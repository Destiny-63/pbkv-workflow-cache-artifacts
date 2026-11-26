"""Replay audited SWE workflows with shared train-only baselines and paired workloads."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .eta import DurationModel
from .learned import PredictionBank
from .predictors import MarkovPredictor, RoleTable
from .replay import ReplayConfig, replay
from .swe_corpus import load_corpus, read_json, sha, write_new
from .swe_evaluation import check_lock, validate_config


def run(corpus_path, partition, out, config=None, bank_path=None, evaluation_lock=None):
    if partition not in {"val", "test"}:
        raise ValueError("Expected val/test replay partition")
    if partition == "test":
        if evaluation_lock is None or config is not None:
            raise ValueError("Test replay requires --evaluation-lock; config overrides are forbidden")
        lock = check_lock(evaluation_lock, corpus_path)
        config = lock["replay"]
    else:
        if config is None or evaluation_lock is not None:
            raise ValueError("Validation replay requires --config")
        lock = None
    validate_config(config)
    if Path(out).exists():
        raise FileExistsError(out)
    subset = bool(lock and lock.get("evaluation_scope") == "available_subset")
    corpus, parts, traces, _ = load_corpus(corpus_path, ("train", partition), require_final=partition == "test" and not subset)
    table = RoleTable(tuple(corpus["architecture"]["roles"]))
    markov = MarkovPredictor(alpha=config["markov_alpha"], roles=table).fit([w.roles() for w in parts["train"]])
    durations = DurationModel.fit(parts["train"], table)
    # Smoothing may place mass on any role, so every role needs train timing support.
    missing = set(table.roles[:-1]) - {s.role for w in parts["train"] for s in w.steps()}
    if missing:
        raise ValueError(f"Training duration model lacks roles: {sorted(missing)}")
    step_len = float(np.median([s.end - s.start for w in parts["train"] for s in w.steps()]))
    bank = None
    if "learned" in config["predictors"]:
        if bank_path is None:
            raise ValueError("Learned SWE replay requires --predictor-bank")
        bank = PredictionBank(bank_path, table)
        p = bank.provenance
        if (p.get("corpus_sha256") != sha(corpus_path)
                or p.get("split_sha256") != corpus["registered_split_sha256"]
                or p.get("partition") != partition
                or p.get("inference_trace_sha256") != {e["trace"]: e["trace_sha256"] for e in corpus["entries"] if e["partition"] == partition}):
            raise ValueError("Prediction bank differs from replay corpus/partition")
        if lock and (bank.checkpoint_sha256 != lock["checkpoint_sha256"]
                     or p.get("evaluation_lock_sha256") != sha(evaluation_lock)):
            raise ValueError("Prediction bank differs from final evaluation lock")
        expected = {f"{w.setting}|{w.workflow_id}|{i}" for w in parts[partition] for i, _ in enumerate(w.steps())}
        if set(bank.rows) != expected:
            raise ValueError("Prediction bank has missing or extra workflows/steps")
    Path(out).mkdir(parents=True)
    provenance = {"corpus_sha256": sha(corpus_path), "partition": partition,
                  "development_only": corpus["development_only"], "config": config,
                  "evaluation_scope": lock.get("evaluation_scope", "complete_formal") if lock else "validation",
                  "coverage": lock.get("coverage") if lock else None,
                  "evaluation_lock_sha256": sha(evaluation_lock) if evaluation_lock else None,
                  "bank_sha256": sha(bank_path) if bank else None,
                  "training_workflows": len(parts["train"]), "evaluation_workflows": len(parts[partition]),
                  "evaluation_instances": len({w.instance_id for w in parts[partition]}),
                  "train_only_models": ["Markov", "DurationModel", "median_step_length"],
                  "learned_available": "current prefill completes; train-only Markov prior beforehand",
                  "limitations": ["recorded durations; misses do not delay subsequent calls",
                                  "predictor latency is not included; no end-to-end latency claim"],
                  "source_sha256": {name: sha(Path(__file__).with_name(name)) for name in
                                    ("run_swe.py", "replay.py", "eta.py", "scores.py", "predictors.py")}}
    write_new(Path(out) / "provenance.json", provenance)
    results = []
    for concurrency, seed in itertools.product(config["concurrencies"], config["seeds"]):
        wfs = sorted(parts[partition], key=lambda w: w.workflow_id)
        order = np.random.default_rng(seed).permutation(len(wfs))
        wfs = [wfs[int(i)] for i in order]
        workload_sha = hashlib.sha256(json.dumps([w.workflow_id for w in wfs]).encode()).hexdigest()
        base = replay(wfs, ReplayConfig("lae", "none", concurrency, None, seed=seed, drop_retired=True), markov, durations)
        for ratio in config["cap_ratios"]:
            peak, locked = base["summary"]["peak_tokens"], base["summary"]["peak_locked_tokens"]
            capacity = int(locked + ratio * (peak - locked))
            for policy in config["policies"]:
                predictors = ["none"] if policy in {"lru", "lae", "he_count", "ref_offline"} else config["predictors"]
                for predictor in predictors:
                    cfg = ReplayConfig(policy, predictor, concurrency, capacity, seed=seed,
                        gamma=config["gamma"], tau=config["tau"], paths=config["paths"], step_len=step_len,
                        candidate_estimator=config["candidate_estimator"], drop_retired=policy == "lae")
                    result = replay(wfs, cfg, markov, durations, bank if predictor == "learned" else None)
                    # Accesses and timestamps must be identical across policies.
                    if result["access_log"] != base["access_log"]:
                        raise ValueError("Policy changed the workload access sequence")
                    if result["summary"]["overflow_tokens"]:
                        raise ValueError("Replay capacity overflow")
                    row = {"policy": policy, "predictor": predictor, "concurrency": concurrency,
                           "seed": seed, "cap_ratio": ratio, "capacity": capacity,
                           "workload_sha256": workload_sha, "access_sequence_verified": True,
                           "diagnostic_future_information": predictor == "oracle" or policy == "ref_offline",
                           "summary": result["summary"]}
                    results.append(row)
                    print(json.dumps(row), flush=True)
    write_new(Path(out) / "results.json", results)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("corpus", "out"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--split-name", choices=("val", "test"), required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--evaluation-lock", type=Path)
    ap.add_argument("--predictor-bank", type=Path)
    args = ap.parse_args()
    run(args.corpus, args.split_name, args.out, read_json(args.config) if args.config else None,
        args.predictor_bank, args.evaluation_lock)


if __name__ == "__main__":
    main()
