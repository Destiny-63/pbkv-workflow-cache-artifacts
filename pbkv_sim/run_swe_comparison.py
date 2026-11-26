"""Frozen five-group SWE replay, retaining G1 scores and true-time semantics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .eta import DurationModel
from .learned import PredictionBank
from .predictors import MarkovPredictor, RoleTable
from .replay import ReplayConfig, replay
from .swe_corpus import load_corpus, read_json, sha, write_new
from .swe_evaluation import check_lock


GROUPS = [
    dict(group="step+predictor", policy="he_step", predictor="learned", oracle_time=False),
    dict(group="step+markov", policy="he_step", predictor="markov", oracle_time=False),
    dict(group="eta+predictor", policy="he_urg", predictor="learned", oracle_time=True),
    dict(group="eta+markov", policy="he_urg", predictor="markov", oracle_time=True),
    dict(group="oracle", policy="he_urg", predictor="oracle", oracle_time=True),
]


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sources():
    # Pin the whole replay package, including the prefix tree and corpus loader.
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))} | {
        "swe_architecture.json": sha(Path(__file__).with_name("swe_architecture.json"))}


def validate_bank(corpus_path, lock_path, bank_path):
    lock = check_lock(lock_path, corpus_path)
    corpus = read_json(corpus_path)
    bank = PredictionBank(bank_path, RoleTable(tuple(corpus["architecture"]["roles"])))
    p = bank.provenance
    if (bank.checkpoint_sha256 != lock["checkpoint_sha256"]
            or p.get("evaluation_lock_sha256") != sha(lock_path)
            or p.get("corpus_sha256") != sha(corpus_path)
            or p.get("split_sha256") != corpus["registered_split_sha256"]
            or p.get("partition") != "test"
            or p.get("inference_trace_sha256") != {
                e["trace"]: e["trace_sha256"] for e in corpus["entries"] if e["partition"] == "test"}):
        raise ValueError("Prediction bank differs from the frozen test model/corpus")
    return lock, corpus, bank


def freeze(corpus_path, lock_path, bank_path, prior_params, out):
    lock, corpus, _ = validate_bank(corpus_path, lock_path, bank_path)
    params = read_json(prior_params)
    protocol = dict(
        schema=1, partition="test", evaluation_scope=lock["evaluation_scope"],
        development_only=corpus["development_only"], coverage=lock["coverage"],
        corpus=str(Path(corpus_path).resolve()), corpus_sha256=sha(corpus_path),
        prediction_lock=str(Path(lock_path).resolve()), prediction_lock_sha256=sha(lock_path),
        predictor_bank=str(Path(bank_path).resolve()), predictor_bank_sha256=sha(bank_path),
        checkpoint_sha256=lock["checkpoint_sha256"], split_sha256=lock["split_sha256"],
        prior_params=str(Path(prior_params).resolve()), prior_params_sha256=sha(prior_params),
        config=dict(concurrencies=[16, 32, 64], cap_ratios=[0.25, 0.5, 0.75], seeds=[0, 1, 2],
                    gamma=params["gamma"], tau=params["tau"], k_max=3, markov_alpha=0.5,
                    groups=GROUPS),
        formula=dict(step="sum_w sum_k gamma^(k-1) s_w(k) sum_a_in_Ow(c) P_w(k)[a]",
                     survival="s_w(1)=1; s_w(k)=product_j<k (1-P_w(j)[END])",
                     eta="same step terms multiplied by exp(-max(t_start_w(k)-now,0)/tau)",
                     true_time="role-independent time for learned/Markov; true role+time for oracle",
                     tail="extend the last non-END start by train median step length; if none, extend END",
                     capacity="int(peak_locked + ratio*(peak_tokens-peak_locked)); unlimited LAE prepass",
                     priorities="unchanged retired-first, summed role score, LRU tie break"),
        parameters="G1 v3 frozen gamma/tau transferred unchanged; no SWE test tuning",
        learned_available="after representative prefill; train-only Markov prior before then",
        source_sha256=sources(), frozen_before_cache_replay=True,
        limitations=["Available subset, not complete registered test coverage",
                     "ETA groups intentionally read future timestamps; diagnostic true-time comparisons",
                     "Oracle is the same score with true roles/time, not an optimal eviction bound",
                     "Recorded LLM durations and tool waits; misses do not delay later calls",
                     "No predictor overhead, SWE solve-rate or end-to-end latency measurement"])
    write_new(out, protocol)
    return protocol


def prepare(protocol_path):
    spec = read_json(protocol_path)
    if spec["schema"] != 1 or spec["source_sha256"] != sources() or spec["config"]["groups"] != GROUPS:
        raise ValueError("Replay protocol or source changed")
    for name in ("corpus", "prediction_lock", "predictor_bank", "prior_params"):
        if sha(spec[name]) != spec[name + "_sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    lock, _, bank = validate_bank(spec["corpus"], spec["prediction_lock"], spec["predictor_bank"])
    if (spec["coverage"] != lock["coverage"] or spec["evaluation_scope"] != lock["evaluation_scope"]
            or spec["checkpoint_sha256"] != lock["checkpoint_sha256"]):
        raise ValueError("Frozen model or evaluation scope changed")
    corpus, parts, _, _ = load_corpus(spec["corpus"], ("train", "test"),
                                    require_final=spec["evaluation_scope"] != "available_subset")
    expected = {f"{w.setting}|{w.workflow_id}|{i}" for w in parts["test"] for i, _ in enumerate(w.steps())}
    if set(bank.rows) != expected:
        raise ValueError("Missing or extra test prediction rows")
    table = bank.roles
    markov = MarkovPredictor(alpha=spec["config"]["markov_alpha"], roles=table).fit(
        [w.roles() for w in parts["train"]])
    durations = DurationModel.fit(parts["train"], table)
    step_len = float(np.median([s.end-s.start for w in parts["train"] for s in w.steps()]))
    return spec, parts, bank, markov, durations, step_len


_INPUTS = None


def init_worker(inputs):
    global _INPUTS
    _INPUTS = inputs


def one(job):
    concurrency, seed, out, protocol_sha = job
    spec, parts, bank, markov, durations, step_len = _INPUTS
    config = spec["config"]
    ordered = sorted(parts["test"], key=lambda w: w.workflow_id)
    wfs = [ordered[int(i)] for i in np.random.default_rng(seed).permutation(len(ordered))]
    start = time.monotonic()
    base = replay(wfs, ReplayConfig("lae", "none", concurrency, None, seed=seed, drop_retired=True),
                  markov, durations)
    access_sha = digest(base["access_log"])
    workload_sha = digest([w.workflow_id for w in wfs])
    peak, locked = base["summary"]["peak_tokens"], base["summary"]["peak_locked_tokens"]
    calibration = dict(concurrency=concurrency, seed=seed, peak_tokens=peak, peak_locked_tokens=locked,
                       step_len=step_len, workload_sha256=workload_sha, access_sha256=access_sha,
                       calls=len(base["calls"]), workflows=len(wfs),
                       instances=len({w.instance_id for w in wfs}),
                       corpus_seeds=sorted({int(w.workflow_id.rsplit('_s', 1)[1]) for w in wfs}),
                       protocol_sha256=protocol_sha)
    path = Path(out) / f"calibration_c{concurrency}_s{seed}.json"
    if path.exists():
        if read_json(path) != calibration:
            raise ValueError("Resume calibration differs")
    else:
        write_new(path, calibration)
    rows = []
    for ratio, group in itertools.product(config["cap_ratios"], GROUPS):
        capacity = int(locked + ratio * (peak-locked))
        target = Path(out) / f"{group['group'].replace('+', '_')}_c{concurrency}_r{ratio}_s{seed}.json"
        if target.exists():
            row = read_json(target)
            if (row["protocol_sha256"] != protocol_sha or row["access_sha256"] != access_sha
                    or row["capacity"] != capacity or row["group"] != group["group"]):
                raise ValueError("Resume result differs")
        else:
            t = time.monotonic()
            cfg = ReplayConfig(group["policy"], group["predictor"], concurrency, capacity, seed=seed,
                               gamma=config["gamma"], tau=config["tau"], k_max=config["k_max"],
                               oracle_time=group["oracle_time"], step_len=step_len)
            result = replay(wfs, cfg, markov, durations, bank if group["predictor"] == "learned" else None)
            summary = result["summary"]
            if result["access_log"] != base["access_log"]:
                raise ValueError("Strategy changed the paired access sequence")
            if (summary["overflow_tokens"] or summary["peak_tokens"] > capacity
                    or summary["workflows"] != len(wfs) or len(result["calls"]) != len(base["calls"])):
                raise ValueError("Capacity or workload acceptance failed")
            if group["predictor"] == "learned" and summary["learned_updates"] != len(bank.rows):
                raise ValueError("Learned predictions were not consumed exactly once per test step")
            row = dict(**group, concurrency=concurrency, seed=seed, cap_ratio=ratio, capacity=capacity,
                       protocol_sha256=protocol_sha, workload_sha256=workload_sha, access_sha256=access_sha,
                       access_sequence_verified=True, calls=len(result["calls"]), summary=summary,
                       elapsed_seconds=time.monotonic()-t)
            write_new(target, row)
            print(json.dumps(dict(completed=target.name, seconds=row["elapsed_seconds"],
                                  refetch_tokens=summary["refetch_tokens"])), flush=True)
        rows.append(row)
    print(json.dumps(dict(workload_complete=[concurrency, seed], seconds=time.monotonic()-start)), flush=True)
    return rows


def run(protocol_path, out, jobs=4):
    inputs = prepare(protocol_path)
    spec, parts, *_ = inputs
    Path(out).mkdir(parents=True, exist_ok=True)
    protocol_sha = sha(protocol_path)
    grid = [(c, s, str(out), protocol_sha) for c, s in itertools.product(
        spec["config"]["concurrencies"], spec["config"]["seeds"])]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=init_worker, initargs=(inputs,)) as pool:
        for result in pool.map(one, grid):
            rows.extend(result)
    expected = len(grid) * len(spec["config"]["cap_ratios"]) * len(GROUPS)
    if len(rows) != expected:
        raise ValueError("Incomplete replay grid")
    write_new(Path(out) / "results.json", rows)
    write_new(Path(out) / "acceptance.json", dict(
        status="passed", runs=len(rows), expected_runs=expected,
        protocol_sha256=protocol_sha, test_workflows=len(parts["test"]),
        test_instances=len({w.instance_id for w in parts["test"]}),
        paired_access_sequences=True, identical_capacity_within_each_grid_point=True,
        zero_capacity_overflow=True, complete_prediction_consumption=True,
        train_validation_test_assignments_unchanged=True,
        evaluation_scope=spec["evaluation_scope"], development_only=spec["development_only"],
        artifact_sha256={p.name: sha(p) for p in sorted(Path(out).glob("*.json"))}))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    for key in ("corpus", "prediction-lock", "predictor-bank", "prior-params", "out"):
        p.add_argument("--"+key, type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.command == "freeze":
        freeze(args.corpus, args.prediction_lock, args.predictor_bank, args.prior_params, args.out)
    else:
        run(args.protocol, args.out, args.jobs)


if __name__ == "__main__":
    main()
