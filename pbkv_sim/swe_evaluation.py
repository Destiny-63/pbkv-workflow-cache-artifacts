"""Freeze the validation-selected checkpoint and replay settings before final test."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .swe_corpus import ARCHITECTURE, load_corpus, read_json, sha, write_new

DEFAULT_REPLAY = {"concurrencies": [4], "cap_ratios": [0.25, 0.5, 0.75],
                  "seeds": [0, 1, 2], "policies": ["lru", "lae", "he_step", "cand"],
                  "predictors": ["markov", "learned"], "gamma": 0.7, "tau": 10.0,
                  "markov_alpha": 0.5, "paths": 256, "candidate_estimator": "exact"}


def validate_config(config):
    if set(config) != set(DEFAULT_REPLAY):
        raise ValueError("Replay config must contain exactly the documented fields")
    for key in ("concurrencies", "seeds", "policies", "predictors", "cap_ratios"):
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"Empty or duplicate {key}")
    if any(type(n) is not int or n < 1 for n in config["concurrencies"]):
        raise ValueError("Invalid concurrency")
    if any(type(n) is not int or n < 0 for n in config["seeds"]):
        raise ValueError("Invalid replay seed")
    if any(not math.isfinite(r) or not 0 < r <= 1 for r in config["cap_ratios"]):
        raise ValueError("Capacity ratios must be in (0,1]")
    if not set(config["policies"]) <= {"lru", "lae", "he_count", "he_step", "he_urg", "cand", "ref_offline"}:
        raise ValueError("Unknown policy")
    if not set(config["predictors"]) <= {"markov", "learned", "oracle"}:
        raise ValueError("Unknown predictor")
    if (not math.isfinite(config["gamma"]) or not 0 <= config["gamma"] <= 1
            or not math.isfinite(config["tau"]) or config["tau"] <= 0
            or not math.isfinite(config["markov_alpha"]) or config["markov_alpha"] <= 0
            or type(config["paths"]) is not int or config["paths"] < 1
            or config["candidate_estimator"] not in {"exact", "mc"}):
        raise ValueError("Invalid replay hyperparameters")
    return config


def freeze(corpus_path, training, config, out, allow_available_subset=False):
    corpus = read_json(corpus_path)
    partial = corpus["development_only"] or not corpus["complete"]
    if partial and not allow_available_subset:
        raise ValueError("Cannot freeze final test on a development/partial corpus")
    # Verify development inputs; test traces and hidden remain unopened here.
    load_corpus(corpus_path, ("train", "val"), require_final=not partial)
    training = Path(training).resolve()
    summary = read_json(training / "summary.json")
    prov = summary["provenance"]
    if (prov.get("corpus_sha256") != sha(corpus_path)
            or prov.get("development_only") is not corpus["development_only"]
            or prov.get("test_evaluated") is not False
            or prov.get("architecture_sha256") != sha(ARCHITECTURE)):
        raise ValueError("Training is not a formal train/validation-only run on this corpus")
    runs = summary["runs"]
    if not runs or any(not math.isfinite(r["val_nll"]) for r in runs):
        raise ValueError("Invalid validation selection report")
    selected = min(runs, key=lambda r: r["val_nll"])
    if selected["seed"] != summary["selected_seed"]:
        raise ValueError("Checkpoint was not selected by validation NLL")
    checkpoint = training / f"seed_{selected['seed']}" / "model.pt"
    # Bind the exact selected validation output, checkpoint and settings.
    bank = read_json(checkpoint.parent / "val_predictions.json")
    if bank["checkpoint_sha256"] != sha(checkpoint) or bank["provenance"].get("corpus_sha256") != sha(corpus_path):
        raise ValueError("Selected validation bank does not match checkpoint/corpus")
    lock = {"schema": 1, "corpus_sha256": sha(corpus_path),
            "architecture_sha256": sha(ARCHITECTURE),
            "split_sha256": corpus["registered_split_sha256"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "training_summary_sha256": sha(training / "summary.json"),
            "validation_bank_sha256": sha(checkpoint.parent / "val_predictions.json"),
            "selected_seed": selected["seed"], "selection": "minimum validation masked NLL",
            "evaluation_scope": "available_subset" if partial else "complete_formal",
            "coverage": {"counts": corpus["counts"], "expected_workflows": corpus["expected_workflows"],
                         "missing_workflows": corpus["missing_workflows"],
                         "quarantined_workflows": corpus["quarantined_workflows"]},
            "replay": validate_config(config), "test_used_for_selection": False}
    write_new(out, lock)
    return lock


def check_lock(path, corpus_path, checkpoint=None):
    lock = read_json(path)
    corpus = read_json(corpus_path)
    subset = lock.get("evaluation_scope") == "available_subset"
    if subset and lock.get("coverage") != {k: corpus[k] for k in
            ("counts", "expected_workflows", "missing_workflows", "quarantined_workflows")}:
        raise ValueError("Available-subset evaluation coverage changed")
    if (lock.get("schema") != 1 or (not subset and (corpus["development_only"] or not corpus["complete"]))
            or lock["corpus_sha256"] != sha(corpus_path)
            or lock["architecture_sha256"] != sha(ARCHITECTURE)
            or lock["split_sha256"] != corpus["registered_split_sha256"]
            or lock["test_used_for_selection"] is not False):
        raise ValueError("Final evaluation lock differs from formal corpus/architecture")
    selected = Path(checkpoint or lock["checkpoint"])
    if sha(selected) != lock["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from final evaluation lock")
    validate_config(lock["replay"])
    return lock


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("corpus", "training", "config", "out"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--allow-available-subset", action="store_true",
                    help="Explicitly evaluate this incomplete snapshot; retain missing coverage in the lock")
    args = ap.parse_args()
    freeze(args.corpus, args.training, read_json(args.config), args.out, args.allow_available_subset)


if __name__ == "__main__":
    main()
