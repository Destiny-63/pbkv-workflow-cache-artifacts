"""Summarize the trained predictor and paired validation midpoint replays."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def stats(values):
    return {"values": values, "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def read_runs(directory):
    out = {}
    for path in sorted(directory.glob("*.json")):
        r = json.loads(path.read_text())
        c, s = r["config"], r["summary"]
        if (c["mix"], c["concurrency"], c["cap_ratio"]) != (0.5, 32, 0.5):
            raise ValueError(f"Not a midpoint result: {path}")
        if s["overflow_tokens"]:
            raise ValueError(f"Capacity overflow: {path}")
        if c["predictor"] == "learned" and s["learned_updates"] <= 0:
            raise ValueError(f"Learned predictor was not consumed: {path}")
        key = (c["name"], c["predictor"])
        out.setdefault(key, {})[c["seed"]] = r
    if not out or any(set(rows) != {0, 1, 2} for rows in out.values()):
        raise ValueError(f"Incomplete three-seed results: {directory}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    training = json.loads((root / "train_v1/summary.json").read_text())
    selected = training["selected_seed"]
    base = read_runs(root / f"replay_mid_frozen_seed{selected}")
    baseline = {f"{name}_{pred}": stats([rows[s]["summary"]["refetch_tokens"] for s in (0, 1, 2)])
                for (name, pred), rows in base.items()}
    per_model = []
    for model_seed in (0, 1, 2):
        rows = read_runs(root / f"replay_mid_frozen_seed{model_seed}")
        step = [rows["he_step", "learned"][s]["summary"]["refetch_tokens"] for s in (0, 1, 2)]
        urg = [rows["he_urg", "learned"][s]["summary"]["refetch_tokens"] for s in (0, 1, 2)]
        per_model.append({"model_seed": model_seed, "he_step": stats(step), "he_urg": stats(urg),
            "urg_reduction_vs_step_pct": 100 * (1 - sum(urg) / sum(step)),
            "paired_reduction_tokens": [a - b for a, b in zip(step, urg)]})
    predictive = {}
    for k in ("k1", "k2", "k3"):
        predictive[k] = stats([r["val"]["all"][k]["accuracy"] for r in training["runs"]])
    report = {
        "status": "complete_validation_development_experiment",
        "scope": "validation only, f=0.5, concurrency=32, capacity ratio=0.5, three replay seeds",
        "gamma": 0.5, "tau": 15.12663293084,
        "parameters_retuned": False, "selected_model_seed": selected,
        "predictor_accuracy_across_training_seeds": predictive,
        "selected_model_replay": baseline, "replay_by_model_seed": per_model,
        "limitations": ["Existing validation set used for checkpoint selection; no independent test conclusion",
            "Grouped Reader steps differ from paper invocation-level prediction",
            "Recorded-duration offline replay measures refetch tokens, not end-to-end latency",
            "Predictions become available at representative prefill completion; predictor inference delay omitted",
            "ETA sampling still uses the original training-only Markov paths and durations"],
        "artifact_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(root.glob("train_v1/seed_*/model.pt"))},
    }
    (root / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
