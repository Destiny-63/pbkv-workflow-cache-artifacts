"""Run replay acceptance, real-hidden retraining, then the HoVer validation grid."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .run_phase0 import atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--predictor-bank", type=Path, required=True)
    ap.add_argument("--reference-training", type=Path, required=True)
    ap.add_argument("--mc-record", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def status(stage, **extra):
        value = {"updated_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
                 "test_evaluated": False, **extra}
        atomic_json(args.out / "status.json", value)
        print(json.dumps(value), flush=True)

    def run(name, command, env=None):
        status(name)
        with (args.out / f"{name}.log").open("w") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, **(env or {})})

    def grid(scope, name, paths=256, hash_seed="0"):
        run(name, [sys.executable, "-u", "-m", "pbkv_sim.run_phase0", "--data", str(args.data),
            "--predictor-bank", str(args.predictor_bank), "--out", str(args.out / name),
            "--scope", scope, "--paths", str(paths), "--estimator", "exact",
            "--jobs", str(args.jobs if scope == "full" else 1)], {"PYTHONHASHSEED": hash_seed})

    try:
        run("unit_tests", [sys.executable, "-m", "pytest", "pbkv_sim/tests", "collect/tests", "-q"])
        run("retrain", [sys.executable, "-u", "-m", "pbkv_sim.train_predictor", "--data", str(args.data),
            "--split", "results/split.json", "--out", str(args.out / "retrain"), "--threads", "4"])
        old = json.loads(args.reference_training.read_text())
        new = json.loads((args.out / "retrain/summary.json").read_text())
        comparisons = []
        for before, after in zip(old["runs"], new["runs"]):
            if before["seed"] != after["seed"]:
                raise RuntimeError("Retraining seed order changed")
            for k in ("k1", "k3"):
                difference = abs(before["val"]["all"][k]["accuracy"] - after["val"]["all"][k]["accuracy"])
                comparisons.append({"seed": before["seed"], "horizon": k, "accuracy_difference": difference})
                if difference > .005:
                    raise RuntimeError(f"Role-table regression: {comparisons[-1]}")
        atomic_json(args.out / "role_table_regression.json", {"passed": True, "comparisons": comparisons})

        for seed in (0, 1):
            grid("determinism", f"hash_seed{seed}", hash_seed=str(seed))
        comparisons = []
        for path in sorted((args.out / "hash_seed0").glob("he_*.json")):
            one = json.loads(path.read_text())["summary"]
            two = json.loads((args.out / "hash_seed1" / path.name).read_text())["summary"]
            if one != two:
                raise RuntimeError(f"Cross-process nondeterminism: {path.name}")
            comparisons.append(path.name)
        if len(comparisons) != 3:
            raise RuntimeError("Missing determinism configurations")
        atomic_json(args.out / "determinism.json", {"passed": True, "hash_seeds": [0, 1], "configs": comparisons})

        for paths in (256, 1024):
            grid("sampling", f"exact_m{paths}", paths=paths)
        def sample(directory):
            path = next((args.out / directory).glob("cand_*.json"))
            return json.loads(path.read_text())["summary"]
        if sample("exact_m256") != sample("exact_m1024"):
            raise RuntimeError("Exact expectation depends on an unused sampling count")
        atomic_json(args.out / "expectation.json", {"estimator": "exact empirical Laplace transform",
            "original_mc_acceptance": json.loads(args.mc_record.read_text()),
            "exact_m256": sample("exact_m256"), "exact_m1024": sample("exact_m1024"),
            "same_score_formula": True, "sampling_count_is_unused_for_exact_estimator": True})
        grid("smoke", "smoke")
        grid("full", "full_grid")
        status("complete", completed=json.loads((args.out / "full_grid/completed.json").read_text()))
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
