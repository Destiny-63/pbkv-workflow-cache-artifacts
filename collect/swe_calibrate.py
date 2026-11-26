"""Preliminary worker calibration with a labeled, behavior-preserving diagnostic.

Uses the same instance/patch repeatedly with unique run IDs and fresh containers.
This is not model evaluation, predictor data, or final worker-count selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-request", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source_request.read_text())
    if source["model_label"] != "synthetic_harness_diagnostic_not_a_model":
        raise ValueError("This preliminary calibration accepts only the labeled harness diagnostic")
    if args.repeats < 4:
        raise ValueError("At least four repetitions are needed to exercise four workers")
    (args.out / "manifest.json").write_text(json.dumps({"registered_utc": time.time(),
        "instance_id": source["instance"]["instance_id"], "repeats_per_worker_count": args.repeats,
        "worker_counts": [1, 2, 4], "preliminary_only": True, "formal_worker_count_frozen": False,
        "model_generated_patch": False, "included_in_predictor_data": False,
        "container_limits": {"cpus": 2, "memory_gib": 8, "pids": 512}}, indent=2) + "\n")
    groups = {}
    for workers in (1, 2, 4):
        queued = time.time()

        def one(i):
            folder = args.out / f"workers_{workers}" / f"repeat_{i}"
            folder.mkdir(parents=True)
            request = deepcopy(source)
            request.update(run_id="pbkv_calibration_" + uuid.uuid4().hex,
                           model_label="synthetic_harness_calibration_not_a_model", queue_start=queued)
            (folder / "request.json").write_text(json.dumps(request, indent=2) + "\n")
            with (folder / "adapter.log").open("w") as handle:
                p = subprocess.run([sys.executable, str(Path(__file__).with_name("swe_harness.py")),
                    "--request", str(folder / "request.json"), "--result", str(folder / "result.json")],
                    cwd=args.out, stdout=handle, stderr=subprocess.STDOUT)
            result = json.loads((folder / "result.json").read_text())
            if p.returncode or result["status"] != "EVALUATED" or not result["tests_executed"]:
                raise RuntimeError(f"No verified fresh test execution: {folder}")
            print(json.dumps({"workers": workers, "repeat": i, "exec_seconds": result["exec_monotonic_seconds"]}), flush=True)
            return result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, range(args.repeats)))
        durations = [r["exec_monotonic_seconds"] for r in results]
        groups[str(workers)] = {"n": len(results), "median_exec_seconds": statistics.median(durations),
            "mean_exec_seconds": statistics.mean(durations), "exec_seconds": durations,
            "run_ids": [r["run_id"] for r in results]}
        (args.out / "progress.json").write_text(json.dumps(groups, indent=2) + "\n")
    baseline = groups["1"]["median_exec_seconds"]
    for group in groups.values():
        group["median_ratio_to_one_worker"] = group["median_exec_seconds"] / baseline
    all_ids = [wid for group in groups.values() for wid in group["run_ids"]]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Run IDs were reused")
    report = {"completed": True, "groups": groups, "preliminary_only": True,
              "formal_worker_count_frozen": False, "included_in_predictor_data": False,
              "limitation": "One development instance with four repetitions per setting; broaden to multiple repositories before freezing formal collection workers."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"completed": True, "unique_evaluations": len(all_ids)}), flush=True)


if __name__ == "__main__":
    main()
