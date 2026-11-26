"""Validation-only HoVer v3 checks and reference grid for the SWE experiment.

Each completed configuration is saved atomically. A manifest pins input/code
hashes so an interrupted grid can resume without mixing different experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .eta import DurationModel
from .learned import PredictionBank
from .replay import ReplayConfig, replay
from .run_g1 import CAPS, CONCS, MIXES, SEEDS, capacity_for, load
from .trace import mix_settings

STATE = None


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def initialize(data, bank, gamma, tau, paths, estimator):
    global STATE
    fast, slow, markov, durations, step_len = load("val", Path(data))
    bank = PredictionBank(Path(bank), markov.roles)
    if bank.provenance["split_sha256"] != hashlib.sha256(Path("results/split.json").read_bytes()).hexdigest():
        raise ValueError("Prediction bank and replay splits differ")
    for setting in ("default", "highdiff"):
        path = Path(data) / f"hover_{setting}.jsonl"
        if bank.provenance["trace_sha256"][path.name] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Prediction bank and replay traces differ")
    STATE = (fast, slow, markov, durations, step_len, bank, gamma, tau, paths, estimator)


def group(job):
    mix, conc, cap_ratio, seed, directory, scope = job
    fast, slow, markov, durations, step_len, bank, gamma, tau, paths, estimator = STATE
    wfs = mix_settings(fast, slow, mix, seed)
    base = replay(wfs, ReplayConfig("lae", "none", conc, None, seed=seed, drop_retired=True), markov, durations)
    capacity = capacity_for(base["summary"]["peak_tokens"], base["summary"]["peak_locked_tokens"], cap_ratio)
    configs = [("lru", "none", False), ("ref_offline", "none", False)]
    for predictor in ("learned", "markov", "oracle"):
        configs.append(("he_step", predictor, False))
        configs.extend(("cand", predictor, true) for true in (False, True))
    if scope == "sampling":
        configs = [("cand", "learned", False)]
    elif scope == "determinism":
        configs = [("he_step", "learned", False), ("he_urg", "learned", False),
                   ("he_step_shadow", "learned", False)]
    records, expected_log = [], base["access_log"]
    for name, pred, true in configs:
        shadow = ("he_urg", "he_urg_oracleT") if name.endswith("_shadow") else ()
        policy = name.removesuffix("_shadow")
        tag = f"{name}_{pred}_{'true' if true else 'est'}_f{mix}_c{conc}_k{cap_ratio}_s{seed}_m{paths}"
        path = Path(directory) / f"{tag}.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue
        cfg = ReplayConfig(policy, pred, conc, capacity, seed=seed, gamma=gamma, tau=tau,
                           oracle_time=true, step_len=step_len, paths=paths, shadow=shadow,
                           candidate_estimator=estimator)
        output = replay(wfs, cfg, markov, durations, bank if pred == "learned" else None)
        if output["access_log"] != expected_log:
            raise RuntimeError(f"Policy-dependent access sequence: {tag}")
        if output["summary"]["overflow_tokens"]:
            raise RuntimeError(f"Capacity overflow: {tag}")
        record = {"tag": tag, "config": asdict(cfg), "mix": mix, "cap_ratio": cap_ratio,
                  "summary": output["summary"], "access_log_matches_lae": True,
                  "instance_count": len({w.workflow_id for w in wfs}), "run_count": len(wfs)}
        atomic_json(path, record)
        records.append(record)
        print(json.dumps({"completed": tag, "refetch_tokens": output["summary"]["refetch_tokens"]}), flush=True)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/20261029_rolefirst"))
    ap.add_argument("--predictor-bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scope", choices=("smoke", "full", "sampling", "determinism"), default="smoke")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--paths", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=15.12663293084)
    ap.add_argument("--estimator", choices=("mc", "exact"), default="exact")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    signature = {"scope": args.scope, "gamma": args.gamma, "tau": args.tau, "paths": args.paths,
                 "estimator": args.estimator,
                 "split_sha256": sha(Path("results/split.json")),
                 "bank_sha256": sha(args.predictor_bank),
                 "trace_sha256": {p.name: sha(p) for p in sorted(args.data.glob("hover_*.jsonl"))},
                 "code_sha256": {p.name: sha(p) for p in sorted(Path("pbkv_sim").glob("*.py"))}}
    manifest = args.out / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["signature"] != signature:
            raise ValueError("Inputs or code changed; use a new output directory")
    else:
        atomic_json(manifest, {"started_utc": datetime.now(timezone.utc).isoformat(), "signature": signature,
                             "test_evaluated": False, "hash_seed": os.environ.get("PYTHONHASHSEED")})
    grid = list(itertools.product(MIXES, CONCS, CAPS, SEEDS)) if args.scope == "full" else [(.5, 32, .5, 2)]
    jobs = [(*values, str(args.out), args.scope) for values in grid]
    init = (str(args.data), str(args.predictor_bank), args.gamma, args.tau, args.paths, args.estimator)
    rows = []
    if args.jobs == 1:
        initialize(*init)
        for job in jobs:
            rows.extend(group(job))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=initialize, initargs=init) as pool:
            for future in as_completed([pool.submit(group, job) for job in jobs]):
                rows.extend(future.result())
    references = [r for r in rows if r["config"]["policy"] == "ref_offline"]
    ratios = {}
    for ref in references:
        key = (ref["mix"], ref["config"]["concurrency"], ref["cap_ratio"], ref["config"]["seed"])
        peers = [r for r in rows if (r["mix"], r["config"]["concurrency"], r["cap_ratio"],
                 r["config"]["seed"]) == key and r is not ref]
        for peer in peers:
            name = peer["config"]["policy"] + "_" + peer["config"]["predictor"] + ("_true" if peer["config"]["oracle_time"] else "_est")
            ratios.setdefault(name, []).append(ref["summary"]["refetch_tokens"] <= peer["summary"]["refetch_tokens"])
    atomic_json(args.out / "completed.json", {"finished_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(rows), "grid_points": len(grid), "test_evaluated": False,
        "reference_no_higher_fraction": {name: sum(v) / len(v) for name, v in sorted(ratios.items())},
        "reference_is_upper_bound": False})


if __name__ == "__main__":
    main()
