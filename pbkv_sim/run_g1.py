"""G1 grid runner. Usage: python3 -m pbkv_sim.run_g1 --split-name val|test --out DIR [--gamma G --tau T --tau-he-time T2] [--jobs N] [--only-mid]"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .eta import DurationModel
from .learned import PredictionBank
from .predictors import MarkovPredictor
from .replay import ReplayConfig, replay
from .trace import load_workflows, mix_settings

DATA = Path("data/hover_rolefirst")
MIXES = (0.0, 0.25, 0.5, 1.0)
CONCS = (16, 32, 64)
CAPS = (0.25, 0.5, 0.75)
SEEDS = (0, 1, 2)
MID = (0.5, 32, 0.5)


def capacity_for(peak: int, peak_locked: int, ratio: float) -> int:
    return int(peak_locked + ratio * (peak - peak_locked))


def load(split_name: str, data: Path = DATA):
    split = json.loads(Path("results/split.json").read_text())
    fast = load_workflows(data / "hover_default.jsonl")
    slow = load_workflows(data / "hover_highdiff.jsonl")
    pick = lambda d, part: {k: v for k, v in d.items() if split.get(k) == part}
    train = list(pick(fast, "train").values()) + list(pick(slow, "train").values())
    markov = MarkovPredictor().fit([w.roles() for w in train])
    durations = DurationModel.fit(train)
    step_len = float(np.median([s.end - s.start for w in train for s in w.steps()]))
    return pick(fast, split_name), pick(slow, split_name), markov, durations, step_len


def runs(gamma: float, tau: float, tau_time: float, predictors=("markov", "oracle")):
    yield ("lru", "none", {})
    yield ("lae", "none", {})
    yield ("he_count", "none", {"gamma": gamma})
    for pred in predictors:
        yield ("he_step", pred, {"gamma": gamma, "tau": tau, "shadow": ("he_urg", "he_urg_oracleT")})
        yield ("he_urg", pred, {"gamma": gamma, "tau": tau})
        yield ("he_urg_rank", pred, {"gamma": gamma})
        yield ("he_time", pred, {"tau": tau_time})
        yield ("he_urg_oracleT", pred, {"gamma": gamma, "tau": tau, "oracle_time": True, "policy_alias": "he_urg"})


def one(job):
    split_name, mix, conc, cap, seed, name, pred, extra, out_dir, data, bank_path = job
    fast, slow, markov, durations, step_len = load(split_name, Path(data))
    wfs = mix_settings(fast, slow, mix, seed=seed)
    base = replay(wfs, ReplayConfig("lae", "none", conc, None, seed=seed, drop_retired=True), markov, durations)
    capacity = capacity_for(base["summary"]["peak_tokens"], base["summary"]["peak_locked_tokens"], cap)
    extra = dict(extra)
    policy = extra.pop("policy_alias", name)
    cfg = ReplayConfig(policy, pred, conc, capacity, seed=seed, step_len=step_len, **extra)
    bank = PredictionBank(Path(bank_path)) if pred == "learned" else None
    if bank:
        if bank.provenance["split_sha256"] != hashlib.sha256(Path("results/split.json").read_bytes()).hexdigest():
            raise ValueError("Prediction bank and replay splits differ")
        for setting in ("default", "highdiff"):
            path = Path(data) / f"hover_{setting}.jsonl"
            if bank.provenance["trace_sha256"][path.name] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Prediction bank and replay traces differ")
    out = replay(wfs, cfg, markov, durations, bank)
    if out["summary"]["overflow_tokens"]:
        raise RuntimeError(f"overflow in {name}_{pred}_f{mix}_c{conc}_k{cap}_s{seed}")
    tag = f"{name}_{pred}_f{mix}_c{conc}_k{cap}_s{seed}"
    record = {"config": {"name": name, "predictor": pred, "mix": mix, "concurrency": conc, "cap_ratio": cap,
                         "capacity": capacity, "seed": seed, **{k: v for k, v in extra.items() if k != "shadow"}},
              "summary": out["summary"]}
    if bank:
        record["predictor_provenance"] = {
            "checkpoint_sha256": bank.checkpoint_sha256,
            "seed": bank.provenance["seed"],
            "availability": "after representative prefill; Markov prior until then",
            "eta_model": "unchanged training-only Markov paths and empirical durations",
        }
    (Path(out_dir) / f"{tag}.json").write_text(json.dumps(record))
    return tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-name", choices=("val", "test"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--tau-he-time", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only-mid", action="store_true")
    ap.add_argument("--policies", default="")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--predictors", default="markov,oracle")
    ap.add_argument("--predictor-bank", type=Path)
    args = ap.parse_args()
    predictors = tuple(args.predictors.split(","))
    if not set(predictors) <= {"markov", "oracle", "learned"}:
        ap.error("--predictors must contain markov, oracle, and/or learned")
    if "learned" in predictors and args.predictor_bank is None:
        ap.error("--predictor-bank is required for learned replay")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    grid = [MID] if args.only_mid else list(itertools.product(MIXES, CONCS, CAPS))
    keep = set(args.policies.split(",")) if args.policies else None
    jobs = [
        (args.split_name, mix, conc, cap, seed, name, pred, extra, args.out, args.data,
         str(args.predictor_bank) if args.predictor_bank else None)
        for (mix, conc, cap) in grid
        for seed in SEEDS
        for name, pred, extra in runs(args.gamma, args.tau, args.tau_he_time, predictors)
        if keep is None or name in keep
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for tag in pool.map(one, jobs):
            print(tag, flush=True)


if __name__ == "__main__":
    main()
