"""v3 validity statistics: role-tag consistency, pinned share, same-role reuse gaps.

Usage: python3 -m pbkv_sim.a3_v3_extra --data DIR --split results/split.json --out results/v3/a3/a3_v3_extra.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .eta import DurationModel
from .predictors import MarkovPredictor
from .replay import ReplayConfig, replay
from .trace import load_workflows, mix_settings

MIXES = (0.0, 0.25, 0.5, 1.0)
CONCS = (16, 32, 64)


def reuse_gaps(wfs) -> dict:
    gaps: dict[str, list[float]] = {}
    for w in wfs:
        last_end: dict[str, float] = {}
        for c in sorted(w.calls, key=lambda c: c.start):
            if c.role in last_end and c.start > last_end[c.role]:
                gaps.setdefault(c.role, []).append(c.start - last_end[c.role])
            last_end[c.role] = max(last_end.get(c.role, 0.0), c.end)
    return {role: {"n": len(v), "mean": float(np.mean(v)), "p50": float(np.percentile(v, 50)),
                   "p95": float(np.percentile(v, 95))} for role, v in sorted(gaps.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    split = json.loads(Path(args.split).read_text())
    fast = load_workflows(data / "hover_default.jsonl")
    slow = load_workflows(data / "hover_highdiff.jsonl")
    train = [w for d in (fast, slow) for k, w in d.items() if split.get(k) == "train"]
    markov = MarkovPredictor().fit([w.roles() for w in train])
    durations = DurationModel.fit(train)
    test_fast = {k: w for k, w in fast.items() if split.get(k) == "test"}
    test_slow = {k: w for k, w in slow.items() if split.get(k) == "test"}

    out: dict = {"reuse_gap_s": {"default": reuse_gaps(test_fast.values()),
                                 "highdiff": reuse_gaps(test_slow.values())}}
    tags, pinned = {}, {}
    for mix in MIXES:
        for conc in CONCS:
            wfs = mix_settings(test_fast, test_slow, mix, seed=0)
            s = replay(wfs, ReplayConfig("lae", "none", conc, None, seed=0, drop_retired=True),
                       markov, durations)["summary"]
            key = f"f{mix}_c{conc}"
            tags[key] = s["untagged_reuse_tokens"] / max(s["reuse_tokens"], 1)
            pinned[key] = s["peak_locked_tokens"] / max(s["peak_tokens"], 1)
    out["untagged_reuse_share"] = tags
    out["peak_locked_over_peak"] = pinned
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"max_untagged_share": max(tags.values()), "max_pinned_share": max(pinned.values())}))


if __name__ == "__main__":
    main()
