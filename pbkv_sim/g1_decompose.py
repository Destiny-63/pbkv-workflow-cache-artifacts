"""Role-information x time-information decomposition of G1 runs.

Usage: python3 -m pbkv_sim.g1_decompose results/v3/g1/test
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TIME_ROWS = (("无", "he_step"), ("Markov 时长", "he_urg"), ("真实时间", "he_urg_oracleT"))
ROLE_COLS = (("Markov 角色", "markov"), ("真实角色", "oracle"))
BASELINES = (("lru", "none"), ("lae", "none"), ("he_count", "none"))


def main() -> None:
    by = defaultdict(list)
    for p in Path(sys.argv[1]).glob("*.json"):
        r = json.loads(p.read_text())
        c = r["config"]
        by[(c["cap_ratio"], c["name"], c["predictor"])].append(r["summary"]["refetch_tokens"])
    for cap in sorted({k[0] for k in by}):
        mean = lambda name, pred: float(np.mean(by[(cap, name, pred)])) if by.get((cap, name, pred)) else float("nan")
        print(f"### 容量比例 {cap}\n")
        print("基线：" + "，".join(f"{n} {mean(n, p):.0f}" for n, p in BASELINES) + "\n")
        print("| 时间信息 | " + " | ".join(label for label, _ in ROLE_COLS) + " |")
        print("|---|" + "---|" * len(ROLE_COLS))
        for label, name in TIME_ROWS:
            cells = []
            for _, pred in ROLE_COLS:
                ref, val = mean("he_step", pred), mean(name, pred)
                cells.append(f"{val:.0f}" if name == "he_step" else f"{val:.0f} ({(val - ref) / ref * 100:+.1f}%)")
            print(f"| {label} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
