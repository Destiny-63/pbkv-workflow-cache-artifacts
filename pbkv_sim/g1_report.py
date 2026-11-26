"""Aggregate G1 runs and apply the pre-registered gate. Usage: python3 -m pbkv_sim.g1_report results/v3/g1/test"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    rows = [json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("*.json")]
    by = defaultdict(list)
    for r in rows:
        c = r["config"]
        by[(c["mix"], c["concurrency"], c["cap_ratio"], c["name"], c["predictor"])].append(r["summary"])
    points = sorted({k[:3] for k in by})
    print("| f | 并发 | 容量 | 预测器 | 分歧率 he_urg | 分歧率 oracleT | 分歧引起的重算占比 | refetch he_step | refetch he_urg | Δ% | refetch oracleT |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    gate = False
    for f, c, k in points:
        for pred in ("markov", "oracle"):
            ref = by.get((f, c, k, "he_step", pred), [])
            urg = by.get((f, c, k, "he_urg", pred), [])
            ora = by.get((f, c, k, "he_urg_oracleT", pred), [])
            if not ref:
                continue
            div = float(np.mean([s.get("diverge_rate_he_urg", 0) for s in ref]))
            div_o = float(np.mean([s.get("diverge_rate_he_urg_oracleT", 0) for s in ref]))
            share = float(np.mean([s.get("refetch_from_divergent_he_urg", 0) / max(s["refetch_tokens"], 1) for s in ref]))
            r_ref = float(np.mean([s["refetch_tokens"] for s in ref]))
            r_urg = float(np.mean([s["refetch_tokens"] for s in urg])) if urg else float("nan")
            r_ora = float(np.mean([s["refetch_tokens"] for s in ora])) if ora else float("nan")
            delta = (r_urg - r_ref) / r_ref * 100 if r_ref else 0.0
            if pred == "markov" and div >= 0.05 and share >= 0.10:
                gate = True
            print(f"| {f} | {c} | {k} | {pred} | {div:.3f} | {div_o:.3f} | {share:.3f} | {r_ref:.0f} | {r_urg:.0f} | {delta:+.1f} | {r_ora:.0f} |")
    missing = [(f, c, k, p) for f, c, k in points for p in ("markov", "oracle")
               if not by.get((f, c, k, "he_urg", p))]
    print()
    if missing:
        print(f"WARNING: {len(missing)} grid points have no he_urg runs; refetch columns there are nan")
    print("G1_GATE", "PASS" if gate else "FAIL")


if __name__ == "__main__":
    main()
