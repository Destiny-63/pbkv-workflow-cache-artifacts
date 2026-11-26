"""A3 trace statistics for HoVer-MA v2. Usage: python3 -m pbkv_sim.a3_stats --data DIR --split FILE --out FILE"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .predictors import MarkovPredictor, evaluate
from .trace import Workflow, load_workflows, mix_settings


def describe(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    mean = float(a.mean())
    return {
        "n": int(a.size), "mean": mean, "cv": float(a.std() / mean) if mean else 0.0,
        "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
    }


def per_role(wfs: list[Workflow]) -> dict:
    out = {}
    roles = sorted({c.role for w in wfs for c in w.calls})
    for role in roles:
        cs = [c for w in wfs for c in w.calls if c.role == role]
        tool = [c.tool_end - c.tool_start for c in cs if c.tool_start is not None and c.tool_end is not None]
        out[role] = {
            "service_s": describe([c.service_time for c in cs]),
            "output_tokens": describe([c.output_tokens for c in cs]),
            "prompt_tokens": describe([c.prompt_tokens for c in cs]),
            "cached_ratio": describe([c.cached_tokens / c.prompt_tokens for c in cs if c.prompt_tokens]),
            "tool_wait_s": describe(tool),
        }
    return out


def structure(wfs: list[Workflow]) -> dict:
    trans = Counter()
    for w in wfs:
        seq = w.roles() + ["END"]
        trans.update(f"{a}->{b}" for a, b in zip(seq, seq[1:]))
    reader_group = Counter(len(s.calls) for w in wfs for s in w.steps() if s.role == "Reader")
    return {
        "workflows": len(wfs),
        "steps_per_workflow": describe([len(w.steps()) for w in wfs]),
        "rounds": dict(Counter(max(c.round for c in w.calls) for w in wfs)),
        "verdicts": dict(Counter(w.verdict for w in wfs)),
        "transitions": dict(sorted(trans.items())),
        "reader_group_size": dict(reader_group),
        "end_signal_rate": 1.0,
    }


def prefix_stats(wfs: list[Workflow]) -> dict:
    first_seg = Counter(c.segments[0] for w in wfs for c in w.calls if c.segments)
    manual_tokens = first_seg.most_common(1)[0][0][1] if first_seg else 0
    final = [max(c.segments[-1][1] for c in w.calls if c.segments) for w in wfs if any(c.segments for c in w.calls)]
    return {
        "manual_tokens": manual_tokens,
        "workflow_max_context_tokens": describe(final),
        "workflow_private_tokens": describe([f - manual_tokens for f in final]),
    }


def progress_dispersion(wfs: list[Workflow], n_active: int, samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    durations = [[s.end - s.start for s in w.steps()] for w in wfs]
    cvs = []
    for _ in range(samples):
        rem = []
        for i in rng.choice(len(wfs), size=min(n_active, len(wfs)), replace=False):
            d = durations[i]
            t = rng.uniform(0, sum(d))
            acc = 0.0
            for x in d:
                if acc + x > t:
                    rem.append(acc + x - t)
                    break
                acc += x
        r = np.asarray(rem)
        if r.size > 1 and r.mean() > 0:
            cvs.append(r.std() / r.mean())
    return float(np.mean(cvs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    split = json.loads(Path(args.split).read_text())
    fast = {k: v for k, v in load_workflows(data / "hover_default.jsonl").items() if k in split}
    slow = {k: v for k, v in load_workflows(data / "hover_highdiff.jsonl").items() if k in split}
    report = {"settings": {}, "markov": {}}
    for name, wfs in (("default", fast), ("highdiff", slow)):
        all_wfs = list(wfs.values())
        report["settings"][name] = {
            "structure": structure(all_wfs),
            "per_role": per_role(all_wfs),
            "prefix": prefix_stats(all_wfs),
        }
    for f in (0.0, 0.25, 0.5, 1.0):
        mixed = mix_settings(fast, slow, f, seed=0)
        report.setdefault("progress_dispersion_cv", {})[str(f)] = {
            str(n): progress_dispersion(mixed, n, 500, seed=1) for n in (16, 32, 64)
        }
    for name, wfs in (("default", fast), ("highdiff", slow), ("mix_0.5", None)):
        pool = dict(zip(fast, mix_settings(fast, slow, 0.5, seed=0))) if wfs is None else wfs
        train = [w.roles() for k, w in pool.items() if split[k] == "train"]
        test = [w.roles() for k, w in pool.items() if split[k] == "test"]
        m = MarkovPredictor(alpha=0.5).fit(train)
        report["markov"][name] = evaluate(lambda h: m.predict(h[-1], 3), test, k_max=3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["markov"], indent=2))


if __name__ == "__main__":
    main()
