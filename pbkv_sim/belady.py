"""Leaf-constrained, variable-size offline reference; not an optimality bound."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from math import inf


@dataclass
class NextUse:
    times: dict[tuple[str, ...], tuple[float, ...]]

    def after(self, prefix_key, t: float) -> float:
        times = self.times.get(tuple(prefix_key), ())
        i = bisect_right(times, t)
        return times[i] if i < len(times) else inf


def build_next_use(access_log) -> NextUse:
    times = defaultdict(list)
    for t, _wid, _idx, keys in access_log:
        for key in keys:
            times[tuple(key)].append(float(t))
    return NextUse({key: tuple(sorted(set(values))) for key, values in times.items()})
