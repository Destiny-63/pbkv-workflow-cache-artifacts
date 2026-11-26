"""HoVer-MA trace loading, step grouping, workflow-level split and setting mix."""

from __future__ import annotations

import hashlib
import json
import random
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Call:
    workflow_id: str
    setting: str
    role: str
    round: int
    call_index: int
    submit: float
    first_token: float
    end: float
    prompt_tokens: int
    output_tokens: int
    cached_tokens: int
    tool_start: float | None
    tool_end: float | None
    thinking: bool
    segments: tuple[tuple[str, int], ...]
    hidden_file: str | None

    @property
    def start(self) -> float:
        return self.tool_start if self.tool_start is not None else self.submit

    @property
    def service_time(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Step:
    role: str
    calls: tuple[Call, ...]

    @property
    def start(self) -> float:
        return min(c.start for c in self.calls)

    @property
    def end(self) -> float:
        return max(c.end for c in self.calls)


@dataclass
class Workflow:
    workflow_id: str
    setting: str
    thinking: bool
    calls: list[Call]
    end: float
    verdict: str | None
    tools: list[ToolEvent] = field(default_factory=list)
    instance_id: str | None = None
    repo: str | None = None

    def tools_after(self, step: Step) -> tuple[ToolEvent, ...]:
        last = max(c.call_index for c in step.calls)
        return tuple(sorted((t for t in self.tools if t.after_call_index == last),
                            key=lambda t: (t.queue_start, t.exec_end, t.tool_type)))

    def delay_after(self, step: Step) -> float:
        tools = self.tools_after(step)
        return max(0.0, max(t.exec_end for t in tools) - step.end) if tools else 0.0

    def steps(self) -> list[Step]:
        steps: list[Step] = []
        for c in self.calls:
            last = steps[-1] if steps else None
            if c.role == "Reader" and last and last.role == "Reader" and last.calls[0].round == c.round:
                steps[-1] = Step("Reader", last.calls + (c,))
            else:
                steps.append(Step(c.role, (c,)))
        return steps

    def roles(self) -> list[str]:
        return [s.role for s in self.steps()]


@dataclass(frozen=True)
class ToolEvent:
    workflow_id: str
    after_call_index: int
    tool_type: str
    queue_start: float
    exec_start: float
    exec_end: float
    container_start: float | None = None

    def __post_init__(self):
        times = [self.queue_start]
        if self.container_start is not None:
            times.append(self.container_start)
        times.extend([self.exec_start, self.exec_end])
        if not all(math.isfinite(t) for t in times) or times != sorted(times):
            raise ValueError(f"Invalid tool timestamps: {self}")

    @property
    def duration(self) -> float:
        return self.exec_end - self.queue_start


def _call(row: dict) -> Call:
    setting = row["setting"]
    return Call(
        workflow_id=row["workflow_id"],
        setting=setting,
        role=row["role"],
        round=int(row.get("round") or 1),
        call_index=int(row["call_index"]),
        submit=float(row["queue_start"]),
        first_token=float(row["prefill_end"]),
        end=float(row["decode_end"]),
        prompt_tokens=int(row.get("prompt_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        cached_tokens=int(row.get("cached_tokens") or 0),
        tool_start=row.get("tool_start"),
        tool_end=row.get("tool_end"),
        thinking=bool(row.get("thinking", setting == "highdiff" and row["role"] == "Verifier")),
        segments=tuple((str(h), int(t)) for h, t in (row.get("segments") or ())),
        hidden_file=row.get("hidden_file"),
    )


def load_workflows(path: Path) -> dict[str, Workflow]:
    calls: dict[str, list[Call]] = {}
    ends: dict[str, dict] = {}
    tools: dict[str, list[ToolEvent]] = {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "workflow_end":
                ends[row["workflow_id"]] = row
            elif row.get("event") == "call":
                calls.setdefault(row["workflow_id"], []).append(_call(row))
            elif row.get("event") == "tool":
                tool = ToolEvent(**{k: row[k] for k in (
                    "workflow_id", "after_call_index", "tool_type", "queue_start", "exec_start", "exec_end")},
                    container_start=row.get("container_start"))
                tools.setdefault(tool.workflow_id, []).append(tool)
    out = {}
    for wid, end in ends.items():
        if end.get("verdict") == "ERROR" or wid not in calls:
            continue
        cs = sorted(calls[wid], key=lambda c: c.call_index)
        thinking = end["setting"] == "highdiff" or any(c.thinking for c in cs)
        wf = Workflow(wid, end["setting"], thinking, cs, float(end["decode_end"]), end.get("verdict"),
                      tools.get(wid, []), end.get("instance_id"), end.get("repo"))
        steps = wf.steps()
        step_ends = {max(c.call_index for c in s.calls): s for s in steps}
        for tool in wf.tools:
            step = step_ends.get(tool.after_call_index)
            if step is None or tool.queue_start < step.end:
                raise ValueError(f"Tool must follow a completed step: {tool}")
            idx = steps.index(step)
            if idx + 1 < len(steps) and tool.exec_end > steps[idx + 1].start:
                raise ValueError(f"Tool overlaps next LLM step: {tool}")
        out[wid] = wf
    return out


def split_ids(ids, n_train: int = 1000, n_val: int = 200, n_test: int = 300, seed: int = 20261027) -> dict[str, str]:
    ordered = sorted(set(ids), key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    need = n_train + n_val + n_test
    if len(ordered) < need:
        raise ValueError(f"need {need} workflows, got {len(ordered)}")
    out = {}
    for i, wid in enumerate(ordered[:need]):
        out[wid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def mix_settings(fast: dict, slow: dict, fraction: float, seed: int) -> list:
    rng = random.Random(seed)
    return [slow[i] if rng.random() < fraction else fast[i] for i in sorted(set(fast) & set(slow))]
