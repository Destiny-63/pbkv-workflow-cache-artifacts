"""Accumulated-context prompts for HoVer-MA v2 (no heavy dependencies)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

IM_START = "<|im_start|>"
PASSAGE_CHARS = 600


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ask_text(role: str, instruction: str, task_input: str) -> str:
    return f"[{role}]\n{instruction}\n\n{task_input}".rstrip()


def answer_for_history(raw: str, parsed: dict | None) -> str:
    if parsed is not None:
        return json.dumps(parsed, ensure_ascii=False)
    return raw.split("</think>")[-1].strip()


@dataclass
class Transcript:
    manual: str
    claim: str
    turns: list[dict] = field(default_factory=list)

    def head(self) -> list[dict]:
        return [
            {"role": "system", "content": self.manual},
            {"role": "user", "content": f"Claim: {self.claim}"},
        ]

    def messages(self, role: str, instruction: str, task_input: str) -> list[dict]:
        ask = {"role": "user", "content": ask_text(role, instruction, task_input)}
        return self.head() + self.turns + [ask]

    def record(self, role: str, instruction: str, task_input: str, answer: str) -> None:
        self.turns.append({"role": "user", "content": ask_text(role, instruction, task_input)})
        self.turns.append({"role": "assistant", "content": answer})


def prefix_segments(prompt: str, count_tokens: Callable[[str], int]) -> list[list]:
    segments = []
    cum_text = ""
    cum_tokens = 0
    for i, piece in enumerate(prompt.split(IM_START)):
        text = piece if i == 0 else IM_START + piece
        if not text:
            continue
        cum_text += text
        cum_tokens += count_tokens(text)
        segments.append([sha16(cum_text), cum_tokens])
    return segments


def passages_block(docs: list[dict], picks: list[int]) -> str:
    blocks = []
    for n, doc_i in enumerate(picks, start=1):
        doc = docs[doc_i]
        blocks.append(f"[{n}] title: {doc['title']}\n{doc['text'][:PASSAGE_CHARS]}")
    return "\n\n".join(blocks)


def role_ask(role: str, task_input: str) -> str:
    return f"[{role}]\n{task_input}".rstrip()


@dataclass
class RoleFirstTranscript:
    claim: str
    turns: list[dict] = field(default_factory=list)

    def messages(self, system: str, role: str, task_input: str) -> list[dict]:
        head = [{"role": "system", "content": system}, {"role": "user", "content": f"Claim: {self.claim}"}]
        return head + self.turns + [{"role": "user", "content": role_ask(role, task_input)}]

    def record(self, role: str, task_input: str, answer: str) -> None:
        self.turns.append({"role": "user", "content": role_ask(role, task_input)})
        self.turns.append({"role": "assistant", "content": answer})


def _common_tokens(a, b) -> int:
    tokens = 0
    for (ha, ta), (hb, _tb) in zip(a, b):
        if ha != hb:
            break
        tokens = ta
    return tokens


def role_reuse_report(rows: list[dict]) -> dict:
    calls = sorted((r for r in rows if r.get("event") == "call" and r.get("segments")),
                   key=lambda r: (r["workflow_id"], r["call_index"]))
    seen = defaultdict(list)
    repeat = private = 0
    for r in calls:
        earlier = seen[(r["workflow_id"], r["role"])]
        if earlier:
            repeat += 1
            if max(_common_tokens(e, r["segments"]) for e in earlier) > r["segments"][0][1]:
                private += 1
        earlier.append(r["segments"])
    later = sorted((r.get("cached_tokens") or 0) / r["prompt_tokens"]
                   for r in calls if r["call_index"] > 1 and r.get("prompt_tokens"))
    mismatch = sum(1 for r in calls if abs((r.get("segment_tokens") or 0) - (r.get("prompt_tokens") or 0)) > 2)
    return {
        "repeat_role_calls": repeat,
        "repeat_role_private_share": private / max(repeat, 1),
        "reuse_median": later[len(later) // 2] if later else 0.0,
        "max_prompt_tokens": max((r.get("prompt_tokens") or 0 for r in calls), default=0),
        "segment_mismatch_rate": mismatch / max(len(calls), 1),
    }
