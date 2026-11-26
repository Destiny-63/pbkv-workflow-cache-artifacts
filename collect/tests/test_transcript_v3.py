import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcript import IM_START, RoleFirstTranscript, prefix_segments, role_reuse_report

GEN = f"{IM_START}assistant\n"
SYS = {"Decomposer": "MANUAL\n\nsplit", "Reader": "MANUAL\n\nread", "Aggregator": "MANUAL\n\nagg"}


def render(messages):
    body = "".join(f"{IM_START}{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return body + GEN


def count(text):
    return len(text.split())


def call(t, role, task_input):
    return render(t.messages(SYS[role], role, task_input))


def test_same_role_next_round_extends_its_own_previous_prompt():
    t = RoleFirstTranscript(claim="C")
    t.record("Decomposer", "Start.", "{}")
    t.record("Reader", "Q1", "{}")
    a1 = call(t, "Aggregator", "notes 1")
    t.record("Aggregator", "notes 1", "{}")
    t.record("Decomposer", "Round 2.", "{}")
    a2 = call(t, "Aggregator", "notes 2")
    assert a2.startswith(a1[: -len(GEN)])


def test_different_roles_share_no_segment():
    t = RoleFirstTranscript(claim="C")
    t.record("Decomposer", "Start.", "{}")
    r = prefix_segments(call(t, "Reader", "Q1"), count)
    a = prefix_segments(call(t, "Aggregator", "notes"), count)
    assert not {h for h, _ in r} & {h for h, _ in a}


def test_parallel_readers_share_until_their_own_ask():
    t = RoleFirstTranscript(claim="C")
    t.record("Decomposer", "Start.", "{}")
    a = prefix_segments(call(t, "Reader", "Q1"), count)
    b = prefix_segments(call(t, "Reader", "Q2"), count)
    assert a[:-2] == b[:-2] and a[-2][0] != b[-2][0]


def test_role_reuse_report_counts_private_same_role_reuse():
    t = RoleFirstTranscript(claim="C")
    rows = []
    for i, (role, inp) in enumerate([("Decomposer", "s"), ("Reader", "q1"), ("Reader", "q2"),
                                     ("Aggregator", "n1"), ("Decomposer", "s2")], start=1):
        segs = prefix_segments(call(t, role, inp), count)
        rows.append({"event": "call", "workflow_id": "w", "call_index": i, "role": role, "segments": segs,
                     "segment_tokens": segs[-1][1], "prompt_tokens": segs[-1][1], "cached_tokens": 0})
        t.record(role, inp, "{}")
    rep = role_reuse_report(rows)
    assert rep["repeat_role_calls"] == 2
    assert rep["repeat_role_private_share"] == 1.0
    assert rep["segment_mismatch_rate"] == 0.0
