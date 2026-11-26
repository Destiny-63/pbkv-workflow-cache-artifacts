import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcript import IM_START, Transcript, answer_for_history, passages_block, prefix_segments

GEN = f"{IM_START}assistant\n"


def render(messages):
    body = "".join(f"{IM_START}{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return body + GEN


def count(text):
    return len(text.split())


def test_next_prompt_extends_previous_prompt_except_generation_tail():
    t = Transcript(manual="MANUAL", claim="X was born in Y.")
    p1 = render(t.messages("Decomposer", "Split it.", "Start."))
    t.record("Decomposer", "Split it.", "Start.", '{"subquestions": ["a", "b"]}')
    p2 = render(t.messages("Reader", "Read.", "Subquestion: a"))
    assert p2.startswith(p1[: -len(GEN)])


def test_segments_share_all_but_generation_tail():
    t = Transcript(manual="MANUAL", claim="C")
    s1 = prefix_segments(render(t.messages("Decomposer", "Split.", "Start.")), count)
    t.record("Decomposer", "Split.", "Start.", "{}")
    p2 = render(t.messages("Reader", "Read.", "Q"))
    s2 = prefix_segments(p2, count)
    n = len(s1) - 1
    assert [h for h, _ in s1[:n]] == [h for h, _ in s2[:n]]
    assert s1[n][0] != s2[n][0]
    pieces = [p if i == 0 else IM_START + p for i, p in enumerate(p2.split(IM_START)) if p]
    assert s2[-1][1] == sum(count(p) for p in pieces)


def test_parallel_siblings_share_transcript_prefix():
    t = Transcript(manual="MANUAL", claim="C")
    t.record("Decomposer", "Split.", "Start.", "{}")
    a = prefix_segments(render(t.messages("Reader", "Read.", "Q1")), count)
    b = prefix_segments(render(t.messages("Reader", "Read.", "Q2")), count)
    assert a[:-2] == b[:-2]
    assert a[-2][0] != b[-2][0]


def test_answer_for_history_drops_thinking():
    raw = "<think>\nlong reasoning\n</think>\n\n{\"verdict\": \"SUPPORTED\"}"
    assert answer_for_history(raw, None) == '{"verdict": "SUPPORTED"}'
    assert answer_for_history(raw, {"verdict": "SUPPORTED"}) == '{"verdict": "SUPPORTED"}'


def test_passages_are_truncated():
    docs = [{"title": "T", "text": "a" * 5000}]
    block = passages_block(docs, [0])
    assert block.startswith("[1] title: T\n")
    assert len(block) == len("[1] title: T\n") + 600
