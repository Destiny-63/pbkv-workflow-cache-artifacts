import json

from pbkv_sim.trace import load_workflows, mix_settings, split_ids


def call(wid, role, idx, rnd, t0, dur, setting="default", tool=None, segments=None, thinking=None):
    row = {
        "event": "call", "workflow_id": wid, "setting": setting, "role": role, "round": rnd,
        "call_index": idx, "queue_start": t0, "prefill_end": t0 + 0.1, "decode_end": t0 + dur,
        "prompt_tokens": 100, "output_tokens": 10, "cached_tokens": 50,
        "tool_start": tool[0] if tool else None, "tool_end": tool[1] if tool else None,
        "hidden_file": None, "segments": segments or [["h0", 50], [f"h{idx}", 100]],
    }
    if thinking is not None:
        row["thinking"] = thinking
    return row


def write(tmp_path, rows):
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_load_and_group_parallel_readers(tmp_path):
    rows = [
        call("w", "Decomposer", 1, 1, 0.0, 1.0),
        call("w", "Reader", 2, 1, 1.1, 2.0, tool=(1.0, 1.05)),
        call("w", "Reader", 3, 1, 1.1, 3.0, tool=(1.0, 1.06)),
        call("w", "Aggregator", 4, 1, 4.2, 1.0),
        call("w", "Verifier", 5, 1, 5.3, 1.0),
        call("w", "Writer", 6, 1, 6.4, 1.0),
        {"event": "workflow_end", "workflow_id": "w", "setting": "default", "decode_end": 7.5, "verdict": "SUPPORTED"},
    ]
    wfs = load_workflows(write(tmp_path, rows))
    steps = wfs["w"].steps()
    assert [s.role for s in steps] == ["Decomposer", "Reader", "Aggregator", "Verifier", "Writer"]
    assert len(steps[1].calls) == 2
    assert steps[1].start == 1.0
    assert abs(steps[1].end - 4.1) < 1e-9
    assert wfs["w"].calls[1].segments == (("h0", 50), ("h2", 100))
    assert abs(wfs["w"].calls[1].service_time - 2.1) < 1e-9


def test_readers_in_different_rounds_are_different_steps(tmp_path):
    rows = [
        call("w", "Reader", 1, 1, 0.0, 1.0),
        call("w", "Reader", 2, 2, 2.0, 1.0),
        {"event": "workflow_end", "workflow_id": "w", "setting": "default", "decode_end": 3.0, "verdict": "NEED_MORE"},
    ]
    assert len(load_workflows(write(tmp_path, rows))["w"].steps()) == 2


def test_error_and_unfinished_workflows_are_dropped(tmp_path):
    rows = [
        call("a", "Writer", 1, 1, 0.0, 1.0),
        {"event": "workflow_end", "workflow_id": "a", "setting": "default", "decode_end": 1.0, "verdict": "ERROR"},
        call("b", "Writer", 1, 1, 0.0, 1.0),
    ]
    assert load_workflows(write(tmp_path, rows)) == {}


def test_thinking_flag_defaults_from_setting(tmp_path):
    rows = [
        call("w", "Verifier", 1, 1, 0.0, 10.0, setting="highdiff"),
        {"event": "workflow_end", "workflow_id": "w", "setting": "highdiff", "decode_end": 10.0, "verdict": "SUPPORTED"},
    ]
    wf = load_workflows(write(tmp_path, rows))["w"]
    assert wf.thinking and wf.calls[0].thinking


def test_split_is_deterministic_and_sized():
    ids = [f"id{i}" for i in range(1500)]
    a = split_ids(ids)
    assert a == split_ids(list(reversed(ids)))
    assert sum(v == "train" for v in a.values()) == 1000
    assert sum(v == "val" for v in a.values()) == 200
    assert sum(v == "test" for v in a.values()) == 300


def test_mix_fraction_extremes(tmp_path):
    fast = {"x": "fast-x", "y": "fast-y"}
    slow = {"x": "slow-x", "y": "slow-y"}
    assert mix_settings(fast, slow, 0.0, seed=1) == ["fast-x", "fast-y"]
    assert mix_settings(fast, slow, 1.0, seed=1) == ["slow-x", "slow-y"]
