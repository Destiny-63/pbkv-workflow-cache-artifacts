"""Isolated synthetic checks; these are not benchmark results."""
import asyncio
import json
import struct
import subprocess
from argparse import Namespace

import httpx
import numpy as np
import pytest

from collect.swe_core import Transcript, MANUAL, INSTRUCTIONS, digest, extract_patch, next_roles, repository_context, terminal_status, token_segments
from collect.swe_ma import Collector, hidden_vector


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"


def issue():
    return {"repo": "synthetic/example", "base_commit": "synthetic-base", "problem_statement": "Support OrderedThing", "instance_id": "synthetic-1",
            "patch": "PRIVATE_GOLD_FIX", "test_patch": "PRIVATE_REFERENCE_TEST", "FAIL_TO_PASS": "PRIVATE_TEST_IDS"}


def test_patch_extraction_preserves_trailing_context_blank_line():
    diff = "--- a/example.py\n+++ b/example.py\n@@ -1,2 +1,3 @@\n x = 1\n+y = 2\n \n"
    assert extract_patch("```diff\n" + diff + "```") == diff
    assert extract_patch(diff) == diff


def test_prompt_does_not_leak_evaluation_metadata():
    tr = Transcript(issue(), "class OrderedThing: pass")
    tr.record("Planner", "plan")
    tr.tool({"resolved": False, "test_output_tail": "real tool failure"})
    text = json.dumps(tr.messages("Coder"))
    assert "PRIVATE_" not in text
    assert text.index("plan") < text.index("real tool failure") < text.rindex("[Coder]")
    assert tr.messages("Coder")[0]["content"].startswith(MANUAL)


def test_exact_edit_protocol_is_opt_in_and_records_matching_history():
    legacy = Transcript(issue(), "source")
    exact = Transcript(issue(), "source", patch_protocol="exact_edits_v1")
    assert "unified diff" in legacy.system("Coder")
    assert '"edits"' in exact.system("Coder")
    exact.record("Coder", '{"edits":[]}')
    prompt = json.dumps(exact.messages("Committer"))
    assert "PRIVATE_" not in prompt
    assert exact.turns[0]["content"] == "[Coder]\n" + exact.instructions["Coder"]


def test_segments_use_exact_token_prefixes_and_share_manual():
    tok = ByteTokenizer()
    first = tok.apply_chat_template(Transcript(issue(), "source").messages("Planner"))
    second = tok.apply_chat_template(Transcript(issue(), "source").messages("Coder"))
    ids, segments = token_segments(first, tok)
    _, other = token_segments(second, tok)
    assert segments[0] == other[0]
    assert segments[-1][1] == len(ids)
    assert all(a[1] < b[1] for a, b in zip(segments, segments[1:]))
    for hashed, n in segments:
        assert hashed == digest(struct.pack(f"<{n}I", *ids[:n]))


def test_patch_paths_and_round_trip(tmp_path):
    patch = "--- a/thing.py\n+++ b/thing.py\n@@ -1 +1 @@\n-old\n+new\n"
    assert extract_patch("```diff\n" + patch + "```\n") == patch
    assert extract_patch("<think>notes</think>\n" + patch) == patch
    for invalid in [patch.replace("a/thing.py", "a/../secret"), patch.replace("b/thing.py", "b/.git/config"), "tests pass"]:
        with pytest.raises(ValueError):
            extract_patch(invalid)


def test_retry_routing_and_cap():
    assert next_roles(1) == ("Planner", "Coder", "Committer", "Tester")
    assert next_roles(2) == ("Analyzer", "Coder", "Committer", "Tester")
    assert terminal_status({"status": "EVALUATED", "resolved": False}, 2) is None
    assert terminal_status({"status": "EVALUATED", "resolved": False}, 3) == "MAX_ATTEMPTS"
    assert terminal_status({"status": "EVALUATED", "resolved": True}, 1) == "RESOLVED"
    assert terminal_status({"status": "ERROR", "resolved": False}, 1) == "ERROR"
    assert terminal_status({"status": "PATCH_APPLY_FAILED", "resolved": False}, 1) == "PATCH_APPLY_FAILED"


def test_stock_schema_extension_preserves_original_fields():
    from collect.patch_swe_request import patched
    source = "from dataclasses import dataclass\nfrom typing import Optional,List\n@dataclass\nclass GenerateReqInput:\n    text: Optional[str] = None\n"
    namespace = {}
    exec(compile(patched(source), "synthetic_schema.py", "exec"), namespace)
    cls = namespace["GenerateReqInput"]
    request = cls(text="synthetic", my_chain_id="w", my_current_agent="Coder", my_chain_history=["Planner"], my_is_selector=False)
    assert request.text == "synthetic" and request.my_chain_history == ["Planner"]
    with pytest.raises(ValueError):
        patched(patched(source))


def test_hidden_requires_one_finite_nonzero_prefill_vector():
    assert hidden_vector({"hidden_states": [[1, 2, 3]]}, 3).shape == (3,)
    for raw in [None, [0, 0, 0], [1, float("nan"), 0], [[1, 2, 3], [4, 5, 6]], [1, 2]]:
        with pytest.raises(ValueError):
            hidden_vector({"hidden_states": raw}, 3)


def test_repository_context_reads_base_not_mutated_worktree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "thing.py").write_text("class OrderedThing:\n    pass\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "thing.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Synthetic Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "synthetic fixture"], check=True)
    base = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    (tmp_path / "thing.py").write_text("NOT_FROM_BASE")
    text = repository_context(tmp_path, base, "OrderedThing cannot be reversed")
    assert "OrderedThing" in text and "NOT_FROM_BASE" not in text
    assert len(text) <= 12000


def test_repository_context_finds_definition_after_many_prose_hits(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for index in range(165):
        (tmp_path / f"a{index:03d}.py").write_text("# Allow calling reversed on collections\n")
    (tmp_path / "z_impl.py").write_text("class OrderedThing:\n    def __init__(self):\n        self.items = {}\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Synthetic Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "synthetic retrieval fixture"], check=True)
    text = repository_context(tmp_path, "HEAD", "Allow calling reversed on OrderedThing")
    assert "class OrderedThing:" in text
    assert text.startswith("\n--- z_impl.py")


def test_actual_stream_contract_and_history_are_recorded(tmp_path):
    async def check():
        args = Namespace(out=tmp_path, workers=1, concurrency=1, context_length=20000, base_url="http://synthetic.invalid", model_label="synthetic")
        collector = Collector(args, ByteTokenizer(), 3)
        await collector.client.aclose()
        seen = []

        def respond(request):
            payload = json.loads(request.content)
            seen.append(payload)
            item = {"text": "synthetic response", "meta_info": {"hidden_states": [[1, 2, 3]], "prompt_tokens": len(payload["input_ids"]), "completion_tokens": 2, "cached_tokens": 0}}
            return httpx.Response(200, text="data: " + json.dumps(item) + "\n\ndata: [DONE]\n\n")

        collector.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        history = ["Planner"]
        await collector.call(issue(), "synthetic_wf", Transcript(issue(), "source"), "Coder", history, 1, 2, 0)
        saved = collector.rows[0]
        assert saved["my_chain_history"] == ["Planner"]
        assert history == ["Planner", "Coder"]
        assert seen[0]["my_current_agent"] == "Coder"
        assert seen[0]["sampling_params"]["sampling_seed"] == 2
        assert "seed" not in seen[0]["sampling_params"]
        assert saved["queue_start"] <= saved["prefill_end"] <= saved["decode_end"]
        np.testing.assert_array_equal(np.load(saved["hidden_file"]), [1, 2, 3])
        assert saved["segments"][-1][1] == saved["prompt_tokens"]
        await collector.client.aclose()

    asyncio.run(check())


def test_server_patch_keeps_prefill_vector_during_decode():
    from collect.patch_swe_hidden import patch_output, patch_schedule
    from types import SimpleNamespace
    original = '''def prefill(req, logits_output, i):
    hidden_state_offset = 0
    if req.return_hidden_states and logits_output.hidden_states is not None:
        pass
    if req.return_hidden_states:
        req.hidden_states.append(
            logits_output.hidden_states[
                hidden_state_offset : (
                    hidden_state_offset := hidden_state_offset
                    + len(req.origin_input_ids)
                )
            ]
            .cpu()
            .clone()
            .tolist()
        )
def decode(req, logits_output, i):
    if req.return_hidden_states and logits_output.hidden_states is not None:
        req.hidden_states.append(
            logits_output.hidden_states[i].cpu().clone().tolist()
        )
'''
    # Match the official prefill condition (it spans multiple lines there),
    # leaving the one-line condition as the unique decode replacement target.
    original = original.replace("    if req.return_hidden_states and logits_output.hidden_states is not None:\n        pass\n", "")
    namespace = {}
    exec(compile(patch_output(original), "synthetic_capture.py", "exec"), namespace)

    class Vector:
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def clone(self): return self
        def tolist(self): return [1, 2, 3]

    req = SimpleNamespace(return_hidden_states=True, hidden_states=[], origin_input_ids=[7, 8])
    output = SimpleNamespace(hidden_states=[Vector()])
    namespace["prefill"](req, output, 0)
    namespace["decode"](req, output, 0)
    assert req.hidden_states == [[1, 2, 3]]
    assert "CaptureHiddenMode.LAST" in patch_schedule("CaptureHiddenMode.FULL\n                if self.return_hidden_states")
    with pytest.raises(ValueError):
        patch_output("unexpected scheduler implementation")


def test_test_markers_split_across_docker_chunks_preserve_output():
    from collect.swe_harness import observed_test_stream
    result = {"exec_start": None, "exec_end": None}
    chunks = [b"setup\nSTART_", b"TEST\nactual tests\nEND", b"_TEST\ncleanup"]
    assert list(observed_test_stream(chunks, result, b"START_TEST", b"END_TEST")) == chunks
    assert result["exec_start"] <= result["exec_end"]
    assert result["exec_monotonic_seconds"] >= 0
    incomplete = {"exec_start": None, "exec_end": None}
    list(observed_test_stream([b"setup failed before tests"], incomplete, b"START_TEST", b"END_TEST"))
    assert incomplete["exec_start"] is None and incomplete["exec_end"] is None
    large = {"exec_start": None, "exec_end": None}
    list(observed_test_stream([b"START_TEST\n" + b"x" * 10000 + b"\nEND_TEST"], large, b"START_TEST", b"END_TEST"))
    assert large["exec_start"] is not None and large["exec_end"] is not None


def test_collected_trace_loads_and_replays_with_external_wait(tmp_path):
    from collect.validate_swe import validate
    from collect.swe_core import ROLES
    (tmp_path / "calls").mkdir()
    (tmp_path / "hidden").mkdir()
    (tmp_path / "evaluations/synthetic_r1").mkdir(parents=True)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"hidden_dimension": 3}))
    (tmp_path / "dataset.json").write_text(json.dumps({"roles": [*ROLES, "END"]}))
    rows = []
    ids = [1, 2, 3]
    for i, (role, start) in enumerate([("Planner", 0), ("Coder", 31)], 1):
        hidden = tmp_path / "hidden" / f"synthetic_{i:02d}.npy"
        np.save(hidden, np.ones(3, dtype=np.float16))
        (tmp_path / "calls" / f"synthetic_{i:02d}.request.json").write_text(json.dumps({"input_ids": ids}))
        rows.append({"event": "call", "workflow_id": "synthetic", "setting": "synthetic", "role": role,
                     "call_index": i, "queue_start": start, "prefill_end": start + .1, "decode_end": start + 1,
                     "prompt_tokens": 3, "segments": [[digest(struct.pack("<3I", *ids)), 3]],
                     "my_chain_history": [] if i == 1 else ["Planner"], "hidden_file": str(hidden)})
    rows.append({"event": "tool", "workflow_id": "synthetic", "after_call_index": 1,
                 "tool_type": "synthetic_wait", "queue_start": 1, "container_start": 1, "exec_start": 1, "exec_end": 31})
    rows.append({"event": "workflow_end", "workflow_id": "synthetic", "instance_id": "synthetic_instance",
                 "setting": "synthetic", "decode_end": 32, "verdict": "MAX_ATTEMPTS"})
    (tmp_path / "evaluations/synthetic_r1/result.json").write_text(json.dumps({
        "purpose": "isolated synthetic unit-test fixture", "tests_executed": True, "status": "EVALUATED", "run_id": "synthetic_test_run"}))
    (tmp_path / "evaluations/synthetic_requests_prepare").mkdir()
    (tmp_path / "evaluations/synthetic_requests_prepare/result.json").write_text(json.dumps({"status": "PREPARED"}))
    path = tmp_path / "swe.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    checked = validate(tmp_path)
    assert checked["completed_workflows"] == 1
    assert checked["observed_tool_events"] == 1
    assert checked["verified_real_evaluations"] == 1
    assert checked["failed_evaluations"] == []
    assert checked["replay_access_policy_independence_passed"]
    rows[1]["my_chain_history"] = ["Tester"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="history"):
        validate(tmp_path)
