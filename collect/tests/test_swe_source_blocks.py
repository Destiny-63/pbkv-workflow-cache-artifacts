"""Isolated synthetic source-ID contract checks, never benchmark results."""
import copy
import json
import subprocess

import pytest

from collect.swe_core import Transcript, digest
from collect.swe_edits import compile_edits
from collect.swe_ma import sampling_params
from collect.swe_source_blocks import source_blocks, resolve_source_blocks


def repository(tmp_path, source):
    (tmp_path / "example.py").write_text(source)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Synthetic", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "synthetic source IDs"], check=True)
    return tmp_path


def build(repo, end):
    return source_blocks(repo, "HEAD", [{"path": "example.py", "start": 0, "end": end}])


def test_source_ids_are_unique_visible_immutable_blocks_and_apply_exactly(tmp_path):
    repo = repository(tmp_path, "def value():\n    return 0\n\ndef private_unseen():\n    return 42\n")
    context, schema, evidence = build(repo, 2)
    sid, item = next(iter(evidence["choices"].items()))
    assert item["search"] == "def value():\n    return 0\n"
    assert "private_unseen" not in context and "42" not in context
    assert evidence["sources"][0]["uncovered_nonblank_lines"] == []
    assert evidence["rendered_context_sha256"] == digest(context)
    payload = {"edits": [{"source_id": sid, "replace": "def value():\n    return 7\n"}]}
    resolved = resolve_source_blocks(payload, evidence)
    assert payload["edits"][0]["source_id"] == sid  # no model-output mutation
    patch, edits = compile_edits(json.dumps(resolved), repo, "HEAD")
    subprocess.run(["git", "-C", str(repo), "apply", "--check", "-"], input=patch, text=True, check=True)
    assert edits["edits"][0]["path"] == "example.py"
    (repo / "example.py").write_text("MUTATED_WORKTREE\n")
    assert build(repo, 2) == (context, schema, evidence)


@pytest.mark.parametrize("payload", [
    {"edits": [{"source_id": "unknown", "replace": "anything\n"}]},
    {"edits": [{"source_id": 0, "replace": "anything\n"}]},
    {"edits": [{"source_id": "unknown", "replace": "anything", "path": "example.py"}]},
    {"edits": []}, {"edits": "invalid"}, {"edits": [], "extra": 1},
])
def test_unknown_inferred_or_malformed_ids_are_rejected(tmp_path, payload):
    repo = repository(tmp_path, "def value():\n    return 0\n")
    _, _, evidence = build(repo, 2)
    with pytest.raises(ValueError):
        resolve_source_blocks(payload, evidence)


def test_repeated_ids_corrupted_mapping_and_noops_fail(tmp_path):
    repo = repository(tmp_path, "def value():\n    return 0\n")
    _, _, evidence = build(repo, 2)
    sid, item = next(iter(evidence["choices"].items()))
    edit = {"source_id": sid, "replace": "def value():\n    return 1\n"}
    with pytest.raises(ValueError, match="overlap"):
        resolve_source_blocks({"edits": [edit, edit]}, evidence)
    corrupted = copy.deepcopy(evidence)
    corrupted["choices"][sid]["search"] = "CORRUPTED\n"
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_source_blocks({"edits": [edit]}, corrupted)
    payload = {"edits": [{"source_id": sid, "replace": item["search"]}]}
    with pytest.raises(ValueError, match="change file content"):
        compile_edits(json.dumps(resolve_source_blocks(payload, evidence)), repo, "HEAD")


def test_all_visible_lines_remain_selectable_beyond_96_options(tmp_path):
    source = "".join(f"v_{i:04d} = {i}\n" for i in range(1200))
    repo = repository(tmp_path, source)
    context, schema, evidence = build(repo, 1200)
    assert len(evidence["choices"]) == 150
    assert evidence["sources"][0]["covered_nonblank_lines"] == 1200
    assert evidence["sources"][0]["uncovered_nonblank_lines"] == []
    assert len(context) <= 64000
    assert set(schema["properties"]["edits"]["items"]["properties"]["source_id"]["enum"]) == set(evidence["choices"])


def test_unselectable_long_line_is_reported_without_truncating_source(tmp_path):
    source = "value = 1\n" + "#" + "x" * 2100 + "\n" + "other = 2\n"
    repo = repository(tmp_path, source)
    context, _, evidence = build(repo, 3)
    assert "x" * 2100 in context and "NOT_SELECTABLE" in context
    assert evidence["sources"][0]["uncovered_nonblank_lines"] == [2]
    assert all(len(c["search"]) <= 2048 for c in evidence["choices"].values())
    with pytest.raises(ValueError, match="outside base"):
        build(repo, 99)


def test_v1_id_schema_and_prompt_are_explicit_opt_in(tmp_path):
    repo = repository(tmp_path, "value = 0\n")
    _, schema, _ = build(repo, 1)
    for role in ("Coder", "Committer"):
        params = sampling_params("source_blocks_v1", role, 2, schema)
        assert json.loads(params["json_schema"]) == schema and params["max_new_tokens"] == 4096
    assert "json_schema" not in sampling_params("source_blocks_v1", "Planner", 1)
    with pytest.raises(ValueError, match="requires a recorded"):
        sampling_params("source_blocks_v1", "Coder", 2)
    assert '"source_id"' in Transcript({}, "", patch_protocol="source_blocks_v1").instructions["Coder"]
    assert '"search"' in Transcript({}, "", patch_protocol="exact_edits_v3").instructions["Coder"]
