"""Isolated synthetic checks for constrained source selection, never fix scoring."""
import json
import subprocess

import pytest

from collect.swe_core import digest
from collect.swe_edits import compile_edits
from collect.swe_ma import sampling_params
from collect.swe_source_schema import source_choice_schema, check_source_choices


@pytest.fixture
def visible_repo(tmp_path):
    source = "class OrderedThing:\n    def __len__(self):\n        return 0\n\n\ndef not_displayed():\n    return 42\n"
    (tmp_path / "example.py").write_text(source)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Synthetic", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "synthetic source choices"], check=True)
    return tmp_path, source


def choices(repo):
    return source_choice_schema(repo, "HEAD", [{"path": "example.py", "start": 0, "end": 3}])


def test_choices_are_unique_visible_source_and_never_replacement_suggestions(visible_repo):
    repo, source = visible_repo
    schema, evidence = choices(repo)
    branch = schema["properties"]["edits"]["items"]["oneOf"][0]
    assert branch["properties"]["path"] == {"const": "example.py"}
    assert branch["properties"]["replace"] == {"type": "string"}
    options = branch["properties"]["search"]["enum"]
    visible = "".join(source.splitlines(keepends=True)[:3])
    assert "    def __len__(self):\n        return 0\n" in options
    assert all(s in visible and source.count(s) == 1 for s in options)
    assert all("not_displayed" not in s and "42" not in s for s in options)
    assert evidence["schema_sha256"] == digest(json.dumps(schema, sort_keys=True))
    assert evidence["sources"][0]["base_source_sha256"] == digest(source)


def test_registered_source_choice_compiles_and_real_git_apply_check_passes(visible_repo):
    repo, _ = visible_repo
    schema, _ = choices(repo)
    payload = {"edits": [{"path": "example.py", "search": "    def __len__(self):\n        return 0\n",
                           "replace": "    def __len__(self):\n        return 1\n"}]}
    check_source_choices(payload, schema)
    patch, _ = compile_edits(json.dumps(payload), repo, "HEAD")
    subprocess.run(["git", "-C", str(repo), "apply", "--check", "-"], input=patch, text=True, check=True)
    for path, search in (("example.py", "    def __len__(self) -> int: return 0"),
                         ("example.py", "def not_displayed():\n    return 42\n"),
                         ("missing.py", payload["edits"][0]["search"])):
        with pytest.raises(ValueError, match="registered visible"):
            check_source_choices({"edits": [{"path": path, "search": search, "replace": "anything\n"}]}, schema)


def test_schema_reads_immutable_base_and_checks_span_bounds(visible_repo):
    repo, _ = visible_repo
    first = choices(repo)
    (repo / "example.py").write_text("WORKTREE_MUTATION\n")
    assert choices(repo) == first
    with pytest.raises(ValueError, match="outside base"):
        source_choice_schema(repo, "HEAD", [{"path": "example.py", "start": 0, "end": 99}])
    with pytest.raises(ValueError, match="No unique visible"):
        source_choice_schema(repo, "HEAD", [])


def test_v3_uses_the_workflows_exact_schema_and_fails_without_one(visible_repo):
    repo, _ = visible_repo
    schema, _ = choices(repo)
    params = sampling_params("exact_edits_v3", "Coder", 2, schema)
    assert json.loads(params["json_schema"]) == schema
    assert params["max_new_tokens"] == 4096
    with pytest.raises(ValueError, match="requires a recorded"):
        sampling_params("exact_edits_v3", "Coder", 2)
    assert "json_schema" not in sampling_params("exact_edits_v3", "Planner", 1)
