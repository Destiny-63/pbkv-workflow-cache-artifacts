"""Isolated synthetic fixtures for an exact-edit development protocol."""
import json
import subprocess

import pytest

from collect.swe_edits import compile_edits


@pytest.fixture
def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "example.py").write_text("def value():\n    return 1\n\n# repeated\n# repeated\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Synthetic", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "synthetic fixture"], check=True)
    return tmp_path


def payload(path="example.py", search="    return 1\n", replace="    return 2\n"):
    return json.dumps({"edits": [dict(path=path, search=search, replace=replace)]})


def test_exact_edits_compile_and_apply_without_mutating_base(repository):
    patch, edits = compile_edits(payload(), repository, "HEAD")
    assert edits["edits"][0]["replace"] == "    return 2\n"
    subprocess.run(["git", "-C", str(repository), "apply", "--check", "-"],
                   input=patch, text=True, check=True)
    assert "return 1" in (repository / "example.py").read_text()
    subprocess.run(["git", "-C", str(repository), "apply", "-"], input=patch, text=True, check=True)
    assert "return 2" in (repository / "example.py").read_text()


@pytest.mark.parametrize("text", [payload(search="return 999\n"),
    payload(search="# repeated\n"), payload(path="../example.py"),
    payload(path=".git/config"), payload(path="missing.py"),
    payload(replace="    return 1\n"), payload(search="# repeated\n# repeated\n", replace="# end"),
    '{"edits":[],"extra":1}'])
def test_invalid_ambiguous_or_inexact_edits_fail(repository, text):
    with pytest.raises(ValueError):
        compile_edits(text, repository, "HEAD")
