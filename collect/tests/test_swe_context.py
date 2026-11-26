"""Synthetic source visibility and native request checks, not benchmark scores."""
import json
import subprocess

from collect.swe_context import issue_terms, repository_context_v2, source_spans
from collect.swe_ma import sampling_params
from collect.swe_edits import EDIT_SCHEMA


def commit(repo):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Synthetic", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "synthetic context"], check=True)


def test_retrieval_includes_constructor_after_large_prose_and_ignores_mutations(tmp_path):
    source = 'import os\n\nclass Blueprint:\n    """' + "Documented behavior.\n" * 1800
    source += '    """\n    def __init__(self, name):\n        self.name = name\n'
    (tmp_path / "blueprints.py").write_text(source)
    commit(tmp_path)
    (tmp_path / "blueprints.py").write_text("MUTATED_NOT_BASE\n")
    result = repository_context_v2(tmp_path, "HEAD", "Blueprint name must reject an empty name")
    assert "def __init__(self, name):\n        self.name = name\n" in result
    assert "class Blueprint:" in result
    assert "MUTATED_NOT_BASE" not in result
    assert len(result) <= 48000


def test_example_test_basename_does_not_displace_source(tmp_path):
    for i in range(12):
        path = tmp_path / "testing" / str(i) / "test_foo.py"
        path.parent.mkdir(parents=True)
        path.write_text("def test_foo():\n    pass\n")
    (tmp_path / "skipping.py").write_text("def evaluate_xfail_marks(item):\n    return item.xfail\n")
    commit(tmp_path)
    result = repository_context_v2(tmp_path, "HEAD", "test_foo.py calls xfail and evaluate_xfail_marks fails")
    assert result.startswith("\n--- skipping.py")
    assert "return item.xfail" in result


def test_issue_title_and_class_beat_broad_generic_usage(tmp_path):
    (tmp_path / "target.py").write_text("class OrderedThing:\n    def __len__(self):\n        return 0\n")
    for i in range(8):
        (tmp_path / f"general{i}.py").write_text("def __reversed__():\n" + "    # reversed natural ordered argument support\n" * 100 + "    return []\n")
    commit(tmp_path)
    result = repository_context_v2(tmp_path, "HEAD", "Support reversed on OrderedThing\nAdd __reversed__ for natural ordered argument support")
    assert result.startswith("\n--- target.py")
    terms = issue_terms("Dynamic xfail marker\nrequest.node.add_marker fails with xfail\n## Versions\n" + "UnrelatedPackage 1.0\n" * 30)
    assert terms.index("xfail") < terms.index("UnrelatedPackage")


def test_source_spans_keep_complete_lines_and_bounded_budget():
    source = "# header\n" * 200 + "def focused():\n    return 'target'\n"
    lines = source.splitlines(keepends=True)
    spans = source_spans(source, ["focused"], 120)
    visible = "".join("".join(lines[a:b]) for a, b in spans)
    assert "def focused():\n    return 'target'\n" in visible
    assert len(visible) <= 120


def test_only_v2_edit_roles_use_native_schema_and_registered_budget():
    for protocol in ("unified_diff_v1", "exact_edits_v1", "exact_edits_v2"):
        for role in ("Planner", "Coder", "Committer", "Tester", "Analyzer"):
            params = sampling_params(protocol, role, 17)
            assert params["sampling_seed"] == 17
            constrained = protocol == "exact_edits_v2" and role in {"Coder", "Committer"}
            assert ("json_schema" in params) == constrained
            if constrained:
                assert json.loads(params["json_schema"]) == EDIT_SCHEMA
                assert params["max_new_tokens"] == 4096
            elif role in {"Coder", "Committer"}:
                assert params["max_new_tokens"] == 3072
