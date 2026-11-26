"""Compile exact model-authored edits to diffs, without fuzzy matching/repair.

Development protocol candidate only; legacy diff runs remain unchanged.
"""
import difflib
import json
from pathlib import PurePosixPath

try:
    from .swe_core import git
except ImportError:
    from swe_core import git

EDIT_SCHEMA = {
    "type": "object", "properties": {"edits": {
        "type": "array", "minItems": 1, "maxItems": 20, "items": {
            "type": "object", "properties": {
                "path": {"type": "string"}, "search": {"type": "string", "minLength": 1},
                "replace": {"type": "string"}},
            "required": ["path", "search", "replace"], "additionalProperties": False}}},
    "required": ["edits"], "additionalProperties": False,
}


def compile_edits(text, repo, commit):
    """Return canonical diff and original parsed edits. All searches must be unique."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-3].strip()
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"edits"}:
        raise ValueError("Expected an object containing only edits")
    edits = payload["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= 20:
        raise ValueError("Expected 1 to 20 exact edits")
    base_paths = set(git(repo, "ls-tree", "-r", "--name-only", commit).splitlines())
    originals, updated = {}, {}
    for edit in edits:
        if (not isinstance(edit, dict) or set(edit) != {"path", "search", "replace"}
                or not all(isinstance(v, str) for v in edit.values())):
            raise ValueError("Each edit must contain string path, search, replace")
        path, search, replacement = edit["path"], edit["search"], edit["replace"]
        pure = PurePosixPath(path)
        if (pure.is_absolute() or ".." in pure.parts or ".git" in pure.parts
                or str(pure) != path or path not in base_paths):
            raise ValueError("Edit must name an existing base-commit file")
        if not search or search == replacement:
            raise ValueError("Search must be nonempty and change file content")
        if git(repo, "ls-tree", commit, "--", path).split()[0] not in ("100644", "100755"):
            raise ValueError("Edit target must be a regular tracked file")
        if path not in originals:
            originals[path] = git(repo, "show", f"{commit}:{path}")
            updated[path] = originals[path]
            if originals[path] and not originals[path].endswith("\n"):
                raise ValueError("No-final-newline files unsupported by this development protocol")
        source = updated[path]
        if source.count(search) != 1:
            raise ValueError("Search must match exactly once; no whitespace/fuzzy repair")
        updated[path] = source.replace(search, replacement, 1)
        if updated[path] and not updated[path].endswith("\n"):
            raise ValueError("Replacement must preserve final newline")
    chunks = []
    for path in sorted(updated):
        chunks.extend(difflib.unified_diff(originals[path].splitlines(keepends=True),
                      updated[path].splitlines(keepends=True), fromfile="a/" + path,
                      tofile="b/" + path))
    patch = "".join(chunks)
    if not patch:
        raise ValueError("Edits have no net effect")
    return patch, payload
