"""Constrain edit sources to literal, visible base-code choices, not fixes."""
import ast
import json
from collections import defaultdict

try:
    from .swe_core import digest, git
except ImportError:
    from swe_core import digest, git


def source_choice_schema(repo, commit, displayed):
    """Return schema and provenance. Replacement text is never supplied here.

    `displayed` comes directly from the context builder, not parsed model output.
    Every search is a unique substring wholly present in a displayed span.
    """
    by_path = defaultdict(list)
    for span in displayed:
        by_path[span["path"]].append((span["start"], span["end"]))
    branches, provenance = [], []
    for path, spans in by_path.items():
        source = git(repo, "show", f"{commit}:{path}")
        lines = source.splitlines(keepends=True)
        visible = set()
        for start, end in spans:
            if not 0 <= start < end <= len(lines):
                raise ValueError("Visible source span is outside base file")
            visible.update(range(start, end))
        candidates = []
        try:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                    candidates.append((0, start, node.end_lineno))
        except (SyntaxError, ValueError):
            pass
        for i in sorted(visible):
            candidates.extend((1, i, i + width) for width in (1, 2, 3))
        options, seen = [], set()
        for kind, start, end in sorted(candidates):
            if not all(i in visible for i in range(start, end)):
                continue
            text = "".join(lines[start:end])
            limit = 2048 if kind == 0 else 512
            if not 8 <= len(text) <= limit or text in seen or source.count(text) != 1:
                continue
            if not text.strip() or text.strip().startswith("#"):
                continue
            options.append(text); seen.add(text)
            if len(options) == 96:
                break
        if not options:
            continue
        branches.append({"type": "object", "properties": {
            "path": {"const": path}, "search": {"type": "string", "enum": options},
            "replace": {"type": "string"}},
            "required": ["path", "search", "replace"], "additionalProperties": False})
        provenance.append({"path": path, "base_source_sha256": digest(source),
                           "visible_spans": [{"start": a, "end": b,
                               "sha256": digest("".join(lines[a:b]))} for a, b in spans],
                           "unique_source_options": len(options)})
    if not branches:
        raise ValueError("No unique visible source edits are available")
    schema = {"type": "object", "properties": {"edits": {
        "type": "array", "minItems": 1, "maxItems": 20,
        "items": {"oneOf": branches}}}, "required": ["edits"], "additionalProperties": False}
    return schema, {"version": "visible_base_sources_v1", "base_commit": commit,
                    "schema_sha256": digest(json.dumps(schema, sort_keys=True)), "sources": provenance,
                    "replacement_source": "unconstrained model-generated string; no supplied fix"}


def check_source_choices(payload, schema):
    """Verify recorded decoder constraints again before ordinary exact compilation."""
    allowed = {b["properties"]["path"]["const"]: set(b["properties"]["search"]["enum"])
               for b in schema["properties"]["edits"]["items"]["oneOf"]}
    if not isinstance(payload, dict) or set(payload) != {"edits"} or not isinstance(payload["edits"], list):
        raise ValueError("Expected a source-choice edits object")
    if not 1 <= len(payload["edits"]) <= 20:
        raise ValueError("Expected 1 to 20 source-choice edits")
    for edit in payload["edits"]:
        if (not isinstance(edit, dict) or set(edit) != {"path", "search", "replace"}
                or not all(isinstance(x, str) for x in edit.values())):
            raise ValueError("Expected string path, search, replace")
        if edit["search"] not in allowed.get(edit["path"], set()):
            raise ValueError("Edit source is not a registered visible base-code choice")
