"""Prospective source_blocks_v1: select immutable visible blocks by ID.

The controller supplies source locations, never replacement content. This is a
versioned output interface, not a repair of earlier model-authored searches.
"""
import json
from collections import defaultdict

try:
    from .swe_core import digest, git
except ImportError:
    from swe_core import digest, git

MAX_BLOCK_CHARS = 2048
MAX_CONTEXT_CHARS = 64000


def source_blocks(repo, commit, displayed):
    by_path = defaultdict(list)
    for span in displayed:
        by_path[span["path"]].append((span["start"], span["end"]))
    choices, parts, sources = {}, [], []
    for path, spans in by_path.items():
        source = git(repo, "show", f"{commit}:{path}")
        lines = source.splitlines(keepends=True)
        visible, covered = set(), set()
        evidence = []
        for start, end in spans:
            if not 0 <= start < end <= len(lines):
                raise ValueError("Visible source span is outside base file")
            visible.update(range(start, end))
            evidence.append(dict(start=start, end=end, sha256=digest("".join(lines[start:end]))))
            cursor = start
            while cursor < end:
                stop = min(cursor + 8, end)
                while stop > cursor + 1 and len("".join(lines[cursor:stop])) > MAX_BLOCK_CHARS:
                    stop -= 1
                first, last = cursor, stop

                def eligible(a, b):
                    text = "".join(lines[a:b])
                    return 8 <= len(text) <= MAX_BLOCK_CHARS and text.strip() and source.count(text) == 1

                # Only expand inside this already displayed span. Never read an
                # unseen line into an option, or supply a suggested correction.
                while not eligible(first, last) and last < min(end, cursor + 16):
                    if len("".join(lines[first:last + 1])) > MAX_BLOCK_CHARS:
                        break
                    last += 1
                while not eligible(first, last) and first > max(start, cursor - 8):
                    if len("".join(lines[first - 1:last])) > MAX_BLOCK_CHARS:
                        break
                    first -= 1
                if eligible(first, last):
                    text = "".join(lines[first:last])
                    sid = "s_" + digest(f"{commit}:{path}:{first}:{last}:" + text)[:16]
                    item = dict(path=path, start=first, end=last, search=text, source_sha256=digest(text))
                    if sid in choices and choices[sid] != item:
                        raise ValueError("Source ID collision")
                    choices[sid] = item
                    covered.update(range(first, last))
                    parts.append(f"\n--- source_id={sid} | {path} | base lines {first + 1}-{last} ---\n" + text)
                    cursor = last
                else:
                    text = "".join(lines[cursor:stop])
                    parts.append(f"\n--- NOT_SELECTABLE | {path} | base lines {cursor + 1}-{stop}; no unique bounded block ---\n" + text)
                    cursor = stop
        nonblank = {i for i in visible if lines[i].strip()}
        sources.append(dict(path=path, base_source_sha256=digest(source), visible_spans=evidence,
                            visible_nonblank_lines=len(nonblank), covered_nonblank_lines=len(nonblank & covered),
                            uncovered_nonblank_lines=[i + 1 for i in sorted(nonblank - covered)]))
    if not choices:
        raise ValueError("No unique visible source blocks are available")
    context = "".join(parts)
    if len(context) > MAX_CONTEXT_CHARS:
        raise ValueError("Source-block context exceeds registered character budget")
    schema = {"type": "object", "properties": {"edits": {
        "type": "array", "minItems": 1, "maxItems": 20, "items": {
            "type": "object", "properties": {
                "source_id": {"type": "string", "enum": list(choices)},
                "replace": {"type": "string"}},
            "required": ["source_id", "replace"], "additionalProperties": False}}},
        "required": ["edits"], "additionalProperties": False}
    provenance = dict(version="visible_source_blocks_v1", base_commit=commit, choices=choices, sources=sources,
                      block_lines=8, expansion_max_right_lines=16, expansion_max_left_lines=8,
                      max_block_chars=MAX_BLOCK_CHARS, max_rendered_context_chars=MAX_CONTEXT_CHARS,
                      rendered_context_sha256=digest(context), schema_sha256=digest(json.dumps(schema, sort_keys=True)),
                      replacement_source="entirely model-generated; exact ID lookup only")
    return context, schema, provenance


def resolve_source_blocks(payload, provenance):
    """Strictly decode the registered interface; never infer or correct an ID."""
    if not isinstance(payload, dict) or set(payload) != {"edits"}:
        raise ValueError("Expected only source-block edits")
    edits = payload["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= 20:
        raise ValueError("Expected 1 to 20 source-block edits")
    resolved, selected = [], []
    for edit in edits:
        if (not isinstance(edit, dict) or set(edit) != {"source_id", "replace"}
                or not all(isinstance(v, str) for v in edit.values())):
            raise ValueError("Expected string source_id and replace")
        item = provenance["choices"].get(edit["source_id"])
        if item is None:
            raise ValueError("Unknown registered source_id; no implicit correction")
        if digest(item["search"]) != item["source_sha256"]:
            raise ValueError("Registered source block hash mismatch")
        if any(item["path"] == old["path"] and max(item["start"], old["start"]) < min(item["end"], old["end"]) for old in selected):
            raise ValueError("Selected source blocks overlap or repeat")
        selected.append(item)
        resolved.append(dict(path=item["path"], search=item["search"], replace=edit["replace"]))
    return {"edits": resolved}
