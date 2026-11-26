"""Prospectively registered visible original-line edits, with no repair/retry."""
import difflib
import json
import subprocess
from pathlib import PurePosixPath

try:
    from .swe_core import digest, git
except ImportError:
    from swe_core import digest, git


def registry_digest(provenance):
    return digest(json.dumps({k: v for k, v in provenance.items() if k != "registry_sha256"}, sort_keys=True))


def regular_base(repo, commit, path):
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or ".git" in pure.parts or str(pure) != path:
        raise ValueError("Invalid source path")
    entry = git(repo, "ls-tree", commit, "--", path).split()
    if not entry or entry[0] not in {"100644", "100755"}:
        raise ValueError("Source must be a regular tracked base file")
    # text=True would normalize CRLF before hashing/slicing. Decode bytes
    # strictly, retaining the original line endings and rejecting lossy input.
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{path}"], timeout=120).decode("utf-8")


def source_positions(repo, commit, displayed, protocol="source_positions_v1"):
    if protocol not in {"source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"}:
        raise ValueError("Unknown source-position protocol")
    spans, bases, parts = {}, {}, []
    for shown in displayed:
        path, start, end = (shown[k] for k in ("path", "start", "end"))
        if path not in bases:
            bases[path] = regular_base(repo, commit, path)
        lines = bases[path].splitlines(keepends=True)
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(lines):
            raise ValueError("Invalid displayed source bounds")
        text = "".join(lines[start:end])
        sid = "v_" + digest(f"{commit}:{path}:{start}:{end}:" + text)[:16]
        if sid in spans:
            raise ValueError("Duplicate visible span ID")
        spans[sid] = dict(path=path, first_line=start + 1, last_line=end, text=text, source_sha256=digest(text))
        parts.append(f"\n--- span_id={sid} | {path} | original base lines {start + 1}-{end}; replace exactly selected inclusive line range ---\n")
        parts.extend(f"{i + 1:06d} | " + lines[i] for i in range(start, end))
        if text and not text.endswith("\n"):
            parts.append("\n")
    if not spans:
        raise ValueError("No visible source spans")
    context = "".join(parts)
    if len(context) > 64000:
        raise ValueError("Numbered visible context exceeds registered budget")
    maximum = max(s["last_line"] for s in spans.values())
    replacement = ({"replace_lines": {"type": "array", "items": {"type": "string"}}}
                   if protocol in {"source_positions_v2", "source_positions_v3"} else {"replace": {"type": "string"}})
    replacement_field = next(iter(replacement))
    schema = {"type": "object", "properties": {"edits": {
        "type": "array", "minItems": 1, "maxItems": 20, "items": {
            "type": "object", "properties": {
                "span_id": {"type": "string", "enum": list(spans)},
                "start_line": {"type": "integer", "minimum": 1, "maximum": maximum},
                "end_line": {"type": "integer", "minimum": 1, "maximum": maximum},
                **replacement},
            "required": ["span_id", "start_line", "end_line", replacement_field], "additionalProperties": False}}},
        "required": ["edits"], "additionalProperties": False}
    provenance = dict(version="visible_" + protocol, base_commit=commit, spans=spans,
                      base_file_sha256={p: digest(text) for p, text in bases.items()},
                      max_rendered_context_chars=64000, rendered_context_sha256=digest(context),
                      schema_sha256=digest(json.dumps(schema, sort_keys=True)),
                      replacement_source="entirely model-generated; exact original inclusive line range only")
    if protocol in {"source_positions_v2", "source_positions_v3"}:
        provenance["replacement_source"] = "all line content model-generated; registered deterministic line serialization"
        provenance["line_serialization"] = "empty array deletes; each string excludes CR/LF; join with first selected original line terminator and append it; no content repair"
        if protocol == "source_positions_v3":
            provenance["line_serialization"] = "empty array deletes; each string has at most one optional terminal LF/CRLF and no interior CR/LF; retain provided terminal bytes, otherwise append first selected original line terminator; no content repair"
    if protocol == "source_positions_text_v1":
        provenance["replacement_source"] = "all multiline text model-generated; registered deterministic line-range boundary serialization"
        provenance["text_serialization"] = "empty string deletes; preserve all provided text bytes and interior newlines; if nonempty text lacks terminal LF, append the final selected original line LF/CRLF; already terminated text is unchanged; no content repair"
    provenance["registry_sha256"] = registry_digest(provenance)
    return context, schema, provenance


def resolve_position_edits(payload, provenance):
    """Validate every model selection against only the archived visible source."""
    if registry_digest(provenance) != provenance["registry_sha256"]:
        raise ValueError("Source-position registry hash mismatch")
    if not isinstance(payload, dict) or set(payload) != {"edits"}:
        raise ValueError("Expected only position edits")
    edits = payload["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= 20:
        raise ValueError("Expected 1 to 20 position edits")
    resolved = []
    text_protocol = provenance["version"] == "visible_source_positions_text_v1"
    optional_terminators = provenance["version"] == "visible_source_positions_v3"
    line_protocol = provenance["version"] in {"visible_source_positions_v2", "visible_source_positions_v3"}
    replacement_field = "replace_lines" if line_protocol else "replace"
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"span_id", "start_line", "end_line", replacement_field}:
            raise ValueError("Invalid position-edit fields")
        sid, first, last, value = (edit[k] for k in ("span_id", "start_line", "end_line", replacement_field))
        if not isinstance(sid, str) or type(first) is not int or type(last) is not int:
            raise ValueError("Invalid position-edit types")
        if line_protocol:
            if not isinstance(value, list) or not all(isinstance(line, str) for line in value):
                raise ValueError("replace_lines must be an array of strings")
            bodies = [(line[:-2] if line.endswith("\r\n") else line[:-1] if line.endswith("\n") else line)
                      for line in value] if optional_terminators else value
            if any("\n" in line or "\r" in line for line in bodies):
                raise ValueError("replace_lines items must exclude CR and LF")
        elif not isinstance(value, str):
            raise ValueError("Invalid position-edit types")
        item = provenance["spans"].get(sid)
        if item is None:
            raise ValueError("Unknown registered span_id")
        if not item["first_line"] <= first <= last <= item["last_line"]:
            raise ValueError("Range must stay inside one visible original span")
        text = item["text"]
        if digest(text) != item["source_sha256"]:
            raise ValueError("Visible span hash mismatch")
        expected_id = "v_" + digest(f"{provenance['base_commit']}:{item['path']}:{item['first_line'] - 1}:{item['last_line']}:" + text)[:16]
        if sid != expected_id:
            raise ValueError("Visible span ID provenance mismatch")
        old = "".join(text.splitlines(keepends=True)[first - item["first_line"]:last - item["first_line"] + 1])
        if line_protocol:
            first_original_line = old.splitlines(keepends=True)[0]
            terminator = "\r\n" if first_original_line.endswith("\r\n") else "\n"
            if optional_terminators:
                replacement = "".join(line if line.endswith("\n") else line + terminator for line in value)
            else:
                replacement = terminator.join(value) + terminator if value else ""
        else:
            replacement = value
        boundary = ""
        if text_protocol and replacement and not replacement.endswith("\n"):
            if not old.endswith("\n"):
                raise ValueError("Selected original range lacks a complete LF line boundary")
            boundary = "\r\n" if old.endswith("\r\n") else "\n"
            replacement += boundary
        if old == replacement:
            raise ValueError("Position edit has no effect")
        if replacement and not replacement.endswith("\n"):
            raise ValueError("Replacement must preserve complete final lines")
        if any(item["path"] == e["path"] and max(first, e["start_line"]) <= min(last, e["end_line"]) for e in resolved):
            raise ValueError("Overlapping or duplicate original line ranges")
        entry = dict(path=item["path"], start_line=first, end_line=last,
                     search=old, source_sha256=digest(old), replace=replacement)
        if text_protocol:
            entry.update(raw_replace=value, raw_replace_sha256=digest(value),
                         serialized_replace_sha256=digest(replacement),
                         boundary_terminator_added=bool(boundary), boundary_terminator=boundary)
        resolved.append(entry)
    return {"edits": resolved}


def compile_positions(text, repo, commit, provenance):
    if commit != provenance["base_commit"]:
        raise ValueError("Position registry belongs to a different base")
    resolved = resolve_position_edits(json.loads(text), provenance)
    originals, updates, changes = {}, {}, {}
    for edit in resolved["edits"]:
        path = edit["path"]
        if path not in originals:
            originals[path] = regular_base(repo, commit, path)
            if digest(originals[path]) != provenance["base_file_sha256"][path]:
                raise ValueError("Base file hash mismatch")
            if originals[path] and not originals[path].endswith("\n"):
                raise ValueError("No-final-newline base unsupported")
        first, last = edit["start_line"], edit["end_line"]
        if "".join(originals[path].splitlines(keepends=True)[first - 1:last]) != edit["search"]:
            raise ValueError("Visible original range differs from base")
        changes.setdefault(path, []).append(edit)
    chunks = []
    for path, edits in sorted(changes.items()):
        lines = originals[path].splitlines(keepends=True)
        for edit in sorted(edits, key=lambda e: e["start_line"], reverse=True):
            lines[edit["start_line"] - 1:edit["end_line"]] = [edit["replace"]] if edit["replace"] else []
        updates[path] = "".join(lines)
        chunks.extend(difflib.unified_diff(originals[path].splitlines(keepends=True), updates[path].splitlines(keepends=True),
                                          fromfile="a/" + path, tofile="b/" + path))
    diff = "".join(chunks)
    if not diff:
        raise ValueError("Position edits have no net effect")
    return diff, resolved
