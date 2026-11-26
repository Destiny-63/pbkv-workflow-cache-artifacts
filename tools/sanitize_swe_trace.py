#!/usr/bin/env python3
"""Create a metadata-only SWE trace from SHA-256-verified source trace files."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import secrets
from collections import Counter
from pathlib import Path

TIME_FIELDS = (
    "queue_start",
    "container_start",
    "exec_start",
    "exec_end",
    "prefill_end",
    "decode_end",
)


def number(value):
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Trace contains a non-finite timestamp")
    return result


def opaque(key: bytes, namespace: str, value: str) -> str:
    digest = hmac.new(key, f"{namespace}:{value}".encode(), hashlib.sha256).hexdigest()
    return digest[:24]


def normalize_time(value, origin):
    timestamp = number(value)
    return None if timestamp is None else round(timestamp - origin, 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.corpus_manifest.read_text())
    split = manifest["split"]
    entries = {entry["trace_sha256"]: entry for entry in manifest["entries"]}
    key = secrets.token_bytes(32)
    workflow_ids = {}
    instance_ids = {}
    prefix_ids = {}
    output_rows = []
    released_splits = {}
    counts = Counter()
    seen_workflows = set()

    for source in args.source:
        raw_bytes = source.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        entry = entries.get(digest)
        if entry is None:
            raise ValueError("A source trace does not match an accepted corpus hash")
        if entry["partition"] not in {"train", "val", "test"}:
            raise ValueError("Unknown source partition")
        expected = set(entry["workflows"])
        rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
        raw_workflows = {row["workflow_id"] for row in rows}
        if raw_workflows != expected:
            raise ValueError("Source trace workflow coverage differs from its manifest")
        if raw_workflows & seen_workflows:
            raise ValueError("Accepted source traces contain duplicate workflows")

        origins = {}
        for row in rows:
            wid = row["workflow_id"]
            times = [number(row.get(field)) for field in TIME_FIELDS]
            available = [value for value in times if value is not None]
            if available:
                origins[wid] = min(origins.get(wid, available[0]), min(available))
        if set(origins) != expected:
            raise ValueError("Every accepted workflow must contain a timestamp")

        for raw in rows:
            wid = raw["workflow_id"]
            instance = raw.get("instance_id")
            if not instance or instance not in split:
                raise ValueError("Workflow instance is absent from the registered split")
            partition = split[instance]
            if partition != entry["partition"]:
                raise ValueError("Trace partition differs from the registered split")
            seed_match = re.search(r"_s(\d+)$", wid)
            if seed_match is None:
                raise ValueError("Workflow seed is missing")

            public_wid = workflow_ids.setdefault(wid, opaque(key, "workflow", wid))
            public_instance = instance_ids.setdefault(instance, opaque(key, "instance", instance))
            released_splits[f"inst-{public_instance}"] = partition
            event = raw["event"]
            public = {
                "event": event,
                "workflow_id": f"wf-{public_wid}",
                "instance_id": f"inst-{public_instance}",
                "partition": partition,
                "seed": int(seed_match.group(1)),
                "setting": "swe-trace",
            }
            origin = origins[wid]
            if event == "call":
                segments = []
                for source_id, token_count in raw.get("segments", []):
                    public_id = prefix_ids.setdefault(
                        str(source_id), opaque(key, "prefix", str(source_id))
                    )
                    segments.append([f"prefix-{public_id}", int(token_count)])
                public.update(
                    role=raw["role"],
                    round=int(raw.get("round") or 1),
                    call_index=int(raw["call_index"]),
                    thinking=str(raw.get("thinking", "false")).lower() == "true",
                    queue_start=normalize_time(raw.get("queue_start"), origin),
                    prefill_end=normalize_time(raw.get("prefill_end"), origin),
                    decode_end=normalize_time(raw.get("decode_end"), origin),
                    prompt_tokens=int(raw.get("prompt_tokens") or 0),
                    output_tokens=int(raw.get("output_tokens") or 0),
                    cached_tokens=int(raw.get("cached_tokens") or 0),
                    segments=segments,
                    segment_tokens=int(raw.get("segment_tokens") or 0),
                )
            elif event == "tool":
                public.update(
                    tool_type="tool",
                    after_call_index=int(raw["after_call_index"]),
                    round=int(raw.get("round") or 1),
                    queue_start=normalize_time(raw.get("queue_start"), origin),
                    container_start=normalize_time(raw.get("container_start"), origin),
                    exec_start=normalize_time(raw.get("exec_start"), origin),
                    exec_end=normalize_time(raw.get("exec_end"), origin),
                )
            elif event == "workflow_end":
                public["decode_end"] = normalize_time(raw.get("decode_end"), origin)
            else:
                raise ValueError(f"Unsupported trace event: {event}")

            output_rows.append(public)
            counts[event] += 1
            seen_workflows.add(wid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output_rows))
    args.split_out.parent.mkdir(parents=True, exist_ok=True)
    args.split_out.write_text(json.dumps(released_splits, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"workflows": len(workflow_ids), "instances": len(instance_ids), "events": dict(counts)}))


if __name__ == "__main__":
    main()
