"""Explicit, audited SWE corpus import. No directory scanning or collection side effects."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .predictors import RoleTable
from .trace import load_workflows

ARCHITECTURE = Path(__file__).with_name("swe_architecture.json")
PARTITIONS = {"train", "val", "test"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {relative}")
    return path


def hidden_file(recorded, root):
    if not recorded:
        raise ValueError("Missing prefill hidden state")
    path = Path(recorded)
    if "hidden" in path.parts:
        path = Path(*path.parts[path.parts.index("hidden"):])
    return inside(root, path)


def read_split(path):
    content = read_json(path)
    if "instances" in content:
        rows = content["instances"]
        split = {r["instance_id"]: r["partition"] for r in rows}
        if len(split) != len(rows):
            raise ValueError("Duplicate instance in split")
        if set(split) & set(content.get("excluded_development_instances", [])):
            raise ValueError("Development instances leaked into formal split")
    else:
        split = content
    if not split or any(not isinstance(k, str) or v not in PARTITIONS for k, v in split.items()):
        raise ValueError("Expected instance_id -> train/val/test split")
    return split


def convert_split(source, out):
    split = read_split(source)
    write_new(out, split)
    write_new(str(out) + ".provenance.json", {
        "source": str(Path(source).resolve()), "source_sha256": sha(source),
        "derived_sha256": sha(out), "instances": dict(Counter(split.values())),
        "rule": "Preserve registered instance partitions; never resplit calls or seeds"})
    return split


def build_corpus(root, split_path, runs_path, scope_path=None, development=False, timeout_records=()):
    """Consume completed strict audits; additionally check portable training/replay inputs.

    The original audit proves request/schema/prefix/harness semantics. This import
    checks its artifact hashes, coverage, identity and hidden vectors, not capability.
    A partial formal corpus is development-only until exact registered coverage exists.
    """
    root = Path(root).resolve()
    split = read_split(split_path)
    registered = read_json(split_path)
    architecture = read_json(ARCHITECTURE)
    table = RoleTable(tuple(architecture["roles"]))
    seeds = registered.get("seeds", [0, 1, 2]) if "instances" in registered else [0, 1, 2]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate registered seeds")
    expected = {f"{iid}_s{seed}" for iid in split for seed in seeds}
    excluded = set()
    if scope_path:
        scope = read_json(scope_path)
        # consumed_or_quarantined includes 240 accepted workflows too; only
        # explicitly quarantined chunks may be removed from final expected scope.
        chunks = scope.get("quarantined_whole_failed_chunks", [])
        rows = registered["instances"]
        for chunk in chunks:
            part, offset = chunk.rsplit("-", 1)
            selected = [r for r in rows if r["partition"] == part][int(offset):int(offset) + 20]
            if len(selected) != 20:
                raise ValueError("Invalid quarantined chunk")
            excluded.update(f"{r['instance_id']}_s{s}" for r in selected for s in seeds)
        if len(excluded) != scope["quarantined_workflows"]:
            raise ValueError("Quarantine scope mismatch")
        expected -= excluded
        if len(expected) != scope["effective_final_target"]:
            raise ValueError("Effective final scope mismatch")
    elif not development:
        raise ValueError("Formal corpus requires the registered scope disposition")
    timeout_hashes = {}
    for path in timeout_records:
        record = read_json(path)
        chunk = record['original_chunk']
        pairs = {f'{iid}_s{s}' for iid in record['quarantined_instances'] for s in seeds}
        if (record.get('classification') != 'verified_official_harness_timeout'
                or record.get('model_resampled') is not False or record.get('timeout_changed') is not False
                or record.get('observations_rewritten') is not False
                or set(record['quarantined_pairs']) != pairs
                or not pairs <= expected
                or any(split.get(iid) != chunk['partition'] for iid in record['quarantined_instances'])):
            raise ValueError('Invalid timeout quarantine or changed instance partition')
        excluded.update(pairs)
        expected -= pairs
        timeout_hashes[str(path)] = sha(path)
    entries, seen, counts = [], set(), Counter()
    source_signature = None
    for item in read_json(runs_path)["runs"]:
        data = inside(root, item["data"])
        audit_path = inside(root, item["audit"])
        audit = read_json(audit_path)
        if (audit.get("observation_integrity_passed") is not True
                or audit.get("hidden_history_prefix_replay_and_actual_schema_checks_passed") is not True
                or audit.get("failed_evaluations") or audit.get("error_workflows")):
            raise ValueError(f"Audit has not passed: {audit_path}")
        manifest = read_json(data / "run_manifest.json")
        if not development:
            if manifest.get("stage") != "formal" or manifest.get("formal_split_sha256") != sha(split_path):
                raise ValueError("Run is not from this formal split")
            if audit.get("formal_corpus") is not True and audit.get("formal_split_sha256") != sha(split_path):
                raise ValueError("Missing formal audit provenance")
            if (manifest.get("patch_protocol") != "source_positions_feedback_v3"
                    or audit.get("protocol") != manifest["patch_protocol"]
                    or manifest.get("hidden_dimension") != architecture["emb_dim"]):
                raise ValueError("Formal protocol or hidden dimension differs")
            signature = manifest.get("source_sha256")
            if not signature or (source_signature is not None and signature != source_signature):
                raise ValueError("Formal runs use different collector sources")
            source_signature = signature
        artifacts = audit["artifact_sha256"]
        required = {"swe.jsonl", "dataset.json", "run_manifest.json", "validation.json", "status.json"}
        if not required.issubset(artifacts):
            raise ValueError("Audit missing required artifacts")
        for relative, digest in artifacts.items():
            if sha(inside(data, relative)) != digest:
                raise ValueError(f"Audited artifact changed: {data / relative}")
        if read_json(data / "dataset.json")["roles"] != list(table.roles):
            raise ValueError("Role order differs from fixed SWE architecture")
        if manifest.get("source_sha256") != audit.get("source_sha256"):
            raise ValueError("Audit and run source provenance differ")
        rows = [json.loads(s) for s in (data / "swe.jsonl").read_text().splitlines() if s.strip()]
        ends = [r for r in rows if r.get("event") == "workflow_end"]
        call_rows = [r for r in rows if r.get("event") == "call"]
        if len({r['workflow_id'] for r in ends}) != len(ends):
            raise ValueError("Duplicate workflow_end")
        actual = {r["workflow_id"] for r in ends}
        run_expected = {f"{i}_s{s}" for i in manifest["instances"]
                        for s in map(int, manifest["seeds"].split(","))}
        if (actual != run_expected or actual != set(audit["expected_workflow_ids"])
                or len(ends) != audit["workflows"] or len(call_rows) != audit["calls"]
                or {r["workflow_id"] for r in call_rows} != actual):
            raise ValueError("Incomplete or unexpected run coverage")
        if actual & seen or not actual <= expected:
            raise ValueError("Duplicate, quarantined, or unregistered workflow")
        workflows = load_workflows(data / "swe.jsonl")
        if set(workflows) != actual:
            raise ValueError("Loader dropped an ERROR or malformed workflow")
        hidden_hashes, row_groups = {}, defaultdict(list)
        for row in call_rows:
            row_groups[row["workflow_id"]].append(row)
        partitions = set()
        for wid, wf in workflows.items():
            if not wf.instance_id or wid not in {f"{wf.instance_id}_s{s}" for s in seeds}:
                raise ValueError("Workflow/instance/seed mismatch")
            part = split[wf.instance_id]
            partitions.add(part)
            records = sorted(row_groups[wid], key=lambda r: r["call_index"])
            if [r["call_index"] for r in records] != list(range(1, len(records) + 1)):
                raise ValueError("Duplicate or noncontiguous call_index")
            history = []
            for row in records:
                if row.get("instance_id") != wf.instance_id or row["role"] not in table.roles[:-1]:
                    raise ValueError("Call identity or role mismatch")
                if row.get("my_chain_history") != history:
                    raise ValueError("History contains missing/current/future calls")
                history.append(row["role"])
                if not (row["queue_start"] <= row["prefill_end"] <= row["decode_end"]):
                    raise ValueError("Invalid prefill timestamp")
                hp = hidden_file(row.get("hidden_file"), data)
                rel = str(hp.relative_to(data))
                if rel not in artifacts:
                    raise ValueError("Hidden vector absent from strict audit")
                x = np.load(hp, allow_pickle=False)
                if (x.shape != (architecture["emb_dim"],) or not np.isfinite(x).all()
                        or not np.linalg.norm(x.astype(np.float32))):
                    raise ValueError("Invalid, zero, or wrong-dimension hidden vector")
                hidden_hashes[rel] = artifacts[rel]
            counts[part] += 1
        if len(partitions) != 1:
            raise ValueError("A collection batch crosses registered partitions")
        part = partitions.pop()
        if not development and manifest.get("formal_partition") != part:
            raise ValueError("Manifest partition differs from instance split")
        seen.update(actual)
        entries.append({"trace": str((data / "swe.jsonl").relative_to(root)),
                        "partition": part, "workflows": sorted(actual),
                        "trace_sha256": artifacts["swe.jsonl"], "hidden_sha256": hidden_hashes,
                        "audit": str(audit_path.relative_to(root)), "audit_sha256": sha(audit_path),
                        "artifacts": {k: artifacts[k] for k in required}})
    if not entries:
        raise ValueError("Empty accepted corpus")
    complete = seen == expected
    return {"schema": 1, "root": str(root), "split": split,
            "registered_split_sha256": sha(split_path),
            "architecture": architecture, "architecture_sha256": sha(ARCHITECTURE),
            "scope_sha256": sha(scope_path) if scope_path else None,
            "timeout_recovery_sha256": timeout_hashes,
            "runs_list_sha256": sha(runs_path), "entries": entries,
            "counts": dict(counts), "expected_workflows": len(expected),
            "missing_workflows": sorted(expected - seen), "quarantined_workflows": sorted(excluded),
            "complete": complete, "development_only": development or not complete,
            "checklist": {"strict_audits_passed": True, "artifact_hashes_verified": True,
                          "instance_seed_split_verified": True, "hidden_history_verified": True,
                          "model_failures_preserved": True, "exact_scope_complete": complete}}


def load_corpus(path, partitions, require_final=False):
    """Verify selected partitions only; training never opens test trace or hidden."""
    corpus = read_json(path)
    if corpus.get("schema") != 1 or corpus["architecture_sha256"] != sha(ARCHITECTURE):
        raise ValueError("Unsupported corpus or changed architecture")
    if corpus["architecture"] != read_json(ARCHITECTURE):
        raise ValueError("Corpus architecture content differs")
    if require_final and (corpus["development_only"] or not corpus["complete"]):
        raise ValueError("Final test requires a complete formal corpus")
    split, root = corpus["split"], Path(corpus["root"])
    by_part = {p: [] for p in partitions}
    roots, traces, seen = {}, {}, set()
    for entry in corpus["entries"]:
        if entry["partition"] not in by_part:
            continue
        trace = inside(root, entry["trace"])
        data = trace.parent
        if sha(trace) != entry["trace_sha256"] or sha(inside(root, entry["audit"])) != entry["audit_sha256"]:
            raise ValueError("Corpus trace or audit changed")
        for rel, digest in (entry["artifacts"] | entry["hidden_sha256"]).items():
            if sha(inside(data, rel)) != digest:
                raise ValueError(f"Corpus artifact changed: {rel}")
        wfs = load_workflows(trace)
        if set(wfs) != set(entry["workflows"]):
            raise ValueError("Corpus workflow coverage changed")
        for wid, wf in wfs.items():
            if wid in seen or split.get(wf.instance_id) != entry["partition"]:
                raise ValueError("Duplicate workflow or partition leakage")
            seen.add(wid)
            by_part[entry["partition"]].append(wf)
            roots[(wf.setting, wid)] = data
        traces[entry["trace"]] = entry["trace_sha256"]
    if any(not wfs for wfs in by_part.values()):
        raise ValueError("Requested corpus partition is empty")
    return corpus, by_part, traces, roots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("split")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("build")
    for name in ("root", "split", "runs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--scope", type=Path)
    p.add_argument("--development", action="store_true")
    p.add_argument("--timeout-record", action="append", type=Path, default=[],
                   help="Additive verified-timeout quarantine record; repeat for every incident")
    args = ap.parse_args()
    if args.command == "split":
        convert_split(args.split, args.out)
    else:
        corpus = build_corpus(args.root, args.split, args.runs, args.scope, args.development, args.timeout_record)
        write_new(args.out, corpus)
        print(json.dumps({k: corpus[k] for k in ("counts", "complete", "development_only", "checklist")}))


if __name__ == "__main__":
    main()
