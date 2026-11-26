"""Validate collected observations and replay compatibility before using them."""
import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

from pbkv_sim.eta import DurationModel
from pbkv_sim.predictors import MarkovPredictor, RoleTable
from pbkv_sim.replay import ReplayConfig, replay
from pbkv_sim.trace import load_workflows
try:
    from .swe_core import digest
except ImportError:
    from swe_core import digest


def validate(data):
    manifest = json.loads((data / "run_manifest.json").read_text())
    roles = RoleTable.from_metadata(data / "dataset.json")
    rows = [json.loads(line) for line in (data / "swe.jsonl").read_text().splitlines() if line]
    histories = {}
    calls = [r for r in rows if r["event"] == "call"]
    block_calls = 0
    position_calls = 0
    position_failures = []
    for row in calls:
        previous = histories.setdefault(row["workflow_id"], [])
        if previous != row["my_chain_history"] or row["call_index"] != len(previous) + 1:
            raise ValueError("Role history/call index does not match the actual observed calls")
        if row["role"] not in roles.index or row["role"] == "END":
            raise ValueError("Unknown observed role")
        previous.append(row["role"])
        prefix = data / "calls" / f"{row['workflow_id']}_{row['call_index']:02d}"
        request = json.loads(prefix.with_suffix(".request.json").read_text())
        if manifest.get("patch_protocol") in {"source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"} and row["role"] in {"Coder", "Committer"}:
            try:
                from .swe_source_positions import resolve_position_edits, compile_positions
            except ImportError:
                from swe_source_positions import resolve_position_edits, compile_positions
            archived = json.loads((data / f"{row['workflow_id']}_source_schema.json").read_text())
            if json.loads(request["sampling_params"]["json_schema"]) != archived["schema"]:
                raise ValueError("Actual source-position schema differs from workflow archive")
            if digest(json.dumps(archived["schema"], sort_keys=True)) != archived["provenance"]["schema_sha256"]:
                raise ValueError("Source-position schema hash mismatch")
            context = (data / f"{row['workflow_id']}_context.txt").read_bytes().decode("utf-8")
            if digest(context) != archived["provenance"]["rendered_context_sha256"]:
                raise ValueError("Source-position displayed context hash mismatch")
            semantic_error = None
            try:
                resolve_position_edits(json.loads(row["text"]), archived["provenance"])
            except ValueError as error:
                # v1 retains its original fail-fast contract. Later versions record every
                # rejection while still auditing hidden states and prefixes.
                if manifest.get("patch_protocol") == "source_positions_v1":
                    raise
                semantic_error = str(error)
                position_failures.append(dict(workflow_id=row["workflow_id"], role=row["role"],
                                              call_index=row["call_index"], error=semantic_error))
            if row["role"] == "Committer":
                artifact = data / f"{row['workflow_id']}_r{row['round']}_patch"
                if artifact.with_suffix(".diff").exists():
                    if semantic_error:
                        raise ValueError("Rejected position edit unexpectedly has a submitted diff")
                    diff, resolved = compile_positions(row["text"], data / "workspaces" / row["workflow_id"],
                                                       archived["provenance"]["base_commit"], archived["provenance"])
                    if diff != artifact.with_suffix(".diff").read_bytes().decode("utf-8") or resolved != json.loads(artifact.with_suffix(".resolved_edits.json").read_text()):
                        raise ValueError("Recorded position diff or resolved ranges differ from the actual Committer output")
            position_calls += semantic_error is None
        if manifest.get("patch_protocol") == "source_blocks_v1" and row["role"] in {"Coder", "Committer"}:
            try:
                from .swe_source_blocks import resolve_source_blocks
            except ImportError:
                from swe_source_blocks import resolve_source_blocks
            archived = json.loads((data / f"{row['workflow_id']}_source_schema.json").read_text())
            if json.loads(request["sampling_params"]["json_schema"]) != archived["schema"]:
                raise ValueError("Actual source-block schema differs from workflow archive")
            if digest(json.dumps(archived["schema"], sort_keys=True)) != archived["provenance"]["schema_sha256"]:
                raise ValueError("Source-block schema hash mismatch")
            context = (data / f"{row['workflow_id']}_context.txt").read_text()
            if digest(context) != archived["provenance"]["rendered_context_sha256"]:
                raise ValueError("Source-block displayed context hash mismatch")
            resolve_source_blocks(json.loads(row["text"]), archived["provenance"])
            block_calls += 1
        ids = request["input_ids"]
        if len(ids) != row["prompt_tokens"] or row["segments"][-1][1] != len(ids):
            raise ValueError("Actual request token count differs from recorded prompt")
        for hashed, n in row["segments"]:
            if digest(struct.pack(f"<{n}I", *ids[:n])) != hashed:
                raise ValueError("Segment is not a hash of the actual token prefix")
        x = np.load(data / "hidden" / Path(row["hidden_file"]).name, allow_pickle=False)
        if x.shape != (manifest["hidden_dimension"],) or not np.isfinite(x).all() or not np.linalg.norm(x.astype(np.float32)):
            raise ValueError("Invalid recorded hidden state")
        if not row["queue_start"] <= row["prefill_end"] <= row["decode_end"]:
            raise ValueError("Invalid LLM observation timestamps")
    workflows = list(load_workflows(data / "swe.jsonl").values())
    if not workflows:
        raise ValueError("No completed non-ERROR workflows")
    markov = MarkovPredictor(roles=roles).fit([w.roles() for w in workflows])
    duration = DurationModel.fit(workflows, roles)
    accesses = [replay(workflows, ReplayConfig(policy, "markov", 4, None), markov, duration)["access_log"]
                for policy in ("lru", "lae", "he_step")]
    if not all(access == accesses[0] for access in accesses):
        raise ValueError("Replay call times/paths depend on eviction policy")
    evaluated, failed = [], []
    for path in sorted((data / "evaluations").glob("*_r*/result.json")):
        if not re.search(r"_r[1-3]$", path.parent.name):
            continue
        result = json.loads(path.read_text())
        if result.get("tests_executed") and result["status"] == "EVALUATED":
            evaluated.append(result["run_id"])
        else:
            failed.append(str(path))
    if len(set(evaluated)) != len(evaluated):
        raise ValueError("Evaluation run IDs were reused")
    errors = [r["workflow_id"] for r in rows if r["event"] == "workflow_end" and r["verdict"] == "ERROR"]
    report = {"calls": len(calls), "completed_workflows": len(workflows),
              "unique_instances": len({w.instance_id for w in workflows}), "error_workflows": errors,
              "verified_real_evaluations": len(evaluated), "failed_evaluations": failed,
              "observed_tool_events": sum(r["event"] == "tool" for r in rows),
              "history_and_hidden_checks_passed": True, "exact_token_prefix_checks_passed": True,
              "replay_access_policy_independence_passed": True,
              "smoke_acceptance_passed": bool(evaluated and not failed and not errors and not position_failures),
              "pilot_only": True, "training_started": False}
    if manifest.get("patch_protocol") == "source_blocks_v1":
        report["source_block_editing_calls_verified"] = block_calls
        report["source_block_schema_and_ids_passed"] = True
    if manifest.get("patch_protocol") in {"source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"}:
        report["source_position_editing_calls_verified"] = position_calls
        report["source_position_schema_and_ranges_passed"] = not position_failures
        if manifest.get("patch_protocol") in {"source_positions_v2", "source_positions_v3", "source_positions_text_v1"}:
            report["source_position_rejected_editing_calls"] = position_failures
    (data / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    result = validate(args.data)
    print(json.dumps(result))
    if not result["smoke_acceptance_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
