"""Collect development SWE workflows using native SGLang and official tests.

Each output directory is immutable as a run. Pilot/model-comparison instances
come from the pre-registered manifest; no final holdout is evaluated here.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import numpy as np

try:
    from .swe_edits import compile_edits, EDIT_SCHEMA
    from .swe_context import repository_context_v2
    from .swe_source_schema import source_choice_schema, check_source_choices
    from .swe_source_blocks import source_blocks, resolve_source_blocks
    from .swe_source_positions import source_positions, compile_positions
    from .swe_core import ROLES, MANUAL, INSTRUCTIONS, MAX_NEW, Transcript, digest, git, repository_context, token_segments, extract_patch, next_roles, terminal_status
except ImportError:
    from swe_edits import compile_edits, EDIT_SCHEMA
    from swe_context import repository_context_v2
    from swe_source_schema import source_choice_schema, check_source_choices
    from swe_source_blocks import source_blocks, resolve_source_blocks
    from swe_source_positions import source_positions, compile_positions
    from swe_core import ROLES, MANUAL, INSTRUCTIONS, MAX_NEW, Transcript, digest, git, repository_context, token_segments, extract_patch, next_roles, terminal_status


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def generation_limits(protocol):
    return dict(MAX_NEW, Coder=4096, Committer=4096) if protocol in {"exact_edits_v2", "exact_edits_v3", "source_blocks_v1", "source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"} else MAX_NEW.copy()


def sampling_params(protocol, role, seed, source_schema=None):
    params = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
              "max_new_tokens": generation_limits(protocol)[role], "sampling_seed": seed}
    if protocol == "exact_edits_v2" and role in {"Coder", "Committer"}:
        # Native SGLang 0.5.9 SamplingParams.json_schema takes a JSON string.
        params["json_schema"] = json.dumps(EDIT_SCHEMA, separators=(",", ":"))
    if protocol in {"exact_edits_v3", "source_blocks_v1", "source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"} and role in {"Coder", "Committer"}:
        if source_schema is None:
            raise ValueError("Source-choice protocol requires a recorded workflow schema")
        params["json_schema"] = json.dumps(source_schema, separators=(",", ":"))
    return params


def hidden_vector(meta, dim):
    raw = meta.get("hidden_states")
    if raw is None:
        raise ValueError("SGLang did not return a prefill hidden vector")
    x = np.asarray(raw, dtype=np.float32)
    # The registered server patch returns exactly one last-prefill vector.
    if x.shape not in {(dim,), (1, dim), (1, 1, dim)}:
        raise ValueError(f"Expected one post-norm prefill vector ({dim}), got {x.shape}")
    x = x.reshape(dim)
    if not np.isfinite(x).all() or float(np.linalg.norm(x)) == 0:
        raise ValueError("Non-finite or zero hidden vector")
    half = x.astype(np.float16)
    if not np.isfinite(half).all():
        raise ValueError("Hidden vector overflows float16")
    return half


def prepare_repository(cache: Path, row, workspace: Path):
    repo = cache / row["repo"].replace("/", "__")
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/" + row["repo"] + ".git", str(repo)], check=True, timeout=1800)
    check = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", row["base_commit"] + "^{commit}"], capture_output=True)
    if check.returncode:
        subprocess.run(["git", "-C", str(repo), "fetch", "origin", row["base_commit"]], check=True, timeout=1800)
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(workspace), row["base_commit"]], check=True, timeout=1800)
    return repo


class Collector:
    def __init__(self, args, tokenizer, hidden_dim):
        self.args, self.tokenizer, self.dim = args, tokenizer, hidden_dim
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=10))
        self.tools = asyncio.Semaphore(args.workers)
        self.workflows = asyncio.Semaphore(args.concurrency)
        self.trace = args.out / "swe.jsonl"
        self.uuid = uuid.uuid4().hex
        self.rows = []
        self.repo_locks = {}
        self.workflow_schemas = {}
        self.workflow_source_blocks = {}
        self.workflow_source_positions = {}

    def append(self, row):
        # Synchronous short writes within the event loop cannot interleave.
        with self.trace.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows.append(row)

    async def call(self, row, wid, transcript, role, history, round_i, index, seed):
        prompt = self.tokenizer.apply_chat_template(transcript.messages(role), tokenize=False,
                      add_generation_prompt=True, enable_thinking=False)
        ids, segments = token_segments(prompt, self.tokenizer)
        protocol = getattr(self.args, "patch_protocol", "unified_diff_v1")
        sampling = sampling_params(protocol, role, seed + index, self.workflow_schemas.get(wid))
        limit = sampling["max_new_tokens"]
        if len(ids) + limit > self.args.context_length:
            raise ValueError(f"Context budget exceeded at {role}: {len(ids)} + {limit}")
        payload = {"input_ids": ids, "stream": True, "return_hidden_states": True,
                   "sampling_params": sampling,
                   "my_chain_id": wid, "my_current_agent": role,
                   "my_chain_history": history.copy(), "my_is_selector": False}
        artifact = self.args.out / "calls" / f"{wid}_{index:02d}"
        artifact.parent.mkdir(exist_ok=True)
        save(artifact.with_suffix(".request.json"), payload)
        started, mono, first, last = time.time(), time.monotonic(), None, None
        async with self.client.stream("POST", self.args.base_url.rstrip("/") + "/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                item = json.loads(body)
                if item.get("error"):
                    raise RuntimeError(item["error"])
                if first is None and (item.get("text") or item.get("output_ids")):
                    first = time.time()
                last = item
        ended = time.time()
        if last is None or first is None:
            raise ValueError("No output tokens received from SGLang")
        meta = last.get("meta_info") or {}
        if meta.get("prompt_tokens") != len(ids):
            raise ValueError(f"Prompt token mismatch: {len(ids)} versus {meta.get('prompt_tokens')}")
        vector = hidden_vector(meta, self.dim)
        hidden = self.args.out / "hidden" / f"{wid}_{index:02d}.npy"
        hidden.parent.mkdir(exist_ok=True)
        np.save(hidden, vector, allow_pickle=False)
        text = last.get("text", "")
        save(artifact.with_suffix(".response.json"), {**last, "meta_info": {k: v for k, v in meta.items() if k != "hidden_states"}})
        self.append({"event": "call", "workflow_id": wid, "instance_id": row["instance_id"],
                     "repo": row["repo"], "setting": self.args.model_label, "role": role,
                     "round": round_i, "call_index": index, "thinking": False,
                     "queue_start": started, "prefill_end": first, "decode_end": ended,
                     "wall_monotonic_seconds": time.monotonic() - mono,
                     "prefill_time_source": "client first output token (proxy)",
                     "prompt_tokens": len(ids), "output_tokens": meta.get("completion_tokens"),
                     "cached_tokens": meta.get("cached_tokens"), "segments": segments,
                     "segment_tokens": len(ids), "hidden_file": str(hidden), "hidden_norm": float(np.linalg.norm(vector.astype(np.float32))),
                     "my_chain_history": history.copy(), "my_current_agent": role,
                     "native_role_field_semantics": "stock SGLang; transported fields are not used for PBKV eviction",
                     "finish_reason": meta.get("finish_reason"), "text": text})
        transcript.record(role, text)
        history.append(role)
        print(json.dumps({"workflow": wid, "role": role, "call_index": index, "tokens": meta.get("completion_tokens")}), flush=True)
        return text

    async def harness(self, request, directory, prepare=False):
        directory.mkdir(parents=True)
        save(directory / "request.json", request)
        command = [sys.executable, str(Path(__file__).with_name("swe_harness.py")),
                   "--request", str(directory / "request.json"), "--result", str(directory / "result.json")]
        if prepare:
            command.append("--prepare-only")
        with (directory / "adapter.log").open("w") as handle:
            process = await asyncio.create_subprocess_exec(*command, cwd=self.args.out,
                         stdout=handle, stderr=subprocess.STDOUT)
            await process.wait()
        result = json.loads((directory / "result.json").read_text())
        if result["status"] == "ERROR":
            raise RuntimeError(f"Official harness failed: {directory / 'result.json'}")
        return result

    async def workflow(self, row, seed):
        async with self.workflows:
            return await self._workflow(row, seed)

    async def _workflow(self, row, seed):
        wid = row["instance_id"] + "_s" + str(seed)
        workspace = self.args.out / "workspaces" / wid
        workspace.parent.mkdir(exist_ok=True)
        history, index, verdict, evaluations = [], 0, "ERROR", []
        try:
            async with self.repo_locks.setdefault(row["repo"], asyncio.Lock()):
                repo = await asyncio.to_thread(prepare_repository, self.args.repo_cache, row, workspace)
            if self.args.patch_protocol in {"exact_edits_v3", "source_blocks_v1", "source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"}:
                context, displayed = await asyncio.to_thread(repository_context_v2, repo,
                    row["base_commit"], row["problem_statement"], with_spans=True)
                if self.args.patch_protocol in {"source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"}:
                    context, schema, schema_evidence = await asyncio.to_thread(source_positions, repo, row["base_commit"], displayed, self.args.patch_protocol)
                    self.workflow_source_positions[wid] = schema_evidence
                elif self.args.patch_protocol == "source_blocks_v1":
                    context, schema, schema_evidence = await asyncio.to_thread(source_blocks, repo, row["base_commit"], displayed)
                    self.workflow_source_blocks[wid] = schema_evidence
                else:
                    schema, schema_evidence = await asyncio.to_thread(source_choice_schema, repo, row["base_commit"], displayed)
                self.workflow_schemas[wid] = schema
                save(self.args.out / f"{wid}_source_schema.json", {"schema": schema, "provenance": schema_evidence})
            else:
                retrieve = repository_context_v2 if self.args.patch_protocol == "exact_edits_v2" else repository_context
                context = await asyncio.to_thread(retrieve, repo, row["base_commit"], row["problem_statement"])
            (self.args.out / f"{wid}_context.txt").write_text(context)
            await self.harness({"instance": row}, self.args.out / "evaluations" / f"{wid}_prepare", prepare=True)
            transcript = Transcript(row, context, patch_protocol=self.args.patch_protocol)
            for attempt in range(1, 4):
                patch_text = ""
                for role in next_roles(attempt):
                    index += 1
                    text = await self.call(row, wid, transcript, role, history, attempt, index, seed)
                    if role == "Committer":
                        patch_text = text
                try:
                    artifact = self.args.out / f"{wid}_r{attempt}_patch"
                    artifact.with_suffix(".raw.txt").write_text(patch_text)
                    evidence = {"protocol": self.args.patch_protocol, "base_commit": row["base_commit"],
                                "raw_sha256": digest(patch_text), "apply_check_passed": False}
                    save(artifact.with_suffix(".json"), evidence)
                    if self.args.patch_protocol == "exact_edits_v3":
                        check_source_choices(json.loads(patch_text), self.workflow_schemas[wid])
                    compiler_text = patch_text
                    if self.args.patch_protocol == "source_blocks_v1":
                        resolved = resolve_source_blocks(json.loads(patch_text), self.workflow_source_blocks[wid])
                        save(artifact.with_suffix(".source_ids.json"), json.loads(patch_text))
                        save(artifact.with_suffix(".resolved_edits.json"), resolved)
                        compiler_text = json.dumps(resolved)
                    if self.args.patch_protocol in {"source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"}:
                        save(artifact.with_suffix(".source_positions.json"), json.loads(patch_text))
                        patch, edits = await asyncio.to_thread(compile_positions, patch_text, repo, row["base_commit"], self.workflow_source_positions[wid])
                        save(artifact.with_suffix(".resolved_edits.json"), edits)
                        save(artifact.with_suffix(".edits.json"), edits)
                        evidence["base_file_sha256"] = {e["path"]: self.workflow_source_positions[wid]["base_file_sha256"][e["path"]] for e in edits["edits"]}
                    elif self.args.patch_protocol in {"exact_edits_v1", "exact_edits_v2", "exact_edits_v3", "source_blocks_v1"}:
                        patch, edits = await asyncio.to_thread(compile_edits, compiler_text, repo, row["base_commit"])
                        save(artifact.with_suffix(".edits.json"), edits)
                        evidence["base_file_sha256"] = {e["path"]: digest(git(repo, "show", f"{row['base_commit']}:{e['path']}")) for e in edits["edits"]}
                    else:
                        patch = extract_patch(patch_text)
                    artifact.with_suffix(".diff").write_text(patch)
                    evidence["diff_sha256"] = digest(patch)
                    save(artifact.with_suffix(".json"), evidence)
                    checked = await asyncio.to_thread(subprocess.run, ["git", "-C", str(workspace), "apply", "--check", "-"], input=patch, capture_output=True, text=True)
                    if checked.returncode:
                        raise ValueError(checked.stderr)
                    evidence["apply_check_passed"] = True
                    save(artifact.with_suffix(".json"), evidence)
                except ValueError as error:
                    verdict = "PATCH_APPLY_FAILED"
                    evidence["error"] = str(error)
                    save(artifact.with_suffix(".json"), evidence)
                    save(self.args.out / f"{wid}_patch_error.json", {"attempt": attempt, "error": str(error)})
                    break
                queue_start = time.time()
                async with self.tools:
                    run_id = f"pbkv_{self.uuid}_{wid}_r{attempt}"
                    result = await self.harness({"instance": row, "patch": patch,
                              "model_label": self.args.model_label, "run_id": run_id,
                              "queue_start": queue_start, "timeout": self.args.test_timeout},
                              self.args.out / "evaluations" / f"{wid}_r{attempt}")
                response_received = time.time()
                evaluations.append(result)
                if not result.get("tests_executed"):
                    verdict = result["status"]
                    break
                base = {"event": "tool", "workflow_id": wid, "after_call_index": index,
                        "instance_id": row["instance_id"], "repo": row["repo"], "run_id": run_id}
                self.append(base | {"tool_type": "swe_test", **{k: result[k] for k in ("queue_start", "container_start", "exec_start", "exec_end")},
                                   "exec_monotonic_seconds": result["exec_monotonic_seconds"], "tests_executed": True})
                # Official grading, cleanup and response delivery are real elapsed
                # work after the test subprocess. Preserve them as a separate event.
                self.append(base | {"tool_type": "evaluation_finalize", "queue_start": result["exec_end"],
                           "container_start": result["exec_end"], "exec_start": result["exec_end"],
                           "exec_end": response_received})
                transcript.tool({k: result.get(k) for k in ("status", "resolved", "tests_executed", "test_output_tail")})
                verdict = terminal_status(result, attempt)
                if verdict is not None:
                    break
        except Exception as error:
            verdict = "ERROR"
            save(self.args.out / f"{wid}_error.json", {"error": repr(error), "call_index": index})
        finally:
            self.append({"event": "workflow_end", "workflow_id": wid, "instance_id": row["instance_id"],
                         "repo": row["repo"], "setting": self.args.model_label, "decode_end": time.time(),
                         "verdict": verdict, "calls": index, "evaluations": len(evaluations), "seed": seed})
            print(json.dumps({"workflow": wid, "verdict": verdict, "evaluations": len(evaluations)}), flush=True)
        return {"workflow_id": wid, "instance_id": row["instance_id"], "verdict": verdict, "evaluations": len(evaluations)}


async def run(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, AutoConfig
    manifest_path = args.prepared / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["dataset_inputs"]:
        if digest((args.prepared / item["local_file"]).read_bytes()) != item["sha256"]:
            raise ValueError("Prepared dataset hash mismatch")
    selected = manifest["pilot_instances"][args.offset:args.offset + args.limit]
    if not selected or any(r["partition"] != "dev" for r in selected):
        raise ValueError("This collector requires pre-registered development instances")
    expected = {r["instance_id"]: r for r in selected}
    # Never read the reference solution patch. Official harness metadata stays in
    # the controller and is never passed through Transcript.messages().
    columns = ["instance_id", "repo", "base_commit", "problem_statement", "version",
               "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"]
    rows = pq.read_table(args.prepared / manifest["dataset_inputs"][0]["local_file"], columns=columns).to_pylist()
    rows = {r["instance_id"]: r for r in rows if r["instance_id"] in expected}
    for wid, row in rows.items():
        if any(row[k] != expected[wid][k] for k in ("base_commit", "repo")):
            raise ValueError(f"Manifest mismatch: {wid}")
    if set(rows) != set(expected):
        raise ValueError("Missing registered instances")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    server = json.loads(args.server_manifest.read_text())
    if server["version"] != "0.5.9":
        raise ValueError("The registered model smoke requires actual SGLang 0.5.9")
    model = Path(args.model)
    model_files = {p.name: digest(p.read_bytes()) for p in [model / "config.json", model / "tokenizer_config.json", model / "model.safetensors.index.json"] if p.is_file()}
    index = model / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else [p.name for p in model.glob("*.safetensors")]
    if not shards or any(not (model / name).is_file() for name in shards):
        raise ValueError("Model weight shards are incomplete")
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {"registered_utc": time.time(), "stage": args.stage,
        "gpu": os.environ.get("PBKV_GPU"),
        "model_label": args.model_label, "model_revision": args.model_revision,
        "model_file_sha256": model_files, "weight_shard_bytes": {name: (model / name).stat().st_size for name in shards},
        "server_image": server,
        "pilot_manifest_sha256": digest(manifest_path.read_bytes()), "instances": list(expected),
        "all_instances_are_development": True, "seeds": args.seeds, "max_attempts": 3,
        "orchestration": "httpx native SGLang /generate: explicit hidden state stream and observation timestamps; no compatible stock AutoGen model client for these native fields",
        "patch_protocol": args.patch_protocol,
        "context_retrieval": {"version": "issue_source_v2" if args.patch_protocol in {"exact_edits_v2", "exact_edits_v3", "source_blocks_v1", "source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"} else "issue_source_v1",
                              "max_chars": 48000 if args.patch_protocol in {"exact_edits_v2", "exact_edits_v3", "source_blocks_v1", "source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"} else 12000,
                              "source_block_rendering": "visible_source_blocks_v1, <=64000 chars" if args.patch_protocol == "source_blocks_v1" else None,
                              "source_position_rendering": "visible_" + args.patch_protocol + ", original numbered lines, <=64000 chars" if args.patch_protocol in {"source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"} else None},
        "generation_json_schema": EDIT_SCHEMA if args.patch_protocol == "exact_edits_v2" else None,
        "generation_schema_mode": "per-workflow visible span ID enum and original inclusive line ranges; see *_source_schema.json" if args.patch_protocol in {"source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"} else ("per-workflow source ID enum; see *_source_schema.json" if args.patch_protocol == "source_blocks_v1" else ("per-workflow visible base-code enum; see *_source_schema.json" if args.patch_protocol == "exact_edits_v3" else "static or unconstrained")),
        "prompts": {"manual": MANUAL, "roles": Transcript({}, "", patch_protocol=args.patch_protocol).instructions, "max_new": generation_limits(args.patch_protocol)},
        "context_length": args.context_length, "workers": args.workers, "concurrency": args.concurrency,
        "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "hidden_dimension": config.hidden_size, "hidden_kind": "post-norm last-prefill token; requires registered server patch",
        "source_sha256": {p.name: digest(p.read_bytes()) for p in [
            *Path(__file__).parent.glob("swe_*.py"),
            *(Path(__file__).with_name(name) for name in ("validate_swe.py", "check_swe_gpu.py", "run_collect_swe.sh"))]},
        "dependencies": {n: importlib.metadata.version(n) for n in ("swebench", "transformers", "numpy", "httpx", "pyarrow")}}
    save(args.out / "run_manifest.json", provenance)
    save(args.out / "dataset.json", {"roles": [*ROLES, "END"]})
    collector = Collector(args, tokenizer, config.hidden_size)
    try:
        result = await asyncio.gather(*(collector.workflow(rows[wid], seed) for wid in expected for seed in map(int, args.seeds.split(","))))
        save(args.out / "status.json", {"stage": "collection_complete", "workflows": result,
             "errors": sum(r["verdict"] == "ERROR" for r in result), "test_evaluated": False,
             "formal_training_started": False})
    finally:
        await collector.client.aclose()
    try:
        from .validate_swe import validate
    except ImportError:
        from validate_swe import validate
    checked = validate(args.out)
    if args.stage == "smoke" and not checked["smoke_acceptance_passed"]:
        raise RuntimeError(f"Smoke acceptance failed: {args.out / 'validation.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-manifest", type=Path, required=True)
    parser.add_argument("--repo-cache", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "model_compare", "pilot"], default="smoke")
    parser.add_argument("--patch-protocol", choices=["unified_diff_v1", "exact_edits_v1", "exact_edits_v2", "exact_edits_v3", "source_blocks_v1", "source_positions_v1", "source_positions_v2", "source_positions_v3", "source_positions_text_v1"], default="unified_diff_v1")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--workers", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--test-timeout", type=int, default=1800)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.prepared = args.prepared.resolve()
    args.repo_cache = args.repo_cache.resolve()
    args.server_manifest = args.server_manifest.resolve()
    if args.limit < 1 or args.offset < 0 or args.offset + args.limit > 40 or args.concurrency < 1:
        parser.error("Use a valid subset of the 40 registered pilot instances")
    seeds = list(map(int, args.seeds.split(",")))
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("Sampling seeds must be nonempty and unique")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
