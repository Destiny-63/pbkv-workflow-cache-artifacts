#!/usr/bin/env python3
"""HoVer multi-agent trace collection against a local SGLang /generate server.

Records one JSONL row per call plus a workflow_end row. Prefill hidden states
are stored separately as float16 vectors of the post-norm last prompt token.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from pathlib import Path

import httpx
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROLES = ("Decomposer", "Reader", "Aggregator", "Verifier", "Writer")
VERDICTS = {"SUPPORTED", "NOT_SUPPORTED", "NEED_MORE"}


class BM25:
    def __init__(self, docs: list[str]):
        self.docs = [re.findall(r"[a-z0-9]+", doc.lower()) for doc in docs]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.df: dict[str, int] = {}
        for doc in self.docs:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1
        self.k1 = 1.5
        self.b = 0.75

    def top(self, query: str, k: int = 3) -> list[int]:
        if not self.docs:
            return []
        words = re.findall(r"[a-z0-9]+", query.lower())
        scores = []
        for doc in self.docs:
            tf: dict[str, int] = {}
            for word in doc:
                tf[word] = tf.get(word, 0) + 1
            score = 0.0
            dl = len(doc) or 1
            for word in words:
                freq = tf.get(word)
                if not freq:
                    continue
                n = self.df.get(word, 0)
                idf = math.log(1.0 + (self.n - n + 0.5) / (n + 0.5))
                score += idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                )
            scores.append(score)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:k] if scores[i] > 0] or order[:k]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def progress(phase: str, **fields) -> None:
    extra = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"COLLECT_PROGRESS phase={phase} {extra}".rstrip(), flush=True)


def parse_json(text: str):
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def as_pairs(value) -> list[tuple[str, list[str]]]:
    if value is None:
        return []
    if isinstance(value, dict):
        titles = value.get("title") or value.get("titles") or []
        sents = value.get("sentences") or value.get("text") or []
        pairs = []
        for title, sent in zip(titles, sents):
            if isinstance(sent, list):
                pairs.append((str(title), [str(x) for x in sent]))
            else:
                pairs.append((str(title), [str(sent)]))
        return pairs
    if isinstance(value, list):
        pairs = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sent = item[1]
                if isinstance(sent, list):
                    pairs.append((title, [str(x) for x in sent]))
                else:
                    pairs.append((title, [str(sent)]))
            elif isinstance(item, dict):
                title = str(item.get("title") or "")
                sent = item.get("sentences") or item.get("text") or []
                if isinstance(sent, str):
                    sent = [sent]
                pairs.append((title, [str(x) for x in sent]))
        return pairs
    return []


def fact_titles(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        if isinstance(value.get("title"), list):
            return {str(x) for x in value["title"]}
        if isinstance(value.get("key"), list):
            return {str(x) for x in value["key"]}
    titles = set()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                if item.get("key"):
                    titles.add(str(item["key"]))
                elif item.get("title"):
                    titles.add(str(item["title"]))
            elif isinstance(item, (list, tuple)) and item:
                titles.add(str(item[0]))
    return titles


def load_prompts(prompt_dir: Path) -> dict[str, str]:
    manual = (prompt_dir / "manual.txt").read_text()
    prompts = {}
    for role, name in (
        ("Decomposer", "decomposer.txt"),
        ("Reader", "reader.txt"),
        ("Aggregator", "aggregator.txt"),
        ("Verifier", "verifier.txt"),
        ("Writer", "writer.txt"),
    ):
        prompts[role] = manual.strip() + "\n\n" + (prompt_dir / name).read_text().strip()
    return prompts


def try_split(path: str, config: str | None, splits: tuple[str, ...]):
    last = None
    for split in splits:
        try:
            if config:
                ds = load_dataset(path, config, split=split)
            else:
                ds = load_dataset(path, split=split)
            return ds, split
        except Exception as exc:  # dataset scripts differ by hub revision
            last = exc
            progress("data", status="split_miss", path=path, split=split, error=type(exc).__name__)
    raise RuntimeError(f"no split for {path}: {last}")


def align_claims(cache_path: Path) -> list[dict]:
    if cache_path.exists():
        rows = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
        progress("data", status="cache", claims=len(rows))
        return rows
    progress("data", status="downloading", dataset="vincentkoc/hover-parquet")
    hover, hover_split = try_split("vincentkoc/hover-parquet", None, ("validation", "dev", "test"))
    progress("data", status="downloading", dataset="hotpot_qa", hover_rows=len(hover), hover_split=hover_split)
    try:
        hotpot_val, hotpot_split = try_split("hotpot_qa", "distractor", ("validation", "dev"))
        hotpot_train, _train_split = try_split("hotpot_qa", "distractor", ("train",))
    except Exception:
        hotpot_val, hotpot_split = try_split("hotpotqa/hotpot_qa", "distractor", ("validation", "dev"))
        hotpot_train, _train_split = try_split("hotpotqa/hotpot_qa", "distractor", ("train",))
    progress(
        "data",
        status="indexing",
        hotpot_rows=len(hotpot_val) + len(hotpot_train),
        hotpot_split=hotpot_split + "+train",
    )
    by_id = {}
    for row in hotpot_train:
        by_id[str(row.get("id") or row.get("_id"))] = row
    for row in hotpot_val:
        by_id[str(row.get("id") or row.get("_id"))] = row
    sample = list(hover.select(range(min(50, len(hover)))))
    aligned = []
    for row in hover:
        hpqa = str(row.get("hpqa_id") or row.get("hpqa") or "")
        base = by_id.get(hpqa)
        if base is None:
            continue
        passages = as_pairs(base.get("context"))
        titles = {title for title, _ in passages}
        needed = fact_titles(row.get("supporting_facts"))
        if not needed or not needed.issubset(titles):
            continue
        claim = str(row.get("claim") or "")
        uid = str(row.get("uid") or row.get("id") or hpqa)
        docs = []
        for title, sents in passages:
            docs.append({"title": title, "text": " ".join(sents)})
        aligned.append({"workflow_id": uid, "claim": claim, "hpqa_id": hpqa, "docs": docs})
    sample_ids = {str(s.get("uid") or s.get("id")) for s in sample}
    sample_checked = 0
    sample_ok = 0
    for row in sample:
        sample_checked += 1
        hpqa = str(row.get("hpqa_id") or "")
        base = by_id.get(hpqa)
        if base is None:
            continue
        titles = {title for title, _ in as_pairs(base.get("context"))}
        needed = fact_titles(row.get("supporting_facts"))
        if needed and needed.issubset(titles):
            sample_ok += 1
    progress(
        "data",
        status="aligned",
        sample_ok=sample_ok,
        sample_n=sample_checked,
        aligned=len(aligned),
        hover_split=hover_split,
        hotpot_split=hotpot_split + "+train",
    )
    if sample_checked and sample_ok / sample_checked < 0.3:
        progress("data", status="fallback_hotpot", reason="hover_match_below_30pct")
        aligned = []
        for row in hotpot_val:
            docs = [{"title": title, "text": " ".join(sents)} for title, sents in as_pairs(row.get("context"))]
            if len(docs) < 2:
                continue
            aligned.append(
                {
                    "workflow_id": str(row.get("id") or row.get("_id")),
                    "claim": str(row.get("question") or ""),
                    "hpqa_id": str(row.get("id") or ""),
                    "docs": docs,
                    "source": "hotpot_qa",
                }
            )
    aligned.sort(key=lambda item: item["workflow_id"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as handle:
        for item in aligned:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    progress("data", status="wrote_cache", claims=len(aligned), path=str(cache_path))
    return aligned


def hidden_vector(meta: dict) -> np.ndarray | None:
    raw = (meta or {}).get("hidden_states")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[-1]
    if arr.ndim != 1 or arr.shape[0] != 4096:
        return None
    return arr.astype(np.float16)


class Collector:
    def __init__(self, args, prompts: dict[str, str], tokenizer):
        self.args = args
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        self.sem = asyncio.Semaphore(args.concurrency)
        self.write_lock = asyncio.Lock()
        self.t0 = time.time()
        self.prompt_tokens = {
            role: len(tokenizer.encode(text)) for role, text in prompts.items()
        }

    def render(self, role: str, user: str, thinking: bool) -> str:
        messages = [
            {"role": "system", "content": self.prompts[role]},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    async def generate(self, text: str, max_new: int) -> dict:
        payload = {
            "text": text,
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": max_new,
                "stop": ["<|im_end|>"],
            },
            "stream": True,
            "return_hidden_states": True,
        }
        queue_start = time.time()
        first = None
        last = None
        async with self.client.stream("POST", self.args.base_url.rstrip("/") + "/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                data = json.loads(body)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                if first is None and (data.get("text") or data.get("output_ids")):
                    first = time.time()
                last = data
        end = time.time()
        if last is None:
            raise RuntimeError("empty generate stream")
        if first is None:
            first = end
        meta = last.get("meta_info") or {}
        return {
            "text": last.get("text") or "",
            "meta": meta,
            "queue_start": queue_start,
            "prefill_end": first,
            "decode_end": end,
            "prompt_tokens": meta.get("prompt_tokens"),
            "output_tokens": meta.get("completion_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
        }

    async def call_role(self, role: str, user: str, max_new: int, thinking: bool, hidden_path: Path) -> dict:
        prompt = self.render(role, user, thinking)
        chain = [sha(self.prompts[role])[:16], sha(prompt)[:16]]
        result = None
        parsed = None
        for _attempt in range(2):
            result = await self.generate(prompt, max_new)
            parsed = parse_json(result["text"])
            if parsed is not None:
                break
        vector = hidden_vector(result["meta"])
        if vector is not None:
            hidden_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(hidden_path, vector)
        return {
            "role": role,
            "prefix_hash_chain": chain,
            "prompt_tokens": result["prompt_tokens"],
            "output_tokens": result["output_tokens"],
            "cached_tokens": result["cached_tokens"],
            "queue_start": result["queue_start"],
            "queue_end": None,
            "prefill_start": None,
            "prefill_end": result["prefill_end"],
            "decode_start": result["prefill_end"],
            "decode_end": result["decode_end"],
            "text": result["text"],
            "parsed": parsed,
            "hidden_file": str(hidden_path) if vector is not None else None,
        }

    def row(self, claim: dict, setting: str, event: str, **fields) -> dict:
        base = {
            "event": event,
            "workflow_id": claim["workflow_id"],
            "setting": setting,
            "role": fields.get("role"),
            "prefix_hash_chain": fields.get("prefix_hash_chain"),
            "prompt_tokens": fields.get("prompt_tokens"),
            "output_tokens": fields.get("output_tokens"),
            "cached_tokens": fields.get("cached_tokens"),
            "queue_start": fields.get("queue_start"),
            "queue_end": fields.get("queue_end"),
            "prefill_start": fields.get("prefill_start"),
            "prefill_end": fields.get("prefill_end"),
            "decode_start": fields.get("decode_start"),
            "decode_end": fields.get("decode_end"),
            "tool_start": fields.get("tool_start"),
            "tool_end": fields.get("tool_end"),
            "round": fields.get("round"),
            "call_index": fields.get("call_index"),
            "parse_ok": fields.get("parse_ok"),
            "hidden_file": fields.get("hidden_file"),
        }
        for key in ("verdict", "subquestions", "text"):
            if key in fields:
                base[key] = fields[key]
        return base

    async def append(self, path: Path, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        async with self.write_lock:
            with path.open("a") as handle:
                handle.write(line)
                handle.flush()

    async def one_workflow(self, claim: dict, setting: str, out_path: Path, hidden_dir: Path) -> dict:
        call_index = 0
        missing = ""
        rounds = 0
        json_fail = 0
        calls = 0
        verdict = None
        evidence = ""
        thinking_verifier = setting == "highdiff"
        async with self.sem:
            for round_i in range(1, 4):
                rounds = round_i
                user = f"Claim: {claim['claim']}"
                if missing:
                    user += f"\nMissing from the previous round: {missing}"
                call_index += 1
                calls += 1
                dec = await self.call_role(
                    "Decomposer",
                    user,
                    256,
                    False,
                    hidden_dir / f"{_safe(claim['workflow_id'])}_{call_index}.npy",
                )
                parsed = dec["parsed"] or {}
                subqs = parsed.get("subquestions") if isinstance(parsed.get("subquestions"), list) else None
                if dec["parsed"] is None or not subqs:
                    json_fail += 1
                subqs = [str(item) for item in (subqs or []) if str(item).strip()][:3]
                if len(subqs) < 2 and dec["parsed"] is not None:
                    json_fail += 1
                await self.append(
                    out_path,
                    self.row(
                        claim,
                        setting,
                        "call",
                        **dec,
                        round=round_i,
                        call_index=call_index,
                        parse_ok=bool(subqs),
                        subquestions=subqs,
                        tool_start=None,
                        tool_end=None,
                    ),
                )
                if not subqs:
                    subqs = [claim["claim"]]

                async def read_one(subq: str, idx: int):
                    tool_start = time.time()
                    bm25 = BM25([doc["text"] for doc in claim["docs"]])
                    picks = bm25.top(subq, 3)
                    tool_end = time.time()
                    blocks = []
                    for n, doc_i in enumerate(picks, start=1):
                        doc = claim["docs"][doc_i]
                        blocks.append(f"[{n}] title: {doc['title']}\n{doc['text'][:1800]}")
                    reader_user = f"Subquestion: {subq}\n\nPassages:\n" + "\n\n".join(blocks)
                    result = await self.call_role(
                        "Reader",
                        reader_user,
                        384,
                        False,
                        hidden_dir / f"{_safe(claim['workflow_id'])}_{idx}.npy",
                    )
                    result["tool_start"] = tool_start
                    result["tool_end"] = tool_end
                    result["call_index"] = idx
                    return result

                base_index = call_index
                tasks = []
                for offset, subq in enumerate(subqs, start=1):
                    tasks.append(read_one(subq, base_index + offset))
                call_index += len(subqs)
                readers = await asyncio.gather(*tasks)
                reader_payload = []
                for reader in readers:
                    calls += 1
                    ok = isinstance(reader["parsed"], dict) and "evidence" in reader["parsed"]
                    if not ok:
                        json_fail += 1
                    ev = (reader["parsed"] or {}).get("evidence") or ""
                    src = (reader["parsed"] or {}).get("source") or ""
                    reader_payload.append({"evidence": ev, "source": src})
                    await self.append(
                        out_path,
                        self.row(
                            claim,
                            setting,
                            "call",
                            **reader,
                            round=round_i,
                            parse_ok=ok,
                        ),
                    )

                call_index += 1
                calls += 1
                agg_user = "Claim: " + claim["claim"] + "\nReaders: " + json.dumps(reader_payload, ensure_ascii=False)
                agg = await self.call_role(
                    "Aggregator",
                    agg_user,
                    512,
                    False,
                    hidden_dir / f"{_safe(claim['workflow_id'])}_{call_index}.npy",
                )
                if agg["parsed"] is None or "evidence" not in (agg["parsed"] or {}):
                    json_fail += 1
                    evidence = " ".join(item["evidence"] for item in reader_payload if item["evidence"])
                else:
                    evidence = str(agg["parsed"].get("evidence") or "")
                await self.append(
                    out_path,
                    self.row(claim, setting, "call", **agg, round=round_i, call_index=call_index, parse_ok=agg["parsed"] is not None),
                )

                call_index += 1
                calls += 1
                ver_user = f"Claim: {claim['claim']}\nEvidence: {evidence}"
                ver = await self.call_role(
                    "Verifier",
                    ver_user,
                    2048 if thinking_verifier else 256,
                    thinking_verifier,
                    hidden_dir / f"{_safe(claim['workflow_id'])}_{call_index}.npy",
                )
                verdict_raw = str((ver["parsed"] or {}).get("verdict") or "").strip().upper()
                verdict = verdict_raw if verdict_raw in VERDICTS else None
                if verdict is None:
                    json_fail += 1
                    verdict = "NOT_SUPPORTED"
                missing = str((ver["parsed"] or {}).get("missing") or "")
                await self.append(
                    out_path,
                    self.row(
                        claim,
                        setting,
                        "call",
                        **ver,
                        round=round_i,
                        call_index=call_index,
                        parse_ok=verdict_raw in VERDICTS,
                        verdict=verdict,
                    ),
                )
                if verdict != "NEED_MORE" or round_i == 3:
                    break

            call_index += 1
            calls += 1
            writer_user = f"Claim: {claim['claim']}\nVerdict: {verdict}\nEvidence: {evidence}"
            writer = await self.call_role(
                "Writer",
                writer_user,
                512,
                False,
                hidden_dir / f"{_safe(claim['workflow_id'])}_{call_index}.npy",
            )
            if writer["parsed"] is None:
                json_fail += 1
            await self.append(
                out_path,
                self.row(claim, setting, "call", **writer, round=rounds, call_index=call_index, parse_ok=writer["parsed"] is not None),
            )
            end = time.time()
            await self.append(
                out_path,
                self.row(
                    claim,
                    setting,
                    "workflow_end",
                    role="END",
                    decode_end=end,
                    round=rounds,
                    verdict=verdict,
                    parse_ok=json_fail == 0,
                ),
            )
        return {"json_fail": json_fail, "calls": calls, "rounds": rounds, "verdict": verdict}

    async def run_split(self, claims: list[dict], setting: str, out_path: Path, hidden_dir: Path, phase: str) -> dict:
        done = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") == "workflow_end":
                    done.add(row["workflow_id"])
        pending = [c for c in claims if c["workflow_id"] not in done]
        total = len(claims)
        stats = {"ok": len(done), "json_fail_calls": 0, "calls": 0, "need_more": 0, "finished": len(done)}
        progress(phase, status="start", done=len(done), total=total, pending=len(pending), setting=setting)
        lock = asyncio.Lock()

        async def wrapped(claim):
            try:
                result = await self.one_workflow(claim, setting, out_path, hidden_dir)
            except Exception as exc:
                progress(phase, status="workflow_error", workflow_id=claim["workflow_id"], error=type(exc).__name__)
                await self.append(
                    out_path,
                    self.row(
                        claim,
                        setting,
                        "workflow_end",
                        role="END",
                        decode_end=time.time(),
                        verdict="ERROR",
                        parse_ok=False,
                        text=str(exc)[:300],
                    ),
                )
                result = {"json_fail": 1, "calls": 0, "rounds": 0, "verdict": "ERROR"}
            async with lock:
                stats["finished"] += 1
                stats["calls"] += result["calls"]
                stats["json_fail_calls"] += result["json_fail"]
                if result["json_fail"] == 0 and result["verdict"] != "ERROR":
                    stats["ok"] += 1
                if result["rounds"] >= 2:
                    stats["need_more"] += 1
                if stats["finished"] % 1 == 0:
                    elapsed = int(time.time() - self.t0)
                    progress(
                        phase,
                        done=f"{stats['finished']}/{total}",
                        ok=stats["ok"],
                        json_fail_calls=stats["json_fail_calls"],
                        calls=stats["calls"],
                        retry_wf=stats["need_more"],
                        elapsed_s=elapsed,
                        setting=setting,
                    )
            return result

        await asyncio.gather(*(wrapped(claim) for claim in pending))
        retry_rate = (stats["need_more"] / stats["finished"]) if stats["finished"] else 0.0
        json_rate = 1.0 - (stats["json_fail_calls"] / stats["calls"]) if stats["calls"] else 0.0
        progress(
            phase,
            status="done",
            finished=stats["finished"],
            json_ok_rate=f"{json_rate:.3f}",
            retry_rate=f"{retry_rate:.3f}",
            setting=setting,
        )
        return {"json_ok_rate": json_rate, "retry_rate": retry_rate, **stats}

    async def close(self):
        await self.client.aclose()


def _safe(workflow_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", workflow_id)


async def wait_server(base_url: str) -> None:
    deadline = time.time() + 40 * 60
    while time.time() < deadline:
        try:
            resp = httpx.get(base_url.rstrip("/") + "/health", timeout=3.0)
            if resp.status_code == 200:
                progress("server", status="healthy", url=base_url)
                return
        except Exception:
            pass
        progress("server", status="waiting")
        await asyncio.sleep(15)
    raise SystemExit("server did not become healthy")


async def main_async(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompt_dir))
    claims = align_claims(out / "aligned_claims.jsonl")
    if len(claims) < args.smoke + 1:
        raise SystemExit(f"only {len(claims)} claims after alignment")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    collector = Collector(args, prompts, tokenizer)
    token_report = " ".join(f"{role}={n}" for role, n in collector.prompt_tokens.items())
    progress("prompts", tokens=token_report)
    meta = {
        "model": args.model,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "temperature": 0.7,
        "enable_thinking_default": False,
        "highdiff": "verifier_thinking_only",
        "schedule": "sglang_default_fcfs_radix_lru",
        "prefetch": False,
        "hidden": "post_norm_last_prefill_token_fp16_4096",
        "prompt_sha256": {role: sha(text) for role, text in prompts.items()},
        "prompt_tokens": collector.prompt_tokens,
        "queue_end": "not exported by SGLang; prefill_end is first streamed token",
        "started": time.time(),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    await wait_server(args.base_url)
    smoke_claims = claims[: args.smoke]
    formal = claims[args.smoke : args.smoke + args.limit]
    progress("plan", smoke=len(smoke_claims), formal=len(formal), aligned=len(claims))
    if not args.skip_smoke:
        smoke_stats = await collector.run_split(
            smoke_claims, "default", out / "smoke.jsonl", out / "hidden" / "smoke", "smoke"
        )
        print(
            "SMOKE_RESULT "
            + json.dumps(
                {
                    "json_ok_rate": round(smoke_stats["json_ok_rate"], 3),
                    "retry_rate": round(smoke_stats["retry_rate"], 3),
                    "finished": smoke_stats["finished"],
                }
            ),
            flush=True,
        )
        if smoke_stats["json_ok_rate"] < 0.8:
            progress("smoke", status="stop", reason="json_ok_rate_below_0.8")
            await collector.close()
            raise SystemExit(3)
    if not args.smoke_only:
        await collector.run_split(
            formal, "default", out / "hover_default.jsonl", out / "hidden" / "default", "default"
        )
        await collector.run_split(
            formal, "highdiff", out / "hover_highdiff.jsonl", out / "hidden" / "highdiff", "highdiff"
        )
    progress("all", status="finished")
    await collector.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--smoke", type=int, default=10)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
