"""Pin public dataset/model revisions and register the development-only pilot IDs.

This command downloads dataset inputs, not model weights, and never executes
patches or tests. It requires pyarrow for the pinned Verified parquet file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "PBKV-workflow-pilot/0.1"}), timeout=45)


def get_json(url):
    with request(url) as handle:
        return json.load(handle)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def pilot_ids(rows, count=40):
    salt = "20261105:swe-pilot:"
    hashed = lambda x: hashlib.sha256((salt + x).encode()).hexdigest()
    by_repo = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row)
    queues = {repo: deque(sorted(items, key=lambda x: hashed(x["instance_id"])))
              for repo, items in by_repo.items()}
    selected = []
    while len(selected) < count:
        before = len(selected)
        for repo in sorted(queues, key=hashed):
            if queues[repo] and len(selected) < count:
                row = queues[repo].popleft()
                selected.append({"instance_id": row["instance_id"], "repo": repo,
                                 "base_commit": row["base_commit"], "partition": "dev"})
        if len(selected) == before:
            raise ValueError("Not enough unique instances for pilot")
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    manifest = {"started_utc": now(), "status": "preparing", "tests_executed": False,
                "model_weights_downloaded": False, "swe_final_evaluation_performed": False}
    save(args.out / "status.json", manifest)
    try:
        models = []
        for model in ("Qwen/Qwen3-8B", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"):
            meta = get_json("https://huggingface.co/api/models/" + model + "?blobs=true")
            revision = meta["sha"]
            config = get_json(f"https://huggingface.co/{model}/resolve/{revision}/config.json")
            models.append({"model_id": model, "revision": revision, "config": config,
                           "safetensors_bytes": sum(x.get("size", 0) for x in meta.get("siblings", [])
                                                    if x["rfilename"].endswith(".safetensors")),
                           "requires_gpu_load_smoke": True})
        save(args.out / "model_metadata.json", models)

        name = "princeton-nlp/SWE-bench_Verified"
        meta = get_json("https://huggingface.co/api/datasets/" + name)
        revision = meta["sha"]
        candidates = [item["rfilename"] for item in meta["siblings"]
                      if item["rfilename"].endswith(".parquet") and "test" in item["rfilename"]]
        if not candidates:
            raise ValueError("Pinned dataset has no test parquet files")
        inputs, rows = [], []
        import pyarrow.parquet as pq
        for filename in sorted(candidates):
            url = f"https://huggingface.co/datasets/{name}/resolve/{revision}/{urllib.parse.quote(filename)}"
            target = args.out / Path(filename).name
            temporary = target.with_suffix(".partial")
            with request(url) as source, temporary.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            inputs.append({"repository_file": filename, "local_file": target.name, "sha256": digest,
                           "bytes": target.stat().st_size})
            rows.extend(pq.read_table(target, columns=["instance_id", "repo", "base_commit"]).to_pylist())
        if len(rows) != 500 or len({row["instance_id"] for row in rows}) != 500:
            raise ValueError(f"Expected 500 unique Verified instances, got {len(rows)}")
        selected = pilot_ids(rows)
        save(args.out / "pilot_manifest.json", {"registered_utc": now(), "dataset": name,
             "dataset_revision": revision, "dataset_inputs": inputs, "total_instances": len(rows),
             "rule": "repo round robin; SHA256(20261105:swe-pilot: + id/repo)",
             "pilot_instances": selected, "model_comparison_ids": [x["instance_id"] for x in selected[:12]],
             "sampling_seeds": [0, 1], "pilot_in_final_test": False,
             "candidate_results_used_for_selection": False})
        manifest.update(status="prepared", finished_utc=now(), pilot_instances=len(selected),
                        pilot_repositories=len({x["repo"] for x in selected}),
                        gpu_smoke_status="blocked_by_gpu_query_hang", collection_started=False)
        save(args.out / "status.json", manifest)
        print(json.dumps(manifest), flush=True)
    except Exception as error:
        manifest.update(status="failed", error=repr(error), updated_utc=now())
        save(args.out / "status.json", manifest)
        raise


if __name__ == "__main__":
    main()
