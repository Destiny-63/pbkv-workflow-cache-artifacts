"""Download and verify the plan's community AWQ fallback without allocating GPUs."""
import argparse
import json
import os
from pathlib import Path
import time

MODEL_ID = "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
REVISION = "c58857a7f41c0920f73d1b56678640f9c02017d7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import snapshot_download

    path = args.root / "models/Qwen3-Coder-30B-A3B-Instruct-AWQ"
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=path, max_workers=2,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "README.md", "LICENSE"])
    config = json.loads((path / "config.json").read_text())
    quant = config["quantization_config"]
    if (config["model_type"] != "qwen3_moe" or config["hidden_size"] != 2048
            or config["max_position_embeddings"] < 65536
            or quant["quant_method"] != "awq" or quant["bits"] != 4
            or quant["group_size"] != 128 or quant["version"].lower() != "gemm"):
        raise ValueError("Unexpected pinned AWQ candidate configuration")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(not (path / name).is_file() for name in shards):
        raise ValueError("Incomplete AWQ model shards")
    manifest = dict(model_id=MODEL_ID, revision=REVISION, local_path=str(path),
                    completed_unix=time.time(), quantization_config=quant,
                    hidden_dimension=2048, context_length=65536,
                    shards={name: (path / name).stat().st_size for name in shards},
                    gpu_compatibility_verified=False, actual_model_smoke_complete=False)
    (args.root / "runtime/awq_model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("AWQ_DOWNLOAD_COMPLETE_NOT_YET_GPU_VALIDATED", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
