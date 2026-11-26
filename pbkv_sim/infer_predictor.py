"""Export fixed-checkpoint SWE probabilities; test requires a final evaluation lock."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .predictors import MarkovPredictor, RoleTable
from .swe_corpus import ARCHITECTURE, load_corpus, read_json, sha, write_new
from .swe_evaluation import check_lock
from .train_predictor import (SOURCE, make_observations, observation_tensors, predict,
                              sliced_metrics, targets, upstream)


def export(corpus_path, checkpoint, partition, out, evaluation_lock=None, metrics_out=None, batch_size=256):
    if partition not in {"val", "test"} or batch_size < 1:
        raise ValueError("Expected val/test and a positive batch size")
    if partition == "test" and evaluation_lock is None:
        raise ValueError("Test inference requires --evaluation-lock")
    lock = check_lock(evaluation_lock, corpus_path, checkpoint) if evaluation_lock else None
    if Path(out).exists() or (metrics_out and Path(metrics_out).exists()):
        raise FileExistsError("Refusing to overwrite inference artifacts")
    subset = bool(lock and lock.get("evaluation_scope") == "available_subset")
    corpus, parts, traces, roots = load_corpus(corpus_path, (partition,), require_final=partition == "test" and not subset)
    arch = corpus["architecture"]
    table = RoleTable(tuple(arch["roles"]))
    wrapper = upstream().GraphSAGEChainPredictor(str(checkpoint), device="cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prov = ckpt["provenance"]
    if (prov.get("corpus_sha256") != sha(corpus_path)
            or prov["split_sha256"] != corpus["registered_split_sha256"]
            or prov.get("architecture_sha256") != sha(ARCHITECTURE)
            or ckpt["cfg"] != arch["cfg"] or sha(SOURCE) != arch["source_sha256"]
            or ckpt["meta"]["agents"] != arch["roles"]
            or ckpt["meta"]["future_k"] != arch["future_k"]
            or ckpt["meta"]["max_hist"] != arch["max_hist"]
            or ckpt["meta"]["emb_dim"] != arch["emb_dim"]):
        raise ValueError("Checkpoint training provenance or architecture differs from corpus")
    records, vectors = make_observations(parts[partition], Path(corpus["root"]), arch["max_hist"], table, roots)
    if vectors.shape[1] != arch["emb_dim"]:
        raise ValueError("Inference hidden dimension differs from checkpoint")
    batch = observation_tensors(records, vectors, arch["max_hist"], table)
    # No label builder is called before inference; history is prior calls only.
    probs = torch.softmax(predict(wrapper.model, batch, batch_size), -1).numpy()
    first = wrapper.predict_distributions(batch[3][0],
                [table.roles[i] for i in records[0]["history"]], table.roles[records[0]["current"]])
    np.testing.assert_allclose(first, probs[0], rtol=1e-5, atol=1e-6)
    bank = {"schema": 1, "roles": list(table.roles), "checkpoint_sha256": sha(checkpoint),
            "provenance": prov | {"partition": partition, "inference_trace_sha256": traces,
                "evaluation_lock_sha256": sha(evaluation_lock) if evaluation_lock else None,
                "inference_inputs": "current agent, prior agents, current prefill hidden only",
                "predictor_latency_included": False},
            "probabilities": {r["key"]: p.tolist() for r, p in zip(records, probs)}}
    bank["provenance"]["evaluation_scope"] = lock.get("evaluation_scope", "complete_formal") if lock else "validation"
    # Persist fixed predictions before building test labels or calculating metrics.
    write_new(out, bank)
    if metrics_out:
        # Labels are used solely after fixed predictions. The model stays in eval mode.
        labels = {f"{w.setting}|{w.workflow_id}|{t}": targets(w.roles(), t, 3, table).tolist()
                  for w in parts[partition] for t, _ in enumerate(w.steps())}
        labelled = [r | {"targets": labels[r["key"]]} for r in records]
        _, train, _, _ = load_corpus(corpus_path, ("train",))
        alpha = lock["replay"]["markov_alpha"] if lock else 0.5
        markov = MarkovPredictor(alpha=alpha, roles=table).fit([w.roles() for w in train["train"]])
        baseline = np.stack([markov.predict(table.roles[r["current"]], 3) for r in records])
        report = {"partition": partition, "development_only": corpus["development_only"],
                  "evaluation_scope": bank["provenance"]["evaluation_scope"],
                  "coverage": lock.get("coverage") if lock else None,
                  "corpus_sha256": sha(corpus_path), "checkpoint_sha256": sha(checkpoint),
                  "evaluation_lock_sha256": sha(evaluation_lock) if evaluation_lock else None,
                  "weighting": "call-level micro; instances and workflow seeds are not independent",
                  "instances": len({w.instance_id for w in parts[partition]}),
                  "workflows": len(parts[partition]), "samples": len(records),
                  "learned": sliced_metrics(probs, labelled), "markov": sliced_metrics(baseline, labelled)}
        write_new(metrics_out, report)
    return bank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("corpus", "checkpoint", "out"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--partition", choices=("val", "test"), required=True)
    ap.add_argument("--evaluation-lock", type=Path)
    ap.add_argument("--metrics-out", type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    export(args.corpus, args.checkpoint, args.partition, args.out, args.evaluation_lock,
           args.metrics_out, args.batch_size)


if __name__ == "__main__":
    main()
