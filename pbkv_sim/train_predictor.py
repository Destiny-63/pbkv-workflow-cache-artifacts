"""Train the pinned upstream GraphSAGEHistSmall on recorded prefill states.

Only train/val IDs are read. A Reader group is one simulator step, represented
by its first call in call_index order. This is a documented adaptation of the
paper's invocation-level predictor to the existing step-level replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .predictors import DEFAULT_ROLES, END_ID, N_ROLES, ROLE_INDEX, STEP_ROLES, MarkovPredictor, RoleTable, _ece
from .trace import load_workflows
from .swe_corpus import ARCHITECTURE, load_corpus, read_json, read_split, sha

SOURCE = Path(__file__).resolve().parents[1] / "vendor/pbkv_0213e13/chain_predictor.py"


def upstream():
    spec = importlib.util.spec_from_file_location("pbkv_upstream_predictor", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def targets(roles: list[str], t: int, k: int, table=DEFAULT_ROLES) -> np.ndarray:
    """END is supervised once; positions strictly after END are ignored."""
    future = roles[t + 1:] + ["END"]
    return np.array([table.index[future[i]] if i < len(future) else -100 for i in range(k)])


def transition_matrix(workflows, table=DEFAULT_ROLES) -> np.ndarray:
    """Training-only, unsmoothed forward agent graph; sinks get a self-loop."""
    a = np.zeros((table.end_id, table.end_id), dtype=np.float32)
    for wf in workflows:
        roles = wf.roles()
        for current, nxt in zip(roles, roles[1:]):
            a[table.index[current], table.index[nxt]] += 1
    for i in range(table.end_id):
        if not a[i].sum():
            a[i, i] = 1
    return a / a.sum(axis=1, keepdims=True)


def hidden_path(recorded: str | None, data: Path) -> Path:
    if not recorded:
        raise ValueError("Missing prefill hidden state; refusing a zero-vector substitute")
    p = Path(recorded)
    if "hidden" in p.parts:
        p = data / Path(*p.parts[p.parts.index("hidden"):])
    elif not p.is_absolute():
        p = data / p
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_training_inputs(data: Path, filenames, split):
    """Keep each trace's artifact directory and provenance when combining runs."""
    by_part = {name: [] for name in ("train", "val")}
    inputs, source_roots, seen = {}, {}, set()
    for filename in filenames:
        path = data / filename
        inputs[str(Path(filename))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for wid, wf in load_workflows(path).items():
            key = (wf.setting, wid)
            if key in seen:
                raise ValueError(f"Repeated workflow in trace inputs: {key}")
            seen.add(key)
            part = split.get(wf.instance_id or wid)
            if part in by_part:
                by_part[part].append(wf)
                source_roots[key] = path.parent
    return by_part, inputs, source_roots


def make_observations(workflows, data: Path, max_hist: int, table=DEFAULT_ROLES, source_roots=None):
    """Build inference inputs from the current call and prior roles only."""
    records, vectors = [], []
    dim = None
    for wf in workflows:
        artifact_root = source_roots[(wf.setting, wf.workflow_id)] if source_roots is not None else data
        steps = wf.steps()
        prior_roles = []
        for t, step in enumerate(steps):
            call = step.calls[0]
            p = hidden_path(call.hidden_file, artifact_root)
            x = np.load(p, allow_pickle=False).astype(np.float32)
            if (x.ndim != 1 or not np.isfinite(x).all() or not np.linalg.norm(x)
                    or (dim is not None and len(x) != dim)):
                raise ValueError(f"Invalid hidden vector: {p} {x.shape}")
            dim = len(x)
            history = [table.index[r] for r in prior_roles[-max_hist:]]
            records.append({"key": f"{wf.setting}|{wf.workflow_id}|{t}",
                            "current": table.index[step.role], "history": history,
                            "instance_id": wf.instance_id or wf.workflow_id,
                            "setting": wf.setting, "role": step.role,
                            "call_index": call.call_index,
                            "hidden_sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
            vectors.append(x)
            prior_roles.append(step.role)
    if not vectors:
        raise ValueError("Empty training or validation partition")
    return records, np.stack(vectors)


def make_samples(workflows, data: Path, max_hist: int, k: int, table=DEFAULT_ROLES, source_roots=None):
    records, vectors = make_observations(workflows, data, max_hist, table, source_roots)
    labels = {f"{wf.setting}|{wf.workflow_id}|{t}": targets(wf.roles(), t, k, table).tolist()
              for wf in workflows for t, _ in enumerate(wf.steps())}
    return [r | {"targets": labels[r["key"]]} for r in records], vectors


def observation_tensors(records, x, max_hist, table=DEFAULT_ROLES):
    hist = [r["history"] + [table.end_id] * (max_hist - len(r["history"])) for r in records]
    return (torch.tensor([r["current"] for r in records]), torch.tensor(hist),
            torch.tensor([len(r["history"]) for r in records]), torch.from_numpy(x))


def tensors(records, x, max_hist, table=DEFAULT_ROLES):
    return (*observation_tensors(records, x, max_hist, table),
            torch.tensor([r["targets"] for r in records]))


@torch.no_grad()
def predict(model, batch, batch_size):
    model.eval()
    return torch.cat([model.forward_batch(*(v[i:i + batch_size] for v in batch[:4]))
                      for i in range(0, len(batch[0]), batch_size)])


def metrics(probs, records):
    y = np.asarray([r["targets"] for r in records])
    out = {}
    for k in range(y.shape[1]):
        valid = y[:, k] != -100
        p = probs[valid, k]
        target = y[valid, k]
        if not len(target):
            out[f"k{k + 1}"] = {"n": 0, "accuracy": None, "nll": None, "ece_10": None}
            continue
        correct = p.argmax(1) == target
        out[f"k{k + 1}"] = {
            "n": int(valid.sum()), "accuracy": float(correct.mean()),
            "nll": float(-np.log(np.maximum(p[np.arange(len(p)), target], 1e-30)).mean()),
            "ece_10": _ece(p.max(1).tolist(), correct.tolist()),
        }
    return out


def sliced_metrics(probs, records):
    out = {"all": metrics(probs, records)}
    for setting in sorted({r["setting"] for r in records}):
        for role in (None, *sorted({r["role"] for r in records if r["setting"] == setting})):
            ids = [i for i, r in enumerate(records) if r["setting"] == setting and
                   (role is None or r["role"] == role)]
            if ids:
                out[setting + ("_" + role if role else "")] = metrics(probs[ids], [records[i] for i in ids])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--corpus", type=Path, help="Audited SWE corpus; locks architecture and split")
    ap.add_argument("--development-check", action="store_true", help="Label direct-data training development only")
    ap.add_argument("--split", type=Path, default=Path("results/split.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-hist", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--metadata", type=Path, help="Dataset JSON with an ordered roles list ending in END")
    ap.add_argument("--trace", action="append", help="Dataset-relative trace filename; repeat for multiple files")
    args = ap.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.max_hist < 1:
        ap.error("epochs, patience, batch-size and max-hist must be positive")
    corpus = None
    if args.corpus:
        if args.data or args.trace or args.metadata or args.split != Path("results/split.json"):
            ap.error("--corpus supplies data, traces, roles and split; do not override them")
        corpus, by_part, inputs, source_roots = load_corpus(args.corpus, ("train", "val"))
        arch = corpus["architecture"]
        if args.max_hist != arch["max_hist"] or sha(SOURCE) != arch["source_sha256"]:
            raise ValueError("Architecture history or upstream source mismatch")
        args.data = Path(corpus["root"])
        table = RoleTable(tuple(arch["roles"]))
        split_digest = corpus["registered_split_sha256"]
    else:
        if not args.data:
            ap.error("--data or --corpus is required")
        table = RoleTable.from_metadata(args.metadata) if args.metadata else DEFAULT_ROLES
        if "Planner" in table.roles and not args.development_check:
            ap.error("Direct SWE data is development only; use --development-check or an audited --corpus")
        split = read_split(args.split)
        if not {"train", "val"}.issubset(set(split.values())):
            raise ValueError("Split must contain train and val partitions")
        by_part, inputs, source_roots = load_training_inputs(
            args.data, args.trace or ["hover_default.jsonl", "hover_highdiff.jsonl"], split)
        split_digest = sha(args.split)
    end_id, n_roles, role_names = table.end_id, len(table.roles), table.roles
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite training artifacts: {args.out}")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    # Both settings share the claim-level split. Test hidden vectors are never loaded.
    samples = {name: make_samples(wfs, args.data, args.max_hist, 3, table, source_roots)
               for name, wfs in by_part.items()}
    batch = {name: tensors(*samples[name], args.max_hist, table) for name in samples}
    dim = samples["train"][1].shape[1]
    if samples["val"][1].shape[1] != dim:
        raise ValueError("Train/val hidden dimensions differ")
    if corpus and dim != arch["emb_dim"]:
        raise ValueError("Hidden dimension differs from fixed SWE architecture")
    a = torch.from_numpy(transition_matrix(by_part["train"], table))
    meta = {"agents": list(role_names), "pad_id": end_id, "end_id": end_id,
            "n_classes": n_roles, "max_hist": args.max_hist, "future_k": 3, "emb_dim": dim}
    # Values are reproduction choices, not unpublished author hyperparameters.
    cfg = {"d_node": 32, "text_proj_dim": 80, "head_dims": [128], "dropout": 0.1,
           "use_attn": True, "use_bidir": False}
    if corpus:
        cfg = arch["cfg"]
    mod = upstream()
    markov = MarkovPredictor(roles=table).fit([w.roles() for w in by_part["train"]])
    mr = np.stack([markov.predict(role_names[r["current"]], 3) for r in samples["val"][0]])
    provenance = {
        "label": "PBKV (reimpl. predictor), step-level adaptation",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_commit": "0213e132efd5c49eb736bc08d032d4207cdcedf6",
        "split_sha256": split_digest,
        "corpus_sha256": sha(args.corpus) if args.corpus else None,
        "architecture_sha256": sha(ARCHITECTURE) if corpus else None,
        "development_only": args.development_check or (corpus["development_only"] if corpus else True),
        "trace_sha256": inputs, "hyperparameters": vars(args) | {"data": str(args.data),
            "split": str(args.split), "out": str(args.out),
            "metadata": str(args.metadata) if args.metadata else None,
            "corpus": str(args.corpus) if args.corpus else None},
        "torch": torch.__version__, "numpy": np.__version__,
        "partitions": {name: {"workflows": len(wfs), "unique_claims": len({w.instance_id or w.workflow_id for w in wfs}),
            "samples": len(samples[name][0])} for name, wfs in by_part.items()},
        "reader_rule": "first call_index in grouped Reader step, prior grouped steps only",
        "graph_rule": "train-only non-END transitions, no smoothing, self-loop at sinks",
        "validation_is_development_data": True, "test_evaluated": False,
        "selection": "minimum masked validation cross-entropy, across epochs and seeds",
    }
    args.out.mkdir(parents=True)
    summary = {"provenance": provenance, "markov_val": sliced_metrics(mr, samples["val"][0]), "runs": []}
    best_overall = (float("inf"), None)
    for seed in map(int, args.seeds.split(",")):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = mod.GraphSAGEHistSmall(end_id, cfg["d_node"], dim, cfg["text_proj_dim"],
                                      cfg["head_dims"], n_roles, 3, cfg["dropout"], a)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, state, epoch_best, stale, history = float("inf"), None, 0, 0, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(batch["train"][0]))
            for ids in order.split(args.batch_size):
                logits = model.forward_batch(*(v[ids] for v in batch["train"][:4]))
                loss = F.cross_entropy(logits.flatten(0, 1), batch["train"][4][ids].flatten(), ignore_index=-100)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            logits = predict(model, batch["val"], args.batch_size)
            val_loss = float(F.cross_entropy(logits.flatten(0, 1), batch["val"][4].flatten(), ignore_index=-100))
            history.append({"epoch": epoch, "val_nll": val_loss})
            if val_loss < best:
                best, state, epoch_best, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                stale += 1
            print(json.dumps({"seed": seed, "epoch": epoch, "val_nll": val_loss}), flush=True)
            if stale >= args.patience:
                break
        model.load_state_dict(state)
        ckpt = {"meta": meta, "cfg": cfg, "A_fwd": a, "state_dict": state,
                "provenance": provenance | {"seed": seed, "best_epoch": epoch_best}}
        seed_dir = args.out / f"seed_{seed}"
        seed_dir.mkdir()
        torch.save(ckpt, seed_dir / "model.pt")
        # Require the unmodified upstream serving wrapper to load our checkpoint.
        wrapper = mod.GraphSAGEChainPredictor(str(seed_dir), device="cpu")
        check = wrapper.predict_distributions(batch["val"][3][0],
                    [role_names[i] for i in samples["val"][0][0]["history"]],
                    role_names[samples["val"][0][0]["current"]])
        probs = torch.softmax(predict(model, batch["val"], args.batch_size), -1).numpy()
        np.testing.assert_allclose(check, probs[0], rtol=1e-5, atol=1e-6)
        report = {"seed": seed, "best_epoch": epoch_best, "val_nll": best,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "upstream_load_verified": True, "val": sliced_metrics(probs, samples["val"][0]),
                  "history": history}
        (seed_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        # Bank contains observations only; no target labels or future paths.
        bank = {"schema": 1, "roles": list(role_names), "provenance": ckpt["provenance"],
                "checkpoint_sha256": hashlib.sha256((seed_dir / "model.pt").read_bytes()).hexdigest(),
                "probabilities": {r["key"]: p.tolist() for r, p in zip(samples["val"][0], probs)}}
        (seed_dir / "val_predictions.json").write_text(json.dumps(bank) + "\n")
        summary["runs"].append({k: v for k, v in report.items() if k != "history"})
        if best < best_overall[0]:
            best_overall = (best, seed)
    summary["selected_seed"] = best_overall[1]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": True, "selected_seed": best_overall[1]}), flush=True)


if __name__ == "__main__":
    main()
