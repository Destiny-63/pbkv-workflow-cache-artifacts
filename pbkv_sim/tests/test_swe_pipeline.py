"""Exercise split/audit integrity and the full train -> freeze -> infer -> replay path."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pbkv_sim.infer_predictor import export
from pbkv_sim.predictors import RoleTable
from pbkv_sim.swe_corpus import (ARCHITECTURE, build_corpus, convert_split, load_corpus,
                                  read_json, read_split, sha, write_new)
from pbkv_sim.swe_evaluation import DEFAULT_REPLAY, freeze, check_lock
from pbkv_sim.run_swe import run
from pbkv_sim.tests.test_trace import call
from pbkv_sim.train_predictor import make_observations, metrics, targets, upstream


def fixture(root):
    arch = read_json(ARCHITECTURE)
    rows = [dict(instance_id=f"{part}-instance", partition=part, repo="sample/repo")
            for part in ("train", "val", "test")]
    split_path = root / "split.json"
    write_new(split_path, {"instances": rows, "seeds": [0, 1, 2], "excluded_development_instances": []})
    scope = root / "scope.json"
    write_new(scope, {"quarantined_whole_failed_chunks": [], "quarantined_workflows": 0,
                      "effective_final_target": 9})
    runs = []
    for instance in rows:
        part, iid = instance["partition"], instance["instance_id"]
        folder = root / part
        (folder / "hidden").mkdir(parents=True)
        records, wids = [], []
        for seed in range(3):
            wid = f"{iid}_s{seed}"; wids.append(wid)
            history = []
            for idx, role in enumerate(arch["roles"][:-1], 1):
                hidden = f"hidden/{wid}_{idx}.npy"
                vector = np.random.default_rng(seed * 10 + idx).normal(size=arch["emb_dim"]).astype(np.float16)
                np.save(folder / hidden, vector)
                row = call(wid, role, idx, 1, idx * 4.0, 1.0,
                           setting="qwen3_coder-awq", segments=[["manual", 10], [f"{wid}-{idx}", 100]])
                row.update(instance_id=iid, hidden_file=f"/original/server/{part}/{hidden}",
                           my_chain_history=history.copy())
                history.append(role); records.append(row)
                if role == "Tester":
                    records.append(dict(event="tool", workflow_id=wid, after_call_index=idx,
                        tool_type="swe_test", queue_start=idx * 4 + 1, container_start=idx * 4 + 1,
                        exec_start=idx * 4 + 1, exec_end=idx * 4 + 3))
            records.append(dict(event="workflow_end", workflow_id=wid, instance_id=iid,
                setting="qwen3_coder-awq", decode_end=21.0, verdict="MAX_ATTEMPTS", repo="sample/repo"))
        (folder / "swe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        write_new(folder / "dataset.json", {"roles": arch["roles"]})
        write_new(folder / "run_manifest.json", {"stage": "formal", "formal_split_sha256": sha(split_path),
            "formal_partition": part, "instances": [iid], "seeds": "0,1,2", "source_sha256": {"collector": "fixture"},
            "hidden_dimension": arch["emb_dim"], "patch_protocol": "source_positions_feedback_v3"})
        for name in ("validation.json", "status.json"):
            write_new(folder / name, {"fixture": True})
        artifacts = {str(p.relative_to(folder)): sha(p) for p in folder.rglob("*") if p.is_file()}
        audit = {"observation_integrity_passed": True, "hidden_history_prefix_replay_and_actual_schema_checks_passed": True,
                 "formal_corpus": True, "source_sha256": {"collector": "fixture"},
                 "protocol": "source_positions_feedback_v3", "failed_evaluations": [], "error_workflows": [],
                 "artifact_sha256": artifacts, "expected_workflow_ids": wids, "workflows": 3, "calls": 15}
        write_new(root / f"{part}.audit.json", audit)
        runs.append({"data": part, "audit": f"{part}.audit.json"})
    runs_path = root / "runs.json"
    write_new(runs_path, {"runs": runs})
    return split_path, scope, runs_path


def corpus_fixture(root):
    split, scope, runs = fixture(root)
    corpus = build_corpus(root, split, runs, scope)
    path = root / "corpus.json"
    write_new(path, corpus)
    return path


def test_split_preserves_instance_groups_and_rejects_development_overlap(tmp_path):
    split, _, _ = fixture(tmp_path)
    out = tmp_path / "derived.json"
    assert convert_split(split, out) == {f"{p}-instance": p for p in ("train", "val", "test")}
    with pytest.raises(FileExistsError):
        convert_split(split, out)
    d = read_json(split)
    d["instances"].append(d["instances"][0])
    split.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Duplicate instance"):
        read_split(split)
    d["instances"].pop(); d["excluded_development_instances"] = ["test-instance"]
    split.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Development instances"):
        read_split(split)


def test_audit_rejects_tampering_unfinished_and_duplicate_runs(tmp_path):
    split, scope, runs = fixture(tmp_path)
    corpus = build_corpus(tmp_path, split, runs, scope)
    assert corpus["complete"] and not corpus["development_only"]
    assert corpus["counts"] == dict(train=3, val=3, test=3)
    d = read_json(runs); d["runs"].append(d["runs"][0]); runs.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Duplicate"):
        build_corpus(tmp_path, split, runs, scope)
    d["runs"].pop(); runs.write_text(json.dumps(d))
    audit = tmp_path / "train.audit.json"
    a = read_json(audit); a["observation_integrity_passed"] = False; audit.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="Audit has not passed"):
        build_corpus(tmp_path, split, runs, scope)
    a["observation_integrity_passed"] = True; audit.write_text(json.dumps(a))
    (tmp_path / "train" / "swe.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        build_corpus(tmp_path, split, runs, scope)


def test_training_loader_never_opens_test_artifacts_and_inference_has_no_targets(tmp_path):
    path = corpus_fixture(tmp_path)
    # Missing test hidden/trace must not affect train/val loading.
    (tmp_path / "test" / "swe.jsonl").unlink()
    for p in (tmp_path / "test" / "hidden").iterdir(): p.unlink()
    _, parts, _, roots = load_corpus(path, ("train", "val"))
    assert len(parts["train"]) == 3
    records, x = make_observations(parts["train"], tmp_path, 32,
                                  source_roots=roots, table=RoleTable(tuple(read_json(ARCHITECTURE)["roles"])))
    assert x.shape == (15, 2048) and all("targets" not in r for r in records)
    assert records[0]["history"] == [] and records[1]["history"] == [0]
    altered = copy.deepcopy(parts["train"])
    from dataclasses import replace
    altered[0].calls[1] = replace(altered[0].calls[1], role="Tester")
    other, other_x = make_observations(altered, tmp_path, 32, RoleTable(tuple(read_json(ARCHITECTURE)["roles"])), roots)
    assert records[0] == other[0]  # changing the future cannot change the current input
    np.testing.assert_array_equal(x[0], other_x[0])
    with pytest.raises(ValueError, match="evaluation-lock"):
        export(path, tmp_path / "nonexistent.pt", "test", tmp_path / "test-bank.json")


def test_partial_corpus_is_development_only_and_empty_horizon_metrics_are_finite(tmp_path):
    split, scope, runs = fixture(tmp_path)
    d = read_json(runs); d["runs"] = d["runs"][:2]; runs.write_text(json.dumps(d))
    corpus = build_corpus(tmp_path, split, runs, scope)
    assert corpus["development_only"] and not corpus["complete"]
    p = tmp_path / "partial.json"; write_new(p, corpus)
    with pytest.raises(ValueError, match="development/partial"):
        freeze(p, tmp_path, DEFAULT_REPLAY, tmp_path / "lock.json")
    probs = np.full((1, 3, 6), 1 / 6)
    result = metrics(probs, [{"targets": [5, -100, -100]}])
    assert result["k2"] == {"n": 0, "accuracy": None, "nll": None, "ece_10": None}


def test_fixed_architecture_and_end_masking():
    import torch
    a = read_json(ARCHITECTURE); c = a["cfg"]
    model = upstream().GraphSAGEHistSmall(5, c["d_node"], a["emb_dim"], c["text_proj_dim"],
        c["head_dims"], 6, 3, c["dropout"], torch.eye(5))
    assert model.head[0].in_features == 144 and model.head[-1].out_features == 18
    assert sum(p.numel() for p in model.parameters()) == 190178
    table = RoleTable(tuple(a["roles"]))
    assert targets(["Planner", "Coder"], 0, 3, table).tolist() == [1, 5, -100]
    assert targets(["Planner", "Coder"], 1, 3, table).tolist() == [5, -100, -100]


def test_timeout_scope_excludes_whole_instance_without_resplitting(tmp_path):
    split, scope, runs = fixture(tmp_path)
    d=read_json(runs);d['runs']=d['runs'][:2];runs.write_text(json.dumps(d))
    record=tmp_path/'timeout.json'
    write_new(record,dict(classification='verified_official_harness_timeout',
        original_chunk={'partition':'test'},quarantined_instances=['test-instance'],
        quarantined_pairs=[f'test-instance_s{s}' for s in [0,1,2]],
        model_resampled=False,timeout_changed=False,observations_rewritten=False))
    corpus=build_corpus(tmp_path,split,runs,scope,timeout_records=[record])
    assert corpus['complete'] and corpus['expected_workflows']==6
    assert corpus['split']['test-instance']=='test'
    bad=read_json(record);bad['original_chunk']['partition']='train';record.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='Invalid timeout'):
        build_corpus(tmp_path,split,runs,scope,timeout_records=[record])


def test_complete_formal_fixture_trains_freezes_and_replays_test(tmp_path):
    corpus = corpus_fixture(tmp_path)
    training = tmp_path / "training"
    subprocess.run([sys.executable, "-m", "pbkv_sim.train_predictor", "--corpus", str(corpus),
                    "--out", str(training), "--epochs", "2", "--patience", "1", "--seeds", "0,1", "--threads", "1"],
                   check=True, capture_output=True, text=True)
    config = copy.deepcopy(DEFAULT_REPLAY)
    config.update(concurrencies=[2], cap_ratios=[0.5], seeds=[0],
                  policies=["lru", "lae", "he_step", "cand", "ref_offline"])
    checkpoint = training / f"seed_{read_json(training / 'summary.json')['selected_seed']}" / "model.pt"
    export(corpus, checkpoint, "val", tmp_path / "val-bank.json", metrics_out=tmp_path / "val-metrics.json")
    lock_path = tmp_path / "lock.json"
    freeze(corpus, training, config, lock_path)
    out = tmp_path / "test-bank.json"
    bank = export(corpus, checkpoint, "test", out, lock_path, tmp_path / "test-metrics.json")
    assert len(bank["probabilities"]) == 15
    assert all("targets" not in v for v in bank["provenance"])
    rows = run(corpus, "test", tmp_path / "replay", bank_path=out, evaluation_lock=lock_path)
    assert len(rows) == 7 and all(r["access_sequence_verified"] for r in rows)
    assert {r["summary"]["workflows"] for r in rows} == {3}
    assert len({r["workload_sha256"] for r in rows}) == 1
    assert read_json(tmp_path / "test-metrics.json")["workflows"] == 3
    with pytest.raises(ValueError, match="config overrides"):
        run(corpus, "test", tmp_path / "override", config=config, bank_path=out, evaluation_lock=lock_path)
    wrong = read_json(out); wrong["provenance"]["partition"] = "val"
    corrupt = tmp_path / "wrong-bank.json"; write_new(corrupt, wrong)
    with pytest.raises(ValueError, match="corpus/partition"):
        run(corpus, "test", tmp_path / "wrong-replay", bank_path=corrupt, evaluation_lock=lock_path)


def test_available_subset_requires_explicit_scope_and_keeps_test_coverage(tmp_path):
    corpus_path = corpus_fixture(tmp_path)
    corpus = read_json(corpus_path)
    corpus.update(complete=False, development_only=True, expected_workflows=12,
                  missing_workflows=[f'missing-instance_s{s}' for s in range(3)])
    corpus_path.write_text(json.dumps(corpus))
    training = tmp_path / 'partial-training'
    subprocess.run([sys.executable, '-m', 'pbkv_sim.train_predictor', '--corpus', str(corpus_path),
                    '--out', str(training), '--epochs', '1', '--patience', '1', '--seeds', '0', '--threads', '1'],
                   check=True, capture_output=True, text=True)
    lock_path = tmp_path / 'subset-lock.json'
    with pytest.raises(ValueError, match='development/partial'):
        freeze(corpus_path, training, DEFAULT_REPLAY, lock_path)
    before = corpus_path.read_bytes()
    lock = freeze(corpus_path, training, DEFAULT_REPLAY, lock_path, allow_available_subset=True)
    assert corpus_path.read_bytes() == before
    assert lock['evaluation_scope'] == 'available_subset'
    assert lock['coverage']['missing_workflows'] == corpus['missing_workflows']
    checkpoint = training / 'seed_0/model.pt'
    bank_path, metrics_path = tmp_path / 'test-bank.json', tmp_path / 'metrics.json'
    export(corpus_path, checkpoint, 'test', bank_path, lock_path, metrics_path)
    assert read_json(metrics_path)['evaluation_scope'] == 'available_subset'
    assert read_json(metrics_path)['development_only'] is True
    assert read_json(metrics_path)['workflows'] == 3
    assert read_json(bank_path)['provenance']['partition'] == 'test'
    lock['coverage']['counts']['test'] = 6
    bad = tmp_path / 'bad-coverage-lock.json'
    write_new(bad, lock)
    with pytest.raises(ValueError, match='coverage changed'):
        check_lock(bad, corpus_path, checkpoint)
    lock['evaluation_scope'] = 'complete_formal'
    promoted = tmp_path / 'promoted-lock.json'
    write_new(promoted, lock)
    with pytest.raises(ValueError, match='formal corpus'):
        check_lock(promoted, corpus_path, checkpoint)
