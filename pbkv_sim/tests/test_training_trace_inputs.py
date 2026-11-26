"""Import independent SWE runs without training a predictor."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pbkv_sim.predictors import DEFAULT_ROLES, RoleTable
from pbkv_sim.tests.test_trace import call
from pbkv_sim.train_predictor import load_training_inputs, make_samples


SWE_ROLES = RoleTable(("Planner", "Coder", "Committer", "Tester", "Analyzer", "END"))


def trace(root, directory, instance, seed, value=None, setting="qwen3_coder-awq", role="Planner"):
    folder = root / directory
    folder.mkdir(exist_ok=True)
    wid = f"{instance}_s{seed}"
    row = call(wid, role, 1, 1, 0, 1, setting=setting)
    row.update(instance_id=instance, hidden_file=f"/original/server/{directory}/hidden/{wid}.npy")
    end = dict(event="workflow_end", workflow_id=wid, instance_id=instance,
               setting=setting, decode_end=1, verdict="MAX_ATTEMPTS")
    path = folder / "swe.jsonl"
    with path.open("a") as handle:
        for record in (row, end):
            handle.write(json.dumps(record) + "\n")
    if value is not None:
        (folder / "hidden").mkdir(exist_ok=True)
        np.save(folder / "hidden" / f"{wid}.npy", np.full(4, value, dtype=np.float16))
    return str(path.relative_to(root))


def test_multiple_same_named_traces_keep_their_own_hidden_and_hashes(tmp_path):
    filenames = [trace(tmp_path, "batch", "first", 0, 1),
                 trace(tmp_path, "continuation", "second", 0, 2)]
    parts, hashes, roots = load_training_inputs(tmp_path, filenames, {"first": "train", "second": "val"})
    assert set(hashes) == {"batch/swe.jsonl", "continuation/swe.jsonl"}
    assert all(hashes[name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in filenames)
    for partition, value in [("train", 1), ("val", 2)]:
        records, vectors = make_samples(parts[partition], tmp_path, 4, 3, SWE_ROLES, roots)
        assert len(records) == 1 and np.array_equal(vectors, np.full((1, 4), value))


def test_instance_seeds_stay_together_and_test_hidden_is_not_read(tmp_path):
    first = trace(tmp_path, "batch", "same", 0, 3)
    trace(tmp_path, "batch", "same", 1, 4)
    second = trace(tmp_path, "continuation", "heldout", 0)
    parts, _, roots = load_training_inputs(tmp_path, [first, second], {"same": "train", "heldout": "test"})
    records, vectors = make_samples(parts["train"], tmp_path, 4, 3, SWE_ROLES, roots)
    assert len(records) == 2 and [w.instance_id for w in parts["train"]] == ["same", "same"]
    assert np.array_equal(vectors[:, 0], [3, 4])
    assert not parts["val"] and all("heldout" not in key[1] for key in roots)


def test_duplicate_workflow_across_runs_is_rejected(tmp_path):
    files = [trace(tmp_path, "batch", "same", 0, 1), trace(tmp_path, "continuation", "same", 0, 2)]
    with pytest.raises(ValueError, match="Repeated workflow"):
        load_training_inputs(tmp_path, files, {"same": "train"})


def test_legacy_different_settings_and_flat_artifacts_still_work(tmp_path):
    a = trace(tmp_path, "a", "legacy", 0, 1, setting="default", role="Writer")
    b = trace(tmp_path, "b", "legacy", 0, 2, setting="highdiff", role="Writer")
    parts, _, roots = load_training_inputs(tmp_path, [a, b], {"legacy": "train"})
    records, vectors = make_samples(parts["train"], tmp_path, 4, 3, DEFAULT_ROLES, roots)
    assert len(records) == 2 and np.array_equal(vectors[:, 0], [1, 2])
    flat = make_samples(parts["train"][:1], tmp_path / "a", 4, 3)
    assert np.array_equal(flat[1], vectors[:1])
