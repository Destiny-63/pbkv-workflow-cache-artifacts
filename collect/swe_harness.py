"""Instrument the unmodified SWE-bench 3.0.17 harness, one process per evaluation.

Run from a dedicated output directory. The process owns its unique run_id and
only the instance container that the official harness creates for that run.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path


def observed_test_stream(chunks, result, start_marker, end_marker):
    """Pass Docker bytes through unchanged and timestamp actual test markers."""
    tail = b""
    for chunk in chunks:
        combined = tail + chunk
        if start_marker in combined and result.get("exec_start") is None:
            result["exec_start"] = time.time()
            result["test_monotonic_start"] = time.monotonic()
        if end_marker in combined and result.get("exec_start") is not None and result.get("exec_end") is None:
            result["exec_end"] = time.time()
            result["exec_monotonic_seconds"] = time.monotonic() - result["test_monotonic_start"]
        tail = combined[-max(len(start_marker), len(end_marker)):]
        yield chunk


def evaluate(request, prepare_only=False):
    if importlib.metadata.version("swebench") != "3.0.17":
        raise RuntimeError("This adapter requires the registered swebench==3.0.17 API")
    import docker
    import swebench.harness.run_evaluation as official
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.constants import LOG_REPORT, LOG_TEST_OUTPUT, LOG_INSTANCE, APPLY_PATCH_FAIL, START_TEST_OUTPUT, END_TEST_OUTPUT
    from docker.models.containers import ContainerCollection

    issue = request["instance"]
    spec = make_test_spec(issue, namespace="swebench")
    client = docker.from_env(timeout=1800)
    # Pull/build costs are preparation, outside the measured tool event.
    try:
        image = client.images.get(spec.instance_image_key)
        image_was_local = True
    except docker.errors.ImageNotFound:
        image = client.images.pull(spec.instance_image_key)
        image_was_local = False
    result = {"instance_id": issue["instance_id"], "image": spec.instance_image_key,
              "image_id": image.id, "image_digests": image.attrs.get("RepoDigests", []),
              "image_was_local": image_was_local, "harness_version": "3.0.17"}
    if prepare_only:
        client.close()
        return result | {"status": "PREPARED", "tests_executed": False}

    pred = {"instance_id": issue["instance_id"], "model_name_or_path": request["model_label"],
            "model_patch": request["patch"]}
    log_dir = official.RUN_EVALUATION_LOG_DIR / request["run_id"] / request["model_label"].replace("/", "__") / issue["instance_id"]
    if log_dir.exists():
        raise FileExistsError(f"Refusing existing evaluation path/cache: {log_dir}")
    result.update(queue_start=request["queue_start"], container_start=None,
                  exec_start=None, exec_end=None, tests_executed=False, resolved=False,
                  status="ERROR", run_id=request["run_id"], log_dir=str(log_dir.resolve()))
    original_create = ContainerCollection.create
    original_build = official.build_container
    original_exec = official.exec_run_with_timeout
    original_api_exec_start = client.api.exec_start
    inside_eval = False

    def limited_create(collection, *args, **kwargs):
        kwargs.update(nano_cpus=2_000_000_000, mem_limit="8g", pids_limit=512)
        env = kwargs.get("environment", {})
        if isinstance(env, list):
            env = dict(item.split("=", 1) for item in env)
        kwargs["environment"] = dict(env, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        container = original_create(collection, *args, **kwargs)
        result["container_id"] = container.id
        return container

    def timed_build(*args, **kwargs):
        result["container_start"] = time.time()
        result["container_monotonic_start"] = time.monotonic()
        return original_build(*args, **kwargs)

    def timed_exec(*args, **kwargs):
        nonlocal inside_eval
        result["eval_script_start"] = time.time()
        start = time.monotonic()
        inside_eval = True
        try:
            value = original_exec(*args, **kwargs)
            result["timed_out"] = value[1]
            result["harness_runtime_seconds"] = value[2]
            return value
        finally:
            inside_eval = False
            result["eval_script_end"] = time.time()
            result["eval_script_monotonic_seconds"] = time.monotonic() - start

    def observed_exec_start(*args, **kwargs):
        value = original_api_exec_start(*args, **kwargs)
        if inside_eval and kwargs.get("stream"):
            return observed_test_stream(value, result, START_TEST_OUTPUT.encode(), END_TEST_OUTPUT.encode())
        return value

    ContainerCollection.create = limited_create
    official.build_container = timed_build
    official.exec_run_with_timeout = timed_exec
    client.api.exec_start = observed_exec_start
    try:
        returned = official.run_instance(spec, pred, False, False, client, request["run_id"], request.get("timeout", 1800))
        test_path, report_path = log_dir / LOG_TEST_OUTPUT, log_dir / LOG_REPORT
        log_path = log_dir / LOG_INSTANCE
        output = test_path.read_text(errors="replace") if test_path.exists() else ""
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        executed = bool(result["exec_start"] is not None and result["exec_end"] is not None
                        and test_path.exists() and test_path.stat().st_mtime >= result["exec_start"] - 1
                        and output.strip())
        result.update(tests_executed=executed, test_output_tail=output[-6000:], harness_log_tail=log[-3000:])
        result["execution_boundary_source"] = "Docker stdout receipt of official START_TEST_OUTPUT/END_TEST_OUTPUT markers; eval.sh setup is included in startup, not test execution"
        if executed and returned and report_path.exists() and report_path.stat().st_mtime >= result["exec_start"] - 1:
            report = json.loads(report_path.read_text())
            result.update(status="EVALUATED", resolved=bool(report[issue["instance_id"]]["resolved"]),
                          report_path=str(report_path.resolve()), report_mtime=report_path.stat().st_mtime,
                          test_output_mtime=test_path.stat().st_mtime, report=report)
        elif APPLY_PATCH_FAIL in log:
            result["status"] = "PATCH_APPLY_FAILED"
        else:
            result["error"] = "No verified fresh evaluation report (or official harness error/timeout)"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        ContainerCollection.create = original_create
        official.build_container = original_build
        official.exec_run_with_timeout = original_exec
        client.api.exec_start = original_api_exec_start
        client.close()
        result["response_ready"] = time.time()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.result.exists():
        raise FileExistsError(args.result)
    try:
        result = evaluate(json.loads(args.request.read_text()), args.prepare_only)
    except Exception:
        result = {"status": "ERROR", "resolved": False, "tests_executed": False, "error": traceback.format_exc()}
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result.get(k) for k in ("instance_id", "status", "tests_executed", "resolved")}))
    if result["status"] == "ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
