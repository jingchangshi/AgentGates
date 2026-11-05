"""H harness tests: workload/scope, launch-after failure, phases/raw/artifacts, FINISHED.

Fake drivers + owned temp dirs only; no card is touched.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
SCRIPTS = PACK / "scripts"
sys.path.insert(0, str(SCRIPTS))

import device_harness as H  # noqa: E402

H.register_workload(H.WorkloadSpec(
    workload_id="vadd1row", canonical_recipe="tilefoundry-npu.device",
    x_mode="X_VADD_1ROW", output_spec={"dtype": "f32", "shape": [4]},
    executor="canonical.x.vadd1row"))


def expect_code(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


def make_request(scope=H.SELECT_ONE, ids=(0,), allow=True):
    X = {"workload_id": "vadd1row", "recipe_id": "tilefoundry-npu.device",
         "allow_device": allow, "artifacts": {"coverage": {"scope": scope,
                                                           "physical_ids": list(ids)}}}
    return H.make_run_request(
        {"case_id": "case-vadd", "run_id": "run-vadd-1"},
        X, {"x": {"bytes": b"\x00" * 16, "dtype": "f32", "shape": [4]},
            "eps_f32_bits": 0x358637bd},
        {"protocol": "frozen-v1"}, {"adapter_commit": "6c7b0e6", "manifest_hash": "m" * 16})


def run_context(tmp):
    return H.RunContext("run-vadd-1", Path(tmp))


def fake_launch(pid, artifacts, abi):
    return {"synced": True, "output": b"\x01" * 16, "dtype": "f32", "shape": [4]}


def fill_observations(identity):
    return {key: {"status": H.OBSERVED, "path": "/tmp/%s.bin" % key, "sha256": "x" * 64}
            for key in H.ARTIFACT_KEYS}


def positive_evidence(tmp, scope=H.SELECT_ONE, ids=(0,)):
    ctx = run_context(tmp)
    selected = H.SelectedDevice((0, 1), tuple(range(len(ids))), tuple(ids), scope, "test")
    outputs = H.execute_attempt(selected, {}, {}, {"x": {"bytes": b"\x00" * 16,
                                                         "dtype": "f32", "shape": [4]},
                                                   "eps_f32_bits": 0x358637bd},
                                ctx, launch_fn=fake_launch)
    evidence = H.collect_evidence(ctx, ctx.phases, outputs,
                                  artifact_identity=fill_observations(None))
    return ctx, evidence


def test_missing_workload_is_named():
    expect_code("UNSUPPORTED_WORKLOAD", lambda: H.get_workload("not-registered"))


def test_non_canonical_generic_entry_is_refused():
    req = make_request()
    expect_code("NON_CANONICAL_DEVICE_ENTRY",
                lambda: H.dispatch_via_adapter(req, "generic.torch.add"))


def test_cpu_verify_defaults_to_skip_device_and_device_needs_switch():
    assert H.default_cpu_verify_args() == ["--side", "recipe", "--skip-device"]
    req = make_request(allow=False)
    expect_code("DEVICE_NOT_REQUESTED", lambda: H.dispatch_via_adapter(req, "tilefoundry-npu.device"))


def test_positive_select_one_run_is_passed_with_raw_inputs_and_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        req, ctx, evidence = None, None, None
        req = make_request()
        ctx, evidence = positive_evidence(tmp)
        result = H.evaluate_run(req, evidence, {"status": "PASS"})
        assert result["finished"] is True and result["passed"] is True, result
        assert any(item.get("name") == "eps" and item.get("f32_bits") == 0x358637bd
                   for item in evidence["inputs"])
        assert all(evidence["artifacts"][key]["status"] == H.OBSERVED for key in H.ARTIFACT_KEYS)
        assert evidence["outputs"][0]["nbytes"] == 16
        assert result["service_receipt"] is None and result["local_only"] is True


def test_all_requested_scope_requires_every_device_output():
    with tempfile.TemporaryDirectory() as tmp:
        req = make_request(scope=H.ALL_REQUESTED, ids=(0, 1))
        ctx, evidence = positive_evidence(tmp, scope=H.ALL_REQUESTED, ids=(0,))
        result = H.evaluate_run(req, evidence, {"status": "PASS"})
        assert result["finished"] is True and result["passed"] is False
        assert result["reason_code"] == "DEVICE_COVERAGE_MISMATCH", result


def test_launch_failure_after_start_does_not_fail_over_and_is_not_pass():
    with tempfile.TemporaryDirectory() as tmp:
        req = make_request()
        ctx = run_context(tmp)
        selected = H.SelectedDevice((0, 1), (0,), (0,), H.SELECT_ONE, "test")
        calls = []

        def bad_launch(pid, artifacts, abi):
            calls.append(pid)
            raise RuntimeError("kernel launch failed in fake driver")

        expect_code("LAUNCH_FAILED", lambda: H.execute_attempt(
            selected, {}, {}, {"x": {"bytes": b"\x00" * 16}, "eps_f32_bits": 1},
            ctx, launch_fn=bad_launch))
        assert calls == [0], "launch started once; no failover to another card"
        evidence = H.collect_evidence(ctx, ctx.phases, [],
                                      artifact_identity=fill_observations(None))
        result = H.evaluate_run(req, evidence, {"status": "PASS"})
        assert result["finished"] is True and result["passed"] is False
        assert result["reason_code"] == "LAUNCH_FAILED", result


def test_finished_without_output_is_not_pass():
    with tempfile.TemporaryDirectory() as tmp:
        req = make_request()
        ctx = run_context(tmp)
        for name in ("launch", "sync", "readback"):
            ctx.record(name, "PASS")
        evidence = H.collect_evidence(ctx, ctx.phases, [],
                                      artifact_identity=fill_observations(None))
        result = H.evaluate_run(req, evidence, {"status": "PASS"})
        assert result["finished"] is True and result["passed"] is False
        assert result["reason_code"] == "OUTPUT_MISSING", result


def test_missing_artifact_identity_is_not_pass():
    with tempfile.TemporaryDirectory() as tmp:
        req = make_request()
        ctx, evidence = positive_evidence(tmp)
        for key in H.ARTIFACT_KEYS:
            evidence["artifacts"][key]["status"] = H.UNOBSERVED
        result = H.evaluate_run(req, evidence, {"status": "PASS"})
        assert result["finished"] is True and result["passed"] is False
        assert result["reason_code"] == "ARTIFACT_IDENTITY_UNOBSERVED", result


def test_shared_busy_response_is_selected_not_switched():
    probes = {
        0: {"responsive": True, "classification": "RESPONSIVE", "timed_out": False,
            "unreaped": False, "rc": 0, "wchan": None, "occupancy": "shared(2)"}}
    attempts, selected = H.select_device(
        H.DeviceRequest((0, 1), H.SELECT_ONE, 0, {"probe": lambda pid: probes[pid],
                                                  "backoff_seconds": 30,
                                                  "sleeper": lambda _: None}),
        {"visible": [0, 1]}, {"probe": lambda pid: probes[pid], "backoff_seconds": 30,
                              "sleeper": lambda _: None})
    assert len(attempts) == 1 and selected.physical_ids == (0,)


if __name__ == "__main__":
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", _name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
