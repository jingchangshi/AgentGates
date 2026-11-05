#!/usr/bin/env python3
"""H4 seam bad controls (simulated). No device and no TL compile here."""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import device_harness as H


def expect(code, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        got = getattr(exc, "code", None)
        if got != code:
            raise AssertionError("expected %s got %r: %s" % (code, got, exc))
        return exc
    raise AssertionError("expected %s, nothing refused" % code)


def test_missing_workload():
    expect("UNSUPPORTED_WORKLOAD", lambda: H.make_run_request(
        {"run_id": "missing"}, {"workload_id": "does-not-exist", "allow_device": True},
        {}, {"protocol": "x"}, {"adapter_commit": "x", "manifest_hash": "x"}))


def test_generic_add_impersonation_refused():
    expect("NON_CANONICAL_DEVICE_ENTRY", lambda: H.register_workload(H.WorkloadSpec(
        "generic-fake", "tilefoundry-npu.device", "TL", {}, "generic_add")))


def test_launch_failure_has_no_failover():
    selected = H.SelectedDevice((0,), (0,), (0,), H.SELECT_ONE, "fake")
    ctx = H.RunContext("h4-simulated-launch", Path(tempfile.mkdtemp(prefix="h4seam-")))
    calls = []

    def launch(physical_id, artifacts, abi):
        calls.append(physical_id)
        raise RuntimeError("simulated launch failure")

    expect("LAUNCH_FAILED", lambda: H.execute_attempt(selected, {}, {}, {}, ctx, launch_fn=launch))
    assert calls == [0], calls
    return {"simulated": True, "case": "launch_failure", "code": "LAUNCH_FAILED", "calls": calls}


def test_missing_output():
    selected = H.SelectedDevice((0,), (0,), (0,), H.SELECT_ONE, "fake")
    ctx = H.RunContext("h4-simulated-output", Path(tempfile.mkdtemp(prefix="h4seam-")))
    expect("OUTPUT_MISSING", lambda: H.execute_attempt(
        selected, {}, {}, {}, ctx, launch_fn=lambda *a: {"synced": True, "output": b""}))


def test_missing_identity_cannot_pass():
    H.register_workload(H.WorkloadSpec("tl-vadd-one-64", "tilefoundry-npu.device",
                                       "TL_SOURCE_RECOMPILE", {"dtype": "float32", "shape": [64]},
                                       "canonical.tl.vadd_one"))
    req = H.make_run_request({"run_id": "identity-sim"}, {"workload_id": "tl-vadd-one-64"},
                             {}, {"protocol": "x"}, {"adapter_commit": "x", "manifest_hash": "x"})
    evidence = {"manifest_path": "simulated", "phases": [
        {"name": "launch", "status": "PASS"}, {"name": "sync", "status": "PASS"},
        {"name": "readback", "status": "PASS"}], "outputs": [{"physical_id": 0}],
        "artifacts": {k: {"status": "UNOBSERVED"} for k in H.ARTIFACT_KEYS}}
    result = H.evaluate_run(req, evidence, {"status": "PASS"})
    assert result["passed"] is False and result["reason_code"] == "ARTIFACT_IDENTITY_UNOBSERVED", result
    return {"simulated": True, "case": "missing_identity", "result": result}


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                result = fn()
                print("SIMULATED PASS", name, result if result else "")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("SIMULATED FAIL", name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
