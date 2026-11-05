#!/usr/bin/env python3
"""H4: one real TL-compiled vadd-one-64 run through the H harness canonical entry.

The script registers the workload, selects one card via the approved pre-launch
probe, compiles the real TL kernel, executes, synchronizes, reads back raw bytes,
records phases/inputs/outputs and four identities, and prints a local result.
It does not use a generic torch add and does not claim service receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_harness as H


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--visible", default=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0"))
    parser.add_argument("--evidence-root", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    visible = [int(x) for x in args.visible.split(",") if x.strip()]
    if not (0 <= args.device_index < len(visible)):
        raise SystemExit("DEVICE_INDEX_NOT_VISIBLE")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(x) for x in visible)
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    if os.geteuid() != 0:
        print(json.dumps({"finished": True, "passed": False, "verdict": "BLOCKED_ENV",
                          "reason_code": "DEVICE_ACCESS_REQUIRES_ROOT",
                          "detail": "Ascend950PR device access must use root inside the canonical "
                                    "docker container; non-root can cause exclusive behavior"}, sort_keys=True))
        return 75

    from tl_vadd_one import compile_vadd_one, readback_vadd_one
    H.register_workload(H.WorkloadSpec(
        workload_id="tl-vadd-one-64", canonical_recipe="tilefoundry-npu.device",
        x_mode="TL_SOURCE_RECOMPILE", output_spec={"dtype": "float32", "shape": [64]},
        executor="canonical.tl.vadd_one"))
    x = __import__("numpy").arange(64, dtype="float32")
    input_bytes = x.tobytes()
    request = H.make_run_request(
        {"run_id": "h4-tl-vadd-%d" % os.getpid(), "case_id": "h4-vadd-one-64",
         "profile_ref": {"kind": "tfnpu-v2"}, "source_ref": {"file": "scripts/tl_vadd_one.py"}},
        {"workload_id": "tl-vadd-one-64", "recipe_id": "tilefoundry-npu.device",
         "allow_device": True, "abi": {"args": ["x", "out"], "output_index": 1},
         "artifacts": {"coverage": {"scope": H.SELECT_ONE,
                                    "physical_ids": [visible[args.device_index]]}}},
        {"x": {"bytes": input_bytes, "dtype": "float32", "shape": [64]},
         "eps_f32_bits": None},
        {"protocol": "tl-vadd-one-f32"}, {"adapter_commit": "6c7b0e6", "manifest_hash": "h4"})
    H.dispatch_via_adapter(request, "tilefoundry-npu.device")

    # preflight selection uses the approved driver probe code path, before launch.
    from verify_environment import probe_card
    device_request = H.DeviceRequest(tuple(visible), H.SELECT_ONE, args.device_index,
                                     {"probe": lambda physical: probe_card(physical, 20.0),
                                      "backoff_seconds": 30.0})
    try:
        attempts, selected = H.select_device(device_request, {"visible": visible},
                                                  {"probe": lambda physical: probe_card(physical, 20.0),
                                                   "backoff_seconds": 30.0})
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"finished": True, "passed": False, "verdict": "NOT_RUN",
                          "reason_code": getattr(error, "code", "DEVICE_PROBE_FAILED"),
                          "detail": str(error)}, sort_keys=True))
        return 75
    if selected is None:
        print(json.dumps({"finished": True, "passed": False, "verdict": "NOT_RUN",
                          "reason_code": "DEVICE_UNRESPONSIVE",
                          "attempts": [a.__dict__ if hasattr(a, "__dict__") else a for a in attempts]},
                         sort_keys=True))
        return 75
    run_root = Path(tempfile.mkdtemp(prefix="h4-tl-vadd-"))
    if args.evidence_root:
        run_root = Path(args.evidence_root) / request.run_id
        run_root.mkdir(parents=True, exist_ok=True)
    ctx = H.RunContext(request.run_id, run_root)
    ctx.attempts = attempts
    ctx.record("preflight", "PASS")
    ctx.record("compile", "START")
    kernel = compile_vadd_one(64)
    ctx.record("compile", "PASS")

    def launch(physical_id, artifacts, abi):
        output, shape, dtype = readback_vadd_one(kernel, input_bytes)
        return {"synced": True, "output": output, "dtype": dtype, "shape": list(shape)}

    outputs = H.execute_attempt(selected, {}, {}, {"x": {"bytes": input_bytes, "dtype": "float32",
                                                          "shape": [64]}},
                                ctx, launch_fn=launch)
    source_path = Path(__file__).resolve().parent / "tl_vadd_one.py"
    cce_bytes = kernel.kernel_source.encode()
    artifacts = {
        "source": {"status": H.OBSERVED, "path": str(source_path),
                   "sha256": sha(source_path.read_bytes())},
        "cce": {"status": H.OBSERVED, "sha256": sha(cce_bytes),
                "reason": "TileLang JITKernel.kernel_source (Ascend C source)"},
        "binary": {"status": H.UNOBSERVED,
                   "reason": "JITKernel.artifact is None; no binary path exposed by this TL build"},
        "load": {"status": H.OBSERVED, "sha256": sha(str(type(kernel)).encode()),
                 "reason": "compiled JITKernel callable in this process"},
    }
    evidence = H.collect_evidence(ctx, ctx.phases, outputs, artifact_identity=artifacts)
    out_bytes = outputs[0]["path"]
    got = __import__("numpy").fromfile(out_bytes, dtype="float32")
    expected = x + 1.0
    mismatch = int((got != expected).sum())
    t_report = {"status": "PASS" if mismatch == 0 else "FAIL", "mismatch_count": mismatch,
                "n_elements": int(got.size), "atol": 0.0, "rtol": 0.0}
    result = H.evaluate_run(request, evidence, t_report, allow_unobserved=("binary",))
    result["t_report"] = t_report
    result["nonroot_override"] = False
    result["device_access_policy"] = "ROOT_ONLY_DOCKER"
    result["input_sha256"] = sha(input_bytes)
    result["output_sha256"] = outputs[0]["sha256"]
    result["attempts"] = [a.__dict__ if hasattr(a, "__dict__") else a for a in attempts]
    result["phases"] = [p.as_dict() if hasattr(p, "as_dict") else p for p in ctx.phases]
    result["identities"] = artifacts
    if args.evidence_root:
        (run_root / "h4_result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
