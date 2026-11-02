#!/usr/bin/env python3
"""Minimal real device work: allocate, compute, synchronize, report.

Scope
-----

This is the smallest thing that proves a card is actually usable: import the
backend, find a device, move tensors onto it, run elementwise arithmetic,
synchronize, and compare against a CPU reference. It is NOT a kernel benchmark,
it does NOT compile TileLang, and it does NOT need ForgeKernel installed.

Why a separate file
-------------------

It runs as a CHILD of the gate, immediately inside the device window, so the
lease is held around real device work rather than around a sleep. Keeping it
separate makes the device work reviewable on its own, and lets it be run and
tested directly without a manifest, an engine or a lease.

Exit codes
----------

  0  the device ran the work and the numbers matched
  1  the device was usable but the work or the numbers failed (a real defect)
  3  no usable device environment: the backend would not import, or no device
     is present. This maps to BLOCKED_ENV upstream: nothing was measured, so
     calling it a failed test would be a lie.

What it does NOT claim
----------------------

It does not identify which PHYSICAL board ran. Logical device 0 is not
necessarily card 0, and the visible-device echo is a value this process could
have set itself. What is proved is that the one device the operator made
visible ran real arithmetic with correct numbers. Board identity is recorded
only where the platform reports it, and is never gated.
"""
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

EXIT_UNAVAILABLE = 3

# The RMSNorm workload this Adapter exists to serve is BF16, so both a strict
# FP32 pass (bit-level reproduction of the CPU result) and a BF16 pass are run.
# The BF16 bound is the format's own rounding floor: 2**-7 = 0.0078125, the
# largest relative error a single rounding of a normal BF16 value can produce.
ELEMENTS = 4096
FP32_TOLERANCE = 1e-6
BF16_TOLERANCE = 2.0 ** -7


def failure(code, message):
    """Report why the device could not be used, then exit with the given code."""
    print(message, file=sys.stderr)
    raise SystemExit(code)


def relative_error(actual, reference):
    """Largest relative error, ignoring entries whose reference is zero."""
    import torch
    denominator = reference.abs()
    mask = denominator > 0
    if not bool(mask.any()):
        return 0.0
    difference = (actual - reference).abs()[mask] / denominator[mask]
    return float(difference.max().item())


def main():
    started = time.monotonic()
    report = {
        "npuAvailable": False,
        "deviceCount": 0,
        "tensorMovedToDevice": 0,
        "opsRun": [],
        "synchronizeOk": False,
        "synchronizeError": None,
        "deviceMemoryAllocated": 0,
        "numericalOk": False,
        "fp32MaxRelativeError": None,
        "bf16MaxRelativeError": None,
        "torchVersion": None,
        "torchNpuVersion": None,
        "authorizedDeviceId": os.environ.get("FK_ASCEND_DEVICE_ID"),
        "visibleDevices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }

    # ── Phase 1: everything that does not run a kernel, BEFORE the lease ──
    #
    # Importing torch_npu costs around ten seconds and creating the device
    # context a couple more, and neither runs a kernel. Doing them inside the
    # window would make the window almost entirely initialization - exactly the
    # "holding a card while not using it" the window exists to prevent - and
    # would let a receipt report a 20-second "device time" that measured an
    # import. The warmup therefore happens first, and its cost is published so
    # the reader can see it was not counted as device work.
    phase = time.monotonic()
    try:
        import torch
    except BaseException as error:
        failure(EXIT_UNAVAILABLE, "cannot import torch: %s: %s"
                % (type(error).__name__, error))
    report["torchVersion"] = torch.__version__
    report["importSeconds"] = round(time.monotonic() - phase, 3)

    phase = time.monotonic()
    try:
        import torch_npu  # noqa: F401  (registers the "npu" backend)
    except BaseException as error:
        failure(EXIT_UNAVAILABLE, "cannot import torch_npu: %s: %s"
                % (type(error).__name__, error))
    report["torchNpuVersion"] = getattr(torch_npu, "__version__", "")
    report["backendSeconds"] = round(time.monotonic() - phase, 3)

    phase = time.monotonic()
    try:
        available = bool(torch.npu.is_available())
        count = int(torch.npu.device_count())
    except BaseException as error:
        failure(EXIT_UNAVAILABLE, "probing the NPU backend failed: %s: %s"
                % (type(error).__name__, error))
    report["npuAvailable"] = available
    report["deviceCount"] = count
    if not available:
        failure(EXIT_UNAVAILABLE, "torch.npu.is_available() is False: no Ascend device")
    if count < 1:
        failure(EXIT_UNAVAILABLE, "torch.npu.device_count() is %d: no Ascend device" % count)
    report["probeSeconds"] = round(time.monotonic() - phase, 3)

    # Creating the context needs the device but executes no kernel of ours. The
    # tiny allocation is what forces the context to exist, so the window below
    # opens on a device that is already usable.
    phase = time.monotonic()
    try:
        torch.npu.set_device(0)
        torch.zeros(1, device="npu")
        torch.npu.synchronize()
    except BaseException as error:
        failure(EXIT_UNAVAILABLE, "cannot initialize the device context: %s: %s"
                % (type(error).__name__, error))
    report["contextSeconds"] = round(time.monotonic() - phase, 3)
    report["warmupSeconds"] = round(time.monotonic() - started, 3)

    # ── Phase 2: the device window, real work only ──
    #
    # The window helper is imported from the Gate rather than reimplemented, so
    # there is exactly one lease implementation in this Pack. The lease PATH is
    # still given by the engine and never derived here.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fk_gate

    try:
        with fk_gate.device_window() as window:
            report["deviceWindow"] = window
            work_started = time.monotonic()
            try:
                report.update(device_work(torch))
            except BaseException as error:
                traceback.print_exc()
                report["opsError"] = "%s: %s" % (type(error).__name__, error)
            report["workSeconds"] = round(time.monotonic() - work_started, 3)
        # Read releasedAt AFTER the block: inside it the window is still open by
        # definition, so checking there would always report "not released".
        window_released = window.get("releasedAt") is not None
    except fk_gate.DeviceWindowBusy as error:
        # The lease did not open. That is a resource refusal: nothing ran, so it
        # must not be reported as a failed test.
        print(str(error), file=sys.stderr)
        raise SystemExit(EXIT_UNAVAILABLE)

    # The window must be dominated by work, not by setup. Publishing both lets a
    # reader check that claim instead of trusting it.
    report["windowHeldSeconds"] = window.get("heldSeconds")
    report["windowOpened"] = True
    report["windowReleased"] = window_released
    report["occupancyRecorded"] = (window.get("occupancy") in {"free", "unknown"}
                                   or bool(re.fullmatch(r"shared\([0-9]+\)",
                                                         window.get("occupancy") or "")))
    report["tensorOnDevice"] = report["tensorMovedToDevice"] > 0
    report["deviceMemoryPositive"] = report["deviceMemoryAllocated"] > 0
    report["totalSeconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, sort_keys=True, default=str))
    ok = (report["npuAvailable"] and report["tensorOnDevice"]
          and report["synchronizeOk"] and report["numericalOk"]
          and report["deviceMemoryPositive"] and report["windowOpened"]
          and report["windowReleased"] and report["occupancyRecorded"]
          and "opsError" not in report)
    return 0 if ok else 1


def device_work(torch):
    """Allocate, compute, synchronize, verify - the whole body of the window.

    Everything here needs the card and runs a kernel. Nothing here is needed to
    decide whether the card exists, which is why the availability probe and the
    context warmup ran outside the window.
    """
    result = {"tensorMovedToDevice": 0, "opsRun": [], "numericalOk": False,
              "synchronizeOk": False, "deviceMemoryAllocated": 0}
    with torch.no_grad():
        left = torch.randn(ELEMENTS, device="npu", dtype=torch.float32)
        right = torch.randn(ELEMENTS, device="npu", dtype=torch.float32)
        result["tensorMovedToDevice"] += 2
        torch.npu.synchronize()

        combined = (left + right) * 2.0
        result["opsRun"].append("add")
        result["opsRun"].append("mul")
        reduced = combined.sum()
        result["opsRun"].append("sum")
        result["tensorMovedToDevice"] += 1
        torch.npu.synchronize()

        # The reference is computed the same way on the CPU: for FP32 these
        # operations are exactly reproducible, so a tight bound is honest
        # rather than generous.
        expected = (left.cpu() + right.cpu()) * 2.0
        result["fp32MaxRelativeError"] = relative_error(combined.cpu(), expected)
        expected_sum = expected.sum()
        # A reduction across 4096 elements accumulates rounding in a different
        # order on either side, so it is compared relatively with a bound
        # appropriate to that rather than bit for bit.
        scale = float(expected_sum.abs().item()) or 1.0
        result["reductionRelativeError"] = (
            abs(float(reduced.cpu().item()) - float(expected_sum.item())) / scale)

        # BF16 is the dtype the real workload uses, so it is exercised too.
        bf16_source = torch.randn(ELEMENTS, device="npu", dtype=torch.bfloat16)
        result["tensorMovedToDevice"] += 1
        bf16_result = bf16_source * 2.0
        bf16_expected = (bf16_source.cpu().float() * 2.0).to(torch.bfloat16)
        result["bf16MaxRelativeError"] = relative_error(
            bf16_result.cpu().float(), bf16_expected.float())
        torch.npu.synchronize()

    result["synchronizeOk"] = True
    try:
        result["deviceMemoryAllocated"] = int(torch.npu.max_memory_allocated())
    except BaseException as error:
        result["memoryError"] = "%s: %s" % (type(error).__name__, error)
    result["numericalOk"] = bool(
        result["fp32MaxRelativeError"] is not None
        and result["fp32MaxRelativeError"] <= FP32_TOLERANCE
        and result["bf16MaxRelativeError"] is not None
        and result["bf16MaxRelativeError"] <= BF16_TOLERANCE
        and result.get("reductionRelativeError", 1.0) <= BF16_TOLERANCE)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
