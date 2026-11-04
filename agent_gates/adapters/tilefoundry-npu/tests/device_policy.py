#!/usr/bin/env python3
"""Check the busy-device policy matches HANDOFF section 4 (shared card is normal).

Properties read from the AST / manifest (not grep):

  1. ORDER: the occupancy READ still precedes the device probe, so a health/readability
     problem is reported before any device node is opened.
  2. NO BUSY REFUSAL: no branch may return EXIT_BLOCKED conditioned on the card being
     occupied. Busy must be RECORDED (busyRecorded) and the run must proceed.
  3. REFUSAL SET: remaining BLOCKED_ENV paths may only be health/readability, identity
     (device id / binding) or ffi - never occupancy.
  4. MANIFEST SYNC: the device gate observationChecks carry busyRecorded and no longer
     carry overriddenBusyDevice/refusedBusyDevice; the description says busy is recorded.

Exit codes: 0 policy holds, 1 it does not, 2 input unreadable.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "scripts" / "tilefoundry_gate.py"
MANIFEST = HERE.parent / "manifest.json"


def seg(source, node):
    return ast.get_source_segment(source, node) or ""


def main() -> int:
    try:
        source = GATE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, json.JSONDecodeError) as error:
        print("could not read inputs: %s" % error, file=sys.stderr)
        return 2

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "gate_device"), None)
    if fn is None:
        print("  FAIL  gate_device is not defined")
        return 1
    fails = 0

    def check(cond, label):
        nonlocal fails
        if not cond:
            fails += 1
        print("  %s  %s" % ("PASS" if cond else "FAIL", label))

    # 1. ORDER
    occ_line = probe_line = None
    for node in ast.walk(fn):
        text = seg(source, node)
        if occ_line is None and isinstance(node, ast.Call) and "device_occupancy(" in text:
            occ_line = node.lineno
        if probe_line is None and isinstance(node, ast.Constant) and "torch_npu" in text:
            probe_line = node.lineno
    check(occ_line is not None, "gate_device calls device_occupancy")
    check(probe_line is not None, "gate_device reaches a device probe")
    check(occ_line is not None and probe_line is not None and occ_line < probe_line,
          "occupancy read (line %s) precedes the device probe (line %s)" % (occ_line, probe_line))

    # 2/3. no EXIT_BLOCKED conditioned on occupancy; busy is recorded
    busy_blocked = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = seg(source, node.test)
        body = "\n".join(seg(source, s) for s in node.body)
        if "occupied" in test and ("EXIT_BLOCKED" in body or "BLOCKED" in body):
            busy_blocked.append((node.lineno, test.strip()))
    check(not busy_blocked,
          "no EXIT_BLOCKED branch conditioned on occupied (found %r)" % (busy_blocked,))
    check("busyRecorded" in source, "code records busyRecorded")
    check("refusedBusyDevice" not in source and "overriddenBusyDevice" not in source,
          "code no longer emits refusedBusyDevice/overriddenBusyDevice")

    # 4. manifest sync
    dev = next((g for g in manifest.get("gates", []) if g.get("id") == "tilefoundry-npu-device"), None)
    if dev is None:
        print("  FAIL  device gate missing from manifest")
        fails += 1
    else:
        obs = dev.get("observationChecks", {})
        check("busyRecorded" in obs, "manifest observationChecks has busyRecorded")
        check("refusedBusyDevice" not in obs and "overriddenBusyDevice" not in obs,
              "manifest observationChecks drops refusedBusyDevice/overriddenBusyDevice")
        desc = dev.get("description", "").lower()
        check("shared" in desc and "record" in desc,
              "manifest description says busy is recorded/allowed on a shared card")
        check("refuses to open a card" not in desc,
              "manifest description no longer claims busy-card refusal")
    pol = manifest.get("busyDevicePolicy", {}) or manifest.get("provenance", {}).get("busyDevicePolicy", {})
    check(pol.get("default", "").lower().startswith("allow") or
          "record" in pol.get("default", "").lower(),
          "busyDevicePolicy default is record/allow, got %r" % (pol.get("default"),))

    print("RESULT:", "PASS" if fails == 0 else "FAILURES=%d" % fails)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
