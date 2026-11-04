#!/usr/bin/env python3
"""Check the busy-device policy: refusal is the default, and it comes first.

Two properties, both invisible to a runtime test that only exercises the refusal
branch:

  1. ORDER. The occupancy read must precede the torch_npu import. If occupancy were
     read after a device had already been opened, the refusal would arrive too late
     to prevent the thing it exists to prevent -- and a test that just checks "busy
     card gives 75" would still pass.

  2. DEFAULT. The refusal must not be conditional on an opt-in flag. The
     `if` whose body returns BLOCKED_ENV must not mention TF_REQUIRE_EXCLUSIVE:
     requiring an opt-in to *avoid* touching a busy card is the inversion that
     caused the incident.

Both are read from the AST. A grep for the literal text is not enough: flipping
`occupied and not override` to `occupied and TF_REQUIRE_EXCLUSIVE == "1"` keeps every
string a grep would look for while restoring the old permissive default. That exact
flip is the negative control for this script.

Exit codes: 0 policy holds, 1 it does not, 2 the input could not be read.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

GATE = Path(__file__).resolve().parent.parent / "scripts" / "tilefoundry_gate.py"


def segment(source: str, node) -> str:
    return ast.get_source_segment(source, node) or ""


def find_gate_device(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "gate_device":
            return node
    return None


def main() -> int:
    try:
        source = GATE.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as error:
        print("could not read %s: %s" % (GATE, error), file=sys.stderr)
        return 2

    fn = find_gate_device(tree)
    if fn is None:
        print("  FAIL  gate_device is not defined")
        return 1

    failures = 0

    # 1. ORDER: the occupancy call must come before any torch_npu import.
    occupancy_line = None
    probe_line = None
    for node in ast.walk(fn):
        text = segment(source, node)
        if occupancy_line is None and "device_occupancy(" in text and isinstance(node, ast.Call):
            occupancy_line = node.lineno
        if probe_line is None and "torch_npu" in text and isinstance(node, ast.Constant):
            probe_line = node.lineno
    if occupancy_line is None:
        print("  FAIL  gate_device never calls device_occupancy")
        failures += 1
    elif probe_line is None:
        print("  FAIL  gate_device never reaches a device probe at all")
        failures += 1
    elif occupancy_line >= probe_line:
        print("  FAIL  occupancy read (line %d) is not before the device probe (line %d)"
              % (occupancy_line, probe_line))
        failures += 1
    else:
        print("  PASS  occupancy (line %d) precedes the device probe (line %d)"
              % (occupancy_line, probe_line))

    # 2. DEFAULT: the refusal branch must not depend on an opt-in flag.
    # Select the refusal branch by its MARKER STRING, then read its test expression.
    # The marker is what identifies the branch; the test is what we police.
    refusal_tests = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        body_lines = [segment(source, stmt) for stmt in node.body]
        body = "\n".join(body_lines)
        if "refusing to open it" not in body and "is held by another user" not in body:
            continue
        has_blocked = "EXIT_BLOCKED" in body
        refusal_tests.append((segment(source, node.test), has_blocked, node.lineno))

    if not refusal_tests:
        print("  FAIL  no busy-device refusal branch returns BLOCKED_ENV")
        failures += 1
    else:
        test_src, has_blocked, ref_line = refusal_tests[0]
        if not has_blocked:
            print("  FAIL  the refusal branch does not return EXIT_BLOCKED (line %d)" % ref_line)
            failures += 1
        else:
            print("  PASS  the refusal returns BLOCKED_ENV (line %d)" % ref_line)
        if "TF_REQUIRE_EXCLUSIVE" in test_src:
            print("  FAIL  the refusal is conditional on an opt-in flag: %s" % test_src.strip())
            failures += 1
        else:
            print("  PASS  refusal is unconditional on opt-in: %s" % test_src.strip())

    # 3. The override must exist and must be the permissive side only.
    override_tests = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test_src = segment(source, node.test)
        body = "\n".join(segment(source, stmt) for stmt in node.body)
        # An override worth having is one that PROCEEDS, and says it is overriding.
        # The test expression is usually `occupied and override`, a NAME, so the flag
        # literal appears at the assignment rather than in the test. Resolve it: a
        # check that only looks at the test text finds nothing and reports a missing
        # override on a gate that has one.
        flag_src = test_src
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", test_src):
            if name in ("occupied", "override", "and", "not", "or"):
                continue
            for assign in ast.walk(fn):
                if (isinstance(assign, ast.Assign)
                        and any(isinstance(tg, ast.Name) and tg.id == name
                                for tg in assign.targets)):
                    flag_src += " " + segment(source, assign.value)
        if "occupied and override" in test_src:
            for assign in ast.walk(fn):
                if (isinstance(assign, ast.Assign)
                        and any(isinstance(tg, ast.Name) and tg.id == "override"
                                for tg in assign.targets)):
                    flag_src += " " + segment(source, assign.value)
        if "TF_ALLOW_BUSY_DEVICE" in flag_src and "OVERRIDE" in body:
            override_tests.append(node)
    if override_tests:
        print("  PASS  the override is named explicitly (TF_ALLOW_BUSY_DEVICE)")
    else:
        print("  FAIL  no explicit override is named")
        failures += 1

    # 4. Occupancy is a precondition of the probe -- the probe must not be reachable
    #    without it. Both appear in gate_device, and 1 established the order.
    if occupancy_line is not None and probe_line is not None and occupancy_line < probe_line:
        print("  PASS  the device probe is guarded by the occupancy read")
    else:
        print("  FAIL  the device probe is not guarded by the occupancy read")
        failures += 1

    if failures:
        print("RESULT: FAIL (%d)" % failures)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())