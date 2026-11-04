#!/usr/bin/env python3
"""Prove the interpreter-identity check refuses a non-interpreter like /bin/true.

This is the negative test for the FALSE PASS the Architect reproduced: with
IN_CONTAINER_PYTHON=/bin/true the Gate used to exit 0 with every field "true" and
nothing ran. The identity check is what closes that, so it must be shown to REFUSE,
not merely to exist.

Runs without a container on purpose: the thing under test is the decision, and a
test for "a bad interpreter is refused" must be runnable where a bad interpreter is
harmless.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

GATE = Path(__file__).resolve().parent.parent / "scripts" / "tilefoundry_gate.py"
spec = importlib.util.spec_from_file_location("tfgate", GATE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

failures = 0


def check(label, ok):
    global failures
    print(("  PASS  " if ok else "  FAIL  ") + label)
    if not ok:
        failures += 1


# A. The REAL /bin/true binary, given the identity script, must produce nothing the
#    parser can read -- i.e. it cannot answer the challenge at all.
run = subprocess.run(["/bin/true", "-c", gate.IDENTITY_SCRIPT],
                     capture_output=True, text=True, check=False)
info = gate._last_json_line(run.stdout)
check("real /bin/true yields no identity JSON (exit %d, stdout=%r)"
      % (run.returncode, run.stdout[:40]), info is None)

# B. With no identity, the verdict is refusal -- and the reason names the cause.
ok, detail = gate.identity_verdict(None, "/bin/true")
check("/bin/true is REFUSED by identity_verdict", ok is False)
check("refusal names the non-interpreter cause", "true" in json.dumps(detail).lower())

# C. A plausible-looking identity with the WRONG challenge is still refused: the
#    challenge is derived, not a constant a stub could echo.
fake = {"executable": os.path.realpath(sys.executable), "version": "3.12",
        "challenge": "0" * 64}
ok, detail = gate.identity_verdict(fake, sys.executable)
check("wrong challenge is refused", ok is False)

# D. A correct identity is accepted (so the check is not just "always refuse").
import hashlib
real = {"executable": os.path.realpath(sys.executable),
        "version": "%d.%d" % sys.version_info[:2],
        "challenge": gate._EXPECTED_CHALLENGE}
ok, detail = gate.identity_verdict(real, sys.executable)
check("a genuine Python identity is accepted", ok is True)

# E. The expected challenge is DERIVED, not the literal sum. A stub that echoes the
#    sum cannot match it.
check("expected challenge is a sha256, not the raw sum",
      gate._EXPECTED_CHALLENGE == hashlib.sha256(
          (gate._CHALLENGE_SALT + str(sum(range(1000)))).encode()).hexdigest()
      and gate._EXPECTED_CHALLENGE != str(sum(range(1000))))

print()
if failures:
    print("RESULT: FAIL (%d)" % failures)
    sys.exit(1)
print("RESULT: PASS")
