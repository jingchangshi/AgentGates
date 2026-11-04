#!/usr/bin/env python3
"""Prove a tool that exits 0 with no structured PASS report is a FAILURE.

Mirror of identity_check.py for the tool layer: the Gate must not accept "exit 0"
as a result. Container-free, because the decision is pure.
"""
from __future__ import annotations

import importlib.util
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


good = {"status": "PASS", "exit_code": 0}
bad_status = {"status": "FAIL", "exit_code": 0}
bad_exit = {"status": "PASS", "exit_code": 3}

check("exit 0 but NO report is a FAILURE", gate.tool_report_ok("regressions", 0, None) is False)
check("exit 0 with non-PASS status is a FAILURE", gate.tool_report_ok("regressions", 0, bad_status) is False)
check("PASS status with non-zero exit_code is a FAILURE", gate.tool_report_ok("regressions", 0, bad_exit) is False)
check("non-zero returncode is a FAILURE even with a PASS report",
      gate.tool_report_ok("regressions", 1, good) is False)
check("a genuine structured PASS is accepted", gate.tool_report_ok("regressions", 0, good) is True)

# cpu_equality: needs real cases with all checks true.
ce_bad = {"status": "PASS", "exit_code": 0, "cases": []}
ce_good = {"status": "PASS", "exit_code": 0,
           "cases": [{"status": "PASS", "checks": {"shape": True, "dtype": True, "bits": True}}]}
ce_falsebit = {"status": "PASS", "exit_code": 0,
               "cases": [{"status": "PASS", "checks": {"shape": True, "bits": False}}]}
check("cpu_equality with no cases is a FAILURE", gate.tool_report_ok("cpu_equality", 0, ce_bad) is False)
check("cpu_equality with a false check is a FAILURE", gate.tool_report_ok("cpu_equality", 0, ce_falsebit) is False)
check("cpu_equality with real passing cases is accepted", gate.tool_report_ok("cpu_equality", 0, ce_good) is True)

# check_copy_slice: the three structural checks must all be true.
ccs_bad = {"status": "PASS", "exit_code": 0, "checks": {"two_copies": True, "one_smem_buffer": False}}
ccs_good = {"status": "PASS", "exit_code": 0,
            "checks": {"two_copies": True, "one_smem_buffer": True, "stages_through_same_buffer": True}}
check("check_copy_slice with a false structural check is a FAILURE",
      gate.tool_report_ok("check_copy_slice", 0, ccs_bad) is False)
check("check_copy_slice with all structural checks true is accepted",
      gate.tool_report_ok("check_copy_slice", 0, ccs_good) is True)

# Frozen empty-checks: an empty checks dict must FAIL, not default-true via all().
ce_empty = {"status": "PASS", "exit_code": 0, "cases": [{"status": "PASS", "checks": {}}]}
check("cpu_equality with EMPTY checks is a FAILURE",
      gate.tool_report_ok("cpu_equality", 0, ce_empty) is False)
ce_missing = {"status": "PASS", "exit_code": 0,
              "cases": [{"status": "PASS", "checks": {"shape": True, "dtype": True}}]}
check("cpu_equality missing a mandatory check (bits) is a FAILURE",
      gate.tool_report_ok("cpu_equality", 0, ce_missing) is False)

# Staleness: a report older than the run start is the PREVIOUS run's file.
import time as _time
_now = _time.time()
check("report newer than run start is fresh", gate.report_is_fresh(_now, _now - 1) is True)
check("report older than run start is STALE (rejected as a skipped writer)",
      gate.report_is_fresh(_now - 100, _now) is False)

# SKIPPED_WRITER_E2E: exits 0 and writes NO report. In a fresh run directory there
# is no file to read, so it must be rejected -- the staleness scenario in miniature.
import subprocess as _sp
import tempfile as _tf
import os as _os
with _tf.TemporaryDirectory() as _td:
    _out = _os.path.join(_td, "skipped.json")
    _r = _sp.run(["/bin/true", "--out", _out], capture_output=True, text=True)
    _wrote = _os.path.exists(_out)
    _rep = None
    if _wrote:
        import json as _json
        _rep = _json.load(open(_out))
    check("SKIPPED_WRITER_E2E: a writer that exits 0 and writes nothing is REJECTED",
          _r.returncode == 0 and not _wrote and gate.tool_report_ok("regressions", 0, _rep) is False)

# UID_COUNTED: a wrong uid must be a counted failure, not an informational field.
check("UID_COUNTED: uid 0 (root) is not acceptable",
      gate.uid_is_acceptable(0, 1071) is False)
check("UID_COUNTED: uid 999 != 1071 is not acceptable",
      gate.uid_is_acceptable(999, 1071) is False)
check("UID_COUNTED: root is unacceptable even if expected is 0",
      gate.uid_is_acceptable(0, 0) is False)
check("UID_COUNTED: None uid is not acceptable",
      gate.uid_is_acceptable(None, 1071) is False)
check("UID_COUNTED: uid 1071 is acceptable",
      gate.uid_is_acceptable(1071, 1071) is True)

print()
if failures:
    print("RESULT: FAIL (%d)" % failures)
    sys.exit(1)
print("RESULT: PASS")
