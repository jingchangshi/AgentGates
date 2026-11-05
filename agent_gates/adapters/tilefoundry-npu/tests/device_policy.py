#!/usr/bin/env python3
"""Static + two-state checks for the shared-card device policy.

The contract: occupancy is read once and RECORDED. It never changes the selected
devices, the workload, the number of executions, or the verdict. There is no
wait, poll, retry, skip, downgrade, or exclusive refusal.

CPU only: the two-state harness replaces container I/O and the payload result;
no docker, npu-smi, torch_npu, or device node is touched.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "scripts" / "tilefoundry_gate.py"
MANIFEST = HERE.parent / "manifest.json"
FAILS = []


def check(cond, label):
    if not cond:
        FAILS.append(label)
    print("  %s  %s" % ("PASS" if cond else "FAIL", label))


def load_gate():
    spec = importlib.util.spec_from_file_location("tilefoundry_gate_under_test", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_result(stdout="", returncode=0):
    return types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def exercise(busy):
    gate = load_gate()
    calls = []
    exposed = "/dev/davinci0\n/dev/davinci1\n/dev/davinci_manager\n"
    payload = json.dumps({
        "devices": [
            {"logical": 0, "physical": 0, "device_name": "dev0", "count": 1024,
             "sum": 3072.0, "ok": True},
            {"logical": 1, "physical": 1, "device_name": "dev1", "count": 1024,
             "sum": 3072.0, "ok": True},
        ],
        "all_ok": True,
    })
    def record(is_busy):
        rows = ["| 0 | 123 | other | 100 |"] if is_busy else []
        return {"occupied": True if is_busy else None,
                "idle_established": False,
                "holders": 1 if is_busy else None,
                "process_table": "present",
                "scope": "container-pid-namespace",
                "process_rows": rows}
    occupancy = {0: record(busy), 1: record(False)}
    gate.assert_not_host = lambda *args: None
    gate.container_exists = lambda config: True
    gate.in_container_python = lambda config: "python"
    gate.device_occupancy = lambda config, ids: (occupancy, "npu-smi raw")
    def exec_(config, argv, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["sh", "-c", "ls /dev/davinci* 2>/dev/null"]:
            return fake_result(exposed)
        return fake_result(payload)
    gate.container_exec = exec_
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = gate.gate_device({"TF_REPO": "/tmp", "CONTAINER_NAME": "fake"}, "0,1", None)
    return rc, calls, out.getvalue(), occupancy


def audit_source(source: str):
    """Structural violations of the no-busy-dispatch contract.

    Returns a list of human-readable findings. This is the detector the
    mutation controls below run against deliberately broken copies, because a
    detector that has never failed has not been shown to detect anything.
    """
    tree = ast.parse(source)
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in {"sleep", "waitpid", "poll", "wait"}:
                findings.append("wait primitive %r at line %d" % (name, node.lineno))
        if isinstance(node, ast.If):
            test = ast.get_source_segment(source, node.test) or ""
            body = "\n".join(ast.get_source_segment(source, s) or "" for s in node.body)
            if any(word in test for word in ("occupied", "holders", "busy")) and (
                    "EXIT_BLOCKED" in body or "EXIT_FAIL" in body
                    or "continue" in body or "return" in body):
                findings.append("busy-controlled branch at line %d: %r"
                                % (node.lineno, test.strip()))
    return findings


def main():
    try:
        source = GATE.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        print("cannot read gate: %s" % exc, file=sys.stderr)
        return 2

    gate = load_gate()
    for raw in ("0", "1", "0,1", "1,0", "2"):
        try:
            gate.parse_device_list(raw)
            check(True, "ordered list accepted: %r" % raw)
        except Exception as exc:  # noqa: BLE001
            check(False, "ordered list accepted: %r (%s)" % (raw, exc))
    for raw, why in (("", "empty"), ("a,b", "non-numeric"), ("0,0", "duplicate"),
                     ("0,,1", "empty item")):
        try:
            gate.parse_device_list(raw)
            check(False, "%s list refused: %r" % (why, raw))
        except gate.GateError:
            check(True, "%s list refused: %r" % (why, raw))

    prefix = gate.device_assertion_prefix("0,1")
    check('expected 0,1' in prefix and "count" not in prefix,
          "assertion carries the ordered list, not a count")

    # No wait primitive and no while-only polling in the device path.
    waits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in {"sleep", "waitpid", "poll", "wait"}:
                waits.append((name, node.lineno))
    check(not waits, "no sleep/waitpid/poll/wait call in the adapter (found %r)" % waits)

    # Detector controls: each injection must be caught, and the clean source
    # must stay clean. Without these the checks above could pass by looking at
    # nothing.
    clean = audit_source(source)
    check(not clean, "the shipped source has no busy-dispatch findings: %r" % (clean,))
    # The legacy keys must be accepted (so old files parse) and never read.
    reads = [node.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
             and node.slice.value in {"TF_REQUIRE_EXCLUSIVE", "TF_ALLOW_BUSY_DEVICE"}]
    calls = [node.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and ast.get_source_segment(source, node) is not None
             and "TF_REQUIRE_EXCLUSIVE" in (ast.get_source_segment(source, node) or "")
             and isinstance(node.func, ast.Attribute) and node.func.attr in {"get", "pop"}]
    check(not reads and not calls,
          "the retired exclusive/busy keys are never read (%r, %r)" % (reads, calls))
    injected_wait = source.replace(
        "    occupancy, occupancy_raw = device_occupancy(config, device_ids)",
        "    time.sleep(5)  # injected idle wait\n"
        "    occupancy, occupancy_raw = device_occupancy(config, device_ids)", 1)
    check(bool(audit_source(injected_wait)), "an injected sleep is detected")
    injected_skip = source.replace(
        "    occupancy, occupancy_raw = device_occupancy(config, device_ids)",
        "    occupancy, occupancy_raw = device_occupancy(config, device_ids)\n"
        "    if occupancy[device_ids[0]][\"occupied\"]:\n"
        "        return EXIT_PASS  # injected busy skip", 1)
    check(bool(audit_source(injected_skip)), "an injected busy skip is detected")

    # Occupancy parsing must separate the health table from the process table. A
    # dump with "No running processes" is idle; a dump with a process row is
    # busy; and the two device rows must not be merged into one.
    sample = """\
+--------------------------------+
| NPU ID | Name | Health | Power |
| 0      | Ascend950PR | OK | 275.9 |
| 1      | Ascend950PR | OK | 206.5 |
+---------------------------+---------------+------------------+
| NPU ID                    | Process id    | Process name     |
+===========================+===============+==================+
| No running processes found in NPU 0                              |
+===========================+===============+==================+
"""
    health, holders, seen = gate.parse_npu_smi(sample)
    check(health == {0: "OK", 1: "OK"}, "health is read per physical id: %r" % (health,))
    check(holders == {}, "an idle dump has no holders")
    check(seen, "the process table is reported as present")

    busy = sample.replace(
        "| No running processes found in NPU 0                              |",
        "| 0      | 4242      | other-tenant     | 1000     |")
    health_b, holders_b, seen_b = gate.parse_npu_smi(busy)
    check(holders_b.get(0) and 0 not in holders_b.get(1, []),
          "a holder row lands on device 0 only: %r" % (holders_b,))
    check(seen_b, "a busy dump still reports the process table as present")

    unreadable = gate.parse_npu_smi("not a table at all")
    check(unreadable[0] == {}, "an unparseable dump yields no health rows")

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "gate_device"), None)
    check(fn is not None, "gate_device exists")
    if fn is not None:
        bad = []
        for node in ast.walk(fn):
            if isinstance(node, ast.If):
                test = ast.get_source_segment(source, node.test) or ""
                body = "\n".join(ast.get_source_segment(source, s) or "" for s in node.body)
                if any(word in test for word in ("occupied", "holders", "busy")) and (
                        "EXIT_BLOCKED" in body or "EXIT_FAIL" in body
                        or "continue" in body or "return" in body):
                    bad.append((node.lineno, test.strip()))
        check(not bad, "no occupied/busy branch controls refusal, skip, or return: %r" % bad)

    busy_rc, busy_calls, busy_out, _ = exercise(True)
    idle_rc, idle_calls, idle_out, _ = exercise(False)
    check(busy_rc == idle_rc, "busy and idle return the same exit code (%s/%s)"
          % (busy_rc, idle_rc))
    check(len(busy_calls) == len(idle_calls),
          "busy and idle execute the same number of work commands (%d/%d)"
          % (len(busy_calls), len(idle_calls)))
    check(busy_calls == idle_calls, "busy and idle execute the same commands")
    busy_obj = json.loads(busy_out)
    idle_obj = json.loads(idle_out)
    busy_obj.pop("occupancy", None)
    idle_obj.pop("occupancy", None)
    check(busy_obj == idle_obj,
          "busy and idle emit the same verdict object (occupancy metadata excluded)")
    check('"occupancyRecorded": true' in busy_out,
          "occupancyRecorded is a true record, independent of occupied")
    check('"occupied"' in busy_out and '"holders"' in busy_out,
          "occupied/holders are present as metadata")
    busy_meta = json.loads(busy_out)["occupancy"]
    check(busy_meta["0"]["occupied"] is True and busy_meta["0"]["holders"] == 1,
          "a busy card records occupied=true with a holder count")
    check(busy_meta["1"]["occupied"] is None and busy_meta["1"]["idle_established"] is False,
          "an empty table records UNKNOWN, never idle: %r" % (busy_meta["1"],))
    check("busyRecorded" not in source and "overriddenBusyDevice" not in source,
          "no legacy busyRecorded/overriddenBusyDevice in the gate")

    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print("cannot read manifest: %s" % exc, file=sys.stderr)
        return 2
    dev = next(g for g in manifest["gates"] if g["id"] == "tilefoundry-npu-device")
    obs = dev["observationChecks"]
    check("occupancyRecorded" in obs and "busyRecorded" not in obs,
          "manifest uses occupancyRecorded, not busyRecorded")
    check("multiDeviceCoverage" in obs and "deviceCountOne" not in obs,
          "manifest requests full multi-device coverage, not count=1")
    policy = manifest.get("provenance", {}).get("busyDevicePolicy", {})
    check("RECORD" in policy.get("default", "") and "same workload" in policy.get("default", ""),
          "manifest policy is record-and-proceed")

    print("RESULT:", "PASS" if not FAILS else "FAILURES=%d" % len(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
