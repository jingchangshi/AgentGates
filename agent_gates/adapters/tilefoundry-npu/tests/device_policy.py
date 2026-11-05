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


#: The complete minimal configuration the device Gate needs. Deliberately
#: explicit rather than relying on env.sh: this tool must run from a BARE SHELL.
#:
#: It used to pass {"TF_REPO", "CONTAINER_NAME"} only, so once the P0-A preflight
#: landed the very first thing gate_device did was refuse with
#: ENV_CANN_SCRIPT_MISSING -- a crash from a bare shell, not a policy finding.
#: A test tool that needs an ambient environment is a test tool nobody can run.
MINIMAL_CONFIG = {
    "TF_REPO": "/tmp",
    "CONTAINER_NAME": "fake",
    "IMAGE": "img",
    "CANN_ROOT": "/data/cann",
    "CANN_SET_ENV": "/data/cann/set_env.sh",
    "IN_CONTAINER_PYTHON": "python",
    "CONTAINER_USER": "u",
    "DEVICE_ID": "",
    # RETIRED and IGNORED by the Gate; present so the "legacy keys still parse"
    # contract is exercised with a real key rather than an invented one.
    "TF_ALLOW_BUSY_DEVICE": "",
}


def _stub_container(gate, calls, occupancy, payload, exposed):
    """Replace every container-facing dependency of gate_device.

    gate_device grew mandatory preflight/identity/profile/cache steps. Each has to
    be stubbed here for the two-state comparison to reach the verdict at all; the
    assertions in main() then still prove busy and idle produce the SAME result.
    """
    gate.assert_not_host = lambda *args: None
    gate.container_exists = lambda config: True
    gate.in_container_python = lambda config: "python"
    gate.environment_preflight = lambda config: {
        "env": {"LD_LIBRARY_PATH": "/cann/lib64", "ASCEND_HOME_PATH": "/cann",
                "PATH": "/usr/bin"},
        "libraries": {}, "smi_probe_exit": 0, "ascend_home_path": "/cann"}
    gate.deployment_verify = lambda config, ids, evidence: {
        "verified": True, "verified_scope": "char-nodes-only",
        "nodes": {"/dev/davinci0": "CHAR", "/dev/davinci1": "CHAR",
                  "/dev/davinci_manager": "CHAR", "/dev/hisi_hdc": "CHAR"}}
    gate.capture_runtime_identity = lambda config, python, env: {
        "acceptable": True, "failures": [], "version": "3.12.0",
        "executable": "/usr/bin/python3", "pythonpath": "/p",
        "torch_backend_autoload": None,
        "modules": {"torch": {"imported": True, "version": "2.12.0",
                              "file": "/t.py", "file_sha256": "aa"},
                    "torch_npu": {"imported": True, "version": "2.12.0",
                                  "file": "/n.py", "file_sha256": "bb"}}}
    gate.capture_compile_profile = lambda config, python, env: {
        "digest": "deadbeef", "roots": {}, "libraries": {}, "tools": {}}
    gate.prepare_empty_cache = lambda config, path: {
        "cache_dir": path, "empty_at_start": True, "disable_cache": True}
    gate.device_occupancy = lambda config, ids, facts: (occupancy, "npu-smi raw")

    def exec_(config, argv, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["sh", "-c", "ls /dev/davinci* 2>/dev/null"]:
            return fake_result(exposed)
        return fake_result(payload)

    gate.container_exec = exec_


def exercise(busy):
    gate = load_gate()
    calls = []
    exposed = "/dev/davinci0\n/dev/davinci1\n/dev/davinci_manager\n"
    # The payload reports NUMBERS, not a verdict: the host recomputes it. An "ok"
    # flag here would model the very self-report the Gate no longer trusts.
    payload = json.dumps({
        "devices": [
            {"logical": 0, "physical": 0, "device_name": "dev0", "count": 1024,
             "sum": 3072.0},
            {"logical": 1, "physical": 1, "device_name": "dev1", "count": 1024,
             "sum": 3072.0},
        ],
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
    _stub_container(gate, calls, occupancy, payload, exposed)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = gate.gate_device(dict(MINIMAL_CONFIG), "0,1", None)
    return rc, calls, out.getvalue(), occupancy


def _write_partial_env(omit):
    """Write a complete env file with one key omitted; return its path."""
    import tempfile
    lines = []
    for key, value in MINIMAL_CONFIG.items():
        if key == omit:
            continue
        lines.append("%s=%s" % (key, value))
    handle = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return Path(handle.name)


def refuse_reason(config, argv=None):
    """Return (exit_code, message) for a configuration that must be REFUSED.

    GateError is what the Gate raises for a bad configuration, but this tool must
    not die with a raw traceback from a bare shell: it reports the refusal and
    keeps going. A missing key must surface as a USAGE error naming the key, not
    as an internal exception with a container stack trace.
    """
    gate = load_gate()
    try:
        gate.load_config  # noqa: B018 - existence check only
    except AttributeError:
        pass
    try:
        code = gate.gate_device(dict(config), "0,1", None)
    except gate.GateError as error:
        return error.code, str(error)
    return code, ""


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
    # The injection anchor is DERIVED from the source, not hardcoded: a hardcoded
    # signature silently stopped matching when device_occupancy gained the preflight
    # argument, so the two controls below "detected nothing" while reporting a pass,
    # and the injected source was byte-identical to the clean one.
    # Match by SHAPE, not by exact spelling: the call is now wrapped in
    # run_stage("occupancy_before", ...) for the P0-E watchdog, so an exact-substring
    # anchor stopped matching and both injections below became no-ops that
    # nonetheless reported a pass. A control that silently stops controlling is
    # worse than a missing one, because it reads as evidence.
    import re as _re
    # The call is now `run_stage("occupancy_before", device_occupancy, ...)`, where
    # the function name appears WITHOUT a following parenthesis, so matching
    # "device_occupancy(" misses it entirely. Match the bare name on a non-def line.
    # The call is `run_stage("occupancy_before", device_occupancy,` CONTINUED on
    # the next line. Injecting after the first physical line splits the call and
    # produces a SyntaxError, so the anchor must be the whole STATEMENT: find the
    # opening line, then consume continuation lines until the call closes.
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if "device_occupancy" in line
                  and not line.lstrip().startswith("def ")
                  and not line.lstrip().startswith("#")), None)
    if start is not None:
        end = start
        while not lines[end].rstrip().endswith(")") and end + 1 < len(lines):
            end += 1
        anchor = chr(10).join(lines[start:end + 1])
    else:
        anchor = None
    if anchor is None:
        # The call site was renamed or moved. Say so plainly instead of crashing:
        # this tool must never die with a TypeError from its own instrumentation,
        # because that looks like a defect in the Gate rather than in the harness.
        check(False, "the occupancy call site is findable for injection "
                    "(device_occupancy(config, device_ids...) not found; the two "
                    "controls below cannot run, so they prove nothing)")
        indent = "    "
    else:
        check(True, "the occupancy call site is findable for injection")
        indent = anchor[:len(anchor) - len(anchor.lstrip())]

    if anchor is not None:
        injected_wait = source.replace(
            anchor,
            indent + "time.sleep(5)  # injected idle wait" + chr(10) + anchor, 1)
        check(injected_wait != source,
              "the sleep injection actually changed the source")
        check(bool(audit_source(injected_wait)), "an injected sleep is detected")
        injected_skip = source.replace(
            anchor,
            anchor + chr(10) + indent + "if occupancy[device_ids[0]][\"occupied\"]:"
            + chr(10) + indent + "    return EXIT_PASS  # injected busy skip", 1)
        check(injected_skip != source,
              "the skip injection actually changed the source")
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

    # A stub that stops matching must produce a FAIL, not an uncaught crash. Without
    # this, removing any single stub killed the tool with GateError and the run showed
    # "0 failures" with rc=1 -- indistinguishable from a clean pass at a glance.
    try:
        busy_rc, busy_calls, busy_out, _ = exercise(True)
        idle_rc, idle_calls, idle_out, _ = exercise(False)
    except Exception as error:  # noqa: BLE001
        check(False, "the two-state harness could not run (%s: %s). A stub no longer "
                     "matches the Gate, so the busy/idle comparison below proves "
                     "nothing" % (type(error).__name__, str(error)[:120]))
        busy_rc = idle_rc = None
        busy_calls = idle_calls = None
        busy_out = idle_out = ""
    check(busy_rc == idle_rc, "busy and idle return the same exit code (%s/%s)"
          % (busy_rc, idle_rc))
    check(len(busy_calls) == len(idle_calls),
          "busy and idle execute the same number of work commands (%d/%d)"
          % (len(busy_calls), len(idle_calls)))
    check(busy_calls == idle_calls, "busy and idle execute the same commands")
    if busy_out and idle_out:
        busy_obj = json.loads(busy_out)
        idle_obj = json.loads(idle_out)
    else:
        busy_obj = idle_obj = {"unavailable": True}
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

    # A BARE SHELL must be enough to run this tool, and a missing configuration key
    # must surface as a NAMED usage error rather than an internal crash. This tool
    # used to die with a raw GateError traceback (ENV_CANN_SCRIPT_MISSING) because
    # exercise() supplied only TF_REPO/CONTAINER_NAME.
    for missing in ("CANN_SET_ENV", "TF_REPO", "CONTAINER_NAME"):
        broken = dict(MINIMAL_CONFIG)
        broken.pop(missing, None)
        code, message = refuse_reason(broken)
        check(code != 0,
              "a configuration missing %s is refused (exit %s)" % (missing, code))
        check(message is not None and "Traceback" not in str(message),
              "a missing %s reports a message, not a traceback" % missing)

    # The refusal must NAME the missing key. Two DIFFERENT layers own two different
    # keys, and the test must respect that instead of demanding one layer do both:
    #   - TF_REPO / CONTAINER_NAME are structural, so load_config owns them;
    #   - CANN_SET_ENV is an ENVIRONMENT fact, so the preflight owns it. It is
    #     deliberately not a load_config requirement: a machine may legitimately
    #     carry the path elsewhere for non-device work.
    # Asking gate_device directly would test the HOST guard, not the usage error,
    # because it refuses on the host before it ever reads the configuration.
    for missing in ("TF_REPO", "CONTAINER_NAME"):
        try:
            gate.load_config(_write_partial_env(missing))
        except gate.GateError as error:
            check(missing in str(error),
                  "a missing %s is named in the refusal: %r"
                  % (missing, str(error)[:80]))
        except Exception as error:  # noqa: BLE001
            check(False, "a missing %s must raise GateError, got %s: %s"
                  % (missing, type(error).__name__, str(error)[:60]))
        else:
            check(False, "a configuration missing %s must be refused" % missing)

    try:
        gate.environment_preflight({"CANN_SET_ENV": "", "CONTAINER_NAME": "c",
                                    "TF_REPO": "/tmp"})
    except gate.GateError as error:
        check("CANN_SET_ENV" in str(error) and error.code == gate.EXIT_BLOCKED,
              "an unset CANN_SET_ENV is named and BLOCKED_ENV: %r" % (str(error)[:70],))
    except Exception as error:  # noqa: BLE001
        check(False, "unset CANN_SET_ENV must raise GateError, got %s"
              % type(error).__name__)
    else:
        check(False, "an unset CANN_SET_ENV must be refused")

    # A bare shell really is enough: the config this tool supplies must be
    # complete on its own. Assert every key the Gate declares as required.
    try:
        required = gate.REQUIRED_CONFIG_KEYS
    except AttributeError:
        required = None
    if required:
        absent = sorted(k for k in required if k not in MINIMAL_CONFIG)
        check(not absent,
              "MINIMAL_CONFIG supplies every required key (absent: %r)" % (absent,))
    else:
        for key in ("TF_REPO", "CONTAINER_NAME", "CANN_SET_ENV"):
            check(key in MINIMAL_CONFIG, "MINIMAL_CONFIG supplies %s" % key)
    print("RESULT:", "PASS" if not FAILS else "FAILURES=%d" % len(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
