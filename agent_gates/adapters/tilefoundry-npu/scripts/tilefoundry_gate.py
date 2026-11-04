#!/usr/bin/env python3
"""The tilefoundry-npu Gates.

Four subcommands, each a distinct Gate:

    --check-config   validate the configuration file without touching anything
    build            build the image and create the container
    cpu              the CPU Gate: import, analyze, finalize, and the tools
    device           the device Gate: run inside the container, as root

One rule runs through all of them: TileFoundry work happens INSIDE the
container. There is no host path here. A Gate that would run the toolchain on
the host refuses instead, because the environment that produced a result is part
of the result, and a host run and a container run are not the same claim.

Exit codes
    0    PASS
    1    FAIL
    2    configuration error
    75   BLOCKED_ENV -- the run did not happen, and no result is reported
    97   the in-payload device assertion failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
import sys
from pathlib import Path

PROG = "tilefoundry_gate"
EXIT_PASS, EXIT_FAIL, EXIT_CONFIG, EXIT_BLOCKED, EXIT_ASSERT = 0, 1, 2, 75, 97

KNOWN_KEYS = {
    "TF_REPO", "IMAGE", "CONTAINER_NAME", "CANN_ROOT", "IN_CONTAINER_PYTHON",
    "CONTAINER_USER", "CONTAINER_UID", "DEVICE_ID", "TF_REQUIRE_EXCLUSIVE", "TF_VENV", "TF_WORKSPACE",
    "TF_ALLOW_BUSY_DEVICE",
    "TF_NPU_ENV", "CONDA_SH", "VENDOR_WHEELS", "TF_CONDA_ENV",
    "CANN_SET_ENV", "PROXY_SH",
}

class GateError(Exception):
    """A refusal with an exit code. Never a pass, never a silent default."""

    def __init__(self, message: str, code: int = EXIT_CONFIG):
        super().__init__(message)
        self.code = code

def log(message: str) -> None:
    print("%s: %s" % (PROG, message), file=sys.stderr)

# ---------------------------------------------------------------- configuration

def parse_env_file(path: Path) -> dict:
    """Parse a strict NAME=value file. Parsed as data; never executed."""
    if not path.is_file():
        raise GateError(
            "configuration file not found: %s\n"
            "  create it from env.example.sh:\n"
            "    cp %s %s\n"
            "    chmod 600 %s" % (path, path.with_name("env.example.sh"), path, path),
        )
    values: dict = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise GateError("%s:%d: not NAME=value: %s" % (path, number, raw))
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise GateError("%s:%d: bad key name: %s" % (path, number, key))
        if key in values:
            raise GateError("%s:%d: duplicate key: %s" % (path, number, key))
        if key not in KNOWN_KEYS:
            raise GateError(
                "%s:%d: unknown key: %s\n"
                "  known keys: %s" % (path, number, key, ", ".join(sorted(KNOWN_KEYS))),
            )
        values[key] = value
    return values

def load_config(path: Path) -> dict:
    values = parse_env_file(path)
    required = ["TF_REPO", "IMAGE", "CONTAINER_NAME", "CANN_ROOT",
                "IN_CONTAINER_PYTHON", "CONTAINER_USER"]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise GateError(
            "required configuration missing: %s" % ", ".join(missing),
        )
    repo = Path(values["TF_REPO"])
    if not repo.is_dir():
        raise GateError("TF_REPO is not a directory: %s" % repo)
    for name in ("Dockerfile", "create-container.sh", "verify-container.sh"):
        candidate = repo / "docker" / "ascend950pr" / name
        if not candidate.is_file():
            raise GateError(
                "the docker recipe is missing from TF_REPO: %s\n"
                "  this adapter reuses %s/docker/ascend950pr instead of duplicating it"
                % (candidate, repo),
            )
    return values
# ------------------------------------------------------------- the host guard

CONTAINER_MARKER = "/.dockerenv"

def assert_not_host(operation: str) -> None:
    """Refuse to run the TileFoundry toolchain on the host.

    This adapter exists because a host run and a container run are different
    claims. Allowing the host path "just for a quick check" is how a result ends
    up in the evidence with no record of which environment produced it, so the
    guard fails closed and says exactly what to do instead.

    The check is the container marker the kernel and docker provide, not a
    variable this code sets itself: a marker we exported would be satisfied by
    the act of exporting it.
    """
    # docker creates /.dockerenv in every container. It is set by the runtime,
    # not by this code, so it cannot be satisfied by exporting a variable.
    if os.path.exists(CONTAINER_MARKER):
        return
    # NOTE: an earlier version of this message advertised "run it from the host and it
    # will docker exec for you". That was never true -- this guard runs first and
    # refuses -- and advertising a path that cannot work is worse than saying nothing,
    # because the operator concludes the tool is broken rather than misused. The gate
    # is run INSIDE the container; the docker exec is the caller's.
    script = Path(sys.argv[0]).name
    raise GateError(
        "refusing to run %s on the HOST.\n"
        "  TileFoundry analyse/finalize/test must run inside the container, so the\n"
        "  environment that produced a result is the environment that was verified.\n"
        "  This gate does NOT re-exec itself through docker; run it in the container:\n"
        "    docker exec -u <user> <container> python3 <path-to>/%s --cpu\n"
        "  (swap --cpu for --device / --build as needed.)\n"
        "  /home is bind-mounted, so this adapter is already visible inside.\n"
        % (operation, script),
        EXIT_CONFIG,
    )

# ------------------------------------------------------------------- docker io

def run(argv: list, timeout: int = 1800, check: bool = False,
        env: dict | None = None) -> subprocess.CompletedProcess:
    log("$ " + " ".join(argv))
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False, env=env,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if check and completed.returncode != 0:
        raise GateError(
            "command failed with status %d: %s"
            % (completed.returncode, " ".join(argv)),
            EXIT_FAIL,
        )
    return completed

def inside_container() -> bool:
    """Are we already running inside the container under test?

    This adapter supports exactly ONE mode: execute inside the container. But
    "inside" can be reached two ways, and they need different plumbing:

      * the caller is on the host and this script re-execs each command with
        `docker exec` -- but then this process is NOT in the container; or
      * this process itself is running in the container (`/.dockerenv` present),
        in which case there is no docker CLI and none is needed.

    The guard in assert_not_host only permits the second. Supporting both without
    asking which one we are in is how the gate crashed with
    "FileNotFoundError: 'docker'" the first time it was run the documented way.
    """
    return os.path.exists(CONTAINER_MARKER)


def container_exists(config: dict) -> bool:
    if inside_container():
        # We are in the container; there is nothing to ask docker about, and no
        # docker binary to ask it with.
        return True
    result = subprocess.run(
        ["docker", "inspect", config["CONTAINER_NAME"]],
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0

def in_container_python(config: dict) -> str:
    return config["IN_CONTAINER_PYTHON"]

def container_exec(config: dict, argv: list, *, user: str | None = None,
                   env: dict | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run argv inside the container. The only way this adapter executes anything.

    Dual mode -- see inside_container(). Inside the container the argv runs
    directly; the `user` argument is then a no-op, because a process cannot change
    its own uid without privilege it should not have. That is why the device Gate
    requires being launched as root inside the container rather than asking this
    function to become root.
    """
    if inside_container():
        merged = dict(os.environ)
        merged.update(env or {})
        return run(list(argv), timeout=timeout, env=merged)
    command = ["docker", "exec"]
    if user is not None:
        command += ["-u", user]
    for key, value in (env or {}).items():
        command += ["-e", "%s=%s" % (key, value)]
    command += [config["CONTAINER_NAME"]] + argv
    return run(command, timeout=timeout)

# ----------------------------------------------------------------- CPU Gate

CPU_TOOLS = [
    ("cpu_equality", "examples/rmsnorm_950pr/tools/cpu_equality.py"),
    ("regressions", "examples/rmsnorm_950pr/tools/regressions.py"),
    ("verify_tir", "examples/rmsnorm_950pr/tools/verify_tir.py"),
    ("check_copy_slice", "examples/rmsnorm_950pr/tools/check_copy_slice.py"),
]

#: A challenge only a real Python can answer.
#:
#: The Gate used to accept exit-code-only success, and the Architect reproduced a
#: FALSE PASS with IN_CONTAINER_PYTHON=/bin/true: every check "passed" because
#: /bin/true exits 0 and nothing ran. So the interpreter must prove what it is.
#: /bin/true prints nothing, fails to produce this JSON, and is refused.
_CHALLENGE_SALT = "tf-gate-identity:"
_EXPECTED_CHALLENGE = hashlib.sha256(
    (_CHALLENGE_SALT + str(sum(range(1000)))).encode()
).hexdigest()

IDENTITY_SCRIPT = (
    "import sys,os,json,hashlib;print(json.dumps({"
    "'executable':os.path.realpath(sys.executable),"
    "'version':'%d.%d'%sys.version_info[:2],"
    "'challenge':hashlib.sha256(('" + _CHALLENGE_SALT + "'+str(sum(range(1000)))).encode()).hexdigest()"
    "}))"
)


def container_read_text(config: dict, path: str, *, user: str | None = None):
    """Read a file from the container. Returns (ok, text)."""
    result = container_exec(config, ["cat", path], user=user)
    return result.returncode == 0, result.stdout


def _last_json_line(text: str):
    for line in reversed([l for l in (text or "").splitlines() if l.strip()]):
        try:
            return json.loads(line)
        except Exception:  # noqa: BLE001
            continue
    return None


def identity_verdict(info, configured_python):
    """Decide, from the parsed identity, whether this interpreter is acceptable.

    Split out as a pure function so it can be UNIT-TESTED WITHOUT A CONTAINER.
    That matters: the failure this guards against is a config pointing at a
    non-interpreter, and a test for it must be runnable where such a config is
    harmless. /bin/true yields info=None here and is refused.
    """
    if info is None:
        return False, {
            "executableMatches": False,
            "versionIsPython3": False,
            "challengeMatches": False,
            "why": "no identity JSON: not a Python interpreter (e.g. /bin/true)",
        }
    try:
        want_exec = os.path.realpath(configured_python)
    except Exception:  # noqa: BLE001
        want_exec = configured_python
    version = info.get("version")
    checks = {
        "executableMatches": info.get("executable") == want_exec,
        "versionIsPython3": isinstance(version, str)
        and version.split(".")[0].isdigit()
        and int(version.split(".")[0]) >= 3,
        "challengeMatches": info.get("challenge") == _EXPECTED_CHALLENGE,
    }
    return all(checks.values()), checks


def uid_is_acceptable(actual_uid, expected_uid):
    """Wrong uid is a FAILURE, not an informational field.

    In the previous revision the uid comparison was recorded in the evidence but
    never counted toward failures, so a wrong-uid run could still PASS. This predicate
    is the single source of truth the Gate counts on.
    """
    return actual_uid is not None and actual_uid == expected_uid and actual_uid != 0


def report_is_fresh(mtime, run_start, slack=2.0):
    """A report written before this run started is the PREVIOUS run's file.

    Staleness guard. With a fixed work directory and fixed filenames, an exit-0 run
    that writes nothing could be "accepted" by reading the previous run's report.
    A unique per-run directory already prevents that; this is the second lock, for
    the case where a directory is ever reused.
    """
    try:
        return float(mtime) >= (float(run_start) - slack)
    except (TypeError, ValueError):
        return False


#: The mandatory per-case checks for cpu_equality. Named explicitly so a missing one
#: is a FAILURE. all({}.values()) is True, so an empty checks dict must not pass.
REQUIRED_CASE_CHECKS = ("shape", "dtype", "bits")


def tool_report_ok(name, returncode, report):
    """A tool "ran" only if it emitted a structured PASS report with real checks.

    Pure, so unit-testable without a container. Exit 0 with a missing or non-PASS
    report is a FAILURE: a tool that exits 0 and prints nothing is the vacuous
    success this Gate refuses.
    """
    ok = (
        returncode == 0
        and report is not None
        and report.get("status") == "PASS"
        and report.get("exit_code") == 0
    )
    if name == "cpu_equality" and report is not None:
        cases = report.get("cases") or []
        if not cases:
            return False
        for c in cases:
            checks = c.get("checks")
            if c.get("status") != "PASS" or not isinstance(checks, dict):
                return False
            # every mandatory check MUST be present and exactly True; a missing key
            # is a failure, not a default-true via all().
            if any(k not in checks or checks[k] is not True for k in REQUIRED_CASE_CHECKS):
                return False
    if name == "check_copy_slice" and report is not None:
        checks = report.get("checks") or {}
        ok = (ok and checks.get("two_copies") is True
              and checks.get("one_smem_buffer") is True
              and checks.get("stages_through_same_buffer") is True)
    return ok


def check_interpreter_identity(config, python, env, user):
    """Prove the configured interpreter is a real Python at the expected path."""
    result = container_exec(config, [python, "-c", IDENTITY_SCRIPT], user=user, env=env)
    info = _last_json_line(result.stdout)
    ok, checks = identity_verdict(info, python)
    detail = {
        "configured": python,
        "returncode": result.returncode,
        "stdout_bytes": len(result.stdout or ""),
        "observed": info,
        "checks": checks,
        "expected_challenge": _EXPECTED_CHALLENGE,
    }
    if info is None:
        detail["why"] = (
            "the interpreter produced no JSON. It did not run Python: this is what a"
            " non-interpreter such as /bin/true looks like, and it must not pass."
        )
    return ok, detail


def gate_cpu(config: dict, evidence_dir: Path | None) -> int:
    """The CPU Gate: every claim is read from evidence, not from an exit code."""
    assert_not_host("the CPU Gate")
    if not container_exists(config):
        raise GateError(
            "container %s does not exist; run the build Gate first"
            % config["CONTAINER_NAME"], EXIT_BLOCKED,
        )
    repo = config["TF_REPO"]
    python = in_container_python(config)
    user = config["CONTAINER_USER"]
    expected_uid = int(config.get("CONTAINER_UID", "1071"))
    env = {"PYTHONPATH": "%s/src:%s/targets/ascend" % (repo, repo),
           "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    failures = 0
    run_id = uuid.uuid4().hex[:12]
    run_start = time.time()
    # UNIQUE directory per run. A fixed path let a skipped writer with rc 0 be
    # "accepted" by reusing a previous run's report; mktemp makes the directory new
    # and empty, so a writer that writes nothing has no file to be accepted from.
    mk = container_exec(config, ["mktemp", "-d", "/tmp/tfgate.XXXXXXXX"], user=user)
    workdir = (mk.stdout or "").strip()
    if mk.returncode != 0 or not workdir.startswith("/tmp/"):
        raise GateError("could not allocate a run directory: %r" % workdir, EXIT_BLOCKED)

    # 1. ACTUAL uid. The configured user string is NOT evidence: inside the
    #    container the exec helper's user argument is a no-op, so "ran as X" must
    #    be read from the process, not asserted from configuration.
    log("--- actual identity ---")
    uid_result = container_exec(config, ["id", "-u"], user=user)
    uid_text = (uid_result.stdout or "").strip()
    actual_uid = int(uid_text) if uid_result.returncode == 0 and uid_text.isdigit() else None
    if actual_uid is None:
        raise GateError(
            "could not read the actual uid inside the container (id -u returned %r)"
            % uid_text, EXIT_BLOCKED,
        )
    log("actual uid = %d (expected %d, must be non-root)" % (actual_uid, expected_uid))
    # EARLY precondition: refuse BEFORE any CPU tool runs, not at the end. Running
    # the whole suite as the wrong account and only then objecting wastes the run
    # and leaves tool output produced under the wrong identity.
    if not uid_is_acceptable(actual_uid, expected_uid):
        # COUNTED FAILURE, not an output-only note. The previous revision recorded the
        # uid but never added it to failures, so a wrong-uid run could still PASS. Now
        # a wrong uid fails: nothing runs, evidence is written, and the Gate exits 1.
        log("FAIL: the CPU Gate must run as uid %d (non-root); id -u reported %d."
            " Refusing before any tool runs; this counts as a failure."
            % (expected_uid, actual_uid))
        if evidence_dir is not None:
            write_evidence(evidence_dir, "cpu", {
                "run_id": run_id,
                "container": config["CONTAINER_NAME"],
                "image": config["IMAGE"],
                "python": python,
                "configured_user": user,
                "ran_as_uid": actual_uid,
                "expected_uid": expected_uid,
                "refused_before_tools": True,
                "failures": 1,
                "observations": {
                    "exitCodeZero": False,
                    "ranInsideContainer": inside_container(),
                    "inContainerPythonUsed": False,
                    "importOk": False,
                    "analyzeOk": False,
                    "finalizeOk": False,
                    "cpuEqualityRuns": False,
                    "regressionsRun": False,
                    "verifyTirRuns": False,
                    "checkCopySliceRuns": False,
                    "noHostInterpreter": True,
                    "interpreterIdentityOk": False,
                    "actualUidIsExpected": False,
                    "toolReportsStructured": False,
                    "finalizedArtifactTyped": False,
                },
            })
        return EXIT_FAIL

    # 2. Interpreter identity challenge. A config pointing at /bin/true dies here.
    ident_ok, ident_detail = check_interpreter_identity(config, python, env, user)
    if not ident_ok:
        log("FAIL interpreter identity: " + json.dumps(ident_detail))
        failures += 1

    # 3. Import both roots, and require the success token in the OUTPUT, not just
    #    exit 0: a process that exits 0 without printing is not proof of an import.
    imported = container_exec(
        config,
        [python, "-c",
         "import tilefoundry, tilefoundry_ascend; print('IMPORT-OK', tilefoundry.__file__)"],
        user=user, env=env,
    )
    import_ok = imported.returncode == 0 and "IMPORT-OK" in (imported.stdout or "")
    if not import_ok:
        failures += 1

    # 4. analyze: require HIR in the OUTPUT, not merely exit 0.
    analyze = container_exec(
        config, [python, "-m", "tilefoundry.cli", "analyze",
                 "%s/examples/rmsnorm_950pr/copy_slice.py:CopySlice" % repo],
        user=user, env=env,
    )
    analyze_ok = (
        analyze.returncode == 0
        and "@func" in (analyze.stdout or "")
        and "copy1" in (analyze.stdout or "")
    )
    if not analyze_ok:
        failures += 1

    # 5. finalize: the TYPED artifact must exist and look like finalized TIR, with a
    #    recorded hash and size. "exit 0" alone could mean an empty file was written.
    container_exec(config, ["mkdir", "-p", workdir], user=user)
    tir_path = workdir + "/tir.py"
    finalize = container_exec(
        config, [python, "-m", "tilefoundry.cli", "schedule", "finalize",
                 "%s/examples/rmsnorm_950pr/copy_slice.py:CopySlice" % repo, tir_path],
        user=user, env=env,
    )
    tir_read_ok, tir_text = container_read_text(config, tir_path, user=user)
    tir_sha = hashlib.sha256((tir_text or "").encode()).hexdigest() if tir_read_ok else None
    finalize_ok = (
        finalize.returncode == 0
        and tir_read_ok
        and len(tir_text or "") > 200
        and "@prim_func" in (tir_text or "")
        and "tensor_view" in (tir_text or "")
    )
    if not finalize_ok:
        failures += 1

    # 6. The tool suite. Each tool must emit a STRUCTURED report whose status is PASS
    #    and whose exit_code is 0, plus the tool-specific fields its own report
    #    carries. A bare exit 0 is not a result.
    log("--- tool suite ---")
    tool_status = {}
    tool_reports = {}
    for name, relative in CPU_TOOLS:
        target = "%s/%s" % (repo, relative)
        out_path = "%s/%s.json" % (workdir, name)
        record = container_exec(config, [python, target, "--out", out_path], user=user, env=env)
        read_ok, text = container_read_text(config, out_path, user=user)
        fresh = False
        if read_ok:
            try:
                fresh = report_is_fresh(os.stat(out_path).st_mtime, run_start)
            except OSError:
                fresh = False
        report = None
        if read_ok and fresh:
            try:
                report = json.loads(text)
            except Exception:  # noqa: BLE001
                report = None
        tool_reports[name] = {
            "returncode": record.returncode,
            "report_present": report is not None,
            "report_fresh": fresh,
            "status": (report or {}).get("status"),
            "exit_code": (report or {}).get("exit_code"),
        }
        # fresh=False means the file predates this run (stale) or is absent: a
        # writer that wrote nothing cannot be accepted from a previous run's file.
        ok = fresh and tool_report_ok(name, record.returncode, report)
        tool_status[name] = ok
        if ok:
            print("PASS %s" % name)
        else:
            log("FAIL %s: " + json.dumps(tool_reports[name]))
            failures += 1

    tool_reports_ok = bool(tool_status) and all(tool_status.values())

    observations = {
        "exitCodeZero": failures == 0,
        "ranInsideContainer": inside_container(),
        "inContainerPythonUsed": ident_ok,
        "importOk": import_ok,
        "analyzeOk": analyze_ok,
        "finalizeOk": finalize_ok,
        "cpuEqualityRuns": tool_status.get("cpu_equality", False),
        "regressionsRun": tool_status.get("regressions", False),
        "verifyTirRuns": tool_status.get("verify_tir", False),
        "checkCopySliceRuns": tool_status.get("check_copy_slice", False),
        "noHostInterpreter": True,
        "interpreterIdentityOk": ident_ok,
        "actualUidIsExpected": actual_uid == expected_uid and actual_uid != 0,
        "toolReportsStructured": tool_reports_ok,
        "finalizedArtifactTyped": finalize_ok,
    }
    if evidence_dir is not None:
        archive = Path(evidence_dir) / ("tool_reports_" + run_id)
        archive.mkdir(parents=True, exist_ok=True)
        for _n in tool_reports:
            _src = Path(workdir) / (_n + ".json")
            if _src.is_file():
                shutil.copy(_src, archive / (_n + ".json"))
        write_evidence(evidence_dir, "cpu", {
            "run_id": run_id,
            "workdir": workdir,
            "container": config["CONTAINER_NAME"],
            "image": config["IMAGE"],
            "python": python,
            "configured_user": user,
            "ran_as_uid": actual_uid,
            "expected_uid": expected_uid,
            "interpreter": ident_detail.get("observed"),
            "interpreter_checks": ident_detail.get("checks"),
            "finalized_artifact": {"sha256": tir_sha,
                                   "bytes": len(tir_text or "") if tir_read_ok else 0,
                                   "path": tir_path},
            "tool_reports": tool_reports,
            "failures": failures,
            "observations": observations,
        })
    # Remove the per-run directory: it is scratch, and leaving it invites the next
    # run to read a stale file. Raw reports were archived above when evidence was
    # requested.
    shutil.rmtree(workdir, ignore_errors=True)
    log("cpu gate: %s" % ("PASS" if failures == 0 else "FAIL"))
    return EXIT_PASS if failures == 0 else EXIT_FAIL

# ---------------------------------------------------------------- device Gate

def device_assertion_prefix(expected: str) -> str:
    """A shell prefix that asserts the device from INSIDE the payload.

    The wrapper's own environment proves nothing about what the workload sees:
    the wrapper never enters the container. This runs there, reads what the
    workload will actually inherit, and refuses before the workload starts.

    It also prints the effective uid, because on this platform device work must
    enter as root and a payload running as the wrong account is not the run that
    was requested.
    """
    return (
        'v="${ASCEND_RT_VISIBLE_DEVICES:-<UNSET>}"; '
        'n=$(printf "%s" "$v" | tr "," "\n" | grep -c .); '
        'u=$(id -u); '
        'echo "DEVICE-ASSERT: visible=$v count=$n wanted=' + expected + ' uid=$u"; '
        'if [ "$v" != "' + expected + '" ] || [ "$n" -ne 1 ]; then '
        'echo "DEVICE-ASSERT-FAIL: payload sees ASCEND_RT_VISIBLE_DEVICES=$v ($n device(s)), expected exactly 1 matching ' + expected + '" >&2; '
        'exit 97; fi; '
        'if [ "$u" -ne 0 ]; then '
        'echo "DEVICE-ASSERT-FAIL: device work must enter as root, payload uid is $u" >&2; '
        'exit 97; fi; '
        'exec "$@"'
    )

def device_occupancy(config: dict, device_id: str) -> tuple:
    """Report who is on the card.

    The caller refuses when it is held. Occupancy is READ HERE, BEFORE anything
    imports torch_npu or reaches aclInit, because the cost of probing a held card is
    not hypothetical: in this program's own incident a device open at 13:46:26 was
    followed 32 s later by a host-wide exec stall. A card someone else is using is not
    a fact about the code under test, and it is also not ours to test on.
    """
    result = container_exec(config, ["npu-smi", "info"], user="root")
    if result.returncode != 0:
        raise GateError(
            "npu-smi failed with status %d; occupancy is UNKNOWN, not empty"
            % result.returncode, EXIT_BLOCKED,
        )
    rows = []
    for line in result.stdout.splitlines():
        cells = [c.strip() for c in line.replace("|", " ").split()]
        if len(cells) >= 3 and cells[0] == device_id and cells[1].isdigit():
            rows.append(line.strip())
    return rows, result.stdout

def gate_device(config: dict, device_id: str, evidence_dir) -> int:
    """The device Gate: root inside the container, explicit device, asserted there."""
    assert_not_host("the device Gate")
    if not device_id:
        raise GateError(
            "no device id given. Pass --device-id N or set DEVICE_ID=N.\n"
            "  There is no default and no search: a run against a device the\n"
            "  operator did not name is not evidence about the device they meant.",
        )
    if not re.fullmatch(r"[0-9]+", device_id):
        raise GateError("device id must be numeric, got: %s" % device_id)
    if not container_exists(config):
        raise GateError(
            "container %s does not exist; run the build Gate first"
            % config["CONTAINER_NAME"], EXIT_BLOCKED,
        )

    repo = config["TF_REPO"]
    python = in_container_python(config)
    env = {"PYTHONPATH": "%s/src:%s/targets/ascend" % (repo, repo),
           "ASCEND_RT_VISIBLE_DEVICES": device_id}

    # OCCUPANCY FIRST, AND IT REFUSES BY DEFAULT.
    #
    # This runs before torch_npu is imported and before aclInit is ever reached. That
    # ordering is the whole point, so it must not be moved below the probe.
    #
    # The default used to be the other way round: record occupancy, proceed, and only
    # refuse when TF_REQUIRE_EXCLUSIVE=1 was set. That made "somebody else is using
    # the card" the normal case to walk into. The evidence against it is our own: a
    # device open at 13:46:26 was followed 32 s later by a host-wide stall, with the
    # card already held by other tenants. Probing a held card is not a neutral act.
    #
    # A busy card is also not a fact about the code under test, so proceeding cannot
    # produce a result about that code: it produces a result about a shared,
    # contention-affected card, which is why the override is loud rather than quiet.
    holders, _ = device_occupancy(config, device_id)
    occupied = bool(holders)
    if occupied:
        log("device %s is OCCUPIED by %d process(es):" % (device_id, len(holders)))
        for row in holders:
            log("  " + row)
    else:
        log("device %s reports no active process" % device_id)

    override = config.get("TF_ALLOW_BUSY_DEVICE") == "1"
    if occupied and not override:
        log("BLOCKED_ENV: device %s is held by another user; refusing to open it." % device_id)
        log("  The run did not happen. This is not a pass and not a failure: there is")
        log("  no result to report, and reporting one would be a fabrication.")
        log("  Refusal is the default because opening a held card is not a neutral act:")
        log("  in this program a device open at 13:46:26 preceded a host-wide stall 32 s")
        log("  later, with the card already held by other tenants.")
        log("  To override deliberately, set TF_ALLOW_BUSY_DEVICE=1. Anything the run")
        log("  then produces is a SHARED-CARD result: timings are PROVISIONAL and the")
        log("  evidence is marked as overridden.")
        return EXIT_BLOCKED

    if occupied and override:
        log("=" * 72)
        log("OVERRIDE: TF_ALLOW_BUSY_DEVICE=1 -- proceeding on a card held by others.")
        log("  Anything below describes a SHARED card, not exclusive use.")
        log("  Timings are PROVISIONAL; numerics are unaffected by contention.")
        log("  This override is recorded in the evidence as overriddenBusyDevice=true.")
        log("=" * 72)

    # The device must be bound INTO the container. Presence on the host is not
    # enough: a container fixed to device 0 cannot reach device 1, and running
    # there anyway would be evidence about a different board.
    bound = container_exec(config, ["test", "-e", "/dev/davinci%s" % device_id], user="root")
    if bound.returncode != 0:
        exposed = container_exec(config, ["sh", "-c", "ls /dev/davinci* 2>/dev/null"], user="root")
        log("BLOCKED_ENV: device %s is not bound into %s." % (device_id, config["CONTAINER_NAME"]))
        log("  the container exposes: %s" % " ".join(exposed.stdout.split()))
        return EXIT_BLOCKED

    # The assertion runs INSIDE the container, around the real workload.
    argv = ["sh", "-c", device_assertion_prefix(device_id), "--",
            python, "-c",
            "import torch, torch_npu; print('torch_npu ok', torch_npu.__version__)"]
    log("--- device probe, asserted inside the payload ---")
    probe = container_exec(config, argv, user="root", env=env, timeout=600)
    if probe.returncode == EXIT_ASSERT:
        log("FAIL: the in-payload device assertion failed; the workload did not run")
        return EXIT_FAIL
    if probe.returncode != 0:
        log("FAIL: the device probe exited %d" % probe.returncode)
        return EXIT_FAIL

    # Same rule as the CPU gate: the keys are exactly the manifest observationChecks.
    observations = {
        "exitCodeZero": True,
        "explicitDeviceId": bool(device_id),
        "ranAsRoot": True,              # docker exec -u root, asserted in-payload
        "deviceCountOne": True,         # the in-payload assertion requires exactly 1
        "assertionInsidePayload": True, # the prefix runs inside the container
        "deviceBoundIntoContainer": True,
        "occupancyReported": True,
        # Only true when the card was held AND the run was allowed to continue via the
        # override. Reporting True when the card was free would claim an override that
        # never happened; reporting False when it was used would hide it.
        "overriddenBusyDevice": occupied and config.get("TF_ALLOW_BUSY_DEVICE") == "1",
        # The refusal path returns 75 before this point, so if we are here it was not
        # taken. Recorded explicitly rather than left as an absent key.
        "refusedBusyDevice": False,
    }
    if evidence_dir is not None:
        write_evidence(evidence_dir, "device", {
            "container": config["CONTAINER_NAME"],
            "device_id": device_id,
            "occupied": occupied,
            "holders": len(holders),
            "ran_as": "root",
            "assertion": "in-payload, exactly one visible device and uid 0",
            "observations": observations,
        })
    log("device gate: PASS (schema-level; this is not device attestation)")
    return EXIT_PASS

# ------------------------------------------------------------------ evidence

def write_evidence(directory, name: str, record: dict) -> None:
    """Write one JSON record per Gate run."""
    import datetime
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["gate"] = name
    record["recorded_at"] = datetime.datetime.now().astimezone().isoformat()
    target = directory / ("gate-%s.json" % name)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log("evidence: %s" % target)

# -------------------------------------------------------------- build Gate

def gate_build(config: dict, evidence_dir) -> int:
    """Build the image and create the container, reusing the TF_REPO recipe."""
    repo = Path(config["TF_REPO"])
    recipe = repo / "docker" / "ascend950pr"
    image = config["IMAGE"]
    container = config["CONTAINER_NAME"]

    if container_exists(config):
        log("container %s already exists; nothing to create" % container)
    else:
        script = recipe / "create-container.sh"
        if not script.is_file():
            raise GateError("the recipe script is missing: %s" % script)
        log("creating %s from %s" % (container, image))
        created = run(
            ["bash", "-c",
             "cd %s && IMAGE=%s CONTAINER_NAME=%s ./create-container.sh"
             % (recipe, image, container)],
            timeout=1800,
        )
        if created.returncode != 0:
            raise GateError(
                "container creation failed with status %d" % created.returncode,
                EXIT_FAIL,
            )

    # The adapter does not duplicate the recipe: it reuses the repo one, so the
    # container that runs the Gate is the same container the repo verifies.
    log("reusing the recipe from %s" % recipe)
    observations = {
        "exitCodeZero": True,
        "imagePresent": True,       # the recipe refuses otherwise
        "containerCreated": container_exists(config),
        "recipeReused": True,       # never duplicated here
    }
    if evidence_dir is not None:
        write_evidence(evidence_dir, "build", {
            "image": image, "container": container,
            "recipe": str(recipe), "reused": True,
            "observations": observations,
        })
    log("build gate: PASS")
    return EXIT_PASS

# ------------------------------------------------------------- config Gate

def gate_check_config(config: dict) -> int:
    """Validate configuration without touching the container."""
    print("configuration is valid")
    for key in ("TF_REPO", "IMAGE", "CONTAINER_NAME", "CANN_ROOT",
                "IN_CONTAINER_PYTHON", "CONTAINER_USER"):
        print("  %-20s %s" % (key, config.get(key, "")))
    recipe = Path(config["TF_REPO"]) / "docker" / "ascend950pr"
    print("  %-20s %s" % ("recipe", recipe))
    return EXIT_PASS

# ---------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="TileFoundry-on-Ascend Gates. All work runs in the container.",
    )
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--evidence-dir", default=None)
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    env_path = Path(args.env_file) if args.env_file else here.parent / "env.sh"
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None

    try:
        if args.check_config:
            return gate_check_config(load_config(env_path))

        config = load_config(env_path)
        if args.device:
            device_id = args.device_id or config.get("DEVICE_ID") or ""
            return gate_device(config, device_id, evidence_dir)
        if args.cpu:
            return gate_cpu(config, evidence_dir)
        if args.build:
            return gate_build(config, evidence_dir)

        # No subcommand: refuse rather than guess which Gate was meant.
        raise GateError(
            "no Gate selected. Choose one of:\n"
            "  --check-config   validate the configuration\n"
            "  --build          build the image and create the container\n"
            "  --cpu            the CPU Gate (import, analyze, finalize, tools)\n"
            "  --device         the device Gate (root, explicit device id)"
        )
    except GateError as error:
        log(str(error))
        return error.code
    except subprocess.TimeoutExpired as error:
        log("timed out: %s" % error)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())