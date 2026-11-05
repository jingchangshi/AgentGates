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

#: Configuration keys this adapter accepts. TF_REQUIRE_EXCLUSIVE and
#: TF_ALLOW_BUSY_DEVICE are listed ONLY so an older file still parses: they are
#: retired, they are read by nothing below, and no value of theirs can select,
#: skip, refuse, downgrade, or retry anything. Removing them from this set would
#: reject old files with a confusing "unknown key", which is not the same as
#: enforcing the shared-card policy.
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
        env: dict | None = None, echo_stdout: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(argv))
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False, env=env,
    )
    if completed.stdout and echo_stdout:
        print(completed.stdout, end="", file=sys.stderr)
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
                   env: dict | None = None, timeout: int = 1800,
                   echo_stdout: bool = True) -> subprocess.CompletedProcess:
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
        return run(list(argv), timeout=timeout, env=merged, echo_stdout=echo_stdout)
    command = ["docker", "exec"]
    if user is not None:
        command += ["-u", user]
    for key, value in (env or {}).items():
        command += ["-e", "%s=%s" % (key, value)]
    command += [config["CONTAINER_NAME"]] + argv
    return run(command, timeout=timeout, echo_stdout=echo_stdout)

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
            log("PASS %s" % name)
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
    print(json.dumps(observations, sort_keys=True))
    log("cpu gate: %s" % ("PASS" if failures == 0 else "FAIL"))
    return EXIT_PASS if failures == 0 else EXIT_FAIL

# ---------------------------------------------------------------- device Gate

def _sq(value: str) -> str:
    """Single-quote a value for sh, so a path with spaces or quotes is inert."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def parse_device_list(raw: str) -> list:
    """An ordered, duplicate-free list of nonnegative device ids."""
    if raw is None:
        raw = ""
    parts = [part.strip() for part in str(raw).split(",")]
    if not parts or any(not part for part in parts):
        raise GateError("device list is empty or has an empty item: %r" % (raw,))
    ids = []
    for part in parts:
        if not re.fullmatch(r"[0-9]+", part):
            raise GateError("device list item is not numeric: %r" % (part,))
        value = int(part)
        if value in ids:
            raise GateError("device list has duplicate id %d: %r" % (value, raw))
        ids.append(value)
    return ids


def device_assertion_prefix(expected: str, cann_set_env: str = "") -> str:
    """Assert the exact visible list and root INSIDE the payload.

    The list is passed through unchanged; no single-device narrowing is done
    here. The assertion writes to stderr so the workload stdout stays one JSON
    object. A mismatch, including an empty or unreadable list, exits 97 before
    torch_npu is imported.

    CANN is sourced first, and only when the configuration names a script that
    exists. torch_npu imports libhccl.so from it, so without this the workload
    fails with an import error that has nothing to do with the device it was
    asked for. A CANN path that is configured and missing is a hard refusal
    rather than a warning: silently importing without it would report a library
    fault as a device result.
    """
    source = ""
    if cann_set_env:
        source = (
            'if [ ! -f ' + _sq(cann_set_env) + ' ]; then '
            'echo "DEVICE-ENV-FAIL: CANN_SET_ENV=' + cann_set_env + ' does not exist" >&2; '
            'exit 97; fi; '
            '. ' + _sq(cann_set_env) + ' >&2 || { '
            'echo "DEVICE-ENV-FAIL: sourcing CANN_SET_ENV failed" >&2; exit 97; }; '
        )
    return (
        source +
        'v="$ASCEND_RT_VISIBLE_DEVICES"; '
        'u=$(id -u); '
        'echo "DEVICE-ASSERT: visible=$v uid=$u" >&2; '
        'if [ "$v" != "' + expected + '" ]; then '
        'echo "DEVICE-ASSERT-FAIL: payload sees ASCEND_RT_VISIBLE_DEVICES=$v, expected ' + expected + '" >&2; '
        'exit 97; fi; '
        'if [ "$u" -ne 0 ]; then '
        'echo "DEVICE-ASSERT-FAIL: device work must enter as root, payload uid is $u" >&2; '
        'exit 97; fi; '
        'exec "$@"'
    )


def parse_npu_smi(text: str):
    """Return (health by physical id, holders by physical id, process table seen).

    A table that cannot be parsed is reported as UNKNOWN, never as "no process".
    The previous revision picked the first numeric cell of every line across the
    whole dump, which merged the health table into the process table and could
    read a busy card as idle.
    """
    health = {}
    holders = {}
    in_process_table = False
    for line in text.splitlines():
        parts = [cell.strip() for cell in line.split("|")]
        if len(parts) >= 4 and re.fullmatch(r"[0-9]+", parts[1]) and parts[2].startswith("Ascend"):
            health[int(parts[1])] = parts[3]
        if "Process id" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        if len(parts) >= 3 and re.fullmatch(r"[0-9]+", parts[1]) and parts[2] != "Process id":
            holders.setdefault(int(parts[1]), []).append(line.strip())
    return health, holders, in_process_table


OCCUPANCY_TIMEOUT = 90


def device_occupancy(config: dict, device_ids: list):
    """Read occupancy ONCE, with one short bound; it is metadata, not a decision.

    There is no repeat sample, no wait, and no re-read until the card looks
    free. A failure to read is reported as an unreadable environment, not as an
    idle card, and a card that is not OK is a real health fault.
    """
    # npu-smi needs the CANN libraries on its own loader path. Reading occupancy
    # without them fails with a loader error that says nothing about the card,
    # so the same configured script is sourced here as for the workload. A
    # missing npu-smi after that is a real environment fault.
    cann = config.get("CANN_SET_ENV") or ""
    prefix = (". %s >/dev/null 2>&1; " % _sq(cann)) if cann else ""
    result = container_exec(config, ["sh", "-c",
                                     prefix + 'exec "$(command -v npu-smi || echo /usr/local/sbin/npu-smi)" info'],
                            user="root", timeout=OCCUPANCY_TIMEOUT, echo_stdout=False)
    if result.returncode != 0:
        raise GateError(
            "npu-smi failed with status %d; occupancy is UNKNOWN, not empty"
            % result.returncode, EXIT_BLOCKED)
    health, holders, table_seen = parse_npu_smi(result.stdout)
    if not health:
        raise GateError(
            "npu-smi produced no readable device health table; occupancy is "
            "UNKNOWN, not empty", EXIT_BLOCKED)
    for device_id in device_ids:
        state = health.get(device_id)
        if state is None:
            raise GateError(
                "npu-smi reports no row for device %d; the requested id does not "
                "match this machine" % device_id, EXIT_BLOCKED)
        if state != "OK":
            raise GateError(
                "device %d health is %r, not OK; a real environment fault"
                % (device_id, state), EXIT_BLOCKED)
    # A holder row PROVES the card is busy. The absence of one does NOT prove it
    # is idle: npu-smi inside this container sees the container's PID namespace,
    # and a measured container dump reported "no running processes" on device 0
    # while the host reported four processes on that same card. So an empty
    # table is recorded as UNKNOWN idle, never as idle.
    occupancy = {}
    for device_id in device_ids:
        rows = holders.get(device_id, [])
        occupancy[device_id] = {
            "occupied": True if rows else None,
            "idle_established": not rows and False,
            "holders": len(rows) if rows else None,
            "process_table": ("present" if table_seen else "UNKNOWN"),
            "scope": "container-pid-namespace",
            "why": (
                "a holder row proves occupancy; an empty table cannot prove idle "
                "because the container's npu-smi cannot see other tenants' processes"
            ),
            "process_rows": rows,
        }
    return occupancy, result.stdout


def gate_device(config: dict, device_raw: str, evidence_dir) -> int:
    """Run a real device workload on every requested visible device.

    Occupancy is READ and RECORDED per device. It never changes the selected
    set, the workload, the verdict, or a retry. All requested devices are
    exercised in one payload; there is no first-device-only shortcut.
    """
    assert_not_host("the device Gate")
    device_ids = parse_device_list(device_raw)
    if not container_exists(config):
        raise GateError(
            "container %s does not exist; run the build Gate first"
            % config["CONTAINER_NAME"], EXIT_BLOCKED)

    visible = ",".join(str(i) for i in device_ids)
    run_id = uuid.uuid4().hex[:12]
    repo = config["TF_REPO"]
    python = in_container_python(config)
    env = {
        "PYTHONPATH": "%s/src:%s/targets/ascend" % (repo, repo),
        "ASCEND_RT_VISIBLE_DEVICES": visible,
        "TF_DEVICE_AUTH": "1",
    }

    occupancy, occupancy_raw = device_occupancy(config, device_ids)
    for device_id in device_ids:
        item = occupancy[device_id]
        log("device %d occupancy: occupied=%s holders=%s idle_established=%s"
            " (RECORDED only; no refusal, no wait)"
            % (device_id, item["occupied"], item["holders"],
               item["idle_established"]))
        for row in item["process_rows"]:
            log("  " + row)
        if item["occupied"] is None:
            log("  unknown-idle: the container's npu-smi cannot see other tenants")

    exposed = container_exec(config, ["sh", "-c", "ls /dev/davinci* 2>/dev/null"],
                             user="root", echo_stdout=False)
    exposed_ids = []
    for token in exposed.stdout.split():
        match = re.fullmatch(r"/dev/davinci([0-9]+)", token)
        if match:
            exposed_ids.append(int(match.group(1)))
    missing = [i for i in device_ids if i not in exposed_ids]
    if missing:
        log("BLOCKED_ENV: requested device(s) %r are not bound into %s"
            % (missing, config["CONTAINER_NAME"]))
        log("  the container exposes: %r" % (exposed_ids,))
        return EXIT_BLOCKED

    payload = """
import json, os, torch, torch_npu
visible = [int(x) for x in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")]
rows = []
for logical, physical in enumerate(visible):
    torch.npu.set_device(logical)
    a = torch.ones((1024,), dtype=torch.float32, device="npu:%d" % logical)
    b = torch.full((1024,), 2.0, dtype=torch.float32, device="npu:%d" % logical)
    c = a + b
    torch.npu.synchronize()
    total = float(c.sum().item())
    rows.append({
        "logical": logical,
        "physical": physical,
        "device_name": torch.npu.get_device_name(logical),
        "count": int(c.numel()),
        "sum": total,
        "ok": bool(total == 3072.0 and int(c.numel()) == 1024),
    })
print(json.dumps({"devices": rows, "all_ok": all(row["ok"] for row in rows)}, sort_keys=True))
"""
    cann_set_env = config.get("CANN_SET_ENV") or ""
    argv = ["sh", "-c", device_assertion_prefix(visible, cann_set_env), "--",
            python, "-c", payload]
    log("--- device workload, exact list asserted inside the payload ---")
    probe = container_exec(config, argv, user="root", env=env, timeout=600,
                           echo_stdout=False)
    if probe.returncode == EXIT_ASSERT:
        log("FAIL: the in-payload device assertion failed; the workload did not run")
        return EXIT_FAIL
    if probe.returncode != 0:
        log("FAIL: the device workload exited %d" % probe.returncode)
        log(probe.stdout[-2000:])
        log(probe.stderr[-2000:])
        return EXIT_FAIL

    try:
        payload_result = json.loads(probe.stdout.strip().splitlines()[-1])
        rows = payload_result["devices"]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        log("FAIL: device workload did not emit one JSON result: %r" % (exc,))
        return EXIT_FAIL

    by_logical = sorted(rows, key=lambda row: int(row.get("logical", -1)))
    coverage = len(by_logical) == len(device_ids)
    identity_ok = coverage and all(
        int(row.get("logical", -1)) == index
        and int(row.get("physical", -1)) == device_ids[index]
        for index, row in enumerate(by_logical))
    workload_ok = coverage and all(bool(row.get("ok")) for row in by_logical)
    observations = {
        "exitCodeZero": True,
        "ranAsRoot": True,
        "multiDeviceCoverage": coverage,
        "assertionInsidePayload": True,
        "deviceBoundIntoContainer": True,
        "occupancyRecorded": True,
        "deviceWorkloadExecuted": workload_ok,
        "deviceIdentityRecorded": identity_ok,
    }
    result = {
        "run_id": run_id,
        "gate": "device",
        "requested_devices": device_ids,
        "visible_devices": visible,
        "logical_physical": {str(i): device_ids[i] for i in range(len(device_ids))},
        "container_exposed_davinci": exposed_ids,
        "devices": by_logical,
        "occupancy": {str(k): {"occupied": v["occupied"],
                                "holders": v["holders"],
                                "idle_established": v["idle_established"],
                                "scope": v["scope"]}
                      for k, v in occupancy.items()},
        "occupancy_raw_sha256": hashlib.sha256(occupancy_raw.encode()).hexdigest(),
        "observations": observations,
    }
    stdout_result = dict(observations)
    stdout_result.update({
        "requested_devices": device_ids,
        "visible_devices": visible,
        "logical_physical": result["logical_physical"],
        "devices": by_logical,
        "occupancy": result["occupancy"],
    })
    print(json.dumps(stdout_result, sort_keys=True))
    if evidence_dir is not None:
        write_evidence(evidence_dir, "device", {
            "run_id": run_id,
            "requested_devices": device_ids,
            "visible_devices": visible,
            "logical_physical": result["logical_physical"],
            "container_exposed_davinci": exposed_ids,
            "devices": by_logical,
            "occupancy": result["occupancy"],
            "occupancy_raw_sha256": result["occupancy_raw_sha256"],
            "observations": observations,
        }, run_id=run_id)
    if not (coverage and identity_ok and workload_ok):
        log("device gate: FAIL (coverage=%s identity=%s workload=%s)"
            % (coverage, identity_ok, workload_ok))
        return EXIT_FAIL
    log("device gate: PASS for %d requested device(s); occupancy is metadata only" % len(device_ids))
    return EXIT_PASS

# ------------------------------------------------------------------ evidence

def write_evidence(directory, name: str, record: dict, run_id: str | None = None) -> None:
    """Write one JSON record per Gate run."""
    import datetime
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["gate"] = name
    record["recorded_at"] = datetime.datetime.now().astimezone().isoformat()
    suffix = ("-" + run_id) if run_id else ""
    target = directory / ("gate-%s%s.json" % (name, suffix))
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
    print(json.dumps(observations, sort_keys=True))
    log("build gate: PASS")
    return EXIT_PASS

# ------------------------------------------------------------- config Gate

def gate_check_config(config: dict) -> int:
    """Validate configuration without touching the container."""
    out = {"configuration": "valid"}
    for key in ("TF_REPO", "IMAGE", "CONTAINER_NAME", "CANN_ROOT",
                "IN_CONTAINER_PYTHON", "CONTAINER_USER"):
        out[key] = config.get(key, "")
    out["recipe"] = str(Path(config["TF_REPO"]) / "docker" / "ascend950pr")
    print(json.dumps(out, sort_keys=True))
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
    parser.add_argument("--device-id", "--device-ids", dest="device_id", default=None,
                        help="ordered comma-separated visible device list, e.g. 0,1")
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
            device_raw = (args.device_id or config.get("DEVICE_ID")
                          or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or "0,1")
            return gate_device(config, device_raw, evidence_dir)
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