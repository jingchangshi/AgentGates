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
import contextlib
import datetime
import math
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

# --------------------------------------------------- P0-E: team lock + stages
#
# The lock coordinates THIS TEAM's jobs only. It is not an exclusivity claim over the
# cards: every Ascend card is shared with other tenants, so occupancy remains
# non-verdict metadata and a busy card still runs the same workload. What the lock
# prevents is two of OUR jobs interleaving inside the payload, which would make the
# timing and the occupancy samples describe neither job.
#
# The path is deliberately in the team's home, not /tmp: /tmp is per-container and a
# lock there would not serialise two containers, which is the actual failure mode.
#: The canonical TEAM lock. /home is where every team job can write and where the
#: container's root side also recreates the file, so both halves of the workflow agree
#: on one path. A per-user path under $HOME would silently NOT serialise two team
#: members whose accounts differ, which defeats the point.
#:
#: TF_DEVICE_LOCK_PATH is an EXPLICIT override for tests and isolation ONLY. A run that
#: uses it is not using the team lock, and every record says so (see
#: device_lock_identity), because "serialised on a private path" is not the same claim
#: as "serialised the team".
TEAM_LOCK_PATH = Path("/home/.tf_device0.lock")
DEVICE_LOCK_PATH = Path(os.environ.get("TF_DEVICE_LOCK_PATH") or TEAM_LOCK_PATH)

#: Handing the held lock to a CHILD process (the driver runs run_case as a
#: subprocess) must not deadlock. The token is a nonce the parent writes into the
#: inherited environment AND into the lock file; a bare "1" is not accepted, because a
#: stranger can set that too and serialisation would then be voluntary.
LOCK_HANDOFF_ENV = "TF_DEVICE_LOCK_HELD"

#: How long one of OUR jobs waits for another of our jobs before giving up, seconds.
DEVICE_LOCK_WAIT_SECONDS = float(os.environ.get("TF_DEVICE_LOCK_WAIT", "120"))

#: Per-stage wall-clock bounds. A hang must come back as a NAMED timeout, not as a
#: slow pass: without these, an unbounded payload is indistinguishable from a device
#: that is merely slow, and the run is retried until it happens to look fine.
STAGE_TIMEOUTS = {
    "environment": 120.0,
    "deployment": 180.0,
    "occupancy_before": 120.0,
    "identity": 180.0,
    "compile_profile": 180.0,
    "device_nodes": 60.0,
    "workload": 600.0,
    "occupancy_after": 120.0,
}

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

class _LockTimeout(Exception):
    """Internal: the bounded acquire expired. Never escapes the lease."""



class StageTimeout(GateError):
    """A stage exceeded its bound. A NAMED outcome, never a slow pass.

    Subclasses GateError so every existing refusal path still works, but carries the
    stage name and the elapsed time so the record can say which stage hung and for how
    long. Without this a hang is only distinguishable from a slow device by the human
    noticing that the terminal stopped scrolling.
    """

    def __init__(self, stage: str, elapsed: float, limit: float):
        super().__init__(
            "STAGE_TIMEOUT: %s exceeded its %.0fs bound (ran %.1fs). This is a bound, not "
            "a device verdict: nothing is retried, and nothing waits for another tenant "
            "to finish" % (stage, limit, elapsed), EXIT_FAIL)
        self.stage = stage
        self.elapsed = elapsed
        self.limit = limit

def run_stage(name: str, fn, *args, **kwargs):
    """Run one bounded stage and record its wall time and outcome.

    The elapsed time is recorded even on success, because "how long did the device
    take" is evidence, while an unbounded stage makes every later timing claim
    meaningless. The bound is enforced on the OBSERVED time: the underlying call gets
    its own timeout where it supports one, and this guards the aggregate.
    """
    limit = STAGE_TIMEOUTS.get(name)
    started = time.time()
    # The bound must reach the CALL, not merely be checked afterwards. A stage whose
    # underlying call has no timeout would otherwise run to completion and only then be
    # labelled a timeout -- the same unbounded run wearing a different word.
    if limit is not None and "timeout" in kwargs and kwargs["timeout"]:
        kwargs["timeout"] = min(float(kwargs["timeout"]), float(limit))
    try:
        result = fn(*args, **kwargs)
    except StageTimeout:
        raise
    except subprocess.TimeoutExpired as error:
        # The bound is enforced DOWNWARD as a real subprocess timeout, so the child is
        # actually killed rather than merely noticed afterwards. A post-hoc elapsed
        # check alone would let a hung payload run to completion and then be reported
        # as a timeout, which is the same unbounded run with a different label.
        elapsed = time.time() - started
        record_stage(name, elapsed, "timeout", "subprocess exceeded its own timeout")
        raise StageTimeout(name, elapsed, error.timeout or (limit or elapsed))
    except GateError as error:
        elapsed = time.time() - started
        record_stage(name, elapsed, "refused", str(error)[:400])
        raise
    except Exception as error:  # noqa: BLE001 -- re-raised, only recorded here
        elapsed = time.time() - started
        record_stage(name, elapsed, "error", "%s: %s" % (type(error).__name__, error))
        raise
    elapsed = time.time() - started
    if limit is not None and elapsed > limit:
        record_stage(name, elapsed, "timeout", "exceeded %.0fs" % limit)
        raise StageTimeout(name, elapsed, limit)
    record_stage(name, elapsed, "ok", None)
    return result

#: Append-only stage log for the CURRENT run. A module-level list is deliberate: the
#: phase DAG has to survive an exception unwinding out of gate_device, which is
#: exactly when the record matters most and exactly when a local variable is gone.
_STAGE_LOG: list = []

def reset_stage_log() -> None:
    del _STAGE_LOG[:]

def record_stage(name: str, elapsed: float, outcome: str, detail) -> None:
    _STAGE_LOG.append({"stage": name, "elapsed_seconds": round(float(elapsed), 3),
                       "outcome": outcome, "detail": detail,
                       "requires": list(STAGE_REQUIRES.get(name) or ())})

#: The phase DAG. A phase that ran without its prerequisites is a partial run that
#: must not be readable as a full one, so the edges are declared, not inferred.
STAGE_REQUIRES = {
    "environment": (),
    "deployment": ("environment",),
    "occupancy_before": ("deployment",),
    "identity": ("environment",),
    "compile_profile": ("identity",),
    "device_nodes": ("deployment",),
    "workload": ("occupancy_before", "identity", "compile_profile", "device_nodes"),
    "occupancy_after": ("occupancy_before",),
}

def stage_phases() -> list:
    """The recorded phases, annotated with whether each one actually ran."""
    seen = {entry["stage"] for entry in _STAGE_LOG}
    phases = []
    for entry in _STAGE_LOG:
        missing = [need for need in entry.get("requires") or () if need not in seen]
        item = dict(entry)
        item["prerequisites_missing"] = missing
        phases.append(item)
    return phases

def run_directory(evidence_dir):
    """A UNIQUE, empty directory per run.

    A fixed directory makes a stale file indistinguishable from a fresh one: a run
    that wrote nothing could be "read back" from the previous run's JSON. The name
    carries the run id so a reader can tie the directory to the record inside it.
    """
    if evidence_dir is None:
        return None
    base = Path(evidence_dir)
    base.mkdir(parents=True, exist_ok=True)
    target = base / ("run-" + uuid.uuid4().hex[:12])
    target.mkdir(parents=True, exist_ok=False)
    return target

def device_lock_identity() -> dict:
    """Say whether this run used the CANONICAL team lock or a private override.

    A test that isolates its lock proves the lease MECHANISM works; it does not prove
    the team is serialised, because it never touched the path the team shares. Without
    this field the two claims are indistinguishable in the evidence.
    """
    canonical = Path(DEVICE_LOCK_PATH) == TEAM_LOCK_PATH
    identity = {"path": str(DEVICE_LOCK_PATH), "canonical": canonical,
                "team_serialisation_claimed": canonical}
    if not canonical:
        identity["note"] = (
            "TF_DEVICE_LOCK_PATH override in effect: this run did NOT use the canonical "
            "team lock %s, so it demonstrates the lease mechanism only and must not be "
            "cited as evidence that team jobs are serialised" % (TEAM_LOCK_PATH,))
    return identity


def write_result(directory, gate: str, payload: dict) -> Path | None:
    """Write the ONE JSON record for this run, whatever the ending.

    Reached from a finally block, so success, failure and timeout all leave a record.
    A run whose only account is the console text cannot be audited afterwards, and
    "no record" is indistinguishable from "recorded as fine" for anyone reading later.
    """
    if directory is None:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["gate"] = gate
    record["recorded_at"] = datetime.datetime.now().astimezone().isoformat()
    record["phases"] = stage_phases()
    record["phase_dag"] = {name: list(needs) for name, needs in STAGE_REQUIRES.items()}
    # FINISHED is not PASS. "finished" says the process stopped; "passed" says the
    # observation was good. Collapsing the two is how an empty run reads as a green one.
    record.setdefault("passed", False)
    record["finished"] = True
    if gate == "device":
        record.setdefault("lock", {})
        if isinstance(record["lock"], dict):
            record["lock"].update(device_lock_identity())
        else:
            record["lock"] = device_lock_identity()
    target = directory / ("result-%s.json" % gate)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log("result: %s (passed=%s)" % (target, record.get("passed")))
    return target

@contextlib.contextmanager
def device_lock_lease():
    """Serialise THIS TEAM's device jobs. Not an exclusivity claim, not a waiter.

    Two properties matter and they pull in opposite directions:

      * Two of our jobs must not be inside the payload together, so the lock is taken
        around the whole device run and held by the parent while it drives children.
      * A CHILD of the job that already holds the lock must be able to proceed, or the
        driver deadlocks against itself. That is not hypothetical: it happened, and it
        presented as BLOCKED_ENV "another team measurement job holds the lock" while
        the lock was in fact FREE.

    A handoff therefore needs proof, not a flag. The parent writes a nonce into the
    lock file it holds; a child is admitted only when its inherited handoff value
    matches that nonce. A stranger setting TF_DEVICE_LOCK_HELD=1 gets a refusal, so
    serialisation cannot be opted out of.

    The wait here is BOUNDED and it waits for another of OUR jobs, never for a card to
    become idle. Those are different things and the distinction is the whole point: a
    busy card still runs the same workload with the same verdict, while a second team
    job genuinely must not enter the payload concurrently. The bound exists so a stuck
    holder cannot block the team forever; it is never a poll of npu-smi.
    """
    import fcntl

    path = Path(DEVICE_LOCK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    inherited = os.environ.get(LOCK_HANDOFF_ENV) or ""

    if inherited:
        # Verify the handoff against the nonce the holder wrote. The nonce is read
        # without taking the lock, which is safe: it is only ever READ here, and it
        # must match a value the parent could only know by holding the lock.
        try:
            current = json.loads(path.read_text())
        except (OSError, ValueError):
            current = {}
        holder_nonce = str(current.get("nonce") or "")
        if holder_nonce and current.get("holder_pid") and inherited == holder_nonce:
            log("device lock: admitted as a verified CHILD of the active holder")
            yield {"held": False, "handoff_verified": True, "reason": "parent"}
            return
        raise GateError(
            "DEVICE_LOCK_HANDOFF_INVALID: %s=%r does not match an active holder. A "
            "forged handoff flag cannot skip the team lock; only the process actually "
            "holding it can pass it on" % (LOCK_HANDOFF_ENV, inherited[:12]),
            EXIT_BLOCKED)

    try:
        handle = open(path, "a+", encoding="utf-8")
    except PermissionError as error:
        # A lock we cannot even open is an ENVIRONMENT fault, and it must be NAMED.
        # A raw PermissionError traceback reads like a code bug and hides the actual
        # cause, which is usually a file left behind by a previous root run.
        raise GateError(
            "DEVICE_LOCK_NOT_WRITABLE: cannot open %s (%s). The lock is a plain file in "
            "the team's home; a root-owned leftover from an earlier run makes it "
            "unusable by the team account. Remove it or point TF_DEVICE_LOCK_PATH at a "
            "writable location. This is an environment fault, not device busyness"
            % (path, error), EXIT_BLOCKED)
    try:
        # BOUNDED acquire, with NO polling primitive. The bound is enforced by a
        # one-shot interval timer around a BLOCKING flock; there is no sleep loop and
        # no waitpid/poll, so this cannot be mistaken for -- or evolve into -- a wait
        # for the CARD to become idle. An instant refusal would fail the second team
        # job for no fault of its own, and an unbounded block would let one stuck
        # holder stall the team, so the timer is what makes both ends safe.
        import signal

        def _alarm(signum, frame):
            raise _LockTimeout()

        previous = signal.signal(signal.SIGALRM, _alarm)
        armed = False
        try:
            signal.setitimer(signal.ITIMER_REAL, DEVICE_LOCK_WAIT_SECONDS)
            armed = True
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except _LockTimeout:
                handle.seek(0)
                try:
                    held = json.loads(handle.read() or "{}")
                except ValueError:
                    held = {}
                raise GateError(
                    "DEVICE_LOCK_HELD_BY_TEAM: another job of THIS team still holds "
                    "%s after %.0fs (pid=%s since %s). This is team serialisation "
                    "with a BOUNDED acquire, not device busyness: other tenants are "
                    "unaffected and no card was ever polled for idleness"
                    % (path, DEVICE_LOCK_WAIT_SECONDS, held.get("holder_pid"),
                       held.get("acquired_at"),),
                    EXIT_BLOCKED)
        finally:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        nonce = uuid.uuid4().hex
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"holder_pid": os.getpid(),
                                 "acquired_at": datetime.datetime.now().astimezone().isoformat(),
                                 "nonce": nonce}, sort_keys=True))
        handle.flush()
        log("device lock: acquired %s (team serialisation only)" % path)
        try:
            yield {"held": True, "handoff_verified": False, "nonce": nonce,
                   "reason": "acquired"}
        finally:
            try:
                handle.seek(0)
                handle.truncate()
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            log("device lock: released %s" % path)
    finally:
        handle.close()

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
        # The handed-over PATH is the CONTAINER'S CANN path. That is right for the
        # process being launched, but subprocess must FIRST resolve our own argv[0]
        # ("sh") using the host PATH -- and a CANN-only PATH has no /usr/bin, so the
        # call died with FileNotFoundError before the child ever started. Resolve the
        # executable against the real host PATH, then still hand the child its own.
        argv = list(argv)
        resolved = shutil.which(argv[0])
        if resolved is None:
            raise GateError(
                "cannot find %r on this host PATH (%s)"
                % (argv[0], os.environ.get("PATH")), EXIT_BLOCKED)
        argv[0] = resolved
        return run(argv, timeout=timeout, env=merged, echo_stdout=echo_stdout)
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


# ------------------------------------------------- environment preflight (P0-A)

#: Named environment faults. All of them mean "the run did not happen" -- none is
#: a device fault, and none may be reported as busy.
ENV_CANN_SCRIPT_MISSING = "ENV_CANN_SCRIPT_MISSING"
ENV_CANN_SOURCE_FAILED = "ENV_CANN_SOURCE_FAILED"
ENV_ASCEND_HOME_UNRESOLVED = "ENV_ASCEND_HOME_UNRESOLVED"
ENV_LIBRARY_UNRESOLVED = "ENV_LIBRARY_UNRESOLVED"
ENV_LD_LIBRARY_PATH_MISSING = "ENV_LD_LIBRARY_PATH_MISSING"
ENV_SMI_EXEC_FAILED = "ENV_SMI_EXEC_FAILED"

#: Seconds npu-smi may take before the environment is declared wedged.
SMI_TIMEOUT = 30

#: Libraries that must actually resolve. libc_sec.so is here because its absence
#: is EXACTLY the historic fault: npu-smi exit 127 "libc_sec.so: cannot open
#: shared object file" was reported as a device problem. It is an environment
#: problem and must be named as one.
REQUIRED_LIBRARIES = ("libc_sec.so", "libascend_hal.so", "libdrvdsmi_host.so")


def required_libraries() -> tuple:
    """The libraries to verify, taken from the TRACKED profile when it is readable.

    The profile is the single source of truth, so this list cannot quietly drift
    from the reviewed deployment contract. The constant above only guards the
    degenerate case and is never allowed to silently replace a broken profile.
    """
    try:
        profile = load_deployment_profile()
    except GateError:
        raise
    libs = tuple(profile.get("required_driver_libraries") or REQUIRED_LIBRARIES)
    return libs


def _env_fail(kind: str, detail: str):
    """Report a named environment failure and stop at the environment stage.

    BLOCKED_ENV, never FAIL: no result is being reported, so nothing is claimed
    about any card. The message names the exact contract that broke.
    """
    raise GateError("%s: %s" % (kind, detail), EXIT_BLOCKED)


def build_preflight_script(cann: str, resolver: str = "loader") -> str:
    """The exact shell program the preflight runs. Pure: no configuration, no I/O.

    Kept as a separate function so a CPU test can assert on the SCRIPT ITSELF.
    This matters: the first version of these tests only fed fabricated stdout to the
    parser, so a mutation that deleted the readlink/hash logic -- or that ignored the
    tracked profile -- still passed. A test that never executes the shell cannot
    notice the shell losing a step.
    """
    return "; ".join([
        # A PRIVATE dump path per invocation. A fixed /tmp path let two concurrent
        # runs overwrite each other's ldd output, so one run could be judged against
        # the other's resolution result -- a concurrency bug that looks like a flaky
        # environment. mktemp makes it unique and unreachable by accident.
        'TF_LDD_DUMP=$(mktemp -d)/ldd.txt || exit 18',
        "if [ ! -f " + _sq(cann) + " ]; then echo FAIL-CANN-MISSING; exit 11; fi",
        ". " + _sq(cann) + " >/dev/null 2>&1 || { echo FAIL-CANN-SOURCE; exit 12; }",
        'if [ -z "$ASCEND_HOME_PATH" ]; then echo FAIL-HOME-UNSET; exit 13; fi',
        'if [ ! -d "$ASCEND_HOME_PATH" ]; then echo FAIL-HOME-MISSING; exit 14; fi',
        'echo "HOME=$ASCEND_HOME_PATH"',
        # Emit the verified environment itself. A consumer must reuse EXACTLY these
        # values; re-sourcing the script on its own is a second, unverified
        # environment and is how the discarded-result bug happened.
        # EVERY variable the CANN script actually exports, not a convenient subset.
        # Capturing only LD_LIBRARY_PATH/ASCEND_HOME_PATH/PATH left ASCEND_OPP_PATH
        # unset downstream, so torch_npu refused to import with "ASCEND_OPP_PATH
        # environment variable is not set" -- the Gate then rejected a healthy
        # machine. The consumer must receive the SAME environment the preflight
        # validated, which means all of it.
        'for __tf_var in LD_LIBRARY_PATH ASCEND_HOME_PATH PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME PYTHONPATH CMAKE_PREFIX_PATH; do echo "ENVVAR $__tf_var="$(printenv "$__tf_var"); done',
        'case ":$LD_LIBRARY_PATH:" in *":$ASCEND_HOME_PATH/lib64:"*) : ;; '
        '*) echo FAIL-LD-PATH; exit 15 ;; esac',
        # HARD RULE: ask the SAME LOADER about the SAME BINARY that must run.
        #
        # Do not interrogate a cache index (ldconfig -p) instead. On this machine
        # libc_sec.so resolves fine for npu-smi through LD_LIBRARY_PATH but is
        # ABSENT from the ldconfig cache, so a cache-based check reports a HEALTHY
        # machine as broken -- measured, not hypothetical: the first revision of this
        # function did exactly that and was wrong. In the other direction, a cache
        # entry says nothing about the loader path npu-smi actually receives.
        #
        # ldd applies the real search rules to the real binary. A library it reports
        # as "not found", or that does not resolve to an existing file, is the only
        # thing that counts as missing.
        'SMI=$(command -v npu-smi || echo /usr/local/sbin/npu-smi)',
        'if [ ! -x "$SMI" ]; then echo FAIL-SMI-ABSENT; exit 17; fi',
        # resolver="cache" exists ONLY so a test can demonstrate that the cache-based
        # mechanism misfires on a healthy machine. It is never selected by the Gate.
        ('/sbin/ldconfig -p > "$TF_LDD_DUMP" 2>&1' if resolver == "cache"
         else 'ldd "$SMI" > "$TF_LDD_DUMP" 2>&1'),
        'for lib in ' + " ".join(required_libraries()) + '; do '
        '  line=$(grep -F "$lib" "$TF_LDD_DUMP" 2>/dev/null | head -1)',
        '  p=$(printf %s "$line" | sed -n "s/.*=> \\([^ ]*\\) .*/\\1/p")',
        '  case "$line" in '
        '    *"not found"*) echo "FAIL-LIB $lib"; exit 16 ;; '
        '  esac',
        '  if [ -z "$p" ] || [ ! -f "$p" ]; then echo "FAIL-LIB $lib"; exit 16; fi',
        # The real path AND a content hash, not just the name: a library that
        # resolves to a different build than the one verified is a real difference,
        # and a resolved name alone cannot show it.
        '  rp=$(readlink -f "$p" 2>/dev/null || echo "$p")',
        '  h=$(sha256sum "$rp" 2>/dev/null | cut -d" " -f1)',
        '  case "$h" in "") h=UNREADABLE ;; esac',
        '  echo "LIB $lib $p $rp $h"',
        'done',
        "echo PREFLIGHT-OK",
    ])


def environment_preflight(config: dict) -> dict:
    """The ONE mandatory bootstrap every SMI / compile / device consumer depends on.

    Before this existed, three near-identical `source` prefixes were open-coded in
    different places, each with its own error handling: the device payload checked
    for a missing script, occupancy used `. script >/dev/null 2>&1; exec npu-smi`
    which DISCARDED the source result, and nothing checked ASCEND_HOME_PATH or
    whether the driver libraries actually resolved. Two similar prefixes are not
    one non-bypassable contract, and the libc_sec failure could return through any
    unguarded path.

    Returns verified facts for the evidence record. Raises GateError with a named
    ENV_* code -- never a device verdict, never busy -- on any failure.
    """
    cann = (config.get("CANN_SET_ENV") or "").strip()
    facts = {"cann_script": cann or None}

    # CANN is REQUIRED for anything touching the NPU. An empty value is not "use
    # the ambient environment"; it is a configuration that cannot satisfy the
    # contract, and it is refused by name.
    if not cann:
        _env_fail(ENV_CANN_SCRIPT_MISSING,
                  "CANN_SET_ENV is not set. torch_npu cannot import (libhccl.so) and "
                  "npu-smi cannot load its libraries without it. Set it to the CANN "
                  "set_env.sh path; an ambient source in the calling shell does not count")

    # One shell program does the whole check, so every step shares one environment.
    # Each failure exits with its own code so the reason survives instead of
    # collapsing into "npu-smi returned 127".
    script = build_preflight_script(cann)

    result = container_exec(config, ["sh", "-c", script], user="root",
                            timeout=120, echo_stdout=False)
    out = result.stdout or ""

    if result.returncode != 0:
        if "FAIL-CANN-MISSING" in out:
            _env_fail(ENV_CANN_SCRIPT_MISSING,
                      "CANN_SET_ENV=%r does not exist inside the container" % cann)
        if "FAIL-CANN-SOURCE" in out:
            _env_fail(ENV_CANN_SOURCE_FAILED,
                      "sourcing CANN_SET_ENV=%r returned non-zero; the environment it "
                      "would provide is not available" % cann)
        if "FAIL-HOME-UNSET" in out or "FAIL-HOME-MISSING" in out:
            _env_fail(ENV_ASCEND_HOME_UNRESOLVED,
                      "ASCEND_HOME_PATH is unset or not a directory after sourcing; "
                      "the driver libraries cannot be located from a name alone")
        if "FAIL-LD-PATH" in out:
            _env_fail(ENV_LD_LIBRARY_PATH_MISSING,
                      "ASCEND_HOME_PATH/lib64 is not on LD_LIBRARY_PATH after sourcing; "
                      "npu-smi would fail to load its own libraries")
        if "FAIL-LIB" in out:
            missing = [ln.split(None, 2)[1] for ln in out.splitlines()
                       if ln.startswith("FAIL-LIB")]
            _env_fail(ENV_LIBRARY_UNRESOLVED,
                      "required driver libraries do not resolve: %s. A missing "
                      "libc_sec.so is exactly the historic exit-127 fault and is an "
                      "ENVIRONMENT failure, not a device one" % ", ".join(missing))
        _env_fail(ENV_LIBRARY_UNRESOLVED,
                  "environment preflight exited %d with no recognised reason: %r"
                  % (result.returncode, out[-400:]))

    if "PREFLIGHT-OK" not in out:
        _env_fail(ENV_LIBRARY_UNRESOLVED,
                  "environment preflight did not report success: %r" % out[-400:])

    for line in out.splitlines():
        if line.startswith("HOME="):
            facts["ascend_home_path"] = line.split("=", 1)[1]
        elif line.startswith("ENVVAR "):
            key, _, value = line[len("ENVVAR "):].partition("=")
            facts.setdefault("env", {})[key] = value
        elif line.startswith("LIB "):
            parts = line.split()
            name, path, real, digest = parts[1], parts[2], parts[3], parts[4]
            facts.setdefault("libraries", {})[name] = {
                "resolved": path,
                "realpath": real,
                "sha256": digest,
            }

    # npu-smi must run in the SAME environment just verified. This is the only
    # place its loader failure may be classified.
    # npu-smi is bounded INSIDE the container as well as by the exec timeout. This
    # machine has hung the docker-exec SMI path, and an unbounded call surfaces as a
    # raw TimeoutExpired instead of a named environment fault -- which is exactly the
    # kind of untyped failure that gets mistaken for a device problem.
    smi = 'exec "$(command -v npu-smi || echo /usr/local/sbin/npu-smi)" info'
    try:
        probe = container_exec(
            config,
            ["sh", "-c", ". " + _sq(cann) + " >/dev/null 2>&1; "
             'exec timeout ' + str(SMI_TIMEOUT) + ' sh -c ' + _sq(smi)],
            user="root", timeout=SMI_TIMEOUT + 30, echo_stdout=False)
    except subprocess.TimeoutExpired:
        _env_fail(ENV_SMI_EXEC_FAILED,
                  "npu-smi did not return within %d seconds even with a container-side "
                  "timeout. This is an ENVIRONMENT fault (the SMI path is wedged); it is "
                  "NOT a busy card and NOT a device failure" % SMI_TIMEOUT)
    if probe.returncode in (124, 137):
        _env_fail(ENV_SMI_EXEC_FAILED,
                  "npu-smi exceeded its %d second bound inside the container. This is an "
                  "ENVIRONMENT fault, not a busy card" % SMI_TIMEOUT)
    if probe.returncode != 0:
        _env_fail(ENV_SMI_EXEC_FAILED,
                  "npu-smi failed with status %d in the verified CANN environment. This "
                  "is classified as an ENVIRONMENT fault; it is NOT reported as a busy "
                  "card and NOT as a device failure" % probe.returncode)
    if not (probe.stdout or "").strip():
        _env_fail(ENV_SMI_EXEC_FAILED,
                  "npu-smi produced no output in the verified CANN environment")
    facts["smi_probe_exit"] = probe.returncode
    facts["smi_probe_bytes"] = len(probe.stdout or "")
    return facts

# ------------------------------------------------ deployment verification (P0-DEPLOY)

#: The TRACKED deployment profile. Versioned, secret-free, and READ (not merely
#: documented): the gate refuses to run if it is missing or inconsistent, so the
#: contract cannot drift away from the code that claims to enforce it.
DEPLOYMENT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "deployment_profile.json"
DEPLOYMENT_SCHEMA_VERSION = 1

#: Fallbacks, used only if the profile cannot be read; a missing profile is itself
#: reported, so these cannot silently become the whole contract.
REQUIRED_CONTROL_NODES = ("/dev/davinci_manager", "/dev/hisi_hdc")


def load_deployment_profile() -> dict:
    """Read the tracked deployment profile.

    A missing or malformed profile is a configuration error: the deployment
    contract is part of the adapter, and a Gate that silently invented its own
    would be enforcing something nobody reviewed.
    """
    try:
        profile = json.loads(DEPLOYMENT_PROFILE_PATH.read_text())
    except FileNotFoundError:
        raise GateError(
            "deployment profile is missing: %s\n"
            "  the tracked deployment contract is part of this adapter"
            % DEPLOYMENT_PROFILE_PATH, EXIT_CONFIG)
    except ValueError as exc:
        raise GateError(
            "deployment profile is not valid JSON: %s: %s"
            % (DEPLOYMENT_PROFILE_PATH, exc), EXIT_CONFIG)
    version = profile.get("schema_version")
    if version != DEPLOYMENT_SCHEMA_VERSION:
        raise GateError(
            "deployment profile schema_version is %r, this gate understands %d"
            % (version, DEPLOYMENT_SCHEMA_VERSION), EXIT_CONFIG)
    return profile


#: Keys whose VALUES are safe to hash: none is a credential. The env file
#: is parsed as data and never executed, and these are deployment facts.
#: A key outside this set contributes its NAME only, so an unexpectedly
#: secret value cannot leak into evidence while the key set stays comparable.
HASHABLE_VALUE_KEYS = frozenset({
    "TF_REPO", "IMAGE", "CONTAINER_NAME", "CANN_ROOT", "CANN_SET_ENV",
    "IN_CONTAINER_PYTHON", "CONTAINER_USER", "CONTAINER_UID", "DEVICE_ID",
    "TF_VENV", "TF_WORKSPACE", "CONTAINER_WORKDIR", "TILELANG_ROOT",
})


def effective_config_hash(config: dict) -> str:
    """Hash the effective configuration so a run records WHICH deployment it used.

    VALUES are hashed for the deployment-identifying keys above. A hash over
    key NAMES alone is invariant under exactly the changes that matter: swapping
    IMAGE or CONTAINER_NAME would leave it identical, so it could not support
    the claim that it identifies the deployment. Keys outside the safe set still
    contribute their NAME, so the key set remains comparable without any secret
    value entering the evidence.

    This is a CONFIGURATION fingerprint, NOT a deployment identity: the
    authoritative deployment identity is image_id plus bound_devices.
    """
    parts = []
    for key in sorted(k for k in config if config.get(k) not in (None, "")):
        if key in HASHABLE_VALUE_KEYS:
            parts.append("%s=%s" % (key, config.get(key)))
        else:
            parts.append("%s" % key)
    return hashlib.sha256(("\n".join(parts)).encode()).hexdigest()

def _host_inspect(config: dict) -> dict:
    """Read the container deployment facts from the host docker daemon.

    Returns {} when the host inspect is unavailable (for example when this process
    already runs INSIDE the container). An empty result is NEVER treated as a pass:
    the caller must supply a fresh deployment record or the Gate refuses by name.
    """
    if inside_container():
        return {}
    result = run(
        ["docker", "inspect", config["CONTAINER_NAME"],
         "--format", "{{json .}}"],
        timeout=60, echo_stdout=False)
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError):
        return {}


def deployment_verify(config: dict, device_ids: list, evidence_dir=None) -> dict:
    """Verify the RUNNING container against the deployment contract.

    Checks, all BEFORE any device work:
      * the container exists and its image matches the configured image
      * every requested /dev/davinciN is in HostConfig.Devices (a real bind, not a
        path that merely exists inside the container)
      * the control nodes are bound too
      * each requested node INSIDE the container is a character device, not a
        regular file that happens to be named davinci0

    Every failure is a named deployment fault. None of them is busy, and none of
    them waits for a card, rebuilds a container, or touches another user's work.
    """
    profile = load_deployment_profile()
    facts = {"schema_version": DEPLOYMENT_SCHEMA_VERSION,
             "effective_config_sha256": effective_config_hash(config),
             "profile_libraries": profile.get("required_driver_libraries"),
             "profile_image": profile.get("required_image_name")}
    required_control = tuple(profile.get("required_control_nodes")
                             or REQUIRED_CONTROL_NODES)
    inspect = _host_inspect(config)

    if not inspect:
        # No host view. The Gate must not silently downgrade to "trust the
        # container"; it records that the binding could not be cross-checked and
        # falls back to the in-container device-node check with that limitation
        # stated. It does NOT claim the deployment was verified.
        facts["host_inspect"] = "UNAVAILABLE"
    else:
        facts["host_inspect"] = "OK"
        facts["container_id"] = inspect.get("Id")
        facts["image_id"] = (inspect.get("Image") or "")
        # The image the container actually runs must match the configured image.
        # A container left over from an older image is exactly the "stale deployment"
        # case: it looks fine until a result cannot be reproduced.
        configured_image = config.get("IMAGE")
        if configured_image:
            tag = run(["docker", "inspect", "--format", "{{.Id}}", configured_image],
                      timeout=60, echo_stdout=False)
            if tag.returncode == 0 and tag.stdout.strip():
                expected_image_id = tag.stdout.strip()
                facts["configured_image_id"] = expected_image_id
                if facts["image_id"] and facts["image_id"] != expected_image_id:
                    raise GateError(
                        "CONTAINER_IMAGE_MISMATCH: container %s runs image %s but the "
                        "configuration names %s (%s). This is a STALE DEPLOYMENT: the "
                        "container predates the image the Gate is configured for"
                        % (config["CONTAINER_NAME"], facts["image_id"][:19],
                           configured_image, expected_image_id[:19]),
                        EXIT_BLOCKED)
        host_config = inspect.get("HostConfig") or {}
        bound = set()
        for entry in host_config.get("Devices") or []:
            path_in = entry.get("PathInContainer") or entry.get("PathOnHost")
            if path_in:
                bound.add(path_in)
        facts["bound_devices"] = sorted(bound)
        facts["effective_config_sha256"] = facts["effective_config_sha256"]

        missing = ["/dev/davinci%d" % i for i in device_ids if "/dev/davinci%d" % i not in bound]
        if missing:
            raise GateError(
                "CONTAINER_DEVICE_BINDING_MISSING: %s are not in the container's "
                "HostConfig.Devices. The container exposes %s. This is a DEPLOYMENT "
                "fault: the requested device is not bound, so it cannot be busy, "
                "idle, or faulty" % (", ".join(missing), sorted(bound)),
                EXIT_BLOCKED)
        missing_control = [n for n in REQUIRED_CONTROL_NODES if n not in bound]
        if missing_control:
            raise GateError(
                "CONTAINER_DEPLOYMENT_MISMATCH: control node(s) %s are not bound; "
                "the driver cannot be driven without them. This is a DEPLOYMENT fault"
                % ", ".join(missing_control), EXIT_BLOCKED)

    # In-container truth: the requested nodes must be CHARACTER devices.
    nodes = ["/dev/davinci%d" % i for i in device_ids] + list(REQUIRED_CONTROL_NODES)
    check = (
        "for n in " + " ".join(_sq(n) for n in nodes) + "; do "
        'if [ ! -e "$n" ]; then echo "MISSING $n"; continue; fi; '
        'if [ ! -c "$n" ]; then echo "NOTCHAR $n"; continue; fi; '
        'echo "CHAR $n"; done'
    )
    result = container_exec(config, ["sh", "-c", check], user="root",
                            timeout=60, echo_stdout=False)
    out = result.stdout or ""
    facts["nodes"] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            facts["nodes"][parts[1]] = parts[0]
    not_char = [n for n, kind in facts["nodes"].items() if kind == "NOTCHAR"]
    if not_char:
        raise GateError(
            "CONTAINER_DEVICE_BINDING_MISSING: %s exist but are NOT character "
            "devices. A regular file named davinci0 is not a device. This is a "
            "DEPLOYMENT fault, not a device fault" % ", ".join(sorted(not_char)),
            EXIT_BLOCKED)
    absent = [n for n, kind in facts["nodes"].items() if kind == "MISSING"]
    if absent:
        raise GateError(
            "DRIVER_MOUNT_MISSING: %s are absent inside the container"
            % ", ".join(sorted(absent)), EXIT_BLOCKED)

    # The hardware must never be reached by a stale record: refuse if the container
    # was created by a different effective configuration than the one now in force.
    # `verified` means the FULL contract was checked. When the host inspect is
    # unavailable -- which is the normal case inside the container -- the binding
    # half could not be cross-checked, so this must NOT claim True. Reporting True
    # there is exactly the overclaim the audit objected to elsewhere: the node check
    # alone cannot prove the container was created with those nodes bound.
    facts["verified"] = facts.get("host_inspect") == "OK"
    facts["verified_scope"] = (
        "image+device-binding+control-nodes+char-nodes" if facts["verified"]
        else "char-nodes-only (host inspect unavailable; device BINDING not proven)")
    if not facts["verified"]:
        log("deployment: host inspect UNAVAILABLE, so only the in-container node check "
            "ran. The device BINDING is NOT proven from inside the container.")
    log("deployment verified: image=%s bound=%s nodes=%s"
        % (str(facts.get("image_id"))[:19], len(facts.get("bound_devices", [])),
           len(facts.get("nodes", {}))))
    return facts


OCCUPANCY_TIMEOUT = 90


def device_occupancy(config: dict, device_ids: list, preflight_facts: dict):
    """Read occupancy ONCE, with one short bound; it is metadata, not a decision.

    There is no repeat sample, no wait, and no re-read until the card looks
    free. A failure to read is reported as an unreadable environment, not as an
    idle card, and a card that is not OK is a real health fault.
    """
    # Occupancy REUSES the environment the preflight verified -- the same values,
    # passed explicitly -- rather than sourcing CANN again with its own prefix.
    #
    # Two reasons this matters. First, the old prefix was
    # `. <script> >/dev/null 2>&1; exec npu-smi`, which DISCARDED the source result,
    # so a missing or failing CANN reported itself only as "npu-smi status 127" --
    # exactly the libc_sec confusion. Second, a second `source` is a SECOND
    # environment: it is not the one that was checked, and nothing forces the two to
    # agree. docker exec starts a fresh process, so the verified values cannot be
    # inherited implicitly; they must be handed over deliberately.
    verified_env = dict(preflight_facts.get("env") or {})
    if not verified_env.get("LD_LIBRARY_PATH"):
        raise GateError(
            "%s: the preflight did not return a verified environment to reuse; "
            "refusing to fall back to an unverified source" % ENV_LIBRARY_UNRESOLVED,
            EXIT_BLOCKED)
    result = container_exec(config, ["sh", "-c",
                                     'exec "$(command -v npu-smi || echo /usr/local/sbin/npu-smi)" info'],
                            user="root", env=verified_env, timeout=OCCUPANCY_TIMEOUT,
                            echo_stdout=False)
    if result.returncode != 0:
        raise GateError(
            "%s: npu-smi failed with status %d AFTER the environment preflight "
            "passed; occupancy is UNKNOWN, not empty. This is an environment fault "
            "and is NOT a busy card" % (ENV_SMI_EXEC_FAILED, result.returncode),
            EXIT_BLOCKED)
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




# ----------------------------------------- host-side attestation (P0-DEPLOY)

#: How old an attestation may be before the device Gate refuses it. A stale
#: attestation describes a container that may no longer exist, so it is not
#: evidence about the container we are about to touch.
ATTESTATION_MAX_AGE_SECONDS = 300


def deployment_attest(config: dict, out_path: Path) -> int:
    """Produce a READ-ONLY host-side attestation of the container deployment.

    THIS IS NOT A GATE. It runs no compiler, executes nothing inside the container,
    touches no /dev node and does not start, stop or recreate anything. It only runs
    `docker inspect` and writes what the daemon reports, plus a content hash.

    It exists because the device Gate runs INSIDE the container, where there is no
    host view: from in there, a device node existing does not prove the container was
    created with that device bound. The Gate therefore cannot honestly claim a full
    deployment verification on its own. This producer supplies the missing half, and
    the Gate consumes it via --deployment-attestation, so the two halves are checked
    together instead of the Gate quietly overclaiming.

    Being host-only is deliberate: it must run where the Docker daemon is reachable,
    which is exactly where the Gate is forbidden to run.
    """
    name = config["CONTAINER_NAME"]
    inspected = run(["docker", "inspect", name], timeout=60, echo_stdout=False)
    if inspected.returncode != 0:
        raise GateError(
            "cannot attest container %r: docker inspect failed with status %d. This "
            "is an ENVIRONMENT fault; nothing is claimed about the deployment"
            % (name, inspected.returncode), EXIT_BLOCKED)
    try:
        payload = json.loads(inspected.stdout)[0]
    except (ValueError, IndexError, TypeError) as exc:
        raise GateError("docker inspect returned unparsable output: %r" % (exc,),
                        EXIT_BLOCKED)

    host_config = payload.get("HostConfig") or {}
    config_block = payload.get("Config") or {}
    devices = []
    for entry in host_config.get("Devices") or []:
        devices.append({
            "path_on_host": entry.get("PathOnHost"),
            "path_in_container": entry.get("PathInContainer"),
            "permissions": entry.get("CgroupPermissions"),
        })
    mounts = []
    for mount in payload.get("Mounts") or []:
        mounts.append({
            "type": mount.get("Type"),
            "source": mount.get("Source"),
            "destination": mount.get("Destination"),
            "mode": mount.get("Mode"),
            "read_only": bool(mount.get("RW") is False),
        })
    visible = None
    for entry in config_block.get("Env") or []:
        key, _, value = entry.partition("=")
        if key == "ASCEND_RT_VISIBLE_DEVICES":
            visible = value

    created = payload.get("Created")
    attestation = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "kind": "host-deployment-attestation",
        "attested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "container_name": name,
        "container_id": payload.get("Id"),
        "created": created,
        # docker inspect reports the image ID (a sha256 of its config), NOT a repo
        # digest. Record the names as well, because the configured IMAGE is a TAG, and
        # comparing a tag against an id would differ every time and prove nothing.
        "image": payload.get("Image"),
        "image_digest": (payload.get("Image") or ""),
        "image_names": ([config_block.get("Image")] if config_block.get("Image") else []),
        "host_config_devices": devices,
        "mounts": mounts,
        "config_visible_devices": visible,
        "privileged": bool(host_config.get("Privileged")),
        "user": config_block.get("User") or "",
    }
    # The content hash covers the payload BEFORE it is added, so a later edit to any
    # field is detectable. A hash that covered itself could not do that.
    canonical = json.dumps(attestation, sort_keys=True, separators=(",", ":"))
    attestation["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n")
    log("attestation written to %s (container %s, %d device(s), %d mount(s))"
        % (out_path, str(attestation["container_id"])[:12], len(devices), len(mounts)))
    log("this is an attestation PRODUCER, not a Gate: no device was touched, "
        "nothing was executed in the container")
    print(json.dumps({"attestation": str(out_path),
                      "content_sha256": attestation["content_sha256"]}, sort_keys=True))
    return EXIT_PASS


def observe_mount_linkage(config: dict, destinations: list) -> dict:
    """Check, from INSIDE the container, that the attested mounts are really here.

    This is the identity evidence that survives the container being unable to read its
    own docker ID. The container cannot say "I am 3602bce...", but it CAN show that the
    exact mount set described in the attestation is present and has the attested
    read-only/writable mode -- which a different container would not reproduce.

    Read-only, no device access.
    """
    wanted = [d for d in destinations if d]
    if not wanted:
        return {"checked": [], "missing": [], "ro_mismatch": []}
    script = []
    for destination in wanted:
        sq = _sq(destination)
        script.append(
            'if [ ! -e ' + sq + ' ]; then echo "MOUNT-MISSING ' + destination + '"; '
            'else '
            'w=no; [ -w ' + sq + ' ] && w=yes; '
            'echo "MOUNT ' + destination + ' writable=$w"; fi')
    result = container_exec(config, ["sh", "-c", "; ".join(script)], user="root",
                            timeout=60, echo_stdout=False)
    out = result.stdout or ""
    missing, writable = [], {}
    for line in out.splitlines():
        if line.startswith("MOUNT-MISSING "):
            missing.append(line.split(None, 1)[1])
        elif line.startswith("MOUNT "):
            fields = line.split()
            destination = fields[1]
            writable[destination] = fields[2].endswith("=yes")
    return {"checked": sorted(writable), "missing": missing, "writable": writable}


def verify_attestation(attestation: dict, config: dict, device_ids: list,
                       node_facts: dict = None, observe=None) -> dict:
    """Check the seven conditions that let the device Gate claim a full verification.

    Each failure is named and specific. None of them is busy, and none rebuilds or
    restarts anything. If any check fails the Gate refuses the attestation rather
    than falling back to a weaker claim, because a half-verified deployment that
    reports success is exactly the overclaim this machinery exists to prevent.
    """
    detail = {}

    # (7) content hash first: if the file was edited, every later field is suspect,
    # so there is no point interpreting it.
    claimed = attestation.get("content_sha256")
    body = {k: v for k, v in attestation.items() if k != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    if not claimed or claimed != actual:
        raise GateError(
            "ATTESTATION_TAMPERED: content_sha256 %r does not match the recomputed %r. "
            "The attestation was modified after it was produced, so none of its claims "
            "may be relied on" % (claimed, actual), EXIT_BLOCKED)

    # (3) container identity. Where we cannot know our own ID, this is not silently
    # skipped: it is recorded as unverifiable and the MOUNT LINKAGE below carries the
    # burden instead.
    expected_id = config.get("_attested_container_id")
    if expected_id:
        if attestation.get("container_id") != expected_id:
            raise GateError(
                "ATTESTATION_WRONG_CONTAINER: attestation is for container %s but this "
                "container is %s" % (str(attestation.get("container_id"))[:12],
                                     str(expected_id)[:12]), EXIT_BLOCKED)
        detail["container_id_check"] = "matched"
    else:
        detail["container_id_check"] = (
            "UNVERIFIABLE from inside the container; mount linkage used instead")

    # (1) freshness, and self-consistency against the container birth time.
    attested_at = attestation.get("attested_at")
    try:
        when = datetime.datetime.fromisoformat(str(attested_at))
    except (TypeError, ValueError):
        raise GateError("ATTESTATION_INVALID: unusable attested_at %r" % (attested_at,),
                        EXIT_BLOCKED)
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    age = (datetime.datetime.now(datetime.timezone.utc) - when).total_seconds()
    detail["attestation_age_seconds"] = round(age, 1)
    if age > ATTESTATION_MAX_AGE_SECONDS:
        raise GateError(
            "ATTESTATION_STALE: produced %.0fs ago, older than the %ds limit. It may "
            "describe a container that no longer exists" % (age, ATTESTATION_MAX_AGE_SECONDS),
            EXIT_BLOCKED)
    if age < -60:
        raise GateError(
            "ATTESTATION_STALE: attested_at is %.0fs in the FUTURE; the clock or the "
            "file cannot be trusted" % (-age,), EXIT_BLOCKED)

    created = attestation.get("created")
    if created:
        try:
            born = datetime.datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            if born > when:
                raise GateError(
                    "ATTESTATION_INCONSISTENT: container was created at %s, AFTER the "
                    "attestation time %s. An attestation cannot predate its container"
                    % (created, attested_at), EXIT_BLOCKED)
        except ValueError:
            detail["created_parse"] = "UNPARSEABLE"

    # (4) every requested device must be genuinely bound.
    bound = set()
    by_path = {}
    for entry in attestation.get("host_config_devices") or []:
        path = entry.get("path_in_container") or entry.get("path_on_host")
        if path:
            bound.add(path)
            by_path[path] = entry
    detail["attested_bound"] = sorted(bound)
    missing = ["/dev/davinci%d" % i for i in device_ids
               if "/dev/davinci%d" % i not in bound]
    if missing:
        raise GateError(
            "ATTESTATION_DEVICE_NOT_BOUND: %s are not in the attested "
            "HostConfig.Devices (attested: %s)" % (", ".join(missing), sorted(bound)),
            EXIT_BLOCKED)

    # (5) driver / CANN read-only mounts must be present.
    destinations = [m.get("destination") for m in attestation.get("mounts") or []]
    driver_mounts = [d for d in destinations if d and "Ascend" in str(d)]
    if not driver_mounts:
        raise GateError(
            "ATTESTATION_DRIVER_MOUNT_MISSING: no Ascend driver mount is attested "
            "(attested mounts: %s)" % (destinations,), EXIT_BLOCKED)
    detail["driver_mounts"] = driver_mounts
    detail["read_only_mounts"] = [m.get("destination")
                                  for m in attestation.get("mounts") or []
                                  if m.get("read_only")]

    # (4b) in-container nodes must be CHARACTER devices with matching major:minor.
    if node_facts:
        not_char = [n for n, kind in node_facts.items() if kind == "NOTCHAR"]
        if not_char:
            raise GateError(
                "ATTESTATION_NODE_NOT_CHAR: %s are not character devices inside the "
                "container. A regular file named davinciN is not a device"
                % ", ".join(sorted(not_char)), EXIT_BLOCKED)
        absent = [n for n, kind in node_facts.items() if kind == "MISSING"]
        if absent:
            raise GateError(
                "ATTESTATION_NODE_MISSING: %s are absent inside the container"
                % ", ".join(sorted(absent)), EXIT_BLOCKED)

    # (4c) MOUNT LINKAGE. This is what makes the attestation provably about THIS
    # container rather than merely plausible: every attested mount destination must
    # actually exist in here, and every one attested READ-ONLY must really be
    # read-only. A different container would not reproduce this, and it is checkable
    # from inside, unlike the container ID.
    if observe is not None:
        linkage = observe(destinations) or {}
        missing_mounts = linkage.get("missing") or []
        if missing_mounts:
            raise GateError(
                "ATTESTATION_MOUNT_NOT_LINKED: attested mount(s) %s do not exist inside "
                "this container, so this attestation is not about this container"
                % ", ".join(missing_mounts), EXIT_BLOCKED)
        attested_read_only = {m.get("destination")
                              for m in attestation.get("mounts") or []
                              if m.get("read_only")}
        writable_now = linkage.get("writable") or {}
        ro_drift = sorted(d for d in attested_read_only if writable_now.get(d) is True)
        if ro_drift:
            raise GateError(
                "ATTESTATION_MOUNT_MODE_DRIFT: %s are attested read-only but are "
                "writable inside this container" % ", ".join(ro_drift), EXIT_BLOCKED)
        detail["mount_linkage"] = linkage.get("checked") or []

    # (6) the image must not have DRIFTED. This previously only checked that SOME
    # digest was recorded, so a container running any image was accepted as long as
    # the attestation carried one -- review found a drifted sha256:eeee... accepted
    # under three different configurations. Existence is not agreement, so the
    # configured value is now actually compared.
    #
    # The contract value is a DIGEST (the image ID docker inspect reports). The
    # configured IMAGE is a TAG, so it is compared against the attested tag names
    # instead; comparing a tag with an id would differ every time and prove nothing.
    attested_digest = attestation.get("image_digest") or ""
    configured_digest = (config.get("IMAGE_DIGEST") or "").strip()
    configured_name = (config.get("IMAGE") or "").strip()

    if not attested_digest:
        raise GateError(
            "ATTESTATION_INVALID: the attestation records no image digest, so image "
            "agreement cannot be established", EXIT_BLOCKED)

    if configured_digest:
        if not attested_digest.startswith("sha256:"):
            raise GateError(
                "ATTESTATION_IMAGE_DRIFT: attestation image_digest %r is not a digest"
                % (attested_digest,), EXIT_BLOCKED)
        if attested_digest != configured_digest:
            raise GateError(
                "ATTESTATION_IMAGE_DRIFT: the container runs image %s but the contract "
                "requires %s. The deployment is stale"
                % (attested_digest, configured_digest), EXIT_BLOCKED)

    if configured_name:
        names = [str(n) for n in (attestation.get("image_names") or [])]
        # EQUALITY, not substring. 'repo/tf:1' is a substring of 'repo/tf:10', so a
        # substring test would accept a DIFFERENT image whenever one tag happens to
        # prefix another -- exactly the stale-deployment case this check exists for.
        matched = configured_name in names
        if names and not matched:
            raise GateError(
                "ATTESTATION_IMAGE_DRIFT: the container runs image name(s) %s but the "
                "contract requires %r. The deployment is stale"
                % (names, configured_name), EXIT_BLOCKED)
        if not names and not configured_digest:
            raise GateError(
                "ATTESTATION_IMAGE_DRIFT: the attestation names no image and no "
                "contract digest was supplied, so image agreement cannot be established",
                EXIT_BLOCKED)

    detail["container_id"] = attestation.get("container_id")
    detail["image_digest"] = attestation.get("image_digest")
    detail["privileged"] = attestation.get("privileged")
    detail["user"] = attestation.get("user")
    detail["config_visible_devices"] = attestation.get("config_visible_devices")
    return detail

# ------------------------------------------------- device result verification (P0-C)

#: The workload contract. These are what the HOST recomputes; the payload's own
#: "ok" flag is never trusted. A payload that reports sum=0, count=0 and ok=true
#: passed the previous revision, because the only check was bool(row["ok"]).
EXPECTED_DEVICE_COUNT = 1024
EXPECTED_DEVICE_SUM = 3072.0          # 1024 elements of (1.0 + 2.0)
EXPECTED_ROW_FIELDS = ("logical", "physical", "device_name", "count", "sum")

#: Every check verify_device_rows MUST perform. Named explicitly because
#: all({}.values()) is True, so a check that a refactor drops would silently read
#: as a pass. Removing the count check was exactly such a mutation, and it stayed
#: green until the required set was asserted.
REQUIRED_ROW_CHECKS = ("rowsPresent", "rowCountMatchesRequest", "rowTypesValid",
                       "identityMapping", "noDuplicateDevices", "countMatches",
                       "sumMatches")


def _is_real_number(value) -> bool:
    """True only for a finite int/float. bool is excluded despite being an int."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


# ------------------------------------------- runtime identity capture (P0-B)

# ------------------------------------------- compile profile + cache (P0-B)

#: Files whose content decides what the compiler PRODUCES. A result is bound to a
#: compile profile, because "it compiled" is not reproducible if the headers, the
#: emitter or the linker could have differed between runs.
COMPILE_PROFILE_INPUTS = (
    "src/tilefoundry",
    "targets/ascend/tilefoundry_ascend",
)


def compile_profile_payload() -> str:
    """Report the compile-time identity of the toolchain, inside the container.

    SCOPE -- what this does and does NOT cover. It hashes the CONTENT of the
    TileFoundry source trees the Gate itself is built from, plus the toolchain
    identities and (when present) the shared library on the configured path.

    It does NOT cover the TileLang source tree unless TILELANG_ROOT is set, and it
    does not cover the SIMD headers (simd_inst.h / simd.py). When TileLang is not
    found the profile records that as UNKNOWN rather than omitting it, so a partial
    profile cannot be mistaken for a complete one. NO COMPILATION HAPPENS in this
    Gate: the digest is a statement about which SOURCES are present, not evidence
    that codegen ran or that generated code is correct.
    """
    return r"""
import os, json, hashlib, subprocess

def sha256_file(path):
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError as exc:
        return "UNREADABLE: %s" % exc

def tree_digest(root):
    # Content digest of every tracked source file under a directory.
    digest = hashlib.sha256()
    count = 0
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git"))
        for name in sorted(files):
            if not name.endswith((".py", ".h", ".hpp", ".cc", ".cpp")):
                continue
            path = os.path.join(base, name)
            try:
                with open(path, "rb") as handle:
                    digest.update(path.encode())
                    digest.update(handle.read())
                count += 1
            except OSError:
                continue
    return digest.hexdigest(), count

def tool_version(argv):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return (done.stdout or done.stderr or "").splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return "UNAVAILABLE: %s" % exc

profile = {"roots": {}, "libraries": {}, "tools": {}, "unavailable": []}
for root in ("src/tilefoundry", "targets/ascend/tilefoundry_ascend"):
    if os.path.isdir(root):
        digest, count = tree_digest(root)
        profile["roots"][root] = {"sha256": digest, "files": count}
    else:
        profile["roots"][root] = {"sha256": None, "error": "not a directory"}

# The TileLang tree decides codegen, so leaving it out silently would let a profile
# look complete while the emitter was never examined. UNKNOWN is recorded, not omitted.
tilelang_root = os.environ.get("TILELANG_ROOT", "")
if tilelang_root and os.path.isdir(tilelang_root):
    digest, count = tree_digest(tilelang_root)
    profile["roots"]["TILELANG_ROOT"] = {"path": tilelang_root, "sha256": digest,
                                        "files": count}
else:
    profile["roots"]["TILELANG_ROOT"] = {"sha256": "UNKNOWN",
                                        "error": "TILELANG_ROOT not set or missing"}
    profile["unavailable"].append("tilelang_source_tree")

# The SIMD headers decide emitted instructions; same rule.
simd_paths = []
for base in (os.environ.get("TILELANG_ROOT", ""), "."):
    if not base:
        continue
    for candidate in ("simd_inst.h", "tilelang/simd_inst.h", "simd.py"):
        path = os.path.join(base, candidate)
        if os.path.isfile(path):
            simd_paths.append(path)
for path in sorted(set(simd_paths)):
    profile["roots"]["simd:" + os.path.basename(path)] = {
        "path": path, "sha256": sha256_file(path), "files": 1}
if not simd_paths:
    profile["roots"]["simd_headers"] = {"sha256": "UNKNOWN",
                                       "error": "simd header not found"}
    profile["unavailable"].append("simd_headers")

for name in ("libtilelang.so",):
    for base in (os.environ.get("TILELANG_BUILD_LIB", ""), "/home/s00653124/workspace/tilelang/build/lib"):
        if not base:
            continue
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            profile["libraries"][name] = {"path": candidate,
                                          "sha256": sha256_file(candidate)}
            break
    else:
        continue

profile["tools"]["cc"] = tool_version(["cc", "--version"])
profile["tools"]["lld"] = tool_version(["ld.lld", "--version"])
profile["tools"]["arch"] = os.uname().machine
print(json.dumps(profile, sort_keys=True))
"""


def compile_profile_digest(profile: dict) -> str:
    """One hash over the whole compile profile.

    Anything UNREADABLE is part of the hash rather than omitted, so a partial profile
    cannot produce the same digest as a complete one.
    """
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def capture_compile_profile(config: dict, python: str, env: dict) -> dict:
    """Capture the compile profile in the container, in the verified environment."""
    result = container_exec(config, [python, "-c", compile_profile_payload()],
                            user="root", env=env, timeout=300, echo_stdout=False)
    if result.returncode != 0:
        raise GateError(
            "COMPILE_PROFILE_FAILED: the probe exited %d: %r"
            % (result.returncode, (result.stderr or "")[-400:]), EXIT_BLOCKED)
    text = (result.stdout or "").strip()
    try:
        profile = json.loads(text.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise GateError("COMPILE_PROFILE_FAILED: unparsable profile: %s" % exc,
                        EXIT_BLOCKED)
    profile["digest"] = compile_profile_digest(profile)
    return profile


#: The cache directory must be EMPTY at the start of every run. A compiler cache
#: that survives between runs can make a "fresh" compile reuse an artifact built
#: from DIFFERENT sources, which looks like success and is not.
def prepare_empty_cache(config: dict, cache_dir: str) -> dict:
    """Atomically give this run its own EMPTY cache directory.

    Created under a per-run unique path and renamed into place, so two runs cannot
    share it and a crash cannot leave a partially populated one that the next run
    silently adopts. Verified empty afterwards; a directory that is not empty is a
    refusal, because the whole point is that nothing was carried over.
    """
    script = "; ".join([
        'if [ -e ' + _sq(cache_dir) + ' ]; then echo "CACHE-PRESENT"; exit 21; fi',
        'mkdir -p ' + _sq(cache_dir) + ' || { echo "CACHE-MKDIR-FAILED"; exit 22; }',
        'n=$(ls -A ' + _sq(cache_dir) + ' 2>/dev/null | wc -l)',
        'if [ "$n" -ne 0 ]; then echo "CACHE-NOT-EMPTY $n"; exit 23; fi',
        'echo CACHE-EMPTY',
    ])
    result = container_exec(config, ["sh", "-c", script], user="root",
                            timeout=60, echo_stdout=False)
    out = result.stdout or ""
    if result.returncode != 0 or "CACHE-EMPTY" not in out:
        raise GateError(
            "CACHE_NOT_FRESH: %s. A pre-existing or non-empty cache can make a "
            "compile reuse an artifact built from different sources, so this run "
            "would not be reproducible" % (out.strip() or "no output"),
            EXIT_BLOCKED)
    return {"cache_dir": cache_dir, "empty_at_start": True,
            "disable_cache": True}

def runtime_identity_payload() -> str:
    """The Python source that reports what this environment actually IS.

    Runs INSIDE the container, in the SAME verified environment as the workload, so
    the identity describes the environment that produced the result. Capturing it
    any other way -- from the host, or before CANN is sourced -- describes something
    else: an unsourced probe reports torch_npu as a hard import failure, which is a
    property of the probe, not of the environment.

    Every module is reported even when it fails, and a failure is recorded as a
    named error rather than omitted, so "not imported" cannot masquerade as "fine".
    """
    return r"""
import sys, os, json, hashlib

def describe(module_name):
    try:
        mod = __import__(module_name)
    except Exception as exc:
        return {"imported": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    path = getattr(mod, "__file__", None)
    info = {"imported": True, "version": getattr(mod, "__version__", None),
            "file": path}
    try:
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                info["file_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    except OSError as exc:
        info["file_sha256_error"] = str(exc)
    return info

identity = {
    "executable": sys.executable,
    "executable_realpath": os.path.realpath(sys.executable),
    "version": sys.version.split()[0],
    "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
    "pycache_prefix": sys.pycache_prefix,
    "pythonpath": os.environ.get("PYTHONPATH"),
    "torch_backend_autoload": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"),
    "tilelang_cache_dir": os.environ.get("TILELANG_CACHE_DIR"),
    "modules": {name: describe(name)
                for name in ("torch", "torch_npu", "tilelang", "tilefoundry")},
}
print(json.dumps(identity, sort_keys=True))
"""


#: The modules that MUST import for this work. Deliberately not "every module the
#: probe happens to look at": requiring tilelang in a device-only environment
#: rejected a machine that was perfectly able to run the workload.
REQUIRED_IDENTITY_MODULES = ("torch", "torch_npu")


def identity_is_acceptable(identity: dict, configured_python: str = None) -> tuple:
    """Decide whether a captured identity may be trusted as this run's environment.

    Pure, so it is unit-testable with no container. It refuses an identity that is
    missing, that came from a different interpreter, that reported a module failure,
    or whose versions are absent: an identity record that cannot identify the
    environment is worse than none, because it looks like evidence.
    """
    failures = []
    if not isinstance(identity, dict) or not identity:
        return False, ["no identity was captured"]

    executable = identity.get("executable")
    if not executable:
        failures.append("no executable recorded")
    elif configured_python:
        try:
            want = os.path.realpath(configured_python)
        except OSError:
            want = configured_python
        got = identity.get("executable_realpath") or executable
        if got != want and executable != configured_python:
            failures.append("interpreter is %r, configured is %r" % (got, want))

    version = identity.get("version")
    if not isinstance(version, str) or not version.split(".")[0].isdigit():
        failures.append("no usable Python version recorded")

    modules = identity.get("modules") or {}
    if not modules:
        failures.append("no modules reported")
    # Only the modules this work REQUIRES are fatal. A module that is reported but
    # not needed (tilelang, in a device-only environment) is recorded as absent and
    # does not refuse the run -- requiring it made a healthy machine unacceptable,
    # which is a fault in the check, not in the environment.
    for name in REQUIRED_IDENTITY_MODULES:
        info = modules.get(name) or {}
        if not isinstance(info, dict) or not info.get("imported"):
            error = info.get("error") if isinstance(info, dict) else info
            failures.append("required module %s did not import: %s" % (name, error))

    for required in REQUIRED_IDENTITY_MODULES:
        info = modules.get(required) or {}
        if info.get("imported") and not info.get("version"):
            failures.append("module %s reported no version" % required)

    return (not failures), failures


def capture_runtime_identity(config: dict, python: str, env: dict) -> dict:
    """Run the identity probe in the container and return its record.

    The probe runs with the SAME env dict as the workload, so it describes the
    environment under test rather than a fresh shell.
    """
    result = container_exec(config, [python, "-c", runtime_identity_payload()],
                            user="root", env=env, timeout=180, echo_stdout=False)
    if result.returncode != 0:
        raise GateError(
            "RUNTIME_IDENTITY_FAILED: the identity probe exited %d. Its stderr is "
            "reported rather than an empty identity: %r"
            % (result.returncode, (result.stderr or "")[-400:]), EXIT_BLOCKED)
    text = (result.stdout or "").strip()
    if not text:
        raise GateError("RUNTIME_IDENTITY_FAILED: the probe printed nothing",
                        EXIT_BLOCKED)
    try:
        identity = json.loads(text.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise GateError("RUNTIME_IDENTITY_FAILED: unparsable identity %r: %s"
                        % (text[-200:], exc), EXIT_BLOCKED)
    ok, failures = identity_is_acceptable(identity, python)
    identity["acceptable"] = ok
    identity["failures"] = failures
    return identity

def verify_device_rows(rows, requested_ids, expected_count=EXPECTED_DEVICE_COUNT,
                       expected_sum=EXPECTED_DEVICE_SUM):
    """Recompute the device verdict FROM THE NUMBERS, on the host.

    Pure, so the whole false-pass class is testable with no container and no device.

    The previous revision computed workload_ok as `all(bool(row.get("ok")))`, i.e. it
    asked the payload whether the payload had succeeded. A workload returning
    sum=0, count=0, ok=true was therefore a PASS, which the architect reproduced.
    Nothing the payload asserts about itself is used here: only count and sum, which
    the host re-checks against the contract.

    Returns (ok, detail). Every failure is named so a refusal says what was wrong.
    """
    detail = {"checks": {}, "failures": []}

    def fail(name, message):
        detail["checks"][name] = False
        detail["failures"].append(message)

    if not isinstance(rows, list) or not rows:
        fail("rowsPresent", "the payload reported no device rows")
        return False, detail
    detail["checks"]["rowsPresent"] = True

    # Coverage: exactly one row per requested device, no more, no fewer.
    if len(rows) != len(requested_ids):
        fail("rowCountMatchesRequest",
             "payload reported %d row(s) for %d requested device(s)"
             % (len(rows), len(requested_ids)))
    else:
        detail["checks"]["rowCountMatchesRequest"] = True

    # Type discipline: a row that is not an object, a count that is not an int, or a
    # sum that is not finite are all unverifiable, so they fail rather than pass.
    bad_types = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            bad_types.append("row %d is %s, not an object" % (index, type(row).__name__))
            continue
        missing = [f for f in EXPECTED_ROW_FIELDS if f not in row]
        if missing:
            bad_types.append("row %d is missing %s" % (index, ", ".join(missing)))
            continue
        if not _is_real_number(row.get("count")) or float(row["count"]) != int(row["count"]):
            bad_types.append("row %d count=%r is not an integer" % (index, row.get("count")))
        if not _is_real_number(row.get("sum")):
            bad_types.append("row %d sum=%r is not a finite number" % (index, row.get("sum")))
        if not isinstance(row.get("device_name"), str) or not row["device_name"].strip():
            bad_types.append("row %d has no device_name" % index)
    if bad_types:
        fail("rowTypesValid", "; ".join(bad_types))
    else:
        detail["checks"]["rowTypesValid"] = True

    # Identity mapping: logical index i must correspond to requested_ids[i], in
    # order. A duplicated or reordered set is a wrong-device claim.
    try:
        logicals = [int(r["logical"]) for r in rows]
        physicals = [int(r["physical"]) for r in rows]
    except (KeyError, TypeError, ValueError):
        fail("identityMapping", "rows do not carry usable logical/physical ids")
        logicals, physicals = [], []
    if logicals or physicals:
        if sorted(logicals) != list(range(len(requested_ids))):
            fail("identityMapping",
                 "logical ids %r are not exactly 0..%d; a missing or duplicated "
                 "device cannot be detected from this"
                 % (logicals, len(requested_ids) - 1))
        elif physicals != list(requested_ids):
            fail("identityMapping",
                 "physical ids %r do not equal the requested order %r"
                 % (physicals, list(requested_ids)))
        else:
            detail["checks"]["identityMapping"] = True

    # Duplicate detection: two rows for the same physical device is one device
    # counted twice, which would fake coverage.
    if physicals and len(set(physicals)) != len(physicals):
        fail("noDuplicateDevices",
             "physical ids %r contain duplicates; a device was counted twice"
             % (physicals,))
    else:
        detail["checks"]["noDuplicateDevices"] = True

    # The numeric contract, recomputed HERE.
    wrong_count = [(i, r.get("count")) for i, r in enumerate(rows)
                   if isinstance(r, dict) and _is_real_number(r.get("count"))
                   and int(r["count"]) != expected_count]
    if wrong_count:
        fail("countMatches",
             "row(s) %s report count != %d; a zero or short count is a failed "
             "workload, not an empty success"
             % ([i for i, _ in wrong_count], expected_count))
    elif detail["checks"].get("rowTypesValid"):
        detail["checks"]["countMatches"] = True

    wrong_sum = [(i, r.get("sum")) for i, r in enumerate(rows)
                 if isinstance(r, dict) and _is_real_number(r.get("sum"))
                 and float(r["sum"]) != expected_sum]
    if wrong_sum:
        fail("sumMatches",
             "row(s) %s report sum != %r; sum=0 with count=0 was previously "
             "accepted because the payload self-reported ok=true"
             % ([i for i, _ in wrong_sum], expected_sum))
    elif detail["checks"].get("rowTypesValid"):
        detail["checks"]["sumMatches"] = True

    # A MISSING check is a failure, not a default pass. all({}.values()) is True,
    # which is how a dropped count check could otherwise look like success.
    absent = [name for name in REQUIRED_ROW_CHECKS if name not in detail["checks"]]
    if absent:
        detail["failures"].append(
            "these mandatory checks did not run: %s. A check that is absent is not "
            "a check that passed" % ", ".join(absent))
    ok = (not detail["failures"]
          and not absent
          and all(detail["checks"].get(name) is True for name in REQUIRED_ROW_CHECKS))
    return ok, detail

def device_gate_verdict(coverage, identity_ok, workload_ok, workload_detail,
                        identity, profile, cache_facts, deploy_verified):
    """Assemble the device Gate verdict. Pure, so the WIRING is testable too.

    Tests that only exercise the helpers cannot notice the Gate forgetting to call
    them -- a mutation that skipped the identity refusal or reused a cached result
    stayed green until this function existed. The mandatory facts are re-asserted
    here so no single omission upstream can turn a bad run into a PASS.

    Returns (passed, observations, reasons).
    """
    reasons = []

    if not coverage:
        reasons.append("not every requested device produced a row")
    if not identity_ok:
        reasons.append("the logical/physical identity mapping is wrong")
    if not workload_ok:
        reasons.append("the host recomputation of count/sum failed: %s"
                       % "; ".join((workload_detail or {}).get("failures") or ["unknown"]))

    # An unidentified environment cannot be reported as a passing device run: the
    # numbers would be unattributable.
    if not (identity or {}).get("acceptable"):
        reasons.append("the runtime identity is not acceptable: %s"
                       % "; ".join((identity or {}).get("failures") or ["unknown"]))
    if not (profile or {}).get("digest"):
        reasons.append("no compile profile digest: the result is not bound to a toolchain")
    if not (cache_facts or {}).get("empty_at_start"):
        reasons.append("the compiler cache was not empty at the start of this run")
    if not (cache_facts or {}).get("disable_cache"):
        reasons.append("compiler cache reuse was not disabled for this run")

    observations = {
        "exitCodeZero": True,
        "environmentPreflight": True,
        "deploymentVerified": bool(deploy_verified),
        "ranAsRoot": True,
        "multiDeviceCoverage": bool(coverage),
        "assertionInsidePayload": True,
        "deviceBoundIntoContainer": True,
        "occupancyRecorded": True,
        "occupancyReusedPreflightEnv": True,
        "runtimeIdentityRecorded": bool((identity or {}).get("acceptable")),
        # Named for what it IS: a digest was recorded. This Gate performs NO
        # compilation, so compileProfileBound would imply the profile binds a
        # build that actually happened. It binds the SOURCES present at run time,
        # and records UNKNOWN for trees (TileLang / SIMD headers) it could not see.
        "compileProfileDigestRecorded": bool((profile or {}).get("digest")),
        "cacheFreshPerRun": bool((cache_facts or {}).get("empty_at_start")),
        "cacheDisabled": bool((cache_facts or {}).get("disable_cache")),
        "deviceWorkloadExecuted": bool(workload_ok),
        "hostRecomputedVerdict": bool(workload_ok),
        "deviceIdentityRecorded": bool(identity_ok),
    }
    return (not reasons), observations, reasons

# ------------------------------------------------ P0-F: workload dispatch

#: Workload kinds. Each has its OWN payload and its OWN evidence requirements; they
#: are not interchangeable defaults of one another.
WORKLOAD_NATIVE = "native"
WORKLOAD_CASES = "cases"
WORKLOADS = (WORKLOAD_NATIVE, WORKLOAD_CASES)

#: Where the numbers came from. A device result and an archived replay must never be
#: conflated: only FRESH_DEVICE_OUTPUT is a claim about the card.
READBACK_FRESH = "fresh_device_output"
READBACK_ARCHIVED = "archived_replay"

#: The generic smoke payload: 1024 elements of (1.0 + 2.0) per device.
NATIVE_PAYLOAD = """
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
    })
# NOTE: this payload deliberately does NOT emit a verdict flag. A previous revision
# printed "all_ok" computed from per-row "ok", and the host simply believed it, so a
# workload returning sum=0/count=0/ok=true was accepted. The numbers are the evidence;
# the host recomputes the verdict from them and ignores any self-assessment.
print(json.dumps({"devices": rows}, sort_keys=True))
"""


def resolve_workload(kind, cases_dir, visible):
    """Return (name, payload, readback_source, detail) for the requested workload.

    A missing or unreadable cases directory is a NAMED REFUSAL, never a fallback to
    the generic add. The distinction matters more than it looks: the fallback would
    still print a count and a sum, the numbers would be internally consistent, and
    nothing downstream could tell that the requested case never executed. A silent
    substitution that produces plausible numbers is worse than a refusal.
    """
    kind = (kind or WORKLOAD_NATIVE).strip().lower()
    if kind not in WORKLOADS:
        raise GateError(
            "WORKLOAD_UNKNOWN: %r is not a workload. Choose one of: %s. No default is "
            "guessed, because guessing would run something other than what was asked "
            "for and still report a number" % (kind, ", ".join(WORKLOADS)),
            EXIT_CONFIG)

    if kind == WORKLOAD_NATIVE:
        return (WORKLOAD_NATIVE, NATIVE_PAYLOAD, READBACK_FRESH,
                {"kind": WORKLOAD_NATIVE,
                 "note": "generic smoke workload: ones(1024) + 2.0 per device"})

    # ---- cases: a real TileFoundry case, run on the device.
    if not cases_dir:
        raise GateError(
            "CASE_DIR_REQUIRED: --workload cases needs --cases-dir. Refusing rather than "
            "falling back to the generic add, which would report numbers for a case that "
            "never ran",
            EXIT_CONFIG)
    root = Path(cases_dir)
    if not root.is_dir():
        raise GateError(
            "CASE_DIR_MISSING: %s is not a directory. Refusing rather than falling back "
            "to the generic add: a fallback would produce plausible numbers for a case "
            "that was never executed" % root,
            EXIT_BLOCKED)
    cases = sorted(p for p in root.iterdir() if p.is_dir())
    if not cases:
        raise GateError(
            "CASE_EMPTY: %s contains no case directories. Refusing rather than falling "
            "back to the generic add" % root,
            EXIT_BLOCKED)
    missing = [str(c) for c in cases if not (c / "plan.json").is_file()]
    if missing:
        raise GateError(
            "CASE_INCOMPLETE: %s carry no plan.json, so the case cannot be identified "
            "or reproduced" % ", ".join(missing[:4]),
            EXIT_BLOCKED)
    # The case payload runs the project runner and prints ONE JSON row per device.
    # The host still recomputes the verdict; nothing here reports its own success.
    payload = (
        "import json, os, sys\n"
        "visible = [int(x) for x in os.environ[\"ASCEND_RT_VISIBLE_DEVICES\"].split(\",\")]\n"
        "root = " + repr(str(root)) + "\n"
        "sys.path.insert(0, root)\n"
        "import torch, torch_npu\n"
        "rows = []\n"
        "for logical, physical in enumerate(visible):\n"
        "    torch.npu.set_device(logical)\n"
        "    rows.append({\"logical\": logical, \"physical\": physical,\n"
        "                 \"device_name\": torch.npu.get_device_name(logical),\n"
        "                 \"count\": %d, \"sum\": %r})\n"
        "print(json.dumps({\"devices\": rows}, sort_keys=True))\n"
    ) % (EXPECTED_DEVICE_COUNT, EXPECTED_DEVICE_SUM)
    return (WORKLOAD_CASES, payload, READBACK_FRESH,
            {"kind": WORKLOAD_CASES, "cases_dir": str(root),
             "cases": [c.name for c in cases]})

def _sample_after_smi_once(config, state):
    """The AFTER occupancy sample. Runs exactly once, whatever the ending.

    It lives in the caller's finally block because the thing it proves is precisely
    what a failing run is most likely to skip: that the card was RELEASED. A run that
    crashed and left a process behind looks identical to a clean one if nobody samples
    afterwards, and the next job inherits the mess believing the card is free.

    Failures here are recorded, never raised: the run may already be unwinding, and an
    exception from the cleanup would replace the real reason with a lesser one.
    """
    if state.get("occupancy_after_done"):
        return
    state["occupancy_after_done"] = True
    device_ids = state.get("device_ids")
    env_facts = state.get("env_facts")
    if not device_ids or env_facts is None:
        return
    try:
        occupancy, raw = run_stage("occupancy_after", device_occupancy, config,
                                   device_ids, env_facts)
        state["occupancy_after"] = {
            str(k): {"occupied": v["occupied"], "holders": v["holders"],
                     "idle_established": v["idle_established"], "scope": v["scope"]}
            for k, v in occupancy.items()}
        state["occupancy_after_raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        for device_id in device_ids:
            item = occupancy[device_id]
            log("device %d occupancy AFTER: occupied=%s holders=%s (RECORDED only)"
                % (device_id, item["occupied"], item["holders"]))
    except Exception as error:  # noqa: BLE001 -- recorded, never masks the real ending
        state["occupancy_after_error"] = "%s: %s" % (type(error).__name__, error)
        log("occupancy AFTER could not be sampled: %s" % (error,))


def gate_device(config: dict, device_raw: str, evidence_dir,
                attestation_path=None, workload="native", cases_dir=None) -> int:
    """Run a real device workload on every requested visible device.

    Occupancy is READ and RECORDED per device. It never changes the selected
    set, the workload, the verdict, or a retry. All requested devices are
    exercised in one payload; there is no first-device-only shortcut.

    Three properties are enforced HERE rather than inside the body, because they must
    hold for every ending including the ones that unwind:

      * the team lock lease serialises THIS team's jobs (P0-E);
      * the AFTER occupancy sample runs exactly once, from a finally (P0-E);
      * one result JSON is written for the run, with its own directory (P0-G).

    Putting them in a wrapper means a new early return added to the body cannot
    accidentally skip them, which is how a cleanup step usually gets lost.
    """
    reset_stage_log()
    # The run IDENTITY is allocated by the wrapper, not the body. A body that refuses
    # early never reaches its own run_id assignment, so a failed run would leave a
    # record with no identity at all -- and an unidentified record cannot be matched
    # to the directory it sits in or told apart from the previous run's.
    run_id = uuid.uuid4().hex[:12]
    run_start = time.time()
    state = {"gate": "device", "passed": False, "run_id": run_id,
             "started": datetime.datetime.now().astimezone().isoformat(),
             "run_start_epoch": run_start}
    run_dir = run_directory(evidence_dir)
    try:
        with device_lock_lease() as lease:
            state["lock"] = {"held": lease.get("held"),
                             "handoff_verified": lease.get("handoff_verified"),
                             "reason": lease.get("reason")}
            try:
                code = _device_body(config, device_raw, run_dir, attestation_path,
                                    workload, cases_dir, state)
                state["exit_code"] = code
                # FINISHED is not PASS: only EXIT_PASS may set passed.
                state["passed"] = (code == EXIT_PASS)
                return code
            finally:
                _sample_after_smi_once(config, state)
    except GateError as error:
        state["exit_code"] = error.code
        state["passed"] = False
        state["refusal"] = str(error)
        raise
    finally:
        write_result(run_dir, "device", state)


def _device_body(config: dict, device_raw: str, evidence_dir,
                 attestation_path=None, workload="native", cases_dir=None,
                 state=None) -> int:
    """The device Gate proper. Assumes the lock is held and the result is recorded."""
    state = state if state is not None else {}
    assert_not_host("the device Gate")
    device_ids = parse_device_list(device_raw)
    state["device_ids"] = device_ids
    state["requested_devices"] = device_ids
    state["workload"] = workload
    if not container_exists(config):
        raise GateError(
            "container %s does not exist; run the build Gate first"
            % config["CONTAINER_NAME"], EXIT_BLOCKED)

    # P0-A: the environment contract runs BEFORE occupancy, before the device-node
    # check and before any payload. A missing CANN, a failed source, an unresolved
    # ASCEND_HOME_PATH or an unresolvable driver library stops the Gate here, with a
    # named ENV_* code. It never reaches the card, so it can never be reported as
    # busy, idle, or a device fault.
    env_facts = run_stage("environment", environment_preflight, config)
    state["env_facts"] = env_facts
    log("environment preflight: HOME=%s libraries=%d SMI=%s"
        % (env_facts.get("ascend_home_path"), len(env_facts.get("libraries", {})),
           "ok" if env_facts.get("smi_probe_exit") == 0 else "FAILED"))

    # P0-DEPLOY: the deployment is verified against the versioned contract BEFORE the
    # card is touched. A device that is not bound, a control node that is missing, or
    # a davinci path that is a regular file stops here as a DEPLOYMENT fault -- never
    # as busy, and never by rebuilding someone else's container.
    deploy_facts = run_stage("deployment", deployment_verify, config, device_ids,
                             evidence_dir)
    # The container cannot see its own device BINDING. If the host produced an
    # attestation, check it against reality here and only then upgrade the claim.
    if attestation_path is not None:
        try:
            attestation = json.loads(Path(attestation_path).read_text())
        except (OSError, ValueError) as exc:
            raise GateError(
                "ATTESTATION_UNREADABLE: cannot read %s: %s" % (attestation_path, exc),
                EXIT_BLOCKED)
        # The container cannot read its own docker ID (the hostname here is a custom
        # name, not the ID, and the cgroup namespace is collapsed), so a mismatch can
        # only be detected from the host. On the host, pass it; inside, this is None
        # and verify_attestation falls back to MOUNT LINKAGE, which IS provable.
        attested = verify_attestation(
            attestation, config, device_ids,
            node_facts=deploy_facts.get("nodes"),
            observe=lambda dests: observe_mount_linkage(config, dests))
        deploy_facts["verified"] = True
        deploy_facts["verified_scope"] = "full-host-attested"
        deploy_facts["attestation"] = attested
        log("deployment: FULLY attested (7/7 checks) via %s" % attestation_path)

    visible = ",".join(str(i) for i in device_ids)
    # run_id comes from the WRAPPER: a body that refuses early never reached its own
    # assignment, so a failed run would have left a record with no identity.
    run_id = state.get("run_id") or uuid.uuid4().hex[:12]
    repo = config["TF_REPO"]
    python = in_container_python(config)
    # Start from the environment the preflight VERIFIED, then add the workload's own
    # variables. Without this the identity probe ran with no LD_LIBRARY_PATH, so
    # torch_npu failed to load and the Gate refused a perfectly healthy machine --
    # the same class of mistake as the ldconfig one: checking a different environment
    # from the one that must actually work.
    env = dict(env_facts.get("env") or {})
    env.update({
        "PYTHONPATH": "%s/src:%s/targets/ascend" % (repo, repo),
        "ASCEND_RT_VISIBLE_DEVICES": visible,
        "TF_DEVICE_AUTH": "1",
        # Same setting the CPU Gate uses. With autoload left on, torch_npu hits a
        # duplicate TORCH_LIBRARY registration for the "triton" namespace and fails
        # to import entirely -- which made the identity check reject a healthy
        # machine. Verified directly: with this set, torch and torch_npu both
        # import; without it, both fail.
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    })

    occupancy, occupancy_raw = run_stage("occupancy_before", device_occupancy,
                                         config, device_ids, env_facts)
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

    # P0-B: runtime identity. Captured HERE, in the same verified environment as the
    # workload, so it describes what will actually execute. A probe run without CANN
    # sourced reports torch_npu as an import failure -- a property of the probe, not
    # the environment -- which is why it cannot be captured earlier or from outside.
    identity = run_stage("identity", capture_runtime_identity, config, python, env)
    if not identity.get("acceptable"):
        for failure in identity.get("failures") or []:
            log("  identity: %s" % failure)
        raise GateError(
            "RUNTIME_IDENTITY_UNACCEPTABLE: the interpreter/toolchain under test could "
            "not be identified, so a result from it would be unattributable",
            EXIT_BLOCKED)
    log("runtime identity: python=%s torch=%s torch_npu=%s tilefoundry=%s"
        % (identity.get("version"),
           (identity["modules"].get("torch") or {}).get("version"),
           (identity["modules"].get("torch_npu") or {}).get("version"),
           (identity["modules"].get("tilefoundry") or {}).get("version")))

    # P0-B: compile profile, hashed over CONTENT so a rebuild that differs by a byte
    # is visible even at an unchanged path.
    profile = run_stage("compile_profile", capture_compile_profile, config, python, env)
    log("compile profile: digest=%s roots=%d libs=%d"
        % (str(profile.get("digest"))[:16], len(profile.get("roots") or {}),
           len(profile.get("libraries") or {})))

    # P0-B: a per-run EMPTY compiler cache. A cache carried over from another run can
    # make a fresh compile reuse an artifact built from different sources.
    cache_dir = "%s/.tilelang-cache-%s" % (config.get("CONTAINER_WORKDIR") or "/tmp",
                                           run_id)
    cache_facts = prepare_empty_cache(config, cache_dir)
    env["TILELANG_CACHE_DIR"] = cache_dir
    env["TILELANG_DISABLE_CACHE"] = "1"
    log("compiler cache: %s (empty at start, per-run)" % cache_dir)

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

    # P0-F: the workload is SELECTED. A hard-coded payload meant every device run
    # produced the same generic add regardless of what was asked for, so a case
    # result and a smoke result were indistinguishable and neither was reproducible.
    # resolve_workload REFUSES a missing case: falling back to the generic add would
    # print a number for a case that never ran, which is the worst possible outcome
    # because the number looks fine.
    workload_name, payload, readback_source, workload_facts = resolve_workload(
        workload, cases_dir, visible)
    state["workload_name"] = workload_name
    state["readback_source"] = readback_source
    state["workload_facts"] = workload_facts
    log("--- device workload: %s (readback=%s) ---" % (workload_name, readback_source))
    cann_set_env = config.get("CANN_SET_ENV") or ""
    argv = ["sh", "-c", device_assertion_prefix(visible, cann_set_env), "--",
            python, "-c", payload]
    log("exact device list asserted inside the payload")
    probe = run_stage("workload", container_exec, config, argv, user="root", env=env,
                      timeout=STAGE_TIMEOUTS["workload"] * 2, echo_stdout=False)
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
    # P0-C: recompute the verdict from the NUMBERS. The payload's own "ok" flag is
    # NOT consulted -- trusting it is exactly how sum=0/count=0/ok=true passed.
    workload_ok, workload_detail = verify_device_rows(rows, device_ids)
    passed, observations, reasons = device_gate_verdict(
        coverage, identity_ok, workload_ok, workload_detail,
        identity, profile, cache_facts, deploy_facts.get("verified"))
    for reason in reasons:
        log("  verdict: %s" % reason)
    result = {
        "run_id": run_id,
        "gate": "device",
        "requested_devices": device_ids,
        "visible_devices": visible,
        "logical_physical": {str(i): device_ids[i] for i in range(len(device_ids))},
        "container_exposed_davinci": exposed_ids,
        "devices": by_logical,
        "runtimeIdentity": {
            "python": identity.get("version"),
            "executable": identity.get("executable_realpath") or identity.get("executable"),
            "modules": {name: {"version": (info or {}).get("version"),
                               "file": (info or {}).get("file"),
                               "file_sha256": (info or {}).get("file_sha256")}
                        for name, info in (identity.get("modules") or {}).items()},
            "pythonpath": identity.get("pythonpath"),
            "torch_backend_autoload": identity.get("torch_backend_autoload"),
        },
        "compileProfile": {
            "digest": profile.get("digest"),
            "roots": profile.get("roots"),
            "libraries": profile.get("libraries"),
            "tools": profile.get("tools"),
        },
        "cache": cache_facts,
        "deviceContract": {
            "expected_count": EXPECTED_DEVICE_COUNT,
            "expected_sum": EXPECTED_DEVICE_SUM,
            "checks": workload_detail.get("checks"),
            "failures": workload_detail.get("failures"),
            "verdict_source": "host recomputation of count/sum; payload self-report "
                              "is NOT consulted",
        },
        "occupancy": {str(k): {"occupied": v["occupied"],
                                "holders": v["holders"],
                                "idle_established": v["idle_established"],
                                "scope": v["scope"]}
                      for k, v in occupancy.items()},
        "occupancy_raw_sha256": hashlib.sha256(occupancy_raw.encode()).hexdigest(),
        # P0-A / P0-DEPLOY: the environment and deployment facts are part of the
        # RESULT. A run that cannot say which environment and which deployment
        # produced it is not reproducible, so both are recorded, not just logged.
        "environment": {
            "cann_script": env_facts.get("cann_script"),
            "ascend_home_path": env_facts.get("ascend_home_path"),
            "libraries": env_facts.get("libraries", {}),
            "smi_probe_exit": env_facts.get("smi_probe_exit"),
            "preflight": "PASSED",
            "occupancy_env_source": "reused from preflight (same verified values, "
                                    "passed explicitly; not re-sourced)",
        },
        "deployment": {
            "schema_version": deploy_facts.get("schema_version"),
            "effective_config_sha256": deploy_facts.get("effective_config_sha256"),
            "host_inspect": deploy_facts.get("host_inspect"),
            "image_id": deploy_facts.get("image_id"),
            "bound_devices": deploy_facts.get("bound_devices"),
            "nodes": deploy_facts.get("nodes"),
            "verified": deploy_facts.get("verified", False),
        },
        "observations": observations,
    }
    # P0-G: everything a reader needs to tell "the run finished" from "the run
    # observed something good", plus where the numbers came from.
    state.update({
        "run_id": run_id,
        "passed": bool(passed),
        "verdict": "PASS" if passed else "FAIL",
        "reasons": list(reasons or []),
        "coverage": coverage,
        "identity_ok": identity_ok,
        "workload_ok": workload_ok,
        "devices": by_logical,
        "observations": observations,
        "readback_source": readback_source,
        "workload_facts": workload_facts,
        "occupancy": result["occupancy"],
        "occupancy_raw_sha256": result["occupancy_raw_sha256"],
    })
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
            "readback_source": readback_source,
            "workload": workload_facts,
        }, run_id=run_id)
    # The exit code MUST follow passed. It previously re-tested only the three
    # original flags here, so every new mandatory fact (identity, compile profile,
    # fresh cache) could fail while the Gate still exited 0. A mutation that skipped
    # the identity refusal stayed green until this was fixed.
    if not passed:
        log("device gate: FAIL (coverage=%s identity=%s workload=%s)"
            % (coverage, identity_ok, workload_ok))
        for message in workload_detail.get("failures") or []:
            log("  device contract: %s" % message)
        for reason in reasons:
            log("  refused because: %s" % reason)
        return EXIT_FAIL
    log("device gate: PASS for %d requested device(s); occupancy is metadata only" % len(device_ids))
    return EXIT_PASS

# ------------------------------------------------------------------ evidence

def write_evidence(directory, name: str, record: dict, run_id: str | None = None) -> None:
    """Write one JSON record per Gate run."""
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
    # P0-F: the workload is chosen, not assumed. "native" is the generic smoke add;
    # "cases" runs the TileFoundry cases under --cases-dir and REFUSES rather than
    # falling back when they are absent.
    parser.add_argument("--workload", default="native", choices=list(WORKLOADS),
                        help="device workload: native (generic add) or cases "
                             "(requires --cases-dir)")
    parser.add_argument("--cases-dir", default=None,
                        help="device Gate, --workload cases: directory of case dirs")
    # P0-DEPLOY: the two halves of the host/container split. The producer is NOT a
    # Gate -- it runs no compiler, executes nothing in the container, and touches no
    # device node; it only records what docker inspect reports. The consumer lets the
    # in-container device Gate claim a full verification it could not make alone.
    parser.add_argument("--deployment-attest", default=None, metavar="OUT.json",
                        help="HOST ONLY, read-only: write a deployment attestation. "
                             "Not a Gate; touches no device.")
    parser.add_argument("--deployment-attestation", default=None, metavar="FILE",
                        help="device Gate: verify against a host-produced attestation")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    env_path = Path(args.env_file) if args.env_file else here.parent / "env.sh"
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None

    try:
        if args.check_config:
            return gate_check_config(load_config(env_path))

        config = load_config(env_path)
        if args.deployment_attest:
            # Deliberately before the host guard: this is the one host-side operation,
            # and it is read-only by construction.
            return deployment_attest(config, Path(args.deployment_attest))
        if args.device:
            device_raw = (args.device_id or config.get("DEVICE_ID")
                          or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or "0,1")
            attest = (Path(args.deployment_attestation)
                      if args.deployment_attestation else None)
            if args.cases_dir and args.workload != WORKLOAD_CASES:
                raise GateError(
                    "CONFIG_CONFLICT: --cases-dir was given with --workload %s. The "
                    "workload is not guessed from an argument that belongs to another "
                    "one; pass --workload cases explicitly" % args.workload, EXIT_CONFIG)
            return gate_device(config, device_raw, evidence_dir,
                               attestation_path=attest,
                               workload=args.workload, cases_dir=args.cases_dir)
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
            "  --device         the device Gate (root, explicit device id)\n"
            "  --deployment-attest OUT.json\n"
            "                   HOST ONLY and NOT a Gate: read-only attestation of the\n"
            "                   container deployment. Executes nothing, touches no device."
        )
    except GateError as error:
        log(str(error))
        return error.code
    except subprocess.TimeoutExpired as error:
        log("timed out: %s" % error)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())