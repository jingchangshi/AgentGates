#!/usr/bin/env python3
"""P0-E/F/G/H contract tests: behavioural, not textual.

Every test here drives the REAL gate_device() with all container I/O stubbed
(container_exec, device_occupancy, the preflights), then asserts on what the gate
ACTUALLY DID: how many times it sampled occupancy, whether two concurrent jobs
entered the payload together, what record it left on disk, how long a hung stage
took to come back.

That matters because a test can go green by accident. Searching the source for the
word "workload" passes on a docstring; searching for "archived" passes on a comment.
Both did, in the first version of this file, and both were rewritten. A test that
fails by AttributeError is just as useless in the other direction: it reports a
crash, not a behaviour. So a missing mechanism is reported as an AssertionError
that names the behaviour that did not happen.

These are written to the CONTRACT (audit section 5, P0-E/F/G/H). On the committed
gate they are expected to FAIL: that failure is the gap the P0 work has to close.
See P0_EFGH_BASELINE.txt for the recorded run and P0_EFGH_IMPLEMENTATION_PLAN.md
for the plan.

Hard invariants that must still hold after the implementation:
  * busy is RECORDED only -- no wait, poll, retry, skip, downgrade, or
    busy-dependent card choice (device_policy.py enforces the static half);
  * the team lock coordinates THIS team only; another tenant being busy is metadata.

Run: python -B agent_gates/adapters/tilefoundry-npu/tests/test_e2e_chain.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT_DIR = HERE.parent / "scripts"
GATE_PATH = SCRIPT_DIR / "tilefoundry_gate.py"
sys.path.insert(0, str(SCRIPT_DIR))

PASS: list = []
FAIL: list = []

PAYLOAD_OK = json.dumps({
    "devices": [
        {"logical": 0, "physical": 0, "device_name": "dev0", "count": 1024,
         "sum": 3072.0, "ok": True},
        {"logical": 1, "physical": 1, "device_name": "dev1", "count": 1024,
         "sum": 3072.0, "ok": True},
    ],
    "all_ok": True,
})
EXPOSED = "/dev/davinci0\n/dev/davinci1\n/dev/davinci_manager\n"


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        text = "%s: %s" % (type(exc).__name__, str(exc)[:240])
        FAIL.append("%s: %s" % (name, text))
        print("  FAIL  %s\n        %s" % (name, text))
        return
    PASS.append(name)
    print("  PASS  %s" % name)


def require(condition, message):
    """A behavioural assertion. Never an AttributeError: say what did NOT happen."""
    if not condition:
        raise AssertionError(message)


def load_gate():
    """A FRESH module per test: no stub leaks from one case into the next."""
    spec = importlib.util.spec_from_file_location("gate_under_test", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResult:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


class Harness:
    """Run the real gate_device with every container boundary replaced.

    Only the container boundary is stubbed. The control flow under test -- the lock,
    the stage bounds, the finally, the record -- is the gate's own code.
    """

    def __init__(self, payload=PAYLOAD_OK, payload_rc=0, payload_delay=0.0,
                 occupancy_delay=0.0, occupancy_raises=None, on_payload=None):
        self.gate = load_gate()
        self.payload = payload
        self.payload_rc = payload_rc
        self.payload_delay = payload_delay
        self.occupancy_delay = occupancy_delay
        self.occupancy_raises = occupancy_raises
        self.on_payload = on_payload
        self.occupancy_calls = 0
        self.payload_calls = 0
        self.exec_calls: list = []
        self.gate.assert_not_host = lambda *a, **k: None
        self.gate.container_exists = lambda config: True
        self.gate.in_container_python = lambda config: "python"
        # Preflights exist only after P0-A/P0-DEPLOY; stub them when present so the
        # same harness works before and after that work lands.
        if hasattr(self.gate, "environment_preflight"):
            self.gate.environment_preflight = lambda config: {
                "ascend_home_path": "/fake/cann", "libraries": {}, "smi_probe_exit": 0,
                "cann_script": "/fake/set_env.sh"}
        if hasattr(self.gate, "deployment_verify"):
            self.gate.deployment_verify = lambda config, ids, evidence=None: {
                "schema_version": 1, "effective_config_sha256": "0" * 64,
                "host_inspect": {}, "image_id": "sha256:fake", "bound_devices": list(ids),
                "nodes": len(ids), "verified": True}
        self.gate.device_occupancy = self._occupancy
        self.gate.container_exec = self._exec

    def _occupancy(self, config, device_ids):
        self.occupancy_calls += 1
        if self.occupancy_delay:
            time.sleep(self.occupancy_delay)
        if self.occupancy_raises is not None:
            raise self.occupancy_raises
        record = {
            "occupied": None, "idle_established": False, "holders": None,
            "process_table": "present", "scope": "container-pid-namespace",
            "process_rows": [],
        }
        return ({device_id: dict(record) for device_id in device_ids}, "npu-smi raw")

    def _exec(self, config, argv, **kwargs):
        self.exec_calls.append(list(argv))
        if argv[:3] == ["sh", "-c", "ls /dev/davinci* 2>/dev/null"]:
            return FakeResult(EXPOSED)
        self.payload_calls += 1
        if self.on_payload is not None:
            self.on_payload()
        if self.payload_delay:
            time.sleep(self.payload_delay)
        return FakeResult(self.payload, self.payload_rc)

    def run(self, device_raw="0,1", evidence_dir=None, **kwargs):
        """Call the gate; return (exit code, captured stdout, elapsed seconds)."""
        out = io.StringIO()
        started = time.time()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                code = self.gate.gate_device(
                    {"TF_REPO": "/tmp", "CONTAINER_NAME": "fake"}, device_raw, evidence_dir,
                    **kwargs)
            except TypeError as exc:
                raise AssertionError(
                    "gate_device does not accept the P0-F workload selector (%s); the device "
                    "Gate can only run its one hard-coded payload" % exc)
            except self.gate.GateError as exc:
                code = exc.code
                out.write(str(exc))
        return code, out.getvalue(), time.time() - started


def result_files(directory: Path):
    return sorted(p for p in Path(directory).rglob("*.json")) if Path(directory).is_dir() else []


# ------------------------------------------------------------- P0-E: the lock

WORKER = '''\
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gate_under_test", sys.argv[1])
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
try:
    with gate.device_lock_lease() as lease:
        print("ENTERED", getattr(lease, "handoff_verified", "n/a"), flush=True)
except gate.GateError as exc:
    print("REFUSED", exc, flush=True)
'''


def _two_real_workers(env_extra=None, workers=2):
    """Run N SEPARATE PROCESSES competing for the one team lock.

    Threads would be the wrong instrument twice over: a flock is per open file
    description, and contextlib.redirect_stdout is process-global and not
    thread-safe (in the first version of this file two threads corrupted stdout
    and silently swallowed every later result). Real processes also model what a
    second team job actually is.
    """
    import subprocess
    env = dict(os.environ)
    env.pop("TF_DEVICE_LOCK_HELD", None)
    env.update(env_extra or {})
    procs = [
        subprocess.Popen(
            [sys.executable, "-B", "-c", WORKER, str(GATE_PATH)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        for _ in range(workers)
    ]
    out = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=120)
        out.append(stdout.strip() + stderr.strip())
    return out


def test_two_workers_never_enter_the_payload_together():
    """The decisive property: two cooperating jobs must not hold the lock at once.

    Measured across real processes, not read out of the source.
    """
    gate = load_gate()
    require(hasattr(gate, "device_lock_lease"),
            "the gate has no team lock lease at all, so two jobs can enter the device "
            "payload concurrently and neither would notice")
    results = _two_real_workers()
    entered = [line for line in results if line.startswith("ENTERED")]
    refused = [line for line in results if "REFUSED" in line]
    require(len(entered) + len(refused) == 2,
            "the workers did not both report an outcome: %r" % (results,))
    require(len(refused) == 1,
            "with two jobs and one lock exactly one must be refused; got %d entered and %d "
            "refused (%r)" % (len(entered), len(refused), results))


def test_forged_lock_held_env_cannot_bypass_the_lock():
    """TF_DEVICE_LOCK_HELD=1 must not be a self-service pass.

    Any stranger can set that variable. If it alone admitted a job, serialisation
    would be voluntary, so a forged flag must not produce a VERIFIED handoff.
    """
    gate = load_gate()
    require(hasattr(gate, "device_lock_lease"),
            "the gate has no team lock lease, so a forged TF_DEVICE_LOCK_HELD=1 skips "
            "nothing because there is nothing to skip")
    results = _two_real_workers(env_extra={"TF_DEVICE_LOCK_HELD": "1"})
    entered = [line for line in results if line.startswith("ENTERED")]
    require(len(entered) < 2,
            "with TF_DEVICE_LOCK_HELD=1 forged by both jobs, %d entered at once; the flag "
            "was trusted as proof of a real parent (%r)" % (len(entered), results))
    for line in entered:
        require("False" in line or "n/a" in line,
                "an unverified handoff was recorded as verified: %r" % (line,))


def test_lock_is_released_after_a_failing_payload():
    """A job that fails must not wedge the lock for the team forever.

    Tested against the LEASE itself: a "released" claim is meaningless while no
    lock exists, which is why the first version of this test passed vacuously.
    """
    gate = load_gate()
    require(hasattr(gate, "device_lock_lease"),
            "the gate has no team lock lease, so 'the lock is released after a failure' "
            "cannot even be asked -- there is no lock to release")
    try:
        with gate.device_lock_lease():
            raise RuntimeError("payload blew up")
    except RuntimeError:
        pass
    results = _two_real_workers(workers=1)
    require(any(line.startswith("ENTERED") for line in results),
            "after a failed payload no other job could take the lock: %r" % (results,))


def _quiet(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------- P0-E: watchdog and after-SMI

def test_hung_stage_is_bounded_and_recorded_as_timeout():
    """A payload that never returns must come back as a named TIMEOUT.

    Today it simply blocks for as long as the payload does and reports whatever
    the process eventually prints, which is indistinguishable from a slow pass.
    """
    h = Harness(payload_delay=3.0)
    code, output, elapsed = h.run()
    require(elapsed < 2.5,
            "a 3.0s payload was allowed to run %.2fs; no stage bound exists, so a hang is "
            "indistinguishable from a slow device" % elapsed)
    require("TIMEOUT" in output.upper() or code != 0,
            "the hung stage returned success with no TIMEOUT record")


def test_a_timeout_is_not_retried():
    """A timeout must not be retried, and never until the card looks idle."""
    h = Harness(payload_delay=3.0)
    elapsed = 0.0
    with contextlib.suppress(Exception):
        _, _, elapsed = h.run()
    require(h.payload_calls == 1,
            "the payload was executed %d times; a bound must not be retried, and certainly "
            "not until another tenant leaves" % h.payload_calls)
    # "Ran once" alone is vacuously true while no bound exists, so also require that the
    # overrun was actually cut short: a timeout that still waits the full time is not one.
    require(elapsed < 2.5,
            "the 3.0s payload ran %.2fs before returning; nothing bounded it, so 'not "
            "retried' is true only because it never came back" % elapsed)


def test_occupancy_is_sampled_before_and_after_exactly_once():
    """before AND after, one each. The after sample is what proves the card was released."""
    h = Harness()
    h.run()
    require(h.occupancy_calls == 2,
            "occupancy was sampled %d time(s); P0-E requires exactly one BEFORE and one "
            "AFTER sample" % h.occupancy_calls)


def test_after_sample_runs_even_when_the_payload_fails():
    """The after-SMI lives in a finally, so a failure still proves release."""
    h = Harness(payload="boom", payload_rc=1)
    h.run()
    require(h.occupancy_calls == 2,
            "the payload failed and occupancy was sampled %d time(s); without a finally "
            "block a failing run never records the card state it left behind"
            % h.occupancy_calls)


# --------------------------------------------------- P0-G: the result record

def test_failure_still_writes_exactly_one_result_json():
    """Every ending is recorded. A crash with no record is how "ok" gets assumed."""
    directory = Path(tempfile.mkdtemp(prefix="p0g-fail-"))
    h = Harness(payload="boom", payload_rc=1)
    h.run(evidence_dir=directory)
    files = result_files(directory)
    require(len(files) >= 1,
            "a FAILED run wrote no JSON at all; the only account of the failure is the "
            "console text")


def test_each_run_gets_its_own_directory():
    """Two runs must not share a directory, or a stale result reads as a fresh one."""
    directory = Path(tempfile.mkdtemp(prefix="p0g-runs-"))
    Harness().run(evidence_dir=directory)
    first = result_files(directory)
    Harness().run(evidence_dir=directory)
    files = result_files(directory)
    require(len(files) > len(first),
            "the second run added no record (%d -> %d files in %s); a run that overwrites "
            "the previous one makes a stale result read as fresh"
            % (len(first), len(files), directory))
    # Uniqueness is only useful if a reader can tell the two apart: the record must carry
    # the identity and timing of THIS run, not just be a differently named file.
    record = json.loads(Path(files[-1]).read_text())
    blob = json.dumps(record).lower()
    missing = [key for key in ("run_id", "started")
               if key not in blob and key.replace("_", "") not in blob]
    require(not missing,
            "the per-run record does not identify the run (%r absent); uniqueness of the "
            "file name is not uniqueness of the evidence" % (missing,))


def test_record_separates_finished_from_passed():
    """FINISHED means the process stopped. PASS means the observation was good."""
    directory = Path(tempfile.mkdtemp(prefix="p0g-finished-"))
    # A payload that reports success at the transport level but observed nothing. An
    # exit code of 0 here must not be able to claim a pass.
    h = Harness(payload=json.dumps({"devices": [], "all_ok": False}))
    h.run(evidence_dir=directory)
    files = result_files(directory)
    require(files, "no result record was written, so nothing distinguishes a finished "
                   "run from a passing one")
    record = json.loads(Path(files[-1]).read_text())
    require("finished" in json.dumps(record).lower() or "status" in record,
            "the record has no status/finished field: %r" % (list(record),))
    require(record.get("passed") is False,
            "a run that observed no device result still recorded passed=%r; FINISHED was "
            "treated as PASS" % (record.get("passed"),))


def test_record_carries_the_phase_graph():
    """Phases must declare prerequisites, so a skipped phase is visible, not absent."""
    directory = Path(tempfile.mkdtemp(prefix="p0g-dag-"))
    Harness().run(evidence_dir=directory)
    files = result_files(directory)
    require(files, "no result record was written, so there is no phase graph to inspect")
    record = json.loads(Path(files[-1]).read_text())
    blob = json.dumps(record)
    require("phase" in blob.lower(),
            "the record carries no phases: a partial run looks identical to a full one")
    require("dag" in blob.lower() or "requires" in blob.lower() or "depends" in blob.lower(),
            "the phases have no dependency edges, so a skipped prerequisite cannot be "
            "detected")


# ------------------------------------------- P0-F: reusable workloads, no fallback

def test_a_missing_case_refuses_and_never_falls_back_to_the_generic_add():
    """A missing case must be a named refusal.

    The dangerous alternative is falling back to the generic smoke payload: that
    prints a number for a case that never ran, and the number looks fine.
    """
    h = Harness()
    try:
        code, output, _ = h.run(workload="cases", cases_dir="/nonexistent/p0f")
    except AssertionError:
        raise
    require(h.payload_calls == 0,
            "with a nonexistent cases directory the gate still executed %d payload(s); that "
            "is a fallback, and any number it produces belongs to a different workload"
            % h.payload_calls)
    require(code != 0 and "CASE" in output.upper(),
            "a missing case produced code=%r without a named CASE_* refusal" % (code,))


def test_the_device_payload_is_selected_not_hard_coded():
    """The gate must run the requested workload, not always its own smoke add."""
    h = Harness()
    try:
        h.run(workload="smoke")
    except AssertionError:
        raise
    smoke_calls = h.payload_calls
    require(smoke_calls >= 1, "the smoke workload did not execute at all")
    source = GATE_PATH.read_text(encoding="utf-8")
    require("torch.ones((1024,)" not in source,
            "the generic ones(1024)+add payload is still baked into the gate, so no other "
            "workload can be selected")


def test_a_device_run_labels_where_its_numbers_came_from():
    """A device result and an archived readback must be distinguishable in the record."""
    directory = Path(tempfile.mkdtemp(prefix="p0f-label-"))
    Harness().run(evidence_dir=directory)
    files = result_files(directory)
    require(files, "no result record was written, so the readback cannot be labelled")
    blob = json.dumps(json.loads(Path(files[-1]).read_text()))
    require("readback" in blob.lower() or "device_output" in blob.lower(),
            "the record never says whether the numbers are a fresh device readback or an "
            "archived replay; the two claims are indistinguishable")


# ------------------------ P0-DEPLOY boundary: the host-only attestation channel
#
# --deployment-attest is a READ-ONLY host channel: it runs no device work and only
# inspects the container. The device Gate consumes its output and may only claim a full
# deployment verification when all seven properties below hold. Each one is a real way a
# stale or forged record could be passed off as a fresh verified deployment.
#
# These tests are written against the IMPLEMENTED contract: verify_attestation(
# attestation, config, device_ids, node_facts, observe) raising GateError with a named
# code. The seven DEFECTS are the contract; the exact code spelling is the
# implementation's, and the code actually raised is reported on failure so a naming
# difference can be told apart from a missing check.

ATTEST_FN = "verify_attestation"


def _digest(record):
    body = {k: v for k, v in record.items() if k != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.sha256(canonical.encode()).hexdigest()


def _stamp(record):
    """Seal a record the way the producer does, so only the tested field differs."""
    record["content_sha256"] = _digest(record)
    return record


def _iso(seconds_ago):
    import datetime
    when = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds_ago)
    return when.isoformat()


def _attest(**overrides):
    """A well-formed attestation; each test breaks exactly ONE property."""
    record = {
        "schema_version": 1,
        "kind": "host-deployment-attestation",
        "attested_at": _iso(5),
        "container_name": "fake",
        "container_id": "c" * 64,
        "created": _iso(600),
        "image": "fake:1",
        "image_digest": "sha256:" + "a" * 64,
        "host_config_devices": [
            {"path_on_host": "/dev/davinci0", "path_in_container": "/dev/davinci0",
             "permissions": "rwm"},
            {"path_on_host": "/dev/davinci1", "path_in_container": "/dev/davinci1",
             "permissions": "rwm"},
        ],
        "mounts": [
            {"type": "bind", "source": "/usr/local/Ascend",
             "destination": "/usr/local/Ascend", "mode": "ro", "read_only": True},
            {"type": "bind", "source": "/home/s00653124",
             "destination": "/home/s00653124", "mode": "rw", "read_only": False},
        ],
        "config_visible_devices": "0,1",
        "privileged": False,
        "user": "root",
    }
    record.update(overrides)
    return _stamp(record)


def _verify(record, device_ids=(0, 1), config=None, node_facts=None, observe=None):
    """Run the real verifier; return its detail dict, or {"refused": code}."""
    gate = load_gate()
    fn = getattr(gate, ATTEST_FN, None)
    require(callable(fn),
            "the gate has no %s, so a host-only attestation cannot be verified at all "
            "and a full deployment claim could only ever be asserted by hand" % ATTEST_FN)
    kwargs = dict(config=config if config is not None else {"CONTAINER_NAME": "fake"},
                  device_ids=list(device_ids))
    if node_facts is not None:
        kwargs["node_facts"] = node_facts
    if observe is not None:
        kwargs["observe"] = observe
    try:
        detail = fn(record, **kwargs)
    except gate.GateError as exc:
        return {"refused": str(exc)}
    return {"detail": detail}


def _expect_refused(label, wanted, **kwargs):
    outcome = _verify(**kwargs)
    refused = outcome.get("refused")
    require(bool(refused),
            "%s was ACCEPTED; a deployment record that survives this is indistinguishable "
            "from a verified one" % label)
    require(wanted in refused,
            "%s was refused, but for %r, not %s -- either the wrong check fired or the "
            "code is spelled differently" % (label, refused[:120], wanted))


def test_attestation_accepts_a_well_formed_fresh_record():
    """The positive control. Without it a verifier that refuses EVERYTHING would pass
    all seven negatives below and prove nothing."""
    outcome = _verify(_attest())
    require("refused" not in outcome,
            "a well-formed fresh attestation was refused (%r); every negative below "
            "would then pass vacuously" % (outcome.get("refused", "")[:150],))


def test_attestation_refuses_a_stale_record():
    """An attestation from a previous window is not evidence about this container."""
    _expect_refused("an attestation produced an hour ago", "ATTESTATION_STALE",
                    record=_attest(attested_at=_iso(3600)))


def test_attestation_refuses_a_container_id_mismatch():
    """A record about a DIFFERENT container says nothing about this one."""
    _expect_refused(
        "an attestation whose container_id belongs to another container",
        "ATTESTATION_WRONG_CONTAINER",
        record=_attest(container_id="d" * 64),
        config={"CONTAINER_NAME": "fake", "_attested_container_id": "c" * 64})


def test_attestation_refuses_a_missing_device():
    """A deployment that does not bind a requested card cannot run on it."""
    _expect_refused(
        "an attestation whose HostConfig.Devices omit device 1",
        "ATTESTATION_DEVICE_NOT_BOUND",
        record=_attest(host_config_devices=[
            {"path_on_host": "/dev/davinci0", "path_in_container": "/dev/davinci0",
             "permissions": "rwm"}]),
        device_ids=(0, 1))


def test_attestation_refuses_a_pseudo_character_node():
    """A davinci path that is a regular file is the classic fake: it exists, so a
    path-existence check passes, but nothing can be launched through it."""
    _expect_refused(
        "a /dev/davinci1 that is not a character device inside the container",
        "ATTESTATION_NODE_NOT_CHAR",
        record=_attest(),
        node_facts={"/dev/davinci0": "CHAR", "/dev/davinci1": "NOTCHAR"})


def test_attestation_refuses_a_missing_mount():
    """Without the driver mount the container cannot load its own runtime."""
    _expect_refused(
        "an attestation attesting no Ascend driver mount",
        "ATTESTATION_DRIVER_MOUNT_MISSING",
        record=_attest(mounts=[{"type": "bind", "source": "/home/s00653124",
                              "destination": "/home/s00653124", "mode": "rw",
                              "read_only": False}]))


def test_attestation_refuses_an_unlinked_mount():
    """An attested mount that is absent inside this container proves the record is
    about a different container, even when the IDs cannot be read from in here."""
    _expect_refused(
        "an attestation whose attested mount is absent inside the container",
        "ATTESTATION_MOUNT_NOT_LINKED",
        record=_attest(),
        observe=lambda destinations: {"checked": [], "missing": ["/usr/local/Ascend"],
                                     "writable": {}})


def test_attestation_refuses_image_drift():
    """The running image must be the contracted one, not merely a well-formed value.

    A record that carries SOME image digest and is accepted regardless of which image
    the contract names is the drift this case exists to catch: the gate would verify
    that a digest exists, not that it is the right one.
    """
    drifted = _attest(image_digest="sha256:" + "e" * 64)
    contract_image = "sha256:" + "a" * 64
    _expect_refused(
        "an attestation whose image digest drifted from the contracted image",
        "ATTESTATION_IMAGE_DRIFT",
        record=drifted,
        config={"CONTAINER_NAME": "fake", "IMAGE": contract_image,
                "IMAGE_DIGEST": contract_image})


def test_attestation_refuses_a_tampered_hash():
    """A record edited after the fact must not verify against its own digest."""
    record = _attest()
    record["host_config_devices"] = record["host_config_devices"][:1]  # edit
    # NOTE: the digest is deliberately NOT recomputed here. Recomputing would repair
    # the tampering this test exists to detect.
    _expect_refused("an attestation edited after its content hash was taken",
                    "ATTESTATION_TAMPERED", record=record)

# --------------------------------------------------- controls: these must PASS

def test_control_busy_is_recorded_not_decided():
    """An empty process table is UNKNOWN idle, never "idle"."""
    gate = load_gate()
    sample = """\
+---------------------------+
| NPU ID | Name        | Health |
| 0      | Ascend950PR | OK     |
+---------------------------+---------------+------------------+
| NPU ID                    | Process id    | Process name     |
+===========================+===============+==================+
| No running processes found in NPU 0                              |
+===========================+===============+==================+
"""
    parsed = gate.parse_npu_smi(sample)
    require(parsed is not None,
            "parse_npu_smi returned None, so occupancy cannot be recorded at all")
    health, holders, seen = parsed
    require(health == {0: "OK"}, "health must be read per physical id: %r" % (health,))
    require(holders == {}, "an empty table has no holders: %r" % (holders,))
    require(seen is True, "the process table must be reported present")


def test_control_a_busy_dependent_verdict_is_detectable():
    """The detector must catch a busy-dependent verdict change.

    This runs the REAL detector from device_policy over a broken copy, so a
    regression in the detector shows up here instead of silently blessing every
    future revision.
    """
    spec = importlib.util.spec_from_file_location("devpol", HERE / "device_policy.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    source = GATE_PATH.read_text(encoding="utf-8")
    # Match the call by shape, not by its exact argument list: device_occupancy gained
    # a preflight_facts parameter and an exact-string anchor reported that as a broken
    # control. The control is about the busy-skip detector, not about the signature.
    import re as _re
    match = _re.search(r"^[ \t]*(occupancy, occupancy_raw) = device_occupancy\([^)]*\)[ \t]*$",
                       source, _re.MULTILINE)
    require(match is not None,
            "no 'occupancy, occupancy_raw = device_occupancy(...)' assignment exists, so this "
            "control cannot inject a busy-skip; update the control rather than deleting it")
    anchor = match.group(0)
    indent = " " * (len(anchor) - len(anchor.lstrip()))
    broken = source.replace(
        anchor,
        anchor + "\n%sif occupancy[device_ids[0]][\"occupied\"]:\n"
                 "%s    return EXIT_PASS  # injected busy skip" % (indent, indent),
        1)
    require(bool(policy.audit_source(broken)),
            "the busy-skip detector did not fire on an injected skip; every later green "
            "result would be untrustworthy")


# ------------------- attestation negatives must DISCRIMINATE, not merely fail
#
# The negatives above could all go green because the verifier refuses EVERYTHING, and
# they could go red because ONE check is missing while the other six are fine. Neither
# tells us the matrix is any good. So the same defects are replayed against a contract
# REFERENCE: each must be refused, each for its OWN code, and a clean record accepted.
# If the reference accepts a bad record, the corresponding negative is decoration.
#
# This control is what caught two real problems in this very file: a tamper case that
# recomputed the digest it was meant to detect, and a node case whose record all seven
# checks had agreed to accept. It stays PERMANENTLY.

REFERENCE_VERIFIER = '''\
import datetime, hashlib, json

SCHEMA = 1
MAX_AGE = 300


def digest(record):
    body = {k: v for k, v in record.items() if k != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify(attestation, config, device_ids, node_facts=None, observe=None):
    """Each branch maps to exactly one negative in the test file."""
    # tampered: the record must verify against its own contents
    if attestation.get("content_sha256") != digest(attestation):
        return "ATTESTATION_TAMPERED"
    # container identity
    want_id = config.get("_attested_container_id")
    if want_id and attestation.get("container_id") != want_id:
        return "ATTESTATION_WRONG_CONTAINER"
    # stale / future
    try:
        when = datetime.datetime.fromisoformat(str(attestation.get("attested_at")))
    except (TypeError, ValueError):
        return "ATTESTATION_INVALID"
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    age = (datetime.datetime.now(datetime.timezone.utc) - when).total_seconds()
    if age > MAX_AGE or age < -60:
        return "ATTESTATION_STALE"
    # requested devices genuinely bound
    bound = set()
    for entry in attestation.get("host_config_devices") or []:
        path = entry.get("path_in_container") or entry.get("path_on_host")
        if path:
            bound.add(path)
    missing = ["/dev/davinci%d" % i for i in device_ids
               if "/dev/davinci%d" % i not in bound]
    if missing:
        return "ATTESTATION_DEVICE_NOT_BOUND"
    # driver mount present
    destinations = [m.get("destination") for m in attestation.get("mounts") or []]
    if not [d for d in destinations if d and "Ascend" in str(d)]:
        return "ATTESTATION_DRIVER_MOUNT_MISSING"
    # nodes must be character devices
    if node_facts:
        if [n for n, k in node_facts.items() if k == "NOTCHAR"]:
            return "ATTESTATION_NODE_NOT_CHAR"
        if [n for n, k in node_facts.items() if k == "MISSING"]:
            return "ATTESTATION_NODE_MISSING"
    # attested mounts must be linked here, and ro must still be ro
    if observe is not None:
        linkage = observe(destinations) or {}
        if linkage.get("missing"):
            return "ATTESTATION_MOUNT_NOT_LINKED"
        attested_ro = {m.get("destination") for m in attestation.get("mounts") or []
                       if m.get("read_only")}
        writable = linkage.get("writable") or {}
        if [d for d in attested_ro if writable.get(d) is True]:
            return "ATTESTATION_MOUNT_MODE_DRIFT"
    # the running image must be the CONTRACTED one, not merely present
    if config.get("IMAGE") and not attestation.get("image_digest"):
        return "ATTESTATION_INVALID"
    contracted = config.get("IMAGE_DIGEST")
    if contracted and attestation.get("image_digest") != contracted:
        return "ATTESTATION_IMAGE_DRIFT"
    return ""
'''


def _reference():
    namespace: dict = {}
    exec(compile(REFERENCE_VERIFIER, "<attestation-reference>", "exec"), namespace)  # noqa: S102
    return namespace


def _reference_digest(record):
    return _reference()["digest"](record)


# label -> (record mutation, verifier kwargs, expected refusal code)
ATTESTATION_CASES = {
    "stale": (dict(attested_at=_iso(3600)), {}, "ATTESTATION_STALE"),
    "container_id_mismatch": (dict(container_id="d" * 64),
                              dict(config={"_attested_container_id": "c" * 64}),
                              "ATTESTATION_WRONG_CONTAINER"),
    "missing_device": (dict(host_config_devices=[
                            {"path_in_container": "/dev/davinci0"}]),
                        dict(device_ids=(0, 1)), "ATTESTATION_DEVICE_NOT_BOUND"),
    "pseudo_character_node": (dict(),
                              dict(node_facts={"/dev/davinci1": "NOTCHAR"}),
                              "ATTESTATION_NODE_NOT_CHAR"),
    "missing_mount": (dict(mounts=[{"destination": "/home/s00653124",
                                    "read_only": False}]),
                       {}, "ATTESTATION_DRIVER_MOUNT_MISSING"),
    "image_drift": (dict(image_digest="sha256:" + "e" * 64),
                    dict(config={"IMAGE_DIGEST": "sha256:" + "a" * 64}),
                    "ATTESTATION_IMAGE_DRIFT"),
    "tampered_hash": (dict(), {}, "ATTESTATION_TAMPERED"),
}


def test_reference_verifier_discriminates_every_negative():
    """The negative matrix has discriminating power, proven against a reference.

    This is the "can this probe fail" control for the whole deployment boundary:
    each defect must be caught for its OWN reason and a clean record must pass. If
    this goes red, the negatives above are decoration and prove nothing.
    """
    reference = _reference()
    clean = _attest()
    require(reference["verify"](clean, {}, (0, 1), None, None) == "",
            "the reference refused a well-formed record (%r); every negative below would "
            "then pass by refusing everything"
            % (reference["verify"](clean, {}, (0, 1), None, None),))
    for label in sorted(ATTESTATION_CASES):
        mutation, kwargs, expected = ATTESTATION_CASES[label]
        record = _attest(**mutation)
        if label == "tampered_hash":
            # Edit AFTER sealing, and do NOT re-seal: re-sealing would repair exactly
            # the tampering this case exists to detect.
            record["host_config_devices"] = record["host_config_devices"][:1]
        config = {"CONTAINER_NAME": "fake"}
        config.update(kwargs.get("config") or {})
        device_ids = kwargs.get("device_ids", (0, 1))
        refused = reference["verify"](record, config, device_ids,
                                      kwargs.get("node_facts"), kwargs.get("observe"))
        require(refused != "",
                "the reference ACCEPTED the %s record, so that negative proves nothing"
                % label)
        require(expected in refused,
                "the %s record was refused as %r, not %s, so that case does not isolate "
                "its defect" % (label, refused, expected))

def main() -> int:
    print("P0-E/F/G/H behavioural contract against the gate at:")
    print("  %s" % GATE_PATH)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print()
        print("GAPS (each line is a behaviour that did not happen):")
        for item in FAIL:
            print("  - " + item)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())