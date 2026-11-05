#!/usr/bin/env python3
"""CPU-only negative controls for the P0-A environment preflight and P0-DEPLOY
verification.

Every case here is a REAL failure mode already observed on this machine:
  - npu-smi exit 127 from a missing libc_sec.so, reported as a device problem;
  - a CANN_SET_ENV that was configured but absent, or present but failing to source;
  - a container whose npu-smi saw "no running processes" while the host saw four.

None of these tests touches a device. They drive the gate's own functions with
stubbed container_exec so a regression fails loudly on the host.

Run: python -B agent_gates/adapters/tilefoundry-npu/tests/test_environment_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT_DIR = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import tilefoundry_gate as G  # noqa: E402

PASS: list = []
FAIL: list = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
    except Exception as exc:  # noqa: BLE001
        FAIL.append("%s: %s: %s" % (name, type(exc).__name__, exc))


def stubs(stdout="", returncode=0):
    """Replace container_exec so the gate runs with no container and no device."""
    calls = []

    def fake(config, argv, **kwargs):
        calls.append((argv, kwargs))

        class R:
            pass

        r = R()
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = ""
        return r

    return fake, calls


def with_stub(fn, *args, **kwargs):
    original = G.container_exec
    fake, _ = stubs(*args, **kwargs)
    G.container_exec = fake
    try:
        return fn()
    finally:
        G.container_exec = original


def expect_blocked(fn, needle, code=None):
    """The contract: a named ENV_* fault, BLOCKED_ENV, and never a device verdict."""
    try:
        fn()
    except G.GateError as exc:
        assert needle in str(exc), "wrong message: %s" % exc
        assert exc.code == G.EXIT_BLOCKED, (
            "an environment fault must be BLOCKED_ENV, got %s" % exc.code)
        if code is not None:
            assert code in str(exc), "missing named code %s in: %s" % (code, exc)
        # The message may MENTION busy, but only to deny it. What must never happen is
        # the gate ATTRIBUTING the failure to occupancy -- that is precisely how the
        # libc_sec exit-127 fault was originally misreported as a device problem.
        low = str(exc).lower()
        for attributing in ("is busy", "card is busy", "device is busy",
                            "because it is busy", "was busy", "occupied by"):
            assert attributing not in low, (
                "an environment fault must never be ATTRIBUTED to occupancy: %s" % exc)
        return str(exc)
    raise AssertionError("expected BLOCKED_ENV containing %r, none raised" % needle)


def with_deploy_stubs(inspect, node_stdout, image_id="sha256:cafe"):
    """Patch every way deployment_verify can reach the outside world.

    All three are stubbed so this file never shells out to docker: a unit test that
    silently depends on a live daemon is a test that passes for the wrong reason on
    a machine that happens to have one.
    """
    class _Ctx:
        def __enter__(self):
            self.saved = (G._host_inspect, G.container_exec, G.run)
            G._host_inspect = lambda config: inspect
            G.container_exec = lambda *a, **k: type("R", (), {
                "returncode": 0, "stderr": "", "stdout": node_stdout})()
            G.run = lambda argv, **k: type("R", (), {
                "returncode": 0, "stderr": "", "stdout": image_id + "\n"})()
            return self

        def __exit__(self, *exc):
            G._host_inspect, G.container_exec, G.run = self.saved
            return False

    return _Ctx()


def cfg(**over):
    base = {"CONTAINER_NAME": "tf-test", "CANN_SET_ENV": "/data/cann/set_env.sh",
            "TF_REPO": "/tmp", "IMAGE": "img", "CANN_ROOT": "/data/cann",
            "IN_CONTAINER_PYTHON": "/usr/bin/python3", "CONTAINER_USER": "u"}
    base.update(over)
    return base


# --- NEGATIVE: empty CANN_SET_ENV must NOT silently use the ambient environment --

def test_empty_cann_is_refused_by_name():
    expect_blocked(lambda: with_stub(lambda: G.environment_preflight(cfg(CANN_SET_ENV="")),
                                     stdout="", returncode=0),
                   "CANN_SET_ENV is not set", G.ENV_CANN_SCRIPT_MISSING)


def test_empty_cann_never_runs_a_command():
    fake, calls = stubs()
    original = G.container_exec
    G.container_exec = fake
    try:
        try:
            G.environment_preflight(cfg(CANN_SET_ENV=""))
        except G.GateError:
            pass
    finally:
        G.container_exec = original
    assert not calls, "an empty CANN must be refused before anything is executed"


# --- NEGATIVE: the historic libc_sec exit-127 fault, classified as ENVIRONMENT --

def test_missing_cann_script_is_named():
    expect_blocked(lambda: with_stub(lambda: G.environment_preflight(cfg()),
                                     stdout="FAIL-CANN-MISSING", returncode=11),
                   "does not exist inside the container", G.ENV_CANN_SCRIPT_MISSING)


def test_failed_source_is_named():
    expect_blocked(lambda: with_stub(lambda: G.environment_preflight(cfg()),
                                     stdout="FAIL-CANN-SOURCE", returncode=12),
                   "returned non-zero", G.ENV_CANN_SOURCE_FAILED)


def test_unresolved_ascend_home_is_named():
    expect_blocked(lambda: with_stub(lambda: G.environment_preflight(cfg()),
                                     stdout="FAIL-HOME-UNSET", returncode=13),
                   "ASCEND_HOME_PATH", G.ENV_ASCEND_HOME_UNRESOLVED)


def test_missing_ld_library_path_is_named():
    expect_blocked(lambda: with_stub(lambda: G.environment_preflight(cfg()),
                                     stdout="FAIL-LD-PATH", returncode=15),
                   "LD_LIBRARY_PATH", G.ENV_LD_LIBRARY_PATH_MISSING)


def test_missing_libc_sec_is_an_environment_fault_not_a_device_fault():
    msg = expect_blocked(
        lambda: with_stub(lambda: G.environment_preflight(cfg()),
                          stdout="FAIL-LIB libc_sec.so", returncode=16),
        "libc_sec.so", G.ENV_LIBRARY_UNRESOLVED)
    assert "not a device" in msg, msg


def test_missing_driver_libraries_are_all_named():
    msg = expect_blocked(
        lambda: with_stub(
            lambda: G.environment_preflight(cfg()),
            stdout="FAIL-LIB libc_sec.so\nFAIL-LIB libascend_hal.so",
            returncode=16),
        "libascend_hal.so", G.ENV_LIBRARY_UNRESOLVED)
    assert "libc_sec.so" in msg and "libascend_hal.so" in msg, msg


def test_smi_failure_after_preflight_is_environment_not_busy():
    """The exact historic misreport: npu-smi exit 127 must never read as a card fault."""
    seq = ["PREFLIGHT-OK\nHOME=/data/cann/ascend-toolkit", ""]
    codes = [0, 127]
    state = {"i": 0}

    def fake(config, argv, **kwargs):
        i = state["i"]
        state["i"] = i + 1

        class R:
            pass

        r = R()
        r.returncode = codes[min(i, len(codes) - 1)]
        r.stdout = seq[min(i, len(seq) - 1)]
        r.stderr = ""
        return r

    original = G.container_exec
    G.container_exec = fake
    try:
        msg = expect_blocked(lambda: G.environment_preflight(cfg()),
                             "npu-smi failed with status 127", G.ENV_SMI_EXEC_FAILED)
    finally:
        G.container_exec = original
    low = msg.lower()
    assert "environment fault" in low, msg
    assert "not reported as a busy card" in low, msg
    assert "not as a device failure" in low, msg


def test_preflight_without_ok_marker_is_refused():
    expect_blocked(lambda: with_stub(lambda: G.environment_preflight(cfg()),
                                     stdout="something else entirely", returncode=0),
                   "did not report success", G.ENV_LIBRARY_UNRESOLVED)


def test_successful_preflight_reports_verified_facts():
    out = ("HOME=/data/cann/ascend-toolkit\n"
           "LIB libc_sec.so /usr/lib64/libc_sec.so /data/cann/lib64/libc_sec.so %s\n"
           "LIB libascend_hal.so /usr/lib64/libascend_hal.so /usr/lib64/libascend_hal.so %s\n"
           "PREFLIGHT-OK\nDEVICE 0 OK")
    sha = "a" * 64
    facts = with_stub(lambda: G.environment_preflight(cfg()),
                      stdout=out % (sha, sha), returncode=0)
    assert facts["ascend_home_path"] == "/data/cann/ascend-toolkit", facts
    assert facts["libraries"]["libc_sec.so"]["resolved"] == "/usr/lib64/libc_sec.so", facts
    # The real path is recorded SEPARATELY: a symlinked library can resolve to a
    # different build than the name suggests, and that difference must be visible.
    assert (facts["libraries"]["libc_sec.so"]["realpath"]
            == "/data/cann/lib64/libc_sec.so"), facts
    assert facts["libraries"]["libc_sec.so"]["sha256"] == sha, facts
    assert facts["smi_probe_exit"] == 0


def test_unreadable_library_hash_is_recorded_not_silently_empty():
    out = ("HOME=/data/cann\n"
           "LIB libc_sec.so /usr/lib64/libc_sec.so /usr/lib64/libc_sec.so UNREADABLE\n"
           "PREFLIGHT-OK")
    facts = with_stub(lambda: G.environment_preflight(cfg()), stdout=out, returncode=0)
    assert facts["libraries"]["libc_sec.so"]["sha256"] == "UNREADABLE", facts


def test_smi_absence_is_a_named_environment_fault():
    expect_blocked(lambda: with_stub(lambda: G.environment_preflight(cfg()),
                                     stdout="FAIL-SMI-ABSENT", returncode=17),
                   "environment preflight exited 17", G.ENV_LIBRARY_UNRESOLVED)


def test_wedged_smi_is_bounded_and_named_not_a_raw_timeout():
    """This machine HAS hung npu-smi; it must surface as ENV_SMI, never a traceback."""
    import subprocess
    original = G.container_exec
    state = {"n": 0}

    def fake(config, argv, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            class R:
                returncode = 0
                stderr = ""
                stdout = "PREFLIGHT-OK\nHOME=/data/cann"
            return R()
        raise subprocess.TimeoutExpired(cmd="npu-smi", timeout=60)

    G.container_exec = fake
    try:
        msg = expect_blocked(lambda: G.environment_preflight(cfg()),
                             "did not return within", G.ENV_SMI_EXEC_FAILED)
    finally:
        G.container_exec = original
    assert "busy" in msg.lower() and "not a busy" in msg.lower(), msg


# --- P0-DEPLOY negatives -------------------------------------------------------

def _full_binding(extra=None):
    devices = [{"PathInContainer": "/dev/davinci0"},
               {"PathInContainer": "/dev/davinci1"},
               {"PathInContainer": "/dev/davinci_manager"},
               {"PathInContainer": "/dev/hisi_hdc"}]
    if extra:
        devices = extra
    return {"Id": "abc", "Image": "sha256:cafe",
            "HostConfig": {"Devices": devices}}


ALL_CHAR = ("CHAR /dev/davinci0\nCHAR /dev/davinci1\n"
            "CHAR /dev/davinci_manager\nCHAR /dev/hisi_hdc\n")


def test_missing_binding_is_a_deployment_fault():
    """A requested device absent from HostConfig.Devices is DEPLOYMENT, never busy."""
    inspect = _full_binding([{"PathOnContainer": "/dev/davinci0",
                              "PathOnHost": "/dev/davinci0"}])
    with with_deploy_stubs(inspect, ALL_CHAR):
        msg = expect_blocked(lambda: G.deployment_verify(cfg(), [0, 1]),
                             "/dev/davinci1", "CONTAINER_DEVICE_BINDING_MISSING")
    assert "DEPLOYMENT" in msg, msg


def test_missing_control_node_is_a_deployment_mismatch():
    inspect = _full_binding([{"PathInContainer": "/dev/davinci0"},
                             {"PathInContainer": "/dev/davinci1"}])
    with with_deploy_stubs(inspect, ALL_CHAR):
        expect_blocked(lambda: G.deployment_verify(cfg(), [0, 1]),
                       "control node", "CONTAINER_DEPLOYMENT_MISMATCH")


def test_regular_file_named_davinci_is_refused():
    """A regular file called davinci0 is not a device; the audit asked for this."""
    with with_deploy_stubs(_full_binding(),
                           "NOTCHAR /dev/davinci0\nCHAR /dev/davinci1\n"
                           "CHAR /dev/davinci_manager\nCHAR /dev/hisi_hdc\n"):
        msg = expect_blocked(lambda: G.deployment_verify(cfg(), [0, 1]),
                             "regular file", "CONTAINER_DEVICE_BINDING_MISSING")
    assert "not a device" in msg, msg


def test_absent_driver_node_is_a_mount_fault():
    with with_deploy_stubs(_full_binding(),
                           "MISSING /dev/hisi_hdc\nCHAR /dev/davinci0\n"
                           "CHAR /dev/davinci1\nCHAR /dev/davinci_manager\n"):
        expect_blocked(lambda: G.deployment_verify(cfg(), [0, 1]),
                       "hisi_hdc", "DRIVER_MOUNT_MISSING")


def test_stale_image_is_refused():
    """A container left over from an older image is a STALE DEPLOYMENT."""
    inspect = _full_binding()
    inspect["Image"] = "sha256:oldcontainer"
    with with_deploy_stubs(inspect, ALL_CHAR, image_id="sha256:newimage"):
        msg = expect_blocked(lambda: G.deployment_verify(cfg(), [0, 1]),
                             "STALE DEPLOYMENT", "CONTAINER_IMAGE_MISMATCH")
    assert "predates" in msg, msg


def test_matching_image_is_accepted():
    with with_deploy_stubs(_full_binding(), ALL_CHAR, image_id="sha256:cafe"):
        facts = G.deployment_verify(cfg(), [0, 1])
    assert facts["verified"] is True, facts
    assert facts["image_id"] == "sha256:cafe", facts
    assert facts["schema_version"] == G.DEPLOYMENT_SCHEMA_VERSION


def test_verified_is_false_when_host_inspect_is_unavailable():
    """Inside the container there is no host view. Claiming verified=True there
    would assert a device BINDING that was never checked -- the node-existence check
    cannot prove the container was CREATED with those nodes bound."""
    original = G._host_inspect
    original_exec = G.container_exec
    G._host_inspect = lambda config: {}
    G.container_exec = lambda *a, **k: type("R", (), {
        "returncode": 0, "stderr": "",
        "stdout": "CHAR /dev/davinci1\nCHAR /dev/davinci_manager\nCHAR /dev/hisi_hdc\n"})()
    try:
        facts = G.deployment_verify(cfg(), [1])
    finally:
        G._host_inspect = original
        G.container_exec = original_exec
    assert facts["verified"] is False, facts
    assert facts["host_inspect"] == "UNAVAILABLE", facts
    assert "not proven" in facts["verified_scope"], facts


def test_deployment_profile_is_tracked_valid_and_secret_free():
    """The contract must be versioned and must not carry machine-local facts."""
    import json as _json
    profile = _json.loads(G.DEPLOYMENT_PROFILE_PATH.read_text())
    assert profile["schema_version"] == G.DEPLOYMENT_SCHEMA_VERSION
    assert profile["required_control_nodes"] == list(G.REQUIRED_CONTROL_NODES)
    text = _json.dumps(profile)
    for secretish in ("/home/", "/data/pri", "s00653124", "y00897153"):
        assert secretish not in text, "profile leaks machine-local fact %r" % secretish


def test_missing_deployment_profile_is_a_config_error():
    original = G.DEPLOYMENT_PROFILE_PATH
    G.DEPLOYMENT_PROFILE_PATH = Path("/nonexistent/deployment_profile.json")
    try:
        try:
            G.load_deployment_profile()
        except G.GateError as exc:
            assert exc.code == G.EXIT_CONFIG, exc.code
            assert "deployment profile is missing" in str(exc), exc
        else:
            raise AssertionError("a missing profile must be refused")
    finally:
        G.DEPLOYMENT_PROFILE_PATH = original


def test_profile_libraries_match_the_gate_constant():
    """The profile drives the check, so it must not disagree with the fallback."""
    assert tuple(G.required_libraries()) == G.REQUIRED_LIBRARIES, (
        G.required_libraries(), G.REQUIRED_LIBRARIES)

def test_effective_config_hash_never_leaks_values():
    """The evidence may record WHICH keys were set, never their values."""
    h = G.effective_config_hash(cfg(PROXY_SH="/secret/path/proxy.sh"))
    assert len(h) == 64, h
    assert "secret" not in h and "proxy" not in h, h
    a = G.effective_config_hash(cfg(PROXY_SH="/a"))
    b = G.effective_config_hash(cfg(PROXY_SH="/b"))
    assert a == b, "the hash must cover the key set, not the values"


# --- the SCRIPT ITSELF, not just its parser output --------------------------
#
# These exist because a mutation study showed the parser-only tests stayed green when
# the readlink/hash step was deleted and when the tracked profile was ignored. A test
# that never looks at the shell cannot notice the shell losing a step.

def test_preflight_script_hashes_and_realpaths_each_library():
    script = G.build_preflight_script("/data/cann/set_env.sh")
    assert "readlink -f" in script, "the real path must be resolved, not assumed"
    assert "sha256sum" in script, "each library must be content-hashed"
    assert "UNREADABLE" in script, "a failed hash must be recorded, not silently blank"
    assert 'echo "LIB $lib $p $rp $h"' in script, (
        "the resolved path, real path and hash must all be emitted")


def test_preflight_script_uses_the_tracked_profile_libraries():
    """The profile drives the check, so the script must list the profile's names."""
    profile = G.load_deployment_profile()
    script = G.build_preflight_script("/data/cann/set_env.sh")
    for lib in profile["required_driver_libraries"]:
        assert lib in script, "profile library %r missing from the preflight script" % lib


def test_the_profile_actually_drives_the_script_not_the_constant():
    """Prove the tracked profile is a live input, not decoration.

    The two lists happen to agree today, so simply deleting the profile lookup is
    behaviourally invisible. This test replaces the profile with a DIFFERENT set and
    asserts the script follows it -- which is only true if the profile is consulted.
    """
    import json as _json
    import shutil
    import tempfile
    original = G.DEPLOYMENT_PROFILE_PATH
    tmpdir = Path(tempfile.mkdtemp())
    try:
        profile = _json.loads(original.read_text())
        profile["required_driver_libraries"] = ["libdecoy_one.so", "libdecoy_two.so"]
        fake = tmpdir / "deployment_profile.json"
        fake.write_text(_json.dumps(profile))
        G.DEPLOYMENT_PROFILE_PATH = fake
        script = G.build_preflight_script("/data/cann/set_env.sh")
        assert "libdecoy_one.so" in script, (
            "the preflight ignored the tracked profile; it is hardcoded somewhere")
        assert "libdecoy_two.so" in script, script
        assert "libc_sec.so" not in script, (
            "the preflight still used the built-in list instead of the profile")
    finally:
        G.DEPLOYMENT_PROFILE_PATH = original
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_preflight_script_uses_the_loader_not_ldconfig():
    """Regression: ldconfig-only reporting was wrong for this machine."""
    script = G.build_preflight_script("/data/cann/set_env.sh")
    assert "ldd" in script, "resolution must go through the loader"
    assert "ldconfig" not in script, (
        "ldconfig alone misses LD_LIBRARY_PATH-only libraries such as libc_sec.so")
    assert '"not found"' in script, "an unresolved library must be detected explicitly"


def test_preflight_script_requires_sourcing_to_succeed():
    script = G.build_preflight_script("/data/cann/set_env.sh")
    assert "FAIL-CANN-SOURCE" in script
    assert "exit 12" in script
    # The failure must not be discarded. The old occupancy prefix sent both streams
    # to /dev/null and could not tell a broken source from a healthy one.
    assert ">/dev/null 2>&1 || { echo FAIL-CANN-SOURCE" in script, script


def test_preflight_script_is_actually_executable_shell():
    """Run it for real: a script that cannot even parse is not a check."""
    import subprocess
    script = G.build_preflight_script("/definitely/not/here.sh")
    done = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 11, (done.returncode, done.stdout, done.stderr)
    assert "FAIL-CANN-MISSING" in done.stdout, done.stdout


def test_preflight_script_reports_success_and_facts_on_a_real_run():
    """End-to-end on THIS host: a real file, real ldconfig/ldd, real hashing."""
    import subprocess
    import shutil
    import tempfile
    smi = shutil.which("npu-smi") or "/usr/local/sbin/npu-smi"
    if not (Path(smi).exists()):
        print("    SKIP: npu-smi is not present on this host")
        return
    # A real, regular, world-readable file: /dev/null is a CHARACTER device and the
    # preflight correctly refuses it under -f, which is the behaviour under test
    # elsewhere. Using it here would have tested the wrong branch.
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write("export ASCEND_HOME_PATH=/tmp\n"
                     "export LD_LIBRARY_PATH=/tmp/lib64:$LD_LIBRARY_PATH\n")
        cann = handle.name
    try:
        done = subprocess.run(["sh", "-c", G.build_preflight_script(cann)],
                              capture_output=True, text=True, timeout=120)
    finally:
        Path(cann).unlink(missing_ok=True)
    if "APSPOOL" in done.stdout or "No such" in done.stderr:
        print("    SKIP: host driver environment not usable")
        return
    assert "LIB " in done.stdout, (done.returncode, done.stdout[-500:], done.stderr[-300:])
    for line in done.stdout.splitlines():
        if line.startswith("LIB "):
            parts = line.split()
            assert len(parts) == 5, "LIB line must carry name, path, realpath, hash: %r" % line
            assert len(parts[4]) == 64, "hash must be a sha256: %r" % line


# --- HARD RULE: ask the loader, not the cache index --------------------------
#
# This is the rule the audit asked to be pinned, and the test below is the
# discriminating one: a cache-index check must MISFIRE ON A HEALTHY MACHINE.

def test_hard_rule_cache_resolver_misfires_where_loader_succeeds():
    """THE discriminating test: on a HEALTHY environment the cache check misfires.

    Both resolvers run over the SAME real npu-smi in the SAME environment. If the
    cache-based mechanism ever stops reporting a missing library where the loader
    resolves it fine, the justification for the ldd rule is gone and it must be
    re-derived rather than kept on faith.

    Runs in the CONTAINER when reachable, because that is where the case is real:
    libc_sec.so is on npu-smi path via LD_LIBRARY_PATH but absent from the container
    ldconfig cache. This host happens to have it cached, so testing only the host
    would pass without demonstrating anything -- which the first version of this
    test did.
    """
    import shutil
    import subprocess

    def resolution_in(prefix):
        out = {}
        for kind in ("loader", "cache"):
            script = G.build_preflight_script(
                "/data/pri/cann-9.2.0/ascend-toolkit/set_env.sh", resolver=kind)
            done = subprocess.run(prefix + ["sh", "-c", script],
                                  capture_output=True, text=True, timeout=180)
            out[kind] = (done.returncode, done.stdout)
        return out

    if not shutil.which("docker"):
        print("    SKIP: docker unavailable")
        return
    alive = subprocess.run(["docker", "exec", "-u", "root", "tf-ascend950pr",
                            "sh", "-c", "true"], capture_output=True, timeout=60)
    if alive.returncode != 0:
        print("    SKIP: container not reachable; the host does not exhibit the case")
        return
    result = resolution_in(["docker", "exec", "-u", "root", "tf-ascend950pr"])
    loader_rc, loader_out = result["loader"]
    cache_rc, cache_out = result["cache"]
    assert loader_rc == 0, "the loader must ACCEPT this healthy environment: %r" % (
        loader_out[-300:],)
    assert "PREFLIGHT-OK" in loader_out
    assert cache_rc != 0, (
        "the cache resolver unexpectedly ACCEPTED the environment, so it shows no "
        "weakness and the ldd rule loses its justification")
    assert "FAIL-LIB" in cache_out, (
        "expected the cache resolver to name a missing library on a healthy "
        "machine; got %r" % cache_out[-300:])
    print("    proven: loader=PASS but cache=FAIL (%s) on the same healthy npu-smi"
          % cache_out.strip().splitlines()[-1][:44])

def test_loader_rule_is_documented_as_a_hard_rule():
    """The hard rule must be written where the next maintainer will see it."""
    src = (SCRIPT_DIR / "tilefoundry_gate.py").read_text()
    assert "HARD RULE" in src, "the loader rule must be stated as a rule, not a note"


def test_no_idle_wait_primitives_in_the_gate():
    """The shared-card policy: nothing may wait, poll, or retry until a card is free."""
    src = (SCRIPT_DIR / "tilefoundry_gate.py").read_text()
    for banned in ("while True", "time.sleep(", "wait_for_idle", "until_idle"):
        assert banned not in src, "found idle-wait primitive %r in the gate" % banned


# --- P0-DEPLOY attestation: eight negative controls -------------------------
#
# Each of these is a way the host half could lie to the container half. All are
# CPU-only: they operate on a synthetic attestation and never touch a device.

def _attestation(**over):
    """A well-formed attestation; tests mutate exactly one field."""
    body = {
        "schema_version": G.DEPLOYMENT_SCHEMA_VERSION,
        "kind": "host-deployment-attestation",
        "attested_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "container_name": "tf-test",
        "container_id": "a" * 64,
        "created": "2026-11-05T00:00:00.000000000Z",
        "image": "sha256:" + "b" * 64,
        "image_digest": "sha256:" + "b" * 64,
        "host_config_devices": [
            {"path_in_container": "/dev/davinci0", "path_on_host": "/dev/davinci0",
             "permissions": "rwm"},
            {"path_in_container": "/dev/davinci1", "path_on_host": "/dev/davinci1",
             "permissions": "rwm"},
            {"path_in_container": "/dev/davinci_manager",
             "path_on_host": "/dev/davinci_manager", "permissions": "rwm"},
            {"path_in_container": "/dev/hisi_hdc", "path_on_host": "/dev/hisi_hdc",
             "permissions": "rwm"},
        ],
        "mounts": [
            {"type": "bind", "source": "/data", "destination": "/data",
             "mode": "ro", "read_only": True},
            {"type": "bind", "source": "/home", "destination": "/home",
             "mode": "rw", "read_only": False},
            {"type": "bind", "source": "/usr/local/Ascend/driver",
             "destination": "/usr/local/Ascend/driver", "mode": "ro",
             "read_only": True},
        ],
        "config_visible_devices": "0,1",
        "privileged": False,
        "user": "",
    }
    body.update(over)
    canonical = __import__("json").dumps(body, sort_keys=True, separators=(",", ":"))
    body["content_sha256"] = __import__("hashlib").sha256(
        canonical.encode()).hexdigest()
    return body


def _reseal(attestation):
    """Recompute the hash as an EDITOR would, to test non-hash checks in isolation."""
    import hashlib as _h
    import json as _j
    body = {k: v for k, v in attestation.items() if k != "content_sha256"}
    attestation["content_sha256"] = _h.sha256(
        _j.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return attestation


def _linkage_ok(destinations):
    return {"checked": list(destinations), "missing": [],
            "writable": {"/data": False, "/home": True,
                         "/usr/local/Ascend/driver": False}}


def _verify(attestation, devices=(0, 1), observe=None, **cfg_over):
    return G.verify_attestation(attestation, cfg(**cfg_over), list(devices),
                                node_facts={"/dev/davinci0": "CHAR",
                                            "/dev/davinci1": "CHAR",
                                            "/dev/davinci_manager": "CHAR",
                                            "/dev/hisi_hdc": "CHAR"},
                                observe=observe)


def test_attestation_well_formed_is_accepted():
    """Positive control: with nothing wrong, all seven checks pass."""
    detail = _verify(_attestation(), observe=_linkage_ok)
    assert detail["mount_linkage"], detail
    assert detail["image_digest"].startswith("sha256:"), detail


def test_negative_stale_attestation_is_refused():
    """(1) An old attestation describes a container that may no longer exist."""
    import datetime as _dt
    old = (_dt.datetime.now(_dt.timezone.utc)
           - _dt.timedelta(seconds=G.ATTESTATION_MAX_AGE_SECONDS + 60)).isoformat()
    att = _attestation(attested_at=old)
    expect_blocked(lambda: _verify(att, observe=_linkage_ok),
                   "ATTESTATION_STALE", "ATTESTATION_STALE")


def test_negative_container_id_mismatch_is_refused():
    """(2) An attestation about a DIFFERENT container must not be accepted."""
    cfg_with_id = cfg()
    cfg_with_id["_attested_container_id"] = "c" * 64
    try:
        G.verify_attestation(_attestation(), cfg_with_id, [0, 1],
                             node_facts={"/dev/davinci0": "CHAR",
                                         "/dev/davinci1": "CHAR"},
                             observe=_linkage_ok)
    except G.GateError as exc:
        assert "ATTESTATION_WRONG_CONTAINER" in str(exc), exc
        assert exc.code == G.EXIT_BLOCKED
        return
    raise AssertionError("a container id mismatch must be refused")


def test_negative_requested_device_absent_from_attested_devices():
    """(3) Asking for davinci1 when Devices never bound it."""
    att = _attestation(host_config_devices=[
        {"path_in_container": "/dev/davinci0", "path_on_host": "/dev/davinci0",
         "permissions": "rwm"},
        {"path_in_container": "/dev/davinci_manager",
         "path_on_host": "/dev/davinci_manager", "permissions": "rwm"},
        {"path_in_container": "/dev/hisi_hdc", "path_on_host": "/dev/hisi_hdc",
         "permissions": "rwm"}])
    msg = expect_blocked(lambda: _verify(att, observe=_linkage_ok),
                         "/dev/davinci1", "ATTESTATION_DEVICE_NOT_BOUND")
    assert "not in the attested" in msg, msg


def test_negative_regular_file_pretending_to_be_davinci():
    """(4) A regular file named davinci1 is not a device."""
    def verify_with_notchar():
        return G.verify_attestation(
            _attestation(), cfg(), [0, 1],
            node_facts={"/dev/davinci0": "CHAR", "/dev/davinci1": "NOTCHAR"},
            observe=_linkage_ok)
    msg = expect_blocked(verify_with_notchar, "not character devices",
                         "ATTESTATION_NODE_NOT_CHAR")
    assert "regular file" in msg, msg


def test_negative_missing_driver_mount():
    """(5) No Ascend driver mount attested."""
    att = _attestation(mounts=[
        {"type": "bind", "source": "/data", "destination": "/data",
         "mode": "ro", "read_only": True}])
    expect_blocked(lambda: _verify(att, observe=_linkage_ok),
                   "no Ascend driver mount", "ATTESTATION_DRIVER_MOUNT_MISSING")


def test_negative_image_digest_missing():
    """(6) Image drift: the attestation carries no digest to compare."""
    att = _attestation(image_digest="")
    expect_blocked(lambda: _verify(att, observe=_linkage_ok),
                   "no image_digest", "ATTESTATION_INVALID")


def test_negative_tampered_attestation_hash_mismatch():
    """(7) Editing any field after sealing must invalidate the whole file."""
    att = _attestation()
    att["config_visible_devices"] = "0,1,2,3,4,5,6,7"   # edit WITHOUT resealing
    msg = expect_blocked(lambda: _verify(att, observe=_linkage_ok),
                         "ATTESTATION_TAMPERED", "ATTESTATION_TAMPERED")
    assert "modified after it was produced" in msg, msg


def test_negative_created_after_attested_at():
    """(8) An attestation cannot predate the container it describes."""
    att = _attestation(created="2099-01-01T00:00:00.000000000Z")
    expect_blocked(lambda: _verify(att, observe=_linkage_ok),
                   "AFTER the attestation time", "ATTESTATION_INCONSISTENT")


def test_negative_mount_not_linked_to_this_container():
    """(4c) An mount that is not present in here means this is another container."""
    def linkage(destinations):
        return {"checked": list(destinations), "missing": ["/usr/local/Ascend/driver"],
                "writable": {}}
    expect_blocked(lambda: _verify(_attestation(), observe=linkage),
                   "ATTESTATION_MOUNT_NOT_LINKED", "ATTESTATION_MOUNT_NOT_LINKED")


def test_negative_read_only_mount_is_actually_writable():
    """(4c) A mount attested ro but writable in here is mode drift."""
    def linkage(destinations):
        return {"checked": list(destinations), "missing": [],
                "writable": {"/data": True, "/home": True,
                             "/usr/local/Ascend/driver": False}}
    expect_blocked(lambda: _verify(_attestation(), observe=linkage),
                   "ATTESTATION_MOUNT_MODE_DRIFT", "ATTESTATION_MOUNT_MODE_DRIFT")


# --- npu-smi parsing: the function whose return I once deleted --------------
#
# Real output captured from this machine, so the parser is exercised against the
# actual table shape rather than a convenient invention.

SMI_SAMPLE = """
+-------------------------------------------------------------------------------------------------+
| npu-smi 25.7.rc1                 Version: 25.7.rc1                                              |
+--------+------------------+---------------+-----------------------------------------------------+
| NPU ID | Name             | Health        | Power(W)    Temp(C)           Hugepages-Usage(page) |
|        |                  | Bus-Id        | NPU Util(%) Memory-Usage(MB)  HBM-Usage(MB)         |
+========+==================+===============+=====================================================+
| 0      | Ascend950PR      | OK            | 463.8       70                0     / 0             |
|        |                  | 0000:55:00.0  | 100         0    / 0          16312 / 131072        |
+========+==================+===============+=====================================================+
| 1      | Ascend950PR      | OK            | 455.0       71                0     / 0             |
|        |                  | 0000:56:00.0  | 0           0    / 0          100   / 131072        |
+========+==================+===============+=====================================================+
| NPU ID | Process id       | Process name  | Process memory(MB)                                  |
+========+==================+===============+=====================================================+
| 1      | 12345            | python        | 2048                                                |
+========+==================+===============+=====================================================+
"""


def test_parse_npu_smi_returns_a_three_tuple_of_real_tables():
    """The contract is a THREE-tuple. Deleting the return once broke the device Gate.

    That deletion produced a ValueError only on a real run, because no CPU test called
    this function at all. This test exists so the absence of that return fails here.
    """
    parsed = G.parse_npu_smi(SMI_SAMPLE)
    assert parsed is not None, (
        "parse_npu_smi returned None: its return statement is missing")
    assert isinstance(parsed, tuple) and len(parsed) == 3, (
        "parse_npu_smi must return (health, holders, table_seen); got %r" % (parsed,))
    health, holders, table_seen = parsed
    assert health == {0: "OK", 1: "OK"}, health
    assert table_seen is True, table_seen
    assert 1 in holders and not holders.get(0), holders


def test_parse_npu_smi_unreadable_table_is_not_reported_as_idle():
    """An unparsable dump must yield NO health rows, never "no processes, idle"."""
    health, holders, table_seen = G.parse_npu_smi("complete garbage")
    assert health == {}, health
    assert holders == {}, holders
    assert table_seen is False, table_seen


def test_parse_npu_smi_health_row_is_not_confused_with_process_row():
    """A holder row must not be read as a health row (the old bug could)."""
    health, holders, table_seen = G.parse_npu_smi(SMI_SAMPLE)
    # The process row starts with id 1 too; it must not overwrite health.
    assert health[1] == "OK", health
    assert table_seen is True


# --- occupancy must REUSE the verified environment, not re-source ------------

def test_occupancy_reuses_the_verified_environment_verbatim():
    """The occupancy read must run in the SAME environment the preflight verified.

    A second `source` is a second environment -- not the one that was checked -- and
    the old open-coded prefix discarded the source result entirely. This pins the
    reuse: the exact verified values must reach npu-smi, and no source call may be
    issued for it.
    """
    captured = {}
    original = G.container_exec

    def fake(config, argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env")

        class R:
            pass

        r = R()
        r.returncode = 0
        r.stderr = ""
        r.stdout = SMI_SAMPLE
        return r

    verified = {"LD_LIBRARY_PATH": "/data/cann/lib64:/usr/local/Ascend/driver/lib64",
                "ASCEND_HOME_PATH": "/data/cann",
                "PATH": "/usr/local/sbin:/usr/bin"}
    G.container_exec = fake
    try:
        G.device_occupancy(cfg(), [0], {"env": verified})
    finally:
        G.container_exec = original

    assert captured["env"] == verified, (
        "occupancy did not receive the verified environment verbatim: %r"
        % (captured["env"],))
    command = " ".join(captured["argv"])
    assert "set_env.sh" not in command, (
        "occupancy re-sourced CANN instead of reusing the verified environment: %r"
        % command)
    assert "npu-smi" in command, command


def test_occupancy_refuses_without_a_verified_environment():
    """No verified env means NO silent fallback to an unverified source."""
    expect_blocked(lambda: G.device_occupancy(cfg(), [0], {}),
                   "did not return a verified environment", "ENV_LIBRARY_UNRESOLVED")


TESTS = [(k, v) for k, v in sorted(globals().items())
         if k.startswith("test_") and callable(v)]


def main() -> int:
    for name, fn in TESTS:
        check(name, fn)
    print("checks:", len(TESTS), "passed:", len(PASS), "failed:", len(FAIL))
    for line in FAIL:
        print("  FAIL", line)
    print("RESULT:", "PASS" if not FAIL else "FAIL")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())