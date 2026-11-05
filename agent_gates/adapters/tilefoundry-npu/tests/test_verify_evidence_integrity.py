"""Evidence integrity: test seams are not evidence, and the lock is released.

Three measured requirements:
  * a seam (TFNU_PROBE_OVERRIDE / --negative-control) must be opt-in, recorded, and
    its run must NOT carry a PASS conclusion;
  * env_evidence_check.py must reject such a run;
  * os._exit skips interpreter teardown, so the team flock must be proved
    RE-ACQUIRABLE after a verify process exits (the kernel closes the fd).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
SCRIPTS = PACK / "scripts"
sys.path.insert(0, str(SCRIPTS))

import env_evidence_check as E  # noqa: E402
import verify_environment as V  # noqa: E402


def _run_verify(args, extra_env):
    env = dict(os.environ)
    env.update(extra_env)
    for name in ("TFNU_VERIFY_REEXEC", "TFNU_VERIFY_LAUNCHED_BY",
                 "TFNU_ALLOW_TEST_SEAMS"):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_environment.py")] + args,
        env=env, capture_output=True, text=True, timeout=120)


def test_a_probe_override_without_the_switch_is_refused(tmp_path):
    result = _run_verify(
        ["--out", str(tmp_path / "out"), "--check-config"],
        {"TFNU_PROBE_OVERRIDE": "0=unresponsive"})
    assert result.returncode == 2, result.stderr[-400:]
    assert "TEST_SEAM_NOT_ALLOWED" in result.stdout, result.stdout[-400:]


def test_a_negative_control_without_the_switch_is_refused(tmp_path):
    result = _run_verify(
        ["--out", str(tmp_path / "out"), "--check-config",
         "--negative-control", "skip-compile"], {})
    assert result.returncode == 2, result.stderr[-400:]
    assert "TEST_SEAM_NOT_ALLOWED" in result.stdout


def test_env_evidence_check_rejects_a_seam_run(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"verdict": "PASS", "code": None,
                                "test_seam_used": False, "unreaped_children": []}))
    seam = tmp_path / "seam.json"
    seam.write_text(json.dumps({"verdict": "PASS_TEST_SEAM_NOT_EVIDENCE",
                                "code": "TEST_SEAM_NOT_EVIDENCE",
                                "test_seam_used": True, "unreaped_children": []}))
    assert E.check(str(good)) == []
    assert "TEST_SEAM_USED" in E.check(str(seam))
    assert E.main([str(good)]) == 0
    assert E.main([str(seam)]) == 1


def test_the_team_lock_is_reacquirable_after_an_os_exit(tmp_path):
    lock = tmp_path / "team.lock"
    child = (
        "import os, sys\n"
        "sys.path.insert(0, %r)\n"
        "import verify_environment as V\n"
        "V.acquire_team_lock(%r)\n"
        "os._exit(0)\n" % (str(SCRIPTS), str(lock)))
    subprocess.run([sys.executable, "-c", child], check=True, timeout=60)
    # If os._exit did NOT release the flock, this non-blocking acquire would fail.
    handle = V.acquire_team_lock(lock)
    handle.close()


def test_evidence_is_fsynced_and_unreaped_children_recorded():
    text = (SCRIPTS / "verify_environment.py").read_text()
    assert "os.fsync(handle.fileno())" in text, "the evidence file must be fsynced"
    assert "os.fsync(directory_fd)" in text, "the run directory must be fsynced"
    assert '"unreaped_children"' in text, "abandoned child PIDs must be in evidence"
