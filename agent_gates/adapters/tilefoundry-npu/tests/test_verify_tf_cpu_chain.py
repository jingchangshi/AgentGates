"""ENV-04: the TileFoundry CPU chain is decided by ARTEFACTS, not by exit 0.

The failure this pins is the classic false green: point the chain at `/bin/true`,
which exits 0 and does nothing, and a checker that trusts exit codes reports PASS
with no program checked, no analysis and no typed TIR.

Run against the REAL deployment to prove the positive direction:
    python -B tests/test_verify_tf_cpu_chain.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
SCRIPTS = PACK / "scripts"
sys.path.insert(0, str(SCRIPTS))

import env_context as E  # noqa: E402
import verify_environment as V  # noqa: E402


def _program(tmp_path: Path) -> Path:
    program = tmp_path / "checked_program.py"
    program.write_text(V.TINY_CHECKED_PROGRAM)
    return program


def test_bin_true_exits_zero_and_must_still_fail(tmp_path):
    result = V.run_tf_cpu_chain(["/bin/true"], _program(tmp_path), tmp_path,
                                env=dict(os.environ), timeout=30)
    assert not result["ok"], "a command that produces nothing was accepted: %r" % (result,)
    assert result["code"] == "TF_CHECKED_NO_STRUCTURED_RESULT", result
    # The point: the step DID exit 0, and the verdict is still FAIL.
    assert result["facts"]["steps"][0]["rc"] == 0, result["facts"]["steps"]
    assert result["facts"]["steps"][0]["structured"] is False


def test_reports_without_a_typed_artifact_must_fail(tmp_path):
    """checked and analyze can print valid JSON; finalize writing nothing still fails."""
    fake = tmp_path / "fake_tf.py"
    fake.write_text(
        "import json, sys\n"
        "argv = sys.argv[1:]\n"
        "if 'finalize' in argv:\n"
        "    raise SystemExit(0)  # exit 0, write NOTHING\n"
        "if 'analyze' in argv:\n"
        "    print(json.dumps({'executed': ['compute-cost'],"
        " 'functions': [{'name': 'copy1'}]}))\n"
        "    raise SystemExit(0)\n"
        "print(json.dumps({'module': 'CopySlice', 'entry': 'copy1', 'functions': 1}))\n")
    result = V.run_tf_cpu_chain([sys.executable, str(fake)], _program(tmp_path),
                                tmp_path, env=dict(os.environ), timeout=30)
    assert not result["ok"], result
    assert result["code"] == "TF_FINALIZE_NO_TYPED_ARTIFACT", result
    assert result["facts"]["steps"][-1]["rc"] == 0
    assert result["facts"]["finalized_artifact"]["bytes"] == 0
    assert result["facts"]["finalized_artifact"]["typed"] is False


def test_a_structured_report_without_analysis_fails(tmp_path):
    """An analyze that exits 0 with an empty report is not evidence."""
    fake = tmp_path / "fake_empty_analysis.py"
    fake.write_text(
        "import json, sys\n"
        "argv = sys.argv[1:]\n"
        "if 'analyze' in argv:\n"
        "    print(json.dumps({'executed': [], 'functions': []}))\n"
        "    raise SystemExit(0)\n"
        "print(json.dumps({'module': 'CopySlice', 'entry': 'copy1', 'functions': 1}))\n")
    result = V.run_tf_cpu_chain([sys.executable, str(fake)], _program(tmp_path),
                                tmp_path, env=dict(os.environ), timeout=30)
    assert not result["ok"], result
    assert result["code"] == "TF_ANALYZE_NO_STRUCTURED_RESULT", result


def test_the_script_reexecs_under_the_deployment_interpreter(tmp_path):
    """The canonical recipe launches python3; the deployment interpreter must
    still be the one that measures. A fake prefix proves the re-exec happens before
    any phase or run directory exists, so no device and no real install is touched."""
    prefix = tmp_path / "fake-prefix"
    (prefix / "bin").mkdir(parents=True)
    marker = tmp_path / "executed.marker"
    wrapper = prefix / "bin" / "python"
    wrapper.write_text('#!/bin/sh\nprintf %s "$0" > "{}"\nexit 0\n'.format(marker))
    wrapper.chmod(0o755)

    env = dict(os.environ)
    env["TFNU_PREFIX"] = str(prefix)
    env.pop("TFNU_VERIFY_REEXEC", None)
    env.pop("TFNU_VERIFY_LAUNCHED_BY", None)
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_environment.py"),
         "--out", str(out), "--skip-device"],
        env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert marker.is_file(), (
        "the deployment interpreter was not exec'd; the script kept the launcher "
        "interpreter: " + proc.stderr[-400:])
    assert not out.exists(), (
        "the re-exec must happen BEFORE a run directory is created; work was done "
        "under the launching interpreter")


def test_the_real_chain_produces_a_typed_artifact(tmp_path):
    """Positive control: the deployment interpreter really runs the chain."""
    interpreter = E.interpreter()
    if not interpreter.exists():
        pytest.skip("no deployment interpreter at %s" % interpreter)
    if not (E.TF_ROOT / "src" / "tilefoundry").is_dir():
        pytest.skip("no TileFoundry source at %s" % E.TF_ROOT)
    env = E.runtime_env(base=dict(os.environ))
    result = V.run_tf_cpu_chain([str(interpreter)], _program(tmp_path), tmp_path,
                                env=env, timeout=180)
    assert result["ok"], result
    facts = result["facts"]
    assert facts["steps"][0]["structured"] is True
    assert facts["analyze"]["executed"], facts["analyze"]
    assert facts["analyze"]["functions"] >= 1
    artifact = facts["finalized_artifact"]
    assert artifact["typed"] is True, artifact
    assert artifact["bytes"] >= 200, artifact
    assert artifact["sha256"], artifact
    assert artifact["contains_prim_func"] and artifact["contains_tensor_view"], artifact
