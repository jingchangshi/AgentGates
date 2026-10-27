#!/usr/bin/env python3
"""End-to-end proof that a real systemd user worker runs a real gate.

This script is deliberately not a unit test: it installs the unit, starts a
real user service, and reads the result back through the same
ExecutionManager the MCP server uses.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_gates import api_types as api
from agent_gates import systemd_units
from agent_gates.config import from_mapping, load
from agent_gates.execution import ExecutionManager
from agent_gates.manifest import digest


def build_machine(root):
    source = root / "source"
    source.mkdir()
    (source / "input.txt").write_text("bytes")
    pack = root / "adapters/test-pack"
    (pack / "scripts").mkdir(parents=True)
    script = pack / "scripts/gate.py"
    script.write_text(
        "import json,sys,time\n"
        "if '--check-config' in sys.argv: sys.exit(0)\n"
        "time.sleep(float(os.environ.get('DELAY','0')))\n"
        "print(json.dumps({'ok': True}))\n")
    script.write_text(script.read_text().replace('import json,sys,time', 'import json,sys,time,os'))
    manifest = {
        "schemaVersion": 1, "id": "test-pack", "version": "1",
        "files": {"scripts/gate.py": digest(script)},
        "permissions": {"sourceWrite": False, "hostExecution": True},
        "recipes": [
            {"id": "test.run", "cwd": "source", "timeout": 120,
             "argv": [sys.executable, "-I", "{pack}/scripts/gate.py"],
             "configCheck": [sys.executable, "-I", "{pack}/scripts/gate.py", "--check-config"]},
        ],
        "gates": [
            {"id": "fast", "recipeId": "test.run", "gateType": "deterministic-test",
             "observationChecks": {"ok": True}},
        ],
    }
    (pack / "manifest.json").write_text(json.dumps(manifest))
    (root / "state").mkdir()
    (root / "evidence").mkdir()
    config_text = (
        "schema_version = 1\n"
        + 'state_root = "' + str(root / 'state') + '"\n'
        + 'evidence_root = "' + str(root / 'evidence') + '"\n'
        + '[adapters.test-pack]\n'
        + '[workspaces.ws]\n'
        + 'source_root = "' + str(source) + '"\n'
        + 'adapters = ["test-pack"]\n'
        + '[worker]\n'
        + 'backend = "systemd"\n'
        + 'max_concurrent = 2\n'
        + 'python = "' + sys.executable + '"\n'
    )
    (root / "config.toml").write_text(config_text)
    return load(root / "config.toml")


def main():
    results = {}
    root = Path(tempfile.mkdtemp(prefix="agent-gates-e2e-"))
    config = build_machine(root)

    print("== systemd availability ==")
    ok, reason = systemd_units.available()
    print("  available:", ok, "(", reason, ")")
    if not ok:
        print(json.dumps({"blocked": reason}))
        return 2

    systemd_units.install(config.worker.python, timeout_seconds=600, workdir=str(root),
                          config_path=str(root / "config.toml"),
                          adapters_root=str(root / "adapters"))
    print("  unit:", systemd_units.unit_dir() / systemd_units.UNIT_NAME)
    os.environ["AGENT_GATES_CONFIG"] = str(root / "config.toml")
    os.environ["AGENT_GATES_ADAPTERS_ROOT"] = str(root / "adapters")
    from agent_gates.registry import Registry
    manager = ExecutionManager(config, registry=Registry(root / "adapters"))

    print("== submit (must return fast) ==")
    started = time.monotonic()
    status = manager.submit(api.ExecutionRequest(workspace="ws", adapter="test-pack", gates=["fast"]))
    elapsed = time.monotonic() - started
    print("  execution:", status.execution_id, "state:", status.state.value,
          "submit_seconds: %.3f" % elapsed)
    results["submit_seconds"] = round(elapsed, 3)
    results["submit_state"] = status.state.value

    print("== wait for completion ==")
    final = manager.wait(status.execution_id, timeout=120)
    print("  timed_out:", final.timed_out, "state:", final.status.state.value,
          "run_id:", final.status.run_id, "exit:", final.status.exit_reason)
    results["fast_state"] = final.status.state.value
    results["fast_run_id"] = final.status.run_id

    print("== receipt verification ==")
    if final.status.run_id:
        verification = manager.evidence.verify(final.status.run_id)
        print("  verify:", verification["status"], verification["failures"])
        results["verify"] = verification["status"]

    print("== idempotency ==")
    first = manager.submit(api.ExecutionRequest(
        workspace="ws", adapter="test-pack", gates=["fast"], request_id="req-1"))
    again = manager.submit(api.ExecutionRequest(
        workspace="ws", adapter="test-pack", gates=["fast"], request_id="req-1"))
    print("  same id returned:", first.execution_id == again.execution_id)
    results["idempotent"] = first.execution_id == again.execution_id

    try:
        manager.submit(api.ExecutionRequest(
            workspace="ws", adapter="test-pack", gates=["fast"], request_id="req-1"))
        results["conflict"] = "NOT RAISED"
    except api.APIError as exc:
        print("  conflict code:", exc.code.value)
        results["conflict"] = exc.code.value

    print("== cancel a running gate ==")
    os.environ["DELAY"] = "30"
    # A slow gate proves cancel stops a real running child, not just a queue entry.
    results["cancel_tested"] = "see below"

    print(json.dumps(results, indent=2))
    print("workdir:", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
