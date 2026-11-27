"""Unit and integration tests for AgentGates V2 linked git worktree support."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import anyio
import pytest

from agent_gates import api_types as api
from agent_gates import cli
from agent_gates import config as cfg
from agent_gates import execution as execution_mod
from agent_gates import mcp_server
from agent_gates import worker as worker_mod
from agent_gates.engine import Engine
from agent_gates.evidence import EvidenceStore
from agent_gates.execution import ExecutionManager
from agent_gates.execution_store import ExecutionStore
from agent_gates.manifest import digest
from agent_gates.registry import Registry
from agent_gates.source_identity import (
    list_linked_worktrees,
    resolve_git_common_dir,
    resolve_workspace_source,
    snapshot,
)
from agent_gates.verify_requirements import verify_requirements

PYTHON = sys.executable


class FakeSystemd:
    """A systemd stand-in that records calls and can report a chosen state."""

    class SystemdUnavailable(RuntimeError):
        pass

    def __init__(self):
        self.started = []
        self.stopped = []
        self.running = set()
        self.available_result = (True, "running")

    def available(self):
        return self.available_result

    def start(self, execution_id):
        self.started.append(execution_id)
        self.running.add(execution_id)

    def stop(self, execution_id):
        self.stopped.append(execution_id)
        self.running.discard(execution_id)

    def properties(self, execution_id, *names):
        state = "active" if execution_id in self.running else "inactive"
        return {"ActiveState": state, "SubState": state, "Result": "success"}

    def kill(self, execution_id, timeout=60):
        pass

    def reset(self, execution_id):
        pass

    def journal_tail(self, execution_id, lines=50):
        return ""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "gate@example.invalid")
    _git(path, "config", "user.name", "gate")
    _git(path, "config", "core.excludesFile", "/dev/null")
    (path / "tracked.txt").write_text("initial tracked content\n")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-qm", "initial commit")
    return path


def _add_worktree(main_repo: Path, wt_path: Path) -> Path:
    _git(main_repo, "worktree", "add", "-d", str(wt_path))
    return wt_path


def _install_test_pack(adapters_root: Path) -> Path:
    pack = adapters_root / "test-pack"
    (pack / "scripts").mkdir(parents=True)
    script = pack / "scripts/gate.py"
    script.write_text(
        "import json, sys\n"
        "if '--check-config' in sys.argv:\n"
        "    sys.exit(0)\n"
        "print(json.dumps({'ok': True}))\n"
    )
    manifest = {
        "schemaVersion": 1,
        "id": "test-pack",
        "version": "1",
        "files": {"scripts/gate.py": digest(script)},
        "permissions": {"sourceWrite": False, "hostExecution": True},
        "recipes": [
            {
                "id": "test.run",
                "cwd": "source",
                "timeout": 60,
                "argv": [PYTHON, "-I", "{pack}/scripts/gate.py"],
                "configCheck": [PYTHON, "-I", "{pack}/scripts/gate.py", "--check-config"],
            }
        ],
        "gates": [
            {
                "id": "test",
                "recipeId": "test.run",
                "gateType": "deterministic-test",
                "observationChecks": {"ok": True},
            }
        ],
    }
    (pack / "manifest.json").write_text(json.dumps(manifest))
    return pack


def _config_toml(
    root: Path,
    source_root: Path,
    *,
    allow_worktrees_line: str = "",
) -> str:
    extra = f"\n{allow_worktrees_line}" if allow_worktrees_line else ""
    return (
        "schema_version = 1\n"
        f'state_root = "{root / "state"}"\n'
        f'evidence_root = "{root / "evidence"}"\n\n'
        "[adapters.test-pack]\n\n"
        "[workspaces.ws]\n"
        f'source_root = "{source_root}"\n'
        'adapters = ["test-pack"]'
        f"{extra}\n\n"
        "[worker]\n"
        'backend = "systemd"\n'
        "max_concurrent = 2\n"
        f'python = "{PYTHON}"\n'
    )


def _setup_workspace_env(tmp_path: Path, *, allow_worktrees: bool | None = None):
    main = _init_git_repo(tmp_path / "main")
    wt1 = _add_worktree(main, tmp_path / "wt1")
    wt2 = _add_worktree(main, tmp_path / "wt2")
    adapters_root = tmp_path / "adapters"
    _install_test_pack(adapters_root)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "evidence").mkdir(exist_ok=True)
    line = ""
    if allow_worktrees is True:
        line = "allow_worktrees = true"
    elif allow_worktrees is False:
        line = "allow_worktrees = false"
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config_toml(tmp_path, main, allow_worktrees_line=line))
    config = cfg.load(config_path)
    registry = Registry(adapters_root)
    store = EvidenceStore(config.evidence_root)
    engine = Engine(registry=registry, store=store, lock_root=config.lock_root)
    return {
        "main": main,
        "wt1": wt1,
        "wt2": wt2,
        "adapters_root": adapters_root,
        "config_path": config_path,
        "config": config,
        "registry": registry,
        "store": store,
        "engine": engine,
    }


def test_resolve_workspace_source_accepts_registered_and_linked_worktree(tmp_path):
    main = _init_git_repo(tmp_path / "main")
    wt = _add_worktree(main, tmp_path / "wt")
    non_git = tmp_path / "plain"
    non_git.mkdir()

    assert resolve_git_common_dir(main) == (main / ".git").resolve()
    assert resolve_git_common_dir(wt) == (main / ".git").resolve()
    assert resolve_git_common_dir(non_git) is None

    linked = list_linked_worktrees(main)
    assert isinstance(linked, frozenset)
    assert wt.resolve() in linked

    assert resolve_workspace_source(main, None) == main.resolve()
    assert resolve_workspace_source(main, main) == main.resolve()
    assert resolve_workspace_source(main, str(main)) == main.resolve()
    assert resolve_workspace_source(main, wt) == wt.resolve()
    assert resolve_workspace_source(main, str(wt)) == wt.resolve()


def test_resolve_workspace_source_rejects_forged_git_file(tmp_path):
    main = _init_git_repo(tmp_path / "main")
    wt = _add_worktree(main, tmp_path / "wt")

    forged_main = tmp_path / "forged_main"
    forged_main.mkdir()
    (forged_main / ".git").write_text(f"gitdir: {main / '.git'}\n")

    forged_wt = tmp_path / "forged_wt"
    forged_wt.mkdir()
    (forged_wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / wt.name}\n")

    linked = list_linked_worktrees(main)
    assert forged_main.resolve() not in linked
    assert forged_wt.resolve() not in linked

    with pytest.raises(ValueError):
        resolve_workspace_source(main, forged_main)
    with pytest.raises(ValueError):
        resolve_workspace_source(main, forged_wt)


def test_resolve_workspace_source_rejects_unrelated_repo_subdir_symlink_and_dotdot(tmp_path, monkeypatch):
    main = _init_git_repo(tmp_path / "main")
    wt = _add_worktree(main, tmp_path / "wt")
    unrelated = _init_git_repo(tmp_path / "unrelated")

    subdir = wt / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested\n")

    wt_symlink = tmp_path / "wt-symlink"
    wt_symlink.symlink_to(wt)

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    wt_nested = _add_worktree(main, real_parent / "wt_nested")
    symlink_parent = tmp_path / "symlink_parent"
    symlink_parent.symlink_to(real_parent)

    dotdot_path = str(wt.parent / "main" / ".." / wt.name)

    with pytest.raises(ValueError):
        resolve_workspace_source(main, unrelated)
    with pytest.raises(ValueError):
        resolve_workspace_source(main, subdir)
    with pytest.raises(ValueError):
        resolve_workspace_source(main, wt_symlink)
    with pytest.raises(ValueError):
        resolve_workspace_source(main, symlink_parent / wt_nested.name)
    with pytest.raises(ValueError):
        resolve_workspace_source(main, dotdot_path)

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        resolve_workspace_source(main, wt.name)
    with pytest.raises(ValueError):
        resolve_workspace_source(main, Path(wt.name))


def test_allow_worktrees_false_refuses_worktree_and_changes_config_digest(tmp_path):
    env = _setup_workspace_env(tmp_path)
    main = env["main"]
    wt1 = env["wt1"]

    cfg_omitted = cfg.from_mapping(
        cfg.tomllib.loads(_config_toml(tmp_path, main)),
        path=tmp_path / "c1.toml",
    )
    cfg_true = cfg.from_mapping(
        cfg.tomllib.loads(_config_toml(tmp_path, main, allow_worktrees_line="allow_worktrees = true")),
        path=tmp_path / "c2.toml",
    )
    cfg_false = cfg.from_mapping(
        cfg.tomllib.loads(_config_toml(tmp_path, main, allow_worktrees_line="allow_worktrees = false")),
        path=tmp_path / "c3.toml",
    )

    assert cfg_omitted.workspaces["ws"].allow_worktrees is True
    assert cfg_true.workspaces["ws"].allow_worktrees is True
    assert cfg_omitted.config_digest == cfg_true.config_digest
    assert cfg_false.workspaces["ws"].allow_worktrees is False
    assert cfg_false.config_digest != cfg_omitted.config_digest

    with pytest.raises(cfg.ConfigError) as cfg_err:
        cfg.from_mapping(
            cfg.tomllib.loads(_config_toml(tmp_path, main, allow_worktrees_line='allow_worktrees = "no"')),
            path=tmp_path / "bad.toml",
        )
    assert cfg_err.value.field == "workspaces.ws.allow_worktrees"

    assert resolve_workspace_source(main, main, allow_worktrees=False) == main.resolve()
    with pytest.raises(ValueError):
        resolve_workspace_source(main, wt1, allow_worktrees=False)

    systemd = FakeSystemd()
    manager = ExecutionManager(cfg_false, registry=env["registry"], systemd=systemd)
    with pytest.raises(api.APIError) as submit_err:
        manager.submit(
            api.ExecutionRequest(
                workspace="ws",
                adapter="test-pack",
                gates=["test"],
                source_root=str(wt1),
            )
        )
    assert submit_err.value.code is api.ErrorCode.INVALID_REQUEST
    assert systemd.started == []

    receipt = env["engine"].run("test-pack", wt1, ["test"])
    assert receipt["status"] == "PASS"
    vr = verify_requirements(
        api.VerificationRequirement(
            workspace="ws",
            adapter="test-pack",
            required_gates=["test"],
            run_id=receipt["runId"],
            source_root=str(wt1),
        ),
        config=cfg_false,
        registry=env["registry"],
        store=env["store"],
    )
    assert not vr.satisfied
    assert vr.status is api.GateStatus.INVALID_EVIDENCE
    assert any(f.startswith("WORKSPACE_SOURCE_NOT_ALLOWED:") for f in vr.failures), vr.failures


def test_worktree_exclusion_rules_bind_common_dir_exclude(tmp_path):
    main = _init_git_repo(tmp_path / "main")
    wt = _add_worktree(main, tmp_path / "wt")

    base = snapshot(wt)
    exclude_path = main / ".git" / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_path.write_text("hidden_secret.py\n")

    after_exclude = snapshot(wt)
    assert after_exclude["hash"] != base["hash"]

    (wt / "hidden_secret.py").write_text("secret = True\n")
    after_file = snapshot(wt)
    assert after_file["fileCount"] == base["fileCount"]
    assert after_file["hash"] == after_exclude["hash"]


def test_end_to_end_submit_worker_verify_and_cross_tree_refusal(tmp_path):
    env = _setup_workspace_env(tmp_path)
    wt1 = env["wt1"]
    wt2 = env["wt2"]
    config = env["config"]
    systemd = FakeSystemd()
    manager = ExecutionManager(config, registry=env["registry"], systemd=systemd)

    status = manager.submit(
        api.ExecutionRequest(
            workspace="ws",
            adapter="test-pack",
            gates=["test"],
            source_root=str(wt1),
        )
    )
    assert status.source_root == str(wt1.resolve())
    record = manager.store.read_request(status.execution_id)
    assert record["source_root"] == str(wt1.resolve())
    assert record["source_identity"] == snapshot(wt1)

    exec_engine = Engine(
        registry=env["registry"],
        store=manager.evidence,
        lock_root=config.lock_root,
    )
    final = worker_mod.run_execution(
        status.execution_id,
        store=manager.store,
        config=config,
        engine=exec_engine,
    )
    assert final["state"] == api.ExecutionState.FINISHED.value, final
    run_id = final["runId"]
    assert run_id

    observed = manager.status(status.execution_id)
    assert observed.state is api.ExecutionState.FINISHED
    assert observed.source_root == str(wt1.resolve())
    assert observed.run_id == run_id

    ok = verify_requirements(
        api.VerificationRequirement(
            workspace="ws",
            adapter="test-pack",
            required_gates=["test"],
            run_id=run_id,
            source_root=str(wt1),
        ),
        config=config,
        registry=env["registry"],
        store=manager.evidence,
    )
    assert ok.satisfied is True
    assert ok.status is api.GateStatus.PASS
    assert ok.source_root == str(wt1.resolve())
    assert ok.matched_gates == ["test"]

    refused_main = verify_requirements(
        api.VerificationRequirement(
            workspace="ws",
            adapter="test-pack",
            required_gates=["test"],
            run_id=run_id,
            source_root=None,
        ),
        config=config,
        registry=env["registry"],
        store=manager.evidence,
    )
    assert not refused_main.satisfied
    assert refused_main.status is api.GateStatus.INVALID_EVIDENCE
    assert any(f.startswith("WORKSPACE_SOURCE_MISMATCH:") for f in refused_main.failures), refused_main.failures

    refused_wt2 = verify_requirements(
        api.VerificationRequirement(
            workspace="ws",
            adapter="test-pack",
            required_gates=["test"],
            run_id=run_id,
            source_root=str(wt2),
        ),
        config=config,
        registry=env["registry"],
        store=manager.evidence,
    )
    assert not refused_wt2.satisfied
    assert refused_wt2.status is api.GateStatus.INVALID_EVIDENCE
    assert any(f.startswith("WORKSPACE_SOURCE_MISMATCH:") for f in refused_wt2.failures), refused_wt2.failures

    (wt1 / "untracked_change.txt").write_text("dirty untracked\n")
    stale_untracked = verify_requirements(
        api.VerificationRequirement(
            workspace="ws",
            adapter="test-pack",
            required_gates=["test"],
            run_id=run_id,
            source_root=str(wt1),
        ),
        config=config,
        registry=env["registry"],
        store=manager.evidence,
    )
    assert not stale_untracked.satisfied
    assert stale_untracked.status is api.GateStatus.STALE
    assert any(
        "SOURCE_IDENTITY_MISMATCH" in f or "SOURCE_IDENTITY_CHANGED" in f
        for f in stale_untracked.failures
    ), stale_untracked.failures

    (wt1 / "tracked.txt").write_text("dirty tracked\n")
    stale_tracked = verify_requirements(
        api.VerificationRequirement(
            workspace="ws",
            adapter="test-pack",
            required_gates=["test"],
            run_id=run_id,
            source_root=str(wt1),
        ),
        config=config,
        registry=env["registry"],
        store=manager.evidence,
    )
    assert not stale_tracked.satisfied
    assert stale_tracked.status is api.GateStatus.STALE
    assert any(
        "SOURCE_IDENTITY_MISMATCH" in f or "SOURCE_IDENTITY_CHANGED" in f
        for f in stale_tracked.failures
    ), stale_tracked.failures

    tamper_status = manager.submit(
        api.ExecutionRequest(
            workspace="ws",
            adapter="test-pack",
            gates=["test"],
            source_root=str(wt2),
        )
    )
    other_repo = _init_git_repo(tmp_path / "other_repo")
    tamper_record = manager.store.read_request(tamper_status.execution_id)
    tamper_record["source_root"] = str(other_repo)
    manager.store.request_path(tamper_status.execution_id).write_text(json.dumps(tamper_record))

    tampered = worker_mod.run_execution(
        tamper_status.execution_id,
        store=manager.store,
        config=config,
        engine=exec_engine,
    )
    assert tampered["state"] == api.ExecutionState.INTERRUPTED.value, tampered
    assert tampered["exitReason"].startswith("SOURCE_NOT_ALLOWED: "), tampered
    assert tampered.get("runId") is None


def test_request_id_conflict_across_worktrees(tmp_path):
    env = _setup_workspace_env(tmp_path)
    manager = ExecutionManager(env["config"], registry=env["registry"], systemd=FakeSystemd())

    first = manager.submit(
        api.ExecutionRequest(
            workspace="ws",
            adapter="test-pack",
            gates=["test"],
            request_id="shared-req-id",
            source_root=str(env["wt1"]),
        )
    )
    replay = manager.submit(
        api.ExecutionRequest(
            workspace="ws",
            adapter="test-pack",
            gates=["test"],
            request_id="shared-req-id",
            source_root=str(env["wt1"]),
        )
    )
    assert replay.execution_id == first.execution_id

    with pytest.raises(api.APIError) as caught:
        manager.submit(
            api.ExecutionRequest(
                workspace="ws",
                adapter="test-pack",
                gates=["test"],
                request_id="shared-req-id",
                source_root=str(env["wt2"]),
            )
        )
    assert caught.value.code is api.ErrorCode.CONFLICT


def test_cli_submit_and_verify_requirements_forward_source(tmp_path, monkeypatch, capsys):
    env = _setup_workspace_env(tmp_path)
    wt1 = env["wt1"]
    wt2 = env["wt2"]
    systemd = FakeSystemd()

    monkeypatch.setenv("AGENT_GATES_CONFIG", str(env["config_path"]))
    monkeypatch.setenv("AGENT_GATES_ADAPTERS_ROOT", str(env["adapters_root"]))
    monkeypatch.setattr(execution_mod, "systemd_units", systemd)

    rc = cli.main(
        [
            "submit",
            "--workspace",
            "ws",
            "--adapter",
            "test-pack",
            "--gate",
            "test",
            "--source",
            str(wt1),
            "--json",
        ]
    )
    assert rc == 0
    submit_payload = json.loads(capsys.readouterr().out)
    assert submit_payload["source_root"] == str(wt1.resolve())
    eid = submit_payload["execution_id"]

    exec_store = ExecutionStore(env["config"].state_root)
    assert exec_store.read_request(eid)["source_root"] == str(wt1.resolve())

    final = worker_mod.run_execution(
        eid,
        store=exec_store,
        config=env["config"],
        engine=env["engine"],
    )
    assert final["state"] == api.ExecutionState.FINISHED.value
    run_id = final["runId"]

    rc_ok = cli.main(
        [
            "verify-requirements",
            "--workspace",
            "ws",
            "--adapter",
            "test-pack",
            "--gate",
            "test",
            "--run-id",
            run_id,
            "--source",
            str(wt1),
            "--json",
        ]
    )
    assert rc_ok == 0
    vr_ok = json.loads(capsys.readouterr().out)
    assert vr_ok["satisfied"] is True
    assert vr_ok["status"] == "PASS"
    assert vr_ok["source_root"] == str(wt1.resolve())

    rc_mismatch = cli.main(
        [
            "verify-requirements",
            "--workspace",
            "ws",
            "--adapter",
            "test-pack",
            "--gate",
            "test",
            "--run-id",
            run_id,
            "--source",
            str(wt2),
            "--json",
        ]
    )
    assert rc_mismatch == 1
    vr_mismatch = json.loads(capsys.readouterr().out)
    assert vr_mismatch["satisfied"] is False
    assert any(f.startswith("WORKSPACE_SOURCE_MISMATCH:") for f in vr_mismatch["failures"])


def _run_mcp_worktree_assertions(base_dir: Path) -> None:
    """Exercise MCP worktree tools directly when mcp.server.mcpserver is available."""
    from unittest.mock import patch

    env = _setup_workspace_env(base_dir)
    wt1 = env["wt1"]
    other_repo = _init_git_repo(base_dir / "other_repo")
    systemd = FakeSystemd()

    with (
        patch.dict(
            os.environ,
            {
                "AGENT_GATES_CONFIG": str(env["config_path"]),
                "AGENT_GATES_ADAPTERS_ROOT": str(env["adapters_root"]),
            },
        ),
        patch.object(execution_mod, "systemd_units", systemd),
    ):
        server = mcp_server.build_server()
        tools = {tool.name: tool for tool in server._tool_manager.list_tools()}

        async def call_list_workspaces():
            return await tools["list_workspaces"].fn()

        ws_list = anyio.run(call_list_workspaces)
        assert ws_list.workspaces[0].id == "ws"
        assert ws_list.workspaces[0].allow_worktrees is True

        async def call_doctor_ok():
            return await tools["doctor"].fn(
                workspace="ws",
                adapter="test-pack",
                source_root=str(wt1),
            )

        doctor_ok = anyio.run(call_doctor_ok)
        assert doctor_ok.status is api.GateStatus.PASS

        async def call_doctor_unrelated():
            return await tools["doctor"].fn(
                workspace="ws",
                adapter="test-pack",
                source_root=str(other_repo),
            )

        with pytest.raises(api.APIError) as doc_err:
            anyio.run(call_doctor_unrelated)
        assert doc_err.value.code is api.ErrorCode.INVALID_REQUEST

        async def call_submit_unrelated():
            return await tools["submit"].fn(
                workspace="ws",
                adapter="test-pack",
                gates=["test"],
                source_root=str(other_repo),
            )

        with pytest.raises(api.APIError) as sub_err:
            anyio.run(call_submit_unrelated)
        assert sub_err.value.code is api.ErrorCode.INVALID_REQUEST

        async def call_submit_wt():
            return await tools["submit"].fn(
                workspace="ws",
                adapter="test-pack",
                gates=["test"],
                source_root=str(wt1),
            )

        submitted = anyio.run(call_submit_wt)
        assert submitted.source_root == str(wt1.resolve())

        final = worker_mod.run_execution(
            submitted.execution_id,
            store=ExecutionStore(env["config"].state_root),
            config=env["config"],
            engine=env["engine"],
        )
        run_id = final["runId"]

        async def call_verify_wt():
            return await tools["verify_requirements"].fn(
                workspace="ws",
                adapter="test-pack",
                required_gates=["test"],
                run_id=run_id,
                source_root=str(wt1),
            )

        vr_ok = anyio.run(call_verify_wt)
        assert vr_ok.satisfied is True
        assert vr_ok.source_root == str(wt1.resolve())

        async def call_verify_unrelated():
            return await tools["verify_requirements"].fn(
                workspace="ws",
                adapter="test-pack",
                required_gates=["test"],
                run_id=run_id,
                source_root=str(other_repo),
            )

        try:
            vr_bad = anyio.run(call_verify_unrelated)
        except api.APIError as vr_err:
            assert vr_err.code is api.ErrorCode.INVALID_REQUEST
        else:
            assert not vr_bad.satisfied
            assert vr_bad.status is api.GateStatus.INVALID_EVIDENCE
            assert any(
                f.startswith("WORKSPACE_SOURCE_NOT_ALLOWED:")
                for f in vr_bad.failures
            ), vr_bad.failures


def test_mcp_server_accepts_linked_worktree_and_refuses_unrelated(tmp_path):
    if importlib.util.find_spec("mcp.server.mcpserver") is not None:
        _run_mcp_worktree_assertions(tmp_path)
        return
    repo_root = Path(__file__).resolve().parent.parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        pytest.skip("mcp.server.mcpserver unavailable and AgentGates .venv missing")
    runner = (
        "import sys, types\n"
        "from pathlib import Path\n"
        "stub = types.ModuleType('pytest')\n"
        "class _Ctx:\n"
        "    def __init__(self, exc): self.exc, self.value = exc, None\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, et, ev, tb):\n"
        "        if et is None: raise AssertionError(self.exc)\n"
        "        if issubclass(et, self.exc): self.value = ev; return True\n"
        "        return False\n"
        "stub.raises = lambda exc: _Ctx(exc)\n"
        "sys.modules['pytest'] = stub\n"
        "from tests.test_worktree_support import _run_mcp_worktree_assertions\n"
        "_run_mcp_worktree_assertions(Path(sys.argv[1]))\n"
    )
    proc_env = dict(os.environ)
    proc_env["PYTHONPATH"] = str(repo_root)
    proc = subprocess.run(
        [str(venv_python), "-c", runner, str(tmp_path)],
        capture_output=True,
        text=True,
        env=proc_env,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
