"""Verification that legacy Docker adapters, routes, and scripts are quarantined."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from agent_gates import config as cfg
from agent_gates.engine import Engine
from agent_gates.evidence import EvidenceStore


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = REPO_ROOT / "agent_gates" / "adapters"


@pytest.fixture
def machine(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "evidence").mkdir()
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "init",
        ],
        cwd=source,
        check=True,
    )
    env_file = tmp_path / "env.sh"
    env_file.write_text("# empty\n", encoding="utf-8")
    return tmp_path


def _base_config(machine):
    return {
        "schema_version": 1,
        "state_root": str(machine / "state"),
        "evidence_root": str(machine / "evidence"),
        "workspaces": {
            "ws": {
                "source_root": str(machine / "source"),
                "adapters": ["ascendnpu-ir"],
            }
        },
        "adapters": {
            "ascendnpu-ir": {"env_file": str(machine / "env.sh")},
        },
        "worker": {
            "backend": "systemd",
            "max_concurrent": 1,
            "python": sys.executable,
            "start_timeout_seconds": 30,
        },
    }


def test_quarantine_doc_exists_and_nonempty():
    doc = REPO_ROOT / "docs" / "docker-mode-quarantine.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "DOCKER_MODE_DISABLED" in text
    assert "s00653124_tla" in text
    assert "cannbot-ops" in text


@pytest.mark.parametrize("adapter_id", ["tilelang-hivm", "tilelang-hivm-npu"])
def test_config_rejects_quarantined_adapter_in_adapters_and_workspaces(machine, adapter_id):
    raw = _base_config(machine)
    raw["adapters"][adapter_id] = {"env_file": str(machine / "env.sh")}
    with pytest.raises(cfg.ConfigError) as caught:
        cfg.from_mapping(raw)
    assert "DOCKER_MODE_DISABLED" in caught.value.reason

    raw2 = _base_config(machine)
    raw2["workspaces"]["ws"]["adapters"] = [adapter_id]
    with pytest.raises(cfg.ConfigError) as caught2:
        cfg.from_mapping(raw2)
    assert "DOCKER_MODE_DISABLED" in caught2.value.reason


@pytest.mark.parametrize("gate_id", ["ascend-build", "ascend-regression"])
def test_config_rejects_quarantined_allowed_gate(machine, gate_id):
    raw = _base_config(machine)
    raw["workspaces"]["ws"]["allowed_gates"] = [gate_id]
    with pytest.raises(cfg.ConfigError) as caught:
        cfg.from_mapping(raw)
    assert "DOCKER_MODE_DISABLED" in caught.value.reason


@pytest.mark.parametrize(
    "adapter_id,gates,device_id",
    [
        ("tilelang-hivm", ["tilelang-hivm-build"], None),
        ("tilelang-hivm-npu", ["tilelang-hivm-npu-sanity"], "1"),
        ("ascendnpu-ir", ["ascend-build"], None),
        ("ascendnpu-ir", ["ascend-regression"], None),
    ],
)
def test_engine_doctor_and_run_block_quarantined_routes(machine, adapter_id, gates, device_id):
    engine = Engine(
        store=EvidenceStore(machine / "evidence"),
        lock_root=machine / "state" / "locks",
    )
    doc = engine.doctor(
        adapter_id,
        machine / "source",
        selected=gates,
        env_file=machine / "env.sh",
        device_id=device_id,
    )
    assert doc["status"] == "BLOCKED_ENV"
    assert any("DOCKER_MODE_DISABLED" in item for item in doc["failures"])

    res = engine.run(
        adapter_id,
        machine / "source",
        gates,
        env_file=machine / "env.sh",
        device_id=device_id,
    )
    assert res["status"] == "BLOCKED_ENV"
    assert any("DOCKER_MODE_DISABLED" in item for item in res["gates"][0]["failures"])


@pytest.mark.parametrize(
    "rel_script,args",
    [
        ("ascendnpu-ir/scripts/docker_gate.py", ["--check-config"]),
        ("tilelang-hivm/scripts/docker_gate.py", ["--check-config"]),
        ("tilelang-hivm-npu/scripts/npu_gate.py", ["--check-config"]),
        ("tilefoundry-npu/scripts/tilefoundry_gate.py", ["--check-config"]),
        ("tilefoundry-npu/scripts/provision.py", ["--help"]),
    ],
)
def test_quarantined_python_scripts_fail_closed_when_executed(rel_script, args):
    proc = subprocess.run(
        [sys.executable, str(ADAPTERS / rel_script), *args],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 75
    assert "DOCKER_MODE_DISABLED" in proc.stderr


@pytest.mark.parametrize(
    "rel_script",
    [
        "ascendnpu-ir/docker/create-container.sh",
        "tilelang-hivm/docker/create-container.sh",
        "tla-npu/docker/create-container.sh",
    ],
)
def test_create_container_scripts_fail_closed_outside_dry_run(tmp_path, rel_script):
    home = tmp_path / "home" / "user"
    home.mkdir(parents=True)
    config = tmp_path / "config.env"
    config.write_text(
        "\n".join(
            [
                "IMAGE=ubuntu:24.04",
                "CONTAINER_NAME=probe",
                "CONTAINER_USER=probe",
                f"CONTAINER_UID={home.stat().st_uid}",
                f"CONTAINER_GID={home.stat().st_gid}",
                "CONTAINER_HOME=/home/user",
                f'BIND_MOUNTS="{tmp_path}/home:/home"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(ADAPTERS / rel_script), "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0
    assert "DOCKER_MODE_DISABLED" in proc.stderr
