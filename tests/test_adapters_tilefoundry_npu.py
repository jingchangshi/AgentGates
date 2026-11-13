"""Tests for the host-local tilefoundry-npu adapter."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from agent_gates import engine, machine_env
from agent_gates.manifest import Manifest

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "agent_gates" / "adapters" / "tilefoundry-npu"
SCRIPT = ADAPTER / "scripts" / "tilefoundry_host_gate.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("tilefoundry_host_gate_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_gate()


class ManifestAndParityCase(unittest.TestCase):
    def test_manifest_pins_digest_and_host_recipes(self):
        raw = json.loads((ADAPTER / "manifest.json").read_text(encoding="utf-8"))
        manifest = Manifest(ADAPTER / "manifest.json")
        self.assertEqual(manifest.id, "tilefoundry-npu")
        self.assertIs(raw["permissions"]["hostExecution"], True)
        self.assertIs(raw["permissions"]["sourceWrite"], False)
        for rel, digest in raw["files"].items():
            self.assertEqual(hashlib.sha256((ADAPTER / rel).read_bytes()).hexdigest(), digest)

        cpu_recipe = manifest.resolve_recipe("tilefoundry-npu.cpu", manifest.root)
        dev_recipe = manifest.resolve_recipe("tilefoundry-npu.device", manifest.root)
        self.assertFalse(cpu_recipe.uses_device)
        self.assertIsNone(cpu_recipe.device_window)
        self.assertTrue(dev_recipe.uses_device)
        self.assertEqual(dev_recipe.device_window, 60.0)

    def test_key_and_engine_wiring_parity(self):
        self.assertEqual(tuple(sorted(GATE.TOOLCHAIN_KEYS)), tuple(sorted(machine_env.TILEFOUNDRY_NPU_KEYS)))
        self.assertEqual(machine_env.ADAPTERS["tilefoundry-npu"]["prefix"], "TILEFOUNDRY_NPU_")
        self.assertEqual(engine.ENV_KEYS["tilefoundry-npu"], "TILEFOUNDRY_NPU_ENV_FILE")
        self.assertEqual(engine.DEVICE_ADAPTERS["tilefoundry-npu"], "TILEFOUNDRY_NPU_DEVICE_ID")
        self.assertEqual(engine.EXCLUSIVE_VARIABLE["tilefoundry-npu"], "TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE")


class CheckConfigAndRunCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / "tf-repo"
        for sub in (
            "src/tilefoundry",
            "tests/tilefoundry",
            "tests/ascend/target",
            "tests/ascend/ir",
            "tests/ascend/passes",
            "tests/ascend/operators",
        ):
            (self.src / sub).mkdir(parents=True)
        self.cann = self.root / "set_env.sh"
        self.cann.write_text("#!/bin/sh\n", encoding="utf-8")
        self.conda_sh = self.root / "conda.sh"
        self.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        self.py = self.root / "python3"
        self.py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.py.chmod(0o755)
        self.env_file = self.root / "machine.env"
        self.write_env()

    def write_env(self, **overrides):
        data = {
            "TILEFOUNDRY_NPU_CANN_SETUP": str(self.cann),
            "TILEFOUNDRY_NPU_CONDA_SH": str(self.conda_sh),
            "TILEFOUNDRY_NPU_CONDA_ENV": "tilefoundry-npu",
            "TILEFOUNDRY_NPU_PYTHON": str(self.py),
            "TILEFOUNDRY_NPU_BUILD_SUBDIR": "build",
        }
        data.update(overrides)
        self.env_file.write_text("".join(f"{k}={v}\n" for k, v in data.items()), encoding="utf-8")

    def test_check_config_passes_without_spawning_and_refuses_export_conflict(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            with patch.dict(os.environ, {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file)}, clear=True):
                self.assertEqual(GATE.main(["--check-config"]), 0)
                with patch.dict(os.environ, {"TILEFOUNDRY_NPU_CONDA_ENV": "other"}, clear=False):
                    with self.assertRaises(SystemExit) as caught:
                        GATE.main(["--check-config"])
                    self.assertEqual(caught.exception.code, 75)
        finally:
            os.chdir(old_cwd)

    def test_check_config_device_checks_exclusivity(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            env = {
                "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
                "TILEFOUNDRY_NPU_DEVICE_ID": "1",
                "TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE": "1",
            }
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(GATE.adapter_runtime, "check_device_ready", side_effect=GATE.adapter_runtime.BlockedEnv("busy")):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config-device"])
                self.assertEqual(caught.exception.code, 75)
        finally:
            os.chdir(old_cwd)

    def test_run_device_requires_lease_and_publishes_window_metadata(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            lease = self.root / "locks" / "npu.lock"
            probe_json = json.dumps({
                "tilefoundryImported": True,
                "npuAvailable": True,
                "deviceCountOne": True,
                "tensorOnDevice": True,
                "fp32VectorOk": True,
                "bf16MatmulOk": True,
                "numericalOk": True,
                "synchronizeOk": True,
                "deviceMemoryPositive": True,
            })
            fake_completed = types.SimpleNamespace(returncode=0, stdout=probe_json + "\n", stderr="")
            env = {
                "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
                "TILEFOUNDRY_NPU_DEVICE_ID": "1",
                "AGENT_GATES_DEVICE_LEASE_PATH": str(lease),
                "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:1",
                "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "2",
            }
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", return_value=fake_completed):
                rc, obs = GATE.run_device(GATE.machine_values(), 30.0)
            self.assertEqual(rc, 0)
            self.assertTrue(obs["exitCodeZero"])
            self.assertTrue(obs["windowOpened"])
            self.assertTrue(obs["windowReleased"])
            self.assertEqual(obs["deviceWindow"]["resource"], "npu:1")
            self.assertEqual(obs["deviceWindow"]["occupancy"], "free")
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
