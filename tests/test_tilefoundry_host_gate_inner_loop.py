"""Host-only tests for the tilefoundry-npu inner-loop device gate."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent_gates import engine, machine_env
from agent_gates.registry import Registry

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "agent_gates" / "adapters" / "tilefoundry-npu"
SCRIPT = ADAPTER / "scripts" / "tilefoundry_host_gate.py"


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_script(SCRIPT, "tilefoundry_host_gate_inner_loop_test")

PROBE = """
import json
print(json.dumps({
    "tilefoundryImported": True,
    "npuAvailable": True,
    "deviceCountOne": True,
    "tensorOnDevice": True,
    "fp32VectorOk": True,
    "bf16MatmulOk": True,
    "numericalOk": True,
    "synchronizeOk": True,
    "deviceMemoryPositive": True,
}))
"""


class ManifestContractCase(unittest.TestCase):
    def test_registry_loads_fast_gate_and_pins_digest(self):
        manifest = Registry(ADAPTER.parent).get("tilefoundry-npu")
        manifest.validate_files()
        self.assertEqual(manifest.id, "tilefoundry-npu")
        self.assertEqual(manifest.order(["tilefoundry-npu-device-fast"]),
                         ["tilefoundry-npu-device-fast"])
        self.assertEqual(manifest.order(["tilefoundry-npu-device"]),
                         ["tilefoundry-npu-cpu", "tilefoundry-npu-device"])

        raw = json.loads((ADAPTER / "manifest.json").read_text(encoding="utf-8"))
        recipes = {item["id"]: item for item in raw["recipes"]}
        fast_recipe = recipes["tilefoundry-npu.device-fast"]
        self.assertEqual(fast_recipe["argv"], [
            "python3", "-B", "{pack}/scripts/tilefoundry_host_gate.py",
            "--device-fast", "--timeout", "600",
        ])
        self.assertEqual(fast_recipe["deviceWindow"], {"waitSeconds": 60})
        self.assertEqual(fast_recipe["configCheck"], [
            "python3", "-B", "{pack}/scripts/tilefoundry_host_gate.py",
            "--check-config-device",
        ])
        self.assertEqual(fast_recipe["blockedExitCodes"], [75])

        gates = {item["id"]: item for item in raw["gates"]}
        fast_gate = gates["tilefoundry-npu-device-fast"]
        self.assertEqual(fast_gate["recipeId"], "tilefoundry-npu.device-fast")
        self.assertEqual(fast_gate["observationChecks"], {
            "exitCodeZero": True,
            "windowOpened": True,
            "windowReleased": True,
            "occupancyRecorded": True,
        })
        self.assertNotIn("dependsOn", fast_gate)
        self.assertEqual(gates["tilefoundry-npu-device"]["dependsOn"],
                         ["tilefoundry-npu-cpu"])

    def test_runner_rows_carry_literal_case_metrics(self):
        self.assertIn('"cases": case_rows', GATE.CANNBENCH_RUNNER)
        self.assertIn('"case_id": str(c.get("case_id", ""))', GATE.CANNBENCH_RUNNER)
        self.assertIn('"speedup": float(c.get("speedup") or 0.0)', GATE.CANNBENCH_RUNNER)
        self.assertIn('"elapsed_us": float(c.get("elapsed_us") or 0.0)',
                      GATE.CANNBENCH_RUNNER)

    def test_binary_flags_are_exact(self):
        self.assertTrue(GATE._binary_flag({"K": "1"}, "K"))
        self.assertFalse(GATE._binary_flag({"K": "0"}, "K"))
        self.assertFalse(GATE._binary_flag({}, "K"))
        with self.assertRaises(GATE.BridgeError):
            GATE._binary_flag({"K": "true"}, "K")


class MachineEnvCase(unittest.TestCase):
    def test_engine_build_directory_accepts_new_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tf-repo"
            source.mkdir()
            env_file = root / "machine.env"
            env_file.write_text(
                "TILEFOUNDRY_NPU_CANN_SETUP=/bin/true\n"
                "TILEFOUNDRY_NPU_CONDA_SH=/bin/true\n"
                "TILEFOUNDRY_NPU_CONDA_ENV=tilefoundry-npu\n"
                "TILEFOUNDRY_NPU_MSPROF_LOCK=1\n"
                "TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE=1\n",
                encoding="utf-8",
            )
            env = {"TILEFOUNDRY_NPU_ENV_FILE": str(env_file)}
            manifest = Registry(ADAPTER.parent).get("tilefoundry-npu")

            machine = machine_env.read("tilefoundry-npu", env=env)
            self.assertEqual(machine.values["TILEFOUNDRY_NPU_MSPROF_LOCK"], "1")
            self.assertEqual(machine.values["TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE"], "1")
            self.assertEqual(engine.build_directory(source, manifest, env),
                             (source / "build").resolve())

            with patch.dict(os.environ, env, clear=True):
                values = GATE.machine_values()
            self.assertEqual(values["TILEFOUNDRY_NPU_MSPROF_LOCK"], "1")
            self.assertEqual(values["TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE"], "1")


class DeviceModeCase(unittest.TestCase):
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
            "tests/device",
        ):
            (self.src / sub).mkdir(parents=True)
        (self.src / "tests" / "device" / "test_ok.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        (self.src / "tests" / "device" / "test_skip.py").write_text(
            "import pytest\n\n@pytest.mark.skip(reason='host-only skip')\n"
            "def test_skip():\n    assert False\n",
            encoding="utf-8",
        )
        self.cann = self.root / "set_env.sh"
        self.cann.write_text(":\n", encoding="utf-8")
        self.conda_sh = self.root / "conda.sh"
        self.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        self.env_file = self.root / "machine.env"
        self.lock_root = self.root / "locks"
        self.lease = self.lock_root / "npu-1.lock"
        self.shim = self.root / "shim"
        self.shim.mkdir()
        self.tl_cache = self.root / "tl-cache"
        self.write_env()

    def write_env(self, **overrides):
        values = {
            "TILEFOUNDRY_NPU_CANN_SETUP": str(self.cann),
            "TILEFOUNDRY_NPU_CONDA_SH": str(self.conda_sh),
            "TILEFOUNDRY_NPU_CONDA_ENV": "tilefoundry-npu",
            "TILEFOUNDRY_NPU_PYTHON": sys.executable,
            "TILEFOUNDRY_NPU_DEVICE_TESTS": "tests/device/test_ok.py",
        }
        values.update(overrides)
        self.env_file.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )

    def environment(self):
        return {
            "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
            "TILEFOUNDRY_NPU_DEVICE_ID": "1",
            "AGENT_GATES_DEVICE_LEASE_PATH": str(self.lease),
            "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:1",
            "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "2",
            "TF_SURROGATE_SHIM_DIR": str(self.shim),
            "TF_SURROGATE_TL_CACHE_DIR": str(self.tl_cache),
        }

    @contextmanager
    def _in_source(self):
        old = os.getcwd()
        os.chdir(self.src)
        try:
            yield
        finally:
            os.chdir(old)

    @contextmanager
    def _record_host_call(self):
        real = GATE.adapter_runtime.run_host_bash
        calls = []

        def recording(cmd, **kwargs):
            calls.append((cmd, dict(kwargs["env"])))
            return real(cmd, **kwargs)

        with patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=recording):
            yield calls

    def run_fast(self, *, extra_env=None, **env_file_overrides):
        self.write_env(**env_file_overrides)
        env = self.environment()
        env.update(extra_env or {})
        with self._in_source(), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            return GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_fast_gate_runs_configured_tests_without_probe(self):
        with self._record_host_call() as calls:
            rc, obs = self.run_fast()
        self.assertEqual(rc, 0)
        self.assertTrue(obs["exitCodeZero"])
        self.assertEqual(obs["testsCollected"], 1)
        self.assertEqual(obs["testsPassed"], 1)
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])
        self.assertTrue(obs["occupancyRecorded"])
        self.assertEqual(len(calls), 1)
        self.assertIn("pytest", calls[0][0])
        self.assertNotIn("probe.py", calls[0][0])

    def test_fast_gate_refuses_empty_or_all_skipped_tests(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="")
        with self._in_source(), \
                patch.dict(os.environ, self.environment(), clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(SystemExit) as caught:
                GATE.main(["--device-fast", "--timeout", "30"])
        self.assertEqual(caught.exception.code, 75)
        self.assertFalse((self.lock_root / ".msprof.lock").exists())

        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="tests/device/test_skip.py")
        rc, obs = self.run_fast(TILEFOUNDRY_NPU_DEVICE_TESTS="tests/device/test_skip.py")
        self.assertEqual(rc, 1)
        self.assertFalse(obs["exitCodeZero"])
        self.assertEqual(obs["testsCollected"], 1)
        self.assertEqual(obs["testsSkipped"], 1)

    def test_fast_gate_uses_cache_and_shim_env_when_requested(self):
        with self._record_host_call() as calls:
            rc, _ = self.run_fast(TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE="1")
        self.assertEqual(rc, 0)
        env = calls[0][1]
        self.assertEqual(env["TILELANG_DISABLE_CACHE"], "0")
        self.assertEqual(env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"], "1")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(self.shim))
        self.assertEqual(env["TF_SURROGATE_TL_CACHE_DIR"], str(self.tl_cache))

        with self._record_host_call() as calls:
            rc, _ = self.run_fast(TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE="0")
        self.assertEqual(rc, 0)
        env = calls[0][1]
        self.assertEqual(env["TILELANG_DISABLE_CACHE"], "1")
        self.assertNotIn(str(self.shim), env["PYTHONPATH"].split(os.pathsep))

    def test_fast_gate_forces_torch_backend_autoload(self):
        with self._record_host_call() as calls:
            rc, _ = self.run_fast(extra_env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][1]["TORCH_DEVICE_BACKEND_AUTOLOAD"], "1")

    def test_fast_failure_output_redacts_llm_credential(self):
        (self.src / "tests" / "device" / "test_fail.py").write_text(
            "import os\n\n"
            "def test_fail():\n"
            "    print('leaked:' + os.environ['MY_KEY'])\n"
            "    assert False\n",
            encoding="utf-8",
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc, obs = self.run_fast(
                TILEFOUNDRY_NPU_DEVICE_TESTS="tests/device/test_fail.py",
                TILEFOUNDRY_NPU_LLM_API_KEY_ENV="MY_KEY",
                extra_env={"MY_KEY": "gate-redaction-token"},
            )
        self.assertEqual(rc, 1)
        self.assertFalse(obs["exitCodeZero"])
        text = stderr.getvalue()
        self.assertNotIn("gate-redaction-token", text)
        self.assertIn("[redacted]", text)

    def test_full_device_route_keeps_probe(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="")
        env = self.environment()
        with patch.object(GATE, "DEVICE_PROBE", PROBE), \
                self._record_host_call() as calls, \
                self._in_source(), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            rc, obs = GATE.run_device(GATE.machine_values(), 60.0)
        self.assertEqual(rc, 0)
        self.assertTrue(obs["exitCodeZero"])
        self.assertTrue(obs["npuAvailable"])
        self.assertTrue(obs["deviceMemoryPositive"])
        self.assertIn("probe.py", calls[0][0])
        self.assertNotIn("pytest", calls[0][0])

    def test_msprof_lock_is_acquired_before_device_window(self):
        events = []
        original_lock = GATE._open_msprof_lock
        original_window = GATE._open_device_window

        @contextmanager
        def lock(*args, **kwargs):
            events.append("msprof")
            with original_lock(*args, **kwargs):
                yield

        @contextmanager
        def window(*args, **kwargs):
            events.append("window")
            with original_window(*args, **kwargs) as finalize:
                yield finalize

        with patch.object(GATE, "_open_msprof_lock", lock), \
                patch.object(GATE, "_open_device_window", window):
            rc, _ = self.run_fast(TILEFOUNDRY_NPU_MSPROF_LOCK="1")
        self.assertEqual(rc, 0)
        self.assertEqual(events[:2], ["msprof", "window"])
        self.assertTrue((self.lock_root / ".msprof.lock").is_file())

    def test_msprof_lock_is_host_shared_across_processes(self):
        with GATE._open_msprof_lock(self.lock_root):
            blocked = self._child_lock_result()
        self.assertEqual(blocked, "blocked")
        self.assertEqual(self._child_lock_result(), "acquired")

    def _child_lock_result(self):
        code = (
            "import importlib.util, sys\n"
            "from pathlib import Path\n"
            f"spec = importlib.util.spec_from_file_location('gate', {str(SCRIPT)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "sys.modules['gate'] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "try:\n"
            f"    with mod._open_msprof_lock(Path({str(self.lock_root)!r}), "
            "timeout_seconds=0.2):\n"
            "        print('acquired')\n"
            "except mod.BridgeError:\n"
            "    print('blocked')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()


if __name__ == "__main__":
    unittest.main()
