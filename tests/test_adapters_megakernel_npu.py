"""Offline unit tests for the host-local megakernel-npu Adapter."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_gates import engine, machine_env
from agent_gates.manifest import Manifest
from agent_gates.registry import Registry

ADAPTER = ROOT / "agent_gates" / "adapters" / "megakernel-npu"
SCRIPT = ADAPTER / "scripts" / "megakernel_npu_gate.py"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_script(SCRIPT, "megakernel_npu_gate_test")


class ManifestAndWiringCase(unittest.TestCase):
    def test_manifest_pins_digest_and_recipes(self):
        raw = json.loads((ADAPTER / "manifest.json").read_text(encoding="utf-8"))
        manifest = Manifest(ADAPTER / "manifest.json")
        self.assertEqual(manifest.id, "megakernel-npu")
        self.assertEqual(Registry().get("megakernel-npu").id, "megakernel-npu")
        self.assertIs(raw["permissions"]["hostExecution"], True)
        self.assertIs(raw["permissions"]["sourceWrite"], False)
        for rel, digest in raw["files"].items():
            self.assertEqual(hashlib.sha256((ADAPTER / rel).read_bytes()).hexdigest(), digest)

        build_recipe = manifest.resolve_recipe("megakernel-npu.build", manifest.root)
        cpu_recipe = manifest.resolve_recipe("megakernel-npu.cpu", manifest.root)
        dev_recipe = manifest.resolve_recipe("megakernel-npu.device", manifest.root)
        self.assertFalse(build_recipe.uses_device)
        self.assertIsNone(build_recipe.device_window)
        self.assertFalse(cpu_recipe.uses_device)
        self.assertIsNone(cpu_recipe.device_window)
        self.assertTrue(dev_recipe.uses_device)
        self.assertEqual(dev_recipe.device_window, 60.0)
        self.assertIn(75, build_recipe.blocked_exit_codes)
        self.assertIn(75, cpu_recipe.blocked_exit_codes)
        self.assertIn(75, dev_recipe.blocked_exit_codes)
        self.assertEqual(
            manifest.order(["megakernel-npu-device"]),
            ["megakernel-npu-cpu", "megakernel-npu-device"],
        )

    def test_engine_and_machine_env_parity(self):
        self.assertEqual(
            tuple(sorted(GATE.TOOLCHAIN_KEYS)),
            tuple(sorted(machine_env.MEGAKERNEL_NPU_KEYS)),
        )
        self.assertEqual(machine_env.ADAPTERS["megakernel-npu"]["prefix"], "MEGAKERNEL_NPU_")
        self.assertEqual(machine_env.ADAPTERS["megakernel-npu"]["selector"], "MEGAKERNEL_NPU_ENV_FILE")
        self.assertEqual(engine.ENV_KEYS["megakernel-npu"], "MEGAKERNEL_NPU_ENV_FILE")
        self.assertEqual(engine.OWN_PREFIX["megakernel-npu"], "MEGAKERNEL_NPU_")
        self.assertEqual(engine.DEVICE_ADAPTERS["megakernel-npu"], "MEGAKERNEL_NPU_DEVICE_ID")
        self.assertEqual(
            engine.EXCLUSIVE_VARIABLE["megakernel-npu"],
            "MEGAKERNEL_NPU_REQUIRE_EXCLUSIVE_DEVICE",
        )


class ConfigAndValidationCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / "td-repo"
        for sub in (
            "python/triton_dist/mega_kernel_ascend/test/host",
            "python/triton_dist/mega_kernel_ascend/test/ops",
            "3rdparty/triton-ascend/third_party/ascend/AscendNPU-IR",
            "3rdparty/shmem/include",
        ):
            (self.src / sub).mkdir(parents=True)
        (self.src / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.18)\n", encoding="utf-8")
        (self.src / "python/setup.py").write_text("# setup\n", encoding="utf-8")
        (self.src / "3rdparty/triton-ascend/CMakeLists.txt").write_text("# ta\n", encoding="utf-8")
        (self.src / "python/triton_dist/mega_kernel_ascend/test/ops/test_megakernel_dtype_contracts.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        self.cann = self.root / "set_env.sh"
        self.cann.write_text("#!/bin/sh\n", encoding="utf-8")
        self.conda_sh = self.root / "conda.sh"
        self.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        for tool in ("python3", "clang", "clang++", "ld.lld", "ninja"):
            p = self.bin_dir / tool
            p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            p.chmod(0o755)
        self.llvm = self.root / "llvm-install"
        for sub in ("include", "lib", "bin"):
            (self.llvm / sub).mkdir(parents=True)
        (self.llvm / "bin/FileCheck").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.work_root = self.root / "work-root"
        self.runtime_home = self.root / "mk-output"
        self.runtime_home.mkdir()
        (self.runtime_home / "libprint_callback.so").write_text("stub", encoding="utf-8")
        self.env_file = self.root / "machine.env"
        self.write_env()

    def write_env(self, **overrides):
        data = {
            "MEGAKERNEL_NPU_CANN_SETUP": str(self.cann),
            "MEGAKERNEL_NPU_CONDA_SH": str(self.conda_sh),
            "MEGAKERNEL_NPU_CONDA_ENV": "megakernel-npu",
            "MEGAKERNEL_NPU_PYTHON": str(self.bin_dir / "python3"),
            "MEGAKERNEL_NPU_LLVM_SYSPATH": str(self.llvm),
            "MEGAKERNEL_NPU_WORK_ROOT": str(self.work_root),
            "MEGAKERNEL_NPU_RUNTIME_HOME": str(self.runtime_home),
            "MEGAKERNEL_NPU_BUILD_SUBDIR": "build",
            "MEGAKERNEL_NPU_NINJA_JOBS": "16",
            "MEGAKERNEL_NPU_LINK_JOBS": "2",
        }
        data.update(overrides)
        self.env_file.write_text("".join(f"{k}={v}\n" for k, v in data.items()), encoding="utf-8")

    def test_preflights_pass_and_enforce_16_core_cap_and_isolation(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            base_env = {"MEGAKERNEL_NPU_ENV_FILE": str(self.env_file), "PATH": str(self.bin_dir)}
            with patch.dict(os.environ, base_env, clear=True):
                self.assertEqual(GATE.main(["--check-config-build"]), 0)
                self.assertEqual(GATE.main(["--check-config-cpu"]), 0)
                self.assertEqual(GATE.main(["--check-config-device"]), 0)

            # Reject > 16 cores.
            self.write_env(MEGAKERNEL_NPU_NINJA_JOBS="17")
            with patch.dict(os.environ, base_env, clear=True):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config-build"])
                self.assertEqual(caught.exception.code, 75)

            # Reject empty NINJA_JOBS.
            self.write_env(MEGAKERNEL_NPU_NINJA_JOBS="")
            with patch.dict(os.environ, base_env, clear=True):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config-build"])
                self.assertEqual(caught.exception.code, 75)

            # Reject WORK_ROOT inside --source.
            self.write_env(MEGAKERNEL_NPU_WORK_ROOT=str(self.src / "stage-inside"))
            with patch.dict(os.environ, base_env, clear=True):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config-build"])
                self.assertEqual(caught.exception.code, 75)

            # Reject path traversal in test lists.
            self.write_env(MEGAKERNEL_NPU_CPU_TESTS="../outside.py")
            with patch.dict(os.environ, base_env, clear=True):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config-cpu"])
                self.assertEqual(caught.exception.code, 75)

            # Reject launch keys inside env.sh.
            self.env_file.write_text("MEGAKERNEL_NPU_DEVICE_ID=0\n", encoding="utf-8")
            with patch.dict(os.environ, base_env, clear=True):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config-cpu"])
                self.assertEqual(caught.exception.code, 75)

            # Reject contradicting shell export.
            self.write_env()
            with patch.dict(
                os.environ, {**base_env, "MEGAKERNEL_NPU_CONDA_ENV": "other"}, clear=True
            ):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config-cpu"])
                self.assertEqual(caught.exception.code, 75)
        finally:
            os.chdir(old_cwd)


class StagingAndWindowCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_incremental_sync_preserves_source_and_existing_build_cache(self):
        src = self.root / "source"
        dst = self.root / "stage"
        (src / "python/triton_dist").mkdir(parents=True)
        (src / "python/triton_dist/mod.py").write_text("x = 1\n", encoding="utf-8")
        (src / "old.txt").write_text("old\n", encoding="utf-8")
        GATE._sync_tree_incremental(
            src, dst, ["python/triton_dist/mod.py", "old.txt"],
            preserve_prefixes=("python/build",),
        )
        (dst / "python/build").mkdir(parents=True)
        cached = dst / "python/build/cache.o"
        cached.write_text("obj", encoding="utf-8")

        (src / "old.txt").unlink()
        (src / "python/triton_dist/mod.py").write_text("x = 2\n", encoding="utf-8")
        GATE._sync_tree_incremental(
            src, dst, ["python/triton_dist/mod.py"],
            preserve_prefixes=("python/build",),
        )
        self.assertEqual((dst / "python/triton_dist/mod.py").read_text(encoding="utf-8"), "x = 2\n")
        self.assertFalse((dst / "old.txt").exists())
        self.assertTrue(cached.is_file())

    def test_device_window_and_occupancy_modes(self):
        lock = self.root / "locks/npu-0.lock"
        with patch.dict(
            os.environ,
            {
                "AGENT_GATES_DEVICE_LEASE_PATH": str(lock),
                "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:0",
                "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "1.0",
            },
            clear=True,
        ), patch.object(GATE, "occupancy", return_value="free"):
            with GATE.device_window() as win:
                self.assertEqual(win["resource"], "npu:0")
                self.assertEqual(win["occupancy"], "free")
                self.assertIsNotNone(win["startedAt"])
            self.assertIsNotNone(win["releasedAt"])

        with patch.dict(
            os.environ,
            {
                "AGENT_GATES_DEVICE_LEASE_PATH": str(lock),
                "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:0",
                "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "1.0",
                "MEGAKERNEL_NPU_REQUIRE_EXCLUSIVE_DEVICE": "1",
            },
            clear=True,
        ), patch.object(GATE, "occupancy", return_value="shared(1)"):
            with self.assertRaises(GATE.DeviceWindowBusy):
                with GATE.device_window():
                    pass

    def test_occupancy_parser_handles_proc_mem_outputs(self):
        def fake_smi(args, timeout=15.0):
            if args == ["info", "-m"]:
                return "0 0 0 0 Ascend950PR\n1 1 0 1 Ascend950PR\n"
            if args == ["info", "-t", "proc-mem", "-i", "0"]:
                return "NPU ID : 0\nNo process in device.\n"
            if args == ["info", "-t", "proc-mem", "-i", "1"]:
                return "NPU ID : 1\nProcess id : 1234 Process name : python3 Process memory(MB) : 512\n"
            return None

        with patch.object(GATE, "_run_smi", side_effect=fake_smi):
            self.assertEqual(GATE.occupancy("npu:0"), "free")
            self.assertEqual(GATE.occupancy("npu:1"), "shared(1)")
            self.assertEqual(GATE.occupancy("npu:9"), "unknown")

    def test_cpu_observations_contract(self):
        obs = GATE._cpu_observations(True, 5, 5, 0, 0, ["t.py"])
        self.assertTrue(obs["exitCodeZero"])
        self.assertTrue(obs["testsCollectedPositive"])
        self.assertTrue(obs["notAllSkipped"])
        zero = GATE._cpu_observations(False, 0, 0, 0, 0, ["t.py"])
        self.assertFalse(zero["testsCollectedPositive"])
        all_skipped = GATE._cpu_observations(True, 3, 0, 0, 3, ["t.py"])
        self.assertFalse(all_skipped["notAllSkipped"])


if __name__ == "__main__":
    unittest.main()
