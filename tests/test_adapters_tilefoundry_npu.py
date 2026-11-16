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
        prep_recipe = manifest.resolve_recipe("tilefoundry-npu.cannbench-prep", manifest.root)
        bench_recipe = manifest.resolve_recipe("tilefoundry-npu.cannbench", manifest.root)
        self.assertFalse(cpu_recipe.uses_device)
        self.assertIsNone(cpu_recipe.device_window)
        self.assertTrue(dev_recipe.uses_device)
        self.assertEqual(dev_recipe.device_window, 60.0)
        self.assertTrue(prep_recipe.uses_device)
        self.assertEqual(prep_recipe.device_window, 60.0)
        self.assertTrue(bench_recipe.uses_device)
        self.assertEqual(bench_recipe.device_window, 60.0)

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


class CannbenchIntegrationCase(unittest.TestCase):
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
            "examples/ascend950pr_ops/attention_residual",
        ):
            (self.src / sub).mkdir(parents=True)
        (self.src / "examples/ascend950pr_ops/attention_residual/kernels.py").write_text("# candidate\n", encoding="utf-8")

        self.repo = self.root / "cannbot-ops"
        for sub in (
            "tools",
            "vendor/cannbench/src/kernel_eval",
            "vendor/cannbench/src/cann_bench_utils",
            "operators/metadata",
            "operators/level4/attention_residual",
        ):
            (self.repo / sub).mkdir(parents=True)
        (self.repo / "tools/run_cannbench.py").write_text("# runner\n", encoding="utf-8")
        (self.repo / "vendor/cannbench/src/cann_bench_utils/CMakeLists.txt").write_text("# cmake\n", encoding="utf-8")
        (self.repo / "vendor/cannbench/src/cann_bench_utils/ext.cpp").write_text("// ext\n", encoding="utf-8")
        for fname in ("proto.yaml", "cases.yaml", "golden.py"):
            (self.repo / "operators/level4/attention_residual" / fname).write_text(f"# {fname}\n", encoding="utf-8")

        self.work_root = self.root / "tf-work"
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
            "TILEFOUNDRY_NPU_CANNBENCH_REPO": str(self.repo),
            "TILEFOUNDRY_NPU_WORK_ROOT": str(self.work_root),
            "TILEFOUNDRY_NPU_CANNBENCH_ARCH": "ascend950",
            "TILEFOUNDRY_NPU_CANNBENCH_TARGETS": "level4/attention_residual:examples/ascend950pr_ops/attention_residual/kernels.py",
        }
        data.update(overrides)
        self.env_file.write_text("".join(f"{k}={v}\n" for k, v in data.items()), encoding="utf-8")

    def test_targets_resolution_prefers_repo_file_and_falls_back_to_env(self):
        values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
        origin, targets = GATE.resolve_cannbench_targets(values, self.src, self.repo)
        self.assertEqual(origin, "env-file")
        self.assertEqual(
            targets,
            (("level4/attention_residual", "examples/ascend950pr_ops/attention_residual/kernels.py"),),
        )

        # In-repo cannbench_targets.json overrides env.sh fallback cleanly.
        (self.repo / "operators/level4/mla").mkdir(parents=True)
        for fname in ("proto.yaml", "cases.yaml", "golden.py"):
            (self.repo / "operators/level4/mla" / fname).write_text(f"# {fname}\n", encoding="utf-8")
        (self.src / "examples/ascend950pr_ops/mla").mkdir(parents=True)
        (self.src / "examples/ascend950pr_ops/mla/kernels.py").write_text("# mla\n", encoding="utf-8")
        (self.src / "examples/ascend950pr_ops/cannbench_targets.json").write_text(
            json.dumps({"targets": [{"task": "level4/mla", "candidate": "examples/ascend950pr_ops/mla/kernels.py"}]}),
            encoding="utf-8",
        )
        origin2, targets2 = GATE.resolve_cannbench_targets(values, self.src, self.repo)
        self.assertEqual(origin2, "repo-file")
        self.assertEqual(targets2, (("level4/mla", "examples/ascend950pr_ops/mla/kernels.py"),))

    def test_work_root_cannot_overlap_source_or_cannbench_repo(self):
        for bad_root in (str(self.src / "work"), str(self.repo / ".cannbench-build")):
            self.write_env(TILEFOUNDRY_NPU_WORK_ROOT=bad_root)
            values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
            with self.assertRaises(GATE.BridgeError):
                GATE.work_root(values, self.src, self.repo)

    def test_check_config_cannbench_validates_and_rejects_bad_candidate_path(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            env = {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file), "TILEFOUNDRY_NPU_DEVICE_ID": "1"}
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(GATE.adapter_runtime, "check_device_ready", return_value=[]):
                self.assertEqual(GATE.main(["--check-config-cannbench"]), 0)

            self.write_env(TILEFOUNDRY_NPU_CANNBENCH_TARGETS="level4/attention_residual:../escape.py")
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(GATE.adapter_runtime, "check_device_ready", return_value=[]):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config-cannbench"])
                self.assertEqual(caught.exception.code, 75)
        finally:
            os.chdir(old_cwd)

    def test_cannbench_prep_and_run_publish_window_and_lean_metrics(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
            info = {
                "missing": [],
                "pythonVersion": "3.12.16",
                "torchVersion": "2.13.0+cpu",
                "torchNpuVersion": "2.13.0",
                "torchCmakeDir": "/tmp/Torch",
                "torchNpuDir": "/tmp/torch_npu",
            }
            _, slot_dir, utils_dir = GATE._abi_slot(values, self.src, self.repo, self.work_root, info)
            (utils_dir / "cann_bench_utils").mkdir(parents=True)
            (utils_dir / "cann_bench_utils" / "_C.so").write_bytes(b"ELF")
            (slot_dir / "build_meta.json").write_text("{}", encoding="utf-8")

            lease = self.root / "locks" / "npu.lock"
            env = {
                "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
                "TILEFOUNDRY_NPU_DEVICE_ID": "1",
                "AGENT_GATES_DEVICE_LEASE_PATH": str(lease),
                "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:1",
                "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "2",
            }
            prep_res = types.SimpleNamespace(returncode=0, stdout='{"cannBenchUtilsCallable": true}\n', stderr="")
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(GATE, "_inspect_conda_cannbench_env", return_value=info), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", return_value=prep_res):
                rc_p, obs_p = GATE.run_cannbench_prep(values, 60.0)
            self.assertEqual(rc_p, 0)
            self.assertTrue(obs_p["utilsReused"])
            self.assertTrue(obs_p["cannBenchUtilsCallable"])
            self.assertEqual(obs_p["deviceWindow"]["occupancy"], "free")

            bench_payload = {
                "officialReportValid": True,
                "allCasesPassed": True,
                "allTimingsValid": True,
                "targetCount": 1,
                "caseCount": 4,
                "passedCases": 4,
                "timedCases": 4,
                "score": 81.25,
                "avgSpeedup": 0.92,
                "results": [{
                    "task": "level4/attention_residual",
                    "candidate": "examples/ascend950pr_ops/attention_residual/kernels.py",
                    "reportValid": True,
                    "reportDir": str(self.work_root / "reports/r1"),
                    "caseCount": 4,
                    "passedCases": 4,
                    "timedCases": 4,
                    "score": 81.25,
                    "avgSpeedup": 0.92,
                }],
            }
            bench_res = types.SimpleNamespace(returncode=0, stdout=json.dumps(bench_payload) + "\n", stderr="")
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(GATE, "_inspect_conda_cannbench_env", return_value=info), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", return_value=bench_res):
                rc_b, obs_b = GATE.run_cannbench(values, 60.0)
            self.assertEqual(rc_b, 0)
            self.assertTrue(obs_b["officialReportValid"])
            self.assertEqual(obs_b["score"], 81.25)
            self.assertEqual(obs_b["targetsSource"], "env-file")
            self.assertEqual(obs_b["deviceWindow"]["resource"], "npu:1")
        finally:
            os.chdir(old_cwd)


class TilelangReuseCase(unittest.TestCase):
    """Downstream TileLang reuse honors tilelang-npu build stamps and fails closed when stale."""

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
            "examples/ascend950pr_ops/attention_residual",
        ):
            (self.src / sub).mkdir(parents=True)
        (self.src / "examples/ascend950pr_ops/attention_residual/kernels.py").write_text("# candidate\n", encoding="utf-8")

        self.repo = self.root / "cannbot-ops"
        for sub in (
            "tools",
            "vendor/cannbench/src/kernel_eval",
            "vendor/cannbench/src/cann_bench_utils",
            "operators/metadata",
            "operators/level4/attention_residual",
        ):
            (self.repo / sub).mkdir(parents=True)
        (self.repo / "tools/run_cannbench.py").write_text("# runner\n", encoding="utf-8")
        (self.repo / "vendor/cannbench/src/cann_bench_utils/CMakeLists.txt").write_text("# cmake\n", encoding="utf-8")
        for fname in ("proto.yaml", "cases.yaml", "golden.py"):
            (self.repo / "operators/level4/attention_residual" / fname).write_text(f"# {fname}\n", encoding="utf-8")

        self.work_root = self.root / "tf-work"
        self.tl_src = self.root / "tilelang"
        for sub in ("tilelang/ascend/language", "3rdparty/tvm/python/tvm", "src/tl", "cmake", "build/lib", "build/tvm"):
            (self.tl_src / sub).mkdir(parents=True)
        (self.tl_src / "CMakeLists.txt").write_text("# tl\n", encoding="utf-8")
        (self.tl_src / "src/tl/kernel.cc").write_text("// native v1\n", encoding="utf-8")
        (self.tl_src / "tilelang/__init__.py").write_text("# tl init\n", encoding="utf-8")
        (self.tl_src / "tilelang/ascend/language/vmi.py").write_text("# vmi\n", encoding="utf-8")
        (self.tl_src / "tilelang/wrap.pyx").write_text("# cython\n", encoding="utf-8")
        (self.tl_src / "3rdparty/tvm/python/tvm/__init__.py").write_text("# tvm\n", encoding="utf-8")
        (self.tl_src / "build/lib/libtilelang.so").write_bytes(b"0" * ((1 << 20) + 16))
        for ext in GATE.TILELANG_SABI_EXTENSIONS:
            (self.tl_src / f"build/lib/{ext}.abi3.so").write_bytes(b"ELF")
        self.refresh_stamp()

        self.cann = self.root / "set_env.sh"
        self.cann.write_text("#!/bin/sh\n", encoding="utf-8")
        self.conda_sh = self.root / "conda.sh"
        self.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        self.py = self.root / "python3"
        self.py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.py.chmod(0o755)
        self.env_file = self.root / "machine.env"
        self.write_env()

    def refresh_stamp(self):
        stamp = {
            "adapter": "tilelang-npu",
            "sabi": "cp39",
            "gitHead": "deadbeef" * 5,
            "nativeInputsSha256": GATE._tilelang_native_inputs_sha256(self.tl_src),
            "libtilelangSha256": GATE._sha256_file(self.tl_src / "build/lib/libtilelang.so"),
        }
        (self.tl_src / "build" / GATE.TILELANG_BUILD_STAMP_FILE).write_text(
            json.dumps(stamp), encoding="utf-8"
        )

    def write_env(self, **overrides):
        data = {
            "TILEFOUNDRY_NPU_CANN_SETUP": str(self.cann),
            "TILEFOUNDRY_NPU_CONDA_SH": str(self.conda_sh),
            "TILEFOUNDRY_NPU_CONDA_ENV": "tilefoundry-npu",
            "TILEFOUNDRY_NPU_PYTHON": str(self.py),
            "TILEFOUNDRY_NPU_TILELANG_SOURCE": str(self.tl_src),
            "TILEFOUNDRY_NPU_TILELANG_BUILD": str(self.tl_src / "build"),
            "TILEFOUNDRY_NPU_BUILD_SUBDIR": "build",
            "TILEFOUNDRY_NPU_CANNBENCH_REPO": str(self.repo),
            "TILEFOUNDRY_NPU_WORK_ROOT": str(self.work_root),
            "TILEFOUNDRY_NPU_CANNBENCH_TARGETS": "level4/attention_residual:examples/ascend950pr_ops/attention_residual/kernels.py",
        }
        data.update(overrides)
        self.env_file.write_text("".join(f"{k}={v}\n" for k, v in data.items()), encoding="utf-8")

    def test_native_digest_matches_tilelang_npu_gate(self):
        spec = importlib.util.spec_from_file_location(
            "tl_gate_for_parity",
            ROOT / "agent_gates" / "adapters" / "tilelang-npu" / "scripts" / "tilelang_npu_gate.py",
        )
        tl_gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tl_gate)
        self.assertEqual(
            GATE._tilelang_native_inputs_sha256(self.tl_src),
            tl_gate._tilelang_native_inputs_sha256(self.tl_src),
        )

    def test_check_config_and_runners_inject_tilelang_and_respect_py_edits(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            with patch.dict(os.environ, {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file)}, clear=True):
                self.assertEqual(GATE.main(["--check-config"]), 0)
                # Editing a pure-Python file inside tilelang/ leaves the native build stamp valid.
                (self.tl_src / "tilelang/ascend/language/vmi.py").write_text("# vmi edited\n", encoding="utf-8")
                self.assertEqual(GATE.main(["--check-config"]), 0)

                captured_cpu_env = {}

                def fake_cpu(cmd, *, cwd, env, **kwargs):
                    captured_cpu_env.update(env)
                    junit = Path(cmd.split("--junit-xml=")[1].split()[0])
                    junit.write_text('<testsuite tests="2" errors="0" failures="0" skipped="0"/>', encoding="utf-8")
                    return types.SimpleNamespace(returncode=0, stdout="", stderr="")

                with patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=fake_cpu):
                    rc_c, obs_c = GATE.run_cpu(GATE.machine_values(), 30.0)
                self.assertEqual(rc_c, 0)
                self.assertTrue(captured_cpu_env["PYTHONPATH"].startswith(str(self.tl_src.resolve()) + ":"))
                self.assertEqual(captured_cpu_env["USE_CUDA"], "0")
                self.assertEqual(obs_c["tilelang"]["source"], str(self.tl_src.resolve()))

            lease = self.root / "locks" / "npu.lock"
            dev_env = {
                "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
                "TILEFOUNDRY_NPU_DEVICE_ID": "1",
                "AGENT_GATES_DEVICE_LEASE_PATH": str(lease),
                "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:1",
                "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "2",
            }
            captured_dev_env = {}
            dev_probe = json.dumps({
                "tilefoundryImported": True,
                "tilelangReused": True,
                "tilelangVersion": "0.1.15+cpu.gitdeadbeef",
                "npuAvailable": True,
                "deviceCountOne": True,
                "tensorOnDevice": True,
                "fp32VectorOk": True,
                "bf16MatmulOk": True,
                "numericalOk": True,
                "synchronizeOk": True,
                "deviceMemoryPositive": True,
            })

            def fake_dev(cmd, *, cwd, env, **kwargs):
                captured_dev_env.update(env)
                return types.SimpleNamespace(returncode=0, stdout=dev_probe + "\n", stderr="")

            with patch.dict(os.environ, dev_env, clear=True), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=fake_dev):
                rc_d, obs_d = GATE.run_device(GATE.machine_values(), 30.0)
            self.assertEqual(rc_d, 0)
            self.assertTrue(obs_d["tilelangReused"])
            self.assertEqual(obs_d["tilelangVersion"], "0.1.15+cpu.gitdeadbeef")
            self.assertEqual(captured_dev_env["TILEFOUNDRY_EXPECTED_TILELANG_SOURCE"], str(self.tl_src.resolve()))
            self.assertTrue(captured_dev_env["PYTHONPATH"].startswith(str(self.tl_src.resolve()) + ":"))

            values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
            info = {
                "missing": [],
                "pythonVersion": "3.12.16",
                "torchVersion": "2.13.0+cpu",
                "torchNpuVersion": "2.13.0",
                "torchCmakeDir": "/tmp/Torch",
                "torchNpuDir": "/tmp/torch_npu",
            }
            _, slot_dir, utils_dir = GATE._abi_slot(values, self.src, self.repo, self.work_root, info)
            (utils_dir / "cann_bench_utils").mkdir(parents=True)
            (utils_dir / "cann_bench_utils" / "_C.so").write_bytes(b"ELF")
            (slot_dir / "build_meta.json").write_text("{}", encoding="utf-8")

            captured_cfg = {}

            def fake_bench(cmd, *, cwd, env, **kwargs):
                cfg_path = Path(cmd.split()[-1])
                captured_cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))
                payload = {
                    "officialReportValid": True,
                    "allCasesPassed": True,
                    "allTimingsValid": True,
                    "targetCount": 1,
                    "caseCount": 1,
                    "passedCases": 1,
                    "timedCases": 1,
                    "score": 80.0,
                    "avgSpeedup": 1.2,
                    "results": [],
                }
                return types.SimpleNamespace(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

            with patch.dict(os.environ, dev_env, clear=True), \
                    patch.object(GATE, "_inspect_conda_cannbench_env", return_value=info), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=fake_bench):
                rc_b, obs_b = GATE.run_cannbench(values, 60.0)
            self.assertEqual(rc_b, 0)
            self.assertEqual(captured_cfg["tilelang_source"], str(self.tl_src.resolve()))
            self.assertEqual(obs_b["tilelang"]["source"], str(self.tl_src.resolve()))
        finally:
            os.chdir(old_cwd)

    def test_missing_sabi_or_stamp_or_mutated_native_fails_closed(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            env = {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file)}

            # 1. Native source edit invalidates stamp -> BLOCKED_ENV (75) naming tilelang-npu-build.
            (self.tl_src / "src/tl/kernel.cc").write_text("// native v2\n", encoding="utf-8")
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(GATE.BridgeError) as err:
                    GATE.tilelang_reuse_identity(GATE.machine_values())
                self.assertIn("tilelang-npu-build", str(err.exception))
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config"])
                self.assertEqual(caught.exception.code, 75)

            # Restore matching stamp, then drop stamp -> fails closed.
            self.refresh_stamp()
            (self.tl_src / "build" / GATE.TILELANG_BUILD_STAMP_FILE).unlink()
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(GATE.BridgeError) as err2:
                    GATE.tilelang_reuse_identity(GATE.machine_values())
                self.assertIn("tilelang-npu-build", str(err2.exception))

            # Restore stamp, then drop one .abi3.so (Python-3.11-only tree) -> fails closed.
            self.refresh_stamp()
            (self.tl_src / "build/lib/tilelang_cython_wrapper.abi3.so").unlink()
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(GATE.BridgeError) as err3:
                    GATE.tilelang_reuse_identity(GATE.machine_values())
                self.assertIn("tilelang-npu-build", str(err3.exception))
        finally:
            os.chdir(old_cwd)


class OptionalAgentKeysCase(unittest.TestCase):
    """Optional agy LLM and reference repo keys are validated and forwarded when non-empty."""

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

    def test_optional_agent_and_repo_keys_forward_and_validate(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            self.write_env(
                TILEFOUNDRY_NPU_LLM_BACKEND="agy_cli",
                TILEFOUNDRY_NPU_AGY_BIN="/opt/agy/bin/agy",
                TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL="gemini-4-hydrogen-medium",
                TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL="gemini-4-hydrogen-high",
                TILEFOUNDRY_NPU_AGY_COMMON_ARGS='"--print --mode plan"',
                TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY="3",
                TILEFOUNDRY_NPU_TILELANG_REPO="/refs/tilelang",
                TILEFOUNDRY_NPU_CANNBOT_DSL_REPO="/refs/cannbot-dsl",
                TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO="/refs/ops-transformer",
                TILEFOUNDRY_NPU_ASC_DEVKIT_REPO="",
            )
            with patch.dict(os.environ, {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file)}, clear=True):
                self.assertEqual(GATE.main(["--check-config"]), 0)
                captured_env = {}

                def fake_cpu(cmd, *, cwd, env, **kwargs):
                    captured_env.update(env)
                    junit = Path(cmd.split("--junit-xml=")[1].split()[0])
                    junit.write_text('<testsuite tests="1" errors="0" failures="0" skipped="0"/>', encoding="utf-8")
                    return types.SimpleNamespace(returncode=0, stdout="", stderr="")

                with patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=fake_cpu):
                    rc, _obs = GATE.run_cpu(GATE.machine_values(), 30.0)
                self.assertEqual(rc, 0)
                self.assertEqual(captured_env["TILEFOUNDRY_NPU_LLM_BACKEND"], "agy_cli")
                self.assertEqual(captured_env["TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL"], "gemini-4-hydrogen-medium")
                self.assertEqual(captured_env["TILEFOUNDRY_NPU_AGY_COMMON_ARGS"], "--print --mode plan")
                self.assertEqual(captured_env["TILEFOUNDRY_NPU_TILELANG_REPO"], "/refs/tilelang")
                self.assertNotIn("TILEFOUNDRY_NPU_ASC_DEVKIT_REPO", captured_env)

            self.write_env(TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY="0")
            with patch.dict(os.environ, {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file)}, clear=True):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config"])
                self.assertEqual(caught.exception.code, 75)

            self.write_env(TILEFOUNDRY_NPU_TILELANG_REPO="relative/tilelang")
            with patch.dict(os.environ, {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file)}, clear=True):
                with self.assertRaises(SystemExit) as caught2:
                    GATE.main(["--check-config"])
                self.assertEqual(caught2.exception.code, 75)
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
