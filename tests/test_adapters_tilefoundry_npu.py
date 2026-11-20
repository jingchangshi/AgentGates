"""Tests for the host-local tilefoundry-npu adapter."""
from __future__ import annotations

import hashlib
import importlib.util
import io
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


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_script(SCRIPT, "tilefoundry_host_gate_test")
ENV_CONTEXT = load_script(SCRIPT.with_name("env_context.py"), "tilefoundry_env_context_test")


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
        remote_recipe = manifest.resolve_recipe("tilefoundry-npu.cannbench-remote", manifest.root)
        self.assertFalse(cpu_recipe.uses_device)
        self.assertIsNone(cpu_recipe.device_window)
        self.assertTrue(dev_recipe.uses_device)
        self.assertEqual(dev_recipe.device_window, 60.0)
        self.assertTrue(prep_recipe.uses_device)
        self.assertEqual(prep_recipe.device_window, 60.0)
        self.assertTrue(bench_recipe.uses_device)
        self.assertEqual(bench_recipe.device_window, 60.0)
        # The L4 prep and official gates accept the managed target/candidate
        # pair so the two recipes stay bound to one target, while an unmanaged
        # invocation still runs the configured targets file.
        self.assertFalse(prep_recipe.dynamic_target)
        self.assertTrue(prep_recipe.dynamic_target_optional)
        self.assertFalse(bench_recipe.dynamic_target)
        self.assertTrue(bench_recipe.dynamic_target_optional)
        self.assertFalse(remote_recipe.uses_device)
        self.assertIsNone(remote_recipe.device_window)
        # L3 and L4 prep/official control processes must not load project
        # sitecustomize or inherited PYTHON* hooks.
        for recipe in (prep_recipe, bench_recipe):
            self.assertEqual(recipe.argv[0], "bash")
            self.assertTrue(recipe.argv[2].endswith("/scripts/host_control.sh"))
            self.assertTrue(recipe.config_check[2].endswith("/scripts/host_control.sh"))
        wrapper = (ADAPTER / "scripts/host_control.sh").read_text(encoding="utf-8")
        self.assertIn("-S", wrapper)
        self.assertIn("-E", wrapper)

    def test_key_and_engine_wiring_parity(self):
        self.assertEqual(tuple(sorted(GATE.TOOLCHAIN_KEYS)), tuple(sorted(machine_env.TILEFOUNDRY_NPU_KEYS)))
        self.assertEqual(set(ENV_CONTEXT.OPTIONAL_FORWARDED_KEYS), set(GATE.OPTIONAL_FORWARDED_KEYS))
        self.assertEqual(machine_env.ADAPTERS["tilefoundry-npu"]["prefix"], "TILEFOUNDRY_NPU_")
        self.assertEqual(engine.ENV_KEYS["tilefoundry-npu"], "TILEFOUNDRY_NPU_ENV_FILE")
        self.assertEqual(engine.DEVICE_ADAPTERS["tilefoundry-npu"], "TILEFOUNDRY_NPU_DEVICE_ID")
        self.assertEqual(engine.EXCLUSIVE_VARIABLE["tilefoundry-npu"], "TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE")


    def test_l3_preflight_uses_the_exported_tilelang_environment_api(self):
        # tilelang/__init__.py exports `env` as an Environment instance, which
        # shadows `import tilelang.env as ...`; the cache API belongs to that
        # exported instance. Reverting to `tilelang_env.env` breaks real runs.
        self.assertIn("from tilelang.env import env as tilelang_env", GATE.L3_PREFLIGHT)
        self.assertIn("tilelang_env.get_cache_write_dir()", GATE.L3_PREFLIGHT)
        self.assertNotIn("tilelang_env.env", GATE.L3_PREFLIGHT)


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

    def test_provider_settings_are_parsed_and_forwarded_from_the_machine_file(self):
        settings = {
            "TILEFOUNDRY_NPU_LLM_BACKEND": "openai_compat",
            "TILEFOUNDRY_NPU_LLM_API_BASE": "https://llm.example.invalid/v1",
            "TILEFOUNDRY_NPU_LLM_API_KEY_ENV": "MODEL_PROVIDER_BEARER",
            "TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL": "exploit-model",
            "TILEFOUNDRY_NPU_LLM_EXPLORE_MODEL": "explore-model",
            "TILEFOUNDRY_NPU_LLM_ESCALATE_MODEL": "escalate-model",
            "TILEFOUNDRY_NPU_LLM_REPAIR_MODEL": "repair-model",
            "TILEFOUNDRY_NPU_LLM_MAX_EFFORT": "xhigh",
            "TILEFOUNDRY_NPU_LLM_EFFORT_CAP": "xhigh",
            "TILEFOUNDRY_NPU_LLM_PASS_REASONING_EFFORT": "1",
            "TILEFOUNDRY_NPU_LLM_STREAM": "1",
            "TILEFOUNDRY_NPU_LLM_MAX_CONCURRENCY": "2",
            "TILEFOUNDRY_NPU_LLM_EXPLOIT_TIMEOUT_SEC": "120",
            "TILEFOUNDRY_NPU_LLM_REASONING_TIMEOUT_SEC": "180.5",
            "TILEFOUNDRY_NPU_LLM_CLI_BIN": "model-cli",
            "TILEFOUNDRY_NPU_LLM_CLI_TEMPLATE": "{bin} exec -m {model} {prompt}",
            "TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL": "legacy-escalate-model",
        }
        file_values = {**settings, "TILEFOUNDRY_NPU_DEVICE_TESTS": "tests/ascend/target"}
        template_key = "TILEFOUNDRY_NPU_LLM_CLI_TEMPLATE"
        file_values[template_key] = f'"{settings[template_key]}"'
        self.write_env(**file_values)
        credential_name = settings["TILEFOUNDRY_NPU_LLM_API_KEY_ENV"]
        credential = "fixture-only-key"
        other_key = "OTHER_PROVIDER_API_KEY"
        worker_env = {
            "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
            "TILEFOUNDRY_NPU_DEVICE_ID": "0",
            "AGENT_GATES_DEVICE_LEASE_PATH": str(self.root / "npu.lock"),
            credential_name: credential,
            other_key: "unrelated-key",
        }
        with patch.dict(os.environ, worker_env, clear=True):
            values = GATE.machine_values()
            machine = machine_env.read("tilefoundry-npu")
            self.assertEqual(values, machine.values)
            self.assertEqual({key: values[key] for key in settings}, settings)
            self.assertEqual(ENV_CONTEXT.forward_optional_machine_env({}, values), {
                **settings, "TILEFOUNDRY_NPU_PYTHON": str(self.py),
            })

            cpu_env = GATE._host_child_env(values)
            GATE._forward_optional_env(cpu_env, values, include_agent_llm=False)
            self.assertTrue(settings.keys().isdisjoint(cpu_env))
            self.assertNotIn(credential_name, cpu_env)
            self.assertNotIn(other_key, cpu_env)

            stderr = io.StringIO()
            failed = types.SimpleNamespace(returncode=1, stdout=credential, stderr=credential)
            with patch.object(sys, "stderr", stderr), \
                    patch.object(GATE, "_source_root", return_value=self.src), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", return_value=failed) as child:
                rc, observations = GATE.run_device(values, 30.0)
            self.assertEqual(rc, 1)
            device_env = child.call_args.kwargs["env"]
            self.assertEqual({key: device_env[key] for key in settings}, settings)
            self.assertEqual(device_env[credential_name], credential)
            self.assertNotIn(other_key, device_env)
            self.assertNotIn(credential, json.dumps(observations))
            self.assertEqual(stderr.getvalue(), "[redacted][redacted]")

    def test_provider_key_assignment_is_rejected_without_printing_its_value(self):
        raw_key = "TILEFOUNDRY_NPU_LLM_API_KEY"
        credential = "fixture-only-key"
        selector = {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file)}
        for file_values, exports in (
            ({raw_key: credential}, {}),
            ({}, {raw_key: credential}),
            ({"TILEFOUNDRY_NPU_LLM_API_KEY_ENV": credential}, {}),
        ):
            with self.subTest(file_keys=tuple(file_values), export_keys=tuple(exports)):
                self.write_env(**file_values)
                stderr = io.StringIO()
                with patch.dict(os.environ, {**selector, **exports}, clear=True), \
                        patch.object(sys, "stderr", stderr), self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config"])
                self.assertEqual(caught.exception.code, 75)
                self.assertNotIn(credential, stderr.getvalue())

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
            credential_name = "MODEL_PROVIDER_BEARER"
            self.write_env(TILEFOUNDRY_NPU_LLM_API_KEY_ENV=credential_name)
            env = {
                "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
                "TILEFOUNDRY_NPU_DEVICE_ID": "1",
                "AGENT_GATES_DEVICE_LEASE_PATH": str(lease),
                "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:1",
                "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "2",
                credential_name: "example-credential",
            }
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", return_value=fake_completed) as child:
                rc, obs = GATE.run_device(GATE.machine_values(), 30.0)
            self.assertEqual(rc, 0)
            self.assertNotIn(credential_name, child.call_args.kwargs["env"])
            self.assertTrue(obs["exitCodeZero"])
            self.assertTrue(obs["windowOpened"])
            self.assertTrue(obs["windowReleased"])
            self.assertEqual(obs["deviceWindow"]["resource"], "npu:1")
            self.assertEqual(obs["deviceWindow"]["occupancy"], "free")
        finally:
            os.chdir(old_cwd)

    def test_run_cpu_isolates_user_site_and_excludes_agent_llm_keys(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            self.write_env(
                TILEFOUNDRY_NPU_LLM_BACKEND="agy_cli",
                TILEFOUNDRY_NPU_AGY_BIN=str(self.py),
                TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL="custom-exploit-model",
                TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL="custom-explore-model",
                TILEFOUNDRY_NPU_AGY_COMMON_ARGS="--print",
                TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY="4",
            )
            captured_envs = []

            def fake_run_host_bash(cmd, *, cwd, env, **kwargs):
                captured_envs.append(dict(env))
                # Write minimal passing JUnit XML referenced by --junit-xml=...
                for part in cmd.split():
                    if part.startswith("--junit-xml="):
                        xml_path = Path(part.split("=", 1)[1].strip("'\""))
                        xml_path.write_text(
                            '<testsuite tests="1" failures="0" errors="0" skipped="0"></testsuite>',
                            encoding="utf-8",
                        )
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.dict(os.environ, {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file)}, clear=True), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=fake_run_host_bash):
                rc, obs = GATE.run_cpu(GATE.machine_values(), 30.0)
            self.assertEqual(rc, 0)
            self.assertTrue(obs["exitCodeZero"])
            self.assertEqual(len(captured_envs), 1)
            cpu_env = captured_envs[0]
            self.assertEqual(cpu_env.get("PYTHONNOUSERSITE"), "1")
            for key in GATE.OPTIONAL_AGENT_LLM_KEYS:
                self.assertNotIn(key, cpu_env)
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

        self.repo = self.root / "cann-bench"
        for sub in (
            "scripts",
            "src/kernel_eval",
            "src/cann_bench_utils",
            "tasks/metadata",
            "tasks/level4/attention_residual",
        ):
            (self.repo / sub).mkdir(parents=True)
        (self.repo / "scripts/run_evaluation.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.repo / "src/cann_bench_utils/CMakeLists.txt").write_text("# cmake\n", encoding="utf-8")
        (self.repo / "src/cann_bench_utils/ext.cpp").write_text("// ext\n", encoding="utf-8")
        for fname in ("proto.yaml", "cases.yaml", "golden.py"):
            (self.repo / "tasks/level4/attention_residual" / fname).write_text(f"# {fname}\n", encoding="utf-8")

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
        (self.repo / "tasks/level4/mla").mkdir(parents=True)
        for fname in ("proto.yaml", "cases.yaml", "golden.py"):
            (self.repo / "tasks/level4/mla" / fname).write_text(f"# {fname}\n", encoding="utf-8")
        (self.src / "examples/ascend950pr_ops/mla").mkdir(parents=True)
        (self.src / "examples/ascend950pr_ops/mla/kernels.py").write_text("# mla\n", encoding="utf-8")
        (self.src / "examples/ascend950pr_ops/cannbench_targets.json").write_text(
            json.dumps({"targets": [{"task": "level4/mla", "candidate": "examples/ascend950pr_ops/mla/kernels.py"}]}),
            encoding="utf-8",
        )
        origin2, targets2 = GATE.resolve_cannbench_targets(values, self.src, self.repo)
        self.assertEqual(origin2, "repo-file")
        self.assertEqual(targets2, (("level4/mla", "examples/ascend950pr_ops/mla/kernels.py"),))

    def test_managed_request_narrows_l4_to_the_requested_pair(self):
        values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
        requested_candidate = "examples/ascend950pr_ops/attention_residual/kernels_v2.py"
        (self.src / requested_candidate).write_text("# candidate v2\n", encoding="utf-8")
        with patch.dict(
            os.environ,
            {
                "AGENT_GATES_TARGET": "level4/attention_residual",
                "AGENT_GATES_CANDIDATE": requested_candidate,
            },
            clear=False,
        ):
            origin, narrowed = GATE.resolve_cannbench_targets(
                values, self.src, self.repo, allow_request=True
            )
        self.assertEqual(origin, "dynamic-request")
        self.assertEqual(narrowed, (("level4/attention_residual", requested_candidate),))

    def test_managed_request_accepts_a_task_outside_the_targets_file(self):
        values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
        (self.repo / "tasks/level4/mla").mkdir(parents=True)
        for fname in ("proto.yaml", "cases.yaml", "golden.py"):
            (self.repo / "tasks/level4/mla" / fname).write_text(f"# {fname}\n", encoding="utf-8")
        (self.src / "examples/ascend950pr_ops/mla").mkdir(parents=True)
        (self.src / "examples/ascend950pr_ops/mla/kernels.py").write_text("# mla\n", encoding="utf-8")
        with patch.dict(
            os.environ,
            {
                "AGENT_GATES_TARGET": "level4/mla",
                "AGENT_GATES_CANDIDATE": "examples/ascend950pr_ops/mla/kernels.py",
            },
            clear=False,
        ):
            origin, narrowed = GATE.resolve_cannbench_targets(
                values, self.src, self.repo, allow_request=True
            )
        self.assertEqual(origin, "dynamic-request")
        self.assertEqual(narrowed, (("level4/mla", "examples/ascend950pr_ops/mla/kernels.py"),))

    def test_managed_request_keeps_the_targets_file_without_a_pair(self):
        values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
        with patch.dict(os.environ, {}, clear=True):
            origin, targets = GATE.resolve_cannbench_targets(
                values, self.src, self.repo, allow_request=True
            )
        self.assertEqual(origin, "env-file")
        self.assertEqual(
            targets,
            (("level4/attention_residual", "examples/ascend950pr_ops/attention_residual/kernels.py"),),
        )

    def test_a_partial_managed_request_is_refused(self):
        values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
        with patch.dict(os.environ, {"AGENT_GATES_TARGET": "level4/attention_residual"}, clear=True):
            with self.assertRaises(GATE.BridgeError):
                GATE.resolve_cannbench_targets(
                    values, self.src, self.repo, allow_request=True
                )
        with patch.dict(
            os.environ,
            {"AGENT_GATES_CANDIDATE": "examples/ascend950pr_ops/attention_residual/kernels.py"},
            clear=True,
        ):
            with self.assertRaises(GATE.BridgeError):
                GATE.resolve_cannbench_targets(
                    values, self.src, self.repo, allow_request=True
                )

    def test_prep_uses_only_the_requested_pair_not_the_machine_target(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            values = GATE._parse_env_file(
                self.env_file.read_text(encoding="utf-8"), self.env_file
            )
            info = {
                "missing": [],
                "pythonVersion": "3.12.16",
                "torchVersion": "2.13.0+cpu",
                "torchNpuVersion": "2.13.0",
                "torchCmakeDir": "/tmp/Torch",
                "torchNpuDir": "/tmp/torch_npu",
            }
            (self.repo / "tasks/level4/mla").mkdir(parents=True)
            for fname in ("proto.yaml", "cases.yaml", "golden.py"):
                (self.repo / "tasks/level4/mla" / fname).write_text(
                    f"# {fname}\n", encoding="utf-8"
                )
            (self.src / "examples/ascend950pr_ops/mla").mkdir(parents=True)
            mla_candidate = "examples/ascend950pr_ops/mla/kernels.py"
            (self.src / mla_candidate).write_text("# mla\n", encoding="utf-8")
            # Force a prebuilt ABI slot: prep then performs only its common
            # cann_bench_utils probe, never target inspection or execution.
            _, slot_dir, utils_dir = GATE._abi_slot(
                values, self.src, self.repo, self.work_root, info
            )
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
                "AGENT_GATES_TARGET": "level4/mla",
                "AGENT_GATES_CANDIDATE": mla_candidate,
            }
            commands: list[str] = []

            def fake_run_host_bash(cmd, *, cwd, env, **kwargs):
                commands.append(cmd)
                return types.SimpleNamespace(
                    returncode=0,
                    stdout='{"cannBenchUtilsCallable": true}\n',
                    stderr="",
                )

            with patch.dict(os.environ, env, clear=True), \
                    patch.object(GATE, "_inspect_conda_cannbench_env", return_value=info), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(
                        GATE.adapter_runtime, "run_host_bash", side_effect=fake_run_host_bash
                    ):
                rc, obs = GATE.run_cannbench_prep(values, 60.0)
            self.assertEqual(rc, 0, obs)
            self.assertEqual(obs["targetsSource"], "dynamic-request")
            self.assertEqual(obs["targets"], [{
                "task": "level4/mla",
                "candidate": mla_candidate,
            }])
            rendered = json.dumps(obs, sort_keys=True) + "\n" + "\n".join(commands)
            self.assertNotIn("attention_residual", rendered)
            for command in commands:
                self.assertNotIn("tasks/level4/", command)
        finally:
            os.chdir(old_cwd)

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
                "PYTHONPATH": "/project/src",
                "PYTHONHOME": "/project/python",
                "PYTHONSTARTUP": "/project/sitecustomize.py",
                "PYTHONUSERBASE": "/project/userbase",
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
            self.assertEqual(obs_p["targetsSource"], "managed")
            self.assertEqual(obs_p["targets"], [])
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
            captured_bench_envs = []
            def fake_bench_bash(cmd, *, cwd, env, **kwargs):
                captured_bench_envs.append((cmd, dict(env)))
                return bench_res

            with patch.dict(os.environ, env, clear=True), \
                    patch.object(GATE, "_inspect_conda_cannbench_env", return_value=info), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.locking, "lock_root", return_value=self.root / "locks"), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=fake_bench_bash):
                rc_b, obs_b = GATE.run_cannbench(values, 60.0)
            self.assertEqual(rc_b, 0)
            self.assertTrue(obs_b["officialReportValid"])
            self.assertEqual(obs_b["score"], 81.25)
            self.assertEqual(obs_b["targetsSource"], "env-file")
            self.assertEqual(obs_b["deviceWindow"]["resource"], "npu:1")
            captured_env = captured_bench_envs[0][1]
            self.assertEqual(captured_env.get("PYTHONNOUSERSITE"), "1")
            for hook in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
                self.assertNotIn(hook, captured_env)
            self.assertIn("-E -s", captured_bench_envs[0][0])
            self.assertIn('AGENT_GATES_ACTIVATION_PYTHONPATH="${PYTHONPATH:-}"', captured_bench_envs[0][0])
            self.assertIn('env["PYTHONNOUSERSITE"] = "1"', GATE.CANNBENCH_RUNNER)
            self.assertIn('for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):', GATE.CANNBENCH_RUNNER)
            self.assertIn('env.pop("AGENT_GATES_ACTIVATION_PYTHONPATH", "")', GATE.CANNBENCH_RUNNER)
            self.assertNotIn('pp_entries.append(env["PYTHONPATH"])', GATE.CANNBENCH_RUNNER)
            self.assertTrue((self.root / "locks" / ".msprof.lock").is_file())

            # A managed request narrows the official run to its pair; the runner
            # config proves exactly one task/candidate reaches the workload.
            managed_candidate = "examples/ascend950pr_ops/attention_residual/kernels_v2.py"
            (self.src / managed_candidate).write_text("# candidate v2\n", encoding="utf-8")
            stage = self.root / "runner-stage"
            stage.mkdir()

            class KeepStage:
                def __enter__(self):
                    return str(stage)

                def __exit__(self, *exc_info):
                    return False

            managed_env = dict(env)
            managed_env["AGENT_GATES_TARGET"] = "level4/attention_residual"
            managed_env["AGENT_GATES_CANDIDATE"] = managed_candidate
            with patch.dict(os.environ, managed_env, clear=True), \
                    patch.object(GATE, "_inspect_conda_cannbench_env", return_value=info), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.locking, "lock_root", return_value=self.root / "locks"), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=fake_bench_bash), \
                    patch.object(GATE.tempfile, "TemporaryDirectory", return_value=KeepStage()):
                rc_m, obs_m = GATE.run_cannbench(values, 60.0)
            self.assertEqual(rc_m, 0)
            self.assertEqual(obs_m["targetsSource"], "dynamic-request")
            runner_cfg = json.loads((stage / "runner_cfg.json").read_text(encoding="utf-8"))
            self.assertEqual(runner_cfg["targets"], [{
                "task": "level4/attention_residual",
                "candidate": managed_candidate,
            }])

            # Prep accepts the same pair and binds itself to it before the
            # official recipe runs.
            with patch.dict(os.environ, managed_env, clear=True), \
                    patch.object(GATE, "_inspect_conda_cannbench_env", return_value=info), \
                    patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                    patch.object(GATE.adapter_runtime, "run_host_bash", return_value=prep_res):
                rc_pm, obs_pm = GATE.run_cannbench_prep(values, 60.0)
            self.assertEqual(rc_pm, 0)
            self.assertEqual(obs_pm["targetsSource"], "dynamic-request")
            self.assertEqual(obs_pm["targets"], [{
                "task": "level4/attention_residual",
                "candidate": managed_candidate,
            }])
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

        self.repo = self.root / "cann-bench"
        for sub in (
            "scripts",
            "src/kernel_eval",
            "src/cann_bench_utils",
            "tasks/metadata",
            "tasks/level4/attention_residual",
        ):
            (self.repo / sub).mkdir(parents=True)
        (self.repo / "scripts/run_evaluation.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.repo / "src/cann_bench_utils/CMakeLists.txt").write_text("# cmake\n", encoding="utf-8")
        for fname in ("proto.yaml", "cases.yaml", "golden.py"):
            (self.repo / "tasks/level4/attention_residual" / fname).write_text(f"# {fname}\n", encoding="utf-8")

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
                TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL="agy-escalation-model",
                TILEFOUNDRY_NPU_AGY_COMMON_ARGS='"--print --mode plan"',
                TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY="3",
                TILEFOUNDRY_NPU_CPU_TESTS="",
                TILEFOUNDRY_NPU_DEVICE_TESTS="",
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

                with patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=fake_cpu) as cpu:
                    rc, _obs = GATE.run_cpu(GATE.machine_values(), 30.0)
                self.assertEqual(rc, 0)
                self.assertTrue(cpu.call_args.args[0].endswith(" " + " ".join(GATE.DEFAULT_CPU_TESTS)))
                self.assertNotIn("TILEFOUNDRY_NPU_LLM_BACKEND", captured_env)
                self.assertNotIn("TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL", captured_env)
                self.assertEqual(captured_env["TILEFOUNDRY_NPU_TILELANG_REPO"], "/refs/tilelang")
                self.assertNotIn("TILEFOUNDRY_NPU_ASC_DEVKIT_REPO", captured_env)

                captured_dev_env = {}
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

                def fake_dev(cmd, *, cwd, env, **kwargs):
                    captured_dev_env.update(env)
                    return types.SimpleNamespace(returncode=0, stdout=probe_json + "\n", stderr="")

                lease = self.root / "locks" / "npu.lock"
                dev_env = {
                    "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
                    "TILEFOUNDRY_NPU_DEVICE_ID": "0",
                    "AGENT_GATES_DEVICE_LEASE_PATH": str(lease),
                    "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:0",
                    "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "2",
                }
                with patch.dict(os.environ, dev_env, clear=True), \
                        patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                        patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=fake_dev) as device:
                    rc_d, _obs_d = GATE.run_device(GATE.machine_values(), 30.0)
                self.assertEqual(rc_d, 0)
                self.assertNotIn("-m pytest", device.call_args.args[0])
                self.assertEqual(captured_dev_env["TILEFOUNDRY_NPU_LLM_BACKEND"], "agy_cli")
                self.assertEqual(captured_dev_env["TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL"], "gemini-4-hydrogen-medium")
                self.assertEqual(captured_dev_env["TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL"], "agy-escalation-model")
                self.assertEqual(captured_dev_env["TILEFOUNDRY_NPU_AGY_COMMON_ARGS"], "--print --mode plan")
                self.assertEqual(captured_dev_env["TILEFOUNDRY_NPU_TILELANG_REPO"], "/refs/tilelang")

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


class CannbenchRemoteCase(unittest.TestCase):
    """Remote CANNBench submission validates MCP rules, packages compliant ZIPs, and submits via MCP."""

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
        (self.src / "examples/ascend950pr_ops/attention_residual/kernels.py").write_text(
            "def attention_residual(x):\n    return x\n",
            encoding="utf-8",
        )

        self.repo = self.root / "cann-bench"
        for sub in (
            "scripts",
            "src/kernel_eval",
            "src/cann_bench_utils",
            "tasks/metadata",
            "tasks/level4/attention_residual",
        ):
            (self.repo / sub).mkdir(parents=True)
        (self.repo / "scripts/run_evaluation.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.repo / "src/cann_bench_utils/CMakeLists.txt").write_text("# cmake\n", encoding="utf-8")
        (self.repo / "tasks/level4/attention_residual/proto.yaml").write_text(
            "operator:\n  name: AttentionResidual\n  schema: attention_residual(Tensor x) -> Tensor\n",
            encoding="utf-8",
        )
        for fname in ("cases.yaml", "golden.py"):
            (self.repo / "tasks/level4/attention_residual" / fname).write_text(f"# {fname}\n", encoding="utf-8")

        self.work_root = self.root / "tf-work"
        self.cann = self.root / "set_env.sh"
        self.cann.write_text("#!/bin/sh\n", encoding="utf-8")
        self.conda_sh = self.root / "conda.sh"
        self.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        self.py = self.root / "python3"
        self.py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.py.chmod(0o755)
        self.token_file = self.root / "mcp.token"
        self.token_file.write_text("secret-bearer-token\n", encoding="utf-8")
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
            "TILEFOUNDRY_NPU_CANNBENCH_TARGETS": "level4/attention_residual:examples/ascend950pr_ops/attention_residual/kernels.py",
            "TILEFOUNDRY_NPU_CANNBENCH_MCP_URL": "https://cannbench.com/mcp",
            "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE": str(self.token_file),
        }
        data.update(overrides)
        self.env_file.write_text("".join(f"{k}={v}\n" for k, v in data.items()), encoding="utf-8")

    def test_check_config_cannbench_remote_requires_token_and_validates_consent(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            env = {"TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file)}
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(GATE.main(["--check-config-cannbench-remote"]), 0)

            self.write_env(TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE="", TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN="")
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit) as caught:
                    GATE.main(["--check-config-cannbench-remote"])
                self.assertEqual(caught.exception.code, 75)

            self.write_env(TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT="invalid")
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit) as caught2:
                    GATE.main(["--check-config-cannbench-remote"])
                self.assertEqual(caught2.exception.code, 75)
        finally:
            os.chdir(old_cwd)

    def test_remote_compliance_blocks_cheat_patterns_without_network_call(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            (self.src / "examples/ascend950pr_ops/attention_residual/kernels.py").write_text(
                "import torch.nn.functional as F\ndef attention_residual(x):\n    return F.softmax(x.cpu(), dim=-1)\n",
                encoding="utf-8",
            )
            values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
            with patch.object(GATE, "_mcp_http_post") as mock_post:
                rc, obs = GATE.run_cannbench_remote(values, 60.0)
            self.assertEqual(rc, 1)
            self.assertFalse(obs["compliancePassed"])
            self.assertFalse(obs["mcpSubmitted"])
            self.assertTrue(obs["results"][0]["violations"])
            mock_post.assert_not_called()
        finally:
            os.chdir(old_cwd)

    def test_remote_submit_packages_zip_and_forwards_mcp_rules(self):
        import zipfile
        old_cwd = os.getcwd()
        try:
            os.chdir(self.src)
            self.write_env(
                TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION="2.1.0",
                TILEFOUNDRY_NPU_CANNBENCH_POOL_ID="pool-950pr",
                TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT="v1",
            )
            values = GATE._parse_env_file(self.env_file.read_text(encoding="utf-8"), self.env_file)
            calls = []

            def fake_post(url, token, payload, *, session_id="", timeout=30.0):
                calls.append((url, token, payload, session_id))
                method = payload.get("method")
                if method == "initialize":
                    return {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-03-26"}}, "sid-123"
                if method == "notifications/initialized":
                    return {}, "sid-123"
                return {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "structuredContent": {
                            "submission_id": "sub-999",
                            "status": "queued",
                            "benchmark_version": "2.1.0",
                            "pool_id": "pool-950pr",
                        }
                    },
                }, "sid-123"

            with patch.object(GATE, "_mcp_http_post", side_effect=fake_post):
                rc, obs = GATE.run_cannbench_remote(values, 60.0)
            self.assertEqual(rc, 0)
            self.assertTrue(obs["compliancePassed"])
            self.assertTrue(obs["archiveValid"])
            self.assertTrue(obs["mcpSubmitted"])
            self.assertEqual(obs["results"][0]["submissionId"], "sub-999")
            self.assertEqual(obs["results"][0]["benchmarkVersion"], "2.1.0")
            self.assertEqual(obs["results"][0]["poolId"], "pool-950pr")

            zip_path = Path(obs["results"][0]["archivePath"])
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())
            self.assertEqual(names, {"build.sh", "setup.py", "cann_bench/__init__.py", "cann_bench/attention_residual.py"})

            submit_call = calls[-1][2]["params"]
            self.assertEqual(submit_call["name"], "submit_kernel")
            self.assertEqual(submit_call["arguments"]["benchmark_version"], "2.1.0")
            self.assertEqual(submit_call["arguments"]["requested_pool_id"], "pool-950pr")
            self.assertEqual(submit_call["arguments"]["independent_environment_consent_version"], "v1")
            self.assertEqual(calls[-1][1], "secret-bearer-token")
            self.assertEqual(calls[-1][3], "sid-123")
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
