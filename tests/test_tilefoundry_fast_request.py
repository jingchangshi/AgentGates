"""Host-only tests for the V2-published, workspace-relative device-fast request."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "agent_gates" / "adapters" / "tilefoundry-npu"
SCRIPT = ADAPTER / "scripts" / "tilefoundry_host_gate.py"

REQUEST_ID = "0123456789abcdef0123456789abcdef"
PROFILE_ID = "fedcba9876543210fedcba9876543210"
PROBE_REQUEST_ID = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
PROBE_ID = "probe_unit_0"


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_script(SCRIPT, "tilefoundry_fast_request_test")

FAST_TEST_HEADER = """\
import hashlib
import json
import os
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(response: Path, payload: dict) -> None:
    response.parent.mkdir(parents=True, exist_ok=True)
    tmp = response.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, response)


def _envelope(request: dict, request_path: Path, status: str) -> dict:
    return {
        "schema": "tilefoundry-fast-response/v1",
        "request_id": request["request_id"],
        "request_sha256": _sha(request_path),
        "mode": request["mode"],
        "status": status,
        "operators": {},
        "msprof_telemetry": {},
        "candidates": {},
        "coverage": {
            "expected_candidates": 0,
            "evaluated_candidates": 0,
            "expected_msprof_targets": 0,
            "observed_msprof_targets": 0,
        },
    }
"""

BATCH_RESPONSE = """\
def test_surrogate_anchor_batch_npu() -> None:
    request_path = Path(os.environ["TF_FAST_REQUEST"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source = Path(os.environ["TF_FAST_SOURCE_ROOT"])
    response = source / request["response_path"]
    payload = _envelope(request, request_path, "ok")
    expected = 0
    for batch in request["batches"]:
        operator = batch["operator_name"]
        payload["operators"][operator] = {}
        payload["candidates"][operator] = {}
        for candidate in batch["candidates"]:
            module = source / candidate["module_path"]
            payload["operators"][operator][candidate["candidate_id"]] = {
                "all_passed": True,
                "score": 1.0,
            }
            payload["candidates"][operator][candidate["candidate_id"]] = {
                "module_path": candidate["module_path"],
                "sha256": candidate["sha256"],
                "post_sha256": _sha(module),
            }
            expected += 1
    payload["coverage"] = {
        "expected_candidates": expected,
        "evaluated_candidates": expected,
        "expected_msprof_targets": 0,
        "observed_msprof_targets": 0,
    }
    _write(response, payload)
"""

PROFILE_RESPONSE = """\
def test_surrogate_anchor_batch_npu() -> None:
    request_path = Path(os.environ["TF_FAST_REQUEST"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source = Path(os.environ["TF_FAST_SOURCE_ROOT"])
    response = source / request["response_path"]
    payload = _envelope(request, request_path, "ok")
    expected = 0
    for target in request["msprof_targets"]:
        operator = target["operator_name"]
        candidate_id = target["candidate_id"]
        module = source / target["module_path"]
        payload["msprof_telemetry"].setdefault(operator, {}).setdefault(
            candidate_id, {}
        )[str(target["case_num"])] = {"dur": 1.0}
        payload["candidates"].setdefault(operator, {})[candidate_id] = {
            "module_path": target["module_path"],
            "sha256": target["sha256"],
            "post_sha256": _sha(module),
        }
        expected += 1
    payload["coverage"] = {
        "expected_candidates": len(payload["candidates"][operator]),
        "evaluated_candidates": len(payload["candidates"][operator]),
        "expected_msprof_targets": expected,
        "observed_msprof_targets": expected,
    }
    _write(response, payload)
"""

MICRO_PROBE_RESPONSE = """\
def test_surrogate_anchor_batch_npu() -> None:
    request_path = Path(os.environ["TF_FAST_REQUEST"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source = Path(os.environ["TF_FAST_SOURCE_ROOT"])
    response = source / request["response_path"]
    payload = _envelope(request, request_path, "ok")
    payload["candidates"]["micro_probe"] = {}
    for candidate in request["candidates"]:
        module = source / candidate["module_path"]
        payload["candidates"]["micro_probe"][candidate["candidate_id"]] = {
            "module_path": candidate["module_path"],
            "sha256": candidate["sha256"],
            "post_sha256": _sha(module),
        }
    payload["coverage"] = {
        "expected_candidates": len(request["candidates"]),
        "evaluated_candidates": len(request["candidates"]),
        "expected_msprof_targets": 0,
        "observed_msprof_targets": 0,
    }
    payload["micro_probe"] = {
        "probe_id": request["probe"]["probe_id"],
        "metrics": {
            "baseline_dur_us": 10.0,
            "optimized_dur_us": 8.0,
            "speedup": 1.25,
            "numerical_ok": 1.0,
        },
    }
    _write(response, payload)
"""

MICRO_PROBE_MISSING_METRIC_RESPONSE = MICRO_PROBE_RESPONSE + """\
    del payload["micro_probe"]["metrics"]["speedup"]
    _write(response, payload)
"""

MICRO_PROBE_WRONG_ID_RESPONSE = MICRO_PROBE_RESPONSE + """\
    payload["micro_probe"]["probe_id"] = "probe_other"
    _write(response, payload)
"""

CANDIDATE_FAILURE_RESPONSE = """\
def test_surrogate_anchor_batch_npu() -> None:
    request_path = Path(os.environ["TF_FAST_REQUEST"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source = Path(os.environ["TF_FAST_SOURCE_ROOT"])
    response = source / request["response_path"]
    payload = _envelope(request, request_path, "candidate_failure")
    expected = 0
    for batch in request["batches"]:
        operator = batch["operator_name"]
        payload["operators"][operator] = {}
        payload["candidates"][operator] = {}
        for candidate in batch["candidates"]:
            module = source / candidate["module_path"]
            payload["operators"][operator][candidate["candidate_id"]] = {
                "all_passed": False,
            }
            payload["candidates"][operator][candidate["candidate_id"]] = {
                "module_path": candidate["module_path"],
                "sha256": candidate["sha256"],
                "post_sha256": _sha(module),
            }
            expected += 1
    payload["coverage"] = {
        "expected_candidates": expected,
        "evaluated_candidates": expected,
        "expected_msprof_targets": 0,
        "observed_msprof_targets": 0,
    }
    _write(response, payload)
    raise AssertionError("fast batch candidate(s) failed")
"""

REQUEST_ERROR_RESPONSE = """\
def test_surrogate_anchor_batch_npu() -> None:
    request_path = Path(os.environ["TF_FAST_REQUEST"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source = Path(os.environ["TF_FAST_SOURCE_ROOT"])
    payload = _envelope(request, request_path, "request_error")
    payload["error"] = "bad request"
    _write(source / request["response_path"], payload)
    raise RuntimeError("bad request")
"""

INFRASTRUCTURE_RESPONSE = """\
def test_surrogate_anchor_batch_npu() -> None:
    request_path = Path(os.environ["TF_FAST_REQUEST"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source = Path(os.environ["TF_FAST_SOURCE_ROOT"])
    payload = _envelope(request, request_path, "infrastructure_error")
    payload["error"] = "device lost"
    _write(source / request["response_path"], payload)
    raise RuntimeError("device lost")
"""

EXTRA_CANDIDATE_RESPONSE = BATCH_RESPONSE + """\
    for operator in payload["operators"]:
        payload["operators"][operator]["unexpected"] = {"score": 1.0}
    _write(response, payload)
"""

COVERAGE_MISMATCH_RESPONSE = BATCH_RESPONSE + """\
    payload["coverage"]["expected_candidates"] += 1
    _write(response, payload)
"""


class FastRequestCase(unittest.TestCase):
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
            "tests/ascend/agent",
            "tests/device",
            "build/agent_gates/fast",
        ):
            (self.src / sub).mkdir(parents=True)
        self.cann = self.root / "set_env.sh"
        self.cann.write_text(":\n", encoding="utf-8")
        self.conda_sh = self.root / "conda.sh"
        self.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        self.cannbench = self.root / "cannbench"
        for sub in (
            "scripts",
            "src/kernel_eval",
            "src/cann_bench_utils",
            "tasks/metadata",
        ):
            (self.cannbench / sub).mkdir(parents=True)
        (self.cannbench / "scripts" / "run_evaluation.sh").write_text(
            ":\n", encoding="utf-8"
        )
        (self.cannbench / "src" / "cann_bench_utils" / "CMakeLists.txt").write_text(
            "\n", encoding="utf-8"
        )
        self.work = self.root / "work"
        self.env_file = self.root / "machine.env"
        self.lock_root = self.root / "locks"
        self.lease = self.lock_root / "npu-1.lock"
        self.write_test_file(BATCH_RESPONSE)
        self.write_env()

    def write_env(self, **overrides):
        values = {
            "TILEFOUNDRY_NPU_CANN_SETUP": str(self.cann),
            "TILEFOUNDRY_NPU_CONDA_SH": str(self.conda_sh),
            "TILEFOUNDRY_NPU_CONDA_ENV": "tilefoundry-npu",
            "TILEFOUNDRY_NPU_PYTHON": sys.executable,
            "TILEFOUNDRY_NPU_DEVICE_TESTS": "tests/device/test_fast.py",
            "TILEFOUNDRY_NPU_CANNBENCH_REPO": str(self.cannbench),
            "TILEFOUNDRY_NPU_WORK_ROOT": str(self.work),
        }
        values.update(overrides)
        self.env_file.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )

    def write_test_file(self, body: str) -> None:
        (self.src / GATE.FAST_DEVICE_TEST_FILE).write_text(
            FAST_TEST_HEADER + body, encoding="utf-8"
        )

    def environment(self, request_id: str = REQUEST_ID):
        return {
            "TILEFOUNDRY_NPU_ENV_FILE": str(self.env_file),
            "TILEFOUNDRY_NPU_DEVICE_ID": "1",
            "AGENT_GATES_DEVICE_LEASE_PATH": str(self.lease),
            "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:1",
            "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "2",
            "AGENT_GATES_TARGET": f"fast/{request_id}",
            "AGENT_GATES_CANDIDATE": (
                f"build/agent_gates/fast/{request_id}.request.json"
            ),
        }

    @contextmanager
    def _in_source(self):
        old = os.getcwd()
        os.chdir(self.src)
        try:
            yield
        finally:
            os.chdir(old)

    def candidate_sha(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def stage_candidate(self, request_id: str, name: str, source: str) -> tuple[str, Path]:
        relative = f"build/agent_gates/fast/{request_id}/candidates/{name}"
        path = self.src / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.chmod(0o644)
        path.write_text(source, encoding="utf-8")
        return relative, path

    def batch_request(
        self,
        *,
        request_id: str = REQUEST_ID,
        response_path: str | None = None,
        sha_a: str | None = None,
        sha_b: str | None = None,
        require_all_passed: bool = False,
        module_a: str | None = None,
        module_b: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        default_a, path_a = self.stage_candidate(
            request_id, "cand_0.py", "def entry():\n    return 1\n"
        )
        default_b, path_b = self.stage_candidate(
            request_id, "cand_1.py", "def entry():\n    return 2\n"
        )
        self.candidate_a = path_a
        self.candidate_b = path_b
        payload = {
            "schema": "tilefoundry-fast-request/v1",
            "mode": "batch",
            "request_id": request_id,
            "response_path": response_path
            or f"build/agent_gates/fast/{request_id}.response.json",
            "warmup": 1,
            "repeat": 3,
            "require_all_passed": require_all_passed,
            "batches": [
                {
                    "operator_name": "unit_op",
                    "entrypoint": "entry",
                    "candidates": [
                        {
                            "candidate_id": "cand_0",
                            "module_path": module_a or default_a,
                            "sha256": sha_a or self.candidate_sha(path_a),
                        },
                        {
                            "candidate_id": "cand_1",
                            "module_path": module_b or default_b,
                            "sha256": sha_b or self.candidate_sha(path_b),
                        },
                    ],
                    "anchors": [
                        {
                            "task": "level4/unit_op",
                            "case_num": 1,
                            "regime_tag": "compute",
                            "ref_perf_us": 12.5,
                            "t_hw_us": 10.0,
                        }
                    ],
                }
            ],
        }
        payload.update(extra or {})
        return payload

    def profile_request(
        self,
        *,
        request_id: str = PROFILE_ID,
        response_path: str | None = None,
    ) -> dict:
        relative, path = self.stage_candidate(
            request_id, "probe.py", "def entry():\n    return 3\n"
        )
        self.candidate_a = path
        return {
            "schema": "tilefoundry-fast-request/v1",
            "mode": "profile",
            "request_id": request_id,
            "response_path": response_path
            or f"build/agent_gates/fast/{request_id}.response.json",
            "warmup": 1,
            "repeat": 3,
            "require_all_passed": False,
            "batches": [],
            "msprof_targets": [
                {
                    "operator_name": "unit_op",
                    "candidate_id": "bottleneck_probe",
                    "module_path": relative,
                    "sha256": self.candidate_sha(path),
                    "case_num": 2,
                    "warmup": 1,
                    "repeat": 3,
                }
            ],
        }

    def micro_probe_request(
        self,
        *,
        request_id: str = PROBE_REQUEST_ID,
        probe: dict | None = None,
        candidate_ids: tuple[str, str] = ("baseline", "optimized"),
        read_only: bool = True,
    ) -> dict:
        candidates = []
        for candidate_id, body in zip(
            candidate_ids,
            ("def make_flywheel_probe():\n    return 1\n", "def make_flywheel_probe():\n    return 2\n"),
        ):
            relative, path = self.stage_candidate(
                request_id, f"{candidate_id}.py", body
            )
            if read_only:
                path.chmod(0o444)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "module_path": relative,
                    "sha256": self.candidate_sha(path),
                }
            )
        payload = {
            "schema": "tilefoundry-fast-request/v1",
            "mode": "micro_probe",
            "request_id": request_id,
            "response_path": f"build/agent_gates/fast/{request_id}.response.json",
            "warmup": 1,
            "repeat": 1,
            "require_all_passed": False,
            "probe": probe
            if probe is not None
            else {
                "probe_id": PROBE_ID,
                "probe_kind": "stage_a",
                "factory_name": "make_flywheel_probe",
                "line_count_baseline": 12,
                "line_count_optimized": 9,
                "tensor_shape": [16, 32],
                "tensor_dtype": "float16",
                "min_speedup": 1.08,
            },
            "candidates": candidates,
        }
        self.probe_candidates = candidates
        return payload

    def write_request(self, request_id: str, payload: dict) -> str:
        relative = f"build/agent_gates/fast/{request_id}.request.json"
        path = self.src / relative
        if path.exists():
            path.chmod(0o644)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        path.chmod(0o444)
        return relative

    def run_fast(self, *, env=None, extra_env=None):
        run_env = dict(self.environment() if env is None else env)
        run_env.update(extra_env or {})
        free_processes = {
            "ok": True,
            "processTableSeen": True,
            "processes": [],
            "error": None,
        }
        with self._in_source(), \
                patch.dict(os.environ, run_env, clear=True), \
                patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 1, "error": None},
                ), \
                patch.object(
                    GATE, "_probe_device_health",
                    return_value={"ok": True, "healthKind": "ok", "healthValue": "OK", "error": None},
                ), \
                patch.object(GATE, "_probe_device_processes", return_value=free_processes), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            return GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_manifest_declares_optional_dynamic_fast_gate(self):
        raw = json.loads((ADAPTER / "manifest.json").read_text(encoding="utf-8"))
        recipes = {item["id"]: item for item in raw["recipes"]}
        self.assertEqual(
            recipes["tilefoundry-npu.device-fast"]["dynamicTarget"], "optional"
        )

    def test_cpu_defaults_include_optional_suites_only_when_present(self):
        shutil.rmtree(self.src / "tests" / "ascend" / "agent")
        shutil.rmtree(self.src / "tests" / "ascend" / "ir")
        self.assertEqual(
            GATE.default_cpu_tests(self.src),
            GATE.DEFAULT_CPU_TESTS,
        )
        (self.src / "tests" / "ascend" / "cost_model").mkdir(parents=True)
        (self.src / "tests" / "ascend" / "agent").mkdir(parents=True)
        (self.src / "tests" / "ascend" / "sil").mkdir(parents=True)
        (self.src / "tests" / "ascend" / "benchmarks").mkdir(parents=True)
        (self.src / "tests" / "ascend" / "benchmarks" / "test_l3_contract_host.py").write_text(
            "def test_host():\n    assert True\n", encoding="utf-8"
        )
        tests = GATE.default_cpu_tests(self.src)
        self.assertNotIn("tests/ascend/ir", tests)
        for rel in (
            "tests/ascend/cost_model",
            "tests/ascend/agent",
            "tests/ascend/sil",
            "tests/ascend/benchmarks/test_l3_contract_host.py",
        ):
            self.assertIn(rel, tests)
        (self.src / "tests" / "ascend" / "ir").mkdir(parents=True)
        self.assertIn("tests/ascend/ir", GATE.default_cpu_tests(self.src))
        self.assertEqual(
            GATE.cpu_tests({"TILEFOUNDRY_NPU_CPU_TESTS": "tests/device"}, self.src),
            ("tests/device",),
        )
        self.assertEqual(GATE.cpu_deselects(self.src), ())
        for node in GATE.CPU_DEVICE_NODE_DESELECTS:
            path = self.src / node.split("::", 1)[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
        self.assertEqual(GATE.cpu_deselects(self.src), GATE.CPU_DEVICE_NODE_DESELECTS)

    def test_batch_request_binds_candidates_and_response_artifacts(self):
        request_relative = self.write_request(REQUEST_ID, self.batch_request())
        with self._record_host_call() as calls:
            rc, obs = self.run_fast()
        self.assertEqual(rc, 0)
        self.assertTrue(obs["exitCodeZero"])
        self.assertEqual(obs["fastRequest"]["path"], request_relative)
        self.assertEqual(obs["fastRequest"]["requestId"], REQUEST_ID)
        self.assertTrue(obs["requestStable"])
        self.assertTrue(obs["fastResponse"]["bound"])
        self.assertEqual(obs["fastResponse"]["status"], "ok")
        self.assertGreater(obs["fastResponse"]["bytes"], 0)
        self.assertEqual(
            [row["candidateId"] for row in obs["fastCandidates"]],
            ["cand_0", "cand_1"],
        )
        self.assertTrue(all(row["stable"] for row in obs["fastCandidates"]))
        self.assertEqual(
            [row["postSha256"] for row in obs["fastCandidates"]],
            [row["preSha256"] for row in obs["fastCandidates"]],
        )
        self.assertEqual(obs["fastCoverage"]["expected_candidates"], 2)
        self.assertIsNone(obs["failureCategory"])
        self.assertEqual(obs["errorCategory"], "none")
        self.assertEqual(len(calls), 1)
        self.assertIn(GATE.FAST_DEVICE_TEST_NODE, calls[0][0])
        self.assertNotIn("tests/device/test_fast.py", calls[0][0])
        child_env = calls[0][1]
        self.assertNotEqual(child_env["TF_FAST_REQUEST"], str(self.src / request_relative))
        self.assertEqual(child_env["TF_FAST_SOURCE_ROOT"], str(self.src.resolve()))
        self.assertEqual(
            child_env["TILEFOUNDRY_NPU_CANNBENCH_REPO"], str(self.cannbench.resolve())
        )
        self.assertEqual(
            child_env["TILELANG_CACHE_DIR"], str((self.work / "surrogate" / "tilelang").resolve())
        )
        self.assertEqual(
            child_env["CANNBENCH_CACHE_DIR"], str((self.work / "surrogate" / "reference").resolve())
        )
        self.assertNotIn("TF_SURROGATE_BATCH_REQUEST", child_env)
        self.assertNotIn("TF_MICRO_PROBE_REQUEST", child_env)
        artifact_paths = {row["path"] for row in obs["artifacts"]}
        self.assertIn(str((self.src / request_relative).resolve()), artifact_paths)
        self.assertIn(str(self.candidate_a.resolve()), artifact_paths)
        self.assertIn(str(self.candidate_b.resolve()), artifact_paths)
        self.assertIn(
            str((self.src / "build/agent_gates/fast" / f"{REQUEST_ID}.response.json").resolve()),
            artifact_paths,
        )
        self.assertFalse((self.lock_root / ".msprof.lock").exists())

    def test_typed_fast_ignores_machine_device_tests_for_both_modes(self):
        for device_tests in ("", "tests/device/test_fast.py"):
            for request_id in (REQUEST_ID, PROFILE_ID):
                response = self.src / f"build/agent_gates/fast/{request_id}.response.json"
                response.unlink(missing_ok=True)
            self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS=device_tests)
            self.write_test_file(BATCH_RESPONSE)
            self.write_request(REQUEST_ID, self.batch_request())
            with self._record_host_call() as calls:
                rc, obs = self.run_fast()
            self.assertEqual(rc, 0)
            self.assertTrue(obs["fastResponse"]["bound"])
            self.assertIn(GATE.FAST_DEVICE_TEST_NODE, calls[0][0])
            self.assertNotIn("tests/device/test_fast.py", calls[0][0])

            self.write_test_file(PROFILE_RESPONSE)
            self.write_request(PROFILE_ID, self.profile_request())
            with self._record_host_call() as calls:
                rc, obs = self.run_fast(env=self.environment(PROFILE_ID))
            self.assertEqual(rc, 0)
            self.assertTrue(obs["fastResponse"]["bound"])
            self.assertIn(GATE.FAST_DEVICE_TEST_NODE, calls[0][0])
            self.assertNotIn("tests/device/test_fast.py", calls[0][0])

    def test_check_config_device_accepts_typed_fast_without_device_tests(self):
        requests = {
            REQUEST_ID: self.batch_request(),
            PROFILE_ID: self.profile_request(),
        }
        for request_id, payload in requests.items():
            self.write_request(request_id, payload)
        for device_tests in ("", "tests/device/test_fast.py"):
            self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS=device_tests)
            for request_id in requests:
                with self._in_source(), \
                        patch.dict(os.environ, self.environment(request_id), clear=True), \
                        patch.object(GATE.adapter_runtime, "check_device_ready", return_value=None):
                    self.assertEqual(GATE.main(["--check-config-device"]), 0)

    def test_check_config_device_requires_the_fixed_harness_file(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="")
        self.write_request(REQUEST_ID, self.batch_request())
        (self.src / GATE.FAST_DEVICE_TEST_FILE).unlink()
        with self._in_source(), \
                patch.dict(os.environ, self.environment(REQUEST_ID), clear=True):
            with self.assertRaises(SystemExit) as caught:
                GATE.main(["--check-config-device"])
        self.assertEqual(caught.exception.code, 75)

    def stage_canonical_softmax(self) -> None:
        task_dir = self.cannbench / "tasks" / "level2" / "softmax"
        task_dir.mkdir(parents=True)
        for name in ("proto.yaml", "cases.yaml"):
            (task_dir / name).write_text("# cases\n", encoding="utf-8")
        (task_dir / "golden.py").write_text("# golden\n", encoding="utf-8")
        candidate = self.src / "examples" / "ascend950pr_ops" / "softmax" / "kernels.py"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("# candidate\n", encoding="utf-8")

    def canonical_environment(self) -> dict[str, str]:
        env = self.environment()
        env["AGENT_GATES_TARGET"] = "level2/softmax"
        env["AGENT_GATES_CANDIDATE"] = "examples/ascend950pr_ops/softmax/kernels.py"
        return env

    def blocked_config_reason(self, argv: list[str]) -> str:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                GATE.main(argv)
        self.assertEqual(caught.exception.code, 75)
        return stderr.getvalue()

    def test_canonical_l3_config_accepts_the_canonical_pair(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="")
        self.stage_canonical_softmax()
        with self._in_source(), \
                patch.dict(os.environ, self.canonical_environment(), clear=True), \
                patch.object(GATE.adapter_runtime, "check_device_ready", return_value=None):
            self.assertEqual(GATE.main(["--check-config-l3"]), 0)

    def test_canonical_cannbench_config_accepts_the_canonical_pair(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="")
        self.stage_canonical_softmax()
        with self._in_source(), \
                patch.dict(os.environ, self.canonical_environment(), clear=True), \
                patch.object(GATE.adapter_runtime, "check_device_ready", return_value=None):
            self.assertEqual(GATE.main(["--check-config-cannbench"]), 0)

    def test_canonical_l3_config_rejects_a_fast_pair(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="")
        self.write_request(REQUEST_ID, self.batch_request())
        with self._in_source(), \
                patch.dict(os.environ, self.environment(REQUEST_ID), clear=True), \
                patch.object(GATE.adapter_runtime, "check_device_ready", return_value=None):
            reason = self.blocked_config_reason(["--check-config-l3"])
        self.assertIn("invalid CANNBench task 'fast/", reason)
        self.assertNotIn("must be fast/", reason)

    def test_canonical_cannbench_config_rejects_a_fast_pair(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="")
        self.write_request(REQUEST_ID, self.batch_request())
        with self._in_source(), \
                patch.dict(os.environ, self.environment(REQUEST_ID), clear=True), \
                patch.object(GATE.adapter_runtime, "check_device_ready", return_value=None):
            reason = self.blocked_config_reason(["--check-config-cannbench"])
        self.assertIn("invalid CANNBench task 'fast/", reason)
        self.assertNotIn("must be fast/", reason)

    def test_device_fast_config_rejects_a_canonical_pair(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="")
        self.stage_canonical_softmax()
        with self._in_source(), \
                patch.dict(os.environ, self.canonical_environment(), clear=True), \
                patch.object(GATE.adapter_runtime, "check_device_ready", return_value=None):
            reason = self.blocked_config_reason(["--check-config-device"])
        self.assertIn("must be fast/<32 lowercase hex>", reason)

    def test_legacy_device_config_without_a_pair_still_passes(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="")
        env = self.environment()
        del env["AGENT_GATES_TARGET"]
        del env["AGENT_GATES_CANDIDATE"]
        with self._in_source(), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(GATE.adapter_runtime, "check_device_ready", return_value=None):
            self.assertEqual(GATE.main(["--check-config-device"]), 0)

    def test_manifest_keeps_canonical_gates_on_their_own_config_checks(self):
        raw = json.loads((ADAPTER / "manifest.json").read_text(encoding="utf-8"))
        checks = {recipe["id"]: recipe["configCheck"][-1] for recipe in raw["recipes"]}
        self.assertEqual(checks["tilefoundry-npu.l3-promotion"], "--check-config-l3")
        self.assertEqual(checks["tilefoundry-npu.cannbench-prep"], "--check-config-cannbench")
        self.assertEqual(checks["tilefoundry-npu.cannbench"], "--check-config-cannbench")
        self.assertEqual(checks["tilefoundry-npu.device"], "--check-config-device")
        self.assertEqual(checks["tilefoundry-npu.device-fast"], "--check-config-device")

    def test_device_fast_requires_the_fixed_harness_file(self):
        self.write_env(TILEFOUNDRY_NPU_DEVICE_TESTS="tests/device/test_fast.py")
        self.write_request(REQUEST_ID, self.batch_request())
        (self.src / GATE.FAST_DEVICE_TEST_FILE).unlink()
        with self._in_source(), \
                patch.dict(os.environ, self.environment(REQUEST_ID), clear=True):
            with self.assertRaises(SystemExit) as caught:
                GATE.main(["--device-fast", "--timeout", "5"])
        self.assertEqual(caught.exception.code, 75)

    def test_profile_request_holds_msprof_lock_without_machine_flag(self):
        self.write_test_file(PROFILE_RESPONSE)
        self.write_request(PROFILE_ID, self.profile_request())
        events = []
        original_lock = GATE._open_msprof_lock
        original_window = GATE._open_device_window

        @contextmanager
        def lock(*args, **kwargs):
            events.append("msprof")
            with original_lock(*args, **kwargs):
                yield

        @contextmanager
        def window(*_args, **_kwargs):
            events.append("window")
            with original_window(*_args, **_kwargs) as finalize:
                yield finalize

        with patch.object(GATE, "_open_msprof_lock", lock), \
                patch.object(GATE, "_open_device_window", window):
            rc, obs = self.run_fast(env=self.environment(PROFILE_ID))
        self.assertEqual(rc, 0)
        self.assertEqual(events[:2], ["msprof", "window"])
        self.assertTrue(obs["fastResponse"]["bound"])
        self.assertEqual(obs["fastCoverage"]["expected_msprof_targets"], 1)
        self.assertTrue((self.lock_root / ".msprof.lock").is_file())

    def test_micro_probe_binds_staged_candidates_and_returns_metrics(self):
        self.write_test_file(MICRO_PROBE_RESPONSE)
        self.write_request(PROBE_REQUEST_ID, self.micro_probe_request())
        with self._record_host_call() as calls:
            rc, obs = self.run_fast(env=self.environment(PROBE_REQUEST_ID))
        self.assertEqual(rc, 0)
        self.assertEqual(obs["fastRequest"]["mode"], "micro_probe")
        self.assertTrue(obs["fastResponse"]["bound"])
        self.assertEqual(obs["fastResponse"]["status"], "ok")
        self.assertEqual(obs["fastCoverage"]["expected_candidates"], 2)
        self.assertEqual(obs["fastCoverage"]["evaluated_candidates"], 2)
        self.assertEqual(obs["fastCoverage"]["expected_msprof_targets"], 0)
        self.assertEqual(
            [row["candidateId"] for row in obs["fastCandidates"]],
            ["baseline", "optimized"],
        )
        self.assertTrue(all(row["stable"] for row in obs["fastCandidates"]))
        self.assertNotIn("TF_MICRO_PROBE_REQUEST", calls[0][1])
        self.assertTrue((self.lock_root / ".msprof.lock").is_file())

    def test_micro_probe_rejects_writable_candidate_modules(self):
        self.write_request(
            PROBE_REQUEST_ID, self.micro_probe_request(read_only=False)
        )
        with self._in_source(), \
                patch.dict(os.environ, self.environment(PROBE_REQUEST_ID), clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError) as caught:
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)
        self.assertIn("must be read-only", str(caught.exception))

    def test_micro_probe_requires_both_probe_candidates(self):
        missing = self.micro_probe_request()
        missing["candidates"] = missing["candidates"][:1]
        duplicate = self.micro_probe_request()
        duplicate["candidates"][1]["candidate_id"] = "baseline"
        for label, payload in (("missing", missing), ("duplicate", duplicate)):
            with self.subTest(case=label):
                self.write_request(PROBE_REQUEST_ID, payload)
                with self._in_source(), \
                        patch.dict(
                            os.environ,
                            self.environment(PROBE_REQUEST_ID),
                            clear=True,
                        ), \
                        patch.object(
                            GATE.locking, "lock_root", return_value=self.lock_root
                        ):
                    with self.assertRaises(GATE.BridgeError):
                        GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_micro_probe_rejects_invalid_probe_fields(self):
        valid = self.micro_probe_request()["probe"]
        cases = {
            "factory_name": {**valid, "factory_name": "not a name"},
            "tensor_shape": {**valid, "tensor_shape": []},
            "min_speedup": {**valid, "min_speedup": 0},
            "extra_key": {**valid, "extra": 1},
        }
        for label, probe in cases.items():
            with self.subTest(field=label):
                self.write_request(
                    PROBE_REQUEST_ID, self.micro_probe_request(probe=probe)
                )
                with self._in_source(), \
                        patch.dict(
                            os.environ,
                            self.environment(PROBE_REQUEST_ID),
                            clear=True,
                        ), \
                        patch.object(
                            GATE.locking, "lock_root", return_value=self.lock_root
                        ):
                    with self.assertRaises(GATE.BridgeError):
                        GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_micro_probe_response_requires_core_metrics(self):
        self.write_test_file(MICRO_PROBE_MISSING_METRIC_RESPONSE)
        self.write_request(PROBE_REQUEST_ID, self.micro_probe_request())
        rc, obs = self.run_fast(env=self.environment(PROBE_REQUEST_ID))
        self.assertEqual(rc, 1)
        self.assertFalse(obs["fastResponse"]["bound"])
        self.assertIn(
            "FAST_RESPONSE_MICRO_PROBE_METRIC_MISSING:speedup",
            obs["fastBindingFailures"],
        )
        self.assertEqual(obs["failureCategory"], "infra")

    def test_micro_probe_response_probe_id_must_match(self):
        self.write_test_file(MICRO_PROBE_WRONG_ID_RESPONSE)
        self.write_request(PROBE_REQUEST_ID, self.micro_probe_request())
        rc, obs = self.run_fast(env=self.environment(PROBE_REQUEST_ID))
        self.assertEqual(rc, 1)
        self.assertIn("FAST_RESPONSE_MICRO_PROBE_ID", obs["fastBindingFailures"])

    def test_candidate_mutation_during_run_is_infrastructure(self):
        self.write_request(REQUEST_ID, self.batch_request())
        real = GATE.adapter_runtime.run_host_bash

        def mutating(cmd, **kwargs):
            result = real(cmd, **kwargs)
            self.candidate_a.write_text("def entry():\n    return 99\n", encoding="utf-8")
            return result

        with patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=mutating):
            rc, obs = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertFalse(obs["exitCodeZero"])
        self.assertIn("FAST_CANDIDATE_CHANGED:unit_op:cand_0", obs["fastBindingFailures"])
        self.assertFalse(obs["fastCandidates"][0]["stable"])
        self.assertEqual(obs["failureCategory"], "infra")

    def test_request_mutation_during_run_is_infrastructure(self):
        request_relative = self.write_request(REQUEST_ID, self.batch_request())
        request_path = self.src / request_relative
        request_path.chmod(0o644)
        real = GATE.adapter_runtime.run_host_bash

        def mutating(cmd, **kwargs):
            result = real(cmd, **kwargs)
            request_path.chmod(0o644)
            request_path.write_text(
                request_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            return result

        with patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=mutating):
            rc, obs = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertFalse(obs["requestStable"])
        self.assertIn("REQUEST_IDENTITY_CHANGED", obs["fastBindingFailures"])
        self.assertEqual(obs["failureCategory"], "infra")

    def test_response_coverage_mismatch_fails_binding(self):
        self.write_test_file(EXTRA_CANDIDATE_RESPONSE)
        self.write_request(REQUEST_ID, self.batch_request())
        rc, obs = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertFalse(obs["fastResponse"]["bound"])
        self.assertIn(
            "FAST_RESPONSE_CANDIDATE_EXTRA:unit_op:unexpected",
            obs["fastBindingFailures"],
        )

    def test_coverage_counts_must_match_declared_candidates(self):
        self.write_test_file(COVERAGE_MISMATCH_RESPONSE)
        self.write_request(REQUEST_ID, self.batch_request())
        rc, obs = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertIn(
            "FAST_RESPONSE_COVERAGE_EXPECTED_CANDIDATES", obs["fastBindingFailures"]
        )
        self.assertEqual(obs["failureCategory"], "infra")

    def test_missing_response_is_infrastructure(self):
        self.write_test_file("def test_surrogate_anchor_batch_npu() -> None:\n    assert True\n")
        self.write_request(REQUEST_ID, self.batch_request())
        rc, obs = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertFalse(obs["fastResponse"]["bound"])
        self.assertTrue(
            any(item.startswith("FAST_RESPONSE_UNREADABLE:") for item in obs["fastBindingFailures"])
        )
        self.assertEqual(obs["failureCategory"], "infra")

    def test_request_error_status_is_infrastructure_not_candidate(self):
        self.write_test_file(REQUEST_ERROR_RESPONSE)
        self.write_request(REQUEST_ID, self.batch_request())
        rc, obs = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertIn("FAST_RESPONSE_STATUS:request_error", obs["fastBindingFailures"])
        self.assertEqual(obs["failureCategory"], "infra")

    def test_infrastructure_status_is_infrastructure(self):
        self.write_test_file(INFRASTRUCTURE_RESPONSE)
        self.write_request(REQUEST_ID, self.batch_request())
        rc, obs = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertIn(
            "FAST_RESPONSE_STATUS:infrastructure_error", obs["fastBindingFailures"]
        )
        self.assertEqual(obs["failureCategory"], "infra")

    def test_candidate_failure_status_with_failed_run_is_candidate(self):
        self.write_test_file(CANDIDATE_FAILURE_RESPONSE)
        self.write_request(
            REQUEST_ID, self.batch_request(require_all_passed=True)
        )
        rc, obs = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertTrue(obs["fastResponse"]["bound"])
        self.assertEqual(obs["fastResponse"]["status"], "candidate_failure")
        self.assertEqual(obs["failureCategory"], "candidate")

    def test_candidate_failure_status_with_passing_run_is_not_a_block(self):
        self.write_test_file(CANDIDATE_FAILURE_RESPONSE.replace(
            '    raise AssertionError("fast batch candidate(s) failed")\n', ""
        ))
        self.write_request(REQUEST_ID, self.batch_request())
        rc, obs = self.run_fast()
        self.assertEqual(rc, 0)
        self.assertTrue(obs["fastResponse"]["bound"])
        self.assertIsNone(obs["failureCategory"])

    def test_invalid_candidate_sha_blocks_before_the_device_window(self):
        request_id = "11111111111111111111111111111111"
        self.write_request(request_id, self.batch_request(request_id=request_id, sha_a="0" * 64))
        window = MagicMock()
        window.return_value = nullcontext(lambda: ("free", {}))

        with patch.object(GATE, "_open_device_window", window), \
                self._in_source(), \
                patch.dict(os.environ, self.environment(request_id), clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError):
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)
        self.assertFalse(window.called)

    def test_check_config_rejects_invalid_request_with_blocked_exit(self):
        request_id = "22222222222222222222222222222222"
        self.write_request(request_id, self.batch_request(request_id=request_id, sha_b="0" * 64))
        with self._in_source(), \
                patch.dict(os.environ, self.environment(request_id), clear=True):
            with self.assertRaises(SystemExit) as caught:
                GATE.main(["--check-config-device"])
        self.assertEqual(caught.exception.code, 75)

    def test_request_with_retired_machine_keys_is_rejected(self):
        request_id = "33333333333333333333333333333333"
        payload = self.batch_request(
            request_id=request_id,
            extra={
                "source_root": str(self.src),
                "cannbench_repo": str(self.cannbench),
                "tilelang_cache_dir": str(self.root / "tl"),
            },
        )
        self.write_request(request_id, payload)
        with self._in_source(), \
                patch.dict(os.environ, self.environment(request_id), clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError):
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_candidate_outside_the_staging_directory_is_rejected(self):
        request_id = "44444444444444444444444444444444"
        outside = self.src / "candidates" / "elsewhere.py"
        outside.parent.mkdir(parents=True)
        outside.write_text("def entry():\n    return 1\n", encoding="utf-8")
        payload = self.batch_request(
            request_id=request_id,
            module_a="candidates/elsewhere.py",
            sha_a=self.candidate_sha(outside),
        )
        self.write_request(request_id, payload)
        with self._in_source(), \
                patch.dict(os.environ, self.environment(request_id), clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError):
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_response_path_must_match_the_derived_path(self):
        request_id = "55555555555555555555555555555555"
        payload = self.batch_request(
            request_id=request_id,
            response_path=f"build/agent_gates/fast/{request_id}/other.response.json",
        )
        self.write_request(request_id, payload)
        with self._in_source(), \
                patch.dict(os.environ, self.environment(request_id), clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError):
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_target_must_be_the_fast_request_id(self):
        request_id = "66666666666666666666666666666666"
        self.write_request(request_id, self.batch_request(request_id=request_id))
        env = self.environment(request_id)
        env["AGENT_GATES_TARGET"] = "level4/unit_op"
        with self._in_source(), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError):
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_candidate_must_be_the_derived_request_path(self):
        request_id = "77777777777777777777777777777777"
        self.write_request(request_id, self.batch_request(request_id=request_id))
        env = self.environment(request_id)
        env["AGENT_GATES_CANDIDATE"] = "candidates/cand_0.py"
        with self._in_source(), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError):
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_partial_dynamic_pair_is_refused(self):
        request_id = "88888888888888888888888888888888"
        self.write_request(request_id, self.batch_request(request_id=request_id))
        env = self.environment(request_id)
        del env["AGENT_GATES_CANDIDATE"]
        with self._in_source(), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError):
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_symlinked_fast_directory_is_refused(self):
        request_id = "99999999999999999999999999999999"
        external = self.root / "external-fast"
        external.mkdir()
        fast_dir = self.src / "build" / "agent_gates" / "fast"
        fast_dir.rmdir()
        fast_dir.symlink_to(external, target_is_directory=True)
        payload = self.batch_request(request_id=request_id)
        self.write_request(request_id, payload)
        with self._in_source(), \
                patch.dict(os.environ, self.environment(request_id), clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError):
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_parent_environment_request_channel_is_refused(self):
        self.write_test_file("def test_surrogate_anchor_batch_npu() -> None:\n    assert True\n")
        env = self.environment()
        del env["AGENT_GATES_TARGET"]
        del env["AGENT_GATES_CANDIDATE"]
        env["TF_FAST_REQUEST"] = str(self.root / "request.json")
        with self._in_source(), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            with self.assertRaises(GATE.BridgeError):
                GATE.run_device(GATE.machine_values(), 60.0, fast=True)

    def test_legacy_fixed_mode_remains_accepted(self):
        (self.src / "tests" / "device" / "test_fast.py").write_text(
            "def test_legacy():\n    assert True\n", encoding="utf-8"
        )
        legacy = self.root / "legacy-request.json"
        legacy.write_text("{}", encoding="utf-8")
        env = self.environment()
        del env["AGENT_GATES_TARGET"]
        del env["AGENT_GATES_CANDIDATE"]
        legacy_env = {
            "TF_SURROGATE_BATCH_REQUEST": str(legacy),
            "TF_MICRO_PROBE_REQUEST": str(legacy),
        }
        with self._record_host_call() as calls:
            rc, obs = self.run_fast(env=env, extra_env=legacy_env)
        self.assertEqual(rc, 0)
        self.assertNotIn("fastRequest", obs)
        self.assertNotIn("fastBindingFailures", obs)
        self.assertEqual(calls[0][1]["TF_SURROGATE_BATCH_REQUEST"], str(legacy))
        self.assertEqual(calls[0][1]["TF_MICRO_PROBE_REQUEST"], str(legacy))
        self.assertFalse((self.lock_root / ".msprof.lock").exists())

    def test_device_fast_blocks_preoccupied_card_and_outputs_occupancy_json(self):
        import io
        from contextlib import redirect_stdout

        self.write_request(REQUEST_ID, self.batch_request())
        env = self.environment(REQUEST_ID)
        env["TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE"] = "0"
        stdout = io.StringIO()
        with self._in_source(), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(
                    GATE.adapter_runtime,
                    "processes_on",
                    return_value=([{"pid": "4321", "name": "python", "memoryMb": "32768"}], 1),
                ), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root), \
                redirect_stdout(stdout):
            rc = GATE.main(["--device-fast", "--timeout", "30"])
        self.assertEqual(rc, 75)
        obs = json.loads(stdout.getvalue().strip())
        self.assertEqual(obs["errorCategory"], "occupancy")
        self.assertEqual(obs["failureCategory"], "occupancy")
        self.assertEqual(obs["deviceOccupancy"], "shared(1)")
        self.assertTrue(obs["occupancyRecorded"])
        self.assertFalse(obs["windowOpened"])

    def test_device_fast_window_intrusion_outputs_occupancy_json_and_exits_75(self):
        import io
        from contextlib import redirect_stdout

        self.write_request(REQUEST_ID, self.batch_request())
        env = self.environment(REQUEST_ID)

        def intrusion_abort(_cmd, **kwargs):
            kwargs["monitor_sink"].update({
                "abort": {
                    "kind": "external_occupancy",
                    "reason": "FAST lost exclusivity after the lease opened",
                    "phase": None,
                    "device": "1",
                    "npuId": 1,
                    "occupancy": "shared(1)",
                },
                "outcome": "aborted",
            })
            raise GATE.BridgeError(
                "runtime health abort (external_occupancy)",
                category="occupancy",
                occupancy="shared(1)",
            )

        stdout = io.StringIO()
        with self._in_source(), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                patch.object(GATE, "_run_host_bash_monitored", side_effect=intrusion_abort), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root), \
                redirect_stdout(stdout):
            rc = GATE.main(["--device-fast", "--timeout", "30"])
        self.assertEqual(rc, 75)
        obs = json.loads(stdout.getvalue().strip())
        self.assertEqual(obs["errorCategory"], "occupancy")
        self.assertEqual(obs["failureCategory"], "occupancy")
        self.assertEqual(obs["deviceOccupancy"], "shared(1)")
        self.assertTrue(obs["windowOpened"])
        self.assertTrue(obs["windowReleased"])
        self.assertEqual(obs["healthMonitoring"]["abort"]["kind"], "external_occupancy")

    def test_device_fast_classifies_npu_device_fault_signatures_as_blocked_env_75(self):
        cases = (
            ("E39006", "stdout", 'print("EE9999 E39006: Inner_Error_Failed_Send_OPP_To_Device")'),
            ("TsdOpen failed", "stderr", 'import sys; print("TsdOpen failed, devId=1", file=sys.stderr)'),
            ("507033", "stdout", 'print("aclrtSynchronizeStream failed, ret=507033")'),
        )
        for sig, stream_name, emit in cases:
            with self.subTest(sig=sig, stream=stream_name):
                (self.src / f"build/agent_gates/fast/{REQUEST_ID}.response.json").unlink(
                    missing_ok=True
                )
                self.write_test_file(
                    f"def test_surrogate_anchor_batch_npu() -> None:\n"
                    f"    {emit}\n"
                    f"    raise RuntimeError('npu runtime failed')\n"
                )
                self.write_request(REQUEST_ID, self.batch_request())
                rc, obs = self.run_fast()
                self.assertEqual(rc, 75)
                self.assertFalse(obs["exitCodeZero"])
                self.assertTrue(obs["windowOpened"])
                self.assertEqual(obs["errorCategory"], "device_fault")
                self.assertEqual(obs["failureCategory"], "device_fault")
                self.assertIs(obs["deviceRejected"], True)
                self.assertEqual(obs["deviceFaultSignature"], sig)

    def test_device_fast_keeps_aicore_trap_and_assertion_error_as_candidate(self):
        for exc_expr in ('RuntimeError("AICore trap")', 'AssertionError("numeric mismatch")'):
            with self.subTest(exc=exc_expr):
                (self.src / f"build/agent_gates/fast/{REQUEST_ID}.response.json").unlink(
                    missing_ok=True
                )
                self.write_test_file(
                    CANDIDATE_FAILURE_RESPONSE.replace(
                        'raise AssertionError("fast batch candidate(s) failed")',
                        f"raise {exc_expr}",
                    )
                )
                self.write_request(
                    REQUEST_ID, self.batch_request(require_all_passed=True)
                )
                rc, obs = self.run_fast()
                self.assertEqual(rc, 1)
                self.assertFalse(obs["exitCodeZero"])
                self.assertEqual(obs["errorCategory"], "candidate")
                self.assertEqual(obs["failureCategory"], "candidate")
                self.assertNotIn("deviceRejected", obs)
                self.assertNotIn("deviceFaultSignature", obs)

    def test_batch_accepts_optional_positive_poison_pill_timeout_sec_and_rejects_invalid(self):
        payload = self.batch_request()
        payload["batches"][0]["poison_pill_timeout_sec"] = 12.5
        self.write_request(REQUEST_ID, payload)
        with self._in_source(), \
                patch.dict(os.environ, self.environment(REQUEST_ID), clear=True), \
                patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
            loaded = GATE._load_fast_request(
                self.src,
                GATE.machine_values(),
                (f"fast/{REQUEST_ID}", f"build/agent_gates/fast/{REQUEST_ID}.request.json"),
            )
            self.assertIsNotNone(loaded)

        for bad_val in (0, -5.0, True, "10"):
            with self.subTest(bad_val=bad_val):
                bad_payload = self.batch_request()
                bad_payload["batches"][0]["poison_pill_timeout_sec"] = bad_val
                self.write_request(REQUEST_ID, bad_payload)
                with self._in_source(), \
                        patch.dict(os.environ, self.environment(REQUEST_ID), clear=True), \
                        patch.object(GATE.locking, "lock_root", return_value=self.lock_root):
                    with self.assertRaises(GATE.BridgeError):
                        GATE._load_fast_request(
                            self.src,
                            GATE.machine_values(),
                            (f"fast/{REQUEST_ID}", f"build/agent_gates/fast/{REQUEST_ID}.request.json"),
                        )

    @contextmanager
    def _record_host_call(self):
        real = GATE.adapter_runtime.run_host_bash
        calls = []

        def recording(cmd, **kwargs):
            calls.append((cmd, dict(kwargs["env"])))
            return real(cmd, **kwargs)

        with patch.object(GATE.adapter_runtime, "run_host_bash", side_effect=recording):
            yield calls


if __name__ == "__main__":
    unittest.main()
