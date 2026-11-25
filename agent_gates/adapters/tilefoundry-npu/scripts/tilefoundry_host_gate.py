#!/usr/bin/env python3
"""Host-local TileFoundry CPU, Ascend NPU, and CANNBench verification gates."""
from __future__ import annotations

import ast
import base64
import concurrent.futures
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, NamedTuple, NoReturn
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from agent_gates import adapter_runtime, locking

PROG = "tilefoundry-npu-gate"
ADAPTER_DIR = Path(__file__).resolve().parent.parent
LLM_API_KEY_ENV = "TILEFOUNDRY_NPU_LLM_API_KEY_ENV"

OPTIONAL_AGENT_LLM_KEYS = (
    "TILEFOUNDRY_NPU_LLM_BACKEND",
    "TILEFOUNDRY_NPU_LLM_API_BASE",
    LLM_API_KEY_ENV,
    "TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL",
    "TILEFOUNDRY_NPU_LLM_EXPLORE_MODEL",
    "TILEFOUNDRY_NPU_LLM_ESCALATE_MODEL",
    "TILEFOUNDRY_NPU_LLM_REPAIR_MODEL",
    "TILEFOUNDRY_NPU_LLM_FALLBACK_MODEL",
    "TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES",
    "TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC",
    "TILEFOUNDRY_NPU_LLM_MAX_EFFORT",
    "TILEFOUNDRY_NPU_LLM_EFFORT_CAP",
    "TILEFOUNDRY_NPU_LLM_PASS_REASONING_EFFORT",
    "TILEFOUNDRY_NPU_LLM_STREAM",
    "TILEFOUNDRY_NPU_LLM_MAX_CONCURRENCY",
    "TILEFOUNDRY_NPU_LLM_EXPLOIT_TIMEOUT_SEC",
    "TILEFOUNDRY_NPU_LLM_REASONING_TIMEOUT_SEC",
    "TILEFOUNDRY_NPU_LLM_STARTUP_TIMEOUT_SEC",
    "TILEFOUNDRY_NPU_LLM_IDLE_STALL_TIMEOUT_SEC",
    "TILEFOUNDRY_NPU_LLM_CLI_BIN",
    "TILEFOUNDRY_NPU_LLM_CLI_TEMPLATE",
    "TILEFOUNDRY_NPU_AGY_BIN",
    "TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL",
    "TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL",
    "TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL",
    "TILEFOUNDRY_NPU_AGY_COMMON_ARGS",
    "TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY",
)

VALID_CANN_SERIES = ("cann_9_2", "cann_9_3")

PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
OPTIONAL_PROXY_KEYS = tuple(f"TILEFOUNDRY_NPU_LLM_{key}" for key in PROXY_KEYS)

TOOLCHAIN_KEYS = (
    "TILEFOUNDRY_NPU_CANN_SETUP",
    "TILEFOUNDRY_NPU_CONDA_SH",
    "TILEFOUNDRY_NPU_CONDA_ENV",
    "TILEFOUNDRY_NPU_CONDA_ENVS_PATH",
    "TILEFOUNDRY_NPU_PYTHON",
    "TILEFOUNDRY_NPU_TILELANG_SOURCE",
    "TILEFOUNDRY_NPU_TILELANG_BUILD",
    "TILEFOUNDRY_NPU_CPU_TESTS",
    "TILEFOUNDRY_NPU_CPU_WORKERS",
    "TILEFOUNDRY_NPU_DEVICE_TESTS",
    "TILEFOUNDRY_NPU_MSPROF_LOCK",
    "TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE",
    "TILEFOUNDRY_NPU_L3_TIMEOUT_SEC",
    "TILEFOUNDRY_NPU_BUILD_SUBDIR",
    "TILEFOUNDRY_NPU_CANNBENCH_REPO",
    "TILEFOUNDRY_NPU_WORK_ROOT",
    "TILEFOUNDRY_NPU_CANNBENCH_ARCH",
    "TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE",
    "TILEFOUNDRY_NPU_CANNBENCH_TARGETS",
    "TILEFOUNDRY_NPU_CANNBENCH_MCP_URL",
    "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN",
    "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE",
    "TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION",
    "TILEFOUNDRY_NPU_CANNBENCH_POOL_ID",
    "TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN",
    "TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT",
    *OPTIONAL_AGENT_LLM_KEYS,
    *OPTIONAL_PROXY_KEYS,
    "TILEFOUNDRY_NPU_L4_EXTERNAL_PROBE_LIMIT",
    "TILEFOUNDRY_NPU_OCCUPANCY_POLICY",
    "TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS",
    "TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB",
    "TILEFOUNDRY_NPU_CANN_SERIES",
    "TILEFOUNDRY_NPU_TILELANG_REPO",
    "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO",
    "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO",
    "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS",
)

OPTIONAL_REPO_AND_PYTHON_KEYS = (
    "TILEFOUNDRY_NPU_PYTHON",
    "TILEFOUNDRY_NPU_CANN_SERIES",
    "TILEFOUNDRY_NPU_TILELANG_REPO",
    "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO",
    "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO",
)

OPTIONAL_FORWARDED_KEYS = (
    OPTIONAL_REPO_AND_PYTHON_KEYS + OPTIONAL_AGENT_LLM_KEYS + OPTIONAL_PROXY_KEYS
)

OPTIONAL_ABS_PATH_KEYS = (
    "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE",
    "TILEFOUNDRY_NPU_AGY_BIN",
    "TILEFOUNDRY_NPU_TILELANG_REPO",
    "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO",
    "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO",
)

NOT_MACHINE_KEYS = frozenset({
    "TILEFOUNDRY_NPU_ENV_FILE",
    "TILEFOUNDRY_NPU_DEVICE_ID",
    "TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE",
})

REQUIRED = (
    "TILEFOUNDRY_NPU_CANN_SETUP",
    "TILEFOUNDRY_NPU_CONDA_SH",
    "TILEFOUNDRY_NPU_CONDA_ENV",
)

DEFAULT_CPU_TESTS = (
    "tests/tilefoundry",
    "tests/ascend/target",
    "tests/ascend/passes",
    "tests/ascend/operators",
)
# Collected when present so the V2 native CPU gate covers the agent, cost_model,
# host L3, and legacy IR suites without breaking checkouts that omit or predate them.
OPTIONAL_CPU_TESTS = (
    "tests/ascend/ir",
    "tests/ascend/cost_model",
    "tests/ascend/agent",
    "tests/ascend/sil",
    "tests/ascend/benchmarks/test_l3_contract_host.py",
)
# Device-only test nodes that the CPU gate would still collect once
# tests/ascend/agent and tests/ascend/target were added to the default suite.
# The CPU gate runs with TORCH_DEVICE_BACKEND_AUTOLOAD=0, but an explicit
# torch_npu import still binds a device when a driver is present, so these
# nodes are deselected by exact id and the device gate owns them.
CPU_DEVICE_NODE_DESELECTS = (
    "tests/ascend/agent/test_ablation_distiller_npu.py"
    "::test_live_flywheel_micro_probe_verification_and_kb_admission",
    "tests/ascend/agent/test_e2e_evolution_npu.py::test_live_e2e_agent_evolution_on_npu",
    "tests/ascend/agent/test_funnel_smoke_npu.py::test_live_one_op_funnel_stage_a_b_c_on_npu",
    "tests/ascend/agent/test_surrogate_batch_npu.py::test_surrogate_anchor_batch_npu",
    "tests/ascend/target/test_micro_probes_npu.py::test_ascend950_calibration_micro_probes",
)
DEFAULT_CPU_WORKERS = 12
MAX_CPU_WORKERS = 16
CPU_WORKER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
# A typed fast request fixes the one harness node it runs. The request carries
# data only; it cannot select a pytest node or command. Batch and profile share
# this node so the device-fast gate has one stable synthetic-device workload.
FAST_DEVICE_TEST_FILE = "tests/ascend/agent/test_surrogate_batch_npu.py"
FAST_DEVICE_TEST_NODE = f"{FAST_DEVICE_TEST_FILE}::test_surrogate_anchor_batch_npu"

DEFAULT_PYTHONPATH_PARTS = ("src", "targets/ascend", "targets/apple", "targets/nvidia")
DEFAULT_PYTHONPATH = ":".join(DEFAULT_PYTHONPATH_PARTS)
TILELANG_SABI_EXTENSIONS = (
    "tilelang_cython_wrapper",
    "tilelang_pto_wrapper",
    "tilelang_ascend_npu_exchange",
    "tilelang_deepgemm_heuristics",
)
TILELANG_BUILD_STAMP_FILE = ".agent_gates_build_stamp.json"
DEFAULT_TARGETS_FILE = "examples/ascend950pr_ops/cannbench_targets.json"
DEFAULT_CANNBENCH_ARCH = "ascend950"
TASK_RE = re.compile(r"level[1-9][0-9]*/[a-z][a-z0-9_]*")
ARCH_RE = re.compile(r"ascend[0-9a-z_]+")

DEFAULT_CPU_TIMEOUT = 600.0
DEFAULT_DEVICE_TIMEOUT = 600.0
DEFAULT_PREP_TIMEOUT = 840.0
DEFAULT_CANNBENCH_TIMEOUT = 3300.0
DEFAULT_REMOTE_TIMEOUT = 840.0
DEFAULT_L3_TIMEOUT = 1200.0
MAX_L3_TIMEOUT = 1800.0
# The engine publishes the dynamic target/candidate pair to a dynamicTarget
# recipe only; these names mirror agent_gates.dynamic_target.
L3_TARGET_REQUEST_ENV = "AGENT_GATES_TARGET"
L3_CANDIDATE_REQUEST_ENV = "AGENT_GATES_CANDIDATE"
L3_EVIDENCE_ENV = "TF_L3_EVIDENCE_PATH"
L3_PROFILE_ENV = "TF_L3_PROFILE"
L3_TARGET_OPERATOR_ENV = "TF_L3_TARGET_OPERATOR"
L3_TARGET_MODULE_ENV = "TF_L3_TARGET_MODULE_PATH"
L3_EXPECTED_PYTHON_ENV = "TF_L3_EXPECTED_PYTHON"
L3_CANNBENCH_REPO_ENV = "TF_L3_CANNBENCH_REPO"
L3_CACHE_EXPECTATION_ENV = "TF_L3_CACHE_EXPECTATION"
L3_CACHE_MODE_ENV = "CANNBENCH_CACHE_MODE"
L3_EVIDENCE_SCHEMA = "tilefoundry-l3-evidence/v1"
# The dynamic fast gate consumes the V2 target/candidate pair only: the target
# is fast/<request_id> and the candidate is the immutable request descriptor at
# <build>/agent_gates/fast/<request_id>.request.json. The adapter snapshots its
# bytes before the run and re-hashes the request, every candidate module, and
# the response after the run, so a receipt can only describe inputs that did not
# change while the device was held. Machine paths (cannbench checkout, caches)
# come from machine_values, never from the request.
FAST_REQUEST_ENV = "TF_FAST_REQUEST"
FAST_REQUEST_SOURCE_ENV = "TF_FAST_SOURCE_ROOT"
FAST_REQUEST_LEGACY_ENV = "TF_SURROGATE_BATCH_REQUEST"
FAST_MICRO_PROBE_LEGACY_ENV = "TF_MICRO_PROBE_REQUEST"
FAST_REQUEST_SCHEMA = "tilefoundry-fast-request/v1"
FAST_RESPONSE_SCHEMA = "tilefoundry-fast-response/v1"
FAST_REQUEST_MODES = ("batch", "profile", "micro_probe")
FAST_RESPONSE_STATUSES = (
    "ok",
    "candidate_failure",
    "request_error",
    "infrastructure_error",
)
FAST_REQUEST_MAX_BYTES = 4 << 20
FAST_RESPONSE_MAX_BYTES = 64 << 20
FAST_REQUEST_ID_RE = re.compile(r"[0-9a-f]{32}")
FAST_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
FAST_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
FAST_SHA256_RE = re.compile(r"[0-9a-f]{64}")
FAST_REQUEST_COMMON_KEYS = (
    "schema",
    "mode",
    "request_id",
    "response_path",
    "warmup",
    "repeat",
    "require_all_passed",
)
FAST_ANCHOR_KEYS = ("task", "case_num", "regime_tag", "ref_perf_us", "t_hw_us")
FAST_PROBE_KEYS = (
    "probe_id",
    "probe_kind",
    "factory_name",
    "line_count_baseline",
    "line_count_optimized",
    "tensor_shape",
    "tensor_dtype",
    "min_speedup",
)
FAST_PROBE_CANDIDATE_IDS = ("baseline", "optimized")
FAST_PROBE_METRIC_KEYS = (
    "baseline_dur_us",
    "optimized_dur_us",
    "speedup",
    "numerical_ok",
)
FAST_COVERAGE_KEYS = (
    "expected_candidates",
    "evaluated_candidates",
    "expected_msprof_targets",
    "observed_msprof_targets",
)
L3_PROFILE = "l3-promotion"
L3_ACCEPTANCE_TEST = "tests/ascend/benchmarks/test_npu_acceptance.py"
L3_GENERIC_TEST = "test_target_operator_full_cannbench_contract"
# The promotion profile fixes the acceptance tests that must pass for a target.
# AR and MLA each have their own case suite and dynamic-cache proof; every other
# canonical target uses the generic full-contract test.
L3_PROFILE_TESTS = {
    "attention_residual": (
        "test_attention_residual_cannbench_cases",
        "test_attention_residual_baseline_vs_tuned_parity",
        "test_dynamic_shape_kernel_cache_reuse",
    ),
    "mla": (
        "test_mla_cannbench_cases",
        "test_mla_supplemental_cases",
        "test_dynamic_shape_kernel_cache_reuse",
    ),
}
L3_CACHE_COMPONENTS = ("tilelang", "reference")
DEFAULT_CANNBENCH_MCP_URL = "https://cannbench.com/mcp"
MCP_CONSENT_VERSION_RE = re.compile(r"v[1-9][0-9]*")

DEVICE_PROBE = r"""
import json, os, pathlib, sys, torch, torch_npu
facts = {
    "tilefoundryImported": False,
    "npuAvailable": False,
    "deviceCountOne": False,
    "tensorOnDevice": False,
    "fp32VectorOk": False,
    "bf16MatmulOk": False,
    "numericalOk": False,
    "synchronizeOk": False,
    "deviceMemoryPositive": False,
}
try:
    import tilefoundry  # noqa: F401
    facts["tilefoundryImported"] = True
except Exception as exc:
    print(f"tilefoundry import failed: {exc}", file=sys.stderr)

expected_tl = os.environ.get("TILEFOUNDRY_EXPECTED_TILELANG_SOURCE", "").strip()
if expected_tl:
    facts["tilelangReused"] = False
    try:
        import tilelang, tilelang_cython_wrapper  # noqa: F401
        tl_file = pathlib.Path(tilelang.__file__).resolve()
        wrapper_file = pathlib.Path(getattr(tilelang_cython_wrapper, "__file__", "")).resolve()
        exp_root = pathlib.Path(expected_tl).resolve()
        ok_tl = tl_file.is_relative_to(exp_root) and wrapper_file.is_relative_to(exp_root / "build" / "lib")
        if not ok_tl:
            print(
                f"tilelang imported from ({tl_file}, {wrapper_file}), expected under {exp_root}",
                file=sys.stderr,
            )
        facts["tilelangReused"] = ok_tl
        facts["tilelangVersion"] = getattr(tilelang, "__version__", "")
    except Exception as exc:
        print(f"tilelang reuse import failed: {exc}", file=sys.stderr)

try:
    avail = bool(torch.npu.is_available())
    facts["npuAvailable"] = avail
    count = int(torch.npu.device_count()) if avail else 0
    facts["deviceCountOne"] = (count == 1)
    if avail and count >= 1:
        torch.npu.set_device(0)
        a_cpu = torch.linspace(-2.0, 2.0, 256, dtype=torch.float32)
        b_cpu = torch.linspace(1.0, 3.0, 256, dtype=torch.float32)
        a_npu = a_cpu.to("npu:0")
        b_npu = b_cpu.to("npu:0")
        facts["tensorOnDevice"] = (a_npu.device.type == "npu" and b_npu.device.type == "npu")
        c_npu = (a_npu + b_npu) * 2.0
        torch.npu.synchronize()
        facts["fp32VectorOk"] = bool(torch.allclose(c_npu.cpu(), (a_cpu + b_cpu) * 2.0, atol=1e-5, rtol=1e-5))

        m1_cpu = torch.randn((64, 64), dtype=torch.bfloat16)
        m2_cpu = torch.randn((64, 64), dtype=torch.bfloat16)
        out_npu = torch.matmul(m1_cpu.to("npu:0"), m2_cpu.to("npu:0"))
        torch.npu.synchronize()
        ref = torch.matmul(m1_cpu.float(), m2_cpu.float())
        facts["bf16MatmulOk"] = bool(torch.allclose(out_npu.cpu().float(), ref, atol=1.0, rtol=5e-2))
        facts["numericalOk"] = facts["fp32VectorOk"] and facts["bf16MatmulOk"]
        facts["synchronizeOk"] = True
        facts["deviceMemoryPositive"] = int(torch.npu.max_memory_allocated(0)) > 0
except Exception as exc:
    print(f"npu probe failed: {exc}", file=sys.stderr)

print(json.dumps(facts, sort_keys=True))
sys.exit(0 if all(v for v in facts.values() if isinstance(v, bool)) else 1)
"""

DEVICE_PROBE_FLAGS = (
    "tilefoundryImported",
    "npuAvailable",
    "deviceCountOne",
    "tensorOnDevice",
    "fp32VectorOk",
    "bf16MatmulOk",
    "numericalOk",
    "synchronizeOk",
    "deviceMemoryPositive",
)

CANNBENCH_ENV_INSPECT = r"""
import importlib, importlib.metadata as md, json, pathlib, sys
missing = []
for mod in ("yaml", "ruamel.yaml", "expecttest", "build", "safetensors"):
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
try:
    import torch
    torch_cmake = torch.utils.cmake_prefix_path + "/Torch"
    torch_ver = getattr(torch, "__version__", "")
except Exception as exc:
    torch_cmake = ""
    torch_ver = ""
    missing.append(f"torch({exc})")
try:
    torch_npu_ver = md.version("torch-npu")
except Exception:
    torch_npu_ver = ""
try:
    spec = importlib.util.find_spec("torch_npu")
    torch_npu_dir = str(pathlib.Path(spec.origin).parent) if spec and spec.origin else ""
except Exception:
    torch_npu_dir = ""
if not torch_npu_dir:
    missing.append("torch_npu")
print(json.dumps({
    "missing": missing,
    "pythonVersion": sys.version.split()[0],
    "torchVersion": torch_ver,
    "torchNpuVersion": torch_npu_ver,
    "torchCmakeDir": torch_cmake,
    "torchNpuDir": torch_npu_dir,
}, sort_keys=True))
"""

CANNBENCH_ABI_PROBE = r"""
import json, sys, torch, torch_npu
ok = False
try:
    import cann_bench_utils
    torch.npu.set_device(0)
    probe = torch.arange(4096, dtype=torch.float16, device="npu:0")
    copied = cann_bench_utils.cann_bench_clone(probe)
    torch.npu.synchronize()
    ok = bool(torch.equal(copied.cpu(), probe.cpu()))
except Exception as exc:
    print(f"cann_bench_utils probe failed: {exc}", file=sys.stderr)
print(json.dumps({"cannBenchUtilsCallable": ok}, sort_keys=True))
sys.exit(0 if ok else 1)
"""

CANNBENCH_RUNNER = r"""
import datetime as _dt, json, math, os, pathlib, shutil, subprocess, sys, tempfile, yaml

cfg = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
repo = pathlib.Path(cfg["repo"])
source = pathlib.Path(cfg["source"])
utils_dir = pathlib.Path(cfg["utils_dir"])
reports_root = pathlib.Path(cfg["reports_root"])
bench_src = repo / "src"
per_op_timeout = max(60, int(cfg.get("per_op_timeout", 3300)))
tf_paths = [str((source / p).resolve()) for p in cfg["pythonpath_parts"] if (source / p).exists()]

reports_root.mkdir(parents=True, exist_ok=True)
stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

results = []
all_valid = True
all_passed = True
all_timed = True
total_cases = 0
total_passed = 0
total_timed = 0

for index, item in enumerate(cfg["targets"]):
    task = item["task"]
    candidate_rel = item["candidate"]
    task_dir = repo / "tasks" / task
    candidate = (source / candidate_rel).resolve()

    suffix = f"{stamp}_{task.replace('/', '__')}"
    if index > 0:
        suffix = f"{suffix}_{index}"
    report_dir = reports_root / suffix
    dedup = 1
    while report_dir.exists():
        report_dir = reports_root / f"{suffix}_{dedup}"
        dedup += 1
    report_dir.mkdir(parents=True, exist_ok=False)

    schema = yaml.safe_load((task_dir / "proto.yaml").read_text(encoding="utf-8"))["operator"]
    function = schema["schema"].split("(")[0].strip()
    expected_ids = {
        f"{task}_{case['case_id']}"
        for case in yaml.safe_load((task_dir / "cases.yaml").read_text(encoding="utf-8"))["cases"]
    }

    with tempfile.TemporaryDirectory(prefix="cannbench-tf-") as temp:
        stage = pathlib.Path(temp)
        l4_cache = stage / "tilelang-cache"
        l4_cache.mkdir()
        staged_task = stage / "tasks" / task
        staged_task.mkdir(parents=True)
        for fname in ("golden.py", "cases.yaml", "proto.yaml"):
            shutil.copy2(task_dir / fname, staged_task / fname)
        shutil.copytree(repo / "tasks" / "metadata", stage / "tasks" / "metadata")
        pkg = stage / "submission" / "cann_bench"
        pkg.mkdir(parents=True)
        shutil.copy2(candidate, pkg / "candidate.py")
        (pkg / "__init__.py").write_text(f"from .candidate import {function}\n", encoding="utf-8")

        env = os.environ.copy()
        # The adapter exports only the activation-derived PYTHONPATH (CANN
        # toolkit paths). Everything else from the parent shell is dropped so
        # a project hook cannot reach the official workload.
        activation_pythonpath = env.pop("AGENT_GATES_ACTIVATION_PYTHONPATH", "")
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(cfg["device"])
        env["TILELANG_CACHE_DIR"] = str(l4_cache)
        env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
        env["TILELANG_DISABLE_CACHE"] = "0"
        env["PYTHONNOUSERSITE"] = "1"
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            env.pop(name, None)
        tl_src = (cfg.get("tilelang_source") or "").strip()
        if tl_src:
            env.setdefault("USE_CUDA", "0")
        pp_entries = [str(stage / "submission"), str(utils_dir), str(bench_src)]
        if tl_src:
            pp_entries.append(tl_src)
        pp_entries.extend(tf_paths)
        if activation_pythonpath:
            pp_entries.append(activation_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pp_entries)

        cmd = [
            sys.executable, "-m", "kernel_eval.staged_eval",
            "--bench-name", "cann",
            "--task-dir", str(staged_task),
            "--operator", schema["name"],
            "--device-id", "0",
            "--processes-per-card", "1",
            "--eval-seed", "0",
            "--input-dist", "uniform",
            "--warmup", "3",
            "--repeat", "5",
            "--timeout-per-operator", str(per_op_timeout),
            "--reports-dir", str(report_dir),
            "--eval-code", "official",
        ]
        print(f"=== CANNBench running {task} ({candidate_rel}) -> {report_dir} ===", file=sys.stderr, flush=True)
        rc = subprocess.call(cmd, cwd=str(repo), env=env, stdout=sys.stderr, stderr=sys.stderr)

    final_path = report_dir / "official_final.json"
    count = passed = timed = 0
    score = 0.0
    avg_speedup = 0.0
    valid = False
    case_rows = []
    if final_path.is_file():
        try:
            report = json.loads(final_path.read_text(encoding="utf-8"))
            op = report["operators"][0]
            cases = op["cases"]
            count = len(cases)
            case_rows = [
                {
                    "case_id": str(c.get("case_id", "")),
                    "speedup": float(c.get("speedup") or 0.0),
                    "elapsed_us": float(c.get("elapsed_us") or 0.0),
                }
                for c in cases
            ]
            passed = sum(
                1 for c in cases
                if c.get("status") == "success" and (c.get("accuracy") or {}).get("passed") is True
            )
            timed = sum(
                1 for c in cases
                if isinstance(c.get("elapsed_us"), (int, float))
                and math.isfinite(c["elapsed_us"])
                and c["elapsed_us"] > 0
                and not c.get("perf_collection_failed")
            )
            score = float(op.get("score") or 0.0)
            avg_speedup = float(op.get("avg_speedup") or 0.0)
            valid = (
                rc == 0
                and count == len(expected_ids)
                and {c.get("case_id") for c in cases} == expected_ids
                and passed == count
                and timed == count
            )
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            valid = False

    total_cases += count
    total_passed += passed
    total_timed += timed
    all_valid = all_valid and valid
    all_passed = all_passed and (count > 0 and passed == count)
    all_timed = all_timed and (count > 0 and timed == count)
    results.append({
        "task": task,
        "candidate": candidate_rel,
        "reportValid": valid,
        "reportDir": str(report_dir),
        "caseCount": count,
        "passedCases": passed,
        "timedCases": timed,
        "cases": case_rows,
        "score": round(score, 6),
        "avgSpeedup": round(avg_speedup, 6),
    })

mean_score = round(sum(r["score"] for r in results) / len(results), 6) if results else 0.0
mean_speedup = round(sum(r["avgSpeedup"] for r in results) / len(results), 6) if results else 0.0
print(json.dumps({
    "officialReportValid": bool(results) and all_valid,
    "allCasesPassed": bool(results) and all_passed,
    "allTimingsValid": bool(results) and all_timed,
    "targetCount": len(results),
    "caseCount": total_cases,
    "passedCases": total_passed,
    "timedCases": total_timed,
    "score": mean_score,
    "avgSpeedup": mean_speedup,
    "results": results,
}, sort_keys=True))
sys.exit(0 if (bool(results) and all_valid and all_passed and all_timed) else 1)
"""


# Preflight runs in the target interpreter BEFORE the device lease is opened:
# it proves the CANN/torch/TileLang comparator stack imports, the comparator
# agrees on equal tensors, and the two dedicated cache roots accept a real
# write/read/delete.  It performs no device work.
L3_PREFLIGHT = r'''
import importlib
import importlib.metadata as metadata
import json
import os
import pathlib
import platform
import sys
import traceback


def _module_facts(name, distribution=None):
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "") or ""
    if not version and distribution:
        try:
            version = metadata.version(distribution)
        except Exception:
            version = ""
    return {
        "name": name,
        "file": str(getattr(module, "__file__", "") or ""),
        "version": str(version),
    }


def _cache_probe(name, raw):
    if not raw:
        raise RuntimeError(f"{name} cache directory was not supplied by the adapter")
    root = pathlib.Path(raw)
    if not root.is_absolute():
        raise RuntimeError(f"{name} cache directory must be absolute: {raw!r}")
    root.mkdir(parents=True, exist_ok=True)
    payload = b"agent-gates-l3-cache-probe"
    probe = root / f".agent_gates_probe_{os.getpid()}"
    try:
        probe.write_bytes(payload)
        read_back = probe.read_bytes()
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass
    return {
        "path": str(root),
        "writable": read_back == payload,
        "probeBytes": len(payload),
    }


facts = {
    "ok": False,
    "python": sys.executable,
    "pythonVersion": platform.python_version(),
    "cannSetup": os.environ.get("TILEFOUNDRY_NPU_CANN_SETUP", ""),
    "modules": {},
    "cache": {},
}
try:
    import numpy
    import torch

    facts["modules"]["torch"] = _module_facts("torch", "torch")
    facts["modules"]["numpy"] = _module_facts("numpy", "numpy")
    # Importing torch_npu loads the CANN runtime libraries.  No device work is
    # performed here; ASCEND_RT_VISIBLE_DEVICES is empty for this process.
    facts["modules"]["torch_npu"] = _module_facts("torch_npu", "torch-npu")
    facts["modules"]["tilelang"] = _module_facts("tilelang", "tilelang")

    vector = torch.arange(8, dtype=torch.float32)
    facts["torchNumpyInterop"] = bool(vector.numpy().tolist() == list(range(8)))

    try:
        from npu_compare import compare_case_result
        facts["modules"]["npu_compare"] = _module_facts("npu_compare")
        verdict = compare_case_result(
            case_id="preflight",
            actual=vector,
            golden=vector.clone(),
            accuracy_mode="cann-bench",
        )
        facts["comparator"] = {
            "module": "npu_compare",
            "callable": callable(compare_case_result),
            "passedOnEqual": bool(verdict.passed),
        }
    except ImportError:
        # The acceptance suite falls back to the same CANNBench comparator
        # when the official npu_compare module is not installed. The adapter
        # passes the checkout's src directory on PYTHONPATH, so import it here
        # without generating any shim module.
        facts["modules"]["npu_compare"] = None
        from kernel_eval.utils.compare import compare_tensors
        facts["modules"]["kernel_eval.utils.compare"] = _module_facts(
            "kernel_eval.utils.compare"
        )
        verdict = compare_tensors(
            vector, vector.clone(), dtype="float32", native_output=None
        )
        facts["comparator"] = {
            "module": "kernel_eval.utils.compare",
            "callable": callable(compare_tensors),
            "passedOnEqual": bool(verdict.passed),
        }

    # tilelang/__init__.py exports ``env`` as an Environment instance, which
    # shadows ``import tilelang.env as ...``; read the cache API from that
    # exported instance instead of assuming the attribute is the submodule.
    from tilelang.env import env as tilelang_env

    facts["tilelangCacheDir"] = tilelang_env.get_cache_write_dir()
    facts["cache"]["tilelang"] = _cache_probe(
        "tilelang", os.environ.get("TILELANG_CACHE_DIR", "")
    )
    facts["cache"]["reference"] = _cache_probe(
        "reference", os.environ.get("CANNBENCH_CACHE_DIR", "")
    )

    import yaml

    task = sys.argv[2]
    repo = pathlib.Path(sys.argv[3])
    cases = yaml.safe_load(
        (repo / "tasks" / task / "cases.yaml").read_text(encoding="utf-8")
    )["cases"]
    facts["caseIds"] = [str(case.get("case_id")) for case in cases]
    facts["ok"] = (
        facts["torchNumpyInterop"]
        and facts["comparator"]["callable"]
        and facts["comparator"]["passedOnEqual"]
        and all(root["writable"] for root in facts["cache"].values())
        and bool(facts["caseIds"])
    )
except Exception as exc:  # noqa: BLE001 - the fact is the failure, not a trace
    facts["error"] = f"{type(exc).__name__}: {exc}"
    facts["traceback"] = traceback.format_exc()[-2000:]

pathlib.Path(sys.argv[1]).write_text(json.dumps(facts, sort_keys=True), encoding="utf-8")
sys.exit(0 if facts["ok"] else 1)
'''


# The actual promotion run executes pytest in the SAME target interpreter that
# records its own identity.  Facts are written before pytest starts so they are
# available even when pytest is killed by the deadline.
L3_PYTEST_WRAPPER = r'''
import importlib
import importlib.metadata as metadata
import json
import os
import pathlib
import platform
import sys
import traceback


def _module_facts(name, distribution=None):
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "") or ""
    if not version and distribution:
        try:
            version = metadata.version(distribution)
        except Exception:
            version = ""
    return {
        "name": name,
        "file": str(getattr(module, "__file__", "") or ""),
        "version": str(version),
    }


facts = {
    "python": sys.executable,
    "pythonVersion": platform.python_version(),
    "sysPathHead": [str(item) for item in sys.path[:6]],
    "modules": {},
}
try:
    facts["modules"]["torch"] = _module_facts("torch", "torch")
    facts["modules"]["numpy"] = _module_facts("numpy", "numpy")
    facts["modules"]["torch_npu"] = _module_facts("torch_npu", "torch-npu")
    facts["modules"]["tilelang"] = _module_facts("tilelang", "tilelang")
    try:
        facts["modules"]["npu_compare"] = _module_facts("npu_compare")
    except ImportError:
        facts["modules"]["npu_compare"] = None
    try:
        facts["modules"]["kernel_eval.utils.compare"] = _module_facts(
            "kernel_eval.utils.compare"
        )
    except ImportError:
        facts["modules"]["kernel_eval.utils.compare"] = None
    expected_source = (os.environ.get("TILEFOUNDRY_EXPECTED_TILELANG_SOURCE") or "").strip()
    tilelang_file = facts["modules"]["tilelang"]["file"]
    facts["tilelangSourceMatches"] = (
        None
        if not expected_source
        else pathlib.Path(tilelang_file).resolve().is_relative_to(pathlib.Path(expected_source).resolve())
    )
    facts["tilelangCacheDir"] = os.environ.get("TILELANG_CACHE_DIR", "")
    facts["referenceCacheDir"] = os.environ.get("CANNBENCH_CACHE_DIR", "")
except Exception as exc:  # noqa: BLE001 - pytest still runs and reports
    facts["importError"] = f"{type(exc).__name__}: {exc}"
    facts["traceback"] = traceback.format_exc()[-2000:]

facts_path = pathlib.Path(sys.argv[1])
facts_path.write_text(json.dumps(facts, sort_keys=True), encoding="utf-8")

import pytest

report = sys.argv[2]
test_file = sys.argv[3]
sys.exit(int(pytest.main([
    "-p", "no:cacheprovider",
    "-q",
    f"--junit-xml={report}",
    test_file,
])))
'''


ERROR_CATEGORIES = (
    "environment",
    "occupancy",
    "deadline",
    "numeric",
    "candidate",
    "coverage",
    "signal",
    "infra",
    "none",
)


class BridgeError(RuntimeError):
    EXIT = adapter_runtime.BLOCKED_ENV

    def __init__(
        self,
        message: str,
        *,
        category: str = "environment",
        occupancy: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category if category in ERROR_CATEGORIES else "infra"
        self.occupancy = occupancy


class _Deadline:
    """One monotonic deadline paid by every step of a recipe."""

    def __init__(self, seconds: float) -> None:
        self.started = time.monotonic()
        self.deadline = self.started + max(0.0, float(seconds))

    def elapsed(self) -> float:
        return round(time.monotonic() - self.started, 3)

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def cap(self, limit: float) -> float:
        return max(0.0, min(float(limit), self.remaining()))

    def require(self, label: str) -> None:
        if self.remaining() <= 0:
            raise BridgeError(
                f"{label} exceeded the recipe deadline", category="deadline"
            )


def _die(code: int, message: str) -> NoReturn:
    print(f"{PROG}: {message}", file=sys.stderr)
    raise SystemExit(code)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_env_file(text: str, path: Path) -> dict[str, str]:
    candidate = Path(path)
    if candidate.is_file() and not candidate.is_symlink():
        try:
            if candidate.read_text(encoding="utf-8") == text:
                return adapter_runtime.parse_env_file(candidate, TOOLCHAIN_KEYS)
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=True) as tmp:
        tmp.write(text)
        tmp.flush()
        try:
            return adapter_runtime.parse_env_file(Path(tmp.name), TOOLCHAIN_KEYS)
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc


def resolved_env_file() -> Path:
    try:
        path = adapter_runtime.resolve_env_file("TILEFOUNDRY_NPU_ENV_FILE", ADAPTER_DIR)
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
    if path is None:
        raise BridgeError(
            f"no machine file: set TILEFOUNDRY_NPU_ENV_FILE or create {ADAPTER_DIR / 'env.sh'}"
        )
    return path


def machine_values() -> dict[str, str]:
    path = resolved_env_file()
    try:
        values = adapter_runtime.parse_env_file(path, TOOLCHAIN_KEYS)
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
    for name in sorted(os.environ):
        if not name.startswith("TILEFOUNDRY_NPU_") or name in NOT_MACHINE_KEYS:
            continue
        if name not in TOOLCHAIN_KEYS:
            raise BridgeError(f"{name} is exported but is not a machine-file key")
        if name not in values:
            raise BridgeError(f"{name} is exported in shell but absent from {path}")
        if os.environ[name] != values[name]:
            raise BridgeError(f"{name} export disagrees with {path}")
    _llm_credential_name(values)
    return values


def _llm_credential_name(values: dict[str, str]) -> str:
    """Return the configured credential variable name."""
    name = (values.get(LLM_API_KEY_ENV) or "").strip()
    if name and (
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name.startswith("TILEFOUNDRY_NPU_")
    ):
        raise BridgeError(f"{LLM_API_KEY_ENV} must name a separate environment variable")
    return name


def _required(values: dict[str, str], name: str) -> str:
    value = (values.get(name) or "").strip()
    if not value:
        raise BridgeError(f"{name} is required in the machine file")
    return value


def _binary_flag(values: dict[str, str], name: str) -> bool:
    raw = (values.get(name) or "").strip()
    if raw == "1":
        return True
    if raw in {"", "0"}:
        return False
    raise BridgeError(f"{name} must be 0 or 1")


def build_subdir(values: dict[str, str]) -> str:
    if "TILEFOUNDRY_NPU_BUILD_SUBDIR" in values:
        raw = values["TILEFOUNDRY_NPU_BUILD_SUBDIR"].strip()
        if not raw:
            raise BridgeError("TILEFOUNDRY_NPU_BUILD_SUBDIR is set but empty")
        try:
            return adapter_runtime.validate_build_subdir(raw)
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc
    return "build"


def python_path(values: dict[str, str]) -> Path:
    explicit = (values.get("TILEFOUNDRY_NPU_PYTHON") or "").strip()
    if explicit:
        return Path(explicit)
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
    if envs_path:
        return Path(envs_path) / conda_env / "bin" / "python3"
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    return conda_sh.parent.parent.parent / "envs" / conda_env / "bin" / "python3"


def _parse_test_list(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return default
    parts = [item.strip() for item in re.split(r"[\s,]+", raw.strip()) if item.strip()]
    if not parts:
        raise BridgeError("test list is set but names no path")
    for item in parts:
        p = Path(item)
        if p.is_absolute() or ".." in p.parts:
            raise BridgeError(f"test path must be relative inside source root: {item!r}")
    return tuple(parts)


def _is_active_cpu_suite_path(path: Path) -> bool:
    """Return True when path is a file, an empty mock dir, or has non-cache entries."""
    if not path.exists():
        return False
    if path.is_file():
        return True
    entries = list(path.iterdir())
    if not entries:
        return True
    for child in entries:
        if child.name in ("__pycache__", ".pytest_cache"):
            continue
        if child.is_dir():
            if _is_active_cpu_suite_path(child):
                return True
        elif child.suffix != ".pyc":
            return True
    return False


def default_cpu_tests(source: Path) -> tuple[str, ...]:
    """CPU suite paths from default and optional lists that exist in this checkout."""
    return tuple(
        rel
        for rel in (*DEFAULT_CPU_TESTS, *OPTIONAL_CPU_TESTS)
        if _is_active_cpu_suite_path(source / rel)
    )


def cpu_tests(values: dict[str, str], source: Path) -> tuple[str, ...]:
    return _parse_test_list(values.get("TILEFOUNDRY_NPU_CPU_TESTS"), default_cpu_tests(source))


def cpu_deselects(source: Path) -> tuple[str, ...]:
    """Return exact device-only node ids present in this checkout for CPU deselection."""
    return tuple(
        node
        for node in CPU_DEVICE_NODE_DESELECTS
        if (source / node.split("::", 1)[0]).is_file()
    )


def cpu_workers(values: dict[str, str]) -> int:
    """Resolve the bounded worker count for multi-process CPU pytest shards."""
    raw = (values.get("TILEFOUNDRY_NPU_CPU_WORKERS") or "").strip()
    if not raw:
        return min(DEFAULT_CPU_WORKERS, max(1, os.cpu_count() or 1))
    if not re.fullmatch(r"\d+", raw):
        raise BridgeError(
            f"TILEFOUNDRY_NPU_CPU_WORKERS must be an integer between 1 and {MAX_CPU_WORKERS}: {raw!r}"
        )
    workers = int(raw)
    if workers < 1 or workers > MAX_CPU_WORKERS:
        raise BridgeError(
            f"TILEFOUNDRY_NPU_CPU_WORKERS must be an integer between 1 and {MAX_CPU_WORKERS}: {raw!r}"
        )
    return workers


def _expand_cpu_test_shards(
    source: Path, tests: tuple[str, ...], workers: int
) -> list[tuple[str, ...]]:
    """Expand relative test directories into size-balanced file shards when safe."""
    if workers <= 1 or any("::" in item for item in tests):
        return [tuple(tests)]
    discovered: list[str] = []
    seen: set[str] = set()
    for entry in tests:
        target = source / entry
        if target.is_dir():
            dir_files = [
                path
                for path in sorted(target.rglob("*.py"))
                if "__pycache__" not in path.parts
                and (path.name.startswith("test_") or path.name.endswith("_test.py"))
            ]
            if not dir_files:
                return [tuple(tests)]
            for path in dir_files:
                rel = path.relative_to(source).as_posix()
                if rel not in seen:
                    seen.add(rel)
                    discovered.append(rel)
        elif target.is_file():
            rel = target.relative_to(source).as_posix()
            if rel not in seen:
                seen.add(rel)
                discovered.append(rel)
        else:
            return [tuple(tests)]
    if len(discovered) <= 1:
        return [tuple(tests)]

    items = sorted(
        ((source / rel).stat().st_size, rel) for rel in discovered
    )
    items.sort(key=lambda item: (-item[0], item[1]))
    num_shards = min(workers, len(items))
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    shard_bytes = [0 for _ in range(num_shards)]
    for size, rel in items:
        idx = min(
            range(num_shards),
            key=lambda shard_i: (shard_bytes[shard_i], len(shards[shard_i]), shard_i),
        )
        shards[idx].append(rel)
        shard_bytes[idx] += size
    return [tuple(sorted(shard)) for shard in shards if shard]


def fast_device_tests(source: Path) -> tuple[str, ...]:
    """Return the fixed harness node a typed fast request runs, requiring its file."""
    if not (source / FAST_DEVICE_TEST_FILE).is_file():
        raise BridgeError(
            f"typed fast request harness is missing in {source}: {FAST_DEVICE_TEST_FILE}"
        )
    return (FAST_DEVICE_TEST_NODE,)


def device_tests(values: dict[str, str]) -> tuple[str, ...]:
    return _parse_test_list(values.get("TILEFOUNDRY_NPU_DEVICE_TESTS"), ())


def cannbench_repo(values: dict[str, str]) -> Path:
    raw = _required(values, "TILEFOUNDRY_NPU_CANNBENCH_REPO")
    path = Path(raw)
    if not path.is_absolute():
        raise BridgeError(f"TILEFOUNDRY_NPU_CANNBENCH_REPO must be an absolute path: {raw!r}")
    resolved = path.resolve()
    required_paths = (
        resolved / "scripts" / "run_evaluation.sh",
        resolved / "src" / "kernel_eval",
        resolved / "src" / "cann_bench_utils" / "CMakeLists.txt",
        resolved / "tasks" / "metadata",
    )
    for req in required_paths:
        if not req.exists():
            raise BridgeError(f"TILEFOUNDRY_NPU_CANNBENCH_REPO missing expected path {req}")
    return resolved


def work_root(values: dict[str, str], source: Path, repo: Path) -> Path:
    raw = _required(values, "TILEFOUNDRY_NPU_WORK_ROOT")
    path = Path(raw)
    if not path.is_absolute():
        raise BridgeError(f"TILEFOUNDRY_NPU_WORK_ROOT must be an absolute path: {raw!r}")
    resolved = path.resolve()
    src_resolved = source.resolve()
    repo_resolved = repo.resolve()
    if resolved == src_resolved or resolved.is_relative_to(src_resolved):
        raise BridgeError(f"TILEFOUNDRY_NPU_WORK_ROOT must be outside TileFoundry source root: {resolved}")
    if resolved == repo_resolved or resolved.is_relative_to(repo_resolved):
        raise BridgeError(f"TILEFOUNDRY_NPU_WORK_ROOT must be outside CANNBench repo: {resolved}")
    return resolved


def cannbench_arch(values: dict[str, str]) -> str:
    raw = (values.get("TILEFOUNDRY_NPU_CANNBENCH_ARCH") or "").strip() or DEFAULT_CANNBENCH_ARCH
    if not ARCH_RE.fullmatch(raw):
        raise BridgeError(f"TILEFOUNDRY_NPU_CANNBENCH_ARCH is not a valid build target: {raw!r}")
    return raw


def cannbench_targets_file_rel(values: dict[str, str]) -> str:
    raw = (values.get("TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE") or "").strip() or DEFAULT_TARGETS_FILE
    p = Path(raw)
    if p.is_absolute() or ".." in p.parts:
        raise BridgeError(f"TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE must be relative inside source: {raw!r}")
    return raw


def _validate_target_pair(task: str, candidate_rel: str, source: Path, repo: Path) -> tuple[str, str]:
    task = task.strip()
    candidate_rel = candidate_rel.strip()
    if not TASK_RE.fullmatch(task):
        raise BridgeError(f"invalid CANNBench task {task!r}; expected levelN/operator_name")
    task_dir = repo / "tasks" / task
    for fname in ("proto.yaml", "cases.yaml", "golden.py"):
        if not (task_dir / fname).is_file():
            raise BridgeError(f"CANNBench operator {task!r} missing {fname} in {repo}")
    cand_path = Path(candidate_rel)
    if not candidate_rel or cand_path.is_absolute() or ".." in cand_path.parts:
        raise BridgeError(f"CANNBench candidate path must be relative inside source: {candidate_rel!r}")
    resolved = (source / cand_path).resolve()
    src_resolved = source.resolve()
    if not resolved.is_relative_to(src_resolved) or not resolved.is_file() or resolved.suffix != ".py":
        raise BridgeError(f"CANNBench candidate .py file missing under source: {candidate_rel!r}")
    return task, cand_path.as_posix()


def resolve_cannbench_targets(
    values: dict[str, str], source: Path, repo: Path, *, allow_request: bool = False
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Resolve the configured target list, or the one pair a managed request names.

    The prep and official recipes declare an optional dynamic target, so an
    unmanaged invocation keeps evaluating the configured targets file. A managed
    V2 request names exactly one canonical task and workspace-relative
    candidate; the adapter validates that pair itself, never evaluates other
    targets alongside it, and does not depend on unrelated targets-file entries.
    """
    if allow_request:
        request = _read_cannbench_request()
        if request is not None:
            pair = _validate_target_pair(request[0], request[1], source, repo)
            return "dynamic-request", (pair,)
    rel_file = cannbench_targets_file_rel(values)
    repo_manifest = (source / rel_file).resolve()
    if not repo_manifest.is_relative_to(source.resolve()):
        raise BridgeError(f"CANNBench targets file escapes source root: {rel_file!r}")

    pairs: list[tuple[str, str]] = []
    origin = ""
    if repo_manifest.exists():
        if not repo_manifest.is_file():
            raise BridgeError(f"CANNBench targets path is not a regular file: {repo_manifest}")
        try:
            payload = json.loads(repo_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BridgeError(f"cannot parse {repo_manifest}: {exc}") from exc
        raw_items = payload.get("targets") if isinstance(payload, dict) else payload
        if not isinstance(raw_items, list) or not raw_items:
            raise BridgeError(f"{repo_manifest} must define a non-empty 'targets' list")
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict) or not isinstance(item.get("task"), str) or not isinstance(item.get("candidate"), str):
                raise BridgeError(f"{repo_manifest} target #{idx + 1} must be {{'task': str, 'candidate': str}}")
            pairs.append(_validate_target_pair(item["task"], item["candidate"], source, repo))
        origin = "repo-file"
    else:
        raw_env = (values.get("TILEFOUNDRY_NPU_CANNBENCH_TARGETS") or "").strip()
        if not raw_env:
            raise BridgeError(
                f"no CANNBench targets configured: create {rel_file} in source or set TILEFOUNDRY_NPU_CANNBENCH_TARGETS in env.sh"
            )
        entries = [e.strip() for e in re.split(r"[\s,;]+", raw_env) if e.strip()]
        if not entries:
            raise BridgeError("TILEFOUNDRY_NPU_CANNBENCH_TARGETS is empty")
        for entry in entries:
            if ":" not in entry:
                raise BridgeError(
                    f"TILEFOUNDRY_NPU_CANNBENCH_TARGETS entry {entry!r} must be formatted as levelN/op:relative/candidate.py"
                )
            task, cand = entry.split(":", 1)
            pairs.append(_validate_target_pair(task, cand, source, repo))
        origin = "env-file"
    return origin, tuple(pairs)


def _read_cannbench_request() -> tuple[str, str] | None:
    """Read the optional L4 pair the engine publishes for a managed request."""
    target = (os.environ.get(L3_TARGET_REQUEST_ENV) or "").strip()
    candidate = (os.environ.get(L3_CANDIDATE_REQUEST_ENV) or "").strip()
    if not target and not candidate:
        return None
    if not target or not candidate:
        raise BridgeError(
            f"{L3_TARGET_REQUEST_ENV} and {L3_CANDIDATE_REQUEST_ENV} "
            "must be supplied together"
        )
    return target, candidate


def _utils_source_digest(repo: Path) -> str:
    src_root = (repo / "src" / "cann_bench_utils").resolve()
    h = hashlib.sha256()
    files = sorted(p for p in src_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        rel = path.relative_to(src_root).as_posix()
        h.update(rel.encode("utf-8") + b"\0")
        h.update(path.read_bytes() + b"\0")
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_head_pure(root: Path) -> str:
    dot_git = Path(root) / ".git"
    if dot_git.is_file():
        try:
            text = dot_git.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return "NO_GIT"
        if not text.startswith("gitdir:"):
            return "NO_GIT"
        raw_dir = text[len("gitdir:"):].strip()
        git_dir = Path(raw_dir) if Path(raw_dir).is_absolute() else (Path(root) / raw_dir)
    elif dot_git.is_dir():
        git_dir = dot_git
    else:
        return "NO_GIT"

    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return "NO_GIT"
    try:
        head_text = head_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return "NO_GIT"
    if not head_text.startswith("ref:"):
        return head_text if re.fullmatch(r"[0-9a-fA-F]{40,64}", head_text) else "NO_GIT"

    ref_name = head_text[len("ref:"):].strip()
    search_dirs = [git_dir]
    commondir_file = git_dir / "commondir"
    if commondir_file.is_file():
        try:
            raw_common = commondir_file.read_text(encoding="utf-8").strip()
            common_dir = Path(raw_common) if Path(raw_common).is_absolute() else (git_dir / raw_common)
            if common_dir.is_dir():
                search_dirs.append(common_dir)
        except (OSError, UnicodeDecodeError):
            pass

    for candidate_dir in search_dirs:
        ref_file = candidate_dir / ref_name
        if ref_file.is_file():
            try:
                commit = ref_file.read_text(encoding="utf-8").strip()
                if commit:
                    return commit
            except (OSError, UnicodeDecodeError):
                pass
        packed = candidate_dir / "packed-refs"
        if packed.is_file():
            try:
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if not line or line.startswith(("#", "^")):
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2 and parts[1].strip() == ref_name:
                        return parts[0].strip()
            except (OSError, UnicodeDecodeError):
                pass
    return "NO_GIT"


def _tilelang_native_inputs_sha256(source: Path) -> str:
    """Deterministic digest matching tilelang-npu's native/CMake/Cython inputs."""
    src_root = Path(source).resolve()
    candidates: list[Path] = []
    for rel_file in ("CMakeLists.txt", "examples/ascend/deepgemm/_heuristics.pyx"):
        p = src_root / rel_file
        if p.is_file():
            candidates.append(p)
    for rel_dir in ("cmake", "src"):
        d = src_root / rel_dir
        if d.is_dir():
            candidates.extend(p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    tl_pkg = src_root / "tilelang"
    if tl_pkg.is_dir():
        for ext in ("*.pyx", "*.pxd", "*.pxi"):
            candidates.extend(p for p in tl_pkg.rglob(ext) if p.is_file() and "__pycache__" not in p.parts)

    by_rel = {p.relative_to(src_root).as_posix(): p for p in candidates}
    h = hashlib.sha256()
    for rel in sorted(by_rel):
        p = by_rel[rel]
        h.update(rel.encode("utf-8") + b"\0")
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        h.update(b"\0")
    tvm_head = _git_head_pure(src_root / "3rdparty" / "tvm")
    h.update(b"submodule:3rdparty/tvm\0" + tvm_head.encode("utf-8") + b"\0")
    return h.hexdigest()


def tilelang_reuse_identity(values: dict[str, str]) -> dict[str, str] | None:
    """Validate and bind the reused TileLang checkout built by tilelang-npu."""
    raw_src = (values.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or "").strip()
    raw_bld = (values.get("TILEFOUNDRY_NPU_TILELANG_BUILD") or "").strip()
    if not raw_src and not raw_bld:
        return None
    if raw_bld and not raw_src:
        raise BridgeError("TILEFOUNDRY_NPU_TILELANG_BUILD requires TILEFOUNDRY_NPU_TILELANG_SOURCE")
    src_path = Path(raw_src)
    if not src_path.is_absolute():
        raise BridgeError(f"TILEFOUNDRY_NPU_TILELANG_SOURCE must be an absolute path: {raw_src!r}")
    if not src_path.is_dir():
        raise BridgeError(f"TILEFOUNDRY_NPU_TILELANG_SOURCE does not exist: {src_path}")
    src = src_path.resolve()

    if raw_bld:
        bld_path = Path(raw_bld)
        if not bld_path.is_absolute():
            raise BridgeError(f"TILEFOUNDRY_NPU_TILELANG_BUILD must be an absolute path: {raw_bld!r}")
        bld = bld_path.resolve()
    else:
        bld = (src / "build").resolve()

    expected_bld = (src / "build").resolve()
    if bld != expected_bld:
        raise BridgeError(
            f"TILEFOUNDRY_NPU_TILELANG_BUILD must match {expected_bld} (got {bld}) because "
            "TileLang DEV imports always load native artifacts from <source>/build"
        )
    if not bld.is_dir():
        raise BridgeError(
            f"TILEFOUNDRY_NPU_TILELANG_BUILD does not exist: {bld}; "
            "run tilelang-npu (gate tilelang-npu-build) first — tilefoundry-npu reuses TileLang and never rebuilds it"
        )

    tl_init = src / "tilelang" / "__init__.py"
    tvm_init = src / "3rdparty" / "tvm" / "python" / "tvm" / "__init__.py"
    if not tl_init.is_file():
        raise BridgeError(f"TILEFOUNDRY_NPU_TILELANG_SOURCE missing tilelang/__init__.py: {src}")
    if not tvm_init.is_file():
        raise BridgeError(f"TILEFOUNDRY_NPU_TILELANG_SOURCE missing 3rdparty/tvm/python/tvm/__init__.py: {src}")

    lib_path = bld / "lib" / "libtilelang.so"
    if not lib_path.is_file() or lib_path.stat().st_size <= (1 << 20) or not (bld / "tvm").is_dir():
        raise BridgeError(
            f"reused TileLang build at {bld} is missing libtilelang.so or tvm/; "
            "run tilelang-npu (gate tilelang-npu-build) first — tilefoundry-npu reuses TileLang and never rebuilds it"
        )
    missing_sabi = [
        f"{ext}.abi3.so"
        for ext in TILELANG_SABI_EXTENSIONS
        if not (bld / "lib" / f"{ext}.abi3.so").is_file()
    ]
    if missing_sabi:
        raise BridgeError(
            f"reused TileLang build at {bld} is missing cp39-abi3 extension(s) ({', '.join(missing_sabi)}); "
            "run tilelang-npu (gate tilelang-npu-build) first — tilefoundry-npu reuses TileLang and never rebuilds it"
        )

    native_sha = _tilelang_native_inputs_sha256(src)
    stamp_file = bld / TILELANG_BUILD_STAMP_FILE
    if not stamp_file.is_file():
        raise BridgeError(
            f"reused TileLang build at {bld} is missing {TILELANG_BUILD_STAMP_FILE}; "
            "run tilelang-npu (gate tilelang-npu-build) first — tilefoundry-npu reuses TileLang and never rebuilds it"
        )
    try:
        stamp = json.loads(stamp_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BridgeError(f"cannot read {stamp_file}: {exc}") from exc
    if not isinstance(stamp, dict) or stamp.get("nativeInputsSha256") != native_sha:
        raise BridgeError(
            f"TileLang native/Cython sources in {src} changed since {bld} was built; "
            "run tilelang-npu (gate tilelang-npu-build) first — tilefoundry-npu reuses TileLang and never rebuilds it"
        )

    return {
        "source": str(src),
        "build": str(bld),
        "gitHead": _git_head_pure(src),
        "moduleSha256": _sha256_file(tl_init),
        "libtilelangSha256": _sha256_file(lib_path),
        "nativeInputsSha256": native_sha,
    }


def _effective_pythonpath(
    tl_identity: dict[str, str] | None, *, shim: str | None = None
) -> str:
    parts = [shim] if shim else []
    if tl_identity is not None:
        parts.append(tl_identity["source"])
    parts.append(DEFAULT_PYTHONPATH)
    return ":".join(parts)


def device_id() -> str | None:
    raw = os.environ.get("TILEFOUNDRY_NPU_DEVICE_ID")
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if not re.fullmatch(r"[0-9]+", cleaned):
        raise BridgeError(f"TILEFOUNDRY_NPU_DEVICE_ID must be a nonnegative decimal ID, got {raw!r}")
    return str(int(cleaned))


def require_device() -> str:
    dev = device_id()
    if dev is None:
        raise BridgeError("explicit TILEFOUNDRY_NPU_DEVICE_ID or --device-id authorization required")
    return dev


def exclusive_requested() -> bool:
    raw = (os.environ.get("TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _source_root() -> Path:
    source = Path.cwd().resolve()
    if not ((source / "src" / "tilefoundry").is_dir() or (source / "pyproject.toml").is_file()):
        raise BridgeError(f"{source} does not look like a TileFoundry checkout")
    return source


class FastCandidate(NamedTuple):
    """One candidate module bound by a typed fast request."""

    operator: str
    candidate_id: str
    relative_path: str
    sha256: str


class FastProbe(NamedTuple):
    """Validated micro-probe specification carried by a typed fast request."""

    probe_id: str
    probe_kind: str
    factory_name: str
    line_count_baseline: int
    line_count_optimized: int
    tensor_shape: tuple[int, ...]
    tensor_dtype: str
    min_speedup: float


class FastRequest(NamedTuple):
    """Validated, immutable snapshot identity of one typed fast request."""

    request_id: str
    mode: str
    path: Path
    relative_path: str
    sha256: str
    raw: bytes
    response_path: Path
    response_relative: str
    candidates: tuple[FastCandidate, ...]
    profile_cases: tuple[tuple[str, str, tuple[int, ...]], ...]
    probe: FastProbe | None = None


def _fast_path(source: Path, raw: object, field: str) -> tuple[str, Path]:
    """Resolve one canonical workspace-relative fast-request path without symlinks."""
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise BridgeError(f"fast request {field} must be a workspace-relative POSIX path")
    parsed = PurePosixPath(raw)
    if parsed.is_absolute() or str(parsed) != raw or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise BridgeError(
            f"fast request {field} must be a canonical relative path "
            "with no '.' or '..' segment"
        )
    current = source
    for part in parsed.parts:
        current = current / part
        if current.is_symlink():
            raise BridgeError(f"fast request {field} must not traverse a symlink: {raw}")
    return raw, source / raw


def _fast_read_bytes(path: Path, *, limit: int, field: str) -> bytes:
    """Read one regular non-symlink file with a hard byte limit."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise BridgeError(f"cannot read fast request {field} {path}: {exc.strerror}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise BridgeError(f"fast request {field} is not a regular file: {path}")
        if info.st_size > limit:
            raise BridgeError(f"fast request {field} exceeds {limit} bytes: {path}")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            block = os.read(fd, min(remaining, 1 << 20))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        data = b"".join(chunks)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise BridgeError(f"fast request {field} exceeds {limit} bytes: {path}")
    return data


def _fast_no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _fast_json_object(data: bytes, *, field: str) -> dict[str, object]:
    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_fast_no_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as exc:
        raise BridgeError(f"fast request {field} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise BridgeError(f"fast request {field} must be a JSON object")
    return parsed


def _fast_object(
    value: object,
    *,
    context: str,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> dict[str, object]:
    """Narrow one JSON value to an object with exactly the declared keys."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BridgeError(f"fast request {context} must be a JSON object")
    unknown = sorted(set(value) - set(required) - set(optional))
    missing = sorted(set(required) - set(value))
    if unknown:
        raise BridgeError(f"fast request {context} has unknown keys: {', '.join(unknown)}")
    if missing:
        raise BridgeError(f"fast request {context} is missing keys: {', '.join(missing)}")
    return value


def _fast_name(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not FAST_NAME_RE.fullmatch(value)
        or value in {".", ".."}
    ):
        raise BridgeError(
            f"fast request {field} must match [A-Za-z0-9][A-Za-z0-9._-]* and be at most 128 chars"
        )
    return value


def _fast_int(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BridgeError(f"fast request {field} must be an integer >= {minimum}")
    return value


def _fast_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise BridgeError(f"fast request {field} must be a boolean")
    return value


def _fast_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 256:
        raise BridgeError(f"fast request {field} must be a non-empty string of at most 256 chars")
    return value


def _fast_declared_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not FAST_SHA256_RE.fullmatch(value):
        raise BridgeError(f"fast request {field} must be a lowercase SHA-256 digest")
    return value


def _fast_candidate(
    value: object,
    *,
    source: Path,
    candidates_prefix: str,
    operator: str,
    context: str,
) -> FastCandidate:
    row = _fast_object(
        value,
        context=context,
        required=("candidate_id", "module_path", "sha256"),
    )
    candidate_id = _fast_name(row["candidate_id"], field=f"{context}.candidate_id")
    relative_path, module_path = _fast_path(
        source, row["module_path"], f"{context}.module_path"
    )
    if (
        relative_path == candidates_prefix
        or not PurePosixPath(relative_path).is_relative_to(
            PurePosixPath(candidates_prefix)
        )
    ):
        raise BridgeError(
            f"fast request {context}.module_path must stay under "
            f"{candidates_prefix}/: {relative_path}"
        )
    declared_sha = _fast_declared_sha(row["sha256"], field=f"{context}.sha256")
    if not module_path.is_file():
        raise BridgeError(f"fast request {context}.module_path does not name a file: {relative_path}")
    actual_sha = _sha256_file(module_path)
    if actual_sha != declared_sha:
        raise BridgeError(
            f"fast request {context}.sha256 does not match candidate bytes: "
            f"declared {declared_sha}, actual {actual_sha}"
        )
    return FastCandidate(operator, candidate_id, relative_path, declared_sha)


def _fast_anchor(value: object, *, context: str) -> None:
    row = _fast_object(value, context=context, required=FAST_ANCHOR_KEYS)
    _fast_text(row["task"], field=f"{context}.task")
    _fast_int(row["case_num"], field=f"{context}.case_num", minimum=0)
    regime_tag = row["regime_tag"]
    if not isinstance(regime_tag, str) or "\x00" in regime_tag or len(regime_tag) > 256:
        raise BridgeError(
            f"fast request {context}.regime_tag must be a string of at most 256 chars"
        )
    for field in ("ref_perf_us", "t_hw_us"):
        if isinstance(row[field], bool) or not isinstance(row[field], (int, float)):
            raise BridgeError(f"fast request {context}.{field} must be a number")


def _fast_probe(value: object, *, context: str) -> FastProbe:
    """Narrow one micro-probe specification declared by a typed fast request."""
    row = _fast_object(value, context=context, required=FAST_PROBE_KEYS)
    probe_id = _fast_text(row["probe_id"], field=f"{context}.probe_id")
    probe_kind = _fast_text(row["probe_kind"], field=f"{context}.probe_kind")
    factory_name = row["factory_name"]
    if not isinstance(factory_name, str) or not FAST_IDENTIFIER_RE.fullmatch(factory_name):
        raise BridgeError(f"fast request {context}.factory_name must be a Python identifier")
    tensor_dtype = _fast_text(row["tensor_dtype"], field=f"{context}.tensor_dtype")
    shape = row["tensor_shape"]
    if not isinstance(shape, list) or not 1 <= len(shape) <= 8:
        raise BridgeError(f"fast request {context}.tensor_shape must list 1 to 8 dimensions")
    dims = tuple(
        _fast_int(dim, field=f"{context}.tensor_shape[{index}]", minimum=1)
        for index, dim in enumerate(shape)
    )
    min_speedup = row["min_speedup"]
    if (
        isinstance(min_speedup, bool)
        or not isinstance(min_speedup, (int, float))
        or not math.isfinite(min_speedup)
        or min_speedup <= 0.0
    ):
        raise BridgeError(
            f"fast request {context}.min_speedup must be a finite positive number"
        )
    return FastProbe(
        probe_id=probe_id,
        probe_kind=probe_kind,
        factory_name=factory_name,
        line_count_baseline=_fast_int(
            row["line_count_baseline"], field=f"{context}.line_count_baseline", minimum=0
        ),
        line_count_optimized=_fast_int(
            row["line_count_optimized"], field=f"{context}.line_count_optimized", minimum=0
        ),
        tensor_shape=dims,
        tensor_dtype=tensor_dtype,
        min_speedup=float(min_speedup),
    )


def _fast_pair() -> tuple[str, str] | None:
    """Read the engine-published dynamic target/candidate pair, if any."""
    target = (os.environ.get(L3_TARGET_REQUEST_ENV) or "").strip()
    candidate = (os.environ.get(L3_CANDIDATE_REQUEST_ENV) or "").strip()
    if not target and not candidate:
        return None
    if not target or not candidate:
        raise BridgeError(
            f"{L3_TARGET_REQUEST_ENV} and {L3_CANDIDATE_REQUEST_ENV} "
            "must be supplied together"
        )
    return target, candidate


def _reject_parent_fast_request() -> None:
    """Refuse the retired parent-environment request channel."""
    if (os.environ.get(FAST_REQUEST_ENV) or "").strip():
        raise BridgeError(
            f"{FAST_REQUEST_ENV} is not read from the parent environment; submit "
            "--target fast/<request_id> with the request JSON as --candidate"
        )


def _fast_request_id(target: str) -> str:
    request_id = target[len("fast/"):] if target.startswith("fast/") else ""
    if not FAST_REQUEST_ID_RE.fullmatch(request_id):
        raise BridgeError(
            f"{L3_TARGET_REQUEST_ENV} must be fast/<32 lowercase hex>: {target!r}"
        )
    return request_id


def _fast_surrogate_caches(wroot: Path) -> dict[str, Path]:
    """Create the adapter-owned cache roots the fast child may write into."""
    base = wroot / "surrogate"
    caches = {
        "base": base,
        "tilelang": base / "tilelang",
        "reference": base / "reference",
    }
    for path in (base, caches["tilelang"], caches["reference"]):
        if path.is_symlink():
            raise BridgeError(f"surrogate cache path must not be a symlink: {path}")
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir() or not path.resolve().is_relative_to(wroot):
            raise BridgeError(f"surrogate cache path escapes the work root: {path}")
    return caches


def _load_fast_request(
    source: Path,
    values: dict[str, str],
    pair: tuple[str, str] | None,
) -> FastRequest | None:
    """Read and fully validate the V2-published typed fast request, if any."""
    if pair is None:
        return None
    target, candidate = pair
    request_id = _fast_request_id(target)
    build = build_subdir(values)
    expected_request = f"{build}/agent_gates/fast/{request_id}.request.json"
    if candidate != expected_request:
        raise BridgeError(
            f"{L3_CANDIDATE_REQUEST_ENV} must name the immutable request "
            f"{expected_request!r} for {target!r}: {candidate!r}"
        )
    relative_path, path = _fast_path(source, candidate, "path")
    raw = _fast_read_bytes(path, limit=FAST_REQUEST_MAX_BYTES, field="request")
    digest = hashlib.sha256(raw).hexdigest()
    payload = _fast_json_object(raw, field="request")
    mode = payload.get("mode")
    if mode not in FAST_REQUEST_MODES:
        raise BridgeError(
            f"fast request mode must be one of {', '.join(FAST_REQUEST_MODES)}: {mode!r}"
        )
    if mode == "batch":
        request = _fast_object(
            payload,
            context="request",
            required=(*FAST_REQUEST_COMMON_KEYS, "batches"),
            optional=("msprof_targets",),
        )
        if request.get("msprof_targets") not in (None, []):
            raise BridgeError("fast batch request must not declare msprof_targets")
    elif mode == "profile":
        request = _fast_object(
            payload,
            context="request",
            required=(*FAST_REQUEST_COMMON_KEYS, "msprof_targets"),
            optional=("batches",),
        )
        if request.get("batches") not in (None, []):
            raise BridgeError("fast profile request must not declare batches")
    else:
        request = _fast_object(
            payload,
            context="request",
            required=(*FAST_REQUEST_COMMON_KEYS, "probe", "candidates"),
        )
    if request["schema"] != FAST_REQUEST_SCHEMA:
        raise BridgeError(
            f"fast request schema must be {FAST_REQUEST_SCHEMA}: {request['schema']!r}"
        )
    if request["request_id"] != request_id:
        raise BridgeError(
            f"fast request request_id must equal the target request id {request_id}: "
            f"{request['request_id']!r}"
        )
    _fast_int(request["warmup"], field="warmup", minimum=1)
    _fast_int(request["repeat"], field="repeat", minimum=1)
    _fast_bool(request["require_all_passed"], field="require_all_passed")
    response_relative = f"{build}/agent_gates/fast/{request_id}.response.json"
    if request["response_path"] != response_relative:
        raise BridgeError(
            f"fast request response_path must be {response_relative!r}: "
            f"{request['response_path']!r}"
        )
    _, response_path = _fast_path(source, response_relative, "response_path")
    response_path.parent.mkdir(parents=True, exist_ok=True)
    if response_path.parent.is_symlink() or not response_path.parent.resolve().is_relative_to(source):
        raise BridgeError(
            f"fast request response_path escapes the source root: {response_relative}"
        )
    if response_path.exists():
        raise BridgeError(f"fast request response_path already exists: {response_relative}")

    candidates_prefix = f"{build}/agent_gates/fast/{request_id}/candidates"
    _, candidates_dir = _fast_path(source, candidates_prefix, "request candidates directory")
    if not candidates_dir.is_dir():
        raise BridgeError(f"fast request candidates directory is missing: {candidates_prefix}")

    candidates: list[FastCandidate] = []
    candidate_keys: set[tuple[str, str]] = set()
    if mode == "batch":
        raw_batches = request["batches"]
        if not isinstance(raw_batches, list) or not raw_batches:
            raise BridgeError("fast request batches must be a non-empty list")
        operators: set[str] = set()
        for index, raw_batch in enumerate(raw_batches):
            context = f"batches[{index}]"
            batch = _fast_object(
                raw_batch,
                context=context,
                required=("operator_name", "entrypoint", "candidates", "anchors"),
            )
            operator = _fast_name(batch["operator_name"], field=f"{context}.operator_name")
            if operator in operators:
                raise BridgeError(f"fast request duplicates operator {operator!r}")
            operators.add(operator)
            _fast_text(batch["entrypoint"], field=f"{context}.entrypoint")
            raw_anchors = batch["anchors"]
            if not isinstance(raw_anchors, list) or not raw_anchors:
                raise BridgeError(f"fast request {context}.anchors must be a non-empty list")
            for anchor_index, raw_anchor in enumerate(raw_anchors):
                _fast_anchor(raw_anchor, context=f"{context}.anchors[{anchor_index}]")
            raw_candidates = batch["candidates"]
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise BridgeError(f"fast request {context}.candidates must be a non-empty list")
            for candidate_index, raw_candidate in enumerate(raw_candidates):
                candidate_context = f"{context}.candidates[{candidate_index}]"
                row = _fast_candidate(
                    raw_candidate,
                    source=source,
                    candidates_prefix=candidates_prefix,
                    operator=operator,
                    context=candidate_context,
                )
                key = (operator, row.candidate_id)
                if key in candidate_keys:
                    raise BridgeError(
                        f"fast request duplicates candidate {operator}/{row.candidate_id}"
                    )
                candidate_keys.add(key)
                candidates.append(row)
        return FastRequest(
            request_id=request_id,
            mode=mode,
            path=path,
            relative_path=relative_path,
            sha256=digest,
            raw=raw,
            response_path=response_path,
            response_relative=response_relative,
            candidates=tuple(candidates),
            profile_cases=(),
        )

    if mode == "micro_probe":
        probe = _fast_probe(request["probe"], context="probe")
        raw_candidates = request["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) != len(
            FAST_PROBE_CANDIDATE_IDS
        ):
            raise BridgeError(
                "fast request micro_probe candidates must declare exactly "
                + " and ".join(FAST_PROBE_CANDIDATE_IDS)
            )
        probe_candidates: list[FastCandidate] = []
        for index, raw_candidate in enumerate(raw_candidates):
            context = f"candidates[{index}]"
            row = _fast_candidate(
                raw_candidate,
                source=source,
                candidates_prefix=candidates_prefix,
                operator="micro_probe",
                context=context,
            )
            if row.candidate_id not in FAST_PROBE_CANDIDATE_IDS:
                raise BridgeError(
                    f"fast request {context}.candidate_id must be one of "
                    + ", ".join(FAST_PROBE_CANDIDATE_IDS)
                    + f": {row.candidate_id!r}"
                )
            if stat.S_IMODE((source / row.relative_path).stat().st_mode) & 0o222:
                raise BridgeError(
                    f"fast request {context}.module_path must be read-only: "
                    f"{row.relative_path}"
                )
            probe_candidates.append(row)
        if {row.candidate_id for row in probe_candidates} != set(FAST_PROBE_CANDIDATE_IDS):
            raise BridgeError(
                "fast request micro_probe candidates must declare both baseline and optimized"
            )
        return FastRequest(
            request_id=request_id,
            mode=mode,
            path=path,
            relative_path=relative_path,
            sha256=digest,
            raw=raw,
            response_path=response_path,
            response_relative=response_relative,
            candidates=tuple(probe_candidates),
            profile_cases=(),
            probe=probe,
        )

    raw_targets = request["msprof_targets"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise BridgeError("fast request msprof_targets must be a non-empty list")
    candidates_by_key: dict[tuple[str, str], FastCandidate] = {}
    profile_cases: dict[tuple[str, str], list[int]] = {}
    for index, raw_target in enumerate(raw_targets):
        context = f"msprof_targets[{index}]"
        target_row = _fast_object(
            raw_target,
            context=context,
            required=(
                "operator_name",
                "candidate_id",
                "module_path",
                "sha256",
                "case_num",
                "warmup",
                "repeat",
            ),
        )
        operator = _fast_name(target_row["operator_name"], field=f"{context}.operator_name")
        candidate_id = _fast_name(
            target_row["candidate_id"], field=f"{context}.candidate_id"
        )
        row = _fast_candidate(
            {
                "candidate_id": candidate_id,
                "module_path": target_row["module_path"],
                "sha256": target_row["sha256"],
            },
            source=source,
            candidates_prefix=candidates_prefix,
            operator=operator,
            context=context,
        )
        key = (operator, candidate_id)
        previous = candidates_by_key.setdefault(key, row)
        if previous != row:
            raise BridgeError(
                f"fast request declares conflicting modules for {operator}/{candidate_id}"
            )
        case_num = _fast_int(target_row["case_num"], field=f"{context}.case_num", minimum=0)
        _fast_int(target_row["warmup"], field=f"{context}.warmup", minimum=1)
        _fast_int(target_row["repeat"], field=f"{context}.repeat", minimum=1)
        cases = profile_cases.setdefault(key, [])
        if case_num in cases:
            raise BridgeError(
                f"fast request duplicates msprof case {operator}/{candidate_id}/{case_num}"
            )
        cases.append(case_num)
    return FastRequest(
        request_id=request_id,
        mode=mode,
        path=path,
        relative_path=relative_path,
        sha256=digest,
        raw=raw,
        response_path=response_path,
        response_relative=response_relative,
        candidates=tuple(candidates_by_key.values()),
        profile_cases=tuple(
            (operator, candidate_id, tuple(sorted(cases)))
            for (operator, candidate_id), cases in profile_cases.items()
        ),
    )


def _fast_response_envelope(
    fast_request: FastRequest, payload: dict[str, object]
) -> tuple[str | None, list[str]]:
    """Require the response envelope to describe exactly this request."""
    failures: list[str] = []
    schema = payload.get("schema")
    if schema != FAST_RESPONSE_SCHEMA:
        failures.append(f"FAST_RESPONSE_SCHEMA:{schema!r}")
    if payload.get("request_id") != fast_request.request_id:
        failures.append("FAST_RESPONSE_REQUEST_ID")
    if payload.get("request_sha256") != fast_request.sha256:
        failures.append("FAST_RESPONSE_REQUEST_SHA")
    if payload.get("mode") != fast_request.mode:
        failures.append("FAST_RESPONSE_MODE")
    status = payload.get("status")
    if not isinstance(status, str) or status not in FAST_RESPONSE_STATUSES:
        failures.append(f"FAST_RESPONSE_STATUS_INVALID:{status!r}")
        return None, failures
    return status, failures


def _fast_response_bindings(
    fast_request: FastRequest, payload: dict[str, object]
) -> list[str]:
    """Require one pre/post SHA binding row for every declared candidate."""
    raw = payload.get("candidates")
    if not isinstance(raw, dict):
        return ["FAST_RESPONSE_BINDINGS_MISSING"]
    expected = {
        (candidate.operator, candidate.candidate_id): candidate
        for candidate in fast_request.candidates
    }
    rows: dict[tuple[str, str], object] = {}
    failures: list[str] = []
    for operator, row in raw.items():
        if not isinstance(operator, str) or not isinstance(row, dict):
            failures.append("FAST_RESPONSE_BINDINGS_INVALID")
            continue
        for candidate_id, item in row.items():
            if not isinstance(candidate_id, str):
                failures.append("FAST_RESPONSE_BINDINGS_INVALID")
                continue
            rows[(operator, candidate_id)] = item
    for operator, candidate_id in sorted(set(rows) - set(expected)):
        failures.append(f"FAST_RESPONSE_BINDING_EXTRA:{operator}:{candidate_id}")
    for key, candidate in expected.items():
        operator, candidate_id = key
        item = rows.get(key)
        if item is None:
            failures.append(f"FAST_RESPONSE_BINDING_MISSING:{operator}:{candidate_id}")
            continue
        if (
            not isinstance(item, dict)
            or set(item) != {"module_path", "sha256", "post_sha256"}
        ):
            failures.append(f"FAST_RESPONSE_BINDING_INVALID:{operator}:{candidate_id}")
            continue
        if item["module_path"] != candidate.relative_path:
            failures.append(f"FAST_RESPONSE_BINDING_PATH:{operator}:{candidate_id}")
        for field in ("sha256", "post_sha256"):
            value = item[field]
            if not isinstance(value, str) or value != candidate.sha256:
                failures.append(f"FAST_RESPONSE_BINDING_SHA:{operator}:{candidate_id}")
                break
    return failures


def _fast_response_coverage(
    fast_request: FastRequest, payload: dict[str, object]
) -> tuple[list[str], dict[str, object]]:
    """Require coverage counts that match the declared candidates and cases."""
    raw = payload.get("coverage")
    if not isinstance(raw, dict):
        return ["FAST_RESPONSE_COVERAGE_MISSING"], {}
    coverage: dict[str, object] = {}
    failures: list[str] = []
    for key in FAST_COVERAGE_KEYS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            failures.append(f"FAST_RESPONSE_COVERAGE_INVALID:{key}")
            continue
        coverage[key] = value
    if failures:
        return failures, {}
    expected_candidates = len(fast_request.candidates)
    expected_targets = sum(len(cases) for _, _, cases in fast_request.profile_cases)
    checks = {
        "expected_candidates": expected_candidates,
        "evaluated_candidates": expected_candidates,
        "expected_msprof_targets": expected_targets,
        "observed_msprof_targets": expected_targets,
    }
    for key, expected in checks.items():
        if coverage[key] != expected:
            failures.append(f"FAST_RESPONSE_COVERAGE_{key.upper()}")
    return failures, coverage


def _fast_response_micro_probe(
    fast_request: FastRequest, payload: dict[str, object]
) -> list[str]:
    """Require one probe result whose identity and metrics match the request."""
    probe = fast_request.probe
    assert probe is not None
    row = payload.get("micro_probe")
    if not isinstance(row, dict):
        return ["FAST_RESPONSE_MICRO_PROBE_MISSING"]
    if set(row) != {"probe_id", "metrics"}:
        return ["FAST_RESPONSE_MICRO_PROBE_INVALID"]
    failures: list[str] = []
    if row["probe_id"] != probe.probe_id:
        failures.append("FAST_RESPONSE_MICRO_PROBE_ID")
    metrics = row["metrics"]
    if not isinstance(metrics, dict) or not all(
        isinstance(key, str) for key in metrics
    ):
        return [*failures, "FAST_RESPONSE_MICRO_PROBE_METRICS_INVALID"]
    for key, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            failures.append(f"FAST_RESPONSE_MICRO_PROBE_METRIC_INVALID:{key}")
    for key in FAST_PROBE_METRIC_KEYS:
        if key not in metrics:
            failures.append(f"FAST_RESPONSE_MICRO_PROBE_METRIC_MISSING:{key}")
    return failures


def _fast_response_producers(
    fast_request: FastRequest, payload: dict[str, object]
) -> list[str]:
    """Require exactly the declared operators, candidates, and cases in the results."""
    failures: list[str] = []
    if fast_request.mode == "batch":
        operators = payload.get("operators")
        if not isinstance(operators, dict):
            return ["FAST_RESPONSE_OPERATORS_MISSING"]
        expected: dict[str, set[str]] = {}
        for candidate in fast_request.candidates:
            expected.setdefault(candidate.operator, set()).add(candidate.candidate_id)
        for operator in sorted(set(operators) - set(expected)):
            failures.append(f"FAST_RESPONSE_OPERATOR_EXTRA:{operator}")
        for operator, candidate_ids in expected.items():
            row = operators.get(operator)
            if not isinstance(row, dict):
                failures.append(f"FAST_RESPONSE_OPERATOR_MISSING:{operator}")
                continue
            for candidate_id in sorted(candidate_ids - set(row)):
                failures.append(
                    f"FAST_RESPONSE_CANDIDATE_MISSING:{operator}:{candidate_id}"
                )
            for candidate_id in sorted(set(row) - candidate_ids):
                failures.append(
                    f"FAST_RESPONSE_CANDIDATE_EXTRA:{operator}:{candidate_id}"
                )
        return failures

    if fast_request.mode == "micro_probe":
        return _fast_response_micro_probe(fast_request, payload)

    telemetry = payload.get("msprof_telemetry")
    if not isinstance(telemetry, dict):
        return ["FAST_RESPONSE_TELEMETRY_MISSING"]
    expected_cases: dict[str, dict[str, set[int]]] = {}
    for operator, candidate_id, cases in fast_request.profile_cases:
        expected_cases.setdefault(operator, {})[candidate_id] = set(cases)
    for operator in sorted(set(telemetry) - set(expected_cases)):
        failures.append(f"FAST_RESPONSE_OPERATOR_EXTRA:{operator}")
    for operator, candidate_map in expected_cases.items():
        row = telemetry.get(operator)
        if not isinstance(row, dict):
            failures.append(f"FAST_RESPONSE_OPERATOR_MISSING:{operator}")
            continue
        for candidate_id in sorted(set(row) - set(candidate_map)):
            failures.append(
                f"FAST_RESPONSE_CANDIDATE_EXTRA:{operator}:{candidate_id}"
            )
        for candidate_id, cases in candidate_map.items():
            case_row = row.get(candidate_id)
            if not isinstance(case_row, dict):
                failures.append(
                    f"FAST_RESPONSE_CANDIDATE_MISSING:{operator}:{candidate_id}"
                )
                continue
            actual_cases: set[int] = set()
            for raw_case in case_row:
                if not isinstance(raw_case, str) or not raw_case.isdecimal():
                    failures.append(
                        f"FAST_RESPONSE_CASE_INVALID:{operator}:{candidate_id}"
                    )
                    continue
                actual_cases.add(int(raw_case))
            for case_num in sorted(cases - actual_cases):
                failures.append(
                    f"FAST_RESPONSE_CASE_MISSING:{operator}:{candidate_id}:{case_num}"
                )
            for case_num in sorted(actual_cases - cases):
                failures.append(
                    f"FAST_RESPONSE_CASE_EXTRA:{operator}:{candidate_id}:{case_num}"
                )
    return failures


def _fast_response_check(
    fast_request: FastRequest, payload: dict[str, object]
) -> tuple[str | None, list[str], dict[str, object]]:
    """Validate one response; a request/infrastructure status is a block, not a verdict."""
    status, failures = _fast_response_envelope(fast_request, payload)
    if status is None:
        return None, failures, {}
    if status in {"request_error", "infrastructure_error"}:
        failures.append(f"FAST_RESPONSE_STATUS:{status}")
        return status, failures, {}
    failures.extend(_fast_response_bindings(fast_request, payload))
    coverage_failures, coverage = _fast_response_coverage(fast_request, payload)
    failures.extend(coverage_failures)
    failures.extend(_fast_response_producers(fast_request, payload))
    return status, failures, coverage


def _fast_verify_binding(
    fast_request: FastRequest, source: Path
) -> tuple[dict[str, object], list[dict[str, str]], list[str]]:
    """Re-hash the request, candidates, and response; bind them as gate artifacts."""
    failures: list[str] = []
    artifacts: dict[str, dict[str, str]] = {}

    def bind(path: Path, digest: str) -> None:
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return
        artifacts.setdefault(str(resolved), {"path": str(resolved), "sha256": digest})

    request_stable = False
    try:
        current = _fast_read_bytes(
            fast_request.path, limit=FAST_REQUEST_MAX_BYTES, field="request"
        )
        request_stable = hashlib.sha256(current).hexdigest() == fast_request.sha256
    except BridgeError:
        request_stable = False
    if request_stable:
        bind(fast_request.path, fast_request.sha256)
    else:
        failures.append("REQUEST_IDENTITY_CHANGED")

    candidate_rows: list[dict[str, object]] = []
    for candidate in fast_request.candidates:
        module_path = source / candidate.relative_path
        post_sha = ""
        try:
            _fast_path(source, candidate.relative_path, "candidate module_path")
            post_sha = _sha256_file(module_path)
        except (BridgeError, OSError):
            post_sha = ""
        stable = post_sha == candidate.sha256
        candidate_rows.append(
            {
                "operator": candidate.operator,
                "candidateId": candidate.candidate_id,
                "modulePath": candidate.relative_path,
                "preSha256": candidate.sha256,
                "postSha256": post_sha,
                "stable": stable,
            }
        )
        if stable:
            bind(module_path, candidate.sha256)
        else:
            failures.append(
                f"FAST_CANDIDATE_CHANGED:{candidate.operator}:{candidate.candidate_id}"
            )

    response_sha = ""
    response_bytes = 0
    response_bound = False
    response_status: str | None = None
    coverage: dict[str, object] = {}
    try:
        response_data = _fast_read_bytes(
            fast_request.response_path,
            limit=FAST_RESPONSE_MAX_BYTES,
            field="response",
        )
        response_sha = hashlib.sha256(response_data).hexdigest()
        response_bytes = len(response_data)
        bind(fast_request.response_path, response_sha)
        response_payload = _fast_json_object(response_data, field="response")
        response_status, response_failures, coverage = _fast_response_check(
            fast_request, response_payload
        )
        failures.extend(response_failures)
        response_bound = response_status is not None and not response_failures
    except BridgeError as exc:
        failures.append(f"FAST_RESPONSE_UNREADABLE:{exc}")
    observed: dict[str, object] = {
        "fastRequest": {
            "path": fast_request.relative_path,
            "sha256": fast_request.sha256,
            "requestId": fast_request.request_id,
            "mode": fast_request.mode,
        },
        "requestStable": request_stable,
        "fastCandidates": candidate_rows,
        "fastResponse": {
            "path": fast_request.response_relative,
            "sha256": response_sha,
            "bytes": response_bytes,
            "status": response_status,
            "bound": response_bound,
        },
        "fastCoverage": coverage or None,
    }
    return observed, list(artifacts.values()), failures


def _fast_failure_fields(
    binding_obs: dict[str, object],
    binding_failures: list[str],
    *,
    ok: bool,
    timed_out: bool,
) -> dict[str, object]:
    """Classify a fast outcome; only an explicit candidate status is a candidate failure."""
    if timed_out or binding_failures:
        category = "infra"
    else:
        response = binding_obs.get("fastResponse")
        status = response.get("status") if isinstance(response, dict) else None
        if status in {"request_error", "infrastructure_error"}:
            category = "infra"
        elif not ok:
            category = "candidate" if status == "candidate_failure" else "infra"
        else:
            category = None
    return {
        "errorCategory": category or "none",
        "failureCategory": category,
    }


def _junit_counts(path: Path) -> dict[str, int] | None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        return None
    collected = sum(int(s.attrib.get("tests", 0)) for s in suites)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
    failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    failed = errors + failures
    passed = max(0, collected - failed - skipped)
    return {
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }


def _cpu_observations(succeeded: bool, collected: int, passed: int, failed: int, skipped: int) -> dict:
    return {
        "exitCodeZero": bool(succeeded),
        "testsCollected": collected,
        "testsPassed": passed,
        "testsFailed": failed,
        "testsSkipped": skipped,
        "testsCollectedPositive": collected > 0,
        "notAllSkipped": collected > 0 and skipped < collected,
    }


def _device_occupancy_snapshot(
    device: str, *, raise_on_blocked_exclusive: bool = False
) -> tuple[str, list[dict[str, str]], int, int | None]:
    try:
        processes, npu_id = adapter_runtime.processes_on(int(device))
    except adapter_runtime.BlockedEnv as exc:
        if exclusive_requested():
            if raise_on_blocked_exclusive:
                raise BridgeError(str(exc)) from exc
            raise BridgeError(str(exc), category="occupancy", occupancy="unknown") from exc
        return "unknown", [], 0, None
    occ = "free" if not processes else f"shared({len(processes)})"
    total_mem_mb = 0
    for row in processes:
        if not isinstance(row, dict):
            continue
        try:
            total_mem_mb += int(row.get("memoryMb") or 0)
        except (TypeError, ValueError):
            pass
    return occ, processes, total_mem_mb, int(npu_id)


def _device_occupancy(device: str) -> str:
    occ, _, _, _ = _device_occupancy_snapshot(device, raise_on_blocked_exclusive=True)
    return occ


_RAW_DEVICE_OCCUPANCY = _device_occupancy


def _device_occupancy_details(device: str) -> tuple[str, list[dict[str, str]], int]:
    occ, processes, total_mem_mb, _ = _device_occupancy_snapshot(device)
    return occ, processes, total_mem_mb


OCCUPANCY_POLICIES = frozenset({"exclusive", "shared_quiet"})
SHARED_MAX_EXT_PROCS_DEFAULT = 6
SHARED_MAX_HBM_MB_DEFAULT = 16384


def _quarantine_path(lease_path: str | Path) -> Path:
    return Path(f"{lease_path}{QUARANTINE_SUFFIX}")


def _l4_idle_preflight(
    device: str,
    *,
    confirmations: int,
    occupancy_policy: str = "exclusive",
    max_ext_procs: int = SHARED_MAX_EXT_PROCS_DEFAULT,
    max_hbm_mb: int = SHARED_MAX_HBM_MB_DEFAULT,
) -> dict:
    """Refuse official L4 timing while disallowed occupancy holds the leased card."""
    streak = 0
    processes: list[dict[str, str]] = []
    total_mem_mb = 0
    npu_id = 0
    for attempt in range(1, confirmations + 1):
        try:
            processes, npu_id = adapter_runtime.processes_on(int(device))
        except adapter_runtime.BlockedEnv as exc:
            raise BridgeError(
                f"L4 idle preflight could not determine device occupancy: {exc}",
                category="occupancy",
                occupancy="unknown",
            ) from exc
        total_mem_mb = 0
        for row in processes:
            try:
                total_mem_mb += int(row.get("memoryMb") or 0)
            except (TypeError, ValueError):
                pass
        if occupancy_policy == "shared_quiet":
            if total_mem_mb <= max_hbm_mb and len(processes) <= max_ext_procs:
                return {
                    "npuId": npu_id,
                    "occupancy": "free" if not processes else f"shared({len(processes)})",
                    "policy": "shared_quiet",
                    "externalProcessCount": len(processes),
                    "externalMemoryMb": total_mem_mb,
                    "attempts": attempt,
                    "confirmations": confirmations,
                }
        elif not processes:
            return {
                "npuId": npu_id,
                "occupancy": "free",
                "attempts": attempt,
                "confirmations": confirmations,
            }
        streak += 1
        if streak < confirmations:
            time.sleep(HEALTH_PROBE_INTERVAL_SECONDS)
    summary = ", ".join(
        f"pid={row.get('pid')}({row.get('name')})" for row in processes[:6]
    )
    if occupancy_policy == "shared_quiet":
        raise BridgeError(
            "official L4 timing under shared_quiet exceeded budget: external process(es) "
            f"{summary} ({len(processes)} procs, {total_mem_mb}MB; limit {max_ext_procs} procs / "
            f"{max_hbm_mb}MB) remained on NPU {npu_id} for {streak} consecutive probes",
            category="occupancy",
            occupancy=f"shared({len(processes)})",
        )
    raise BridgeError(
        "official L4 timing requires an idle card: external process(es) "
        f"{summary} remained on NPU {npu_id} for {streak} consecutive probes",
        category="occupancy",
        occupancy=f"shared({len(processes)})",
    )


def _process_identity(pid: int) -> dict | None:
    """Snapshot a process identity that stays meaningful across pid reuse."""
    stat = _read_proc_stat(pid)
    if stat is None or stat["state"] == "Z":
        return None
    return {
        "pid": int(pid),
        "startTime": int(stat["startTime"]),
        "state": stat["state"],
        "comm": _read_proc_comm(pid),
    }


def _identity_status(identity: object) -> str:
    """Return alive/gone/unknown for a recorded process identity."""
    if not isinstance(identity, dict):
        return "unknown"
    pid = identity.get("pid")
    started = identity.get("startTime")
    if not isinstance(pid, int) or not isinstance(started, int):
        return "unknown"
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return "gone"
    except OSError:
        return "unknown"
    try:
        fields = raw.rsplit(")", 1)[1].split()
        state, start_time = fields[0], int(fields[19])
    except (IndexError, ValueError):
        return "unknown"
    if state == "Z":
        return "gone"
    if start_time != started:
        # The pid now belongs to an unrelated process; the recorded one exited.
        return "gone"
    return "alive"


def _read_quarantine(lease_path: str | Path) -> dict | None:
    """Check the per-card quarantine marker and clear it once every identity is gone."""
    path = _quarantine_path(lease_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        return {
            "path": str(path),
            "blocked": True,
            "reason": f"quarantine marker is unreadable: {exc}",
            "processes": [],
            "alive": [],
            "unknown": [],
        }
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        return {
            "path": str(path),
            "blocked": True,
            "reason": f"quarantine marker is not valid JSON: {exc}",
            "processes": [],
            "alive": [],
            "unknown": [],
        }
    if not isinstance(payload, dict) or payload.get("schema") != QUARANTINE_SCHEMA:
        return {
            "path": str(path),
            "blocked": True,
            "reason": "quarantine marker has an unexpected schema",
            "processes": [],
            "alive": [],
            "unknown": [],
        }
    processes = payload.get("processes")
    if not isinstance(processes, list) or not processes:
        # Without a process identity no successor can confirm that the residue
        # exited, so this stays blocked rather than guessing.
        return {
            "path": str(path),
            "blocked": True,
            "reason": "quarantine marker records no process identity",
            "payload": payload,
            "processes": [],
            "alive": [],
            "unknown": [],
        }
    alive: list[dict] = []
    unknown: list[dict] = []
    for item in processes:
        status = _identity_status(item)
        if status == "alive":
            alive.append(item)
        elif status == "unknown":
            unknown.append(item)
    if alive or unknown:
        return {
            "path": str(path),
            "blocked": True,
            "reason": payload.get("reason") or "an earlier run left device processes",
            "payload": payload,
            "processes": processes,
            "alive": alive,
            "unknown": unknown,
        }
    removed = True
    try:
        path.unlink()
    except (FileNotFoundError, ProcessLookupError):
        pass
    except OSError as exc:
        removed = False
        return {
            "path": str(path),
            "blocked": True,
            "reason": f"quarantine marker became unreadable while clearing: {exc}",
            "payload": payload,
            "processes": processes,
            "alive": [],
            "unknown": [],
            "removed": removed,
        }
    return {
        "path": str(path),
        "blocked": False,
        "cleared": True,
        "removed": removed,
        "payload": payload,
        "processes": processes,
        "alive": [],
        "unknown": [],
    }


def _quarantine_cleanup(
    cleanup: dict,
    *,
    lease_path: str | None,
    resource: str,
    device: str,
    token: str,
    phase: str | None,
    reason: str,
) -> dict:
    """Persist surviving device work before the lease is released.

    A successor must not reuse the same physical card while these identities
    are still alive or cannot be inspected. The marker is per-card because the
    lease path is per-card.
    """
    if not lease_path:
        return {"written": False, "skipped": "no device lease path in this run"}
    candidates: list[dict] = [
        item for item in (cleanup.get("survivingProcesses") or [])
        if isinstance(item, dict)
    ]
    known_pids = {item.get("pid") for item in candidates}
    for pid in cleanup.get("surviving") or []:
        if not isinstance(pid, int) or pid in known_pids:
            continue
        identity = _process_identity(pid)
        if identity is not None:
            candidates.append(identity)
    identities: list[dict] = []
    for item in candidates:
        status = _identity_status(item)
        if status in {"alive", "unknown"}:
            entry = dict(item)
            entry["status"] = status
            identities.append(entry)
    path = _quarantine_path(lease_path)
    if not identities:
        # Every survivor observed during cleanup is confirmed exited.
        cleanup["surviving"] = []
        return {"path": str(path), "written": False, "cleared": True, "processes": []}
    payload = {
        "schema": QUARANTINE_SCHEMA,
        "resource": resource,
        "device": device,
        "leasePath": str(lease_path),
        "token": token,
        "phase": phase,
        "reason": reason,
        "recordedAt": _timestamp(),
        "processes": identities,
        "cleanup": {
            "pids": cleanup.get("pids"),
            "sigterm": cleanup.get("sigterm"),
            "sigkill": cleanup.get("sigkill"),
            "seconds": cleanup.get("seconds"),
        },
    }
    record = {
        "path": str(path),
        "written": False,
        "schema": QUARANTINE_SCHEMA,
        "reason": reason,
        "phase": phase,
        "processes": identities,
    }
    shared = locking.is_shared_device_lock_root(path.parent)
    tmp_path: str | None = None
    try:
        locking.ensure_lock_dir(path.parent, shared=shared)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o666 if shared else 0o600)
        os.replace(tmp_path, path)
        tmp_path = None
        record["written"] = True
    except OSError as exc:
        record["writeError"] = f"{type(exc).__name__}: {exc}"
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return record


def _quarantine_block_message(lease_resource: str, quarantine: dict) -> str:
    unresolved = [
        f"pid={item.get('pid')}({item.get('comm') or '?'},{item.get('status', 'unknown')})"
        for item in (quarantine.get("alive") or []) + (quarantine.get("unknown") or [])
    ]
    raw_payload = quarantine.get("payload")
    payload: dict = raw_payload if isinstance(raw_payload, dict) else {}
    detail = f"reason={quarantine.get('reason')}"
    if payload.get("phase"):
        detail += f", phase={payload.get('phase')}"
    if payload.get("recordedAt"):
        detail += f", recordedAt={payload.get('recordedAt')}"
    if unresolved:
        detail += f", unresolved={', '.join(unresolved[:8])}"
    return (
        f"device lease {lease_resource} is quarantined by an earlier run "
        f"({detail}); marker {quarantine.get('path')}"
    )


@contextmanager
def _open_device_window(
    device: str,
    *,
    warmup_seconds: float | None = None,
    wait_seconds: float | None = None,
    require_idle: bool = False,
    occupancy_policy: str = "exclusive",
    max_ext_procs: int = SHARED_MAX_EXT_PROCS_DEFAULT,
    max_hbm_mb: int = SHARED_MAX_HBM_MB_DEFAULT,
    enforce_proc_count: bool = True,
):
    lease_path = os.environ.get("AGENT_GATES_DEVICE_LEASE_PATH", "").strip()
    lease_resource = os.environ.get("AGENT_GATES_DEVICE_LEASE_RESOURCE", f"npu:{device}").strip()
    if not lease_path:
        raise BridgeError("AGENT_GATES_DEVICE_LEASE_PATH is required for windowed device runs")
    if wait_seconds is None:
        try:
            wait_seconds = float(os.environ.get("AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS", "60"))
        except ValueError as exc:
            raise BridgeError("invalid AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS") from exc
    wait_seconds = max(0.0, float(wait_seconds))

    lease_file = Path(lease_path)
    shared = locking.is_shared_device_lock_root(lease_file.parent)
    fd = locking.open_lock_file(lease_file, shared=shared)
    started = time.monotonic()
    deadline = started + wait_seconds
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BridgeError(
                        f"device lease {lease_resource} did not open within {wait_seconds}s",
                        category="occupancy",
                        occupancy="unknown",
                    ) from exc
                time.sleep(min(0.05, remaining))

        # The marker is written by the previous lease holder before it releases
        # the flock. Checking it here, before occupancy and before any workload,
        # keeps residue on this physical card from being reused by a successor.
        quarantine = _read_quarantine(lease_path)
        if quarantine is not None and quarantine.get("blocked"):
            raise BridgeError(
                _quarantine_block_message(lease_resource, quarantine),
                category="occupancy",
                occupancy="quarantined",
            )

        opened = time.monotonic()
        window_started = _timestamp()
        ext_procs: list[dict[str, str]] = []
        ext_mem_mb = 0
        npu_id: int | None = None
        if occupancy_policy == "shared_quiet":
            occ, ext_procs, ext_mem_mb, npu_id = _device_occupancy_snapshot(device)
        elif _device_occupancy is _RAW_DEVICE_OCCUPANCY:
            occ, _, _, npu_id = _device_occupancy_snapshot(
                device, raise_on_blocked_exclusive=True
            )
        else:
            occ = _device_occupancy(device)
        probe_seconds = round(time.monotonic() - opened, 3)
        if exclusive_requested():
            if occ != "free":
                raise BridgeError(
                    f"device {lease_resource} is {occ} and TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE is set",
                    category="occupancy",
                    occupancy=occ,
                )
        elif occupancy_policy == "shared_quiet":
            if require_idle:
                if occ == "unknown":
                    raise BridgeError(
                        f"device {lease_resource} occupancy is unknown under shared_quiet",
                        category="occupancy",
                        occupancy="unknown",
                    )
                if ext_mem_mb > max_hbm_mb or (enforce_proc_count and len(ext_procs) > max_ext_procs):
                    raise BridgeError(
                        f"device {lease_resource} exceeds shared_quiet budget ({occ}, "
                        f"{ext_mem_mb}MB; limit {max_ext_procs} procs / {max_hbm_mb}MB)",
                        category="occupancy",
                        occupancy=occ,
                    )
        elif require_idle:
            if occ != "free":
                raise BridgeError(
                    f"device {lease_resource} is {occ} and requires an idle device",
                    category="occupancy",
                    occupancy=occ,
                )
        work_started = time.monotonic()

        def finalize_window() -> tuple[str, dict]:
            win = {
                "resource": lease_resource,
                "path": lease_path,
                "startedAt": window_started,
                "releasedAt": _timestamp(),
                "waited": round(opened - started, 3),
                "heldSeconds": round(time.monotonic() - work_started, 3),
                "probeSeconds": probe_seconds,
                "occupancy": occ,
            }
            if npu_id is not None:
                win["npuId"] = npu_id
            if warmup_seconds is not None:
                win["warmupSeconds"] = round(float(warmup_seconds), 3)
            if occupancy_policy != "exclusive":
                win["occupancyPolicy"] = occupancy_policy
                win["externalProcessCount"] = len(ext_procs)
                win["externalMemoryMb"] = ext_mem_mb
            elif require_idle:
                win["requiredIdle"] = True
            if quarantine is not None:
                win["quarantine"] = {
                    "path": quarantine.get("path"),
                    "cleared": bool(quarantine.get("cleared")),
                    "removed": quarantine.get("removed"),
                    "reason": quarantine.get("reason"),
                    "processes": quarantine.get("processes") or [],
                }
            return occ, win

        yield finalize_window
    finally:
        os.close(fd)


# Runtime health supervision. This is deliberately not another process
# supervisor: every host command still runs through adapter_runtime.run_host_bash
# and the engine-owned process supervisor remains the outer backstop. This layer
# only samples card health/occupancy, and on a confirmed fault signals the
# processes carrying this run's opaque token so the existing runner can reap its
# direct child and release the window. Liveness is not guessed from CPU time or
# log silence: a silent NPU kernel is left to the recipe deadline. An
# unreclaimable token process is recorded in a per-card quarantine marker so the
# next worker blocks before using that card.
HEALTH_TOKEN_ENV = "AGENT_GATES_RUN_HEALTH_TOKEN"
HEALTH_POLL_SECONDS = 0.5
HEALTH_PROBE_INTERVAL_SECONDS = 2.0
HEALTH_PROBE_TIMEOUT_SECONDS = 5.0
HEALTH_PROBE_FAILURE_LIMIT = 3
HEALTH_FATAL_CONFIRMATIONS = 2
HEALTH_EXTERNAL_CONFIRMATIONS = 2
L4_EXTERNAL_PROBE_LIMIT_DEFAULT = 3
HEALTH_MISSING_CONFIRMATIONS = 2
HEALTH_PID_RESCAN_SECONDS = 1.0
HEALTH_CLEANUP_GRACE_SECONDS = 2.0
HEALTH_CLEANUP_SECONDS = 10.0
HEALTH_WORKER_JOIN_SECONDS = 8.0
HEALTH_DEADLINE_SWEEP_GRACE_SECONDS = 0.25
HEALTH_DEADLINE_EXIT_SECONDS = 5.0
QUARANTINE_SUFFIX = ".quarantine.json"
QUARANTINE_SCHEMA = "agent-gates-device-quarantine/v1"

_PROC_MEM_LINE = re.compile(
    r"Process id\s*:\s*(\d+)\s+Process name\s*:\s*(.*?)\s+Process memory\(MB\)\s*:\s*(\d+)"
)
_PROC_MEM_NPU = re.compile(r"^\s*NPU ID\s*:\s*(\d+)\s*$", re.MULTILINE)
_HEALTH_STATUS = re.compile(r"Health(?:\s+Status)?\s*:\s*([A-Za-z][A-Za-z _-]*)", re.IGNORECASE)


def _bounded_host_command(argv: list[str], timeout: float) -> dict:
    """Run a small probe command with a hard wall-clock bound.

    The caller gets ``unreaped`` and ``wchan`` when even SIGKILL cannot make the
    probe exit (typically D-state in the driver). The probe process is put in its
    own session and is never allowed to become an unbounded wait.
    """
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        return {
            "argv": list(argv),
            "pid": None,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "seconds": round(time.monotonic() - started, 3),
            "timedOut": False,
            "unreaped": False,
            "wchan": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        stdout, stderr = proc.communicate(timeout=max(0.1, float(timeout)))
        return {
            "argv": list(argv),
            "pid": proc.pid,
            "returncode": proc.returncode,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "seconds": round(time.monotonic() - started, 3),
            "timedOut": False,
            "unreaped": False,
            "wchan": None,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        wchan = _read_proc_wchan(proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        unreaped = False
        stdout = stderr = ""
        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            unreaped = True
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
        return {
            "argv": list(argv),
            "pid": proc.pid,
            "returncode": proc.returncode,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "seconds": round(time.monotonic() - started, 3),
            "timedOut": True,
            "unreaped": unreaped,
            "wchan": wchan,
            "error": "probe timed out",
        }


def _bounded_smi_command(argv: list[str], timeout: float) -> dict:
    """Run a bounded npu-smi probe command, retrying transient exit code 215."""
    backoffs = adapter_runtime.SMI_RETRY_BACKOFF_SECONDS
    retry_codes = adapter_runtime.SMI_TRANSIENT_RETRY_CODES
    result: dict | None = None
    for attempt in range(len(backoffs) + 1):
        result = _bounded_host_command(argv, timeout)
        if (
            result["error"] is None
            and not result["timedOut"]
            and not result["unreaped"]
            and result["returncode"] in retry_codes
            and attempt < len(backoffs)
        ):
            time.sleep(backoffs[attempt])
            continue
        return result
    assert result is not None
    return result


def _npu_id_from_mapping(text: str, device: str) -> tuple[int | None, str | None]:
    matches: list[int] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 4 and all(field.isdigit() for field in fields[:4]):
            if int(fields[3]) == int(device):
                matches.append(int(fields[0]))
        elif len(fields) >= 3 and all(field.isdigit() for field in fields[:3]):
            if int(fields[2]) == int(device):
                matches.append(int(fields[0]))
    unique = sorted(set(matches))
    if not unique:
        return None, f"no Ascend NPU mapped for device index {device} in `npu-smi info -m`"
    if len(unique) > 1:
        return None, f"physical device {device} is claimed by multiple npu-smi NPU IDs {unique}"
    return unique[0], None


def _probe_npu_mapping(device: str, timeout: float) -> dict:
    smi = adapter_runtime.smi_command()
    result = _bounded_smi_command([smi, "info", "-m"], timeout)
    base = {
        "kind": "mapping",
        "command": f"{smi} info -m",
        "probePid": result["pid"],
        "seconds": result["seconds"],
        "timedOut": result["timedOut"],
        "unreaped": result["unreaped"],
        "wchan": result["wchan"],
        "returncode": result["returncode"],
        "npuId": None,
    }
    if result["error"] is not None or result["timedOut"] or result["unreaped"]:
        return {
            **base,
            "ok": False,
            "missing": False,
            "error": result["error"] or "mapping probe did not complete",
            "stderrTail": (result["stderr"] or "")[-500:],
        }
    if result["returncode"] != 0:
        return {
            **base,
            "ok": False,
            "missing": False,
            "error": f"mapping probe exited {result['returncode']}",
            "stderrTail": (result["stderr"] or result["stdout"] or "")[-500:],
        }
    npu_id, error = _npu_id_from_mapping(result["stdout"], device)
    missing = bool(npu_id is None and error and error.startswith("no Ascend NPU mapped"))
    return {
        **base,
        "ok": npu_id is not None,
        "missing": missing,
        "npuId": npu_id,
        "error": error,
    }


def _health_kind(value: object) -> str:
    if value is None:
        return "unknown"
    token = re.sub(r"[^A-Za-z_]+", "_", str(value).strip()).strip("_").upper()
    if token in {"OK", "NORMAL", "HEALTHY", "GOOD"}:
        return "ok"
    if token in {
        "ERROR", "FAULT", "FAULTY", "FAIL", "FAILED", "OFFLINE", "DOWN",
        "CRITICAL", "UNHEALTHY", "ABNORMAL", "NOT_OK", "NOTOK",
    }:
        return "fatal"
    return "unknown"


def _health_value_from_health_output(text: str) -> str | None:
    match = _HEALTH_STATUS.search(text)
    return match.group(1).strip() if match else None


def _health_value_from_info(text: str, npu_id: int) -> str | None:
    health: dict[int, str] = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) >= 4 and cells[1].isdigit() and cells[2].startswith("Ascend"):
            health[int(cells[1])] = cells[3]
            continue
        if len(cells) >= 3:
            match = re.fullmatch(r"(\d+)\s+\S+", cells[1])
            if match and cells[2]:
                health.setdefault(int(match.group(1)), cells[2])
    return health.get(npu_id)


_USAGE_AICORE_RATE = re.compile(r"Aicore\s+Usage\s+Rate\(\%\)\s*:\s*(\d+)", re.IGNORECASE)
_USAGE_HBM_BANDWIDTH_RATE = re.compile(
    r"HBM\s+Bandwidth\s+Usage\s+Rate\(\%\)\s*:\s*(\d+)", re.IGNORECASE
)


def _probe_npu_usages(
    npu_id: int, chip_id: int | None = None, timeout: float = 3.0
) -> dict[str, int]:
    """Probe AICore and HBM bandwidth usage percentages via npu-smi, degrading to {}."""
    smi = adapter_runtime.smi_command()
    base_cmd = [smi, "info", "-t", "usages", "-i", str(npu_id)]
    if chip_id is not None:
        result = _bounded_smi_command([*base_cmd, "-c", str(chip_id)], timeout)
        if result.get("error") is not None or result.get("timedOut") or result.get("unreaped"):
            return {}
        combined = f"{result.get('stdout') or ''}\n{result.get('stderr') or ''}"
        if result.get("returncode") != 0 or "does not support" in combined.lower():
            result = _bounded_smi_command(base_cmd, timeout)
    else:
        result = _bounded_smi_command(base_cmd, timeout)

    if (
        result.get("error") is not None
        or result.get("timedOut")
        or result.get("unreaped")
        or result.get("returncode") != 0
    ):
        return {}

    stdout = result.get("stdout") or ""
    usages: dict[str, int] = {}
    aicore_match = _USAGE_AICORE_RATE.search(stdout)
    if aicore_match is not None:
        usages["aicoreUsagePct"] = int(aicore_match.group(1))
    hbm_match = _USAGE_HBM_BANDWIDTH_RATE.search(stdout)
    if hbm_match is not None:
        usages["hbmBandwidthUsagePct"] = int(hbm_match.group(1))
    return usages


def _probe_device_health(npu_id: int, timeout: float) -> dict:
    smi = adapter_runtime.smi_command()
    command = [smi, "info", "-t", "health", "-i", str(npu_id)]
    result = _bounded_smi_command(command, timeout)
    base = {
        "kind": "health",
        "command": " ".join(command),
        "probePid": result["pid"],
        "seconds": result["seconds"],
        "timedOut": result["timedOut"],
        "unreaped": result["unreaped"],
        "wchan": result["wchan"],
        "returncode": result["returncode"],
        "healthValue": None,
        "healthKind": "unavailable",
        "source": "health",
    }
    if result["timedOut"] or result["unreaped"]:
        return {
            **base,
            "ok": False,
            "error": "device health probe timed out",
            "stderrTail": (result["stderr"] or "")[-500:],
        }
    if result["error"] is None and result["returncode"] == 0:
        value = _health_value_from_health_output(result["stdout"])
        if value is not None:
            return {
                **base,
                "ok": True,
                "healthValue": value,
                "healthKind": _health_kind(value),
            }

    # Older npu-smi builds do not support `-t health`; fall back to the health
    # table in the full dump. Missing/ambiguous rows remain unknown.
    fallback = _bounded_smi_command([smi, "info"], timeout)
    fallback_base = {
        **base,
        "command": f"{smi} info",
        "probePid": fallback["pid"],
        "source": "info-table",
        "seconds": round(result["seconds"] + fallback["seconds"], 3),
        "timedOut": fallback["timedOut"],
        "unreaped": fallback["unreaped"],
        "wchan": fallback["wchan"],
        "returncode": fallback["returncode"],
    }
    if fallback["error"] is not None or fallback["timedOut"] or fallback["unreaped"]:
        return {
            **fallback_base,
            "ok": False,
            "error": fallback["error"] or "device health probe did not complete",
            "stderrTail": (fallback["stderr"] or "")[-500:],
        }
    if fallback["returncode"] != 0:
        return {
            **fallback_base,
            "ok": False,
            "error": f"device health probe exited {fallback['returncode']}",
            "stderrTail": (fallback["stderr"] or fallback["stdout"] or "")[-500:],
        }
    value = _health_value_from_info(fallback["stdout"], npu_id)
    if value is None:
        return {
            **fallback_base,
            "ok": False,
            "error": f"device health table did not report NPU ID {npu_id}",
        }
    return {
        **fallback_base,
        "ok": True,
        "healthValue": value,
        "healthKind": _health_kind(value),
    }


def _probe_device_processes(npu_id: int, timeout: float) -> dict:
    smi = adapter_runtime.smi_command()
    command = [smi, "info", "-t", "proc-mem", "-i", str(npu_id)]
    result = _bounded_smi_command(command, timeout)
    base = {
        "kind": "occupancy",
        "command": " ".join(command),
        "probePid": result["pid"],
        "seconds": result["seconds"],
        "timedOut": result["timedOut"],
        "unreaped": result["unreaped"],
        "wchan": result["wchan"],
        "returncode": result["returncode"],
        "processTableSeen": False,
    }
    if result["error"] is not None or result["timedOut"] or result["unreaped"]:
        return {
            **base,
            "ok": False,
            "error": result["error"] or "occupancy probe did not complete",
            "stderrTail": (result["stderr"] or "")[-500:],
        }
    if result["returncode"] != 0:
        return {
            **base,
            "ok": False,
            "error": f"occupancy probe exited {result['returncode']}",
            "stderrTail": (result["stderr"] or result["stdout"] or "")[-500:],
        }
    if _PROC_MEM_NPU.search(result["stdout"]) is None:
        return {
            **base,
            "ok": False,
            "error": f"proc-mem output did not report NPU ID {npu_id}",
        }
    seen = "Process id" in result["stdout"] or "No process" in result["stdout"]
    processes = [
        {"pid": match.group(1), "name": match.group(2), "memoryMb": match.group(3)}
        for match in _PROC_MEM_LINE.finditer(result["stdout"])
    ]
    return {
        **base,
        "ok": seen,
        "processTableSeen": seen,
        "processes": processes,
        "error": None if seen else "proc-mem output had no process table",
    }


def _read_proc_stat(pid: int) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        fields = raw.rsplit(")", 1)[1].split()
        return {
            "state": fields[0],
            "startTime": int(fields[19]),
        }
    except (IndexError, ValueError):
        return None


def _read_proc_wchan(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/wchan").read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None


def _read_proc_comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _pid_has_token(pid: int, token: str) -> bool | None:
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError:
        return None
    needle = f"{HEALTH_TOKEN_ENV}={token}".encode("utf-8")
    return needle in environ.split(b"\0")


def _pid_owner_matches(pid: int) -> bool | None:
    try:
        return Path(f"/proc/{pid}").stat().st_uid == os.geteuid()
    except OSError:
        return None


def _scan_token_pids(token: str) -> set[int]:
    found: set[int] = set()
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return found
    own_pid = os.getpid()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own_pid:
            continue
        try:
            if entry.stat().st_uid != os.geteuid():
                continue
        except OSError:
            continue
        if _pid_has_token(pid, token) is True:
            found.add(pid)
    return found


def _classify_device_processes(processes: list[dict], token: str) -> dict:
    own: list[int] = []
    external: list[int] = []
    unknown: list[int] = []
    stale: list[int] = []
    for row in processes:
        try:
            pid = int(row.get("pid", ""))
        except (TypeError, ValueError):
            continue
        if pid == os.getpid():
            continue
        stat = _read_proc_stat(pid)
        if stat is None:
            stale.append(pid)
            continue
        if stat["state"] == "Z":
            continue
        if _read_proc_comm(pid) in {"npu-smi", "npu_smi"}:
            # A concurrent SMI query is not a device workload.
            continue
        has_token = _pid_has_token(pid, token)
        if has_token is True:
            own.append(pid)
        elif has_token is False:
            external.append(pid)
        elif _pid_owner_matches(pid) is False:
            external.append(pid)
        else:
            unknown.append(pid)
    return {
        "own": sorted(set(own)),
        "external": sorted(set(external)),
        "unknown": sorted(set(unknown)),
        "stale": sorted(set(stale)),
    }


def _kill_token_processes(token: str) -> dict:
    """Signal only processes carrying this run's token, bounded and targeted."""
    started = time.monotonic()
    seen: set[int] = set()
    sigterm: set[int] = set()
    sigkill: set[int] = set()

    def _alive() -> list[int]:
        alive: list[int] = []
        for pid in sorted(seen):
            stat = _read_proc_stat(pid)
            if stat is None or stat["state"] == "Z":
                continue
            if _pid_has_token(pid, token) is not True:
                continue
            alive.append(pid)
        return alive

    grace_end = started + max(0.0, HEALTH_CLEANUP_GRACE_SECONDS)
    while True:
        seen.update(_scan_token_pids(token))
        alive = _alive()
        if not alive:
            break
        now = time.monotonic()
        if now >= grace_end:
            break
        for pid in alive:
            try:
                os.kill(pid, signal.SIGTERM)
                sigterm.add(pid)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.1)

    kill_end = time.monotonic() + max(0.0, HEALTH_CLEANUP_SECONDS)
    while True:
        seen.update(_scan_token_pids(token))
        alive = _alive()
        if not alive:
            break
        now = time.monotonic()
        if now >= kill_end:
            break
        for pid in alive:
            try:
                os.kill(pid, signal.SIGKILL)
                sigkill.add(pid)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.1)

    surviving = _alive()
    return {
        "attempted": bool(seen),
        "seconds": round(time.monotonic() - started, 3),
        "pids": sorted(seen)[:20],
        "sigterm": sorted(sigterm)[:20],
        "sigkill": sorted(sigkill)[:20],
        "surviving": surviving[:20],
        "survivingProcesses": [
            identity
            for pid in surviving[:20]
            if (identity := _process_identity(pid)) is not None
        ],
        "incomplete": bool(surviving),
    }


def _retire_stuck_probe(pid: int) -> None:
    """Reap our own exited probe child without ever blocking on a D-state probe."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


class _RunHealthMonitor:
    """Sample one run's card health and occupancy without owning it."""

    def __init__(
        self,
        *,
        device: str,
        mode: str,
        token: str,
        phase_getter: Callable[[], str | None] | None = None,
        external_limit: int | None = None,
        occupancy_policy: str = "exclusive",
        shared_max_ext_procs: int = SHARED_MAX_EXT_PROCS_DEFAULT,
        shared_max_hbm_mb: int = SHARED_MAX_HBM_MB_DEFAULT,
        allow_external_in_shared: bool = False,
    ) -> None:
        self.device = device
        self.mode = mode
        self.token = token
        self.phase_getter = phase_getter
        self.occupancy_policy = occupancy_policy
        self.shared_max_ext_procs = max(1, int(shared_max_ext_procs))
        self.shared_max_hbm_mb = max(256, int(shared_max_hbm_mb))
        self.allow_external_in_shared = bool(allow_external_in_shared)
        self.started = time.monotonic()
        self.probe_interval = max(0.01, float(HEALTH_PROBE_INTERVAL_SECONDS))
        self.probe_timeout = max(0.05, float(HEALTH_PROBE_TIMEOUT_SECONDS))
        self.failure_limit = max(1, int(HEALTH_PROBE_FAILURE_LIMIT))
        self.fatal_limit = max(1, int(HEALTH_FATAL_CONFIRMATIONS))
        self.external_limit = max(
            1,
            int(
                HEALTH_EXTERNAL_CONFIRMATIONS
                if external_limit is None
                else external_limit
            ),
        )
        self.missing_limit = max(1, int(HEALTH_MISSING_CONFIRMATIONS))
        self.started_at = _timestamp()
        self.last_scan = 0.0
        self.known_pids: set[int] = set()
        self.last_probe_at = 0.0
        self.last_probe: dict | None = None
        self.last_mapping_check: dict | None = None
        self.npu_id: int | None = None
        self.health = "not_probed"
        self.health_unknown = False
        self.probe_failures = 0
        self.occupancy_failures = 0
        self.fatal_streak = 0
        self.non_ok_streak = 0
        self.missing_streak = 0
        self.external_streak = 0
        self.occupancy = "not_checked"
        self.own_pids: list[int] = []
        self.external_pids: list[int] = []
        self.unknown_pids: list[int] = []
        self.stuck_probes: dict[int, dict] = {}
        self.stuck_probe_history: set[int] = set()
        self.unknown_signals: set[str] = set()
        if mode == "l4":
            # L4's internal device-fault counters are not exposed by the host
            # SMI interface; record that limitation instead of inferring health.
            self.unknown_signals.add("l4_internal_fault_counters_unavailable")
        self.abort: dict | None = None
        self.cleanup_result: dict | None = None
        self.outcome = "running"

    def _phase_name(self) -> str | None:
        if self.phase_getter is None:
            return None
        try:
            return self.phase_getter()
        except Exception:
            return None

    def _refresh_processes(self, now: float) -> None:
        if now - self.last_scan >= max(0.05, float(HEALTH_PID_RESCAN_SECONDS)):
            current = _scan_token_pids(self.token)
            self.last_scan = now
        else:
            current = {
                pid for pid in self.known_pids
                if _pid_has_token(pid, self.token) is True
            }
        self.known_pids = current

    def _abort_run(
        self,
        kind: str,
        reason: str,
        *,
        occupancy: str | None = None,
        health_unknown: bool = False,
    ) -> dict:
        if self.abort is None:
            self.abort = {
                "kind": kind,
                "reason": reason,
                "at": _timestamp(),
                "phase": self._phase_name(),
                "device": self.device,
                "npuId": self.npu_id,
                "occupancy": occupancy,
                "healthUnknown": bool(health_unknown),
                "stuckProbePids": sorted(self.stuck_probes)[:20],
            }
            self.health_unknown = self.health_unknown or health_unknown
        return self.abort

    def _note_stuck_probe(self, probe: dict) -> None:
        """Track a probe that survived SIGKILL so new probes are not stacked on it."""
        if not probe.get("unreaped"):
            return
        pid = probe.get("probePid")
        if not isinstance(pid, int):
            return
        identity = _process_identity(pid)
        if identity is None:
            # The probe exited (or is a zombie) between SIGKILL and this check.
            _retire_stuck_probe(pid)
            return
        self.stuck_probes[pid] = identity
        self.stuck_probe_history.add(pid)
        self.unknown_signals.add("device_probe_unreaped")

    def _stuck_probe_alive(self) -> bool:
        active: dict[int, dict] = {}
        for pid, identity in self.stuck_probes.items():
            status = _identity_status(identity)
            if status in {"alive", "unknown"}:
                # Indeterminate is treated like alive: never stack more probes
                # on a driver that may be wedged.
                active[pid] = identity
            else:
                _retire_stuck_probe(pid)
        self.stuck_probes = active
        return bool(active)

    def _count_probe_failure(self, reason: str) -> None:
        self.probe_failures += 1
        if self.probe_failures >= self.failure_limit:
            self._abort_run(
                "probe_unavailable",
                f"device probe unavailable for {self.probe_failures} "
                f"consecutive attempts: {reason}",
                health_unknown=True,
            )

    def _note_missing_mapping(self, *, detail: str) -> None:
        self.missing_streak += 1
        self.probe_failures = 0
        self.health = "missing"
        if self.missing_streak >= self.missing_limit:
            self._abort_run(
                "device_missing",
                f"target device {self.device} is no longer mapped by "
                f"`npu-smi info -m` on {self.missing_streak} consecutive probes "
                f"({detail}); this run's work cannot be scheduled there",
                occupancy="missing",
            )

    def _recheck_mapping_after_failure(self) -> bool:
        """Distinguish a disappeared card from an unreadable health probe."""
        mapping = _probe_npu_mapping(self.device, self.probe_timeout)
        self.last_mapping_check = mapping
        self._note_stuck_probe(mapping)
        if mapping.get("ok"):
            self.npu_id = int(mapping["npuId"])
            self.missing_streak = 0
            return False
        if mapping.get("missing"):
            self._note_missing_mapping(detail=str(mapping.get("error")))
            return True
        self.missing_streak = 0
        return False

    def _probe(self) -> None:
        probe_timeout = self.probe_timeout
        if self._stuck_probe_alive():
            self.last_probe = {
                "at": _timestamp(),
                "kind": "probe_stuck",
                "probePids": sorted(self.stuck_probes)[:20],
                "error": "a previous probe did not exit after SIGKILL; "
                "new probes are suppressed",
            }
            self.health = "unknown"
            self._count_probe_failure(
                f"probe process(es) {sorted(self.stuck_probes)[:8]} did not exit"
            )
            return

        if self.npu_id is None:
            mapping = _probe_npu_mapping(self.device, probe_timeout)
            self.last_mapping_check = mapping
            self._note_stuck_probe(mapping)
            if not mapping.get("ok"):
                self.last_probe = {
                    "at": _timestamp(),
                    "npuId": None,
                    "healthKind": "missing" if mapping.get("missing") else "unavailable",
                    "mapping": mapping,
                    "error": mapping.get("error"),
                }
                if mapping.get("missing"):
                    self._note_missing_mapping(detail=str(mapping.get("error")))
                else:
                    self.missing_streak = 0
                    self._count_probe_failure(str(mapping.get("error")))
                return
            self.npu_id = int(mapping["npuId"])
            self.missing_streak = 0

        health = _probe_device_health(self.npu_id, probe_timeout)
        self._note_stuck_probe(health)
        row: dict = {
            "at": _timestamp(),
            "npuId": self.npu_id,
            "health": health,
        }
        self.last_probe = row
        kind = health.get("healthKind")
        if kind == "ok":
            self.probe_failures = 0
            self.fatal_streak = 0
            self.non_ok_streak = 0
            self.missing_streak = 0
            self.health = "ok"
        elif kind == "fatal":
            self.fatal_streak += 1
            self.probe_failures = 0
            self.non_ok_streak = 0
            self.missing_streak = 0
            self.health = "fatal"
            if self.fatal_streak >= self.fatal_limit:
                self._abort_run(
                    "device_error",
                    "device health reports "
                    f"{health.get('healthValue')!r} for NPU {self.npu_id} "
                    f"on {self.fatal_streak} consecutive probes",
                )
                return
        elif health.get("ok") and health.get("healthValue") is not None:
            # Parsed but not a recognised healthy value (e.g. Warning). Record
            # it as non-OK; it is not a confirmed fatal fault and never a
            # reason to kill a possibly long-running kernel.
            self.probe_failures = 0
            self.fatal_streak = 0
            self.missing_streak = 0
            self.non_ok_streak += 1
            self.health = "non_ok"
            self.unknown_signals.add("device_health_non_ok")
        else:
            self.fatal_streak = 0
            self.non_ok_streak = 0
            self.health = "unknown"
            if health.get("unreaped"):
                # Do not launch another probe against a wedged driver; the
                # stuck-probe check at the top of the next interval takes over.
                self._count_probe_failure(str(health.get("error")))
                return
            if self._recheck_mapping_after_failure():
                return
            self._count_probe_failure(
                str(health.get("error") or health.get("healthValue"))
            )
            return

        occupancy = _probe_device_processes(self.npu_id, probe_timeout)
        self._note_stuck_probe(occupancy)
        row["occupancy"] = occupancy
        if not occupancy.get("ok"):
            self.occupancy = "unknown"
            self.occupancy_failures += 1
            self.unknown_signals.add("device_occupancy_unavailable")
            if self.mode == "l4" and self.occupancy_failures >= self.failure_limit:
                self._abort_run(
                    "occupancy_unknown",
                    "L4 occupancy could not be observed for "
                    f"{self.occupancy_failures} consecutive attempts: "
                    f"{occupancy.get('error')}",
                    occupancy="unknown",
                    health_unknown=True,
                )
            return

        self.occupancy_failures = 0
        classes = _classify_device_processes(
            list(occupancy.get("processes") or []), self.token
        )
        self.own_pids = classes["own"]
        self.external_pids = classes["external"]
        self.unknown_pids = classes["unknown"]
        if self.unknown_pids:
            self.unknown_signals.add("device_process_environ_unreadable")
        if self.external_pids:
            self.occupancy = f"shared({len(self.external_pids)})"
        elif self.unknown_pids:
            self.occupancy = "unknown"
        elif self.own_pids:
            self.occupancy = "own"
        else:
            self.occupancy = "free"

        if self.external_pids:
            if self.occupancy_policy == "shared_quiet":
                if self.allow_external_in_shared:
                    self.external_streak = 0
                else:
                    ext_pid_strs = {str(pid) for pid in self.external_pids}
                    ext_mem_mb = 0
                    for proc_row in occupancy.get("processes") or []:
                        if str(proc_row.get("pid") or "").strip() in ext_pid_strs:
                            try:
                                ext_mem_mb += int(proc_row.get("memoryMb") or 0)
                            except (TypeError, ValueError):
                                pass
                    if (
                        ext_mem_mb > self.shared_max_hbm_mb
                        or len(self.external_pids) > self.shared_max_ext_procs + 4
                    ):
                        self.external_streak += 1
                        if self.mode in {"fast", "l3", "l4"} and self.external_streak >= self.external_limit:
                            self._abort_run(
                                "external_occupancy",
                                f"{self.mode.upper()} under shared_quiet exceeded heavy contention "
                                f"limit ({len(self.external_pids)} procs, {ext_mem_mb}MB > "
                                f"{self.shared_max_ext_procs + 4} procs / {self.shared_max_hbm_mb}MB) "
                                f"on NPU {self.npu_id} for {self.external_streak} consecutive probes",
                                occupancy=f"shared({len(self.external_pids)})",
                            )
                    else:
                        self.external_streak = 0
            else:
                self.external_streak += 1
                if self.mode in {"fast", "l3", "l4"} and self.external_streak >= self.external_limit:
                    self._abort_run(
                        "external_occupancy",
                        f"{self.mode.upper()} lost exclusivity after the lease opened: external device "
                        f"process(es) {self.external_pids[:8]} remained on NPU {self.npu_id} "
                        f"for {self.external_streak} consecutive probes",
                        occupancy=f"shared({len(self.external_pids)})",
                    )
        else:
            self.external_streak = 0

    def poll(self) -> dict | None:
        now = time.monotonic()
        self._refresh_processes(now)
        if self.known_pids and now - self.last_probe_at >= self.probe_interval:
            self.last_probe_at = now
            self._probe()
        return self.abort

    def cleanup_probes(self) -> dict:
        """Stop and reap our own stuck probes; never touches workload processes."""
        started = time.monotonic()
        pids = sorted(self.stuck_probes)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = started + 1.0
        while time.monotonic() < deadline:
            if not self._stuck_probe_alive():
                break
            time.sleep(0.05)
        remaining = sorted(self.stuck_probes)
        for pid in remaining:
            _retire_stuck_probe(pid)
        return {
            "pids": pids[:20],
            "remaining": remaining[:20],
            "seconds": round(time.monotonic() - started, 3),
        }

    def cleanup(self, *, aborted: bool) -> dict:
        result = _kill_token_processes(self.token)
        result["probeCleanup"] = self.cleanup_probes()
        self.cleanup_result = result
        if aborted:
            self.outcome = "aborted"
        elif result.get("surviving"):
            self.outcome = "cleanup_incomplete"
        return result

    def finish(self, outcome: str, *, force: bool = False) -> None:
        if force or self.outcome == "running":
            self.outcome = outcome

    def snapshot(self) -> dict:
        data: dict = {
            "enabled": True,
            "mode": self.mode,
            "device": self.device,
            "npuId": self.npu_id,
            "startedAt": self.started_at,
            "probeIntervalSeconds": self.probe_interval,
            "probeTimeoutSeconds": self.probe_timeout,
            "health": self.health,
            "healthUnknown": bool(self.health_unknown),
            "occupancy": self.occupancy,
            "probeFailures": self.probe_failures,
            "occupancyFailures": self.occupancy_failures,
            "fatalHealthStreak": self.fatal_streak,
            "nonOkHealthStreak": self.non_ok_streak,
            "missingMappingStreak": self.missing_streak,
            "externalOccupancyStreak": self.external_streak,
            "ownProcessCount": len(self.known_pids),
            "ownPids": sorted(self.known_pids)[:20],
            "externalProcessCount": len(self.external_pids),
            "externalPids": list(self.external_pids)[:20],
            "unknownProcessCount": len(self.unknown_pids),
            "unknownPids": list(self.unknown_pids)[:20],
            "stuckProbePids": sorted(self.stuck_probes)[:20],
            "stuckProbePidsObserved": sorted(self.stuck_probe_history)[:20],
            "lastProbe": self.last_probe,
            "lastMappingCheck": self.last_mapping_check,
            "unknownSignals": sorted(self.unknown_signals),
            "outcome": self.outcome,
        }
        if self.abort is not None:
            data["abort"] = dict(self.abort)
        if self.cleanup_result is not None:
            data["cleanup"] = dict(self.cleanup_result)
        return data


def _run_host_bash_monitored(
    command: str,
    *,
    mode: str,
    device: str,
    monitor_sink: dict,
    phase_getter: Callable[[], str | None] | None = None,
    lease_path: str | None = None,
    external_probe_limit: int | None = None,
    occupancy_policy: str = "exclusive",
    shared_max_ext_procs: int = SHARED_MAX_EXT_PROCS_DEFAULT,
    shared_max_hbm_mb: int = SHARED_MAX_HBM_MB_DEFAULT,
    allow_external_in_shared: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess[str]:
    """Run one existing host command while sampling this run's health.

    The command and its timeout/cleanup semantics are unchanged: it still runs
    through adapter_runtime.run_host_bash. The monitor only adds (a) bounded
    health/occupancy probes and (b) a targeted signal sweep over processes that
    inherited this run's opaque token when a confirmed fault must end the run.
    Surrendered device work that survives that sweep is recorded in the
    per-card quarantine marker before the caller releases the lease.
    """
    token = secrets.token_hex(16)
    child_env = dict(kwargs.pop("env", None) or os.environ)
    child_env[HEALTH_TOKEN_ENV] = token
    kwargs["env"] = child_env
    monitor = _RunHealthMonitor(
        device=device,
        mode=mode,
        token=token,
        phase_getter=phase_getter,
        external_limit=external_probe_limit,
        occupancy_policy=occupancy_policy,
        shared_max_ext_procs=shared_max_ext_procs,
        shared_max_hbm_mb=shared_max_hbm_mb,
        allow_external_in_shared=allow_external_in_shared,
    )
    resolved_lease = lease_path
    if resolved_lease is None:
        resolved_lease = os.environ.get("AGENT_GATES_DEVICE_LEASE_PATH", "").strip() or None
    lease_resource = os.environ.get(
        "AGENT_GATES_DEVICE_LEASE_RESOURCE", f"npu:{device}"
    ).strip()
    result: dict = {}
    started = time.monotonic()
    run_timeout = kwargs.get("timeout")
    deadline = (
        started + float(run_timeout)
        if isinstance(run_timeout, (int, float)) and float(run_timeout) > 0
        else None
    )
    deadline_sweep: dict | None = None
    deadline_exit_at: float | None = None
    deadline_unreapable = False

    def _runner() -> None:
        try:
            result["completed"] = adapter_runtime.run_host_bash(command, **kwargs)
        except BaseException as exc:  # re-raised in the calling thread
            result["error"] = exc

    thread = threading.Thread(
        target=_runner,
        name=f"tilefoundry-health-run-{os.getpid()}",
        daemon=True,
    )
    thread.start()

    def _cleanup(aborted: bool, reason: str) -> dict:
        if monitor.cleanup_result is not None:
            return monitor.cleanup_result
        cleanup = monitor.cleanup(aborted=aborted)
        cleanup["workerJoined"] = not thread.is_alive()
        cleanup["quarantine"] = _quarantine_cleanup(
            cleanup,
            lease_path=resolved_lease,
            resource=lease_resource,
            device=device,
            token=token,
            phase=monitor._phase_name(),
            reason=reason,
        )
        if deadline_sweep is not None:
            cleanup["deadlineSweep"] = deadline_sweep
        monitor.cleanup_result = cleanup
        return cleanup

    def _quarantine_note(cleanup: dict) -> str:
        quarantine = cleanup.get("quarantine") or {}
        if quarantine.get("written"):
            return (
                f"; card quarantined for successors: {quarantine.get('path')} "
                f"(reason={quarantine.get('reason')})"
            )
        if quarantine.get("writeError"):
            return f"; quarantine marker could not be written: {quarantine.get('writeError')}"
        return ""

    try:
        while True:
            thread.join(timeout=max(0.01, float(HEALTH_POLL_SECONDS)))
            if not thread.is_alive():
                break
            if (
                deadline is not None
                and deadline_sweep is None
                and time.monotonic()
                >= deadline + max(0.0, float(HEALTH_DEADLINE_SWEEP_GRACE_SECONDS))
            ):
                # run_host_bash has already applied its own timeout. An escaped
                # descendant can still hold the stdout/stderr pipes open and
                # make its final communicate() wait for that descendant, so the
                # recipe deadline alone would not bound this call. Sweep only
                # processes carrying this run's opaque token; this is the same
                # bounded cleanup, applied at the deadline instead of after it.
                deadline_sweep = _kill_token_processes(token)
                deadline_exit_at = (
                    time.monotonic()
                    + max(0.0, float(HEALTH_DEADLINE_EXIT_SECONDS))
                )
            if (
                deadline_exit_at is not None
                and thread.is_alive()
                and time.monotonic() >= deadline_exit_at
            ):
                # A D-state descendant can survive SIGKILL and keep the runner
                # blocked on its inherited pipes. Do not wait for the recipe's
                # outer bound: return a bounded result and fence the card with
                # the surviving identities recorded by the quarantine marker.
                abort_reason = (
                    "recipe deadline exceeded and this run's process cleanup "
                    "could not complete within "
                    f"{HEALTH_DEADLINE_EXIT_SECONDS}s after the deadline sweep"
                )
                cleanup = _cleanup(True, abort_reason)
                cleanup["workerJoined"] = False
                monitor.finish("deadline_unreclaimable", force=True)
                monitor_sink.update(monitor.snapshot())
                quarantine = cleanup.get("quarantine") or {}
                deadline_unreapable = True
                raise BridgeError(
                    f"host command did not exit within {run_timeout}s and its "
                    f"remaining processes could not be reaped: "
                    f"{cleanup.get('surviving')}"
                    + _quarantine_note(cleanup),
                    category="infra",
                    occupancy=(
                        "quarantined" if quarantine.get("written") else None
                    ),
                )
            if monitor.poll() is not None:
                if not thread.is_alive():
                    break
                abort = monitor.abort or {}
                cleanup = _cleanup(
                    True,
                    f"runtime health abort ({abort.get('kind')}) in phase "
                    f"{abort.get('phase') or 'unknown'}",
                )
                thread.join(timeout=max(0.0, float(HEALTH_WORKER_JOIN_SECONDS)))
                cleanup["workerJoined"] = not thread.is_alive()
                monitor_sink.update(monitor.snapshot())
                completed = result.get("completed")
                tail = ""
                if completed is not None:
                    tail = (getattr(completed, "stderr", "") or "")[-1500:]
                category = (
                    "occupancy"
                    if abort.get("kind") in {"external_occupancy", "occupancy_unknown"}
                    else "environment"
                )
                message = (
                    f"runtime health abort ({abort.get('kind')}) in phase "
                    f"{abort.get('phase') or 'unknown'} on device {device} "
                    f"(npu {monitor.npu_id}): {abort.get('reason')}"
                )
                if cleanup.get("surviving"):
                    message += (
                        f"; process cleanup incomplete: {cleanup.get('surviving')}"
                    )
                message += _quarantine_note(cleanup)
                if tail.strip():
                    message += f"; last stderr: {tail.strip()}"
                raise BridgeError(
                    message,
                    category=category,
                    occupancy=abort.get("occupancy"),
                )
    except KeyboardInterrupt:
        _cleanup(True, "interrupted by KeyboardInterrupt")
        thread.join(timeout=max(0.0, float(HEALTH_WORKER_JOIN_SECONDS)))
        monitor_sink.update(monitor.snapshot())
        raise
    except BaseException:
        # The explicit abort below raises BridgeError from inside this try;
        # its cleanup result must not be replaced by an empty second sweep.
        _cleanup(True, "run ended by an exception")
        if not deadline_unreapable:
            thread.join(timeout=max(0.0, float(HEALTH_WORKER_JOIN_SECONDS)))
        monitor_sink.update(monitor.snapshot())
        raise

    if "error" in result:
        error = result["error"]
        if not isinstance(error, KeyboardInterrupt):
            cleanup = _cleanup(False, f"runner error ({type(error).__name__})")
            cleanup["workerJoined"] = not thread.is_alive()
        monitor.finish("runner_error")
        monitor_sink.update(monitor.snapshot())
        raise error
    completed = result["completed"]
    if mode in {"fast", "l4"} and monitor.known_pids:
        monitor._probe()
        if monitor.abort is not None and monitor.abort.get("kind") == "external_occupancy":
            abort = monitor.abort
            cleanup = _cleanup(True, f"post-run intrusion fence ({abort.get('kind')})")
            monitor_sink.update(monitor.snapshot())
            raise BridgeError(
                f"runtime health abort ({abort.get('kind')}) before window release on device "
                f"{device} (npu {monitor.npu_id}): {abort.get('reason')}"
                + _quarantine_note(cleanup),
                category="occupancy",
                occupancy=abort.get("occupancy"),
            )
    cleanup = _cleanup(
        False, f"host command finished (exit {completed.returncode})"
    )
    cleanup["workerJoined"] = not thread.is_alive()
    if cleanup.get("surviving"):
        monitor.finish("cleanup_incomplete")
        monitor_sink.update(monitor.snapshot())
        raise BridgeError(
            "host command finished but this run's processes could not be "
            f"terminated before releasing the device window: {cleanup.get('surviving')}"
            + _quarantine_note(cleanup),
            category="infra",
            occupancy="quarantined" if (cleanup.get("quarantine") or {}).get("written") else None,
        )
    monitor.finish("completed")
    monitor_sink.update(monitor.snapshot())
    return completed


@contextmanager
def _open_msprof_lock(wroot: Path, *, timeout_seconds: float = 3600.0):
    """Serialize host-wide CANN msprof profiling across concurrent NPU device runs."""
    shared = locking.is_shared_device_lock_root(wroot)
    lock_path = wroot / ".msprof.lock"
    fd = locking.open_lock_file(lock_path, shared=shared)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BridgeError(
                        f"host msprof lock {lock_path} did not open within {timeout_seconds}s"
                    ) from exc
                time.sleep(min(0.25, remaining))
        yield lock_path
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _extract_last_json(stdout: str, marker_key: str) -> dict:
    for line in reversed((stdout or "").splitlines()):
        cleaned = line.strip()
        if cleaned.startswith("{"):
            try:
                parsed = json.loads(cleaned)
            except ValueError:
                continue
            if isinstance(parsed, dict) and marker_key in parsed:
                return parsed
    return {}


def _abi_slot(
    values: dict[str, str], source: Path, repo: Path, wroot: Path, info: dict[str, str]
) -> tuple[str, Path, Path]:
    arch = cannbench_arch(values)
    cann_setup = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP")).resolve()
    python = python_path(values).resolve()
    key_material = "\0".join([
        arch,
        str(python),
        str(cann_setup),
        str(info.get("pythonVersion", "")),
        str(info.get("torchVersion", "")),
        str(info.get("torchNpuVersion", "")),
        _utils_source_digest(repo),
    ])
    digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:12]
    slot_dir = wroot / "cannbench-utils" / f"{arch}-{digest}"
    return f"{arch}-{digest}", slot_dir, slot_dir / "cann_bench_utils"


def _host_child_env(values: dict[str, str], *, include_llm_credential: bool = False) -> dict[str, str]:
    """Limit provider API keys when a credential variable is configured."""
    name = _llm_credential_name(values)
    if not name:
        return dict(os.environ)
    env = {
        key: value for key, value in os.environ.items()
        if key != name and key.upper() != "API_KEY" and not key.upper().endswith("_API_KEY")
    }
    if include_llm_credential and name in os.environ:
        env[name] = os.environ[name]
    return env


def _scrub_python_startup(env: dict[str, str]) -> None:
    """Drop parent-shell Python hooks so only adapter-built paths reach a child."""
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "AGENT_GATES_ACTIVATION_PYTHONPATH",
    ):
        env.pop(name, None)


def _inspect_conda_cannbench_env(
    values: dict[str, str], source: Path, *, timeout: float = 60.0
) -> dict:
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)
    run_env = _host_child_env(values)
    _scrub_python_startup(run_env)
    envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
    if envs_path:
        run_env["CONDA_ENVS_PATH"] = envs_path
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    run_env["PYTHONNOUSERSITE"] = "1"
    completed = adapter_runtime.run_host_bash(
        f"{shlex.quote(str(python))} -c {shlex.quote(CANNBENCH_ENV_INSPECT)}",
        cwd=source,
        env=run_env,
        cann_setup=None,
        conda_sh=conda_sh,
        conda_env=conda_env,
        timeout=max(1.0, timeout),
        isolate_temp=True,
    )
    info = _extract_last_json(completed.stdout, "missing")
    if completed.returncode != 0 or not info:
        sys.stderr.write((completed.stderr or completed.stdout)[-4000:])
        raise BridgeError("failed to inspect TileFoundry Conda Python dependencies")
    return info


def _forward_proxy_env(
    run_env: dict[str, str],
    values: dict[str, str],
    *,
    include_agent_llm: bool,
) -> None:
    """Forward explicitly configured proxies in both variable cases."""
    for key in PROXY_KEYS:
        scoped = f"TILEFOUNDRY_NPU_LLM_{key}"
        scoped_value = (values.get(scoped) or "").strip()
        value = scoped_value if include_agent_llm else ""
        value = (
            value
            or (run_env.get(key) or "").strip()
            or (run_env.get(key.lower()) or "").strip()
        )
        if not value:
            continue
        run_env[key] = value
        run_env[key.lower()] = value
        if scoped_value and include_agent_llm:
            run_env[scoped] = scoped_value


def _detect_cann_series_from_setup(cann_setup: str) -> str:
    # cann_9_3 is the sole active mainline; cann_9_2 is an inert schema
    # placeholder selectable ONLY when explicitly set via
    # TILEFOUNDRY_NPU_CANN_SERIES=cann_9_2.
    _ = cann_setup
    return "cann_9_3"


def resolve_cann_series(values: dict[str, str]) -> str:
    explicit = (values.get("TILEFOUNDRY_NPU_CANN_SERIES") or "").strip()
    if explicit:
        if explicit not in VALID_CANN_SERIES:
            raise BridgeError(
                f"TILEFOUNDRY_NPU_CANN_SERIES must be one of {VALID_CANN_SERIES}: {explicit!r}"
            )
        return explicit
    return _detect_cann_series_from_setup(values.get("TILEFOUNDRY_NPU_CANN_SETUP") or "")


def _environment_observations(values: dict[str, str]) -> dict[str, str]:
    return {
        "cannSeries": resolve_cann_series(values),
        "bishengArch": "dav-3510",
    }


def _forward_optional_env(
    run_env: dict[str, str],
    values: dict[str, str],
    *,
    include_agent_llm: bool = True,
) -> None:
    keys = (
        OPTIONAL_FORWARDED_KEYS
        if include_agent_llm
        else OPTIONAL_REPO_AND_PYTHON_KEYS
    )
    for key in keys:
        val = (values.get(key) or "").strip()
        if val:
            run_env[key] = val
    run_env["TILEFOUNDRY_CANN_SERIES"] = resolve_cann_series(values)
    if not include_agent_llm:
        for key in (*OPTIONAL_AGENT_LLM_KEYS, *OPTIONAL_PROXY_KEYS):
            run_env.pop(key, None)
    _forward_proxy_env(run_env, values, include_agent_llm=include_agent_llm)


def _validate_optional_agent_keys(values: dict[str, str]) -> None:
    resolve_cann_series(values)
    conc = (values.get("TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY") or "").strip()
    if conc and (not conc.isdigit() or int(conc) < 1):
        raise BridgeError(
            f"TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY must be a positive integer: {conc!r}"
        )
    raw_fallback = values.get("TILEFOUNDRY_NPU_LLM_FALLBACK_MODEL") or ""
    fallback_model = raw_fallback.strip()
    if raw_fallback and not fallback_model:
        raise BridgeError(
            "TILEFOUNDRY_NPU_LLM_FALLBACK_MODEL must not be whitespace-only"
        )
    if fallback_model and any(ch.isspace() for ch in fallback_model):
        raise BridgeError(
            "TILEFOUNDRY_NPU_LLM_FALLBACK_MODEL must be a single-token model ID: "
            f"{fallback_model!r}"
        )
    raw_503_retries = (values.get("TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES") or "").strip()
    if raw_503_retries and not re.fullmatch(r"[0-9]+", raw_503_retries):
        raise BridgeError(
            "TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES must be a non-negative integer: "
            f"{raw_503_retries!r}"
        )
    raw_503_backoff = (values.get("TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC") or "").strip()
    if raw_503_backoff:
        parts = [part.strip() for part in raw_503_backoff.split(",")]
        if not parts or any(not part for part in parts):
            raise BridgeError(
                "TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC must be a comma-separated list "
                f"of non-negative finite floats: {raw_503_backoff!r}"
            )
        for part in parts:
            try:
                val = float(part)
            except ValueError as exc:
                raise BridgeError(
                    "TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC must contain only "
                    f"non-negative finite floats: {raw_503_backoff!r}"
                ) from exc
            if not math.isfinite(val) or val < 0.0:
                raise BridgeError(
                    "TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC entries must be "
                    f"non-negative finite floats: {raw_503_backoff!r}"
                )
    for key in OPTIONAL_ABS_PATH_KEYS:
        val = (values.get(key) or "").strip()
        if val and not Path(val).is_absolute():
            raise BridgeError(f"{key} must be an absolute path when set: {val!r}")


def l3_timeout(values: dict[str, str]) -> float:
    raw = (values.get("TILEFOUNDRY_NPU_L3_TIMEOUT_SEC") or "").strip()
    if not raw:
        return DEFAULT_L3_TIMEOUT
    try:
        value = float(raw)
    except ValueError as exc:
        raise BridgeError(
            f"TILEFOUNDRY_NPU_L3_TIMEOUT_SEC must be a number: {raw!r}"
        ) from exc
    if not 0 < value <= MAX_L3_TIMEOUT:
        raise BridgeError(
            f"TILEFOUNDRY_NPU_L3_TIMEOUT_SEC must be in (0, {MAX_L3_TIMEOUT:g}]: {raw!r}"
        )
    return value


def l4_external_probe_limit(values: dict[str, str]) -> int:
    """Resolve how many consecutive external-occupancy probes block L4 timing."""
    raw = (values.get("TILEFOUNDRY_NPU_L4_EXTERNAL_PROBE_LIMIT") or "").strip()
    if not raw:
        return L4_EXTERNAL_PROBE_LIMIT_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise BridgeError(
            "TILEFOUNDRY_NPU_L4_EXTERNAL_PROBE_LIMIT must be an integer: "
            f"{raw!r}"
        ) from exc
    if value < 1:
        raise BridgeError(
            "TILEFOUNDRY_NPU_L4_EXTERNAL_PROBE_LIMIT must be at least 1: "
            f"{raw!r}"
        )
    return value


def occupancy_policy(values: dict[str, str]) -> str:
    """Resolve the device-window occupancy admission policy."""
    raw = (values.get("TILEFOUNDRY_NPU_OCCUPANCY_POLICY") or "").strip().lower()
    if not raw:
        return "exclusive"
    if raw not in OCCUPANCY_POLICIES:
        raise BridgeError(
            "TILEFOUNDRY_NPU_OCCUPANCY_POLICY must be one of "
            f"{sorted(OCCUPANCY_POLICIES)}: {raw!r}"
        )
    return raw


def shared_max_ext_procs(values: dict[str, str]) -> int:
    """Resolve max external device process count admitted under shared_quiet."""
    raw = (values.get("TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS") or "").strip()
    if not raw:
        return SHARED_MAX_EXT_PROCS_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise BridgeError(
            f"TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS must be an integer: {raw!r}"
        ) from exc
    if value < 1:
        raise BridgeError(
            f"TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS must be at least 1: {raw!r}"
        )
    return value


def shared_max_hbm_mb(values: dict[str, str]) -> int:
    """Resolve max total external device HBM (MB) admitted under shared_quiet."""
    raw = (values.get("TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB") or "").strip()
    if not raw:
        return SHARED_MAX_HBM_MB_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise BridgeError(
            f"TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB must be an integer: {raw!r}"
        ) from exc
    if value < 256:
        raise BridgeError(
            f"TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB must be at least 256: {raw!r}"
        )
    return value


def check_config(
    *,
    check_device: bool = False,
    check_cannbench: bool = False,
    check_cannbench_remote: bool = False,
    check_l3: bool = False,
) -> int:
    try:
        values = machine_values()
        for name in REQUIRED:
            _required(values, name)
        build_subdir(values)
        _validate_optional_agent_keys(values)
        _binary_flag(values, "TILEFOUNDRY_NPU_MSPROF_LOCK")
        _binary_flag(values, "TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE")
        l3_timeout(values)
        l4_external_probe_limit(values)
        occupancy_policy(values)
        shared_max_ext_procs(values)
        shared_max_hbm_mb(values)
        cann = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
        if not cann.is_file():
            raise BridgeError(f"TILEFOUNDRY_NPU_CANN_SETUP does not exist: {cann}")
        conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
        if not conda_sh.is_file():
            raise BridgeError(f"TILEFOUNDRY_NPU_CONDA_SH does not exist: {conda_sh}")
        python = python_path(values)
        if not python.is_file():
            raise BridgeError(f"TileFoundry Python interpreter does not exist: {python}")
        source = _source_root()
        cpu_workers(values)
        for rel in cpu_tests(values, source):
            if not (source / rel).exists():
                raise BridgeError(f"CPU test path missing in {source}: {rel}")
        # Only the device route consumes the typed fast descriptor: the canonical
        # device route shares this entry point without a pair, while the L3 and
        # CANNBench routes validate their own canonical target/candidate below.
        # Routing those canonical pairs through fast parsing would reject every
        # managed L3/L4 diagnosis before its own validator ever sees the pair.
        fast_request: FastRequest | None = None
        if check_device and not (check_l3 or check_cannbench):
            _reject_parent_fast_request()
            fast_request = _load_fast_request(source, values, _fast_pair())
        if fast_request is None:
            for rel in device_tests(values):
                if not (source / rel).exists():
                    raise BridgeError(f"device test path missing in {source}: {rel}")
        tilelang_reuse_identity(values)
        if check_cannbench or check_cannbench_remote:
            repo = cannbench_repo(values)
            work_root(values, source, repo)
            if check_cannbench:
                cannbench_arch(values)
            resolve_cannbench_targets(
                values, source, repo, allow_request=check_cannbench
            )
        if check_cannbench_remote:
            _resolve_mcp_config(values)
        if check_l3:
            request = _read_l3_request()
            repo = cannbench_repo(values)
            wroot = work_root(values, source, repo)
            task, _candidate = _validate_target_pair(
                request["task"], request["candidate"], source, repo
            )
            operator = _l3_operator(task)
            _l3_cache_layout(wroot, operator)
        if check_device:
            if fast_request is not None:
                repo = cannbench_repo(values)
                _fast_surrogate_caches(work_root(values, source, repo))
                fast_device_tests(source)
            dev = device_id()
            if dev is not None:
                try:
                    adapter_runtime.check_device_ready(int(dev), exclusive=exclusive_requested())
                except adapter_runtime.BlockedEnv as exc:
                    raise BridgeError(str(exc)) from exc
    except BridgeError as exc:
        _die(75, str(exc))
    print(f"{PROG}: configuration ok", file=sys.stderr)
    return 0


def run_cpu(values: dict[str, str], timeout: float) -> tuple[int, dict]:
    source = _source_root()
    tl_identity = tilelang_reuse_identity(values)
    tests = cpu_tests(values, source)
    workers = cpu_workers(values)
    shards = _expand_cpu_test_shards(source, tests, workers)
    all_deselects = cpu_deselects(source)
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)
    deadline = _Deadline(timeout)

    with tempfile.TemporaryDirectory(prefix="tilefoundry-cpu-") as tmp:
        base_env = _host_child_env(values)
        envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
        if envs_path:
            base_env["CONDA_ENVS_PATH"] = envs_path
        _forward_optional_env(base_env, values, include_agent_llm=False)
        base_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        base_env["PYTHONDONTWRITEBYTECODE"] = "1"
        base_env["PYTHONNOUSERSITE"] = "1"
        base_env["TILELANG_DISABLE_CACHE"] = "1"
        base_env["PYTHONPATH"] = _effective_pythonpath(tl_identity)
        if tl_identity is not None:
            base_env.setdefault("USE_CUDA", "0")
        if len(shards) > 1:
            base_env.update(CPU_WORKER_THREAD_ENV)

        deselect_files = {node.split("::", 1)[0] for node in all_deselects}

        def _run_shard(
            shard_idx: int, shard_targets: tuple[str, ...]
        ) -> tuple[tuple[str, ...], Path, subprocess.CompletedProcess[str]]:
            shard_env = dict(base_env)
            if len(shards) > 1:
                for tmp_key in ("TMPDIR", "TEMP", "TMP", "ASCEND_WORK_PATH"):
                    shard_env.pop(tmp_key, None)
                report = Path(tmp) / f"junit_shard_{shard_idx}.xml"
                target_set = set(shard_targets)
                shard_deselects = [
                    node for node in all_deselects if node.split("::", 1)[0] in target_set
                ]
            else:
                report = Path(tmp) / "junit.xml"
                shard_deselects = list(all_deselects)
            cmd = " ".join(
                [shlex.quote(str(python)), "-m", "pytest", "-p", "no:cacheprovider",
                 f"--junit-xml={shlex.quote(str(report))}"]
                + [f"--deselect={shlex.quote(node)}" for node in shard_deselects]
                + [shlex.quote(t) for t in shard_targets]
            )
            # CPU suite deliberately does NOT source CANN set_env.sh because
            # CANN's native libraries collide with ortools CP-SAT in this env.
            completed = adapter_runtime.run_host_bash(
                cmd,
                cwd=source,
                env=shard_env,
                cann_setup=None,
                conda_sh=conda_sh,
                conda_env=conda_env,
                timeout=max(1.0, deadline.remaining()),
                isolate_temp=True,
            )
            return shard_targets, report, completed

        shard_results: list[tuple[tuple[str, ...], Path, subprocess.CompletedProcess[str]]] = []
        try:
            if len(shards) == 1:
                shard_results.append(_run_shard(0, shards[0]))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(shards)) as pool:
                    futures = [
                        pool.submit(_run_shard, idx, shard)
                        for idx, shard in enumerate(shards)
                    ]
                    for fut in futures:
                        shard_results.append(fut.result())
        except subprocess.TimeoutExpired as exc:
            if exc.stderr:
                sys.stderr.write(str(exc.stderr)[-4000:])
            print(f"{PROG}: CPU pytest timed out after {timeout}s", file=sys.stderr)
            timeout_obs = _cpu_observations(False, 0, 0, 0, 0)
            timeout_obs["cpuWorkers"] = len(shards)
            timeout_obs["environment"] = _environment_observations(values)
            if tl_identity is not None:
                timeout_obs["tilelang"] = tl_identity
            return 1, timeout_obs

        total_collected = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        all_rc_ok = True
        combined_tails: list[str] = []
        for shard_targets, report, completed in shard_results:
            counts = _junit_counts(report)
            if counts is None:
                sys.stderr.write((completed.stderr or completed.stdout)[-4000:])
                print(f"{PROG}: missing JUnit report at {report}", file=sys.stderr)
                missing_obs = _cpu_observations(False, 0, 0, 0, 0)
                missing_obs["cpuWorkers"] = len(shards)
                missing_obs["environment"] = _environment_observations(values)
                if tl_identity is not None:
                    missing_obs["tilelang"] = tl_identity
                return 1, missing_obs
            empty_deselected_shard = (
                completed.returncode == 5
                and counts["collected"] == 0
                and all(target in deselect_files for target in shard_targets)
            )
            if completed.returncode != 0 and not empty_deselected_shard:
                all_rc_ok = False
                combined_tails.append((completed.stdout + "\n" + completed.stderr)[-2000:])
            total_collected += counts["collected"]
            total_passed += counts["passed"]
            total_failed += counts["failed"]
            total_skipped += counts["skipped"]

        succeeded = (
            all_rc_ok
            and total_collected > 0
            and total_failed == 0
            and total_skipped < total_collected
        )
        obs = _cpu_observations(
            succeeded,
            total_collected,
            total_passed,
            total_failed,
            total_skipped,
        )
        obs["cpuWorkers"] = len(shards)
        obs["environment"] = _environment_observations(values)
        if tl_identity is not None:
            obs["tilelang"] = tl_identity
        if not succeeded:
            if combined_tails:
                sys.stderr.write("\n".join(combined_tails)[-4000:])
            else:
                tail = "\n".join(
                    (res.stdout + "\n" + res.stderr)[-1500:]
                    for _, _, res in shard_results
                )
                sys.stderr.write(tail[-4000:])
        return (0 if succeeded else 1), obs


def _redact_llm_credential(text: str | bytes, credential: str) -> str:
    """Redact the credential from device subprocess output."""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    if credential:
        text = text.replace(credential, "[redacted]")
    return text


def _window_evidence(
    occupancy: str, window: dict, *, npu_id: int | None = None
) -> dict:
    evidence: dict[str, object] = {
        "windowOpened": True,
        "windowReleased": True,
        "occupancyRecorded": occupancy in {"free", "unknown"} or occupancy.startswith("shared("),
        "deviceWindow": window,
    }
    resolved_npu_id = npu_id
    if resolved_npu_id is None and isinstance(window, dict):
        resolved_npu_id = _int_or_none(
            window.get("npuId") if "npuId" in window else window.get("npu_id")
        )
    if resolved_npu_id is not None:
        usages = _probe_npu_usages(resolved_npu_id)
        if usages:
            evidence.update(usages)
    return evidence


@contextmanager
def _phase(phases: list[dict], name: str):
    """Record one adapter-controlled phase, including its failure."""
    entry: dict[str, object] = {"name": name, "startedAt": _timestamp(), "status": "running"}
    phases.append(entry)
    started = time.monotonic()
    try:
        yield entry
    except BaseException as exc:
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        raise
    else:
        entry["status"] = "ok"
    finally:
        entry["endedAt"] = _timestamp()
        entry["seconds"] = round(time.monotonic() - started, 3)


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _read_json_file(path: Path, *, limit: int = 1 << 20) -> dict | None:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if len(data) > limit:
        return None
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_l3_request() -> dict:
    """Read the engine-published L3 request: canonical target and candidate."""
    target = (os.environ.get(L3_TARGET_REQUEST_ENV) or "").strip()
    candidate = (os.environ.get(L3_CANDIDATE_REQUEST_ENV) or "").strip()
    if not target or not candidate:
        raise BridgeError(
            f"{L3_TARGET_REQUEST_ENV} and {L3_CANDIDATE_REQUEST_ENV} "
            "must be supplied together by the engine for --l3-promotion"
        )
    return {"task": target, "candidate": candidate}


def _l3_operator(task: str) -> str:
    operator = task.rsplit("/", 1)[-1]
    if operator not in L3_PROFILE_TESTS and not re.fullmatch(r"[a-z][a-z0-9_]*", operator):
        raise BridgeError(f"cannot derive an L3 operator from target {task!r}")
    return operator


def _l3_expected_tests(operator: str) -> tuple[str, ...]:
    return L3_PROFILE_TESTS.get(operator, (L3_GENERIC_TEST,))


def _l3_cache_layout(wroot: Path, operator: str) -> dict[str, Path]:
    base = wroot / "l3" / "cache" / operator
    return {
        "base": base,
        "tilelang": base / "tilelang",
        "reference": base / "reference",
        "warmMarker": base / "reference-warm.json",
    }


def _tree_stats(root: Path) -> dict:
    files = 0
    total = 0
    try:
        for current, _dirs, names in os.walk(root):
            for name in names:
                item = Path(current) / name
                try:
                    if item.is_file():
                        files += 1
                        total += item.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return {"path": str(root), "files": files, "bytes": total}


def _l3_expectation(
    reference: Path, marker: Path, *, task: str, operator: str
) -> tuple[str, str]:
    """Ordinary promotion accepts any legal mix of cache hits and rebuilds.

    A marker is not authoritative here: only the Lead-owned strict cold/warm
    acceptance may read or write it. The observed directory state is still
    reported below so the receipt keeps the pre-run cache identity.
    """
    _ = (marker, task, operator)
    entries = _tree_stats(reference / "operators")["files"]
    if entries == 0:
        return "any", "ordinary-gate-empty-cache-allows-cold-rebuild"
    return "any", "ordinary-gate-observed-entries-allow-legal-mixed-rebuild"


def _l3_junit(path: Path) -> dict | None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    rows: list[dict] = []
    for case in root.iter("testcase"):
        name = str(case.attrib.get("name", ""))
        if case.find("failure") is not None:
            status = "failed"
        elif case.find("error") is not None:
            status = "error"
        elif case.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        try:
            seconds = float(case.attrib.get("time", "0") or 0)
        except ValueError:
            seconds = 0.0
        rows.append({
            "name": name.split("[", 1)[0],
            "fullName": name,
            "className": str(case.attrib.get("classname", "")),
            "status": status,
            "seconds": round(seconds, 3),
        })
    if not suites and not rows:
        return None
    collected = sum(int(s.attrib.get("tests", 0)) for s in suites) if suites else len(rows)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites) if suites else 0
    failures = sum(int(s.attrib.get("failures", 0)) for s in suites) if suites else 0
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites) if suites else 0
    failed = errors + failures
    return {
        "collected": collected,
        "passed": max(0, collected - failed - skipped),
        "failed": failed,
        "errors": errors,
        "failures": failures,
        "skipped": skipped,
        "rows": rows,
    }


def _l3_expected_rows(junit: dict | None, expected: tuple[str, ...]) -> tuple[list[dict], list[str], list[str]]:
    """Return expected rows plus missing/duplicate test names."""
    if junit is None:
        return [], list(expected), []
    rows = [row for row in junit["rows"] if row["name"] in expected]
    names = [row["name"] for row in rows]
    missing = [name for name in expected if name not in names]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    return rows, missing, duplicates


def _l3_evidence_summary(
    evidence: dict | None,
    *,
    expected_module: Path,
    candidate_relative: str,
    candidate_digest: str,
) -> dict | None:
    if evidence is None:
        return None
    target = _as_dict(evidence.get("target"))
    candidate = dict(_as_dict(evidence.get("candidate")))
    # relativePath is not a producer field. It is the canonical path this
    # adapter accepted from the engine request and bound to the candidate
    # bytes, so keep it engine-signed. Producer modulePath/digest fields stay
    # exactly as observed; missing identity is handled by _l3_classify.
    candidate["relativePath"] = candidate_relative
    nodes = _as_list(evidence.get("nodes"))
    # Keep the nested target block, including only the module path the
    # producer recorded. A downstream promotion client binds the receipt to
    # that observed path; the request-derived value is reported separately
    # under expectedIdentity and never substituted into this field.
    return {
        "schema": evidence.get("schema"),
        "profile": evidence.get("profile"),
        "task": target.get("task"),
        "operator": target.get("operator"),
        "target": {
            "operator": target.get("operator"),
            "task": target.get("task"),
            "taskDir": target.get("taskDir"),
            "modulePath": target.get("modulePath"),
            "cases": list(target.get("cases") or []),
        },
        "targetCaseCount": len(_as_list(target.get("cases"))),
        "candidate": candidate,
        "expectedIdentity": {
            "target": {"modulePath": str(expected_module)},
            "candidate": {
                "modulePath": str(expected_module),
                "relativePath": candidate_relative,
                "sha256": candidate_digest,
            },
        },
        "interpreter": _as_dict(evidence.get("interpreter")),
        "coverage": _as_dict(evidence.get("coverage")),
        "referenceCache": _as_dict(evidence.get("referenceCache")),
        "writeErrors": list(_as_list(evidence.get("writeErrors"))),
        "interruption": _as_dict(evidence.get("interruption")),
        "timing": _as_dict(evidence.get("timing")),
        "phase": evidence.get("phase"),
        # Producer-structured failure classification. Only a candidate/numeric
        # comparison failure may become a candidate verdict; everything else
        # stays an infrastructure block.
        "failure": _as_dict(evidence.get("failure")) or None,
        "failures": [row for row in _as_list(evidence.get("failures")) if isinstance(row, dict)],
        "nodes": [
            {
                "name": node.get("name"),
                "status": node.get("status"),
                "nodeId": node.get("nodeId"),
                "requiredCases": list(_as_list(node.get("requiredCases"))),
                "cases": _as_dict(node.get("cases")),
                "failure": _as_dict(node.get("failure")) or None,
                "caseCount": len(_as_dict(node.get("cases"))),
                "error": node.get("error"),
            }
            for node in nodes
            if isinstance(node, dict)
        ],
    }


def run_device(values: dict[str, str], timeout: float, *, fast: bool = False) -> tuple[int, dict]:
    source = _source_root()
    tl_identity = tilelang_reuse_identity(values)
    device = require_device()
    cann_setup = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)
    _reject_parent_fast_request()
    fast_request = _load_fast_request(source, values, _fast_pair())
    tests = fast_device_tests(source) if fast_request is not None else device_tests(values)
    if fast and not tests:
        raise BridgeError(
            "tilefoundry-npu-device-fast requires TILEFOUNDRY_NPU_DEVICE_TESTS "
            "for the legacy fixed mode"
        )
    fast_repo: Path | None = None
    fast_caches: dict[str, Path] | None = None
    if fast_request is not None:
        if not fast:
            raise BridgeError(
                "a dynamic fast request target is only supported by "
                "tilefoundry-npu-device-fast"
            )
        fast_repo = cannbench_repo(values)
        fast_caches = _fast_surrogate_caches(work_root(values, source, fast_repo))
    msprof_mode = fast_request is not None and fast_request.mode in (
        "profile",
        "micro_probe",
    )
    msprof_lock = _binary_flag(values, "TILEFOUNDRY_NPU_MSPROF_LOCK") or msprof_mode
    cache_enabled = _binary_flag(values, "TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE")

    started = time.monotonic()
    lock = (
        _open_msprof_lock(
            locking.device_lock_root(), timeout_seconds=max(1.0, timeout - 30.0)
        )
        if msprof_lock
        else nullcontext()
    )
    policy = occupancy_policy(values)
    max_procs = shared_max_ext_procs(values)
    max_hbm = shared_max_hbm_mb(values)
    require_idle_window = bool(fast_request is not None)
    health_sink: dict = {}
    occ = "not_checked"
    win: dict | None = None
    try:
        with lock:
            with _open_device_window(
                device,
                require_idle=require_idle_window,
                occupancy_policy=policy,
                max_ext_procs=max_procs,
                max_hbm_mb=max_hbm,
                enforce_proc_count=True,
            ) as finalize_window:
                with tempfile.TemporaryDirectory(prefix="tilefoundry-npu-") as tmp:
                    report = Path(tmp) / "junit.xml"
                    run_env = _host_child_env(values, include_llm_credential=bool(tests))
                    credential = run_env.get(_llm_credential_name(values), "").strip()
                    envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
                    if envs_path:
                        run_env["CONDA_ENVS_PATH"] = envs_path
                    _forward_optional_env(run_env, values)
                    run_env["ASCEND_RT_VISIBLE_DEVICES"] = device
                    # A caller that imported tilefoundry_ascend without CANN sourced
                    # exports "0"; torch only autoloads torch_npu when this is "1".
                    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
                    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
                    run_env["PYTHONNOUSERSITE"] = "1"
                    run_env["TILELANG_DISABLE_CACHE"] = "0" if cache_enabled else "1"
                    if cache_enabled:
                        run_env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
                    shim = (run_env.get("TF_SURROGATE_SHIM_DIR") or "").strip() if cache_enabled else ""
                    run_env["PYTHONPATH"] = _effective_pythonpath(tl_identity, shim=shim)
                    if tl_identity is not None:
                        run_env.setdefault("USE_CUDA", "0")
                        run_env["TILEFOUNDRY_EXPECTED_TILELANG_SOURCE"] = tl_identity["source"]

                    if fast_request is not None:
                        # The child reads only the validated snapshot. The original
                        # request is re-hashed after the run, so a caller cannot
                        # rewrite the terms of the run while the device is held.
                        assert fast_repo is not None and fast_caches is not None
                        snapshot = Path(tmp) / "fast_request.json"
                        snapshot.write_bytes(fast_request.raw)
                        snapshot.chmod(0o400)
                        run_env[FAST_REQUEST_ENV] = str(snapshot)
                        run_env[FAST_REQUEST_SOURCE_ENV] = str(source)
                        run_env["TILEFOUNDRY_NPU_CANNBENCH_REPO"] = str(fast_repo)
                        run_env["TILELANG_CACHE_DIR"] = str(fast_caches["tilelang"])
                        run_env["CANNBENCH_CACHE_DIR"] = str(fast_caches["reference"])
                        run_env.pop(FAST_REQUEST_LEGACY_ENV, None)
                        run_env.pop(FAST_MICRO_PROBE_LEGACY_ENV, None)

                    if fast:
                        cmd = " ".join(
                            [shlex.quote(str(python)), "-m", "pytest", "-p", "no:cacheprovider",
                             f"--junit-xml={shlex.quote(str(report))}"]
                            + [shlex.quote(t) for t in tests]
                        )
                    else:
                        probe_path = Path(tmp) / "probe.py"
                        probe_path.write_text(DEVICE_PROBE, encoding="utf-8")
                        cmd = f"{shlex.quote(str(python))} {shlex.quote(str(probe_path))}"
                        if tests:
                            cmd += " && " + " ".join(
                                [shlex.quote(str(python)), "-m", "pytest", "-p", "no:cacheprovider"]
                                + [shlex.quote(t) for t in tests]
                            )
                    remaining = max(1.0, timeout - (time.monotonic() - started))
                    try:
                        try:
                            completed = _run_host_bash_monitored(
                                cmd,
                                mode="fast" if fast_request is not None else "device",
                                device=device,
                                monitor_sink=health_sink,
                                external_probe_limit=1 if fast_request is not None else None,
                                occupancy_policy=policy,
                                shared_max_ext_procs=max_procs,
                                shared_max_hbm_mb=max_hbm,
                                allow_external_in_shared=False,
                                cwd=source,
                                env=run_env,
                                cann_setup=cann_setup,
                                conda_sh=conda_sh,
                                conda_env=conda_env,
                                timeout=remaining,
                                isolate_temp=True,
                            )
                        except subprocess.TimeoutExpired as exc:
                            # run_host_bash SIGTERM/SIGKILLs the whole process group and
                            # reaps it before raising, so no profiling child survives
                            # this return and the msprof lock is safe to release.
                            occ, win = finalize_window()
                            if exc.stderr:
                                sys.stderr.write(_redact_llm_credential(exc.stderr, credential)[-4000:])
                            print(f"{PROG}: device run timed out after {timeout}s", file=sys.stderr)
                            binding_obs: dict[str, object] = {}
                            binding_artifacts: list[dict[str, str]] = []
                            binding_failures: list[str] = []
                            if fast_request is not None:
                                binding_obs, binding_artifacts, binding_failures = _fast_verify_binding(
                                    fast_request, source
                                )
                            if fast:
                                obs = {
                                    "exitCodeZero": False,
                                    "errorCategory": "deadline",
                                    "testsCollected": 0,
                                    "testsPassed": 0,
                                    "testsFailed": 0,
                                    "testsSkipped": 0,
                                    "healthMonitoring": health_sink,
                                    **_window_evidence(occ, win),
                                    **binding_obs,
                                }
                            else:
                                obs = {
                                    "exitCodeZero": False,
                                    "errorCategory": "deadline",
                                    "healthMonitoring": health_sink,
                                    **dict.fromkeys(DEVICE_PROBE_FLAGS, False),
                                    **_window_evidence(occ, win),
                                }
                                if tl_identity is not None:
                                    obs["tilelangReused"] = False
                            if fast_request is not None:
                                obs["fastBindingFailures"] = binding_failures
                                obs["fastRunTimedOut"] = True
                                obs["artifacts"] = binding_artifacts
                                obs.update(
                                    _fast_failure_fields(
                                        binding_obs, binding_failures, ok=False, timed_out=True
                                    )
                                )
                            if tl_identity is not None:
                                obs["tilelang"] = tl_identity
                            obs["environment"] = _environment_observations(values)
                            return 1, obs
                    finally:
                        if win is None:
                            occ, win = finalize_window()

                    binding_obs = {}
                    binding_artifacts = []
                    binding_failures = []
                    if fast_request is not None:
                        binding_obs, binding_artifacts, binding_failures = _fast_verify_binding(
                            fast_request, source
                        )
                    if fast:
                        counts = _junit_counts(report)
                        if counts is None:
                            sys.stderr.write(
                                _redact_llm_credential(
                                    completed.stderr or completed.stdout, credential
                                )[-4000:]
                            )
                            print(f"{PROG}: missing JUnit report at {report}", file=sys.stderr)
                            counts = {"collected": 0, "passed": 0, "failed": 0, "skipped": 0}
                        ok = (
                            completed.returncode == 0
                            and counts["collected"] > 0
                            and counts["failed"] == 0
                            and counts["skipped"] < counts["collected"]
                            and not binding_failures
                        )
                        if not ok:
                            if binding_failures:
                                print(
                                    f"{PROG}: fast request binding failed: {', '.join(binding_failures)}",
                                    file=sys.stderr,
                                )
                            sys.stderr.write(
                                _redact_llm_credential(
                                    completed.stdout + "\n" + completed.stderr, credential
                                )[-4000:]
                            )
                        obs = {
                            "exitCodeZero": ok,
                            "errorCategory": "none" if ok else "candidate",
                            "testsCollected": counts["collected"],
                            "testsPassed": counts["passed"],
                            "testsFailed": counts["failed"],
                            "testsSkipped": counts["skipped"],
                            "environment": _environment_observations(values),
                            "healthMonitoring": health_sink,
                            **_window_evidence(occ, win),
                            **binding_obs,
                        }
                        if fast_request is not None:
                            obs["fastBindingFailures"] = binding_failures
                            obs["artifacts"] = binding_artifacts
                            obs.update(
                                _fast_failure_fields(
                                    binding_obs, binding_failures, ok=ok, timed_out=False
                                )
                            )
                        if tl_identity is not None:
                            obs["tilelang"] = tl_identity
                        return (0 if ok else (completed.returncode or 1)), obs

                    if completed.stderr.strip():
                        sys.stderr.write(_redact_llm_credential(completed.stderr, credential)[-4000:])

                    parsed = _extract_last_json(completed.stdout, "npuAvailable")
                    tilelang_reused_ok = True if tl_identity is None else (parsed.get("tilelangReused") is True)
                    ok = completed.returncode == 0 and bool(parsed) and tilelang_reused_ok and all(
                        parsed.get(k) is True for k in DEVICE_PROBE_FLAGS
                    )
                    if not ok:
                        sys.stderr.write(_redact_llm_credential(completed.stdout or "", credential)[-4000:])

                    obs = {
                        "exitCodeZero": ok,
                        "errorCategory": "none" if ok else "candidate",
                        "environment": _environment_observations(values),
                        "healthMonitoring": health_sink,
                        **{key: bool(parsed.get(key, False)) for key in DEVICE_PROBE_FLAGS},
                        **_window_evidence(occ, win),
                    }
                    if tl_identity is not None:
                        obs["tilelangReused"] = bool(parsed.get("tilelangReused", False))
                        obs["tilelangVersion"] = _redact_llm_credential(
                            str(parsed.get("tilelangVersion", "")), credential
                        )
                        obs["tilelang"] = tl_identity
                    return (0 if ok else (completed.returncode or 1)), obs
    except BridgeError as exc:
        obs = {
            "exitCodeZero": False,
            "errorCategory": exc.category,
            "failureCategory": exc.category,
            "error": str(exc),
            "windowOpened": win is not None,
            "windowReleased": win is not None,
            "occupancyRecorded": False,
            "healthMonitoring": health_sink,
            "environment": _environment_observations(values),
        }
        if win is not None:
            obs.update(_window_evidence(occ, win))
        if exc.occupancy:
            obs["deviceOccupancy"] = exc.occupancy
            obs["occupancyRecorded"] = (
                obs["occupancyRecorded"]
                or exc.occupancy in {"free", "unknown"}
                or exc.occupancy.startswith("shared(")
            )
        rc_out = (
            1
            if (exc.category == "deadline" and obs.get("windowOpened"))
            else adapter_runtime.BLOCKED_ENV
        )
        return rc_out, obs


def _lease_wait_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS", "60")))
    except ValueError as exc:
        raise BridgeError("invalid AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS") from exc


def _l3_child_env(
    values: dict[str, str],
    *,
    source: Path,
    device: str,
    task: str,
    candidate: Path,
    python: Path,
    repo: Path,
    run_dir: Path,
    caches: dict[str, Path],
    tl_identity: dict[str, str] | None,
    expectation: str,
    workload: bool,
) -> dict[str, str]:
    """Build the one activation environment shared by preflight and pytest."""
    env = _host_child_env(values)
    envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
    if envs_path:
        env["CONDA_ENVS_PATH"] = envs_path
    for name in list(env):
        if (
            name.startswith(("TF_L3_", "TILELANG_", "CANNBENCH_CACHE_"))
            or name in {
                "PYTHONPATH",
                "PYTHONHOME",
                "PYTHONSTARTUP",
                "ASCEND_RT_VISIBLE_DEVICES",
                "TORCH_DEVICE_BACKEND_AUTOLOAD",
                "AGENT_GATES_TARGET",
                "AGENT_GATES_CANDIDATE",
            }
        ):
            env.pop(name, None)
    # The target interpreter gets the adapter-resolved source root and the
    # CANNBench checkout's official comparator path. No shim module is
    # generated: the acceptance suite imports npu_compare when installed and
    # otherwise falls back to kernel_eval.utils.compare from this checkout.
    parts = [str(source.resolve()), str(repo.resolve() / "src")]
    if tl_identity is not None:
        parts.append(tl_identity["source"])
    parts.append(DEFAULT_PYTHONPATH)
    env.update({
        "PYTHONPATH": ":".join(parts),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TILELANG_CACHE_DIR": str(caches["tilelang"]),
        "TILELANG_KERNEL_CACHE_USE_LIB_STAMP": "1",
        "TILELANG_DISABLE_CACHE": "0",
        "CANNBENCH_CACHE_DIR": str(caches["reference"]),
        "TILEFOUNDRY_CANN_SERIES": resolve_cann_series(values),
        L3_CACHE_MODE_ENV: "required",
        L3_CACHE_EXPECTATION_ENV: expectation,
    })
    if (values.get("TILEFOUNDRY_NPU_CANN_SERIES") or "").strip():
        env["TILEFOUNDRY_NPU_CANN_SERIES"] = values["TILEFOUNDRY_NPU_CANN_SERIES"].strip()
    if workload:
        env.update({
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
            "ASCEND_RT_VISIBLE_DEVICES": device,
            L3_EVIDENCE_ENV: str(run_dir / "evidence.json"),
            L3_PROFILE_ENV: L3_PROFILE,
            L3_TARGET_OPERATOR_ENV: task,
            L3_TARGET_MODULE_ENV: str(candidate),
            L3_EXPECTED_PYTHON_ENV: str(python.resolve()),
            L3_CANNBENCH_REPO_ENV: str(repo),
        })
        if tl_identity is not None:
            env["TILEFOUNDRY_EXPECTED_TILELANG_SOURCE"] = tl_identity["source"]
    else:
        # Preflight imports and probes only; it must not touch the granted card.
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    return env


def _l3_preflight_errors(
    facts: dict | None, python: Path, caches: dict[str, Path]
) -> list[str]:
    if not isinstance(facts, dict):
        return ["preflight produced no facts"]
    errors: list[str] = []
    if facts.get("ok") is not True:
        errors.append(str(facts.get("error") or "preflight reported failure"))
    try:
        if Path(str(facts.get("python", ""))).resolve() != python.resolve():
            errors.append("preflight interpreter differs from the adapter-resolved Python")
    except OSError:
        errors.append("preflight interpreter path is unusable")
    cache = facts.get("cache") if isinstance(facts.get("cache"), dict) else {}
    for name in L3_CACHE_COMPONENTS:
        row = cache.get(name) if isinstance(cache, dict) else None
        if not isinstance(row, dict) or row.get("writable") is not True:
            errors.append(f"{name} cache probe failed")
            continue
        try:
            if Path(str(row.get("path", ""))).resolve() != caches[name].resolve():
                errors.append(f"{name} cache path differs from the dedicated directory")
        except OSError:
            errors.append(f"{name} cache path is unusable")
    tilelang_cache = str(facts.get("tilelangCacheDir") or "").strip()
    try:
        if not tilelang_cache or Path(tilelang_cache).resolve() != caches["tilelang"].resolve():
            errors.append("TileLang effective cache directory differs from the dedicated directory")
    except OSError:
        errors.append("TileLang cache directory is unusable")
    case_ids = facts.get("caseIds")
    if not isinstance(case_ids, list) or not case_ids:
        errors.append("CANNBench task declares no case IDs")
    elif len(set(case_ids)) != len(case_ids):
        errors.append("CANNBench task declares duplicate case IDs")
    return errors


def _l3_failure_rows(evidence: dict | None) -> list[dict]:
    """Collect the producer's structured failure rows without inventing any."""
    if not isinstance(evidence, dict):
        return []
    rows: list[dict] = []
    for key in ("failure",):
        item = evidence.get(key)
        if isinstance(item, dict):
            rows.append(item)
    for item in _as_list(evidence.get("failures")):
        if isinstance(item, dict):
            rows.append(item)
    for node in _as_list(evidence.get("nodes")):
        if isinstance(node, dict) and isinstance(node.get("failure"), dict):
            rows.append(node["failure"])
    cases = evidence.get("cases")
    if isinstance(cases, dict):
        for item in cases.values():
            if isinstance(item, dict) and isinstance(item.get("failure"), dict):
                rows.append(item["failure"])
    unique: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        try:
            key = json.dumps(row, sort_keys=True, default=str)
        except (TypeError, ValueError):
            continue
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _candidate_failed_node_names(rows: list[dict]) -> set[str]:
    """Return pytest node names that recorded an explicit candidate failure row."""
    return {
        str(row.get("node")).strip()
        for row in rows
        if isinstance(row, dict)
        and str(row.get("kind") or "").strip().lower() == "candidate"
        and str(row.get("node") or "").strip()
    }


def _is_collateral_missing_case(row: dict, candidate_nodes: set[str]) -> bool:
    """Return True only for downstream missing_case rows on an already failed candidate node."""
    if str(row.get("kind") or "").strip().lower() != "coverage":
        return False
    if str(row.get("category") or "").strip().lower() != "missing_case":
        return False
    node_name = str(row.get("node") or "").strip()
    return bool(node_name and node_name in candidate_nodes)


def _l3_producer_block_kind(rows: list[dict]) -> str | None:
    """Producer's own precedence: infrastructure/non-collateral coverage outrank candidate."""
    cand_nodes = _candidate_failed_node_names(rows)
    has_coverage_block = False
    for row in rows:
        kind = str(row.get("kind") or "").strip().lower()
        if kind == "infrastructure":
            return "infrastructure"
        if kind == "coverage" and not _is_collateral_missing_case(row, cand_nodes):
            has_coverage_block = True
    return "coverage" if has_coverage_block else None


_CANDIDATE_KERNEL_STAGES = frozenset({"kernel", "compile", "cache_idempotence"})
_CANDIDATE_KERNEL_DETAIL_SIGNALS = (
    "exceeded vf stack size",
    "ascend compilation failed",
    "bisheng",
    "ld.lld",
    "undefined symbol",
    "tvmerror",
    "l3 cache idempotence violation",
    "syntaxerror",
    "nameerror",
    "typeerror",
    "attributeerror",
    "keyerror",
    "valueerror",
    "importerror",
    "ub overflow",
    "out of ubuf",
    "kernel launch",
)
_INFRA_KERNEL_DETAIL_EXCLUDES = (
    "device offline",
    "device unavailable",
    "npu-smi",
    "drverror",
    "e39999",
    "ez9999",
    "aclrtgetdevice",
    "timeout",
    "reference cache",
    "no space left",
    "permission denied",
    "segmentation fault",
    "bus error",
)


def _is_candidate_kernel_failure_row(
    row: dict,
    *,
    node_records: dict[str, dict],
    junit_rows: list[dict],
) -> bool:
    """Return True iff *row* binds a deterministic candidate compile/kernel/cache defect."""
    role = str(row.get("role") or "").strip().lower()
    if role == "baseline":
        return False
    stage = str(row.get("stage") or "").strip().lower()
    if stage not in _CANDIDATE_KERNEL_STAGES:
        return False
    node_name = row.get("node")
    if not isinstance(node_name, str) or not node_name:
        return False
    if not any(
        j.get("status") == "failed" and j.get("name") == node_name
        for j in junit_rows
    ):
        return False
    case_id = row.get("case")
    if case_id is not None:
        node = node_records.get(node_name)
        cases = node.get("cases") if isinstance(node, dict) else None
        if (
            not isinstance(cases, dict)
            or str(cases.get(str(case_id)) or "").strip().lower() not in {"failed", "error"}
        ):
            return False
    detail_text = f"{row.get('exceptionType') or ''} {row.get('detail') or ''}".lower()
    if any(excl in detail_text for excl in _INFRA_KERNEL_DETAIL_EXCLUDES):
        return False
    kind = str(row.get("kind") or "").strip().lower()
    if kind == "candidate" and (role == "candidate" or stage in {"kernel", "compile"}):
        return True
    return any(sig in detail_text for sig in _CANDIDATE_KERNEL_DETAIL_SIGNALS)


def _l3_candidate_kernel_failure(
    evidence: dict | None, junit_rows: list[dict]
) -> dict | None:
    """Return a deterministic candidate compile/kernel/idempotence failure if unambiguous.

    Any genuine non-candidate infrastructure or coverage failure row outranks
    candidate kernel failures so hardware/environment/coverage blocks stay intact.
    """
    if not isinstance(evidence, dict):
        return None
    failures = _l3_failure_rows(evidence)
    if not failures:
        return None
    node_records = {
        str(node.get("name")): node
        for node in _as_list(evidence.get("nodes"))
        if isinstance(node, dict) and node.get("name")
    }
    cand_nodes = _candidate_failed_node_names(failures)
    matched: list[dict] = []
    for row in failures:
        if _is_candidate_kernel_failure_row(
            row, node_records=node_records, junit_rows=junit_rows
        ):
            matched.append(row)
            continue
        if _is_collateral_missing_case(row, cand_nodes):
            continue
        kind = str(row.get("kind") or "").strip().lower()
        if kind in {"infrastructure", "coverage"}:
            return None
    if not matched:
        return None
    first = matched[0]
    return {
        "kind": "candidate",
        "category": "candidate",
        "role": first.get("role") or "candidate",
        "stage": str(first.get("stage") or "").strip().lower(),
        "node": first.get("node"),
        "case": None if first.get("case") is None else str(first.get("case")),
        "detail": first.get("detail"),
        "exceptionType": first.get("exceptionType"),
    }


def _l3_candidate_numeric_failure(
    evidence: dict | None, junit_rows: list[dict]
) -> dict | None:
    """Return the only producer-explicit candidate numeric comparison failure.

    A bare JUnit ``failed`` row, a missing field, or a marker whose role,
    applicable node, case or comparison does not line up is never numeric.
    """
    if not isinstance(evidence, dict):
        return None
    failures = _l3_failure_rows(evidence)
    if _l3_producer_block_kind(failures) is not None:
        return None
    node_records = {
        str(node.get("name")): node
        for node in _as_list(evidence.get("nodes"))
        if isinstance(node, dict) and node.get("name")
    }
    for marker in failures:
        if str(marker.get("kind") or "").strip().lower() != "candidate":
            continue
        if str(marker.get("category") or "").strip().lower() != "numeric":
            continue
        if str(marker.get("role") or "").strip().lower() != "candidate":
            continue
        if str(marker.get("stage") or "").strip().lower() != "comparison":
            continue
        if not marker.get("comparator"):
            continue
        comparisons = marker.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            continue
        if not any(
            isinstance(row, dict) and row.get("passed") is False
            for row in comparisons
        ):
            continue
        node_name = marker.get("node")
        case_id = marker.get("case")
        if not isinstance(node_name, str) or not node_name or case_id is None:
            continue
        node = node_records.get(node_name)
        if not isinstance(node, dict):
            continue
        cases = node.get("cases")
        if not isinstance(cases, dict):
            continue
        if str(cases.get(str(case_id)) or "").strip().lower() not in {"failed", "error"}:
            continue
        # The matching JUnit row must be a failure of the same node. A failed
        # row for another node/case does not transfer the candidate verdict.
        if not any(
            row.get("status") == "failed" and row.get("name") == node_name
            for row in junit_rows
        ):
            continue
        return {
            "kind": "candidate",
            "category": "numeric",
            "role": "candidate",
            "stage": "comparison",
            "node": node_name,
            "case": str(case_id),
            "comparator": marker.get("comparator"),
            "detail": marker.get("detail"),
            "exceptionType": marker.get("exceptionType"),
        }
    return None


def _l3_classify(
    *,
    rc: int,
    expected: tuple[str, ...],
    junit: dict | None,
    evidence: dict | None,
    task: str,
    operator: str,
    candidate: Path,
    candidate_digest: str,
    python: Path,
) -> tuple[str, bool, bool, list[str]]:
    """Classify a finished L3 pytest run without turning gaps into zeros."""
    notes: list[str] = []
    rows, missing, duplicates = _l3_expected_rows(junit, expected)
    if junit is None:
        notes.append("junit report missing")
    if missing:
        notes.append(f"expected tests missing: {', '.join(missing)}")
    if duplicates:
        notes.append(f"expected tests duplicated: {', '.join(duplicates)}")
    junit_ok = (
        junit is not None
        and bool(rows)
        and not missing
        and not duplicates
        and all(row["status"] == "passed" for row in rows)
    )

    ev = _as_dict(evidence) if evidence is not None else None
    ev_target = _as_dict(ev.get("target")) if ev else {}
    ev_coverage = _as_dict(ev.get("coverage")) if ev else {}
    ev_candidate = _as_dict(ev.get("candidate")) if ev else {}
    ev_interpreter = _as_dict(ev.get("interpreter")) if ev else {}
    ev_cache = _as_dict(ev.get("referenceCache")) if ev else {}
    ev_interruption = _as_dict(ev.get("interruption")) if ev else {}
    ev_write_errors = _as_list(ev.get("writeErrors")) if ev else []
    ev_timing = _as_dict(ev.get("timing")) if ev else {}
    target_cases = _as_list(ev_target.get("cases"))
    identity_ok = bool(
        ev
        and ev.get("schema") == L3_EVIDENCE_SCHEMA
        and ev.get("profile") == L3_PROFILE
        and ev_target.get("task") == task
        and ev_target.get("operator") == operator
        and target_cases
        and len(set(str(case) for case in target_cases)) == len(target_cases)
    )
    candidate_ok = bool(
        ev_candidate.get("sha256") == candidate_digest
        and ev_candidate.get("modulePath") is not None
        and str(ev_candidate.get("modulePath")) == str(candidate)
    )
    # The producer records the module it actually imported as the target.
    # That observed path is required identity for promotion; the adapter's
    # request-derived expectation is reported separately and never substituted
    # here.
    declared_target_module = ev_target.get("modulePath")
    target_module_ok = (
        declared_target_module is not None
        and str(declared_target_module) == str(candidate)
    )
    try:
        interpreter_ok = bool(
            ev_interpreter.get("match") is True
            and Path(str(ev_interpreter.get("sysExecutable") or "")).resolve() == python.resolve()
        )
    except OSError:
        interpreter_ok = False
    cache_ok = bool(
        ev_cache.get("required") is True
        and ev_cache.get("usable") is True
        and not (ev_cache.get("errors") or [])
    )
    coverage_problem = ev_coverage.get("referenceCacheProblem")
    evidence_qualified = ev_coverage.get("qualified") is True
    # A snapshot is authoritative for this run only when the producer reached
    # its strict final write. A failed final write leaves the previous
    # intermediate payload (possibly without writeErrors), so phase and
    # finishedAt are the trusted terminal state for promotion.
    evidence_terminal = bool(ev and ev.get("phase") == "finished" and ev_timing.get("finishedAt"))
    if ev_write_errors:
        notes.append(f"evidence write errors: {len(ev_write_errors)}")
    if ev is not None and not evidence_terminal:
        notes.append("evidence did not reach a finished terminal state")
    failure_rows = _l3_failure_rows(ev)
    producer_block = _l3_producer_block_kind(failure_rows)
    numeric_failure = _l3_candidate_numeric_failure(ev, rows)
    kernel_failure = _l3_candidate_kernel_failure(ev, rows)
    junit_failed = [row for row in rows if row["status"] == "failed"]
    if junit_failed and numeric_failure is None and kernel_failure is None:
        notes.append(
            "JUnit failure(s) have no matching producer candidate/numeric "
            "comparison failure: "
            + ", ".join(sorted({str(row["name"]) for row in junit_failed}))
        )
    if numeric_failure is not None:
        notes.append(
            "producer candidate numeric failure: "
            f"{numeric_failure.get('node')}:{numeric_failure.get('case')}"
        )
    if kernel_failure is not None:
        notes.append(
            f"producer candidate {kernel_failure.get('stage')} failure: "
            f"{kernel_failure.get('node')}:{kernel_failure.get('case')}"
        )
    interrupted_category = str(ev_interruption.get("errorCategory") or "").strip()
    coverage_complete = bool(
        junit_ok
        and identity_ok
        and candidate_ok
        and target_module_ok
        and interpreter_ok
        and cache_ok
        and evidence_qualified
        and evidence_terminal
        and not ev_write_errors
    )
    promotion_eligible = bool(
        coverage_complete and rc == 0 and not interrupted_category and not coverage_problem
    )
    if promotion_eligible:
        return "none", True, True, notes

    category = "infra"
    if interrupted_category:
        category = "environment"
    elif not interpreter_ok or not candidate_ok or not target_module_ok:
        category = "environment"
    elif not cache_ok or coverage_problem in {"reference_cache_unavailable", "reference_cache_error"}:
        category = "environment"
    elif rc < 0:
        category = "signal"
    elif ev is not None and not evidence_terminal:
        # A stale or unfinalized snapshot is insufficient evidence, not a
        # candidate verdict, even if it happens to contain a numeric marker.
        category = "environment"
    elif numeric_failure is not None and rc != 0:
        # An explicit producer-marked candidate comparison failure is a real
        # candidate verdict; an unrelated evidence write error cannot mask it.
        category = "numeric"
    elif ev_write_errors:
        category = "environment"
    elif missing or duplicates:
        category = "coverage"
    elif coverage_problem:
        category = "coverage"
    elif any(row["status"] in {"skipped"} for row in rows):
        category = "coverage"
    elif any(row["status"] == "error" for row in rows):
        category = "infra"
    elif kernel_failure is not None and identity_ok and rc != 0:
        category = "candidate"
    elif junit_failed:
        # No valid producer marker binds this failure to the candidate. Use
        # the producer's structured kind; an unclassified failure is infra.
        if producer_block == "coverage":
            category = "coverage"
        elif producer_block == "infrastructure":
            category = "environment"
        else:
            category = "infra"
    elif not coverage_complete:
        category = "coverage" if ev is not None else "infra"
    return category, coverage_complete, False, notes


def run_l3_promotion(values: dict[str, str], timeout: float) -> tuple[int, dict]:
    """Run the L3 promotion acceptance profile under adapter-owned activation."""
    timeout = max(1.0, min(float(timeout), MAX_L3_TIMEOUT))
    deadline = _Deadline(timeout)
    phases: list[dict] = []
    obs: dict = {
        "profile": L3_PROFILE,
        "exitCodeZero": False,
        "task": None,
        "operator": None,
        "candidate": None,
        "sourceRoot": None,
        "sourceGitHead": None,
        "tilelang": None,
        "dependencies": None,
        "cachePolicy": None,
        "cachePreflight": {"ok": False},
        "cachePreflightOk": False,
        "cache": {},
        "referenceCache": None,
        "phases": phases,
        "elapsedSeconds": 0.0,
        "testsCollected": None,
        "testsPassed": None,
        "testsFailed": None,
        "testsSkipped": None,
        "testsCollectedPositive": None,
        "notAllSkipped": None,
        "expectedTests": [],
        "testRows": [],
        "caseCount": None,
        "passedCases": None,
        "coverage": None,
        "coverageComplete": False,
        "promotionEligible": False,
        "candidateDigestRecorded": False,
        "errorCategory": "environment",
        "environment": _environment_observations(values),
        "windowOpened": False,
        "windowReleased": False,
        "occupancyRecorded": False,
        "healthMonitoring": {},
    }
    source = None
    repo = None
    caches: dict[str, Path] | None = None
    run_dir: Path | None = None
    junit: dict | None = None
    evidence: dict | None = None
    evidence_path: Path | None = None
    workload_facts_path: Path | None = None
    report: Path | None = None
    candidate_abs: Path | None = None
    candidate_rel = ""
    health_sink: dict = obs["healthMonitoring"]
    rc = 1
    try:
        with _phase(phases, "config"):
            source = _source_root()
            device = require_device()
            repo = cannbench_repo(values)
            wroot = work_root(values, source, repo)
            python = python_path(values)
            if not python.is_file():
                raise BridgeError(f"TileFoundry Python interpreter does not exist: {python}")
            cann_setup = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
            conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
            conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
            tl_identity = tilelang_reuse_identity(values)
            obs["sourceRoot"] = str(source)
            obs["sourceGitHead"] = _git_head_pure(source)
            obs["tilelang"] = tl_identity

        with _phase(phases, "request"):
            request = _read_l3_request()
            task, candidate_rel = _validate_target_pair(
                request["task"], request["candidate"], source, repo
            )
            candidate = (source / candidate_rel).resolve()
            candidate_abs = candidate
            digest = _sha256_file(candidate)
            size = candidate.stat().st_size
            operator = _l3_operator(task)
            expected = _l3_expected_tests(operator)
            obs.update({
                "task": task,
                "operator": operator,
                "candidate": {
                    "path": candidate_rel,
                    "absolutePath": str(candidate),
                    "sha256": digest,
                    "sizeBytes": size,
                },
                "expectedTests": list(expected),
                "candidateDigestRecorded": True,
            })

        with _phase(phases, "activation"):
            assert caches is None
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = wroot / "l3" / "runs" / operator / (
                f"{stamp}_{os.getpid()}_{time.time_ns() % 1_000_000_000:09d}"
            )
            run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            caches = _l3_cache_layout(wroot, operator)
            for name in ("tilelang", "reference"):
                caches[name].mkdir(parents=True, exist_ok=True, mode=0o700)
            expectation, basis = _l3_expectation(
                caches["reference"],
                caches["warmMarker"],
                task=task,
                operator=operator,
            )
            preflight_py = run_dir / "preflight.py"
            preflight_py.write_text(L3_PREFLIGHT, encoding="utf-8")
            wrapper_py = run_dir / "pytest_l3.py"
            wrapper_py.write_text(L3_PYTEST_WRAPPER, encoding="utf-8")
            facts_path = run_dir / "preflight_facts.json"
            workload_facts_path = run_dir / "workload_facts.json"
            report = run_dir / "junit.xml"
            evidence_path = run_dir / "evidence.json"
            obs["cachePolicy"] = {
                "scope": "dedicated",
                "tilelangDir": str(caches["tilelang"]),
                "referenceDir": str(caches["reference"]),
                "referenceMode": "required",
                "expectation": expectation,
                "expectationBasis": basis,
                "strictMode": None,
                "warmMarkerRead": False,
                "warmMarkerWritten": False,
                "sharedCacheTouched": None,
            }
            obs["cache"]["before"] = {
                "tilelang": _tree_stats(caches["tilelang"]),
                "reference": _tree_stats(caches["reference"]),
            }
            obs["paths"] = {
                "runDir": str(run_dir),
                "junit": str(report),
                "evidence": str(evidence_path),
            }
            preflight_env = _l3_child_env(
                values, source=source, device=device, task=task, candidate=candidate,
                python=python, repo=repo, run_dir=run_dir, caches=caches,
                tl_identity=tl_identity, expectation=expectation, workload=False,
            )

        with _phase(phases, "preflight"):
            deadline.require("preflight")
            preflight_cmd = " ".join([
                shlex.quote(str(python)), "-B", shlex.quote(str(preflight_py)),
                shlex.quote(str(facts_path)), shlex.quote(task), shlex.quote(str(repo)),
            ])
            try:
                checked = adapter_runtime.run_host_bash(
                    preflight_cmd,
                    cwd=source,
                    env=preflight_env,
                    cann_setup=cann_setup,
                    conda_sh=conda_sh,
                    conda_env=conda_env,
                    timeout=max(1.0, deadline.cap(300.0)),
                    isolate_temp=True,
                )
            except subprocess.TimeoutExpired as exc:
                if exc.stderr:
                    sys.stderr.write(str(exc.stderr)[-4000:])
                raise BridgeError(
                    "L3 preflight exceeded its time budget",
                    category="deadline" if deadline.remaining() <= 0 else "environment",
                ) from exc
            facts = _read_json_file(facts_path)
            obs["dependencies"] = facts
            errors = _l3_preflight_errors(facts, python, caches)
            obs["cachePreflight"] = {
                "ok": not errors,
                "probeBytes": ((facts or {}).get("cache") or {}).get("tilelang", {}).get("probeBytes"),
                "errors": errors,
            }
            obs["cachePreflightOk"] = not errors
            if checked.returncode != 0 or errors:
                sys.stderr.write((checked.stderr or checked.stdout or "")[-4000:])
                raise BridgeError(
                    "L3 preflight failed: " + "; ".join(errors or [f"exit {checked.returncode}"])
                )

        policy = occupancy_policy(values)
        max_procs = shared_max_ext_procs(values)
        max_hbm = shared_max_hbm_mb(values)
        with _phase(phases, "device-workload"):
            deadline.require("device lease")
            lease_wait = deadline.cap(_lease_wait_seconds())
            with _open_device_window(
                device,
                wait_seconds=lease_wait,
                require_idle=True,
                occupancy_policy=policy,
                max_ext_procs=max_procs,
                max_hbm_mb=max_hbm,
                enforce_proc_count=True,
            ) as finalize_window:
                try:
                    if deadline.remaining() <= 0:
                        raise BridgeError(
                            "L3 workload exceeded the recipe deadline", category="deadline"
                        )
                    workload_env = _l3_child_env(
                        values, source=source, device=device, task=task, candidate=candidate,
                        python=python, repo=repo, run_dir=run_dir, caches=caches,
                        tl_identity=tl_identity, expectation=expectation, workload=True,
                    )
                    workload_cmd = " ".join([
                        shlex.quote(str(python)), "-B", shlex.quote(str(wrapper_py)),
                        shlex.quote(str(workload_facts_path)), shlex.quote(str(report)),
                        shlex.quote(L3_ACCEPTANCE_TEST),
                    ])
                    try:
                        completed = _run_host_bash_monitored(
                            workload_cmd,
                            cwd=source,
                            env=workload_env,
                            cann_setup=cann_setup,
                            conda_sh=conda_sh,
                            conda_env=conda_env,
                            timeout=max(1.0, deadline.remaining()),
                            isolate_temp=True,
                            mode="l3",
                            device=device,
                            monitor_sink=health_sink,
                            external_probe_limit=2,
                            occupancy_policy=policy,
                            shared_max_ext_procs=max_procs,
                            shared_max_hbm_mb=max_hbm,
                            allow_external_in_shared=False,
                            phase_getter=lambda: phases[-1]["name"] if phases else None,
                        )
                    except subprocess.TimeoutExpired as exc:
                        if exc.stderr:
                            sys.stderr.write(str(exc.stderr)[-4000:])
                        raise BridgeError(
                            "L3 promotion workload exceeded the recipe deadline",
                            category="deadline",
                        ) from exc
                finally:
                    occ, win = finalize_window()
                    obs.update(_window_evidence(occ, win))
                if completed.stderr.strip():
                    sys.stderr.write(completed.stderr[-4000:])
                if completed.returncode != 0:
                    sys.stderr.write((completed.stdout or "")[-4000:])
                obs["workloadFacts"] = _read_json_file(workload_facts_path)

        with _phase(phases, "evidence"):
            junit = _l3_junit(report)
            evidence = _read_json_file(evidence_path)
            candidate_abs = Path(str((obs["candidate"] or {}).get("absolutePath", "")))
            category, coverage_complete, eligible, notes = _l3_classify(
                rc=int(completed.returncode),
                expected=expected,
                junit=junit,
                evidence=evidence,
                task=task,
                operator=operator,
                candidate=candidate_abs,
                candidate_digest=str((obs["candidate"] or {}).get("sha256", "")),
                python=python,
            )
            obs.update({
                "errorCategory": category,
                "classificationNotes": notes,
                "coverageComplete": coverage_complete,
                "promotionEligible": eligible,
                "exitCodeZero": eligible,
            })
            candidate_digest = str((obs["candidate"] or {}).get("sha256", ""))
            evidence_summary = _l3_evidence_summary(
                evidence,
                expected_module=candidate_abs,
                candidate_relative=candidate_rel,
                candidate_digest=candidate_digest,
            )
            obs["evidence"] = evidence_summary
            rc = 0 if eligible else 1
    except BridgeError as exc:
        obs["errorCategory"] = exc.category
        obs["error"] = str(exc)
        obs["exitCodeZero"] = False
        obs["promotionEligible"] = False
        obs["coverageComplete"] = False
        if exc.occupancy:
            obs["deviceOccupancy"] = exc.occupancy
            obs["occupancyRecorded"] = (
                obs["occupancyRecorded"]
                or exc.occupancy in {"free", "unknown"}
                or exc.occupancy.startswith("shared(")
            )
    except subprocess.TimeoutExpired as exc:
        obs["errorCategory"] = "deadline" if deadline.remaining() <= 0 else "infra"
        obs["error"] = f"subprocess timeout: {exc}"
    except Exception as exc:  # noqa: BLE001 - the observation is the report
        obs["errorCategory"] = "infra"
        obs["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if evidence is None and evidence_path is not None:
            evidence = _read_json_file(evidence_path)
        if junit is None and report is not None:
            junit = _l3_junit(report)
        if workload_facts_path is not None:
            partial_facts = _read_json_file(workload_facts_path)
            if partial_facts is not None:
                obs.setdefault("workloadFacts", partial_facts)
        if (
            isinstance(evidence, dict)
            and candidate_abs is not None
            and candidate_rel
            and "evidence" not in obs
        ):
            obs["evidence"] = _l3_evidence_summary(
                evidence,
                expected_module=candidate_abs,
                candidate_relative=candidate_rel,
                candidate_digest=str((obs.get("candidate") or {}).get("sha256", "")),
            )
        obs["healthMonitoring"] = health_sink
        if caches is not None:
            obs["cache"]["after"] = {
                "tilelang": _tree_stats(caches["tilelang"]),
                "reference": _tree_stats(caches["reference"]),
            }
        obs["elapsedSeconds"] = deadline.elapsed()
        obs["phases"] = phases

    if junit is not None:
        obs["testsCollected"] = junit["collected"]
        obs["testsPassed"] = junit["passed"]
        obs["testsFailed"] = junit["failed"]
        obs["testsSkipped"] = junit["skipped"]
        obs["testsCollectedPositive"] = junit["collected"] > 0
        obs["notAllSkipped"] = junit["collected"] > 0 and junit["passed"] > 0
        obs["testRows"] = _l3_expected_rows(junit, tuple(obs["expectedTests"]))[0]
    if isinstance(evidence, dict):
        ev_coverage = _as_dict(evidence.get("coverage"))
        ev_cache = _as_dict(evidence.get("referenceCache"))
        ev_cases = _as_dict(ev_coverage.get("cases"))
        obs["coverage"] = ev_coverage or None
        obs["caseCount"] = ev_cases.get("expected")
        obs["passedCases"] = ev_cases.get("completed")
        if ev_cache:
            obs["referenceCache"] = {
                "hits": ev_cache.get("hits"),
                "misses": ev_cache.get("misses"),
                "writes": ev_cache.get("writes"),
                "errors": ev_cache.get("errors"),
                "byKind": ev_cache.get("byKind"),
            }
    if obs["errorCategory"] != "none":
        if obs["errorCategory"] == "occupancy":
            rc = adapter_runtime.BLOCKED_ENV
        else:
            rc = 1 if obs.get("windowOpened") else adapter_runtime.BLOCKED_ENV
    # Promotion clients treat "numeric" and "candidate" as candidate failures.
    # "coverage", "environment", "occupancy", "deadline", "signal", and "infra"
    # are infrastructure blocks that preserve the candidate.
    obs["failureCategory"] = (
        None if obs["errorCategory"] in {None, "none"} else obs["errorCategory"]
    )
    return rc, obs


def _report_evidence(reports_root: Path, since: float) -> dict:
    """Preserve what the official runner already wrote before a failure."""
    rows: list[dict] = []
    try:
        entries = sorted(reports_root.iterdir(), key=lambda p: p.stat().st_mtime)
    except OSError:
        return {"root": str(reports_root), "reports": []}
    for entry in entries:
        try:
            if not entry.is_dir() or entry.stat().st_mtime < since - 2:
                continue
            files: list[str] = []
            for item in sorted(entry.rglob("*")):
                if item.is_file():
                    files.append(item.relative_to(entry).as_posix())
                    if len(files) >= 40:
                        break
            rows.append({
                "name": entry.name,
                "files": files,
                "hasOfficialFinal": (entry / "official_final.json").is_file(),
            })
        except OSError:
            continue
    return {"root": str(reports_root), "reports": rows[-10:]}


def run_cannbench_prep(values: dict[str, str], timeout: float) -> tuple[int, dict]:
    deadline = _Deadline(timeout)
    phases: list[dict] = []
    source = _source_root()
    tl_identity = tilelang_reuse_identity(values)
    device = require_device()
    repo = cannbench_repo(values)
    wroot = work_root(values, source, repo)
    request = _read_cannbench_request()
    request_target = (
        _validate_target_pair(request[0], request[1], source, repo)
        if request is not None
        else None
    )
    arch = cannbench_arch(values)
    cann_setup = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)

    abi_tag, slot_dir, utils_dir = _abi_slot(
        values, source, repo, wroot,
        {"pythonVersion": "", "torchVersion": "", "torchNpuVersion": ""},
    )
    ext_so = utils_dir / "cann_bench_utils" / "_C.so"
    meta_file = slot_dir / "build_meta.json"
    info: dict = {}
    utils_built = False
    utils_reused = False
    obs: dict = {
        "exitCodeZero": False,
        "depsReady": False,
        "cannBenchUtilsPresent": False,
        "cannBenchUtilsCallable": False,
        "utilsBuilt": False,
        "utilsReused": False,
        "utilsRoot": str(utils_dir),
        "arch": arch,
        "targetsSource": "dynamic-request" if request_target else "managed",
        "targets": (
            [{"task": request_target[0], "candidate": request_target[1]}]
            if request_target
            else []
        ),
        "pythonVersion": None,
        "torchVersion": None,
        "torchNpuVersion": None,
        "windowOpened": False,
        "windowReleased": False,
        "occupancyRecorded": False,
        "healthMonitoring": {},
        "phases": phases,
        "elapsedSeconds": 0.0,
        "errorCategory": "environment",
        "environment": _environment_observations(values),
    }
    health_sink: dict = obs["healthMonitoring"]
    if tl_identity is not None:
        obs["tilelang"] = tl_identity
    policy = occupancy_policy(values)
    max_procs = shared_max_ext_procs(values)
    max_hbm = shared_max_hbm_mb(values)
    try:
        deadline.require("device lease")
        with _open_device_window(
            device,
            wait_seconds=deadline.cap(_lease_wait_seconds()),
            require_idle=True,
            occupancy_policy=policy,
            max_ext_procs=max_procs,
            max_hbm_mb=max_hbm,
            enforce_proc_count=False,
        ) as finalize_window:
            try:
                with _phase(phases, "inspect"):
                    if deadline.remaining() <= 0:
                        raise BridgeError(
                            "CANNBench prep exceeded the recipe deadline", category="deadline"
                        )
                    info = _inspect_conda_cannbench_env(
                        values, source, timeout=deadline.cap(60.0)
                    )
                    missing = info.get("missing") or []
                    if missing:
                        raise BridgeError(
                            "missing CANNBench Python packages in TileFoundry env "
                            f"({', '.join(missing)}); run "
                            f"{ADAPTER_DIR / 'scripts' / 'setup_cannbench_deps.sh'}"
                        )
                    obs.update({
                        "depsReady": True,
                        "pythonVersion": info.get("pythonVersion"),
                        "torchVersion": info.get("torchVersion"),
                        "torchNpuVersion": info.get("torchNpuVersion"),
                    })
                    abi_tag, slot_dir, utils_dir = _abi_slot(values, source, repo, wroot, info)
                    ext_so = utils_dir / "cann_bench_utils" / "_C.so"
                    meta_file = slot_dir / "build_meta.json"
                    obs.update({
                        "utilsRoot": str(utils_dir),
                        "cannBenchUtilsPresent": ext_so.is_file(),
                    })
                    if ext_so.is_file() and meta_file.is_file():
                        utils_reused = True

                with _phase(phases, "build"):
                    if not utils_reused:
                        build_parent = wroot / "cannbench-utils"
                        build_parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.TemporaryDirectory(
                            prefix=f".tmp-{abi_tag}-", dir=str(build_parent)
                        ) as stage_tmp:
                            stage_slot = Path(stage_tmp) / "slot"
                            stage_utils = stage_slot / "cann_bench_utils"
                            stage_slot.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(repo / "src" / "cann_bench_utils", stage_utils)
                            build_cmd = " && ".join([
                                " ".join([
                                    "cmake", "-S", shlex.quote(str(stage_utils)),
                                    "-B", shlex.quote(str(stage_utils / "build")),
                                    "-DCMAKE_BUILD_TYPE=Release",
                                    f"-DPython3_EXECUTABLE={shlex.quote(str(python))}",
                                    f"-DTorch_DIR={shlex.quote(info['torchCmakeDir'])}",
                                    f"-DTORCH_NPU_PATH={shlex.quote(info['torchNpuDir'])}",
                                    f"-DNPU_ARCH={shlex.quote(arch)}",
                                ]),
                                " ".join([
                                    "cmake", "--build",
                                    shlex.quote(str(stage_utils / "build")),
                                    "--parallel", "4",
                                ]),
                            ])
                            build_env = _host_child_env(values)
                            _scrub_python_startup(build_env)
                            envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
                            if envs_path:
                                build_env["CONDA_ENVS_PATH"] = envs_path
                            build_env["PYTHONDONTWRITEBYTECODE"] = "1"
                            build_env["PYTHONNOUSERSITE"] = "1"
                            build_env["TILEFOUNDRY_CANN_SERIES"] = resolve_cann_series(values)
                            if deadline.remaining() <= 0:
                                raise BridgeError(
                                    "CANNBench prep build exceeded the recipe deadline",
                                    category="deadline",
                                )
                            try:
                                built = _run_host_bash_monitored(
                                    build_cmd,
                                    cwd=repo,
                                    env=build_env,
                                    cann_setup=cann_setup,
                                    conda_sh=conda_sh,
                                    conda_env=conda_env,
                                    timeout=deadline.cap(600.0),
                                    isolate_temp=True,
                                    mode="l4",
                                    device=device,
                                    monitor_sink=health_sink,
                                    occupancy_policy=policy,
                                    shared_max_ext_procs=max_procs,
                                    shared_max_hbm_mb=max_hbm,
                                    allow_external_in_shared=True,
                                    phase_getter=(
                                        lambda: phases[-1]["name"] if phases else None
                                    ),
                                )
                            except subprocess.TimeoutExpired as exc:
                                if exc.stderr:
                                    sys.stderr.write(str(exc.stderr)[-4000:])
                                raise BridgeError(
                                    "cann_bench_utils CMake build timed out",
                                    category=(
                                        "deadline" if deadline.remaining() <= 0 else "infra"
                                    ),
                                ) from exc
                            if built.stderr.strip():
                                sys.stderr.write(built.stderr[-4000:])
                            if built.returncode != 0 or not (
                                stage_utils / "cann_bench_utils" / "_C.so"
                            ).is_file():
                                sys.stderr.write((built.stdout or "")[-4000:])
                                raise BridgeError(
                                    f"cann_bench_utils CMake build failed (exit {built.returncode})"
                                )
                            (stage_slot / "build_meta.json").write_text(
                                json.dumps({
                                    "abiTag": abi_tag,
                                    "arch": arch,
                                    "python": str(python.resolve()),
                                    "cannSetup": str(cann_setup.resolve()),
                                    "pythonVersion": info["pythonVersion"],
                                    "torchVersion": info["torchVersion"],
                                    "torchNpuVersion": info["torchNpuVersion"],
                                    "builtAt": _timestamp(),
                                }, indent=2),
                                encoding="utf-8",
                            )
                            if slot_dir.exists():
                                shutil.rmtree(slot_dir)
                            shutil.move(str(stage_slot), str(slot_dir))
                        utils_built = True

                with _phase(phases, "abi-probe"):
                    if deadline.remaining() <= 0:
                        raise BridgeError(
                            "CANNBench ABI probe exceeded the recipe deadline",
                            category="deadline",
                        )
                    run_env = _host_child_env(values)
                    _scrub_python_startup(run_env)
                    envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
                    if envs_path:
                        run_env["CONDA_ENVS_PATH"] = envs_path
                    run_env["ASCEND_RT_VISIBLE_DEVICES"] = device
                    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
                    run_env["PYTHONNOUSERSITE"] = "1"
                    run_env["PYTHONPATH"] = str(utils_dir)
                    run_env["TILEFOUNDRY_CANN_SERIES"] = resolve_cann_series(values)
                    probe_cmd = f"{shlex.quote(str(python))} -c {shlex.quote(CANNBENCH_ABI_PROBE)}"
                    try:
                        probed = _run_host_bash_monitored(
                            probe_cmd,
                            cwd=source,
                            env=run_env,
                            cann_setup=cann_setup,
                            conda_sh=conda_sh,
                            conda_env=conda_env,
                            timeout=max(1.0, deadline.remaining()),
                            isolate_temp=True,
                            mode="l4",
                            device=device,
                            monitor_sink=health_sink,
                            occupancy_policy=policy,
                            shared_max_ext_procs=max_procs,
                            shared_max_hbm_mb=max_hbm,
                            allow_external_in_shared=True,
                            phase_getter=(
                                lambda: phases[-1]["name"] if phases else None
                            ),
                        )
                    except subprocess.TimeoutExpired as exc:
                        if exc.stderr:
                            sys.stderr.write(str(exc.stderr)[-4000:])
                        raise BridgeError(
                            "cann_bench_utils ABI probe timed out",
                            category="deadline" if deadline.remaining() <= 0 else "infra",
                        ) from exc
                    if probed.stderr.strip():
                        sys.stderr.write(probed.stderr[-4000:])
                    parsed = _extract_last_json(probed.stdout, "cannBenchUtilsCallable")
                    callable_ok = bool(parsed.get("cannBenchUtilsCallable", False))
                    ok = probed.returncode == 0 and ext_so.is_file() and callable_ok
                    if not ok:
                        sys.stderr.write((probed.stdout or "")[-4000:])
                    obs.update({
                        "cannBenchUtilsPresent": ext_so.is_file(),
                        "cannBenchUtilsCallable": callable_ok,
                    })
            finally:
                occ, win = finalize_window()
                obs.update(_window_evidence(occ, win))

        obs.update({
            "exitCodeZero": ok,
            "utilsBuilt": utils_built,
            "utilsReused": utils_reused,
            "errorCategory": "none" if ok else "infra",
        })
        return (0 if ok else (probed.returncode or 1)), obs
    except BridgeError as exc:
        obs["exitCodeZero"] = False
        obs["errorCategory"] = exc.category
        obs["error"] = str(exc)
        if exc.occupancy:
            obs["deviceOccupancy"] = exc.occupancy
            obs["occupancyRecorded"] = (
                exc.occupancy in {"free", "unknown"} or exc.occupancy.startswith("shared(")
            )
        return 75, obs
    except Exception as exc:  # noqa: BLE001 - the observation is the report
        obs["exitCodeZero"] = False
        obs["errorCategory"] = "infra"
        obs["error"] = f"{type(exc).__name__}: {exc}"
        return 75, obs
    finally:
        obs.update({
            "utilsBuilt": utils_built,
            "utilsReused": utils_reused,
            "tests": None,
            "healthMonitoring": health_sink,
            "elapsedSeconds": deadline.elapsed(),
            "phases": phases,
        })


def run_cannbench(values: dict[str, str], timeout: float) -> tuple[int, dict]:
    deadline = _Deadline(timeout)
    phases: list[dict] = []
    source = _source_root()
    tl_identity = tilelang_reuse_identity(values)
    device = require_device()
    repo = cannbench_repo(values)
    wroot = work_root(values, source, repo)
    targets_source, targets = resolve_cannbench_targets(
        values, source, repo, allow_request=True
    )
    cann_setup = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)
    external_probe_limit = l4_external_probe_limit(values)
    reports_root = wroot / "reports"
    report_epoch = time.time()

    obs: dict = {
        "exitCodeZero": False,
        "officialReportValid": False,
        "allCasesPassed": False,
        "allTimingsValid": False,
        "targetCount": len(targets),
        "targetsSource": targets_source,
        "caseCount": None,
        "passedCases": None,
        "timedCases": None,
        "score": None,
        "avgSpeedup": None,
        "results": [],
        "windowOpened": False,
        "windowReleased": False,
        "occupancyRecorded": False,
        "healthMonitoring": {},
        "phases": phases,
        "elapsedSeconds": 0.0,
        "errorCategory": "environment",
        "environment": _environment_observations(values),
        "reportEvidence": {"root": str(reports_root), "reports": []},
    }
    health_sink: dict = obs["healthMonitoring"]
    if tl_identity is not None:
        obs["tilelang"] = tl_identity
    policy = occupancy_policy(values)
    max_procs = shared_max_ext_procs(values)
    max_hbm = shared_max_hbm_mb(values)
    try:
        deadline.require("official CANNBench run")
        # Host profiler serialization comes before the device lease; the two
        # locks protect different resources and are paid from the same budget.
        with _phase(phases, "msprof-lock"):
            with _open_msprof_lock(
                locking.device_lock_root(),
                timeout_seconds=max(1.0, deadline.cap(_lease_wait_seconds())),
            ):
                with _phase(phases, "device-lease"):
                    with _open_device_window(
                        device,
                        wait_seconds=deadline.cap(_lease_wait_seconds()),
                        require_idle=True,
                        occupancy_policy=policy,
                        max_ext_procs=max_procs,
                        max_hbm_mb=max_hbm,
                        enforce_proc_count=True,
                    ) as finalize_window:
                        try:
                            with _phase(phases, "inspect"):
                                if deadline.remaining() <= 0:
                                    raise BridgeError(
                                        "official CANNBench run exceeded the recipe deadline",
                                        category="deadline",
                                    )
                                info = _inspect_conda_cannbench_env(
                                    values, source, timeout=deadline.cap(60.0)
                                )
                                if info.get("missing"):
                                    raise BridgeError(
                                        "missing CANNBench Python packages in TileFoundry env "
                                        f"({', '.join(info['missing'])}); run "
                                        "tilefoundry-npu-cannbench-prep first"
                                    )
                                _, _, utils_dir = _abi_slot(
                                    values, source, repo, wroot, info
                                )
                                if not (
                                    utils_dir / "cann_bench_utils" / "_C.so"
                                ).is_file():
                                    raise BridgeError(
                                        "isolated cann_bench_utils _C.so missing under "
                                        f"{utils_dir}; run tilefoundry-npu-cannbench-prep first"
                                    )

                            with _phase(phases, "idle-preflight"):
                                obs["idlePreflight"] = _l4_idle_preflight(
                                    device,
                                    confirmations=external_probe_limit,
                                    occupancy_policy=policy,
                                    max_ext_procs=max_procs,
                                    max_hbm_mb=max_hbm,
                                )

                            runner_cfg = {
                                "repo": str(repo),
                                "source": str(source),
                                "utils_dir": str(utils_dir),
                                "reports_root": str(reports_root),
                                "device": device,
                                "per_op_timeout": int(
                                    max(1.0, deadline.remaining() - 60.0)
                                ),
                                "pythonpath_parts": list(DEFAULT_PYTHONPATH_PARTS),
                                "tilelang_source": (
                                    tl_identity["source"] if tl_identity else ""
                                ),
                                "targets": [{"task": t, "candidate": c} for t, c in targets],
                            }

                            with _phase(phases, "workload"):
                                with tempfile.TemporaryDirectory(
                                    prefix="tilefoundry-cannbench-"
                                ) as tmp:
                                    tmp_dir = Path(tmp)
                                    runner_py = tmp_dir / "runner.py"
                                    cfg_json = tmp_dir / "runner_cfg.json"
                                    runner_py.write_text(CANNBENCH_RUNNER, encoding="utf-8")
                                    cfg_json.write_text(
                                        json.dumps(runner_cfg), encoding="utf-8"
                                    )

                                    run_env = _host_child_env(values)
                                    _scrub_python_startup(run_env)
                                    run_env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
                                    envs_path = (
                                        values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or ""
                                    ).strip()
                                    if envs_path:
                                        run_env["CONDA_ENVS_PATH"] = envs_path
                                    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
                                    run_env["PYTHONNOUSERSITE"] = "1"
                                    run_env["TILEFOUNDRY_CANN_SERIES"] = resolve_cann_series(values)
                                    cmd = (
                                        'export AGENT_GATES_ACTIVATION_PYTHONPATH="${PYTHONPATH:-}"; '
                                        f"{shlex.quote(str(python))} -E -s "
                                        f"{shlex.quote(str(runner_py))} "
                                        f"{shlex.quote(str(cfg_json))}"
                                    )
                                    if deadline.remaining() <= 0:
                                        raise BridgeError(
                                            "official CANNBench run exceeded the recipe deadline",
                                            category="deadline",
                                        )
                                    try:
                                        completed = _run_host_bash_monitored(
                                            cmd,
                                            cwd=source,
                                            env=run_env,
                                            cann_setup=cann_setup,
                                            conda_sh=conda_sh,
                                            conda_env=conda_env,
                                            timeout=max(1.0, deadline.remaining()),
                                            isolate_temp=True,
                                            mode="l4",
                                            device=device,
                                            monitor_sink=health_sink,
                                            external_probe_limit=external_probe_limit,
                                            occupancy_policy=policy,
                                            shared_max_ext_procs=max_procs,
                                            shared_max_hbm_mb=max_hbm,
                                            allow_external_in_shared=False,
                                            phase_getter=(
                                                lambda: phases[-1]["name"] if phases else None
                                            ),
                                        )
                                    except subprocess.TimeoutExpired as exc:
                                        if exc.stderr:
                                            sys.stderr.write(str(exc.stderr)[-4000:])
                                        raise BridgeError(
                                            "official CANNBench run timed out",
                                            category="deadline",
                                        ) from exc
                        finally:
                            occ, win = finalize_window()
                            obs.update(_window_evidence(occ, win))

        if completed.stderr.strip():
            sys.stderr.write(completed.stderr[-4000:])
        parsed = _extract_last_json(completed.stdout, "officialReportValid")
        report_valid = bool(parsed.get("officialReportValid", False))
        cases_passed = bool(parsed.get("allCasesPassed", False))
        timings_valid = bool(parsed.get("allTimingsValid", False))
        ok = (
            completed.returncode == 0
            and report_valid
            and cases_passed
            and timings_valid
        )
        if not ok:
            sys.stderr.write((completed.stdout or "")[-4000:])
        if ok:
            category = "none"
        elif report_valid and not cases_passed:
            category = "numeric"
        else:
            category = "infra"
        reported_targets = _int_or_none(parsed.get("targetCount"))
        obs.update({
            "exitCodeZero": ok,
            "officialReportValid": report_valid,
            "allCasesPassed": cases_passed,
            "allTimingsValid": timings_valid,
            "targetCount": len(targets) if reported_targets is None else reported_targets,
            "caseCount": _int_or_none(parsed.get("caseCount")),
            "passedCases": _int_or_none(parsed.get("passedCases")),
            "timedCases": _int_or_none(parsed.get("timedCases")),
            "score": _float_or_none(parsed.get("score")),
            "avgSpeedup": _float_or_none(parsed.get("avgSpeedup")),
            "results": list(parsed.get("results") or []),
            "errorCategory": category,
        })
        return (0 if ok else (completed.returncode or 1)), obs
    except BridgeError as exc:
        obs["errorCategory"] = exc.category
        obs["error"] = str(exc)
        if exc.occupancy:
            obs["deviceOccupancy"] = exc.occupancy
            obs["occupancyRecorded"] = (
                exc.occupancy in {"free", "unknown"}
                or exc.occupancy.startswith("shared(")
            )
        # A workload timeout after the lease opened is a bounded failure with
        # preserved partial evidence, not an environment block; the receipt
        # keeps the deadline category so promotion can pause distinctly.
        if exc.category == "deadline" and obs.get("windowOpened"):
            return 1, obs
        return 75, obs
    except Exception as exc:  # noqa: BLE001 - the observation is the report
        obs["errorCategory"] = "infra"
        obs["error"] = f"{type(exc).__name__}: {exc}"
        return 75, obs
    finally:
        obs["reportEvidence"] = _report_evidence(reports_root, report_epoch)
        obs["healthMonitoring"] = health_sink
        obs["elapsedSeconds"] = deadline.elapsed()
        obs["phases"] = phases


_BANNED_TORCH_DELEGATES = frozenset({
    "matmul", "mm", "bmm", "conv2d", "conv3d", "linear", "softmax",
    "layer_norm", "rms_norm", "scaled_dot_product_attention", "silu", "gelu",
})


def _resolve_mcp_config(values: dict[str, str]) -> dict[str, str]:
    mcp_url = (values.get("TILEFOUNDRY_NPU_CANNBENCH_MCP_URL") or "").strip() or DEFAULT_CANNBENCH_MCP_URL
    if not (mcp_url.startswith("https://") or mcp_url.startswith("http://")):
        raise BridgeError(f"TILEFOUNDRY_NPU_CANNBENCH_MCP_URL must start with https:// or http://: {mcp_url!r}")

    token = (values.get("TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN") or "").strip()
    token_file = (values.get("TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE") or "").strip()
    if not token and token_file:
        tpath = Path(token_file)
        if not tpath.is_file():
            raise BridgeError(f"TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE does not exist: {token_file}")
        token = tpath.read_text(encoding="utf-8").strip()
    if not token:
        raise BridgeError(
            "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN or TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE is required for remote CANNBench submission"
        )

    consent = (values.get("TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT") or "").strip()
    if consent and not MCP_CONSENT_VERSION_RE.fullmatch(consent):
        raise BridgeError(
            f"TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT must match v<N> (e.g. v1): {consent!r}"
        )

    return {
        "mcp_url": mcp_url,
        "token": token,
        "benchmark_version": (values.get("TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION") or "").strip(),
        "pool_id": (values.get("TILEFOUNDRY_NPU_CANNBENCH_POOL_ID") or "").strip(),
        "aggregation_token": (values.get("TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN") or "").strip(),
        "independent_env_consent": consent,
    }


def _parse_proto_operator(proto_path: Path) -> tuple[str, str]:
    text = proto_path.read_text(encoding="utf-8")
    name_m = re.search(r"^\s*name:\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*$", text, re.MULTILINE)
    schema_m = re.search(r"^\s*schema:\s*['\"]?([A-Za-z0-9_]+)\s*\(", text, re.MULTILINE)
    if not name_m or not schema_m:
        raise BridgeError(f"cannot parse operator name/schema from {proto_path}")
    return name_m.group(1), schema_m.group(1)


def _check_candidate_compliance(candidate: Path, function_name: str) -> list[str]:
    try:
        src = candidate.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(candidate))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return [f"EVAL-CANN-004: invalid Python candidate ({exc})"]

    violations: list[str] = []
    exported_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            exported_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    exported_names.add(tgt.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exported_names.add(alias.asname or alias.name)
    if function_name not in exported_names:
        violations.append(f"EVAL-CANN-005: missing callable {function_name!r} in {candidate.name}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "cpu" and not node.args and not node.keywords:
                violations.append("SUB-BEH-004: .cpu() call detected (CPU fallback forbidden)")
            elif attr == "data_ptr":
                violations.append("SUB-BEH-005: data_ptr() call detected (pointer caching forbidden)")
            elif attr in _BANNED_TORCH_DELEGATES:
                owner = node.func.value
                if isinstance(owner, ast.Name) and owner.id in {"F", "torch"}:
                    violations.append(f"SUB-BEH-001: {owner.id}.{attr}() delegation forbidden")
                elif isinstance(owner, ast.Attribute) and owner.attr in {"functional", "aten"}:
                    violations.append(f"SUB-BEH-001: *.{owner.attr}.{attr}() delegation forbidden")
    return violations


def _build_remote_submission_zip(
    candidate: Path, op_name: str, func_name: str, out_dir: Path
) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", func_name).strip("_") or "operator"
    pkg_root = out_dir / "pkg"
    cann_pkg = pkg_root / "cann_bench"
    cann_pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, cann_pkg / f"{slug}.py")
    (cann_pkg / "__init__.py").write_text(
        f"from .{slug} import {func_name}\n\n__all__ = [{func_name!r}]\n",
        encoding="utf-8",
    )
    (pkg_root / "setup.py").write_text(
        "from setuptools import find_packages, setup\n\n"
        "setup(name='cann_bench', version='1.0.0', packages=find_packages())\n",
        encoding="utf-8",
    )
    build_sh = pkg_root / "build.sh"
    build_sh.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "python3 -m pip wheel --no-deps --no-build-isolation -w dist .\n",
        encoding="utf-8",
    )
    build_sh.chmod(0o755)

    zip_path = out_dir / f"{slug}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(p for p in pkg_root.rglob("*") if p.is_file()):
            rel = file_path.relative_to(pkg_root).as_posix()
            zf.write(file_path, arcname=rel)
    return zip_path


def _parse_mcp_response(raw_body: str, content_type: str) -> dict:
    if "text/event-stream" in content_type:
        last_payload: dict | None = None
        for line in raw_body.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    last_payload = json.loads(data)
        if last_payload is None:
            raise BridgeError("empty SSE response from BenchSite MCP server")
        return last_payload
    return json.loads(raw_body)


def _mcp_http_post(
    url: str,
    token: str,
    payload: dict,
    *,
    session_id: str = "",
    timeout: float = 30.0,
) -> tuple[dict, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            ctype = resp.headers.get("Content-Type", "")
            next_sid = resp.headers.get("Mcp-Session-Id", session_id)
            if not body.strip():
                return {}, next_sid
            return _parse_mcp_response(body, ctype), next_sid
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise BridgeError(f"BenchSite MCP HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise BridgeError(f"BenchSite MCP request failed: {exc}") from exc


def _extract_mcp_tool_json(resp: dict) -> dict:
    if "error" in resp:
        raise BridgeError(f"BenchSite MCP JSON-RPC error: {resp['error']}")
    result = resp.get("result") or {}
    if result.get("isError"):
        raise BridgeError(f"BenchSite MCP tool error: {result}")
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            text = (item.get("text") or "").strip()
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except ValueError:
                return {"message": text}
    return dict(result)


def _mcp_submit_kernel(
    mcp_cfg: dict[str, str],
    *,
    task: str,
    op_name: str,
    zip_path: Path,
    timeout: float,
) -> dict:
    url = mcp_cfg["mcp_url"]
    token = mcp_cfg["token"]
    init_resp, sid = _mcp_http_post(
        url,
        token,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "tilefoundry-npu-adapter", "version": "1.0.0"},
            },
        },
        timeout=min(timeout, 30.0),
    )
    if "error" in init_resp:
        raise BridgeError(f"BenchSite MCP initialize failed: {init_resp['error']}")
    _mcp_http_post(
        url,
        token,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        session_id=sid,
        timeout=min(timeout, 15.0),
    )

    submit_args: dict[str, object] = {
        "task": task,
        "operator": op_name,
        "filename": zip_path.name,
        "archive_base64": base64.b64encode(zip_path.read_bytes()).decode("ascii"),
    }
    if mcp_cfg["benchmark_version"]:
        submit_args["benchmark_version"] = mcp_cfg["benchmark_version"]
    if mcp_cfg["pool_id"]:
        submit_args["requested_pool_id"] = mcp_cfg["pool_id"]
    if mcp_cfg["independent_env_consent"]:
        submit_args["independent_environment_consent_version"] = mcp_cfg["independent_env_consent"]

    call_resp, _ = _mcp_http_post(
        url,
        token,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "submit_kernel", "arguments": submit_args},
        },
        session_id=sid,
        timeout=timeout,
    )
    return _extract_mcp_tool_json(call_resp)


def run_cannbench_remote(values: dict[str, str], timeout: float) -> tuple[int, dict]:
    source = _source_root()
    repo = cannbench_repo(values)
    wroot = work_root(values, source, repo)
    targets_source, targets = resolve_cannbench_targets(values, source, repo)
    mcp_cfg = _resolve_mcp_config(values)

    submissions_root = wroot / "submissions"
    submissions_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_compliant = True
    all_archives_valid = True
    all_submitted = True
    results = []

    for index, (task, candidate_rel) in enumerate(targets):
        op_name, func_name = _parse_proto_operator(repo / "tasks" / task / "proto.yaml")
        candidate = (source / candidate_rel).resolve()
        violations = _check_candidate_compliance(candidate, func_name)

        sub_dir = submissions_root / f"{stamp}_{task.replace('/', '__')}_{index}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        zip_path = _build_remote_submission_zip(candidate, op_name, func_name, sub_dir)
        archive_valid = zip_path.is_file() and zip_path.stat().st_size > 0

        if violations:
            all_compliant = False
            all_submitted = False
            results.append({
                "task": task,
                "operator": op_name,
                "candidate": candidate_rel,
                "archivePath": str(zip_path),
                "archiveValid": archive_valid,
                "compliant": False,
                "violations": violations,
                "submitted": False,
                "submissionId": "",
                "status": "compliance_failed",
            })
            continue

        if not archive_valid:
            all_archives_valid = False
            all_submitted = False
            results.append({
                "task": task,
                "operator": op_name,
                "candidate": candidate_rel,
                "archivePath": str(zip_path),
                "archiveValid": False,
                "compliant": True,
                "violations": [],
                "submitted": False,
                "submissionId": "",
                "status": "archive_failed",
            })
            continue

        receipt = _mcp_submit_kernel(
            mcp_cfg,
            task=task,
            op_name=op_name,
            zip_path=zip_path,
            timeout=max(30.0, timeout / max(1, len(targets))),
        )
        (sub_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        sub_id = str(receipt.get("submission_id") or receipt.get("id") or "")
        status = str(receipt.get("status") or "submitted")
        submitted = bool(sub_id or status)
        if not submitted:
            all_submitted = False
        results.append({
            "task": task,
            "operator": op_name,
            "candidate": candidate_rel,
            "archivePath": str(zip_path),
            "archiveValid": True,
            "compliant": True,
            "violations": [],
            "submitted": submitted,
            "submissionId": sub_id,
            "status": status,
            "benchmarkVersion": mcp_cfg["benchmark_version"] or str(receipt.get("benchmark_version") or ""),
            "poolId": mcp_cfg["pool_id"] or str(receipt.get("pool_id") or ""),
        })

    ok = all_compliant and all_archives_valid and all_submitted and bool(results)
    obs = {
        "exitCodeZero": ok,
        "compliancePassed": all_compliant,
        "archiveValid": all_archives_valid,
        "mcpSubmitted": all_submitted and bool(results),
        "targetCount": len(targets),
        "targetsSource": targets_source,
        "mcpUrl": mcp_cfg["mcp_url"],
        "environment": _environment_observations(values),
        "results": results,
    }
    return (0 if ok else 1), obs


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    timeout = None
    if "--timeout" in args:
        idx = args.index("--timeout")
        try:
            timeout = float(args[idx + 1])
        except (IndexError, ValueError):
            _die(2, "--timeout needs a number of seconds")
        del args[idx : idx + 2]

    if args == ["--check-config"]:
        return check_config(check_device=False)
    if args == ["--check-config-device"]:
        return check_config(check_device=True)
    if args == ["--check-config-cannbench"]:
        return check_config(check_device=True, check_cannbench=True)
    if args == ["--check-config-cannbench-remote"]:
        return check_config(check_device=False, check_cannbench_remote=True)
    if args == ["--check-config-l3"]:
        return check_config(check_device=True, check_l3=True)

    try:
        values = machine_values()
    except BridgeError as exc:
        _die(75, str(exc))

    if args == ["--cpu"]:
        rc, obs = run_cpu(values, DEFAULT_CPU_TIMEOUT if timeout is None else timeout)
        print(json.dumps(obs, sort_keys=True))
        return rc
    if args == ["--device"]:
        try:
            rc, obs = run_device(values, DEFAULT_DEVICE_TIMEOUT if timeout is None else timeout)
        except BridgeError as exc:
            _die(75, str(exc))
        print(json.dumps(obs, sort_keys=True))
        return rc
    if args == ["--device-fast"]:
        try:
            rc, obs = run_device(
                values, DEFAULT_DEVICE_TIMEOUT if timeout is None else timeout, fast=True
            )
        except BridgeError as exc:
            _die(75, str(exc))
        print(json.dumps(obs, sort_keys=True))
        return rc
    if args == ["--l3-promotion"]:
        try:
            rc, obs = run_l3_promotion(
                values, l3_timeout(values) if timeout is None else timeout
            )
        except BridgeError as exc:
            _die(75, str(exc))
        print(json.dumps(obs, sort_keys=True))
        return rc
    if args == ["--cannbench-prep"]:
        try:
            rc, obs = run_cannbench_prep(values, DEFAULT_PREP_TIMEOUT if timeout is None else timeout)
        except BridgeError as exc:
            _die(75, str(exc))
        print(json.dumps(obs, sort_keys=True))
        return rc
    if args == ["--cannbench"]:
        try:
            rc, obs = run_cannbench(values, DEFAULT_CANNBENCH_TIMEOUT if timeout is None else timeout)
        except BridgeError as exc:
            _die(75, str(exc))
        print(json.dumps(obs, sort_keys=True))
        return rc
    if args == ["--cannbench-remote"]:
        try:
            rc, obs = run_cannbench_remote(values, DEFAULT_REMOTE_TIMEOUT if timeout is None else timeout)
        except BridgeError as exc:
            _die(75, str(exc))
        print(json.dumps(obs, sort_keys=True))
        return rc

    _die(
        2,
        f"usage: {PROG} (--cpu | --device | --device-fast | --l3-promotion | --cannbench-prep | "
        "--cannbench | --cannbench-remote | --check-config | --check-config-device | "
        "--check-config-l3 | --check-config-cannbench | --check-config-cannbench-remote) "
        "[--timeout SECONDS]",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
