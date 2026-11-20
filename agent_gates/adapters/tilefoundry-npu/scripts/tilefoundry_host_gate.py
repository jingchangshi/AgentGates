#!/usr/bin/env python3
"""Host-local TileFoundry CPU, Ascend NPU, and CANNBench verification gates."""
from __future__ import annotations

import ast
import base64
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import NoReturn
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
    "TILEFOUNDRY_NPU_LLM_MAX_EFFORT",
    "TILEFOUNDRY_NPU_LLM_EFFORT_CAP",
    "TILEFOUNDRY_NPU_LLM_PASS_REASONING_EFFORT",
    "TILEFOUNDRY_NPU_LLM_STREAM",
    "TILEFOUNDRY_NPU_LLM_MAX_CONCURRENCY",
    "TILEFOUNDRY_NPU_LLM_EXPLOIT_TIMEOUT_SEC",
    "TILEFOUNDRY_NPU_LLM_REASONING_TIMEOUT_SEC",
    "TILEFOUNDRY_NPU_LLM_CLI_BIN",
    "TILEFOUNDRY_NPU_LLM_CLI_TEMPLATE",
    "TILEFOUNDRY_NPU_AGY_BIN",
    "TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL",
    "TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL",
    "TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL",
    "TILEFOUNDRY_NPU_AGY_COMMON_ARGS",
    "TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY",
)

TOOLCHAIN_KEYS = (
    "TILEFOUNDRY_NPU_CANN_SETUP",
    "TILEFOUNDRY_NPU_CONDA_SH",
    "TILEFOUNDRY_NPU_CONDA_ENV",
    "TILEFOUNDRY_NPU_CONDA_ENVS_PATH",
    "TILEFOUNDRY_NPU_PYTHON",
    "TILEFOUNDRY_NPU_TILELANG_SOURCE",
    "TILEFOUNDRY_NPU_TILELANG_BUILD",
    "TILEFOUNDRY_NPU_CPU_TESTS",
    "TILEFOUNDRY_NPU_DEVICE_TESTS",
    "TILEFOUNDRY_NPU_MSPROF_LOCK",
    "TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE",
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
    "TILEFOUNDRY_NPU_TILELANG_REPO",
    "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO",
    "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO",
    "TILEFOUNDRY_NPU_ASC_DEVKIT_REPO",
)

OPTIONAL_REPO_AND_PYTHON_KEYS = (
    "TILEFOUNDRY_NPU_PYTHON",
    "TILEFOUNDRY_NPU_TILELANG_REPO",
    "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO",
    "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO",
    "TILEFOUNDRY_NPU_ASC_DEVKIT_REPO",
)

OPTIONAL_FORWARDED_KEYS = OPTIONAL_REPO_AND_PYTHON_KEYS + OPTIONAL_AGENT_LLM_KEYS

OPTIONAL_ABS_PATH_KEYS = (
    "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE",
    "TILEFOUNDRY_NPU_AGY_BIN",
    "TILEFOUNDRY_NPU_TILELANG_REPO",
    "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO",
    "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO",
    "TILEFOUNDRY_NPU_ASC_DEVKIT_REPO",
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
    "tests/ascend/ir",
    "tests/ascend/passes",
    "tests/ascend/operators",
)

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
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(cfg["device"])
        env["TILELANG_DISABLE_CACHE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        tl_src = (cfg.get("tilelang_source") or "").strip()
        if tl_src:
            env.setdefault("USE_CUDA", "0")
        pp_entries = [str(stage / "submission"), str(utils_dir), str(bench_src)]
        if tl_src:
            pp_entries.append(tl_src)
        pp_entries.extend(tf_paths)
        if env.get("PYTHONPATH"):
            pp_entries.append(env["PYTHONPATH"])
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


class BridgeError(RuntimeError):
    EXIT = adapter_runtime.BLOCKED_ENV


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


def cpu_tests(values: dict[str, str]) -> tuple[str, ...]:
    return _parse_test_list(values.get("TILEFOUNDRY_NPU_CPU_TESTS"), DEFAULT_CPU_TESTS)


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
    values: dict[str, str], source: Path, repo: Path
) -> tuple[str, tuple[tuple[str, str], ...]]:
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


def _device_occupancy(device: str) -> str:
    try:
        processes, _ = adapter_runtime.processes_on(int(device))
    except adapter_runtime.BlockedEnv as exc:
        if exclusive_requested():
            raise BridgeError(str(exc)) from exc
        return "unknown"
    return "free" if not processes else f"shared({len(processes)})"


@contextmanager
def _open_device_window(device: str, *, warmup_seconds: float | None = None):
    lease_path = os.environ.get("AGENT_GATES_DEVICE_LEASE_PATH", "").strip()
    lease_resource = os.environ.get("AGENT_GATES_DEVICE_LEASE_RESOURCE", f"npu:{device}").strip()
    if not lease_path:
        raise BridgeError("AGENT_GATES_DEVICE_LEASE_PATH is required for windowed device runs")
    try:
        wait_seconds = float(os.environ.get("AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS", "60"))
    except ValueError as exc:
        raise BridgeError("invalid AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS") from exc

    Path(lease_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lease_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    started = time.monotonic()
    deadline = started + max(0.0, wait_seconds)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BridgeError(
                        f"device lease {lease_resource} did not open within {wait_seconds}s"
                    ) from exc
                time.sleep(min(0.05, remaining))

        opened = time.monotonic()
        window_started = _timestamp()
        occ = _device_occupancy(device)
        probe_seconds = round(time.monotonic() - opened, 3)
        if exclusive_requested() and occ.startswith("shared("):
            raise BridgeError(
                f"device {lease_resource} is {occ} and TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE is set"
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
            if warmup_seconds is not None:
                win["warmupSeconds"] = round(float(warmup_seconds), 3)
            return occ, win

        yield finalize_window
    finally:
        os.close(fd)


@contextmanager
def _open_msprof_lock(wroot: Path, *, timeout_seconds: float = 3600.0):
    """Serialize host-wide CANN msprof profiling across concurrent NPU device runs."""
    wroot.mkdir(parents=True, exist_ok=True)
    lock_path = wroot / ".msprof.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
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


def _inspect_conda_cannbench_env(values: dict[str, str], source: Path) -> dict:
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)
    run_env = _host_child_env(values)
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
        timeout=60.0,
        isolate_temp=True,
    )
    info = _extract_last_json(completed.stdout, "missing")
    if completed.returncode != 0 or not info:
        sys.stderr.write((completed.stderr or completed.stdout)[-4000:])
        raise BridgeError("failed to inspect TileFoundry Conda Python dependencies")
    return info


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
    if not include_agent_llm:
        for key in OPTIONAL_AGENT_LLM_KEYS:
            run_env.pop(key, None)


def _validate_optional_agent_keys(values: dict[str, str]) -> None:
    conc = (values.get("TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY") or "").strip()
    if conc and (not conc.isdigit() or int(conc) < 1):
        raise BridgeError(
            f"TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY must be a positive integer: {conc!r}"
        )
    for key in OPTIONAL_ABS_PATH_KEYS:
        val = (values.get(key) or "").strip()
        if val and not Path(val).is_absolute():
            raise BridgeError(f"{key} must be an absolute path when set: {val!r}")


def check_config(
    *,
    check_device: bool = False,
    check_cannbench: bool = False,
    check_cannbench_remote: bool = False,
) -> int:
    try:
        values = machine_values()
        for name in REQUIRED:
            _required(values, name)
        build_subdir(values)
        _validate_optional_agent_keys(values)
        _binary_flag(values, "TILEFOUNDRY_NPU_MSPROF_LOCK")
        _binary_flag(values, "TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE")
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
        for rel in cpu_tests(values):
            if not (source / rel).exists():
                raise BridgeError(f"CPU test path missing in {source}: {rel}")
        for rel in device_tests(values):
            if not (source / rel).exists():
                raise BridgeError(f"device test path missing in {source}: {rel}")
        tilelang_reuse_identity(values)
        if check_cannbench or check_cannbench_remote:
            repo = cannbench_repo(values)
            work_root(values, source, repo)
            if check_cannbench:
                cannbench_arch(values)
            resolve_cannbench_targets(values, source, repo)
        if check_cannbench_remote:
            _resolve_mcp_config(values)
        if check_device:
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
    tests = cpu_tests(values)
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)

    with tempfile.TemporaryDirectory(prefix="tilefoundry-cpu-") as tmp:
        report = Path(tmp) / "junit.xml"
        run_env = _host_child_env(values)
        envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
        if envs_path:
            run_env["CONDA_ENVS_PATH"] = envs_path
        _forward_optional_env(run_env, values, include_agent_llm=False)
        run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        run_env["PYTHONDONTWRITEBYTECODE"] = "1"
        run_env["PYTHONNOUSERSITE"] = "1"
        run_env["TILELANG_DISABLE_CACHE"] = "1"
        run_env["PYTHONPATH"] = _effective_pythonpath(tl_identity)
        if tl_identity is not None:
            run_env.setdefault("USE_CUDA", "0")
        # CPU suite deliberately does NOT source CANN set_env.sh because
        # CANN's native libraries collide with ortools CP-SAT in this env.
        cmd = " ".join(
            [shlex.quote(str(python)), "-m", "pytest", "-p", "no:cacheprovider",
             f"--junit-xml={shlex.quote(str(report))}"]
            + [shlex.quote(t) for t in tests]
        )
        try:
            completed = adapter_runtime.run_host_bash(
                cmd,
                cwd=source,
                env=run_env,
                cann_setup=None,
                conda_sh=conda_sh,
                conda_env=conda_env,
                timeout=timeout,
                isolate_temp=True,
            )
        except subprocess.TimeoutExpired as exc:
            if exc.stderr:
                sys.stderr.write(str(exc.stderr)[-4000:])
            print(f"{PROG}: CPU pytest timed out after {timeout}s", file=sys.stderr)
            timeout_obs = _cpu_observations(False, 0, 0, 0, 0)
            if tl_identity is not None:
                timeout_obs["tilelang"] = tl_identity
            return 1, timeout_obs

        counts = _junit_counts(report)
        if counts is None:
            sys.stderr.write((completed.stderr or completed.stdout)[-4000:])
            print(f"{PROG}: missing JUnit report at {report}", file=sys.stderr)
            missing_obs = _cpu_observations(False, 0, 0, 0, 0)
            if tl_identity is not None:
                missing_obs["tilelang"] = tl_identity
            return 1, missing_obs

        succeeded = (
            completed.returncode == 0
            and counts["collected"] > 0
            and counts["failed"] == 0
            and counts["skipped"] < counts["collected"]
        )
        obs = _cpu_observations(
            succeeded,
            counts["collected"],
            counts["passed"],
            counts["failed"],
            counts["skipped"],
        )
        if tl_identity is not None:
            obs["tilelang"] = tl_identity
        if not succeeded:
            sys.stderr.write((completed.stdout + "\n" + completed.stderr)[-4000:])
        return (0 if succeeded else 1), obs


def _redact_llm_credential(text: str | bytes, credential: str) -> str:
    """Redact the credential from device subprocess output."""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    if credential:
        text = text.replace(credential, "[redacted]")
    return text


def _window_evidence(occupancy: str, window: dict) -> dict:
    return {
        "windowOpened": True,
        "windowReleased": True,
        "occupancyRecorded": occupancy in {"free", "unknown"} or occupancy.startswith("shared("),
        "deviceWindow": window,
    }


def run_device(values: dict[str, str], timeout: float, *, fast: bool = False) -> tuple[int, dict]:
    source = _source_root()
    tl_identity = tilelang_reuse_identity(values)
    device = require_device()
    cann_setup = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)
    tests = device_tests(values)
    if fast and not tests:
        raise BridgeError(
            "tilefoundry-npu-device-fast requires TILEFOUNDRY_NPU_DEVICE_TESTS"
        )
    msprof_lock = _binary_flag(values, "TILEFOUNDRY_NPU_MSPROF_LOCK")
    cache_enabled = _binary_flag(values, "TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE")

    started = time.monotonic()
    lock = (
        _open_msprof_lock(locking.lock_root(), timeout_seconds=max(1.0, timeout - 30.0))
        if msprof_lock
        else nullcontext()
    )
    with lock:
        with _open_device_window(device) as finalize_window:
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
                shim = (run_env.get("TF_SURROGATE_SHIM_DIR") or "").strip() if cache_enabled else ""
                run_env["PYTHONPATH"] = _effective_pythonpath(tl_identity, shim=shim)
                if tl_identity is not None:
                    run_env.setdefault("USE_CUDA", "0")
                    run_env["TILEFOUNDRY_EXPECTED_TILELANG_SOURCE"] = tl_identity["source"]

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
                    completed = adapter_runtime.run_host_bash(
                        cmd,
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
                    if fast:
                        obs = {
                            "exitCodeZero": False,
                            "testsCollected": 0,
                            "testsPassed": 0,
                            "testsFailed": 0,
                            "testsSkipped": 0,
                            **_window_evidence(occ, win),
                        }
                    else:
                        obs = {
                            "exitCodeZero": False,
                            **dict.fromkeys(DEVICE_PROBE_FLAGS, False),
                            **_window_evidence(occ, win),
                        }
                        if tl_identity is not None:
                            obs["tilelangReused"] = False
                    if tl_identity is not None:
                        obs["tilelang"] = tl_identity
                    return 1, obs

                occ, win = finalize_window()
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
                    )
                    if not ok:
                        sys.stderr.write(
                            _redact_llm_credential(
                                completed.stdout + "\n" + completed.stderr, credential
                            )[-4000:]
                        )
                    obs = {
                        "exitCodeZero": ok,
                        "testsCollected": counts["collected"],
                        "testsPassed": counts["passed"],
                        "testsFailed": counts["failed"],
                        "testsSkipped": counts["skipped"],
                        **_window_evidence(occ, win),
                    }
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


def run_cannbench_prep(values: dict[str, str], timeout: float) -> tuple[int, dict]:
    source = _source_root()
    tl_identity = tilelang_reuse_identity(values)
    device = require_device()
    repo = cannbench_repo(values)
    wroot = work_root(values, source, repo)
    arch = cannbench_arch(values)
    cann_setup = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)

    warmup_started = time.monotonic()
    info = _inspect_conda_cannbench_env(values, source)
    missing = info.get("missing") or []
    if missing:
        raise BridgeError(
            "missing CANNBench Python packages in TileFoundry env "
            f"({', '.join(missing)}); run {ADAPTER_DIR / 'scripts' / 'setup_cannbench_deps.sh'}"
        )

    abi_tag, slot_dir, utils_dir = _abi_slot(values, source, repo, wroot, info)
    ext_so = utils_dir / "cann_bench_utils" / "_C.so"
    meta_file = slot_dir / "build_meta.json"

    utils_built = False
    utils_reused = False
    if ext_so.is_file() and meta_file.is_file():
        utils_reused = True
    else:
        build_parent = wroot / "cannbench-utils"
        build_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".tmp-{abi_tag}-", dir=str(build_parent)) as stage_tmp:
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
                " ".join(["cmake", "--build", shlex.quote(str(stage_utils / "build")), "--parallel", "4"]),
            ])
            build_env = _host_child_env(values)
            envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
            if envs_path:
                build_env["CONDA_ENVS_PATH"] = envs_path
            build_env["PYTHONDONTWRITEBYTECODE"] = "1"
            build_env["PYTHONNOUSERSITE"] = "1"
            try:
                built = adapter_runtime.run_host_bash(
                    build_cmd,
                    cwd=repo,
                    env=build_env,
                    cann_setup=cann_setup,
                    conda_sh=conda_sh,
                    conda_env=conda_env,
                    timeout=min(timeout, 600.0),
                    isolate_temp=True,
                )
            except subprocess.TimeoutExpired as exc:
                if exc.stderr:
                    sys.stderr.write(str(exc.stderr)[-4000:])
                raise BridgeError("cann_bench_utils CMake build timed out") from exc
            if built.stderr.strip():
                sys.stderr.write(built.stderr[-4000:])
            if built.returncode != 0 or not (stage_utils / "cann_bench_utils" / "_C.so").is_file():
                sys.stderr.write((built.stdout or "")[-4000:])
                raise BridgeError(f"cann_bench_utils CMake build failed (exit {built.returncode})")
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

    warmup_seconds = round(time.monotonic() - warmup_started, 3)

    with _open_device_window(device, warmup_seconds=warmup_seconds) as finalize_window:
        run_env = _host_child_env(values)
        envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
        if envs_path:
            run_env["CONDA_ENVS_PATH"] = envs_path
        run_env["ASCEND_RT_VISIBLE_DEVICES"] = device
        run_env["PYTHONDONTWRITEBYTECODE"] = "1"
        run_env["PYTHONNOUSERSITE"] = "1"
        run_env["PYTHONPATH"] = str(utils_dir)
        probe_cmd = f"{shlex.quote(str(python))} -c {shlex.quote(CANNBENCH_ABI_PROBE)}"
        try:
            probed = adapter_runtime.run_host_bash(
                probe_cmd,
                cwd=source,
                env=run_env,
                cann_setup=cann_setup,
                conda_sh=conda_sh,
                conda_env=conda_env,
                timeout=max(60.0, timeout - warmup_seconds),
                isolate_temp=True,
            )
        except subprocess.TimeoutExpired as exc:
            occ, win = finalize_window()
            if exc.stderr:
                sys.stderr.write(str(exc.stderr)[-4000:])
            return 1, {
                "exitCodeZero": False,
                "depsReady": True,
                "cannBenchUtilsPresent": ext_so.is_file(),
                "cannBenchUtilsCallable": False,
                "utilsBuilt": utils_built,
                "utilsReused": utils_reused,
                "utilsRoot": str(utils_dir),
                "arch": arch,
                "pythonVersion": info["pythonVersion"],
                "torchVersion": info["torchVersion"],
                "torchNpuVersion": info["torchNpuVersion"],
                "windowOpened": True,
                "windowReleased": True,
                "occupancyRecorded": occ in {"free", "unknown"} or occ.startswith("shared("),
                "deviceWindow": win,
            }

        occ, win = finalize_window()
        if probed.stderr.strip():
            sys.stderr.write(probed.stderr[-4000:])
        parsed = _extract_last_json(probed.stdout, "cannBenchUtilsCallable")
        callable_ok = bool(parsed.get("cannBenchUtilsCallable", False))
        ok = probed.returncode == 0 and ext_so.is_file() and callable_ok
        if not ok:
            sys.stderr.write((probed.stdout or "")[-4000:])
        obs = {
            "exitCodeZero": ok,
            "depsReady": True,
            "cannBenchUtilsPresent": ext_so.is_file(),
            "cannBenchUtilsCallable": callable_ok,
            "utilsBuilt": utils_built,
            "utilsReused": utils_reused,
            "utilsRoot": str(utils_dir),
            "arch": arch,
            "pythonVersion": info["pythonVersion"],
            "torchVersion": info["torchVersion"],
            "torchNpuVersion": info["torchNpuVersion"],
            "windowOpened": True,
            "windowReleased": True,
            "occupancyRecorded": occ in {"free", "unknown"} or occ.startswith("shared("),
            "deviceWindow": win,
        }
        if tl_identity is not None:
            obs["tilelang"] = tl_identity
        return (0 if ok else (probed.returncode or 1)), obs


def run_cannbench(values: dict[str, str], timeout: float) -> tuple[int, dict]:
    source = _source_root()
    tl_identity = tilelang_reuse_identity(values)
    device = require_device()
    repo = cannbench_repo(values)
    wroot = work_root(values, source, repo)
    targets_source, targets = resolve_cannbench_targets(values, source, repo)
    cann_setup = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)

    warmup_started = time.monotonic()
    info = _inspect_conda_cannbench_env(values, source)
    if info.get("missing"):
        raise BridgeError(
            "missing CANNBench Python packages in TileFoundry env "
            f"({', '.join(info['missing'])}); run tilefoundry-npu-cannbench-prep first"
        )
    _, _, utils_dir = _abi_slot(values, source, repo, wroot, info)
    if not (utils_dir / "cann_bench_utils" / "_C.so").is_file():
        raise BridgeError(
            f"isolated cann_bench_utils _C.so missing under {utils_dir}; run tilefoundry-npu-cannbench-prep first"
        )
    warmup_seconds = round(time.monotonic() - warmup_started, 3)

    reports_root = wroot / "reports"
    runner_cfg = {
        "repo": str(repo),
        "source": str(source),
        "utils_dir": str(utils_dir),
        "reports_root": str(reports_root),
        "device": device,
        "per_op_timeout": int(max(60.0, timeout - warmup_seconds - 60.0)),
        "pythonpath_parts": list(DEFAULT_PYTHONPATH_PARTS),
        "tilelang_source": tl_identity["source"] if tl_identity else "",
        "targets": [{"task": t, "candidate": c} for t, c in targets],
    }

    with _open_msprof_lock(locking.lock_root()):
        with _open_device_window(device, warmup_seconds=warmup_seconds) as finalize_window:
            with tempfile.TemporaryDirectory(prefix="tilefoundry-cannbench-") as tmp:
                tmp_dir = Path(tmp)
                runner_py = tmp_dir / "runner.py"
                cfg_json = tmp_dir / "runner_cfg.json"
                runner_py.write_text(CANNBENCH_RUNNER, encoding="utf-8")
                cfg_json.write_text(json.dumps(runner_cfg), encoding="utf-8")

                run_env = _host_child_env(values)
                run_env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
                envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
                if envs_path:
                    run_env["CONDA_ENVS_PATH"] = envs_path
                run_env["PYTHONDONTWRITEBYTECODE"] = "1"
                run_env["PYTHONNOUSERSITE"] = "1"
                cmd = f"{shlex.quote(str(python))} {shlex.quote(str(runner_py))} {shlex.quote(str(cfg_json))}"
                try:
                    completed = adapter_runtime.run_host_bash(
                        cmd,
                        cwd=source,
                        env=run_env,
                        cann_setup=cann_setup,
                        conda_sh=conda_sh,
                        conda_env=conda_env,
                        timeout=max(60.0, timeout - warmup_seconds),
                        isolate_temp=True,
                    )
                except subprocess.TimeoutExpired as exc:
                    occ, win = finalize_window()
                    if exc.stderr:
                        sys.stderr.write(str(exc.stderr)[-4000:])
                    timeout_obs = {
                        "exitCodeZero": False,
                        "officialReportValid": False,
                        "allCasesPassed": False,
                        "allTimingsValid": False,
                        "targetCount": len(targets),
                        "targetsSource": targets_source,
                        "caseCount": 0,
                        "passedCases": 0,
                        "timedCases": 0,
                        "score": 0.0,
                        "avgSpeedup": 0.0,
                        "results": [],
                        "windowOpened": True,
                        "windowReleased": True,
                        "occupancyRecorded": occ in {"free", "unknown"} or occ.startswith("shared("),
                        "deviceWindow": win,
                    }
                    if tl_identity is not None:
                        timeout_obs["tilelang"] = tl_identity
                    return 1, timeout_obs

            occ, win = finalize_window()
            if completed.stderr.strip():
                sys.stderr.write(completed.stderr[-4000:])
            parsed = _extract_last_json(completed.stdout, "officialReportValid")
            report_valid = bool(parsed.get("officialReportValid", False))
            cases_passed = bool(parsed.get("allCasesPassed", False))
            timings_valid = bool(parsed.get("allTimingsValid", False))
            ok = completed.returncode == 0 and report_valid and cases_passed and timings_valid
            if not ok:
                sys.stderr.write((completed.stdout or "")[-4000:])

            obs = {
                "exitCodeZero": ok,
                "officialReportValid": report_valid,
                "allCasesPassed": cases_passed,
                "allTimingsValid": timings_valid,
                "targetCount": int(parsed.get("targetCount", len(targets))),
                "targetsSource": targets_source,
                "caseCount": int(parsed.get("caseCount", 0)),
                "passedCases": int(parsed.get("passedCases", 0)),
                "timedCases": int(parsed.get("timedCases", 0)),
                "score": float(parsed.get("score", 0.0)),
                "avgSpeedup": float(parsed.get("avgSpeedup", 0.0)),
                "results": list(parsed.get("results") or []),
                "windowOpened": True,
                "windowReleased": True,
                "occupancyRecorded": occ in {"free", "unknown"} or occ.startswith("shared("),
                "deviceWindow": win,
            }
            if tl_identity is not None:
                obs["tilelang"] = tl_identity
            return (0 if ok else (completed.returncode or 1)), obs


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
        f"usage: {PROG} (--cpu | --device | --device-fast | --cannbench-prep | --cannbench | --cannbench-remote | "
        "--check-config | --check-config-device | --check-config-cannbench | --check-config-cannbench-remote) [--timeout SECONDS]",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
