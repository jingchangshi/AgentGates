#!/usr/bin/env python3
"""Host-local Triton-distributed-ascend (MegaKernel) build, CPU, and NPU gates.

Scope
-----

This Adapter serves the ``megakernel-npu`` profile for
``Triton-distributed-ascend`` (``feature/megakernel`` branch):

* ``--build`` (``megakernel-npu-build``): incrementally stages ``--source``
  into ``MEGAKERNEL_NPU_WORK_ROOT/stage`` (keeping ``--source`` strictly
  read-only, ``sourceWrite: false``), applies the pinned ``triton-ascend``
  patches offline inside staging, compiles ``triton_dist`` with ``<= 16``
  parallel jobs against ``MEGAKERNEL_NPU_LLVM_SYSPATH``, installs the wheel into
  the ``megakernel-npu`` Conda environment, and verifies native products, opt
  tools, MegaKernel device bitcode, and ``ModelBuilder`` imports.
* ``--cpu`` (``megakernel-npu-cpu``): runs Host/CPU ``pytest`` suites without
  sourcing CANN (``TORCH_DEVICE_BACKEND_AUTOLOAD=0``) against live ``--source``
  ``triton_dist`` Python modules and emits JUnit XML counts.
* ``--device`` (``megakernel-npu-device``): opens an ``npu:<id>``
  ``deviceWindow`` advisory lock, probes card occupancy via ``npu-smi``,
  sources CANN and activates ``megakernel-npu``, verifies real NPU allocation,
  FP32/BF16 numerical ops, synchronization, and ``ModelBuilder`` readiness, and
  optionally executes configured MegaKernel device tests inside the window.

Output contract
---------------

Standard output carries exactly one JSON object and nothing else. Diagnostics
go to stderr. Exit code 75 maps to ``BLOCKED_ENV``.
"""
from __future__ import annotations

import contextlib
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
import xml.etree.ElementTree as ET

PROG = 'megakernel-npu-gate'
ADAPTER_DIR = Path(__file__).resolve().parent.parent

MAX_NINJA_JOBS = 16
DEFAULT_NINJA_JOBS = '16'
MAX_LINK_JOBS = 4
DEFAULT_LINK_JOBS = '2'
DEFAULT_TIMEOUT = 3300.0
DEFAULT_CPU_TIMEOUT = 600.0
DEFAULT_DEVICE_TIMEOUT = 1200.0
DEFAULT_WINDOW_WAIT_SECONDS = 60.0
SMI_TRANSIENT_RETRY_CODES = frozenset({215})
SMI_RETRY_BACKOFF_SECONDS = (0.15, 0.35)

# Exact AscendNPU-IR commit pinned by python/setup.py in Triton-distributed-ascend.
PINNED_ASCEND_NPU_IR_COMMIT = 'af366381db32f9c4c73521a2fe54647bf77e0614'

# Kept in strict parity with agent_gates.machine_env.MEGAKERNEL_NPU_KEYS.
MEGAKERNEL_NPU_KEYS = (
    'MEGAKERNEL_NPU_CANN_SETUP',
    'MEGAKERNEL_NPU_CONDA_SH',
    'MEGAKERNEL_NPU_CONDA_ENV',
    'MEGAKERNEL_NPU_CONDA_ENVS_PATH',
    'MEGAKERNEL_NPU_PYTHON',
    'MEGAKERNEL_NPU_LLVM_SYSPATH',
    'MEGAKERNEL_NPU_WORK_ROOT',
    'MEGAKERNEL_NPU_RUNTIME_HOME',
    'MEGAKERNEL_NPU_BISHENGIR_BIN',
    'MEGAKERNEL_NPU_CMAKE_BIN',
    'MEGAKERNEL_NPU_BUILD_SUBDIR',
    'MEGAKERNEL_NPU_NINJA_JOBS',
    'MEGAKERNEL_NPU_LINK_JOBS',
    'MEGAKERNEL_NPU_ENABLE_ASCENDC_OPS',
    'MEGAKERNEL_NPU_BUILD_SHMEM',
    'MEGAKERNEL_NPU_CPU_TESTS',
    'MEGAKERNEL_NPU_DEVICE_TESTS',
    'MEGAKERNEL_NPU_DEVICE_TEST_MODE',
)
TOOLCHAIN_KEYS = MEGAKERNEL_NPU_KEYS
ALLOWED = frozenset(MEGAKERNEL_NPU_KEYS)

REQUIRED = (
    'MEGAKERNEL_NPU_CANN_SETUP',
    'MEGAKERNEL_NPU_CONDA_SH',
    'MEGAKERNEL_NPU_CONDA_ENV',
)

NOT_MACHINE_KEYS = frozenset({
    'MEGAKERNEL_NPU_ENV_FILE',
    'MEGAKERNEL_NPU_DEVICE_ID',
    'MEGAKERNEL_NPU_REQUIRE_EXCLUSIVE_DEVICE',
    'MEGAKERNEL_NPU_SMI',
})

DEFAULT_CPU_TESTS = (
    'python/triton_dist/mega_kernel_ascend/test/host',
    'python/triton_dist/mega_kernel_ascend/test/ops/test_megakernel_dtype_contracts.py',
)

MEGAKERNEL_BITCODES = (
    'megakernel_op.aic.c310.bc',
    'megakernel_op.aiv.c310.bc',
    'megakernel_op.mix.aic.c310.bc',
    'megakernel_op.mix.aiv.c310.bc',
)

SAFE_BUILD_SUBDIR = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
_ASSIGN = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
_PROC_MEM_LINE = re.compile(
    r'Process id\s*:\s*(\d+)\s+Process name\s*:\s*(.*?)\s+Process memory\(MB\)\s*:\s*(\d+)'
)
GIT_SAFE = (
    '-c', 'core.fsmonitor=false',
    '-c', 'core.hooksPath=/dev/null',
    '-c', 'core.pager=cat',
    '-c', 'core.untrackedCache=false',
)


class BridgeError(ValueError):
    """A machine file or environment fault; mapped to BLOCKED_ENV (exit 75)."""


class DeviceWindowBusy(BridgeError):
    """Raised when the device window lease cannot be acquired in time."""


def _die(code: int, message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_env_file(text: str, path: Path) -> dict[str, str]:
    """Parse strict NAME=VALUE data without executing or expanding shell code."""
    values: dict[str, str] = {}
    for number, raw in enumerate(str(text).splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = _ASSIGN.match(line)
        if match is None:
            raise BridgeError(f'{path}: line {number} is not a NAME=value assignment: {raw!r}')
        key, value = match.group(1), match.group(2).strip()
        if key in NOT_MACHINE_KEYS:
            raise BridgeError(
                f'{path}: line {number} sets {key}, which is a launch value, '
                f'not machine configuration; pass it on the command line or environment instead'
            )
        if key not in ALLOWED:
            raise BridgeError(
                f'{path}: line {number} sets unknown key {key!r}; '
                f'allowed keys are {", ".join(sorted(ALLOWED))}'
            )
        if key in values:
            raise BridgeError(f'{path}: line {number} sets {key} a second time')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        elif value[:1] in ('"', "'") or value[-1:] in ('"', "'"):
            raise BridgeError(f'{path}: line {number} has an unbalanced quote in {raw!r}')
        values[key] = value
    return values


def default_env_file() -> Path | None:
    """Return sibling env.sh beside this adapter if present."""
    candidate = ADAPTER_DIR / 'env.sh'
    return candidate if candidate.is_file() else None


def resolved_env_file() -> Path | None:
    """Resolve explicit MEGAKERNEL_NPU_ENV_FILE or sibling env.sh."""
    explicit = (os.environ.get('MEGAKERNEL_NPU_ENV_FILE') or '').strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise BridgeError(f'MEGAKERNEL_NPU_ENV_FILE must be an absolute path: {explicit!r}')
        if path.is_symlink() or not path.is_file():
            raise BridgeError(f'MEGAKERNEL_NPU_ENV_FILE is not a regular non-symlink file: {explicit!r}')
        return path
    return default_env_file()


def machine_values() -> dict[str, str]:
    """Read machine configuration strictly from env.sh and reject shell conflicts."""
    path = resolved_env_file()
    if path is None:
        raise BridgeError(
            f'no machine file: set MEGAKERNEL_NPU_ENV_FILE or create {ADAPTER_DIR / "env.sh"}'
        )
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc
    values = _parse_env_file(text, path)

    for name in sorted(os.environ):
        if not name.startswith('MEGAKERNEL_NPU_') or name in NOT_MACHINE_KEYS:
            continue
        exported = (os.environ.get(name) or '').strip()
        if not exported:
            continue
        if name not in ALLOWED:
            raise BridgeError(
                f'{name} is exported in the environment but is not a machine-file key'
            )
        if name not in values or not values[name].strip():
            raise BridgeError(
                f'{name} is exported as {exported!r}, but machine values are read only from {path}'
            )
        if exported != values[name].strip():
            raise BridgeError(
                f'{name} is {exported!r} in the launching environment but {values[name]!r} in '
                f'{path}; machine values are read from that file only'
            )
    return values


def _required(values: dict[str, str], name: str) -> str:
    value = (values.get(name) or '').strip()
    if not value:
        raise BridgeError(f'{name} is required in the machine file')
    return value


def _abs_path(raw: str, name: str) -> Path:
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        raise BridgeError(f'{name} must be an absolute path: {raw!r}')
    if '\n' in raw or '\r' in raw or '\x00' in raw:
        raise BridgeError(f'{name} contains invalid control characters')
    return path


def _required_file(values: dict[str, str], name: str) -> Path:
    path = _abs_path(_required(values, name), name)
    if not path.is_file():
        raise BridgeError(f'{name} does not name an existing file: {path}')
    return path


def _required_dir(values: dict[str, str], name: str) -> Path:
    path = _abs_path(_required(values, name), name)
    if not path.is_dir():
        raise BridgeError(f'{name} does not name an existing directory: {path}')
    return path


def build_subdir(values: dict[str, str]) -> str:
    """Validate single-level build subdirectory name matching machine_env."""
    if 'MEGAKERNEL_NPU_BUILD_SUBDIR' in values:
        raw = values['MEGAKERNEL_NPU_BUILD_SUBDIR']
        if not raw.strip():
            raise BridgeError('MEGAKERNEL_NPU_BUILD_SUBDIR is set but empty')
        raw = raw.strip()
        if not SAFE_BUILD_SUBDIR.fullmatch(raw) or raw in {'.', '..'}:
            raise BridgeError(
                f'MEGAKERNEL_NPU_BUILD_SUBDIR must be a plain directory name: {raw!r}'
            )
        return raw
    return 'build'


def ninja_jobs(values: dict[str, str]) -> str:
    """Validate build job count, strictly enforcing the <= 16-core limit."""
    if 'MEGAKERNEL_NPU_NINJA_JOBS' in values:
        raw = values['MEGAKERNEL_NPU_NINJA_JOBS']
        if not raw.strip():
            raise BridgeError('MEGAKERNEL_NPU_NINJA_JOBS is set but empty')
        raw = raw.strip()
        if not raw.isdigit() or int(raw) < 1:
            raise BridgeError(f'MEGAKERNEL_NPU_NINJA_JOBS must be a positive integer: {raw!r}')
        jobs = int(raw)
        if jobs > MAX_NINJA_JOBS:
            raise BridgeError(
                f'MEGAKERNEL_NPU_NINJA_JOBS={jobs} exceeds the maximum allowed {MAX_NINJA_JOBS} cores'
            )
        return str(jobs)
    return DEFAULT_NINJA_JOBS


def link_jobs(values: dict[str, str]) -> str:
    """Validate TRITON_PARALLEL_LINK_JOBS (1..4)."""
    raw = (values.get('MEGAKERNEL_NPU_LINK_JOBS') or '').strip()
    if not raw:
        return DEFAULT_LINK_JOBS
    if not raw.isdigit() or not (1 <= int(raw) <= MAX_LINK_JOBS):
        raise BridgeError(
            f'MEGAKERNEL_NPU_LINK_JOBS must be an integer in 1..{MAX_LINK_JOBS}: {raw!r}'
        )
    return str(int(raw))


def on_off_flag(values: dict[str, str], key: str, default: str = 'OFF') -> str:
    """Validate an optional ON/OFF build flag."""
    raw = (values.get(key) or '').strip()
    if not raw:
        return default
    upper = raw.upper()
    if upper not in {'ON', 'OFF'}:
        raise BridgeError(f'{key} must be ON or OFF: {raw!r}')
    return upper


def device_test_mode(values: dict[str, str]) -> str:
    """Validate MEGAKERNEL_NPU_DEVICE_TEST_MODE."""
    raw = (values.get('MEGAKERNEL_NPU_DEVICE_TEST_MODE') or '').strip().lower()
    if not raw:
        return 'auto'
    if raw not in {'auto', 'script', 'pytest'}:
        raise BridgeError(
            f'MEGAKERNEL_NPU_DEVICE_TEST_MODE must be auto, script, or pytest: {raw!r}'
        )
    return raw


def _python_executable(values: dict[str, str]) -> Path:
    """Resolve the python3 binary inside the configured Conda environment."""
    explicit = (values.get('MEGAKERNEL_NPU_PYTHON') or '').strip()
    if explicit:
        path = _abs_path(explicit, 'MEGAKERNEL_NPU_PYTHON')
        if not path.is_file() or not os.access(path, os.X_OK):
            raise BridgeError(f'MEGAKERNEL_NPU_PYTHON is not an executable file: {path}')
        return path

    conda_sh = _required_file(values, 'MEGAKERNEL_NPU_CONDA_SH')
    conda_env = _required(values, 'MEGAKERNEL_NPU_CONDA_ENV')
    env_as_path = Path(conda_env)
    if env_as_path.is_absolute():
        candidate = env_as_path / 'bin' / 'python3'
    else:
        envs_root = (values.get('MEGAKERNEL_NPU_CONDA_ENVS_PATH') or '').strip()
        if envs_root:
            base = _abs_path(envs_root, 'MEGAKERNEL_NPU_CONDA_ENVS_PATH')
            candidate = base / conda_env / 'bin' / 'python3'
        else:
            conda_base = conda_sh.resolve().parents[2]
            candidate = conda_base / 'envs' / conda_env / 'bin' / 'python3'
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise BridgeError(
            f'cannot locate executable python3 for conda env {conda_env!r}: {candidate}'
        )
    return candidate


def _conda_envs_path(values: dict[str, str]) -> str | None:
    raw = (values.get('MEGAKERNEL_NPU_CONDA_ENVS_PATH') or '').strip()
    if not raw:
        return None
    return str(_abs_path(raw, 'MEGAKERNEL_NPU_CONDA_ENVS_PATH'))


def _validate_work_root(values: dict[str, str], source: Path) -> Path:
    raw = _required(values, 'MEGAKERNEL_NPU_WORK_ROOT')
    path = _abs_path(raw, 'MEGAKERNEL_NPU_WORK_ROOT')
    if path.is_symlink():
        raise BridgeError(f'MEGAKERNEL_NPU_WORK_ROOT must not be a symlink: {path}')
    src_resolved = source.resolve()
    work_resolved = path.resolve(strict=False)
    if work_resolved == src_resolved or work_resolved.is_relative_to(src_resolved):
        raise BridgeError(
            f'MEGAKERNEL_NPU_WORK_ROOT must be outside --source ({src_resolved}): {work_resolved}'
        )
    if src_resolved.is_relative_to(work_resolved):
        raise BridgeError(
            f'--source must not live inside MEGAKERNEL_NPU_WORK_ROOT: {work_resolved}'
        )
    return path


def _relative_test_paths(
    values: dict[str, str],
    key: str,
    source: Path,
    *,
    defaults: tuple[str, ...] = (),
    require_exists: bool = True,
) -> list[str]:
    raw = (values.get(key) or '').strip()
    if not raw:
        items = list(defaults)
    else:
        items = [part.strip() for part in re.split(r'[,\s]+', raw) if part.strip()]
        if not items:
            raise BridgeError(f'{key} names no test path')
    src_resolved = source.resolve()
    validated: list[str] = []
    for item in items:
        rel = Path(item)
        if rel.is_absolute() or '..' in rel.parts:
            raise BridgeError(f'{key} test path must be relative inside source root: {item!r}')
        target = (src_resolved / rel).resolve()
        if not target.is_relative_to(src_resolved):
            raise BridgeError(f'{key} test path escapes source root: {item!r}')
        if require_exists and not target.exists():
            raise BridgeError(f'{key} test path does not exist: {item}')
        validated.append(rel.as_posix())
    return validated


def _find_tool(name: str, extra_dirs: list[Path] | None = None) -> str | None:
    search_parts = [str(d) for d in (extra_dirs or []) if d]
    existing = os.environ.get('PATH', '')
    if existing:
        search_parts.append(existing)
    return shutil.which(name, path=os.pathsep.join(search_parts))


def _check_common(values: dict[str, str], source: Path) -> Path:
    for name in REQUIRED:
        _required(values, name)
    _required_file(values, 'MEGAKERNEL_NPU_CANN_SETUP')
    _required_file(values, 'MEGAKERNEL_NPU_CONDA_SH')
    _conda_envs_path(values)
    py = _python_executable(values)
    build_subdir(values)
    ninja_jobs(values)
    link_jobs(values)
    on_off_flag(values, 'MEGAKERNEL_NPU_ENABLE_ASCENDC_OPS', 'OFF')
    on_off_flag(values, 'MEGAKERNEL_NPU_BUILD_SHMEM', 'OFF')
    device_test_mode(values)
    if not (source / 'python' / 'triton_dist').is_dir():
        raise BridgeError(f'{source} does not look like Triton-distributed-ascend (missing python/triton_dist)')
    return py


def check_config_build(values: dict[str, str], source: Path) -> int:
    """Validate build toolchain, LLVM, work root, and concurrency cap without compiling."""
    py = _check_common(values, source)
    if not (source / 'CMakeLists.txt').is_file() or not (source / 'python' / 'setup.py').is_file():
        raise BridgeError(f'{source} is missing CMakeLists.txt or python/setup.py')
    if not (source / '3rdparty' / 'triton-ascend' / 'CMakeLists.txt').is_file():
        raise BridgeError(
            f'{source}/3rdparty/triton-ascend is not initialized; run git submodule update --init'
        )
    llvm_dir = _required_dir(values, 'MEGAKERNEL_NPU_LLVM_SYSPATH')
    for sub in ('include', 'lib'):
        if not (llvm_dir / sub).is_dir():
            raise BridgeError(f'MEGAKERNEL_NPU_LLVM_SYSPATH missing {sub}/ directory: {llvm_dir}')
    if not (llvm_dir / 'bin' / 'FileCheck').is_file():
        raise BridgeError(f'MEGAKERNEL_NPU_LLVM_SYSPATH missing bin/FileCheck: {llvm_dir}')
    _validate_work_root(values, source)

    cmake_cfg = (values.get('MEGAKERNEL_NPU_CMAKE_BIN') or '').strip()
    if cmake_cfg:
        cmake_path = _abs_path(cmake_cfg, 'MEGAKERNEL_NPU_CMAKE_BIN')
        if not cmake_path.is_file() or not os.access(cmake_path, os.X_OK):
            raise BridgeError(f'MEGAKERNEL_NPU_CMAKE_BIN is not executable: {cmake_path}')

    conda_bin = py.parent
    for tool in ('clang', 'clang++', 'ninja'):
        if _find_tool(tool, [conda_bin]) is None:
            raise BridgeError(f'required build tool not found on PATH: {tool}')
    if _find_tool('ld.lld', [conda_bin]) is None and _find_tool('lld', [conda_bin]) is None:
        raise BridgeError('required linker ld.lld or lld not found on PATH')
    print(f'{PROG}: build configuration ok (jobs<={ninja_jobs(values)})', file=sys.stderr)
    return 0


def check_config_cpu(values: dict[str, str], source: Path) -> int:
    """Validate Host/CPU test configuration without sourcing CANN or running pytest."""
    _check_common(values, source)
    tests = _relative_test_paths(
        values, 'MEGAKERNEL_NPU_CPU_TESTS', source, defaults=DEFAULT_CPU_TESTS, require_exists=True
    )
    if not tests:
        raise BridgeError('no CPU test paths resolved')
    print(f'{PROG}: cpu configuration ok ({len(tests)} target(s))', file=sys.stderr)
    return 0


def check_config_device(values: dict[str, str], source: Path) -> int:
    """Validate NPU device gate configuration without acquiring hardware."""
    _check_common(values, source)
    _relative_test_paths(
        values, 'MEGAKERNEL_NPU_DEVICE_TESTS', source, defaults=(), require_exists=True
    )
    runtime_home = (values.get('MEGAKERNEL_NPU_RUNTIME_HOME') or '').strip()
    if runtime_home:
        rt_dir = _abs_path(runtime_home, 'MEGAKERNEL_NPU_RUNTIME_HOME')
        if not rt_dir.is_dir():
            raise BridgeError(f'MEGAKERNEL_NPU_RUNTIME_HOME is not a directory: {rt_dir}')
        if not any(
            (rt_dir / cb).is_file()
            for cb in ('libprint_callback.so', 'libdefault_callback.so', 'libskip_inactive_callback.so')
        ):
            raise BridgeError(
                f'MEGAKERNEL_NPU_RUNTIME_HOME does not contain MegaKernel callback libraries: {rt_dir}'
            )
    bisheng_bin = (values.get('MEGAKERNEL_NPU_BISHENGIR_BIN') or '').strip()
    if bisheng_bin:
        _abs_path(bisheng_bin, 'MEGAKERNEL_NPU_BISHENGIR_BIN')
        if not Path(bisheng_bin).is_dir():
            raise BridgeError(f'MEGAKERNEL_NPU_BISHENGIR_BIN is not a directory: {bisheng_bin}')
    dev = device_id()
    if dev is not None and not dev.isdigit():
        raise BridgeError(f'MEGAKERNEL_NPU_DEVICE_ID must be a nonnegative integer: {dev!r}')
    print(f'{PROG}: device configuration ok', file=sys.stderr)
    return 0


def device_id() -> str | None:
    raw = (os.environ.get('MEGAKERNEL_NPU_DEVICE_ID') or '').strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise BridgeError(f'MEGAKERNEL_NPU_DEVICE_ID must be a nonnegative decimal ID: {raw!r}')
    return str(int(raw))


def exclusive_requested() -> bool:
    raw = (os.environ.get('MEGAKERNEL_NPU_REQUIRE_EXCLUSIVE_DEVICE') or '').strip().lower()
    return raw in {'1', 'true', 'yes', 'on'}


def _smi_cmd() -> str:
    return (
        (os.environ.get('MEGAKERNEL_NPU_SMI') or '').strip()
        or (os.environ.get('AGENT_GATES_SMI_COMMAND') or '').strip()
        or 'npu-smi'
    )


def _run_smi(args: list[str], timeout: float = 15.0) -> str | None:
    cmd = _smi_cmd()
    resolved = shutil.which(cmd) if '/' not in cmd else (cmd if Path(cmd).exists() else None)
    if resolved is None:
        return None
    for attempt in range(len(SMI_RETRY_BACKOFF_SECONDS) + 1):
        try:
            done = subprocess.run(
                [resolved, *args], capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode == 0:
            return done.stdout or ''
        if done.returncode in SMI_TRANSIENT_RETRY_CODES and attempt < len(SMI_RETRY_BACKOFF_SECONDS):
            time.sleep(SMI_RETRY_BACKOFF_SECONDS[attempt])
            continue
        return None
    return None


def _npu_id_for_physical(device: int) -> int | None:
    output = _run_smi(['info', '-m'])
    if not output:
        return None
    matches: list[int] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 4 and all(f.isdigit() for f in fields[:4]):
            if int(fields[3]) == device:
                matches.append(int(fields[0]))
        elif len(fields) >= 3 and all(f.isdigit() for f in fields[:3]):
            if int(fields[2]) == device:
                matches.append(int(fields[0]))
    unique = sorted(set(matches))
    return unique[0] if len(unique) == 1 else None


def occupancy(resource: str) -> str:
    """Measure target NPU occupancy without exposing other tenants' PIDs or cmdlines."""
    raw_dev = resource.partition(':')[2]
    if not raw_dev.isdigit():
        return 'unknown'
    dev = int(raw_dev)
    npu_id = _npu_id_for_physical(dev)
    if npu_id is None:
        return 'unknown'
    output = _run_smi(['info', '-t', 'proc-mem', '-i', str(npu_id)])
    if not output:
        return 'unknown'
    if not re.search(rf'^\s*NPU ID\s*:\s*{npu_id}\s*$', output, re.MULTILINE):
        return 'unknown'
    idle = bool(re.search(r'^\s*No process in device\.?\s*$', output, re.MULTILINE))
    announced = len(re.findall(r'Process id\s*:', output))
    parsed = sum(1 for line in output.splitlines() if _PROC_MEM_LINE.search(line))
    if parsed == 0 and idle and announced == 0:
        return 'free'
    if parsed > 0 and not idle and parsed == announced:
        return f'shared({parsed})'
    return 'unknown'


def _lease_path() -> Path:
    value = (os.environ.get('AGENT_GATES_DEVICE_LEASE_PATH') or '').strip()
    if not value:
        raise BridgeError(
            'AGENT_GATES_DEVICE_LEASE_PATH is not set: this Recipe declares a deviceWindow'
        )
    return _abs_path(value, 'AGENT_GATES_DEVICE_LEASE_PATH')


def _lease_resource() -> str:
    value = (os.environ.get('AGENT_GATES_DEVICE_LEASE_RESOURCE') or '').strip()
    if not value:
        raise BridgeError('AGENT_GATES_DEVICE_LEASE_RESOURCE is not set')
    return value


def _lease_wait_seconds() -> float:
    value = (os.environ.get('AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS') or '').strip()
    if not value:
        return DEFAULT_WINDOW_WAIT_SECONDS
    try:
        seconds = float(value)
    except ValueError as exc:
        raise BridgeError(f'AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS is invalid: {value!r}') from exc
    if seconds <= 0:
        raise BridgeError(f'AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS must be positive: {value!r}')
    return seconds


@contextlib.contextmanager
def device_window():
    """Acquire the engine-issued device lease around card work and emit window metadata."""
    path = _lease_path()
    resource = _lease_resource()
    wait = _lease_wait_seconds()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    started = time.monotonic()
    deadline = started + wait
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeviceWindowBusy(
                        f'device lease {resource} did not open within {wait}s; another run holds it'
                    ) from None
                time.sleep(min(0.05, remaining))
        opened = time.monotonic()
        occ = occupancy(resource)
        probe_seconds = round(time.monotonic() - opened, 3)
        if exclusive_requested() and (occ.startswith('shared(') or occ == 'unknown'):
            raise DeviceWindowBusy(
                f'device {resource} occupancy is {occ} and MEGAKERNEL_NPU_REQUIRE_EXCLUSIVE_DEVICE is set'
            )
        report = {
            'resource': resource,
            'path': str(path),
            'startedAt': _timestamp(),
            'waited': round(opened - started, 3),
            'occupancy': occ,
            'releasedAt': None,
            'probeSeconds': probe_seconds,
        }
        work_started = time.monotonic()
        try:
            yield report
        finally:
            report['heldSeconds'] = round(time.monotonic() - work_started, 3)
            report['releasedAt'] = _timestamp()
            os.close(fd)
            fd = -1
    finally:
        if fd >= 0:
            os.close(fd)


def _activation(values: dict[str, str], *, with_cann: bool) -> str:
    """Build shell prelude activating CANN (optional) then Conda."""
    lines = ['set +u']
    if with_cann:
        lines.append('source ' + shlex.quote(_required(values, 'MEGAKERNEL_NPU_CANN_SETUP')))
    envs_path = _conda_envs_path(values)
    if envs_path:
        lines.append('export CONDA_ENVS_PATH=' + shlex.quote(envs_path))
    lines.append('source ' + shlex.quote(_required(values, 'MEGAKERNEL_NPU_CONDA_SH')))
    lines.append('conda activate ' + shlex.quote(_required(values, 'MEGAKERNEL_NPU_CONDA_ENV')))
    return ' && '.join(lines)


def _run_shell(
    values: dict[str, str],
    body: str,
    *,
    cwd: Path,
    timeout: float,
    log,
    with_cann: bool,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, float]:
    started = time.monotonic()
    script = 'set -o pipefail' + os.linesep + _activation(values, with_cann=with_cann) + os.linesep + body
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    try:
        result = subprocess.run(
            ['/bin/bash', '-c', script],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        log.write((exc.output or b'').decode('utf-8', 'replace'))
        log.flush()
        raise BridgeError(f'command timed out after {timeout}s') from exc
    log.write(result.stdout.decode('utf-8', 'replace'))
    log.flush()
    return result.returncode, time.monotonic() - started


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git_clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env['GIT_OPTIONAL_LOCKS'] = '0'
    return env


def _git_head(repo: Path) -> str:
    try:
        done = subprocess.run(
            ['git', *GIT_SAFE, '-C', str(repo), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, env=_git_clean_env(), timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return 'NO_GIT'
    if done.returncode != 0:
        return 'NO_GIT'
    return done.stdout.strip() or 'NO_GIT'


def _list_repo_files(root: Path) -> list[str]:
    """List files in root via git ls-files (including untracked non-ignored), or os.walk."""
    try:
        done = subprocess.run(
            ['git', *GIT_SAFE, '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
            capture_output=True, env=_git_clean_env(), timeout=120,
        )
        if done.returncode == 0:
            return sorted({os.fsdecode(name) for name in done.stdout.split(b'\0') if name})
    except (OSError, subprocess.SubprocessError):
        pass
    names: list[str] = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d != '.git']
        for name in files:
            names.append((Path(directory) / name).relative_to(root).as_posix())
    return sorted(names)


def _copy_if_changed(src: Path, dst: Path) -> None:
    if src.is_symlink():
        target = os.readlink(src)
        if dst.is_symlink() and os.readlink(dst) == target:
            return
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(target)
        return
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and not dst.is_symlink():
        if dst.stat().st_size == src.stat().st_size and _sha256_file(dst) == _sha256_file(src):
            return
    elif dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _sync_tree_incremental(
    src_root: Path,
    dst_root: Path,
    files: list[str],
    *,
    preserve_prefixes: tuple[str, ...] = (),
) -> None:
    """Sync listed files from src_root to dst_root, preserving mtimes on unchanged files."""
    dst_root.mkdir(parents=True, exist_ok=True)
    wanted = set(files)
    # Remove files in dst_root that disappeared from src_root (outside preserved caches).
    for directory, dirs, existing_files in os.walk(dst_root, topdown=True, followlinks=False):
        rel_dir = Path(directory).relative_to(dst_root).as_posix()
        rel_prefix = '' if rel_dir == '.' else rel_dir + '/'
        if any(rel_dir == p or rel_prefix.startswith(p + '/') for p in preserve_prefixes):
            dirs[:] = []
            continue
        kept_dirs = []
        for d in dirs:
            if d == '.git':
                continue
            d_rel = f'{rel_prefix}{d}'
            if any(d_rel == p or d_rel.startswith(p + '/') for p in preserve_prefixes):
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs
        for f in existing_files:
            f_rel = f'{rel_prefix}{f}'
            if f_rel not in wanted:
                (Path(directory) / f).unlink(missing_ok=True)

    for rel in files:
        if any(rel == p or rel.startswith(p + '/') for p in preserve_prefixes):
            continue
        _copy_if_changed(src_root / rel, dst_root / rel)


def _triton_ascend_patch_digest(source: Path) -> str:
    """Digest inputs that determine patched stage/3rdparty/triton-ascend state."""
    ta_dir = source / '3rdparty' / 'triton-ascend'
    h = hashlib.sha256(b'stage-v2\0')
    h.update(_git_head(ta_dir).encode('utf-8') + b'\0')
    ta_patch = source / '3rdparty' / 'triton-ascend.patch'
    if ta_patch.is_file():
        h.update(b'triton-ascend.patch\0' + ta_patch.read_bytes() + b'\0')
    patch_dir = ta_dir / 'third_party' / 'ascend' / 'patch'
    if patch_dir.is_dir():
        for p in sorted(patch_dir.iterdir()):
            if p.is_file():
                h.update(p.name.encode('utf-8') + b'\0' + p.read_bytes() + b'\0')
    try:
        dirty = subprocess.run(
            ['git', *GIT_SAFE, '-C', str(ta_dir), 'status', '--porcelain', '-z'],
            capture_output=True, env=_git_clean_env(), timeout=60,
        )
        if dirty.returncode == 0:
            h.update(dirty.stdout + b'\0')
    except (OSError, subprocess.SubprocessError):
        pass
    return h.hexdigest()


def _prepare_triton_ascend_stage(source: Path, stage: Path, work_root: Path) -> None:
    """Prepare isolated patched stage/3rdparty/triton-ascend without mutating --source."""
    ta_src = source / '3rdparty' / 'triton-ascend'
    ta_dst = stage / '3rdparty' / 'triton-ascend'
    stamp_file = work_root / '.triton_ascend_patch_stamp'
    digest = _triton_ascend_patch_digest(source)
    if (
        stamp_file.is_file()
        and stamp_file.read_text(encoding='utf-8').strip() == digest
        and (ta_dst / '.git').is_dir()
    ):
        return

    if ta_dst.exists() or ta_dst.is_symlink():
        shutil.rmtree(ta_dst)
    ta_dst.mkdir(parents=True, exist_ok=True)

    # Copy triton-ascend excluding .git pointers and nested AscendNPU-IR.
    for directory, dirs, files in os.walk(ta_src, followlinks=False):
        rel_dir = Path(directory).relative_to(ta_src).as_posix()
        dirs[:] = [
            d for d in dirs
            if d not in {'.git', 'build'}
            and not (rel_dir == 'third_party/ascend' and d == 'AscendNPU-IR')
            and not (rel_dir == 'python' and d == 'build')
        ]
        for fname in files:
            if fname == '.git':
                continue
            rel = fname if rel_dir == '.' else f'{rel_dir}/{fname}'
            _copy_if_changed(ta_src / rel, ta_dst / rel)

    # Populate AscendNPU-IR at exact pinned commit offline via git archive.
    npu_ir_src = ta_src / 'third_party' / 'ascend' / 'AscendNPU-IR'
    npu_ir_dst = ta_dst / 'third_party' / 'ascend' / 'AscendNPU-IR'
    npu_ir_dst.mkdir(parents=True, exist_ok=True)
    archived = False
    if (npu_ir_src / '.git').exists():
        archive = subprocess.run(
            ['git', *GIT_SAFE, '-C', str(npu_ir_src), 'archive', PINNED_ASCEND_NPU_IR_COMMIT],
            capture_output=True, env=_git_clean_env(), timeout=120,
        )
        if archive.returncode == 0 and archive.stdout:
            extract = subprocess.run(
                ['tar', '-xf', '-', '-C', str(npu_ir_dst)],
                input=archive.stdout, capture_output=True, timeout=120,
            )
            if extract.returncode == 0:
                archived = True
    if not archived:
        for directory, dirs, files in os.walk(npu_ir_src, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {'.git', 'build'}]
            rel_dir = Path(directory).relative_to(npu_ir_src).as_posix()
            for fname in files:
                if fname == '.git':
                    continue
                rel = fname if rel_dir == '.' else f'{rel_dir}/{fname}'
                _copy_if_changed(npu_ir_src / rel, npu_ir_dst / rel)

    git_env = _git_clean_env()
    for cmd in (
        ['git', 'init', '-q'],
        ['git', 'add', '-A'],
        ['git', '-c', 'user.name=agent-gates', '-c', 'user.email=agent-gates@local',
         'commit', '-q', '-m', 'staged-base'],
    ):
        done = subprocess.run(cmd, cwd=str(ta_dst), env=git_env, capture_output=True, text=True)
        if done.returncode != 0:
            raise BridgeError(f'failed to initialize staged triton-ascend git tree: {done.stderr}')

    # Apply upstream triton-ascend patches + repo triton-ascend.patch offline so
    # setup.py sees TA_dirty != 0 and never runs git fetch over the network.
    patch_dir = ta_dst / 'third_party' / 'ascend' / 'patch'
    for patch_name in ('triton-ascend-dev-3.6.0.patch', 'triton-ascend-3.6.0.patch'):
        p = patch_dir / patch_name
        if p.is_file() and p.stat().st_size > 0:
            done = subprocess.run(
                ['git', 'apply', str(p)], cwd=str(ta_dst), env=git_env, capture_output=True, text=True
            )
            if done.returncode != 0:
                raise BridgeError(f'failed to apply {patch_name} in staging: {done.stderr}')
    ta_patch = source / '3rdparty' / 'triton-ascend.patch'
    if not ta_patch.is_file():
        raise BridgeError(f'missing required patch file: {ta_patch}')
    if ta_patch.stat().st_size > 0:
        done = subprocess.run(
            ['git', 'apply', '--whitespace=fix', str(ta_patch)],
            cwd=str(ta_dst), env=git_env, capture_output=True, text=True,
        )
        if done.returncode != 0:
            raise BridgeError(f'failed to apply triton-ascend.patch in staging: {done.stderr}')

    stamp_file.write_text(digest + os.linesep, encoding='utf-8')


def _prepare_stage(source: Path, work_root: Path) -> Path:
    """Incrementally sync --source into WORK_ROOT/stage without touching --source."""
    stage = work_root / 'stage'
    stage.mkdir(parents=True, exist_ok=True)
    top_files = [
        name for name in _list_repo_files(source)
        if name not in {'3rdparty/triton-ascend', '3rdparty/shmem', 'python/triton'}
        and not name.startswith(('3rdparty/triton-ascend/', '3rdparty/shmem/', 'python/triton/'))
    ]
    _sync_tree_incremental(
        source, stage, top_files,
        preserve_prefixes=(
            '3rdparty/triton-ascend', '3rdparty/shmem',
            'python/build', 'python/dist', 'python/triton', 'python/triton_dist.egg-info',
        ),
    )

    shmem_src = source / '3rdparty' / 'shmem'
    shmem_dst = stage / '3rdparty' / 'shmem'
    if shmem_src.is_dir():
        shmem_files = [
            name for name in _list_repo_files(shmem_src)
            if not name.startswith(('build/', 'install/', 'dist/', 'package/'))
        ]
        _sync_tree_incremental(
            shmem_src, shmem_dst, shmem_files,
            preserve_prefixes=('build', 'install', 'dist', 'package', '3rdparty'),
        )

    _prepare_triton_ascend_stage(source, stage, work_root)
    return stage


def _sync_missing_triton_dist_subpackages(stage: Path, python_exe: Path) -> None:
    """Ensure pure-Python mega_kernel_ascend subtrees omitted by setup.py are present in site-packages."""
    src_pkg = stage / 'python' / 'triton_dist'
    cmd = [
        str(python_exe),
        '-c',
        'import sysconfig; p = sysconfig.get_paths(); print(p.get("purelib") or p.get("platlib") or "")',
    ]
    done = subprocess.run(cmd, cwd='/tmp', capture_output=True, text=True, timeout=60)
    if done.returncode != 0 or not done.stdout.strip():
        raise BridgeError(f'failed to locate site-packages for {python_exe}: {done.stderr.strip()}')
    dst_pkg = Path(done.stdout.strip()) / 'triton_dist'
    if not dst_pkg.is_dir():
        raise BridgeError(f'installed triton_dist package directory missing: {dst_pkg}')
    if dst_pkg.resolve() == src_pkg.resolve():
        return
    exts = ('.py', '.bin', '.h', '.hpp', '.c', '.cpp', '.inc', '.conf', '.json', '.yaml', '.md')
    for root, dirs, files in os.walk(src_pkg, followlinks=False):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d != '__pycache__' and not (root_path / d).is_symlink()]
        rel_dir = root_path.relative_to(src_pkg)
        for fname in files:
            if not fname.endswith(exts):
                continue
            src_file = root_path / fname
            if src_file.is_symlink() or not src_file.is_file():
                continue
            target = dst_pkg / rel_dir / fname
            if target.is_symlink():
                continue
            if not target.exists() or src_file.read_bytes() != target.read_bytes():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, target)


def _write_nproc_shim(directory: Path, jobs: str) -> Path:
    shim_dir = directory / 'shims'
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / 'nproc'
    shim.write_text(f'#!/bin/sh\necho {jobs}\n', encoding='utf-8')
    shim.chmod(0o755)
    return shim_dir


@contextlib.contextmanager
def _triton_dist_overlay(source: Path):
    """Temporary PYTHONPATH entry exposing ONLY live triton_dist without shadowing triton."""
    with tempfile.TemporaryDirectory(prefix='mk-npu-overlay-') as td:
        overlay = Path(td)
        (overlay / 'triton_dist').symlink_to(source / 'python' / 'triton_dist')
        yield overlay


def _probe_build_products(values: dict[str, str], stage: Path, python_exe: Path) -> dict:
    probe_code = (
        'import json, pathlib, triton, triton_dist\n'
        'from triton_dist.mega_kernel_ascend.models_blade.model_builder import ModelBuilder\n'
        'triton_root = pathlib.Path(triton.__file__).resolve().parent\n'
        'c_dir = triton_root / "_C"\n'
        'lib = c_dir / "libtriton.so"\n'
        'opt_ok = all((c_dir / n).is_file() for n in ("triton-opt", "triton-mlir-opt"))\n'
        'bc_dir = triton_root / "backends" / "ascend" / "lib"\n'
        'bcs = ["megakernel_op.aic.c310.bc", "megakernel_op.aiv.c310.bc",\n'
        '       "megakernel_op.mix.aic.c310.bc", "megakernel_op.mix.aiv.c310.bc"]\n'
        'bc_paths = {n: str(bc_dir / n) for n in bcs if (bc_dir / n).is_file()}\n'
        'print(json.dumps({\n'
        '  "libtriton": str(lib) if lib.is_file() else "",\n'
        '  "libtritonSize": lib.stat().st_size if lib.is_file() else 0,\n'
        '  "optToolsPresent": bool(opt_ok),\n'
        '  "megakernelBitcodePresent": len(bc_paths) == len(bcs),\n'
        '  "tritonFile": str(triton.__file__),\n'
        '  "tritonDistFile": str(triton_dist.__file__),\n'
        '  "modelBuilderImported": ModelBuilder is not None,\n'
        '  "bcPaths": bc_paths,\n'
        '}))\n'
    )
    with tempfile.TemporaryDirectory(prefix='mk-npu-probe-') as tmp:
        log_path = Path(tmp) / 'probe.log'
        body = shlex.quote(str(python_exe)) + ' -c ' + shlex.quote(probe_code)
        with log_path.open('w', encoding='utf-8') as log:
            rc, _ = _run_shell(values, body, cwd=Path('/tmp'), timeout=180, log=log, with_cann=True)
        if rc != 0:
            return {'rc': rc, 'ok': False, 'tail': log_path.read_text(errors='replace')[-1000:]}
        lines = [line.strip() for line in log_path.read_text(errors='replace').splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    data['rc'] = 0
                    data['ok'] = True
                    return data
                except ValueError:
                    pass
    return {'rc': 1, 'ok': False}


def build(values: dict[str, str], source: Path, timeout: float) -> tuple[int, dict]:
    """Incrementally stage and build Triton-distributed-ascend with <= 16 cores."""
    py = _python_executable(values)
    work_root = _validate_work_root(values, source)
    work_root.mkdir(parents=True, exist_ok=True)
    jobs = ninja_jobs(values)
    ljobs = link_jobs(values)
    llvm_dir = _required_dir(values, 'MEGAKERNEL_NPU_LLVM_SYSPATH')
    ascendc_ops = on_off_flag(values, 'MEGAKERNEL_NPU_ENABLE_ASCENDC_OPS', 'OFF')
    build_shmem = on_off_flag(values, 'MEGAKERNEL_NPU_BUILD_SHMEM', 'OFF')
    cmake_bin = (values.get('MEGAKERNEL_NPU_CMAKE_BIN') or '').strip()

    stage = _prepare_stage(source, work_root)
    wheel_dir = work_root / 'wheels'
    wheel_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = work_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    build_tmp = work_root / 'tmp'
    build_tmp.mkdir(parents=True, exist_ok=True)
    triton_cache = work_root / 'triton_cache'
    triton_cache.mkdir(parents=True, exist_ok=True)
    shim_dir = _write_nproc_shim(work_root, jobs)

    extra_path = [str(shim_dir)]
    if cmake_bin:
        extra_path.insert(0, str(Path(cmake_bin).parent))

    build_log = logs_dir / 'build.log'
    rc = 0
    started = time.monotonic()
    with build_log.open('w', encoding='utf-8') as log:
        if build_shmem == 'ON':
            shmem_dir = stage / '3rdparty' / 'shmem'
            shmem_cmd = os.linesep.join([
                'export PATH=' + shlex.quote(os.pathsep.join(extra_path)) + ':"$PATH"',
                'bash scripts/build.sh -python_extension -soc_type Ascend950',
                shlex.quote(str(py)) + ' -m pip install --no-deps --force-reinstall --no-index dist/shmem-*.whl',
            ])
            rc, _ = _run_shell(values, shmem_cmd, cwd=shmem_dir, timeout=timeout, log=log, with_cann=True)

        if rc == 0:
            for old_whl in wheel_dir.glob('triton_dist-*.whl'):
                old_whl.unlink(missing_ok=True)
            statements = [
                'export PATH=' + shlex.quote(os.pathsep.join(extra_path)) + ':"$PATH"',
                'export LLVM_SYSPATH=' + shlex.quote(str(llvm_dir)),
                'export TRITON_USE_ASCEND=ON',
                'export TRITON_BUILD_DISTRIBUTED=ON',
                'export TRITON_BUILD_WITH_CLANG_LLD=ON',
                'export TRITON_BUILD_PROTON=OFF',
                'export TRITON_BUILD_LITTLE_KERNEL=OFF',
                'export TRITON_BUILD_WITH_CCACHE=ON',
                f'export TRITON_ENABLE_ASCENDC_OPS={ascendc_ops}',
                'export TRITON_OFFLINE_BUILD=1',
                'export TRITON_PTXAS_PATH=',
                'export TRITON_PTXAS_BLACKWELL_PATH=',
                'export TRITON_CUOBJDUMP_PATH=',
                'export TRITON_NVDISASM_PATH=',
                'export TRITON_CUDACRT_PATH=',
                'export TRITON_CUDART_PATH=',
                'export TRITON_CUPTI_INCLUDE_PATH=',
                'export TRITON_CUPTI_LIB_PATH=',
                f'export MAX_JOBS={jobs}',
                f'export CMAKE_BUILD_PARALLEL_LEVEL={jobs}',
                f'export TRITON_PARALLEL_LINK_JOBS={ljobs}',
                'export TMPDIR=' + shlex.quote(str(build_tmp)),
                'export TEMP=' + shlex.quote(str(build_tmp)),
                'export TMP=' + shlex.quote(str(build_tmp)),
                'export TRITON_CACHE_DIR=' + shlex.quote(str(triton_cache)),
                (
                    f'{shlex.quote(str(py))} -m pip wheel ./python --verbose '
                    f'--no-build-isolation --no-deps --wheel-dir {shlex.quote(str(wheel_dir))}'
                ),
                (
                    f'{shlex.quote(str(py))} -m pip install --no-deps --force-reinstall '
                    f'--no-index {shlex.quote(str(wheel_dir))}/triton_dist-*.whl'
                ),
            ]
            rc, _ = _run_shell(
                values, os.linesep.join(statements), cwd=stage, timeout=timeout, log=log, with_cann=True
            )
            if rc == 0:
                _sync_missing_triton_dist_subpackages(stage, py)

    build_seconds = round(time.monotonic() - started, 3)
    if rc != 0:
        print(f'{PROG}: build failed rc={rc}; see {build_log}', file=sys.stderr)
        sys.stderr.write(build_log.read_text(errors='replace')[-4000:])

    probe = _probe_build_products(values, stage, py) if rc == 0 else {'ok': False}
    if rc == 0 and not probe.get('ok'):
        print(f'{PROG}: post-install probe failed', file=sys.stderr)
        if probe.get('tail'):
            sys.stderr.write(probe['tail'] + '\n')
    lib_path = Path(probe.get('libtriton', '')) if probe.get('libtriton') else None
    native_ok = bool(lib_path and lib_path.is_file() and probe.get('libtritonSize', 0) > (1 << 20))
    opt_ok = bool(probe.get('optToolsPresent'))
    bc_ok = bool(probe.get('megakernelBitcodePresent'))
    imported_ok = bool(probe.get('ok'))
    builder_ok = bool(probe.get('modelBuilderImported'))
    succeeded = rc == 0 and native_ok and opt_ok and bc_ok and imported_ok and builder_ok

    if succeeded:
        stamp = {
            'adapter': 'megakernel-npu',
            'gitHead': _git_head(source),
            'python': str(py),
            'jobs': int(jobs),
            'libtritonSha256': _sha256_file(lib_path),
            'bitcodeSha256': {k: _sha256_file(Path(v)) for k, v in sorted((probe.get('bcPaths') or {}).items())},
            'builtAt': _timestamp(),
        }
        (work_root / 'build-stamp.json').write_text(
            json.dumps(stamp, indent=2, sort_keys=True) + os.linesep, encoding='utf-8'
        )

    observations = {
        'exitCodeZero': bool(succeeded),
        'nativeLibraryPresent': native_ok,
        'optToolsPresent': opt_ok,
        'megakernelBitcodePresent': bc_ok,
        'tritonDistImported': imported_ok,
        'modelBuilderImported': builder_ok,
        'jobs': int(jobs),
        'workRoot': str(work_root),
        'buildLog': str(build_log),
        'buildSeconds': build_seconds,
    }
    return (0 if succeeded else (rc or 1)), observations


def _junit_counts(path: Path) -> dict[str, int] | None:
    try:
        root = ET.parse(str(path)).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == 'testsuite' else list(root.iter('testsuite'))
    if not suites:
        return None
    collected = passed = failed = skipped = 0
    for suite in suites:
        for case in suite.iter('testcase'):
            collected += 1
            if case.find('failure') is not None or case.find('error') is not None:
                failed += 1
            elif case.find('skipped') is not None:
                skipped += 1
            else:
                passed += 1
    return {'collected': collected, 'passed': passed, 'failed': failed, 'skipped': skipped}


def _cpu_observations(succeeded: bool, collected: int, passed: int, failed: int, skipped: int, tests: list[str]) -> dict:
    return {
        'exitCodeZero': bool(succeeded),
        'testsCollected': collected,
        'testsPassed': passed,
        'testsFailed': failed,
        'testsSkipped': skipped,
        'notAllSkipped': collected > 0 and skipped < collected,
        'testsCollectedPositive': collected > 0,
        'testPaths': tests,
    }


def cpu_tests(values: dict[str, str], source: Path, timeout: float) -> tuple[int, dict]:
    """Run Host/CPU pytest suite with CANN libs on LD_LIBRARY_PATH and NPU devices hidden."""
    py = _python_executable(values)
    tests = _relative_test_paths(
        values, 'MEGAKERNEL_NPU_CPU_TESTS', source, defaults=DEFAULT_CPU_TESTS, require_exists=True
    )
    with tempfile.TemporaryDirectory(prefix='mk-npu-cpu-') as tmp, _triton_dist_overlay(source) as overlay:
        tmp_path = Path(tmp)
        report = tmp_path / 'junit.xml'
        log_path = tmp_path / 'pytest.log'
        statements = [
            'export TORCH_DEVICE_BACKEND_AUTOLOAD=0',
            'export ASCEND_RT_VISIBLE_DEVICES=""',
            'export TMPDIR=' + shlex.quote(str(tmp_path)),
            'export TEMP=' + shlex.quote(str(tmp_path)),
            'export TMP=' + shlex.quote(str(tmp_path)),
            'export PYTHONPATH=' + shlex.quote(str(overlay)) + ':"${PYTHONPATH:-}"',
            ' '.join(
                [shlex.quote(str(py)), '-m', 'pytest']
                + [shlex.quote(t) for t in tests]
                + ['-p', 'no:cacheprovider', '--junit-xml=' + shlex.quote(str(report))]
            ),
        ]
        with log_path.open('w', encoding='utf-8') as log:
            try:
                rc, _ = _run_shell(
                    values, os.linesep.join(statements), cwd=source, timeout=timeout, log=log, with_cann=True
                )
            except BridgeError:
                print(f'{PROG}: cpu pytest timed out', file=sys.stderr)
                return 1, _cpu_observations(False, 0, 0, 0, 0, tests)

        counts = _junit_counts(report)
        if counts is None:
            print(f'{PROG}: no readable JUnit report at {report}', file=sys.stderr)
            sys.stderr.write(log_path.read_text(errors='replace')[-4000:])
            return 1, _cpu_observations(False, 0, 0, 0, 0, tests)

        collected, passed, failed, skipped = (
            counts['collected'], counts['passed'], counts['failed'], counts['skipped']
        )
        ok = rc == 0 and collected > 0 and skipped < collected and failed == 0
        if not ok:
            sys.stderr.write(log_path.read_text(errors='replace')[-4000:])
        return (0 if ok else (rc or 1)), _cpu_observations(ok, collected, passed, failed, skipped, tests)


DEVICE_PROBE = r"""
import json, pathlib, sys
facts = {
    "tritonDistImported": False,
    "modelBuilderImported": False,
    "npuAvailable": False,
    "deviceCountOne": False,
    "tensorOnDevice": False,
    "numericalOk": False,
    "synchronizeOk": False,
    "deviceMemoryPositive": False,
    "nativeLibraryPresent": False,
    "megakernelBitcodePresent": False,
}
try:
    import torch, torch_npu  # noqa: F401
    import triton
    import triton_dist
    facts["tritonDistImported"] = True
    from triton_dist.mega_kernel_ascend.models_blade.model_builder import ModelBuilder
    facts["modelBuilderImported"] = ModelBuilder is not None
    triton_root = pathlib.Path(triton.__file__).resolve().parent
    lib = triton_root / "_C" / "libtriton.so"
    facts["nativeLibraryPresent"] = lib.is_file() and lib.stat().st_size > (1 << 20)
    bc_dir = triton_root / "backends" / "ascend" / "lib"
    bcs = ("megakernel_op.aic.c310.bc", "megakernel_op.aiv.c310.bc",
           "megakernel_op.mix.aic.c310.bc", "megakernel_op.mix.aiv.c310.bc")
    facts["megakernelBitcodePresent"] = all((bc_dir / n).is_file() for n in bcs)

    avail = bool(torch.npu.is_available())
    facts["npuAvailable"] = avail
    count = int(torch.npu.device_count()) if avail else 0
    facts["deviceCountOne"] = (count == 1)
    if avail and count >= 1:
        torch.npu.set_device(0)
        torch.npu.reset_peak_memory_stats(0)
        probe_t = torch.rand((4,)).npu()
        facts["tensorOnDevice"] = (probe_t.device.type == "npu")
        a = torch.linspace(1.0, 16.0, 16, dtype=torch.float32).npu()
        b = torch.linspace(0.5, 8.0, 16, dtype=torch.float32).npu()
        vec_ok = torch.allclose((a + b).cpu(), a.cpu() + b.cpu(), atol=1e-5, rtol=1e-5)
        m1 = torch.randn((32, 64), dtype=torch.bfloat16).npu()
        m2 = torch.randn((64, 32), dtype=torch.bfloat16).npu()
        mm = torch.matmul(m1, m2)
        torch.npu.synchronize()
        facts["synchronizeOk"] = True
        mm_ref = torch.matmul(m1.cpu().float(), m2.cpu().float())
        mm_ok = torch.allclose(mm.cpu().float(), mm_ref, atol=1e-1, rtol=1e-1)
        facts["numericalOk"] = bool(vec_ok and mm_ok)
        facts["deviceMemoryPositive"] = int(torch.npu.max_memory_allocated(0)) > 0
except Exception as exc:
    print(f"device probe error: {exc}", file=sys.stderr)
print(json.dumps(facts))
"""


def _check_built_artifacts(values: dict[str, str], py: Path) -> None:
    """Fail fast with BLOCKED_ENV (exit 75) if triton_dist build products are absent."""
    check_code = (
        'import json, pathlib, triton, triton_dist\n'
        'root = pathlib.Path(triton.__file__).resolve().parent\n'
        'lib = root / "_C" / "libtriton.so"\n'
        'bc = root / "backends" / "ascend" / "lib" / "megakernel_op.mix.aiv.c310.bc"\n'
        'print(json.dumps({"lib": lib.is_file(), "bc": bc.is_file()}))\n'
    )
    with tempfile.TemporaryDirectory(prefix='mk-npu-precheck-') as tmp:
        log_path = Path(tmp) / 'precheck.log'
        with log_path.open('w', encoding='utf-8') as log:
            rc, _ = _run_shell(
                values,
                f'{shlex.quote(str(py))} -c {shlex.quote(check_code)}',
                cwd=Path('/tmp'),
                timeout=120,
                log=log,
                with_cann=True,
            )
        if rc != 0:
            raise BridgeError(
                'triton/triton_dist is not installed in megakernel-npu env; run megakernel-npu-build first'
            )
        for line in reversed(log_path.read_text(errors='replace').splitlines()):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                data = json.loads(line)
                if not data.get('lib') or not data.get('bc'):
                    raise BridgeError(
                        'built libtriton.so or megakernel_op.*.c310.bc missing; run megakernel-npu-build first'
                    )
                return
    raise BridgeError('unable to verify built triton_dist artifacts')


def device_run(values: dict[str, str], source: Path, timeout: float) -> tuple[int, dict]:
    """Run windowed NPU verification and optional MegaKernel device tests."""
    dev = device_id()
    if dev is None:
        raise BridgeError('MEGAKERNEL_NPU_DEVICE_ID or --device-id is required for device gate')
    py = _python_executable(values)
    tests = _relative_test_paths(
        values, 'MEGAKERNEL_NPU_DEVICE_TESTS', source, defaults=(), require_exists=True
    )
    mode = device_test_mode(values)
    runtime_home = (values.get('MEGAKERNEL_NPU_RUNTIME_HOME') or '').strip()
    bisheng_bin = (values.get('MEGAKERNEL_NPU_BISHENGIR_BIN') or '').strip()

    _check_built_artifacts(values, py)

    probe_facts: dict = {}
    device_tests_passed = 0
    device_tests_failed = 0
    tests_ok = True

    with device_window() as window, tempfile.TemporaryDirectory(prefix='mk-npu-dev-') as tmp, _triton_dist_overlay(source) as overlay:
        tmp_path = Path(tmp)
        ascend_work = tmp_path / 'ascend_work'
        triton_cache = tmp_path / 'triton_cache'
        ascend_work.mkdir(parents=True, exist_ok=True)
        triton_cache.mkdir(parents=True, exist_ok=True)

        env_lines = [
            f'export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(dev)}',
            'export DEVICE_ID=0',
            'export NPU_ID=0',
            'export TMPDIR=' + shlex.quote(str(tmp_path)),
            'export TEMP=' + shlex.quote(str(tmp_path)),
            'export TMP=' + shlex.quote(str(tmp_path)),
            'export ASCEND_WORK_PATH=' + shlex.quote(str(ascend_work)),
            'export TRITON_CACHE_DIR=' + shlex.quote(str(triton_cache)),
            'export PYTHONPATH=' + shlex.quote(str(overlay)) + ':"${PYTHONPATH:-}"',
        ]
        if bisheng_bin:
            env_lines.append('export PATH=' + shlex.quote(bisheng_bin) + ':"$PATH"')
        if runtime_home:
            env_lines.append('export MEGA_KERNEL_HOME=' + shlex.quote(runtime_home))
            env_lines.append(
                'export LD_LIBRARY_PATH='
                + shlex.quote(runtime_home)
                + ':'
                + shlex.quote(str(Path(runtime_home) / 'lib'))
                + ':"${LD_LIBRARY_PATH:-}"'
            )

        probe_log = tmp_path / 'probe.log'
        probe_body = os.linesep.join(env_lines + [f'{shlex.quote(str(py))} -c {shlex.quote(DEVICE_PROBE)}'])
        with probe_log.open('w', encoding='utf-8') as log:
            probe_rc, _ = _run_shell(
                values, probe_body, cwd=source, timeout=min(timeout, 300.0), log=log, with_cann=True
            )
        for line in reversed(probe_log.read_text(errors='replace').splitlines()):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    probe_facts = json.loads(line)
                    break
                except ValueError:
                    pass
        if probe_rc != 0 or not probe_facts:
            sys.stderr.write(probe_log.read_text(errors='replace')[-4000:])

        for idx, rel_test in enumerate(tests):
            test_path = source / rel_test
            run_pytest = mode == 'pytest' or (mode == 'auto' and test_path.is_dir())
            test_log = tmp_path / f'device_test_{idx}.log'
            if run_pytest:
                junit = tmp_path / f'device_junit_{idx}.xml'
                cmd = (
                    f'{shlex.quote(str(py))} -m pytest {shlex.quote(rel_test)} '
                    f'-p no:cacheprovider --junit-xml={shlex.quote(str(junit))}'
                )
            else:
                cmd = f'{shlex.quote(str(py))} {shlex.quote(rel_test)}'
            body = os.linesep.join(env_lines + [cmd])
            with test_log.open('w', encoding='utf-8') as log:
                t_rc, _ = _run_shell(values, body, cwd=source, timeout=timeout, log=log, with_cann=True)
            if run_pytest:
                counts = _junit_counts(junit)
                t_ok = bool(t_rc == 0 and counts and counts['collected'] > 0 and counts['failed'] == 0 and counts['skipped'] < counts['collected'])
            else:
                t_ok = t_rc == 0
            if t_ok:
                device_tests_passed += 1
            else:
                device_tests_failed += 1
                tests_ok = False
                sys.stderr.write(test_log.read_text(errors='replace')[-4000:])

    all_probe_ok = (
        probe_rc == 0
        and all(
            probe_facts.get(k) is True
            for k in (
                'tritonDistImported',
                'modelBuilderImported',
                'npuAvailable',
                'deviceCountOne',
                'tensorOnDevice',
                'numericalOk',
                'synchronizeOk',
                'deviceMemoryPositive',
            )
        )
    )
    succeeded = all_probe_ok and tests_ok
    observations = {
        'exitCodeZero': bool(succeeded),
        'tritonDistImported': bool(probe_facts.get('tritonDistImported')),
        'modelBuilderImported': bool(probe_facts.get('modelBuilderImported')),
        'npuAvailable': bool(probe_facts.get('npuAvailable')),
        'deviceCountOne': bool(probe_facts.get('deviceCountOne')),
        'tensorOnDevice': bool(probe_facts.get('tensorOnDevice')),
        'numericalOk': bool(probe_facts.get('numericalOk')),
        'synchronizeOk': bool(probe_facts.get('synchronizeOk')),
        'deviceMemoryPositive': bool(probe_facts.get('deviceMemoryPositive')),
        'nativeLibraryPresent': bool(probe_facts.get('nativeLibraryPresent')),
        'megakernelBitcodePresent': bool(probe_facts.get('megakernelBitcodePresent')),
        'windowOpened': window.get('startedAt') is not None,
        'windowReleased': window.get('releasedAt') is not None,
        'occupancyRecorded': window.get('occupancy') in {'free', 'unknown'} or str(window.get('occupancy', '')).startswith('shared('),
        'deviceWindow': window,
        'deviceTestsRun': len(tests),
        'deviceTestsPassed': device_tests_passed,
        'deviceTestsFailed': device_tests_failed,
    }
    return (0 if succeeded else 1), observations


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    timeout = None
    if '--timeout' in args:
        idx = args.index('--timeout')
        try:
            timeout = float(args[idx + 1])
        except (IndexError, ValueError):
            _die(2, f'{PROG}: --timeout needs a number of seconds')
        del args[idx:idx + 2]

    try:
        values = machine_values()
        source = Path.cwd()
        if '--check-config-build' in args:
            return check_config_build(values, source)
        if '--check-config-cpu' in args:
            return check_config_cpu(values, source)
        if '--check-config-device' in args:
            return check_config_device(values, source)
        if '--build' in args:
            rc, observations = build(values, source, timeout or DEFAULT_TIMEOUT)
            print(json.dumps(observations))
            return rc
        if '--cpu' in args:
            rc, observations = cpu_tests(values, source, timeout or DEFAULT_CPU_TIMEOUT)
            print(json.dumps(observations))
            return rc
        if '--device' in args:
            rc, observations = device_run(values, source, timeout or DEFAULT_DEVICE_TIMEOUT)
            print(json.dumps(observations))
            return rc
    except BridgeError as exc:
        _die(75, f'{PROG}: {exc}')

    _die(
        2,
        f'usage: {PROG} (--build | --cpu | --device | --check-config-build | '
        f'--check-config-cpu | --check-config-device) [--timeout SECONDS]',
    )
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
