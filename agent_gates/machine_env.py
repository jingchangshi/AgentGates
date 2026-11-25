"""One machine file per adapter: the only source of that machine's parameters.

The three Adapters all answer the same question - "which container, which build
tree, how many jobs, which toolchain" - from one operator-edited `env.sh`, so a
Gate, the engine's resource locks and a human's shell cannot disagree about the
same machine. Before this module the Ascend Adapter read its container identity
and its build directory from the launching shell while both TileLang Adapters
read the equivalent values from the file; the same run could therefore hold a
lock on `build` while building into `build-release`, and an inherited export
could point a Gate at a container nobody had created.

Two layers share the grammar but not the code path, on purpose:

* the Adapter scripts parse the file themselves, because a Pack must run from
  an arbitrary install prefix and must never import the core package;
* this module parses it for the engine, which needs the *same* build
  subdirectory to take the *same* resource lock.

Both sides are pinned to the same key sets by tests, so a key added to one side
does not silently become machine configuration the other side ignores.

The file is data. Only `NAME=value`, comments and blank lines are accepted; no
expansion, no substitution, no line continuations, and a malformed or unknown
key fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

# A conservative `NAME=value` grammar, identical to the Adapters'.
_ASSIGNMENT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')

# Keys each Adapter's machine file may define. Keep in step with
# `adapters/*/scripts/*.py`; `tests/test_machine_env.py` compares the two.
ASCENDNPU_KEYS = (
    'ASCENDNPU_CANN_SETUP',
    'ASCENDNPU_CONDA_SH',
    'ASCENDNPU_CONDA_ENV',
    'ASCENDNPU_CCEC_BIN',
    'ASCENDNPU_CONTAINER_NAME',
    'ASCENDNPU_CONTAINER_USER',
    'ASCENDNPU_CONTAINER_HOME',
    'ASCENDNPU_BUILD_SUBDIR',
    'ASCENDNPU_NINJA_JOBS',
    # The container path of the source root, when it differs from the derived
    # bind-mount mapping. It names a location inside the container, so it is a
    # machine fact like the identity above, not an invocation option.
    'ASCENDNPU_CONTAINER_SOURCE',
)
TILELANG_HIVM_KEYS = (
    'TILELANG_HIVM_CANN_SETUP',
    'TILELANG_HIVM_CONDA_SH',
    'TILELANG_HIVM_CONDA_ENV',
    'TILELANG_HIVM_HIVM_COMPILER',
    'TILELANG_HIVM_HIVM_TARGET',
    'TILELANG_HIVM_CONDA_ENVS_PATH',
    'TILELANG_HIVM_WHEEL_DIR',
    'TILELANG_HIVM_PIP_INDEX',
    'TILELANG_HIVM_PYTHON_VERSION',
    'TILELANG_HIVM_SOURCE_ROOT',
    'TILELANG_HIVM_CONTAINER_NAME',
    'TILELANG_HIVM_CONTAINER_USER',
    'TILELANG_HIVM_CONTAINER_HOME',
    'TILELANG_HIVM_BUILD_SUBDIR',
    'TILELANG_HIVM_NINJA_JOBS',
    # The container path of the source root, when it differs from the derived
    # bind-mount mapping. A machine fact, like the identity above.
    'TILELANG_HIVM_CONTAINER_SOURCE',
)

# The TLA Adapter has its own key set: it configures a different toolchain
# (a vendor Triton wheel rather than a TileLang build tree), so it must not share
# the TILELANG_HIVM_ prefix. It has no build tree of its own, but the engine
# resolves a build directory name for every Adapter so it can take a lease, which
# is why the BUILD_SUBDIR key is accepted even though the Gate never writes one.
TLA_NPU_KEYS = (
    'TLA_NPU_CANN_SETUP',
    'TLA_NPU_CONDA_SH',
    'TLA_NPU_CONDA_ENV',
    'TLA_NPU_CONDA_ENVS_PATH',
    'TLA_NPU_CONTAINER_NAME',
    'TLA_NPU_CONTAINER_USER',
    'TLA_NPU_CONTAINER_HOME',
    'TLA_NPU_CONTAINER_SOURCE',
    'TLA_NPU_PYTHON_VERSION',
    'TLA_NPU_PIP_INDEX',
    'TLA_NPU_WHEEL_DIR',
    'TLA_NPU_TRITON_WHEEL',
    'TLA_NPU_TEST_DIR',
    'TLA_NPU_CACHE_DIR',
    'TLA_NPU_BUILD_SUBDIR',
)
# The FK-Ascend Adapter configures a ForgeKernel checkout plus a REUSED
# TileLang build, not a build tree of its own: the engine resolves a BUILD_SUBDIR
# for every Adapter so it can take a lease, but this one installs into a private
# virtual environment and never runs Ninja in the source.
#
# The keys are deliberately split by who reads them. `env.sh` holds the machine
# (paths, interpreter, toolchain); the device id is NOT here, because it
# authorizes one run rather than describing a machine.
FK_ASCEND_KEYS = (
    'FK_ASCEND_CANN_SETUP',
    'FK_ASCEND_PYTHON',
    'FK_ASCEND_TILELANG_SOURCE',
    'FK_ASCEND_TILELANG_BUILD',
    'FK_ASCEND_WORK_ROOT',
    'FK_ASCEND_BUILD_SUBDIR',
    'FK_ASCEND_VENV',
    'FK_ASCEND_PIP_INDEX',
    'FK_ASCEND_CLAUDE_BIN',
    'FK_ASCEND_PROVIDER_ENDPOINT',
    'FK_ASCEND_PROVIDER_MODEL',
    'FK_ASCEND_PROVIDER_KEY_VARIABLE',
    'FK_ASCEND_AGENT_SANDBOX',
    'FK_ASCEND_AGENT_NETWORK',
    'FK_ASCEND_DEPENDENCY_POLICY',
    'FK_ASCEND_TILELANG_CACHE_DIR',
)
# The BSKA-NPU Adapter proves the BiShengKernelAgent (helix) runtime and the
# vendored official CANNBench evaluator on the HOST: a complete CANN setup, one
# conda environment, the CANNBench checkout, and the node/claude executables.
# There is no container key set: the Adapter runs as the invoking user so the
# interpreter that imports Torch is the one the native extension was compiled
# against. The device id is a launch value (BSKA_NPU_DEVICE_ID), not a key.
BSKA_NPU_KEYS = (
    'BSKA_NPU_CANN_SETUP',
    'BSKA_NPU_CONDA_ENV',
    'BSKA_NPU_CONDA_ENVS_PATH',
    'BSKA_NPU_CANNBENCH_REPO',
    'BSKA_NPU_CANNBENCH_TASK',
    'BSKA_NPU_CANNBENCH_ARCH',
    'BSKA_NPU_BUILD_SUBDIR',
    'BSKA_NPU_NODE_BIN',
    'BSKA_NPU_CLAUDE_BIN',
    # Pin one exact triton_ascend build. The version string is what the Gate
    # enforces at run time.
    'BSKA_NPU_TRITON_ASCEND_VERSION',
)
# The TileLang-NPU Adapter builds and tests a plain TileLang checkout ON THE HOST:
# there is no container and no HIVM backend involved, so it owns its own prefix
# rather than sharing the TILELANG_HIVM_ file. It is the only TileLang Adapter
# that writes a host build tree through its own Gate, which is why it declares a
# job count the build actually consumes.
#
# TILELANG_NPU_CMAKE_BIN exists because this host's conda environment ships a
# CMake whose major version differs from the one that generated an existing build
# tree; naming the binary keeps the operator in control of which generator
# regenerates that tree. TILELANG_NPU_EXAMPLE_TESTS names the fixed example tests
# the regression runs, so the selection is reviewed configuration rather than a
# glob that silently grows.
# The device id (TILELANG_NPU_DEVICE_ID) is a launch value, not a machine key.
TILELANG_NPU_KEYS = (
    'TILELANG_NPU_CANN_SETUP',
    'TILELANG_NPU_CONDA_SH',
    'TILELANG_NPU_CONDA_ENV',
    'TILELANG_NPU_CONDA_ENVS_PATH',
    'TILELANG_NPU_PYTHON',
    'TILELANG_NPU_CMAKE_BIN',
    'TILELANG_NPU_BUILD_SUBDIR',
    'TILELANG_NPU_NINJA_JOBS',
    'TILELANG_NPU_EXAMPLE_TESTS',
)

# TileFoundry host adapter: CANN 9.3.0 + conda tilefoundry-npu env; CPU tests
# run without CANN sourced (to avoid ortools/CANN native library collision)
# and device tests run windowed on the card under CANN + torch_npu. Optional
# TILEFOUNDRY_NPU_TILELANG_{SOURCE,BUILD} keys reuse a tilelang-npu checkout
# build (never rebuilding TileLang locally).
TILEFOUNDRY_NPU_KEYS = (
    'TILEFOUNDRY_NPU_CANN_SETUP',
    'TILEFOUNDRY_NPU_CONDA_SH',
    'TILEFOUNDRY_NPU_CONDA_ENV',
    'TILEFOUNDRY_NPU_CONDA_ENVS_PATH',
    'TILEFOUNDRY_NPU_PYTHON',
    'TILEFOUNDRY_NPU_TILELANG_SOURCE',
    'TILEFOUNDRY_NPU_TILELANG_BUILD',
    'TILEFOUNDRY_NPU_CPU_TESTS',
    'TILEFOUNDRY_NPU_CPU_WORKERS',
    'TILEFOUNDRY_NPU_DEVICE_TESTS',
    'TILEFOUNDRY_NPU_MSPROF_LOCK',
    'TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE',
    'TILEFOUNDRY_NPU_BUILD_SUBDIR',
    'TILEFOUNDRY_NPU_CANNBENCH_REPO',
    'TILEFOUNDRY_NPU_WORK_ROOT',
    'TILEFOUNDRY_NPU_CANNBENCH_ARCH',
    'TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE',
    'TILEFOUNDRY_NPU_CANNBENCH_TARGETS',
    'TILEFOUNDRY_NPU_CANNBENCH_MCP_URL',
    'TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN',
    'TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE',
    'TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION',
    'TILEFOUNDRY_NPU_CANNBENCH_POOL_ID',
    'TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN',
    'TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT',
    'TILEFOUNDRY_NPU_LLM_BACKEND',
    'TILEFOUNDRY_NPU_LLM_API_BASE',
    'TILEFOUNDRY_NPU_LLM_API_KEY_ENV',
    'TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL',
    'TILEFOUNDRY_NPU_LLM_EXPLORE_MODEL',
    'TILEFOUNDRY_NPU_LLM_ESCALATE_MODEL',
    'TILEFOUNDRY_NPU_LLM_REPAIR_MODEL',
    'TILEFOUNDRY_NPU_LLM_FALLBACK_MODEL',
    'TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES',
    'TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC',
    'TILEFOUNDRY_NPU_L3_TIMEOUT_SEC',
    'TILEFOUNDRY_NPU_LLM_MAX_EFFORT',
    'TILEFOUNDRY_NPU_LLM_EFFORT_CAP',
    'TILEFOUNDRY_NPU_LLM_PASS_REASONING_EFFORT',
    'TILEFOUNDRY_NPU_LLM_STREAM',
    'TILEFOUNDRY_NPU_LLM_MAX_CONCURRENCY',
    'TILEFOUNDRY_NPU_LLM_EXPLOIT_TIMEOUT_SEC',
    'TILEFOUNDRY_NPU_LLM_REASONING_TIMEOUT_SEC',
    'TILEFOUNDRY_NPU_LLM_STARTUP_TIMEOUT_SEC',
    'TILEFOUNDRY_NPU_LLM_IDLE_STALL_TIMEOUT_SEC',
    'TILEFOUNDRY_NPU_LLM_CLI_BIN',
    'TILEFOUNDRY_NPU_LLM_CLI_TEMPLATE',
    'TILEFOUNDRY_NPU_LLM_HTTP_PROXY',
    'TILEFOUNDRY_NPU_LLM_HTTPS_PROXY',
    'TILEFOUNDRY_NPU_LLM_ALL_PROXY',
    'TILEFOUNDRY_NPU_LLM_NO_PROXY',
    'TILEFOUNDRY_NPU_L4_EXTERNAL_PROBE_LIMIT',
    'TILEFOUNDRY_NPU_OCCUPANCY_POLICY',
    'TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS',
    'TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB',
    'TILEFOUNDRY_NPU_AGY_BIN',
    'TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL',
    'TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL',
    'TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL',
    'TILEFOUNDRY_NPU_AGY_COMMON_ARGS',
    'TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY',
    'TILEFOUNDRY_NPU_CANN_SERIES',
    'TILEFOUNDRY_NPU_TILELANG_REPO',
    'TILEFOUNDRY_NPU_CANNBOT_DSL_REPO',
    'TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO',
    'TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS',
)

# MegaKernel host adapter: CANN + Python 3.11 conda megakernel-npu env + LLVM;
# sourceWrite=false so builds stage incrementally into MEGAKERNEL_NPU_WORK_ROOT
# and ninja concurrency is strictly capped to <= 16 cores.
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

# The adapter ID -> the prefix it owns, the selector variable it honours and the
# machine file the checkout keeps beside its template.
ADAPTERS = {
    'ascendnpu-ir': {
        'prefix': 'ASCENDNPU_',
        'selector': 'ASCENDNPU_ENV_FILE',
        'directory': 'ascendnpu-ir',
        'keys': ASCENDNPU_KEYS,
    },
    'tilelang-hivm': {
        'prefix': 'TILELANG_HIVM_',
        'selector': 'TILELANG_HIVM_ENV_FILE',
        'directory': 'tilelang-hivm',
        'keys': TILELANG_HIVM_KEYS,
    },
    'tilelang-hivm-npu': {
        'prefix': 'TILELANG_HIVM_',
        'selector': 'TILELANG_HIVM_ENV_FILE',
        'directory': 'tilelang-hivm',
        'keys': TILELANG_HIVM_KEYS,
    },
    'tla-npu': {
        'prefix': 'TLA_NPU_',
        'selector': 'TLA_NPU_ENV_FILE',
        'directory': 'tla-npu',
        'keys': TLA_NPU_KEYS,
    },
    'forgekernel-ascend-npu': {
        'prefix': 'FK_ASCEND_',
        'selector': 'FK_ASCEND_ENV_FILE',
        'directory': 'forgekernel-ascend-npu',
        'keys': FK_ASCEND_KEYS,
    },
    'bska-npu': {
        'prefix': 'BSKA_NPU_',
        'selector': 'BSKA_NPU_ENV_FILE',
        'directory': 'bska-npu',
        'keys': BSKA_NPU_KEYS,
    },
    # The host-local TileLang Adapter owns both its file and its prefix. Unlike
    # the two HIVM Adapters it shares nothing with them: a build directory it
    # leases must not be confused with the one the docker gates write.
    'tilelang-npu': {
        'prefix': 'TILELANG_NPU_',
        'selector': 'TILELANG_NPU_ENV_FILE',
        'directory': 'tilelang-npu',
        'keys': TILELANG_NPU_KEYS,
    },
    'tilefoundry-npu': {
        'prefix': 'TILEFOUNDRY_NPU_',
        'selector': 'TILEFOUNDRY_NPU_ENV_FILE',
        'directory': 'tilefoundry-npu',
        'keys': TILEFOUNDRY_NPU_KEYS,
    },
    'megakernel-npu': {
        'prefix': 'MEGAKERNEL_NPU_',
        'selector': 'MEGAKERNEL_NPU_ENV_FILE',
        'directory': 'megakernel-npu',
        'keys': MEGAKERNEL_NPU_KEYS,
    },
}


class MachineEnvError(ValueError):
    """A machine file that cannot be honoured; never a Gate result."""


@dataclass(frozen=True)
class Machine:
    """The resolved machine parameters an Adapter and the engine both use."""

    adapter: str
    path: Path | None
    values: dict

    def get(self, name: str) -> str:
        return self.values.get(name, '')

    @property
    def build_subdir(self) -> str:
        if self.build_key in self.values and not self.values[self.build_key].strip():
            # An empty value in the file is a typo, not "use the default": the
            # file spells "unset" by leaving the key out. The Adapters refuse it
            # too, and the engine must agree, or a run would lease a directory the
            # Gate then refuses to build in.
            raise MachineEnvError(f'{self.path}: {self.build_key} is set but empty')
        return self.values.get(self.build_key, 'build') or 'build'

    @property
    def build_key(self) -> str:
        return ADAPTERS[self.adapter]['prefix'] + 'BUILD_SUBDIR'

    @property
    def jobs_key(self) -> str:
        return ADAPTERS[self.adapter]['prefix'] + 'NINJA_JOBS'

    @property
    def container_keys(self) -> tuple:
        prefix = ADAPTERS[self.adapter]['prefix']
        return (prefix + 'CONTAINER_NAME', prefix + 'CONTAINER_USER', prefix + 'CONTAINER_HOME')

    def container_identity(self):
        """(name, user, home) or None when this machine names no container."""
        values = [self.get(name).strip() for name in self.container_keys]
        if not any(values):
            return None
        missing = [name for name, value in zip(self.container_keys, values) if not value]
        if missing:
            raise MachineEnvError(
                f'{self.path or "no machine file"} does not define {", ".join(missing)}; '
                f'container identity is read from that file only')
        return tuple(values)

    def ninja_jobs(self) -> str:
        if self.jobs_key in self.values and not self.values[self.jobs_key].strip():
            raise MachineEnvError(f'{self.path}: {self.jobs_key} is set but empty')
        raw = self.get(self.jobs_key).strip()
        if not raw:
            return '$(nproc)'
        if not raw.isdigit() or int(raw) < 1:
            raise MachineEnvError(f'{self.jobs_key} must be a positive integer: {raw!r}')
        return raw


def parse(text: str, path: Path, keys) -> dict:
    """Parse the strict `NAME=value` file; the file is never executed."""
    values: dict = {}
    allowed = tuple(keys)
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = _ASSIGNMENT.match(line)
        if match is None:
            raise MachineEnvError(f'{path}: line {number} is not a NAME=value assignment: {raw!r}')
        key, value = match.group(1), match.group(2).strip()
        if key not in allowed:
            raise MachineEnvError(f'{path}: line {number} sets unknown key {key!r}; '
                                  f'allowed keys are {", ".join(allowed)}')
        if key in values:
            raise MachineEnvError(f'{path}: line {number} sets {key} a second time')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        elif value[:1] in ('"', "'") or value[-1:] in ('"', "'"):
            raise MachineEnvError(f'{path}: line {number} has an unbalanced quote in {raw!r}')
        values[key] = value
    return values


def adapters_root() -> Path:
    return Path(__file__).resolve().parent / 'adapters'


def find_file(adapter: str, env=None) -> Path | None:
    """The machine file for one Adapter, or None when this machine keeps none.

    Order: the selector variable (`<PREFIX>ENV_FILE`, which must be an absolute
    existing path when set), then the packaged `adapters/<dir>/env.sh` beside
    the committed template. There is deliberately no home-directory search: no
    Gate may read a file nobody reviewed.
    """
    environ = os.environ if env is None else env
    spec = ADAPTERS[adapter]
    explicit = (environ.get(spec['selector']) or '').strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise MachineEnvError(f"{spec['selector']} must be an absolute path: {explicit!r}")
        if not path.exists():
            raise MachineEnvError(f"{spec['selector']} does not exist: {path}")
        if not path.is_file():
            raise MachineEnvError(f"{spec['selector']} is not a regular file: {path}")
        return path
    candidate = adapters_root() / spec['directory'] / 'env.sh'
    return candidate if candidate.is_file() else None


def read(adapter: str, env=None, env_file: Path | None = None) -> Machine:
    """Resolve one Adapter's machine parameters, failing closed on a bad file."""
    spec = ADAPTERS[adapter]
    path = Path(env_file) if env_file is not None else find_file(adapter, env)
    if path is not None:
        if not path.is_absolute():
            raise MachineEnvError(f'{spec["selector"]} must be an absolute path: {path!r}')
        if not path.is_file():
            raise MachineEnvError(f'{path} is not a regular file')
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            raise MachineEnvError(f'cannot read {path}: {exc}') from exc
        values = parse(text, path, spec['keys'])
    else:
        values = {}
    build = values.get(spec['prefix'] + 'BUILD_SUBDIR')
    if build is not None and not build.strip():
        raise MachineEnvError(
            f'{path}: {spec["prefix"]}BUILD_SUBDIR is set but empty; leave the key out '
            f'to use the default')
    build = (build or 'build').strip()
    if not SAFE_NAME.fullmatch(build) or build in {'.', '..'}:
        raise MachineEnvError(
            f'{path or "no machine file"}: {spec["prefix"]}BUILD_SUBDIR is not a plain '
            f'directory name: {build!r}')
    return Machine(adapter, path, values)
