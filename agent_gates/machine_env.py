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
