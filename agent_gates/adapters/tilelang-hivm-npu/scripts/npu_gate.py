#!/usr/bin/env python3
"""Run the Ascend hardware sanity Gate inside the build container.

This Adapter is selected separately. It is **not** an automatic SKIP mode of
`tilelang-hivm`: the owner selects `tilelang-hivm-npu` and sets
`TILELANG_HIVM_DEVICE_ID`, and it contributes its own Recipe and its own Gate.
Without that selection no hardware test runs at all.

Scope warning
-------------

The Gate runs `examples/ascend/test_vecadd.py -k simd`, which exercises the
**plain Ascend SIMD path** (AscendC plus BiSheng, target `ascend`). It does
**not** exercise the HIVM CodeGen path, which today is a TEST_ONLY PREBUILT
module (`tilelang/ascend/codegen.py`, `HIVM_CODEGEN`). A PASS here proves that
this machine can build and run a real TileLang Ascend kernel on a real,
authorized device with a real numerical comparison. It does **not** prove HIVM
End-to-End CodeGen, and it must never be reported as such.

Device authorization
--------------------

`TILELANG_HIVM_DEVICE_ID` is mandatory and has no default. This Adapter never
discovers, chooses or substitutes a device. A machine with several cards is
shared, so silently picking another card would run someone else's work. The
authorized id is mapped to the physical device id through the driver
(`acl.rt.get_phy_dev_id_by_logic_dev_id`) both before and after the tests; the
Gate fails when that mapping is not exactly the authorized id. A failure to
synchronize the device is reported with its original message and fails the Gate.
No device reset, no device substitution and no automatic retry is attempted.

Environment variables (all read at run time):

  TILELANG_HIVM_DEVICE_ID         required, the authorized NPU device id. Not in
                                  the machine file: it authorizes one run, not a
                                  machine.
  TILELANG_HIVM_CONTAINER_NAME    optional, name of a running build container.
                                  The machine file's TILELANG_HIVM_CONTAINER_NAME
                                  is the only source, exactly as in
                                  `tilelang-hivm`. A shell export has no effect.
  TILELANG_HIVM_CONTAINER_HOME    optional, HOME inside the container; read from
                                  the same machine file
  TILELANG_HIVM_CONTAINER_USER    ignored; the device Gate runs as root, because
                                  the driver refuses a non-root opener once any
                                  namespace holds the card (`uda_occupy_dev_by_ns`)
  TILELANG_HIVM_CONTAINER_SOURCE  optional, explicit container path of the source
                                  root; when set it must match the bind-mount
                                  mapping, when unset the mapping is used directly
  TILELANG_HIVM_GATE_TIMEOUT      optional, container-side timeout in seconds; used
                                  ONLY when no `--timeout` argument is passed

Toolchain configuration comes from a file, not from the launching shell:

  TILELANG_HIVM_ENV_FILE          path of the env.sh that holds CANN/Conda;
                                  default the checkout's own
                                  adapters/tilelang-hivm/env.sh

The file contains plain `NAME=value` assignments for these keys, and nothing
else is read from it:

  TILELANG_HIVM_CANN_SETUP        optional, CANN set_env.sh inside the container
  TILELANG_HIVM_CONDA_SH          optional, conda.sh inside the container
  TILELANG_HIVM_CONDA_ENV         optional, conda environment to activate

The file is the single source of truth for CANN and Conda; this Gate shares it
with `tilelang-hivm`, because both use the `TILELANG_HIVM_` prefix. Exporting the
three toolchain variables in the launching shell has no effect.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PROG = 'tilelang-hivm-npu-gate'
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
# Grace between `timeout` sending TERM and the container-side SIGKILL; the outer
# Harness Recipe timeout must still exceed `--timeout + KILL_AFTER_SECONDS`.
KILL_AFTER_SECONDS = 15
# Read-only Host probe of the authorized card. `npu-smi info` answers in well
# under a second; a hung driver must block the Gate rather than delay it.
NPU_SMI = 'npu-smi'
PROBE_TIMEOUT_SECONDS = 60

# The Gate's test selection: repository-relative, fixed in this digest-pinned
# file, and never read from the environment. `-k simd` selects the SIMD
# parameterization of the vector-add example.
GATE_TEST = 'examples/ascend/test_vecadd.py'
GATE_SELECT = 'simd'

# Environment variables that must be identical for the base Adapter and this
# one: they describe the same container and the same checkout.
CONTAINER_VARIABLES = ('TILELANG_HIVM_CONTAINER_NAME', 'TILELANG_HIVM_CONTAINER_USER',
                       'TILELANG_HIVM_CONTAINER_HOME', 'TILELANG_HIVM_CANN_SETUP',
                       'TILELANG_HIVM_CONDA_SH', 'TILELANG_HIVM_CONDA_ENV',
                       'TILELANG_HIVM_CONTAINER_SOURCE')


class BridgeError(Exception):
    """A configuration, mapping or launch failure, not a Gate result."""

    EXIT = 2

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def env_value(name: str, *, required: bool = False, default: str | None = None) -> str | None:
    value = os.environ.get(name, '')
    value = value.strip() if isinstance(value, str) else ''
    if not value:
        if required:
            raise BridgeError(f'missing required environment variable {name}')
        return default
    return value


def env_path(name: str, *, required: bool = False) -> Path | None:
    value = env_value(name, required=required)
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        raise BridgeError(f'{name} must be an absolute path inside the container: {value!r}')
    if '\n' in value or '\x00' in value:
        raise BridgeError(f'{name} contains an illegal character')
    return path


# Keys the toolchain file may define. Anything else is an error: a typo must not
# silently leave a component unconfigured.
# Keys the toolchain file may define. Anything else is an error: a typo must not
# silently leave a component unconfigured. The first three are this Gate's own;
# the rest belong to `scripts/setup-python-env.sh` in the `tilelang-hivm`
# Adapter, which creates the environment those three point at. This Gate reads
# the same machine file, so it must accept the keys it does not use.
TOOLCHAIN_KEYS = (
    'TILELANG_HIVM_CANN_SETUP',
    'TILELANG_HIVM_CONDA_SH',
    'TILELANG_HIVM_CONDA_ENV',
    'TILELANG_HIVM_CONDA_ENVS_PATH',
    'TILELANG_HIVM_WHEEL_DIR',
    'TILELANG_HIVM_PIP_INDEX',
    'TILELANG_HIVM_PYTHON_VERSION',
    'TILELANG_HIVM_SOURCE_ROOT',
    # Container identity, shared with `tilelang-hivm` because both Gates enter
    # the same container. The device Gate ignores the user key (it runs as root)
    # but must accept it, or one file could not serve both Adapters.
    'TILELANG_HIVM_CONTAINER_NAME',
    'TILELANG_HIVM_CONTAINER_USER',
    'TILELANG_HIVM_CONTAINER_HOME',
    # The Late-hivm compiler, read by `tilelang-hivm`'s Gate to forward
    # `HIVM_COMPILER` and `HIVM_TARGET` into the container. This Gate runs the
    # vector-add example, which does not read them, but it reads the same machine
    # file and must accept every key that file may define.
    'TILELANG_HIVM_HIVM_COMPILER',
    'TILELANG_HIVM_HIVM_TARGET',
)
# The keys this Gate needs for the container it enters. The user key is excluded
# from the requirement on purpose: the device Gate runs as root, so a machine
# that only ever runs the device Gate need not name a container user.
CONTAINER_KEYS = ('TILELANG_HIVM_CONTAINER_NAME', 'TILELANG_HIVM_CONTAINER_HOME')
# The provisioning keys that must be absolute paths. `PIP_INDEX` is a URL, and
# `PYTHON_VERSION` and `CONDA_ENV` are plain names, so neither is checked here.
PROVISIONING_PATHS = (
    'TILELANG_HIVM_CONDA_ENVS_PATH',
    'TILELANG_HIVM_WHEEL_DIR',
    'TILELANG_HIVM_SOURCE_ROOT',
)
PROVISIONING_KEYS = tuple(key for key in TOOLCHAIN_KEYS
                          if key not in ('TILELANG_HIVM_CANN_SETUP',
                                         'TILELANG_HIVM_CONDA_SH',
                                         'TILELANG_HIVM_CONDA_ENV',
                                         'TILELANG_HIVM_CONTAINER_NAME',
                                         'TILELANG_HIVM_CONTAINER_USER',
                                         'TILELANG_HIVM_CONTAINER_HOME',
                                         'TILELANG_HIVM_HIVM_COMPILER',
                                         'TILELANG_HIVM_HIVM_TARGET'))
# Where the machine toolchain file lives. It is never committed: `env.sh` is
# git-ignored because it names one machine's paths. The search order is:
#
#   1. $TILELANG_HIVM_ENV_FILE, explicit. It must exist when set.
#   2. <checkout>/adapters/tilelang-hivm/env.sh, next to that Adapter's
#      committed env.example.sh. Found by walking up from this file, so the same
#      file resolves from a checkout and from an installed Pack inside one. Both
#      TileLang Gates read that one file, because both use the TILELANG_HIVM_
#      prefix; this Adapter has no template of its own.
#
# A step that finds nothing is not an error: CANN and Conda are optional, so an
# absent file means "configure nothing".
ADAPTER_DIR = Path(__file__).resolve().parent.parent
CHECKOUT_MARKER = Path('adapters') / 'tilelang-hivm' / 'env.example.sh'


def checkout_env_file() -> Path | None:
    """The in-repository machine file, when this Pack belongs to a checkout."""
    # Packaged adapters share a local adapters directory, including the NPU sibling.
    packaged = ADAPTER_DIR.parent / 'tilelang-hivm' / 'env.example.sh'
    if ADAPTER_DIR.parent.name == 'adapters' and packaged.is_file():
        return packaged.with_name('env.sh')
    for ancestor in (ADAPTER_DIR, *ADAPTER_DIR.parents):
        if (ancestor / CHECKOUT_MARKER).is_file():
            return ancestor / CHECKOUT_MARKER.parent / 'env.sh'
    return None


def default_env_file() -> Path | None:
    """The machine file the checkout owns, or None outside a checkout.

    There is no second location. A machine that keeps its file elsewhere names
    it in `TILELANG_HIVM_ENV_FILE`; guessing a home-directory path silently
    picked up a file nobody was reviewing.
    """
    return checkout_env_file()

def resolved_env_file() -> Path | None:
    """The machine file to read, or None when this machine has none.

    `TILELANG_HIVM_ENV_FILE` wins and must be an absolute path that exists. With
    nothing set, the checkout's own `env.sh` is used, and its absence is not an
    error: CANN and Conda are optional, so "no file" means "configure nothing".
    """
    explicit = env_value('TILELANG_HIVM_ENV_FILE')
    if not explicit:
        return default_env_file()
    path = Path(explicit).expanduser()
    if not path.is_absolute():
        raise BridgeError(f'TILELANG_HIVM_ENV_FILE must be an absolute path: {explicit!r}')
    if not path.exists():
        raise BridgeError(f'TILELANG_HIVM_ENV_FILE does not exist: {path}')
    return path

# A conservative `NAME=value` grammar. Values may be bare, single- or
# double-quoted. No expansion, no command substitution, no line continuations:
# the file is data, so reading it never executes Host code.
_ASSIGNMENT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')


def _parse_env_file(text: str, path: Path) -> dict[str, str]:
    """Parse the strict `NAME=value` toolchain file; never executes it."""
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = _ASSIGNMENT.match(line)
        if match is None:
            raise BridgeError(f'{path}: line {number} is not a NAME=value assignment: {raw!r}')
        key, value = match.group(1), match.group(2).strip()
        if key not in TOOLCHAIN_KEYS:
            raise BridgeError(f'{path}: line {number} sets unknown key {key!r}; '
                              f'allowed keys are {", ".join(TOOLCHAIN_KEYS)}')
        if key in values:
            raise BridgeError(f'{path}: line {number} sets {key} a second time')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        elif value[:1] in ('"', "'") or value[-1:] in ('"', "'"):
            raise BridgeError(f'{path}: line {number} has an unbalanced quote in {raw!r}')
        values[key] = value
    return values


def toolchain_from_env_file() -> tuple[Path | None, Path | None, str | None]:
    """Read CANN/Conda from the configured file. The file is authoritative.

    The same file serves `tilelang-hivm` and `tilelang-hivm-npu`, because both
    use the `TILELANG_HIVM_` prefix. A missing default file is not an error; an
    explicitly selected one that is missing or malformed is. Exporting the three
    toolchain variables in the launching shell has no effect.
    """
    path = resolved_env_file()
    if path is None or not path.exists():
        return None, None, None
    if not path.is_file():
        raise BridgeError(f'TILELANG_HIVM_ENV_FILE is not a regular file: {path}')
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc

    values = _parse_env_file(text, path)

    def as_path(name: str, raw: str) -> Path | None:
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            raise BridgeError(f'{path}: {name} must be an absolute path: {raw!r}')
        if '\n' in raw or '\x00' in raw:
            raise BridgeError(f'{path}: {name} contains an illegal character')
        return candidate

    cann = as_path('TILELANG_HIVM_CANN_SETUP', values.get('TILELANG_HIVM_CANN_SETUP', ''))
    conda_sh = as_path('TILELANG_HIVM_CONDA_SH', values.get('TILELANG_HIVM_CONDA_SH', ''))
    conda_env = values.get('TILELANG_HIVM_CONDA_ENV', '') or None
    print(f'{PROG}: toolchain file {path}', file=sys.stderr)
    return cann, conda_sh, conda_env


def conda_envs_path_from_env_file() -> Path | None:
    """The extra Conda environment directory, when the machine file names one.

    A Conda installation searches only the environments under its own prefix. A
    user's environments normally live in `~/.conda/envs`, so without this value
    `conda activate` fails with `EnvironmentNameNotFound: Could not find conda
    environment: <name>` even though `conda env list` shows the environment.
    """
    path = resolved_env_file()
    if path is None or not path.exists():
        return None
    try:
        values = _parse_env_file(path.read_text(encoding='utf-8'), path)
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc
    raw = values.get('TILELANG_HIVM_CONDA_ENVS_PATH', '')
    if not raw:
        return None
    if not Path(raw).is_absolute():
        raise BridgeError(f'{path}: TILELANG_HIVM_CONDA_ENVS_PATH must be an absolute path: {raw!r}')
    return Path(raw)


def provisioning_from_env_file() -> dict[str, str]:
    """The provisioning values from the same toolchain file, as raw strings.

    `setup-python-env.sh` is handed these values, so a machine declares its
    wheel directory, index and interpreter version in the one file it already
    edits rather than in a shell export that leaves no record. This Gate does
    not use them for its own run; it exposes them so that the machine file has
    one reader, which is what keeps the two Adapters from disagreeing about
    which keys are legal.
    """
    path = resolved_env_file()
    if path is None or not path.exists():
        return {}
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc
    values = _parse_env_file(text, path)
    for name in PROVISIONING_PATHS:
        raw = values.get(name, '')
        if raw and not Path(raw).is_absolute():
            raise BridgeError(f'{path}: {name} must be an absolute path: {raw!r}')
    return {key: value for key, value in values.items() if key in PROVISIONING_KEYS}


def env_file_values() -> dict[str, str]:
    """Every key the machine file defines, parsed once for the whole Gate."""
    path = resolved_env_file()
    if path is None or not path.exists():
        return {}
    if not path.is_file():
        raise BridgeError(f'TILELANG_HIVM_ENV_FILE is not a regular file: {path}')
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc
    return _parse_env_file(text, path)


def container_identity() -> tuple[str, str, Path | None]:
    """The container this Gate enters, read from the machine file.

    `tilelang-hivm` creates the container and reads its identity from the same
    file, so the two Adapters cannot address two different containers. The
    launching shell is not consulted: a stale `TILELANG_HIVM_CONTAINER_NAME`
    export used to be the only source, and it could name a stopped container
    while the machine file named the live one.

    The user key is not required here. This Gate runs as root, so it needs the
    name and HOME only, and refusing a machine that leaves the user out would
    reject a configuration `tilelang-hivm` already accepts.
    """
    path = resolved_env_file()
    values = env_file_values()
    missing = [name for name in CONTAINER_KEYS if not values.get(name)]
    if missing:
        where = str(path) if path is not None else 'no machine file was found'
        raise BridgeError(
            f'{where} does not define {", ".join(missing)}. Container identity is read '
            f'from that file only; copy env.example.sh to env.sh beside it, set '
            f'TILELANG_HIVM_CONTAINER_NAME and TILELANG_HIVM_CONTAINER_HOME to the '
            f'container docker/config.env created, and run the Gate again.')
    container = values['TILELANG_HIVM_CONTAINER_NAME'].strip()
    home = values['TILELANG_HIVM_CONTAINER_HOME'].strip()
    if not SAFE_NAME.fullmatch(container):
        raise BridgeError('container name must be a plain name, not an option-like string')
    if not home.startswith('/'):
        raise BridgeError(f'TILELANG_HIVM_CONTAINER_HOME must be an absolute container path: {home!r}')
    if any('\n' in value or '\x00' in value for value in (container, home)):
        raise BridgeError('container identity values must not contain a newline or NUL')
    return container, home, path


def device_id() -> str:
    """The authorized device. There is no discovery and no default."""
    value = env_value('TILELANG_HIVM_DEVICE_ID', required=True) or ''
    if not value.isdigit() or int(value) >= 64:
        raise BridgeError(f'TILELANG_HIVM_DEVICE_ID must be a plain device number: {value!r}')
    return str(int(value))


def npu_smi(*args: str) -> str:
    """Run one read-only `npu-smi info` query on the Host.

    Only `info` subcommands are ever issued: the probe must not reset, set or
    otherwise change a device. Every way of not getting an answer - a missing
    binary, a non-zero exit, a timeout, a signal - is a BridgeError, because
    "unknown" is not "free" and a card whose state cannot be read must not be
    handed to a test that would then share it with another user's job.
    """
    try:
        done = subprocess.run([NPU_SMI, 'info', *args], capture_output=True, text=True,
                              timeout=PROBE_TIMEOUT_SECONDS)
    except FileNotFoundError as error:
        raise BridgeError(f'cannot determine NPU occupancy: {NPU_SMI} is not installed on the Host '
                          f'({error})') from error
    except subprocess.TimeoutExpired as error:
        raise BridgeError(f'cannot determine NPU occupancy: `{NPU_SMI} info {" ".join(args)}` '
                          f'did not answer within {PROBE_TIMEOUT_SECONDS}s') from error
    except OSError as error:
        raise BridgeError(f'cannot determine NPU occupancy: {NPU_SMI} could not be started: {error}') from error
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise BridgeError(f'cannot determine NPU occupancy: `{NPU_SMI} info {" ".join(args)}` exited '
                          f'{done.returncode}: {detail[0] if detail else "no message"}')
    return done.stdout


def npu_id_of_physical(device: str) -> str:
    """The `npu-smi` NPU ID whose Chip Phy-ID is the authorized device.

    The Gate authorizes a physical id: the container side asserts the driver
    maps its logical device onto exactly that id. `npu-smi` numbers cards by
    NPU ID, which is not guaranteed to equal the physical id, so the mapping
    table is consulted instead of assuming the two namespaces coincide. A
    device missing from the table, or claimed by more than one row, cannot be
    probed and is refused rather than guessed.
    """
    table = npu_smi('-m')
    npu_ids = []
    for line in table.splitlines():
        fields = line.split()
        if len(fields) < 4 or not all(field.isdigit() for field in fields[:4]):
            continue
        if fields[3] == device:
            npu_ids.append(fields[0])
    unique = sorted(set(npu_ids))
    if not unique:
        raise BridgeError(f'cannot determine NPU occupancy: no device in the npu-smi mapping table '
                          f'reports physical id {device}')
    if len(unique) > 1:
        raise BridgeError(f'cannot determine NPU occupancy: physical id {device} is claimed by several '
                          f'npu-smi NPU IDs {unique}')
    return unique[0]


def processes_on(npu_id: str) -> list[tuple[str, str]]:
    """The `(pid, name)` pairs `npu-smi` reports for one NPU ID.

    The answer must be structurally recognisable in full: it has to name the
    queried NPU ID, and every process row it prints has to match the documented
    layout, or the table has to say in the driver's own words that the device
    is idle. A row this Gate cannot read means the driver's format changed, and
    a format this Gate cannot read is unknown occupancy, never a free card.
    Anything else - an empty table, a partially matching table, a different
    layout - is a BridgeError.
    """
    text = npu_smi('-t', 'proc-mem', '-i', npu_id)
    if not re.search(rf'^\s*NPU ID\s*:\s*{re.escape(npu_id)}\s*$', text, re.MULTILINE):
        raise BridgeError(f'cannot determine NPU occupancy: `{NPU_SMI} info -t proc-mem -i {npu_id}` '
                          f'did not report NPU ID {npu_id}')
    # The driver prints this sentence, and only this sentence, when the device
    # holds nothing. It is an explicit statement, not the absence of rows.
    idle = len(re.findall(r'^\s*No process in device\.?\s*$', text, re.MULTILINE))
    pattern = re.compile(r'Process id:(\d+)\s+Process name:(\S+)\s+Process memory\(MB\):(\d+)')
    rows = [(match.group(1), match.group(2)) for match in pattern.finditer(text)]
    announced = len(re.findall(r'Process id:', text))
    if rows and idle:
        raise BridgeError(f'cannot determine NPU occupancy: `{NPU_SMI} info -t proc-mem -i {npu_id}` '
                          f'reported both processes and an idle device')
    if len(rows) != announced:
        raise BridgeError(f'cannot determine NPU occupancy: `{NPU_SMI} info -t proc-mem -i {npu_id}` '
                          f'reported {announced} process row(s) but only {len(rows)} matched the known '
                          f'layout; the driver answer is not understood')
    if not rows and not idle:
        raise BridgeError(f'cannot determine NPU occupancy: `{NPU_SMI} info -t proc-mem -i {npu_id}` '
                          f'neither listed a process nor said the device is idle')
    return rows


def require_device_free(device: str) -> str:
    """Refuse to dispatch onto a card another process is already using.

    A5 cards are shared by every user of the machine. An idle card is not an
    assignment either, so this check only removes the case where the device is
    demonstrably busy; authorization is still the explicit variable. Only the
    count and the `npu-smi` identity are reported - other users' process ids
    and command lines are not copied into the receipt.
    """
    npu_id = npu_id_of_physical(device)
    holders = processes_on(npu_id)
    if holders:
        raise BridgeError(f'NPU device {device} (npu-smi NPU ID {npu_id}) is already in use by '
                          f'{len(holders)} process(es) this Gate did not start; refusing to share a '
                          f'card that was not explicitly released to this task')
    return npu_id


def parse_argv(argv: list[str]) -> int:
    """Optional `--timeout SECONDS`; the manifest always passes one."""
    value = None
    if argv[:1] == ['--timeout']:
        if len(argv) != 2:
            raise BridgeError('--timeout requires exactly one value and no other argument')
        value = argv[1]
    elif argv:
        raise BridgeError(f'unexpected arguments: {argv!r}')
    else:
        value = env_value('TILELANG_HIVM_GATE_TIMEOUT', default='1500')
    if value is None or not value.isdigit() or int(value) < 1:
        raise BridgeError(f'gate timeout must be a positive integer number of seconds: {value!r}')
    return int(value)


def docker_exec_prefix(container: str, home: str, workdir: str) -> list[str]:
    """Run the container side as root.

    The Ascend driver holds `davinci0`/`davinci1` per user namespace
    (`uda_occupy_dev_by_ns`). Once any namespace holds the card, a non-root
    opener is refused with EBUSY even when it is a member of the device group,
    so an ordinary container user cannot reliably reach the device. root is
    not subject to that check; the device Gate therefore runs as root. The
    build Gate keeps the container user, because building must stay provable
    for an ordinary user.
    """
    return ['docker', 'exec', '-u', 'root', '-e', f'HOME={home}', '-w', workdir, container]


def container_mounts(container: str) -> list[dict]:
    done = subprocess.run(['docker', 'inspect', '--format', '{{json .Mounts}}', container],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise BridgeError(f'docker inspect failed for {container!r}: '
                          f'{done.stderr.strip() or done.stdout.strip()}')
    try:
        mounts = json.loads(done.stdout)
    except ValueError as exc:
        raise BridgeError(f'docker inspect returned unparsable mounts for {container!r}') from exc
    if not isinstance(mounts, list):
        raise BridgeError(f'docker inspect returned no mount list for {container!r}')
    return [item for item in mounts if isinstance(item, dict)]


def map_host_to_container(host_root: Path, mounts: list[dict], container: str = '<container>') -> Path:
    """Resolve the Host source root through bind mounts; the deepest bind wins."""
    host_root = Path(os.path.realpath(host_root))
    candidates = []
    for mount in mounts:
        if mount.get('Type') != 'bind':
            continue
        source, destination = mount.get('Source'), mount.get('Destination')
        if not source or not destination:
            continue
        source = Path(os.path.realpath(source))
        destination = Path(destination)
        if host_root == source:
            mapped = destination
        elif host_root.is_relative_to(source):
            mapped = destination / host_root.relative_to(source)
        else:
            continue
        candidates.append((len(source.parts), mapped))
    if not candidates:
        raise BridgeError(
            f'host source root {host_root} is not exposed by any bind mount of container '
            f'{container!r}; refusing to substitute a different clone inside the container')
    depth = max(rank for rank, _ in candidates)
    best = [path for rank, path in candidates if rank == depth]
    unique = []
    for path in best:
        if path not in unique:
            unique.append(path)
    if len(unique) != 1:
        raise BridgeError(f'ambiguous container mapping for {host_root}: '
                          f'{[str(path) for path in unique]}')
    return unique[0]


def identity_probe(prefix: list[str], path: str) -> str:
    """`dev:ino` of `path` inside the container, using `stat` (not Python)."""
    expression = 'stat -c "%d:%i" ' + shlex.quote(path)
    done = subprocess.run([*prefix, 'bash', '-c', expression], capture_output=True, text=True)
    if done.returncode != 0:
        raise BridgeError(f'container cannot stat {path}: {done.stderr.strip() or done.stdout.strip()}')
    lines = done.stdout.strip().splitlines()
    if not lines or ':' not in lines[-1]:
        raise BridgeError(f'container identity probe for {path} returned {done.stdout.strip()!r}')
    return lines[-1]


def verify_same_tree(host_root: Path, prefix: list[str], container_root: str) -> None:
    host_stat = host_root.stat()
    remote = identity_probe(prefix, container_root)
    expected = f'{host_stat.st_dev}:{host_stat.st_ino}'
    if remote != expected:
        raise BridgeError(
            f'container source root {container_root} is not the host source root {host_root} '
            f'(container {remote} != host {expected}); refusing to run against a different tree')


# Container-side runner. It sets the visible device *before* importing the
# runtime, observes the driver's logical-to-physical mapping before and after the
# tests, instruments the move-to-device call and the numerical comparison, then
# prints one raw report on standard output. pytest's own output is redirected to
# standard error so the report stays the only line on standard output. Every
# failure path still prints a report and exits non-zero, so an unavailable
# device is a FAIL with a reason, never an unexplained bridge error.
RUNNER = r'''
import json, os, sys, xml.etree.ElementTree as ET

source, device, junit, select, test = sys.argv[1:6]
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device
expected = int(device)
report = {"command": "npu", "authorizedDevice": expected, "fatal": None}
report["junitPresent"] = False
report["tests"] = {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
report["pytestExit"] = None
report["phyBefore"] = None
report["phyAfter"] = None
report["npuAvailable"] = False
report["npuDeviceCount"] = 0
report["currentDevice"] = None
report["deviceName"] = None
report["tensorMovesToDevice"] = 0
report["numericalAssertions"] = 0
report["npuPeakMemoryBytes"] = 0
# Default to a failed synchronization. The block below sets it True only when the
# runtime confirms that all device work completed, so a missing value, a skipped
# block or a fatal error all keep it False and the Gate cannot pass on silence.
report["synchronizeOk"] = False
report["synchronizeError"] = None

saved = os.dup(1)
os.dup2(2, 1)
try:
    try:
        import pytest
        import torch
        import torch_npu

        moves = []
        original_move = torch.Tensor.npu

        def observed_move(self, *args, **kwargs):
            moves.append(1)
            return original_move(self, *args, **kwargs)

        torch.Tensor.npu = observed_move
        assertions = []
        original_assert = torch.testing.assert_close

        def observed_assert(*args, **kwargs):
            assertions.append(1)
            return original_assert(*args, **kwargs)

        torch.testing.assert_close = observed_assert

        report["npuAvailable"] = bool(torch.npu.is_available())
        report["npuDeviceCount"] = int(torch.npu.device_count())
        if report["npuAvailable"]:
            report["currentDevice"] = int(torch.npu.current_device())
            report["deviceName"] = torch.npu.get_device_name(0)

        import acl
        acl.init()
        report["driverDeviceCount"] = acl.rt.get_device_count()[0]
        report["phyBefore"] = acl.rt.get_phy_dev_id_by_logic_dev_id(0)[0]

        report["pytestExit"] = int(pytest.main(
            ["-q", "-p", "no:cacheprovider", "--junitxml=" + junit, "-k", select, test]))

        # A synchronization failure means the device work did not complete, so it
        # is a Gate failure with its original message preserved. No device reset, no
        # device substitution and no retry is attempted: this Gate runs on exactly
        # the authorized device and reports what that device did.
        try:
            torch.npu.synchronize()
        except BaseException as error:
            report["synchronizeOk"] = False
            report["synchronizeError"] = "%s: %s" % (type(error).__name__, error)
        else:
            report["synchronizeOk"] = True
        report["tensorMovesToDevice"] = len(moves)
        report["numericalAssertions"] = len(assertions)
        try:
            report["npuPeakMemoryBytes"] = int(torch.npu.max_memory_allocated(0))
        except Exception:
            pass
        report["phyAfter"] = acl.rt.get_phy_dev_id_by_logic_dev_id(0)[0]
    except BaseException as error:
        report["fatal"] = "%s: %s" % (type(error).__name__, error)
finally:
    os.dup2(saved, 1)
    os.close(saved)

if report["pytestExit"] is not None:
    try:
        root = ET.parse(junit).getroot()
    except Exception:
        pass
    else:
        stats = {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        report["junitPresent"] = True
        suites = [root] if root.tag == "testsuite" else list(root)
        for suite in suites:
            if suite.tag == "testsuite" and suite.get("tests"):
                stats["collected"] += int(suite.get("tests"))
            for case in suite.iter("testcase"):
                if not suite.get("tests"):
                    stats["collected"] += 1
                outcome = "passed"
                for entry in case:
                    if entry.tag == "skipped":
                        outcome = "skipped"
                    elif entry.tag in ("failure", "error"):
                        outcome = "failed" if entry.tag == "failure" else "errors"
                stats[outcome] += 1
        report["tests"] = stats

print(json.dumps(report, sort_keys=True))
# The runner's own exit status encodes the same conditions the bridge checks, so a
# synchronization failure or a fatal error cannot be read as success even if the
# report is ignored downstream. A pytest failure already makes this non-zero.
sys.exit(0 if report["fatal"] is None and report["synchronizeOk"] is True
         and report["pytestExit"] == 0 else 1)
'''


def container_script(source_root: str, timeout: int, device: str,
                     cann: Path | None, conda_sh: Path | None, conda_env: str | None,
                     conda_envs_path: Path | None = None) -> str:
    """Fixed container-side Bash: toolchain order, then one runner."""
    lines = [
        'set -o pipefail',
        f'fail() {{ echo "{PROG}: $*" >&2; exit 3; }}',
        'export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH',
        'export PYTHONDONTWRITEBYTECODE=1',
    ]
    # Every message that carries a machine-file or command-line value is
    # emitted as a single-quoted literal. `fail "text {value}"` would expand
    # `$(...)` or backticks inside the container, which would make a value read
    # from env.sh executable code and break the rule that the file is data.
    if cann is not None:
        lines.append(f'[ -r {shlex.quote(str(cann))} ] '
                     f'|| fail {shlex.quote(f"CANN setup not readable: {cann}")}')
        lines.append(f'. {shlex.quote(str(cann))}')
    if conda_sh is not None and conda_env:
        lines.append(f'[ -r {shlex.quote(str(conda_sh))} ] '
                     f'|| fail {shlex.quote(f"conda.sh not readable: {conda_sh}")}')
        # The directory comes from the machine file and from nowhere else. A
        # guessed `$HOME/.conda/envs` used to stand in for it, which meant the
        # Gate searched a directory the operator had not written down.
        if conda_envs_path is not None:
            # Concatenation, not a quoted value inside double quotes: the
            # latter would keep the quotes as literal characters in the path.
            lines.append(f'export CONDA_ENVS_PATH={shlex.quote(str(conda_envs_path))}'
                         f'"${{CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}}"')
        lines.append(f'. {shlex.quote(str(conda_sh))}')
        lines.append(f'conda activate {shlex.quote(conda_env)} '
                     f'|| fail {shlex.quote(f"conda activate {conda_env} failed")}')
    lines.append(f'cd {shlex.quote(source_root)} '
                 f'|| fail {shlex.quote(f"cannot enter {source_root}")}')
    lines.append(f'[ -f {shlex.quote(GATE_TEST)} ] || fail "Gate test file is missing: {GATE_TEST}"')
    lines.append('tmp=$(mktemp -d) || fail "cannot create a temporary directory"')
    lines.append(f'timeout --signal=TERM --kill-after={KILL_AFTER_SECONDS} {timeout} '
                 f'python3 - "$PWD" {shlex.quote(device)} "$tmp/junit.xml" '
                 f'{shlex.quote(GATE_SELECT)} {shlex.quote(GATE_TEST)} <<\'TILELANG_HIVM_NPU\'')
    lines.extend(RUNNER.strip('\n').splitlines())
    lines.append('TILELANG_HIVM_NPU')
    lines.append('rc=$?')
    lines.append('exit $rc')
    return '\n'.join(lines) + '\n'


def unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate report key')
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError('non-finite report number')


def read_report(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.startswith('{')]
    if not lines:
        raise BridgeError('container produced no NPU Gate report')
    try:
        report = json.loads(lines[-1], object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
    except ValueError as exc:
        raise BridgeError('container produced an unparsable NPU Gate report') from exc
    if not isinstance(report, dict) or report.get('command') != 'npu':
        raise BridgeError('container NPU Gate report has the wrong shape')
    return report


def observations(report: dict, device: str) -> dict:
    """Only what the container measured. Missing data is false, never assumed."""
    tests = report.get('tests') if isinstance(report.get('tests'), dict) else {}
    keys = ('collected', 'passed', 'failed', 'errors', 'skipped')
    valid_counts = all(type(tests.get(key)) is int and tests[key] >= 0 for key in keys)
    collected, passed, failed, errors, skipped = (
        [tests[key] for key in keys] if valid_counts else [0] * len(keys))
    valid_counts = valid_counts and collected == passed + failed + errors + skipped
    expected = int(device)
    moves = report.get('tensorMovesToDevice')
    assertions = report.get('numericalAssertions')
    peak = report.get('npuPeakMemoryBytes')
    result = {
        'deviceAuthorized': True,
        'junitPresent': report.get('junitPresent') is True,
        'testsCollected': collected,
        'testsPassed': passed,
        'testsFailed': failed + errors,
        'testsSkipped': skipped,
        'testsCollectedPositive': valid_counts and collected > 0,
        'notAllSkipped': valid_counts and collected > 0 and skipped < collected,
        'allPassed': valid_counts and passed > 0 and failed == 0 and errors == 0
        and report.get('fatal') is None,
        'pytestExitZero': type(report.get('pytestExit')) is int and report['pytestExit'] == 0,
        # The tests really moved tensors to the device and really compared
        # numbers; a stub that never calls the kernel cannot satisfy these.
        'tensorMovesToDevice': moves if type(moves) is int and moves >= 0 else 0,
        'numericalAssertions': assertions if type(assertions) is int and assertions >= 0 else 0,
        'npuPeakMemoryBytes': peak if type(peak) is int and peak >= 0 else 0,
        'npuAvailable': report.get('npuAvailable') is True,
        # Synchronization is a real device operation, not bookkeeping: when the
        # runtime cannot confirm that device work completed, the run is a failure
        # and the original message is carried into the receipt for review.
        'synchronizeOk': report.get('synchronizeOk') is True,
        'synchronizeError': report.get('synchronizeError')
        if isinstance(report.get('synchronizeError'), str) else None,
        'deviceCountOne': type(report.get('npuDeviceCount')) is int
        and report['npuDeviceCount'] == 1,
        'driverDeviceCountMatches': type(report.get('driverDeviceCount')) is int
        and report['driverDeviceCount'] == 1,
        # The driver's own logical-to-physical mapping must equal the authorized
        # id both before and after the run; this is not a value the test sets.
        'deviceMatchesAuthorized': type(report.get('phyBefore')) is int
        and type(report.get('phyAfter')) is int
        and report['phyBefore'] == expected and report['phyAfter'] == expected,
    }
    result['tensorMovedToDevice'] = result['tensorMovesToDevice'] > 0
    result['numericalAssertionsObserved'] = result['numericalAssertions'] > 0
    result['npuDeviceUsed'] = result['npuPeakMemoryBytes'] > 0
    return result


def passed(result: dict) -> bool:
    """The same conditions the manifest declares, so the exit code cannot lie."""
    return all(result[key] is True for key in (
        'testsCollectedPositive', 'notAllSkipped', 'allPassed', 'pytestExitZero',
        'tensorMovedToDevice', 'numericalAssertionsObserved', 'npuDeviceUsed',
        'npuAvailable', 'deviceMatchesAuthorized', 'synchronizeOk', 'junitPresent',
        'deviceCountOne', 'driverDeviceCountMatches'))


def require_conda_pair(conda_sh: Path | None, conda_env: str | None) -> None:
    """CONDA_SH and CONDA_ENV are both-or-neither; a half configuration fails closed.

    Silently ignoring one half would run the tests against a different
    interpreter than the one pytest was configured for, which is exactly the
    failure the Adapter is supposed to prevent. The container script below also
    reads `conda_sh is not None and conda_env`, so without this check a half
    configuration would drop Conda on the container side and still look fine.
    """
    if (conda_sh is None) != (conda_env is None):
        raise BridgeError('TILELANG_HIVM_CONDA_SH and TILELANG_HIVM_CONDA_ENV must be provided '
                          'together; set both or neither')


def check_config() -> int:
    """Report whether this machine is configured, without entering the container.

    Preflight runs this before the device Gate starts. The device id is part of
    the check: it authorizes one run, and a run without it must be refused by
    the Host rather than by a Gate that has already begun.
    """
    container, home, identity_file = container_identity()
    device = device_id()
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError(f'TILELANG_HIVM_CONDA_ENV is not a plain environment name: {conda_env!r}')
    # Occupancy is a Host fact, so it is checked here, before the Recipe is
    # dispatched at all. The container's own `npu-smi` is not used: inside the
    # image it fails to load its libraries and would report nothing.
    npu_id = require_device_free(device)
    print(f'{PROG}: configuration ready: identity file {identity_file} container={container} '
          f'home={home} device={device} npuId={npu_id} occupancy=free', file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    if list(argv) == ['--check-config']:
        return check_config()
    host_root = Path.cwd()
    container, home, identity_file = container_identity()

    device = device_id()
    timeout = parse_argv(list(argv))
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError(f'TILELANG_HIVM_CONDA_ENV is not a plain environment name: {conda_env!r}')

    # Occupancy is decided before anything touches Docker. The configuration
    # check already probed the card, but that ran earlier and another job may
    # have taken it since; this second read-only probe is the one that guards
    # the dispatch. A busy or unreadable card stops the Gate with no `docker
    # exec` at all, so no test process is ever started next to another user's.
    require_device_free(device)

    mapped = map_host_to_container(host_root, container_mounts(container), container)
    declared = env_path('TILELANG_HIVM_CONTAINER_SOURCE')
    if declared is not None and os.path.realpath(str(declared)) != os.path.realpath(str(mapped)):
        raise BridgeError(f'TILELANG_HIVM_CONTAINER_SOURCE {declared} does not match the bind-mount '
                          f'mapping {mapped} of host source root {host_root}')
    container_root = str(declared if declared is not None else mapped)
    prefix = docker_exec_prefix(container, home, container_root)
    verify_same_tree(host_root, prefix, container_root)
    print(f'{PROG}: identity file {identity_file}', file=sys.stderr)
    print(f'{PROG}: container={container} user=root source={container_root} '
          f'timeout={timeout}s device={device} test={GATE_TEST} select={GATE_SELECT}', file=sys.stderr)
    script = container_script(container_root, timeout, device, cann, conda_sh, conda_env,
                              conda_envs_path_from_env_file())
    completed = subprocess.run([*prefix, 'bash', '-c', script], capture_output=True, text=True)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    report = read_report(completed.stdout)
    result = observations(report, device)
    if report.get('fatal'):
        result['fatal'] = True
        result['fatalReason'] = str(report['fatal'])
    print(json.dumps(result, sort_keys=True))
    # A failure inside the container keeps its non-zero exit code; the declared
    # observations are false as well, so an unavailable device, an all-skipped
    # suite, a zero-test run and a timeout can never be a PASS.
    return 0 if (passed(result) and completed.returncode == 0) else (completed.returncode or 1)


if __name__ == '__main__':
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BridgeError as error:
        print(f'{PROG}: {error.message}', file=sys.stderr)
        raise SystemExit(BridgeError.EXIT)
