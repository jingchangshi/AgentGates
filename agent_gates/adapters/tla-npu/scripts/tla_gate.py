#!/usr/bin/env python3
"""Run the TLA (Triton Language for Ascend) end-to-end Gate in its container.

Scope
-----

This Adapter runs exactly one script, test_tla.py from the pinned TLA package
directory, and it runs it on a real Ascend device. That script is a 3-in-1
end-to-end check of the tla namespace shipped by the vendor Triton wheel:

  * it builds a cast_back kernel with tla.jit, using tla.alloc_buffer,
    tla.view, tla.copy, tla.scope.simd() and tla.simd.*;
  * it launches it on an NPU with a real fp8 input and checks the result
    against a Torch reference (CAST_BACK OK);
  * it runs the same semantics as a plain triton.jit kernel and checks that
    too (TRITON_CAST_BACK OK).

A PASS therefore proves that this machine can compile and execute a real
Triton-on-Ascend kernel through the TLA namespace and get correct numbers, and
that the underlying community Triton backend still works alongside it. It does
NOT prove the whole TLA surface, and it is not a compiler test suite.

The script is edited by nobody here
-----------------------------------

The Gate does not modify, patch or regenerate the test file. It runs the bytes
that are on disk. If the bytes are not valid Python the Gate fails and says so,
because a test that cannot be imported is not a test that passed.

Device selection
----------------

TLA_NPU_DEVICE_ID must name the device; this Adapter never discovers, chooses or
substitutes one. Silently picking a different card would run a test on hardware
nobody asked for, so the id is explicit here and AgentGates supplies it from the
local configuration.

Sharing is normal. On this class of machine a card is shared by every user, and a
process holding a device does not reserve it: refusing to run because someone
else is on the card would block legitimate work for a reason unrelated to the code
under test. Occupancy is therefore REPORTED (occupancy=free or shared(n)) rather
than enforced.

An operator who needs an uncontended card sets TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE=1
(AgentGates forwards devices.require_exclusive_device). A held card then stops
the Gate as BLOCKED_ENV, which is honest: the test never ran.

Device use is proved by counters measured in the process that ran the test:
tensors placed on the device, a positive peak allocation, and kernel images in a
fresh cache. The Gate does NOT claim to re-derive which physical board ran.
Board identity is not observable from inside this container (its npu-smi cannot
load libc_sec.so), torch reports the same UUID for both cards on this host, and
acl.rt.get_phy_dev_id_by_logic_dev_id answers from the host physical table
regardless of the selection - so none of those is gated. The visible-device echo
and the device UUID/name are recorded in the receipt for a reader, and are
deliberately excluded from passed(), because a value this Gate supplied and then
read back cannot fail for a hardware reason.
A failure to synchronize the device is reported with its original message and
fails the Gate. No device reset, no device substitution and no automatic retry
is attempted.

Environment variables (all read at run time):

  TLA_NPU_DEVICE_ID               required, the NPU device id to use.
  TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE optional; when truthy a card held by another
                                  process is BLOCKED_ENV instead of shared.
  TLA_NPU_CONTAINER_NAME          optional, name of a running build container.
                                  The machine file is the only source. A shell
                                  export has no effect.
  TLA_NPU_CONTAINER_HOME          optional, HOME inside the container; read from
                                  the same machine file
  TLA_NPU_CONTAINER_USER          ignored; the device Gate runs as root, because
                                  the driver refuses a non-root opener once any
                                  namespace holds the card (uda_occupy_dev_by_ns)
  TLA_NPU_CONTAINER_SOURCE        optional, explicit container path of the
                                  source root; when set it must match the
                                  bind-mount mapping
  TLA_NPU_GATE_TIMEOUT            optional, container-side timeout in seconds;
                                  used ONLY when no --timeout argument is passed

Toolchain configuration comes from a file, not from the launching shell:

  TLA_NPU_ENV_FILE                path of the env.sh that holds CANN/Conda;
                                  default the checkout's own adapters/tla-npu/env.sh

The file contains plain NAME=value assignments for the keys listed in
TOOLCHAIN_KEYS, and nothing else is read from it. It is the single source of
truth for CANN and Conda.

Why the test lives outside a source checkout
--------------------------------------------

The script under test is shipped in a package directory (~/Packages/tla-...), not
inside a Git checkout. The Gate therefore takes --source to mean "the directory
holding the test", derives its container path from the bind mounts, and proves
both sides see the same directory by comparing its device/inode pair before
running anything. That is the same guarantee the sibling Adapters give for a
compiler checkout: the bytes tested are the bytes on disk.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PROG = 'tla-npu-gate'
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
# Grace between timeout sending TERM and the container-side SIGKILL; the outer
# Harness Recipe timeout must still exceed --timeout + KILL_AFTER_SECONDS.
KILL_AFTER_SECONDS = 15
# Read-only Host probe of the authorized card. npu-smi info answers in well under
# a second; a hung driver must block the Gate rather than delay it.
NPU_SMI = 'npu-smi'
PROBE_TIMEOUT_SECONDS = 60

# The Gate's test file: fixed in this digest-pinned file, and never read from the
# environment. It lives in the TEST directory (TLA_NPU_TEST_DIR), which is not the
# source root: the source root is the triton-ascend checkout whose TLA
# implementation is under test, and the test exercising it is kept separately.
GATE_TEST = 'test_tla.py'
# Markers the test prints on success. Their absence is a failure, never a pass
# by silence.
CAST_BACK_MARK = 'CAST_BACK OK'
TRITON_MARK = 'TRITON_CAST_BACK OK'

# Keys the toolchain file may define. Anything else is an error: a typo must not
# silently leave a component unconfigured.
TOOLCHAIN_KEYS = (
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
    # The directory holding the Gate test, kept separate from the source root.
    'TLA_NPU_TEST_DIR',
    # A dedicated, stable cache directory. TRITON_CACHE_DIR must not point inside
    # either input tree: a JIT cache is written to, and the whole point of the
    # source and test roots is that a run never writes into the bytes it tests.
    'TLA_NPU_CACHE_DIR',
    # The sibling Adapters read their build subdirectory from their file so the
    # engine leases the directory they write. This Gate writes no build tree, but
    # the engine still resolves a build directory for every adapter, so the key
    # is accepted and defaults to 'build'.
    'TLA_NPU_BUILD_SUBDIR',
)
# The keys this Gate needs for the container it enters. The user key is excluded
# from the requirement on purpose: the device Gate runs as root.
CONTAINER_KEYS = ('TLA_NPU_CONTAINER_NAME', 'TLA_NPU_CONTAINER_HOME')
BUILD_KEY = 'TLA_NPU_BUILD_SUBDIR'


class BridgeError(Exception):
    """A configuration, mapping or launch failure, not a Gate result."""

    EXIT = 2

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class OccupancyBlocked(BridgeError):
    """The card is held by a job this Gate did not start.

    This is an environment prerequisite that is missing, not a test that ran and
    failed: nothing was dispatched and no assertion was evaluated. It therefore
    exits with its own code so the runner can record BLOCKED_ENV rather than
    FAIL. The code is 75 (sysexits EX_TEMPFAIL), chosen because pytest's own
    statuses are 0-5 and the container script uses 3, so this value cannot be
    produced by a test run that failed.
    """

    EXIT = 75


def env_value(name: str, *, required: bool = False, default: str | None = None) -> str | None:
    value = os.environ.get(name, '')
    value = value.strip() if isinstance(value, str) else ''
    if not value:
        if required:
            raise BridgeError(f'missing required environment variable {name}')
        return default
    return value


# A conservative NAME=value grammar. Values may be bare, single- or
# double-quoted. No expansion, no command substitution, no line continuations:
# the file is data, so reading it never executes Host code.
_ASSIGNMENT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')


def _parse_env_file(text: str, path: Path) -> dict[str, str]:
    """Parse the strict NAME=value toolchain file; never executes it."""
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


ADAPTER_DIR = Path(__file__).resolve().parent.parent
CHECKOUT_MARKER = Path('adapters') / 'tla-npu' / 'env.example.sh'


def checkout_env_file() -> Path | None:
    """The in-repository machine file, when this Pack belongs to a checkout."""
    for ancestor in (ADAPTER_DIR, *ADAPTER_DIR.parents):
        if (ancestor / CHECKOUT_MARKER).is_file():
            return ancestor / CHECKOUT_MARKER.parent / 'env.sh'
    return None


def default_env_file() -> Path | None:
    """The machine file the checkout owns, or None outside a checkout.

    There is no second location. A machine that keeps its file elsewhere names it
    in TLA_NPU_ENV_FILE; guessing a home-directory path silently picked up a file
    nobody was reviewing.
    """
    return checkout_env_file()


def resolved_env_file() -> Path | None:
    """The machine file to read, or None when this machine has none."""
    explicit = env_value('TLA_NPU_ENV_FILE')
    if not explicit:
        return default_env_file()
    path = Path(explicit).expanduser()
    if not path.is_absolute():
        raise BridgeError(f'TLA_NPU_ENV_FILE must be an absolute path: {explicit!r}')
    if not path.exists():
        raise BridgeError(f'TLA_NPU_ENV_FILE does not exist: {path}')
    return path


def env_file_values() -> dict[str, str]:
    """Every key the machine file defines, parsed once for the whole Gate."""
    path = resolved_env_file()
    if path is None or not path.exists():
        return {}
    if not path.is_file():
        raise BridgeError(f'TLA_NPU_ENV_FILE is not a regular file: {path}')
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc
    return _parse_env_file(text, path)


def machine_value(name: str) -> str | None:
    """One machine value, read from the machine file and nowhere else.

    A contradicting export is refused, and an export the file does not define is
    refused too: a value that looks like configuration but has no effect is worse
    than an error.
    """
    values = env_file_values()
    defined = name in values
    from_file = values.get(name, '').strip()
    exported = env_value(name)
    if defined and from_file:
        if exported and exported.strip() != from_file:
            raise BridgeError(
                f'{name} is {exported!r} in the launching environment but {from_file!r} in '
                f'{resolved_env_file()}; machine values are read from that file only. '
                f'Remove the export or edit the file, so one run cannot use two values.')
        return from_file
    if exported and exported.strip():
        path = resolved_env_file()
        where = str(path) if path is not None else 'no machine file was found'
        raise BridgeError(
            f'{name} is exported as {exported!r}, but machine values are read only from '
            f'{where}. Define it in that file instead of the launching environment.')
    if defined:
        raise BridgeError(f'{resolved_env_file()}: {name} is set but empty')
    return None


def toolchain_from_env_file() -> tuple[Path | None, Path | None, str | None]:
    """Read CANN/Conda from the configured file. The file is authoritative."""
    path = resolved_env_file()
    if path is None or not path.exists():
        return None, None, None
    if not path.is_file():
        raise BridgeError(f'TLA_NPU_ENV_FILE is not a regular file: {path}')
    values = env_file_values()

    def as_path(name: str, raw: str) -> Path | None:
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            raise BridgeError(f'{path}: {name} must be an absolute path: {raw!r}')
        if '\n' in raw or '\x00' in raw:
            raise BridgeError(f'{path}: {name} contains an illegal character')
        return candidate

    cann = as_path('TLA_NPU_CANN_SETUP', values.get('TLA_NPU_CANN_SETUP', ''))
    conda_sh = as_path('TLA_NPU_CONDA_SH', values.get('TLA_NPU_CONDA_SH', ''))
    conda_env = values.get('TLA_NPU_CONDA_ENV', '') or None
    print(f'{PROG}: toolchain file {path}', file=sys.stderr)
    return cann, conda_sh, conda_env


def conda_envs_path_from_env_file() -> Path | None:
    """The extra Conda environment directory, when the machine file names one.

    A Conda installation searches only the environments under its own prefix. A
    user's environments normally live in ~/.conda/envs, so without this value
    conda activate fails with EnvironmentNameNotFound even though
    conda env list shows the environment.
    """
    values = env_file_values()
    raw = values.get('TLA_NPU_CONDA_ENVS_PATH', '')
    if not raw:
        return None
    if not Path(raw).is_absolute():
        raise BridgeError(f'{resolved_env_file()}: TLA_NPU_CONDA_ENVS_PATH must be an '
                          f'absolute path: {raw!r}')
    return Path(raw)


def test_dir_from_env_file() -> Path:
    """The directory holding the Gate test, from the machine file.

    Deliberately not the source root. The source root is the triton-ascend
    checkout under test; the test that exercises its TLA implementation lives in
    its own directory. Keeping the two apart lets a PASS name both the
    implementation that ran and the bytes that exercised it, instead of
    conflating one directory with both roles.
    """
    value = machine_value('TLA_NPU_TEST_DIR')
    if value is None:
        raise BridgeError('TLA_NPU_TEST_DIR is required; it names the directory holding '
                          f'{GATE_TEST} (the source root is the checkout under test)')
    path = Path(value)
    if not path.is_absolute():
        raise BridgeError(f'TLA_NPU_TEST_DIR must be an absolute path: {value}')
    return path


def cache_dir_from_env_file() -> Path | None:
    """A dedicated TRITON_CACHE_DIR, from the machine file if it names one.

    A JIT cache is written to, so it must not land in either input tree. That is
    not hypothetical: Triton wrote a cache/ tree holding npu_utils.so into the
    former test package directory and moved that directory's provenance hash
    between two runs of the same Gate.
    """
    value = machine_value('TLA_NPU_CACHE_DIR')
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        raise BridgeError(f'TLA_NPU_CACHE_DIR must be an absolute path: {value}')
    return path


def declared_container_source() -> Path | None:
    """The container source root the machine file names, if it names one.

    It confirms a path; it cannot invent one. The bind-mount mapping is still
    derived from docker inspect and compared, so a value that disagrees with the
    mounts is refused rather than trusted.
    """
    raw = machine_value('TLA_NPU_CONTAINER_SOURCE') or ''
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise BridgeError(f'{resolved_env_file()}: TLA_NPU_CONTAINER_SOURCE must be an '
                          f'absolute container path: {raw!r}')
    if chr(10) in raw or chr(0) in raw:
        raise BridgeError('TLA_NPU_CONTAINER_SOURCE contains an illegal character')
    return candidate


def container_identity() -> tuple[str, str, Path | None]:
    """The container this Gate enters, read from the machine file.

    The launching shell is not consulted: a stale export used to be the only
    source, and it could name a stopped container while the machine file named
    the live one.
    """
    path = resolved_env_file()
    values = env_file_values()
    missing = [name for name in CONTAINER_KEYS if not values.get(name)]
    if missing:
        where = str(path) if path is not None else 'no machine file was found'
        raise BridgeError(
            f'{where} does not define {", ".join(missing)}. Container identity is read '
            f'from that file only; copy env.example.sh to env.sh beside it, set '
            f'TLA_NPU_CONTAINER_NAME and TLA_NPU_CONTAINER_HOME to the container '
            f'docker/config.env created, and run the Gate again.')
    container = values['TLA_NPU_CONTAINER_NAME'].strip()
    home = values['TLA_NPU_CONTAINER_HOME'].strip()
    if not SAFE_NAME.fullmatch(container):
        raise BridgeError('container name must be a plain name, not an option-like string')
    if not home.startswith('/'):
        raise BridgeError(f'TLA_NPU_CONTAINER_HOME must be an absolute container path: {home!r}')
    if any('\n' in value or '\x00' in value for value in (container, home)):
        raise BridgeError('container identity values must not contain a newline or NUL')
    return container, home, path


def device_id() -> str:
    """The authorized device. There is no discovery and no default."""
    value = env_value('TLA_NPU_DEVICE_ID', required=True) or ''
    if not value.isdigit() or int(value) >= 64:
        raise BridgeError(f'TLA_NPU_DEVICE_ID must be a plain device number: {value!r}')
    return str(int(value))


def npu_smi(*args: str) -> str:
    """Run one read-only npu-smi info query on the Host.

    Only info subcommands are ever issued: the probe must not reset, set or
    otherwise change a device. Every way of not getting an answer - a missing
    binary, a non-zero exit, a timeout, a signal - is an OccupancyBlocked,
    because unknown is not free.
    """
    try:
        done = subprocess.run([NPU_SMI, 'info', *args], capture_output=True, text=True,
                              timeout=PROBE_TIMEOUT_SECONDS)
    except FileNotFoundError as error:
        raise OccupancyBlocked(f'cannot determine NPU occupancy: {NPU_SMI} is not installed on '
                               f'the Host ({error})') from error
    except subprocess.TimeoutExpired as error:
        raise OccupancyBlocked(f'cannot determine NPU occupancy: {NPU_SMI} info '
                               f'{" ".join(args)} did not answer within '
                               f'{PROBE_TIMEOUT_SECONDS}s') from error
    except OSError as error:
        raise OccupancyBlocked(f'cannot determine NPU occupancy: {NPU_SMI} could not be '
                               f'started: {error}') from error
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise OccupancyBlocked(f'cannot determine NPU occupancy: {NPU_SMI} info '
                               f'{" ".join(args)} exited {done.returncode}: '
                               f'{detail[-1] if detail else "no output"}')
    return done.stdout


def npu_id_of_physical(device: str) -> str:
    """The npu-smi NPU ID whose Chip Phy-ID is the authorized device.

    The Gate authorizes a physical id: the container side asserts the driver maps
    its logical device onto exactly that id. npu-smi numbers cards by NPU ID,
    which is not guaranteed to equal the physical id, so the mapping table is
    consulted instead of assuming the two namespaces coincide. A device missing
    from the table, or claimed by more than one row, cannot be probed and is
    refused rather than guessed.

    The table layout is: NPU ID, Slot ID, Chip ID, Chip Phy-ID, Chip Name. A line
    whose first four fields are not all numbers is a header and is skipped; a
    table this Gate cannot read yields no match, which is refused, never treated
    as a free card.
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
        raise OccupancyBlocked(f'cannot determine NPU occupancy: no device in the npu-smi '
                               f'mapping table reports physical id {device}')
    if len(unique) > 1:
        raise OccupancyBlocked(f'cannot determine NPU occupancy: physical id {device} is '
                               f'claimed by several npu-smi NPU IDs {unique}')
    return unique[0]


def processes_on(npu_id: str) -> list[tuple[str, str]]:
    """The (pid, name) pairs npu-smi reports for one NPU ID.

    The answer must be structurally recognisable in full: it has to name the
    queried NPU ID, and every process row it prints has to match the documented
    layout, or the table has to say in the driver's own words that the device is
    idle. A row this Gate cannot read means the driver's format changed, and a
    format this Gate cannot read is unknown occupancy, never a free card.
    """
    text = npu_smi('-t', 'proc-mem', '-i', npu_id)
    if not re.search(rf'^\s*NPU ID\s*:\s*{re.escape(npu_id)}\s*$', text, re.MULTILINE):
        raise OccupancyBlocked(f'cannot determine NPU occupancy: {NPU_SMI} info -t proc-mem '
                               f'-i {npu_id} did not report NPU ID {npu_id}')
    idle = len(re.findall(r'^\s*No process in device\.?\s*$', text, re.MULTILINE))
    pattern = re.compile(r'Process id:(\d+)\s+Process name:(\S+)\s+Process memory\(MB\):(\d+)')
    rows = [(match.group(1), match.group(2)) for match in pattern.finditer(text)]
    announced = len(re.findall(r'Process id:', text))
    if rows and idle:
        raise OccupancyBlocked(f'cannot determine NPU occupancy: {NPU_SMI} info -t proc-mem '
                               f'-i {npu_id} reported both processes and an idle device')
    if len(rows) != announced:
        raise OccupancyBlocked(f'cannot determine NPU occupancy: {NPU_SMI} info -t proc-mem '
                               f'-i {npu_id} reported {announced} process row(s) but only '
                               f'{len(rows)} matched the known layout')
    if not rows and not idle:
        raise OccupancyBlocked(f'cannot determine NPU occupancy: {NPU_SMI} info -t proc-mem '
                               f'-i {npu_id} neither listed a process nor said the device is '
                               f'idle')
    return rows


def exclusive_requested() -> bool:
    """Whether this run demands a card no other process is using."""
    value = env_value('TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE', default='') or ''
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def device_occupancy(device: str) -> tuple[str, list]:
    """The npu-smi id for a physical device, and the processes holding it."""
    npu_id = npu_id_of_physical(device)
    return npu_id, processes_on(npu_id)


def require_device_free(device: str) -> str:
    """Report the card's occupancy; refuse only when a quiet card was demanded.

    Sharing an Ascend NPU is normal operation. Another job running on a device
    does not make that device unusable for this Gate, and refusing on that basis
    would block legitimate work for a reason that has nothing to do with the code
    under test. So the DEFAULT is to proceed, and the occupancy is reported so a
    reader can tell a quiet run from a shared one.

    Only the count and the npu-smi identity are reported: other users' process ids
    and command lines are not copied into the receipt.
    """
    npu_id, holders = device_occupancy(device)
    if holders and exclusive_requested():
        raise OccupancyBlocked(f'NPU device {device} (npu-smi NPU ID {npu_id}) is already in '
                               f'use by {len(holders)} process(es) and this run requires an '
                               f'exclusive card (TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE is set)')
    return npu_id


def parse_argv(argv: list[str]) -> int:
    """Optional --timeout SECONDS; the manifest always passes one."""
    value = None
    if argv[:1] == ['--timeout']:
        if len(argv) != 2:
            raise BridgeError('--timeout requires exactly one value and no other argument')
        value = argv[1]
    elif argv:
        raise BridgeError(f'unexpected arguments: {argv!r}')
    else:
        value = env_value('TLA_NPU_GATE_TIMEOUT', default='1500')
    if value is None or not value.isdigit() or int(value) < 1:
        raise BridgeError(f'gate timeout must be a positive integer number of seconds: {value!r}')
    return int(value)


def docker_exec_prefix(container: str, home: str, workdir: str) -> list[str]:
    """Run the container side as root.

    The Ascend driver holds davinci0/davinci1 per user namespace
    (uda_occupy_dev_by_ns). Once any namespace holds the card, a non-root opener
    is refused with EBUSY even when it is a member of the device group, so an
    ordinary container user cannot reliably reach the device. root is not subject
    to that check; the device Gate therefore runs as root.
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
    """dev:ino of path inside the container, using stat (not Python)."""
    expression = 'stat -c "%d:%i" ' + shlex.quote(path)
    done = subprocess.run([*prefix, 'bash', '-c', expression], capture_output=True, text=True)
    if done.returncode != 0:
        raise BridgeError(f'container cannot stat {path}: '
                          f'{done.stderr.strip() or done.stdout.strip()}')
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

# Container-side runner. It sets the visible device *before* importing the runtime,
# observes the driver's logical-to-physical mapping before and after the test,
# instruments the move-to-device call, runs the test as a child process so the
# markers it prints can be read, then prints one raw report on standard output.
# The test's own stderr is forwarded; its stdout is carried in the report. Every
# failure path still prints a report and exits non-zero, so an unavailable device
# is a FAIL with a reason, never an unexplained bridge error.
RUNNER = r'''
import json, os, sys, tempfile

source, device, test = sys.argv[1:4]
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = device
expected = int(device)
report = {'command': 'tla', 'authorizedDevice': expected, 'fatal': None}
report['npuAvailable'] = False
report['npuDeviceCount'] = 0
report['currentDevice'] = None
report['deviceName'] = None
report['visibleDevices'] = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
report['scriptExit'] = None
report['scriptStdout'] = ''
report['artifacts'] = {'object': 0, 'npubin': 0}
report['tensorMovesToDevice'] = 0
report['npuPeakMemoryBytes'] = 0
report['deviceUuid'] = None
# Default to a failed synchronization. The child sets it True only when the
# runtime confirms that all device work completed, so a missing value, a skipped
# block or a fatal error all keep it False and the Gate cannot pass on silence.
report['synchronizeOk'] = False
report['synchronizeError'] = None

WATCHER = "# Instrumented wrapper. The test runs in THIS process so the counters it\n# produces describe the run that actually happened. A parent-process counter\n# would be structurally zero: torch.npu.max_memory_allocated and a patched\n# torch.Tensor.npu are both process-local.\nimport json, runpy, sys, traceback\n\ntest, out = sys.argv[1:3]\nimport torch\nimport torch_npu\n\n# Count the ways this test actually puts a tensor on the device: the tensor\n# factory device= keyword and the Tensor.npu() call. Counting a call the test\n# never makes would report zero on a perfect run, so every factory that accepts\n# device= is wrapped.\nmoves = []\nfor factory_name in ('randn', 'rand', 'zeros', 'ones', 'full', 'empty'):\n    original_factory = getattr(torch, factory_name)\n\n    def make_observed(original):\n        def observed_factory(*args, **kwargs):\n            device = kwargs.get('device')\n            if isinstance(device, str) and device.split(':')[0] == 'npu':\n                moves.append(1)\n            return original(*args, **kwargs)\n        return observed_factory\n\n    setattr(torch, factory_name, make_observed(original_factory))\n\noriginal_move = torch.Tensor.npu\n\n\ndef observed_move(self, *args, **kwargs):\n    moves.append(1)\n    return original_move(self, *args, **kwargs)\n\n\ntorch.Tensor.npu = observed_move\nreport = {'moves': 0, 'peak': 0, 'exit': 1, 'synchronizeOk': False,\n          'synchronizeError': None, 'uuid': None}\ntry:\n    runpy.run_path(test, run_name='__main__')\nexcept SystemExit as error:\n    code = error.code\n    report['exit'] = 0 if not code else (code if isinstance(code, int) else 1)\nexcept BaseException:\n    traceback.print_exc()\n    report['exit'] = 1\nelse:\n    report['exit'] = 0\nfinally:\n    report['moves'] = len(moves)\n    try:\n        torch.npu.synchronize()\n    except BaseException as error:\n        report['synchronizeError'] = '%s: %s' % (type(error).__name__, error)\n    else:\n        report['synchronizeOk'] = True\n    try:\n        report['peak'] = int(torch.npu.max_memory_allocated(0))\n    except Exception:\n        pass\n    try:\n        report['uuid'] = str(torch.npu.get_device_properties(0).uuid)\n    except Exception:\n        pass\n    with open(out, 'w') as handle:\n        json.dump(report, handle)\n    sys.exit(report['exit'] if isinstance(report['exit'], int) else 1)"
wait_marker = 1
saved = os.dup(1)
os.dup2(2, 1)
try:
    try:
        import subprocess
        import torch
        import torch_npu

        report['npuAvailable'] = bool(torch.npu.is_available())
        report['npuDeviceCount'] = int(torch.npu.device_count())

        # The TLA implementation the test actually imports. A PASS must name the
        # code that produced it, so this is hashed rather than read from a version
        # string: the wheel and the checkout can disagree, and a wheel reinstalled
        # at the same version with different contents would otherwise let an older
        # PASS verify as current.
        try:
            import hashlib
            import triton
            import triton.language.extra.tla as _tla
            _tla_dir = os.path.dirname(os.path.abspath(_tla.__file__))
            _digest = hashlib.sha256()
            _files = 0
            for _root, _dirs, _names in os.walk(_tla_dir):
                _dirs.sort()
                for _name in sorted(_names):
                    if not _name.endswith('.py'):
                        continue
                    _full = os.path.join(_root, _name)
                    _rel = os.path.relpath(_full, _tla_dir)
                    _digest.update(_rel.encode() + b'\0')
                    with open(_full, 'rb') as _handle:
                        _digest.update(_handle.read() + b'\0')
                    _files += 1
            report['tlaImplementationSha256'] = _digest.hexdigest()
            report['tlaImplementationFileCount'] = _files
            report['tlaImplementationPath'] = _tla_dir
            report['tritonVersion'] = triton.__version__
            report['tlaHasDma'] = hasattr(_tla, 'dma')
        except BaseException as _error:
            # Recorded, not gated: the counters below decide the verdict. A missing
            # identity must still be visible in the receipt rather than silently
            # absent, so the failure text is carried.
            report['tlaImplementationSha256'] = None
            report['tlaImplementationError'] = '%s: %s' % (
                type(_error).__name__, _error)
        if report['npuAvailable']:
            report['currentDevice'] = int(torch.npu.current_device())
            report['deviceName'] = torch.npu.get_device_name(0)

        # The instrumented child runs the test in its own process so the
        # move-to-device counter and the peak-memory reading describe the run
        # that actually happened. A separate child also keeps a failed import in
        # the test from leaking state into this report.
        work = tempfile.mkdtemp(prefix='tla-gate-')
        wrapper = os.path.join(work, 'run_tla.py')
        child_report = os.path.join(work, 'child.json')
        with open(wrapper, 'w') as handle:
            handle.write(WATCHER)
        # Run from the scratch directory, NOT from the source tree. CANN writes a
        # crash-dump directory (extra-info/) into the process working directory
        # when a kernel fault occurs, and the source tree is read-only by contract:
        # a run must never deposit artifacts into the bytes it is testing. The
        # test file is addressed by absolute path, so its own directory does not
        # need to be the working directory. ASCEND_WORK_PATH pins the dump
        # location explicitly for the same reason.
        child_env = dict(os.environ, ASCEND_WORK_PATH=work)
        completed = subprocess.run(
            [sys.executable, '-B', wrapper, os.path.abspath(test), child_report],
            cwd=work, env=child_env, capture_output=True, text=True)
        report['scriptStdout'] = completed.stdout
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        child = None
        try:
            with open(child_report) as handle:
                child = json.load(handle)
        except Exception as error:
            report['fatal'] = 'the instrumented run produced no report: %s' % error
        if isinstance(child, dict):
            report['scriptExit'] = child.get('exit')
            report['tensorMovesToDevice'] = child.get('moves')
            report['npuPeakMemoryBytes'] = child.get('peak')
            report['synchronizeOk'] = child.get('synchronizeOk') is True
            report['synchronizeError'] = child.get('synchronizeError')
            report['deviceUuid'] = child.get('uuid')
        else:
            report['scriptExit'] = int(completed.returncode)

        # Compiler products. The Ascend backend emits an .npubin image per
        # compiled kernel; a run that only interpreted produces none. The
        # container script points TRITON_CACHE_DIR at a fresh directory, so
        # these counts belong to this run alone.
        root = os.environ.get('TRITON_CACHE_DIR', '')
        if root and os.path.isdir(root):
            for directory, _dirs, files in os.walk(root):
                for name in files:
                    if name.endswith('.npubin'):
                        report['artifacts']['npubin'] += 1
                    elif name.endswith('.o'):
                        report['artifacts']['object'] += 1
    except BaseException as error:
        report['fatal'] = '%s: %s' % (type(error).__name__, error)
finally:
    os.dup2(saved, 1)
    os.close(saved)

print(json.dumps(report, sort_keys=True))
sys.exit(0 if report['fatal'] is None and report['synchronizeOk'] is True
         and report['scriptExit'] == 0 else 1)
'''


def container_script(test_dir: str, source_root: str, timeout: int, device: str,
                     cann: Path | None, conda_sh: Path | None, conda_env: str | None,
                     conda_envs_path: Path | None = None,
                     cache_dir: Path | None = None) -> str:
    """Fixed container-side Bash: toolchain order, then one runner.

    Two directories with two roles: the runner executes the test from test_dir,
    and source_root is the checkout whose TLA implementation is under test.
    Neither is a working directory, and neither receives the JIT cache.
    """
    lines = [
        'set -o pipefail',
        f'fail() {{ echo "{PROG}: $*" >&2; exit 3; }}',
        'export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH',
        'export PYTHONDONTWRITEBYTECODE=1',
    ]
    if cache_dir is not None:
        # Created here rather than assumed, so a missing parent is a configuration
        # fault instead of a silent fall back to the default location.
        lines.append(f'export TRITON_CACHE_DIR={shlex.quote(str(cache_dir))}')
        lines.append('mkdir -p "$TRITON_CACHE_DIR" '
                     '|| fail "cannot create TRITON_CACHE_DIR=$TRITON_CACHE_DIR"')
    else:
        # No dedicated directory configured: a private temporary one keeps the
        # artifact count attributable to this run and writes into neither tree.
        lines.append('export TRITON_CACHE_DIR=$(mktemp -d) || exit 3')
    if cann is not None:
        lines.append(f'[ -r {shlex.quote(str(cann))} ] '
                     f'|| fail {shlex.quote(f"CANN setup not readable: {cann}")}')
        # A readability check is not a load check. set_env.sh can be present and
        # still fail, and without this status check the shell would carry on into
        # the runner with no CANN on the loader path - surfacing much later as an
        # ImportError inside the test, which the Gate would then record as a test
        # failure rather than a configuration fault.
        lines.append(f'. {shlex.quote(str(cann))} '
                     f'|| fail {shlex.quote(f"failed to source CANN setup: {cann}")}')
    if conda_sh is not None and conda_env:
        lines.append(f'[ -r {shlex.quote(str(conda_sh))} ] '
                     f'|| fail {shlex.quote(f"conda.sh not readable: {conda_sh}")}')
        if conda_envs_path is not None:
            lines.append(f'export CONDA_ENVS_PATH={shlex.quote(str(conda_envs_path))}'
                         f'"${{CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}}"')
        lines.append(f'. {shlex.quote(str(conda_sh))} '
                     f'|| fail {shlex.quote(f"failed to source conda.sh: {conda_sh}")}')
        lines.append(f'conda activate {shlex.quote(conda_env)} '
                     f'|| fail {shlex.quote(f"conda activate {conda_env} failed")}')
    # The checkout under test must exist, because its TLA implementation is what
    # the test exercises, but it is not entered: neither tree is a working
    # directory, and chdir into one invites exactly the artifact deposit that
    # previously moved a provenance hash.
    lines.append(f'[ -d {shlex.quote(source_root)} ] '
                 f'|| fail {shlex.quote(f"source root is not a directory: {source_root}")}')
    lines.append(f'cd {shlex.quote(test_dir)} '
                 f'|| fail {shlex.quote(f"cannot enter test directory: {test_dir}")}')
    lines.append(f'[ -f {shlex.quote(GATE_TEST)} ] '
                 f'|| fail {shlex.quote(f"Gate test file is missing: {test_dir}/{GATE_TEST}")}')
    lines.append(f'timeout --signal=TERM --kill-after={KILL_AFTER_SECONDS} {timeout} '
                 f'python3 - {shlex.quote(source_root)} {shlex.quote(device)} '
                 f'{shlex.quote(GATE_TEST)} '
                 f'<<\'TLA_NPU\'')
    lines.extend(RUNNER.strip('\n').splitlines())
    lines.append('TLA_NPU')
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
        raise BridgeError('container produced no TLA Gate report')
    try:
        report = json.loads(lines[-1], object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
    except ValueError as exc:
        raise BridgeError('container produced an unparsable TLA Gate report') from exc
    if not isinstance(report, dict) or report.get('command') != 'tla':
        raise BridgeError('container TLA Gate report has the wrong shape')
    return report


def observations(report: dict, device: str) -> dict:
    """Only what the container measured. Missing data is false, never assumed."""
    expected = int(device)
    moves = report.get('tensorMovesToDevice')
    peak = report.get('npuPeakMemoryBytes')
    artifacts = report.get('artifacts') if isinstance(report.get('artifacts'), dict) else {}
    objects = artifacts.get('object')
    npubins = artifacts.get('npubin')
    stdout = report.get('scriptStdout')
    # The markers are read from the captured stdout the test itself produced, not
    # from a flag the test could set: a claim without the matching output cannot
    # pass.
    # CAST_BACK_MARK is a literal substring of TRITON_MARK, so a plain "in"
    # test would credit the TLA route for output that only the plain-Triton
    # reference produced. Each marker is therefore matched where it stands on
    # its own, so one route passing cannot stand in for the other.
    cast_back = isinstance(stdout, str) and bool(re.search(r'(?<!TRITON_)' + CAST_BACK_MARK, stdout))
    triton = isinstance(stdout, str) and TRITON_MARK in stdout
    result = {
        'deviceAuthorized': True,
        'npuAvailable': report.get('npuAvailable') is True,
        'scriptExitZero': type(report.get('scriptExit')) is int and report['scriptExit'] == 0,
        'castBackOk': cast_back,
        'tritonCastBackOk': triton,
        # The test really moved tensors to the device and the allocator really
        # raised peak usage. Both are measured in the process that ran the test,
        # so a stub that never calls the kernel cannot satisfy them.
        'tensorMovesToDevice': moves if type(moves) is int and moves >= 0 else 0,
        'npuPeakMemoryBytes': peak if type(peak) is int and peak >= 0 else 0,
        # Compiler products: a run that produced no kernel image did not compile
        # for this device. The Ascend backend emits .npubin; a .o is counted too
        # but is not required, because requiring an artifact the backend does not
        # produce would make this check unsatisfiable rather than strict.
        'compiledArtifactsPresent': type(npubins) is int and npubins > 0,
        'compiledObjectFiles': objects if type(objects) is int and objects >= 0 else 0,
        # Synchronization is a real device operation, not bookkeeping: when the
        # runtime cannot confirm that device work completed, the run is a failure
        # and the original message is carried into the receipt for review.
        'synchronizeOk': report.get('synchronizeOk') is True,
        'synchronizeError': report.get('synchronizeError')
        if isinstance(report.get('synchronizeError'), str) else None,
        # The runtime honoured the single-device filter. This is not vacuous: the
        # runner exports ASCEND_RT_VISIBLE_DEVICES before importing torch, and a
        # runtime that ignored the mask would report every card on the host. It
        # proves the filter took effect. It does NOT prove the process could not
        # have reached another card, because the runner itself set the mask.
        'deviceCountOne': type(report.get('npuDeviceCount')) is int
        and report['npuDeviceCount'] == 1,
        # RECORDED, NOT GATED. These are echoes of configuration this Gate
        # supplied, so they cannot fail for a hardware reason and must not be
        # presented as proof of which board ran. visibleDevices is the value the
        # runner itself exported and then read back; deviceUuid/deviceName come
        # from torch.npu.get_device_properties and name the model, not the board
        # (both cards report the same uuid on this host). They are kept because a
        # reader of the receipt benefits from seeing them, and they are explicitly
        # excluded from passed().
        'visibleDeviceEcho': report.get('visibleDevices')
        if isinstance(report.get('visibleDevices'), str) else None,
        # RECORDED, NOT GATED, for the same reason as the echo above: the counter
        # checks prove the test really ran, and these fields only let a reader tie
        # a PASS to the TLA code that produced it.
        'tlaImplementationSha256': report.get('tlaImplementationSha256')
        if isinstance(report.get('tlaImplementationSha256'), str) else None,
        'tlaImplementationFileCount': report.get('tlaImplementationFileCount')
        if type(report.get('tlaImplementationFileCount')) is int else None,
        'tritonVersion': report.get('tritonVersion')
        if isinstance(report.get('tritonVersion'), str) else None,
        'tlaHasDma': report.get('tlaHasDma')
        if isinstance(report.get('tlaHasDma'), bool) else None,
        'tlaImplementationError': report.get('tlaImplementationError')
        if isinstance(report.get('tlaImplementationError'), str) else None,
        'deviceModelUuid': report.get('deviceUuid')
        if isinstance(report.get('deviceUuid'), str) else None,
        'deviceName': report.get('deviceName')
        if isinstance(report.get('deviceName'), str) else None,
    }
    result['tensorMovedToDevice'] = result['tensorMovesToDevice'] > 0
    result['npuDeviceUsed'] = result['npuPeakMemoryBytes'] > 0
    result['numericalAssertionsObserved'] = result['castBackOk'] and result['tritonCastBackOk']
    return result

# Every condition this script gates on. It is the manifest's observationChecks
# minus exitCodeZero, which only the evaluator can decide because it describes the
# host process status rather than anything the container measured. Keeping the
# list here short of the manifest would let this script exit 0 while the evaluator
# still records a FAIL, so the two must agree.
REQUIRED = (
    'scriptExitZero', 'castBackOk', 'tritonCastBackOk', 'tensorMovedToDevice',
    'npuDeviceUsed', 'compiledArtifactsPresent', 'npuAvailable',
    'synchronizeOk', 'deviceCountOne', 'numericalAssertionsObserved')


def passed(result: dict) -> bool:
    """Exactly REQUIRED, so the exit status cannot disagree with the manifest."""
    return all(result[key] is True for key in REQUIRED)


def require_conda_pair(conda_sh: Path | None, conda_env: str | None) -> None:
    """CONDA_SH and CONDA_ENV are both-or-neither; a half configuration fails closed.

    Silently ignoring one half would run the test against a different interpreter
    than the one configured, which is exactly the failure this Adapter prevents.
    """
    if (conda_sh is None) != (conda_env is None):
        raise BridgeError('TLA_NPU_CONDA_SH and TLA_NPU_CONDA_ENV must be provided '
                          'together; set both or neither')


def check_config() -> int:
    """Report whether this machine is configured, without entering the container.

    Preflight runs this before the device Gate starts. The device id is part of
    the check: it authorizes one run, and a run without it must be refused by the
    Host rather than by a Gate that has already begun.
    """
    container, home, identity_file = container_identity()
    device = device_id()
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError(f'TLA_NPU_CONDA_ENV is not a plain environment name: {conda_env!r}')
    conda_envs_path_from_env_file()
    # The build key is part of the file grammar and the engine leases by it; an
    # invalid value must fail here rather than in the engine.
    build = machine_value(BUILD_KEY)
    if build is not None and (not SAFE_NAME.fullmatch(build) or build in {'.', '..'}):
        raise BridgeError(f'{resolved_env_file()}: {BUILD_KEY} is not a plain directory '
                          f'name: {build!r}')
    # Occupancy is a Host fact, so it is checked here, before the Recipe is
    # dispatched at all. The container's own npu-smi is not used: inside the image
    # it fails to load its libraries and would report nothing.
    npu_id = require_device_free(device)
    _, holders = device_occupancy(device)
    occupancy = 'free' if not holders else f'shared({len(holders)})'
    print(f'{PROG}: configuration ready: identity file {identity_file} '
          f'container={container} home={home} device={device} npuId={npu_id} '
          f'occupancy={occupancy} exclusive={exclusive_requested()}', file=sys.stderr)
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
        raise BridgeError(f'TLA_NPU_CONDA_ENV is not a plain environment name: {conda_env!r}')

    # Occupancy is re-read before anything touches Docker: the configuration check
    # ran earlier and another job may have taken the card since. A held card only
    # stops the Gate when this run demanded exclusivity.
    require_device_free(device)

    mounts = container_mounts(container)
    mapped = map_host_to_container(host_root, mounts, container)
    declared = declared_container_source()
    if declared is not None and os.path.realpath(str(declared)) != os.path.realpath(str(mapped)):
        raise BridgeError(f'TLA_NPU_CONTAINER_SOURCE {declared} does not match the bind-mount '
                          f'mapping {mapped} of host source root {host_root}')
    container_root = str(declared if declared is not None else mapped)

    # The test directory is a separate input with its own identity: it holds the
    # bytes that exercise the implementation. Both are mapped through the same
    # bind mounts and both are proven identical on each side.
    test_dir = test_dir_from_env_file()
    if not test_dir.is_dir():
        raise BridgeError(f'TLA_NPU_TEST_DIR does not exist: {test_dir}')
    container_test_dir = str(map_host_to_container(test_dir, mounts, container))

    # The cache is written to, so it must live outside both input trees. That is
    # enforced, not merely documented: a cache inside the source or test root
    # would reintroduce the pollution this Gate was changed to prevent.
    cache_dir = cache_dir_from_env_file()
    for label, tree in (('source root', host_root), ('test directory', test_dir)):
        if cache_dir is not None and (cache_dir == tree or cache_dir.is_relative_to(tree)):
            raise BridgeError(f'TLA_NPU_CACHE_DIR {cache_dir} is inside the {label} {tree}; '
                              f'the cache is written to and would pollute an input tree')
    container_cache_dir = None
    if cache_dir is not None:
        container_cache_dir = map_host_to_container(cache_dir, mounts, container)

    prefix = docker_exec_prefix(container, home, container_root)
    verify_same_tree(host_root, prefix, container_root)
    verify_same_tree(test_dir, prefix, container_test_dir)
    print(f'{PROG}: identity file {identity_file}', file=sys.stderr)
    print(f'{PROG}: container={container} user=root source={container_root} '
          f'testDir={container_test_dir} cache={container_cache_dir} '
          f'timeout={timeout}s device={device} test={GATE_TEST}', file=sys.stderr)
    script = container_script(str(container_test_dir), container_root, timeout, device, cann,
                              conda_sh, conda_env, conda_envs_path_from_env_file(),
                              container_cache_dir)
    completed = subprocess.run([*prefix, 'bash', '-c', script], capture_output=True, text=True)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    report = read_report(completed.stdout)
    result = observations(report, device)
    if report.get('fatal'):
        result['fatal'] = True
        result['fatalReason'] = str(report['fatal'])
    published = dict(result)
    # The host process status is the whole contract for the shell that ran; the
    # evaluator keeps a produced claim only when the host agrees, so this states
    # the intent rather than overriding it.
    published['exitCodeZero'] = True
    print(json.dumps(published, sort_keys=True))
    return 0 if (passed(result) and completed.returncode == 0) else (completed.returncode or 1)


if __name__ == '__main__':
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BridgeError as error:
        print(f'{PROG}: {error.message}', file=sys.stderr)
        # The instance, not the class: an OccupancyBlocked carries its own code so
        # the runner can record BLOCKED_ENV instead of a failed test. Reading
        # BridgeError.EXIT here would silently flatten every subclass to 2.
        raise SystemExit(error.EXIT)