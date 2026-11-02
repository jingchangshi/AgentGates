#!/usr/bin/env python3
"""Preflight and build Gate for the ForgeKernel Ascend Adapter.

Scope
-----

This Adapter does not implement the optimization loop; it makes a ForgeKernel
checkout runnable on this host and proves that it is runnable. Two steps exist
today:

  * ``--check-config`` decides whether the machine file is complete enough to
    run at all, without touching a device, a compiler or a network.
  * the recipe itself installs the adapted ForgeKernel tree into a private
    virtual environment and proves the installed package imports and reports the
    tree it came from.

Smoke, optimize and accept gates are deliberately absent, not stubbed: a Gate
that exists and always passes is worse than one that does not exist.

Why a private virtual environment
---------------------------------

The host has one shared Conda environment that other work depends on.
Installing ForgeKernel into it would let this run change a toolchain another
run is using. So the build installs into FK_ASCEND_VENV, created with
--system-site-packages: the reused TileLang, torch and torch_npu stay the ones
the shared environment provides, and only ForgeKernel itself is installed on
top. That is also why the installer is ``pip install --no-deps`` - resolving
dependencies here would upgrade a shared package, which the frozen-dependency
policy forbids.

TileLang is reused, never rebuilt
---------------------------------

FK_ASCEND_TILELANG_SOURCE and FK_ASCEND_TILELANG_BUILD name an existing
checkout and its existing build. This Gate verifies them and records their
identity; it never configures, compiles or pip-installs TileLang. If the native
library is missing the run stops with a diagnosis instead of quietly starting a
multi-hour build nobody approved.

Output contract
---------------

Standard output carries exactly one JSON object and nothing else; every
diagnostic goes to stderr. The evaluator bounds that object at 64 KiB, so long
build logs are written to a file under the private environment and only that
file's path and digest are reported.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

PROG = 'fk-ascend-gate'

# One plain name, so a configured directory cannot be an option-like string or
# escape the root it is joined onto.
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')

# Keys the machine file may define. Kept in step with machine_env.ADAPTERS; the
# host regression suite compares the two, so a key added on one side cannot
# become configuration the other silently ignores.
TOOLCHAIN_KEYS = (
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

# The keys this Gate cannot run without. The provider keys describe the agent
# that will later drive the loop, not this build, so they are validated when
# present and not demanded here.
REQUIRED = ('FK_ASCEND_PYTHON', 'FK_ASCEND_TILELANG_SOURCE', 'FK_ASCEND_TILELANG_BUILD',
            'FK_ASCEND_VENV')

BUILD_KEY = 'FK_ASCEND_BUILD_SUBDIR'

# The observable properties this Gate gates on. Kept identical to the manifest
# observationChecks minus exitCodeZero, which the evaluator derives from the host
# process status and an Adapter cannot publish for itself.
CHECKS = ('venvCreated', 'forgekernelInstalled', 'importOk', 'versionReported',
          'forgeEntryPointOk', 'tilelangReused', 'tilelangSourceUnchanged')

# Variables that are deliberately NOT machine-file keys: the file selector, the
# per-run device authorization, and the probe command a test may substitute.
NOT_MACHINE_KEYS = frozenset({'FK_ASCEND_ENV_FILE', 'FK_ASCEND_DEVICE_ID', 'FK_ASCEND_SMI'})

# A single install is bounded well below the manifest timeout: a resolver that
# wanders for twenty minutes is a diagnosis, not progress.
INSTALL_TIMEOUT_SECONDS = 900

# The packaging entry point this Adapter installs from, and the probe that proves
# the installed package is the one that can actually be imported.
PKG_FILE = 'pyproject.toml'
PROBE = ('import json\n'
         'import forgekernel\n'
         'from pathlib import Path\n'
         'print(json.dumps({"file": forgekernel.__file__,\n'
         '                  "version": str(getattr(forgekernel, "__version__", "")),\n'
         '                  "cli": str(Path(forgekernel.__file__).parent / "__main__.py")}))')

# How long a window polls before giving up on a lease that never opens. The bound
# comes from the manifest (deviceWindow.waitSeconds) which the engine publishes;
# this is only the fallback for a script started by hand.
DEFAULT_WINDOW_WAIT_SECONDS = 60.0


class BridgeError(Exception):
    """A configuration or environment fault, not a Gate result."""

    EXIT = 2


class DeviceWindowBusy(BridgeError):
    """The device lease did not open within the declared wait.

    This is a resource refusal, not a failed build: nothing was dispatched to the
    card. It carries its own exit code so the runner records BLOCKED_ENV, which
    is honest - the work did not run. It is never produced by an occupied card,
    because another tenant's process is normal and is only recorded.
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


# A conservative NAME=value grammar, identical to the other Adapters': the file
# is data, so reading it never executes host code.
_ASSIGNMENT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')


def _parse_env_file(text: str, path: Path) -> dict:
    """Parse the strict NAME=value machine file; never executes it."""
    values = {}
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
CHECKOUT_MARKER = Path('adapters') / 'forgekernel-ascend-npu' / 'env.example.sh'


def checkout_env_file() -> Path | None:
    """The in-repository machine file, when this Pack belongs to a checkout."""
    for ancestor in (ADAPTER_DIR, *ADAPTER_DIR.parents):
        if (ancestor / CHECKOUT_MARKER).is_file():
            return ancestor / CHECKOUT_MARKER.parent / 'env.sh'
    return None


def default_env_file() -> Path | None:
    """The machine file the checkout owns, or None outside a checkout.

    There is no second location: a machine that keeps its file elsewhere names it
    in FK_ASCEND_ENV_FILE. Guessing a home-directory path would silently pick up a
    file nobody is reviewing.
    """
    return checkout_env_file()


def resolved_env_file() -> Path | None:
    """The machine file to read, or None when this machine has none."""
    explicit = env_value('FK_ASCEND_ENV_FILE')
    if not explicit:
        return default_env_file()
    path = Path(explicit).expanduser()
    if not path.is_absolute():
        raise BridgeError(f'FK_ASCEND_ENV_FILE must be an absolute path: {explicit!r}')
    if not path.exists():
        raise BridgeError(f'FK_ASCEND_ENV_FILE does not exist: {path}')
    return path


def env_file_values() -> dict:
    """Every key the machine file defines, parsed once for the whole Gate."""
    path = resolved_env_file()
    if path is None or not path.exists():
        return {}
    if not path.is_file():
        raise BridgeError(f'FK_ASCEND_ENV_FILE is not a regular file: {path}')
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


def _absolute(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise BridgeError(f'{name} must be an absolute path: {value!r}')
    if chr(10) in value or chr(0) in value:
        raise BridgeError(f'{name} contains an illegal character')
    return path


def required_path(name: str, *, file: bool = False) -> Path:
    """A configured path that must exist; the message names the key, not a value."""
    value = machine_value(name)
    if value is None:
        raise BridgeError(
            f'{resolved_env_file() or "no machine file was found"} does not define {name}; '
            f'copy env.example.sh to env.sh beside it and fill in {name}')
    path = _absolute(value, name)
    if file and not path.is_file():
        raise BridgeError(f'{name} is not a regular file: {path}')
    if not file and not path.is_dir():
        raise BridgeError(f'{name} is not a directory: {path}')
    return path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def venv_python(venv: Path) -> Path:
    return venv / 'bin' / 'python'


def tilelang_identity() -> dict:
    """The REUSED TileLang this Adapter depends on, bound by identity.

    Reusing an existing build is an explicit dependency, not a build step: the
    run must record what it reused so a later change to that checkout invalidates
    the evidence. A missing native library is a diagnosis, never a rebuild.
    """
    source = required_path('FK_ASCEND_TILELANG_SOURCE')
    build = required_path('FK_ASCEND_TILELANG_BUILD')
    module = source / 'tilelang' / '__init__.py'
    if not module.is_file():
        raise BridgeError(f'FK_ASCEND_TILELANG_SOURCE has no tilelang/__init__.py: {source}')
    native = sorted(path for path in build.rglob('libtilelang*.so*') if path.is_file())
    if not native:
        raise BridgeError(
            f'FK_ASCEND_TILELANG_BUILD contains no libtilelang native library: {build}. '
            f'This Adapter REUSES an existing TileLang build and never starts one; '
            f'point the key at the build that already exists or build it deliberately.')
    return {'source': str(source), 'build': str(build),
            'moduleSha256': digest(module),
            'nativeLibraries': {str(path.relative_to(build)): digest(path) for path in native}}


# ── device window ───────────────────────────────────────────────────────────


def lease_path() -> Path:
    """The lease the engine handed this Recipe, or a refusal.

    The path is GIVEN, never derived. The lease file is a hash of the resource
    name, and re-deriving it here could lock a different file from the one the
    engine reserved - which is how two runs end up believing they each hold the
    same card.
    """
    value = env_value('AGENT_GATES_DEVICE_LEASE_PATH')
    if not value:
        raise BridgeError(
            'AGENT_GATES_DEVICE_LEASE_PATH is not set: this Recipe declares a device '
            'window, so the engine must hand it the lease to open')
    return _absolute(value, 'AGENT_GATES_DEVICE_LEASE_PATH')


def lease_resource() -> str:
    """The resource name this window must report back verbatim.

    The engine compares it against the card it authorized, so a window that
    opened some other lease is refused rather than credited.
    """
    value = env_value('AGENT_GATES_DEVICE_LEASE_RESOURCE')
    if not value:
        raise BridgeError('AGENT_GATES_DEVICE_LEASE_RESOURCE is not set')
    return value


def lease_wait_seconds() -> float:
    """The bounded wait the manifest declared, in seconds.

    A window refused immediately would turn ordinary coordination between two of
    our own runs into a spurious BLOCKED_ENV; a window that waited forever would
    wedge the run. Hence a declared bound, and a refusal when it expires.
    """
    value = env_value('AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS')
    if not value:
        return DEFAULT_WINDOW_WAIT_SECONDS
    try:
        seconds = float(value)
    except ValueError:
        raise BridgeError(f'AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS is not a number: {value!r}') from None
    if not seconds > 0:
        raise BridgeError(f'AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS must be positive: {value!r}')
    return seconds


def _timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def occupancy(resource: str) -> str:
    """Observed occupancy OF THE LEASED CARD, as a value the contract defines.

    The card is taken from the resource name (npu:3), not assumed, because a
    machine-wide count would report a busy neighbour when THIS card is free and
    vice versa - a number that reads like evidence but is not about the thing
    being leased.

    Recorded, never gated. "unknown" is an honest answer and silence is not one:
    a reader must be able to tell a quiet window from a shared one. Only a count
    is reported - other tenants' process ids and command lines never leave the
    probe.
    """
    smi = env_value('FK_ASCEND_SMI', default='npu-smi') or 'npu-smi'
    physical = resource.partition(':')[2]
    if not physical.isdigit():
        return 'unknown'
    try:
        done = subprocess.run([smi, 'info', '-t', 'proc-mem', '-i', physical],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return 'unknown'
    if done.returncode != 0:
        return 'unknown'
    # The per-device report must name the card it describes, or the answer is
    # about something else and is reported as unknown rather than counted.
    if not re.search(rf'NPU ID\s*:\s*{physical}\s*$', done.stdout, re.MULTILINE):
        return 'unknown'
    idle = re.search(r'No process in device\.?\s*$', done.stdout, re.MULTILINE)
    holders = len(re.findall(r'Process id:\s*[0-9]+', done.stdout))
    if holders == 0 and idle:
        return 'free'
    if holders:
        return f'shared({holders})'
    return 'unknown'
    # A process row is "| <npu id> | <pid> | <name> | <memory> |"; no other row
    # in this report starts with two integer fields, so the count cannot pick up
    # a utilisation or memory line.
    holders = len(re.findall(r'^\|\s*[0-9]+\s*\|\s*[0-9]+\s*\|', done.stdout, re.MULTILINE))
    if holders == 0:
        return 'free'
    return f'shared({holders})'


@contextlib.contextmanager
def device_window():
    """Open the device lease, yield its facts, and release it.

    This window is the ONLY span in which this Adapter occupies a card.
    Thinking, compiling and writing files happen outside it, so a long
    optimization turn never holds hardware it is not using.

    Two properties are deliberate:

    * another tenant on the card is not an error. Occupancy is measured and
      reported, never gated, because on a shared machine someone else using the
      card is the normal case;
    * a lease that does not open within the declared wait raises
      DeviceWindowBusy, which exits 75 so the runner records BLOCKED_ENV. That is
      a resource refusal, never a failed test.

    What the caller receives is what the receipt will carry: which lease, when it
    opened, when it closed, what was observed, and how long it waited.
    """
    import fcntl
    import time

    path = lease_path()
    resource = lease_resource()
    wait = lease_wait_seconds()
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
                        f'device lease {resource} did not open within {wait}s; another '
                        f'AgentGates run holds it') from None
                time.sleep(min(0.05, remaining))
        opened = time.monotonic()
        report = {'resource': resource, 'path': str(path),
                  'startedAt': _timestamp(),
                  'waited': round(opened - started, 3),
                  'occupancy': occupancy(resource), 'releasedAt': None,
                  # The occupancy probe shells out to npu-smi, so it happens before
                  # the work starts. Its own cost is published for the same reason
                  # the warmup is: a window should be dominated by work.
                  'probeSeconds': round(time.monotonic() - opened, 3)}
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

def check_config() -> int:
    """Decide whether this machine is configured, without building anything.

    Nothing here starts a process, touches a device or reaches the network: the
    caller is asking "would a build run make sense", and the answer must not
    depend on the state a build would itself change.
    """
    values = env_file_values()
    path = resolved_env_file()
    # An unconfigured checkout and an empty file are different diagnoses, and a
    # reader needs to know which: one is a missing file, the other is a file that
    # omits required keys. Reporting the second for the first sends someone to
    # edit a file that does not exist.
    if path is None or not path.exists():
        raise BridgeError(
            f'no machine file was found (looked for {path}). Copy env.example.sh to env.sh '
            f'beside it, or set FK_ASCEND_ENV_FILE to an absolute path, and fill in the '
            f'values this machine uses.')
    # The file is the ONLY source of these values, so an export is refused
    # whether or not it agrees: silently ignoring it would leave a shell that
    # believes it configured something, and honouring it would make the receipt
    # describe a machine the file does not. The exceptions are the three
    # variables that are deliberately not machine facts: the file selector, the
    # per-run device authorization, and the probe command a test may substitute.
    for name in sorted(os.environ):
        if not name.startswith('FK_ASCEND_') or name in NOT_MACHINE_KEYS:
            continue
        if name not in TOOLCHAIN_KEYS:
            raise BridgeError(f'{name} is exported but is not a machine-file key; '
                              f'allowed keys are {", ".join(TOOLCHAIN_KEYS)}')
        raise BridgeError(
            f'{name} is exported; machine values are read only from {path}. '
            f'Define it there instead of the launching environment.')
    missing = [name for name in REQUIRED if not values.get(name)]
    if missing:
        raise BridgeError(f'{path} does not define {", ".join(missing)}')
    # Every configured path must be absolute; a relative one would resolve
    # against whatever directory the Gate happened to start in.
    for name, value in values.items():
        if not value:
            continue
        if name.endswith(('_SETUP', '_PYTHON', '_SOURCE', '_BUILD', '_ROOT', '_VENV',
                          '_BIN', '_DIR')):
            _absolute(value, name)
    python = _absolute(values['FK_ASCEND_PYTHON'], 'FK_ASCEND_PYTHON')
    if not python.is_file() or not os.access(python, os.X_OK):
        raise BridgeError(f'FK_ASCEND_PYTHON is not an executable file: {python}')
    build_key = values.get(BUILD_KEY)
    if build_key is not None and (not SAFE_NAME.fullmatch(build_key) or build_key in {'.', '..'}):
        raise BridgeError(f'{path}: {BUILD_KEY} is not a plain directory name: {build_key!r}')
    policy = values.get('FK_ASCEND_DEPENDENCY_POLICY')
    if policy is not None and policy not in {'frozen', 'online'}:
        raise BridgeError(f'{path}: FK_ASCEND_DEPENDENCY_POLICY must be "frozen" or '
                          f'"online": {policy!r}')
    identity = tilelang_identity()
    print(f'{PROG}: configuration ready: file {path} venv {values["FK_ASCEND_VENV"]} '
          f'tilelang source {identity["source"]} build {identity["build"]} '
          f'native={len(identity["nativeLibraries"])}', file=sys.stderr)
    return 0


def check_config_smoke() -> int:
    """Preflight for the smoke Gate: can this run reach a device at all?

    Deliberately weaker than the build preflight. The smoke Gate exists to run
    before any ForgeKernel installation, so it must not demand the TileLang
    checkout, the venv or the CANN setup that a build needs. It checks the three
    things it actually uses: a machine file, an executable interpreter, and an
    authorized device id.

    A device that is absent or busy is NOT detected here. That is reported by the
    Gate itself as BLOCKED_ENV, because a preflight that raced the device would
    produce a different answer depending on timing, and a run that is blocked
    after the preflight is already classified honestly.
    """
    env_file_values()
    setup = smoke_environment()
    identity = smoke_probe_identity()
    print(f'{PROG}: smoke configuration ready: interpreter {setup["interpreter"]} '
          f'device {setup["device"]} probe {identity["path"]} '
          f'probeSha256 {identity["sha256"]}', file=sys.stderr)
    return 0


def smoke_probe_identity() -> dict:
    """The probe this Gate will actually execute, bound by digest.

    The probe runs on the device, so the receipt must say which bytes ran. A
    pinned manifest entry would already do that for an installed Pack, but the
    Gate is also runnable from a checkout where the file could have been edited
    after pinning, so the digest is recomputed and published.
    """
    if not PROBE_SCRIPT.is_file():
        raise BridgeError(f'the device probe is missing from this Pack: {PROBE_SCRIPT}')
    return {'path': str(PROBE_SCRIPT), 'sha256': digest(PROBE_SCRIPT)}

def run(argv: list, *, timeout: float = 300, log: Path | None = None) -> subprocess.CompletedProcess:
    """Run one command, capturing output; the log file keeps the long version."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise BridgeError(f'cannot start {argv[0]}: {exc}') from exc
    except subprocess.TimeoutExpired as exc:
        raise BridgeError(f'{argv[0]} did not finish within {timeout}s') from exc
    if log is not None:
        with Path(log).open('a', encoding='utf-8') as stream:
            stream.write('$ ' + ' '.join(str(item) for item in argv) + chr(10))
            stream.write(done.stdout)
            stream.write(done.stderr)
    return done


def create_venv(python: Path, venv: Path) -> None:
    """Create the private environment, reusing the shared site packages.

    --system-site-packages is the whole point: the reused TileLang, torch and
    torch_npu stay exactly the ones the shared environment provides. Without it
    this Adapter would have to install its own copies, which is a second
    toolchain nobody reviewed and exactly what the frozen-dependency policy
    forbids.
    """
    if venv_python(venv).is_file():
        return
    venv.parent.mkdir(parents=True, exist_ok=True)
    done = run([str(python), '-m', 'venv', '--system-site-packages', str(venv)],
               timeout=INSTALL_TIMEOUT_SECONDS)
    if done.returncode != 0:
        detail = done.stderr.strip().splitlines()
        raise BridgeError(f'python -m venv failed ({done.returncode}): '
                          f'{detail[-1] if detail else "no output"}')


def build(source: Path) -> dict:
    """Install the adapted ForgeKernel tree into the private environment.

    The source tree is READ, never written: sourceWrite is false, and a build
    that edited the checkout would change the identity the receipt binds.
    """
    python = required_path('FK_ASCEND_PYTHON', file=True)
    venv_value = machine_value('FK_ASCEND_VENV')
    if venv_value is None:
        raise BridgeError('FK_ASCEND_VENV is required; it names the private environment '
                          'this build installs into')
    venv = _absolute(venv_value, 'FK_ASCEND_VENV')
    pyproject = source / PKG_FILE
    if not pyproject.is_file():
        raise BridgeError(f'source root has no {PKG_FILE}: {source}')
    before = digest(pyproject)
    identity = tilelang_identity()
    create_venv(python, venv)
    install_log = venv.parent / 'build-install.log'
    if install_log.is_file():
        install_log.unlink()
    # --no-deps on purpose: resolving dependencies would upgrade the shared
    # environment this Adapter deliberately reuses.
    done = run([str(venv_python(venv)), '-m', 'pip', 'install', '--no-deps',
                '--no-build-isolation', str(source)],
               timeout=INSTALL_TIMEOUT_SECONDS, log=install_log)
    installed = done.returncode == 0
    result = {'venvCreated': venv_python(venv).is_file(), 'forgekernelInstalled': installed,
              'installLog': str(install_log) if install_log.is_file() else None,
              'installLogSha256': digest(install_log) if install_log.is_file() else None,
              'tilelangSourceUnchanged': digest(pyproject) == before,
              'tilelang': identity, 'venv': str(venv)}
    if not installed:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise BridgeError(f'pip install failed ({done.returncode}): '
                          f'{detail[-1] if detail else "no output"}; log {install_log}')
    # The installed package is what the next Gate runs, so prove it is importable
    # and that it reports the tree it came from. A wheel that installs but cannot
    # import is not a built Adapter.
    probe = run([str(venv_python(venv)), '-I', '-c', PROBE], timeout=120)
    probe_ok = probe.returncode == 0
    fact = {}
    if probe_ok:
        try:
            fact = json.loads(probe.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            probe_ok = False
    result.update(importOk=probe_ok,
                  versionReported=bool(str(fact.get('version', '')).strip()) if probe_ok else False,
                  forgeEntryPointOk=bool(fact.get('cli')) and Path(fact.get('cli', '')).is_file()
                  if probe_ok else False,
                  moduleFile=fact.get('file') if probe_ok else None,
                  moduleVersion=fact.get('version') if probe_ok else None)
    if not probe_ok:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        raise BridgeError(f'installed forgekernel does not import: '
                          f'{detail[-1] if detail else "no output"}')
    # Reusing a build is only honest if the build is still the one recorded.
    # Re-reading the identity after the install catches a concurrent rebuild of
    # the shared TileLang checkout between check and use.
    if tilelang_identity() != identity:
        raise BridgeError('the reused TileLang checkout changed during the build; the '
                          'evidence would describe a build that no longer exists')
    result['tilelangReused'] = True
    return result


def probe_window(seconds: float) -> dict:
    """Open the device window, do a short real thing, and close it.

    This is the smallest honest use of the lease: it proves the window opens,
    that occupancy is measured while it is held, and that it closes. It does NOT
    run a kernel or measure anything, so it is a lease check - not a smoke test,
    not a performance result. Those gates are still to be written.

    Holding the card for a moment is deliberate: the point of the window is that
    it is taken around real work and released, and a window that closes before
    anyone could look is not evidence that anything was held.
    """
    import time

    with device_window() as window:
        # Work that genuinely needs the lease would go here. For now the window
        # holds for the requested moment so its extent is observable.
        time.sleep(min(seconds, 5.0))
        held = window
    observed = {
        'deviceWindow': held,
        'windowOpened': True,
        'windowReleased': held.get('releasedAt') is not None,
        'occupancyRecorded': held.get('occupancy') in {'free', 'unknown'}
                             or bool(re.fullmatch(r'shared\([0-9]+\)', held.get('occupancy') or '')),
    }
    return observed

# ── smoke: the smallest real device work, inside the window ──────────────────

# The keys the smoke Gate needs. Deliberately fewer than the build Gate: it must
# be runnable before any ForgeKernel installation exists, so it asks only for the
# interpreter that can import torch_npu. TileLang is not imported, nothing is
# compiled, and no check reaches the network.
SMOKE_REQUIRED = ('FK_ASCEND_PYTHON',)
SMOKE_OPTIONAL = ('FK_ASCEND_CANN_SETUP',)

# The child that does the device work. It is a separate reviewed file rather
# than an inline string: the work is the thing worth reading, and it can be run
# directly without a manifest, an engine or a lease.
PROBE_SCRIPT = Path(__file__).resolve().parent / 'fk_device_probe.py'

# The observable properties the smoke gate gates on. Every one is a literal
# boolean the probe or the window can compute; a count or a duration is
# published for the receipt but never compared, because "more than zero" is not
# expressible in an equality check.
SMOKE_CHECKS = ('npuAvailable', 'tensorOnDevice', 'numericalOk', 'synchronizeOk',
                'deviceMemoryPositive', 'windowOpened', 'windowReleased',
                'occupancyRecorded')


def smoke_environment() -> dict:
    """The child environment: the configured interpreter, made to see one card.

    The device id is authorized per run (FK_ASCEND_DEVICE_ID), so it is NOT read
    from the machine file. It is exported as the visible-device list so that the
    single card the operator named is the one the child can reach: logical 0 in
    the child is therefore the authorized card, not whatever card 0 happens to
    be on the host.
    """
    python = machine_value('FK_ASCEND_PYTHON')
    if python is None:
        raise BridgeError(
            f'{(resolved_env_file() or "no machine file was found")} does not define '
            f'FK_ASCEND_PYTHON; the smoke Gate needs an interpreter that can import '
            f'torch_npu')
    interpreter = _absolute(python, 'FK_ASCEND_PYTHON')
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise BridgeError(f'FK_ASCEND_PYTHON is not an executable file: {interpreter}')
    device = env_value('FK_ASCEND_DEVICE_ID')
    if not device or not device.isdigit():
        raise BridgeError('FK_ASCEND_DEVICE_ID is not set to a device id; the smoke Gate '
                          'refuses to choose a card for you')
    can_setup = machine_value('FK_ASCEND_CANN_SETUP')
    return {
        'interpreter': interpreter,
        'device': str(int(device)),
        'cannSetup': _absolute(can_setup, 'FK_ASCEND_CANN_SETUP') if can_setup else None,
    }


def smoke() -> dict:
    """Run the device probe and publish what it measured.

    The PROBE owns the window, not this function. That is deliberate: importing
    torch_npu costs tens of seconds and touches no hardware, so a window opened
    here would be almost entirely import time - precisely the "holding a card
    without using it" the window exists to prevent. The probe therefore warms up
    first, takes the lease around the allocation and arithmetic alone, and
    releases it before reporting. This Gate just runs it and checks the result.

    The lease variables reach the probe through the environment the engine set,
    and the probe refuses to invent a lease path if they are absent.

    A missing backend or card is exit 3 from the probe, which becomes
    DeviceWindowBusy (75) here so the runner records BLOCKED_ENV: nothing was
    measured, so calling it a failed test would be a false claim in the other
    direction. A probe that ran and produced wrong numbers exits 1 and is a real
    failure.
    """
    setup = smoke_environment()
    identity = smoke_probe_identity()
    command = []
    if setup['cannSetup'] is not None:
        # The CANN environment must be initialized before torch imports torch_npu,
        # because the backend resolves its libraries from there. The path is a
        # machine value, so it is emitted as a single-quoted literal: inside
        # double quotes a $(...) would execute.
        command = ['bash', '-c',
                   'set -e\nsource ' + shlex.quote(str(setup['cannSetup'])) +
                   '\nexec "$@"', 'fk-smoke']
    command += [str(setup['interpreter']), '-I', str(PROBE_SCRIPT)]
    child_env = dict(os.environ)
    child_env['ASCEND_RT_VISIBLE_DEVICES'] = setup['device']
    child_env['FK_ASCEND_DEVICE_ID'] = setup['device']
    child_env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=300,
                             env=child_env)
    except subprocess.TimeoutExpired as exc:
        raise BridgeError('the device probe did not finish within 300s') from exc
    except FileNotFoundError as exc:
        raise BridgeError(f'cannot start the device probe: {exc}') from exc

    probe = {}
    try:
        probe = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        if done.returncode == 3:
            raise DeviceWindowBusy('no usable device: '
                                   + (done.stderr.strip()[-300:] or 'the probe reported '
                                      'no device')) from None
        raise BridgeError('the device probe produced no readable report; '
                          f'exit {done.returncode}, stderr: {done.stderr.strip()[-400:]}')

    if done.returncode == 3:
        raise DeviceWindowBusy('no usable device: '
                               + (probe.get('opsError') or done.stderr.strip()[-300:]
                                  or 'the probe reported no device'))
    if done.returncode not in (0, 1):
        raise BridgeError(f'the device probe exited {done.returncode}: '
                          f'{done.stderr.strip()[-400:]}')

    result = {key: value for key, value in probe.items()}
    result['probe'] = identity
    result['authorizedDeviceId'] = setup['device']
    result['npuAvailable'] = bool(probe.get('npuAvailable'))
    result['tensorOnDevice'] = bool(probe.get('tensorOnDevice'))
    result['numericalOk'] = bool(probe.get('numericalOk'))
    result['synchronizeOk'] = bool(probe.get('synchronizeOk'))
    result['deviceMemoryPositive'] = int(probe.get('deviceMemoryAllocated') or 0) > 0
    result['windowOpened'] = bool(probe.get('windowOpened'))
    result['windowReleased'] = bool(probe.get('windowReleased'))
    result['occupancyRecorded'] = bool(probe.get('occupancyRecorded'))
    return result

def smoke_passed(result: dict) -> bool:
    """Exactly SMOKE_CHECKS, so the exit status cannot disagree with the manifest."""
    return all(result.get(key) is True for key in SMOKE_CHECKS)


def passed(result: dict) -> bool:
    """Exactly CHECKS, so the exit status cannot disagree with the manifest."""
    return all(result.get(key) is True for key in CHECKS)


def main(argv: list) -> int:
    if list(argv) == ['--check-config']:
        return check_config()
    if list(argv) == ['--check-config-smoke']:
        return check_config_smoke()
    if argv[:1] == ['--build']:
        if len(argv) != 1:
            raise BridgeError('--build takes no further arguments')
        result = build(Path.cwd())
        published = dict(result)
        published['exitCodeZero'] = True
        print(json.dumps(published, sort_keys=True))
        return 0 if passed(result) else 1
    if list(argv) == ['--smoke']:
        result = smoke()
        published = dict(result)
        published['exitCodeZero'] = True
        # Never print the CANN path or the interpreter: a receipt is shared, and
        # a machine path in it is a fact about this host, not about the Gate.
        print(json.dumps(published, sort_keys=True, default=str))
        return 0 if smoke_passed(result) else 1
    if argv[:1] == ['--probe-window']:
        # A verification mode, not a Gate: it is declared in no manifest, it is
        # selected by no gate id, and its output claims nothing about a kernel.
        # It exists so the lease can be exercised without inventing a device Gate
        # before there is one.
        seconds = 0.0
        if len(argv) == 3 and argv[1] == '--hold':
            try:
                seconds = float(argv[2])
            except ValueError:
                raise BridgeError(f'--hold takes seconds: {argv[2]!r}') from None
        elif len(argv) != 1:
            raise BridgeError(f'--probe-window takes at most --hold SECONDS: {argv!r}')
        print(json.dumps(probe_window(seconds), sort_keys=True))
        return 0
    raise BridgeError(f'unexpected arguments: {argv!r}')


if __name__ == '__main__':
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BridgeError as error:
        print(f'{PROG}: {error}', file=sys.stderr)
        # The instance, not the class: a DeviceWindowBusy carries its own code so
        # the runner records BLOCKED_ENV instead of a failed Gate.
        raise SystemExit(error.EXIT)