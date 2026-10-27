#!/usr/bin/env python3
"""Run one TileLang Gate inside the build container via `docker exec`.

The Harness Recipe runs this file with the TileLang source root as its working
directory. Everything about the container is supplied at run time through
environment variables, so this Adapter never hard-codes a container name, user,
host path, repository name, Git branch, CANN installation path, Conda
installation path or device id.

How a Gate is decided
---------------------

The container measures; this bridge decides. `docker exec` runs one fixed Bash
program that initializes the toolchain, runs the command under `timeout
--signal=TERM --kill-after=15`, and prints a small raw report as the *only*
line on standard output. Every other byte of container output goes to this
bridge's standard error, which the Harness keeps as the Gate's `stderr` log.

The bridge then prints exactly one JSON object on its own standard output. The
Harness reads that object as the Gate's observations and compares the declared
keys literally. A missing, unparsable or non-object report is a FAIL, so the
Gate fails closed. The container exit code is returned unchanged (124 on
timeout, 125/126/127 for Docker-level errors); this bridge only adds 2 for a
configuration or mapping error and 3 for a structural failure inside the
container.

Host and container must see one tree
------------------------------------

The Host source root is the process working directory. The corresponding
container path is derived from `docker inspect` bind mounts and then *proved* by
comparing the device/inode pair seen by both sides. After that proof, every
relative artifact path the container reports is hashed again on the Host, so the
Gate attests the bytes of the reviewed checkout rather than a file that merely
has the expected name.

Environment variables (all read at run time):

  TILELANG_HIVM_CONTAINER_NAME    optional, name of a running build container. The
                                  machine file's TILELANG_HIVM_CONTAINER_NAME is
                                  the only source; when that line is absent the
                                  Gate refuses to run. See below.
  TILELANG_HIVM_CONTAINER_USER    optional, user to run as inside the container
  TILELANG_HIVM_CONTAINER_HOME    optional, HOME inside the container
  TILELANG_HIVM_CONTAINER_SOURCE  optional, explicit container path of the source
                                  root; when set it must match the bind-mount
                                  mapping, when unset the mapping is used directly
  TILELANG_HIVM_BUILD_SUBDIR      optional, build directory name (default: build)
  TILELANG_HIVM_NINJA_JOBS        optional, ninja -j value (default: container CPUs)
  TILELANG_HIVM_GATE_TIMEOUT      optional, container-side timeout in seconds; used
                                  ONLY when no `--timeout` argument is passed

Container identity comes from the same machine file as CANN and Conda, not from
the launching shell. `env.sh` holds the command that creates the container
(`docker/config.env`) and the command that uses it, so one edit cannot leave
them disagreeing. A shell export of the three identity variables has no effect:
that is what made two agents in one run address two different containers.

Toolchain configuration comes from a file, not from the launching shell. The
CANN and Conda sequence itself is written only in `scripts/init-toolchain.sh`,
which this Gate inlines into the container-side script and calls with the three
values it validated on the Host, so the sequence a Gate runs and the sequence a
human sources are one reviewed, digest-pinned file instead of two copies that
can drift apart. That file is the only place the order lives: PATH, then CANN,
then Conda, then the interpreter check. Neither CANN nor Conda is required; a
half Conda configuration (conda.sh without an environment name) is refused.

  TILELANG_HIVM_ENV_FILE          path of the env.sh that holds CANN/Conda;
                                  default the checkout's own
                                  adapters/tilelang-hivm/env.sh

The file contains plain `NAME=value` assignments for these keys, and nothing
else is read from it:

  TILELANG_HIVM_CANN_SETUP        optional, CANN set_env.sh inside the container
  TILELANG_HIVM_CONDA_SH          optional, conda.sh inside the container
  TILELANG_HIVM_CONDA_ENV         optional, conda environment to activate
  TILELANG_HIVM_HIVM_COMPILER     optional, the Late-hivm compiler inside the
                                  container, forwarded as HIVM_COMPILER
  TILELANG_HIVM_HIVM_TARGET       optional, forwarded as HIVM_TARGET; only the
                                  codegen's own device is accepted

The file is the single source of truth for CANN and Conda; the whole
`tilelang-hivm` and `tilelang-hivm-npu` pair share it, because both use the
`TILELANG_HIVM_` prefix. Editing it changes the next Gate without recreating the
container. Exporting the toolchain variables in the launching shell has no
effect: a stale export must never mask a file edit.

`HIVM_COMPILER` is forwarded, not inherited
-------------------------------------------

`docker exec` starts a process that inherits nothing from this Gate, so every
variable the container-side TileLang reads has to be named on that command line.
Test subject `testing/ascend/codegen/test_hivm_build.py` skips itself when
`HIVM_COMPILER` is unset, so a dropped value used to turn "no codegen test ran"
into a run a reader could mistake for "the codegen tests passed". The Gate now
forwards the value from the machine file and, when it is set, refuses to run
unless the container can execute it. Leaving it empty keeps the previous
behaviour, which is correct on a machine without the compiler.

Usage:

  docker_gate.py --check-config
  docker_gate.py [--timeout SECONDS] build
  docker_gate.py [--timeout SECONDS] provenance
  docker_gate.py [--timeout SECONDS] regression

`--check-config` resolves the machine configuration and exits without entering
the container. Preflight runs it, so a Gate is never dispatched into a
configuration the Host could already have refused.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PROG = 'tilelang-hivm-docker-gate'
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
COMMANDS = ('build', 'provenance', 'regression')
# Grace period between `timeout` sending TERM and the container-side SIGKILL. It is
# deliberately short: the Harness Recipe timeout is the outer bound, so the whole
# `timeout + KILL_AFTER_SECONDS` envelope must still fit inside it. The manifest
# pins both numbers and the Recipes always pass `--timeout` explicitly, so
# `TILELANG_HIVM_GATE_TIMEOUT` can never silently widen the envelope.
KILL_AFTER_SECONDS = 15

# The Host-only regression set: repository-relative test files, not
# machine-specific values, and the exact command measured when this Adapter was
# authored. Widening it means editing this digest-pinned file and therefore
# re-trusting the Adapter; it cannot be changed from the environment, because a
# Gate that can be pointed at a different test set is not the Gate that was
# validated.
REGRESSION_TESTS = (
    'testing/ascend/target/test_tilelang_ascend_target.py',
    'testing/ascend/cache/test_tilelang_ascend_binary_cache.py',
)
# The whole container-side Bash program, probe and report included, is built
# from these constants only. Nothing here is read from the environment.

# Container-side provenance probe. It prints one raw report that describes what
# the container would actually load: the imported Python module, the native
# shared objects reachable through `tilelang.env.TL_LIBS`, and the Bisheng
# compiler that the toolkit exposes. It computes no digest from anything other
# than the files it read.
PROBE = r'''
import hashlib, importlib, importlib.metadata, json, os, shutil, sys

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def real(path):
    return os.path.realpath(path)

command, source = sys.argv[1], real(sys.argv[2])
facts = {"sourceRoot": source}
info = os.stat(source)
facts["sourceDevIno"] = "%d:%d" % (info.st_dev, info.st_ino)

import tilelang

module = real(tilelang.__file__)
facts["pythonModuleRel"] = os.path.relpath(module, source)
facts["pythonModuleSha256"] = sha256(module)
facts["version"] = getattr(tilelang, "__version__", None)
try:
    facts["distVersion"] = importlib.metadata.version("tilelang")
except Exception:
    facts["distVersion"] = None

environment = importlib.import_module("tilelang.env")
libs = [real(path) for path in environment.TL_LIBS]
facts["tlLibDirs"] = libs
facts["tlLibDirsRel"] = [os.path.relpath(path, source) for path in libs]
facts["devMode"] = bool(environment.DEV)

import tilelang.testing as testing
facts["ascendBackendCompiled"] = bool(testing.ascend_backend_compiled())

native = {}
for directory in libs:
    if not os.path.isdir(directory):
        continue
    for name in sorted(os.listdir(directory)):
        candidate = os.path.join(directory, name)
        if name.endswith(".so") and os.path.isfile(candidate):
            native[os.path.relpath(candidate, source)] = sha256(candidate)
facts["nativeLibs"] = native

bisheng = shutil.which("bishengir-compile")
facts["bishengirCompile"] = real(bisheng) if bisheng else None
facts["bishengirCompileSha256"] = sha256(real(bisheng)) if bisheng else None
facts["ascendHomePath"] = os.environ.get("ASCEND_HOME_PATH")
facts["condaDefaultEnv"] = os.environ.get("CONDA_DEFAULT_ENV")
facts["pythonExecutable"] = real(sys.executable)
print(json.dumps({"command": command, "facts": facts}, sort_keys=True))
'''

# Container-side report merge for the regression Gate: read the probe report and
# the JUnit XML that pytest wrote, then print one raw report. Test statistics
# come from the XML file, never from parsing pytest's human-readable summary.
MERGE = r'''
import json, sys, xml.etree.ElementTree as ET

report = json.load(open(sys.argv[1]))
junit, rc = sys.argv[2], int(sys.argv[3])
report["pytestExit"] = rc
counts = {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0,
          "xfailed": 0, "xpassed": 0}
try:
    root = ET.parse(junit).getroot()
except Exception:
    report["junitPresent"] = False
else:
    report["junitPresent"] = True
    suites = [root] if root.tag == "testsuite" else list(root)
    for suite in suites:
        if suite.tag == "testsuite" and suite.get("tests"):
            counts["collected"] += int(suite.get("tests"))
        for case in suite.iter("testcase"):
            if not suite.get("tests"):
                counts["collected"] += 1
            outcome = "passed"
            for entry in case:
                if entry.tag == "skipped":
                    outcome = "skipped"
                elif entry.tag in ("failure", "error"):
                    outcome = "failed" if entry.tag == "failure" else "errors"
            counts[outcome] += 1
report["tests"] = counts
print(json.dumps(report, sort_keys=True))
'''


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
#
# The first three are the Gate's own toolchain values. The rest belong to
# `scripts/setup-python-env.sh`, which creates the environment those three point
# at: the wheel directory, the index for the small dependencies, the interpreter
# version, the checkout to install editable, and the extra directory a Conda
# installation must search for environments outside its own prefix. They live in
# the same file because one machine file is the whole configuration; a shell
# export is invisible in review and leaves no record of what was set. The Gate
# ignores them and reads only its own three.
TOOLCHAIN_KEYS = (
    'TILELANG_HIVM_CANN_SETUP',
    'TILELANG_HIVM_CONDA_SH',
    'TILELANG_HIVM_CONDA_ENV',
    # The Late-hivm compiler the HIVM codegen calls. It lives here, not in the
    # launching shell, for the same reason CANN does: it names where this
    # machine's compiler is, and a shell export leaves no record in review.
    'TILELANG_HIVM_HIVM_COMPILER',
    'TILELANG_HIVM_HIVM_TARGET',
    'TILELANG_HIVM_CONDA_ENVS_PATH',
    'TILELANG_HIVM_WHEEL_DIR',
    'TILELANG_HIVM_PIP_INDEX',
    'TILELANG_HIVM_PYTHON_VERSION',
    'TILELANG_HIVM_SOURCE_ROOT',
    # Container identity. It lives here, beside the toolchain, because the same
    # `env.sh` is the one file a machine has to fill in and review: the container
    # the Gate addresses must be the container `docker/config.env` created. Two
    # sources let one run address a stopped container while another agent
    # addressed the live one, and neither record showed the disagreement.
    'TILELANG_HIVM_CONTAINER_NAME',
    'TILELANG_HIVM_CONTAINER_USER',
    'TILELANG_HIVM_CONTAINER_HOME',
)
# The container identity keys. All three are required: a container name without a
# user and HOME cannot be entered, and a user or HOME without a name names no
# container. The Gate refuses a partial identity instead of guessing the rest.
CONTAINER_KEYS = (
    'TILELANG_HIVM_CONTAINER_NAME',
    'TILELANG_HIVM_CONTAINER_USER',
    'TILELANG_HIVM_CONTAINER_HOME',
)
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
                                         'TILELANG_HIVM_HIVM_COMPILER',
                                         'TILELANG_HIVM_HIVM_TARGET',
                                         *CONTAINER_KEYS))
# The names the container-side process actually reads. The machine file spells
# them with the Adapter prefix so one file carries the whole machine; TileLang
# reads the bare names, so the Gate forwards the value under the bare name.
HIVM_COMPILER_VAR = 'HIVM_COMPILER'
HIVM_TARGET_VAR = 'HIVM_TARGET'
# `tilelang/contrib/hivm.py` treats HIVM_TARGET as an assertion, not an
# override: the emitted MLIR, the compiler's `--target` and the output ELF must
# all name one device, so any other value is refused before the compiler runs.
# The Gate repeats that rule on the Host so a mistyped target is a configuration
# error instead of a Gate failure.
HIVM_DEFAULT_TARGET = 'Ascend950PR_9579'
# Where the machine toolchain file lives. It is never committed: `env.sh` is
# git-ignored because it names one machine's paths. The search order is:
#
#   1. $TILELANG_HIVM_ENV_FILE, explicit. It must exist when set.
#   2. <checkout>/adapters/tilelang-hivm/env.sh, next to the committed
#      env.example.sh. Found by walking up from this file, so the same file
#      resolves from a checkout and from an installed Pack inside one. Both
#      TileLang Gates read that one file, because both use the TILELANG_HIVM_
#      prefix.
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


# The committed initialization sequence. The Gate inlines this file's text into
# the container-side script, because the Pack tree is not bind-mounted into the
# build container. Its content is pinned by the manifest, so the sequence a Gate
# runs and the sequence a human sources are one reviewed file rather than two
# copies that can drift apart.
INIT_SCRIPT = Path(__file__).resolve().with_name('init-toolchain.sh')
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

    A missing default file is not an error: CANN and Conda are optional. An
    explicitly selected file that is missing or malformed is an error. Exporting
    the three toolchain variables in the launching shell has no effect, because
    a stale export must never mask an edit to the file.
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


def env_file_values() -> dict[str, str]:
    """Every key the machine file defines, parsed once for the whole Gate.

    The file is data, not a script: `_parse_env_file` rejects a line that is not
    `NAME=value`, an unknown key, and a duplicated key, so reading it never runs
    Host code. An absent default file is `{}`; an explicitly selected file that
    is missing is already refused by `resolved_env_file`.
    """
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


def container_identity() -> tuple[str, str, str, Path | None]:
    """The container the Gate addresses, read from the machine file.

    The name, user and HOME come from `env.sh`, which is also where the machine
    records where that container's toolchain lives. One file, one answer: the
    container `docker/config.env` created and the container this Gate enters
    cannot drift apart, and two agents in one run cannot pick two containers.

    The launching shell is not consulted. Exporting
    `TILELANG_HIVM_CONTAINER_NAME` used to be the only way to set it, so a shell
    that carried a stale export silently addressed a stopped container while an
    `env.sh` beside it named the live one. A stale export must never mask the
    file, for the same reason it must not mask the CANN and Conda values.
    """
    path = resolved_env_file()
    values = env_file_values()
    missing = [name for name in CONTAINER_KEYS if not values.get(name)]
    if missing:
        where = str(path) if path is not None else 'no machine file was found'
        raise BridgeError(
            f'{where} does not define {", ".join(missing)}. Container identity is read '
            f'from that file only; copy env.example.sh to env.sh beside it, set the '
            f'three TILELANG_HIVM_CONTAINER_* keys to the container docker/config.env '
            f'created, and run the Gate again.')
    container = values['TILELANG_HIVM_CONTAINER_NAME'].strip()
    user = values['TILELANG_HIVM_CONTAINER_USER'].strip()
    home = values['TILELANG_HIVM_CONTAINER_HOME'].strip()
    if not SAFE_NAME.fullmatch(container) or not SAFE_NAME.fullmatch(user):
        raise BridgeError('container name and user must be plain names, not option-like strings')
    if not home.startswith('/'):
        raise BridgeError(f'TILELANG_HIVM_CONTAINER_HOME must be an absolute container path: {home!r}')
    if any('\n' in value or '\x00' in value for value in (container, user, home)):
        raise BridgeError('container identity values must not contain a newline or NUL')
    return container, user, home, path


def conda_envs_path_from_env_file() -> Path | None:
    """The extra Conda environment directory, when the machine file names one.

    A Conda installation searches only the environments under its own prefix. A
    user's environments normally live in `~/.conda/envs`, so without this value
    `conda activate` fails with `EnvironmentNameNotFound: Could not find conda
    environment: <name>` even though `conda env list` shows the environment.
    Kept separate from `toolchain_from_env_file` so the three-value contract the
    Gates already use stays unchanged.
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


def hivm_from_env_file() -> tuple[Path | None, str | None]:
    """The Late-hivm compiler and target the container process must see.

    TileLang reads `HIVM_COMPILER` from the process environment
    (`tilelang/contrib/hivm.py`), and `docker exec` starts a process with only
    the variables it is given. This Adapter forwarded just `HOME`, so a compiler
    set on the Host never reached the container: `testing/ascend/codegen/
    test_hivm_build.py` then skipped every test that needs it, and a run that
    proved nothing could still look green. Forwarding the value explicitly is
    what makes that skip impossible to mistake for a pass.

    The value comes from the machine file and never from the launching shell,
    for the same reason CANN and the container identity do: an export is
    invisible in review and a stale one silently describes a compiler that is no
    longer installed. Both values are CONTAINER paths, because they are checked
    and used inside the container.

    Returning `(None, None)` is legitimate: a machine without the compiler
    forwards nothing, and the codegen tests keep skipping honestly. A value that
    is set but unusable is an error, so a typo cannot silently degrade to that
    skip.
    """
    path = resolved_env_file()
    if path is None or not path.exists():
        return None, None
    try:
        values = _parse_env_file(path.read_text(encoding='utf-8'), path)
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc

    raw_compiler = values.get('TILELANG_HIVM_HIVM_COMPILER', '').strip()
    raw_target = values.get('TILELANG_HIVM_HIVM_TARGET', '').strip()
    if not raw_compiler:
        if raw_target:
            raise BridgeError(
                f'{path}: TILELANG_HIVM_HIVM_TARGET is set but '
                f'TILELANG_HIVM_HIVM_COMPILER is empty; a target names a device for a '
                f'compiler that is not configured')
        return None, None
    compiler = Path(raw_compiler)
    if not compiler.is_absolute():
        raise BridgeError(f'{path}: TILELANG_HIVM_HIVM_COMPILER must be an absolute '
                          f'container path: {raw_compiler!r}')
    if '\n' in raw_compiler or '\x00' in raw_compiler:
        raise BridgeError(f'{path}: TILELANG_HIVM_HIVM_COMPILER contains an illegal character')
    if raw_target and raw_target != HIVM_DEFAULT_TARGET:
        raise BridgeError(
            f'{path}: TILELANG_HIVM_HIVM_TARGET is {raw_target!r}, but the HIVM codegen '
            f'emits for {HIVM_DEFAULT_TARGET!r} only. The module target, the compiler '
            f'`--target` and the output ELF must name one device, so a different value is '
            f'refused here rather than silently retargeting the build.')
    return compiler, (raw_target or None)


def provisioning_from_env_file() -> dict[str, str]:
    """The provisioning values from the same toolchain file, as raw strings.

    `setup-python-env.sh` is handed these values, so a machine declares its
    wheel directory, index and interpreter version in the one file it already
    edits rather than in a shell export that leaves no record. The Gate does not
    call this: it needs none of these to run a build or a test.

    An absent or empty file yields an empty mapping, and the provisioning script
    then reports which value it still needs. A malformed file is an error here,
    because a provisioning run that silently proceeded on defaults would create
    an environment other than the one the file describes.
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


def build_subdir() -> str:
    """A single relative directory name; traversal and absolute paths are refused."""
    if 'TILELANG_HIVM_BUILD_SUBDIR' in os.environ and not os.environ['TILELANG_HIVM_BUILD_SUBDIR'].strip():
        raise BridgeError('TILELANG_HIVM_BUILD_SUBDIR is set but empty')
    value = env_value('TILELANG_HIVM_BUILD_SUBDIR', default='build') or 'build'
    if not SAFE_NAME.fullmatch(value):
        raise BridgeError(f'TILELANG_HIVM_BUILD_SUBDIR is not a safe relative directory name: {value!r}')
    return value


def ninja_jobs() -> str:
    value = env_value('TILELANG_HIVM_NINJA_JOBS')
    if value is None:
        # Let the container decide from its own CPU count and cgroup limits.
        return '$(nproc)'
    if not value.isdigit() or int(value) < 1:
        raise BridgeError(f'TILELANG_HIVM_NINJA_JOBS must be a positive integer: {value!r}')
    return value


def parse_argv(argv: list[str]) -> tuple[str, int]:
    """Optional leading `--timeout SECONDS`, then exactly one known command."""
    value = None
    if argv[:1] == ['--timeout']:
        # An explicit `--timeout` is authoritative; it is what the manifest passes.
        if len(argv) < 3:
            raise BridgeError('--timeout requires a value and a command')
        value, argv = argv[1], argv[2:]
    else:
        # Only a bare invocation falls back to the environment.
        value = env_value('TILELANG_HIVM_GATE_TIMEOUT', default='1500')
    if value is None or not value.isdigit() or int(value) < 1:
        raise BridgeError(f'gate timeout must be a positive integer number of seconds: {value!r}')
    if len(argv) != 1 or argv[0] not in COMMANDS:
        raise BridgeError(f'expected exactly one command from {COMMANDS}: {argv!r}')
    return argv[0], int(value)


def docker_exec_prefix(container: str, user: str, home: str, workdir: str,
                       forwarded: dict[str, str] | None = None) -> list[str]:
    """The fixed `docker exec` argv, plus the few variables the container needs.

    `docker exec` starts a process that inherits nothing from this Gate's own
    environment, so every variable the container-side program reads has to be
    named here. `HOME` is one; the Late-hivm compiler and target are the others,
    because `docker exec` silently dropping them turned "the codegen tests
    skipped" into something a reader could mistake for "they passed".

    Only values the Gate resolved from the machine file are forwarded. Nothing is
    copied from `os.environ`, so an export in the launching shell cannot change
    what the container measures.
    """
    # Fixed argv. No `-t` and no interactive terminal is used.
    prefix = ['docker', 'exec', '-u', user, '-e', f'HOME={home}']
    for name, value in sorted((forwarded or {}).items()):
        prefix.extend(['-e', f'{name}={value}'])
    return [*prefix, '-w', workdir, container]


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
    """Resolve the Host source root to a container path through bind mounts.

    The most specific covering bind mount wins: candidates are ranked by the
    depth of the bound *source* directory, so a bind of the worktree itself
    beats a bind of the user's home. Equally specific candidates with different
    container paths are an error rather than a coin flip; a Host root that no
    bind mount covers is refused outright.
    """
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
    """Return `dev:ino` of `path` as observed inside the container.

    `stat` is used rather than Python: the interpreter is only on the Conda
    PATH, which is not yet initialized at this point, and the proof that both
    sides see one tree must not depend on that initialization succeeding.
    """
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


def require_conda_pair(conda_sh: Path | None, conda_env: str | None) -> None:
    """CONDA_SH and CONDA_ENV are both-or-neither; a half configuration fails closed.

    Silently ignoring one half would run the tests against a different
    interpreter than the one pytest was configured for, which is exactly the
    failure the Adapter is supposed to prevent.
    """
    if (conda_sh is None) != (conda_env is None):
        raise BridgeError('TILELANG_HIVM_CONDA_SH and TILELANG_HIVM_CONDA_ENV must be provided '
                          'together; set both or neither')


def init_snippet() -> str:
    """The committed init script's text, for inlining into the container script."""
    try:
        return INIT_SCRIPT.read_text(encoding='utf-8').rstrip('\n')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {INIT_SCRIPT}: {exc}') from exc


def init_args(cann: Path | None, conda_sh: Path | None, conda_env: str | None,
              conda_envs_path: Path | None = None) -> str:
    """The toolchain values as positional arguments; `-` means "not set"."""
    values = (str(cann) if cann is not None else '-',
              str(conda_sh) if conda_sh is not None else '-',
              conda_env if conda_env is not None else '-',
              str(conda_envs_path) if conda_envs_path is not None else '-')
    return ' '.join(shlex.quote(value) for value in values)


def container_script(source_root: str, build: str, jobs: str, timeout: int,
                     cann: Path | None, conda_sh: Path | None, conda_env: str | None,
                     conda_envs_path: Path | None = None,
                     hivm_compiler: Path | None = None) -> str:
    """Fixed container-side Bash: initialize the toolchain, then run one command.

    Only the Gate-owned part is written here. The toolchain sequence lives in
    `scripts/init-toolchain.sh`, which is inlined and then called with the three
    values already validated on the Host, so a human sourcing that file and a
    Gate running it execute the same reviewed initialization. That file returns
    non-zero instead of exiting, which is why `fail` turns its result into the
    Gate's own exit 3.

    Neither CANN nor Conda is required: both unset means "configure nothing and
    continue", because the image already provides cmake, ninja and python3.
    Only a half Conda configuration fails closed. `set -u` is intentionally not
    used, because the vendor scripts read variables that are legitimately unset
    on first entry.

    The Late-hivm compiler is checked here as well as forwarded. `docker exec`
    carries the value, but a path the container cannot execute is a broken
    configuration, and the honest outcome is the Gate's own exit 3 rather than a
    pytest skip that reads like a pass. An unconfigured compiler is not checked:
    the codegen tests then skip for a stated reason, which is accurate.
    """
    lines = [
        'set -o pipefail',
        f'fail() {{ echo "{PROG}: $*" >&2; exit 3; }}',
        'export PYTHONDONTWRITEBYTECODE=1',
        init_snippet(),
        f'init_toolchain {init_args(cann, conda_sh, conda_env, conda_envs_path)} '
        '|| fail "toolchain initialization failed"',
    ]
    if hivm_compiler is not None:
        quoted = shlex.quote(str(hivm_compiler))
        lines.append(f'[ -x {quoted} ] '
                     f'|| fail {shlex.quote(f"{HIVM_COMPILER_VAR} is not an executable in the container: {hivm_compiler}")}')
        lines.append(f'export {HIVM_COMPILER_VAR}={quoted}')
    # Messages that carry an env-file value are single-quoted literals: inside
    # double quotes a `$(...)` in the value would execute in the container,
    # which would make the machine file code rather than data.
    lines.append(f'cd {shlex.quote(source_root)} '
                 f'|| fail {shlex.quote(f"cannot enter {source_root}")}')
    lines.append(f'[ -d {shlex.quote(build)} ] '
                 f'|| fail {shlex.quote(f"no {build} directory under {source_root}; configure the build directory first")}')
    lines.append('command=$1; shift')
    lines.append('case "$command" in')
    # Build: an incremental Ninja run of the one stable phony target. The build
    # tree writes only inside the build directory, which the repository ignores,
    # so the Harness source-identity check still holds afterwards.
    lines.append('  build)')
    lines.append(f'    [ -f {shlex.quote(build)}/build.ninja ] '
                 f'|| fail {shlex.quote(f"no {build}/build.ninja; initialize the build directory first")}')
    lines.append(f'    timeout --signal=TERM --kill-after={KILL_AFTER_SECONDS} {timeout} '
                 f'ninja -C {shlex.quote(build)} -j {jobs} tilelang >&2')
    lines.append('    rc=$?')
    lines.append('    printf \'{"command": "build", "exitCode": %d}\\n\' "$rc"')
    lines.append('    exit $rc')
    lines.append('    ;;')
    # Provenance: report what the container would load. This command runs the
    # probe only; the bridge compares the digests against the Host.
    lines.append('  provenance)')
    lines.append('    python3 - provenance "$PWD" <<\'TILELANG_HIVM_PROBE\'')
    lines.extend(PROBE.strip('\n').splitlines())
    lines.append('TILELANG_HIVM_PROBE')
    lines.append('    ;;')
    # Regression: probe, then the fixed test files, then a report built from the
    # JUnit XML. Test statistics never come from pytest's summary text.
    lines.append('  regression)')
    lines.append(f'    [ -f {shlex.quote(build)}/build.ninja ] '
                 f'|| fail {shlex.quote(f"no {build}/build.ninja; initialize the build directory first")}')
    for test in REGRESSION_TESTS:
        lines.append(f'    [ -f {shlex.quote(test)} ] '
                     f'|| fail {shlex.quote(f"regression test file is missing: {test}")}')
    lines.append('    tmp=$(mktemp -d) || fail "cannot create a temporary directory"')
    lines.append('    python3 - regression "$PWD" >"$tmp/report.json" <<\'TILELANG_HIVM_PROBE\'')
    lines.extend(PROBE.strip('\n').splitlines())
    lines.append('TILELANG_HIVM_PROBE')
    lines.append('    [ -s "$tmp/report.json" ] || fail "container provenance probe produced no facts"')
    lines.append('    tests="' + ' '.join(REGRESSION_TESTS) + '"')
    lines.append(f'    timeout --signal=TERM --kill-after={KILL_AFTER_SECONDS} {timeout} '
                 'python3 -m pytest -q -p no:cacheprovider '
                 '--junitxml="$tmp/junit.xml" $tests >&2')
    lines.append('    rc=$?')
    lines.append('    python3 - "$tmp/report.json" "$tmp/junit.xml" "$rc" <<\'TILELANG_HIVM_MERGE\'')
    lines.extend(MERGE.strip('\n').splitlines())
    lines.append('TILELANG_HIVM_MERGE')
    lines.append('    mrc=$?')
    lines.append('    [ $mrc -eq 0 ] || fail "container could not produce a Gate report"')
    lines.append('    exit $rc')
    lines.append('    ;;')
    lines.append('  *) fail "unknown command $command" ;;')
    lines.append('esac')
    return '\n'.join(lines) + '\n'


def relative_inside(relative: object, *, what: str) -> str:
    """Every artifact this Adapter attests must live inside the source root."""
    if not isinstance(relative, str) or not relative or relative.startswith('/') \
            or '..' in Path(relative).parts:
        raise BridgeError(f'{what} resolves outside the source root: {relative!r}')
    return relative


def host_digest(host_root: Path, relative: str) -> str:
    import hashlib
    path = (host_root / relative).resolve()
    if not path.is_relative_to(host_root.resolve()):
        raise BridgeError(f'artifact {relative!r} resolves outside the source root')
    if not path.is_file():
        raise BridgeError(f'artifact {relative!r} is not a readable file on the Host')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate report key')
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError('non-finite report number')


def read_report(stdout: str, *, command: str) -> dict:
    """The container prints its raw report as the only JSON object it emits."""
    lines = [line for line in stdout.splitlines() if line.startswith('{')]
    if not lines:
        raise BridgeError(f'container produced no {command} report')
    try:
        report = json.loads(lines[-1], object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
    except ValueError as exc:
        raise BridgeError(f'container produced an unparsable {command} report') from exc
    if not isinstance(report, dict) or report.get('command') != command:
        raise BridgeError(f'container {command} report has the wrong shape')
    return report


def identity_observations(facts: dict, host_root: Path) -> dict:
    """Compare what the container loaded with the bytes the Host can read.

    Host and container share one bind mount, and that was proved by device and
    inode before this ran. Hashing the same relative path on both sides then
    says something stronger than "a file with this name exists": the imported
    Python module and every native shared object the built package points at are
    the very bytes of the reviewed checkout.
    """
    observations = {}
    module_rel = relative_inside(facts.get('pythonModuleRel'), what='imported Python module')
    container_module = facts.get('pythonModuleSha256')
    observations['pythonModuleRel'] = module_rel
    module_digest = host_digest(host_root, module_rel)
    observations['importedModuleSha256MatchesHost'] = (
        isinstance(container_module, str) and container_module == module_digest)
    observations['artifacts'] = [
        {'path': str((host_root / module_rel).resolve()), 'sha256': module_digest}]

    native = facts.get('nativeLibs')
    if not isinstance(native, dict) or not native:
        observations['nativeLibsPresent'] = False
        observations['loadedNativeSha256MatchesHost'] = False
        return observations
    observations['nativeLibsPresent'] = True
    observations['nativeLibCount'] = len(native)
    matched = True
    for relative, digest in sorted(native.items()):
        if not isinstance(digest, str):
            matched = False
            break
        relative = relative_inside(relative, what='native library')
        actual = host_digest(host_root, relative)
        observations['artifacts'].append(
            {'path': str((host_root / relative).resolve()), 'sha256': actual})
        if actual != digest:
            matched = False
    observations['loadedNativeSha256MatchesHost'] = matched
    return observations


def regression_observations(report: dict, host_root: Path) -> dict:
    """Identity plus test statistics. A count that cannot be trusted is false."""
    facts = report.get('facts')
    if not isinstance(facts, dict):
        raise BridgeError('container regression report has no provenance facts')
    observations = identity_observations(facts, host_root)
    observations['ascendBackendCompiled'] = facts.get('ascendBackendCompiled') is True
    observations['devMode'] = facts.get('devMode') is True
    tests = report.get('tests')
    if not isinstance(tests, dict):
        tests = {}
    keys = ('collected', 'passed', 'failed', 'errors', 'skipped')
    valid_counts = all(type(tests.get(key)) is int and tests[key] >= 0 for key in keys)
    collected, passed, failed, errors, skipped = (
        [tests[key] for key in keys] if valid_counts else [0] * len(keys))
    valid_counts = valid_counts and collected == passed + failed + errors + skipped
    observations['junitPresent'] = report.get('junitPresent') is True
    observations['testsCollected'] = collected
    observations['testsPassed'] = passed
    observations['testsFailed'] = failed + errors
    observations['testsSkipped'] = skipped
    observations['testsCollectedPositive'] = valid_counts and collected > 0
    # A suite that collected nothing, or skipped everything, proves no behaviour,
    # so it can never be a PASS even when pytest exits 0.
    observations['notAllSkipped'] = valid_counts and collected > 0 and skipped < collected
    observations['allPassed'] = valid_counts and passed > 0 and failed == 0 and errors == 0
    observations['pytestExitZero'] = type(report.get('pytestExit')) is int and report['pytestExit'] == 0
    # The suite is gated at collection time by `testing/ascend/conftest.py` when
    # the Ascend backend is missing; the backend flag being true is the direct
    # witness that the directory was not ignored wholesale.
    return observations


def emit(observations: dict) -> None:
    print(json.dumps(observations, sort_keys=True))


def check_config() -> int:
    """Report whether this machine is configured, without entering the container.

    Preflight runs this before any Gate process starts, so a missing container
    identity or an unreadable toolchain file blocks the task in `BLOCKED_ENV`
    instead of surfacing later as a Gate failure. It performs exactly the Host
    work `main` does before its first `docker` call, and nothing else: no
    `docker inspect`, no `docker exec`, no file written.

    Exit 0 means the configuration resolved. Every message goes to stderr, so
    the caller can quote the last line as the refusal.
    """
    container, user, home, identity_file = container_identity()
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError(f'TILELANG_HIVM_CONDA_ENV is not a plain environment name: {conda_env!r}')
    build = build_subdir()
    hivm_compiler, hivm_target = hivm_from_env_file()
    print(f'{PROG}: configuration ready: identity file {identity_file} container={container} '
          f'user={user} home={home} build={build} '
          f'hivmCompiler={hivm_compiler if hivm_compiler is not None else "unset"} '
          f'hivmTarget={hivm_target if hivm_target is not None else "default"}', file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    if list(argv) == ['--check-config']:
        return check_config()
    host_root = Path.cwd()
    container, user, home, identity_file = container_identity()

    command, timeout = parse_argv(list(argv))
    build = build_subdir()
    jobs = ninja_jobs()
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError(f'TILELANG_HIVM_CONDA_ENV is not a plain environment name: {conda_env!r}')
    conda_envs_path = conda_envs_path_from_env_file()
    hivm_compiler, hivm_target = hivm_from_env_file()

    mapped = map_host_to_container(host_root, container_mounts(container), container)
    declared = env_path('TILELANG_HIVM_CONTAINER_SOURCE')
    if declared is not None and os.path.realpath(str(declared)) != os.path.realpath(str(mapped)):
        raise BridgeError(f'TILELANG_HIVM_CONTAINER_SOURCE {declared} does not match the bind-mount '
                          f'mapping {mapped} of host source root {host_root}')
    container_root = str(declared if declared is not None else mapped)
    # The container process reads these two by their bare names, so they are the
    # only variables copied from the machine file into `docker exec`.
    forwarded = {}
    if hivm_compiler is not None:
        forwarded[HIVM_COMPILER_VAR] = str(hivm_compiler)
    if hivm_target is not None:
        forwarded[HIVM_TARGET_VAR] = hivm_target
    prefix = docker_exec_prefix(container, user, home, container_root, forwarded)
    verify_same_tree(host_root, prefix, container_root)

    print(f'{PROG}: identity file {identity_file}', file=sys.stderr)
    print(f'{PROG}: container={container} user={user} source={container_root} build={build} '
          f'jobs={jobs} timeout={timeout}s command={command} '
          f'{HIVM_COMPILER_VAR}={hivm_compiler if hivm_compiler is not None else "unset"}',
          file=sys.stderr)
    script = container_script(container_root, build, jobs, timeout, cann, conda_sh, conda_env,
                              conda_envs_path, hivm_compiler)
    completed = subprocess.run([*prefix, 'bash', '-c', script, 'bash', command], capture_output=True, text=True)
    if completed.stderr:
        sys.stderr.write(completed.stderr)

    if command == 'provenance':
        report = read_report(completed.stdout, command=command)
        facts = report.get('facts')
        if not isinstance(facts, dict):
            raise BridgeError('container provenance report has no facts')
        observations = identity_observations(facts, host_root)
        emit(observations)
        return completed.returncode

    if command == 'build':
        report = read_report(completed.stdout, command=command)
        emit({'exitCodeZero': type(report.get('exitCode')) is int and report['exitCode'] == 0})
        return completed.returncode

    report = read_report(completed.stdout, command=command)
    observations = regression_observations(report, host_root)
    emit(observations)
    # Preserve real failures and reject malformed raw facts even if pytest exited 0.
    accepted = all(observations[key] is True for key in (
        'importedModuleSha256MatchesHost', 'loadedNativeSha256MatchesHost',
        'nativeLibsPresent', 'ascendBackendCompiled', 'devMode', 'junitPresent',
        'testsCollectedPositive', 'notAllSkipped', 'allPassed', 'pytestExitZero'))
    return completed.returncode or (0 if accepted else 1)


if __name__ == '__main__':
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BridgeError as error:
        print(f'{PROG}: {error.message}', file=sys.stderr)
        raise SystemExit(BridgeError.EXIT)
