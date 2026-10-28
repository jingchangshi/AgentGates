#!/usr/bin/env python3
"""Run one AscendNPU-IR Ninja target inside the build container via `docker exec`.

The Harness Recipe runs this file with the project source root as its working
directory. Everything about the container is supplied at run time through
environment variables, so this Adapter never hard-codes a container name, user,
host path, repository name or Git branch.

The bridge is deliberately narrow:

1. the Host source root is the process working directory;
2. the container is expected to bind-mount that directory; the mapping is
   derived from `docker inspect` bind mounts and then *proved* by comparing the
   device/inode pair seen by both sides, so the container builds exactly the
   bytes the Host sees, including uncommitted edits;
3. exactly one fixed `docker exec` argv is issued; no shell string is built
   from user input (values are always `shlex.quote`d);
4. the container-side command runs under `timeout --signal=TERM
   --kill-after=15`, and the container exit code is returned unchanged (124 on
   timeout, 125/126/127 for Docker-level errors);
5. the timeout ceiling is fixed by the manifest: each Recipe passes `--timeout`
   explicitly, so the Adapter's own value is stable and the Host's outer Recipe
   timeout always exceeds `timeout + 15s`.

Machine configuration comes from one file, not from the launching shell:

  ASCENDNPU_ENV_FILE         explicit path of the machine `env.sh`. Must exist
                             when set.

Without it the file is found by walking up from this script to the checkout that
owns it, and reading `<checkout>/adapters/ascendnpu-ir/env.sh`, next to the
committed env.example.sh. That resolves the same file from a checkout and from an
installed Pack inside one, so a human or an agent edits the configuration in the
repository, beside the template it was copied from. There is no other location:
a machine that keeps the file elsewhere names it with ASCENDNPU_ENV_FILE, so no
Gate can read a file nobody reviewed.

The file is the single source for every machine fact this Adapter needs, exactly
as it already is for the two TileLang Adapters:

  ASCENDNPU_CONTAINER_NAME   required, name of a running build container
  ASCENDNPU_CONTAINER_USER   required, user to run as inside the container
  ASCENDNPU_CONTAINER_HOME   required, HOME inside the container
  ASCENDNPU_BUILD_SUBDIR     optional, build directory name (default: build)
  ASCENDNPU_NINJA_JOBS       optional, ninja -j value (default: container CPUs)
  ASCENDNPU_CANN_SETUP       optional, CANN set_env.sh
  ASCENDNPU_CONDA_SH         optional, conda.sh
  ASCENDNPU_CONDA_ENV        optional, conda environment to activate
  ASCENDNPU_CCEC_BIN         optional, ccec bin directory used by the full build
  ASCENDNPU_CONTAINER_SOURCE optional, explicit container path of the source
                             root; when set it must match the bind-mount
                             mapping, when unset the mapping is used directly

One value still comes from the launching environment because it selects which
file to read, and one because it bounds the invocation rather than describing
the machine:

  ASCENDNPU_ENV_FILE         the selector above
  ASCENDNPU_GATE_TIMEOUT     optional, container-side incremental timeout in
                             seconds; used ONLY when no `--timeout` is passed

Exporting a machine variable has no effect: a stale export must never mask a
file edit, and a machine/shell disagreement is refused instead of silently
picking one side.

A build tree that does not exist is built from scratch
------------------------------------------------------

The Gate has always required `build/build.ninja` and refused otherwise, so a
fresh checkout could never be gated at all - the only from-zero path was a
manual helper the Gate did not call. It now examines the configured build
directory inside the container:

* `build/build.ninja` exists -> the pinned incremental Ninja run, unchanged,
  with the `--timeout` the manifest passes;
* the directory has neither `build.ninja` nor `CMakeCache.txt` -> the fixed
  full build in `scripts/full-build.sh`, which configures and builds with the
  same command a human runs, under its own larger bound;
* `CMakeCache.txt` without `build.ninja` -> refused. That tree is neither
  incrementally buildable nor safe to reconfigure without deleting state, and a
  Gate must not delete an operator's build directory.

The full build supersedes the incremental targets (it builds every target and
installs the published products), so the Gate proves the products exist
afterwards rather than claiming the named targets ran.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PROG = 'ascendnpu-ir-docker-gate'
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
# Grace period between `timeout` sending TERM and the container-side SIGKILL. It is
# deliberately short: the Harness Recipe timeout is the outer bound, so the whole
# `timeout + KILL_AFTER_SECONDS` envelope must still fit inside it. The manifest
# pins both numbers (840/1140s here) and the Recipes always pass `--timeout`
# explicitly, so `ASCENDNPU_GATE_TIMEOUT` can never silently widen the envelope.
KILL_AFTER_SECONDS = 15
# Container-side bound for a from-zero full build. A cold tree compiles LLVM,
# BiShengIR, triton and the shmem template, so the incremental bound (840 s) is
# not in the same league; the manifest passes its own larger value and the
# worker's RuntimeMaxSec is sized above both.
FULL_BUILD_TIMEOUT = 21600


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


# Keys the machine file may define. Anything else is an error: a typo such as
# ASCENDNPU_CANN_SETUP= (double underscore) must not be ignored.
TOOLCHAIN_KEYS = ('ASCENDNPU_CANN_SETUP', 'ASCENDNPU_CONDA_SH', 'ASCENDNPU_CONDA_ENV')
# Keys the same file may also define but the incremental Gate does not use.
# scripts/full-build.sh reads ASCENDNPU_CCEC_BIN to configure a fresh tree, and
# the Gate consumes it only on its from-zero branch.
OPTIONAL_FILE_KEYS = ('ASCENDNPU_CCEC_BIN',)
# Container identity. It is a machine fact, so it lives in the machine file: the
# container `docker/config.env` created and the container a Gate enters cannot
# drift apart, and two agents in one run cannot pick two containers. The three
# TileLang Adapters already worked this way; Ascend read the launching shell, so
# the same run could gate a tree in a container nobody had created.
CONTAINER_KEYS = ('ASCENDNPU_CONTAINER_NAME', 'ASCENDNPU_CONTAINER_USER',
                  'ASCENDNPU_CONTAINER_HOME')
# Build selection. These decide which tree the Gate builds in and how wide it
# builds, so they are machine facts too; the engine reads the same file for the
# same reason, because a lock on `build` must not cover a run into `build-fast`.
BUILD_KEYS = ('ASCENDNPU_BUILD_SUBDIR', 'ASCENDNPU_NINJA_JOBS')
# The container path of the source root, when the bind-mount mapping is not the
# one this machine uses. It names a location inside the container, so it is a
# machine fact exactly like the container identity: a shell export of it decides
# which tree the Gate runs in, and a stale one would gate a different directory
# than the engine recorded. The mapping is still verified against the mounts, so
# this key can only confirm a path, never invent one.
SOURCE_KEYS = ('ASCENDNPU_CONTAINER_SOURCE',)
# Every key the file may define. A machine file that sets a key outside this set
# is refused rather than half-honoured.
FILE_KEYS = TOOLCHAIN_KEYS + OPTIONAL_FILE_KEYS + CONTAINER_KEYS + BUILD_KEYS + SOURCE_KEYS
# Machine keys whose shell export must never disagree with the file. The file
# wins for its own keys; a conflicting export is refused so that neither side can
# silently steer a Gate the other side does not describe.
MACHINE_KEYS = CONTAINER_KEYS + BUILD_KEYS + SOURCE_KEYS
# Where the machine toolchain file lives. It is never committed: `env.sh` is
# git-ignored, like `docker/config.env`, because it names one machine's paths.
# The search order is:
#
#   1. $ASCENDNPU_ENV_FILE, explicit. It must exist: the operator asked for it.
#   2. <checkout>/adapters/ascendnpu-ir/env.sh, next to the committed
#      env.example.sh. Found by walking up from this file, so the same file
#      resolves from a checkout and from an installed Pack inside one.
#
# Step 2 keeps human and agent work in the repository, beside the template the
# file was copied from. A step that finds nothing is not an error: CANN and Conda
# are optional, so an absent file means "configure nothing" and the Gate passes
# nothing to `init_toolchain`.
ADAPTER_DIR = Path(__file__).resolve().parent.parent
CHECKOUT_MARKER = Path('adapters') / 'ascendnpu-ir' / 'env.example.sh'


def checkout_env_file() -> Path | None:
    """The in-repository machine file, when this Pack belongs to a checkout."""
    # Packaged adapters share a local adapters directory, including the NPU sibling.
    packaged = ADAPTER_DIR.parent / 'ascendnpu-ir' / 'env.example.sh'
    if ADAPTER_DIR.parent.name == 'adapters' and packaged.is_file():
        return packaged.with_name('env.sh')
    for ancestor in (ADAPTER_DIR, *ADAPTER_DIR.parents):
        if (ancestor / CHECKOUT_MARKER).is_file():
            return ancestor / CHECKOUT_MARKER.parent / 'env.sh'
    return None


def default_env_file() -> Path | None:
    """The machine file, or None.

    There is no second location: a machine that keeps the file elsewhere names
    it with `ASCENDNPU_ENV_FILE`, so no Gate can read a file nobody
    reviewed.
    """
    return checkout_env_file()

# The committed initialization sequence. The Gate inlines this file's text into
# the container-side script, because the Pack tree is not bind-mounted into the
# build container. Its content is pinned by the manifest, so the sequence a Gate
# runs and the sequence a human sources are one reviewed file rather than two
# copies that can drift apart.
INIT_SCRIPT = Path(__file__).resolve().with_name('init-toolchain.sh')
# The from-zero build command. It is committed and pinned by the manifest, so the
# path a Gate takes on an empty tree and the command a human runs are one file.
FULL_BUILD_SCRIPT = Path(__file__).resolve().with_name('full-build.sh')
# A conservative `NAME=value` grammar. Values may be bare, single- or
# double-quoted. No expansion, no command substitution, no line continuations:
# the file is data, so reading it never executes Host code.
_ASSIGNMENT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')


def _parse_env_file(text: str, path: Path) -> dict[str, str]:
    """Parse the strict `NAME=value` toolchain file; never executes it.

    `#` comments and blank lines are allowed. Every other line must be a plain
    assignment to a known key, so a malformed or misspelled file fails closed
    instead of silently leaving a component unconfigured.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = _ASSIGNMENT.match(line)
        if match is None:
            raise BridgeError(f'{path}: line {number} is not a NAME=value assignment: {raw!r}')
        key, value = match.group(1), match.group(2).strip()
        if key not in FILE_KEYS:
            raise BridgeError(f'{path}: line {number} sets unknown key {key!r}; '
                              f'allowed keys are {", ".join(FILE_KEYS)}')
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

    A missing default file is not an error: CANN and Conda are optional, and a
    machine that needs neither leaves the file absent. An explicitly selected
    file that is missing or malformed is an error, because that means the
    operator asked for configuration that cannot be honoured.
    """
    path, values = machine_values()
    if path is None:
        return None, None, None
    cann_raw, conda_sh_raw = values.get('ASCENDNPU_CANN_SETUP', ''), values.get('ASCENDNPU_CONDA_SH', '')
    conda_env_raw = values.get('ASCENDNPU_CONDA_ENV', '')
    # An explicitly empty value disables that optional component.
    def as_path(name: str, raw: str) -> Path | None:
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            raise BridgeError(f'{path}: {name} must be an absolute path: {raw!r}')
        if '\n' in raw or '\x00' in raw:
            raise BridgeError(f'{path}: {name} contains an illegal character')
        return candidate

    cann = as_path('ASCENDNPU_CANN_SETUP', cann_raw)
    conda_sh = as_path('ASCENDNPU_CONDA_SH', conda_sh_raw)
    conda_env = conda_env_raw or None
    print(f'{PROG}: toolchain file {path}', file=sys.stderr)
    return cann, conda_sh, conda_env


def machine_values() -> tuple[Path | None, dict[str, str]]:
    """The machine file's values, parsed once, and the path they came from."""
    explicit = env_value('ASCENDNPU_ENV_FILE')
    path = Path(explicit).expanduser() if explicit else default_env_file()
    if path is None:
        return None, {}
    if not path.is_absolute():
        raise BridgeError(f'ASCENDNPU_ENV_FILE must be an absolute path: {explicit!r}')
    if not path.exists():
        if explicit:
            raise BridgeError(f'ASCENDNPU_ENV_FILE does not exist: {path}')
        return None, {}
    if not path.is_file():
        raise BridgeError(f'ASCENDNPU_ENV_FILE is not a regular file: {path}')
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc
    return path, _parse_env_file(text, path)


def machine_value(name: str, *, path: Path | None = None,
                  values: dict[str, str] | None = None) -> str | None:
    """One machine value, read from the machine file and nowhere else.

    Every machine fact this Adapter uses - the container it enters, the build
    directory, the Ninja width, the toolchain - has exactly one source: `env.sh`.
    An inherited export is never consulted as a value. That is what makes the
    engine's build lease trustworthy: the engine resolves the same file, so it
    cannot lock `build` while the Gate writes `build-release` because some
    caller had exported a different directory.

    Two failures are still reported rather than ignored, because silently
    dropping a value an operator set is worse than refusing:

    * the file defines a key and the environment exports a *different* value -
      one of the two is stale, and neither order of precedence reveals which;
    * the file does not define a key but the environment exports one - the
      export looks like configuration but has no effect, so it is refused with
      a pointer at the file instead of being obeyed or ignored.

    Sourcing `env.sh` and running a Gate therefore still works: the export then
    equals the file, which is not a conflict.
    """
    if values is None:
        path, values = machine_values()
    defined = name in values
    from_file = values.get(name, '').strip()
    exported = env_value(name)
    if defined and from_file:
        if exported and exported.strip() != from_file:
            raise BridgeError(
                f'{name} is {exported!r} in the launching environment but {from_file!r} in '
                f'{path}; machine values are read from that file only. Remove the export '
                f'or edit the file, so one run cannot use two values.')
        return from_file
    if exported and exported.strip():
        where = str(path) if path is not None else 'no machine file was found'
        raise BridgeError(
            f'{name} is exported as {exported!r}, but machine values are read only from '
            f'{where}. Define it in that file instead of the launching environment.')
    if defined and name in BUILD_KEYS:
        # An empty build directory or job count is a typo, not "use default": the
        # file spells "unset" by leaving the key out.
        raise BridgeError(f'{path}: {name} is set but empty')
    return '' if defined else None


def build_subdir() -> str:
    """A single relative directory name; traversal and absolute paths are refused."""
    path, values = machine_values()
    if 'ASCENDNPU_BUILD_SUBDIR' in os.environ and not os.environ['ASCENDNPU_BUILD_SUBDIR'].strip() \
            and 'ASCENDNPU_BUILD_SUBDIR' not in values:
        # A blank export is a typo, not "use the default": the file spells
        # "unset" by leaving the key out, and the shell spells it by not setting
        # the variable at all.
        raise BridgeError('ASCENDNPU_BUILD_SUBDIR is set but empty')
    value = machine_value('ASCENDNPU_BUILD_SUBDIR', path=path, values=values) or 'build'
    if not SAFE_NAME.fullmatch(value) or value in ('.', '..'):
        raise BridgeError(f'ASCENDNPU_BUILD_SUBDIR is not a safe relative directory name: {value!r}')
    return value


def ninja_jobs() -> str:
    path, values = machine_values()
    if 'ASCENDNPU_NINJA_JOBS' in os.environ and not os.environ['ASCENDNPU_NINJA_JOBS'].strip() \
            and 'ASCENDNPU_NINJA_JOBS' not in values:
        raise BridgeError('ASCENDNPU_NINJA_JOBS is set but empty')
    value = machine_value('ASCENDNPU_NINJA_JOBS', path=path, values=values)
    if not value:
        # Let the container decide from its own CPU count and cgroup limits.
        return '$(nproc)'
    if not value.isdigit() or int(value) < 1:
        raise BridgeError(f'ASCENDNPU_NINJA_JOBS must be a positive integer: {value!r}')
    return value


def container_identity() -> tuple[str, str, str]:
    """The container this Gate addresses, read from the machine file.

    The name, user and HOME are machine facts, so they live beside the CANN and
    Conda values in `env.sh`. Reading them from the launching shell - which is
    what this Adapter used to do - let one run address a stopped container while
    the file named the live one, and let two agents in a single run address two
    different containers, with neither record showing the disagreement.
    """
    path, values = machine_values()
    identity = [machine_value(name, path=path, values=values) for name in CONTAINER_KEYS]
    missing = [name for name, value in zip(CONTAINER_KEYS, identity) if not value]
    if missing:
        where = str(path) if path is not None else 'no machine file was found'
        raise BridgeError(
            f'{where} does not define {", ".join(missing)}. Container identity is read '
            f'from that file; copy env.example.sh to env.sh beside it, set the three '
            f'ASCENDNPU_CONTAINER_* keys to the container docker/config.env created, and '
            f'run the Gate again.')
    container, user, home = identity
    if not SAFE_NAME.fullmatch(container) or not SAFE_NAME.fullmatch(user):
        raise BridgeError('container name and user must be plain names, not option-like strings')
    if not home.startswith('/'):
        raise BridgeError(f'ASCENDNPU_CONTAINER_HOME must be an absolute container path: {home!r}')
    return container, user, home


def timeout_seconds(argv: list[str]) -> tuple[int, int, list[str], bool]:
    """Leading Gate options, then the Ninja argv.

    Two bounds are needed because the two modes are different jobs: an
    incremental Ninja run of a few targets, and a from-zero configure plus full
    build. Reusing the incremental bound would kill a full build long before it
    could finish, which would make "the Gate can build from an empty tree" true
    only on paper.

    `--allow-full-build` is opt-in per recipe, and only the build gate passes it.
    The regression gate is a test: if it could trigger a full build, an unbuilt
    tree would report PASS for a command that never ran. The default is therefore
    "incremental only", and the refusal is what the regression gate reports when
    the tree it needs does not exist.
    """
    incremental = full = None
    allow_full = False
    rest = list(argv)
    while rest[:1] and rest[0].startswith('--'):
        flag = rest[0]
        if flag == '--allow-full-build':
            if allow_full:
                raise BridgeError('--allow-full-build was given twice')
            allow_full = True
            rest = rest[1:]
            continue
        if flag not in ('--timeout', '--full-build-timeout'):
            break
        if len(rest) < 2:
            raise BridgeError(f'{flag} requires a value')
        if flag == '--timeout':
            if incremental is not None:
                raise BridgeError('--timeout was given twice')
            incremental = rest[1]
        else:
            if full is not None:
                raise BridgeError('--full-build-timeout was given twice')
            full = rest[1]
        rest = rest[2:]
    if incremental is None:
        # Only a bare invocation falls back to the environment.
        incremental = env_value('ASCENDNPU_GATE_TIMEOUT', default='1500')
    if full is None:
        full = env_value('ASCENDNPU_FULL_BUILD_TIMEOUT', default=str(FULL_BUILD_TIMEOUT))
    for name, value in (('--timeout', incremental), ('--full-build-timeout', full)):
        if value is None or not value.isdigit() or int(value) < 1:
            raise BridgeError(f'{name} must be a positive integer number of seconds: {value!r}')
    if not rest:
        raise BridgeError('no Ninja target was given')
    return int(incremental), int(full), rest, allow_full


def docker_exec_prefix(container: str, user: str, home: str, workdir: str) -> list[str]:
    # Fixed argv. No `-t` and no interactive terminal is used.
    return ['docker', 'exec', '-u', user, '-e', f'HOME={home}', '-w', workdir, container]


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
    """Return `dev:ino` of `path` as observed inside the container."""
    expression = 'import os,sys;st=os.stat(sys.argv[1]);print("%d:%d" % (st.st_dev, st.st_ino))'
    done = subprocess.run([*prefix, 'bash', '-c',
                           'python3 -c ' + shlex.quote(expression) + ' ' + shlex.quote(path)],
                          capture_output=True, text=True)
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


def init_snippet() -> str:
    """The committed init script's text, for inlining into the container script."""
    try:
        return INIT_SCRIPT.read_text(encoding='utf-8').rstrip('\n')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {INIT_SCRIPT}: {exc}') from exc


def init_args(cann: Path | None, conda_sh: Path | None, conda_env: str | None) -> str:
    """The three toolchain values as positional arguments; `-` means "not set"."""
    values = (str(cann) if cann is not None else '-',
              str(conda_sh) if conda_sh is not None else '-',
              conda_env if conda_env is not None else '-')
    return ' '.join(shlex.quote(value) for value in values)


# The container-side choice between the two modes. A tree with `build.ninja` is
# built incrementally, exactly as before; a tree with neither `build.ninja` nor
# `CMakeCache.txt` is configured and built from scratch by the committed helper;
# a `CMakeCache.txt` without `build.ninja` is refused, because that tree can
# neither be built incrementally nor reconfigured without deleting state the Gate
# did not create.
MODE_PROBE = r'''
mode=incremental
if [ ! -f {build}/build.ninja ]; then
    if [ -e {build}/CMakeCache.txt ]; then
        fail {half}
    fi
    mode=full
fi
'''
# Refusal used by a recipe that must run an existing tree. The regression target
# is a test, not a build: satisfying it with a fresh full build would report PASS
# for a gate whose specified command never ran.
INCREMENTAL_ONLY_REFUSAL = (
    'no {build}/build.ninja under {source}; this recipe runs an existing build '
    'tree and never builds from scratch - run the build gate first')
# Products the full build must have produced, relative to the configured build
# directory - which is where `build-tools/build.sh` puts them (`<build>/bin`,
# with the published set also installed under `<build>/install`). The source
# root never holds them, so only this location is checked.
FULL_BUILD_PRODUCTS = ('bin/bishengir-opt', 'bin/bishengir-compile')


def container_script(source_root: str, build: str, jobs: str, timeout: int,
                     cann: Path | None, conda_sh: Path | None, conda_env: str | None,
                     *, full_timeout: int | None = None, ccec_bin: Path | None = None,
                     targets: tuple[str, ...] = (), allow_full_build: bool = False) -> str:
    """Fixed container-side Bash: initialize the toolchain, then build.

    Only the Gate-owned part is written here. The toolchain sequence lives in
    `scripts/init-toolchain.sh` and the from-zero command in
    `scripts/full-build.sh`; both are inlined and called with values already
    validated on the Host, so a human sourcing those files and a Gate running
    them execute the same reviewed commands. Each returns non-zero instead of
    exiting, which is why `fail` turns a failure into the Gate's own exit 3.

    The order is fixed: initialization, then `cd`, then the mode probe, then
    either the incremental Ninja exec or the full build. The checks for cmake and
    ninja happen inside the initialization, so they resolve from the effective
    PATH, after CANN and `conda activate` have extended it.
    """
    if full_timeout is None:
        full_timeout = FULL_BUILD_TIMEOUT
    # A recipe that is not the build gate runs an existing tree only: the
    # regression target is a test, and satisfying it with a fresh build would
    # report PASS for a command that never ran. Its script therefore carries no
    # full-build branch at all, rather than a branch that fails after the fact.
    lines = [
        'set -o pipefail',
        'fail() { echo "ascendnpu-ir-docker-gate: $*" >&2; exit 3; }',
        init_snippet(),
    ]
    if allow_full_build:
        lines.append(full_build_snippet())
    lines += [
        f'init_toolchain {init_args(cann, conda_sh, conda_env)} '
        '|| fail "toolchain initialization failed"',
        # Messages that carry an env-file value are single-quoted literals:
        # inside double quotes a `$(...)` in the value would execute in the
        # container, which would make the machine file code rather than data.
        f'cd {shlex.quote(source_root)} '
        f'|| fail {shlex.quote(f"cannot enter {source_root}")}',
        MODE_PROBE.format(build=shlex.quote(build),
                          half=shlex.quote(
                              f'{build}/CMakeCache.txt exists without {build}/build.ninja under '
                              f'{source_root}; refusing to reconfigure or delete that tree')),
    ]
    if not allow_full_build:
        lines += [
            'if [ "$mode" = full ]; then',
            f'    fail {shlex.quote(INCREMENTAL_ONLY_REFUSAL.format(build=build, source=source_root))}',
            'fi',
        ]
    lines.append(
        f'exec timeout --signal=TERM --kill-after={KILL_AFTER_SECONDS} {timeout} '
        f'ninja -C {shlex.quote(build)} -j {jobs} "$@"' if not allow_full_build else
        'if [ "$mode" = incremental ]; then')
    if allow_full_build:
        lines += [
            f'    exec timeout --signal=TERM --kill-after={KILL_AFTER_SECONDS} {timeout} '
            f'ninja -C {shlex.quote(build)} -j {jobs} "$@"',
            'fi',
            # The full build supersedes the named targets, so "$@" is not
            # forwarded: a flag-shaped target would otherwise reach build.sh as
            # an option. `full_build` is a shell function and `timeout` can only
            # execute a program, so it is exported and run through one more bash
            # whose only job is to call it. That keeps the container-side bound
            # real instead of relying on the outer supervisor to notice a wedged
            # docker exec.
            f'export ASCENDNPU_BUILD_SUBDIR={shlex.quote(build)} '
            f'ASCENDNPU_CCEC_BIN={shlex.quote(str(ccec_bin) if ccec_bin is not None else "")} '
            f'ASCENDNPU_NINJA_JOBS={shlex.quote(jobs)} '
            'ASCENDNPU_FULL_BUILD_LOG=-',
            'export -f full_build _full_build_error',
            f'timeout --signal=TERM --kill-after={KILL_AFTER_SECONDS} {full_timeout} '
            f'bash -c {shlex.quote("full_build " + shlex.quote(source_root))} '
            '|| fail "full build failed"',
            *[f'[ -f {shlex.quote(build)}/{shlex.quote(name)} ] '
              f'|| fail {shlex.quote(f"full build did not produce {build}/{name}")}'
              for name in FULL_BUILD_PRODUCTS],
        ]
    return '\n'.join(lines) + '\n'


def full_build_snippet() -> str:
    """The committed full-build script's text, for inlining."""
    try:
        return FULL_BUILD_SCRIPT.read_text(encoding='utf-8').rstrip('\n')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {FULL_BUILD_SCRIPT}: {exc}') from exc


def require_conda_pair(conda_sh: Path | None, conda_env: str | None) -> None:
    """CONDA_SH and CONDA_ENV are both-or-neither; a half configuration fails closed.

    Silently ignoring one half would run Ninja against a different interpreter
    than the one the build tree was configured with, which is exactly the failure
    the Adapter is supposed to prevent.
    """
    if (conda_sh is None) != (conda_env is None):
        raise BridgeError('ASCENDNPU_CONDA_SH and ASCENDNPU_CONDA_ENV must be provided together; '
                          'set both or neither')


def ccec_bin() -> Path | None:
    """The ccec bin directory a from-zero build needs, when the file names one.

    An incremental build never reconfigures, so it needs no compiler path; the
    from-zero branch does, and it refuses to start without one rather than
    letting `build.sh` configure a tree against a guessed toolchain.
    """
    path, values = machine_values()
    raw = machine_value('ASCENDNPU_CCEC_BIN', path=path, values=values) or ''
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise BridgeError(f'{path or "no machine file"}: ASCENDNPU_CCEC_BIN must be an '
                          f'absolute container path: {raw!r}')
    if '\n' in raw or '\x00' in raw:
        raise BridgeError('ASCENDNPU_CCEC_BIN contains an illegal character')
    return candidate


def declared_container_source() -> Path | None:
    """The container source root the machine file names, if it names one.

    This key exists for the case where the container sees the checkout at a
    different path than the bind-mount mapping derives. It confirms a path, it
    cannot invent one: the derived mapping is still computed from `docker
    inspect` and compared, so a value here that does not match the mounts is
    refused rather than trusted.

    It lives in `env.sh` rather than the launching shell for the same reason the
    container identity does: it decides which tree the Gate runs in, and a stale
    export would gate a directory nobody recorded.
    """
    path, values = machine_values()
    raw = machine_value('ASCENDNPU_CONTAINER_SOURCE', path=path, values=values) or ''
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise BridgeError(f'{path or "no machine file"}: ASCENDNPU_CONTAINER_SOURCE must be '
                          f'an absolute container path: {raw!r}')
    if chr(10) in raw or chr(0) in raw:
        raise BridgeError('ASCENDNPU_CONTAINER_SOURCE contains an illegal character')
    return candidate


def check_config() -> int:
    """Validate machine configuration on the host without Docker or writes."""
    container, user, home = container_identity()
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError('ASCENDNPU_CONDA_ENV is not a plain environment name')
    build_subdir()
    ninja_jobs()
    ccec_bin()
    declared_container_source()
    print(f'{PROG}: configuration valid (container={container} user={user})', file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    if argv == ['--check-config']:
        return check_config()
    host_root = Path.cwd()
    container, user, home = container_identity()
    timeout, full_timeout, targets, allow_full = timeout_seconds(list(argv))
    build = build_subdir()
    jobs = ninja_jobs()
    ccec = ccec_bin()
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError(f'ASCENDNPU_CONDA_ENV is not a plain environment name: {conda_env!r}')

    mapped = map_host_to_container(host_root, container_mounts(container), container)
    declared = declared_container_source()
    if declared is not None and os.path.realpath(str(declared)) != os.path.realpath(str(mapped)):
        raise BridgeError(f'ASCENDNPU_CONTAINER_SOURCE {declared} does not match the bind-mount '
                          f'mapping {mapped} of host source root {host_root}')
    container_root = str(declared if declared is not None else mapped)
    workdir = container_root
    prefix = docker_exec_prefix(container, user, home, workdir)
    verify_same_tree(host_root, prefix, container_root)

    print(f'{PROG}: container={container} user={user} source={container_root} '
          f'build={build} jobs={jobs} timeout={timeout}s fullBuildTimeout={full_timeout}s '
          f'allowFullBuild={allow_full} targets={" ".join(targets)}', file=sys.stderr)
    script = container_script(container_root, build, jobs, timeout, cann, conda_sh, conda_env,
                              full_timeout=full_timeout, ccec_bin=ccec, targets=tuple(targets),
                              allow_full_build=allow_full)
    completed = subprocess.run([*prefix, 'bash', '-c', script, 'bash', *targets])
    return completed.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BridgeError as error:
        print(f'{PROG}: {error.message}', file=sys.stderr)
        raise SystemExit(BridgeError.EXIT)
