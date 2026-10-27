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

Environment variables (all read at run time):

  ASCENDNPU_CONTAINER_NAME   required, name of a running build container
  ASCENDNPU_CONTAINER_USER   required, user to run as inside the container
  ASCENDNPU_CONTAINER_HOME   required, HOME inside the container
  ASCENDNPU_CONTAINER_SOURCE optional, explicit container path of the source
                             root; when set it must match the bind-mount
                             mapping, when unset the mapping is used directly
  ASCENDNPU_BUILD_SUBDIR     optional, build directory name (default: build)
  ASCENDNPU_NINJA_JOBS       optional, ninja -j value (default: container CPUs)
  ASCENDNPU_GATE_TIMEOUT     optional, container-side timeout in seconds; used
                             ONLY when no `--timeout` argument is passed

Toolchain configuration comes from a file, not from the launching shell:

  ASCENDNPU_ENV_FILE         explicit path of the env.sh that holds CANN/Conda.
                             Must exist when set.

Without it the file is found by walking up from this script to the checkout that
owns it, and reading `<checkout>/adapters/ascendnpu-ir/env.sh`, next to the
committed env.example.sh. That resolves the same file from a checkout and from an
installed Pack inside one, so a human or an agent edits the configuration in the
repository, beside the template it was copied from. There is no other location:
a machine that keeps the file elsewhere names it with ASCENDNPU_ENV_FILE, so no
Gate can read a file nobody reviewed.

Editing the file changes the next Gate without recreating the container.
Exporting those three variables in the launching shell has no effect: a stale
export must never mask a file edit.
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


# Keys the toolchain file may define. Anything else in the file is an error: a
# typo such as ASCENDNPU_CANN_SETUP= (double underscore) must not be ignored.
TOOLCHAIN_KEYS = ('ASCENDNPU_CANN_SETUP', 'ASCENDNPU_CONDA_SH', 'ASCENDNPU_CONDA_ENV')
# Keys the same file may also define but the Gate does not use. scripts/full-build.sh
# reads ASCENDNPU_CCEC_BIN to configure a fresh build tree; an incremental build
# never reconfigures, so the Gate neither requires nor consumes it.
OPTIONAL_FILE_KEYS = ('ASCENDNPU_CCEC_BIN',)
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
        if key not in TOOLCHAIN_KEYS and key not in OPTIONAL_FILE_KEYS:
            raise BridgeError(f'{path}: line {number} sets unknown key {key!r}; '
                              f'allowed keys are '
                              f'{", ".join(TOOLCHAIN_KEYS + OPTIONAL_FILE_KEYS)}')
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
    explicit = env_value('ASCENDNPU_ENV_FILE')
    path = Path(explicit).expanduser() if explicit else default_env_file()
    if path is None:
        return None, None, None
    if not path.is_absolute():
        raise BridgeError(f'ASCENDNPU_ENV_FILE must be an absolute path: {explicit!r}')
    if not path.exists():
        if explicit:
            raise BridgeError(f'ASCENDNPU_ENV_FILE does not exist: {path}')
        return None, None, None
    if not path.is_file():
        raise BridgeError(f'ASCENDNPU_ENV_FILE is not a regular file: {path}')
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc

    values = _parse_env_file(text, path)
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


def build_subdir() -> str:
    """A single relative directory name; traversal and absolute paths are refused."""
    if 'ASCENDNPU_BUILD_SUBDIR' in os.environ and not os.environ['ASCENDNPU_BUILD_SUBDIR'].strip():
        raise BridgeError('ASCENDNPU_BUILD_SUBDIR is set but empty')
    value = env_value('ASCENDNPU_BUILD_SUBDIR', default='build') or 'build'
    if not SAFE_NAME.fullmatch(value):
        raise BridgeError(f'ASCENDNPU_BUILD_SUBDIR is not a safe relative directory name: {value!r}')
    return value


def ninja_jobs() -> str:
    value = env_value('ASCENDNPU_NINJA_JOBS')
    if value is None:
        # Let the container decide from its own CPU count and cgroup limits.
        return '$(nproc)'
    if not value.isdigit() or int(value) < 1:
        raise BridgeError(f'ASCENDNPU_NINJA_JOBS must be a positive integer: {value!r}')
    return value


def timeout_seconds(argv: list[str]) -> tuple[int, list[str]]:
    """Optional leading `--timeout SECONDS`; the remainder is the Ninja argv."""
    value = None
    if argv[:1] == ['--timeout']:
        # An explicit `--timeout` is authoritative; it is what the manifest passes.
        if len(argv) < 3:
            raise BridgeError('--timeout requires a value and at least one Ninja target')
        value, argv = argv[1], argv[2:]
    else:
        # Only a bare invocation falls back to the environment.
        value = env_value('ASCENDNPU_GATE_TIMEOUT', default='1500')
    if value is None or not value.isdigit() or int(value) < 1:
        raise BridgeError(f'gate timeout must be a positive integer number of seconds: {value!r}')
    if not argv:
        raise BridgeError('no Ninja target was given')
    return int(value), argv


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


def container_script(source_root: str, build: str, jobs: str, timeout: int,
                     cann: Path | None, conda_sh: Path | None, conda_env: str | None) -> str:
    """Fixed container-side Bash: initialize the toolchain, then run Ninja.

    Only the Gate-owned part is written here. The toolchain sequence lives in
    `scripts/init-toolchain.sh`, which is inlined and then called with the three
    values already validated on the Host, so a human sourcing that file and a
    Gate running it execute the same reviewed initialization. It returns non-zero
    instead of exiting, which is why `fail` turns its result into the Gate's own
    exit 3.

    The rest of the order is fixed: initialization, then `cd`, then the build
    tree check, then the Ninja exec. The checks for cmake and ninja happen inside
    the initialization, so they resolve from the effective PATH, after CANN and
    `conda activate` have extended it.
    """
    lines = [
        'set -o pipefail',
        'fail() { echo "ascendnpu-ir-docker-gate: $*" >&2; exit 3; }',
        init_snippet(),
        f'init_toolchain {init_args(cann, conda_sh, conda_env)} '
        '|| fail "toolchain initialization failed"',
        f'cd {shlex.quote(source_root)} || fail "cannot enter {source_root}"',
        f'[ -f {shlex.quote(build)}/build.ninja ] '
        f'|| fail "no {build}/build.ninja under {source_root}; initialize the build directory first"',
        f'exec timeout --signal=TERM --kill-after={KILL_AFTER_SECONDS} {timeout} '
        f'ninja -C {shlex.quote(build)} -j {jobs} "$@"',
    ]
    return '\n'.join(lines) + '\n'


def require_conda_pair(conda_sh: Path | None, conda_env: str | None) -> None:
    """CONDA_SH and CONDA_ENV are both-or-neither; a half configuration fails closed.

    Silently ignoring one half would run Ninja against a different interpreter
    than the one the build tree was configured with, which is exactly the failure
    the Adapter is supposed to prevent.
    """
    if (conda_sh is None) != (conda_env is None):
        raise BridgeError('ASCENDNPU_CONDA_SH and ASCENDNPU_CONDA_ENV must be provided together; '
                          'set both or neither')


def check_config() -> int:
    """Validate machine configuration on the host without Docker or writes."""
    container = env_value('ASCENDNPU_CONTAINER_NAME', required=True) or ''
    user = env_value('ASCENDNPU_CONTAINER_USER', required=True) or ''
    home = env_value('ASCENDNPU_CONTAINER_HOME', required=True) or ''
    if not SAFE_NAME.fullmatch(container) or not SAFE_NAME.fullmatch(user):
        raise BridgeError('container name and user must be plain names, not option-like strings')
    if not home.startswith('/'):
        raise BridgeError('ASCENDNPU_CONTAINER_HOME must be an absolute container path')
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError('ASCENDNPU_CONDA_ENV is not a plain environment name')
    build_subdir()
    ninja_jobs()
    env_path('ASCENDNPU_CONTAINER_SOURCE')
    print(f'{PROG}: configuration valid', file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    if argv == ['--check-config']:
        return check_config()
    host_root = Path.cwd()
    container = env_value('ASCENDNPU_CONTAINER_NAME', required=True) or ''
    user = env_value('ASCENDNPU_CONTAINER_USER', required=True) or ''
    home = env_value('ASCENDNPU_CONTAINER_HOME', required=True) or ''
    if not SAFE_NAME.fullmatch(container) or not SAFE_NAME.fullmatch(user):
        raise BridgeError('container name and user must be plain names, not option-like strings')
    if not home.startswith('/'):
        raise BridgeError(f'ASCENDNPU_CONTAINER_HOME must be an absolute container path: {home!r}')

    timeout, targets = timeout_seconds(list(argv))
    build = build_subdir()
    jobs = ninja_jobs()
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError(f'ASCENDNPU_CONDA_ENV is not a plain environment name: {conda_env!r}')

    mapped = map_host_to_container(host_root, container_mounts(container), container)
    declared = env_path('ASCENDNPU_CONTAINER_SOURCE')
    if declared is not None and os.path.realpath(str(declared)) != os.path.realpath(str(mapped)):
        raise BridgeError(f'ASCENDNPU_CONTAINER_SOURCE {declared} does not match the bind-mount '
                          f'mapping {mapped} of host source root {host_root}')
    container_root = str(declared if declared is not None else mapped)
    workdir = container_root
    prefix = docker_exec_prefix(container, user, home, workdir)
    verify_same_tree(host_root, prefix, container_root)

    print(f'{PROG}: container={container} user={user} source={container_root} '
          f'build={build} jobs={jobs} timeout={timeout}s targets={" ".join(targets)}', file=sys.stderr)
    script = container_script(container_root, build, jobs, timeout, cann, conda_sh, conda_env)
    completed = subprocess.run([*prefix, 'bash', '-c', script, 'bash', *targets])
    return completed.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BridgeError as error:
        print(f'{PROG}: {error.message}', file=sys.stderr)
        raise SystemExit(BridgeError.EXIT)
