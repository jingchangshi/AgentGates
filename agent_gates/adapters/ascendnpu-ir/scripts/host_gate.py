#!/usr/bin/env python3
"""Run one AscendNPU-IR Ninja target on the Host, with no Docker in the path.

This Adapter used to have exactly one execution route: `docker exec` into a build
container, with the source root proved to be the same bytes on both sides. That
route is correct and stays unchanged, but it has a precondition nothing in this
repository can satisfy - a running container and permission to enter it. On a
machine whose operator cannot reach the Docker socket the Adapter is not
degraded, it is unusable, and the work it exists to gate (does the tree in front
of me still build, and still pass `check-bishengir`?) is exactly the work that
does not depend on a container at all.

So this file adds the second route: the same tree, built on the Host, as the
operator who launched the Gate. Host mode is not a different Gate. It is the same
Gate with the transport removed, and that is the whole reason it is a *wrapper*:

* the strict `NAME=value` machine file, the build-directory and job-count rules,
  the three-state build-tree probe, the `timeout --signal=TERM --kill-after=15`
  chain and the from-zero `full_build` command are NOT re-implemented here;
* they are bound from the pinned sibling `scripts/docker_gate.py`, which this
  module loads by its own resolved path. The manifest pins that file's digest, so
  the parser and the script generator a Host Gate runs are the reviewed ones. A
  copy would be a second implementation no digest covers, and the first
  divergence would be a Run that reports PASS for a command nobody reviewed;
* importing it dispatches nothing: the sibling guards its dispatch with
  `if __name__ == '__main__':`, so loading the module is not running it.

What this file adds is the part Docker mode does not need and cannot provide: a
preflight of the *Host* toolchain. Docker mode inherits whatever the image has,
so cmake, ninja, clang and `ld.lld` are the image's problem. On the Host they are
the operator's problem, and the failure arrives late and unrecognisably - the
source requires ninja >= 1.12 (CMakeLists.txt:6-13) and cmake >= 3.28
(CMakeLists.txt:2), while a stock Ubuntu 24.04 puts ninja 1.11 first on PATH and
dies inside a configure that has already spent minutes on LLVM. The verification
here resolves each tool from the EFFECTIVE PATH - after CANN and `conda activate`
have had their say - and prints the path it found, because "ninja 1.11.1" is not
actionable while "the first ninja on PATH is /usr/bin/ninja, and 1.13.2 is
installed elsewhere" is.

The container path is never touched: no container identity, no bind-mount
mapping, no `dev:ino` comparison, no `docker exec` prefix. The Host *is* the
tree, so the identity proof the container route exists to perform has nothing to
prove here, and calling any of it would make a Host Gate depend on the very
precondition it was written to avoid.

Two configuration checks, because the two recipes through this file need
different machines:

* `--check-config-build` - a from-zero tree needs everything listed above plus a
  ccec bin directory holding executable `ccec` and `llvm-link`
  (bishengir/lib/Template/CMakeLists.txt:573-578) and populated
  `third-party/llvm-project` and `third-party/shmem`. This reports that at t=0.
* `--check-config-regression` - must be useful BEFORE a build tree exists, which
  is precisely when an operator runs `doctor` on a new machine. It validates the
  same file and the same tools and only *reports* whether `<build>/build.ninja`
  exists, because the regression recipe itself refuses to build from scratch and
  an absent tree is not a broken configuration.

Both checks are read-only: they never configure, never run cmake, and never write
into the source tree.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# New messages say "host" so an operator reading a receipt can tell which route
# refused. The generated script keeps the sibling's own prefix in its `fail`
# messages: it is the same reviewed generator, and rewriting its text would be
# the fork this file exists to avoid.
PROG = 'ascendnpu-ir-host-gate'

# The pinned sibling, loaded through the *resolved* path of this file. Not
# PYTHONPATH: search-path order is something the launching shell controls, so a
# same-named module placed earlier on it could replace the reviewed parser. Not
# an `agent_gates` package import either: an installed Pack must run without the
# package that shipped it, exactly as the sibling does.
SIBLING = Path(__file__).resolve().with_name('docker_gate.py')
_SHARED_MODULE_NAME = 'ascendnpu_shared_gate'


def _load_shared(path: Path):
    """Import the sibling bridge as a module; it executes no Gate on import."""
    spec = importlib.util.spec_from_file_location(_SHARED_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise OSError(f'not a loadable python module: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    shared = _load_shared(SIBLING)
except (OSError, SyntaxError) as error:
    print(f'{PROG}: cannot load the pinned sibling {SIBLING}: {error}', file=sys.stderr)
    raise SystemExit(2)

# Reuse, deliberately by binding instead of wrapping or copying. Every name below
# resolves to the sibling's own object, so Host mode and Docker mode cannot
# disagree about the file grammar, the default build directory, the conda pairing
# rule, the timeout flags or a single byte of the script that runs. Wrapping would
# let the two drift one argument at a time; copying would put an unpinned copy in
# the Pack. `_parse_env_file` and `FILE_KEYS` are bound for the same reason - the
# parity tests drive both modules over one corpus of files, and they have to reach
# one parser, not two look-alikes.
#
# `ccec_bin` is the one exception, and only its wording differs. The sibling's
# message says the value "must be an absolute container path" (docker_gate.py:709),
# which is true where it was written and wrong here: on this route every path is a
# Host path (env.example.sh and docs/adapters.md both say so). A host-only operator
# holding a relative value would be told to fix the wrong thing. The rule stays the
# sibling's; `ccec_bin` below is the sibling's function with this route's noun.
BridgeError = shared.BridgeError
FILE_KEYS = shared.FILE_KEYS
_parse_env_file = shared._parse_env_file
machine_values = shared.machine_values
machine_value = shared.machine_value
build_subdir = shared.build_subdir
ninja_jobs = shared.ninja_jobs
toolchain_from_env_file = shared.toolchain_from_env_file
require_conda_pair = shared.require_conda_pair
_shared_ccec_bin = shared.ccec_bin
timeout_seconds = shared.timeout_seconds
init_snippet = shared.init_snippet
init_args = shared.init_args
container_script = shared.container_script
SAFE_NAME = shared.SAFE_NAME


def ccec_bin():
    """`ASCENDNPU_CCEC_BIN`, with a refusal that names the right kind of path.

    The validation itself is the sibling's: absoluteness and the illegal-character
    check are properties of the value, not of the transport, and re-deriving them
    here is how two rules for one key appear. Only the sentence is replaced, and
    only when it is the container-path refusal that was raised - any other
    `BridgeError` from that function is a different problem and is re-raised
    untouched so its original wording keeps pointing at its own cause.
    """
    try:
        return _shared_ccec_bin()
    except BridgeError as error:
        if 'absolute container path' not in error.message:
            raise
        raise BridgeError(error.message.replace(
            'absolute container path', 'absolute host path (this route has no container)')) from error

# Floors the source itself enforces. They are the reason a Host preflight can say
# "this machine cannot build this tree" before anything is configured: cmake asks
# for 3.28 at CMakeLists.txt:2 and the Ninja generator refuses below 1.12 at
# CMakeLists.txt:6-13.
MINIMUM_CMAKE = (3, 28, 0)
MINIMUM_NINJA = (1, 12, 0)

# Every tool that must resolve from the effective PATH, with the reason the tree
# needs it. `bash`, `timeout` and `nproc` are here because the generated script is
# Bash and its own bound is `timeout`: a Gate that cannot run them would report a
# confusing Ninja failure instead of a missing tool. clang, clang++, ld.lld and g++
# are the compilers and linker `scripts/full-build.sh` passes to build.sh; python3
# drives lit for `check-bishengir`; git is a hard prefix check inside build.sh:75-77.
REQUIRED_TOOLS = ('bash', 'timeout', 'nproc', 'cmake', 'ninja', 'clang', 'clang++',
                  'g++', 'python3', 'ld.lld', 'git')

# The prefix `scripts/init-toolchain.sh` puts on every reason it refuses. Matching
# it is how a failure here keeps the specific cause ("cmake not found in PATH
# after environment initialization") instead of the generic line this probe's own
# `fail` appends afterwards.
INIT_PREFIX = 'ascendnpu-ir-init:'

# The two submodules a from-zero build cannot proceed without. `torch-mlir` is
# deliberately absent: `scripts/full-build.sh` passes no `--build-torch-mlir`, so
# demanding it would refuse a tree that builds perfectly well.
FROM_ZERO_TREES = ('third-party/llvm-project', 'third-party/shmem')

# The lit target the regression recipe runs, and the only target whose coverage
# depends on a tool the generator never runs. Named here so the diagnostic below
# cannot drift from the manifest: the recipe passes exactly this target.
REGRESSION_TARGET = 'check-bishengir'

# The capability lit decides from a PATH lookup, and the directive that makes a
# test depend on it. A file carrying the directive is not failed when the feature
# is absent - lit reports it UNSUPPORTED - so a PASS can cover fewer files than
# the tree holds. Counting those files is how an operator learns how many.
HIVMC_TOOL = 'hivmc'
HIVMC_DIRECTIVE = re.compile(r'^[ \t]*(?://|#|;|!)[ \t]*REQUIRES:[^\n]*\bhivmc\b',
                             re.MULTILINE)
LIT_TEST_ROOT = 'bishengir/test'

# A file this large is not a lit test: it is a build or data artifact that happens
# to sit under the test root, and reading it would cost more than the count is
# worth on the path of every regression dispatch.
_MAX_SCANNED_BYTES = 1024 * 1024

# First dotted numeric run in a tool's version output. Deliberately not anchored:
# `cmake --version` says "cmake version 3.28.3", clang says "Ubuntu clang version
# 18.1.3 (1ubuntu1)", and g++ puts its Debian revision first. What matters below is
# that the comparison is numeric - 3.9 is older than 3.28 - which a string
# comparison gets backwards.
_VERSION = re.compile(r'([0-9]+(?:\.[0-9]+)*)')


def version_parts(text: str) -> tuple[int, ...] | None:
    """The first dotted version in `text`, as integers, or None."""
    match = _VERSION.search(text or '')
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split('.'))


def version_at_least(reported: str, minimum: tuple[int, ...]) -> bool:
    """Numeric comparison, with missing components read as zero.

    `ninja --version` prints three components and a floor may name two, so the
    shorter tuple is padded rather than compared on length.
    """
    parts = version_parts(reported)
    if parts is None:
        return False
    width = max(len(parts), len(minimum))
    return parts + (0,) * (width - len(parts)) >= minimum + (0,) * (width - len(minimum))


def tree_mode(build: Path) -> str:
    """Which of the shared generator's three tree states this directory is in.

    The generated script decides the same three states inside the shell before it
    runs anything. Deciding them here, without cmake and without writing, is what
    lets the configuration check refuse a half-built tree and demand the from-zero
    prerequisites up front instead of after a long configure.
    """
    if (build / 'build.ninja').is_file():
        return 'incremental'
    if (build / 'CMakeCache.txt').exists():
        # Same refusal, same reason, as the generated MODE_PROBE: that tree can
        # neither be built incrementally nor reconfigured without deleting state
        # the Gate did not create - and no Gate may delete a tree.
        raise BridgeError(
            f'{build}/CMakeCache.txt exists without {build}/build.ninja; refusing to '
            f'reconfigure or delete that tree - it is neither incrementally buildable '
            f'nor safe to reconfigure without deleting state this Gate did not create')
    return 'full'


def _populated(directory: Path) -> bool:
    """True when the directory holds at least one entry, without listing it all."""
    try:
        return next(directory.iterdir(), None) is not None
    except OSError:
        return False


def require_from_zero_prerequisites(source: Path, ccec: Path | None) -> None:
    """The compiler and submodule facts only a from-zero configure needs.

    An incremental run never reconfigures, so it needs none of this and a machine
    that only ever builds incrementally may leave `ASCENDNPU_CCEC_BIN` empty - the
    same rule `ccec_bin()` already documents. This function is reached only when
    the tree really has neither `build.ninja` nor `CMakeCache.txt`.

    `ccec` must be executable, not merely present. The CMake check is `EXISTS`
    (bishengir/lib/Template/CMakeLists.txt:575) and this machine's copy is mode
    0750, so a directory carrying the right names but the wrong mode would pass
    CMake and then fail inside the tool, naming none of it.
    """
    if ccec is None:
        raise BridgeError(
            f'{source} has no build.ninja and no CMakeCache.txt, so the build Gate '
            f'would configure it from scratch - which needs a compiler. Set '
            f'ASCENDNPU_CCEC_BIN in the machine file (see env.example.sh); an existing '
            f'build tree needs nothing here')
    for name in ('ccec', 'llvm-link'):
        tool = ccec / name
        if not tool.is_file():
            raise BridgeError(
                f'{ccec} does not contain {name}; a from-zero build sets '
                f'BISHENG_COMPILER_PATH to that directory and CMake refuses to '
                f'configure without both tools '
                f'(bishengir/lib/Template/CMakeLists.txt:573-578)')
        if not os.access(tool, os.X_OK):
            raise BridgeError(
                f'{tool} exists but is not executable by this user; CMake only tests '
                f'that it exists, so this would fail later inside the toolchain - fix '
                f'its mode')
    for relative in FROM_ZERO_TREES:
        tree = source / relative
        if not tree.is_dir():
            raise BridgeError(
                f'{tree} does not exist; a from-zero build compiles the sources '
                f'checked out there - run: git submodule update --init {relative}')
        if not _populated(tree):
            raise BridgeError(
                f'{tree} exists but is empty; a from-zero build compiles the sources '
                f'that submodule should hold - run: git submodule update --init '
                f'{relative}')


_PROBE_BODY = r'''
set -o pipefail
fail() { echo "ascendnpu-ir-host-gate: $*" >&2; exit 3; }
@INIT@
init_toolchain @ARGS@ || fail "toolchain initialization failed"
probe_tool() {
    key=$1
    cmd=$2
    path=$(command -v "$cmd" 2>/dev/null) || path=
    line=
    if [ -n "$path" ]; then
        line=$("$path" --version 2>/dev/null) || line=
    fi
    first=${line%%$'\n'*}
    printf 'tool\t%s\t%s\t%s\n' "$key" "$path" "$first"
}
@PROBES@
root=$(git rev-parse --show-toplevel 2>/dev/null) || root=
printf 'gitroot\t%s\n' "$root"
'''


def probe_script(cann: Path | None, conda_sh: Path | None, conda_env: str | None) -> str:
    """Bash that initializes the toolchain, then reports tools as TAB records.

    The toolchain sequence is the committed `scripts/init-toolchain.sh`, inlined by
    the sibling so the probes below see the EFFECTIVE PATH: a cmake that only
    `conda activate` provides, or a ninja that only CANN's setup prepends, is the
    one a build would use and therefore the one that must be judged. It also means
    a broken CANN setup fails here, with the init script's own message, instead of
    surfacing as a mysterious missing-tool complaint.

    Records are parsed in Python on purpose. A version comparison written in Bash
    is a string comparison, and `[ 3.9 -lt 3.28 ]` is not even a valid integer
    test; keeping the numbers on this side is what makes the check numeric.
    """
    probes = '\n'.join(f'probe_tool {name} {name}' for name in REQUIRED_TOOLS)
    return (_PROBE_BODY.replace('@INIT@', init_snippet())
            .replace('@ARGS@', init_args(cann, conda_sh, conda_env))
            .replace('@PROBES@', probes))


def run_probes(cann: Path | None, conda_sh: Path | None,
               conda_env: str | None) -> tuple[dict[str, tuple[str, str]], str]:
    """Resolve every required tool from the initialized PATH, plus the git root."""
    completed = subprocess.run(['bash', '-c', probe_script(cann, conda_sh, conda_env), 'bash'],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        # init_toolchain reports its own reason ("cmake not found in PATH after
        # environment initialization", "sourcing CANN setup failed: ..."), and this
        # probe's own `fail` then adds one generic line after it. Keep the specific
        # reason: the generic one says only *that* initialization failed, which is
        # the one thing the reader already knows.
        lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
        reason = next((line for line in reversed(lines) if line.startswith(INIT_PREFIX)), '')
        raise BridgeError(f'toolchain initialization failed: '
                          f'{reason or (lines[-1] if lines else "no output")}')
    tools: dict[str, tuple[str, str]] = {}
    git_root = ''
    for line in completed.stdout.splitlines():
        fields = line.split('\t')
        if fields[0] == 'tool' and len(fields) >= 4:
            tools[fields[1]] = (fields[2], fields[3])
        elif fields[0] == 'gitroot':
            git_root = fields[1] if len(fields) > 1 else ''
    return tools, git_root


def report_tools(tools: dict[str, tuple[str, str]]) -> None:
    """Print what resolved where, before judging any of it.

    A version complaint is only actionable beside the path it came from: this
    class of machine has ninja 1.13.2 in `/usr/local/bin` and 1.11.1 in
    `/usr/bin`, so the same message is a false alarm or a real blocker depending
    on which one PATH found.
    """
    for name in REQUIRED_TOOLS:
        path, version = tools.get(name, ('', ''))
        detail = f'{path} [{version.strip()}]' if path else 'NOT FOUND on the effective PATH'
        print(f'{PROG}: {name} = {detail}', file=sys.stderr)


def validate_tools(tools: dict[str, tuple[str, str]], git_root: str) -> None:
    """Judge the resolved tools against what the source actually enforces."""
    missing = [name for name in REQUIRED_TOOLS if not tools.get(name, ('', ''))[0]]
    if missing:
        raise BridgeError(
            f'the effective PATH cannot resolve {", ".join(missing)}; the build needs '
            f'them (compilers from scripts/full-build.sh:138-152, python3 for lit, git '
            f'from build-tools/build.sh:75-77). Install them, or let the CANN/Conda '
            f'initialization put them on PATH')
    for name, minimum, where in (('cmake', MINIMUM_CMAKE, 'CMakeLists.txt:2'),
                                 ('ninja', MINIMUM_NINJA, 'CMakeLists.txt:6-13')):
        path, version = tools[name]
        if not version_at_least(version, minimum):
            floor = '.'.join(str(part) for part in minimum)
            raise BridgeError(
                f'{name} at {path} reports {version.strip() or "no version"}, older than '
                f'{floor}, which this source requires ({where}). That path is only the '
                f'first {name} on the effective PATH, so a newer installation has to be '
                f'placed ahead of it')
    if not git_root:
        raise BridgeError(
            f'git rev-parse --show-toplevel failed in {Path.cwd()}; '
            f'build-tools/build.sh:75-77 exits 1 outside a git repository, so the '
            f'source root must be a work tree')


def count_hivmc_tests(source: Path) -> int | None:
    """How many lit files under the source carry `REQUIRES: hivmc`, or None.

    Best-effort by contract: this is a diagnostic, so an absent or unreadable
    directory is reported as "unknown", never raised. A Gate that refused because
    it could not count its own tests would be failing for the one reason that
    says nothing about the tree.

    The walk is a regex over the files themselves - no build, no lit, no cmake -
    because the alternative is parsing lit's configuration on the path of every
    regression dispatch. It reads the same directive lit reads: the marker comment
    of any lit dialect, then `REQUIRES:`, then the feature in the list. The
    directive is anchored at the start of a line so a RUN line that merely
    mentions the tool is not counted: on this tree that distinction is 54 files
    against 67, and over-reporting would make the number useless.
    """
    root = source / LIT_TEST_ROOT
    if not root.is_dir():
        return None
    count = 0
    for path in root.rglob('*'):
        try:
            if not path.is_file() or path.is_symlink():
                continue
            if path.stat().st_size > _MAX_SCANNED_BYTES:
                continue
            if HIVMC_DIRECTIVE.search(path.read_text(encoding='utf-8', errors='replace')):
                count += 1
        except OSError:
            # One unreadable file must not cost the count. It is skipped, and the
            # number below is then a lower bound for whoever reads it.
            continue
    return count


def report_lit_coverage(source: Path, build: str) -> None:
    """Say whether `hivmc` is reachable, and how much coverage depends on it.

    Why this exists at all: the regression Gate passes on `exitCodeZero` alone, so
    "the tests ran and passed" and "the tests were skipped" produce the same
    receipt. They do not mean the same thing. `bishengir/test/lit.cfg.py:333` adds
    the `hivmc` feature only when `shutil.which("hivmc")` succeeds, and it resolves
    that against the PYTHON PROCESS `os.environ` PATH. The lit configuration's own
    tool directories are appended to `config.environment` a few lines earlier
    (`lit.cfg.py:89-93`), and those two are NOT the same object: a `hivmc` that the
    build produced under `<build>/bin` is visible to a RUN line and still invisible
    to `shutil.which`. That is why the second half of the report matters even when
    the first half finds the tool.

    LOGGING ONLY, deliberately. A missing `hivmc` is a legitimate host - NVHPMC is
    part of the toolchain, not the Adapter - and Docker mode has exactly the same
    exposure, so making this fail would invent a refusal neither route had. No
    exit code changes, and no PATH is prepended: this change reports what lit will
    see, and quietly changing what lit sees is a different, larger change.

    Nothing here writes either. The count is a read of the source tree, so a Gate
    that is otherwise read-only stays read-only, and the block can be emitted
    before dispatch without changing what dispatch does.
    """
    print(f'{PROG}: regression coverage report for {REGRESSION_TARGET} '
          f'(lit feature "{HIVMC_TOOL}"):', file=sys.stderr)

    # The inherited PATH, and only that. The generated script sources CANN and
    # Conda before it execs Ninja, so the PATH lit finally inherits is the
    # POST-initialization one, which this Gate cannot re-derive without running
    # that initialization a second time. The consequence is stated in the message
    # below rather than hidden: the answer here is the pre-initialization truth,
    # so "not resolved" is a warning that initialization may still lift, never a
    # verdict about the run.
    resolved = shutil.which(HIVMC_TOOL)
    if resolved:
        print(f'{PROG}:   {HIVMC_TOOL} = {resolved} '
              f'(resolved on the inherited PATH; lit will add the "{HIVMC_TOOL}" '
              f'feature)', file=sys.stderr)
    else:
        print(f'{PROG}:   {HIVMC_TOOL} = NOT RESOLVED on the inherited PATH '
              f'(os.environ["PATH"]), checked before CANN/Conda initialization adds '
              f'anything; lit.cfg.py:333 runs shutil.which("{HIVMC_TOOL}") against that '
              f'object, so unless the initialization provides the tool the '
              f'"{HIVMC_TOOL}" feature is unavailable and every test carrying '
              f'"REQUIRES: {HIVMC_TOOL}" will be reported UNSUPPORTED rather than run',
              file=sys.stderr)

    # The build product, reported separately because it is the case that misleads:
    # a tree can hold the tool, pass every build, and still skip those tests. The
    # wording depends on what the lookup above already found - claiming lit will
    # not see a tool that is on PATH would be the kind of diagnostic an operator
    # stops believing.
    built = source / build / 'bin' / HIVMC_TOOL
    if built.exists() and not resolved:
        print(f'{PROG}:   {built} exists, but lit will STILL not see it while it is '
              f'absent from the inherited PATH: lit.cfg.py:89-93 append the build tool '
              f'directories to lit config.environment, while lit.cfg.py:333 asks '
              f'shutil.which, which reads os.environ. Those are different objects, so a '
              f'RUN line finds the tool and the feature probe does not',
              file=sys.stderr)
    elif built.exists() and Path(resolved) != built:
        print(f'{PROG}:   {built} exists; the inherited PATH resolves {HIVMC_TOOL} to '
              f'{resolved} instead, so the feature is on and the build product is merely '
              f'shadowed', file=sys.stderr)

    count = count_hivmc_tests(source)
    if count is None:
        print(f'{PROG}:   {source / LIT_TEST_ROOT} does not exist; the number of tests '
              f'depending on "{HIVMC_TOOL}" is unknown (this is not a failure - the '
              f'directory is absent, not unreadable)', file=sys.stderr)
    else:
        consequence = ('they will run' if resolved else
                       f'they will be skipped as UNSUPPORTED while this Gate can still '
                       f'report PASS')
        print(f'{PROG}:   {count} file(s) under {LIT_TEST_ROOT} carry '
              f'"REQUIRES: {HIVMC_TOOL}"; {consequence}', file=sys.stderr)


def check_config(mode: str) -> int:
    """Validate machine configuration on the Host: no Docker, no cmake, no writes.

    `mode` is `build` or `regression`. Both validate the file grammar, the build
    directory, the job count, the conda pairing and the whole toolchain; only
    `build` may demand the from-zero prerequisites, and only when the tree really
    needs a from-zero build.
    """
    source = Path.cwd()
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError(f'ASCENDNPU_CONDA_ENV is not a plain environment name: {conda_env!r}')
    build = source / build_subdir()
    jobs = ninja_jobs()
    ccec = ccec_bin()
    state = tree_mode(build)

    tools, git_root = run_probes(cann, conda_sh, conda_env)
    report_tools(tools)
    validate_tools(tools, git_root)

    if mode == 'build' and state == 'full':
        require_from_zero_prerequisites(source, ccec)

    conda = '-'.join(part for part in (str(conda_sh or ''), conda_env or '') if part)
    print(f'{PROG}: configuration valid ({mode}) source={source} build={build} '
          f'buildState={state} jobs={jobs} cann={cann or "-"} conda={conda or "-"} '
          f'ccec={ccec or "-"} gitRoot={git_root}', file=sys.stderr)
    if mode == 'regression' and state == 'full':
        # Not a failure: a new machine is exactly where `doctor` runs first. The
        # regression recipe is the thing that refuses, and it says so itself.
        print(f'{PROG}: note: {build}/build.ninja does not exist yet; the regression '
              f'recipe runs an existing tree only and will refuse until the build '
              f'recipe has produced one', file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] and argv[0].startswith('--check-config'):
        flag = argv[0]
        if flag not in ('--check-config-build', '--check-config-regression'):
            raise BridgeError(f'unknown configuration check {flag!r}; expected '
                              f'--check-config-build or --check-config-regression')
        if len(argv) > 1:
            raise BridgeError(f'{flag} takes no further arguments: {argv[1:]!r}')
        return check_config(flag.removeprefix('--check-config-'))

    source = Path.cwd()
    timeout, full_timeout, targets, allow_full = timeout_seconds(list(argv))
    build = build_subdir()
    jobs = ninja_jobs()
    ccec = ccec_bin()
    cann, conda_sh, conda_env = toolchain_from_env_file()
    require_conda_pair(conda_sh, conda_env)
    if conda_env is not None and not SAFE_NAME.fullmatch(conda_env):
        raise BridgeError(f'ASCENDNPU_CONDA_ENV is not a plain environment name: {conda_env!r}')

    print(f'{PROG}: host source={source} build={build} jobs={jobs} '
          f'timeout={timeout}s+{shared.KILL_AFTER_SECONDS}s '
          f'fullBuildTimeout={full_timeout}s allowFullBuild={allow_full} '
          f'targets={" ".join(targets)}', file=sys.stderr)
    # The regression target is the only one whose PASS depends on a capability lit
    # decides for itself, so it is the only one that gets the full report. It is
    # emitted here, before the generated script runs, so the block is in the
    # receipt's stderr even when the run later times out or is killed - a
    # diagnostic that only appears after a completed run would be missing exactly
    # when the operator is trying to explain the run they did not get.
    if REGRESSION_TARGET in targets:
        report_lit_coverage(source, build)
    elif not shutil.which(HIVMC_TOOL):
        # Build mode's job is the build, so one line: enough to explain later why
        # the regression that follows reports fewer tests than the tree holds.
        print(f'{PROG}: note: {HIVMC_TOOL} is not on the inherited PATH; a following '
              f'{REGRESSION_TARGET} will report its "{HIVMC_TOOL}" tests as UNSUPPORTED, '
              f'and that run prints the full coverage report', file=sys.stderr)
    # The generated script is the sibling's own, unchanged: same init sequence,
    # same three-state probe, same `exec timeout --signal=TERM --kill-after=15`
    # chain, same from-zero command. Only the transport is gone, so the Host source
    # root is passed where Docker mode passes the container one.
    script = container_script(str(source), build, jobs, timeout, cann, conda_sh, conda_env,
                              full_timeout=full_timeout, ccec_bin=ccec,
                              targets=tuple(targets), allow_full_build=allow_full)
    completed = subprocess.run(['bash', '-c', script, 'bash', *targets])
    # Unchanged, and for the same reasons Docker mode returns it unchanged: 124 is
    # `timeout` saying the tree was too slow, 3 is this Gate saying the build it
    # started failed, and neither may be laundered into a PASS.
    return completed.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BridgeError as error:
        print(f'{PROG}: {error.message}', file=sys.stderr)
        raise SystemExit(BridgeError.EXIT)
