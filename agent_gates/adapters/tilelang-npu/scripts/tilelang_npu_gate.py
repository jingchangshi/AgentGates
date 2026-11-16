#!/usr/bin/env python3
"""Host-local TileLang Ascend build and example Gate (tilelang-npu Adapter).

Scope
-----

This Adapter answers one question: does *this* TileLang checkout build from
source on *this* host against *this* conda environment, and do the example
kernels then run correctly on the NPU?

It is deliberately unlike the two tilelang-hivm Adapters. Those drive a
container, reach a separate HIVM compiler and require an existing
build/build.ninja. This one runs on the host, has no container and no HIVM
backend in its dependency set, and it *configures* its own build tree.
Requiring a pre-existing build.ninja here would refuse exactly the work the
Adapter exists to do.

Why configure the tree instead of reusing one
---------------------------------------------

A build tree records the compiler, the CMake that generated it and the Python
header paths it was configured against. Reusing a tree produced by another
Python or by a different CMake major version silently rebuilds against the
wrong headers, or - worse - links objects compiled from sources that do not
exist on the current branch. Both were observed on this host: a tree from
another branch carried objects for files absent from the checkout under test.
So the build Gate runs CMake first, always, from the sources actually present.

It never deletes the tree. An incremental re-run reuses it, which is what makes
the second run cheap after the first has paid the full cost.

Two steps, one direction
------------------------

--build compiles; --examples runs the pinned example tests on a device and
depends on the build. Neither step claims the other result: a green build says
nothing about numerical correctness, and a passing example says nothing about
whether the tree was built from the sources under test.

Output contract
---------------

Standard output carries exactly one JSON object and nothing else. Every
diagnostic goes to stderr, because the evaluator parses stdout as observations
and anything else on that stream becomes an unparseable report. Exit code 75
means "this environment cannot run the Gate" - an unusable interpreter, an
unactivatable environment - which the runner records as BLOCKED_ENV rather than
as a failed test.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

PROG = 'tilelang-npu-gate'

ADAPTER_DIR = Path(__file__).resolve().parent.parent

# Keys the machine file may define. Kept in step with machine_env.TILELANG_NPU_KEYS;
# the host regression suite compares the two, so a key added on one side cannot
# become configuration the other silently ignores.
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

# Alias read by the repository machine-file parity test.
TOOLCHAIN_KEYS = TILELANG_NPU_KEYS

# The keys this Gate cannot run without. The CMake binary and the example
# selection are optional: an unset CMake binary means "the cmake the activated
# environment provides", and an unset selection means the reviewed default list.
REQUIRED = (
    'TILELANG_NPU_CANN_SETUP',
    'TILELANG_NPU_CONDA_SH',
    'TILELANG_NPU_CONDA_ENV',
    'TILELANG_NPU_CONDA_ENVS_PATH',
    'TILELANG_NPU_PYTHON',
)

# Values that select a file or authorize one run. They are launch parameters,
# never machine facts: a device id authorizes one Gate run, and putting it in a
# file would make a machine look permanently entitled to a card.
NOT_MACHINE_KEYS = frozenset({'TILELANG_NPU_ENV_FILE', 'TILELANG_NPU_DEVICE_ID'})

# The example tests the regression runs when the machine file does not narrow
# them. These are the application examples that exercise real numerics; the
# language-feature suite under testing/ is a different contract.
DEFAULT_EXAMPLE_TESTS = (
    'examples/ascend/test_vecadd.py',
    'examples/ascend/test_rmsnorm.py',
    'examples/ascend/test_per_token_cast_to_fp8.py',
)

# CMake options this Adapter always passes. An Ascend build on a host with no
# CUDA toolkit needs CUDA off and the Ascend stubs on, or configure stops at
# target detection before compiling anything.
CMAKE_OPTIONS = (
    '-DCMAKE_BUILD_TYPE=Release',
    '-DUSE_ASCEND=ON',
    '-DUSE_CUDA=OFF',
    '-DUSE_ROCM=ON',
    '-DUSE_LLVM=OFF',
    '-DTILELANG_USE_ASCEND_STUBS=ON',
    '-DTILELANG_USE_CUDA_STUBS=ON',
    '-DTILELANG_USE_HIP_STUBS=ON',
    '-DUSE_Z3=ON',
    '-DUSE_PYPI_Z3=ON',
    '-DUSE_CUSTOM_LOGGING=ON',
    '-DSKBUILD_SABI_COMPONENT=Development.SABIModule',
    '-DSKBUILD_SABI_VERSION=3.9',
)

# Cython extension modules compiled by CMake under build/lib as Python Stable
# ABI (.abi3.so) shared objects so both Python 3.11 (tilelang-npu) and
# downstream Python >= 3.10 consumers (tilefoundry-npu) can load the same build.
SABI_EXTENSIONS = (
    'tilelang_cython_wrapper',
    'tilelang_pto_wrapper',
    'tilelang_ascend_npu_exchange',
    'tilelang_deepgemm_heuristics',
)
BUILD_STAMP_FILE = '.agent_gates_build_stamp.json'

_ASSIGN = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')

# Only machine facts may appear in the file. The launch values are named above so
# this parser can REFUSE them by name with a useful message: a file that pinned a
# device id would turn a per-run authorization into a standing entitlement, and
# the refusal is only helpful if it says which key is wrong and why.
ALLOWED = set(TILELANG_NPU_KEYS)

# A build is bounded well below the manifest timeout: a ninja that has not
# finished in that window is a diagnosis, not slow progress.
DEFAULT_TIMEOUT = 3300.0


class BridgeError(ValueError):
    """A machine file that cannot be honoured; mapped to BLOCKED_ENV."""


def _die(code, message):
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _parse_env_file(text, path):
    """Parse the strict NAME=value file. Never executes or expands it."""
    values = {}
    for number, raw in enumerate(str(text).splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = _ASSIGN.match(line)
        if match is None:
            raise BridgeError(f'{path}: line {number} is not a NAME=value assignment: {raw!r}')
        key, value = match.group(1), match.group(2).strip()
        if key in NOT_MACHINE_KEYS:
            raise BridgeError(f'{path}: line {number} sets {key}, which is a launch '
                              f'value, not machine configuration; pass it on the '
                              f'command line or the environment instead')
        if key not in ALLOWED:
            raise BridgeError(f'{path}: line {number} sets unknown key {key!r}; '
                              f'allowed keys are {", ".join(sorted(ALLOWED))}')
        if key in values:
            raise BridgeError(f'{path}: line {number} sets {key} a second time')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        elif value[:1] in ('"', "'") or value[-1:] in ('"', "'"):
            raise BridgeError(f'{path}: line {number} has an unbalanced quote in {raw!r}')
        values[key] = value
    return values


def default_env_file():
    """The machine file beside this Adapter, or None when the machine keeps none."""
    candidate = ADAPTER_DIR / 'env.sh'
    return candidate if candidate.is_file() else None


def resolved_env_file():
    explicit = (os.environ.get('TILELANG_NPU_ENV_FILE') or '').strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise BridgeError(f'TILELANG_NPU_ENV_FILE must be an absolute path: {explicit!r}')
        if not path.is_file():
            raise BridgeError(f'TILELANG_NPU_ENV_FILE is not a regular file: {explicit!r}')
        return path
    return default_env_file()


def machine_values():
    """The machine parameters, from the file only.

    The launching environment is deliberately not a source: an inherited export
    would let a stale shell point this Gate at another interpreter while the
    engine leased a different build directory.
    """
    path = resolved_env_file()
    if path is None:
        raise BridgeError(f'no machine file: set TILELANG_NPU_ENV_FILE or create '
                          f'{ADAPTER_DIR / "env.sh"}')
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f'cannot read {path}: {exc}') from exc
    return _parse_env_file(text, path)


def _required(values, name):
    value = (values.get(name) or '').strip()
    if not value:
        raise BridgeError(f'{name} is required in the machine file')
    return value


def build_subdir(values):
    """The build directory name this Adapter writes, relative to the source root.

    An empty value is a typo, not "use the default": the file spells "unset" by
    leaving the key out, and the engine resolves the same default from the same
    file, so both sides must agree on which tree is leased.
    """
    if 'TILELANG_NPU_BUILD_SUBDIR' in values:
        raw = values['TILELANG_NPU_BUILD_SUBDIR']
        if not raw.strip():
            raise BridgeError('TILELANG_NPU_BUILD_SUBDIR is set but empty')
        raw = raw.strip()
        # Refused here as well as in the engine: the engine decides which tree it
        # leases, but this Gate decides which tree it writes, and a value that
        # climbs out of the checkout must be stopped by both. Relying on the
        # caller leaves a path-traversal build reachable for anyone who runs the
        # script directly.
        if Path(raw).is_absolute() or '..' in Path(raw).parts or Path(raw).name != raw:
            raise BridgeError(f'TILELANG_NPU_BUILD_SUBDIR must be a single directory '
                              f'name inside the checkout: {raw!r}')
        return raw
    return 'build'


def ninja_jobs(values):
    """The job count, as text so the default can stay a Ninja-side expression."""
    if 'TILELANG_NPU_NINJA_JOBS' in values:
        raw = values['TILELANG_NPU_NINJA_JOBS']
        if not raw.strip():
            raise BridgeError('TILELANG_NPU_NINJA_JOBS is set but empty')
        raw = raw.strip()
        if not raw.isdigit() or int(raw) < 1:
            raise BridgeError(f'TILELANG_NPU_NINJA_JOBS must be a positive integer: {raw!r}')
        return raw
    return '$(nproc)'


def example_tests(values):
    """The fixed example test files, relative to the source root."""
    raw = (values.get('TILELANG_NPU_EXAMPLE_TESTS') or '').strip()
    if not raw:
        return list(DEFAULT_EXAMPLE_TESTS)
    names = [item.strip() for item in raw.split(',') if item.strip()]
    if not names:
        raise BridgeError('TILELANG_NPU_EXAMPLE_TESTS names no test file')
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise BridgeError('TILELANG_NPU_EXAMPLE_TESTS must name files inside the '
                              f'source root: {name!r}')
    return names


def _activation(values):
    """The shell prelude that makes CANN and the conda environment current.

    CANN is sourced before Conda because the conda environment activation is what
    provides the interpreter, and the toolkit variables must already be set for
    the libraries that interpreter will load. Every value is emitted as a single
    quoted literal: these strings reach a shell, so an unquoted command
    substitution inside a path would execute.
    """
    lines = [
        'set +u',
        'source ' + shlex.quote(_required(values, 'TILELANG_NPU_CANN_SETUP')),
        'export CONDA_ENVS_PATH=' + shlex.quote(_required(values, 'TILELANG_NPU_CONDA_ENVS_PATH')),
        'source ' + shlex.quote(_required(values, 'TILELANG_NPU_CONDA_SH')),
        'conda activate ' + shlex.quote(_required(values, 'TILELANG_NPU_CONDA_ENV')),
    ]
    return ' && '.join(lines)


def _run_shell(values, body, *, cwd, timeout, log):
    """Run a shell body inside the activated environment; return (rc, seconds)."""
    started = time.monotonic()
    script = 'set -o pipefail' + os.linesep + _activation(values) + os.linesep + body
    try:
        result = subprocess.run(['/bin/bash', '-c', script], cwd=str(cwd),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        log.write((exc.output or b'').decode('utf-8', 'replace'))
        log.flush()
        raise BridgeError('command timed out after ' + str(timeout) + ' s') from exc
    log.write(result.stdout.decode('utf-8', 'replace'))
    log.flush()
    return result.returncode, time.monotonic() - started


def _python(values):
    return _required(values, 'TILELANG_NPU_PYTHON')


def _source_root():
    """The checkout under test: the recipe working directory."""
    return Path.cwd()


def _sha256_file(path):
    """Streaming SHA-256 digest of a regular file."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git_head_pure(root):
    """Resolve Git HEAD commit without spawning a subprocess; 'NO_GIT' if unavailable."""
    dot_git = Path(root) / '.git'
    if dot_git.is_file():
        try:
            text = dot_git.read_text(encoding='utf-8').strip()
        except (OSError, UnicodeDecodeError):
            return 'NO_GIT'
        if not text.startswith('gitdir:'):
            return 'NO_GIT'
        raw_dir = text[len('gitdir:'):].strip()
        git_dir = Path(raw_dir) if Path(raw_dir).is_absolute() else (Path(root) / raw_dir)
    elif dot_git.is_dir():
        git_dir = dot_git
    else:
        return 'NO_GIT'

    head_file = git_dir / 'HEAD'
    if not head_file.is_file():
        return 'NO_GIT'
    try:
        head_text = head_file.read_text(encoding='utf-8').strip()
    except (OSError, UnicodeDecodeError):
        return 'NO_GIT'
    if not head_text.startswith('ref:'):
        return head_text if re.fullmatch(r'[0-9a-fA-F]{40,64}', head_text) else 'NO_GIT'

    ref_name = head_text[len('ref:'):].strip()
    search_dirs = [git_dir]
    commondir_file = git_dir / 'commondir'
    if commondir_file.is_file():
        try:
            raw_common = commondir_file.read_text(encoding='utf-8').strip()
            common_dir = Path(raw_common) if Path(raw_common).is_absolute() else (git_dir / raw_common)
            if common_dir.is_dir():
                search_dirs.append(common_dir)
        except (OSError, UnicodeDecodeError):
            pass

    for candidate_dir in search_dirs:
        ref_file = candidate_dir / ref_name
        if ref_file.is_file():
            try:
                commit = ref_file.read_text(encoding='utf-8').strip()
                if commit:
                    return commit
            except (OSError, UnicodeDecodeError):
                pass
        packed = candidate_dir / 'packed-refs'
        if packed.is_file():
            try:
                for line in packed.read_text(encoding='utf-8').splitlines():
                    if not line or line.startswith(('#', '^')):
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) == 2 and parts[1].strip() == ref_name:
                        return parts[0].strip()
            except (OSError, UnicodeDecodeError):
                pass
    return 'NO_GIT'


def _tilelang_native_inputs_sha256(source):
    """Deterministic digest over TileLang native/CMake/Cython build inputs."""
    src_root = Path(source).resolve()
    candidates = []
    for rel_file in ('CMakeLists.txt', 'examples/ascend/deepgemm/_heuristics.pyx'):
        p = src_root / rel_file
        if p.is_file():
            candidates.append(p)
    for rel_dir in ('cmake', 'src'):
        d = src_root / rel_dir
        if d.is_dir():
            candidates.extend(p for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    tl_pkg = src_root / 'tilelang'
    if tl_pkg.is_dir():
        for ext in ('*.pyx', '*.pxd', '*.pxi'):
            candidates.extend(p for p in tl_pkg.rglob(ext) if p.is_file() and '__pycache__' not in p.parts)

    by_rel = {}
    for p in candidates:
        rel = p.relative_to(src_root).as_posix()
        by_rel[rel] = p

    h = hashlib.sha256()
    for rel in sorted(by_rel):
        p = by_rel[rel]
        h.update(rel.encode('utf-8') + b'\0')
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        h.update(b'\0')
    tvm_head = _git_head_pure(src_root / '3rdparty' / 'tvm')
    h.update(b'submodule:3rdparty/tvm\0' + tvm_head.encode('utf-8') + b'\0')
    return h.hexdigest()


PROBE = (
    'import json, sys' + os.linesep
    + 'import tilelang' + os.linesep
    + 'import tilelang.testing as tt' + os.linesep
    + 'import tilelang_cython_wrapper, tilelang_ascend_npu_exchange' + os.linesep
    + 'print(json.dumps({"file": tilelang.__file__,' + os.linesep
    + '                  "wrapper": getattr(tilelang_cython_wrapper, "__file__", ""),' + os.linesep
    + '                  "ascend": bool(tt.ascend_backend_compiled())}))' + os.linesep
)


def _probe_observations(values, lib, source):
    """Measure what the built tree actually provides.

    Both facts are measured, never assumed. nativeLibraryPresent requires a real
    shared object of plausible size - a zero-byte file named libtilelang.so is
    not a build product - together with all four cp39-abi3 Cython extensions
    (.abi3.so). ascendBackendCompiled asks the interpreter that will run the
    examples whether its tilelang reports an Ascend-enabled build, and requires
    that answer to come from the tree under test (and load its extension from
    build/lib): a green flag from a different installed copy would describe
    code this Gate never built.
    """
    lib_dir = lib.parent
    sabi_present = all((lib_dir / f'{ext}.abi3.so').is_file() for ext in SABI_EXTENSIONS)
    present = lib.is_file() and lib.stat().st_size > (1 << 20) and sabi_present
    compiled = False
    if present:
        # The probe runs inside the SAME activated environment as the build.
        # Importing tilelang preloads torch, which loads torch_npu, which needs
        # the CANN libraries on the loader path. A probe started with the plain
        # host environment fails on a missing libhccl.so and would report "no
        # Ascend backend" for a perfectly good build. Prepending source on
        # PYTHONPATH ensures the live checkout is tested even if site-packages
        # carries another install.
        try:
            with tempfile.TemporaryDirectory(prefix='tilelang-npu-probe-') as tmp:
                log_path = Path(tmp) / 'probe.log'
                script = (
                    'export PYTHONPATH=' + shlex.quote(str(source)) + ':"${PYTHONPATH:-}"' + os.linesep
                    + shlex.quote(_python(values)) + ' -c ' + shlex.quote(PROBE)
                )
                with log_path.open('w') as log:
                    rc, _ = _run_shell(values, script, cwd=source, timeout=300, log=log)
                if rc == 0:
                    text = log_path.read_text(errors='replace').strip().splitlines()
                    payload = json.loads(text[-1])
                    resolved = Path(str(payload.get('file', ''))).resolve()
                    wrapper_raw = str(payload.get('wrapper', '')).strip()
                    wrapper_resolved = Path(wrapper_raw).resolve() if wrapper_raw else None
                    if (
                        resolved.is_relative_to(source.resolve())
                        and wrapper_resolved is not None
                        and wrapper_resolved.is_relative_to(lib_dir.resolve())
                    ):
                        compiled = payload.get('ascend') is True
        except (BridgeError, OSError, ValueError, IndexError):
            compiled = False
    return present, compiled


def check_config():
    """Preflight: is the machine file complete and is the source a checkout?

    Deliberately does not compile, does not touch a device and does not spawn
    CMake or Ninja. A configuration check that costs minutes is one operators
    skip, and a skipped check is not a check.
    """
    try:
        values = machine_values()
    except BridgeError as exc:
        _die(75, PROG + ': ' + str(exc))
    try:
        for name in REQUIRED:
            _required(values, name)
        build_subdir(values)
        ninja_jobs(values)
        example_tests(values)
        python = Path(_required(values, 'TILELANG_NPU_PYTHON'))
        if not python.is_file():
            raise BridgeError(f'TILELANG_NPU_PYTHON does not exist: {python}')
        for name in ('TILELANG_NPU_CANN_SETUP', 'TILELANG_NPU_CONDA_SH'):
            path = Path(_required(values, name))
            if not path.is_file():
                raise BridgeError(f'{name} does not exist: {path}')
        source = _source_root()
        if not (source / 'CMakeLists.txt').is_file():
            raise BridgeError(f'{source} is not a TileLang checkout (no CMakeLists.txt)')
    except BridgeError as exc:
        _die(75, PROG + ': ' + str(exc))
    print(PROG + ': configuration ok', file=sys.stderr)
    return 0


def check_config_examples():
    rc = check_config()
    if rc != 0:
        return rc
    try:
        dev = device_id()
        if dev is not None:
            require_device_free(dev)
    except BridgeError as exc:
        _die(75, PROG + ': ' + str(exc))
    return 0


def build(values, timeout):
    """Configure and compile the checkout, then measure the products."""
    source = _source_root()
    subdir = build_subdir(values)
    cmake = (values.get('TILELANG_NPU_CMAKE_BIN') or '').strip() or 'cmake'
    build_dir = source / subdir
    jobs = ninja_jobs(values)

    with tempfile.TemporaryDirectory(prefix='tilelang-npu-build-') as tmp:
        log_path = Path(tmp) / 'build.log'
        with log_path.open('w') as log:
            configure_body = ' '.join(
                [shlex.quote(cmake), '-S', shlex.quote(str(source)),
                 '-B', shlex.quote(str(build_dir)), '-G', 'Ninja']
                + list(CMAKE_OPTIONS)
                + ['-DPython_EXECUTABLE=' + shlex.quote(_python(values)),
                   '-DPython3_EXECUTABLE=' + shlex.quote(_python(values))])
            configure_rc, configure_seconds = _run_shell(
                values, configure_body, cwd=source, timeout=min(600.0, timeout), log=log)
            build_rc, build_seconds = configure_rc, 0.0
            if configure_rc != 0:
                print(PROG + f': configure failed rc={configure_rc}', file=sys.stderr)
                sys.stderr.write(log_path.read_text(errors='replace')[-4000:])
            else:
                # Drop any stale Python-version-specific extension modules so
                # only the freshly built cp39-abi3 (.abi3.so) modules remain in
                # build/lib and downstream Python 3.11 / 3.12+ runtimes agree.
                lib_dir = build_dir / 'lib'
                if lib_dir.is_dir():
                    for stale in lib_dir.glob('tilelang_*.cpython-*.so'):
                        stale.unlink(missing_ok=True)
                # The default target, not the narrower tilelang phony target: the
                # phony target builds only lib/libtilelang.so, while the Python
                # extension modules that make "import tilelang" work are produced
                # by the default target. Importing the result is the point of this
                # Gate, so the narrower target would report a green build whose
                # library cannot be imported.
                build_body = 'cmake --build ' + shlex.quote(str(build_dir)) + ' -- -j ' + jobs
                build_rc, build_seconds = _run_shell(
                    values, build_body, cwd=source, timeout=timeout, log=log)
                if build_rc != 0:
                    print(PROG + f': build failed rc={build_rc}', file=sys.stderr)
                    sys.stderr.write(log_path.read_text(errors='replace')[-4000:])

    # The claim is derived, not asserted. The evaluator also binds the host
    # process status, but a payload that says True while the compile failed
    # would be a receipt a reader could not trust on its own.
    succeeded = configure_rc == 0 and build_rc == 0

    lib_path = build_dir / 'lib' / 'libtilelang.so'
    present, compiled = _probe_observations(values, lib_path, source)
    if succeeded and present and compiled and lib_path.is_file():
        stamp = {
            'adapter': 'tilelang-npu',
            'sabi': 'cp39',
            'gitHead': _git_head_pure(source),
            'nativeInputsSha256': _tilelang_native_inputs_sha256(source),
            'libtilelangSha256': _sha256_file(lib_path),
            'builtAt': datetime.now(timezone.utc).isoformat(),
        }
        (build_dir / BUILD_STAMP_FILE).write_text(
            json.dumps(stamp, indent=2, sort_keys=True) + os.linesep,
            encoding='utf-8',
        )
    observations = {
        'exitCodeZero': bool(succeeded),
        'nativeLibraryPresent': bool(present),
        'ascendBackendCompiled': bool(compiled),
        'buildDir': str(build_dir),
        'configureSeconds': round(configure_seconds, 3),
        'buildSeconds': round(build_seconds, 3),
    }
    return build_rc, observations


def _junit_counts(path):
    """Collected/passed/failed/skipped from a JUnit report, or None."""
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


def device_id():
    """The card this run may use, or None when the engine leased none.

    Read from the environment rather than the machine file: this authorizes one
    run, and a machine file would turn that into a standing entitlement. The
    engine writes it only for a gate that declares it uses a device, so its
    absence is meaningful - it means no lease was taken and this Gate must not
    reach for a card anyway.
    """
    raw = (os.environ.get('TILELANG_NPU_DEVICE_ID') or '').strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise BridgeError(f'TILELANG_NPU_DEVICE_ID must be a nonnegative decimal ID: {raw!r}')
    return raw


def exclusive_requested():
    """Whether this run demands a card no other process is using."""
    raw = (os.environ.get('TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE') or '').strip().lower()
    return raw in {'1', 'true', 'yes', 'on'}


def device_occupancy(device):
    """The processes holding the card, as npu-smi reports them.

    Returns an empty list when npu-smi is absent or its output cannot be read:
    this Gate runs a numerical test, not an occupancy monitor, and a missing tool
    must not be able to fail a run that would otherwise pass.
    """
    smi = os.environ.get('AGENT_GATES_SMI_COMMAND') or 'npu-smi'
    try:
        result = subprocess.run([smi, 'info'], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    holders, in_process_table = [], False
    for line in result.stdout.decode('utf-8', 'replace').splitlines():
        if 'Process id' in line or 'Process name' in line:
            in_process_table = True
            continue
        if not in_process_table or '|' not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        # The process table names the device id in the first cell; a row for
        # another card says nothing about this run.
        if len(cells) >= 2 and cells[0] == str(device) and cells[1].isdigit():
            holders.append(cells[1])
    return holders


def require_device_free(device):
    """Refuse a contended card only when the operator demanded exclusivity.

    Sharing an Ascend NPU is normal: another job on a card does not make it
    unusable for this Gate, and refusing by default would block legitimate work
    for a reason unrelated to the code under test. So the default proceeds and
    the occupancy is reported, which lets a reader tell a quiet run from a shared
    one. Only the count is reported - other tenants' process ids are not copied
    into a receipt.
    """
    holders = device_occupancy(device)
    if holders and exclusive_requested():
        raise BridgeError(f'NPU device {device} is already in use by {len(holders)} '
                          f'process(es) and this run requires an exclusive card '
                          f'(TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE is set)')
    return len(holders)


def _examples_observations(succeeded, collected, passed, failed, skipped):
    """The example observations for one outcome.

    One constructor for every path, including the ones that never reach pytest,
    so the payload shape cannot drift between success and failure and a reader of
    a failed receipt sees the same fields as a passing one.
    """
    return {
        'exitCodeZero': bool(succeeded),
        'testsCollected': collected,
        'testsPassed': passed,
        'testsFailed': failed,
        'testsSkipped': skipped,
        'notAllSkipped': collected > 0 and skipped < collected,
        'testsCollectedPositive': collected > 0,
    }


def examples(values, timeout):
    """Run the pinned example tests on one authorized device."""
    source = _source_root()
    device = device_id()
    if device is not None:
        require_device_free(device)
    tests = example_tests(values)
    missing = [name for name in tests if not (source / name).is_file()]
    if missing:
        # 75, not a failure: the tests this Gate was told to run are not in this
        # checkout, so nothing was attempted. The manifest maps 75 to BLOCKED_ENV,
        # which is the honest verdict for "not applicable", and it must be reachable
        # as a status rather than as a crash.
        print(PROG + ': example test file(s) absent from the checkout: ' + ', '.join(missing),
              file=sys.stderr)
        return 75, _examples_observations(False, 0, 0, 0, 0)

    with tempfile.TemporaryDirectory(prefix='tilelang-npu-examples-') as tmp:
        tmp_path = Path(tmp)
        tmp_dir = tmp_path / 'tmp'
        ascend_work = tmp_path / 'ascend_work'
        triton_cache = tmp_path / 'triton_cache'
        tilelang_cache = tmp_path / 'tilelang_cache'
        for d in (tmp_dir, ascend_work, triton_cache, tilelang_cache):
            d.mkdir(parents=True, exist_ok=True)
        report = tmp_path / 'junit.xml'
        log_path = tmp_path / 'pytest.log'

        # The example modules import each other by bare name
        # (for example "from example_vecadd import vector_add"), so every
        # directory holding a selected test is on PYTHONPATH. Rootdir stays the
        # checkout: that is where the shared conftest and the installed package
        # live.
        directories = sorted({str((source / name).parent) for name in tests})
        pp_dirs = [str(source)] + [d for d in directories if d != str(source)]
        # Only the leased card is visible. Without this the examples would see
        # every card on the host and could initialize one this run never leased,
        # which is exactly the collision the engine lease exists to prevent.
        # Each export is its own shell STATEMENT, separated by a newline. Joined
        # with spaces, the pytest invocation became part of the last export, so
        # every pytest argument was parsed as a variable assignment and no test
        # ever ran - a gate that reported "nothing collected" for a reason that
        # had nothing to do with the code under test.
        statements = [
            'export PYTHONPATH=' + shlex.quote(os.pathsep.join(pp_dirs)) + ':"${PYTHONPATH:-}"',
            'export TMPDIR=' + shlex.quote(str(tmp_dir)),
            'export TEMP=' + shlex.quote(str(tmp_dir)),
            'export TMP=' + shlex.quote(str(tmp_dir)),
            'export ASCEND_WORK_PATH=' + shlex.quote(str(ascend_work)),
            'export TRITON_CACHE_DIR=' + shlex.quote(str(triton_cache)),
            'export TILELANG_CACHE_DIR=' + shlex.quote(str(tilelang_cache)),
        ]
        if device:
            statements.append('export ASCEND_RT_VISIBLE_DEVICES=' + shlex.quote(device))
        statements.append(' '.join(
            [shlex.quote(_python(values)), '-m', 'pytest']
            + [shlex.quote(name) for name in tests]
            + ['-p', 'no:cacheprovider',
               '--junit-xml=' + shlex.quote(str(report))]))
        body = os.linesep.join(statements)
        with log_path.open('w') as log:
            try:
                rc, _ = _run_shell(values, body, cwd=source, timeout=timeout, log=log)
            except BridgeError:
                print(PROG + ': example run timed out', file=sys.stderr)
                return 1, _examples_observations(False, 0, 0, 0, 0)

        counts = _junit_counts(report)
        if counts is None:
            # Without the report the counters are unknown. Emitting zeros would
            # claim "no tests were collected", which is a different statement from
            # "the report is missing", so say the real thing and fail.
            print(PROG + f': no readable JUnit report at {report}', file=sys.stderr)
            sys.stderr.write(log_path.read_text(errors='replace')[-4000:])
            return 1, _examples_observations(False, 0, 0, 0, 0)

        collected, skipped, failed = counts['collected'], counts['skipped'], counts['failed']
        observations = _examples_observations(rc == 0, collected, counts['passed'],
                                              failed, skipped)
        if collected == 0 or skipped >= collected or failed > 0:
            sys.stderr.write(log_path.read_text(errors='replace')[-4000:])
    return rc, observations


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    timeout = DEFAULT_TIMEOUT
    if '--timeout' in args:
        index = args.index('--timeout')
        try:
            timeout = float(args[index + 1])
        except (IndexError, ValueError):
            _die(2, PROG + ': --timeout needs a number of seconds')
        del args[index:index + 2]

    try:
        values = machine_values()
    except BridgeError as exc:
        _die(75, PROG + ': ' + str(exc))

    if '--check-config' in args:
        return check_config()
    if '--check-config-examples' in args:
        return check_config_examples()
    if '--build' in args:
        try:
            rc, observations = build(values, timeout)
        except BridgeError as exc:
            _die(75, PROG + ': ' + str(exc))
        print(json.dumps(observations))
        return rc
    if '--examples' in args:
        try:
            rc, observations = examples(values, timeout)
        except BridgeError as exc:
            _die(75, PROG + ': ' + str(exc))
        print(json.dumps(observations))
        return rc
    _die(2, 'usage: ' + PROG + ' (--build | --examples | --check-config | '
             '--check-config-examples) [--timeout SECONDS]')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
