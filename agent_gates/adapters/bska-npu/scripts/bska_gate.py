#!/usr/bin/env python3
"""Host-local NPU and CANNBench Gate for BiShengKernelAgent (bska-npu Adapter).

Scope
-----

This Adapter answers two questions about *this* host and *this* conda
environment, in order:

  * --smoke      does the BiShengKernelAgent runtime import and run a real
                 Triton kernel on an authorized card, and does the CANNBench
                 native extension load and execute on that card?
  * --cannbench  does the vendored official CANNBench evaluator produce a
                 report this project would accept as a score?

There is no container. An earlier revision of this Adapter entered a Docker
container as root; that is removed rather than kept as a fallback, because the
container's login shell pinned ASCEND_RT_VISIBLE_DEVICES and its root identity
resolved a different Torch than the one the native extension was compiled
against. Both produced a green Gate over a run that could not score. Running on
the host as the invoking user removes the whole class: the interpreter that
imports Torch is the interpreter that compiled the extension, and the card the
Gate names is the card the run uses.

Why the extension is checked by calling it
------------------------------------------

A present `_C.so` proves the build finished, not that it fits the Torch this
interpreter loaded. The two are separate installs on this machine (a user-site
Torch and an environment-site Torch), and an extension compiled against one
raises `tensor does not have a device` under the other while every import
still succeeds. So the Gate calls cann_bench_clone on the card and compares the
result: that single call is the ABI check, and it is the failure this Adapter
exists to catch.

Two steps, one direction
------------------------

--cannbench depends on --smoke. A green smoke run says the runtime works; it
says nothing about whether the evaluator can produce a score. Neither step
claims the other's result.

Output contract
---------------

Standard output carries exactly one JSON object and nothing else; every
diagnostic goes to stderr, because the evaluator parses stdout as observations.
Exit code 75 means this machine cannot run the Gate at all - a missing
interpreter, a missing CANN setup, a machine file that cannot be honoured -
which the runner records as BLOCKED_ENV rather than as a failed test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import re
import shlex
import subprocess
import sys
import tempfile

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from agent_gates import adapter_runtime

PROG = 'bska-npu-gate'

ADAPTER_DIR = Path(__file__).resolve().parent.parent

# Keys the machine file may define. Kept in step with machine_env.BSKA_NPU_KEYS;
# the host regression suite compares the two, so a key added on one side cannot
# become configuration the other silently ignores.
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
    'BSKA_NPU_TRITON_ASCEND_VERSION',
)

# Alias read by the repository machine-file parity test.
TOOLCHAIN_KEYS = BSKA_NPU_KEYS

# The keys this Gate cannot run without. The node, claude and build-directory
# keys are optional: an unset executable means "the one on PATH", and an unset
# build directory means the engine's default.
REQUIRED = (
    'BSKA_NPU_CANN_SETUP',
    'BSKA_NPU_CONDA_ENV',
    'BSKA_NPU_CONDA_ENVS_PATH',
    'BSKA_NPU_CANNBENCH_REPO',
    'BSKA_NPU_CANNBENCH_TASK',
    'BSKA_NPU_TRITON_ASCEND_VERSION',
)

# Values that select a file or authorize one run. They are launch parameters,
# never machine facts: a device id authorizes one Gate run, and putting it in a
# file would make a machine look permanently entitled to a card.
NOT_MACHINE_KEYS = frozenset({
    'BSKA_NPU_ENV_FILE',
    'BSKA_NPU_DEVICE_ID',
    'BSKA_NPU_REQUIRE_EXCLUSIVE_DEVICE',
})

_ASSIGN = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
ALLOWED = set(BSKA_NPU_KEYS)

# The native extension is built here, relative to the CANNBench checkout. This
# exact parent directory is what goes on PYTHONPATH: putting .cannbench-build
# itself there makes cann_bench_utils a namespace package whose relative import
# of _C fails.
UTILS_SUBDIR = Path('.cannbench-build') / 'cann_bench_utils'

# A CANNBench floor is bounded well below the manifest timeout: an evaluator
# still running in that window is a diagnosis, not slow progress.
DEFAULT_SMOKE_TIMEOUT = 600.0
DEFAULT_CANNBENCH_TIMEOUT = 3300.0


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
    explicit = (os.environ.get('BSKA_NPU_ENV_FILE') or '').strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise BridgeError(f'BSKA_NPU_ENV_FILE must be an absolute path: {explicit!r}')
        if not path.is_file():
            raise BridgeError(f'BSKA_NPU_ENV_FILE is not a regular file: {explicit!r}')
        return path
    return default_env_file()


def machine_values():
    """The machine parameters, from the file only.

    The launching environment is deliberately not a source: an inherited export
    would let a stale shell point this Gate at another interpreter than the one
    the engine leased for.
    """
    path = resolved_env_file()
    if path is None:
        raise BridgeError(f'no machine file: set BSKA_NPU_ENV_FILE or create '
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
    """The build directory name this Adapter leases, relative to the source root.

    An empty value is a typo, not "use the default": the file spells "unset" by
    leaving the key out, and the engine resolves the same default from the same
    file, so both sides must agree on which tree is leased.
    """
    if 'BSKA_NPU_BUILD_SUBDIR' in values:
        raw = values['BSKA_NPU_BUILD_SUBDIR']
        if not raw.strip():
            raise BridgeError('BSKA_NPU_BUILD_SUBDIR is set but empty')
        raw = raw.strip()
        if Path(raw).is_absolute() or '..' in Path(raw).parts or Path(raw).name != raw:
            raise BridgeError(f'BSKA_NPU_BUILD_SUBDIR must be a single directory '
                              f'name inside the checkout: {raw!r}')
        return raw
    return 'build'


def python_path(values):
    """The interpreter of the configured conda environment."""
    return (Path(_required(values, 'BSKA_NPU_CONDA_ENVS_PATH'))
            / _required(values, 'BSKA_NPU_CONDA_ENV') / 'bin' / 'python')


def cannbench_repo(values):
    return Path(_required(values, 'BSKA_NPU_CANNBENCH_REPO'))


def cannbench_task(values):
    raw = _required(values, 'BSKA_NPU_CANNBENCH_TASK')
    if not re.fullmatch(r'level[1-9][0-9]*/[a-z][a-z0-9_]*', raw):
        raise BridgeError(f'BSKA_NPU_CANNBENCH_TASK must be levelN/operator: {raw!r}')
    return raw


def cannbench_arch(values):
    raw = (values.get('BSKA_NPU_CANNBENCH_ARCH') or '').strip() or 'ascend950'
    if not re.fullmatch(r'ascend[0-9a-z_]+', raw):
        raise BridgeError(f'BSKA_NPU_CANNBENCH_ARCH is not a build target: {raw!r}')
    return raw


def expected_triton_version(values):
    return _required(values, 'BSKA_NPU_TRITON_ASCEND_VERSION')


def device_id():
    """The card this run may use, or None when the engine leased none.

    Read from the environment rather than the machine file: this authorizes one
    run, and a machine file would turn that into a standing entitlement. The
    engine writes it only for a gate that declares it uses a device, so its
    absence is meaningful - it means no lease was taken and this Gate must not
    reach for a card anyway.
    """
    raw = (os.environ.get('BSKA_NPU_DEVICE_ID') or '').strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise BridgeError(f'BSKA_NPU_DEVICE_ID must be a nonnegative decimal ID: {raw!r}')
    return raw


def exclusive_requested():
    raw = (os.environ.get('BSKA_NPU_REQUIRE_EXCLUSIVE_DEVICE') or '').strip().lower()
    return raw in {'1', 'true', 'yes', 'on'}


def require_device():
    device = device_id()
    if device is None:
        raise BridgeError('explicit BSKA_NPU_DEVICE_ID or --device-id authorization '
                          'required for this Gate')
    try:
        adapter_runtime.check_device_ready(int(device), exclusive=exclusive_requested())
    except adapter_runtime.BlockedEnv as exc:
        raise BridgeError(str(exc)) from exc
    return device


def _run(values, body, *, cwd, timeout, extra_env=None):
    """Run a shell body with CANN sourced; return (rc, stdout, stderr).

    HOME is pinned to the invoking user's real home. The layered Torch install
    this Adapter depends on lives in a user-site directory, which Python finds
    through HOME; a run launched under another HOME (sudo, a service unit that
    sets its own) silently resolves the environment-site Torch instead and the
    native extension then fails on the first tensor it touches.
    """
    script = 'set +u' + os.linesep + 'source ' + shlex.quote(_required(values, 'BSKA_NPU_CANN_SETUP')) + os.linesep + body
    env = dict(os.environ)
    env.pop('ASCEND_RT_VISIBLE_DEVICES', None)
    # pwd, not Path.home(): Path.home() reads HOME, so under an overridden HOME
    # it returns the very value this line exists to replace.
    env['HOME'] = pwd.getpwuid(os.getuid()).pw_dir
    env.update(extra_env or {})
    result = subprocess.run(['/bin/bash', '-c', script], cwd=str(cwd), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return result.returncode, result.stdout.decode('utf-8', 'replace'),         result.stderr.decode('utf-8', 'replace')


PROBE = r'''
import json, importlib.metadata as md, os, shutil, subprocess, sys
facts = {}
def note(name, value):
    facts[name] = bool(value)

torch = torchvision = None
triton = tl = None
try:
    import torch as torch
    note("torchImported", True)
except Exception as exc:
    note("torchImported", False)
    print("torch import failed: %s" % exc, file=sys.stderr)
try:
    import torch_npu
    note("torchNpuImported", True)
except Exception as exc:
    note("torchNpuImported", False)
    print("torch_npu import failed: %s" % exc, file=sys.stderr)
try:
    import triton as triton
    import triton.language as tl
    note("tritonImported", True)
except Exception as exc:
    note("tritonImported", False)
    print("triton import failed: %s" % exc, file=sys.stderr)
try:
    import torchvision as torchvision
    note("torchvisionImported", True)
except Exception as exc:
    note("torchvisionImported", False)
    print("torchvision import failed: %s" % exc, file=sys.stderr)

facts["torchVersion"] = getattr(torch, "__version__", "")
facts["torchvisionVersion"] = getattr(torchvision, "__version__", "")
facts["torchNpuVersion"] = getattr(sys.modules.get("torch_npu"), "__version__", "")
facts["pythonVersion"] = sys.version.split()[0]

expected = os.environ.get("BSKA_EXPECT_TRITON_VERSION", "").strip()
installed = ""
for dist in ("triton-ascend", "triton_ascend"):
    try:
        installed = md.version(dist) or ""
        break
    except Exception:
        continue
note("tritonAscendInstalled", bool(installed))
facts["tritonAscendVersion"] = installed
facts["tritonAscendExpectedVersion"] = expected
note("tritonAscendVersionMatch", installed == expected if expected else False)

utils = os.environ.get("BSKA_UTILS_PATH", "")
note("cannBenchUtilsPresent", bool(utils) and os.path.isfile(os.path.join(utils, "cann_bench_utils", "_C.so")))

if torch is not None:
    try:
        note("npuAvailable", torch.npu.is_available())
        note("deviceCountPositive", torch.npu.device_count() > 0)
    except Exception as exc:
        note("npuAvailable", False)
        note("deviceCountPositive", False)
        print("npu query failed: %s" % exc, file=sys.stderr)
    try:
        torch.npu.set_device(0)
        x = torch.randn(1024, device="npu", dtype=torch.float32)
        y = torch.randn(1024, device="npu", dtype=torch.float32)
        out = torch.empty_like(x)
        note("tensorOnDevice", True)
        if tl is not None:
            @triton.jit
            def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                tl.store(out_ptr + offs,
                         tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
                         mask=mask)
            add_kernel[(triton.cdiv(1024, 128),)](x, y, out, 1024, BLOCK=128)
            torch.npu.synchronize()
            note("synchronizeOk", True)
            note("numericalOk", bool(torch.allclose(out, x + y, atol=1e-3, rtol=1e-3)))
        else:
            note("numericalOk", False)
            note("synchronizeOk", False)
    except Exception as exc:
        print("device workload failed: %s" % exc, file=sys.stderr)
        facts.setdefault("tensorOnDevice", False)
        facts.setdefault("numericalOk", False)
        facts.setdefault("synchronizeOk", False)

# The ABI check: an extension built against a different Torch imports cleanly and
# only fails when it touches a tensor. Calling it on the card is the measurement.
callable_ok = False
if facts.get("cannBenchUtilsPresent") and facts.get("tensorOnDevice"):
    try:
        import cann_bench_utils
        probe = torch.arange(4096, dtype=torch.float16, device="npu")
        copied = cann_bench_utils.cann_bench_clone(probe)
        torch.npu.synchronize()
        callable_ok = bool(torch.equal(copied, probe))
    except Exception as exc:
        print("native extension call failed: %s" % exc, file=sys.stderr)
note("cannBenchUtilsCallable", callable_ok)

helix = os.environ.get("BSKA_HELIX_BIN", "")
note("helixAvailable", bool(helix) and os.path.isfile(helix)
     and subprocess.run([helix, "--help"], capture_output=True).returncode == 0)
for label, key in (("nodeAvailable", "BSKA_NODE_BIN"), ("claudeAvailable", "BSKA_CLAUDE_BIN")):
    binary = os.environ.get(key, "")
    note(label, bool(binary) and shutil.which(binary) is not None
         and subprocess.run([binary, "--version"], capture_output=True).returncode == 0)

print(json.dumps(facts))
'''


def _smoke_facts(values, device):
    python = python_path(values)
    utils = cannbench_repo(values) / UTILS_SUBDIR
    helix = python.parent / 'helix'
    # The probe must live in a real file: Triton rejects a @jit kernel defined in
    # a "python -c" string because it resolves the kernel's source by file name.
    with tempfile.TemporaryDirectory(prefix='bska-smoke-') as tmp:
        tmp_path = Path(tmp)
        triton_cache = tmp_path / 'triton-cache'
        ascend_work = tmp_path / 'ascend-work'
        triton_cache.mkdir(parents=True, exist_ok=True)
        ascend_work.mkdir(parents=True, exist_ok=True)
        extra = {
            'ASCEND_RT_VISIBLE_DEVICES': device,
            'PYTHONPATH': str(utils),
            'BSKA_UTILS_PATH': str(utils),
            'BSKA_EXPECT_TRITON_VERSION': expected_triton_version(values),
            'BSKA_HELIX_BIN': str(helix),
            'BSKA_NODE_BIN': (values.get('BSKA_NPU_NODE_BIN') or '').strip() or 'node',
            'BSKA_CLAUDE_BIN': (values.get('BSKA_NPU_CLAUDE_BIN') or '').strip() or 'claude',
            'TRITON_CACHE_DIR': str(triton_cache),
            'ASCEND_WORK_PATH': str(ascend_work),
            'TMPDIR': str(tmp_path),
        }
        probe = tmp_path / 'probe.py'
        probe.write_text(PROBE)
        rc, stdout, stderr = _run(
            values,
            'exec ' + ' '.join([shlex.quote(str(python)), shlex.quote(str(probe))]),
            cwd=Path.cwd(), timeout=DEFAULT_SMOKE_TIMEOUT, extra_env=extra)
    facts = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{'):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                facts = parsed
                break
    if stderr.strip():
        sys.stderr.write(stderr[-4000:])
    if not facts:
        raise BridgeError(f'probe produced no observations (exit {rc})')
    return facts


# Observations the probe measures. The two path facts are established by
# check_config, which already ran as this recipe's configCheck, so they are not
# re-measured here and must not be listed.
SMOKE_CHECKS = (
    'torchImported', 'torchNpuImported', 'tritonImported', 'torchvisionImported',
    'tritonAscendInstalled', 'tritonAscendVersionMatch', 'npuAvailable',
    'deviceCountPositive', 'tensorOnDevice', 'numericalOk', 'synchronizeOk',
    'cannBenchUtilsPresent', 'cannBenchUtilsCallable', 'nodeAvailable',
    'claudeAvailable', 'helixAvailable',
)

SMOKE_FACTS = ('torchVersion', 'torchNpuVersion', 'torchvisionVersion',
               'tritonAscendVersion', 'tritonAscendExpectedVersion', 'pythonVersion')


def smoke(values):
    """Prove the runtime and the native extension work on one authorized card."""
    device = require_device()
    facts = _smoke_facts(values, device)
    observations = {
        'cannSetupReadable': True,
        'condaEnvPresent': True,
        'deviceId': device,
    }
    for key in SMOKE_CHECKS:
        observations[key] = bool(facts.get(key, False))
    for key in SMOKE_FACTS:
        observations[key] = str(facts.get(key, ''))
    ok = all(observations[key] is True for key in SMOKE_CHECKS)
    return (0 if ok else 1), observations


def _report_facts(path):
    """Counts from an official report, or None when it cannot be read.

    `accuracy` is null in a report whose case never reached the accuracy stage,
    so it is normalised before it is queried rather than assumed to be a dict.
    """
    try:
        report = json.loads(Path(path).read_text())
        operator = report['operators'][0]
        cases = operator['cases']
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    passed = sum(1 for case in cases
                 if (case.get('accuracy') or {}).get('passed') is True
                 and case.get('status') == 'success')
    timed = sum(1 for case in cases
                if isinstance(case.get('elapsed_us'), (int, float))
                and case['elapsed_us'] > 0
                and not case.get('perf_collection_failed'))
    score = operator.get('score')
    return len(cases), passed, timed, score


def cannbench(values, timeout):
    """Run the vendored official evaluator and report what it produced.

    The runner already refuses a report it considers incomplete, so its exit
    status is the verdict; the parsed counts are published so a reader can tell
    a full pass from a run that never started. They are facts, never gate
    checks: the case count belongs to the operator, not to this machine.
    """
    device = require_device()
    repo = cannbench_repo(values)
    task = cannbench_task(values)
    python = python_path(values)
    runner = repo / 'tools' / 'run_cannbench.py'
    if not runner.is_file():
        raise BridgeError(f'{repo} does not hold tools/run_cannbench.py')
    with tempfile.TemporaryDirectory(prefix='bska-cannbench-') as tmp:
        reports = Path(tmp) / 'reports'
        body = 'exec ' + ' '.join([
            shlex.quote(str(python)), shlex.quote(str(runner)), shlex.quote(task),
            '--device', shlex.quote(device), '--reports-dir', shlex.quote(str(reports)),
        ])
        rc, stdout, stderr = _run(values, body, cwd=repo, timeout=timeout)
        sys.stderr.write(stdout[-2000:])
        sys.stderr.write(stderr[-4000:])
        parsed = _report_facts(reports / 'official_final.json')

    # A missing report and an empty one are the same verdict - the run produced
    # no score - so the counts default to zero and the observations still get
    # published. A receipt is owed on every path, including the failed ones.
    count, passed, timed, score = parsed or (0, 0, 0, None)
    observations = {
        'officialReportValid': rc == 0,
        'allCasesPassed': count > 0 and passed == count,
        'allTimingsValid': count > 0 and timed == count,
        'caseCount': count,
        'passedCases': passed,
        'timedCases': timed,
        'score': round(float(score), 6) if isinstance(score, (int, float)) else 0.0,
        'deviceId': device,
        'taskId': task,
    }
    if rc != 0:
        print(PROG + f': official report rejected by the runner (exit {rc})', file=sys.stderr)
    return rc, observations


def check_config():
    """Preflight: is the machine file complete and are the checkouts present?

    Deliberately does not import Torch, does not touch a card and does not run
    the evaluator. A configuration check that costs minutes is one operators
    skip, and a skipped check is not a check.
    """
    try:
        values = machine_values()
        for name in REQUIRED:
            _required(values, name)
        build_subdir(values)
        cannbench_arch(values)
        task = cannbench_task(values)
        python = python_path(values)
        if not python.is_file():
            raise BridgeError(f'{python} does not exist; BSKA_NPU_CONDA_ENV and '
                              f'BSKA_NPU_CONDA_ENVS_PATH name no interpreter')
        for name in ('BSKA_NPU_CANN_SETUP',):
            path = Path(_required(values, name))
            if not path.is_file():
                raise BridgeError(f'{name} does not exist: {path}')
        repo = cannbench_repo(values)
        if not (repo / 'tools' / 'run_cannbench.py').is_file():
            raise BridgeError(f'{repo} does not hold tools/run_cannbench.py '
                              f'(BSKA_NPU_CANNBENCH_REPO)')
        if not (repo / 'operators' / task).is_dir():
            raise BridgeError(f'{repo} has no operator {task} '
                              f'(BSKA_NPU_CANNBENCH_TASK)')
        utils = repo / UTILS_SUBDIR / 'cann_bench_utils' / '_C.so'
        if not utils.is_file():
            raise BridgeError(f'native extension missing at {utils}; run '
                              f'tools/build_cannbench_utils.sh {cannbench_arch(values)} '
                              f'with PYTHON={python}')
        if (os.environ.get('BSKA_NPU_DEVICE_ID') or '').strip():
            require_device()
    except BridgeError as exc:
        _die(75, PROG + ': ' + str(exc))
    print(PROG + ': configuration ok', file=sys.stderr)
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    timeout = DEFAULT_CANNBENCH_TIMEOUT
    if '--timeout' in args:
        index = args.index('--timeout')
        try:
            timeout = float(args[index + 1])
        except (IndexError, ValueError):
            _die(2, PROG + ': --timeout needs a number of seconds')
        del args[index:index + 2]

    if '--check-config' in args:
        return check_config()

    try:
        values = machine_values()
    except BridgeError as exc:
        _die(75, PROG + ': ' + str(exc))

    if '--smoke' in args:
        try:
            rc, observations = smoke(values)
        except BridgeError as exc:
            _die(75, PROG + ': ' + str(exc))
        except subprocess.TimeoutExpired:
            _die(75, PROG + ': smoke probe timed out')
        print(json.dumps(observations))
        return rc
    if '--cannbench' in args:
        try:
            rc, observations = cannbench(values, timeout)
        except BridgeError as exc:
            _die(75, PROG + ': ' + str(exc))
        except subprocess.TimeoutExpired:
            _die(75, PROG + ': CANNBench run timed out')
        print(json.dumps(observations))
        return rc
    _die(2, 'usage: ' + PROG + ' (--smoke | --cannbench | --check-config) '
             '[--timeout SECONDS]')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
