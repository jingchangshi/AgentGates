#!/usr/bin/env python3
"""Deterministic NPU smoke gate for BiShengKernelAgent (bska-npu adapter).

This script is executed by AgentGates on the HOST. It enters the configured
container with docker exec and proves that the BiShengKernelAgent (helix)
runtime is usable on a real Ascend NPU:

  * CANN is sourced and the configured conda environment imports torch,
    torch_npu and triton;
  * a real Triton vector-add kernel runs on the authorized device and the
    result matches the Torch reference;
  * node, claude and helix are available inside the container.

The machine file is parsed as strict NAME=value data and never executed.
Only --smoke prints the machine-readable observation object to stdout; every
diagnostic goes to stderr. Exit code 75 means the environment is unavailable,
which AgentGates records as BLOCKED_ENV rather than a failed test.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PREFIX = "BSKA_NPU_"

# Keep in step with agent_gates/machine_env.py BSKA_NPU_KEYS.
BSKA_NPU_KEYS = (
    "BSKA_NPU_CONTAINER_NAME",
    "BSKA_NPU_CONTAINER_USER",
    "BSKA_NPU_CONTAINER_HOME",
    "BSKA_NPU_CANN_SETUP",
    "BSKA_NPU_CANN_EXTRA_LIB",
    "BSKA_NPU_CONDA_SH",
    "BSKA_NPU_CONDA_ENV",
    "BSKA_NPU_CONDA_ENVS_PATH",
    "BSKA_NPU_PYTHON_VERSION",
    "BSKA_NPU_HELIX_REPO",
    "BSKA_NPU_BUILD_SUBDIR",
    "BSKA_NPU_NODE_BIN",
    "BSKA_NPU_CLAUDE_BIN",
)

# Alias used by the repository machine-file parity test: the keys this script
# accepts are exactly the keys machine_env declares for this Adapter.
TOOLCHAIN_KEYS = BSKA_NPU_KEYS

REQUIRED = (
    "BSKA_NPU_CONTAINER_NAME",
    "BSKA_NPU_CONTAINER_USER",
    "BSKA_NPU_CONTAINER_HOME",
    "BSKA_NPU_CANN_SETUP",
    "BSKA_NPU_CONDA_ENV",
    "BSKA_NPU_CONDA_ENVS_PATH",
)

_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
ALLOWED = set(BSKA_NPU_KEYS) | {PREFIX + "ENV_FILE", PREFIX + "DEVICE_ID"}


class BridgeError(ValueError):
    """A machine file that cannot be honoured; mapped to BLOCKED_ENV by the CLI."""


def _die(code, message):
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _env_file_path():
    explicit = (os.environ.get(PREFIX + "ENV_FILE") or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute() or not path.is_file():
            _die(75, f"{PREFIX}ENV_FILE must be an existing absolute file: {explicit!r}")
        return path
    candidate = Path(__file__).resolve().parent.parent / "env.sh"
    if not candidate.is_file():
        _die(75, f"no machine file: set {PREFIX}ENV_FILE or create {candidate}")
    return candidate


def _parse_env_file(text, path):
    values = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGN.match(line)
        if match is None:
            raise BridgeError(f"{path}: line {number} is not a NAME=value assignment: {raw!r}")
        key, value = match.group(1), match.group(2).strip()
        if key not in BSKA_NPU_KEYS:
            raise BridgeError(f"{path}: line {number} sets unknown key {key!r}")
        if key in values:
            raise BridgeError(f"{path}: line {number} sets {key} a second time")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        elif value[:1] in ('"', "'") or value[-1:] in ('"', "'"):
            raise BridgeError(f"{path}: line {number} has an unbalanced quote")
        values[key] = value
    return values


def _load(path):
    try:
        return _parse_env_file(path.read_text(encoding="utf-8"), path)
    except BridgeError as exc:
        _die(75, str(exc))


def _docker(argv, timeout=60, stdin=None):
    return subprocess.run(
        ["docker", *argv],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _check_config(values):
    for key in REQUIRED:
        if not values.get(key, "").strip():
            _die(75, f"{key} is required in the machine file")
    name = values["BSKA_NPU_CONTAINER_NAME"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        _die(75, f"BSKA_NPU_CONTAINER_NAME is not a plain Docker name: {name!r}")
    try:
        state = _docker(["inspect", "--format", "{{.State.Running}}", name], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _die(75, f"docker is unavailable: {exc}")
    if state.returncode != 0:
        _die(75, f"container {name} does not exist: {state.stderr.strip()}")
    if state.stdout.strip() != "true":
        _die(75, f"container {name} is not running")

    envs_path = values["BSKA_NPU_CONDA_ENVS_PATH"]
    conda_env = values["BSKA_NPU_CONDA_ENV"]
    python = f"{envs_path}/{conda_env}/bin/python"
    checks = [
        ("CANN setup", "test -r " + shlex.quote(values["BSKA_NPU_CANN_SETUP"])),
        ("conda python", "test -x " + shlex.quote(python)),
    ]
    for label, command in checks:
        result = _docker(["exec", name, "bash", "-lc", command], timeout=30)
        if result.returncode != 0:
            _die(75, f"{label} is not available in container {name}")
    print("bska-npu config OK", file=sys.stderr)
    return 0


_PROBE = r'''
import json, os, shutil, subprocess, sys, traceback

facts = {}
def record(name, value):
    facts[name] = bool(value)

torch = None
triton = None
tl = None
try:
    import torch  # noqa: F401
    record("torchImported", True)
except Exception as exc:
    record("torchImported", False)
    print("torch import failed:", exc, file=sys.stderr)
try:
    import torch_npu  # noqa: F401
    record("torchNpuImported", True)
except Exception as exc:
    record("torchNpuImported", False)
    print("torch_npu import failed:", exc, file=sys.stderr)
try:
    import triton
    import triton.language as tl  # noqa: F401
    record("tritonImported", True)
except Exception as exc:
    record("tritonImported", False)
    print("triton import failed:", exc, file=sys.stderr)

if torch is not None and facts.get("torchNpuImported"):
    try:
        record("npuAvailable", torch.npu.is_available())
        record("deviceCountPositive", torch.npu.device_count() > 0)
    except Exception as exc:
        record("npuAvailable", False)
        record("deviceCountPositive", False)
        print("npu query failed:", exc, file=sys.stderr)
    try:
        torch.npu.set_device(0)
        N = 1024
        x = torch.randn(N, device="npu", dtype=torch.float32)
        y = torch.randn(N, device="npu", dtype=torch.float32)
        out = torch.empty_like(x)
        record("tensorOnDevice", True)
        if tl is not None:
            @triton.jit
            def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                a = tl.load(x_ptr + offs, mask=mask)
                b = tl.load(y_ptr + offs, mask=mask)
                tl.store(out_ptr + offs, a + b, mask=mask)
            add_kernel[(triton.cdiv(N, 128),)](x, y, out, N, BLOCK=128)
            torch.npu.synchronize()
            record("synchronizeOk", True)
            ref = x + y
            record("numericalOk", bool(torch.allclose(out, ref, atol=1e-3, rtol=1e-3)))
        else:
            record("numericalOk", False)
            record("synchronizeOk", False)
    except Exception as exc:
        print("device workload failed:", exc, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        facts.setdefault("tensorOnDevice", False)
        facts.setdefault("numericalOk", False)
        facts.setdefault("synchronizeOk", False)

node = shutil.which("node")
record("nodeAvailable", bool(node))
if node:
    try:
        record("nodeAvailable", subprocess.run([node, "--version"], capture_output=True).returncode == 0)
    except Exception:
        pass

claude = os.environ.get("BSKA_CLAUDE_BIN") or "/usr/local/bin/claude"
try:
    env = dict(os.environ)
    env["HOME"] = os.environ.get("BSKA_CONTAINER_HOME", env.get("HOME", "/root"))
    record("claudeAvailable", subprocess.run([claude, "--version"], capture_output=True, env=env).returncode == 0)
except Exception as exc:
    print("claude check failed:", exc, file=sys.stderr)
    record("claudeAvailable", False)

helix = os.path.join(os.environ.get("BSKA_ENVS_PATH", ""), os.environ.get("BSKA_CONDA_ENV", ""), "bin", "helix")
try:
    record("helixAvailable", os.path.isfile(helix) and subprocess.run([helix, "--help"], capture_output=True).returncode == 0)
except Exception:
    record("helixAvailable", False)

print(json.dumps(facts))
'''


def _smoke(values):
    device = (os.environ.get(PREFIX + "DEVICE_ID") or "").strip()
    if not re.fullmatch(r"[0-9]+", device):
        _die(75, f"explicit {PREFIX}DEVICE_ID or --device-id authorization required")
    name = values["BSKA_NPU_CONTAINER_NAME"]
    envs_path = values["BSKA_NPU_CONDA_ENVS_PATH"]
    conda_env = values["BSKA_NPU_CONDA_ENV"]
    home = values["BSKA_NPU_CONTAINER_HOME"]
    cann = values["BSKA_NPU_CANN_SETUP"]
    extra_lib = values.get("BSKA_NPU_CANN_EXTRA_LIB", "").strip()
    node_bin = values.get("BSKA_NPU_NODE_BIN", "/usr/local/bin/node")
    claude_bin = values.get("BSKA_NPU_CLAUDE_BIN", "/usr/local/bin/claude")

    inner = (
        'source "$BSKA_CANN"; '
        'if [ -n "$BSKA_CANN_EXTRA_LIB" ]; then export LD_LIBRARY_PATH="$BSKA_CANN_EXTRA_LIB:$LD_LIBRARY_PATH"; fi; '
        'export ASCEND_RT_VISIBLE_DEVICES="$BSKA_DEVICE"; '
        'export PATH="/usr/local/bin:$PATH"; '
        'export TRITON_CACHE_DIR=/tmp/bska-triton-cache; '
        'cat > /tmp/bska_probe.py; '
        '"$BSKA_ENVS_PATH/$BSKA_CONDA_ENV/bin/python" /tmp/bska_probe.py; rc=$?; '
        'rm -f /tmp/bska_probe.py; exit $rc'
    )
    argv = [
        "exec", "-i", "-u", "root",
        "-e", "BSKA_CANN=" + cann,
        "-e", "BSKA_CANN_EXTRA_LIB=" + extra_lib,
        "-e", "BSKA_ENVS_PATH=" + envs_path,
        "-e", "BSKA_CONDA_ENV=" + conda_env,
        "-e", "BSKA_CONTAINER_HOME=" + home,
        "-e", "BSKA_DEVICE=" + device,
        "-e", "BSKA_NODE_BIN=" + node_bin,
        "-e", "BSKA_CLAUDE_BIN=" + claude_bin,
        name, "bash", "-lc", inner,
    ]
    try:
        result = _docker(argv, timeout=1500, stdin=_PROBE)
    except subprocess.TimeoutExpired:
        _die(75, "in-container probe timed out")
    except OSError as exc:
        _die(75, f"docker exec failed: {exc}")

    facts = {}
    for line in reversed((result.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                facts = parsed
                break
    if (result.stderr or "").strip():
        print((result.stderr or "")[-4000:], file=sys.stderr)
    if not facts:
        _die(75, f"in-container probe produced no observations (exit {result.returncode})")

    observations = {
        "containerRunning": True,
        "cannSetupReadable": True,
        "condaEnvPresent": True,
    }
    for key in (
        "torchImported", "torchNpuImported", "tritonImported", "npuAvailable",
        "deviceCountPositive", "tensorOnDevice", "numericalOk", "synchronizeOk",
        "nodeAvailable", "claudeAvailable", "helixAvailable",
    ):
        observations[key] = bool(facts.get(key, False))
    print(json.dumps(observations))
    return 0


def main(argv):
    mode = None
    for arg in argv:
        if arg == "--check-config":
            mode = "check"
        elif arg == "--smoke":
            mode = "smoke"
        elif arg.startswith("--device-id="):
            os.environ[PREFIX + "DEVICE_ID"] = arg.split("=", 1)[1]
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            _die(2, f"unknown argument: {arg}")
    if mode is None:
        _die(2, "one of --check-config or --smoke is required")
    values = _load(_env_file_path())
    if mode == "check":
        return _check_config(values)
    return _smoke(values)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
