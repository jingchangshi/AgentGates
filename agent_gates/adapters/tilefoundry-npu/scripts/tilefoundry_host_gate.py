#!/usr/bin/env python3
"""Host-local TileFoundry CPU and Ascend NPU verification gate."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
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

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from agent_gates import adapter_runtime

PROG = "tilefoundry-npu-gate"
ADAPTER_DIR = Path(__file__).resolve().parent.parent

TOOLCHAIN_KEYS = (
    "TILEFOUNDRY_NPU_CANN_SETUP",
    "TILEFOUNDRY_NPU_CONDA_SH",
    "TILEFOUNDRY_NPU_CONDA_ENV",
    "TILEFOUNDRY_NPU_CONDA_ENVS_PATH",
    "TILEFOUNDRY_NPU_PYTHON",
    "TILEFOUNDRY_NPU_CPU_TESTS",
    "TILEFOUNDRY_NPU_DEVICE_TESTS",
    "TILEFOUNDRY_NPU_BUILD_SUBDIR",
)

NOT_MACHINE_KEYS = frozenset({
    "TILEFOUNDRY_NPU_ENV_FILE",
    "TILEFOUNDRY_NPU_DEVICE_ID",
    "TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE",
})

REQUIRED = (
    "TILEFOUNDRY_NPU_CANN_SETUP",
    "TILEFOUNDRY_NPU_CONDA_SH",
    "TILEFOUNDRY_NPU_CONDA_ENV",
)

DEFAULT_CPU_TESTS = (
    "tests/tilefoundry",
    "tests/ascend/target",
    "tests/ascend/ir",
    "tests/ascend/passes",
    "tests/ascend/operators",
)

DEFAULT_PYTHONPATH = "src:targets/ascend:targets/apple:targets/nvidia"
DEFAULT_CPU_TIMEOUT = 600.0
DEFAULT_DEVICE_TIMEOUT = 600.0

DEVICE_PROBE = r"""
import json, sys, torch, torch_npu
facts = {
    "tilefoundryImported": False,
    "npuAvailable": False,
    "deviceCountOne": False,
    "tensorOnDevice": False,
    "fp32VectorOk": False,
    "bf16MatmulOk": False,
    "numericalOk": False,
    "synchronizeOk": False,
    "deviceMemoryPositive": False,
}
try:
    import tilefoundry  # noqa: F401
    facts["tilefoundryImported"] = True
except Exception as exc:
    print(f"tilefoundry import failed: {exc}", file=sys.stderr)

try:
    avail = bool(torch.npu.is_available())
    facts["npuAvailable"] = avail
    count = int(torch.npu.device_count()) if avail else 0
    facts["deviceCountOne"] = (count == 1)
    if avail and count >= 1:
        torch.npu.set_device(0)
        a_cpu = torch.linspace(-2.0, 2.0, 256, dtype=torch.float32)
        b_cpu = torch.linspace(1.0, 3.0, 256, dtype=torch.float32)
        a_npu = a_cpu.to("npu:0")
        b_npu = b_cpu.to("npu:0")
        facts["tensorOnDevice"] = (a_npu.device.type == "npu" and b_npu.device.type == "npu")
        c_npu = (a_npu + b_npu) * 2.0
        torch.npu.synchronize()
        facts["fp32VectorOk"] = bool(torch.allclose(c_npu.cpu(), (a_cpu + b_cpu) * 2.0, atol=1e-5, rtol=1e-5))

        m1_cpu = torch.randn((64, 64), dtype=torch.bfloat16)
        m2_cpu = torch.randn((64, 64), dtype=torch.bfloat16)
        out_npu = torch.matmul(m1_cpu.to("npu:0"), m2_cpu.to("npu:0"))
        torch.npu.synchronize()
        ref = torch.matmul(m1_cpu.float(), m2_cpu.float())
        facts["bf16MatmulOk"] = bool(torch.allclose(out_npu.cpu().float(), ref, atol=1.0, rtol=5e-2))
        facts["numericalOk"] = facts["fp32VectorOk"] and facts["bf16MatmulOk"]
        facts["synchronizeOk"] = True
        facts["deviceMemoryPositive"] = int(torch.npu.max_memory_allocated(0)) > 0
except Exception as exc:
    print(f"npu probe failed: {exc}", file=sys.stderr)

print(json.dumps(facts, sort_keys=True))
sys.exit(0 if all(facts.values()) else 1)
"""


class BridgeError(RuntimeError):
    EXIT = adapter_runtime.BLOCKED_ENV


def _die(code: int, message: str) -> None:
    print(f"{PROG}: {message}", file=sys.stderr)
    raise SystemExit(code)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_env_file(text: str, path: Path) -> dict[str, str]:
    candidate = Path(path)
    if candidate.is_file() and not candidate.is_symlink():
        try:
            if candidate.read_text(encoding="utf-8") == text:
                return adapter_runtime.parse_env_file(candidate, TOOLCHAIN_KEYS)
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=True) as tmp:
        tmp.write(text)
        tmp.flush()
        try:
            return adapter_runtime.parse_env_file(Path(tmp.name), TOOLCHAIN_KEYS)
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc


def resolved_env_file() -> Path:
    try:
        path = adapter_runtime.resolve_env_file("TILEFOUNDRY_NPU_ENV_FILE", ADAPTER_DIR)
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
    if path is None:
        raise BridgeError(
            f"no machine file: set TILEFOUNDRY_NPU_ENV_FILE or create {ADAPTER_DIR / 'env.sh'}"
        )
    return path


def machine_values() -> dict[str, str]:
    path = resolved_env_file()
    try:
        values = adapter_runtime.parse_env_file(path, TOOLCHAIN_KEYS)
    except ValueError as exc:
        raise BridgeError(str(exc)) from exc
    for name in sorted(os.environ):
        if not name.startswith("TILEFOUNDRY_NPU_") or name in NOT_MACHINE_KEYS:
            continue
        if name not in TOOLCHAIN_KEYS:
            raise BridgeError(f"{name} is exported but is not a machine-file key")
        if name not in values:
            raise BridgeError(f"{name} is exported in shell but absent from {path}")
        if os.environ[name] != values[name]:
            raise BridgeError(f"{name} export disagrees with {path}")
    return values


def _required(values: dict[str, str], name: str) -> str:
    value = (values.get(name) or "").strip()
    if not value:
        raise BridgeError(f"{name} is required in the machine file")
    return value


def build_subdir(values: dict[str, str]) -> str:
    if "TILEFOUNDRY_NPU_BUILD_SUBDIR" in values:
        raw = values["TILEFOUNDRY_NPU_BUILD_SUBDIR"].strip()
        if not raw:
            raise BridgeError("TILEFOUNDRY_NPU_BUILD_SUBDIR is set but empty")
        try:
            return adapter_runtime.validate_build_subdir(raw)
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc
    return "build"


def python_path(values: dict[str, str]) -> Path:
    explicit = (values.get("TILEFOUNDRY_NPU_PYTHON") or "").strip()
    if explicit:
        return Path(explicit)
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
    if envs_path:
        return Path(envs_path) / conda_env / "bin" / "python3"
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    return conda_sh.parent.parent.parent / "envs" / conda_env / "bin" / "python3"


def _parse_test_list(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return default
    parts = [item.strip() for item in re.split(r"[\s,]+", raw.strip()) if item.strip()]
    if not parts:
        raise BridgeError("test list is set but names no path")
    for item in parts:
        p = Path(item)
        if p.is_absolute() or ".." in p.parts:
            raise BridgeError(f"test path must be relative inside source root: {item!r}")
    return tuple(parts)


def cpu_tests(values: dict[str, str]) -> tuple[str, ...]:
    return _parse_test_list(values.get("TILEFOUNDRY_NPU_CPU_TESTS"), DEFAULT_CPU_TESTS)


def device_tests(values: dict[str, str]) -> tuple[str, ...]:
    return _parse_test_list(values.get("TILEFOUNDRY_NPU_DEVICE_TESTS"), ())


def device_id() -> str | None:
    raw = os.environ.get("TILEFOUNDRY_NPU_DEVICE_ID")
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if not re.fullmatch(r"[0-9]+", cleaned):
        raise BridgeError(f"TILEFOUNDRY_NPU_DEVICE_ID must be a nonnegative decimal ID, got {raw!r}")
    return str(int(cleaned))


def require_device() -> str:
    dev = device_id()
    if dev is None:
        raise BridgeError("explicit TILEFOUNDRY_NPU_DEVICE_ID or --device-id authorization required")
    return dev


def exclusive_requested() -> bool:
    raw = (os.environ.get("TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _source_root() -> Path:
    source = Path.cwd().resolve()
    if not ((source / "src" / "tilefoundry").is_dir() or (source / "pyproject.toml").is_file()):
        raise BridgeError(f"{source} does not look like a TileFoundry checkout")
    return source


def _junit_counts(path: Path) -> dict[str, int] | None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        return None
    collected = sum(int(s.attrib.get("tests", 0)) for s in suites)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
    failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    failed = errors + failures
    passed = max(0, collected - failed - skipped)
    return {
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }


def _cpu_observations(succeeded: bool, collected: int, passed: int, failed: int, skipped: int) -> dict:
    return {
        "exitCodeZero": bool(succeeded),
        "testsCollected": collected,
        "testsPassed": passed,
        "testsFailed": failed,
        "testsSkipped": skipped,
        "testsCollectedPositive": collected > 0,
        "notAllSkipped": collected > 0 and skipped < collected,
    }


def check_config(*, check_device: bool = False) -> int:
    try:
        values = machine_values()
        for name in REQUIRED:
            _required(values, name)
        build_subdir(values)
        cann = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
        if not cann.is_file():
            raise BridgeError(f"TILEFOUNDRY_NPU_CANN_SETUP does not exist: {cann}")
        conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
        if not conda_sh.is_file():
            raise BridgeError(f"TILEFOUNDRY_NPU_CONDA_SH does not exist: {conda_sh}")
        python = python_path(values)
        if not python.is_file():
            raise BridgeError(f"TileFoundry Python interpreter does not exist: {python}")
        source = _source_root()
        for rel in cpu_tests(values):
            if not (source / rel).exists():
                raise BridgeError(f"CPU test path missing in {source}: {rel}")
        for rel in device_tests(values):
            if not (source / rel).exists():
                raise BridgeError(f"device test path missing in {source}: {rel}")
        if check_device:
            dev = device_id()
            if dev is not None:
                try:
                    adapter_runtime.check_device_ready(int(dev), exclusive=exclusive_requested())
                except adapter_runtime.BlockedEnv as exc:
                    raise BridgeError(str(exc)) from exc
    except BridgeError as exc:
        _die(75, str(exc))
    print(f"{PROG}: configuration ok", file=sys.stderr)
    return 0


def run_cpu(values: dict[str, str], timeout: float) -> tuple[int, dict]:
    source = _source_root()
    tests = cpu_tests(values)
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)

    with tempfile.TemporaryDirectory(prefix="tilefoundry-cpu-") as tmp:
        report = Path(tmp) / "junit.xml"
        run_env = dict(os.environ)
        envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
        if envs_path:
            run_env["CONDA_ENVS_PATH"] = envs_path
        run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        run_env["PYTHONDONTWRITEBYTECODE"] = "1"
        run_env["TILELANG_DISABLE_CACHE"] = "1"
        run_env["PYTHONPATH"] = DEFAULT_PYTHONPATH
        # CPU suite deliberately does NOT source CANN set_env.sh because
        # CANN's native libraries collide with ortools CP-SAT in this env.
        cmd = " ".join(
            [shlex.quote(str(python)), "-m", "pytest", "-p", "no:cacheprovider",
             f"--junit-xml={shlex.quote(str(report))}"]
            + [shlex.quote(t) for t in tests]
        )
        try:
            completed = adapter_runtime.run_host_bash(
                cmd,
                cwd=source,
                env=run_env,
                cann_setup=None,
                conda_sh=conda_sh,
                conda_env=conda_env,
                timeout=timeout,
                isolate_temp=True,
            )
        except subprocess.TimeoutExpired as exc:
            if exc.stderr:
                sys.stderr.write(str(exc.stderr)[-4000:])
            print(f"{PROG}: CPU pytest timed out after {timeout}s", file=sys.stderr)
            return 1, _cpu_observations(False, 0, 0, 0, 0)

        counts = _junit_counts(report)
        if counts is None:
            sys.stderr.write((completed.stderr or completed.stdout)[-4000:])
            print(f"{PROG}: missing JUnit report at {report}", file=sys.stderr)
            return 1, _cpu_observations(False, 0, 0, 0, 0)

        succeeded = (
            completed.returncode == 0
            and counts["collected"] > 0
            and counts["failed"] == 0
            and counts["skipped"] < counts["collected"]
        )
        obs = _cpu_observations(
            succeeded,
            counts["collected"],
            counts["passed"],
            counts["failed"],
            counts["skipped"],
        )
        if not succeeded:
            sys.stderr.write((completed.stdout + "\n" + completed.stderr)[-4000:])
        return (0 if succeeded else 1), obs


def _device_occupancy(device: str) -> str:
    try:
        processes, _ = adapter_runtime.processes_on(int(device))
    except adapter_runtime.BlockedEnv as exc:
        if exclusive_requested():
            raise BridgeError(str(exc)) from exc
        return "unknown"
    return "free" if not processes else f"shared({len(processes)})"


def run_device(values: dict[str, str], timeout: float) -> tuple[int, dict]:
    source = _source_root()
    device = require_device()
    lease_path = os.environ.get("AGENT_GATES_DEVICE_LEASE_PATH", "").strip()
    lease_resource = os.environ.get("AGENT_GATES_DEVICE_LEASE_RESOURCE", f"npu:{device}").strip()
    if not lease_path:
        raise BridgeError("AGENT_GATES_DEVICE_LEASE_PATH is required for windowed device runs")
    try:
        wait_seconds = float(os.environ.get("AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS", "60"))
    except ValueError as exc:
        raise BridgeError("invalid AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS") from exc

    cann_setup = Path(_required(values, "TILEFOUNDRY_NPU_CANN_SETUP"))
    conda_sh = Path(_required(values, "TILEFOUNDRY_NPU_CONDA_SH"))
    conda_env = _required(values, "TILEFOUNDRY_NPU_CONDA_ENV")
    python = python_path(values)
    extra_tests = device_tests(values)

    Path(lease_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lease_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    started = time.monotonic()
    deadline = started + max(0.0, wait_seconds)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BridgeError(
                        f"device lease {lease_resource} did not open within {wait_seconds}s"
                    ) from exc
                time.sleep(min(0.05, remaining))

        opened = time.monotonic()
        window_started = _timestamp()
        occ = _device_occupancy(device)
        probe_seconds = round(time.monotonic() - opened, 3)
        if exclusive_requested() and occ.startswith("shared("):
            raise BridgeError(
                f"device {lease_resource} is {occ} and TILEFOUNDRY_NPU_REQUIRE_EXCLUSIVE_DEVICE is set"
            )

        work_started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="tilefoundry-npu-") as tmp:
            probe_path = Path(tmp) / "probe.py"
            probe_path.write_text(DEVICE_PROBE, encoding="utf-8")
            run_env = dict(os.environ)
            envs_path = (values.get("TILEFOUNDRY_NPU_CONDA_ENVS_PATH") or "").strip()
            if envs_path:
                run_env["CONDA_ENVS_PATH"] = envs_path
            run_env["ASCEND_RT_VISIBLE_DEVICES"] = device
            run_env["PYTHONDONTWRITEBYTECODE"] = "1"
            run_env["TILELANG_DISABLE_CACHE"] = "1"
            run_env["PYTHONPATH"] = DEFAULT_PYTHONPATH

            cmd = f"{shlex.quote(str(python))} {shlex.quote(str(probe_path))}"
            if extra_tests:
                cmd += " && " + " ".join(
                    [shlex.quote(str(python)), "-m", "pytest", "-p", "no:cacheprovider"]
                    + [shlex.quote(t) for t in extra_tests]
                )
            try:
                completed = adapter_runtime.run_host_bash(
                    cmd,
                    cwd=source,
                    env=run_env,
                    cann_setup=cann_setup,
                    conda_sh=conda_sh,
                    conda_env=conda_env,
                    timeout=timeout,
                    isolate_temp=True,
                )
            except subprocess.TimeoutExpired as exc:
                held = round(time.monotonic() - work_started, 3)
                released = _timestamp()
                if exc.stderr:
                    sys.stderr.write(str(exc.stderr)[-4000:])
                print(f"{PROG}: device probe timed out after {timeout}s", file=sys.stderr)
                return 1, {
                    "exitCodeZero": False,
                    "tilefoundryImported": False,
                    "npuAvailable": False,
                    "deviceCountOne": False,
                    "tensorOnDevice": False,
                    "fp32VectorOk": False,
                    "bf16MatmulOk": False,
                    "numericalOk": False,
                    "synchronizeOk": False,
                    "deviceMemoryPositive": False,
                    "windowOpened": True,
                    "windowReleased": True,
                    "occupancyRecorded": occ in {"free", "unknown"} or occ.startswith("shared("),
                    "deviceWindow": {
                        "resource": lease_resource,
                        "path": lease_path,
                        "startedAt": window_started,
                        "releasedAt": released,
                        "waited": round(opened - started, 3),
                        "heldSeconds": held,
                        "probeSeconds": probe_seconds,
                        "occupancy": occ,
                    },
                }

        held = round(time.monotonic() - work_started, 3)
        released = _timestamp()
        if completed.stderr.strip():
            sys.stderr.write(completed.stderr[-4000:])

        parsed: dict = {}
        for line in reversed(completed.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    candidate = json.loads(line)
                except ValueError:
                    continue
                if isinstance(candidate, dict) and "npuAvailable" in candidate:
                    parsed = candidate
                    break

        ok = completed.returncode == 0 and bool(parsed) and all(
            parsed.get(k) is True
            for k in (
                "tilefoundryImported",
                "npuAvailable",
                "deviceCountOne",
                "tensorOnDevice",
                "fp32VectorOk",
                "bf16MatmulOk",
                "numericalOk",
                "synchronizeOk",
                "deviceMemoryPositive",
            )
        )
        if not ok:
            sys.stderr.write((completed.stdout or "")[-4000:])

        obs = {
            "exitCodeZero": ok,
            "tilefoundryImported": bool(parsed.get("tilefoundryImported", False)),
            "npuAvailable": bool(parsed.get("npuAvailable", False)),
            "deviceCountOne": bool(parsed.get("deviceCountOne", False)),
            "tensorOnDevice": bool(parsed.get("tensorOnDevice", False)),
            "fp32VectorOk": bool(parsed.get("fp32VectorOk", False)),
            "bf16MatmulOk": bool(parsed.get("bf16MatmulOk", False)),
            "numericalOk": bool(parsed.get("numericalOk", False)),
            "synchronizeOk": bool(parsed.get("synchronizeOk", False)),
            "deviceMemoryPositive": bool(parsed.get("deviceMemoryPositive", False)),
            "windowOpened": True,
            "windowReleased": True,
            "occupancyRecorded": occ in {"free", "unknown"} or occ.startswith("shared("),
            "deviceWindow": {
                "resource": lease_resource,
                "path": lease_path,
                "startedAt": window_started,
                "releasedAt": released,
                "waited": round(opened - started, 3),
                "heldSeconds": held,
                "probeSeconds": probe_seconds,
                "occupancy": occ,
            },
        }
        return (0 if ok else (completed.returncode or 1)), obs
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    timeout = None
    if "--timeout" in args:
        idx = args.index("--timeout")
        try:
            timeout = float(args[idx + 1])
        except (IndexError, ValueError):
            _die(2, "--timeout needs a number of seconds")
        del args[idx : idx + 2]

    if args == ["--check-config"]:
        return check_config(check_device=False)
    if args == ["--check-config-device"]:
        return check_config(check_device=True)

    try:
        values = machine_values()
    except BridgeError as exc:
        _die(75, str(exc))

    if args == ["--cpu"]:
        rc, obs = run_cpu(values, DEFAULT_CPU_TIMEOUT if timeout is None else timeout)
        print(json.dumps(obs, sort_keys=True))
        return rc
    if args == ["--device"]:
        try:
            rc, obs = run_device(values, DEFAULT_DEVICE_TIMEOUT if timeout is None else timeout)
        except BridgeError as exc:
            _die(75, str(exc))
        print(json.dumps(obs, sort_keys=True))
        return rc

    _die(
        2,
        f"usage: {PROG} (--cpu | --device | --check-config | --check-config-device) [--timeout SECONDS]",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
