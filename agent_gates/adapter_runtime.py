"""Shared host-local execution, environment, and Ascend NPU helpers for adapters."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

BLOCKED_ENV = 75

SAFE_BUILD_SUBDIR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_PROC_MEM_LINE = re.compile(
    r"Process id\s*:\s*(\d+)\s+Process name\s*:\s*(.*?)\s+Process memory\(MB\)\s*:\s*(\d+)"
)


class BlockedEnv(RuntimeError):
    """Raised when the host machine or NPU card cannot run a gate (exit 75)."""


def smi_command(explicit: str | None = None) -> str:
    """Resolve the npu-smi executable name or path."""
    return explicit or os.environ.get("AGENT_GATES_SMI_COMMAND") or "npu-smi"


def parse_env_file(
    path: Path,
    allowed_keys: tuple[str, ...] | frozenset[str] | set[str],
    *,
    allow_empty: bool = False,
) -> dict[str, str]:
    """Strictly parse a plain-data adapter env.sh without executing shell code."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"machine env file must be a regular non-symlink file: {candidate}")
    try:
        raw_text = candidate.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read machine env file {candidate}: {exc}") from exc

    allowed = set(allowed_keys)
    values: dict[str, str] = {}
    for lineno, raw_line in enumerate(raw_text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT.match(stripped)
        if not match:
            raise ValueError(
                f"{candidate}:{lineno}: expected KEY=VALUE assignment, got {stripped[:80]!r}"
            )
        key, raw_value = match.group(1), match.group(2).strip()
        if key not in allowed:
            raise ValueError(f"{candidate}:{lineno}: unknown configuration key {key}")
        if key in values:
            raise ValueError(f"{candidate}:{lineno}: duplicate key {key}")
        if raw_value == "":
            value = ""
        else:
            if not (raw_value.startswith("'") or raw_value.startswith('"')):
                if any(token in raw_value for token in ("$(", "`", ";", "|", "&")):
                    raise ValueError(f"{candidate}:{lineno}: shell metacharacters forbidden for {key}")
            try:
                tokens = shlex.split(raw_value, comments=True, posix=True)
            except ValueError as exc:
                raise ValueError(f"{candidate}:{lineno}: invalid quoting for {key}: {exc}") from exc
            if len(tokens) > 1:
                raise ValueError(
                    f"{candidate}:{lineno}: {key} value must be a single token (quote spaces)"
                )
            value = tokens[0] if tokens else ""
        if "\n" in value or "\r" in value or "\x00" in value:
            raise ValueError(f"{candidate}:{lineno}: {key} must be single-line")
        values[key] = value

    if not allow_empty and not values:
        raise ValueError(f"machine env file is empty: {candidate}")
    return values


def resolve_env_file(env_var: str, adapter_dir: Path) -> Path | None:
    """Resolve the explicit selector env var or fall back to sibling env.sh."""
    explicit = os.environ.get(env_var, "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise ValueError(f"{env_var} must be an absolute path: {explicit}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{env_var} does not name an existing regular file: {explicit}")
        return path.resolve()
    sibling = Path(adapter_dir) / "env.sh"
    if sibling.exists() or sibling.is_symlink():
        if sibling.is_symlink() or not sibling.is_file():
            raise ValueError(f"sibling env file must be a regular non-symlink file: {sibling}")
        return sibling.resolve()
    return None


def validate_build_subdir(value: str | None, *, default: str = "build") -> str:
    """Validate a single safe build subdirectory component."""
    if value is None:
        return default
    if not value or not SAFE_BUILD_SUBDIR.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid build subdirectory name: {value!r}")
    return value


def validate_ninja_jobs(value: str | None) -> int | None:
    """Validate an optional NINJA_JOBS value."""
    if value is None or value == "":
        return None
    if not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"NINJA_JOBS must be a positive integer, got {value!r}")
    jobs = int(value)
    if jobs < 1 or jobs > 256:
        raise ValueError(f"NINJA_JOBS out of range [1, 256]: {jobs}")
    return jobs


def _run_smi(args: list[str], *, smi_cmd: str | None = None, timeout: float = 15.0) -> str:
    cmd = smi_command(smi_cmd)
    resolved = shutil.which(cmd) if "/" not in cmd else (cmd if Path(cmd).exists() else None)
    if resolved is None:
        raise BlockedEnv(f"Ascend SMI tool not found on host: {cmd}")
    try:
        completed = subprocess.run(
            [resolved, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BlockedEnv(f"{cmd} {' '.join(args)} failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[:300]
        raise BlockedEnv(f"{cmd} {' '.join(args)} exited {completed.returncode}: {detail}")
    return completed.stdout or ""


def npu_id_of_physical(device: int, *, smi_cmd: str | None = None) -> int:
    """Map a logical/physical Ascend device index to its npu-smi NPU ID."""
    output = _run_smi(["info", "-m"], smi_cmd=smi_cmd)
    matches: list[int] = []
    for line in output.splitlines():
        fields = line.split()
        # 5-column layout (e.g. Ascend950PR): NPU ID, Slot ID, Chip ID, Chip Phy-ID, Chip Name
        if len(fields) >= 4 and all(f.isdigit() for f in fields[:4]):
            if int(fields[3]) == device:
                matches.append(int(fields[0]))
        # 4-column layout (no Slot ID): NPU ID, Chip ID, Chip Phy-ID, Chip Name
        elif len(fields) >= 3 and all(f.isdigit() for f in fields[:3]):
            if int(fields[2]) == device:
                matches.append(int(fields[0]))
    unique = sorted(set(matches))
    if not unique:
        raise BlockedEnv(f"no Ascend NPU mapped for device index {device} in `npu-smi info -m`")
    if len(unique) > 1:
        raise BlockedEnv(
            f"physical device {device} is claimed by multiple npu-smi NPU IDs {unique}"
        )
    return unique[0]


def processes_on(device: int, *, smi_cmd: str | None = None) -> tuple[list[dict[str, str]], int]:
    """Return active device compute processes (handling blank process names cleanly)."""
    npu_id = npu_id_of_physical(device, smi_cmd=smi_cmd)
    output = _run_smi(["info", "-t", "proc-mem", "-i", str(npu_id)], smi_cmd=smi_cmd)
    if not re.search(rf"^\s*NPU ID\s*:\s*{npu_id}\s*$", output, re.MULTILINE):
        raise BlockedEnv(
            f"cannot determine NPU occupancy: proc-mem output did not report NPU ID {npu_id}"
        )
    idle = bool(re.search(r"^\s*No process in device\.?\s*$", output, re.MULTILINE))
    announced = len(re.findall(r"Process id\s*:", output))
    processes: list[dict[str, str]] = []
    for line in output.splitlines():
        match = _PROC_MEM_LINE.search(line)
        if not match:
            continue
        pid = match.group(1)
        name = match.group(2).strip() or "<unknown>"
        memory_mb = match.group(3)
        processes.append({"pid": pid, "name": name, "memoryMb": memory_mb})
    if not (
        (processes and not idle and len(processes) == announced)
        or (not processes and idle and announced == 0)
    ):
        raise BlockedEnv(
            f"cannot determine NPU occupancy on NPU ID {npu_id}: "
            f"parsed {len(processes)}/{announced} process rows (idle={idle})"
        )
    return processes, npu_id


def check_device_ready(
    device: int,
    *,
    exclusive: bool = False,
    smi_cmd: str | None = None,
) -> dict:
    """Inspect device occupancy and enforce exclusivity when requested."""
    processes, npu_id = processes_on(device, smi_cmd=smi_cmd)
    if exclusive and processes:
        summary = ", ".join(f"pid={p['pid']}({p['name']},{p['memoryMb']}MB)" for p in processes[:6])
        raise BlockedEnv(f"Ascend device {device} (npu {npu_id}) is busy: {summary}")
    return {"device": device, "npuId": npu_id, "processes": processes}


@contextmanager
def device_window(
    lock_root: Path,
    device: int,
    *,
    exclusive: bool = False,
    smi_cmd: str | None = None,
    wait_seconds: float = 0.0,
):
    """Acquire an advisory flock on npu-<device>.lock and check occupancy."""
    directory = Path(lock_root)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = directory / f"npu-{device}.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise BlockedEnv(
                        f"device window lock busy for device {device}: {lock_path}"
                    ) from exc
                time.sleep(0.05)
        info = check_device_ready(device, exclusive=exclusive, smi_cmd=smi_cmd)
        yield info
    finally:
        os.close(fd)


def build_host_bash_script(
    command: str,
    *,
    cann_setup: Path | str | None = None,
    conda_sh: Path | str | None = None,
    conda_env: str | None = None,
    extra_path: list[str] | None = None,
) -> str:
    """Assemble a non-login bash command that sources CANN and activates Conda cleanly."""
    lines = ["set -e"]
    if extra_path:
        joined = os.pathsep.join(extra_path)
        lines.append(f'export PATH={shlex.quote(joined)}${{PATH:+:$PATH}}')
    if cann_setup:
        lines.append("set +u")
        lines.append(f"source {shlex.quote(str(cann_setup))}")
    if conda_env:
        if conda_sh:
            lines.append("set +u")
            lines.append(f"source {shlex.quote(str(conda_sh))}")
            lines.append(f"conda activate {shlex.quote(str(conda_env))}")
        else:
            env_path = Path(conda_env)
            if env_path.is_absolute() and (env_path / "bin").is_dir():
                lines.append(f'export PATH={shlex.quote(str(env_path / "bin"))}:"$PATH"')
            else:
                lines.append(
                    'CONDA_BASE=$(conda info --base 2>/dev/null) || { echo "conda unavailable" >&2; exit 75; }'
                )
                lines.append('source "$CONDA_BASE/etc/profile.d/conda.sh"')
                lines.append(f"conda activate {shlex.quote(str(conda_env))}")
    lines.append(command)
    return "\n".join(lines)


def run_host_bash(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    cann_setup: Path | str | None = None,
    conda_sh: Path | str | None = None,
    conda_env: str | None = None,
    timeout: float = 600.0,
    extra_path: list[str] | None = None,
    isolate_temp: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a host bash script cleanly in its own process group with temp isolation."""
    script = build_host_bash_script(
        command,
        cann_setup=cann_setup,
        conda_sh=conda_sh,
        conda_env=conda_env,
        extra_path=extra_path,
    )
    run_env = dict(os.environ if env is None else env)
    temp_ctx = tempfile.TemporaryDirectory(prefix="agent-gates-host-") if isolate_temp else None
    try:
        if temp_ctx is not None:
            temp_root = Path(temp_ctx.name)
            ascend_work = temp_root / "ascend_work"
            ascend_work.mkdir(parents=True, exist_ok=True)
            for key in ("TMPDIR", "TEMP", "TMP"):
                run_env.setdefault(key, str(temp_root))
            run_env.setdefault("ASCEND_WORK_PATH", str(ascend_work))

        proc = subprocess.Popen(
            ["/bin/bash", "-c", script],
            cwd=str(cwd),
            env=run_env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                out_tail, err_tail = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                out_tail, err_tail = proc.communicate()
            raise subprocess.TimeoutExpired(
                exc.cmd, exc.timeout, output=out_tail, stderr=err_tail
            ) from exc
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
