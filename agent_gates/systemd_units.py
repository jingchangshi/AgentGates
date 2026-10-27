"""User-level systemd units for the asynchronous execution worker.

One template unit, agent-gates-worker@.service, with one instance per execution
id. The unit never interpolates anything but a validated execution id, and it
runs a fixed interpreter with a module entry point: nothing a model can say
reaches ExecStart.

Why a user unit rather than a thread or a child process: objective A1 requires
that a DSH Web or MCP server restart does not end a submitted gate. A child of
the MCP server dies with it. A user-normal systemd instance outlives every
process that talks to AgentGates and keeps the outcome in journald.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

EXECUTION_ID = re.compile(r"^[0-9a-f]{32}$")

UNIT_NAME = "agent-gates-worker@.service"

# %i is the execution id systemd expands from the instance name. It is the ONLY
# substitution in this file, and callers validate the id before a unit is named.
UNIT_TEMPLATE = """[Unit]\nDescription=AgentGates gate execution worker (%i)\nDocumentation=https://github.com/jingchangshi/AgentGates\nAfter=network.target\n\n[Service]\n# Type=simple (not oneshot): with oneshot, `systemctl start` blocks until the\n# gate finishes, which would make submit() wait for the build. The unit is\n# still a one-shot job in effect: Restart=no and it exits when the worker\n# exits.\nType=simple\n# Fixed interpreter and fixed module entry point. Never built from input.
# -s drops the user site directory so a stray ~/.local copy cannot shadow the
# installed package. -E is deliberately NOT used: it would also ignore the
# PYTHONPATH line below, which is what points the interpreter at the install.
# The caller's own environment still cannot leak in: systemd starts this unit
# from the manager, not from the process that submitted the request.\nExecStart=__PYTHON__ -s -m agent_gates.worker %i\n# A gate owns its own recipe timeout; RuntimeMaxSec is the backstop that stops a\n# wedged worker even when its own supervisor never returns.\nRuntimeMaxSec=__TIMEOUT__\n# KillMode=control-group is what makes cancel() stop the compiler, the docker\n# client and every descendant rather than only the worker's own pid. The\n# resource leases are flock descriptors held by those processes, so they are\n# released exactly when the process tree is gone - not one moment earlier.\nKillMode=control-group\nKillSignal=SIGTERM\n# Grace period before SIGKILL: enough for a docker client to forward, short\n# enough that a cancelled execution does not linger.\nTimeoutStopSec=30s\n# Never restart: a gate is a one-shot job, and a restart would re-run a test\n# after an operator cancelled it.\nRestart=no\n# Keep stdout/stderr in journald, which outlives the worker.\nStandardOutput=journal\nStandardError=journal\nSyslogIdentifier=agent-gates-worker\n# Run from a neutral directory so a recipe's relative paths cannot resolve\n# against the DSH server's cwd.\nWorkingDirectory=__WORKDIR__\n# -s drops the user site directory so a stray ~/.local copy cannot shadow the
# installed package. -E/-I are deliberately NOT used: they also ignore the
# PYTHONPATH line below, which is what points the interpreter at the install.
# The caller's environment still cannot leak in: systemd starts this unit from
# the manager, not from the process that submitted the request.
Environment=PYTHONPATH=__PACKAGE_ROOT__
# The worker must read the SAME configuration the submitter resolved. The
# path is baked in at install time; a caller cannot override it through the
# environment because systemd starts this unit from the manager.
Environment=AGENT_GATES_CONFIG=__CONFIG_PATH__
Environment=AGENT_GATES_ADAPTERS_ROOT=__ADAPTERS_ROOT__
"""


class SystemdUnavailable(RuntimeError):
    """The user systemd manager cannot be used on this machine."""


def unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def _user_env():
    env = dict(os.environ)
    # A systemd --user manager is reached through the per-user runtime dir.
    # Adding it when absent lets a test process (or a supervisor started
    # without a login session) still address the same manager.
    if not env.get("XDG_RUNTIME_DIR"):
        env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    return env


def run(*args, check=True, timeout=30):
    """Invoke systemctl --user with a fixed argv and no shell."""
    try:
        completed = subprocess.run(["systemctl", "--user", *args], capture_output=True,
                                   text=True, timeout=timeout, env=_user_env(), check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemdUnavailable(f"systemctl --user unavailable: {exc}") from exc
    if check and completed.returncode != 0:
        raise SystemdUnavailable(
            f"systemctl --user {' '.join(args)} failed ({completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()[:400]}")
    return completed


def available() -> tuple[bool, str]:
    """Probe the user manager without raising.

    @returns (usable, reason). A caller that gets False must report BLOCKED and
        must not fall back to in-process execution.
    """
    if shutil.which("systemctl") is None:
        return False, "systemctl is not installed"
    runtime = Path(f"/run/user/{os.getuid()}")
    if not runtime.is_dir():
        return False, f"no user runtime directory {runtime}"
    result = run("is-system-running", check=False, timeout=15)
    state = (result.stdout or "").strip()
    # `running` and `degraded` both mean the bus answers; a degraded manager can
    # still start units. `offline`/`unknown` mean no manager is reachable.
    if state in {"running", "degraded", "starting"}:
        return True, state
    return False, f"user manager state {state or 'unknown'}: {(result.stderr or '').strip()[:200]}"


def package_root() -> str:
    """The directory to put on PYTHONPATH so agent_gates is importable.

    Resolved at install time from the interpreter running the installer, which
    removes the guesswork about whether a given interpreter already has the
    package on its own path (an editable install, a checkout, or a venv whose
    base interpreter lives elsewhere).
    """
    import agent_gates
    return str(Path(agent_gates.__file__).resolve().parent.parent)


def install(python: str, *, timeout_seconds: int, workdir: str = "/", root: Path | None = None,
            package: str | None = None, config_path: str | None = None,
            adapters_root: str | None = None) -> Path:
    """Write the unit template and reload the user manager.

    @param python - absolute interpreter that has agent_gates importable.
    @param timeout_seconds - backstop timeout for one worker.
    @param workdir - neutral working directory for the worker.
    @param root - unit directory override, used by tests.
    @param package - PYTHONPATH root; defaults to the installer's package root.
    @param config_path - the configuration the worker must read; defaults to the
        documented per-user location.
    @param adapters_root - adapter directory override, for fixture installs.
    """
    directory = Path(root) if root is not None else unit_dir()
    directory.mkdir(parents=True, exist_ok=True)
    resolved_config = config_path or os.environ.get("AGENT_GATES_CONFIG") or str(
        Path.home() / ".config/agent-gates/config.toml")
    resolved_adapters = adapters_root or os.environ.get("AGENT_GATES_ADAPTERS_ROOT") or ""
    rendered = (UNIT_TEMPLATE
                .replace("__PYTHON__", str(python))
                .replace("__TIMEOUT__", str(int(timeout_seconds)))
                .replace("__WORKDIR__", workdir)
                .replace("__PACKAGE_ROOT__", package or package_root())
                .replace("__CONFIG_PATH__", str(resolved_config))
                .replace("__ADAPTERS_ROOT__", str(resolved_adapters)))
    path = directory / UNIT_NAME
    path.write_text(rendered)
    os.chmod(path, 0o644)
    run("daemon-reload", timeout=60)
    return path


def unit_for(execution_id: str) -> str:
    """Name the instance unit for one execution, refusing anything but an id."""
    if not EXECUTION_ID.fullmatch(execution_id):
        # An execution id becomes a unit name, i.e. an argument to systemctl.
        # Refusing here is what stops an id from becoming a flag or a path.
        raise ValueError("invalid execution id")
    return UNIT_NAME.replace("@.service", f"@{execution_id}.service")


def start(execution_id: str, *, timeout: int = 30):
    return run("start", unit_for(execution_id), timeout=timeout)


def stop(execution_id: str, *, timeout: int = 60):
    return run("stop", unit_for(execution_id), timeout=timeout)


def kill(execution_id: str, *, timeout: int = 60):
    """Send SIGKILL to the whole control group."""
    return run("kill", "--signal=KILL", unit_for(execution_id), check=False, timeout=timeout)


def properties(execution_id: str, *names: str) -> dict[str, str]:
    """Read unit properties, or an empty mapping when the unit is unknown.

    Unit properties are what distinguishes 'still running' from 'finished
    successfully' from 'never started', without trusting a stale pid file.
    """
    try:
        unit = unit_for(execution_id)
    except ValueError:
        return {}
    result = run("show", unit, f"--property={','.join(names)}", check=False, timeout=30)
    if result.returncode != 0:
        return {}
    values = {}
    for line in (result.stdout or "").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def reset(execution_id: str) -> None:
    """Clear a failed state so a new attempt for the same id can start.

    Idempotent executions never start one id twice, so this exists for the
    recovery path (a worker that failed to start) rather than for retries.
    """
    run("reset-failed", unit_for(execution_id), check=False, timeout=30)


def journal_tail(execution_id: str, *, lines: int = 50) -> str:
    """Bounded journal excerpt for one execution, for diagnostics only."""
    unit = unit_for(execution_id)
    try:
        completed = subprocess.run(
            ["journalctl", "--user", "-u", unit, "-n", str(int(lines)), "--no-pager",
             "--output=short-iso"],
            capture_output=True, text=True, timeout=30, env=_user_env(), check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "")[-8000:]
