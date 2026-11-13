"""Versioned local machine configuration for AgentGates V2.

Every machine parameter an agent-facing API needs is resolved from here, never
from model input. The file is standard TOML; its default location is
C:Users... -> XDG style C:~/.config/agent-gates/config.toml, and an explicit
path always wins so unit and integration tests never depend on a developer's
home directory.

Validation reports the exact dotted field that failed, because a configuration
error that only says "invalid config" cannot be repaired by the operator who
wrote the file.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tomllib

from .docker_quarantine import (
    QUARANTINED_ADAPTERS,
    QUARANTINED_GATES,
    quarantine_reason,
)

CONFIG_SCHEMA_VERSION = 1

DEFAULT_CONFIG_PATH = Path.home() / ".config/agent-gates/config.toml"

# Adapter, workspace and gate identifiers already have manifest-level rules.
# They are repeated here so a configuration file cannot smuggle a path through
# an identifier that AgentGates would later join onto a directory.
ID = re.compile(r"^[a-z][a-z0-9-]{0,63}$")

EXECUTION_BACKENDS = ("systemd",)

# The interpreter a worker unit runs under. Recorded at install time so a
# template unit never has to guess which Python has agent_gates installed.
DEFAULT_PYTHON = sys.executable


class ConfigError(ValueError):
    """A configuration fault, carrying the dotted field that caused it."""

    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


def _check_unknown_keys(mapping, allowed, prefix=""):
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        field = f"{prefix}.{unknown[0]}" if prefix else str(unknown[0])
        raise ConfigError(field, "unknown field")


def _require(mapping, key, field):
    if key not in mapping:
        raise ConfigError(field, "missing required field")
    return mapping[key]


def _string(mapping, key, field, *, required=True):
    if key not in mapping:
        if required:
            raise ConfigError(field, "missing required field")
        return None
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(field, "must be a nonempty string")
    return value


def _integer(mapping, key, field, *, default=None, minimum=None, maximum=None):
    value = mapping.get(key, default)
    if value is None:
        raise ConfigError(field, "missing required field")
    # TOML booleans are Python bools, which are also ints; rejecting them keeps
    # "concurrency = true" from silently meaning 1.
    if type(value) is not int:
        raise ConfigError(field, "must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(field, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(field, f"must be <= {maximum}")
    return value


def _canonical_directory(value, field, *, must_exist=True, create=False):
    """Resolve a configured directory to a canonical absolute path.

    Symlinked components are rejected rather than followed: the whole point of
    the check is that the recorded path names the bytes AgentGates will use, and
    a symlink lets another user or a later step redirect them.
    """
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(field, "must be a nonempty path string")
    raw = Path(os.path.expanduser(value))
    if not raw.is_absolute():
        raise ConfigError(field, "must be an absolute path")
    if ".." in raw.parts:
        raise ConfigError(field, "must not contain '..' components")
    if raw.is_symlink():
        raise ConfigError(field, "must not be a symlink")
    if create:
        raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    if must_exist:
        if not raw.exists():
            raise ConfigError(field, "does not exist")
        if not raw.is_dir():
            raise ConfigError(field, "must be a directory")
        resolved = raw.resolve(strict=True)
        # A parent component may still be a symlink even when the leaf is not.
        for parent in raw.parents:
            if parent.is_symlink():
                raise ConfigError(field, f"parent component is a symlink: {parent}")
        return resolved
    return raw.resolve(strict=False)


def _canonical_file(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(field, "must be a nonempty path string")
    raw = Path(os.path.expanduser(value))
    if not raw.is_absolute():
        raise ConfigError(field, "must be an absolute path")
    if ".." in raw.parts:
        raise ConfigError(field, "must not contain '..' components")
    if raw.is_symlink():
        raise ConfigError(field, "must not be a symlink")
    if not raw.is_file():
        raise ConfigError(field, "must name an existing regular file")
    for parent in raw.parents:
        if parent.is_symlink():
            raise ConfigError(field, f"parent component is a symlink: {parent}")
    return raw.resolve(strict=True)


class Workspace:
    """One registered source tree and the machine resources it may use."""

    __slots__ = ("id", "source_root", "adapters", "allowed_gates")

    def __init__(self, identifier, source_root, adapters, allowed_gates):
        self.id = identifier
        self.source_root = source_root
        self.adapters = adapters
        self.allowed_gates = allowed_gates

    def __repr__(self):
        return f"Workspace({self.id!r}, {str(self.source_root)!r})"


class Adapter:
    """One executable adapter and the environment file it is allowed to read."""

    __slots__ = ("id", "env_file")

    def __init__(self, identifier, env_file):
        self.id = identifier
        self.env_file = env_file

    def __repr__(self):
        return f"Adapter({self.id!r})"


class Worker:
    """How the asynchronous executor is started."""

    __slots__ = ("backend", "max_concurrent", "python", "timeout_start_seconds")

    def __init__(self, backend, max_concurrent, python, timeout_start_seconds):
        self.backend = backend
        self.max_concurrent = max_concurrent
        self.python = python
        self.timeout_start_seconds = timeout_start_seconds

    def __repr__(self):
        return f"Worker({self.backend!r}, max_concurrent={self.max_concurrent})"


class Devices:
    """Which NPU devices a gate may use.

    Sharing a card is normal operation, not an error: another job running on a
    device does not make that device unavailable to this one. So the DEFAULT is
    every device the machine exposes, and a configuration can only ever NARROW
    that set.

    Exclusivity is therefore opt-in. An operator who needs a quiet card sets
    ``require_exclusive_device``; the gate then refuses a card another process
    holds. That is a policy choice, not a precondition for running at all.
    """

    __slots__ = ("allowed", "default_device", "require_exclusive", "smi_command")

    def __init__(self, allowed, default_device, require_exclusive, smi_command):
        self.allowed = allowed
        self.default_device = default_device
        self.require_exclusive = require_exclusive
        self.smi_command = smi_command

    def permits(self, device_id):
        """Whether this configuration allows one device id."""
        return self.allowed is None or str(device_id) in self.allowed

    def select(self):
        """The device a request uses when nothing narrower selects one.

        Deterministic on purpose: a gate that silently picked "whatever looks
        idle" would make two identical requests run on different hardware, and
        the receipt would stop describing a reproducible machine state.
        """
        if self.default_device is not None:
            return self.default_device
        if self.allowed:
            return sorted(self.allowed, key=int)[0]
        return "0"

    def __repr__(self):
        return (f"Devices(allowed={self.allowed if self.allowed is None else sorted(self.allowed)}, "
                f"default={self.default_device!r}, exclusive={self.require_exclusive})")


class Config:
    """A validated, immutable view of one configuration file."""

    def __init__(self, *, path, schema_version, state_root, evidence_root,
                 workspaces, adapters, worker, devices, lock_root, config_digest):
        self.path = path
        self.schema_version = schema_version
        self.state_root = state_root
        self.evidence_root = evidence_root
        self.workspaces = workspaces
        self.adapters = adapters
        self.worker = worker
        self.devices = devices
        self.lock_root = lock_root
        self.config_digest = config_digest

    def workspace(self, workspace_id):
        try:
            return self.workspaces[workspace_id]
        except KeyError:
            raise ConfigError(f"workspaces.{workspace_id}", "unknown workspace") from None

    def adapter(self, adapter_id):
        try:
            return self.adapters[adapter_id]
        except KeyError:
            raise ConfigError(f"adapters.{adapter_id}", "unknown adapter") from None

    def __repr__(self):
        return (f"Config(path={str(self.path)!r}, digest={self.config_digest[:12]}, "
                f"workspaces={sorted(self.workspaces)}, adapters={sorted(self.adapters)})")


def _digest(payload):
    """Stable digest of the effective configuration.

    Only values that change what a gate would execute are bound: the file's own
    location and digest identity describe *where* the configuration lives, not
    *what* it asks the machine to do. Two configs that agree on every operational
    field accept each other's executions even if one is at a different path.
    """
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()


def load(path=None):
    """Load and validate one configuration file.

    @param path - explicit override; falls back to the AGENT_GATES_CONFIG
        environment variable, then to the per-user default.
    """
    if path is None:
        path = os.environ.get("AGENT_GATES_CONFIG") or DEFAULT_CONFIG_PATH
    path = Path(os.path.expanduser(str(path)))
    if path.is_symlink():
        raise ConfigError("config", "configuration file must not be a symlink")
    if not path.is_file():
        raise ConfigError("config", f"configuration file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError("config", f"not valid TOML: {exc}") from exc
    return from_mapping(raw, path=path)


def from_mapping(raw, *, path=None):
    """Validate an already-parsed configuration mapping.

    @param raw - parsed TOML document.
    @param path - location recorded on the result, used only for reporting.
    """
    if not isinstance(raw, dict):
        raise ConfigError("config", "top level must be a table")
    _check_unknown_keys(
        raw,
        ("schema_version", "state_root", "evidence_root", "lock_root",
         "workspaces", "adapters", "worker", "devices"),
    )
    schema_version = _integer(raw, "schema_version", "schema_version", minimum=1)
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigError("schema_version",
                          f"unsupported schema_version {schema_version} (expected {CONFIG_SCHEMA_VERSION})")

    state_root = _canonical_directory(_require(raw, "state_root", "state_root"), "state_root", create=True)
    evidence_root = _canonical_directory(_require(raw, "evidence_root", "evidence_root"), "evidence_root", create=True)
    lock_root = _canonical_directory(raw.get("lock_root", str(state_root / "locks")), "lock_root", create=True)

    workspaces_raw = _require(raw, "workspaces", "workspaces")
    if not isinstance(workspaces_raw, dict) or not workspaces_raw:
        raise ConfigError("workspaces", "must be a nonempty table")
    workspaces = {}
    for workspace_id, body in workspaces_raw.items():
        field = f"workspaces.{workspace_id}"
        if not ID.fullmatch(workspace_id):
            raise ConfigError(field, "workspace id must match [a-z][a-z0-9-]{0,63}")
        if not isinstance(body, dict):
            raise ConfigError(field, "must be a table")
        _check_unknown_keys(body, ("source_root", "adapters", "allowed_gates"), field)
        source_root = _canonical_directory(_require(body, "source_root", f"{field}.source_root"),
                                           f"{field}.source_root")
        adapters_raw = _require(body, "adapters", f"{field}.adapters")
        if not isinstance(adapters_raw, list) or not adapters_raw:
            raise ConfigError(f"{field}.adapters", "must be a nonempty list of adapter ids")
        adapters = []
        for index, adapter_id in enumerate(adapters_raw):
            entry = f"{field}.adapters[{index}]"
            if not isinstance(adapter_id, str) or not ID.fullmatch(adapter_id):
                raise ConfigError(entry, "must be an adapter id matching [a-z][a-z0-9-]{0,63}")
            if adapter_id in QUARANTINED_ADAPTERS:
                raise ConfigError(entry, quarantine_reason(f"adapter {adapter_id!r}"))
            if adapter_id in adapters:
                raise ConfigError(entry, "duplicate adapter id")
            adapters.append(adapter_id)
        allowed_raw = body.get("allowed_gates")
        allowed_gates = None
        if allowed_raw is not None:
            if not isinstance(allowed_raw, list) or not allowed_raw:
                raise ConfigError(f"{field}.allowed_gates", "must be a nonempty list of gate ids when present")
            allowed_gates = set()
            for index, gate_id in enumerate(allowed_raw):
                entry = f"{field}.allowed_gates[{index}]"
                if not isinstance(gate_id, str) or not gate_id.strip():
                    raise ConfigError(entry, "must be a gate id")
                if any(gate_id in QUARANTINED_GATES.get(aid, ()) for aid in adapters):
                    raise ConfigError(entry, quarantine_reason(f"gate {gate_id!r}"))
                if gate_id in allowed_gates:
                    raise ConfigError(entry, "duplicate gate id")
                allowed_gates.add(gate_id)
        workspaces[workspace_id] = Workspace(workspace_id, source_root, tuple(adapters), allowed_gates)

    adapters_raw = _require(raw, "adapters", "adapters")
    if not isinstance(adapters_raw, dict) or not adapters_raw:
        raise ConfigError("adapters", "must be a nonempty table")
    adapters = {}
    for adapter_id, body in adapters_raw.items():
        field = f"adapters.{adapter_id}"
        if not ID.fullmatch(adapter_id):
            raise ConfigError(field, "adapter id must match [a-z][a-z0-9-]{0,63}")
        if adapter_id in QUARANTINED_ADAPTERS:
            raise ConfigError(field, quarantine_reason(f"adapter {adapter_id!r}"))
        if not isinstance(body, dict):
            raise ConfigError(field, "must be a table")
        _check_unknown_keys(body, ("env_file",), field)
        env_file = None
        if body.get("env_file") is not None:
            env_file = _canonical_file(body["env_file"], f"{field}.env_file")
        adapters[adapter_id] = Adapter(adapter_id, env_file)

    # A workspace may name only adapters that have a machine entry. This is the
    # "unauthorized resource combination" check: a workspace listing an adapter
    # nobody configured would otherwise fail later, inside the worker.
    for workspace in workspaces.values():
        for adapter_id in workspace.adapters:
            if adapter_id not in adapters:
                raise ConfigError(f"workspaces.{workspace.id}.adapters",
                                  f"names adapter {adapter_id!r} with no [adapters.{adapter_id}] entry")

    worker_raw = _require(raw, "worker", "worker")
    if not isinstance(worker_raw, dict):
        raise ConfigError("worker", "must be a table")
    _check_unknown_keys(worker_raw, ("backend", "max_concurrent", "python", "start_timeout_seconds"), "worker")
    backend = _require(worker_raw, "backend", "worker.backend")
    if backend not in EXECUTION_BACKENDS:
        raise ConfigError("worker.backend", f"must be one of {', '.join(EXECUTION_BACKENDS)}")
    max_concurrent = _integer(worker_raw, "max_concurrent", "worker.max_concurrent",
                              default=2, minimum=1, maximum=64)
    python = _string(worker_raw, "python", "worker.python", required=False)
    if python is None:
        python = DEFAULT_PYTHON
    python_path = Path(python)
    if not python_path.is_absolute():
        raise ConfigError("worker.python", "must be an absolute path")
    if not python_path.is_file():
        raise ConfigError("worker.python", "does not exist")
    timeout_start = _integer(worker_raw, "start_timeout_seconds", "worker.start_timeout_seconds",
                             default=60, minimum=1, maximum=3600)
    worker = Worker(backend, max_concurrent, python_path.resolve(strict=True), timeout_start)

    devices_raw = raw.get("devices", {})
    if not isinstance(devices_raw, dict):
        raise ConfigError("devices", "must be a table")
    _check_unknown_keys(
        devices_raw,
        ("allowed_npu_devices", "authorized_npu_devices", "default_npu_device",
         "require_exclusive_device", "smi_command"),
        "devices",
    )
    # "allowed" is an optional RESTRICTION. Absent, empty, or explicitly null
    # all mean the same thing: every device the machine exposes is usable.
    # `authorized_npu_devices` is accepted as a legacy spelling of the same
    # key so an existing file keeps loading; its empty value now means "no
    # restriction" rather than "no device", which is the policy this machine
    # settled on: a card another job uses is still a card this gate may use.
    allowed_raw = devices_raw.get("allowed_npu_devices",
                                  devices_raw.get("authorized_npu_devices"))
    allowed = None
    if allowed_raw is not None:
        if not isinstance(allowed_raw, list):
            raise ConfigError("devices.allowed_npu_devices", "must be a list of device ids")
        collected = set()
        for index, device in enumerate(allowed_raw):
            entry = f"devices.allowed_npu_devices[{index}]"
            # Integers keep "device 0" from being written as the string "0" in
            # one place and the number in another; the engine normalises to "0".
            if type(device) is not int or device < 0 or device >= 64:
                raise ConfigError(entry, "must be an integer device id between 0 and 63")
            collected.add(str(device))
        allowed = frozenset(collected) if collected else None

    default_raw = devices_raw.get("default_npu_device")
    default_device = None
    if default_raw is not None:
        if type(default_raw) is not int or default_raw < 0 or default_raw >= 64:
            raise ConfigError("devices.default_npu_device",
                              "must be an integer device id between 0 and 63")
        default_device = str(default_raw)
        if allowed is not None and default_device not in allowed:
            raise ConfigError("devices.default_npu_device",
                              "must also appear in devices.allowed_npu_devices")

    exclusive_raw = devices_raw.get("require_exclusive_device", False)
    if type(exclusive_raw) is not bool:
        raise ConfigError("devices.require_exclusive_device", "must be a boolean")

    smi = devices_raw.get("smi_command", "npu-smi")
    if not isinstance(smi, str) or not smi.strip():
        raise ConfigError("devices.smi_command", "must be a nonempty string")
    devices = Devices(allowed, default_device, exclusive_raw, smi)

    payload = {
        "schema_version": schema_version,
        "workspaces": {key: {"source_root": str(ws.source_root), "adapters": list(ws.adapters),
                             "allowed_gates": sorted(ws.allowed_gates) if ws.allowed_gates else None}
                       for key, ws in workspaces.items()},
        "adapters": {key: {"env_file": str(ad.env_file) if ad.env_file else None}
                     for key, ad in adapters.items()},
        "worker": {"backend": worker.backend, "max_concurrent": worker.max_concurrent,
                   "python": str(worker.python), "start_timeout_seconds": worker.timeout_start_seconds},
        "devices": {"allowed": sorted(devices.allowed) if devices.allowed else None,
                    "default": devices.default_device,
                    "require_exclusive": devices.require_exclusive,
                    "smi_command": devices.smi_command},
    }
    # state_root/evidence_root/lock_root are deliberately OUTSIDE the digest.
    # They select where evidence is stored, not what a gate executes, so moving
    # the evidence directory must not make every in-flight execution invalid.
    return Config(path=path, schema_version=schema_version, state_root=state_root,
                  evidence_root=evidence_root, workspaces=workspaces, adapters=adapters,
                  worker=worker, devices=devices, lock_root=lock_root, config_digest=_digest(payload))
