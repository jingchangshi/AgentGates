"""Durable execution records for asynchronous gate runs.

The store is the only authority on what happened to an execution. It survives
every process that talks to it: the MCP server, the CLI, and the worker itself
may all restart, and a reader that comes back later must be able to tell a
finished run from an interrupted one without trusting anything in memory.

Two invariants drive the implementation:

1. A state file is replaced atomically (write, fsync, rename, fsync the
   directory), so a crash leaves the old value or the new one, never a torn one.
2. A terminal state is final. A late event from a worker that was declared
   interrupted must not be able to rewrite FINISHED over INTERRUPTED or the
   other way round, because that is exactly how a crashed run would come back
   looking successful.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import time

from . import api_types as api

EXECUTION_ID = re.compile(r"^[0-9a-f]{32}$")

FORMAT_VERSION = 1


def new_execution_id() -> str:
    """A fresh execution id: 32 hex characters, so it is safe as a unit name."""
    return secrets.token_hex(16)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, value) -> None:
    """Replace one JSON file atomically, durably.

    The temporary file lives in the same directory so the rename cannot cross
    a filesystem boundary, and it is fsynced before the rename so a power loss
    cannot leave the new name pointing at empty bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + "." + secrets.token_hex(4) + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + chr(10)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


class ExecutionStore:
    """The state_root/executions/<executionId>/ record set."""

    def __init__(self, state_root):
        self.root = Path(state_root)
        self.executions = self.root / "executions"

    def initialize(self) -> None:
        self.executions.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.executions.is_symlink():
            raise ValueError("execution store must not be a symlink")

    def directory(self, execution_id: str) -> Path:
        """Resolve one execution directory, refusing anything but a real id."""
        if not isinstance(execution_id, str) or not EXECUTION_ID.fullmatch(execution_id):
            raise ValueError("invalid execution id")
        path = self.executions / execution_id
        if path.is_symlink():
            raise ValueError("execution directory must not be a symlink")
        return path

    # ── paths ───────────────────────────────────────────────────────────────

    def request_path(self, execution_id):
        return self.directory(execution_id) / "request.json"

    def status_path(self, execution_id):
        return self.directory(execution_id) / "status.json"

    def events_path(self, execution_id):
        return self.directory(execution_id) / "events.jsonl"

    def worker_path(self, execution_id):
        return self.directory(execution_id) / "worker.json"

    def cancel_path(self, execution_id):
        return self.directory(execution_id) / "cancel.json"

    # ── creation ────────────────────────────────────────────────────────────

    def create(self, execution_id: str, request: dict, status: dict) -> Path:
        """Create the record set for a new execution.

        Uses mkdir without exist_ok on purpose: two processes racing to submit
        the same execution id must not both believe they created it.
        """
        self.initialize()
        directory = self.directory(execution_id)
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            raise FileExistsError("execution " + execution_id + " already exists") from None
        _fsync_dir(self.executions)
        atomic_write_json(directory / "request.json", request)
        atomic_write_json(directory / "status.json", status)
        self.append_event(execution_id, "submitted", {"state": status.get("state")})
        return directory

    # ── reads ───────────────────────────────────────────────────────────────

    def exists(self, execution_id: str) -> bool:
        try:
            return self.status_path(execution_id).is_file()
        except ValueError:
            return False

    def read_request(self, execution_id: str):
        return self._read(self.request_path(execution_id))

    def read_status(self, execution_id: str):
        return self._read(self.status_path(execution_id))

    def read_worker(self, execution_id: str):
        return self._read(self.worker_path(execution_id))

    def read_cancel(self, execution_id: str):
        return self._read(self.cancel_path(execution_id))

    @staticmethod
    def _read(path: Path):
        try:
            if path.is_symlink() or not path.is_file():
                return None
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def events(self, execution_id: str, limit: int = 500) -> list:
        """The last few events; progress is diagnostic, never evidence."""
        path = self.events_path(execution_id)
        try:
            if not path.is_file() or path.is_symlink():
                return []
            lines = path.read_text().splitlines()
        except OSError:
            return []
        result = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                result.append(item)
        return result

    def list_ids(self) -> list:
        self.initialize()
        return sorted(entry.name for entry in self.executions.iterdir()
                      if entry.is_dir() and EXECUTION_ID.fullmatch(entry.name))

    # ── writes ──────────────────────────────────────────────────────────────

    def append_event(self, execution_id: str, kind: str, payload=None) -> None:
        """Append one event line, fsynced, so a crash cannot lose a transition."""
        path = self.events_path(execution_id)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        line = json.dumps({"at": time.time(), "kind": kind, "payload": payload or {}},
                          sort_keys=True, allow_nan=False)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, (line + chr(10)).encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def write_status(self, execution_id: str, status: dict) -> None:
        atomic_write_json(self.status_path(execution_id), status)

    def write_worker(self, execution_id: str, worker: dict) -> None:
        atomic_write_json(self.worker_path(execution_id), worker)

    def write_cancel(self, execution_id: str, cancel: dict) -> None:
        atomic_write_json(self.cancel_path(execution_id), cancel)

    # ── transitions ─────────────────────────────────────────────────────────

    def transition(self, execution_id: str, state, *, event=None, event_payload=None, **fields) -> dict:
        """Move an execution to a new state, refusing to leave a terminal one.

        Returns the written status. A terminal state is returned unchanged
        rather than overwritten: the first terminal outcome is the truth, and
        everything after it is a late straggler.
        """
        current = self.read_status(execution_id) or {}
        current_state = current.get("state")
        terminal = {item.value for item in api.TERMINAL_STATES}
        if current_state in terminal:
            return current
        updated = {**current, **fields}
        updated["state"] = state.value if isinstance(state, api.ExecutionState) else str(state)
        updated["executionId"] = execution_id
        updated["updatedAt"] = time.time()
        self.write_status(execution_id, updated)
        if event is not None:
            self.append_event(execution_id, event, event_payload or {})
        return updated


class AdmissionLock:
    """A non-blocking cross-process lease over one execution id.

    It guarantees that a single execution is started by exactly one submitter
    even when two MCP calls arrive at the same instant.
    """

    def __init__(self, state_root, execution_id: str):
        directory = Path(state_root) / "locks"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / ("submit-" + execution_id + ".lock")
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            raise BlockingIOError(errno.EAGAIN, "execution is being submitted") from exc
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
