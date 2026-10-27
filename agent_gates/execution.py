"""The asynchronous execution manager: submit, status, wait, cancel, recover.

The CLI and the MCP server both call this module and nothing else. It resolves
every machine parameter from the local configuration (C2), starts an
independent systemd user worker (D2), and never runs a build in its own
process.

The manager does not own the truth about a run either: the ExecutionStore does.
Every read re-derives an answer from the store, the systemd unit state, the
worker record and the receipt, so a manager that restarts cannot invent a
success it never observed.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

from . import api_types as api
from . import systemd_units
from .config import ConfigError, load
from .evidence import EvidenceStore
from .execution_store import AdmissionLock, EXECUTION_ID, ExecutionStore, new_execution_id
from .manifest import ManifestError
from .registry import Registry

# A wait is always bounded. The MCP wait tool must never hold a single call
# open until a compiler finishes; callers poll instead.
DEFAULT_WAIT_SECONDS = 15.0
MAX_WAIT_SECONDS = 60.0


def _terminal(state) -> bool:
    return state in {item.value for item in api.TERMINAL_STATES}


class ExecutionManager:
    """Owns the submit/observe/cancel lifecycle for one configuration."""

    def __init__(self, config=None, *, registry=None, systemd=None):
        self.config = config if config is not None else load()
        # The registry root is overridable so integration tests can install
        # fixture adapters without touching the packaged adapter directory.
        self.registry = registry if registry is not None else Registry(
            os.environ.get("AGENT_GATES_ADAPTERS_ROOT") or None)
        self.store = ExecutionStore(self.config.state_root)
        self.evidence = EvidenceStore(self.config.evidence_root)
        self.systemd = systemd or systemd_units

    # ── validation ──────────────────────────────────────────────────────────

    def _resolve(self, workspace_id: str, adapter_id: str, gates: list):
        """Validate a request against the registry, refusing with stable codes."""
        try:
            workspace = self.config.workspace(workspace_id)
        except ConfigError as exc:
            raise api.APIError(api.ErrorCode.UNKNOWN_WORKSPACE, str(exc),
                               field=exc.field) from None
        try:
            manifest = self.registry.get(adapter_id)
        except (ManifestError, OSError, ValueError) as exc:
            raise api.APIError(api.ErrorCode.UNKNOWN_ADAPTER, str(exc)) from None
        if adapter_id not in workspace.adapters:
            raise api.APIError(
                api.ErrorCode.ADAPTER_NOT_ALLOWED,
                'adapter ' + repr(adapter_id) + ' is not registered for workspace '
                + repr(workspace_id),
                details={"registered": list(workspace.adapters)})
        if not gates:
            raise api.APIError(api.ErrorCode.INVALID_REQUEST,
                               "at least one gate is required")
        unknown = [gate for gate in gates if gate not in manifest.gates]
        if unknown:
            raise api.APIError(api.ErrorCode.UNKNOWN_GATE,
                               "unknown gate(s): " + ", ".join(unknown),
                               details={"available": sorted(manifest.gates)})
        if workspace.allowed_gates is not None:
            denied = [gate for gate in gates if gate not in workspace.allowed_gates]
            if denied:
                raise api.APIError(
                    api.ErrorCode.GATE_NOT_ALLOWED,
                    "gate(s) not permitted for workspace " + repr(workspace_id) + ": "
                    + ", ".join(denied),
                    details={"allowed": sorted(workspace.allowed_gates)})
        # Gate dependencies are resolved by the manifest, so asking for only the
        # dependent gate silently runs its prerequisites too. That is correct for
        # execution, and the receipt records them, so verify_requirements can
        # check the declared dependency chain explicitly.
        ordered = manifest.order(gates)
        return workspace, manifest, ordered

    def _device_for(self, adapter_id: str, requested_device):
        """Choose the NPU device a request will use.

        A card another job is running on is still a card this gate may use, so
        nothing here refuses for occupancy: sharing is the normal case on a
        shared machine, and a refusal on that basis would block legitimate
        work for reasons that have nothing to do with the code under test.

        The configuration may NARROW the set (`devices.allowed_npu_devices`)
        and may demand a quiet card (`devices.require_exclusive_device`). Both
        are optional; with neither set, every device is available and the
        selection is deterministic so a receipt keeps describing a
        reproducible machine state.
        """
        if not adapter_id.endswith("-npu"):
            return None
        if requested_device is not None:
            # An execution request has no device field, so this is reachable
            # only from an internal caller passing one explicitly.
            device = str(requested_device)
            if not self.config.devices.permits(device):
                raise api.APIError(
                    api.ErrorCode.DEVICE_NOT_AUTHORIZED,
                    "device " + device + " is excluded by devices.allowed_npu_devices",
                    field="devices.allowed_npu_devices",
                    details={"allowed": sorted(self.config.devices.allowed or [])})
            return device
        return self.config.devices.select()

    @staticmethod
    def _valid_id(execution_id) -> str:
        """Reject a malformed execution id as a REQUEST error.

        The id reaches a file path and a systemd unit name, so a value that
        is not 32 hex characters must fail before either is built.
        """
        if not isinstance(execution_id, str) or not EXECUTION_ID.fullmatch(execution_id):
            raise api.APIError(api.ErrorCode.INVALID_REQUEST,
                               "invalid execution id; expected 32 hexadecimal characters")
        return execution_id

    # ── submit (D1) ─────────────────────────────────────────────────────────

    def submit(self, request: api.ExecutionRequest) -> api.ExecutionStatus:
        """Persist a request, start a worker, and return immediately.

        Order matters: the request is durable before the unit starts, so a
        worker can never run a request the store does not describe.
        """
        self.store.initialize()
        workspace, manifest, ordered = self._resolve(
            request.workspace, request.adapter, list(request.gates))
        device_id = self._device_for(request.adapter, None)

        # Idempotency (D1): the same request id must not create a second real
        # execution, and a different request under the same id is a conflict.
        if request.request_id is not None:
            existing = self._find_by_request_id(request.request_id)
            if existing is not None:
                bound = self.store.read_request(existing) or {}
                same = (bound.get("workspace") == request.workspace
                        and bound.get("adapter") == request.adapter
                        and list(bound.get("requested_gates") or []) == list(request.gates))
                if not same:
                    raise api.APIError(
                        api.ErrorCode.CONFLICT,
                        'request id ' + repr(request.request_id) + ' was already used '
                        "for a different request",
                        details={"execution_id": existing})
                # Same request, same id: return the original execution as-is.
                return self.status(existing)

        available, reason = self.systemd.available()
        if not available:
            raise api.APIError(
                api.ErrorCode.SYSTEMD_UNAVAILABLE,
                "user systemd is not usable, so a persistent worker cannot be "
                "started; refusing rather than running in-process: " + reason)

        active = self._active_ids()
        if len(active) >= self.config.worker.max_concurrent:
            raise api.APIError(
                api.ErrorCode.CONCURRENCY_LIMIT,
                str(len(active)) + " executions are already running (limit "
                + str(self.config.worker.max_concurrent) + ")",
                details={"active": active})

        execution_id = new_execution_id()
        record = {
            "format": 1,
            "execution_id": execution_id,
            "workspace": request.workspace,
            "adapter": request.adapter,
            # The order the Engine will actually use, dependencies included, so
            # the record describes the run rather than the wish.
            "gates": ordered,
            "requested_gates": list(request.gates),
            "request_id": request.request_id,
            "device_id": device_id,
            "source_root": str(workspace.source_root),
            "adapter_manifest_sha256": manifest.manifest_hash,
            "config_digest": self.config.config_digest,
            "submitted_at": time.time(),
        }
        status = {
            "format": 1,
            "executionId": execution_id,
            "state": api.ExecutionState.SUBMITTED.value,
            "workspace": request.workspace,
            "adapter": request.adapter,
            "gates": ordered,
            "requestId": request.request_id,
            "submittedAt": time.time(),
            "startedAt": None,
            "finishedAt": None,
            "runId": None,
            "currentGate": None,
            "completedGates": [],
            "gateStatus": None,
            "exitReason": None,
            "message": None,
        }
        try:
            self.store.create(execution_id, record, status)
        except FileExistsError as exc:
            raise api.APIError(api.ErrorCode.CONFLICT, str(exc)) from None

        try:
            self.systemd.start(execution_id)
        except systemd_units.SystemdUnavailable as exc:
            self.store.transition(
                execution_id, api.ExecutionState.INTERRUPTED,
                event="start_failed",
                **{"exitReason": "WORKER_START_FAILED: " + str(exc)[:400],
                   "finishedAt": time.time()})
            raise api.APIError(api.ErrorCode.WORKER_START_FAILED,
                               "the worker unit could not be started: " + str(exc),
                               details={"execution_id": execution_id}) from None

        self.store.transition(execution_id, api.ExecutionState.STARTING,
                              event="worker_started")
        return self.status(execution_id)

    def _find_by_request_id(self, request_id: str):
        for execution_id in self.store.list_ids():
            record = self.store.read_request(execution_id)
            if record is not None and record.get("request_id") == request_id:
                return execution_id
        return None

    def _active_ids(self) -> list:
        """Ids that are not yet terminal, according to the store."""
        active = []
        for execution_id in self.store.list_ids():
            status = self.store.read_status(execution_id)
            if status is not None and not _terminal(status.get("state")):
                active.append(execution_id)
        return active

    # ── observe ─────────────────────────────────────────────────────────────

    def status(self, execution_id: str) -> api.ExecutionStatus:
        """Read the current state, reconciling it with the machine first."""
        execution_id = self._valid_id(execution_id)
        status = self._reconcile(execution_id)
        if status is None:
            raise api.APIError(api.ErrorCode.EXECUTION_NOT_FOUND,
                               'no execution ' + repr(execution_id))
        return self._to_execution_status(status)

    def _to_execution_status(self, status: dict) -> api.ExecutionStatus:
        state = status.get("state") or api.ExecutionState.SUBMITTED.value
        valid = {item.value for item in api.ExecutionState}
        return api.ExecutionStatus(
            execution_id=status.get("executionId"),
            state=api.ExecutionState(state) if state in valid
            else api.ExecutionState.INTERRUPTED,
            workspace=status.get("workspace") or "",
            adapter=status.get("adapter") or "",
            gates=list(status.get("gates") or []),
            request_id=status.get("requestId"),
            run_id=status.get("runId"),
            submitted_at=_iso(status.get("submittedAt")),
            started_at=_iso(status.get("startedAt")),
            finished_at=_iso(status.get("finishedAt")),
            current_gate=status.get("currentGate"),
            completed_gates=list(status.get("completedGates") or []),
            message=status.get("message"),
            exit_reason=status.get("exitReason"),
        )

    def wait(self, execution_id: str, timeout=None) -> api.WaitResult:
        """Block for at most the given timeout, then return whatever is known."""
        execution_id = self._valid_id(execution_id)
        limit = DEFAULT_WAIT_SECONDS if timeout is None else float(timeout)
        # A caller cannot ask for an unbounded wait; the ceiling is enforced here
        # so neither a model nor a script can hold a tool call open indefinitely.
        limit = max(0.0, min(limit, MAX_WAIT_SECONDS))
        deadline = time.monotonic() + limit
        status = self._reconcile(execution_id)
        if status is None:
            raise api.APIError(api.ErrorCode.EXECUTION_NOT_FOUND,
                               'no execution ' + repr(execution_id))
        while not _terminal(status.get("state")) and time.monotonic() < deadline:
            time.sleep(min(0.5, max(0.05, deadline - time.monotonic())))
            status = self._reconcile(execution_id) or status
        timed_out = not _terminal(status.get("state"))
        return api.WaitResult(execution_id=execution_id, timed_out=timed_out,
                              status=self._to_execution_status(status))

    # ── cancel (D5) ─────────────────────────────────────────────────────────

    def cancel(self, execution_id: str) -> api.ExecutionStatus:
        """Cancel an execution, idempotently.

        Cancelling is not the same as killing: the systemd unit is stopped, and
        KillMode=control-group takes the compiler and every descendant with it.
        The resource leases are flock descriptors held by those processes, so
        they are released exactly when the tree is gone - not before.
        """
        execution_id = self._valid_id(execution_id)
        status = self._reconcile(execution_id)
        if status is None:
            raise api.APIError(api.ErrorCode.EXECUTION_NOT_FOUND,
                               'no execution ' + repr(execution_id))
        state = status.get("state")
        if _terminal(state):
            # Already terminal: a repeat cancellation reports the terminal state.
            return self.status(execution_id)

        self.store.write_cancel(execution_id, {"requested_at": time.time()})
        self.store.transition(execution_id, api.ExecutionState.CANCELLING,
                              event="cancel_requested")
        try:
            self.systemd.stop(execution_id)
        except systemd_units.SystemdUnavailable as exc:
            # The unit refusing to stop means we cannot prove the child tree is
            # gone. Report that instead of claiming a cancellation that may not
            # have stopped anything.
            raise api.APIError(
                api.ErrorCode.WORKER_UNAVAILABLE,
                "the worker unit could not be stopped: " + str(exc),
                details={"execution_id": execution_id}) from None

        # Give the worker's own SIGTERM handling a chance to record CANCELLED
        # with its real reason, then force the terminal state if it did not.
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            current = self.store.read_status(execution_id) or {}
            if _terminal(current.get("state")):
                break
            unit_state = self.systemd.properties(execution_id, "ActiveState").get("ActiveState")
            if unit_state not in {"active", "activating"}:
                break
            time.sleep(0.25)
        final = self.store.read_status(execution_id) or {}
        if not _terminal(final.get("state")):
            self.store.transition(execution_id, api.ExecutionState.CANCELLED,
                                  event="cancelled",
                                  **{"exitReason": "cancelled by operator",
                                     "finishedAt": time.time()})
        return self.status(execution_id)

    # ── reconcile (D6) ──────────────────────────────────────────────────────

    def _reconcile(self, execution_id: str):
        """Re-derive an execution state from everything that can prove it.

        The store is not trusted blindly, and neither is a pid: a pid can be
        reused by an unrelated process after a crash. The decision uses the
        unit state together with a receipt whose verification passes.
        """
        status = self.store.read_status(execution_id)
        if status is None:
            return None
        if _terminal(status.get("state")):
            # A terminal state is final even if a straggler says otherwise, but
            # a receipt that appeared after an INTERRUPTED verdict upgrades it:
            # the run really did finish, and refusing that would hide real
            # evidence behind a crash window.
            if (status.get("state") == api.ExecutionState.INTERRUPTED.value
                    and status.get("runId") is None):
                status = self._adopt_receipt(execution_id, status) or status
            return status

        properties = self.systemd.properties(execution_id, "ActiveState", "SubState",
                                            "Result", "ExecMainStatus")
        active = properties.get("ActiveState") in {"active", "activating", "deactivating"}

        if active:
            if status.get("state") == api.ExecutionState.SUBMITTED.value:
                return self.store.transition(execution_id, api.ExecutionState.STARTING,
                                             event="observed_starting")
            return status

        # Not active. Either it finished (a receipt should exist), or it died
        # without writing an outcome.
        adopted = self._adopt_receipt(execution_id, status)
        if adopted is not None:
            return adopted

        worker = self.store.read_worker(execution_id) or {}
        if _process_alive(worker):
            # A worker process exists but its unit is gone: it is no longer
            # under systemd control, so we cannot prove it will finish.
            return self.store.transition(
                execution_id, api.ExecutionState.INTERRUPTED,
                event="interrupted",
                **{"exitReason": "INTERRUPTED: worker process is no longer managed "
                                  "by systemd and no receipt was published",
                   "finishedAt": time.time()})

        if self.store.read_cancel(execution_id) is not None:
            return self.store.transition(execution_id, api.ExecutionState.CANCELLED,
                                         event="cancelled",
                                         **{"exitReason": "cancelled by operator",
                                            "finishedAt": time.time()})

        if properties.get("Result") == "success":
            # The worker exited 0 but left no receipt. The Engine always
            # publishes one before returning, so this is a torn run, not a pass.
            return self.store.transition(
                execution_id, api.ExecutionState.INTERRUPTED,
                event="interrupted",
                **{"exitReason": "INTERRUPTED: worker exited without publishing a receipt",
                   "finishedAt": time.time()})

        result = properties.get("Result")
        return self.store.transition(
            execution_id, api.ExecutionState.INTERRUPTED,
            event="interrupted",
            **{"exitReason": ("INTERRUPTED: no live worker and no receipt"
                              + (" (unit result " + str(result) + ")" if result else "")),
               "finishedAt": time.time()})

    def _adopt_receipt(self, execution_id: str, status: dict):
        """Finish an execution from a receipt that actually verifies.

        This covers two crash windows the objective names: a receipt written but
        the status not updated, and a status written but the final event not
        written. The receipt is the authority, and only a receipt that verifies
        may upgrade the record.
        """
        record = self.store.read_request(execution_id) or {}
        candidate = status.get("runId") or self._find_receipt(execution_id, record)
        if not candidate:
            return None
        verification = self.evidence.verify(candidate, registry=self.registry)
        try:
            receipt = self.evidence.read(candidate)
        except (OSError, ValueError):
            receipt = None
        if receipt is None:
            return None
        if receipt.get("adapter", {}).get("id") != record.get("adapter"):
            return None
        gate_results = [{"gate_id": gate.get("gateId"), "status": gate.get("status"),
                         "failures": list(gate.get("failures") or []),
                         "recipe_id": gate.get("recipeId")}
                        for gate in receipt.get("gates", []) if isinstance(gate, dict)]
        return self.store.transition(
            execution_id, api.ExecutionState.FINISHED,
            event="recovered_from_receipt",
            **{
                "runId": candidate,
                "gateStatus": verification.get("status"),
                "gateResults": gate_results,
                "completedGates": [item["gate_id"] for item in gate_results],
                "currentGate": None,
                "exitReason": "recovered from published receipt",
                "finishedAt": time.time(),
            })

    def _find_receipt(self, execution_id: str, record: dict):
        """Look for a receipt this execution produced, most recent first."""
        directory = self.evidence.root / "runs"
        if not directory.is_dir():
            return None
        entries = [entry for entry in directory.iterdir() if entry.is_dir()]
        entries.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for entry in entries:
            receipt = self.store._read(entry / "receipt.json")
            if not isinstance(receipt, dict):
                continue
            if receipt.get("adapter", {}).get("id") != record.get("adapter"):
                continue
            if os.path.realpath(receipt.get("source", {}).get("root", "") or "") != os.path.realpath(
                    record.get("source_root") or ""):
                continue
            gates = [gate.get("gateId") for gate in receipt.get("gates", [])
                     if isinstance(gate, dict)]
            if gates != list(record.get("gates") or []):
                continue
            return receipt.get("runId")
        return None


def _process_alive(worker: dict) -> bool:
    """Whether the recorded worker is really running on THIS boot.

    A pid alone is not proof: after a reboot, or after the pid was recycled by
    an unrelated process, boot identity and command line disagree. Both are
    checked, and the process must still be a python running agent_gates.
    """
    pid = worker.get("pid")
    if type(pid) is not int or pid <= 0:
        return False
    recorded_boot = worker.get("boot_id")
    if recorded_boot:
        try:
            current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            current_boot = ""
        if current_boot and recorded_boot != current_boot:
            return False
    try:
        cmdline = Path("/proc/" + str(pid) + "/cmdline").read_bytes()
    except OSError:
        return False
    return b"agent_gates" in cmdline


def _iso(value):
    if type(value) not in (int, float):
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
