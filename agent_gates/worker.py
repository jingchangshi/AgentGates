"""The gate worker: one process, one execution, started by systemd.

This module is the ONLY place a submitted execution actually runs a compiler.
It is a plain entry point - systemd passes a validated execution id as argv[1] -
so the unit file never interpolates a command.

Progress events are written for an operator watching a long build. They are
explicitly NOT evidence: a gate's PASS comes from the receipt the Engine
publishes, and nothing here can manufacture one.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback

from . import api_types as api
from .config import ConfigError, load
from .engine import Engine
from .evidence import EvidenceStore
from .execution_store import EXECUTION_ID, ExecutionStore
from .locking import LockBusy
from .registry import Registry


class WorkerError(RuntimeError):
    pass


def _config_for_execution(request, config):
    """Refuse to run when the machine configuration no longer matches.

    The request bound a configuration digest at submit time. If the operator
    edited the configuration while the execution was queued, running under the
    new one would execute a different machine contract than the caller was
    told. Reporting that explicitly is the C3 requirement; silently adopting
    the new configuration is not allowed.
    """
    bound = request.get("config_digest")
    if not bound:
        raise WorkerError("request carries no configuration digest")
    if bound != config.config_digest:
        raise WorkerError(
            "configuration changed after submission: bound " + str(bound)[:12]
            + " now " + config.config_digest[:12])
    return config


def run_execution(execution_id: str, *, store: ExecutionStore, config, engine=None,
                  cancelled=None) -> dict:
    """Execute one persisted request and record every transition.

    @param execution_id - a validated 32-hex-character execution id.
    @param store - the durable execution store.
    @param config - the configuration to run under; it must match the binding.
    @param engine - engine override, used by tests.
    @param cancelled - callable returning True when a cancel was requested.
    """
    request = store.read_request(execution_id)
    if request is None:
        raise WorkerError("no request record for execution " + execution_id)

    # A cancel requested before the worker ever started must not be ignored
    # and must not be followed by a build.
    if store.read_cancel(execution_id) is not None:
        store.transition(execution_id, api.ExecutionState.CANCELLED,
                         event="cancelled_before_start",
                         **{"exitReason": "cancelled before the worker started",
                            "finishedAt": time.time()})
        return store.read_status(execution_id)

    store.transition(execution_id, api.ExecutionState.STARTING, event="starting",
                     **{"startedAt": time.time(), "workerPid": os.getpid()})
    try:
        _config_for_execution(request, config)
    except ConfigError as exc:
        store.transition(execution_id, api.ExecutionState.INTERRUPTED,
                         event="config_mismatch",
                         **{"exitReason": "CONFIG_IDENTITY_CHANGED: " + str(exc),
                            "finishedAt": time.time()})
        return store.read_status(execution_id)

    workspace = request["workspace"]
    adapter = request["adapter"]
    gates = list(request["gates"])

    workspace_entry = config.workspace(workspace)
    source = workspace_entry.source_root
    adapter_entry = config.adapter(adapter)
    device_id = request.get("device_id")

    registry = Registry(os.environ.get("AGENT_GATES_ADAPTERS_ROOT") or None)
    evidence = EvidenceStore(config.evidence_root)
    engine = engine or Engine(registry=registry, store=evidence, lock_root=config.lock_root)

    store.transition(execution_id, api.ExecutionState.RUNNING, event="running")

    try:
        receipt = engine.run(adapter, source, gates,
                             env_file=adapter_entry.env_file,
                             device_id=device_id,
                             require_exclusive_device=config.devices.require_exclusive)
    except LockBusy as exc:
        # A competing AgentGates run holds the source or build lease. This is
        # a resource refusal, not an environment fault, and no gate ran.
        store.transition(execution_id, api.ExecutionState.FINISHED,
                         event="resource_busy",
                         **{"exitReason": "RESOURCE_BUSY: " + str(exc),
                            "finishedAt": time.time()})
        return store.read_status(execution_id)
    except KeyboardInterrupt:
        store.transition(execution_id, api.ExecutionState.CANCELLED,
                         event="cancelled",
                         **{"exitReason": "cancelled by signal",
                            "finishedAt": time.time()})
        return store.read_status(execution_id)
    except (OSError, ValueError, RuntimeError) as exc:
        # The Engine was reached but could not produce a receipt. Record the
        # real reason; never dress it up as a gate verdict.
        store.transition(execution_id, api.ExecutionState.INTERRUPTED,
                         event="failed",
                         **{"exitReason": "EXECUTION_FAILED: " + str(exc)[:400],
                            "finishedAt": time.time()})
        return store.read_status(execution_id)

    # A cancel that arrived while the gate ran makes this execution CANCELLED.
    # The Engine still publishes a receipt (the interrupted gate is FAIL), but
    # calling that outcome FINISHED would report an operator-initiated stop as
    # a completed run. The receipt is kept: it is real evidence about what the
    # gate did before it was stopped, and a FAIL receipt can never satisfy a
    # completion requirement.
    if store.read_cancel(execution_id) is not None:
        store.transition(execution_id, api.ExecutionState.CANCELLED,
                         event="cancelled_while_running",
                         event_payload={"run_id": receipt.get("runId")},
                         **{"runId": receipt.get("runId"),
                            "gateStatus": receipt.get("status"),
                            "exitReason": "cancelled by operator while the gate was running",
                            "currentGate": None,
                            "finishedAt": time.time()})
        return store.read_status(execution_id)

    run_id = receipt.get("runId")
    gate_status = receipt.get("status")
    gate_results = [{"gate_id": gate.get("gateId"), "status": gate.get("status"),
                     "failures": list(gate.get("failures") or []),
                     "recipe_id": gate.get("recipeId")}
                    for gate in receipt.get("gates", []) if isinstance(gate, dict)]

    # FINISHED says the worker stopped. gate_status says what the gates decided.
    # They are deliberately separate: a finished execution whose gates FAIL is
    # the normal outcome of a broken tree, not an execution error.
    # The status record uses ONE key spelling. Writing run_id here as well as
    # runId would leave two spellings for one fact, and a reader that trusts
    # the wrong one reports a finished run with no evidence.
    store.transition(execution_id, api.ExecutionState.FINISHED,
                     event="finished",
                     event_payload={"run_id": run_id, "gate_status": gate_status},
                     **{
                         "runId": run_id,
                         "gateStatus": gate_status,
                         "gateResults": gate_results,
                         "completedGates": [item["gate_id"] for item in gate_results],
                         "currentGate": None,
                         "exitReason": "completed",
                         "finishedAt": time.time(),
                     })
    return store.read_status(execution_id)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or not EXECUTION_ID.fullmatch(argv[0]):
        print("agent-gates worker: expected exactly one execution id", file=sys.stderr)
        return 2
    execution_id = argv[0]

    # A cancelled execution is stopped with SIGTERM. Converting it to an
    # exception here lets the Engine's own interrupt handling run, so the
    # supervisor tears down the child process tree instead of leaving a
    # compiler running past the end of its worker.
    def _interrupt(signum, frame):
        raise KeyboardInterrupt("worker signalled")

    signal.signal(signal.SIGTERM, _interrupt)

    try:
        config = load()
    except ConfigError as exc:
        print("agent-gates worker: configuration unusable: " + str(exc), file=sys.stderr)
        return 3

    store = ExecutionStore(config.state_root)
    store.initialize()
    store.write_worker(execution_id, {"pid": os.getpid(), "started_at": time.time(),
                                      "python": sys.executable, "cwd": os.getcwd(),
                                      "boot_id": _boot_id()})
    try:
        status = run_execution(execution_id, store=store, config=config)
    except KeyboardInterrupt:
        store.transition(execution_id, api.ExecutionState.CANCELLED,
                         event="cancelled_by_signal",
                         **{"exitReason": "cancelled by signal",
                            "finishedAt": time.time()})
        status = store.read_status(execution_id)
    except (OSError, ValueError, RuntimeError) as exc:
        traceback.print_exc()
        store.transition(execution_id, api.ExecutionState.INTERRUPTED,
                         event="worker_failed",
                         **{"exitReason": "WORKER_FAILED: " + str(exc)[:400],
                            "finishedAt": time.time()})
        return 1
    print("agent-gates worker: " + execution_id + " " + str(status.get("state")))
    return 0 if status.get("state") in {item.value for item in api.TERMINAL_STATES} else 0


def _boot_id():
    """Identify the boot, so a pid from a previous boot is never trusted."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
