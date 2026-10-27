"""Requirement verification: does one receipt satisfy a declared requirement NOW?

This is the module a Team Task completion policy calls before allowing a
completed state. It is deliberately separate from `Engine`: it never runs a
gate, it only inspects a receipt that a gate already produced.

The eight checks are the ones the objective names (E3). Two of them exist
because the obvious shortcuts are unsound:

- No `latest` fallback. A caller must name a run id, because "the newest receipt"
  is a different question from "the receipt this task produced".
- A doctor PASS cannot satisfy a requirement. Doctor proves the machine is
  configured; it runs no compiler test and produces no gate verdict.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from . import api_types as api
from .evidence import EvidenceStore
from .manifest import ManifestError
from .registry import Registry
from .source_identity import source_root, snapshot

PASS = api.GateStatus.PASS


def _now():
    return datetime.now(timezone.utc).isoformat()


def _result(*, workspace, adapter, required_gates, run_id, failures, status,
            matched=None, identity=None, satisfied=False):
    return api.RequirementVerificationResult(
        satisfied=satisfied,
        status=status,
        failures=list(failures),
        run_id=run_id,
        verified_at=_now(),
        current_source_identity=identity,
        matched_gates=list(matched or []),
        workspace=workspace,
        adapter=adapter,
        required_gates=list(required_gates),
    )


def verify_requirements(requirement: api.VerificationRequirement, *, config,
                        registry: Registry | None = None, store: EvidenceStore | None = None):
    """Check one explicit requirement against the current machine state.

    @param requirement - workspace, adapter, required gates and an explicit run id.
    @param config - the validated local configuration; supplies the workspace's
        registered source root and the evidence root.
    @param registry - adapter registry, defaulted to the installed adapters.
    @param store - evidence store, defaulted to the configured evidence root.
    @returns a result whose `satisfied` is true only when every check passed.
    """
    workspace_id = requirement.workspace
    adapter_id = requirement.adapter
    required = list(requirement.required_gates)
    run_id = requirement.run_id

    def refuse(failures, status=api.GateStatus.INVALID_EVIDENCE, matched=None, identity=None):
        return _result(workspace=workspace_id, adapter=adapter_id, required_gates=required,
                       run_id=run_id, failures=failures, status=status,
                       matched=matched, identity=identity, satisfied=False)

    # ── request shape ────────────────────────────────────────────────────────
    if not required:
        return refuse(["REQUIRED_GATES_EMPTY"], api.GateStatus.NOT_RUN)
    if not run_id or not isinstance(run_id, str):
        # The explicit run id is the whole point; "latest" is not accepted.
        return refuse(["RUN_ID_REQUIRED"], api.GateStatus.NOT_RUN)

    registry = registry or Registry()
    store = store or EvidenceStore(config.evidence_root)

    # ── (3) workspace binding ────────────────────────────────────────────────
    try:
        workspace = config.workspace(workspace_id)
    except Exception as exc:  # ConfigError
        return refuse([f"UNKNOWN_WORKSPACE: {exc}"], api.GateStatus.INVALID_EVIDENCE)
    try:
        current_root = source_root(workspace.source_root)
    except (OSError, ValueError) as exc:
        return refuse([f"WORKSPACE_SOURCE_UNREADABLE: {exc}"], api.GateStatus.BLOCKED_ENV)
    if adapter_id not in workspace.adapters:
        # (4) the adapter must be one this workspace is allowed to use.
        return refuse([f"ADAPTER_NOT_REGISTERED_FOR_WORKSPACE: {adapter_id}"])
    if workspace.allowed_gates is not None:
        disallowed = [gate for gate in required if gate not in workspace.allowed_gates]
        if disallowed:
            return refuse([f"GATE_NOT_ALLOWED: {gate}" for gate in disallowed])

    # ── (1) run id resolves to a real receipt ────────────────────────────────
    try:
        receipt = store.read(run_id)
    except (OSError, ValueError) as exc:
        return refuse([f"RECEIPT_NOT_FOUND: {exc}"], api.GateStatus.NOT_RUN)

    # ── (2) the receipt is complete and valid ────────────────────────────────
    verification = store.verify(run_id, registry=registry)
    if verification.get("status") != PASS.value:
        status = verification.get("status")
        return refuse([f"RECEIPT_INVALID: {verification.get('status')}"]
                      + [f"RECEIPT_FAILURE: {item}" for item in verification.get("failures", [])],
                      api.GateStatus(status) if status in api.GateStatus._value2member_map_
                      else api.GateStatus.INVALID_EVIDENCE)

    # ── (4) adapter matches the requirement ─────────────────────────────────
    recorded_adapter = receipt.get("adapter", {}).get("id")
    if recorded_adapter != adapter_id:
        return refuse([f"ADAPTER_MISMATCH: receipt={recorded_adapter} required={adapter_id}"])

    # ── (3) receipt-bound source root matches the CURRENT configuration ─────
    recorded_root = receipt.get("source", {}).get("root")
    if not isinstance(recorded_root, str):
        return refuse(["RECEIPT_SOURCE_ROOT_MISSING"])
    if os.path.realpath(recorded_root) != os.path.realpath(str(current_root)):
        return refuse([f"WORKSPACE_SOURCE_MISMATCH: receipt={recorded_root} config={current_root}"])

    # ── (7) current source identity equals the receipt's ────────────────────
    try:
        identity = snapshot(Path(recorded_root))
    except (OSError, ValueError) as exc:
        return refuse([f"SOURCE_IDENTITY_UNREADABLE: {exc}"], api.GateStatus.BLOCKED_ENV)
    if identity != receipt.get("source"):
        return refuse(["SOURCE_IDENTITY_MISMATCH"], api.GateStatus.STALE, identity=identity)

    # ── (5) and (6) every required gate ran, passed, with its dependencies ──
    by_id = {gate.get("gateId"): gate for gate in receipt.get("gates", [])
             if isinstance(gate, dict)}
    try:
        manifest = registry.get(adapter_id)
    except (OSError, ValueError, ManifestError) as exc:
        return refuse([f"ADAPTER_UNREADABLE: {exc}"])
    failures = []
    matched = []
    for gate_id in required:
        gate = by_id.get(gate_id)
        if gate is None:
            failures.append(f"GATE_NOT_IN_RECEIPT: {gate_id}")
            continue
        status = gate.get("status")
        if status != PASS.value:
            failures.append(f"GATE_NOT_PASS: {gate_id}={status}")
            continue
        if gate_id not in manifest.gates:
            failures.append(f"GATE_UNKNOWN_TO_ADAPTER: {gate_id}")
            continue
        # A passing gate whose declared dependency did not also pass cannot be
        # relied on. verify() already enforces this for the receipt as a whole;
        # it is repeated here so the failure names the requirement that broke.
        missing = [dep for dep in manifest.gates[gate_id].get("dependsOn", [])
                   if (by_id.get(dep) or {}).get("status") != PASS.value]
        if missing:
            failures.append(f"GATE_DEPENDENCY_NOT_PASS: {gate_id} missing {','.join(missing)}")
            continue
        matched.append(gate_id)
    if failures:
        return refuse(failures, api.GateStatus.FAIL, matched=matched, identity=identity)

    # ── (8) current configuration and artifact identity are still valid ─────
    # verify() above re-hashed every bound artifact and configuration file, so
    # reaching here means both are current. Recording it keeps the result
    # explicit for a reader rather than implied by a function call.
    return _result(workspace=workspace_id, adapter=adapter_id, required_gates=required,
                   run_id=run_id, failures=[], status=api.GateStatus.PASS,
                   matched=matched, identity=identity, satisfied=True)
