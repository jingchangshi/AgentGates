"""The versioned, agent-facing API contract for AgentGates V2.

This module is the ONLY place that decides what a caller is told about a run.
The CLI renders these models as JSON and the MCP server renders the same models
as `structuredContent`; neither recomputes a verdict. Keeping one model set
is what stops "the CLI said PASS and the tool said FAIL" from being possible.

The internal Evidence Receipt schema is deliberately NOT modelled here: a receipt
is a signed historical artifact whose format is pinned by published files on
disk, while this contract is free to evolve. See docs/v2/adr.md ADR-005.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

API_SCHEMA_VERSION = 1


class _Model(BaseModel):
    """Base for every API value: closed, strict, and JSON-ready."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ── execution lifecycle (B2) ──────────────────────────────────────────────────

class ExecutionState(str, Enum):
    """Where an execution is in its lifecycle.

    This is NOT the gate verdict. An execution reaching FINISHED says the worker
    stopped, not that the compiler tests passed; only a gate verdict says that.
    """

    SUBMITTED = "SUBMITTED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


TERMINAL_STATES = frozenset({
    ExecutionState.FINISHED,
    ExecutionState.CANCELLED,
    ExecutionState.INTERRUPTED,
})


class GateStatus(str, Enum):
    """A gate's verdict, unchanged from the pre-V2 engine."""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED_ENV = "BLOCKED_ENV"
    NOT_RUN = "NOT_RUN"
    STALE = "STALE"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"


# ── error taxonomy (B2, ADR-007) ──────────────────────────────────────────────

class ErrorFamily(str, Enum):
    """Which layer refused. The family is what a caller branches on."""

    REQUEST = "request"
    CONFIG = "config"
    RESOURCE = "resource"
    EXECUTION = "execution"
    COMMUNICATION = "communication"
    RECEIPT = "receipt"


class ErrorCode(str, Enum):
    """Stable codes. Callers may match on these; prose never changes them."""

    # request
    INVALID_REQUEST = "INVALID_REQUEST"
    UNKNOWN_WORKSPACE = "UNKNOWN_WORKSPACE"
    UNKNOWN_ADAPTER = "UNKNOWN_ADAPTER"
    UNKNOWN_GATE = "UNKNOWN_GATE"
    GATE_NOT_ALLOWED = "GATE_NOT_ALLOWED"
    ADAPTER_NOT_ALLOWED = "ADAPTER_NOT_ALLOWED"
    DEVICE_NOT_AUTHORIZED = "DEVICE_NOT_AUTHORIZED"
    CONFLICT = "CONFLICT"
    # config
    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_MISSING_FIELD = "CONFIG_MISSING_FIELD"
    CONFIG_UNSAFE_PATH = "CONFIG_UNSAFE_PATH"
    # resource
    RESOURCE_BUSY = "RESOURCE_BUSY"
    CONCURRENCY_LIMIT = "CONCURRENCY_LIMIT"
    # execution
    EXECUTION_NOT_FOUND = "EXECUTION_NOT_FOUND"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    EXECUTION_INTERRUPTED = "EXECUTION_INTERRUPTED"
    # communication
    WORKER_UNAVAILABLE = "WORKER_UNAVAILABLE"
    SYSTEMD_UNAVAILABLE = "SYSTEMD_UNAVAILABLE"
    WORKER_START_FAILED = "WORKER_START_FAILED"
    # receipt
    RECEIPT_INVALID = "RECEIPT_INVALID"
    RECEIPT_STALE = "RECEIPT_STALE"
    RECEIPT_NOT_FOUND = "RECEIPT_NOT_FOUND"
    REQUIREMENTS_UNSATISFIED = "REQUIREMENTS_UNSATISFIED"
    VERIFIER_UNAVAILABLE = "VERIFIER_UNAVAILABLE"


FAMILY_OF_CODE = {
    ErrorCode.INVALID_REQUEST: ErrorFamily.REQUEST,
    ErrorCode.UNKNOWN_WORKSPACE: ErrorFamily.REQUEST,
    ErrorCode.UNKNOWN_ADAPTER: ErrorFamily.REQUEST,
    ErrorCode.UNKNOWN_GATE: ErrorFamily.REQUEST,
    ErrorCode.GATE_NOT_ALLOWED: ErrorFamily.REQUEST,
    ErrorCode.ADAPTER_NOT_ALLOWED: ErrorFamily.REQUEST,
    ErrorCode.DEVICE_NOT_AUTHORIZED: ErrorFamily.REQUEST,
    ErrorCode.CONFLICT: ErrorFamily.REQUEST,
    ErrorCode.CONFIG_INVALID: ErrorFamily.CONFIG,
    ErrorCode.CONFIG_MISSING_FIELD: ErrorFamily.CONFIG,
    ErrorCode.CONFIG_UNSAFE_PATH: ErrorFamily.CONFIG,
    ErrorCode.RESOURCE_BUSY: ErrorFamily.RESOURCE,
    ErrorCode.CONCURRENCY_LIMIT: ErrorFamily.RESOURCE,
    ErrorCode.EXECUTION_NOT_FOUND: ErrorFamily.EXECUTION,
    ErrorCode.EXECUTION_FAILED: ErrorFamily.EXECUTION,
    ErrorCode.EXECUTION_CANCELLED: ErrorFamily.EXECUTION,
    ErrorCode.EXECUTION_INTERRUPTED: ErrorFamily.EXECUTION,
    ErrorCode.WORKER_UNAVAILABLE: ErrorFamily.COMMUNICATION,
    ErrorCode.SYSTEMD_UNAVAILABLE: ErrorFamily.COMMUNICATION,
    ErrorCode.WORKER_START_FAILED: ErrorFamily.COMMUNICATION,
    ErrorCode.RECEIPT_INVALID: ErrorFamily.RECEIPT,
    ErrorCode.RECEIPT_STALE: ErrorFamily.RECEIPT,
    ErrorCode.RECEIPT_NOT_FOUND: ErrorFamily.RECEIPT,
    ErrorCode.REQUIREMENTS_UNSATISFIED: ErrorFamily.RECEIPT,
    ErrorCode.VERIFIER_UNAVAILABLE: ErrorFamily.RECEIPT,
}


class APIError(Exception):
    """One refusal, with a stable code and a machine-readable family.

    Raised everywhere inside AgentGates so adapters (CLI, MCP, policy plugin)
    never have to guess which layer failed.
    """

    def __init__(self, code: ErrorCode, message: str, *, field: str | None = None,
                 details: dict[str, Any] | None = None):
        self.code = ErrorCode(code)
        self.family = FAMILY_OF_CODE[self.code]
        self.message = message
        self.field = field
        self.details = dict(details or {})
        super().__init__(f"{self.code.value}: {message}")

    def to_model(self) -> "APIErrorModel":
        return APIErrorModel(code=self.code.value, family=self.family.value,
                             message=self.message, field=self.field, details=self.details)


class APIErrorModel(_Model):
    """Serialised {@link APIError}."""

    code: str
    family: str
    message: str
    field: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


# ── registry surfaces ─────────────────────────────────────────────────────────

class GateInfo(_Model):
    """One gate a registered adapter can run."""

    id: str
    recipe_id: str
    depends_on: list[str] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)


class AdapterInfo(_Model):
    """One installed adapter as this configuration exposes it."""

    id: str
    manifest_sha256: str
    gates: list[GateInfo] = Field(default_factory=list)


class WorkspaceInfo(_Model):
    """One registered workspace and the adapters/gates it may use here."""

    id: str
    source_root: str
    adapters: list[AdapterInfo] = Field(default_factory=list)
    allowed_gates: list[str] | None = None


class DoctorCheck(_Model):
    """One configuration probe, mirroring the engine's doctor output."""

    recipe_id: str
    ok: bool
    exit_code: int | None = None
    reason: str | None = None


class DoctorReport(_Model):
    """Result of a read-only environment probe.

    A doctor PASS proves the machine is configured. It is NOT evidence that any
    gate ran, and it can never satisfy a completion requirement (E3).
    """

    workspace: str
    adapter: str
    status: GateStatus
    checks: list[DoctorCheck] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


# ── execution ────────────────────────────────────────────────────────────────

class ExecutionRequest(_Model):
    """What a caller asked for. Every machine parameter was resolved by config."""

    workspace: str
    adapter: str
    gates: list[str]
    request_id: str | None = None


class GateResult(_Model):
    """One gate's verdict inside an execution."""

    gate_id: str
    status: GateStatus
    failures: list[str] = Field(default_factory=list)
    recipe_id: str | None = None


class ExecutionStatus(_Model):
    """A point-in-time view of one execution. Safe to poll."""

    execution_id: str
    state: ExecutionState
    workspace: str
    adapter: str
    gates: list[str]
    request_id: str | None = None
    run_id: str | None = None
    submitted_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    current_gate: str | None = None
    completed_gates: list[str] = Field(default_factory=list)
    message: str | None = None
    exit_reason: str | None = None


class ExecutionResult(_Model):
    """The terminal outcome of one execution.

    `gate_status` is the aggregated gate verdict and is `None` unless the
    execution actually finished and produced a receipt. `state` being FINISHED
    with `gate_status` FAIL is the normal, expected combination.
    """

    execution_id: str
    state: ExecutionState
    workspace: str
    adapter: str
    gates: list[str]
    run_id: str | None = None
    gate_status: GateStatus | None = None
    gate_results: list[GateResult] = Field(default_factory=list)
    submitted_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_reason: str | None = None
    message: str | None = None


# ── evidence verification ────────────────────────────────────────────────────

class EvidenceVerification(_Model):
    """The result of re-verifying one receipt against the present machine."""

    run_id: str
    status: GateStatus
    valid: bool
    failures: list[str] = Field(default_factory=list)


class VerificationRequirement(_Model):
    """Explicit conditions a completion must satisfy.

    This is what a Team Task declares and what the completion policy forwards to
    `verify_requirements`. It contains no machine parameter: the workspace id
    is resolved through the local configuration.
    """

    workspace: str
    adapter: str
    required_gates: list[str]
    # Optional so that "a task arrived with no evidence reference" is a
    # verifier refusal (RUN_ID_REQUIRED) the policy can report precisely,
    # rather than a schema error that never reaches the verifier.
    run_id: str | None = None


class RequirementVerificationResult(_Model):
    """The answer to `does this run satisfy these requirements right now?`.

    `satisfied` is true only when every server-side check passed. `failures`
    names each failed check with a stable token so a policy plugin can report
    precisely why completion was refused.
    """

    satisfied: bool
    status: GateStatus
    failures: list[str] = Field(default_factory=list)
    run_id: str | None = None
    verified_at: str
    current_source_identity: dict[str, Any] | None = None
    matched_gates: list[str] = Field(default_factory=list)
    workspace: str
    adapter: str
    required_gates: list[str] = Field(default_factory=list)


# ── wait ─────────────────────────────────────────────────────────────────────

class WaitResult(_Model):
    """A bounded wait. `timed_out` true means nothing is decided yet."""

    execution_id: str
    timed_out: bool
    status: ExecutionStatus


def jsonable(value):
    """Render any API value as plain JSON, keeping enums as their tokens."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
