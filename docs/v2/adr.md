# AgentGates V2 — Architecture Decision Records

These records fix the decisions the objective declares immutable, and record the
decisions taken during implementation where the objective left a choice open.

---

## ADR-001 — DSH integration through AgentGates' native MCP server

**Status:** accepted (mandated by objective A1).

**Decision.** AgentGates exposes a Model Context Protocol server implemented inside
the Python package, speaking stdio. DSH consumes it through the official
`@deepseek-ai/dsh-mcp-client` plugin with `serverName: agentgates`, producing public
tool names `mcp__agentgates__<tool>`.

**Rejected.** A TypeScript CLI-parsing bridge in `dsh-compiler-mode`. It would
duplicate the validation logic the objective forbids duplicating and would give two
places to compute PASS/FAIL.

**Consequence.** The CLI and the MCP server must be thin adapters over the same
Python API — one place computes verdicts (`docs/v2/api-schema.md`).

---

## ADR-002 — Gate execution is asynchronous, in a systemd user worker

**Status:** accepted (mandated by objective A1).

**Decision.** `submit()` persists an ExecutionRequest and starts
`agent-gates-worker@<executionId>.service`, a user-level systemd template instance,
then returns an execution id with status SUBMITTED/STARTING without waiting for the
build. The worker calls the existing `Engine`; the MCP server never holds a build
process.

**Why systemd, not an in-process asyncio task.** Objective A1 requires that a DSH
Web or MCP server restart does not end a submitted Gate. An in-process task dies with
its process; a user unit survives independently and records its outcome in journald.

**Enforcement.** If user-level systemd is unavailable the Execution Manager must
record `BLOCKED_ENV` with a stable error code. It must **not** silently fall back to
in-process execution and claim the persistence objective is met.

---

## ADR-003 — Completion gating is opt-in per Team Task

**Status:** accepted (mandated by objective A1).

**Decision.** A Team Task is gated only when it explicitly declares verification
requirements in its durable record. Tasks without requirements keep byte-identical
behaviour. The completion policy is a generic DSH extension point; the AgentGates
implementation is a plugin in `dsh-compiler-mode`.

**Consequence.** No parsing of Markdown descriptions for gate requirements (also
forbidden by G3). The requirement payload is opaque to DSH.

---

## ADR-004 — Policy runs outside the Team transaction, then re-checks

**Status:** accepted — decision taken during implementation.

**Context.** `TeamJournal.transact` is a per-Lead serial queue. G4 requires that
verification cannot be bypassed and that a revision cannot be silently swapped
between check and commit, while also warning that an external verify may be slow.
Holding the queue for the duration of a source re-hash would block unrelated task
mutations.

**Decision.** Run the completion policy *before* entering the transaction, passing
it a snapshot of the task (revision, requirement digest, evidence reference). Enter
the transaction, re-read the task, and require all three to be unchanged before
appending `completed`. A mismatch is a rejection, not a retry.

**Guarantees this preserves (the ones G4 actually names):**

1. Same task revision — enforced by the in-transaction CAS *and* the re-check.
2. Same completion requirement — compared by canonical digest.
3. Same evidence reference — compared by value.
4. Verification failure produces no `completed` event — the policy throws before
   the append.
5. Only successful verification permits the commit.
6. Validator timeout is a rejection (the policy receives an abort signal and must
   surface it as the same error type as a failure).
7. Concurrent updates recheck conditions — the in-transaction re-read does this.

**Cost.** The verify runs concurrently with other transactions, so two tasks may
verify simultaneously. That is acceptable: verification is read-only against
AgentGates state, and the commit remains serialized.

---

## ADR-005 — Two distinct schemas: API and Receipt

**Status:** accepted (mandated by objective B1).

**Decision.** `agent_gates/api_types.py` models the agent-facing surface
(WorkspaceInfo, AdapterInfo, GateInfo, ExecutionRequest, ExecutionStatus,
ExecutionResult, GateResult, EvidenceVerification, VerificationRequirement,
RequirementVerificationResult, APIError). The internal receipt schema stays in
`engine.py`/`evidence.py` and is unchanged.

**Why they must not merge.** The receipt is a signed historical artifact whose
format is pinned by existing tests and by published receipts on disk. The API is a
stable contract agents bind to and will evolve independently. Collapsing them would
make a receipt-format change a breaking API change, and an API change a receipt
invalidation.

---

## ADR-006 — Machine parameters never come from the model

**Status:** accepted (mandated by objective C2).

**Decision.** Tool inputs are limited to: workspace, adapter, gates,
execution_id/run_id, request_id, and a bounded wait duration. Everything else —
source root, env_file, python executable, CANN paths, conda init, evidence_root,
device id — is resolved from `~/.config/agent-gates/config.toml`.

**Enforcement.** The API models for tool input simply have no field for those
values, so there is nothing to validate away. Workspace ids are looked up in the
registry; unknown ids are a stable `UNKNOWN_WORKSPACE` error, not a path fallback.

---

## ADR-007 — Error taxonomy is differentiated, not collapsed

**Status:** accepted (mandated by objective B2).

**Decision.** `APIError` carries a stable `code` in six families:

| Family | Codes (examples) | Meaning |
|---|---|---|
| request | `INVALID_REQUEST`, `UNKNOWN_WORKSPACE`, `UNKNOWN_ADAPTER`, `UNKNOWN_GATE`, `CONFLICT` | caller asked for something malformed or ungranted |
| config | `CONFIG_INVALID`, `CONFIG_MISSING_FIELD`, `CONFIG_UNSAFE_PATH` | local configuration is wrong; names the field |
| resource | `RESOURCE_BUSY` | a source/build/npu lease is held |
| execution | `EXECUTION_FAILED`, `EXECUTION_NOT_FOUND`, `EXECUTION_CANCELLED`, `EXECUTION_INTERRUPTED` | the run's own outcome |
| communication | `WORKER_UNAVAILABLE`, `SYSTEMD_UNAVAILABLE`, `WORKER_START_FAILED` | we could not reach or start the executor |
| receipt | `RECEIPT_INVALID`, `RECEIPT_STALE`, `REQUIREMENTS_UNSATISFIED` | evidence exists but does not satisfy |

**Rule.** `BLOCKED_ENV` is reserved for a *gate verdict* meaning the environment
prevented the test from running. It is never used as a catch-all for transport or
code faults.

---

## ADR-008 — NPU device authorization is explicit-only

**Status:** accepted (mandated by objective C4).

**Decision.** An idle device is not an authorized device. NPU gates run only on a
device id listed in the local configuration under an explicit operator grant. With
no grant, the NPU gate is `BLOCKED_ENV` and its test command is never invoked.

**Rationale.** Selecting a free card silently would make a regression run on
hardware the operator did not choose, and would make `verify_requirements` report a
PASS for a machine state nobody approved.

---

## ADR-009 — Completion gating verifies submission-time source only

**Status:** accepted (mandated by objective G4, final paragraph).

**Decision.** `verify_requirements` binds a receipt to the source identity captured
at gate time and compares it against the source identity *now*. A task completing
therefore proves the evidence matched the tree at completion. If the tree changes
afterwards, the receipt becomes `STALE` and must be re-verified before final
delivery.

**Documentation duty.** This scope limit is stated in the user-facing docs, because
a reader could otherwise believe a completed task certifies the tree forever.
