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
execution_id/run_id, request_id, the typed target/candidate request pair, an
optional `device_id` that can only select from the operator-authorized cards,
and a bounded wait duration. Everything else — source root, env_file, python
executable, CANN paths, conda init, evidence_root, the device allowlist and the
default device — is resolved from `~/.config/agent-gates/config.toml`. The
device selector narrows the configured set; it can never widen it.

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

## ADR-008 — NPU device policy: sharing is the default, exclusivity is opt-in

**Status:** revised — supersedes the original "explicit authorization only" rule.

### What changed and why

The first version required an operator to list `authorized_npu_devices` before any
NPU gate could run, and refused a card another process was using. On a real
shared machine that was wrong in two ways:

1. **It blocked legitimate work.** A job already running on a card does not
   reserve it. Every other workload on this host shares the devices; a gate that
   refused to start would be the only thing on the machine that could not.
2. **"An idle card is not an authorized card" was solving the wrong problem.**
   The concern is a *reproducible* run, not an *exclusive* one. Exclusivity is one
   way to get reproducibility and it is the wrong default.

### The policy now

| Setting | Default | Meaning |
|---|---|---|
| `devices.allowed_npu_devices` | absent = **every device** | Optional restriction. Present and non-empty narrows the set. |
| `devices.default_npu_device` | lowest permitted id | Pins the selection. Must appear in the allowlist. |
| `devices.require_exclusive_device` | `false` | Opt-in. When true, a card held by another process is `BLOCKED_ENV`. |

Selection is deterministic: with no configuration a gate uses device `0`. A gate
that silently picked "whatever looks idle" would make two identical requests run
on different hardware, and the receipt would stop describing a reproducible
machine state.

A request may still name one permitted card explicitly (`--device-id` in the
CLI, `device_id` in the V2 API) so an operator can move work off a failing card.
The id is canonicalised and checked against the same allowlist; omitting it keeps
the deterministic default. The resolved id is part of the request identity: the
record, the execution status and the receipt all name it, and replaying the same
`request_id` on another card is a `CONFLICT` rather than a silent switch.

### Occupancy is reported, not enforced

`npu-smi` is still consulted — but to *describe* the card, not to veto it. The
configuration check says exactly what it found:

```
configuration ready: ... device=0 npuId=0 occupancy=shared(6) exclusive=False
```

A reader can therefore tell a quiet run from a contended one, which is the
information exclusivity was costing. An unreadable driver answer is still a
refusal: "unknown" is not "shared".

### Exclusivity remains available

An operator who needs an uncontended card sets `require_exclusive_device = true`.
AgentGates forwards it to the adapter as
`TILELANG_HIVM_REQUIRE_EXCLUSIVE_DEVICE=1`, and a held card then reports
`BLOCKED_ENV` (exit 75) rather than running a test whose timing or memory another
job could distort. That is a deliberate choice, not a precondition.

### What this costs, stated plainly

A shared card can make a *timing-sensitive* or memory-heavy test flaky, and the
gate will report that flakiness as a verdict about the tree. The mitigation is to
opt into exclusivity for those runs, or to read `occupancy=shared(n)` in the
configuration check before treating a failure as a code regression.

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
