# AgentGates V2 — MCP API Schema

Server name: `agentgates` (legal for DSH: `[A-Za-z0-9_-]{1,32}`).
Version: `2.0.0`. Transport: stdio. Entry point:

```
<interpreter> -s -m agent_gates.mcp_server
```

DSH exposes the tools as `mcp__agentgates__<tool>`.

## Shared rules

1. **Every tool declares an input schema and an output schema.** The output
   schema is derived from the same Pydantic model the CLI serialises, so the two
   surfaces cannot drift.
2. **`structuredContent` carries the machine-readable answer.** A model never
   has to parse prose to learn a verdict.
3. **A failure is an error, never a success-shaped object.** A tool that cannot
   answer raises; the client sees `isError: true`.
4. **Undeclared arguments are refused.** The SDK would silently drop them, which
   would let a caller believe a machine parameter had been honoured. AgentGates
   rejects them by name.
5. **No tool accepts a machine parameter.** There is no field for a command, a
   container, an interpreter, an env file, a source directory or a device id.

## Tools

### `list_workspaces`

| Input | Type | Notes |
|---|---|---|
| `workspace` | string? | omit to list all |

`structuredContent`: `{ workspaces: [{ id, source_root, adapters: [{ id,
manifest_sha256, gates: [{ id, recipe_id, depends_on, checks }] }],
allowed_gates }] }`

### `list_gates`

| Input | Type |
|---|---|
| `workspace` | string |
| `adapter` | string |

`structuredContent`: `{ workspace, adapter, gates: [{ id, recipe_id, depends_on,
checks }] }`. Refuses an adapter the workspace is not registered for.

### `doctor`

Probes the machine configuration for one adapter. Runs **no compiler test** and
**can never satisfy a completion requirement** — it produces no run id.

`structuredContent`: `{ workspace, adapter, status, checks: [{ recipe_id, ok,
exit_code }], failures }`

### `submit`

| Input | Type | Notes |
|---|---|---|
| `workspace` | string | registered id |
| `adapter` | string | must be registered for that workspace |
| `gates` | string[] | dependencies are added automatically |
| `request_id` | string? | idempotency key |

Returns immediately with an `execution_id`. `request_id` semantics: same request
+ same id → the original execution; same id + different request → `CONFLICT`; new
id → new execution.

`structuredContent`: `ExecutionStatus` — `{ execution_id, state, workspace,
adapter, gates, request_id, run_id, submitted_at, started_at, finished_at,
current_gate, completed_gates, message, exit_reason }`

### `status` / `cancel`

| Input | Type |
|---|---|
| `execution_id` | string (32 hex) |

``status`` returns `ExecutionStatus`. `cancel` is idempotent, stops the unit's
whole process group, and never yields `PASS`.

### `wait`

| Input | Type | Notes |
|---|---|---|
| `execution_id` | string | |
| `timeout_seconds` | number, 0–60 | clamped server-side |

``timed_out: true`` is a normal answer meaning "not decided yet". A single call
can never block until a build finishes.

### `report` / `verify`

| Input | Type |
|---|---|
| `run_id` | string |

`report` returns the receipt plus its current verification. `verify` returns
`EvidenceVerification` — `{ run_id, status, valid, failures }`.

### `verify_requirements` — the completion gate

| Input | Type | Notes |
|---|---|---|
| `workspace` | string | |
| `adapter` | string | the run must be bound to it |
| `required_gates` | string[] | every one must have passed |
| `run_id` | string | explicit; **no `latest` shortcut** |

`structuredContent`: `RequirementVerificationResult`

```json
{
  "satisfied": false,
  "status": "STALE",
  "failures": ["RECEIPT_INVALID: STALE", "RECEIPT_FAILURE: SOURCE_IDENTITY_CHANGED"],
  "run_id": "8866f8ff996362209cc8574cb33ec1ea",
  "verified_at": "2026-10-27T12:19:44.981472+00:00",
  "current_source_identity": { ... },
  "matched_gates": [],
  "workspace": "tilelang",
  "adapter": "tilelang-hivm",
  "required_gates": ["tilelang-hivm-build", "tilelang-hivm-regression"]
}
```

Server-side checks, in the order they run:

1. the run id resolves to a real receipt (no `latest`),
2. the receipt is complete and re-verifies,
3. the receipt's source root equals the workspace's **current** configured root,
4. the receipt's adapter equals the required adapter,
5. every required gate is present and `PASS`,
6. every required gate's declared dependencies also passed,
7. the current source identity equals the receipt's,
8. the current configuration and artifact identities are still valid.

A `doctor` PASS and a historical PASS both fail these checks by construction.

## Error taxonomy

Refusals carry a stable `code` and a `family` (see `adr.md` ADR-007). `family`
tells the caller which layer refused: `request`, `config`, `resource`,
`execution`, `communication` or `receipt`. `BLOCKED_ENV` is a gate verdict and
never an error code.
