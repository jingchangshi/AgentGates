# AgentGates V2 x DSH Compiler Mode — Baseline Audit (P0)

Audited at the start of implementation, against the three pinned baselines. Every
statement below was read from the working tree, not from earlier narration.

## 1. Pinned baselines

| Repo | Path | Branch | Baseline SHA | Tree at audit |
|---|---|---|---|---|
| AgentGates | `/home/s00653124/workspace/AgentGates` | `migration/p0-p2` | `f88da67` | clean |
| dsh-compiler-mode | `/home/s00653124/workspace/dsh-compiler-mode` | `main` | `b83c179` (contains `b459980`) | clean |
| deepseek-harness | `/home/s00653124/workspace/deepseek-harness` | `dev` | `b62215edfa` | clean |

Work branches created from these baselines, one per repo:

- AgentGates -> `feat/v2-mcp-async`
- dsh-compiler-mode -> `feat/agentgates-mcp`
- deepseek-harness -> `feat/team-completion-policy`

## 2. AgentGates — what already exists

**470 tests pass** (`.venv/bin/python -m pytest tests -q`) plus 88 subtests, before any change.

| Module | Role | Reused by V2 as |
|---|---|---|
| `manifest.py` | Digest-pinned adapter contracts, fixed recipe argv, gate DAG ordering | **unchanged** |
| `registry.py` | Only installed adapters are executable | extended into the workspace registry |
| `engine.py` | Runs one adapter's gate DAG under source/build leases | **called by the worker**, not copied |
| `executor.py` | Fixed argv through an independent supervisor, `shell=False` | unchanged |
| `process_supervisor.py` | Descendant supervision + lease inheritance | unchanged |
| `locking.py` | `flock` leases per source/build/npu resource | unchanged |
| `evaluator.py` | Observation assertions -> PASS/FAIL | unchanged |
| `evidence.py` | Receipt publish, HMAC attestation, re-verify | unchanged |
| `source_identity.py` | Source tree snapshot identity | unchanged |
| `cli.py` | `list/doctor/run/report/verify` | kept, extended |

Key invariants already established in code that V2 must not break:

1. A receipt is published once with `open("x")` — it is never rewritten.
2. `aggregate()` is positive: PASS requires *every* gate literally PASS. An unknown
   gate status becomes `INVALID_EVIDENCE`, never PASS.
3. A recipe executes only a manifest-pinned `argv` with `shell=False`; placeholders
   are limited to `{pack}` naming a hash-pinned file.
4. Source leases are `flock` descriptors **inherited by the supervisor**, so the
   lease outlives the launching process until descendants are reaped.
5. `engine.run` re-snapshots source, manifest hash, and configuration after the
   gates and demotes PASS -> `STALE` on any drift.

**Conclusion:** the objective's "do not reimplement the Engine" constraint is
satisfiable literally. V2 adds a layer *around* `Engine.run`, it does not fork it.

## 3. AgentGates — what is genuinely missing

| Objective section | Gap found in code |
|---|---|
| B (versioned API) | No data models at all. `cli.py` builds ad-hoc dicts; `engine` returns raw receipt dicts. No stable error codes (a single `except` maps everything to `BLOCKED_ENV`). |
| C (local config) | No config file. Machine parameters arrive as CLI flags (`--env-file`, `--device-id`, `--source`), which is exactly what C2 forbids for agent input. |
| D (async worker) | Entirely absent. `Engine.run` is synchronous and in-process. |
| E (MCP server) | Entirely absent. No `mcp` dependency. |
| — | `EvidenceStore.latest()` exists and is what `report`/`verify` default to — E3 explicitly forbids `latest` for requirement verification. |

## 4. deepseek-harness — Team task surface

- `task-board.ts` (255 lines): the `complete` case is unconditional — authorization,
  an `in_progress` check, then `next = {...current, status:'completed'}`. **There is no
  seam before the state commit.**
- `journal.ts` (73 lines): `transact()` serializes per-Lead operations on a promise
  chain; `appendAndFlush()` appends then flushes. The CAS check
  (`current.revision !== request.expectedRevision`) happens **inside** the transaction.
- `projection.ts` (394 lines): the durable `TeamTaskSnapshot` is zod-validated with
  `.strict()` and a literal `version: 2`. Adding a snapshot field **requires** a
  projection schema change or old events fail validation.
- `tool-agent-team/src/index.ts` (279 lines): hand-written JSON `parameters` and
  `ValueSchemaSpec` output schemas.

**Baseline:** `npx vitest run packages/experimental/agent-team packages/experimental/tool-agent-team`
-> 110 passed. `npx tsc -b tsconfig.host.json` -> clean.

### 4.1 The transaction-vs-verification tension (G4)

The objective warns that an external verify may need to re-read a large source tree
and must not hold the Team transaction queue indefinitely. The existing design runs
CAS **and** append inside `journal.transact`, which is a per-Lead serial queue. A
verification that re-hashes a compiler tree takes seconds to minutes and would block
every other task mutation in the Team.

Decision (ADR-004): the completion policy runs **outside** the transaction, and the
transaction then **re-checks** revision, requirement digest and evidence reference
before committing. This preserves G4's guarantees (no silent revision swap, no
completed event on failure) without holding the queue for the duration of an
external verify. Measured verification latency is recorded in the test report.

## 5. deepseek-harness — MCP client surface

`packages/mcp/mcp-client` (`@deepseek-ai/dsh-mcp-client`):

```ts
export const name = 'mcp-client'
export const inject = ['tools']
const SERVER_NAME_PATTERN = /^[A-Za-z0-9_-]{1,32}$/
const activeServerNames = new WeakMap<object, Set<string>>()
```

- One instance connects to **one** MCP server; tools register as
  `mcp__<serverName>__<rawName>`. `agentgates` is a legal namespace, so
  `mcp__agentgates__submit` is the public tool name.
- `activeServerNames` is scoped to the **registration scope**: "agent-scoped MCP
  servers may reuse a namespace in another Agent, while global instances and
  duplicates inside one Agent remain mutually exclusive."
- A stdio instance spawns a child process with `reconnect` defaults,
  `toolCallTimeoutMs` (default 60 000), and `failOnStartupError`.

**Consequence for F2:** confining AgentGates tools to the two compiler presets
depends on the registration scope of the mounted row. A bundle-level (host) row in
`bundles.patch.yml` registers into the root scope and is visible to every preset.
Preset-only visibility requires the preset to own the row, which requires the
preset realm to isolate the MCP client — the same constraint already documented in
`cordis.patch.yml` for `lsp`. The resolution and its real measurements are in
`docs/v2/mcp-scoping.md`.

## 6. dsh-compiler-mode — current bundle

- Two presets: `ai-compiler` (order 5) and `ai-compiler-magpie` (order 6), both
  `@deepseek-ai/dsh-agent-preset` rows in `config/bundle-ai-compiler/cordis.patch.yml`.
- Host layer `bundles.patch.yml` mounts `lsp`, `lsp-stdio`, `tool-lsp`,
  `agent-team`, `tool-agent-team`, `ui-agent-team`.
- Scripts: `install.sh`, `uninstall.sh`, `rollback.sh`, `verify.sh`, plus
  `patches/0001-team-agent-options.patch`.
- Both presets must gain any new row symmetrically to satisfy F2.

## 7. Environment facts established before design

| Fact | Measurement |
|---|---|
| user systemd available | `systemctl --user is-system-running` -> `running`; `/run/user/1071` exists; `dbus.service` active |
| existing user units | `dbus.service`, `dsh-web.service` (DSH already runs as a user unit) |
| Python | `/home/s00653124/workspace/AgentGates/.venv/bin/python` = 3.11.14 |
| MCP Python SDK | `mcp 2.3.0` installs cleanly from PyPI into that venv |
| DSH tooling | `npx vitest run <pkgs>` and `npx tsc -b tsconfig.host.json` both work |
| docker | available; images `npuir/cann9.1.0-950-py311`, `npuir/cann9.0.0-py311`, `npuir-cann950:9.1.0`, `tilelang-hivm-dev:24.04` present |
| NPU | `npu-smi` present at `/usr/local/bin/npu-smi` |

**D2 is therefore achievable with real user-level systemd** — it is not a BLOCKED
path. Whether the *compiler* gates are runnable is a separate question, answered by
the P6 reconnaissance and recorded in the test report.
