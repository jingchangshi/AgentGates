# Migration final report

Repository `jingchangshi/AgentGates`, branch `migration/p0-p2` (the tip of that branch is the commit that carries this file), execution date 2026-10-27. Prior commits: `cc573c7` (inventory), `3994c69` (audited baseline and extraction semantics). `main` was deliberately not advanced; see [the delivery note](progress.md#delivery).

## Objective and scope

Extract a standalone AgentGates from PiEngineeringHarness (`9ca04ba6814154a778ecf30168c793d73352e643`): Python 3.11+, an `agent-gates` CLI, no runtime dependency on PiEngineeringHarness, Claude Code or an agent runtime. Scope: P0 audit and baseline, P1 standalone core (three adapters, five gates), P2 Claude Code plugin, docs, this report. Not migrated: Harness task stores, approvals, sessions, model routing, orchestration.

## What was delivered

### P0 — original baseline

Original digest-verified recipes executed and independently evaluated; manifests and pinned files matched the source checkout byte-for-byte.

| Gate | Result | Seconds | Evidence |
| --- | --- | ---: | --- |
| `ascend-build` | PASS | 80.670 | Three Ninja targets completed |
| `ascend-regression` | PASS | 410.713 | 1142 discovered / 1023 passed / 119 unsupported |
| `tilelang-hivm-build` | PASS | 0.615 | Incremental `tilelang` target |
| `tilelang-hivm-regression` | PASS | 8.727 | 14 collected / 14 passed / 0 skipped, 12 native libraries |
| `tilelang-hivm-npu-sanity` | BLOCKED_ENV | — | No assigned device id |

Source HEADs: AscendNPU-IR `bfefe18011152cca671e7e6ddf03793afbacaa2a`; TileLang `6fe6f09182c854e680a5488f4d8461fe5d5a76a6` with five pre-existing tracked modifications preserved. **P0 gap:** the 12 individual native `.so` hashes were not persisted and were not retroactively repaired.

### P1 — standalone CLI

| Adapter | `doctor` | `run` | Run ID | Gates | `verify` |
| --- | --- | --- | --- | --- | --- |
| `ascendnpu-ir` | PASS 0.196 s `4f713ee9129568c83fde8b2887180360` | PASS 57.873 s | `52ea3b30495dba9d086cabe0dc9c38bb` | build PASS, regression PASS | `valid: true` |
| `tilelang-hivm` | PASS 0.195 s `59f4093c1f2c1c44b7318b2919bbc2dc` | PASS 21.057 s | `61ed95d533a314da934e4e8e6cd6eca7` | build PASS, regression PASS (14/14, 12 native libs) | `valid: true` |
| `tilelang-hivm-npu` | BLOCKED_ENV `2f5ee7dc81905e8805acc2ac7f017cd8` | BLOCKED_ENV | `e6d1680cd9b2972d645729089cc5a1ae` | no device assignment | `valid: false` |

These are the round-2 runs, taken after the independent review's fixes changed the adapter scripts. Round 1 (`c1610b67…`, `181cad5a…`, `414c2ca7…`) is retained under `.evidence/p1-runs` and now verifies as `STALE` (`ADAPTER_IDENTITY_CHANGED`), which is the correct outcome for evidence taken against different adapter bytes; the current evidence root is `.evidence/p1-runs-r2`.

An occupancy probe with explicit device ids (read-only host `npu-smi`; container `npu-smi` cannot load `libc_sec.so`) blocked both visible cards with `CONFIG_CHECK_FAILED`: device 0, run `57063f7d46de74f1cee9b8ac87c761ff`, "already in use by 7 process(es) this Gate did not start"; device 1, run `b767bf002a52e80b721ab1180f927e72`, 4 such processes. Both `verify` `valid: false`. The holder counts differ from round 1 because they are a live reading of a shared machine.

Offline suite: **452 passed, 88 subtests passed, exit 0**, logged to ignored `.evidence/final-pytest.log`. The count grew from 441 because the review fixes added eleven regression tests, one or more per confirmed defect; the earlier 439 figure was the suite before the two packaging tests. Isolated wheel proof (`.evidence/check_wheel.py`): PASS — the wheel contains every manifest-pinned file and the adapter deployment files, and no `env.sh`, `config.env`, `__pycache__` or machine configuration; it installs into a fresh venv away from the checkout where `agent-gates list --json` works and `engineering`, `anthropic`, `claude_code` are not importable. A fresh clone of the branch tip also passes the suite (452 tests) and `claude plugin validate`.

### P2 — Claude Code plugin

`claude plugin validate ./integrations/claude-code` passes with no warnings (re-run during this report). Two fresh sessions were run. Round 1 (Claude Code 2.1.296, session `63e5fbf2-2815-4f03-995f-3b8d3bff4117`) exited 0 after 243.3 s and was taken before the review fixes changed the adapter scripts; its receipts now verify `STALE` / `ADAPTER_IDENTITY_CHANGED`. Round 2 (session `e8fbc41e-0bbf-42b4-b522-15d8bf8e5381`) exited 0 after 201.6 s and is the current evidence. Both `init` events record the plugin as `agent-gates@inline` v0.1.0 contributing exactly three AgentGates skills — `agent-gates:doctor`, `agent-gates:verify`, `agent-gates:report`, all `disable-model-invocation: true` — alongside the harness's own global skills; `hooks/hooks.json` declares the `TaskCompleted` hook, though no `TaskCompleted` event occurred in either session, so it was never invoked there. Both sessions reported model `group/auto-deepseek-v4-1-flash`, flagged `unrecognized_model`; neither was Opus.

Round-2 plugin-initiated runs on the TileLang checkout (evidence root `.evidence/p2-plugin-runs-r2`): doctor PASS `9d59fc2378fcadaafb5d0f949f4498ba`; run PASS `a62906a815e77f9001c706c0c932b7dc` (build 0.616 s, regression 8.13 s, 13 bound artifacts); verify `valid: true`; report PASS. This duplicates the P1 TileLang source, **not** a third independent validation. Round 1's `1bab01b8…` and `74e428e9…` are retained and verify `STALE`.

Two isolated worktree exercises both FAILED in both rounds; round-2 run IDs (`.evidence/p2-worktree-runs-r2`):

| Worktree | Change | Run ID | Result |
| --- | --- | --- | --- |
| `ascend-dev` | Whitespace-only fix in `bishengir/test/Transforms/canonicalize-module.mlir:9` | `b2ef7ee46e15cfe90313f099de2a8afd` | FAIL: build `NONZERO_EXIT`, recipe exit 3 (no `build/build.ninja`); regression `NOT_RUN` |
| `tilelang-dev` | One assertion at `testing/ascend/target/test_tilelang_ascend_target.py:30` | `62eec6254852af279ace6b5651e365e1` | FAIL: build `NONZERO_EXIT`, recipe exit 2 (no build directory); regression `NOT_RUN` |

Both `verify` results were `valid: false`. Neither run was redirected to the original checkouts; both worktree diffs remain uncommitted. Round 1's `6b14a3dc…` and `b1e29f54…` are retained and verify `STALE`.

## Architecture

`manifest.py` and `registry.py` load digest-pinned adapter data and reject schema, dependency, path-escape and pin violations. `engine.py` resolves configuration, takes leases, runs preflight, captures source identity before and after, and publishes evidence; `executor.py` dispatches fixed argv with `shell=False`. `evaluator.py` is type-strict; `source_identity.py` hashes HEAD plus tracked, dirty and untracked bytes; `evidence.py` signs and revalidates receipts without reusing a historical PASS. The plugin is CLI-only with an opt-in task-completion receipt check.

## Verification evidence

| Claim | Evidence | Status |
| --- | --- | --- |
| Original recipes still pass | P0 baseline, four gates | Verified |
| CLI reproduces gate outcomes | P1 runs, matching counts | Verified |
| No Harness/Claude runtime dependency | Wheel install + import assertions | Verified |
| Offline correctness | 452 tests, 88 subtests, exit 0 (`.evidence/final-pytest.log`) | Verified |
| Plugin loads and runs gates | Two real sessions, three plugin skills discovered and invoked, round-2 run PASS | Verified |
| Repository private | Auth `ls-remote` exit 0; nonexistent-repo control 128; anonymous API 404 vs public 200; 102-repo public listing omits it | Proven private |
| NPU gate on hardware | None in any phase; the gate is covered only by offline fixtures and negative tests | Not measured |
| Independent Opus review | `claude-opus-5` `model_not_found`; exact-name retry `333e2661-73d9-4bed-b7f8-eaaca366281a` gave no verdict | **No completed Opus review** |
| Independent review (default model, not Opus) | Seven confirmed defects, all reproduced by the lead and fixed, with eleven new regression tests; round-2 real-machine runs reproduce every verdict | Completed, see [ledger](progress.md#independent-review) |

## Deliberately stricter than the legacy path

1. **Type-strict observations** — legacy `observed.get(k) == expected` accepted `1` for `true`; the migrated evaluator requires identical types and rejects duplicate keys and non-finite numbers.
2. **Positive PASS aggregation** — an overall PASS now requires every gate to be literally `PASS`; an unrecognised status is `INVALID_EVIDENCE` rather than falling through to a pass, so a receipt written by a bug cannot verify as valid.
3. **Ascend preflight** — the legacy Ascend manifest had no `configCheck`; a read-only `--check-config` now yields `BLOCKED_ENV` before any Ninja target starts.
4. **Artifact binding** — products are recorded and re-checked after the run; a changed `.so` is `STALE`, an unbound native artifact `INVALID_EVIDENCE`, closing the P0 gap going forward.
5. **NPU occupancy probe** — a read-only host `npu-smi` probe runs in preflight and again before dispatch; a busy card, an unreadable driver, or an answer the parser cannot read in full is `BLOCKED_ENV`. The legacy adapter had none.
6. **Receipt revalidation** — a PASS is current only while `verify` returns `valid: true`.

## Known limitations and what is not proven

- **No real NPU execution in any phase.** Both cards were held by other users and no device was assigned; the NPU gate is covered only by offline fixtures and negative tests. It is basic Ascend SIMD vector-add and does not prove HIVM CodeGen E2E.
- **Warm build trees.** P1 used pre-existing configured builds; cold-build timeouts (840 s / 1140 s inner) remain unvalidated assumptions.
- **Local trust only.** Receipts are per-user HMAC (`local-user-integrity`), not externally signed, and cannot resist a same-OS-user process.
- **Occupancy is a snapshot** and cannot see a job starting after the probe.
- **The plugin's full isolated-worktree loop was not demonstrated** — both exercises stopped at missing build configuration.
- **The `TaskCompleted` hook was never fired by a real task-completion event.** It is covered by offline tests and by direct invocation against real receipts — a round-2 plugin PASS (exit 0), a round-2 `BLOCKED_ENV` receipt (exit 2) and an event with no receipt block (exit 0, inactive) — but no `TaskCompleted` event occurred in any recorded session.
- Also unestablished: compiler semantic correctness, coordination with other users' jobs, and a completed Opus review. Equivalence does not extend to Harness scheduling, approvals or model routing.

## Reproduction

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install -e .
.venv/bin/agent-gates list --json
.venv/bin/agent-gates doctor --adapter ADAPTER --source ABS_SOURCE --env-file ABS_ENV --evidence-root DIR --json
.venv/bin/agent-gates run --adapter ADAPTER --source ABS_SOURCE --env-file ABS_ENV --evidence-root DIR --all --json
.venv/bin/agent-gates report --run-id RUN_ID --evidence-root DIR --json
.venv/bin/agent-gates verify --run-id RUN_ID --evidence-root DIR --json
python3 -m pytest -q tests/
claude plugin validate ./integrations/claude-code
```

For `tilelang-hivm-npu`, add `--device-id ASSIGNED_ID`; without assignment, and on a busy card, the correct result is `BLOCKED_ENV`.

## Using AgentGates from AscendNPU-IR and TileLang

AgentGates is not installed into either compiler checkout; it runs beside them and reads them as data. Paths below are placeholders — the runs in this report used the workspace checkouts `<ascend-checkout>` (HEAD `bfefe180`) and `<tilelang-checkout>` (HEAD `6fe6f09`).

```bash
cd <agent-gates-checkout> && python3.11 -m venv .venv && .venv/bin/python -m pip install -e .
export PATH="<agent-gates-checkout>/.venv/bin:$PATH"

# AscendNPU-IR (gate IDs: ascend-build, ascend-regression)
agent-gates doctor --adapter ascendnpu-ir --source <ascend-checkout> \
  --env-file <machine-dir>/ascend.env --json
agent-gates run --adapter ascendnpu-ir --source <ascend-checkout> \
  --env-file <machine-dir>/ascend.env --all --json

# TileLang (gate IDs: tilelang-hivm-build, tilelang-hivm-regression)
agent-gates run --adapter tilelang-hivm --source <tilelang-checkout> \
  --env-file <machine-dir>/tilelang-hivm.env.sh --all --json

# NPU sanity: only with a device an operator has assigned to you
agent-gates run --adapter tilelang-hivm-npu --source <tilelang-checkout> \
  --env-file <machine-dir>/tilelang-hivm.env.sh --device-id ASSIGNED_ID --json
```

Rules that follow from the implementation:

- `--source` must be the compiler checkout root; the checkout must already have a configured `build/build.ninja`. A fresh worktree without one fails at the build gate — that is a real result, not a setup bug.
- `--env-file` must be an absolute path to a machine file outside the checkout. The adapters parse it as `NAME=value` data; it is never sourced. Keep the real file out of Git.
- `--all` applies only to the chosen adapter and never pulls in the NPU gate.
- Evidence lands in `~/.local/state/agent-gates` unless `--evidence-root` names a private directory. Pass the returned `runId` to `agent-gates report` / `verify` to revalidate; a receipt turns `STALE` once the source, adapter bytes or a bound artifact change.
- From Claude Code, the same commands are available as `/agent-gates:doctor`, `/agent-gates:verify` and `/agent-gates:report` after loading the plugin with `claude --plugin-dir <agent-gates-checkout>/integrations/claude-code`.
