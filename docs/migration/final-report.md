# Migration final report

Repository `jingchangshi/AgentGates`, branch `migration/p0-p2`, execution date 2026-10-27. Prior commits: `cc573c7` (inventory), `3994c69` (audited baseline and extraction semantics).

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
| `ascendnpu-ir` | PASS 0.192 s `408db982743005773b0720bc32eca0c7` | PASS 56.473 s | `c1610b674026bc2156c1a3ccedb48ee0` | build PASS, regression PASS | `valid: true` |
| `tilelang-hivm` | PASS 0.197 s `e674eb1a85d46179c0d89ba22e6996cb` | PASS 20.247 s | `181cad5aa72c8893043f4a53f7b2e79b` | build PASS, regression PASS (14/14, 12 native libs) | `valid: true` |
| `tilelang-hivm-npu` | BLOCKED_ENV `3b5f392b58f3f5fde72b0b9ec3db9b4b` | BLOCKED_ENV | `414c2ca7aaf5880ec7e7711545e8475e` | no device assignment | `valid: false` |

An occupancy probe with explicit device ids (read-only host `npu-smi`; container `npu-smi` cannot load `libc_sec.so`) blocked both visible cards with `CONFIG_CHECK_FAILED`: device 0, run `a0c167250ab7c79a1e286ddf4e8cd7c2`, "already in use by 2 process(es) this Gate did not start"; device 1, run `a3428763b378f1b6a0297b4bf42c8c97`, 5 such processes. Both `verify` `valid: false`.

Offline suite: **441 passed, 88 subtests passed, exit 0**, logged to ignored `.evidence/final-pytest.log`; the earlier 439 figure was the same suite before two packaging tests were added. Isolated wheel proof (`.evidence/check_wheel.py`): PASS — the wheel contains every manifest-pinned file and the adapter deployment files, and no `env.sh`, `config.env`, `__pycache__` or machine configuration; it installs into a fresh venv away from the checkout where `agent-gates list --json` works and `engineering`, `anthropic`, `claude_code` are not importable. A fresh clone of the branch also passes the suite (441 tests) and `claude plugin validate`.

### P2 — Claude Code plugin

`claude plugin validate ./integrations/claude-code` passes with no warnings (re-run during this report). A fresh session (Claude Code 2.1.296, session `63e5fbf2-2815-4f03-995f-3b8d3bff4117`) exited 0 after 243.3 s. Its `init` event records the plugin as `agent-gates@inline` v0.1.0 contributing exactly three AgentGates skills — `agent-gates:doctor`, `agent-gates:verify`, `agent-gates:report`, all `disable-model-invocation: true` — alongside the harness's own global skills; the `TaskCompleted` hook is registered from `hooks/hooks.json`, though no `TaskCompleted` event occurred in this session. The session reported model `group/auto-deepseek-v4-1-flash`, flagged `unrecognized_model`; it was **not** Opus.

Plugin-initiated runs on the TileLang checkout: doctor PASS `1bab01b8adab80325a3becf9ee9ecf0d`; run PASS `74e428e9818aa75ad7fb1b95f20032bc`; verify `valid: true`; report PASS. This duplicates the P1 TileLang source, **not** a third independent validation.

Two isolated worktree exercises both FAILED:

| Worktree | Change | Run ID | Result |
| --- | --- | --- | --- |
| `ascend-dev` | Whitespace-only fix in `bishengir/test/Transforms/canonicalize-module.mlir:9` | `6b14a3dc48720c0ef9738823554ae358` | FAIL: build `NONZERO_EXIT`, recipe exit 3 (no `build/build.ninja`); regression `NOT_RUN` |
| `tilelang-dev` | One assertion at `testing/ascend/target/test_tilelang_ascend_target.py:30` | `b1e29f54789df1a328bb3fd891aaf8e3` | FAIL: build `NONZERO_EXIT`, recipe exit 2 (no build directory); regression `NOT_RUN` |

Both `verify` results were `valid: false`. Neither run was redirected to the original checkouts; both worktree diffs remain uncommitted.

## Architecture

`manifest.py` and `registry.py` load digest-pinned adapter data and reject schema, dependency, path-escape and pin violations. `engine.py` resolves configuration, takes leases, runs preflight, captures source identity before and after, and publishes evidence; `executor.py` dispatches fixed argv with `shell=False`. `evaluator.py` is type-strict; `source_identity.py` hashes HEAD plus tracked, dirty and untracked bytes; `evidence.py` signs and revalidates receipts without reusing a historical PASS. The plugin is CLI-only with an opt-in task-completion receipt check.

## Verification evidence

| Claim | Evidence | Status |
| --- | --- | --- |
| Original recipes still pass | P0 baseline, four gates | Verified |
| CLI reproduces gate outcomes | P1 runs, matching counts | Verified |
| No Harness/Claude runtime dependency | Wheel install + import assertions | Verified |
| Offline correctness | 441 tests, 88 subtests, exit 0 (`.evidence/final-pytest.log`) | Verified |
| Plugin loads and runs gates | Real session, three plugin skills, run PASS | Verified |
| Repository private | Auth `ls-remote` exit 0; nonexistent-repo control 128; anonymous API 404 vs public 200; 102-repo public listing omits it | Proven private |
| NPU gate on hardware | None in any phase | Not measured |
| Independent Opus review | `claude-opus-5` `model_not_found`; exact-name retry `333e2661-73d9-4bed-b7f8-eaaca366281a` gave no verdict | **No completed review** |

## Deliberately stricter than the legacy path

1. **Type-strict observations** — legacy `observed.get(k) == expected` accepted `1` for `true`; the migrated evaluator requires identical types and rejects duplicate keys and non-finite numbers.
2. **Ascend preflight** — the legacy Ascend manifest had no `configCheck`; a read-only `--check-config` now yields `BLOCKED_ENV` before any Ninja target starts.
3. **Artifact binding** — products are recorded and re-checked after the run; a changed `.so` is `STALE`, an unbound native artifact `INVALID_EVIDENCE`, closing the P0 gap going forward.
4. **NPU occupancy probe** — a read-only host `npu-smi` probe runs in preflight and again before dispatch; a busy or unreadable card is `BLOCKED_ENV`. The legacy adapter had none.
5. **Receipt revalidation** — a PASS is current only while `verify` returns `valid: true`.

## Known limitations and what is not proven

- **No real NPU execution in any phase.** Both cards were held by other users and no device was assigned; the NPU gate is covered only by offline fixtures and negative tests. It is basic Ascend SIMD vector-add and does not prove HIVM CodeGen E2E.
- **Warm build trees.** P1 used pre-existing configured builds; cold-build timeouts (840 s / 1140 s inner) remain unvalidated assumptions.
- **Local trust only.** Receipts are per-user HMAC (`local-user-integrity`), not externally signed, and cannot resist a same-OS-user process.
- **Occupancy is a snapshot** and cannot see a job starting after the probe.
- **The plugin's full isolated-worktree loop was not demonstrated** — both exercises stopped at missing build configuration.
- **The `TaskCompleted` hook was exercised only by offline tests**, not by a real task-completion event.
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
