# P2 validation: real Claude Code plugin session

A fresh, non-interactive Claude Code session was launched against the repository plugin with the installed CLI on `PATH`. This is a real session, not a simulated hook test. Raw transcript and logs stay in ignored `.evidence/p2-session/` and `.evidence/p2-session-r2/`.

Two sessions were run. Round 1 (below, session `63e5fbf2…`) was taken before the independent review's fixes changed the adapter scripts; its receipts now verify as `STALE` with `ADAPTER_IDENTITY_CHANGED`, which is the correct outcome for evidence bound to different adapter bytes. Round 2 is the current evidence and is documented first.

## Round 2 (current evidence)

```text
claude -p --plugin-dir <repo>/integrations/claude-code \
  --dangerously-skip-permissions --output-format stream-json --verbose \
  --include-hook-events --tools Bash,Read,Edit,Skill --strict-mcp-config \
  --name agent-gates-p2-validation
```

- Claude Code `2.1.296`, session `e8fbc41e-0bbf-42b4-b522-15d8bf8e5381`, exit code 0, 201.6 s, no timeout.
- Model reported by the session: `group/auto-deepseek-v4-1-flash`, flagged `unrecognized_model`. **Not Opus**; no Opus claim is made anywhere in this document.
- `system/init` again loads `agent-gates@inline` v0.1.0 and lists exactly the three AgentGates skills `doctor`, `verify`, `report` among the harness's own skills.

| Step | Skill invoked | Exit | Run ID | Status |
| --- | --- | ---: | --- | --- |
| `doctor` | `agent-gates:doctor` | 0 | `9d59fc2378fcadaafb5d0f949f4498ba` | PASS (both `--check-config` exit 0) |
| `run --all` | `agent-gates:verify` | 0 | `a62906a815e77f9001c706c0c932b7dc` | PASS |
| `verify` | `agent-gates:verify` | 0 | same | `valid: true`, `failures: []` |
| `report` | `agent-gates:report` | 0 | same | recordedStatus PASS / current PASS |

Evidence root `.evidence/p2-plugin-runs-r2`. Source `<tilelang-checkout>`, HEAD `6fe6f09182c854e680a5488f4d8461fe5d5a76a6`, 1990 files, source hash `8f9ffe6d…` unchanged before and after, five pre-existing tracked modifications preserved. `tilelang-hivm-build` PASS (0.616 s, warm build tree) and `tilelang-hivm-regression` PASS (8.13 s) with all five required observations and 13 bound artifacts. `trust: local-user-integrity`, `historicalPassReuse: false`.

`doctor` publishes no receipt — it writes only its config-check logs — so there is nothing for `verify` to check on that run ID in either round; the PASS is the CLI's own exit status and log tree.

The 0.6 s build is an incremental `tilelang` target against a pre-existing warm build tree. It attests that the current tree compiles and links and that the suite passes; it is not a cold-build measurement.

### Round-2 worktree exercises

Same two detached worktrees, same uncommitted single-file edits, evidence root `.evidence/p2-worktree-runs-r2`.

| Worktree | Change | Run ID | Result |
| --- | --- | --- | --- |
| `ascend-dev` | `bishengir/test/Transforms/canonicalize-module.mlir:9`, whitespace only | `b2ef7ee46e15cfe90313f099de2a8afd` | FAIL: `ascend-build` `NONZERO_EXIT`, recipe exit 3 (`no build/build.ninja under …/ascend-dev; initialize the build directory first`); `ascend-regression` `NOT_RUN` `DEPENDENCY_NOT_PASSED` |
| `tilelang-dev` | one assertion at `testing/ascend/target/test_tilelang_ascend_target.py:30` | `62eec6254852af279ace6b5651e365e1` | FAIL: `tilelang-hivm-build` `NONZERO_EXIT`, recipe exit 2 (`no build directory under …/tilelang-dev; configure the build directory first`, then `container produced no build report`); regression `NOT_RUN` |

Both `verify` → exit 1, `valid: false`, the same three failures. Both worktrees have no `build/`, no `CMakeCache.txt`/`build.ninja`, and empty submodules (`third-party/llvm-project`, `third-party/torch-mlir`, `third-party/shmem`, `3rdparty/tvm`, `3rdparty/cutlass`). Neither run was redirected to the original checkouts, nothing was built elsewhere, no checkout was symlinked in, and both diffs remain uncommitted at their original HEADs.

## Round 1 (retained, now STALE)

```text
claude -p --plugin-dir <repo>/integrations/claude-code \
  --dangerously-skip-permissions --output-format stream-json --verbose \
  --include-hook-events --tools Bash,Read,Edit,Skill --strict-mcp-config \
  --name agent-gates-p2-validation
```

- Claude Code `2.1.296`, session `63e5fbf2-2815-4f03-995f-3b8d3bff4117`, exit code 0, 243.3 s, no timeout.
- Actual model reported by the session: `group/auto-deepseek-v4-1-flash`. Claude Code flagged it as `unrecognized_model`. This is **not** Opus; no claim of an Opus-run review is made.
- `PATH` was prefixed with the project virtualenv so `agent-gates` resolved to the editable install.

## Round-1 discovery

From that session's `system/init` event, the plugin was loaded as `agent-gates@inline` (path `integrations/claude-code`, version `0.1.0`) and contributed exactly three skills:

| Skill | `allowed-tools` | `disable-model-invocation` |
| --- | --- | --- |
| `agent-gates:doctor` | `Bash` | true |
| `agent-gates:verify` | `Bash, Read` | true |
| `agent-gates:report` | `Bash` | true |

`hooks/hooks.json` declares the `TaskCompleted` command hook (`python3 -I "${CLAUDE_PLUGIN_ROOT}/hooks/task_completed.py"`, 75 s), and the session read that file; the session's `init` event carries no hook list, so registration was not independently observed. No `TaskCompleted` event occurred, so the hook was never invoked in either session. No other AgentGates command or marketplace entry existed; `installed_plugins.json` lists only `pyright-lsp`, and `agent-gates@inline` appeared only under `pluginUsage`.

`claude plugin validate ./integrations/claude-code` passes with no warnings.

## Round-1 plugin-initiated runs (STALE)

Source `<tilelang-checkout>`, HEAD `6fe6f09`, five pre-existing tracked modifications preserved. Evidence root `.evidence/p2-plugin-runs`. The round-1 `run` ID `74e428e9…` now verifies as `STALE` / `ADAPTER_IDENTITY_CHANGED`; the round-1 `doctor` ID `1bab01b8…` publishes no receipt, so `verify` reports `INVALID_EVIDENCE` for it — the same "no receipt" answer that the round-2 doctor ID gives. The round-2 table above is the current evidence for the same steps.

| Step | Exit | Run ID | Status |
| --- | ---: | --- | --- |
| `doctor` | 0 | `1bab01b8adab80325a3becf9ee9ecf0d` | PASS |
| `run --all` | 0 | `74e428e9818aa75ad7fb1b95f20032bc` | PASS |
| `verify` | 0 | same | `valid: true`, PASS |
| `report` | 0 | same | recordedStatus PASS / current status PASS |

Both recipes' `--check-config` exited 0, resolving `container=<user>-tilelang-hivm`, `hivmTarget=Ascend950PR_9579`. `tilelang-hivm-build` passed on `exitCodeZero`; `tilelang-hivm-regression` passed with all five required observations, including `loadedNativeSha256MatchesHost` and `notAllSkipped`. `historicalPassReuse: false`, trust `local-user-integrity`.

This result is a duplicate measurement of the same source as the P1 TileLang run, performed through the plugin path. It is **not** an independent third validation of TileLang correctness, and it is not counted as one.

## Round-1 isolated worktree exercises (STALE)

Two pre-created worktrees under `.evidence/worktrees/` were used. Both are detached at the original HEADs and visible through the existing container `/home` bind mount. Neither has its own configured build, and neither has initialized submodules.

Both round-1 run IDs below now verify as `STALE` / `ADAPTER_IDENTITY_CHANGED`, because the review fixes changed the adapter scripts after they were recorded. The round-2 runs above reproduce every verdict against the current adapter bytes.

### Ascend: `ascend-dev` (round 1)

Change: `bishengir/test/Transforms/canonicalize-module.mlir:9`, `func.func @func_1(%arg0: memref<8xf32>){` → `…>) {`. Whitespace in test source only; no compiler behaviour touched.

| Step | Exit | Run ID | Status |
| --- | ---: | --- | --- |
| `run --all` | 1 | `6b14a3dc48720c0ef9738823554ae358` | FAIL |
| `verify` | 1 | same | `valid: false` |

- `ascend-build` FAIL: `NONZERO_EXIT`, `OBSERVATION_MISMATCH: exitCodeZero`; recipe exit 3 with `no build/build.ninja under …/worktrees/ascend-dev; initialize the build directory first`.
- `ascend-regression` NOT_RUN: `DEPENDENCY_NOT_PASSED`.
- `third-party/{llvm-project,torch-mlir,shmem}` are empty (submodules uninitialized).

This is a genuine FAIL of the worktree, not a PASS borrowed from the original checkout. The session did not redirect execution to `<ascend-checkout>` and did not claim that checkout's build attests this change. The original checkout's `ascend-build` PASS in P0/P1 covers different source bytes and a different source root.

### TileLang: `tilelang-dev` (round 1)

Change: `testing/ascend/target/test_tilelang_ascend_target.py:30`, added `assert normalize_ascend_target({"kind": "c"}) is None` next to the existing non-string normalization checks.

| Step | Exit | Run ID | Status |
| --- | ---: | --- | --- |
| `run --all` | 1 | `b1e29f54789df1a328bb3fd891aaf8e3` | FAIL |
| `verify` | 1 | same | `valid: false` |

- `tilelang-hivm-build` FAIL: `NONZERO_EXIT`, `OBSERVATION_MISMATCH: exitCodeZero`; recipe exit 2 with `no build directory under …/worktrees/tilelang-dev; configure the build directory first`, then `container produced no build report`.
- `tilelang-hivm-regression` NOT_RUN: `DEPENDENCY_NOT_PASSED`.
- `3rdparty/tvm` and `3rdparty/cutlass` are empty.

No unrelated source was built and no checkout was symlinked in.

### Why these are blocked rather than completed

The worktrees are correct isolated copies of the source, but an isolated worktree needs its own configured build and initialized submodules to produce gate evidence. Neither exists, and the goal forbids substituting the original checkout's build. Creating them would mean a full cold build (and submodule initialization) in a directory the container mounts, which is outside what was authorized for this migration and would not be evidence about the change anyway. The honest outcome is the FAIL recorded above.

Consequence for the P2 claim: **the plugin's execution mode, receipt reading and verification path are validated end to end on a real source tree; the two development-loop exercises are validated only up to the point where the worktree's missing build configuration stops them.** A complete isolated-worktree development loop was not demonstrated.

## Integrity of the session

- The two worktree diffs are retained uncommitted; `git status` in each shows exactly the intended single-file modification.
- AgentGates source was not edited by the session; no commits were created.
- No dependencies installed, no containers created or stopped, no NPU or hardware calls, no subagents spawned.
- The session was restricted to `Bash, Read, Edit, Skill` with `--strict-mcp-config`.

## What remains unproven

- No Opus-model independent review. The earlier Opus reviewer failed with `model_not_found` for `claude-opus-5`, and the exact-name retry session `333e2661-73d9-4bed-b7f8-eaaca366281a` produced no completed verdict.
- The `TaskCompleted` hook was never fired by a real task-completion event: neither session completed a task, so no `TaskCompleted` event occurred in either round. It is covered by `tests/test_plugin_receipts.py` and was invoked directly, with `PATH` pointing at the installed CLI, on three events: a round-2 plugin PASS receipt (exit 0), a round-2 `BLOCKED_ENV` receipt (exit 2, "agent-gates rejected the requested receipt"), and an event with no receipt block (exit 0, hook inactive).
- A full isolated-worktree development loop, as described above.

## Observed harness behaviour worth recording

`disable-model-invocation: true` is declared on all three skills, yet both sessions launched them through the `Skill` tool without any bypass. The flag is not being enforced for these plugin skills in Claude Code 2.1.296. This is a property of the harness, not of the plugin; the skills remain CLI-only in the sense that they take explicit `--adapter`/`--source` arguments and never guess them.
