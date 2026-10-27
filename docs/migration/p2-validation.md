# P2 validation: real Claude Code plugin session

A fresh, non-interactive Claude Code session was launched against the repository plugin with the installed CLI on `PATH`. This is a real session, not a simulated hook test. Raw transcript and logs stay in ignored `.evidence/p2-session/`.

## Session setup

```text
claude -p --plugin-dir <repo>/integrations/claude-code \
  --dangerously-skip-permissions --output-format stream-json --verbose \
  --include-hook-events --tools Bash,Read,Edit,Skill --strict-mcp-config \
  --name agent-gates-p2-validation
```

- Claude Code `2.1.296`, session `63e5fbf2-2815-4f03-995f-3b8d3bff4117`, exit code 0, 243.3 s, no timeout.
- Actual model reported by the session: `group/auto-deepseek-v4-1-flash`. Claude Code flagged it as `unrecognized_model`. This is **not** Opus; no claim of an Opus-run review is made.
- `PATH` was prefixed with the project virtualenv so `agent-gates` resolved to the editable install.

## What the session actually discovered

From the `system/init` event, the plugin was loaded as `agent-gates@inline` (path `integrations/claude-code`, version `0.1.0`) and contributed exactly three skills:

| Skill | `allowed-tools` | `disable-model-invocation` |
| --- | --- | --- |
| `agent-gates:doctor` | `Bash` | true |
| `agent-gates:verify` | `Bash, Read` | true |
| `agent-gates:report` | `Bash` | true |

The `TaskCompleted` hook was registered from `hooks/hooks.json`. No other AgentGates command or marketplace entry existed; `installed_plugins.json` lists only `pyright-lsp`, and `agent-gates@inline` appeared only under `pluginUsage`.

`claude plugin validate ./integrations/claude-code` passes with no warnings.

## Plugin-initiated runs

Source `<tilelang-checkout>`, HEAD `6fe6f09`, five pre-existing tracked modifications preserved. Evidence root `.evidence/p2-plugin-runs`. These are new runs, separate from P0 and P1.

| Step | Exit | Run ID | Status |
| --- | ---: | --- | --- |
| `doctor` | 0 | `1bab01b8adab80325a3becf9ee9ecf0d` | PASS |
| `run --all` | 0 | `74e428e9818aa75ad7fb1b95f20032bc` | PASS |
| `verify` | 0 | same | `valid: true`, PASS |
| `report` | 0 | same | recordedStatus PASS / current status PASS |

Both recipes' `--check-config` exited 0, resolving `container=<user>-tilelang-hivm`, `hivmTarget=Ascend950PR_9579`. `tilelang-hivm-build` passed on `exitCodeZero`; `tilelang-hivm-regression` passed with all five required observations, including `loadedNativeSha256MatchesHost` and `notAllSkipped`. `historicalPassReuse: false`, trust `local-user-integrity`.

This result is a duplicate measurement of the same source as the P1 TileLang run, performed through the plugin path. It is **not** an independent third validation of TileLang correctness, and it is not counted as one.

## Isolated worktree exercises

Two pre-created worktrees under `.evidence/worktrees/` were used. Both are detached at the original HEADs and visible through the existing container `/home` bind mount. Neither has its own configured build, and neither has initialized submodules.

### Ascend: `ascend-dev`

Change: `bishengir/test/Transforms/canonicalize-module.mlir:9`, `func.func @func_1(%arg0: memref<8xf32>){` → `…>) {`. Whitespace in test source only; no compiler behaviour touched.

| Step | Exit | Run ID | Status |
| --- | ---: | --- | --- |
| `run --all` | 1 | `6b14a3dc48720c0ef9738823554ae358` | FAIL |
| `verify` | 1 | same | `valid: false` |

- `ascend-build` FAIL: `NONZERO_EXIT`, `OBSERVATION_MISMATCH: exitCodeZero`; recipe exit 3 with `no build/build.ninja under …/worktrees/ascend-dev; initialize the build directory first`.
- `ascend-regression` NOT_RUN: `DEPENDENCY_NOT_PASSED`.
- `third-party/{llvm-project,torch-mlir,shmem}` are empty (submodules uninitialized).

This is a genuine FAIL of the worktree, not a PASS borrowed from the original checkout. The session did not redirect execution to `<ascend-checkout>` and did not claim that checkout's build attests this change. The original checkout's `ascend-build` PASS in P0/P1 covers different source bytes and a different source root.

### TileLang: `tilelang-dev`

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
- The `TaskCompleted` hook was exercised only by offline tests in `tests/test_plugin_receipts.py`, not by a real task-completion event in this session.
- A full isolated-worktree development loop, as described above.
