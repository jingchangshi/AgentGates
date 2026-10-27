---
name: verify
description: Run explicitly selected AgentGates gates and verify their current receipt, or verify an existing receipt by run ID.
disable-model-invocation: true
allowed-tools: Bash, Read
argument-hint: --adapter ADAPTER --source ABSOLUTE_SOURCE (--gate GATE | --all) [--env-file FILE] [--device-id ID] [--evidence-root PATH] OR --run-id RUN_ID [--evidence-root PATH]
---

Use the installed `agent-gates` CLI. Treat arguments as data: quote each value and never evaluate embedded shell expressions. Choose exactly one of the following explicit modes. Do not silently select a source, adapter, gate, device, or latest receipt.

Execution mode requires `--adapter`, an absolute `--source`, and either repeated `--gate` selections or `--all`. Honor explicit `--env-file` and `--evidence-root` values. First identify the selected source/version with `git -C SOURCE rev-parse HEAD` and `git -C SOURCE status --short` (report a non-Git source honestly), then run `agent-gates run $ARGUMENTS --json`. This invocation authorizes only the selected adapter's fixed gates. Use the actual run JSON even when the CLI exits nonzero. Read the returned `receiptPath`, check its adapter/source/gate identities, and take its actual `runId`. Run `agent-gates verify --run-id RUN_ID --json`, preserving the same explicit evidence root. Receipt verification checks the final source and current artifacts. Do not redirect a missing worktree build to another checkout, provision dependencies, retry failures, edit evidence, or claim a different checkout's build covers this source.

The `tilelang-hivm-npu` adapter requires both explicit selection of that adapter and an explicitly task-assigned `--device-id`. Never infer allocation from an idle card or choose a fallback device. Base adapter `--all` stays within that adapter and never adds the NPU adapter. If allocation is absent, report the blocker without running hardware.

Receipt mode requires `--run-id` and permits `--evidence-root`; run only `agent-gates verify $ARGUMENTS --json`. It does not run gates. Do not mix a run ID with execution-mode selectors.

Report exact command exit codes, actual run ID, recorded gate statuses, and verification `status`, `valid`, and `failures`. A recorded PASS is current only when verification returns `valid: true` and `status: PASS`. Missing, stale, failed, blocked, and invalid evidence do not satisfy a task's gate requirement. Preserve the real failure details and stop without automatic repair or retry.
