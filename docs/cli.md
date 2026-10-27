# CLI reference

Install with `python3.11 -m pip install -e .` in a dedicated virtual environment. Each subcommand accepts `--json` and `--evidence-root PATH`. JSON mode prints a structured result. Exit 0 indicates PASS/success, 1 indicates a recorded non-PASS result, and 2 indicates an invocation/environment error.

| Command | Purpose |
| --- | --- |
| `agent-gates list` | List bundled adapter IDs, manifest digests and gate IDs |
| `agent-gates doctor --adapter ID --source ROOT` | Run configuration checks without compiler build or NPU execution |
| `agent-gates run --adapter ID --source ROOT --gate GATE` | Run a gate and its dependencies, then publish a receipt |
| `agent-gates run --adapter ID --source ROOT --all` | Run all gates belonging to that adapter |
| `agent-gates report --run-id ID --json` | Read receipt and revalidate its current status |
| `agent-gates verify --run-id ID --json` | Verify receipt integrity, rules and current identities |

`--source` names the actual checkout root. A subdirectory of a Git checkout is rejected. A worktree must have its own configured build and dependencies; the tool does not silently use the original checkout's build.

`doctor` and `run` accept `--env-file /absolute/path/env.sh`. This maps to `ASCENDNPU_ENV_FILE` for AscendNPU-IR and `TILELANG_HIVM_ENV_FILE` for both TileLang adapters. Omit it only when the selected adapter's compatible local configuration discovery is appropriate. A migrated tool never automatically borrows another checkout's private machine file. Machine configuration files are data, not shell programs.

`run --gate` may be repeated. Select exactly one of `--gate` or `--all`. Gate dependencies are executed in topological order. Selecting `tilelang-hivm --all` never selects `tilelang-hivm-npu-sanity`.

For a device run, provide `--device-id ID` or explicitly set `TILELANG_HIVM_DEVICE_ID` to a device assigned to this task. This declaration is not an external scheduler or proof of allocation; the operator must have assignment. Do not choose a device only because it appears idle. The NPU Adapter then probes the card read-only through the Host's `npu-smi` before `doctor` reports ready and again immediately before the container command: a card holding any process the Gate did not start, or a driver that cannot be read, yields `BLOCKED_ENV` with the Adapter's one-line reason on the Gate (`configReason`). The two probes are classified identically even though they fail at different points — `CONFIG_CHECK_FAILED` when the check sees the busy card, `BLOCKED_EXIT_CODE` when it becomes busy after the check — because in both cases nothing was dispatched and no assertion ran. The AgentGates device lease is an additional, weaker guarantee — it constrains only AgentGates runs.

`report` and `verify` default to the most recent run when `--run-id` is omitted. Use the same evidence store on every command. Preserve stdout/stderr and process records in place; moving or editing them invalidates the receipt. Source, configuration, script or product changes may make a previously recorded PASS become STALE or INVALID_EVIDENCE. A successful `doctor` is an environment diagnosis, not a completed compiler gate receipt.

Example:

```bash
agent-gates run --adapter ascendnpu-ir --source /absolute/AscendNPU-IR --env-file /absolute/ascend.env --gate ascend-regression --evidence-root /absolute/evidence --json
agent-gates report --run-id RUN_ID --evidence-root /absolute/evidence --json
agent-gates verify --run-id RUN_ID --evidence-root /absolute/evidence --json
```

No command installs CANN/Conda, rebuilds Docker images, creates a compiler build tree, modifies compiler source, stops a shared container, or reuses an old PASS in place of a new run.
