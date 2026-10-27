# AgentGates

AgentGates is a standalone Python 3.11+ CLI for deterministic compiler build and test gates. Claude Code can plan and modify code; AgentGates executes digest-pinned recipes and records machine results, source identity, artifacts, logs, and verifiable local receipts. The CLI has no dependency on PiEngineeringHarness, Claude Code, or an agent runtime.

The initial adapters are `ascendnpu-ir`, `tilelang-hivm`, and `tilelang-hivm-npu`. The NPU sanity gate exercises basic Ascend SIMD behavior; it does not prove full HIVM CodeGen correctness.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/agent-gates list
.venv/bin/agent-gates doctor --adapter tilelang-hivm --source /absolute/compiler/checkout --env-file /absolute/machine/env.sh
.venv/bin/agent-gates run --adapter tilelang-hivm --source /absolute/compiler/checkout --env-file /absolute/machine/env.sh --all --json
.venv/bin/agent-gates report --run-id RUN_ID --json
.venv/bin/agent-gates verify --run-id RUN_ID --json
```

`--all` applies only to the chosen adapter. It does not implicitly invoke the NPU adapter. NPU runs require an explicitly assigned device and a successful availability check; a visible idle card alone does not authorize its use. Machine configuration is parsed as data, never sourced as arbitrary shell code. Existing CANN/Conda setup scripts are initialized by the reviewed adapter inside the existing container.

Evidence defaults to `~/.local/state/agent-gates`; use `--evidence-root /absolute/private/directory` on an individual command to select another store. Retain receipts and logs together. A report revalidates the recorded source and artifacts and may become `STALE` after a change. Historical PASS results are not reused for new executions. The local HMAC protects integrity within the stated trust boundary; it does not protect against malicious processes with the same user's privileges.

For the optional Claude Code plugin, load `integrations/claude-code` with the installed CLI on `PATH`:

```bash
PATH="$PWD/.venv/bin:$PATH" claude --plugin-dir "$PWD/integrations/claude-code"
```

Use `/agent-gates:doctor`, `/agent-gates:verify`, and `/agent-gates:report`. Plugin loading and hardware validation results are recorded separately from ordinary CLI runs.

See [architecture](docs/architecture.md), [CLI](docs/cli.md), [adapters](docs/adapters.md), [evidence](docs/evidence.md), [testing](docs/testing.md), and [Claude Code](docs/claude-code.md). Migration evidence and limitations are tracked in [the execution ledger](docs/migration/progress.md) and [final report](docs/migration/final-report.md).
