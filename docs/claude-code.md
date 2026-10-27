# Claude Code integration

The repository plugin lives in `integrations/claude-code`. Install AgentGates in the Python environment used by Claude Code, ensure `agent-gates` is on PATH, then start a fresh session with an absolute plugin directory:

```bash
claude --plugin-dir /absolute/AgentGates/integrations/claude-code
```

This does not modify global settings. The plugin adds three manually invoked skills:

```text
/agent-gates:doctor --adapter tilelang-hivm --source /absolute/source --env-file /absolute/machine.env
/agent-gates:verify --adapter tilelang-hivm --source /absolute/source --env-file /absolute/machine.env --all
/agent-gates:verify --run-id RUN_ID
/agent-gates:report --run-id RUN_ID
```

Doctor checks configuration and report reads an explicit run and its current verification result. Verify has two modes: explicit adapter/source/gate selectors execute the fixed gates, read their actual receipt, then verify it against the final source and artifacts; an explicit run ID verifies an existing receipt without execution. Execution mode identifies the selected source HEAD and modifications first. A missing worktree build remains a real blocker or failure; the skill never redirects execution to another checkout. These skills wrap the standalone CLI and do not provision environments or retry failed gates. NPU execution additionally requires the explicit `tilelang-hivm-npu` adapter and a task-assigned `--device-id`; no card is selected automatically. Report and receipt-only verify require an explicit run ID to avoid attaching an unrelated latest result. Optional `--evidence-root /absolute/private/store` selects a nondefault evidence store.

## Optional task completion policy

The bundled `TaskCompleted` command hook is inactive for ordinary tasks. It enforces receipt requirements only when the task subject or description contains exactly one explicit JSON block of this form:

````text
```agent-gates-receipt
{
  "adapter": "tilelang-hivm",
  "source": "/absolute/source",
  "gates": ["tilelang-hivm-build", "tilelang-hivm-regression"],
  "runId": "0123456789abcdef0123456789abcdef"
}
```
````

Replace the example ID with the actual run ID. Optional `evidenceRoot` names an absolute evidence-store path. The hook calls only `agent-gates verify --run-id ... --json` and `agent-gates report --run-id ... --json`. It requires a current valid PASS, matching adapter/source/run identity, and a PASS for every required gate. Missing, failed, blocked, stale, malformed and invalid evidence block completion with exit 2 and an explanation. The hook never executes `run`, chooses hardware, retries a gate or edits receipts. Set the task's run ID after a separately authorized gate run.

The hook trusts the local installed CLI and its local-user evidence protections; another process with the same user's privileges can modify the tool or its attestation material. This is a local integrity boundary, not isolation from the account owner.

The plugin is optional: use the standalone CLI without it when no task hook is wanted. Do not add receipt blocks to unrelated tasks. Removing the plugin from the session removes its completion policy without changing evidence.

## Validation

Static schema validation uses the actual installed CLI:

```bash
claude plugin validate /absolute/AgentGates/integrations/claude-code --json
python3 -m unittest discover -s tests -p 'test_plugin*.py' -q
```

Static manifest validation does not prove a fresh Claude session loaded the skills or executed the hook. Fresh-session loading and actual skill invocation must be recorded separately by the integration run. The offline hook tests simulate CLI results and do not establish compiler or real-device PASS.
