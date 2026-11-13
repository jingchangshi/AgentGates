"""The AgentGates MCP server: one stdio process exposing the same API as the CLI.

Design rules this module exists to satisfy:

- It shares the Python API. Every tool is a thin wrapper over config.py,
  execution.py, verify_requirements.py and the EvidenceStore, so the CLI and
  the MCP surface can never disagree about a verdict.
- It holds no build process. submit starts a systemd user worker and returns;
  the server may be restarted at any moment without affecting a running gate.
- Tool inputs are bounded to ids and small scalars (C2). There is no parameter
  for a shell command, a container, an interpreter, an env file or a source
  directory, because there is nowhere for one to be accepted.
- A failing tool raises; it never returns a success-shaped object that a model
  could mistake for a result.
"""
from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from . import api_types as api
from .config import ConfigError, load
from .docker_quarantine import QUARANTINED_GATES, is_quarantined_recipe
from .evidence import EvidenceStore
from .execution import MAX_WAIT_SECONDS, ExecutionManager
from .verify_requirements import verify_requirements

SERVER_NAME = "agentgates"


def _manager() -> ExecutionManager:
    return ExecutionManager(load())


def _gate_info(gate_id, definition):
    return api.GateInfo(id=gate_id,
                        recipe_id=definition["recipeId"],
                        depends_on=list(definition.get("dependsOn") or []),
                        checks=dict(definition.get("observationChecks") or {}))


def _allowed_manifest_gates(workspace, manifest):
    blocked = QUARANTINED_GATES.get(manifest.id, ())
    allowed = []
    for gate_id, definition in manifest.gates.items():
        if gate_id in blocked:
            continue
        if workspace.allowed_gates is not None and gate_id not in workspace.allowed_gates:
            continue
        recipe = manifest.resolve_recipe(definition["recipeId"], workspace.source_root)
        if is_quarantined_recipe(manifest.id, recipe.id, recipe.argv):
            continue
        allowed.append((gate_id, definition))
    return allowed


def _status_enum(value):
    return (api.GateStatus(value) if value in api.GateStatus._value2member_map_
            else api.GateStatus.INVALID_EVIDENCE)


# ---- input models ---------------------------------------------------------
#
# Each model is the COMPLETE input surface of its tool. extra="forbid" means an
# attempt to pass an undeclared machine parameter is rejected rather than
# ignored, which turns a design violation into a visible error.


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceInput(_Input):
    workspace: str | None = Field(
        None, description="Registered workspace id; omit to list every workspace.")


class GatesInput(_Input):
    workspace: str = Field(description="Registered workspace id.")
    adapter: str = Field(description="Adapter id available to that workspace.")


class DoctorInput(_Input):
    workspace: str = Field(description="Registered workspace id.")
    adapter: str = Field(description="Adapter id available to that workspace.")
    # Optional and additive: omitting it diagnoses every route exactly as before.
    # A gate id is not a machine parameter - it names a fixpoint of the adapter's
    # own dependency graph, so it cannot steer a command line.
    gates: list[str] | None = Field(
        None, description="Gate ids to diagnose (dependencies included); omit to probe every route.")


class SubmitInput(_Input):
    workspace: str = Field(description="Registered workspace id.")
    adapter: str = Field(description="Adapter id available to that workspace.")
    gates: list[str] = Field(description="Gate ids; dependencies are added automatically.")
    request_id: str | None = Field(
        None, description="Idempotency key: the same id with the same request returns the original execution.")


class ExecutionInput(_Input):
    execution_id: str = Field(description="Execution id returned by submit.")


class WaitInput(_Input):
    execution_id: str = Field(description="Execution id returned by submit.")
    timeout_seconds: float = Field(
        15.0, ge=0, le=MAX_WAIT_SECONDS,
        description="Bounded wait; the server clamps this and never blocks until a build finishes.")


class RunInput(_Input):
    run_id: str = Field(description="Run id of a published receipt.")


class RequirementsInput(_Input):
    workspace: str = Field(description="Registered workspace id.")
    adapter: str = Field(description="Adapter id the run must be bound to.")
    required_gates: list[str] = Field(description="Every gate that must have passed.")
    run_id: str = Field(description="Explicit run id; there is no latest shortcut.")


# ---- output models --------------------------------------------------------


class WorkspaceList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspaces: list[api.WorkspaceInfo]


class GateList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace: str
    adapter: str
    gates: list[api.GateInfo]


class ReportView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    status: str
    adapter: str
    source_root: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    gates: list[api.GateResult]
    failures: list[str]
    verification: api.EvidenceVerification | None = None


def _receipt_report(run_id, store, registry):
    """Read one published receipt and re-verify it without recomputing a verdict."""
    receipt = store.read(run_id)
    verification = store.verify(run_id, registry=registry)
    gates = []
    for gate in receipt.get("gates", []):
        if isinstance(gate, dict):
            gates.append(api.GateResult(
                gate_id=gate.get("gateId"),
                status=_status_enum(gate.get("status")),
                failures=list(gate.get("failures") or []),
                recipe_id=gate.get("recipeId")))
    return ReportView(
        run_id=run_id,
        status=str(verification.get("status")),
        adapter=str((receipt.get("adapter") or {}).get("id", "")),
        source_root=(receipt.get("source") or {}).get("root"),
        started_at=receipt.get("startedAt"),
        finished_at=receipt.get("finishedAt"),
        gates=gates,
        failures=list(verification.get("failures") or []),
        verification=api.EvidenceVerification(
            run_id=run_id,
            status=_status_enum(verification.get("status")),
            valid=bool(verification.get("valid")),
            failures=list(verification.get("failures") or [])))


def _reject_unknown(received: dict, declared: set) -> None:
    """Refuse undeclared arguments instead of silently dropping them.

    The MCP server SDK validates the DECLARED parameters and ignores anything
    else, which would let a caller pass a machine parameter such as a shell
    command and receive a normal-looking success while the field vanished. C2
    requires that such a call FAIL, so the caller learns the parameter does
    not exist rather than believing it was honoured.
    """
    unknown = sorted(set(received) - declared)
    if unknown:
        raise ValueError(
            "undeclared argument(s): " + ", ".join(unknown)
            + "; AgentGates accepts only ids and bounded scalars, and every "
              "machine parameter is resolved from local configuration")


def build_server():
    """Construct the MCP server and register every tool."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name=SERVER_NAME,
        title="AgentGates",
        version="2.0.0",
        instructions=(
            "Deterministic compiler engineering gates. Call list_workspaces, then "
            "list_gates for one adapter, then submit and poll status/wait. A gate "
            "verdict is PASS/FAIL/BLOCKED_ENV/NOT_RUN/STALE/INVALID_EVIDENCE; an "
            "execution being FINISHED does NOT mean its gates passed. "
            "verify_requirements answers whether one explicit run id satisfies "
            "required gates for the workspace current source state."),
    )

    # Each handler below is registered with the SDK, which validates the
    # declared parameters. The undeclared-argument check happens in
    # _strict(), which every tool is wrapped with before registration.
    @server.tool(name="list_workspaces",
                 description="List registered workspaces and the adapters each may use.")
    async def list_workspaces(workspace: str | None = None) -> WorkspaceList:
        payload = WorkspaceInput(workspace=workspace)
        manager = _manager()
        config = manager.config
        wanted = [payload.workspace] if payload.workspace else sorted(config.workspaces)
        infos = []
        for workspace_id in wanted:
            workspace = config.workspace(workspace_id)
            adapters = []
            for adapter_id in workspace.adapters:
                manifest = manager.registry.get(adapter_id)
                adapters.append(api.AdapterInfo(
                    id=manifest.id,
                    manifest_sha256=manifest.manifest_hash,
                    gates=[_gate_info(gate_id, definition)
                           for gate_id, definition in _allowed_manifest_gates(workspace, manifest)]))
            infos.append(api.WorkspaceInfo(
                id=workspace.id,
                source_root=str(workspace.source_root),
                adapters=adapters,
                allowed_gates=(sorted(workspace.allowed_gates)
                               if workspace.allowed_gates else None)))
        return WorkspaceList(workspaces=infos)

    @server.tool(name="list_gates",
                 description="List the gates one workspace adapter can run.")
    async def list_gates(workspace: str, adapter: str) -> GateList:
        payload = GatesInput(workspace=workspace, adapter=adapter)
        manager = _manager()
        workspace = manager.config.workspace(payload.workspace)
        if payload.adapter not in workspace.adapters:
            raise ValueError("adapter " + payload.adapter
                             + " is not registered for workspace " + payload.workspace)
        manifest = manager.registry.get(payload.adapter)
        return GateList(workspace=payload.workspace, adapter=payload.adapter,
                        gates=[_gate_info(gate_id, definition)
                               for gate_id, definition in _allowed_manifest_gates(workspace, manifest)])

    @server.tool(name="doctor",
                 description="Probe the machine configuration for one workspace and "
                             "adapter. Runs no compiler test and never satisfies a requirement. "
                             "Name gates to diagnose one route only; the reply states its scope.")
    async def doctor(workspace: str, adapter: str,
                     gates: list[str] | None = None) -> api.DoctorReport:
        payload = DoctorInput(workspace=workspace, adapter=adapter, gates=gates)
        from .engine import Engine, config_reason
        manager = _manager()
        workspace = manager.config.workspace(payload.workspace)
        if payload.adapter not in workspace.adapters:
            raise ValueError("adapter " + payload.adapter
                             + " is not registered for workspace " + payload.workspace)
        engine = Engine(registry=manager.registry,
                        store=EvidenceStore(manager.config.evidence_root),
                        lock_root=manager.config.lock_root)
        adapter = manager.config.adapter(payload.adapter)
        manifest = manager.registry.get(payload.adapter)
        selected_gates = payload.gates
        if selected_gates is None:
            allowed_pairs = _allowed_manifest_gates(workspace, manifest)
            if allowed_pairs and len(allowed_pairs) < len(manifest.gates):
                selected_gates = [gid for gid, _ in allowed_pairs]
        elif workspace.allowed_gates is not None:
            denied = [g for g in selected_gates if g not in workspace.allowed_gates]
            if denied:
                raise ValueError(
                    "gate(s) not permitted for workspace " + repr(payload.workspace)
                    + ": " + ", ".join(denied)
                )
        device_id = manager._device_for(payload.adapter, None)
        import os
        had_smi = "AGENT_GATES_SMI_COMMAND" in os.environ
        prev_smi = os.environ.get("AGENT_GATES_SMI_COMMAND")
        if manager.config.devices.smi_command:
            os.environ["AGENT_GATES_SMI_COMMAND"] = manager.config.devices.smi_command
        try:
            result = engine.doctor(
                payload.adapter, workspace.source_root,
                selected=selected_gates, env_file=adapter.env_file,
                device_id=device_id,
                require_exclusive_device=manager.config.devices.require_exclusive)
        finally:
            if had_smi:
                os.environ["AGENT_GATES_SMI_COMMAND"] = prev_smi
            else:
                os.environ.pop("AGENT_GATES_SMI_COMMAND", None)
        checks = []
        for item in result.get("checks", []):
            ok = (item.get("exitCode") == 0 and not item.get("timedOut")
                  and not item.get("interrupted"))
            reason = None if ok else (config_reason(item) or None)
            checks.append(api.DoctorCheck(
                recipe_id=item.get("recipeId", ""),
                ok=ok,
                exit_code=item.get("exitCode"),
                reason=reason))
        return api.DoctorReport(workspace=payload.workspace, adapter=payload.adapter,
                                status=_status_enum(result.get("status")),
                                checks=checks,
                                failures=list(result.get("failures") or []),
                                all_routes=bool(result.get("allRoutes", True)),
                                selected_gates=list(result.get("selectedGates") or []),
                                checked_recipes=list(result.get("checkedRecipes") or []),
                                deduplicated_recipes=dict(result.get("deduplicatedRecipes") or {}))

    @server.tool(name="submit",
                 description="Submit gates for asynchronous execution. Returns an execution "
                             "id immediately; the gates run in an independent systemd worker.")
    async def submit(workspace: str, adapter: str, gates: list[str],
                     request_id: str | None = None) -> api.ExecutionStatus:
        payload = SubmitInput(workspace=workspace, adapter=adapter,
                              gates=gates, request_id=request_id)
        manager = _manager()
        return manager.submit(api.ExecutionRequest(
            workspace=payload.workspace, adapter=payload.adapter,
            gates=payload.gates, request_id=payload.request_id))

    @server.tool(name="status", description="Read one execution current state.")
    async def status(execution_id: str) -> api.ExecutionStatus:
        payload = ExecutionInput(execution_id=execution_id)
        return _manager().status(payload.execution_id)

    @server.tool(name="wait",
                 description="Bounded wait for one execution. A timed-out wait is a normal "
                             "answer; call status or wait again.")
    async def wait(execution_id: str, timeout_seconds: float = 15.0) -> api.WaitResult:
        payload = WaitInput(execution_id=execution_id, timeout_seconds=timeout_seconds)
        return _manager().wait(payload.execution_id, payload.timeout_seconds)

    @server.tool(name="cancel",
                 description="Cancel an execution and stop its worker process tree. "
                             "Idempotent; a cancelled execution never reports PASS.")
    async def cancel(execution_id: str) -> api.ExecutionStatus:
        payload = ExecutionInput(execution_id=execution_id)
        return _manager().cancel(payload.execution_id)

    @server.tool(name="report",
                 description="Read one published receipt with its current verification. "
                             "Requires an explicit run id.")
    async def report(run_id: str) -> ReportView:
        payload = RunInput(run_id=run_id)
        manager = _manager()
        return _receipt_report(payload.run_id,
                               EvidenceStore(manager.config.evidence_root),
                               manager.registry)

    @server.tool(name="verify",
                 description="Re-verify one published receipt against this machine.")
    async def verify(run_id: str) -> api.EvidenceVerification:
        payload = RunInput(run_id=run_id)
        manager = _manager()
        store = EvidenceStore(manager.config.evidence_root)
        result = store.verify(payload.run_id, registry=manager.registry)
        return api.EvidenceVerification(
            run_id=payload.run_id,
            status=_status_enum(result.get("status")),
            valid=bool(result.get("valid")),
            failures=list(result.get("failures") or []))

    @server.tool(name="verify_requirements",
                 description="Decide whether one explicit run id satisfies a set of required "
                             "gates for the current source state. This is what a completion gate calls.")
    async def verify_requirements_tool(workspace: str, adapter: str,
                                        required_gates: list[str], run_id: str,
                                        ) -> api.RequirementVerificationResult:
        payload = RequirementsInput(workspace=workspace, adapter=adapter,
                                    required_gates=required_gates, run_id=run_id)
        manager = _manager()
        requirement = api.VerificationRequirement(
            workspace=payload.workspace, adapter=payload.adapter,
            required_gates=payload.required_gates, run_id=payload.run_id)
        return verify_requirements(requirement, config=manager.config,
                                   registry=manager.registry, store=manager.evidence)

    _enforce_strict_arguments(server)
    return server


def _enforce_strict_arguments(server) -> None:
    """Make every tool refuse arguments it did not declare.

    The SDK validates the declared parameters and silently ignores the rest.
    Silently ignoring is the wrong default for this server specifically: a
    caller that passes a machine parameter (a shell command, a container name,
    an interpreter) would receive a normal-looking success and reasonably
    believe the parameter was honoured. Rejecting turns a design violation into
    the error the objective requires.

    The check is installed on the ToolManager, because the Tool records
    themselves are frozen models and cannot be patched per tool.
    """
    manager = server._tool_manager
    allowed_by_name = {
        tool.name: set(tool.fn_metadata.arg_model.model_fields)
        for tool in manager.list_tools()
    }
    original = manager.call_tool

    async def call_tool(name, arguments, context, convert_result=False):
        allowed = allowed_by_name.get(name)
        if allowed is not None:
            _reject_unknown(arguments or {}, allowed)
        return await original(name, arguments, context, convert_result=convert_result)

    manager.call_tool = call_tool


def main(argv=None) -> int:
    """Run the stdio server. This process never holds a build."""
    import sys
    try:
        server = build_server()
    except (ConfigError, OSError) as exc:
        print("agent-gates mcp: cannot start: " + str(exc), file=sys.stderr)
        return 2
    asyncio.run(server.run_stdio_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
