"""The agent-gates command line; execution uses installed fixed recipes only."""
import argparse
import json
from pathlib import Path
import sys
from .engine import Engine
from .evidence import EvidenceStore
from .registry import Registry
from . import dynamic_target
from .api_types import APIError, ExecutionRequest, VerificationRequirement, jsonable
from .config import ConfigError
from .docker_quarantine import QUARANTINED_ADAPTERS, QUARANTINED_GATES
from .execution import ExecutionManager
from .verify_requirements import verify_requirements


def parser():
    root = argparse.ArgumentParser(prog="agent-gates")
    subcommands = root.add_subparsers(dest="command", required=True)
    for name in ("list", "doctor", "run", "report", "verify",
                 "submit", "status", "wait", "cancel", "verify-requirements"):
        command = subcommands.add_parser(name)
        command.add_argument("--json", action="store_true", help="print structured JSON")
        command.add_argument("--evidence-root", type=Path, help="local private evidence directory")
        if name in {"doctor", "run"}:
            command.add_argument("--adapter", required=True)
            command.add_argument("--source", required=True, type=Path)
            command.add_argument("--env-file", type=Path)
            command.add_argument("--device-id", help="explicitly authorize this NPU device for the run")
            command.add_argument("--target", help="typed workload target for a dynamic-target route")
            command.add_argument("--candidate", help="source-relative candidate file for the target")
        if name == "run":
            selection = command.add_mutually_exclusive_group(required=True)
            selection.add_argument("--gate", action="append", help="gate ID; dependencies run first")
            selection.add_argument("--all", action="store_true", help="run every gate in this adapter")
        if name == "doctor":
            # Optional, and named exactly as `run` names it: the same operator
            # scoping a diagnosis and a run should not learn two spellings. No
            # `--all` counterpart, because every route is already the default.
            command.add_argument("--gate", action="append",
                                 help="gate ID; diagnose only this route and its dependencies")
        if name in {"report", "verify"}:
            command.add_argument("--run-id", help="run ID (default: most recent)")
        if name == "submit":
            command.add_argument("--workspace", required=True)
            command.add_argument("--adapter", required=True)
            command.add_argument("--gate", action="append", required=True)
            command.add_argument("--request-id")
            command.add_argument("--target")
            command.add_argument("--candidate")
            command.add_argument(
                "--source",
                type=Path,
                help="linked git worktree root belonging to --workspace; omit for the registered root",
            )
            command.add_argument(
                "--device-id",
                help="authorized NPU card for this execution; omit for the configured default")
        if name in {"status", "wait", "cancel"}:
            command.add_argument("--execution-id", required=True)
        if name == "wait":
            command.add_argument("--timeout", type=float, default=15.0)
        if name == "verify-requirements":
            command.add_argument("--workspace", required=True)
            command.add_argument("--adapter", required=True)
            command.add_argument("--gate", action="append", required=True)
            command.add_argument("--run-id", required=True)
            command.add_argument("--target")
            command.add_argument("--candidate")
            command.add_argument(
                "--source",
                type=Path,
                help="linked git worktree root belonging to --workspace; omit for the registered root",
            )
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    registry = Registry()
    store = EvidenceStore(args.evidence_root)
    engine = Engine(registry=registry, store=store)
    # Asynchronous verbs resolve machine parameters from the local
    # configuration and delegate to the worker; they must not construct an
    # Engine, which would run gates in this process.
    if args.command in {"submit", "status", "wait", "cancel", "verify-requirements"}:
        return _async_command(args)
    try:
        if args.command == "list":
            result = {
                "adapters": [
                    {
                        "id": manifest.id,
                        "manifestSha256": manifest.manifest_hash,
                        "gates": list(manifest.gates),
                        "quarantined": manifest.id in QUARANTINED_ADAPTERS,
                        "quarantinedGates": [
                            g for g in manifest.gates
                            if g in QUARANTINED_GATES.get(manifest.id, ())
                        ],
                    }
                    for manifest in registry.list()
                ]
            }
        elif args.command in {"doctor", "run"}:
            options = {"env_file": args.env_file, "device_id": args.device_id,
                       "target": args.target, "candidate": args.candidate}
            if args.command == "doctor":
                result = engine.doctor(args.adapter, args.source, selected=args.gate, **options)
            else:
                all_gates = list(registry.get(args.adapter).gates)
                host_only = [
                    g for g in all_gates
                    if g not in QUARANTINED_GATES.get(args.adapter, ())
                ]
                selected = (host_only if host_only else all_gates) if args.all else args.gate
                result = engine.run(args.adapter, args.source, selected, **options)
        else:
            run_id = args.run_id or store.latest()
            verified = store.verify(run_id, registry=registry)
            if args.command == "verify":
                result = verified
            else:
                result = store.read(run_id)
                result["verification"] = verified
                # A receipt with no recorded status is malformed. It has already
                # failed verification, but reading `["status"]` would raise a
                # KeyError the handler below does not catch, so the command
                # would exit 1 with empty stdout instead of a structured report.
                if "status" not in result:
                    raise ValueError("receipt has no recorded status")
                result["recordedStatus"] = result["status"]
                result["status"] = verified["status"]
                # A gate's recorded PASS describes the run that produced it, not
                # the current state of the evidence. Once verification has
                # failed the whole receipt is no longer a passing claim, so no
                # gate may still present itself as PASS to a reader who iterates
                # `gates` instead of the top-level status.
                if verified["status"] != "PASS":
                    result["gates"] = [
                        {**gate, "recordedStatus": gate.get("status"), "status": verified["status"]}
                        for gate in result.get("gates", [])]
                result["receiptPath"] = str(store.directory(run_id) / "receipt.json")
        if args.json:
            print(json.dumps(result, indent=2))
        elif args.command == "list":
            for adapter in result["adapters"]:
                print(f"{adapter['id']}: {', '.join(adapter['gates'])}")
        else:
            print(f"{result['status']}  run {result['runId']}")
            if result.get("receiptPath"):
                print(f"Receipt: {result['receiptPath']}")
            if "allRoutes" in result:
                # A scoped PASS must not be quotable as a whole-adapter PASS, and
                # the human-readable output is what an operator pastes into a
                # ticket. The JSON carries the same facts; this line keeps the
                # text form from being the one place the scope is invisible.
                scope = ("all routes" if result["allRoutes"]
                         else "gates " + ", ".join(result.get("selectedGates") or []))
                print(f"Scope: {scope}; checked " + ", ".join(result.get("checkedRecipes") or []))
            for gate in result.get("gates", []):
                print(f"  {gate['gateId']}: {gate['status']}")
                for failure in gate.get("failures", []):
                    print(f"    {failure}")
            for failure in result.get("failures", []):
                print(f"  {failure}")
        return 0 if result.get("status", "PASS") == "PASS" else 1
    except dynamic_target.TargetRequestError as exc:
        # A target/candidate pair is request data: an unusable one is refused
        # before a run exists, never recorded as a machine verdict.
        result = {"status": "INVALID_REQUEST", "failures": [str(exc)]}
        if args.json:
            print(json.dumps(result))
        else:
            print(f"INVALID_REQUEST: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        result = {"status": "BLOCKED_ENV", "failures": [str(exc)]}
        if args.json:
            print(json.dumps(result))
        else:
            print(f"BLOCKED_ENV: {exc}", file=sys.stderr)
        return 2


def _emit(payload, args, status_ok=True):
    """Print one API model as JSON, identically for every asynchronous verb."""
    print(json.dumps(jsonable(payload), indent=2))
    return 0 if status_ok else 1


def _async_command(args):
    """Run one asynchronous verb; every failure becomes a stable error code."""
    try:
        manager = ExecutionManager()
        source_root = str(args.source) if getattr(args, "source", None) is not None else None
        if args.command == "submit":
            request = ExecutionRequest(workspace=args.workspace, adapter=args.adapter,
                                       gates=args.gate, request_id=args.request_id,
                                       target=args.target, candidate=args.candidate,
                                       source_root=source_root,
                                       device_id=args.device_id)
            return _emit(manager.submit(request), args)
        if args.command == "status":
            return _emit(manager.status(args.execution_id), args)
        if args.command == "wait":
            result = manager.wait(args.execution_id, args.timeout)
            # A bounded wait that returned without a verdict is a normal result,
            # not a command failure: the caller asked to wait only so long.
            return _emit(result, args)
        if args.command == "cancel":
            return _emit(manager.cancel(args.execution_id), args)
        requirement = VerificationRequirement(workspace=args.workspace, adapter=args.adapter,
                                              required_gates=args.gate, run_id=args.run_id,
                                              target=args.target, candidate=args.candidate,
                                              source_root=source_root)
        result = verify_requirements(requirement, config=manager.config,
                                     registry=manager.registry, store=manager.evidence)
        return _emit(result, args, status_ok=result.satisfied)
    except APIError as exc:
        payload = jsonable(exc.to_model())
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(exc.code.value + ": " + exc.message, file=sys.stderr)
        return 2
    except ConfigError as exc:
        payload = {"code": "CONFIG_INVALID", "family": "config", "message": str(exc),
                   "field": exc.field, "details": {}}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("CONFIG_INVALID: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
