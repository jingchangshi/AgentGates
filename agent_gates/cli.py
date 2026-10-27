"""The agent-gates command line; execution uses installed fixed recipes only."""
import argparse
import json
from pathlib import Path
import sys
from .engine import Engine
from .evidence import EvidenceStore
from .registry import Registry


def parser():
    root = argparse.ArgumentParser(prog="agent-gates")
    subcommands = root.add_subparsers(dest="command", required=True)
    for name in ("list", "doctor", "run", "report", "verify"):
        command = subcommands.add_parser(name)
        command.add_argument("--json", action="store_true", help="print structured JSON")
        command.add_argument("--evidence-root", type=Path, help="local private evidence directory")
        if name in {"doctor", "run"}:
            command.add_argument("--adapter", required=True)
            command.add_argument("--source", required=True, type=Path)
            command.add_argument("--env-file", type=Path)
            command.add_argument("--device-id", help="explicitly authorize this NPU device for the run")
        if name == "run":
            selection = command.add_mutually_exclusive_group(required=True)
            selection.add_argument("--gate", action="append", help="gate ID; dependencies run first")
            selection.add_argument("--all", action="store_true", help="run every gate in this adapter")
        if name in {"report", "verify"}:
            command.add_argument("--run-id", help="run ID (default: most recent)")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    registry = Registry()
    store = EvidenceStore(args.evidence_root)
    engine = Engine(registry=registry, store=store)
    try:
        if args.command == "list":
            result = {"adapters": [{"id": manifest.id, "manifestSha256": manifest.manifest_hash,
                                     "gates": list(manifest.gates)} for manifest in registry.list()]}
        elif args.command in {"doctor", "run"}:
            options = {"env_file": args.env_file, "device_id": args.device_id}
            if args.command == "doctor":
                result = engine.doctor(args.adapter, args.source, **options)
            else:
                selected = list(registry.get(args.adapter).gates) if args.all else args.gate
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
            for gate in result.get("gates", []):
                print(f"  {gate['gateId']}: {gate['status']}")
                for failure in gate.get("failures", []):
                    print(f"    {failure}")
            for failure in result.get("failures", []):
                print(f"  {failure}")
        return 0 if result.get("status", "PASS") == "PASS" else 1
    except (OSError, ValueError, RuntimeError) as exc:
        result = {"status": "BLOCKED_ENV", "failures": [str(exc)]}
        if args.json:
            print(json.dumps(result))
        else:
            print(f"BLOCKED_ENV: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
