"""Local receipt integrity and fresh source, adapter, log and artifact checks.

The per-user HMAC catches accidental edits and data-only forgery. It does not
isolate receipts from another process running with the same user's privileges.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
from .evaluator import equal, evaluate
from .executor import file_reference
from .manifest import Manifest, strict_json
from .source_identity import snapshot

RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def default_root():
    return Path.home() / ".local/state/agent-gates"


class EvidenceStore:
    def __init__(self, root: Path | None = None):
        self.root = (Path(root) if root is not None else default_root()).absolute()

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink():
            raise ValueError("evidence root must not be a symlink")
        key = self.root / ".receipt-key"
        try:
            fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            if key.is_symlink() or not key.is_file():
                raise ValueError("invalid receipt key")
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
        runs = self.root / "runs"
        if runs.is_symlink():
            raise ValueError("evidence runs directory must not be a symlink")
        runs.mkdir(exist_ok=True, mode=0o700)

    def create_run(self):
        self.initialize()
        run_id = secrets.token_hex(16)
        directory = self.root / "runs" / run_id
        directory.mkdir(mode=0o700)
        return run_id, directory

    def directory(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError("invalid run ID")
        path = self.root / "runs" / run_id
        if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("run not found or unsafe run path")
        return path

    def latest(self):
        directories = [path for path in (self.root / "runs").glob("*")
                       if path.is_dir() and RUN_ID.fullmatch(path.name) and (path / "receipt.json").is_file()]
        if not directories:
            raise ValueError("no runs available")
        return max(directories, key=lambda path: path.stat().st_mtime_ns).name

    def publish(self, receipt):
        directory = self.directory(receipt["runId"])
        payload = canonical(receipt)
        signature = hmac.new((self.root / ".receipt-key").read_bytes(), payload, hashlib.sha256).hexdigest()
        for name, value in (("receipt.json", receipt), ("receipt.attestation.json", {"signature": signature, "algorithm": "HMAC-SHA256", "trust": "local-user-integrity"})):
            with (directory / name).open("x") as stream:
                os.chmod(directory / name, 0o600)
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        return directory / "receipt.json"

    def read(self, run_id):
        path = self.directory(run_id) / "receipt.json"
        if path.is_symlink():
            raise ValueError("receipt must not be a symlink")
        raw = strict_json(path.read_text())
        if not isinstance(raw, dict) or raw.get("runId") != run_id:
            raise ValueError("receipt identity mismatch")
        return raw

    def verify(self, run_id, *, registry=None):
        try:
            directory = self.directory(run_id)
            receipt = self.read(run_id)
            attestation = strict_json((directory / "receipt.attestation.json").read_text())
            expected = hmac.new((self.root / ".receipt-key").read_bytes(), canonical(receipt), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, attestation["signature"]):
                raise ValueError("receipt authentication failed")
            if receipt.get("schemaVersion") != 1 or not isinstance(receipt.get("gates"), list):
                raise ValueError("invalid receipt schema")
            manifest = registry.get(receipt["adapter"]["id"]) if registry else Manifest(Path(receipt["adapter"]["path"]))
            if manifest.manifest_hash != receipt["adapter"]["sha256"]:
                return self._result(run_id, "STALE", ["ADAPTER_IDENTITY_CHANGED"])
            current = snapshot(Path(receipt["source"]["root"]))
            if current != receipt["source"]:
                return self._result(run_id, "STALE", ["SOURCE_IDENTITY_CHANGED"])
            for reference in receipt.get("configuration", {}).get("files", []):
                if not self._matches(reference):
                    return self._result(run_id, "STALE", ["CONFIGURATION_IDENTITY_CHANGED"])
            gate_ids = [gate["gateId"] for gate in receipt["gates"]]
            if gate_ids != manifest.order(receipt["selectedGates"]):
                raise ValueError("receipt gate dependency set or order mismatch")
            passed = set()
            for gate in receipt["gates"]:
                definition = manifest.gates[gate["gateId"]]
                recipe = manifest.resolve_recipe(definition["recipeId"], Path(receipt["source"]["root"]))
                if gate["recipeId"] != recipe.id or not equal(gate["checks"], definition["observationChecks"]):
                    raise ValueError("receipt gate rules differ from manifest")
                if gate.get("status") == "PASS":
                    if not gate.get("executable") or gate.get("failures"):
                        raise ValueError("passed gate missing executable identity or contains failures")
                    if any(dependency not in passed for dependency in definition.get("dependsOn", [])):
                        raise ValueError("passed gate has an unpassed dependency")
                    if gate.get("sourceBefore") != receipt["source"] or gate.get("sourceAfter") != receipt["source"]:
                        raise ValueError("passed gate source bindings missing or inconsistent")
                    process = gate["process"]
                    if process["argv"] != list(recipe.argv) or process["cwd"] != recipe.cwd:
                        raise ValueError("receipt argv or cwd differs from manifest")
                    if recipe.config_check and not gate.get("configCheck"):
                        raise ValueError("passed gate missing configCheck")
                for process_key in ("process", "configCheck"):
                    process = gate.get(process_key)
                    if not process:
                        continue
                    for key in ("stdout", "stderr", "processResult"):
                        reference = process[key]
                        if not Path(reference["path"]).resolve().is_relative_to(directory.resolve()) or not self._matches(reference):
                            raise ValueError("log or process evidence changed or escaped run directory")
                    terminal = strict_json(Path(process["processResult"]["path"]).read_text())
                    if any(not equal(process.get(key), value) for key, value in terminal.items()):
                        raise ValueError("receipt differs from terminal process evidence")
                if gate.get("executable") and not self._matches(gate["executable"]):
                    return self._result(run_id, "STALE", ["EXECUTABLE_IDENTITY_CHANGED"])
                for artifact in gate.get("artifacts", []):
                    if not self._matches(artifact):
                        return self._result(run_id, "STALE", ["ARTIFACT_IDENTITY_CHANGED"])
                if gate.get("status") == "PASS":
                    config = gate.get("configCheck")
                    if config and (config["argv"] != list(recipe.config_check) or config["cwd"] != recipe.cwd or config["exitCode"] != 0 or config["timedOut"] or config["interrupted"]):
                        raise ValueError("passed gate configuration check did not pass")
                    result = evaluate(gate["process"], definition["observationChecks"])
                    if result["status"] != "PASS" or not equal(result["observations"], gate["observations"]):
                        raise ValueError("gate no longer passes its recorded machine output")
                    from .engine import collect_artifacts, is_observation_only_recipe
                    observed_artifacts = collect_artifacts(result["observations"], Path(receipt["source"]["root"]))
                    if observed_artifacts and observed_artifacts != gate["artifacts"]:
                        raise ValueError("receipt artifacts differ from machine observations")
                    if is_observation_only_recipe(manifest.id, recipe.id, gate["checks"]):
                        before = {item["path"]: item for item in gate.get("artifactInputs", [])}
                        if any(item["path"] in before and item != before[item["path"]] for item in gate["artifacts"]):
                            raise ValueError("artifact changed during observation-only test")
                        if any(".so" in Path(item["path"]).name and item["path"] not in before for item in gate["artifacts"]):
                            raise ValueError("native artifact pretest identity missing")
                    passed.add(gate["gateId"])
            expected_status = aggregate(receipt["gates"])
            if receipt["status"] != expected_status:
                raise ValueError("receipt aggregate status mismatch")
            return self._result(run_id, expected_status, [failure for gate in receipt["gates"] for failure in gate["failures"]])
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            return self._result(run_id, "INVALID_EVIDENCE", [str(exc)])

    @staticmethod
    def _matches(reference):
        path = Path(reference["path"])
        return not path.is_symlink() and path.is_file() and file_reference(path) == reference

    @staticmethod
    def _result(run_id, status, failures):
        return {"runId": run_id, "status": status, "valid": status == "PASS", "failures": failures}


# Every status the engine can write. A receipt carrying anything else - a
# typo, a truncated write, a status a future gate path invents - is not a
# result this module understands, so it must never be read as a pass.
STATUSES = ("PASS", "FAIL", "BLOCKED_ENV", "NOT_RUN", "STALE", "INVALID_EVIDENCE")


def aggregate(gates):
    """The overall status of a run: PASS only when every gate literally passed.

    The previous form blocked a fixed list of failure literals and returned
    PASS for everything else, so an unknown status such as SKIPPED or ERROR
    became an overall pass and `verify` recomputed the same pass. The
    condition is now positive: an empty run is NOT_RUN, any unrecognised
    status is INVALID_EVIDENCE, and PASS requires every gate to be PASS.
    """
    statuses = [gate["status"] for gate in gates]
    if not statuses:
        return "NOT_RUN"
    unknown = [status for status in statuses if status not in STATUSES]
    if unknown:
        return "INVALID_EVIDENCE"
    for status in ("INVALID_EVIDENCE", "STALE", "FAIL", "BLOCKED_ENV", "NOT_RUN"):
        if status in statuses:
            return status
    return "PASS"
