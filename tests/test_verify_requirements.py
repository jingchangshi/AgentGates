"""Requirement verification: the checks a completion gate depends on.

Each test builds a real receipt by running a real gate through the real Engine,
then asks the verifier a question. Nothing here mocks the verification path,
because the point of the module is what it concludes about actual evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

from agent_gates import api_types as api
from agent_gates.config import from_mapping
from agent_gates.engine import Engine
from agent_gates.evidence import EvidenceStore
from agent_gates.manifest import digest
from agent_gates.registry import Registry
from agent_gates.verify_requirements import verify_requirements


class RequirementFixture(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "input.txt").write_text("source bytes")
        self.pack = self.root / "adapters/test-pack"
        (self.pack / "scripts").mkdir(parents=True)
        self.script = self.pack / "scripts/gate.py"
        # A gate whose observation can be steered, so PASS and FAIL are both real.
        self.script.write_text(
            "import json,sys,os\n"
            "if '--check-config' in sys.argv: sys.exit(0)\n"
            "ok = os.environ.get('GATE_OK', '1') == '1'\n"
            "print(json.dumps({'ok': ok}))\n")
        self.manifest = {
            "schemaVersion": 1, "id": "test-pack", "version": "1", "files": {},
            "permissions": {"sourceWrite": False, "hostExecution": True},
            "recipes": [
                {"id": "test.run", "cwd": "source", "timeout": 20,
                 "argv": [sys.executable, "-I", "{pack}/scripts/gate.py"],
                 "configCheck": [sys.executable, "-I", "{pack}/scripts/gate.py", "--check-config"]},
                {"id": "test.build", "cwd": "source", "timeout": 20,
                 "argv": [sys.executable, "-I", "{pack}/scripts/gate.py"],
                 "configCheck": [sys.executable, "-I", "{pack}/scripts/gate.py", "--config-check"]},
            ],
            "gates": [
                {"id": "build", "recipeId": "test.build", "gateType": "deterministic-test",
                 "observationChecks": {"ok": True}},
                {"id": "test", "recipeId": "test.run", "gateType": "deterministic-test",
                 "observationChecks": {"ok": True}, "dependsOn": ["build"]},
            ],
        }
        self.manifest["files"] = {"scripts/gate.py": digest(self.script)}
        (self.pack / "manifest.json").write_text(json.dumps(self.manifest))
        self.registry = Registry(self.root / "adapters")
        self.store = EvidenceStore(self.root / "evidence")
        self.engine = Engine(registry=self.registry, store=self.store, lock_root=self.root / "locks")

    def config(self, *, adapters=("test-pack",), allowed_gates=None):
        (self.root / "state").mkdir(exist_ok=True)
        (self.root / "evidence").mkdir(exist_ok=True)
        body = {
            "schema_version": 1,
            "state_root": str(self.root / "state"),
            "evidence_root": str(self.root / "evidence"),
            "adapters": {name: {} for name in adapters},
            "workspaces": {"ws": {"source_root": str(self.source),
                                  "adapters": list(adapters),
                                  **({"allowed_gates": list(allowed_gates)}
                                     if allowed_gates is not None else {})}},
            "worker": {"backend": "systemd", "max_concurrent": 1, "python": sys.executable},
        }
        return from_mapping(body, path=self.root / "config.toml")

    def requirement(self, **overrides):
        values = {"workspace": "ws", "adapter": "test-pack",
                  "required_gates": ["test"], "run_id": None}
        values.update(overrides)
        return api.VerificationRequirement(**values)

    def passing_run(self):
        receipt = self.engine.run("test-pack", self.source, ["test"])
        self.assertEqual(receipt["status"], "PASS")
        return receipt["runId"]

    def verify(self, requirement, **kwargs):
        return verify_requirements(requirement, config=kwargs.pop("config", self.config()),
                                   registry=self.registry, store=self.store)


class PositiveTests(RequirementFixture):
    def test_satisfied_run_reports_pass(self):
        run_id = self.passing_run()
        result = self.verify(self.requirement(run_id=run_id))
        self.assertTrue(result.satisfied)
        self.assertIs(result.status, api.GateStatus.PASS)
        self.assertEqual(result.run_id, run_id)
        self.assertEqual(result.matched_gates, ["test"])
        self.assertEqual(result.failures, [])
        self.assertIsNotNone(result.verified_at)
        self.assertIsNotNone(result.current_source_identity)

    def test_dependency_is_also_matched(self):
        run_id = self.passing_run()
        result = self.verify(self.requirement(run_id=run_id, required_gates=["test", "build"]))
        self.assertTrue(result.satisfied)
        self.assertEqual(sorted(result.matched_gates), ["build", "test"])


class NegativeTests(RequirementFixture):
    """Every case must leave satisfied false."""

    def test_no_run_id_refused(self):
        self.passing_run()
        result = self.verify(self.requirement(run_id=None))
        self.assertFalse(result.satisfied)
        self.assertIn("RUN_ID_REQUIRED", result.failures)
        self.assertIs(result.status, api.GateStatus.NOT_RUN)
        self.assertIsNone(result.run_id)

    def test_latest_is_not_accepted_as_a_run_id(self):
        self.passing_run()
        result = self.verify(self.requirement(run_id="latest"))
        self.assertFalse(result.satisfied)
        self.assertIn("RECEIPT_NOT_FOUND", result.failures[0])

    def test_wrong_run_id_refused(self):
        self.passing_run()
        result = self.verify(self.requirement(run_id="0" * 32))
        self.assertFalse(result.satisfied)
        self.assertIn("RECEIPT_NOT_FOUND", result.failures[0])

    def test_empty_required_gates_refused(self):
        run_id = self.passing_run()
        result = self.verify(self.requirement(run_id=run_id, required_gates=[]))
        self.assertFalse(result.satisfied)
        self.assertIn("REQUIRED_GATES_EMPTY", result.failures)

    def test_missing_required_gate_refused(self):
        run_id = self.passing_run()
        result = self.verify(self.requirement(run_id=run_id, required_gates=["test", "absent"]))
        self.assertFalse(result.satisfied)
        self.assertIn("GATE_NOT_IN_RECEIPT: absent", result.failures)

    def test_failing_gate_refused(self):
        import agent_gates.engine as engine_module
        # Force the gate's observation to fail by patching the adapter script's
        # environment for this one run.
        original = engine_module.environment
        engine_module.environment = lambda manifest, env_file=None, device_id=None,             require_exclusive_device=False: {
            **original(manifest, env_file, device_id, require_exclusive_device), "GATE_OK": "0"}
        try:
            receipt = self.engine.run("test-pack", self.source, ["test"])
        finally:
            engine_module.environment = original
        self.assertEqual(receipt["status"], "FAIL")
        result = self.verify(self.requirement(run_id=receipt["runId"]))
        self.assertFalse(result.satisfied)
        # A FAILing receipt is rejected at the receipt-validity check, before
        # per-gate inspection; the report must still name the gate status.
        self.assertTrue(any("RECEIPT_INVALID: FAIL" in item for item in result.failures),
                        result.failures)

    def test_adapter_mismatch_refused(self):
        run_id = self.passing_run()
        # The workspace now names a second adapter; verify against the wrong one.
        config = self.config(adapters=("test-pack", "other-pack"))
        result = self.verify(self.requirement(run_id=run_id, adapter="other-pack"), config=config)
        self.assertFalse(result.satisfied)
        self.assertTrue(any("ADAPTER_MISMATCH" in item or "ADAPTER_NOT_REGISTERED" in item
                            for item in result.failures))

    def test_unknown_workspace_refused(self):
        run_id = self.passing_run()
        result = self.verify(self.requirement(run_id=run_id, workspace="nope"))
        self.assertFalse(result.satisfied)
        self.assertTrue(any("UNKNOWN_WORKSPACE" in item for item in result.failures))

    def test_workspace_source_root_mismatch_refused(self):
        """The receipt was produced from a different tree than config names now."""
        run_id = self.passing_run()
        other = self.root / "other-source"
        other.mkdir()
        (other / "input.txt").write_text("different")
        config = from_mapping({
            "schema_version": 1,
            "state_root": str(self.root / "state"),
            "evidence_root": str(self.root / "evidence"),
            "adapters": {"test-pack": {}},
            "workspaces": {"ws": {"source_root": str(other), "adapters": ["test-pack"]}},
            "worker": {"backend": "systemd", "max_concurrent": 1, "python": sys.executable},
        }, path=self.root / "c.toml")
        result = verify_requirements(self.requirement(run_id=run_id), config=config,
                                     registry=self.registry, store=self.store)
        self.assertFalse(result.satisfied)
        self.assertTrue(any("WORKSPACE_SOURCE_MISMATCH" in item for item in result.failures))

    def test_source_changed_after_gate_refused_as_stale(self):
        run_id = self.passing_run()
        (self.source / "input.txt").write_text("modified after the gate")
        result = self.verify(self.requirement(run_id=run_id))
        self.assertFalse(result.satisfied)
        self.assertIs(result.status, api.GateStatus.STALE)
        # The store's own re-verification notices the drift first and reports
        # it as STALE; the requirement result must carry that same verdict.
        self.assertTrue(any("SOURCE_IDENTITY_CHANGED" in item for item in result.failures),
                        result.failures)

    def test_tampered_receipt_refused(self):
        """Editing a published receipt breaks its HMAC attestation."""
        run_id = self.passing_run()
        receipt_path = self.store.directory(run_id) / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        # A change that would otherwise read as MORE successful than reality.
        receipt["gates"][0]["failures"] = []
        receipt["startedAt"] = "1999-01-01T00:00:00+00:00"
        receipt_path.write_text(json.dumps(receipt))
        result = self.verify(self.requirement(run_id=run_id))
        self.assertFalse(result.satisfied)
        self.assertIs(result.status, api.GateStatus.INVALID_EVIDENCE)
        self.assertTrue(any("RECEIPT_INVALID" in item for item in result.failures),
                        result.failures)

    def test_gate_outside_workspace_allowlist_refused(self):
        run_id = self.passing_run()
        config = self.config(allowed_gates=["build"])
        result = self.verify(self.requirement(run_id=run_id), config=config)
        self.assertFalse(result.satisfied)
        self.assertIn("GATE_NOT_ALLOWED: test", result.failures)

    def test_receipt_does_not_satisfy_a_different_workspace_tree(self):
        """A PASS from tree A must never satisfy a requirement bound to tree B."""
        run_id = self.passing_run()
        second = self.root / "second"
        second.mkdir()
        (second / "input.txt").write_text("second")
        config = from_mapping({
            "schema_version": 1,
            "state_root": str(self.root / "state"),
            "evidence_root": str(self.root / "evidence"),
            "adapters": {"test-pack": {}},
            "workspaces": {"two": {"source_root": str(second), "adapters": ["test-pack"]}},
            "worker": {"backend": "systemd", "max_concurrent": 1, "python": sys.executable},
        }, path=self.root / "c.toml")
        result = verify_requirements(self.requirement(run_id=run_id, workspace="two"),
                                     config=config, registry=self.registry, store=self.store)
        self.assertFalse(result.satisfied)


if __name__ == "__main__":
    unittest.main()
