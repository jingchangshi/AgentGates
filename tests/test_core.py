"""Offline extraction contracts; compiler checkouts and hardware are never touched."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from agent_gates.cli import main
from agent_gates.engine import Engine, environment
from agent_gates.evaluator import equal, evaluate
from agent_gates.evidence import EvidenceStore
from agent_gates.executor import execute
from agent_gates.locking import ResourceLock, LockBusy
from agent_gates.manifest import Manifest, ManifestError, digest
from agent_gates.registry import Registry
from agent_gates.source_identity import snapshot


class CoreFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "input.txt").write_text("source bytes")
        self.pack = self.root / "adapters/test-pack"
        (self.pack / "scripts").mkdir(parents=True)
        self.script = self.pack / "scripts/gate.py"
        self.script.write_text('import json,sys\nif "--check-config" in sys.argv: sys.exit(0)\nprint(json.dumps({"ok": True}))\n')
        self.manifest = {"schemaVersion": 1, "id": "test-pack", "version": "1", "files": {},
                         "permissions": {"sourceWrite": False, "hostExecution": True},
                         "recipes": [{"id": "test.run", "cwd": "source", "timeout": 5,
                                      "argv": [sys.executable, "-I", "{pack}/scripts/gate.py"],
                                      "configCheck": [sys.executable, "-I", "{pack}/scripts/gate.py", "--check-config"]}],
                         "gates": [{"id": "test", "recipeId": "test.run", "gateType": "deterministic-test",
                                    "observationChecks": {"ok": True}}]}
        self.write_manifest()
        self.registry = Registry(self.root / "adapters")
        self.store = EvidenceStore(self.root / "evidence")
        self.engine = Engine(registry=self.registry, store=self.store, lock_root=self.root / "locks")

    def write_manifest(self):
        self.manifest["files"] = {"scripts/gate.py": digest(self.script)}
        (self.pack / "manifest.json").write_text(json.dumps(self.manifest))

    def run_gate(self):
        return self.engine.run("test-pack", self.source, ["test"])


class ManifestTests(CoreFixture):
    def test_fixed_argv_source_cwd_and_config_check(self):
        recipe = self.registry.get("test-pack").resolve_recipe("test.run", self.source)
        self.assertEqual(recipe.argv[2], str(self.script))
        self.assertEqual(recipe.cwd, str(self.source))
        self.assertEqual(recipe.config_check[-1], "--check-config")

    def test_hash_drift_and_manifest_drift_rejected(self):
        loaded = self.registry.get("test-pack")
        self.script.write_text("print('changed')")
        with self.assertRaisesRegex(ManifestError, "hash mismatch"):
            loaded.validate_files()
        self.write_manifest()
        loaded = self.registry.get("test-pack")
        (self.pack / "manifest.json").write_text("{}")
        with self.assertRaisesRegex(ManifestError, "manifest changed"):
            loaded.validate_files()

    def test_pinned_path_symlink_and_escape_rejected(self):
        self.manifest["recipes"][0]["argv"][2] = "{pack}/../outside.py"
        self.write_manifest()
        with self.assertRaises(ManifestError):
            self.registry.get("test-pack")
        self.manifest["recipes"][0]["argv"][2] = "{pack}/scripts/gate.py"
        target = self.root / "external.py"
        target.write_text("print('{}')")
        self.script.unlink()
        self.script.symlink_to(target)
        self.write_manifest()
        with self.assertRaisesRegex(ManifestError, "symlink"):
            self.registry.get("test-pack")

    def test_invalid_adapter_ids_and_schema(self):
        for value in ("../test-pack", "/tmp/test-pack", "Test", "test/pack"):
            with self.assertRaises(ManifestError):
                self.registry.get(value)
        self.manifest["schemaVersion"] = True
        self.write_manifest()
        with self.assertRaises(ManifestError):
            self.registry.get("test-pack")

    def test_host_execution_must_be_granted_explicitly(self):
        # The permission is the manifest's consent to run commands on this
        # machine, so an omitted or merely truthy grant must not read as yes.
        for value in (None, {}, {"hostExecution": 0}, {"hostExecution": "true"},
                      {"hostExecution": 1}, {"hostExecution": False}):
            if value is None:
                self.manifest.pop("permissions", None)
            else:
                self.manifest["permissions"] = value
            self.write_manifest()
            with self.assertRaisesRegex(ManifestError, "hostExecution"):
                self.registry.get("test-pack")
        self.manifest["permissions"] = {"sourceWrite": False, "hostExecution": True}
        self.write_manifest()
        self.assertEqual(self.registry.get("test-pack").id, "test-pack")

    def test_a_manifest_without_permissions_never_executes(self):
        # End to end: the refusal happens at load, so no recipe process starts.
        marker = self.root / "EXECUTED"
        self.script.write_text(
            'import json,sys\nfrom pathlib import Path\n'
            f'Path({str(marker)!r}).write_text("ran")\n'
            'print(json.dumps({"ok": True}))\n')
        self.manifest["permissions"] = {}
        self.write_manifest()
        with self.assertRaises(ManifestError):
            self.engine.run("test-pack", self.source, ["test"])
        self.assertFalse(marker.exists(), "an ungranted adapter executed on the host")

    def test_dag_closure_order_and_cycle(self):
        self.manifest["gates"].append({"id": "second", "recipeId": "test.run", "gateType": "deterministic-test",
                                       "observationChecks": {"ok": True}, "dependsOn": ["test"]})
        self.write_manifest()
        self.assertEqual(self.registry.get("test-pack").order(["second"]), ["test", "second"])
        self.manifest["gates"][0]["dependsOn"] = ["second"]
        self.write_manifest()
        with self.assertRaisesRegex(ManifestError, "cycle"):
            self.registry.get("test-pack")

    def test_malformed_argv_and_timeout_rejected(self):
        for value in ([sys.executable, 1], [], "python"):
            self.manifest["recipes"][0]["argv"] = value
            self.write_manifest()
            with self.assertRaises(ManifestError):
                self.registry.get("test-pack")
        self.manifest["recipes"][0]["argv"] = [sys.executable]
        self.manifest["recipes"][0]["timeout"] = True
        self.write_manifest()
        with self.assertRaises(ManifestError):
            self.registry.get("test-pack")


class EvaluationTests(CoreFixture):
    def process(self, value):
        path = self.root / "stdout"
        path.write_text(value)
        return {"exitCode": 0, "timedOut": False, "interrupted": False, "stdout": {"path": str(path)}}

    def test_observations_are_type_strict(self):
        for raw in ('{"ok":1}', '{"ok":"true"}', '{}', '[]', 'garbage', '{"ok":true,"ok":false}', '{"ok":NaN}'):
            result = evaluate(self.process(raw), {"ok": True})
            self.assertEqual(result["status"], "FAIL", raw)
        self.assertEqual(evaluate(self.process('{"ok":true}'), {"ok": True})["status"], "PASS")
        self.assertFalse(equal({"ok": 1}, {"ok": True}))

    def test_nonzero_timeout_and_signal_cannot_pass(self):
        for key, value in (("exitCode", 1), ("timedOut", True), ("interrupted", True), ("exitCode", -15)):
            process = self.process('{"ok":true}')
            process[key] = value
            self.assertEqual(evaluate(process, {"ok": True})["status"], "FAIL")

    def test_zero_and_all_skip_counts_cannot_pass(self):
        for collected, skipped in ((0, 0), (2, 2), (True, 0), (2, "0")):
            raw = json.dumps({"testsCollectedPositive": True, "notAllSkipped": True,
                              "testsCollected": collected, "testsSkipped": skipped})
            self.assertEqual(evaluate(self.process(raw), {"testsCollectedPositive": True, "notAllSkipped": True})["status"], "FAIL")

    def test_a_missing_counter_cannot_opt_out_of_the_cross_check(self):
        # Omitting `testsCollected` and `testsSkipped` used to skip the
        # cross-check entirely, so a report with the two boolean claims and no
        # counts passed - which is the zero-test case those checks exist for.
        for raw in ({"testsCollectedPositive": True, "notAllSkipped": True},
                    {"testsCollectedPositive": True},
                    {"notAllSkipped": True}):
            result = evaluate(self.process(json.dumps(raw)),
                              {"testsCollectedPositive": True, "notAllSkipped": True})
            self.assertEqual(result["status"], "FAIL", raw)
            self.assertTrue(result["failures"], raw)
        self.assertEqual(evaluate(self.process(json.dumps(
            {"testsCollectedPositive": True, "notAllSkipped": True,
             "testsCollected": 3, "testsSkipped": 0})),
            {"testsCollectedPositive": True, "notAllSkipped": True})["status"], "PASS")

    def test_a_produced_exit_code_claim_cannot_outvote_the_real_one(self):
        # An adapter writes `exitCodeZero` for a command *inside* the container,
        # but the receipt binds the host process. An exit-code-only gate ignores
        # the produced value entirely - ordinary build output must not speak.
        # Any other gate keeps it, so a container failure cannot be hidden by an
        # adapter that exited 0, and only a literal `true` agrees with a pass.
        process = self.process('{"exitCodeZero": true, "ok": true}')
        process["exitCode"] = 1
        self.assertEqual(evaluate(process, {"exitCodeZero": True})["status"], "FAIL")
        self.assertEqual(evaluate(process, {"ok": True, "exitCodeZero": True})["status"], "FAIL")
        self.assertEqual(evaluate(self.process('{"exitCodeZero": false}'), {"exitCodeZero": True})["status"], "PASS")
        for raw in ('{"exitCodeZero": false, "ok": true}', '{"exitCodeZero": 1, "ok": true}',
                    '{"exitCodeZero": "true", "ok": true}'):
            self.assertEqual(evaluate(self.process(raw), {"ok": True, "exitCodeZero": True})["status"], "FAIL", raw)
        self.assertEqual(evaluate(self.process('{"exitCodeZero": true, "ok": true}'),
                                  {"ok": True, "exitCodeZero": True})["status"], "PASS")

    def test_overflowing_json_number_is_rejected_not_read_as_infinity(self):
        # `parse_constant` rejects the NaN/Infinity literals, but `1e999` is a
        # syntactically finite spelling that parses to `inf`. Such a value would
        # only fail later, when the receipt is serialised with allow_nan=False.
        path = self.root / "overflow.json"
        path.write_text('{"value": 1e999}')
        from agent_gates.manifest import strict_json
        with self.assertRaises(ValueError):
            strict_json(path.read_text())
        process = dict(exitCode=0, timedOut=False, interrupted=False, stdout={"path": str(path)})
        self.assertEqual(evaluate(process, {"value": 1.0})["status"], "FAIL")

    def test_exit_only_gate_preserves_old_build_contract(self):
        process = self.process("ordinary build output")
        self.assertEqual(evaluate(process, {"exitCodeZero": True})["status"], "PASS")


class EngineTests(CoreFixture):
    def test_pass_receipt_and_fresh_verification(self):
        run = self.run_gate()
        self.assertEqual(run["status"], "PASS", run)
        self.assertTrue(self.store.verify(run["runId"], registry=self.registry)["valid"])
        self.assertEqual(run["gates"][0]["process"]["argv"][2], str(self.script))

    def test_failed_config_blocks_before_recipe(self):
        self.script.write_text('import sys\nif "--check-config" in sys.argv: sys.exit(2)\nraise RuntimeError("must not execute")\n')
        self.write_manifest()
        run = self.run_gate()
        self.assertEqual(run["status"], "BLOCKED_ENV")
        self.assertNotIn("process", run["gates"][0])
        self.assertIn("CONFIG_CHECK_FAILED", run["gates"][0]["failures"])

    def test_config_check_cannot_change_source_before_dispatch(self):
        self.script.write_text('import sys\nfrom pathlib import Path\nif "--check-config" in sys.argv:\n Path("input.txt").write_text("changed by preflight")\n sys.exit(0)\nraise RuntimeError("must not start recipe after mutation")\n')
        self.write_manifest()
        run = self.run_gate()
        self.assertEqual(run["status"], "STALE")
        self.assertNotIn("process", run["gates"][0])

    def test_failed_dependency_does_not_dispatch(self):
        self.script.write_text('import sys\nsys.exit(0 if "--check-config" in sys.argv else 1)\n')
        self.manifest["gates"].append({"id": "second", "recipeId": "test.run", "gateType": "deterministic-test",
                                       "observationChecks": {"ok": True}, "dependsOn": ["test"]})
        self.write_manifest()
        run = self.engine.run("test-pack", self.source, ["second"])
        self.assertEqual([gate["status"] for gate in run["gates"]], ["FAIL", "NOT_RUN"])
        self.assertNotIn("process", run["gates"][1])

    def test_a_declared_blocked_exit_code_is_not_a_failed_test(self):
        # The Adapter uses a dedicated exit code for "the environment is not
        # available" - a card another job holds. Nothing was dispatched, so
        # recording FAIL would claim a test failed that never started. The
        # classification must match what the configuration check produces for
        # the same condition, and it must still never pass.
        self.manifest["recipes"][0]["blockedExitCodes"] = [75]
        self.script.write_text('import sys\nif "--check-config" in sys.argv: sys.exit(0)\n'
                               'print("the card is held by another job", file=sys.stderr)\nsys.exit(75)\n')
        self.write_manifest()
        run = self.run_gate()
        gate = run["gates"][0]
        self.assertEqual(run["status"], "BLOCKED_ENV")
        self.assertEqual(gate["status"], "BLOCKED_ENV")
        self.assertIn("BLOCKED_EXIT_CODE: 75", gate["failures"])
        self.assertIn("held by another job", gate["configReason"])
        self.assertEqual(gate["process"]["exitCode"], 75)
        self.assertFalse(self.store.verify(run["runId"], registry=self.registry)["valid"])

    def test_an_undeclared_exit_code_is_still_a_failure(self):
        # The classification is opt-in per recipe: without the declaration the
        # same exit status is an ordinary non-zero result, so an adapter cannot
        # make arbitrary failures disappear into BLOCKED_ENV.
        self.script.write_text('import sys\nif "--check-config" in sys.argv: sys.exit(0)\nsys.exit(75)\n')
        self.write_manifest()
        run = self.run_gate()
        self.assertEqual(run["status"], "FAIL")
        self.assertEqual(run["gates"][0]["status"], "FAIL")
        self.assertIn("NONZERO_EXIT", run["gates"][0]["failures"])

    def test_a_blocked_exit_code_still_cannot_pass(self):
        # A gate that exits with the blocked code after doing the work is not a
        # pass even if its stdout claims every observation succeeded: the
        # declaration is a statement about the environment, not a bypass.
        self.manifest["recipes"][0]["blockedExitCodes"] = [75]
        self.script.write_text('import json,sys\nif "--check-config" in sys.argv: sys.exit(0)\n'
                               'print(json.dumps({"ok": True}))\nsys.exit(75)\n')
        self.write_manifest()
        run = self.run_gate()
        self.assertEqual(run["status"], "BLOCKED_ENV")
        self.assertNotEqual(run["gates"][0]["status"], "PASS")

    def test_a_timed_out_recipe_stays_a_failure_despite_its_exit_status(self):
        # The supervisor kills a timed-out child, and the status of a killed
        # child can coincide with a declared blocked code. A timeout means work
        # started and was cut short, which is a different event from "the
        # environment is not available", so it must not be reclassified.
        self.manifest["recipes"][0]["blockedExitCodes"] = [75]
        self.script.write_text('import sys\nif "--check-config" in sys.argv: sys.exit(0)\nsys.exit(75)\n')
        self.write_manifest()
        from agent_gates import engine as engine_module
        real = engine_module.execute

        def timed_out(argv, **kwargs):
            # Only the recipe is timed out; the configuration check still runs.
            if "--check-config" in argv:
                return real(argv, **kwargs)
            return {**real(argv, **kwargs), "timedOut": True}

        with patch("agent_gates.engine.execute", side_effect=timed_out):
            run = self.run_gate()
        gate = run["gates"][0]
        self.assertEqual(gate["status"], "FAIL")
        self.assertIn("TIMEOUT", gate["failures"])
        self.assertNotIn("BLOCKED_EXIT_CODE: 75", gate["failures"])

    def test_an_unwritable_receipt_names_the_directory_instead_of_vanishing(self):
        # Everything before publishing is already on disk. A publish fault used
        # to escape as a bare ValueError from outside the run's handler, so the
        # caller saw only an exception and the run directory went unmentioned.
        with patch.object(self.store, "publish", side_effect=OSError("disk full")):
            with self.assertRaises(RuntimeError) as error:
                self.run_gate()
        self.assertIn("disk full", str(error.exception))
        self.assertIn("retained in", str(error.exception))
        runs = list((self.root / "evidence" / "runs").iterdir())
        self.assertEqual(len(runs), 1, "the run directory must survive a publish fault")
        self.assertTrue(any(runs[0].rglob("stdout.log")), "gate logs must survive a publish fault")

    def test_a_malformed_receipt_reports_a_status_instead_of_crashing(self):
        # `report` read `receipt["status"]` unguarded, so a receipt without one
        # raised KeyError outside the CLI's handler: exit 1, empty stdout.
        run = self.run_gate()
        self.assertEqual(run["status"], "PASS")
        receipt_path = Path(run["receiptPath"])
        receipt = json.loads(receipt_path.read_text())
        del receipt["status"]
        receipt_path.write_text(json.dumps(receipt))
        with patch("agent_gates.cli.Registry", return_value=self.registry), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["report", "--run-id", run["runId"],
                         "--evidence-root", str(self.root / "evidence"), "--json"])
        self.assertEqual(code, 2)
        reported = json.loads(output.getvalue())
        self.assertEqual(reported["status"], "BLOCKED_ENV")
        self.assertIn("recorded status", " ".join(reported["failures"]))

    def test_source_changed_during_execution_cannot_pass(self):
        self.script.write_text('import json,sys\nfrom pathlib import Path\nif "--check-config" in sys.argv: sys.exit(0)\nPath("input.txt").write_text("changed")\nprint(json.dumps({"ok": True}))\n')
        self.write_manifest()
        run = self.run_gate()
        self.assertEqual(run["status"], "STALE")

    def test_source_changes_after_run_stale(self):
        run = self.run_gate()
        (self.source / "input.txt").write_text("different")
        self.assertEqual(self.store.verify(run["runId"], registry=self.registry)["status"], "STALE")

    def test_artifact_identity_recorded_and_revalidated(self):
        product = self.source / "native.so"
        product.write_bytes(b"native bytes")
        self.script.write_text('import json,sys,hashlib\nfrom pathlib import Path\nif "--check-config" in sys.argv: sys.exit(0)\np=Path("native.so").resolve()\nprint(json.dumps({"ok":True,"artifacts":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}]}))\n')
        self.write_manifest()
        run = self.run_gate()
        self.assertEqual(run["status"], "PASS")
        self.assertEqual(run["gates"][0]["artifacts"][0]["path"], str(product))
        product.write_bytes(b"wrong bytes")
        self.assertEqual(self.store.verify(run["runId"], registry=self.registry)["status"], "STALE")

    def test_ignored_artifact_changes_invalidate_receipt(self):
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        (self.source / ".gitignore").write_text("native.so\\n")
        subprocess.run(["git", "-C", str(self.source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        product = self.source / "native.so"
        product.write_bytes(b"native bytes")
        self.script.write_text('import json,sys,hashlib\nfrom pathlib import Path\nif "--check-config" in sys.argv: sys.exit(0)\np=Path("native.so").resolve()\nprint(json.dumps({"ok":True,"artifacts":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}]}))\n')
        self.write_manifest()
        run = self.run_gate()
        self.assertEqual(run["status"], "PASS")
        product.write_bytes(b"changed")
        verified = self.store.verify(run["runId"], registry=self.registry)
        self.assertEqual(verified["status"], "STALE")
        self.assertIn("ARTIFACT_IDENTITY_CHANGED", verified["failures"])

    def test_corrupted_receipt_or_logs_invalid(self):
        run = self.run_gate()
        receipt_path = Path(run["receiptPath"])
        original = receipt_path.read_text()
        receipt_path.write_text(original.replace('"status": "PASS"', '"status": "FAIL"'))
        self.assertEqual(self.store.verify(run["runId"], registry=self.registry)["status"], "INVALID_EVIDENCE")
        receipt_path.write_text(original)
        Path(run["gates"][0]["process"]["stdout"]["path"]).write_text('{"ok":true}')
        self.assertEqual(self.store.verify(run["runId"], registry=self.registry)["status"], "INVALID_EVIDENCE")

    def test_adapter_drift_invalidates_verification(self):
        run = self.run_gate()
        self.script.write_text("changed")
        self.assertEqual(self.store.verify(run["runId"], registry=self.registry)["status"], "INVALID_EVIDENCE")

    def test_doctor_checks_without_running_gate(self):
        self.script.write_text('import sys\nif "--check-config" in sys.argv: sys.exit(0)\nraise RuntimeError("doctor launched gate")\n')
        self.write_manifest()
        result = self.engine.doctor("test-pack", self.source)
        self.assertEqual(result["status"], "PASS")

    def test_evidence_directory_inside_source_is_rejected(self):
        engine = Engine(registry=self.registry, store=EvidenceStore(self.source / "evidence"))
        with self.assertRaisesRegex(ValueError, "outside"):
            engine.run("test-pack", self.source, ["test"])

    def test_source_lock_blocks_dispatch(self):
        with ResourceLock("source:" + str(self.source), self.root / "locks"):
            run = self.run_gate()
        self.assertEqual(run["status"], "BLOCKED_ENV")
        self.assertNotIn("process", run["gates"][0])

    def test_cli_invalid_adapter_and_run_id(self):
        self.assertEqual(main(["run", "--adapter", "../bad", "--source", str(self.source), "--all", "--json"]), 2)
        self.assertEqual(main(["verify", "--run-id", "../bad", "--evidence-root", str(self.root), "--json"]), 1)

    def test_report_never_shows_a_passing_gate_once_verification_fails(self):
        # `report` keeps the recorded gate statuses for traceability, but a
        # reader that iterates `gates` must not see PASS for a receipt that no
        # longer verifies. The recorded value moves to `recordedStatus`.
        run = self.run_gate()
        self.assertEqual(run["status"], "PASS")
        (self.source / "input.txt").write_text("source bytes changed after the run")
        with patch("agent_gates.cli.Registry", return_value=self.registry), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["report", "--run-id", run["runId"],
                         "--evidence-root", str(self.root / "evidence"), "--json"])
        self.assertEqual(code, 1)
        reported = json.loads(output.getvalue())
        self.assertEqual(reported["status"], "STALE")
        self.assertTrue(reported["gates"], "report dropped the gate list")
        for gate in reported["gates"]:
            self.assertEqual(gate["status"], "STALE")
            self.assertEqual(gate["recordedStatus"], "PASS")


class ProcessTests(CoreFixture):
    def execute(self, code, timeout=5):
        return execute([sys.executable, "-I", "-u", "-c", code], cwd=self.source, env=dict(os.environ),
                       timeout=timeout, log_dir=self.root / "process-logs")

    def test_timeout_retains_partial_logs_and_signal(self):
        result = self.execute('import time; print("checkpoint"); time.sleep(30)', timeout=0.25)
        self.assertTrue(result["timedOut"])
        self.assertEqual(result["signal"], signal.SIGKILL)
        self.assertIn("checkpoint", Path(result["stdout"]["path"]).read_text())

    def test_relative_path_entries_resolve_against_the_recipe_cwd(self):
        # A relative PATH entry means "relative to the process's own cwd". The
        # recipe runs with cwd=recipe.cwd, not with the directory that launched
        # agent-gates, so resolving it against this process would bind a binary
        # the recipe never executed.
        from agent_gates.executor import executable_identity
        tool_dir = self.source / "bin"
        tool_dir.mkdir()
        tool = tool_dir / "fixture-tool"
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
        identity = executable_identity(["fixture-tool"], str(self.source), {"PATH": "bin"})
        self.assertEqual(identity["sha256"], digest(tool))
        # A relative entry that only exists under the launching directory is
        # not a resolution: the recipe's cwd is the one that counts.
        self.assertFalse((self.root / "bin" / "fixture-tool").exists())

    def test_normal_exit_reaps_detached_descendant(self):
        result = self.execute('import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],start_new_session=True); Path("child-pid").write_text(str(p.pid))')
        self.assertEqual(result["exitCode"], 0)
        pid = (self.source / "child-pid").read_text()
        self.assertFalse(Path("/proc/" + pid).exists())

    def test_lock_retained_by_inherited_descriptor(self):
        lock = ResourceLock("device:0", self.root / "locks")
        lock.__enter__()
        inherited = os.dup(lock.fileno())
        try:
            lock.__exit__()
            with self.assertRaises(LockBusy):
                with ResourceLock("device:0", self.root / "locks"):
                    pass
        finally:
            os.close(inherited)
        with ResourceLock("device:0", self.root / "locks"):
            pass


class DeviceTests(CoreFixture):
    def setUp(self):
        super().setUp()
        self.manifest["id"] = "tilelang-hivm-npu"
        self.pack = self.pack.rename(self.root / "adapters/tilelang-hivm-npu")
        self.script = self.pack / "scripts/gate.py"
        self.write_manifest()

    def test_missing_device_authorization_blocks(self):
        with patch.dict(os.environ, {}, clear=True):
            run = self.engine.run("tilelang-hivm-npu", self.source, ["test"])
        self.assertEqual(run["status"], "BLOCKED_ENV")
        self.assertIn("authorization required", run["gates"][0]["failures"][0])
        self.assertNotIn("process", run["gates"][0])

    def test_device_lease_prevents_duplicate_occupancy(self):
        with ResourceLock("npu:3", self.root / "locks"):
            run = self.engine.run("tilelang-hivm-npu", self.source, ["test"], device_id="3")
        self.assertEqual(run["status"], "BLOCKED_ENV")
        self.assertNotIn("process", run["gates"][0])

    def test_env_file_is_data_and_device_is_explicit(self):
        path = self.root / "env.sh"
        path.write_text("TILELANG_HIVM_CONTAINER_NAME=$(touch /must-not-exist)\\n")
        env = environment(self.registry.get("tilelang-hivm-npu"), path, "03")
        self.assertEqual(env["TILELANG_HIVM_ENV_FILE"], str(path))
        self.assertEqual(env["TILELANG_HIVM_DEVICE_ID"], "3")
        with self.assertRaisesRegex(ValueError, "absolute"):
            environment(self.registry.get("tilelang-hivm-npu"), Path("env.sh"))


class ParentDeathTests(CoreFixture):
    def test_sigkill_runner_reaps_detached_child_and_releases_lease(self):
        helper = self.root / "host.py"
        helper.write_text('''import os,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from agent_gates.executor import execute
from agent_gates.locking import ResourceLock
root=Path(sys.argv[2])
with ResourceLock("parent-death", root/"locks") as lock:
    execute([sys.executable,"-I","-c", "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); Path('escaped-pid').write_text(str(p.pid)); time.sleep(60)"], cwd=root, env=dict(os.environ), timeout=60, log_dir=root/"logs", lease_fds=(lock.fileno(),))
''')
        host = subprocess.Popen([sys.executable, "-I", str(helper), str(Path(__file__).resolve().parents[1]), str(self.root)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: host.kill() if host.poll() is None else None)
        deadline = time.monotonic() + 5
        while not (self.root / "escaped-pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue((self.root / "escaped-pid").exists())
        pid = (self.root / "escaped-pid").read_text()
        host.kill()
        host.wait(timeout=5)
        terminal = self.root / "logs/process-result.json"
        while not terminal.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(terminal.exists())
        self.assertTrue(json.loads(terminal.read_text())["interrupted"])
        self.assertFalse(Path("/proc/" + pid).exists())
        while time.monotonic() < deadline:
            try:
                with ResourceLock("parent-death", self.root / "locks"):
                    break
            except LockBusy:
                time.sleep(0.01)
        else:
            self.fail("supervisor retained lease after terminal cleanup")


class SnapshotTests(CoreFixture):
    def test_git_bytes_ignore_clean_filter_and_cover_untracked(self):
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        subprocess.run(["git", "-C", str(self.source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        before = snapshot(self.source)
        (self.source / "input.txt").write_text("dirty")
        self.assertNotEqual(before["hash"], snapshot(self.source)["hash"])
        (self.source / "new.txt").write_text("untracked")
        after = snapshot(self.source)
        (self.source / "new.txt").write_text("new untracked bytes")
        self.assertNotEqual(after["hash"], snapshot(self.source)["hash"])

    def test_source_symlink_hashes_link_text_without_traversal(self):
        link = self.source / "link"
        link.symlink_to(self.script)
        before = snapshot(self.source)
        self.script.write_text("external contents do not belong to this tree")
        self.assertEqual(before, snapshot(self.source))
        link.unlink()
        link.symlink_to(self.root / "missing-fixture")
        self.assertNotEqual(before["hash"], snapshot(self.source)["hash"])


if __name__ == "__main__":
    unittest.main()
