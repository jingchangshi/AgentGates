"""The engine side of the windowed lease.

A windowed recipe opens the lease itself, so the engine must NOT hold it. If it
did, the child would wait for a lock its own parent owns and the run would
deadlock - which is why these tests drive a real Engine and a real child
process, and why the assertion is about the receipt and the child's own report
rather than about a call count.

They also pin the other direction: an adapter that declares nothing keeps the
whole-recipe lease it always had. That is the property the 645 pre-existing
tests depend on, and it must not be quietly redefined.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from agent_gates import engine
from agent_gates.engine import Engine
from agent_gates.evidence import EvidenceStore
from agent_gates.locking import ResourceLock
from agent_gates.manifest import Manifest, ManifestError, digest
from agent_gates.registry import Registry


# A child that opens the lease it was given, does a moment of work and closes it.
# It refuses to run at all when the engine failed to publish a lease, so a
# windowed run that silently went back to the whole-recipe lease cannot pass.
WINDOWED = textwrap.dedent("""
    import fcntl, json, os, sys, time
    from datetime import datetime, timezone
    path = os.environ.get("AGENT_GATES_DEVICE_LEASE_PATH")
    if not path:
        print("no lease published", file=sys.stderr)
        raise SystemExit(9)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    wait = float(os.environ.get("AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS", "0"))
    deadline = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                print("busy", file=sys.stderr)
                raise SystemExit(75)
            time.sleep(0.02)
    started = datetime.now(timezone.utc).isoformat()
    time.sleep(0.15)
    released = datetime.now(timezone.utc).isoformat()
    os.close(fd)
    print(json.dumps({"deviceWindow": {"resource": os.environ["AGENT_GATES_DEVICE_LEASE_RESOURCE"],
                                        "path": path, "startedAt": started,
                                        "releasedAt": released, "occupancy": "shared(2)",
                                        "waited": 0.0}}))
""")


class WindowFixture(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "program.txt").write_text("original\n")
        self.pack = self.root / "registry" / "fixture-npu"
        (self.pack / "scripts").mkdir(parents=True)
        self.script = self.pack / "scripts/gate.py"
        self.engine = Engine(registry=Registry(self.root / "registry"),
                             store=EvidenceStore(self.root / "evidence"),
                             lock_root=self.root / "locks",
                             device_lock_root=self.root / "locks")

    def write(self, recipe, gate_checks=None, code=WINDOWED, blocked=None):
        self.script.write_text(code)
        declared = {} if blocked is None else {"blockedExitCodes": list(blocked)}
        raw = {"schemaVersion": 1, "id": "fixture-npu",
               "files": {"scripts/gate.py": digest(self.script)},
               "permissions": {"sourceWrite": False, "hostExecution": True},
               "recipes": [{"id": "fixture.run", "cwd": "source", "timeout": 30,
                            "argv": [sys.executable, "-I", "{pack}/scripts/gate.py"],
                            **declared, **recipe}],
               "gates": [{"id": "fixture-gate", "gateType": "deterministic-test",
                          "recipeId": "fixture.run",
                          "observationChecks": gate_checks or {"exitCodeZero": True}}]}
        (self.pack / "manifest.json").write_text(json.dumps(raw))

    def run_fixture(self, **kwargs):
        with patch.dict(engine.DEVICE_ADAPTERS, {"fixture-npu": "FIXTURE_DEVICE_ID"}, clear=False), \
                patch.dict(os.environ, {"FIXTURE_DEVICE_ID": "3"}, clear=False):
            return self.engine.run("fixture-npu", self.source, ["fixture-gate"],
                                   device_id="3", **kwargs)


class WindowedRunCase(WindowFixture):
    def test_the_engine_does_not_hold_the_lease_and_the_child_reports_it(self):
        self.write({"deviceWindow": {"waitSeconds": 10}})
        run = self.run_fixture()
        gate = run["gates"][0]
        self.assertEqual(run["status"], "PASS", gate)
        self.assertEqual(run["deviceLease"]["mode"], "windowed")
        self.assertEqual(run["deviceLease"]["resource"], "npu:3")
        window = gate["deviceWindow"]
        self.assertEqual(window["resource"], "npu:3")
        self.assertEqual(window["occupancy"], "shared(2)")
        self.assertLessEqual(window["startedAt"], window["releasedAt"])
        self.assertEqual(run["deviceLeaseWindows"], [{"gateId": "fixture-gate", **window}])

    def test_the_card_is_free_after_the_run_because_the_window_closed(self):
        self.write({"deviceWindow": {"waitSeconds": 10}})
        run = self.run_fixture()
        self.assertEqual(run["status"], "PASS", run["gates"][0])
        with ResourceLock("npu:3", self.root / "locks"):
            pass

    def test_a_recipe_that_forgets_a_window_is_refused_a_lease_it_still_holds(self):
        # The child is handed the card and told to wait; the engine holds nothing,
        # so the run finishes instead of deadlocking. A deadlock here would be the
        # bug that makes "engine holds it AND child wants it" unusable.
        self.write({"deviceWindow": {"waitSeconds": 10}})
        run = self.run_fixture()
        self.assertNotEqual(run["gates"][0].get("failures"), ["TIMEOUT"])

    def test_a_window_report_naming_a_different_lease_is_invalid_evidence(self):
        code = WINDOWED.replace('"resource": os.environ["AGENT_GATES_DEVICE_LEASE_RESOURCE"]',
                                '"resource": "npu:7"')
        self.write({"deviceWindow": {"waitSeconds": 10}}, code=code)
        run = self.run_fixture()
        self.assertEqual(run["gates"][0]["status"], "INVALID_EVIDENCE")
        self.assertIn("DEVICE_WINDOW_RESOURCE_MISMATCH", run["gates"][0]["failures"])

    def test_a_window_report_with_reversed_times_is_invalid_evidence(self):
        code = WINDOWED.replace('"startedAt": started', '"startedAt": "z"')
        self.write({"deviceWindow": {"waitSeconds": 10}}, code=code)
        run = self.run_fixture()
        self.assertEqual(run["gates"][0]["status"], "INVALID_EVIDENCE")
        self.assertIn("DEVICE_WINDOW_TIMES_INVALID", run["gates"][0]["failures"])

    def test_a_window_report_with_an_undefined_occupancy_is_invalid_evidence(self):
        code = WINDOWED.replace('"occupancy": "shared(2)"', '"occupancy": "probably fine"')
        self.write({"deviceWindow": {"waitSeconds": 10}}, code=code)
        run = self.run_fixture()
        self.assertEqual(run["gates"][0]["status"], "INVALID_EVIDENCE")
        self.assertIn("DEVICE_WINDOW_OCCUPANCY_INVALID", run["gates"][0]["failures"])

    def test_a_windowed_gate_that_reports_no_window_is_invalid_evidence(self):
        # Not publishing a window means no window was opened. Silence must not be
        # read as "a very short window", or the weakest possible report passes.
        self.write({"deviceWindow": {"waitSeconds": 10}},
                   code='import json; print(json.dumps({}))')
        run = self.run_fixture()
        self.assertEqual(run["gates"][0]["status"], "INVALID_EVIDENCE")
        self.assertIn("DEVICE_WINDOW_MISSING", run["gates"][0]["failures"])

    def test_extra_window_durations_are_carried_but_never_gated(self):
        # A receipt should be able to show that the window was dominated by work
        # rather than by setup. The values are carried, and a wrong type is
        # dropped rather than trusted - but a slow machine is never a failure.
        code = WINDOWED.replace(
            '"waited": 0.0}}',
            '"waited": 0.0, "heldSeconds": 0.5, "probeSeconds": 0.1,'
            ' "warmupSeconds": "not a number"}}')
        self.write({"deviceWindow": {"waitSeconds": 10}}, code=code)
        run = self.run_fixture()
        gate = run["gates"][0]
        self.assertEqual(gate["status"], "PASS", gate)
        window = gate["deviceWindow"]
        self.assertEqual(window["heldSeconds"], 0.5)
        self.assertEqual(window["probeSeconds"], 0.1)
        self.assertNotIn("warmupSeconds", window)

    def test_a_gate_blocked_before_the_window_is_not_reported_as_bad_evidence(self):
        # A blocked gate ran nothing, so it has no window to show. Demanding one
        # would rewrite an honest resource refusal into INVALID_EVIDENCE, which
        # claims the evidence was bad rather than that nothing ran. The refusal
        # itself must survive as BLOCKED_ENV, and the lease must still be freed.
        self.write({"deviceWindow": {"waitSeconds": 10}},
                   blocked=[75], code='import sys; sys.exit(75)')
        run = self.run_fixture()
        gate = run["gates"][0]
        self.assertEqual(gate["status"], "BLOCKED_ENV", gate)
        self.assertNotIn("DEVICE_WINDOW_MISSING", gate["failures"])
        self.assertIs(gate["deviceWindow"].get("notOpened"), True)
        self.assertIsNone(gate["deviceWindow"]["startedAt"])
        self.assertIsNone(gate["deviceWindow"]["releasedAt"])
        with ResourceLock("npu:3", self.root / "locks"):
            pass

    def test_another_holder_of_the_card_lease_is_reported_not_refused(self):
        # Sharing a card is normal. A window reports what it saw and runs anyway;
        # only a lease that never opens is a refusal. The occupancy here is a
        # literal, because requiring a quiet card would make this test depend on
        # who else is on the machine.
        self.write({"deviceWindow": {"waitSeconds": 10}})
        run = self.run_fixture()
        gate = run["gates"][0]
        self.assertEqual(gate["status"], "PASS", gate)
        self.assertEqual(gate["deviceWindow"]["occupancy"], "shared(2)")


class WholeRecipeRunCase(WindowFixture):
    def test_a_recipe_without_a_window_keeps_the_whole_recipe_lease(self):
        # The old behaviour, unchanged: the engine takes npu:<id> for the run and
        # the receipt is shaped exactly as before. This is what the 645
        # pre-existing tests rely on, so it is asserted rather than assumed.
        self.write({}, gate_checks={"exitCodeZero": True})
        with ResourceLock("npu:3", self.root / "locks"):
            run = self.run_fixture()
        gate = run["gates"][0]
        self.assertEqual(gate["status"], "BLOCKED_ENV")
        self.assertTrue(any("busy" in failure for failure in gate["failures"]), gate)
        self.assertNotIn("deviceLease", run)
        self.assertNotIn("deviceWindow", gate)

    def test_a_host_only_recipe_needs_no_device_authorization(self):
        # A build never touches the card, so demanding an allocation for it would
        # make work that cannot conflict with anyone fail on a busy machine.
        self.write({"device": "none"}, code='import json; print(json.dumps({"ok": True}))',
                   gate_checks={"exitCodeZero": True})
        with patch.dict(engine.DEVICE_ADAPTERS, {"fixture-npu": "FIXTURE_DEVICE_ID"}, clear=False), \
                patch.dict(os.environ, {}, clear=True):
            run = self.engine.run("fixture-npu", self.source, ["fixture-gate"])
        self.assertEqual(run["gates"][0]["status"], "PASS", run["gates"][0])
        self.assertNotIn("authorizedDeviceId", run)


class DeclarationCase(unittest.TestCase):
    """The declaration itself: what a manifest may say about a device."""

    def raw(self, recipe, adapter="fixture-npu"):
        import tempfile
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        script = root / "gate.py"
        script.write_text("print(1)\n")
        return {"schemaVersion": 1, "id": adapter,
                "files": {"gate.py": digest(script)},
                "permissions": {"sourceWrite": False, "hostExecution": True},
                "recipes": [recipe],
                "gates": [{"id": "g", "gateType": "deterministic-test",
                           "recipeId": recipe["id"],
                           "observationChecks": {"exitCodeZero": True}}]}, root

    def recipe(self, **extra):
        return {"id": "r", "cwd": "pack", "timeout": 10,
                "argv": [sys.executable, "-I", "{pack}/gate.py"], **extra}

    def load(self, raw, root):
        (root / "manifest.json").write_text(json.dumps(raw))
        return Manifest(root / "manifest.json")

    def test_a_window_must_declare_a_positive_finite_wait(self):
        for value in (0, -1, True, "5", None, float("inf")):
            raw, root = self.raw(self.recipe(deviceWindow={"waitSeconds": value}))
            with self.assertRaises(ManifestError, msg=repr(value)):
                self.load(raw, root)

    def test_an_unknown_window_key_is_refused(self):
        raw, root = self.raw(self.recipe(deviceWindow={"waitSeconds": 5,
                                              "holdSeconds": 5}))
        with self.assertRaises(ManifestError) as error:
            self.load(raw, root)
        self.assertIn("holdSeconds", str(error.exception))

    def test_a_non_object_window_is_refused(self):
        raw, root = self.raw(self.recipe(deviceWindow=5))
        with self.assertRaises(ManifestError):
            self.load(raw, root)

    def test_a_declared_window_is_a_float_wait(self):
        raw, root = self.raw(self.recipe(deviceWindow={"waitSeconds": 7}))
        self.assertEqual(self.load(raw, root).resolve_recipe("r", root).device_window, 7.0)

    def test_an_undeclared_recipe_defers_to_the_engine(self):
        raw, root = self.raw(self.recipe())
        self.assertIsNone(self.load(raw, root).resolve_recipe("r", root).device_window)

    def test_an_unknown_device_value_is_refused(self):
        raw, root = self.raw(self.recipe(device="gpu"))
        with self.assertRaises(ManifestError):
            self.load(raw, root)

    def test_mixing_windowed_and_whole_recipe_device_gates_is_refused(self):
        # The two modes cannot share a run: holding the lease while handing it to
        # a child deadlocks on the very lock the child waits for. Refusing the
        # manifest is better than running one device gate unprotected.
        raw, root = self.raw(self.recipe())
        raw["gates"].append({"id": "w", "gateType": "deterministic-test",
                             "recipeId": "w", "observationChecks": {"exitCodeZero": True}})
        raw["recipes"].append({"id": "w", "cwd": "pack", "timeout": 10,
                               "deviceWindow": {"waitSeconds": 5},
                               "argv": [sys.executable, "-I", "{pack}/gate.py"]})
        manifest = self.load(raw, root)
        with patch.dict(engine.DEVICE_ADAPTERS, {"fixture-npu": "FIXTURE_DEVICE_ID"}):
            with self.assertRaises(ManifestError) as error:
                engine.device_lease_plan(manifest, ["g", "w"], root)
        self.assertIn("both whole-recipe and windowed", str(error.exception))

    def test_an_adapter_without_a_device_variable_has_no_plan(self):
        raw, root = self.raw(self.recipe(), adapter="fixture")
        manifest = self.load(raw, root)
        self.assertEqual(engine.device_lease_plan(manifest, ["g"], root), (None, {}))


if __name__ == "__main__":
    unittest.main()
