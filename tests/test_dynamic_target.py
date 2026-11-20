"""The typed target/candidate request and its receipt binding.

A dynamic-target route names one workload and one candidate file as request
data. These tests pin the whole narrow contract: the grammar, the both-or-
neither rule, the environment the recipe actually receives, the candidate bytes
the receipt binds, the structured configuration observation, and the shared
device lease namespace that keeps V1 and V2 off one card at the same time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest.mock import patch

from agent_gates import api_types as api
from agent_gates import cli, dynamic_target, engine
from agent_gates import worker as worker_mod
from agent_gates.config import from_mapping
from agent_gates.engine import Engine
from agent_gates.evidence import EvidenceStore
from agent_gates.execution import ExecutionManager
from agent_gates.locking import ResourceLock, device_lock_root
from agent_gates.manifest import Manifest, digest
from agent_gates.registry import Registry
from agent_gates.verify_requirements import verify_requirements


TARGET = "level4/attention_residual"
CANDIDATE = "kernels/candidate.py"

SCRIPT = textwrap.dedent("""
    import json, os, sys, time

    target = os.environ.get("AGENT_GATES_TARGET")
    candidate = os.environ.get("AGENT_GATES_CANDIDATE")

    if "--check" in sys.argv:
        mode = os.environ.get("FIXTURE_CHECK", "ok")
        if mode == "observe":
            print('AGENT_GATES_CONFIG: {"category": "DEPENDENCY", "detail": "tilelang import failed"}',
                  file=sys.stderr)
            print("install the tilelang wheel", file=sys.stderr)
            raise SystemExit(3)
        if mode == "broken-observation":
            print('AGENT_GATES_CONFIG: {"category": "NONSENSE", "detail": "x"}', file=sys.stderr)
            print("plain human reason", file=sys.stderr)
            raise SystemExit(3)
        if mode == "slow":
            time.sleep(float(os.environ.get("FIXTURE_CHECK_SLEEP", "0.6")))
        if mode == "require":
            if target != "level4/attention_residual" or candidate != "kernels/candidate.py":
                print("missing target request", file=sys.stderr)
                raise SystemExit(4)
        raise SystemExit(0)

    if os.environ.get("FIXTURE_RECIPE_MODE") == "blocked":
        print('AGENT_GATES_CONFIG: {"category": "DEVICE", "detail": "card held by another job"}',
              file=sys.stderr)
        print("card held by another job", file=sys.stderr)
        raise SystemExit(75)

    if os.environ.get("FIXTURE_RECIPE_MODE") == "hold":
        open(os.environ["FIXTURE_HOLD_MARKER"], "w").write("held")
        while not os.path.exists(os.environ["FIXTURE_RELEASE_MARKER"]):
            time.sleep(0.02)

    print(json.dumps({
        "target": target,
        "candidate": candidate,
        "inheritedTarget": os.environ.get("FIXTURE_INHERITED_TARGET"),
    }))
""")


class _RunningSystemd:
    """The smallest systemd stand-in a submission needs to stay SUBMITTED."""

    def __init__(self):
        self.started = []
        self.running = set()

    def available(self):
        return True, "running"

    def start(self, execution_id):
        self.started.append(execution_id)
        self.running.add(execution_id)

    def stop(self, execution_id):
        self.running.discard(execution_id)

    def properties(self, execution_id, *names):
        state = "active" if execution_id in self.running else "inactive"
        return {"ActiveState": state, "SubState": state, "Result": "success"}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "kernels").mkdir(parents=True)
        self.candidate = self.source / CANDIDATE
        self.candidate.write_text("# candidate\n")
        self.pack = self.root / "registry/fixture-host"
        (self.pack / "scripts").mkdir(parents=True)
        self.script = self.pack / "scripts/gate.py"
        self.script.write_text(SCRIPT)
        self.engine = Engine(registry=Registry(self.root / "registry"),
                             store=EvidenceStore(self.root / "evidence"),
                             lock_root=self.root / "locks",
                             device_lock_root=self.root / "locks")

    def write(self, *, dynamic: bool | str = True, timeout=30, check_timeout=None,
              gate_id="fixture-gate", recipe_id="fixture.run", config_check=True,
              blocked=(75,)):
        recipe = {"id": recipe_id, "cwd": "source", "timeout": timeout,
                  "argv": [sys.executable, "-I", "{pack}/scripts/gate.py"]}
        if config_check:
            recipe["configCheck"] = [sys.executable, "-I", "{pack}/scripts/gate.py", "--check"]
        if blocked:
            recipe["blockedExitCodes"] = list(blocked)
        if dynamic is True:
            recipe["dynamicTarget"] = True
        elif dynamic == "optional":
            recipe["dynamicTarget"] = "optional"
        if check_timeout is not None:
            recipe["configCheckTimeout"] = check_timeout
        raw = {"schemaVersion": 1, "id": "fixture-host",
               "files": {"scripts/gate.py": digest(self.script)},
               "permissions": {"sourceWrite": False, "hostExecution": True},
               "recipes": [recipe],
               "gates": [{"id": gate_id, "gateType": "deterministic-test",
                          "recipeId": recipe_id,
                          "observationChecks": {"exitCodeZero": True}}]}
        (self.pack / "manifest.json").write_text(json.dumps(raw))

    def write_stages(self, modes):
        """A prep -> official graph, marking each recipe's dynamicTarget mode.

        `True` requires the pair, `"optional"` accepts it, and `False` leaves
        the recipe static, so one helper can build the real L4 shape as well as
        a mixed graph that proves the environment is published per recipe.
        """
        recipes, gates = [], []
        for name, mode in zip(("prep", "official"), modes):
            recipe = {"id": f"fixture.{name}", "cwd": "source", "timeout": 30,
                      "argv": [sys.executable, "-I", "{pack}/scripts/gate.py"]}
            if mode is True:
                recipe["dynamicTarget"] = True
            elif mode == "optional":
                recipe["dynamicTarget"] = "optional"
            recipes.append(recipe)
            gate = {"id": f"fixture-{name}", "gateType": "deterministic-test",
                    "recipeId": f"fixture.{name}",
                    "observationChecks": {"exitCodeZero": True}}
            if name == "official":
                gate["dependsOn"] = ["fixture-prep"]
            gates.append(gate)
        raw = {"schemaVersion": 1, "id": "fixture-host",
               "files": {"scripts/gate.py": digest(self.script)},
               "permissions": {"sourceWrite": False, "hostExecution": True},
               "recipes": recipes, "gates": gates}
        (self.pack / "manifest.json").write_text(json.dumps(raw))

    def run_fixture(self, *, target=TARGET, candidate=CANDIDATE, selected=None, **kwargs):
        self.write(**kwargs)
        return self.engine.run("fixture-host", self.source,
                               selected or ["fixture-gate"],
                               target=target, candidate=candidate)

    def manifest(self):
        return self.engine.registry.get("fixture-host")


class GrammarTests(unittest.TestCase):
    def test_target_accepts_canonical_slash_separated_segments(self):
        self.assertEqual(dynamic_target.validate_target(TARGET), TARGET)
        self.assertEqual(dynamic_target.validate_target("level4"), "level4")

    def test_target_refuses_path_syntax_and_empty_segments(self):
        for value in ("", "/absolute", "a/../b", "a//b", "a/", "a b",
                      "a\x00b", "a\\b", "x" * 257, None, 1):
            with self.subTest(value=value):
                with self.assertRaises(dynamic_target.TargetRequestError):
                    dynamic_target.validate_target(value)

    def test_candidate_refuses_noncanonical_paths(self):
        for value in ("", "/abs.py", "a/../b.py", "./a.py", "a//b.py", "a/",
                      "a\\b.py", "a\x00b.py", None, 1):
            with self.subTest(value=value):
                with self.assertRaises(dynamic_target.TargetRequestError):
                    dynamic_target.validate_candidate(value)

    def test_candidate_must_be_a_regular_file_under_the_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").mkdir()
            (root / "a/file.py").write_text("x")
            self.assertEqual(
                dynamic_target.candidate_reference(root, "a/file.py")["relativePath"],
                "a/file.py")
            for value in ("a", "absent.py"):
                with self.subTest(value=value):
                    with self.assertRaises(dynamic_target.TargetRequestError):
                        dynamic_target.candidate_reference(root, value)
            (root / "link.py").symlink_to(root / "a/file.py")
            with self.assertRaises(dynamic_target.TargetRequestError):
                dynamic_target.candidate_reference(root, "link.py")


class RequestResolutionTests(Fixture):
    def setUp(self):
        super().setUp()
        self.write()

    def test_a_dynamic_route_requires_both_halves(self):
        for target, candidate in ((None, None), (TARGET, None), (None, CANDIDATE)):
            with self.subTest(target=target, candidate=candidate):
                with self.assertRaises(dynamic_target.TargetRequestError):
                    self.engine.run("fixture-host", self.source, ["fixture-gate"],
                                    target=target, candidate=candidate)

    def test_a_static_route_refuses_the_pair(self):
        self.write(dynamic=False)
        with self.assertRaises(dynamic_target.TargetRequestError):
            self.engine.run("fixture-host", self.source, ["fixture-gate"],
                            target=TARGET, candidate=CANDIDATE)

    def test_doctor_requires_the_pair_only_for_an_explicit_dynamic_scope(self):
        with self.assertRaises(dynamic_target.TargetRequestError):
            self.engine.doctor("fixture-host", self.source, selected=["fixture-gate"])
        report = self.engine.doctor("fixture-host", self.source,
                                    selected=["fixture-gate"],
                                    target=TARGET, candidate=CANDIDATE)
        self.assertEqual(report["status"], "PASS", report["failures"])

    def test_manifest_refuses_dynamic_target_with_a_pack_cwd(self):
        raw = json.loads((self.pack / "manifest.json").read_text())
        raw["recipes"][0]["cwd"] = "pack"
        (self.pack / "manifest.json").write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            Manifest(self.pack / "manifest.json")

    def test_manifest_refuses_a_false_dynamic_target_and_an_orphan_timeout(self):
        raw = json.loads((self.pack / "manifest.json").read_text())
        raw["recipes"][0]["dynamicTarget"] = False
        (self.pack / "manifest.json").write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            Manifest(self.pack / "manifest.json")
        raw["recipes"][0]["dynamicTarget"] = True
        raw["recipes"][0].pop("configCheck")
        raw["recipes"][0]["configCheckTimeout"] = 5
        (self.pack / "manifest.json").write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            Manifest(self.pack / "manifest.json")


class RunTests(Fixture):
    def test_the_receipt_binds_the_target_and_the_candidate_bytes(self):
        run = self.run_fixture()
        self.assertEqual(run["status"], "PASS", run["gates"])
        bound = run["dynamicTarget"]
        self.assertEqual(bound["target"], TARGET)
        self.assertEqual(bound["candidate"]["relativePath"], CANDIDATE)
        self.assertEqual(bound["candidate"]["sha256"],
                         digest(self.candidate))
        self.assertEqual(bound["candidate"]["bytes"], self.candidate.stat().st_size)
        self.assertEqual(self.engine.store.verify(run["runId"],
                                                  registry=self.engine.registry)["status"],
                         "PASS")

    def test_the_recipe_receives_the_pair_only_because_it_declares_it(self):
        run = self.run_fixture()
        observations = run["gates"][0]["observations"]
        self.assertEqual(observations["target"], TARGET)
        self.assertEqual(observations["candidate"], CANDIDATE)

    def test_a_launching_shell_cannot_inject_the_pair(self):
        with patch.dict(os.environ, {"AGENT_GATES_TARGET": "level4/from-shell",
                                     "AGENT_GATES_CANDIDATE": "from-shell.py"}, clear=False):
            run = self.run_fixture()
        observations = run["gates"][0]["observations"]
        self.assertEqual(observations["target"], TARGET)
        self.assertEqual(observations["candidate"], CANDIDATE)

    def test_the_configuration_check_is_probed_with_the_request(self):
        with patch.dict(os.environ, {"FIXTURE_CHECK": "require"}, clear=False):
            run = self.run_fixture()
        self.assertEqual(run["status"], "PASS", run["gates"])

    def test_a_static_recipe_never_sees_the_request_environment(self):
        self.write(dynamic=False)
        (self.source / "static.py").write_text("x")
        run = self.engine.run("fixture-host", self.source, ["fixture-gate"])
        observations = run["gates"][0]["observations"]
        self.assertIsNone(observations["target"])
        self.assertIsNone(observations["candidate"])

    def test_changing_the_candidate_makes_the_receipt_stale(self):
        run = self.run_fixture()
        self.candidate.write_text("# replaced\n")
        result = self.engine.store.verify(run["runId"],
                                          registry=self.engine.registry)
        self.assertEqual(result["status"], "STALE")
        self.assertIn("CANDIDATE_IDENTITY_CHANGED", result["failures"])

    def test_deleting_the_candidate_makes_the_receipt_stale(self):
        run = self.run_fixture()
        self.candidate.unlink()
        result = self.engine.store.verify(run["runId"],
                                          registry=self.engine.registry)
        self.assertEqual(result["status"], "STALE")
        self.assertIn("CANDIDATE_IDENTITY_CHANGED", result["failures"])

    def test_unrelated_source_change_still_reports_the_source(self):
        run = self.run_fixture()
        (self.source / "other.txt").write_text("changed\n")
        result = self.engine.store.verify(run["runId"],
                                          registry=self.engine.registry)
        self.assertEqual(result["status"], "STALE")
        self.assertIn("SOURCE_IDENTITY_CHANGED", result["failures"])


class OptionalRequestTests(Fixture):
    """The managed fixed-target route accepts a pair but never demands one."""

    def test_a_managed_fixed_target_route_runs_without_a_pair(self):
        self.write(dynamic="optional")
        run = self.engine.run("fixture-host", self.source, ["fixture-gate"])
        self.assertEqual(run["status"], "PASS", run["gates"])
        self.assertNotIn("dynamicTarget", run)
        observations = run["gates"][0]["observations"]
        self.assertIsNone(observations["target"])
        self.assertIsNone(observations["candidate"])

    def test_a_launching_shell_cannot_supply_the_pair_to_an_optional_route(self):
        self.write(dynamic="optional")
        with patch.dict(os.environ, {"AGENT_GATES_TARGET": "level4/from-shell",
                                     "AGENT_GATES_CANDIDATE": "from-shell.py"}, clear=False):
            run = self.engine.run("fixture-host", self.source, ["fixture-gate"])
        observations = run["gates"][0]["observations"]
        self.assertIsNone(observations["target"])
        self.assertIsNone(observations["candidate"])

    def test_the_managed_prep_official_graph_runs_without_a_pair(self):
        self.write_stages(("optional", "optional"))
        run = self.engine.run("fixture-host", self.source, ["fixture-official"])
        self.assertEqual(run["status"], "PASS", run["gates"])
        self.assertEqual([gate["gateId"] for gate in run["gates"]],
                         ["fixture-prep", "fixture-official"])
        self.assertNotIn("dynamicTarget", run)
        for gate in run["gates"]:
            self.assertIsNone(gate["observations"]["target"])

    def test_an_explicit_pair_is_effective_in_both_optional_stages(self):
        self.write_stages(("optional", "optional"))
        run = self.engine.run("fixture-host", self.source, ["fixture-official"],
                              target=TARGET, candidate=CANDIDATE)
        self.assertEqual(run["status"], "PASS", run["gates"])
        self.assertEqual([gate["gateId"] for gate in run["gates"]],
                         ["fixture-prep", "fixture-official"])
        for gate in run["gates"]:
            self.assertEqual(gate["observations"]["target"], TARGET)
            self.assertEqual(gate["observations"]["candidate"], CANDIDATE)
        self.assertEqual(run["dynamicTarget"]["candidate"]["sha256"],
                         digest(self.candidate))
        self.assertEqual(self.engine.store.verify(run["runId"],
                                                  registry=self.engine.registry)["status"],
                         "PASS")

    def test_a_static_recipe_in_the_same_graph_does_not_see_the_pair(self):
        self.write_stages(("optional", False))
        run = self.engine.run("fixture-host", self.source, ["fixture-official"],
                              target=TARGET, candidate=CANDIDATE)
        self.assertEqual(run["status"], "PASS", run["gates"])
        prep, official = run["gates"]
        self.assertEqual(prep["observations"]["target"], TARGET)
        self.assertIsNone(official["observations"]["target"])
        self.assertIsNone(official["observations"]["candidate"])

    def test_a_replaced_candidate_makes_an_optional_receipt_stale(self):
        self.write(dynamic="optional")
        run = self.engine.run("fixture-host", self.source, ["fixture-gate"],
                              target=TARGET, candidate=CANDIDATE)
        self.assertEqual(run["status"], "PASS", run["gates"])
        self.candidate.write_text("# replaced\n")
        result = self.engine.store.verify(run["runId"],
                                          registry=self.engine.registry)
        self.assertEqual(result["status"], "STALE")
        self.assertIn("CANDIDATE_IDENTITY_CHANGED", result["failures"])

    def test_optional_still_requires_a_source_recipe(self):
        self.write(dynamic="optional")
        raw = json.loads((self.pack / "manifest.json").read_text())
        raw["recipes"][0]["cwd"] = "pack"
        (self.pack / "manifest.json").write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            Manifest(self.pack / "manifest.json")

    def test_manifest_refuses_other_dynamic_target_values(self):
        self.write(dynamic="optional")
        for value in ("yes", 1):
            with self.subTest(value=value):
                raw = json.loads((self.pack / "manifest.json").read_text())
                raw["recipes"][0]["dynamicTarget"] = value
                (self.pack / "manifest.json").write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    Manifest(self.pack / "manifest.json")

    def test_an_optional_scope_still_accepts_doctor_without_the_pair(self):
        self.write(dynamic="optional")
        report = self.engine.doctor("fixture-host", self.source,
                                    selected=["fixture-gate"])
        self.assertEqual(report["status"], "PASS", report["failures"])


class ConfigObservationTests(Fixture):
    def test_a_structured_observation_is_recorded_beside_the_human_reason(self):
        with patch.dict(os.environ, {"FIXTURE_CHECK": "observe"}, clear=False):
            run = self.run_fixture()
        gate = run["gates"][0]
        self.assertEqual(gate["status"], "BLOCKED_ENV")
        self.assertEqual(gate["failures"], ["CONFIG_CHECK_FAILED"])
        self.assertEqual(gate["configReason"], "install the tilelang wheel")
        self.assertEqual(gate["configObservation"],
                         {"category": "DEPENDENCY",
                          "detail": "tilelang import failed"})

    def test_an_unknown_category_is_not_an_observation(self):
        with patch.dict(os.environ, {"FIXTURE_CHECK": "broken-observation"}, clear=False):
            run = self.run_fixture()
        gate = run["gates"][0]
        self.assertEqual(gate["status"], "BLOCKED_ENV")
        self.assertEqual(gate["configReason"], "plain human reason")
        self.assertNotIn("configObservation", gate)

    def test_a_blocked_exit_code_keeps_the_human_reason_without_a_preflight_category(self):
        # The observation grammar describes the configuration PREFLIGHT. A
        # recipe's own blocked exit happens during execution, so even when the
        # Adapter prints an observation line the Gate must not relabel an
        # execution-time refusal with a preflight category.
        with patch.dict(os.environ, {"FIXTURE_RECIPE_MODE": "blocked"}, clear=False):
            run = self.run_fixture()
        gate = run["gates"][0]
        self.assertEqual(gate["status"], "BLOCKED_ENV")
        self.assertEqual(gate["failures"], ["BLOCKED_EXIT_CODE: 75"])
        self.assertEqual(gate["configReason"], "card held by another job")
        self.assertNotIn("configObservation", gate)

    def test_the_configuration_check_has_an_independent_timeout(self):
        with patch.dict(os.environ, {"FIXTURE_CHECK": "slow",
                                     "FIXTURE_CHECK_SLEEP": "0.6"}, clear=False):
            run = self.run_fixture(timeout=0.1, check_timeout=5)
        self.assertEqual(run["status"], "PASS", run["gates"])
        with patch.dict(os.environ, {"FIXTURE_CHECK": "slow",
                                     "FIXTURE_CHECK_SLEEP": "0.6"}, clear=False):
            run = self.run_fixture(timeout=0.1)
        self.assertEqual(run["gates"][0]["status"], "BLOCKED_ENV")
        self.assertTrue(run["gates"][0]["configCheck"]["timedOut"])


class SharedDeviceLockTests(Fixture):
    def setUp(self):
        super().setUp()
        self.write()
        # The request pair is irrelevant here; only the lease namespace is.
        self.env = {"FIXTURE_DEVICE_ID": "3"}
        self.device = patch.dict(engine.DEVICE_ADAPTERS,
                                 {"fixture-host": "FIXTURE_DEVICE_ID"}, clear=False)
        self.device.start()
        self.addCleanup(self.device.stop)

    def test_the_engine_leases_the_card_in_its_own_namespace(self):
        configured = self.root / "configured-locks"
        host = self.root / "host-locks"
        engine_with_split_roots = Engine(
            registry=self.engine.registry, store=self.engine.store,
            lock_root=configured, device_lock_root=host)
        with patch.dict(os.environ, self.env, clear=False):
            with ResourceLock("npu:3", configured):
                # A lock in the evidence/state namespace is not the card.
                run = engine_with_split_roots.run(
                    "fixture-host", self.source, ["fixture-gate"],
                    device_id="3", target=TARGET, candidate=CANDIDATE)
                self.assertEqual(run["status"], "PASS", run["gates"])
            with ResourceLock("npu:3", host):
                run = engine_with_split_roots.run(
                    "fixture-host", self.source, ["fixture-gate"],
                    device_id="3", target=TARGET, candidate=CANDIDATE)
                self.assertEqual(run["status"], "BLOCKED_ENV")

    def test_the_environment_override_is_the_one_shared_root(self):
        # No explicit device_lock_root: the canonical resolver must read the
        # operator override, exactly as a real V1 or V2 process does.
        engine_using_override = Engine(registry=self.engine.registry,
                                       store=self.engine.store,
                                       lock_root=self.root / "configured-locks")
        with patch.dict(os.environ, {**self.env,
                                     "AGENT_GATES_DEVICE_LOCK_ROOT": str(self.root / "host-locks")},
                        clear=False):
            with ResourceLock("npu:3", self.root / "host-locks"):
                run = engine_using_override.run(
                    "fixture-host", self.source, ["fixture-gate"],
                    device_id="3", target=TARGET, candidate=CANDIDATE)
        self.assertEqual(run["status"], "BLOCKED_ENV")
        self.assertTrue(all(gate["status"] == "BLOCKED_ENV" for gate in run["gates"]))

    def test_the_override_must_be_absolute(self):
        with self.assertRaises(ValueError):
            device_lock_root({"AGENT_GATES_DEVICE_LOCK_ROOT": "relative/locks"})

    def test_v1_holding_the_card_blocks_v2_through_the_managed_root(self):
        """One real V1 run and one real V2 worker must share one card lock.

        Both sides resolve the canonical managed root (the single environment
        override), never a private directory of their own entry point. V1 runs
        the actual Engine path in a thread and holds `npu:3` while its recipe is
        alive; the V2 worker, on a different source, must not reach the recipe
        until V1 releases. No NPU is involved: the recipe is a Python fixture.
        """
        self.write()
        managed = self.root / "managed-device-locks"
        managed.mkdir()
        other = self.root / "source2"
        (other / "kernels").mkdir(parents=True)
        (other / CANDIDATE).write_text("# second candidate\n")
        (self.root / "state2").mkdir()
        (self.root / "evidence2").mkdir()
        config = from_mapping({
            "schema_version": 1,
            "state_root": str(self.root / "state2"),
            "evidence_root": str(self.root / "evidence2"),
            "adapters": {"fixture-host": {}},
            "workspaces": {
                "ws1": {"source_root": str(self.source), "adapters": ["fixture-host"]},
                "ws2": {"source_root": str(other), "adapters": ["fixture-host"]},
            },
            "worker": {"backend": "systemd", "max_concurrent": 2,
                       "python": sys.executable},
        }, path=self.root / "config.toml")
        manager = ExecutionManager(config, registry=self.engine.registry,
                                   systemd=_RunningSystemd())
        hold = self.root / "held.marker"
        release = self.root / "release.marker"
        environment = {
            "AGENT_GATES_DEVICE_LOCK_ROOT": str(managed),
            "FIXTURE_DEVICE_ID": "3",
            "FIXTURE_RECIPE_MODE": "hold",
            "FIXTURE_HOLD_MARKER": str(hold),
            "FIXTURE_RELEASE_MARKER": str(release),
        }
        v1 = Engine(registry=self.engine.registry,
                    store=EvidenceStore(self.root / "evidence-v1"),
                    lock_root=self.root / "locks-v1")
        outcomes = {}

        def _run_v1():
            try:
                outcomes["run"] = v1.run("fixture-host", self.source,
                                         ["fixture-gate"],
                                         target=TARGET, candidate=CANDIDATE)
            except BaseException as exc:  # surfaced by the assertion below
                outcomes["error"] = exc

        def _request():
            return api.ExecutionRequest(
                workspace="ws2", adapter="fixture-host", gates=["fixture-gate"],
                target=TARGET, candidate=CANDIDATE)

        with patch.dict(os.environ, environment, clear=False):
            thread = threading.Thread(target=_run_v1, daemon=True)
            thread.start()
            deadline = time.monotonic() + 20
            while not hold.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(hold.exists(), "V1 never opened its device lease")

            status = manager.submit(_request())
            v2 = Engine(registry=self.engine.registry,
                        store=EvidenceStore(config.evidence_root),
                        lock_root=config.lock_root)
            busy = worker_mod.run_execution(status.execution_id,
                                            store=manager.store,
                                            config=config, engine=v2)
            self.assertEqual(busy["state"], "FINISHED", busy)
            self.assertEqual(busy["gateStatus"], "BLOCKED_ENV", busy)
            receipt = manager.evidence.read(busy["runId"])
            self.assertNotIn("PASS", [gate["status"] for gate in receipt["gates"]])
            self.assertIn("resource busy: npu:3",
                          " ".join(failure for gate in receipt["gates"]
                                   for failure in gate["failures"]))

            release.write_text("go")
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive())
            self.assertNotIn("error", outcomes)
            self.assertEqual(outcomes["run"]["status"], "PASS",
                             outcomes["run"]["gates"])

            after = manager.submit(_request())
            released = worker_mod.run_execution(after.execution_id,
                                                store=manager.store,
                                                config=config, engine=v2)
        self.assertEqual(released["state"], "FINISHED", released)
        self.assertEqual(released["gateStatus"], "PASS", released)


class VerifyRequirementTests(Fixture):
    def setUp(self):
        super().setUp()
        self.execution = self.run_fixture()
        self.assertEqual(self.execution["status"], "PASS", self.execution["gates"])
        self.config = from_mapping({
            "schema_version": 1,
            "state_root": str(self.root / "state"),
            "evidence_root": str(self.root / "evidence"),
            "adapters": {"fixture-host": {}},
            "workspaces": {"ws": {"source_root": str(self.source),
                                  "adapters": ["fixture-host"]}},
            "worker": {"backend": "systemd", "max_concurrent": 2,
                       "python": sys.executable},
        }, path=self.root / "config.toml")

    def requirement(self, **overrides):
        values = {"workspace": "ws", "adapter": "fixture-host",
                  "required_gates": ["fixture-gate"], "run_id": self.execution["runId"],
                  "target": TARGET, "candidate": CANDIDATE}
        values.update(overrides)
        return api.VerificationRequirement(**values)

    def verify(self, requirement):
        return verify_requirements(requirement, config=self.config,
                                   registry=self.engine.registry,
                                   store=self.engine.store)

    def test_the_bound_pair_satisfies_the_requirement(self):
        result = self.verify(self.requirement())
        self.assertTrue(result.satisfied, result.failures)
        self.assertEqual(result.target, TARGET)
        self.assertEqual(result.candidate, CANDIDATE)

    def test_a_different_target_is_refused(self):
        result = self.verify(self.requirement(target="level3/other"))
        self.assertFalse(result.satisfied)
        self.assertTrue(any(item.startswith("TARGET_MISMATCH") for item in result.failures),
                        result.failures)

    def test_a_different_candidate_is_refused(self):
        result = self.verify(self.requirement(candidate="kernels/other.py"))
        self.assertFalse(result.satisfied)
        self.assertTrue(any(item.startswith("CANDIDATE_MISMATCH") for item in result.failures),
                        result.failures)

    def test_a_requirement_without_a_pair_does_not_constrain_one(self):
        result = self.verify(self.requirement(target=None, candidate=None))
        self.assertTrue(result.satisfied, result.failures)
        self.assertIsNone(result.target)

    def test_half_a_pair_is_refused_before_the_receipt(self):
        result = self.verify(self.requirement(candidate=None))
        self.assertFalse(result.satisfied)
        self.assertIn("TARGET_CANDIDATE_PAIR_REQUIRED", result.failures)

    def test_a_receipt_without_a_binding_cannot_satisfy_the_pair(self):
        self.write(dynamic=False)
        (self.source / "static.py").write_text("x")
        static = self.engine.run("fixture-host", self.source, ["fixture-gate"])
        result = self.verify(self.requirement(run_id=static["runId"]))
        self.assertFalse(result.satisfied)
        self.assertIn("TARGET_NOT_BOUND", result.failures)


class CliTests(Fixture):
    def setUp(self):
        super().setUp()
        self.write()
        self.patch = patch.object(cli, "Registry", lambda: self.engine.registry)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def invoke(self, *extra):
        return cli.main(["run", "--json", "--evidence-root", str(self.root / "evidence"),
                         "--adapter", "fixture-host", "--source", str(self.source),
                         "--gate", "fixture-gate", *extra])

    def test_half_a_pair_is_an_invalid_request_not_a_machine_verdict(self):
        from io import StringIO
        from contextlib import redirect_stdout
        output = StringIO()
        with redirect_stdout(output):
            code = self.invoke("--target", TARGET)
        payload = json.loads(output.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "INVALID_REQUEST")
        self.assertIn("together", payload["failures"][0])

    def test_a_grammar_error_is_an_invalid_request(self):
        from io import StringIO
        from contextlib import redirect_stdout
        output = StringIO()
        with redirect_stdout(output):
            code = self.invoke("--target", "../escape", "--candidate", CANDIDATE)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "INVALID_REQUEST")

    def test_the_pair_runs_through_the_cli(self):
        from io import StringIO
        from contextlib import redirect_stdout
        output = StringIO()
        with redirect_stdout(output):
            code = self.invoke("--target", TARGET, "--candidate", CANDIDATE)
        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0, payload)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["dynamicTarget"]["target"], TARGET)


if __name__ == "__main__":
    unittest.main()
