"""Execution manager semantics, with a fake systemd and a real execution store.

These tests avoid starting real user units so they can run in any CI, while
still exercising the properties that matter: a terminal state is final, a
recovery decision never invents a pass, idempotency is exact, and cancel is
idempotent. The real systemd path is proven separately by
${B}tests/live_systemd_e2e.py${B} and by the real compiler runs in the test report.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from agent_gates import api_types as api
from agent_gates import engine
from agent_gates import worker as worker_mod
from agent_gates.config import from_mapping
from agent_gates.execution import ExecutionManager
from agent_gates.execution_store import ExecutionStore, new_execution_id
from agent_gates.manifest import digest
from agent_gates.registry import Registry
from agent_gates.source_identity import snapshot

PYTHON = sys.executable


class FakeSystemd:
    """A systemd stand-in that records calls and can report a chosen state."""

    class SystemdUnavailable(RuntimeError):
        pass

    def __init__(self):
        self.started = []
        self.stopped = []
        self.running = set()
        self.available_result = (True, "running")

    def available(self):
        return self.available_result

    def start(self, execution_id):
        self.started.append(execution_id)
        self.running.add(execution_id)

    def stop(self, execution_id):
        self.stopped.append(execution_id)
        self.running.discard(execution_id)

    def properties(self, execution_id, *names):
        state = "active" if execution_id in self.running else "inactive"
        return {"ActiveState": state, "SubState": state, "Result": "success"}

    def kill(self, execution_id, timeout=60):
        pass

    def reset(self, execution_id):
        pass

    def journal_tail(self, execution_id, lines=50):
        return ""


class ExecutionFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "input.txt").write_text("bytes")
        self.pack = self.root / "adapters/test-pack"
        (self.pack / "scripts").mkdir(parents=True)
        script = self.pack / "scripts/gate.py"
        script.write_text("import json,sys\nprint(json.dumps({'ok': True}))\n")
        manifest = {
            "schemaVersion": 1, "id": "test-pack", "version": "1",
            "files": {"scripts/gate.py": digest(script)},
            "permissions": {"sourceWrite": False, "hostExecution": True},
            "recipes": [{"id": "test.run", "cwd": "source", "timeout": 60,
                         "argv": [PYTHON, "-I", "{pack}/scripts/gate.py"]}],
            "gates": [{"id": "test", "recipeId": "test.run",
                       "gateType": "deterministic-test",
                       "observationChecks": {"ok": True}}],
        }
        (self.pack / "manifest.json").write_text(json.dumps(manifest))
        (self.root / "state").mkdir()
        (self.root / "evidence").mkdir()
        self.config = from_mapping({
            "schema_version": 1,
            "state_root": str(self.root / "state"),
            "evidence_root": str(self.root / "evidence"),
            "adapters": {"test-pack": {}},
            "workspaces": {"ws": {"source_root": str(self.source),
                                    "adapters": ["test-pack"]}},
            "worker": {"backend": "systemd", "max_concurrent": 2,
                       "python": PYTHON},
        }, path=self.root / "config.toml")
        self.systemd = FakeSystemd()
        self.manager = ExecutionManager(self.config,
                                        registry=Registry(self.root / "adapters"),
                                        systemd=self.systemd)
        self.store = ExecutionStore(self.config.state_root)

    def request(self, **overrides):
        values = {"workspace": "ws", "adapter": "test-pack", "gates": ["test"]}
        values.update(overrides)
        return api.ExecutionRequest(**values)


class SubmitTests(ExecutionFixture):
    def test_submit_returns_without_waiting_for_the_gate(self):
        started = time.monotonic()
        status = self.manager.submit(self.request())
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNot(status.state, api.ExecutionState.FINISHED)
        self.assertEqual(len(self.systemd.started), 1)

    def test_submit_persists_the_request_before_starting_the_worker(self):
        status = self.manager.submit(self.request(request_id="r1"))
        record = self.store.read_request(status.execution_id)
        self.assertEqual(record["workspace"], "ws")
        self.assertEqual(record["request_id"], "r1")
        # The configuration binding is recorded, so a worker cannot adopt a
        # different machine contract after the fact.
        self.assertEqual(record["config_digest"], self.config.config_digest)

    def test_unknown_workspace_is_a_request_error(self):
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.request(workspace="nope"))
        self.assertIs(caught.exception.code, api.ErrorCode.UNKNOWN_WORKSPACE)

    def test_adapter_not_registered_for_the_workspace(self):
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.request(adapter="other-pack"))
        self.assertIn(caught.exception.code,
                      {api.ErrorCode.UNKNOWN_ADAPTER, api.ErrorCode.ADAPTER_NOT_ALLOWED})

    def test_unknown_gate_is_a_request_error(self):
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.request(gates=["absent"]))
        self.assertIs(caught.exception.code, api.ErrorCode.UNKNOWN_GATE)

    def test_systemd_unavailable_is_not_silently_degraded(self):
        self.systemd.available_result = (False, "no manager")
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.request())
        self.assertIs(caught.exception.code, api.ErrorCode.SYSTEMD_UNAVAILABLE)
        # Nothing may have been started in-process as a fallback.
        self.assertEqual(self.systemd.started, [])

    def test_concurrency_limit_refuses_rather_than_queueing(self):
        self.config.worker.max_concurrent = 1
        self.manager.submit(self.request(request_id="a"))
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.request(request_id="b"))
        self.assertIs(caught.exception.code, api.ErrorCode.CONCURRENCY_LIMIT)


class IdempotencyTests(ExecutionFixture):
    def test_same_request_id_returns_the_original_execution(self):
        first = self.manager.submit(self.request(request_id="same"))
        second = self.manager.submit(self.request(request_id="same"))
        self.assertEqual(first.execution_id, second.execution_id)
        self.assertEqual(len(self.systemd.started), 1)

    def test_same_id_different_request_is_a_conflict(self):
        # Two VALID requests that differ only in their gate selection. The
        # second one must be refused as a conflict, because the id already
        # names a real execution of a different request.
        self.manager.submit(self.request(request_id="same", gates=["test"]))
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.request(request_id="same",
                                             gates=["test", "test"]))
        self.assertIs(caught.exception.code, api.ErrorCode.CONFLICT)
        self.assertIs(caught.exception.code, api.ErrorCode.CONFLICT)

    def test_new_request_id_creates_a_new_execution(self):
        first = self.manager.submit(self.request(request_id="one"))
        second = self.manager.submit(self.request(request_id="two"))
        self.assertNotEqual(first.execution_id, second.execution_id)

    def test_no_request_id_never_deduplicates(self):
        first = self.manager.submit(self.request())
        second = self.manager.submit(self.request())
        self.assertNotEqual(first.execution_id, second.execution_id)


class TerminalStateTests(ExecutionFixture):
    def test_a_terminal_state_cannot_be_overwritten(self):
        execution_id = new_execution_id()
        self.store.create(execution_id, {"workspace": "ws"},
                          {"state": api.ExecutionState.SUBMITTED.value})
        self.store.transition(execution_id, api.ExecutionState.FINISHED,
                              runId="r" * 32)
        late = self.store.transition(execution_id, api.ExecutionState.INTERRUPTED)
        self.assertEqual(late["state"], api.ExecutionState.FINISHED.value)

    def test_cancelled_is_terminal_for_late_events(self):
        execution_id = new_execution_id()
        self.store.create(execution_id, {}, {"state": api.ExecutionState.SUBMITTED.value})
        self.store.transition(execution_id, api.ExecutionState.CANCELLED)
        late = self.store.transition(execution_id, api.ExecutionState.FINISHED,
                                     runId="r" * 32)
        self.assertEqual(late["state"], api.ExecutionState.CANCELLED.value)
        self.assertIsNone(late.get("runId"))


class CancelTests(ExecutionFixture):
    def test_cancel_stops_the_unit_and_marks_cancelled(self):
        status = self.manager.submit(self.request())
        cancelled = self.manager.cancel(status.execution_id)
        self.assertEqual(cancelled.state, api.ExecutionState.CANCELLED)
        self.assertIn(status.execution_id, self.systemd.stopped)
        self.assertIsNone(cancelled.run_id)

    def test_cancel_is_idempotent(self):
        status = self.manager.submit(self.request())
        first = self.manager.cancel(status.execution_id)
        second = self.manager.cancel(status.execution_id)
        self.assertEqual(first.state, second.state)
        self.assertEqual(len(self.systemd.stopped), 1)

    def test_cancelling_an_unknown_execution_is_an_error(self):
        with self.assertRaises(api.APIError) as caught:
            self.manager.cancel("0" * 32)
        self.assertIs(caught.exception.code, api.ErrorCode.EXECUTION_NOT_FOUND)

    def test_cancelling_a_finished_execution_reports_its_terminal_state(self):
        execution_id = new_execution_id()
        self.store.create(execution_id, {}, {"state": api.ExecutionState.SUBMITTED.value})
        self.store.transition(execution_id, api.ExecutionState.FINISHED,
                              runId="r" * 32)
        result = self.manager.cancel(execution_id)
        self.assertEqual(result.state, api.ExecutionState.FINISHED)
        self.assertEqual(self.systemd.stopped, [])


class RecoveryTests(ExecutionFixture):
    """The crash windows from D6: none may be reported as a success."""

    def test_worker_gone_without_a_receipt_becomes_interrupted(self):
        status = self.manager.submit(self.request())
        self.systemd.running.discard(status.execution_id)
        recovered = self.manager.status(status.execution_id)
        self.assertIs(recovered.state, api.ExecutionState.INTERRUPTED)
        self.assertIsNone(recovered.run_id)

    def test_interrupted_never_reports_a_gate_pass(self):
        status = self.manager.submit(self.request())
        self.systemd.running.discard(status.execution_id)
        self.manager.status(status.execution_id)
        finished = self.manager.status(status.execution_id)
        self.assertIs(finished.state, api.ExecutionState.INTERRUPTED)

    def test_a_stale_worker_pid_is_not_treated_as_alive(self):
        status = self.manager.submit(self.request())
        # Record a pid that certainly is not this process's ancestor child.
        self.store.write_worker(status.execution_id,
                                {"pid": 1, "boot_id": "not-this-boot"})
        self.systemd.running.discard(status.execution_id)
        recovered = self.manager.status(status.execution_id)
        self.assertIs(recovered.state, api.ExecutionState.INTERRUPTED)

    def test_status_unknown_execution_is_an_error(self):
        with self.assertRaises(api.APIError) as caught:
            self.manager.status("0" * 32)
        self.assertIs(caught.exception.code, api.ErrorCode.EXECUTION_NOT_FOUND)

    def test_invalid_execution_id_is_refused_before_touching_the_filesystem(self):
        with self.assertRaises(api.APIError):
            self.manager.status("../../etc/passwd")


class DeviceSelectionTests(ExecutionFixture):
    """The device a request uses: the configured default and explicit selection."""

    def npu_config(self, **devices):
        manifest = {
            "schemaVersion": 1, "id": "test-npu", "version": "1",
            "files": {},
            "permissions": {"sourceWrite": False, "hostExecution": True},
            "recipes": [{"id": "npu.run", "cwd": "source", "timeout": 60,
                         "argv": [PYTHON, "-I", "-c", "print(1)"]}],
            "gates": [{"id": "npu", "recipeId": "npu.run",
                       "gateType": "deterministic-test",
                       "observationChecks": {"exitCodeZero": True}}],
        }
        return self.write_npu_config(manifest, **devices)

    def write_npu_config(self, manifest, **devices):
        pack = self.root / "adapters/test-npu"
        pack.mkdir(parents=True, exist_ok=True)
        (pack / "manifest.json").write_text(json.dumps(manifest))
        body = {
            "schema_version": 1,
            "state_root": str(self.root / "state"),
            "evidence_root": str(self.root / "evidence"),
            "adapters": {"test-npu": {}},
            "workspaces": {"ws": {"source_root": str(self.source),
                                  "adapters": ["test-npu"]}},
            "worker": {"backend": "systemd", "max_concurrent": 2, "python": PYTHON},
        }
        if devices:
            body["devices"] = devices
        return from_mapping(body, path=self.root / "config.toml")

    def windowed_npu_config(self, **devices):
        """An NPU recipe that opens the lease itself and reports its window."""
        script = self.root / "adapters/test-npu/scripts/gate.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            "import fcntl, json, os\n"
            "path = os.environ['AGENT_GATES_DEVICE_LEASE_PATH']\n"
            "fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "print(json.dumps({\n"
            "    'device': os.environ.get('TEST_DEVICE_ID'),\n"
            "    'deviceWindow': {\n"
            "        'resource': os.environ['AGENT_GATES_DEVICE_LEASE_RESOURCE'],\n"
            "        'path': path,\n"
            "        'startedAt': '2026-01-01T00:00:00+00:00',\n"
            "        'releasedAt': '2026-01-01T00:00:01+00:00',\n"
            "        'occupancy': 'free', 'waited': 0.0}}))\n"
            "os.close(fd)\n")
        manifest = {
            "schemaVersion": 1, "id": "test-npu", "version": "1",
            "files": {"scripts/gate.py": digest(script)},
            "permissions": {"sourceWrite": False, "hostExecution": True},
            "recipes": [{"id": "npu.run", "cwd": "source", "timeout": 30,
                         "argv": [PYTHON, "-I", "{pack}/scripts/gate.py"],
                         "deviceWindow": {"waitSeconds": 5}}],
            "gates": [{"id": "npu", "recipeId": "npu.run",
                       "gateType": "deterministic-test",
                       "observationChecks": {"exitCodeZero": True}}],
        }
        return self.write_npu_config(manifest, **devices)

    def npu_manager(self, **devices):
        return ExecutionManager(self.npu_config(**devices),
                                registry=Registry(self.root / "adapters"),
                                systemd=self.systemd)

    @staticmethod
    def npu_request(**overrides):
        values = {"workspace": "ws", "adapter": "test-npu", "gates": ["npu"]}
        values.update(overrides)
        return api.ExecutionRequest(**values)

    def receipt_on_disk(self, run_id, **fields):
        """A receipt the recovery scan can really read, unsigned on purpose.

        `_find_receipt` discovers candidates before verification; these tests
        exercise that discovery, so the file only needs the fields it compares.
        """
        directory = self.manager.evidence.root / "runs" / run_id
        directory.mkdir(parents=True)
        receipt = {
            "runId": run_id,
            "startedAt": "2099-01-01T00:00:00+00:00",
            "adapter": {"id": "test-npu"},
            "source": {"root": os.path.realpath(str(self.source))},
            "gates": [{"gateId": "npu"}],
        }
        receipt.update(fields)
        (directory / "receipt.json").write_text(json.dumps(receipt))
        return receipt

    def device_record(self, **overrides):
        values = {
            "adapter": "test-npu",
            "source_root": os.path.realpath(str(self.source)),
            "gates": ["npu"],
            "submitted_at": time.time(),
            "device_id": "2",
            "device_required": True,
        }
        values.update(overrides)
        return values

    def host_only_npu_config(self, **devices):
        """An NPU Adapter whose only recipe never touches the card."""
        manifest = {
            "schemaVersion": 1, "id": "test-npu", "version": "1",
            "files": {},
            "permissions": {"sourceWrite": False, "hostExecution": True},
            "recipes": [{"id": "npu.run", "cwd": "source", "timeout": 60,
                         "device": "none",
                         "argv": [PYTHON, "-I", "-c", "print(1)"]}],
            "gates": [{"id": "npu", "recipeId": "npu.run",
                       "gateType": "deterministic-test",
                       "observationChecks": {"exitCodeZero": True}}],
        }
        return self.write_npu_config(manifest, **devices)

    def run_worker(self, manager, config, execution_id):
        """Run the real Engine through the worker contract, as systemd would."""
        execution_engine = engine.Engine(
            registry=Registry(self.root / "adapters"), store=manager.evidence,
            lock_root=config.lock_root, device_lock_root=self.root / "locks")
        return worker_mod.run_execution(execution_id, store=manager.store,
                                        config=config, engine=execution_engine)

    def hide_worker_outcome(self, manager, execution_id):
        """Leave the signed receipt on disk, but roll the status back to RUNNING.

        This is the crash window recovery exists for: the receipt was published,
        but the transition that names it was never observed.
        """
        status = manager.store.read_status(execution_id)
        for key in ("runId", "gateStatus", "gateResults", "completedGates",
                    "exitReason", "finishedAt"):
            status.pop(key, None)
        status["state"] = api.ExecutionState.RUNNING.value
        manager.store.write_status(execution_id, status)
        self.systemd.running.discard(execution_id)

    def rewrite_record(self, manager, execution_id, **fields):
        record = manager.store.read_request(execution_id)
        record.update(fields)
        manager.store.request_path(execution_id).write_text(json.dumps(record))

    def test_a_non_npu_adapter_gets_no_device(self):
        self.assertIsNone(self.manager._device_for("test-pack", None))

    def test_an_npu_adapter_gets_a_device_with_no_configuration(self):
        """The default policy: a device is available without being named first."""
        manager = ExecutionManager(self.npu_config(),
                                   registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        self.assertEqual(manager._device_for("test-npu", None), "0")

    def test_a_narrowed_configuration_is_respected(self):
        manager = ExecutionManager(self.npu_config(allowed_npu_devices=[4]),
                                   registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        self.assertEqual(manager._device_for("test-npu", None), "4")

    def test_an_excluded_device_is_refused(self):
        manager = ExecutionManager(self.npu_config(allowed_npu_devices=[4]),
                                   registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        with self.assertRaises(api.APIError) as caught:
            manager._device_for("test-npu", "9")
        self.assertIs(caught.exception.code, api.ErrorCode.DEVICE_NOT_AUTHORIZED)

    def test_explicit_zero_is_not_treated_as_the_default(self):
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2],
                                   default_npu_device=1)
        status = manager.submit(self.npu_request(device_id="0"))
        self.assertEqual(status.device_id, "0")
        self.assertEqual(manager.store.read_request(status.execution_id)["device_id"], "0")
        self.assertEqual(manager.store.read_status(status.execution_id)["deviceId"], "0")
        self.assertEqual(manager.wait(status.execution_id, timeout=0).status.device_id, "0")

    def test_an_explicit_authorized_device_is_used(self):
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2],
                                   default_npu_device=1)
        status = manager.submit(self.npu_request(device_id="2"))
        self.assertEqual(status.device_id, "2")
        self.assertEqual(manager.store.read_request(status.execution_id)["device_id"], "2")
        self.assertEqual(manager.status(status.execution_id).device_id, "2")

    def test_an_omitted_device_still_uses_the_configured_default(self):
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2],
                                   default_npu_device=1)
        status = manager.submit(self.npu_request())
        self.assertEqual(status.device_id, "1")

    def test_an_unauthorized_device_is_refused_before_a_worker_starts(self):
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2],
                                   default_npu_device=1)
        with self.assertRaises(api.APIError) as caught:
            manager.submit(self.npu_request(device_id="9"))
        self.assertIs(caught.exception.code, api.ErrorCode.DEVICE_NOT_AUTHORIZED)
        self.assertEqual(self.systemd.started, [])

    def test_a_malformed_device_is_a_request_error(self):
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2])
        with self.assertRaises(api.APIError) as caught:
            manager.submit(self.npu_request(device_id="npu1"))
        self.assertIs(caught.exception.code, api.ErrorCode.INVALID_REQUEST)
        self.assertEqual(self.systemd.started, [])

    def test_replaying_the_same_request_on_another_card_is_a_conflict(self):
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2],
                                   default_npu_device=1)
        first = manager.submit(self.npu_request(request_id="r-card", device_id="1"))
        with self.assertRaises(api.APIError) as caught:
            manager.submit(self.npu_request(request_id="r-card", device_id="2"))
        self.assertIs(caught.exception.code, api.ErrorCode.CONFLICT)
        self.assertEqual(len(self.systemd.started), 1)
        self.assertEqual(manager.status(first.execution_id).device_id, "1")

    def test_replaying_the_same_card_is_the_same_execution(self):
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2],
                                   default_npu_device=1)
        first = manager.submit(self.npu_request(request_id="r-card", device_id="1"))
        again = manager.submit(self.npu_request(request_id="r-card"))
        self.assertEqual(again.execution_id, first.execution_id)
        self.assertEqual(len(self.systemd.started), 1)

    def test_a_profile_that_needs_no_device_keeps_running_without_one(self):
        status = self.manager.submit(self.request(device_id="1"))
        self.assertIsNone(status.device_id)
        record = self.store.read_request(status.execution_id)
        self.assertIsNone(record["device_id"])
        self.assertFalse(record["device_required"])

    def test_the_receipt_card_matches_the_window_the_adapter_opened(self):
        """V2 selection reaches the adapter and both records name one card."""
        config = self.windowed_npu_config(allowed_npu_devices=[0, 1, 2],
                                          default_npu_device=1)
        manager = ExecutionManager(config, registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        status = manager.submit(self.npu_request(device_id="2"))
        device_root = self.root / "device-locks"
        device_root.mkdir()
        execution_engine = engine.Engine(
            registry=Registry(self.root / "adapters"), store=manager.evidence,
            lock_root=config.lock_root, device_lock_root=device_root)
        with patch.dict(engine.DEVICE_ADAPTERS,
                        {"test-npu": "TEST_DEVICE_ID"}, clear=False):
            updated = worker_mod.run_execution(
                status.execution_id, store=manager.store, config=config,
                engine=execution_engine)
        self.assertEqual(updated["state"], "FINISHED", updated)
        receipt = manager.evidence.read(updated["runId"])
        self.assertEqual(receipt["status"], "PASS", receipt["gates"][0])
        self.assertEqual(receipt["authorizedDeviceId"], "2")
        self.assertEqual(receipt["deviceLease"],
                         {"mode": "windowed", "resource": "npu:2", "gates": ["npu"]})
        self.assertEqual(receipt["gates"][0]["deviceWindow"]["resource"], "npu:2")
        self.assertEqual(receipt["gates"][0]["observations"]["device"], "2")
        status_now = manager.status(status.execution_id)
        self.assertEqual(status_now.device_id, "2")
        # The status the caller polls and the signed receipt describe one card,
        # and the lease the recipe opened is that card's lock file.
        self.assertEqual(status_now.device_id, receipt["authorizedDeviceId"])
        verification = manager.evidence.verify(updated["runId"],
                                               registry=manager.registry)
        self.assertEqual(verification["status"], "PASS", verification)
        from agent_gates.locking import lock_path
        self.assertTrue(lock_path("npu:2", device_root).is_file())
        record = manager.store.read_request(status.execution_id)
        self.assertTrue(record["device_required"])
        self.assertEqual(receipt["authorizedDeviceId"], record["device_id"])

    def test_a_host_only_recipe_does_not_claim_a_bound_card(self):
        """An NPU Adapter can ship a profile that never touches the card."""
        config = self.host_only_npu_config(allowed_npu_devices=[0, 1, 2])
        manager = ExecutionManager(config, registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        status = manager.submit(self.npu_request(device_id="2"))
        record = manager.store.read_request(status.execution_id)
        # The id is validated, but the selected gate touches no card, so submit
        # records none rather than claiming hardware the run will never use.
        self.assertIsNone(record["device_id"])
        self.assertFalse(record["device_required"])
        self.assertIsNone(status.device_id)
        self.assertIsNone(manager.status(status.execution_id).device_id)
        self.assertIsNone(manager.wait(status.execution_id, timeout=0).status.device_id)
        self.assertIsNone(manager.store.read_status(status.execution_id)["deviceId"])

    def test_a_device_receipt_without_a_card_is_not_adopted(self):
        """A host-only receipt cannot stand in for a crashed device run."""
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2])
        manager.store.initialize()
        receipt = self.receipt_on_disk("c" * 32)
        record = self.device_record()
        self.assertFalse(manager._receipt_matches_request(receipt, record))
        self.assertIsNone(manager._find_receipt("3" * 32, record))

    def test_a_device_receipt_from_another_card_is_not_adopted(self):
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2])
        manager.store.initialize()
        receipt = self.receipt_on_disk("d" * 32, authorizedDeviceId="1")
        record = self.device_record()
        self.assertFalse(manager._receipt_matches_request(receipt, record))
        self.assertIsNone(manager._find_receipt("4" * 32, record))

    def test_a_legacy_device_record_still_requires_its_card(self):
        """Records written before the flag existed keep the stricter reading."""
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2])
        record = self.device_record()
        record.pop("device_required")
        receipt = self.receipt_on_disk("e" * 32)
        self.assertFalse(manager._receipt_matches_request(receipt, record))

    def test_a_host_only_request_keeps_its_no_device_recovery(self):
        manager = self.npu_manager(allowed_npu_devices=[0, 1, 2])
        manager.store.initialize()
        receipt = self.receipt_on_disk("f" * 32)
        record = self.device_record(device_id=None, device_required=False)
        self.assertTrue(manager._receipt_matches_request(receipt, record))
        self.assertEqual(manager._find_receipt("5" * 32, record), "f" * 32)

    def test_a_completed_device_run_reports_one_card_on_status_wait_and_disk(self):
        """The card resolved at admission is the card every result path shows."""
        config = self.windowed_npu_config(allowed_npu_devices=[0, 1, 2],
                                          default_npu_device=1)
        manager = ExecutionManager(config, registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        status = manager.submit(self.npu_request(device_id="2"))
        with patch.dict(engine.DEVICE_ADAPTERS,
                        {"test-npu": "TEST_DEVICE_ID"}, clear=False):
            final = self.run_worker(manager, config, status.execution_id)
        self.assertEqual(final["state"], "FINISHED", final)
        observed = manager.status(status.execution_id)
        waited = manager.wait(status.execution_id, timeout=0)
        self.assertIs(observed.state, api.ExecutionState.FINISHED)
        self.assertEqual(observed.device_id, "2")
        self.assertEqual(waited.status.device_id, "2")
        self.assertEqual(manager.store.read_request(status.execution_id)["device_id"], "2")
        self.assertEqual(manager.store.read_status(status.execution_id)["deviceId"], "2")
        self.assertEqual(manager.evidence.read(final["runId"])["authorizedDeviceId"], "2")

    def test_a_completed_no_device_run_reports_no_card_on_status_wait_and_disk(self):
        status = self.manager.submit(self.request(device_id="3"))
        final = self.run_worker(self.manager, self.config, status.execution_id)
        self.assertEqual(final["state"], "FINISHED", final)
        observed = self.manager.status(status.execution_id)
        waited = self.manager.wait(status.execution_id, timeout=0)
        self.assertIs(observed.state, api.ExecutionState.FINISHED)
        self.assertIsNone(observed.device_id)
        self.assertIsNone(waited.status.device_id)
        self.assertIsNone(self.store.read_request(status.execution_id)["device_id"])
        self.assertIsNone(self.store.read_status(status.execution_id)["deviceId"])
        self.assertNotIn("authorizedDeviceId", self.manager.evidence.read(final["runId"]))

    def test_recovery_adopts_a_device_receipt_and_keeps_its_card(self):
        """A crash after the receipt still reports the admitted card everywhere."""
        config = self.windowed_npu_config(allowed_npu_devices=[0, 1, 2],
                                          default_npu_device=1)
        manager = ExecutionManager(config, registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        status = manager.submit(self.npu_request(device_id="2"))
        with patch.dict(engine.DEVICE_ADAPTERS,
                        {"test-npu": "TEST_DEVICE_ID"}, clear=False):
            final = self.run_worker(manager, config, status.execution_id)
        self.assertEqual(final["state"], "FINISHED", final)
        self.hide_worker_outcome(manager, status.execution_id)
        # A torn status may have lost the field, but admission is the authority
        # for the resolved card and recovery must publish it again.
        torn = manager.store.read_status(status.execution_id)
        torn.pop("deviceId")
        manager.store.write_status(status.execution_id, torn)
        recovered = manager.status(status.execution_id)
        self.assertIs(recovered.state, api.ExecutionState.FINISHED)
        self.assertEqual(recovered.run_id, final["runId"])
        self.assertEqual(recovered.device_id, "2")
        waited = manager.wait(status.execution_id, timeout=0)
        self.assertIs(waited.status.state, api.ExecutionState.FINISHED)
        self.assertEqual(waited.status.device_id, "2")
        persisted = manager.store.read_status(status.execution_id)
        self.assertEqual(persisted["state"], api.ExecutionState.FINISHED.value)
        self.assertEqual(persisted["runId"], final["runId"])
        self.assertEqual(persisted["deviceId"], "2")
        self.assertEqual(manager.store.read_request(status.execution_id)["device_id"], "2")

    def test_recovery_adopts_a_no_device_receipt_without_a_card(self):
        status = self.manager.submit(self.request(device_id="3"))
        final = self.run_worker(self.manager, self.config, status.execution_id)
        self.assertEqual(final["state"], "FINISHED", final)
        self.hide_worker_outcome(self.manager, status.execution_id)
        recovered = self.manager.status(status.execution_id)
        self.assertIs(recovered.state, api.ExecutionState.FINISHED)
        self.assertEqual(recovered.run_id, final["runId"])
        self.assertIsNone(recovered.device_id)
        self.assertIsNone(self.manager.wait(status.execution_id, timeout=0)
                          .status.device_id)
        persisted = self.store.read_status(status.execution_id)
        self.assertEqual(persisted["state"], api.ExecutionState.FINISHED.value)
        self.assertIsNone(persisted["deviceId"])

    def test_recovery_adopts_a_host_only_receipt_without_a_card(self):
        """The protocol demands a receipt card only for a gate that drives it."""
        config = self.host_only_npu_config(allowed_npu_devices=[0, 1, 2])
        manager = ExecutionManager(config, registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        status = manager.submit(self.npu_request(device_id="2"))
        record = manager.store.read_request(status.execution_id)
        # No selected gate drives the card, so submit binds and claims none.
        self.assertIsNone(record["device_id"])
        self.assertFalse(record["device_required"])
        self.assertIsNone(status.device_id)
        final = self.run_worker(manager, config, status.execution_id)
        self.assertEqual(final["state"], "FINISHED", final)
        self.assertNotIn("authorizedDeviceId", manager.evidence.read(final["runId"]))
        self.hide_worker_outcome(manager, status.execution_id)
        recovered = manager.status(status.execution_id)
        self.assertIs(recovered.state, api.ExecutionState.FINISHED)
        self.assertEqual(recovered.run_id, final["runId"])
        self.assertIsNone(recovered.device_id)
        self.assertIsNone(manager.wait(status.execution_id, timeout=0)
                          .status.device_id)
        self.assertIsNone(manager.store.read_status(status.execution_id)["deviceId"])

    def test_recovery_refuses_a_device_request_whose_receipt_names_no_card(self):
        """A signed host-only receipt cannot complete a run that drives a card."""
        config = self.host_only_npu_config(allowed_npu_devices=[0, 1, 2])
        manager = ExecutionManager(config, registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        status = manager.submit(self.npu_request(device_id="2"))
        execution_id = status.execution_id
        # Model the protocol violation recovery must survive: the record says
        # the selected gate drives the card, while the receipt proves none was
        # bound. The record is the only mutable side; the receipt is signed.
        self.rewrite_record(manager, execution_id,
                            device_id="2", device_required=True)
        status_raw = manager.store.read_status(execution_id)
        status_raw["deviceId"] = "2"
        manager.store.write_status(execution_id, status_raw)
        final = self.run_worker(manager, config, execution_id)
        self.assertEqual(final["state"], "FINISHED", final)
        self.assertNotIn("authorizedDeviceId", manager.evidence.read(final["runId"]))
        self.hide_worker_outcome(manager, execution_id)
        recovered = manager.status(execution_id)
        self.assertIs(recovered.state, api.ExecutionState.INTERRUPTED)
        self.assertIsNone(recovered.run_id)
        self.assertEqual(recovered.device_id, "2")
        self.assertIs(manager.wait(execution_id, timeout=0).status.state,
                      api.ExecutionState.INTERRUPTED)
        persisted = manager.store.read_status(execution_id)
        self.assertEqual(persisted["state"], api.ExecutionState.INTERRUPTED.value)
        self.assertIsNone(persisted.get("runId"))
        self.assertEqual(persisted["deviceId"], "2")

    def test_recovery_refuses_a_device_receipt_from_another_card(self):
        config = self.windowed_npu_config(allowed_npu_devices=[0, 1, 2],
                                          default_npu_device=1)
        manager = ExecutionManager(config, registry=Registry(self.root / "adapters"),
                                   systemd=self.systemd)
        status = manager.submit(self.npu_request(device_id="1"))
        execution_id = status.execution_id
        with patch.dict(engine.DEVICE_ADAPTERS,
                        {"test-npu": "TEST_DEVICE_ID"}, clear=False):
            final = self.run_worker(manager, config, execution_id)
        self.assertEqual(final["state"], "FINISHED", final)
        receipt = manager.evidence.read(final["runId"])
        self.assertEqual(receipt["authorizedDeviceId"], "1")
        # The execution was admitted on card 2; a nearby receipt for card 1
        # cannot stand in for it after the status transition was lost.
        self.rewrite_record(manager, execution_id, device_id="2")
        status_raw = manager.store.read_status(execution_id)
        status_raw["deviceId"] = "2"
        manager.store.write_status(execution_id, status_raw)
        self.hide_worker_outcome(manager, execution_id)
        recovered = manager.status(execution_id)
        self.assertIs(recovered.state, api.ExecutionState.INTERRUPTED)
        self.assertIsNone(recovered.run_id)
        self.assertEqual(recovered.device_id, "2")
        self.assertIs(manager.wait(execution_id, timeout=0).status.state,
                      api.ExecutionState.INTERRUPTED)
        persisted = manager.store.read_status(execution_id)
        self.assertEqual(persisted["state"], api.ExecutionState.INTERRUPTED.value)
        self.assertIsNone(persisted.get("runId"))
        self.assertEqual(persisted["deviceId"], "2")

class WaitTests(ExecutionFixture):
    def test_wait_is_bounded(self):
        status = self.manager.submit(self.request())
        started = time.monotonic()
        result = self.manager.wait(status.execution_id, timeout=0.5)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNotNone(result)

    def test_wait_clamps_an_enormous_timeout(self):
        from agent_gates.execution import MAX_WAIT_SECONDS
        status = self.manager.submit(self.request())
        started = time.monotonic()
        self.manager.wait(status.execution_id, timeout=10 ** 9)
        elapsed = time.monotonic() - started
        # A single call may not be used to wait out a whole build.
        self.assertLess(elapsed, MAX_WAIT_SECONDS + 5)

    def test_wait_on_a_terminal_execution_returns_immediately(self):
        status = self.manager.submit(self.request())
        self.manager.cancel(status.execution_id)
        started = time.monotonic()
        result = self.manager.wait(status.execution_id, timeout=30)
        self.assertFalse(result.timed_out)
        self.assertLess(time.monotonic() - started, 5)

    def test_config_digest_mismatch_marks_execution_interrupted(self):
        status = self.manager.submit(self.request())
        eid = status.execution_id
        from agent_gates import worker as worker_mod
        req = self.manager.store.read_request(eid)
        req["config_digest"] = "0" * 64
        self.manager.store.request_path(eid).write_text(json.dumps(req))
        updated = worker_mod.run_execution(eid, store=self.manager.store, config=self.config)
        self.assertEqual(updated["state"], "INTERRUPTED")
        self.assertIn("CONFIG_IDENTITY_CHANGED", updated["exitReason"])

    def test_find_receipt_ignores_claimed_and_older_receipts(self):
        import os

        run_old = "a" * 32
        run_new = "b" * 32
        old_dir = self.config.evidence_root / "runs" / run_old
        new_dir = self.config.evidence_root / "runs" / run_new
        old_dir.mkdir(parents=True)
        new_dir.mkdir(parents=True)
        source_real = os.path.realpath(str(self.source))
        (old_dir / "receipt.json").write_text(json.dumps({
            "runId": run_old,
            "startedAt": "2020-01-01T00:00:00+00:00",
            "adapter": {"id": "test-pack"},
            "source": {"root": source_real},
            "gates": [{"gateId": "test-gate"}],
        }))
        claimed_id = "1" * 32
        self.manager.store.initialize()
        self.manager.store.create(
            claimed_id,
            {"execution_id": claimed_id, "adapter": "test-pack", "source_root": source_real, "gates": ["test-gate"]},
            {"executionId": claimed_id, "state": "FINISHED", "runId": run_new},
        )
        (new_dir / "receipt.json").write_text(json.dumps({
            "runId": run_new,
            "startedAt": "2099-01-01T00:00:00+00:00",
            "adapter": {"id": "test-pack"},
            "source": {"root": source_real},
            "gates": [{"gateId": "test-gate"}],
        }))
        target_id = "2" * 32
        found = self.manager._find_receipt(target_id, {
            "adapter": "test-pack",
            "source_root": source_real,
            "gates": ["test-gate"],
            "submitted_at": time.time(),
        })
        self.assertIsNone(found)


class DynamicTargetExecutionTests(ExecutionFixture):
    """V2 carries the same typed request V1 does, and records it exactly."""

    def setUp(self):
        super().setUp()
        self.candidate = self.source / "candidates/kernels.py"
        self.candidate.parent.mkdir()
        self.candidate.write_text("# candidate\n")
        script = self.pack / "scripts/gate.py"
        script.write_text(
            "import json, os\n"
            "print(json.dumps({'target': os.environ.get('AGENT_GATES_TARGET'),"
            " 'candidate': os.environ.get('AGENT_GATES_CANDIDATE')}))\n")
        manifest = {
            "schemaVersion": 1, "id": "test-pack", "version": "1",
            "files": {"scripts/gate.py": digest(script)},
            "permissions": {"sourceWrite": False, "hostExecution": True},
            "recipes": [{"id": "test.run", "cwd": "source", "timeout": 60,
                         "dynamicTarget": True,
                         "argv": [PYTHON, "-I", "{pack}/scripts/gate.py"]}],
            "gates": [{"id": "test", "recipeId": "test.run",
                       "gateType": "deterministic-test",
                       "observationChecks": {"exitCodeZero": True}}],
        }
        (self.pack / "manifest.json").write_text(json.dumps(manifest))

    def dynamic_request(self, **overrides):
        values = {"target": "level4/attention_residual",
                  "candidate": "candidates/kernels.py"}
        values.update(overrides)
        return self.request(**values)

    def run_worker(self, execution_id):
        from agent_gates import worker as worker_mod
        from agent_gates.engine import Engine
        from agent_gates.evidence import EvidenceStore
        engine = Engine(registry=self.manager.registry,
                        store=EvidenceStore(self.config.evidence_root),
                        lock_root=self.config.lock_root,
                        device_lock_root=self.root / "locks")
        return worker_mod.run_execution(execution_id, store=self.manager.store,
                                        config=self.config, engine=engine)

    def test_submit_echoes_and_persists_the_pair(self):
        status = self.manager.submit(self.dynamic_request(request_id="dyn"))
        self.assertEqual(status.target, "level4/attention_residual")
        self.assertEqual(status.candidate, "candidates/kernels.py")
        record = self.store.read_request(status.execution_id)
        self.assertEqual(record["target"], "level4/attention_residual")
        self.assertEqual(record["candidate"], "candidates/kernels.py")

    def test_the_worker_runs_the_bound_pair_and_the_receipt_records_it(self):
        status = self.manager.submit(self.dynamic_request())
        final = self.run_worker(status.execution_id)
        self.assertEqual(final["state"], "FINISHED")
        self.assertEqual(final["gateStatus"], "PASS", final)
        receipt = self.manager.evidence.read(final["runId"])
        self.assertEqual(receipt["dynamicTarget"]["target"],
                         "level4/attention_residual")
        self.assertEqual(receipt["dynamicTarget"]["candidate"]["relativePath"],
                         "candidates/kernels.py")
        self.assertEqual(receipt["gates"][0]["observations"]["target"],
                         "level4/attention_residual")
        observed = self.manager.status(status.execution_id)
        self.assertEqual(observed.target, "level4/attention_residual")
        self.assertEqual(observed.candidate, "candidates/kernels.py")

    def test_a_dynamic_route_without_the_pair_is_an_invalid_request(self):
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.request())
        self.assertIs(caught.exception.code, api.ErrorCode.INVALID_REQUEST)

    def test_a_static_route_refuses_the_pair(self):
        raw = json.loads((self.pack / "manifest.json").read_text())
        raw["recipes"][0].pop("dynamicTarget")
        (self.pack / "manifest.json").write_text(json.dumps(raw))
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.request(
                target="level4/attention_residual",
                candidate="candidates/kernels.py"))
        self.assertIs(caught.exception.code, api.ErrorCode.INVALID_REQUEST)

    def test_a_malformed_candidate_is_an_invalid_request(self):
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.dynamic_request(candidate="../escape.py"))
        self.assertIs(caught.exception.code, api.ErrorCode.INVALID_REQUEST)

    def test_idempotency_compares_the_pair(self):
        first = self.manager.submit(self.dynamic_request(request_id="same"))
        again = self.manager.submit(self.dynamic_request(request_id="same"))
        self.assertEqual(first.execution_id, again.execution_id)
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.dynamic_request(
                request_id="same", target="level4/other"))
        self.assertIs(caught.exception.code, api.ErrorCode.CONFLICT)

    def test_recovery_will_not_adopt_a_receipt_for_another_candidate(self):
        status = self.manager.submit(self.dynamic_request())
        final = self.run_worker(status.execution_id)
        receipt = self.manager.evidence.read(final["runId"])
        record = self.store.read_request(status.execution_id)
        self.assertTrue(self.manager._receipt_matches_request(receipt, record))
        other = dict(record, candidate="candidates/other.py")
        self.assertFalse(self.manager._receipt_matches_request(receipt, other))
        self.assertFalse(self.manager._receipt_matches_request(
            receipt, {"adapter": "test-pack"}))

    def test_submit_freezes_the_candidate_digest_and_the_source_identity(self):
        status = self.manager.submit(self.dynamic_request())
        record = self.store.read_request(status.execution_id)
        bound = record["dynamic_target"]
        self.assertEqual(bound["target"], "level4/attention_residual")
        self.assertEqual(bound["candidate"]["relativePath"], "candidates/kernels.py")
        self.assertEqual(bound["candidate"]["sha256"], digest(self.candidate))
        self.assertEqual(bound["candidate"]["bytes"],
                         self.candidate.stat().st_size)
        # The source tree is frozen too, not just the candidate file.
        self.assertEqual(record["source_identity"], snapshot(self.source))

    def test_a_candidate_swapped_while_queued_is_refused_without_running(self):
        status = self.manager.submit(self.dynamic_request())
        self.candidate.write_text("# swapped while queued\n")
        final = self.run_worker(status.execution_id)
        self.assertEqual(final["state"], "INTERRUPTED", final)
        self.assertIsNone(final.get("runId"))
        self.assertIn("REQUEST_IDENTITY_CHANGED", final["exitReason"])
        self.assertIn("CANDIDATE_IDENTITY_CHANGED", final["exitReason"])

    def test_a_candidate_deleted_while_queued_is_refused_as_an_identity_change(self):
        status = self.manager.submit(self.dynamic_request())
        self.candidate.unlink()
        final = self.run_worker(status.execution_id)
        self.assertEqual(final["state"], "INTERRUPTED", final)
        self.assertIsNone(final.get("runId"))
        self.assertIn("REQUEST_IDENTITY_CHANGED", final["exitReason"])
        self.assertIn("CANDIDATE_IDENTITY_CHANGED", final["exitReason"])

    def test_a_source_changed_while_queued_is_refused_without_running(self):
        status = self.manager.submit(self.dynamic_request())
        (self.source / "input.txt").write_text("changed while queued")
        final = self.run_worker(status.execution_id)
        self.assertEqual(final["state"], "INTERRUPTED", final)
        self.assertIsNone(final.get("runId"))
        self.assertIn("REQUEST_IDENTITY_CHANGED", final["exitReason"])
        self.assertIn("SOURCE_IDENTITY_CHANGED", final["exitReason"])

    def test_idempotency_refuses_a_replaced_candidate_with_the_same_path(self):
        self.manager.submit(self.dynamic_request(request_id="same"))
        self.candidate.write_text("# replaced\n")
        with self.assertRaises(api.APIError) as caught:
            self.manager.submit(self.dynamic_request(request_id="same"))
        self.assertIs(caught.exception.code, api.ErrorCode.CONFLICT)

    def make_optional(self):
        raw = json.loads((self.pack / "manifest.json").read_text())
        raw["recipes"][0]["dynamicTarget"] = "optional"
        (self.pack / "manifest.json").write_text(json.dumps(raw))

    def test_an_optional_route_queues_its_managed_target_without_a_pair(self):
        self.make_optional()
        status = self.manager.submit(self.request())
        record = self.store.read_request(status.execution_id)
        self.assertIsNone(record["dynamic_target"])
        final = self.run_worker(status.execution_id)
        self.assertEqual(final["state"], "FINISHED", final)
        self.assertEqual(final["gateStatus"], "PASS", final)
        receipt = self.manager.evidence.read(final["runId"])
        self.assertNotIn("dynamicTarget", receipt)
        self.assertIsNone(receipt["gates"][0]["observations"]["target"])

    def test_an_optional_route_queues_an_explicit_pair_when_given_one(self):
        self.make_optional()
        status = self.manager.submit(self.dynamic_request())
        final = self.run_worker(status.execution_id)
        self.assertEqual(final["state"], "FINISHED", final)
        self.assertEqual(final["gateStatus"], "PASS", final)
        receipt = self.manager.evidence.read(final["runId"])
        self.assertEqual(receipt["dynamicTarget"]["candidate"]["sha256"],
                         digest(self.candidate))
        self.assertEqual(receipt["gates"][0]["observations"]["target"],
                         "level4/attention_residual")


if __name__ == "__main__":
    unittest.main()
