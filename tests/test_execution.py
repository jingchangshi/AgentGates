"""Execution manager semantics, with a fake systemd and a real execution store.

These tests avoid starting real user units so they can run in any CI, while
still exercising the properties that matter: a terminal state is final, a
recovery decision never invents a pass, idempotency is exact, and cancel is
idempotent. The real systemd path is proven separately by
${B}tests/live_systemd_e2e.py${B} and by the real compiler runs in the test report.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from agent_gates import api_types as api
from agent_gates.config import from_mapping
from agent_gates.execution import ExecutionManager
from agent_gates.execution_store import ExecutionStore, new_execution_id
from agent_gates.manifest import digest
from agent_gates.registry import Registry

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


if __name__ == "__main__":
    unittest.main()
