"""Host-simulated tests for bounded runtime health supervision.

These tests exercise the adapter's real host-command runner and the real device
window lock. Only NPU observations are synthetic: no live card is touched.
"""
from __future__ import annotations

import importlib.util
import json
import os
from contextlib import contextmanager
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    ROOT / "agent_gates" / "adapters" / "tilefoundry-npu" / "scripts"
    / "tilefoundry_host_gate.py"
)


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_script(SCRIPT, "tilefoundry_host_gate_runtime_health_test")


def _proc_state(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        return raw.rsplit(")", 1)[1].split()[0]
    except (IndexError, ValueError):
        return None


def _wait_ended(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _proc_state(pid)
        if state is None or state == "Z":
            return True
        time.sleep(0.05)
    return False


def _row(kind: str, value: str) -> dict:
    return {
        "ok": True,
        "healthKind": kind,
        "healthValue": value,
        "error": None,
    }


def _unavailable(error: str) -> dict:
    return {
        "ok": False,
        "healthKind": "unavailable",
        "healthValue": None,
        "timedOut": False,
        "unreaped": False,
        "error": error,
    }


class _AdvancingClock:
    """Controlled monotonic proxy: each observation advances the fake clock.

    Only ``monotonic`` is synthetic; sleep and timestamp calls resolve to the
    real :mod:`time` module, so short real commands can exercise long simulated
    run times without changing the adapter's deadline logic.
    """

    def __init__(self, step: float):
        self.step = float(step)
        self.value = 0.0

    def monotonic(self) -> float:
        self.value += self.step
        return self.value

    def __getattr__(self, name):
        return getattr(time, name)


class RuntimeHealthCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lease = self.root / "locks" / "npu-1.lock"
        self.external: subprocess.Popen | None = None
        self.residue: subprocess.Popen | None = None

    def tearDown(self):
        for proc in (self.external, self.residue):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    @contextmanager
    def lease_environment(self):
        with patch.dict(
            os.environ,
            {
                "AGENT_GATES_DEVICE_LEASE_PATH": str(self.lease),
                "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:1",
                "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "0",
            },
            clear=False,
        ):
            yield

    def _sequence(self, rows: list[dict], default: dict):
        pending = list(rows)

        def _next(*_args, **_kwargs):
            return pending.pop(0) if pending else default

        return _next

    def _free_processes(self, *_args, **_kwargs):
        return {
            "ok": True,
            "processTableSeen": True,
            "processes": [],
            "error": None,
        }

    def _assert_ended(self, pids: list[int]):
        for pid in pids:
            self.assertTrue(_wait_ended(pid), f"pid {pid} still holds the device")

    def test_mid_run_device_error_cleans_this_run_before_releasing_lease(self):
        pidfile = self.root / "workload-pids"
        command = (
            f"echo $$ > {pidfile}; "
            f"setsid bash -c 'echo $$ >> {pidfile}; exec sleep 300' & "
            "sleep 300"
        )
        health = self._sequence(
            [_row("ok", "OK"), _row("fatal", "ERROR"), _row("fatal", "ERROR")],
            _row("fatal", "ERROR"),
        )
        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(GATE, "_probe_device_health", side_effect=health), \
                patch.object(GATE, "_probe_device_processes", side_effect=self._free_processes), \
                self.lease_environment():
            started = time.monotonic()
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    GATE._run_host_bash_monitored(
                        command,
                        cwd=self.root,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                        cann_setup=None,
                        conda_sh=None,
                        conda_env=None,
                        timeout=30,
                        isolate_temp=True,
                        mode="l4",
                        device="1",
                        monitor_sink=sink,
                        phase_getter=lambda: "device-workload",
                    )
            measured = time.monotonic() - started
            self.assertLess(measured, 12.0, f"fault-to-lease-release took {measured:.3f}s")

            # The second task must acquire the same window immediately; the
            # first run's cleanup happened before this context exited above.
            second_started = time.monotonic()
            with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                pass
            self.assertLess(time.monotonic() - second_started, 1.0)

        self.assertIn("device_error", str(caught.exception))
        self.assertEqual(sink["abort"]["kind"], "device_error")
        self.assertEqual(sink["abort"]["phase"], "device-workload")
        self.assertEqual(sink["npuId"], 7)
        self.assertFalse(sink["abort"]["healthUnknown"])
        self.assertTrue(sink["cleanup"]["attempted"])
        self.assertFalse(sink["cleanup"]["incomplete"])
        self.assertTrue(sink["cleanup"]["workerJoined"])
        self.assertTrue(sink["cleanup"]["quarantine"]["cleared"])
        self.assertFalse(
            Path(f"{self.lease}{GATE.QUARANTINE_SUFFIX}").exists(),
            "a fully cleaned run must not quarantine the card",
        )
        pids = [int(value) for value in pidfile.read_text(encoding="utf-8").split()]
        self.assertGreaterEqual(len(pids), 2)
        self._assert_ended(pids)
        self.assertFalse(set(pids).isdisjoint(set(sink["cleanup"]["pids"])))
        print(
            f"runtime-health fault-to-lease-release measured={measured:.3f}s "
            f"pids={pids} cleanup={sink['cleanup']['seconds']:.3f}s"
        )

    def test_l4_external_occupancy_ends_own_run_but_never_kills_tenant(self):
        self.external = subprocess.Popen(
            ["/bin/sleep", "30"],
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        external_row = {
            "ok": True,
            "processTableSeen": True,
            "processes": [{
                "pid": str(self.external.pid),
                "name": "sleep",
                "memoryMb": "1",
            }],
            "error": None,
        }
        processes = self._sequence([self._free_processes(), external_row, external_row], external_row)
        health = self._sequence([_row("ok", "OK")], _row("ok", "OK"))
        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(GATE, "_probe_device_health", side_effect=health), \
                patch.object(GATE, "_probe_device_processes", side_effect=processes), \
                self.lease_environment():
            started = time.monotonic()
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    GATE._run_host_bash_monitored(
                        "sleep 300",
                        cwd=self.root,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                        cann_setup=None,
                        conda_sh=None,
                        conda_env=None,
                        timeout=30,
                        isolate_temp=True,
                        mode="l4",
                        device="1",
                        monitor_sink=sink,
                        phase_getter=lambda: "workload",
                    )
            measured = time.monotonic() - started

        self.assertLess(measured, 12.0)
        self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
        self.assertEqual(sink["abort"]["kind"], "external_occupancy")
        self.assertIn(self.external.pid, sink["externalPids"])
        self.assertIsNone(self.external.poll(), "an unrelated tenant process was killed")
        self.assertFalse(sink["cleanup"]["incomplete"])
        print(f"runtime-health external-occupancy-to-exit measured={measured:.3f}s")

    def test_probe_self_hang_is_bounded_and_reported_as_unavailable(self):
        result = GATE._bounded_host_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            0.2,
        )
        self.assertTrue(result["timedOut"])
        self.assertFalse(result["unreaped"])
        self.assertLess(result["seconds"], 2.0)
        self.assertEqual(result["returncode"], -signal.SIGKILL)

        def hanging_health(_npu_id, timeout):
            probe = GATE._bounded_host_command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout,
            )
            return {
                "ok": False,
                "healthKind": "unavailable",
                "healthValue": None,
                "error": probe["error"],
                "timedOut": probe["timedOut"],
                "unreaped": probe["unreaped"],
                "probePid": probe["pid"],
                "wchan": probe["wchan"],
            }

        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(GATE, "_probe_device_health", side_effect=hanging_health), \
                patch.object(GATE, "_probe_device_processes", side_effect=self._free_processes), \
                patch.object(GATE, "HEALTH_PROBE_TIMEOUT_SECONDS", 0.2), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.05), \
                patch.object(GATE, "HEALTH_PROBE_FAILURE_LIMIT", 2), \
                self.lease_environment():
            started = time.monotonic()
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    GATE._run_host_bash_monitored(
                        "sleep 300",
                        cwd=self.root,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                        cann_setup=None,
                        conda_sh=None,
                        conda_env=None,
                        timeout=30,
                        isolate_temp=True,
                        mode="l4",
                        device="1",
                        monitor_sink=sink,
                    )
            measured = time.monotonic() - started

        self.assertLess(measured, 5.0)
        self.assertEqual(sink["abort"]["kind"], "probe_unavailable")
        self.assertTrue(sink["abort"]["healthUnknown"])
        self.assertTrue(sink["healthUnknown"])
        self.assertIn("probe timed out", str(caught.exception))
        self.assertFalse(sink["cleanup"]["incomplete"])
        print(f"runtime-health probe-self-hang-to-exit measured={measured:.3f}s")

    def test_silent_slow_task_is_left_to_the_recipe_deadline(self):
        """No CPU/log/progress heuristic; a quiet healthy task simply finishes."""
        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(
                    GATE, "_probe_device_health",
                    side_effect=self._sequence([_row("ok", "OK")], _row("ok", "OK")),
                ), \
                patch.object(GATE, "_probe_device_processes", side_effect=self._free_processes), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1), \
                self.lease_environment():
            with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                completed = GATE._run_host_bash_monitored(
                    "sleep 1.5",
                    cwd=self.root,
                    env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                    cann_setup=None,
                    conda_sh=None,
                    conda_env=None,
                    timeout=10,
                    isolate_temp=True,
                    mode="l4",
                    device="1",
                    monitor_sink=sink,
                )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(sink["outcome"], "completed")
        self.assertNotIn("abort", sink)
        self.assertEqual(sink["health"], "ok")
        self.assertIsNotNone(sink["lastProbe"])

    def test_silent_escaped_descendant_waits_for_the_recipe_deadline_then_cleans(self):
        """Silence alone must not invent a kill; the total deadline still bounds the run."""
        pidfile = self.root / "escaped-pid"
        command = (
            f"setsid bash -c 'echo $$ > {pidfile}; exec sleep 30' & "
            "sleep 30"
        )
        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(
                    GATE, "_probe_device_health",
                    side_effect=self._sequence([_row("ok", "OK")], _row("ok", "OK")),
                ), \
                patch.object(GATE, "_probe_device_processes", side_effect=self._free_processes), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1), \
                self.lease_environment():
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    GATE._run_host_bash_monitored(
                        command,
                        cwd=self.root,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                        cann_setup=None,
                        conda_sh=None,
                        conda_env=None,
                        timeout=1.0,
                        isolate_temp=True,
                        mode="l4",
                        device="1",
                        monitor_sink=sink,
                        phase_getter=lambda: "compile",
                    )
            measured = time.monotonic() - started
        self.assertGreaterEqual(measured, 1.0)
        self.assertLess(measured, 8.0)
        self.assertNotIn("abort", sink, "silence must not be classified as a stall")
        self.assertEqual(sink["outcome"], "runner_error")
        self.assertTrue(sink["cleanup"]["deadlineSweep"]["attempted"])
        escaped = int(pidfile.read_text(encoding="utf-8").strip())
        self.assertIn(escaped, sink["cleanup"]["deadlineSweep"]["pids"])
        self._assert_ended([escaped])
        print(f"runtime-health deadline-to-escaped-cleanup measured={measured:.3f}s")

    def test_l4_unobservable_occupancy_is_reported_not_guessed(self):
        unavailable = {
            "ok": False,
            "processTableSeen": False,
            "processes": [],
            "error": "proc-mem table not available",
        }
        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(
                    GATE, "_probe_device_health",
                    side_effect=self._sequence([_row("ok", "OK")], _row("ok", "OK")),
                ), \
                patch.object(GATE, "_probe_device_processes", return_value=unavailable), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1), \
                patch.object(GATE, "HEALTH_PROBE_FAILURE_LIMIT", 2), \
                self.lease_environment():
            started = time.monotonic()
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    GATE._run_host_bash_monitored(
                        "sleep 30",
                        cwd=self.root,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                        cann_setup=None,
                        conda_sh=None,
                        conda_env=None,
                        timeout=30,
                        isolate_temp=True,
                        mode="l4",
                        device="1",
                        monitor_sink=sink,
                    )
            measured = time.monotonic() - started
        self.assertLess(measured, 5.0)
        self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
        self.assertEqual(sink["abort"]["kind"], "occupancy_unknown")
        self.assertTrue(sink["abort"]["healthUnknown"])
        self.assertIn("device_occupancy_unavailable", sink["unknownSignals"])
        self.assertIn("l4_internal_fault_counters_unavailable", sink["unknownSignals"])
        self.assertFalse(sink["cleanup"]["incomplete"])
        print(f"runtime-health occupancy-unknown-to-exit measured={measured:.3f}s")

    def test_l3_legal_sharing_is_recorded_but_not_aborted(self):
        self.external = subprocess.Popen(
            ["/bin/sleep", "30"],
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        shared = {
            "ok": True,
            "processTableSeen": True,
            "processes": [{
                "pid": str(self.external.pid),
                "name": "sleep",
                "memoryMb": "1",
            }],
            "error": None,
        }
        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(
                    GATE, "_probe_device_health",
                    side_effect=self._sequence([_row("ok", "OK")], _row("ok", "OK")),
                ), \
                patch.object(GATE, "_probe_device_processes", return_value=shared), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1), \
                self.lease_environment():
            with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                completed = GATE._run_host_bash_monitored(
                    "sleep 2",
                    cwd=self.root,
                    env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                    cann_setup=None,
                    conda_sh=None,
                    conda_env=None,
                    timeout=10,
                    isolate_temp=True,
                    mode="l3",
                    device="1",
                    monitor_sink=sink,
                )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(sink["outcome"], "completed")
        self.assertNotIn("abort", sink)
        self.assertEqual(sink["occupancy"], "shared(1)")
        self.assertIn(self.external.pid, sink["externalPids"])
        self.assertIsNone(self.external.poll())

    def test_device_missing_is_distinguished_from_probe_unavailable(self):
        missing = {
            "ok": False,
            "missing": True,
            "npuId": None,
            "error": "no Ascend NPU mapped for device index 1 in `npu-smi info -m`",
        }
        health = self._sequence(
            [_row("ok", "OK")], _unavailable("health command failed")
        )
        mappings = self._sequence(
            [{"ok": True, "npuId": 7, "error": None}], missing
        )
        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(GATE, "_probe_npu_mapping", side_effect=mappings), \
                patch.object(GATE, "_probe_device_health", side_effect=health), \
                patch.object(GATE, "_probe_device_processes", side_effect=self._free_processes), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1), \
                self.lease_environment():
            started = time.monotonic()
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    GATE._run_host_bash_monitored(
                        "sleep 30",
                        cwd=self.root,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                        cann_setup=None,
                        conda_sh=None,
                        conda_env=None,
                        timeout=30,
                        isolate_temp=True,
                        mode="l4",
                        device="1",
                        monitor_sink=sink,
                    )
            measured = time.monotonic() - started
        self.assertLess(measured, 5.0)
        self.assertEqual(getattr(caught.exception, "category", None), "environment")
        self.assertEqual(sink["abort"]["kind"], "device_missing")
        self.assertEqual(sink["abort"]["occupancy"], "missing")
        self.assertEqual(sink["missingMappingStreak"], 2)
        self.assertIn("no Ascend NPU mapped", str(caught.exception))
        print(f"runtime-health device-missing-to-exit measured={measured:.3f}s")

    def test_non_ok_health_is_recorded_without_killing_the_run(self):
        sink: dict = {}
        health = self._sequence([_row("unknown", "Warning")], _row("unknown", "Warning"))
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(GATE, "_probe_device_health", side_effect=health), \
                patch.object(GATE, "_probe_device_processes", side_effect=self._free_processes), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1), \
                self.lease_environment():
            with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                completed = GATE._run_host_bash_monitored(
                    "sleep 1.2",
                    cwd=self.root,
                    env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                    cann_setup=None,
                    conda_sh=None,
                    conda_env=None,
                    timeout=10,
                    isolate_temp=True,
                    mode="l4",
                    device="1",
                    monitor_sink=sink,
                )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(sink["outcome"], "completed")
        self.assertNotIn("abort", sink)
        self.assertEqual(sink["health"], "non_ok")
        self.assertGreaterEqual(sink["nonOkHealthStreak"], 1)
        self.assertIn("device_health_non_ok", sink["unknownSignals"])

    def test_stuck_probe_is_not_accumulated_and_is_separate_from_workload(self):
        self.residue = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
        )
        probe_pid = self.residue.pid
        hang = {
            "ok": False,
            "healthKind": "unavailable",
            "healthValue": None,
            "timedOut": True,
            "unreaped": True,
            "probePid": probe_pid,
            "wchan": "D",
            "error": "probe timed out",
        }
        calls: list[int] = []

        def stuck_health(*_args, **_kwargs):
            calls.append(1)
            return dict(hang)

        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(GATE, "_probe_device_health", side_effect=stuck_health), \
                patch.object(GATE, "_probe_device_processes", side_effect=self._free_processes), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.05), \
                patch.object(GATE, "HEALTH_POLL_SECONDS", 0.02), \
                patch.object(GATE, "HEALTH_PROBE_FAILURE_LIMIT", 2), \
                self.lease_environment():
            started = time.monotonic()
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    GATE._run_host_bash_monitored(
                        "sleep 30",
                        cwd=self.root,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                        cann_setup=None,
                        conda_sh=None,
                        conda_env=None,
                        timeout=30,
                        isolate_temp=True,
                        mode="l4",
                        device="1",
                        monitor_sink=sink,
                    )
            measured = time.monotonic() - started
        self.assertLess(measured, 5.0)
        self.assertEqual(sink["abort"]["kind"], "probe_unavailable")
        self.assertEqual(len(calls), 1, "new probes were launched into a wedged driver")
        self.assertIn(probe_pid, sink["stuckProbePidsObserved"])
        self.assertIn("device_probe_unreaped", sink["unknownSignals"])
        # The probe child is cleaned as a probe, never counted as workload residue.
        self.assertIn(probe_pid, sink["cleanup"]["probeCleanup"]["pids"])
        self.assertNotIn(probe_pid, sink["cleanup"]["pids"])
        self._assert_ended([probe_pid])
        self.assertIn("did not exit", str(caught.exception))
        print(
            f"runtime-health stuck-probe-to-exit measured={measured:.3f}s "
            f"probes={len(calls)} probeCleanup={sink['cleanup']['probeCleanup']['seconds']:.3f}s"
        )

    def test_indeterminate_quarantine_marker_stays_blocking(self):
        marker = Path(f"{self.lease}{GATE.QUARANTINE_SUFFIX}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{not-json", encoding="utf-8")
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                self.lease_environment():
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    pass
        self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
        self.assertEqual(getattr(caught.exception, "occupancy", None), "quarantined")
        self.assertIn("not valid JSON", str(caught.exception))
        self.assertTrue(marker.exists(), "an indeterminate marker must not be cleared")

    def test_unreapable_escaped_workload_returns_bounded_and_quarantines(self):
        """A D-state-like survivor must yield a bounded result, not wait for the recipe outer bound."""
        pidfile = self.root / "unreapable-pid"
        command = (
            f"setsid bash -c 'echo $$ > {pidfile}; exec sleep 30' & "
            "sleep 30"
        )
        escaped_pid: int | None = None

        def sweep_without_reaping(_token):
            nonlocal escaped_pid
            try:
                escaped_pid = int(pidfile.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                escaped_pid = None
            identity = (
                GATE._process_identity(escaped_pid)
                if escaped_pid is not None else None
            )
            return {
                "attempted": True,
                "seconds": 0.0,
                "pids": [escaped_pid] if escaped_pid is not None else [],
                "sigterm": [],
                "sigkill": [],
                "surviving": [escaped_pid] if escaped_pid is not None else [],
                "survivingProcesses": [identity] if identity is not None else [],
                "incomplete": True,
            }

        sink: dict = {}
        try:
            with patch.object(GATE, "_device_occupancy", return_value="free"), \
                    patch.object(
                        GATE, "_probe_npu_mapping",
                        return_value={"ok": True, "npuId": 7, "error": None},
                    ), \
                    patch.object(
                        GATE, "_probe_device_health",
                        side_effect=self._sequence([_row("ok", "OK")], _row("ok", "OK")),
                    ), \
                    patch.object(GATE, "_probe_device_processes", side_effect=self._free_processes), \
                    patch.object(GATE, "_kill_token_processes", side_effect=sweep_without_reaping), \
                    patch.object(GATE, "HEALTH_POLL_SECONDS", 0.05), \
                    patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1), \
                    patch.object(GATE, "HEALTH_DEADLINE_SWEEP_GRACE_SECONDS", 0.1), \
                    patch.object(GATE, "HEALTH_DEADLINE_EXIT_SECONDS", 0.2), \
                    self.lease_environment():
                started = time.monotonic()
                with self.assertRaises(GATE.BridgeError) as caught:
                    with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                        GATE._run_host_bash_monitored(
                            command,
                            cwd=self.root,
                            env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                            cann_setup=None,
                            conda_sh=None,
                            conda_env=None,
                            timeout=0.5,
                            isolate_temp=True,
                            mode="l4",
                            device="1",
                            monitor_sink=sink,
                        )
                measured = time.monotonic() - started
            self.assertLess(measured, 5.0)
            self.assertEqual(getattr(caught.exception, "category", None), "infra")
            self.assertEqual(getattr(caught.exception, "occupancy", None), "quarantined")
            self.assertEqual(sink["outcome"], "deadline_unreclaimable")
            self.assertFalse(sink["cleanup"]["workerJoined"])
            self.assertTrue(sink["cleanup"]["quarantine"]["written"])
            self.assertIsNotNone(escaped_pid)
            marker = Path(f"{self.lease}{GATE.QUARANTINE_SUFFIX}")
            self.assertTrue(marker.exists())
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["processes"][0]["pid"], escaped_pid)

            # The successor is fenced by the marker while the residue lives.
            with patch.object(GATE, "_device_occupancy", return_value="free"), \
                    self.lease_environment():
                with self.assertRaises(GATE.BridgeError) as blocked:
                    with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                        pass
            self.assertEqual(getattr(blocked.exception, "occupancy", None), "quarantined")
            print(
                f"runtime-health unreapable-deadline-result measured={measured:.3f}s "
                f"cleanup={sink['cleanup']['seconds']:.3f}s"
            )
        finally:
            if escaped_pid is not None:
                try:
                    os.kill(escaped_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _wait_ended(escaped_pid)
        marker = Path(f"{self.lease}{GATE.QUARANTINE_SUFFIX}")
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                self.lease_environment():
            with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                pass
        self.assertFalse(marker.exists())

    def test_quarantine_identity_survives_pid_reuse(self):
        identity = GATE._process_identity(os.getpid())
        self.assertIsNotNone(identity)
        marker = Path(f"{self.lease}{GATE.QUARANTINE_SUFFIX}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({
                "schema": GATE.QUARANTINE_SCHEMA,
                "resource": "npu:1",
                "device": "1",
                "reason": "simulated residue",
                "recordedAt": "2026-01-01T00:00:00Z",
                "processes": [{
                    **identity,
                    # The pid is alive but belongs to a different process
                    # instance than the one recorded in the marker.
                    "startTime": int(identity["startTime"]) + 1,
                }],
            }),
            encoding="utf-8",
        )
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                self.lease_environment():
            with GATE._open_device_window("1", wait_seconds=0, require_idle=True) as finalize:
                _, window = finalize()
            self.assertTrue(window["quarantine"]["cleared"])
            self.assertTrue(window["quarantine"]["removed"])
        self.assertFalse(marker.exists())

    def test_unreclaimable_residue_blocks_a_successor_until_it_exits(self):
        """Independent previous worker: it exits and releases the flock, but the marker blocks."""
        self.residue = subprocess.Popen(
            ["/bin/sleep", "30"],
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        worker = textwrap.dedent(
            """
            import importlib.util, json, os, sys
            from pathlib import Path
            from unittest.mock import patch

            script, lease, residue_pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
            spec = importlib.util.spec_from_file_location("tf_gate_worker", script)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            os.environ["AGENT_GATES_DEVICE_LEASE_PATH"] = lease
            os.environ["AGENT_GATES_DEVICE_LEASE_RESOURCE"] = "npu:1"
            os.environ["AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS"] = "0"
            identity = module._process_identity(residue_pid)
            assert identity is not None, "residue process is not observable"
            cleanup = {
                "attempted": True,
                "seconds": 0.0,
                "pids": [],
                "sigterm": [],
                "sigkill": [],
                "surviving": [residue_pid],
                "survivingProcesses": [identity],
                "incomplete": True,
            }
            with patch.object(module, "_device_occupancy", return_value="free"), \\
                    patch.object(module, "_kill_token_processes", return_value=cleanup):
                with module._open_device_window("1", wait_seconds=0, require_idle=True):
                    try:
                        module._run_host_bash_monitored(
                            "true",
                            cwd=Path.cwd(),
                            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
                            cann_setup=None,
                            conda_sh=None,
                            conda_env=None,
                            timeout=10,
                            isolate_temp=True,
                            mode="l4",
                            device="1",
                            monitor_sink={},
                        )
                    except module.BridgeError as exc:
                        print(json.dumps({
                            "category": exc.category,
                            "occupancy": exc.occupancy,
                            "message": str(exc),
                        }))
                    else:
                        raise SystemExit("worker did not observe cleanup residue")
            print("worker-exited")
            """
        )
        worker_path = self.root / "quarantine_worker.py"
        worker_path.write_text(worker, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-B", str(worker_path), str(SCRIPT), str(self.lease), str(self.residue.pid)],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ, "PATH": "/usr/bin:/bin", "HOME": str(self.root)},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("worker-exited", completed.stdout)
        record = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(record["category"], "infra")
        self.assertEqual(record["occupancy"], "quarantined")
        marker = Path(f"{self.lease}{GATE.QUARANTINE_SUFFIX}")
        self.assertTrue(marker.exists(), "the previous worker did not leave a marker")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], GATE.QUARANTINE_SCHEMA)
        self.assertEqual(payload["processes"][0]["pid"], self.residue.pid)
        self.assertEqual(payload["resource"], "npu:1")

        # The previous process is gone, so the old flock is released; the
        # marker still fences this card before occupancy/workload.
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                self.lease_environment():
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    pass
            self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
            self.assertEqual(getattr(caught.exception, "occupancy", None), "quarantined")
            self.assertIn(str(self.residue.pid), str(caught.exception))
            self.assertIsNone(self.residue.poll(), "the residue was touched by the successor")

            self.residue.kill()
            self.residue.wait(timeout=5)
            self.residue = None
            self.assertTrue(_wait_ended(payload["processes"][0]["pid"]))
            with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                pass
            self.assertFalse(marker.exists(), "a confirmed-exited marker was not cleared")

    def test_silent_healthy_workload_crosses_former_stall_threshold_and_completes(self):
        """A quiet, healthy workload is bounded only by the total deadline.

        The monotonic clock is a controlled proxy: every observation advances
        20 simulated seconds, so the recorded run crosses the former 120 s
        silence threshold while the real command stays short. The total budget
        is 600 simulated seconds, so crossing that old threshold must not be
        mistaken for a stall and must not end the run early.
        """
        clock = _AdvancingClock(20.0)
        probes: list[float] = []

        def healthy(_npu_id, _timeout):
            probes.append(clock.value)
            return _row("ok", "OK")

        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(GATE, "_probe_device_health", side_effect=healthy), \
                patch.object(GATE, "_probe_device_processes", side_effect=self._free_processes), \
                patch.object(GATE, "time", clock), \
                self.lease_environment():
            with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                before = clock.value
                completed = GATE._run_host_bash_monitored(
                    "sleep 2.2",
                    cwd=self.root,
                    env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                    cann_setup=None,
                    conda_sh=None,
                    conda_env=None,
                    timeout=600,
                    isolate_temp=True,
                    mode="l4",
                    device="1",
                    monitor_sink=sink,
                    phase_getter=lambda: "silent-kernel",
                )
                simulated_elapsed = clock.value - before

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "", "the workload must stay silent")
        self.assertEqual(completed.stderr, "", "the workload must produce no log")
        self.assertEqual(sink["outcome"], "completed")
        self.assertNotIn("abort", sink)
        self.assertEqual(sink["health"], "ok")
        self.assertEqual(
            (sink.get("lastProbe") or {}).get("health", {}).get("healthKind"),
            "ok",
        )
        self.assertGreaterEqual(len(probes), 3, "health was not probed repeatedly")
        self.assertGreater(simulated_elapsed, 120.0, "did not cross the former stall threshold")
        self.assertLess(simulated_elapsed, 600.0, "the total deadline was not preserved")
        self.assertNotIn("deadlineSweep", sink["cleanup"])
        print(
            "runtime-health silent-healthy-cross-threshold "
            f"simulated={simulated_elapsed:.3f}s probes={len(probes)} "
            f"real={2.2:.3f}s"
        )

    def test_quarantine_on_one_card_does_not_block_another_card(self):
        """The marker is per-card; a residue on npu:1 must not fence npu:2."""
        self.residue = subprocess.Popen(
            ["/bin/sleep", "30"],
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        identity = GATE._process_identity(self.residue.pid)
        self.assertIsNotNone(identity)
        marker = Path(f"{self.lease}{GATE.QUARANTINE_SUFFIX}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({
                "schema": GATE.QUARANTINE_SCHEMA,
                "resource": "npu:1",
                "device": "1",
                "reason": "simulated residue on npu:1",
                "recordedAt": "2026-01-01T00:00:00Z",
                "processes": [identity],
            }),
            encoding="utf-8",
        )
        other_lease = self.root / "locks" / "npu-2.lock"
        other_marker = Path(f"{other_lease}{GATE.QUARANTINE_SUFFIX}")

        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                self.lease_environment():
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    pass
            self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
            self.assertEqual(getattr(caught.exception, "occupancy", None), "quarantined")

            with patch.dict(
                os.environ,
                {
                    "AGENT_GATES_DEVICE_LEASE_PATH": str(other_lease),
                    "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:2",
                    "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "0",
                },
                clear=False,
            ):
                with GATE._open_device_window("2", wait_seconds=0, require_idle=True) as finalize:
                    occupancy, window = finalize()
            self.assertEqual(occupancy, "free")
            self.assertEqual(window["resource"], "npu:2")
            self.assertNotIn("quarantine", window)

        self.assertTrue(marker.exists(), "the npu:1 marker was removed by another card's window")
        self.assertFalse(other_marker.exists(), "a marker was created for the unrelated card")
        self.assertIsNone(self.residue.poll(), "another card's window touched this card's residue")
