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

    def test_configured_external_probe_limit_aborts_after_one_probe(self):
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
        processes = self._sequence(
            [self._free_processes(), external_row], external_row
        )
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
                        external_probe_limit=1,
                        phase_getter=lambda: "workload",
                    )
        self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
        self.assertEqual(sink["abort"]["kind"], "external_occupancy")
        self.assertIn("for 1 consecutive probes", str(caught.exception))
        self.assertIsNone(self.external.poll(), "an unrelated tenant process was killed")

    def test_l4_idle_preflight_accepts_a_consecutively_free_card(self):
        with patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 7)):
            observed = GATE._l4_idle_preflight("1", confirmations=3)
        self.assertEqual(
            observed,
            {"npuId": 7, "occupancy": "free", "attempts": 1, "confirmations": 3},
        )

    def test_l4_idle_preflight_refuses_an_occupied_card(self):
        occupied = ([{"pid": "4242", "name": "other"}], 7)
        with patch.object(GATE.adapter_runtime, "processes_on", return_value=occupied), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.01):
            with self.assertRaises(GATE.BridgeError) as caught:
                GATE._l4_idle_preflight("1", confirmations=3)
        self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
        self.assertEqual(getattr(caught.exception, "occupancy", None), "shared(1)")
        self.assertIn("pid=4242(other)", str(caught.exception))
        self.assertIn("3 consecutive probes", str(caught.exception))

    def test_l4_idle_preflight_reports_unobservable_occupancy(self):
        def blocked(_device):
            raise GATE.adapter_runtime.BlockedEnv("npu-smi unavailable")

        with patch.object(GATE.adapter_runtime, "processes_on", side_effect=blocked):
            with self.assertRaises(GATE.BridgeError) as caught:
                GATE._l4_idle_preflight("1", confirmations=3)
        self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
        self.assertEqual(getattr(caught.exception, "occupancy", None), "unknown")
        self.assertIn("could not determine device occupancy", str(caught.exception))

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

    def test_l3_external_occupancy_ends_own_run_but_never_kills_tenant(self):
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
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    GATE._run_host_bash_monitored(
                        "sleep 30",
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
        self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
        self.assertEqual(getattr(caught.exception, "occupancy", None), "shared(1)")
        self.assertEqual(sink["abort"]["kind"], "external_occupancy")
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

    def test_probe_retries_transient_exit_code_215_and_succeeds(self):
        calls = []

        def fake_cmd(argv, _timeout):
            calls.append(argv)
            if len(calls) < 3:
                return {
                    "argv": argv,
                    "pid": 100 + len(calls),
                    "returncode": 215,
                    "stdout": "",
                    "stderr": "transient",
                    "seconds": 0.01,
                    "timedOut": False,
                    "unreaped": False,
                    "wchan": None,
                    "error": None,
                }
            return {
                "argv": argv,
                "pid": 103,
                "returncode": 0,
                "stdout": "\tNPU ID : 7\n\tNo process in device.\n",
                "stderr": "",
                "seconds": 0.01,
                "timedOut": False,
                "unreaped": False,
                "wchan": None,
                "error": None,
            }

        with patch.object(GATE, "_bounded_host_command", side_effect=fake_cmd), \
                patch.object(GATE.time, "sleep") as sleep_mock:
            res = GATE._probe_device_processes(7, 5.0)
        self.assertTrue(res["ok"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [call.args[0] for call in sleep_mock.call_args_list],
            list(GATE.adapter_runtime.SMI_RETRY_BACKOFF_SECONDS),
        )

    def test_fast_post_run_pre_release_intrusion_check_aborts_under_2s(self):
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
        # First probe during run sees free; post-run pre-release check sees intruder.
        processes = self._sequence([self._free_processes(), external_row], external_row)
        health = self._sequence([_row("ok", "OK")], _row("ok", "OK"))
        sink: dict = {}
        with patch.object(GATE, "_device_occupancy", return_value="free"), \
                patch.object(
                    GATE, "_probe_npu_mapping",
                    return_value={"ok": True, "npuId": 7, "error": None},
                ), \
                patch.object(GATE, "_probe_device_health", side_effect=health), \
                patch.object(GATE, "_probe_device_processes", side_effect=processes), \
                patch.object(GATE, "HEALTH_POLL_SECONDS", 0.05), \
                patch.object(GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 10.0), \
                self.lease_environment():
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window("1", wait_seconds=0, require_idle=True):
                    GATE._run_host_bash_monitored(
                        "sleep 0.25",
                        cwd=self.root,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                        cann_setup=None,
                        conda_sh=None,
                        conda_env=None,
                        timeout=10,
                        isolate_temp=True,
                        mode="fast",
                        device="1",
                        monitor_sink=sink,
                        external_probe_limit=1,
                    )
        self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
        self.assertEqual(getattr(caught.exception, "occupancy", None), "shared(1)")
        self.assertEqual(sink["abort"]["kind"], "external_occupancy")
        self.assertIsNone(self.external.poll())

    def test_occupancy_policy_parsing_and_validation(self):
        self.assertEqual(GATE.occupancy_policy({}), "shared_quiet")
        self.assertEqual(
            GATE.occupancy_policy({"TILEFOUNDRY_NPU_OCCUPANCY_POLICY": "EXCLUSIVE"}),
            "exclusive",
        )
        self.assertEqual(
            GATE.occupancy_policy({"TILEFOUNDRY_NPU_OCCUPANCY_POLICY": "SHARED_QUIET"}),
            "shared_quiet",
        )
        self.assertEqual(GATE.shared_max_ext_procs({}), 6)
        self.assertEqual(
            GATE.shared_max_ext_procs({"TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS": "4"}), 4
        )
        self.assertEqual(GATE.shared_max_hbm_mb({}), 16384)
        self.assertEqual(
            GATE.shared_max_hbm_mb({"TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB": "8192"}), 8192
        )
        for bad in (
            {"TILEFOUNDRY_NPU_OCCUPANCY_POLICY": "anything"},
            {"TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS": "0"},
            {"TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS": "abc"},
            {"TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB": "128"},
            {"TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB": "xyz"},
        ):
            with self.assertRaises(GATE.BridgeError):
                if "TILEFOUNDRY_NPU_OCCUPANCY_POLICY" in bad:
                    GATE.occupancy_policy(bad)
                elif "TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS" in bad:
                    GATE.shared_max_ext_procs(bad)
                else:
                    GATE.shared_max_hbm_mb(bad)

    def test_shared_quiet_admission_records_honest_shared_and_rejects_heavy_hbm(self):
        light_procs = [
            {"pid": "111", "name": "python", "memoryMb": "1024"},
            {"pid": "222", "name": "python", "memoryMb": "2048"},
        ]
        heavy_procs = [
            {"pid": "333", "name": "python", "memoryMb": "47134"},
        ]
        with patch.object(
            GATE.adapter_runtime, "processes_on", return_value=(light_procs, 1)
        ), self.lease_environment():
            with GATE._open_device_window(
                "1",
                wait_seconds=0,
                require_idle=True,
                occupancy_policy="shared_quiet",
                max_ext_procs=6,
                max_hbm_mb=16384,
            ) as finalize:
                occ, win = finalize()
            self.assertEqual(occ, "shared(2)")
            self.assertEqual(win["occupancy"], "shared(2)")
            self.assertEqual(win["occupancyPolicy"], "shared_quiet")
            self.assertEqual(win["externalProcessCount"], 2)
            self.assertEqual(win["externalMemoryMb"], 3072)
            self.assertNotIn("requiredIdle", win)

            l4_preflight = GATE._l4_idle_preflight(
                "1",
                confirmations=3,
                occupancy_policy="shared_quiet",
                max_ext_procs=6,
                max_hbm_mb=16384,
            )
            self.assertEqual(l4_preflight["occupancy"], "shared(2)")
            self.assertEqual(l4_preflight["policy"], "shared_quiet")
            self.assertEqual(l4_preflight["externalMemoryMb"], 3072)

        with patch.object(
            GATE.adapter_runtime, "processes_on", return_value=(heavy_procs, 0)
        ), self.lease_environment():
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window(
                    "0",
                    wait_seconds=0,
                    require_idle=True,
                    occupancy_policy="shared_quiet",
                    max_ext_procs=6,
                    max_hbm_mb=16384,
                ):
                    pass
            self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
            self.assertEqual(getattr(caught.exception, "occupancy", None), "shared(1)")

    def test_shared_quiet_monitor_allows_light_tolerates_l3_and_aborts_heavy_hbm(self):
        self.external = subprocess.Popen(
            ["/bin/sleep", "30"],
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        light_row = {
            "ok": True,
            "processTableSeen": True,
            "processes": [{"pid": str(self.external.pid), "name": "sleep", "memoryMb": "1024"}],
            "error": None,
        }
        heavy_row = {
            "ok": True,
            "processTableSeen": True,
            "processes": [{"pid": str(self.external.pid), "name": "sleep", "memoryMb": "20480"}],
            "error": None,
        }
        health = self._sequence([_row("ok", "OK")], _row("ok", "OK"))

        # 1. fast under shared_quiet with light background completes cleanly.
        sink_light: dict = {}
        with patch.object(
            GATE.adapter_runtime,
            "processes_on",
            return_value=([{"pid": str(self.external.pid), "name": "sleep", "memoryMb": "1024"}], 1),
        ), patch.object(
            GATE, "_probe_npu_mapping", return_value={"ok": True, "npuId": 1, "error": None}
        ), patch.object(
            GATE, "_probe_device_health", side_effect=health
        ), patch.object(
            GATE, "_probe_device_processes", return_value=light_row
        ), patch.object(GATE, "HEALTH_POLL_SECONDS", 0.05), patch.object(
            GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1
        ), self.lease_environment():
            with GATE._open_device_window(
                "1", wait_seconds=0, require_idle=True, occupancy_policy="shared_quiet"
            ):
                done = GATE._run_host_bash_monitored(
                    "sleep 0.3",
                    cwd=self.root,
                    env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                    cann_setup=None,
                    conda_sh=None,
                    conda_env=None,
                    timeout=10,
                    isolate_temp=True,
                    mode="fast",
                    device="1",
                    monitor_sink=sink_light,
                    external_probe_limit=1,
                    occupancy_policy="shared_quiet",
                    shared_max_ext_procs=6,
                    shared_max_hbm_mb=16384,
                    allow_external_in_shared=False,
                )
        self.assertEqual(done.returncode, 0)
        self.assertEqual(sink_light["outcome"], "completed")
        self.assertNotIn("abort", sink_light)

        # 2. fast under shared_quiet aborts when mid-run external HBM exceeds budget.
        sink_heavy: dict = {}
        with patch.object(
            GATE.adapter_runtime, "processes_on", return_value=([], 1)
        ), patch.object(
            GATE, "_probe_npu_mapping", return_value={"ok": True, "npuId": 1, "error": None}
        ), patch.object(
            GATE, "_probe_device_health", side_effect=health
        ), patch.object(
            GATE, "_probe_device_processes", return_value=heavy_row
        ), patch.object(GATE, "HEALTH_POLL_SECONDS", 0.05), patch.object(
            GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1
        ), self.lease_environment():
            with self.assertRaises(GATE.BridgeError) as caught:
                with GATE._open_device_window(
                    "1", wait_seconds=0, require_idle=True, occupancy_policy="shared_quiet"
                ):
                    GATE._run_host_bash_monitored(
                        "sleep 5",
                        cwd=self.root,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                        cann_setup=None,
                        conda_sh=None,
                        conda_env=None,
                        timeout=10,
                        isolate_temp=True,
                        mode="fast",
                        device="1",
                        monitor_sink=sink_heavy,
                        external_probe_limit=1,
                        occupancy_policy="shared_quiet",
                        shared_max_ext_procs=6,
                        shared_max_hbm_mb=16384,
                        allow_external_in_shared=False,
                    )
        self.assertEqual(getattr(caught.exception, "category", None), "occupancy")
        self.assertEqual(sink_heavy["abort"]["kind"], "external_occupancy")

        # 3. l3 ignores proc-count cap at admission and allow_external_in_shared tolerates mid-run heavy HBM.
        many_light_procs = [
            {"pid": str(self.external.pid), "name": "sleep", "memoryMb": "128"}
            for _ in range(8)
        ]
        sink_l3: dict = {}
        with patch.object(
            GATE.adapter_runtime,
            "processes_on",
            return_value=(many_light_procs, 1),
        ), patch.object(
            GATE, "_probe_npu_mapping", return_value={"ok": True, "npuId": 1, "error": None}
        ), patch.object(
            GATE, "_probe_device_health", side_effect=health
        ), patch.object(
            GATE, "_probe_device_processes", return_value=heavy_row
        ), patch.object(GATE, "HEALTH_POLL_SECONDS", 0.05), patch.object(
            GATE, "HEALTH_PROBE_INTERVAL_SECONDS", 0.1
        ), self.lease_environment():
            with GATE._open_device_window(
                "1",
                wait_seconds=0,
                require_idle=True,
                occupancy_policy="shared_quiet",
                enforce_proc_count=False,
            ):
                done_l3 = GATE._run_host_bash_monitored(
                    "sleep 0.3",
                    cwd=self.root,
                    env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
                    cann_setup=None,
                    conda_sh=None,
                    conda_env=None,
                    timeout=10,
                    isolate_temp=True,
                    mode="l3",
                    device="1",
                    monitor_sink=sink_l3,
                    external_probe_limit=2,
                    occupancy_policy="shared_quiet",
                    shared_max_ext_procs=6,
                    shared_max_hbm_mb=16384,
                    allow_external_in_shared=True,
                )
        self.assertEqual(done_l3.returncode, 0)
        self.assertEqual(sink_l3["outcome"], "completed")
        self.assertNotIn("abort", sink_l3)

    def test_probe_npu_usages_parses_rates_and_omits_c_when_chip_id_none(self):
        calls: list[list[str]] = []
        sample_output = (
            "\tNPU ID                         : 0\n"
            "\tHBM Usage Rate(%)              : 3\n"
            "\tAicore Usage Rate(%)           : 17\n"
            "\tHBM Bandwidth Usage Rate(%)    : 42\n"
        )

        def fake_cmd(argv, _timeout):
            calls.append(list(argv))
            return {
                "argv": argv,
                "pid": 201,
                "returncode": 0,
                "stdout": sample_output,
                "stderr": "",
                "seconds": 0.01,
                "timedOut": False,
                "unreaped": False,
                "wchan": None,
                "error": None,
            }

        with patch.object(GATE, "_bounded_host_command", side_effect=fake_cmd):
            usages = GATE._probe_npu_usages(0)
        self.assertEqual(usages, {"aicoreUsagePct": 17, "hbmBandwidthUsagePct": 42})
        self.assertEqual(len(calls), 1)
        self.assertNotIn("-c", calls[0])
        self.assertEqual(calls[0][-4:], ["-t", "usages", "-i", "0"])

    def test_probe_npu_usages_retries_without_c_when_unsupported(self):
        calls: list[list[str]] = []
        success_output = (
            "\tNPU ID                         : 0\n"
            "\tAicore Usage Rate(%)           : 5\n"
            "\tHBM Bandwidth Usage Rate(%)    : 12\n"
        )

        def fake_cmd(argv, _timeout):
            calls.append(list(argv))
            if "-c" in argv:
                return {
                    "argv": argv,
                    "pid": 202,
                    "returncode": 255,
                    "stdout": "This device does not support input parameter of -c.\n",
                    "stderr": "",
                    "seconds": 0.01,
                    "timedOut": False,
                    "unreaped": False,
                    "wchan": None,
                    "error": None,
                }
            return {
                "argv": argv,
                "pid": 203,
                "returncode": 0,
                "stdout": success_output,
                "stderr": "",
                "seconds": 0.01,
                "timedOut": False,
                "unreaped": False,
                "wchan": None,
                "error": None,
            }

        with patch.object(GATE, "_bounded_host_command", side_effect=fake_cmd):
            usages = GATE._probe_npu_usages(0, chip_id=0)
        self.assertEqual(usages, {"aicoreUsagePct": 5, "hbmBandwidthUsagePct": 12})
        self.assertEqual(len(calls), 2)
        self.assertIn("-c", calls[0])
        self.assertNotIn("-c", calls[1])

    def test_probe_npu_usages_degrades_gracefully_and_window_evidence_attaches(self):
        def failing_cmd(argv, _timeout):
            return {
                "argv": argv,
                "pid": 204,
                "returncode": 1,
                "stdout": "",
                "stderr": "unsupported",
                "seconds": 0.01,
                "timedOut": False,
                "unreaped": False,
                "wchan": None,
                "error": None,
            }

        with patch.object(GATE, "_bounded_host_command", side_effect=failing_cmd):
            self.assertEqual(GATE._probe_npu_usages(0), {})

        def timeout_cmd(argv, _timeout):
            return {
                "argv": argv,
                "pid": 205,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "seconds": 3.0,
                "timedOut": True,
                "unreaped": False,
                "wchan": None,
                "error": "probe timed out",
            }

        with patch.object(GATE, "_bounded_host_command", side_effect=timeout_cmd):
            self.assertEqual(GATE._probe_npu_usages(0, chip_id=0), {})

        win = {"resource": "npu:0", "npuId": 0, "occupancy": "free"}
        with patch.object(
            GATE,
            "_probe_npu_usages",
            return_value={"aicoreUsagePct": 23, "hbmBandwidthUsagePct": 61},
        ):
            ev = GATE._window_evidence("free", win)
        self.assertTrue(ev["windowOpened"])
        self.assertTrue(ev["windowReleased"])
        self.assertTrue(ev["occupancyRecorded"])
        self.assertEqual(ev["deviceWindow"], win)
        self.assertEqual(ev["aicoreUsagePct"], 23)
        self.assertEqual(ev["hbmBandwidthUsagePct"], 61)

        with patch.object(GATE, "_probe_npu_usages", return_value={}):
            ev_empty = GATE._window_evidence("free", win)
        self.assertNotIn("aicoreUsagePct", ev_empty)
        self.assertNotIn("hbmBandwidthUsagePct", ev_empty)

    def test_detect_npu_device_fault_signatures_and_non_matches(self):
        positives = (
            ("E39006", "EE9999: E39006 init failed", ""),
            (
                "Inner_Error_Failed_Send_OPP_To_Device",
                "",
                "Inner_Error_Failed_Send_OPP_To_Device while loading",
            ),
            ("TsdOpen failed", "TsdOpen failed, devId=0", ""),
            ("tdt error", "", "TDT error occurred on channel"),
            ("507033", "aclrtSynchronizeStream failed, error code: 507033", ""),
            ("ACL_ERROR_RT_DEVICE", "ret=ACL_ERROR_RT_DEVICE", ""),
            ("ACL_ERROR_RT_DEVICE_FAILURE", "", "status=ACL_ERROR_RT_DEVICE_FAILURE"),
            ("drvError", "drvError=0x4 during submit", ""),
            ("HCCL_E_INTERNAL", "", "HCCL_E_INTERNAL: link down"),
            ("uncorrectable ecc", "HBM uncorrectable ecc detected", ""),
            ("ecc error", "", "fatal ECC error on chip 0"),
        )
        for expected, out, err in positives:
            with self.subTest(expected=expected):
                sig = GATE._detect_npu_device_fault(out, err)
                self.assertIsNotNone(sig)
                assert sig is not None
                self.assertEqual(sig.lower(), expected.lower())

        negatives = (
            ("RuntimeError: AICore trap at pc=0x10", ""),
            ("", "AssertionError: numeric mismatch max_diff=0.25"),
            ("max_abs_diff=0.5 > tol=1e-3", "candidate failed"),
            ("sha256=deadbeef507033cafe", ""),
            ("", ""),
        )
        for out, err in negatives:
            with self.subTest(out=out, err=err):
                self.assertIsNone(GATE._detect_npu_device_fault(out, err))

    def test_device_probe_classifies_npu_device_fault_as_75_and_aicore_trap_as_1(self):
        src = self.root / "tf-src"
        (src / "src" / "tilefoundry").mkdir(parents=True)
        cann = self.root / "set_env.sh"
        cann.write_text(":\n", encoding="utf-8")
        conda_sh = self.root / "conda.sh"
        conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        env_file = self.root / "machine.env"
        env_file.write_text(
            "".join(
                f"{k}={v}\n"
                for k, v in {
                    "TILEFOUNDRY_NPU_CANN_SETUP": str(cann),
                    "TILEFOUNDRY_NPU_CONDA_SH": str(conda_sh),
                    "TILEFOUNDRY_NPU_CONDA_ENV": "tilefoundry-npu",
                    "TILEFOUNDRY_NPU_PYTHON": sys.executable,
                }.items()
            ),
            encoding="utf-8",
        )
        run_env = {
            "TILEFOUNDRY_NPU_ENV_FILE": str(env_file),
            "TILEFOUNDRY_NPU_DEVICE_ID": "1",
            "AGENT_GATES_DEVICE_LEASE_PATH": str(self.lease),
            "AGENT_GATES_DEVICE_LEASE_RESOURCE": "npu:1",
            "AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS": "0",
        }
        old_cwd = os.getcwd()
        os.chdir(src)
        self.addCleanup(os.chdir, old_cwd)

        for sig, out, err in (
            ("E39006", "E39006: device init failed", ""),
            ("TsdOpen failed", "", "TsdOpen failed on device 1"),
            ("507033", "aclrtSynchronizeStream failed ret=507033", ""),
        ):
            with self.subTest(sig=sig):
                completed = subprocess.CompletedProcess(
                    args=["bash"], returncode=1, stdout=out, stderr=err
                )
                with patch.dict(os.environ, run_env, clear=True), \
                        patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                        patch.object(GATE, "_run_host_bash_monitored", return_value=completed), \
                        patch.object(GATE.locking, "lock_root", return_value=self.root / "locks"):
                    rc, obs = GATE.run_device(GATE.machine_values(), 60.0, fast=False)
                self.assertEqual(rc, 75)
                self.assertFalse(obs["exitCodeZero"])
                self.assertTrue(obs["windowOpened"])
                self.assertEqual(obs["errorCategory"], "device_fault")
                self.assertEqual(obs["failureCategory"], "device_fault")
                self.assertIs(obs["deviceRejected"], True)
                self.assertEqual(obs["deviceFaultSignature"], sig)

        trap = subprocess.CompletedProcess(
            args=["bash"],
            returncode=1,
            stdout="",
            stderr='RuntimeError: AICore trap during kernel execution',
        )
        with patch.dict(os.environ, run_env, clear=True), \
                patch.object(GATE.adapter_runtime, "processes_on", return_value=([], 1)), \
                patch.object(GATE, "_run_host_bash_monitored", return_value=trap), \
                patch.object(GATE.locking, "lock_root", return_value=self.root / "locks"):
            rc, obs = GATE.run_device(GATE.machine_values(), 60.0, fast=False)
        self.assertEqual(rc, 1)
        self.assertFalse(obs["exitCodeZero"])
        self.assertEqual(obs["errorCategory"], "candidate")
        self.assertNotIn("deviceRejected", obs)
        self.assertNotIn("deviceFaultSignature", obs)


