"""Unit tests for the shared host-local adapter runtime helpers."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agent_gates import adapter_runtime


class ParseEnvFileCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_parses_plain_and_quoted_assignments(self):
        env_file = self.root / "env.sh"
        env_file.write_text(
            "# comment\n"
            "FOO_A=/opt/cann/set_env.sh\n"
            "export FOO_B='hello world'\n"
            'FOO_C=""\n',
            encoding="utf-8",
        )
        parsed = adapter_runtime.parse_env_file(env_file, ("FOO_A", "FOO_B", "FOO_C"))
        self.assertEqual(
            parsed,
            {"FOO_A": "/opt/cann/set_env.sh", "FOO_B": "hello world", "FOO_C": ""},
        )

    def test_rejects_empty_unknown_duplicate_metachar_and_symlink(self):
        empty = self.root / "empty.sh"
        empty.write_text("# only comments\n\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            adapter_runtime.parse_env_file(empty, ("FOO_A",))

        unknown = self.root / "unknown.sh"
        unknown.write_text("FOO_BAD=1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            adapter_runtime.parse_env_file(unknown, ("FOO_A",))

        dup = self.root / "dup.sh"
        dup.write_text("FOO_A=1\nFOO_A=2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            adapter_runtime.parse_env_file(dup, ("FOO_A",))

        sub = self.root / "sub.sh"
        sub.write_text("FOO_A=$(whoami)\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            adapter_runtime.parse_env_file(sub, ("FOO_A",))

        real = self.root / "real.sh"
        real.write_text("FOO_A=1\n", encoding="utf-8")
        link = self.root / "link.sh"
        link.symlink_to(real)
        with self.assertRaises(ValueError):
            adapter_runtime.parse_env_file(link, ("FOO_A",))


class SmiAndDeviceWindowCase(unittest.TestCase):
    MAP_5COL = (
        "\tNPU ID  Slot ID  Chip ID  Chip Phy-ID  Chip Name\n"
        "\t0       74       0        0            Ascend950PR\n"
        "\t1       76       0        1            Ascend950PR\n"
    )
    MAP_4COL = (
        "\tNPU ID  Chip ID  Chip Phy-ID  Chip Name\n"
        "\t3       0        1            Ascend910B\n"
    )

    def test_npu_id_of_physical_supports_5col_and_4col_layouts(self):
        with patch.object(adapter_runtime, "_run_smi", return_value=self.MAP_5COL):
            self.assertEqual(adapter_runtime.npu_id_of_physical(1), 1)
        with patch.object(adapter_runtime, "_run_smi", return_value=self.MAP_4COL):
            self.assertEqual(adapter_runtime.npu_id_of_physical(1), 3)

    def test_processes_on_handles_blank_process_name(self):
        proc_out = (
            "\tNPU ID                         : 0\n"
            "\tProcess id:2330713 Process name:                  Process memory(MB):796\n"
            "\tProcess id:12345   Process name:python3           Process memory(MB):512\n"
        )

        def fake_smi(args, **_kwargs):
            return self.MAP_5COL if args == ["info", "-m"] else proc_out

        with patch.object(adapter_runtime, "_run_smi", side_effect=fake_smi):
            procs, npu_id = adapter_runtime.processes_on(0)
        self.assertEqual(npu_id, 0)
        self.assertEqual(
            procs,
            [
                {"pid": "2330713", "name": "<unknown>", "memoryMb": "796"},
                {"pid": "12345", "name": "python3", "memoryMb": "512"},
            ],
        )

    def test_check_device_ready_and_device_window_enforce_exclusivity(self):
        busy = (
            "\tNPU ID : 1\n"
            "\tProcess id:42 Process name:python3 Process memory(MB):100\n"
        )

        def fake_smi(args, **_kwargs):
            return self.MAP_5COL if args == ["info", "-m"] else busy

        with patch.object(adapter_runtime, "_run_smi", side_effect=fake_smi):
            shared = adapter_runtime.check_device_ready(1, exclusive=False)
            self.assertEqual(len(shared["processes"]), 1)
            with self.assertRaises(adapter_runtime.BlockedEnv):
                adapter_runtime.check_device_ready(1, exclusive=True)

            with tempfile.TemporaryDirectory() as td:
                lock_root = Path(td)
                idle = "\tNPU ID : 1\n\tNo process in device.\n"
                with patch.object(
                    adapter_runtime,
                    "_run_smi",
                    side_effect=lambda args, **kw: self.MAP_5COL if args == ["info", "-m"] else idle,
                ):
                    with adapter_runtime.device_window(lock_root, 1, exclusive=True) as info:
                        self.assertEqual(info["npuId"], 1)
                        with self.assertRaises(adapter_runtime.BlockedEnv):
                            with adapter_runtime.device_window(lock_root, 1, exclusive=False, wait_seconds=0.0):
                                pass


class RunHostBashCase(unittest.TestCase):
    def test_set_plus_u_cann_and_temp_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cann = root / "set_env.sh"
            # Intentionally references an unbound variable under set -u.
            cann.write_text('export TEST_CANN="${UNBOUND_VAR:-ok}"\nTEST_PROBE=$UNBOUND_2\n', encoding="utf-8")
            done = adapter_runtime.run_host_bash(
                'echo "CANN=$TEST_CANN WORK=$ASCEND_WORK_PATH"',
                cwd=root,
                cann_setup=cann,
                isolate_temp=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("CANN=ok WORK=", done.stdout)
            work_path = done.stdout.strip().split("WORK=")[1]
            self.assertFalse(Path(work_path).exists())

    def test_timeout_kills_process_group(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(subprocess.TimeoutExpired):
                adapter_runtime.run_host_bash(
                    "sleep 10 & wait",
                    cwd=Path(td),
                    timeout=0.2,
                    isolate_temp=True,
                )


if __name__ == "__main__":
    unittest.main()
