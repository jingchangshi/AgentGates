"""Two runs, one card: the window must serialise them, never refuse them.

This is the property the shared-machine design exists for, and it cannot be
proved with one process. Two independent processes each declare the same card and
each open a window; the test asserts that

* neither run is refused - no BLOCKED_ENV is produced by our own contention;
* the windows do not overlap, because the lease is still exclusive while held;
* each window was entered, did its work and left, so the card is free between
  them.

The processes are real subprocesses with the real adapter script, so this
exercises the same flock the engine hands a windowed recipe.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / 'agent_gates/adapters/forgekernel-ascend-npu/scripts/fk_gate.py'


class TwoProcessWindowCase(unittest.TestCase):
    """Run the Adapter's window in two processes against one lease file."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lease = self.root / 'locks' / 'npu.lock'

    def spawn(self, *, wait, hold, tag):
        env = dict(os.environ)
        env.update({
            'AGENT_GATES_DEVICE_LEASE_PATH': str(self.lease),
            'AGENT_GATES_DEVICE_LEASE_RESOURCE': 'npu:3',
            'AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS': str(wait),
            'PATH': os.environ.get('PATH', ''),
        })
        return subprocess.Popen([sys.executable, '-I', str(GATE), '--probe-window',
                                 '--hold', str(hold)],
                                cwd=str(self.root), env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_two_processes_sharing_one_card_do_not_block_each_other(self):
        # Both are launched together; the second must WAIT for the first rather
        # than fail. A refusal here would be the bug this design removes.
        first = self.spawn(wait=30, hold=0.5, tag='a')
        second = self.spawn(wait=30, hold=0.2, tag='b')
        results = []
        for proc in (first, second):
            out, err = proc.communicate(timeout=90)
            results.append((proc.returncode, out, err))
        for index, (code, out, err) in enumerate(results):
            self.assertEqual(code, 0, f'run {index} exited {code}: {err}')
            self.assertNotIn('BLOCKED', err, f'run {index} was blocked: {err}')
        windows = [json.loads(out)['deviceWindow'] for _, out, _ in results]
        for window in windows:
            self.assertEqual(window['resource'], 'npu:3')
            self.assertEqual(window['path'], str(self.lease))
            self.assertIsNotNone(window['releasedAt'])
        # Exactly one of them had to wait: that is the serialisation, and it is
        # an expected cost of sharing one card, not an error.
        self.assertTrue(any(window['waited'] > 0.05 for window in windows),
                        f'neither run waited, so they overlapped: {windows}')
        # The windows are disjoint in time: one ends before the other begins.
        ordered = sorted(windows, key=lambda window: window['startedAt'])
        self.assertLessEqual(ordered[0]['releasedAt'], ordered[1]['startedAt'],
                             f'windows overlapped: {windows}')

    def test_an_occupied_card_is_recorded_and_never_refused(self):
        # On this class of machine another tenant on the card is the NORMAL case,
        # so the window must open anyway and report what it saw. This test does
        # not require a quiet card to exist: it requires that sharing one is not
        # an error, which is the property that matters and the one that can be
        # checked on a busy machine.
        first = self.spawn(wait=30, hold=0.2, tag='a')
        out, err = first.communicate(timeout=90)
        self.assertEqual(first.returncode, 0, f'an occupied card refused the run: {err}')
        window = json.loads(out)['deviceWindow']
        occupancy = window['occupancy']
        # A defined value, always: "unknown" is allowed, silence is not.
        self.assertTrue(occupancy == 'free' or occupancy == 'unknown'
                        or re.fullmatch(r'shared\([0-9]+\)', occupancy), occupancy)
        self.assertIsNotNone(window['releasedAt'])

    def test_a_third_process_can_take_the_lease_after_the_windows_close(self):
        # "Occupied only while running" has to mean the lease is genuinely free
        # between windows. A window that leaked its descriptor would keep the
        # card reserved forever and no test above would notice.
        import fcntl
        proc = self.spawn(wait=30, hold=0.1, tag='a')
        out, err = proc.communicate(timeout=90)
        self.assertEqual(proc.returncode, 0, err)
        self.assertTrue(self.lease.is_file(), out)
        fd = os.open(self.lease, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def test_a_window_bounded_wait_expires_as_a_resource_refusal(self):
        # The other half of "waits instead of refusing": a wait that never opens
        # must still end, and end as BLOCKED_ENV (75), not as a hang or a FAIL.
        import fcntl
        self.lease.parent.mkdir(parents=True, exist_ok=True)
        held = os.open(self.lease, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            proc = self.spawn(wait=0.5, hold=0.1, tag='c')
            out, err = proc.communicate(timeout=90)
            self.assertEqual(proc.returncode, 75,
                             f'a held lease must exit 75 (BLOCKED_ENV): {out} {err}')
            self.assertIn('did not open', err)
        finally:
            os.close(held)


class SharedLockRootPermissionsCase(unittest.TestCase):
    """Shared device lock roots must expose 0o1777 dir and 0o666 lock/quarantine files."""

    def test_shared_device_lock_root_uses_1777_dir_and_0666_files(self):
        import importlib.util
        import stat
        from unittest.mock import patch
        from agent_gates import adapter_runtime, locking

        gate_path = ROOT / 'agent_gates/adapters/tilefoundry-npu/scripts/tilefoundry_host_gate.py'
        spec = importlib.util.spec_from_file_location('tf_gate_perm_test', gate_path)
        assert spec is not None and spec.loader is not None
        gate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gate
        spec.loader.exec_module(gate)

        with tempfile.TemporaryDirectory() as td:
            shared_root = Path(td) / 'shared-locks'
            lease_path = shared_root / 'npu-1.lock'
            old_umask = os.umask(0o022)
            try:
                with patch.dict(os.environ, {
                    'AGENT_GATES_DEVICE_LOCK_ROOT': str(shared_root),
                    'AGENT_GATES_DEVICE_LEASE_PATH': str(lease_path),
                    'AGENT_GATES_DEVICE_LEASE_RESOURCE': 'npu:1',
                    'AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS': '0',
                }, clear=False):
                    self.assertTrue(locking.is_shared_device_lock_root(shared_root))
                    self.assertFalse(locking.is_shared_device_lock_root(Path(td) / 'private'))

                    with locking.ResourceLock('npu:1', shared_root):
                        pass
                    self.assertEqual(stat.S_IMODE(shared_root.stat().st_mode), 0o1777)
                    self.assertEqual(
                        stat.S_IMODE(locking.lock_path('npu:1', shared_root).stat().st_mode),
                        0o666,
                    )

                    with patch.object(adapter_runtime, 'processes_on', return_value=([], 1)):
                        with adapter_runtime.device_window(shared_root, 1, exclusive=True):
                            pass
                    self.assertEqual(stat.S_IMODE(lease_path.stat().st_mode), 0o666)

                    with patch.object(gate, '_device_occupancy', return_value='free'):
                        with gate._open_device_window('1', wait_seconds=0, require_idle=True):
                            pass
                    self.assertEqual(stat.S_IMODE(lease_path.stat().st_mode), 0o666)

                    with gate._open_msprof_lock(locking.device_lock_root(), timeout_seconds=1.0) as msprof_lock:
                        self.assertEqual(msprof_lock.parent.resolve(), shared_root.resolve())
                    self.assertEqual(stat.S_IMODE((shared_root / '.msprof.lock').stat().st_mode), 0o666)

                    identity = gate._process_identity(os.getpid())
                    assert identity is not None
                    cleanup = {
                        'surviving': [os.getpid()],
                        'survivingProcesses': [identity],
                    }
                    record = gate._quarantine_cleanup(
                        cleanup,
                        lease_path=str(lease_path),
                        resource='npu:1',
                        device='1',
                        token='tok',
                        phase='test',
                        reason='perm-check',
                    )
                    self.assertTrue(record['written'])
                    q_path = Path(record['path'])
                    self.assertEqual(stat.S_IMODE(q_path.stat().st_mode), 0o666)
                    q_path.unlink()
            finally:
                os.umask(old_umask)


if __name__ == '__main__':
    unittest.main()
