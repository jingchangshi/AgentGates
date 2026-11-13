"""Host-only regressions for the ForgeKernel Ascend Adapter and its device window.

These pin the two properties the shared-machine design depends on:

* a card is occupied only inside the window that actually uses it;
* two runs wanting the same card WAIT for each other instead of one of them
  being refused as if the environment were broken.

Nothing here starts Docker, touches an NPU or runs a compiler.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'agent_gates/adapters/forgekernel-ascend-npu'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load('fk_gate', ADAPTER / 'scripts/fk_gate.py')


class MachineFileCase(unittest.TestCase):
    """The Gate and the engine accept exactly the same machine keys."""

    def test_the_script_and_the_engine_accept_the_same_keys(self):
        from agent_gates import machine_env
        self.assertEqual(tuple(sorted(GATE.TOOLCHAIN_KEYS)),
                         tuple(sorted(machine_env.FK_ASCEND_KEYS)))

    def test_the_adapter_owns_its_own_prefix_and_selector(self):
        from agent_gates import machine_env
        spec = machine_env.ADAPTERS['forgekernel-ascend-npu']
        self.assertEqual(spec['prefix'], 'FK_ASCEND_')
        self.assertEqual(spec['selector'], 'FK_ASCEND_ENV_FILE')
        self.assertEqual(spec['directory'], 'forgekernel-ascend-npu')

    def test_an_unknown_key_is_refused_by_the_gate_and_the_engine(self):
        from agent_gates import machine_env
        text = 'FK_ASCEND_PYTHON=/bin/python3\nNOT_A_MACHINE_KEY=x\n'
        path = Path('/nonexistent/machine.env')
        with self.assertRaises(machine_env.MachineEnvError):
            machine_env.parse(text, path, machine_env.FK_ASCEND_KEYS)
        with self.assertRaises(GATE.BridgeError):
            GATE._parse_env_file(text, path)

    def test_a_duplicate_key_is_refused_by_both_sides(self):
        from agent_gates import machine_env
        text = 'FK_ASCEND_PYTHON=/bin/python3\nFK_ASCEND_PYTHON=/bin/python3\n'
        path = Path('/nonexistent/machine.env')
        with self.assertRaises(machine_env.MachineEnvError):
            machine_env.parse(text, path, machine_env.FK_ASCEND_KEYS)
        with self.assertRaises(GATE.BridgeError):
            GATE._parse_env_file(text, path)


class CheckConfigCase(unittest.TestCase):
    """Preflight decides about configuration only; it builds and probes nothing."""

    def machine(self, values):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        # A TileLang checkout and build that exist, so the identity check has
        # something real to read: preflight must verify, not merely record.
        source = root / 'tilelang'
        (source / 'tilelang').mkdir(parents=True)
        (source / 'tilelang' / '__init__.py').write_text('__version__ = "0"\n')
        build = source / 'build'
        (build / 'lib').mkdir(parents=True)
        (build / 'lib' / 'libtilelang.so').write_bytes(b'native')
        path = root / 'machine.env'
        filled = {'FK_ASCEND_PYTHON': sys.executable,
                  'FK_ASCEND_TILELANG_SOURCE': str(source),
                  'FK_ASCEND_TILELANG_BUILD': str(build),
                  'FK_ASCEND_VENV': str(root / 'venv/fk'),
                  # The build installs from a verified copy, which has to live
                  # outside both the checkout and the evidence directory.
                  'FK_ASCEND_WORK_ROOT': str(root / 'work'),
                  **values}
        path.write_text(''.join(f'{key}={value}\n' for key, value in filled.items()))
        return path, root

    def test_a_complete_configuration_returns_zero_and_spawns_nothing(self):
        path, _ = self.machine({})
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path)}, clear=True), \
                patch.object(GATE.subprocess, 'run') as run:
            self.assertEqual(GATE.main(['--check-config']), 0)
            run.assert_not_called()

    def test_a_missing_key_fails(self):
        for missing in GATE.REQUIRED:
            path, root = self.machine({})
            values = dict(line.split('=', 1) for line in path.read_text().splitlines())
            values.pop(missing)
            path.write_text(''.join(f'{key}={value}\n' for key, value in values.items()))
            with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path)}, clear=True):
                with self.assertRaises(GATE.BridgeError) as error:
                    GATE.main(['--check-config'])
                self.assertIn(missing, str(error.exception))

    def test_an_unknown_key_fails(self):
        path, _ = self.machine({'FK_ASCEND_TYPO': 'x'})
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path)}, clear=True):
            with self.assertRaises(GATE.BridgeError) as error:
                GATE.main(['--check-config'])
            self.assertIn('FK_ASCEND_TYPO', str(error.exception))

    def test_a_duplicate_key_fails(self):
        path, _ = self.machine({})
        path.write_text(path.read_text() + 'FK_ASCEND_VENV=/tmp/other\n')
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path)}, clear=True):
            with self.assertRaises(GATE.BridgeError) as error:
                GATE.main(['--check-config'])
            self.assertIn('second time', str(error.exception))

    def test_a_missing_native_library_is_a_diagnosis_not_a_rebuild(self):
        path, root = self.machine({})
        build = Path(root / 'tilelang/build')
        (build / 'lib' / 'libtilelang.so').unlink()
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path)}, clear=True), \
                patch.object(GATE.subprocess, 'run') as run:
            with self.assertRaises(GATE.BridgeError) as error:
                GATE.main(['--check-config'])
            self.assertIn('never starts one', str(error.exception))
            run.assert_not_called()

    def test_the_check_starts_no_build(self):
        # The property that makes this a preflight: it must not begin the work it
        # is deciding about. Every process launch is refused here.
        path, _ = self.machine({})
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path)}, clear=True), \
                patch.object(GATE.subprocess, 'run', side_effect=AssertionError('spawned')):
            self.assertEqual(GATE.main(['--check-config']), 0)

    def test_an_exported_machine_value_is_refused(self):
        # The file is the only source: an export that agrees is still refused,
        # because a shell that believes it configured a run while the receipt
        # describes the file is two configurations for one build.
        path, _ = self.machine({})
        for exported in ('FK_ASCEND_WORK_ROOT=/tmp', 'FK_ASCEND_NOPE=1'):
            name, _, value = exported.partition('=')
            with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path), name: value}, clear=True):
                with self.assertRaises(GATE.BridgeError) as error:
                    GATE.main(['--check-config'])
                self.assertIn(name, str(error.exception))
                # An unknown key is a typo; a known key is a misplaced
                # configuration. Both are refused, and both name the file.
                if name in GATE.TOOLCHAIN_KEYS:
                    self.assertIn(str(path), str(error.exception))

    def test_the_per_run_device_authorization_is_not_a_machine_key(self):
        # It authorizes one run, so it cannot live in a file shared by every run
        # on the machine - and therefore must not be refused as a stray export.
        path, _ = self.machine({})
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path), 'FK_ASCEND_DEVICE_ID': '3'},
                        clear=True):
            self.assertEqual(GATE.main(['--check-config']), 0)


class WindowCase(unittest.TestCase):
    """The device window: what it holds, for how long, and what it refuses."""

    def lease(self, *, wait='5', resource='npu:3'):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        path = root / 'locks' / 'npu.lock'
        values = {'AGENT_GATES_DEVICE_LEASE_PATH': str(path),
                  'AGENT_GATES_DEVICE_LEASE_RESOURCE': resource,
                  'AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS': wait}
        return path, values

    def test_the_window_reports_what_it_held_and_released(self):
        path, values = self.lease()
        with patch.dict(os.environ, values, clear=True):
            with GATE.device_window() as window:
                self.assertEqual(window['resource'], 'npu:3')
                self.assertEqual(window['path'], str(path))
                self.assertIsNone(window['releasedAt'])
                self.assertIn(window['occupancy'], {'free', 'unknown'})
                self.assertIsInstance(window['waited'], float)
            self.assertIsNotNone(window['releasedAt'])
            self.assertLessEqual(window['startedAt'], window['releasedAt'])

    def test_the_lease_is_free_outside_the_window(self):
        # The whole point of a window: the card is NOT held by the process that
        # is merely configured to use it. Another process can take the lease
        # while this one is between windows, which is what "only occupy while
        # running a kernel" has to mean to be more than a claim.
        import fcntl
        path, values = self.lease()
        with patch.dict(os.environ, values, clear=True):
            with GATE.device_window():
                pass
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)

    def test_the_window_is_held_while_it_is_open(self):
        # The complement of the test above, so "free outside" cannot be satisfied
        # by a window that never held anything.
        import fcntl
        path, values = self.lease()
        with patch.dict(os.environ, values, clear=True):
            with GATE.device_window():
                with self.assertRaises(BlockingIOError):
                    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(fd)

    def test_the_window_publishes_how_long_it_held_and_how_long_it_worked(self):
        # A window is only a window if its duration is observable. Without these
        # a receipt could report a 20-second "device time" that measured an
        # import, and nobody could tell from the evidence.
        path, values = self.lease()
        with patch.dict(os.environ, values, clear=True):
            with GATE.device_window() as window:
                pass
        self.assertIn('heldSeconds', window)
        self.assertIn('probeSeconds', window)
        self.assertGreaterEqual(window['heldSeconds'], 0.0)
        self.assertLess(window['heldSeconds'], 5.0)

    def test_a_window_waits_for_a_holder_instead_of_failing(self):
        import fcntl
        path, values = self.lease(wait='5')
        path.parent.mkdir(parents=True, exist_ok=True)
        held = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        released_at = {}

        def release():
            time.sleep(0.3)
            released_at['t'] = time.monotonic()
            os.close(held)

        import threading
        thread = threading.Thread(target=release)
        thread.start()
        try:
            with patch.dict(os.environ, values, clear=True):
                with GATE.device_window() as window:
                    opened = time.monotonic()
                    self.assertIsNotNone(window['occupancy'])
                # It opened only AFTER the other holder let go, and it says so.
                self.assertGreaterEqual(opened, released_at['t'])
                self.assertGreaterEqual(window['waited'], 0.2)
        finally:
            thread.join()

    def test_a_window_that_never_opens_is_a_resource_refusal(self):
        import fcntl
        path, values = self.lease(wait='0.3')
        path.parent.mkdir(parents=True, exist_ok=True)
        held = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with patch.dict(os.environ, values, clear=True):
                with self.assertRaises(GATE.DeviceWindowBusy) as error:
                    with GATE.device_window():
                        self.fail('a held lease must not open')
                # 75 is the code the runner maps to BLOCKED_ENV: nothing ran.
                self.assertEqual(error.exception.EXIT, 75)
        finally:
            os.close(held)

    def test_a_lease_handle_must_be_given_not_invented(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GATE.BridgeError) as error:
                with GATE.device_window():
                    self.fail('no lease was given')
            self.assertIn('AGENT_GATES_DEVICE_LEASE_PATH', str(error.exception))

    def test_a_nonpositive_wait_is_refused(self):
        path, values = self.lease(wait='0')
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaises(GATE.BridgeError):
                with GATE.device_window():
                    self.fail('a zero wait is not a window')

    def test_occupancy_reports_a_count_and_never_a_process_identity(self):
        # Another tenant on the card is normal, so occupancy is a count. It must
        # not carry the other process's identity into a receipt.
        output = ('\tNPU ID                         : 4\n'
                  '\tProcess id:1976176 Process name:python3  Process memory(MB):303\n'
                  '\tProcess id:1977873 Process name:alice  Process memory(MB):373\n')
        done = subprocess.CompletedProcess(args=[], returncode=0, stdout=output, stderr='')
        with patch.dict(os.environ, {}, clear=True), patch.object(GATE.subprocess, 'run', return_value=done) as run:
            self.assertEqual(GATE.occupancy('npu:4'), 'shared(2)')
            # The probe asks about the LEASED card, not the machine: a neighbour
            # being busy must not be reported as this card being busy.
            self.assertEqual(run.call_args.args[0][-2:], ['-i', '4'])
        self.assertNotIn('alice', 'shared(2)')
        empty = subprocess.CompletedProcess(args=[], returncode=0,
                                            stdout='\tNPU ID                         : 4\n'
                                                   '\tNo process in device.\n', stderr='')
        with patch.dict(os.environ, {}, clear=True), patch.object(GATE.subprocess, 'run', return_value=empty):
            self.assertEqual(GATE.occupancy('npu:4'), 'free')
        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout='', stderr='boom')
        with patch.dict(os.environ, {}, clear=True), patch.object(GATE.subprocess, 'run', return_value=failed):
            self.assertEqual(GATE.occupancy('npu:4'), 'unknown')

    def test_an_answer_about_another_card_is_unknown_not_free(self):
        # A report that does not name the queried card cannot be turned into a
        # count: "free" here would be a claim about a device nobody asked about.
        other = subprocess.CompletedProcess(args=[], returncode=0,
                                            stdout='\tNPU ID                         : 5\n'
                                                   '\tNo process in device.\n', stderr='')
        with patch.dict(os.environ, {}, clear=True), patch.object(GATE.subprocess, 'run', return_value=other):
            self.assertEqual(GATE.occupancy('npu:4'), 'unknown')

    def test_a_nonnumeric_resource_yields_unknown_rather_than_a_guess(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(GATE.subprocess, 'run') as run:
            self.assertEqual(GATE.occupancy('npu:all'), 'unknown')
            run.assert_not_called()


class SmokeGateCase(unittest.TestCase):
    """The smoke Gate actually uses a window, and never fakes a device result."""

    def machine(self, values=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        path = root / 'machine.env'
        filled = {'FK_ASCEND_PYTHON': sys.executable, **(values or {})}
        path.write_text(''.join(f'{key}={value}\n' for key, value in filled.items()))
        return path, root

    def test_an_authorized_device_is_required_and_never_defaulted(self):
        # Choosing a card is the operator's act. A Gate that picks one is how a
        # test quietly runs on somebody else's device.
        path, _ = self.machine()
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path)}, clear=True):
            with self.assertRaises(GATE.BridgeError) as error:
                GATE.main(['--check-config-smoke'])
            self.assertIn('FK_ASCEND_DEVICE_ID', str(error.exception))

    def test_a_non_numeric_device_id_is_refused(self):
        path, _ = self.machine()
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path),
                                     'FK_ASCEND_DEVICE_ID': 'npu0'}, clear=True):
            with self.assertRaises(GATE.BridgeError):
                GATE.main(['--check-config-smoke'])

    def test_the_preflight_needs_no_build_artifacts(self):
        # The point of the smoke Gate is that it runs before any ForgeKernel
        # installation exists: it must not demand TileLang, a venv or a build.
        path, _ = self.machine()
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path),
                                     'FK_ASCEND_DEVICE_ID': '0'}, clear=True):
            self.assertEqual(GATE.main(['--check-config-smoke']), 0)

    def test_an_authorized_device_is_made_visible_to_the_child(self):
        # The child must reach the card the operator named. Logical device 0 in
        # the child is only that card because the Gate made it so.
        path, _ = self.machine()
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({'npuAvailable': True, 'tensorOnDevice': True,
                               'numericalOk': True, 'synchronizeOk': True,
                               'deviceMemoryAllocated': 4096, 'windowOpened': True,
                               'windowReleased': True, 'occupancyRecorded': True}),
            stderr='')
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path),
                                     'FK_ASCEND_DEVICE_ID': '1'}, clear=True), \
                patch.object(GATE.subprocess, 'run', return_value=completed) as run:
            GATE.main(['--smoke'])
        child = run.call_args.kwargs['env']
        self.assertEqual(child['ASCEND_RT_VISIBLE_DEVICES'], '1')
        self.assertEqual(child['FK_ASCEND_DEVICE_ID'], '1')

    def test_a_missing_backend_is_a_resource_refusal_not_a_failure(self):
        # "There is no device here" is not "the code is wrong". Reporting FAIL
        # would claim a test failed that never ran.
        path, _ = self.machine()
        completed = subprocess.CompletedProcess(args=[], returncode=3, stdout='',
                                                stderr='cannot import torch_npu')
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path),
                                     'FK_ASCEND_DEVICE_ID': '0'}, clear=True), \
                patch.object(GATE.subprocess, 'run', return_value=completed):
            with self.assertRaises(GATE.DeviceWindowBusy) as error:
                GATE.main(['--smoke'])
            self.assertEqual(error.exception.EXIT, 75)
            self.assertIn('cannot import torch_npu', str(error.exception))

    def test_a_running_backend_with_wrong_numbers_is_a_real_failure(self):
        # The mirror of the case above: if the device ran and the arithmetic was
        # wrong, that IS a failed test and must exit 1.
        path, _ = self.machine()
        completed = subprocess.CompletedProcess(
            args=[], returncode=1,
            stdout=json.dumps({'npuAvailable': True, 'tensorOnDevice': True,
                               'numericalOk': False, 'synchronizeOk': True,
                               'deviceMemoryAllocated': 4096, 'windowOpened': True,
                               'windowReleased': True, 'occupancyRecorded': True}),
            stderr='')
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path),
                                     'FK_ASCEND_DEVICE_ID': '0'}, clear=True), \
                patch.object(GATE.subprocess, 'run', return_value=completed):
            self.assertEqual(GATE.main(['--smoke']), 1)

    def test_an_unreadable_probe_report_is_refused(self):
        # A probe that ran but published no report is not evidence of success.
        path, _ = self.machine()
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout='not json', stderr='')
        with patch.dict(os.environ, {'FK_ASCEND_ENV_FILE': str(path),
                                     'FK_ASCEND_DEVICE_ID': '0'}, clear=True), \
                patch.object(GATE.subprocess, 'run', return_value=completed):
            with self.assertRaises(GATE.BridgeError):
                GATE.main(['--smoke'])

    def test_the_window_checks_are_exactly_what_the_manifest_declares(self):
        # The child computes these booleans; the manifest gates on them. If the
        # two sets drift, the receipt and the verdict describe different runs.
        manifest = json.loads((ADAPTER / 'manifest.json').read_text())
        gate = {item['id']: item for item in manifest['gates']}[
            'forgekernel-ascend-npu-smoke']
        declared = set(gate['observationChecks']) - {'exitCodeZero'}
        self.assertEqual(declared, set(GATE.SMOKE_CHECKS))
        for name in GATE.SMOKE_CHECKS:
            self.assertIs(gate['observationChecks'][name], True, name)

    def test_the_smoke_recipe_declares_a_window_and_can_be_blocked(self):
        manifest = json.loads((ADAPTER / 'manifest.json').read_text())
        recipe = {item['id']: item for item in manifest['recipes']}[
            'forgekernel-ascend-npu.smoke']
        self.assertNotIn('device', recipe, 'a windowed recipe must not also opt out')
        self.assertGreater(float(recipe['deviceWindow']['waitSeconds']), 0)
        self.assertIn(75, recipe['blockedExitCodes'])

    def test_the_probe_is_pinned_so_the_receipt_names_the_bytes_that_ran(self):
        manifest = json.loads((ADAPTER / 'manifest.json').read_text())
        self.assertIn('scripts/fk_device_probe.py', manifest['files'])
        # The Gate republishes the digest it will execute, so a checkout edited
        # after pinning cannot pass silently.
        self.assertEqual(GATE.smoke_probe_identity()['sha256'],
                         GATE.digest(GATE.PROBE_SCRIPT))


class SourceCopyCase(unittest.TestCase):
    """The build must not write into the checkout it installs from.

    This is the regression for a real defect: `pip install <checkout>` made
    setuptools write `build/` into the ForgeKernel tree, which ForgeKernel does
    not ignore, so the source identity moved during the Gate and its own PASS
    became SOURCE_IDENTITY_CHANGED. The fix installs from a verified copy; these
    tests fail if anyone goes back to installing the checkout directly.
    """

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source"
        (self.source / "pkg").mkdir(parents=True)
        (self.source / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")
        (self.source / "pkg" / "__init__.py").write_text("VALUE = 1\n")
        (self.source / "README.md").write_text("readme\n")
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        # A committed file, an untracked-but-not-ignored file, and an IGNORED
        # file: the listing must take the first two and skip the third.
        (self.source / ".gitignore").write_text("ignored.bin\n")
        (self.source / "untracked.txt").write_text("untracked\n")
        (self.source / "ignored.bin").write_bytes(b"ignored")
        subprocess.run(["git", "-C", str(self.source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.source), "commit", "-qm", "base"],
                       check=True)
        # Recreate the untracked file and the ignored one AFTER the commit.
        (self.source / "untracked.txt").write_text("untracked\n")
        (self.source / "ignored.bin").write_bytes(b"ignored")

    def listing(self):
        return GATE.git_listing(self.source)

    def test_the_copy_contains_exactly_what_the_receipt_would_hash(self):
        destination = self.root / "copy"
        result = GATE.copy_source(self.source, destination)
        copied = sorted(str(path.relative_to(destination))
                        for path in destination.rglob("*") if path.is_file())
        self.assertEqual(copied, self.listing())
        # The ignored file is not part of the source identity, so it must not be
        # installed either: a copy that took it would install bytes no receipt
        # describes.
        self.assertNotIn("ignored.bin", copied)
        # An untracked-but-not-ignored file IS part of the identity, so it must
        # be copied - `git archive` would have dropped it.
        self.assertIn("untracked.txt", copied)
        self.assertEqual(result["fileCount"], len(copied))

    def test_the_copy_is_verified_against_the_source_before_it_is_used(self):
        destination = self.root / "copy"
        result = GATE.copy_source(self.source, destination)
        self.assertEqual(result["sourceHash"], result["copyHash"])
        self.assertIs(result["hashMatches"], True)
        # Independently recomputed: the same bytes hashed the same way agree.
        self.assertEqual(GATE.tree_hash(destination, result["sourceFiles"]),
                         result["sourceHash"])

    def test_a_copy_that_lost_a_file_is_refused_rather_than_installed(self):
        # The check that makes the hash meaningful: if the copy is incomplete the
        # digest differs, and the Gate must refuse instead of installing.
        destination = self.root / "copy"
        GATE.copy_source(self.source, destination)
        (destination / "pkg" / "__init__.py").write_text("VALUE = 2\n")
        self.assertNotEqual(
            GATE.tree_hash(destination, GATE.git_listing(self.source)),
            GATE.tree_hash(self.source, GATE.git_listing(self.source)))

    def test_the_copy_carries_no_git_directory(self):
        # If .git came along, the copy would be a repository and a later listing
        # of it would answer a different question than the one asked.
        destination = self.root / "copy"
        GATE.copy_source(self.source, destination)
        self.assertFalse((destination / ".git").exists())

    def test_the_build_does_not_install_the_checkout_itself(self):
        # The actual regression, asserted on behaviour rather than on source
        # text: run build() with the installer stubbed out and inspect the argv
        # it would have executed. Installing the checkout is what wrote `build/`
        # into the ForgeKernel tree, so the install target must be a copy that
        # lives outside it.
        work = self.root / "work"
        venv = self.root / "venv"
        machine = self.root / "machine.env"
        machine.write_text(
            f"FK_ASCEND_PYTHON={sys.executable}\n"
            f"FK_ASCEND_TILELANG_SOURCE={self.source}\n"
            f"FK_ASCEND_TILELANG_BUILD={self.source}\n"
            f"FK_ASCEND_VENV={venv}\n"
            f"FK_ASCEND_WORK_ROOT={work}\n")
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/python").write_text("")
        (venv / "forgekernel").mkdir(parents=True)
        (venv / "forgekernel/__init__.py").write_text("")
        (venv / "cli.py").write_text("")

        commands = []

        real_run = subprocess.run

        def fake_run(argv, **kwargs):
            argv = [str(item) for item in argv]
            # git must reach the real implementation: the listing is the very
            # thing under test, so stubbing it would test the stub.
            if "git" in argv[0]:
                return real_run(argv, **kwargs)
            commands.append(argv)
            stdout = ""
            if "-c" in argv:
                stdout = json.dumps({"file": str(venv / "forgekernel/__init__.py"),
                                     "version": "0", "cli": str(venv / "cli.py")})
            return subprocess.CompletedProcess(args=argv, returncode=0,
                                               stdout=stdout, stderr="")

        with patch.dict(os.environ, {"FK_ASCEND_ENV_FILE": str(machine)}, clear=True), \
                patch.object(GATE, "create_venv"), \
                patch.object(GATE, "tilelang_identity", return_value={"source": "x"}), \
                patch.object(GATE, "digest", return_value="0" * 64), \
                patch.object(GATE, "venv_python", return_value=venv / "bin/python"), \
                patch.object(GATE.subprocess, "run", side_effect=fake_run):
            GATE.build(self.source)

        installs = [c for c in commands if "install" in c]
        self.assertTrue(installs, "no pip install was executed")
        target = Path(installs[0][-1])
        # The installed path must be the copy, not the checkout.
        self.assertNotEqual(target, self.source)
        self.assertTrue(target.is_relative_to(work),
                        f"installed {target}, which is not a copy under {work}")


class ManifestCase(unittest.TestCase):
    def manifest(self):
        return json.loads((ADAPTER / 'manifest.json').read_text())

    def test_every_pinned_file_matches_its_digest(self):
        import hashlib
        manifest = self.manifest()
        self.assertEqual(manifest['schemaVersion'], 1)
        self.assertEqual(manifest['id'], 'forgekernel-ascend-npu')
        self.assertIs(manifest['permissions']['hostExecution'], True)
        self.assertIs(manifest['permissions']['sourceWrite'], False)
        for name, digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ADAPTER / name).read_bytes()).hexdigest(), digest, name)

    def test_the_build_gate_is_host_only_and_publishes_exactly_what_it_checks(self):
        manifest = self.manifest()
        gate = {item['id']: item for item in manifest['gates']}['forgekernel-ascend-npu-build']
        recipe = {item['id']: item for item in manifest['recipes']}['forgekernel-ascend-npu.build']
        # A build never drives a card, so it declares that and takes no lease.
        self.assertEqual(recipe.get('device'), 'none')
        self.assertNotIn('deviceWindow', recipe)
        self.assertEqual(set(gate['observationChecks']) - {'exitCodeZero'}, set(GATE.CHECKS))
        self.assertEqual(set(GATE.CHECKS) - set(gate['observationChecks']), set())
        for name in GATE.CHECKS:
            self.assertIs(gate['observationChecks'][name], True, name)

    def test_the_recipe_blocks_a_temporary_failure_and_bounds_its_install(self):
        recipe = {item['id']: item for item in self.manifest()['recipes']}['forgekernel-ascend-npu.build']
        self.assertIn(75, recipe['blockedExitCodes'])
        self.assertGreater(float(recipe['timeout']), GATE.INSTALL_TIMEOUT_SECONDS)

    def test_the_adapter_is_registered_and_its_gates_are_reachable(self):
        from agent_gates.registry import Registry
        manifest = Registry().get('forgekernel-ascend-npu')
        self.assertEqual(manifest.id, 'forgekernel-ascend-npu')
        self.assertEqual(list(manifest.gates),
                         ['forgekernel-ascend-npu-smoke', 'forgekernel-ascend-npu-build'])

    def test_the_build_recipe_takes_no_device_and_needs_no_authorization(self):
        from agent_gates.registry import Registry
        manifest = Registry().get('forgekernel-ascend-npu')
        # `recipes` is the raw declaration; `resolve_recipe` is the typed form the
        # engine acts on, and that is what must carry the opt-out.
        self.assertEqual(manifest.recipes['forgekernel-ascend-npu.build'].get('device'), 'none')
        recipe = manifest.resolve_recipe('forgekernel-ascend-npu.build', manifest.root)
        self.assertIs(recipe.device_window, None)
        self.assertIs(recipe.uses_device, False)

    def test_device_window_refuses_shared_card_when_exclusive_requested(self):
        with tempfile.TemporaryDirectory() as td:
            lease = Path(td) / 'npu.lock'
            env = {
                'AGENT_GATES_DEVICE_LEASE_PATH': str(lease),
                'AGENT_GATES_DEVICE_LEASE_RESOURCE': 'npu:1',
                'AGENT_GATES_DEVICE_LEASE_WAIT_SECONDS': '1',
                'FK_ASCEND_REQUIRE_EXCLUSIVE_DEVICE': '1',
            }
            with patch.dict(os.environ, env, clear=False), \
                    patch.object(GATE, 'occupancy', return_value='shared(2)'):
                with self.assertRaises(GATE.DeviceWindowBusy):
                    with GATE.device_window():
                        pass


if __name__ == '__main__':
    unittest.main()