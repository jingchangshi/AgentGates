"""Host-only regressions for the TLA Adapter.

These pin the properties that make the gate honest without a device: the
observations it publishes, the conditions it will not pass on, and the machine
file grammar it shares with the engine. Nothing here touches Docker or an NPU.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'agent_gates/adapters/tla-npu'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load('tla_gate', ADAPTER / 'scripts/tla_gate.py')

# A report that describes a run which really reached the device. Every negative
# case below is this dict with exactly one thing broken, so a test failure names
# the one property that stopped working.
GOOD = {
    'npuAvailable': True,
    'scriptExit': 0,
    'scriptStdout': 'CAST_BACK OK tmax 8 hid 256\nTRITON_CAST_BACK OK num_mismatch: 0\npass\n',
    'tensorMovesToDevice': 6,
    'npuPeakMemoryBytes': 50366464,
    'artifacts': {'object': 0, 'npubin': 2},
    'synchronizeOk': True,
    'synchronizeError': None,
    'npuDeviceCount': 1,
    'visibleDevices': '1',
    'deviceUuid': '0008d80a-d200-0000-0000-008a4ec00000',
    'deviceName': 'Ascend950PR_9579',
}


class ObservationCase(unittest.TestCase):
    def observed(self, **changes):
        report = dict(GOOD)
        report.update(changes)
        return GATE.observations(report, '1')

    def test_a_device_run_passes_every_declared_check(self):
        result = self.observed()
        self.assertTrue(GATE.passed(result), [k for k, v in result.items() if v is not True])
        manifest = json.loads((ADAPTER / 'manifest.json').read_text())
        declared = set(manifest['gates'][0]['observationChecks'])
        # exitCodeZero is supplied by the evaluator from the host process status,
        # so it is the one declared check the bridge does not publish itself.
        self.assertEqual(declared - set(result), {'exitCodeZero'})
        for name in declared - {'exitCodeZero'}:
            self.assertIs(result[name], True, name)

    def test_the_scripts_own_conditions_agree_with_the_manifest(self):
        # If this script gated on fewer conditions than the manifest declares, it
        # could exit 0 on a run the evaluator then records as FAIL. exitCodeZero is
        # the only declared check it cannot compute, because that one describes the
        # host process.
        manifest = json.loads((ADAPTER / 'manifest.json').read_text())
        declared = set(manifest['gates'][0]['observationChecks'])
        self.assertEqual(declared - set(GATE.REQUIRED), {'exitCodeZero'})
        self.assertEqual(set(GATE.REQUIRED) - declared, set())
        # And every one of them must actually be consulted.
        for name in GATE.REQUIRED:
            broken = self.observed()
            broken[name] = False
            self.assertFalse(GATE.passed(broken), name)

    def test_no_tensor_on_the_device_fails(self):
        self.assertFalse(GATE.passed(self.observed(tensorMovesToDevice=0)))

    def test_no_positive_allocation_fails(self):
        self.assertFalse(GATE.passed(self.observed(npuPeakMemoryBytes=0)))

    def test_no_compiled_kernel_fails(self):
        self.assertFalse(GATE.passed(self.observed(artifacts={'object': 0, 'npubin': 0})))

    def test_an_object_file_alone_is_not_required(self):
        # The Ascend backend emits .npubin and no .o. Requiring both would make
        # the check unsatisfiable on a correct run, which is worse than useless.
        result = self.observed(artifacts={'object': 0, 'npubin': 1})
        self.assertTrue(GATE.passed(result))

    def test_silent_stdout_fails_even_with_a_zero_exit(self):
        self.assertFalse(GATE.passed(self.observed(scriptStdout='')))

    def test_the_cast_back_marker_alone_fails(self):
        # The two markers are independent evidence: one route passing must not
        # stand in for both.
        self.assertFalse(GATE.passed(self.observed(scriptStdout='CAST_BACK OK tmax 8 hid 256\n')))

    def test_the_triton_marker_alone_fails(self):
        self.assertFalse(GATE.passed(self.observed(scriptStdout='TRITON_CAST_BACK OK num_mismatch: 0\n')))

    def test_a_nonzero_script_exit_fails(self):
        self.assertFalse(GATE.passed(self.observed(scriptExit=1)))

    def test_a_missing_script_exit_fails(self):
        self.assertFalse(GATE.passed(self.observed(scriptExit=None)))

    def test_a_failed_synchronize_fails(self):
        self.assertFalse(GATE.passed(self.observed(synchronizeOk=False)))

    def test_two_visible_devices_fails(self):
        self.assertFalse(GATE.passed(self.observed(npuDeviceCount=2)))

    def test_the_device_identity_echo_is_recorded_but_not_gated(self):
        # These values are echoes of configuration this Gate supplied, so they
        # cannot fail for a hardware reason. Recording them is useful; gating on
        # them would be a tautology dressed as proof, which is why they are
        # excluded from passed() and from the manifest.
        manifest = json.loads((ADAPTER / 'manifest.json').read_text())
        declared = set(manifest['gates'][0]['observationChecks'])
        self.assertNotIn('deviceMatchesAuthorized', declared)
        self.assertNotIn('visibleDeviceEcho', declared)
        self.assertNotIn('deviceModelUuid', declared)
        self.assertEqual(self.observed(visibleDevices='0')['visibleDeviceEcho'], '0')
        self.assertIsNone(self.observed(visibleDevices=None)['visibleDeviceEcho'])

    def test_a_boolean_in_place_of_a_counter_is_refused(self):
        # A report that claims success as a bare true where the contract wants a
        # count must not be read as a count.
        result = self.observed(tensorMovesToDevice=True, npuPeakMemoryBytes=True)
        self.assertEqual(result['tensorMovesToDevice'], 0)
        self.assertEqual(result['npuPeakMemoryBytes'], 0)
        self.assertFalse(GATE.passed(result))

    def test_a_negative_counter_is_refused(self):
        result = self.observed(tensorMovesToDevice=-1, npuPeakMemoryBytes=-1)
        self.assertEqual(result['tensorMovesToDevice'], 0)
        self.assertEqual(result['npuPeakMemoryBytes'], 0)


class ReportCase(unittest.TestCase):
    def test_a_report_without_a_marker_prefix_is_refused(self):
        with self.assertRaises(GATE.BridgeError):
            GATE.read_report('no report here\n')

    def test_a_duplicate_key_is_refused(self):
        with self.assertRaises(GATE.BridgeError):
            GATE.read_report('{"command": "tla", "command": "tla"}\n')

    def test_a_non_finite_number_is_refused(self):
        with self.assertRaises(GATE.BridgeError):
            GATE.read_report('{"command": "tla", "npuPeakMemoryBytes": NaN}\n')

    def test_the_wrong_command_is_refused(self):
        with self.assertRaises(GATE.BridgeError):
            GATE.read_report('{"command": "something-else"}\n')


class ContainerScriptCase(unittest.TestCase):
    """The container-side sources must be complete, compilable programs.

    They are carried as embedded text, so a truncated or mangled literal would
    only be discovered by a real run. These checks make that a host-only failure.
    """

    def test_the_runner_and_watcher_are_complete_scripts(self):
        match = re.search(r'WATCHER = (".*")', GATE.RUNNER)
        self.assertIsNotNone(match)
        watcher = eval(match.group(1))
        self.assertNotIn('truncated', watcher)
        compile(watcher, 'watcher', 'exec')
        self.assertIn('runpy.run_path', watcher)
        self.assertIn("'uuid'", watcher)
        # The watcher is what counts device use, so both ways the test reaches
        # the device have to be observed.
        self.assertIn('torch.Tensor.npu = observed_move', watcher)
        self.assertIn("device.split(':')[0] == 'npu'", watcher)

    def test_the_runner_compiles_as_the_container_receives_it(self):
        compile(GATE.RUNNER, 'runner', 'exec')
        # The wrapper is written into the container from the runner, so a
        # quoting fault in the handoff is a runtime-only failure without this.
        self.assertIn('handle.write(WATCHER)', GATE.RUNNER)

    def test_the_container_script_names_every_toolchain_step(self):
        script = GATE.container_script('/tests', '/src', 60, '1', Path('/cann/set_env.sh'),
                                       Path('/conda/etc/profile.d/conda.sh'), 'an-env',
                                       Path('/envs'))
        for fragment in ('. /cann/set_env.sh', 'conda activate an-env',
                         'export TRITON_CACHE_DIR=$(mktemp -d)', 'test_tla.py',
                         'CONDA_ENVS_PATH=/envs'):
            self.assertIn(fragment, script)
        # The embedded programs are Python; the surrounding Bash is not, so the
        # runner is compiled and the shell is checked for the quoting that
        # would break it.
        compile(GATE.RUNNER, 'runner', 'exec')


class SourceTreeSafetyCase(unittest.TestCase):
    """A run must not deposit artifacts into the bytes it is testing."""

    def test_the_instrumented_run_does_not_use_the_source_directory_as_cwd(self):
        # CANN writes a crash-dump tree (extra-info/) into the process working
        # directory when a kernel faults. Observed for real: a contended-card run
        # left a root-owned extra-info/ inside the source directory, which both
        # violated sourceWrite=false and made the source snapshot depend on
        # whether an earlier run had failed.
        match = re.search(r'WATCHER = (".*")', GATE.RUNNER)
        self.assertIsNotNone(match)
        watcher = eval(match.group(1))
        self.assertIn('runpy.run_path', watcher)
        runner = GATE.RUNNER
        self.assertIn('tempfile.mkdtemp', runner)
        self.assertIn('cwd=work', runner)
        self.assertNotIn('cwd=os.path.dirname(os.path.abspath(test))', runner)
        # The dump location is pinned explicitly rather than left to the runtime.
        self.assertIn('ASCEND_WORK_PATH=work', runner)

    def test_the_container_script_enters_the_test_directory_not_the_source_root(self):
        script = GATE.container_script('/tests', '/src', 60, '1', Path('/cann/set_env.sh'),
                                       Path('/conda/etc/profile.d/conda.sh'), 'an-env',
                                       Path('/envs'))
        # The runner executes the test from the test directory. The source root is
        # the checkout under test, so the shell only checks that it exists: a chdir
        # into it is what let a compiler cache land inside the tree being hashed.
        self.assertIn('cd /tests', script)
        self.assertIn('[ -d /src ]', script)
        self.assertNotIn('cd /src', script)
        self.assertIn('TLA_NPU', script)

    def test_a_configured_cache_directory_is_used_and_created(self):
        script = GATE.container_script('/tests', '/src', 60, '1', None, None, None, None,
                                       Path('/cache/tla'))
        self.assertIn('export TRITON_CACHE_DIR=/cache/tla', script)
        self.assertIn('mkdir -p "$TRITON_CACHE_DIR"', script)
        # A dedicated directory replaces the per-run temporary one rather than
        # being layered on top of it, so the count describes this run only.
        self.assertNotIn('mktemp -d)', script)

    def test_the_runner_records_which_tla_implementation_it_imported(self):
        # A version string cannot distinguish a wheel rebuilt at the same version,
        # so the implementation is hashed and reported.
        self.assertIn('tlaImplementationSha256', GATE.RUNNER)
        self.assertIn('import triton.language.extra.tla as _tla', GATE.RUNNER)
        self.assertIn('tritonVersion', GATE.RUNNER)

    def test_the_cache_may_not_live_inside_either_input_tree(self):
        # Enforced in the Gate module itself, so a misconfiguration is refused
        # before anything is dispatched rather than polluting a hashed tree.
        source = Path(GATE.__file__).read_text()
        self.assertIn('is inside the {label}', source)
        self.assertIn("('source root', host_root)", source)
        self.assertIn("('test directory', test_dir)", source)


class SourceAndTestInputsCase(unittest.TestCase):
    """The source root and the test directory are different inputs."""

    def test_the_test_directory_is_required(self):
        # A Gate that quietly fell back to the source root would run whatever
        # test happened to be there, which is exactly the conflation this
        # adapter was changed to remove.
        with patch.object(GATE, 'machine_value', return_value=None):
            with self.assertRaises(GATE.BridgeError):
                GATE.test_dir_from_env_file()

    def test_the_test_directory_must_be_absolute(self):
        with patch.object(GATE, 'machine_value', return_value='relative/dir'):
            with self.assertRaises(GATE.BridgeError):
                GATE.test_dir_from_env_file()

    def test_the_cache_directory_is_optional_but_must_be_absolute(self):
        with patch.object(GATE, 'machine_value', return_value=None):
            self.assertIsNone(GATE.cache_dir_from_env_file())
        with patch.object(GATE, 'machine_value', return_value='relative/cache'):
            with self.assertRaises(GATE.BridgeError):
                GATE.cache_dir_from_env_file()


class ProvisioningScriptCase(unittest.TestCase):
    """The provisioning script is the only place the dependency set is written."""

    SETUP = Path(__file__).resolve().parent.parent / 'agent_gates' / 'adapters' \
        / 'tla-npu' / 'scripts' / 'setup-python-env.sh'

    def test_a_malformed_machine_file_stops_provisioning(self):
        # The lookup distinguishes status 2 (malformed) from status 1 (key not
        # set). It used to conflate them: an awk `exit 2` inside the main block
        # still runs END, and END's own exit overrode the status, so every
        # malformed line was reported as "key missing" and silently defaulted.
        source = self.SETUP.read_text()
        self.assertIn('END { exit(bad ? 2 : (found ? 0 : 1)) }', source)
        self.assertIn('bad = 1', source)
        self.assertIn('is malformed', source)

    def test_the_script_is_built_from_named_steps(self):
        # Reading and installing are separate steps, so a typo cannot leave a
        # half-built environment and each phase can be reviewed on its own.
        source = self.SETUP.read_text()
        for step in ('_config_load()', '_wheels_select()', '_env_prepare()',
                     '_wheels_install()', '_verify_environment()'):
            self.assertIn(step, source)
        # main states the order and nothing else.
        main = source.split('main() {', 1)[1].split('}', 1)[0]
        self.assertEqual([line.strip() for line in main.strip().splitlines() if line.strip()],
                         ['_config_load', '_wheels_select', '_env_prepare',
                          '_wheels_install', '_verify_environment',
                          '_setup_error "environment $CFG_CONDA_ENV is ready at $ENV_DIR"',
                          'return 0'])

    def test_the_awk_comment_holds_no_apostrophe(self):
        # The awk program lives inside single quotes. An apostrophe in a comment
        # closes the shell string early and the script stops parsing.
        source = self.SETUP.read_text()
        opening = "    awk -v want=\"$key\" '"
        start = source.index(opening) + len(opening)
        end = source.index("    ' \"$file\"", start)
        self.assertNotIn("'", source[start:end])

    def test_the_setup_script_and_the_gate_read_the_same_machine_file(self):
        # Provisioning and running must agree on which file describes the
        # machine, or an environment is built from one and exercised from another.
        setup = self.SETUP.read_text()
        gate = Path(GATE.__file__).read_text()
        for name in ('TLA_NPU_ENV_FILE', 'env.sh'):
            self.assertIn(name, setup)
            self.assertIn(name, gate)


class MachineFileCase(unittest.TestCase):
    def test_the_script_and_the_engine_accept_the_same_keys(self):
        from agent_gates import machine_env
        self.assertEqual(tuple(sorted(GATE.TOOLCHAIN_KEYS)),
                         tuple(sorted(machine_env.TLA_NPU_KEYS)))

    def test_the_adapter_owns_its_own_prefix(self):
        from agent_gates import machine_env
        self.assertEqual(machine_env.ADAPTERS['tla-npu']['prefix'], 'TLA_NPU_')
        self.assertEqual(machine_env.ADAPTERS['tla-npu']['selector'], 'TLA_NPU_ENV_FILE')

    def test_exclusivity_is_the_variable_this_adapter_reads(self):
        # The engine forwards the operator policy into this exact name. A
        # hardcoded TileLang variable would look forwarded and be ignored.
        from agent_gates import engine
        self.assertEqual(engine.EXCLUSIVE_VARIABLE['tla-npu'],
                         'TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE')
        with patch.dict(os.environ, {'TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE': '1'}, clear=False):
            self.assertTrue(GATE.exclusive_requested())
        with patch.dict(os.environ, {'TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE': '0'}, clear=False):
            self.assertFalse(GATE.exclusive_requested())

    def test_the_device_variable_is_the_one_the_engine_leases(self):
        from agent_gates import engine
        self.assertEqual(engine.DEVICE_ADAPTERS['tla-npu'], 'TLA_NPU_DEVICE_ID')

    def test_half_a_conda_pair_is_refused(self):
        with self.assertRaises(GATE.BridgeError):
            GATE.require_conda_pair(Path('/bin/true'), None)
        with self.assertRaises(GATE.BridgeError):
            GATE.require_conda_pair(None, 'some-env')
        GATE.require_conda_pair(None, None)

    def test_an_empty_machine_file_is_refused_rather_than_guessed(self):
        with machine_file('') as values, patch.dict(os.environ, values, clear=True):
            with self.assertRaises(GATE.BridgeError):
                GATE.main(['--check-config'])


def machine_file(text=''):
    import contextlib
    import tempfile

    @contextlib.contextmanager
    def manager():
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'machine.env'
            path.write_text(text)
            yield {'TLA_NPU_ENV_FILE': str(path)}
    return manager()


class ManifestCase(unittest.TestCase):
    def test_every_pinned_file_matches_its_digest(self):
        manifest = json.loads((ADAPTER / 'manifest.json').read_text())
        self.assertEqual(manifest['schemaVersion'], 1)
        self.assertEqual(manifest['id'], 'tla-npu')
        self.assertIs(manifest['permissions']['hostExecution'], True)
        self.assertIs(manifest['permissions']['sourceWrite'], False)
        for name, digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ADAPTER / name).read_bytes()).hexdigest(), digest, name)

    def test_the_recipe_pins_a_fixed_argv_and_blocks_a_temporary_failure(self):
        manifest = json.loads((ADAPTER / 'manifest.json').read_text())
        recipe = manifest['recipes'][0]
        self.assertEqual(recipe['cwd'], 'source')
        self.assertEqual(recipe['argv'][:3], ['python3', '-B', '{pack}/scripts/tla_gate.py'])
        self.assertIn(75, recipe['blockedExitCodes'])
        # The outer timeout has to exceed the inner one plus the kill grace, or a
        # slow container would be killed by the harness before it could report.
        self.assertGreater(recipe['timeout'], 1200 + GATE.KILL_AFTER_SECONDS)

    def test_the_entrypoint_is_the_reviewed_bytes(self):
        # The container entrypoint is shared with the Ascend IR adapter. A local
        # edit would mean two different entrypoints ship under one review.
        ours = (ADAPTER / 'docker/entry.sh').read_bytes()
        sibling = (ROOT / 'agent_gates/adapters/ascendnpu-ir/docker/entry.sh').read_bytes()
        self.assertEqual(ours, sibling)


class HostExecutionCase(unittest.TestCase):
    def test_processes_on_handles_blank_process_name(self):
        sample = (
            "\tNPU ID                         : 0\n"
            "\tProcess id:2330713 Process name:                  Process memory(MB):796\n"
            "\tProcess id:12345   Process name:python3           Process memory(MB):512\n"
        )
        with patch.object(GATE, 'npu_smi', return_value=sample):
            rows = GATE.processes_on('0')
        self.assertEqual(rows, [('2330713', '<unknown>'), ('12345', 'python3')])

    def test_npu_smi_honors_agent_gates_smi_command(self):
        import types
        fake = types.SimpleNamespace(returncode=0, stdout='ok\n', stderr='')
        with patch.dict(os.environ, {'AGENT_GATES_SMI_COMMAND': '/custom/fake-smi'}), \
                patch.object(GATE.subprocess, 'run', return_value=fake) as run_mock:
            out = GATE.npu_smi('-m')
        self.assertEqual(out, 'ok\n')
        self.assertEqual(run_mock.call_args[0][0][:2], ['/custom/fake-smi', 'info'])


if __name__ == '__main__':
    unittest.main()