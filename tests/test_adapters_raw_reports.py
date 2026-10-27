"""Malformed container facts must not become successful gate observations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ADAPTERS = Path(__file__).parents[1] / 'agent_gates' / 'adapters'


def load(adapter, script):
    spec = importlib.util.spec_from_file_location(adapter.replace('-', '_'), ADAPTERS / adapter / 'scripts' / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load('tilelang-hivm', 'docker_gate.py')
npu = load('tilelang-hivm-npu', 'npu_gate.py')


class RawReportsCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.module = self.root / 'tilelang' / '__init__.py'
        self.native = self.root / 'custom-output' / 'libtilelang.so'
        for path in (self.module, self.native):
            path.parent.mkdir(parents=True)
            path.write_bytes(b'measured bytes')
        self.facts = {
            'pythonModuleRel': 'tilelang/__init__.py',
            'pythonModuleSha256': hashlib.sha256(self.module.read_bytes()).hexdigest(),
            'nativeLibs': {'custom-output/libtilelang.so': hashlib.sha256(self.native.read_bytes()).hexdigest()},
            'ascendBackendCompiled': True, 'devMode': True,
        }
        self.stats = dict(collected=3, passed=3, failed=0, errors=0, skipped=0)

    def npu_report(self):
        return dict(tests=self.stats.copy(), pytestExit=0, junitPresent=True,
                    phyBefore=1, phyAfter=1, tensorMovesToDevice=6, numericalAssertions=3,
                    npuPeakMemoryBytes=1024, npuAvailable=True, synchronizeOk=True,
                    npuDeviceCount=1, driverDeviceCount=1)

    def test_npu_scalar_substitutions_fail(self):
        fields = ('numericalAssertions', 'tensorMovesToDevice', 'npuPeakMemoryBytes',
                  'phyBefore', 'phyAfter', 'npuDeviceCount', 'driverDeviceCount')
        for field in fields:
            for value in (True, False, '1', 1.0, -1, None):
                with self.subTest(field=field, value=value):
                    report = self.npu_report()
                    report[field] = value
                    self.assertFalse(npu.passed(npu.observations(report, '1')))
        for field in ('npuAvailable', 'synchronizeOk', 'junitPresent'):
            for value in ('false', 1, None, False):
                report = self.npu_report()
                report[field] = value
                self.assertFalse(npu.passed(npu.observations(report, '1')))
        for value in (False, '0', 0.0, None):
            report = self.npu_report()
            report['pytestExit'] = value
            self.assertFalse(npu.passed(npu.observations(report, '1')))

    def test_malformed_statistics_fail_for_both_bridges(self):
        for key in self.stats:
            for value in (True, '0', -1, None):
                with self.subTest(key=key, value=value):
                    tests = dict(self.stats, **{key: value})
                    report = self.npu_report()
                    report['tests'] = tests
                    self.assertFalse(npu.passed(npu.observations(report, '1')))
                    report['facts'] = self.facts
                    observed = base.regression_observations(report, self.root)
                    self.assertFalse(observed['testsCollectedPositive'])
                    self.assertFalse(observed['allPassed'])
        report = self.npu_report()
        report['tests']['passed'] = 4
        self.assertFalse(npu.passed(npu.observations(report, '1')))

    def test_truthy_backend_and_junit_values_stay_false(self):
        report = dict(facts=dict(self.facts, devMode='true', ascendBackendCompiled=1),
                      tests=self.stats, junitPresent='true', pytestExit=False)
        observed = base.regression_observations(report, self.root)
        for field in ('devMode', 'ascendBackendCompiled', 'junitPresent', 'pytestExitZero'):
            self.assertIs(observed[field], False)

    def test_duplicate_and_nonfinite_raw_json_is_rejected(self):
        for command, reader in (
                ('regression', lambda text: base.read_report(text, command='regression')),
                ('npu', npu.read_report)):
            for suffix in ('"tests":{"passed":1,"passed":0}', '"value":NaN',
                           '"value":Infinity', '"value":-Infinity'):
                with self.subTest(command=command, suffix=suffix):
                    with self.assertRaises((base.BridgeError, npu.BridgeError)):
                        reader('{"command":' + json.dumps(command) + ',' + suffix + '}')

    def test_measured_artifacts_include_custom_native_location(self):
        observed = base.identity_observations(self.facts, self.root)
        expected = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in (self.module, self.native)]
        self.assertEqual(observed['artifacts'], expected)
        self.assertTrue(observed['loadedNativeSha256MatchesHost'])

    def test_symlink_outside_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / 'lib.so'
            target.write_bytes(b'outside')
            self.native.unlink()
            self.native.symlink_to(target)
            with self.assertRaises(base.BridgeError):
                base.identity_observations(self.facts, self.root)

    def test_base_regression_rejects_raw_failure_with_zero_process_exit(self):
        report = dict(command='regression', facts=self.facts, tests=self.stats,
                      junitPresent=True, pytestExit=False)
        done = type('Completed', (), dict(stdout=json.dumps(report), stderr='', returncode=0))()
        with (patch.object(base, 'container_identity', return_value=('container', 'user', '/home/user', 'machine')),
             patch.object(base, 'toolchain_from_env_file', return_value=(None, None, None)),
             patch.object(base, 'conda_envs_path_from_env_file', return_value=None),
             patch.object(base, 'hivm_from_env_file', return_value=(None, None)),
             patch.object(base, 'container_mounts', return_value=[]),
             patch.object(base, 'map_host_to_container', return_value=self.root),
             patch.object(base, 'verify_same_tree'),
             patch.object(base, 'regression_observations', return_value=dict.fromkeys(
                 ('importedModuleSha256MatchesHost', 'loadedNativeSha256MatchesHost',
                  'nativeLibsPresent', 'ascendBackendCompiled', 'devMode', 'junitPresent',
                  'testsCollectedPositive', 'notAllSkipped', 'allPassed'), True) | {'pytestExitZero': False}),
             patch.object(base.subprocess, 'run', return_value=done)):
            self.assertEqual(base.main(['--timeout', '900', 'regression']), 1)


if __name__ == '__main__':
    unittest.main()
