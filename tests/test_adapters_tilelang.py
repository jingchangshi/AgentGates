"""Offline acceptance checks for the `tilelang-hivm` and `tilelang-hivm-npu` Adapters.

Every Docker interaction is faked, so these tests need neither a container, a
compiler build nor an NPU. They cover the deterministic rules the two Adapters
own:

* manifest validity and published SHA-256 pinning;
* Host-to-container bind-mount resolution, and refusal when the container sees a
  different tree;
* refusal of an unsafe build directory and of path traversal;
* the nesting between the Harness Recipe timeout and the container-side ceiling;
* the real command in each Recipe, which no environment variable may replace;
* that a non-zero exit, a timeout and a missing report can never be a PASS;
* that zero collected tests, an all-SKIPPED suite and an uncompiled Ascend
  backend can never be a PASS;
* that a Host/container identity mismatch in the imported module or in a native
  library can never be a PASS;
* that a missing or malformed NPU device id can never be a PASS, and that a
  device synchronization failure can never be a PASS;
* that a task which does not select the NPU Adapter runs no device test;
* that the three Adapters compose into Gates with the right dependencies;
* that an uninstalled or unauthorized Pack is refused.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / 'agent_gates/adapters'
BASE = ADAPTERS / 'tilelang-hivm'
NPU = ADAPTERS / 'tilelang-hivm-npu'

sys.path.insert(0, str(ROOT))
from agent_gates.manifest import Manifest, ManifestError as ContractError


def load_manifest(path):
    manifest = Manifest(path)
    return dict(manifest.raw, manifestHash=manifest.manifest_hash)


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_script('tilelang_base_bridge', BASE / 'scripts/docker_gate.py')
npu = load_script('tilelang_npu_bridge', NPU / 'scripts/npu_gate.py')

# Values that are true of the machine this Adapter was authored on. None of them
# may appear in the shipped source: the Adapter is configured per server.
MACHINE_SPECIFIC = (
    's00653124', '/home/s00653124', 's00653124_tilelang',
    '/data/pri/cann-9.2.0', '/data/miniconda3', 'AscendNPU-IR-1',
    '/data/pri', 'hivm-poc-control',
)


class ManifestCase(unittest.TestCase):
    def test_base_manifest_is_valid_and_hash_pinned(self):
        manifest = load_manifest(BASE / 'manifest.json')
        self.assertEqual(manifest['id'], 'tilelang-hivm')
        self.assertEqual(manifest['permissions'], {'sourceWrite': False, 'hostExecution': True})
        self.assertEqual(manifest['runtimeCompatibility'], {'engineeringCore': '>=2.0.0'})
        self.assertEqual([r['id'] for r in manifest['recipes']],
                         ['tilelang-hivm.build', 'tilelang-hivm.regression'])
        self.assertEqual({g['id'] for g in manifest['gates']},
                         {'tilelang-hivm-build', 'tilelang-hivm-regression'})
        self.assertEqual(list(manifest['files']),
                         ['scripts/docker_gate.py', 'scripts/full-build.sh',
                          'scripts/init-toolchain.sh', 'scripts/setup-python-env.sh'])

    def test_npu_manifest_is_valid_and_hash_pinned(self):
        manifest = load_manifest(NPU / 'manifest.json')
        self.assertEqual(manifest['id'], 'tilelang-hivm-npu')
        self.assertEqual(manifest['permissions'], {'sourceWrite': False, 'hostExecution': True})
        self.assertEqual(manifest['runtimeCompatibility'], {'engineeringCore': '>=2.0.0'})
        self.assertEqual([r['id'] for r in manifest['recipes']], ['tilelang-hivm-npu.sanity'])
        self.assertEqual({g['id'] for g in manifest['gates']}, {'tilelang-hivm-npu-sanity'})
        self.assertEqual(list(manifest['files']), ['scripts/npu_gate.py'])

    def test_pinned_digest_matches_the_shipped_file(self):
        for adapter, relatives in (
                (BASE, ('scripts/docker_gate.py', 'scripts/full-build.sh',
                        'scripts/init-toolchain.sh', 'scripts/setup-python-env.sh')),
                (NPU, ('scripts/npu_gate.py',))):
            manifest = load_manifest(adapter / 'manifest.json')
            for relative in relatives:
                actual = hashlib.sha256((adapter / relative).read_bytes()).hexdigest()
                self.assertEqual(manifest['files'][relative], actual, relative)

    def test_every_script_is_pinned_and_no_other_file_is(self):
        # The manifest is the Gate contract, so a script that is not pinned could
        # be changed without re-trusting the Adapter.
        for adapter in (BASE, NPU):
            manifest = load_manifest(adapter / 'manifest.json')
            shipped = {str(path.relative_to(adapter))
                       for path in (adapter / 'scripts').rglob('*')
                       if path.is_file() and '__pycache__' not in path.parts}
            self.assertEqual(set(manifest['files']), shipped, adapter.name)

    def test_no_machine_specific_value_is_hard_coded(self):
        for adapter in (BASE, NPU):
            for path in sorted((adapter / 'scripts').glob('*.py')):
                text = path.read_text()
                for forbidden in MACHINE_SPECIFIC:
                    self.assertNotIn(forbidden, text, f'{path}: {forbidden}')
                self.assertNotIn('ASCEND_RT_VISIBLE_DEVICES=', text.replace(
                    'os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device', ''))

    def test_no_committed_adapter_file_carries_a_machine_value(self):
        # The `*.py` scan above is what the repository test enforces; the shell
        # scripts, the Docker files and the templates are read inside the
        # container, so they must stay as machine-neutral as the Gates. The
        # variable *name* ASCEND_RT_VISIBLE_DEVICES is legitimate in a deployment
        # script; a literal device id for it is not.
        # The scan must cover COMMITTED files only. A machine file such as
        # `docker/config.env` is git-ignored by design and legitimately holds
        # this host's paths, so walking the directory tree would fail on every
        # correctly configured machine. `git ls-files` names exactly what a
        # checkout receives, and an untracked new file is caught by the
        # `git status` check that the repository tests run.
        device_literal = re.compile(r'ASCEND_RT_VISIBLE_DEVICES=[0-9]')
        tracked = [str(path.relative_to(ROOT)) for adapter in (BASE, NPU)
                   for path in adapter.rglob('*') if path.is_file()
                   and path.name not in ('env.sh', 'config.env')]
        seen = 0
        for name in tracked:
            if not name:
                continue
            path = ROOT / name
            if '__pycache__' in path.parts or not path.is_file():
                continue
            seen += 1
            text = path.read_text()
            for forbidden in MACHINE_SPECIFIC:
                self.assertNotIn(forbidden, text, f'{path}: {forbidden}')
            self.assertIsNone(device_literal.search(text), path)
        self.assertGreater(seen, 10, 'the tracked-file scan found almost nothing')

    def test_drifted_script_digest_is_refused(self):
        for adapter, relative in (
                (BASE, 'scripts/docker_gate.py'), (BASE, 'scripts/full-build.sh'),
                (BASE, 'scripts/init-toolchain.sh'), (NPU, 'scripts/npu_gate.py')):
            with tempfile.TemporaryDirectory() as tmp:
                copy = Path(tmp) / adapter.name
                shutil.copytree(adapter, copy, ignore=shutil.ignore_patterns('__pycache__'))
                (copy / relative).write_text('# drifted\n')
                with self.assertRaises(ContractError):
                    load_manifest(copy / 'manifest.json')

    def test_recipes_use_a_fixed_bridge_and_interpolate_nothing(self):
        for adapter, script in ((BASE, 'scripts/docker_gate.py'), (NPU, 'scripts/npu_gate.py')):
            manifest = load_manifest(adapter / 'manifest.json')
            for recipe in manifest['recipes']:
                self.assertEqual(recipe['cwd'], 'source', recipe['id'])
                self.assertEqual(recipe['argv'][:3], ['python3', '-B', f'{{pack}}/{script}'],
                                 recipe['id'])
                self.assertEqual(recipe['argv'][3], '--timeout', recipe['id'])
                for item in recipe['argv']:
                    for token in ('{goal', '$', ';', '&&', '|'):
                        self.assertNotIn(token, item, f'{recipe["id"]}: {item}')

    def test_every_recipe_checks_its_configuration_before_it_is_dispatched(self):
        # A Gate reads its container identity from the machine file, so a machine
        # that is not configured fails deep inside Docker. The Recipe declares the
        # check that answers "is this machine configured?" without entering the
        # container, and Preflight runs it before any Gate process starts.
        for adapter, script in ((BASE, 'scripts/docker_gate.py'), (NPU, 'scripts/npu_gate.py')):
            manifest = load_manifest(adapter / 'manifest.json')
            for recipe in manifest['recipes']:
                self.assertEqual(recipe.get('configCheck'),
                                 ['python3', '-B', f'{{pack}}/{script}', '--check-config'],
                                 recipe['id'])
        # The flag the manifests name is a real mode of both scripts, and it is
        # the only argv it accepts: anything else is still an ordinary Gate call
        # and must not be read as a check.
        for module, flag in ((base, 'build'), (npu, '--timeout')):
            with self.assertRaises(Exception):
                with contextlib.redirect_stderr(io.StringIO()):
                    module.main(['--check-config', flag])

    def test_gate_templates_declare_only_supported_fields(self):
        for adapter in (BASE, NPU):
            manifest = load_manifest(adapter / 'manifest.json')
            for gate in manifest['gates']:
                self.assertTrue(set(gate) <= {'id', 'recipeId', 'gateType', 'observationChecks',
                                              'coverageClaim', 'dependsOn', 'allowNotApplicable'})
                self.assertTrue(gate['observationChecks'], gate['id'])
                self.assertTrue(all(value is True for value in gate['observationChecks'].values()),
                                gate['id'])


class TimeoutCase(unittest.TestCase):
    def test_inner_ceiling_plus_grace_fits_inside_the_recipe_timeout(self):
        self.assertEqual(base.KILL_AFTER_SECONDS, 15)
        self.assertEqual(npu.KILL_AFTER_SECONDS, 15)
        for adapter in (BASE, NPU):
            manifest = load_manifest(adapter / 'manifest.json')
            for recipe in manifest['recipes']:
                inner = int(recipe['argv'][4])
                self.assertLess(inner + 15, recipe['timeout'], recipe['id'])

    def test_container_script_pins_the_short_kill_after(self):
        script = base.container_script('/src', 'build', '$(nproc)', 840, None, None, None)
        self.assertIn('--kill-after=15', script)
        self.assertIn('timeout --signal=TERM --kill-after=15 840 ', script)
        self.assertNotIn('--kill-after=60', script)
        npu_script = npu.container_script('/src', 900, '0', None, None, None)
        self.assertIn('timeout --signal=TERM --kill-after=15 900 ', npu_script)

    def test_explicit_timeout_wins_over_the_environment(self):
        os.environ['TILELANG_HIVM_GATE_TIMEOUT'] = '5'
        self.addCleanup(os.environ.pop, 'TILELANG_HIVM_GATE_TIMEOUT', None)
        self.assertEqual(base.parse_argv(['--timeout', '840', 'build']), ('build', 840))
        self.assertEqual(npu.parse_argv(['--timeout', '900']), 900)
        # A bare invocation is the only path that reads the environment.
        self.assertEqual(base.parse_argv(['build'])[1], 5)
        self.assertEqual(npu.parse_argv([]), 5)

    def test_manifest_recipes_never_rely_on_the_environment_default(self):
        for adapter, extra in ((BASE, 1), (NPU, 0)):
            manifest = load_manifest(adapter / 'manifest.json')
            for recipe in manifest['recipes']:
                with patch.dict(os.environ, {'TILELANG_HIVM_GATE_TIMEOUT': '1'}):
                    if adapter is BASE:
                        _, timeout = base.parse_argv(recipe['argv'][3:])
                    else:
                        timeout = npu.parse_argv(recipe['argv'][3:])
                self.assertEqual(timeout, int(recipe['argv'][4]), recipe['id'])
                self.assertNotEqual(timeout, 1, recipe['id'])

    def test_a_non_positive_timeout_is_refused(self):
        for value in ('0', '-1', 'abc', ''):
            with self.assertRaises(base.BridgeError):
                base.parse_argv(['--timeout', value, 'build'])
            with self.assertRaises(npu.BridgeError):
                npu.parse_argv(['--timeout', value])


class MountCase(unittest.TestCase):
    HOST = Path('/home/u/ws/tilelang')

    def both(self):
        return (base, npu)

    def test_parent_bind_produces_a_suffix_mapping(self):
        mounts = [{'Type': 'bind', 'Source': '/home', 'Destination': '/home'}]
        with patch.object(base.os.path, 'realpath', side_effect=lambda p: str(p)):
            for module in self.both():
                self.assertEqual(str(module.map_host_to_container(self.HOST, mounts)),
                                 str(self.HOST))

    def test_deepest_bind_wins(self):
        mounts = [
            {'Type': 'bind', 'Source': '/home', 'Destination': '/mnt/home'},
            {'Type': 'bind', 'Source': '/home/u/ws', 'Destination': '/src'},
        ]
        with patch.object(base.os.path, 'realpath', side_effect=lambda p: str(p)):
            for module in self.both():
                self.assertEqual(str(module.map_host_to_container(self.HOST, mounts)),
                                 '/src/tilelang')

    def test_unmounted_host_root_is_refused(self):
        mounts = [{'Type': 'bind', 'Source': '/home/other', 'Destination': '/home/other'}]
        with patch.object(base.os.path, 'realpath', side_effect=lambda p: str(p)):
            for module in self.both():
                with self.assertRaises(module.BridgeError) as error:
                    module.map_host_to_container(self.HOST, mounts)
                self.assertIn('not exposed by any bind mount', str(error.exception))

    def test_ambiguous_equal_depth_mounts_are_refused(self):
        mounts = [
            {'Type': 'bind', 'Source': str(self.HOST), 'Destination': '/a'},
            {'Type': 'bind', 'Source': str(self.HOST), 'Destination': '/b'},
        ]
        with patch.object(base.os.path, 'realpath', side_effect=lambda p: str(p)):
            for module in self.both():
                with self.assertRaises(module.BridgeError) as error:
                    module.map_host_to_container(self.HOST, mounts)
                self.assertIn('ambiguous', str(error.exception))

    def test_non_bind_mounts_are_ignored(self):
        mounts = [{'Type': 'volume', 'Source': '/var/lib/docker/volumes/x/_data',
                   'Destination': '/home'}]
        with patch.object(base.os.path, 'realpath', side_effect=lambda p: str(p)):
            for module in self.both():
                with self.assertRaises(module.BridgeError):
                    module.map_host_to_container(self.HOST, mounts)


class EnvironmentCase(unittest.TestCase):
    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))

    def test_build_subdir_traversal_is_refused(self):
        for value in ('..', '../build', 'a/b', '/build', '.', '', 'build/..', 'build/../..'):
            os.environ['TILELANG_HIVM_BUILD_SUBDIR'] = value
            with self.assertRaises(base.BridgeError, msg=repr(value)):
                base.build_subdir()

    def test_build_subdir_default_and_plain_names(self):
        os.environ.pop('TILELANG_HIVM_BUILD_SUBDIR', None)
        self.assertEqual(base.build_subdir(), 'build')
        os.environ['TILELANG_HIVM_BUILD_SUBDIR'] = 'build-release'
        self.assertEqual(base.build_subdir(), 'build-release')

    def test_ninja_jobs_defaults_to_container_nproc(self):
        os.environ.pop('TILELANG_HIVM_NINJA_JOBS', None)
        self.assertEqual(base.ninja_jobs(), '$(nproc)')
        os.environ['TILELANG_HIVM_NINJA_JOBS'] = '0'
        with self.assertRaises(base.BridgeError):
            base.ninja_jobs()
        os.environ['TILELANG_HIVM_NINJA_JOBS'] = '64'
        self.assertEqual(base.ninja_jobs(), '64')

    def test_required_container_variables_fail_closed(self):
        for name in ('TILELANG_HIVM_CONTAINER_NAME', 'TILELANG_HIVM_CONTAINER_USER',
                     'TILELANG_HIVM_CONTAINER_HOME'):
            for module in (base, npu):
                with self.assertRaises(module.BridgeError):
                    module.env_value(name, required=True)

    def test_unset_is_not_the_same_as_empty(self):
        for module in (base, npu):
            self.assertIsNone(module.env_value('TILELANG_HIVM_NOT_SET_ANYWHERE'))
            os.environ['TILELANG_HIVM_NOT_SET_ANYWHERE'] = '   '
            self.assertIsNone(module.env_value('TILELANG_HIVM_NOT_SET_ANYWHERE'))

    def test_relative_tool_paths_are_refused(self):
        for module in (base, npu):
            os.environ['TILELANG_HIVM_CANN_SETUP'] = 'set_env.sh'
            with self.assertRaises(module.BridgeError):
                module.env_path('TILELANG_HIVM_CANN_SETUP')

    def test_container_identity_comes_from_the_machine_file_not_the_shell(self):
        # The failure this prevents: one run where two agents address two
        # containers, because a stale shell export named a stopped container
        # while the machine file named the live one. Neither record showed it.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'env.sh'
            path.write_text('TILELANG_HIVM_CONTAINER_NAME=live\n'
                            'TILELANG_HIVM_CONTAINER_USER=builder\n'
                            'TILELANG_HIVM_CONTAINER_HOME=/home/builder\n')
            os.environ['TILELANG_HIVM_ENV_FILE'] = str(path)
            self.addCleanup(os.environ.pop, 'TILELANG_HIVM_ENV_FILE', None)
            for name, value in (('TILELANG_HIVM_CONTAINER_NAME', 'stopped'),
                                ('TILELANG_HIVM_CONTAINER_USER', 'someone'),
                                ('TILELANG_HIVM_CONTAINER_HOME', '/home/someone')):
                os.environ[name] = value
                self.addCleanup(os.environ.pop, name, None)
            self.assertEqual(base.container_identity(),
                             ('live', 'builder', '/home/builder', path))
            self.assertEqual(npu.container_identity(), ('live', '/home/builder', path))

    def test_the_device_gate_does_not_require_a_container_user(self):
        # The device Gate runs as root, so a machine that only ever runs it need
        # not name a container user -- but one file serves both Adapters, so the
        # key is still accepted when present (asserted by the test above).
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'env.sh'
            path.write_text('TILELANG_HIVM_CONTAINER_NAME=live\n'
                            'TILELANG_HIVM_CONTAINER_HOME=/home/builder\n')
            os.environ['TILELANG_HIVM_ENV_FILE'] = str(path)
            self.addCleanup(os.environ.pop, 'TILELANG_HIVM_ENV_FILE', None)
            self.assertEqual(npu.container_identity(), ('live', '/home/builder', path))
            with self.assertRaisesRegex(base.BridgeError, 'CONTAINER_USER'):
                base.container_identity()

    def test_a_missing_identity_names_the_file_and_the_keys(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'env.sh'
            path.write_text('TILELANG_HIVM_CONTAINER_NAME=live\n')
            os.environ['TILELANG_HIVM_ENV_FILE'] = str(path)
            self.addCleanup(os.environ.pop, 'TILELANG_HIVM_ENV_FILE', None)
            for module, missing in ((base, 'CONTAINER_USER, TILELANG_HIVM_CONTAINER_HOME'),
                                    (npu, 'TILELANG_HIVM_CONTAINER_HOME')):
                with self.assertRaises(module.BridgeError) as caught:
                    module.container_identity()
                message = str(caught.exception)
                self.assertIn(str(path), message)
                self.assertIn(missing, message)
                self.assertIn('read from that file only', message)

    def test_the_toolchain_file_is_the_single_source_and_is_only_data(self):
        # Both TileLang gates share one file, because both use the same prefix.
        # Reading it must never execute it, and a stale export must not mask it.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'env.sh'
            path.write_text('# toolchain\nTILELANG_HIVM_CANN_SETUP="/cann/set_env.sh"\n'
                            'TILELANG_HIVM_CONDA_SH=/conda/c.sh\nTILELANG_HIVM_CONDA_ENV=mk\n')
            os.environ['TILELANG_HIVM_ENV_FILE'] = str(path)
            self.addCleanup(os.environ.pop, 'TILELANG_HIVM_ENV_FILE', None)
            os.environ['TILELANG_HIVM_CANN_SETUP'] = '/stale/set_env.sh'
            self.addCleanup(os.environ.pop, 'TILELANG_HIVM_CANN_SETUP', None)
            for module in (base, npu):
                self.assertEqual(module.toolchain_from_env_file(),
                                 (Path('/cann/set_env.sh'), Path('/conda/c.sh'), 'mk'),
                                 module.PROG)
                self.assertFalse(hasattr(module, 'HOME_ENV_FILE'),
                                 f'{module.PROG} must not keep a second file location')

    def test_the_toolchain_file_never_executes_and_rejects_junk(self):
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / 'executed'
            path = Path(td) / 'env.sh'
            os.environ['TILELANG_HIVM_ENV_FILE'] = str(path)
            self.addCleanup(os.environ.pop, 'TILELANG_HIVM_ENV_FILE', None)
            for module in (base, npu):
                path.write_text(f'TILELANG_HIVM_CONDA_ENV=$(touch {marker}; echo x)\n')
                module.toolchain_from_env_file()
                self.assertFalse(marker.exists(), module.PROG)
                for junk in ('rm -rf /\n', 'TILELANG_HIVM_TYPO=1\n', 'TILELANG_HIVM_CONDA_ENV=a\n'
                             'TILELANG_HIVM_CONDA_ENV=b\n', 'TILELANG_HIVM_CANN_SETUP=rel.sh\n'):
                    path.write_text(junk)
                    with self.assertRaises(module.BridgeError, msg=f'{module.PROG} {junk!r}'):
                        module.toolchain_from_env_file()
                path.write_text('# a machine without CANN or Conda\n')
                self.assertEqual(module.toolchain_from_env_file(), (None, None, None))

    def test_a_missing_default_toolchain_file_is_not_an_error(self):
        os.environ.pop('TILELANG_HIVM_ENV_FILE', None)
        for module in (base, npu):
            with patch.object(module, 'default_env_file', return_value=Path('/nonexistent/env.sh')):
                self.assertEqual(module.toolchain_from_env_file(), (None, None, None))
            # No checkout, so there is no default file at all. That is not an
            # error either: CANN and Conda are optional.
            with patch.object(module, 'default_env_file', return_value=None):
                self.assertEqual(module.toolchain_from_env_file(), (None, None, None))
                self.assertEqual(module.conda_envs_path_from_env_file(), None)
                self.assertEqual(module.provisioning_from_env_file(), {})

    def test_the_toolchain_file_also_carries_the_provisioning_values(self):
        # `setup-python-env.sh` must not need environment variables, because a
        # shell export leaves no record and cannot be reviewed. It reads the same
        # `env.sh` the Gates read, so the file the user edits is the whole
        # configuration. The Gate ignores the provisioning keys and uses only
        # CANN, CONDA_SH and CONDA_ENV, but it must not reject the file for
        # carrying them.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'env.sh'
            path.write_text(
                'TILELANG_HIVM_CANN_SETUP=/cann/set_env.sh\n'
                'TILELANG_HIVM_CONDA_SH=/conda/etc/profile.d/conda.sh\n'
                'TILELANG_HIVM_CONDA_ENV=mk\n'
                'TILELANG_HIVM_CONDA_ENVS_PATH=/home/u/.conda/envs\n'
                'TILELANG_HIVM_WHEEL_DIR=/wheels\n'
                'TILELANG_HIVM_PIP_INDEX=https://mirror/simple/\n'
                'TILELANG_HIVM_PYTHON_VERSION=3.11\n'
                'TILELANG_HIVM_SOURCE_ROOT=/src/tilelang\n')
            os.environ['TILELANG_HIVM_ENV_FILE'] = str(path)
            self.addCleanup(os.environ.pop, 'TILELANG_HIVM_ENV_FILE', None)
            for module in (base, npu):
                # The Gate's own three values are unchanged by the extra keys.
                self.assertEqual(module.toolchain_from_env_file(),
                                 (Path('/cann/set_env.sh'),
                                  Path('/conda/etc/profile.d/conda.sh'), 'mk'),
                                 module.PROG)

    def test_a_conda_environment_outside_the_install_prefix_is_supported(self):
        # A conda installation only looks for environments under its own prefix
        # unless `CONDA_ENVS_PATH` says otherwise, and a user's environments
        # normally live in `~/.conda/envs`. Without this value the Gate fails with
        # `EnvironmentNameNotFound: Could not find conda environment: <name>`.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'env.sh'
            path.write_text('TILELANG_HIVM_CONDA_SH=/conda/etc/profile.d/conda.sh\n'
                            'TILELANG_HIVM_CONDA_ENV=mk\n'
                            'TILELANG_HIVM_CONDA_ENVS_PATH=/home/u/.conda/envs\n')
            os.environ['TILELANG_HIVM_ENV_FILE'] = str(path)
            self.addCleanup(os.environ.pop, 'TILELANG_HIVM_ENV_FILE', None)
            for module in (base, npu):
                values = module.provisioning_from_env_file()
                self.assertEqual(values['TILELANG_HIVM_CONDA_ENVS_PATH'],
                                 '/home/u/.conda/envs', module.PROG)
                # The Gate's own three values are not provisioning values; the
                # mapping holds only the keys the provisioning script reads.
                self.assertNotIn('TILELANG_HIVM_CONDA_ENV', values, module.PROG)
                self.assertIsNone(values.get('TILELANG_HIVM_WHEEL_DIR'), module.PROG)

    def test_provisioning_values_refuse_an_unknown_key_and_a_relative_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'env.sh'
            os.environ['TILELANG_HIVM_ENV_FILE'] = str(path)
            self.addCleanup(os.environ.pop, 'TILELANG_HIVM_ENV_FILE', None)
            for module in (base, npu):
                path.write_text('TILELANG_HIVM_WHEEL_DIR=relative/wheels\n')
                with self.assertRaises(module.BridgeError, msg=module.PROG):
                    module.provisioning_from_env_file()
                path.write_text('TILELANG_HIVM_NOT_A_KEY=1\n')
                with self.assertRaises(module.BridgeError, msg=module.PROG):
                    module.provisioning_from_env_file()

    def test_an_option_like_container_name_is_refused(self):
        for module in (base, npu):
            self.assertIsNone(module.SAFE_NAME.fullmatch('--privileged'))
            self.assertIsNone(module.SAFE_NAME.fullmatch('u; rm -rf /'))
            self.assertIsNotNone(module.SAFE_NAME.fullmatch('s00653124_tilelang'))

    def test_missing_or_malformed_device_id_is_refused(self):
        os.environ.pop('TILELANG_HIVM_DEVICE_ID', None)
        with self.assertRaises(npu.BridgeError):
            npu.device_id()
        for value in ('abc', '-1', '1.0', '0x1', '', '  ', '999'):
            os.environ['TILELANG_HIVM_DEVICE_ID'] = value
            with self.assertRaises(npu.BridgeError, msg=repr(value)):
                npu.device_id()

    def test_a_plain_device_number_is_accepted(self):
        for value in ('0', '1', '63'):
            os.environ['TILELANG_HIVM_DEVICE_ID'] = value
            self.assertEqual(npu.device_id(), value)


class TileLangEnvFileLocationCase(unittest.TestCase):
    """Where the machine file is found, without TILELANG_HIVM_ENV_FILE set.

    The file is git-ignored and sits beside the `tilelang-hivm` template. Both
    Adapters read that one file, so `tilelang-hivm-npu` has no template of its
    own and must resolve its neighbour's.
    """

    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        os.environ.pop('TILELANG_HIVM_ENV_FILE', None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkout = Path(self.temp.name) / 'PiEngineeringHarness'
        self.adapter = self.checkout / 'adapters' / 'tilelang-hivm'
        self.adapter.mkdir(parents=True)
        (self.adapter / 'env.example.sh').write_text('TILELANG_HIVM_CANN_SETUP=\n')

    def test_both_adapters_resolve_the_file_next_to_its_template(self):
        for module, name in ((base, 'tilelang-hivm'), (npu, 'tilelang-hivm-npu')):
            pack = self.checkout / 'pack-store' / 'installed' / name
            (pack / 'scripts').mkdir(parents=True)
            with patch.object(module, 'ADAPTER_DIR', pack):
                self.assertEqual(module.default_env_file(), self.adapter / 'env.sh', name)

    def test_outside_any_checkout_there_is_no_default_file(self):
        # A guessed home-directory path used to stand in for the checkout file.
        # It is gone: a machine that keeps its file elsewhere names it in
        # TILELANG_HIVM_ENV_FILE, so no Gate reads a file nobody reviewed.
        loose = Path(self.temp.name) / 'somewhere' / 'tilelang-hivm'
        (loose / 'scripts').mkdir(parents=True)
        for module in (base, npu):
            with patch.object(module, 'ADAPTER_DIR', loose):
                self.assertIsNone(module.default_env_file(), module.PROG)

    def test_the_found_file_is_what_both_gates_read(self):
        (self.adapter / 'env.sh').write_text('TILELANG_HIVM_CANN_SETUP=/cann/set_env.sh\n')
        for module in (base, npu):
            with patch.object(module, 'ADAPTER_DIR', self.adapter):
                self.assertEqual(module.toolchain_from_env_file(),
                                 (Path('/cann/set_env.sh'), None, None), module.PROG)


class DeploymentCase(unittest.TestCase):
    """Container creation, toolchain configuration and the Gate are three layers.

    `docker/config.env` creates the container and uses unprefixed keys. `env.sh`
    configures CANN and Conda. `manifest.json` is the Gate contract. None of the
    three may start reading another, because then editing one file would silently
    change the meaning of a Gate.
    """

    SCRIPT = BASE / 'docker/create-container.sh'
    TEMPLATE = BASE / 'docker/config.example.env'

    def test_create_container_never_reads_env_sh(self):
        for path in (self.SCRIPT,):
            text = path.read_text()
            self.assertNotIn('env.sh', text, path)
            self.assertNotIn('TILELANG_HIVM_', text, path)

    def test_docker_gate_never_reads_config_env(self):
        # `config.env` describes container *creation* and uses unprefixed keys.
        # The Gates must never parse it: the identity reaches them through
        # `env.sh`, which is where a machine states the container it created.
        # Naming the file in prose is allowed and necessary, because that is how
        # the reader is told which values to copy; *reading* it is what would
        # make two files decide one thing.
        for path in (BASE / 'scripts/docker_gate.py', NPU / 'scripts/npu_gate.py'):
            text = path.read_text()
            self.assertNotIn('CONTAINER_UID', text, path)
            self.assertNotIn('CONTAINER_GID', text, path)
            self.assertNotIn('BIND_MOUNTS', text, path)
            # No code path may open, parse or resolve it; only prose may name it.
            compact = text.replace(' ', '')
            for reader in ('open(config.env', 'read_text(config.env',
                            'Path(config.env', "read_text()if"):
                self.assertNotIn(reader, compact, path)

    def test_the_config_keys_are_unprefixed_and_the_gate_keys_are_prefixed(self):
        script = self.SCRIPT.read_text()
        for key in ('IMAGE', 'CONTAINER_NAME', 'CONTAINER_USER', 'CONTAINER_UID',
                    'CONTAINER_GID', 'CONTAINER_HOME', 'BIND_MOUNTS'):
            self.assertRegex(script, rf'(?m)^{key}="\$\{{{key}:?-', key)
        gate = (BASE / 'scripts/docker_gate.py').read_text()
        for key in ('TILELANG_HIVM_CONTAINER_NAME', 'TILELANG_HIVM_CONTAINER_USER',
                    'TILELANG_HIVM_CONTAINER_HOME', 'TILELANG_HIVM_CONTAINER_SOURCE'):
            self.assertIn(key, gate)

    def test_entry_sh_is_byte_identical_to_the_ascendnpu_ir_one(self):
        # The copy is intentional: the file is project-neutral, and this test
        # makes silent divergence impossible. Change both copies or neither.
        ours = hashlib.sha256((BASE / 'docker/entry.sh').read_bytes()).hexdigest()
        theirs = hashlib.sha256(
            (ADAPTERS / 'ascendnpu-ir/docker/entry.sh').read_bytes()).hexdigest()
        self.assertEqual(ours, theirs)

    def test_create_container_matches_the_ascendnpu_ir_copy_beside_the_image_line(self):
        # The safety behaviour (dry run, mount validation, HOME readability,
        # double identity verification) is the value, so the two copies must not
        # drift. Only comments and the image default may differ.
        def masked(path):
            return [line for line in path.read_text().splitlines()
                    if not line.startswith('#') and 'IMAGE="${IMAGE:-' not in line]

        self.assertEqual(masked(self.SCRIPT),
                         masked(ADAPTERS / 'ascendnpu-ir/docker/create-container.sh'))

    def test_the_example_template_covers_every_script_default(self):
        required = set(re.findall(r'^([A-Z_]+)="\$\{[A-Z_]+:?-', self.SCRIPT.read_text(),
                                  re.MULTILINE))
        defined = {line.split('=', 1)[0]
                   for line in self.TEMPLATE.read_text().splitlines()
                   if line and not line.startswith('#') and '=' in line}
        self.assertIn('IMAGE', required)
        self.assertTrue(required <= defined, sorted(required - defined))

    def test_the_example_template_holds_only_empty_values(self):
        for number, raw in enumerate(self.TEMPLATE.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            key, _, value = line.partition('=')
            self.assertEqual(value.strip(), '',
                             f'{self.TEMPLATE}: line {number} sets {key} to a value')

    def test_the_template_carries_no_machine_path(self):
        template = self.TEMPLATE.read_text()
        for forbidden in MACHINE_SPECIFIC:
            self.assertNotIn(forbidden, template)


class InitToolchainCase(unittest.TestCase):
    """Run scripts/init-toolchain.sh the way a human does: sourced, in real Bash.

    `InitScriptInteractionCase` below covers the Gate path, which inlines the
    text. These cases cover the other caller: the values come from `source
    env.sh`, the state the script builds must stay in the caller's shell, and a
    failure must return a status instead of ending that shell.
    """

    SCRIPT = BASE / 'scripts/init-toolchain.sh'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        # A stand-in interpreter: the TileLang tool check is `python3`, and this
        # host's own interpreter has no pytest.
        self.python = self.tools / 'python3'
        self.python.write_text('#!/bin/bash\nexit 0\n')
        self.python.chmod(0o755)
        self.cann = self.root / 'set_env.sh'
        self.cann.write_text('export ASCEND_HOME_PATH=/cann\n')
        self.conda_sh = self.root / 'conda.sh'
        self.conda_sh.write_text('conda() { export CONDA_DEFAULT_ENV="$2"; }\n')

    def run_script(self, script, path=None):
        env = {'HOME': str(self.root), 'PATH': path or f'{self.tools}:/usr/bin:/bin'}
        return subprocess.run([shutil.which('bash'), '-c', script],
                              capture_output=True, text=True, env=env)

    def source(self, *args):
        """Source the init script with `args`, then report the state it built."""
        quoted = ' '.join(shlex.quote(str(arg)) for arg in args)
        script = (f'. {shlex.quote(str(self.SCRIPT))} {quoted}\n'
                  'rc=$?\n'
                  'printf "rc=%s cann=%s conda_env=%s activated=%s python3=%s\\n" "$rc" '
                  '"${TILELANG_HIVM_CANN_SETUP:-}" "${TILELANG_HIVM_CONDA_ENV:-}" '
                  '"${CONDA_DEFAULT_ENV:-}" "$(command -v python3)"\n')
        return self.run_script(script)

    def test_values_from_env_sh_are_used_without_arguments(self):
        env_file = self.root / 'env.sh'
        env_file.write_text(f'TILELANG_HIVM_CANN_SETUP={self.cann}\n'
                            f'TILELANG_HIVM_CONDA_SH={self.conda_sh}\n'
                            'TILELANG_HIVM_CONDA_ENV=py311\n')
        script = (f'. {shlex.quote(str(env_file))}\n'
                  f'. {shlex.quote(str(self.SCRIPT))}\n'
                  'rc=$?\n'
                  'printf "rc=%s activated=%s\\n" "$rc" "${CONDA_DEFAULT_ENV:-}"\n')
        done = self.run_script(script)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=0', done.stdout)
        # The activation ran; it was not merely read from the file.
        self.assertIn('activated=py311', done.stdout)

    def test_cann_is_sourced_before_conda(self):
        events = self.root / 'events'
        cann = self.root / 'ordered-cann.sh'
        cann.write_text('printf "cann\\n" >> ' + shlex.quote(str(events)) + '\n'
                        'export ASCEND_HOME_PATH=/cann\n')
        conda_sh = self.root / 'ordered-conda.sh'
        conda_sh.write_text('printf "conda\\n" >> ' + shlex.quote(str(events)) + '\n'
                            'conda() { printf "activate\\n" >> ' + shlex.quote(str(events)) + '; }\n')
        done = self.source(cann, conda_sh, 'py311', '-')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=0', done.stdout)
        self.assertEqual(events.read_text().split(), ['cann', 'conda', 'activate'])

    def test_cann_without_ascend_home_path_is_not_ignored(self):
        incomplete = self.root / 'incomplete.sh'
        incomplete.write_text('true\n')
        done = self.source(incomplete, '-', '-', '-')
        self.assertIn('rc=1', done.stdout)
        self.assertIn('ASCEND_HOME_PATH', done.stderr)

    def test_a_dash_disables_an_optional_component(self):
        done = self.source('-', '-', '-', '-')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=0', done.stdout)
        self.assertIn('cann= conda_env=', done.stdout)

    def test_half_a_conda_configuration_is_refused(self):
        done = self.source('-', self.conda_sh, '-', '-')
        self.assertIn('rc=1', done.stdout)
        self.assertIn('must be provided together', done.stderr)

    def test_an_unreadable_cann_is_refused(self):
        done = self.source(self.root / 'missing.sh', '-', '-', '-')
        self.assertIn('rc=1', done.stdout)
        self.assertIn('CANN setup not readable', done.stderr)

    def test_a_failure_returns_instead_of_ending_the_callers_shell(self):
        # The report line after the sourced call only prints if the shell is
        # still alive, which is what makes this usable from an interactive session.
        script = (f'. {shlex.quote(str(self.SCRIPT))} missing.sh - - -\n'
                  'rc=$?\n'
                  'printf "survived rc=%s\\n" "$rc"\n')
        done = self.run_script(script)
        self.assertIn('survived rc=1', done.stdout)
        self.assertIn('CANN setup not readable', done.stderr)

    def test_the_tool_check_is_python3_not_cmake(self):
        # The build tree is already configured, so the Gate never reconfigures and
        # never needs cmake or ninja. Only the interpreter is required.
        self.assertFalse((self.tools / 'cmake').exists())
        self.assertFalse((self.tools / 'ninja').exists())
        done = self.source('-', '-', '-', '-')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=0', done.stdout)
        self.assertIn(f'python3={self.tools}/python3', done.stdout)
        text = self.SCRIPT.read_text()
        self.assertIn('command -v python3', text)
        self.assertNotIn('command -v cmake', text)
        self.assertNotIn('command -v ninja', text)

    def test_a_missing_interpreter_is_refused(self):
        # A PATH with no python3 at all: an empty directory first, then only the
        # directories that hold `bash` itself. `command -v python3` must fail and
        # the script must return non-zero with the message that names it.
        empty = self.root / 'empty-tools'
        empty.mkdir()
        done = self.run_script(f'. {shlex.quote(str(self.SCRIPT))} - - - -\n',
                               path=f'{empty}')
        self.assertNotEqual(done.returncode, 0,
                            'a PATH without python3 must make the script return non-zero')
        self.assertNotIn('rc=0', done.stdout)
        self.assertIn('python3', done.stderr)
        self.assertIn('not found', done.stderr)


class FullBuildCase(unittest.TestCase):
    """scripts/full-build.sh creates the fresh build tree the Gate then gates.

    The Gate does not call this file, so these cases run it the way a human does:
    sourced, with no auto-run, and with the values read from the calling shell.
    `cmake` and `ninja` are fakes that record the argv they are given, so the
    command is checked without compiling anything.
    """

    SCRIPT = BASE / 'scripts/full-build.sh'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_root = self.root / 'src'
        self.source_root.mkdir()
        (self.source_root / 'CMakeLists.txt').write_text('# fake\n')
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        self.record = self.root / 'argv.txt'
        self.out = self.root / 'out.txt'

    def write_tool(self, name, status=0):
        path = self.tools / name
        path.write_text('#!/bin/bash\n'
                        f'printf "{name} %s\\n" "$*" >> "$FULL_BUILD_ARGV"\n'
                        f'exit {status}\n')
        path.chmod(0o755)
        return path

    def call(self, args='full_build', *, configure_status=0, **overrides):
        self.write_tool('cmake', configure_status)
        self.write_tool('ninja')
        script = (f'. {shlex.quote(str(self.SCRIPT))}\n'
                  f'{args} > {shlex.quote(str(self.out))} 2>&1\n'
                  'rc=$?\n'
                  'pf=off; shopt -o pipefail | grep -q on && pf=on\n'
                  'printf "rc=%s pipefail=%s\\n" "$rc" "$pf"\n')
        env = {'HOME': str(self.root), 'PATH': f'{self.tools}:/usr/bin:/bin',
               'FULL_BUILD_ARGV': str(self.record),
               'TILELANG_HIVM_BUILD_SUBDIR': 'build',
               'TILELANG_HIVM_NINJA_JOBS': '8'}
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return subprocess.run([shutil.which('bash'), '-c', script],
                              capture_output=True, text=True, env=env,
                              cwd=str(self.source_root))

    def recorded(self):
        return self.record.read_text().splitlines() if self.record.exists() else []

    def test_sourcing_defines_the_function_and_never_builds(self):
        script = (f'. {shlex.quote(str(self.SCRIPT))}\n'
                  'declare -F full_build\n')
        done = subprocess.run([shutil.which('bash'), '-c', script],
                              capture_output=True, text=True,
                              env={'HOME': str(self.root), 'PATH': f'{self.tools}:/usr/bin:/bin'},
                              cwd=str(self.source_root))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('full_build', done.stdout)
        self.assertEqual(self.recorded(), [])
        self.assertFalse((self.source_root / 'build.log').exists())

    def test_the_documented_command_is_passed_to_cmake_and_ninja(self):
        done = self.call()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=0 pipefail=off', done.stdout)
        self.assertEqual(self.recorded(), [
            'cmake -S . -B ./build -G Ninja -DCMAKE_BUILD_TYPE=Release '
            '-DUSE_ASCEND=ON -DUSE_CUDA=OFF',
            'ninja -C ./build -j 8 tilelang',
        ])
        self.assertTrue((self.source_root / 'build.log').is_file())

    def test_extra_configure_arguments_are_appended(self):
        done = self.call(TILELANG_HIVM_CMAKE_ARGS='-DUSE_CUDA=ON')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(self.recorded()[0].endswith('-DUSE_CUDA=OFF -DUSE_CUDA=ON'),
                        self.recorded()[0])

    def test_an_existing_build_directory_is_reused_not_deleted(self):
        kept = self.source_root / 'build' / 'CMakeCache.txt'
        kept.parent.mkdir()
        kept.write_text('# configured earlier\n')
        done = self.call()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(kept.is_file(), 'the existing build directory was removed')
        self.assertIn('cmake -S . -B ./build', self.recorded()[0])
        self.assertNotIn('rm', self.SCRIPT.read_text().split('full_build()')[1])

    def test_a_failing_configure_stops_before_ninja_and_is_returned(self):
        done = self.call(configure_status=7)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=7 pipefail=off', done.stdout)
        self.assertIn('build failed with status 7', self.out.read_text())
        self.assertEqual([line.split()[0] for line in self.recorded()], ['cmake'])

    def test_the_default_job_count_is_bounded_by_available_memory(self):
        # The CPU count is not a usable default: 256 cores starting 256 compilers
        # on a shared server ends in `c++: fatal error: Killed signal terminated
        # program cc1plus`. Both inputs are read from the machine, so the case
        # checks the rule on synthetic figures rather than on this host.
        script = (f'. {shlex.quote(str(self.SCRIPT))}\n'
                  'for mb in 512 2048 4096; do\n'
                  '  printf "%s=%s\\n" "$mb" "$(_full_build_default_jobs "$mb")"\n'
                  'done\n')
        done = subprocess.run([shutil.which('bash'), '-c', script],
                              capture_output=True, text=True,
                              env={'HOME': str(self.root), 'PATH': f'{self.tools}:/usr/bin:/bin'},
                              cwd=str(self.source_root))
        self.assertEqual(done.returncode, 0, done.stderr)
        counts = dict(line.split('=') for line in done.stdout.split())

        # A per-job figure six times larger must ask for far fewer jobs, and no
        # figure may ever exceed the core count.
        cores = int(subprocess.run(['getconf', '_NPROCESSORS_ONLN'],
                                   capture_output=True, text=True).stdout.strip())
        self.assertLess(int(counts['4096']), int(counts['512']))
        for value in counts.values():
            self.assertLessEqual(int(value), cores)
            self.assertGreaterEqual(int(value), 1)

    def test_a_job_count_that_is_not_a_number_falls_back_to_the_core_count(self):
        script = (f'. {shlex.quote(str(self.SCRIPT))}\n'
                  '_full_build_default_jobs notanumber\n')
        done = subprocess.run([shutil.which('bash'), '-c', script],
                              capture_output=True, text=True,
                              env={'HOME': str(self.root), 'PATH': f'{self.tools}:/usr/bin:/bin'},
                              cwd=str(self.source_root))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(),
                         subprocess.run(['getconf', '_NPROCESSORS_ONLN'],
                                        capture_output=True, text=True).stdout.strip())

    def test_an_explicit_job_count_is_never_overridden(self):
        done = self.call(TILELANG_HIVM_NINJA_JOBS='3')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('ninja -C ./build -j 3 tilelang', self.recorded())

    def test_an_unexpected_argument_is_refused(self):
        done = self.call(args='full_build one two')
        self.assertIn('rc=1 pipefail=off', done.stdout)
        self.assertIn('expected no arguments, or exactly one', self.out.read_text())
        self.assertEqual(self.recorded(), [])

    def test_a_source_root_without_cmakelists_is_refused(self):
        empty = self.root / 'elsewhere'
        empty.mkdir()
        done = self.call(args=f'full_build {shlex.quote(str(empty))}')
        self.assertIn('rc=1 pipefail=off', done.stdout)
        self.assertIn('no CMakeLists.txt', self.out.read_text())
        self.assertEqual(self.recorded(), [])

    def test_a_build_subdirectory_with_a_separator_is_refused(self):
        for value in ('..', '../build', 'a/b', '/build', '.'):
            done = self.call(TILELANG_HIVM_BUILD_SUBDIR=value)
            self.assertIn('rc=1 pipefail=off', done.stdout, value)
            self.assertIn('must be one plain directory name', self.out.read_text(), value)
        self.assertEqual(self.recorded(), [])

    def test_an_empty_build_subdirectory_means_the_default(self):
        # The shell's `:-` default applies, so an empty value is the documented
        # `build`, not a directory named ''.
        done = self.call(TILELANG_HIVM_BUILD_SUBDIR='')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('cmake -S . -B ./build', self.recorded()[0])

    def test_missing_cmake_or_ninja_fails_before_configure(self):
        script = (f'. {shlex.quote(str(self.SCRIPT))}\n'
                  f'full_build > {shlex.quote(str(self.out))} 2>&1\n'
                  'printf "rc=%s\\n" "$?"\n')
        done = subprocess.run([shutil.which('bash'), '-c', script],
                              capture_output=True, text=True,
                              env={'HOME': str(self.root), 'PATH': '/nonexistent',
                                   'FULL_BUILD_ARGV': str(self.record)},
                              cwd=str(self.source_root))
        self.assertIn('rc=1', done.stdout)
        self.assertIn('cmake and ninja must be on PATH', self.out.read_text())
        self.assertFalse((self.source_root / 'build.log').exists())

    def test_full_build_never_activates_conda_or_sources_anything(self):
        # Activation belongs to init-toolchain.sh. Doing it here would mask an
        # ordering bug in that file, and sourcing a file would run code the
        # manifest did not pin.
        text = self.SCRIPT.read_text()
        self.assertNotIn('conda activate', text)
        for line in text.splitlines():
            self.assertFalse(re.match(r'\s*(source|\.)\s', line), line)


class InitScriptInteractionCase(unittest.TestCase):
    """The Gate ships scripts/init-toolchain.sh instead of a second copy."""

    def test_the_gate_inlines_the_committed_script_and_calls_it_once(self):
        script = base.container_script('/src', 'build', '4', 840,
                                       Path('/cann/set_env.sh'), None, None)
        committed = base.INIT_SCRIPT.read_text(encoding='utf-8').rstrip('\n')
        self.assertIn(committed, script)
        calls = [line for line in script.splitlines()
                 if line.startswith('init_toolchain ') and '|| fail' in line]
        self.assertEqual(len(calls), 1, calls)
        self.assertIn('init_toolchain /cann/set_env.sh - - - '
                      '|| fail "toolchain initialization failed"', script)

    def test_the_gate_keeps_no_second_copy_of_the_sequence(self):
        script = base.container_script('/src', 'build', '4', 840, None, None, None)
        branch = script.split('init_toolchain')[-1]
        self.assertNotIn('. /cann', script)
        self.assertIn('init_toolchain - - - -', script)
        self.assertNotIn('conda activate', branch)
        # The old hard-coded PATH line is gone: the committed script owns PATH.
        self.assertNotIn('export PATH=/usr/local/sbin:/usr/local/bin', script)

    def test_unset_cann_and_conda_mean_configure_nothing_and_continue(self):
        script = base.container_script('/src', 'build', '4', 840, None, None, None)
        self.assertIn('init_toolchain - - - -', script)
        self.assertNotIn('fail "CANN setup not readable', script)

    def test_a_half_conda_configuration_fails_closed_on_the_host(self):
        with self.assertRaises(base.BridgeError) as error:
            base.require_conda_pair(Path('/conda.sh'), None)
        self.assertIn('must be provided together', str(error.exception))
        with self.assertRaises(base.BridgeError):
            base.require_conda_pair(None, 'py311')
        base.require_conda_pair(None, None)
        base.require_conda_pair(Path('/conda.sh'), 'py311')

    def test_the_three_values_are_forwarded_in_order(self):
        # CANN, conda.sh and the environment name keep their order. The fourth
        # value, the directory Conda searches for environments, is optional and
        # reported as unset, so a machine without one passes the same argument
        # count and the init script sees an unchanged signature.
        self.assertEqual(base.init_args(Path('/cann.sh'), Path('/conda.sh'), 'py311'),
                         '/cann.sh /conda.sh py311 -')
        self.assertEqual(base.init_args(None, None, None), '- - - -')
        self.assertEqual(base.init_args(Path('/cann.sh'), Path('/conda.sh'), 'py311',
                                        Path('/envs')),
                         '/cann.sh /conda.sh py311 /envs')


class _FakeDockerCase(unittest.TestCase):
    """Shared Host-side fake for `docker inspect` and `docker exec`."""

    def setUp(self):
        # The bridge prints its observation JSON to stdout; keep the suite quiet.
        # unittest reports on stderr, so this hides only the bridge output.
        context = contextlib.redirect_stdout(io.StringIO())
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'tilelang'
        (self.source / 'build').mkdir(parents=True)
        (self.source / 'build' / 'build.ninja').write_text('# fake build tree\n')
        (self.source / 'tilelang').mkdir()
        (self.source / 'tilelang' / '__init__.py').write_text('# fake module\n')
        (self.source / 'build' / 'lib').mkdir()
        (self.source / 'build' / 'lib' / 'libtilelang.so').write_bytes(b'fake native\n')
        for test in base.REGRESSION_TESTS:
            path = self.source / test
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fake test\n')
        gate = self.source / npu.GATE_TEST
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text('# fake gate test\n')
        origin = os.getcwd()
        self.addCleanup(os.chdir, origin)
        os.chdir(self.source)
        # Container identity is read from the machine file, never from the
        # launching shell: a stale export used to be the only source and could
        # name a stopped container while the file named the live one. Tests
        # therefore write a machine file rather than exporting the three keys.
        self.identity_file = self.write_identity_file()
        os.environ['TILELANG_HIVM_ENV_FILE'] = str(self.identity_file)
        for name in ('TILELANG_HIVM_CONTAINER_NAME', 'TILELANG_HIVM_CONTAINER_USER',
                     'TILELANG_HIVM_CONTAINER_HOME', 'TILELANG_HIVM_CONTAINER_SOURCE',
                     'TILELANG_HIVM_CANN_SETUP', 'TILELANG_HIVM_CONDA_SH',
                     'TILELANG_HIVM_CONDA_ENV', 'TILELANG_HIVM_BUILD_SUBDIR',
                     'TILELANG_HIVM_NINJA_JOBS', 'TILELANG_HIVM_GATE_TIMEOUT',
                     'TILELANG_HIVM_DEVICE_ID'):
            os.environ.pop(name, None)

    def write_identity_file(self, *, name='fake-container', user='u', home='/home/u', extra=''):
        path = self.root / 'env.sh'
        path.write_text(f'TILELANG_HIVM_CONTAINER_NAME={name}\n'
                        f'TILELANG_HIVM_CONTAINER_USER={user}\n'
                        f'TILELANG_HIVM_CONTAINER_HOME={home}\n' + extra)
        return path

    def digest(self, relative):
        return hashlib.sha256((self.source / relative).read_bytes()).hexdigest()

    @staticmethod
    def script_of(call):
        """The container shell body is the argv element starting with `set -o pipefail`."""
        for item in call:
            if item.startswith('set -o pipefail'):
                return item
        return None

    @classmethod
    def command_of(cls, call):
        """The Host-side `bash -c <script> bash <command>` tail names the command."""
        script = cls.script_of(call)
        if script is None:
            return None
        body = call.index(script)
        return call[body + 2] if len(call) > body + 2 else None

    @classmethod
    def docker_exec_scripts(cls, calls):
        """Only the `docker exec` calls that carry a container script."""
        return [call for call in calls if call[:2] == ['docker', 'exec']
                and cls.script_of(call) is not None]

    def mounts(self):
        return [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]

    def base_facts(self):
        return {
            'pythonModuleRel': 'tilelang/__init__.py',
            'pythonModuleSha256': self.digest('tilelang/__init__.py'),
            'nativeLibs': {'build/lib/libtilelang.so': self.digest('build/lib/libtilelang.so')},
            'ascendBackendCompiled': True,
            'devMode': True,
        }

    def fake_base(self, *, mounts=None, inspect_ok=True, identity=None, exit_code=0,
                  build_exit=None, facts=None, tests=None, junit=True, stdout=None, calls=None):
        mounts = self.mounts() if mounts is None else mounts
        stat = self.source.stat()
        identity = identity or f'{stat.st_dev}:{stat.st_ino}'
        payload = {
            'command': 'regression',
            'facts': self.base_facts() if facts is None else facts,
            'junitPresent': junit,
            'pytestExit': exit_code,
            'tests': tests if tests is not None else {
                'collected': 14, 'passed': 14, 'failed': 0, 'errors': 0, 'skipped': 0},
        }

        def run(argv, **kwargs):
            if calls is not None:
                calls.append(list(argv))
            if argv[:2] == ['docker', 'inspect']:
                if not inspect_ok:
                    return subprocess.CompletedProcess(argv, 1, '', 'no such container')
                return subprocess.CompletedProcess(argv, 0, json.dumps(mounts), '')
            if argv[:2] == ['docker', 'exec']:
                if 'stat -c' in ' '.join(argv):
                    return subprocess.CompletedProcess(argv, 0, identity + '\n', '')
                command = argv[-1]
                if command == 'build':
                    code = exit_code if build_exit is None else build_exit
                    body = {'command': 'build', 'exitCode': code}
                elif command == 'provenance':
                    body = {'command': 'provenance', 'facts': self.base_facts()
                            if facts is None else facts}
                else:
                    body = payload
                text = json.dumps(body) + '\n' if stdout is None else stdout
                return subprocess.CompletedProcess(argv, exit_code, text, 'container output\n')
            return subprocess.CompletedProcess(argv, 0, '', '')
        return run

    def fake_npu(self, *, mounts=None, inspect_ok=True, identity=None, exit_code=0,
                 report=None, stdout=None, calls=None, device='0',
                 occupancy=(), probe_exit=0, probe_text=None):
        """Host-side fake for the NPU bridge, including the `npu-smi` probe.

        `occupancy` lists `(pid, name)` pairs the probe reports for the
        authorized device; the default is an idle card. `probe_exit` and
        `probe_text` model a driver that refuses or answers oddly, which the
        bridge must treat as unknown rather than free.
        """
        mounts = self.mounts() if mounts is None else mounts
        stat = self.source.stat()
        identity = identity or f'{stat.st_dev}:{stat.st_ino}'
        body = {
            'command': 'npu', 'fatal': None, 'junitPresent': True,
            'tests': {'collected': 1, 'passed': 1, 'failed': 0, 'errors': 0, 'skipped': 0},
            'pytestExit': 0, 'phyBefore': int(device), 'phyAfter': int(device), 'npuAvailable': True,
            'npuDeviceCount': 1, 'currentDevice': int(device), 'deviceName': 'Ascend950PR_9579',
            'driverDeviceCount': 1, 'tensorMovesToDevice': 6, 'numericalAssertions': 1,
            'npuPeakMemoryBytes': 4096, 'synchronizeOk': True, 'synchronizeError': None,
        }
        body.update(report or {})

        def probe(argv):
            if probe_text is not None:
                return subprocess.CompletedProcess(argv, probe_exit, probe_text, '')
            if probe_exit != 0:
                return subprocess.CompletedProcess(argv, probe_exit, 'driver unavailable\n', '')
            if argv[2:] == ['-m']:
                return subprocess.CompletedProcess(
                    argv, 0, '\tNPU ID   Slot ID  Chip ID  Chip Phy-ID  Chip Name\n'
                              f'\t{device}  76  0  {device}  Ascend950PR\n', '')
            npu_id = argv[argv.index('-i') + 1]
            lines = [f'\tNPU ID                         : {npu_id}']
            # The real driver prints this sentence, and only this sentence,
            # when the device holds nothing; an empty table with no marker is
            # an answer the bridge refuses.
            if not occupancy:
                lines.append('\tNo process in device.')
            lines += [f'\tProcess id:{pid} Process name:{name}           Process memory(MB):256'
                      for pid, name in occupancy]
            return subprocess.CompletedProcess(argv, 0, '\n'.join(lines) + '\n', '')

        def run(argv, **kwargs):
            if calls is not None:
                calls.append(list(argv))
            if argv[:1] == [npu.NPU_SMI]:
                return probe(argv)
            if argv[:2] == ['docker', 'inspect']:
                if not inspect_ok:
                    return subprocess.CompletedProcess(argv, 1, '', 'no such container')
                return subprocess.CompletedProcess(argv, 0, json.dumps(mounts), '')
            if argv[:2] == ['docker', 'exec']:
                if 'stat -c' in ' '.join(argv):
                    return subprocess.CompletedProcess(argv, 0, identity + '\n', '')
                text = json.dumps(body) + '\n' if stdout is None else stdout
                return subprocess.CompletedProcess(argv, exit_code, text, 'container output\n')
            return subprocess.CompletedProcess(argv, 0, '', '')
        return run


class BaseRunCase(_FakeDockerCase):
    def test_valid_mapping_runs_the_real_command_and_returns_zero(self):
        calls = []
        with patch('subprocess.run', side_effect=self.fake_base(calls=calls)):
            self.assertEqual(base.main(['--timeout', '840', 'build']), 0)
        exec_calls = self.docker_exec_scripts(calls)
        self.assertTrue(exec_calls)
        self.assertFalse(any('-t' in call for call in calls))
        script = self.script_of(exec_calls[-1])
        self.assertIn('ninja -C build', script)
        self.assertIn('$(nproc)', script)
        self.assertIn('tilelang', script)

    def test_ninja_is_never_replaced_by_a_weaker_proxy(self):
        calls = []
        with patch('subprocess.run', side_effect=self.fake_base(calls=calls)):
            base.main(['--timeout', '840', 'build'])
        script = self.script_of(next(call for call in self.docker_exec_scripts(calls)
                                     if self.command_of(call) == 'build'))
        # Only the `build` branch matters here: the `provenance` branch legitimately
        # imports TileLang to hash it, and the `regression` branch runs pytest.
        branch = script.split('  build)')[1].split('    ;;')[0]
        for proxy in ('compileall', 'import tilelang', 'true', 'echo PASS'):
            self.assertNotIn(proxy, branch)
        self.assertIn('ninja -C build', branch)

    def test_regression_runs_the_pinned_test_files(self):
        calls = []
        with patch('subprocess.run', side_effect=self.fake_base(calls=calls)):
            self.assertEqual(base.main(['--timeout', '900', 'regression']), 0)
        script = self.script_of(next(call for call in self.docker_exec_scripts(calls)
                                     if self.command_of(call) == 'regression'))
        for test in base.REGRESSION_TESTS:
            self.assertIn(test, script)
        self.assertIn('python3 -m pytest', script)
        self.assertIn('--junitxml=', script)

    def test_a_nonzero_pytest_exit_is_returned_as_failure(self):
        tests = {'collected': 14, 'passed': 12, 'failed': 2, 'errors': 0, 'skipped': 0}
        with patch('subprocess.run',
                   side_effect=self.fake_base(exit_code=1, tests=tests, junit=True)):
            self.assertEqual(base.main(['--timeout', '900', 'regression']), 1)

    def test_a_timeout_exit_code_is_returned_unchanged(self):
        for command, argv in (('build', ['--timeout', '840', 'build']),
                              ('regression', ['--timeout', '900', 'regression'])):
            with patch('subprocess.run', side_effect=self.fake_base(exit_code=124)):
                self.assertEqual(base.main(argv), 124, command)

    def test_a_nonzero_build_exit_is_reported_by_the_observation(self):
        # The observation the Harness compares is derived from the exit code the
        # container itself reported, so a failed Ninja can never read as zero.
        failed = base.read_report('{"command": "build", "exitCode": 2}\n', command='build')
        passed = base.read_report('{"command": "build", "exitCode": 0}\n', command='build')
        self.assertNotEqual(failed.get('exitCode'), 0)
        self.assertEqual(passed.get('exitCode'), 0)

    def test_a_timed_out_build_is_reported_by_the_observation(self):
        timed_out = base.read_report('{"command": "build", "exitCode": 124}\n', command='build')
        self.assertNotEqual(timed_out.get('exitCode'), 0)

    def test_a_missing_report_fails_closed(self):
        for stdout in ('', 'not json\n', '{"command": "wrong"}\n', '[1, 2, 3]\n'):
            with patch('subprocess.run', side_effect=self.fake_base(stdout=stdout)):
                with self.assertRaises(base.BridgeError, msg=repr(stdout)):
                    base.main(['--timeout', '900', 'regression'])

    def test_identity_mismatch_is_refused(self):
        with patch('subprocess.run', side_effect=self.fake_base(identity='1:1')):
            with self.assertRaises(base.BridgeError) as error:
                base.main(['--timeout', '840', 'build'])
        self.assertIn('not the host source root', str(error.exception))

    def test_missing_container_fails_before_any_exec(self):
        calls = []
        with patch('subprocess.run',
                   side_effect=self.fake_base(inspect_ok=False, calls=calls)):
            with self.assertRaises(base.BridgeError):
                base.main(['--timeout', '840', 'build'])
        self.assertTrue(all(call[:2] == ['docker', 'inspect'] for call in calls))

    def test_unshared_source_tree_fails_before_any_exec(self):
        mounts = [{'Type': 'bind', 'Source': '/somewhere/else', 'Destination': '/somewhere/else'}]
        calls = []
        with patch('subprocess.run', side_effect=self.fake_base(mounts=mounts, calls=calls)):
            with self.assertRaises(base.BridgeError):
                base.main(['--timeout', '840', 'build'])
        self.assertFalse(any(call[:2] == ['docker', 'exec'] for call in calls))

    def test_declared_container_source_must_match_the_mapping(self):
        os.environ['TILELANG_HIVM_CONTAINER_SOURCE'] = '/a/different/path'
        with patch('subprocess.run', side_effect=self.fake_base()):
            with self.assertRaises(base.BridgeError) as error:
                base.main(['--timeout', '840', 'build'])
        self.assertIn('does not match the bind-mount mapping', str(error.exception))

    def test_an_unknown_command_is_refused(self):
        for command in ('frobnicate', 'install', 'test', '--help'):
            with self.assertRaises(base.BridgeError, msg=command):
                base.parse_argv(['--timeout', '840', command])

    def test_no_source_logs_are_written(self):
        before = sorted(path.relative_to(self.source) for path in self.source.rglob('*'))
        with patch('subprocess.run', side_effect=self.fake_base()):
            base.main(['--timeout', '900', 'regression'])
        after = sorted(path.relative_to(self.source) for path in self.source.rglob('*'))
        self.assertEqual(before, after)


class IdentityCase(_FakeDockerCase):
    def test_matching_module_and_native_libraries_pass(self):
        result = base.identity_observations(self.base_facts(), self.source)
        self.assertTrue(result['importedModuleSha256MatchesHost'])
        self.assertTrue(result['nativeLibsPresent'])
        self.assertTrue(result['loadedNativeSha256MatchesHost'])

    def test_a_different_imported_module_cannot_pass(self):
        facts = self.base_facts()
        facts['pythonModuleSha256'] = '0' * 64
        result = base.identity_observations(facts, self.source)
        self.assertFalse(result['importedModuleSha256MatchesHost'])

    def test_a_different_native_library_cannot_pass(self):
        facts = self.base_facts()
        facts['nativeLibs'] = {'build/lib/libtilelang.so': '0' * 64}
        result = base.identity_observations(facts, self.source)
        self.assertFalse(result['loadedNativeSha256MatchesHost'])

    def test_a_missing_native_library_set_cannot_pass(self):
        facts = self.base_facts()
        facts['nativeLibs'] = {}
        result = base.identity_observations(facts, self.source)
        self.assertFalse(result['nativeLibsPresent'])
        self.assertFalse(result['loadedNativeSha256MatchesHost'])

    def test_an_artifact_outside_the_source_root_is_refused(self):
        for relative in ('/etc/passwd', '../../etc/passwd', ''):
            facts = self.base_facts()
            facts['pythonModuleRel'] = relative
            with self.assertRaises(base.BridgeError, msg=repr(relative)):
                base.identity_observations(facts, self.source)

    def test_an_artifact_that_does_not_exist_on_the_host_is_refused(self):
        facts = self.base_facts()
        facts['pythonModuleRel'] = 'tilelang/not_there.py'
        facts['pythonModuleSha256'] = '0' * 64
        with self.assertRaises(base.BridgeError):
            base.identity_observations(facts, self.source)

    def test_module_or_native_mismatch_makes_the_regression_gate_fail(self):
        facts = self.base_facts()
        facts['pythonModuleSha256'] = '0' * 64
        report = {'command': 'regression', 'facts': facts, 'junitPresent': True,
                  'pytestExit': 0, 'tests': {'collected': 5, 'passed': 5, 'failed': 0,
                                             'errors': 0, 'skipped': 0}}
        observed = base.regression_observations(report, self.source)
        self.assertFalse(observed['importedModuleSha256MatchesHost'])


class HivmCompilerForwardingCase(_FakeDockerCase):
    """TileLang reads HIVM_COMPILER from the process environment, docker exec
    inherits nothing, so the Gate has to name it on the exec argv.

    Without the forwarding, `testing/ascend/codegen/test_hivm_build.py` skips
every test that needs the compiler and the run stays green. These cases pin
that the value reaches the container, that an unusable value is refused
rather than degraded to that skip, and that an unset value changes nothing on
a machine that has no compiler.
    """

    def test_the_value_is_forwarded_on_the_exec_argv(self):
        os.environ['TILELANG_HIVM_ENV_FILE'] = str(self.write_identity_file(
            extra='TILELANG_HIVM_HIVM_COMPILER=/opt/ir/bin/bishengir-compile\n'))
        calls = []
        with patch('subprocess.run', side_effect=self.fake_base(calls=calls)):
            self.assertEqual(base.main(['--timeout', '840', 'build']), 0)
        exec_calls = self.docker_exec_scripts(calls)
        self.assertTrue(exec_calls)
        for call in exec_calls:
            self.assertIn('-e', call)
            self.assertIn('HIVM_COMPILER=/opt/ir/bin/bishengir-compile', call)

    def test_the_container_script_exports_it_and_requires_it_to_be_executable(self):
        compiler = Path('/opt/ir/bin/bishengir-compile')
        script = base.container_script('/src', 'build', '4', 840, None, None, None,
                                       None, compiler)
        self.assertIn('[ -x /opt/ir/bin/bishengir-compile ]', script)
        self.assertIn('export HIVM_COMPILER=/opt/ir/bin/bishengir-compile', script)

    def test_an_unset_compiler_forwards_nothing_and_checks_nothing(self):
        os.environ['TILELANG_HIVM_ENV_FILE'] = str(self.write_identity_file())
        calls = []
        with patch('subprocess.run', side_effect=self.fake_base(calls=calls)):
            self.assertEqual(base.main(['--timeout', '840', 'build']), 0)
        for call in calls:
            self.assertFalse(any(str(item).startswith('HIVM_COMPILER=') for item in call), call)
        script = base.container_script('/src', 'build', '4', 840, None, None, None)
        self.assertNotIn('HIVM_COMPILER', script)

    def test_the_target_is_forwarded_only_with_the_compiler(self):
        os.environ['TILELANG_HIVM_ENV_FILE'] = str(self.write_identity_file(
            extra='TILELANG_HIVM_HIVM_COMPILER=/opt/ir/bin/bishengir-compile\n'
                  'TILELANG_HIVM_HIVM_TARGET=Ascend950PR_9579\n'))
        calls = []
        with patch('subprocess.run', side_effect=self.fake_base(calls=calls)):
            base.main(['--timeout', '840', 'build'])
        forwarded = self.docker_exec_scripts(calls)[0]
        self.assertIn('HIVM_TARGET=Ascend950PR_9579', forwarded)

    def test_a_relative_compiler_path_is_refused(self):
        os.environ['TILELANG_HIVM_ENV_FILE'] = str(self.write_identity_file(
            extra='TILELANG_HIVM_HIVM_COMPILER=bin/bishengir-compile\n'))
        with self.assertRaises(base.BridgeError) as error:
            base.hivm_from_env_file()
        self.assertIn('absolute', str(error.exception))

    def test_a_target_without_a_compiler_is_refused(self):
        os.environ['TILELANG_HIVM_ENV_FILE'] = str(self.write_identity_file(
            extra='TILELANG_HIVM_HIVM_TARGET=Ascend950PR_9579\n'))
        with self.assertRaises(base.BridgeError) as error:
            base.hivm_from_env_file()
        self.assertIn('TILELANG_HIVM_HIVM_COMPILER is empty', str(error.exception))

    def test_a_target_for_another_device_is_refused(
            self):
        # tilelang/contrib/hivm.py refuses a differing HIVM_TARGET because the
        # module target, the compiler `--target` and the ELF must name one device.
        # The Gate refuses it on the Host so a typo is a configuration error.
        os.environ['TILELANG_HIVM_ENV_FILE'] = str(self.write_identity_file(
            extra='TILELANG_HIVM_HIVM_COMPILER=/opt/ir/bin/bishengir-compile\n'
                  'TILELANG_HIVM_HIVM_TARGET=Ascend910B\n'))
        with self.assertRaises(base.BridgeError) as error:
            base.hivm_from_env_file()
        self.assertIn('Ascend950PR_9579', str(error.exception))

    def test_check_config_reports_the_compiler_without_entering_the_container(self):
        compiler = self.root / 'bishengir-compile'
        compiler.write_text('#!/bin/sh\nexit 0\n')
        compiler.chmod(0o755)
        os.environ['TILELANG_HIVM_ENV_FILE'] = str(self.write_identity_file(
            extra=f'TILELANG_HIVM_HIVM_COMPILER={compiler}\n'))
        calls = []
        with patch('subprocess.run', side_effect=self.fake_base(calls=calls)):
            self.assertEqual(base.check_config(), 0)
        self.assertEqual(calls, [])

    def test_the_template_documents_the_compiler_and_holds_no_value(self):
        template = (BASE / 'env.example.sh').read_text()
        defined = {line.split('=', 1)[0]: line.split('=', 1)[1].strip()
                   for line in template.splitlines()
                   if line and not line.startswith('#') and '=' in line}
        for key in ('TILELANG_HIVM_HIVM_COMPILER', 'TILELANG_HIVM_HIVM_TARGET'):
            self.assertIn(key, defined)
            self.assertEqual(defined[key], '', key)
        # Both keys are accepted by the strict NAME=value reader, so a machine
        # that fills them in is not refused for using an unknown key.
        for key in ('TILELANG_HIVM_HIVM_COMPILER', 'TILELANG_HIVM_HIVM_TARGET'):
            self.assertIn(key, base.TOOLCHAIN_KEYS)
        # They belong to the Gates, not to the provisioning script.
        for key in ('TILELANG_HIVM_HIVM_COMPILER', 'TILELANG_HIVM_HIVM_TARGET'):
            self.assertNotIn(key, base.PROVISIONING_KEYS)

    def test_both_gates_accept_every_key_the_machine_file_may_define(self):
        # The two Gates read ONE machine file, so a key that only one of them
        # accepts makes that file unusable for the other. The device Gate does
        # not read the compiler itself, which is exactly why the omission is
        # easy to miss: it fails only when a machine sets the key.
        template = (BASE / 'env.example.sh').read_text()
        defined = [line.split('=', 1)[0].strip()
                   for line in template.splitlines()
                   if line and not line.startswith('#') and '=' in line]
        self.assertTrue(defined)
        for key in defined:
            self.assertIn(key, base.TOOLCHAIN_KEYS, key)
            self.assertIn(key, npu.TOOLCHAIN_KEYS, key)
        # The provisioning split must agree as well: neither Gate may treat a
        # Gate-owned key as something `setup-python-env.sh` provisions.
        self.assertEqual(base.PROVISIONING_KEYS, npu.PROVISIONING_KEYS)
        for key in ('TILELANG_HIVM_HIVM_COMPILER', 'TILELANG_HIVM_HIVM_TARGET'):
            self.assertNotIn(key, npu.PROVISIONING_KEYS)


class RegressionStatisticsCase(unittest.TestCase):
    """The statistics contract: zero tests, an all-SKIPPED suite and an offline
    Ascend backend must each be visible as a false observation."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'tilelang').mkdir()
        (self.root / 'tilelang' / '__init__.py').write_text('x\n')
        self.digest = hashlib.sha256((self.root / 'tilelang' / '__init__.py').read_bytes()).hexdigest()

    def observe(self, tests, *, junit=True, ascend=True):
        report = {
            'command': 'regression',
            'facts': {'pythonModuleRel': 'tilelang/__init__.py',
                      'pythonModuleSha256': self.digest,
                      'nativeLibs': {},
                      'ascendBackendCompiled': ascend,
                      'devMode': True},
            'junitPresent': junit,
            'pytestExit': 0,
            'tests': tests,
        }
        return base.regression_observations(report, self.root)

    def test_a_real_passing_run_is_visible(self):
        observed = self.observe({'collected': 14, 'passed': 14, 'failed': 0,
                                 'errors': 0, 'skipped': 0})
        self.assertTrue(observed['testsCollectedPositive'])
        self.assertTrue(observed['notAllSkipped'])
        self.assertTrue(observed['allPassed'])
        self.assertEqual(observed['testsCollected'], 14)
        self.assertEqual(observed['testsPassed'], 14)

    def test_a_zero_test_run_cannot_pass(self):
        observed = self.observe({'collected': 0, 'passed': 0, 'failed': 0,
                                 'errors': 0, 'skipped': 0})
        self.assertFalse(observed['testsCollectedPositive'])
        self.assertFalse(observed['notAllSkipped'])
        self.assertFalse(observed['allPassed'])

    def test_an_all_skipped_suite_cannot_pass(self):
        observed = self.observe({'collected': 3, 'passed': 0, 'failed': 0,
                                 'errors': 0, 'skipped': 3})
        self.assertFalse(observed['notAllSkipped'])
        self.assertFalse(observed['allPassed'])

    def test_a_partly_skipped_suite_with_a_pass_still_reports_the_skip(self):
        observed = self.observe({'collected': 4, 'passed': 3, 'failed': 0,
                                 'errors': 0, 'skipped': 1})
        self.assertTrue(observed['notAllSkipped'])
        self.assertTrue(observed['allPassed'])
        self.assertEqual(observed['testsSkipped'], 1)

    def test_a_missing_junit_report_cannot_pass(self):
        observed = self.observe({'collected': 0, 'passed': 0, 'failed': 0,
                                 'errors': 0, 'skipped': 0}, junit=False)
        self.assertFalse(observed['junitPresent'])
        self.assertFalse(observed['testsCollectedPositive'])

    def test_a_missing_tests_object_cannot_pass(self):
        report = {'command': 'regression',
                  'facts': {'pythonModuleRel': 'tilelang/__init__.py',
                            'pythonModuleSha256': self.digest, 'nativeLibs': {},
                            'ascendBackendCompiled': True},
                  'junitPresent': True, 'pytestExit': 0}
        observed = base.regression_observations(report, self.root)
        self.assertFalse(observed['testsCollectedPositive'])
        self.assertFalse(observed['allPassed'])

    def test_an_uncompiled_ascend_backend_is_reported_as_such(self):
        observed = self.observe({'collected': 0, 'passed': 0, 'failed': 0,
                                 'errors': 0, 'skipped': 0}, ascend=False)
        self.assertFalse(observed['ascendBackendCompiled'])


class NpuRunCase(_FakeDockerCase):
    def test_an_authorized_device_runs_the_pinned_gate_test(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        calls = []
        with patch('subprocess.run', side_effect=self.fake_npu(calls=calls)):
            self.assertEqual(npu.main(['--timeout', '900']), 0)
        script = self.script_of(self.docker_exec_scripts(calls)[0])
        self.assertIn(npu.GATE_TEST, script)
        self.assertIn('"-k", select, test', script)
        self.assertIn('simd', script)
        self.assertIn('ASCEND_RT_VISIBLE_DEVICES', script)

    def test_without_an_authorized_device_no_command_is_spawned(self):
        calls = []
        with patch('subprocess.run', side_effect=self.fake_npu(calls=calls)):
            with self.assertRaises(npu.BridgeError):
                npu.main(['--timeout', '900'])
        self.assertEqual(calls, [])

    def test_a_malformed_device_id_spawns_no_command(self):
        for value in ('abc', '-1', '999'):
            os.environ['TILELANG_HIVM_DEVICE_ID'] = value
            calls = []
            with patch('subprocess.run', side_effect=self.fake_npu(calls=calls)):
                with self.assertRaises(npu.BridgeError, msg=value):
                    npu.main(['--timeout', '900'])
            self.assertEqual(calls, [], value)

    def test_a_task_without_the_npu_adapter_runs_no_device_command(self):
        # The Host-only path must not reach the device script at all.
        calls = []
        with patch('subprocess.run', side_effect=self.fake_base(calls=calls)):
            self.assertEqual(base.main(['--timeout', '900', 'regression']), 0)
        joined = ' '.join(' '.join(call) for call in calls)
        self.assertNotIn('npu_gate.py', joined)
        self.assertNotIn('TILELANG_HIVM_DEVICE_ID', joined)

    def test_a_synchronization_failure_cannot_pass(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        report = {'synchronizeOk': False,
                  'synchronizeError': 'RuntimeError: device synchronize failed'}
        with patch('subprocess.run', side_effect=self.fake_npu(report=report, exit_code=1)):
            self.assertNotEqual(npu.main(['--timeout', '900']), 0)

    def test_a_synchronization_failure_is_visible_in_the_observations(self):
        report = {'synchronizeOk': False,
                  'synchronizeError': 'RuntimeError: device synchronize failed'}
        observed = npu.observations(report, '0')
        self.assertFalse(observed['synchronizeOk'])
        self.assertEqual(observed['synchronizeError'],
                         'RuntimeError: device synchronize failed')
        self.assertFalse(npu.passed(observed))

    def test_a_missing_synchronization_field_cannot_pass(self):
        report = {'synchronizeOk': None}
        observed = npu.observations(report, '0')
        self.assertFalse(observed['synchronizeOk'])
        self.assertFalse(npu.passed(observed))

    def test_the_runner_reports_synchronization_and_exits_nonzero_on_failure(self):
        runner = npu.RUNNER
        self.assertIn('synchronizeOk', runner)
        self.assertIn('synchronizeError', runner)
        self.assertIn('torch.npu.synchronize()', runner)
        # No device reset, no device substitution and no retry loop.
        for forbidden in ('reset_device', 'npu.synchronize()\n            except BaseException:\n                pass',
                          'for attempt', 'while True', 'time.sleep'):
            self.assertNotIn(forbidden, runner)
        self.assertIn('sys.exit(0 if', runner)
        # The synchronization block must not swallow the error.
        self.assertNotIn('except BaseException:\n            pass', runner)

    def test_a_zero_test_run_cannot_pass(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        report = {'tests': {'collected': 0, 'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0},
                  'pytestExit': 5}
        with patch('subprocess.run', side_effect=self.fake_npu(report=report, exit_code=5)):
            self.assertNotEqual(npu.main(['--timeout', '900']), 0)

    def test_an_all_skipped_suite_cannot_pass(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        report = {'tests': {'collected': 2, 'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 2},
                  'pytestExit': 0, 'tensorMovesToDevice': 0, 'npuPeakMemoryBytes': 0}
        with patch('subprocess.run', side_effect=self.fake_npu(report=report, exit_code=0)):
            self.assertNotEqual(npu.main(['--timeout', '900']), 0)

    def test_a_wrong_device_mapping_cannot_pass(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        report = {'phyBefore': 1, 'phyAfter': 1}
        with patch('subprocess.run', side_effect=self.fake_npu(report=report, exit_code=0)):
            self.assertNotEqual(npu.main(['--timeout', '900']), 0)

    def test_an_unavailable_device_cannot_pass(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        report = {'npuAvailable': False, 'npuDeviceCount': 0, 'currentDevice': None,
                  'deviceName': None, 'phyBefore': None, 'phyAfter': None,
                  'tensorMovesToDevice': 0, 'npuPeakMemoryBytes': 0, 'pytestExit': 0}
        with patch('subprocess.run', side_effect=self.fake_npu(report=report, exit_code=0)):
            self.assertNotEqual(npu.main(['--timeout', '900']), 0)

    def test_a_cpu_fallback_cannot_pass(self):
        # pytest may pass while the kernel ran on the host CPU: no tensor reached
        # the device and the device allocated nothing.
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        report = {'tensorMovesToDevice': 0, 'numericalAssertions': 1,
                  'npuPeakMemoryBytes': 0}
        with patch('subprocess.run', side_effect=self.fake_npu(report=report, exit_code=0)):
            self.assertNotEqual(npu.main(['--timeout', '900']), 0)

    def test_missing_numerical_assertions_cannot_pass(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        report = {'numericalAssertions': 0}
        with patch('subprocess.run', side_effect=self.fake_npu(report=report, exit_code=0)):
            self.assertNotEqual(npu.main(['--timeout', '900']), 0)

    def test_a_timeout_cannot_pass(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        with patch('subprocess.run', side_effect=self.fake_npu(exit_code=124)):
            self.assertEqual(npu.main(['--timeout', '900']), 124)

    def test_a_missing_or_wrong_report_fails_closed(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        for stdout in ('', 'no report here\n', '{"command": "build"}\n', '{bad json\n'):
            with patch('subprocess.run', side_effect=self.fake_npu(stdout=stdout, exit_code=0)):
                with self.assertRaises(npu.BridgeError, msg=repr(stdout)):
                    npu.main(['--timeout', '900'])

    def test_a_fatal_container_error_is_recorded_and_fails(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        report = {'fatal': 'ModuleNotFoundError: No module named torch_npu'}
        observed = npu.observations(report, '0')
        self.assertFalse(npu.passed(observed))
        with patch('subprocess.run',
                   side_effect=self.fake_npu(report=report, exit_code=1)):
            self.assertNotEqual(npu.main(['--timeout', '900']), 0)

    def test_identity_mismatch_is_refused_before_the_device_is_touched(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        calls = []
        with patch('subprocess.run', side_effect=self.fake_npu(identity='9:9', calls=calls)):
            with self.assertRaises(npu.BridgeError):
                npu.main(['--timeout', '900'])
        self.assertFalse(any(call[:2] == ['docker', 'exec'] and 'ASCEND' in ' '.join(call)
                             for call in calls))

    def test_a_wrong_device_never_substitutes_another_card(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '3'
        report = {'phyBefore': 0, 'phyAfter': 0, 'authorizedDevice': 3}
        observed = npu.observations(report, '3')
        self.assertFalse(observed['deviceMatchesAuthorized'])
        self.assertFalse(npu.passed(observed))

    def test_a_half_conda_configuration_fails_closed_before_the_device(self):
        # The container script only activates Conda when both values are set, so a
        # half configuration would silently drop it and run the device test against
        # whatever interpreter the image provides. Both Gates must refuse it on the
        # Host first, before any docker call. The toolchain comes from the env file,
        # so each half is written there.
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        conda_sh = self.root / 'conda.sh'
        conda_sh.write_text('conda() { :; }\n')
        identity = ('TILELANG_HIVM_CONTAINER_NAME=fake-container\n'
                    'TILELANG_HIVM_CONTAINER_HOME=/home/u\n')
        for body in (f'TILELANG_HIVM_CONDA_SH={conda_sh}\n',
                     'TILELANG_HIVM_CONDA_ENV=py311\n'):
            env_file = self.root / 'half.env'
            env_file.write_text(identity + body)
            os.environ['TILELANG_HIVM_ENV_FILE'] = str(env_file)
            calls = []
            with patch('subprocess.run', side_effect=self.fake_npu(calls=calls)):
                with self.assertRaises(npu.BridgeError) as error:
                    npu.main(['--timeout', '900'])
            self.assertIn('must be provided together', str(error.exception))
            self.assertEqual(calls, [], 'no docker call may happen on a half configuration')

    def test_the_gate_never_resets_or_selects_a_physical_device(self):
        script = npu.container_script('/src', 900, '0', None, None, None)
        for forbidden in ('npu-smi', 'reset_device', 'kill -9', 'pkill'):
            self.assertNotIn(forbidden, script)
        self.assertIn('ASCEND_RT_VISIBLE_DEVICES', script)


class NpuOccupancyCase(_FakeDockerCase):
    """A card in use by another job must block the Gate before dispatch.

    The A5 cards are shared, and an idle card is not an assignment either. The
    probe therefore only removes the case where the card is demonstrably busy;
    it never selects a card, and it never reports another user's process ids.
    """

    def test_a_busy_card_blocks_before_any_docker_call(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        calls = []
        busy = [(4242, 'python3'), (4243, 'pytest')]
        with patch('subprocess.run', side_effect=self.fake_npu(calls=calls, occupancy=busy)):
            with self.assertRaises(npu.BridgeError) as error:
                npu.main(['--timeout', '900'])
        message = str(error.exception)
        self.assertIn('already in use by 2 process', message)
        # The holder identities are counted, never copied into the receipt.
        self.assertNotIn('4242', message)
        self.assertNotIn('4243', message)
        self.assertFalse([call for call in calls if call[:2] == ['docker', 'exec']])

    def test_a_card_that_became_busy_is_blocked_not_failed(self):
        # The same condition caught at the configuration check is
        # `BLOCKED_ENV`; caught at the pre-dispatch probe it must not become
        # `FAIL`, because no test started. The distinct exit code is what
        # carries that classification to the runner, so it has to be the
        # declared one and not a code a failing test run could produce.
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        with patch('subprocess.run', side_effect=self.fake_npu(occupancy=[(7, 'python')])):
            with self.assertRaises(npu.OccupancyBlocked) as error:
                npu.main(['--timeout', '900'])
        self.assertEqual(type(error.exception).EXIT, 75)
        # pytest's own statuses are 0-5 and the container script uses 3, so 75
        # cannot be a test failure that was mistaken for a blocked environment.
        self.assertNotIn(npu.OccupancyBlocked.EXIT, (0, 1, 2, 3, 4, 5))
        manifest = load_manifest(NPU / 'manifest.json')
        for recipe in manifest['recipes']:
            self.assertEqual(recipe.get('blockedExitCodes'), [npu.OccupancyBlocked.EXIT], recipe['id'])
        for recipe in load_manifest(BASE / 'manifest.json')['recipes']:
            self.assertNotIn('blockedExitCodes', recipe,
                             'the base Adapter has no blocked-environment exit code')

    def test_check_config_blocks_on_a_busy_card(self):
        # Preflight is the first place the Host learns the machine is unusable;
        # a busy card must stop there rather than after the Recipe starts.
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        calls = []
        with patch('subprocess.run', side_effect=self.fake_npu(calls=calls, occupancy=[(7, 'python')])):
            with self.assertRaises(npu.BridgeError):
                npu.check_config()
        self.assertFalse([call for call in calls if call[:2] == ['docker', 'exec']])

    def test_an_idle_card_is_probed_but_never_selected(self):
        # An idle card still needs the explicit variable: the probe answers a
        # question about the authorized device, it does not choose one.
        calls = []
        with patch('subprocess.run', side_effect=self.fake_npu(calls=calls)):
            with self.assertRaises(npu.BridgeError):
                npu.main(['--timeout', '900'])
        self.assertEqual(calls, [])

    def test_an_unreadable_driver_answer_is_not_treated_as_free(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        for probe in (dict(probe_exit=1), dict(probe_exit=255, probe_text=''),
                      dict(probe_text='some other output\n'), dict(probe_text='')):
            calls = []
            with self.subTest(probe=probe):
                with patch('subprocess.run', side_effect=self.fake_npu(calls=calls, **probe)):
                    with self.assertRaises(npu.BridgeError):
                        npu.main(['--timeout', '900'])
                self.assertFalse([call for call in calls if call[:2] == ['docker', 'exec']])

    def test_a_missing_npu_smi_cannot_dispatch(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        with patch('subprocess.run', side_effect=FileNotFoundError(2, 'No such file')):
            with self.assertRaises(npu.BridgeError) as error:
                npu.main(['--timeout', '900'])
        self.assertIn('cannot determine NPU occupancy', str(error.exception))

    def test_a_hanging_probe_cannot_dispatch(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        with patch('subprocess.run',
                   side_effect=subprocess.TimeoutExpired(['npu-smi', 'info', '-m'], 60)):
            with self.assertRaises(npu.BridgeError):
                npu.main(['--timeout', '900'])

    def test_the_physical_id_is_looked_up_in_the_mapping_table(self):
        # `npu-smi` numbers cards by NPU ID, which need not equal the physical
        # id the driver asserts. Here the authorized device 1 is NPU ID 0.
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '1'
        mapping = ('\tNPU ID   Slot ID  Chip ID  Chip Phy-ID  Chip Name\n'
                   '\t0  76  0  1  Ascend950PR\n'
                   '\t1  81  0  0  Ascend950PR\n')
        seen = []

        def run(argv, **kwargs):
            seen.append(list(argv))
            if argv[0] == npu.NPU_SMI:
                if argv[2:] == ['-m']:
                    return subprocess.CompletedProcess(argv, 0, mapping, '')
                return subprocess.CompletedProcess(argv, 0, '\tNPU ID : 0\n\tNo process in device.\n', '')
            return self.fake_npu(calls=None, device='1')(argv, **kwargs)

        with patch('subprocess.run', side_effect=run):
            self.assertEqual(npu.main(['--timeout', '900']), 0)
        probes = [call for call in seen if call[0] == npu.NPU_SMI]
        self.assertEqual(probes[0][2:], ['-m'])
        self.assertIn('-i', probes[1])
        self.assertEqual(probes[1][probes[1].index('-i') + 1], '0',
                         'the NPU ID holding physical id 1 is queried, not the literal id')
    def test_a_device_missing_from_the_mapping_table_is_refused(self):
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '5'
        with patch('subprocess.run', side_effect=self.fake_npu()):
            with self.assertRaises(npu.BridgeError) as error:
                npu.main(['--timeout', '900'])
        self.assertIn('physical id 5', str(error.exception))

    def test_the_probe_only_reads_device_state(self):
        # Only `npu-smi info` may be issued: no reset, no set, no config write.
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        calls = []
        with patch('subprocess.run', side_effect=self.fake_npu(calls=calls)):
            self.assertEqual(npu.main(['--timeout', '900']), 0)
        probes = [call for call in calls if call[0] == npu.NPU_SMI]
        self.assertTrue(probes)
        for call in probes:
            self.assertEqual(call[1], 'info', call)
            self.assertNotIn('reset', ' '.join(call))
            self.assertNotIn('-s', call)
        self.assertEqual({tuple(call[2:3]) for call in probes}, {('-m',), ('-t',)})

    def test_the_device_is_rechecked_immediately_before_dispatch(self):
        # Another job may take the card between the configuration check and the
        # Recipe. The second probe must see it and stop before `docker exec`.
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        calls = []
        state = {'holders': []}

        def run(argv, **kwargs):
            if argv[0] == npu.NPU_SMI and argv[2:3] == ['-t']:
                state['holders'] = [(99, 'python3')]
            return self.fake_npu(calls=calls, occupancy=state['holders'])(argv, **kwargs)

        with patch('subprocess.run', side_effect=run):
            with self.assertRaises(npu.BridgeError):
                npu.main(['--timeout', '900'])
        self.assertFalse([call for call in calls if call[:2] == ['docker', 'exec']])

    def test_an_unrecognised_process_table_is_unknown_occupancy_not_a_free_card(self):
        # The probe used to raise only when the `NPU ID` header was missing.
        # If the header matched but the per-process layout drifted - a driver
        # upgrade adding a space before the colon, a name containing spaces -
        # no row matched, the holder list came back empty, and a busy card was
        # reported free. Any answer this Gate cannot read in full is unknown.
        drift = ('\tNPU ID                         : 0\n'
                 '\tProcess id:4242 Process name:python3 Process memory(MB) : 256\n')
        with patch('subprocess.run', side_effect=self.fake_npu(probe_text=drift)):
            with self.assertRaises(npu.BridgeError) as error:
                npu.processes_on('0')
        self.assertIn('not understood', str(error.exception))

    def test_a_table_with_neither_a_row_nor_the_idle_marker_is_refused(self):
        # Absence of rows is not a statement that the card is idle. The driver
        # says so explicitly; without that sentence the answer is unreadable.
        silent = '\tNPU ID                         : 0\n'
        with patch('subprocess.run', side_effect=self.fake_npu(probe_text=silent)):
            with self.assertRaises(npu.BridgeError):
                npu.processes_on('0')

    def test_the_drivers_idle_marker_is_the_only_empty_table_that_is_free(self):
        idle = '\tNPU ID                         : 0\n\tNo process in device.\n'
        with patch('subprocess.run', side_effect=self.fake_npu(probe_text=idle)):
            self.assertEqual(npu.processes_on('0'), [])

    def test_a_table_claiming_both_processes_and_an_idle_card_is_refused(self):
        contradictory = ('\tNPU ID                         : 0\n'
                         '\tNo process in device.\n'
                         '\tProcess id:4242 Process name:python3           Process memory(MB):256\n')
        with patch('subprocess.run', side_effect=self.fake_npu(probe_text=contradictory)):
            with self.assertRaises(npu.BridgeError):
                npu.processes_on('0')

    def test_a_machine_file_value_cannot_execute_inside_the_container_script(self):
        # `fail "text {value}"` would expand `$(...)` in the value when the
        # container shell runs the script, which would make a value read from
        # env.sh executable code. Values must reach the script as literals.
        hostile = Path('/tmp/$(touch /tmp/PWNED)/cann.sh')
        script = npu.container_script('/src', 900, '0', hostile, None, None, None)
        self.assertNotIn('fail "CANN setup not readable', script)
        for line in script.splitlines():
            if 'CANN setup not readable' in line:
                self.assertIn("'/tmp/$(touch /tmp/PWNED)/cann.sh'", line)
        conda = npu.container_script('/src', 900, '0', None, Path('/tmp/c.sh'), 'env$(touch /tmp/PWNED)', None)
        for line in conda.splitlines():
            if 'conda activate' in line:
                self.assertNotIn('fail "', line)


class NpuIsolationCase(unittest.TestCase):
    """A task that does not select the NPU Adapter must run no device test."""

    def test_the_base_adapter_has_no_device_variable_or_command(self):
        text = (BASE / 'scripts/docker_gate.py').read_text()
        self.assertNotIn('TILELANG_HIVM_DEVICE_ID', text)
        manifest = load_manifest(BASE / 'manifest.json')
        for recipe in manifest['recipes']:
            self.assertNotIn('DEVICE_ID', ' '.join(recipe['argv']))
            self.assertNotIn('npu', ' '.join(recipe['argv']))

    def test_only_the_npu_adapter_declares_the_device_capability(self):
        self.assertEqual(load_manifest(NPU / 'manifest.json')['capabilities'],
                         ['device-regression'])
        self.assertNotIn('device-regression',
                         load_manifest(BASE / 'manifest.json')['capabilities'])

    def test_the_npu_recipe_requires_an_explicit_device_variable(self):
        script = (NPU / 'scripts/npu_gate.py').read_text()
        self.assertIn("env_value('TILELANG_HIVM_DEVICE_ID', required=True)", script)
        # No fallback value: a device is authorized explicitly or the Gate refuses.
        call = script.split("TILELANG_HIVM_DEVICE_ID', required=True)")[1].split('\n')[0]
        self.assertNotIn('default=', call)


class NpuRunsAsRootCase(_FakeDockerCase):
    """The NPU device is exclusively held per user namespace by the driver.

    `uda_occupy_dev_by_ns` rejects a non-root opener with EBUSY once any other
    namespace holds the card, so an ordinary container user cannot reliably
    reach the device. The Gate therefore runs its container side as root.
    """

    def test_the_device_gate_executes_as_root(self):
        script = (NPU / 'scripts/npu_gate.py').read_text()
        self.assertIn("'-u', 'root'", script)

    def test_the_device_gate_does_not_execute_as_a_container_user(self):
        script = (NPU / 'scripts/npu_gate.py').read_text()
        self.assertNotIn("'-u', user", script)
        # The variable may be named in prose that explains why it is ignored, but
        # the Gate must never read it to build the exec prefix.
        self.assertNotIn("env_value('TILELANG_HIVM_CONTAINER_USER'", script)

    def test_the_build_gate_keeps_the_container_user(self):
        # Only the device Gate changes: the build Gate still proves that an
        # ordinary user can build, which is what the container setup promises.
        script = (BASE / 'scripts/docker_gate.py').read_text()
        self.assertIn("'-u', user", script)

    def test_the_root_prefix_still_pins_home_and_workdir(self):
        prefix = npu.docker_exec_prefix('c1', '/home/u', '/src')
        self.assertEqual(prefix[:3], ['docker', 'exec', '-u'])
        self.assertEqual(prefix[3], 'root')
        self.assertIn('HOME=/home/u', prefix)
        self.assertIn('/src', prefix)

    def test_a_container_user_variable_is_no_longer_required(self):
        # It stays accepted so an existing environment keeps working, but the
        # Gate must not refuse to start when it is unset.
        os.environ['TILELANG_HIVM_DEVICE_ID'] = '0'
        os.environ.pop('TILELANG_HIVM_CONTAINER_USER', None)
        calls = []
        with patch('subprocess.run', side_effect=self.fake_npu(calls=calls)):
            self.assertEqual(npu.main(['--timeout', '900']), 0)
        self.assertTrue(self.docker_exec_scripts(calls))

    def test_the_readme_states_why_root_is_required(self):
        text = (NPU / 'README.md').read_text()
        self.assertIn('root', text)
        self.assertIn('uda_occupy_dev_by_ns', text)

    def test_a_conda_environment_outside_root_home_stays_reachable(self):
        # root's HOME is not where a container user keeps its conda envs, so the
        # container script must point conda at the directory the machine file
        # names. Without it a root run reports `Could not find conda environment`.
        script = npu.container_script('/src', 900, '0', None, Path('/data/conda.sh'), 'myenv',
                                      conda_envs_path=Path('/home/someone/.conda/envs'))
        self.assertIn('CONDA_ENVS_PATH', script)
        self.assertIn('/home/someone/.conda/envs', script)

    def test_the_conda_envs_path_is_not_guessed_from_home(self):
        # A guessed `$HOME/.conda/envs` used to stand in for the configured
        # directory, which meant the Gate searched a place the operator had not
        # written down. Only the machine file may name it.
        script = npu.container_script('/src', 900, '0', None, Path('/data/conda.sh'), 'myenv')
        self.assertNotIn('CONDA_ENVS_PATH', script)

    def test_no_conda_envs_path_is_set_without_conda(self):
        script = npu.container_script('/src', 900, '0', None, None, None)
        self.assertNotIn('CONDA_ENVS_PATH', script)

    def test_the_conda_envs_path_preserves_an_existing_value(self):
        # A machine that already exported CONDA_ENVS_PATH must keep its entries,
        # otherwise the Gate silently hides environments that used to be found.
        script = npu.container_script('/src', 900, '0', None, Path('/data/conda.sh'), 'myenv',
                                      conda_envs_path=Path('/home/someone/.conda/envs'))
        self.assertIn('${CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}', script)




class SetupPythonEnvCase(unittest.TestCase):
    """scripts/setup-python-env.sh provisions the interpreter the Gates run in.

    Two properties matter and both are cheap to check without conda, curl or a
    GPU: it refuses a torch wheel that is not a `+cpu` build (the CUDA build is
    several gigabytes of unusable wheels on an Ascend host), and it never guesses
    a wheel directory. The provisioning itself is verified on the machine, not
    here.
    """

    SCRIPT = BASE / 'scripts/setup-python-env.sh'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wheels = self.root / 'wheels'
        self.wheels.mkdir()

    def wheel(self, name):
        path = self.wheels / name
        path.write_bytes(b'fake wheel\n')
        return path

    def run_script(self, **env):
        # An empty machine file by default. Without this the script would read the
        # checkout's own `env.sh`, which names a real wheel directory and a real
        # Conda, so a unit test would provision a real environment and download
        # Miniconda. A test that needs a value writes its own file.
        empty = self.root / 'unit-test-env.sh'
        empty.write_text('# no values: the test supplies what it needs\n')
        base = {'HOME': str(self.root), 'PATH': '/usr/bin:/bin',
                'TILELANG_HIVM_ENV_FILE': str(empty)}
        base.update(env)
        return subprocess.run([shutil.which('bash'), str(self.SCRIPT)],
                              capture_output=True, text=True, env=base,
                              cwd=str(self.root))

    def test_a_missing_wheel_directory_is_refused_with_a_named_error(self):
        done = self.run_script()
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('TILELANG_HIVM_WHEEL_DIR is required', done.stderr)

    def test_a_nonexistent_wheel_directory_is_refused(self):
        done = self.run_script(TILELANG_HIVM_WHEEL_DIR=str(self.root / 'nope'))
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('no such wheel directory', done.stderr)

    def test_a_cuda_torch_wheel_is_refused(self):
        # The plain `torch-2.10.0` wheel from PyPI is the CUDA build; only a
        # `+cpu` wheel is acceptable here.
        self.wheel('torch-2.10.0-cp311-cp311-manylinux_2_28_x86_64.whl')
        self.wheel('torch_npu-2.10.0-cp311-cp311-manylinux_2_28_x86_64.whl')
        done = self.run_script(TILELANG_HIVM_WHEEL_DIR=str(self.wheels))
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('not a +cpu build', done.stderr)

    def test_a_missing_torch_npu_wheel_is_refused(self):
        self.wheel('torch-2.10.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl')
        done = self.run_script(TILELANG_HIVM_WHEEL_DIR=str(self.wheels))
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('no torch_npu-*.whl', done.stderr)

    def test_a_missing_conda_installation_is_refused(self):
        # The installation is reused, never created, so a file that names none
        # is a configuration error rather than an invitation to install a
        # second Conda that the file does not describe.
        self.wheel('torch-2.10.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl')
        self.wheel('torch_npu-2.10.0-cp311-cp311-manylinux_2_28_x86_64.whl')
        done = self.run_script(TILELANG_HIVM_WHEEL_DIR=str(self.wheels))
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('TILELANG_HIVM_CONDA_SH is required', done.stderr)

    def test_a_relative_conda_envs_path_is_refused(self):
        done = self.run_script(TILELANG_HIVM_WHEEL_DIR=str(self.wheels),
                               TILELANG_HIVM_CONDA_ENVS_PATH='conda/envs')
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('must be an absolute path', done.stderr)

    def test_an_environment_name_that_is_a_path_is_refused(self):
        done = self.run_script(TILELANG_HIVM_WHEEL_DIR=str(self.wheels),
                               TILELANG_HIVM_CONDA_ENV='../escape')
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('must be one plain name', done.stderr)

    def write_env_file(self, **values):
        path = self.root / 'env.sh'
        path.write_text(''.join(f'{key}={value}\n' for key, value in values.items()))
        return path

    def test_the_machine_file_is_the_primary_configuration(self):
        # The operator edits one file, the same one both Gates read. A wheel
        # directory named there is used without any export.
        self.wheel('torch-2.10.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl')
        self.write_env_file(TILELANG_HIVM_WHEEL_DIR=str(self.wheels))
        done = self.run_script(TILELANG_HIVM_ENV_FILE=str(self.root / 'env.sh'))
        # The run stops at the interpreter, before anything is installed, but it
        # must not stop for a missing wheel directory.
        self.assertNotIn('TILELANG_HIVM_WHEEL_DIR is required', done.stderr)

    def test_an_export_overrides_the_machine_file(self):
        # The override exists for one run and for tests. It is not the primary
        # mechanism, and the script says so when it applies one, so a run never
        # silently provisions something other than the file describes.
        self.wheel('torch-2.10.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl')
        self.write_env_file(TILELANG_HIVM_WHEEL_DIR=str(self.root / 'from-file'))
        done = self.run_script(TILELANG_HIVM_ENV_FILE=str(self.root / 'env.sh'),
                               TILELANG_HIVM_WHEEL_DIR=str(self.wheels))
        self.assertIn('overrides', done.stderr)
        self.assertNotIn('TILELANG_HIVM_WHEEL_DIR is required', done.stderr)

    def test_a_conda_installation_is_reused_instead_of_installed(self):
        # A machine that already has Conda must not receive a second one. The
        # value that proves it exists is `TILELANG_HIVM_CONDA_SH`, the same one
        # the Gates source, so the file describes the installation that is
        # actually used.
        conda_sh = self.root / 'conda' / 'etc' / 'profile.d' / 'conda.sh'
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text('# stand-in\n')
        self.wheel('torch-2.10.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl')
        self.write_env_file(TILELANG_HIVM_WHEEL_DIR=str(self.wheels),
                            TILELANG_HIVM_CONDA_SH=str(conda_sh),
                            TILELANG_HIVM_CONDA_ENV='mk')
        done = self.run_script(TILELANG_HIVM_ENV_FILE=str(self.root / 'env.sh'))
        self.assertNotIn('installing Miniconda', done.stderr)

    def test_the_conda_prefix_is_derived_from_conda_sh(self):
        # `conda.sh` names an installation, not a prefix. `conda env list` shows
        # environments under the prefix, and a wrong derivation once produced
        # `no conda executable at /data/bin/conda` for a real installation at
        # `/data/miniconda3`. The prefix is two levels above `conda.sh`'s
        # directory, and the derivation is proved by finding `bin/conda` there.
        prefix = self.root / 'miniconda3'
        (prefix / 'bin').mkdir(parents=True)
        conda = prefix / 'bin' / 'conda'
        conda.write_text('#!/bin/sh\nexit 0\n')
        conda.chmod(0o755)
        (prefix / 'etc' / 'profile.d').mkdir(parents=True)
        conda_sh = prefix / 'etc' / 'profile.d' / 'conda.sh'
        conda_sh.write_text('# stand-in\n')
        self.write_env_file(TILELANG_HIVM_WHEEL_DIR=str(self.wheels),
                            TILELANG_HIVM_CONDA_SH=str(conda_sh),
                            TILELANG_HIVM_CONDA_ENV='mk')
        done = self.run_script(TILELANG_HIVM_ENV_FILE=str(self.root / 'env.sh'))
        self.assertNotIn('cannot locate the Conda prefix', done.stderr)
        self.assertNotIn(f'no conda executable at {prefix.parent}/bin/conda', done.stderr)

    def test_an_environment_directory_list_is_read(self):
        # The Conda variable is a colon-separated list. A machine that already
        # exports it from a shell profile writes a list, and that file must not
        # become unreadable just because it is more than one directory.
        first = self.root / 'a' / 'envs'
        second = self.root / 'b' / 'envs'
        (second / 'mk' / 'bin').mkdir(parents=True)
        (second / 'mk' / 'bin' / 'python3').write_text('#!/bin/sh\nexit 0\n')
        self.write_env_file(TILELANG_HIVM_WHEEL_DIR=str(self.wheels),
                            TILELANG_HIVM_CONDA_ENVS_PATH=f'{first}:{second}')
        done = self.run_script(TILELANG_HIVM_ENV_FILE=str(self.root / 'env.sh'))
        self.assertNotIn('must be an absolute path', done.stderr)

    def test_the_environment_directory_is_the_configured_one(self):
        # A Conda installation looks only under its own prefix. When the file
        # names `~/.conda/envs`, the environment is created and reported there,
        # not under the installation prefix, and the printed values point at it.
        envs = self.root / '.conda' / 'envs'
        self.write_env_file(TILELANG_HIVM_WHEEL_DIR=str(self.wheels),
                            TILELANG_HIVM_CONDA_ENVS_PATH=str(envs))
        done = self.run_script(TILELANG_HIVM_ENV_FILE=str(self.root / 'env.sh'))
        self.assertIn(str(envs), done.stderr + done.stdout)

    def test_the_script_never_runs_from_a_gate(self):
        # Provisioning needs the network and minutes of downloads. A Gate must
        # stay offline and fast, so no bridge may execute this script. A bridge
        # may name it in a comment, because that is how a reader learns which
        # file owns the provisioning values; what must not appear is a call.
        import ast
        for bridge in (BASE / 'scripts/docker_gate.py', NPU / 'scripts/npu_gate.py'):
            tree = ast.parse(bridge.read_text())
            # Drop every docstring, so prose that names the script is allowed and
            # only executable text is searched.
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    body = node.body
                    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                            and isinstance(body[0].value.value, str):
                        docstrings.add(id(body[0].value))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and id(node) not in docstrings:
                    self.assertNotIn('setup-python-env', node.value, bridge.name)

    def test_the_default_environment_name_is_named_in_the_script(self):
        # The default environment name is the one the examples use. There is no
        # default installation prefix, because the script installs nothing.
        text = self.SCRIPT.read_text()
        self.assertIn('conda_env=mk', text)
        self.assertNotIn('CONDA_ROOT', text)
        self.assertNotIn('MINICONDA_INSTALLER', text)
        self.assertNotIn('repo.anaconda.com', text)

    def test_the_editable_install_returns_build_tree_ownership(self):
        # Running this script as root is normal, because installing into `/opt`
        # needs it. Configure then writes into `<source>/build`, and a build tree
        # left owned by root makes the next build Gate, which runs as the
        # container user, fail with `Error writing to build log: Permission
        # denied`. The owner must be recorded before the install and restored
        # after it.
        text = self.SCRIPT.read_text()
        self.assertIn("stat -c '%u:%g'", text)
        self.assertIn('chown -R "$owner"', text)
        record = text.index("stat -c '%u:%g'")
        install = text.index('--no-build-isolation -e .')
        restore = text.index('chown -R "$owner"')
        self.assertLess(record, install, 'the owner must be read before the install')
        self.assertLess(install, restore, 'the owner must be restored after the install')

    def test_the_editable_install_forces_the_ascend_backend(self):
        # Autodetection picks CUDA on a machine with no CUDA device, which builds
        # a library that fails only later, inside a kernel.
        text = self.SCRIPT.read_text()
        self.assertIn('USE_ASCEND=ON;USE_CUDA=OFF', text)


if __name__ == '__main__':
    unittest.main()
