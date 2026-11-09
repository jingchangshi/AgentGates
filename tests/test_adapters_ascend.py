"""Offline checks for the ascendnpu-ir Adapter: planning, mapping and failure paths.

Every Docker interaction is faked, so these tests need neither a container nor a
compiler build, and the Host route is exercised against a temporary PATH of fake
tools, so it needs no real cmake, ninja or clang either. They cover the
deterministic rules the Adapter is responsible for: manifest/hash validity, plan
composition with compiler-engineering, Gate dependency order, authorization and
install/enablement refusal, build-subdir traversal refusal, mount-mapping
rejection, declared-path mismatch, the Host toolchain floors, and the requirement
that a non-zero Ninja exit can never be reported as PASS.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / 'agent_gates/adapters'
ADAPTER = ADAPTERS / 'ascendnpu-ir'
BRIDGE = ADAPTER / 'scripts/docker_gate.py'
HOST_BRIDGE = ADAPTER / 'scripts/host_gate.py'
FULL_BUILD = ADAPTER / 'scripts/full-build.sh'

sys.path.insert(0, str(ROOT))
from agent_gates.manifest import Manifest, ManifestError as ContractError


def load_manifest(path):
    manifest = Manifest(path)
    return dict(manifest.raw, manifestHash=manifest.manifest_hash)


def load_bridge(path=BRIDGE, name='ascendnpu_bridge'):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = load_bridge()
# The no-Docker route. Loading it also proves it can find its pinned sibling by
# its own resolved path: the module refuses to import without that file.
host_bridge = load_bridge(HOST_BRIDGE, 'ascendnpu_host_bridge')


class ManifestCase(unittest.TestCase):
    def test_manifest_is_valid_and_hash_pinned(self):
        manifest = load_manifest(ADAPTER / 'manifest.json')
        self.assertEqual(manifest['id'], 'ascendnpu-ir')
        self.assertEqual(manifest['permissions'], {'sourceWrite': False, 'hostExecution': True})
        self.assertEqual(manifest['runtimeCompatibility'], {'engineeringCore': '>=2.0.0'})
        # Two routes, two recipes each. Both are the same Adapter on purpose: the
        # engine derives the build lease from the Adapter ID, so a second ID would
        # let a Docker run and a Host run write one build tree under two locks.
        self.assertEqual([r['id'] for r in manifest['recipes']],
                         ['ascendnpu-ir.incremental-build', 'ascendnpu-ir.regression',
                          'ascendnpu-ir.host-build', 'ascendnpu-ir.host-regression'])
        self.assertEqual({g['id'] for g in manifest['gates']},
                         {'ascend-build', 'ascend-regression',
                          'ascend-build-host', 'ascend-regression-host'})
        self.assertEqual(len(manifest['files']), 4)
        self.assertEqual(set(manifest['files']),
                         {'scripts/docker_gate.py', 'scripts/init-toolchain.sh',
                          'scripts/full-build.sh', 'scripts/host_gate.py'})

    def test_the_docker_route_is_byte_identical_after_the_host_route_exists(self):
        # The non-regression guard: the working Docker route was reviewed and
        # measured, so adding a second route must not touch a single byte of what
        # it runs. Asserted as whole records rather than field by field, because a
        # field-by-field check is what lets a new key slip in unnoticed.
        manifest = load_manifest(ADAPTER / 'manifest.json')
        recipes = {item['id']: item for item in manifest['recipes']}
        self.assertEqual(recipes['ascendnpu-ir.incremental-build'], {
            'id': 'ascendnpu-ir.incremental-build',
            'cwd': 'source',
            'timeout': 23400,
            'argv': ['python3', '-B', '{pack}/scripts/docker_gate.py', '--timeout', '840',
                     '--full-build-timeout', '21600', '--allow-full-build', 'bishengir-opt',
                     'bishengir-compile', 'install-bishengir-publish-products'],
            'configCheck': ['python3', '-B', '{pack}/scripts/docker_gate.py', '--check-config'],
        })
        self.assertEqual(recipes['ascendnpu-ir.regression'], {
            'id': 'ascendnpu-ir.regression',
            'cwd': 'source',
            'timeout': 1200,
            'argv': ['python3', '-B', '{pack}/scripts/docker_gate.py', '--timeout', '1140',
                     'check-bishengir'],
            'configCheck': ['python3', '-B', '{pack}/scripts/docker_gate.py', '--check-config'],
        })
        gates = {item['id']: item for item in manifest['gates']}
        self.assertEqual(gates['ascend-build'], {
            'id': 'ascend-build',
            'recipeId': 'ascendnpu-ir.incremental-build',
            'gateType': 'deterministic-test',
            'observationChecks': {'exitCodeZero': True},
        })
        self.assertEqual(gates['ascend-regression'], {
            'id': 'ascend-regression',
            'recipeId': 'ascendnpu-ir.regression',
            'gateType': 'deterministic-test',
            'observationChecks': {'exitCodeZero': True},
            'dependsOn': ['ascend-build'],
        })
        # No host-only key leaked into the Docker records: the two routes share an
        # Adapter, so a stray "device": "none" here would silently stop the Docker
        # gate from taking the card it needs.
        for gate in ('ascend-build', 'ascend-regression'):
            self.assertNotIn('device', gates[gate])
            self.assertNotIn('deviceWindow', gates[gate])

    def test_the_host_route_mirrors_the_docker_bounds_with_its_own_preflight(self):
        # Same measured numbers - a Host build is the same build - but its own
        # pinned script and its own configuration check, because the preflight a
        # Host needs (cmake/ninja/clang on PATH) is not the preflight a container
        # needs (container identity).
        manifest = load_manifest(ADAPTER / 'manifest.json')
        recipes = {item['id']: item for item in manifest['recipes']}
        docker_build = recipes['ascendnpu-ir.incremental-build']
        docker_regression = recipes['ascendnpu-ir.regression']
        for host_id, docker in (('ascendnpu-ir.host-build', docker_build),
                                ('ascendnpu-ir.host-regression', docker_regression)):
            host = recipes[host_id]
            self.assertEqual(host['cwd'], 'source')
            self.assertEqual(host['argv'][:3],
                             ['python3', '-B', '{pack}/scripts/host_gate.py'])
            # The bound is the same job, so the same numbers. Only the script path
            # and the preflight differ.
            self.assertEqual(host['timeout'], docker['timeout'])
            self.assertEqual(host['argv'][3:], docker['argv'][3:])
            self.assertEqual(host['configCheck'][:3],
                             ['python3', '-B', '{pack}/scripts/host_gate.py'])
            self.assertNotIn('--check-config', host['configCheck'])
        self.assertEqual(recipes['ascendnpu-ir.host-build']['configCheck'],
                         ['python3', '-B', '{pack}/scripts/host_gate.py',
                          '--check-config-build'])
        self.assertEqual(recipes['ascendnpu-ir.host-regression']['configCheck'],
                         ['python3', '-B', '{pack}/scripts/host_gate.py',
                          '--check-config-regression'])

    def test_the_two_routes_are_independent_in_dependency_order(self):
        # A Host-only operator must be able to run both host gates without a
        # container, and a Docker operator must not be made to run a Host build.
        # Dependency closure is what a run walks, so a cross-route edge here
        # would make each route demand the other one's machine.
        contract = Manifest(ADAPTER / 'manifest.json')
        self.assertEqual(contract.order(['ascend-regression-host']),
                         ['ascend-build-host', 'ascend-regression-host'])
        self.assertEqual(contract.order(['ascend-regression']),
                         ['ascend-build', 'ascend-regression'])
        # Each host gate names only its own host dependency, and each docker gate
        # only its own docker one.
        self.assertEqual(contract.gates['ascend-regression-host'].get('dependsOn'),
                         ['ascend-build-host'])
        self.assertEqual(contract.gates['ascend-regression'].get('dependsOn'),
                         ['ascend-build'])
        self.assertNotIn('dependsOn', contract.gates['ascend-build-host'])
        self.assertNotIn('dependsOn', contract.gates['ascend-build'])

    def test_no_absolute_path_container_name_or_branch_is_hard_coded(self):
        # The project may be named; machine- and checkout-specific values may not.
        # All four pinned files are scanned: the two shell scripts are read
        # inside the container, so they must stay as machine-neutral as the Gates.
        for text in (path.read_text()
                     for path in (BRIDGE, HOST_BRIDGE, bridge.INIT_SCRIPT, FULL_BUILD)):
            for forbidden in ('/home/shijingchang', 's00653124', 'AscendNPU-IR-1',
                              'dev-compiletime-skippass', 'bisheng-async', 'triton-py311',
                              '/opt/cmake', '/opt/miniconda3', 'CANN/cann-'):
                self.assertNotIn(forbidden, text)
            self.assertNotIn('shijingchang', text)

    def test_recipes_declare_the_measured_targets_timeouts_and_no_goal_text(self):
        manifest = load_manifest(ADAPTER / 'manifest.json')
        recipes = {item['id']: item for item in manifest['recipes']}
        build = recipes['ascendnpu-ir.incremental-build']
        regression = recipes['ascendnpu-ir.regression']
        for recipe in (build, regression):
            self.assertEqual(recipe['cwd'], 'source')
            self.assertEqual(recipe['argv'][:3], ['python3', '-B', '{pack}/scripts/docker_gate.py'])
            self.assertEqual(recipe['argv'][3], '--timeout')
        # Outer Harness bound must exceed the container-side ceiling. It is the
        # full-build ceiling that sets it: the same Gate takes either path, and a
        # bound smaller than a cold full build would make the from-zero branch
        # unreachable rather than supported.
        self.assertEqual(build['timeout'], 23400)
        self.assertEqual(int(build['argv'][4]) < build['timeout'], True)
        self.assertEqual(build['argv'][5], '--full-build-timeout')
        self.assertLess(int(build['argv'][6]) + bridge.KILL_AFTER_SECONDS, build['timeout'])
        # Measured by a real run: 1142 tests across 256 lit workers in 2.7s wall
        # clock (1009 passed, 0 failed). The original bound is ample, so a larger
        # timeout would only delay noticing a genuine hang.
        self.assertEqual(regression['timeout'], 1200)
        self.assertEqual(int(regression['argv'][4]) < regression['timeout'], True)
        self.assertEqual(build['argv'][5:7], ['--full-build-timeout', '21600'])
        self.assertEqual(build['argv'][7], '--allow-full-build')
        self.assertEqual(build['argv'][8:], ['bishengir-opt', 'bishengir-compile',
                                             'install-bishengir-publish-products'])
        self.assertEqual(regression['argv'][5:], ['check-bishengir'])
        # The Adapter is a fixed bridge: no recipe may interpolate goal text.
        for recipe in (build, regression):
            for item in recipe['argv']:
                self.assertNotIn('{goal', item)
                self.assertNotIn('$', item)
                self.assertNotIn(';', item)

    def test_timeout_layer_nesting_and_grace_period(self):
        manifest = load_manifest(ADAPTER / 'manifest.json')
        recipes = {item['id']: item for item in manifest['recipes']}
        self.assertEqual(bridge.KILL_AFTER_SECONDS, 15)
        # Both container-side bounds of the build recipe nest inside its outer
        # bound, and the regression recipe keeps its measured pair.
        build = recipes['ascendnpu-ir.incremental-build']
        regression = recipes['ascendnpu-ir.regression']
        for inner in (int(build['argv'][4]), int(build['argv'][6])):
            self.assertLess(inner + bridge.KILL_AFTER_SECONDS, build['timeout'])
        self.assertEqual(regression['timeout'], 1200)
        self.assertLess(int(regression['argv'][4]) + bridge.KILL_AFTER_SECONDS,
                        regression['timeout'])
        for recipe in (build, regression):
            self.assertEqual(recipe['argv'][3], '--timeout')

    def test_the_worker_backstop_outlives_every_installed_recipe(self):
        # The unit's RuntimeMaxSec is derived, not guessed: a backstop shorter
        # than a recipe's own bound would kill work the manifest declares
        # legitimate, which is exactly what would happen to a from-zero build
        # under the old fixed 7200 s.
        from agent_gates import systemd_units
        from agent_gates.registry import Registry
        backstop = systemd_units.backstop_seconds(Registry())
        longest = max(float(recipe['timeout'])
                      for manifest in Registry().list()
                      for recipe in manifest.recipes.values())
        self.assertGreater(backstop, longest)
        self.assertEqual(backstop, int(longest) + systemd_units.BACKSTOP_MARGIN_SECONDS)

    def test_container_script_pins_the_short_kill_after(self):
        script = bridge.container_script('/src', 'build', '64', 840, None, None, None)
        self.assertIn('--kill-after=15', script)
        self.assertIn('exec timeout --signal=TERM --kill-after=15 840 ', script)
        self.assertNotIn('--kill-after=60', script)
        self.assertIn('set -o pipefail', script)

    def test_container_script_preserves_an_inherited_path(self):
        # The PATH docker exec provides must survive initialization; only an
        # absent PATH gets the system fallback, and cmake is never pinned. Both
        # rules now live in the committed init script the Gate inlines.
        script = bridge.container_script('/src', 'build', '64', 840, None, None, None)
        self.assertIn('export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"',
                      script)
        self.assertNotIn('export PATH=/usr/local/sbin', script)

    def test_tools_are_probed_from_the_effective_path(self):
        script = bridge.container_script('/src', 'build', '64', 840, None, None, None)
        self.assertIn('command -v cmake >/dev/null', script)
        self.assertIn('command -v ninja >/dev/null', script)
        # Order: the checks come after CANN/Conda, and before the Ninja exec.
        self.assertLess(script.index('command -v cmake'), script.index('exec timeout --signal=TERM'))

    def test_environment_initialization_is_the_committed_sequence_called_with_the_values(self):
        # The Gate ships the committed init script's text and calls it with the
        # three values it validated on the Host. Which of them are set is decided
        # here; the order they are applied in is proven by ScriptExecutionCase,
        # which runs the real script with real Bash.
        script = bridge.container_script('/src', 'build', '64', 840,
                                         Path('/cann/set_env.sh'), Path('/conda/etc/profile.d/conda.sh'),
                                         'env311')
        call = 'init_toolchain /cann/set_env.sh /conda/etc/profile.d/conda.sh env311 || fail'
        self.assertIn(call, script)
        self.assertLess(script.index('init_toolchain()'), script.index(call))
        self.assertLess(script.index(call), script.index('exec timeout --signal=TERM'))

    def test_conda_activation_must_not_be_silently_skipped_when_configured(self):
        script = bridge.container_script('/src', 'build', '64', 840,
                                         None, Path('/conda/etc/profile.d/conda.sh'), 'env311')
        # `-` marks the unset CANN value, and both Conda values are forwarded, so
        # the script cannot initialize nothing while the file wants a Conda env.
        self.assertIn('init_toolchain - /conda/etc/profile.d/conda.sh env311 || fail', script)

    def test_cann_and_conda_source_failures_are_explicit(self):
        script = bridge.container_script('/src', 'build', '64', 840,
                                         Path('/cann/set_env.sh'), Path('/conda.sh'), 'env311')
        # The sequence is the committed file's own text, inlined verbatim, so the
        # checks a Gate runs are the checks a human sources. There is one copy.
        committed = bridge.INIT_SCRIPT.read_text(encoding='utf-8').rstrip('\n')
        self.assertIn(committed, script)
        self.assertIn('|| fail "toolchain initialization failed"', script)
        self.assertNotIn('ASCENDNPU_CMAKE_BIN', script)

    def test_explicit_timeout_wins_over_the_environment(self):
        os.environ['ASCENDNPU_GATE_TIMEOUT'] = '5'
        timeout, full, targets, allow_full = bridge.timeout_seconds(
            ['--timeout', '840', 'check-bishengir'])
        self.assertEqual(timeout, 840)
        self.assertEqual(targets, ['check-bishengir'])
        # A full build is opt-in per recipe; the default is "existing tree only".
        self.assertFalse(allow_full)
        # The from-zero bound is separate: one number cannot serve an incremental
        # Ninja run and a cold full build.
        self.assertEqual(full, bridge.FULL_BUILD_TIMEOUT)
        # A bare invocation is the only path that reads the environment.
        self.assertEqual(bridge.timeout_seconds(['check-bishengir'])[0], 5)
        os.environ.pop('ASCENDNPU_GATE_TIMEOUT', None)
        self.assertEqual(bridge.timeout_seconds(['check-bishengir'])[0], 1500)

    def test_the_full_build_bound_is_explicit_and_separately_bounded(self):
        # The manifest passes both bounds. A full build cannot be expressed by the
        # incremental one, so it is a distinct flag with its own value, and the
        # environment can only supply it when the caller passes nothing.
        os.environ['ASCENDNPU_FULL_BUILD_TIMEOUT'] = '77'
        with self.assertRaises(bridge.BridgeError):
            bridge.timeout_seconds(['--full-build-timeout', '0', 'bishengir-opt'])
        with self.assertRaises(bridge.BridgeError):
            bridge.timeout_seconds(['--timeout', '840', '--timeout', '900', 'bishengir-opt'])
        self.assertEqual(bridge.timeout_seconds(['bishengir-opt'])[1], 77)
        os.environ.pop('ASCENDNPU_FULL_BUILD_TIMEOUT', None)
        _, full, _, _ = bridge.timeout_seconds(['--timeout', '840', 'bishengir-opt'])
        self.assertEqual(full, bridge.FULL_BUILD_TIMEOUT)

    def test_only_the_build_recipe_may_build_from_an_empty_tree(self):
        # The regression gate is a test of an existing tree. If it could trigger a
        # full build, an unbuilt checkout would report PASS for a command naming
        # targets that never ran.
        manifest = load_manifest(ADAPTER / 'manifest.json')
        recipes = {item['id']: item for item in manifest['recipes']}
        self.assertIn('--allow-full-build', recipes['ascendnpu-ir.incremental-build']['argv'])
        self.assertNotIn('--allow-full-build', recipes['ascendnpu-ir.regression']['argv'])
        self.assertFalse(bridge.timeout_seconds(['--timeout', '1140', 'check-bishengir'])[3])
        # The same rule on the Host route: exactly ONE recipe carries the flag,
        # and it is the build one. A host regression gate that could build from
        # scratch would pass on a tree where check-bishengir never ran.
        carrying = {recipe['id'] for recipe in manifest['recipes']
                    if '--allow-full-build' in recipe['argv']}
        self.assertEqual(carrying, {'ascendnpu-ir.incremental-build',
                                    'ascendnpu-ir.host-build'})
        # Stated per route as well: within the Host route exactly one recipe
        # carries it, so a host regression gate can never build from scratch.
        host_carrying = {recipe['id'] for recipe in manifest['recipes']
                         if recipe['argv'][2].endswith('/host_gate.py')
                         and '--allow-full-build' in recipe['argv']}
        self.assertEqual(host_carrying, {'ascendnpu-ir.host-build'})
        self.assertFalse(host_bridge.timeout_seconds(
            ['--timeout', '1140', 'check-bishengir'])[3])

    def test_the_regression_script_refuses_to_build_from_scratch(self):
        script = bridge.container_script('/src', 'build', '4', 840, None, None, None)
        self.assertIn('runs an existing build tree and never builds from scratch', script)
        self.assertNotIn('full_build /src', script)
        # The build gate's own script does carry the branch.
        allowed = bridge.container_script('/src', 'build', '4', 840, None, None, None,
                                          allow_full_build=True)
        self.assertIn('full_build /src', allowed)
        self.assertNotIn('never builds from scratch', allowed)

    def test_manifest_recipes_never_rely_on_the_environment_default(self):
        manifest = load_manifest(ADAPTER / 'manifest.json')
        for recipe in manifest['recipes']:
            with patch.dict(os.environ, {'ASCENDNPU_GATE_TIMEOUT': '1'}):
                timeout = bridge.timeout_seconds(recipe['argv'][3:])[0]
            self.assertEqual(timeout, int(recipe['argv'][4]), recipe['id'])
            self.assertNotEqual(timeout, 1, recipe['id'])

    def test_drifted_script_digest_is_refused(self):
        # All four pinned files are covered: a drifted Gate or a drifted shell
        # script each break the manifest contract, because a Pack that pins a
        # file that no longer matches its own bytes cannot be trusted.
        for name in ('scripts/docker_gate.py', 'scripts/init-toolchain.sh',
                     'scripts/full-build.sh', 'scripts/host_gate.py'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                copy = Path(tmp) / 'ascendnpu-ir'
                shutil.copytree(ADAPTER, copy)
                (copy / name).write_text('# drifted\n')
                with self.assertRaises(ContractError):
                    load_manifest(copy / 'manifest.json')




class EnvironmentCase(unittest.TestCase):
    """Machine values come from an explicit temporary file, never the shell.

    Every case names its own `ASCENDNPU_ENV_FILE`, so the outcome cannot depend
    on whether the machine running the suite happens to keep a real, git-ignored
    `env.sh` beside the packaged adapter.
    """

    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.machine = self.root / 'env.sh'
        for name in list(os.environ):
            if name.startswith('ASCENDNPU_'):
                os.environ.pop(name)

    def use_machine_file(self, **values):
        text = ''.join(f'{key}={value}' + chr(10) for key, value in values.items())
        self.machine.write_text(text)
        os.environ['ASCENDNPU_ENV_FILE'] = str(self.machine)
        return self.machine

    def test_build_subdir_traversal_is_refused(self):
        for value in ('..', '../build', 'a/b', '/build', '.', 'build/..'):
            self.use_machine_file(ASCENDNPU_BUILD_SUBDIR=value)
            with self.assertRaises(bridge.BridgeError, msg=repr(value)):
                bridge.build_subdir()
        # An empty key is a typo, not "use the default".
        self.use_machine_file(ASCENDNPU_BUILD_SUBDIR='')
        with self.assertRaises(bridge.BridgeError):
            bridge.build_subdir()

    def test_build_subdir_default_and_plain_names(self):
        self.use_machine_file(ASCENDNPU_CANN_SETUP='')
        self.assertEqual(bridge.build_subdir(), 'build')
        self.use_machine_file(ASCENDNPU_BUILD_SUBDIR='build-release')
        self.assertEqual(bridge.build_subdir(), 'build-release')

    def test_jobs_defaults_to_container_nproc(self):
        self.use_machine_file(ASCENDNPU_CANN_SETUP='')
        self.assertEqual(bridge.ninja_jobs(), '$(nproc)')
        self.use_machine_file(ASCENDNPU_NINJA_JOBS='0')
        with self.assertRaises(bridge.BridgeError):
            bridge.ninja_jobs()
        self.use_machine_file(ASCENDNPU_NINJA_JOBS='64')
        self.assertEqual(bridge.ninja_jobs(), '64')

    def test_a_non_empty_export_is_never_a_configuration_source(self):
        # The user's requirement: only `env.sh` configures a machine. An export
        # the file does not define looks like configuration but would have no
        # effect, so it is refused rather than obeyed or silently dropped.
        self.use_machine_file(ASCENDNPU_CANN_SETUP='')
        os.environ['ASCENDNPU_BUILD_SUBDIR'] = 'build-fast'
        with self.assertRaises(bridge.BridgeError) as error:
            bridge.build_subdir()
        self.assertIn('read only from', str(error.exception))

    def test_a_contradicting_export_is_refused(self):
        self.use_machine_file(ASCENDNPU_BUILD_SUBDIR='build')
        os.environ['ASCENDNPU_BUILD_SUBDIR'] = 'build-fast'
        with self.assertRaises(bridge.BridgeError) as error:
            bridge.build_subdir()
        self.assertIn('two values', str(error.exception))

    def test_an_export_that_agrees_with_the_file_is_accepted(self):
        # Sourcing `env.sh` and then running a Gate is a supported workflow: the
        # export duplicates the file, so nothing is ambiguous.
        self.use_machine_file(ASCENDNPU_BUILD_SUBDIR='build-release',
                              ASCENDNPU_NINJA_JOBS='8')
        os.environ['ASCENDNPU_BUILD_SUBDIR'] = 'build-release'
        self.assertEqual(bridge.build_subdir(), 'build-release')
        self.assertEqual(bridge.ninja_jobs(), '8')

class EnvFileCase(unittest.TestCase):
    """CANN/Conda come from the toolchain file, which is authoritative.

    The Adapter must read that file as data: reading it may never execute Host
    code, and a stale export in the launching shell may never mask an edit to
    the file. Both properties are what make the file a usable single source of
    truth, so both are tested directly rather than inferred.
    """

    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('ASCENDNPU_CANN_SETUP', 'ASCENDNPU_CONDA_SH', 'ASCENDNPU_CONDA_ENV',
                     'ASCENDNPU_ENV_FILE'):
            os.environ.pop(name, None)

    def env_file(self, text, name='env.sh'):
        path = self.root / name
        path.write_text(text)
        os.environ['ASCENDNPU_ENV_FILE'] = str(path)
        return path

    def test_values_are_read_and_defaults_are_empty(self):
        self.env_file('ASCENDNPU_CANN_SETUP=/cann/set_env.sh\n')
        self.assertEqual(bridge.toolchain_from_env_file(),
                         (Path('/cann/set_env.sh'), None, None))

    def test_the_file_is_authoritative_over_a_stale_export(self):
        os.environ['ASCENDNPU_CANN_SETUP'] = '/stale/set_env.sh'
        self.env_file('ASCENDNPU_CANN_SETUP=/from/file.sh\n')
        self.assertEqual(bridge.toolchain_from_env_file()[0], Path('/from/file.sh'))

    def test_an_edit_is_visible_without_any_other_change(self):
        path = self.env_file('ASCENDNPU_CONDA_ENV=first\n')
        self.assertEqual(bridge.toolchain_from_env_file()[2], 'first')
        path.write_text('ASCENDNPU_CONDA_ENV=second\n')
        self.assertEqual(bridge.toolchain_from_env_file()[2], 'second')

    def test_reading_the_file_never_executes_it(self):
        marker = self.root / 'executed'
        self.env_file(f'ASCENDNPU_CONDA_ENV=$(touch {marker}; echo x)\n')
        _, _, conda_env = bridge.toolchain_from_env_file()
        self.assertFalse(marker.exists(), 'the file was executed, not parsed')
        self.assertIn('touch', conda_env)  # kept literally, not run

    def test_a_backtick_value_is_not_run_and_is_rejected_as_a_path(self):
        # Backticks are kept literal, so the value is not an absolute path and the
        # Gate fails closed. Either way the marker must not exist: the point is
        # that the file is data, so nothing in it executes.
        marker = self.root / 'executed-backtick'
        self.env_file(f'ASCENDNPU_CANN_SETUP=`touch {marker}`\n')
        with self.assertRaises(bridge.BridgeError):
            bridge.toolchain_from_env_file()
        self.assertFalse(marker.exists(), 'the file was executed, not parsed')

    def test_comments_blanks_and_quotes_are_handled(self):
        self.env_file('# a comment\n\nASCENDNPU_CANN_SETUP="/cann/set_env.sh"\n'
                      "ASCENDNPU_CONDA_SH='/conda/c.sh'\nASCENDNPU_CONDA_ENV=mk\n")
        self.assertEqual(bridge.toolchain_from_env_file(),
                         (Path('/cann/set_env.sh'), Path('/conda/c.sh'), 'mk'))

    def test_a_missing_default_file_means_no_toolchain(self):
        # CANN and Conda are optional; an absent default file is not an error.
        os.environ.pop('ASCENDNPU_ENV_FILE', None)
        with patch.object(bridge, 'default_env_file', return_value=self.root / 'absent.env'):
            self.assertEqual(bridge.toolchain_from_env_file(), (None, None, None))

    def test_an_explicitly_selected_missing_file_is_an_error(self):
        os.environ['ASCENDNPU_ENV_FILE'] = str(self.root / 'absent.env')
        with self.assertRaises(bridge.BridgeError) as error:
            bridge.toolchain_from_env_file()
        self.assertIn('does not exist', str(error.exception))

    def test_a_directory_is_not_a_configuration_file(self):
        os.environ['ASCENDNPU_ENV_FILE'] = str(self.root)
        with self.assertRaises(bridge.BridgeError):
            bridge.toolchain_from_env_file()

    def test_a_relative_selector_is_refused(self):
        os.environ['ASCENDNPU_ENV_FILE'] = 'relative/env.sh'
        with self.assertRaises(bridge.BridgeError):
            bridge.toolchain_from_env_file()

    def test_malformed_lines_fail_closed(self):
        for text in ('rm -rf /\n', 'ASCENDNPU_CANN_SETUP =/a\n', 'export X=/a\n',
                     'ASCENDNPU_CANN_SETUP="/a\n', 'ASCENDNPU_CANN_SETUP=/a\nextra\n'):
            self.env_file(text)
            with self.assertRaises(bridge.BridgeError, msg=repr(text)):
                bridge.toolchain_from_env_file()

    def test_an_unknown_key_is_refused_not_ignored(self):
        # A typo such as a doubled underscore must not silently leave CANN unset.
        self.env_file('ASCENDNPU__CANN_SETUP=/a\n')
        with self.assertRaises(bridge.BridgeError) as error:
            bridge.toolchain_from_env_file()
        self.assertIn('unknown key', str(error.exception))

    def test_a_duplicated_key_is_refused(self):
        self.env_file('ASCENDNPU_CONDA_ENV=a\nASCENDNPU_CONDA_ENV=b\n')
        with self.assertRaises(bridge.BridgeError):
            bridge.toolchain_from_env_file()

    def test_a_relative_value_is_refused(self):
        self.env_file('ASCENDNPU_CANN_SETUP=cann/set_env.sh\n')
        with self.assertRaises(bridge.BridgeError):
            bridge.toolchain_from_env_file()

    def test_an_empty_value_disables_that_component(self):
        self.env_file('ASCENDNPU_CANN_SETUP=\nASCENDNPU_CONDA_ENV=\n')
        self.assertEqual(bridge.toolchain_from_env_file(), (None, None, None))

    def test_the_sample_template_is_valid_and_has_no_values(self):
        # The committed template must parse and must not configure this machine.
        # It names every machine fact the Adapter reads - the toolchain, the
        # container it enters and the build selection - and leaves each one for
        # the operator. The build selection is shown commented out, because an
        # empty value there is a typo rather than "use the default".
        sample = ADAPTER / 'env.example.sh'
        values = bridge._parse_env_file(sample.read_text(), sample)
        self.assertEqual(set(values),
                         set(bridge.FILE_KEYS) - set(bridge.BUILD_KEYS) - set(bridge.SOURCE_KEYS))
        for name in bridge.CONTAINER_KEYS + ('ASCENDNPU_CCEC_BIN',):
            self.assertEqual(values[name], '', name)
        # Optional selections are shown commented out: an empty key there is a
        # typo, and leaving the line out is how the file says "use the default".
        for name in bridge.BUILD_KEYS + bridge.SOURCE_KEYS:
            self.assertIn(f'# {name}=', sample.read_text(), name)
        self.assertEqual(set(values.values()), {''})

    def test_a_key_the_gate_does_not_use_is_ignored_not_rejected(self):
        # A machine stores ASCENDNPU_CCEC_BIN for scripts/full-build.sh. The Gate
        # must neither require nor reject it.
        self.env_file('ASCENDNPU_CANN_SETUP=/cann/set_env.sh\n'
                      'ASCENDNPU_CCEC_BIN=/opt/ccec/bin\n')
        self.assertEqual(bridge.toolchain_from_env_file(),
                         (Path('/cann/set_env.sh'), None, None))


class EnvFileLocationCase(unittest.TestCase):
    """Where the machine file is found, without ASCENDNPU_ENV_FILE set.

    The file is git-ignored and sits beside its template, so a human or an agent
    edits it in the repository. The Gate reaches it by walking up from itself,
    which must resolve to the checkout both when the Gate runs from there and
    when it runs from an installed Pack inside it.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkout = Path(self.temp.name) / 'PiEngineeringHarness'
        adapter = self.checkout / 'adapters' / 'ascendnpu-ir'
        adapter.mkdir(parents=True)
        (adapter / 'env.example.sh').write_text('ASCENDNPU_CANN_SETUP=\n')
        os.environ.pop('ASCENDNPU_ENV_FILE', None)

    @property
    def adapter(self):
        return self.checkout / 'adapters' / 'ascendnpu-ir'

    def test_the_file_next_to_its_template_is_used_from_a_checkout(self):
        with patch.object(bridge, 'ADAPTER_DIR', self.adapter):
            self.assertEqual(bridge.default_env_file(), self.adapter / 'env.sh')

    def test_an_installed_pack_inside_the_checkout_resolves_the_same_file(self):
        pack = self.checkout / 'pack-store' / 'installed' / 'ascendnpu-ir'
        (pack / 'scripts').mkdir(parents=True)
        # The Pack carries a copy of the template, so it must not be mistaken for
        # the checkout: only a directory holding `adapters/ascendnpu-ir` counts.
        (pack / 'env.example.sh').write_text('ASCENDNPU_CANN_SETUP=\n')
        with patch.object(bridge, 'ADAPTER_DIR', pack):
            self.assertEqual(bridge.default_env_file(), self.adapter / 'env.sh')

    def test_outside_any_checkout_there_is_no_default_file(self):
        loose = Path(self.temp.name) / 'somewhere' / 'ascendnpu-ir'
        (loose / 'scripts').mkdir(parents=True)
        with patch.object(bridge, 'ADAPTER_DIR', loose):
            self.assertIsNone(bridge.default_env_file())

    def test_the_found_file_is_what_the_gate_reads(self):
        (self.adapter / 'env.sh').write_text('ASCENDNPU_CANN_SETUP=/cann/set_env.sh\n')
        with patch.object(bridge, 'ADAPTER_DIR', self.adapter):
            self.assertEqual(bridge.toolchain_from_env_file(),
                             (Path('/cann/set_env.sh'), None, None))

    def test_the_explicit_selector_still_wins(self):
        elsewhere = Path(self.temp.name) / 'elsewhere.env'
        elsewhere.write_text('ASCENDNPU_CANN_SETUP=/other/set_env.sh\n')
        os.environ['ASCENDNPU_ENV_FILE'] = str(elsewhere)
        (self.adapter / 'env.sh').write_text('ASCENDNPU_CANN_SETUP=/cann/set_env.sh\n')
        with patch.object(bridge, 'ADAPTER_DIR', self.adapter):
            self.assertEqual(bridge.toolchain_from_env_file(),
                             (Path('/other/set_env.sh'), None, None))


class MountCase(unittest.TestCase):
    HOST = Path('/home/u/ws/AscendNPU-IR-1')

    def test_parent_bind_produces_suffix_mapping(self):
        mounts = [{'Type': 'bind', 'Source': '/home', 'Destination': '/home', 'RW': True}]
        with patch.object(bridge.os.path, 'realpath', side_effect=lambda p: str(p)):
            mapped = bridge.map_host_to_container(self.HOST, mounts)
        self.assertEqual(str(mapped), str(self.HOST))

    def test_exact_and_deepest_bind_wins(self):
        mounts = [
            {'Type': 'bind', 'Source': '/home', 'Destination': '/mnt/home'},
            {'Type': 'bind', 'Source': '/home/u/ws', 'Destination': '/src'},
        ]
        with patch.object(bridge.os.path, 'realpath', side_effect=lambda p: str(p)):
            mapped = bridge.map_host_to_container(self.HOST, mounts)
        self.assertEqual(str(mapped), '/src/AscendNPU-IR-1')

    def test_unmounted_host_root_is_refused(self):
        mounts = [{'Type': 'bind', 'Source': '/home/other', 'Destination': '/home/other'}]
        with patch.object(bridge.os.path, 'realpath', side_effect=lambda p: str(p)):
            with self.assertRaises(bridge.BridgeError) as error:
                bridge.map_host_to_container(self.HOST, mounts)
        self.assertIn('not exposed by any bind mount', str(error.exception))

    def test_ambiguous_equal_depth_mounts_are_refused(self):
        mounts = [
            {'Type': 'bind', 'Source': '/home/u/ws/AscendNPU-IR-1', 'Destination': '/a'},
            {'Type': 'bind', 'Source': '/home/u/ws/AscendNPU-IR-1', 'Destination': '/b'},
        ]
        with patch.object(bridge.os.path, 'realpath', side_effect=lambda p: str(p)):
            with self.assertRaises(bridge.BridgeError) as error:
                bridge.map_host_to_container(self.HOST, mounts)
        self.assertIn('ambiguous', str(error.exception))

    def test_non_bind_mounts_are_ignored(self):
        mounts = [{'Type': 'volume', 'Source': '/var/lib/docker/volumes/x/_data', 'Destination': '/home'}]
        with patch.object(bridge.os.path, 'realpath', side_effect=lambda p: str(p)):
            with self.assertRaises(bridge.BridgeError):
                bridge.map_host_to_container(self.HOST, mounts)


class RunCase(unittest.TestCase):
    """The full bridge, with `docker` replaced by a scripted fake."""

    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        (self.source / 'build').mkdir(parents=True)
        (self.source / 'build/build.ninja').write_text('# fake\n')
        origin = os.getcwd()
        self.addCleanup(os.chdir, origin)
        os.chdir(self.source)
        for name in list(os.environ):
            if name.startswith('ASCENDNPU_'):
                os.environ.pop(name)
        # These cases exercise the Docker bridge, so they name their own machine
        # file. Reading the packaged one would make the suite depend on whether
        # this particular machine keeps a real, git-ignored `env.sh`.
        self.machine = self.root / 'env.sh'
        self.write_machine({'ASCENDNPU_CONTAINER_NAME': 'fake-container',
                            'ASCENDNPU_CONTAINER_USER': 'u',
                            'ASCENDNPU_CONTAINER_HOME': '/home/u'})
        os.environ['ASCENDNPU_ENV_FILE'] = str(self.machine)

    def write_machine(self, values):
        self.machine.write_text(''.join(f'{key}={value}' + chr(10)
                                        for key, value in values.items()))
        return self.machine

    def fake_docker(self, *, mounts, inspect_ok=True, identity=None, ninja_exit=0, calls=None):
        stat = self.source.stat()
        identity = identity or f'{stat.st_dev}:{stat.st_ino}'
        def run(argv, **kwargs):
            if calls is not None:
                calls.append(argv)
            if argv[:2] == ['docker', 'inspect']:
                if not inspect_ok:
                    return subprocess.CompletedProcess(argv, 1, '', 'no such container')
                return subprocess.CompletedProcess(argv, 0, json.dumps(mounts), '')
            if argv[:2] == ['docker', 'exec']:
                if 'os.stat' in ' '.join(argv):
                    return subprocess.CompletedProcess(argv, 0, identity + '\n', '')
                return subprocess.CompletedProcess(argv, ninja_exit, 'ninja output\n', '')
            return subprocess.CompletedProcess(argv, 0, '', '')
        return run

    def test_valid_mapping_runs_and_preserves_ninja_exit_code(self):
        stat = self.source.stat()
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        calls = []
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, ninja_exit=0, calls=calls)):
            code = bridge.main(['bishengir-opt'])
        self.assertEqual(code, 0)
        exec_calls = [c for c in calls if c[:2] == ['docker', 'exec']]
        self.assertTrue(exec_calls)
        self.assertNotIn('-t', [item for call in exec_calls for item in call])
        self.assertIn('bishengir-opt', exec_calls[-1])

    def test_nonzero_ninja_is_returned_as_failure(self):
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, ninja_exit=17)):
            self.assertEqual(bridge.main(['check-bishengir']), 17)

    def test_timeout_exit_code_is_returned(self):
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, ninja_exit=124)):
            self.assertEqual(bridge.main(['check-bishengir']), 124)

    def test_a_timed_out_gate_can_never_be_reported_as_pass(self):
        # 124 is non-zero, so the recipe-level `exitCodeZero` check fails and the
        # Harness records FAIL rather than PASS. Assert it end to end through the
        # bridge, not just on the constant.
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, ninja_exit=124)):
            for target in ('bishengir-opt', 'check-bishengir'):
                code = bridge.main(['--timeout', '840', target])
                self.assertNotEqual(code, 0, target)
                self.assertEqual(code, 124, target)

    def test_the_recipes_run_the_bridge_with_the_pinned_timeout(self):
        manifest = load_manifest(ADAPTER / 'manifest.json')
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        # Only the Docker recipes reach the Docker bridge. The Host recipes are
        # covered by HostRunCase, which drives host_gate.py against a fake PATH:
        # running them here would either contact real Docker or prove nothing.
        docker_recipes = [recipe for recipe in manifest['recipes']
                          if recipe['argv'][2] == '{pack}/scripts/docker_gate.py']
        self.assertEqual([recipe['id'] for recipe in docker_recipes],
                         ['ascendnpu-ir.incremental-build', 'ascendnpu-ir.regression'])
        for recipe in docker_recipes:
            # The manifest invokes `python3 -B {pack}/scripts/docker_gate.py ...`;
            # the bridge itself receives only the arguments after the script path.
            self.assertEqual(recipe['argv'][:3], ['python3', '-B', '{pack}/scripts/docker_gate.py'])
            calls = []
            with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, calls=calls)):
                self.assertEqual(bridge.main(recipe['argv'][3:]), 0, recipe['id'])
            script = next(part for call in calls if call[:2] == ['docker', 'exec']
                          for part in call if 'timeout --signal=TERM' in part)
            self.assertIn(f'--kill-after={bridge.KILL_AFTER_SECONDS} {recipe["argv"][4]} ', script)
            self.assertLess(int(recipe['argv'][4]) + bridge.KILL_AFTER_SECONDS, recipe['timeout'])

    def test_no_manifest_recipe_reaches_a_bridge_that_is_not_pinned(self):
        # Every recipe must run a script the manifest pins, whatever route it
        # takes. A recipe naming an unpinned path would execute bytes no digest
        # covers - the exact hole the files map exists to close.
        manifest = load_manifest(ADAPTER / 'manifest.json')
        for recipe in manifest['recipes']:
            self.assertIn(recipe['argv'][2].removeprefix('{pack}/'), manifest['files'],
                          recipe['id'])

    def test_missing_container_fails_before_any_exec(self):
        calls = []
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=[], inspect_ok=False, calls=calls)):
            with self.assertRaises(bridge.BridgeError):
                bridge.main(['bishengir-opt'])
        self.assertTrue(all(call[:2] == ['docker', 'inspect'] for call in calls))

    def test_unshared_source_tree_fails_before_ninja(self):
        mounts = [{'Type': 'bind', 'Source': '/somewhere/else', 'Destination': '/somewhere/else'}]
        calls = []
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, calls=calls)):
            with self.assertRaises(bridge.BridgeError):
                bridge.main(['bishengir-opt'])
        self.assertFalse(any(call[:2] == ['docker', 'exec'] and 'ninja' in ' '.join(call) for call in calls))

    def test_identity_mismatch_is_refused(self):
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, identity='1:1')):
            with self.assertRaises(bridge.BridgeError) as error:
                bridge.main(['bishengir-opt'])
        self.assertIn('not the host source root', str(error.exception))

    def test_declared_container_source_must_match_the_mapping(self):
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        self.write_machine({'ASCENDNPU_CONTAINER_NAME': 'fake-container',
                            'ASCENDNPU_CONTAINER_USER': 'u',
                            'ASCENDNPU_CONTAINER_HOME': '/home/u',
                            'ASCENDNPU_CONTAINER_SOURCE': '/a/different/path'})
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts)):
            with self.assertRaises(bridge.BridgeError) as error:
                bridge.main(['bishengir-opt'])
        self.assertIn('does not match the bind-mount mapping', str(error.exception))

    def test_the_declared_container_source_is_machine_configuration(self):
        # It decides which tree the Gate runs in, so it belongs in the file with
        # the container identity - not in the launching shell, where a stale
        # export would gate a directory nobody recorded.
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        os.environ['ASCENDNPU_CONTAINER_SOURCE'] = str(self.source)
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts)) as run:
            with self.assertRaises(bridge.BridgeError) as error:
                bridge.main(['bishengir-opt'])
        self.assertIn('read only from', str(error.exception))
        self.assertFalse(any(call[:2] == ['docker', 'exec'] for call in run.call_args_list))

    def test_a_file_declared_container_source_is_honoured_when_it_matches(self):
        # A checkout mounted at a different container path: the file confirms the
        # path, and the derived mapping must still agree with it.
        visible = self.root / 'visible'
        visible.mkdir()
        self.source.rename(visible / 'source')
        self.source = visible / 'source'
        self.write_machine({'ASCENDNPU_CONTAINER_NAME': 'fake-container',
                            'ASCENDNPU_CONTAINER_USER': 'u',
                            'ASCENDNPU_CONTAINER_HOME': '/home/u',
                            'ASCENDNPU_CONTAINER_SOURCE': str(self.source)})
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        calls = []
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, calls=calls)):
            self.assertEqual(bridge.main(['bishengir-opt']), 0)
        exec_calls = [c for c in calls if c[:2] == ['docker', 'exec']]
        self.assertTrue(exec_calls)
        self.assertIn(f'-w', exec_calls[0])
        self.assertIn(str(self.source), exec_calls[0])

    def test_no_source_logs_are_written(self):
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        before = {p.name for p in self.source.iterdir()}
        with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, ninja_exit=0)):
            bridge.main(['bishengir-opt'])
        self.assertEqual({p.name for p in self.source.iterdir()}, before)

    def test_main_refuses_partial_conda_configuration_before_any_docker_call(self):
        # The half configuration is discovered in the toolchain file, before any
        # Docker call, so the invalid file must be reported without a container
        # ever being consulted.
        path = self.root / 'partial.env'
        path.write_text('ASCENDNPU_CONDA_SH=/c/conda.sh\n')
        os.environ['ASCENDNPU_ENV_FILE'] = str(path)
        with patch('subprocess.run') as run:
            with self.assertRaises(bridge.BridgeError):
                bridge.main(['bishengir-opt'])
        run.assert_not_called()

    def test_behavior_is_independent_of_the_checked_out_branch(self):
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True)
        mounts = [{'Type': 'bind', 'Source': str(self.root), 'Destination': str(self.root)}]
        observed = []
        for branch in ('main', 'dev-compiletime-skippass', 'some/unrelated-name'):
            subprocess.run(['git', '-C', str(self.source), 'checkout', '-qB', branch], check=True)
            calls = []
            with patch('subprocess.run', side_effect=self.fake_docker(mounts=mounts, calls=calls)):
                self.assertEqual(bridge.main(['check-bishengir']), 0)
            exec_calls = [call for call in calls if call[:2] == ['docker', 'exec']]
            observed.append((exec_calls[0], exec_calls[-1]))
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(observed[0], observed[2])
        for prefix, command in observed:
            self.assertTrue(all(branch not in ' '.join(part) for part in command
                                for branch in ('dev-compiletime-skippass', 'some/unrelated-name')))


class ScriptExecutionCase(unittest.TestCase):
    """Run the generated container script with real Bash and fake tools.

    Mocks prove the argv shape; they cannot prove that an inherited PATH
    survives, that an empty PATH gets a fallback, or that a tool supplied only by
    a sourced setup file becomes visible. These cases stage a small fake
    container root and execute the generated script for real.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'src'
        (self.source / 'build').mkdir(parents=True)
        (self.source / 'build/build.ninja').write_text('# fake\n')
        # A recording `ninja` and a `cmake`, both on the staged PATH.
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        self.ninja_log = self.root / 'ninja-args'
        (self.tools / 'ninja').write_text(
            '#!/bin/bash\nprintf "%s\\n" "$@" > ' + shlex.quote(str(self.ninja_log)) + '\n')
        (self.tools / 'cmake').write_text('#!/bin/bash\nexit 0\n')
        for path in (self.tools / 'ninja', self.tools / 'cmake'):
            path.chmod(0o755)

    def run_script(self, script, path, *args, trace=False):
        env = {'HOME': str(self.root)}
        if path is not None:
            env['PATH'] = path
        flags = '-xc' if trace else '-c'
        return subprocess.run([shutil.which('bash'), flags, script, 'bash', *args],
                              capture_output=True, text=True, env=env)

    def script(self, **kwargs):
        return bridge.container_script(str(self.source), 'build', '4', 840, **kwargs)

    def test_inherited_path_survives_and_a_staged_tool_is_used(self):
        script = self.script(cann=None, conda_sh=None, conda_env=None)
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.ninja_log.read_text().split(),
                         ['-C', 'build', '-j', '4', 'bishengir-opt'])

    def test_empty_path_gets_the_system_fallback(self):
        # With PATH empty the `${PATH:-...}` fallback must run, so the script never
        # proceeds with an empty PATH. `-x` traces the assignment directly, which
        # keeps the assertion independent of what this host happens to install.
        script = self.script(cann=None, conda_sh=None, conda_env=None)
        done = self.run_script(script, '', 'check-bishengir', trace=True)
        self.assertIn('PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                      done.stderr)

    def test_missing_cmake_fails_before_ninja_runs(self):
        only_ninja = self.root / 'only-ninja'
        only_ninja.mkdir()
        shutil.copy2(self.tools / 'ninja', only_ninja / 'ninja')
        empty = self.root / 'empty'
        empty.mkdir()
        script = self.script(cann=None, conda_sh=None, conda_env=None)
        done = self.run_script(script, f'{only_ninja}:{empty}', 'bishengir-opt')
        self.assertEqual(done.returncode, 3)
        self.assertIn('cmake not found in PATH', done.stderr)
        self.assertFalse(self.ninja_log.exists())

    def test_cann_then_conda_order_is_observable(self):
        # A fake CANN setup and a fake conda.sh record when they run, and each
        # only adds its tool to PATH, so the tool checks prove the order.
        events = self.root / 'events'
        cann = self.root / 'set_env.sh'
        conda_sh = self.root / 'conda.sh'
        cann.write_text(
            'printf "cann\\n" >> ' + shlex.quote(str(events)) + '\n'
            'export ASCEND_HOME_PATH=/cann\n'
            'export PATH=' + shlex.quote(str(self.tools)) + ':"$PATH"\n')
        conda_sh.write_text(
            'printf "conda\\n" >> ' + shlex.quote(str(events)) + '\n'
            'conda() { printf "activate\\n" >> ' + shlex.quote(str(events)) + '; }\n')
        script = self.script(cann=Path(cann), conda_sh=Path(conda_sh), conda_env='env311')
        done = self.run_script(script, '/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(events.read_text().split(), ['cann', 'conda', 'activate'])

    def test_cann_source_failure_is_not_ignored(self):
        # A CANN script that fails normally (return non-zero) must hit `fail`.
        cann = self.root / 'broken.sh'
        cann.write_text('return 7\n')
        script = self.script(cann=Path(cann), conda_sh=None, conda_env=None)
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 3, done.stderr)
        self.assertIn('sourcing CANN setup failed', done.stderr)
        self.assertFalse(self.ninja_log.exists())

    def test_cann_script_that_exits_the_shell_still_fails_the_gate(self):
        # Some vendor scripts call `exit`. That ends the shell before `fail` can
        # run, but the code is non-zero, so the Gate still fails and ninja never
        # runs. What must not happen is a zero exit with ninja executed.
        cann = self.root / 'hard-exit.sh'
        cann.write_text('exit 7\n')
        script = self.script(cann=Path(cann), conda_sh=None, conda_env=None)
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertNotEqual(done.returncode, 0)
        self.assertFalse(self.ninja_log.exists())

    def test_cann_without_ascend_home_path_is_not_ignored(self):
        cann = self.root / 'incomplete.sh'
        cann.write_text('true\n')
        script = self.script(cann=Path(cann), conda_sh=None, conda_env=None)
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 3)
        self.assertIn('ASCEND_HOME_PATH', done.stderr)

    def test_conda_activate_failure_is_not_ignored(self):
        conda_sh = self.root / 'conda.sh'
        conda_sh.write_text('conda() { return 1; }\n')
        script = self.script(cann=None, conda_sh=Path(conda_sh), conda_env='env311')
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 3)
        self.assertIn('conda activate env311 failed', done.stderr)

    def test_a_conda_supplied_tool_wins_over_the_system_one(self):
        # PATH precedence: a cmake/ninja that `conda activate` prepends must be
        # the one used, because the image's tools are not pinned to a directory.
        conda_tools = self.root / 'conda-tools'
        conda_tools.mkdir()
        marker = self.root / 'conda-ninja-used'
        (conda_tools / 'ninja').write_text('#!/bin/bash\nprintf used > ' + shlex.quote(str(marker)) + '\n')
        (conda_tools / 'cmake').write_text('#!/bin/bash\nexit 0\n')
        for path in (conda_tools / 'ninja', conda_tools / 'cmake'):
            path.chmod(0o755)
        conda_sh = self.root / 'conda.sh'
        conda_sh.write_text('conda() { export PATH="' + str(conda_tools) + ':$PATH"; }\n')
        script = self.script(cann=None, conda_sh=Path(conda_sh), conda_env='env311')
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'check-bishengir')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(marker.exists(), 'the conda-provided ninja was not used')

    def stage_a_fake_build_script(self, status=0, products=True):
        """A fake build.sh that records its argv and leaves the real products."""
        recorded = self.root / 'build-args'
        tools = self.source / 'build-tools'
        tools.mkdir(exist_ok=True)
        script = ['#!/bin/bash', 'printf "%s\\n" "$@" > ' + shlex.quote(str(recorded))]
        if products:
            # Mirror where build-tools/build.sh actually installs: under the
            # directory named by --build, never at the source root.
            script += [
                'build=.',
                'while [ $# -gt 0 ]; do',
                '  if [ "$1" = --build ]; then build=$2; fi',
                '  shift',
                'done',
                'mkdir -p "$build/bin"',
                'for name in bishengir-opt bishengir-compile; do',
                '  printf "#!/bin/bash\\nexit 0\\n" > "$build/bin/$name"',
                '  chmod 755 "$build/bin/$name"',
                'done',
            ]
        script.append(f'exit {status}')
        (tools / 'build.sh').write_text('\n'.join(script) + '\n')
        return recorded

    def test_an_empty_tree_is_built_from_scratch_and_the_products_are_checked(self):
        # The branch the user asked for: a build directory with neither
        # build.ninja nor CMakeCache.txt is configured and built by the pinned
        # command, and the Gate passes only when the products exist afterwards.
        shutil.rmtree(self.source / 'build')
        recorded = self.stage_a_fake_build_script()
        script = bridge.container_script(
            str(self.source), 'build', '64', 840, None, None, None,
            full_timeout=21600, ccec_bin=Path('/opt/ccec/bin'), allow_full_build=True)
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 0, done.stderr)
        # Ninja is never run against a tree that has no build.ninja.
        self.assertFalse(self.ninja_log.exists())
        # The command is the operator's, with the flags this Adapter owns.
        argv = recorded.read_text().splitlines()
        self.assertEqual(argv, [
            '--c-compiler', 'clang', '--cxx-compiler', 'clang++', '--enable-lld',
            '--build-type', 'Release', '--enable-assertion', '-t',
            '--bisheng-compiler', '/opt/ccec/bin', '--disable-werror',
            '--disable-bishengir-werror', '--build-triton', '--build-shmem-template',
            '--build', './build', '-j', '64',
        ])
        # Nothing destructive, and the machine file is not rewritten.
        for forbidden in ('-r', '--rebuild', '--clean'):
            self.assertNotIn(forbidden, argv)
        self.assertTrue((self.source / 'build/bin/bishengir-opt').is_file())

    def test_a_full_build_that_produces_nothing_cannot_pass(self):
        # "It exited zero" is not the same as "it built the compiler": a build
        # script that returns success without installing products must fail.
        shutil.rmtree(self.source / 'build')
        self.stage_a_fake_build_script(products=False)
        script = bridge.container_script(
            str(self.source), 'build', '4', 840, None, None, None,
            full_timeout=21600, ccec_bin=Path('/opt/ccec/bin'), allow_full_build=True)
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 3)
        self.assertIn('full build did not produce build/bin/bishengir-opt', done.stderr)

    def test_a_failing_full_build_is_reported_as_a_failure(self):
        shutil.rmtree(self.source / 'build')
        self.stage_a_fake_build_script(status=7, products=False)
        script = bridge.container_script(
            str(self.source), 'build', '4', 840, None, None, None,
            full_timeout=21600, ccec_bin=Path('/opt/ccec/bin'), allow_full_build=True)
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 3)
        self.assertIn('full build failed', done.stderr)
        self.assertFalse(self.ninja_log.exists())

    def test_a_full_build_into_another_directory_checks_that_directory(self):
        # The build directory is a machine fact, and the product check has to
        # follow it: a Gate that always looked at `build` would pass a run that
        # wrote `build-release`.
        shutil.rmtree(self.source / 'build')
        self.stage_a_fake_build_script()
        script = bridge.container_script(
            str(self.source), 'build-release', '4', 840, None, None, None,
            full_timeout=21600, ccec_bin=Path('/opt/ccec/bin'), allow_full_build=True)
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue((self.source / 'build-release/bin/bishengir-opt').is_file())
        self.assertFalse((self.source / 'build').exists())

    def test_a_half_configured_tree_is_refused_and_left_alone(self):
        # CMakeCache.txt without build.ninja is neither incrementally buildable
        # nor reconfigured without deleting state. The Gate must refuse rather
        # than remove the operator's tree.
        (self.source / 'build/CMakeCache.txt').write_text('# configured, never built\n')
        (self.source / 'build/build.ninja').unlink()
        script = self.script(cann=None, conda_sh=None, conda_env=None)
        done = self.run_script(script, f'{self.tools}:/usr/bin:/bin', 'bishengir-opt')
        self.assertEqual(done.returncode, 3)
        self.assertIn('CMakeCache.txt exists without build/build.ninja', done.stderr)
        self.assertTrue((self.source / 'build/CMakeCache.txt').is_file())
        self.assertFalse(self.ninja_log.exists())


class HostToolchain:
    """A temporary PATH of fake tools, so a Host preflight needs no real compiler.

    host_gate resolves every tool from the EFFECTIVE PATH, after CANN and Conda.
    Staging one executable per name - each answering --version with a chosen
    string - is what makes the numeric comparison and the refusal paths testable
    without installing or downgrading anything on the machine running the suite.
    """

    # What each tool reports by default. The version strings are the shapes the
    # real tools print, including clang's "Ubuntu clang version 18.1.3 (1ubuntu1)",
    # because a pattern that only matches a bare 1.2.3 would pass here and fail
    # on the machine this Gate exists for.
    VERSIONS = {
        'cmake': 'cmake version 3.28.3',
        'ninja': '1.13.2',
        'clang': 'Ubuntu clang version 18.1.3 (1ubuntu1)',
        'clang++': 'Ubuntu clang version 18.1.3 (1ubuntu1)',
        'g++': 'g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0',
        'ld.lld': 'Ubuntu LLD 18.1.3',
        'python3': 'Python 3.12.3',
        'git': 'git version 2.43.0',
    }
    # bash, timeout and nproc are staged too: the preflight probes them, and a
    # case that only staged the interesting tools would fail for a boring reason.
    TOOLS = ('bash', 'timeout', 'nproc') + tuple(VERSIONS)

    def __init__(self, root, source):
        self.root = Path(root)
        self.source = Path(source)
        self.path = self.root / 'tools'
        self.path.mkdir()

    def write(self, *, versions=None, omit=(), git_root='source'):
        versions = versions or {}
        for name in self.TOOLS:
            if name in omit:
                # Removing the file is what "missing" means. Skipping the write
                # would leave a tool staged by an earlier call in place, and every
                # refusal case here would silently assert nothing.
                (self.path / name).unlink(missing_ok=True)
                continue
            reported = versions.get(name, self.VERSIONS.get(name))
            lines = ['#!/bin/bash']
            if reported is not None:
                lines.append('if [ "$1" = --version ]; then printf "%s\\n" '
                             + shlex.quote(reported) + '; exit 0; fi')
            if name == 'bash':
                # The generated script IS bash, so re-exec the real one rather
                # than pretending to be it.
                lines.append('exec /bin/bash "$@"')
            if name == 'timeout':
                # The generated script runs its work under `timeout`, so a stub
                # that merely exited would never execute the command it wraps,
                # and every run case would assert nothing. Drop the options and
                # the duration, then run the command for real, untimed.
                lines += ['while [ $# -gt 0 ]; do',
                          '  case "$1" in',
                          '    -*) shift ;;',
                          '    *) break ;;',
                          '  esac',
                          'done',
                          'shift',  # the duration
                          'exec "$@"']
            if name == 'nproc':
                # `ninja_jobs()` defaults to $(nproc). A stub that printed
                # nothing would delete the -j value and shift the Ninja argv.
                lines += ['if [ "$1" = --version ]; then exit 0; fi',
                          'printf "4\\n"',
                          'exit 0']
            if name == 'git' and git_root is not None:
                root = self.source if git_root == 'source' else self.root / 'not-a-tree'
                lines.append('if [ "$1" = rev-parse ]; then printf "%s\\n" '
                             + shlex.quote(str(root)) + '; exit 0; fi')
            lines.append('exit 0')
            tool = self.path / name
            tool.write_text('\n'.join(lines) + '\n')
            tool.chmod(0o755)
        return self.path


class HostCase(unittest.TestCase):
    """A staged machine file, a staged source tree and a staged PATH.

    ASCENDNPU_ENV_FILE is always explicit, so the outcome cannot depend on
    whether the machine running the suite keeps a real, git-ignored env.sh
    beside the packaged Adapter.
    """

    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.machine = self.root / 'env.sh'
        self.write_machine('ASCENDNPU_CANN_SETUP=\n')
        self.tools = HostToolchain(self.root, self.source)
        self.tools.write()
        origin = os.getcwd()
        self.addCleanup(os.chdir, origin)
        os.chdir(self.source)

    def write_machine(self, text):
        self.machine.write_text(text)
        os.environ['ASCENDNPU_ENV_FILE'] = str(self.machine)
        return self.machine

    def stage_tree(self, state='incremental', subdir='build'):
        """Put the build directory in one of the generator's three states."""
        build = self.source / subdir
        if state == 'incremental':
            build.mkdir(parents=True, exist_ok=True)
            (build / 'build.ninja').write_text('# fake\n')
        elif state == 'half':
            build.mkdir(parents=True, exist_ok=True)
            (build / 'CMakeCache.txt').write_text('# configured, never built\n')
        return build

    def stage_third_party(self, llvm='populated', shmem='populated'):
        for name, how in (('llvm-project', llvm), ('shmem', shmem)):
            tree = self.source / 'third-party' / name
            if how == 'absent':
                continue
            tree.mkdir(parents=True, exist_ok=True)
            if how == 'populated':
                (tree / 'CMakeLists.txt').write_text('# source\n')

    def stage_ccec(self, executable=('ccec', 'llvm-link')):
        """A ccec directory; names outside executable exist but are not +x."""
        ccec = self.root / 'ccec'
        ccec.mkdir(exist_ok=True)
        for name in ('ccec', 'llvm-link'):
            tool = ccec / name
            tool.write_text('#!/bin/bash\nexit 0\n')
            tool.chmod(0o755 if name in executable else 0o644)
        self.write_machine('ASCENDNPU_CANN_SETUP=\nASCENDNPU_CCEC_BIN=' + str(ccec) + '\n')
        return ccec

    def check(self, mode='build'):
        """Run one configuration check with the staged PATH as the whole PATH."""
        os.environ['PATH'] = str(self.tools.path)
        os.environ['HOME'] = str(self.root)
        return host_bridge.main(['--check-config-' + mode])

    def refusal(self, mode='build'):
        """The message a refused check reported, or a failure if it passed."""
        with self.assertRaises(host_bridge.BridgeError) as caught:
            self.check(mode)
        return caught.exception.message

class HostPreflightCase(HostCase):
    """The Host toolchain floors, proven numerically and with real refusals.

    These are the checks Docker mode cannot provide: it inherits the image's
    cmake and ninja, so a Host run is the only place a too-old tool needs to be
    caught - and catching it here means catching it before a configure that
    would spend minutes on LLVM first.
    """

    def test_a_satisfied_toolchain_validates(self):
        self.stage_tree()
        self.assertEqual(self.check(), 0)
        # Nothing was built, configured or written.
        self.assertEqual(sorted(p.name for p in self.source.iterdir()), ['build'])
        self.assertEqual(sorted(p.name for p in (self.source / 'build').iterdir()),
                         ['build.ninja'])

    def test_ninja_1_11_is_refused_and_1_12_and_1_13_are_accepted(self):
        # The exact boundary the source enforces at CMakeLists.txt:6-13, and the
        # exact pair this machine has installed: /usr/bin/ninja is 1.11.1 and
        # /usr/local/bin/ninja is 1.13.2, so which one PATH found decides.
        self.stage_tree()
        for version in ('1.11.1', '1.11.0', '1.9.0'):
            self.tools.write(versions={'ninja': version})
            message = self.refusal()
            self.assertIn('ninja', message)
            self.assertIn('1.12.0', message)
            self.assertIn('CMakeLists.txt:6-13', message)
        for version in ('1.12.0', '1.12.1', '1.13.2'):
            self.tools.write(versions={'ninja': version})
            self.assertEqual(self.check(), 0, version)

    def test_cmake_3_27_is_refused_and_3_28_is_accepted(self):
        self.stage_tree()
        for version in ('cmake version 3.27.9', 'cmake version 3.27.0'):
            self.tools.write(versions={'cmake': version})
            message = self.refusal()
            self.assertIn('cmake', message)
            self.assertIn('3.28.0', message)
        for version in ('cmake version 3.28.0', 'cmake version 3.28.3',
                        'cmake version 4.0.1'):
            self.tools.write(versions={'cmake': version})
            self.assertEqual(self.check(), 0, version)

    def test_the_version_comparison_is_numeric_not_lexical(self):
        # 3.9 is older than 3.28, but "3.9" > "3.28" as a string, and as a Bash
        # integer test the comparison is not even a number. Both directions are
        # asserted so a comparison that merely reverses the bug cannot pass.
        self.stage_tree()
        self.tools.write(versions={'cmake': 'cmake version 3.9.6'})
        self.assertIn('3.28.0', self.refusal())
        self.tools.write(versions={'cmake': 'cmake version 3.28.0'})
        self.assertEqual(self.check(), 0)
        # And the same for ninja: 1.9 is older than 1.12 although "1.9" > "1.12".
        self.tools.write(versions={'ninja': '1.9.0'})
        self.assertIn('1.12.0', self.refusal())
        self.tools.write(versions={'ninja': '1.12.0'})
        self.assertEqual(self.check(), 0)

    def test_a_version_with_fewer_components_is_padded_not_compared_by_length(self):
        # ninja --version prints three components, but a tool reporting two must
        # not be judged by tuple length: 3.28 has to satisfy a 3.28.0 floor.
        self.stage_tree()
        self.tools.write(versions={'ninja': '1.12', 'cmake': 'cmake version 3.28'})
        self.assertEqual(self.check(), 0)

    def test_every_required_tool_must_resolve_from_the_effective_path(self):
        # Each of these is a tool the build really runs, so a missing one has to
        # be named at t=0 rather than surfacing as a Ninja or lit failure.
        self.stage_tree()
        for name in ('clang', 'clang++', 'g++', 'ld.lld', 'git', 'timeout',
                     'nproc', 'python3'):
            with self.subTest(name=name):
                self.tools.write(omit=(name,))
                self.assertIn(name, self.refusal())
        # cmake and ninja are refused even earlier, by the inlined init script,
        # whose message is more specific. Assert the refusal, not its wording.
        for name in ('cmake', 'ninja'):
            with self.subTest(name=name):
                self.tools.write(omit=(name,))
                self.assertIn(name, self.refusal())

    def test_a_missing_shell_still_fails_the_check(self):
        # bash is not a probed tool: it is what RUNS the probe. Without it Python
        # cannot even launch the check, so the failure is an OSError rather than a
        # BridgeError with a nice message. What matters is that it is not 0 -
        # a preflight that cannot run must never report a valid configuration.
        self.stage_tree()
        self.tools.write(omit=('bash',))
        os.environ['PATH'] = str(self.tools.path)
        os.environ['HOME'] = str(self.root)
        with self.assertRaises((host_bridge.BridgeError, OSError)):
            host_bridge.main(['--check-config-build'])

    def test_a_failure_outside_a_git_work_tree_is_refused(self):
        # build-tools/build.sh:75-77 exits 1 outside a repository, so a Host tree
        # that is not a work tree cannot be built - and saying so here is cheaper
        # than the same failure after Ninja starts.
        self.stage_tree()
        self.tools.write(git_root=None)
        self.assertIn('git rev-parse --show-toplevel failed', self.refusal())

    def test_the_resolved_path_is_reported_so_a_shadowed_tool_is_actionable(self):
        # "ninja 1.11.1" is not actionable; "the first ninja on PATH is
        # /usr/bin/ninja" is. The report goes to stderr, before any judgement.
        self.stage_tree()
        self.tools.write(versions={'ninja': '1.11.1'})
        os.environ['PATH'] = str(self.tools.path)
        os.environ['HOME'] = str(self.root)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(host_bridge.BridgeError):
                host_bridge.main(['--check-config-build'])
        report = stderr.getvalue()
        # The path is printed beside the version, and the refusal repeats it:
        # the operator has to know WHICH ninja was judged.
        self.assertIn(str(self.tools.path / 'ninja'), report)
        self.assertIn('1.11.1', report)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(host_bridge.BridgeError) as caught:
                host_bridge.main(['--check-config-build'])
        self.assertIn(str(self.tools.path / 'ninja'), caught.exception.message)


class HostFromZeroCase(HostCase):
    """What only a from-zero build needs, demanded only when it is needed."""

    def setUp(self):
        super().setUp()
        # state=full means neither build.ninja nor CMakeCache.txt exists.
        self.ccec = self.stage_ccec()
        self.stage_third_party()

    def test_a_populated_from_zero_tree_validates(self):
        self.assertEqual(self.check(), 0)

    def test_a_ccec_directory_without_the_tools_is_refused(self):
        # bishengir/lib/Template/CMakeLists.txt:573-578 makes both tools FATAL, so
        # a half-populated directory would fail after a configure that already ran.
        for missing in ('ccec', 'llvm-link'):
            with self.subTest(missing=missing):
                self.stage_ccec()
                (self.ccec / missing).unlink()
                message = self.refusal()
                self.assertIn(missing, message)
                self.assertIn('bishengir/lib/Template/CMakeLists.txt:573-578', message)

    def test_a_non_executable_ccec_is_refused(self):
        # The CMake check is EXISTS, so mode 0644 passes CMake and then fails
        # inside the tool with nothing naming the cause. This machine's real copy
        # is 0750, which is why the executable bit is the property that matters.
        self.stage_ccec(executable=('llvm-link',))
        message = self.refusal()
        self.assertIn('is not executable', message)
        self.assertIn(str(self.ccec / 'ccec'), message)

    def test_an_absent_ccec_selection_is_refused(self):
        self.write_machine('ASCENDNPU_CANN_SETUP=\n')
        self.assertIn('ASCENDNPU_CCEC_BIN', self.refusal())

    def test_an_empty_or_absent_third_party_tree_is_refused(self):
        # A submodule directory that exists but is empty is not populated: a
        # from-zero configure would compile nothing and fail much later.
        for name in ('llvm-project', 'shmem'):
            for how in ('empty', 'absent'):
                with self.subTest(name=name, how=how):
                    self.stage_third_party()
                    tree = self.source / 'third-party' / name
                    if how == 'empty':
                        for child in tree.iterdir():
                            if child.is_dir():
                                shutil.rmtree(child)
                            else:
                                child.unlink()
                    else:
                        shutil.rmtree(tree)
                    message = self.refusal()
                    self.assertIn(str(tree), message)
                    self.assertIn(name, message)

    def test_an_incremental_tree_needs_none_of_the_from_zero_prerequisites(self):
        # The rule is "a from-zero build needs a compiler", not "a build gate
        # does". A machine that only ever builds an existing tree may leave
        # CCEC_BIN empty, which is what ccec_bin() already documents.
        self.stage_tree('incremental')
        self.write_machine('ASCENDNPU_CANN_SETUP=\n')
        self.assertEqual(self.check(), 0)

    def test_a_half_configured_tree_is_refused_by_both_checks(self):
        # Same refusal as the generated MODE_PROBE, for the same reason: that
        # tree can neither be built incrementally nor reconfigured without
        # deleting state the Gate did not create, and no Gate may delete a tree.
        self.stage_tree('half')
        message = self.refusal('build')
        self.assertIn('CMakeCache.txt exists without', message)
        self.assertIn('build.ninja', message)
        # A refused tree must be left exactly as it was found.
        self.assertTrue((self.source / 'build' / 'CMakeCache.txt').is_file())
        self.assertFalse((self.source / 'build' / 'build.ninja').exists())
        # The regression check refuses it too: this is not a machine that merely
        # lacks a build tree, it is a tree that is in neither state.
        self.assertIn('CMakeCache.txt exists without', self.refusal('regression'))

    def test_the_regression_check_is_tolerable_before_any_build_tree_exists(self):
        # doctor runs on a NEW machine, which is exactly when no build tree exists
        # yet. Refusing there would make the diagnostic useless on the machine it
        # is for; the regression RECIPE is what refuses, and it says so itself.
        self.assertFalse((self.source / 'build').exists())
        self.assertEqual(self.check('regression'), 0)
        # Still nothing written: the check reports the state, it does not create it.
        self.assertFalse((self.source / 'build').exists())

    def test_the_regression_check_does_not_require_the_from_zero_prerequisites(self):
        # A regression run never configures, so demanding CCEC_BIN or populated
        # submodules would refuse a machine that can run the regression gate.
        self.write_machine('ASCENDNPU_CANN_SETUP=\n')
        shutil.rmtree(self.source / 'third-party')
        self.assertEqual(self.check('regression'), 0)

    def test_an_unknown_check_flag_is_refused(self):
        self.stage_tree()
        for flag in ('--check-config', '--check-config-everything'):
            with self.subTest(flag=flag):
                os.environ['PATH'] = str(self.tools.path)
                with self.assertRaises(host_bridge.BridgeError) as caught:
                    host_bridge.main([flag])
                self.assertIn('unknown configuration check', caught.exception.message)

class HostRunCase(HostCase):
    """The Host bridge end to end, with a recording fake ninja on PATH."""

    def setUp(self):
        super().setUp()
        self.ninja_log = self.root / 'ninja-args'
        self.stage_tree()
        self.fake_ninja('exit 0')

    def fake_ninja(self, tail):
        """A ninja that answers --version and then runs `tail` instead of building."""
        (self.tools.path / 'ninja').write_text(
            '#!/bin/bash\n'
            'if [ "$1" = --version ]; then printf "1.13.2\\n"; exit 0; fi\n'
            'printf "%s\\n" "$@" > ' + shlex.quote(str(self.ninja_log)) + '\n'
            + tail + '\n')
        (self.tools.path / 'ninja').chmod(0o755)

    def run_host(self, *args):
        # The staged tools come first so the fake ninja wins, but /usr/bin stays
        # behind them: the from-zero branch runs the committed shell script, which
        # uses coreutils, and a PATH of stubs alone would fail for that reason.
        os.environ['PATH'] = f'{self.tools.path}:/usr/bin:/bin'
        os.environ['HOME'] = str(self.root)
        return host_bridge.main(list(args))

    def test_the_incremental_run_uses_the_shared_command_and_returns_its_code(self):
        self.assertEqual(self.run_host('--timeout', '840', 'bishengir-opt'), 0)
        # `$(nproc)` is the default job count the file leaves unset; the generated
        # script expands it in the shell, which is why ninja sees the number.
        self.assertEqual(self.ninja_log.read_text().split(),
                         ['-C', 'build', '-j', '4', 'bishengir-opt'])

    def test_a_failing_ninja_is_never_reported_as_pass(self):
        self.fake_ninja('exit 17')
        self.assertEqual(self.run_host('check-bishengir'), 17)

    def test_a_timeout_is_returned_unchanged(self):
        # 124 is timeout saying the tree was too slow. Laundering it into 0, or
        # into a generic 3, would report PASS for work that never finished.
        self.fake_ninja('exit 124')
        self.assertEqual(self.run_host('--timeout', '840', 'check-bishengir'), 124)

    def test_the_regression_run_refuses_to_build_from_scratch(self):
        shutil.rmtree(self.source / 'build')
        self.assertEqual(self.run_host('--timeout', '1140', 'check-bishengir'), 3)
        self.assertFalse(self.ninja_log.exists())

    def test_the_timeout_chain_is_the_shared_one(self):
        self.fake_ninja('exit 0')
        self.run_host('--timeout', '840', 'bishengir-opt')
        # The generated script is the sibling's, so the same TERM/15s chain runs.
        script = host_bridge.container_script(str(self.source), 'build', '$(nproc)', 840,
                                             None, None, None, targets=('bishengir-opt',))
        self.assertIn(f'--kill-after={bridge.KILL_AFTER_SECONDS} 840 ', script)

    def test_a_full_build_is_opt_in_and_supersedes_the_named_targets(self):
        # This is the build gate's own flag. A regression gate must never be
        # satisfiable by a build, which is why the branch is opt-in per recipe.
        shutil.rmtree(self.source / 'build')
        tools_dir = self.source / 'build-tools'
        tools_dir.mkdir()
        recorded = self.root / 'build-args'
        (tools_dir / 'build.sh').write_text(
            '#!/bin/bash\n'
            'printf "%s\\n" "$@" > ' + shlex.quote(str(recorded)) + '\n'
            'build=.\n'
            'while [ $# -gt 0 ]; do\n'
            '  if [ "$1" = --build ]; then build=$2; fi\n'
            '  shift\n'
            'done\n'
            'mkdir -p "$build/bin"\n'
            'for name in bishengir-opt bishengir-compile; do\n'
            '  printf "#!/bin/bash\\nexit 0\\n" > "$build/bin/$name"\n'
            '  chmod 755 "$build/bin/$name"\n'
            'done\n')
        (tools_dir / 'build.sh').chmod(0o755)
        self.stage_ccec()
        self.stage_third_party()
        code = self.run_host('--timeout', '840', '--full-build-timeout', '21600',
                             '--allow-full-build', 'bishengir-opt')
        self.assertEqual(code, 0)
        # Ninja is never run against a tree that has no build.ninja.
        self.assertFalse(self.ninja_log.exists())
        self.assertTrue((self.source / 'build/bin/bishengir-opt').is_file())
        # The named targets are NOT forwarded: a flag-shaped target would reach
        # build.sh as an option. The from-zero command builds every target.
        argv = recorded.read_text().split()
        self.assertNotIn('bishengir-opt', argv)
        self.assertIn('--build', argv)


class HostParityCase(unittest.TestCase):
    """Host mode is Docker mode with the transport removed, proven by identity.

    Not "the two agree today" but "there is one implementation": each of these
    names must resolve to the SAME object the pinned sibling defines. A copy -
    even a byte-identical one - would pass a behavioural test this morning and
    drift the first time either side was edited, so identity is the assertion.
    """

    SHARED = ('BridgeError', 'FILE_KEYS', '_parse_env_file', 'machine_values',
              'machine_value', 'build_subdir', 'ninja_jobs',
              'toolchain_from_env_file', 'require_conda_pair',
              'timeout_seconds', 'init_snippet', 'init_args', 'container_script')

    # The one rule this file re-words, and the reason it may: the sibling's
    # refusal says "absolute container path" (docker_gate.py:709), which is wrong
    # on a route with no container. The RULE must stay the sibling's, so the
    # wrapper is asserted to delegate rather than to reimplement.
    REREWORDED = 'ccec_bin'

    def test_every_shared_name_is_the_siblings_own_object(self):
        for name in self.SHARED:
            with self.subTest(name=name):
                self.assertTrue(hasattr(host_bridge, name), name)
                self.assertIs(getattr(host_bridge, name),
                              getattr(host_bridge.shared, name),
                              name + ' is a copy, not the pinned implementation')
        # And the sibling it loaded is the pinned file, resolved from this one.
        self.assertEqual(Path(host_bridge.shared.__file__).resolve(),
                         HOST_BRIDGE.resolve().with_name('docker_gate.py'))

    def test_the_one_rewording_still_delegates_the_rule_to_the_sibling(self):
        # A wrapper that re-derived the absoluteness check would be a second rule
        # for one key. It must call the sibling's function, and it must pass the
        # sibling's verdict through except for the noun it exists to correct.
        self.assertIsNot(getattr(host_bridge, self.REREWORDED),
                         getattr(host_bridge.shared, self.REREWORDED))
        self.assertIs(host_bridge._shared_ccec_bin,
                      host_bridge.shared.ccec_bin)

        seen = []
        original = host_bridge._shared_ccec_bin

        def recorder(*args, **kwargs):
            seen.append((args, kwargs))
            return original(*args, **kwargs)
        # This call must name its own machine file, like every other case in this
        # module: reading the packaged (git-ignored) env.sh would make the outcome
        # depend on the developer's machine - a real CCEC_BIN there, or a stray
        # export in the launching shell, would fail this case for an unrelated
        # reason. The value is also the assertion: the wrapper must return the
        # sibling's result unchanged, not re-derive it.
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        chosen = Path(directory) / 'delegation.env'
        chosen.write_text('ASCENDNPU_CCEC_BIN=/opt/ccec/bin' + chr(10))
        with patch.dict(os.environ, {'ASCENDNPU_ENV_FILE': str(chosen)}, clear=True):
            with patch.object(host_bridge, '_shared_ccec_bin', recorder):
                self.assertEqual(host_bridge.ccec_bin(), Path('/opt/ccec/bin'))
        self.assertEqual(len(seen), 1, 'the wrapper did not delegate to the sibling')

        # The reworded branch: a relative value, still refused, host wording.
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        relative = Path(directory) / 'machine.env'
        relative.write_text('ASCENDNPU_CCEC_BIN=relative/bin\n')
        with patch.dict(os.environ, {'ASCENDNPU_ENV_FILE': str(relative)}, clear=True):
            with self.assertRaises(host_bridge.BridgeError) as caught:
                host_bridge.ccec_bin()
        self.assertIn('absolute host path', caught.exception.message)
        self.assertNotIn('container path', caught.exception.message)
        # ... and the Docker route on the same file still says "container", because
        # the rewording is a difference between ROUTES, not a change to the rule.
        # Asserting only the Host side would let a future edit make both routes say
        # "host path" and pass, leaving the Docker message wrong for its own machine.
        with patch.dict(os.environ, {'ASCENDNPU_ENV_FILE': str(relative)}, clear=True):
            with self.assertRaises(bridge.BridgeError) as docker:
                bridge.ccec_bin()
        self.assertIn('absolute container path', docker.exception.message)
        self.assertNotIn('host path', docker.exception.message)

        # Any OTHER refusal from that function keeps its own wording: the wrapper
        # corrects one sentence, it does not relabel every failure.
        def other(*args, **kwargs):
            raise host_bridge.BridgeError('some unrelated refusal')
        with patch.object(host_bridge, '_shared_ccec_bin', other):
            with self.assertRaises(host_bridge.BridgeError) as caught:
                host_bridge.ccec_bin()
        self.assertIn('some unrelated refusal', caught.exception.message)

    def test_the_host_gate_never_calls_the_container_transport(self):
        # Reusing the parser is the point; reusing the transport is the bug. The
        # names are absent from this module's globals, and a source scan keeps it
        # that way even if a later edit reaches them through `shared.`.
        for name in ('docker_exec_prefix', 'container_mounts', 'verify_same_tree',
                     'identity_probe', 'container_identity', 'map_host_to_container'):
            with self.subTest(name=name):
                self.assertNotIn(name, vars(host_bridge))
                self.assertNotIn(name, HOST_BRIDGE.read_text())

    def test_the_config_checks_never_spawn_docker(self):
        # The whole point of the Host route: it must work on a machine that
        # cannot reach the socket. Every argv the preflight issues is recorded and
        # none of them may name docker - a `docker inspect` here would make the
        # diagnostic refuse on exactly the machine it exists for.
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        source = root / 'source'
        (source / 'build').mkdir(parents=True)
        (source / 'build/build.ninja').write_text('# fake\n')
        for name in ('third-party/llvm-project', 'third-party/shmem'):
            tree = source / name
            tree.mkdir(parents=True)
            (tree / 'CMakeLists.txt').write_text('# source\n')
        tools = HostToolchain(root, source)
        tools.write()
        machine = root / 'env.sh'
        ccec = root / 'ccec'
        ccec.mkdir()
        for name in ('ccec', 'llvm-link'):
            (ccec / name).write_text('#!/bin/bash\nexit 0\n')
            (ccec / name).chmod(0o755)
        machine.write_text('ASCENDNPU_CANN_SETUP=\nASCENDNPU_CCEC_BIN=' + str(ccec) + '\n')
        saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))
        origin = os.getcwd()
        self.addCleanup(os.chdir, origin)
        os.chdir(source)
        real_run = host_bridge.subprocess.run
        issued = []
        def record(argv, *args, **kwargs):
            issued.append(list(argv))
            return real_run(argv, *args, **kwargs)
        with patch.object(host_bridge.subprocess, 'run', side_effect=record):
            os.environ.update({'PATH': str(tools.path), 'HOME': str(root),
                               'ASCENDNPU_ENV_FILE': str(machine)})
            for mode in ('build', 'regression'):
                self.assertEqual(host_bridge.main(['--check-config-' + mode]), 0, mode)
        self.assertTrue(issued, 'the preflight issued no command at all')
        # No argv may name docker as a WORD. Matched per element rather than on
        # the joined text: the probe payload inlines the committed init script,
        # whose comments legitimately mention docker_gate.py as that file's
        # other caller, and a substring test would fail on documentation.
        for argv in issued:
            self.assertNotIn('docker', [part.lower() for part in argv], argv[:2])
        # The only command a config check may run is the bash probe script. Its
        # -c text inlines the committed init script, whose COMMENT mentions
        # docker_gate.py - that is documentation, not transport, so the check
        # above is on the argv that would be executed, not on the payload text.
        self.assertEqual({argv[0] for argv in issued}, {'bash'})
        for argv in issued:
            self.assertEqual(argv[:2], ['bash', '-c'])
            self.assertEqual(argv[-1], 'bash')

    def test_the_generated_script_is_the_shared_generators_own_text(self):
        # Same argv, same tree probe, same timeout chain, same from-zero command:
        # only the source root differs, because the Host root IS the tree.
        arguments = dict(full_timeout=21600, ccec_bin=Path('/opt/ccec/bin'),
                         targets=('bishengir-opt',), allow_full_build=True)
        self.assertEqual(
            host_bridge.container_script('/src', 'build', '64', 840, None, None,
                                        None, **arguments),
            bridge.container_script('/src', 'build', '64', 840, None, None,
                                    None, **arguments))

class InitScriptCase(unittest.TestCase):
    """Run the committed init script the way a human does: sourced, no arguments.

    ScriptExecutionCase covers the Gate, which inlines the text and passes three
    validated values. These cases cover the other caller: the values come from
    `source env.sh`, the state the script builds must stay in the caller's shell,
    and a failure must return a status instead of ending that shell.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        for name in ('cmake', 'ninja'):
            tool = self.tools / name
            tool.write_text('#!/bin/bash\nexit 0\n')
            tool.chmod(0o755)
        # Fake CANN and Conda setup files that record what the script did.
        self.cann = self.root / 'set_env.sh'
        self.cann.write_text('export ASCEND_HOME_PATH=/cann\n')
        self.conda_sh = self.root / 'conda.sh'
        self.conda_sh.write_text('conda() { export CONDA_DEFAULT_ENV="$2"; }\n')

    def _run(self, script):
        env = {'HOME': str(self.root), 'PATH': f'{self.tools}:/usr/bin:/bin'}
        return subprocess.run([shutil.which('bash'), '-c', script],
                              capture_output=True, text=True, env=env)

    def source(self, body='', *args):
        """Source the init script with `args`, then report the state it built."""
        quoted = ' '.join(shlex.quote(arg) for arg in args)
        script = (
            body
            + f'. {shlex.quote(str(bridge.INIT_SCRIPT))} {quoted}\n'
            + 'rc=$?\n'
            + 'printf "rc=%s cann=%s conda_env=%s conda_activated=%s cmake=%s\\n" "$rc" '
            '"${ASCENDNPU_CANN_SETUP:-}" "${ASCENDNPU_CONDA_ENV:-}" '
            '"${CONDA_DEFAULT_ENV:-}" "$(command -v cmake)"\n')
        return self._run(script)

    def test_values_from_env_sh_are_used_without_arguments(self):
        env_file = self.root / 'env.sh'
        env_file.write_text(f'ASCENDNPU_CANN_SETUP={self.cann}\n'
                            f'ASCENDNPU_CONDA_SH={self.conda_sh}\n'
                            'ASCENDNPU_CONDA_ENV=env311\n')
        done = self.source(f'. {shlex.quote(str(env_file))}\n')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=0', done.stdout)
        # The activation ran, it was not merely read from the file.
        self.assertIn('conda_activated=env311', done.stdout)
        self.assertIn(f'cmake={self.tools}/cmake', done.stdout)

    def test_a_dash_disables_an_optional_component(self):
        done = self.source('', '-', '-', '-')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=0', done.stdout)
        self.assertIn('cann= conda_env=', done.stdout)

    def test_half_a_conda_configuration_is_refused(self):
        # `source()` ends with its report line, so the status to judge is the one
        # the sourced call left behind, not the shell's final exit.
        done = self.source('', '-', str(self.conda_sh), '')
        self.assertIn('rc=1', done.stdout)
        self.assertIn('must be provided together', done.stderr)

    def test_unreadable_cann_is_refused(self):
        done = self.source('', str(self.root / 'missing.sh'), '-', '-')
        self.assertIn('rc=1', done.stdout)
        self.assertIn('CANN setup not readable', done.stderr)

    def test_a_failure_returns_instead_of_ending_the_callers_shell(self):
        # The report line after the sourced call only prints if the shell is
        # still alive, which is what makes this usable from an interactive session.
        script = (f'. {shlex.quote(str(bridge.INIT_SCRIPT))} missing.sh - -\n'
                  'rc=$?\n'
                  'printf "survived rc=%s\\n" "$rc"\n')
        done = self._run(script)
        self.assertIn('survived rc=1', done.stdout)
        self.assertIn('CANN setup not readable', done.stderr)

    def test_the_gate_ships_this_file_and_not_a_second_copy(self):
        script = bridge.container_script('/src', 'build', '4', 840,
                                         Path('/cann/set_env.sh'), None, None)
        committed = bridge.INIT_SCRIPT.read_text(encoding='utf-8').rstrip('\n')
        self.assertIn(committed, script)
        self.assertIn(f'init_toolchain {shlex.quote("/cann/set_env.sh")} - -', script)


class FullBuildCase(unittest.TestCase):
    """scripts/full-build.sh creates the fresh build tree the Gate then gates.

    The Gate does not call this file, so these cases run it the way a human does:
    sourced, with no auto-run, and with the values read from the calling shell.
    The build itself is a fake `build-tools/build.sh` that records the argv it is
    given, so the command is checked without compiling anything.
    """

    SCRIPT = ADAPTER / 'scripts/full-build.sh'

    EXPECTED_ARGV = [
        '--c-compiler', 'clang', '--cxx-compiler', 'clang++', '--enable-lld',
        '--build-type', 'Release', '--enable-assertion', '-t',
        '--bisheng-compiler', '/opt/ccec/bin', '--disable-werror',
        '--disable-bishengir-werror', '--build-triton', '--build-shmem-template',
        '--build', './build-fresh', '-j', '8',
    ]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'src'
        (self.source / 'build-tools').mkdir(parents=True)
        self.record = self.root / 'argv.txt'
        self.out = self.root / 'out.txt'

    def write_build_script(self, status=0):
        (self.source / 'build-tools' / 'build.sh').write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\n" "$@" > "$FULL_BUILD_ARGV"\n'
            f'exit {status}\n')

    def call(self, args='full_build', **overrides):
        """Source the file, call `args`, and report the status left behind."""
        script = (f'. {shlex.quote(str(self.SCRIPT))}\n'
                  f'{args} > {shlex.quote(str(self.out))} 2>&1\n'
                  'rc=$?\n'
                  'pf=off\n'
                  'shopt -o pipefail | grep -q on && pf=on\n'
                  'printf "rc=%s pipefail=%s\\n" "$rc" "$pf"\n')
        env = {'HOME': str(self.root), 'PATH': '/usr/bin:/bin',
               'FULL_BUILD_ARGV': str(self.record),
               'ASCENDNPU_CCEC_BIN': '/opt/ccec/bin',
               'ASCENDNPU_BUILD_SUBDIR': 'build-fresh',
               'ASCENDNPU_NINJA_JOBS': '8'}
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return subprocess.run([shutil.which('bash'), '-c', script],
                              capture_output=True, text=True, env=env,
                              cwd=str(self.source))

    def recorded_argv(self):
        return self.record.read_text().splitlines() if self.record.exists() else None

    def test_sourcing_defines_the_function_and_never_builds(self):
        # Sourcing must not start a build: `init-toolchain.sh` has an auto-run
        # guard, this file deliberately has none.
        script = (f'. {shlex.quote(str(self.SCRIPT))}\n'
                  'declare -F full_build\n')
        done = subprocess.run([shutil.which('bash'), '-c', script],
                              capture_output=True, text=True,
                              env={'HOME': str(self.root), 'PATH': '/usr/bin:/bin'})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('full_build', done.stdout)
        self.assertIsNone(self.recorded_argv())

    def test_the_documented_command_is_passed_to_build_sh(self):
        self.write_build_script(0)
        done = self.call()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=0 pipefail=off', done.stdout)
        self.assertEqual(self.recorded_argv(), self.EXPECTED_ARGV)

    def test_a_missing_ccec_bin_is_refused_before_the_build_runs(self):
        done = self.call(ASCENDNPU_CCEC_BIN=None)
        self.assertIn('rc=1 pipefail=off', done.stdout)
        self.assertIn('ASCENDNPU_CCEC_BIN is not set', self.out.read_text())
        self.assertIsNone(self.recorded_argv())

    def test_a_source_root_without_build_sh_is_refused(self):
        empty = self.root / 'elsewhere'
        empty.mkdir()
        done = self.call(args=f'full_build {shlex.quote(str(empty))}')
        self.assertIn('rc=1 pipefail=off', done.stdout)
        self.assertIn('no build-tools/build.sh', self.out.read_text())
        self.assertIsNone(self.recorded_argv())

    def test_a_failing_build_status_is_returned_and_options_are_kept(self):
        # `pipefail` must stay off in the calling shell: the pipeline runs in a
        # subshell, so sourcing this file cannot alter an interactive session.
        self.write_build_script(7)
        done = self.call()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('rc=7 pipefail=off', done.stdout)
        self.assertIn('build failed with status 7', self.out.read_text())

    def test_a_missing_source_root_is_refused(self):
        self.write_build_script(0)
        done = self.call(args=f'full_build {shlex.quote(str(self.root / "absent"))}')
        self.assertIn('rc=1 pipefail=off', done.stdout)
        self.assertIn('no such source root', self.out.read_text())
        self.assertIsNone(self.recorded_argv())

    def test_extra_flags_are_forwarded_after_the_owned_ones(self):
        # A human may add a flag the Adapter does not model; the helper's own
        # flags stay in their fixed order so the argv remains reviewable.
        self.write_build_script(0)
        done = self.call(args='full_build "$PWD" --skip-hivmc')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.recorded_argv(), self.EXPECTED_ARGV + ['--skip-hivmc'])

    def test_a_flag_the_helper_owns_cannot_be_repeated(self):
        for flag in ('-j', '--build', '--bisheng-compiler', '-t', '--enable-lld',
                     '--build-triton', '--c-compiler'):
            self.write_build_script(0)
            done = self.call(args=f'full_build "$PWD" {flag} 99')
            self.assertIn('rc=1 pipefail=off', done.stdout, flag)
            self.assertIn('may not be repeated', self.out.read_text(), flag)
            self.assertIsNone(self.recorded_argv(), flag)

    def test_a_rebuild_request_is_refused_because_it_deletes_the_tree(self):
        # `-r` makes build.sh run clean_build_dir. No Adapter may delete a tree it
        # did not create, so the flag never reaches it from here.
        self.write_build_script(0)
        done = self.call(args='full_build "$PWD" -r')
        self.assertIn('rc=1 pipefail=off', done.stdout)
        self.assertIn('would delete', self.out.read_text())
        self.assertIsNone(self.recorded_argv())

    def test_an_existing_log_is_never_overwritten(self):
        # A previous run's log is the only record of that run. The new log gets a
        # timestamp instead of replacing it.
        previous = self.source / 'build.log'
        previous.write_text('previous run\n')
        self.write_build_script(0)
        done = self.call()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(previous.read_text(), 'previous run\n')
        logs = sorted(path.name for path in self.source.glob('build-*.log'))
        self.assertEqual(len(logs), 1, logs)
        self.assertRegex(logs[0], r'^build-\d{8}T\d{6}Z\.log$')

    def test_a_dash_log_path_writes_nothing_into_the_tree(self):
        # The Gate uses this: a gate may not leave a file in the source tree.
        self.write_build_script(0)
        done = self.call(ASCENDNPU_FULL_BUILD_LOG='-')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertFalse((self.source / 'build.log').exists())
        self.assertEqual(self.recorded_argv(), self.EXPECTED_ARGV)


class CreateContainerCase(unittest.TestCase):
    """create-container.sh must reject an identity that cannot read its own HOME.

    `entry.sh` never chowns HOME (it is normally the host home directory), so a
    UID/GID that does not match the host owner leaves `~/.bashrc` unreadable and
    a login shell fails with "Permission denied" only after the operator is
    already inside the container. The check runs in --dry-run too, so it costs
    nothing to verify here.

    `stat` reports the real ownership of the staged directory, so the check is
    exercised against whatever uid/gid owns the temporary directory, with no
    root and no container needed.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / 'home' / 'user'
        self.home.mkdir(parents=True)
        self.script = ADAPTER / 'docker/create-container.sh'

    def config(self, uid, gid, *, home_mode=None, bind=None, extra=''):
        if home_mode is not None:
            os.chmod(self.home, home_mode)
        bind = bind if bind is not None else f'{self.root}/home:/home'
        path = self.root / 'config.env'
        path.write_text(
            'IMAGE=ubuntu:24.04\n'
            'CONTAINER_NAME=probe\n'
            'CONTAINER_USER=probe\n'
            f'CONTAINER_UID={uid}\n'
            f'CONTAINER_GID={gid}\n'
            f'CONTAINER_HOME=/home/user\n'
            f'BIND_MOUNTS="{bind}"\n'
            f'{extra}')
        return path

    def run_script(self, config):
        return subprocess.run([shutil.which('bash'), str(self.script),
                               '--config', str(config), '--dry-run'],
                              capture_output=True, text=True)

    def test_owner_uid_and_gid_pass(self):
        uid = self.home.stat().st_uid
        gid = self.home.stat().st_gid
        done = self.run_script(self.config(uid, gid))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn('cannot read CONTAINER_HOME', done.stderr)

    def test_a_mismatched_uid_is_refused_with_the_real_owner(self):
        # 750 (the common useradd mode) denies both traversal and read to a
        # non-owner, which is exactly the reported failure.
        uid = self.home.stat().st_uid
        gid = self.home.stat().st_gid
        done = self.run_script(self.config(uid + 1, gid + 1, home_mode=0o750))
        self.assertEqual(done.returncode, 2, done.stdout)
        self.assertIn('cannot read CONTAINER_HOME', done.stderr)
        self.assertIn(f'uid={uid} gid={gid}', done.stderr)
        self.assertIn(f'UID={uid + 1} GID={gid + 1}', done.stderr)
        # The message must tell the operator to fix config.env, not the host home.
        self.assertIn('Do not chown the host home directory', done.stderr)

    def test_an_owner_uid_with_a_different_gid_passes(self):
        # The check is about the effective owner, not about both fields matching.
        uid = self.home.stat().st_uid
        gid = self.home.stat().st_gid
        done = self.run_script(self.config(uid, gid + 1))
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_a_matching_supplementary_gid_passes(self):
        uid = self.home.stat().st_uid
        gid = self.home.stat().st_gid
        done = self.run_script(self.config(uid + 1, gid + 1, home_mode=0o750,
                                           extra=f'SUPPLEMENTARY_GIDS="{gid}"\n'))
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_a_world_readable_home_passes_for_any_uid(self):
        uid = self.home.stat().st_uid
        gid = self.home.stat().st_gid
        done = self.run_script(self.config(uid + 1, gid + 1, home_mode=0o755))
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_home_outside_every_mount_is_not_checked(self):
        uid = self.home.stat().st_uid
        gid = self.home.stat().st_gid
        data = self.root / 'data'
        data.mkdir()
        done = self.run_script(self.config(uid + 1, gid + 1, home_mode=0o750,
                                           bind=f'{data}:/data'))
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_a_remapped_mount_checks_the_host_source_it_points_at(self):
        # HOME is remapped to a different host directory, and that directory is
        # what the check must inspect.
        uid = self.home.stat().st_uid
        gid = self.home.stat().st_gid
        done = self.run_script(self.config(
            uid + 1, gid + 1, home_mode=0o750,
            bind=f'{self.root}/home/user:/home/user'))
        self.assertEqual(done.returncode, 2, done.stdout)
        self.assertIn('cannot read CONTAINER_HOME', done.stderr)
        self.assertIn(f'uid={uid} gid={gid}', done.stderr)

    def test_the_deepest_mount_wins_when_home_is_covered_twice(self):
        # A broad, world-readable mount would pass; the deeper mount of HOME
        # itself must decide, and it is not readable by the configured UID.
        inner = self.root / 'inner'
        inner.mkdir()
        os.chmod(self.home, 0o755)
        os.chmod(inner, 0o750)
        uid = inner.stat().st_uid
        gid = inner.stat().st_gid
        done = self.run_script(self.config(
            uid + 1, gid + 1,
            bind=f'{self.root}/home:/home {inner}:/home/user'))
        self.assertEqual(done.returncode, 2, done.stdout)
        self.assertIn('cannot read CONTAINER_HOME', done.stderr)
        self.assertIn(str(inner), done.stderr)


if __name__ == '__main__':
    unittest.main()
