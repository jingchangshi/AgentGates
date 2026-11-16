"""Offline regressions for the host-local tilelang-npu Adapter.

This Adapter builds and tests a TileLang checkout on THIS machine: no container,
no root and no Docker. Almost everything it claims is therefore decided from
configuration, and these tests pin the properties that make those claims honest
without compiling anything or touching a card:

* the machine keys the Gate accepts are exactly the keys the engine leases on, so
  a key cannot become configuration one side reads and the other ignores;
* the build directory the engine locks is the directory the Gate writes, or the
  lock protects a tree nobody builds in;
* the machine file is data: nothing in it is expanded or executed;
* the shipped template names no machine path, so no other host layout ships;
* the manifest pins the bytes that ran and says which observations it gates on;
* the device id authorizes one run, so it is not a machine key;
* preflight answers a question about configuration and starts no build;
* the evaluator turns the Gate observations into the verdict the manifest
  promised.

Nothing here starts CMake, Ninja, a container or a device.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'agent_gates/adapters/tilelang-npu'
sys.path.insert(0, str(ROOT))

from agent_gates.evaluator import evaluate
from agent_gates import machine_env
from agent_gates.manifest import Manifest


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load('tilelang_npu_gate', ADAPTER / 'scripts/tilelang_npu_gate.py')

NL = chr(10)
BACKTICK = chr(96)


class MachineFileCase(unittest.TestCase):
    """The Gate and the engine accept exactly the same machine keys."""

    def test_the_script_and_the_engine_accept_the_same_keys(self):
        self.assertEqual(tuple(sorted(GATE.TILELANG_NPU_KEYS)),
                         tuple(sorted(machine_env.TILELANG_NPU_KEYS)))

    def test_the_shared_alias_is_the_same_key_set(self):
        # The repository suite and every other Adapter name this set
        # TOOLCHAIN_KEYS, while this Adapter own code names its prefix. An
        # alias that drifted would let one of the two be edited alone.
        self.assertEqual(tuple(GATE.TOOLCHAIN_KEYS), tuple(GATE.TILELANG_NPU_KEYS))

    def test_the_adapter_owns_its_own_prefix_selector_and_file(self):
        spec = machine_env.ADAPTERS['tilelang-npu']
        self.assertEqual(spec['prefix'], 'TILELANG_NPU_')
        self.assertEqual(spec['selector'], 'TILELANG_NPU_ENV_FILE')
        self.assertEqual(spec['directory'], 'tilelang-npu')
        self.assertEqual(spec['keys'], machine_env.TILELANG_NPU_KEYS)
        # It shares nothing with the two HIVM Adapters: a build directory it
        # leases must not be confused with the one the docker gates write.
        self.assertNotEqual(machine_env.ADAPTERS['tilelang-npu']['directory'],
                            machine_env.ADAPTERS['tilelang-hivm']['directory'])

    def test_the_engine_resolves_this_adapter_from_the_same_two_names(self):
        from agent_gates import engine
        self.assertEqual(engine.ENV_KEYS['tilelang-npu'], 'TILELANG_NPU_ENV_FILE')
        self.assertEqual(engine.OWN_PREFIX['tilelang-npu'], 'TILELANG_NPU_')

    def test_the_packaged_machine_file_sits_beside_the_template(self):
        # There is no home-directory search: a machine that keeps its file
        # elsewhere names it in TILELANG_NPU_ENV_FILE, so no Gate reads a file
        # nobody reviewed. The packaged location is the only default.
        expected = ADAPTER / 'env.sh'
        found = GATE.default_env_file()
        if expected.is_file():
            self.assertEqual(Path(found), expected)
        else:
            self.assertIsNone(found, found)

    def test_the_launch_names_are_not_machine_keys(self):
        # The selector and the device id are launch parameters. Naming them is
        # what lets the parser refuse an unknown key without turning these two
        # into file configuration that describes a machine permanently.
        self.assertEqual(GATE.NOT_MACHINE_KEYS,
                         frozenset({'TILELANG_NPU_ENV_FILE',
                                    'TILELANG_NPU_DEVICE_ID'}))
        self.assertEqual(set(GATE.TILELANG_NPU_KEYS) & set(GATE.NOT_MACHINE_KEYS), set())
        # The two launch names are named so the parser can REFUSE them by name,
        # not so it can accept them. A file that accepted a device id would turn
        # a per-run authorization into a standing entitlement, and the comment on
        # NOT_MACHINE_KEYS would describe the opposite of the code.
        self.assertEqual(GATE.ALLOWED, set(GATE.TILELANG_NPU_KEYS))
        for name in GATE.NOT_MACHINE_KEYS:
            self.assertNotIn(name, GATE.ALLOWED)
            with self.assertRaises(GATE.BridgeError) as error:
                GATE._parse_env_file(f'{name}=1\n', Path('/nonexistent/machine.env'))
            self.assertIn('launch value', str(error.exception))


class StrictGrammarCase(unittest.TestCase):
    """The file is data: no execution, no expansion, no ambiguity."""

    KEY = 'TILELANG_NPU_BUILD_SUBDIR'
    PATH = Path('/nonexistent/machine.env')

    def both(self, text):
        """Parse with both readers and require one answer."""
        values = machine_env.parse(text, self.PATH, machine_env.TILELANG_NPU_KEYS)
        self.assertEqual(GATE._parse_env_file(text, self.PATH), values, repr(text))
        return values

    def refused(self, text):
        # The engine and the Gate have separate parsers on purpose, so a file one
        # accepts and the other rejects would let a run lease a tree the Gate then
        # refuses to build in - or build in one the engine never leased.
        with self.assertRaises(machine_env.MachineEnvError, msg=repr(text)):
            machine_env.parse(text, self.PATH, machine_env.TILELANG_NPU_KEYS)
        with self.assertRaises(GATE.BridgeError, msg=repr(text)):
            GATE._parse_env_file(text, self.PATH)

    def test_both_readers_accept_the_same_plain_file(self):
        for text in ('# comment' + NL + NL + self.KEY + '=build' + NL,
                     self.KEY + '="x y"' + NL,
                     self.KEY + "='x y'" + NL,
                     self.KEY + '=' + NL,
                     self.KEY + '=build-release' + NL,
                     self.KEY + '=a=b' + NL):
            values = self.both(text)
            self.assertIn(self.KEY, values)

    def test_a_value_is_never_executed_or_expanded(self):
        # The reason the grammar is strict: a machine file is reviewed as data.
        # If a reader evaluated it, a copied file could run commands on the host
        # and the value a Gate used would not be the one a human read.
        for value in ('$(id -u)', BACKTICK + 'id -u' + BACKTICK,
                      '$' + '{HOME}', '$HOME/build'):
            values = self.both(self.KEY + '=' + value + NL)
            self.assertEqual(values[self.KEY], value)

    def test_an_unknown_key_is_refused_by_name_on_both_sides(self):
        text = 'TILELANG_NPU_TYPO=x' + NL
        with self.assertRaises(machine_env.MachineEnvError) as error:
            machine_env.parse(text, self.PATH, machine_env.TILELANG_NPU_KEYS)
        self.assertIn('TILELANG_NPU_TYPO', str(error.exception))
        with self.assertRaises(GATE.BridgeError) as error:
            GATE._parse_env_file(text, self.PATH)
        self.assertIn('TILELANG_NPU_TYPO', str(error.exception))

    def test_a_duplicate_key_is_refused_on_both_sides(self):
        self.refused(self.KEY + '=one' + NL + self.KEY + '=two' + NL)

    def test_a_malformed_line_is_refused_on_both_sides(self):
        for text in (self.KEY + NL, self.KEY + ' =build' + NL,
                     self.KEY + ':build' + NL,
                     '1' + self.KEY + '=build' + NL,
                     'export ' + self.KEY + '=build' + NL):
            self.refused(text)

    def test_quotes_are_stripped_but_must_balance(self):
        self.assertEqual(self.both(self.KEY + '="x y"' + NL), {self.KEY: 'x y'})
        for text in (self.KEY + '="x' + NL, self.KEY + "=x'" + NL):
            self.refused(text)

    def test_comments_and_blank_lines_are_ignored(self):
        self.assertEqual(self.both('# c' + NL + NL + self.KEY + '=1' + NL),
                         {self.KEY: '1'})


class BuildDirectoryCase(unittest.TestCase):
    """The engine leases the directory the Gate builds in.

    The Gate reads the name through build_subdir(values) - the parsed machine
    file and nothing else - so one file decides both the lease and the build.
    """

    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        for name in list(os.environ):
            if name.startswith('TILELANG_NPU_'):
                os.environ.pop(name)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'tilelang'
        self.source.mkdir()
        self.manifest = Manifest(ADAPTER / 'manifest.json')

    def values(self, text):
        """Parse text the way the Gate will, from a file it could really read."""
        path = self.root / 'machine.env'
        path.write_text(text)
        os.environ['TILELANG_NPU_ENV_FILE'] = str(path)
        return path, GATE._parse_env_file(text, path)

    def lines(self, **values):
        return ''.join('%s=%s%s' % (key, value, NL) for key, value in values.items())

    def test_the_engine_and_the_gate_read_the_same_directory(self):
        from agent_gates import engine
        for subdir in ('build', 'build-release', 'out.dir'):
            path, values = self.values(self.lines(TILELANG_NPU_BUILD_SUBDIR=subdir))
            self.assertEqual(GATE.build_subdir(values), subdir)
            self.assertEqual(engine.build_directory(self.source, self.manifest, os.environ),
                             (self.source / subdir).resolve(), subdir)
            machine_env.read('tilelang-npu', env_file=path)

    def test_the_default_build_directory_is_the_engines_default(self):
        # The two readers must agree about the DEFAULT too, not only about a
        # configured value: a Gate building in its default tree while the engine
        # leases another one is the same defect with no key to blame.
        path, values = self.values('')
        default = machine_env.read('tilelang-npu', env_file=path).build_subdir
        self.assertEqual(GATE.build_subdir(values), default)
        self.assertIsNotNone(machine_env.SAFE_NAME.fullmatch(default))
        self.assertEqual(dict(values), {})

    def test_an_empty_value_is_refused_by_both_readers(self):
        # An empty value is a typo, not "use the default": the file spells unset
        # by leaving the line out. Accepting it would let a run lease a directory
        # the Gate then refuses to build in.
        path, values = self.values(self.lines(TILELANG_NPU_BUILD_SUBDIR=''))
        with self.assertRaises(machine_env.MachineEnvError):
            machine_env.read('tilelang-npu', env_file=path)
        with self.assertRaises(GATE.BridgeError):
            GATE.build_subdir(values)

    def test_a_path_like_subdirectory_never_becomes_a_lease(self):
        # A separator or dot-dot resolves outside the source tree, so the lease
        # and the build would describe different directories. BOTH readers refuse
        # it: the engine decides which tree it leases, but the Gate decides which
        # tree it writes, and relying on the caller would leave a path-traversal
        # build reachable for anyone who runs the script directly.
        from agent_gates import engine
        for value in ('..', '.', 'a/b', '/build', '../etc'):
            path, values = self.values(self.lines(TILELANG_NPU_BUILD_SUBDIR=value))
            with self.assertRaises(machine_env.MachineEnvError, msg=repr(value)):
                machine_env.read('tilelang-npu', env_file=path)
            with self.assertRaises(GATE.BridgeError, msg=repr(value)):
                GATE.build_subdir(values)
            with self.assertRaises(ValueError, msg=repr(value)):
                engine.build_directory(self.source, self.manifest, os.environ)

    def test_a_plain_name_is_still_accepted_by_the_gate(self):
        # The counterpart, so the refusal above cannot be satisfied by a Gate that
        # rejects every name: an ordinary directory must keep working.
        for value in ('build', 'build-main', 'out_build'):
            path, values = self.values(self.lines(TILELANG_NPU_BUILD_SUBDIR=value))
            self.assertEqual(GATE.build_subdir(values), value)
            machine_env.read('tilelang-npu', env_file=path)

    def test_a_contradicting_export_cannot_move_the_lock_off_the_build(self):
        # The engine resolves the file, so a stale export cannot point the lease
        # at a tree the Gate is not writing.
        from agent_gates import engine
        self.values(self.lines(TILELANG_NPU_BUILD_SUBDIR='build'))
        os.environ['TILELANG_NPU_BUILD_SUBDIR'] = 'build-fast'
        self.assertEqual(engine.build_directory(self.source, self.manifest, os.environ),
                         (self.source / 'build').resolve())


class NinjaJobsCase(unittest.TestCase):
    """How many compilers one build starts is a machine fact."""

    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        for name in list(os.environ):
            if name.startswith('TILELANG_NPU_'):
                os.environ.pop(name)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def values(self, text):
        path = self.root / 'machine.env'
        path.write_text(text)
        os.environ['TILELANG_NPU_ENV_FILE'] = str(path)
        return path, GATE._parse_env_file(text, path)

    def test_the_default_job_count_is_the_machines_cpu_count(self):
        # The substitution stays a literal: the count is resolved by the host
        # that runs the build, so it describes the machine doing the work rather
        # than the one that read the file.
        path, values = self.values('')
        self.assertEqual(machine_env.read('tilelang-npu', env_file=path).ninja_jobs(),
                         '$(nproc)')
        self.assertEqual(GATE.ninja_jobs(values), '$(nproc)')

    def test_a_configured_job_count_is_used_by_both_readers(self):
        path, values = self.values('TILELANG_NPU_NINJA_JOBS=32' + NL)
        self.assertEqual(machine_env.read('tilelang-npu', env_file=path).ninja_jobs(), '32')
        self.assertEqual(GATE.ninja_jobs(values), '32')

    def test_a_non_positive_or_non_integer_count_is_refused(self):
        # A job count of 0 or the word sixty is not a slow build but a broken
        # one: Ninja fails after the tree was configured, which loses the
        # configure and reports the failure far from its cause.
        for value in ('0', '-1', 'sixty', '1.5', ''):
            path, values = self.values('TILELANG_NPU_NINJA_JOBS=%s%s' % (value, NL))
            with self.assertRaises(machine_env.MachineEnvError, msg=repr(value)):
                machine_env.read('tilelang-npu', env_file=path).ninja_jobs()
            with self.assertRaises(GATE.BridgeError, msg=repr(value)):
                GATE.ninja_jobs(values)


class TemplateCase(unittest.TestCase):
    """The committed template configures nothing, so it ships no machine."""

    TEMPLATE = ADAPTER / 'env.example.sh'

    def values(self):
        return machine_env.parse(self.TEMPLATE.read_text(), self.TEMPLATE,
                                 machine_env.TILELANG_NPU_KEYS)

    def test_the_template_is_data_both_readers_accept(self):
        text = self.TEMPLATE.read_text()
        self.assertEqual(GATE._parse_env_file(text, self.TEMPLATE), self.values())

    def test_the_template_defines_every_machine_key(self):
        # A key the template omits is a key an operator will not know exists, and
        # a Gate that reads it would run with a value nobody set.
        self.assertEqual(sorted(self.values()), sorted(machine_env.TILELANG_NPU_KEYS))

    def test_every_value_is_empty(self):
        # A committed default would point another host at a path that does not
        # exist there, or worse, at one that does and belongs to somebody else.
        for key, value in self.values().items():
            self.assertEqual(value, '', '%s ships the value %r' % (key, value))

    def test_the_template_carries_no_machine_specific_path(self):
        # Empty values are the rule above; this catches the other shape of the
        # same leak, a path left in a comment for somebody to copy.
        text = self.TEMPLATE.read_text()
        for forbidden in ('/home/', '/data/', 's00653124'):
            self.assertNotIn(forbidden, text, forbidden)

    def test_the_template_is_not_the_live_machine_file(self):
        # env.sh holds this host paths and is git-ignored; only the empty
        # template is committed. Shipping the live file would ship a machine.
        live = ADAPTER / 'env.sh'
        if live.is_file():
            self.assertFalse(self.TEMPLATE.samefile(live))


class DeviceAuthorizationCase(unittest.TestCase):
    """A device id authorizes one run; the machine file describes the machine."""

    def test_the_device_id_and_the_selector_are_not_machine_keys(self):
        # If the device id were a machine key, one file would authorize every run
        # on that machine. It stays a launch value, so the operator names the card
        # for each run and the receipt records that choice.
        for key in ('TILELANG_NPU_DEVICE_ID', 'TILELANG_NPU_ENV_FILE'):
            self.assertNotIn(key, machine_env.TILELANG_NPU_KEYS)
            self.assertNotIn(key, GATE.TILELANG_NPU_KEYS)
            self.assertIn(key, GATE.NOT_MACHINE_KEYS)


class CheckConfigCase(unittest.TestCase):
    """Preflight decides about configuration only; it builds and probes nothing."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Preflight requires the working directory to be a checkout:
        # CMakeLists.txt is how it tells a source tree from any directory.
        self.cmake_lists = self.root / 'CMakeLists.txt'
        self.cmake_lists.write_text('# fake' + NL)
        self.machine = self.root / 'machine.env'
        self.write_machine()

    def write_machine(self, **values):
        """A machine file whose required paths really exist.

        Preflight verifies rather than records, so a made-up path would make it
        refuse for the right reason and the test would prove nothing.
        """
        cann = self.root / 'set_env.sh'
        cann.write_text('export ASCEND_HOME_PATH=/cann' + NL)
        conda_sh = self.root / 'conda.sh'
        conda_sh.write_text('conda() { :; }' + NL)
        python = self.root / 'python3'
        python.write_text('#!/bin/sh' + NL + 'exit 0' + NL)
        python.chmod(0o755)
        filled = {'TILELANG_NPU_CANN_SETUP': str(cann),
                  'TILELANG_NPU_CONDA_SH': str(conda_sh),
                  'TILELANG_NPU_CONDA_ENV': 'an-env',
                  'TILELANG_NPU_CONDA_ENVS_PATH': str(self.root / 'envs'),
                  'TILELANG_NPU_PYTHON': str(python)}
        filled.update(values)
        self.machine.write_text(''.join('%s=%s%s' % (key, value, NL)
                                        for key, value in filled.items()))
        return self.machine

    def run_gate(self, *argv, env=None):
        environment = {'PATH': '/usr/bin:/bin', 'HOME': str(self.root),
                       'TILELANG_NPU_ENV_FILE': str(self.machine)}
        environment.update(env or {})
        return subprocess.run([sys.executable, '-B', str(GATE.__file__), *argv],
                              capture_output=True, text=True, env=environment,
                              cwd=str(self.root), timeout=300)

    def test_a_complete_configuration_returns_zero_and_spawns_nothing(self):
        # The property that makes this a preflight: it must not start the work it
        # is deciding about. Every process launch is refused here.
        import contextlib
        with patch.dict(os.environ, {'TILELANG_NPU_ENV_FILE': str(self.machine)},
                        clear=True), \
                patch.object(GATE.subprocess, 'run') as run, \
                contextlib.chdir(str(self.root)):
            self.assertEqual(GATE.main(['--check-config']), 0)
            run.assert_not_called()

    def test_the_real_process_exits_zero_and_starts_no_tool(self):
        # The same property from outside the module: cmake, ninja and ctest are
        # stand-ins on PATH that record any launch, so a preflight that began
        # configuring the tree is visible even though nothing is compiled.
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        marker = self.root / 'spawned'
        for tool in ('cmake', 'ninja', 'ctest'):
            stub = bin_dir / tool
            stub.write_text('#!/bin/sh' + NL + ': > '
                            + shlex.quote(str(marker)) + NL + 'exit 0' + NL)
            stub.chmod(0o755)
        started = time.monotonic()
        done = self.run_gate('--check-config',
                             env={'PATH': '%s:/usr/bin:/bin' % bin_dir})
        elapsed = time.monotonic() - started
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertFalse(marker.exists(), 'preflight started cmake, ninja or ctest')
        # A preflight that compiles is not a preflight, and a cold configure of a
        # TileLang checkout takes far longer than this bound.
        self.assertLess(elapsed, 120.0)

    def test_the_real_process_creates_no_build_tree(self):
        # Nothing may be written on the way to a verdict: a preflight runs in the
        # source directory, and a configuration it started there would be the very
        # work it exists to decide about.
        before = sorted(path.name for path in self.root.iterdir())
        done = self.run_gate('--check-config')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), before)

    def test_an_unknown_key_makes_preflight_exit_nonzero(self):
        self.write_machine(TILELANG_NPU_TYPO='x')
        done = self.run_gate('--check-config')
        # 75 is the code the runner maps to BLOCKED_ENV: nothing ran. A
        # configuration that cannot be read is a refusal, not a failed build.
        self.assertEqual(done.returncode, 75, done.stderr)
        self.assertIn('TILELANG_NPU_TYPO', done.stderr)

    def test_a_missing_required_key_is_refused(self):
        values = dict(line.split('=', 1)
                      for line in self.machine.read_text().splitlines() if '=' in line)
        for missing in GATE.REQUIRED:
            kept = dict(values)
            kept.pop(missing)
            self.machine.write_text(''.join('%s=%s%s' % (key, value, NL)
                                            for key, value in kept.items()))
            done = self.run_gate('--check-config')
            self.assertEqual(done.returncode, 75, (missing, done.stderr))
            self.assertIn(missing, done.stderr)
        self.write_machine()

    def test_a_missing_source_checkout_is_refused(self):
        # The Gate builds a checkout, so a directory without CMakeLists.txt is a
        # misconfiguration rather than something to discover after configuring.
        self.cmake_lists.unlink()
        done = self.run_gate('--check-config')
        self.assertEqual(done.returncode, 75, done.stderr)
        self.assertIn('CMakeLists.txt', done.stderr)


class ManifestCase(unittest.TestCase):
    """The manifest is the Gate contract, and it pins the bytes that ran."""

    def manifest(self):
        return json.loads((ADAPTER / 'manifest.json').read_text())

    def gates(self):
        return {item['id']: item for item in self.manifest()['gates']}

    def test_the_manifest_parses_and_its_id_is_its_directory(self):
        loaded = Manifest(ADAPTER / 'manifest.json')
        self.assertEqual(loaded.id, 'tilelang-npu')
        # A directory and an ID that disagree would let a run select a Pack whose
        # manifest describes something else.
        self.assertEqual(ADAPTER.name, loaded.id)
        self.assertEqual(loaded.raw['schemaVersion'], 1)
        self.assertIs(loaded.raw['permissions']['hostExecution'], True)
        self.assertIs(loaded.raw['permissions']['sourceWrite'], False)
        from agent_gates.registry import Registry
        self.assertEqual(Registry().get('tilelang-npu').id, 'tilelang-npu')

    def test_every_pinned_file_matches_its_digest(self):
        # The pin is what makes the executed bytes the reviewed bytes; an
        # unpinned edit would run code no receipt describes.
        files = self.manifest()['files']
        self.assertIn('scripts/tilelang_npu_gate.py', files)
        for name, digest in files.items():
            self.assertEqual(hashlib.sha256((ADAPTER / name).read_bytes()).hexdigest(),
                             digest, name)

    def test_every_recipe_checks_its_configuration_first(self):
        loaded = Manifest(ADAPTER / 'manifest.json')
        for recipe_id in loaded.recipes:
            recipe = loaded.resolve_recipe(recipe_id, ADAPTER)
            self.assertTrue(recipe.config_check, recipe_id)
            self.assertEqual(len([item for item in recipe.config_check
                                  if item.startswith('--check-config')]), 1, recipe_id)
            # The check must resolve to the pinned script, or a checkout edited
            # after pinning would be the code that answered the question.
            self.assertTrue([item for item in recipe.config_check
                             if Path(item).is_absolute() and Path(item).is_file()],
                            recipe_id)

    def test_the_examples_recipe_can_be_blocked_and_follows_the_build(self):
        loaded = Manifest(ADAPTER / 'manifest.json')
        # `recipes` is the raw declaration; `resolve_recipe` is the typed form the
        # engine acts on, and that is what must carry the block and the device
        # opt-out.
        examples = loaded.resolve_recipe('tilelang-npu.examples', ADAPTER)
        # 75 means nothing ran: a card that was not available must not be
        # recorded as a test that failed.
        self.assertIn(75, examples.blocked_exit_codes)
        self.assertIn(75, loaded.resolve_recipe('tilelang-npu.build',
                                                ADAPTER).blocked_exit_codes)
        self.assertEqual(loaded.gates['tilelang-npu-examples'].get('dependsOn'),
                         ['tilelang-npu-build'])
        self.assertEqual(loaded.order(['tilelang-npu-examples']),
                         ['tilelang-npu-build', 'tilelang-npu-examples'])

    def test_the_build_recipe_takes_no_device(self):
        # A build never drives a card, so it must not demand an authorized device
        # id for work that cannot use one.
        loaded = Manifest(ADAPTER / 'manifest.json')
        self.assertEqual(loaded.recipes['tilelang-npu.build'].get('device'), 'none')
        self.assertNotIn('deviceWindow', loaded.recipes['tilelang-npu.build'])
        self.assertIs(loaded.resolve_recipe('tilelang-npu.build', ADAPTER).uses_device,
                      False)

    def test_the_build_gate_publishes_exactly_what_a_build_can_observe(self):
        gate = self.gates()['tilelang-npu-build']
        self.assertEqual(gate['recipeId'], 'tilelang-npu.build')
        self.assertEqual(set(gate['observationChecks']),
                         {'exitCodeZero', 'nativeLibraryPresent',
                          'ascendBackendCompiled'})
        for name in ('nativeLibraryPresent', 'ascendBackendCompiled'):
            self.assertIs(gate['observationChecks'][name], True, name)

    def test_the_examples_gate_publishes_what_a_test_run_can_observe(self):
        checks = self.gates()['tilelang-npu-examples']['observationChecks']
        self.assertEqual(set(checks),
                         {'exitCodeZero', 'testsCollectedPositive', 'notAllSkipped',
                          'testsFailed'})
        self.assertIs(checks['testsCollectedPositive'], True)
        self.assertIs(checks['notAllSkipped'], True)
        # A count, not a boolean: no failures is a number of failures, and the
        # evaluator compares it as one.
        self.assertIs(type(checks['testsFailed']), int)
        self.assertEqual(checks['testsFailed'], 0)

    def test_the_two_gates_are_the_ones_the_adapter_registers(self):
        self.assertEqual(sorted(self.gates()),
                         ['tilelang-npu-build', 'tilelang-npu-examples'])


class EvaluatorCase(unittest.TestCase):
    """What the Gate observed becomes the verdict the manifest promised."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gates = Manifest(ADAPTER / 'manifest.json').gates

    def checks(self, gate):
        return self.gates[gate]['observationChecks']

    def process(self, observations, *, exitCode=0):
        path = self.root / 'stdout'
        path.write_text(json.dumps(observations))
        return {'exitCode': exitCode, 'timedOut': False, 'interrupted': False,
                'stdout': {'path': str(path)}}

    def build_report(self, **changes):
        checks = self.checks('tilelang-npu-build')
        report = {name: True for name in checks if name != 'exitCodeZero'}
        report.update(changes)
        return report

    def examples_report(self, **changes):
        report = {'testsFailed': 0, 'testsCollected': 3, 'testsSkipped': 0,
                  'testsCollectedPositive': True, 'notAllSkipped': True}
        report.update(changes)
        return report

    def test_a_build_that_reports_a_native_ascend_backend_passes(self):
        result = evaluate(self.process(self.build_report()),
                          self.checks('tilelang-npu-build'))
        self.assertEqual(result['status'], 'PASS', result['failures'])

    def test_a_build_without_the_native_library_fails(self):
        # The one observation that separates a compiled backend from a checkout
        # that merely configured: without the shared library there is nothing on
        # this host to import.
        result = evaluate(self.process(self.build_report(nativeLibraryPresent=False)),
                          self.checks('tilelang-npu-build'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('OBSERVATION_MISMATCH: nativeLibraryPresent', result['failures'])

    def test_a_build_that_left_the_backend_uncompiled_fails(self):
        result = evaluate(self.process(self.build_report(ascendBackendCompiled=False)),
                          self.checks('tilelang-npu-build'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('OBSERVATION_MISMATCH: ascendBackendCompiled', result['failures'])

    def test_a_nonzero_build_cannot_pass_on_its_own_claim(self):
        result = evaluate(self.process(self.build_report(), exitCode=1),
                          self.checks('tilelang-npu-build'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('NONZERO_EXIT', result['failures'])

    def test_an_examples_run_that_collected_nothing_is_not_a_pass(self):
        # A suite that collected no test is green and proves nothing. The boolean
        # claims alone would pass, which is why the evaluator cross-checks the
        # count instead of trusting them.
        result = evaluate(self.process(self.examples_report(testsCollected=0)),
                          self.checks('tilelang-npu-examples'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('INVALID_TEST_COUNT', result['failures'])

    def test_a_passing_examples_report_passes(self):
        # The counterpart, so the zero-count refusal cannot be satisfied by a gate
        # that fails everything.
        result = evaluate(self.process(self.examples_report()),
                          self.checks('tilelang-npu-examples'))
        self.assertEqual(result['status'], 'PASS', result['failures'])

    def test_a_failed_example_is_never_a_pass(self):
        result = evaluate(self.process(self.examples_report(testsFailed=1)),
                          self.checks('tilelang-npu-examples'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('OBSERVATION_MISMATCH: testsFailed', result['failures'])

    def test_a_missing_counter_cannot_opt_out_of_the_cross_check(self):
        # The shape a stub report has: the two booleans and no counts at all.
        # Without the cross-check the zero-test case could pass by saying nothing.
        report = self.examples_report()
        report.pop('testsCollected')
        result = evaluate(self.process(report), self.checks('tilelang-npu-examples'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('INVALID_TEST_COUNT', result['failures'])

    def test_the_report_the_gate_publishes_is_what_the_manifest_gates_on(self):
        # The Gate emits these booleans plus metadata; the manifest gates on
        # exactly the booleans. A key the Gate stopped emitting would make every
        # build MISSING_OBSERVATION, and a key the Gate never emitted would be a
        # check no run could satisfy.
        source = Path(GATE.__file__).read_text()
        for name in self.checks('tilelang-npu-build'):
            self.assertIn(name, source, name)
        for name in ('testsCollectedPositive', 'notAllSkipped', 'testsFailed',
                     'testsCollected', 'testsSkipped'):
            self.assertIn(name, source, name)


class BuildObservationCase(unittest.TestCase):
    """What the build Gate publishes about itself.

    Both run offline: the shell runner and the native probe are replaced, so the
    question under test is what the Gate REPORTS, not whether this host can
    compile TileLang.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        import contextlib
        self.enter = contextlib.chdir(str(self.root))
        self.enter.__enter__()
        self.addCleanup(self.enter.__exit__, None, None, None)

    def build(self, *, configure_rc=0, build_rc=0):
        """Run build() with the shell and the probe replaced by their answers."""
        answers = iter([(configure_rc, 0.25), (build_rc, 1.5)])
        # Only the interpreter is needed: the shell runner is replaced, so nothing
        # is executed and no toolchain has to exist on this host.
        values = {"TILELANG_NPU_PYTHON": sys.executable}
        with patch.object(GATE, "_run_shell", side_effect=lambda *a, **k: next(answers)), \
                patch.object(GATE, "_probe_observations", return_value=(True, True)):
            return GATE.build(values, 60.0)

    def test_a_successful_build_claims_a_zero_exit(self):
        rc, observations = self.build()
        self.assertEqual(rc, 0)
        self.assertIs(observations["exitCodeZero"], True)
        self.assertIs(observations["nativeLibraryPresent"], True)
        self.assertIs(observations["ascendBackendCompiled"], True)
        # The durations are recorded, not gated: a slow machine is not a failure,
        # but a receipt that cannot show the split cannot be checked either.
        self.assertEqual(observations["configureSeconds"], 0.25)
        self.assertEqual(observations["buildSeconds"], 1.5)

    def test_a_failed_compile_never_publishes_a_zero_exit(self):
        # The claim is derived from the statuses, so a payload cannot say the
        # build succeeded while the compile failed. A constant True here would be
        # a receipt a reader could not trust on its own, even though the evaluator
        # also binds the host process.
        rc, observations = self.build(build_rc=1)
        self.assertEqual(rc, 1)
        self.assertIs(observations["exitCodeZero"], False)

    def test_a_failed_configure_never_publishes_a_zero_exit(self):
        # A configure failure is an ordinary outcome of this Gate, so it must
        # return the same shape as every other outcome. main() unpacks a
        # (rc, observations) pair, and a bare int here makes that path raise
        # TypeError - so the run reports a harness crash instead of the status the
        # manifest maps to BLOCKED_ENV.
        result = self.build(configure_rc=2)
        self.assertIsInstance(result, tuple,
                              "build() must return (rc, observations) on every path")
        rc, observations = result
        self.assertEqual(rc, 2)
        self.assertIs(observations["exitCodeZero"], False)

    def test_a_failed_build_still_reports_what_the_tree_contains(self):
        # The probe answers about the tree, not about the exit status: a partial
        # build that produced a library must say so, and the verdict still fails
        # because the compile did.
        _, observations = self.build(build_rc=1)
        self.assertIs(observations["nativeLibraryPresent"], True)
        result = evaluate(
            self.process(observations),
            Manifest(ADAPTER / "manifest.json").gates["tilelang-npu-build"]
            ["observationChecks"])
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("NONZERO_EXIT", result["failures"])

    def process(self, observations, *, exitCode=1):
        path = self.root / "stdout.json"
        path.write_text(json.dumps(observations))
        return {"exitCode": exitCode, "timedOut": False, "interrupted": False,
                "stdout": {"path": str(path)}}


class ReportShapeCase(unittest.TestCase):
    """Every outcome returns the pair main() unpacks.

    A regression here is invisible until an error path is taken: main() does
    `rc, observations = build(...)`, so a bare int makes the Gate die with a
    TypeError on exactly the paths that are supposed to report 75 (BLOCKED_ENV)
    or 1. The four cases below are the error paths that used to return an int.
    """

    KEYS = ("floats",)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # These cases run the Gate in a temporary checkout, because it reads the
        # working directory as the source root. The restore is registered BEFORE
        # the removal so the directory still exists when the process leaves it.
        self.saved_cwd = os.getcwd()
        self.addCleanup(os.chdir, self.saved_cwd)
        self.addCleanup(self.temp.cleanup)
        os.chdir(str(self.root))

    def assert_report(self, result, rc):
        """A (rc, observations) pair whose claim agrees with the status."""
        self.assertIsInstance(result, tuple,
                              "a Gate outcome must be (rc, observations), not an int")
        status, observations = result
        self.assertEqual(status, rc)
        self.assertIsInstance(observations, dict)
        # The payload must stand on its own: a reader with only the receipt must
        # not be told the run succeeded when it did not.
        self.assertIs(observations["exitCodeZero"], rc == 0)
        return observations

    def values(self):
        return {"TILELANG_NPU_PYTHON": sys.executable}

    def test_a_failed_configure_returns_a_pair(self):
        answers = iter([(2, 0.1)])
        with patch.object(GATE, "_run_shell", side_effect=lambda *a, **k: next(answers)), \
                patch.object(GATE, "_probe_observations", return_value=(False, False)):
            self.assert_report(GATE.build(self.values(), 30.0), 2)

    def test_a_missing_example_file_is_a_refusal_not_a_crash(self):
        # 75 is the code the manifest declares as blockedExitCodes, so this path
        # is how "not applicable here" reaches the runner. A TypeError would be
        # recorded as a gate failure instead.
        self.assert_report(GATE.examples(self.values(), 30.0), 75)

    def test_an_unreadable_junit_report_returns_a_pair(self):
        for name in GATE.DEFAULT_EXAMPLE_TESTS:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text("")
        with patch.object(GATE, "_run_shell", return_value=(1, 0.1)):
            self.assert_report(GATE.examples(self.values(), 30.0), 1)

    def test_a_timed_out_example_run_returns_a_pair(self):
        for name in GATE.DEFAULT_EXAMPLE_TESTS:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text("")
        with patch.object(GATE, "_run_shell", side_effect=GATE.BridgeError("timed out")):
            self.assert_report(GATE.examples(self.values(), 30.0), 1)

    def test_main_can_consume_every_outcome(self):
        # The property the cases above exist for, asserted where it broke: main()
        # unpacks the pair. A shape regression makes this raise TypeError instead
        # of returning a status, which is exactly the mid-run crash it caused.
        (self.root / "CMakeLists.txt").write_text("")
        answers = iter([(2, 0.1)])
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(GATE, "machine_values", return_value=self.values()), \
                patch.object(GATE, "_run_shell", side_effect=lambda *a, **k: next(answers)), \
                patch.object(GATE, "_probe_observations", return_value=(False, False)):
            self.assertEqual(GATE.main(["--build"]), 2)


class DeviceWiringCase(unittest.TestCase):
    """The card this run may use is a launch value, checked and forwarded.

    Occupancy is deliberately NOT asserted as a count: this is a shared host, so
    how many processes hold a card changes between runs. What is pinned is the
    POLICY - a shared card proceeds, and only an operator-demanded exclusivity
    refuses - because that is the part a reader of the receipt must be able to
    rely on.
    """

    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        for name in ("TILELANG_NPU_DEVICE_ID", "TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE"):
            os.environ.pop(name, None)

    def test_no_lease_means_no_device(self):
        # The engine writes the id only for a gate that declares it uses a
        # device, so an absent id means no lease was taken. The Gate must not
        # reach for a card on its own.
        self.assertIsNone(GATE.device_id())

    def test_a_plain_decimal_id_is_accepted(self):
        for value in ("0", "1", "63"):
            os.environ["TILELANG_NPU_DEVICE_ID"] = value
            self.assertEqual(GATE.device_id(), value)

    def test_a_malformed_device_id_is_refused_rather_than_defaulted(self):
        # A junk id must not fall back to card 0: that would silently run on a
        # device the operator never authorized.
        for value in ("npu0", "-1", "1.0", "0x1", "3,4", "three"):
            os.environ["TILELANG_NPU_DEVICE_ID"] = value
            with self.assertRaises(GATE.BridgeError, msg=repr(value)):
                GATE.device_id()
        # Surrounding whitespace is stripped, not rejected: an export padded by a
        # shell is still a plain id, and refusing it would fail a correctly
        # authorized run for a formatting reason.
        os.environ["TILELANG_NPU_DEVICE_ID"] = "  3  "
        self.assertEqual(GATE.device_id(), "3")

    def test_exclusivity_is_opt_in(self):
        self.assertFalse(GATE.exclusive_requested())
        for value in ("1", "true", "yes", "on"):
            os.environ["TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE"] = value
            self.assertTrue(GATE.exclusive_requested(), value)
        for value in ("0", "false", "no", "off", ""):
            os.environ["TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE"] = value
            self.assertFalse(GATE.exclusive_requested(), value)

    def test_a_shared_card_proceeds_and_an_exclusive_run_refuses(self):
        # Sharing a card is normal here, so the default must not block work for a
        # reason unrelated to the code under test. The occupied case is simulated
        # rather than measured, so nothing in this suite depends on this host.
        with patch.object(GATE, "device_occupancy", return_value=["111", "222"]):
            self.assertEqual(GATE.require_device_free("3"), 2)
            os.environ["TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE"] = "1"
            with self.assertRaises(GATE.BridgeError) as error:
                GATE.require_device_free("3")
            self.assertIn("exclusive", str(error.exception))
            # The message reports a count, never another tenant identity.
            self.assertNotIn("111", str(error.exception))
            self.assertNotIn("222", str(error.exception))

    def test_an_absent_npu_smi_never_fails_a_run(self):
        # A missing monitoring tool is not a failed test: the numbers are what
        # this Gate proves, and occupancy is metadata.
        with patch.object(GATE.subprocess, "run", side_effect=OSError("no npu-smi")):
            self.assertEqual(GATE.device_occupancy("0"), [])
        os.environ["TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE"] = "1"
        with patch.object(GATE.subprocess, "run", side_effect=OSError("no npu-smi")):
            self.assertEqual(GATE.require_device_free("0"), 0)

    def test_the_engine_leases_the_variable_this_gate_reads(self):
        # A device id the engine never writes, or a variable the Gate never
        # reads, would mean no lease and no protection while both sides looked
        # configured.
        from agent_gates import engine
        self.assertEqual(engine.DEVICE_ADAPTERS["tilelang-npu"], "TILELANG_NPU_DEVICE_ID")
        self.assertEqual(engine.EXCLUSIVE_VARIABLE["tilelang-npu"],
                         "TILELANG_NPU_REQUIRE_EXCLUSIVE_DEVICE")


class StableAbiAndBuildStampCase(unittest.TestCase):
    """CMake builds cp39-abi3 extensions, drops stale .cpython-*.so, and writes a stamp."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.saved_cwd = os.getcwd()
        self.addCleanup(os.chdir, self.saved_cwd)
        self.addCleanup(self.temp.cleanup)
        os.chdir(str(self.root))
        (self.root / "CMakeLists.txt").write_text("# tilelang\n", encoding="utf-8")
        (self.root / "src").mkdir()
        (self.root / "src" / "stub.cc").write_text("// native\n", encoding="utf-8")
        (self.root / "tilelang").mkdir()
        (self.root / "tilelang" / "w.pyx").write_text("# cython\n", encoding="utf-8")

    def test_cmake_options_enable_cp39_stable_abi(self):
        self.assertIn("-DSKBUILD_SABI_COMPONENT=Development.SABIModule", GATE.CMAKE_OPTIONS)
        self.assertIn("-DSKBUILD_SABI_VERSION=3.9", GATE.CMAKE_OPTIONS)

    def test_build_removes_stale_cpython_so_and_writes_build_stamp(self):
        lib_dir = self.root / "build" / "lib"
        lib_dir.mkdir(parents=True)
        stale_so = lib_dir / "tilelang_cython_wrapper.cpython-311-x86_64-linux-gnu.so"
        stale_so.write_bytes(b"stale")
        libtilelang = lib_dir / "libtilelang.so"
        libtilelang.write_bytes(b"0" * ((1 << 20) + 16))
        for ext in GATE.SABI_EXTENSIONS:
            (lib_dir / f"{ext}.abi3.so").write_bytes(b"abi3")

        answers = iter([(0, 0.2), (0, 1.0)])
        values = {"TILELANG_NPU_PYTHON": sys.executable}
        with patch.object(GATE, "_run_shell", side_effect=lambda *a, **k: next(answers)), \
                patch.object(GATE, "_probe_observations", return_value=(True, True)):
            rc, obs = GATE.build(values, 60.0)

        self.assertEqual(rc, 0)
        self.assertTrue(obs["nativeLibraryPresent"])
        self.assertFalse(stale_so.exists())
        stamp_path = self.root / "build" / GATE.BUILD_STAMP_FILE
        self.assertTrue(stamp_path.is_file())
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        self.assertEqual(stamp["adapter"], "tilelang-npu")
        self.assertEqual(stamp["sabi"], "cp39")
        self.assertEqual(stamp["nativeInputsSha256"],
                         GATE._tilelang_native_inputs_sha256(self.root))
        self.assertEqual(stamp["libtilelangSha256"], GATE._sha256_file(libtilelang))

    def test_probe_requires_all_sabi_extensions(self):
        lib_dir = self.root / "build" / "lib"
        lib_dir.mkdir(parents=True)
        libtilelang = lib_dir / "libtilelang.so"
        libtilelang.write_bytes(b"0" * ((1 << 20) + 16))
        # Missing one .abi3.so -> nativeLibraryPresent is False without running shell.
        for ext in GATE.SABI_EXTENSIONS[:-1]:
            (lib_dir / f"{ext}.abi3.so").write_bytes(b"abi3")
        with patch.object(GATE, "_run_shell") as shell_mock:
            present, compiled = GATE._probe_observations(
                {"TILELANG_NPU_PYTHON": sys.executable}, libtilelang, self.root
            )
        self.assertFalse(present)
        self.assertFalse(compiled)
        shell_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()