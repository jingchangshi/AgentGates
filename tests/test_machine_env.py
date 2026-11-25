"""One machine file is the only source of an Adapter's machine parameters.

These tests pin the two properties that make the arrangement safe:

* the key sets parsed here and by each Adapter's scripts are the same, so a key
  cannot become machine configuration one side reads and the other ignores;
* the build directory the engine leases is the directory the Adapter builds in,
  because a lock on one tree while another is written is not a lock at all.

Everything is offline: a machine file is written into a temporary directory and
the Adapter scripts are imported directly, never executed against Docker.
"""
import importlib.util
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / 'agent_gates/adapters'
sys.path.insert(0, str(ROOT))

from agent_gates import machine_env
from agent_gates.manifest import Manifest


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ASCEND = load_script(ADAPTERS / 'ascendnpu-ir/scripts/docker_gate.py', 'machine_env_ascend')
HOST = load_script(ADAPTERS / 'ascendnpu-ir/scripts/host_gate.py', 'machine_env_ascend_host')
TILELANG = load_script(ADAPTERS / 'tilelang-hivm/scripts/docker_gate.py', 'machine_env_tilelang')
NPU = load_script(ADAPTERS / 'tilelang-hivm-npu/scripts/npu_gate.py', 'machine_env_npu')
TLA = load_script(ADAPTERS / 'tla-npu/scripts/tla_gate.py', 'machine_env_tla')
FK = load_script(ADAPTERS / 'forgekernel-ascend-npu/scripts/fk_gate.py', 'machine_env_fk')
BSKA = load_script(ADAPTERS / 'bska-npu/scripts/bska_gate.py', 'machine_env_bska')
TLNPU = load_script(ADAPTERS / 'tilelang-npu/scripts/tilelang_npu_gate.py', 'machine_env_tlnpu')
TFOUNDRY = load_script(ADAPTERS / 'tilefoundry-npu/scripts/tilefoundry_host_gate.py', 'machine_env_tilefoundry')
MK = load_script(ADAPTERS / 'megakernel-npu/scripts/megakernel_npu_gate.py', 'machine_env_megakernel')


class EnvironmentCase(unittest.TestCase):
    """A saved environment plus a temporary machine file directory."""

    def setUp(self):
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in list(os.environ):
            if name.startswith(('ASCENDNPU_', 'TILELANG_HIVM_', 'TLA_NPU_', 'FK_ASCEND_')):
                os.environ.pop(name)

    def write_env(self, name, values):
        path = self.root / name
        path.write_text(''.join(f'{key}={value}\n' for key, value in values.items()))
        return path


class KeySetParityCase(EnvironmentCase):
    """The engine and the Adapters must accept exactly the same keys."""

    def test_the_ascend_key_set_matches_the_engine(self):
        self.assertEqual(tuple(sorted(ASCEND.FILE_KEYS)),
                         tuple(sorted(machine_env.ASCENDNPU_KEYS)))
        # Both routes read one file, so the Host wrapper must accept exactly the
        # same keys. If it accepted fewer, a key the engine leases on would be a
        # file the Host Gate refuses; if it accepted more, the Host Gate could
        # read machine configuration the engine never leased.
        self.assertEqual(tuple(sorted(HOST.FILE_KEYS)),
                         tuple(sorted(machine_env.ASCENDNPU_KEYS)))
        # And the Host wrapper binds the sibling's own tuple rather than
        # restating it: there is one key set, owned by the pinned bridge.
        self.assertIs(HOST.FILE_KEYS, HOST.shared.FILE_KEYS)

    def test_each_adapter_key_set_matches_the_engine(self):
        # tilelang-hivm and tilelang-hivm-npu share one file, so both scripts must
        # accept every key that file may define - including keys they never read.
        # tla-npu owns its own file and its own prefix, so it is compared against
        # its own engine key set rather than being folded into the TileLang one.
        for module, expected in ((TILELANG, machine_env.TILELANG_HIVM_KEYS),
                                 (NPU, machine_env.TILELANG_HIVM_KEYS),
                                 (TLA, machine_env.TLA_NPU_KEYS),
                                 (FK, machine_env.FK_ASCEND_KEYS),
                                 (BSKA, machine_env.BSKA_NPU_KEYS),
                                 (TLNPU, machine_env.TILELANG_NPU_KEYS),
                                 (TFOUNDRY, machine_env.TILEFOUNDRY_NPU_KEYS),
                                 (MK, machine_env.MEGAKERNEL_NPU_KEYS)):
            self.assertEqual(tuple(sorted(module.TOOLCHAIN_KEYS)),
                             tuple(sorted(expected)), module.__name__)

    def test_the_build_keys_are_machine_configuration_in_every_adapter(self):
        # Every Adapter that owns a build tree declares both the directory the
        # engine leases and the job count the build uses. tla-npu writes no build
        # tree - it compiles through a prebuilt vendor wheel - so it declares the
        # directory (the engine resolves one for every Adapter) but not a job
        # count, and asserting one would demand a setting nothing consumes.
        for keys in (machine_env.ASCENDNPU_KEYS, machine_env.TILELANG_HIVM_KEYS):
            self.assertTrue([key for key in keys if key.endswith('_BUILD_SUBDIR')])
            self.assertTrue([key for key in keys if key.endswith('_NINJA_JOBS')])
        self.assertTrue([key for key in machine_env.TLA_NPU_KEYS
                         if key.endswith('_BUILD_SUBDIR')])
        self.assertEqual([key for key in machine_env.TLA_NPU_KEYS
                          if key.endswith('_NINJA_JOBS')], [],
                         'tla-npu has no build tree, so it must not declare a job count')
        # forgekernel-ascend-npu installs into a private venv and never runs
        # Ninja in the source, so it declares the lease directory but no job
        # count, for the same reason tla-npu does.
        self.assertTrue([key for key in machine_env.FK_ASCEND_KEYS
                         if key.endswith('_BUILD_SUBDIR')])
        self.assertEqual([key for key in machine_env.FK_ASCEND_KEYS
                          if key.endswith('_NINJA_JOBS')], [],
                         'forgekernel-ascend-npu has no Ninja build tree')

    def test_the_adapter_parsers_accept_exactly_what_the_engine_accepts(self):
        # Key-name parity is not enough: the two sides have separate parsers, so a
        # file the engine accepts must also be accepted by the Adapter, and one it
        # rejects must be rejected there too. Otherwise a machine could pass the
        # engine, take a lease, and then fail inside the Gate - or worse, the Gate
        # could accept a key the engine never leases.
        for adapter, spec in machine_env.ADAPTERS.items():
            module = {'ascendnpu-ir': ASCEND, 'tilelang-hivm': TILELANG,
                      'tilelang-hivm-npu': NPU, 'tla-npu': TLA,
                      'forgekernel-ascend-npu': FK, 'bska-npu': BSKA,
                      'tilelang-npu': TLNPU,
                      'tilefoundry-npu': TFOUNDRY,
                      'megakernel-npu': MK}[adapter]
            path = self.root / f'{adapter}.env'
            text = ''.join(f'{key}=' + chr(10) for key in spec['keys'])
            path.write_text(text)
            machine_env.parse(text, path, spec['keys'])          # engine accepts
            module._parse_env_file(text, path)                    # adapter accepts
            bad = text + 'NOT_A_MACHINE_KEY=x' + chr(10)
            with self.assertRaises(machine_env.MachineEnvError):
                machine_env.parse(bad, path, spec['keys'])
            with self.assertRaises(module.BridgeError):
                module._parse_env_file(bad, path)
            del path

    def test_every_shipped_template_parses_with_the_engine_grammar(self):
        seen = 0
        for adapter, spec in machine_env.ADAPTERS.items():
            template = ADAPTERS / spec['directory'] / 'env.example.sh'
            if not template.is_file():
                continue
            seen += 1
            values = machine_env.parse(template.read_text(), template, spec['keys'])
            # A committed template contains only keys the Adapter accepts.
            self.assertTrue(values, adapter)
            # Every machine-specific path a Gate enters is left for the operator
            # to fill in: a committed default would name someone else's machine.
            for key, value in values.items():
                if key.endswith(('_CANN_SETUP', '_CONDA_SH', '_CONDA_ENV', '_CCEC_BIN',
                                 '_CONTAINER_NAME', '_CONTAINER_USER', '_CONTAINER_HOME',
                                 '_SOURCE_ROOT', '_WHEEL_DIR', '_HIVM_COMPILER')):
                    self.assertEqual(value, '', (adapter, key))
        self.assertGreaterEqual(seen, 2)


class StrictGrammarCase(EnvironmentCase):
    """The file is data: no execution, no expansion, no ambiguity."""

    def test_a_value_is_never_expanded_or_executed(self):
        values = machine_env.parse('A=$(id -u)\nB=`id -u`\nC=${HOME}\n',
                                   self.root / 'f', ('A', 'B', 'C'))
        self.assertEqual(values, {'A': '$(id -u)', 'B': '`id -u`', 'C': '${HOME}'})

    def test_an_unknown_key_is_refused_by_name(self):
        with self.assertRaises(machine_env.MachineEnvError) as error:
            machine_env.parse('A=1\nTYPO=2\n', self.root / 'f', ('A',))
        self.assertIn('TYPO', str(error.exception))
        self.assertIn('A', str(error.exception))

    def test_a_duplicate_key_is_refused(self):
        with self.assertRaises(machine_env.MachineEnvError) as error:
            machine_env.parse('A=1\nA=2\n', self.root / 'f', ('A',))
        self.assertIn('a second time', str(error.exception))

    def test_a_malformed_line_is_refused(self):
        for text in ('A\n', 'A =1\n', 'A:1\n', '1A=1\n'):
            with self.assertRaises(machine_env.MachineEnvError, msg=repr(text)):
                machine_env.parse(text, self.root / 'f', ('A',))

    def test_quotes_are_stripped_but_must_balance(self):
        self.assertEqual(machine_env.parse('A="x y"\n', self.root / 'f', ('A',)), {'A': 'x y'})
        for text in ('A="x\n', "A=x'\n"):
            with self.assertRaises(machine_env.MachineEnvError, msg=repr(text)):
                machine_env.parse(text, self.root / 'f', ('A',))

    def test_comments_and_blank_lines_are_ignored(self):
        self.assertEqual(machine_env.parse('# c\n\nA=1\n', self.root / 'f', ('A',)), {'A': '1'})


class ResolutionCase(EnvironmentCase):
    """Which file is read, and what happens when one does not exist."""

    def test_the_packaged_file_is_the_default(self):
        path = machine_env.find_file('ascendnpu-ir', {})
        packaged = ADAPTERS / 'ascendnpu-ir/env.sh'
        self.assertEqual(path, packaged if packaged.is_file() else None)

    def test_the_selector_must_be_absolute_and_exist(self):
        with self.assertRaises(machine_env.MachineEnvError):
            machine_env.find_file('ascendnpu-ir', {'ASCENDNPU_ENV_FILE': 'relative.sh'})
        with self.assertRaises(machine_env.MachineEnvError):
            machine_env.find_file('ascendnpu-ir', {'ASCENDNPU_ENV_FILE': str(self.root / 'no')})
        real = self.write_env('real.env', {'ASCENDNPU_BUILD_SUBDIR': 'b'})
        self.assertEqual(machine_env.find_file('ascendnpu-ir',
                                               {'ASCENDNPU_ENV_FILE': str(real)}), real)

    def test_no_file_means_no_values_and_the_default_build_directory(self):
        # A machine that keeps no file gets the documented defaults. `env_file`
        # is passed explicitly so the result cannot depend on whether the machine
        # running the suite keeps a real, git-ignored `env.sh`.
        machine = machine_env.read('ascendnpu-ir', {}, env_file=None)
        if machine.path is not None:
            # A real file exists here; the empty case is covered by an empty one.
            empty = self.write_env('empty.env', {})
            machine = machine_env.read('ascendnpu-ir', {}, env_file=empty)
        self.assertEqual(machine.values, {})
        self.assertEqual(machine.build_subdir, 'build')
        self.assertEqual(machine.ninja_jobs(), '$(nproc)')
        self.assertIsNone(machine.container_identity())

    def test_the_two_tilelang_adapters_share_one_file(self):
        self.assertEqual(machine_env.ADAPTERS['tilelang-hivm']['directory'],
                         machine_env.ADAPTERS['tilelang-hivm-npu']['directory'])

    def test_a_half_container_identity_fails_closed(self):
        path = self.write_env('p.env', {'ASCENDNPU_CONTAINER_NAME': 'c'})
        machine = machine_env.read('ascendnpu-ir', env_file=path)
        with self.assertRaises(machine_env.MachineEnvError) as error:
            machine.container_identity()
        self.assertIn('ASCENDNPU_CONTAINER_USER', str(error.exception))

    def test_a_full_container_identity_is_returned_in_order(self):
        path = self.write_env('c.env', {'ASCENDNPU_CONTAINER_NAME': 'box',
                                       'ASCENDNPU_CONTAINER_USER': 'u',
                                       'ASCENDNPU_CONTAINER_HOME': '/home/u'})
        machine = machine_env.read('ascendnpu-ir', env_file=path)
        self.assertEqual(machine.container_identity(), ('box', 'u', '/home/u'))

    def test_a_build_subdirectory_must_be_one_plain_name(self):
        for value in ('..', '.', 'a/b', '/build'):
            path = self.write_env('b.env', {'ASCENDNPU_BUILD_SUBDIR': value})
            with self.assertRaises(machine_env.MachineEnvError, msg=repr(value)):
                machine_env.read('ascendnpu-ir', env_file=path)
        # An empty value is a typo, not "use the default": the file spells "unset"
        # by leaving the key out. The Adapter refuses it too, so accepting it here
        # would let a run lease a directory the Gate then refuses to build in.
        path = self.write_env('e.env', {'ASCENDNPU_BUILD_SUBDIR': ''})
        with self.assertRaises(machine_env.MachineEnvError) as error:
            machine_env.read('ascendnpu-ir', env_file=path)
        self.assertIn('set but empty', str(error.exception))

    def test_jobs_must_be_a_positive_integer(self):
        for value in ('0', '-1', 'sixty'):
            path = self.write_env('j.env', {'ASCENDNPU_NINJA_JOBS': value})
            with self.assertRaises(machine_env.MachineEnvError, msg=repr(value)):
                machine_env.read('ascendnpu-ir', env_file=path).ninja_jobs()


class LeaseParityCase(EnvironmentCase):
    """The engine leases the directory the Adapter builds in."""

    def setUp(self):
        super().setUp()
        self.source = self.root / 'source'
        self.source.mkdir()
        self.manifest = Manifest(ADAPTERS / 'ascendnpu-ir/manifest.json')

    def test_the_engine_and_the_adapter_read_the_same_directory(self):
        from agent_gates import engine
        for subdir in ('build', 'build-release', 'out.dir'):
            path = self.write_env('m.env', {'ASCENDNPU_BUILD_SUBDIR': subdir})
            os.environ['ASCENDNPU_ENV_FILE'] = str(path)
            leased = engine.build_directory(self.source, self.manifest, os.environ)
            self.assertEqual(leased, (self.source / subdir).resolve())
            self.assertEqual(ASCEND.build_subdir(), subdir)

    def test_a_contradicting_export_cannot_move_the_lock_away_from_the_build(self):
        # The Adapter refuses the disagreement, so no run can hold a lease on one
        # tree while writing another: the engine resolves the file, and the Gate
        # refuses to start when the shell says something else.
        path = self.write_env('m.env', {'ASCENDNPU_BUILD_SUBDIR': 'build'})
        os.environ.update({'ASCENDNPU_ENV_FILE': str(path),
                           'ASCENDNPU_BUILD_SUBDIR': 'build-fast'})
        with self.assertRaises(ASCEND.BridgeError) as error:
            ASCEND.build_subdir()
        self.assertIn('read from that file only', str(error.exception))
        from agent_gates import engine
        self.assertEqual(engine.build_directory(self.source, self.manifest, os.environ),
                         (self.source / 'build').resolve())

    def test_the_tilelang_build_directory_is_leased_from_the_shared_file(self):
        from agent_gates import engine
        path = self.write_env('t.env', {
            'TILELANG_HIVM_BUILD_SUBDIR': 'build-npu',
            'TILELANG_HIVM_CONTAINER_NAME': 'c',
            'TILELANG_HIVM_CONTAINER_USER': 'u',
            'TILELANG_HIVM_CONTAINER_HOME': '/home/u',
        })
        os.environ['TILELANG_HIVM_ENV_FILE'] = str(path)
        manifest = Manifest(ADAPTERS / 'tilelang-hivm/manifest.json')
        self.assertEqual(engine.build_directory(self.source, manifest, os.environ),
                         (self.source / 'build-npu').resolve())
        self.assertEqual(TILELANG.build_subdir(), 'build-npu')
        # The device Adapter reads the same file, so it must accept the key the
        # build Adapter acts on - one file serves both.
        self.assertEqual(NPU.env_file_values()['TILELANG_HIVM_BUILD_SUBDIR'], 'build-npu')

    def test_a_bad_machine_file_blocks_the_run_instead_of_leasing_a_guess(self):
        from agent_gates import engine
        path = self.write_env('bad.env', {'ASCENDNPU_BUILD_SUBDIR': 'a/b'})
        os.environ['ASCENDNPU_ENV_FILE'] = str(path)
        with self.assertRaises(ValueError):
            engine.build_directory(self.source, self.manifest, os.environ)


class RouteParityCase(EnvironmentCase):
    """The Docker and Host routes must refuse and accept the same machine files.

    Key-name parity alone cannot prevent semantic drift: the Host wrapper binds
    the Docker bridge's parser today, but nothing stops a later edit from forking
    it. These cases therefore drive BOTH modules over one corpus of files and
    compare what they DO - the values they resolve and whether they refuse - so a
    fork that changes one side's behaviour fails here.
    """

    # Each entry is (description, file text, extra environment). The environment
    # is what the launching shell would hold; a case that exports a machine value
    # is exercising the file-wins / disagreement-refused rules.
    CORPUS = (
        ('plain', 'ASCENDNPU_CANN_SETUP=/cann/set_env.sh\n', {}),
        ('quotes', 'ASCENDNPU_CANN_SETUP="/cann/set_env.sh"\n'
                    "ASCENDNPU_CONDA_SH='/conda/c.sh'\n"
                    'ASCENDNPU_CONDA_ENV="mk"\n', {}),
        ('comments and blanks', '# a comment\n\nASCENDNPU_CONDA_ENV=mk\n', {}),
        ('duplicate key', 'ASCENDNPU_CONDA_ENV=a\nASCENDNPU_CONDA_ENV=b\n', {}),
        ('unknown key', 'ASCENDNPU__CANN_SETUP=/a\n', {}),
        ('unbalanced quote', 'ASCENDNPU_CANN_SETUP="/a\n', {}),
        ('unbalanced trailing quote', "ASCENDNPU_CONDA_ENV=mk'\n", {}),
        ('not an assignment', 'rm -rf /\n', {}),
        ('export line', 'export X=/a\n', {}),
        ('space before equals', 'ASCENDNPU_CANN_SETUP =/a\n', {}),
        ('relative cann path', 'ASCENDNPU_CANN_SETUP=cann/set_env.sh\n', {}),
        ('relative ccec path', 'ASCENDNPU_CCEC_BIN=relative/bin\n', {}),
        ('empty cann disables it', 'ASCENDNPU_CANN_SETUP=\nASCENDNPU_CONDA_ENV=\n', {}),
        ('empty build subdir', 'ASCENDNPU_BUILD_SUBDIR=\n', {}),
        ('build subdir traversal', 'ASCENDNPU_BUILD_SUBDIR=../build\n', {}),
        ('build subdir absolute', 'ASCENDNPU_BUILD_SUBDIR=/build\n', {}),
        ('plain build subdir', 'ASCENDNPU_BUILD_SUBDIR=build-release\n', {}),
        ('non-numeric jobs', 'ASCENDNPU_NINJA_JOBS=sixty\n', {}),
        ('zero jobs', 'ASCENDNPU_NINJA_JOBS=0\n', {}),
        ('positive jobs', 'ASCENDNPU_NINJA_JOBS=8\n', {}),
        ('command substitution is literal',
         'ASCENDNPU_CANN_SETUP=$(touch {marker})\n', {}),
        ('export agrees with the file',
         'ASCENDNPU_BUILD_SUBDIR=build-release\n',
         {'ASCENDNPU_BUILD_SUBDIR': 'build-release'}),
        ('export contradicts the file',
         'ASCENDNPU_BUILD_SUBDIR=build\n',
         {'ASCENDNPU_BUILD_SUBDIR': 'build-fast'}),
        ('export not in the file', 'ASCENDNPU_CANN_SETUP=\n',
         {'ASCENDNPU_BUILD_SUBDIR': 'build-fast'}),
        ('container keys', ''.join(f'{key}=value\n' for key in
                                    ('ASCENDNPU_CONTAINER_NAME',
                                     'ASCENDNPU_CONTAINER_USER',
                                     'ASCENDNPU_CONTAINER_HOME')), {}),
    )

    # The values both routes resolve, in a fixed order. `toolchain_from_env_file`
    # covers PATH-valued keys with their own absolute-path rules, so the corpus
    # exercises the grammar and the semantic rules together.
    READERS = ('toolchain_from_env_file', 'build_subdir', 'ninja_jobs', 'ccec_bin',
               'require_conda_pair')

    @staticmethod
    def owner(module):
        """The module that DEFINES the parser: the wrapper's sibling, if any."""
        return getattr(module, 'shared', module)

    def behaviour(self, module, path):
        """What one module resolves: ok+value, or refused, per reader."""
        result = []
        for name in self.READERS:
            reader = getattr(module, name)
            try:
                if name == 'require_conda_pair':
                    cann, conda_sh, conda_env = module.toolchain_from_env_file()
                    reader(conda_sh, conda_env)
                    result.append((name, 'ok', None))
                else:
                    result.append((name, 'ok', repr(reader())))
            except module.BridgeError:
                result.append((name, 'refused', None))
        return result

    def test_the_two_routes_agree_on_one_corpus_of_machine_files(self):
        for number, (label, text, extra) in enumerate(self.CORPUS):
            with self.subTest(label=label):
                text = text.format(marker=self.root / 'executed')
                path = self.root / f'corpus-{number}.env'
                path.write_text(text)
                values = dict(os.environ)
                values['ASCENDNPU_ENV_FILE'] = str(path)
                values.update(extra)
                with patch.dict(os.environ, values, clear=True):
                    docker = self.behaviour(ASCEND, path)
                    self.assertFalse((self.root / 'executed').exists(),
                                     'a machine file was executed, not parsed')
                with patch.dict(os.environ, values, clear=True):
                    host = self.behaviour(HOST, path)
                self.assertEqual(docker, host, label)

    def test_a_missing_selector_is_refused_by_both_routes(self):
        # The engine refuses an explicitly selected file that does not exist, so a
        # route that quietly used the default would gate a machine nobody named.
        values = {'ASCENDNPU_ENV_FILE': str(self.root / 'absent.env')}
        for module in (ASCEND, HOST):
            with self.subTest(module=module.__name__), patch.dict(os.environ, values, clear=True):
                with self.assertRaises(module.BridgeError):
                    module.toolchain_from_env_file()

    def test_a_relative_selector_is_refused_by_both_routes(self):
        values = {'ASCENDNPU_ENV_FILE': 'relative/env.sh'}
        for module in (ASCEND, HOST):
            with self.subTest(module=module.__name__), patch.dict(os.environ, values, clear=True):
                with self.assertRaises(module.BridgeError):
                    module.toolchain_from_env_file()

    def test_no_file_at_all_means_the_same_empty_configuration(self):
        for module in (ASCEND, HOST):
            with self.subTest(module=module.__name__), \
                    patch.dict(os.environ, {'ASCENDNPU_CANN_SETUP': ''}, clear=True):
                # `default_env_file` is the sibling's, so it has to be patched
                # where the parser looks it up - on the module that owns the
                # search path, not on the wrapper that binds the parser.
                with patch.object(self.owner(module), 'default_env_file',
                                  return_value=None):
                    self.assertEqual(module.toolchain_from_env_file(),
                                     (None, None, None))
                    self.assertEqual(module.build_subdir(), 'build')
                    self.assertEqual(module.ninja_jobs(), '$(nproc)')
                    self.assertIsNone(module.ccec_bin())


class RouteLeaseCase(EnvironmentCase):
    """A Host run and a Docker run must contend for the SAME build tree lease.""

    The engine derives a run's resources from the Adapter ID and the resolved
    build directory, and the resource string carries no Adapter identity
    (locking.normalise realpaths only the path). That is what makes the two
    routes serialize correctly: they are the same Adapter writing one tree. A
    second Adapter ID would acquire a second, distinct lease for the same
    directory, and two runs could then write it concurrently.
    """

    def setUp(self):
        super().setUp()
        self.source = self.root / 'source'
        self.source.mkdir()

    def resources(self, adapter, subdir):
        """The leases a run of `adapter` would take, as the engine builds them."""
        from agent_gates import engine, locking
        manifest = Manifest(ADAPTERS / adapter / 'manifest.json')
        name = ADAPTERS / machine_env.ADAPTERS[adapter]['directory']
        path = self.write_env(f'{adapter}.env',
                              {machine_env.ADAPTERS[adapter]['prefix'] + 'BUILD_SUBDIR':
                               subdir})
        env = {'ASCENDNPU_ENV_FILE': str(path),
               'TILELANG_HIVM_ENV_FILE': str(path)}
        with patch.dict(os.environ, env, clear=True):
            build = engine.build_directory(self.source, manifest, os.environ)
        resources = ['source:' + str(self.source), 'build:' + str(build)]
        return [locking.lock_path(resource, self.root / 'locks') for resource in resources], build

    def test_the_host_gate_builds_where_the_engine_leases(self):
        from agent_gates import engine
        manifest = Manifest(ADAPTERS / 'ascendnpu-ir/manifest.json')
        for subdir in ('build', 'build-release', 'out.dir'):
            with self.subTest(subdir=subdir):
                path = self.write_env('m.env', {'ASCENDNPU_BUILD_SUBDIR': subdir})
                with patch.dict(os.environ, {'ASCENDNPU_ENV_FILE': str(path)}, clear=True):
                    leased = engine.build_directory(self.source, manifest, os.environ)
                    # The Host wrapper resolves the same name from the same file,
                    # and the pin is `name`, not a private copy of the default.
                    self.assertEqual(leased, (self.source / subdir).resolve())
                    self.assertEqual(HOST.build_subdir(), subdir)
                    self.assertEqual(leased, (self.source / HOST.build_subdir()).resolve())

    def test_both_routes_take_the_same_lease_for_one_tree(self):
        # One Adapter, two routes: the IDs the engine leases under are identical
        # because the resource string is derived from the PATH, not the route.
        for subdir in ('build', 'build-release'):
            with self.subTest(subdir=subdir):
                from agent_gates import locking
                host_paths, host_build = self.resources('ascendnpu-ir', subdir)
                # The same manifest serves both routes, so there is nothing to
                # compare across Adapter IDs here - the point is that the Host
                # gate's own directory produces the lease the engine takes.
                self.assertEqual(host_build, (self.source / subdir).resolve())
                self.assertEqual(host_paths,
                                 [locking.lock_path('source:' + str(self.source),
                                                    self.root / 'locks'),
                                  locking.lock_path('build:' + str(host_build),
                                                    self.root / 'locks')])
                # No adapter identity appears in the resource name.
                for resource in ('source:' + str(self.source),
                                 'build:' + str(host_build)):
                    self.assertNotIn('ascendnpu', locking.normalise(resource))

    def test_two_adapter_ids_on_one_build_path_still_collide(self):
        # The reason the host route is a new gate on the SAME Adapter rather than
        # a new Adapter: two IDs pointing at one directory must hash to one lock
        # file, or the serialization the lease exists for does not happen.
        from agent_gates import locking
        ascend, ascend_build = self.resources('ascendnpu-ir', 'build')
        tilelang, tilelang_build = self.resources('tilelang-hivm', 'build')
        self.assertEqual(ascend_build, tilelang_build)
        self.assertEqual(ascend, tilelang)
        # Stated as the property, not just as this pair: the file one resource
        # leases is a hash of the resolved path alone.
        self.assertEqual(locking.lock_path('build:' + str(ascend_build), self.root / 'locks'),
                         locking.lock_path('build:' + str(tilelang_build), self.root / 'locks'))

class WorkerBackstopCase(EnvironmentCase):
    """The worker backstop is derived from the Adapters, never guessed."""

    def test_the_backstop_exceeds_every_installed_recipe(self):
        from agent_gates import systemd_units
        from agent_gates.registry import Registry
        registry = Registry()
        backstop = systemd_units.backstop_seconds(registry)
        longest = max(float(recipe['timeout'])
                      for manifest in registry.list()
                      for recipe in manifest.recipes.values())
        self.assertGreater(backstop, longest)
        self.assertEqual(backstop, int(longest) + systemd_units.BACKSTOP_MARGIN_SECONDS)

    def test_the_backstop_accommodates_a_from_zero_full_build(self):
        # The Ascend build gate can take a cold full build; a backstop that only
        # covered an incremental run would kill it mid-configure and make the
        # from-zero branch unusable in practice.
        from agent_gates import systemd_units
        from agent_gates.registry import Registry
        manifest = Registry().get('ascendnpu-ir')
        build = manifest.recipes['ascendnpu-ir.incremental-build']
        self.assertGreater(systemd_units.backstop_seconds(), float(build['timeout']))
        self.assertGreater(float(build['timeout']), 21600)

    def test_a_shorter_explicit_backstop_is_refused(self):
        from agent_gates import systemd_units
        minimum = systemd_units.backstop_seconds()
        with self.assertRaises(ValueError) as error:
            with patch.object(systemd_units, 'run'):
                systemd_units.install(sys.executable, timeout_seconds=minimum - 1,
                                      root=self.root)
        self.assertIn('shorter than', str(error.exception))
        self.assertFalse((self.root / systemd_units.UNIT_NAME).exists())

    def test_an_installed_unit_carries_the_derived_backstop(self):
        from agent_gates import systemd_units
        with patch.object(systemd_units, 'run'):
            systemd_units.install(sys.executable, root=self.root)
        unit = (self.root / systemd_units.UNIT_NAME).read_text()
        match = re.search(r'RuntimeMaxSec=(\d+)', unit)
        self.assertIsNotNone(match, unit)
        self.assertEqual(int(match.group(1)), systemd_units.backstop_seconds())
        self.assertGreater(int(match.group(1)), 7200)

    def test_main_wires_device_lock_root_cli_and_env_fallback(self):
        from agent_gates import systemd_units
        with patch.object(systemd_units, 'unit_dir', return_value=self.root), \
                patch.object(systemd_units, 'run'):
            rc = systemd_units.main([
                '--python', sys.executable,
                '--device-lock-root', '/var/lock/agent-gates',
            ])
        self.assertEqual(rc, 0)
        unit = (self.root / systemd_units.UNIT_NAME).read_text()
        self.assertIn('Environment=AGENT_GATES_DEVICE_LOCK_ROOT=/var/lock/agent-gates', unit)

        with patch.dict(os.environ, {'AGENT_GATES_DEVICE_LOCK_ROOT': '/tmp/shared-locks'}, clear=False), \
                patch.object(systemd_units, 'unit_dir', return_value=self.root), \
                patch.object(systemd_units, 'run'):
            rc_env = systemd_units.main(['--python', sys.executable])
        self.assertEqual(rc_env, 0)
        unit_env = (self.root / systemd_units.UNIT_NAME).read_text()
        self.assertIn('Environment=AGENT_GATES_DEVICE_LOCK_ROOT=/tmp/shared-locks', unit_env)


if __name__ == '__main__':
    unittest.main()
