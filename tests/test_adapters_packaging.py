"""Package discovery and host-only preflight regressions."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / 'agent_gates/adapters'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackagingCase(unittest.TestCase):
    def test_every_bridge_finds_its_packaged_configuration(self):
        # Each Adapter ships its own machine file beside its template. The two
        # TileLang adapters share one file; the TLA adapter owns a separate one,
        # so a shared-file assumption here would make it look configured when it
        # is not.
        shared = {'tilelang-hivm': 'tilelang-hivm', 'tilelang-hivm-npu': 'tilelang-hivm'}
        adapters = ('ascendnpu-ir', 'tilelang-hivm', 'tilelang-hivm-npu', 'tla-npu',
                    'forgekernel-ascend-npu')
        # Each Adapter's bridge file, named explicitly. A suffix rule cannot work
        # here: two Adapter IDs end in '-npu' but ship differently named scripts.
        bridges = {'tla-npu': 'tla_gate.py', 'forgekernel-ascend-npu': 'fk_gate.py'}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'agent_gates/adapters'
            for name in adapters:
                (root / name).mkdir(parents=True)
            for name in ('ascendnpu-ir', 'tilelang-hivm', 'tla-npu', 'forgekernel-ascend-npu'):
                (root / name / 'env.example.sh').write_text('# template\n')
            for name in adapters:
                script = bridges.get(name, 'npu_gate.py' if name.endswith('-npu') else 'docker_gate.py')
                module = load(name, ADAPTERS / name / 'scripts' / script)
                with patch.object(module, 'ADAPTER_DIR', root / name):
                    expected = shared.get(name, name)
                    self.assertEqual(module.default_env_file(), root / expected / 'env.sh')

    def test_ascend_check_config_validates_without_spawning_docker(self):
        module = load('ascend_preflight', ADAPTERS / 'ascendnpu-ir/scripts/docker_gate.py')
        with tempfile.TemporaryDirectory() as td:
            env_file = Path(td) / 'machine.env'
            # The container identity is a machine fact in this file, not in the
            # launching environment: the same file the engine reads to lease the
            # build directory.
            env_file.write_text('ASCENDNPU_CONTAINER_NAME=fixture\n'
                                'ASCENDNPU_CONTAINER_USER=builder\n'
                                'ASCENDNPU_CONTAINER_HOME=/home/builder\n')
            values = {'ASCENDNPU_ENV_FILE': str(env_file)}
            with patch.dict(os.environ, values, clear=True), patch.object(module.subprocess, 'run') as run:
                self.assertEqual(module.main(['--check-config']), 0)
                run.assert_not_called()
            env_file.write_text('ASCENDNPU_CONDA_ENV=fixture\n')
            with patch.dict(os.environ, values, clear=True), patch.object(module.subprocess, 'run') as run:
                with self.assertRaises(module.BridgeError):
                    module.main(['--check-config'])
                run.assert_not_called()

    def test_a_shell_export_cannot_stand_in_for_a_missing_file_key(self):
        # The engine leases the build directory from the machine file. If the Gate
        # honoured an export when the file left a key out, the two could disagree
        # and one run would hold a lock on a tree it does not build in.
        module = load('ascend_preflight', ADAPTERS / 'ascendnpu-ir/scripts/docker_gate.py')
        with tempfile.TemporaryDirectory() as td:
            env_file = Path(td) / 'machine.env'
            env_file.write_text('ASCENDNPU_CANN_SETUP=\n')
            values = {'ASCENDNPU_ENV_FILE': str(env_file),
                      'ASCENDNPU_CONTAINER_NAME': 'fixture',
                      'ASCENDNPU_CONTAINER_USER': 'builder',
                      'ASCENDNPU_CONTAINER_HOME': '/home/builder'}
            with patch.dict(os.environ, values, clear=True), patch.object(module.subprocess, 'run') as run:
                with self.assertRaises(module.BridgeError) as error:
                    module.main(['--check-config'])
                self.assertIn('read only from', str(error.exception))
                run.assert_not_called()

    def test_every_manifest_declares_a_pinned_preflight(self):
        # The property is "every recipe is preceded by a preflight run by the
        # pinned script", not "the flag is literally --check-config". An Adapter
        # whose recipes need different preflights - a device smoke run needs far
        # less than a build - declares one per recipe, and all of them are still
        # preflights driven by a file the manifest pins.
        for path in ADAPTERS.glob('*/manifest.json'):
            manifest = json.loads(path.read_text())
            for recipe in manifest['recipes']:
                check = recipe['configCheck']
                self.assertTrue(check[-1].startswith('--check-config'), (path, check))
                self.assertIn(check[2].removeprefix('{pack}/'), manifest['files'])

    def test_the_distribution_declares_the_deployment_files_it_ships(self):
        # A wheel that omits `docker/` can run Gates but cannot create the
        # container they address. Each file is named explicitly rather than
        # matched with `docker/*`, because that glob also matched a real machine
        # `config.env` - git-ignore rules do not apply to package-data. This
        # test asserts the property, not just the presence of a pattern.
        data = (ROOT / 'pyproject.toml').read_text()
        # Only the data list matters; the comment above it explains the glob.
        listing = data.split('[tool.setuptools.package-data]', 1)[1]
        listing = '\n'.join(line for line in listing.splitlines()
                            if not line.lstrip().startswith('#'))
        self.assertNotIn('docker/*', listing, 'a docker/* glob would ship a real config.env')
        shipped = ('Dockerfile', 'entry.sh', 'create-container.sh', 'config.example.env', 'README.md')
        for name in shipped:
            self.assertIn(f'"adapters/*/docker/{name}"', listing, f'{name} is not packaged')
        self.assertNotIn('config.env"', listing.replace('config.example.env"', ''),
                         'a real config.env must not be packaged')
        for name in shipped:
            for adapter in ('ascendnpu-ir', 'tilelang-hivm'):
                self.assertTrue((ADAPTERS / adapter / 'docker' / name).is_file(),
                                f'{adapter}/docker/{name}')

    def test_the_packaging_list_covers_the_new_adapters_payload(self):
        # An installed Pack must carry the patches it applies and the task it
        # runs, or the wheel ships an Adapter that cannot do its job. These are
        # directory patterns rather than one file at a time because the patch
        # series is expected to grow; unlike docker/, neither directory can hold
        # machine data.
        listing = (ROOT / 'pyproject.toml').read_text().split('[tool.setuptools.package-data]', 1)[1]
        for pattern in ('"adapters/*/patches/*"', '"adapters/*/tasks/*/*"'):
            self.assertIn(pattern, listing, f'{pattern} is not packaged')
        adapter = ADAPTERS / 'forgekernel-ascend-npu'
        self.assertTrue((adapter / 'patches/series.json').is_file())
        self.assertTrue((adapter / 'tasks/rmsnorm-ascend').is_dir())

    def test_the_packaging_list_covers_every_shipped_deployment_file(self):
        # The explicit list is only safe while it is complete: a new deployment
        # file that nobody adds to pyproject.toml would silently drop out of the
        # wheel instead of being noticed here.
        data = (ROOT / 'pyproject.toml').read_text()
        for adapter in ('ascendnpu-ir', 'tilelang-hivm'):
            for path in sorted((ADAPTERS / adapter / 'docker').iterdir()):
                if not path.is_file() or path.name == 'config.env':
                    continue
                self.assertIn(f'"adapters/*/docker/{path.name}"', data,
                              f'{adapter}/docker/{path.name} is not packaged')

    def test_the_example_container_configuration_holds_no_machine_value(self):
        # Both templates are committed, so neither may carry this machine's
        # account, home or toolchain paths. Generic illustrative shapes such as
        # `/home/<user>` or a `/home:/home` bind are fine; the real file beside
        # each template is where actual values belong.
        machine = ('s00653124', '/home/s00653124', 's00653124_tilelang', '/data/pri',
                   '/data/miniconda3', 'AscendNPU-IR-1', 'hivm-poc-control')
        for adapter in ('ascendnpu-ir', 'tilelang-hivm'):
            template = ADAPTERS / adapter / 'docker' / 'config.example.env'
            text = template.read_text()
            for value in machine:
                self.assertNotIn(value, text, f'{template}: {value}')
            self.assertNotRegex(text, r'ASCEND_RT_VISIBLE_DEVICES=[0-9]')
        # The ForgeKernel template is host-local, so it ships no container file;
        # the same rule still applies to it, through its own name.
        template = ADAPTERS / 'forgekernel-ascend-npu' / 'env.example.sh'
        text = template.read_text()
        for value in machine:
            self.assertNotIn(value, text, f'{template}: {value}')
        self.assertNotRegex(text, r'FK_ASCEND_DEVICE_ID=[0-9]')

    def test_the_container_helper_has_no_literal_device_default(self):
        # `ASCEND_VISIBLE_DEVICES=0,1` as a fallback made a configuration file
        # that merely omitted the line expose cards nobody allocated to it. The
        # only safe default is the empty one: no filtering is the operator's
        # explicit choice, not a side effect of forgetting a key. A template
        # value is not enough - the script itself must carry no fallback.
        for adapter in ('ascendnpu-ir', 'tilelang-hivm'):
            script = (ADAPTERS / adapter / 'docker' / 'create-container.sh').read_text()
            self.assertIn('ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES-}"', script,
                          f'{adapter}/docker/create-container.sh has a device default')
            self.assertNotRegex(script, r'ASCEND_VISIBLE_DEVICES:-[0-9]',
                                f'{adapter}/docker/create-container.sh defaults to devices')


if __name__ == '__main__':
    unittest.main()
