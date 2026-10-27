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
    def test_all_three_bridges_find_their_packaged_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'agent_gates/adapters'
            for name in ('ascendnpu-ir', 'tilelang-hivm', 'tilelang-hivm-npu'):
                (root / name).mkdir(parents=True)
            for name in ('ascendnpu-ir', 'tilelang-hivm'):
                (root / name / 'env.example.sh').write_text('# template\n')
            for name in ('ascendnpu-ir', 'tilelang-hivm', 'tilelang-hivm-npu'):
                script = 'npu_gate.py' if name.endswith('-npu') else 'docker_gate.py'
                module = load(name, ADAPTERS / name / 'scripts' / script)
                with patch.object(module, 'ADAPTER_DIR', root / name):
                    expected = 'tilelang-hivm' if name.startswith('tilelang') else name
                    self.assertEqual(module.default_env_file(), root / expected / 'env.sh')

    def test_ascend_check_config_validates_without_spawning_docker(self):
        module = load('ascend_preflight', ADAPTERS / 'ascendnpu-ir/scripts/docker_gate.py')
        with tempfile.TemporaryDirectory() as td:
            env_file = Path(td) / 'machine.env'
            env_file.write_text('# no optional toolchain\n')
            values = {'ASCENDNPU_ENV_FILE': str(env_file), 'ASCENDNPU_CONTAINER_NAME': 'fixture',
                      'ASCENDNPU_CONTAINER_USER': 'builder', 'ASCENDNPU_CONTAINER_HOME': '/home/builder'}
            with patch.dict(os.environ, values, clear=True), patch.object(module.subprocess, 'run') as run:
                self.assertEqual(module.main(['--check-config']), 0)
                run.assert_not_called()
            env_file.write_text('ASCENDNPU_CONDA_ENV=fixture\n')
            with patch.dict(os.environ, values, clear=True), patch.object(module.subprocess, 'run') as run:
                with self.assertRaises(module.BridgeError):
                    module.main(['--check-config'])
                run.assert_not_called()

    def test_every_manifest_declares_a_pinned_preflight(self):
        for path in ADAPTERS.glob('*/manifest.json'):
            manifest = json.loads(path.read_text())
            for recipe in manifest['recipes']:
                check = recipe['configCheck']
                self.assertEqual(check[-1], '--check-config')
                self.assertIn(check[2].removeprefix('{pack}/'), manifest['files'])

    def test_the_distribution_declares_the_deployment_files_it_ships(self):
        # A wheel that omits `docker/` can run Gates but cannot create the
        # container they address. The pattern must stay narrow enough to never
        # match a real `config.env`, which is machine data.
        data = (ROOT / 'pyproject.toml').read_text()
        self.assertIn('adapters/*/docker/*', data)
        for name in ('Dockerfile', 'entry.sh', 'create-container.sh', 'config.example.env'):
            for adapter in ('ascendnpu-ir', 'tilelang-hivm'):
                self.assertTrue((ADAPTERS / adapter / 'docker' / name).is_file(),
                                f'{adapter}/docker/{name}')

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


if __name__ == '__main__':
    unittest.main()
