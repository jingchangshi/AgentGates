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


if __name__ == '__main__':
    unittest.main()
