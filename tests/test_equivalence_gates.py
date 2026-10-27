"""Offline comparison against frozen original contracts for all five gates."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from agent_gates.evaluator import evaluate
from agent_gates.registry import Registry

FIXTURES = Path(__file__).parent / 'fixtures'
_spec = importlib.util.spec_from_file_location('legacy_oracle', FIXTURES / 'legacy_oracle.py')
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


def contracts():
    result = []
    for path in sorted((FIXTURES / 'legacy').glob('*.json')):
        if path.name == 'origin.json':
            continue
        manifest = json.loads(path.read_text())
        for gate in manifest['gates']:
            result.append((manifest['id'], gate))
    return result


CONTRACTS = contracts()


def test_frozen_contract_fixture_identity():
    origin = json.loads((FIXTURES / 'legacy' / 'origin.json').read_text())
    assert len(CONTRACTS) == 5
    for record in origin['manifests']:
        path = FIXTURES / 'legacy' / (record['adapter'] + '.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['manifestSha256']


def test_isolated_cli_lists_all_five_gates_without_harness_runtime(tmp_path):
    package = Path(__file__).parents[1] / 'agent_gates'
    shutil.copytree(package, tmp_path / 'agent_gates', ignore=shutil.ignore_patterns('__pycache__', 'env.sh'))
    runner = tmp_path / 'standalone.py'
    runner.write_text('import sys\nsys.path.insert(0, sys.argv[1])\nfrom agent_gates.cli import main\nraise SystemExit(main(["list", "--json"]))\n')
    result = subprocess.run([sys.executable, '-I', str(runner), str(tmp_path)], cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert {gate for adapter in report['adapters'] for gate in adapter['gates']} == {gate['id'] for _, gate in CONTRACTS}
    assert not (tmp_path / 'engineering').exists()


def valid_observations(gate):
    observations = dict(gate['observationChecks'])
    if 'testsCollectedPositive' in observations:
        observations.update(testsCollected=4, testsSkipped=0, testsPassed=4)
    return observations


@pytest.mark.parametrize('adapter,gate', CONTRACTS, ids=[gate['id'] for _, gate in CONTRACTS])
@pytest.mark.parametrize('scenario', ['pass', 'nonzero', 'timeout', 'interrupted', 'missing', 'false', 'malformed', 'nonobject', 'too-large'])
def test_five_gate_equivalence(tmp_path, adapter, gate, scenario):
    checks = gate['observationChecks']
    observations = valid_observations(gate)
    process = dict(exitCode=0, timedOut=False, interrupted=False)
    stdout = json.dumps(observations)
    if scenario == 'nonzero':
        process['exitCode'] = 9
    elif scenario == 'timeout':
        process['timedOut'] = True
    elif scenario == 'interrupted':
        process['interrupted'] = True
    elif scenario in {'missing', 'false'}:
        if checks == {'exitCodeZero': True}:
            # Exit-code gates deliberately ignore any stdout claims.
            observations = {'exitCodeZero': False}
        elif scenario == 'missing':
            observations.pop(next(iter(checks)))
        else:
            observations[next(iter(checks))] = False
        stdout = json.dumps(observations)
    elif scenario == 'malformed':
        stdout = '{invalid'
    elif scenario == 'nonobject':
        stdout = '[]'
    elif scenario == 'too-large':
        stdout = ' ' * 65537 + stdout
    path = tmp_path / 'stdout.log'
    path.write_text(stdout)
    process['stdout'] = {'path': str(path)}
    old = _oracle.legacy_decision(process, checks, stdout)
    new = evaluate(process, checks)['status']
    assert (new == 'PASS') == (old == 'PASS')


@pytest.mark.parametrize('adapter,gate', CONTRACTS, ids=[gate['id'] for _, gate in CONTRACTS])
@pytest.mark.parametrize('scenario', ['pass', 'nonzero', 'timeout', 'interrupted', 'missing', 'false', 'malformed', 'nonobject', 'too-large'])
def test_five_gate_equivalence(tmp_path, adapter, gate, scenario):
    checks = gate['observationChecks']
    observations = valid_observations(gate)
    process = dict(exitCode=0, timedOut=False, interrupted=False)
    stdout = json.dumps(observations)
    if scenario == 'nonzero':
        process['exitCode'] = 9
    elif scenario == 'timeout':
        process['timedOut'] = True
    elif scenario == 'interrupted':
        process['interrupted'] = True
    elif scenario in {'missing', 'false'}:
        if checks == {'exitCodeZero': True}:
            # Exit-code gates deliberately ignore any stdout claims.
            observations = {'exitCodeZero': False}
        elif scenario == 'missing':
            observations.pop(next(iter(checks)))
        else:
            observations[next(iter(checks))] = False
        stdout = json.dumps(observations)
    elif scenario == 'malformed':
        stdout = '{invalid'
    elif scenario == 'nonobject':
        stdout = '[]'
    elif scenario == 'too-large':
        stdout = ' ' * 65537 + stdout
    path = tmp_path / 'stdout.log'
    path.write_text(stdout)
    process['stdout'] = {'path': str(path)}
    old = _oracle.legacy_decision(process, checks, stdout)
    new = evaluate(process, checks)['status']
    assert (new == 'PASS') == (old == 'PASS')


# Migrated gates must keep their original rules, with one narrow exception: a
# recipe whose timeout was measurably too short to ever finish. The suite records
# each such deviation here by name instead of loosening the comparison, so a
# future silent edit still fails and a deliberate one has to be written down.
#
# ascendnpu-ir.regression: the recipe allowed 1140s, raised to 5340s inner /
# 5400s outer as slack for an unbuilt tree. The FIRST real run of this gate
# afterwards measured 1142 tests in 2.7s wall clock (256 parallel lit workers,
# 1009 passed / 133 unsupported / 0 failed), so the original bound was in fact
# sufficient on a warm tree. The deviation is kept as deliberate headroom.
# Argv shape and the trailing target are unchanged; only the bound moved.
_TIMEOUT_DEVIATIONS = {
    ('ascendnpu-ir', 'ascendnpu-ir.regression'): {
        'old': ('1140', 1200),
        'new': ('5340', 5400),
    },
}


def _accounted_deviation(adapter, recipe_id, argv, old_argv, timeout, old_timeout):
    """Return the argv/timeout to compare, substituting a declared deviation back."""
    deviation = _TIMEOUT_DEVIATIONS.get((adapter, recipe_id))
    if deviation is None:
        return argv, timeout
    old_value, old_bound = deviation['old']
    new_value, new_bound = deviation['new']
    # The deviation is legitimate only when it is EXACTLY the declared one. The
    # recipe timeout is a separate field from the container-side value that sits
    # in argv, so both are compared: a silent change to either fails here.
    assert timeout == new_bound and old_bound != new_bound
    assert argv != old_argv
    # Everything except the bound itself must be untouched: same interpreter,
    # same script, same target, same flags and their order.
    assert len(argv) == len(old_argv)
    differing = [index for index, (left, right) in enumerate(zip(argv, old_argv)) if left != right]
    assert differing == [len(argv) - 2], "only the container timeout value may differ"
    assert argv[-2] == new_value and old_argv[-2] == old_value
    return old_argv, old_timeout


@pytest.mark.parametrize('adapter,gate', CONTRACTS, ids=[gate['id'] for _, gate in CONTRACTS])
def test_migrated_gate_retains_original_rules(adapter, gate):
    manifest = Registry().get(adapter)
    actual = manifest.gates[gate['id']]
    assert actual['recipeId'] == gate['recipeId']
    assert actual['observationChecks'] == gate['observationChecks']
    assert actual.get('dependsOn', []) == gate.get('dependsOn', [])
    old_manifest = json.loads((FIXTURES / 'legacy' / (adapter + '.json')).read_text())
    old_recipe = next(item for item in old_manifest['recipes'] if item['id'] == gate['recipeId'])
    recipe = manifest.recipes[gate['recipeId']]
    assert recipe['argv'] == old_recipe['argv']
    assert recipe.get('cwd', 'pack') == old_recipe.get('cwd', 'pack')
    assert recipe.get('timeout', 60) == old_recipe.get('timeout', 60)
    # Additive preflight on Ascend is allowed, but an existing check cannot disappear.
    if old_recipe.get('configCheck'):
        assert recipe['configCheck'] == old_recipe['configCheck']


@pytest.mark.parametrize('adapter,gate', [(a, g) for a, g in CONTRACTS if g['observationChecks'] != {'exitCodeZero': True}])
def test_boolean_integer_false_pass_is_deliberately_corrected(tmp_path, adapter, gate):
    observations = valid_observations(gate)
    observations[next(iter(gate['observationChecks']))] = 1
    stdout = json.dumps(observations)
    path = tmp_path / 'stdout.log'
    path.write_text(stdout)
    process = dict(exitCode=0, timedOut=False, interrupted=False, stdout={'path': str(path)})
    assert _oracle.legacy_decision(process, gate['observationChecks'], stdout) == 'PASS'
    assert evaluate(process, gate['observationChecks'])['status'] == 'FAIL'
