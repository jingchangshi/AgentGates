"""Isolated negative correctness checks; no Docker, compiler, or device needed."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent_gates.engine import Engine, collect_artifacts
from agent_gates.evaluator import evaluate
from agent_gates.evidence import EvidenceStore
from agent_gates.executor import execute
from agent_gates.locking import ResourceLock
from agent_gates.manifest import Manifest, ManifestError, digest
from agent_gates.registry import Registry
from agent_gates.source_identity import snapshot

ADAPTERS = Path(__file__).parents[1] / 'agent_gates' / 'adapters'


def load_adapter(adapter, filename):
    spec = importlib.util.spec_from_file_location(adapter.replace('-', '_'), ADAPTERS / adapter / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandbox(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'program.txt').write_text('original\n')
    return source


def local_engine(tmp_path, sandbox, *, adapter='fixture', code="print('{}')", config=None, checks=None, timeout=2):
    directory = tmp_path / 'registry' / adapter
    directory.mkdir(parents=True)
    script = directory / 'gate.py'
    script.write_text(code + '\n')
    recipe = {'id': 'fixture.run', 'argv': [sys.executable, '-I', '{pack}/gate.py'], 'cwd': 'source', 'timeout': timeout}
    if config is not None:
        recipe['configCheck'] = [sys.executable, '-I', '{pack}/gate.py', '--check']
        script.write_text('import sys\nif "--check" in sys.argv:\n    raise SystemExit(' + str(config) + ')\n' + code + '\n')
    raw = {'schemaVersion': 1, 'id': adapter, 'files': {'gate.py': digest(script)},
           'permissions': {'sourceWrite': False, 'hostExecution': True}, 'recipes': [recipe],
           'gates': [{'id': 'fixture-gate', 'gateType': 'deterministic-test', 'recipeId': recipe['id'],
                      'observationChecks': checks or {'exitCodeZero': True}}]}
    (directory / 'manifest.json').write_text(json.dumps(raw))
    registry = Registry(tmp_path / 'registry')
    store = EvidenceStore(tmp_path / 'evidence')
    engine = Engine(registry=registry, store=store, lock_root=tmp_path / 'locks')
    return engine, store, directory


@pytest.mark.parametrize('value', [1, 'true', None, [], {}, False])
def test_required_boolean_wrong_types_fail(tmp_path, value):
    path = tmp_path / 'stdout.log'
    path.write_text(json.dumps({'observed': value}))
    process = dict(exitCode=0, timedOut=False, interrupted=False, stdout={'path': str(path)})
    assert evaluate(process, {'observed': True})['status'] == 'FAIL'


@pytest.mark.parametrize('exit_code', [False, True, '0', None, -15, 124])
def test_exit_status_must_be_actual_integer_zero(tmp_path, exit_code):
    process = dict(exitCode=exit_code, timedOut=False, interrupted=False)
    assert evaluate(process, {'exitCodeZero': True})['status'] == 'FAIL'


@pytest.mark.parametrize('text', ['{}', '[]', '{bad', '{"observed":true,"observed":false}', '{"observed":NaN}'])
def test_missing_or_invalid_observation_fails(tmp_path, text):
    path = tmp_path / 'stdout.log'
    path.write_text(text)
    assert evaluate(dict(exitCode=0, timedOut=False, interrupted=False, stdout={'path': str(path)}), {'observed': True})['status'] == 'FAIL'


@pytest.mark.parametrize('collected,skipped', [(0, 0), (3, 3), (True, 0), ('3', 0), (3, -1), (3, '0')])
def test_claimed_positive_boolean_cannot_hide_bad_test_counts(tmp_path, collected, skipped):
    path = tmp_path / 'stdout.log'
    path.write_text(json.dumps(dict(testsCollectedPositive=True, notAllSkipped=True, testsCollected=collected, testsSkipped=skipped)))
    process = dict(exitCode=0, timedOut=False, interrupted=False, stdout={'path': str(path)})
    assert evaluate(process, dict(testsCollectedPositive=True, notAllSkipped=True))['status'] == 'FAIL'


@pytest.mark.parametrize('kind', ['module', 'native'])
def test_host_artifact_mismatch_fails(tmp_path, sandbox, kind):
    base = load_adapter('tilelang-hivm', 'docker_gate.py')
    module = sandbox / 'tilelang' / '__init__.py'
    native = sandbox / 'build' / 'libtilelang.so'
    module.parent.mkdir()
    native.parent.mkdir()
    module.write_text('module')
    native.write_text('library')
    facts = {'pythonModuleRel': 'tilelang/__init__.py', 'pythonModuleSha256': digest(module),
             'nativeLibs': {'build/libtilelang.so': digest(native)}}
    if kind == 'module':
        facts['pythonModuleSha256'] = '0' * 64
    else:
        facts['nativeLibs']['build/libtilelang.so'] = '0' * 64
    observed = base.identity_observations(facts, sandbox)
    field = 'importedModuleSha256MatchesHost' if kind == 'module' else 'loadedNativeSha256MatchesHost'
    assert observed[field] is False


@pytest.mark.parametrize('skipped', ['3', True, -1, None])
def test_regression_invalid_skip_count_cannot_claim_executed_tests(tmp_path, sandbox, skipped):
    base = load_adapter('tilelang-hivm', 'docker_gate.py')
    module = sandbox / 'tilelang' / '__init__.py'
    native = sandbox / 'build' / 'libtilelang.so'
    module.parent.mkdir()
    native.parent.mkdir()
    module.write_text('module')
    native.write_text('library')
    report = {'facts': {'pythonModuleRel': 'tilelang/__init__.py', 'pythonModuleSha256': digest(module),
                        'nativeLibs': {'build/libtilelang.so': digest(native)}},
              'tests': dict(collected=3, passed=3, failed=0, errors=0, skipped=skipped), 'pytestExit': 0, 'junitPresent': True}
    observed = base.regression_observations(report, sandbox)
    path = tmp_path / 'stdout.log'
    path.write_text(json.dumps(observed))
    process = dict(exitCode=0, timedOut=False, interrupted=False, stdout={'path': str(path)})
    checks = dict(importedModuleSha256MatchesHost=True, loadedNativeSha256MatchesHost=True,
                  nativeLibsPresent=True, notAllSkipped=True, testsCollectedPositive=True)
    assert evaluate(process, checks)['status'] == 'FAIL'


def npu_report():
    return dict(tests=dict(collected=3, passed=3, failed=0, errors=0, skipped=0), pytestExit=0, junitPresent=True,
                phyBefore=3, phyAfter=3, tensorMovesToDevice=6, numericalAssertions=3, npuPeakMemoryBytes=1024,
                npuAvailable=True, synchronizeOk=True, npuDeviceCount=1, driverDeviceCount=1)


@pytest.mark.parametrize('change', [dict(tests=dict(collected=0)), dict(tests=dict(collected=3, skipped=3)),
                                   dict(numericalAssertions=0), dict(tensorMovesToDevice=0), dict(npuPeakMemoryBytes=0),
                                   dict(synchronizeOk=False), dict(phyAfter=2), dict(phyBefore=None)])
def test_npu_negative_machine_facts_fail(change):
    npu = load_adapter('tilelang-hivm-npu', 'npu_gate.py')
    report = npu_report()
    report.update(change)
    observed = npu.observations(report, '3')
    assert npu.passed(observed) is False


@pytest.mark.parametrize('field,value', [('numericalAssertions', True), ('tensorMovesToDevice', True),
                                         ('npuPeakMemoryBytes', True), ('phyAfter', True),
                                         ('npuAvailable', 'false')])
def test_npu_raw_fact_types_cannot_manufacture_a_pass(field, value):
    npu = load_adapter('tilelang-hivm-npu', 'npu_gate.py')
    report = npu_report()
    report.update(phyBefore=1, phyAfter=1)
    report[field] = value
    observed = npu.observations(report, '1')
    assert npu.passed(observed) is False


def test_npu_device_assignment_is_mandatory(monkeypatch):
    npu = load_adapter('tilelang-hivm-npu', 'npu_gate.py')
    monkeypatch.delenv('TILELANG_HIVM_DEVICE_ID', raising=False)
    with pytest.raises(npu.BridgeError):
        npu.device_id()


@pytest.mark.parametrize('config', [None, 7])
def test_required_config_blocks_before_recipe(tmp_path, sandbox, config):
    engine, store, _ = local_engine(tmp_path, sandbox, adapter='tilelang-hivm', config=config,
                                     code="from pathlib import Path; Path('recipe-started').write_text('bad')")
    result = engine.run('tilelang-hivm', sandbox, ['fixture-gate'])
    assert result['status'] == 'BLOCKED_ENV'
    assert not (sandbox / 'recipe-started').exists()
    assert store.verify(result['runId'], registry=engine.registry)['valid'] is False


def test_same_device_lease_blocks_second_source_before_dispatch(tmp_path, sandbox):
    engine, _, _ = local_engine(tmp_path, sandbox, adapter='tilelang-hivm-npu', config=0,
                                 code="from pathlib import Path; Path('recipe-started').write_text('bad')")
    with ResourceLock('npu:3', tmp_path / 'locks'):
        result = engine.run('tilelang-hivm-npu', sandbox, ['fixture-gate'], device_id='3')
    assert result['status'] == 'BLOCKED_ENV'
    assert not (sandbox / 'recipe-started').exists()


def test_real_subprocess_nonzero_and_timeout(tmp_path, sandbox):
    for name, code, timeout in [('nonzero', 'raise SystemExit(7)', 2), ('timeout', 'import time; time.sleep(30)', 0.1)]:
        result = execute([sys.executable, '-I', '-c', code], cwd=str(sandbox), env=dict(os.environ), timeout=timeout, log_dir=tmp_path / name)
        assert result['exitCode'] != 0
        assert result['timedOut'] is (name == 'timeout')
        assert evaluate(result, {'exitCodeZero': True})['status'] == 'FAIL'


def test_config_source_mutation_blocks_recipe_before_launch(tmp_path, sandbox):
    engine, _, directory = local_engine(tmp_path, sandbox, config=0,
                                         code="from pathlib import Path; Path('recipe-started').write_text('bad')")
    script = directory / 'gate.py'
    script.write_text("import sys\nfrom pathlib import Path\nif '--check' in sys.argv:\n    Path('program.txt').write_text('changed during config')\n    raise SystemExit(0)\nPath('recipe-started').write_text('bad')\n")
    path = directory / 'manifest.json'
    raw = json.loads(path.read_text())
    raw['files']['gate.py'] = digest(script)
    path.write_text(json.dumps(raw))
    result = engine.run('fixture', sandbox, ['fixture-gate'])
    assert result['status'] == 'STALE'
    assert not (sandbox / 'recipe-started').exists()


def test_source_mutation_during_run_invalidates_pass(tmp_path, sandbox):
    engine, _, _ = local_engine(tmp_path, sandbox, code="from pathlib import Path; Path('program.txt').write_text('changed'); print('{}')")
    result = engine.run('fixture', sandbox, ['fixture-gate'])
    assert result['status'] == 'STALE'


@pytest.mark.parametrize('mutation', ['receipt', 'stdout', 'source', 'manifest', 'script', 'artifact'])
def test_receipt_revalidation_rejects_corruption_and_staleness(tmp_path, sandbox, mutation):
    product = sandbox / 'product.bin'
    product.write_bytes(b'original artifact')
    observed = {'artifacts': [{'path': str(product), 'sha256': digest(product)}]}
    engine, store, directory = local_engine(tmp_path, sandbox, code='print(' + repr(json.dumps(observed)) + ')', checks={'artifacts': observed['artifacts']})
    result = engine.run('fixture', sandbox, ['fixture-gate'])
    assert result['status'] == 'PASS'
    assert store.verify(result['runId'], registry=engine.registry)['status'] == 'PASS'
    if mutation == 'receipt':
        path = Path(result['receiptPath'])
        receipt = json.loads(path.read_text())
        receipt['status'] = 'FAIL'
        path.write_text(json.dumps(receipt))
    elif mutation == 'stdout':
        Path(result['gates'][0]['process']['stdout']['path']).write_text('{}')
    elif mutation == 'source':
        (sandbox / 'program.txt').write_text('changed')
    elif mutation == 'manifest':
        path = directory / 'manifest.json'
        raw = json.loads(path.read_text())
        raw['version'] = 'changed'
        path.write_text(json.dumps(raw))
    elif mutation == 'script':
        (directory / 'gate.py').write_text('changed')
    elif mutation == 'artifact':
        product.write_bytes(b'changed artifact')
    checked = store.verify(result['runId'], registry=engine.registry)
    assert checked['status'] in {'INVALID_EVIDENCE', 'STALE'}
    assert checked['valid'] is False


@pytest.mark.parametrize('relative', ['../outside', '/outside'])
def test_pinned_file_cannot_escape_adapter(tmp_path, sandbox, relative):
    _, _, directory = local_engine(tmp_path, sandbox)
    path = directory / 'manifest.json'
    raw = json.loads(path.read_text())
    raw['files'] = {relative: '0' * 64}
    path.write_text(json.dumps(raw))
    with pytest.raises(ManifestError):
        Manifest(path)


def test_receipt_binds_actual_loaded_native_outside_default_build_dirs(tmp_path, sandbox, monkeypatch):
    import subprocess
    import agent_gates.source_identity as source_identity

    module = sandbox / 'tilelang' / '__init__.py'
    native = sandbox / 'custom-output' / 'libtilelang.so'
    module.parent.mkdir()
    native.parent.mkdir()
    module.write_text('module')
    native.write_bytes(b'actual loaded native bytes')
    base = load_adapter('tilelang-hivm', 'docker_gate.py')
    facts = {'pythonModuleRel': 'tilelang/__init__.py', 'pythonModuleSha256': digest(module),
             'nativeLibs': {'custom-output/libtilelang.so': digest(native)}}
    observed = base.identity_observations(facts, sandbox)
    observed.update(testsCollectedPositive=True, notAllSkipped=True, testsCollected=3, testsSkipped=0)
    checks = {key: True for key in ('importedModuleSha256MatchesHost', 'loadedNativeSha256MatchesHost',
                                   'nativeLibsPresent', 'testsCollectedPositive', 'notAllSkipped')}
    # Model a Git worktree where compiler outputs are correctly git-ignored.
    def fixture_git(root, *args):
        if args == ('rev-parse', '--show-toplevel'):
            stdout = os.fsencode(str(sandbox)) + b'\n'
        elif args == ('rev-parse', 'HEAD'):
            stdout = b'0123456789012345678901234567890123456789\n'
        elif args[:2] == ('config', '--get'):
            # No global ignore file in this fixture; the checkout has none either.
            return subprocess.CompletedProcess(args, 1, stdout=b'', stderr=b'')
        elif args[0] == 'ls-files':
            stdout = b'program.txt\0tilelang/__init__.py\0'
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=b'')
    monkeypatch.setattr(source_identity, 'git', fixture_git)
    engine, store, _ = local_engine(tmp_path, sandbox, adapter='tilelang-hivm', config=0,
                                     code='print(' + repr(json.dumps(observed)) + ')', checks=checks)
    result = engine.run('tilelang-hivm', sandbox, ['fixture-gate'])
    assert result['status'] == 'PASS'
    assert str(native) in {item['path'] for item in result['gates'][0]['artifacts']}
    native.write_bytes(b'replaced loaded native bytes')
    assert store.verify(result['runId'], registry=engine.registry)['valid'] is False


def test_observed_artifact_hash_cannot_disagree_with_file(sandbox):
    product = sandbox / 'product.bin'
    product.write_bytes(b'actual')
    with pytest.raises(ValueError, match='hash differs'):
        collect_artifacts({'artifacts': [{'path': str(product), 'sha256': '0' * 64}]}, sandbox)


@pytest.mark.parametrize('text', ['{not-json', '[]', '{"signature":null}', '{"signature":true}'])
def test_corrupt_attestation_cannot_pass(tmp_path, sandbox, text):
    engine, store, _ = local_engine(tmp_path, sandbox)
    result = engine.run('fixture', sandbox, ['fixture-gate'])
    path = Path(result['receiptPath']).with_name('receipt.attestation.json')
    path.write_text(text)
    assert store.verify(result['runId'], registry=engine.registry)['status'] == 'INVALID_EVIDENCE'


def test_doctor_does_not_hide_latest_completed_receipt(tmp_path, sandbox):
    engine, store, _ = local_engine(tmp_path, sandbox, config=0)
    result = engine.run('fixture', sandbox, ['fixture-gate'])
    assert result['status'] == 'PASS'
    engine.doctor('fixture', sandbox)
    assert store.latest() == result['runId']
    assert store.verify(store.latest(), registry=engine.registry)['valid'] is True


def test_source_byte_identity_detects_untracked_changes(sandbox):
    before = snapshot(sandbox)
    (sandbox / 'new.py').write_text('value = 1\n')
    assert snapshot(sandbox) != before


def _init_repo(path):
    run = lambda *args: subprocess.run(['git', '-C', str(path), *args], check=True,
                                       capture_output=True)
    run('init', '-q')
    run('config', 'user.email', 'gate@example.invalid')
    run('config', 'user.name', 'gate')
    run('config', 'core.excludesFile', '/dev/null')
    (path / 'a.py').write_text('value = 1\n')
    run('add', 'a.py')
    run('commit', '-qm', 'init')


def test_git_exclusion_rules_are_bound_into_source_identity(tmp_path):
    # `ls-files --others --exclude-standard` honours `.git/info/exclude` and
    # `core.excludesFile`. Neither file is part of the tracked tree, so without
    # binding them a writer can hide arbitrary bytes from the snapshot while a
    # previously recorded PASS keeps verifying as current.
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo(repo)
    base = snapshot(repo)
    (repo / '.git' / 'info').mkdir(parents=True, exist_ok=True)
    (repo / '.git' / 'info' / 'exclude').write_text('evil.py\n')
    assert snapshot(repo)['hash'] != base['hash'], 'info/exclude change was invisible'
    (repo / 'evil.py').write_text("value = 'evil'\n")
    assert snapshot(repo)['fileCount'] == base['fileCount'], 'evil.py was enumerated'

    elsewhere = tmp_path / 'global-ignore'
    elsewhere.write_text('other.py\n')
    subprocess.run(['git', '-C', str(repo), 'config', 'core.excludesFile', str(elsewhere)], check=True)
    assert snapshot(repo)['hash'] != base['hash'], 'core.excludesFile change was invisible'


def test_an_unreadable_directory_cannot_hide_from_source_identity(tmp_path):
    # os.walk swallows PermissionError, so before this was handled a directory the
    # snapshot could not list contributed nothing: not to fileCount, not to the
    # digest. Observed for real when a root-owned CANN extra-info/ dump tree landed
    # inside a source directory and every receipt still verified as current. A tree
    # nobody can enumerate must move the hash, not vanish from it.
    work = tmp_path / 'src'
    work.mkdir()
    (work / 'a.py').write_text('value = 1\n')
    base = snapshot(work)
    hidden = work / 'extra-info'
    hidden.mkdir()
    (hidden / 'dump.bin').write_bytes(b'\0' * 8)
    original_mode = hidden.stat().st_mode
    try:
        os.chmod(hidden, 0o000)
        changed = snapshot(work)
        assert changed['hash'] != base['hash'], 'an unreadable directory was invisible to the digest'
    finally:
        os.chmod(hidden, original_mode)
    (hidden / 'dump.bin').unlink()
    hidden.rmdir()
    assert snapshot(work)['hash'] == base['hash'], 'the digest did not return to its original value'


def test_an_unknown_gate_status_can_never_aggregate_to_pass():
    # `aggregate` used to block a fixed list of failure literals and return
    # PASS for everything else, so a gate that recorded SKIPPED or ERROR - a
    # future status, or a receipt written by a bug - produced an overall pass
    # that `verify` then recomputed as valid. PASS must be a positive claim.
    from agent_gates.evidence import aggregate
    for status in ('SKIPPED', 'ERROR', 'PASSED', 'TIMEOUT', '', 'pass', 'Pass'):
        assert aggregate([{'status': status}]) == 'INVALID_EVIDENCE', status
    assert aggregate([]) == 'NOT_RUN'
    assert aggregate([{'status': 'PASS'}]) == 'PASS'
    assert aggregate([{'status': 'PASS'}, {'status': 'PASS'}]) == 'PASS'
    for status in ('FAIL', 'BLOCKED_ENV', 'NOT_RUN', 'STALE', 'INVALID_EVIDENCE'):
        assert aggregate([{'status': 'PASS'}, {'status': status}]) == status


def test_a_receipt_with_an_unknown_status_verifies_as_invalid(tmp_path, sandbox):
    engine, store, _ = local_engine(tmp_path, sandbox)
    result = engine.run('fixture', sandbox, ['fixture-gate'])
    assert result['status'] == 'PASS'
    receipt_path = Path(result['receiptPath'])
    receipt = json.loads(receipt_path.read_text())
    receipt['gates'][0]['status'] = 'SKIPPED'
    receipt_path.write_text(json.dumps(receipt))
    # Re-sign so the only thing wrong is the status, not the HMAC. This is the
    # exact shape a buggy writer would leave behind: a well-formed, correctly
    # signed receipt whose gate status this code has never seen.
    import hashlib, hmac
    from agent_gates.evidence import canonical
    signature = hmac.new((store.root / '.receipt-key').read_bytes(), canonical(receipt), hashlib.sha256).hexdigest()
    (receipt_path.parent / 'receipt.attestation.json').write_text(
        json.dumps({'signature': signature, 'algorithm': 'HMAC-SHA256', 'trust': 'local-user-integrity'}))
    verified = store.verify(result['runId'], registry=engine.registry)
    assert verified['valid'] is False
    assert verified['status'] == 'INVALID_EVIDENCE'


def test_a_supervisor_that_outlives_its_deadline_is_a_runtime_error(tmp_path, monkeypatch):
    # subprocess.TimeoutExpired is a SubprocessError, not an OSError, so it
    # escaped every handler in engine.py and cli.py and lost the receipt
    # instead of recording a failure. The executor must convert it.
    import subprocess as sp
    from agent_gates import executor

    class Hanging:
        returncode = None
        def __init__(self, *a, **k):
            pass
        def wait(self, timeout=None):
            raise sp.TimeoutExpired(cmd='supervisor', timeout=timeout)
        def poll(self):
            return None
        def terminate(self):
            pass

    monkeypatch.setattr(executor.subprocess, 'Popen', Hanging)
    with pytest.raises(RuntimeError):
        executor.execute([sys.executable, '-c', 'pass'], cwd=tmp_path, env={}, timeout=1,
                         log_dir=tmp_path / 'logs')


def test_a_lock_key_normalises_path_spellings(tmp_path):
    # `source:/x`, `source:/x/` and `source:/x/.` hashed to three lock files,
    # so two runs could lease "the same" build tree twice.
    target = tmp_path / 'tree'
    target.mkdir()
    spellings = [str(target), str(target) + '/', str(target) + '/.', str(target / '.' / '')]
    paths = {ResourceLock('source:' + value, tmp_path / 'locks').path for value in spellings}
    assert len(paths) == 1, paths
    # Non-path keys keep their exact spelling: a device lease is not a path.
    assert ResourceLock('npu:3', tmp_path / 'locks').path != ResourceLock('npu:03', tmp_path / 'locks').path


def test_configuration_binds_the_inherited_toolchain_environment(tmp_path, sandbox, monkeypatch):
    # PATH, PYTHONPATH and LD_PRELOAD select the toolchain without appearing in
    # the adapter prefix, so the receipt binds them by hash: two runs under
    # different toolchains must not present the same configuration identity,
    # and no value may be stored. `verify` deliberately does not re-derive
    # this, because the verifying shell is not the running shell.
    from agent_gates.engine import configuration
    from agent_gates.manifest import Manifest
    manifest = Manifest(ADAPTERS / 'tilelang-hivm' / 'manifest.json')
    before = configuration(manifest, {'PATH': '/usr/bin', 'LD_PRELOAD': ''})
    after = configuration(manifest, {'PATH': '/opt/toolchain/bin:/usr/bin', 'LD_PRELOAD': ''})
    assert before['inheritedEnvironmentSha256'] != after['inheritedEnvironmentSha256']
    assert before['environmentSha256'] == after['environmentSha256']
    assert '/opt/toolchain/bin' not in json.dumps(before)
    preload = configuration(manifest, {'PATH': '/usr/bin', 'LD_PRELOAD': '/tmp/inject.so'})
    assert preload['inheritedEnvironmentSha256'] not in (
        before['inheritedEnvironmentSha256'], after['inheritedEnvironmentSha256'])


def test_one_adapter_cannot_inherit_the_other_adapters_prefix(monkeypatch):
    # A stray ASCENDNPU_BUILD_SUBDIR export must not steer a TileLang run
    # through a knob that `configuration()` does not hash for that adapter.
    from agent_gates.engine import environment
    from agent_gates.manifest import Manifest
    monkeypatch.setenv('ASCENDNPU_BUILD_SUBDIR', 'elsewhere')
    monkeypatch.setenv('TILELANG_HIVM_BUILD_SUBDIR', 'build')
    tilelang = Manifest(ADAPTERS / 'tilelang-hivm' / 'manifest.json')
    ascend = Manifest(ADAPTERS / 'ascendnpu-ir' / 'manifest.json')
    assert 'ASCENDNPU_BUILD_SUBDIR' not in environment(tilelang)
    assert environment(tilelang)['TILELANG_HIVM_BUILD_SUBDIR'] == 'build'
    assert 'TILELANG_HIVM_BUILD_SUBDIR' not in environment(ascend)
    assert environment(ascend)['ASCENDNPU_BUILD_SUBDIR'] == 'elsewhere'


def test_nested_commitless_git_directory_is_hashed_via_filesystem_while_top_level_fails_closed(tmp_path):
    empty_top = tmp_path / 'empty_top'
    empty_top.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=empty_top, check=True)
    with pytest.raises(ValueError, match='no readable committed HEAD'):
        snapshot(empty_top)

    parent = tmp_path / 'parent'
    parent.mkdir()
    _init_repo(parent)
    sub = parent / 'work'
    sub.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=sub, check=True)
    (sub / 'k.py').write_text('x = 1\n', encoding='utf-8')
    first = snapshot(parent)
    assert first['mode'] == 'git-files'

    (sub / 'k.py').write_text('x = 2\n', encoding='utf-8')
    second = snapshot(parent)
    assert second['hash'] != first['hash']
