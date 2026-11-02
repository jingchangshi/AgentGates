"""Host-only regressions for the acceptance protocol (accept_rmsnorm.py).

Everything here runs without a device, a compiler or a lease: these are the
DECISIONS the accept Gate makes, pinned against real measured data captured on
this host. A gate whose negative controls are asserted rather than measured is
not a gate, so every control below carries the numbers it was derived from.

The five negative controls the acceptance protocol requires
-----------------------------------------------------------

  #1 a candidate that delegates to torch_npu.npu_rms_norm  -> refused as inauthentic
  #2 a candidate that accumulates in bf16                  -> refused by precision
  #3 a job missing its marker / exiting non-zero / timing out -> FAIL, never PASS
  #4 a device baseline whose host_bound_ratio is [0.04, 0.88] -> invalid evidence
  #5 a candidate that is merely slower                      -> FAIL, keeps its artifact

Controls #2 and #5 are asserted against the measured matrix in
tests/fixtures/precision_matrix.json, which is a recording of a real run at
8192x4096 and 64x4096 over three input families, on this machine, with this
toolchain. The separation it shows is the gate's discriminating power: correct
fp32 implementations read 0.0041-0.0078 max_rel (0.5-1.0 bf16 ulp) and
deliberate bf16 accumulation reads 0.026-0.043 (3.3-5.5 ulp).
"""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
ADAPTER = ROOT / 'agent_gates/adapters/forgekernel-ascend-npu'
ACCEPT = ADAPTER / 'scripts/accept_rmsnorm.py'
FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACCEPTANCE = load('accept_rmsnorm', ACCEPT)


def matrix():
    return json.loads((FIXTURES / 'precision_matrix.json').read_text())


class PrecisionBoundaryCase(unittest.TestCase):
    """2**-7 is one bf16 ulp, so the boundary must PASS, not fail.

    A correctly rounded bf16 result can land EXACTLY on 2**-7: the vendor and this
    repository's own fp32-accumulating kernel both do, at 8192x4096/normal. A
    strict '<' would therefore fail the vendor against itself. That is the single
    easiest way to break this gate, so it is pinned directly.
    """

    def test_the_ulp_floor_is_exactly_one_ulp(self):
        self.assertEqual(ACCEPTANCE.TASK_FLOOR, 2.0 ** -7)
        self.assertEqual(ACCEPTANCE.ulp_multiple(2.0 ** -7), 1.0)

    def test_a_max_rel_exactly_at_the_floor_passes(self):
        row = {'shape': 'x', 'family': 'normal', 'truth': 'float64',
               'vendor': {'rel_l2': 1e-5, 'max_rel': 2.0 ** -7, 'mean_abs': 1e-8},
               'candidate': {'rel_l2': 1e-5, 'max_rel': 2.0 ** -7, 'mean_abs': 1e-8}}
        verdict = ACCEPTANCE.judge_precision([row])
        self.assertEqual(verdict['verdict'], 'pass', verdict['failures'])

    def test_a_max_rel_just_above_the_floor_fails_when_the_vendor_is_lower(self):
        # One quantum past the floor on one side only: the comparison is strict
        # about the VALUE, not about the equality lane above.
        row = {'shape': 'x', 'family': 'normal', 'truth': 'float64',
               'vendor': {'rel_l2': 1e-5, 'max_rel': 1e-8, 'mean_abs': 1e-8},
               'candidate': {'rel_l2': 1e-5, 'max_rel': 2.0 ** -6.5, 'mean_abs': 1e-8}}
        verdict = ACCEPTANCE.judge_precision([row])
        self.assertEqual(verdict['verdict'], 'fail')

    def test_the_measured_matrix_really_lands_on_the_boundary(self):
        # Not an assumption: 8192x4096/normal vendor max_rel is exactly 2**-7 in
        # the recorded run. If that ever stops being true, this test should be
        # re-derived rather than quietly deleted.
        row = matrix()['shapes']['8192x4096']['families']['normal']
        self.assertEqual(row['vendor_max_rel'], 2.0 ** -7)
        self.assertEqual(ACCEPTANCE.ulp_multiple(row['vendor_max_rel']), 1.0)


class NegativeControlOneDelegation(unittest.TestCase):
    """#1: a candidate that calls the vendor operator must be refused.

    The alias is built exactly the way that defeated a name regex once already:
    importlib + string concatenation + getattr. No name in this source appears in
    any deny list, and the walk must still see it.
    """

    DELEGATING = '''import importlib
import torch
_backend = importlib.import_module("torch" + "_npu")
_fusion = getattr(_backend, "npu_rms" + "_norm")
def run(x, w, eps=1e-6):
    return _fusion(x, w, eps)[0]
'''

    GENUINE = '''import torch, tilelang
import tilelang.ascend.language as T
def run(x, w, eps=1e-6):
    return _KERNEL(x, w, eps)
'''

    def artifact(self, source, extra=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / 'kernel.py').write_text(source)
        for name, text in (extra or {}).items():
            (root / name).write_text(text)
        return root

    def test_the_aliased_delegation_is_caught_and_named(self):
        scan, artifact = ACCEPTANCE.scan_artifact(self.artifact(self.DELEGATING))
        verdict = ACCEPTANCE.judge_authenticity(scan, artifact, None, None)
        self.assertEqual(verdict['verdict'], 'fail')
        details = json.dumps(verdict['static']['delegations'])
        self.assertIn('npu_rms_norm', details)
        self.assertIn('getattr', details)
        # The dynamic import of the backend itself is allowed; it is recorded as
        # a warning, not counted as the defect.
        self.assertTrue(verdict['static']['warnings'])
        self.assertFalse(verdict['static']['unresolvedDynamic'])

    def test_a_plain_dotted_call_is_caught(self):
        source = ('import torch_npu\n'
                  'def run(x, w, eps=1e-6):\n'
                  '    return torch_npu.npu_rms_norm(x, w, eps)[0]\n')
        scan, artifact = ACCEPTANCE.scan_artifact(self.artifact(source))
        verdict = ACCEPTANCE.judge_authenticity(scan, artifact, None, None)
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('npu_rms_norm', json.dumps(verdict['static']['delegations']))

    def test_importing_the_backend_for_registration_is_allowed(self):
        # torch_npu must be importable: no npu tensor exists without it. A genuine
        # kernel that imports it and never calls a fusion operator passes the
        # static leg (it still needs its compile and runtime evidence, which this
        # assertion supplies as if it had them).
        source = ('import torch\nimport torch_npu\n'
                  'import tilelang\nimport tilelang.ascend.language as T\n'
                  'def run(x, w, eps=1e-6):\n'
                  '    return x\n')
        scan, artifact = ACCEPTANCE.scan_artifact(self.artifact(source))
        verdict = ACCEPTANCE.judge_authenticity(
            scan, artifact, {'launchedTilelangKernel': True, 'vendorOpObserved': False},
            {'aibinCount': 1})
        self.assertEqual(verdict['verdict'], 'pass', verdict['reasons'])

    def test_a_reachable_helper_is_walked_even_though_it_files_differ(self):
        # "Do not scan only this round's diff": the delegation lives in a helper
        # module the entry imports, and the archive is self-contained, so every
        # module it ships is reachable by definition.
        root = self.artifact(self.GENUINE,
                             {'vendor_bridge.py': self.DELEGATING.replace(
                                 'def run(', 'def compute(')})
        scan, artifact = ACCEPTANCE.scan_artifact(root)
        verdict = ACCEPTANCE.judge_authenticity(scan, artifact, None, None)
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('vendor_bridge.py',
                      json.dumps(verdict['static']['delegations']))

    def test_a_name_built_at_run_time_is_refused_as_unbounded(self):
        source = ('def run(x, w, eps=1e-6):\n'
                  '    name = "".join([chr(110), chr(112), chr(117)])\n'
                  '    return globals().get(name)(x, w, eps)[0]\n')
        scan, artifact = ACCEPTANCE.scan_artifact(self.artifact(source))
        verdict = ACCEPTANCE.judge_authenticity(scan, artifact, None, None)
        self.assertEqual(verdict['verdict'], 'fail')

    def test_the_same_source_without_its_compile_evidence_is_refused(self):
        # Static pass alone is not acceptance: a source that never produced a
        # device binary during THIS run has no evidence it ran.
        scan, artifact = ACCEPTANCE.scan_artifact(self.artifact(self.GENUINE))
        verdict = ACCEPTANCE.judge_authenticity(
            scan, artifact, {'launchedTilelangKernel': True, 'vendorOpObserved': False},
            {'aibinCount': 0})
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('aibin', json.dumps(verdict['reasons']))

    def test_a_run_that_launches_no_kernel_is_refused(self):
        scan, artifact = ACCEPTANCE.scan_artifact(self.artifact(self.GENUINE))
        verdict = ACCEPTANCE.judge_authenticity(
            scan, artifact, {'kernelNames': ['AscendCL@aclrtSynchronizeDevice'],
                             'launchedTilelangKernel': False, 'vendorOpObserved': False},
            {'aibinCount': 1})
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('no TileLang-generated kernel', json.dumps(verdict['reasons']))


class NegativeControlTwoBf16Accumulation(unittest.TestCase):
    """#2: deliberate bf16 accumulation must really be refused.

    The numbers are the recorded run, not an assertion about one: on every one of
    the six (shape, family) combinations a bf16-accumulating kernel reads 3.3-5.5
    ulp while a correct fp32-accumulating kernel and the vendor read 0.0-1.0 ulp.
    """

    def rows(self):
        data = matrix()
        rows = []
        for shape, block in data['shapes'].items():
            for family, row in block['families'].items():
                rows.append({'shape': shape, 'family': family, 'truth': 'float64',
                             'vendor': {'rel_l2': row['vendor_rel_l2'],
                                        'max_rel': row['vendor_max_rel'],
                                        'mean_abs': row['vendor_mean_abs']},
                             'candidate': {'rel_l2': row['bf16_rel_l2'],
                                           'max_rel': row['bf16_max_rel'],
                                           'mean_abs': row['bf16_mean_abs']}})
        return rows

    def test_the_bf16_accumulating_candidate_fails_every_family(self):
        verdict = ACCEPTANCE.judge_precision(self.rows())
        self.assertEqual(verdict['verdict'], 'fail')
        failed = {item['shape'] + '/' + item['family'] for item in verdict['rows']
                  if item['status'] == 'fail'}
        self.assertEqual(len(failed), 6, failed)

    def test_the_separation_is_in_ulps_not_in_percent_margin(self):
        # The honest bound is the MEASURED one, not a remembered one. Across the
        # six combinations the bf16-accumulating kernel reads 2.05-5.53 ulp, and
        # the correct one 0.00-1.00. The worst case (64x4096/near_zero_rows,
        # 2.05 ulp) is still twice the 1.0-ulp floor and 2.05x its limit, so the
        # rejection holds everywhere - but quoting "3.4-5.5 ulp" as the range
        # would overstate the margin, and the gap is what the gate rests on.
        verdict = ACCEPTANCE.judge_precision(self.rows())
        for item in verdict['rows']:
            self.assertGreater(item['ulp_cand'] / item['ulp_limit'], 1.4, item)
            self.assertLessEqual(item['ulp_ref'], 1.0, item)

    def test_the_worst_measured_separation_is_still_a_rejection(self):
        # Pinned separately so a future dataset cannot quietly narrow it.
        verdict = ACCEPTANCE.judge_precision(self.rows())
        worst = min(item['ulp_cand'] / item['ulp_limit'] for item in verdict['rows'])
        self.assertGreater(worst, 1.4)
        self.assertEqual(verdict['verdict'], 'fail')

    def test_the_honest_fp32_candidate_passes_on_the_same_data(self):
        # The control that makes the negative meaningful: same pipeline, same
        # tolerances, and it passes.
        data = matrix()
        rows = []
        for shape, block in data['shapes'].items():
            for family, row in block['families'].items():
                rows.append({'shape': shape, 'family': family, 'truth': 'float64',
                             'vendor': {'rel_l2': row['vendor_rel_l2'],
                                        'max_rel': row['vendor_max_rel'],
                                        'mean_abs': row['vendor_mean_abs']},
                             'candidate': {'rel_l2': row['fp32_rel_l2'],
                                           'max_rel': row['fp32_max_rel'],
                                           'mean_abs': row['fp32_mean_abs']}})
        verdict = ACCEPTANCE.judge_precision(rows)
        self.assertEqual(verdict['verdict'], 'pass', verdict['failures'])

    def test_the_floor_has_room_for_an_honest_kernel(self):
        # The lead's question: when e_ref is tiny the limit degenerates to the
        # floor. It must still be far wider than an honest fp32 kernel's error.
        data = matrix()
        for shape, block in data['shapes'].items():
            for family, row in block['families'].items():
                limit = ACCEPTANCE.precision_limit(row['vendor_rel_l2'])
                self.assertLess(row['fp32_rel_l2'] / limit, 0.05,
                                (shape, family, 'an honest kernel used over 5 percent '
                                                'of the limit'))


class NegativeControlThreeJobEvidence(unittest.TestCase):
    """#3: a job that did not finish cleanly is FAIL, never PASS, never SKIP."""

    GOOD_STDOUT = '{"ok": true, "value": 1}'

    def record(self, **overrides):
        base = {'exitCode': 0, 'timedOut': False, 'interrupted': False, 'reason': None,
                'completionMarker': True, 'collected': 2, 'skipped': 0}
        base.update(overrides)
        return base

    def test_a_clean_job_passes(self):
        verdict = ACCEPTANCE.judge_job_evidence(self.record(), self.GOOD_STDOUT)
        self.assertEqual(verdict['verdict'], 'pass', verdict['reasons'])

    def test_a_missing_marker_fails(self):
        verdict = ACCEPTANCE.judge_job_evidence(
            self.record(completionMarker=False), self.GOOD_STDOUT)
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('completion marker', json.dumps(verdict['reasons']))

    def test_a_non_zero_exit_fails_even_with_a_plausible_report(self):
        verdict = ACCEPTANCE.judge_job_evidence(self.record(exitCode=1), self.GOOD_STDOUT)
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('exited 1', json.dumps(verdict['reasons']))

    def test_a_timeout_fails_even_with_a_log(self):
        verdict = ACCEPTANCE.judge_job_evidence(
            self.record(timedOut=True, exitCode=None), self.GOOD_STDOUT)
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('deadline', json.dumps(verdict['reasons']))

    def test_an_all_skipped_run_is_not_a_pass(self):
        verdict = ACCEPTANCE.judge_job_evidence(
            self.record(collected=3, skipped=3), self.GOOD_STDOUT)
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('skipped', json.dumps(verdict['reasons']))

    def test_a_false_observation_check_fails(self):
        verdict = ACCEPTANCE.judge_job_evidence(
            self.record(), self.GOOD_STDOUT, checks={'numericalOk': False})
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('numericalOk', json.dumps(verdict['reasons']))

    def test_an_oversized_report_fails(self):
        verdict = ACCEPTANCE.judge_job_evidence(
            self.record(), json.dumps({'padding': 'x' * (70 * 1024)}))
        self.assertEqual(verdict['verdict'], 'fail')
        self.assertIn('limit', json.dumps(verdict['reasons']))

    def test_a_report_that_is_not_one_object_fails(self):
        verdict = ACCEPTANCE.judge_job_evidence(self.record(), 'not json at all')
        self.assertEqual(verdict['verdict'], 'fail')

    def test_evidence_is_read_from_a_real_run_directory(self):
        # The same shape the engine writes: a process-result.json beside a
        # stdout.log and the job log carrying the completion marker.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs' / 'abc' / 'gate'
            run.mkdir(parents=True)
            (run / 'process-result.json').write_text(json.dumps(
                {'exitCode': 0, 'timedOut': False, 'interrupted': False, 'reason': None}))
            (run / 'stdout.log').write_text(self.GOOD_STDOUT)
            (run / 'job.log').write_text('[gpu_entry.sh] task=all finished\n')
            evidence = ACCEPTANCE.read_job_evidence(root)
            verdict = ACCEPTANCE.judge_job_evidence(evidence['record'], evidence['stdout'])
            self.assertEqual(verdict['verdict'], 'pass', verdict['reasons'])
            (run / 'job.log').write_text('[gpu_entry.sh] task=all started\n')
            evidence = ACCEPTANCE.read_job_evidence(root)
            verdict = ACCEPTANCE.judge_job_evidence(evidence['record'], evidence['stdout'])
            self.assertEqual(verdict['verdict'], 'fail')


class NegativeControlFourHostBound(unittest.TestCase):
    """#4: a baseline whose worst workload is host-bound is invalid evidence.

    The instance is the one that was measured to flip the verdict: ratios
    [0.04, 0.88]. The maximum (0.88) blocks the baseline; the ratio of geometric
    means (0.1876) passes it. Both numbers are asserted here so a future change
    that "simplifies" the aggregate cannot silently reintroduce the flip.
    """

    WORKLOADS = [{'name': 'decode', 'hostBoundRatio': 0.88},
                 {'name': 'prefill', 'hostBoundRatio': 0.04}]

    def test_the_aggregated_ratio_flips_the_verdict(self):
        audit = ACCEPTANCE.audit_host_bound(self.WORKLOADS)
        self.assertAlmostEqual(audit['maxRatio'], 0.88)
        self.assertEqual(audit['verdictOnMax'], 'invalid')
        self.assertAlmostEqual(audit['rederivedRatioOfGeomeans'], math.sqrt(0.88 * 0.04),
                               places=6)
        self.assertEqual(audit['verdictOnRederived'], 'valid')
        self.assertTrue(audit['verdictFlips'])
        self.assertFalse(audit['valid'])

    def test_the_worst_workload_is_named_in_the_reason(self):
        audit = ACCEPTANCE.audit_host_bound(self.WORKLOADS)
        self.assertIn('decode', json.dumps(audit['reasons']))

    def test_an_aggregate_that_disagrees_with_the_maximum_is_itself_a_defect(self):
        # This is exactly what the archive currently emits: _task_ref_alone
        # republishes a geometric mean while the emitting side used the maximum.
        audit = ACCEPTANCE.audit_host_bound(
            [{'name': 'decode', 'hostBoundRatio': 0.04},
             {'name': 'prefill', 'hostBoundRatio': 0.88, 'hostEnqueueMs': 0.88,
              'deviceMedianMs': 1.0}],
            {'hostBoundRatio': 0.1876})
        self.assertFalse(audit['valid'])
        self.assertTrue(audit['aggregateInconsistent'])

    def test_a_fully_device_bound_baseline_is_valid(self):
        # A four-microsecond decode kernel needs a host path far below 2 us to
        # stay device-bound, which is exactly why this workload is the one that
        # goes host-bound first.
        audit = ACCEPTANCE.audit_host_bound(
            [{'name': 'prefill', 'hostEnqueueMs': 0.02, 'deviceMedianMs': 0.5},
             {'name': 'decode', 'hostEnqueueMs': 0.001, 'deviceMedianMs': 0.004}])
        self.assertAlmostEqual(audit['maxRatio'], 0.25)
        self.assertEqual(audit['verdictOnMax'], 'valid')
        self.assertTrue(audit['valid'], audit['reasons'])

    def test_a_workload_without_an_enqueue_clock_invalidates_the_baseline(self):
        # No clock is not a zero ratio: it means the reading cannot be qualified.
        audit = ACCEPTANCE.audit_host_bound([{'name': 'decode', 'deviceMedianMs': 0.004},
                                            {'name': 'prefill', 'hostEnqueueMs': 0.02,
                                             'deviceMedianMs': 0.5}])
        self.assertFalse(audit['valid'])
        self.assertIn('decode', json.dumps(audit['reasons']))

    def test_an_archived_report_is_audited_through_the_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ref.json'
            path.write_text(json.dumps({'performance': {
                'host_bound_ratio': 0.1876,
                'workloads': [{'name': 'decode', 'host_enqueue_ms': 0.004,
                               'device_median_ms': 0.1},
                              {'name': 'prefill', 'host_enqueue_ms': 0.9,
                               'device_median_ms': 1.0}]}}))
            archive = json.loads(path.read_text())
            audit = ACCEPTANCE.audit_host_bound(
                ACCEPTANCE._archive_workloads(archive),
                ACCEPTANCE._archive_aggregate(archive))
            self.assertAlmostEqual(audit['maxRatio'], 0.9)
            self.assertEqual(audit['verdictOnMax'], 'invalid')
            self.assertFalse(audit['valid'])


class NegativeControlFiveSlowCandidate(unittest.TestCase):
    """#5: a correct but slower candidate is FAIL, and keeps its artifact.

    The windows are shaped like real paired samples: vendor 0.0206-0.0214 ms,
    candidate 0.055-0.057 ms, both drifting together, which is what a shared card
    looks like. The interval must not rescue it.
    """

    def windows(self, candidate, vendor, count=12, spread=0.02):
        out = []
        for index in range(count):
            drift = 1.0 + spread * ((index % 5) - 2) / 2.0
            vendor_ms = vendor * drift
            candidate_ms = candidate * drift
            out.append({'index': index,
                        'order': ['candidate', 'vendor'] if index % 2 == 0
                        else ['vendor', 'candidate'],
                        'vendor_ms': round(vendor_ms, 6),
                        'candidate_ms': round(candidate_ms, 6),
                        'speedup': vendor_ms / candidate_ms,
                        'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}})
        return out

    def test_a_candidate_three_times_slower_fails(self):
        verdict = ACCEPTANCE.judge_timing(self.windows(0.0560, 0.0210), '8192x4096')
        self.assertEqual(verdict['status'], 'FAIL')
        self.assertLess(verdict['interval95'][1], 1.0)

    def test_the_failure_keeps_the_best_correct_artifact(self):
        result = {'authenticity': {'verdict': 'pass'},
                  'precision': {'verdict': 'pass'},
                  'timing': {'8192x4096': {'status': 'FAIL', 'pointSpeedup': 0.36,
                                           'interval95': [0.35, 0.37]}},
                  'candidate': {'root': '/artifacts/candidates/genuine'}}
        verdict = ACCEPTANCE.decide(result, check_only=False)
        self.assertEqual(verdict['status'], 'FAIL')
        self.assertEqual(verdict['stage'], 'performance')
        self.assertEqual(verdict['bestCorrectArtifact'],
                         '/artifacts/candidates/genuine')

    def test_a_candidate_at_the_same_speed_matches(self):
        verdict = ACCEPTANCE.judge_timing(self.windows(0.0210, 0.0210, spread=0.01),
                                          '64x4096')
        self.assertIn(verdict['status'], ('MATCH', 'EXCEED'), verdict)

    def test_a_partial_win_does_not_become_a_pass(self):
        # Per shape, never a geometric mean: 8192 matches, the decode shape does
        # not. The aggregate must be FAIL, not an averaged "on balance".
        result = {'authenticity': {'verdict': 'pass'},
                  'precision': {'verdict': 'pass'},
                  'timing': {'8192x4096': {'status': 'MATCH', 'pointSpeedup': 1.02,
                                           'interval95': [1.0, 1.04]},
                             '64x4096': {'status': 'FAIL', 'pointSpeedup': 0.86,
                                         'interval95': [0.84, 0.88]}},
                  'candidate': {'root': '/artifacts/candidates/genuine'}}
        verdict = ACCEPTANCE.decide(result, check_only=False)
        self.assertEqual(verdict['status'], 'FAIL')
        self.assertIn('64x4096', json.dumps(verdict['reasons']))


class WindowClassificationCase(unittest.TestCase):
    """A window that cannot be classified is VOID, and a run of them decides nothing."""

    def test_a_window_with_no_witness_is_void(self):
        self.assertEqual(ACCEPTANCE.classify_window({}), 'unclassifiable')

    def test_a_window_with_one_witness_is_degraded_not_void(self):
        # Both witnesses are preferred, but one is enough to label a window. Only
        # when neither answers is the window meaningless.
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}), 'quiet')
        self.assertEqual(ACCEPTANCE.classify_window({'coTenants': []}), 'quiet')

    def test_a_trajectory_peak_makes_the_window_contended(self):
        # The before/after readings both say 0; only the samples DURING the window
        # saw the card go busy. This is precisely what a single pair of readings
        # cannot detect, and why the trajectory is a witness in its own right.
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
             'aicoreUtilTrajectory': [0, 0, 97, 88, 0]}), 'contended')

    def test_a_flat_trajectory_is_quiet(self):
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
             'aicoreUtilTrajectory': [0, 0, 1, 0]}), 'quiet')

    def test_a_trajectory_alone_is_a_witness(self):
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilTrajectory': [0, 0, 0]}), 'quiet')

    def test_an_unreadable_npu_smi_leaves_the_window_void(self):
        witness = {'aicoreUtilBefore': None, 'aicoreUtilAfter': None,
                   'heartbeatAdvanced': None}
        self.assertEqual(ACCEPTANCE.classify_window(witness), 'unclassifiable')

    def test_a_quiet_window_is_quiet(self):
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}), 'quiet')

    def test_a_co_tenant_seen_in_the_process_table_makes_the_window_contended(self):
        # The utilisation counter is a coarse, sampled reading; the driver's process
        # table is not. A neighbour that a 1 Hz sample misses entirely still shows up
        # here, which is why it is a witness in its own right.
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
             'coTenants': [4066195]}), 'contended')

    def test_an_empty_co_tenant_list_does_not_make_a_window_contended(self):
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 1, 'aicoreUtilAfter': 2, 'coTenants': []}), 'quiet')

    def test_a_busy_card_makes_the_window_contended(self):
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 90}), 'contended')

    def test_an_advanced_heartbeat_makes_the_window_contended(self):
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
             'heartbeatAdvanced': True}), 'contended')

    def test_contended_windows_are_labelled_and_excluded_from_the_sample(self):
        # Contended windows are recorded and shown, but they do NOT enter the
        # estimate: their shift is systematic (measured at 2.05x), and a median
        # resists noise but not bias.
        windows = [{'vendor_ms': 0.021, 'candidate_ms': 0.021, 'speedup': 1.0,
                    'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}}
                   for _ in range(10)]
        # A contended window so fast it would drag the median upward if counted.
        windows += [{'vendor_ms': 0.021, 'candidate_ms': 0.002, 'speedup': 10.0,
                     'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 99}}
                    for _ in range(2)]
        verdict = ACCEPTANCE.judge_timing(windows, '8192x4096')
        self.assertIn(verdict['status'], ('MATCH', 'EXCEED'))
        self.assertEqual(verdict['contendedWindows'], 2)
        self.assertEqual(verdict['quietWindows'], 10)
        self.assertEqual(verdict['decisionSampleWindows'], 10)
        self.assertAlmostEqual(verdict['pointSpeedup'], 1.0)

    def test_contended_windows_never_reach_the_verdict(self):
        # The quiet-fraction gate alone would let a contaminated window through as
        # long as enough quiet ones exist. This pins the stronger property: with the
        # gate SATISFIED (75%), the verdict is computed from the quiet windows only.
        # A contended arm here reads 0.20x instead of 1.02x, so if any leaked in the
        # point estimate would move off 1.02 and this test would see it.
        windows = [{'vendor_ms': 0.01, 'candidate_ms': 0.01 / 1.02, 'speedup': 1.02,
                    'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}}
                   for _ in range(15)]
        windows += [{'vendor_ms': 0.01, 'candidate_ms': 0.01 / 0.20, 'speedup': 0.20,
                     'witness': {'aicoreUtilBefore': 99, 'aicoreUtilAfter': 99}}
                    for _ in range(5)]
        verdict = ACCEPTANCE.judge_timing(windows, '8192x4096')
        self.assertEqual(verdict['decisionSampleWindows'], 15)
        self.assertEqual(verdict['quietWindows'], 15)
        self.assertEqual(verdict['contendedWindows'], 5)
        self.assertAlmostEqual(verdict['pointSpeedup'], 1.02)
        self.assertNotEqual(verdict['status'], 'FAIL')

    def test_an_unlabelled_shortfall_is_not_reported_as_a_contention_shortfall(self):
        # Both RETEST branches have to stay reachable and keep distinct reasons. The
        # unlabelled check must run FIRST: a quiet sample is a subset of the usable
        # windows, so if the quiet rule were evaluated first it would always fire and
        # this message would be unreachable code that only looked like a guard.
        windows = [{'vendor_ms': 0.01, 'candidate_ms': 0.01, 'speedup': 1.0,
                    'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}}
                   for _ in range(2)]
        windows += [{'vendor_ms': 0.01, 'candidate_ms': 0.01, 'speedup': 1.0,
                     'witness': {}} for _ in range(8)]
        verdict = ACCEPTANCE.judge_timing(windows, '64x4096')
        self.assertEqual(verdict['status'], 'RETEST')
        self.assertIn('could not be labelled', verdict['why'])

    def test_a_high_contention_fraction_invalidates_the_sample(self):
        # 6 quiet of 10 is under the 70 percent rule: the windows being excluded are
        # excluded systematically, so what is left is a selected sample.
        windows = [{'vendor_ms': 0.021, 'candidate_ms': 0.021, 'speedup': 1.0,
                    'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}}
                   for _ in range(6)]
        windows += [{'vendor_ms': 0.021, 'candidate_ms': 0.021, 'speedup': 1.0,
                     'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 99}}
                    for _ in range(4)]
        verdict = ACCEPTANCE.judge_timing(windows, '64x4096')
        self.assertEqual(verdict['status'], 'RETEST')
        self.assertEqual(verdict['quietFractionOfWindows'], 0.6)

    def test_a_thin_but_majority_quiet_sample_is_still_a_retest(self):
        windows = [{'vendor_ms': 0.021, 'candidate_ms': 0.021, 'speedup': 1.0,
                    'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}}
                   for _ in range(5)]
        windows += [{'vendor_ms': 0.021, 'candidate_ms': 0.021, 'speedup': 1.0,
                     'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 99}}
                    for _ in range(2)]
        verdict = ACCEPTANCE.judge_timing(windows, '64x4096')
        self.assertEqual(verdict['status'], 'RETEST')

    def test_a_run_with_no_quiet_window_decides_nothing(self):
        # Measured, not assumed: on this host, sustained AICore contention moved an
        # unchanged candidate's paired ratio from 0.3224 to 0.6615 - a factor of
        # 2.05 - because the two arms suffer unequally. A verdict from such a run
        # would attribute contention to the candidate, so it is a RETEST.
        windows = [{'vendor_ms': 0.141, 'candidate_ms': 0.163, 'speedup': 0.66,
                    'witness': {'aicoreUtilBefore': 99, 'aicoreUtilAfter': 99}}
                   for _ in range(12)]
        verdict = ACCEPTANCE.judge_timing(windows, '8192x4096')
        self.assertEqual(verdict['status'], 'RETEST')
        self.assertEqual(verdict['quietWindows'], 0)
        self.assertIn('2.05', json.dumps(verdict['why']))

    def test_too_few_classifiable_windows_is_a_retest_not_a_failure(self):
        windows = [{'vendor_ms': 0.021, 'candidate_ms': 0.060, 'speedup': 0.35,
                    'witness': {'aicoreUtilBefore': None, 'aicoreUtilAfter': None}}
                   for _ in range(5)]
        windows += [{'vendor_ms': 0.021, 'candidate_ms': 0.060, 'speedup': 0.35,
                     'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}}
                    for _ in range(3)]
        verdict = ACCEPTANCE.judge_timing(windows, '64x4096')
        self.assertEqual(verdict['status'], 'RETEST')
        self.assertEqual(verdict['voidWindows'], 5)

    def test_a_wildly_unstable_window_set_is_a_retest_not_a_pass(self):
        windows = []
        for index in range(12):
            speedup = 1.2 if index % 2 else 0.8
            windows.append({'vendor_ms': 0.021, 'candidate_ms': 0.021 / speedup,
                            'speedup': speedup,
                            'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}})
        verdict = ACCEPTANCE.judge_timing(windows, '8192x4096')
        self.assertEqual(verdict['status'], 'RETEST')


class CardProcessWitnessCase(unittest.TestCase):
    """The co-tenant witness is read from the driver, not authored by the script.

    Pinned against a REAL recorded npu-smi snapshot, because the table format (the
    divider under the header is '+====', the closing divider is '+----') is exactly
    the kind of thing a hand-written parser gets wrong and a fake fixture would not
    catch.
    """

    SNAPSHOT = '''+-------------------------------------------------------------------------------------------------+
| npu-smi 25.7.rc1                                 Version: 25.7.rc1                              |
+--------+------------------+---------------+-----------------------------------------------------+
| NPU ID | Name             | Health        | Power(W)    Temp(C)           Hugepages-Usage(page) |
|        |                  | Bus-Id        | NPU Util(%) Memory-Usage(MB)  HBM-Usage(MB)         |
+========+==================+===============+=====================================================+
| 0      | Ascend950PR      | Alarm         | 207.4       67                0     / 0             |
|        |                  | 0000:D7:00.0  | 0           0    / 0          7204  / 131072        |
+========+==================+===============+=====================================================+
| 1      | Ascend950PR      | OK            | 208.2       70                0     / 0             |
|        |                  | 0000:55:00.0  | 0           0    / 0          6794  / 131072        |
+========+==================+===============+=====================================================+
+---------------------------+---------------+-----------------------------------------------------+
| NPU ID                    | Process id    | Process name             | Process memory(MB)       |
+===========================+===============+=====================================================+
| 0                         | 4066195       | pytest                   | 700                     |
| 0                         | 4072612       | python3                  | 317                     |
+===========================+===============+=====================================================+
| 1                         | 4076382       | python3                  | 303                     |
| 1                         | 4076869       | python3                  | 291                     |
+===========================+===============+=====================================================+
'''

    def run_on(self, snapshot):
        import subprocess
        original = ACCEPTANCE.subprocess

        class Done:
            def __init__(self, text):
                self.returncode, self.stdout, self.stderr = 0, text, ''

        ACCEPTANCE.subprocess = type('S', (), {
            'run': staticmethod(lambda *a, **k: Done(snapshot)),
            'TimeoutExpired': subprocess.TimeoutExpired,
            'SubprocessError': subprocess.SubprocessError})
        self.addCleanup(lambda: setattr(ACCEPTANCE, 'subprocess', original))
        return ACCEPTANCE

    def test_the_parser_reads_the_real_table(self):
        module = self.run_on(self.SNAPSHOT)
        self.assertEqual(module._card_processes(0), [4066195, 4072612])
        self.assertEqual(module._card_processes(1), [4076382, 4076869])

    def test_a_card_with_no_processes_has_an_empty_list_not_None(self):
        # Empty and unknown are different answers: one says the card is clear, the
        # other says nobody can tell.
        module = self.run_on(self.SNAPSHOT)
        self.assertEqual(module._card_processes(9), [])

    def test_card_snapshot_reads_the_two_line_card_row(self):
        # The split is not obvious: power is decimal but temperature is not, and the
        # utilisation/HBM counters live on the SECOND line with the bus id. A parser
        # that expects one line of six integers reads nothing at all.
        module = self.run_on(self.SNAPSHOT)
        snapshot = module._card_snapshot(0)
        self.assertEqual(snapshot['utilPct'], 0)
        self.assertEqual(snapshot['hbmUsedMb'], 7204)
        self.assertEqual(snapshot['hbmCapacityMb'], 131072)
        self.assertEqual(snapshot['pids'], [4066195, 4072612])

    def test_card_snapshot_separates_unknown_from_zero(self):
        # 'garbage' describes no card, so every reading is unknown - not zero, and
        # not "clear". A blind witness that reports clear certifies contention away.
        unreadable = self.run_on('garbage')._card_snapshot(0)
        self.assertIsNone(unreadable['utilPct'])
        self.assertIsNone(unreadable['hbmUsedMb'])
        self.assertIsNone(unreadable['pids'])

    def test_the_header_divider_does_not_end_the_table(self):
        # The regression this pins: closing the table on '+====' - which npu-smi
        # puts directly under the header - yields no rows at all, and a witness that
        # silently returns [] would mark a busy card as clear.
        module = self.run_on(self.SNAPSHOT)
        self.assertNotEqual(module._card_processes(0), [])

    def test_an_unreadable_card_is_unknown_not_clear(self):
        module = self.run_on('garbage')
        self.assertIsNone(module._card_processes(0))


class CanaryWitnessCase(unittest.TestCase):
    """The canary is the only witness that cannot be stale.

    npu-smi costs 0.5-3.2 s per invocation on this host (measured), so a window
    cannot be sampled at 100-200 ms. A canary kernel runs INSIDE the window at the
    resolution the measurement itself runs at, so it observes contention that a
    before/after poll can walk straight past.
    """

    def test_an_inflated_canary_makes_the_window_contended(self):
        # Every npu-smi reading says the card is idle; the canary says otherwise.
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
             'aicoreUtilTrajectory': [0, 0, 0],
             'coTenants': [], 'canaryInflation': 1.9,
             'samplingRateHz': 2.0}), 'contended')

    def test_a_flat_canary_leaves_the_window_quiet(self):
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
             'coTenants': [], 'canaryInflation': 1.01,
             'samplingRateHz': 2.0}), 'quiet')

    def test_the_canary_threshold_is_a_stated_constant(self):
        self.assertEqual(ACCEPTANCE.CANARY_INFLATION, 1.15)
        self.assertEqual(ACCEPTANCE.classify_window(
            {'canaryInflation': 1.14, 'samplingRateHz': 2.0, 'coTenants': []}), 'quiet')
        self.assertEqual(ACCEPTANCE.classify_window(
            {'canaryInflation': 1.16, 'samplingRateHz': 2.0, 'coTenants': []}),
            'contended')

    def test_a_sparse_window_is_not_allowed_to_claim_quiet(self):
        # The lead's requirement, kept: 'quiet' must be falsifiable. A window that
        # was sampled once over three seconds did not show the card was idle
        # throughout, only that nobody looked.
        self.assertEqual(ACCEPTANCE.classify_window(
            {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
             'coTenants': [], 'canaryInflation': 1.0,
             'samplingRateHz': 0.2}), 'sparse')

    def test_a_sparse_window_is_counted_but_not_decided_on(self):
        # Sparse means "not observed well enough to call it quiet", so it cannot
        # stand as an unbiased sample either.
        windows = [{'vendor_ms': 0.021, 'candidate_ms': 0.021, 'speedup': 1.0,
                    'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
                                'coTenants': [], 'canaryInflation': 1.0,
                                'samplingRateHz': 0.2}} for _ in range(4)]
        windows += [{'vendor_ms': 0.021, 'candidate_ms': 0.021, 'speedup': 1.0,
                     'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
                                 'coTenants': [], 'canaryInflation': 1.0,
                                 'samplingRateHz': 2.0}} for _ in range(10)]
        verdict = ACCEPTANCE.judge_timing(windows, '8192x4096')
        self.assertIn(verdict['status'], ('MATCH', 'EXCEED'))
        self.assertEqual(verdict['sparseWindows'], 4)
        self.assertEqual(verdict['quietWindows'], 10)
        self.assertEqual(verdict['decisionSampleWindows'], 10)
        self.assertEqual(verdict['quietWindowsWithSparseSampling'], 0)

    def test_the_report_states_the_sampling_rate_it_actually_achieved(self):
        # The rate is bounded by npu-smi, not chosen. Saying so is what stops a
        # sparse 'quiet' from being read as an observed one.
        windows = [{'vendor_ms': 0.021, 'candidate_ms': 0.021, 'speedup': 1.0,
                    'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0,
                                'coTenants': [], 'canaryInflation': 1.0,
                                'samplingRateHz': rate}}
                   for rate in (0.5, 1.0, 1.5)]
        verdict = ACCEPTANCE.judge_timing(windows, '64x4096')
        self.assertEqual(verdict['witnessSamplingRateHz']['min'], 0.5)
        self.assertEqual(verdict['witnessSamplingRateHz']['max'], 1.5)
        self.assertIn('npu-smi', verdict['witnessSamplingRateHz']['note'])


class ReportShapeCase(unittest.TestCase):
    """The report is one JSON object, under 64 KiB, with samples and verdict."""

    def report(self):
        return {'schema': 1, 'protocol': ACCEPTANCE.PROTOCOL,
                'verdict': {'status': 'MATCH', 'stage': 'performance', 'reasons': []},
                'timing': {'8192x4096': {
                    'status': 'MATCH', 'pointSpeedup': 1.02,
                    'interval95': [1.0, 1.04],
                    'windows': [{'index': i, 'order': ['a', 'b'],
                                 'vendor_ms': 0.02, 'candidate_ms': 0.0196,
                                 'speedup': 1.02,
                                 'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}}
                                for i in range(40)]}}}

    def capture(self, result):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            ACCEPTANCE.emit(result)
        return buffer.getvalue()

    def test_the_report_is_one_object_within_the_limit(self):
        text = self.capture(self.report())
        self.assertLessEqual(len(text.encode('utf-8')), ACCEPTANCE.MAX_OUTPUT_BYTES)
        decoded = json.loads(text)
        self.assertIn('verdict', decoded)
        self.assertIn('windows', decoded['timing']['8192x4096'])

    def test_an_oversized_report_drops_detail_but_keeps_the_samples_and_verdict(self):
        result = self.report()
        result['padding'] = 'x' * (80 * 1024)
        text = self.capture(result)
        self.assertLessEqual(len(text.encode('utf-8')), ACCEPTANCE.MAX_OUTPUT_BYTES)
        decoded = json.loads(text)
        self.assertIn('sizeGuard', decoded)
        self.assertEqual(decoded['verdict']['status'], 'MATCH')
        self.assertEqual(len(decoded['timing']['8192x4096']['windows']), 40)
        self.assertIn('interval95', decoded['timing']['8192x4096'])


class ExitCodeCase(unittest.TestCase):
    """The status-to-exit mapping is the contract the engine acts on."""

    def test_the_status_mapping_is_frozen(self):
        self.assertEqual(ACCEPTANCE.STATUS_EXITS, {'MATCH': 0, 'EXCEED': 0,
                                                   'CHECKS_ONLY': 0, 'RETEST': 4})

    def test_a_failed_authenticity_short_circuits_everything_else(self):
        result = {'authenticity': {'verdict': 'fail', 'reasons': ['delegated']},
                  'precision': {'verdict': 'pass'},
                  'timing': {'8192x4096': {'status': 'EXCEED', 'pointSpeedup': 5.0,
                                           'interval95': [4.8, 5.2]}}}
        verdict = ACCEPTANCE.decide(result, check_only=False)
        self.assertEqual(verdict['status'], 'FAIL')
        self.assertEqual(verdict['stage'], 'authenticity')

    def test_invalid_baseline_evidence_is_its_own_status(self):
        result = {'authenticity': {'verdict': 'pass'},
                  'hostBoundAudit': {'valid': False, 'reasons': ['host-bound']},
                  'precision': {'verdict': 'pass'},
                  'timing': {'8192x4096': {'status': 'MATCH', 'pointSpeedup': 1.01,
                                           'interval95': [1.0, 1.02]}}}
        verdict = ACCEPTANCE.decide(result, check_only=False)
        self.assertEqual(verdict['status'], 'INVALID_EVIDENCE')
        self.assertEqual(ACCEPTANCE.STATUS_EXITS.get(verdict['status'], 1), 1)

    def test_a_precision_failure_is_a_failure_not_a_retest(self):
        result = {'authenticity': {'verdict': 'pass'},
                  'precision': {'verdict': 'fail', 'failures': ['8192x4096/normal: bf16']},
                  'timing': {'8192x4096': {'status': 'MATCH', 'pointSpeedup': 1.01,
                                           'interval95': [1.0, 1.02]}}}
        verdict = ACCEPTANCE.decide(result, check_only=False)
        self.assertEqual(verdict['status'], 'FAIL')
        self.assertEqual(verdict['stage'], 'precision')

    def test_check_only_never_claims_a_performance_verdict(self):
        result = {'authenticity': {'verdict': 'pass'}}
        verdict = ACCEPTANCE.decide(result, check_only=True)
        self.assertEqual(verdict['status'], 'CHECKS_ONLY')

    def test_the_frozen_shapes_are_the_task_shapes(self):
        self.assertEqual(set(ACCEPTANCE.SHAPES), {'8192x4096', '64x4096'})
        self.assertEqual(ACCEPTANCE.SHAPES['8192x4096'], (8192, 4096))
        self.assertEqual(ACCEPTANCE.SHAPES['64x4096'], (64, 4096))
        self.assertEqual(ACCEPTANCE.EPS, 1e-6)


if __name__ == '__main__':
    unittest.main()
