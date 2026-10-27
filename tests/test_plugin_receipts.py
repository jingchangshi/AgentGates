"""Offline TaskCompleted checks with a fake standalone CLI."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'integrations/claude-code/hooks/task_completed.py'
spec = importlib.util.spec_from_file_location('receipt_hook', SCRIPT)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)
RUN_ID = 'a' * 32


class ReceiptHookCase(unittest.TestCase):
    def request(self, **changes):
        value = {'adapter': 'tilelang-hivm', 'source': '/tmp/source',
                 'gates': ['tilelang-hivm-regression'], 'runId': RUN_ID}
        value.update(changes)
        return {'task_description': 'Required evidence:\n```agent-gates-receipt\n'
                + json.dumps(value) + '\n```'}

    def fake(self, *, status='PASS', valid=True, source='/tmp/source', adapter='tilelang-hivm',
             gates=None, returncode=0, stdout=None):
        self.calls = []
        def run(argv, **kwargs):
            self.calls.append(argv)
            result = {'runId': RUN_ID, 'status': status, 'valid': valid}
            if argv[1] == 'report':
                result.update({'verification': {'status': status, 'valid': valid},
                               'adapter': {'id': adapter}, 'source': {'root': source},
                               'gates': gates if gates is not None else [
                                   {'gateId': 'tilelang-hivm-regression', 'status': status}]})
            return subprocess.CompletedProcess(argv, returncode,
                                               json.dumps(result) if stdout is None else stdout, '')
        return run

    def test_unrelated_task_never_spawns_a_command(self):
        def forbidden(*args, **kwargs):
            self.fail('unrelated task invoked CLI')
        hook.check({'task_subject': 'edit documentation', 'task_description': 'no gate requested'}, forbidden)

    def test_current_passing_requested_receipt_is_accepted(self):
        hook.check(self.request(), self.fake())
        self.assertEqual([call[1] for call in self.calls], ['verify', 'report'])
        self.assertTrue(all('--run-id' in call and RUN_ID in call for call in self.calls))
        self.assertFalse(any('run' in call or 'doctor' in call for call in self.calls))

    def test_failed_stale_blocked_invalid_and_not_run_are_rejected(self):
        for status in ('FAIL', 'STALE', 'BLOCKED_ENV', 'INVALID_EVIDENCE', 'NOT_RUN'):
            with self.subTest(status=status), self.assertRaises(hook.ReceiptError):
                hook.check(self.request(), self.fake(status=status, valid=False))

    def test_string_true_is_not_a_valid_receipt(self):
        with self.assertRaises(hook.ReceiptError):
            hook.check(self.request(), self.fake(valid='true'))

    def test_a_different_source_or_adapter_is_rejected(self):
        for options in ({'source': '/tmp/other'}, {'adapter': 'ascendnpu-ir'}):
            with self.subTest(options=options), self.assertRaises(hook.ReceiptError):
                hook.check(self.request(), self.fake(**options))

    def test_missing_or_failed_required_gate_is_rejected(self):
        for gates in ([], [{'gateId': 'tilelang-hivm-regression', 'status': 'FAIL'}]):
            with self.assertRaises(hook.ReceiptError):
                hook.check(self.request(), self.fake(gates=gates))

    def test_malformed_or_duplicate_explicit_requests_fail_closed(self):
        for text in ('```agent-gates-receipt', '```agent-gates-receipt\n{bad\n```',
                     self.request()['task_description'] * 2):
            with self.assertRaises(hook.ReceiptError):
                hook.check({'task_description': text}, self.fake())

    def test_missing_id_or_relative_source_fails_before_cli(self):
        for changes in ({'runId': ''}, {'runId': '--all'}, {'source': 'relative'},
                        {'gates': []}, {'evidenceRoot': 'relative'}):
            fake = self.fake()
            with self.assertRaises(hook.ReceiptError):
                hook.check(self.request(**changes), fake)
            self.assertEqual(self.calls, [])

    def test_cli_failure_or_invalid_json_fails_closed(self):
        for options in ({'returncode': 1}, {'stdout': ''}, {'stdout': '[]'}, {'stdout': 'not json'}):
            with self.subTest(options=options), self.assertRaises(hook.ReceiptError):
                hook.check(self.request(), self.fake(**options))

    def test_private_store_is_passed_as_one_argv_value(self):
        hook.check(self.request(evidenceRoot='/tmp/evidence with spaces'), self.fake())
        for call in self.calls:
            self.assertIn('/tmp/evidence with spaces', call)


if __name__ == '__main__':
    unittest.main()
