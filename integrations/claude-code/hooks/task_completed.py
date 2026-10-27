#!/usr/bin/env python3
"""Check explicitly requested receipts; never dispatch a gate or hardware run."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

MARKER = '```agent-gates-receipt'
RUN_ID = re.compile(r'^[0-9a-f]{32}$')
ID = re.compile(r'^[a-z][a-z0-9-]{0,127}$')


class ReceiptError(ValueError):
    pass


def cli_json(arguments, run=subprocess.run):
    # `verify` re-hashes the whole source tree and every bound artifact. On a
    # large checkout with a cold page cache that takes longer than 30s, and a
    # timeout here would read to the user as "receipt invalid" when the
    # receipt is fine. hooks.json allows 75s for this hook; leave room inside.
    try:
        completed = run(['agent-gates', *arguments, '--json'], capture_output=True,
                        text=True, timeout=60, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ReceiptError('verification timed out; the receipt was not evaluated') from exc
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise ReceiptError('agent-gates returned no valid JSON') from exc
    if completed.returncode != 0 or not isinstance(result, dict):
        raise ReceiptError('agent-gates rejected the requested receipt')
    return result


def check(event, run=subprocess.run):
    text = '\n'.join(value for key in ('task_subject', 'task_description')
                     if isinstance(value := event.get(key), str))
    if MARKER not in text:
        return
    matches = re.findall(r'```agent-gates-receipt\s*\n(.*?)\n```', text, re.DOTALL)
    if len(matches) != 1 or text.count(MARKER) != 1:
        raise ReceiptError('provide exactly one complete agent-gates-receipt JSON block')
    try:
        request = json.loads(matches[0])
    except ValueError as exc:
        raise ReceiptError('receipt request is not valid JSON') from exc
    if not isinstance(request, dict) or set(request) - {'adapter', 'source', 'gates', 'runId', 'evidenceRoot'}:
        raise ReceiptError('unsupported receipt request fields')
    for key in ('adapter', 'source', 'runId'):
        if not isinstance(request.get(key), str) or not request[key]:
            raise ReceiptError(f'receipt request needs {key}')
    if not ID.fullmatch(request['adapter']) or not RUN_ID.fullmatch(request['runId']):
        raise ReceiptError('invalid adapter or run ID')
    if not Path(request['source']).is_absolute():
        raise ReceiptError('receipt source must be absolute')
    gates = request.get('gates')
    if not isinstance(gates, list) or not gates or any(not isinstance(gate, str) or not ID.fullmatch(gate) for gate in gates):
        raise ReceiptError('receipt request needs explicit gate IDs')
    arguments = ['--run-id', request['runId']]
    if 'evidenceRoot' in request:
        evidence = request['evidenceRoot']
        if not isinstance(evidence, str) or not Path(evidence).is_absolute():
            raise ReceiptError('evidenceRoot must be absolute')
        arguments += ['--evidence-root', evidence]
    verified = cli_json(['verify', *arguments], run)
    if verified.get('valid') is not True or verified.get('status') != 'PASS' or verified.get('runId') != request['runId']:
        raise ReceiptError('receipt is missing, failed, stale or invalid')
    report = cli_json(['report', *arguments], run)
    if report.get('runId') != request['runId'] or report.get('status') != 'PASS':
        raise ReceiptError('report does not describe the requested passing run')
    verification = report.get('verification', {})
    if not isinstance(verification, dict) or verification.get('valid') is not True or verification.get('status') != 'PASS':
        raise ReceiptError('report freshness verification failed')
    adapter = report.get('adapter', {})
    source = report.get('source', {})
    if not isinstance(adapter, dict) or adapter.get('id') != request['adapter']:
        raise ReceiptError('receipt adapter does not match the task')
    if not isinstance(source, dict) or not isinstance(source.get('root'), str) or Path(source['root']).resolve() != Path(request['source']).resolve():
        raise ReceiptError('receipt source does not match the task')
    observed = report.get('gates')
    if not isinstance(observed, list) or any(not isinstance(item, dict) for item in observed):
        raise ReceiptError('receipt gates are malformed')
    statuses = {item.get('gateId'): item.get('status') for item in observed}
    if any(statuses.get(gate) != 'PASS' for gate in gates):
        raise ReceiptError('a required gate has no passing receipt')


def main():
    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict):
            raise ReceiptError('invalid TaskCompleted input')
        check(event)
        return 0
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        print(f'AgentGates completion blocked: {exc}. Run the explicitly required gates separately and attach their current receipt.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
