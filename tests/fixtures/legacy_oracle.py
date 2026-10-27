"""Frozen decision rules from PiEngineeringHarness engineering/execution.py:122-165.

This oracle intentionally retains Python's permissive bool/int equality so tests
can distinguish equivalence from documented correctness fixes. It imports no
Harness modules and performs no compiler, Docker, or device execution.
"""
import json


def legacy_decision(process, checks, stdout):
    if process['exitCode'] != 0 or process['timedOut'] or process['interrupted']:
        return 'FAIL'
    if not checks:
        return 'BLOCKED'
    if checks == {'exitCodeZero': True}:
        return 'PASS'
    if len(stdout.encode()) > 65536:
        return 'BLOCKED'
    try:
        observations = json.loads(stdout)
        if not isinstance(observations, dict):
            raise ValueError('observation must be object')
    except ValueError:
        return 'FAIL'
    return 'FAIL' if any(observations.get(key) != expected for key, expected in checks.items()) else 'PASS'
