"""Compare bounded machine stdout with typed manifest observations."""
from pathlib import Path
from .manifest import strict_json


def equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(equal(a, e) for a, e in zip(actual, expected))
    return actual == expected


def evaluate(process: dict, checks: dict):
    failures = []
    if type(process.get("exitCode")) is not int or process["exitCode"] != 0:
        failures.append("NONZERO_EXIT")
    if process.get("timedOut"):
        failures.append("TIMEOUT")
    if process.get("interrupted"):
        failures.append("INTERRUPTED")
    observations = {}
    if checks == {"exitCodeZero": True}:
        # Ordinary build output remains valid; retain structured artifact facts
        # when the selected adapter emits them even for an exit-code gate.
        try:
            path = Path(process["stdout"]["path"])
            if path.stat().st_size <= 65536:
                value = strict_json(path.read_text())
                if isinstance(value, dict):
                    observations = value
        except (OSError, ValueError, KeyError):
            pass
    else:
        try:
            path = Path(process["stdout"]["path"])
            if path.stat().st_size > 65536:
                raise ValueError("observation stdout exceeds 65536 bytes")
            observations = strict_json(path.read_text())
            if not isinstance(observations, dict):
                raise ValueError("observation must be an object")
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f"INVALID_OBSERVATION: {exc}")
            observations = {}
    # Two different claims share this name. An Adapter that emits
    # `exitCodeZero` describes a command *inside* the container; the receipt
    # binds the host process that ran the Adapter. For an exit-code-only gate
    # the runner's own status is the whole contract - ordinary build output
    # must not be able to speak, so any produced value is discarded. For every
    # other gate a produced claim is kept, but only as a literal boolean and
    # only when the host process agrees: otherwise a container-reported failure
    # could be hidden by an Adapter that exited 0.
    host_exit_zero = type(process.get("exitCode")) is int and process["exitCode"] == 0 \
        and not process.get("timedOut") and not process.get("interrupted")
    if checks != {"exitCodeZero": True} and "exitCodeZero" in observations:
        observations["exitCodeZero"] = host_exit_zero and observations["exitCodeZero"] is True
    else:
        observations["exitCodeZero"] = host_exit_zero
    for key, expected in checks.items():
        if key not in observations:
            failures.append(f"MISSING_OBSERVATION: {key}")
        elif not equal(observations[key], expected):
            failures.append(f"OBSERVATION_MISMATCH: {key}")
    # A boolean claim is only accepted alongside the counter it summarises. When
    # the cross-check was skipped for an absent counter, a producer could opt
    # out of it: a report containing `testsCollectedPositive: true` and
    # `notAllSkipped: true` with no counts at all passed, which is exactly the
    # zero-test case those checks exist to catch.
    if "testsCollectedPositive" in checks:
        count = observations.get("testsCollected")
        if type(count) is not int or count <= 0:
            failures.append("INVALID_TEST_COUNT")
    if "notAllSkipped" in checks:
        skipped, collected = observations.get("testsSkipped"), observations.get("testsCollected")
        if type(skipped) is not int or type(collected) is not int or skipped < 0 or skipped >= collected:
            failures.append("ALL_TESTS_SKIPPED_OR_INVALID_COUNT")
    return {"status": "FAIL" if failures else "PASS", "observations": observations, "failures": failures}
