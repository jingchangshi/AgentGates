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
    observations["exitCodeZero"] = type(process.get("exitCode")) is int and process["exitCode"] == 0 and not process.get("timedOut") and not process.get("interrupted")
    for key, expected in checks.items():
        if key not in observations:
            failures.append(f"MISSING_OBSERVATION: {key}")
        elif not equal(observations[key], expected):
            failures.append(f"OBSERVATION_MISMATCH: {key}")
    # Cross-check counters when emitted; boolean claims cannot conceal zero/all-skip runs.
    if "testsCollectedPositive" in checks and "testsCollected" in observations:
        count = observations["testsCollected"]
        if type(count) is not int or count <= 0:
            failures.append("INVALID_TEST_COUNT")
    if "notAllSkipped" in checks and "testsSkipped" in observations:
        skipped, collected = observations["testsSkipped"], observations.get("testsCollected")
        if type(skipped) is not int or type(collected) is not int or skipped < 0 or skipped >= collected:
            failures.append("ALL_TESTS_SKIPPED_OR_INVALID_COUNT")
    return {"status": "FAIL" if failures else "PASS", "observations": observations, "failures": failures}
