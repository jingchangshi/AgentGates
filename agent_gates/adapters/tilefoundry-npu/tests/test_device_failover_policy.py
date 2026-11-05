"""ENV-05 device-failover policy: bounded, per-card, driver-only (H addendum).

Only a probe with ALL of timed_out + unreaped + a driver wchan is
BUSY_DEVICE_NOT_RESPONDING and may back off / switch cards. Every other non-
responsive result is a named failure that stops the run (no backoff, no next card,
never PASS). A card merely OCCUPIED by another tenant is normal shared use: use it
immediately, no retry.

These tests drive the REAL policy function with a fake probe, so none needs hardware.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
SCRIPTS = PACK / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_environment as V  # noqa: E402


def result(physical_id, classification, simulated=False, occupancy=None, stderr=""):
    """A fake bounded_run result with the classification the real probe would assign."""
    if classification == V.BUSY_CODE:
        timed_out, unreaped, rc, out, wchan = True, True, None, "", "devdrv_sync_non_tran"
    elif classification == "DEVICE_PROBE_UNATTRIBUTED":
        timed_out, unreaped, rc, out, wchan = True, True, None, "", None
    elif classification == "DEVICE_PROBE_TIMEOUT":
        timed_out, unreaped, rc, out, wchan = True, False, -9, "", None
    elif classification == "DEVICE_PROBE_FAILED":
        timed_out, unreaped, rc, out, wchan = False, False, 2, "", None
    elif classification == "DEVICE_PROBE_EMPTY_OUTPUT":
        timed_out, unreaped, rc, out, wchan = False, False, 0, "   ", None
    elif classification == "RESPONSIVE":
        timed_out, unreaped, rc, out, wchan = False, False, 0, "ok", None
    else:
        raise AssertionError("unknown fake classification: " + classification)
    return {
        "physical_id": physical_id, "simulated": simulated, "probe": "fake",
        "responsive": classification == "RESPONSIVE", "classification": classification,
        "timed_out": timed_out, "unreaped": unreaped, "rc": rc, "pid": -1,
        "wchan": wchan, "stderr_tail": stderr, "seconds": 0.01, "occupancy": occupancy,
    }


def _run(probes, candidates=(0, 1), visible=(0, 1)):
    sleeps = []
    attempts, selected, failure = V.first_responsive_card(
        list(candidates), list(visible), lambda physical: probes[physical], 30,
        sleeper=sleeps.append)
    return attempts, selected, failure, sleeps


# ---- 1. all three pieces present -> BUSY and card switch is allowed -------------
def test_real_d_state_evidence_is_busy_and_may_switch_to_second_card():
    probes = {0: result(0, V.BUSY_CODE, simulated=True),
              1: result(1, "RESPONSIVE")}
    attempts, selected, failure, sleeps = _run(probes)

    assert [a["physical_id"] for a in attempts] == [0, 1]
    assert attempts[0]["outcome"] == V.BUSY_CODE
    assert attempts[0]["timed_out"] and attempts[0]["unreaped"]
    assert attempts[0]["wchan"] == "devdrv_sync_non_tran"
    assert attempts[0]["backoff_seconds"] == 30
    assert sleeps == [30], "exactly one bounded backoff before the second card"
    assert failure is None
    assert selected == {"logical_index": 1, "physical_device_id": 1,
                        "selection_reason": "first RESPONSIVE entry of the ordered visible list"}


# ---- 2. timeout+unreaped but wchan missing/unsupported -> unattributed, NO switch -
def test_timeout_unreaped_without_driver_wchan_is_unattributed_and_does_not_switch():
    probes = {0: result(0, "DEVICE_PROBE_UNATTRIBUTED", simulated=True),
              1: result(1, "RESPONSIVE")}
    attempts, selected, failure, sleeps = _run(probes)

    assert len(attempts) == 1, "an unattributed timeout must not probe the next card"
    assert attempts[0]["outcome"] == "DEVICE_PROBE_UNATTRIBUTED"
    assert attempts[0]["timed_out"] and attempts[0]["unreaped"] and attempts[0]["wchan"] is None
    assert sleeps == [], "no backoff for an unattributed timeout"
    assert selected is None
    assert failure["code"] == "DEVICE_PROBE_UNATTRIBUTED"


# ---- 3. fast rc!=0 -> named CONFIG/BLOCKED, no backoff, no next card ------------
def test_fast_rc_failure_is_named_config_blocked_with_one_attempt():
    probes = {0: result(0, "DEVICE_PROBE_FAILED", simulated=True,
                        stderr="libc_sec.so: cannot open shared object file"),
              1: result(1, "RESPONSIVE")}
    attempts, selected, failure, sleeps = _run(probes)

    assert len(attempts) == 1, "a fast failure must not walk to the next card"
    assert attempts[0]["outcome"] == "DEVICE_PROBE_FAILED"
    assert sleeps == [], "a fast failure must not back off"
    assert selected is None
    assert failure["code"] == "DEVICE_PROBE_FAILED" and failure["rc"] == 2
    assert "libc_sec.so" in failure["stderr_tail"]


# ---- 4. rc==0 empty output -> named invalid result, no switch ------------------
def test_rc_zero_empty_output_is_named_and_does_not_switch():
    probes = {0: result(0, "DEVICE_PROBE_EMPTY_OUTPUT", simulated=True),
              1: result(1, "RESPONSIVE")}
    attempts, selected, failure, sleeps = _run(probes)

    assert len(attempts) == 1
    assert attempts[0]["outcome"] == "DEVICE_PROBE_EMPTY_OUTPUT"
    assert sleeps == [] and selected is None
    assert failure["code"] == "DEVICE_PROBE_EMPTY_OUTPUT"


# ---- 5. all BUSY -> NOT_RUN (passed False), bounded attempts, one backoff gap --
def test_all_cards_busy_is_not_run_with_bounded_attempts():
    probes = {0: result(0, V.BUSY_CODE, simulated=True),
              1: result(1, V.BUSY_CODE, simulated=True)}
    attempts, selected, failure, sleeps = _run(probes)

    assert selected is None and failure is None
    assert len(attempts) == 2 and all(a["outcome"] == V.BUSY_CODE for a in attempts)
    assert sleeps == [30], "one backoff between the two attempts, never after the last"
    # phase_device maps exactly this state to NOT_RUN with passed=False.
    phase_status = "NOT_RUN" if selected is None and failure is None else "OTHER"
    assert phase_status == "NOT_RUN"
    assert not (phase_status == "PASS")


# ---- 6. occupied but responsive -> use immediately, no retry/refusal -----------
def test_an_occupied_but_responsive_card_is_used_without_any_retry():
    probes = {0: result(0, "RESPONSIVE", occupancy="shared(4)"),
              1: result(1, "RESPONSIVE")}
    attempts, selected, failure, sleeps = _run(probes)

    assert len(attempts) == 1, "occupancy must not trigger a second attempt"
    assert sleeps == [], "occupancy must not trigger a backoff"
    assert selected["physical_device_id"] == 0
    assert attempts[0]["occupancy"] == "shared(4)"


def test_policy_defaults_are_the_documented_ones():
    assert V.DEVICE_ATTEMPT_TIMEOUT_DEFAULT == 20.0
    assert V.DEVICE_BACKOFF_DEFAULT == 30.0


def test_candidates_can_start_at_an_explicit_index():
    probes = {0: result(0, "RESPONSIVE"), 1: result(1, "RESPONSIVE")}
    attempts, selected, failure, sleeps = _run(probes, candidates=[1], visible=[0, 1])
    assert selected["logical_index"] == 1 and selected["physical_device_id"] == 1
    assert len(attempts) == 1 and failure is None


def test_probe_card_override_classifies_each_shape(monkeypatch=None):
    """The real probe_card seam obeys the same classification (no hardware)."""
    old = {key: os.environ.get(key) for key in ("TFNU_ALLOW_TEST_SEAMS", "TFNU_PROBE_OVERRIDE")}
    try:
        os.environ["TFNU_ALLOW_TEST_SEAMS"] = "1"
        expected = {
            "fail": "DEVICE_PROBE_FAILED",
            "empty": "DEVICE_PROBE_EMPTY_OUTPUT",
            "unattributed": "DEVICE_PROBE_UNATTRIBUTED",
            "unresponsive": V.BUSY_CODE,
            "responsive": "RESPONSIVE",
        }
        for forced, classification in expected.items():
            os.environ["TFNU_PROBE_OVERRIDE"] = "0=" + forced
            got = V.probe_card(0, 0.01)
            assert got["classification"] == classification, (forced, got)
            assert got["simulated"] is True
        # unsupported wchan never supports driver-wait attribution
        assert not V.wchan_supports_driver_wait(None)
        assert not V.wchan_supports_driver_wait("do_wait")
        assert V.wchan_supports_driver_wait("devdrv_sync_non_tran")
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
