"""ENV-05 device-failover policy: bounded, per-card, driver-only.

User policy: a card whose DRIVER does not answer may be skipped after a bounded
backoff, because a wedged queue can be attached to ONE card while another answers.
A card that is merely OCCUPIED by another tenant is different: that is normal
shared use, is recorded, and must never cause a retry or a switch.

These tests drive the REAL policy function with a fake probe, so neither scenario
needs hardware and each is deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
SCRIPTS = PACK / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_environment as V  # noqa: E402


def result(physical_id, *, responsive, simulated=False, occupancy=None):
    return {
        "physical_id": physical_id, "simulated": simulated,
        "probe": "fake", "responsive": responsive,
        "timed_out": not responsive, "unreaped": not responsive,
        "rc": 0 if responsive else None, "pid": -1,
        "wchan": None if responsive else "devdrv_sync_non_tran",
        "seconds": 0.01, "occupancy": occupancy,
    }


def test_first_card_wedged_switches_to_the_second_after_one_backoff():
    probes = {0: result(0, responsive=False, simulated=True),
              1: result(1, responsive=True)}
    sleeps = []
    attempts, selected = V.first_responsive_card(
        [0, 1], [0, 1], lambda physical: probes[physical], 30, sleeper=sleeps.append)

    assert [a["physical_id"] for a in attempts] == [0, 1]
    assert attempts[0]["outcome"] == V.BUSY_CODE
    assert attempts[0]["timed_out"] and attempts[0]["wchan"] == "devdrv_sync_non_tran"
    assert attempts[0]["backoff_seconds"] == 30
    assert sleeps == [30], "exactly one bounded backoff before the second card"
    assert selected == {"logical_index": 1, "physical_device_id": 1,
                        "selection_reason": "first RESPONSIVE entry of the ordered visible list"}


def test_all_cards_wedged_is_a_declared_failure():
    probes = {0: result(0, responsive=False, simulated=True),
              1: result(1, responsive=False, simulated=True)}
    sleeps = []
    attempts, selected = V.first_responsive_card(
        [0, 1], [0, 1], lambda physical: probes[physical], 30, sleeper=sleeps.append)

    assert selected is None, "no card answered, so nothing may be selected"
    assert len(attempts) == 2
    assert all(a["outcome"] == V.BUSY_CODE for a in attempts)
    assert sleeps == [30], "one backoff between the two attempts, never after the last"


def test_an_occupied_but_responsive_card_is_used_without_any_retry():
    probes = {0: result(0, responsive=True, occupancy="shared(4)"),
              1: result(1, responsive=True)}
    sleeps = []
    attempts, selected = V.first_responsive_card(
        [0, 1], [0, 1], lambda physical: probes[physical], 30, sleeper=sleeps.append)

    assert len(attempts) == 1, "occupancy must not trigger a second attempt"
    assert sleeps == [], "occupancy must not trigger a backoff"
    assert selected["physical_device_id"] == 0
    assert attempts[0]["occupancy"] == "shared(4)"


def test_the_policy_defaults_are_the_documented_ones():
    # 20 s separates a <=6 s healthy probe from a >20 s wedged one; 30 s is one
    # congestion window. Both are documented in README and the module comment.
    assert V.DEVICE_ATTEMPT_TIMEOUT_DEFAULT == 20.0
    assert V.DEVICE_BACKOFF_DEFAULT == 30.0


def test_candidates_start_at_an_explicit_index():
    probes = {0: result(0, responsive=True), 1: result(1, responsive=True)}
    attempts, selected = V.first_responsive_card(
        [1], [0, 1], lambda physical: probes[physical], 30, sleeper=lambda s: None)
    assert selected["logical_index"] == 1 and selected["physical_device_id"] == 1
    assert len(attempts) == 1
