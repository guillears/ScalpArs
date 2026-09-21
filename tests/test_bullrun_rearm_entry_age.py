"""🕐 Sep-21 (57l) REARM entry-age gate — a bounce is worth trading for its first hour, not its sixth.

Evidence (log-derived door episodes #40/#42/#44): door-age <60min = 14·93%·+$952, ≥60min =
20·35%·−$1,108; within episode #44 alone the running P&L peaked +$562 at 31 min and closed −$358.

Invariants pinned here:
  · GREEN is NEVER gated — it carries the sleeve's P&L and the decay evidence is REARM-only.
  · the gate FAILS OPEN on a missing/unknown clock (an unknown age must not silently halt the
    sleeve; a fail-closed bug here would look identical to "no setups" in the trade data).
  · 0 / absent threshold disables it.
  · age is EPISODE age derived from `rearm_t0`, which the monitor carries forward across a flap —
    so a door that flaps cannot hand itself a fresh entry window.
"""
from types import SimpleNamespace

import pytest

from services.trading_engine import bullrun_door_age_min, bullrun_entry_age_ok


def _th(cap=60.0, flap=30.0):
    return SimpleNamespace(bullrun_rearm_max_entry_age_min=cap, bullrun_rearm_flap_merge_min=flap)


# ---------------------------------------------------------------- age helper

def test_age_is_minutes_since_the_door_armed():
    assert bullrun_door_age_min(1_000_000 + 3600, 1_000_000) == pytest.approx(60.0)
    assert bullrun_door_age_min(1_000_000 + 90, 1_000_000) == pytest.approx(1.5)


def test_age_is_none_without_a_clock():
    """No rearm_t0 = not a REARM episode (GREEN, or the monitor has not armed)."""
    for t0 in (None, 0, ""):
        assert bullrun_door_age_min(1_000_000, t0) is None


def test_age_never_negative_and_survives_garbage():
    assert bullrun_door_age_min(1_000_000, 2_000_000) == 0.0          # clock skew → clamp, not negative
    assert bullrun_door_age_min("nonsense", 1_000_000) is None


# ---------------------------------------------------------------- the gate

def test_green_is_never_gated():
    """The door carrying the P&L must be untouchable — B3 is 10·90%·+$2,426 and must stay whole."""
    for door in ("GREEN", "green", None, "", "AMBER"):
        assert bullrun_entry_age_ok(_th(), door, 10_000.0) is True


def test_rearm_allowed_inside_the_window_blocked_outside():
    assert bullrun_entry_age_ok(_th(), "REARM", 0.0) is True
    assert bullrun_entry_age_ok(_th(), "REARM", 59.9) is True
    assert bullrun_entry_age_ok(_th(), "REARM", 60.0) is True          # boundary is inclusive
    assert bullrun_entry_age_ok(_th(), "REARM", 60.1) is False


def test_the_b11_cohort_is_blocked():
    """Every B11 fill sat at door-age 346-398 min of episode #44 — the batch that prompted this."""
    for age in (346.0, 347.0, 360.0, 369.0, 375.0, 388.0, 398.0):
        assert bullrun_entry_age_ok(_th(), "REARM", age) is False


def test_episode_44_early_winners_are_kept():
    """0/13/13/31 min — 4·100%·+$562, the window the gate exists to preserve."""
    for age in (0.0, 13.0, 31.0):
        assert bullrun_entry_age_ok(_th(), "REARM", age) is True


def test_case_insensitive_door():
    assert bullrun_entry_age_ok(_th(), "rearm", 999.0) is False


def test_zero_or_missing_threshold_disables_the_gate():
    assert bullrun_entry_age_ok(_th(cap=0.0), "REARM", 10_000.0) is True
    assert bullrun_entry_age_ok(SimpleNamespace(), "REARM", 10_000.0) is True


def test_fails_open_on_unknown_age():
    """A missing clock must ALLOW the entry — a silent halt is indistinguishable from 'no setups'."""
    assert bullrun_entry_age_ok(_th(), "REARM", None) is True


def test_gate_composes_with_the_helper_end_to_end():
    t0 = 1_000_000
    th = _th()
    assert bullrun_entry_age_ok(th, "REARM", bullrun_door_age_min(t0 + 1800, t0)) is True    # 30 min
    assert bullrun_entry_age_ok(th, "REARM", bullrun_door_age_min(t0 + 7200, t0)) is False   # 2 h
    # GREEN has no clock at all → age None → allowed, twice over
    assert bullrun_entry_age_ok(th, "GREEN", bullrun_door_age_min(t0 + 7200, None)) is True


# ---------------------------------------------------------------- anti-flap clock
# Deep review 2026-09-21: the carry-forward decision is the highest-risk part of 57l and had ZERO
# coverage — the helper below was extracted from the monitor specifically so it can be pinned here.

from services.trading_engine import rearm_clock_for_arm  # noqa: E402

T0 = 1_000_000.0


def test_fresh_arm_with_no_history_uses_now():
    assert rearm_clock_for_arm(T0, None, None, 30.0) == T0


def test_quick_rearm_carries_the_old_clock_forward():
    """Door ran from T0, dropped at T0+2h, re-arms 5 min later → still the SAME episode."""
    off = T0 + 7200
    assert rearm_clock_for_arm(off + 300, off, T0, 30.0) == T0


def test_rearm_after_a_real_gap_starts_a_fresh_clock():
    off = T0 + 7200
    assert rearm_clock_for_arm(off + 1801, off, T0, 30.0) == off + 1801     # 30min + 1s


def test_carry_boundary_is_inclusive():
    off = T0 + 7200
    assert rearm_clock_for_arm(off + 1800, off, T0, 30.0) == T0             # exactly 30 min
    assert rearm_clock_for_arm(off + 1800.1, off, T0, 30.0) != T0


def test_green_interlude_cannot_poison_the_new_door():
    """The sleeve-disabling path the review found: REARM → GREEN → REARM inside the flap window.

    The monitor clears rearm_t0 AND rearm_off_at whenever GREEN is on, so the post-GREEN arm sees
    no history and must start fresh — otherwise the new door inherits the pre-GREEN clock and is
    blocked from its first second.
    """
    now = T0 + 19_500                                       # ~5h25m later, 25 min after the drop
    assert rearm_clock_for_arm(now, None, None, 30.0) == now


def test_zero_or_missing_flap_window_never_carries():
    off = T0 + 7200
    for flap in (0.0, None, ""):
        assert rearm_clock_for_arm(off + 60, off, T0, flap) == off + 60


def test_garbage_inputs_fall_back_to_a_fresh_clock():
    off = T0 + 7200
    assert rearm_clock_for_arm(off + 60, "nonsense", T0, 30.0) == off + 60
    assert rearm_clock_for_arm(off + 60, off, "nonsense", 30.0) == off + 60
    assert rearm_clock_for_arm(off + 60, off, T0, "nonsense") == off + 60


def test_negative_gap_does_not_carry():
    """Clock skew — an off_at in the FUTURE must not be read as a flap."""
    off = T0 + 7200
    assert rearm_clock_for_arm(off - 60, off, T0, 30.0) == off - 60


def test_carried_clock_still_blocks_a_stale_door_end_to_end():
    """A flapping door cannot buy itself a fresh 60-minute window — 57l's whole point."""
    off = T0 + 21_600                                        # door 6h old when it dropped
    t0 = rearm_clock_for_arm(off + 120, off, T0, 30.0)        # re-arms 2 min later
    assert t0 == T0
    age = bullrun_door_age_min(off + 180, t0)
    assert age > 360
    assert bullrun_entry_age_ok(_th(), "REARM", age) is False
