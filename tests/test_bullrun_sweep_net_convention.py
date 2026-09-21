"""🎯 Sep-21 (57k) — the exit sweep must price in the SAME units the engine trades in.

The engine arms on `peak_pnl` and stops on `pnl_percentage`, and both are fee-NET. The sweep used
to compare a GROSS path against those thresholds, so a fill peaking between arm and arm+FEE gross
was simulated as ARMED when live it took the full stop. That is the SUI 2026-09-21 class: gross
peak 1.052%, net peak 0.988%, live result −1.20% (full stop) but the old sim scored it +0.11%.

These are pure-math invariants of `simulate`; no network, no pool.

⚠ WHICH ASSERTS ACTUALLY GUARD THE BUG (deep review 2026-09-21 — do not trust the others as
regression cover): reverting the net conversion fails `test_near_miss_gross_peak_does_not_arm`,
`test_clearing_arm_net_does_arm_and_locks`, `test_arm_threshold_is_exactly_the_engine_number`,
and the `peak` leg of `test_stop_and_trail_are_both_net_referenced`. The remaining asserts are
tautological w.r.t. this fix — the fee cancels in `peak − band×atr` and in the timeout return —
so they pin the CONVENTION (peak is reported net, peak never goes negative) rather than catch a
regression of it. Kept deliberately: the convention is what the next reader will get wrong.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
B = pytest.importorskip("bullrun_exit_sweep")

ATR = 0.99            # SUI's entry ATR%
SL = -1.2             # ATR-widened stop, floored


def _flat_then(gross_peak, gross_end):
    """One bar that reaches `gross_peak`, then bars walking down to `gross_end`."""
    hi = [gross_peak, gross_peak, gross_end]
    lo = [0.0, gross_end, gross_end]
    return hi, lo


def test_near_miss_gross_peak_does_not_arm():
    """The SUI class: gross peak clears 1.0 but NET peak does not — must take the full stop."""
    hi, lo = _flat_then(1.052, -2.0)
    pnl, peak = B.simulate(hi, lo, ATR, arm=1.0)
    assert peak == pytest.approx(1.052 - B.FEE, abs=1e-9)   # peak is reported NET
    assert peak < 1.0                                        # …and therefore never armed
    assert pnl == pytest.approx(SL, abs=1e-9)                # full stop, not the +0.2 lock


def test_clearing_arm_net_does_arm_and_locks():
    """Same path with enough room to clear the arm NET — the BE lock must hold the floor."""
    hi, lo = _flat_then(1.0 + B.FEE + 0.05, -2.0)
    pnl, peak = B.simulate(hi, lo, ATR, arm=1.0)
    assert peak >= 1.0
    assert pnl == pytest.approx(B.LOCK, abs=1e-9)            # +0.2 net, not the stop


def test_arm_threshold_is_exactly_the_engine_number():
    """Lowering the arm by one fee unit is what rescues the near-miss fill — 57k's whole claim."""
    hi, lo = _flat_then(1.052, -2.0)
    assert B.simulate(hi, lo, ATR, arm=1.0)[0] == pytest.approx(SL, abs=1e-9)
    assert B.simulate(hi, lo, ATR, arm=0.90)[0] == pytest.approx(B.LOCK, abs=1e-9)


def test_exit_pnl_is_net_of_fees():
    """A path that never stops returns the final mid MINUS the round-trip toll, exactly once."""
    hi, lo = [0.5, 0.5], [0.1, 0.1]
    pnl, _ = B.simulate(hi, lo, ATR, arm=1.0)
    assert pnl == pytest.approx((0.5 + 0.1) / 2 - B.FEE, abs=1e-9)


def test_peak_never_goes_negative():
    """The engine seeds peak_pnl at 0; a fill that is red from tick one must report peak 0."""
    hi, lo = [-0.5, -0.6], [-0.7, -0.8]
    _, peak = B.simulate(hi, lo, ATR, arm=1.0)
    assert peak == 0.0


def test_stop_and_trail_are_both_net_referenced():
    """Trail line = peak − N×ATR in NET space, and the BE lock floors it (door-independent here)."""
    gross_peak = 1.5 + B.FEE
    hi, lo = [gross_peak, gross_peak, -5.0], [0.0, 1.4, -5.0]
    pnl, peak = B.simulate(hi, lo, ATR, arm=1.0, trail=0.5)
    assert peak == pytest.approx(1.5, abs=1e-9)
    assert pnl == pytest.approx(max(B.LOCK, 1.5 - 0.5 * ATR), abs=1e-9)
