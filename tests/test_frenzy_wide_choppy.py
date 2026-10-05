"""🌀 Oct-5 (DECISION_LOG 215): FRENZY_WIDE skips a choppy / fading pump — the share of the episode's 5m closes at / above the spike-anchored
VWAP (frenzy_walk 'above_share') at or below frenzy_wide_above_share_min. Fail-open on a missing reading; 0 = off."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.frenzy import frenzy_wide_choppy  # noqa: E402

TH = SimpleNamespace(frenzy_wide_above_share_min=67.8)


def test_blocks_at_and_below_the_frozen_threshold():
    assert frenzy_wide_choppy({"above_share": 20.7}, TH)
    assert frenzy_wide_choppy({"above_share": 67.8}, TH)          # ≤ blocks (the study's cut)
    assert not frenzy_wide_choppy({"above_share": 67.81}, TH)
    assert not frenzy_wide_choppy({"above_share": 96.0}, TH)      # RLC 10-05 at its entry


def test_fail_open_and_off_switch():
    assert not frenzy_wide_choppy({"above_share": None}, TH)
    assert not frenzy_wide_choppy({}, TH)
    assert not frenzy_wide_choppy(None, TH)
    assert not frenzy_wide_choppy({"above_share": 10.0}, SimpleNamespace(frenzy_wide_above_share_min=0.0))
    assert not frenzy_wide_choppy({"above_share": 10.0}, SimpleNamespace())
    assert not frenzy_wide_choppy({"above_share": "x"}, TH)


def test_frenzy_walk_reports_the_share_of_closes_above_the_spike_vwap():
    from services.frenzy import frenzy_walk, BAR_MS
    th = SimpleNamespace(frenzy_spike_ret_pct=5.0, frenzy_spike_vol_mult=10.0, frenzy_spike_min_hour_usd=0.0, frenzy_state_vol_mult=1.0,
                         frenzy_min_hours=0.0, frenzy_max_hours=96.0)
    t0 = 1_790_000_000_000 // BAR_MS * BAR_MS
    bars = [[t0 + i * BAR_MS, 1.0, 1.0, 1.0, 1.0, 100.0] for i in range(400)]                 # quiet history (> the 25 h verify window)
    bars += [[t0 + (400 + i) * BAR_MS, 1.0, 1.2, 1.0, 1.2, 100000.0] for i in range(1)]      # the spike bar
    bars += [[t0 + (401 + i) * BAR_MS, 1.2, 1.3, 1.2, 1.3, 50000.0] for i in range(20)]      # closes above the spike VWAP
    ep = frenzy_walk(bars, 100.0 * 1.0, th)
    assert ep is not None and ep["above_share"] == 100.0
