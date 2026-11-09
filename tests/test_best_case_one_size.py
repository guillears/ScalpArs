"""📏 one-size current-batch DCR (DECISION_LOG 503) — the money-math invariants of scripts/best_case/one_size.py."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "best_case"))
import one_size as OS  # noqa: E402


def test_frac_is_pct_times_size_and_nan_adds_nothing():
    f = OS.frac([3.0, -4.0, np.nan], [0.5, 0.5, 2.0])
    assert np.allclose(f, [0.015, -0.02, 0.0])


def test_dcr_compounds_every_calendar_day_by_close_day():
    closed = ["2026-11-01T10:00:00", "2026-11-01T20:00:00", "2026-11-03T05:00:00+00:00"]
    f = [0.01, 0.01, -0.01]
    d = OS.dcr(closed, f, [True, True, True], pd.Timestamp("2026-11-01 08:00"), pd.Timestamp("2026-11-03 06:00"))
    want = ((1.02 * 1.0 * 0.99) ** (1 / 3) - 1) * 100   # 3 calendar days, the empty one at 0 %
    assert abs(d - want) < 1e-9


def test_dcr_mask_and_tz_aware_span():
    closed = ["2026-11-01T10:00:00", "2026-11-01T11:00:00"]
    d = OS.dcr(closed, [0.05, -0.50], [True, False], pd.Timestamp("2026-11-01", tz="UTC"), pd.Timestamp("2026-11-01 23:00", tz="UTC"))
    assert abs(d - 5.0) < 1e-9   # the masked fill never counts


def test_one_size_does_not_depend_on_account_balance():
    """the WILLY sign flip (DL 503): the same % results give the same DCR whatever the account was when they closed."""
    closed = ["2026-10-24T03:00:00", "2026-10-31T22:00:00"]
    a = OS.dcr(closed, OS.frac([3.69, -4.0], [0.47, 0.47]), [True, True], pd.Timestamp("2026-10-24"), pd.Timestamp("2026-10-31 23:00"))
    b = OS.dcr(closed[::-1], OS.frac([-4.0, 3.69], [0.47, 0.47]), [True, True], pd.Timestamp("2026-10-24"), pd.Timestamp("2026-10-31 23:00"))
    assert abs(a - b) < 1e-9 and a < 0   # a +3.69 and a −4 at one size net negative, in either order


def test_equity_at_open_and_sleeve_x_on_the_account_path():
    # account path: start 2,900, a +100 close before the 2nd open, a 7,100 top-up before the 3rd
    eq = OS.equity_at_open(["2026-10-24T00:00:00", "2026-10-25T00:00:00", "2026-10-27T00:00:00"],
                           ["2026-10-24T12:00:00"], [100.0], 2900.0, [("2026-10-26T13:50:00", 7100.0)])
    assert np.allclose(eq, [2900.0, 3000.0, 10100.0])
    assert abs(OS.sleeve_x([2900.0, 6000.0, 20200.0], eq, "FRENZY_LITE") - 2.0) < 1e-12        # 1.0 · 2.0 · 2.0 → median 2.0
    # WILLY: a 10× fill counts at today's 5× (× 0.5) and today's margin (× willy_inv ÷ the cell multiplier it traded at)
    assert abs(OS.sleeve_x([10000.0], [10000.0], "FRENZY_WILLY", [10.0], [1.0], willy_inv=0.5) - 0.25) < 1e-12
    assert abs(OS.sleeve_x([10000.0], [10000.0], "FRENZY_WILLY_HT", [10.0]) - 0.5) < 1e-12   # HT: 10× → today's 5×
    x = OS.sleeve_x([np.nan], [10000.0], "FRENZY_LITE")
    assert x != x   # NaN


def test_dcr_refuses_a_fill_without_close_time():
    import pytest
    with pytest.raises(ValueError, match="without a closed_at"):
        OS.dcr([None], [0.01], [True], pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-02"))


def test_cap_factor_makes_one_stop_cost_the_cap():
    assert abs(OS.cap_factor(0.98, 3.0) - 5 / 2.94) < 1e-12           # RETRY ≈ ×1.70
    assert abs(OS.cap_factor(1.94, 3.0) * 1.94 * 3.0 - 5.0) < 1e-12    # LITE: after the factor one stop = 5 %
    assert OS.cap_factor(1.94, 3.0) < 1 < OS.cap_factor(0.244, 10.0)   # LITE down · Flatbase up
    assert OS.cap_factor(0.0, 3.0) != OS.cap_factor(0.0, 3.0) and OS.cap_factor(None, 3.0) != OS.cap_factor(None, 3.0)   # NaN


def test_equity_at_open_unsorted_and_tz_aware_closes():
    eq = OS.equity_at_open(["2026-11-02T00:00:00"], ["2026-11-01T10:00:00+00:00", "2026-10-30T10:00:00", "2026-11-03T00:00:00"],
                           [50.0, -20.0, 999.0], 1000.0)
    assert np.allclose(eq, [1030.0])   # only closes before the open count, in any order / tz


def test_sleeve_x_needs_leverage_for_willy():
    import pytest
    with pytest.raises(ValueError, match="leverage column"):
        OS.sleeve_x([1000.0], [1000.0], "FRENZY_WILLY_HT")
