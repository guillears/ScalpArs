"""Best-case CLEAN-ONLY column (DECISION_LOG 453): a line switches on only when its CLEAN fills pass the DECISION_LOG 447 ladder —
past its checkpoint (N fills on D days) with the day-clustered 95 % range on the line's side; before it the in-sample number never counts."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "best_case"))
import confidence as CF  # noqa: E402

DAYS = [f"2026-11-{d:02d}" for d in range(1, 11)]


def _fills(vals, n_days=10):
    days = [DAYS[i % n_days] for i in range(len(vals))]
    return np.asarray(vals, float), days


def test_pre_before_checkpoint():
    v, d = _fills([-2.0] * 14)
    assert CF.ladder(v, d, -1, 15, 8) == "PRE"                    # 14 < 15 fills, however strong
    v, d = _fills([-2.0] * 20, n_days=5)
    assert CF.ladder(v, d, -1, 15, 8) == "PRE"                    # 5 < 8 days


def test_pass_against_straddle():
    rng = np.random.default_rng(1)
    v, d = _fills(-1.0 + rng.normal(0, 0.2, 20))
    assert CF.ladder(v, d, -1, 15, 8) == "PASS"                   # blocked fills clearly lose
    assert CF.ladder(v, d, +1, 15, 8) == "AGAINST"                # a gain line clearly losing
    v, d = _fills(rng.normal(0, 2.0, 20))
    assert CF.ladder(v, d, -1, 15, 8) == "STRADDLE"


def test_nan_fills_dropped_with_their_days():
    v, d = _fills([-1.0] * 15 + [np.nan] * 5)
    assert CF.ladder(v, d, -1, 15, 8) == "PASS"
    v, d = _fills([-1.0] * 14 + [np.nan])
    assert CF.ladder(v, d, -1, 15, 8) == "PRE"                    # a NaN never counts toward the checkpoint


def test_unregistered_line_is_pre():
    assert CF.line_calls([("X", "NOT_A_LINE")], Z=pd.DataFrame(), fb=None, fx=None) == {("X", "NOT_A_LINE"): "PRE"}


def test_deciding_test_pace_and_dates():
    """(DECISION_LOG 484) Market-hot lines are decided on blocked < kept; pace = the whole batch; a line is never decided before its checkpoint."""
    assert CF.YARD["ML_MKT_HOT"] == "rest" and CF.YARD["MKT_HOT"] == "rest" and CF.YARD.get("WILLY_GAP", "money") == "money"
    end = CF.B25_START + pd.Timedelta(days=10)
    p = CF.pace([CF.B25_START + pd.Timedelta(days=i) for i in range(10)], end)
    assert p == (1.0, 1.0)
    assert CF.eta_days(5, 0, p) == 5 and CF.eta_days(5, 8, (1.0, 0.5)) == 16 and CF.eta_days(0, 0, p) == 0
    assert CF.eta_days(3, 0, (0.0, 0.0)) == float("inf")
    assert CF.when(0, end) == "now" and CF.when(float("inf"), end) == "no pace yet" and CF.when(120, end) == "> 3 months"
    assert CF.when(3, end) == f"{end + pd.Timedelta(days=3):%b %d}"
    r = CF.row("MOMENTUM_LONG", "ML_MKT_HOT", np.r_[np.full(40, -0.03), np.full(40, 0.1)] - 0.05, [0.1, 0.2], ["2026-11-01", "2026-11-02"], -1, None,
               rest=np.full(80, 0.3) + np.r_[np.full(40, -0.4), np.full(40, 0.4)], p=(1.0, 0.5), end=end, clean_rest=2, p_rest=(1.0, 0.5))
    assert "**blocked < kept (90 %):" in r and "more kept)" in r
    import re
    b0 = re.search(r'B0, FLOWS = pd.Timestamp\("([^"]+)"\)', open(os.path.join(os.path.dirname(CF.__file__), "best_dcr.py")).read())
    assert b0 and pd.Timestamp(b0[1]) == CF.B25_START                    # one batch start in both files (deep review 484)
