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
