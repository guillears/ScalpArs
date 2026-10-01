"""🐳 Master-pool CF_FADE_CAP05 ticket scale (DECISION_LOG 165/167) — pure-math invariants."""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.build_master_pool import fade_cap05_scale as f  # noqa: E402


def test_desired_bound_matches_the_master_rows():
    assert math.isclose(f(25340.851953, 17219890.0, 17219.892567, True), 25340.851953 / 17219.892567, rel_tol=1e-9)   # WLFI 08-05
    assert math.isclose(93.31 * f(25340.851953, 17219890.0, 17219.892567, "True"), 137.32, abs_tol=0.01)


def test_half_percent_and_ceiling_bounds():
    assert math.isclose(f(200_000, 12_000_000, 12_000, True), 0.005 * 12_000_000 / 12_000)       # 0.5 % of volume binds
    assert math.isclose(f(9e6, 150_000_000, 150_000, True), 500_000 / 150_000)                    # hard ceiling binds


def test_no_ops():
    assert f(25_000, 9_999_999, 9_999, True) == 1.0            # thin pair: lived ticket stays
    assert f(25_000, 15_000_000, 15_000, False) == 1.0         # not capped
    assert f(25_000, 15_000_000, 15_000, None) == 1.0
    assert f(None, 15_000_000, 15_000, True) == 1.0 and f(float("nan"), 15_000_000, 15_000, True) == 1.0
    assert f(25_000, 15_000_000, 0, True) == 1.0               # zero notional
    assert f(15_010, 15_000_000, 15_000, True) == 1.0          # inside the 0.1 % tolerance
    assert f(10_000, 15_000_000, 15_000, True) == 1.0          # desired below the lived ticket: never shrinks


def test_loser_scales_symmetrically():
    sc = f(24_000, 12_000_000, 12_000, True)
    assert sc == 2.0 and -100.0 * sc == -200.0
