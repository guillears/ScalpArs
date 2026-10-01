"""🐳 Master-pool CF_FADE_CAP05 ticket scale (DECISION_LOG 165/167/168) — pure-math invariants."""
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
    assert math.isclose(f(25_000, 4_000_000, 8_000, True), 0.005 * 4_000_000 / 8_000)   # thin pair at the old 0.2 %: re-priced too (168)
    assert f(25_000, 4_000_000, 20_000, True) == 1.0           # already at the 0.5 % ticket
    assert f(25_000, 15_000_000, 15_000, False) == 1.0         # not capped
    assert f(25_000, 15_000_000, 15_000, None) == 1.0
    assert f(None, 15_000_000, 15_000, True) == 1.0 and f(float("nan"), 15_000_000, 15_000, True) == 1.0
    assert f(25_000, 15_000_000, 0, True) == 1.0               # zero notional
    assert f(15_010, 15_000_000, 15_000, True) == 1.0          # inside the 0.1 % tolerance
    assert f(10_000, 15_000_000, 15_000, True) == 1.0          # desired below the lived ticket: never shrinks


def test_loser_scales_symmetrically():
    sc = f(24_000, 12_000_000, 12_000, True)
    assert sc == 2.0 and -100.0 * sc == -200.0


def test_master_csv_scale_invariants():
    """168: the scale lives only on kept, non-probe, capped fades; it never shrinks a ticket; tags stay in the allowed set."""
    import pandas as pd
    m = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    assert m.stack_ticket_scale.notna().all() and (m.stack_ticket_scale >= 1.0).all()
    s = m[m.stack_ticket_scale != 1.0]
    assert len(s) and s.stack_keep.all() and not s.is_probe.any() and (s.entry_strategy == "SPIKE_FADE").all()
    assert s.liquidity_capped.astype(str).str.lower().eq("true").all()
    assert m.stack_keep.dtype == bool and m.is_probe.dtype == bool
    assert set(s.stack_block_reason) <= {"CF_FADE_CAP05", "CF_FADE_SL15", "CF_FADE_LATE_ARM"}   # tripwire: a new CF tag on a scaled fade must be reviewed
    assert ((s.notional_value * s.stack_ticket_scale) <= 0.005 * s.entry_pair_volume_24h_usd * 1.001).all()
    u = s[s.stack_block_reason == "CF_FADE_CAP05"]                      # untagged before: lived P&L × scale, so the pct is the lived pct
    assert ((u.stack_pnl / u.stack_ticket_scale - u.pnl).abs() < 0.02).all()
