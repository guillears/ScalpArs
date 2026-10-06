"""DECISION_LOG 234 (2026-10-06, STACK 2026-10-06d): the master prices FRENZY sleeve fills with the LIVE lock exit (peak ≥ +3 → max(+2,
peak − 2)) instead of the retired fixed +3 TP, and replays the WIDE hold-green rule (DECISION_LOG 231) with the engine's own function.
Data-integrity test on the committed reports/MASTER_POOL_stacked.csv (regenerate it after any builder change)."""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _m():
    return pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)


def test_frenzy_lock_pricing():
    import json
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    arm, floor, trail = th["frenzy_lock_arm_pct"], th["frenzy_lock_floor_pct"], th["frenzy_lock_trail_pct"]
    M = _m()
    f = M[(M.stack_keep == True) & M.entry_strategy.astype(str).str.startswith("FRENZY")]
    assert f.peak_pnl.notna().all()
    pre = f.opened_at.astype(str).str[:19].str.replace(" ", "T") < "2026-10-05T15:49"
    armed = f[pre & (f.peak_pnl.astype(float) >= arm)]
    assert len(armed) > 0 and np.allclose(armed.stack_pct, np.maximum(min(floor, arm), armed.peak_pnl.astype(float) - trail))
    rest = f[~(pre & (f.peak_pnl.astype(float) >= arm))]
    assert np.allclose(rest.stack_pct, rest.pnl_percentage)


def test_wide_hold_green_replayed():
    M = _m()
    w = M[M.entry_strategy.astype(str) == "FRENZY_WIDE"]
    blocked = w[w.stack_block_reason.isin(["FRENZY_WIDE_ATR_HIGH", "FRENZY_WIDE_RECLAIM"])]
    assert len(blocked) >= 3 and (blocked.stack_keep == False).all() and (blocked.stack_pnl == 0).all() and blocked.stack_pct.isna().all()
    kept = w[w.stack_keep == True]
    reb = pd.read_csv(os.path.join(ROOT, "reports", "WIDE_STREAK_REBUILD.csv"))
    st = {(a, p): s_ for a, p, s_ in zip(reb.opened_at.astype(str).str[:19], reb.pair, reb.above_streak)}
    assert (pd.to_numeric(kept.entry_atr_pct) <= 2.5).all() and kept.entry_frenzy_bar_ret_pct.notna().all()
    assert all(st[(str(a)[:19], p)] > 12 for a, p in zip(kept.opened_at, kept.pair))
    assert os.path.exists(os.path.join(ROOT, "reports", "WIDE_STREAK_REBUILD.csv"))


def test_b17_archived():
    M = _m()
    assert (M.era == "B17").sum() == 25 and os.path.exists(os.path.join(ROOT, "reports", "BASELINE17_batch1003-1006_orders.csv"))
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    assert 'STACK_VERSION = "2026-10-06d"' in bld and "max(min(_lk_floor, _lk_arm), float(r.peak_pnl) - _lk_trail)" in bld and "FRENZY_LOCK_LIVE_FROM" in bld
