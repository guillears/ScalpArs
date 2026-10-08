"""10-08a (DECISION_LOG 250): FRENZY sleeve fills priced at the FIXED +3 (lock retired), WIDE hold-green ATR cap 3.0, bearish-day block.
History — DECISION_LOG 234 (2026-10-06, STACK 2026-10-06d): the master prices FRENZY sleeve fills with the LIVE lock exit (peak ≥ +3 → max(+2,
peak − 2)) instead of the retired fixed +3 TP, and replays the WIDE hold-green rule (DECISION_LOG 231) with the engine's own function.
Data-integrity test on the committed reports/MASTER_POOL_stacked.csv (regenerate it after any builder change)."""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _m():
    return pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)


def _th():
    import json
    return json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]


def test_frenzy_fixed_tp_pricing():
    """10-08a (DECISION_LOG 250): the live exit is the fixed +3 again (lock off) — every kept FRENZY fill whose recorded peak reached +3
    books +3 (every era); below +3 the as-traded pct."""
    th = _th()
    tp = float(th["frenzy_tp_pct"])
    assert float(th["frenzy_lock_arm_pct"]) == 0 and tp == 3.0
    M = _m()
    f = M[(M.stack_keep == True) & M.entry_strategy.astype(str).str.startswith("FRENZY")]
    assert f.peak_pnl.notna().all()
    hit = f[f.peak_pnl.astype(float) >= tp]
    assert len(hit) >= 4 and np.allclose(hit.stack_pct, tp)                     # RLC / MOVR / AIN 10-04 / SAND 10-04 05:05 (stopped −3 live after a +3.26 peak)
    rest = f[f.peak_pnl.astype(float) < tp]
    assert len(rest) > 0 and np.allclose(rest.stack_pct, rest.pnl_percentage)    # −3 stops before +3 stay as traded
    assert (np.sign(f.stack_pnl) == np.sign(f.stack_pct)).all()


def test_frenzy_fixed_pct_rule():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from build_master_pool import frenzy_fixed_pct as fp
    assert fp("FRENZY_LONG", 3.0, -3.0) == 3.0 and fp("FRENZY_WIDE", 5.1, 3.47) == 3.0 and fp("FRENZY_LITE", 3.26, -3.0) == 3.0   # peak ≥ 3 books +3
    assert fp("FRENZY_LONG", 2.99, -3.0) == -3.0 and fp("FRENZY_LONG", None, -1.2) == -1.2 and fp("FRENZY_LONG", float("nan"), 0.4) == 0.4
    assert fp("MOMENTUM", 5.0, 1.0) == 1.0                                        # other sleeves never


def test_bearish_day_stack_block():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from build_master_pool import frenzy_bearish_stack_block as bb
    assert bb("FRENZY_LONG", -0.4, -0.02) and bb("FRENZY_WIDE", -0.9, -0.1) and bb("FRENZY_LITE", -0.1, -0.1)   # bearish → refused
    assert not bb("FRENZY_LONG", -0.4, 0.02) and not bb("FRENZY_LONG", 0.3, -0.2)                              # one leg ≥ 0 → kept
    assert not bb("FRENZY_LONG", None, -0.2) and not bb("FRENZY_LONG", float("nan"), float("nan"))             # unreadable → kept (fail-open)
    assert not bb("MOMENTUM", -0.4, -0.2)
    M = _m()
    f = M[M.entry_strategy.astype(str).str.startswith("FRENZY")]
    r1, g = pd.to_numeric(f.entry_btc_1d_ret_pct, errors="coerce"), pd.to_numeric(f.entry_btc_trend_gap_pct, errors="coerce")
    bear = f[(r1 < 0) & (g < 0)]
    assert len(bear) >= 2 and not (bear.stack_keep == True).any()               # FLUID 10-06 / AIN 10-03: refused (hold-green judged first → its label)
    kept = f[f.stack_keep == True]
    assert len(kept) >= 8 and not ((pd.to_numeric(kept.entry_btc_1d_ret_pct, errors="coerce") < 0)
                                   & (pd.to_numeric(kept.entry_btc_trend_gap_pct, errors="coerce") < 0)).any()   # every kept fill is non-bearish
    assert ((kept.pair == "ENJUSDT") & (pd.to_numeric(kept.entry_btc_1d_ret_pct) < 0)).any()                    # ENJ 10-03: day < 0 but gap > 0 → kept


def test_wide_hold_green_replayed():
    M = _m()
    w = M[M.entry_strategy.astype(str) == "FRENZY_WIDE"]
    blocked = w[w.stack_block_reason.isin(["FRENZY_WIDE_ATR_HIGH", "FRENZY_WIDE_RECLAIM"])]
    assert len(blocked) >= 3 and (blocked.stack_keep == False).all() and (blocked.stack_pnl == 0).all() and blocked.stack_pct.isna().all()
    kept = w[w.stack_keep == True]
    reb = pd.read_csv(os.path.join(ROOT, "reports", "WIDE_STREAK_REBUILD.csv"))
    st = {(a, p): s_ for a, p, s_ in zip(reb.opened_at.astype(str).str[:19], reb.pair, reb.above_streak)}
    assert (pd.to_numeric(kept.entry_atr_pct) <= float(_th()["frenzy_max_atr_pct"])).all() and kept.entry_frenzy_bar_ret_pct.notna().all()   # 3.0 since 10-08a
    assert all(st[(str(a)[:19], p)] > 12 for a, p in zip(kept.opened_at, kept.pair))
    assert os.path.exists(os.path.join(ROOT, "reports", "WIDE_STREAK_REBUILD.csv"))


def test_b17_archived():
    M = _m()
    assert (M.era == "B17").sum() == 25 and os.path.exists(os.path.join(ROOT, "reports", "BASELINE17_batch1003-1006_orders.csv"))
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    assert 'STACK_VERSION = "2026-10-08b"' in bld and "frenzy_fixed_pct(strat," in bld and 'reason[n] = "FRENZY_BEARISH_DAY"' in bld
    assert "WIDE_HG_MAX_ATR_FROZEN = 12.0, 3.0" in bld and "Prior 10-06d — 10-06d:" in bld   # the STACK history keeps every step


def test_wide_hold_green_cap_is_era_aware():
    """10-08a deep review: a PRE-raise WIDE fill with ATR in (2.5, 3.0] on a red candle must not be re-coded GREEN_BAR and kept as WIDE —
    the cap in force when it opened (2.5) codes it ATR_HIGH → refused (FRENZY_WIDE_ATR_HIGH); from the raise the 3.0 cap applies."""
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import build_master_pool as B
    from services.frenzy import frenzy_wide_hold_green_block
    th = SimpleNamespace(frenzy_wide_hold_green_streak=B.WIDE_HG_STREAK_FROZEN)
    pre, post = "2026-10-05T12:00:00", "2026-10-10 12:00:00"
    assert B.wide_hg_cap_at(pre) == 2.5 and B.wide_hg_cap_at(post) == 3.0 and B.wide_hg_cap_at(B.FRENZY_ATR30_FROM) == 3.0
    assert B.wide_hg_code(pre, 2.7) == "FRENZY_ATR_HIGH" and B.wide_hg_code(post, 2.7) == "FRENZY_GREEN_BAR"
    assert B.wide_hg_code(pre, 2.5) == "FRENZY_GREEN_BAR" and B.wide_hg_code(post, 3.01) == "FRENZY_ATR_HIGH" and B.wide_hg_code(post, None) == "FRENZY_ATR_HIGH"
    red_pre = frenzy_wide_hold_green_block({"above_streak": 20, "bar_ret_pct": -0.3}, B.wide_hg_code(pre, 2.7), th)
    assert red_pre == "FRENZY_WIDE_ATR_HIGH"                                     # the synthetic pre-raise red ATR-2.7 WIDE row is refused
    green_post = frenzy_wide_hold_green_block({"above_streak": 20, "bar_ret_pct": 0.3}, B.wide_hg_code(post, 2.7), th)
    assert green_post is None                                                    # a post-raise hold-green ATR-2.7 WIDE fill is kept
