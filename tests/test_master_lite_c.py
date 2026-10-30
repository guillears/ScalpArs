"""10-16a (DECISION_LOG 301 in the master STACK): FRENZY_LITE fills are priced at their LITE-C take profit — +3 inside the zone
(entry_frenzy_vol_mult ≥ 20 ∧ entry_btc_1h_slope > 0), else +2 (a missing stamp → +2, the engine's fail-to-lower) — through the engine's own
services.frenzy.frenzy_lite_c_tp on values FROZEN with STACK_VERSION. Pins the frozen values to trading_config.json (a retune must bump the
stack), the rule on synthetic rows, and the committed master's B21 / B22 rows (data-integrity test — regenerate the master after a builder change)."""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_master_pool as B   # noqa: E402


def test_frozen_lite_c_matches_live_json():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert B.FRENZY_LITE_C_ENABLED_FROZEN is bool(th["frenzy_lite_c_enabled"])
    assert B.FRENZY_LITE_C_VOL_FROZEN == float(th["frenzy_lite_c_vol_mult"])
    assert B.FRENZY_LITE_C_SLOPE_FROZEN == float(th["frenzy_lite_c_btc_slope_min"])
    assert B.FRENZY_LITE_C_TP_FROZEN == float(th["frenzy_lite_c_tp_pct"])
    assert B.FRENZY_LITE_TP_FROZEN == float(th["frenzy_lite_tp_pct"])


def _lite_cf():
    c = pd.read_csv(os.path.join(ROOT, "reports", "LITE_C_REVERT_CF.csv"))
    return {(a[:19], p): v for a, p, v in zip(c.opened_at.astype(str), c.pair, c.cf_pct)}


def test_lite_c_rule():
    tp = B.frenzy_lite_c_tp_frozen                                                # LITE C off since DL 420: +3 for every fill
    assert tp(20.0, 0.01) == 3.0 and tp(19.9, 0.5) == 3.0 and tp(50.0, 0.0) == 3.0 and tp(None, 0.3) == 3.0
    fp = B.frenzy_fixed_pct
    assert fp("FRENZY_LITE", 2.59, -3.43, tp(16.4, -0.2)) == -3.43                # EPIC 10-15: back to the stop
    assert fp("FRENZY_LITE", 3.02, 2.80, tp(7.7, -0.27)) == 3.0                   # DEEP 10-14: +3
    assert fp("FRENZY_LITE", 1.23, -2.54, tp(50.8, -0.03)) == -2.54               # DUSK 10-15: as traded


def test_master_lite_rows_priced_at_plus_3():
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    k = M[(M.stack_keep == True) & (M.entry_strategy.astype(str) == "FRENZY_LITE")]
    assert len(k) >= 24
    cf = _lite_cf()
    pk = pd.to_numeric(k.peak_pnl, errors="coerce").values
    exp = np.where(pk >= 3.0, 3.0, k.pnl_percentage.values)
    exp = np.array([cf.get((str(a)[:19], p), e) for a, p, e in zip(k.opened_at, k.pair, exp)])
    assert np.allclose(k.stack_pct.values, exp)
    b22 = k[k.era == "B22"].set_index("pair")
    assert np.isclose(b22.loc["EPICUSDT", "stack_pct"], -3.43, atol=0.01) and np.isclose(b22.loc["DEEPUSDT", "stack_pct"], 3.0)


def test_b22_archived_live_batch():
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    b = M[M.era == "B22"]
    assert len(b) == 22 and (b.is_paper.astype(str).str.lower() == "false").all() and b.opened_at.min() >= "2026-10-14T18:36:30"
    assert "BASELINE22_batch1014-1016_LIVE_orders.csv" in dict((e, p) for e, p, _, _ in B.discover_eras())["B22"]
