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


def test_lite_c_rule():
    tp = B.frenzy_lite_c_tp_frozen
    assert tp(20.0, 0.01) == 3.0 and tp(82.7, 0.0865) == 3.0                     # zone (B21 WAL)
    assert tp(19.9, 0.5) == 2.0 and tp(50.0, 0.0) == 2.0 and tp(71.1, -0.23) == 2.0   # vol < 20 / slope not > 0 (B18 MET)
    assert tp(None, 0.3) == 2.0 and tp(float("nan"), 0.3) == 2.0 and tp(30.0, None) == 2.0   # missing stamp → the lower TP
    fp = B.frenzy_fixed_pct
    assert fp("FRENZY_LITE", 2.59, -3.43, tp(16.4, -0.2)) == 2.0                  # EPIC 10-15: saved
    assert fp("FRENZY_LITE", 3.02, 2.80, tp(7.7, -0.27)) == 2.0                   # DEEP 10-14: capped
    assert fp("FRENZY_LITE", 1.23, -2.54, tp(50.8, -0.03)) == -2.54               # DUSK 10-15: peak < 2 → as traded


def test_master_lite_rows_priced_at_c_tp():
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    k = M[(M.stack_keep == True) & (M.entry_strategy.astype(str) == "FRENZY_LITE")]
    assert len(k) >= 24
    t = np.array([B.frenzy_lite_c_tp_frozen(v, s) for v, s in zip(k.entry_frenzy_vol_mult, k.entry_btc_1h_slope)])
    pk = pd.to_numeric(k.peak_pnl, errors="coerce").values
    exp = np.where(pk >= t, t, k.pnl_percentage.values)
    assert np.allclose(k.stack_pct.values, exp)
    b22 = k[k.era == "B22"].set_index("pair")
    assert np.isclose(b22.loc["EPICUSDT", "stack_pct"], 2.0) and np.isclose(b22.loc["DEEPUSDT", "stack_pct"], 2.0)


def test_b22_archived_live_batch():
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    b = M[M.era == "B22"]
    assert len(b) == 22 and (b.is_paper.astype(str).str.lower() == "false").all() and b.opened_at.min() >= "2026-10-14T18:36:30"
    assert "BASELINE22_batch1014-1016_LIVE_orders.csv" in dict((e, p) for e, p, _, _ in B.discover_eras())["B22"]
