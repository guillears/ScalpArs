"""🪤 Sep-29 FADE LAGGARD gate (DECISION_LOG 128) — pure-rule truth table, live −DI == research feature, fail-open, wiring parity."""
import json, os, sys
import numpy as np, pandas as pd
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from services.indicators import closed_wilder_ndi, fade_laggard_block, closed_ema_gap_pct

TH = SimpleNamespace(spike_fade_lag_ndi_min=16.1, spike_fade_lag_btc_gap_min=0.0)


def test_truth_table_strict_and_both_legs():
    assert fade_laggard_block(TH, 16.2, 0.01) is True
    assert fade_laggard_block(TH, 16.1, 0.01) is False          # strict >
    assert fade_laggard_block(TH, 16.2, 0.0) is False           # strict > on the BTC leg
    assert fade_laggard_block(TH, 30.0, -0.5) is False          # BTC not trending up
    assert fade_laggard_block(TH, 5.0, 2.0) is False            # pair not in a daily downtrend
    assert fade_laggard_block(SimpleNamespace(spike_fade_lag_ndi_min=16.1, spike_fade_lag_btc_gap_min=1.0), 20.0, 0.5) is False


def test_fail_open_and_off():
    assert fade_laggard_block(TH, None, 1.0) is False and fade_laggard_block(TH, 20.0, None) is False
    assert fade_laggard_block(TH, float("nan"), 1.0) is False and fade_laggard_block(TH, "x", 1.0) is False
    assert fade_laggard_block(SimpleNamespace(spike_fade_lag_ndi_min=0.0), 20.0, 1.0) is False
    assert fade_laggard_block(SimpleNamespace(), 20.0, 1.0) is False


def _synthetic_ohlcv(n=260, seed=3):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    h = c * (1 + rng.uniform(0, 0.02, n)); l = c * (1 - rng.uniform(0, 0.02, n)); o = np.r_[c[0], c[:-1]]
    return [[i * 86400000, float(o[i]), float(h[i]), float(l[i]), float(c[i]), 1.0] for i in range(n)]


def test_live_ndi_equals_the_research_feature_on_closed_bars():
    """The engine reading must equal entry_feature_factory's PAIR_1d_ndi (the column the rule was found on), forming bar dropped."""
    import entry_feature_factory as EF
    raw = _synthetic_ohlcv()
    k = pd.DataFrame({"o": [r[1] for r in raw], "h": [r[2] for r in raw], "l": [r[3] for r in raw], "c": [r[4] for r in raw], "vol": 1.0})
    ref = float(EF._ind(k.iloc[:-1]).ndi.iloc[-1])              # factory on the closed bars only
    live = closed_wilder_ndi(raw)
    assert live is not None and abs(live - ref) < 1e-3
    assert closed_wilder_ndi(raw[:30]) is None and closed_wilder_ndi(None) is None and closed_wilder_ndi([[0, 1, 1, 1, 1, 1]] * 60) is None
    # the BTC leg reuses the zone-stamp helper (closed bars, ewm adjust=False) — pin its closed-bar semantics too
    assert closed_ema_gap_pct(raw, 50, 200) is not None and closed_ema_gap_pct(raw[:100], 50, 200) is None


def test_live_json_armed_and_wiring_parity():
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["spike_fade_lag_ndi_min"] == 16.1 and cfg["spike_fade_lag_btc_gap_min"] == 0.0     # frozen thresholds
    import config as C
    assert C.SignalThresholds.model_fields["spike_fade_lag_ndi_min"].default == 0.0            # code default = off
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert eng.count('_record_filter_block("SPIKE_FADE_LAGGARD", "SHORT")') == 2                # scanner + top-50 hook
    assert eng.count("fade_laggard_block(") == 2 and "get_ohlcv('BTC/USDT:USDT', '4h', 1000)" in eng
    assert eng.count("entry_pair_1d_ndi=") >= 3 and eng.count("await _fade_pair_1d_ndi(") == 2                                                 # 2 call sites + Order(...)
    assert "entry_pair_1d_ndi = Column(Float" in open(os.path.join(ROOT, "models.py")).read()
    assert "'entry_pair_1d_ndi', 'entry_btc_4h_ema50_200_gap_pct'" in open(os.path.join(ROOT, "database.py")).read()
    assert open(os.path.join(ROOT, "main.py")).read().count('"entry_pair_1d_ndi": getattr(') == 2
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert ui.count("config-spike-fade-lag-ndi-min") == 3 and ui.count("config-spike-fade-lag-btc-gap-min") == 3
    assert "Fade laggard gate (Sep 29)" in ui
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    assert "FADE_LAGGARD" in bld and "spike_fade_lag_ndi_min=16.1" in bld and bld.startswith("#!") and 'STACK_VERSION = "2026-10-01a"' in bld


def test_builder_inputs_prefer_stamps_and_fail_open():
    from scripts.build_master_pool import fade_laggard_inputs
    df = pd.DataFrame({"entry_strategy": ["SPIKE_FADE", "MOMENTUM"], "pair": ["ZZZNOPAIRUSDT", "ZZZNOPAIRUSDT"],
                       "opened_at": ["2026-09-01 00:00:00"] * 2, "entry_pair_1d_ndi": [20.0, np.nan], "entry_btc_4h_ema50_200_gap_pct": [0.5, np.nan]})
    ndi, gap = fade_laggard_inputs(df)
    assert ndi.iloc[0] == 20.0 and gap.iloc[0] == 0.5 and np.isnan(ndi.iloc[1])
    assert fade_laggard_block(TH, ndi.iloc[1], gap.iloc[1]) is False
