"""🧱 (DECISION_LOG 527) the ARMED WATCH-LINE FILTERS — services/watch_filters.py rules (frozen cuts, toggles, fail-open on unreadable inputs),
their wiring in the engine (judged last, recorded as blocked signals), the config / UI parity (D11), the decisions-export BLOCKED rows (D12)
and the master builder's use of the same rules. Hermetic: no network, no DB."""
import json
import math
import os
import re
from types import SimpleNamespace as NS

import pytest

from services import watch_filters as WF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ON = NS(**{k: True for k in WF.TOGGLES.values()})
OFF = NS(**{k: False for k in WF.TOGGLES.values()})


def _src(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def test_frozen_cuts():
    assert (WF.WILLY_GAP_ADX_MAX, WF.WILLY_GAP_EMA_GAP_MAX, WF.FAILED_SPIKE_VWAP_MAX, WF.RETRY_BTC1H_MAX, WF.LITE_EMA20_MAX) == (55.0, 0.90, 0.0, 0.0, 0.0)
    assert (WF.LONG_WEAK_VOL_MAX, WF.FADE_LATE_GAP_MIN, WF.FADE_POS_DI_MAX, WF.FADE_EXT_MAX, WF.ML_HOT_SHARE_MIN, WF.ML_BULL_MIN) == (119.0, 0.0, 48.5, 1.80, 0.50, 90.0)
    from services import mkt_up4h
    assert WF.ML_HOT_SHARE_MIN == mkt_up4h.HOT_CUT


def test_willy_and_ht():
    assert WF.willy_blocks(ON, 55.0, 0.91, 0.1) == []                              # boundaries: ADX 55 / gap 0.91 / vwap > 0 pass
    assert WF.willy_blocks(ON, 55.01, 0.91, 0.1) == ["WILLY_GAP"]
    assert WF.willy_blocks(ON, 30, 0.90, 0.1) == ["WILLY_GAP"]                     # gap ≤ 0.90 (inclusive)
    assert WF.willy_blocks(ON, 30, -0.5, 0.1) == ["WILLY_GAP"]                     # |gap| — a signed value still reads its size
    assert WF.willy_blocks(ON, 30, 2.0, 0.0) == ["WILLY_FAILED_SPIKE"]             # vwap ≤ 0 (inclusive)
    assert WF.willy_blocks(ON, 60, 0.5, -1) == ["WILLY_GAP", "WILLY_FAILED_SPIKE"]
    assert WF.willy_blocks(ON, None, 0.5, None) == ["WILLY_GAP"]                   # one readable leg is enough for the GAP line
    assert WF.willy_blocks(ON, None, None, None) == []                             # unreadable → never blocks
    assert WF.willy_blocks(ON, float("nan"), "x", float("inf")) == []
    assert WF.willy_blocks(ON, 60, 0.5, -1, ht=True) == ["WILLY_FAILED_SPIKE"]     # HT: FAILED_SPIKE only
    ht_off = NS(**{**vars(ON), "watch_block_willy_ht_failed_spike": False})
    assert WF.willy_blocks(ht_off, 60, 0.5, -1, ht=True) == [] and WF.willy_blocks(ht_off, 60, 0.5, -1) == ["WILLY_GAP", "WILLY_FAILED_SPIKE"]
    assert WF.willy_blocks(OFF, 60, 0.5, -1) == []


def test_lite_retry_long():
    assert WF.retry_blocks(ON, 0.0) == ["RETRY_BTC1H"] and WF.retry_blocks(ON, 0.0001) == [] and WF.retry_blocks(ON, None) == []
    assert WF.lite_blocks(ON, 0.0) == ["LITE_BTC_EMA20_DOWN"] and WF.lite_blocks(ON, 0.01) == [] and WF.lite_blocks(ON, None) == []
    assert WF.long_blocks(ON, 119.0) == ["LONG_WEAK_FRENZY"] and WF.long_blocks(ON, 119.01) == [] and WF.long_blocks(ON, None) == []
    for f, v in ((WF.retry_blocks, -1), (WF.lite_blocks, -1), (WF.long_blocks, 100)):
        assert f(OFF, v) == []


def test_fade_and_ml():
    assert WF.fade_blocks(ON, 0.5, 48.5, 1.80) == ["FADE_LATE_WEAK"]               # every leg at its inclusive edge
    assert WF.fade_blocks(ON, 0.0, 40, 1.0) == []                                  # early tape (gap ≤ 0) → outside the line
    assert WF.fade_blocks(ON, 0.5, 48.6, 1.0) == [] and WF.fade_blocks(ON, 0.5, 40, 1.81) == []
    assert WF.fade_blocks(ON, None, 40, 1.0) == [] and WF.fade_blocks(ON, 0.5, None, 1.0) == []   # a conjunction: any unreadable leg = no block
    assert WF.ml_blocks(ON, 0.5, 89.9) == [] and WF.ml_blocks(ON, 0.5001, 89.9) == ["ML_MKT_HOT"]
    assert WF.ml_blocks(ON, 0.2, 90.0) == ["ML_BULL90"] and WF.ml_blocks(ON, 0.9, 95) == ["ML_MKT_HOT", "ML_BULL90"]
    assert WF.ml_blocks(ON, None, None) == [] and WF.ml_blocks(OFF, 0.9, 95) == []


def test_frenzy_dispatch_and_code():
    kw = dict(adx=60, ema_gap_8_13=0.5, vs_vwap_pct=-1, btc_1h_slope=-0.1, btc_ema20_slope_raw=-0.1, frenzy_vol_mult=100)
    assert WF.frenzy_blocks(ON, "FRENZY_WILLY", **kw) == ["WILLY_GAP", "WILLY_FAILED_SPIKE"]
    assert WF.frenzy_blocks(ON, "FRENZY_WILLY_HT", **kw) == ["WILLY_FAILED_SPIKE"]
    assert WF.frenzy_blocks(ON, "FRENZY_LITE_RETRY", **kw) == ["RETRY_BTC1H"]
    assert WF.frenzy_blocks(ON, "FRENZY_LITE", **kw) == ["LITE_BTC_EMA20_DOWN"]
    assert WF.frenzy_blocks(ON, "FRENZY_LONG", **kw) == ["LONG_WEAK_FRENZY"]
    assert WF.frenzy_blocks(ON, "FRENZY_WIDE", **kw) == [] and WF.frenzy_blocks(ON, "MOMENTUM", **kw) == []
    assert WF.block_code("FRENZY_WILLY", ["WILLY_GAP", "WILLY_FAILED_SPIKE"]) == "FRENZY_WILLY_BLOCK_WILLY_GAP+WILLY_FAILED_SPIKE"


def test_watch_zones_agree_with_the_armed_rules():
    """the best case's zones (scripts/watch_zones.py) and the armed rules must block the same fills on the same stamps."""
    import sys
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import watch_zones as WZ
    d = pd.DataFrame(dict(entry_adx=[60, 30, 30, None], entry_ema_gap_8_13=[2.0, 0.5, 2.0, None], entry_frenzy_vs_vwap_pct=[1, 1, -1, None],
                          entry_btc_1h_slope=[0.1, -0.1, 0.0, None], entry_btc_ema20_slope=[0.1, -0.1, 0.0, None],
                          entry_frenzy_vol_mult=[200, 119, 119.5, None], entry_bull_pct=[89, 90, 95, None]))
    z = WZ.zones(d)
    for i in range(len(d)):
        r = d.iloc[i]
        w = WF.frenzy_blocks(ON, "FRENZY_WILLY", adx=r.entry_adx, ema_gap_8_13=r.entry_ema_gap_8_13, vs_vwap_pct=r.entry_frenzy_vs_vwap_pct)
        assert ("WILLY_GAP" in w) == bool(z["FRENZY_WILLY"]["WILLY_GAP"].iloc[i])
        assert ("WILLY_FAILED_SPIKE" in w) == bool(z["FRENZY_WILLY"]["WILLY_FAILED_SPIKE"].iloc[i])
        assert bool(WF.retry_blocks(ON, r.entry_btc_1h_slope)) == bool(z["FRENZY_LITE_RETRY"]["RETRY_BTC1H"].iloc[i])
        assert bool(WF.lite_blocks(ON, r.entry_btc_ema20_slope)) == bool(z["FRENZY_LITE"]["LITE_BTC_EMA20_DOWN"].iloc[i])
        assert bool(WF.long_blocks(ON, r.entry_frenzy_vol_mult)) == bool(z["FRENZY_LONG"]["LONG_WEAK_FRENZY"].iloc[i])
        assert ("ML_BULL90" in WF.ml_blocks(ON, None, r.entry_bull_pct)) == bool(z["MOMENTUM_LONG"]["ML_BULL90"].iloc[i])


def test_config_ui_parity():
    import config
    tj = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    html = _src("templates/index.html")
    for line, key in WF.TOGGLES.items():
        assert config.SignalThresholds.model_fields[key].default is False, key            # default OFF in code
        assert tj[key] is True, key                                                       # armed in the shipped config
        m = re.search(rf"{key}: document\.getElementById\('([a-z0-9-]+)'\)\?\.checked", html)
        assert m, f"no save handler for {key}"
        iid = m.group(1)
        assert html.count(f'id="{iid}"') == 1 and f"getElementById('{iid}'); if (_e) _e.checked = config.thresholds.{key} === true" in html, key


def test_engine_wiring():
    src = _src("services/trading_engine.py")
    fo = src[src.index("    async def _frenzy_open("):]
    fo = fo[:fo.index("\n    async def ", 10)]
    i_bear, i_watch, i_open = fo.index('"FRENZY_BEARISH_UNREAD"'), fo.index("watch_frenzy_blocks("), fo.index("order = await self.open_position(")
    assert i_bear < i_watch < i_open                                                     # judged LAST, right before the open
    blk = fo[i_watch:i_open]
    assert "_btc_ema20_slope_pct_raw" in blk and "self._blocked_record(" in blk and "self._record_filter_block(_wcode" in blk and "return" in blk
    op = src[src.index("    async def open_position("):]
    i_mega, i_wf = op.index('self._record_filter_block("LONG_MEGACAP_BLOCK"'), op.index("watch_ml_blocks(")
    assert i_mega < i_wf and "watch_fade_blocks(" in op[i_wf:i_wf + 3000] and "self._blocked_record(_wf_sleeve" in op[i_wf:i_wf + 4000]
    assert "elif spike_fade and direction == \"SHORT\"" in op[i_wf - 2000:i_wf + 2000]
    assert "watch_block_fade_late_weak" in src and "_current_btc_1d_ema50_200_gap_pct = closed_ema_gap_pct(" in src


def test_blocked_record_is_deduped_and_never_raises(monkeypatch):
    from services.trading_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)
    spawned = []
    monkeypatch.setattr(TradingEngine, "is_paper_mode", property(lambda self: True), raising=False)
    eng._wh_spawn = lambda coro: (spawned.append(coro), coro.close())
    ok1 = eng._blocked_record("FRENZY_WILLY", "LONG", "XUSDT", 1.23, ["WILLY_GAP"], {"entry_adx": 60.0, "bad": float("nan")}, bar_ms=1_700_000_100_000)
    ok2 = eng._blocked_record("FRENZY_WILLY", "LONG", "XUSDT", 1.23, ["WILLY_GAP"], {}, bar_ms=1_700_000_100_000)       # same bar → dropped
    ok3 = eng._blocked_record("FRENZY_WILLY_HT", "LONG", "XUSDT", 1.23, ["WILLY_FAILED_SPIKE"], {}, bar_ms=1_700_000_100_000)   # other sleeve → kept
    assert (ok1, ok2, ok3) == (True, False, True) and len(spawned) == 2
    assert eng._blocked_record("S", "LONG", "P", object(), ["L"], {}) is False                      # a bad price never raises


def test_decisions_export_and_model():
    from models import BlockedSignal
    cols = {c.name for c in BlockedSignal.__table__.columns}
    assert {"pair", "sleeve", "direction", "at", "bar_at", "price", "lines", "sole", "entry_fields_json", "is_paper"} <= cols
    main_src = _src("main.py")
    assert "BLOCKED_EXPORT_COLS" in main_src and "+ _mr_cols + _bl_cols + _xd_cols" in main_src and "delete(BlockedSignal)" in main_src
    import main
    b = NS(at=None, pair="XUSDT", direction="LONG", lines="WILLY_GAP", sleeve="FRENZY_WILLY", price=1.0, bar_at=None, sole=True, is_paper=True,
           entry_fields_json="{}")
    row = main.blocked_export_row(b, lambda d: d)
    assert row["e"] == "BLOCKED" and row["gate"] == "WILLY_GAP" and set(main.BLOCKED_EXPORT_COLS) <= set(row)


def test_master_builder_uses_the_same_rules():
    src = _src("scripts/build_master_pool.py")
    assert 'STACK_VERSION = "2026-11-11a"' in src and "from services import watch_filters as WF" in src
    assert "WF.frenzy_blocks(" in src and "WF.fade_blocks(" in src and "WF.ml_blocks(" in src and "reason[n] = WF.block_code(" in src


def test_master_flatbase_confirm_and_long_minus3():
    """(DL 527 master fix) FLATBASE on FB_CONFIRM + +5 / −10 (reports/FLATBASE_CONFIRM_CF.csv: SKIP refused, CONFIRMED_LIVE as traded, else cf510)
    and FRENZY_LONG −4-era fills on today's −3 — the master equals the best case's B25 numbers."""
    import pandas as pd
    import numpy as np
    src = _src("scripts/build_master_pool.py")
    assert "LONG_STOP_FROZEN = 3.0" in src and '"FLATBASE_CONFIRM_SKIP"' in src and "FLATBASE_CONFIRM_CF.csv" in src
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    cf = pd.read_csv(os.path.join(ROOT, "reports", "FLATBASE_CONFIRM_CF.csv"), comment="#")
    fb = M[M.entry_strategy == "FLATBASE_SHORT"]
    key = set(zip(cf.opened_at.astype(str).str[:19], cf.pair))
    assert all((str(o)[:19], p) in key for o, p in zip(fb.opened_at, fb.pair))            # every master Flatbase fill is covered
    skip = set(zip(cf[cf.cf_why == "SKIP"].opened_at.astype(str).str[:19], cf[cf.cf_why == "SKIP"].pair))
    for r in fb.itertuples():
        if (str(r.opened_at)[:19], r.pair) in skip:
            assert r.stack_block_reason == "FLATBASE_CONFIRM_SKIP" and not bool(r.stack_keep)
    b = fb[(fb.era == "B25") & (fb.stack_keep == True)]   # noqa: E712
    p = pd.to_numeric(b.stack_pct)
    assert len(b) == 22 and round(p.sum(), 2) == 93.74
    l4 = M[(M.entry_strategy == "FRENZY_LONG") & (pd.to_numeric(M.entry_frenzy_stop_pct, errors="coerce") >= 3.5) & (M.stack_keep == True)]   # noqa: E712
    assert len(l4) == 7 and (pd.to_numeric(l4.stack_pct) >= -3.05).all() and np.isclose(pd.to_numeric(l4.stack_pct).sum(), 1.48, atol=0.01)
