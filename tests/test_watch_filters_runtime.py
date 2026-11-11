"""🧱 (DECISION_LOG 527, deep review) the armed watch-line filters at RUNTIME — the real TradingEngine._frenzy_open (SQLite harness of
tests/test_willy_global_hold_switch) and the real open_position (its _e2e harness): each line blocks (one counter, no open, one blocked-signal
record), unreadable inputs and the exempt paths (WIDE, probes, momentum shorts, a stale BTC 1 d gap) never block. Hermetic: no network."""
import asyncio
import time

import pytest

from tests.test_willy_global_hold_switch import _off, _hold_db, _e2e, BAR_OPEN

CASES = [
    ({}, dict(vol_mult=40.0), {}, "FRENZY_LONG_BLOCK_LONG_WEAK_FRENZY"),
    ({}, dict(vol_mult=400.0), {}, None),
    ({"lite": True}, dict(vol_mult=40.0), {"_btc_ema20_slope_pct_raw": -0.01}, "FRENZY_LITE_BLOCK_LITE_BTC_EMA20_DOWN"),
    ({"lite": True}, dict(vol_mult=40.0), {"_btc_ema20_slope_pct_raw": None}, None),
    ({"lite_retry": dict(green=BAR_OPEN - 300_000, ret=0.4, exp=BAR_OPEN + 600_000)}, {}, {"_current_btc_1h_slope": -0.1}, "FRENZY_LITE_RETRY_BLOCK_RETRY_BTC1H"),
    ({"wide": True}, dict(vol_mult=40.0), {"_btc_ema20_slope_pct_raw": -0.01, "_current_btc_1h_slope": -0.1}, None),
    ({"willy": "A"}, dict(vs_vwap_pct=-0.5), {}, "FRENZY_WILLY_BLOCK_WILLY_GAP+WILLY_FAILED_SPIKE"),
    ({"willy": "A", "willy_ht": True}, dict(vs_vwap_pct=-0.5), {}, "FRENZY_WILLY_HT_BLOCK_WILLY_FAILED_SPIKE"),
    ({"catchup_now": BAR_OPEN + 600_000}, dict(vol_mult=40.0), {}, "FRENZY_LONG_BLOCK_LONG_WEAK_FRENZY"),
]

@pytest.mark.parametrize("kw,fx,gl,expect", CASES)
def test_frenzy_block(monkeypatch, kw, fx, gl, expect):
    import services.trading_engine as TE
    _off(monkeypatch, TE)
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**12)
    monkeypatch.setattr(TE, "_current_btc_trend_gap_pct", 0.5)
    monkeypatch.setattr(TE, "_current_btc_1d_ret_pct", 1.0)
    monkeypatch.setattr(TE, "_btc_ema20_slope_pct_raw", 0.02)
    monkeypatch.setattr(TE, "_current_btc_1h_slope", 0.05)
    for k, v in gl.items(): monkeypatch.setattr(TE, k, v)
    th = TE.config.trading_config.thresholds
    for k in ("watch_block_willy_gap","watch_block_willy_failed_spike","watch_block_willy_ht_failed_spike","watch_block_retry_btc1h",
              "watch_block_lite_btc_ema20_down","watch_block_long_weak_frenzy"):
        monkeypatch.setattr(th, k, True)
    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=False)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks, opened, spawned = [], [], []
        e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
        e._flip_entry_fields = lambda *a, **k: {"entry_ema_gap_8_13": 0.5}
        e._sanitize_open_kwargs = lambda ef, s_, d: ef
        e._wh_spawn = lambda coro: (spawned.append(coro), coro.close())
        async def gvv(sig, wait=True): return 0.5
        e._frenzy_gvol_value = gvv; e._frenzy_gvol_catchup_value = gvv
        async def fake_open(**k): opened.append(k); return object()
        e.open_position = fake_open
        flag = dict(pair="FOOUSDT", spike_ts=BAR_OPEN - 600_000, hours=5.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=400.0, run_pct=8.0, atr_pct=1.0,
                    volume_24h=5e7, price=1.0, live_price=1.0, adx_delta=1.0, di_spread=2.0, above_streak=14, above_share=90.0, bar_ret_pct=-0.1, peak=1.1)
        flag.update(fx)
        async with SL() as db:
            r = await e._frenzy_open(db, flag, {"adx": 60.0, "rsi": 55.0}, BAR_OPEN, **kw)
        await eng.dispose()
        return r, blocks, opened, flag, spawned
    r, blocks, opened, flag, spawned = asyncio.run(go())
    wb = [b for b in blocks if "_BLOCK_" in b]
    if expect:
        assert wb == [expect] and opened == [] and len(spawned) == 1 and r is None
    else:
        assert wb == []



def _setup(monkeypatch, share, gap):
    import services.trading_engine as TE
    monkeypatch.setattr(TE, "mkt_up4h_stamp", lambda now_ms=None: {"entry_mkt_up4h_pct": share, "entry_mkt_up4h_n": 90})
    monkeypatch.setattr(TE, "_current_btc_1d_ema50_200_gap_pct", gap)
    monkeypatch.setattr(TE, "_btc_1d_gap_at", time.time())
    th = TE.config.trading_config.thresholds
    for k in ("watch_block_ml_mkt_hot","watch_block_ml_bull90","watch_block_fade_late_weak"):
        monkeypatch.setattr(th, k, True)

def test_ml_hot_blocks(monkeypatch):
    _setup(monkeypatch, 0.7, None)
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", entry_bull_pct=50.0)
    assert r is None and "MOMENTUM_LONG_BLOCK_ML_MKT_HOT" in blocks and ("FOOUSDT","MOMENTUM") not in opened

def test_ml_bull90_blocks(monkeypatch):
    _setup(monkeypatch, None, None)
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", entry_bull_pct=92.0)
    assert r is None and "MOMENTUM_LONG_BLOCK_ML_BULL90" in blocks

def test_ml_cold_opens(monkeypatch):
    _setup(monkeypatch, 0.3, None)
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", entry_bull_pct=50.0)

def test_probe_exempt(monkeypatch):
    _setup(monkeypatch, 0.7, None)
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", entry_bull_pct=95.0, gap_probe=True)

def test_fade_blocks(monkeypatch):
    _setup(monkeypatch, 0.7, 5.0)
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", direction="SHORT", spike_fade=True, entry_pos_di=40.0, entry_dist_from_ema13_pct=1.0)
    assert r is None and "SPIKE_FADE_BLOCK_FADE_LATE_WEAK" in blocks

def test_fade_stale_gap_fail_open(monkeypatch):
    _setup(monkeypatch, 0.7, 5.0)
    import services.trading_engine as TE
    monkeypatch.setattr(TE, "_btc_1d_gap_at", time.time() - 9000)
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", direction="SHORT", spike_fade=True, entry_pos_di=40.0, entry_dist_from_ema13_pct=1.0)
    assert "SPIKE_FADE_BLOCK_FADE_LATE_WEAK" not in blocks

def test_short_momentum_untouched(monkeypatch):
    _setup(monkeypatch, 0.7, 5.0)
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", direction="SHORT", entry_bull_pct=95.0, entry_pos_di=40.0, entry_dist_from_ema13_pct=1.0)
