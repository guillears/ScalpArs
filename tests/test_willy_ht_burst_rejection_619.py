"""💥 (DECISION_LOG 619) WILLY_HT_BURST_REJECTION — FRENZY_WILLY_HT refused when, on the 1m bars CLOSED before the decision minute,
pre_vol15_vs60 > 1.60 AND cloc15 < 0.50 (a declared override of the DL 618 watch line). The pure rule + the shared cloc15 (services), the
rolling 1m buffer, the engine at RUNTIME (the real _frenzy_open / _frenzy_willy_ht_try on the SQLite harness of
tests/test_willy_global_hold_switch, with the 1m read stubbed), D11 / D13 wiring, and the pricer's TIGHT absolute revert gate (10 / 5).
Hermetic: no network."""
import asyncio
import json
import pathlib
import sys
import time
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from services import frenzy as FZ
from services import watch_filters as WF
from tests.test_willy_global_hold_switch import _off, _hold_db, BAR_OPEN

ROOT = pathlib.Path(__file__).resolve().parents[1]
ON = NS(watch_block_willy_ht_failed_spike=False, watch_block_willy_failed_spike=False, watch_block_willy_gap=False,
        frenzy_willy_ht_min_dip_pct=0, frenzy_willy_ht_burst_rejection_enabled=True)
M = 60_000


def _bars(m0, recent_vol=200.0, base_vol=100.0, last_close=0.95, hi=1.1, lo=0.9, n_base=60, n_recent=15, forming=True):
    """1m klines ending at the bar before minute m0 (+ the forming bar at m0): n_base bars at base_vol, then n_recent at recent_vol
    whose range is [lo, hi] and whose last close is last_close."""
    out = []
    t = m0 - (n_base + n_recent) * M
    for i in range(n_base):
        out.append([t, 1.0, 1.0, 1.0, 1.0, base_vol]); t += M
    for i in range(n_recent):
        c = last_close if i == n_recent - 1 else 1.0
        out.append([t, 1.0, hi, lo, c, recent_vol]); t += M
    if forming:
        out.append([m0, 1.0, 5.0, 0.1, 0.2, 9e9])   # the forming bar: never read (opened at the decision minute)
    return out


# ── ① the pure functions ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_cloc15_window_flat_short_and_lookahead():
    m0 = 1_700_000_000_000 // M * M
    b = _bars(m0)
    assert FZ.cloc15(b, m0 + 7_000) == pytest.approx(0.25)                 # (0.95 − 0.9) ÷ 0.2; the forming bar at m0 is ignored
    assert FZ.pre_vol15_vs60(b, m0 + 7_000) == pytest.approx(2.0)
    assert FZ.cloc15(b[-15:], m0) is None                                  # 14 closed bars before m0 (+ the forming one) → None
    assert FZ.cloc15(_bars(m0, hi=1.0, lo=1.0, last_close=1.0), m0) is None   # flat range
    assert FZ.cloc15(None, m0) is None and FZ.cloc15([["x"]], m0) is None
    assert FZ.pre_vol15_vs60(b[-60:], m0) is None                          # < 75 bars → None
    assert FZ.CLOC_BARS == 15


def test_state_strict_cuts():
    S = WF.ht_burst_state
    assert S(2.0, 0.25) == "zone"
    assert S(1.60, 0.10) == "rest"                                         # pre_vol exactly 1.60 → no burst (strict >)
    assert S(1.6001, 0.4999) == "zone"
    assert S(2.0, 0.50) == "rest"                                          # cloc exactly 0.50 → trade (strict <)
    assert S(1.2, None) == "rest"                                          # no burst: cloc not needed
    assert S(None, 0.1) == "unread" and S(2.0, None) == "unread" and S(float("nan"), 0.1) == "unread"
    assert WF.HT_BURST_PRE_VOL_MIN == 1.60 and WF.HT_BURST_CLOC_MAX == 0.50  # frozen


def test_rule_in_frenzy_blocks():
    blk = lambda pv, c, th=ON, sl="FRENZY_WILLY_HT": WF.frenzy_blocks(th, sl, vs_vwap_pct=1.0, price=1.0, spike_high=1.1, pre_vol=pv, cloc=c)
    assert blk(2.0, 0.25) == ["WILLY_HT_BURST_REJECTION"]
    assert blk(2.0, 0.75) == [] and blk(1.2, 0.1) == [] and blk(None, None) == [] and blk(2.0, None) == []
    for v in (False, None, "true", 1):                                     # only a real True arms it
        assert blk(2.0, 0.25, NS(**{**vars(ON), "frenzy_willy_ht_burst_rejection_enabled": v})) == []
    assert WF.ht_burst_on(NS()) is False
    for sl in ("FRENZY_WILLY", "FRENZY_LITE", "FRENZY_LONG", "FRENZY_LITE_RETRY"):   # HT only
        assert blk(2.0, 0.25, sl=sl) == []
    both = NS(**{**vars(ON), "frenzy_willy_ht_min_dip_pct": 1.4})
    assert WF.frenzy_blocks(both, "FRENZY_WILLY_HT", price=1.095, spike_high=1.1, pre_vol=2.0, cloc=0.25) == ["WILLY_HT_NEAR_HIGH", "WILLY_HT_BURST_REJECTION"]


def test_scout_reuses_the_shared_cloc15():
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_willy_ht_lines as L
    m0 = 1_700_000_000_000 // M * M
    b = _bars(m0, last_close=0.97)
    assert L.cloc15(b, m0) == FZ.cloc15(b, m0) and L.CLOC_BARS == FZ.CLOC_BARS and L.CLOC_CUT == WF.HT_BURST_CLOC_MAX
    assert L.PRE_VOL_CUT == WF.HT_BURST_PRE_VOL_MIN and L.DL_ARMED_BURST == 619
    for pv, c in ((2.0, 0.25), (2.0, 0.75), (1.2, None), (None, 0.1), (2.0, None)):   # the line's zone ⇔ the armed rule's zone
        assert (L.zone_burst_rejection(pv, c) == "zone") == (WF.ht_burst_state(pv, c) == "zone")
    src = (ROOT / "scripts/scout_willy_ht_lines.py").read_text()
    assert "ARMED for WILLY_HT since DECISION_LOG 619" in src and "WILLY_HT_BURST_REJECTION" in src


# ── ② the rolling buffer ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_buffer_merge_final_coverage():
    m0 = 1_700_000_000_000 // M * M
    buf = {}
    FZ.burst_buf_merge(buf, _bars(m0), m0 + 1_500)                         # requested 1.5 s after m0 began
    B = buf["bars"]
    assert B[m0][1] is False and B[m0 - M][1] is True                      # the forming bar is not final
    assert buf["cov"] == [[m0 - 75 * M, m0]]
    rows = FZ.burst_buf_rows(buf, m0 + 5_000)
    assert rows is not None and len(rows) == 75 and rows[-1][0] == m0 - M
    assert FZ.burst_buf_rows(buf, m0 + M + 5_000) is None                  # next minute: coverage ends at m0 → read once
    FZ.burst_buf_merge(buf, [[m0, 1, 1, 1, 1, 100.0], [m0 + M, 1, 1, 1, 1, 1.0]], m0 + M + 1_500)   # the refresh after m0 closes
    assert B[m0] == ([m0, 1.0, 1.0, 1.0, 1.0, 100.0], True) and buf["cov"] == [[m0 - 75 * M, m0 + M]]
    assert FZ.burst_buf_rows(buf, m0 + M + 5_000)[-1][0] == m0
    FZ.burst_buf_merge(buf, [[m0, 9, 9, 9, 9, 9.0]], m0 + 10)              # a non-final read never overwrites a final bar
    assert B[m0][0][5] == 100.0
    big = {}
    FZ.burst_buf_merge(big, [[m0 + i * M, 1, 1, 1, 1, 1] for i in range(300)], m0 + 400 * M)
    assert len(big["bars"]) == FZ.BURST_BUF_KEEP and big["cov"] == [[m0 + 150 * M, m0 + 400 * M]]   # coverage clamped to the bars kept


def test_finality_is_judged_at_the_request_time():
    """619 review: a bar still forming when the request was SENT is never final, even if the answer arrives after it closed."""
    m0 = 1_700_000_000_000 // M * M
    buf = {}
    FZ.burst_buf_merge(buf, _bars(m0), m0 - 200)                           # sent 0.2 s before m0 (bar m0 − 1 still forming)
    assert buf["bars"][m0 - M][1] is False and FZ.burst_buf_rows(buf, m0 + 5_000) is None


def test_buf_equals_fetch_on_the_same_binance_rows_gaps_included():
    """619 review: BUF and FETCH give the SAME values on the same Binance rows — the raw definition (the last 75 / 15 bars opened before the
    minute, as returned). A minute Binance omits is omitted in both; a gap only the BUFFER has (a missed refresh) is never read as complete."""
    m0 = 1_700_000_000_000 // M * M
    raw = _bars(m0, n_base=70)                                             # 85 closed bars + the forming one
    raw = [r for r in raw if r[0] != m0 - 40 * M]                          # Binance itself omits one minute
    buf = {}
    FZ.burst_buf_merge(buf, raw, m0 + 1_500)
    rows = FZ.burst_buf_rows(buf, m0 + 5_000)
    assert rows is not None and rows[0][0] == m0 - 76 * M                  # the 75 rows as returned (one older bar fills the omitted minute)
    for f in (FZ.pre_vol15_vs60, FZ.cloc15):
        assert f(rows, m0 + 5_000) == f(raw, m0 + 5_000)
    hole = {}
    FZ.burst_buf_merge(hole, [r for r in raw if r[0] < m0 - 10 * M], m0 - 10 * M + 1_500)   # an old read …
    FZ.burst_buf_merge(hole, [r for r in raw if r[0] >= m0 - 5 * M], m0 + 1_500)          # … then a tail read: 4 minutes never read
    assert FZ.burst_buf_rows(hole, m0 + 5_000) is None


def test_thin_coin_zero_volume_minutes_are_rows():
    """619 review (checked live 2026-11-21 on two USDⓈ-M coins with ~500–800 trades / 24 h: 1,000 1m bars, 0 gaps, ~88 % zero-volume rows):
    zero-volume minutes come back as rows → they sit in the window and in the base mean like any bar."""
    m0 = 1_700_000_000_000 // M * M
    raw = _bars(m0, base_vol=0.0)
    assert FZ.pre_vol15_vs60(raw, m0) is None                              # an all-zero base hour → unscored (study parity)
    raw[0][5] = 60.0                                                       # one trade in the hour → base mean 1.0
    assert FZ.pre_vol15_vs60(raw, m0) == pytest.approx(200.0)


def test_read_uses_buffer_then_one_capped_fetch(monkeypatch):
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    now = int(time.time() * 1000); m0 = now // M * M
    calls = []
    monkeypatch.setattr(TE.binance_service, "burst_limited", lambda: False)

    async def fake(pair, limit=90):
        calls.append(limit); return _bars(m0)
    monkeypatch.setattr(TE.binance_service, "get_klines_1m_recent", fake)
    monkeypatch.setattr(TE, "_ht_burst_buf", {})
    pv, c, ms, src = asyncio.run(e._ht_burst_read("FOOUSDT", now))         # empty buffer → one fetch, merged
    assert (src, calls) == ("FETCH", [TE.HT_BURST_LIMIT_FULL]) and pv == pytest.approx(2.0) and c == pytest.approx(0.25)
    pv2, c2, ms, src = asyncio.run(e._ht_burst_read("FOOUSDT", now))       # now from memory — identical values
    assert src == "BUF" and calls == [TE.HT_BURST_LIMIT_FULL] and (pv2, c2) == (pv, c) and ms < 50

    async def slow(pair, limit=90):
        await asyncio.sleep(5); return _bars(m0)
    monkeypatch.setattr(TE.binance_service, "get_klines_1m_recent", slow)
    monkeypatch.setattr(TE, "HT_BURST_FETCH_TIMEOUT_S", 0.05)
    monkeypatch.setattr(TE, "_ht_burst_buf", {})
    pv, c, ms, src = asyncio.run(e._ht_burst_read("FOOUSDT", now))
    assert (pv, c, src) == (None, None, "TIMEOUT") and ms < 1000           # capped → fail-open upstream
    monkeypatch.setattr(TE.binance_service, "burst_limited", lambda: True)
    assert asyncio.run(e._ht_burst_read("FOOUSDT", now))[3] == "LIMITED"   # ban / back-off → no read at all


def test_candidates_pending_first_and_capped(monkeypatch):
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    e._fz_willy_pending = {"PENDUSDT": {}}
    monkeypatch.setattr(TE, "_frenzy_flags", {f"P{i}USDT": {"spike_ts": i} for i in range(60)} | {"PENDUSDT": {"spike_ts": 0}})
    c = e._ht_burst_candidates()
    assert c[0] == "PENDUSDT" and c[1] == "P59USDT" and len(c) == TE.HT_BURST_BUF_MAX_PAIRS


def test_buffer_refresh_tail_vs_full(monkeypatch):
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    now = int(time.time() * 1000); m0 = now // M * M
    calls = []

    async def fake(pair, limit=90):
        calls.append(limit)
        return _bars(m0) if limit == TE.HT_BURST_LIMIT_FULL else [[m0 - k * M, 1, 1, 1, 1, 1] for k in range(limit - 1, -1, -1)]
    monkeypatch.setattr(TE.binance_service, "get_klines_1m_recent", fake)
    monkeypatch.setattr(TE, "_ht_burst_buf", {})
    asyncio.run(e._ht_burst_refresh_pair("FOOUSDT"))
    asyncio.run(e._ht_burst_refresh_pair("FOOUSDT"))
    assert calls == [TE.HT_BURST_LIMIT_FULL, TE.HT_BURST_LIMIT_TAIL] and TE.HT_BURST_LIMIT_TAIL == 6   # full once, then the tail (weight 1 each)


def test_tail_ok_needs_coverage_from_m0_minus_75_to_m0_minus_5(monkeypatch):
    """619 review #7: the tail read (m0 − 5 … m0 − 1 + the forming bar) is enough only when the buffer already covers m0 − 75 … m0 − 5."""
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    now = int(time.time() * 1000); m0 = now // M * M
    calls = []

    async def fake(pair, limit=90):
        calls.append(limit); return None
    monkeypatch.setattr(TE.binance_service, "get_klines_1m_recent", fake)
    for cov, want in (([[m0 - 75 * M, m0 - 5 * M]], TE.HT_BURST_LIMIT_TAIL), ([[m0 - 74 * M, m0]], TE.HT_BURST_LIMIT_FULL),
                      ([[m0 - 80 * M, m0 - 6 * M]], TE.HT_BURST_LIMIT_FULL)):
        calls.clear()
        monkeypatch.setattr(TE, "_ht_burst_buf", {"FOOUSDT": {"bars": {}, "cov": cov}})
        asyncio.run(e._ht_burst_refresh_pair("FOOUSDT"))
        assert calls == [want], cov


def test_loop_sleep_never_skips_a_minute():
    import services.trading_engine as TE
    f = TE.TradingEngine._ht_burst_sleep_s
    off = TE.HT_BURST_REFRESH_OFFSET_S
    base = 1_700_000_000.0 - (1_700_000_000.0 % 60)
    assert f(base) == pytest.approx(off)                                   # at :00 → :01.5 of the SAME minute
    assert f(base + 1.0) == pytest.approx(off - 1.0)
    assert f(base + off) == pytest.approx(60.0)                            # exactly at the moment → the next minute (never 0)
    assert f(base + 59.9) == pytest.approx(off + 0.1)
    for t in (0.0, 0.7, 1.49, 1.51, 30.0, 59.99):
        assert 0 < f(base + t) <= 60.0 and ((base + t + f(base + t)) % 60.0) == pytest.approx(off, abs=1e-6)


def test_shutdown_cancels_the_loop(monkeypatch):
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)

    async def go():
        async def forever():
            await asyncio.sleep(3600)
        e._ht_burst_task = asyncio.get_running_loop().create_task(forever())
        t = e._ht_burst_task
        await asyncio.sleep(0)
        await e.stop_ht_burst_loop()
        return t
    t = asyncio.run(go())
    assert t.cancelled() and e._ht_burst_task is None
    src = (ROOT / "main.py").read_text()
    assert "await trading_engine.stop_ht_burst_loop()" in src


def test_replay_without_a_1m_source_is_not_applied(monkeypatch):
    """619 review #10: an engine replay's fake exchange has no get_klines_1m_recent → source NA, no UNREAD count, no loop."""
    import services.trading_engine as TE
    monkeypatch.setattr(TE, "binance_service", NS())
    e = object.__new__(TE.TradingEngine)
    assert TE.TradingEngine._ht_burst_source_ok() is False
    assert asyncio.run(e._ht_burst_read("FOOUSDT", int(time.time() * 1000))) == (None, None, 0, "NA")
    th = TE.config.trading_config.thresholds
    monkeypatch.setattr(th, "frenzy_willy_enabled", True, raising=False)
    monkeypatch.setattr(th, "frenzy_willy_ht_burst_rejection_enabled", True, raising=False)

    async def go():
        e._ht_burst_buf_ensure()
        return e.__dict__.get('_ht_burst_task')
    assert asyncio.run(go()) is None


# ── ③ the engine at runtime ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _run(monkeypatch, read=(2.0, 0.25, 3, "BUF"), on=True, via_try=False, min_dip=0, peak=1.05):
    import services.trading_engine as TE
    _off(monkeypatch, TE)
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**12)
    monkeypatch.setattr(TE, "_current_btc_trend_gap_pct", 0.5)
    monkeypatch.setattr(TE, "_current_btc_1d_ret_pct", 1.0)
    th = TE.config.trading_config.thresholds
    monkeypatch.setattr(th, "frenzy_willy_ht_min_dip_pct", min_dip, raising=False)
    monkeypatch.setattr(th, "frenzy_willy_ht_burst_rejection_enabled", on, raising=False)
    monkeypatch.setattr(th, "watch_block_willy_ht_failed_spike", False)

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=False)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks, opened, spawned, rec, ev, reads = [], [], [], [], [], []
        e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
        e._flip_entry_fields = lambda *a, **k: {"entry_ema_gap_8_13": 0.5}
        e._sanitize_open_kwargs = lambda ef, s_, d: ef
        e._wh_spawn = lambda coro: (spawned.append(coro), coro.close())
        _br = e._blocked_record
        e._blocked_record = lambda *a, **k: (rec.append((a, k)), _br(*a, **k))[1]
        e._willy_event = lambda gate, pair, reason=None, price=None, tally=True: ev.append((gate, pair, reason, price, tally))

        async def fake_read(pair, ms):
            reads.append((pair, ms)); return read
        e._ht_burst_read = fake_read

        async def gvv(sig, wait=True): return 0.5
        e._frenzy_gvol_value = gvv; e._frenzy_gvol_catchup_value = gvv

        async def fake_open(**k): opened.append(k); return object()
        e.open_position = fake_open
        flag = dict(pair="FOOUSDT", spike_ts=BAR_OPEN - 600_000, hours=5.0, vwap=0.9, vs_vwap_pct=1.0, vol_mult=400.0, run_pct=8.0, atr_pct=1.0,
                    volume_24h=5e7, price=1.0, live_price=1.0, adx_delta=1.0, di_spread=2.0, above_streak=14, above_share=90.0,
                    bar_ret_pct=-0.1, peak=peak)
        async with SL() as db:
            if via_try:
                pd_ = dict(trig="A", armed=BAR_OPEN - 600_000)
                r = await e._frenzy_willy_ht_try(db, "FOOUSDT", dict(bar_ret_pct=-0.1), flag, {"adx": 30.0, "rsi": 55.0}, BAR_OPEN, pd_, 0.5, "x")
            else:
                r = await e._frenzy_open(db, flag, {"adx": 30.0, "rsi": 55.0}, BAR_OPEN, willy="A", willy_ht=True)
        await eng.dispose()
        return r, blocks, opened, spawned, rec, ev, flag, reads
    return asyncio.run(go())


def test_engine_refuses_burst_with_low_close(monkeypatch):
    r, blocks, opened, spawned, rec, ev, flag, reads = _run(monkeypatch)
    assert r is None and opened == [] and len(reads) == 1
    assert [b for b in blocks if "_BLOCK_" in b] == ["FRENZY_WILLY_HT_BLOCK_WILLY_HT_BURST_REJECTION"]
    assert len(rec) == 1 and len(spawned) == 1
    (sl, d, pair, px, lines, ef), kw = rec[0]
    assert (sl, d, pair, lines) == ("FRENZY_WILLY_HT", "LONG", "FOOUSDT", ["WILLY_HT_BURST_REJECTION"])
    assert ef["entry_frenzy_pre_vol15_dec"] == 2.0 and ef["entry_frenzy_cloc15"] == 0.25 and ef["entry_frenzy_burst_read_ms"] == 3
    assert len(ev) == 1 and ev[0][0] == "WILLY_HT_BURST_REJECTION" and ev[0][4] is False
    assert "pre_vol 2.00" in ev[0][2] and "cloc15 0.25" in ev[0][2]
    assert "armed filter WILLY_HT_BURST_REJECTION" in flag["willy_ht_last"]


def test_engine_trades_burst_with_high_close_and_stamps(monkeypatch):
    r, blocks, opened, _, rec, ev, _, _ = _run(monkeypatch, read=(2.0, 0.75, 0, "BUF"))
    assert r is True and len(opened) == 1 and rec == [] and ev == [] and not [b for b in blocks if "BURST" in b]
    k = opened[0]
    assert k["entry_frenzy_pre_vol15_dec"] == 2.0 and k["entry_frenzy_cloc15"] == 0.75 and k["entry_frenzy_burst_read_ms"] == 0


def test_engine_trades_without_burst(monkeypatch):
    r, blocks, opened, _, rec, *_ = _run(monkeypatch, read=(1.1, 0.05, 0, "BUF"))
    assert r is True and len(opened) == 1 and rec == [] and not [b for b in blocks if "BURST" in b]


def test_engine_fail_open_unread(monkeypatch):
    for read in ((None, None, 1500, "TIMEOUT"), (2.0, None, 2, "BUF"), (None, 0.1, 2, "FETCH")):
        r, blocks, opened, _, rec, *_ = _run(monkeypatch, read=read)
        assert r is True and len(opened) == 1 and rec == [] and "WILLY_HT_BURST_REJECTION_UNREAD" in blocks, read


def test_engine_switch_off_never_reads_or_refuses(monkeypatch):
    r, blocks, opened, _, rec, ev, _, reads = _run(monkeypatch, on=False)
    assert r is True and len(opened) == 1 and rec == [] and reads == [] and not [b for b in blocks if "BURST" in b]
    assert opened[0]["entry_frenzy_cloc15"] is None


def test_ht_pending_entry_ends_on_the_block(monkeypatch):
    r, blocks, opened, *_ = _run(monkeypatch, via_try=True)
    assert r == "ended" and opened == [] and "FRENZY_WILLY_HT_BLOCK_WILLY_HT_BURST_REJECTION" in blocks
    r, _, opened, *_ = _run(monkeypatch, read=(2.0, 0.9, 0, "BUF"), via_try=True)
    assert r == "opened" and len(opened) == 1


def test_near_high_refusal_skips_the_burst_read(monkeypatch):
    """619 review #3: NEAR_HIGH (or any earlier HT line) already refuses → no 1m read; the row's burst stamps are None; one counter."""
    r, blocks, opened, _, rec, ev, flag, reads = _run(monkeypatch, min_dip=1.4, peak=1.01)
    assert r is None and opened == [] and reads == []
    assert [b for b in blocks if "_BLOCK_" in b] == ["FRENZY_WILLY_HT_BLOCK_WILLY_HT_NEAR_HIGH"] and "WILLY_HT_BURST_REJECTION_UNREAD" not in blocks
    (sl, d, pair, px, lines, ef), kw = rec[0]
    assert lines == ["WILLY_HT_NEAR_HIGH"] and ef["entry_frenzy_cloc15"] is None and ef["entry_frenzy_pre_vol15_dec"] is None
    assert [x[0] for x in ev] == ["WILLY_HT_NEAR_HIGH"]
    r, blocks, opened, _, rec, ev, _, reads = _run(monkeypatch, min_dip=1.4, peak=1.05)   # near-high passes → the burst line judges
    assert len(reads) == 1 and [b for b in blocks if "_BLOCK_" in b] == ["FRENZY_WILLY_HT_BLOCK_WILLY_HT_BURST_REJECTION"]
    assert [x[0] for x in ev] == ["WILLY_HT_BURST_REJECTION"]


def test_engine_na_source_is_not_unread(monkeypatch):
    r, blocks, opened, _, rec, *_ = _run(monkeypatch, read=(None, None, 0, "NA"))
    assert r is True and len(opened) == 1 and rec == [] and not [b for b in blocks if "BURST" in b]


def test_open_position_writes_the_stamps(monkeypatch):
    from tests.test_frenzy_willy_ht import _open
    r, _, _ = _open(monkeypatch, "FOOUSDT", frenzy_willy=True, frenzy_willy_ht=True, entry_frenzy_pre_vol15_dec=2.5, entry_frenzy_cloc15=0.3,
                    entry_frenzy_burst_read_ms=4)
    assert r is not None and r.entry_strategy == "FRENZY_WILLY_HT"
    assert r.entry_frenzy_pre_vol15_dec == pytest.approx(2.5) and r.entry_frenzy_cloc15 == pytest.approx(0.3) and r.entry_frenzy_burst_read_ms == 4


# ── ④ D11 / D13 wiring ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_config_json_ui_validator_and_stamp_contract():
    import config
    import models
    from services.trading_engine import TradingEngine
    from services.frenzy import frenzy_willy_ht_config_errors as E
    mf = type(config.trading_config.thresholds).model_fields
    assert mf["frenzy_willy_ht_burst_rejection_enabled"].default is False           # code default OFF
    src = (ROOT / "config.py").read_text()
    assert "DECISION_LOG 619" in src and "first 10 priced blocked HT trades on" in src and "≥ 5 days" in src
    assert json.loads((ROOT / "trading_config.json").read_text())["thresholds"]["frenzy_willy_ht_burst_rejection_enabled"] is True
    html = (ROOT / "templates/index.html").read_text()
    assert html.count('id="config-fz-willy-ht-burst-rejection"') == 1                # the input
    assert ("getElementById('config-fz-willy-ht-burst-rejection'); if (_e) _e.checked = config.thresholds.frenzy_willy_ht_burst_rejection_enabled === true;"
            in html)                                                                  # load
    assert "frenzy_willy_ht_burst_rejection_enabled: document.getElementById('config-fz-willy-ht-burst-rejection')?.checked ?? false," in html   # save
    assert "HT burst-rejection block ${" in html                                      # the config block (shared by both exports)
    for c in ("entry_frenzy_pre_vol15_dec", "entry_frenzy_cloc15", "entry_frenzy_burst_read_ms"):
        assert c in models.Order.__table__.columns and TradingEngine.stamp_class(c) == "sleeve"
        assert f"'{c}'" in (ROOT / "database.py").read_text()                          # migration
    base = dict(frenzy_willy_tp_pct=1.25, frenzy_willy_ht_tp_pct=6.0)
    bad = lambda v: [e for e in E(NS(**base, frenzy_willy_ht_burst_rejection_enabled=v)) if "burst" in e]
    assert not bad(True) and not bad(False) and bad("yes")


# ── ⑤ the pricer: the line + the TIGHT absolute revert gate ───────────────────────────────────────────────────────────────────────────
def _S():
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_blocked_filters as S
    return S


def _B(pcts, days=None, pend=0, sole=True, line="WILLY_HT_BURST_REJECTION"):
    days = days if days is not None else list(range(len(pcts)))
    rows = [dict(te=i, lines=line, status="PRICED", pct=p, day=d, sleeve="FRENZY_WILLY_HT", sole=sole) for i, (p, d) in enumerate(zip(pcts, days))]
    rows += [dict(te=-1, lines=line, status="PENDING", pct=None, day=0, sleeve="FRENZY_WILLY_HT", sole=sole)] * pend
    return pd.DataFrame(rows)


def test_pricer_line_and_tight_gate():
    S = _S()
    assert S.LINES["FRENZY_WILLY_HT"] == ("WILLY_FAILED_SPIKE", "WILLY_HT_NEAR_HIGH", "WILLY_HT_BURST_REJECTION")
    assert S.ABS_GATE["WILLY_HT_BURST_REJECTION"] == (10, 5) and S.ABS_GATE["WILLY_HT_NEAR_HIGH"] == (15, 8)   # tighter than NEAR_HIGH
    assert "WILLY_HT_BURST_REJECTION" in S.SOLE_GATED and S.LINE_FLOOR_DL["WILLY_HT_BURST_REJECTION"] == 619
    K = pd.DataFrame(columns=["te", "status", "pct", "sleeve"])
    g = lambda B, **k: S.gate_read(B, K, "WILLY_HT_BURST_REJECTION", **k)[0]
    assert g(_B([-1.0] * 9)) == "COLLECTING"
    assert g(_B([-1.0] * 10)) == "KEEP"
    assert g(_B([-1.0] * 10, days=[i // 3 for i in range(10)])) == "COLLECTING"       # 10 on 4 days → waits for the 5th day
    assert g(_B([-1.0] * 12, days=[i // 3 for i in range(12)])) == "COLLECTING"
    assert g(_B([-1.0] * 13, days=[i // 3 for i in range(13)])) == "KEEP"             # decided at the first prefix with 5 days
    assert g(_B([1.0] * 4 + [-1.0] * 6)) == "KEEP"
    assert g(_B([1.0] * 5 + [-1.0] * 5)).startswith("REVERT")                         # avg exactly 0 → revert (≥ 0)
    assert g(_B([-1.0] * 10, pend=1)) == "COLLECTING"
    assert g(_B([-1.0] * 10, sole=False), sole=True) == "COLLECTING"


def _dec_rows(T0, n, line):
    out = []
    for i in range(n):
        t = T0 + pd.Timedelta(days=i // 2, minutes=i * 7)
        out.append(dict(t=t.strftime("%Y-%m-%dT%H:%M:%S.000"), e="BLOCKED", pair=f"P{i}USDT", dir="LONG", price=1.0, gate=line,
                        strategy="FRENZY_WILLY_HT", bl_sleeve="FRENZY_WILLY_HT", bl_bar_at=t.strftime("%Y-%m-%dT%H:%M:00.000"),
                        bl_lines=line, bl_sole=True, bl_is_paper=True, bl_entry_fields="{}"))
    return out


def test_pricer_run_floor_and_freeze(tmp_path, monkeypatch):
    S = _S()
    T0 = pd.Timestamp("2026-11-22 00:00"); since = int(T0.value // 10**6)
    p = tmp_path / "scalpars_decisions_paper_2026-12-10_00-00-00.csv"
    pd.DataFrame(_dec_rows(T0, 12, "WILLY_HT_BURST_REJECTION")).to_csv(p, index=False)
    monkeypatch.setattr(S, "price_all", lambda B, now_ms, fetch=True, store_path=None: B.assign(pct=-1.0, why="STOP", status="PRICED", exit_ms=B.te + 60_000))
    st = tmp_path / "g.json"
    kw = dict(now_ms=since + 30 * 86_400_000, dec_paths=[str(p)], ord_paths=[], store_path=None, since_ms=since)
    txt = "\n".join(S.run(state_path=str(st), line_floors={"WILLY_HT_BURST_REJECTION": None}, **kw))
    assert "NOT EVALUABLE — the DECISION_LOG 619 commit is not in git" in txt and not st.exists()
    txt = "\n".join(S.run(state_path=str(st), line_floors={"WILLY_HT_BURST_REJECTION": since + 3 * 86_400_000}, **kw))   # 6 refusals before the floor
    assert not st.exists() and "6/10 priced on 3/5 days" in txt                                  # only the 6 after the floor count
    txt = "\n".join(S.run(state_path=str(st), line_floors={"WILLY_HT_BURST_REJECTION": since}, **kw))
    g = json.load(open(st))
    assert g["FRENZY_WILLY_HT@608:WILLY_HT_BURST_REJECTION"]["state"] == "KEEP"
    assert g["FRENZY_WILLY_HT@608:WILLY_HT_BURST_REJECTION|sole"]["state"] == "KEEP"
    assert "FRENZY_WILLY_HT · WILLY_HT_BURST_REJECTION (sole)" in txt and "TIGHTER absolute gate" in txt


# ── ⑥ the burst client (binance_service), called unbound with a fake exchange ──────────────────────────────────────────────────────────
class _FakeEx:
    def __init__(self, rows=None, err=None, weight="12"):
        self.rows, self.err, self.calls = rows or [], err, []
        self.last_response_headers = {"X-MBX-USED-WEIGHT-1M": weight} if weight is not None else {}

    async def fapiPublicGetKlines(self, params):
        self.calls.append(params)
        if self.err:
            raise self.err
        return self.rows


def _svc(ex):
    import services.binance_service as BS
    svc = object.__new__(BS.BinanceService)
    svc.burst_exchange, svc.burst_limited_until, svc.burst_used_weight = ex, 0.0, 0
    return BS, svc


def test_burst_client_read_weight_symbol_and_clamp(monkeypatch):
    raw = [[1, "1.0", "1.2", "0.9", "1.1", "50", 2, "x"]]
    BS, svc = _svc(_FakeEx(rows=raw))
    monkeypatch.setattr(BS, "_ban_until", 0)
    out = asyncio.run(BS.BinanceService.get_klines_1m_recent(svc, "ABC/USDT:USDT", 500))
    assert out == [[1, 1.0, 1.2, 0.9, 1.1, 50.0]]
    assert svc.burst_exchange.calls == [{"symbol": "ABCUSDT", "interval": "1m", "limit": 99}]   # normalised · clamped ≤ 99 (weight 1)
    assert svc.burst_used_weight == 12 and svc.burst_limited_until == 0.0
    asyncio.run(BS.BinanceService.get_klines_1m_recent(svc, "XYZUSDT", 0))
    assert svc.burst_exchange.calls[-1] == {"symbol": "XYZUSDT", "interval": "1m", "limit": 1}


def test_burst_client_weight_stand_down_to_the_next_minute(monkeypatch):
    BS, svc = _svc(_FakeEx(rows=[], weight="1500"))
    monkeypatch.setattr(BS, "_ban_until", 0)
    t0 = time.time()
    asyncio.run(BS.BinanceService.get_klines_1m_recent(svc, "ABCUSDT", 90))
    nxt = (int(t0 // 60) + 1) * 60
    assert svc.burst_used_weight == 1500 and svc.burst_limited_until == pytest.approx(nxt, abs=60)
    assert svc.burst_limited_until % 60 == 0 and BS.BinanceService.burst_limited(svc) is True
    n = len(svc.burst_exchange.calls)
    assert asyncio.run(BS.BinanceService.get_klines_1m_recent(svc, "ABCUSDT", 90)) is None and len(svc.burst_exchange.calls) == n   # no read
    BS2, svc2 = _svc(_FakeEx(rows=[], weight="1499"))
    asyncio.run(BS.BinanceService.get_klines_1m_recent(svc2, "ABCUSDT", 90))
    assert svc2.burst_limited_until == 0.0


def test_burst_client_429_backoff_and_418_ban(monkeypatch):
    from ccxt.base.errors import RateLimitExceeded, DDoSProtection
    BS, svc = _svc(_FakeEx(err=RateLimitExceeded("429 Too many requests")))
    monkeypatch.setattr(BS, "_ban_until", 0)
    assert asyncio.run(BS.BinanceService.get_klines_1m_recent(svc, "ABCUSDT", 90)) is None
    assert svc.burst_limited_until >= time.time() + BS.BinanceService.BURST_LIMIT_BACKOFF_S - 5
    ban_ms = int((time.time() + 600) * 1000)
    BS, svc = _svc(_FakeEx(err=DDoSProtection(f"418 IP banned until {ban_ms}")))
    monkeypatch.setattr(BS, "_ban_persist_callback", None, raising=False)
    assert asyncio.run(BS.BinanceService.get_klines_1m_recent(svc, "ABCUSDT", 90)) is None
    assert BS._ban_until == pytest.approx(ban_ms / 1000) and BS.BinanceService.burst_limited(svc) is True


def test_burst_limited_reads_an_active_ban_only(monkeypatch):
    """619 review #2: an EXPIRED ban (_ban_until in the past, not yet cleared) no longer stops the burst reads — the EXH helpers' check."""
    BS, svc = _svc(_FakeEx())
    monkeypatch.setattr(BS, "_ban_until", time.time() - 10)
    assert BS.BinanceService.burst_limited(svc) is False
    monkeypatch.setattr(BS, "_ban_until", time.time() + 10)
    assert BS.BinanceService.burst_limited(svc) is True


def test_burst_client_in_the_ip_weight_view():
    import services.binance_service as BS
    svc = object.__new__(BS.BinanceService)
    svc.exchange = svc.public_exchange = svc.regime_exchange = None
    svc.burst_exchange = NS(last_response_headers={"x-mbx-used-weight-1m": "777"}, lastRestRequestTimestamp=time.time() * 1000)
    assert BS.BinanceService.ip_used_weight_1m(svc) == 777
