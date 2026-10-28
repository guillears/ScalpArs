"""🩳 FLATBASE_SHORT (DECISION_LOG 348 — operator discipline override): short a FRENZY episode whose coin was FLAT for the 14 days before
the spike, at the study's S4 bar (first 4-h close below the spike VWAP ≥ 4 h after the spike, 24 h volume ≥ $20M, ≤ 96 h), turnover > 1
→ trade · ≤ 1 → observe-only shadow · unknown mcap → trade; exit −8 % TP (resting maker) / +15 % stop (+ backstop) / 72 h; size = the
momentum-long mults; no slot of its own. Pure rules + the REAL engine paths on SQLite; every exchange read stubbed (no network)."""
import asyncio
import datetime as dt
import json
import os
import re
import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from services import flatbase as FB  # noqa: E402
from tests.test_frenzy_willy import _hold_db  # noqa: E402
from tests.test_willy_global_hold_switch import _offline  # noqa: E402

BAR, HOUR, DAY = 300_000, 3_600_000, 86_400_000
T0 = int(pd.Timestamp("2026-01-01").value // 10**6)        # a 4-hour boundary: a bar k CLOSES on one when (k + 1) % 48 == 0
PAIR = "FBSUSDT"


def _th(**kw):
    d = dict(flatbase_short_enabled=True, flatbase_flat_max=0.444, flatbase_min_vol_usd=20e6, flatbase_turnover_min=1.0, flatbase_tp_pct=8.0,
             flatbase_stop_pct=15.0, flatbase_max_hold_hours=72.0, flatbase_min_hours_after_spike=4.0, flatbase_max_hours_after_spike=96.0,
             flatbase_invest_mult=0.2, flatbase_lev_mult=0.25, resting_tp_flatbase_enabled=True, broker_backstop_pct=2.5)
    d.update(kw)
    return NS(**d)


def _series(hist_days=16, after=900, base_vol=1.2e5, pump_vol=3e7, decline_to=0.95, stay_above=False, early_dump=True, after_vol=2e5):
    """a full 5m grid: a FLAT history (±2 %), a spike bar `on` (+10 % on huge volume), a short pump, then a slide. Returns (rows, on)."""
    n_hist = hist_days * 288
    on = n_hist + ((24 - (n_hist + 1)) % 48)                 # (on + 1) % 48 == 24 → boundaries at on + 24 (too early), on + 72 (S4) …
    rows, k = [], 0
    for k in range(on):
        c = 1.0 + 0.02 * np.sin(k / 37.0)
        rows.append([T0 + k * BAR, c, c * 1.001, c * 0.999, c, base_vol])
    rows.append([T0 + on * BAR, 1.0, 1.12, 1.0, 1.10, pump_vol])
    pump = [1.15, 1.2, 1.18, 1.12, 1.08] if early_dump else [1.2, 1.3, 1.4, 1.5, 1.6]
    for j, c in enumerate(pump):
        rows.append([T0 + (on + 1 + j) * BAR, c, c * 1.01, c * 0.99, c, pump_vol / 3])
    last = pump[-1]
    for j in range(after):
        i = on + 6 + j
        c = (last * 1.0005 ** j) if stay_above else (last * (decline_to / last) ** (min(j, 300) / 300.0))
        rows.append([T0 + i * BAR, c, c * 1.002, c * 0.998, c, after_vol])
    return rows, on


def _window(rows, k, n=1499):
    return [list(r) for r in rows[max(0, k + 1 - n): k + 1]]


def _spike_ms(rows, on):
    return int(rows[on][0]) + BAR


# ── ① FLAT (14 days, 5m closes, ≥ 90 % present, read once) ─────────────────────────────────────────────────────────────────────────

def _flat_rows(n=4500, amp=0.10, drop=(), start=T0):
    return [[start + k * BAR, 1.0, 1.0, 1.0, 1.0 + amp * (k % 97) / 96.0, 1.0] for k in range(n) if k not in set(drop)]


def test_flat_label_coverage_and_young_listing():
    rows = _flat_rows()
    bc = int(rows[-1][0]) + BAR                               # the base bar = the last row
    rng, cov, n = FB.flat14_from_bars(rows, bc)
    assert n == 4032 and cov == 1.0 and abs(rng - 0.10) < 1e-9
    assert FB.flat_ok(rng, _th()) and not FB.flat_ok(0.4441, _th()) and FB.flat_ok(0.444, _th()) and not FB.flat_ok(None, _th())
    # 90 %: 403 missing bars inside the window (89.99 % present) → unreadable; 403 → 3,629 = 90.0 % → scored
    lo = len(rows) - 4032
    assert FB.flat14_from_bars(_flat_rows(drop=range(lo, lo + 404)), bc)[0] is None
    assert FB.flat14_from_bars(_flat_rows(drop=range(lo, lo + 403)), bc)[0] is not None
    # a listing younger than 14 days (history starts inside the window) → the study's NaN, whatever the coverage
    assert FB.flat14_from_bars(_flat_rows(n=4000), int(_flat_rows(n=4000)[-1][0]) + BAR)[0] is None
    # bars AFTER the base bar never count (the window ENDS at the base bar's close)
    late = rows + [[int(rows[-1][0]) + BAR, 1, 1, 1, 9.0, 1]]
    assert FB.flat14_from_bars(late, bc)[0] == rng
    assert FB.flat14_from_bars([[0, 1, 1, 1, -1.0, 1]] * 3, 10**9) == (None, None, 0)    # bad data → unreadable, never raises
    assert FB.base_close_ms(1_000_000_000_000) == 1_000_000_000_000 - 30 * 60_000


def test_15m_closes_deviation_documented():
    """the 15m alternative sees every 3rd 5m close: its range is NEVER wider, so it can only label MORE episodes FLAT — measured on the
    study's 6,903 episodes: 216 more FLAT, 0 fewer → the live rule reads the study's own 5m closes."""
    rows = _flat_rows(amp=0.0)
    k = next(k for k in range(len(rows) - 100, len(rows)) if (int(rows[k][0]) + BAR) % 900_000)
    rows[k][4] = 1.50                                        # one 5m close spike between two 15m closes → 50 % range on 5m
    bc = int(rows[-1][0]) + BAR
    r5 = FB.flat14_from_bars(rows, bc)[0]
    m15 = [[int(r[0]) - 2 * BAR, 1, 1, 1, r[4], 1] for r in rows if (int(r[0]) + BAR) % 900_000 == 0]
    r15 = FB.flat14_from_bars(m15, bc, tf_ms=900_000)[0]
    assert r5 == pytest.approx(0.5) and r15 <= r5 and FB.flat_ok(r15, _th()) and not FB.flat_ok(r5, _th())
    assert FB.FLAT_TF == "5m" and FB.FLAT_PAGES * FB.FLAT_PAGE_LIMIT >= FB.FLAT_EXPECTED == 4032


# ── ② S4 entry (pure) ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _ref_s4(rows, on, fs=None, th=None):
    """an INDEPENDENT reference of the study's cond4 (the sim's vectorised S4) on a full grid."""
    p = FB.flatbase_params(th or _th())
    t = np.array([r[0] for r in rows]); h = np.array([r[2] for r in rows]); l = np.array([r[3] for r in rows]); c = np.array([r[4] for r in rows])
    v = np.array([r[5] for r in rows]); tp = (h + l + c) / 3; q = v * tp
    vol24 = pd.Series(q).rolling(288, min_periods=288).sum().values
    w = np.arange(on, min(len(t) - 1, on + int(p['max_h'] * 12)) + 1)
    vw = np.cumsum(tp[w] * q[w]) / np.cumsum(q[w])
    ok = ((t[w] + BAR) % (4 * HOUR) == 0) & (w >= on + int(p['min_h'] * 12)) & (c[w] < vw) & (np.nan_to_num(vol24[w]) >= p['min_vol'])
    if fs is not None:
        ok &= w >= fs
    k = np.nonzero(ok)[0]
    return int(w[k[0]]) if len(k) else None


def test_s4_boundary_four_hours_vwap_and_first_bar_only():
    rows, on = _series()
    sp = _spike_ms(rows, on)
    ref = _ref_s4(rows, on)
    assert ref == on + 72                                     # on + 24 is a boundary below the VWAP but < 4 h after the spike
    for k in (on + 24, on + 50, on + 71):
        assert FB.s4_scan(_window(rows, k), sp, _th())['status'] == FB.ACT_WAIT, k
    s = FB.s4_scan(_window(rows, ref), sp, _th())
    assert s['status'] == FB.ACT_ENTER and s['bar_open'] == rows[ref][0] and FB.is_boundary_close(rows[ref][0])
    assert s['close'] < s['vwap'] and s['vs_vwap_pct'] < 0 and s['off_peak_pct'] < 0 and s['hours'] == pytest.approx(72 / 12)
    pk = max(r[2] for r in rows[on: ref + 1])
    assert s['base'] == pytest.approx(rows[on - 6][4]) and s['peak'] == pk and s['run_pct'] == pytest.approx((pk / rows[on - 6][4] - 1) * 100)
    k_pk = next(k for k in range(on, ref + 1) if rows[k][2] == pk)          # ⏱ (401) the FIRST row printing the peak
    assert s['peak_age_h'] == pytest.approx((rows[ref][0] - rows[k_pk][0]) / HOUR) and s['peak_age_h'] >= 0
    # the first qualifying boundary passed unjudged (outage / restart) → MISSED at the next one, never a later entry
    s2 = FB.s4_scan(_window(rows, ref + 48), sp, _th())
    assert s2['status'] == FB.ACT_MISSED and s2['idx'] is not None and s2['bar_open'] == rows[ref][0]
    # …unless the episode was first seen AFTER it (the study's first-seen gate): then the next boundary is the entry
    s3 = FB.s4_scan(_window(rows, ref + 48), sp, _th(), first_seen_open_ms=rows[ref + 1][0])
    assert s3['status'] == FB.ACT_ENTER and s3['bar_open'] == rows[ref + 48][0]
    # min hours is configurable (2 h → the on + 24 boundary qualifies)
    assert FB.s4_scan(_window(rows, on + 24), sp, _th(flatbase_min_hours_after_spike=1.0))['status'] == FB.ACT_ENTER


def test_s4_volume_floor_and_96h_window():
    rows, on = _series(base_vol=2e4, pump_vol=2e5, after_vol=2e4)   # 24 h volume far below $20M at every boundary
    sp = _spike_ms(rows, on)
    s = FB.s4_scan(_window(rows, on + 72), sp, _th())
    assert s['status'] == FB.ACT_WAIT and s['vol_low_last'] < 20e6 and _ref_s4(rows, on) is None
    assert FB.s4_scan(_window(rows, on + 72), sp, _th(flatbase_min_vol_usd=0))['status'] == FB.ACT_ENTER
    rows, on = _series(stay_above=True, early_dump=False, after=1300)   # never below the VWAP → expired after 96 h
    sp = _spike_ms(rows, on)
    assert FB.s4_scan(_window(rows, on + 1151), sp, _th())['status'] == FB.ACT_WAIT
    assert FB.s4_scan(_window(rows, on + 1152), sp, _th())['status'] == FB.ACT_EXPIRED
    assert FB.s4_scan(_window(rows, on + 10), sp + 999 * BAR, _th())['status'] == FB.ACT_NO_DATA   # spike not in the window: never guessed


def test_s4_gap_filled_like_the_study():
    rows, on = _series()
    sp = _spike_ms(rows, on)
    w = _window(rows, on + 72)
    gapped = [r for r in w if r[0] != rows[on + 30][0]]       # one missing bar: filled flat at the previous close, zero volume
    assert FB.s4_scan(gapped, sp, _th())['status'] == FB.ACT_ENTER


# ── ③ turnover / exits (pure) ──────────────────────────────────────────────────────────────────────────────────────────────────────

def test_turnover_route():
    th = _th()
    assert FB.turnover_route(50e6, 20e6, th) == (FB.ROUTE_TRADE, 2.5)
    assert FB.turnover_route(20e6, 20e6, th) == (FB.ROUTE_SHADOW, 1.0)                      # ≤ 1 → observe only
    assert FB.turnover_route(10e6, 20e6, th)[0] == FB.ROUTE_SHADOW
    assert FB.turnover_route(50e6, None, th) == (FB.ROUTE_TRADE, None)                      # unknown mcap → trade
    assert FB.turnover_route(50e6, 0, th) == (FB.ROUTE_TRADE, None)
    assert FB.turnover_route(None, 20e6, th)[0] == FB.ROUTE_SHADOW                          # known cap, unmeasured volume → never a real order


def test_exit_levels_are_price_moves():
    th = _th()
    assert FB.flatbase_exit_for(100.0, 92.0, 10, th) == (True, "FRENZY_TP", 8.0)            # −8 % price
    assert FB.flatbase_exit_for(100.0, 92.0, 10, th, resting_tp=True)[0] is False            # the resting limit owns the TP
    assert FB.flatbase_exit_for(100.0, 115.0, 10, th) == (True, "STOP_LOSS", -15.0)          # +15 % price
    assert FB.flatbase_exit_for(100.0, 115.0, 10, th, resting_tp=True)[1] == "STOP_LOSS"
    assert FB.flatbase_exit_for(100.0, 114.9, 10, th)[0] is False
    c, r, line = FB.flatbase_exit_for(100.0, 101.0, 72 * 60, th)
    assert (c, r) == (True, "MAX_HOLD_TIME") and line == pytest.approx(-1.0)
    assert FB.flatbase_exit_for(100.0, 101.0, 72 * 60 - 1, th)[0] is False
    assert FB.flatbase_exit_for(100.0, None, None, th)[0] is False                           # unreadable → no close (backstop / cap run)
    c, r, line = FB.flatbase_exit_for(100.0, None, 72 * 60, th)                               # unreadable price at the time cap: still closes,
    assert (c, r) == (True, "MAX_HOLD_TIME") and f"{line:.2f}" == "-15.00"                    # and the line is a number (the log format never raises)
    assert FB.net_of_funding_pct(1.0, -2.0, 2.0, 50.0) == (pytest.approx(-1.0), True)        # $2 paid on $100 notional = −2 pts
    assert FB.net_of_funding_pct(1.0, None, 2.0, 50.0) == (1.0, False)                         # paper: funding unknown
    assert FB.price_path_exit(100.0, [[0, 100, 101, 99, 100, 1]] * 864, th, slip_pct=0.1)['gross_pct'] == pytest.approx((1 - 100.1 / 99.9) * 100)
    assert FB.max_hold_minutes(th) == 4320
    # a bad hand-edited value never widens: the study's level is used
    assert FB.flatbase_params(_th(flatbase_stop_pct=0, flatbase_tp_pct=-1, flatbase_max_hours_after_spike=500))['stop'] == 10.0   # 🔁 (408) fallback = the live exit


def test_backstop_and_resting_tp_price():
    import services.trading_engine as TE
    from services.resting_tp import rtp_row_target, RTP_SLEEVES, RTP_ROW_STRATEGIES
    from services.exec_shadow import resting_tp_price
    import config
    th = config.trading_config.thresholds
    assert TE.bot_own_stop_pct(th, "FLATBASE_SHORT") == 10.0   # 🔁 DL 408 (+5 / −10; DL 348 shipped −15)
    assert TE.order_backstop_pct(th, "FLATBASE_SHORT", leverage=5) == 10.5                   # stop + 0.5, inside 92 % of liquidation at 5×
    assert TE.row_backstop_pct(th, {"entry_strategy": "FLATBASE_SHORT", "leverage": 5}) == 10.5   # the heal / re-price sweeps' per-row rule
    assert RTP_SLEEVES["FLATBASE_SHORT"] == "resting_tp_flatbase_enabled" and "FLATBASE_SHORT" in RTP_ROW_STRATEGIES
    row = NS(entry_strategy="FLATBASE_SHORT", entry_price=2.0, quantity=50.0, entry_fee=0.045, exit_override_at=None, direction="SHORT")
    net, why = rtp_row_target(th, row)
    assert why == "FRENZY_TP" and resting_tp_price("SHORT", 2.0, 0.045, net, config.trading_config.maker_fee, 0.0001) == pytest.approx(1.90)
    assert rtp_row_target(NS(**dict(th.model_dump(), resting_tp_flatbase_enabled=False)), row) is None   # own switch


def test_price_path_exit_matches_the_sim_order():
    th = _th()
    mk = lambda o, h, l, c: [0, o, h, l, c, 1]
    r = FB.price_path_exit(100.0, [mk(100, 101, 99, 100), mk(100, 116, 91, 100)], th)   # one bar touches both → STOP first (the sim)
    assert r['reason'] == "STOP_LOSS" and r['gross_pct'] == pytest.approx(-15.0)
    r = FB.price_path_exit(100.0, [mk(100, 101, 99, 100), mk(90, 91, 89, 90)], th)       # gap through the TP → filled at the open (better)
    assert r['reason'] == "FRENZY_TP" and r['gross_pct'] == pytest.approx(10.0)
    r = FB.price_path_exit(100.0, [mk(100, 101, 99, 99)] * 864, th)
    assert r['reason'] == "MAX_HOLD_TIME" and r['bars'] == 864 and r['gross_pct'] == pytest.approx(1.0)
    assert FB.price_path_exit(100.0, [mk(100, 101, 99, 99)] * 10, th)['reason'] == "OPEN"


# ── ④ the REAL engine: tracking, FLAT read, decisions, one per episode across a restart ────────────────────────────────────────────

def _engine(TE, th=None):
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True; e.is_running = True
    e.blocks = []
    e._record_filter_block = lambda n, d, had_room=True: e.blocks.append(n)
    return e


def _patch_th(monkeypatch, TE, **kw):
    monkeypatch.setattr(TE, "FLATBASE_PAGE_PACE_S", 0.0)                                       # the 2 s page pace (live) — instant in tests
    for k, v in dict(vars(_th()), **kw).items():
        monkeypatch.setattr(TE.config.trading_config.thresholds, k, v, raising=False)


def _ep(sp):
    return dict(spike_ts=sp, verified=True, hours=1.0)


def test_track_flat_read_and_restart_resume(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    _patch_th(monkeypatch, TE)
    rows, on = _series()
    sp = _spike_ms(rows, on)
    calls = []

    async def until(sym, tf, end, limit):
        calls.append((sym, tf, end, limit))
        return [r[:6] for r in rows if r[0] <= end][-limit:]
    monkeypatch.setattr(TE.binance_service, "get_ohlcv_until", until)
    from services import mcap_service
    monkeypatch.setattr(mcap_service, "get", lambda p: (None, None)); req = []
    monkeypatch.setattr(mcap_service, "request_pair", lambda p: req.append(p) or True)

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        e = _engine(TE)
        async with SL() as db:
            await e._flatbase_eval(db, PAIR, _ep(sp), _window(rows, on + 2), {"symbol": "FBS/USDT:USDT"}, rows[on + 2][0] + BAR)
            await asyncio.gather(*list(e.__dict__.get('_fbs_tasks', set())))
        async with SL() as s:
            led = (await s.execute(select(models.FlatbaseEpisode))).scalars().all()
        # restart: a new engine resumes the ledger row (first-seen + FLAT label) without a second history read
        e2 = _engine(TE)
        async with SL() as db:
            await e2._flatbase_eval(db, PAIR, None, _window(rows, on + 3), {"symbol": "FBS/USDT:USDT"}, rows[on + 3][0] + BAR)
        st2 = dict(e2._fbs_state())
        await eng.dispose()
        return e, led, st2
    e, led, st2 = asyncio.run(go())
    assert len(led) == 1 and led[0].status == "TRACKING" and led[0].flat14_pct == pytest.approx((1.02 / 0.98 - 1) * 100, abs=0.01) and led[0].flat_cov_pct == 100.0
    assert led[0].first_seen_at == dt.datetime.utcfromtimestamp(rows[on + 2][0] / 1000)
    assert len(calls) == 3 and all(c[1] == "5m" and c[3] == 1500 for c in calls) and calls[0][2] == FB.base_close_ms(sp) - 1   # 3 public pages
    assert req == [PAIR] and e.blocks == []
    rec = st2[(PAIR, sp)]
    assert rec['flat'] == pytest.approx(1.02 / 0.98 - 1, abs=1e-4) and rec['first_seen_ms'] == rows[on + 2][0] and len(calls) == 3


def _run_bars(monkeypatch, rows, on, ks, *, flat=0.04, mcap=10e6, open_ok=True, held=False, existing_order=False, th_kw=None, fs_k=None,
              status="TRACKING"):
    """drives the REAL _flatbase_eval on each bar index of ks (the live window ending there) with a pre-seeded TRACKING episode; the
    exchange / mcap / funding reads and open_position are stubbed. → (engine, ledger rows, opens, blocks)."""
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    _patch_th(monkeypatch, TE, **(th_kw or {}))
    sp = _spike_ms(rows, on)
    from services import mcap_service
    monkeypatch.setattr(mcap_service, "get", lambda p: (mcap, 1))

    async def _fetch_now(p, timeout_s=20.0):
        return (mcap, 1)
    monkeypatch.setattr(mcap_service, "fetch_pair_now", _fetch_now)

    async def fr(sym, max_age_s=None):
        return -0.0003
    monkeypatch.setattr(TE.binance_service, "fetch_funding_rate", fr)
    opens = []

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        async with SL() as s:
            row = models.FlatbaseEpisode(pair=PAIR, spike_at=dt.datetime.utcfromtimestamp(sp / 1000), is_paper=True, status=status,
                                         first_seen_at=dt.datetime.utcfromtimestamp(rows[fs_k if fs_k is not None else on][0] / 1000),
                                         flat14_pct=(flat * 100 if flat is not None else None), flat_cov_pct=100.0,
                                         flat_attempts=(3 if flat is None else 1))
            s.add(row)
            if held:
                s.add(models.Order(pair=PAIR, direction="LONG", status="OPEN", is_paper=True, entry_strategy="FRENZY_LONG", confidence="STRONG_BUY",
                                   entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0))
            if existing_order:
                s.add(models.Order(pair=PAIR, direction="SHORT", status="CLOSED", is_paper=True, entry_strategy="FLATBASE_SHORT", confidence="STRONG_BUY",
                                   entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0,
                                   entry_frenzy_spike_at=dt.datetime.utcfromtimestamp(sp / 1000)))
            await s.commit()
        e = _engine(TE)

        async def fake_open(db=None, **kw):
            opens.append(kw)
            return NS(id=77) if open_ok else None
        e.open_position = fake_open
        async with SL() as db:
            for k in ks:
                await e._flatbase_eval(db, PAIR, ({"spike_ts": sp, "verified": True, "hours": 1.0} if status != "TRACKING" else None),
                                       _window(rows, k), {"symbol": "FBS/USDT:USDT", "price": rows[k][4]}, rows[k][0] + BAR)
                await e._flatbase_enter_queued(db)                      # the pass judges S4 entries after its per-pair loop
        async with SL() as s:
            led = (await s.execute(select(models.FlatbaseEpisode))).scalars().all()
        await eng.dispose()
        return e, led
    e, led = asyncio.run(go())
    return e, led, opens, e.blocks


def test_engine_enters_on_the_s4_bar_turnover_above_one():
    rows, on = _series()
    import pytest as _p
    mp = _p.MonkeyPatch()
    try:
        e, led, opens, blocks = _run_bars(mp, rows, on, range(on + 1, on + 73), mcap=10e6)   # vol24 ≈ $90M+ / $10M → turnover > 1
    finally:
        mp.undo()
    assert len(opens) == 1 and led[0].status == "OPENED" and led[0].order_id == 77
    o = opens[0]
    assert o['direction'] == "SHORT" and o['flatbase_short'] is True and o['confidence'] == "STRONG_BUY"
    assert o['flatbase_bar_close_ms'] == rows[on + 72][0] + BAR and o['current_price'] == rows[on + 72][4]
    assert o['entry_frenzy_spike_at'] == dt.datetime.utcfromtimestamp(_spike_ms(rows, on) / 1000) and o['entry_funding_rate'] == -0.0003
    assert o['entry_flatbase_flat14_pct'] == pytest.approx(4.0) and o['entry_flatbase_off_peak_pct'] < 0 and o['entry_frenzy_vs_vwap_pct'] < 0
    pk = max(r[2] for r in rows[on: on + 73])
    k_pk = next(k for k in range(on, on + 73) if rows[k][2] == pk)
    assert o['entry_flatbase_peak_age_h'] == pytest.approx((rows[on + 72][0] - rows[k_pk][0]) / HOUR, abs=1e-3)   # ⏱ (401)
    assert led[0].peak_age_h == o['entry_flatbase_peak_age_h']
    assert o['entry_pair_volume_24h_usd'] > 20e6 and led[0].turnover > 1 and led[0].funding_rate == -0.0003 and blocks == []


def test_engine_turnover_at_or_below_one_is_shadow_only(monkeypatch):
    rows, on = _series()
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, [on + 72], mcap=1e12)
    assert opens == [] and led[0].status == "SHADOW" and led[0].turnover < 1 and led[0].entry_price == rows[on + 72][4]
    assert led[0].entry_bar_at == dt.datetime.utcfromtimestamp(rows[on + 72][0] / 1000) and led[0].mcap_usd == 1e12
    assert led[0].vwap is not None and led[0].off_peak_pct < 0 and led[0].vol24_usd > 20e6 and blocks == ["FLATBASE_TURNOVER_SHADOW"]


def test_engine_unknown_mcap_trades(monkeypatch):
    rows, on = _series()
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, [on + 72], mcap=None)
    assert len(opens) == 1 and led[0].status == "OPENED" and led[0].turnover is None and led[0].mcap_usd is None


def test_engine_pair_held_refused(monkeypatch):
    rows, on = _series()
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, [on + 72, on + 120], held=True)
    assert opens == [] and led[0].status == "REFUSED" and "pair held" in led[0].reason and blocks == ["FLATBASE_PAIR_HELD"]


def test_engine_volume_floor_no_entry(monkeypatch):
    rows, on = _series(base_vol=2e4, pump_vol=2e5, after_vol=2e4)
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, [on + 72, on + 120])
    assert opens == [] and led[0].status == "TRACKING" and blocks == ["FLATBASE_VOL24_LOW", "FLATBASE_VOL24_LOW"]


def test_engine_missed_boundary_never_caught_up(monkeypatch):
    rows, on = _series()
    # the process was down over the S4 bar (on + 72) and judges again from on + 80: the next boundary (on + 120) is MISSED, no order
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, range(on + 80, on + 200))
    assert opens == [] and led[0].status == "MISSED" and blocks == ["FLATBASE_MISSED_BOUNDARY"]


def test_engine_one_per_episode_order_on_record(monkeypatch):
    rows, on = _series()
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, [on + 72], existing_order=True)
    assert opens == [] and led[0].status == "OPENED" and "already on record" in led[0].reason


def test_engine_flat_unknown_or_not_flat_fail_closed(monkeypatch):
    rows, on = _series()
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, [on + 72], flat=None)
    assert opens == [] and led[0].status == "UNREADABLE" and blocks == ["FLATBASE_FLAT_UNREADABLE"]


def test_engine_open_refused_ends_the_episode(monkeypatch):
    rows, on = _series()
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, [on + 72, on + 120], open_ok=False)
    assert len(opens) == 1 and led[0].status == "REFUSED" and blocks == ["FLATBASE_OPEN_REFUSED"]


def test_flat_read_unreadable_and_not_flat(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    _patch_th(monkeypatch, TE)
    rows, on = _series()
    sp = _spike_ms(rows, on)
    from services import mcap_service
    monkeypatch.setattr(mcap_service, "get", lambda p: (1.0, 1))

    def run(until):
        monkeypatch.setattr(TE.binance_service, "get_ohlcv_until", until)

        async def go():
            eng, SL = await _hold_db(monkeypatch)()
            e = _engine(TE)
            async with SL() as db:
                for k in (on + 1, on + 2, on + 3, on + 4):
                    await e._flatbase_eval(db, PAIR, _ep(sp), _window(rows, k), {"symbol": "FBS/USDT:USDT"}, rows[k][0] + BAR)
                    await asyncio.gather(*list(e.__dict__.get('_fbs_tasks', set())))
            async with SL() as s:
                led = (await s.execute(select(models.FlatbaseEpisode))).scalars().all()
            await eng.dispose()
            return e.blocks, led
        return asyncio.run(go())

    async def fail(*a, **k):
        return None
    blocks, led = run(fail)
    assert led[0].status == "UNREADABLE" and led[0].flat_attempts == FB.FLAT_MAX_ATTEMPTS and blocks == ["FLATBASE_FLAT_UNREADABLE"]
    wild = [list(r) for r in rows]
    for r in wild[on - 1000: on - 900]:
        r[4] = 3.0                                             # a 200 % range in the 14 days before the base

    async def wild_until(sym, tf, end, limit):
        return [r[:6] for r in wild if r[0] <= end][-limit:]
    blocks, led = run(wild_until)
    assert led[0].status == "NOT_FLAT" and led[0].flat14_pct == pytest.approx((3.0 / 0.98 - 1) * 100, abs=0.01) and blocks == ["FLATBASE_NOT_FLAT"]


# ── ⑤ parity: the live entry (engine _flatbase_eval, bar by bar on the 1,499-bar window) == the study sim's S4 on the same series ─────

def _sim_s4(monkeypatch, rows, on):
    import study_flatbase_short_common as C
    import study_flatbase_short_sim as SIM
    t = np.array([r[0] for r in rows], dtype="int64"); o = np.array([r[1] for r in rows]); h = np.array([r[2] for r in rows])
    l = np.array([r[3] for r in rows]); c = np.array([r[4] for r in rows]); v = np.array([r[5] for r in rows])
    qv = v * (h + l + c) / 3.0                                 # the exchange quote volume ≡ volume × typical price on this series
    monkeypatch.setattr(C, "load_full", lambda pair: (t, o, h, l, c, v, qv, np.ones(len(t), bool)))
    monkeypatch.setattr(C, "funding_table", lambda pair: None)
    eps = pd.DataFrame([dict(on_t=int(t[on]), off_t=int(t[on + 300]), first_seen_t=int(t[on]), new_ep_at_end=False, base=float(c[on - 6]))])
    sigs, _, _ = SIM.worker((PAIR, eps, int(t[-1]) + BAR))
    s4 = [s for s in sigs if s['entry'] == "S4"]
    return (int((s4[0]['i_t'] - t[0]) // BAR) if s4 else None)


@pytest.mark.parametrize("kw", [dict(), dict(early_dump=False, decline_to=0.9), dict(base_vol=2e4, pump_vol=2e5, after_vol=2e4, after=1300), dict(base_vol=2e4, pump_vol=2e5),
                                dict(stay_above=True, early_dump=False, after=1300)])
def test_parity_live_entry_vs_study_sim_s4(monkeypatch, kw):
    rows, on = _series(**kw)
    sim_k = _sim_s4(monkeypatch, rows, on)
    assert sim_k == _ref_s4(rows, on)                          # the independent reference agrees with the study
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, range(on, min(len(rows), on + 1160)), mcap=None)
    live_k = next((k for k in range(len(rows)) if opens and rows[k][0] + BAR == opens[0]['flatbase_bar_close_ms']), None)
    assert live_k == sim_k, (kw, live_k, sim_k)
    assert led[0].status == ("OPENED" if sim_k is not None else "EXPIRED")


# ── ⑥ open_position: no slot, the momentum-long size, the price stop + backstop record (REAL open path, paper) ─────────────────────

def _e2e_open(monkeypatch, n_open=4, seed_strategy="MOMENTUM", th_kw=None, **kw):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    monkeypatch.setattr(TE.TradingEngine, "_willy_global_hold_on", lambda self: False)
    monkeypatch.setattr(TE.TradingEngine, "_rtp_tick_cached", lambda self, pair, px: 0.0001)   # the exchange tick (market metadata, stubbed)
    net = _offline(monkeypatch, TE)
    _patch_th(monkeypatch, TE, **(th_kw or {}))
    monkeypatch.setattr(TE.config.trading_config.thresholds, "broker_backstop_enabled", True)
    monkeypatch.setattr(TE.config.trading_config.investment, "max_open_positions", 4)
    monkeypatch.setattr(TE.config.trading_config.investment, "redeploy_leftover_enabled", False, raising=False)

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        async with SL() as s:
            for i in range(n_open):
                s.add(models.Order(pair=f"P{i}USDT", direction="LONG", status="OPEN", is_paper=True, entry_strategy=seed_strategy, confidence="STRONG_BUY",
                                   entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0))
            await s.commit()
        e = TE.TradingEngine(); e.is_running = True; e.is_paper_mode = True
        blocks = []
        _orig = e._record_filter_block
        e._record_filter_block = lambda n, d, had_room=True: (blocks.append(n), _orig(n, d, had_room))

        async def bal(db):
            return 5000.0
        e.get_available_balance = bal
        async with SL() as db:
            r = await e.open_position(db=db, pair=PAIR, direction="SHORT", confidence="STRONG_BUY", current_price=1.0,
                                      entry_pair_volume_24h_usd=5e9, **kw)
            n_slot = (await db.execute(select(TE.func.count(models.Order.id)).where(TE.and_(models.Order.status == "OPEN", TE._bot_slot_filter())))).scalar()
            n_bot = (await db.execute(select(TE.func.count(models.Order.id)).where(TE.and_(models.Order.status == "OPEN", TE._bot_open_filter())))).scalar()
            ref = e.calculate_position_size(5000.0, "STRONG_BUY", total_portfolio=5000.0 + n_open, base_lev_multiplier=0.25)
            ref_m = e.momentum_long_resize("X", ref[0], ref[1], 5000.0, "STRONG_BUY", 5000.0 + n_open, 1.0, 1.0, "investment", "MOMENTUM", "LONG", 0.25)
        await eng.dispose()
        return r, blocks, n_slot, n_bot, ref, ref_m
    out = asyncio.run(go())
    assert net == []
    return out


def test_open_no_slot_momentum_size_stop_and_backstop(monkeypatch):
    now_ms = int(__import__("time").time() * 1000)
    r, blocks, n_slot, n_bot, ref, ref_m = _e2e_open(monkeypatch, flatbase_short=True, flatbase_bar_close_ms=now_ms,
                                                    entry_frenzy_spike_at=dt.datetime(2026, 10, 22, 8, 0), entry_flatbase_flat14_pct=31.5,
                                                    entry_flatbase_flat_cov_pct=100.0, entry_flatbase_off_peak_pct=-22.0,
                                                    entry_flatbase_peak_age_h=7.5)
    assert r.entry_flatbase_peak_age_h == 7.5                                              # ⏱ (401) the real Order carries the stamp
    assert r is not None and r.entry_strategy == "FLATBASE_SHORT" and r.direction == "SHORT"   # the book was FULL (4/4): no slot needed
    assert n_slot == 4 and n_bot == 5                                                       # never counted in the bot's slots (fee burn keeps it)
    assert r.leverage == ref[1] == 5 and r.investment == pytest.approx(ref_m[0], rel=1e-6)     # = the momentum-long size (0.2 margin at 5×)
    assert r.notional_value == pytest.approx(ref_m[0] * 5, rel=1e-6)
    assert r.cell_multiplier_source == "FLATBASE_SHORT" and r.entry_frenzy_stop_pct == 15.0 and r.backstop_pct == 15.5
    assert r.backstop_trigger_price == pytest.approx(r.entry_price * 1.155)
    assert r.entry_flatbase_flat14_pct == 31.5 and r.entry_flatbase_off_peak_pct == -22.0 and r.entry_frenzy_spike_at == dt.datetime(2026, 10, 22, 8, 0)
    assert r.resting_tp_price == pytest.approx(r.entry_price * 0.92, rel=2e-3)                 # paper records the resting TP live would hold
    assert "PAIR_HELD" not in blocks


def test_open_refused_late_and_manual_block(monkeypatch):
    late = int(__import__("time").time() * 1000) - 301_000
    r, blocks, *_ = _e2e_open(monkeypatch, n_open=0, flatbase_short=True, flatbase_bar_close_ms=late)
    assert r is None and "FLATBASE_LATE" in blocks


def test_slot_filter_sql_and_momentum_still_capped():
    import services.trading_engine as TE
    s = str(TE._bot_slot_filter().compile(compile_kwargs={"literal_binds": True}))
    assert "FLATBASE_SHORT" in s and "MANUAL" in s
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng.count("_bot_slot_filter())") == 3                                            # open_position + both scan had-room counts
    assert "if _open_count_now >= _eff_max_pos and not _fbs:" in eng


# ── ⑦ exits through the REAL realtime path · candle path / restart / heal wiring ─────────────────────────────────────────────────────

@pytest.mark.parametrize("price,held_min,want", [(0.919, 10, "FRENZY_TP"), (1.151, 10, "STOP_LOSS"), (1.01, 72 * 60 + 1, "MAX_HOLD_TIME"), (1.05, 10, None)])
def test_realtime_exit_path(monkeypatch, price, held_min, want):
    import services.trading_engine as TE
    _patch_th(monkeypatch, TE, resting_tp_flatbase_enabled=False)
    closed = []

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        import models
        opened = dt.datetime.utcnow() - dt.timedelta(minutes=held_min)
        async with SL() as s:
            o = models.Order(pair=PAIR, direction="SHORT", status="OPEN", is_paper=True, entry_strategy="FLATBASE_SHORT", confidence="STRONG_BUY",
                             entry_price=1.0, quantity=100.0, investment=20.0, leverage=5, notional_value=100.0, entry_fee=0.045, opened_at=opened)
            s.add(o); await s.commit(); oid = o.id
        e = _engine(TE)

        async def cp(db, order, px, reason):
            closed.append(reason); order.status = "CLOSED"; return order
        e.close_position = cp
        TE._open_orders_cache[PAIR] = [dict(id=oid, direction="SHORT", entry_price=1.0, quantity=100.0, entry_fee=0.045, stop_loss=-15.0,
                                            entry_strategy="FLATBASE_SHORT", opened_at=opened, peak_pnl=0.0, trough_pnl=0.0, confidence="STRONG_BUY")]
        await e.check_realtime_stop_loss(PAIR, price)
        await eng.dispose()
    asyncio.run(go())
    assert closed == ([want] if want else [])


def test_exit_and_protection_sites_wired():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    # candle path: own 72 h cap (never the global 20 h), own exit branch inside the sleeve intercept, resting-TP gate
    i = eng.index("            elif (order.entry_strategy or \"\") in FLATBASE_CLASS:   # 🩳 (348) its OWN 72 h time exit")
    assert "max_hold = flatbase_max_hold_minutes(config.trading_config.thresholds)" in eng[i:i + 300]
    assert eng.count("+ FRENZY_STRATEGIES + FLATBASE_CLASS:") == 2                          # candle + realtime intercepts
    assert eng.count("in FRENZY_STRATEGIES + FLATBASE_CLASS) and self._rtp_state(") == 2    # the resting-TP safety net on both paths
    assert eng.count("_br_close, _br_reason, _br_stop = flatbase_exit_for(") == 2
    assert eng.count("in FRENZY_STRATEGIES + FLATBASE_CLASS or reason.startswith(RH_STOP_CLASS)") == 2   # every FLATBASE close is taker
    # cached stops (open + every cache rebuild after a restart), resting TP armed at the open, backstop per order at open
    assert "else -flatbase_params(config.trading_config.thresholds)['stop'] if _fbs" in eng
    assert "else -flatbase_params(config.trading_config.thresholds)['stop'] if (order.entry_strategy or '') in FLATBASE_CLASS" in eng
    assert "if (_frenzy and (_fz_es in RTP_SLEEVES)) or _fbs:" in eng
    assert eng.count("(_fz_es if _frenzy else (FLATBASE_SHORT if _fbs else None))") == 2
    # restart re-arm / heal / re-price / reconcile read the ROW: row_backstop_pct + rtp_row_target cover FLATBASE (tested above)
    assert "_flatbase_follow_seed(db)" in eng and "await self._flatbase_sweep(bar_open, seen | _fbs_seen)" in eng
    p = eng.index("    async def _update_frenzy_pass(self, db):"); body = eng[p:eng.index("\n    def _frenzy_wide_off_note(", p)]
    assert "if not (_on or _obs or _wide or _lite or _willy or _fbs):" in body
    # nothing FLATBASE runs on the FRENZY decision path: the loop only QUEUES the pair; tracking / reads / S4 / entries run after it
    assert "self.__dict__.setdefault('_fbs_eval_q', []).append((pair, ep, closed, by[pair], bar_open))" in body and "self._flatbase_eval(" not in body
    assert body.index("await self._frenzy_willy_eval(") < body.index("await self._flatbase_run_queue(db)") < body.index("await self._flatbase_enter_queued(db)") \
        < body.index("await self._flatbase_sweep(bar_open, seen | _fbs_seen)")
    assert "self._fbs_eval_q = []; self._fbs_enter_q = []" in body and "FLATBASE_EXTRA_READS_MAX" in body          # stale work cleared · extra reads capped
    assert "es in (FRENZY_LITE_RETRY, FRENZY_WILLY_HT) or es in FLATBASE_CLASS:" in eng                            # EMA13-cross exit excluded too
    import services.trading_engine as TE
    assert TE._ema13_cross_exit_applies("FLATBASE_SHORT") is False and TE.FLATBASE_EXTRA_READS_MAX == 20


# ── ⑧ config / UI / exports parity (D11 / D12) ───────────────────────────────────────────────────────────────────────────────────────

FIELDS = {"flatbase_max_open": 10, "flatbase_short_enabled": True, "flatbase_flat_max": 0.444, "flatbase_min_vol_usd": 20e6, "flatbase_turnover_min": 1.0,
          "flatbase_tp_pct": 5.0, "flatbase_stop_pct": 10.0, "flatbase_max_hold_hours": 72.0, "flatbase_min_hours_after_spike": 4.0,
          "flatbase_max_hours_after_spike": 96.0, "flatbase_invest_mult": 0.2, "flatbase_lev_mult": 0.25, "resting_tp_flatbase_enabled": True}


def test_config_defaults_json_ui_and_exports():
    import config as C
    mf = type(C.trading_config.thresholds).model_fields
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for k, v in FIELDS.items():
        assert mf[k].default == v and cfg[k] == v, k
    assert cfg["momentum_long_invest_mult"] == cfg["flatbase_invest_mult"] and cfg["momentum_long_lev_mult"] == cfg["flatbase_lev_mult"]
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 348" in src and "DISCIPLINE OVERRIDE" in src
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    ids = ["config-fbs-enabled", "config-rtp-flatbase-enabled", "config-fbs-flat-max", "config-fbs-min-vol", "config-fbs-turnover-min", "config-fbs-tp",
           "config-fbs-stop", "config-fbs-max-hold", "config-fbs-min-hours", "config-fbs-max-hours", "config-fbs-invest-mult", "config-fbs-lev-mult",
           "config-fbs-max-open"]
    for i in ids:
        assert html.count(f'id="{i}"') == 1, i                                               # one input each, in the FLATBASE box
    for i in ids[2:]:
        assert f"['{i}', 'flatbase_" in html, i                                                 # FLATBASE_NUM_FIELDS = load AND save
    assert "for (const [_id, _key, _def] of FLATBASE_NUM_FIELDS)" in html and "...Object.fromEntries(FLATBASE_NUM_FIELDS.map(" in html
    assert "config.thresholds.flatbase_short_enabled !== false" in html and "flatbase_short_enabled: document.getElementById('config-fbs-enabled')" in html
    assert "config.thresholds.resting_tp_flatbase_enabled !== false" in html and "resting_tp_flatbase_enabled: document.getElementById('config-rtp-flatbase-enabled')" in html
    a = html.index("<!-- ===== FRENZY SLEEVE BOX END ===== -->"); b = html.index("<!-- ===== 🩳 FLATBASE_SHORT BOX")
    assert 0 < b - a < 100                                                                     # its own box, right after FRENZY
    assert '<option value="FLATBASE_SHORT">' in html and 'data-ss="fbs"' in html and "S.fbs = {" in html
    assert html.count("_stratRaw === 'FLATBASE_SHORT' ? 'FLATBASE-S'") == 2 and "(o.entry_strategy || '') === 'FLATBASE_SHORT'" in html
    # both text exports: the config line (shared _buildConfigLines, called by both) + the FRENZY monitor line + the FRENZY Fills rows
    assert html.count("..._buildConfigLines(cfg, changelog, hr, hr2, status)") == 2 and "🩳 FLATBASE_SHORT (DECISION_LOG 348, discipline override)" in html
    assert html.count("...frenzyReportLines(perf, hr2)") == 2 and "+ flatbaseMonitorStr(m);" in html
    assert html.count("FRENZY Fills (FRENZY_LONG · FRENZY_WIDE · FRENZY_LITE · FRENZY_WILLY) · 🔁 FRENZY_LITE_RETRY") == 2   # UI + the shared export block
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '("FLATBASE_SHORT", "🩳 FLATBASE_SHORT (' in main and 'if _fz_es == "FLATBASE_SHORT":' in main      # FRENZY Fills row + gate
    assert "_fb_cfg_errors(new_config.thresholds, _sb_lev)" in main and "'FLATBASE_SHORT': ('flatbase_short_enabled',)" in main
    assert 'e="FLATBASE"' in main and "fb_status" in main                                       # decisions export rows (the shadow cohort)


def test_config_errors():
    import config
    th = config.trading_config.thresholds
    assert FB.flatbase_config_errors(th, 20.0) == []
    bad = NS(**dict(th.model_dump(), flatbase_tp_pct=0, flatbase_max_hours_after_spike=150, flatbase_short_enabled="yes"))
    errs = " | ".join(FB.flatbase_config_errors(bad, 20.0))
    assert "flatbase_tp_pct" in errs and "flatbase_max_hours_after_spike" in errs and "flatbase_short_enabled" in errs
    wide = NS(**dict(th.model_dump(), flatbase_lev_mult=1.0))                                   # 20×: the backstop cannot sit beyond +15 %
    assert "too wide" in " ".join(FB.flatbase_config_errors(wide, 20.0))

    import main
    from fastapi import HTTPException
    saved = []
    mp = pytest.MonkeyPatch()
    mp.setattr(main, "save_trading_config", lambda c: saved.append(c) or True)
    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(main.update_config(main.ConfigUpdate(thresholds={"flatbase_stop_pct": -1})))
    finally:
        mp.undo()
    assert ei.value.status_code == 400 and "flatbase_stop_pct" in ei.value.detail and saved == []


def test_sleeve_rows_and_labels():
    import main as M
    o = NS(entry_strategy="FLATBASE_SHORT", direction="SHORT", pnl_percentage=1.5, pnl=3.0, opened_at=dt.datetime(2026, 10, 22, 8),
           closed_at=dt.datetime(2026, 10, 22, 9), investment=60.0, notional_value=300.0, leverage=5, peak_pnl=2.0, import_source=None)
    rows = M._compute_sleeve_performance([o], start_balance=1000.0)
    assert any(r.get('sleeve') == 'Flatbase-Short' for r in rows)
    st = M._compute_strategy_performance([o])
    assert st[0]["strategy"].startswith("FLATBASE_SHORT")
    assert M._SLEEVE_LABELS[("FLATBASE_SHORT", "SHORT")] == "FLATBASE Short"
    assert M._sleeve_enabled(NS(flatbase_short_enabled=False), "FLATBASE_SHORT") is False


# ── ⑨ master pool / screen / archive / scout classification ─────────────────────────────────────────────────────────────────────

def test_master_screen_archive_classification():
    import build_master_pool as BMP
    import screen_pool as SP
    import archive_batch as AB
    assert BMP.stack_pair_blacklist_reason("FLATBASE_SHORT", "BTCUSDT") == "PAIR_NO_TRADE"          # the FRENZY pass's universe
    assert BMP.stack_pair_blacklist_reason("FLATBASE_SHORT", "FBSUSDT") == ""
    src = open(os.path.join(ROOT, "scripts", "build_master_pool.py"), encoding="utf-8").read()
    assert "elif strat == 'FLATBASE_SHORT':\n                why = 'FLATBASE_SLEEVE'" in src and 'if strat == "FLATBASE_SHORT":' in src
    row = dict(status="CLOSED", pair="FBSUSDT", entry_strategy="FLATBASE_SHORT", direction="SHORT", cell_multiplier_source="FLATBASE_SHORT")
    assert SP.sleeve(row) is None                                                                   # never MOM_SHORT / FLIP
    d = pd.DataFrame([dict(entry_strategy="FLATBASE_SHORT", direction="SHORT"), dict(entry_strategy="MOMENTUM", direction="SHORT")])
    assert list(AB._sleeve(d)) == ["FLATBASE_SHORT", "MOM-short"] and not AB._probe(d).any()    # full size: its own headline row


def test_scout_gates_and_shadow_pricing(tmp_path):
    import scout_flatbase_short as SC
    t0 = dt.datetime(2026, 10, 23, 0, 0)
    orders = tmp_path / "scalpars_orders_paper_x.csv"
    rows = []
    for i in range(12):
        rows.append(dict(opened_at=(t0 + dt.timedelta(hours=8 * i)).isoformat(), closed_at=(t0 + dt.timedelta(hours=8 * i + 4)).isoformat(),
                         pair=f"P{i}USDT", direction="SHORT", status="CLOSED", entry_strategy="FLATBASE_SHORT", pnl=-1.0,
                         pnl_percentage=(-15.0 if i < 3 else 1.0), close_reason=("STOP_LOSS" if i < 3 else "FRENZY_TP"), is_paper=True))
    rows.append(dict(opened_at=(t0 + dt.timedelta(hours=1)).isoformat(), closed_at=None, pair="OPUSDT", direction="SHORT", status="CLOSED",
                     entry_strategy="FLATBASE_SHORT", pnl=5.0, pnl_percentage=5.0, close_reason="MANUAL", is_paper=True))   # operator-closed: out
    pd.DataFrame(rows).to_csv(orders, index=False)
    dec = tmp_path / "scalpars_decisions_paper_x.csv"
    pd.DataFrame([dict(e="FLATBASE", pair="SHUSDT", price=1.0, fb_spike_at="2026-10-23T02:00:00", fb_status="SHADOW",
                       fb_entry_bar_at="2026-10-23T07:55:00", fb_turnover=0.4),
                  dict(e="FLATBASE", pair="SHUSDT", price=1.0, fb_spike_at="2026-10-23T02:00:00", fb_status="SHADOW",
                       fb_entry_bar_at="2026-10-23T07:55:00", fb_turnover=0.4),                          # duplicate across exports
                  dict(e="FLATBASE", pair="XXUSDT", price=1.0, fb_spike_at="2026-10-23T02:00:00", fb_status="OPENED",
                       fb_entry_bar_at="2026-10-23T07:55:00")]).to_csv(dec, index=False)
    got = []

    def kl(sym, start, limit):
        got.append((sym, start, limit))
        return [[start, 1.0, 1.0, 0.91, 0.91, 1.0]]                                                  # the TP touches on the first bar
    since = int(pd.Timestamp("2026-10-22").value // 10**6)
    fund = []
    L = SC.run(now_ms=since + 30 * 86_400_000, paths=[str(orders)], dec_paths=[str(dec)], since_ms=since, klines=kl,
               funding=lambda sym, a, b: fund.append((sym, a, b)) or -0.5,
               tp_klines=lambda *a: [], fbx_state_path=str(tmp_path / "fbx.json"), tp_state_path=str(tmp_path / "tp.json"), tp_final_path=str(tmp_path / "fin.json"),
                obs_fb2_state_path=str(tmp_path / "fb2.json"))   # (367) own path
    txt = "\n".join(L)
    assert "FBX — +5 / −10 SWITCH-BACK GATE" in txt                                                         # 🔁 (408) the section runs inside run()
    assert "real (net of funding)" in txt and "funding unknown on 12 fill(s)" in txt                       # paper rows: no funding column
    assert "slippage 0.030 %/taker side, funding incl." in txt and fund and fund[0][0] == "SHUSDT"
    # a live fill's funding is netted: −$2 paid on a $100 notional turns +1.0 % into −1.0 %
    live = tmp_path / "scalpars_orders_live_y.csv"
    pd.DataFrame([dict(rows[3], opened_at=(t0 + dt.timedelta(days=9)).isoformat(), pair="LVUSDT", pnl_percentage=1.0, funding_fee_usd=-2.0,
                       entry_price=2.0, quantity=50.0, is_paper=False)]).to_csv(live, index=False)
    a = SC.load_orders([str(live)])
    assert a.net.iloc[0] == pytest.approx(-1.0) and bool(a.fund_known.iloc[0])
    assert "operator-closed: 1" in txt and "N 12" in txt and "3 stop-outs in a row" in txt and "Gate verdict: REVIEW" in txt
    assert set(got) == {("SHUSDT", int(pd.Timestamp("2026-10-23T08:00:00").value // 10**6), 864)}      # the bar after the S4 bar, 72 h (also OBS-FB2's two cells, DL 407)
    assert "1 setups · 1 priced" in txt and "⏳ 12/30 real fills" in txt
    assert SC.run(paths=[str(orders)], dec_paths=[str(dec)], since_ms=None, klines=kl, fbx_state_path=str(tmp_path / "fbx.json"), tp_state_path=None, tp_final_path=None) is not None


def test_gate_rules():
    t = lambda d, h=0: dt.datetime(2026, 10, 23) + dt.timedelta(days=d, hours=h)
    v, txt = FB.flatbase_gate([(t(0), 1.0, "FRENZY_TP")] * 5)
    assert v == "COLLECTING"
    v, _ = FB.flatbase_gate([(t(0, i), -11.0, "MAX_HOLD_TIME") for i in range(3)])                      # Σ −33 ≤ −30
    assert v == "REVIEW"
    v, _ = FB.flatbase_gate([(t(0, i), -1.0, "MAX_HOLD_TIME") for i in range(10)])                      # first 10 avg < 0
    assert v == "REVIEW"
    main = [(t(i // 3, i % 3), 1.0 if i % 5 else -1.0, "FRENZY_TP") for i in range(30)]                 # 30 fills on 10 days, avg +0.6
    assert FB.flatbase_gate(main)[0] == "STEP_UP"
    neg = [(t(i // 3, i % 3), -0.2 if i % 2 else 0.1, "MAX_HOLD_TIME") for i in range(30)]
    assert FB.flatbase_gate(neg)[0] in ("OFF", "REVIEW")
    mid = [(t(i // 3, i % 3), 0.2, "MAX_HOLD_TIME") for i in range(30)]
    v, txt = FB.flatbase_gate(mid)
    assert v == "REREAD" and "re-read at 60" in txt
    assert FB.flatbase_observe_gate([1.0] * 29, [5.0])[0] == "COLLECTING"
    assert FB.flatbase_observe_gate([1.0] * 30, [2.0, 0.5])[0] == "REVIEW"
    assert FB.flatbase_observe_gate([1.0] * 30, [0.5])[0] == "KEEP_OUT"


def test_reset_never_reopens_a_decided_episode(monkeypatch):
    """a reset (ALL / SHORT) keeps every episode whose S4 window may still be open — a decided one stays decided (no second entry), only
    old rows go; a LONG reset never touches the ledger (mirrors WILLY's kept seen-episode map)."""
    import models, main
    from sqlalchemy import select
    rows, on = _series()
    main_src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    w = main_src[main_src.index("async def _willy_reset_state("):main_src.index("\n\n\n", main_src.index("async def _willy_reset_state("))]
    assert "FlatbaseEpisode" not in w and main_src.count("await _flatbase_reset_state(db, is_paper)") == 2   # ALL + SHORT only
    now = dt.datetime.utcnow().replace(microsecond=0)

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        async with SL() as s:
            s.add(models.FlatbaseEpisode(pair="NEWUSDT", spike_at=now - dt.timedelta(hours=10), is_paper=True, status="SHADOW",
                                         first_seen_at=now - dt.timedelta(hours=10)))
            s.add(models.FlatbaseEpisode(pair="OLDUSDT", spike_at=now - dt.timedelta(hours=200), is_paper=True, status="OPENED",
                                         first_seen_at=now - dt.timedelta(hours=200)))
            await s.commit()
        async with SL() as s:
            await main._flatbase_reset_state(s, True); await s.commit()
            kept = sorted(r.pair for r in (await s.execute(select(models.FlatbaseEpisode))).scalars().all())
        await eng.dispose()
        return kept
    assert asyncio.run(go()) == ["NEWUSDT"]
    # the kept decided row stops a second entry: the engine re-sees the flagged episode, resumes it as SHADOW, never opens
    e, led, opens, blocks = _run_bars(monkeypatch, rows, on, [on + 72], mcap=None, status="SHADOW")
    assert opens == [] and led[0].status == "SHADOW"


def test_max_open_cap_and_dashboard_count(monkeypatch):
    now_ms = int(__import__("time").time() * 1000)
    r, blocks, *_ = _e2e_open(monkeypatch, n_open=1, seed_strategy="FLATBASE_SHORT", flatbase_short=True, flatbase_bar_close_ms=now_ms)
    assert r is not None                                                                          # 1 open < the cap (10, operator)
    mp = pytest.MonkeyPatch()
    try:
        r2, b2, *_ = _e2e_open(mp, n_open=1, seed_strategy="FLATBASE_SHORT", th_kw=dict(flatbase_max_open=1), flatbase_short=True,
                               flatbase_bar_close_ms=now_ms)
    finally:
        mp.undo()
    assert r2 is None and "FLATBASE_MAX_OPEN" in b2                                               # its OWN cap, counted
    assert FB.flatbase_params(NS(flatbase_max_open=2))['max_open'] == 2 and FB.flatbase_params(NS(flatbase_max_open=-1))['max_open'] == 0
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert main.count("not in ('MANUAL', 'FLATBASE_SHORT')]),   # bot SLOT lane") == 2 and main.count('"flatbase_open_count": len(') == 2
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "ordCountEl.textContent = `Open Positions: ${_bo + _mo + _fo}`;" in html                  # 🩳 (362) total first
    assert "`Bot slots ${_bo} / ${data.max_open_positions ?? '–'}` + (_rdp > Number(data.max_open_positions || 0) ? ` (redeploy ≤ ${_rdp})` : '')" in html
    assert main.count('"max_open_positions_redeploy": (int(getattr(config.trading_config.investment, \'max_open_positions_hard\'') == 2 and "`FLATBASE ${_fo}${_fcap > 0 ? ' / ' + _fcap : ''}`" in html
    assert main.count('"flatbase_max_open": int(getattr(config.trading_config.thresholds, \'flatbase_max_open\', 0) or 0)') == 2


def _real_pass(monkeypatch, fbs_on, tracked=("TRKUSDT",)):
    """the REAL _update_frenzy_pass on stubbed reads: AAAUSDT is on FRENZY's shortlist, TRKUSDT is only FLATBASE-tracked; every read pair
    walks to a FLAGGED episode (so any pair reaching FRENZY's judgement would be flagged and judged). → what the long sleeves saw."""
    import services.trading_engine as TE
    now = (int(__import__("time").time()) // 300) * 300 + 10.0
    bar_open = int(now * 1000) // 300_000 * 300_000
    monkeypatch.setattr(TE._leash_time, "time", lambda: now)
    monkeypatch.setattr(TE, "_frenzy_flags", {}); monkeypatch.setattr(TE, "_frenzy_status", {"seeded": True})
    monkeypatch.setattr(TE, "_frenzy_kl_cache", {}); monkeypatch.setattr(TE, "_frenzy_norm_cache", {})
    th = TE.config.trading_config.thresholds
    for k, v in dict(frenzy_long_enabled=True, frenzy_short_observe=False, frenzy_wide_enabled=False, frenzy_lite_enabled=True,
                     frenzy_willy_enabled=True, flatbase_short_enabled=fbs_on, frenzy_catchup_max_bars=0).items():
        monkeypatch.setattr(th, k, v, raising=False)
    allp = [dict(pair="AAAUSDT", symbol="AAA/USDT:USDT", volume_24h=5e7, change_24h=30.0, range_24h=30.0, price=1.0),
            dict(pair="TRKUSDT", symbol="TRK/USDT:USDT", volume_24h=5e7, change_24h=1.0, range_24h=2.0, price=1.0)]

    async def tops(*a, **k):
        return allp
    monkeypatch.setattr(TE.binance_service, "get_top_futures_pairs", tops)

    async def ohlcv(sym, tf, limit):
        step = 300_000 if tf == '5m' else 3_600_000
        last = bar_open if tf == '5m' else bar_open // 3_600_000 * 3_600_000
        return [[last - (limit - 1 - i) * step, 1.0, 1.01, 0.99, 1.0, 1000.0] for i in range(limit)]
    monkeypatch.setattr(TE.binance_service, "get_ohlcv", ohlcv)
    monkeypatch.setattr(TE, "frenzy_walk", lambda closed, nh, th_: dict(spike_ts=int(closed[-1][0]) - 3_600_000, verified=True, hours=1.0,
                                                                         in_state=False, above_hour=False, fresh_on=False, vs_vwap_pct=-1.0,
                                                                         price=1.0, vwap=1.01, vol_mult=1.0))
    e = _engine(TE)
    seen = dict(willy=[], lite=[], fbs=[], unread=[])

    async def noop(*a, **k):
        return None
    for n in ("_frenzy_judged_get", "_frenzy_judged_set", "_frenzy_persist_flags", "_frenzy_willy_sweep", "_frenzy_lite_retry_sweep",
              "_flatbase_enter_queued", "_flatbase_sweep"):
        setattr(e, n, noop)
    e._frenzy_gvol_start = lambda *a, **k: None

    async def willy(db, pair, *a, **k):
        seen['willy'].append(pair)
    e._frenzy_willy_eval = willy

    async def lite(db, pair, *a, **k):
        seen['lite'].append(pair)
    e._frenzy_lite_eval = lite

    async def fbe(db, pair, *a, **k):
        seen['fbs'].append(pair)
    e._flatbase_eval = fbe
    e._flatbase_pairs = lambda: set(tracked)
    e._frenzy_note_unread = lambda names, s_, *a: seen['unread'].append(sorted(names))
    asyncio.run(e._update_frenzy_pass(None))
    return seen, sorted(TE._frenzy_flags)


def test_frenzy_candidate_set_unchanged_by_flatbase_tracking(monkeypatch):
    """the FLATBASE-only extra reads never reach FRENZY: same flags, same LITE / WILLY judgements, same catch-up horizons as with the
    sleeve OFF; the tracked pair is read for FLATBASE only."""
    off, flags_off = _real_pass(monkeypatch, False)
    mp = pytest.MonkeyPatch()
    try:
        on, flags_on = _real_pass(mp, True)
    finally:
        mp.undo()
    assert flags_on == flags_off == ["AAAUSDT"]
    assert on['willy'] == off['willy'] == ["AAAUSDT"] and on['lite'] == off['lite'] == ["AAAUSDT"]
    assert on['unread'] == off['unread'] == [["AAAUSDT"]]
    assert sorted(on['fbs']) == ["AAAUSDT", "TRKUSDT"] and off['fbs'] == []


def test_scout_peak_age_rebuild_matches_the_engine_rule():
    """⏱ (DECISION_LOG 401) the scout's kline rebuild = s4_scan's peak_age_h / off_peak_pct on the same rows; a stamped value wins."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import scout_flatbase_short as SF
    rows, on = _series()
    ref = _ref_s4(rows, on)
    sp = _spike_ms(rows, on)
    s = FB.s4_scan(_window(rows, ref), sp, _th())
    h, op = SF.peak_age_from_rows(rows, sp, rows[ref][0])
    assert h == pytest.approx(s['peak_age_h']) and op == pytest.approx(s['off_peak_pct'])
    assert SF.peak_age_from_rows(rows[on + 1:], sp, rows[ref][0]) is None        # spike bar missing → unscored, never a guess
    o = pd.Timestamp(rows[ref][0] + BAR + 7_000, unit="ms")                          # the fill lands 7 s after the S4 close
    r = NS(pair="XUSDT", o=o, entry_flatbase_peak_age_h=float("nan"), entry_frenzy_spike_at=pd.Timestamp(sp, unit="ms").isoformat(),
           entry_flatbase_off_peak_pct=s['off_peak_pct'])
    kl = lambda pair, start, n: [x for x in rows if start <= x[0] < start + n * BAR]
    got = SF.peak_age_of(r, None, 0, klines=kl)
    assert got[0] == pytest.approx(s['peak_age_h']) and got[1] == "rebuilt"
    assert SF.peak_age_of(NS(**{**vars(r), "entry_flatbase_peak_age_h": 3.5}), None, 0, klines=kl)[:2] == (3.5, "stamped")
    bad = SF.peak_age_of(NS(**{**vars(r), "entry_flatbase_off_peak_pct": s['off_peak_pct'] + 5}), None, 0, klines=kl)
    assert bad[1] == "rebuilt ⚠ off-peak Δ"
    late = SF.peak_age_of(NS(**{**vars(r), "o": pd.Timestamp(rows[ref][0] + BAR + 7 * 60_000, unit="ms")}), None, 0, klines=kl)
    assert late[0] == pytest.approx(s['peak_age_h'])                                 # 7 min late (past the next 5m close) → still the S4 bar
    assert SF.peak_age_of(NS(**{**vars(r), "o": pd.Timestamp(rows[ref][0] + BAR + 20 * 60_000, unit="ms")}), None, 0, klines=kl)[2]
    assert SF.peak_age_of(r, None, 0, klines=lambda *a: None)[2] == "fetch stopped / --no-fetch"
    tie = [list(x) for x in rows]
    k_pk = next(k for k in range(on, ref + 1) if rows[k][2] == s['peak'])
    tie[ref - 1][2] = s['peak']                                                      # a later row equals the peak → the FIRST row still wins
    if ref - 1 != k_pk:
        assert FB.s4_scan(_window(tie, ref), sp, _th())['peak_age_h'] == pytest.approx(s['peak_age_h'])
        assert SF.peak_age_from_rows(tie, sp, rows[ref][0])[0] == pytest.approx(s['peak_age_h'])


# ── 🩳🔒 OBS-FB2 LOW-TURNOVER SWITCH (DECISION_LOG 407) ───────────────────────────────────────────────────────────────────────────
def test_obs_fb2_cells_price_both_exits_with_hold():
    import scout_flatbase_short as SF2
    t = pd.Timestamp("2026-10-30 08:00")
    row = NS(t=t, pair="AUSDT", px=100.0)
    s0 = int(t.value // 1_000_000) + 300_000

    def kl(sym, start, n):   # price slides 6 % down over 12 bars, then flat: +5 (TP5_SL10) hits first, +8 never
        return [[s0 + i * 300_000, 100 - 0.5 * min(i, 12), 100 - 0.5 * min(i, 12) + 0.1, 100 - 0.5 * min(i, 12) - 0.1, 100 - 0.5 * min(i, 12), 1]
                for i in range(n)]
    a = SF2.price_shadow_cell(row, "LOW×TP5_SL10", None, 0, fetch=False, klines=kl, slip_pct=0.0, funding=lambda *x: -0.05)
    b = SF2.price_shadow_cell(row, "LOW×TODAY", None, 0, fetch=False, klines=kl, slip_pct=0.0, funding=lambda *x: -0.05)
    assert a is not None and a[0] == pytest.approx(5.0 - 0.045 - 0.018 - 0.05, abs=0.1) and a[1] < 120   # +5 maker TP, funding incl.
    assert b is not None and b[1] >= 71 * 60 and b[0] > a[0]               # +8 never → held to the 72 h close (≈ +6 %)


def test_obs_fb2_verdict_bar():
    import scout_flatbase_short as SF2
    days = pd.date_range("2026-11-01", periods=12, freq="D")
    good = [(days[i % 12] + pd.Timedelta(hours=i), f"P{i}USDT", 4.0 + (i % 3)) for i in range(30)]
    v, txt = SF2.obs_fb2_verdict(good, [-2.7] * 13)
    assert v == "CANDIDATE" and "operator decides" in txt
    assert SF2.obs_fb2_verdict(good[:29], [-2.7])[0] == "COLLECTING"
    assert SF2.obs_fb2_verdict([(t, p, -1.0) for t, p, _ in good], [0.5] * 6)[0] == "RETIRE"
    assert SF2.obs_fb2_verdict(good, [-2.7] * 4)[0] == "COLLECTING"            # comparator too thin
    assert SF2.obs_fb2_verdict(good[:28], [-2.7] * 13, unpriced=2)[0] == "COLLECTING"   # an open window blocks the frozen read
    one_day = [(days[0] + pd.Timedelta(minutes=i), f"P{i}USDT", 4.0) for i in range(30)]
    assert SF2.obs_fb2_verdict(one_day, [-2.7] * 13)[0] == "COLLECTING"             # 1 day < 10 days
    conc = [(t, "SAMEUSDT", x) for t, _, x in good]
    assert SF2.obs_fb2_verdict(conc, [-2.7] * 13)[0] == "KEEP_OBSERVING"            # one pair carries the whole gain


def test_obs_fb2_prefix_grows_until_the_day_count():
    import scout_flatbase_short as SF2
    days = pd.date_range("2026-11-01", periods=12, freq="D")
    clustered = [(days[i // 4] + pd.Timedelta(minutes=i), f"P{i}USDT", 4.0) for i in range(32)]   # 4 a day → 30 on 8 days
    assert SF2.obs_fb2_verdict(clustered[:30], [-2.7] * 13)[0] == "COLLECTING"
    more = clustered + [(days[8] + pd.Timedelta(hours=1), "XUSDT", 4.0), (days[9] + pd.Timedelta(hours=1), "YUSDT", 4.0)]
    assert SF2.obs_fb2_verdict(more, [-2.7] * 13)[0] in ("CANDIDATE", "KEEP_OBSERVING")       # the grown prefix reaches 10 days
    assert SF2.obs_fb2_verdict(more[:28], [-2.7] * 13, unpriced=2, unpriced_days=[days[10], days[11]])[0] == "COLLECTING"


# ── 🔁🔒 FBX — the +5 / −10 SWITCH-BACK GATE (DECISION_LOG 408) ───────────────────────────────────────────────────────────────────
def test_fbx_verdict_early_and_final():
    import scout_flatbase_short as SF2
    days = pd.date_range("2026-11-01", periods=10, freq="D")
    rows = lambda o, n, k=16: [(days[i % 10] + pd.Timedelta(hours=i), f"P{i}USDT", o, n, 500.0) for i in range(k)]
    r = SF2.fbx_verdict(rows(2.0, 0.5))                                            # +8 / −15 ahead by 1.5 pt → early revert
    assert r["early"][0] == "REVERT" and r["final"][0] == "REVERT"
    r = SF2.fbx_verdict(rows(1.0, 0.5))                                            # 0.5 pt behind: early passes, final reverts
    assert r["early"][0] == "EARLY_OK" and r["final"][0] == "REVERT"
    assert SF2.fbx_verdict(rows(-1.0, 0.5))["final"][0] == "KEEP"
    assert SF2.fbx_verdict(rows(0.5, 0.5))["final"][0] == "KEEP"                   # a tie keeps +5 / −10 (≥)
    assert SF2.fbx_verdict(rows(1.0, 0.5, 9))["early"][0] == "COLLECTING"
    one_day = [(days[0] + pd.Timedelta(minutes=i), f"P{i}USDT", 0.0, 1.0, 500.0) for i in range(20)]
    assert SF2.fbx_verdict(one_day)["final"][0] == "COLLECTING"                    # 15 fills but 1 day < 8 days
    held = rows(-1.0, 0.5)
    held[3] = held[3][:2] + (None, None, 500.0)
    r = SF2.fbx_verdict(held)
    assert r["early"][0] == "HOLD" and r["final"][0] == "HOLD"                     # an unpriced fill holds the read, never skipped
    # $ at the actual notional: avg % ties-or-better but the bigger fills lose → revert
    mix = [(days[i % 10] + pd.Timedelta(hours=i), f"P{i}USDT", 0.0, (3.0 if i % 2 else -2.0), (100.0 if i % 2 else 1000.0)) for i in range(16)]
    assert SF2.fbx_verdict(mix)["final"][0] == "REVERT"


def test_real_by_exit_books_the_exit_a_fill_ran_else_rewalks():
    import scout_flatbase_short as SF2
    sw = pd.Timestamp("2026-10-29 12:00")
    real = pd.DataFrame(dict(o=[pd.Timestamp("2026-10-28 00:00"), pd.Timestamp("2026-10-29 08:00"), pd.Timestamp("2026-10-30 00:00")],
                             c=[pd.Timestamp("2026-10-28 06:00"), pd.Timestamp("2026-10-29 14:00"), pd.Timestamp("2026-10-30 03:00")],
                             pair=["AUSDT", "BUSDT", "CUSDT"], net=[7.9, -9.0, 4.9]))
    w = lambda a, b: {"LIVE": {"net_pct": a}, "TP5_SL10": {"net_pct": b}}
    walks = {(real.o[0], "AUSDT"): w(7.95, 4.95), (real.o[1], "BUSDT"): w(-15.1, -10.1), (real.o[2], "CUSDT"): w(7.9, 4.92)}
    f = SF2.real_by_exit_fn(real, walks, int(sw.value // 1_000_000))
    assert [x for _, x in f((8.0, 15.0))[0]] == [7.9, -15.1, 7.9]   # A booked (ran +8 / −15) · B straddled → re-walk · C re-walk
    assert [x for _, x in f((5.0, 10.0))[0]] == [4.95, -10.1, 4.9]  # A re-walk · B re-walk · C booked (ran +5 / −10)
    g = SF2.real_by_exit_fn(real, {}, None)                          # DL 408 not in git → everything ran +8 / −15
    assert [x for _, x in g((8.0, 15.0))[0]] == [7.9, -9.0, 4.9] and g((5.0, 10.0)) == ([], 3)   # no walk → counted missing
    h = SF2.real_by_exit_fn(real.assign(net=[None, -9.0, 4.9]), walks, int(sw.value // 1_000_000))
    assert h((8.0, 15.0)) == ([(real.o[1], -15.1), (real.o[2], 7.9)], 1)   # an unreadable booked net is counted, never raises


def test_fbx_lines_counts_fresh_and_shows_straddlers(tmp_path):
    import scout_flatbase_short as SF2
    sw = pd.Timestamp("2026-10-29 12:00")
    real = pd.DataFrame(dict(o=[pd.Timestamp("2026-10-29 08:00"), pd.Timestamp("2026-10-30 00:00")],
                             c=[pd.Timestamp("2026-10-29 14:00"), pd.Timestamp("2026-10-30 03:00")], pair=["XANUSDT", "CUSDT"],
                             pct=[-9.0, 4.9], net=[-9.0, 4.9], entry_price=[1.0, 2.0], quantity=[500.0, 250.0]))
    walks = {(real.o[1], "CUSDT"): {"LIVE": {"net_pct": 7.9, "bars": 30}, "TP5_SL10": {"net_pct": 4.92, "bars": 10}}}
    out = " ".join(SF2.fbx_lines(real, walks, int(sw.value // 1_000_000), state_path=str(tmp_path / "fbx.json")))
    assert "fresh fills" in out and "priced 1" in out and "XANUSDT 10-29 08:00 booked -9.00" in out and "EARLY: COLLECTING" in out
    assert "not in git" in " ".join(SF2.fbx_lines(real, walks, None))


def test_registry_flatbase_line_matches_the_live_exit():
    import observe_registry as OR
    d = OR.diffs(OR.EXIT["scout_flatbase_short.py"]["walks"], OR.thresholds())
    assert not [k for k, _, _ in d if k.startswith("flatbase_")]     # re-based on DL 408 (the WILLY leg's own era is separate)


def test_fbx_lines_freeze_supersede_finals_and_open(tmp_path):
    import scout_flatbase_short as SF2
    sw = pd.Timestamp("2026-10-29 12:00")
    sw_ms = int(sw.value // 1_000_000)
    o = [sw + pd.Timedelta(hours=1 + 13 * i) for i in range(12)]
    real = pd.DataFrame(dict(o=o, c=[x + pd.Timedelta(hours=2) for x in o], pair=[f"P{i}USDT" for i in range(12)], pct=[4.9] * 12,
                             net=[4.9] * 12, entry_price=[1.0] * 12, quantity=[500.0] * 12))
    wk = lambda a, b: {"LIVE": {"net_pct": a, "bars": 20}, "TP5_SL10": {"net_pct": b, "bars": 8}}
    walks = {(x, f"P{i}USDT"): wk(7.9, 4.9) for i, x in enumerate(o)}
    st = str(tmp_path / "fbx.json")
    t1 = " ".join(SF2.fbx_lines(real, walks, sw_ms, state_path=st))
    assert "EARLY: REVERT" in t1 and "FINAL: SUPERSEDED" in t1                     # +8 / −15 ahead by 3 pt → revert, final superseded
    walks2 = {k: wk(0.0, 4.9) for k in walks}                                      # the data changes — the frozen reads do not
    t2 = " ".join(SF2.fbx_lines(real, walks2, sw_ms, state_path=st))
    assert "EARLY: REVERT" in t2 and "frozen" in t2 and "FINAL: SUPERSEDED" in t2
    # an UNPRICEABLE_FINAL fill is out (counted); an earlier still-open fill holds the read
    fin = {(o[0], "P0USDT")}
    t3 = " ".join(SF2.fbx_lines(real, walks2, sw_ms, state_path=None, finals=fin))
    assert "UNPRICEABLE_FINAL 1" in t3 and "EARLY: EARLY_OK" in t3
    opn = pd.DataFrame(dict(o=[o[0] + pd.Timedelta(minutes=5)], pair=["OPENUSDT"]))
    t4 = " ".join(SF2.fbx_lines(real, walks2, sw_ms, state_path=None, open_rows=opn))
    assert "still open 1" in t4 and "EARLY: HOLD" in t4


def test_fbx_ran_eras():
    import scout_flatbase_short as SF2
    sw = pd.Timestamp("2026-10-29 12:00")
    ms = int(sw.value // 1_000_000)
    R = lambda o, c: NS(o=pd.Timestamp(o), c=(pd.NaT if c is None else pd.Timestamp(c)))
    assert SF2.fbx_ran(R("2026-10-29 08:00", "2026-10-29 11:00"), ms) == "OLD"
    assert SF2.fbx_ran(R("2026-10-29 13:00", "2026-10-29 15:00"), ms) == "NEW"
    assert SF2.fbx_ran(R("2026-10-29 08:00", "2026-10-29 15:00"), ms) == "MIX"
    assert SF2.fbx_ran(R("2026-10-29 08:00", "2026-10-29 15:00"), None) == "OLD"
    end = int(pd.Timestamp("2026-11-05").value // 1_000_000)                      # a revert closes the +5 / −10 era
    assert SF2.fbx_ran(R("2026-11-04 20:00", "2026-11-05 03:00"), ms, end) == "MIX"
    assert SF2.fbx_ran(R("2026-11-04 20:00", "2026-11-04 23:00"), ms, end) == "NEW"


def test_tp_shadow_parity_reads_the_exit_each_fill_ran():
    import scout_flatbase_short as SF2
    sw = pd.Timestamp("2026-10-29 12:00")
    ms = int(sw.value // 1_000_000)
    rows = [NS(o=pd.Timestamp("2026-10-29 13:00"), c=pd.Timestamp("2026-10-29 15:00"), pct=4.94, close_reason="FRENZY_TP"),
            NS(o=pd.Timestamp("2026-10-28 13:00"), c=pd.Timestamp("2026-10-28 15:00"), pct=7.94, close_reason="FRENZY_TP")]
    w = {"LIVE": {"net_pct": 7.94, "reason": "FRENZY_TP"}, "TP5_SL10": {"net_pct": 4.94, "reason": "FRENZY_TP"}}
    for r in rows:   # the booked number matches the walk of the exit the fill ran (a post-switch +5 TP vs +8 walk would fail by 3 pts)
        ref = w["LIVE"]["net_pct"] if SF2.fbx_ran(r, ms) == "OLD" else w[SF2.FBX_KEY]["net_pct"]
        assert abs(ref - r.pct) <= SF2.TP_SH_PARITY_TOL
    k = {"TP5_SL10": {"net_pct": -10.06, "reason": "STOP_LOSS"}, "LIVE": {"net_pct": 7.94, "reason": "FRENZY_TP"}}
    line = SF2.tp_split_line("TP5_SL10", [(NS(close_reason="STOP_LOSS"), k)], kind=lambda r, w_: SF2._live_kind(w_["LIVE"]["reason"]))
    assert "TP→cut winner 1" in line                                               # a post-switch stop is classed by the +8 rule's outcome
