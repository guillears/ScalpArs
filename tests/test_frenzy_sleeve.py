"""🔥 FRENZY sleeve (Oct-2, DECISION_LOG 176) — pure rules (services/frenzy.py) + the wiring every surface must carry."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.frenzy import (BAR_MS, frenzy_breaks, frenzy_exit_for, frenzy_flagged, frenzy_long_status, frenzy_walk,  # noqa: E402
                             normal_hour_usd)


class TH:
    frenzy_spike_ret_pct = 5.0; frenzy_spike_vol_mult = 20.0; frenzy_spike_min_hour_usd = 2e6; frenzy_state_vol_mult = 100.0
    frenzy_min_hours = 2.0; frenzy_max_hours = 96.0; frenzy_min_volume_usd = 20e6; frenzy_max_atr_pct = 2.0
    frenzy_stop_pct = 3.0; frenzy_trail_arm_pct = 5.0; frenzy_trail_giveback_pct = 1.5; frenzy_long_skip_green_bar = False


NORM = 10_000.0   # the pair's normal hourly quote volume in these fixtures


def _bars(n_quiet=320, after=40, spike_vol=6_000_000.0, run_vol=2_000_000.0, drift=0.002, dip_at=None):
    """Quiet tape at 1.00, then a spike bar (+8 % on heavy volume), then `after` bars drifting up on heavy volume.
    dip_at = bars after the spike where price drops below the anchored VWAP for 3 bars."""
    rows = []; px = 1.0
    for i in range(n_quiet):
        rows.append([i * BAR_MS, px, px * 1.001, px * 0.999, px, NORM / 12])          # normal volume: NORM per hour
    px2 = px * 1.08
    rows.append([n_quiet * BAR_MS, px, px2, px, px2, spike_vol / px2])
    px = px2
    for k in range(1, after + 1):
        if dip_at is not None and dip_at <= k < dip_at + 3:
            c = 1.0
        else:
            px = px * (1 + drift); c = px
        rows.append([(n_quiet + k) * BAR_MS, c, c * 1.001, c * 0.999, c, run_vol / c])
    return rows


def test_spike_flags_the_pair_and_the_setup_turns_on_two_hours_later():
    b = _bars(after=23)                                    # 23 bars after the spike: not yet 2 h
    ep = frenzy_walk(b, NORM, TH)
    assert ep is not None and ep["verified"] and frenzy_flagged(ep, TH)
    assert ep["spike_ts"] == 320 * BAR_MS + BAR_MS and abs(ep["base"] - 1.0) < 1e-9
    assert ep["above_hour"] and not ep["in_state"] and not ep["fresh_on"]
    ok, code, _ = frenzy_long_status(ep, 1.5, 50e6, TH)
    assert (ok, code) == (False, "FRENZY_TOO_EARLY")
    ep = frenzy_walk(_bars(after=24), NORM, TH)            # the 24th bar = 2 h: the setup turns ON on this bar only
    assert ep["in_state"] and ep["fresh_on"] and ep["vol_mult"] >= 100
    assert frenzy_long_status(ep, 1.5, 50e6, TH) == (True, "FRENZY_READY", "READY")
    ep = frenzy_walk(_bars(after=25), NORM, TH)            # one bar later: still ON, no longer an entry bar
    assert ep["in_state"] and not ep["fresh_on"] and frenzy_long_status(ep, 1.5, 50e6, TH)[1] == "FRENZY_ON"


def test_setup_needs_an_hour_above_the_average_price_and_re_arms_after_an_hour_off():
    b = _bars(after=60, dip_at=30)
    ep = frenzy_walk(b[:320 + 1 + 33], NORM, TH)           # inside / just after the dip: below the average price
    assert not ep["in_state"] and frenzy_long_status(ep, 1.5, 50e6, TH)[1] == "FRENZY_BELOW_AVG"
    # the text follows the price: under the line → "below …"; back above but not for an hour → "back above … N of 12 closes"
    for k in range(30, 44):
        e = frenzy_walk(b[:320 + 1 + k], NORM, TH); code, txt = frenzy_long_status(e, 1.5, 50e6, TH)[1:]
        assert code == "FRENZY_BELOW_AVG" and not e["above_hour"]
        if e["vs_vwap_pct"] < 0:
            assert txt.startswith("below its average price (") and e["above_streak"] == 0
        else:
            assert e["above_streak"] >= 1 and txt.startswith("back above its average (+") and txt.endswith(f" · {e['above_streak']} of 12 closes")
    e = dict(in_state=False, above_hour=False, vs_vwap_pct=None, above_streak=0)
    assert frenzy_long_status(e, 1.5, 50e6, TH)[2] == "below its average price"
    e = dict(in_state=False, above_hour=False, vs_vwap_pct=0.4, above_streak=30)       # capped: an unbroken hour would be above_hour
    assert frenzy_long_status(e, 1.5, 50e6, TH)[2] == "back above its average (+0.4%) · 11 of 12 closes"
    e = dict(in_state=False, above_hour=False, vs_vwap_pct=-0.004, above_streak=0)
    assert frenzy_long_status(e, 1.5, 50e6, TH)[2] == "below its average price (-0.00%)"           # just under the line: the sign stays
    fresh = [k for k in range(34, 61) if (frenzy_walk(b[:320 + 1 + k], NORM, TH) or {}).get("fresh_on")]
    assert fresh == [44]                                   # 3 dip bars (30–32) + 12 closes back above → ON again at bar 44, once


def test_volume_and_atr_gates_and_the_24h_floor():
    ep = frenzy_walk(_bars(after=24, run_vol=50_000.0), NORM, TH)         # 60× normal: flagged, setup off
    assert frenzy_flagged(ep, TH) and not ep["in_state"] and frenzy_long_status(ep, 1.5, 50e6, TH)[1] == "FRENZY_VOL_FADED"
    assert frenzy_long_status(ep, 1.5, 50e6, TH)[2].startswith("held above 1 h · volume ")
    ep = frenzy_walk(_bars(after=24), NORM, TH)
    assert frenzy_long_status(ep, 2.01, 50e6, TH)[1] == "FRENZY_ATR_HIGH" and frenzy_long_status(ep, None, 50e6, TH)[1] == "FRENZY_ATR_HIGH"
    assert frenzy_long_status(ep, 2.0, 50e6, TH)[0] is True
    assert frenzy_long_status(ep, 1.0, 19e6, TH)[1] == "FRENZY_VOL24_LOW" and frenzy_long_status(ep, 1.0, None, TH)[1] == "FRENZY_VOL24_LOW"

    class NoAtr(TH):
        frenzy_max_atr_pct = 0.0
    assert frenzy_long_status(ep, 9.0, 50e6, NoAtr)[0] is True            # 0 = no ATR gate


def test_no_spike_no_flag_and_bad_input_fails_closed():
    quiet = _bars(after=0)[:-1]
    assert frenzy_walk(quiet, NORM, TH) is None
    assert frenzy_walk(_bars(after=24, spike_vol=100_000.0, run_vol=100_000.0), NORM, TH) is None   # volume below 20× normal
    assert frenzy_walk(_bars(after=24), None, TH) is None and frenzy_walk(_bars(after=24), 0, TH) is None
    assert frenzy_walk([], NORM, TH) is None and frenzy_walk([[0, "x", 1, 1, 1, 1]] * 50, NORM, TH) is None
    assert frenzy_flagged(None, TH) is False


def test_flag_needs_a_verifiable_spike_and_ends_at_the_max_age_or_24h_without_the_setup():
    ep = frenzy_walk(_bars(n_quiet=100, after=24), NORM, TH)             # < 25 h of history before the spike
    assert ep is not None and not ep["verified"] and not frenzy_flagged(ep, TH)

    class Short(TH):
        frenzy_max_hours = 1.0
    assert not frenzy_flagged(frenzy_walk(_bars(after=24), NORM, TH), Short)
    b = _bars(after=24) + [[(345 + k) * BAR_MS, 1.2, 1.2, 1.2, 1.2, NORM / 12 / 1.2] for k in range(300)]   # volume back to normal for 25 h
    assert frenzy_walk(b, NORM, TH) is None                               # 24 h without a state bar: the episode is over


def test_exit_stop_trail_and_live_floor():
    f = frenzy_exit_for
    assert f(-2.99, 0.0, TH) == (False, "STOP_LOSS", -3.0) and f(-3.0, 0.0, TH)[:2] == (True, "STOP_LOSS")
    assert f(2.0, 4.99, TH) == (False, "STOP_LOSS", -3.0)                 # below the arm: only the stop
    c, why, line = f(3.5, 5.0, TH)                                        # armed at +5: line = 5 − 1.5 × 1.05 = 3.425
    assert (c, why) == (False, "RUNNER_TRAIL") and abs(line - 3.425) < 1e-9
    assert f(3.42, 5.0, TH)[:2] == (True, "RUNNER_TRAIL")
    assert abs(f(0.0, 20.0, TH)[2] - (20.0 - 1.5 * 1.2)) < 1e-9           # the give-back is % of PRICE at the best point
    assert f(-2.3, 0.0, TH, stop_floor=-2.2)[:2] == (True, "STOP_LOSS") and f(-2.1, 0.0, TH, stop_floor=-2.2)[0] is False   # live: inside the backstop
    assert f(-2.9, 0.0, TH, stop_floor=-9.0)[0] is False                  # a deeper floor never widens the stop
    assert f(None, 0.0, TH)[0] is False and f("x", 0.0, TH)[0] is False   # unreadable P&L never closes

    class Neg(TH):
        frenzy_stop_pct = -3.0                                            # a sign slip in the JSON must not disable the stop
    assert f(-3.0, 0.0, Neg)[:2] == (True, "STOP_LOSS")


def test_short_observation_break_rule():
    up = [[i * BAR_MS, 0, 0, 0, 100 + i * 0.1, 1] for i in range(300)]    # steady rise: closes above both EMAs
    assert frenzy_breaks(up) == []
    last = up[-1][4]
    assert frenzy_breaks(up + [[300 * BAR_MS, 0, 0, 0, last - 4.0, 1]]) == [50]          # below EMA50 (≈ last − 2.5), still above EMA200
    assert frenzy_breaks(up + [[300 * BAR_MS, 0, 0, 0, last - 20.0, 1]]) == [50, 200]
    two = up + [[300 * BAR_MS, 0, 0, 0, last - 20.0, 1], [301 * BAR_MS, 0, 0, 0, last - 21.0, 1]]
    assert frenzy_breaks(two) == []                                        # only the FIRST close below counts
    assert frenzy_breaks(up[:100]) == [] and frenzy_breaks(None) == []


def test_normal_hour_volume_ignores_the_last_day_and_needs_ten_days():
    H = 3_600_000; last = 400 * H
    hb = [[i * H, 1, 1, 1, 1, 100.0] for i in range(400)]
    assert normal_hour_usd(hb, last) == 100.0
    hot = [r[:] for r in hb]
    for r in hot[-24:]:
        r[5] = 1e9                                                         # the frenzy day itself never moves the normaliser
    assert normal_hour_usd(hot, last) == 100.0
    assert normal_hour_usd(hb[:200], 200 * H) is None and normal_hour_usd(None, last) is None


def test_config_parity_and_every_surface():
    import config as C
    th = C.trading_config.thresholds
    fields = sorted(k for k in type(th).model_fields if k.startswith("frenzy_"))
    assert len(fields) == 28
    cfgj = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert sorted(k for k in cfgj if k.startswith("frenzy_")) == fields                   # every field has a JSON value
    assert type(th).model_fields["frenzy_long_enabled"].default is False                  # OFF in code; the JSON arms it
    assert cfgj["frenzy_long_enabled"] is True and cfgj["frenzy_long_invest_mult"] == 1.0 and cfgj["frenzy_long_lev_mult"] == 0.32
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    listed = dict(re.findall(r"\['(config-fz-[a-z0-9-]+)', '(frenzy_[a-z0-9_]+)'", html))
    by_key = {v: k for k, v in listed.items()}
    by_key.update(frenzy_long_enabled="config-fz-long-enabled", frenzy_short_observe="config-fz-short-observe", frenzy_pair_blacklist="config-fz-blacklist",
                  frenzy_long_skip_green_bar="config-fz-skip-green", frenzy_wide_enabled="config-fz-wide-enabled")
    assert sorted(by_key) == fields                                                       # every field has a UI input
    for key, _id in by_key.items():
        assert html.count(f'id="{_id}"') == 1, _id
        assert html.count(key) >= (1 if key in listed.values() else 2), key               # list-driven fields appear once; the rest are loaded and saved by name
    assert "FRENZY_NUM_FIELDS.map(" in html and "of FRENZY_NUM_FIELDS)" in html           # one list drives the load and the save
    for _id in ("frenzy-monitor-line", "frenzy-flags-body", "frenzy-body", "frenzy-breaks-body"):
        assert html.count(f'id="{_id}"') == 1 and html.count(f"getElementById('{_id}')") == 1
    assert html.count("lines.push(...frenzyReportLines(perf, hr2));") == 2                # clipboard AND saved-file exports
    for title in ("## 🔥 FRENZY Flagged Pairs (now)", "## 🔥 FRENZY Fills (FRENZY_LONG · FRENZY_WIDE)", "## 🔥 FRENZY Short Observations"):
        assert html.count(title) == 1
    assert '<option value="FRENZY_LONG">' in html and "frenzyMark(_fz)" in html


def test_engine_and_api_wiring():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng.count("frenzy_exit_for(") == 4                                             # sleeve: candle + realtime · MANUAL "Frenzy exit": candle + realtime
    assert "await self._update_frenzy(db, wait=False)" in eng and eng.count('(_fz_es if _frenzy else f"SURGE_{direction}" if _surge') == 2
    assert "_sg_pref = _fz_es.lower() if _frenzy else" in eng and 'cell_src = _fz_es if _frenzy else' in eng
    assert '"SURGE_SHORT", "SURGE_LONG", "FRENZY_LONG", "FRENZY_WIDE")' in eng                           # no pair-EMA exit on a FRENZY fill
    assert "_frenzy_kill" not in eng and "frenzy_kill_verdict" not in eng                 # operator: no automatic off
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert main.count("'FRENZY_LONG'") >= 3 and "'Frenzy-Long'" in main and "'Frenzy-Wide'" in main and main.count("'FRENZY_WIDE'") >= 3
    for key in ('"frenzy_rows"', '"frenzy_flags"', '"frenzy_breaks"', '"frenzy_monitor"'):
        assert main.count(key) == 3, key                                                  # the payload + both fallback payloads
    import models as M
    cols = {c.name for c in M.Order.__table__.columns}
    need = {"entry_frenzy_spike_at", "entry_frenzy_hours", "entry_frenzy_vwap", "entry_frenzy_vs_vwap_pct", "entry_frenzy_vol_mult",
            "entry_frenzy_run_pct", "entry_frenzy_stop_atr", "entry_frenzy_bar_ret_pct", "entry_frenzy_di_spread"}
    assert need <= cols and M.FrenzyBreak.__tablename__ == "frenzy_breaks"
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert all(f"'{c}'" in db for c in need)                                              # the migration adds every column
    import inspect
    from services.trading_engine import TradingEngine
    params = set(inspect.signature(TradingEngine.open_position).parameters)
    assert {"frenzy_long"} | need <= params


def test_flagged_pairs_are_pinned_and_marked_in_the_pairs_payload(monkeypatch):
    """Behavioural: a flagged pair inside the Top list carries the mark and moves to the top; a flagged pair OUTSIDE the list gets
    its own row from the sleeve's reading; the FRENZY tables' payload builders run on real rows."""
    import asyncio, datetime as dt
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models
    import services.trading_engine as TE
    mk = lambda pair, **kw: dict(dict(pair=pair, spike_ts=1_700_000_000_000, base=1.0, vwap=1.5, price=1.6, peak=2.0, hours=40.0, vol_mult=150.0,
                                      above_hour=True, in_state=True, fresh_on=False, verified=True, vs_vwap_pct=6.7, run_pct=100.0, gain_pct=60.0,
                                      off_peak_pct=-20.0, atr_pct=1.8, volume_24h=3e8, live_price=1.61, ready=False, code="FRENZY_ON",
                                      text="ON (entry bar passed)", misses=0, ema5=1.6, ema8=1.59, ema13=1.58, ema20=1.57, rsi=61.0, adx=30.0), **kw)
    monkeypatch.setattr(TE, "_frenzy_flags", {"QNTUSDT": mk("QNTUSDT"), "TINYUSDT": mk("TINYUSDT", in_state=False, text="below its average price", volume_24h=4e7, spike_ts=1_700_000_600_000)})

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with async_sessionmaker(eng, expire_on_commit=False)() as db:
            now = dt.datetime.utcnow()
            for pair, vol in (("BTCUSDT", 9e9), ("NEARUSDT", 5e8), ("QNTUSDT", 4e8)):
                db.add(models.PairData(pair=pair, price=4.9, volume_24h=vol, ema5=4.9, ema8=4.91, ema13=4.92, ema20=4.94, rsi=43.0, adx=24.0,
                                       signal="NO_TRADE", confidence="NO_TRADE", updated_at=now))
            db.add(models.FrenzyBreak(pair="QNTUSDT", line=50, bar_close_at=now.replace(microsecond=0), price=1.5, hours=41.0, run_pct=100.0,
                                      off_peak_pct=-25.0, vs_vwap_pct=-1.0, vol_mult=120.0, atr_pct=1.9, volume_24h=3e8, btc_rsi=48.0, bull_pct=40.0, bear_pct=35.0))
            await db.commit()
            return await main.get_pairs(db=db, limit=50), await main._frenzy_break_rows(db)
    rows, breaks = asyncio.run(run())
    # flagged first with the NEWEST spike on top (TINY's spike is 10 min later, though it has less volume and is outside the Top list); the rest by volume
    assert [r["pair"] for r in rows] == ["TINYUSDT", "QNTUSDT", "BTCUSDT", "NEARUSDT"]
    q, t = rows[1], rows[0]
    assert q["frenzy"]["in_state"] and q["frenzy"]["late"] and not q.get("frenzy_only")
    assert t["frenzy_only"] is True and t["price"] == 1.61 and t["volume_24h"] == 4e7 and t["atr_pct"] == 1.8 and t["signal"] is None
    assert rows[2]["frenzy"] is None
    assert breaks and breaks[0]["pair"] == "QNTUSDT" and breaks[0]["line"] == 50
    fl = main._frenzy_flag_rows()
    assert [f["pair"] for f in fl] == ["QNTUSDT", "TINYUSDT"] and main._frenzy_monitor_payload()["flagged"] == 2


def test_a_spike_inside_an_older_episode_that_left_the_window_is_never_trusted():
    """Review I-1: a long frenzy whose FIRST spike scrolled out of the 1500-bar window must not re-anchor on a later spike. The
    spike is trusted only when the 25 h before it hold no bar at the state volume, or the previous in-window episode was trusted."""
    hot = _bars(n_quiet=320, after=24)
    for r in hot[10:300]:
        r[5] = 2_000_000.0                       # state-level volume before the visible spike, no visible spike bar: an older frenzy
    ep = frenzy_walk(hot, NORM, TH)
    assert ep is not None and not ep["verified"] and not frenzy_flagged(ep, TH)
    assert frenzy_long_status(ep, 1.5, 50e6, TH)[0] is True          # the setup itself reads ON — only the flag refuses it
    # chained: a trusted episode ends (25 h of normal volume), hot-but-below-average bars follow, then a new spike → trusted
    a = _bars(n_quiet=320, after=24)
    n = len(a); px = a[-1][4]
    a += [[(n + k) * BAR_MS, px, px, px, px, NORM / 12 / px] for k in range(300)]                 # the first episode dies
    n = len(a)
    a += [[(n + k) * BAR_MS, px, px * 1.08, px, px * 1.08, 6_000_000.0 / (px * 1.08)] for k in range(1)]   # second spike
    px2 = px * 1.08; n = len(a)
    for k in range(1, 25):
        px2 *= 1.002; a.append([(n + k - 1) * BAR_MS, px2, px2 * 1.001, px2 * 0.999, px2, 2_000_000.0 / px2])
    ep2 = frenzy_walk(a, NORM, TH)
    assert ep2 is not None and ep2["verified"] and ep2["fresh_on"] and ep2["spike_ts"] == 645 * BAR_MS + BAR_MS


def test_exit_through_the_engine_intercept_and_the_late_entry_guard():
    """Behavioural: frenzy_exit_for is what both engine paths call; the entry guard constant exists and FRENZY closes are urgent."""
    import services.trading_engine as TE
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert 'f"{_bk}_PAIR_DAY_CAP"' in eng and "frenzy_max_entries_per_pair_day" in eng                    # per-pair daily ceiling (DECISION_LOG 177)
    assert TE.FRENZY_ENTRY_MAX_LATE_S == 120 and 'f"{_fz_bk}_LATE"' in eng and 'f"{_fz_bk}_DISLOC" if _frenzy else "SURGE_DISLOC"' in eng and 'f"{_bk}_DISLOC"' in eng
    assert eng.count('(order.entry_strategy or "") in FRENZY_STRATEGIES or reason.startswith(RH_STOP_CLASS)') == 2   # live + paper: every FRENZY close is taker
    assert "self._journal_pair = pair; self._journal_ctx = None" in eng and "await self._frenzy_persist_flags()" in eng
    import models as M
    assert M.FrenzyFlag.__tablename__ == "frenzy_flags"


def test_pair_day_cap_counts_todays_frenzy_fills_only(monkeypatch):
    """Behavioural (DECISION_LOG 177): three FRENZY_LONG fills on the pair today refuse the next setup; yesterday's fills, other
    pairs, other strategies and cap 0 do not."""
    import asyncio, datetime as dt, time
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, config as C
    import services.trading_engine as TE
    bar_open = int(time.time() // 300) * 300_000
    bar_dt = dt.datetime.utcfromtimestamp(bar_open / 1000); day0 = bar_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if (bar_dt - day0).total_seconds() < 900:
        return   # too close to UTC midnight to seat three earlier fills inside the day
    th = C.trading_config.thresholds
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**6)   # the late-entry guard is not under test (the wall clock is mid-bar)

    async def run(rows, cap):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        old = th.frenzy_max_entries_per_pair_day
        try:
            th.frenzy_max_entries_per_pair_day = cap
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                for pair, strat, when in rows:
                    db.add(models.Order(pair=pair, direction="LONG", status="CLOSED", is_paper=True, entry_strategy=strat, confidence="STRONG_BUY",
                                        entry_price=1.0, quantity=1.0, investment=1.0, leverage=20, notional_value=20.0, opened_at=when))
                await db.commit()
                e = object.__new__(TE.TradingEngine); e.is_paper_mode = True; blocks = []
                e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
                e._flip_entry_fields = lambda *a, **k: {}
                e._sanitize_open_kwargs = lambda ef, s, d: ef

                async def quiet_market(sig_open, wait=True):
                    return 0.5   # 🌊 the market-volume gate (DECISION_LOG 194) is not under test: a quiet market passes it
                e._frenzy_gvol_value = quiet_market
                opened = []

                async def fake_open(**kw):
                    opened.append(kw["pair"]); return None
                e.open_position = fake_open
                flag = dict(pair="FOOUSDT", spike_ts=bar_open - 7_200_000, hours=2.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=150.0, run_pct=20.0,
                            atr_pct=1.5, volume_24h=5e7, price=1.0, live_price=1.0)
                await e._frenzy_open(db, flag, {}, bar_open)
        finally:
            th.frenzy_max_entries_per_pair_day = old
            await eng.dispose()
        return blocks, opened, flag.get("last_fire")
    t1 = day0 + dt.timedelta(minutes=5); yday = day0 - dt.timedelta(seconds=1)
    three = [("FOOUSDT", "FRENZY_LONG", t1)] * 3
    b, o, lf = asyncio.run(run(three, 3))
    assert "FRENZY_PAIR_DAY_CAP" in b and o == [] and "3 entries on this pair today (max 3)" in lf
    assert asyncio.run(run(three, 0))[1] == ["FOOUSDT"] and asyncio.run(run(three, 4))[1] == ["FOOUSDT"]          # 0 = no cap · under the cap
    assert asyncio.run(run([("FOOUSDT", "FRENZY_LONG", yday)] * 3, 3))[1] == ["FOOUSDT"]                            # yesterday does not count
    assert asyncio.run(run([("BARUSDT", "FRENZY_LONG", t1)] * 3, 3))[1] == ["FOOUSDT"]                              # another pair
    assert asyncio.run(run([("FOOUSDT", "MANUAL", t1), ("FOOUSDT", "SURGE_LONG", t1), ("FOOUSDT", None, t1)], 3))[1] == ["FOOUSDT"]   # other strategies
    assert asyncio.run(run([("FOOUSDT", "FRENZY_LONG", day0)] * 3, 3))[1] == []                                     # a fill in the first second of the day counts


def test_manual_entry_can_use_the_frenzy_exit():
    """Oct-2 (operator): the manual panel's "🔥 Frenzy exit" — a MANUAL trade closed by the FRENZY stop / trailing exit (live settings)."""
    f = frenzy_exit_for
    # a SHORT measures the give-back from its LOWEST price: at +10 % the line is 10 − 1.5 × 0.90, a LONG's is 10 − 1.5 × 1.10
    assert abs(f(0.0, 10.0, TH, short=True)[2] - (10.0 - 1.5 * 0.90)) < 1e-9 and abs(f(0.0, 10.0, TH)[2] - (10.0 - 1.5 * 1.10)) < 1e-9
    assert f(-3.0, 0.0, TH, short=True)[:2] == (True, "STOP_LOSS") and f(8.6, 10.0, TH, short=True)[:2] == (True, "RUNNER_TRAIL")
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert 'if exit_mode not in ("FIXED", "MOMENTUM", "FLOOR", "FRENZY", "BULLRUN", "BULLRUN_SL"):' in eng
    assert "in ('FIXED', 'FLOOR', 'FRENZY', 'BULLRUN', 'BULLRUN_SL')) or _ovr_rt):" in eng and 'in ("FIXED", "FLOOR", "FRENZY", "BULLRUN", "BULLRUN_SL")) or _ovr_m:' in eng   # realtime intercept + candle-loop branch
    assert '{"RUNNER_TRAIL": "MANUAL_TRAIL", "FRENZY_TP": "MANUAL_TP"}.get(_fz_why, "MANUAL_SL")' in eng and 'short=(direction == "SHORT")' in eng
    assert "getattr(order, 'manual_exit_mode', None) == \"FRENZY\")" in eng                                   # the 12 h cap applies
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('<option value="FRENZY">🔥 Frenzy exit</option>') == 1 and "m === 'MOMENTUM' || m === 'FRENZY'" in html
    # the leverage safety check: the 3 % stop is refused where the widest safe stop is tighter
    import asyncio, config as C, services.trading_engine as TE
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
    lev = next((L for L in (20, 25, 33, 50, 75, 100, 125) if TE.manual_floor_for_leverage(C.trading_config.thresholds, L) > -3.0), None)
    fn = getattr(TE.TradingEngine.open_manual_position, "__wrapped__", None)
    assert lev and fn is not None
    try:
        asyncio.run(fn(e, None, "FOOUSDT", "LONG", 100.0, float(lev), exit_mode="FRENZY"))
        raise AssertionError("expected the FRENZY stop to be refused at high leverage")
    except ValueError as err:
        assert "FRENZY stop" in str(err)
    # review: the candle path persists the peak (a restart must not disarm the trail) and carries the stop when the websocket is silent
    assert "order.peak_pnl = _mf_peak" in eng and '{"RUNNER_TRAIL": "MANUAL_TRAIL", "FRENZY_TP": "MANUAL_TP"}.get(_mf_why, "MANUAL_SL")' in eng


def test_shipped_atr_limit_is_2_5_everywhere():
    """DECISION_LOG 179: the ATR entry limit ships at 2.5 % — code default, JSON, rule fallback and the page default agree."""
    import config as C
    th = C.trading_config.thresholds
    assert type(th).model_fields["frenzy_max_atr_pct"].default == 2.5
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]["frenzy_max_atr_pct"] == 2.5

    class Bare:   # no field at all → the rule's own fallback
        frenzy_state_vol_mult = 100.0; frenzy_min_volume_usd = 20e6
    ep = frenzy_walk(_bars(after=24), NORM, TH)
    assert frenzy_long_status(ep, 2.5, 50e6, Bare)[0] is True and frenzy_long_status(ep, 2.51, 50e6, Bare)[2] == "ATR 2.51% > 2.5%"
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "['config-fz-max-atr', 'frenzy_max_atr_pct', 2.5]" in html and 'id="config-fz-max-atr"' in html


def test_long_skips_a_green_signal_candle_when_the_switch_is_on():
    """DECISION_LOG 180: the setup bar must have closed at or below its open (red / flat). Unreadable → refused (fail-closed)."""
    class On(TH):
        frenzy_long_skip_green_bar = True
    flat0 = _bars(after=24)                               # the fixture's bars open at their close (flat)
    b = [r[:] for r in flat0]; b[-1][1] = b[-1][4] * 0.995   # the setup bar opened 0.5 % lower → a green candle
    ep = frenzy_walk(b, NORM, On)
    assert ep["fresh_on"] and ep["bar_red"] is False and ep["bar_ret_pct"] > 0
    ok, code, text = frenzy_long_status(ep, 1.5, 50e6, On)
    assert (ok, code) == (False, "FRENZY_GREEN_BAR") and text.startswith("signal candle green (+")
    assert frenzy_long_status(ep, 1.5, 50e6, TH)[0] is True                      # switch off → enters after any candle
    red = [r[:] for r in b]; red[-1][1] = red[-1][4] * 1.001                    # same close, opened higher → a red candle still above the VWAP
    ep2 = frenzy_walk(red, NORM, On)
    assert ep2["fresh_on"] and ep2["bar_red"] is True and ep2["bar_ret_pct"] < 0 and frenzy_long_status(ep2, 1.5, 50e6, On) == (True, "FRENZY_READY", "READY")
    assert frenzy_long_status(frenzy_walk(flat0, NORM, On), 1.5, 50e6, On)[0] is True   # close == open counts as flat → allowed
    assert frenzy_long_status(dict(ep2, bar_red=None, bar_ret_pct=None), 1.5, 50e6, On)[1] == "FRENZY_GREEN_BAR"   # unreadable → refused
    assert frenzy_long_status(ep, 9.0, 50e6, On)[1] == "FRENZY_ATR_HIGH"         # the ATR refusal is reported first
    bad = [r[:] for r in flat0]; bad[-1][1] = "x"                                # an unreadable open keeps the flag and only refuses the entry
    epb = frenzy_walk(bad, NORM, On)
    assert epb is not None and frenzy_flagged(epb, On) and epb["bar_red"] is False and frenzy_long_status(epb, 1.5, 50e6, On)[2] == "signal candle unreadable"

    class Missing:   # no field at all → armed
        frenzy_state_vol_mult = 100.0; frenzy_min_volume_usd = 20e6; frenzy_max_atr_pct = 2.5
    assert frenzy_long_status(ep, 1.5, 50e6, Missing)[1] == "FRENZY_GREEN_BAR"
    import config as C
    assert type(C.trading_config.thresholds).model_fields["frenzy_long_skip_green_bar"].default is True
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]["frenzy_long_skip_green_bar"] is True
