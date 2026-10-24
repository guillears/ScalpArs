"""🔁 2026-10-19 FRENZY_LITE_RETRY (operator DECLARED OVERRIDE, DECISION_LOG 316): a FRENZY_LITE stretch refused ONLY for a green signal candle
records ONE pending retry; the first red / flat closed 5m bar within frenzy_lite_retry_wait_minutes (15) of the green bar's close on which the
SAME stretch is still LITE-READY opens FRENZY_LITE_RETRY (own sleeve, 4×, LITE's exit, no sleeve slot limit). Pure rule + engine path + every
surface (D11 / D12)."""
import asyncio
import datetime as dt
import json
import os
import time
from types import SimpleNamespace as NS

import pytest

from services.frenzy import (BAR_MS, FRENZY_LITE_RETRY, FRENZY_LITE_FAMILY, LITE_RETRY_DROP, LITE_RETRY_OPEN, LITE_RETRY_WAIT,
                             frenzy_exit_for, frenzy_lite_c_tp, frenzy_lite_retry_config_errors, frenzy_lite_retry_gate,
                             frenzy_lite_retry_pending, frenzy_lite_retry_pending_clean, frenzy_lite_retry_step, frenzy_lite_retry_wait_ms,
                             frenzy_tp_for)
from tests.test_frenzy_lite import LAST, NEXT, NORM, SID, _DB, _engine, _th as _lth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIKE = LAST - 5 * 3_600_000
GREEN = LAST + BAR_MS                      # the green bar's CLOSE (the pass bar the refusal happened on)
MIN = 60_000


def _th(**kw):
    base = dict(frenzy_lite_retry_enabled=True, frenzy_lite_retry_wait_minutes=15, frenzy_lite_retry_invest_mult=1.0, frenzy_lite_retry_lev_mult=0.2)
    base.update(kw)
    return _lth(**base)


def _ep(**kw):
    base = dict(verified=True, in_state=False, fresh_on=False, above_hour=True, above_streak=14, vol_mult=63.0, hours=5.2, bar_red=True,
                bar_ret_pct=-0.05, last_bar_ts=LAST, atr_pct=1.0, spike_ts=SPIKE)
    base.update(kw)
    return base


def _later(k, **kw):
    """the k-th bar after the green bar, same stretch (streak grows by k)."""
    return _ep(**dict(dict(above_streak=14 + k, last_bar_ts=LAST + k * BAR_MS), **kw))


def _pending(th=None):
    return frenzy_lite_retry_pending(SID, SPIKE, GREEN, 0.21, th or _th())


# ── pure rule ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_pending_record_and_window():
    p = _pending()
    assert p == dict(sid=SID, spike=SPIKE, green=GREEN, exp=GREEN + 15 * MIN, ret=0.21)
    assert frenzy_lite_retry_wait_ms(_th(frenzy_lite_retry_wait_minutes=0)) == 15 * MIN                       # ≤ 0 → 15
    assert frenzy_lite_retry_wait_ms(_th(frenzy_lite_retry_wait_minutes=None)) == 15 * MIN
    assert frenzy_lite_retry_wait_ms(_th(frenzy_lite_retry_wait_minutes="x")) == 15 * MIN
    assert frenzy_lite_retry_pending(None, SPIKE, GREEN, 0.2, _th()) is None                                 # unreadable id: no retry
    assert frenzy_lite_retry_pending_clean(json.loads(json.dumps(p))) == p
    assert frenzy_lite_retry_pending_clean({"sid": "x"}) is None and frenzy_lite_retry_pending_clean([1]) is None


def test_first_red_or_flat_bar_within_15_min_opens():
    th, p = _th(), _pending()
    for k in (1, 2, 3):                                                                                       # closes at +5 / +10 / +15 min
        act, code, _ = frenzy_lite_retry_step(_later(k), th, 50e6, p, GREEN + k * BAR_MS)
        assert act == LITE_RETRY_OPEN and code == "FRENZY_LITE_READY", k
    act, _, _ = frenzy_lite_retry_step(_later(1, bar_red=True, bar_ret_pct=0.0), th, 50e6, p, GREEN + BAR_MS)   # flat = close ≤ open
    assert act == LITE_RETRY_OPEN


def test_green_and_non_ready_bars_are_skipped():
    th, p = _th(), _pending()
    assert frenzy_lite_retry_step(_later(1, bar_red=False, bar_ret_pct=0.3), th, 50e6, p, GREEN + BAR_MS)[:2] == (LITE_RETRY_WAIT, "FRENZY_LITE_GREEN_BAR")
    assert frenzy_lite_retry_step(_later(1, vol_mult=120.0), th, 50e6, p, GREEN + BAR_MS)[:2] == (LITE_RETRY_WAIT, "FRENZY_LITE_VOL_HIGH")
    assert frenzy_lite_retry_step(_later(1, in_state=True), th, 50e6, p, GREEN + BAR_MS)[:2] == (LITE_RETRY_WAIT, "FRENZY_LITE_IN_STATE")
    assert frenzy_lite_retry_step(_later(1), th, 5e6, p, GREEN + BAR_MS)[:2] == (LITE_RETRY_WAIT, "FRENZY_LITE_VOL24_LOW")
    assert frenzy_lite_retry_step(_ep(), th, 50e6, p, GREEN)[0] == LITE_RETRY_WAIT                           # the green bar itself never acts


def test_drops_expiry_break_end_window():
    th, p = _th(), _pending()
    assert frenzy_lite_retry_step(_later(4), th, 50e6, p, GREEN + 4 * BAR_MS)[:2] == (LITE_RETRY_DROP, "FRENZY_LITE_RETRY_EXPIRED")   # +20 min
    broken = _ep(above_streak=1, last_bar_ts=LAST + 2 * BAR_MS)                                                # a close below VWAP → new stretch
    assert frenzy_lite_retry_step(broken, th, 50e6, p, GREEN + 2 * BAR_MS)[:2] == (LITE_RETRY_DROP, "FRENZY_LITE_RETRY_BROKEN")
    assert frenzy_lite_retry_step(_later(1, above_streak=0), th, 50e6, p, GREEN + BAR_MS)[:2] == (LITE_RETRY_DROP, "FRENZY_LITE_RETRY_BROKEN")
    assert frenzy_lite_retry_step(_later(1, verified=False), th, 50e6, p, GREEN + BAR_MS)[:2] == (LITE_RETRY_DROP, "FRENZY_LITE_RETRY_ENDED")
    assert frenzy_lite_retry_step(_later(1, spike_ts=SPIKE + BAR_MS), th, 50e6, p, GREEN + BAR_MS)[0] == LITE_RETRY_DROP
    assert frenzy_lite_retry_step(_later(1, hours=17.0), th, 50e6, p, GREEN + BAR_MS)[:2] == (LITE_RETRY_DROP, "FRENZY_LITE_RETRY_WINDOW_OVER")
    assert frenzy_lite_retry_step(_later(1), _th(frenzy_lite_enabled=False), 50e6, p, GREEN + BAR_MS)[:2] == (LITE_RETRY_DROP, "FRENZY_LITE_RETRY_OFF")


def test_gate_text_frozen_reads():
    d = lambda i: dt.datetime(2026, 10, 20) + dt.timedelta(days=i)
    assert frenzy_lite_retry_gate([], [])[0] == "COLLECTING"
    assert frenzy_lite_retry_gate([(d(0), -3.1), (d(1), -3.0)], [])[0] == "OFF"                               # Σ −6.1 ≤ −6 early kill
    assert frenzy_lite_retry_gate([(d(i), (2.0 if i % 2 else -1.9)) for i in range(9)], [])[0] == "COLLECTING"
    assert frenzy_lite_retry_gate([(d(i), (1.0 if i < 4 else -0.9)) for i in range(10)], [])[0] == "OFF"     # first 10 avg < 0
    good = [(d(i // 2), (2.0 if i % 3 else -0.5)) for i in range(20)]                                       # 20 fills on 10 days, avg > 0
    assert frenzy_lite_retry_gate(good, [(d(1), 0.1)])[0] == "STEP_UP"
    assert frenzy_lite_retry_gate(good, [(d(1), 5.0)])[0] == "OFF"                                          # below live LITE same period
    assert frenzy_lite_retry_gate(good, [(d(-5), 0.1)])[0] == "OFF"                                         # no same-period LITE fill
    assert frenzy_lite_retry_gate([(d(0), 1.0)] * 20, [])[0] == "COLLECTING"                                 # 20 fills but 1 day


# ── engine: arm only on a GREEN_BAR refusal ────────────────────────────────────────────────────────────────────────────────────────

def _lite_eval(TE, ep, outcome="opened", vol24=50e6, e=None):
    blocks = []
    e = e or _engine(TE, blocks)
    e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
    e._frenzy_lite_prev_missed = lambda *a, **k: False

    async def fake_open(db, fl, ind, bar_open, wide=False, catchup_now=None, lite=False):
        fl["last_fire"] = f"{dt.datetime.utcfromtimestamp(bar_open / 1000):%m-%d %H:%M} LITE {outcome}"
    e._frenzy_open = fake_open

    async def done_db(db, pair, sid):
        return False
    e._frenzy_lite_done_db = done_db
    flag = dict(ep, pair="FOOUSDT", volume_24h=vol24, text="x")
    bars = [[LAST - (299 - i) * BAR_MS, 1, 1.01, 0.99, 1.0, 10.0] for i in range(300)]
    asyncio.run(e._frenzy_lite_eval(_DB(), "FOOUSDT", ep, flag, {}, bars, int(ep["last_bar_ts"]) + BAR_MS, norm_hour=NORM))
    return e, blocks


def test_pending_created_only_on_a_green_bar_refusal(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e, blocks = _lite_eval(TE, _ep(bar_red=False, bar_ret_pct=0.21))
    assert blocks == ["FRENZY_LITE_GREEN_BAR"] and e._fz_lite_done["FOOUSDT"] == SID                       # LITE stays judged (consumed)
    assert e._fz_lite_retry["FOOUSDT"] == _pending()
    for kw in (dict(outcome="refused: the pair already has an open position"), dict(outcome="opened"), dict(vol24=5e6)):
        e2, _ = _lite_eval(TE, _ep(), **kw)
        assert not (getattr(e2, "_fz_lite_retry", None) or {}), kw                                           # never on another refusal / a fill


def test_switch_off_no_pending_no_behaviour_change(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_lite_retry_enabled=False)))
    e, blocks = _lite_eval(TE, _ep(bar_red=False, bar_ret_pct=0.21))
    assert blocks == ["FRENZY_LITE_GREEN_BAR"] and not (getattr(e, "_fz_lite_retry", None) or {})
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_lth()))                                  # a config without the field = off
    e, _ = _lite_eval(TE, _ep(bar_red=False, bar_ret_pct=0.21))
    assert not (getattr(e, "_fz_lite_retry", None) or {})


# ── engine: the pending retry on later bars ────────────────────────────────────────────────────────────────────────────────────────

def _retry_runner(TE, th, open_result=True):
    blocks, calls = [], []
    e = _engine(TE, blocks)
    e._fz_lite_retry = {"FOOUSDT": _pending(th)}

    async def fake_open(db, fl, ind, bar_open, wide=False, catchup_now=None, lite=False, lite_retry=None, **k):
        calls.append(dict(bar=bar_open, lite_retry=lite_retry, lite=lite))
        fl["lite_retry_last"] = f"{dt.datetime.utcfromtimestamp(bar_open / 1000):%m-%d %H:%M} LITE RETRY " + ("opened" if open_result else "refused: x")
        return open_result
    e._frenzy_open = fake_open
    bars = [[LAST - (299 - i) * BAR_MS, 1, 1.01, 0.99, 1.0, 10.0] for i in range(300)]

    def step(ep, bar_close):
        del blocks[:]
        flag = dict(ep, pair="FOOUSDT", volume_24h=50e6)
        asyncio.run(e._frenzy_lite_retry_eval(_DB(), "FOOUSDT", ep, flag, {}, bars, bar_close))
        return list(blocks), flag
    return e, calls, step


def test_red_bar_opens_once_green_bars_skipped(monkeypatch):
    import services.trading_engine as TE
    th = _th(); monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th))
    e, calls, step = _retry_runner(TE, th)
    b, flag = step(_later(1, bar_red=False, bar_ret_pct=0.1), GREEN + BAR_MS)                                  # green again: skipped
    assert calls == [] and b == [] and "FOOUSDT" in e._fz_lite_retry and "waiting" in flag["lite_retry_last"]
    b, flag = step(_later(2), GREEN + 2 * BAR_MS)                                                              # first red bar
    assert len(calls) == 1 and calls[0]["lite_retry"] == _pending(th) and "FOOUSDT" not in e._fz_lite_retry
    assert "di_spread" in flag and "adx_delta" in flag                                                        # the stamps LITE gets
    step(_later(3), GREEN + 3 * BAR_MS)
    assert len(calls) == 1                                                                                     # first qualifying bar only


def test_expiry_and_break_drop_with_counters(monkeypatch):
    import services.trading_engine as TE
    th = _th(); monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th))
    e, calls, step = _retry_runner(TE, th)
    b, _ = step(_later(4), GREEN + 4 * BAR_MS)
    assert b == ["FRENZY_LITE_RETRY_EXPIRED"] and calls == [] and "FOOUSDT" not in e._fz_lite_retry
    e, calls, step = _retry_runner(TE, th)
    b, _ = step(_ep(above_streak=1, last_bar_ts=LAST + BAR_MS), GREEN + BAR_MS)
    assert b == ["FRENZY_LITE_RETRY_BROKEN"] and calls == [] and "FOOUSDT" not in e._fz_lite_retry


def test_refusal_at_the_red_bar_ends_it(monkeypatch):
    import services.trading_engine as TE
    th = _th(); monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th))
    e, calls, step = _retry_runner(TE, th, open_result=None)
    step(_later(1), GREEN + BAR_MS)
    assert len(calls) == 1 and "FOOUSDT" not in e._fz_lite_retry
    step(_later(2), GREEN + 2 * BAR_MS)
    assert len(calls) == 1                                                                                     # one retry per green refusal


def test_sweep_drops_expired_unflagged_and_switched_off(monkeypatch):
    import services.trading_engine as TE
    th = _th(); monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th))
    blocks = []; e = _engine(TE, blocks)
    e._fz_lite_retry = {"AUSDT": _pending(th), "BUSDT": _pending(th), "CUSDT": _pending(th)}
    monkeypatch.setattr(TE, "_frenzy_flags", {"AUSDT": {}, "CUSDT": {}})
    asyncio.run(e._frenzy_lite_retry_sweep(GREEN + BAR_MS, seen={"AUSDT", "BUSDT"}))
    assert set(e._fz_lite_retry) == {"AUSDT", "CUSDT"} and blocks == ["FRENZY_LITE_RETRY_ENDED"]             # C not read: waits for its clock
    asyncio.run(e._frenzy_lite_retry_sweep(GREEN + 4 * BAR_MS, seen=set()))
    assert e._fz_lite_retry == {} and blocks.count("FRENZY_LITE_RETRY_EXPIRED") == 2
    e._fz_lite_retry = {"AUSDT": _pending(th)}
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_lite_retry_enabled=False)))
    asyncio.run(e._frenzy_lite_retry_sweep(GREEN + BAR_MS, seen=set()))
    assert e._fz_lite_retry == {}


def test_pending_survives_a_restart(monkeypatch):
    import services.trading_engine as TE
    from tests.test_frenzy_lite import _mem_sl
    e = object.__new__(TE.TradingEngine)
    e._fz_lite_retry = {"FOOUSDT": _pending()}
    js = json.loads(e._frenzy_state_json())
    assert js["lite_retry"] == {"FOOUSDT": _pending()}

    async def go():
        eng, SL = await _mem_sl()(json.dumps(js)); monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
        e2 = object.__new__(TE.TradingEngine); await e2._frenzy_judged_get()
        await eng.dispose()
        return e2
    assert asyncio.run(go())._fz_lite_retry == {"FOOUSDT": _pending()}


# ── engine: _frenzy_open(lite_retry=…) — LITE's filters, no slot limit, own tag / size / stamps ──────────────────────────────────

def _open(monkeypatch, rows, flag_kw=None, th_kw=None, held=False, bearish=False, ok=True, lite_only=False):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, config as C
    import services.trading_engine as TE
    th = C.trading_config.thresholds
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**6)
    monkeypatch.setattr(TE, "_current_btc_trend_gap_pct", -0.05 if bearish else 0.10)
    monkeypatch.setattr(TE, "_current_btc_1d_ret_pct", -0.8 if bearish else 0.5)
    monkeypatch.setattr(TE, "_zone_stamps_at", time.time())
    monkeypatch.setattr(C.trading_config, "entry_zone_stamps_enabled", True, raising=False)
    for k, v in dict(dict(frenzy_gvol_max=0.0, frenzy_lite_max_slots=2, frenzy_max_entries_per_pair_day=3, frenzy_bearish_day_block=True,
                          frenzy_max_entry_dislocation_pct=1.0), **(th_kw or {})).items():
        monkeypatch.setattr(th, k, v, raising=False)
    # ONE clock read. Seeded fills sit on TODAY's UTC date (the day caps count per UTC day — "30 min ago" fell on yesterday 00:00-00:30 UTC),
    # and the bar is ≥ 00:05 so the seed at midnight stays before it (the "already entered this bar" guard checks opened_at > bar − 1 s)
    _ts = time.time()
    now = dt.datetime.utcfromtimestamp(_ts)
    _d0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    seed_at = max(now - dt.timedelta(minutes=30), _d0)
    bar_open = max(int(_ts // 300) * 300_000, int((_d0 - dt.datetime(1970, 1, 1)).total_seconds() * 1000) + 300_000)

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        try:
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                for pair, strat, status in rows:
                    db.add(models.Order(pair=pair, direction="LONG", status=status, is_paper=True, entry_strategy=strat, confidence="STRONG_BUY",
                                        entry_price=1.0, quantity=1.0, investment=1.0, leverage=4, notional_value=4.0, opened_at=seed_at))
                await db.commit()
                blocks, opened = [], []
                e = _engine(TE, blocks)

                async def gvv(sig, wait=True):
                    return 0.5
                e._frenzy_gvol_value = gvv

                async def hold(*a, **k):
                    return held
                e._willy_hold_block = hold

                async def fake_open(**kw):
                    opened.append(kw); return object() if ok else None
                e.open_position = fake_open
                flag = dict(dict(pair="FOOUSDT", spike_ts=bar_open - 7_200_000, hours=5.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=63.0, run_pct=20.0,
                                 atr_pct=1.0, volume_24h=5e7, price=1.0, live_price=1.0, adx_delta=1.0, di_spread=2.0, above_streak=16, above_share=90.0,
                                 bar_ret_pct=-0.1), **(flag_kw or {}))
                pend = dict(sid=1, spike=1, green=bar_open - 10 * MIN, exp=bar_open + 5 * MIN, ret=0.21)
                if lite_only:
                    r = await e._frenzy_open(db, flag, {}, bar_open, lite=True)
                else:
                    r = await e._frenzy_open(db, flag, {}, bar_open, lite_retry=pend)
        finally:
            await eng.dispose()
        return r, blocks, opened, flag
    return asyncio.run(run())


def test_open_tags_retry_with_stamps(monkeypatch):
    r, b, o, flag = _open(monkeypatch, [])
    assert r is True and b == [] and len(o) == 1
    kw = o[0]
    assert kw["frenzy_lite"] is True and kw["frenzy_lite_retry"] is True and kw["frenzy_long"] is False and kw["frenzy_strong"] is False
    assert kw["entry_frenzy_retry_wait_min"] == 10.0 and kw["entry_frenzy_retry_green_ret"] == 0.21
    assert "LITE RETRY opened" in flag["lite_retry_last"] and not flag.get("last_fire")                       # its own outcome field


def test_no_slot_limit_and_lite_slots_unaffected(monkeypatch):
    full = [("A", "FRENZY_LITE", "OPEN"), ("B", "FRENZY_LITE", "OPEN"), ("C", FRENZY_LITE_RETRY, "OPEN"), ("D", FRENZY_LITE_RETRY, "OPEN"),
            ("E", FRENZY_LITE_RETRY, "OPEN")]
    r, b, o, _ = _open(monkeypatch, full)
    assert r is True and b == [] and len(o) == 1                                                               # LITE's 2 slots full: the retry still opens
    r, b, o, _ = _open(monkeypatch, [("C", FRENZY_LITE_RETRY, "OPEN"), ("D", FRENZY_LITE_RETRY, "OPEN")], lite_only=True)
    assert b == [] and len(o) == 1 and o[0].get("frenzy_lite_retry") is False                                 # retries never take LITE's slots
    r, b, o, _ = _open(monkeypatch, [("A", "FRENZY_LITE", "OPEN"), ("B", "FRENZY_LITE", "OPEN")], lite_only=True)
    assert b == ["FRENZY_LITE_MAX_SLOTS"] and o == []                                                          # LITE's own cap unchanged


@pytest.mark.parametrize("case,kw,want", [
    ("pair held", dict(rows=[("FOOUSDT", "MOMENTUM", "OPEN")]), "FRENZY_LITE_RETRY_PAIR_HELD"),
    ("day cap", dict(rows=[("FOOUSDT", "FRENZY_LITE", "CLOSED")] * 2 + [("FOOUSDT", FRENZY_LITE_RETRY, "CLOSED")]), "FRENZY_LITE_RETRY_PAIR_DAY_CAP"),
    ("dislocation", dict(rows=[], flag_kw=dict(live_price=1.02)), "FRENZY_LITE_RETRY_DISLOC"),
    ("bearish", dict(rows=[], bearish=True), "FRENZY_LITE_RETRY_BEARISH_DAY"),
    ("open refused", dict(rows=[], ok=False), "FRENZY_LITE_RETRY_OPEN_REFUSED"),
])
def test_filters_rechecked_at_the_red_bar(monkeypatch, case, kw, want):
    r, b, o, flag = _open(monkeypatch, **kw)
    assert not r and b == [want] and (o == [] or case == "open refused"), case
    assert "LITE RETRY refused" in flag["lite_retry_last"]


def test_willy_hold_refuses(monkeypatch):
    r, b, o, _ = _open(monkeypatch, [], held=True)
    assert r is False and o == []


def test_retry_counts_toward_the_lite_day_cap(monkeypatch):
    rows = [("FOOUSDT", "FRENZY_LITE", "CLOSED"), ("FOOUSDT", FRENZY_LITE_RETRY, "CLOSED"), ("FOOUSDT", FRENZY_LITE_RETRY, "CLOSED")]
    r, b, o, _ = _open(monkeypatch, rows, lite_only=True)
    assert b == ["FRENZY_LITE_PAIR_DAY_CAP"] and o == []                                                      # one LITE-family count


def test_frenzy_long_refused_on_a_pair_held_by_the_retry(monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models
    import services.trading_engine as TE
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**6)
    bar_open = int(time.time() // 300) * 300_000

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with async_sessionmaker(eng, expire_on_commit=False)() as db:
            db.add(models.Order(pair="FOOUSDT", direction="LONG", status="OPEN", is_paper=True, entry_strategy=FRENZY_LITE_RETRY, confidence="STRONG_BUY",
                                entry_price=1.0, quantity=1.0, investment=1.0, leverage=4, notional_value=4.0, opened_at=dt.datetime.utcnow()))
            await db.commit()
            blocks = []; e = _engine(TE, blocks)

            async def no_open(**kw):
                raise AssertionError("must not open")
            e.open_position = no_open
            flag = dict(pair="FOOUSDT", spike_ts=bar_open - 7_200_000, hours=3.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=150.0, run_pct=20.0,
                        atr_pct=1.5, volume_24h=5e7, price=1.0, live_price=1.0, above_streak=20, bar_ret_pct=-0.1)
            await e._frenzy_open(db, flag, {}, bar_open)
        await eng.dispose()
        return blocks
    assert asyncio.run(run()) == ["FRENZY_PAIR_HELD_BY_LITE"]


# ── size / exit identical to LITE ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_size_resolves_to_its_own_fields():
    import config as C
    th = C.trading_config.thresholds
    pref = FRENZY_LITE_RETRY.lower()
    assert pref == "frenzy_lite_retry" and getattr(th, f"{pref}_lev_mult") == 0.2 and getattr(th, f"{pref}_invest_mult") == 1.0
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "_sg_pref = _fz_es.lower() if _frenzy else" in eng
    assert 'FRENZY_LITE_RETRY if (_frenzy and frenzy_lite and frenzy_lite_retry) else "FRENZY_LITE" if (_frenzy and frenzy_lite)' in eng


def test_lite_c_tp_and_exit_identical_to_lite():
    th = NS(frenzy_tp_pct=1.5, frenzy_lite_tp_pct=3.0, frenzy_lite_c_enabled=True, frenzy_lite_c_vol_mult=20.0, frenzy_lite_c_btc_slope_min=0.0,
            frenzy_lite_c_tp_pct=2.0, frenzy_stop_pct=3.0, frenzy_lock_arm_pct=0.0)
    assert frenzy_lite_c_tp(th, 25.0, 0.1) == (3.0, True) and frenzy_lite_c_tp(th, 10.0, 0.1) == (2.0, False)
    for tp_used in (2.0, 3.0, None):
        assert frenzy_tp_for(th, FRENZY_LITE_RETRY, tp_used) == frenzy_tp_for(th, "FRENZY_LITE", tp_used)
        for pnl, pk in ((2.1, 2.1), (-3.05, 0.5), (1.0, 1.2)):
            assert frenzy_exit_for(pnl, pk, th, None, use_tp=True, sleeve=FRENZY_LITE_RETRY, tp_override=tp_used) == \
                   frenzy_exit_for(pnl, pk, th, None, use_tp=True, sleeve="FRENZY_LITE", tp_override=tp_used)
    assert FRENZY_LITE_FAMILY == ("FRENZY_LITE", "FRENZY_LITE_RETRY")
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "if _frenzy and _fz_es in FRENZY_LITE_FAMILY:" in eng                                              # the LITE C decision at entry


def test_resting_tp_stop_backstop_and_strategy_sets():
    import services.trading_engine as TE
    from services.resting_tp import RTP_SLEEVES, rtp_target
    assert RTP_SLEEVES[FRENZY_LITE_RETRY] == "resting_tp_lite_retry_enabled"
    th = NS(resting_tp_lite_enabled=True, resting_tp_lite_retry_enabled=True, frenzy_tp_pct=1.5, frenzy_lite_tp_pct=3.0, frenzy_lock_arm_pct=0.0,
            frenzy_stop_pct=3.0, broker_backstop_pct=2.5)
    assert rtp_target(th, FRENZY_LITE_RETRY, None, 2.0) == rtp_target(th, "FRENZY_LITE", None, 2.0) == (2.0, "FRENZY_TP")
    assert rtp_target(NS(**dict(vars(th), resting_tp_lite_retry_enabled=False)), FRENZY_LITE_RETRY, None, 2.0) is None   # own switch
    assert TE.bot_own_stop_pct(th, FRENZY_LITE_RETRY) == TE.bot_own_stop_pct(th, "FRENZY_LITE") == 3.0
    assert TE.order_backstop_pct(th, FRENZY_LITE_RETRY, leverage=4) == TE.order_backstop_pct(th, "FRENZY_LITE", leverage=4)
    assert FRENZY_LITE_RETRY in TE.FRENZY_STRATEGIES                                                           # exit intercept / 12 h cap / urgent close


# ── D11 / D12 parity ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_config_d11():
    import config as C
    mf = C.SignalThresholds.model_fields
    assert mf["frenzy_lite_retry_enabled"].default is False and mf["frenzy_lite_retry_wait_minutes"].default == 15.0
    assert mf["frenzy_lite_retry_invest_mult"].default == 1.0 and mf["frenzy_lite_retry_lev_mult"].default == 0.2
    assert mf["resting_tp_lite_retry_enabled"].default is True
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_lite_retry_enabled"] is True and cfg["frenzy_lite_retry_wait_minutes"] == 15.0
    assert cfg["frenzy_lite_retry_invest_mult"] == 1.0 and cfg["frenzy_lite_retry_lev_mult"] == 0.2 and cfg["resting_tp_lite_retry_enabled"] is True
    assert "DECISION_LOG 316" in open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert frenzy_lite_retry_config_errors(NS(frenzy_lite_retry_enabled=True, frenzy_lite_retry_wait_minutes=15)) == []
    assert frenzy_lite_retry_config_errors(NS(frenzy_lite_retry_enabled="yes", frenzy_lite_retry_wait_minutes=120, frenzy_lite_retry_lev_mult=-1)) and \
        len(frenzy_lite_retry_config_errors(NS(frenzy_lite_retry_enabled="yes", frenzy_lite_retry_wait_minutes=120, frenzy_lite_retry_lev_mult=-1))) == 3
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "_cfg_errs.extend(_lr_cfg_errors(new_config.thresholds))" in main


def test_ui_inputs_load_save():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for _id in ("config-fz-lite-retry-enabled", "config-fz-lite-retry-invest-mult", "config-fz-lite-retry-lev-mult", "config-fz-lite-retry-wait",
                "config-rtp-lite-retry-enabled"):
        assert html.count(f'id="{_id}"') == 1, _id
    for row in ("['config-fz-lite-retry-wait', 'frenzy_lite_retry_wait_minutes', 15]", "['config-fz-lite-retry-invest-mult', 'frenzy_lite_retry_invest_mult', 1.0]",
                "['config-fz-lite-retry-lev-mult', 'frenzy_lite_retry_lev_mult', 0.2]"):
        assert html.count(row) == 1, row                                                                       # FRENZY_NUM_FIELDS: load + save
    assert "frenzy_lite_retry_enabled: document.getElementById('config-fz-lite-retry-enabled')?.checked ?? false" in html
    assert "getElementById('config-fz-lite-retry-enabled'); if (_e) _e.checked = config.thresholds.frenzy_lite_retry_enabled === true" in html
    assert "resting_tp_lite_retry_enabled: document.getElementById('config-rtp-lite-retry-enabled')?.checked ?? true" in html
    assert "getElementById('config-rtp-lite-retry-enabled'); if (_e) _e.checked = config.thresholds.resting_tp_lite_retry_enabled !== false" in html
    assert html.count('data-ss="fzlr"') == 1 and "S.fzlr = {" in html


def test_reports_d12():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    i = html.index("function frenzyReportLines(perf, hr2)"); body = html[i:html.index("\n        }\n", i)]
    assert "FRENZY_LITE_RETRY" in body and "lite_retry_last" in body                                         # the shared export block
    assert html.count("lines.push(...frenzyReportLines(perf, hr2));") == 2                                    # clipboard + saved file
    assert html.count("lines.push(...liteCReportLines(perf, hr2));") == 2
    assert html.count("FRENZY Fills (FRENZY_LONG · FRENZY_WIDE · FRENZY_LITE · FRENZY_WILLY) · 🔁 FRENZY_LITE_RETRY") == 2   # UI + export
    assert "LITE RETRY ${m.lite_retry_enabled ? 'ON' : 'OFF'}" in html and "LITE RETRY ${_bt.frenzy_lite_retry_enabled === true ? 'ON' : 'OFF'}" in html
    assert "LITE RETRY ${_bt.resting_tp_lite_retry_enabled !== false ? 'ON' : 'OFF'}" in html
    assert '<option value="FRENZY_LITE_RETRY">' in html and "strat === 'FRENZY_LITE_RETRY'" in html
    assert "LITE RETRY opened/.exec((f && f.lite_retry_last)" in html and "(o.entry_strategy || '') === 'FRENZY_LITE_RETRY'" in html
    import models
    cols = [c.name for c in models.Order.__table__.columns]
    assert "entry_frenzy_retry_wait_min" in cols and "entry_frenzy_retry_green_ret" in cols                    # the orders CSV carries them
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "'entry_frenzy_retry_wait_min', 'entry_frenzy_retry_green_ret'" in db                              # migration


def test_sleeve_and_strategy_rows():
    import main as M
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "'Frenzy-Lite', 'Frenzy-Lite-Retry', 'Frenzy-Willy', 'Frenzy-Willy-HT', 'Flatbase-Short', 'Manual']" in src   # 🩳 (348) Flatbase-Short   # 🎲¼ (341) + HT after WILLY                            # own sleeve row, right after LITE
    rows = M._compute_sleeve_performance([NS(entry_strategy=FRENZY_LITE_RETRY, direction="LONG", pnl_percentage=1.0, pnl=10.0,
                                             opened_at=dt.datetime(2026, 10, 19), closed_at=dt.datetime(2026, 10, 19, 1)),
                                          NS(entry_strategy="FRENZY_LITE", direction="LONG", pnl_percentage=-1.0, pnl=-10.0,
                                             opened_at=dt.datetime(2026, 10, 19), closed_at=dt.datetime(2026, 10, 19, 1))]) \
        if hasattr(M, "_compute_sleeve_performance") else None
    if rows is not None:
        names = [r["sleeve"] for r in rows]
        assert names.index("Frenzy-Lite") + 1 == names.index("Frenzy-Lite-Retry")
    st = M._compute_strategy_performance([NS(entry_strategy=FRENZY_LITE_RETRY, direction="LONG", pnl_percentage=1.0, pnl=10.0,
                                             opened_at=dt.datetime(2026, 10, 19), closed_at=dt.datetime(2026, 10, 19, 1))])
    assert st[0]["strategy"] == "FRENZY_LITE_RETRY · L"
    assert "'FRENZY_LITE', 'FRENZY_LITE_RETRY', 'FRENZY_WILLY']" in src                                       # strategy table rank
    assert M._SLEEVE_LABELS[(FRENZY_LITE_RETRY, "LONG")] == "FRENZY LITE RETRY Long"
    assert any("RETRY" in str(r) for r in M._open_by_sleeve([NS(entry_strategy=FRENZY_LITE_RETRY, direction="LONG")]))
    assert '("FRENZY_LITE_RETRY", "FRENZY_LITE_RETRY (LITE green refusal' in src and "_lrg((await db.execute(" in src   # FRENZY Fills row + gate
    assert '("FRENZY_LITE_RETRY", "RETRY · ")' in src                                                       # LITE C tally rows


def test_master_builder_and_scout_note():
    src = open(os.path.join(ROOT, "scripts", "build_master_pool.py"), encoding="utf-8").read()
    assert "elif strat == 'FRENZY_LITE_RETRY':\n                why = 'FRENZY_SLEEVE'" in src and '+ ("FRENZY_LITE_RETRY",)' in src
    assert 'if strat == "FRENZY_LITE_RETRY":' in src
    sc = open(os.path.join(ROOT, "scripts", "scout_lite_green_retry.py"), encoding="utf-8").read()
    assert "live FRENZY_LITE_RETRY fills now exist" in sc and 'GATE_N, GATE_DAYS, P_MIN, SHARE_MAX = 20, 10, 0.90, 0.50' in sc


def test_pass_wiring_and_reset():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("async def _update_frenzy_pass(self, db):"); body = eng[i:eng.index("def _frenzy_wide_off_note(", i)]
    assert "await self._frenzy_lite_retry_eval(db, pair, ep, flag, ind, closed, bar_open)" in body                 # after LITE, every flagged pair
    assert body.index("self._frenzy_lite_eval(") < body.index("self._frenzy_lite_retry_eval(") < body.index("self._frenzy_willy_eval(")
    assert "await self._frenzy_lite_retry_sweep(bar_open, seen)" in body
    assert 'if code == "FRENZY_LITE_GREEN_BAR":   # 🔁 (316)' in eng and "await self._frenzy_lite_retry_arm(pair, ep, sid, bar_open)" in eng
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "'_fz_lite_hold_first', '_fz_lite_retry')" in main                                                    # a LONG reset clears pending retries
    assert '"lite_retry_pending": [dict(pair=k' in main and '"lite_retry_last": f.get(\'lite_retry_last\')' in main


# ── dual review ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_main_read_is_the_shortest_prefix_with_20_fills_on_10_days():
    d = lambda i: dt.datetime(2026, 10, 20) + dt.timedelta(days=i)
    burst = [(d(i // 4), 1.0) for i in range(20)]                                                              # 20 fills on 5 days
    assert frenzy_lite_retry_gate(burst, [(d(0), 0.1)])[0] == "COLLECTING"
    more = burst + [(d(5 + i), 1.0) for i in range(4)]                                                         # 9 days → still collecting
    assert frenzy_lite_retry_gate(more, [(d(0), 0.1)])[0] == "COLLECTING"
    done = more + [(d(9), 1.0)]                                                                                # the 10th day → a verdict
    v, txt = frenzy_lite_retry_gate(done, [(d(0), 0.1)])
    assert v == "STEP_UP" and "first 25 fills" in txt
    assert frenzy_lite_retry_gate(done + [(d(10), -50.0)], [(d(0), 0.1)])[0] == "OFF"                         # Σ kill still wins later


def test_drop_hygiene(monkeypatch):
    import services.trading_engine as TE
    th = _th(frenzy_lite_enabled=False); monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th))
    e, calls, step = _retry_runner(TE, th)
    monkeypatch.setitem(TE._frenzy_flags, "FOOUSDT", {"lite_retry_pending": {"exp_ms": 1}})
    b, _ = step(_later(1), GREEN + BAR_MS)
    assert b == [] and "FOOUSDT" not in e._fz_lite_retry                                                       # LITE off: dropped, not counted
    assert "lite_retry_pending" not in TE._frenzy_flags["FOOUSDT"]                                            # no stale monitor text
    th2 = _th(); monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th2))
    e, calls, step = _retry_runner(TE, th2)

    async def held_open(db, fl, ind, bar_open, lite_retry=None, **k):
        fl["_willy_hold"] = True
        return False
    e._frenzy_open = held_open
    _, flag = step(_later(1), GREEN + BAR_MS)
    assert "_willy_hold" not in flag and "FOOUSDT" not in e._fz_lite_retry


def test_family_checks_include_the_retry():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert 'Order.entry_strategy.in_(FRENZY_LITE_FAMILY),   # (dual review) + the retry' in eng                   # LITE judged-stretch DB backstop
    assert 'Order.entry_strategy.in_(("FRENZY_LONG", "FRENZY_WIDE") + FRENZY_LITE_FAMILY)' in eng               # WILLY B's question
    assert "this pending-retry change is NOT persisted now" in eng
