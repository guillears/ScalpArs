"""🔓 2026-10-21 (DECISION_LOG 340, operator: "we should revert it asap") — frenzy_willy_global_hold: the FRENZY_WILLY GLOBAL HOLD (DL 251)
behind a switch. OFF (trading_config.json) → no automated open is refused / counted / recorded because a WILLY is open; every WILLY-own guard
stays (own slots — a pending entry stays armed while they are full —, pair held, lateness / dislocation, turnover) and open_position's
one-position-per-pair rule still keeps every other sleeve off the WILLY's pair. ON → the shipped hold, unchanged (tests/test_frenzy_willy.py,
marked willy_hold, pin it ON). Paper = live: the switch is read the same in both modes."""
import asyncio
import datetime as dt
import json
import os
import sys
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_frenzy_willy import (BAR_OPEN, LAST, SLEEVES, _engine as _willy_engine, _hold_db, _run as _willy_run,  # noqa: E402
                                     _th as _willy_th)


def _off(monkeypatch, TE, on=False):
    monkeypatch.setattr(TE.TradingEngine, "_willy_global_hold_on", lambda self: on)


# ── config / UI wiring (D11) ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_config_default_json_and_ui_wiring():
    import config as C
    mf = type(C.trading_config.thresholds).model_fields
    assert mf["frenzy_willy_global_hold"].default is True                                          # code default = the shipped semantics
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_willy_global_hold"] is False                                                # operator 2026-10-21: OFF
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 340" in src and "SCOUT_WILLY_HOLD.csv" in src and "10 % of the account" in src   # evidence + revert gate
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-fz-willy-global-hold"') == 1
    assert "WILLY global hold (block all other bot entries while a WILLY is open)" in html
    assert ("getElementById('config-fz-willy-global-hold'); if (_e) _e.checked = config.thresholds.frenzy_willy_global_hold !== false;"
            in html)                                                                               # load (missing key = ON)
    assert "frenzy_willy_global_hold: document.getElementById('config-fz-willy-global-hold')?.checked ?? true," in html   # save
    assert "${m.willy_global_hold === false ? 'global hold OFF" in html                            # monitor line (dashboard + both exports)
    assert "${_bt.frenzy_willy_global_hold === false ? 'global hold OFF" in html                   # config block (both exports)
    assert html.count("lines.push(...frenzyReportLines(perf, hr2));") == 2
    assert html.count("lines.push(..._buildConfigLines(cfg, changelog, hr, hr2, status));") == 2
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "\"willy_global_hold\": bool(getattr(_th, 'frenzy_willy_global_hold', True))," in main


def test_switch_reader_defaults_on(monkeypatch):
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=NS()))
    assert e._willy_global_hold_on() is True                                                       # key missing → ON (shipped semantics)
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=NS(frenzy_willy_global_hold=False)))
    assert e._willy_global_hold_on() is False
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=NS(frenzy_willy_global_hold=True)))
    assert e._willy_global_hold_on() is True
    monkeypatch.setattr(TE.config, "trading_config", None)                                          # unreadable → ON
    assert e._willy_global_hold_on() is True


# ── the choke point ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_off_never_blocks_counts_or_records_any_sleeve(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    _off(monkeypatch, TE)

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=True)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks = []; reads = []
        e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
        _real = e._willy_hold_state

        async def spy(db=None):
            reads.append(1); return await _real(db)
        e._willy_hold_state = spy
        res = []
        async with SL() as db:
            for sl in SLEEVES:
                d = "SHORT" if sl in ("SPIKE_FADE", "BEARRUN_SHORT", "SURGE_SHORT") else "LONG"
                res.append(await e._willy_hold_block(db, "FOOUSDT", d, sl, price=1.0, signal_ms=BAR_OPEN))
            await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        async with SL() as s_:
            rows = (await s_.execute(select(models.WillyHoldBlock))).scalars().all()
        await eng.dispose()
        return res, blocks, reads, rows, getattr(e, "_wh_events", None)
    res, blocks, reads, rows, ev = asyncio.run(go())
    assert not any(res) and blocks == [] and reads == [] and rows == [] and not ev


def test_on_still_blocks(monkeypatch):
    import services.trading_engine as TE
    _off(monkeypatch, TE, on=True)

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=True)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks = []
        e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
        async with SL() as db:
            r = [await e._willy_hold_block(db, "FOOUSDT", d, sl) for sl, d in (("FRENZY_LITE", "LONG"), ("MOMENTUM", "LONG"), ("SPIKE_FADE", "SHORT"))]
            await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        await eng.dispose()
        return r, blocks
    r, blocks = asyncio.run(go())
    assert all(r) and blocks == ["FRENZY_WILLY_HOLD"] * 3


def _offline(monkeypatch, TE):
    """review (340): the open path never reaches the network — every exchange read it makes is stubbed, and any aiohttp session or ccxt
    exchange construction fails the test."""
    import aiohttp
    bs = TE.binance_service

    async def _none(*a, **k):
        return None

    async def _empty(*a, **k):
        return {}

    async def _tick(*a, **k):
        return 0.0001
    for n, f in (("load_markets", _none), ("load_public_markets", _none), ("get_leverage_brackets", _empty), ("get_tick_size", _tick),
                 ("get_ohlcv", _none), ("fetch_orderbook", _none), ("fetch_orderbook_depth", _none), ("get_current_price", _none)):
        monkeypatch.setattr(bs, n, f, raising=False)

    net = []

    def _no_net(*a, **k):   # recorded too: a caller that swallows the exception cannot hide it
        net.append(1)
        raise AssertionError("network: an aiohttp session was created in an offline test")
    monkeypatch.setattr(aiohttp.ClientSession, "__init__", _no_net)
    monkeypatch.setattr(TE.config.trading_config, "maker_entry_enabled", False)   # the paper maker wait polls the book for 15–45 s
    return net


def _e2e(monkeypatch, pair, direction="LONG", on=False, **kw):
    """the REAL open_position on real SQLite with an open FRENZY_WILLY on WWWUSDT."""
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    _off(monkeypatch, TE, on=on)
    net = _offline(monkeypatch, TE)

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=True)()
        e = TE.TradingEngine(); e.is_running = True; e.is_paper_mode = True
        blocks = []
        _orig = e._record_filter_block
        e._record_filter_block = lambda n, d, had_room=True: (blocks.append(n), _orig(n, d, had_room))

        async def bal(db):
            return 5000.0
        e.get_available_balance = bal
        async with SL() as db:
            r = await e.open_position(db=db, pair=pair, direction=direction, confidence="STRONG_BUY", current_price=1.0, **kw)
        await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        async with SL() as s_:
            rows = (await s_.execute(select(models.WillyHoldBlock))).scalars().all()
            opened = (await s_.execute(select(models.Order).where(models.Order.status == "OPEN"))).scalars().all()
        await eng.dispose()
        return r, blocks, rows, [(o.pair, o.entry_strategy) for o in opened]
    out = asyncio.run(go())
    assert net == []                                                                               # offline: no aiohttp session at all
    return out


def test_e2e_off_momentum_long_opens_next_to_a_willy(monkeypatch):
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT")
    assert r is not None and "FRENZY_WILLY_HOLD" not in blocks and rows == []
    assert sorted(opened) == [("FOOUSDT", "MOMENTUM"), ("WWWUSDT", "FRENZY_WILLY")]


def test_e2e_off_pair_held_still_refuses_the_willys_pair(monkeypatch):
    for d in ("LONG", "SHORT"):                                                                    # one-way mode would net them: never two rows
        r, blocks, rows, opened = _e2e(monkeypatch, "WWWUSDT", direction=d)
        assert r is None and "PAIR_HELD" in blocks and "FRENZY_WILLY_HOLD" not in blocks and rows == []
        assert opened == [("WWWUSDT", "FRENZY_WILLY")]


def test_e2e_on_still_refused(monkeypatch):
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", on=True)
    assert r is None and "FRENZY_WILLY_HOLD" in blocks and len(rows) == 1 and opened == [("WWWUSDT", "FRENZY_WILLY")]


def test_under_lock_recheck_and_catchup_stand_down_when_off():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "if self._willy_global_hold_on() and (await self._willy_hold_state(db))[0]:   # 🔓 (340)" in eng      # book-lock re-check
    assert "if status != FRENZY_CATCHUP_STALE and self._willy_global_hold_on():" in eng                           # catch-up deferral
    assert ("_held = (await self._willy_hold_state(db))[0] if self._willy_global_hold_on() else await self._willy_slots_full(db)"
            in eng)                                                                                                # WILLY red-candle quiet check
    # every remaining _willy_hold_state caller is behind the switch (the block itself checks it first)
    i = eng.index("    async def _willy_hold_block("); blk = eng[i:eng.index("    def _willy_hold_record(", i)]
    assert blk.index("if not self._willy_global_hold_on():") < blk.index("await self._willy_hold_state(db)")
    assert eng.count("await self._willy_hold_state(db)") == 4


# ── FRENZY sleeves: _frenzy_open / LITE / LITE_RETRY / WILLY own guards ───────────────────────────────────────────────────────────

def test_frenzy_open_off_lite_retry_and_long_not_refused_for_the_hold(monkeypatch):
    import services.trading_engine as TE
    _off(monkeypatch, TE)
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**12)
    monkeypatch.setattr(TE, "_current_btc_trend_gap_pct", 0.5)
    monkeypatch.setattr(TE, "_current_btc_1d_ret_pct", 1.0)

    async def go(kw):
        eng, SL = await _hold_db(monkeypatch, open_willy=True)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks = []; opened = []
        e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
        e._flip_entry_fields = lambda *a, **k: {}
        e._sanitize_open_kwargs = lambda ef, s_, d: ef

        async def gvv(sig, wait=True):
            return 0.5
        e._frenzy_gvol_value = gvv

        async def fake_open(**k):
            opened.append(k); return object()
        e.open_position = fake_open
        flag = dict(pair="FOOUSDT", spike_ts=BAR_OPEN - 600_000, hours=5.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=40.0, run_pct=8.0, atr_pct=1.0,
                    volume_24h=5e7, price=1.0, live_price=1.0, adx_delta=1.0, di_spread=2.0, above_streak=14, above_share=90.0, bar_ret_pct=-0.1)
        async with SL() as db:
            r = await e._frenzy_open(db, flag, {}, BAR_OPEN, **kw)
        await eng.dispose()
        return r, blocks, opened, flag
    for kw in ({"lite": True}, {"lite_retry": dict(green=BAR_OPEN - 300_000, ret=0.4, exp=BAR_OPEN + 600_000)}, {}):
        r, blocks, opened, flag = asyncio.run(go(kw))
        assert "FRENZY_WILLY_HOLD" not in blocks and "_willy_hold" not in flag, (kw, blocks)
        assert "WILLY hold" not in str(flag.get("last_fire") or "") + str(flag.get("lite_retry_last") or ""), kw
        assert len(opened) == 1, (kw, blocks, flag.get("last_fire"), flag.get("lite_retry_last"))


def test_lite_eval_off_opens_while_a_willy_is_open(monkeypatch):
    import services.trading_engine as TE
    from tests.test_frenzy_lite import _ep, _th as _lth, _lite_runner, SID
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_lth(frenzy_willy_global_hold=False)))
    e, run = _lite_runner(TE)

    async def held(db=None):
        return True, 5, "WWWUSDT", "OPEN"
    e._willy_hold_state = held
    e._fz_lite_hold_first = {"FOOUSDT": (SID, LAST - 30 * 300_000)}                              # a stale marker from before the switch
    calls, blocks, flag = run(_ep())
    assert len(calls) == 1 and blocks == [] and e._fz_lite_done["FOOUSDT"] == SID                  # judged + opened — never HOLD_EXPIRED
    assert "FOOUSDT" not in e._fz_lite_hold_first and "WILLY hold" not in str(flag.get("last_fire"))


def test_willy_pending_off_slots_full_stays_armed_then_opens_ignoring_the_hold(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_willy_th(frenzy_willy_global_hold=False)))
    e = _willy_engine(TE)
    e._held = True                                                                                 # a WILLY is open (the hold read would say held)
    full = {"v": True}

    async def slots_full(db):
        return full["v"]
    e._willy_slots_full = slots_full
    sp = LAST - 30 * 300_000
    e._open_result = "opened"
    o, f = _willy_run(e, dict(spike_ts=sp, bar_ret_pct=-0.2))                                       # red, own slots full → recorded once, stays armed
    assert len(o) == 1 and "blocked" in (o[0]["note"] or "") and e._fz_willy_pending["FOOUSDT"]["blocked"] is True
    o, f = _willy_run(e, dict(spike_ts=sp, bar_ret_pct=-0.2), prev=sp, bar_open=BAR_OPEN + 300_000)
    assert o == [] and "blocked (hold / slots full / pair held)" in f["willy_last"]
    full["v"] = False                                                                              # the other WILLY closed; the hold read still says held
    o, _ = _willy_run(e, dict(spike_ts=sp, bar_ret_pct=-0.1), prev=sp, bar_open=BAR_OPEN + 2 * 300_000)
    assert [x["willy"] for x in o] == ["A"] and "FOOUSDT" not in e._fz_willy_pending               # opened: the hold is not consulted when OFF


def test_willy_slots_full_reader():
    import services.trading_engine as TE
    import models
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    async def go(n_open, slots):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        out = None
        _old = TE.config.trading_config.thresholds.frenzy_willy_max_slots
        try:
            TE.config.trading_config.thresholds.frenzy_willy_max_slots = slots
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                for i in range(n_open):
                    db.add(models.Order(pair=f"W{i}USDT", direction="LONG", status="OPEN", is_paper=True, entry_strategy="FRENZY_WILLY",
                                        confidence="STRONG_BUY", entry_price=1.0, quantity=1.0, investment=1.0, leverage=10, notional_value=10.0,
                                        opened_at=dt.datetime(2026, 10, 21)))
                db.add(models.Order(pair="LUSDT", direction="LONG", status="OPEN", is_paper=True, entry_strategy="FRENZY_LITE", confidence="STRONG_BUY",
                                    entry_price=1.0, quantity=1.0, investment=1.0, leverage=10, notional_value=10.0, opened_at=dt.datetime(2026, 10, 21)))
                await db.commit()
                e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
                out = await e._willy_slots_full(db)
        finally:
            TE.config.trading_config.thresholds.frenzy_willy_max_slots = _old
            await eng.dispose()
        return out
    assert asyncio.run(go(0, 1)) is False and asyncio.run(go(1, 1)) is True and asyncio.run(go(1, 2)) is False and asyncio.run(go(1, 0)) is True

    class Bad:
        async def execute(self, *a, **k):
            raise RuntimeError("db down")
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
    assert asyncio.run(e._willy_slots_full(Bad())) is True                                         # unreadable → stays armed (fail-closed)


def test_e2e_open_position_hard_single_willy_guard(monkeypatch):
    """(340 review) open_position re-counts open FRENZY_WILLY rows before any exchange order and refuses at ≥ frenzy_willy_max_slots —
    with the switch OFF and ON alike (ON: the hold refuses first). Raising the slot cap lets a second WILLY through (the guard is the cap)."""
    import config as C
    th = C.trading_config.thresholds
    kw = dict(frenzy_willy=True, entry_frenzy_willy_trigger="A")
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", **kw)
    assert r is None and "FRENZY_WILLY_MAX_SLOTS" in blocks and "FRENZY_WILLY_HOLD" not in blocks and opened == [("WWWUSDT", "FRENZY_WILLY")]
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", on=True, **kw)
    assert r is None and "FRENZY_WILLY_HOLD" in blocks and opened == [("WWWUSDT", "FRENZY_WILLY")]
    monkeypatch.setattr(th, "frenzy_willy_max_slots", 2)
    r, blocks, rows, opened = _e2e(monkeypatch, "FOOUSDT", **kw)
    assert r is not None and "FRENZY_WILLY_MAX_SLOTS" not in blocks and sorted(opened) == [("FOOUSDT", "FRENZY_WILLY"), ("WWWUSDT", "FRENZY_WILLY")]
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("    async def open_position("); body = eng[i:eng.index("\n    async def ", i + 10)]
    g = body.index('if _wh_es == "FRENZY_WILLY":\n            _wslots')
    assert body.index("if await self._willy_hold_block(db, pair, direction, _wh_es") < g < body.index("result = await self._simulate_maker_entry_paper(")
    assert g < body.index("_book_release = await self._book_hold()")                              # before any exchange order / booking
