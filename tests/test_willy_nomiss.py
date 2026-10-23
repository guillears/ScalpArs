"""🧭 2026-10-23 R_NOMISS (DECISION_LOG 350, operator-approved fix — BTRUSDT 10-22): WILLY entry A for a cold pair read PAST the late window
(frenzy_willy_a_cold_max_bars) is allowed iff NO red 5m bar (close < open, strictly — frenzy_willy_red's rule; flat is not red) closed from
the spike bar up to the bar just read. Armed exactly as the on-time A (armed = the spike close, exp = spike + red wait); the 24 h-volume floor
is re-checked at the red bar; turnover routing (WILLY / WILLY_HT), hold, slots, pair held, lateness, dislocation unchanged; one A per episode;
restart-safe. Fills stamp Order.entry_frenzy_willy_nomiss. Real engine paths (_frenzy_willy_eval / _frenzy_open / state persistence); every
exchange read is stubbed — no network. Bar times are fixed (no wall-clock dependence)."""
import asyncio
import datetime as dt
import json
import os
import sys
from types import SimpleNamespace as NS

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.frenzy import BAR_MS, frenzy_willy_no_red_missed, frenzy_willy_a_nomiss_on, frenzy_willy_pending_clean  # noqa: E402
from tests.test_frenzy_willy import _DB, _mem, _th  # noqa: E402

# BTRUSDT 10-22 shape: the spike bar closed at SP (17:30 in the log); every bar after it green until the first red bar, which is the bar
# just closed when the pair is first read (18:05 → 7 bars after the spike). bar_open = the forming bar's open = the close of the bar just read.
SP = 1_790_000_000_000 // BAR_MS * BAR_MS
LATE = 7
READ = SP + LATE * BAR_MS
OLD_MSG = "already flagged before this process judged the pair on the previous bar"


def _bars(bar_open, colours, pre=40):
    """closed 5m rows ending at the bar just judged (open = bar_open − BAR_MS). colours = one char per bar from the spike bar (open SP −
    BAR_MS) to the bar just judged: 'g' green · 'r' red · 'f' flat. `pre` RED bars before the spike: they never count."""
    first = SP - BAR_MS
    assert first + (len(colours) - 1) * BAR_MS == bar_open - BAR_MS
    rows = [[first - (pre - k) * BAR_MS, 1.0, 1.0, 0.9, 0.95, 10.0] for k in range(pre)]
    for k, c in enumerate(colours):
        o, cl = (1.0, 1.02) if c == "g" else (1.0, 0.99) if c == "r" else (1.0, 1.0)
        rows.append([first + k * BAR_MS, o, max(o, cl), min(o, cl), cl, 10.0])
    return rows


def _engine(TE):
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
    e.blocks, e.opens, e.marks = [], [], []
    e._record_filter_block = lambda name, d, had_room=True: e.blocks.append(name)
    e._fz_judged_loaded = True
    e._done = {"A": False, "B": False}

    async def fake_open(db, fl, ind, bar_open, willy=None, willy_note=None, willy_wait_bars=None, willy_ht=False, **kw):
        # what the real _frenzy_open stamps: the pending entry's nomiss tag (it is still pending while the open path runs)
        pd_ = (e.__dict__.get('_fz_willy_pending') or {}).get(fl['pair']) or {}
        e.opens.append(dict(willy=willy, ht=willy_ht, wait=willy_wait_bars, bar=bar_open, nomiss=bool(pd_.get('nomiss'))))
        return True
    e._frenzy_open = fake_open

    async def mark(pair, sp):
        e.marks.append((pair, sp)); e.__dict__.setdefault('_fz_willy_seen', {})[pair] = sp
    e._frenzy_willy_mark = mark

    async def save():
        pass
    e._frenzy_willy_save = save

    async def opened(db, pair, bar_open):
        return False
    e._frenzy_willy_sleeve_opened = opened

    async def done(db, pair, sp, trig):
        return e._done[trig]
    e._frenzy_willy_done_db = done

    async def free(db=None):
        return False, None, None, None
    e._willy_hold_state = free

    async def slots(db):
        return False
    e._willy_slots_full = slots
    e._willy_ht_slots_full = slots
    return e


def _run(e, colours, bar_open=READ, prev=None, warm=False, vol=5e7, closed=True):
    del e.opens[:]
    ret = {"g": 0.4, "r": -0.3, "f": 0.0}[colours[-1]]
    ep = dict(spike_ts=SP, bar_ret_pct=ret)
    flag = dict(ep, pair="BTRUSDT", volume_24h=vol, price=1.0, live_price=1.0)
    asyncio.run(e._frenzy_willy_eval(_DB(), "BTRUSDT", ep, flag, {}, bar_open, prev, True,
                                     dict(on=False, ready=False, code="FRENZY_ON", text="x"), pair_warm=warm,
                                     closed=(_bars(bar_open, colours) if closed else None)))
    return list(e.opens), flag


def _cfg(monkeypatch, TE, **kw):
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(**kw)))


def _ev(e):
    return dict(getattr(e, "_willy_event_counts", None) or {})


# ── the pure rule ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_pure_no_red_missed():
    g7 = "g" * LATE                                                                  # spike bar + 6 greens before the bar just judged
    assert frenzy_willy_no_red_missed(_bars(READ, g7 + "r"), SP, READ) is True        # BTR: first red = the read bar
    assert frenzy_willy_no_red_missed(_bars(READ, g7 + "g"), SP, READ) is True        # the read bar itself is not checked here
    assert frenzy_willy_no_red_missed(_bars(READ, "ggrgggg" + "r"), SP, READ) is False   # a red bar already closed before the read
    assert frenzy_willy_no_red_missed(_bars(READ, "r" + "g" * 6 + "r"), SP, READ) is False   # a red SPIKE bar: an on-time A opens on it
    assert frenzy_willy_no_red_missed(_bars(READ, "gfgfggg" + "r"), SP, READ) is True    # flat is NOT red (WILLY's strict rule)
    rows = _bars(READ, g7 + "r")
    assert frenzy_willy_no_red_missed([r for r in rows if r[0] != SP + 2 * BAR_MS], SP, READ) is None   # a gap → unreadable
    assert frenzy_willy_no_red_missed(rows[:-1], SP, READ) is None                   # not ending at the bar just judged
    bad = [list(r) for r in rows]; bad[-3][1] = 0.0
    assert frenzy_willy_no_red_missed(bad, SP, READ) is None                         # a bad open
    bad[-3][1] = float("nan")
    assert frenzy_willy_no_red_missed(bad, SP, READ) is None
    assert frenzy_willy_no_red_missed(None, SP, READ) is None and frenzy_willy_no_red_missed(rows, None, READ) is None
    assert frenzy_willy_no_red_missed(_bars(SP, "r"), SP, SP) is True                # on-time: nothing before the spike bar counts
    assert frenzy_willy_a_nomiss_on(NS()) is True and frenzy_willy_a_nomiss_on(NS(frenzy_willy_a_nomiss=False)) is False
    assert frenzy_willy_a_nomiss_on(NS(frenzy_willy_a_nomiss=None)) is True


# ── engine: allowed / refused ──────────────────────────────────────────────────────────────────────────────────────────────────────

def test_btr_shape_opens_at_the_read(monkeypatch, caplog):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)
    with caplog.at_level("INFO", logger=TE.logger.name):
        o, f = _run(e, "g" * LATE + "r")
    assert [(x["willy"], x["ht"], x["nomiss"], x["wait"]) for x in o] == [("A", False, True, LATE)]   # wait counted from the spike (on-time)
    assert f"[FRENZY_WILLY] BTRUSDT: read {LATE} bars after the spike but no red bar missed — entry A allowed (R_NOMISS)" in caplog.text
    assert OLD_MSG not in caplog.text and e.blocks == [] and _ev(e).get("FRENZY_WILLY_A_NOMISS") == 1
    assert e.marks == [("BTRUSDT", SP)]


def test_green_at_the_read_arms_like_on_time_then_opens_on_the_first_red(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)
    o, _ = _run(e, "g" * LATE + "g")
    pd_ = e._fz_willy_pending["BTRUSDT"]
    assert o == [] and pd_["armed"] == SP and pd_["exp"] == SP + 3_600_000 and pd_["nomiss"] is True and pd_["trig"] == "A"
    assert frenzy_willy_pending_clean(pd_)["nomiss"] is True                         # the tag survives the persisted round trip
    o, _ = _run(e, "g" * (LATE + 1) + "r", bar_open=READ + BAR_MS, prev=SP, warm=True)
    assert [(x["willy"], x["nomiss"], x["wait"]) for x in o] == [("A", True, LATE + 1)] and "BTRUSDT" not in e._fz_willy_pending


def test_a_red_bar_before_the_read_is_refused_with_the_old_message(monkeypatch, caplog):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)
    with caplog.at_level("INFO", logger=TE.logger.name):
        o, _ = _run(e, "ggrgggg" + "r")
    assert o == [] and not getattr(e, "_fz_willy_pending", {})
    assert OLD_MSG in caplog.text and f"{LATE} bars old > the 3-bar late window — no entry A" in caplog.text
    assert _ev(e).get("FRENZY_WILLY_A_NOMISS_MISSED") == 1 and "FRENZY_WILLY_A_NOMISS" not in _ev(e)
    assert _run(e, "ggrgggg" + "r" + "r", bar_open=READ + BAR_MS, prev=SP, warm=True)[0] == []   # seen: never later either


def test_unreadable_bars_refused_as_before(monkeypatch, caplog):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)
    with caplog.at_level("INFO", logger=TE.logger.name):
        o, _ = _run(e, "g" * LATE + "r", closed=False)                               # no bars handed in → fail-closed
    assert o == [] and OLD_MSG in caplog.text and not _ev(e)


def test_first_red_later_than_the_wait_from_the_spike_is_refused(monkeypatch, caplog):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)                                                                  # read 13 bars (65 min) after the spike, all green
    with caplog.at_level("INFO", logger=TE.logger.name):
        o, _ = _run(e, "g" * 13 + "r", bar_open=SP + 13 * BAR_MS)
    assert o == [] and OLD_MSG in caplog.text and _ev(e).get("FRENZY_WILLY_A_NOMISS_LATE") == 1
    e = _engine(TE)                                                                  # read in time, green, then green past 60 min → expires
    assert _run(e, "g" * LATE + "g")[0] == []
    o, _ = _run(e, "g" * 13 + "r", bar_open=SP + 13 * BAR_MS, prev=SP, warm=True)
    assert o == [] and e.blocks == ["FRENZY_WILLY_RED_EXPIRED"] and "BTRUSDT" not in e._fz_willy_pending
    e = _engine(TE)                                                                  # exactly at the wait edge (on-time rule) → opens
    assert _run(e, "g" * 12 + "r", bar_open=SP + 12 * BAR_MS)[0][0]["nomiss"] is True


def test_volume_floor_rechecked_at_the_entry_bar(monkeypatch, caplog):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)
    with caplog.at_level("INFO", logger=TE.logger.name):
        o, f = _run(e, "g" * LATE + "r", vol=1.5e7)
    assert o == [] and e.blocks == ["FRENZY_WILLY_NOMISS_VOL24_LOW"] and "BTRUSDT" not in e._fz_willy_pending
    assert "R_NOMISS entry A refused at the red bar — 24 h volume $15.0M < $20M" in caplog.text and "R_NOMISS" in f["willy_last"]
    e = _engine(TE)
    assert _run(e, "g" * LATE + "r", vol=None)[0] == [] and e.blocks == ["FRENZY_WILLY_NOMISS_VOL24_LOW"]   # unreadable → refused
    e = _engine(TE)                                                                  # an on-time WILLY is unchanged (no floor at its red bar)
    assert [x["nomiss"] for x in _run(e, "r", bar_open=SP, vol=1.5e7)[0]] == [False] and e.blocks == []


def test_turnover_above_one_routes_to_willy_ht(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE, frenzy_willy_max_vol_mcap_ratio=1.0, frenzy_willy_ht_enabled=True, frenzy_willy_ht_max_slots=1)
    e = _engine(TE); e._willy_turnover = lambda pair, vol: 2.0
    o, _ = _run(e, "g" * LATE + "r")
    assert [(x["willy"], x["ht"], x["nomiss"]) for x in o] == [("A", True, True)] and e.blocks == ["FRENZY_WILLY_TURNOVER"]
    e = _engine(TE); e._willy_turnover = lambda pair, vol: None                      # unknown mcap → as today: refused, stays armed
    o, _ = _run(e, "g" * LATE + "r")
    assert o == [] and e.blocks == ["FRENZY_WILLY_TURNOVER_UNREAD"] and e._fz_willy_pending["BTRUSDT"]["nomiss"] is True
    e = _engine(TE); e._willy_turnover = lambda pair, vol: 0.5                       # below the cut → full-size WILLY
    assert [(x["ht"], x["nomiss"]) for x in _run(e, "g" * LATE + "r")[0]] == [(False, True)]


def test_switch_off_is_the_old_behaviour(monkeypatch, caplog):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE, frenzy_willy_a_nomiss=False)
    e = _engine(TE)
    with caplog.at_level("INFO", logger=TE.logger.name):
        o, _ = _run(e, "g" * LATE + "r")
    assert o == [] and not getattr(e, "_fz_willy_pending", {}) and OLD_MSG in caplog.text
    assert _ev(e).get("FRENZY_WILLY_A_NOMISS_OFF") == 1 and "FRENZY_WILLY_A_NOMISS" not in _ev(e)   # eligible refusal tracked
    assert "R_NOMISS switched off" in caplog.text


def test_inside_the_late_window_unchanged(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)                                                                  # 2 bars late (inside the window): armed from NOW, no tag
    o, _ = _run(e, "g" * 2 + "g", bar_open=SP + 2 * BAR_MS)
    pd_ = e._fz_willy_pending["BTRUSDT"]
    assert o == [] and pd_["armed"] == SP + 2 * BAR_MS and "nomiss" not in pd_ and not _ev(e).get("FRENZY_WILLY_A_NOMISS")


def test_one_per_episode(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)
    assert len(_run(e, "g" * LATE + "r")[0]) == 1
    assert _run(e, "g" * LATE + "rr", bar_open=READ + BAR_MS, prev=SP, warm=True)[0] == []      # warm, same episode: no second A
    assert _run(e, "g" * LATE + "rrr", bar_open=READ + 2 * BAR_MS, prev=None, warm=False)[0] == []   # cold again (FRENZY paused): seen
    e = _engine(TE); e._done["A"] = True                                             # DB backstop: an A fill on this episode already
    assert _run(e, "g" * LATE + "r")[0] == [] and "FRENZY_WILLY_A_NOMISS" not in _ev(e)
    e = _engine(TE); e._fz_judged_loaded = False                                     # stored state unread → refused as before the fix
    assert _run(e, "g" * LATE + "r")[0] == [] and e.blocks == [] and not _ev(e)       # (today's late-window refusal; no R_NOMISS check)


def test_restart(monkeypatch):
    """① a fresh process (empty seen map) first reading the pair after the window: the same bars → the same decision · ② a pending R_NOMISS
    entry persisted by the last process keeps its tag and its spike-based wait · ③ an episode the last process already saw never fires."""
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)
    assert [x["nomiss"] for x in _run(e, "g" * LATE + "r")[0]] == [True]
    e = _engine(TE)
    assert _run(e, "g" * LATE + "g")[0] == [] and e._fz_willy_pending["BTRUSDT"]["nomiss"] is True
    pend = dict(e._fz_willy_pending["BTRUSDT"])

    async def go():
        eng, SL = await _mem(monkeypatch, None)()
        e1 = object.__new__(TE.TradingEngine)
        await e1._frenzy_judged_get()
        await e1._frenzy_willy_mark("BTRUSDT", SP)
        e1._fz_willy_pending = {"BTRUSDT": pend}
        await e1._frenzy_willy_save()
        e2 = object.__new__(TE.TradingEngine)                                        # the restart
        await e2._frenzy_judged_get()
        await eng.dispose()
        return e2
    e2 = asyncio.run(go())
    assert e2._fz_willy_pending["BTRUSDT"] == frenzy_willy_pending_clean(pend) and e2._fz_willy_pending["BTRUSDT"]["nomiss"] is True and e2._fz_willy_seen == {"BTRUSDT": SP}
    e = _engine(TE); e._fz_willy_seen = dict(e2._fz_willy_seen); e._fz_willy_pending = dict(e2._fz_willy_pending)
    o, _ = _run(e, "g" * (LATE + 2) + "r", bar_open=READ + 2 * BAR_MS, prev=None, warm=False)   # cold after the restart: the pending opens
    assert [(x["nomiss"], x["wait"]) for x in o] == [(True, LATE + 2)]
    e = _engine(TE); e._fz_willy_seen = {"BTRUSDT": SP}                              # seen, nothing pending → never fires
    assert _run(e, "g" * LATE + "r")[0] == [] and "FRENZY_WILLY_A_NOMISS" not in _ev(e)


# ── the real open path stamps the column ───────────────────────────────────────────────────────────────────────────────────────────

def test_frenzy_open_stamps_entry_frenzy_willy_nomiss(monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models
    import services.trading_engine as TE
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**12)

    async def run(pending):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        opened = []
        try:
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
                e._record_filter_block = lambda name, d, had_room=True: None
                e._flip_entry_fields = lambda *a, **k: {}
                e._sanitize_open_kwargs = lambda ef, s_, d: ef

                async def gvv(sig, wait=True):
                    return 1.0
                e._frenzy_gvol_value = gvv

                async def free(db=None):
                    return False, None, None, None
                e._willy_hold_state = free

                async def fake_open(**kw):
                    opened.append(kw); return object()
                e.open_position = fake_open
                e._fz_willy_pending = {"BTRUSDT": pending} if pending else {}
                flag = dict(pair="BTRUSDT", spike_ts=SP, hours=0.6, vwap=1.0, vs_vwap_pct=1.0, vol_mult=40.0, run_pct=8.0, atr_pct=2.0,
                            volume_24h=5e7, price=1.0, live_price=1.0, bar_ret_pct=-0.3)
                await e._frenzy_open(db, flag, {}, READ, willy="A", willy_note="red candle", willy_wait_bars=LATE)
        finally:
            await eng.dispose()
        return opened
    o = asyncio.run(run(dict(trig="A", spike=SP, armed=SP, exp=SP + 3_600_000, nomiss=True)))
    assert len(o) == 1 and o[0]["entry_frenzy_willy_nomiss"] is True and o[0]["entry_frenzy_willy_trigger"] == "A"
    o = asyncio.run(run(dict(trig="A", spike=SP, armed=SP, exp=SP + 3_600_000)))
    assert len(o) == 1 and o[0]["entry_frenzy_willy_nomiss"] is False


def test_order_column_migration_and_open_position_stamp():
    import models
    assert "entry_frenzy_willy_nomiss" in {c.name for c in models.Order.__table__.columns}   # rides the orders CSV for free (D12)
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "('entry_frenzy_willy_nomiss', 'BOOLEAN')" in db
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "entry_frenzy_willy_nomiss=(bool(entry_frenzy_willy_nomiss) if (_frenzy and _fz_es in WILLY_EXIT_CLASS) else None)," in eng
    assert "closed=closed)   # 🧭 (350) R_NOMISS reads the bars since the spike" in eng
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert main.count("\"entry_frenzy_willy_nomiss\": getattr(o, 'entry_frenzy_willy_nomiss', None),") == 2


# ── D11 / D12 parity ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_config_ui_export_parity():
    import config as C
    mf = type(C.trading_config.thresholds).model_fields
    assert mf["frenzy_willy_a_nomiss"].default is True
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]["frenzy_willy_a_nomiss"] is True
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 350" in src and "BTRUSDT 10-22" in src and "−0.07" in src and "0.30" in src   # evidence + gate
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-fz-willy-a-nomiss"') == 1
    assert ("getElementById('config-fz-willy-a-nomiss'); if (_e) _e.checked = config.thresholds.frenzy_willy_a_nomiss !== false;" in html)
    assert "frenzy_willy_a_nomiss: document.getElementById('config-fz-willy-a-nomiss')?.checked ?? true," in html
    assert "${_bt.frenzy_willy_a_nomiss === false ? ' (R_NOMISS OFF)'" in html                  # config block of _buildConfigLines …
    assert html.count("lines.push(..._buildConfigLines(cfg, changelog, hr, hr2, status));") == 2   # … which both exports print
    assert "o.entry_frenzy_willy_nomiss === true ? ' · R_NOMISS" in html                         # the fill's badge names it
    i = html.index('id="config-fz-willy-a-cold"'); j = html.index('id="config-fz-willy-a-nomiss"')
    assert 0 < j - i < 3000                                                                       # next to the WILLY late-window input


# ── scout gate (alert-only, frozen) ────────────────────────────────────────────────────────────────────────────────────────────────

def test_scout_gate_and_wiring(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_willy_nomiss as S
    t0 = pd.Timestamp("2026-11-01")

    def fr(vals, nm, start=0):
        return pd.DataFrame(dict(o=[t0 + pd.Timedelta(hours=start + k) for k in range(len(vals))], pct=vals, nomiss=nm))
    assert S.gate(fr([0.1] * 14, True), fr([0.1] * 20, False))[0] == "COLLECTING"
    assert S.gate(fr([-0.3] * 15, True), fr([0.1] * 20, False))[0] == "ALERT"                     # −0.30 < +0.10 − 0.30
    assert S.gate(fr([-0.15] * 15, True), fr([0.1] * 20, False))[0] == "KEEP"
    assert S.gate(fr([-0.3] * 15, True), fr([0.1] * 5, False, start=100))[0] == "NO_COMPARATOR"   # comparator = same period only
    assert S.gate(fr([-0.3] * 15, True), fr([0.1] * 9, False))[0] == "NO_COMPARATOR"             # < 10 comparator fills → no ALERT
    assert S.gate(fr([-0.3] * 15, True), fr([0.1] * 10, False))[0] == "ALERT"
    v, txt = S.gate(fr([-0.3] * 15 + [5.0] * 10, True), fr([0.1] * 30, False))                   # frozen first 15 — later fills never rescue it
    assert v == "ALERT" and "first 15" in txt
    p = tmp_path / "scalpars_orders_paper_x.csv"
    pd.DataFrame([dict(opened_at="2026-11-01T10:00:00", pair="A", direction="LONG", status="CLOSED", entry_strategy="FRENZY_WILLY",
                       pnl=1.0, pnl_percentage=0.5, close_reason="FRENZY_TP", is_paper=True, entry_frenzy_willy_nomiss=True),
                  dict(opened_at="2026-11-01T11:00:00", pair="B", direction="LONG", status="CLOSED", entry_strategy="FRENZY_WILLY_HT",
                       pnl=-1.0, pnl_percentage=-0.5, close_reason="FRENZY_SL", is_paper=True, entry_frenzy_willy_nomiss=False),
                  dict(opened_at="2026-11-01T12:00:00", pair="C", direction="LONG", status="CLOSED", entry_strategy="FRENZY_WILLY",
                       pnl=1.0, pnl_percentage=0.5, close_reason="MANUAL", is_paper=True, entry_frenzy_willy_nomiss=True),
                  dict(opened_at="2026-11-01T13:00:00", pair="D", direction="LONG", status="CLOSED", entry_strategy="FRENZY_LONG",
                       pnl=1.0, pnl_percentage=0.5, close_reason="FRENZY_TP", is_paper=True, entry_frenzy_willy_nomiss=None)]).to_csv(p, index=False)
    out = "\n".join(S.run(paths=[str(p)], dec_paths=[], since_ms=int(t0.timestamp() * 1000)))
    assert "R_NOMISS (entry_frenzy_willy_nomiss): N 1 " in out and "normal WILLY (same period, WILLY + HT): N 1 " in out
    assert "operator-closed: 1 — excluded" in out and "COLLECTING" in out
    assert "allowed (A_NOMISS) 0 (0d) · refused: missed 0 (0d) / late 0 (0d) / vol24 0 (0d) / off 0 (0d)" in out
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_willy_nomiss as _wnm" in src and "_rg_sec += _wnm.run(now_ms)" in src


def test_vol24_refusal_comes_before_the_hold_and_unread_state_journals_nothing(monkeypatch):
    """review fixes: a sub-floor R_NOMISS pair ends as VOL24_LOW even while the hold / slot is full (no FRENZY_WILLY_HOLD block, no open
    path); an on-time WILLY under the same hold stays armed exactly as before · a pass refused STATE_UNREAD journals no R_NOMISS event."""
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _engine(TE)
    calls = []

    async def held(db=None):
        calls.append(1); return True, 9, "ZZZUSDT", "OPEN"
    e._willy_hold_state = held

    async def full(db):
        calls.append(1); return True
    e._willy_slots_full = full
    o, f = _run(e, "g" * LATE + "r", vol=1.5e7)
    assert o == [] and e.blocks == ["FRENZY_WILLY_NOMISS_VOL24_LOW"] and calls == [] and "BTRUSDT" not in e._fz_willy_pending
    e2 = _engine(TE); e2._willy_hold_state = held; e2._willy_slots_full = full   # on-time WILLY, same hold: unchanged (blocked, stays armed)
    o, f = _run(e2, "r", bar_open=SP, vol=1.5e7)
    assert [x["nomiss"] for x in o] == [False] and e2._fz_willy_pending["BTRUSDT"]["blocked"] is True and "blocked" in f["willy_last"]
    e3 = _engine(TE); e3._fz_judged_loaded = False
    for cols in ("ggrgggg" + "r", "g" * LATE + "r"):                                 # missed / eligible: nothing journalled, old refusal
        assert _run(e3, cols)[0] == [] and not _ev(e3)
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_a_nomiss=False)))
    e4 = _engine(TE); e4._fz_judged_loaded = False
    assert _run(e4, "g" * LATE + "r")[0] == [] and not _ev(e4)                         # no OFF event either
