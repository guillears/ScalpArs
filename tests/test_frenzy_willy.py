"""🎲 Oct-8 FRENZY_WILLY (DECISION_LOG 251 — operator-directed DECLARED EXCEPTION, armed against the evidence, no automatic off).
LONG only, no filters. Entry A = the pass bar a pair becomes FRENZY-flagged for an episode (engine view, once per episode, restart-proof);
entry B = an episode's fresh ON bar that FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE did not take. A trigger ARMS a pending entry that opens on
the FIRST closed RED 5m candle within frenzy_willy_red_max_wait_minutes (60). Own exit: TP +1 % net · NO stop · 120-min cap. GLOBAL HOLD:
while a WILLY is open no other automated trade opens (manual exempt), every refusal counted + recorded. Pure rules + the engine paths +
every surface (D11 / D12). No wall-clock dependence: bar times are fixed or derived from the same `bar_open` the code under test uses."""
import asyncio
import datetime as dt
import json
import os
import re
import sys
import time
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.frenzy import (BAR_MS, frenzy_willy_exit_for, frenzy_willy_levels, frenzy_willy_new_flag, frenzy_willy_reads,  # noqa: E402
                             frenzy_willy_red, frenzy_willy_wait_ms, frenzy_willy_pending_clean)

LAST = 1_790_000_000_000 // BAR_MS * BAR_MS
BAR_OPEN = LAST + BAR_MS            # the forming bar's open = the close of the bar just judged


def _th(**kw):
    base = dict(frenzy_willy_enabled=True, frenzy_willy_entry_a=True, frenzy_willy_entry_b=True, frenzy_willy_tp_pct=1.0, frenzy_willy_stop_pct=0.0,
                frenzy_willy_max_hold_minutes=120, frenzy_willy_max_slots=1, frenzy_willy_invest_mult=1.0, frenzy_willy_lev_mult=1.0,
                frenzy_willy_red_max_wait_minutes=60)
    base.update(kw)
    return NS(**base)


# ── pure exit (no stop · TP +1 · 120-min cap) ─────────────────────────────────────────────────────────────────────────────────────

def test_exit_no_stop_tp_and_time_cap():
    th = _th()
    assert frenzy_willy_exit_for(1.0, 1.0, th) == (True, "FRENZY_TP", 1.0)                       # +1 net closes
    assert frenzy_willy_exit_for(0.99, 0.99, th)[0] is False
    for p in (-3.0, -10.0, -40.0):                                                                 # NO stop: a −10 % trade stays open …
        assert frenzy_willy_exit_for(p, 0.2, th, held_minutes=119.9)[0] is False, p
    assert frenzy_willy_exit_for(-10.0, 0.2, th, held_minutes=120.0) == (True, "MAX_HOLD_TIME", -10.0)   # … until the 120-min cap
    assert frenzy_willy_exit_for(0.3, 0.5, th, held_minutes=120.0) == (True, "MAX_HOLD_TIME", 0.3)
    assert frenzy_willy_exit_for(0.3, 1.2, th) == (True, "FRENZY_TP", 0.3)                       # a missed +1 peak closes at once …
    assert frenzy_willy_exit_for(-0.4, 1.2, th) == (True, "FRENZY_TP_LATE", -0.4)                # … never labelled a TP at a loss
    assert frenzy_willy_exit_for(-2.3, 0.0, th, stop_floor=-2.2)[:2] == (True, "STOP_LOSS")     # LIVE: the backstop line (paper: None)
    assert frenzy_willy_exit_for(-2.1, 0.0, th, stop_floor=-2.2)[0] is False
    assert frenzy_willy_exit_for(-3.0, 0.0, _th(frenzy_willy_stop_pct=3.0))[:2] == (True, "STOP_LOSS")   # > 0 re-arms a stop
    assert frenzy_willy_exit_for("x", 0.0, th)[0] is False


def test_levels_defaults():
    assert frenzy_willy_levels(_th()) == (1.0, None, 120)
    assert frenzy_willy_levels(_th(frenzy_willy_tp_pct=0, frenzy_willy_stop_pct=0, frenzy_willy_max_hold_minutes=0)) == (1.0, None, 120)
    assert frenzy_willy_levels(_th(frenzy_willy_tp_pct=None, frenzy_willy_stop_pct="x", frenzy_willy_max_hold_minutes=-5)) == (1.0, None, 120)
    assert frenzy_willy_levels(_th(frenzy_willy_tp_pct=2.0, frenzy_willy_stop_pct=-2.5, frenzy_willy_max_hold_minutes=45)) == (2.0, -2.5, 45)
    assert frenzy_willy_levels(NS()) == (1.0, None, 120)
    assert frenzy_willy_wait_ms(_th()) == 3_600_000 and frenzy_willy_wait_ms(NS()) == 3_600_000 and frenzy_willy_wait_ms(_th(frenzy_willy_red_max_wait_minutes=0)) == 0


def test_exit_matches_the_json():
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert frenzy_willy_levels(NS(**cfg)) == (1.0, None, 120) and frenzy_willy_wait_ms(NS(**cfg)) == 3_600_000


def test_red_candle_rule():
    assert frenzy_willy_red(dict(bar_ret_pct=-0.01)) is True
    assert frenzy_willy_red(dict(bar_ret_pct=0.0)) is False                                         # flat is not red
    assert frenzy_willy_red(dict(bar_ret_pct=0.3)) is False
    assert frenzy_willy_red(dict(bar_ret_pct=None)) is None and frenzy_willy_red({}) is None        # unreadable → no entry
    assert frenzy_willy_red(dict(bar_ret_pct=float("nan"))) is None and frenzy_willy_red(dict(bar_ret_pct="x")) is None


def test_new_flag_rule():
    assert frenzy_willy_new_flag(None, None, LAST) is True
    assert frenzy_willy_new_flag(LAST, None, LAST) is False
    assert frenzy_willy_new_flag(None, LAST, LAST) is False
    assert frenzy_willy_new_flag(LAST - 10 * BAR_MS, LAST - 10 * BAR_MS, LAST) is True
    assert frenzy_willy_new_flag(None, LAST + BAR_MS, LAST) is False
    assert frenzy_willy_new_flag(None, None, None) is False and frenzy_willy_new_flag("x", None, LAST) is False


def test_frozen_reads_shared():
    k = lambda i: f"2026-10-10T{i:02d}:00:00"
    rd = frenzy_willy_reads([(k(i), 1.0, "FRENZY_TP") for i in range(12)] + [(k(12 + i), -3.0, "MAX_HOLD_TIME") for i in range(8)])
    assert rd["revert"] == "revert" and rd["review"] == "ok" and rd["capped_loss_share"] == 0.0
    rd = frenzy_willy_reads([(k(i), -1.0, "MAX_HOLD_TIME") for i in range(6)] + [(k(6 + i), 1.0, "FRENZY_TP") for i in range(14)])
    assert rd["review"] == "review" and abs(rd["capped_loss_share"] - 60.0) < 1e-9 and rd["revert"] == "holds"
    rd = frenzy_willy_reads([(k(i), 1.0, "FRENZY_TP") for i in range(19)] + [(k(19), None, "")])  # the 20th still open
    assert rd["revert"] == "collecting"
    rd = frenzy_willy_reads([(k(5), -1.0, "MAX_HOLD_TIME")] * 1 + [(k(i), 1.0, "FRENZY_TP") for i in range(5)])  # sorted by OPEN time
    assert rd["revert_closed"] == 6


def test_pending_clean():
    assert frenzy_willy_pending_clean(dict(trig="A", spike=1, armed=2, exp=3)) == dict(trig="A", spike=1, armed=2, exp=3, why="", blocked=False, turnover=None, tkinds=[])
    assert frenzy_willy_pending_clean(dict(trig="A", spike=1, armed=2, exp=3, turnover="FRENZY_WILLY_TURNOVER"))["turnover"] == "FRENZY_WILLY_TURNOVER"
    assert frenzy_willy_pending_clean(dict(trig="C", spike=1, armed=2, exp=3)) is None
    assert frenzy_willy_pending_clean(dict(trig="A", spike="x", armed=2, exp=3)) is None and frenzy_willy_pending_clean("x") is None


# ── engine: arming + the first red candle ─────────────────────────────────────────────────────────────────────────────────────────

class _DB:
    async def rollback(self):
        pass

    async def execute(self, *a, **k):           # the pending "still blocked?" pair re-check: no open position
        class _R:
            def first(self):
                return None
        return _R()


def _engine(TE):
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
    e.blocks = []; e.opens = []; e.marks = []; e.saves = 0
    e._record_filter_block = lambda name, d, had_room=True: e.blocks.append(name)
    e._fz_judged_loaded = True
    e._sleeve_opened = False; e._done = {"A": False, "B": False}; e._open_result = "opened"; e._held = False

    async def fake_open(db, fl, ind, bar_open, wide=False, catchup_now=None, lite=False, willy=None, willy_note=None, willy_wait_bars=None):
        e.opens.append(dict(willy=willy, note=willy_note, bar=bar_open, wait=willy_wait_bars))
        if e._open_result == "hold":
            fl["_willy_hold"] = True; return False
        if e._open_result == "pair_held":
            fl["_willy_pair_held"] = True; return False
        if e._open_result == "disloc":
            return False
        fl["willy_last"] = f"{dt.datetime.utcfromtimestamp(bar_open / 1000):%m-%d %H:%M} WILLY {willy} opened"
        return True
    e._frenzy_open = fake_open

    async def mark(pair, sp):
        e.marks.append((pair, sp)); e.__dict__.setdefault('_fz_willy_seen', {})[pair] = sp
    e._frenzy_willy_mark = mark

    async def save():
        e.saves += 1
    e._frenzy_willy_save = save

    async def opened(db, pair, bar_open):
        return e._sleeve_opened
    e._frenzy_willy_sleeve_opened = opened

    async def done(db, pair, sp, trig):
        return e._done[trig]
    e._frenzy_willy_done_db = done

    async def hold_state(db=None):
        return (e._held, 9, "ZZZUSDT", "OPEN") if e._held else (False, None, None, None)
    e._willy_hold_state = hold_state
    return e


def _run(e, ep, bar_open=BAR_OPEN, prev=None, enabled=True, last_fire=None, fz=None, warm=True):
    del e.opens[:]
    ep = dict(dict(bar_ret_pct=-0.2), **ep)
    flag = dict(ep, pair="FOOUSDT", last_fire=last_fire)
    asyncio.run(e._frenzy_willy_eval(_DB(), "FOOUSDT", ep, flag, {}, bar_open, prev, enabled,
                                     fz or dict(on=True, ready=False, code="FRENZY_ATR_HIGH", text="ATR 3.4% > 3%"), pair_warm=warm))
    return list(e.opens), flag


def test_red_trigger_bar_opens_at_once(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    sp = LAST - 30 * BAR_MS
    o, f = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.35))
    assert [x["willy"] for x in o] == ["A"] and o[0]["wait"] == 0 and "red candle (-0.35%)" in o[0]["note"]
    assert "FRENZY_WILLY_ARMED_A" in e._willy_event_counts and not getattr(e, "_fz_willy_pending", {}) and "WILLY A opened" in f["willy_last"]
    assert f["last_fire"] is None                                                                  # FRENZY's own field untouched
    o, _ = _run(e, dict(spike_ts=sp), prev=sp, bar_open=BAR_OPEN + BAR_MS)                         # the same episode never re-arms
    assert o == []


def test_green_trigger_then_red_two_bars_later(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    sp = LAST - 30 * BAR_MS
    o, f = _run(e, dict(spike_ts=sp, bar_ret_pct=0.4))                                             # trigger bar green → armed, waiting
    assert o == [] and e._fz_willy_pending["FOOUSDT"]["trig"] == "A" and "armed · waiting red" in f["willy_last"]
    assert f["willy_pending"]["exp_ms"] == BAR_OPEN + 3_600_000
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=0.1), prev=sp, bar_open=BAR_OPEN + BAR_MS)          # bar 2 green
    assert o == []
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.05), prev=sp, bar_open=BAR_OPEN + 2 * BAR_MS)    # bar 3 red → opens, wait 2
    assert [x["willy"] for x in o] == ["A"] and o[0]["wait"] == 2 and "FOOUSDT" not in e._fz_willy_pending


def test_unreadable_bar_no_entry_then_red(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    sp = LAST - 30 * BAR_MS
    o, f = _run(e, dict(spike_ts=sp, bar_ret_pct=None))
    assert o == [] and "candle unreadable" in f["willy_last"] and "FOOUSDT" in e._fz_willy_pending
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.1), prev=sp, bar_open=BAR_OPEN + BAR_MS)
    assert o and o[0]["wait"] == 1


def test_no_red_within_60_min_expires_no_rearm(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    sp = LAST - 30 * BAR_MS
    _run(e, dict(spike_ts=sp, bar_ret_pct=0.2))
    for k in range(1, 13):                                                                          # 12 green bars = 60 min: still armed
        o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=0.2), prev=sp, bar_open=BAR_OPEN + k * BAR_MS)
        assert o == [] and "FOOUSDT" in e._fz_willy_pending
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.5), prev=sp, bar_open=BAR_OPEN + 13 * BAR_MS)    # a red bar AFTER the wait
    assert o == [] and "FOOUSDT" not in e._fz_willy_pending and e.blocks.count("FRENZY_WILLY_RED_EXPIRED") == 1
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.5), prev=None, bar_open=BAR_OPEN + 14 * BAR_MS)  # dropped + re-read: never re-armed
    assert o == [] and "FOOUSDT" not in e._fz_willy_pending


def test_red_exactly_at_the_wait_edge_opens(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    sp = LAST - 30 * BAR_MS
    _run(e, dict(spike_ts=sp, bar_ret_pct=0.2))
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.5), prev=sp, bar_open=BAR_OPEN + 12 * BAR_MS)
    assert o and o[0]["wait"] == 12


def test_sweep_expires_unflagged_and_switched_off(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    monkeypatch.setattr(TE, "_frenzy_flags", {"KEEPUSDT": {}})
    e = _engine(TE)
    e._fz_willy_pending = {"GONEUSDT": dict(trig="A", spike=1, armed=BAR_OPEN, exp=BAR_OPEN + 3_600_000, why="", blocked=False),
                           "KEEPUSDT": dict(trig="B", spike=1, armed=BAR_OPEN, exp=BAR_OPEN + 3_600_000, why="", blocked=False)}
    asyncio.run(e._frenzy_willy_sweep(BAR_OPEN + BAR_MS, True))
    assert list(e._fz_willy_pending) == ["KEEPUSDT"] and e.blocks == ["FRENZY_WILLY_RED_EXPIRED"]
    asyncio.run(e._frenzy_willy_sweep(BAR_OPEN + 13 * BAR_MS, True))                               # its wait passed (pair not read)
    assert e._fz_willy_pending == {}
    e._fz_willy_pending = {"KEEPUSDT": dict(trig="B", spike=1, armed=BAR_OPEN, exp=BAR_OPEN + 3_600_000, why="", blocked=False)}
    asyncio.run(e._frenzy_willy_sweep(BAR_OPEN + BAR_MS, False))                                   # WILLY switched off
    assert e._fz_willy_pending == {}


def test_global_hold_during_pending_stays_then_opens(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    sp = LAST - 30 * BAR_MS
    e._open_result = "hold"; e._held = True
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.2))                                            # red, but a WILLY is open → stays pending
    assert len(o) == 1 and e._fz_willy_pending["FOOUSDT"]["blocked"] is True
    o, f = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.2), prev=sp, bar_open=BAR_OPEN + BAR_MS)        # still held: re-checked quietly
    assert o == [] and "blocked (hold / pair held)" in f["willy_last"]
    e._held = False; e._open_result = "opened"                                                     # the other WILLY closed
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=0.3), prev=sp, bar_open=BAR_OPEN + 2 * BAR_MS)     # green: waits
    assert o == []
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.1), prev=sp, bar_open=BAR_OPEN + 3 * BAR_MS)
    assert [x["willy"] for x in o] == ["A"] and o[0]["wait"] == 3 and "FOOUSDT" not in e._fz_willy_pending


def test_pair_held_keeps_pending_other_refusal_ends_it(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE); e._open_result = "pair_held"
    _run(e, dict(spike_ts=LAST, bar_ret_pct=-0.2))
    assert e._fz_willy_pending["FOOUSDT"]["blocked"] is True
    e = _engine(TE); e._open_result = "disloc"
    _run(e, dict(spike_ts=LAST, bar_ret_pct=-0.2))
    assert "FOOUSDT" not in (getattr(e, "_fz_willy_pending", None) or {})                         # judged at the red bar: done


def test_entry_a_once_per_episode_and_warm_rule(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    sp = LAST - 30 * BAR_MS
    assert [x["willy"] for x in _run(e, dict(spike_ts=sp))[0]] == ["A"]
    assert _run(e, dict(spike_ts=sp), prev=sp, bar_open=BAR_OPEN + BAR_MS)[0] == []
    assert _run(e, dict(spike_ts=sp), prev=None, bar_open=BAR_OPEN + 2 * BAR_MS)[0] == []           # re-read after a drop: seen
    assert [x["willy"] for x in _run(e, dict(spike_ts=sp + 40 * BAR_MS), prev=sp, bar_open=BAR_OPEN + 50 * BAR_MS)[0]] == ["A"]   # new episode
    e = _engine(TE)
    o, _ = _run(e, dict(spike_ts=LAST - 6 * BAR_MS), warm=False)                                   # not judged on the previous bar → no A
    assert o == [] and e.marks == [("FOOUSDT", LAST - 6 * BAR_MS)] and "FRENZY_WILLY_ARMED_A" not in (getattr(e, "_willy_event_counts", None) or {})
    e = _engine(TE)
    assert [x["willy"] for x in _run(e, dict(spike_ts=BAR_OPEN), warm=False)[0]] == ["A"]         # spike bar = the bar just closed


def test_cold_a_window_pure():
    from services.frenzy import frenzy_willy_a_cold_bars, frenzy_willy_late_bars
    assert frenzy_willy_a_cold_bars(NS()) == 3 and frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars=None)) == 3
    assert frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars=-1)) == 3 and frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars="x")) == 3
    assert frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars=0)) == 0 and frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars=5)) == 5
    assert frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars=12)) == 12 and frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars=13)) == 12
    assert frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars=10_000)) == 12 and frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars=float("inf"))) == 12
    assert frenzy_willy_a_cold_bars(_th(frenzy_willy_a_cold_max_bars=float("nan"))) == 3
    assert frenzy_willy_late_bars(BAR_OPEN, BAR_OPEN) == 0 and frenzy_willy_late_bars(LAST, BAR_OPEN) == 1
    assert frenzy_willy_late_bars(BAR_OPEN - 3 * BAR_MS, BAR_OPEN) == 3 and frenzy_willy_late_bars(None, BAR_OPEN) is None


def test_cold_pair_gtc_late_a_arms(monkeypatch, caplog):
    """GTCUSDT 2026-10-08: the pair joined FRENZY's read (shortlist) only AFTER its spike bar → cold; the spike bar closed 1 bar before the
    bar just closed (also 2 — the log's 'spike 17:30' read at 17:40) → A arms on THIS bar, the red-candle wait counts from arming."""
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    for k in (1, 2):
        e = _engine(TE)
        sp = BAR_OPEN - k * BAR_MS
        with caplog.at_level("INFO", logger=TE.logger.name):
            caplog.clear()
            o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=0.4), warm=False)                       # green bar: armed, waits for red
        assert o == [] and e.marks == [("FOOUSDT", sp)] and e._willy_event_counts.get("FRENZY_WILLY_ARMED_A") == 1
        pd = e._fz_willy_pending["FOOUSDT"]
        assert pd["trig"] == "A" and pd["armed"] == BAR_OPEN and pd["exp"] == BAR_OPEN + 3_600_000   # wait from arming, not the spike
        assert f"[FRENZY_WILLY] FOOUSDT: entry A armed (flag {k} bars late: joined the read late) → waiting for the first red 5m candle" in caplog.text
        o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.3), prev=sp, bar_open=BAR_OPEN + BAR_MS)    # next bar red → opens, 1 bar after arming
        assert [x["willy"] for x in o] == ["A"] and o[0]["wait"] == 1


def test_cold_pair_window_edges(monkeypatch, caplog):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    assert [x["willy"] for x in _run(e, dict(spike_ts=BAR_OPEN - 3 * BAR_MS), warm=False)[0]] == ["A"]   # 3 bars back → arms (red → opens)
    e = _engine(TE)
    with caplog.at_level("INFO", logger=TE.logger.name):
        o, _ = _run(e, dict(spike_ts=BAR_OPEN - 4 * BAR_MS), warm=False)                         # 4 bars back → no A, logged as before
    assert o == [] and not getattr(e, "_fz_willy_pending", {}) and "FRENZY_WILLY_ARMED_A" not in (getattr(e, "_willy_event_counts", None) or {})
    assert e.marks == [("FOOUSDT", BAR_OPEN - 4 * BAR_MS)]                                       # marked seen: it never fires later
    assert "already flagged before this process judged the pair on the previous bar" in caplog.text and "no entry A" in caplog.text
    assert _run(e, dict(spike_ts=BAR_OPEN - 4 * BAR_MS), prev=BAR_OPEN - 4 * BAR_MS, bar_open=BAR_OPEN + BAR_MS)[0] == []
    e = _engine(TE)                                                                               # a warm pair is unaffected by the window
    assert [x["willy"] for x in _run(e, dict(spike_ts=BAR_OPEN - 10 * BAR_MS), warm=True)[0]] == ["A"]


def test_cold_window_restart_seen_episode_never_fires(monkeypatch):
    """restart: a 2-bar-old spike the last process already saw (persisted willy_seen, or the FrenzyFlag seed) → no A, even inside the window."""
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    sp = BAR_OPEN - 2 * BAR_MS
    e = _engine(TE); e._fz_willy_seen = {"FOOUSDT": sp}                                           # stored willy_seen (or seeded from FrenzyFlag)
    o, _ = _run(e, dict(spike_ts=sp, bar_ret_pct=-0.5), prev=None, warm=False)
    assert o == [] and not getattr(e, "_fz_willy_pending", {}) and "FRENZY_WILLY_ARMED_A" not in (getattr(e, "_willy_event_counts", None) or {})
    sa = dt.datetime.utcfromtimestamp(sp / 1000)

    async def go():
        eng, SL = await _mem(monkeypatch, None, flags=[("FOOUSDT", sa)])()
        e2 = object.__new__(TE.TradingEngine)
        async with SL() as db:
            await e2._frenzy_willy_seed_from_flags(db)
        await eng.dispose()
        return e2._fz_willy_seen
    seen = asyncio.run(go())
    e = _engine(TE); e._fz_willy_seen = dict(seen)
    assert _run(e, dict(spike_ts=sp, bar_ret_pct=-0.5), prev=None, warm=False)[0] == []
    e = _engine(TE); e._done["A"] = True                                                          # DB backstop: a fill before the restart
    assert _run(e, dict(spike_ts=sp, bar_ret_pct=-0.5), prev=None, warm=False)[0] == []
    e = _engine(TE); e._fz_judged_loaded = False                                                  # stored state unread → fail-closed
    assert _run(e, dict(spike_ts=sp, bar_ret_pct=-0.5), prev=None, warm=False)[0] == [] and e.blocks == ["FRENZY_WILLY_STATE_UNREAD"]


def test_cold_window_clamped_to_one_hour(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_a_cold_max_bars=999)))
    e = _engine(TE)
    assert [x["willy"] for x in _run(e, dict(spike_ts=BAR_OPEN - 12 * BAR_MS), warm=False)[0]] == ["A"]   # 12 bars = the ceiling
    e = _engine(TE)
    assert _run(e, dict(spike_ts=BAR_OPEN - 13 * BAR_MS), warm=False)[0] == []                    # 999 never widens past 1 h
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert 'min="0" max="12" id="config-fz-willy-a-cold"' in html
    assert "return [_key, (_key === 'frenzy_willy_a_cold_max_bars' && x > 12) ? 12 : (x === null" in html   # the save caps it at 12 too
    assert "['config-fz-willy-a-cold', 'frenzy_willy_a_cold_max_bars', 3]" in html                      # blank / negative → 3 on save


def test_cold_window_zero_is_the_old_rule(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_a_cold_max_bars=0)))
    e = _engine(TE)
    assert _run(e, dict(spike_ts=BAR_OPEN - BAR_MS), warm=False)[0] == []                         # 1 bar late → no A
    e = _engine(TE)
    assert [x["willy"] for x in _run(e, dict(spike_ts=BAR_OPEN), warm=False)[0]] == ["A"]         # the bar just closed → A


def test_entry_a_switch_off_db_backstop_unreadable(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    assert _run(e, dict(spike_ts=LAST), enabled=False)[0] == [] and e.marks == [("FOOUSDT", LAST)]
    assert _run(e, dict(spike_ts=LAST), enabled=True)[0] == []                                     # switched on later: seen
    e = _engine(TE); e._done["A"] = True
    assert _run(e, dict(spike_ts=LAST))[0] == []
    e = _engine(TE); e._done["A"] = None
    assert _run(e, dict(spike_ts=LAST))[0] == []
    e = _engine(TE); e._fz_judged_loaded = False
    assert _run(e, dict(spike_ts=LAST))[0] == [] and e.blocks == ["FRENZY_WILLY_STATE_UNREAD"]
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_entry_a=False)))
    e = _engine(TE)
    assert _run(e, dict(spike_ts=LAST))[0] == []


def test_entry_b_only_when_frenzy_wide_lite_did_not_open(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    sp = LAST - 40 * BAR_MS
    ep = dict(spike_ts=sp, fresh_on=True)
    e = _engine(TE); e._fz_willy_seen = {"FOOUSDT": sp}; e._sleeve_opened = True
    assert _run(e, ep, prev=sp)[0] == []
    _stamp = f"{dt.datetime.utcfromtimestamp(BAR_OPEN / 1000):%m-%d %H:%M} "
    for lf, fz, want in (
        (_stamp + "refused: ATR 3.40% > 3%", None, "refused: ATR 3.40% > 3%"),
        (_stamp + "refused: bearish day (BTC day −0.80%, trend gap −0.05%)", None, "bearish day"),
        (_stamp + "refused: market volume 1.20× normal ≥ 1×", None, "market volume"),
        (_stamp + "WIDE refused: green reclaim — 5 closes above its average (needs > 12)", None, "WIDE refused: green reclaim"),
        (_stamp + "refused: 2 FRENZY_LONG positions open (max 2)", None, "positions open"),
        (_stamp + "refused: WILLY hold — FRENZY_WILLY ZZZUSDT open", None, "WILLY hold"),
        (None, dict(on=False, ready=True, code="FRENZY_READY", text="READY"), "FRENZY_LONG off"),
        (None, dict(on=True, ready=False, code="FRENZY_GREEN_BAR", text="signal candle green (+0.200%)"), "FRENZY_GREEN_BAR — signal candle green"),
        ("10-01 00:00 refused: an OLD refusal", dict(on=True, ready=False, code="FRENZY_VOL24_LOW", text="24 h volume $5M < $20M"), "FRENZY_VOL24_LOW"),
    ):
        e = _engine(TE); e._fz_willy_seen = {"FOOUSDT": sp}
        o, f = _run(e, dict(ep, bar_ret_pct=0.3), prev=sp, last_fire=lf, fz=fz)                   # green: stays armed, the reason readable
        assert o == [] and "FRENZY_WILLY_ARMED_B" in e._willy_event_counts and want in e._fz_willy_pending["FOOUSDT"]["why"], want
        assert e._fz_willy_pending["FOOUSDT"]["why"].startswith("ON not taken: ")
        assert f["last_fire"] == lf                                                                # never overwritten by WILLY
        o, _ = _run(e, dict(ep, bar_ret_pct=-0.3, fresh_on=False), prev=sp, bar_open=BAR_OPEN + BAR_MS)
        assert [x["willy"] for x in o] == ["B"] and o[0]["wait"] == 1
    # the B reason is FRENZY's own record — WILLY's text in willy_last is never read
    assert TE.TradingEngine._frenzy_willy_b_reason(None, BAR_OPEN, dict(on=True, ready=False, code="X", text="t")) == "X — t"
    # one B per episode — used even when it expires
    e = _engine(TE); e._fz_willy_seen = {"FOOUSDT": sp}
    _run(e, dict(ep, bar_ret_pct=0.3), prev=sp)
    assert e._fz_willy_pending["FOOUSDT"]["trig"] == "B" and e._fz_willy_b_used["FOOUSDT"] == sp
    e._fz_willy_pending.clear()
    assert _run(e, dict(ep, bar_ret_pct=-0.3), prev=sp, bar_open=BAR_OPEN + 30 * BAR_MS)[0] == []
    e = _engine(TE); e._fz_willy_seen = {"FOOUSDT": sp}; e._done["B"] = True
    assert _run(e, ep, prev=sp)[0] == []
    e = _engine(TE); e._fz_willy_seen = {"FOOUSDT": sp}; e._sleeve_opened = None
    assert _run(e, ep, prev=sp)[0] == []
    assert _run(_engine(TE), dict(spike_ts=sp, fresh_on=False), prev=sp)[0] == []
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_entry_b=False)))
    e = _engine(TE); e._fz_willy_seen = {"FOOUSDT": sp}
    assert _run(e, ep, prev=sp)[0] == []


def test_no_b_on_a_bar_where_a_was_considered(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    o, _ = _run(e, dict(spike_ts=LAST - 30 * BAR_MS, fresh_on=True))
    assert [x["willy"] for x in o] == ["A"] and "FRENZY_WILLY_ARMED_B" not in (getattr(e, "_willy_event_counts", None) or {})
    e = _engine(TE); e._done["A"] = True                                                          # A considered (already filled) → still no B
    o, _ = _run(e, dict(spike_ts=LAST - 30 * BAR_MS, fresh_on=True))
    assert o == [] and "FRENZY_WILLY_ARMED_B" not in (getattr(e, "_willy_event_counts", None) or {})


def test_crash_is_isolated(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)

    async def boom(*a, **k):
        raise RuntimeError("boom")
    e._frenzy_open = boom
    _run(e, dict(spike_ts=LAST))
    assert e.blocks[-1] == "FRENZY_WILLY_FAILED"


def test_pass_runs_willy_last():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("async def _update_frenzy_pass(")
    body = eng[i:eng.index("async def _frenzy_persist_flags(", i)]
    w = body.index("await self._frenzy_willy_eval(")
    for earlier in ("await self._frenzy_open(db, flag, ind, bar_open)", "await self._frenzy_open(db, flag, ind, bar_open, wide=True)",
                    "await self._frenzy_lite_eval("):
        assert body.index(earlier) < w, earlier
    assert "if not (_on or _obs or _wide or _lite or _willy):" in body and "await self._frenzy_willy_sweep(bar_open, _willy, seen)" in body
    assert "pair_warm=(_lo_prev is not None and int(_lo_prev) == bar_open - 600_000)" in body
    assert body.index("_lo_prev = self.__dict__.setdefault('_fz_pair_last_ok', {}).get(pair)") < body.index("self.__dict__.setdefault('_fz_pair_last_ok', {})[pair] = bar_open - 300_000")
    assert "await self._frenzy_willy_seed_from_flags(db)" in body


# ── restart: BotState JSON (seen / B used / pending) + the FrenzyFlag seed ──────────────────────────────────────────────────────────

def _mem(monkeypatch, state_json=None, flags=()):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models
    import services.trading_engine as TE

    async def mk():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        async with SL() as s_:
            s_.add(models.BotState(is_running=False, frenzy_last_judged_bar_ms=LAST - BAR_MS, frenzy_unjudged_json=state_json))
            for p, sa in flags:
                s_.add(models.FrenzyFlag(pair=p, spike_at=sa))
            await s_.commit()
        monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
        return eng, SL
    return mk


def test_seen_b_used_and_pending_survive_a_restart(monkeypatch):
    import services.trading_engine as TE
    pend = dict(trig="A", spike=LAST, armed=BAR_OPEN, exp=BAR_OPEN + 3_600_000, why="new flag", blocked=False, turnover=None, tkinds=[])

    async def go():
        eng, SL = await _mem(monkeypatch, json.dumps({"lite_done": {"X": 1}, "willy_pending": {"BAD": {"trig": "Q"}}}))()
        e1 = object.__new__(TE.TradingEngine)
        await e1._frenzy_judged_get()
        await e1._frenzy_willy_mark("FOOUSDT", LAST)
        e1._fz_willy_b_used = {"BARUSDT": LAST - BAR_MS}
        e1._fz_willy_pending = {"FOOUSDT": dict(pend)}
        await e1._frenzy_willy_save()
        e2 = object.__new__(TE.TradingEngine)                                                       # a restart
        await e2._frenzy_judged_get()
        await eng.dispose()
        return e2
    e2 = asyncio.run(go())
    assert e2._fz_willy_seen == {"FOOUSDT": LAST} and e2._fz_lite_done == {"X": 1} and e2._fz_willy_b_used == {"BARUSDT": LAST - BAR_MS}
    assert e2._fz_willy_pending == {"FOOUSDT": pend}                                               # a bad stored entry is dropped, not fatal
    js = json.loads(e2._frenzy_state_json())
    assert {"willy_seen", "willy_b_used", "willy_pending", "lite_done", "on_done", "unread"} <= set(js)


def test_pending_after_restart_opens_on_red_or_expires_past_its_wait(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    e._fz_willy_seen = {"FOOUSDT": LAST}
    e._fz_willy_pending = {"FOOUSDT": dict(trig="A", spike=LAST, armed=BAR_OPEN, exp=BAR_OPEN + 3_600_000, why="", blocked=False)}
    o, _ = _run(e, dict(spike_ts=LAST, bar_ret_pct=-0.2), prev=None, bar_open=BAR_OPEN + 4 * BAR_MS, warm=False)
    assert o and o[0]["wait"] == 4                                                                 # cold pass: the pending still opens
    e = _engine(TE)
    e._fz_willy_seen = {"FOOUSDT": LAST}
    e._fz_willy_pending = {"FOOUSDT": dict(trig="A", spike=LAST, armed=BAR_OPEN, exp=BAR_OPEN + 3_600_000, why="", blocked=False)}
    o, _ = _run(e, dict(spike_ts=LAST, bar_ret_pct=-0.2), prev=None, bar_open=BAR_OPEN + 20 * BAR_MS, warm=False)
    assert o == [] and e.blocks == ["FRENZY_WILLY_RED_EXPIRED"]                                    # stale pending never fires


def test_judged_get_bad_willy_value_never_wipes_other_maps(monkeypatch):
    import services.trading_engine as TE
    stored = json.dumps({"unread": {"AUSDT": LAST}, "on_done": {"NMRUSDT": LAST}, "willy_seen": {"X": "bad", "Y": LAST}, "willy_b_used": [1, 2]})

    async def go():
        eng, _ = await _mem(monkeypatch, stored)()
        e = object.__new__(TE.TradingEngine)
        await e._frenzy_judged_get()
        await eng.dispose()
        return e
    e = asyncio.run(go())
    assert e._fz_pair_unjudged == {"AUSDT": LAST} and e._fz_on_done == {"NMRUSDT": LAST} and e._fz_willy_seen == {"Y": LAST}


def test_frenzyflag_seed_restart_no_late_a(monkeypatch):
    """restart with an EMPTY seen map + a pre-existing FrenzyFlag row (the last process had flagged the episode) read late → no A."""
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    sp = LAST - 30 * BAR_MS
    sa = dt.datetime.utcfromtimestamp(sp / 1000)

    async def go():
        eng, SL = await _mem(monkeypatch, None, flags=[("FOOUSDT", sa)])()
        e = object.__new__(TE.TradingEngine)
        async with SL() as db:
            await e._frenzy_willy_seed_from_flags(db)
        await eng.dispose()
        return e._fz_willy_seen
    seen = asyncio.run(go())
    assert seen == {"FOOUSDT": sp}
    e = _engine(TE); e._fz_willy_seen = dict(seen)
    o, _ = _run(e, dict(spike_ts=sp), prev=None, warm=True)                                       # even a warm read: seen → no A
    assert o == [] and "FRENZY_WILLY_ARMED_A" not in (getattr(e, "_willy_event_counts", None) or {})


# ── the GLOBAL HOLD ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _hold_db(monkeypatch, open_willy=False):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models
    import services.trading_engine as TE

    async def mk():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        if open_willy:
            async with SL() as s_:
                s_.add(models.Order(pair="WWWUSDT", direction="LONG", status="OPEN", is_paper=True, entry_strategy="FRENZY_WILLY",
                                    confidence="STRONG_BUY", entry_price=1.0, quantity=1.0, investment=1.0, leverage=20, notional_value=20.0,
                                    opened_at=dt.datetime(2026, 10, 9, 1, 0)))
                await s_.commit()
        monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
        monkeypatch.setattr(TE, "_open_orders_cache", {})
        return eng, SL
    return mk


SLEEVES = ("MOMENTUM", "FLIP:FAN_RATIO_GATE", "SPIKE_FADE", "SPIKE_BOUNCE", "SPIKE_CHASE", "BULLRUN_LONG", "BEARRUN_SHORT", "SURGE_LONG",
           "SURGE_SHORT", "FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY", "BULL_LONG", "BOUNCE_LONG")


@pytest.mark.willy_hold
def test_hold_blocks_every_sleeve_records_and_dedupes(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=True)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks = []
        e._record_filter_block = lambda name, d, had_room=True: blocks.append((name, d))
        res = []
        async with SL() as db:
            for sl in SLEEVES:
                d = "SHORT" if sl in ("SPIKE_FADE", "BEARRUN_SHORT", "SURGE_SHORT") else "LONG"
                res.append(await e._willy_hold_block(db, "FOOUSDT", d, sl, price=1.23, invest_mult=1.5, lev_mult=0.32, signal_ms=BAR_OPEN))
            await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))                              # the first events are written in the background
            for _ in range(3):                                                                    # the same setup again during the same hold
                res.append(await e._willy_hold_block(db, "FOOUSDT", "LONG", "MOMENTUM", price=1.3))
            await e._willy_hold_flush()                                                           # repeat counts: in memory, flushed ≤ 1/min + at the close
            await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        async with SL() as s_:
            rows = (await s_.execute(select(models.WillyHoldBlock))).scalars().all()
        await eng.dispose()
        return res, blocks, rows
    res, blocks, rows = asyncio.run(go())
    assert all(res) and len(blocks) == len(SLEEVES) + 3 and all(b[0] == "FRENZY_WILLY_HOLD" for b in blocks)
    assert len(rows) == len(SLEEVES)                                                              # one event per (WILLY, pair, dir, sleeve)
    m = [r for r in rows if r.sleeve == "MOMENTUM"][0]
    assert m.repeats == 4 and m.price == 1.23 and m.invest_mult == 1.5 and m.lev_mult == 0.32 and m.willy_pair == "WWWUSDT" and m.willy_order_id > 0
    assert m.signal_at == dt.datetime.utcfromtimestamp(BAR_OPEN / 1000)


@pytest.mark.willy_hold
def test_hold_free_cache_shortcut_memo_and_fail_closed(monkeypatch):
    import services.trading_engine as TE

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=False)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks = []
        e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
        out = {}
        async with SL() as db:
            out["free"] = await e._willy_hold_block(db, "FOOUSDT", "LONG", "MOMENTUM")
            TE._open_orders_cache["QQQUSDT"] = [dict(id=77, entry_strategy="FRENZY_WILLY")]         # opened a moment ago: the cache knows first
            out["cache"] = await e._willy_hold_state(db)
            TE._open_orders_cache.clear()

            class Bad:
                async def execute(self, *a, **k):
                    raise RuntimeError("db down")
            e._wh_free_at = None
            out["unread"] = await e._willy_hold_block(Bad(), "FOOUSDT", "LONG", "MOMENTUM")
        await eng.dispose()
        return out, blocks
    out, blocks = asyncio.run(go())
    assert out["free"] is False and out["cache"] == (True, 77, "QQQUSDT", "OPEN")
    assert out["unread"] is True and blocks == ["FRENZY_WILLY_HOLD_UNREAD"]                       # fail-closed, counted


def test_hold_choke_point_in_open_position_and_not_manual():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("    async def open_position("); j = eng.index("\n    async def ", i + 10)
    body = eng[i:j]
    h = body.index("if await self._willy_hold_block(db, pair, direction, _wh_es, price=current_price, invest_mult=cell_mult, lev_mult=cell_lev_mult,")
    assert body.index("investment, leverage, cell_capped = self.calculate_position_size(") < h
    assert h < body.index("maker_fee_rate = getattr(tc, 'maker_fee', tc.trading_fee)") < body.index("order = Order(")
    m0 = eng.index("    async def open_manual_position("); m1 = eng.index("\n    async def ", m0 + 10)
    assert "_willy_hold" not in eng[m0:m1]                                                          # MANUAL opens are never blocked
    # every automated open path reaches open_position (no other OPEN Order construction)
    _oc = [m.start() for m in re.finditer(r"order = Order\(", eng)]   # SIGNAL_EXPIRED record · open_position · open_manual_position
    assert len(_oc) == 3 and eng[_oc[0]:_oc[0] + 200].count('status="SIGNAL_EXPIRED"') == 1 and i < _oc[1] < j and m0 < _oc[2] < m1
    # the hold's sleeve label = the Order's entry_strategy for every sleeve flag combination
    expr = body[body.index("_wh_es = ") + len("_wh_es = "):body.index("\n", body.index("_wh_es = "))]
    base = dict(_frenzy=False, _fz_es="FRENZY_LONG", _surge=False, direction="LONG", bearrun_short=False, bullrun_long=False, spike_bounce=False,
                spike_fade=False, spike_chase_probe=False, bounce_long=False, bull_long=False, flip_source=None)
    cases = [({}, "MOMENTUM"), ({"flip_source": "FAN"}, "FLIP:FAN"), ({"spike_fade": True}, "SPIKE_FADE"), ({"bullrun_long": True}, "BULLRUN_LONG"),
             ({"_surge": True, "direction": "SHORT"}, "SURGE_SHORT"), ({"_frenzy": True, "_surge": True, "_fz_es": "FRENZY_WILLY"}, "FRENZY_WILLY"),
             ({"bearrun_short": True}, "BEARRUN_SHORT"), ({"spike_chase_probe": True}, "SPIKE_CHASE"), ({"bounce_long": True}, "BOUNCE_LONG")]
    for kw, want in cases:
        assert eval(expr, {}, dict(base, **kw)) == want, (kw, want)
    o = eng.index("    async def _frenzy_open("); o1 = eng.index("    async def _maybe_open_surge(", o)
    fo = eng[o:o1]
    assert fo.index("await self._willy_hold_block(db, pair, \"LONG\", _es,") < fo.index("_slots = max(1, int(getattr(th, 'frenzy_willy_max_slots'")


@pytest.mark.willy_hold
def test_frenzy_open_every_frenzy_sleeve_names_the_hold(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**6)
    bar_open = BAR_OPEN

    async def go(kw):
        eng, SL = await _hold_db(monkeypatch, open_willy=True)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks = []
        e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)

        async def no_open(**k):
            raise AssertionError("open_position must not be reached")
        e.open_position = no_open
        flag = dict(pair="FOOUSDT", spike_ts=bar_open - 600_000, price=1.0, live_price=1.0, hours=1.0)
        async with SL() as db:
            r = await e._frenzy_open(db, flag, {}, bar_open, **kw)
        await eng.dispose()
        return r, blocks, flag
    for kw, fld in (({}, "last_fire"), ({"wide": True}, "last_fire"), ({"lite": True}, "last_fire"), ({"willy": "A"}, "willy_last")):
        r, blocks, flag = asyncio.run(go(kw))
        assert r is False and blocks == ["FRENZY_WILLY_HOLD"] and flag["_willy_hold"] is True and "WILLY hold" in flag[fld], kw


def test_lite_and_catchup_defer_on_the_hold():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    li = eng.index("    async def _frenzy_lite_eval("); lb = eng[li:eng.index("    @staticmethod", li)]
    assert lb.index("if ready and await self._willy_hold_block(db, pair, \"LONG\", \"FRENZY_LITE\"") < lb.index("await self._frenzy_lite_mark_done(pair, sid)\n            _notes")
    ci = eng.index("    async def _frenzy_catchup("); cb = eng[ci:eng.index("    async def _frenzy_open(", ci)] if "    async def _frenzy_open(" in eng[ci:] else eng[ci:ci + 6000]
    assert cb.index("self._frenzy_willy_defer_on(pair, ep.get('on_bar_ts'))") < cb.index("_tried[_key] = bar_open")   # never burns the chance
    assert "if flag.pop('_willy_hold', False) and ep.get('on_bar_ts') is not None:" in eng


@pytest.mark.willy_hold
def test_lite_ready_signal_on_hold_does_not_consume_the_stretch(monkeypatch):
    import services.trading_engine as TE
    from tests.test_frenzy_lite import _ep, _th as _lth, _lite_runner, SID
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_lth()))
    e, run = _lite_runner(TE)

    async def held(db=None):
        return True, 5, "WWWUSDT", "OPEN"
    e._willy_hold_state = held

    e._willy_hold_record = lambda *a, **k: True   # synchronous, like the real one (round-3 M7)
    calls, blocks, flag = run(_ep())
    assert calls == [] and blocks == ["FRENZY_WILLY_HOLD"] and "FOOUSDT" not in (getattr(e, "_fz_lite_done", None) or {})
    assert "stretch NOT judged" in flag["last_fire"]

    async def free(db=None):
        return False, None, None, None
    e._willy_hold_state = free
    calls, blocks, _ = run(_ep(above_streak=15, last_bar_ts=_ep()["last_bar_ts"] + BAR_MS))       # next bar: judged normally
    assert len(calls) == 1 and e._fz_lite_done["FOOUSDT"] == SID


def test_defer_on_sets_the_catchup_horizon():
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    e._frenzy_willy_defer_on("FOOUSDT", LAST)
    assert e._fz_pair_unjudged == {"FOOUSDT": LAST - BAR_MS}
    e._frenzy_willy_defer_on("FOOUSDT", LAST + 3 * BAR_MS)                                         # never moves forward
    assert e._fz_pair_unjudged == {"FOOUSDT": LAST - BAR_MS}
    from services.frenzy import frenzy_catchup_check
    st, age = frenzy_catchup_check(dict(in_state=True, fresh_on=False, on_bar_ts=LAST, last_bar_ts=LAST + 2 * BAR_MS), LAST - BAR_MS, 6)
    assert st is not None and age == 2                                                             # → re-judged as a catch-up later


# ── engine: _frenzy_open(willy=…) guards ──────────────────────────────────────────────────────────────────────────────────────────

def test_frenzy_open_willy_path(monkeypatch):
    """own tag / slots; ANY open position on the pair refuses it (pending kept); NO gvol gate, NO bearish block, NO pair-day cap; the
    dislocation guard; open_position gets frenzy_willy=True + the trigger + the wait bars, never frenzy_long / the strong bump. Bar time is
    the fixed BAR_OPEN (lateness disabled) — no wall-clock dependence."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, config as C
    import services.trading_engine as TE
    th = C.trading_config.thresholds
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**12)
    monkeypatch.setattr(TE, "_current_btc_trend_gap_pct", -0.5)       # a BEARISH day: WILLY ignores it
    monkeypatch.setattr(TE, "_current_btc_1d_ret_pct", -2.0)
    bar_open = BAR_OPEN
    bar_dt = dt.datetime.utcfromtimestamp(bar_open / 1000)

    async def run(rows, gv=5.0, slots=1, ok=True, live=1.0, trig="A", hold_free=True):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        old = (th.frenzy_gvol_max, th.frenzy_willy_max_slots, th.frenzy_bearish_day_block, th.frenzy_max_entries_per_pair_day)
        try:
            th.frenzy_gvol_max = 1.0; th.frenzy_willy_max_slots = slots; th.frenzy_bearish_day_block = True; th.frenzy_max_entries_per_pair_day = 1
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                for pair, strat, status, at in rows:
                    db.add(models.Order(pair=pair, direction="LONG", status=status, is_paper=True, entry_strategy=strat, confidence="STRONG_BUY",
                                        entry_price=1.0, quantity=1.0, investment=1.0, leverage=20, notional_value=20.0, opened_at=at))
                await db.commit()
                blocks = []; opened = []
                e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
                e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
                e._flip_entry_fields = lambda *a, **k: {}
                e._sanitize_open_kwargs = lambda ef, s_, d: ef

                async def gvv(sig, wait=True):
                    assert wait is False                                         # stamp only — never waits for the market-volume read
                    return gv
                e._frenzy_gvol_value = gvv
                if hold_free:   # the hold has its own tests; here the WILLY guards behind it
                    async def _free(db=None):
                        return False, None, None, None
                    e._willy_hold_state = _free

                async def fake_open(**kw):
                    opened.append(kw); return object() if ok else None
                e.open_position = fake_open
                flag = dict(pair="FOOUSDT", spike_ts=bar_open - 600_000, hours=0.1, vwap=1.0, vs_vwap_pct=1.0, vol_mult=40.0, run_pct=8.0,
                            atr_pct=6.0, volume_24h=5e6, price=1.0, live_price=live, adx_delta=1.0, di_spread=2.0, above_streak=2, above_share=90.0,
                            bar_ret_pct=-0.4, last_fire="FRENZY's own note")
                r = await e._frenzy_open(db, flag, {}, bar_open, willy=trig, willy_note="red candle (-0.40%) · 2 bars after the trigger", willy_wait_bars=2)
        finally:
            th.frenzy_gvol_max, th.frenzy_willy_max_slots, th.frenzy_bearish_day_block, th.frenzy_max_entries_per_pair_day = old
            await eng.dispose()
        return blocks, opened, r, flag
    b, o, r, f = asyncio.run(run([("FOOUSDT", "FRENZY_WILLY", "CLOSED", bar_dt - dt.timedelta(hours=3))]))   # earlier WILLY today: no day cap
    assert b == [] and len(o) == 1 and r is True and "WILLY A opened" in f["willy_last"] and f["last_fire"] == "FRENZY's own note"
    k = o[0]
    assert k["frenzy_willy"] is True and k["entry_frenzy_willy_trigger"] == "A" and k["entry_frenzy_willy_wait_bars"] == 2 and k["frenzy_long"] is False
    assert k["frenzy_wide"] is False and k["frenzy_lite"] is False and k["frenzy_strong"] is False and k["entry_frenzy_gvol"] == 5.0
    assert k["entry_atr_pct"] == 6.0 and k["entry_frenzy_stop_atr"] is None                       # no stop → no stop/ATR stamp
    b, o, r, f = asyncio.run(run([("FOOUSDT", "MOMENTUM", "OPEN", bar_dt - dt.timedelta(minutes=30))]))
    assert b == ["FRENZY_WILLY_PAIR_HELD"] and o == [] and f["_willy_pair_held"] is True and "refused: the pair already has an open position" in f["willy_last"]
    b, o, _, _ = asyncio.run(run([("A", "FRENZY_WILLY", "OPEN", bar_dt - dt.timedelta(minutes=5))]))
    assert b == ["FRENZY_WILLY_MAX_SLOTS"] and o == []
    b, o, _, f = asyncio.run(run([("A", "FRENZY_WILLY", "OPEN", bar_dt - dt.timedelta(minutes=5))], hold_free=False))   # the real read: the hold first
    assert b == ["FRENZY_WILLY_HOLD"] and o == [] and f["_willy_hold"] is True
    b, o, _, _ = asyncio.run(run([("A", "FRENZY_LITE", "OPEN", bar_dt - dt.timedelta(minutes=5)), ("B", "FRENZY_LITE", "OPEN", bar_dt)]))
    assert b == [] and len(o) == 1                                                                # LITE's slots are not WILLY's (hold stubbed free)
    b, o, _, f = asyncio.run(run([], live=1.02))
    assert b == ["FRENZY_WILLY_DISLOC"] and o == [] and "price moved" in f["willy_last"]
    b, o, _, _ = asyncio.run(run([], gv=None))
    assert b == [] and len(o) == 1                                                                # market volume unreadable: no gate
    b, o, r, f = asyncio.run(run([], ok=False, trig="B"))
    assert b == ["FRENZY_WILLY_OPEN_REFUSED"] and r is False and "WILLY B refused" in f["willy_last"]
    b, o, _, _ = asyncio.run(run([("FOOUSDT", "FRENZY_WILLY", "CLOSED", bar_dt + dt.timedelta(seconds=5))]))
    assert o == [] and b == []                                                                    # this bar's entry already taken (retry pass)


def test_frenzy_open_willy_structure():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("async def _frenzy_open("); body = eng[i:eng.index("async def _maybe_open_surge(", i)]
    assert 'self._record_filter_block(f"{_bk}_LATE", "LONG")' in body and '_bk = "FRENZY" if _es == "FRENZY_LONG" else _es' in body
    assert "_day_cap = 0 if _wl else" in body and "if _wl:\n                _gvb = None" in body and "if _wl:\n                _bb = None" in body
    assert "flag['last_fire'] =" not in body                                                      # every outcome goes through _lfk


# ── exit wiring (both paths · hold cap · urgent close) ─────────────────────────────────────────────────────────────────────────────

def test_exit_wiring_every_path():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert 'FRENZY_STRATEGIES = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY")' in eng
    assert eng.count("frenzy_willy_exit_for(") == 2
    assert eng.count("_held_minutes(order.opened_at)") == 1 and eng.count("_held_minutes(order_info.get('opened_at'))") == 1
    assert "max_hold = frenzy_willy_levels(config.trading_config.thresholds)[2]" in eng
    for a, b in (('elif (order.entry_strategy or "") == "FRENZY_WILLY":   # 🎲 Oct-8 (251): its OWN exit', 'elif (order.entry_strategy or "") in FRENZY_STRATEGIES:   # 🔥 its own stop'),
                 ("elif (order_info.get('entry_strategy') or '') == 'FRENZY_WILLY':", "elif (order_info.get('entry_strategy') or '') in FRENZY_STRATEGIES:   # 🔥 its own stop")):
        assert eng.index(a) < eng.index(b)
    assert "_urgent_exit = (order.entry_strategy or \"\") in FRENZY_STRATEGIES" in eng and "_urgent_exit_paper = (order.entry_strategy or \"\") in FRENZY_STRATEGIES" in eng


def test_held_minutes():
    import services.trading_engine as TE
    assert TE._held_minutes(None) is None and TE._held_minutes("x") is None
    assert 59.9 < TE._held_minutes(dt.datetime.utcnow() - dt.timedelta(minutes=60)) < 60.1


def test_ema13_exit_excluded():
    import services.trading_engine as TE
    assert TE._ema13_cross_exit_applies("FRENZY_WILLY") is False


def test_sizing_1x_1_0():
    import config as C
    mf = C.SignalThresholds.model_fields
    assert mf["frenzy_willy_invest_mult"].default == 1.0 and mf["frenzy_willy_lev_mult"].default == 1.0
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert '_fz_es = ("FRENZY_WILLY" if (_frenzy and frenzy_willy)' in eng
    assert "_frenzy = bool(frenzy_long or frenzy_wide or frenzy_lite or frenzy_willy)" in eng
    assert "_sg_pref = _fz_es.lower() if _frenzy else" in eng and 'if _frenzy and _fz_es == "FRENZY_LONG" and frenzy_strong:' in eng


# ── config / UI / exports (D11 · D12) ──────────────────────────────────────────────────────────────────────────────────────────────

FIELDS = {"frenzy_willy_enabled": True, "frenzy_willy_entry_a": True, "frenzy_willy_entry_b": True, "frenzy_willy_invest_mult": 1.0,
          "frenzy_willy_lev_mult": 1.0, "frenzy_willy_tp_pct": 1.0, "frenzy_willy_stop_pct": 0.0, "frenzy_willy_max_hold_minutes": 120,
          "frenzy_willy_max_slots": 1, "frenzy_willy_red_max_wait_minutes": 60, "frenzy_willy_a_cold_max_bars": 3}


def test_config_d11():
    import config as C
    mf = C.SignalThresholds.model_fields
    assert mf["frenzy_willy_enabled"].default is False
    assert mf["frenzy_willy_tp_pct"].default == 1.0 and mf["frenzy_willy_stop_pct"].default == 0.0 and mf["frenzy_willy_max_hold_minutes"].default == 120
    assert mf["frenzy_willy_max_slots"].default == 1 and mf["frenzy_willy_red_max_wait_minutes"].default == 60
    assert mf["frenzy_willy_a_cold_max_bars"].default == 3 and "GTCUSDT 2026-10-08" in open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert mf["frenzy_willy_entry_a"].default is True and mf["frenzy_willy_entry_b"].default is True
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for k, v in FIELDS.items():
        assert cfg[k] == v, k
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    for s_ in ("DECISION_LOG 251", "FRENZY_NEW_FLAG_X_STUDY_2026-10-07.md", "FRENZY_FLAG_TRADE_MATH_2026-10-08.md", "FRENZY_SECONDS_DELAY_STUDY_2026-10-08.md",
               "−$380", "14 %", "−0.01…−0.08", "NO STOP LOSS", "GLOBAL HOLD", "FIRST closed RED 5m bar"):
        assert s_ in src, s_


def test_ui_inputs_load_save_and_both_exports():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for _id in ("config-fz-willy-enabled", "config-fz-willy-entry-a", "config-fz-willy-entry-b", "config-fz-willy-invest-mult", "config-fz-willy-lev-mult",
                "config-fz-willy-tp", "config-fz-willy-stop", "config-fz-willy-max-hold", "config-fz-willy-slots", "config-fz-willy-red-wait",
                "config-fz-willy-a-cold"):
        assert html.count(f'id="{_id}"') == 1, _id
    for row in ("['config-fz-willy-invest-mult', 'frenzy_willy_invest_mult', 1.0]", "['config-fz-willy-lev-mult', 'frenzy_willy_lev_mult', 1.0]",
                "['config-fz-willy-tp', 'frenzy_willy_tp_pct', 1.0]", "['config-fz-willy-stop', 'frenzy_willy_stop_pct', 0]",
                "['config-fz-willy-max-hold', 'frenzy_willy_max_hold_minutes', 120]", "['config-fz-willy-slots', 'frenzy_willy_max_slots', 1]",
                "['config-fz-willy-red-wait', 'frenzy_willy_red_max_wait_minutes', 60]", "['config-fz-willy-a-cold', 'frenzy_willy_a_cold_max_bars', 3]"):
        assert html.count(row) == 1, row
    assert "_key === 'frenzy_willy_max_slots' || _key === 'frenzy_willy_max_hold_minutes' || _key === 'frenzy_willy_red_max_wait_minutes' || _key === 'frenzy_willy_a_cold_max_bars') ? Math.round(x)" in html
    assert "new-flag A for a pair read late ≤ ${_bt.frenzy_willy_a_cold_max_bars ?? 3} candles after its spike" in html
    for k, d in (("enabled", "false"), ("entry-a", "true"), ("entry-b", "true")):
        key = "frenzy_willy_" + k.replace("-", "_")
        assert f"{key}: document.getElementById('config-fz-willy-{k}')?.checked ?? {d}" in html, key
        assert f"getElementById('config-fz-willy-{k}'); if (_e) _e.checked = config.thresholds.{key}" in html, key
    assert html.count('data-ss="fzy"') == 1 and "S.fzy = {" in html
    assert "WILLY ${_bt.frenzy_willy_enabled === true ? 'ON' : 'OFF'}" in html
    assert "enters on the first red 5m candle ≤ ${_bt.frenzy_willy_red_max_wait_minutes ?? 60} min after the trigger" in html
    assert "TP +${_bt.frenzy_willy_tp_pct ?? 1} / ${(_bt.frenzy_willy_stop_pct ?? 0) > 0 ? 'SL −' + _bt.frenzy_willy_stop_pct : 'no SL'} / ${_bt.frenzy_willy_max_hold_minutes ?? 120} min" in html
    assert "global hold (no other automated open while a WILLY is open; manual exempt) · declared exception" in html
    assert html.count("lines.push(..._buildConfigLines(cfg, changelog, hr, hr2, status));") == 2
    assert "WILLY ${m.willy_enabled ? 'ON' : 'OFF'}" in html and "first red candle ≤ ${m.willy_red_wait ?? '-'} min" in html and "waiting red · expires ' + frenzyWillyClock(x.exp_ms)" in html
    assert html.count("lines.push(...frenzyReportLines(perf, hr2));") == 2
    assert html.count("FRENZY Fills (FRENZY_LONG · FRENZY_WIDE · FRENZY_LITE · FRENZY_WILLY)") == 2
    assert '<option value="FRENZY_WILLY">' in html and "strat === 'FRENZY_WILLY') return tags" in html
    assert "if ((o.entry_strategy || '') === 'FRENZY_WILLY') {" in html and ">🎯 TP +${_wTp}% · ${_wSl} · ${_wMh}m</span>" in html and "'no SL'" in html
    assert "(?:WIDE |LITE |WILLY [AB] )?opened/.exec((f && f.willy_last) || '')" in html
    assert "${f.willy_last ? ' | 🎲 ' + frenzyLastFireLocal(f.willy_last) : ''}${f.willy_pending ? ' | 🟥 ' + frenzyWillyPendingText(f) : ''}" in html   # export
    assert "frenzyWillyPendingText(f)) + '</span>'" in html                                       # UI cell


def test_api_payload_and_sleeve_rows(monkeypatch):
    import main as M
    rows = M._open_by_sleeve([NS(entry_strategy="FRENZY_WILLY", direction="LONG")])
    assert rows and rows[0][0] == "FRENZY WILLY Long"
    perf = M._compute_sleeve_performance([NS(entry_strategy="FRENZY_WILLY", direction="LONG", pnl_percentage=1.0, pnl=6.3,
                                             opened_at=dt.datetime(2026, 10, 9), closed_at=dt.datetime(2026, 10, 9, 0, 20))])
    names = [r["sleeve"] for r in (perf.get("rows") if isinstance(perf, dict) else perf)] if perf else []
    assert "Frenzy-Willy" in names
    monkeypatch.setattr(M.trading_engine, "_fz_willy_pending", {"FOOUSDT": dict(trig="A", spike=1, armed=2, exp=3, why="", blocked=False)}, raising=False)
    m = M._frenzy_monitor_payload()
    assert m["willy_tp"] == 1.0 and m["willy_stop"] is None and m["willy_max_hold"] == 120 and m["willy_max_slots"] == 1 and m["willy_red_wait"] == 60
    assert m["willy_pending"] == [dict(pair="FOOUSDT", trig="A", armed_ms=2, exp_ms=3, blocked=False)]
    v = M._frenzy_flag_view(dict(pair="X", spike_ts=LAST, willy_last="w", willy_pending=dict(trig="B", armed_ms=1, exp_ms=2)))
    assert v["willy_last"] == "w" and v["willy_pending"]["trig"] == "B"
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '("FRENZY_WILLY", "FRENZY_WILLY (new flag + ON not taken · red-candle entry · turnover < 1× mcap · TP +1 / no SL / 120 min · global hold · declared exception)")' in main
    assert "  A · new flag" in main and "_wrt(_wrd(" in main and ".order_by(Order.opened_at.asc())" in main and "'FRENZY_WILLY', 'MANUAL')" in main
    assert "🔒 WILLY hold (lifetime) — automated setups refused while a WILLY was open" in main and 'e="WILLY_HOLD"' in main
    assert "getattr(_th, 'frenzy_willy_enabled', False)):" in main
    assert '"entry_frenzy_willy_trigger": getattr(o, \'entry_frenzy_willy_trigger\', None),   # 🎲 Oct-8 (251): FRENZY_WILLY entry A / B (closed orders: the A / B split)' in main


# ── stamps: the new columns + migration + the hold table ───────────────────────────────────────────────────────────────────────────

def test_columns_migration_and_hold_table():
    import models as Mo
    cols = {c.name for c in Mo.Order.__table__.columns}
    assert {"entry_frenzy_willy_trigger", "entry_frenzy_willy_wait_bars"} <= cols
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "('entry_frenzy_willy_trigger', 'VARCHAR(2)'), ('entry_frenzy_willy_wait_bars', 'INTEGER')" in db
    assert Mo.WillyHoldBlock.__tablename__ == "willy_hold_blocks"
    hc = {c.name for c in Mo.WillyHoldBlock.__table__.columns}
    assert {"willy_order_id", "willy_pair", "pair", "direction", "sleeve", "signal_at", "first_at", "last_at", "price", "invest_mult", "lev_mult",
            "repeats", "is_paper"} <= hc


# ── master builder + scouts ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_master_builder_tags_willy_as_frenzy_sleeve():
    src = open(os.path.join(ROOT, "scripts", "build_master_pool.py"), encoding="utf-8").read()
    assert "elif strat in ('FRENZY_LONG', 'FRENZY_WIDE', 'FRENZY_LITE', 'FRENZY_WILLY'):" in src
    assert "frenzy_willy_invest_mult=1.0, frenzy_willy_lev_mult=1.0" in src and 'if strat == "FRENZY_WILLY":' in src
    assert re.search(r'FRENZY3 = \("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE"\)', src)


def test_scouts_know_willy():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_frenzy_exits as S
    import scout_revert_gates as RG
    import scout_willy_hold as WH
    assert RG.DEPLOYS["FRENZY_WILLY"][0] == "grep:(DECISION_LOG 251)"
    assert S._wait_bucket(0) == "0 (red trigger bar)" and S._hours_bucket(2.0) == "1–3 h"
    assert WH.exit_rule("FRENZY_WILLY") == (1.0, None, 120)
    src = open(os.path.join(ROOT, "scripts", "scout_stop_slip.py"), encoding="utf-8").read()
    assert 'if es == "FRENZY_WILLY":' in src and "FRENZY_TP / FRENZY_TP_LATE" in src


# ── round 2: turnover filter · hold regression / race · e2e · events · review fixes ────────────────────────────────────────────────

def test_vol_mcap_ratio_pure_never_raises():
    from services.frenzy import vol_mcap_ratio
    assert vol_mcap_ratio(5e7, 1e8) == 0.5 and vol_mcap_ratio(2e8, 1e8) == 2.0
    for v, m in ((None, 1e8), (1e8, None), (0, 1e8), (1e8, 0), (-1, 1e8), ("x", 1), (float("nan"), 1e8), (float("inf"), 1e8), (object(), 1)):
        assert vol_mcap_ratio(v, m) is None, (v, m)


def _turn_engine(TE, monkeypatch, mcap):
    from services import mcap_service
    monkeypatch.setattr(mcap_service, "get", lambda pair: (mcap, 50))
    e = _engine(TE)
    e._fz_willy_seen = {"FOOUSDT": LAST}
    e._fz_willy_pending = {"FOOUSDT": dict(trig="A", spike=LAST, armed=BAR_OPEN, exp=BAR_OPEN + 3_600_000, why="", blocked=False, turnover=None)}
    return e


def _run_t(e, bar_open, ret=-0.2, vol=5e7):
    del e.opens[:]
    ep = dict(spike_ts=LAST, bar_ret_pct=ret)
    flag = dict(ep, pair="FOOUSDT", volume_24h=vol, price=1.0, live_price=1.0)
    asyncio.run(e._frenzy_willy_eval(_DB(), "FOOUSDT", ep, flag, {}, bar_open, LAST, True, None, pair_warm=True))
    return list(e.opens), flag


def test_turnover_below_cut_opens(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_max_vol_mcap_ratio=1.0)))
    e = _turn_engine(TE, monkeypatch, mcap=1e8)
    o, _ = _run_t(e, BAR_OPEN + BAR_MS, vol=5e7)                                                   # R = 0.5 < 1
    assert o and o[0]["willy"] == "A" and e.blocks == []


def test_turnover_refused_stays_pending_then_opens_when_r_drops(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_max_vol_mcap_ratio=1.0)))
    e = _turn_engine(TE, monkeypatch, mcap=1e8)
    o, f = _run_t(e, BAR_OPEN + BAR_MS, vol=2e8)                                                   # R = 2.0 ≥ 1 → refused, armed
    assert o == [] and e.blocks == ["FRENZY_WILLY_TURNOVER"] and e._fz_willy_pending["FOOUSDT"]["turnover"] == "FRENZY_WILLY_TURNOVER"
    assert "turnover 2.00× mcap ≥ 1×" in f["willy_last"] and not (getattr(e, "_willy_event_counts", None) or {})   # a filter block, not a tally event
    o, _ = _run_t(e, BAR_OPEN + 2 * BAR_MS, vol=1.5e8)                                             # still ≥ 1: recorded ONCE per pending
    assert o == [] and e.blocks == ["FRENZY_WILLY_TURNOVER"]
    o, _ = _run_t(e, BAR_OPEN + 3 * BAR_MS, vol=9e7)                                               # R = 0.9 → opens on this red bar
    assert o and o[0]["wait"] == 3 and "FOOUSDT" not in e._fz_willy_pending


def test_turnover_unreadable_fail_closed_and_expiry_reason(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_max_vol_mcap_ratio=1.0)))
    e = _turn_engine(TE, monkeypatch, mcap=None)                                                   # missing / stale mcap
    o, f = _run_t(e, BAR_OPEN + BAR_MS)
    assert o == [] and e.blocks == ["FRENZY_WILLY_TURNOVER_UNREAD"] and "turnover unreadable" in f["willy_last"]
    notes = []
    monkeypatch.setattr(TE._djournal, "note", lambda ev, **k: notes.append((ev, k)))
    _run_t(e, BAR_OPEN + 13 * BAR_MS)                                                               # wait passed → expired, reason TURNOVER_UNREAD
    assert "FRENZY_WILLY_RED_EXPIRED" in e.blocks and any(k.get("gate") == "FRENZY_WILLY_RED_EXPIRED" and k.get("reason", "").startswith("TURNOVER_UNREAD") for ev, k in notes)


def test_turnover_ratio_zero_is_off(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_max_vol_mcap_ratio=0)))
    e = _turn_engine(TE, monkeypatch, mcap=None)                                                   # even unreadable: the filter is off
    o, _ = _run_t(e, BAR_OPEN + BAR_MS, vol=9e9)
    assert o and e.blocks == []


def test_vol_mcap_stamp_on_every_order():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("    async def open_position("); j = eng.index("\n    async def ", i + 10)
    assert "entry_vol_mcap_ratio=vol_mcap_ratio(entry_pair_volume_24h_usd, _mcap_usd)," in eng[i:j]   # the shared Order() of every bot sleeve
    m0 = eng.index("    async def open_manual_position(")
    assert "st['entry_vol_mcap_ratio'] = vol_mcap_ratio(st.get('entry_pair_volume_24h_usd'), st.get('entry_mcap_usd'))" in eng
    import models as Mo
    assert "entry_vol_mcap_ratio" in {c.name for c in Mo.Order.__table__.columns}
    assert "('entry_vol_mcap_ratio', 'FLOAT')" in open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()


@pytest.mark.willy_hold
def test_hold_regression_no_memo_after_a_commit(monkeypatch):
    """deep review I1: free → an OPEN FRENZY_WILLY row appears (empty cache, as after a stale cache rebuild) → the very next check blocks."""
    import models
    import services.trading_engine as TE

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=False)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        async with SL() as db:
            a = await e._willy_hold_state(db)
            db.add(models.Order(pair="WWWUSDT", direction="LONG", status="OPEN", is_paper=True, entry_strategy="FRENZY_WILLY", confidence="STRONG_BUY",
                                entry_price=1.0, quantity=1.0, investment=1.0, leverage=20, notional_value=20.0, opened_at=dt.datetime(2026, 10, 9, 1, 0)))
            await db.commit()
            b = await e._willy_hold_state(db)
        await eng.dispose()
        return a, b
    a, b = asyncio.run(go())
    assert a[0] is False and b[0] is True and b[3] == "OPEN"


def test_hold_rechecked_under_the_book_lock():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("        _book_release = await self._book_hold()\n        # 🔒 (251, round-2 review)")
    blk = eng[i:i + 1600]
    assert "if (await self._willy_hold_state(db))[0]:" in blk and "if self.is_paper_mode:" in blk and "_book_release()\n                return None" in blk
    assert 'self._willy_event("FRENZY_WILLY_HOLD_RACE", pair,' in blk
    assert "WILLY_HOLD_MEMO_S" not in eng and "_wh_free_at" not in eng                             # no "free" memo anywhere
    # the cache rebuild keeps an order opened after its snapshot
    u = eng.index("    async def update_orders_cache(")
    ub = eng[u:eng.index("\nasync def realtime_stop_loss_callback", u)]
    assert "Order.id.in_(_cands), Order.status == \"OPEN\", Order.is_paper == self.is_paper_mode" in ub and "if _ci.get('id') in _keep_ids:" in ub


@pytest.mark.willy_hold
def test_e2e_open_position_refused_by_the_hold_and_recorded(monkeypatch):
    """the REAL open_position on real SQLite with an open FRENZY_WILLY: a MOMENTUM long is refused at the choke point, the block row written."""
    import models
    import services.trading_engine as TE
    from sqlalchemy import select

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
            r = await e.open_position(db=db, pair="FOOUSDT", direction="LONG", confidence="STRONG_BUY", current_price=1.0)
        await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        async with SL() as s_:
            rows = (await s_.execute(select(models.WillyHoldBlock))).scalars().all()
            n_open = len((await s_.execute(select(models.Order).where(models.Order.status == "OPEN"))).scalars().all())
        await eng.dispose()
        return r, blocks, rows, n_open
    r, blocks, rows, n_open = asyncio.run(go())
    assert r is None and "FRENZY_WILLY_HOLD" in blocks and n_open == 1
    assert len(rows) == 1 and rows[0].sleeve == "MOMENTUM" and rows[0].pair == "FOOUSDT" and rows[0].willy_pair == "WWWUSDT" and rows[0].reason == "OPEN"


def test_manual_open_never_reaches_the_hold():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    m0 = eng.index("    async def open_manual_position("); m1 = eng.index("\n    async def ", m0 + 10)
    assert "_willy_hold" not in eng[m0:m1] and "self.open_position(" not in eng[m0:m1]
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    a = main.index('@app.post("/api/orders/manual_open")')
    assert "trading_engine.open_manual_position(" in main[a:a + 1500] and "open_position(" not in main[a:a + 1500].replace("open_manual_position(", "")


@pytest.mark.willy_hold
def test_unread_hold_rows_dedupe_per_minute(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=False)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks = []
        e._record_filter_block = lambda n, d, had_room=True: blocks.append(n)

        class Bad:
            async def execute(self, *a, **k):
                raise RuntimeError("db down")
        for _ in range(3):
            assert await e._willy_hold_block(Bad(), "FOOUSDT", "LONG", "MOMENTUM", price=1.0)
        await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        async with SL() as s_:
            rows = (await s_.execute(select(models.WillyHoldBlock))).scalars().all()
        await eng.dispose()
        return blocks, rows
    blocks, rows = asyncio.run(go())
    assert blocks == ["FRENZY_WILLY_HOLD_UNREAD"] * 3 and len(rows) == 1 and rows[0].reason == "UNREAD" and rows[0].willy_order_id is None


def test_open_path_crash_keeps_the_pending_entry(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    e._fz_willy_seen = {"FOOUSDT": LAST}
    e._fz_willy_pending = {"FOOUSDT": dict(trig="A", spike=LAST, armed=BAR_OPEN, exp=BAR_OPEN + 3_600_000, why="", blocked=False, turnover=None)}

    async def failing(db, fl, ind, bar_open, **k):
        fl["_willy_open_failed"] = True; return None
    e._frenzy_open = failing
    _run(e, dict(spike_ts=LAST, bar_ret_pct=-0.2), prev=LAST, bar_open=BAR_OPEN + BAR_MS)
    assert "FOOUSDT" in e._fz_willy_pending                                                      # not burnt — the next red bar retries
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "flag['_willy_open_failed'] = True" in eng


def test_older_episode_pending_expires_before_new_triggers(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e = _engine(TE)
    old = LAST - 400 * BAR_MS
    e._fz_willy_pending = {"FOOUSDT": dict(trig="A", spike=old, armed=BAR_OPEN - BAR_MS, exp=BAR_OPEN + 3_600_000, why="", blocked=False, turnover=None)}
    e._fz_willy_seen = {"FOOUSDT": old}
    o, _ = _run(e, dict(spike_ts=LAST - 30 * BAR_MS, bar_ret_pct=0.3), prev=old)                    # the new episode arms A at once (M2)
    assert e._fz_willy_pending["FOOUSDT"]["spike"] == LAST - 30 * BAR_MS and "FRENZY_WILLY_RED_EXPIRED" in e.blocks


def test_no_b_when_hold_deferred_or_pair_held(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    sp = LAST - 40 * BAR_MS
    e = _engine(TE); e._fz_willy_seen = {"FOOUSDT": sp}
    del e.opens[:]
    ep = dict(spike_ts=sp, fresh_on=True, bar_ret_pct=0.3)
    flag = dict(ep, pair="FOOUSDT", _on_hold_deferred=True)
    asyncio.run(e._frenzy_willy_eval(_DB(), "FOOUSDT", ep, flag, {}, BAR_OPEN, sp, True, None, pair_warm=True))
    assert "FOOUSDT" not in (getattr(e, "_fz_willy_pending", None) or {})

    class HeldDB(_DB):
        async def execute(self, *a, **k):
            class _R:
                def first(self):
                    return (1,)
            return _R()
    e = _engine(TE); e._fz_willy_seen = {"FOOUSDT": sp}
    flag = dict(ep, pair="FOOUSDT")
    asyncio.run(e._frenzy_willy_eval(HeldDB(), "FOOUSDT", ep, flag, {}, BAR_OPEN, sp, True, None, pair_warm=True))
    assert "FOOUSDT" not in (getattr(e, "_fz_willy_pending", None) or {})                         # FRENZY refused PAIR_HELD: not "not taken"


def test_sweep_keeps_unread_pairs_after_restart(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE, "_frenzy_flags", {})
    e = _engine(TE)
    e._fz_willy_pending = {"FOOUSDT": dict(trig="A", spike=1, armed=BAR_OPEN, exp=BAR_OPEN + 3_600_000, why="", blocked=False, turnover=None)}
    asyncio.run(e._frenzy_willy_sweep(BAR_OPEN + BAR_MS, True, seen=set()))                         # klines failed: not read → kept
    assert "FOOUSDT" in e._fz_willy_pending
    asyncio.run(e._frenzy_willy_sweep(BAR_OPEN + BAR_MS, True, seen={"FOOUSDT"}))                  # read and not flagged → episode over
    assert e._fz_willy_pending == {}


def test_lite_hold_deferral_capped(monkeypatch):
    import services.trading_engine as TE
    from tests.test_frenzy_lite import _ep, _th as _lth, _lite_runner, SID
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_lth(frenzy_catchup_max_bars=6)))
    e, run = _lite_runner(TE)

    async def held(db=None):
        return True, 5, "WWWUSDT", "OPEN"
    e._willy_hold_state = held
    e._willy_hold_record = lambda *a, **k: False
    calls, blocks, _ = run(_ep())
    assert calls == [] and blocks == ["FRENZY_WILLY_HOLD"]
    lb = _ep()["last_bar_ts"]
    calls, blocks, _ = run(_ep(above_streak=20, last_bar_ts=lb + 6 * BAR_MS))                       # 6 bars later: still deferred
    assert blocks == ["FRENZY_WILLY_HOLD"] and "FOOUSDT" not in (getattr(e, "_fz_lite_done", None) or {})
    calls, blocks, flag = run(_ep(above_streak=21, last_bar_ts=lb + 7 * BAR_MS))                    # past the catch-up window → consumed
    assert calls == [] and blocks == ["FRENZY_LITE_HOLD_EXPIRED"] and e._fz_lite_done["FOOUSDT"] == SID


def test_willy_events_not_filter_blocks_and_ui_notes():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert 'self._willy_event(f"FRENZY_WILLY_ARMED_{trig}", pair, reason=why)' in eng and '_record_filter_block(f"FRENZY_WILLY_ARMED' not in eng
    dj = open(os.path.join(ROOT, "services", "decision_journal.py"), encoding="utf-8").read()
    assert "elif ev in ('OPEN', 'ADMIT', 'EXPIRED', 'WILLY'):" in dj
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "paper: no stop · live: exchange backstop ≈ −${" in html and "paper: no stop · live: ${m.willy_backstop_live != null" in html
    assert "${(_bt.frenzy_willy_max_vol_mcap_ratio ?? 0) > 0 ? 'turnover < ' + _bt.frenzy_willy_max_vol_mcap_ratio + '× mcap (unknown mcap = no entry)' : 'turnover filter OFF'}" in html
    assert html.count('id="config-fz-willy-turnover"') == 1 and html.count("['config-fz-willy-turnover', 'frenzy_willy_max_vol_mcap_ratio', 0]") == 1
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert main.count('"entry_frenzy_willy_wait_bars": getattr(o, \'entry_frenzy_willy_wait_bars\', None), "entry_vol_mcap_ratio"') == 2
    assert "await db.execute(delete(WillyHoldBlock).where(WillyHoldBlock.is_paper == is_paper))" in main
    assert "avg wait {sum(_wwb) / len(_wwb):.1f} bars to the red candle" in main



# ── round 3 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_cache_rebuild_keep_rule_confirms_open_in_mode(monkeypatch):
    """a cached order absent from the rebuild's snapshot is kept ONLY when a same-rebuild read confirms it OPEN in this mode."""
    import models
    import services.trading_engine as TE
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    now = dt.datetime.utcnow()

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        async with SL() as s_:
            for oid, st, paper in ((101, "OPEN", True), (102, "CLOSED", True), (103, "OPEN", False)):
                s_.add(models.Order(id=oid, pair=f"P{oid}USDT", direction="LONG", status=st, is_paper=paper, entry_strategy="FRENZY_WILLY",
                                    confidence="STRONG_BUY", entry_price=1.0, quantity=1.0, investment=1.0, leverage=20, notional_value=20.0, opened_at=now))
            await s_.commit()
        cache = {f"P{oid}USDT": [dict(id=oid, entry_strategy="FRENZY_WILLY", opened_at=now)] for oid in (101, 102, 103)}
        monkeypatch.setattr(TE, "_open_orders_cache", cache)
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True

        class SnapDB:   # the rebuild's first read (the snapshot) saw NO open order; later reads go to the real DB
            def __init__(self, real):
                self.real, self.n = real, 0

            async def execute(self, q, *a, **k):
                self.n += 1
                if self.n == 1:
                    class _R:
                        def scalars(self):
                            class _S:
                                def all(self):
                                    return []
                            return _S()
                    return _R()
                return await self.real.execute(q, *a, **k)
        async with SL() as db:
            try:
                await e.update_orders_cache(SnapDB(db))
            except Exception:
                pass
        await eng.dispose()
        return {k: [x["id"] for x in v] for k, v in TE._open_orders_cache.items()}
    out = asyncio.run(go())
    assert out == {"P101USDT": [101]}                                                              # closed → gone · other mode → gone


def test_close_position_pops_the_cache_after_commit():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("        if (order.status or '') == \"CLOSED\":   # 🔒 (251, round-3) the committed close leaves the realtime cache NOW")
    blk = eng[i:i + 700]
    assert "async with _cache_lock:" in blk and "_open_orders_cache.pop(order.pair, None)" in blk
    assert eng.index("_db_commit_success") < i                                                     # after the commit only
    import models as Mo
    assert any(ix.name == "ix_orders_status_strategy" for ix in Mo.Order.__table__.indexes)
    assert "CREATE INDEX IF NOT EXISTS ix_orders_status_strategy ON orders (status, entry_strategy, is_paper)" in open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()


def test_turnover_kinds_recorded_once_despite_flapping(monkeypatch):
    import services.trading_engine as TE
    from services import mcap_service
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_max_vol_mcap_ratio=1.0)))
    monkeypatch.setattr(mcap_service, "request_pair", lambda pair: False)
    e = _turn_engine(TE, monkeypatch, mcap=1e8)
    cur = {"cap": 1e8}
    monkeypatch.setattr(mcap_service, "get", lambda pair: (cur["cap"], 1))
    for k, cap in enumerate([1e8, None, 1e8, None, 1e8], start=1):                                # R ≥ 1 / unreadable / R ≥ 1 / …
        cur["cap"] = cap
        _run_t(e, BAR_OPEN + k * BAR_MS, vol=2e8)
    assert e.blocks.count("FRENZY_WILLY_TURNOVER") == 1 and e.blocks.count("FRENZY_WILLY_TURNOVER_UNREAD") == 1
    assert "FRENZY_WILLY_TURNOVER" not in (getattr(e, "_willy_event_counts", None) or {})          # a filter block, not a tally event


@pytest.mark.mcap_real
def test_mcap_request_pair_dedupes_and_respects_switch(monkeypatch):
    from services import mcap_service as M
    import config as C
    started = []

    class _Loop:
        def create_task(self, coro):
            coro.close(); started.append(1)

            class _T:
                def add_done_callback(self, cb):
                    pass
            return _T()
    monkeypatch.setattr(M.asyncio, "get_running_loop", lambda: _Loop())
    monkeypatch.setattr(M, "_one_req", {})
    monkeypatch.setattr(M, "_cache", {})
    monkeypatch.setattr(C.trading_config, "mcap_fetch_enabled", True, raising=False)
    assert M.request_pair("FOOUSDT") is True and M.request_pair("FOOUSDT") is False and len(started) == 1     # deduped 10 min
    monkeypatch.setattr(C.trading_config, "mcap_fetch_enabled", False, raising=False)
    assert M.request_pair("BARUSDT") is False                                                      # fetching OFF
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "mcap fetch OFF → WILLY can" in html


def test_arming_requests_the_pairs_cap(monkeypatch):
    import services.trading_engine as TE
    from services import mcap_service
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_max_vol_mcap_ratio=1.0)))
    asked = []
    monkeypatch.setattr(mcap_service, "request_pair", lambda pair: asked.append(pair) or True)
    monkeypatch.setattr(mcap_service, "get", lambda pair: (None, None))
    e = _engine(TE)
    _run(e, dict(spike_ts=LAST - 30 * BAR_MS, bar_ret_pct=0.3))
    assert asked and set(asked) == {"FOOUSDT"}


def test_lite_hold_first_cleared_at_mark_done():
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    e._fz_lite_hold_first = {"FOOUSDT": (1, 2)}
    asyncio.run(e._frenzy_lite_mark_done("FOOUSDT", 5))
    assert e._fz_lite_hold_first == {}


@pytest.mark.willy_hold
def test_close_flush_awaits_the_first_insert(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=True)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        e._record_filter_block = lambda n, d, had_room=True: None
        async with SL() as db:
            for _ in range(4):
                await e._willy_hold_block(db, "FOOUSDT", "LONG", "MOMENTUM", price=1.0)
            wid = list(e._wh_events)[0][0]
        await e._willy_hold_flush(drop_wid=wid)                                                       # the WILLY closed right away
        await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        async with SL() as s_:
            r = (await s_.execute(select(models.WillyHoldBlock))).scalars().all()
        await eng.dispose()
        return r, e._wh_events
    rows, ev = asyncio.run(go())
    assert len(rows) == 1 and rows[0].repeats == 4 and ev == {}


def test_reset_clears_willy_state():
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    i = main.index("async def _willy_reset_state(db, is_paper):")
    blk = main[i:i + 1200]
    assert "await db.execute(delete(WillyHoldBlock).where(WillyHoldBlock.is_paper == is_paper))" in blk
    assert "'_willy_event_counts', '_fz_lite_hold_first'" in blk
    d = main.index("async def _willy_drain_hold_writes():")
    assert "await asyncio.wait(_wht, timeout=10.0)" in main[d:d + 600]



# ── round-3 deep review ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_turnover_recorded_only_when_the_entry_could_otherwise_open(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th(frenzy_willy_max_vol_mcap_ratio=1.0)))
    e = _turn_engine(TE, monkeypatch, mcap=1e8)
    e._held = True; e._open_result = "hold"                                                       # a WILLY is open AND R would refuse
    o, f = _run_t(e, BAR_OPEN + BAR_MS, vol=2e8)
    assert len(o) == 1 and "FRENZY_WILLY_TURNOVER" not in e.blocks and e._fz_willy_pending["FOOUSDT"]["blocked"] is True
    o, _ = _run_t(e, BAR_OPEN + 2 * BAR_MS, vol=2e8)                                               # still held: quiet, no turnover record
    assert o == [] and "FRENZY_WILLY_TURNOVER" not in e.blocks
    e._held = False; e._open_result = "opened"                                                    # the hold ended — NOW turnover decides
    o, _ = _run_t(e, BAR_OPEN + 3 * BAR_MS, vol=2e8)
    assert o == [] and e.blocks.count("FRENZY_WILLY_TURNOVER") == 1
    o, _ = _run_t(e, BAR_OPEN + 4 * BAR_MS, vol=5e7)
    assert o and o[0]["wait"] == 4


@pytest.mark.willy_hold
def test_flush_increments_across_a_restart_mid_hold(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=True)()
        e1 = object.__new__(TE.TradingEngine); e1.is_paper_mode = True
        e1._record_filter_block = lambda n, d, had_room=True: None
        async with SL() as db:
            for _ in range(3):
                await e1._willy_hold_block(db, "FOOUSDT", "LONG", "MOMENTUM", price=1.0)
            await asyncio.gather(*list(getattr(e1, "_wh_tasks", set())))
            await e1._willy_hold_flush()                                                           # stored 3
            e2 = object.__new__(TE.TradingEngine); e2.is_paper_mode = True                       # a restart mid-hold
            e2._record_filter_block = lambda n, d, had_room=True: None
            for _ in range(2):
                await e2._willy_hold_block(db, "FOOUSDT", "LONG", "MOMENTUM", price=1.0)
            await asyncio.gather(*list(getattr(e2, "_wh_tasks", set())))
            await e2._willy_hold_flush()
        async with SL() as s_:
            r = (await s_.execute(select(models.WillyHoldBlock))).scalars().all()
        await eng.dispose()
        return r
    rows = asyncio.run(go())
    assert len(rows) == 1 and rows[0].repeats == 5                                                 # 3 + (bump on insert) 1 + 1 flushed = 5


def test_cache_keeps_late_arrivals_confirmed_open(monkeypatch):
    """round-3 M3: a cached order absent from the snapshot is a candidate whatever its age; only the DB read keeps it."""
    import models
    import services.trading_engine as TE
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    old = dt.datetime.utcnow() - dt.timedelta(hours=2)

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        async with SL() as s_:
            s_.add(models.Order(id=201, pair="LATEUSDT", direction="LONG", status="OPEN", is_paper=True, entry_strategy="MOMENTUM",
                                confidence="STRONG_BUY", entry_price=1.0, quantity=1.0, investment=1.0, leverage=20, notional_value=20.0, opened_at=old))
            await s_.commit()
        monkeypatch.setattr(TE, "_open_orders_cache", {"LATEUSDT": [dict(id=201, entry_strategy="MOMENTUM", opened_at=None)]})
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True

        class SnapDB:
            def __init__(self, real):
                self.real, self.n = real, 0

            async def execute(self, q, *a, **k):
                self.n += 1
                if self.n == 1:
                    class _R:
                        def scalars(self):
                            class _S:
                                def all(self):
                                    return []
                            return _S()
                    return _R()
                return await self.real.execute(q, *a, **k)
        async with SL() as db:
            try:
                await e.update_orders_cache(SnapDB(db))
            except Exception:
                pass
        await eng.dispose()
        return {k: [x["id"] for x in v] for k, v in TE._open_orders_cache.items()}
    assert asyncio.run(go()) == {"LATEUSDT": [201]}


def test_reconcile_and_partial_reset_and_adopt_stamp():
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert 'trading_engine._wh_spawn(trading_engine._willy_hold_flush(drop_wid=info["id"]))' in main
    r = main.index('@app.post("/api/reset")')
    body = main[r:main.index("# ----- Balance -----", r)]
    assert body.index("await _willy_drain_hold_writes()") < body.index("await trading_engine.pause(db)")   # drained BEFORE any DML / pause
    assert 'if direction in ("ALL", "LONG"):' in body and 'if direction == "LONG":   # 🔒 (251) FRENZY_WILLY is LONG-only' in body
    assert body.count("await _willy_reset_state(db, is_paper)") == 2
    assert "entry_vol_mcap_ratio=_adopt_vol_mcap(pair)," in main
    import main as M
    assert M._adopt_vol_mcap("NOPEUSDT") is None                                                   # never raises


def test_index_created_on_an_existing_db(tmp_path, monkeypatch):
    """round-3 M2: an EXISTING orders table without the index gets it from init_db's migration (PRAGMA index_list)."""
    import sqlite3
    import database, models
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    dbp = tmp_path / "mig.db"

    async def go():
        eng = create_async_engine(f"sqlite+aiosqlite:///{dbp}")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with eng.begin() as c:
            await c.execute(text("DROP INDEX ix_orders_status_strategy"))
        await eng.dispose()
        before = [r[1] for r in sqlite3.connect(dbp).execute("PRAGMA index_list(orders)")]
        e2 = create_async_engine(f"sqlite+aiosqlite:///{dbp}")
        monkeypatch.setattr(database, "engine", e2)
        await database.init_db()
        await e2.dispose()
        return before, [r[1] for r in sqlite3.connect(dbp).execute("PRAGMA index_list(orders)")]
    before, after = asyncio.run(go())
    assert "ix_orders_status_strategy" not in before and "ix_orders_status_strategy" in after
