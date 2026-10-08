"""🪶 Oct-7 FRENZY_LITE (DECISION_LOG 243 — operator ARMED as a declared exception): FRENZY without the ≥ frenzy_state_vol_mult setup volume,
first ~17 h of the episode, one entry per above-VWAP stretch, NO ATR filter, FRENZY exit, no automatic off.
Pure rules (services/frenzy.py: frenzy_lite_status, frenzy_lite_stretch_id, frenzy_gvol_block) + the engine path + every surface (D11 / D12)."""
import asyncio
import datetime as dt
import json
import os
import re
import sys
import time
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.frenzy import (BAR_MS, FRENZY_LITE_COUNTED, FRENZY_LITE_READY, frenzy_gvol_block, frenzy_lite_status,  # noqa: E402
                             frenzy_lite_stretch_id, frenzy_walk)

NORM = 10_000.0
LAST = 1_790_000_000_000 // BAR_MS * BAR_MS


def _th(**kw):
    base = dict(frenzy_lite_enabled=True, frenzy_lite_min_above_closes=12, frenzy_lite_max_hours=16.75, frenzy_state_vol_mult=100.0,
                frenzy_min_hours=2.0, frenzy_max_hours=96.0, frenzy_min_volume_usd=20e6, frenzy_long_skip_green_bar=True, frenzy_max_atr_pct=2.5,
                frenzy_gvol_max=1.0, frenzy_spike_ret_pct=5.0, frenzy_spike_vol_mult=20.0, frenzy_spike_min_hour_usd=2e6)
    base.update(kw)
    return NS(**base)


def _ep(**kw):
    """a verified flagged episode, NOT in the FRENZY state, held above its average for 14 closes with volume 63× (LITE territory)."""
    base = dict(verified=True, in_state=False, fresh_on=False, above_hour=True, above_streak=14, vol_mult=63.0, hours=5.2, bar_red=True,
                bar_ret_pct=-0.05, last_bar_ts=LAST, atr_pct=1.0)
    base.update(kw)
    return base


# ── pure rules ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_ready_case_and_text():
    ok, code, text = frenzy_lite_status(_ep(), _th(), 50e6)
    assert ok and code == FRENZY_LITE_READY == "FRENZY_LITE_READY"
    assert text == "LITE: held above · vol 63× · 5.2 h"                                       # the monitor's status cell


def test_stretch_id_arithmetic():
    assert frenzy_lite_stretch_id(_ep(above_streak=1)) == LAST                                # the first bar of the streak IS the last bar
    assert frenzy_lite_stretch_id(_ep(above_streak=14)) == LAST - 13 * BAR_MS
    assert frenzy_lite_stretch_id(_ep(above_streak=0)) is None and frenzy_lite_stretch_id(_ep(last_bar_ts=None)) is None
    assert frenzy_lite_stretch_id(None) is None and frenzy_lite_stretch_id(_ep(above_streak="x")) is None
    # the next bar of the same stretch keeps the id; a broken streak that resumes is a NEW stretch
    a = frenzy_lite_stretch_id(_ep(above_streak=14)); b = frenzy_lite_stretch_id(_ep(above_streak=15, last_bar_ts=LAST + BAR_MS))
    c = frenzy_lite_stretch_id(_ep(above_streak=1, last_bar_ts=LAST + 5 * BAR_MS))
    assert a == b and c > a


def test_refusals_in_order():
    th = _th()
    cases = [
        (_ep(in_state=True), "FRENZY_LITE_IN_STATE"),                                        # FRENZY ON → FRENZY / WIDE territory
        (_ep(above_streak=11), "FRENZY_LITE_NOT_ABOVE"),                                     # < 12 closes above the average
        (_ep(vol_mult=100.0), "FRENZY_LITE_VOL_HIGH"),                                       # ≥ 100× = FRENZY's volume (its own sleeve)
        (_ep(vol_mult=None), "FRENZY_LITE_VOL_HIGH"),                                        # unreadable volume: fail closed
        (_ep(hours=1.99), "FRENZY_LITE_TOO_EARLY"),
        (_ep(hours=16.76), "FRENZY_LITE_TOO_LATE"),
        (_ep(hours=None), "FRENZY_LITE_NOT_FLAGGED"),                                     # unreadable age: not a verifiable flag
        (_ep(verified=False), "FRENZY_LITE_NOT_FLAGGED"),
        (_ep(hours=97.0, verified=True), "FRENZY_LITE_NOT_FLAGGED"),                         # flag expired (frenzy_max_hours)
    ]
    for ep, want in cases:
        ok, code, _ = frenzy_lite_status(ep, th, 50e6)
        assert not ok and code == want, (ep, code)
    assert frenzy_lite_status(_ep(hours=2.0), th, 50e6)[0] and frenzy_lite_status(_ep(hours=16.75), th, 50e6)[0]   # both ends inclusive
    assert frenzy_lite_status(_ep(vol_mult=99.9), th, 50e6)[0]
    ok, code, _ = frenzy_lite_status(_ep(), th, 5e6)
    assert not ok and code == "FRENZY_LITE_VOL24_LOW" and code in FRENZY_LITE_COUNTED
    ok, code, _ = frenzy_lite_status(_ep(), th, None)
    assert not ok and code == "FRENZY_LITE_VOL24_LOW"                                         # unreadable 24 h volume: fail closed


def test_green_bar_refused_like_frenzy():
    ok, code, text = frenzy_lite_status(_ep(bar_red=False, bar_ret_pct=0.12), _th(), 50e6)
    assert not ok and code == "FRENZY_LITE_GREEN_BAR" and code in FRENZY_LITE_COUNTED and "+0.120%" in text
    ok, code, text = frenzy_lite_status(_ep(bar_red=False, bar_ret_pct=None), _th(), 50e6)
    assert not ok and code == "FRENZY_LITE_GREEN_BAR" and "unreadable" in text              # unreadable candle: fail closed
    assert frenzy_lite_status(_ep(bar_red=False, bar_ret_pct=0.12), _th(frenzy_long_skip_green_bar=False), 50e6)[0]   # the FRENZY switch rules it


def test_stretch_already_entered():
    sid = frenzy_lite_stretch_id(_ep())
    ok, code, text = frenzy_lite_status(_ep(), _th(), 50e6, judged_stretch_ms=sid)
    assert not ok and code == "FRENZY_LITE_STRETCH_DONE" and text.startswith("LITE: stretch judged · ")
    assert "stretch judged at 13:40 (refused: GREEN_BAR)" in frenzy_lite_status(_ep(), _th(), 50e6, sid, "13:40 (refused: GREEN_BAR)")[2]
    assert frenzy_lite_status(_ep(), _th(), 5e6, judged_stretch_ms=sid)[1] == "FRENZY_LITE_STRETCH_DONE"   # judged beats the 24 h volume read
    assert not frenzy_lite_status(_ep(above_streak=20, last_bar_ts=LAST + 6 * BAR_MS), _th(), 50e6, judged_stretch_ms=sid)[0]   # later bar, same stretch
    assert frenzy_lite_status(_ep(above_streak=12, last_bar_ts=LAST + 40 * BAR_MS), _th(), 50e6, judged_stretch_ms=sid)[0]      # a new stretch may enter
    assert frenzy_lite_status(_ep(), _th(), 50e6, judged_stretch_ms=sid - BAR_MS)[0]                                             # an older stretch's fill


def test_disabled_flag():
    ok, code, _ = frenzy_lite_status(_ep(), _th(frenzy_lite_enabled=False), 50e6)
    assert not ok and code == "FRENZY_LITE_OFF"
    ok, code, _ = frenzy_lite_status(_ep(), NS(), 50e6)                                      # a config without the field = OFF
    assert not ok and code == "FRENZY_LITE_OFF"


def test_no_atr_refusal_even_at_atr_6_pct():
    ok, code, _ = frenzy_lite_status(_ep(atr_pct=6.0), _th(frenzy_max_atr_pct=2.5), 50e6)
    assert ok and code == FRENZY_LITE_READY                                                   # ATR is stamped only (scout LITE_ATR reads it)
    import inspect
    assert "atr" not in inspect.signature(frenzy_lite_status).parameters


def test_min_above_closes_6_and_12():
    for need, streak, want in ((12, 11, False), (12, 12, True), (6, 5, False), (6, 6, True), (6, 11, True)):
        assert frenzy_lite_status(_ep(above_streak=streak, above_hour=streak >= 12), _th(frenzy_lite_min_above_closes=need), 50e6)[0] is want, (need, streak)
    # the rule reads the streak, NOT above_hour (hard-wired to 12 inside frenzy_walk for FRENZY)
    assert frenzy_lite_status(_ep(above_streak=7, above_hour=False), _th(frenzy_lite_min_above_closes=6), 50e6)[0]
    assert not frenzy_lite_status(_ep(above_streak=7, above_hour=True), _th(frenzy_lite_min_above_closes=12), 50e6)[0]


def test_gvol_block_pure_rule():
    th = _th()
    assert frenzy_gvol_block(0.8, th) is None
    assert frenzy_gvol_block(1.0, th) == "GVOL_HIGH" and frenzy_gvol_block(1.4, th) == "GVOL_HIGH"   # ≥ max refuses
    assert frenzy_gvol_block(None, th) == "GVOL_UNREAD" and frenzy_gvol_block("x", th) == "GVOL_UNREAD"   # fail closed
    assert frenzy_gvol_block(None, _th(frenzy_gvol_max=0)) is None and frenzy_gvol_block(5.0, _th(frenzy_gvol_max=0)) is None   # 0 = gate off


def _walk_rows(after=30):
    """quiet tape, a +8 % spike bar, then `after` flat ('red') bars drifting up on ≈ 50× normal volume (below the 100× setup volume)."""
    rows = []; px = 1.0
    for i in range(320):
        rows.append([i * BAR_MS, px, px * 1.001, px * 0.999, px, NORM / 12])
    px2 = px * 1.08
    rows.append([320 * BAR_MS, px, px2, px, px2, 6_000_000.0 / px2])
    px = px2
    for k in range(1, after):
        px *= 1.002
        rows.append([(320 + k) * BAR_MS, px, px * 1.001, px * 0.999, px, 50 * NORM / 12 / px])   # ≈ 50× normal per hour
    return rows


def test_on_a_real_walk():
    """A spike, then the price drifts up on volume BELOW the FRENZY setup volume → never in state; LITE becomes ready at 2 h, the stretch id
    is the first bar of the above streak, and a bar past 16.75 h is refused."""
    rows = _walk_rows()
    th = _th()
    ep = frenzy_walk(rows[:320 + 25], NORM, th)                                               # 24 bars after the spike = 2.0 h
    assert ep and not ep["in_state"] and ep["vol_mult"] < 100
    ok, code, _ = frenzy_lite_status(ep, th, 50e6)
    assert ok, code
    assert frenzy_lite_stretch_id(ep) == ep["last_bar_ts"] - (ep["above_streak"] - 1) * BAR_MS
    early = frenzy_walk(rows[:320 + 20], NORM, th)
    assert frenzy_lite_status(early, th, 50e6)[1] == "FRENZY_LITE_TOO_EARLY"
    assert frenzy_lite_status(dict(ep, hours=17.0), th, 50e6)[1] == "FRENZY_LITE_TOO_LATE"


# ── engine path ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _engine(TE, blocks):
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
    e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
    e._flip_entry_fields = lambda *a, **k: {}
    e._sanitize_open_kwargs = lambda ef, s, d: ef
    return e


class _DB:
    async def rollback(self):
        pass


def _lite_runner(TE):
    """drive _frenzy_lite_eval on one engine across bars. fake_open plays the open path: outcome "opened" or "refused: …" (a refusal at
    the gvol gate / slots / pair held / late …), or raises when boom."""
    bars = [[LAST - (299 - i) * BAR_MS, 1, 1.01, 0.99, 1.0, 10.0] for i in range(300)]
    blocks = []; calls = []
    e = _engine(TE, blocks)

    def run(ep, outcome="opened", boom=False, db_done=False, vol24=50e6, closed=None, real_prev=False):
        del blocks[:]; del calls[:]
        if not real_prev:   # the first-bar re-walk has its own tests (test_first_bar_guarantee_*)
            e._frenzy_lite_prev_missed = lambda *a, **k: False
        else:
            e.__dict__.pop('_frenzy_lite_prev_missed', None)

        async def fake_open(db, fl, ind, bar_open, wide=False, catchup_now=None, lite=False):
            if boom:
                raise RuntimeError("boom")
            calls.append(dict(lite=lite, wide=wide, bar=bar_open, di=fl.get("di_spread"), adx=fl.get("adx_delta")))
            fl["last_fire"] = f"{dt.datetime.utcfromtimestamp(bar_open / 1000):%m-%d %H:%M} LITE {outcome}"
        e._frenzy_open = fake_open

        async def done_db(db, pair, sid):
            return db_done
        e._frenzy_lite_done_db = done_db
        flag = dict(ep, pair="FOOUSDT", volume_24h=vol24, text="held above 1 h · volume 63× < 100×")
        asyncio.run(e._frenzy_lite_eval(_DB(), "FOOUSDT", ep, flag, {}, closed or bars, int(ep["last_bar_ts"]) + BAR_MS, norm_hour=NORM))
        return list(calls), list(blocks), flag
    return e, run


NEXT = dict(above_streak=15, last_bar_ts=LAST + BAR_MS)   # the next bar of the same stretch
SID = LAST - 13 * BAR_MS                                  # _ep()'s stretch id


def test_lite_eval_one_judgement_per_stretch(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e, run = _lite_runner(TE)
    calls, blocks, flag = run(_ep())
    assert len(calls) == 1 and calls[0]["lite"] and not calls[0]["wide"] and blocks == [] and e._fz_lite_done["FOOUSDT"] == SID
    assert flag["lite_ready"] and flag["text"].startswith("LITE: held above") and "di_spread" in flag   # the monitor shows LITE's read
    calls, blocks, flag = run(_ep(**NEXT))                                                          # next bar, same stretch
    assert calls == [] and blocks == [] and flag["lite_code"] == "FRENZY_LITE_STRETCH_DONE" and "(opened)" in flag["text"]
    # 🕒 Oct-7: the judged bar's time rides as UTC ms (the UI shows the operator's clock); the server text says UTC explicitly
    assert flag["lite_judged_ms"] == int(LAST) + BAR_MS and " UTC (opened)" in flag["text"]
    calls, _, _ = run(_ep(above_streak=12, last_bar_ts=LAST + 40 * BAR_MS))                         # a new stretch
    assert len(calls) == 1 and e._fz_lite_done["FOOUSDT"] == LAST + 29 * BAR_MS
    calls, blocks, flag = run(_ep(above_streak=5))                                                  # not a candidate: no counter spam
    assert calls == [] and blocks == [] and flag["text"].startswith("held above 1 h")               # FRENZY's own text kept


def test_green_bar_then_next_bar_no_entry(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e, run = _lite_runner(TE)
    calls, blocks, flag = run(_ep(bar_red=False, bar_ret_pct=0.2))
    assert calls == [] and blocks == ["FRENZY_LITE_GREEN_BAR"] and "stretch judged" in flag["last_fire"] and e._fz_lite_done["FOOUSDT"] == SID
    calls, blocks, flag = run(_ep(**NEXT))                                                          # red next bar — the stretch is done
    assert calls == [] and blocks == [] and flag["lite_code"] == "FRENZY_LITE_STRETCH_DONE" and "(refused: GREEN_BAR)" in flag["text"]
    assert flag["lite_judged_ms"] == int(LAST) + BAR_MS and " UTC (refused: GREEN_BAR)" in flag["text"]   # 🕒 Oct-7


def test_open_path_refusal_then_next_bar_no_entry(monkeypatch):
    """gvol high / unread, slots, pair held, late, open refused: the open path refused the first signal bar → the stretch is done."""
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    for why in ("refused: market volume 1.20× normal ≥ 1×", "refused: market volume unreadable", "refused: 2 FRENZY_LITE positions open (max 2)",
                "refused: the pair already has an open position", "refused by the open path (slots / balance / cooldown / price moved)"):
        e, run = _lite_runner(TE)
        calls, _, _ = run(_ep(), outcome=why)
        assert len(calls) == 1 and e._fz_lite_done["FOOUSDT"] == SID, why
        calls, blocks, flag = run(_ep(**NEXT))
        assert calls == [] and blocks == [] and flag["lite_code"] == "FRENZY_LITE_STRETCH_DONE" and why in flag["text"], why


def test_vol24_low_then_next_bar_can_enter(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e, run = _lite_runner(TE)
    calls, blocks, flag = run(_ep(), vol24=5e6)                                                     # pre-signal: counted, not judged
    assert calls == [] and blocks == ["FRENZY_LITE_VOL24_LOW"] and "FOOUSDT" not in (getattr(e, "_fz_lite_done", None) or {})
    calls, blocks, _ = run(_ep(**NEXT))
    assert len(calls) == 1 and blocks == [] and e._fz_lite_done["FOOUSDT"] == SID


def test_crash_in_lite_path_does_not_burn_the_stretch(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e, run = _lite_runner(TE)
    real_mark = e._frenzy_lite_mark_done

    async def mark_then_crash(pair, sid, restore=False, prev=None):
        await real_mark(pair, sid, restore=restore, prev=prev)
        if not restore:
            raise RuntimeError("boom after the mark")   # our own exception before the open ran
    e._frenzy_lite_mark_done = mark_then_crash
    calls, blocks, _ = run(_ep())
    assert calls == [] and blocks == ["FRENZY_LITE_FAILED"] and "FOOUSDT" not in e._fz_lite_done   # mark rolled back
    e._frenzy_lite_mark_done = real_mark
    calls, blocks, _ = run(_ep(**NEXT))
    assert len(calls) == 1 and blocks == []                                                         # the next bar may still enter
    e2, run2 = _lite_runner(TE)
    calls, blocks, _ = run2(_ep(), boom=True)                                                       # the open itself raised → crash-isolated
    assert blocks == ["FRENZY_LITE_FAILED"]


def test_first_bar_guarantee_cold_start_mid_stretch(monkeypatch):
    """the process (re)starts mid-stretch: the previous bar was already a signal bar nobody evaluated → no entry, stretch judged."""
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    rows = _walk_rows()
    closed = rows[:320 + 27]                                                    # bar 26: 2.17 h; bar 25 (2.08 h) was already a signal bar
    ep = frenzy_walk(closed, NORM, _th())
    assert ep["above_streak"] > 12 and frenzy_lite_status(ep, _th(), 50e6)[0]
    e, run = _lite_runner(TE)
    calls, blocks, flag = run(ep, closed=closed, real_prev=True)
    assert calls == [] and blocks == [] and flag["lite_code"] == "FRENZY_LITE_STRETCH_DONE" and "first signal bar not seen" in flag["text"]
    assert e._fz_lite_done["FOOUSDT"] == frenzy_lite_stretch_id(ep)
    assert flag.get("lite_judged_ms") is None   # 🕒 Oct-7: a backstop note carries no bar time
    nxt = frenzy_walk(rows[:320 + 28], NORM, _th())
    calls, _, flag = run(nxt, closed=rows[:320 + 28], real_prev=True)           # and the stretch stays done
    assert calls == [] and flag["lite_code"] == "FRENZY_LITE_STRETCH_DONE"


def test_first_bar_guarantee_previous_bar_too_early_enters(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    closed = _walk_rows()[:320 + 25]                                            # bar 24 = 2.0 h; bar 23 (1.92 h) was too early
    ep = frenzy_walk(closed, NORM, _th())
    assert ep["above_streak"] > 12
    _, run = _lite_runner(TE)
    calls, blocks, _ = run(ep, closed=closed, real_prev=True)
    assert len(calls) == 1 and blocks == []


def test_first_bar_guarantee_streak_equal_need_skips_the_walk(monkeypatch):
    import services.trading_engine as TE
    closed = _walk_rows()[:320 + 27]
    ep = frenzy_walk(closed, NORM, _th())
    th = _th(frenzy_lite_min_above_closes=int(ep["above_streak"]))             # streak == need: this bar IS the stretch's first signal bar
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th))
    monkeypatch.setattr(TE, "frenzy_walk", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no extra walk when streak == need")))
    _, run = _lite_runner(TE)
    calls, blocks, _ = run(ep, closed=closed, real_prev=True)
    assert len(calls) == 1 and blocks == []


def test_first_bar_guarantee_previous_bar_seen_skips_the_walk(monkeypatch):
    """a previous bar this process evaluated (e.g. refused for the 24 h volume) is never a 'missed' first bar."""
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    rows = _walk_rows()
    e, run = _lite_runner(TE)
    p = frenzy_walk(rows[:320 + 26], NORM, _th())
    calls, blocks, _ = run(p, closed=rows[:320 + 26], real_prev=True, vol24=5e6)
    assert calls == [] and blocks == ["FRENZY_LITE_VOL24_LOW"]
    monkeypatch.setattr(TE, "frenzy_walk", lambda *a, **k: (_ for _ in ()).throw(AssertionError("previous bar was seen")))
    ep = frenzy_walk(rows[:320 + 27], NORM, _th())   # this module's own import — the engine's frenzy_walk is the patched one
    calls, blocks, _ = run(ep, closed=rows[:320 + 27], real_prev=True)
    assert len(calls) == 1 and blocks == []


def test_first_bar_walk_unreadable_fails_closed(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    monkeypatch.setattr(TE, "frenzy_walk", lambda *a, **k: None)
    e, run = _lite_runner(TE)
    calls, blocks, flag = run(_ep(), real_prev=True)
    assert calls == [] and "first signal bar not seen" in flag["text"] and e._fz_lite_done["FOOUSDT"] == SID


def test_vol24_low_counted_once_per_stretch(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    _, run = _lite_runner(TE)
    assert run(_ep(), vol24=5e6)[1] == ["FRENZY_LITE_VOL24_LOW"]
    calls, blocks, flag = run(_ep(**NEXT), vol24=5e6)
    assert blocks == [] and calls == [] and "24 h volume" in flag["last_fire"]                 # same stretch: not counted again
    assert run(_ep(above_streak=12, last_bar_ts=LAST + 40 * BAR_MS), vol24=5e6)[1] == ["FRENZY_LITE_VOL24_LOW"]   # a new stretch counts


def test_db_unreadable_marks_the_stretch(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e2, _ = _lite_runner(TE)

    async def unread(db, pair, sid):
        return None   # the tri-state backstop: the DB read failed
    flag = dict(_ep(), pair="FOOUSDT", volume_24h=50e6)
    e2._frenzy_lite_prev_missed = lambda *a, **k: False
    e2._frenzy_lite_done_db = unread
    opened = []

    async def no_open(*a, **k):
        opened.append(1)
    e2._frenzy_open = no_open
    asyncio.run(e2._frenzy_lite_eval(_DB(), "FOOUSDT", _ep(), flag, {}, [], LAST + BAR_MS, norm_hour=NORM))
    assert opened == [] and "DB unreadable" in flag["text"] and e2._fz_lite_done["FOOUSDT"] == SID


def test_stale_last_fire_never_becomes_the_note(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    e, _ = _lite_runner(TE)

    async def silent_open(db, fl, ind, bar_open, wide=False, catchup_now=None, lite=False):
        return None   # an open path that records no outcome
    flag = dict(_ep(), pair="FOOUSDT", volume_24h=50e6, last_fire="10-06 09:00 LITE opened")
    e._frenzy_lite_prev_missed = lambda *a, **k: False

    async def nodb(db, pair, sid):
        return False
    e._frenzy_lite_done_db = nodb; e._frenzy_open = silent_open
    asyncio.run(e._frenzy_lite_eval(_DB(), "FOOUSDT", _ep(), flag, {}, [], LAST + BAR_MS, norm_hour=NORM))
    assert "opened" not in e._fz_lite_note["FOOUSDT"][1] and "no outcome recorded" in e._fz_lite_note["FOOUSDT"][1]


def test_db_backstop_after_restart(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_th()))
    _, run = _lite_runner(TE)
    calls, blocks, flag = run(_ep(), db_done=True)                                                  # a LITE fill on this stretch before a restart
    assert calls == [] and blocks == [] and flag["lite_code"] == "FRENZY_LITE_STRETCH_DONE"
    _, run = _lite_runner(TE)
    calls, blocks, _ = run(_ep(bar_red=False, bar_ret_pct=0.2), db_done=True)
    assert calls == [] and blocks == []                                                             # already judged: no green counter either


def test_lite_done_survives_a_restart_via_botstate_json():
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    e._fz_lite_done = {"FOOUSDT": 123}
    js = json.loads(e._frenzy_state_json())
    assert js["lite_done"] == {"FOOUSDT": 123} and "on_done" in js and "unread" in js


def test_frenzy_open_lite_path(monkeypatch):
    """_frenzy_open(lite=True): own tag / slots / day count; refused while the pair holds ANY open position (FRENZY_LITE_PAIR_HELD);
    market volume fail-closed with the LITE prefix; open_position gets frenzy_lite=True, never frenzy_long / the strong bump. The stretch is
    marked by the caller BEFORE this runs (test_open_path_refusal_then_next_bar_no_entry) — _frenzy_open never touches it."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, config as C
    import services.trading_engine as TE
    th = C.trading_config.thresholds
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**6)
    monkeypatch.setattr(TE, "_current_btc_trend_gap_pct", 0.10)   # 🐻 (250) a non-bearish day: the bearish gate passes (its own tests: test_frenzy_bearish_day.py)
    bar_open = int(time.time() // 300) * 300_000
    now = dt.datetime.utcnow()

    async def run(rows, gv=0.5, slots=2, ok=True):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        old = (th.frenzy_gvol_max, th.frenzy_lite_max_slots)
        try:
            th.frenzy_gvol_max = 1.0; th.frenzy_lite_max_slots = slots
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                for pair, strat, status in rows:
                    db.add(models.Order(pair=pair, direction="LONG", status=status, is_paper=True, entry_strategy=strat, confidence="STRONG_BUY",
                                        entry_price=1.0, quantity=1.0, investment=1.0, leverage=20, notional_value=20.0, opened_at=now - dt.timedelta(hours=3)))
                await db.commit()
                blocks = []; opened = []
                e = _engine(TE, blocks)

                async def gvv(sig, wait=True):
                    return gv
                e._frenzy_gvol_value = gvv

                async def fake_open(**kw):
                    opened.append(kw); return object() if ok else None
                e.open_position = fake_open

                async def mark(*a, **k):
                    raise AssertionError("_frenzy_open must not mark the stretch")
                e._frenzy_lite_mark_done = mark
                flag = dict(pair="FOOUSDT", spike_ts=bar_open - 7_200_000, hours=5.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=63.0, run_pct=20.0,
                            atr_pct=6.0, volume_24h=5e7, price=1.0, live_price=1.0, adx_delta=1.0, di_spread=2.0, above_streak=14, above_share=90.0,
                            bar_ret_pct=-0.1)
                await e._frenzy_open(db, flag, {}, bar_open, lite=True)
        finally:
            th.frenzy_gvol_max, th.frenzy_lite_max_slots = old
            await eng.dispose()
        return blocks, opened, None, flag.get("last_fire") or ""
    b, o, d, lf = asyncio.run(run([]))
    assert b == [] and len(o) == 1 and o[0]["frenzy_lite"] is True and o[0]["frenzy_long"] is False and o[0]["frenzy_wide"] is False
    assert o[0]["frenzy_strong"] is False and o[0]["entry_atr_pct"] == 6.0 and o[0]["entry_frenzy_gvol"] == 0.5     # ATR 6 % stamped, not refused
    assert "LITE opened" in lf
    b, o, d, lf = asyncio.run(run([], ok=False))
    assert "FRENZY_LITE_OPEN_REFUSED" in b and "LITE refused" in lf                          # the stretch was judged by the caller: done
    b, o, d, lf = asyncio.run(run([("FOOUSDT", "MOMENTUM", "OPEN")]))
    assert b == ["FRENZY_LITE_PAIR_HELD"] and o == [] and "already has an open position" in lf
    b, o, _, _ = asyncio.run(run([("FOOUSDT", "MOMENTUM", "CLOSED")]))
    assert b == [] and len(o) == 1                                                            # a closed position does not hold the pair
    b, o, _, lf = asyncio.run(run([("A", "FRENZY_LITE", "OPEN"), ("B", "FRENZY_LITE", "OPEN")]))
    assert b == ["FRENZY_LITE_MAX_SLOTS"] and o == [] and "LITE refused" in lf
    b, o, _, _ = asyncio.run(run([("A", "FRENZY_WIDE", "OPEN"), ("B", "FRENZY_WIDE", "OPEN")]))
    assert b == [] and len(o) == 1                                                            # WIDE's slots are not LITE's
    b, o, _, lf = asyncio.run(run([], gv=1.2))
    assert b == ["FRENZY_LITE_GVOL_HIGH"] and o == [] and "LITE refused: market volume" in lf
    b, o, _, _ = asyncio.run(run([], gv=None))
    assert b == ["FRENZY_LITE_GVOL_UNREAD"] and o == []


# ── wiring / every surface ─────────────────────────────────────────────────────────────────────────────────────────────────────────

FIELDS = {"frenzy_lite_enabled": True, "frenzy_lite_invest_mult": 1.0, "frenzy_lite_lev_mult": 0.32, "frenzy_lite_max_slots": 2,   # 0.32: DECISION_LOG 247
          "frenzy_lite_max_hours": 16.75, "frenzy_lite_min_above_closes": 12}


def test_config_d11():
    import config as C
    mf = C.SignalThresholds.model_fields
    assert mf["frenzy_lite_enabled"].default is False                                          # OFF in code; the JSON arms it
    assert mf["frenzy_lite_lev_mult"].default == 0.2 and mf["frenzy_lite_max_hours"].default == 16.75 and mf["frenzy_lite_min_above_closes"].default == 12
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for k, v in FIELDS.items():
        assert cfg[k] == v, k
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 243" in src and "HOLD_LOWVOL_EARLY_FRENZY_FILTERS_2026-10-06.md" in src


def test_ui_inputs_load_save_and_reports():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for _id in ("config-fz-lite-enabled", "config-fz-lite-invest-mult", "config-fz-lite-lev-mult", "config-fz-lite-slots",
                "config-fz-lite-max-hours", "config-fz-lite-min-above"):
        assert html.count(f'id="{_id}"') == 1, _id
    for row in ("['config-fz-lite-invest-mult', 'frenzy_lite_invest_mult', 1.0]", "['config-fz-lite-lev-mult', 'frenzy_lite_lev_mult', 0.2]",
                "['config-fz-lite-slots', 'frenzy_lite_max_slots', 2]", "['config-fz-lite-max-hours', 'frenzy_lite_max_hours', 16.75]",
                "['config-fz-lite-min-above', 'frenzy_lite_min_above_closes', 12]"):
        assert html.count(row) == 1, row                                                       # FRENZY_NUM_FIELDS drives load + save
    assert "_key === 'frenzy_lite_max_slots' || _key === 'frenzy_lite_min_above_closes' || _key === 'frenzy_willy_max_slots' || _key === 'frenzy_willy_max_hold_minutes' || _key === 'frenzy_willy_red_max_wait_minutes') ? Math.round(x)" in html   # ints
    assert "frenzy_lite_enabled: document.getElementById('config-fz-lite-enabled')?.checked ?? false" in html
    assert "getElementById('config-fz-lite-enabled'); if (_e) _e.checked = config.thresholds.frenzy_lite_enabled === true" in html
    assert html.count('data-ss="fzl"') == 1 and "S.fzl = {" in html                           # ⚖️ Sleeve Sizing row
    # the config text-report line (_buildConfigLines → BOTH exports) and the monitor line (dashboard + BOTH exports)
    assert "LITE ${_bt.frenzy_lite_enabled === true ? 'ON' : 'OFF'}" in html and "no ATR cap · no auto-off) · WILLY ${_bt.frenzy_willy_enabled === true" in html
    assert html.count("lines.push(..._buildConfigLines(cfg, changelog, hr, hr2, status));") == 2
    assert "LITE ${m.lite_enabled ? 'ON' : 'OFF'}" in html and html.count("lines.push(...frenzyReportLines(perf, hr2));") == 2
    assert html.count("FRENZY Fills (FRENZY_LONG · FRENZY_WIDE · FRENZY_LITE · FRENZY_WILLY)") == 2            # UI header + the shared export block
    assert '<option value="FRENZY_LITE">' in html and "strat === 'FRENZY_LITE' || strat === 'FRENZY_WILLY') return tags" in html
    assert "['FRENZY_LONG', 'FRENZY_WIDE', 'FRENZY_LITE'].includes(o.entry_strategy" in html     # FRENZY exit badge
    assert "(?:WIDE |LITE |WILLY [AB] )?opened/" in html and "f.ready || f.lite_ready" in html


def test_engine_and_api_wiring():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert 'FRENZY_STRATEGIES = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY")' in eng            # FRENZY exit / hold cap / urgent close
    assert "if _lite and not ep.get('in_state'):" in eng and "await self._frenzy_lite_eval(db, pair, ep, flag, ind, closed, bar_open, norm_hour=_nc[1])" in eng
    assert "if not (_on or _obs or _wide or _lite or _willy):" in eng and "if _on or _wide or _lite or _willy:" in eng
    assert '"FRENZY_LITE" if (_frenzy and frenzy_lite)' in eng                        # tag → frenzy_lite_* size fields
    i = eng.index("async def _frenzy_lite_eval("); body = eng[i:eng.index("@staticmethod", i)]
    assert "atr" not in re.sub(r'""".*?"""', "", body, flags=re.S).replace("ATR", "")          # no ATR filter in the LITE path
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "'Frenzy-Lite'" in main and main.count("'FRENZY_LITE'") >= 3 and '("FRENZY_LITE", "LONG"): "FRENZY LITE Long"' in main
    assert '"lite_enabled": bool(getattr(_th, \'frenzy_lite_enabled\', False))' in main and "getattr(_th, 'frenzy_lite_enabled', False) or getattr(_th, 'frenzy_willy_enabled', False)):" in main
    assert '("FRENZY_LITE", "FRENZY_LITE (held above' in main and '"lite_ready": bool(f.get(\'lite_ready\'))' in main


def test_open_by_sleeve_and_sleeve_rows_know_lite():
    import main as M
    rows = M._open_by_sleeve([NS(entry_strategy="FRENZY_LITE", direction="LONG")])
    assert any("LITE" in str(r) for r in (rows if isinstance(rows, list) else [rows]))


# ── 243 review fixes ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _mem_sl():
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models

    async def mk(state_json=None):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        async with SL() as s:
            s.add(models.BotState(is_running=False, frenzy_last_judged_bar_ms=LAST - BAR_MS, frenzy_unjudged_json=state_json)); await s.commit()
        return eng, SL
    return mk


def test_judged_set_before_get_never_wipes_stored_marks(monkeypatch):
    """lost-state wipe: a _frenzy_judged_set from a fresh process (e.g. main.frenzy_loop's 'FRENZY off' call, clear_pairs=True) merges the
    stored JSON first — lite_done / on_done survive; and a restarted process reads them back."""
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    stored = json.dumps({"unread": {"AUSDT": LAST - 10 * BAR_MS}, "on_done": {"NMRUSDT": LAST - 30 * BAR_MS}, "lite_done": {"FOOUSDT": SID}})

    async def go():
        eng, SL = await _mem_sl()(stored); monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
        e1 = object.__new__(TE.TradingEngine)
        await e1._frenzy_judged_set(LAST, clear_pairs=True)                     # set BEFORE any get
        async with SL() as s:
            js = json.loads((await s.execute(select(models.BotState.frenzy_unjudged_json))).scalar())
        e2 = object.__new__(TE.TradingEngine)                                    # a restart: round-trip through _frenzy_judged_get
        ms = await e2._frenzy_judged_get()
        await eng.dispose()
        return js, e2, ms
    js, e2, ms = asyncio.run(go())
    assert js["lite_done"] == {"FOOUSDT": SID} and js["on_done"] == {"NMRUSDT": LAST - 30 * BAR_MS}
    assert js["unread"] == {}                                                    # clear_pairs still clears the unread map (after the merge)
    assert e2._fz_lite_done == {"FOOUSDT": SID} and e2._fz_on_done == {"NMRUSDT": LAST - 30 * BAR_MS} and ms == LAST


def test_judged_set_merges_memory_and_store(monkeypatch):
    import services.trading_engine as TE
    stored = json.dumps({"on_done": {"NMRUSDT": LAST - 30 * BAR_MS}, "lite_done": {"FOOUSDT": SID}})

    async def go():
        eng, SL = await _mem_sl()(stored); monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
        e = object.__new__(TE.TradingEngine)
        e._fz_lite_done = {"BARUSDT": LAST}                                      # marked in memory before the first load (not persisted)
        await e._frenzy_judged_set(LAST)
        e2 = object.__new__(TE.TradingEngine); await e2._frenzy_judged_get()
        await eng.dispose()
        return e2
    e2 = asyncio.run(go())
    assert e2._fz_lite_done == {"FOOUSDT": SID, "BARUSDT": LAST} and "NMRUSDT" in e2._fz_on_done


def test_frenzy_and_wide_refused_on_a_pair_held_by_lite(monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, config as C
    import services.trading_engine as TE
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**6)
    bar_open = int(time.time() // 300) * 300_000
    now = dt.datetime.utcnow()

    async def run(wide, strat="FRENZY_LITE"):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with async_sessionmaker(eng, expire_on_commit=False)() as db:
            db.add(models.Order(pair="FOOUSDT", direction="LONG", status="OPEN", is_paper=True, entry_strategy=strat, confidence="STRONG_BUY",
                                entry_price=1.0, quantity=1.0, investment=1.0, leverage=4, notional_value=4.0, opened_at=now - dt.timedelta(hours=1)))
            await db.commit()
            blocks = []; opened = []
            e = _engine(TE, blocks)

            async def gvv(sig, wait=True):
                return 0.5
            e._frenzy_gvol_value = gvv

            async def fake_open(**kw):
                opened.append(kw); return None
            e.open_position = fake_open
            flag = dict(pair="FOOUSDT", spike_ts=bar_open - 7_200_000, hours=3.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=150.0, run_pct=20.0,
                        atr_pct=1.5, volume_24h=5e7, price=1.0, live_price=1.0, above_streak=20, bar_ret_pct=0.1, code="FRENZY_GREEN_BAR")
            await e._frenzy_open(db, flag, {}, bar_open, wide=wide)
        await eng.dispose()
        return blocks, opened, flag.get("last_fire") or ""
    b, o, lf = asyncio.run(run(False))
    assert b == ["FRENZY_PAIR_HELD_BY_LITE"] and o == [] and "open FRENZY_LITE position" in lf
    b, o, _ = asyncio.run(run(True))
    assert b == ["FRENZY_WIDE_PAIR_HELD_BY_LITE"] and o == []
    b, o, _ = asyncio.run(run(False, strat="MOMENTUM"))
    assert "FRENZY_PAIR_HELD_BY_LITE" not in b and len(o) == 1                  # another sleeve's position: open_position's PAIR_HELD as before


def test_gvol_block_non_finite_is_unread():
    th = _th()
    for bad in (float("nan"), float("inf"), float("-inf"), "nan"):
        assert frenzy_gvol_block(bad, th) == "GVOL_UNREAD", bad


def test_min_above_and_max_hours_guards():
    from services.frenzy import frenzy_lite_need, frenzy_lite_hmax
    assert frenzy_lite_need(_th(frenzy_lite_min_above_closes=0)) == 12 and frenzy_lite_need(_th(frenzy_lite_min_above_closes=-3)) == 12
    assert frenzy_lite_need(_th(frenzy_lite_min_above_closes="x")) == 12 and frenzy_lite_need(_th(frenzy_lite_min_above_closes=6)) == 6
    assert not frenzy_lite_status(_ep(above_streak=11), _th(frenzy_lite_min_above_closes=0), 50e6)[0]   # 0 → 12 used, not "any streak"
    assert frenzy_lite_hmax(_th(frenzy_lite_max_hours=2.0)) == 16.75 and frenzy_lite_hmax(_th(frenzy_lite_max_hours=0)) == 16.75
    assert frenzy_lite_hmax(_th(frenzy_lite_max_hours=10.0)) == 10.0
    assert frenzy_lite_status(_ep(hours=12.0), _th(frenzy_lite_max_hours=1.0), 50e6)[0]                 # ≤ min hours → 16.75 used
    import main as M, config as C
    th = C.trading_config.thresholds; old = (th.frenzy_lite_min_above_closes, th.frenzy_lite_max_hours)
    try:
        th.frenzy_lite_min_above_closes, th.frenzy_lite_max_hours = 0, 1.0
        m = M._frenzy_monitor_payload()
        assert m["lite_min_above"] == 12 and m["lite_max_hours"] == 16.75                              # the monitor shows what is used
    finally:
        th.frenzy_lite_min_above_closes, th.frenzy_lite_max_hours = old
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert 'min="2" id="config-fz-lite-max-hours"' in html and 'min="1" id="config-fz-lite-min-above"' in html


def test_lite_logs_and_flags_cell():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("async def _frenzy_open("); body = eng[i:eng.index("async def _maybe_open_surge(", i)]
    assert '_sig = "LITE signal" if lite else "setup ON"' in body and 'f"[{_es}] {pair}: setup ON' not in body
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "${(f.ready || f.lite_ready) ? 'text-emerald-400 font-semibold' : (f.in_state ? 'text-orange-300' : 'text-gray-400')}" in html
