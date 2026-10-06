"""⏪ Oct-6 FRENZY catch-up — a fresh ON bar that closed while no pass judged (pause / restart / outage) is judged once, late, as of THAT bar.
Pure rules (services/frenzy.py: on_bar_ts, frenzy_catchup_check, frenzy_catchup_moved, frenzy_vol24_at) + the engine path + every surface."""
import asyncio
import json
import os
import re
import sys
import time
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.frenzy import (BAR_MS, FRENZY_CATCHUP_OK, FRENZY_CATCHUP_STALE, frenzy_catchup_check, frenzy_catchup_moved,  # noqa: E402
                             frenzy_vol24_at, frenzy_walk)

NORM = 10_000.0


class TH:
    frenzy_spike_ret_pct = 5.0; frenzy_spike_vol_mult = 20.0; frenzy_spike_min_hour_usd = 2e6; frenzy_state_vol_mult = 100.0
    frenzy_min_hours = 2.0; frenzy_max_hours = 96.0; frenzy_min_volume_usd = 20e6; frenzy_max_atr_pct = 2.0
    frenzy_stop_pct = 3.0; frenzy_long_skip_green_bar = True; frenzy_wide_enabled = False; frenzy_max_entry_dislocation_pct = 1.0


def _bars(n_quiet=320, after=40, spike_vol=6_000_000.0, run_vol=2_000_000.0, drift=0.002, dip_at=None):
    """test_frenzy_sleeve's fixture: quiet tape, a +8 % spike bar on heavy volume, then `after` bars drifting up on heavy volume
    (open == close → every bar is flat = 'red' for the candle rule). The setup turns ON 24 bars after the spike."""
    rows = []; px = 1.0
    for i in range(n_quiet):
        rows.append([i * BAR_MS, px, px * 1.001, px * 0.999, px, NORM / 12])
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


def T(k):   # open ms of the k-th bar after the spike bar
    return (320 + k) * BAR_MS


# ── pure rules ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_on_bar_ts_is_the_bar_the_stretch_turned_on():
    assert frenzy_walk(_bars(after=23), NORM, TH)["on_bar_ts"] is None                  # not in state yet
    ep = frenzy_walk(_bars(after=24), NORM, TH)
    assert ep["fresh_on"] and ep["on_bar_ts"] == ep["last_bar_ts"] == T(24)
    for a in (25, 27, 35):                                                               # still ON, entry bar passed: the ON bar stays 24
        ep = frenzy_walk(_bars(after=a), NORM, TH)
        assert ep["in_state"] and not ep["fresh_on"] and ep["on_bar_ts"] == T(24)
    b = _bars(after=60, dip_at=30)                                                       # an hour off, then ON again at bar 44
    for k in range(24, 61):
        ep = frenzy_walk(b[:320 + 1 + k], NORM, TH)
        assert bool(ep["fresh_on"]) == (ep["on_bar_ts"] == ep["last_bar_ts"])           # fresh_on ⇔ the ON bar is the last bar
        if ep["in_state"]:
            assert ep["on_bar_ts"] == (T(24) if k < 44 else T(44))
        else:
            assert ep["on_bar_ts"] is None


def test_catchup_check_only_an_unjudged_on_bar():
    ep = frenzy_walk(_bars(after=27), NORM, TH)                                          # ON at 24, now at 27 (3 bars later)
    assert frenzy_catchup_check(ep, T(22), 6) == (FRENZY_CATCHUP_OK, 3)                  # bars 23–26 never judged → recover bar 24
    assert frenzy_catchup_check(ep, T(23), 6) == (FRENZY_CATCHUP_OK, 3)
    assert frenzy_catchup_check(ep, T(24), 6) == (None, None)                            # bar 24 WAS judged (normal path) → never
    assert frenzy_catchup_check(ep, T(26), 6) == (None, None)
    assert frenzy_catchup_check(ep, None, 6) == (None, None)                             # cold start: no stored bar → do nothing
    assert frenzy_catchup_check(ep, T(22), 0) == (None, None)                            # 0 = off
    assert frenzy_catchup_check(ep, T(22), 2) == (FRENZY_CATCHUP_STALE, 3)               # older than max → stale
    assert frenzy_catchup_check(frenzy_walk(_bars(after=24), NORM, TH), T(22), 6) == (None, None)   # fresh → the normal path
    assert frenzy_catchup_check(frenzy_walk(_bars(after=23), NORM, TH), T(10), 6) == (None, None)   # not in state
    assert frenzy_catchup_check(None, T(22), 6) == (None, None) and frenzy_catchup_check({}, T(22), 6) == (None, None)


def test_simulated_pause_recovers_exactly_the_missed_fresh_bar():
    """Walk the tape bar by bar; passes run except inside a pause window. A catch-up fires only when the fresh bar fell inside the window,
    on the first pass after it — never for a bar a pass judged."""
    b = _bars(after=40)

    def run(paused):
        last = None; fires = []
        for k in range(0, 41):
            if k in paused:
                continue
            ep = frenzy_walk(b[:320 + 1 + k], NORM, TH)
            if ep:
                st, _ = frenzy_catchup_check(ep, last, 6)
                if st:
                    fires.append((k, st))
            last = T(k)                                                                   # the pass completed: bar k judged
        return fires
    assert run(set()) == []                                                               # no pause: the fresh bar was judged normally
    assert run({23, 24, 25, 26}) == [(27, FRENZY_CATCHUP_OK)]                            # NMR-like: ON bar inside the pause → once
    assert run({20, 21, 22, 23}) == []                                                    # pause ended before the ON bar
    assert run(set(range(22, 34))) == [(34, FRENZY_CATCHUP_STALE)]                       # resumed 10 bars later → stale
    assert run({25, 26, 27}) == []                                                        # pause after the ON bar was judged


def test_dislocation_guard_and_24h_volume_on_the_bar():
    assert frenzy_catchup_moved(1.0, 1.009, 1.0) is False and frenzy_catchup_moved(1.0, 0.991, 1.0) is False
    assert frenzy_catchup_moved(1.0, 1.011, 1.0) is True and frenzy_catchup_moved(1.0, 0.98, 1.0) is True
    assert frenzy_catchup_moved(1.0, 1.0, 0) is True                                     # guard off → refuse (fail-closed)
    assert frenzy_catchup_moved(0, 1.0, 1.0) is True and frenzy_catchup_moved(1.0, None, 1.0) is True
    rows = [[i * BAR_MS, 1, 2.0, 1.0, 3.0, 10.0] for i in range(300)]                    # typical price 2 × 10 = 20 per bar
    assert frenzy_vol24_at(rows) == 288 * 20.0 and frenzy_vol24_at(rows[:287]) is None and frenzy_vol24_at(None) is None
    import services.trading_engine as TE
    bars = TE.TradingEngine._frenzy_catchup_bars
    assert bars(T(22), T(27), 6) == [T(23), T(24), T(25), T(26)]
    assert bars(T(10), T(27), 3) == [T(24), T(25), T(26)]                                 # capped at max bars
    assert bars(T(26), T(27), 6) == [] and bars(None, T(27), 6) == [] and bars(T(22), T(27), 0) == []


# ── the engine path ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _engine(TE, blocks):
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
    e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
    e._flip_entry_fields = lambda *a, **k: {}
    e._sanitize_open_kwargs = lambda ef, s, d: ef
    return e


def test_catchup_judges_the_on_bar_once_and_is_crash_isolated(monkeypatch):
    import services.trading_engine as TE
    th = TH()
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th))
    closed = _bars(after=27); now_bar = T(28)                                             # this pass judges bar 27 (closed at T(28))
    info = dict(price=closed[-1][4], change_24h=20.0, range_24h=30.0)

    def run(prev, max_bars=6, boom=False, wide=False, e=None, blocks=None):
        blocks = [] if blocks is None else blocks; calls = []
        e = e or _engine(TE, blocks)

        async def fake_open(db, fl, ind, sig, wide=False, catchup_now=None):
            if boom:
                raise RuntimeError("boom")
            calls.append(dict(sig=sig, wide=wide, catchup_now=catchup_now, price=fl["price"], atr=fl["atr_pct"], vol24=fl["volume_24h"],
                              fresh=fl["fresh_on"], bar_ret=fl["bar_ret_pct"]))
            fl["last_fire"] = f"{'WIDE ' if wide else ''}opened"
        e._frenzy_open = fake_open
        ep = frenzy_walk(closed, NORM, th); flag = dict(ep, pair="NMRUSDT")

        class DB:
            async def rollback(self):
                pass
        asyncio.run(e._frenzy_catchup(DB(), "NMRUSDT", ep, closed, NORM, flag, info, now_bar, prev, max_bars, True, wide))
        return calls, blocks, flag.get("last_fire") or "", e

    calls, blocks, lf, e = run(T(22))
    assert len(calls) == 1 and calls[0]["sig"] == T(24) + BAR_MS and calls[0]["catchup_now"] == now_bar and not calls[0]["wide"]
    assert calls[0]["fresh"] and abs(calls[0]["price"] - closed[320 + 24][4]) < 1e-12                # judged AS OF the ON bar
    assert calls[0]["vol24"] == frenzy_vol24_at(closed[:320 + 25]) and lf.startswith("⏪ catch-up · ") and "opened" in lf
    calls2, _, _, _ = run(T(22), e=e)                                                                 # once per (pair, spike, ON bar)
    assert calls2 == []
    assert run(T(24))[0] == [] and run(T(26))[0] == [] and run(None)[0] == []                       # judged normally / cold start
    calls, blocks, lf, _ = run(T(22), max_bars=2)
    assert calls == [] and blocks == ["FRENZY_CATCHUP_STALE"] and "3 bars old (max 2)" in lf
    calls, blocks, _, _ = run(T(22), boom=True)                                                       # a crash stays inside the pair
    assert blocks == ["FRENZY_CATCHUP_FAILED"]
    th.frenzy_max_atr_pct = 0.01                                                                      # FRENZY refuses on ATR (ON bar) …
    calls, blocks, lf, _ = run(T(22))
    assert calls == [] and blocks == ["FRENZY_ATR_HIGH", "FRENZY_CATCHUP_REFUSED"] and "refused: ATR" in lf   # M-8: shared reason + the cohort's own
    th.frenzy_wide_enabled = True                                                                     # … and WIDE takes it, also as of the ON bar
    calls, blocks, lf, _ = run(T(22), wide=True)
    assert len(calls) == 1 and calls[0]["wide"] and calls[0]["sig"] == T(24) + BAR_MS and calls[0]["catchup_now"] == now_bar


def test_frenzy_open_catchup_guards(monkeypatch):
    """Catch-up: no signal-bar lateness refusal, market volume read ON the ON bar, live price within the dislocation of the ON close
    (FRENZY_CATCHUP_MOVED), stamped frenzy_catchup=True, counted FRENZY_CATCHUP_OPEN. The normal path still refuses a late bar."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, config as C
    import services.trading_engine as TE
    th = C.trading_config.thresholds
    now_bar = int(time.time() // 300) * 300_000; on_sig = now_bar - 15 * 60_000                     # ON bar closed 15 min ago

    async def run(live, catchup=True, gmax=1.0, dmax=1.0, wide=False):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        old = (th.frenzy_gvol_max, th.frenzy_max_entry_dislocation_pct, th.frenzy_wide_hold_green_streak, th.frenzy_wide_above_share_min)
        try:
            th.frenzy_gvol_max = gmax; th.frenzy_max_entry_dislocation_pct = dmax; th.frenzy_wide_hold_green_streak = 0; th.frenzy_wide_above_share_min = 0
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                blocks = []; opened = []; asked = []
                e = _engine(TE, blocks)

                async def gv_now(sig, wait=True):
                    asked.append(("now", sig)); return 0.5

                async def gv_cu(sig, wait=True):
                    asked.append(("cu", sig)); return 0.6
                e._frenzy_gvol_value = gv_now; e._frenzy_gvol_catchup_value = gv_cu

                async def fake_open(**kw):
                    opened.append(kw); return object()
                e.open_position = fake_open
                flag = dict(pair="NMRUSDT", spike_ts=on_sig - 7_200_000, hours=2.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=343.0, run_pct=20.0,
                            atr_pct=2.1, volume_24h=5e7, price=1.0, live_price=live, above_streak=25, bar_ret_pct=-0.1, code="FRENZY_READY")
                await e._frenzy_open(db, flag, {}, on_sig, wide=wide, catchup_now=(now_bar if catchup else None))
        finally:
            th.frenzy_gvol_max, th.frenzy_max_entry_dislocation_pct, th.frenzy_wide_hold_green_streak, th.frenzy_wide_above_share_min = old
            await eng.dispose()
        return blocks, opened, asked, flag.get("last_fire") or ""
    b, o, a, lf = asyncio.run(run(1.005))
    assert len(o) == 1 and o[0]["frenzy_catchup"] is True and o[0]["entry_frenzy_gvol"] == 0.6 and a == [("cu", on_sig - 300_000)]
    assert o[0]["entry_frenzy_catchup_bars"] == 3 and o[0]["entry_frenzy_catchup_move_pct"] == 0.5    # (235) ON close 15 min before this pass's bar · +0.5 %
    assert abs(o[0]["frenzy_bar_open_ms"] - time.time() * 1000) < 60_000                             # the lane wait counts from the decision
    assert "FRENZY_LATE" not in b and "FRENZY_CATCHUP_OPEN" in b and lf.endswith("opened")
    b, o, _, lf = asyncio.run(run(1.02))
    assert o == [] and b == ["FRENZY_CATCHUP_MOVED"] and "moved +2.00% from the ON bar's close" in lf
    b, o, _, _ = asyncio.run(run(1.0, dmax=0))                                                          # guard off → catch-up refused
    assert o == [] and b == ["FRENZY_CATCHUP_MOVED"]
    b, o, _, _ = asyncio.run(run(1.02, wide=True))
    assert o == [] and b == ["FRENZY_WIDE_CATCHUP_MOVED"]
    b, o, a, _ = asyncio.run(run(1.0, catchup=False))                                                   # the normal path: still late
    assert o == [] and "FRENZY_LATE" in b and a == [("now", on_sig - 300_000)]
    b, o, _, _ = asyncio.run(run(0.995))
    assert o[0]["entry_frenzy_catchup_move_pct"] == -0.5                                               # signed


def test_last_judged_bar_survives_a_restart(monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models
    import services.trading_engine as TE

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        async with SL() as s:
            s.add(models.BotState(is_running=False)); await s.commit()
        monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
        e1 = object.__new__(TE.TradingEngine)
        cold = await e1._frenzy_judged_get()                                                   # fresh DB: unknown → no catch-up
        await e1._frenzy_judged_set(T(22)); await e1._frenzy_judged_set(T(21))               # monotonic
        e2 = object.__new__(TE.TradingEngine)                                                  # a restarted process
        got = await e2._frenzy_judged_get()
        await eng.dispose()
        return cold, got
    assert asyncio.run(go()) == (None, T(22))


# ── wiring / every surface (D11, D12) ───────────────────────────────────────────────────────────────────────────────────────────────

def test_wiring_and_surfaces():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("async def _update_frenzy_pass("); body = eng[i:eng.index("async def _frenzy_persist_flags(", i)]
    assert "catchup_prev=await self._frenzy_judged_get()" in body                          # captured once per bar (retry passes reuse it)
    _elif = "elif (_on or _wide) and _cu_max > 0 and _pp is not None and ep.get('in_state'):"
    assert _elif in body and "await self._frenzy_catchup(db, pair, ep, closed, _nc[1], flag, by[pair], bar_open, _pp, _cu_max, _on, _wide," in body
    assert body.index("if ep.get('fresh_on'):") < body.index("await self._frenzy_mark_on_done(pair, ep.get('on_bar_ts'))") < body.index(_elif)
    assert body.index("_pp = self._frenzy_pair_prev(pair, _cu_prev)") < body.index("self.__dict__.setdefault('_fz_pair_unjudged', {}).pop(pair, None)")
    assert body.index("self._frenzy_note_unread(names, seen, bar_open - 300_000, _cu_prev)") < body.index("await self._frenzy_judged_set(bar_open - 300_000)")
    assert "self._frenzy_gvol_catchup_start(" not in body                                       # fix 8: lazy, inside _frenzy_catchup only
    # stored only after the pair loop completed (an exception before it leaves the bar unjudged)
    assert body.index("for _fut in asyncio.as_completed(_fz_tasks):") < body.index("await self._frenzy_judged_set(bar_open - 300_000)") < body.index("except Exception as e:\n            logger.error(f\"[FRENZY] update failed")
    assert "entry_frenzy_catchup=(bool(frenzy_catchup) if _frenzy else None)" in eng
    assert "entry_frenzy_catchup_bars=(entry_frenzy_catchup_bars if (_frenzy and frenzy_catchup) else None)" in eng
    assert "entry_frenzy_catchup_move_pct=(entry_frenzy_catchup_move_pct if (_frenzy and frenzy_catchup) else None)" in eng
    assert "· bars {_bars_n} · move {_mv_txt} vs the ON close" in eng                         # both values in the [FRENZY_CATCHUP] line
    import models as M, config as C
    assert "entry_frenzy_catchup" in {c.name for c in M.Order.__table__.columns}
    assert "frenzy_last_judged_bar_ms" in {c.name for c in M.BotState.__table__.columns}
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert {"entry_frenzy_catchup_bars", "entry_frenzy_catchup_move_pct"} <= {c.name for c in M.Order.__table__.columns}
    assert "('entry_frenzy_catchup_bars', 'INTEGER'), ('entry_frenzy_catchup_move_pct', 'FLOAT')" in db
    assert "('entry_frenzy_catchup', 'BOOLEAN')" in db and "ADD COLUMN frenzy_last_judged_bar_ms BIGINT" in db
    assert C.SignalThresholds.model_fields["frenzy_catchup_max_bars"].default == 0
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_catchup_max_bars"] == 6
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert ui.count('id="config-fz-catchup"') == 1 and "['config-fz-catchup', 'frenzy_catchup_max_bars', 6]" in ui
    assert "_key === 'frenzy_catchup_max_bars' ||" in ui                                    # integer: rounded on save
    i = ui.index("function _buildConfigLines("); assert "frenzy_catchup_max_bars" in ui[i:ui.index("lines.push(..._buildConfigLines(")]
    for name in ("FRENZY_CATCHUP_STALE", "_CATCHUP_MOVED", "_CATCHUP_OPEN", "[FRENZY_CATCHUP]"):
        assert name in eng, name
    assert '"entry_frenzy_catchup"' in open(os.path.join(ROOT, "tests", "test_manual_entry_stamps.py"), encoding="utf-8").read()


# ── ⏪ ON time in the live panel / exports (operator Oct-6: "ON HH:MM · hace …", exports "ON desde MM-DD HH:MM", Argentina h23) ────

def _js_src():
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    i = ui.index("const FRENZY_TZ"); return ui, ui[i:ui.index("function frenzyMark(f)", i)]


def test_on_time_reaches_the_flag_payload():
    import main
    ep = frenzy_walk(_bars(after=27), NORM, TH)
    v = main._frenzy_flag_view(dict(ep, pair="NMRUSDT", code="FRENZY_ON", text="ON (entry bar passed)"))
    assert v["on_ms"] == T(24) + BAR_MS and v["code"] == "FRENZY_ON"                       # the ON bar's CLOSE
    assert main._frenzy_flag_view(dict(frenzy_walk(_bars(after=23), NORM, TH), pair="X"))["on_ms"] is None   # not ON → no time


def test_on_time_text_live_and_exports():
    import shutil, subprocess
    ui, fn = _js_src()
    assert "timeZone: 'America/Argentina/Buenos_Aires', hourCycle: 'h23'" in fn             # the app's convention, never the browser default
    for bad in ("getTimezoneOffset", "getUTCHours", "toLocaleTimeString([]", "(entry bar passed)"):
        assert bad not in fn, bad
    assert "${_fzE(frenzyStatusText(f))}" in ui and "${_escAttr(frenzyStatusText(_fz))}" in ui   # table cell + pairs-table Block Reason
    i = ui.index("function frenzyReportLines("); rep = ui[i:ui.index("function surgeTime(", i)]
    assert "${frenzyReportStatus(f)}" in rep and "frenzyStatusText" not in rep and "hace" not in rep   # exports: "ON desde", no stale "hace"
    assert ui.count("lines.push(...frenzyReportLines(perf, hr2));") == 2                     # both text exports
    node = shutil.which("node")
    if not node:
        return
    on = 1791294000000   # 2026-10-06 13:40 UTC = 10:40 Buenos Aires
    js = fn + f"""
    const on = {on};
    console.log(JSON.stringify([frenzyStatusText({{code:'FRENZY_ON', on_ms:on}}, on + 25*60000), frenzyStatusText({{code:'FRENZY_ON', on_ms:on}}, on + 135*60000),
      frenzyStatusText({{code:'FRENZY_ON', on_ms:on}}, on + 27*3600000), frenzyStatusText({{code:'FRENZY_ON', text:'ON (entry bar passed)'}}),
      frenzyStatusText({{code:'FRENZY_READY', text:'READY', on_ms:on}}), frenzyReportStatus({{code:'FRENZY_ON', on_ms:on}}),
      frenzyReportStatus({{code:'FRENZY_ON'}}), frenzyReportStatus({{code:'FRENZY_VOL_FADED', text:'faded'}}),
      frenzyStatusText({{code:'FRENZY_ON', on_ms: {on} - 13*3600000 - 40*60000 + 3*3600000 + 5*60000}}, on)]));"""
    for tz in ("UTC", "Asia/Tokyo", "America/Sao_Paulo"):                                   # the viewer's zone never changes the text
        out = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, env={**os.environ, "TZ": tz}, check=True).stdout)
        assert out[:8] == ["ON 10:40 · hace 25 min", "ON 10:40 · hace 2 h 15 min", "ON 10:40 · hace 1 d 3 h", "ON", "READY",
                           "ON desde 10-06 10:40", "ON", "faded"], (tz, out)
        assert out[8].startswith("ON 00:05 · hace 10 h 35 min"), out[8]                          # h23: never "24:05"


# ── review fixes ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_unreadable_pair_keeps_its_own_horizon():
    """Fix 1: a completed pass with the pair unreadable (n_bad > 0) advances the global bar, but the pair keeps its last judged bar → its
    fresh bar is still caught up on the next bar it is read."""
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    e._fz_pair_last_ok = {"NMRUSDT": T(23), "AUSDT": T(23)}
    e._frenzy_note_unread(["NMRUSDT", "AUSDT"], {"AUSDT"}, T(24), T(23))                 # bar 24 (NMR's fresh bar): NMR unread
    e._frenzy_note_unread(["NMRUSDT", "AUSDT"], {"AUSDT"}, T(25), T(24))                 # still unread on 25: the FIRST failure wins
    assert e._fz_pair_unjudged == {"NMRUSDT": T(23)}
    ep = frenzy_walk(_bars(after=26), NORM, TH)                                           # read again on bar 26: global horizon = 25
    assert frenzy_catchup_check(ep, T(25), 6) == (None, None)                             # the global bar alone would lose it …
    assert frenzy_catchup_check(ep, e._frenzy_pair_prev("NMRUSDT", T(25)), 6) == (FRENZY_CATCHUP_OK, 2)   # … the pair's own horizon recovers it
    assert e._frenzy_pair_prev("AUSDT", T(25)) == T(25) and e._frenzy_pair_prev("X", None) is None
    e._frenzy_note_unread([], set(), T(23) + 25 * 3600_000, None)                         # > 24 h old → dropped
    assert e._fz_pair_unjudged == {}


def _mem_db(monkeypatch, with_row=True):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models
    import services.trading_engine as TE

    async def mk():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        if with_row:
            async with SL() as s:
                s.add(models.BotState(is_running=False)); await s.commit()
        return eng, SL
    return mk


def test_restart_never_rejudges_an_on_bar_judged_before_a_cancelled_pass(monkeypatch):
    """M-2: the pass judging NMR's fresh bar 24 is cancelled before it stores the bar (global stays 23). The ON bar was recorded the moment
    it was judged, so the restarted process does not re-judge it as a catch-up (which would skip the lateness check)."""
    import services.trading_engine as TE
    mk = _mem_db(monkeypatch)
    th = TH(); monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th))

    async def go():
        eng, SL = await mk(); monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
        e1 = object.__new__(TE.TradingEngine)
        await e1._frenzy_judged_set(T(23))                                                  # last completed pass: bar 23
        await e1._frenzy_mark_on_done("NMRUSDT", T(24))                                     # bar 24 judged normally, then the pass is cancelled
        e2 = object.__new__(TE.TradingEngine); blocks = []
        e2._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
        opened = []

        async def fake_open(*a, **k):
            opened.append(1)
        e2._frenzy_open = fake_open
        prev = await e2._frenzy_judged_get()
        closed = _bars(after=27); ep = frenzy_walk(closed, NORM, th)
        await e2._frenzy_catchup(None, "NMRUSDT", ep, closed, NORM, dict(ep), dict(price=closed[-1][4]), T(28), prev, 6, True, False)
        e3 = object.__new__(TE.TradingEngine); e3._record_filter_block = e2._record_filter_block; e3._frenzy_open = fake_open
        await e3._frenzy_judged_get(); e3._fz_on_done = {}                                  # control: without the record it WOULD catch up
        await e3._frenzy_catchup(None, "NMRUSDT", ep, closed, NORM, dict(ep), dict(price=closed[-1][4]), T(28), prev, 6, True, False)
        await eng.dispose()
        return prev, opened
    prev, opened = asyncio.run(go())
    assert prev == T(23) and opened == [1]                                                   # only the control opened


def test_zero_row_write_is_not_marked_saved(monkeypatch):
    """Fix 6: no BotState row → 0 rows updated → warned, not marked saved (retried next pass)."""
    import services.trading_engine as TE
    mk = _mem_db(monkeypatch, with_row=False)

    async def go():
        eng, SL = await mk(); monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
        e = object.__new__(TE.TradingEngine)
        await e._frenzy_judged_set(T(22))
        r = getattr(e, "_fz_judged_saved", None)
        await eng.dispose()
        return r
    assert asyncio.run(go()) is None


def test_gvol_catchup_read_is_lazy_and_retried_when_empty():
    """Fix 8: started by the first candidate only; a read that came back None is started again; a good one is reused."""
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine); calls = []

    async def fake_bars(allp, limit):
        calls.append(limit); return (None, None) if len(calls) == 1 else ({}, 50)
    e._market_gvol_bars = fake_bars

    async def go():
        e._frenzy_gvol_catchup_start([{}], T(27), 6); await asyncio.sleep(0.01)
        assert await e._frenzy_gvol_catchup_value(T(24)) is None                            # first read failed
        e._frenzy_gvol_catchup_start([{}], T(27), 6); await asyncio.sleep(0.01)            # → read again
        e._frenzy_gvol_catchup_start([{}], T(27), 6); await asyncio.sleep(0.01)            # → reused (it returned a dict)
        return calls
    assert asyncio.run(go()) == [68, 68]


def test_order_book_rechecked_against_the_on_close_and_other_review_fixes():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("async def open_position("); op = eng[i:eng.index("binance_order_id = None", i)]
    # fix 2: the book price at the order vs the ON close (no 1 % + 1 % stacking) — fail-closed predicate, its own counter
    assert "if _frenzy and frenzy_catchup and frenzy_catchup_ref_price and frenzy_catchup_moved(frenzy_catchup_ref_price, _sg_live_px or current_price, _sg_max or 0):" in op
    assert op.index("frenzy_catchup_moved(frenzy_catchup_ref_price") > op.index("_sg_live_px = float(_sg_ob['best_ask']")
    assert "frenzy_catchup_ref_price=(_close if _cu else None)," in eng
    # fix 4: its own lane-wait wording
    assert 'f"catch-up order reached {_fz_late2:.0f}s after the decision (max {FRENZY_ENTRY_MAX_LATE_S}s)" if frenzy_catchup' in op
    j = eng.index("async def _frenzy_open("); fo = eng[j:eng.index("async def _maybe_open_surge(", j)]
    # fix 3: the old _DISLOC check skipped for catch-ups · fix 5: the already-taken return says so
    assert "if not _cu and _dmax > 0 and _close > 0 and abs(price / _close - 1) * 100 > _dmax:" in fo
    assert "already entered — a {_es} on this pair opened at / after this bar" in fo
    # fix 10: FRENZY fully off counts as judged (and clears the per-pair map)
    main_src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    k = main_src.index("async def frenzy_loop():"); loop = main_src[k:main_src.index("async def orderbook_loop():", k)]
    assert "await trading_engine._frenzy_judged_set(int(time.time() // 300) * 300_000 - 300_000, clear_pairs=True)" in loop
    assert loop.index("clear_pairs=True") < loop.index("if trading_engine.is_running:")
    # 13: the catch-up cohort row rides frenzy_rows → UI table + both exports (frenzyReportLines ×2)
    assert '"row": "⏪ recuperados (catch-up)' in main_src
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "data.frenzy_rows.map(r =>" in ui and "for (const r of perf.frenzy_rows)" in ui and ui.count("lines.push(...frenzyReportLines(perf, hr2));") == 2
    # I-2: no "10 fills" wording; the pre-registered gate
    cfg = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "review at 20 catch-up" in cfg and "10 catch-up" not in cfg and "+2 %" not in cfg[cfg.index("FRENZY CATCH-UP"):cfg.index("frenzy_catchup_max_bars: int")]
