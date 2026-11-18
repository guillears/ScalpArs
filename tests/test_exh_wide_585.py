"""🌐 EXHAUSTION WIDE universe (DECISION_LOG 585 — operator-approved PAPER run, declared override).

The WIDE detector (services.frenzy.frenzy_walk at exhaustion_wide_spike_vol_mult, services.exhaustion.wide_walk_th) checked against the
EXPAND_EDGES study's 15×-only spikes (tests/fixtures/exh_wide_parity.npz, built by scripts/exh_wide_parity_fixture.py), the REAL engine
tracking / judging paths on SQLite (exchange reads stubbed), the universe stamp on the open path, config / UI / exports, the scout gate."""
import asyncio
import datetime as dt
import json
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services import exhaustion as X  # noqa: E402
from services.frenzy import frenzy_walk, frenzy_flagged  # noqa: E402
from tests.test_exhaustion_short import _th as _exth, _row as _exrow, _engine  # noqa: E402
from tests.test_frenzy_willy import _hold_db  # noqa: E402

FIX = np.load(os.path.join(ROOT, "tests", "fixtures", "exh_wide_parity.npz"))
BAR, M15, HOUR = 300_000, 900_000, 3_600_000


def _th(**kw):
    d = dict(vars(_exth()), exhaustion_wide_enabled=True, exhaustion_wide_spike_vol_mult=15.0, exhaustion_wide_min_vol_usd=5e6,
             frenzy_spike_ret_pct=5.0, frenzy_spike_vol_mult=20.0, frenzy_spike_min_hour_usd=2e6, frenzy_state_vol_mult=100.0, frenzy_min_hours=2.0,
             frenzy_max_hours=96.0)
    d.update(kw)
    return NS(**d)


def _wrow(k):
    m = FIX[f"r{k}_meta"]
    return str(FIX[f"r{k}_pair"]), [[int(r[0])] + [float(x) for x in r[1:]] for r in FIX[f"r{k}_k5"]], int(m[1]), float(m[2]), float(m[3])


# ── ① parity: the WIDE walk finds the study's 15×-only spikes ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [0, 1, 2, 3, 5])
def test_wide_walk_finds_the_study_spike(k):
    pair, rows, spike_ms, norm, volx = _wrow(k)
    e15 = frenzy_walk(rows, norm, X.wide_walk_th(_th()))
    assert e15 is not None and e15['spike_ts'] == spike_ms and frenzy_flagged(e15, _th())
    e20 = frenzy_walk(rows, norm, _th())
    assert e20 is None or e20['spike_ts'] != spike_ms                                       # the 20× FRENZY flag does not hold this spike


def test_wide_walk_chains_like_the_frenzy_walk():
    """row 4 (GRASS): the walk's episode chaining (24 h without a state bar) anchors an EARLIER 15× spike than the study's 72 h-chained
    bar-rule list — the same realisable-vs-bar-rule gap EXHAUSTION already carries (EXPAND_EDGES §method: 3.6 % more bar-rule trades)."""
    pair, rows, spike_ms, norm, volx = _wrow(4)
    e15 = frenzy_walk(rows, norm, X.wide_walk_th(_th()))
    assert e15 is not None and e15['spike_ts'] < spike_ms and frenzy_flagged(e15, _th())


def test_wide_helpers():
    th = _th()
    w = X.wide_walk_th(th)
    assert w.frenzy_spike_vol_mult == 15.0 and w.frenzy_spike_ret_pct == 5.0 and th.frenzy_spike_vol_mult == 20.0
    assert X.universe_min_vol("WIDE", th) == 5e6 and X.universe_min_vol("FRENZY", th) == 20e6 and X.universe_min_vol(None, th) == 20e6
    assert X.pick_episode([(100, "WIDE"), (50, "FRENZY"), (200, "WIDE")]) == (50, "FRENZY")       # FRENZY always wins
    assert X.pick_episode([(100, "WIDE"), (200, "WIDE")]) == (200, "WIDE") and X.pick_episode([]) == (None, None)
    allp = [dict(pair="AUSDT", change_24h=20, range_24h=25, volume_24h=6e6), dict(pair="BUSDT", change_24h=1, range_24h=30, volume_24h=8e6),
            dict(pair="CUSDT", change_24h=30, range_24h=30, volume_24h=4e6), dict(pair="DUSDT", change_24h=50, range_24h=50, volume_24h=9e7),
            dict(pair="EUSDT", change_24h=2, range_24h=3, volume_24h=9e7), dict(pair="币USDT", change_24h=50, range_24h=50, volume_24h=9e7)]
    took, nq = X.wide_shortlist(allp, {"ZUSDT"}, 15.0, 5e6, exclude={"DUSDT"}, cap=30)
    assert took == ["BUSDT", "AUSDT"] and nq == 2                                                # move-ranked; excluded / thin / small / non-ASCII out
    assert X.wide_shortlist(allp, {"BUSDT"}, 15.0, 5e6, cap=1) == (["DUSDT"], 2)
    assert X.wide_enabled(th) and not X.wide_enabled(_th(exhaustion_wide_enabled="yes"))
    f = dict(ok=True, cross=True, vol24=8e6, has_spot=True, spot_share_chg=-0.02, vol_decay=0.8, rsi_prev=72, rsi=69, atr_pct=1.0)
    assert X.decide(f, th)[1] == X.B_VOL24 and X.decide(f, th, min_vol=5e6)[0] is True


# ── ② the REAL engine path: tracking WIDE, FRENZY wins, the WIDE floor at the judgement ──────────────────────────────────────────────

def _patch(monkeypatch, TE, **kw):
    for k, v in dict(vars(_th()), **kw).items():
        monkeypatch.setattr(TE.config.trading_config.thresholds, k, v, raising=False)


def test_engine_tracks_a_wide_episode_on_a_15m_close(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    pair, rows, spike_ms, norm, volx = _wrow(1)                                               # LUNA2: no 20× flag at all
    T = (rows[-1][0] + BAR) // M15 * M15
    closed = [r for r in rows if r[0] + BAR <= T]
    _patch(monkeypatch, TE)
    monkeypatch.setitem(TE._frenzy_norm_cache, pair, (0, norm, 10 ** 12))

    async def go(wide_on):
        eng, SL = await _hold_db(monkeypatch)()
        monkeypatch.setattr(TE.config.trading_config.thresholds, "exhaustion_wide_enabled", wide_on, raising=False)
        e = _engine(TE)
        async with SL() as db:
            await e._exhaustion_eval(db, pair, None, closed, {}, T - BAR)                    # a 5m close: the WIDE detector waits for a 15m close
            n5 = len(e._exh_state())
            await e._exhaustion_eval(db, pair, None, closed, {}, T)
        async with SL() as s:
            eps = (await s.execute(select(models.ExhaustionEpisode))).scalars().all()
        await eng.dispose()
        return n5, eps, dict(e._exh_state())
    n5, eps, st = asyncio.run(go(True))
    assert n5 == 0 and len(eps) == 1 and eps[0].universe == "WIDE" and st[(pair, spike_ms)]['universe'] == "WIDE"
    n5, eps, st = asyncio.run(go(False))
    assert eps == [] and st == {}                                                              # WIDE off → nothing tracked


def test_frenzy_flag_upgrades_a_wide_row_and_wins_the_signal(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    _patch(monkeypatch, TE)
    sp = 1_763_000_100_000 // BAR * BAR

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        e = _engine(TE)
        async with SL() as db:
            await e._exhaustion_ready(db)
            await e._exhaustion_track("UPGUSDT", sp, sp, "WIDE")
            await e._exhaustion_track("UPGUSDT", sp, sp + BAR, "FRENZY")
        async with SL() as s:
            rows = (await s.execute(select(models.ExhaustionEpisode))).scalars().all()
        await eng.dispose()
        return e, rows
    e, rows = asyncio.run(go())
    assert len(rows) == 1 and rows[0].universe == "FRENZY" and e._exh_state()[("UPGUSDT", sp)]['universe'] == "FRENZY"
    e2 = _engine(TE)
    e2._exh_cand_q = []
    e2._exhaustion_eval_signal_args = None
    sig = {"cross": True, "rsi_prev": 72.0, "rsi": 69.0, "close": 1.0}
    monkeypatch.setattr(TE._exh_rules, "rsi_cross", lambda closed, T, th: sig)
    asyncio.run(e2._exhaustion_eval_signal("UPGUSDT", [(sp, "WIDE"), (sp - HOUR, "FRENZY")], [], {}, sp + 2 * HOUR, _th()))
    assert e2._exh_cand_q[-1][1] == sp - HOUR and e2._exh_cand_q[-1][6] == "FRENZY"


def _judge(monkeypatch, universe, th_kw=None):
    """drives the REAL _exhaustion_judge on the EXHAUSTION parity row 1 for an episode of `universe` (reads + open stubbed)."""
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    pair, T, on, sp, rows, spot, m, _ = _exrow(1)
    _patch(monkeypatch, TE, **(th_kw or {}))

    async def kl(pid, interval, limit, end_ms=None, spot=False):
        if spot:
            return [[r[0], 0, 0, 0, 0, 0, r[0] + M15 - 1, r[1]] for r in _exrow(1)[5]]
        return [r for r in rows if r[0] <= end_ms][-limit:]

    async def syms():
        return {pair}

    async def fr(sym, max_age_s=None):
        return 0.0001

    async def oi(pid, end_ms, limit=12):
        return None
    for k, f in (("fetch_exh_klines", kl), ("fetch_exh_spot_symbols", syms), ("fetch_funding_rate", fr), ("fetch_exh_open_interest", oi)):
        monkeypatch.setattr(TE.binance_service, k, f)
    monkeypatch.setattr(TE.binance_service, "exh_used_weight", 0, raising=False)
    opens = []

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        e = _engine(TE)
        e._exh_loaded = True; e._exh_mode = True
        e._exh_eps = {(pair, sp): dict(id=None, status="TRACKING", first_seen_ms=sp, universe=universe)}

        async def fake_open(db=None, **kw):
            opens.append(kw)
            return NS(id=55)
        e.open_position = fake_open
        closed = [r[:6] for r in rows if r[0] + BAR <= T]
        sig = X.rsi_cross(closed, T, _th())
        async with SL() as db:
            await e._exhaustion_judge(db, pair, sp, closed, {}, T, sig, universe=universe)
        async with SL() as s:
            j = (await s.execute(select(models.ExhaustionSignal))).scalars().all()
        await eng.dispose()
        return j
    return opens, asyncio.run(go()), m


def test_judgement_uses_the_universe_floor_and_stamps_it(monkeypatch):
    vol24 = float(_exrow(1)[6][9])
    floor = vol24 * 1.5                                                                        # above this row's 24 h volume
    opens, j, m = _judge(monkeypatch, "WIDE", th_kw=dict(exhaustion_min_vol_usd=floor, exhaustion_wide_min_vol_usd=vol24 * 0.5))
    assert len(opens) == 1 and opens[0]['entry_exh_universe'] == "WIDE" and j[0].universe == "WIDE" and j[0].status == "OPENED"
    mp = pytest.MonkeyPatch()
    try:
        opens, j, m = _judge(mp, "FRENZY", th_kw=dict(exhaustion_min_vol_usd=floor, exhaustion_wide_min_vol_usd=vol24 * 0.5))
    finally:
        mp.undo()
    assert opens == [] and j[0].block == "EXH_VOL24_LOW" and j[0].universe == "FRENZY"           # a FRENZY episode keeps its $20M-class floor


def test_open_position_stamps_the_universe(monkeypatch):
    from tests.test_flatbase_short import _e2e_open
    now_ms = int(__import__("time").time() * 1000)
    r, *_ = _e2e_open(monkeypatch, n_open=0, th_kw=dict(vars(_th())), exhaustion_short=True, exhaustion_bar_close_ms=now_ms,
                      entry_exh_tp_pct=3.0, entry_exh_sl_pct=1.5, entry_exh_universe="WIDE")
    assert r is not None and r.entry_exh_universe == "WIDE"
    mp = pytest.MonkeyPatch()
    try:
        r, *_ = _e2e_open(mp, n_open=0, th_kw=dict(vars(_th())), exhaustion_short=True, exhaustion_bar_close_ms=now_ms,
                          entry_exh_tp_pct=3.0, entry_exh_sl_pct=1.5, entry_exh_universe="junk")
    finally:
        mp.undo()
    assert r.entry_exh_universe == "FRENZY"                                                    # anything else → FRENZY


def test_pass_wiring():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    p = eng.index("    async def _update_frenzy_pass(self, db):"); body = eng[p:eng.index("\n    def _frenzy_wide_off_note(", p)]
    assert "if _exh and _exh_rules.wide_enabled(th) and bar_open % 900_000 == 0:" in body and "exclude=set(names), cap=EXH_WIDE_READS_MAX" in body
    assert "_exh_only |= set(_wd)" in body and "self._record_filter_block(_exh_rules.B_WIDE_READ_CAP, \"SHORT\")" in body
    import services.trading_engine as TE
    assert TE.EXH_WIDE_READS_MAX == 30 and TE.TradingEngine.stamp_class("entry_exh_universe") == "sleeve"


# ── ③ config / UI / exports / scout ────────────────────────────────────────────────────────────────────────────────────────────────

def test_config_json_ui_and_exports():
    import config as C
    mf = C.SignalThresholds.model_fields
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for k, (d, j) in {"exhaustion_wide_enabled": (False, True), "exhaustion_wide_spike_vol_mult": (15.0, 15.0),
                      "exhaustion_wide_min_vol_usd": (5e6, 5e6)}.items():
        assert mf[k].default == d and cfg[k] == j, k
    assert "DECISION_LOG 585" in open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for i in ("config-exh-wide-enabled", "config-exh-wide-vol-mult", "config-exh-wide-min-vol"):
        assert html.count(f'id="{i}"') == 1, i
    assert ", 'exhaustion_wide_spike_vol_mult', " in html and ", 'exhaustion_wide_min_vol_usd', " in html          # EXHAUSTION_NUM_FIELDS (load + save)
    assert "config.thresholds.exhaustion_wide_enabled" in html and "exhaustion_wide_enabled: document.getElementById('config-exh-wide-enabled')" in html
    assert "🌐 EXHAUSTION WIDE universe (DECISION_LOG 585" in html and "m.exhaustion_wide_enabled" in html   # both exports (config lines + monitor line)
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '"exh_universe"' in main and "🌐 universe WIDE (15× / $5M, DL 585)" in main and "WIDE gate (DL 585)" in main
    assert X.exhaustion_config_errors(NS(**dict(C.trading_config.thresholds.model_dump(), exhaustion_wide_enabled="on")), 20.0)


def test_scout_gate_and_split(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_exh_wide_585 as S
    t0 = pd.Timestamp("2026-11-20")
    base = dict(direction="SHORT", status="CLOSED", entry_strategy="EXHAUSTION_SHORT", pnl=1.0, notional_value=1000.0, entry_exh_model_slip_pct=0.05,
                entry_exh_q5_usd=1e6, entry_depth_mode=None, entry_depth_walk_pct=None)
    rows = [dict(base, opened_at=str(t0 + pd.Timedelta(hours=9 * i)), pair=f"W{i}USDT", pnl_percentage=1.5, close_reason="FRENZY_TP",
                 entry_exh_universe="WIDE") for i in range(31)]
    rows += [dict(base, opened_at=str(t0 + pd.Timedelta(hours=i)), pair=f"F{i}USDT", pnl_percentage=-0.5, close_reason="STOP_LOSS",
                  entry_exh_universe=None) for i in range(5)]
    fr = pd.DataFrame(rows); fr["o"] = pd.to_datetime(fr.opened_at)
    st = str(tmp_path / "w.json")
    txt = S.report(frame=fr, since_ms=int(t0.value // 10**6), state_path=st)
    assert "FROZEN" in txt and "KEEP" in txt and "WIDE  : N 31" in txt and "FRENZY: N 5" in txt
    assert json.load(open(st))["verdict"] == "KEEP"
    assert S.net_of(fr.itertuples().__next__()) == pytest.approx(1.5 - 0.05)                     # maker TP: only the entry leg pays
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import observe_registry as R
    assert "scout_exh_wide_585.py" in R.BOOKED
    assert "import scout_exh_wide_585 as _xw585" in open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "reports/SCOUT_EXH_WIDE_585.json" in open(os.path.join(ROOT, ".gitignore")).read()


# ── ④ review fixes (DECISION_LOG 585 review I2 / I3) ──────────────────────────────────────────────────────────────────────────────────

def _fills_frame(n_fz, n_wide):
    t0 = pd.Timestamp("2026-11-20")
    base = dict(closed_at=None, direction="SHORT", status="CLOSED", entry_strategy="EXHAUSTION_SHORT", pnl=1.0, pnl_percentage=1.0,
                close_reason="FRENZY_TP", notional_value=1000.0, entry_exh_model_slip_pct=0.0, entry_exh_q5_usd=1e6, entry_exh_tp_pct=9.0,
                entry_exh_sl_pct=3.0, entry_slippage_pct=0.0, entry_exh_atr_pct=1.0, entry_depth_mode=None, entry_depth_walk_pct=None,
                entry_price=1.0)
    rows = [dict(base, opened_at=str(t0 + pd.Timedelta(hours=i)), pair=f"F{i}USDT", entry_exh_universe=(None if i % 2 else "FRENZY")) for i in range(n_fz)]
    rows += [dict(base, opened_at=str(t0 + pd.Timedelta(hours=i, minutes=30)), pair=f"W{i}USDT", entry_exh_universe="WIDE") for i in range(n_wide)]
    a = pd.DataFrame(rows); a["o"] = pd.to_datetime(a.opened_at); a["slip"] = 0.0
    return a


def test_pre585_gates_count_frenzy_universe_only():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_exhaustion_short as SC
    import scout_exh_exit_override as XO
    import scout_exh_ssc_oivol as SO
    f = _fills_frame(3, 4)
    assert len(SC.frenzy_universe(f, "entry_exh_universe")) == 3                                   # FRENZY + NULL kept, WIDE out
    txt = SC.report(f, pd.DataFrame(columns=list(SC.DEC_COLS)))
    assert "FRENZY universe only" in txt and "W0USDT" not in txt
    j = pd.DataFrame([dict(e="EXHAUSTION", pair="AUSDT", exh_signal_at="2026-11-21T00:00:00", exh_status="BLOCKED", exh_block="EXH_SSC_OIVOL",
                           exh_universe=u) for u in ("WIDE", None, "FRENZY")])
    assert len(SO.cohort(j, 0)) == 2                                                              # the 582 shadow gate: WIDE signals out
    called = []
    orig = XO.price_fill
    try:
        XO.price_fill = lambda pair, *a, **k: called.append(pair) or "NO_DATA"
        XO.report(f, now_ms=0, state_path=None)
    finally:
        XO.price_fill = orig
    assert called and not any(p.startswith("W") for p in called)                                  # the 579 exit gate never prices a WIDE fill
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert 'or_(Order.entry_exh_universe.is_(None), Order.entry_exh_universe != "WIDE")' in main      # the dashboard 578 gate


def test_read_order_frenzy_first_and_wide_off(monkeypatch):
    import services.trading_engine as TE
    e = _engine(TE)
    e._exh_mode = True; e._exh_loaded = True
    e._exh_eps = {("ZUSDT", 100): dict(status="TRACKING", universe="FRENZY"), ("AUSDT", 300): dict(status="TRACKING", universe="WIDE"),
                  ("BUSDT", 200): dict(status="TRACKING", universe="WIDE"), ("CUSDT", 50): dict(status="TRACKING", universe=None),
                  ("DUSDT", 400): dict(status="ENDED", universe="FRENZY")}
    assert e._exh_read_order(True) == ["ZUSDT", "CUSDT", "AUSDT", "BUSDT"]                         # FRENZY newest first, then WIDE newest first
    assert e._exh_read_order(False) == ["ZUSDT", "CUSDT"]                                           # WIDE-only pairs skipped while WIDE is off


def test_frenzy_episode_ends_the_wide_twin(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    _patch(monkeypatch, TE)
    sp = 1_763_000_100_000 // BAR * BAR

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        e = _engine(TE)
        async with SL() as db:
            await e._exhaustion_ready(db)
            await e._exhaustion_track("TWNUSDT", sp, sp, "WIDE")
            await e._exhaustion_track("TWNUSDT", sp + HOUR, sp + HOUR, "FRENZY")                  # a later 20× spike on the same coin
        async with SL() as s:
            rows = {(r.universe, r.status) for r in (await s.execute(select(models.ExhaustionEpisode))).scalars().all()}
        await eng.dispose()
        return e, rows
    e, rows = asyncio.run(go())
    assert rows == {("WIDE", "ENDED"), ("FRENZY", "TRACKING")} and list(e._exh_state()) == [("TWNUSDT", sp + HOUR)]
