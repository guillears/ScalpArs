"""🎲² WILLY EXIT V2 (DECISION_LOG 356, operator-approved declared override, 2026-10-23): a FRENZY_WILLY / FRENZY_WILLY_HT fill's take profit
is aimed just below the episode's spike high (net at H × (1 − k %), maker exit; ≥ the WILLY TP, ≤ the cap), decided ONCE at entry and stored
(Order.frenzy_willy_tp_target_pct); a runner target (> the arm) gets a profit floor that arms at peak ≥ the arm (stored peak → restart-safe)
and closes at net ≤ the floor (WILLY_FLOOR) — no stop, no late TP once armed; the cap still closes. Target == +1.25 → today's exit exactly.
The stop is unchanged (the BTC-falling stop was NOT shipped). No network."""
import asyncio, json, os, random, sys
from datetime import datetime, timedelta
from types import SimpleNamespace as NS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.frenzy import (frenzy_willy_exit_for, frenzy_willy_high_target, frenzy_willy_floor_armed, frenzy_willy_floor_levels,
                             frenzy_willy_tp_at, frenzy_willy_v2_config_errors, frenzy_willy_row_tp)
from services.resting_tp import rtp_row_target, rtp_target, rtp_reason_for_limit, stop_class
from services.exec_shadow import net_pct_at, resting_tp_price

TK, MK = 0.00045, 0.00018
EF = TK * 100   # entry taker fee, % of notional


def _th(**kw):
    base = dict(frenzy_willy_tp_pct=1.25, frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, frenzy_willy_late_tp_minutes=30.0,
                frenzy_willy_late_tp_pct=0.25, frenzy_willy_high_target_enabled=True, frenzy_willy_high_k_pct=1.0,
                frenzy_willy_high_tp_max_pct=8.0, frenzy_willy_floor_arm_pct=1.25, frenzy_willy_floor_pct=1.0, resting_tp_willy_enabled=True)
    base.update(kw)
    return NS(**base)


def _row(tp=None, peak=0.0, armed_at=None, es="FRENZY_WILLY"):
    return dict(entry_strategy=es, frenzy_willy_tp_target_pct=tp, peak_pnl=peak, willy_floor_armed_at=armed_at)


# ─────────────────────────────────────────────────────────────── ① target math

def test_target_far_near_missing_switch_cap():
    th = _th()
    e = 100.0
    # H far: 5 % above entry → P_H = 103.95 (k 1 %) → the net at that price with the taker entry + maker exit
    t_used, t_raw = frenzy_willy_high_target(th, e, 105.0, EF, MK)
    assert abs(t_raw - net_pct_at("LONG", e, 105.0 * 0.99, EF, MK)) < 1e-3 and abs(t_used - t_raw) < 1e-9 and 3.8 < t_used < 4.0
    # the resting limit for T_used lands on P_H (rounded UP to the tick — never books less than the target)
    lim = resting_tp_price("LONG", e, EF, t_used, MK, 0.01)
    assert abs(lim - 103.95) <= 0.011
    # H near: target below +1.25 → floored at the WILLY TP
    assert frenzy_willy_high_target(th, e, 101.5, EF, MK)[0] == 1.25
    # H below the entry (the red bar broke the high? never — but a bad stamp) → +1.25
    assert frenzy_willy_high_target(th, e, 99.0, EF, MK)[0] == 1.25
    # missing / unreadable H or entry or fees → +1.25, no raw
    for args in ((e, None, EF, MK), (e, float("nan"), EF, MK), (None, 105.0, EF, MK), (e, 105.0, None, MK), (e, "x", EF, MK)):
        assert frenzy_willy_high_target(th, *args) == (1.25, None)
    # switch off → +1.25 always
    assert frenzy_willy_high_target(_th(frenzy_willy_high_target_enabled=False), e, 105.0, EF, MK) == (1.25, None)
    # cap 8
    assert frenzy_willy_high_target(th, e, 130.0, EF, MK)[0] == 8.0
    assert frenzy_willy_high_target(_th(frenzy_willy_high_tp_max_pct=5.0), e, 130.0, EF, MK)[0] == 5.0
    # k moves the price: k 2 % < k 1 %
    assert frenzy_willy_high_target(_th(frenzy_willy_high_k_pct=2.0), e, 105.0, EF, MK)[0] < t_used


# ─────────────────────────────────────────────────────────────── ② floor / exit

def test_unarmed_and_legacy_rows_identical_to_today():
    """pre-V2 row (NULL target), target 1.25 and an un-armed runner below the arm all decide exactly like the pre-356 exit (row=None)."""
    th = _th()
    rnd = random.Random(3)
    for _ in range(4000):
        p = rnd.uniform(-5, 4); pk = max(p, rnd.uniform(-1, 4)); h = rnd.choice([None, rnd.uniform(0, 120)]); rt = rnd.random() < 0.5
        old = frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt)
        assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt, row=_row(None, pk)) == old
        assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt, row=_row(1.25, pk)) == old   # T_used == +1.25: today, incl. a touch at +1.25
        assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt, row={"entry_strategy": "FRENZY_WILLY"}) == old
        if pk < 1.25 and p < 1.25:   # a runner not yet armed: everything as today (stop, late TP, cap) — only the TP level differs
            assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=True, row=_row(3.5, pk)) == frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=True)


def test_target_125_with_peak_at_arm_and_resting_tp_is_today():
    th = _th()
    # a touch of +1.25 that the resting limit did not fill, then a slide: today rides on to the stop — V2 identical (no floor at T == arm)
    for p in (1.0, 0.5, -3.9, -4.0):
        assert frenzy_willy_exit_for(p, 1.3, th, 10, None, resting_tp=True, row=_row(1.25, 1.3)) == frenzy_willy_exit_for(p, 1.3, th, 10, None, resting_tp=True)
    assert not frenzy_willy_floor_armed(th, _row(1.25, 5.0))


def test_runner_tp_and_floor_exit():
    th = _th()
    r = _row(3.5)
    assert frenzy_willy_exit_for(3.6, 3.6, th, 10, row=r) == (True, "FRENZY_TP", 3.5)          # software TP at the row's own target
    assert frenzy_willy_exit_for(3.6, 3.6, th, 10, resting_tp=True, row=r)[0] is False          # resting limit alive → the limit books it
    assert frenzy_willy_exit_for(1.3, 1.3, th, 10, row=r)[0] is False                           # armed at +1.25, holding
    assert frenzy_willy_exit_for(1.0, 1.4, th, 10, row=r) == (True, "WILLY_FLOOR", 1.0)          # floor
    assert frenzy_willy_exit_for(1.01, 1.4, th, 10, row=r)[0] is False
    assert frenzy_willy_exit_for(-4.5, 1.4, th, 10, resting_tp=True, row=r) == (True, "WILLY_FLOOR", 1.0)   # a gap through: floor (market), never "STOP_LOSS"


def test_armed_ignores_late_tp_and_stop_cap_still_closes():
    th = _th()
    r = _row(3.5, peak=1.4)
    assert frenzy_willy_exit_for(1.2, 1.4, th, 45, row=r)[0] is False                           # late TP +0.25 would have fired — armed: no
    assert frenzy_willy_exit_for(1.2, 1.4, th, 90, row=r) == (True, "MAX_HOLD_TIME", 1.2)
    # unarmed runner: the late TP and the stop as today
    assert frenzy_willy_exit_for(0.3, 0.9, th, 45, row=_row(3.5, 0.9)) == (True, "WILLY_LATE_TP", 0.25)
    assert frenzy_willy_exit_for(-4.0, 0.9, th, 20, row=_row(3.5, 0.9)) == (True, "STOP_LOSS", -4.0)
    assert frenzy_willy_tp_at(th, 45, tp_used=3.5, armed=True) == (3.5, "FRENZY_TP")
    assert frenzy_willy_tp_at(th, 45, tp_used=3.5, armed=False) == (0.25, "WILLY_LATE_TP")
    assert frenzy_willy_tp_at(th, 10, tp_used=3.5) == (3.5, "FRENZY_TP")


def test_floor_arm_restart_safe_and_switch():
    th = _th()
    assert not frenzy_willy_floor_armed(th, _row(3.5, peak=1.2))
    assert frenzy_willy_floor_armed(th, _row(3.5, peak=1.2), peak=1.3)                          # the live peak arms it
    assert frenzy_willy_floor_armed(th, _row(3.5, peak=1.3))                                    # restart: the STORED peak
    assert frenzy_willy_floor_armed(th, _row(3.5, peak=0.0, armed_at=datetime(2026, 10, 23)))   # restart: the recorded arm time
    assert frenzy_willy_floor_armed(th, NS(**_row(3.5, peak=1.3)))                              # an ORM-like row
    # switch OFF → new fills store +1.25 (no floor); an OPEN runner keeps its stored target + floor (never re-decided)
    off = _th(frenzy_willy_high_target_enabled=False)
    assert frenzy_willy_high_target(off, 100.0, 105.0, EF, MK)[0] == 1.25
    assert frenzy_willy_floor_armed(off, _row(3.5, peak=1.3)) and frenzy_willy_row_tp(off, _row(3.5)) == 3.5
    # bad floor config never produces a floor ≥ the arm
    assert frenzy_willy_floor_levels(_th(frenzy_willy_floor_pct=2.0)) == (1.25, 0.625)
    assert frenzy_willy_floor_levels(_th(frenzy_willy_floor_pct=-1)) == (1.25, 0.625)
    assert frenzy_willy_floor_levels(_th(frenzy_willy_floor_arm_pct=0)) == (1.25, 1.0)


def test_ht_runs_the_same_exit():
    th = _th()
    for p, pk, h in ((3.6, 3.6, 5), (1.0, 1.4, 10), (1.2, 1.4, 45), (0.3, 0.9, 45), (-4.0, 0.9, 20)):
        assert (frenzy_willy_exit_for(p, pk, th, h, row=_row(3.5, pk, es="FRENZY_WILLY_HT"))
                == frenzy_willy_exit_for(p, pk, th, h, row=_row(3.5, pk)))
    assert rtp_row_target(th, NS(**_row(3.5, es="FRENZY_WILLY_HT"), exit_override_at=None), 10) == (3.5, "FRENZY_TP")


def test_badge_facts():
    th = _th()


# ─────────────────────────────────────────────────────────────── resting TP (stored T_used, armed = no late swap)

def _orm(tp=None, peak=0.0, es="FRENZY_WILLY", armed_at=None):
    return NS(entry_strategy=es, frenzy_willy_tp_target_pct=tp, peak_pnl=peak, willy_floor_armed_at=armed_at, exit_override_at=None,
              direction="LONG", entry_price=100.0)


def test_resting_target_uses_the_stored_target_and_armed_blocks_the_late_swap():
    th = _th()
    assert rtp_row_target(th, _orm(3.5), 0.0) == (3.5, "FRENZY_TP")
    assert rtp_row_target(th, _orm(3.5, peak=0.5), 40) == (0.25, "WILLY_LATE_TP")             # not armed: today's late swap
    assert rtp_row_target(th, _orm(3.5, peak=1.3), 40) == (3.5, "FRENZY_TP")                  # armed (stored peak): stays at T_used
    assert rtp_row_target(th, _orm(3.5, peak=0.5), 40, willy_armed=True) == (3.5, "FRENZY_TP")   # armed in memory (realtime) before the peak is stored
    assert rtp_row_target(th, _orm(None), 0.0) == (1.25, "FRENZY_TP") and rtp_row_target(th, _orm(None), 40) == (0.25, "WILLY_LATE_TP")
    assert rtp_row_target(_th(resting_tp_willy_enabled=False), _orm(3.5), 0.0) is None
    assert rtp_target(th, "FRENZY_WILLY", 40, willy_tp=3.5, willy_armed=True) == (3.5, "FRENZY_TP")
    # the recorded limit's close reason is judged against the row's own TP (a +3.5 limit is a FRENZY_TP, a +0.25 limit is the late one)
    lim35 = resting_tp_price("LONG", 100.0, EF, 3.5, MK, 0.01)
    lim025 = resting_tp_price("LONG", 100.0, EF, 0.25, MK, 0.01)
    assert rtp_reason_for_limit(th, "FRENZY_WILLY", "LONG", 100.0, EF, lim35, MK, willy_tp=3.5) == "FRENZY_TP"
    assert rtp_reason_for_limit(th, "FRENZY_WILLY", "LONG", 100.0, EF, lim025, MK, willy_tp=3.5) == "WILLY_LATE_TP"
    assert stop_class("WILLY_FLOOR")   # a floor exit never waits on a blocked resting-TP cancel


def _harness(monkeypatch, **kw):
    from tests.test_resting_tp import _setup
    return _setup(monkeypatch, **kw)


LIM35 = resting_tp_price("LONG", 100.0, EF, 3.5, MK, 0.01)
LIM125 = resting_tp_price("LONG", 100.0, EF, 1.25, MK, 0.01)


def test_engine_arm_places_the_stored_target_live_and_paper(monkeypatch):
    rows = [dict(pair="RUNUSDT", entry_strategy="FRENZY_WILLY", frenzy_willy_tp_target_pct=3.5),
            dict(pair="HTXUSDT", entry_strategy="FRENZY_WILLY_HT", frenzy_willy_tp_target_pct=3.5),
            dict(pair="OLDUSDT", entry_strategy="FRENZY_WILLY")]                                  # pre-V2 row → config +1.25

    def go(paper):
        T, eng, engine, S, seed, read, log, blocks = _harness(monkeypatch, paper=paper, rows=rows,
                                                              place=lambda p, px: {"id": "O-" + p, "price": px, "amount": 10.0})
        async def run():
            await seed()
            from sqlalchemy import select
            import models
            async with S() as s:
                for o in (await s.execute(select(models.Order).order_by(models.Order.id))).scalars().all():
                    await eng._rtp_arm(s, o)
            r = await read(); await engine.dispose(); return r
        return asyncio.run(run()), log
    r, log = go(False)
    assert [x[-1] for x in log if x[0] == "place"] == [LIM35, LIM35, LIM125]
    assert (r[1].resting_tp_price, r[2].resting_tp_price, r[3].resting_tp_price) == (LIM35, LIM35, LIM125)
    rp, logp = go(True)                                                                          # paper = live
    assert logp == [] and (rp[1].resting_tp_price, rp[2].resting_tp_price, rp[3].resting_tp_price) == (LIM35, LIM35, LIM125)


def test_engine_sweep_late_swap_only_when_not_armed(monkeypatch):
    old = datetime.utcnow() - timedelta(minutes=40)
    rows = [dict(pair="ARMUSDT", entry_strategy="FRENZY_WILLY", frenzy_willy_tp_target_pct=3.5, peak_pnl=1.4, protective_tp_order_id="A1",
                 resting_tp_price=LIM35, resting_tp_filled_qty=0.0, opened_at=old),            # armed: stays at +3.5
            dict(pair="UNAUSDT", entry_strategy="FRENZY_WILLY", frenzy_willy_tp_target_pct=3.5, peak_pnl=0.6, protective_tp_order_id="B1",
                 resting_tp_price=LIM35, resting_tp_filled_qty=0.0, opened_at=old)]            # not armed at 40 min: → late +0.25
    T, eng, engine, S, seed, read, log, blocks = _harness(monkeypatch, rows=rows, place=lambda p, px: {"id": "N-" + p, "price": px, "amount": 10.0},
                                                          fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0}, close_px=100.1)
    async def go():
        await seed(); out = await eng.resting_tp_sweep(now=100.0)
        r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert [x for x in out if x[1] == 1] == [] and r[1].protective_tp_order_id == "A1" and r[1].resting_tp_price == LIM35
    assert [x[0] for x in out if x[1] == 2] == ["reprice"]
    assert not [x for x in log if x[1:2] == ("ARMUSDT",)]


def test_engine_arm_note_records_once(monkeypatch):
    import services.trading_engine as T
    monkeypatch.setattr(T.config.trading_config, "thresholds", _th(), raising=False)
    eng = T.TradingEngine.__new__(T.TradingEngine)
    o = NS(pair="XUSDT", entry_strategy="FRENZY_WILLY", frenzy_willy_tp_target_pct=3.5, peak_pnl=1.3, willy_floor_armed_at=None)
    T._rtp_rt[777] = {}
    assert eng._willy_v2_arm_note(777, o, 1.3, persist_row=o) is True and o.willy_floor_armed_at is not None and T._rtp_rt[777]["warm"] is True
    t0 = o.willy_floor_armed_at
    assert eng._willy_v2_arm_note(777, o, 1.5, persist_row=o) is True and o.willy_floor_armed_at == t0       # recorded ONCE
    d = dict(pair="YUSDT", entry_strategy="FRENZY_WILLY", frenzy_willy_tp_target_pct=3.5, peak_pnl=0.2, willy_floor_armed_at=None)
    assert eng._willy_v2_arm_note(778, d, 1.3) is True and d["willy_floor_armed_at"] is not None              # realtime: the cache dict
    assert eng._willy_v2_arm_note(779, dict(d, frenzy_willy_tp_target_pct=1.25, willy_floor_armed_at=None), 2.0) is False   # T == arm: no floor
    T._rtp_rt.pop(777, None)


def test_backstop_follows_the_unchanged_stop():
    import services.trading_engine as T
    th = NS(**vars(_th()), broker_backstop_pct=2.5)
    assert T.row_backstop_pct(th, dict(_row(3.5), leverage=10)) == T.row_backstop_pct(th, dict(_row(None), leverage=10)) == T.order_backstop_pct(th, "FRENZY_WILLY", leverage=10)
    assert T.order_backstop_pct(th, "FRENZY_WILLY", leverage=10) == 4.5     # stop −4 + 0.5 margin (BTC-falling stop not shipped)


# ─────────────────────────────────────────────────────────────── config / migration / UI / export parity

def _src(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


FIELDS = {"frenzy_willy_high_target_enabled": True, "frenzy_willy_high_k_pct": 1.0, "frenzy_willy_high_tp_max_pct": 8.0,
          "frenzy_willy_floor_arm_pct": 1.25, "frenzy_willy_floor_pct": 1.0}


def test_config_fields_json_validation():
    import config as C
    f = type(C.trading_config.thresholds).model_fields
    j = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for k, v in FIELDS.items():
        assert f[k].default == v and j[k] == v, k
    assert "frenzy_willy_btcfall_stop_pct" not in f and "frenzy_willy_btcfall_stop_pct" not in j      # component B not shipped
    assert frenzy_willy_v2_config_errors(_th()) == []
    bad = frenzy_willy_v2_config_errors(_th(frenzy_willy_floor_pct=1.25, frenzy_willy_high_tp_max_pct=1.0, frenzy_willy_high_k_pct=-1,
                                            frenzy_willy_high_target_enabled="yes", frenzy_willy_floor_arm_pct=0))
    assert len(bad) == 6                                                                            # + arm < WILLY TP (deep review)
    assert "frenzy_willy_v2_config_errors as _wv2_cfg_errors" in _src("main.py") and "_cfg_errs.extend(_wv2_cfg_errors(new_config.thresholds))" in _src("main.py")


def test_ui_inputs_load_save_and_export_line():
    h = _src("templates", "index.html")
    for _id in ("config-fz-willy-high-target", "config-fz-willy-high-k", "config-fz-willy-high-tp-max", "config-fz-willy-floor-arm", "config-fz-willy-floor"):
        assert h.count(f'id="{_id}"') == 1, _id
    for frag in ("['config-fz-willy-high-k', 'frenzy_willy_high_k_pct', 1.0]", "['config-fz-willy-high-tp-max', 'frenzy_willy_high_tp_max_pct', 8.0]",
                 "['config-fz-willy-floor-arm', 'frenzy_willy_floor_arm_pct', 1.25]", "['config-fz-willy-floor', 'frenzy_willy_floor_pct', 1.0]",
                 "_e.checked = config.thresholds.frenzy_willy_high_target_enabled !== false",
                 "frenzy_willy_high_target_enabled: document.getElementById('config-fz-willy-high-target')?.checked ?? true"):
        assert h.count(frag) == 1, frag
    assert h.count("🎲² WILLY exit V2 (DECISION_LOG 356): TP at the spike high") == 1       # one line in _buildConfigLines …
    assert h.count("lines.push(..._buildConfigLines(cfg, changelog, hr, hr2, status));") == 2   # … which feeds BOTH text exports
    assert "btcfall" not in h
    assert "floor +${_wFl} ✓armed" in h and "floor +${_wFl} after +${_wArm}" in h                # the open-row badge
    assert "🎲² WILLY_FLOOR</span>" in h                                                         # the close reason rendered


def test_model_columns_payload_and_migration(tmp_path, monkeypatch):
    import models
    cols = models.Order.__table__.columns
    for c in ("entry_frenzy_willy_spike_high", "frenzy_willy_tp_target_pct", "willy_floor_armed_at"):
        assert c in cols and cols[c].nullable, c
    assert "entry_frenzy_willy_btc_falling" not in cols and "entry_frenzy_willy_stop_pct" not in cols
    m = _src("main.py")
    assert m.count('"frenzy_willy_tp_target_pct": getattr(o, \'frenzy_willy_tp_target_pct\', None)') == 2 and m.count('"willy_floor_armed_at":') == 2
    # the real migration adds the columns to an OLD orders table (sqlite)
    import database
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.ext.asyncio import create_async_engine
    path = tmp_path / "old.db"
    se = create_engine(f"sqlite:///{path}")
    models.Base.metadata.create_all(se)
    with se.begin() as c:
        for col in ("entry_frenzy_willy_spike_high", "frenzy_willy_tp_target_pct", "willy_floor_armed_at"):
            c.execute(text(f"ALTER TABLE orders DROP COLUMN {col}"))
    assert "frenzy_willy_tp_target_pct" not in [c["name"] for c in inspect(se).get_columns("orders")]
    se.dispose()
    ae = create_async_engine(f"sqlite+aiosqlite:///{path}")
    monkeypatch.setattr(database, "engine", ae)
    asyncio.run(database.init_db())
    asyncio.run(ae.dispose())
    se = create_engine(f"sqlite:///{path}")
    have = [c["name"] for c in inspect(se).get_columns("orders")]
    se.dispose()
    for col in ("entry_frenzy_willy_spike_high", "frenzy_willy_tp_target_pct", "willy_floor_armed_at"):
        assert col in have, col


def test_engine_source_pins():
    eng = _src("services", "trading_engine.py")
    assert "entry_frenzy_willy_spike_high=(flag.get('peak') if _wl else None)" in eng                       # H = frenzy_walk's peak
    assert "_wv2_tp, _wv2_traw = frenzy_willy_high_target(_wv2_th, actual_price, entry_frenzy_willy_spike_high, _wv2_ef," in eng
    assert "frenzy_willy_tp_target_pct=_wv2_tp," in eng and "'frenzy_willy_tp_target_pct': _wv2_tp, 'willy_floor_armed_at': None," in eng
    assert "'frenzy_willy_tp_target_pct': getattr(order, 'frenzy_willy_tp_target_pct', None)," in eng          # the cache rebuild (restart)
    assert eng.count("resting_tp=_rtp_on_m, row=order)") == 1 and eng.count("resting_tp=_rtp_on_rt, row=order_info)") == 1
    assert eng.count("self._willy_v2_arm_note(") == 2 and eng.count('[WILLY_V2]') >= 4
    assert '_reason_base.startswith("WILLY_FLOOR")' in eng
    assert "_rtp_tp = frenzy_willy_row_tp(_rtp_th, order) if _rtp_es in WILLY_EXIT_CLASS" in eng             # 📏 295 shadow at the row's TP


def test_armed_floor_survives_a_later_config_edit():
    """(review) once willy_floor_armed_at is recorded, raising the arm level above the order's target never un-arms it (never re-decided)."""
    row = _row(1.5, peak=1.3, armed_at=datetime(2026, 10, 23))
    assert frenzy_willy_floor_armed(_th(frenzy_willy_floor_arm_pct=2.0), row)
    assert not frenzy_willy_floor_armed(_th(frenzy_willy_floor_arm_pct=2.0), _row(1.5, peak=1.3))   # not yet recorded → today's rule


def test_arm_must_not_sit_below_the_willy_tp():
    """(deep review) arm < frenzy_willy_tp_pct would give every +TP fill a floor; refused on save."""
    assert any("floor_arm_pct must be ≥ frenzy_willy_tp_pct" in e for e in frenzy_willy_v2_config_errors(_th(frenzy_willy_tp_pct=1.5)))
    assert not frenzy_willy_v2_config_errors(_th())
