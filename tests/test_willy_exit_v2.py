"""🎲²ᵇ WILLY EXIT V2b (DECISION_LOG 356 → 358): a FRENZY_WILLY / FRENZY_WILLY_HT fill's take profit is aimed just below the episode's spike
high (net at H × (1 − k %), maker exit; ≥ the WILLY TP, ≤ the cap), decided ONCE at entry and stored (Order.frenzy_willy_tp_target_pct).
DECISION_LOG 358 REMOVED the V2 profit floor from the engine (no off switch kept): stop −4 throughout, the late TP (+0.25 after 30 min, the
resting limit re-priced) for EVERY trade, the 90-min cap. Target == +1.25 → the pre-356 exit exactly. A row armed under V2 before the deploy
simply runs V2b (willy_floor_armed_at is historical, never read). No network."""
import asyncio, inspect, json, os, random, sys
from datetime import datetime, timedelta
from types import SimpleNamespace as NS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

import services.frenzy as FZ
import services.resting_tp as RTP
from services.frenzy import (frenzy_willy_exit_for, frenzy_willy_high_target, frenzy_willy_tp_at, frenzy_willy_v2_config_errors,
                             frenzy_willy_row_tp)
from services.resting_tp import rtp_row_target, rtp_target, rtp_reason_for_limit, stop_class
from services.exec_shadow import net_pct_at, resting_tp_price

TK, MK = 0.00045, 0.00018
EF = TK * 100   # entry taker fee, % of notional


def _th(**kw):
    base = dict(frenzy_willy_tp_pct=1.25, frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, frenzy_willy_late_tp_minutes=30.0,
                frenzy_willy_late_tp_pct=0.25, frenzy_willy_high_target_enabled=True, frenzy_willy_high_k_pct=1.0,
                frenzy_willy_high_tp_max_pct=8.0, resting_tp_willy_enabled=True)
    base.update(kw)
    return NS(**base)


def _row(tp=None, peak=0.0, armed_at=None, es="FRENZY_WILLY"):
    return dict(entry_strategy=es, frenzy_willy_tp_target_pct=tp, peak_pnl=peak, willy_floor_armed_at=armed_at)


# ─────────────────────────────────────────────────────────────── target math (unchanged from DL 356)

def test_target_far_near_missing_switch_cap():
    th = _th()
    e = 100.0
    t_used, t_raw = frenzy_willy_high_target(th, e, 105.0, EF, MK)
    assert abs(t_raw - net_pct_at("LONG", e, 105.0 * 0.99, EF, MK)) < 1e-3 and abs(t_used - t_raw) < 1e-9 and 3.8 < t_used < 4.0
    lim = resting_tp_price("LONG", e, EF, t_used, MK, 0.01)
    assert abs(lim - 103.95) <= 0.011
    assert frenzy_willy_high_target(th, e, 101.5, EF, MK)[0] == 1.25
    assert frenzy_willy_high_target(th, e, 99.0, EF, MK)[0] == 1.25
    for args in ((e, None, EF, MK), (e, float("nan"), EF, MK), (None, 105.0, EF, MK), (e, 105.0, None, MK), (e, "x", EF, MK)):
        assert frenzy_willy_high_target(th, *args) == (1.25, None)
    assert frenzy_willy_high_target(_th(frenzy_willy_high_target_enabled=False), e, 105.0, EF, MK) == (1.25, None)
    assert frenzy_willy_high_target(th, e, 130.0, EF, MK)[0] == 8.0
    assert frenzy_willy_high_target(_th(frenzy_willy_high_tp_max_pct=5.0), e, 130.0, EF, MK)[0] == 5.0
    assert frenzy_willy_high_target(_th(frenzy_willy_high_k_pct=2.0), e, 105.0, EF, MK)[0] < t_used


# ─────────────────────────────────────────────────────────────── V2b exit: the old exit with the row's own TP

def test_every_row_is_the_old_exit_at_its_own_target():
    """V2b = the pre-356 exit with the TP level replaced by the row's stored target — for EVERY row (legacy, +1.25, runner, V2-armed)."""
    th = _th()
    rnd = random.Random(3)
    for _ in range(4000):
        p = rnd.uniform(-5, 9); pk = max(p, rnd.uniform(-1, 9)); h = rnd.choice([None, rnd.uniform(0, 120)]); rt = rnd.random() < 0.5
        old = frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt)
        assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt, row=_row(None, pk)) == old
        assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt, row=_row(1.25, pk)) == old
        assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt, row={"entry_strategy": "FRENZY_WILLY"}) == old
        T = rnd.choice([1.5, 3.5, 7.42])
        at_t = frenzy_willy_exit_for(p, pk, _th(frenzy_willy_tp_pct=T), h, None, resting_tp=rt)
        assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt, row=_row(T, pk)) == at_t
        assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt, row=_row(T, pk, armed_at=datetime(2026, 10, 23))) == at_t
        assert frenzy_willy_exit_for(p, pk, th, h, None, resting_tp=rt, row=_row(T, pk))[1] != "WILLY_FLOOR"


def test_runner_that_touched_285_and_falls_rides_to_the_stop_or_the_late_tp():
    """the INIT 10-23 shape: target +7.42, peak +2.85, then a fall — V2b never floors."""
    th = _th()
    r = _row(7.42, peak=2.85)
    for p in (2.0, 1.06, 1.0, 0.5, 0.0, -3.99):
        assert frenzy_willy_exit_for(p, 2.85, th, 10, resting_tp=True, row=r)[0] is False, p      # no floor: holds
    assert frenzy_willy_exit_for(-4.0, 2.85, th, 10, resting_tp=True, row=r) == (True, "STOP_LOSS", -4.0)
    assert frenzy_willy_exit_for(-4.5, 2.85, th, 10, resting_tp=True, row=r) == (True, "STOP_LOSS", -4.0)
    assert frenzy_willy_exit_for(1.0, 2.85, th, 45, row=r) == (True, "WILLY_LATE_TP", 0.25)        # late TP for every trade (software)
    assert frenzy_willy_exit_for(1.0, 2.85, th, 45, resting_tp=True, row=r)[0] is False           # resting limit re-priced to +0.25 books it
    assert frenzy_willy_exit_for(0.1, 2.85, th, 90, resting_tp=True, row=r) == (True, "MAX_HOLD_TIME", 0.1)
    assert frenzy_willy_exit_for(7.5, 7.5, th, 20, row=r) == (True, "FRENZY_TP", 7.42)              # the stored target
    assert frenzy_willy_exit_for(7.5, 7.5, th, 20, resting_tp=True, row=r)[0] is False             # the limit books it


def test_row_armed_under_v2_before_the_deploy_runs_v2b():
    th = _th()
    r = NS(**_row(3.5, peak=1.4, armed_at=datetime(2026, 10, 23, 13)))
    assert frenzy_willy_exit_for(1.0, 1.4, th, 10, resting_tp=True, row=r)[0] is False          # V2 would have floored here
    assert frenzy_willy_exit_for(-4.0, 1.4, th, 10, resting_tp=True, row=r) == (True, "STOP_LOSS", -4.0)
    assert frenzy_willy_exit_for(0.3, 1.4, th, 45, row=r) == (True, "WILLY_LATE_TP", 0.25)
    assert rtp_row_target(th, NS(**vars(r), exit_override_at=None, direction="LONG", entry_price=100.0), 40) == (0.25, "WILLY_LATE_TP")


def test_tp_at_has_no_armed_exception():
    th = _th()
    assert "armed" not in inspect.signature(frenzy_willy_tp_at).parameters
    assert frenzy_willy_tp_at(th, 45, tp_used=3.5) == (0.25, "WILLY_LATE_TP")
    assert frenzy_willy_tp_at(th, 10, tp_used=3.5) == (3.5, "FRENZY_TP")
    assert frenzy_willy_tp_at(th, 29.9) == (1.25, "FRENZY_TP") and frenzy_willy_tp_at(th, 30.0) == (0.25, "WILLY_LATE_TP")


def test_switch_off_is_the_pre_356_exit_and_open_runners_keep_their_target():
    off = _th(frenzy_willy_high_target_enabled=False)
    assert frenzy_willy_high_target(off, 100.0, 105.0, EF, MK)[0] == 1.25
    assert frenzy_willy_row_tp(off, _row(3.5)) == 3.5                                          # never re-decided
    assert frenzy_willy_exit_for(1.3, 1.3, off, 10, row=_row(1.25)) == (True, "FRENZY_TP", 1.25)


def test_ht_runs_the_same_exit():
    th = _th()
    for p, pk, h in ((3.6, 3.6, 5), (1.0, 1.4, 10), (1.2, 1.4, 45), (0.3, 0.9, 45), (-4.0, 0.9, 20), (0.2, 1.4, 90)):
        assert (frenzy_willy_exit_for(p, pk, th, h, row=_row(3.5, pk, es="FRENZY_WILLY_HT"))
                == frenzy_willy_exit_for(p, pk, th, h, row=_row(3.5, pk)))
    assert rtp_row_target(th, NS(**_row(3.5, es="FRENZY_WILLY_HT"), exit_override_at=None), 10) == (3.5, "FRENZY_TP")


def test_floor_code_is_gone():
    for name in ("frenzy_willy_floor_levels", "frenzy_willy_floor_applies", "frenzy_willy_floor_armed", "WILLY_V2_ARM_DEF", "WILLY_V2_FLOOR_DEF"):
        assert not hasattr(FZ, name), name
    assert not hasattr(RTP, "rtp_willy_row_args")
    assert "willy_armed" not in inspect.signature(rtp_target).parameters
    assert "willy_armed" not in inspect.signature(rtp_row_target).parameters
    assert "WILLY_FLOOR" not in inspect.getsource(frenzy_willy_exit_for).split('"""')[-1]       # the body never emits it


# ─────────────────────────────────────────────────────────────── resting TP (stored T_used, late re-price for every row)

def _orm(tp=None, peak=0.0, es="FRENZY_WILLY", armed_at=None):
    return NS(entry_strategy=es, frenzy_willy_tp_target_pct=tp, peak_pnl=peak, willy_floor_armed_at=armed_at, exit_override_at=None,
              direction="LONG", entry_price=100.0)


def test_resting_target_uses_the_stored_target_and_reprices_late_for_runners():
    th = _th()
    assert rtp_row_target(th, _orm(3.5), 0.0) == (3.5, "FRENZY_TP")
    assert rtp_row_target(th, _orm(3.5, peak=0.5), 40) == (0.25, "WILLY_LATE_TP")
    assert rtp_row_target(th, _orm(3.5, peak=2.85), 40) == (0.25, "WILLY_LATE_TP")             # peak above +1.25: no exception any more
    assert rtp_row_target(th, _orm(None), 0.0) == (1.25, "FRENZY_TP") and rtp_row_target(th, _orm(None), 40) == (0.25, "WILLY_LATE_TP")
    assert rtp_row_target(_th(resting_tp_willy_enabled=False), _orm(3.5), 0.0) is None
    assert rtp_target(th, "FRENZY_WILLY", 40, willy_tp=3.5) == (0.25, "WILLY_LATE_TP")
    assert rtp_target(th, "FRENZY_WILLY_HT", 10, willy_tp=3.5) == (3.5, "FRENZY_TP")
    lim35 = resting_tp_price("LONG", 100.0, EF, 3.5, MK, 0.01)
    lim025 = resting_tp_price("LONG", 100.0, EF, 0.25, MK, 0.01)
    assert rtp_reason_for_limit(th, "FRENZY_WILLY", "LONG", 100.0, EF, lim35, MK, willy_tp=3.5) == "FRENZY_TP"
    assert rtp_reason_for_limit(th, "FRENZY_WILLY", "LONG", 100.0, EF, lim025, MK, willy_tp=3.5) == "WILLY_LATE_TP"
    assert stop_class("WILLY_FLOOR")   # historical V2-era reason still classified (never waits on a blocked cancel)


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


def test_engine_sweep_reprices_every_runner_late_incl_a_v2_armed_row(monkeypatch):
    """after a restart into V2b: a row armed under V2 (peak 1.4 + willy_floor_armed_at) and an un-armed runner both re-price to +0.25."""
    old = datetime.utcnow() - timedelta(minutes=40)
    rows = [dict(pair="ARMUSDT", entry_strategy="FRENZY_WILLY", frenzy_willy_tp_target_pct=3.5, peak_pnl=1.4, protective_tp_order_id="A1",
                 resting_tp_price=LIM35, resting_tp_filled_qty=0.0, opened_at=old, willy_floor_armed_at=old + timedelta(minutes=2)),
            dict(pair="UNAUSDT", entry_strategy="FRENZY_WILLY", frenzy_willy_tp_target_pct=3.5, peak_pnl=0.6, protective_tp_order_id="B1",
                 resting_tp_price=LIM35, resting_tp_filled_qty=0.0, opened_at=old)]
    T, eng, engine, S, seed, read, log, blocks = _harness(monkeypatch, rows=rows, place=lambda p, px: {"id": "N-" + p, "price": px, "amount": 10.0},
                                                          fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0}, close_px=100.1)
    async def go():
        await seed(); out = await eng.resting_tp_sweep(now=100.0)
        r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert [x[0] for x in out if x[1] == 1] == ["reprice"] and [x[0] for x in out if x[1] == 2] == ["reprice"]
    lim025 = resting_tp_price("LONG", 100.0, EF, 0.25, MK, 0.01)
    assert [x[-1] for x in log if x[0] == "place"] == [lim025, lim025]
    assert r[1].willy_floor_armed_at is not None                                                 # historical stamp untouched


def test_backstop_follows_the_unchanged_stop():
    import services.trading_engine as T
    th = NS(**vars(_th()), broker_backstop_pct=2.5)
    assert T.row_backstop_pct(th, dict(_row(3.5), leverage=10)) == T.row_backstop_pct(th, dict(_row(None), leverage=10)) == T.order_backstop_pct(th, "FRENZY_WILLY", leverage=10)
    assert T.order_backstop_pct(th, "FRENZY_WILLY", leverage=10) == 4.5


# ─────────────────────────────────────────────────────────────── config / migration / UI / export parity

def _src(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


FIELDS = {"frenzy_willy_high_target_enabled": True, "frenzy_willy_high_k_pct": 1.0, "frenzy_willy_high_tp_max_pct": 8.0}
REMOVED = ("frenzy_willy_floor_arm_pct", "frenzy_willy_floor_pct")


def test_config_fields_json_validation():
    import config as C
    f = type(C.trading_config.thresholds).model_fields
    j = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for k, v in FIELDS.items():
        assert f[k].default == v and j[k] == v, k
    for k in REMOVED + ("frenzy_willy_btcfall_stop_pct",):
        assert k not in f and k not in j, k
    # an old saved config that still carries the floor keys loads (keys ignored)
    t = type(C.trading_config.thresholds)(frenzy_willy_floor_arm_pct=1.25, frenzy_willy_floor_pct=1.0)
    assert not hasattr(t, "frenzy_willy_floor_pct") and not hasattr(t, "frenzy_willy_floor_arm_pct")
    assert frenzy_willy_v2_config_errors(_th()) == []
    assert frenzy_willy_v2_config_errors(_th(frenzy_willy_floor_pct=5.0, frenzy_willy_floor_arm_pct=0)) == []   # no floor validation any more
    bad = frenzy_willy_v2_config_errors(_th(frenzy_willy_high_tp_max_pct=1.0, frenzy_willy_high_k_pct=-1, frenzy_willy_high_target_enabled="yes"))
    assert len(bad) == 3
    assert "frenzy_willy_v2_config_errors as _wv2_cfg_errors" in _src("main.py") and "_cfg_errs.extend(_wv2_cfg_errors(new_config.thresholds))" in _src("main.py")


def test_put_config_refuses_bad_v2_values(monkeypatch):
    """PUT /api/config runs the V2 validator (the floor keys are simply dropped by the model)."""
    import main
    src = inspect.getsource(main.update_config)
    assert "_wv2_cfg_errors(new_config.thresholds)" in src and "floor_arm" not in src


def test_ui_inputs_load_save_and_export_line():
    h = _src("templates", "index.html")
    for _id in ("config-fz-willy-high-target", "config-fz-willy-high-k", "config-fz-willy-high-tp-max"):
        assert h.count(f'id="{_id}"') == 1, _id
    for _id in ("config-fz-willy-floor-arm", "config-fz-willy-floor"):
        assert _id not in h, _id
    for k in REMOVED:
        assert k not in h, k
    for frag in ("['config-fz-willy-high-k', 'frenzy_willy_high_k_pct', 1.0]", "['config-fz-willy-high-tp-max', 'frenzy_willy_high_tp_max_pct', 8.0]",
                 "_e.checked = config.thresholds.frenzy_willy_high_target_enabled !== false",
                 "frenzy_willy_high_target_enabled: document.getElementById('config-fz-willy-high-target')?.checked ?? true"):
        assert h.count(frag) == 1, frag
    assert h.count("🎲² WILLY exit V2b (DECISION_LOG 356 → 358): TP at the spike high") == 1     # one line in _buildConfigLines …
    assert h.count("lines.push(..._buildConfigLines(cfg, changelog, hr, hr2, status));") == 2   # … which feeds BOTH text exports
    assert "btcfall" not in h and "_wFl" not in h and "_wArm" not in h and "willy_floor_armed_at" not in h
    assert "🎲² WILLY_FLOOR</span>" in h                                                         # historical close reason still rendered


def test_model_columns_payload_and_migration(tmp_path, monkeypatch):
    import models
    cols = models.Order.__table__.columns
    for c in ("entry_frenzy_willy_spike_high", "frenzy_willy_tp_target_pct", "willy_floor_armed_at"):   # kept (no drops; floor stamp historical)
        assert c in cols and cols[c].nullable, c
    assert "HISTORICAL ONLY" in _src("models.py")
    m = _src("main.py")
    assert m.count('"frenzy_willy_tp_target_pct": getattr(o, \'frenzy_willy_tp_target_pct\', None)') == 2 and m.count('"willy_floor_armed_at":') == 2
    import database
    from sqlalchemy import create_engine, inspect as sinspect, text
    from sqlalchemy.ext.asyncio import create_async_engine
    path = tmp_path / "old.db"
    se = create_engine(f"sqlite:///{path}")
    models.Base.metadata.create_all(se)
    with se.begin() as c:
        for col in ("entry_frenzy_willy_spike_high", "frenzy_willy_tp_target_pct", "willy_floor_armed_at"):
            c.execute(text(f"ALTER TABLE orders DROP COLUMN {col}"))
    se.dispose()
    ae = create_async_engine(f"sqlite+aiosqlite:///{path}")
    monkeypatch.setattr(database, "engine", ae)
    asyncio.run(database.init_db())
    asyncio.run(ae.dispose())
    se = create_engine(f"sqlite:///{path}")
    have = [c["name"] for c in sinspect(se).get_columns("orders")]
    se.dispose()
    for col in ("entry_frenzy_willy_spike_high", "frenzy_willy_tp_target_pct", "willy_floor_armed_at"):
        assert col in have, col


def test_engine_source_pins():
    eng = _src("services", "trading_engine.py")
    assert "entry_frenzy_willy_spike_high=(flag.get('peak') if _wl else None)" in eng
    assert "_wv2_tp, _wv2_traw = frenzy_willy_high_target(_wv2_th, actual_price, entry_frenzy_willy_spike_high, _wv2_ef," in eng
    assert "frenzy_willy_tp_target_pct=_wv2_tp," in eng and "'frenzy_willy_tp_target_pct': _wv2_tp," in eng
    assert "'frenzy_willy_tp_target_pct': getattr(order, 'frenzy_willy_tp_target_pct', None)," in eng          # the cache rebuild (restart)
    assert eng.count("resting_tp=_rtp_on_m, row=order)") == 1 and eng.count("resting_tp=_rtp_on_rt, row=order_info)") == 1
    for gone in ("_willy_v2_arm_note", "willy_armed", "frenzy_willy_floor", "'warm'", " warm=", "willy_floor_armed_at", "floor ARMED", "floor exit"):
        assert gone not in eng, gone
    assert "wtp=getattr(row, 'frenzy_willy_tp_target_pct', None))" in eng                                     # realtime gate: row TP only
    assert "_rtp_tp = frenzy_willy_row_tp(_rtp_th, order) if _rtp_es in WILLY_EXIT_CLASS" in eng
