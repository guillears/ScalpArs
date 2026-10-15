"""🎯 LITE C (DECISION_LOG 301, operator 2026-10-15): a FRENZY_LITE fill's take profit is decided ONCE at entry from its stamped
entry_frenzy_vol_mult / entry_btc_1h_slope and stored on the order (Order.frenzy_lite_tp_used + lite_c_zone).

Pure rule (zone / outside / boundaries / NaN / switch off / fallbacks) · the stored TP's precedence in frenzy_exit_for, frenzy_tp_for and the
resting-TP target · legacy orders (NULL) → the config TP · non-LITE sleeves unaffected · the engine's resting TP placed / kept at the stored TP
(live + paper) · source pins for the entry stamp, the exit paths and the caches · D11 / D12 parity. No network."""
import asyncio, json, os, re, sys
from types import SimpleNamespace as NS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.frenzy import frenzy_lite_c_tp, frenzy_tp_for, frenzy_exit_for, frenzy_lite_tp_override, frenzy_willy_exit_for
from services.resting_tp import rtp_target
from services.exec_shadow import resting_tp_price

NAN = float("nan")


def _th(**kw):
    base = dict(frenzy_tp_pct=1.5, frenzy_lite_tp_pct=3.0, frenzy_lock_arm_pct=0.0, frenzy_stop_pct=3.0, frenzy_trail_arm_pct=5.0,
                frenzy_trail_giveback_pct=1.5, frenzy_lite_c_enabled=True, frenzy_lite_c_vol_mult=20.0, frenzy_lite_c_btc_slope_min=0.0,
                frenzy_lite_c_tp_pct=2.0, resting_tp_willy_enabled=True, resting_tp_lite_enabled=True,
                frenzy_willy_tp_pct=1.25, frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90)
    base.update(kw)
    return NS(**base)


# ─────────────────────────────────────────────────────────────── the pure rule

def test_zone_and_outside():
    th = _th()
    assert frenzy_lite_c_tp(th, 35.0, 0.12) == (3.0, True)          # zone → the normal LITE TP
    assert frenzy_lite_c_tp(th, 19.9, 0.12) == (2.0, False)         # volume short
    assert frenzy_lite_c_tp(th, 35.0, -0.05) == (2.0, False)        # BTC 1h falling
    assert frenzy_lite_c_tp(th, 5.0, -0.3) == (2.0, False)


def test_boundaries():
    th = _th()
    assert frenzy_lite_c_tp(th, 20.0, 0.0001) == (3.0, True)        # vol 20 exactly is IN (≥)
    assert frenzy_lite_c_tp(th, 20.0, 0.0) == (2.0, False)          # slope 0 exactly is OUT (strictly >)
    assert frenzy_lite_c_tp(th, 20.0, -0.0) == (2.0, False)


def test_missing_or_unreadable_fails_to_the_lower_tp():
    th = _th()
    for vm, sl in ((None, 0.2), (40.0, None), (None, None), (NAN, 0.2), (40.0, NAN), ("x", 0.2), (40.0, "bad"),
                   (float("inf"), 0.2), (40.0, float("-inf"))):
        assert frenzy_lite_c_tp(th, vm, sl) == (2.0, False), (vm, sl)
    assert frenzy_lite_c_tp(th, "25", "0.1") == (3.0, True)          # numeric strings are readable


def test_switch_off_is_todays_tp_for_every_fill():
    th = _th(frenzy_lite_c_enabled=False)
    assert frenzy_lite_c_tp(th, 35.0, 0.12) == (3.0, True)
    assert frenzy_lite_c_tp(th, 5.0, -0.3) == (3.0, False)
    assert frenzy_lite_c_tp(th, None, None) == (3.0, False)
    assert frenzy_lite_c_tp(_th(frenzy_lite_c_enabled="true"), 5.0, -0.3)[0] == 3.0   # only a real True arms it


def test_tp_fallbacks():
    assert frenzy_lite_c_tp(_th(frenzy_lite_tp_pct=0.0), 35.0, 0.1) == (1.5, True)          # LITE TP 0 → the shared frenzy_tp_pct (284)
    assert frenzy_lite_c_tp(_th(frenzy_lite_tp_pct=0.0), 5.0, 0.1) == (1.5, False)          # never ABOVE the zone's TP (min)
    assert frenzy_lite_c_tp(_th(frenzy_lite_tp_pct=1.5), 5.0, 0.1) == (1.5, False)          # a LITE TP below 2 is not inverted
    assert frenzy_lite_c_tp(_th(frenzy_lite_tp_pct=0.0, frenzy_tp_pct=0.0), 5.0, 0.1) == (0.0, False)   # TP off stays off
    for bad in (0.0, -1.0, None, NAN, "x"):                                                  # a broken LITE C TP never means "no TP"
        assert frenzy_lite_c_tp(_th(frenzy_lite_c_tp_pct=bad), 5.0, 0.1) == (3.0, False), bad
    assert frenzy_lite_c_tp(_th(frenzy_lite_c_vol_mult=None, frenzy_lite_c_btc_slope_min=NAN), 20.0, 0.01) == (3.0, True)   # cut defaults 20 / 0
    assert frenzy_lite_c_tp(_th(frenzy_lite_c_vol_mult=50.0), 40.0, 0.1) == (2.0, False)
    assert frenzy_lite_c_tp(_th(frenzy_lite_c_btc_slope_min=-0.05), 40.0, -0.01) == (3.0, True)   # a negative floor is allowed
    th = NS(frenzy_tp_pct=1.5, frenzy_lite_tp_pct=3.0)                                                 # fields absent → off
    assert frenzy_lite_c_tp(th, 5.0, -1.0) == (3.0, False)


# ─────────────────────────────────────────────────────────────── stored TP precedence

def test_override_helper():
    assert frenzy_lite_tp_override(2.0) == 2.0 and frenzy_lite_tp_override("3") == 3.0
    for bad in (None, 0, -2.0, NAN, float("inf"), "x"):
        assert frenzy_lite_tp_override(bad) is None, bad


def test_frenzy_tp_for_override_lite_only():
    th = _th()
    assert frenzy_tp_for(th, "FRENZY_LITE", 2.0) == 2.0
    assert frenzy_tp_for(th, "FRENZY_LITE", None) == 3.0            # legacy row → config TP
    assert frenzy_tp_for(th, "FRENZY_LITE", NAN) == 3.0 and frenzy_tp_for(th, "FRENZY_LITE", 0) == 3.0
    assert frenzy_tp_for(th, "FRENZY_LONG", 2.0) == 1.5 and frenzy_tp_for(th, "FRENZY_WIDE", 2.0) == 1.5 and frenzy_tp_for(th, None, 2.0) == 1.5
    assert frenzy_tp_for(th, "FRENZY_LITE") == 3.0                  # the old two-arg call is unchanged


def test_frenzy_exit_for_uses_the_stored_tp():
    th = _th()
    assert frenzy_exit_for(2.0, 2.0, th, use_tp=True, sleeve="FRENZY_LITE", tp_override=2.0) == (True, "FRENZY_TP", 2.0)
    assert frenzy_exit_for(1.9, 1.9, th, use_tp=True, sleeve="FRENZY_LITE", tp_override=2.0)[0] is False
    assert frenzy_exit_for(0.5, 2.3, th, use_tp=True, sleeve="FRENZY_LITE", tp_override=2.0) == (True, "FRENZY_TP", 2.0)   # missed peak catch-up
    assert frenzy_exit_for(2.0, 2.0, th, use_tp=True, sleeve="FRENZY_LITE", tp_override=None)[0] is False   # legacy: +3
    assert frenzy_exit_for(3.0, 3.0, th, use_tp=True, sleeve="FRENZY_LITE", tp_override=None) == (True, "FRENZY_TP", 3.0)
    assert frenzy_exit_for(3.0, 3.0, th, use_tp=True, sleeve="FRENZY_LITE", tp_override=3.0) == (True, "FRENZY_TP", 3.0)   # zone
    assert frenzy_exit_for(2.5, 2.5, th, use_tp=True, sleeve="FRENZY_LITE", tp_override=2.0, resting_tp=True)[0] is False   # a resting TP owns it
    assert frenzy_exit_for(-3.0, 1.0, th, use_tp=True, sleeve="FRENZY_LITE", tp_override=2.0)[:2] == (True, "STOP_LOSS")   # stop unchanged
    assert frenzy_exit_for(-2.9, 1.0, th, use_tp=True, sleeve="FRENZY_LITE", tp_override=2.0)[0] is False
    # non-LITE sleeves / manual FRENZY exit: an override can never move them
    assert frenzy_exit_for(1.5, 1.5, th, use_tp=True, sleeve="FRENZY_LONG", tp_override=2.0) == (True, "FRENZY_TP", 1.5)
    assert frenzy_exit_for(1.5, 1.5, th, use_tp=True, tp_override=2.0) == (True, "FRENZY_TP", 1.5)
    # lock-then-trail armed: no fixed TP for anyone (override irrelevant)
    assert frenzy_exit_for(2.1, 2.1, _th(frenzy_lock_arm_pct=3.0), use_tp=True, sleeve="FRENZY_LITE", tp_override=2.0)[0] is False
    # WILLY's exit has no override path at all
    assert frenzy_willy_exit_for(1.25, 1.25, th, 5)[:2] == (True, "FRENZY_TP")


def test_resting_target_uses_the_stored_tp():
    th = _th()
    assert rtp_target(th, "FRENZY_LITE", 0.0, 2.0) == (2.0, "FRENZY_TP")
    assert rtp_target(th, "FRENZY_LITE", 0.0, None) == (3.0, "FRENZY_TP") and rtp_target(th, "FRENZY_LITE", 0.0) == (3.0, "FRENZY_TP")
    assert rtp_target(th, "FRENZY_LITE", 0.0, NAN) == (3.0, "FRENZY_TP")
    assert rtp_target(th, "FRENZY_WILLY", 0.0, 2.0) == (1.25, "FRENZY_TP")      # WILLY ignores it
    assert rtp_target(th, "FRENZY_LONG", 0.0, 2.0) is None                      # not a resting-TP sleeve
    assert rtp_target(_th(frenzy_lock_arm_pct=3.0), "FRENZY_LITE", 0.0, 2.0) is None
    assert rtp_target(_th(resting_tp_lite_enabled=False), "FRENZY_LITE", 0.0, 2.0) is None


# ─────────────────────────────────────────────────────────────── engine: resting TP placed / kept at the stored TP

def _harness(monkeypatch, **kw):
    from tests.test_resting_tp import _setup
    return _setup(monkeypatch, **kw)


EF = 0.045
MK = 0.00018
TICK = 0.01
LIM3 = resting_tp_price("LONG", 100.0, EF, 3.0, MK, TICK)
LIM2 = resting_tp_price("LONG", 100.0, EF, 2.0, MK, TICK)


def _arm_all(monkeypatch, paper, rows):
    T, eng, engine, S, seed, read, log, blocks = _harness(monkeypatch, paper=paper, rows=rows,
                                                          place=lambda p, px: {"id": "O-" + p, "price": px, "amount": 10.0})
    async def go():
        await seed()
        from sqlalchemy import select
        import models
        async with S() as s:
            for o in (await s.execute(select(models.Order).order_by(models.Order.id))).scalars().all():
                await eng._rtp_arm(s, o)
        r = await read(); await engine.dispose(); return r
    return asyncio.run(go()), log


def test_engine_arm_places_the_stored_tp_live_and_paper(monkeypatch):
    assert LIM2 < LIM3
    rows = [dict(pair="OUTUSDT", entry_strategy="FRENZY_LITE", frenzy_lite_tp_used=2.0, lite_c_zone=False),
            dict(pair="ZONUSDT", entry_strategy="FRENZY_LITE", frenzy_lite_tp_used=3.0, lite_c_zone=True),
            dict(pair="OLDUSDT", entry_strategy="FRENZY_LITE")]                                       # pre-301 row: NULL → config +3
    r, log = _arm_all(monkeypatch, False, rows)
    assert [x for x in log if x[0] == "place"] == [("place", "OUTUSDT", "LONG", 10.0, LIM2), ("place", "ZONUSDT", "LONG", 10.0, LIM3),
                                                   ("place", "OLDUSDT", "LONG", 10.0, LIM3)]
    assert r[1].resting_tp_price == LIM2 and r[2].resting_tp_price == LIM3 and r[3].resting_tp_price == LIM3
    rp, logp = _arm_all(monkeypatch, True, rows)                                                      # paper = live
    assert logp == [] and rp[1].resting_tp_price == LIM2 and rp[2].resting_tp_price == LIM3 and rp[3].resting_tp_price == LIM3


def test_engine_sweep_keeps_the_stored_tp_and_never_redecides_a_legacy_row(monkeypatch):
    from datetime import datetime, timedelta
    old = datetime.utcnow() - timedelta(minutes=5)
    rows = [dict(pair="OUTUSDT", entry_strategy="FRENZY_LITE", frenzy_lite_tp_used=2.0, protective_tp_order_id="A1", resting_tp_price=LIM2,
                 resting_tp_filled_qty=0.0, opened_at=old),                                           # stored +2 at its limit → untouched
            dict(pair="OLDUSDT", entry_strategy="FRENZY_LITE", protective_tp_order_id="B1", resting_tp_price=LIM3,
                 resting_tp_filled_qty=0.0, opened_at=old),                                           # legacy +3 at its limit → untouched
            dict(pair="BADUSDT", entry_strategy="FRENZY_LITE", frenzy_lite_tp_used=2.0, protective_tp_order_id="C1", resting_tp_price=LIM3,
                 resting_tp_filled_qty=0.0, opened_at=old),                                           # stored +2 resting at +3 → re-priced to +2
            dict(pair="HEAUSDT", entry_strategy="FRENZY_LITE", frenzy_lite_tp_used=2.0, opened_at=old)]   # none → healed at +2
    T, eng, engine, S, seed, read, log, blocks = _harness(monkeypatch, rows=rows, place=lambda p, px: {"id": "N-" + p, "price": px, "amount": 10.0},
                                                          fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    async def go():
        await seed(); out = await eng.resting_tp_sweep(now=100.0)
        r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert sorted(out) == [("heal", 4, "placed"), ("reprice", 3, "replaced")]
    assert r[1].protective_tp_order_id == "A1" and r[1].resting_tp_price == LIM2
    assert r[2].protective_tp_order_id == "B1" and r[2].resting_tp_price == LIM3
    assert r[3].protective_tp_order_id == "N-BADUSDT" and r[3].resting_tp_price == LIM2
    assert r[4].protective_tp_order_id == "N-HEAUSDT" and r[4].resting_tp_price == LIM2
    assert not [x for x in log if x[1:2] in (("OUTUSDT",), ("OLDUSDT",))]


# ─────────────────────────────────────────────────────────────── source pins (entry stamp, exits, caches)

def _src(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def test_entry_decision_uses_the_stamped_values_and_every_consumer_reads_the_row():
    eng = _src("services", "trading_engine.py")
    # decided from the SAME variables the Order constructor stamps (entry_frenzy_vol_mult / entry_btc_1h_slope), LITE only, before Order(
    i = eng.index("_lc_v, _lc_zone = frenzy_lite_c_tp(_lc_th, entry_frenzy_vol_mult, entry_btc_1h_slope)")
    j = eng.index("        order = Order(\n            binance_order_id=binance_order_id,")
    assert i < j and 'if _frenzy and _fz_es == "FRENZY_LITE":' in eng[i - 600:i]
    assert "entry_btc_1h_slope=entry_btc_1h_slope," in eng[j:j + 6000]
    assert "entry_frenzy_vol_mult=(entry_frenzy_vol_mult if _frenzy else None)" in eng[j:j + 9000]
    assert "frenzy_lite_tp_used=_lc_tp_used, lite_c_zone=_lc_zone," in eng[j:j + 12000]
    assert eng.count("frenzy_lite_tp_used=_lc_tp_used") == 1
    # exits: the sleeve's candle + realtime paths pass the stored TP; the manual FRENZY exit does not
    assert eng.count("frenzy_exit_for(") == 4 and eng.count("tp_override=") == 2
    assert "resting_tp=_rtp_on_m, tp_override=getattr(order, 'frenzy_lite_tp_used', None))" in eng
    assert "resting_tp=_rtp_on_rt, tp_override=order_info.get('frenzy_lite_tp_used'))" in eng
    # both cache builders (open + rebuild) carry it — the realtime path reads order_info
    assert "'frenzy_lite_tp_used': _lc_tp_used," in eng and "'frenzy_lite_tp_used': getattr(order, 'frenzy_lite_tp_used', None)," in eng
    # every resting-TP target that reads a row passes the stored TP (open, re-price, heal, sweep); the WILLY-only late swap needs none
    calls = re.findall(r"rtp_target\(([^\n]*)\)", eng)
    lite_calls = [c for c in calls if "st['es']" not in c]
    assert len(lite_calls) == 4 and all("frenzy_lite_tp_used" in c for c in lite_calls), calls
    assert "frenzy_tp_for(_rtp_th, _rtp_es, getattr(order, 'frenzy_lite_tp_used', None))" in eng   # 📏 295 shadow context
    rtp = _src("services", "resting_tp.py")
    assert 'frenzy_tp_for(th, es, lite_tp), "FRENZY_TP"' in rtp


def test_model_migration_config_ui_and_report_parity():
    import models, config as C
    cols = models.Order.__table__.columns
    assert "frenzy_lite_tp_used" in cols and "lite_c_zone" in cols and cols["frenzy_lite_tp_used"].nullable and cols["lite_c_zone"].nullable
    db = _src("database.py")
    assert 'ADD COLUMN frenzy_lite_tp_used FLOAT' in db and 'ADD COLUMN lite_c_zone BOOLEAN' in db
    f = type(C.trading_config.thresholds).model_fields
    assert f["frenzy_lite_c_enabled"].default is False and f["frenzy_lite_c_vol_mult"].default == 20.0
    assert f["frenzy_lite_c_btc_slope_min"].default == 0.0 and f["frenzy_lite_c_tp_pct"].default == 2.0
    j = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert (j["frenzy_lite_c_enabled"], j["frenzy_lite_c_vol_mult"], j["frenzy_lite_c_btc_slope_min"], j["frenzy_lite_c_tp_pct"]) == (True, 20.0, 0.0, 2.0)
    h = _src("templates", "index.html")
    for _id in ("config-fz-lite-c-enabled", "config-fz-lite-c-vol", "config-fz-lite-c-slope", "config-fz-lite-c-tp", "lite-c-body"):
        assert h.count(f'id="{_id}"') == 1, _id
    assert "['config-fz-lite-c-vol', 'frenzy_lite_c_vol_mult', 20], ['config-fz-lite-c-slope', 'frenzy_lite_c_btc_slope_min', 0], ['config-fz-lite-c-tp', 'frenzy_lite_c_tp_pct', 2.0]" in h
    assert "config.thresholds.frenzy_lite_c_enabled === true" in h and "frenzy_lite_c_enabled: document.getElementById('config-fz-lite-c-enabled')" in h
    assert "_key !== 'frenzy_lite_c_btc_slope_min'" in h                                   # a negative slope floor survives the save
    assert "LITE C: TP +" in h and "(DECISION_LOG 301)" in h
    assert "Number(o.frenzy_lite_tp_used) > 0) ? Number(o.frenzy_lite_tp_used)" in h       # open-order badge shows the stored TP
    # D12: one UI table + BOTH text exports (clipboard copyAnalytics + saved-file _buildPerfLines)
    assert h.count("lines.push(...liteCReportLines(perf, hr2));") == 2 and h.count("getElementById('lite-c-body')") == 1
    assert h.count("LITE C Tally") >= 2
    m = _src("main.py")
    assert m.count('"lite_c_rows": [],') == 2 and '"lite_c_rows": lite_c_rows,' in m
    assert m.count('"frenzy_lite_tp_used": getattr(o, \'frenzy_lite_tp_used\', None)') == 2
    assert '"lite_c_enabled": getattr(_th, \'frenzy_lite_c_enabled\', False) is True' in m
