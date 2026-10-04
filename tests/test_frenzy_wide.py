"""🔥🌐 Oct-3 FRENZY-WIDE: takes ONLY the fresh FRENZY setups refused for the ATR cap or a green signal candle; own tag / size / slots."""
import os
from types import SimpleNamespace as NS

from services.frenzy import frenzy_wide_ready, frenzy_vol_trend, frenzy_adx_delta, frenzy_long_status, FRENZY_WIDE_CODES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = NS(frenzy_wide_enabled=True, frenzy_state_vol_mult=100.0, frenzy_min_volume_usd=20e6, frenzy_max_atr_pct=2.5, frenzy_long_skip_green_bar=True)
EP = dict(in_state=True, above_hour=True, fresh_on=True, vol_mult=150.0, hours=3.0, bar_red=True, bar_ret_pct=-0.1)


def test_wide_takes_only_atr_and_green_refusals():
    ok, code, _ = frenzy_long_status(EP, 6.2, 50e6, TH)                         # AIN Oct-3: ATR 6.2 %
    assert not ok and code == "FRENZY_ATR_HIGH" and frenzy_wide_ready(EP, code, TH, 6.2)
    ep = dict(EP, bar_red=False, bar_ret_pct=0.3)
    ok, code, _ = frenzy_long_status(ep, 1.0, 50e6, TH)
    assert not ok and code == "FRENZY_GREEN_BAR" and frenzy_wide_ready(ep, code, TH, 1.0)
    ok, code, _ = frenzy_long_status(EP, 1.0, 50e6, TH)
    assert ok and not frenzy_wide_ready(EP, code, TH, 6.2)                           # FRENZY takes it — WIDE never doubles it
    for ep, vol24 in ((dict(EP, fresh_on=False), 50e6), (EP, 5e6), (dict(EP, in_state=False), 50e6)):
        ok, code, _ = frenzy_long_status(ep, 6.2, vol24, TH)
        assert not ok and code not in FRENZY_WIDE_CODES and not frenzy_wide_ready(ep, code, TH, 1.0)   # every other gate still binds
    assert not frenzy_wide_ready(EP, "FRENZY_ATR_HIGH", NS(frenzy_wide_enabled=False), 6.2)       # the switch
    ok, code, _ = frenzy_long_status(EP, None, 50e6, TH)
    assert code == "FRENZY_ATR_HIGH" and not frenzy_wide_ready(EP, code, TH, None)            # unreadable ATR: fail closed (review)
    assert not frenzy_wide_ready(dict(EP, fresh_on=False), "FRENZY_ATR_HIGH", TH, 6.2)                 # first candle only


def test_observe_stamps():
    bars = [[i * 300_000, 1, 1.01, 0.99, 1.0, 100.0] for i in range(12)] + [[i * 300_000, 1, 1.01, 0.99, 1.0, 250.0] for i in range(12, 24)]
    assert frenzy_vol_trend(bars) == 2.5 and frenzy_vol_trend(bars[:10]) is None and frenzy_vol_trend([[0, 1, 1, 1, 1, "x"]] * 30) is None
    up = [[i, 1 + i * 0.01, 1 + i * 0.01 + 0.02, 1 + i * 0.01 - 0.005, 1 + i * 0.01 + 0.015, 10] for i in range(120)]
    assert frenzy_adx_delta(up) is not None and frenzy_adx_delta(up[:30]) is None


def test_wiring():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert 'FRENZY_STRATEGIES = ("FRENZY_LONG", "FRENZY_WIDE")' in eng
    assert "if frenzy_wide_ready(ep, code, th, atr):" in eng and "await self._frenzy_open(db, flag, ind, bar_open, wide=True)" in eng
    assert "'frenzy_wide_max_slots' if wide else 'frenzy_max_slots'" in eng and "_sg_pref = _fz_es.lower() if _frenzy else" in eng
    assert '(order.entry_strategy or "") == "FRENZY_LONG"' not in eng                    # every exit / hold / urgent path takes both tags
    import models as M
    cols = {c.name for c in M.Order.__table__.columns}
    assert {"entry_frenzy_adx_delta", "entry_frenzy_vol_trend"} <= cols
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "('entry_frenzy_adx_delta', 'FLOAT')" in db and "('entry_frenzy_vol_trend', 'FLOAT')" in db


def test_global_volume_ratio_on_the_closed_signal_bar():
    from services.frenzy import global_volume_ratio
    def pair(last_vol, n=60, t0=0):
        return [[t0 + i * 300_000, 1, 1, 1, 1, (last_vol if i == n - 1 else 100.0)] for i in range(n)]
    sig = 59 * 300_000
    bars = {f"P{i}": pair(200.0) for i in range(30)}
    v = global_volume_ratio(bars, sig)
    assert v is not None and abs(v - 200.0 / (100.0 * 47 / 48 + 200.0 / 48)) < 1e-3          # Σ bar ÷ Σ 48-bar mean incl. the bar
    assert global_volume_ratio({f"P{i}": pair(200.0) for i in range(29)}, sig) is None        # < 30 readable pairs
    assert global_volume_ratio(bars, sig + 300_000) is None                                    # that bar not in the data → unreadable
    assert global_volume_ratio({**bars, "BAD": [[sig, 1, 1, 1, 1, "x"]]}, sig) == v            # a broken pair is skipped, never raises


def test_gvol_gate_wired_fail_closed():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("async def _frenzy_open(")
    body = eng[i:eng.index("async def _maybe_open_surge(", i)]
    assert "_gv = await self._frenzy_gvol_value(bar_open - 300_000, wait=_gvmax > 0)" in body
    assert "if _gvmax > 0 and (_gv is None or _gv >= _gvmax):" in body                         # unreadable = no entry
    assert body.index("_frenzy_gvol_value(") < body.index("FRENZY_ENTRY_MAX_LATE_S")            # the wait counts toward lateness
    assert "self._frenzy_gvol_start(allp, bar_open - 300_000)" in eng and "entry_frenzy_gvol=_gv" in body
    import json
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_gvol_max"] == 1.0


def test_gvol_gate_behaviour(monkeypatch):
    """A busy market (≥ max) and an unreadable one refuse with their own counters; a quiet one reaches the open; 0 = gate off."""
    import asyncio, time
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, config as C
    import services.trading_engine as TE
    th = C.trading_config.thresholds
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**6)
    bar_open = int(time.time() // 300) * 300_000

    async def run(gv, gmax, wide=False):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        old = th.frenzy_gvol_max
        try:
            th.frenzy_gvol_max = gmax
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                e = object.__new__(TE.TradingEngine); e.is_paper_mode = True; blocks = []; opened = []
                e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
                e._flip_entry_fields = lambda *a, **k: {}
                e._sanitize_open_kwargs = lambda ef, s, d: ef

                async def gvv(sig, wait=True):
                    return gv
                e._frenzy_gvol_value = gvv

                async def fake_open(**kw):
                    opened.append(kw.get("entry_frenzy_gvol")); return None
                e.open_position = fake_open
                flag = dict(pair="FOOUSDT", spike_ts=bar_open - 7_200_000, hours=2.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=150.0, run_pct=20.0,
                            atr_pct=1.5, volume_24h=5e7, price=1.0, live_price=1.0)
                await e._frenzy_open(db, flag, {}, bar_open, wide=wide)
        finally:
            th.frenzy_gvol_max = old
            await eng.dispose()
        return blocks, opened, flag.get("last_fire") or ""
    b, o, lf = asyncio.run(run(1.3, 1.0))
    assert "FRENZY_GVOL_HIGH" in b and o == [] and "market volume 1.30× normal ≥ 1×" in lf
    b, o, lf = asyncio.run(run(None, 1.0, wide=True))
    assert "FRENZY_WIDE_GVOL_UNREAD" in b and o == [] and "WIDE refused: market volume unreadable" in lf
    b, o, _ = asyncio.run(run(0.8, 1.0))
    assert not [x for x in b if "GVOL" in x] and o == [0.8]                                    # quiet: reaches the open, stamped
    b, o, _ = asyncio.run(run(1.5, 0.0))
    assert not [x for x in b if "GVOL" in x] and o == [1.5]                                    # 0 = off (still stamped)


def test_gvol_value_never_leaks_a_cancelled_read():
    """Review (Oct-3): a read task that ended CANCELLED (ccxt's shared load_markets) must give None, never raise into frenzy_loop."""
    import asyncio
    import services.trading_engine as TE

    async def run():
        e = object.__new__(TE.TradingEngine)
        t = asyncio.create_task(asyncio.sleep(10)); await asyncio.sleep(0); t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        e._fz_gvol_task = (1000, t)
        a = await e._frenzy_gvol_value(1000)
        done = asyncio.create_task(asyncio.sleep(0, result=0.7)); await done
        e._fz_gvol_task = (1000, done)
        return a, await e._frenzy_gvol_value(1000), await e._frenzy_gvol_value(2000), await e._frenzy_gvol_value(1000, wait=False)
    assert asyncio.run(run()) == (None, 0.7, None, 0.7)


def test_fixed_take_profit():
    """🎯 Oct-4 (196): frenzy_tp_pct closes at +tp net (reason FRENZY_TP, before the trail) when use_tp — sleeve AND manual FRENZY exit."""
    from services.frenzy import frenzy_exit_for
    th = NS(frenzy_stop_pct=3.0, frenzy_trail_arm_pct=5.0, frenzy_trail_giveback_pct=1.5, frenzy_tp_pct=4.0)
    assert frenzy_exit_for(4.0, 4.0, th, use_tp=True) == (True, "FRENZY_TP", 4.0)
    assert frenzy_exit_for(3.99, 3.99, th, use_tp=True)[0] is False
    assert frenzy_exit_for(-3.0, 1.0, th, use_tp=True) == (True, "STOP_LOSS", -3.0)
    assert frenzy_exit_for(4.2, 4.2, th, short=True, use_tp=True) == (True, "FRENZY_TP", 4.0)   # a manual SHORT on the FRENZY exit
    assert frenzy_exit_for(5.5, 7.0, th)[1] == "RUNNER_TRAIL"                   # use_tp False → the trail (kept for completeness)
    assert frenzy_exit_for(2.0, 4.3, th, use_tp=True) == (True, "FRENZY_TP", 4.0)   # missed TP (peak passed it): close now, never ride to the stop
    assert frenzy_exit_for(-1.0, 4.3, th, use_tp=True, stop_floor=-2.5)[0] is True
    assert frenzy_exit_for(6.0, 9.0, NS(**{**th.__dict__, "frenzy_tp_pct": 0.0}), use_tp=True)[1] == "RUNNER_TRAIL"   # tp off → trail fallback
    assert frenzy_exit_for(4.5, 4.5, NS(**{**th.__dict__, "frenzy_tp_pct": -4.0}), use_tp=True)[0] is False   # negative = off
    assert frenzy_exit_for(4.0, 6.0, NS(**{**th.__dict__, "frenzy_tp_pct": 3.0}), use_tp=True) == (True, "FRENZY_TP", 3.0)   # tp < arm: a peak past the arm still takes the TP
    assert frenzy_exit_for(4.5, 4.5, NS(**{**th.__dict__, "frenzy_tp_pct": 0.0}), use_tp=True)[0] is False   # 0 = off → trail


def test_fixed_tp_wiring():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng.count("use_tp=True)   # 🎯 fixed TP (196)\n") == 2                # sleeve candle + realtime paths
    assert eng.count("use_tp=True)   # 🎯 fixed TP (196): manual FRENZY exit too") == 2   # manual candle + realtime paths
    assert eng.count('"FRENZY_TP": "MANUAL_TP"') == 2
    assert eng.count('_reason_base.startswith("FRENZY_TP")') == 2               # post-exit tracking (live + recovery whitelists)
    import json
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]["frenzy_tp_pct"] == 3.0


def test_strong_signal_leverage_wiring():
    """💪 Oct-4 (197): FRENZY_LONG with ADX rising ∧ +DI above −DI → frenzy_long_lev_mult_strong (absolute); WIDE never; unreadable = normal."""
    import json
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "frenzy_strong=(not wide and flag.get('adx_delta') is not None and flag.get('di_spread') is not None" in eng
    assert "and float(flag['adx_delta']) > 0 and float(flag['di_spread']) > 0)" in eng
    i = eng.index("if _frenzy and _fz_es == \"FRENZY_LONG\" and frenzy_strong:")
    assert eng.index("_sg_inv = getattr(_th_sg") < i < eng.index("cell_lev_mult = max(0.05, min(1.0 if _sg_lev is None")   # replaces the lev mult BEFORE the clamp
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_long_lev_mult_strong"] == 0.5 and cfg["frenzy_long_lev_mult"] == 0.32
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-fz-lev-mult-strong"') == 1 and "['config-fz-lev-mult-strong', 'frenzy_long_lev_mult_strong', 0.5]" in html



def test_frenzy_open_keeps_every_stamp_kwarg():
    """Oct-4 review: an inline comment once swallowed entry_frenzy_spike_at= (the episode id) — every stamp must be a live kwarg in _frenzy_open."""
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("async def _frenzy_open("); body = eng[i:eng.index("async def _maybe_open_surge(", i)]
    code = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())          # comments stripped
    for kw in ("entry_frenzy_spike_at=datetime.utcfromtimestamp(flag['spike_ts'] / 1000)", "entry_frenzy_hours=", "entry_frenzy_vwap=",
               "entry_frenzy_vol_mult=", "entry_frenzy_di_spread=flag.get('di_spread')", "entry_frenzy_adx_delta=flag.get('adx_delta')",
               "entry_frenzy_gvol=_gv", "frenzy_strong=("):
        assert kw in code, kw
