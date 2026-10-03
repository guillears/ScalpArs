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
