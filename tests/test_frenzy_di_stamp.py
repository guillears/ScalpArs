"""🔬 FRENZY observe-only stamp: +DI − −DI on the signal bar (DECISION_LOG 187)."""
import os
import sys

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.frenzy import frenzy_di_spread  # noqa: E402


def _bars(n=300, drift=0.002, seed=1):
    r = np.random.default_rng(seed); c = 1.0 * np.cumprod(1 + drift + r.normal(0, 0.004, n))
    return [[i * 300_000, c[i - 1] if i else c[0], c[i] * 1.003, c[i] * 0.997, c[i], 1000.0] for i in range(n)]


def test_matches_the_ta_formula_the_backtest_used_and_has_the_right_sign():
    b = _bars()
    a = ADXIndicator(high=pd.Series([x[2] for x in b]), low=pd.Series([x[3] for x in b]), close=pd.Series([x[4] for x in b]), window=14)
    assert abs(frenzy_di_spread(b) - round(float(a.adx_pos().iloc[-1] - a.adx_neg().iloc[-1]), 3)) < 1e-9
    assert frenzy_di_spread(_bars(drift=0.003)) > 0 > frenzy_di_spread(_bars(drift=-0.003))


def test_the_live_300_bar_window_gives_the_full_history_value():
    b = _bars(n=1500, drift=0.0005, seed=4)
    a = ADXIndicator(high=pd.Series([x[2] for x in b]), low=pd.Series([x[3] for x in b]), close=pd.Series([x[4] for x in b]), window=14)
    assert abs(frenzy_di_spread(b[-300:]) - float(a.adx_pos().iloc[-1] - a.adx_neg().iloc[-1])) < 1e-2


def test_unreadable_input_is_none_never_raises():
    assert frenzy_di_spread(None) is None and frenzy_di_spread([]) is None and frenzy_di_spread(_bars(n=30)) is None
    bad = _bars(); bad[-1] = [0, "x", "y", "z", "w", 1]
    assert frenzy_di_spread(bad) is None


def test_wired_as_an_entry_stamp_only():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "di_spread=(frenzy_di_spread(closed[-300:]) if (ready or code in FRENZY_WIDE_CODES) else None)" in eng
    assert "entry_frenzy_di_spread=flag.get('di_spread')" in eng and "entry_frenzy_di_spread=(entry_frenzy_di_spread if _frenzy else None)" in eng
    lines = [x.strip() for x in eng.splitlines() if "di_spread" in x and not x.strip().startswith("#")]
    allowed = lambda x: (("frenzy_di_spread" in x and ("import" in x or "if (ready or code in FRENZY_WIDE_CODES)" in x
                                                        or "frenzy_di_spread(trunc[-300:]) if _stamp else None" in x)) or "entry_frenzy_di_spread" in x   # ⏪ Oct-6 catch-up: the same stamp on the ON bar
                         or "flag.get('di_spread') is not None" in x or "float(flag['di_spread']) > 0" in x)   # 💪 Oct-4 (197): the ONE rule that reads it (strong-signal leverage)
    assert all(allowed(x) for x in lines), lines
    import models as M
    assert "entry_frenzy_di_spread" in {c.name for c in M.Order.__table__.columns}
    assert "('entry_frenzy_di_spread', 'FLOAT')" in open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
