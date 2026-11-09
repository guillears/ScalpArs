"""📈 capacity_path (DECISION_LOG 513): the table ① parser reads ONLY table ① (①b / ①c reuse the sleeve names) and reports unknown rows;
the reserve line is monotone and ≤ the balance; the day walk starts at the start rate with every sleeve at 100 % size, a cap that never
binds leaves the plain compound path, and a binding cap / leverage step slows it."""
import importlib.util, math, os

import numpy as np
import pandas as pd

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "best_case", "capacity_path.py")
_s = importlib.util.spec_from_file_location("capacity_path", _P)
CP = importlib.util.module_from_spec(_s)
_s.loader.exec_module(CP)

MD = """# x
## ① Each sleeve with its best watch line
| Sleeve | Line switched on | Current batch, before → best | Master |
|---|---|---|---|
| LITE RETRY | RETRY_BTC1H ¹ | +0.10 → **+2.08** | +1.15 → +0.00 |
| FRENZY_LONG | LONG_ADX35 | −0.07 → **−0.02** | +0.06 → +0.07 |
| Momentum Short | no line helps | +0.52 | +0.37 |
| Spike Fade | whole | +0.10 → **+2** | +1 |
| New Sleeve | x | +1.00 → **+3.00** | +1 |
## ①b Hold time and trades per day, before → best
| Sleeve | Line | Trades per day |
|---|---|---|
| LITE RETRY | x | 6.0 → 6.0 |
"""
INV = {"reserve_mode": "schedule", "reserve_schedule": "10000:9750, 25000:24250, 100000:90000, 1000000:500000",
       "fee_reserve_usd": 15.0, "fee_reserve_pct": 2.5}


def test_parser_reads_table_one_only_and_reports_unknown(tmp_path):
    p = tmp_path / "BEST_CASE_x.md"
    p.write_text(MD, encoding="utf-8")
    seen = []
    s = CP.best_case_shares(str(p), warn=seen.append)
    assert s == {"FRENZY_LITE_RETRY": 2.08, "FRENZY_LONG": -0.02, "MOMENTUM_SHORT": 0.52, "SPIKE_FADE": 2.0}
    assert len(seen) == 1 and "New Sleeve" in seen[0]


def test_reserve_line_monotone_and_below_balance():
    T = CP.tradeable_fn(inv=INV)
    prev = 0.0
    for e in (5e3, 10e3, 25e3, 1e5, 5e5, 1e6, 3e6):
        t = T(e)
        assert 0 <= t <= e and t >= prev
        prev = t
    assert CP.tradeable_fn(no_reserve=True)(1234.0) == 1234.0


def _fills(cap, lev=10.0):
    return pd.DataFrame(dict(sl=["A", "A"], frac=[0.5, 0.5], cap=[cap, cap], lev=[lev, lev]))


def test_walk_starts_at_rate_and_uncapped_equals_plain_compound():
    T = CP.tradeable_fn(no_reserve=True)
    rate, rel = CP.build(_fills(np.nan), {"A": 1.0}, T, 10000.0)
    assert abs(rate(10000.0, 8.0) - 8.0) < 1e-9 and rel("A", 10000.0) == 1.0
    days = CP.days_to(rate, 8.0, 10000.0, targets=(50e3, 1e6))
    assert days == {t: math.ceil(math.log(t / 1e4) / math.log(1.08)) for t in (50e3, 1e6)}


def test_binding_cap_and_leverage_step_slow_the_path():
    T = CP.tradeable_fn(no_reserve=True)
    rate_c, rel_c = CP.build(_fills(20000.0), {"A": 1.0}, T, 10000.0)          # wanted 0.5 × E per fill: binds above $40k
    assert rel_c("A", 40000.0) == 1.0 and abs(rel_c("A", 80000.0) - 0.5) < 1e-9
    assert CP.days_to(rate_c, 8.0, 10000.0)[1e6] > CP.days_to(CP.build(_fills(np.nan), {"A": 1.0}, T, 10000.0)[0], 8.0, 10000.0)[1e6]
    rate_l, rel_l = CP.build(_fills(np.nan, lev=20.0), {"A": 1.0}, T, 10000.0, "0:20, 25000:15")
    assert rel_l("A", 20000.0) == 1.0 and abs(rel_l("A", 30000.0) - 0.75) < 1e-9
