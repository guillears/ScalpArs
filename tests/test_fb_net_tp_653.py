"""💸 DECISION_LOG 653 — FLATBASE net take profit: the TP moves out by the funding PAID (resting limit + software TP), Route B parity,
config / JSON / UI (D11), the revert scout gate and hooks."""
import json
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from services import flatbase as FB  # noqa: E402
from tests.test_flatbase_short import _th  # noqa: E402

ON = dict(flatbase_stop_pct=10.0, flatbase_tp_pct=5.0, flatbase_stop_counts_funding=True, flatbase_tp_counts_funding=True)


def test_software_tp_moves_out_by_paid_funding_only():
    th, off = _th(**ON), _th(**dict(ON, flatbase_tp_counts_funding=False))
    assert FB.flatbase_exit_for(100.0, 95.0, 10, th, funding_paid_pct=3.0)[0] is False            # +5 % price, 3 % paid → not yet
    assert FB.flatbase_exit_for(100.0, 91.9, 10, th, funding_paid_pct=3.04)[2] == pytest.approx(8.0)   # 0.05 % steps (3.04 → 3.00)
    assert FB.flatbase_exit_for(100.0, 92.0, 10, th, funding_paid_pct=3.0) == (True, "FRENZY_TP", 8.0)
    assert FB.flatbase_exit_for(100.0, 95.0, 10, off, funding_paid_pct=3.0) == (True, "FRENZY_TP", 5.0)   # switch off → price TP
    for fp in (None, 0.0, -2.0, "x", float("nan"), 150.0):                                       # received / garbage → price TP
        assert FB.flatbase_exit_for(100.0, 95.0, 10, th, funding_paid_pct=fp) == (True, "FRENZY_TP", 5.0)
    assert FB.flatbase_exit_for(100.0, 92.0, 10, th, funding_paid_pct=3.0, resting_tp=True)[0] is False   # the resting limit books it
    assert FB.flatbase_exit_for(100.0, 108.5, 10, th, funding_paid_pct=1.6)[1] == "STOP_LOSS"     # the DL 607 stop is unchanged


def test_tp_price_pct_and_net_target():
    th, off = _th(**ON), _th(**dict(ON, flatbase_tp_counts_funding=False))
    assert FB.tp_price_pct(th, 4.23) == pytest.approx(9.20) and FB.tp_price_pct(th, 4.2499) == pytest.approx(9.20) and FB.tp_price_pct(off, 4.23) == 5.0
    assert FB.tp_price_pct(th, None) == 5.0 and FB.tp_price_pct(th, -1) == 5.0 and FB.tp_price_pct(th, 60) == 5.0
    n0, n1 = FB.rtp_net_tp(1.0, 0.045, 0.00018, th), FB.rtp_net_tp(1.0, 0.045, 0.00018, th, funding_paid_pct=2.0)
    assert n1 == pytest.approx(7.0 - 0.045 - 0.018 * (1 - 0.07)) and n0 == pytest.approx(5.0 - 0.045 - 0.018 * 0.95)


def test_resting_limit_reads_the_rows_paid_funding(monkeypatch):
    from services import resting_tp as RT
    monkeypatch.setattr(RT, "rtp_enabled", lambda th, es: True)
    th = _th(**ON)
    row = lambda f: NS(entry_strategy="FLATBASE_SHORT", entry_price=1.0, quantity=1000.0, entry_fee=0.45, funding_fee_usd=f)   # noqa: E731
    base = RT._flatbase_row_target(th, row(None))[0]
    paid = RT._flatbase_row_target(th, row(-30.0))[0]                    # 30 USD paid on a 1,000 USD notional = 3 %
    recv = RT._flatbase_row_target(th, row(+30.0))[0]
    assert paid == pytest.approx(base + 3.0 * (1 + 0.00018), abs=1e-3) and recv == pytest.approx(base)
    assert RT._flatbase_row_target(_th(**dict(ON, flatbase_tp_counts_funding=False)), row(-30.0))[0] == pytest.approx(base)


def test_route_b_walk_parity():
    sys.path.insert(0, os.path.join(ROOT, "scripts", "route_b_signals"))
    import exits as EX
    n = 30
    t = np.arange(n) * 60_000.0
    c = np.r_[np.full(10, 100.0), np.full(n - 10, 94.0)]          # short: −6 % after bar 10
    o = h = l = c
    fund = [(5 * 60_000.0, -0.02, 100.0)]                          # shorts pay 2 % once
    a = EX.walk(o, h, l, c, t, 100.0, -1, 5.0, 10.0, n, funding=fund, stop_counts_funding=True, tp_counts_funding=False, entry_ms=0)
    b = EX.walk(o, h, l, c, t, 100.0, -1, 5.0, 10.0, n, funding=fund, stop_counts_funding=True, tp_counts_funding=True, entry_ms=0)
    assert a["reason"] == "TP" and b["reason"] != "TP"             # the net TP needs −7 % here
    import common as RC
    import inspect
    assert "tp_counts_funding=bool(g(\"flatbase_tp_counts_funding\", False))" in inspect.getsource(RC)


def test_config_json_ui():
    import config as C
    assert type(C.trading_config.thresholds).model_fields["flatbase_tp_counts_funding"].default is False
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]["flatbase_tp_counts_funding"] is True
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-fbs-tp-funding"') == 1
    assert "config.thresholds.flatbase_tp_counts_funding === true" in html
    assert "flatbase_tp_counts_funding: document.getElementById('config-fbs-tp-funding')?.checked ?? false" in html
    assert "+ funding paid, DL 653" in html
    assert "flatbase_tp_counts_funding must be true or false" in " | ".join(FB.flatbase_config_errors(NS(flatbase_tp_counts_funding="yes"), 20.0))


def test_scout_gate_and_hooks():
    import scout_fb_net_tp_653 as S
    assert S.gate([(1.0, False)] * 19)[0] == "COLLECTING"
    assert S.gate([(0.5, False)] * 20)[0] == "KEEP"
    assert S.gate([(-0.5, False)] * 20)[0] == "REVERT"
    assert S.gate([(1.0, False)] * 17 + [(1.0, True)] * 3)[0] == "REVERT"           # ≥ 3 touched-then-stopped even with Σ > 0
    import pandas as pd
    assert "NOT EVALUABLE" in S.report(frame=pd.DataFrame(), state_path=None, floor_ms=None)
    assert "scout_fb_net_tp_653" in open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    assert "scout_fb_net_tp_653.py" in open(os.path.join(ROOT, "scripts", "observe_registry.py")).read()
    assert "reports/SCOUT_FB_NET_TP_653.json" in open(os.path.join(ROOT, ".gitignore")).read()
