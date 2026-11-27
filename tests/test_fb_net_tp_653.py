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


def _arrays(n, closes):
    BAR = 300_000
    t = (np.arange(n, dtype="int64") * BAR) + 1_800_000_000_000 // BAR * BAR
    c = np.asarray(closes, float)
    return t, c.copy(), c.copy(), c.copy(), c.copy()


def test_walk_arrays_net_tp_moves_out_by_paid_funding_656():
    """🔁 (DECISION_LOG 656) the ONE Flatbase net-exit pricer: real funding paid moves the TP out (NTP) — the price TP (base) does not."""
    import scout_fb_net_tp_653 as S
    n = 900
    closes = np.r_[np.full(10, 100.0), np.full(n - 10, 94.0)]       # −6 % after bar 10, then flat
    t, o, h, l, c = _arrays(n, closes)
    fs_t, fs_r = np.array([int(t[5]) + 1]), np.array([-0.02])          # shorts PAY 2 % once (a negative rate)
    w = S.walk_arrays(t, o, h, l, c, 1, 100.0, int(t[0]), fs_t, fs_r, 5.0, 10.0, int(t[1]) + 72 * 3_600_000)
    assert w["base_why"] == "TP" and w["NTP_why"] != "TP"             # the net TP needs −7 %: today's exit never fills at −6
    w0 = S.walk_arrays(t, o, h, l, c, 1, 100.0, int(t[0]), np.array([], "int64"), np.array([]), 5.0, 10.0, int(t[1]) + 72 * 3_600_000)
    assert w0["NTP_why"] == "TP" and abs(w0["NTP_net"] - w0["base_net"]) < 1e-9   # no funding → the two TPs coincide
    w6 = S.walk_arrays(t, o, h, l, c, 1, 100.0, int(t[0]), np.array([], "int64"), np.array([]), 6.0, 10.0, int(t[1]) + 72 * 3_600_000)
    assert w6["NTP_why"] == "TIME"                                      # TP +6 needs −6.05 % through: never fills, held to 72 h
    assert S.walk_arrays(t[:0], o[:0], h[:0], l[:0], c[:0], 0, 100.0, 0, None, None, 5.0, 10.0, 0) is None


def test_walk_fill_readiness_entry_modes_and_fetch_stop_656(monkeypatch, tmp_path):
    import scout_fb_net_tp_653 as S
    monkeypatch.setattr(S, "FUND_CACHE", str(tmp_path / "fund"))
    monkeypatch.setattr(S, "_STOP", {"why": None})
    n = 900
    t, o, h, l, c = _arrays(n, np.r_[np.full(30, 100.0), np.full(n - 30, 94.0)])
    calls = []

    class ET:
        @staticmethod
        def real_arrays(pair, a, b):
            calls.append(pair)
            if pair == "BADUSDT":
                raise SystemExit("Binance 429 — stopped")
            if pair == "NONEUSDT":
                raise IndexError("arrays_from on an empty frame")   # the real study helper on a delisted / invalid symbol
            return t, o, h, l, c, c

    fake = type("RR", (), {"ET": ET, "fetch_funding": staticmethod(lambda p, a, b: (np.array([], "int64"), np.array([])))})
    monkeypatch.setitem(S._MODS, "fbfr", fake)
    monkeypatch.setattr(S, "_PATHS", {})
    te = int(t[2]) + 12_000                                            # a real fill 12 s into bar 2
    late = int(t[-1])
    assert S.walk_fill("AUSDT", te, 100.0, now_ms=te + 3_600_000) is None and calls == []   # the 72 h path is not out yet → no fetch
    w = S.walk_fill("AUSDT", te, 100.0, now_ms=late)
    assert isinstance(w, dict) and w["NTP_why"] == "TP"
    w6 = S.walk_fill("AUSDT", te, 100.0, 6.0, now_ms=late)            # tp 6 re-uses the cached path
    assert calls == ["AUSDT"] and w6["NTP_why"] == "TIME"
    s = S.walk_fill("AUSDT", int(t[29]), None, now_ms=late, entry="open")   # a shadow at bar 29's open: E = 100
    assert s["NTP_why"] == "TP"
    off = S.walk_fill("AUSDT", int(t[29]) + 1, None, now_ms=late, entry="open")   # off the 5m grid → the bar at its floor
    assert off["NTP_why"] == "TP"
    assert S.walk_fill("NONEUSDT", te, 100.0, now_ms=late) == "NO_DATA"
    with pytest.raises(RuntimeError):
        S.walk_fill("BADUSDT", te, 100.0, now_ms=late)                 # a 429 never kills the scout — the caller holds the read
    monkeypatch.setattr(S, "_STOP", {"why": None})                     # (a new run)
    assert S.price_fill("NONEUSDT", te, 100.0) is None
    # closed windows' funding is cached on disk: a new run (empty memory) reads it back without a fetch
    n_f = []
    fake2 = type("RR", (), {"ET": ET, "fetch_funding": staticmethod(lambda p, a, b: n_f.append(p) or (np.array([1], "int64"), np.array([0.001])))})
    monkeypatch.setitem(S._MODS, "fbfr", fake2)
    monkeypatch.setattr(S, "_PATHS", {})
    S.walk_fill("CUSDT", te, 100.0, now_ms=late)
    monkeypatch.setattr(S, "_PATHS", {})
    S.walk_fill("CUSDT", te, 100.0, now_ms=late)
    assert n_f == ["CUSDT"] and sum(f.startswith("CUSDT_") for f in os.listdir(tmp_path / "fund")) == 1


def test_walk_fill_stops_after_429_and_fills_a_cut_tail_656(monkeypatch, tmp_path):
    """(656 review) a 418 / 429 stops every later fetch of the run · a cached path missing its tail is completed, never priced as TIME."""
    import pandas as pd
    import scout_fb_net_tp_653 as S
    monkeypatch.setattr(S, "FUND_CACHE", str(tmp_path / "fund"))
    monkeypatch.setattr(S, "_STOP", {"why": None})
    monkeypatch.setattr(S, "_PATHS", {})
    n = 900
    t, o, h, l, c = _arrays(n, np.r_[np.full(30, 100.0), np.full(n - 30, 94.0)])
    calls, tails = [], []

    class ET:
        @staticmethod
        def real_arrays(pair, a, b):
            calls.append(pair)
            if pair == "RLUSDT":
                raise SystemExit("Binance 429 — stopped")
            k = 840 if pair == "CUTUSDT" else n                          # CUT: the cached file ends ≈ 2 h before the walk ends
            return t[:k], o[:k], h[:k], l[:k], c[:k], c[:k]

        @staticmethod
        def fetch(pair, a, b):
            tails.append(pair)
            m = (t >= a) & (t < b)
            return pd.DataFrame(dict(open_time=t[m], o=o[m], h=h[m], l=l[m], c=c[m], vol=np.zeros(m.sum())))

        arrays_from = staticmethod(lambda d: _arrays_from(d))
    fake = type("RR", (), {"ET": ET, "fetch_funding": staticmethod(lambda p, a, b: (np.array([], "int64"), np.array([])))})
    monkeypatch.setitem(S._MODS, "fbfr", fake)
    te, late = int(t[2]) + 12_000, int(t[-1]) + 30 * 3_600_000
    w = S.walk_fill("CUTUSDT", te, 100.0, now_ms=late)
    assert tails == ["CUTUSDT"] and isinstance(w, dict) and w["NTP_why"] == "TP"
    with pytest.raises(RuntimeError):
        S.walk_fill("RLUSDT", te, 100.0, now_ms=late)
    with pytest.raises(RuntimeError, match="earlier this run"):
        S.walk_fill("OKUSDT", te, 100.0, now_ms=late)                  # nothing more is fetched after a 429
    assert calls == ["CUTUSDT", "RLUSDT"]
    assert isinstance(S.walk_fill("CUTUSDT", te, 100.0, 6.0, now_ms=late), dict)   # a cached path still prices


def _arrays_from(d):
    d = d.drop_duplicates("open_time", keep="last").sort_values("open_time")
    return (d.open_time.values.astype("int64"), d.o.values.astype(float), d.h.values.astype(float), d.l.values.astype(float),
            d.c.values.astype(float), d.vol.values.astype(float))


def test_walk_fill_cut_path_waits_then_goes_no_data_656(monkeypatch, tmp_path):
    """(656 review) a path whose tail cannot be fetched waits a day, then is NO_DATA (out) — never priced as an early TIME exit."""
    import pandas as pd
    import scout_fb_net_tp_653 as S
    monkeypatch.setattr(S, "FUND_CACHE", str(tmp_path / "fund"))
    monkeypatch.setattr(S, "_STOP", {"why": None})
    n = 900
    t, o, h, l, c = _arrays(n, np.full(n, 100.0))

    class ET:
        real_arrays = staticmethod(lambda pair, a, b: (t[:840], o[:840], h[:840], l[:840], c[:840], c[:840]))
        fetch = staticmethod(lambda pair, a, b: pd.DataFrame(columns=["open_time", "o", "h", "l", "c", "vol"]))
        arrays_from = staticmethod(lambda d: _arrays_from(d))
    monkeypatch.setitem(S._MODS, "fbfr", type("RR", (), {"ET": ET, "fetch_funding": staticmethod(lambda p, a, b: (np.array([], "int64"), np.array([])))}))
    te = int(t[2]) + 12_000
    monkeypatch.setattr(S, "_PATHS", {})
    assert S.walk_fill("XUSDT", te, 100.0, now_ms=te + 74 * 3_600_000) is None          # not a day past the walk's end → wait
    monkeypatch.setattr(S, "_PATHS", {})
    assert S.walk_fill("XUSDT", te, 100.0, now_ms=te + 100 * 3_600_000) == "NO_DATA"    # a day later, still cut → out
