"""📖 (DECISION_LOG 551) entry depth guard — pure sizing math + the engine / config / UI wiring."""
import json
import pathlib

import pytest

from services import entry_depth as D

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASKS = [[100.0, 50.0], [100.05, 30.0], [100.10, 20.0], [100.20, 1000.0]]   # $5,000 · $3,001.5 · $2,002 · beyond 0.15 %
BIDS = [[100.0, 50.0], [99.95, 30.0], [99.90, 20.0], [99.80, 1000.0]]


def test_fillable_long_and_short():
    assert D.depth_fillable_usd(ASKS, "LONG", 0.15) == pytest.approx(5000 + 3001.5 + 2002)
    assert D.depth_fillable_usd(BIDS, "SHORT", 0.15) == pytest.approx(5000 + 2998.5 + 1998)
    assert D.depth_fillable_usd(ASKS, "LONG", 0.0) == pytest.approx(5000)            # best level only
    assert D.depth_fillable_usd(ASKS, "LONG", 0.25) is None                         # every level inside the limit: truncated read → unmeasured
    assert D.depth_fillable_usd(ASKS[:3], "LONG", 0.15) is None and D.depth_fillable_usd([[100.0, 1.0]], "LONG", 0.15) is None
    assert D.depth_fillable_usd([], "LONG", 0.15) is None and D.depth_fillable_usd(None, "LONG", 0.15) is None
    assert D.depth_fillable_usd([["x", 1]], "LONG", 0.15) is None and D.depth_fillable_usd([[0, 1]], "LONG", 0.15) is None


def test_guard_actions_only_shrink_or_skip():
    assert D.depth_guard_size(8000, 10003.5, 0.5) == (D.FULL, 8000)
    assert D.depth_guard_size(16575, 10003.5, 0.5) == (D.SHRINK, 10003.5)
    assert D.depth_guard_size(30000, 10003.5, 0.5) == (D.SKIP, 10003.5)
    assert D.depth_guard_size(16575, None, 0.5) == (D.FULL, 16575)                    # unreadable book → fail-open
    assert D.depth_guard_size(20000, 10000, 0.5) == (D.SHRINK, 10000)                 # exactly the floor still trades
    for n, f in ((100, 50), (100, 99.99), (100, 1e9), (5, 0)):
        assert D.depth_guard_size(n, f, 0.5)[1] <= n


def test_engine_wiring():
    src = (ROOT / "services/trading_engine.py").read_text()
    assert "limit=(50 if _dg_on else 5)" in src                                     # the one book read carries the levels
    assert 'self._record_filter_block("DEPTH_GUARD_SKIP", direction)' in src
    assert 'self._blocked_record(_wh_es, direction, pair, current_price, ["DEPTH_GUARD"], _dg_ef)' in src
    assert "entry_depth_fillable_usd=_dg_fill, entry_depth_shrunk_from=_dg_from" in src
    i_guard = src.index("📖 (DECISION_LOG 551 → 565) ENTRY DEPTH GUARD")
    i_order = src.index("result = await binance_service.create_market_order(", i_guard)   # the open path's taker order (other paths use it too)
    assert src.index("notional_value = investment * leverage\n        _req_qty_snapshot = None") < i_guard < i_order   # sized after notional, before the order
    b = (ROOT / "services/binance_service.py").read_text()
    assert "'bids': ob['bids'], 'asks': ob['asks']" in b


def test_config_ui_wiring():
    import config
    inv = config.InvestmentConfig() if hasattr(config, "InvestmentConfig") else None
    j = json.loads((ROOT / "trading_config.json").read_text())["investment"]
    assert j["entry_depth_guard_enabled"] is True and j["entry_depth_guard_max_sweep_pct"] == 0.15 and j["entry_depth_guard_min_frac"] == 0.25   # DL 583
    if inv is not None:
        assert inv.entry_depth_guard_enabled is False and inv.entry_depth_guard_max_sweep_pct == 0.15 and inv.entry_depth_guard_min_frac == 0.5
    html = (ROOT / "templates/index.html").read_text()
    for el in ("config-depth-guard-enabled", "config-depth-guard-max-sweep", "config-depth-guard-min-frac"):
        assert html.count(el) == 3                                                    # input + loader + save
    assert "⑦ depth guard" in html


def test_stamp_columns_classified():
    import models
    from services.trading_engine import TradingEngine
    cols = {c.name for c in models.Order.__table__.columns}
    for c in ("entry_depth_fillable_usd", "entry_depth_shrunk_from"):
        assert c in cols and c in TradingEngine.STAMP_SLEEVE_ONLY


LIMITS = []
MODE = ["sweep"]


def _run_flatbase(monkeypatch, bids):
    """the REAL open_position (paper, offline harness of test_flatbase_short) with a stubbed book: a FLATBASE_SHORT sells into `bids`."""
    import datetime as dt
    import time
    from tests import test_flatbase_short as T
    _orig_off = T._offline

    def _off_with_book(mp, TE):
        net = _orig_off(mp, TE)

        async def book(*a, **k):
            LIMITS.append(k.get('limit'))
            if bids is None:
                return None
            return {'best_bid': bids[0][0], 'best_ask': bids[0][0] * 1.0001, 'bid_qty': bids[0][1], 'ask_qty': 1e9,
                    'bids': bids, 'asks': [[bids[0][0] * 1.0001, 1e9]], 'limit': k.get('limit')}
        mp.setattr(TE.binance_service, "fetch_orderbook", book)
        mp.setattr(TE.config.trading_config.investment, "entry_depth_guard_enabled", True, raising=False)
        mp.setattr(TE.config.trading_config.investment, "entry_depth_guard_max_sweep_pct", 0.15, raising=False)
        mp.setattr(TE.config.trading_config.investment, "entry_depth_guard_min_frac", 0.5, raising=False)
        mp.setattr(TE.config.trading_config.investment, "entry_depth_guard_mode", MODE[0], raising=False)   # 📖📐 (584) the 551 band rule unless a test asks
        return net
    monkeypatch.setattr(T, "_offline", _off_with_book)
    r, blocks, *_ = T._e2e_open(monkeypatch, n_open=0, th_kw=dict(flatbase_shares_bot_slots=False, flatbase_invest_mult=0.2, flatbase_lev_mult=0.25,
                                                                  momentum_long_invest_mult=0.2, momentum_long_lev_mult=0.25),
                                flatbase_short=True, flatbase_bar_close_ms=int(time.time() * 1000),
                                entry_frenzy_spike_at=dt.datetime(2026, 10, 22, 8, 0), entry_flatbase_flat14_pct=31.5,
                                entry_flatbase_flat_cov_pct=100.0, entry_flatbase_off_peak_pct=-22.0, entry_flatbase_peak_age_h=7.5)
    return r, blocks


def test_e2e_full_shrink_skip_and_unreadable(monkeypatch):
    deep = [[1.0, 1e9], [0.9980, 1.0]]
    r, blocks = _run_flatbase(monkeypatch, deep)
    assert r is not None and r.entry_depth_shrunk_from is None and r.entry_depth_fillable_usd == pytest.approx(1e9)
    n = r.notional_value
    lev, inv = r.leverage, r.investment
    half_plus = [[1.0, n * 0.6], [0.9980, n * 10]]                      # 0.6 × planned within 0.15 %, the rest beyond it
    r2, blocks2 = _run_flatbase(monkeypatch, half_plus)
    assert r2 is not None and r2.entry_depth_shrunk_from == pytest.approx(n, rel=1e-6)
    assert r2.notional_value == pytest.approx(n * 0.6, rel=1e-6) and r2.leverage == lev and r2.investment == pytest.approx(inv * 0.6, rel=1e-6)
    assert r2.quantity == pytest.approx(r2.notional_value / r2.entry_price, rel=1e-6)
    r3, blocks3 = _run_flatbase(monkeypatch, [[1.0, n * 0.3], [0.9980, n * 10]])
    assert r3 is None and "DEPTH_GUARD_SKIP" in blocks3
    r4, blocks4 = _run_flatbase(monkeypatch, None)                        # no book → unchanged, never blocked
    assert r4 is not None and r4.notional_value == pytest.approx(n, rel=1e-6) and r4.entry_depth_fillable_usd is None
    assert "DEPTH_GUARD_SKIP" not in blocks + blocks2 + blocks4
    assert LIMITS and set(LIMITS) == {50}                                    # the one book read asks for 50 levels (same weight as 5)
    trunc = [[1.0 + 0.0001 * i, 1.0] for i in range(50)]                  # 50 levels all inside 0.15 %: unmeasured → the deeper read (stubbed: same book) → full + counted
    r5, blocks5 = _run_flatbase(monkeypatch, trunc)
    assert r5 is not None and r5.notional_value == pytest.approx(n, rel=1e-6) and "DEPTH_GUARD_TRUNCATED" in blocks5 and 500 in LIMITS


def test_skip_refusal_text_reaches_flatbase_and_frenzy():
    src = (ROOT / "services/trading_engine.py").read_text()
    assert "if _frenzy or _fbs or _exh:   # (551 deep review) FLATBASE too" in src   # 🪫 (DECISION_LOG 578) + EXHAUSTION
    assert "quantity = notional_value / ((_sg_live_px if _surge else _dg_bpx) or current_price)" in src   # (565 review) the book price the depth was measured at


import pytest as _pt


@_pt.mark.parametrize("plan, fill, frac, act, n", [(1000, 300, 0.25, "SHRINK", 300), (1000, 300, 0.5, "SKIP", 300), (1000, 200, 0.25, "SKIP", 200),
                                                   (1000, 1200, 0.25, "FULL", 1000), (1000, 250, 0.25, "SHRINK", 250),
                                                   (6200, 2758, 0.25, "SHRINK", 2758), (6094, 2843, 0.25, "SHRINK", 2843),      # 11-18 ZIL / ZKP
                                                   (6200, 2758, 0.5, "SKIP", 2758), (6094, 2843, 0.5, "SKIP", 2843)])
def test_dl583_quarter_book_shrinks(plan, fill, frac, act, n):
    """📖 (DECISION_LOG 583) a book holding 25–50 % of the plan now SHRINKS (min_frac 0.25) where it used to SKIP (0.5); the boundary shrinks."""
    from services import entry_depth as ED
    a, got = ED.depth_guard_size(plan, fill, frac)
    assert a == getattr(ED, act) and got == n



def test_dl583_band_scout(tmp_path):
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts"))
    import scout_depth_band_583 as B
    import pandas as pd
    assert B.in_band(2758, 6200) and not B.in_band(3500, 6200) and not B.in_band(2758, None) and not B.in_band(None, 6200)
    assert B.gate([(1.0, 0.05, 0.05)] * 9)[0] == "COLLECTING"
    assert B.gate([(5.0, 0.05, 0.08)] * 10)[0] == "KEEP"
    assert B.gate([(-5.0, 0.05, 0.08)] * 10)[0] == "REVERT"
    assert B.gate([(5.0, 0.05, 0.30)] * 10)[0] == "REVERT"                                  # thin exit side slips → flag
    o = pd.Timestamp("2026-11-20")
    fr = pd.DataFrame([dict(opened_at=str(o + pd.Timedelta(hours=i)), pair=f"P{i}USDT", direction="SHORT", status="CLOSED",
                            entry_strategy="EXHAUSTION_SHORT", pnl=10.0, pnl_percentage=1.0, close_reason="FRENZY_TP",
                            notional_value=2700.0, entry_depth_shrunk_from=6100.0, entry_slippage_pct=0.05, exit_slippage_pct=0.0,
                            is_paper=True, o=o + pd.Timedelta(hours=i)) for i in range(12)])
    st = tmp_path / "s.json"
    txt = B.report(frame=fr, since_ms=int(pd.Timestamp("2026-11-19").value // 1_000_000), state_path=str(st), now_ms=0)
    assert "FROZEN" in txt and "KEEP" in txt
    import json as _j
    assert _j.load(open(st))["verdict"] == "KEEP"
