"""📖📐 (DECISION_LOG 584) impact-capped depth sizing + 📖📊 capacity stamps — pure walk math, the engine guard in both modes (REAL open
path, paper), the live capped-limit price, paper fill at the walked VWAP, capacity stamps on the FLATBASE / EXHAUSTION open paths and journals,
config / UI / exports parity, the scout gate + capacity table."""
import datetime as dt
import json
import pathlib
import sys
import time

import pandas as pd
import pytest

from services import entry_depth as D

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASKS = [[100.0, 50.0], [100.05, 30.0], [100.10, 20.0], [100.20, 1000.0]]   # $5,000 · $3,001.5 · $2,002 · $100,200
BIDS = [[100.0, 50.0], [99.95, 30.0], [99.90, 20.0], [99.80, 1000.0]]


# ── ① pure math ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_walk_exact_vwap():
    assert D.walk_vwap_pct(ASKS, "LONG", 5000) == pytest.approx(0.0)
    n = 5000 + 3001.5                                                          # two full levels: qty 80
    assert D.walk_vwap_pct(ASKS, "LONG", n) == pytest.approx((n / 80 / 100 - 1) * 100)
    w, vwap, deep = D.walk_detail(ASKS, "LONG", 6000.0)                        # 50 @ 100 + $1,000 of the 100.05 level
    q = 50 + 1000 / 100.05
    assert vwap == pytest.approx(6000 / q) and deep == 100.05 and w == pytest.approx((6000 / q / 100 - 1) * 100)
    ws = D.walk_vwap_pct(BIDS, "SHORT", 5000 + 2998.5)
    assert ws == pytest.approx((1 - (5000 + 2998.5) / 80 / 100) * 100) and ws > 0
    assert D.walk_vwap_pct(ASKS, "LONG", 0) == 0.0
    assert D.walk_vwap_pct(ASKS, "LONG", 1e9) is None                          # shallower than the notional
    for bad in (None, [], [["x", 1]], [[0, 1]], [[-1, 5]]):
        assert D.walk_vwap_pct(bad, "LONG", 100) is None
    assert D.walk_detail(ASKS, "LONG", float("nan")) == (None, None, None)


def test_impact_fillable_at_the_cap_boundary():
    u, deep = D.impact_fillable_usd(ASKS, "LONG", 0.10)
    tgt = 100.0 * 1.001
    q_prev, c_prev = 100.0, 5000 + 3001.5 + 2002                               # levels 1–3 keep the average under 100.10
    x = (tgt * q_prev - c_prev) / (100.20 - tgt)
    assert u == pytest.approx(c_prev + 100.20 * x) and deep == 100.20
    assert D.walk_vwap_pct(ASKS, "LONG", u) == pytest.approx(0.10, abs=1e-9)  # exactly the cap
    assert D.walk_vwap_pct(ASKS, "LONG", u * 1.01) > 0.10
    us, ds = D.impact_fillable_usd(BIDS, "SHORT", 0.10)
    assert D.walk_vwap_pct(BIDS, "SHORT", us) == pytest.approx(0.10, abs=1e-9) and ds == 99.80
    assert D.impact_fillable_usd(ASKS, "LONG", 0.0) == (pytest.approx(5000.0), 100.0)   # a zero cap = the best level only
    # a thin book that runs out under the cap: everything it shows, flagged exhausted
    u2, d2, ex = D.impact_fillable_detail(ASKS[:2], "LONG", 0.10)
    assert u2 == pytest.approx(5000 + 3001.5) and d2 == 100.05 and ex is True
    assert D.impact_fillable_detail(ASKS, "LONG", 0.10)[2] is False
    for bad in (None, [], [["x", 1]], [[0, 1]]):
        assert D.impact_fillable_usd(bad, "LONG", 0.10) == (None, None)
    assert D.impact_fillable_usd(ASKS, "LONG", None) == (None, None) and D.impact_fillable_usd(ASKS, "LONG", -1) == (None, None)


def test_exit_side_and_thinner_binds():
    book = {'asks': ASKS, 'bids': BIDS[:2]}
    e, _ = D.impact_fillable_usd(book['asks'], "LONG", 0.10)
    x, _ = D.exit_side_impact_fillable_usd(book, "LONG", 0.10)               # a LONG exits into the bids
    assert x == pytest.approx(5000 + 2998.5) and D.thinner_side_usd(e, x) == x
    xs, _ = D.exit_side_impact_fillable_usd(book, "SHORT", 0.10)             # a SHORT exits into the asks
    assert xs == pytest.approx(e)
    assert D.exit_side_impact_fillable_usd(None, "LONG", 0.1) == (None, None) and D.exit_side_impact_fillable_usd("x", "LONG", 0.1) == (None, None)
    assert D.depth_guard_size(20000, D.thinner_side_usd(e, x), 0.25) == (D.SHRINK, pytest.approx(x))


def test_impact_limit_price_one_tick_toward_fill():
    assert D.impact_limit_price(100.20, "LONG", 0.01) == pytest.approx(100.21)
    assert D.impact_limit_price(99.80, "SHORT", 0.01) == pytest.approx(99.79)
    assert D.impact_limit_price(0.1264, "LONG", 0.0001) == pytest.approx(0.1265)
    assert D.impact_limit_price(1.0, "LONG", 0) == 1.0 and D.impact_limit_price(None, "LONG", 0.01) is None
    assert D.impact_limit_price(0.0001, "SHORT", 0.0001) == 0.0001                # never a non-positive price


def test_capacity_helpers():
    b = D.book_bands({'asks': ASKS, 'bids': BIDS}, "LONG")
    assert b['entry_book_usd_02'] == pytest.approx(5000 + 3001.5 + 2002 + 100200) and b['entry_book_usd_05'] == b['entry_book_usd_02']
    assert b['entry_book_exit_usd_02'] == pytest.approx(5000 + 2998.5 + 1998 + 99800)
    assert D.book_band_usd(ASKS, "LONG", 0.07) == pytest.approx(5000 + 3001.5)
    assert all(v is None for v in D.book_bands(None, "LONG").values()) and len(D.book_bands(None, "SHORT")) == 6
    assert D.binding_side(100, 200, 300) == "NONE" and D.binding_side(100, 50, 300) == "ENTRY" and D.binding_side(100, 300, 50) == "EXIT"
    assert D.binding_side(100, 50, 50) == "ENTRY" and D.binding_side(100, None, 50) == "EXIT" and D.binding_side(100, None, None) == "NONE"
    rows = D.capacity_rows([dict(sleeve="FLATBASE_SHORT", planned=100, final=50, binding="EXIT", walk=0.1, slip=0.2),
                            dict(sleeve="FLATBASE_SHORT", planned=300, final=300, binding="NONE", walk=0.03, slip=0.0),
                            dict(sleeve="MOMENTUM", planned=10, final=10, binding="NONE", walk=0.0, slip=0.0),
                            dict(sleeve="X", planned=None)], {"FLATBASE_SHORT": 2})
    assert [r['sleeve'] for r in rows] == ["EXHAUSTION_SHORT", "FLATBASE_SHORT", "DISTRIBUTION_SHORT", "MOMENTUM"]   # 📉 (586) a focus sleeve
    fb = rows[1]
    assert fb['n'] == 2 and fb['skipped'] == 2 and fb['skipped_pct'] == 50.0 and fb['shrunk_pct'] == 50.0 and fb['med_planned'] == 200
    assert (fb['bind_entry'], fb['bind_exit'], fb['bind_none']) == (0, 1, 1) and fb['med_walk'] == pytest.approx(0.065)
    assert rows[0]['n'] == 0 and rows[0]['shrunk_pct'] is None


# ── ② the REAL open path (paper, FLATBASE_SHORT / EXHAUSTION_SHORT harness) ──────────────────────────────────────────────────────────
CAPS = []


def _open(monkeypatch, bids, asks=None, mode="impact", sleeve="FLATBASE", exit_side=True, impact=0.10, min_frac=0.25):
    from tests import test_flatbase_short as T
    import services.trading_engine as TE
    _orig_off = T._offline

    def _off_with_book(mp, TE_):
        net = _orig_off(mp, TE_)

        async def book(*a, **k):
            if bids is None:
                return None
            ak = asks if asks is not None else [[bids[0][0] * 1.0001, 1e12]]
            return {'best_bid': bids[0][0], 'best_ask': ak[0][0], 'bid_qty': bids[0][1], 'ask_qty': ak[0][1], 'bids': bids, 'asks': ak,
                    'limit': k.get('limit')}
        mp.setattr(TE_.binance_service, "fetch_orderbook", book)
        inv = TE_.config.trading_config.investment
        for k_, v in (("entry_depth_guard_enabled", True), ("entry_depth_guard_max_sweep_pct", 0.15), ("entry_depth_guard_min_frac", min_frac),
                      ("entry_depth_guard_mode", mode), ("entry_depth_guard_max_avg_impact_pct", impact),
                      ("entry_depth_guard_exit_side_enabled", exit_side), ("entry_depth_guard_all_enabled", True)):
            mp.setattr(inv, k_, v, raising=False)
        return net
    monkeypatch.setattr(T, "_offline", _off_with_book)
    CAPS.clear()
    _orig_rec = TE.TradingEngine._depth_cap_record
    monkeypatch.setattr(TE.TradingEngine, "_depth_cap_record", staticmethod(lambda *a: (CAPS.append(_orig_rec(*a)), CAPS[-1])[1]))
    th = dict(flatbase_shares_bot_slots=False, flatbase_invest_mult=0.2, flatbase_lev_mult=0.25, momentum_long_invest_mult=0.2, momentum_long_lev_mult=0.25)
    if sleeve == "FLATBASE":
        kw = dict(flatbase_short=True, flatbase_bar_close_ms=int(time.time() * 1000), entry_frenzy_spike_at=dt.datetime(2026, 10, 22, 8, 0),
                  entry_flatbase_flat14_pct=31.5, entry_flatbase_flat_cov_pct=100.0, entry_flatbase_off_peak_pct=-22.0, entry_flatbase_peak_age_h=7.5)
        r, blocks, *_ = T._e2e_open(monkeypatch, n_open=0, th_kw=th, **kw)
    else:
        from tests import test_exhaustion_short as X
        r, blocks, *_ = X._e2e(monkeypatch, n_open=0, exhaustion_short=True, exhaustion_bar_close_ms=int(time.time() * 1000),
                               entry_frenzy_spike_at=dt.datetime(2026, 11, 17, 8, 0), entry_exh_tp_pct=3.6, entry_exh_sl_pct=1.8,
                               entry_exh_decision_price=1.0, entry_exh_q5_usd=2e6)
    return r, blocks


def _plan(monkeypatch, sleeve="FLATBASE"):
    r, _ = _open(monkeypatch, [[1.0, 1e12]], sleeve=sleeve)
    return r.notional_value


def _ladder(n_usd, step=0.0005, levels=40, frac=0.1):
    """a SHORT's entry side (bids): levels of n_usd × frac each, 0.05 % apart, best 1.0."""
    return [[round(1.0 * (1 - step * i), 8), n_usd * frac / (1.0 * (1 - step * i))] for i in range(levels)]


def test_sweep_mode_unchanged_and_stamped(monkeypatch):
    n = _plan(monkeypatch)
    r, blocks = _open(monkeypatch, [[1.0, n * 0.6], [0.9980, n * 10]], mode="sweep")   # DL 551 band rule: 0.6 × planned within 0.15 %
    assert r is not None and r.notional_value == pytest.approx(n * 0.6, rel=1e-6) and r.entry_depth_mode == "sweep"
    assert r.entry_price == pytest.approx(1.0)                                            # sweep: paper fill at the best price, as before
    assert r.entry_cap_planned_usd == pytest.approx(n, rel=1e-6) and r.entry_cap_final_usd == pytest.approx(r.notional_value, rel=1e-6)
    assert r.entry_cap_entry_side_usd == r.entry_depth_fillable_usd == pytest.approx(n * 0.6, rel=1e-6)
    assert r.entry_cap_binding_side == "ENTRY" and r.entry_cap_frac == pytest.approx(0.6, abs=1e-4)
    assert r.entry_book_usd_02 == pytest.approx(n * 0.6 + 0.998 * n * 10, rel=1e-6)            # bands from the same read (0.998 is inside 0.2 %)
    assert r.entry_book_exit_usd_1 is not None and r.entry_depth_walk_pct is not None


def test_impact_mode_sizes_to_the_average_walk_and_books_the_vwap(monkeypatch):
    n = _plan(monkeypatch)
    bids = _ladder(n)                                                                     # 10 % of the plan per level, 0.05 % apart
    r, blocks = _open(monkeypatch, bids, mode="impact", impact=0.10)
    u, _ = D.impact_fillable_usd(bids, "SHORT", 0.10)
    assert u < n and r is not None and r.entry_depth_mode == "impact"
    assert r.notional_value == pytest.approx(u, rel=1e-4) and r.entry_cap_entry_side_usd == pytest.approx(u, rel=1e-4)
    assert r.entry_depth_walk_pct == pytest.approx(0.10, abs=1e-4)                        # the modelled walk at the final notional = the cap
    _, vwap, _ = D.walk_detail(bids, "SHORT", r.notional_value)
    assert r.entry_price == pytest.approx(vwap, rel=1e-9) and r.entry_price < 1.0        # paper sells at the walked VWAP, not the best bid
    assert r.quantity == pytest.approx(r.notional_value / r.entry_price, rel=1e-9)
    assert r.entry_cap_binding_side == "ENTRY" and r.entry_cap_frac == pytest.approx(u / n, rel=1e-3)
    # the sweep band on the same book would have kept only ~4 levels (0.15 %) → impact keeps more
    assert u > D.depth_fillable_usd(bids, "SHORT", 0.15)
    assert CAPS and CAPS[-1]['cap_action'] == "SHRINK" and CAPS[-1]['cap_mode'] == "impact"


def test_impact_mode_thin_exit_side_binds_and_skip(monkeypatch):
    n = _plan(monkeypatch)
    deep = [[1.0, 1e12]]
    asks = [[1.0001, n * 0.4 / 1.0001], [1.01, 1e12]]                                      # the EXIT side (a SHORT buys the asks) holds ~0.4 × plan
    r, _ = _open(monkeypatch, deep, asks=asks, mode="impact")
    assert r is not None and r.entry_cap_binding_side == "EXIT" and r.notional_value == pytest.approx(r.entry_cap_exit_side_usd, rel=1e-6)
    assert r.entry_cap_exit_side_usd < n * 0.45
    asks2 = [[1.0001, n * 0.1 / 1.0001], [1.01, 1e12]]
    r2, blocks2 = _open(monkeypatch, deep, asks=asks2, mode="impact")                      # 0.1 × plan < min_frac 0.25 → skip
    assert r2 is None and "DEPTH_GUARD_SKIP" in blocks2
    assert CAPS[-1]['cap_action'] == "SKIP" and CAPS[-1]['cap_binding_side'] == "EXIT" and CAPS[-1]['cap_final_usd'] == 0.0


def test_impact_mode_unreadable_book_fail_open(monkeypatch):
    n = _plan(monkeypatch)
    r, blocks = _open(monkeypatch, None, mode="impact")
    assert r is not None and r.notional_value == pytest.approx(n, rel=1e-6) and r.entry_depth_walk_pct is None
    assert r.entry_cap_planned_usd is None and "DEPTH_GUARD_SKIP" not in blocks


def test_exhaustion_open_path_carries_the_capacity_stamps(monkeypatch):
    n = _plan(monkeypatch, sleeve="EXH")
    bids = _ladder(n)
    r, _ = _open(monkeypatch, bids, mode="impact", sleeve="EXH")
    assert r is not None and r.entry_strategy == "EXHAUSTION_SHORT" and r.entry_depth_mode == "impact"
    for c in ("entry_cap_planned_usd", "entry_cap_entry_side_usd", "entry_cap_exit_side_usd", "entry_cap_binding_side", "entry_cap_final_usd",
              "entry_cap_frac", "entry_depth_walk_pct", "entry_book_usd_02", "entry_book_usd_05", "entry_book_usd_1",
              "entry_book_exit_usd_02", "entry_book_exit_usd_05", "entry_book_exit_usd_1"):
        assert getattr(r, c) is not None, c
    assert r.entry_cap_final_usd < r.entry_cap_planned_usd and r.entry_price < 1.0


# ── ③ the live capped LIMIT price in impact mode + wiring ───────────────────────────────────────────────────────────────────────────
def test_engine_wiring():
    src = (ROOT / "services/trading_engine.py").read_text()
    assert "impact_cap_limit(_dg_elv, direction, _dg_deep, _cap_tick, _dg_imp_cap, _cap_px)" in src
    i_lim = src.index("if _dg_impact and _cap_px and _dg_deep and _dg_elv:")
    assert i_lim < src.index("elif _cap_px:\n                # 📖🧢 (DECISION_LOG 553) CAPPED ENTRY")   # re-priced before the capped order is sent
    assert "if _dg_impact and _dg_vwap and _dg_vwap > 0:" in src                                        # paper books the walked VWAP
    assert "entry_depth_walk_pct=_dg_walk, entry_depth_mode=_dg_mode_st" in src
    assert "self._depth_cap_last = None   # 📖📊 (584)" in src and src.count("self._depth_cap_for(pair)") == 6   # 📉 (586) + DISTRIBUTION opened / refused
    assert 'cap_action="NOFILL"' in src


def test_stamp_columns_classified():
    import models
    from services.trading_engine import TradingEngine
    cols = {c.name for c in models.Order.__table__.columns}
    new = ("entry_depth_walk_pct", "entry_depth_mode", "entry_cap_planned_usd", "entry_cap_entry_side_usd", "entry_cap_exit_side_usd",
           "entry_cap_binding_side", "entry_cap_final_usd", "entry_cap_frac", "entry_book_usd_02", "entry_book_usd_05", "entry_book_usd_1",
           "entry_book_exit_usd_02", "entry_book_exit_usd_05", "entry_book_exit_usd_1")
    for c in new:
        assert c in cols, c
        assert c in TradingEngine.STAMP_SLEEVE_ONLY or c.startswith(TradingEngine.STAMP_SLEEVE_PREFIXES)
    for t in (models.ExhaustionSignal, models.FlatbaseEpisode):
        tc = {c.name for c in t.__table__.columns}
        assert set(TradingEngine.DEPTH_CAP_JOURNAL_KEYS) <= tc
    db = (ROOT / "database.py").read_text()
    assert "entry_depth_walk_pct" in db and "entry_cap_planned_usd" in db and "for _t584 in ('exhaustion_signals', 'flatbase_episodes')" in db


def test_journals_carry_the_capacity_read(monkeypatch):
    import services.trading_engine as TE
    from tests import test_exhaustion_short as X
    rec = TE.TradingEngine._depth_cap_record(X._row(1)[0], "SKIP", "impact", 6000.0, 9000.0, 2500.0, "EXIT", 0.0, 0.0, None)

    def hook(e):
        _o = e.open_position

        async def op(**kw):
            r = await _o(**kw)
            e._depth_cap_last = dict(rec)
            return r
        e.open_position = op
    e, opens, sig, eps, calls = X._run(monkeypatch, k=1, open_ok=False, engine_hook=hook)
    assert sig[0].status == "REFUSED" and sig[0].cap_action == "SKIP" and sig[0].cap_binding_side == "EXIT" and sig[0].cap_planned_usd == 6000.0
    from tests import test_flatbase_short as F
    rows, on = F._series()
    monkeypatch.setattr(TE.TradingEngine, "_depth_cap_for", lambda self, pair: {k: v for k, v in rec.items() if k != 'pair'})
    e, led, opens, blocks = F._run_bars(monkeypatch, rows, on, [on + 72])
    assert led[0].status == "OPENED" and led[0].cap_action == "SKIP" and led[0].cap_exit_side_usd == 2500.0 and led[0].cap_mode == "impact"


# ── ④ config / UI / exports parity ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_config_defaults_json_ui_and_exports():
    import config
    inv = config.InvestmentConfig()
    assert inv.entry_depth_guard_mode == "sweep" and inv.entry_depth_guard_max_avg_impact_pct == 0.15
    assert config.InvestmentConfig(entry_depth_guard_mode="IMPACT").entry_depth_guard_mode == "impact"
    assert config.InvestmentConfig(entry_depth_guard_mode="bogus").entry_depth_guard_mode == "sweep"
    for bad in (0, -0.1, 1.5, "x", None):
        assert config.InvestmentConfig(entry_depth_guard_max_avg_impact_pct=bad).entry_depth_guard_max_avg_impact_pct == 0.15
    assert config.InvestmentConfig(entry_depth_guard_max_avg_impact_pct=1).entry_depth_guard_max_avg_impact_pct == 1.0
    j = json.loads((ROOT / "trading_config.json").read_text())["investment"]
    assert j["entry_depth_guard_mode"] == "impact" and j["entry_depth_guard_max_avg_impact_pct"] == 0.15
    html = (ROOT / "templates/index.html").read_text()
    for el in ("config-depth-guard-mode", "config-depth-guard-max-impact"):
        assert html.count(el) == 3                                                         # input + loader + save
    assert "entry_depth_guard_mode: (document.getElementById('config-depth-guard-mode')" in html
    assert "impact mode: avg walk ≤ " in html                                              # the shared ⑦ config line of both exports
    assert html.count("lines.push(...depthCapacityReportLines(perf, hr2));") == 2          # D12: both text exports
    assert html.count("depth-capacity-body") == 2 and html.count("${depthCapBadgeHtml(o)}") == 2   # UI table + the order-row badge
    m = (ROOT / "main.py").read_text()
    assert m.count('"depth_capacity_rows"') >= 2 and "DEPTH_CAP_ORDER_FIELDS" in m
    assert '["fb_" + _k for _k in TradingEngine.DEPTH_CAP_JOURNAL_KEYS]' in m and '["exh_" + _k for _k in TradingEngine.DEPTH_CAP_JOURNAL_KEYS]' in m


# ── ⑤ the scout ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _scout():
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_depth_impact_584 as S
    return S


def test_scout_gate_and_registry():
    S = _scout()
    import observe_registry as OR
    assert "scout_depth_impact_584.py" in OR.BOOKED and "import scout_depth_impact_584" in (ROOT / "scripts/opportunity_scout.py").read_text()
    assert "reports/SCOUT_DEPTH_IMPACT_584.json" in (ROOT / ".gitignore").read_text()
    assert S.gate([(0.1, 0.08)] * 19)[0] == "COLLECTING"
    assert S.gate([(0.1, 0.08)] * 20)[0] == "KEEP"
    assert S.gate([(0.14, 0.08)] * 20)[0] == "REVERT"                                   # realised − modelled +0.06 > +0.05
    assert S.gate([(0.16, 0.14)] * 20, cap=0.10)[0] == "REVERT"                          # mean slip 0.16 > cap + 0.05
    assert S.gate([(0.13, 0.08)] * 20, cap=0.10)[0] == "KEEP"
    assert S.gate([(None, 0.1)] * 25)[0] == "COLLECTING"
    assert S.is_live(False) and S.is_live("False") and not S.is_live(True) and not S.is_live(None) and S.is_live(True, "live")


def test_scout_report_freezes_and_prints_the_capacity_table(tmp_path):
    S = _scout()
    t0 = pd.Timestamp("2026-11-19 00:00")
    rows = []
    for i in range(22):
        rows.append(dict(opened_at=str(t0 + pd.Timedelta(minutes=i)), pair=f"P{i}USDT", direction="SHORT", status="CLOSED",
                         entry_strategy=("EXHAUSTION_SHORT" if i % 2 else "FLATBASE_SHORT"), notional_value=1000.0, entry_slippage_pct=0.2,
                         entry_depth_walk_pct=0.08, entry_depth_mode="impact", is_paper=False, trading_mode=None,
                         entry_cap_planned_usd=2000.0, entry_cap_final_usd=1000.0, entry_cap_binding_side="EXIT",
                         entry_fill_vs_book_pct=0.2, entry_order_type="TAKER", entry_cap_fill_frac=1.0))
    a = pd.DataFrame(rows)
    a["o"] = pd.to_datetime(a.opened_at)
    st = tmp_path / "s.json"
    txt = S.report(frame=a, since_ms=int(t0.timestamp() * 1000), state_path=str(st), cap=0.10, skips={"EXHAUSTION_SHORT": 3}, nofills=0)
    assert "FROZEN" in txt and "REVERT" in txt and json.loads(st.read_text())["verdict"] == "REVERT"
    assert "Capacity at entry — EXHAUSTION_SHORT · FLATBASE_SHORT" in txt
    line = [l for l in txt.splitlines() if l.startswith("EXHAUSTION_SHORT")][0]
    assert " 11 " in line and "$2,000" in line and "$1,000" in line and "3 (21.4)" in line and "0/11/0" in line


# ── ⑥ review fixes (C1 · I5 · I2 · I3 · I4 · M) ─────────────────────────────────────────────────────────────────────────────────────
GAP_ASKS = [[1.0, 10000.0], [1.05, 1000.0]]     # the deep review's gapped book: $10k at the best, the next level 5 % away


def test_gapped_book_marginal_bound_and_clamped_limit():
    u, deep = D.impact_fillable_usd(GAP_ASKS, "LONG", 0.15)
    assert deep == 1.05 and u > 10000                                              # the average alone would reach the far level
    assert D.book_band_usd(GAP_ASKS, "LONG", 0.45) == pytest.approx(10000)         # the 3 × cap marginal bound keeps it at the best level
    assert D.impact_cap_limit(GAP_ASKS, "LONG", 1.05, 0.0001, 0.15, 1.0015) == pytest.approx(1.0045)   # clamped (was 1.0501)
    assert D.impact_cap_limit(GAP_ASKS, "LONG", 1.0, 0.0001, 0.15, 1.0015) == pytest.approx(1.0015)    # never tighter than the sweep cap
    bids = [[1.0, 1e4], [0.95, 1e3]]
    assert D.impact_cap_limit(bids, "SHORT", 0.95, 0.0001, 0.15, 0.9985) == pytest.approx(0.9955)
    assert D.impact_cap_limit([[1.0, 1e9], [0.9990, 1e9]], "SHORT", 0.9990, 0.0001, 0.15, 0.9985) == pytest.approx(0.9985)   # sweep floor (−tick 0.9989 is tighter)
    assert D.impact_cap_limit([[1.0, 1e9], [0.9980, 1e9]], "SHORT", 0.9980, 0.0001, 0.15, 0.9985) == pytest.approx(0.9979)   # deepest − 1 tick
    assert D.impact_limit_price(0.12345678, "LONG", 1e-7) == pytest.approx(0.1234569) and len(repr(D.impact_limit_price(0.1264, "LONG", 0.0001))) <= 6


def test_impact_is_a_superset_of_sweep_at_the_default_cap():
    import config
    cap = config.InvestmentConfig().entry_depth_guard_max_avg_impact_pct
    assert cap == 0.15 == config.InvestmentConfig().entry_depth_guard_max_sweep_pct
    for asks in ([[100.0, 10.0], [100.14, 500.0], [100.30, 1e6]], ASKS, [[1.0, 1.0], [1.0014, 1e5], [1.0016, 1e5], [1.01, 1e9]]):
        sw = D.depth_fillable_usd(asks, "LONG", 0.15)
        im = D.impact_fillable_usd(asks, "LONG", cap)[0]
        assert sw is not None and im >= sw - 1e-6
        assert min(im, D.book_band_usd(asks, "LONG", 3 * cap)) >= sw - 1e-6           # the marginal bound keeps it (0.45 % ≥ 0.15 %)


def test_walk_float_dust_tolerance():
    lv = [[0.1, 1 / 3], [0.2, 1 / 3]]
    tot = 0.1 / 3 + 0.2 / 3
    assert D.walk_vwap_pct(lv, "LONG", tot) is not None and D.walk_vwap_pct(lv, "LONG", tot + 0.005) is not None
    assert D.walk_vwap_pct(lv, "LONG", tot + 1.0) is None


def test_impact_50_to_500_fallback_and_truncated(monkeypatch):
    n = _plan(monkeypatch)
    thin50 = [[round(1.0 - 1e-5 * i, 8), 1.0] for i in range(50)]                   # 50 tiny levels, all under the cap → exhausted, < plan
    r, blocks = _open(monkeypatch, thin50, mode="impact")
    assert r is None and "DEPTH_GUARD_TRUNCATED" in blocks and "DEPTH_GUARD_SKIP" in blocks   # the deeper read (stub: same book) still short
    deep50 = [[round(1.0 - 1e-5 * i, 8), n] for i in range(50)]                      # exhausted but already ≥ plan → no deeper read, not counted
    r2, blocks2 = _open(monkeypatch, deep50, mode="impact")
    assert r2 is not None and "DEPTH_GUARD_TRUNCATED" not in blocks2 and r2.entry_book_bands_levels == 50
    assert r2.entry_cap_binding_side == "NONE"


def test_engine_fail_open_writes_no_capacity_record(monkeypatch):
    r, _ = _open(monkeypatch, None, mode="impact")
    assert r is not None and CAPS == []


# the LIVE capped path in impact mode (fake exchange): the limit price and quantity sent
SENT = []


def _live_open(monkeypatch, book, direction="SHORT", mode="impact"):
    import models
    import services.trading_engine as TE
    from tests import test_flatbase_short as T
    import asyncio
    monkeypatch.setattr(TE.TradingEngine, "_willy_global_hold_on", lambda self: False)
    monkeypatch.setattr(TE.TradingEngine, "_rtp_tick_cached", lambda self, pair, px: 0.0001)
    net = T._offline(monkeypatch, TE)
    T._patch_th(monkeypatch, TE, flatbase_shares_bot_slots=False, flatbase_invest_mult=0.2, flatbase_lev_mult=0.25)
    inv = TE.config.trading_config.investment
    for k_, v in (("entry_depth_guard_enabled", True), ("entry_depth_cap_enabled", True), ("entry_depth_cap_wait_s", 0.0),
                  ("entry_depth_guard_max_sweep_pct", 0.15), ("entry_depth_guard_min_frac", 0.25), ("entry_depth_guard_mode", mode),
                  ("entry_depth_guard_max_avg_impact_pct", 0.15), ("entry_depth_guard_exit_side_enabled", False), ("max_open_positions", 4),
                  ("redeploy_leftover_enabled", False)):
        monkeypatch.setattr(inv, k_, v, raising=False)
    bs = TE.binance_service

    async def ob(*a, **k):
        return book

    async def lev(sym, l):
        return int(l)

    async def tick(*a, **k):
        return 0.0001

    async def capped(symbol, side, amount, price, wait_s=0.0, status=None):
        SENT.append(dict(side=side, amount=amount, price=price))
        return {'id': None, 'amount': 0.0, 'requested': amount, 'price': price, 'immediate': 0.0, 'wait_s': 0.0, 'unconfirmed': False}
    for n_, f_ in (("fetch_orderbook", ob), ("set_leverage", lev), ("get_tick_size", tick), ("create_capped_entry", capped)):
        monkeypatch.setattr(bs, n_, f_, raising=False)
    SENT.clear()
    CAPS.clear()
    _orig_rec = TE.TradingEngine._depth_cap_record
    monkeypatch.setattr(TE.TradingEngine, "_depth_cap_record", staticmethod(lambda *a: (CAPS.append(_orig_rec(*a)), CAPS[-1])[1]))

    async def go():
        from tests.test_frenzy_willy import _hold_db
        eng, SL = await _hold_db(monkeypatch)()
        e = TE.TradingEngine(); e.is_running = True; e.is_paper_mode = False
        blocks = []
        e._record_filter_block = lambda n, d, had_room=True: blocks.append(n)

        async def bal(db):
            return 5000.0
        e.get_available_balance = bal
        async with SL() as db:
            kw = (dict(flatbase_short=True, flatbase_bar_close_ms=int(time.time() * 1000), entry_frenzy_spike_at=dt.datetime(2026, 10, 22, 8, 0),
                       entry_flatbase_flat14_pct=31.5, entry_flatbase_flat_cov_pct=100.0, entry_flatbase_off_peak_pct=-22.0,
                       entry_flatbase_peak_age_h=7.5) if direction == "SHORT" else dict(surge_dir="LONG"))   # LONG: a SURGE long (same book path)
            r = await e.open_position(db=db, pair=T.PAIR, direction=direction, confidence="STRONG_BUY", current_price=1.0,
                                      entry_pair_volume_24h_usd=5e9, **kw)
        await eng.dispose()
        return r, blocks, e
    out = asyncio.run(go())
    assert net == []
    return out


def _book2(entry, direction):
    far = [[1.0001 if direction == "SHORT" else 0.9999, 1e12]]
    return ({'best_bid': entry[0][0], 'best_ask': far[0][0], 'bids': entry, 'asks': far} if direction == "SHORT"
            else {'best_bid': far[0][0], 'best_ask': entry[0][0], 'bids': far, 'asks': entry})


@pytest.mark.parametrize("direction", ["SHORT", "LONG"])
def test_live_capped_limit_in_impact_mode(monkeypatch, direction):
    sg = -1 if direction == "SHORT" else 1
    r0, b0, _ = _live_open(monkeypatch, _book2([[1.0, 1e12]], direction), direction)
    assert SENT, "the capped order was not sent"
    n = SENT[0]['amount'] * 1.0                                                     # plan at price 1.0 (deep book → full size)
    assert SENT[0]['price'] == pytest.approx(1.0 + sg * 0.0015)                       # deep book: the walk stays at the best → the sweep floor
    assert r0 is None and "DEPTH_CAP_NOFILL" in b0 and CAPS[-1]['cap_action'] == "FULL"
    # a ladder: 10 % of the plan per level, 0.05 % apart → the walk reaches past the sweep band, limit = deepest + 1 tick
    lad = [[round(1.0 + sg * 0.0005 * i, 8), n * 0.1] for i in range(40)]
    r1, b1, e1 = _live_open(monkeypatch, _book2(lad, direction), direction)
    u, _ = D.impact_fillable_usd(lad, direction, 0.15)
    _, _, deep = D.walk_detail(lad, direction, u)
    assert SENT[0]['price'] == pytest.approx(round(deep + sg * 0.0001, 4)) and abs(SENT[0]['price'] - 1.0) > 0.0015
    assert SENT[0]['amount'] == pytest.approx(u, rel=2e-3)                              # the shrunk quantity at ~1.0
    assert e1._depth_cap_last['cap_action'] == "NOFILL" and e1._depth_cap_last['cap_final_usd'] == 0.0
    # the gapped book: $ at the best then a level 5 % away — the limit never reaches it
    gap = [[1.0, n * 0.9], [round(1.0 + sg * 0.05, 4), n * 10]]
    _live_open(monkeypatch, _book2(gap, direction), direction)
    assert SENT[0]['price'] == pytest.approx(1.0 + sg * 0.0015)                        # deepest = best (+1 tick) → the sweep floor
    assert SENT[0]['amount'] == pytest.approx(n * 0.9, rel=2e-3)                       # sized to the $ inside the marginal bound
    _live_open(monkeypatch, _book2(gap, direction), direction, mode="sweep")
    assert SENT[0]['price'] == pytest.approx(1.0 + sg * 0.0015)                        # sweep mode: the 553 ceiling as before


def test_exhaustion_gate_entry_leg_after_walk():
    from services import exhaustion as X
    assert X.entry_leg_slip(0.3, "impact", 0.1) == pytest.approx(0.2) and X.entry_leg_slip(0.1, "impact", 0.3) == 0.0
    assert X.entry_leg_slip(0.3, "sweep", 0.1) == 0.3 and X.entry_leg_slip(0.3, None, None) == 0.3 and X.entry_leg_slip(None, "impact", 0.1) is None
    assert X.gate_net_pct(1.0, "STOP_LOSS", 0.3, 0.2) == pytest.approx(0.5) and X.gate_net_pct(1.0, "STOP_LOSS", 0.3) == pytest.approx(0.4)
    t = [dt.datetime(2026, 11, 19, 0, i) for i in range(20)]
    v4, _ = X.exhaustion_gate([(a, 0.3, "STOP_LOSS", 0.2) for a in t])                  # 0.3 − 0.2 − 0.2 < 0 → OFF
    v5, _ = X.exhaustion_gate([(a, 0.3, "STOP_LOSS", 0.2, 0.0) for a in t])             # impact rows: entry leg paid → +0.1 → KEEP
    assert (v4, v5) == ("OFF", "KEEP")
    src = (ROOT / "scripts/scout_exhaustion_short.py").read_text() + (ROOT / "main.py").read_text()
    assert "entry_leg_slip(r.slip, r.entry_depth_mode, r.entry_depth_walk_pct)" in src and "_exels(s_, dm_, wk_)" in src


def test_band_583_slip_legs_live_only_and_mode_split():
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_depth_band_583 as B
    t0 = pd.Timestamp("2026-11-19 00:00")
    rows = [dict(opened_at=str(t0 + pd.Timedelta(minutes=i)), pair=f"P{i}", direction="SHORT", status="CLOSED", entry_strategy="EXHAUSTION_SHORT",
                 pnl=1.0, pnl_percentage=0.1, close_reason="TP", notional_value=30.0, entry_depth_shrunk_from=100.0,
                 entry_slippage_pct=0.5, exit_slippage_pct=0.5, is_paper=True, entry_depth_mode=("impact" if i % 2 else None)) for i in range(10)]
    a = pd.DataFrame(rows)
    a["o"] = pd.to_datetime(a.opened_at)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        txt = B.report(frame=a, since_ms=int(t0.timestamp() * 1000), state_path=d + "/s.json")
    assert "KEEP" in txt and "by depth mode: " in txt and "impact 5" in txt and "pre-584 5" in txt   # paper 0.5 % slips never flag


def test_scout_584_reads_fill_vs_book_takers_only(tmp_path):
    S = _scout()
    t0 = pd.Timestamp("2026-11-19 00:00")
    base = dict(direction="SHORT", status="CLOSED", entry_strategy="EXHAUSTION_SHORT", notional_value=1000.0, entry_slippage_pct=0.9,
                entry_depth_walk_pct=0.08, entry_depth_mode="impact", is_paper=False, trading_mode=None, entry_cap_planned_usd=2000.0,
                entry_cap_final_usd=1000.0, entry_cap_binding_side="EXIT", entry_fill_vs_book_pct=0.09, entry_cap_fill_frac=1.0)
    rows = [dict(base, opened_at=str(t0 + pd.Timedelta(minutes=i)), pair=f"P{i}", entry_order_type=("MAKER" if i < 5 else ("TAKER+MAKER" if i % 2 else "TAKER"))) for i in range(26)]
    a = pd.DataFrame(rows)
    a["o"] = pd.to_datetime(a.opened_at)
    txt = S.report(frame=a, since_ms=int(t0.timestamp() * 1000), state_path=str(tmp_path / "s.json"), cap=0.15, skips={}, nofills=2)
    assert "KEEP" in txt and "19/20" not in txt                  # decision slip 0.9 % is NOT the ruler; fill vs book 0.09 vs walk 0.08 → keep
    assert "NO-FILL 2" in txt and "fill share mean 1.00" in txt
    a2 = a.assign(entry_order_type="MAKER")
    txt2 = S.report(frame=a2, since_ms=int(t0.timestamp() * 1000), state_path=str(tmp_path / "s2.json"), cap=0.15, skips={}, nofills=0)
    assert "COLLECTING" in txt2                                  # maker fills never enter the gate


def test_perf_capacity_rows_skips_and_zero_fills(db):
    import asyncio
    import main
    import models
    from types import SimpleNamespace as NS

    async def go():
        now = dt.datetime.utcnow()
        for i, (sl, ln, ago) in enumerate((("EXHAUSTION_SHORT", "DEPTH_GUARD", 1), ("EXHAUSTION_SHORT", "DEPTH_GUARD", 2),
                                           ("FLATBASE_SHORT", "WILLY_GAP", 1), ("EXHAUSTION_SHORT", "DEPTH_GUARD", 100))):
            db.add(models.BlockedSignal(pair=f"P{i}USDT", sleeve=sl, direction="SHORT", at=now - dt.timedelta(hours=ago),
                                        bar_at=now - dt.timedelta(hours=ago), price=1.0, lines=ln, sole=True, entry_fields_json="{}", is_paper=True))
        await db.commit()
        o = NS(entry_cap_planned_usd=100.0, entry_strategy="FLATBASE_SHORT", entry_cap_final_usd=50.0, entry_cap_binding_side="EXIT",
               entry_depth_walk_pct=0.1, entry_slippage_pct=0.12)
        r_all = await main._depth_capacity_rows(db, [o], None, True)
        r_win = await main._depth_capacity_rows(db, [], now - dt.timedelta(hours=10), True)
        return r_all, r_win
    r_all, r_win = asyncio.get_event_loop().run_until_complete(go()) if False else asyncio.run(go())
    ex = {r['sleeve']: r for r in r_all}
    assert ex['EXHAUSTION_SHORT']['n'] == 0 and ex['EXHAUSTION_SHORT']['skipped'] == 3 and ex['EXHAUSTION_SHORT']['skipped_pct'] == 100.0
    assert ex['FLATBASE_SHORT']['n'] == 1 and ex['FLATBASE_SHORT']['skipped'] == 0 and ex['FLATBASE_SHORT']['shrunk_pct'] == 100.0
    w = {r['sleeve']: r for r in r_win}
    assert w['EXHAUSTION_SHORT']['skipped'] == 2 and w['FLATBASE_SHORT']['n'] == 0     # the window start, not the first fill
