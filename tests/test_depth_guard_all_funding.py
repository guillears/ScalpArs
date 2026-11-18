"""📖↔ (DECISION_LOG 565) depth guard on EVERY bot entry, sized to the THINNER side of the book · 💸 (566) funding guard ·
👁 (567) FAN_BTC_EMA13_DEEP observe line."""
import json
import pathlib
import sys

import pytest

from services import entry_depth as ED
from services.funding_guard import funding_guard_block, funding_against

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── pure rules ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_exit_side_and_thinner():
    book = {'bids': [[1.0, 1000.0], [0.997, 1e9]], 'asks': [[1.001, 200.0], [1.004, 1e9]]}
    assert ED.exit_side_fillable_usd(book, "LONG", 0.15) == pytest.approx(1000.0)        # a long exits into the bids
    assert ED.exit_side_fillable_usd(book, "SHORT", 0.15) == pytest.approx(200.2)        # a short exits buying the asks
    assert ED.exit_side_fillable_usd(None, "LONG", 0.15) is None
    assert ED.thinner_side_usd(5000.0, 300.0) == 300.0 and ED.thinner_side_usd(None, 300.0) == 300.0
    assert ED.thinner_side_usd(5000.0, None) == 5000.0 and ED.thinner_side_usd(None, None) is None


@pytest.mark.parametrize("d,rate,mins,block", [
    ("SHORT", -0.00183, 1.6, True),     # live INIT 11-16: −0.183 % hourly, 96 s before the payment
    ("SHORT", -0.00183, 16, False),     # outside 15 min
    ("SHORT", -0.0004, 5, False),       # 0.04 % < 0.05 %
    ("SHORT", 0.00183, 5, False),       # a short RECEIVES a positive rate
    ("LONG", 0.0006, 5, True), ("LONG", -0.0006, 5, False),
    ("SHORT", -0.0005, 15, True),       # both cuts inclusive
])
def test_funding_rule(d, rate, mins, block):
    now = 1_700_000_000_000
    assert funding_guard_block(d, rate, now + mins * 60000, now, 0.05, 15)[0] is block


def test_funding_unreadable_never_blocks():
    now = 1_700_000_000_000
    for args in ((None, now + 60000), (-0.002, None), (float("nan"), now + 60000), (-0.002, now - 60000)):
        assert funding_guard_block("SHORT", args[0], args[1], now, 0.05, 15)[0] is False
    assert funding_guard_block("SHORT", -0.002, now + 60000, now, 0, 15)[0] is False     # cut 0 = off
    assert funding_against("SHORT", -0.001) and not funding_against("LONG", -0.001)


# ── wiring (D11) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_config_ui_wiring():
    import config
    inv, th = config.InvestmentConfig(), config.SignalThresholds()
    assert inv.entry_depth_guard_all_enabled is False and inv.entry_depth_guard_exit_side_enabled is False
    assert th.funding_guard_enabled is False and th.funding_guard_rate_min_pct == 0.05 and th.funding_guard_minutes == 15.0
    j = json.loads((ROOT / "trading_config.json").read_text())
    assert j["investment"]["entry_depth_guard_all_enabled"] is True and j["investment"]["entry_depth_guard_exit_side_enabled"] is True
    assert j["thresholds"]["funding_guard_enabled"] is True and j["thresholds"]["funding_guard_rate_min_pct"] == 0.05
    assert j["thresholds"]["funding_guard_minutes"] == 15.0
    html = (ROOT / "templates/index.html").read_text()
    for el in ("config-depth-guard-all", "config-depth-guard-exit-side", "config-funding-guard-enabled", "config-funding-guard-rate",
               "config-funding-guard-minutes"):
        assert html.count(el) == 3, el                                                  # input + loader + save
    assert "💸 funding guard ${" in html and "sized to the thinner side" in html          # the shared export line


def test_engine_wiring_and_stamp():
    import models
    from services.trading_engine import TradingEngine
    src = (ROOT / "services/trading_engine.py").read_text()
    assert "funding_guard_block(direction, _fg_rate, _fg_next" in src and 'self._record_filter_block("FUNDING_GUARD", direction)' in src
    i_f = src.index("💸 (DECISION_LOG 566) FUNDING GUARD")
    i_g = src.index("📖 (DECISION_LOG 551 → 565) ENTRY DEPTH GUARD")
    i_o = src.index("        if not self.is_paper_mode:\n            symbol = pair.replace('USDT', '/USDT:USDT')\n            side = ", i_g)
    assert i_f < i_g < i_o                                                               # both guards run before any order
    assert "if not _surge and _dg_on2 and _dg_all and notional_value > 0 and leverage > 0:" in src
    assert "thinner_side_usd(_dg_fill, _dg_xfill)" in src and "entry_depth_exit_fillable_usd=_dg_xfill" in src
    assert "entry_depth_exit_fillable_usd" in {c.name for c in models.Order.__table__.columns}
    assert "entry_depth_exit_fillable_usd" in TradingEngine.STAMP_SLEEVE_ONLY
    assert "fetch_funding_next" in (ROOT / "services/binance_service.py").read_text()


# ── the REAL open_position (paper, offline) ─────────────────────────────────────────────────────────────────────────────────────
MODE = ["sweep"]


def _run(monkeypatch, book=None, funding=None, direction="LONG", **kw):
    from tests import test_willy_global_hold_switch as W
    _orig = W._offline

    def _off(mp, TE):
        net = _orig(mp, TE)

        async def ob(*a, **k):
            return book

        async def fn(*a, **k):
            return funding
        mp.setattr(TE.binance_service, "fetch_orderbook", ob)
        mp.setattr(TE.binance_service, "fetch_funding_next", fn)
        inv = TE.config.trading_config.investment
        for k_, v in (("entry_depth_guard_enabled", True), ("entry_depth_guard_all_enabled", True), ("entry_depth_guard_exit_side_enabled", True),
                      ("entry_depth_guard_max_sweep_pct", 0.15), ("entry_depth_guard_min_frac", 0.5),
                      ("entry_depth_guard_mode", MODE[0])):   # 📖📐 (584) sweep unless a test asks
            mp.setattr(inv, k_, v, raising=False)
        mp.setattr(TE.config.trading_config.thresholds, "funding_guard_enabled", True, raising=False)
        return net
    monkeypatch.setattr(W, "_offline", _off)
    import services.trading_engine as TE
    th = TE.config.trading_config.thresholds
    for k_ in ("watch_block_ml_mkt_hot", "watch_block_ml_bull90", "watch_block_ml_weak_rescue", "watch_block_ml_weak_market",
               "watch_block_ml_fading_quiet", "watch_block_ms_regime_ends"):
        monkeypatch.setattr(th, k_, False, raising=False)
    r, blocks, rows, opened = W._e2e(monkeypatch, "FOOUSDT", direction=direction, entry_bull_pct=50.0, **kw)
    return r, blocks


def _book(entry_usd, exit_usd, direction="LONG"):
    """a book at price 1.0 whose ENTRY side holds entry_usd and EXIT side exit_usd within 0.15 %, then deep beyond it."""
    near_e, near_x = [[1.0, entry_usd], [1.01, 1e12]], [[1.0, exit_usd], [0.99, 1e12]]
    if direction == "LONG":
        return {'best_bid': 1.0, 'best_ask': 1.0, 'asks': near_e, 'bids': near_x}
    return {'best_bid': 1.0, 'best_ask': 1.0, 'bids': [[1.0, entry_usd], [0.99, 1e12]], 'asks': [[1.0, exit_usd], [1.01, 1e12]]}


def test_momentum_long_unreadable_book_opens_unchanged(monkeypatch):
    r, blocks = _run(monkeypatch, book=None)
    assert r is not None and r.entry_depth_shrunk_from is None and r.entry_depth_exit_fillable_usd is None
    assert "DEPTH_GUARD_SKIP" not in blocks and "FUNDING_GUARD" not in blocks


def test_momentum_long_thin_exit_side_shrinks_or_skips(monkeypatch):
    r0, _ = _run(monkeypatch, book=None)
    n = r0.notional_value
    r, blocks = _run(monkeypatch, book=_book(n * 100, n * 0.6))                  # entry side deep, EXIT side 0.6 × planned → shrink to it
    assert r is not None and r.entry_depth_shrunk_from == pytest.approx(n, rel=1e-6)
    assert r.notional_value == pytest.approx(n * 0.6, rel=1e-3) and r.entry_depth_exit_fillable_usd == pytest.approx(n * 0.6, rel=1e-3)
    r2, blocks2 = _run(monkeypatch, book=_book(n * 100, n * 0.3))                # exit side < half → skip
    assert r2 is None and "DEPTH_GUARD_SKIP" in blocks2
    r3, _ = _run(monkeypatch, book=_book(n * 100, n * 100))                       # both deep → full size, both sides stamped
    assert r3 is not None and r3.entry_depth_shrunk_from is None and r3.entry_depth_exit_fillable_usd == pytest.approx(n * 100, rel=1e-3)


def test_funding_guard_blocks_and_passes(monkeypatch):
    import time
    soon = int(time.time() * 1000) + 90_000
    r, blocks = _run(monkeypatch, funding=(0.0018, soon))                         # a LONG pays +0.18 % in 1.5 min → skipped
    assert r is None and "FUNDING_GUARD" in blocks
    r2, blocks2 = _run(monkeypatch, funding=(-0.0018, soon))                      # a LONG receives a negative rate → opens
    assert r2 is not None and "FUNDING_GUARD" not in blocks2


def test_fan_btc_ema13_observe_rule():
    sys.path.insert(0, str(ROOT / "scripts"))
    import observe_registry as OR
    import scout_fan_btc_ema13_watch as S
    assert S.CUT == -0.16 and S.tagged(-0.1657) and S.tagged(-0.16) and not S.tagged(-0.0834) and not S.tagged(None) and not S.tagged("x")
    assert (S.N_MIN, S.DAYS_MIN, S.P_MIN) == (15, 8, 0.95)
    assert "scout_fan_btc_ema13_watch.py" in OR.BOOKED
    import pandas as pd
    c = pd.DataFrame(dict(pct=[-0.5] * 14, tag=[True] * 14, day=[f"d{i}" for i in range(14)]))
    assert S.verdict(c)[0] == "COLLECTING"
