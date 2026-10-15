"""📏 (DECISION_LOG 295) execution-quality shadow — observe only. Entry: where the entry cost comes from (drift / spread / our fill vs
the book / lag). Exit: what a resting post-only take-profit limit would have booked. Pure rules + wiring pins (the shadow never changes
an order and never raises into the open / close paths)."""
import inspect, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.exec_shadow import entry_book_metrics, resting_tp_price, net_pct_at, resting_tp_shadow, cached_tick

MK = 0.00018


def test_entry_book_metrics_signs_and_values():
    # LONG: decision 100, book 100.10 / 100.20, filled 100.30, read 7.5 s after the decision clock
    m = entry_book_metrics("LONG", 100.0, 100.10, 100.20, 100.30, 7.5)
    assert m == {"entry_book_drift_pct": 0.2, "entry_book_spread_pct": round(0.10 / 100.15 * 100, 4),
                 "entry_fill_vs_book_pct": round((100.30 / 100.20 - 1) * 100, 4), "entry_order_lag_s": 7.5}
    # SHORT: the far side is the BID; selling lower than the decision = a cost (positive)
    s = entry_book_metrics("SHORT", 100.0, 99.80, 99.90, 99.70, 1.0)
    assert s["entry_book_drift_pct"] == 0.2 and s["entry_fill_vs_book_pct"] == round((1 - 99.70 / 99.80) * 100, 4)
    # a better fill than the book / a book that moved our way → negative
    assert entry_book_metrics("LONG", 100.0, 99.8, 99.9, 99.85, 0)["entry_book_drift_pct"] == -0.1
    # unusable input never raises
    assert entry_book_metrics("LONG", None, None, None, None, None) == {"entry_book_drift_pct": None, "entry_book_spread_pct": None,
                                                                          "entry_fill_vs_book_pct": None, "entry_order_lag_s": None}
    assert entry_book_metrics("LONG", 100, 101, 100, 100, -3)["entry_book_spread_pct"] is None   # crossed book → no spread
    assert entry_book_metrics("LONG", 100, 101, 100, 100, -3)["entry_order_lag_s"] is None        # negative lag → unknown


def test_resting_tp_price_books_the_same_net_target_with_a_maker_fee():
    for direction in ("LONG", "SHORT"):
        for e, tick, ef, tp in ((0.0178089, 1e-7, 0.045, 1.25), (104.05, 0.01, 0.045, 3.0), (0.10733, 1e-5, 0.045, 1.25)):
            lim = resting_tp_price(direction, e, ef, tp, MK, tick)
            net = net_pct_at(direction, e, lim, ef, MK)
            assert net >= tp - 1e-6                                     # never books less than the target
            one_tick_back = lim - tick if direction == "LONG" else lim + tick
            assert net_pct_at(direction, e, one_tick_back, ef, MK) < tp  # …and is the closest tick that does
    assert resting_tp_price("LONG", 0, 0.045, 1.25, MK, 0.01) is None
    assert resting_tp_price("LONG", 1, 0.045, 0, MK, 0.01) is None
    assert resting_tp_price("LONG", 1, None, 1.25, MK, 0.01) is None


def test_net_pct_matches_the_engine_convention():
    # engine: pnl_percentage = (raw − entry_fee − exit_fee) / entry notional × 100, exit fee on the exit notional
    e, x, q = 0.0178089, 0.0180238, 139736.0
    notional, entry_fee = e * q, e * q * 0.00045
    exit_fee = x * q * 0.00045
    engine = ((x - e) * q - entry_fee - exit_fee) / notional * 100
    assert abs(net_pct_at("LONG", e, x, entry_fee / notional * 100, 0.00045) - engine) < 1e-4


def test_resting_tp_shadow_needs_a_trade_through():
    e, ef, tick = 0.0178089, 1.1198450026799998 / 2488.5444503999997 * 100, 1e-7      # live ZK 2026-10-15
    lim = resting_tp_price("LONG", e, ef, 1.25, MK, tick)
    # live ZK: high 0.018048 went through the limit → +1.25 instead of the +1.116 the market exit booked
    l2, filled, pnl = resting_tp_shadow("LONG", e, ef, 1.25, MK, tick, 0.018048, None, 1.1162)
    assert l2 == lim and filled is True and abs(pnl - 1.25) < 0.01
    # a touch (high == limit) is NOT a fill → the trade keeps its actual result
    assert resting_tp_shadow("LONG", e, ef, 1.25, MK, tick, lim, None, -3.79)[1:] == (False, -3.79)
    assert resting_tp_shadow("LONG", e, ef, 1.25, MK, tick, lim + tick, None, -3.79)[1] is True
    # SHORT mirror uses the LOW
    ls = resting_tp_price("SHORT", 100.0, 0.045, 1.25, MK, 0.01)
    assert resting_tp_shadow("SHORT", 100.0, 0.045, 1.25, MK, 0.01, None, ls - 0.01, -1.0)[1] is True
    assert resting_tp_shadow("SHORT", 100.0, 0.045, 1.25, MK, 0.01, None, ls, -1.0)[1] is False
    # missing extreme → unknown, never a guess; bad input → all None
    assert resting_tp_shadow("LONG", e, ef, 1.25, MK, tick, None, None, 1.0) == (lim, None, None)
    assert resting_tp_shadow("LONG", None, ef, 1.25, MK, tick, 1, 1, 1.0) == (None, None, None)


def test_model_columns_migration_and_csv():
    import models, database
    cols = models.Order.__table__.columns
    for c in ("entry_book_drift_pct", "entry_book_spread_pct", "entry_fill_vs_book_pct", "entry_order_lag_s",
              "shadow_rtp_price", "shadow_rtp_filled", "shadow_rtp_pnl_pct"):
        assert c in cols and cols[c].nullable                       # rides the orders CSV via column introspection
        assert f"ALTER TABLE orders ADD COLUMN {c} " in inspect.getsource(database)


def test_engine_wiring_is_observe_only():
    import services.trading_engine as T
    src = inspect.getsource(T.TradingEngine.open_position)
    assert "_eb_m = entry_book_metrics(direction, current_price, _eb_bid, _eb_ask, actual_price, _eb_lag)" in src
    assert "entry_book_drift_pct=_eb_m.get('entry_book_drift_pct')" in src
    assert "_eb_bid = _eb_ask = _eb_lag = None" in src             # defined on every path before use
    csrc = inspect.getsource(T.TradingEngine._close_position_locked)
    # the tick lookup runs BEFORE the DB retry loop (no HTTP inside the loop), the shadow inside it is wrapped
    assert csrc.index("_rtp_ctx = None") < csrc.index("for _db_attempt in range(1, _max_db_retries + 1):") < csrc.index("resting_tp_shadow(")
    ctx = csrc[csrc.index("_rtp_ctx = None"):csrc.index("for _db_attempt in range(1, _max_db_retries + 1):")]
    import re
    assert not re.search(r"\bawait\s", ctx) and "cached_tick(" in ctx                # the close holds the global close lock: no await for a shadow
    assert "frenzy_lock_arm_pct" in ctx                                 # lock-then-trail replaces the fixed TP → no shadow for it
    assert "order.shadow_rtp_price, order.shadow_rtp_filled, order.shadow_rtp_pnl_pct = _rtp_lim, _rtp_fill, _rtp_pnl" in csrc


def test_cached_tick_reads_the_loaded_cache_and_rejects_implausible_ticks():
    mk = {"ZK/USDT:USDT": {"precision": {"price": 1e-07}}, "BTC/USDT:USDT": {"precision": {"price": 0.1}},
          "OLD/USDT:USDT": {"precision": {"price": 4}}, "BAD/USDT:USDT": {"precision": {}}}
    assert cached_tick(mk, "ZK/USDT:USDT", 0.0178) == 1e-07
    assert cached_tick(mk, "BTC/USDT:USDT", 60000) == 0.1
    assert cached_tick(mk, "OLD/USDT:USDT", 5.0) == 1e-4                 # integer precision = decimals
    assert cached_tick(mk, "BAD/USDT:USDT", 1.0) is None                 # no precision → unknown, never the 0.01 default
    assert cached_tick({"X/USDT:USDT": {"precision": {"price": 0.01}}}, "X/USDT:USDT", 0.02) is None   # 50 %/tick → implausible
    assert cached_tick(None, "ZK/USDT:USDT", 1) is None and cached_tick(mk, "NONE/USDT:USDT", 1) is None
