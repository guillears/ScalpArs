"""📖 Order-book research stamps (DECISION_LOG 192): metrics, manual-click stamps, the minute recorder, BOOK rows in the Decisions CSV."""
import asyncio
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.orderbook_stats import OB_FIELDS, orderbook_metrics, snapshot_pairs  # noqa: E402

BOOK = {"bids": [[100.0, 10], [99.8, 50], [99.0, 5]], "asks": [[100.2, 10], [100.5, 1], [101.5, 300]]}


def test_metrics_on_a_known_book():
    m = orderbook_metrics(BOOK["bids"], BOOK["asks"])
    close = lambda x, y: abs(x - y) <= 1e-5 * max(1.0, abs(y))          # values are kept to 6 significant figures
    assert close(m["mid"], 100.1) and close(m["spread_pct"], 0.2 / 100.1 * 100)
    assert close(m["bid_usd_05"], 100 * 10 + 99.8 * 50) and close(m["ask_usd_05"], 100.2 * 10 + 100.5 * 1)
    assert close(m["imb_05"], (5990 - 1102.5) / (5990 + 1102.5))
    assert close(m["wall_ask_usd"], 101.5 * 300) and close(m["wall_ask_dist_pct"], (101.5 / 100.1 - 1) * 100)
    assert m["imb_2"] is None                                              # the bids reach only ~1.1 % → no 2 % reading
    assert set(OB_FIELDS) <= set(m) and all(v is None or math.isfinite(v) for v in m.values())


def test_bands_beyond_the_returned_depth_are_blank_and_3_element_entries_work():
    bids = [[100.0, 5, 3], [99.9, 5, 1]]; asks = [[100.1, 5, 2], [100.2, 5, 1]]          # reaches ~0.15 % on each side
    m = orderbook_metrics(bids, asks)
    assert m["imb_025"] is None and m["imb_2"] is None and m["bid_usd_1"] is None and 0.1 < m["reach_bid_pct"] < 0.2
    assert m["spread_pct"] is not None and m["wall_bid_usd"] is not None


def test_random_books_match_a_reference():
    import random
    rnd = random.Random(7)
    for _ in range(300):
        mid = 10 ** rnd.uniform(-4, 4); n = rnd.randint(5, 400); step = mid * rnd.uniform(1e-4, 2e-3)
        bids = [[mid - step * (i + 1), rnd.uniform(0.1, 100)] for i in range(n)]; asks = [[mid + step * (i + 1), rnd.uniform(0.1, 100)] for i in range(n)]
        m = orderbook_metrics(bids, asks); md = (bids[0][0] + asks[0][0]) / 2
        for band, k in ((0.5, "05"), (2.0, "2")):
            if m[f"imb_{k}"] is None:
                continue
            bu = sum(p * q for p, q in bids if p >= md * (1 - band / 100)); au = sum(p * q for p, q in asks if p <= md * (1 + band / 100))
            assert abs(m[f"imb_{k}"] - (bu - au) / (bu + au)) < 1e-4


def test_bad_books_are_none_never_raise():
    assert orderbook_metrics([], BOOK["asks"]) is None and orderbook_metrics(BOOK["bids"], None) is None
    assert orderbook_metrics([[101, 1]], [[100, 1]]) is None                     # crossed
    assert orderbook_metrics([["x", 1]], [[100, 1]]) is None


def test_snapshot_pairs_manual_first_dedup_cap():
    assert snapshot_pairs(["A", "B", "C"], ["C", "D"], cap=3) == ["C", "D", "A"]
    assert snapshot_pairs([], [], cap=12) == [] and snapshot_pairs(None, None) == []


def test_model_migration_and_wiring():
    import models
    cols = {c.name for c in models.Order.__table__.columns}
    assert {f"manual_ob_{k}" for k in OB_FIELDS} <= cols
    snap = {c.name for c in models.OrderbookSnap.__table__.columns}
    assert {"pair", "at", "reason", "mid"} | set(OB_FIELDS) <= snap and models.OrderbookSnap.__tablename__ == "orderbook_snaps"
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert 'ALTER TABLE orders ADD COLUMN manual_ob_{_obc} FLOAT' in db
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "async def orderbook_loop():" in main and "_orderbook_task = asyncio.create_task(orderbook_loop())" in main
    assert 'e="BOOK"' in main and '_cols = list(_dj.EXPORT_COLS) + _ob_cols' in main      # rides the Decisions CSV
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "orderbook" not in ui.lower()                                       # operator: no new download button
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert 'asyncio.wait_for(binance_service.fetch_orderbook_depth(symbol, 500), 3.0)' in eng       # bounded, never blocks the open


def test_manual_open_stamps_the_book(monkeypatch):
    import services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = True

    async def _bal(db): return 10_000.0
    async def _bnb(db): return 1_000.0
    eng.get_available_balance = _bal; eng._recalculate_paper_bnb = _bnb

    class _Trk: last_price = 100.1
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: _Trk())
    monkeypatch.setattr(T.websocket_tracker, "pair_silence_seconds", lambda p: 1.0)

    async def _stamps(pair, symbol, direction, price): return {"entry_atr_pct": 1.0}
    eng._manual_entry_stamps = _stamps

    async def _brk(): return {}
    monkeypatch.setattr(T.binance_service, "get_leverage_brackets", _brk)
    book = {"v": BOOK}

    async def _depth(symbol, limit=500): return book["v"]
    monkeypatch.setattr(T.binance_service, "fetch_orderbook_depth", _depth)

    class _Accepted(Exception): pass
    captured = []

    class _Res:
        def scalar(self): return 0
        def first(self): return None
        def scalar_one_or_none(self): return None

    class _DB:
        async def execute(self, *a, **k): return _Res()
        def add(self, o): captured.append(o); raise _Accepted()

    def attempt():
        try:
            asyncio.run(TE.open_manual_position(eng, _DB(), pair="SANDUSDT", direction="LONG", investment=100, leverage=20, exit_mode="FIXED", sl_pct=2.0))
        except _Accepted:
            return captured[-1]

    o = attempt()
    assert abs(o.manual_ob_spread_pct - 0.2 / 100.1 * 100) < 1e-5 and o.manual_ob_imb_05 is not None and abs(o.manual_ob_wall_ask_usd - 101.5 * 300) < 1
    book["v"] = None
    o = attempt()
    assert o.manual_ob_spread_pct is None and o.manual_ob_imb_05 is None             # unreadable book: the open still happens
