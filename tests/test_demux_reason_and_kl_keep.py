"""DECISION_LOG 224 (2026-10-06): (1) the pattern-cell de-mux reason is stamped on the order (cell_demux_reason) so a 2× cell
sized down to 1× is visible in the CSV; (2) DECISION_LOG 230: get_ohlcv reads > 1000 bars from the raw klines endpoint (newer ccxt caps
fetch_ohlcv at 1000), the FRENZY 5m cache keeps FRENZY_KL_LIMIT = 1500 bars (the studies' window) and a truncated full read keeps the cache."""
import os
from services.frenzy import merge_klines, BAR_MS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows(t0, n):
    return [[t0 + i * BAR_MS, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(n)]


def test_cached_window_stays_at_full_read_length():
    full = _rows(0, 1500)                              # a full read now returns the requested 1500 (raw klines, DECISION_LOG 230)
    keep = 1500                                        # FRENZY_KL_LIMIT
    cache = full
    for k in range(12):                                # 12 incremental passes between full reads
        tail = _rows(cache[-1][0], 5) if k == 0 else _rows(cache[-1][0] - 3 * BAR_MS, 5)
        cache = merge_klines(cache, tail, keep)
        assert cache is not None and len(cache) == 1500
    short = _rows(0, 1000)                             # the old ccxt-capped read with keep 1500 grew past the full read
    assert len(merge_klines(short, _rows(short[-1][0], 5), 1500)) == 1004


def test_engine_wiring():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert "keep=FRENZY_KL_LIMIT)" in eng and "FRENZY_KL_LIMIT = 1500" in eng and "[FRENZY_KL_SHORT]" in eng
    bs = open(os.path.join(ROOT, "services", "binance_service.py")).read()
    assert "fapiPublicGetKlines" in bs and "OHLCV_CCXT_CAP = 1000" in bs
    assert "merge_klines(_c5['rows'], _full[-FRENZY_KL_TAIL:], _c5['keep'])" in eng
    for tag in ("C1_DEMUX_BREADTH", "UNMATCHED_SPRINT_DEMUX", "UNMATCHED_DEMUX_PVR"):
        assert f'_cell_demux_reason = "{tag}"' in eng
    assert "cell_demux_reason=(_cell_demux_reason if (_pcell_src is not None and cell_src == _pcell_src) else None)" in eng
    assert "cell_demux_reason = Column(String(32)" in open(os.path.join(ROOT, "models.py")).read()
    assert "ADD COLUMN cell_demux_reason VARCHAR(32)" in open(os.path.join(ROOT, "database.py")).read()


def test_get_ohlcv_raw_path(monkeypatch):
    import asyncio
    from services import binance_service as B
    svc = B.binance_service
    calls = {}

    class Ex:
        def market(self, sym): return {'id': 'RLCUSDT'}
        async def fapiPublicGetKlines(self, p): calls['raw'] = p; return [['1', '1.5', '2', '1', '1.8', '100']]
        async def fetch_ohlcv(self, sym, tf, limit=None): calls['ccxt'] = limit; return [[1, 1.0, 1.0, 1.0, 1.0, 1.0]]

    async def _noop(*a, **k): return None
    monkeypatch.setattr(svc, "public_exchange", Ex(), raising=False)
    monkeypatch.setattr(svc, "_check_ban", _noop, raising=False)
    monkeypatch.setattr(svc, "load_public_markets", _noop, raising=False)
    rows = asyncio.run(svc.get_ohlcv('RLC/USDT:USDT', '5m', 1500))
    assert calls['raw'] == {'symbol': 'RLCUSDT', 'interval': '5m', 'limit': 1500} and rows == [[1, 1.5, 2.0, 1.0, 1.8, 100.0]]
    assert isinstance(rows[0][0], int)
    asyncio.run(svc.get_ohlcv('RLC/USDT:USDT', '5m', 1000))
    assert calls['ccxt'] == 1000                       # ≤ 1000 stays on ccxt
