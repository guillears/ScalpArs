"""DECISION_LOG 245 (2026-10-07): Top-pairs display price — one cached all-symbols price read (main._live_prices →
BinanceService.fetch_all_prices on its own client), display only."""
import asyncio
import pytest
import main as M
from services import binance_service as BS


class _Src:
    def __init__(self, px=None, exc=None, delay=0.0):
        self.px, self.exc, self.delay, self.calls = px, exc, delay, 0

    async def __call__(self, *a, **k):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.px


@pytest.fixture
def setup(monkeypatch):
    def _go(src, now=1000.0):
        M._live_px.update(t=0.0, px={})
        M._live_px["try"] = 0.0
        monkeypatch.setattr(M, "_live_px_lock", asyncio.Lock())
        monkeypatch.setattr(M.binance_service, "fetch_all_prices", src)
        clock = {"t": now}
        monkeypatch.setattr(M.time, "time", lambda: clock["t"])
        return clock
    yield _go
    M._live_px.update(t=0.0, px={})
    M._live_px["try"] = 0.0


def test_reads_once_per_ttl(setup):
    src = _Src(px={"METUSDT": 0.3699})
    clock = setup(src)
    assert asyncio.run(M._live_prices()) == {"METUSDT": 0.3699} and src.calls == 1
    clock["t"] += M.LIVE_PX_TTL_S - 1
    asyncio.run(M._live_prices())
    assert src.calls == 1
    clock["t"] += 2
    asyncio.run(M._live_prices())
    assert src.calls == 2


def test_failure_keeps_last_map_then_expires_and_backs_off(setup):
    src = _Src(px={"METUSDT": 0.36})
    clock = setup(src)
    asyncio.run(M._live_prices())
    src.px = None                              # a failed / banned read
    clock["t"] += M.LIVE_PX_TTL_S + 0.1
    assert asyncio.run(M._live_prices()) == {"METUSDT": 0.36}
    calls = src.calls
    asyncio.run(M._live_prices())
    assert src.calls == calls                  # not retried inside the TTL
    clock["t"] += M.LIVE_PX_MAX_AGE_S + 1
    assert asyncio.run(M._live_prices()) == {}  # too old → the scan close


def test_timeout_keeps_last_map(setup, monkeypatch):
    src = _Src(px={"METUSDT": 0.36})
    clock = setup(src)
    asyncio.run(M._live_prices())
    real_wait_for = asyncio.wait_for

    async def fast_timeout(aw, t):
        aw.close()
        raise asyncio.TimeoutError()
    monkeypatch.setattr(M.asyncio, "wait_for", fast_timeout)
    clock["t"] += M.LIVE_PX_TTL_S + 0.1
    assert asyncio.run(M._live_prices()) == {"METUSDT": 0.36}
    monkeypatch.setattr(M.asyncio, "wait_for", real_wait_for)


def test_concurrent_refreshes_share_one_read(setup):
    src = _Src(px={"METUSDT": 0.36}, delay=0.01)   # suspends, so the other refreshes hit the lock / try stamp mid-read

    async def go():
        setup(src)
        return await asyncio.gather(*[M._live_prices() for _ in range(5)])
    out = asyncio.run(go())
    assert src.calls == 1
    assert out[0] == {"METUSDT": 0.36}


def test_fetch_all_prices_parses_drops_stale_and_records_ban(monkeypatch):
    svc = BS.binance_service
    _real = BS.BinanceService.fetch_all_prices   # the conftest stubs the instance attribute (no network) — exercise the real method
    now = 1_000_000.0

    class _Ex:
        async def fapiPublicGetTickerPrice(self):
            return [{"symbol": "METUSDT", "price": "0.3699", "time": str(int(now * 1000) - 1000)},
                    {"symbol": "OLDUSDT", "price": "1.0", "time": str(int(now * 1000) - 600_000)},
                    {"symbol": "BAD", "price": "x", "time": "0"}]
    monkeypatch.setattr(BS.time, "time", lambda: now)
    monkeypatch.setattr(svc, "display_exchange", _Ex())
    monkeypatch.setattr(BS, "_ban_until", 0)
    assert asyncio.run(_real(svc)) == {"METUSDT": 0.3699}

    seen = []
    class _Boom:
        async def fapiPublicGetTickerPrice(self):
            raise RuntimeError("418 banned until 1999999999999")
    monkeypatch.setattr(svc, "display_exchange", _Boom())
    monkeypatch.setattr(svc, "_detect_ban", lambda e: seen.append(str(e)))
    assert asyncio.run(_real(svc)) is None and seen
    monkeypatch.setattr(BS, "_ban_until", now + 60)
    monkeypatch.setattr(svc, "display_exchange", _Ex())
    assert asyncio.run(_real(svc)) is None   # banned → no call
