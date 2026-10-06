"""DECISION_LOG 240 (2026-10-06): opens (manual and bot) no longer wait for the WebSocket reconnect that adds a new pair (~5 s live).
The subscription runs as a background task; a failure is logged, never raised into the open."""
import asyncio, os
import services.trading_engine as T
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_open_paths_do_not_await_the_reconnect():
    src = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert src.count("ws_subscribe_background(pair)") == 2
    assert "await websocket_tracker.subscribe_pair(pair, actual_price)" not in src


def test_background_subscribe_returns_at_once_and_logs_failures(monkeypatch, caplog):
    calls = []

    async def slow_ok(pair, price=None):
        await asyncio.sleep(0.2); calls.append(pair)

    async def boom(pair, price=None):
        raise RuntimeError("ws down")

    async def run():
        monkeypatch.setattr(T.websocket_tracker, "subscribe_pair", slow_ok)
        loop = asyncio.get_running_loop(); t0 = loop.time()
        t = T.ws_subscribe_background("AAAUSDT")
        assert loop.time() - t0 < 0.05                 # the caller is not held
        await t; await asyncio.sleep(0)
        assert calls == ["AAAUSDT"] and t not in T._WS_SUB_TASKS
        monkeypatch.setattr(T.websocket_tracker, "subscribe_pair", boom)
        await T.ws_subscribe_background("BBBUSDT")  # must not raise
    asyncio.run(run())
    assert "background subscribe of BBBUSDT failed" in caplog.text


def test_real_tracker_keeps_the_live_price():
    """force_reset_tracking seeds the entry price; the background subscribe must not overwrite a newer tick (no price passed)."""
    from services.websocket_tracker import WebSocketTracker as W
    async def run():
        w = W()
        w.force_reset_tracking("CCCUSDT", 1.00)
        w.trackers["CCCUSDT"].update(1.05)          # a live tick arrives before the background task runs
        await w.subscribe_pair("CCCUSDT")   # what the task does (websocket None → no reconnect)
        assert "CCCUSDT" in w.subscribed_pairs and w.trackers["CCCUSDT"].last_price == 1.05
    asyncio.run(run())
