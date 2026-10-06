"""DECISION_LOG 241 (2026-10-06): an intentional reconnect (new pair / healing / prune) skips the 1 s error backoff — at most one such
skip per FAST_RECONNECT_MIN_GAP_S — and the close waits at most CLOSE_TIMEOUT = 1 s (was 5 s). Real errors keep the backoff."""
import asyncio
import services.websocket_tracker as WT


class _FakeWS:
    """recv() blocks until close() is called (like a live stream), then raises ConnectionClosed; or fails at once with `error`."""

    def __init__(self, error=False):
        self.error = error
        self._closed = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def recv(self):
        if self.error:
            raise WT.ConnectionClosed(None, None)      # a network drop, not our close
        await self._closed.wait()
        raise WT.ConnectionClosed(None, None)

    async def close(self):
        self._closed.set()


def _run(monkeypatch, plan, actions, max_connects):
    """plan: list of 'ok' / 'err' sockets per connect; actions: coroutine(t) run as a separate task (the engine / scan side)."""
    t = WT.WebSocketTracker()
    t.subscribed_pairs = {"AAAUSDT"}
    connects, sleeps = [], []

    def fake_connect(url, **kw):
        connects.append(kw.get("close_timeout"))
        if len(connects) >= max_connects:
            t.running = False
        return _FakeWS(error=plan[min(len(connects) - 1, len(plan) - 1)] == "err")

    real_sleep = asyncio.sleep

    async def fake_sleep(d):
        sleeps.append(d)
        await real_sleep(0)

    monkeypatch.setattr(WT.websockets, "connect", fake_connect)
    monkeypatch.setattr(WT, "_sleep", fake_sleep)

    async def go():
        t.running = True
        runner = asyncio.create_task(t._run_forever())
        await actions(t)
        await asyncio.wait_for(runner, 5)
    asyncio.run(go())
    return t, connects, sleeps


async def _wait_ws(t):
    for _ in range(200):
        if t.websocket is not None:
            return
        await asyncio.sleep(0)
    raise AssertionError("no socket")


def test_intentional_reconnect_skips_backoff_errors_keep_it(monkeypatch):
    async def actions(t):
        await _wait_ws(t)
        await t._reconnect()                 # from another task, like the engine: intentional → no backoff
    t, connects, sleeps = _run(monkeypatch, ["ok", "err", "ok"], actions, 3)
    assert connects[0] == WT.WebSocketTracker.CLOSE_TIMEOUT == 1.0
    assert sleeps == [0, 1]                   # intentional = immediate; the following network error still backs off
    assert t._intentional_reconnect is False


def test_second_intentional_within_gap_backs_off(monkeypatch):
    async def actions(t):
        await _wait_ws(t)
        await t._reconnect()
        await _wait_ws(t)
        await t._reconnect()                 # a second intentional close within FAST_RECONNECT_MIN_GAP_S
    t, connects, sleeps = _run(monkeypatch, ["ok", "ok", "ok"], actions, 3)
    assert sleeps == [0, 1]                   # rate-limited: no fast reconnect loop


def test_reconnect_never_clears_a_newer_socket():
    async def go():
        t = WT.WebSocketTracker()
        old, new = _FakeWS(), _FakeWS()
        t.websocket = old

        async def slow_close():
            t.websocket = new                 # the run loop opened the next socket while our close was awaited
        old.close = slow_close
        await t._reconnect()
        assert t.websocket is new             # not wiped
    asyncio.run(go())
