"""SSE event-stream endpoint for Live Terminal.

Each connected client gets:
  1. Replay of the current ring buffer (last 500 events) so the feed is
     populated immediately on view-switch.
  2. Streaming of new events as they arrive (polled at ~6Hz from the
     shared ring buffer).
  3. A `server_time` event every 1 second — the client uses inter-event
     gap variance as its latency-bar proxy. No backend exchange ping.
  4. A `: keepalive` comment every 15 seconds — drives the heartbeat dot.

A separate background asyncio task (start_heartbeat) injects a synthetic
[HEARTBEAT] event every 3 seconds containing the bot's current state
(regime, breadth, open positions, btc adx/rsi, etc.). It reads from
trading_engine module-level globals only — no function calls into
trading logic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from .handler import get_events_since, get_snapshot, push_event, _parse_message

logger = logging.getLogger("live_terminal.stream")

_HEARTBEAT_TASK: asyncio.Task | None = None
_HEARTBEAT_INTERVAL_SEC = 3.0
_STREAM_POLL_INTERVAL_SEC = 0.15
_KEEPALIVE_INTERVAL_SEC = 15.0
_SERVER_TIME_INTERVAL_SEC = 1.0
_SNAPSHOT_REPLAY_CAP = 200  # send last N on connect (not full 500 — first paint speed)


def _gather_bot_state() -> dict:
    """Read-only snapshot of bot state from existing trading_engine module
    globals. NEVER calls into trading_engine functions; only reads attributes.

    Imports are local to avoid module-import-order coupling with main.py.
    """
    state: dict = {
        "is_running": None,
        "is_paper": None,
        "open_positions": 0,
        "regime": None,
        "btc_adx": None,
        "btc_rsi": None,
        "btc_trend_gap_pct": None,
        "btc_1h_slope": None,
        "btc_atr_pct": None,
        "bull_pct": None,
        "bear_pct": None,
        # May 16: counts + global vol exposed for Live Terminal HEAT COMPASS panel.
        # Already-computed module globals — no new computation, just exposure.
        "breadth_n_bull": None,
        "breadth_n_bear": None,
        "breadth_n_total": None,
        "global_volume_ratio": None,
        "ban_active": False,
    }
    health = {"ws": True, "api": True, "log": True, "ex": True}

    try:
        from services.trading_engine import trading_engine, _open_orders_cache
        import services.trading_engine as te
        state["is_running"] = bool(getattr(trading_engine, "is_running", False))
        state["is_paper"] = bool(getattr(trading_engine, "is_paper_mode", True))
        state["open_positions"] = sum(
            len(orders) for orders in _open_orders_cache.values()
        )
        state["regime"] = getattr(te, "_current_btc_regime", None)
        state["btc_adx"] = getattr(te, "_current_btc_adx", None)
        state["btc_rsi"] = getattr(te, "_current_btc_rsi", None)
        state["btc_trend_gap_pct"] = getattr(te, "_current_btc_trend_gap_pct", None)
        state["btc_1h_slope"] = getattr(te, "_current_btc_1h_slope", None)
        state["bull_pct"] = getattr(te, "_market_bull_pct", None)
        state["bear_pct"] = getattr(te, "_market_bear_pct", None)
        state["breadth_n_bull"] = getattr(te, "_breadth_n_bull", None)
        state["breadth_n_bear"] = getattr(te, "_breadth_n_bear", None)
        state["breadth_n_total"] = getattr(te, "_breadth_n_total", None)
        state["global_volume_ratio"] = getattr(te, "_global_volume_ratio", None)
    except Exception as exc:
        logger.debug(f"[HEARTBEAT_STATE] read failed (non-fatal): {exc}")
        health["log"] = False

    try:
        from services.binance_service import get_ban_status
        ban_until = get_ban_status() or 0
        state["ban_active"] = bool(ban_until and ban_until > time.time())
        health["api"] = not state["ban_active"]
        health["ex"] = not state["ban_active"]
    except Exception:
        # If ban-status call fails, that itself is a yellow flag.
        health["api"] = False

    # WS health: services.websocket_tracker exposes a subscribed-count
    try:
        from services.websocket_tracker import websocket_tracker
        # Use whatever shape the tracker has; if subscribed_pairs attr exists, count it
        if hasattr(websocket_tracker, "subscribed_pairs"):
            health["ws"] = len(websocket_tracker.subscribed_pairs) > 0
        elif hasattr(websocket_tracker, "_trackers"):
            health["ws"] = len(websocket_tracker._trackers) > 0
    except Exception:
        # Don't penalise — assume green if we can't introspect
        pass

    return {"state": state, "health": health}


async def _heartbeat_loop() -> None:
    """Single background task that pushes [HEARTBEAT] every 3s into the
    ring buffer. All connected SSE streams see the same heartbeats
    automatically through their polling loop.
    """
    while True:
        try:
            payload = _gather_bot_state()
            # Build a human-readable message line so terminal feed shows
            # something sensible even without the structured payload.
            s = payload["state"]
            msg = (
                f"[HEARTBEAT] open={s['open_positions']} "
                f"regime={s['regime']} "
                f"btc_adx={s['btc_adx']} "
                f"btc_rsi={s['btc_rsi']} "
                f"running={s['is_running']} paper={s['is_paper']}"
            )
            parsed = _parse_message(msg)
            push_event("INFO", parsed, msg, extra={"hb": payload})
        except Exception as exc:
            logger.warning(f"[HEARTBEAT] failed (non-fatal): {exc}")
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SEC)


def start_heartbeat() -> None:
    """Start the background heartbeat task. Idempotent."""
    global _HEARTBEAT_TASK
    if _HEARTBEAT_TASK is not None and not _HEARTBEAT_TASK.done():
        return
    loop = asyncio.get_event_loop()
    _HEARTBEAT_TASK = loop.create_task(_heartbeat_loop())
    logger.info("[STARTUP] Live Terminal heartbeat task started")


def stop_heartbeat() -> None:
    """Cancel the heartbeat task on shutdown."""
    global _HEARTBEAT_TASK
    if _HEARTBEAT_TASK is not None and not _HEARTBEAT_TASK.done():
        _HEARTBEAT_TASK.cancel()
        _HEARTBEAT_TASK = None


def _sse_format(event: dict, event_name: str | None = None) -> str:
    """Format a dict as one SSE record. SSE-safe JSON (no embedded newlines)."""
    data_line = "data: " + json.dumps(event, separators=(",", ":")) + "\n"
    if event_name:
        return f"event: {event_name}\n{data_line}\n"
    return data_line + "\n"


async def event_stream_generator(request) -> "AsyncGenerator[str, None]":  # noqa: F821
    """The async generator passed to StreamingResponse.

    Lifecycle:
      - On connect: yield a `replay` event for each historical event in the
        ring buffer, in id order.
      - Then poll for new events (~6 Hz) and yield them as default-type
        SSE messages.
      - In parallel emit `server_time` events every 1s and `: keepalive`
        comments every 15s.
      - Exits cleanly when the client disconnects.
    """
    # Phase 1: snapshot replay (only the last N for quick first paint)
    snapshot = get_snapshot()
    if snapshot:
        replay_slice = snapshot[-_SNAPSHOT_REPLAY_CAP:]
        last_id = replay_slice[-1]["id"]
        for evt in replay_slice:
            yield _sse_format(evt, event_name="replay")
    else:
        last_id = 0

    # Phase 2: live tail
    next_keepalive = time.time() + _KEEPALIVE_INTERVAL_SEC
    next_server_time = time.time() + _SERVER_TIME_INTERVAL_SEC

    try:
        while True:
            # Disconnect check
            try:
                if await request.is_disconnected():
                    break
            except Exception:
                # Some test contexts don't implement is_disconnected; degrade gracefully
                pass

            now = time.time()

            # Server-time pulse for client-side latency proxy
            if now >= next_server_time:
                yield _sse_format({"t": now}, event_name="server_time")
                next_server_time = now + _SERVER_TIME_INTERVAL_SEC

            # Keepalive comment (drives heartbeat dot, also keeps proxies open)
            if now >= next_keepalive:
                yield f": keepalive {int(now)}\n\n"
                next_keepalive = now + _KEEPALIVE_INTERVAL_SEC

            # New events since last cursor
            new_events = get_events_since(last_id)
            for evt in new_events:
                yield _sse_format(evt)
                last_id = evt["id"]

            await asyncio.sleep(_STREAM_POLL_INTERVAL_SEC)
    except asyncio.CancelledError:
        # Client disconnected mid-stream — silent exit
        return
    except Exception as exc:
        logger.warning(f"[SSE_STREAM] generator exited unexpectedly: {exc}")
        return
