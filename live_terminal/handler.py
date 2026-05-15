"""LiveTerminalHandler — captures every LogRecord into a thread-safe ring
buffer parsed for tag/symbol/side/key-value pairs. Never raises.

Attaches to the root logger so existing logger.info/.warning/.error calls
across the codebase are captured without any modification at the call site.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from typing import Any

from .tag_map import category_for

# ─── Ring buffer ──────────────────────────────────────────────────────────
# 500-event ring, thread-safe. _EVENT_ID monotonically increases so SSE
# consumers can track their cursor and request "events since last_id".
_BUFFER: deque = deque(maxlen=500)
_LOCK = threading.Lock()
_EVENT_ID = 0

# ─── Parsers ──────────────────────────────────────────────────────────────
# Tag must be at the very start: "[TAG_NAME] rest of message"
_TAG_RE = re.compile(r"^\[([A-Z][A-Z0-9_]*)\]\s*")
# Symbol: matches XYZUSDT (no leading-digit pairs because regex is anchored on [A-Z])
# Allowing 2-12 chars before USDT; covers 1000PEPEUSDT etc. via a separate digit-prefix variant.
_SYMBOL_RE = re.compile(r"\b((?:\d+)?[A-Z]{2,10}USDT)\b")
_SIDE_RE = re.compile(r"\b(LONG|SHORT)\b")
# key=value where value has no spaces or commas — common in our log style
_KV_RE = re.compile(r"\b([A-Za-z_][\w]*)=([^\s,]+)")


def _parse_message(msg: str) -> dict[str, Any]:
    """Extract tag, symbol, side, kv-pairs, and tail from a log message.

    Returns a dict even on parse failure (never raises).
    """
    try:
        tag_m = _TAG_RE.match(msg)
        if tag_m:
            tag = tag_m.group(1)
            rest = msg[tag_m.end():]
        else:
            tag = "RAW"
            rest = msg

        sym_m = _SYMBOL_RE.search(rest)
        side_m = _SIDE_RE.search(rest)
        # First 12 kv pairs only — guards against absurd messages
        kvs = {}
        for k, v in _KV_RE.findall(rest):
            if k in kvs:
                continue
            kvs[k] = v
            if len(kvs) >= 12:
                break

        return {
            "tag": tag,
            "symbol": sym_m.group(1) if sym_m else None,
            "side": side_m.group(1) if side_m else None,
            "kv": kvs,
            "tail": rest.strip(),
        }
    except Exception:
        # Belt-and-suspenders: handler must never raise
        return {
            "tag": "RAW",
            "symbol": None,
            "side": None,
            "kv": {},
            "tail": msg[:500] if isinstance(msg, str) else "",
        }


def push_event(level: str, parsed: dict, raw_msg: str, extra: dict | None = None) -> dict:
    """Append a parsed event to the ring buffer. Returns the event with its id."""
    global _EVENT_ID
    with _LOCK:
        _EVENT_ID += 1
        evt = {
            "id": _EVENT_ID,
            "ts": time.time(),
            "level": level,
            "tag": parsed["tag"],
            "category": category_for(parsed["tag"], level),
            "symbol": parsed["symbol"],
            "side": parsed["side"],
            "kv": parsed["kv"],
            "tail": parsed["tail"],
            "msg": raw_msg if len(raw_msg) <= 500 else raw_msg[:500] + "…",
        }
        if extra:
            evt.update(extra)
        _BUFFER.append(evt)
        return evt


def get_events_since(last_id: int) -> list[dict]:
    """Return ring-buffer events with id > last_id, oldest first."""
    with _LOCK:
        return [e for e in _BUFFER if e["id"] > last_id]


def get_snapshot() -> list[dict]:
    """Return a copy of the current ring buffer (oldest first)."""
    with _LOCK:
        return list(_BUFFER)


def current_event_id() -> int:
    with _LOCK:
        return _EVENT_ID


class LiveTerminalHandler(logging.Handler):
    """Captures every LogRecord, parses it, pushes to the ring buffer.

    Critical invariant: emit() MUST NOT raise. Trading code is the caller;
    a crash in this handler would crash the calling code.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # getMessage() returns the bare formatted message string
            # (without the asctime/name/level prefix).
            actual_msg = record.getMessage()
            parsed = _parse_message(actual_msg)
            push_event(record.levelname, parsed, actual_msg)
        except Exception:
            # Swallow everything. Trading must continue regardless.
            pass


# ─── Installer ────────────────────────────────────────────────────────────
_INSTALLED = False


def install() -> LiveTerminalHandler | None:
    """Attach LiveTerminalHandler to the root logger. Idempotent.

    Safe to call multiple times. Returns the handler instance, or None if
    already installed.
    """
    global _INSTALLED
    if _INSTALLED:
        return None
    root = logging.getLogger()
    handler = LiveTerminalHandler()
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    _INSTALLED = True
    return handler
