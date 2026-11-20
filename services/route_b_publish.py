"""📡 Route B (DECISION_LOG 597) — the bot's side: publish what it is watching so the recorder records those coins in full detail.

During a FRENZY pass the engine calls maybe_publish(...) when due() (at most once every PUBLISH_EVERY_S): ONE background task reads the
open positions (its own DB session, 5 s cap) and writes one small JSON to s3://scalpars-route-b-recorder-euw1/bot/frenzy_list.json —
  {"at": ms, "paper": bool, "open": {pair: {"s": sleeve (entry_strategy), "d": direction, "o": opened ISO}}  (OPEN positions),
   "flagged": {pair: spike ISO | null}  (FRENZY-flagged episodes), "tracked": [episodes EXHAUSTION / FLATBASE still follow],
   "followed": [every pair FRENZY reads this pass], "sleeves": {sleeve: on/off from the live config}}
  The recorder keeps every list it reads (events "bot") → the daily per-sleeve report (scripts/route_b/daily_report.py).
OBSERVE-ONLY and isolated from trading: the pass only builds the in-memory lists (no await); the S3 PUT runs on its OWN single thread
(never the shared default executor) with a cached boto3 client (connect 3 s · read 5 s · 2 attempts), and a new publish waits until the
previous PUT has really finished. Never raises. Off in replays (SCALPARS_REPLAY=1), under pytest (PYTEST_CURRENT_TEST) and with
SCALPARS_ROUTE_B_OFF=1. The instance role may write ONLY bot/* in that bucket (scripts/route_b/deploy_aws.sh). No publish happens while
every FRENZY sleeve is off (the pass returns early) — the recorder then runs on its own surge list (health: ⚠ DEGRADED, bot_n 0)."""
import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

BUCKET = "scalpars-route-b-recorder-euw1"
KEY = "bot/frenzy_list.json"
PUBLISH_EVERY_S = 240.0
OPEN_READ_TIMEOUT_S = 5.0
RECENT_CLOSED_S = 1800                            # positions closed within this window ride the list too (fast fills between publishes)
_state = {"at": 0.0, "task": None, "fut": None, "fails": 0}
_pool = None
_client = None
_lock = threading.Lock()


def enabled() -> bool:
    return not (os.environ.get("SCALPARS_REPLAY") == "1" or os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("SCALPARS_ROUTE_B_OFF") == "1")


def _iso(v):
    if v is None:
        return None
    try:
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%dT%H:%M:%S")
        x = float(v)
        return datetime.fromtimestamp(x / 1000.0 if x > 1e11 else x, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


# sleeve → (the config switches that must all be on, how the engine's entry_strategy names its fills): exact names, a prefix, and
# a direction where one strategy name serves both (MOMENTUM). Display / report only — never read by trading.
SLEEVE_SWITCHES = (
    ("EXHAUSTION_SHORT", ("exhaustion_short_enabled",), ("EXHAUSTION_SHORT",), None, None),
    ("DISTRIBUTION_SHORT", ("distribution_short_enabled",), ("DISTRIBUTION_SHORT",), None, None),
    ("FLATBASE_SHORT", ("flatbase_short_enabled",), ("FLATBASE_SHORT",), None, None),
    ("FRENZY_WILLY_HT", ("frenzy_willy_enabled", "frenzy_willy_ht_enabled"), ("FRENZY_WILLY_HT",), None, None),
    ("FRENZY_WILLY", ("frenzy_willy_enabled", "frenzy_willy_plain_entries_enabled"), ("FRENZY_WILLY",), None, None),
    ("FRENZY_LONG", ("frenzy_long_enabled",), ("FRENZY_LONG",), None, None),
    ("FRENZY_WIDE", ("frenzy_wide_enabled",), ("FRENZY_WIDE",), None, None),
    ("FRENZY_LITE", ("frenzy_lite_enabled",), ("FRENZY_LITE",), None, None),
    ("FRENZY_LITE_RETRY", ("frenzy_lite_retry_enabled",), ("FRENZY_LITE_RETRY",), None, None),
    ("MOMENTUM_LONG", ("momentum_long_entries_enabled",), ("MOMENTUM",), None, "LONG"),
    ("MOMENTUM_SHORT", ("momentum_short_entries_enabled",), ("MOMENTUM",), None, "SHORT"),
    ("FLIP", ("flip_entry_enabled",), (), "FLIP:", None),
    ("SURGE_LONG", ("surge_long_enabled",), ("SURGE_LONG",), None, None),
    ("SURGE_SHORT", ("surge_short_enabled",), ("SURGE_SHORT",), None, None),
    ("BULL_RUN", ("bullrun_sleeve_enabled",), ("BULLRUN_LONG",), None, None),
    ("BEAR_RUN", ("bearrun_sleeve_enabled",), ("BEARRUN_SHORT",), None, None),
    ("SPIKE_FADE", ("spike_fade_enabled",), ("SPIKE_FADE",), None, None),
    ("SPIKE_BOUNCE", ("spike_bounce_enabled",), ("SPIKE_BOUNCE",), None, None),
    ("SPIKE_CHASE", ("spike_chase_enabled",), ("SPIKE_CHASE",), None, None),
)


def sleeve_of(strategy, direction=None) -> str:
    """an Order's entry_strategy (+ direction) → the SLEEVE_SWITCHES name; MANUAL / unknown → "MANUAL" / "OTHER". Pure."""
    st = str(strategy or "").strip()
    d = str(direction or "").strip().upper()
    if not st:
        return "OTHER"
    if st.upper().startswith("MANUAL"):
        return "MANUAL"
    for name, _, exact, prefix, dirn in SLEEVE_SWITCHES:
        if (st in exact or (prefix and st.startswith(prefix))) and (dirn is None or d == dirn):
            return name
    return "OTHER"


def sleeves_on(th) -> dict:
    """{sleeve: True/False} from the live thresholds (a missing switch counts as off). Pure, never raises."""
    out = {}
    for name, keys, *_ in SLEEVE_SWITCHES:
        try:
            out[name] = all(bool(getattr(th, k, False)) for k in keys)
        except Exception:
            out[name] = False
    return out


def _open_map(rows) -> dict:
    """[(pair, entry_strategy, direction, opened_at[, status, closed_at])] or [pair] → {pair: {"s" sleeve, "e" entry_strategy, "d", "o"}}
    for the OPEN rows (the engine holds one position per coin). Pure, never raises."""
    out = {}
    for r in rows or []:
        try:
            if isinstance(r, str):
                out[r] = {"s": None, "e": None, "d": None, "o": None}
            elif len(r) < 5 or str(r[4]).upper() == "OPEN":
                out[str(r[0])] = {"s": sleeve_of(r[1], r[2]), "e": (str(r[1]) if r[1] is not None else None),
                                  "d": (str(r[2]) if r[2] is not None else None), "o": _iso(r[3])}
        except Exception:
            continue
    return out


def _recent_map(rows) -> dict:
    """the CLOSED rows of the same read (closed in the last RECENT_CLOSED_S) → {"pair|opened": {...+ "c" closed ISO}} — a fill that
    opened and closed between two publishes still reaches the daily report. Pure, never raises."""
    out = {}
    for r in rows or []:
        try:
            if isinstance(r, str) or len(r) < 6 or str(r[4]).upper() == "OPEN":
                continue
            o = _iso(r[3])
            out[f"{r[0]}|{o}"] = {"p": str(r[0]), "s": sleeve_of(r[1], r[2]), "e": (str(r[1]) if r[1] is not None else None),
                                  "d": (str(r[2]) if r[2] is not None else None), "o": o, "c": _iso(r[5])}
        except Exception:
            continue
    return out


def payload(now_ms, paper, open_rows, flags, tracked, followed, sleeves=None) -> dict:
    """the published document. flags = the engine's {pair: flag dict} (spike_ts read when present). Pure, never raises."""
    try:
        fl = {}
        for p, f in (flags or {}).items():
            fl[str(p)] = _iso((f or {}).get("spike_ts")) if isinstance(f, dict) else None
        return {"at": int(now_ms), "paper": bool(paper), "open": _open_map(open_rows), "recent": _recent_map(open_rows), "flagged": fl,
                "tracked": sorted({str(p) for p in (tracked or [])}), "followed": [str(p) for p in (followed or [])],
                "sleeves": dict(sleeves or {})}
    except Exception:
        return {"at": int(now_ms), "paper": bool(paper), "open": {}, "recent": {}, "flagged": {}, "tracked": [], "followed": [], "sleeves": {}}


def _s3():
    global _client
    with _lock:
        if _client is None:
            import boto3                                      # lazy (requirements.txt already carries it for the DB backup)
            from botocore.config import Config
            _client = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "eu-west-1",
                                   config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 2}))
        return _client


def _put(body: bytes) -> None:
    _s3().put_object(Bucket=BUCKET, Key=KEY, Body=body, ContentType="application/json")


def _executor():
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="route-b")
    return _pool


async def _publish(doc: dict, open_fn) -> None:
    try:
        if open_fn is not None:
            try:
                rows = await asyncio.wait_for(open_fn(), OPEN_READ_TIMEOUT_S)
                doc["open"], doc["recent"] = _open_map(rows), _recent_map(rows)
            except Exception:
                doc["open_unread"] = True                     # published anyway: the FRENZY list is the main content
        fut = _executor().submit(_put, json.dumps(doc, separators=(",", ":")).encode())
        _state["fut"] = fut
        await asyncio.wait_for(asyncio.wrap_future(fut), 20.0)
        _state["fails"] = 0
    except Exception as e:
        _state["fails"] += 1
        if _state["fails"] in (1, 10) or _state["fails"] % 100 == 0:   # quiet: a missing permission must not flood the log
            logger.warning(f"[ROUTE_B] frenzy list not published ({type(e).__name__}: {str(e)[:120]}) — fail #{_state['fails']} (observe-only)")


def due(now_s=None) -> bool:
    """True when a publish would be scheduled now: enabled, the interval passed, the last task AND its S3 thread finished. Never raises."""
    try:
        t, f = _state.get("task"), _state.get("fut")
        return (enabled() and ((time.time() if now_s is None else float(now_s)) - _state["at"] >= PUBLISH_EVERY_S)
                and (t is None or t.done()) and (f is None or f.done()))
    except Exception:
        return False


def maybe_publish(paper, open_fn, flags, tracked, followed, now_s=None, th=None) -> bool:
    """schedule one publish when due (open_fn = an async callable → the open pairs, read inside the task). → True when scheduled.
    Never raises; never awaits."""
    try:
        now_s = time.time() if now_s is None else float(now_s)
        if not due(now_s):
            return False
        _state["at"] = now_s
        doc = payload(int(now_s * 1000), paper, [], dict(flags or {}), list(tracked or []), list(followed or []),
                      sleeves_on(th) if th is not None else {})   # snapshot now
        _state["task"] = asyncio.get_running_loop().create_task(_publish(doc, open_fn))
        return True
    except Exception:
        return False
