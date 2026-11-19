"""📡 Route B (DECISION_LOG 597) — the bot's side: publish what it is watching so the recorder records those coins in full detail.

During a FRENZY pass the engine calls maybe_publish(...) when due() (at most once every PUBLISH_EVERY_S): ONE background task reads the
open positions (its own DB session, 5 s cap) and writes one small JSON to s3://scalpars-route-b-recorder-euw1/bot/frenzy_list.json —
  {"at": ms, "paper": bool, "open": [pairs with an OPEN position], "flagged": {pair: spike ISO | null}  (FRENZY-flagged episodes),
   "tracked": [episodes EXHAUSTION / FLATBASE still follow], "followed": [every pair FRENZY reads this pass]}
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


def payload(now_ms, paper, open_pairs, flags, tracked, followed) -> dict:
    """the published document. flags = the engine's {pair: flag dict} (spike_ts read when present). Pure, never raises."""
    try:
        fl = {}
        for p, f in (flags or {}).items():
            fl[str(p)] = _iso((f or {}).get("spike_ts")) if isinstance(f, dict) else None
        return {"at": int(now_ms), "paper": bool(paper), "open": sorted({str(p) for p in (open_pairs or [])}), "flagged": fl,
                "tracked": sorted({str(p) for p in (tracked or [])}), "followed": [str(p) for p in (followed or [])]}
    except Exception:
        return {"at": int(now_ms), "paper": bool(paper), "open": [], "flagged": {}, "tracked": [], "followed": []}


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
                doc["open"] = sorted({str(p) for p in (await asyncio.wait_for(open_fn(), OPEN_READ_TIMEOUT_S) or [])})
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


def maybe_publish(paper, open_fn, flags, tracked, followed, now_s=None) -> bool:
    """schedule one publish when due (open_fn = an async callable → the open pairs, read inside the task). → True when scheduled.
    Never raises; never awaits."""
    try:
        now_s = time.time() if now_s is None else float(now_s)
        if not due(now_s):
            return False
        _state["at"] = now_s
        doc = payload(int(now_s * 1000), paper, [], dict(flags or {}), list(tracked or []), list(followed or []))   # snapshot now
        _state["task"] = asyncio.get_running_loop().create_task(_publish(doc, open_fn))
        return True
    except Exception:
        return False
