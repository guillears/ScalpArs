#!/usr/bin/env python3
"""📡 Route B — live Binance USDT-M futures recorder (DECISION_LOG 597; RESEARCH ONLY — never trades, never talks to the bot's API).

WHY   Order-book history cannot be downloaded afterwards; Route A (reports/ROUTE_A_*) showed the trade tape separates Frenzy runners from
      fizzles (AUC 0.60–0.65) without a profitable rule — the book is the untested piece, for every sleeve (Frenzy longs, EXH / DIST /
      Flatbase shorts, momentum). Phase 1 = record; a live trigger only after a pre-registered study passes.
WHAT  three websocket connections (public market data — no API key, no REST weight):
        ① MARKET (all coins, /market/stream): !forceOrder@arr — liquidations, a SNAPSHOT stream (Binance sends at most the latest
           liquidation per symbol per second, so it under-counts bursts) · !markPrice@arr@1s (mark / index / funding, one row per second)
        ② BOOK (hot coins, /public/stream): <sym>@depth20@500ms (top-20 book)
        ③ TRADE (hot coins, /market/stream): <sym>@aggTrade (every trade, taker side)
        ④ FULL BOOK (the bot's PRIORITY coins, ≤ FULL_MAX, /public/stream): <sym>@depth@100ms diffs + a REST snapshot (limit 1000, weight
           20) → a LOCAL full book per Binance's U / u / pu procedure (fullbook.py) → kinds fbook (derived, 1 s) + fsnap (±5 % levels, 30 s).
           PRIORITY = the bot list's open · flagged · tracked (not 'followed'), read every FULL_REFRESH_S, sticky 30 min. Snapshot REST is
           budgeted (FB_WEIGHT_PER_MIN), stops on 418 / 429 / a high X-MBX-USED-WEIGHT-1M. Without fullbook.py the recorder runs without ④.
        ⑤ OPEN INTEREST (REST, every 60 s, priority + hot coins ≤ 150, weight 1 each) → kind oi · POSITIONING ratios (REST /futures/data,
           every 5 min, latest 5m row of 4 endpoints, ≤ 500 requests / 5 min) → kind ratio (positioning.py; POS_OFF=1 turns ⑤ off).
      HOT list (≤ MAX_HOT), refreshed every 5 min, in this priority: BTC + ETH · the BOT's own list (s3://<BUCKET>/bot/frenzy_list.json,
      written by the bot during its FRENZY pass: open positions · FRENZY-flagged episodes · episodes EXHAUSTION / FLATBASE still track ·
      every pair FRENZY follows; ignored when > 30 min old) · coins already in (≥ 60 min, no churn) · SURGE coins (|move vs ≥ 30 min ago|
      ≥ 3 %, from stream ①, kept 6 h — the backup when the bot list is stale) · top 24 h movers (quote volume ≥ $5M) · top 24 h volume.
      REST: the 24 h ticker every 5 min (weight 40) + exchangeInfo once a day (weight 1).
FILES <DATA_DIR>/<kind>/<UTC date>/<HH>-<process start ms>.jsonl.gz (kinds: liq · mark · book · trade · events · fbook · fsnap · oi · ratio), one JSON object per
      line {"r": local receive ms, ...payload}. ONE file per process and hour (a restart never appends to a file a killed process left
      half-written). A CLOSED hour's files go to s3://<BUCKET>/data/<kind>/<date>/<same name> and are deleted locally only after a
      successful upload. A reader should still tolerate a truncated last gzip member (power loss).
HEALTH status.json every 60 s (uploaded every 5 min → s3://<BUCKET>/health/status.json) · daily summary at the UTC day change →
      health/daily/<date>.json · every disconnect / close / reconnect / start / stop → the events kind (the gap log).
SAFETY SIGTERM / SIGINT close the files cleanly · a read silent for READ_TIMEOUT_S reconnects (half-open sockets) · every disconnect —
      clean close included — backs off · below MIN_FREE_GB of disk the recorder stops WRITING but keeps uploading (frees the disk).
Env   BUCKET (empty → no upload / no bot list: local test) · DATA_DIR (default /var/lib/scalpars-recorder) · MAX_HOT (default 120) ·
      FULL_MAX (default 30; 0 = no full book) · FULL_SYMS (comma list — a fixed priority set, local smoke tests only).
Usage python3 recorder.py [--seconds N]   (N > 0 → stop after N seconds: a local smoke test)"""
import argparse
import asyncio
import gzip
import http.client
import math
import json
import os
import shutil
import signal
import socket
import time
import sys
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the sibling modules, also under python -I

try:                                              # ④ the full book (a missing file must never stop the rest of the recorder)
    import fullbook as FB
except Exception:                                 # pragma: no cover
    FB = None

try:                                              # ⑤ open interest + positioning (same rule)
    import positioning as PS
except Exception:                                 # pragma: no cover
    PS = None

try:                                              # websockets ≥ 13: the new asyncio client
    from websockets.asyncio.client import connect as ws_connect
except Exception:                                 # pragma: no cover — older websockets
    from websockets import connect as ws_connect

WS_MARKET = "wss://fstream.binance.com/market/stream"   # Binance's 2025 split (checked live 2026-11-19): aggTrade / markPrice / forceOrder
WS_PUBLIC = "wss://fstream.binance.com/public/stream"   # … and the order book (depth); the legacy /stream no longer carries the /market ones
REST = "https://fapi.binance.com"
REST_HOST = "fapi.binance.com"
MARKET_STREAMS = ("!forceOrder@arr", "!markPrice@arr@1s")
HOT_CONNS = {"book": (WS_PUBLIC, "@depth20@500ms"), "trade": (WS_MARKET, "@aggTrade")}   # kind → (endpoint, stream suffix)
CORE = ("BTCUSDT", "ETHUSDT")
SURGE_PCT, SURGE_MIN, SURGE_MAX_MIN, SURGE_KEEP_S = 3.0, 30, 40, 6 * 3600
MOVER_MIN_QV = 5_000_000.0
N_MOVERS, N_VOLUME = 40, 15
STICKY_S = 3600
HOT_REFRESH_S, INFO_REFRESH_S = 300, 86_400
SUB_CHUNK, SUB_GAP_S = 50, 0.3                    # Binance: ≤ 10 incoming messages / s per connection, ≤ 1,024 streams
FLUSH_S, STATUS_S, UPLOAD_S = 5, 60, 300
READ_TIMEOUT_S = 60                               # mark arrives every second, BTC's book every 500 ms → 60 s of silence = a dead socket
S3_TIMEOUT_S = 120
UPLOAD_MAX_FILES = 20                             # per pass — a backlog never stalls housekeeping for long (status keeps flowing)
MIN_FREE_GB = 1.0
KINDS = ("liq", "mark", "book", "trade", "events", "fbook", "fsnap", "oi", "ratio")
FEATURE_KINDS = ("book", "trade", "liq", "mark", "fbook", "oi", "ratio")   # the inputs of features.py (events / fsnap are not)
FULL_REFRESH_S = 60                               # the priority set is re-read from the bot list every minute
FB_REST_TIMEOUT_S = 10
BOT_DOC_TTL_S = 55                                # in-process (boto3) bot-list read: once a minute for both managers …
BOT_DOC_TTL_CLI_S = 300                           # … without boto3 the aws CLI is spawned at most every 5 min (the bot publishes every 4)
OI_SLOW_S = 45                                    # health ⚠ when one OI sweep takes longer (it must fit its 60 s slot)
FB_SYNC_FRESH_S = 10                              # "synced" in the status = an fbook line in the last 10 s
PAUSE_FILE = "rest_pause.json"                    # a 418 / 429 pause (and the ratio 429 pause) survives a restart
MAX_PAUSE_MS = 3 * 86_400_000                     # a saved / announced pause longer than 3 days is ignored / clamped
_tls = threading.local()                          # one keep-alive HTTPS connection per REST worker thread


def rest_pause_s(status, headers, now_s):
    """seconds to stop the weight-counted REST reads after a response (fullbook.rest_pause_s; a minimal copy when that file is missing)."""
    if FB is not None:
        return FB.rest_pause_s(status, headers, now_s)
    return 600.0 if status == 418 else 120.0 if status == 429 else 0.0
FEATURE_TIMEOUT_S = 600
FEATURE_MAX_TRIES = 2                             # a hour whose summary fails twice is uploaded raw anyway (rebuildable from S3)


def now_ms() -> int:
    return int(time.time() * 1000)


def hour_key(ms: int):
    """(UTC date 'YYYY-MM-DD', hour 'HH') of a ms timestamp. Pure."""
    t = time.gmtime(ms / 1000.0)
    return time.strftime("%Y-%m-%d", t), time.strftime("%H", t)


def backoff_s(n: int) -> float:
    """seconds to wait before reconnect attempt n (1, 2, …): 2, 3, … capped at 30. Pure."""
    return float(min(30, 1 + max(1, int(n))))


# ── pure selection logic (tested) ──────────────────────────────────────────────────────────────────────────────────────────────

class Surge:
    """per-symbol 1-minute mark samples (≤ SURGE_MAX_MIN + 1 minutes) → symbols whose |move vs the newest sample ≥ 30 min old (and ≤ 40)|
    ≥ SURGE_PCT; each kept SURGE_KEEP_S. A missing minute never blinds it (nearest older sample, not an exact 30-min one)."""

    def __init__(self):
        self.px = {}          # sym → deque[(minute, price)]
        self.hot = {}         # sym → last surge ms

    def feed(self, sym, price, ms):
        try:
            p = float(price)
        except (TypeError, ValueError):
            return
        if not p > 0:
            return
        m = int(ms) // 60_000
        d = self.px.setdefault(sym, deque(maxlen=SURGE_MAX_MIN + 1))
        if d and d[-1][0] == m:
            d[-1] = (m, p)
            return
        d.append((m, p))
        ref = None
        for mm, pp in d:                              # oldest → newest: the LAST sample still ≥ 30 min old
            if mm <= m - SURGE_MIN:
                ref = (mm, pp)
            else:
                break
        if ref is not None and ref[0] >= m - SURGE_MAX_MIN and abs(p / ref[1] - 1.0) * 100.0 >= SURGE_PCT:
            self.hot[sym] = int(ms)

    def active(self, ms):
        """symbols surged within SURGE_KEEP_S, newest first (expired ones dropped)."""
        for s in [s for s, t in self.hot.items() if ms - t > SURGE_KEEP_S * 1000]:
            del self.hot[s]
        return [s for s, _ in sorted(self.hot.items(), key=lambda kv: -kv[1])]


BOT_KEYS = ("open", "flagged", "tracked", "followed")   # the bot list's groups, in priority order
BOT_MAX_AGE_S = 1800


def bot_pairs(doc, ms):
    """the bot's list (bot/frenzy_list.json) → its pairs in priority order; [] when missing, malformed or older than BOT_MAX_AGE_S. Pure."""
    try:
        if not doc or ms - int(doc.get("at") or 0) > BOT_MAX_AGE_S * 1000:
            return []
        out = []
        for k in BOT_KEYS:
            v = doc.get(k) or []
            for p in (v.keys() if isinstance(v, dict) else v):
                if isinstance(p, str) and p and p not in out:
                    out.append(p)
        return out
    except (TypeError, ValueError, AttributeError):
        return []


def pick_hot(tickers, tradable, surge, current, added_ms, ms, max_hot, bot=()):
    """→ the new hot list (ordered). tickers = /fapi/v1/ticker/24hr rows · tradable = USDT perpetual TRADING symbols (None = no filter) ·
    surge = Surge.active() · current = the subscribed list · added_ms = sym → when it entered · bot = bot_pairs(). Pure."""
    ok = (lambda s: s in tradable) if tradable else (lambda s: s.endswith("USDT"))
    rows = []
    for t in tickers or []:
        try:
            rows.append((t["symbol"], abs(float(t["priceChangePercent"])), float(t["quoteVolume"])))
        except (KeyError, TypeError, ValueError):
            continue
    rows = [r for r in rows if ok(r[0])]
    movers = [s for s, ch, qv in sorted(rows, key=lambda r: -r[1]) if qv >= MOVER_MIN_QV][:N_MOVERS]
    volume = [s for s, _, _ in sorted(rows, key=lambda r: -r[2])][:N_VOLUME]
    want = []
    sticky = [s for s in current if ok(s) and ms - added_ms.get(s, ms) < STICKY_S * 1000]   # a delisted coin leaves at once
    for s in list(CORE) + [x for x in bot if ok(x)] + sticky + [x for x in surge if ok(x)] + movers + volume:
        if s not in want:
            want.append(s)
    return want[:max_hot]


def hot_streams(syms, suffix):
    return [s.lower() + suffix for s in syms]


def compact_mark(arr):
    """!markPrice@arr rows → [[symbol, mark, index, funding, next funding ms]]. Pure."""
    out = []
    for x in arr or []:
        try:
            out.append([x["s"], x["p"], x.get("i"), x.get("r"), x.get("T")])
        except (KeyError, TypeError):
            continue
    return out


def file_hour(name):
    """'HH-<ms>.jsonl.gz' (or legacy 'HH.jsonl.gz') → 'HH', else None. Pure."""
    if not name.endswith(".jsonl.gz") or len(name) < 11 or not name[:2].isdigit():
        return None
    return name[:2]


def hour_groups(files):
    """closed_files() rows → {(date, HH): {kind: [paths]}}. Pure."""
    out = {}
    for kind, day, fn, path in files:
        h = file_hour(fn)
        out.setdefault((day, h), {}).setdefault(kind, []).append(path)
    return out


def closed_files(data_dir, cur_date, cur_hour):
    """every <kind>/<date>/<HH…>.jsonl.gz whose hour is before the current UTC hour. Pure on the file tree."""
    out = []
    for kind in KINDS:
        kd = os.path.join(data_dir, kind)
        if not os.path.isdir(kd):
            continue
        for d in sorted(os.listdir(kd)):
            dd = os.path.join(kd, d)
            if not os.path.isdir(dd):
                continue
            for f in sorted(os.listdir(dd)):
                h = file_hour(f)
                if h is not None and (d, h) < (cur_date, cur_hour):
                    out.append((kind, d, f, os.path.join(dd, f)))
    return out


# ── the recorder ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

class Writer:
    """one gzip file per (kind, UTC hour, process): <dir>/<kind>/<date>/<HH>-<tag>.jsonl.gz."""

    def __init__(self, data_dir, tag=None):
        self.dir = data_dir
        self.tag = str(now_ms() if tag is None else tag)
        self.fh = {}           # kind → (date, hour, handle)
        self.n = {k: 0 for k in KINDS}
        self.bytes = 0
        self.paused = False    # disk nearly full → drop writes (counted) until the uploader frees space
        self.dropped = 0
        self.closed = set()    # (kind, date, hour) this process already closed — never reopened (it may be uploaded already)

    def write(self, kind, obj, ms=None):
        if self.paused:
            self.dropped += 1
            return
        ms = now_ms() if ms is None else ms
        d, h = hour_key(ms)
        if (kind, d, h) in self.closed:                              # a late line for a closed (maybe uploaded) hour → file it NOW
            d, h = hour_key(now_ms())
        cur = self.fh.get(kind)
        if cur is None or cur[0] != d or cur[1] != h:
            if cur is not None:
                cur[2].close()
                self.closed.add((kind, cur[0], cur[1]))
            p = os.path.join(self.dir, kind, d)
            os.makedirs(p, exist_ok=True)
            fp = os.path.join(p, f"{h}-{self.tag}.jsonl.gz")
            cur = (d, h, gzip.open(fp, "at", compresslevel=6), fp)
            self.fh[kind] = cur
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        cur[2].write(line)
        self.n[kind] += 1
        self.bytes += len(line)

    def flush(self):
        for v in self.fh.values():
            try:
                v[2].flush()
            except Exception:
                pass

    def roll(self, d=None, h=None):
        """close handles of a past hour (so the uploader can take them)."""
        if d is None:
            d, h = hour_key(now_ms())
        for k in list(self.fh):
            if (self.fh[k][0], self.fh[k][1]) != (d, h):
                try:
                    self.fh[k][2].close()
                finally:
                    self.closed.add((k, self.fh[k][0], self.fh[k][1]))
                    del self.fh[k]
        if len(self.closed) > 5000:                                  # bounded: keep only today's and yesterday's keys
            keep = {hour_key(now_ms() - x)[0] for x in (0, 86_400_000)}
            self.closed = {c for c in self.closed if c[1] in keep}

    def close(self):
        for v in self.fh.values():
            try:
                v[2].close()
            except Exception:
                pass
        self.fh = {}

    def open_paths(self):
        return {v[3] for v in self.fh.values()}


class Recorder:
    def __init__(self, data_dir, bucket, max_hot, stop_at=None, full_max=None, full_syms=None):
        self.dir, self.bucket, self.max_hot, self.stop_at = data_dir, bucket, max_hot, stop_at
        self.started = now_ms()
        self.w = Writer(data_dir, self.started)
        self.surge = Surge()
        self.hot, self.added = [], {}
        self.tradable = None
        self.last = {k: None for k in KINDS}
        self.reconnects = {"market": 0, "book": 0, "trade": 0, "fbook": 0}
        self.host = socket.gethostname()
        self.hot_ws = {}       # kind → the live websocket (book / trade)
        self.day = hour_key(self.started)[0]
        self.day_n = {k: 0 for k in KINDS}
        self.day_gap_s = {"market": 0.0, "book": 0.0, "trade": 0.0, "fbook": 0.0}
        self.day_hot = set()
        self.bot_n = 0
        self.uploaded = 0
        self.upload_fail = 0
        self.hk_errors = 0
        self.feat_tries = {}     # (date, HH) → failed attempts
        self.feat_done = 0
        self.feat_fail = 0
        self.feat_last = None    # "date HH" of the newest summary uploaded
        self.feat_build_s = None  # seconds the last summary took
        self.bot_at = None       # the last bot list logged (its "at")
        # ④ full book
        self.full_req = (FB.FULL_MAX if FB else 30) if full_max is None else max(0, int(full_max))   # what the unit asked for
        self.full_max = self.full_req if FB is not None else 0
        self.full_on = FB is not None and self.full_max > 0
        self.full_fixed = [s for s in (full_syms or []) if s]
        self.full, self.full_seen, self.full_want_n = [], {}, 0
        self.books = {}          # sym → FB.LocalBook
        self.fb_bucket = FB.WeightBucket() if FB else None
        self.fb_rest = {"paused_until": 0, "pause_why": None, "used_1m": None, "snapshots": 0, "weight": 0, "n429": 0, "n418": 0,
                        "errors": 0}
        self.snap_pool = None    # the snapshotter's own thread (sweeps never delay a snapshot)
        self.sweep_pool = None   # the OI / ratio sweeps' own threads (never the default executor hot_manager uses)
        self.s3 = None           # boto3 client for the bot list (None = not installed → the aws CLI every BOT_DOC_TTL_CLI_S)
        self.bot_doc, self.bot_doc_ms = None, 0
        self.last_tick = None    # the last 24 h ticker (reused during a REST pause)
        self.fb_tot = self._fb_zero()
        self.fb_day = self._fb_zero()
        # ⑤ open interest + positioning
        self.pos_off = os.environ.get("POS_OFF") == "1"
        self.pos_on = PS is not None and not self.pos_off
        self.oi_track = PS.OiTrack() if PS else None
        self.ratio_win = PS.Rolling(300) if PS else None
        self.ratio_paused_until = 0
        self.pos = self._pos_zero()
        self.pos_day = {"oi_want": 0, "oi_ok": 0, "ratio_want": 0, "ratio_ok": 0}
        self._load_pause()
        self.stop = asyncio.Event()

    @staticmethod
    def _pos_zero():
        return {"oi_sweeps": 0, "oi_last_ms": None, "oi_coins": 0, "oi_ok": 0, "oi_err": 0, "oi_skipped": 0, "oi_sweep_s": None,
                "ratio_sweeps": 0, "ratio_last_ms": None, "ratio_tried_ms": None, "ratio_coins": 0, "ratio_ok": 0, "ratio_err": 0,
                "ratio_skipped": 0, "ratio_unread": 0, "ratio_n429": 0}

    @staticmethod
    def _fb_zero():
        return {"syncs": 0, "refreshes": 0, "gaps": 0, "stale": 0, "crossed": 0, "bad": 0, "silent": 0, "empty": 0, "snap_err": 0,
                "want_s": 0, "synced_s": 0}

    def _fb_count(self, what, n=1):
        if what in self.fb_tot:
            self.fb_tot[what] += n
            self.fb_day[what] += n

    def event(self, what, **kw):
        try:
            self.w.write("events", dict(r=now_ms(), ev=what, host=self.host, **kw))
        except Exception:
            pass

    def _count(self, kind):
        self.last[kind] = now_ms()
        self.day_n[kind] += 1

    async def _read(self, ws):
        """the next message, or TimeoutError after READ_TIMEOUT_S of silence (a half-open socket)."""
        return await asyncio.wait_for(ws.recv(), READ_TIMEOUT_S)

    async def _conn_loop(self, name, url, on_open, on_msg, idle_ok=None):
        """connect → on_open(ws) → on_msg(raw) per message; ANY end (error, clean close, read timeout) → logged gap + backoff + reconnect."""
        down, n_fail = None, 0
        while not self.stop.is_set():
            why, opened = "closed by the server", False
            try:
                async with ws_connect(url, max_size=2 ** 24, open_timeout=20) as ws:   # default keepalive pings stay on
                    opened = True
                    await on_open(ws)
                    if down is not None:
                        gap = (now_ms() - down) / 1000
                        self.event("reconnect", conn=name, gap_s=round(gap, 1))
                        self.day_gap_s[name] += gap
                    down, n_fail = None, 0
                    while not self.stop.is_set():
                        try:
                            raw = await self._read(ws)
                        except asyncio.TimeoutError:
                            if idle_ok is not None and idle_ok():
                                continue                             # nothing subscribed: silence is normal (pings keep it alive)
                            raise
                        on_msg(raw)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                why = f"no data for {READ_TIMEOUT_S} s" if opened else "connect timeout"
            except Exception as e:
                why = f"{type(e).__name__}: {str(e)[:160]}"
            finally:
                if name in self.hot_ws:
                    self.hot_ws.pop(name, None)
            if self.stop.is_set():
                break
            if down is None:
                down = now_ms()
            n_fail += 1
            self.reconnects[name] += 1
            self.event("disconnect", conn=name, why=why)
            try:
                await asyncio.wait_for(self.stop.wait(), backoff_s(n_fail))
            except asyncio.TimeoutError:
                pass

    # ① all-market
    async def market(self):
        async def on_open(ws):
            return None

        def on_msg(raw):
            r = now_ms()
            m = json.loads(raw)
            st, d = m.get("stream", ""), m.get("data")
            if st.startswith("!forceOrder"):
                self.w.write("liq", {"r": r, "d": d}, r)
                self._count("liq")
            elif st.startswith("!markPrice"):
                rows = compact_mark(d)
                for s, p, *_ in rows:
                    self.surge.feed(s, p, r)
                ev = (d[0].get("E") if isinstance(d, list) and d and isinstance(d[0], dict) else None)
                self.w.write("mark", {"r": r, "E": ev, "m": rows}, r)
                self._count("mark")
        await self._conn_loop("market", WS_MARKET + "?streams=" + "/".join(MARKET_STREAMS), on_open, on_msg)

    # ②③ hot coins
    async def _sub(self, ws, method, streams):
        for i in range(0, len(streams), SUB_CHUNK):
            await ws.send(json.dumps({"method": method, "params": streams[i:i + SUB_CHUNK], "id": now_ms() % 1_000_000_000}))
            await asyncio.sleep(SUB_GAP_S)

    async def hotconn(self, kind):
        """one connection per hot kind (book on /public, trade on /market): subscribe the hot list on every (re)connect, then record."""
        url, suffix = HOT_CONNS[kind]

        async def on_open(ws):
            self.hot_ws[kind] = ws
            if self.hot:
                await self._sub(ws, "SUBSCRIBE", hot_streams(list(self.hot), suffix))

        def on_msg(raw):
            r = now_ms()
            m = json.loads(raw)
            st = m.get("stream")
            if not st:
                if m.get("error"):
                    self.event("sub_error", conn=kind, why=str(m.get("error"))[:200])
                return                # SUBSCRIBE acks
            d = m.get("data") or {}
            if kind == "trade":
                self.w.write("trade", {"r": r, "d": d}, r)
            else:
                self.w.write("book", {"r": r, "s": d.get("s"), "E": d.get("E"), "T": d.get("T"), "b": d.get("b"), "a": d.get("a")}, r)
            self._count(kind)
        await self._conn_loop(kind, url, on_open, on_msg)

    async def _rest_json(self, path):
        """a weight-counted REST read through the shared pause / used-weight bookkeeping → body; raises on any non-200 (the caller's
        loop logs it and retries next pass)."""
        status, headers, body = await self._in_pool("sweep", self._rest, path)
        self._rest_result(status, headers)
        if status != 200 or body is None:
            raise RuntimeError(f"HTTP {status} on {path.split('?')[0]}")
        return body

    async def _bot_doc(self):
        """the bot list, read at most once per BOT_DOC_TTL_S (hot_manager + full_manager share it)."""
        ttl = BOT_DOC_TTL_S if self._boto() is not None else BOT_DOC_TTL_CLI_S
        if self.bot_doc_ms and time.time() * 1000 - self.bot_doc_ms < ttl * 1000:
            return self.bot_doc
        self.bot_doc, self.bot_doc_ms = await self._bot_list(), now_ms()
        return self.bot_doc

    def _boto(self):
        """a boto3 S3 client (instance-role credentials, short timeouts) or None when boto3 is not installed / no bucket."""
        if self.s3 is None and self.bucket:
            try:
                import boto3
                from botocore.config import Config
                self.s3 = boto3.client("s3", config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 2}))
            except Exception:
                self.s3 = False
        return self.s3 or None

    def _load_pause(self):
        """a 418 / 429 pause written by a previous process → honoured at start. Never raises."""
        try:
            with open(os.path.join(self.dir, PAUSE_FILE)) as f:
                d = json.load(f)
        except Exception:
            return
        if not isinstance(d, dict):
            return
        now = now_ms()

        def ok(v):                                                   # finite, in the future, ≤ 3 days ahead (1e30 / garbage ignored)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return int(v) if math.isfinite(v) and now < v <= now + MAX_PAUSE_MS else None
        u = ok(d.get("until"))
        if u is not None:
            self.fb_rest["paused_until"], self.fb_rest["pause_why"] = u, d.get("why")
        r = ok(d.get("ratio_until"))
        if r is not None:
            self.ratio_paused_until = r

    def _save_pause(self):
        try:
            os.makedirs(self.dir, exist_ok=True)
            self._write_json(PAUSE_FILE, {"until": self.fb_rest["paused_until"], "why": self.fb_rest["pause_why"],
                                          "ratio_until": self.ratio_paused_until})
        except Exception:
            pass

    async def hot_manager(self):
        last_info = 0
        while not self.stop.is_set():
            try:
                paused = now_ms() < self.fb_rest["paused_until"]
                if paused:                                           # a 418 / 429 / weight pause: no REST — the websockets still
                    tick = self.last_tick or []                      # record core + bot + surge coins (they cost no weight)
                else:
                    if time.time() - last_info > INFO_REFRESH_S or self.tradable is None:
                        info = await self._rest_json("/fapi/v1/exchangeInfo")
                        self.tradable = {s["symbol"] for s in info.get("symbols", []) if s.get("contractType") == "PERPETUAL"
                                         and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"}
                        last_info = time.time()
                    tick = self.last_tick = await self._rest_json("/fapi/v1/ticker/24hr")
                doc = await self._bot_doc()
                if isinstance(doc, dict) and doc.get("at") != self.bot_at:   # 📋 history of what the bot watched / held (the daily report)
                    self.bot_at = doc.get("at")
                    self.event("bot", at=doc.get("at"), paper=doc.get("paper"), open=doc.get("open"),
                               flagged=sorted((doc.get("flagged") or {}).keys()) if isinstance(doc.get("flagged"), dict) else doc.get("flagged"),
                               tracked=doc.get("tracked"), n_followed=len(doc.get("followed") or []), sleeves=doc.get("sleeves"))
                bot = bot_pairs(doc, now_ms())
                ms = now_ms()
                new = pick_hot(tick, self.tradable, self.surge.active(ms), self.hot, self.added, ms, self.max_hot, bot)
                self.bot_n = len(bot)
                add, drop = [s for s in new if s not in self.hot], [s for s in self.hot if s not in new]
                for s in add:
                    self.added[s] = ms
                for s in drop:
                    self.added.pop(s, None)
                self.hot = new        # FIRST: a connection that comes back during the sends below subscribes the NEW list
                self.day_hot.update(new)
                for kind, (_, suffix) in HOT_CONNS.items():
                    ws = self.hot_ws.get(kind)
                    if ws is None:
                        continue          # (re)connecting — on_open subscribes self.hot
                    try:
                        if drop:
                            await self._sub(ws, "UNSUBSCRIBE", hot_streams(drop, suffix))
                        if add:
                            await self._sub(ws, "SUBSCRIBE", hot_streams(add, suffix))   # a duplicate SUBSCRIBE is harmless
                    except Exception as e:
                        self.event("sub_error", conn=kind, why=f"{type(e).__name__}: {str(e)[:160]}")
                if add or drop:
                    self.event("hot", add=add, drop=drop, n=len(new), bot=len(bot), surge=[s for s in self.surge.active(ms) if s in new])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event("hot_error", why=f"{type(e).__name__}: {str(e)[:160]}")
            left = (self.fb_rest["paused_until"] - now_ms()) / 1000
            await self._sleep(min(HOT_REFRESH_S, max(5.0, left)) if left > 0 else HOT_REFRESH_S)   # refresh soon after a pause ends

    # ④ full book for the bot's priority coins ───────────────────────────────────────────────────────────────────────────────────
    def full_status(self, ms=None):
        """the health block: priority coins, how many have a synced book, sync / gap / resync counts, today's coverage, REST state."""
        ms = now_ms() if ms is None else ms
        synced = [s for s in self.full if s in self.books and self.books[s].synced
                  and ms - self.books[s].last_emit_ms <= FB_SYNC_FRESH_S * 1000]       # synced = an fbook line in the last 10 s
        d = self.fb_day
        return {"on": self.full_on, "requested": self.full_req, "module": FB is not None, "cap": self.full_max,
                "n_want": self.full_want_n, "n": len(self.full), "synced_n": len(synced),
                "not_synced": [s for s in self.full if s not in synced][:40], "conn": "fbook" in self.hot_ws,
                "silent_now": sum(1 for s in self.full if s in self.books and self.books[s].synced and self.books[s].last_ev_ms is not None
                                  and ms - self.books[s].last_ev_ms > 60_000),
                "totals": dict(self.fb_tot), "day": dict(d),
                "coverage_day_pct": (round(100.0 * d["synced_s"] / d["want_s"], 1) if d["want_s"] else None),
                "buffer_overflow": sum(b.overflow for b in self.books.values()),
                "rest": dict(self.fb_rest, paused_s=max(0, round((self.fb_rest["paused_until"] - ms) / 1000)))}

    def set_full(self, new):
        """apply a new priority list → (added, dropped); books are created / dropped here (subscribe sends happen in full_manager)."""
        add, drop = [s for s in new if s not in self.full], [s for s in self.full if s not in new]
        for s in drop:
            self.books.pop(s, None)
        for s in add:
            self.books[s] = FB.LocalBook(s)
        self.full = list(new)
        return add, drop

    def on_fb_msg(self, raw):
        """one /public/stream message of the full-book connection → its coin's LocalBook (no file write: the emitter writes)."""
        try:
            m = json.loads(raw)
        except ValueError:
            self._fb_count("bad")                    # one bad frame never resets the connection (= every book)
            return
        st = m.get("stream") if isinstance(m, dict) else None
        if not st:
            if isinstance(m, dict) and m.get("error"):
                self.event("sub_error", conn="fbook", why=str(m.get("error"))[:200])
            return
        d = m.get("data") or {}
        bk = self.books.get(d.get("s")) if isinstance(d, dict) else None
        if bk is None:
            return                                   # an unsubscribed coin's last events
        ms = now_ms()
        try:                                         # hold: no snapshot can come during a REST pause → keep 1 event, not a backlog
            self._fb_result(bk.on_event(d, ms, hold=ms < self.fb_rest["paused_until"]))
        except Exception:
            bk.unsync()                              # anything unexpected: THIS coin resyncs, the other books are untouched
            self._fb_count("bad")

    def _fb_result(self, res):
        """count what a LocalBook call returned."""
        k = {"gap": "gaps", "synced": "syncs", "refreshed": "refreshes", "stale": "stale", "bad": "bad"}.get(res)
        if k:
            self._fb_count(k)

    async def fullconn(self):
        async def on_open(ws):
            self.hot_ws["fbook"] = ws
            for s in self.full:                      # a new connection = every book out of sync until its new snapshot
                self.books[s] = FB.LocalBook(s)
            if self.full:
                await self._sub(ws, "SUBSCRIBE", hot_streams(list(self.full), FB.FB_SUFFIX))
        while not self.stop.is_set() and not self.full:              # connect only once there are coins to subscribe
            await self._sleep(2)
        await self._conn_loop("fbook", WS_PUBLIC, on_open, self.on_fb_msg, idle_ok=lambda: not self.full)

    async def full_manager(self):
        """every FULL_REFRESH_S: the bot list → the priority set (pick_full) → subscribe / unsubscribe on the full-book connection."""
        while not self.stop.is_set():
            try:
                ms = now_ms()
                if self.full_fixed:
                    new, self.full_want_n = self.full_fixed[:self.full_max], len(self.full_fixed)
                else:
                    doc = await self._bot_doc()
                    new, self.full_want_n = FB.pick_full(doc, ms, self.full, self.full_seen, self.tradable, self.full_max)
                add, drop = self.set_full(new)
                ws = self.hot_ws.get("fbook")
                if ws is not None:
                    try:
                        if drop:
                            await self._sub(ws, "UNSUBSCRIBE", hot_streams(drop, FB.FB_SUFFIX))
                        if add:
                            await self._sub(ws, "SUBSCRIBE", hot_streams(add, FB.FB_SUFFIX))
                    except Exception as e:
                        self.event("sub_error", conn="fbook", why=f"{type(e).__name__}: {str(e)[:160]}")
                if add or drop:
                    self.event("full", add=add, drop=drop, n=len(new), n_want=self.full_want_n, cap=self.full_max)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event("full_error", why=f"{type(e).__name__}: {str(e)[:160]}")
            await self._sleep(FULL_REFRESH_S)

    def _get_depth(self, sym):
        """GET /fapi/v1/depth (limit 1000) → (HTTP status, headers dict, body or None). Never raises."""
        return self._rest(f"/fapi/v1/depth?symbol={sym}&limit={FB.FB_LIMIT}")

    async def _in_pool(self, which, fn, *args):
        """run fn(*args) on the snapshotter's own thread ('snap') or the sweeps' threads ('sweep')."""
        if which == "snap":
            if self.snap_pool is None:
                self.snap_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fb-snap")
            pool = self.snap_pool
        else:
            if self.sweep_pool is None:
                self.sweep_pool = ThreadPoolExecutor(max_workers=PS.CONCURRENCY + 1 if PS else 2, thread_name_prefix="pos")
            pool = self.sweep_pool
        return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)

    def _rest(self, path):
        """GET REST + path over a KEEP-ALIVE HTTPS connection (one per worker thread; from eu-west-1 a new TLS handshake to Binance
        costs more than the request) → (HTTP status, headers dict, body or None); status 0 = network error. Never raises."""
        for attempt in (0, 1):
            conn = getattr(_tls, "conn", None)
            if conn is None:
                conn = _tls.conn = http.client.HTTPSConnection(REST_HOST, timeout=FB_REST_TIMEOUT_S)
            try:
                conn.request("GET", path, headers={"User-Agent": "route-b-recorder", "Connection": "keep-alive"})
                r = conn.getresponse()
                raw = r.read()
                hdr = dict(r.getheaders())
                if r.will_close:
                    conn.close()
                    _tls.conn = None
            except (http.client.HTTPException, OSError):
                try:
                    conn.close()
                except Exception:
                    pass
                _tls.conn = None
                if attempt:                                          # a stale kept-alive socket gets ONE fresh retry
                    return 0, {}, None
                continue
            if r.status != 200:
                return r.status, hdr, None
            try:
                return 200, hdr, json.loads(raw.decode())
            except ValueError:
                return 200, hdr, None
        return 0, {}, None

    def next_snapshot(self, ms):
        """the priority coin to snapshot next: of those that need one, the one asked LONGEST ago (round-robin — a coin that keeps
        failing or resyncing never starves the others); None when none."""
        todo = [s for s in self.full if s in self.books and self.books[s].needs_snapshot(ms)]
        return min(todo, key=lambda x: (self.books[x].last_req_ms, self.full.index(x))) if todo else None

    async def snapshotter(self):
        """one REST snapshot at a time for the first priority coin that needs one, inside the weight budget; 418 / 429 / high used weight
        stop every snapshot for the time rest_pause_s() says (logged)."""
        while not self.stop.is_set():
            try:
                ms = now_ms()
                if "fbook" not in self.hot_ws or ms < self.fb_rest["paused_until"]:
                    await self._sleep(1)
                    continue
                sym = self.next_snapshot(ms)
                if sym is None or not self.fb_bucket.take(FB.FB_SNAP_WEIGHT, time.time()):
                    await self._sleep(0.25)
                    continue
                self.books[sym].mark_requested(ms)
                status, headers, body = await self._in_pool("snap", self._get_depth, sym)
                self.fb_rest["snapshots"] += 1
                self.fb_rest["weight"] += FB.FB_SNAP_WEIGHT
                self._rest_result(status, headers)
                bk = self.books.get(sym)
                if bk is None:
                    continue                                         # dropped from the set while the request ran
                if status != 200 or body is None:
                    self.fb_rest["errors"] += 1
                    self._fb_count("snap_err")
                    bk.request_failed(now_ms())                      # this coin backs off (5 s … 5 min); the others go on
                    continue
                self._fb_result(bk.on_snapshot(body, now_ms()))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.fb_rest["errors"] += 1
                self.event("snapshot_error", why=f"{type(e).__name__}: {str(e)[:160]}")
                await self._sleep(1)

    def fb_tick(self, ms):
        """one emitter pass: per synced coin the fbook line (+ the fsnap line every FB_SNAP_S), coverage counters, crossed books → resync,
        a book drifting toward its edge (or an hour old) → refresh. Returns lines written."""
        conn = "fbook" in self.hot_ws
        n = 0
        self.fb_day["want_s"] += len(self.full)
        self.fb_tot["want_s"] += len(self.full)
        for s in list(self.full):
            bk = self.books.get(s)
            if bk is None or not bk.synced or not conn:
                continue
            if bk.stale(ms):                                         # its diff stream went silent: never emit a frozen book
                bk.unsync()
                self._fb_count("silent")
                continue
            snap_due = bk.snap_due(ms)
            m = FB.book_metrics(bk.bids, bk.asks, bk.edge_bid, bk.edge_ask, with_levels=snap_due)
            if m is None:
                if FB.crossed(bk.bids, bk.asks):
                    bk.unsync()
                    self._fb_count("crossed")
                elif bk.begin_refresh(ms):                           # a side emptied inside the trusted region (price beyond the
                    self._fb_count("empty")                          # edge): re-snapshot — gated by the refresh backoff
                continue
            self.w.write("fbook", FB.fbook_record(bk, m, ms), ms)
            bk.last_emit_ms = ms
            n += 1
            if snap_due:
                self.w.write("fsnap", FB.fsnap_record(bk, m, ms), ms)
                bk.mark_snap(ms)
                self._count("fsnap")
            if bk.refresh_due(ms, min(m["rb"], m["ra"])):
                bk.begin_refresh(ms)
        if n:
            self.fb_day["synced_s"] += n
            self.fb_tot["synced_s"] += n
            self.last["fbook"] = ms
            self.day_n["fbook"] += n
        return n

    async def fb_emitter(self):
        nxt = time.time()
        while not self.stop.is_set():
            try:
                self.fb_tick(now_ms())
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event("fbook_error", why=f"{type(e).__name__}: {str(e)[:160]}")
            nxt += FB.FB_EMIT_S
            if nxt < time.time():                                    # fell behind (a slow pass): skip, never burst
                nxt = time.time() + FB.FB_EMIT_S
            await self._sleep(max(0.0, nxt - time.time()))

    # ⑤ open interest + positioning ─────────────────────────────────────────────────────────────────────────────────────────────
    def pos_status(self, ms=None):
        ms = now_ms() if ms is None else ms
        p, d = dict(self.pos), self.pos_day
        p.update(on=self.pos_on, module=PS is not None, off=self.pos_off, oi_age_s=(None if p["oi_last_ms"] is None else round((ms - p["oi_last_ms"]) / 1000)),
                 ratio_age_s=(None if p["ratio_last_ms"] is None else round((ms - p["ratio_last_ms"]) / 1000)), day=dict(d),
                 oi_coverage_day_pct=(round(100.0 * d["oi_ok"] / d["oi_want"], 1) if d["oi_want"] else None),
                 ratio_coverage_day_pct=(round(100.0 * d["ratio_ok"] / d["ratio_want"], 1) if d["ratio_want"] else None),
                 ratio_paused_s=max(0, round((self.ratio_paused_until - ms) / 1000)),
                 ratio_5m=(self.ratio_win.count(time.time()) if self.ratio_win else 0))
        return p

    def _rest_result(self, status, headers):
        """bookkeeping of a WEIGHT-counted REST response (depth snapshots, open interest): used weight · 418 / 429 / high used weight →
        pause every weight-counted read (rest_pause_s). → the pause in seconds."""
        h = {str(k).lower(): v for k, v in (headers or {}).items()}
        if h.get("x-mbx-used-weight-1m"):
            self.fb_rest["used_1m"] = h["x-mbx-used-weight-1m"]
        if status == 0:
            return 0.0                                               # network error: no answer from Binance, nothing to obey
        try:
            pause = float(rest_pause_s(status, headers, time.time()))
        except Exception:
            pause = 600.0 if status == 418 else 120.0 if status == 429 else 0.0
        if not math.isfinite(pause):
            pause = 600.0
        pause = min(max(0.0, pause), MAX_PAUSE_MS / 1000)
        if pause > 0:
            until = now_ms() + int(pause * 1000)
            if until > self.fb_rest["paused_until"]:
                self.fb_rest["paused_until"] = until
                self.fb_rest["pause_why"] = FB.pause_reason(status, headers) if FB else status   # the CURRENT pause's cause
        if status in (418, 429):
            self.fb_rest["n418" if status == 418 else "n429"] += 1
            self.event("rest_limit", status=status, pause_s=round(pause), used_1m=self.fb_rest["used_1m"])
            self._save_pause()
        return pause

    async def oi_sweep(self, ms):
        """one open-interest pass over the priority + hot coins (PS.CONCURRENCY requests at a time) → one 'oi' line. Stops asking on a
        REST pause. → rows written."""
        coins = PS.order_coins(self.full, self.hot, PS.OI_MAX_COINS)
        minute = ms // 60_000
        got, t0, sem = {}, time.time(), asyncio.Semaphore(PS.CONCURRENCY)

        async def one(s):
            async with sem:
                if self.stop.is_set() or now_ms() < self.fb_rest["paused_until"]:
                    return                                           # not asked → counted as skipped below
                status, headers, body = await self._in_pool("sweep", self._rest, PS.OI_PATH.format(s))
                self._rest_result(status, headers)
                got[s] = PS.parse_oi(body) if status == 200 else None
                await asyncio.sleep(PS.REQ_GAP_S)
        res = await asyncio.gather(*(one(s) for s in coins), return_exceptions=True)
        for s, r in zip(coins, res):                                 # a worker that raised = that coin unread (counted), others go on
            if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                got.setdefault(s, None)
        rows = []
        for s in coins:                                              # priority order kept
            v = got.get(s)
            if v is not None:
                c1, c5 = self.oi_track.feed(s, v[0], minute)
                rows.append([s, v[0], v[1], c1, c5])
        err = sum(1 for s in coins if s in got and got[s] is None)  # asked, answered badly (a 400 delisted coin, 5xx): per coin
        skipped = sum(1 for s in coins if s not in got)              # never asked (a REST pause / shutdown)
        self.oi_track.prune(set(coins))
        if rows:                                                     # file hour = NOW (a sweep crossing the hour never reopens an
            self.w.write("oi", {"r": ms, "m": rows})                 # uploaded file); "r" keeps the sweep's minute
            self._count("oi")
        self.pos.update(oi_sweeps=self.pos["oi_sweeps"] + 1, oi_coins=len(coins), oi_ok=len(rows), oi_skipped=skipped,
                        oi_err=self.pos["oi_err"] + err, oi_sweep_s=round(time.time() - t0, 1))
        if rows:
            self.pos["oi_last_ms"] = ms
        self.pos_day["oi_want"] += len(coins)
        self.pos_day["oi_ok"] += len(rows)
        return len(rows)

    async def ratio_sweep(self, ms):
        """one positioning pass: 4 endpoints × the coins the 5-min budget allows (PS.CONCURRENCY coins at a time) → one 'ratio' line.
        A /futures/data 429 pauses only these reads; a 418 pauses every REST read. → rows written."""
        coins = PS.order_coins(self.full, self.hot, PS.OI_MAX_COINS)
        take, skipped = PS.ratio_coins(coins, self.ratio_win.count(time.time()))
        got, sem, halt = {}, asyncio.Semaphore(PS.CONCURRENCY), [False]

        def blocked():
            return halt[0] or self.stop.is_set() or now_ms() < self.ratio_paused_until or \
                (now_ms() < self.fb_rest["paused_until"] and self.fb_rest["pause_why"] == 418)   # /fapi weight limits are not ours

        async def one(s):
            async with sem:
                parsed = {}
                for key, ep, field in PS.RATIO_ENDPOINTS:
                    if blocked():
                        break
                    status, headers, body = await self._in_pool("sweep", self._rest, PS.ratio_path(ep, s))
                    self.ratio_win.add(time.time())
                    if status == 429:                                # the /futures/data limit (its own) → pause only these
                        self.pos["ratio_n429"] += 1
                        pause = min(max(rest_pause_s(429, headers, time.time()), PS.RATIO_PAUSE_429_S), MAX_PAUSE_MS / 1000)
                        self.ratio_paused_until = max(self.ratio_paused_until, now_ms() + int(pause * 1000))
                        self.event("rest_limit", status=429, conn="ratio", pause_s=round(pause))
                        self._save_pause()
                        halt[0] = True
                        break
                    if status == 418:                                # an IP ban stops every REST read
                        self._rest_result(status, headers)
                        halt[0] = True
                        break
                    parsed[key] = PS.parse_ratio(body, field) if status == 200 else None
                    await asyncio.sleep(PS.REQ_GAP_S)
                if parsed:
                    got[s] = PS.ratio_row(s, parsed)
        res = await asyncio.gather(*(one(s) for s in take), return_exceptions=True)
        for s, r in zip(take, res):
            if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                got.setdefault(s, None)
        rows = [got[s] for s in take if got.get(s) is not None]
        err = sum(1 for s in take if s in got and got[s] is None)
        if rows:
            self.w.write("ratio", {"r": ms, "m": rows})              # file hour = NOW (see oi_sweep)
            self._count("ratio")
            self.pos["ratio_last_ms"] = ms                           # only a sweep that READ coins counts as the last one
        self.pos.update(ratio_sweeps=self.pos["ratio_sweeps"] + 1, ratio_tried_ms=ms, ratio_coins=len(coins), ratio_ok=len(rows),
                        ratio_err=self.pos["ratio_err"] + err, ratio_skipped=skipped, ratio_unread=len(take) - len(rows))
        self.pos_day["ratio_want"] += len(coins)
        self.pos_day["ratio_ok"] += len(rows)
        return len(rows)

    async def _every(self, period_s, fn, name, offset_s=0):
        """run fn(sweep ms) at the start of every period (UTC-aligned + offset); a slow sweep skips the missed slot. Never dies."""
        while not self.stop.is_set():
            t = time.time()
            nxt = (t - offset_s) // period_s * period_s + period_s + offset_s
            await self._sleep(nxt - t)
            if self.stop.is_set():
                break
            try:
                if self.hot or self.full:
                    await fn(now_ms())
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event(f"{name}_error", why=f"{type(e).__name__}: {str(e)[:160]}")

    async def oi_poller(self):
        await self._every(PS.OI_EVERY_S, self.oi_sweep, "oi", offset_s=2)

    async def ratio_poller(self):
        await self._every(PS.RATIO_EVERY_S, self.ratio_sweep, "ratio", offset_s=20)   # the 5m row is published a little after the close

    async def _aws(self, *args, capture=False):
        """run the aws CLI with a hard timeout → (ok, stdout bytes, stderr text). Never raises."""
        try:
            p = await asyncio.create_subprocess_exec("aws", *args, stdout=(asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL),
                                                     stderr=asyncio.subprocess.PIPE)
        except Exception as e:
            return False, b"", f"{type(e).__name__}: {e}"
        try:
            out, err = await asyncio.wait_for(p.communicate(), S3_TIMEOUT_S)
        except asyncio.TimeoutError:
            try:
                p.kill()
                await p.wait()
            except Exception:
                pass
            return False, b"", f"timeout after {S3_TIMEOUT_S} s"
        return p.returncode == 0, (out or b""), (err or b"").decode(errors="replace")[:200]

    async def _bot_list(self):
        """the bot's list from S3 (None when unreadable / no bucket)."""
        if not self.bucket:
            return None
        cli = self._boto()
        if cli is not None:                                          # in-process: no aws-CLI process per read
            try:
                body = await self._in_pool("sweep", lambda: cli.get_object(Bucket=self.bucket, Key="bot/frenzy_list.json")["Body"].read())
                return json.loads(body.decode())
            except Exception:
                return None
        ok, out, _ = await self._aws("s3", "cp", "--only-show-errors", f"s3://{self.bucket}/bot/frenzy_list.json", "-", capture=True)
        if not ok:
            return None
        try:
            return json.loads(out.decode())
        except ValueError:
            return None

    async def _sleep(self, s):
        try:
            await asyncio.wait_for(self.stop.wait(), s)
        except asyncio.TimeoutError:
            pass

    # files, health, uploads
    def backlog_age_s(self):
        """age (s) of the oldest CLOSED-hour file still waiting for upload — 0 when none. Never raises."""
        try:
            cd, ch = hour_key(now_ms())
            files = [x for x in closed_files(self.dir, cd, ch) if x[3] not in self.w.open_paths()]
            if not files:
                return 0
            return round(time.time() - min(os.path.getmtime(x[3]) for x in files), 0)
        except Exception:
            return None

    def status(self):
        ms = now_ms()
        try:
            free = round(shutil.disk_usage(self.dir).free / 1e9, 2)
        except Exception:
            free = None
        return dict(now=ms, host=self.host, started=self.started, hot_n=len(self.hot), hot=self.hot, bot_n=self.bot_n,
                    last=dict(self.last), age_s={k: (None if v is None else round((ms - v) / 1000, 1)) for k, v in self.last.items()},
                    reconnects=self.reconnects, day=self.day, day_n=self.day_n, day_gap_s={k: round(v, 1) for k, v in self.day_gap_s.items()},
                    day_hot_unique=len(self.day_hot), bytes_raw_total=self.w.bytes, disk_free_gb=free, writes_paused=self.w.paused,
                    writes_dropped=self.w.dropped, uploaded=self.uploaded, upload_fail=self.upload_fail, housekeeping_errors=self.hk_errors,
                    features_done=self.feat_done, features_fail=self.feat_fail, features_last=self.feat_last, features_build_s=self.feat_build_s,
                    upload_backlog_age_s=self.backlog_age_s(), full=self.full_status(ms), pos=self.pos_status(ms),
                    note="day_* counters restart with the process (a mid-day restart under-counts that day)")

    async def _s3_put(self, src, key):
        ok, _, err = await self._aws("s3", "cp", "--only-show-errors", src, f"s3://{self.bucket}/{key}")
        if not ok:
            self.event("upload_error", key=key, why=err)
        return ok

    async def _upload(self):
        """the status first (health stays visible), then ≤ UPLOAD_MAX_FILES closed-hour files (oldest first), then daily summaries."""
        sp = os.path.join(self.dir, "status.json")
        if os.path.exists(sp):
            await self._s3_put(sp, "health/status.json")
        cd, ch = hour_key(now_ms())
        self.w.roll(cd, ch)                                          # same hour key as the pick below (no turn of the hour in between)
        busy = self.w.open_paths()
        ready = [x for x in closed_files(self.dir, cd, ch) if x[3] not in busy]
        groups = hour_groups(ready)
        held = set()
        for key in sorted(groups):                                   # 📊 raw files wait for their hour's summary (built by featurer())
            if not self._summary_settled(key, groups[key]):
                held.update(p for k, ps in groups[key].items() if k in FEATURE_KINDS for p in ps)
        todo = sorted((x for x in ready if x[3] not in held), key=lambda x: (x[1], x[2]))[:UPLOAD_MAX_FILES]
        for kind, day, fn, path in todo:                             # never a file a writer still holds open
            if await self._s3_put(path, f"data/{kind}/{day}/{fn}"):
                os.remove(path)
                self.uploaded += 1
            else:
                self.upload_fail += 1
        for f in sorted(x for x in os.listdir(self.dir) if x.startswith("daily_") and x.endswith(".json")):
            if await self._s3_put(os.path.join(self.dir, f), f"health/daily/{f[6:]}"):
                os.remove(os.path.join(self.dir, f))

    def _fpaths(self, key):
        day, hh = key
        fdir = os.path.join(self.dir, "_features", day)
        return fdir, os.path.join(fdir, f"{hh}.done"), os.path.join(fdir, f"{hh}.csv.gz")

    def _summary_settled(self, key, kinds):
        """True when the hour's raw files may leave the disk: summary uploaded (marker), nothing to summarise, or the BUILD gave up."""
        if not any(kinds.get(k) for k in FEATURE_KINDS):
            return True
        return os.path.exists(self._fpaths(key)[1]) or self.feat_tries.get(key, 0) >= FEATURE_MAX_TRIES

    async def _features(self, key, kinds):
        """(date, HH) → build features/<date>/<HH>.csv.gz once (low CPU priority) and upload it; a marker file remembers it. A failed
        BUILD counts toward FEATURE_MAX_TRIES (then the hour goes raw-only, rebuildable from S3); a failed UPLOAD keeps the built file and
        is simply retried next pass. Without a bucket (local run) the file is kept. → True when settled. Never raises."""
        try:
            day, hh = key
            fdir, marker, out = self._fpaths(key)
            if self._summary_settled(key, kinds):
                return True
            os.makedirs(fdir, exist_ok=True)
            if not os.path.exists(out):
                args = (["nice", "-n", "10"] if shutil.which("nice") else []) + [
                    sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.py"), out + ".tmp.gz"]
                for k in FEATURE_KINDS:
                    if kinds.get(k):
                        args += [f"--{k}"] + sorted(kinds[k])
                ok, why, t0 = False, "", time.time()
                try:
                    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
                    try:
                        _, err = await asyncio.wait_for(p.communicate(), FEATURE_TIMEOUT_S)
                        ok, why = p.returncode == 0, (err or b"").decode(errors="replace")[-200:]
                    except asyncio.TimeoutError:
                        try:
                            p.kill()
                            await p.wait()
                        except Exception:
                            pass
                        why = f"timeout after {FEATURE_TIMEOUT_S} s"
                    except asyncio.CancelledError:          # shutdown mid-build: never leave the child running
                        try:
                            p.kill()
                        except Exception:
                            pass
                        raise
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    why = f"{type(e).__name__}: {str(e)[:160]}"
                self.feat_build_s = round(time.time() - t0, 1)
                if not ok:
                    self.feat_tries[key] = self.feat_tries.get(key, 0) + 1
                    self.feat_fail += 1
                    self.event("features_error", hour=f"{day} {hh}", tries=self.feat_tries[key], why=why)
                    return self._summary_settled(key, kinds)
                os.replace(out + ".tmp.gz", out)
            if not self.bucket:
                open(marker, "w").close()                            # local run: keep the summary file
                self.feat_done += 1
                self.feat_last = f"{day} {hh}"
                return True
            if await self._s3_put(out, f"features/{day}/{hh}.csv.gz"):
                open(marker, "w").close()
                os.remove(out)
                self.feat_done += 1
                self.feat_last = f"{day} {hh}"
                return True
            return False                                             # upload failed: retried next pass (the built file is kept)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.feat_fail += 1
            self.event("features_error", hour=str(key), why=f"{type(e).__name__}: {str(e)[:160]}")
            return False

    def _prune_features(self, keep_days=3):
        """drop _features/<day> folders (markers + leftovers) older than keep_days — their raw hours are long gone. Never raises."""
        try:
            root = os.path.join(self.dir, "_features")
            cut = time.strftime("%Y-%m-%d", time.gmtime(time.time() - keep_days * 86400))
            for d in os.listdir(root):
                if d < cut:
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            for k in [k for k in self.feat_tries if k[0] < cut]:
                del self.feat_tries[k]
        except Exception:
            pass

    async def featurer(self):
        """📊 the hourly per-minute summaries, OFF the housekeeping path: one closed hour per pass, oldest first, then a short pause."""
        while not self.stop.is_set():
            try:
                cd, ch = hour_key(now_ms())
                self.w.roll(cd, ch)                                  # close the past hour's handles first (as _upload does)
                busy = self.w.open_paths()
                groups = hour_groups([x for x in closed_files(self.dir, cd, ch) if x[3] not in busy])
                pending = [k for k in sorted(groups) if not self._summary_settled(k, groups[k])]
                if pending:
                    await self._features(pending[0], groups[pending[0]])
                    await self._sleep(5)
                    continue
                self._prune_features()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event("featurer_error", why=f"{type(e).__name__}: {str(e)[:160]}")
            await self._sleep(60)

    def _write_json(self, name, obj):
        p = os.path.join(self.dir, name)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, p)

    async def housekeeping(self):
        last_up = last_st = 0.0
        while not self.stop.is_set():
            await self._sleep(FLUSH_S)
            try:
                self.w.flush()
                self.w.roll()
                try:
                    free = shutil.disk_usage(self.dir).free / 1e9
                except Exception:
                    free = None
                if free is not None and (free < MIN_FREE_GB) != self.w.paused:
                    self.w.paused = free < MIN_FREE_GB
                    if not self.w.paused:
                        self.event("writes_resumed", disk_free_gb=round(free, 2))
                d = hour_key(now_ms())[0]
                if d != self.day:                                    # the UTC day changed → the daily summary
                    self._write_json(f"daily_{self.day}.json", dict(self.status(), day=self.day, summary_at=now_ms()))
                    self.day, self.day_n = d, {k: 0 for k in KINDS}
                    self.day_gap_s, self.day_hot = {k: 0.0 for k in self.day_gap_s}, set(self.hot)
                    self.fb_day = self._fb_zero()
                    self.pos_day = {k: 0 for k in self.pos_day}
                if time.time() - last_st >= STATUS_S:
                    last_st = time.time()
                    self._write_json("status.json", self.status())
                if self.bucket and time.time() - last_up >= UPLOAD_S:
                    last_up = time.time()
                    await self._upload()
            except asyncio.CancelledError:
                raise
            except Exception as e:                                   # never let housekeeping die (it is what frees the disk)
                self.hk_errors += 1
                self.event("housekeeping_error", why=f"{type(e).__name__}: {str(e)[:160]}")

    async def run(self):
        os.makedirs(self.dir, exist_ok=True)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        for b in BAD_ENV:
            self.event("bad_env", **b)
        self.event("start", bucket=bool(self.bucket), max_hot=self.max_hot, full_max=self.full_max, full_on=self.full_on, pos_on=self.pos_on)
        coros = [self.market(), self.hotconn("book"), self.hotconn("trade"), self.hot_manager(), self.housekeeping(), self.featurer()]
        if self.full_on:
            coros += [self.fullconn(), self.full_manager(), self.snapshotter(), self.fb_emitter()]
        if self.pos_on:
            coros += [self.oi_poller(), self.ratio_poller()]
        tasks = [asyncio.create_task(c) for c in coros]
        if self.stop_at:
            try:
                await asyncio.wait_for(self.stop.wait(), max(0.0, self.stop_at - time.time()))
            except asyncio.TimeoutError:
                self.stop.set()
        else:
            done, _ = await asyncio.wait([asyncio.create_task(self.stop.wait())] + tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:                                           # a task that died unexpectedly → log, then exit (systemd restarts)
                if t in tasks and t.exception() is not None:
                    self.event("task_died", why=f"{type(t.exception()).__name__}: {str(t.exception())[:160]}")
            self.stop.set()
        await asyncio.sleep(0.5)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for pool in (self.snap_pool, self.sweep_pool):
            if pool is not None:
                pool.shutdown(wait=False)
        self.event("stop")
        self.w.close()                                               # clean gzip trailers
        try:
            self._write_json("status.json", self.status())
        except Exception:
            pass


BAD_ENV = []


def env_int(name, default):
    """an int env var; a malformed one falls back to the default (logged as a 'bad_env' event at start) — never a crash loop."""
    v = os.environ.get(name, "")
    if not v.strip():
        return default
    try:
        return max(0, int(float(v)))
    except (TypeError, ValueError, OverflowError):
        BAD_ENV.append({"name": name, "value": v[:40], "used": default})
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.0)
    a = ap.parse_args()
    rec = Recorder(os.environ.get("DATA_DIR", "/var/lib/scalpars-recorder"), os.environ.get("BUCKET", ""),
                   env_int("MAX_HOT", 120), (time.time() + a.seconds) if a.seconds > 0 else None,
                   full_max=env_int("FULL_MAX", None),
                   full_syms=[x.strip().upper() for x in os.environ.get("FULL_SYMS", "").split(",") if x.strip()])
    asyncio.run(rec.run())


if __name__ == "__main__":
    main()
