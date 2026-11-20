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
      HOT list (≤ MAX_HOT), refreshed every 5 min, in this priority: BTC + ETH · the BOT's own list (s3://<BUCKET>/bot/frenzy_list.json,
      written by the bot during its FRENZY pass: open positions · FRENZY-flagged episodes · episodes EXHAUSTION / FLATBASE still track ·
      every pair FRENZY follows; ignored when > 30 min old) · coins already in (≥ 60 min, no churn) · SURGE coins (|move vs ≥ 30 min ago|
      ≥ 3 %, from stream ①, kept 6 h — the backup when the bot list is stale) · top 24 h movers (quote volume ≥ $5M) · top 24 h volume.
      REST: the 24 h ticker every 5 min (weight 40) + exchangeInfo once a day (weight 1).
FILES <DATA_DIR>/<kind>/<UTC date>/<HH>-<process start ms>.jsonl.gz (kinds: liq · mark · book · trade · events), one JSON object per
      line {"r": local receive ms, ...payload}. ONE file per process and hour (a restart never appends to a file a killed process left
      half-written). A CLOSED hour's files go to s3://<BUCKET>/data/<kind>/<date>/<same name> and are deleted locally only after a
      successful upload. A reader should still tolerate a truncated last gzip member (power loss).
HEALTH status.json every 60 s (uploaded every 5 min → s3://<BUCKET>/health/status.json) · daily summary at the UTC day change →
      health/daily/<date>.json · every disconnect / close / reconnect / start / stop → the events kind (the gap log).
SAFETY SIGTERM / SIGINT close the files cleanly · a read silent for READ_TIMEOUT_S reconnects (half-open sockets) · every disconnect —
      clean close included — backs off · below MIN_FREE_GB of disk the recorder stops WRITING but keeps uploading (frees the disk).
Env   BUCKET (empty → no upload / no bot list: local test) · DATA_DIR (default /var/lib/scalpars-recorder) · MAX_HOT (default 120).
Usage python3 recorder.py [--seconds N]   (N > 0 → stop after N seconds: a local smoke test)"""
import argparse
import asyncio
import gzip
import json
import os
import shutil
import signal
import socket
import time
import sys
import urllib.request
from collections import deque

try:                                              # websockets ≥ 13: the new asyncio client
    from websockets.asyncio.client import connect as ws_connect
except Exception:                                 # pragma: no cover — older websockets
    from websockets import connect as ws_connect

WS_MARKET = "wss://fstream.binance.com/market/stream"   # Binance's 2025 split (checked live 2026-11-19): aggTrade / markPrice / forceOrder
WS_PUBLIC = "wss://fstream.binance.com/public/stream"   # … and the order book (depth); the legacy /stream no longer carries the /market ones
REST = "https://fapi.binance.com"
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
KINDS = ("liq", "mark", "book", "trade", "events")
FEATURE_KINDS = ("book", "trade", "liq", "mark")   # the inputs of features.py (events are not)
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

    def write(self, kind, obj, ms=None):
        if self.paused:
            self.dropped += 1
            return
        ms = now_ms() if ms is None else ms
        d, h = hour_key(ms)
        cur = self.fh.get(kind)
        if cur is None or cur[0] != d or cur[1] != h:
            if cur is not None:
                cur[2].close()
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
                    del self.fh[k]

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
    def __init__(self, data_dir, bucket, max_hot, stop_at=None):
        self.dir, self.bucket, self.max_hot, self.stop_at = data_dir, bucket, max_hot, stop_at
        self.started = now_ms()
        self.w = Writer(data_dir, self.started)
        self.surge = Surge()
        self.hot, self.added = [], {}
        self.tradable = None
        self.last = {k: None for k in KINDS}
        self.reconnects = {"market": 0, "book": 0, "trade": 0}
        self.host = socket.gethostname()
        self.hot_ws = {}       # kind → the live websocket (book / trade)
        self.day = hour_key(self.started)[0]
        self.day_n = {k: 0 for k in KINDS}
        self.day_gap_s = {"market": 0.0, "book": 0.0, "trade": 0.0}
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
        self.stop = asyncio.Event()

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

    async def _conn_loop(self, name, url, on_open, on_msg):
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
                        on_msg(await self._read(ws))
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

    def _get(self, path):
        with urllib.request.urlopen(urllib.request.Request(REST + path, headers={"User-Agent": "route-b-recorder"}), timeout=20) as r:
            return json.loads(r.read().decode())

    async def hot_manager(self):
        last_info = 0
        while not self.stop.is_set():
            try:
                if time.time() - last_info > INFO_REFRESH_S or self.tradable is None:
                    info = await asyncio.to_thread(self._get, "/fapi/v1/exchangeInfo")
                    self.tradable = {s["symbol"] for s in info.get("symbols", []) if s.get("contractType") == "PERPETUAL"
                                     and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"}
                    last_info = time.time()
                tick = await asyncio.to_thread(self._get, "/fapi/v1/ticker/24hr")
                doc = await self._bot_list()
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
            await self._sleep(HOT_REFRESH_S)

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
                    upload_backlog_age_s=self.backlog_age_s(),
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
        self.event("start", bucket=bool(self.bucket), max_hot=self.max_hot)
        tasks = [asyncio.create_task(c) for c in (self.market(), self.hotconn("book"), self.hotconn("trade"), self.hot_manager(), self.housekeeping(), self.featurer())]
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
        self.event("stop")
        self.w.close()                                               # clean gzip trailers
        try:
            self._write_json("status.json", self.status())
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.0)
    a = ap.parse_args()
    rec = Recorder(os.environ.get("DATA_DIR", "/var/lib/scalpars-recorder"), os.environ.get("BUCKET", ""),
                   int(os.environ.get("MAX_HOT", "120")), (time.time() + a.seconds) if a.seconds > 0 else None)
    asyncio.run(rec.run())


if __name__ == "__main__":
    main()
