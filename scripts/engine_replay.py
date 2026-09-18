#!/usr/bin/env python3
"""ENGINE-CODE REPLAY — runs the REAL TradingEngine (services/trading_engine.py) over
historical klines with a simulated clock. Nothing in the signal / gate / sizing / exit
code is reimplemented: the exchange layer (binance_service) is replaced by a kline
server that answers from reports/backtest_cache, the WebSocket price feed is replaced
by 1-minute bar ticks, and every wall-clock read inside the engine is redirected to
the simulated clock.

Usage:
  venv/bin/python scripts/engine_replay.py --start 2026-01-01 --end 2026-02-01 --tag jan
Outputs (reports/backtest_cache/replay/<tag>_*.csv|json):
  orders (every Order column), monitor periods, filter-block counters, run meta.

Fidelity notes (declared, see reports/ENGINE_REPLAY_METHOD.md):
  * scan cadence = --scan-step seconds (live ≈ 30-95 s per full cycle); each scan's
    batch delays advance the clock exactly like the live OHLCV_BATCH_DELAY sleeps.
  * the in-progress 5m candle is rebuilt from 1m bars for pairs with 1m data (the
    top-50 scan universe); scanner-only pairs are evaluated on completed candles.
  * exits: every open position receives 4 ticks per minute (open, adverse extreme,
    favourable extreme, close) through the SAME realtime stop-loss path as live.
  * paper maker entry: fills at the limit if the 1m bar touches it, else taker
    fallback at that minute's close (live: 20 s WebSocket window).
  * no funding (paper live does not charge it either); fees exactly as the engine.
"""
import os, sys, argparse, asyncio, json, math, glob, types, logging, time as _realtime
from datetime import datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "reports", "backtest_cache")
OUT = os.path.join(CACHE, "replay")
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True)            # first day whose entries COUNT (UTC)
ap.add_argument("--end", required=True)              # exclusive
ap.add_argument("--tag", required=True)
ap.add_argument("--warm-days", type=float, default=3.0)
ap.add_argument("--follow-hours", type=float, default=24.0)
ap.add_argument("--scan-step", type=int, default=120)   # seconds between scan cycles
ap.add_argument("--balance", type=float, default=5000.0)
ap.add_argument("--log", default="WARNING")
ap.add_argument("--db-dir", default=None)
ap.add_argument("--no-scanner", action="store_true")
ap.add_argument("--tick-steps", type=int, default=6)      # interpolation points per intra-minute leg (0 = 4 raw ticks)
ap.add_argument("--no-persist", action="store_true")
ap.add_argument("--tick-order", default="random", choices=["adverse", "favourable", "random"])
ap.add_argument("--tick-mode", default="stepped", choices=["stepped", "wickstop"])
ap.add_argument("--ticks", action="store_true")
ap.add_argument("--config", default=None)   # replay with a HISTORICAL trading_config.json (era-config calibration); fields missing in it take today's defaults   # use real aggTrades (reports/backtest_cache/ticks) for open positions + maker window  # wickstop: trails/locks see open→close only; the adverse extreme is fed ONLY if it breaches the hard stop  # intra-minute extreme ordering (unknowable from 1m bars)     # disable the maker-window persistence emulation
A = ap.parse_args()

def _ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
START_MS, END_MS = _ms(A.start), _ms(A.end)
WARM_MS = START_MS - int(A.warm_days * 86400_000)
FOLLOW_MS = END_MS + int(A.follow_hours * 3600_000)
DAY = 86400_000

db_dir = A.db_dir or os.path.join(OUT, "db")
os.makedirs(db_dir, exist_ok=True)
DB_PATH = os.path.join(db_dir, f"replay_{A.tag}.db")
for f in glob.glob(DB_PATH + "*"):
    os.remove(f)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{DB_PATH}"
os.environ["DEBUG"] = "false"

logging.basicConfig(level=getattr(logging, A.log.upper()), format="%(levelname)s %(name)s: %(message)s")
for _n in ("sqlalchemy", "aiosqlite", "asyncio"):
    logging.getLogger(_n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import random as _random
_RNG = _random.Random(20260917)

# ---------------------------------------------------------------- simulated clock
class Clock:
    ms = WARM_MS

class FakeDatetime(datetime):
    @classmethod
    def utcnow(cls):
        return datetime(1970, 1, 1) + timedelta(milliseconds=Clock.ms)
    @classmethod
    def now(cls, tz=None):
        d = (datetime(1970, 1, 1) + timedelta(milliseconds=Clock.ms)).replace(tzinfo=timezone.utc)
        return d.astimezone(tz) if tz else d.replace(tzinfo=None)

class FakeTime(types.ModuleType):
    def __getattr__(self, n):
        return getattr(_realtime, n)
    def time(self):
        return Clock.ms / 1000.0
FAKE_TIME = FakeTime("time")

async def fake_sleep(s=0, *a, **k):
    if s and s > 0:
        Clock.ms += int(s * 1000)
    await asyncio.sleep(0)

# ---------------------------------------------------------------- config + engine import
import config
config.save_trading_config = lambda *_a, **_k: True   # a replay must NEVER write the live trading_config.json (spike tripwire auto-disable calls it)
if A.config:
    import json as _json
    config.trading_config = config.TradingConfig(**_json.load(open(A.config, encoding="utf-8")))
    print(f"[REPLAY] config override: {A.config}", flush=True)
tc = config.trading_config
tc.paper_trading = True
tc.paper_balance = A.balance
tc.paper_bnb_initial_usd = 1e9          # fees are netted inside Order.pnl; BNB ledger must never bind
tc.bnb_swap_enabled = False
tc.bnb_auto_sell_enabled = False
tc.fee_auto_fetch = False
if A.no_scanner:
    tc.thresholds.spike_scanner_enabled = False

import database, models
from sqlalchemy import event, select
from services import trading_engine as te
from services import websocket_tracker as wst
from services import binance_service as bsvc

te.datetime = FakeDatetime
wst.datetime = FakeDatetime
te.time = FAKE_TIME
te._leash_time = FAKE_TIME
bsvc.time = FAKE_TIME
_ns = types.SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("__")})
_ns.sleep = fake_sleep
te.asyncio = _ns

@event.listens_for(models.Order, "before_insert")
def _stamp_opened(mapper, conn, target):
    if target.opened_at is None:
        target.opened_at = FakeDatetime.utcnow()

@event.listens_for(models.PairData, "before_insert")
def _stamp_pd(mapper, conn, target):
    if getattr(target, "updated_at", None) is None:
        target.updated_at = FakeDatetime.utcnow()

@event.listens_for(models.Transaction, "before_insert")
def _stamp_tx(mapper, conn, target):
    if getattr(target, "timestamp", None) is None:
        target.timestamp = FakeDatetime.utcnow()

# ---------------------------------------------------------------- historical data server
EXINFO = json.load(open(os.path.join(CACHE, "exchange_info.json")))

def _sym(s):
    return s.replace("/USDT:USDT", "USDT") if "/" in s else s

class Series:
    __slots__ = ("ts", "o", "h", "l", "c", "v", "q")
    def __init__(self, df):
        df = df.drop_duplicates("open_time").sort_values("open_time")
        self.ts = df["open_time"].values.astype(np.int64)
        for k, col in (("o", "o"), ("h", "h"), ("l", "l"), ("c", "c"), ("v", "vol"), ("q", "qvol")):
            setattr(self, k, df[col].values.astype(float))
    def __len__(self):
        return len(self.ts)

class KlineServer:
    def __init__(self):
        self.k5, self.k1, self.k1h = {}, {}, {}
        self.no1m = set()
        self.lo, self.hi = WARM_MS - 6 * DAY, FOLLOW_MS + 2 * DAY
        self.fetches = 0

    def _load(self, sub, pair, lo, hi):
        fp = os.path.join(CACHE, sub, f"{pair}.csv")
        if not os.path.exists(fp):
            return None
        df = pd.read_csv(fp)
        if sub == "k1m":
            od = os.path.join(CACHE, "k1m_ondemand", f"{pair}.csv")
            if os.path.exists(od):
                df = pd.concat([df, pd.read_csv(od)])
        df = df[(df["open_time"] >= lo) & (df["open_time"] < hi)]
        return Series(df) if len(df) else None

    def get5(self, pair):
        if pair not in self.k5:
            s = self._load("k5m_full", pair, self.lo, self.hi) or self._load("k5m", pair, self.lo, self.hi)
            if s is None and pair == "BTCUSDT":
                s = self._load("", "btc_5m", self.lo, self.hi)
            self.k5[pair] = s
        return self.k5[pair]

    def get1(self, pair):
        if pair not in self.k1:
            s = self._load("k1m", pair, self.lo, self.hi)
            if s is None:
                od = os.path.join(CACHE, "k1m_ondemand", f"{pair}.csv")
                if os.path.exists(od):
                    df = pd.read_csv(od); df = df[(df.open_time >= self.lo) & (df.open_time < self.hi)]
                    s = Series(df) if len(df) else None
            self.k1[pair] = s
        return self.k1[pair]

    def ensure1m(self, pair, t_ms, quiet=False):
        """1m data needed (open position / scan-universe partial candle) on a pair outside the
        cached spans → fetch once per (pair, day) and cache under k1m_ondemand/."""
        s = self.get1(pair)
        if s is not None and len(s):
            j = np.searchsorted(s.ts, t_ms - 180_000, side="left")
            if j < len(s) and s.ts[j] <= t_ms and s.ts[0] <= t_ms - 3600_000:
                return s                      # a bar within the last 3 minutes: covered
        key = (pair, t_ms // DAY)
        if key in self.no1m:
            return s
        self.no1m.add(key)
        import urllib.request
        from urllib.parse import quote
        lo, hi = t_ms - 2 * 3600_000, min(t_ms + 3 * DAY, FOLLOW_MS + DAY)
        rows, cur = [], lo
        try:
            while cur < hi:
                url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={quote(pair)}&interval=1m"
                       f"&startTime={cur}&endTime={hi}&limit=1500")
                with urllib.request.urlopen(url, timeout=30) as r:
                    kl = json.loads(r.read())
                if not kl:
                    break
                rows.extend(kl)
                nxt = kl[-1][0] + 1
                if nxt <= cur:
                    break
                cur = nxt
                _realtime.sleep(0.15)
            self.fetches += 1
        except Exception as ex:
            if not quiet:
                logging.warning(f"[REPLAY] 1m fetch failed {pair}: {ex}")
            return s
        od = os.path.join(CACHE, "k1m_ondemand"); os.makedirs(od, exist_ok=True)
        fp = os.path.join(od, f"{pair}.csv")
        new = pd.DataFrame([[k[0], k[1], k[2], k[3], k[4], k[5], k[7]] for k in rows],
                           columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
        if os.path.exists(fp):
            new = pd.concat([pd.read_csv(fp), new])
        new = new.drop_duplicates("open_time").sort_values("open_time")
        new.to_csv(fp, index=False)
        base = self.k1.get(pair)
        frames = [new[(new["open_time"] >= self.lo) & (new["open_time"] < self.hi)]]
        if base is not None:
            frames.append(pd.DataFrame({"open_time": base.ts, "o": base.o, "h": base.h, "l": base.l,
                                        "c": base.c, "vol": base.v, "qvol": base.q}))
        self.k1[pair] = Series(pd.concat(frames))
        return self.k1[pair]

    def bar1(self, pair, m_ms):
        """The completed 1m bar opening at m_ms, or None."""
        s = self.get1(pair)
        if s is None or not len(s):
            return None
        i = np.searchsorted(s.ts, m_ms)
        if i < len(s) and s.ts[i] == m_ms:
            return (s.o[i], s.h[i], s.l[i], s.c[i])
        return None

    def last_price(self, pair, t_ms):
        s = self.get1(pair)
        if s is not None and len(s):
            i = np.searchsorted(s.ts, t_ms - 60_000, side="right") - 1
            if i >= 0:
                return float(s.c[i])
        s5 = self.get5(pair)
        if s5 is not None and len(s5):
            i = np.searchsorted(s5.ts, t_ms - 300_000, side="right") - 1
            if i >= 0:
                return float(s5.c[i])
        return None

    def ohlcv(self, pair, tf, limit, t_ms):
        if tf == "5m":
            return self._ohlcv5(pair, limit, t_ms)
        if tf == "1h":
            return self._ohlcv1h(pair, limit, t_ms)
        raise ValueError(f"unsupported timeframe {tf}")

    def _ohlcv5(self, pair, limit, t_ms):
        s = self.get5(pair)
        if s is None or not len(s):
            return []
        cur_open = (t_ms // 300_000) * 300_000
        n_done = np.searchsorted(s.ts, cur_open)           # bars with open_time < cur_open
        s1 = self.get1(pair)
        has1 = s1 is not None and len(s1) and s1.ts[0] <= cur_open and s1.ts[-1] >= cur_open - 60_000
        if has1:
            lo = max(0, n_done - (limit - 1))
            out = [[int(s.ts[i]), s.o[i], s.h[i], s.l[i], s.c[i], s.v[i]] for i in range(lo, n_done)]
            m_end = (t_ms // 60_000) * 60_000
            a = np.searchsorted(s1.ts, cur_open); b = np.searchsorted(s1.ts, m_end)
            if b > a:
                out.append([int(cur_open), float(s1.o[a]), float(s1.h[a:b].max()),
                            float(s1.l[a:b].min()), float(s1.c[b - 1]), float(s1.v[a:b].sum())])
            else:
                px = out[-1][4] if out else None
                if px is None:
                    return []
                out.append([int(cur_open), px, px, px, px, 0.0])
            return out
        # no 1m data: completed candles only, the last completed candle plays the live one
        lo = max(0, n_done - limit)
        return [[int(s.ts[i]), s.o[i], s.h[i], s.l[i], s.c[i], s.v[i]] for i in range(lo, n_done)]

    def _ohlcv1h(self, pair, limit, t_ms):
        s = self.get5(pair)
        if s is None or not len(s):
            return []
        cur_open = (t_ms // 3600_000) * 3600_000
        hr = (s.ts // 3600_000) * 3600_000
        cut = np.searchsorted(s.ts, (t_ms // 300_000) * 300_000)   # completed 5m bars only
        if cut == 0:
            return []
        hrs = hr[:cut]
        uniq, first = np.unique(hrs, return_index=True)
        out = []
        for j, (h0, i0) in enumerate(zip(uniq, first)):
            i1 = first[j + 1] if j + 1 < len(first) else cut
            out.append([int(h0), s.o[i0], float(s.h[i0:i1].max()), float(s.l[i0:i1].min()),
                        s.c[i1 - 1], float(s.v[i0:i1].sum())])
        if out and out[-1][0] < cur_open:
            px = out[-1][4]
            out.append([int(cur_open), px, px, px, px, 0.0])
        return out[-limit:]

KS = KlineServer()

class TickStore:
    """Real trades (Binance aggTrades daily archives → ticks/<PAIR>/<date>.npz, t ms + price)."""
    def __init__(self):
        self.cache = {}; self.missing = set(); self.hits = 0; self.misses = 0
    def _day(self, pair, day_ms):
        key = (pair, day_ms)
        if key in self.cache:
            return self.cache[key]
        if key in self.missing:
            return None
        fp = os.path.join(CACHE, "ticks", pair, f"{datetime.utcfromtimestamp(day_ms / 1000):%Y-%m-%d}.npz")
        if not os.path.exists(fp):
            self.missing.add(key); return None
        z = np.load(fp); d = (z["t"], z["p"].astype(float)); self.cache[key] = d
        if len(self.cache) > 400:
            self.cache.pop(next(iter(self.cache)))
        return d
    def window(self, pair, t0, t1, max_pts=60):
        """ticks with t0 <= t < t1, downsampled to <= max_pts keeping first/last/high/low."""
        day = (t0 // DAY) * DAY
        d = self._day(pair, day)
        if d is None:
            self.misses += 1; return None
        t, pr = d
        a = np.searchsorted(t, t0); b = np.searchsorted(t, t1)
        if t1 > day + DAY:                                   # window crosses midnight
            d2 = self._day(pair, day + DAY)
            if d2 is not None:
                t = np.concatenate([t, d2[0]]); pr = np.concatenate([pr, d2[1]]); b = np.searchsorted(t, t1)
        if b <= a:
            return []
        self.hits += 1
        tt, pp = t[a:b], pr[a:b]
        if len(tt) <= max_pts:
            return list(zip(tt.tolist(), pp.tolist()))
        idx = set(np.linspace(0, len(tt) - 1, max_pts).astype(int).tolist())
        idx.add(int(np.argmax(pp))); idx.add(int(np.argmin(pp)))
        idx = sorted(idx)
        return [(int(tt[i]), float(pp[i])) for i in idx]
TS = TickStore()

class Universe:
    """Live rule: get_top_futures_pairs ranks ALL USDT perps by the ticker's rolling 24h quote
    volume at scan time. Rebuilt here from 5m klines (Σ qvol of the 288 completed bars before
    t) for every pair with a full-range 5m file; pairs without one fall back to prior-day volume."""
    def __init__(self):
        self.daily = {}
        for fp in glob.glob(os.path.join(CACHE, "daily", "*.csv")):
            d = {}
            for line in open(fp):
                p = line.strip().split(",")
                if len(p) == 3:
                    d[int(p[1])] = float(p[2])
            if d:
                self.daily[os.path.basename(fp)[:-4]] = d
        self.full = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(CACHE, "k5m_full", "*.csv")))
        self.cum = {}
        self._cache_t = None; self._cache_rows = None
        print(f"[REPLAY] universe: {len(self.full)} pairs with full-range 5m (rolling-24h ranking), "
              f"{len(self.daily)} with daily volume (fallback)", flush=True)

    def _vol24(self, pair, t_ms):
        if pair in self.full:
            if pair not in self.cum:
                s = KS.get5(pair)
                self.cum[pair] = None if s is None else (s.ts, np.concatenate([[0.0], np.cumsum(s.q)]))
            c = self.cum[pair]
            if c is not None:
                ts, cq = c
                i = np.searchsorted(ts, t_ms - 300_000, side="right")   # completed bars only
                if i >= 288:
                    return float(cq[i] - cq[i - 288])
                return None
        d = self.daily.get(pair)
        if d:
            return d.get(((t_ms // DAY) - 1) * DAY)
        return None

    def ranked(self, t_ms):
        key = (t_ms // 60_000)
        if key == self._cache_t:
            return self._cache_rows
        rows = []
        for pair in set(self.full) | set(self.daily):
            meta = EXINFO.get(pair)
            if meta is None:
                continue
            ob = meta.get("onboardDate")
            if ob is not None and ob > t_ms:
                continue
            v = self._vol24(pair, t_ms)
            if v is None or v <= 0:
                continue
            rows.append((v, pair, meta))
        rows.sort(reverse=True)
        self._cache_t, self._cache_rows = key, rows
        return rows

    def top(self, limit, t_ms, new_listing_days=0, alpha=False, coin_only=False):
        out = []
        for qv, pair, meta in self.ranked(t_ms):
            if coin_only and meta.get("underlyingType") not in (None, "COIN"):
                continue
            ob = meta.get("onboardDate")
            age = (t_ms - ob) / DAY if ob else None
            if new_listing_days and age is not None and age < new_listing_days:
                continue
            if alpha and any("Alpha" in x for x in (meta.get("underlyingSubType") or [])):
                continue
            out.append({"symbol": f"{pair[:-4]}/USDT:USDT", "pair": pair, "price": 0.0,
                        "volume_24h": float(qv), "change_24h": 0.0, "age_days": age})
            if len(out) >= limit:
                break
        if limit <= 60:
            for p in out:
                KS.ensure1m(p["pair"], t_ms, quiet=True)
                p["price"] = KS.last_price(p["pair"], t_ms) or 0.0
        return out

UNI = Universe()

class FakeBinance:
    def __init__(self):
        self.calls = {}
    def _n(self, k):
        self.calls[k] = self.calls.get(k, 0) + 1
    async def get_top_futures_pairs(self, limit=20, new_listing_filter_days=0,
                                    alpha_subtype_filter_enabled=False, coin_underlying_only=False):
        self._n("top")
        return UNI.top(limit, Clock.ms, new_listing_filter_days, alpha_subtype_filter_enabled, coin_underlying_only)
    async def get_ohlcv(self, symbol, timeframe="5m", limit=100):
        self._n(f"ohlcv_{timeframe}_{limit}")
        return KS.ohlcv(_sym(symbol), timeframe, limit, Clock.ms)
    async def get_current_price(self, symbol):
        self._n("price")
        return KS.last_price(_sym(symbol), Clock.ms) or 0.0
    async def get_bnb_price(self):
        return 600.0
    async def fetch_funding_rate(self, symbol):
        return None
    async def fetch_orderbook(self, symbol, limit=5):
        return None
    def get_tick_size(self, symbol):
        return 0.0001
    async def get_balance(self):
        return {"ok": False, "usdt_free": 0, "usdt_used": 0, "usdt_total": 0, "bnb_free": 0,
                "bnb_total": 0, "total_portfolio": 0, "maint_margin": 0, "margin_balance": 0}
    async def get_commission_rates(self):
        raise RuntimeError("replay: no exchange")
    async def get_open_positions(self):
        return []
    async def get_funding_fees_usd(self, *a, **k):
        return 0.0
    def __getattr__(self, name):
        raise AttributeError(f"replay FakeBinance: unexpected call to binance_service.{name}")

FB = FakeBinance()
te.binance_service = FB

WT = wst.websocket_tracker
WT.websocket = None
WT.running = False
WT.pair_silence_seconds = lambda pair: 0.0
WT.is_pair_streamed = lambda pair: True
async def _noop(*a, **k):
    return None
WT.start = _noop
WT.force_reconnect = _noop
WT._reconnect = _noop

ENG = te.trading_engine
_real_open = ENG.open_position
async def _gated_open(*a, **k):
    if Clock.ms >= END_MS:
        return None
    return await _real_open(*a, **k)
ENG.open_position = _gated_open

PENDING = {}
PERSIST_STATS = {"attempt": 0, "filled": 0, "expired": 0}
async def _maker_paper(pair, direction, current_price, notional_value, maker_fee_rate,
                       taker_fee_rate, confidence=None):
    """Live: post a limit 1-2 ticks inside, watch WS prices for 20 s, then re-validate the signal
    with fresh klines before falling back to taker. Replay (1m data): a candidate must be reached
    by the engine again on the NEXT scan (signal persisted >= one scan step) before it fills —
    a stricter stand-in for the 20 s re-validation. Fill price = limit if that minute's bar
    touches it (maker), else the minute close (taker fallback)."""
    key = (pair, direction)
    if not A.no_persist:
        PERSIST_STATS["attempt"] += 1
        last = PENDING.get(key)
        if last is None or Clock.ms - last > 3 * A.scan_step * 1000:
            PENDING[key] = Clock.ms
            PERSIST_STATS["expired"] += 1
            return {"entry_order_type": "SIGNAL_EXPIRED", "skipped": True, "reason": "replay_persistence",
                    "price": current_price, "entry_fee": 0.0, "wait_seconds": 20.0}
        PENDING.pop(key, None)
        PERSIST_STATS["filled"] += 1
    tcfg = config.trading_config
    offset_ticks = getattr(tcfg, "maker_offset_ticks", 2)
    if current_price >= 10000: tick = 0.10
    elif current_price >= 100: tick = 0.01
    elif current_price >= 1: tick = 0.001
    else: tick = 0.0001
    limit_price = current_price - offset_ticks * tick if direction == "LONG" else current_price + offset_ticks * tick
    limit_price = round(limit_price / tick) * tick
    if A.ticks:
        w = TS.window(pair, Clock.ms, Clock.ms + 20_000, max_pts=100000)
        if w:                                                # real 20 s WebSocket window
            Clock.ms += 20_000
            for _t, _p in w:
                if (direction == "LONG" and _p <= limit_price) or (direction == "SHORT" and _p >= limit_price):
                    return {"price": limit_price, "entry_fee": notional_value * maker_fee_rate, "entry_order_type": "MAKER"}
            return {"price": float(w[-1][1]), "entry_fee": notional_value * taker_fee_rate, "entry_order_type": "TAKER_FALLBACK"}
    m = (Clock.ms // 60_000) * 60_000
    bar = KS.bar1(pair, m)
    Clock.ms += 20_000
    if bar is not None:
        o, h, l, c = bar
        # maker fill if the minute's extreme touches the limit. Calibration: this gives 53% maker
        # fills (live 39%) and sleeve totals on target; the close-based rule gave 18% maker and
        # under-shot the momentum sleeve by 40% (pass 6). Declared optimistic by ~+14 pp maker share.
        if (direction == "LONG" and l <= limit_price) or (direction == "SHORT" and h >= limit_price):
            return {"price": limit_price, "entry_fee": notional_value * maker_fee_rate, "entry_order_type": "MAKER"}
        fb = float(c)
    else:
        fb = current_price
    return {"price": fb, "entry_fee": notional_value * taker_fee_rate, "entry_order_type": "TAKER_FALLBACK"}
ENG._simulate_maker_entry_paper = _maker_paper
async def _no_expired_row(*a, **k):
    return None
ENG._record_signal_expired_order = _no_expired_row   # live: this call raises (kwarg mismatch) and the open aborts — same net effect, no traceback spam

# ---------------------------------------------------------------- driver
def _tick_seq(direction, bar, steps):
    """Intra-minute price path: open → adverse extreme → favourable extreme → close, each leg
    interpolated in `steps` points so a stop / trail / BE level is crossed near the level
    (live fills at the first tick past the level) instead of at the minute's extreme or close."""
    o, h, l, c = bar
    adv, fav = (l, h) if direction == "LONG" else (h, l)
    first, second = adv, fav
    if A.tick_order == "favourable" or (A.tick_order == "random" and _RNG.random() < 0.5):
        first, second = fav, adv
    pts = [o]
    for a, b in ((o, first), (first, second), (second, c)):
        # adaptive resolution: a leg is split so no step exceeds 0.10% of price — a stop / trail /
        # lock level is then crossed within ~0.1% (live fills at the first tick past the level;
        # coarse 6-step legs over-shot fade stops by 0.25%/fill on wild pairs)
        n = max(1, steps, int(math.ceil(abs(b - a) / max(a, 1e-12) / 0.001)))
        pts.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return pts

async def feed_ticks(m_ms, db):
    """Deliver the completed minute [m_ms, m_ms+60s) to every pair with an open position
    (stepped path through the realtime stop-loss path + the position monitor) or a
    post-exit tracker (close only)."""
    pairs = {}
    fresh = set()          # opened inside this minute: only the close is post-entry information
    for pair, lst in list(te._open_orders_cache.items()):
        if lst:
            pairs[pair] = lst[0].get("direction", "LONG")
            oa = lst[0].get("opened_at")
            try:
                oa_ms = int(pd.Timestamp(oa).timestamp() * 1000) if oa is not None else None
            except Exception:
                oa_ms = None
            if oa_ms is not None and oa_ms >= m_ms:
                fresh.add(pair)
    for info in list(ENG._post_exit_tracking.values()):
        pairs.setdefault(info.get("pair"), None)
    paths = {}
    for pair, direction in pairs.items():
        if not pair:
            continue
        bar = KS.bar1(pair, m_ms)
        if bar is None:
            KS.ensure1m(pair, m_ms)
            bar = KS.bar1(pair, m_ms)
            if bar is None:
                continue
        if direction is None:
            WT.update_price(pair, float(bar[3]), tick=True)
            continue
        if A.ticks:
            oa = None
            try:
                oa = int(pd.Timestamp(te._open_orders_cache[pair][0].get("opened_at")).timestamp() * 1000)
            except Exception:
                oa = None
            w = TS.window(pair, max(m_ms, oa or 0), m_ms + 60_000)
            if w is not None:
                paths[pair] = w if w else []          # list of (t, p): real ticks after the fill only
                continue
        if pair in fresh:
            paths[pair] = [float(bar[3])]
        elif A.tick_mode == "wickstop":
            # adverse wick first (stops / BE levels can fire on it), then the open→close path only:
            # the favourable extreme is NOT fed, so a peak can only register on closes and a
            # "peak then pullback inside the same minute" (the stepped path's artefact) cannot occur.
            o, h, l, c = bar
            adv = l if direction == "LONG" else h
            n = max(1, int(math.ceil(abs(c - o) / max(o, 1e-12) / 0.001)))
            paths[pair] = [float(o), float(adv)] + [o + (c - o) * k / n for k in range(1, n + 1)]
        else:
            paths[pair] = _tick_seq(direction, bar, A.tick_steps)
    if not paths:
        return
    paths = {k: v for k, v in paths.items() if len(v)}
    if not paths:
        return
    n = max(len(v) for v in paths.values())
    t_end = Clock.ms
    for k in range(n):
        Clock.ms = m_ms + int((k + 1) * 60_000 / n)     # live-like time inside the minute (confirmation timers, leash clocks)
        touched = False
        for pair, pts in paths.items():
            if k < len(pts):
                pt = pts[k]
                if isinstance(pt, tuple):                # real tick: (t, price)
                    Clock.ms = int(pt[0]); px = float(pt[1])
                else:
                    px = float(pt)
                WT.update_price(pair, px, tick=True)
                await ENG.check_realtime_stop_loss(pair, px)
                touched = True
        if touched:
            await ENG.update_open_positions(db)
    Clock.ms = t_end

async def dump(db, meta):
    from sqlalchemy import text
    import sqlite3
    con = sqlite3.connect(DB_PATH)
    for tbl, name in (("orders", "orders"), ("monitor_periods", "bull_periods"),
                      ("bear_monitor_periods", "bear_periods"), ("phantom_flips", "phantom_flips"),
                      ("transactions", "transactions")):
        try:
            df = pd.read_sql_query(f"SELECT * FROM {tbl}", con)
            df.to_csv(os.path.join(OUT, f"{A.tag}_{name}.csv"), index=False)
        except Exception as ex:
            logging.warning(f"[REPLAY] dump {tbl} failed: {ex}")
    con.close()
    try:
        meta["filter_blocks"] = ENG._get_filter_block_summary()
    except Exception as ex:
        meta["filter_blocks_error"] = str(ex)
    meta["binance_calls"] = FB.calls
    meta["persistence"] = PERSIST_STATS
    meta["ticks"] = {"minutes_with_ticks": TS.hits, "minutes_missing": TS.misses}
    meta["ondemand_1m_fetches"] = KS.fetches
    json.dump(meta, open(os.path.join(OUT, f"{A.tag}_meta.json"), "w"), indent=1, default=str)

async def run():
    t0 = _realtime.time()
    await database.init_db()
    db = database.AsyncSessionLocal()
    await ENG.initialize(db)
    ENG.is_running = True
    ENG.is_paper_mode = True
    ENG.paper_balance = A.balance
    ENG.paper_bnb_balance_usd = 1e9
    await ENG.save_state(db)
    scans = 0
    next_scan = WARM_MS
    m = WARM_MS
    last_day = None
    meta = {"args": vars(A), "start_ms": START_MS, "end_ms": END_MS, "warm_ms": WARM_MS}
    while m < FOLLOW_MS:
        Clock.ms = m
        day = m // DAY
        if day != last_day:
            last_day = day
            await db.commit(); await db.close()
            db = database.AsyncSessionLocal()
            n_open = len([1 for l in te._open_orders_cache.values() for _ in l])
            print(f"[REPLAY {A.tag}] {FakeDatetime.utcnow():%Y-%m-%d} scans={scans} open={n_open} "
                  f"bal={ENG.paper_balance:.0f} elapsed={_realtime.time() - t0:.0f}s", flush=True)
        await feed_ticks(m - 60_000, db)
        Clock.ms = m
        await ENG.update_open_positions(db)
        await ENG.update_orders_cache(db)
        await ENG.update_post_exit_tracking(db)
        if m >= next_scan and m < END_MS:
            ENG._last_scan_time = 0.0
            Clock.ms = m
            try:
                await ENG.scan_and_trade(db)
            except Exception as ex:
                logging.exception(f"[REPLAY] scan failed at {FakeDatetime.utcnow()}: {ex}")
            scans += 1
            next_scan = m + A.scan_step * 1000
        if m >= END_MS:
            if not any(te._open_orders_cache.values()) and not ENG._post_exit_tracking:
                break
        m += 60_000
    Clock.ms = m
    await ENG.update_orders_cache(db)
    meta["scans"] = scans
    meta["end_clock"] = str(FakeDatetime.utcnow())
    meta["elapsed_s"] = round(_realtime.time() - t0, 1)
    meta["final_balance"] = ENG.paper_balance
    await db.commit()
    await dump(db, meta)
    await db.close()
    await database.engine.dispose()
    print(f"[REPLAY {A.tag}] done scans={scans} final_bal={ENG.paper_balance:.2f} "
          f"elapsed={meta['elapsed_s']}s", flush=True)

if __name__ == "__main__":
    asyncio.run(run())
