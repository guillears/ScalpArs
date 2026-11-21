#!/usr/bin/env python3
"""📡 Route B — open interest + crowd positioning for the HOT coins. Pure logic; recorder.py does the I/O.

WHY   the liquidation stream (!forceOrder@arr) is a SAMPLE (≤ 1 per coin per second); open interest DROPPING during a sharp move is the
      cleaner cascade signal, and the crowd's positioning (top-trader / global long-short ratios, taker buy/sell) is the "before" picture.
WHAT  ① OPEN INTEREST every OI_EVERY_S = 60 s: GET /fapi/v1/openInterest?symbol=X (request weight 1) for ≤ OI_MAX_COINS coins
        (the full-book PRIORITY coins first, then the hot list) → kind 'oi' (one line per sweep):
        {"r": sweep start ms, "m": [[sym, oi (contracts), T (exchange ms), chg_1m %, chg_5m %], …]}  (chg vs this process's own sweep
        1 / 5 minutes earlier; null when that sweep is missing)
      ② POSITIONING every RATIO_EVERY_S = 300 s, period 5m, the LATEST row of /futures/data/topLongShortAccountRatio (tla) ·
        topLongShortPositionRatio (tpa) · globalLongShortAccountRatio (gla) · takerlongshortRatio (tkr) → kind 'ratio':
        {"r": sweep start ms, "m": [[sym, ts (the row's period ms), tla, tpa, gla, tkr buySellRatio, buyVol, sellVol], …]}
BUDGET /fapi/v1/openInterest: weight 1 → ≤ 150 / min. The /futures/data/* endpoints have their OWN IP limit (Binance docs: 1,000 requests
      per 5 min per IP); one sweep uses at most RATIO_MAX_PER_SWEEP = 480 (4 calls per coin → ≤ 120 coins; the rest are SKIPPED and
      counted) and a rolling guard keeps ≤ RATIO_GUARD_PER_5M = 960 in any 300 s (two overlapping sweeps fit: no every-other-sweep
      starvation).
      Every response's X-MBX-USED-WEIGHT-1M is read; 418 / 429 stop the requests (see recorder.py)."""
import math
from collections import deque

OI_EVERY_S, RATIO_EVERY_S = 60, 300
OI_MAX_COINS = 150
OI_PATH = "/fapi/v1/openInterest?symbol={}"
RATIO_PERIOD = "5m"
RATIO_ENDPOINTS = (("tla", "/futures/data/topLongShortAccountRatio", "longShortRatio"),
                   ("tpa", "/futures/data/topLongShortPositionRatio", "longShortRatio"),
                   ("gla", "/futures/data/globalLongShortAccountRatio", "longShortRatio"),
                   ("tkr", "/futures/data/takerlongshortRatio", "buySellRatio"))
RATIO_MAX_PER_SWEEP = 480                         # < half of Binance's 1,000 / 5 min IP limit for /futures/data
RATIO_GUARD_PER_5M = 960                          # safety: never more than this in any rolling 300 s
RATIO_MAX_PER_5M = RATIO_MAX_PER_SWEEP            # (old name)
RATIO_PAUSE_429_S = 300
REQ_GAP_S = 0.05                                  # spacing between two requests of one worker
CONCURRENCY = 3                                   # parallel requests per sweep (Tokyo is ~0.3–0.4 s away: 150 coins ≈ 20 s, not 60);
                                                  # the depth snapshots have their own thread, so sweeps never delay them


def ratio_path(endpoint, sym):
    return f"{endpoint}?symbol={sym}&period={RATIO_PERIOD}&limit=1"


def order_coins(full, hot, cap):
    """the priority (full-book) coins first, then the hot list, de-duplicated, ≤ cap. Pure."""
    out = []
    for s in list(full or []) + list(hot or []):
        if s and s not in out:
            out.append(s)
    return out[:max(0, int(cap))]


def ratio_coins(coins, used_5m, per_sweep=RATIO_MAX_PER_SWEEP, guard=RATIO_GUARD_PER_5M, per_coin=len(RATIO_ENDPOINTS)):
    """→ (coins to read this sweep, coins skipped): ≤ per_sweep requests this sweep AND ≤ guard in the rolling 300 s. Pure."""
    n = max(0, min(int(per_sweep), int(guard) - int(used_5m)) // per_coin)
    return list(coins[:n]), max(0, len(coins) - n)


class Rolling:
    """requests in a rolling window (seconds)."""

    def __init__(self, window_s=300):
        self.w, self.t = window_s, deque()

    def add(self, now_s, n=1):
        for _ in range(n):
            self.t.append(now_s)

    def count(self, now_s):
        while self.t and now_s - self.t[0] >= self.w:
            self.t.popleft()
        return len(self.t)


class OiTrack:
    """per coin the last ≤ 7 one-minute OI samples → % change vs 1 and 5 minutes earlier (a missed sweep → the sample one minute older,
    never further; else None)."""

    def __init__(self):
        self.d = {}

    def feed(self, sym, oi, minute):
        q = self.d.setdefault(sym, deque(maxlen=7))
        if q and q[-1][0] == minute:
            q[-1] = (minute, oi)
        else:
            q.append((minute, oi))

        def chg(k):
            ref = None
            for m, v in q:
                if m in (minute - k, minute - k - 1) and (ref is None or m > ref[0]):
                    ref = (m, v)
            return None if ref is None or not ref[1] else round((oi / ref[1] - 1.0) * 100.0, 4)
        return chg(1), chg(5)

    def prune(self, keep):
        for s in [s for s in self.d if s not in keep]:
            del self.d[s]


def parse_oi(body):
    """/fapi/v1/openInterest body → (oi float, time ms) or None. Pure."""
    try:
        oi = float(body["openInterest"])
        return (oi, int(body.get("time") or 0)) if math.isfinite(oi) else None
    except (KeyError, TypeError, ValueError):
        return None


def parse_ratio(body, field):
    """a /futures/data list → (timestamp ms, value, row) of the NEWEST row, or None. Pure."""
    try:
        rows = [r for r in (body or []) if isinstance(r, dict) and field in r]
        if not rows:
            return None
        r = max(rows, key=lambda x: int(x.get("timestamp") or 0))
        return int(r.get("timestamp") or 0), float(r[field]), r
    except (TypeError, ValueError):
        return None


def ratio_row(sym, parsed):
    """{key: parse_ratio() result or None} → [sym, ts, tla, tpa, gla, tkr, buyVol, sellVol] (None where unread); None if all unread. Pure."""
    if not any(parsed.get(k) for k, _, _ in RATIO_ENDPOINTS):
        return None
    ts = max((parsed[k][0] for k, _, _ in RATIO_ENDPOINTS if parsed.get(k)), default=None)
    vals = [(parsed[k][1] if parsed.get(k) else None) for k, _, _ in RATIO_ENDPOINTS]
    tk = parsed.get("tkr")

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    return [sym, ts] + vals + [f(tk[2].get("buyVol")) if tk else None, f(tk[2].get("sellVol")) if tk else None]
