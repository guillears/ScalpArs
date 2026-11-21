#!/usr/bin/env python3
"""📡 Route B — FULL order book for the bot's own coins (the PRIORITY set). Pure logic; recorder.py does the I/O.

WHY   the top-20 stream (<sym>@depth20@500ms) reaches only 0.05–0.4 % from the price on fine-tick coins (EVAA 0.05 %, BLESS 0.17 %,
      PHAROS 0.38 %), so the cost of a $6–10k order and the depth within 0.5 / 1 / 2 % cannot be measured. For the coins the bot HOLDS or
      WATCHES (open positions · FRENZY-flagged · EXHAUSTION / DIST / FLATBASE tracked episodes) the recorder keeps a LOCAL FULL BOOK.
HOW   Binance's documented futures procedure: subscribe <sym>@depth@100ms (/public/stream) and BUFFER the events · GET
      /fapi/v1/depth?symbol=X&limit=1000 (weight 20) · drop events with u < lastUpdateId · the first applied event must have
      U ≤ lastUpdateId ≤ u · every next event's pu == the previous u, else the book is out of sync → a NEW snapshot (resync). Quantities
      are absolute (0 = remove the level).
TRUST the snapshot holds ≤ 1,000 levels a side, so the book is exact only between the snapshot's deepest bid and deepest ask (the EDGES;
      a side with < 1,000 levels is complete — edge 0 / ∞). Levels beyond an edge are ignored (never half-known), every derived record
      carries how far the trusted book reaches (rb / ra, % from mid; 100 = the whole side) — a band past the reach is a LOWER bound.
      The book is re-snapshotted (without a gap in the records) every FB_REFRESH_S, or sooner when the price drifts toward an edge.
RECORDS (two new kinds, s3 data/fbook/ and data/fsnap/ — the top-20 'book' kind is unchanged):
  fbook (every FB_EMIT_S = 1 s per synced coin) {"r" local ms, "s", "E" last event ms, "u" last update id, "bb", "ba", "sp" spread bps,
         "rb" / "ra" trusted reach %, "b" / "a" [$ resting within FB_BANDS % of mid], "cb" / "cs" [bps vs mid to BUY / SELL FB_SIZES $ at
         market by walking the book; null = more than the trusted book within 5 %]}
  fsnap (every FB_SNAP_S = 30 s per synced coin) {"r", "s", "E", "u", "eb" / "ea" the edges (null = complete side), "b" / "a" every
         [price, qty] within ±FB_SNAP_PCT % of mid (trusted region)} — raw reconstruction at 30 s resolution."""
import math
import time
from collections import deque

FB_SUFFIX = "@depth@100ms"
FB_LIMIT = 1000                                   # snapshot levels a side (weight 20)
FB_SNAP_WEIGHT = 20
FB_BANDS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)         # % from mid
FB_BAND_TAGS = ("01", "02", "05", "1", "2", "5")
FB_SIZES = (1000, 3000, 6000, 10000, 25000)       # $ market order each side
FB_WINDOW_PCT = 5.0                               # derived records look ±5 % around mid
FB_SNAP_PCT = 5.0                                 # fsnap keeps ±5 %
FB_EMIT_S, FB_SNAP_S = 1.0, 30.0
FB_REFRESH_S = 3600                               # re-snapshot a synced book every hour (keeps the edges wide)
FB_REFRESH_MIN_GAP_S = 300                        # … and at most every 5 min when the price drifts toward an edge
FB_MAX_BUFFER = 300                               # events kept while waiting for a snapshot (~30 s of a busy coin); measured worst case
                                                  # 40–140 MB at 30 coins all unsynced (10–40 levels per event), typical far less
FB_STALE_S = 300                                  # a synced book with no diff event for 5 min is unsynced (re-snapshot) — a dead stream
FB_PENDING_TIMEOUT_S = 10                         # a snapshot the stream never caught up with → ask again
FB_MIN_SNAP_GAP_S = 5                             # never two snapshot requests for one coin within 5 s
FB_WEIGHT_PER_MIN = 300                           # snapshot REST budget (15 / min; Binance's IP limit is 2,400, our rule ≤ 1,200)
FB_WEIGHT_BURST = 100
FB_USED_WEIGHT_STOP = 800                         # X-MBX-USED-WEIGHT-1M above this → wait for the next minute
MAX_PAUSE_S = 3 * 86400                           # no pause (Retry-After, saved file) longer than 3 days
FB_SNAP_STAGGER_S = 30                            # first fsnap of a coin lands at a hashed offset (30 coins never in the same second)
FULL_MAX = 30                                     # priority coins at most
FULL_STICKY_S = 1800                              # a coin that leaves the bot list is kept 30 min (the exit side of a closed trade)
BOT_MAX_AGE_S = 1800
INF = float("inf")


# ── the priority set ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def bot_priority(doc, ms, max_age_s=BOT_MAX_AGE_S):
    """the bot list → (open, flagged, tracked) symbol lists: open in list order, flagged newest spike first, tracked in list order;
    ([], [], []) when missing / malformed / older than max_age_s. 'followed' (every pair FRENZY reads) is NOT a priority group. Pure."""
    try:
        if not isinstance(doc, dict) or ms - int(doc.get("at") or 0) > max_age_s * 1000:
            return [], [], []

        def keys(v):
            return [p for p in (v.keys() if isinstance(v, dict) else (v or [])) if isinstance(p, str) and p]
        op = keys(doc.get("open"))
        fl = doc.get("flagged") or {}
        if isinstance(fl, dict):
            fl = [p for p, _ in sorted(((p, str(t or "")) for p, t in fl.items() if isinstance(p, str) and p), key=lambda x: x[1], reverse=True)]
        else:
            fl = keys(fl)
        return op, fl, keys(doc.get("tracked"))
    except (TypeError, ValueError, AttributeError):
        return [], [], []


def pick_full(doc, ms, current, seen, tradable=None, cap=FULL_MAX, sticky_s=FULL_STICKY_S):
    """→ (new priority list, n wanted before the cap). Order: OPEN · FLAGGED (newest first) · TRACKED (coins already recorded first —
    no churn) · STICKY (left the bot list < sticky_s ago, most recent first). `seen` (sym → last ms in the bot list) is UPDATED in place
    and pruned. A non-tradable (delisted) coin leaves at once. Pure apart from `seen`."""
    ok = (lambda s: s in tradable) if tradable else (lambda s: s.endswith("USDT"))
    op, fl, tr = bot_priority(doc, ms)
    cur = set(current or [])
    tr = [s for s in tr if s in cur] + [s for s in tr if s not in cur]
    want = []
    for s in op + fl + tr:
        if ok(s) and s not in want:
            want.append(s)
    for s in want:
        seen[s] = ms
    for s in [s for s, t in seen.items() if ms - t >= sticky_s * 1000]:
        del seen[s]
    sticky = sorted((s for s in (current or []) if s not in want and ok(s) and s in seen), key=lambda s: -seen[s])
    n_want = len(want)
    return (want + sticky)[:max(0, int(cap))], n_want


# ── REST budget ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

class WeightBucket:
    """token bucket: `per_min` weight per minute, at most `burst` saved up."""

    def __init__(self, per_min=FB_WEIGHT_PER_MIN, burst=FB_WEIGHT_BURST, now_s=None):
        self.rate, self.cap = per_min / 60.0, float(burst)
        self.tokens = float(burst)
        self.t = time.time() if now_s is None else now_s

    def take(self, w, now_s):
        self.tokens = min(self.cap, self.tokens + max(0.0, now_s - self.t) * self.rate)
        self.t = now_s
        if self.tokens >= w:
            self.tokens -= w
            return True
        return False


def pause_reason(status, headers, used_stop=FB_USED_WEIGHT_STOP):
    """418 / 429 / 'weight' / None — why rest_pause_s() pauses. Pure."""
    if status in (418, 429):
        return status
    h = {str(k).lower(): v for k, v in (headers or {}).items()}
    try:
        return "weight" if int(h.get("x-mbx-used-weight-1m")) > used_stop else None
    except (TypeError, ValueError):
        return None


def rest_pause_s(status, headers, now_s, used_stop=FB_USED_WEIGHT_STOP):
    """seconds to stop ALL snapshot requests after a response: 418 (IP banned) → Retry-After or 600 · 429 (too many) → Retry-After or
    120 · used weight (X-MBX-USED-WEIGHT-1M) above used_stop → until the next minute · else 0 (a 400 / 5xx is that coin's problem:
    counted and retried with a per-coin backoff, never a global stop). Pure."""
    h = {str(k).lower(): v for k, v in (headers or {}).items()}

    def ra(default):
        try:
            v = float(h.get("retry-after"))
        except (TypeError, ValueError):
            return float(default)
        return min(MAX_PAUSE_S, max(1.0, v)) if math.isfinite(v) else float(default)   # inf / nan / 1e30 never freeze us
    if status == 418:
        return ra(600)
    if status == 429:
        return ra(120)
    try:
        used = int(h.get("x-mbx-used-weight-1m"))
    except (TypeError, ValueError):
        used = None
    if used is not None and used > used_stop:
        return 60.0 - (now_s % 60.0) + 1.0
    return 0.0


# ── the local book ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _apply(bids, asks, ev, edge_bid, edge_ask):
    for p, q in ev.get("b") or ():
        p, q = float(p), float(q)
        if q == 0.0:
            bids.pop(p, None)
        elif p >= edge_bid:
            bids[p] = q
    for p, q in ev.get("a") or ():
        p, q = float(p), float(q)
        if q == 0.0:
            asks.pop(p, None)
        elif p <= edge_ask:
            asks[p] = q


def snap_offset_ms(sym):
    """a stable per-coin offset in [0, FB_SNAP_S) s — the coin's fsnap phase. Pure."""
    h = 0
    for ch in str(sym):
        h = (h * 131 + ord(ch)) % 1_000_003
    return int((h % int(FB_SNAP_STAGGER_S * 1000)))


class LocalBook:
    """one coin's local full book, synced per Binance's U / u / pu rules. on_event / on_snapshot return what happened:
    'buffered' · 'applied' · 'old' · 'gap' (out of sync → a new snapshot is needed) · 'waiting' (snapshot newer than the stream) ·
    'stale' (snapshot older than the buffered stream → ask again) · 'synced' · 'refreshed' (a new snapshot swapped in with no gap) ·
    'bad' (a malformed event / snapshot: THIS coin is unsynced, nothing raises).
    MEMORY: the buffer holds ≤ FB_MAX_BUFFER events (~30 s of a busy coin) and, while no snapshot can be asked (hold=True: the REST pause
    of a 418 / 429 / weight stop), only the newest event — events older than the coming snapshot are useless anyway (u < lastUpdateId)."""

    def __init__(self, sym, max_buffer=FB_MAX_BUFFER):
        self.sym = sym
        self.bids, self.asks = {}, {}
        self.synced = False
        self.last_u = None
        self.last_E = None
        self.last_ev_ms = None       # local time of the last diff event (staleness)
        self.last_emit_ms = 0        # local time of the last fbook line (the recorder sets it)
        self.buf = deque(maxlen=max_buffer)
        self.overflow = 0
        self.pending = None          # (lastUpdateId, bids, asks, limit, at ms) — a snapshot the stream has not reached yet
        self.want = True             # a snapshot should be requested
        self.refreshing = False      # synced, and a new snapshot is being fetched (events applied AND buffered)
        self.edge_bid, self.edge_ask = 0.0, INF
        self.synced_ms = None
        self.reach0 = None           # min trusted reach % right after the last sync
        self.last_req_ms = 0
        self.next_req_ms = 0         # per-coin backoff after failed snapshot requests (a delisted / erroring coin)
        self.req_fails = 0
        self.last_refresh_ms = 0
        self.refresh_next_ms = 0     # no refresh before this (set when a refresh gives up)
        self.last_snap_rec_ms = 0
        self.snap_off = snap_offset_ms(sym)                         # fsnap phase: every FB_SNAP_S at this per-coin offset (stagger)
        self.last_snap_slot = None
        self.refresh_tries = 0

    # stream
    def _push(self, ev, hold=False):
        if hold:                                                     # no snapshot can come soon: keep only the newest event
            self.buf.clear()
        elif len(self.buf) == self.buf.maxlen:
            self.overflow += 1
        self.buf.append(ev)

    def unsync(self):
        self.synced = self.refreshing = False
        self.bids, self.asks = {}, {}
        self.last_u = None
        self.pending = None
        self.want = True
        self.buf.clear()

    def on_event(self, ev, now_ms=0, hold=False):
        try:
            u = int(ev["u"])
            int(ev["U"])
        except (KeyError, TypeError, ValueError):
            return "bad"
        self.last_E = ev.get("E", self.last_E)
        self.last_ev_ms = now_ms
        if self.synced:
            if u <= self.last_u:
                return "old"
            pu = ev.get("pu")
            try:
                chained = pu is not None and int(pu) == self.last_u
            except (TypeError, ValueError):
                chained = False
            if not chained:
                self.unsync()
                self._push(ev, hold)
                return "gap"
            try:
                _apply(self.bids, self.asks, ev, self.edge_bid, self.edge_ask)
            except (TypeError, ValueError, AttributeError):
                self.unsync()                                        # a half-applied event: this coin resyncs, nothing else
                return "bad"
            self.last_u = u
            if self.refreshing:
                self._push(ev, hold and self.pending is None)
                if self.pending is not None:
                    return self._try_sync(now_ms)
            return "applied"
        self._push(ev, hold and self.pending is None)
        if self.pending is not None:
            return self._try_sync(now_ms)
        return "buffered"

    # snapshot
    def needs_snapshot(self, now_ms):
        if self.synced and not self.refreshing:
            return False
        if self.pending is not None and now_ms - self.pending[4] > FB_PENDING_TIMEOUT_S * 1000:
            self.pending, self.want = None, True                     # a snapshot the stream never caught up with: ask again
        if not self.buf or now_ms - self.last_req_ms < FB_MIN_SNAP_GAP_S * 1000 or now_ms < self.next_req_ms:
            return False
        return self.want

    def mark_requested(self, now_ms):
        self.last_req_ms = now_ms
        self.want = False

    def request_failed(self, now_ms=0):
        """a failed REST snapshot (4xx / 5xx / network): ask again after 5, 10, 20 … ≤ 300 s."""
        self.want = True
        self.req_fails += 1
        self.next_req_ms = now_ms + int(min(300, FB_MIN_SNAP_GAP_S * 2 ** (self.req_fails - 1)) * 1000)

    def begin_refresh(self, now_ms):
        """→ True when a refresh starts (never inside the give-up backoff)."""
        if self.synced and not self.refreshing and now_ms >= self.refresh_next_ms:
            self.refreshing, self.want = True, True
            self.buf.clear()
            self.last_refresh_ms = now_ms
            return True
        return False

    def on_snapshot(self, snap, now_ms=0, limit=FB_LIMIT):
        """snap = the REST body {"lastUpdateId", "bids", "asks"}."""
        try:
            L = int(snap["lastUpdateId"])
            b = [(float(p), float(q)) for p, q in snap.get("bids") or []]
            a = [(float(p), float(q)) for p, q in snap.get("asks") or []]
        except (KeyError, TypeError, ValueError):
            self.request_failed(now_ms)
            return "bad"
        if not (self.refreshing or not self.synced):
            return "old"
        self.req_fails, self.next_req_ms = 0, 0
        self.pending = (L, b, a, int(limit), now_ms)
        return self._try_sync(now_ms)

    def _try_sync(self, now_ms):
        try:
            return self._try_sync_(now_ms)
        except (KeyError, TypeError, ValueError, AttributeError):
            self.unsync()                                            # a malformed buffered event: start over for this coin only
            return "bad"

    def _try_sync_(self, now_ms):
        L, b, a, limit, _ = self.pending
        while self.buf and int(self.buf[0]["u"]) < L:              # older than the snapshot: drop
            self.buf.popleft()
        if not self.buf:
            return "waiting"
        evs = list(self.buf)
        if int(evs[0]["U"]) > L:                                     # events between the snapshot and the buffer are missing
            self.pending, self.want = None, True
            return self._fail("stale", now_ms)
        for i in range(1, len(evs)):
            try:
                ok = int(evs[i]["pu"]) == int(evs[i - 1]["u"])
            except (KeyError, TypeError, ValueError):
                ok = False
            if not ok:                                               # a gap INSIDE the buffer: keep what follows it, ask again
                self.buf = deque(evs[i:], maxlen=self.buf.maxlen)
                self.pending, self.want = None, True
                return self._fail("gap", now_ms)
        eb = min(p for p, _ in b) if len(b) >= limit and b else 0.0
        ea = max(p for p, _ in a) if len(a) >= limit and a else INF
        bids = {p: q for p, q in b if q > 0}
        asks = {p: q for p, q in a if q > 0}
        for ev in evs:
            _apply(bids, asks, ev, eb, ea)
        was = self.synced
        self.bids, self.asks, self.edge_bid, self.edge_ask = bids, asks, eb, ea
        self.last_u = int(evs[-1]["u"])
        self.synced, self.refreshing, self.pending, self.want = True, False, None, False
        self.buf.clear()
        self.synced_ms = now_ms
        self.refresh_tries, self.refresh_next_ms = 0, 0
        m = book_metrics(self.bids, self.asks, eb, ea)
        self.reach0 = min(m["rb"], m["ra"]) if m else None
        return "refreshed" if was else "synced"

    def _fail(self, what, now_ms=0):
        if self.refreshing:                                          # the live book is still exact — give up after 3 tries
            self.refresh_tries += 1
            if self.refresh_tries >= 3:
                self.refreshing, self.want, self.refresh_tries = False, False, 0
                self.buf.clear()
                self.refresh_next_ms = now_ms + FB_REFRESH_MIN_GAP_S * 1000   # back off: no retry storm
        return what

    def refresh_due(self, now_ms, reach_now=None):
        if not self.synced or self.refreshing or now_ms < self.refresh_next_ms:
            return False
        since = max(self.last_refresh_ms, now_ms if self.synced_ms is None else self.synced_ms)
        if now_ms - since >= FB_REFRESH_S * 1000:
            return True
        if reach_now is not None and self.reach0 and reach_now < min(0.5 * self.reach0, FB_WINDOW_PCT) \
                and now_ms - since >= FB_REFRESH_MIN_GAP_S * 1000:
            return True
        return False

    def snap_due(self, now_ms):
        """an fsnap is due when the coin's FB_SNAP_S slot (shifted by its own offset) changed — 30 coins spread over 30 s. Pure."""
        return (now_ms - self.snap_off) // int(FB_SNAP_S * 1000) != self.last_snap_slot

    def mark_snap(self, now_ms):
        self.last_snap_slot = (now_ms - self.snap_off) // int(FB_SNAP_S * 1000)
        self.last_snap_rec_ms = now_ms

    def stale(self, now_ms, stale_s=None):
        """synced, but no diff event for stale_s (FB_STALE_S): the stream for this coin went silent → its book cannot be trusted."""
        s = FB_STALE_S if stale_s is None else stale_s
        return self.synced and self.last_ev_ms is not None and now_ms - self.last_ev_ms > s * 1000


# ── derived numbers ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def crossed(bids, asks):
    """True when the best bid ≥ the best ask — a book that cannot exist on the exchange (out of sync). Pure."""
    return bool(bids) and bool(asks) and max(bids) >= min(asks)


def walk_cost_bps(levels, usd, mid, side):
    """bps vs mid to fill `usd` by walking `levels` [(price, qty)] best first (asks for a buy, bids for a sell); None if not enough. Pure."""
    left, qty, spent = float(usd), 0.0, 0.0
    for p, q in levels:
        take = min(left, p * q)
        spent += take
        qty += take / p
        left -= take
        if left <= 1e-9:
            avg = spent / qty
            return (avg / mid - 1.0) * 1e4 if side == "buy" else (1.0 - avg / mid) * 1e4
    return None


def book_metrics(bids, asks, edge_bid=0.0, edge_ask=INF, window_pct=FB_WINDOW_PCT, with_levels=False):
    """the trusted book → {bb, ba, mid, sp, rb, ra, b[bands], a[bands], cb[sizes], cs[sizes]} (+ bl / al sorted levels within the window
    when with_levels); None when a side is empty or the book is CROSSED (see crossed()). Pure."""
    if not bids or not asks:
        return None
    bb, ba = max(bids), min(asks)
    if not (bb > 0 and ba > bb):
        return None
    mid = (bb + ba) / 2.0
    lo, hi = max(mid * (1 - window_pct / 100.0), edge_bid), min(mid * (1 + window_pct / 100.0), edge_ask)
    bl = sorted(((p, q) for p, q in bids.items() if p >= lo), reverse=True)
    al = sorted((p, q) for p, q in asks.items() if p <= hi)
    bands_b, bands_a = [], []
    for pct in FB_BANDS:
        blo, ahi = mid * (1 - pct / 100.0), mid * (1 + pct / 100.0)
        bands_b.append(sum(p * q for p, q in bl if p >= blo))
        bands_a.append(sum(p * q for p, q in al if p <= ahi))
    out = {"bb": bb, "ba": ba, "mid": mid, "sp": (ba - bb) / mid * 1e4,
           "rb": (1 - edge_bid / mid) * 100.0 if edge_bid > 0 else 100.0,
           "ra": (edge_ask / mid - 1) * 100.0 if edge_ask < INF else 100.0,
           "b": bands_b, "a": bands_a,
           "cb": [walk_cost_bps(al, s, mid, "buy") for s in FB_SIZES],
           "cs": [walk_cost_bps(bl, s, mid, "sell") for s in FB_SIZES]}
    if with_levels:
        out["bl"], out["al"] = bl, al
    return out


def _r(x, nd):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, nd)


def fbook_record(book, m, r_ms):
    """book + book_metrics() → the compact 'fbook' line. Pure."""
    return {"r": r_ms, "s": book.sym, "E": book.last_E, "u": book.last_u, "bb": m["bb"], "ba": m["ba"], "sp": _r(m["sp"], 3),
            "rb": _r(m["rb"], 3), "ra": _r(m["ra"], 3), "b": [round(x) for x in m["b"]], "a": [round(x) for x in m["a"]],
            "cb": [_r(x, 2) for x in m["cb"]], "cs": [_r(x, 2) for x in m["cs"]]}


def fsnap_record(book, m, r_ms, pct=FB_SNAP_PCT):
    """the trusted levels within ±pct % of mid → the 'fsnap' line (needs book_metrics(..., with_levels=True)). Pure."""
    lo, hi = m["mid"] * (1 - pct / 100.0), m["mid"] * (1 + pct / 100.0)
    return {"r": r_ms, "s": book.sym, "E": book.last_E, "u": book.last_u,
            "eb": (book.edge_bid if book.edge_bid > 0 else None), "ea": (book.edge_ask if book.edge_ask < INF else None),
            "b": [[p, q] for p, q in m["bl"] if p >= lo], "a": [[p, q] for p, q in m["al"] if p <= hi]}
