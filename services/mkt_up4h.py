"""🌡 (DECISION_LOG 489) MARKET-HOT BREADTH on a ROLLING top-100 (V2) — OBSERVE-ONLY. Changes no trading decision.

WHAT  Two readings stamped on EVERY fill (orders.entry_mkt_up4h_pct / orders.entry_mkt_up4h_n) — the ML_MKT_HOT measure (DECISION_LOG
      416) on the rolling universe V2 chosen by pre-registered rules in reports/ML_MKT_HOT_ROLLING_2026-11-06.md §7 / §10. It REPLACED the
      frozen 108-coin list everywhere on 2026-11-06 (no dual measure).
      · universe (one list per UTC day D) = the 100 USDT-margined PERPETUAL symbols with status TRADING and the highest quote volume of UTC
        day D − 1, after the exclusions: BTCUSDT, ETHUSDT, stablecoin / USD-pegged perps (base in STABLE_BASES, or ≥ 10 of the last 30 daily
        closes with > 95 % of them inside 0.97 … 1.03). NOT excluded: the pair blacklist, Alpha coins, new listings, non-COIN underlyings
        (the measure is the MARKET, not the tradable list). Ties in volume → symbol name (deterministic).
      · the up-4h share at an entry = on the last CLOSED 1h bar H (NULL when H closed > 2 h before the entry): #(close_H / close_{H−4h} − 1 > 0) ÷
        #(members with both closes); NULL when < 30 members have both closes. hot = share > 0.50 (the frozen DL 416 cut; never re-fit).
      · entry_mkt_up4h_n = the members with both closes (set whenever the hour was computed — a NULL share WITH an n means < 30).
ONE RULE  select_universe() and up4h_from_closes() below are the ONLY implementation: the engine's live stamp and every historical
      rebuild (scripts/scout_ml_mkt_hot_watch.py — the scouts / best case read it) call these same pure functions.
HOW   MktUp4hTracker.refresh() runs once per hour in its own task (main.mkt_up4h_loop, HH:00:05; 00:00:30 at the UTC day turn):
      ① the day's list once per UTC day — one /fapi/v1/ticker/24hr (weight 40) + exchangeInfo (weight 1); in the first 10 min of the day the
      24 h ticker volume IS day D − 1; later (a cold start mid-day with no stored list) the top-150 by 24 h volume are re-ranked on their exact
      D − 1 quote volume (1d klines, weight 1 each, once). Near-$1 candidates get a 1d-kline peg check (weight 1 each, usually 0–3 a day).
      The list is FROZEN per day in table mkt_up4h_universes (first build wins — a restart reuses it). On a failed build yesterday's list
      stays in use (logged). ② the 1h closes — /fapi/v1/klines interval 1h limit 6 for the 100 members (weight 1 each) → ≈ 100 weight / hour
      + ≈ 41 / day. stamp() is a pure memory read at the Order build (no I/O, never raises): a value only for the entry's own last closed
      hour, else NULL — the hourly 🩹 repair (main.mkt_up4h_repair_pass) recomputes NULL stamps of recent bot fills from the stored list
      + historical klines (endTime), the same functions. Paper and live read the same tracker (mode-agnostic).
FAIL-SAFE  a failed member read is retried once; still > MAX_FAILED_READS failures → no value for that hour (never a share built on a
           visibly partial list → NULL stamps, repaired later); ≤ MAX_FAILED_READS failures count as "no data" exactly like the rebuild.
           🧱 Since DECISION_LOG 527 the trading path READS stamp() once: the armed ML_MKT_HOT filter (services/watch_filters.ml_blocks via
           TradingEngine.open_position) refuses a momentum LONG when the share is > 0.50. A NULL stamp (start of the hour, a failed read,
           switched off) never blocks — fail-open, the filter simply does not fire.
"""
import asyncio
import json
import logging
import math
import time

logger = logging.getLogger(__name__)

HOUR = 3_600_000
DAY = 86_400_000
TOP_N = 100
MIN_MEMBERS = 30                 # ≥ 30 members with both closes, else NULL (the DL 416 MIN_COINS)
HOT_CUT = 0.50                   # frozen DL 416 cut: hot = share > 0.50
LOOKBACK_MS = 4 * HOUR           # close vs the close 4 h earlier
STALE_MS = 2 * HOUR              # the last closed bar may be at most 2 h old at the entry
KLINE_LIMIT = 6                  # 1h klines per member per hour (5 closed bars H−4 … H + the forming one)
EXCLUDED = frozenset({"BTCUSDT", "ETHUSDT"})
STABLE_BASES = frozenset({"USDC", "FDUSD", "TUSD", "USDP", "BUSD", "USD1", "USDE", "DAI", "PYUSD", "RLUSD", "U", "XUSD", "USDS", "EUR", "EURI",
                          "AEUR", "USDD", "BFUSD"})   # the study's named list (scripts/study_ml_mkt_hot_rolling.py STABLE_BASES), verbatim
PEG_LO, PEG_HI, PEG_SHARE = 0.97, 1.03, 0.95         # the study's empirical peg test: > 95 % of daily closes within 0.97 … 1.03 …
PEG_MIN_CLOSES, PEG_WINDOW = 10, 30                  # … on ≥ 10 of the last 30 daily closes
PEG_PREFILTER = (0.90, 1.10)     # live: only candidates whose last price sits here get the 1d-kline peg read (a pegged coin trades near $1)
PEG_CHECK_DEPTH = 20             # live: peg reads for candidates down to rank TOP_N + this (a pegged coin above the cut is replaced by #101 …)
FRESH_DAY_MS = 10 * 60_000       # in the first 10 min of a UTC day the 24 h ticker volume = the previous UTC day (spec §10.2)
EXACT_CANDIDATES = 150           # a mid-day cold build re-ranks this many (top by 24 h volume) on their exact previous-day 1d volume
MAX_FAILED_READS = 10            # member reads still failing after the retry above this → no value for the hour
UNIVERSE_RETRY_S = 600
UNIVERSE_KEEP_DAYS = 3           # day lists kept in memory for the repair (older days are read back from the table)
WEIGHT_CEILING = 500             # 🔒 the IP's X-MBX-USED-WEIGHT-1M (shared with trading) above this → no read starts (deferred). 11-06 (DL 491):
                                 # 600 → 500 — the live 18:00 refresh read IP weight 313 incl. its own 100 reads (engine ≈ 210–310 / min); 400
                                 # would skip busy hours, 500 stays far below Binance's 2,400 / min with the reads now paced
READ_GAP_S = 0.5                 # 🔒 (11-06 tuning) the live tracker starts at most ~2 reads / s → an hour's ~100 reads spread over ~1 min
COLD_GAP_S = 1.5                 # 🔒 … and the one-off mid-day cold ranking (~150 1d reads) over ~4 min, never a burst next to the engine
CONCURRENCY = 3                  # 🔒 reads in flight; with READ_GAP_S the live hour's ~100 reads take ≈ 50 s (never a burst on the shared IP budget)
CALL_TIMEOUT_S = 5.0


# ─────────────────────────────── pure functions (the ONE rule: engine stamp = historical rebuild) ───────────────────────────────
def universe_day_ms(now_ms: int) -> int:
    """the UTC day whose list applies at now_ms (the list is ranked on the day BEFORE it)."""
    return (int(now_ms) // DAY) * DAY


def last_closed_bar_ms(entry_ms: int) -> int:
    """close time of the last CLOSED 1h bar at entry_ms (a bar closing exactly at entry_ms counts as closed)."""
    return (int(entry_ms) // HOUR) * HOUR


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def is_pegged(base, closes=None) -> bool:
    """stablecoin / USD-pegged: base in STABLE_BASES, or ≥ PEG_MIN_CLOSES of the last PEG_WINDOW daily closes with > PEG_SHARE inside
    PEG_LO … PEG_HI (the study's is_stable). PURE."""
    if str(base or "") in STABLE_BASES:
        return True
    c = [x for x in (_num(v) for v in (closes or [])[-PEG_WINDOW:]) if x is not None]
    if len(c) >= PEG_MIN_CLOSES:
        return sum(1 for x in c if PEG_LO <= x <= PEG_HI) / len(c) > PEG_SHARE
    return False


def select_universe(day_vols: dict, statuses: dict = None, bases: dict = None, daily_closes: dict = None, top_n: int = TOP_N) -> list:
    """{symbol: quote volume of the previous UTC day} → the V2 list (rank order, ≤ top_n). PURE, never raises.
    statuses {symbol: exchangeInfo status} of every USDT PERPETUAL — given → only status TRADING is eligible and a symbol missing from it is
    out (not a live perpetual); None (historical rebuild: as-of status unknown) → no status filter. bases {symbol: baseAsset} (missing →
    symbol minus 'USDT'). daily_closes {symbol: [closes, oldest first, the days BEFORE the list's day]} for the peg test (missing → named
    list only). Volume must be > 0 and finite. Ties → symbol name."""
    rows = []
    for s, v in (day_vols or {}).items():
        s = str(s)
        f = _num(v)
        if f is None or f <= 0 or not s.endswith("USDT") or s in EXCLUDED:
            continue
        if statuses is not None and statuses.get(s) != "TRADING":
            continue
        base = (bases or {}).get(s) or s[:-4]
        if is_pegged(base, (daily_closes or {}).get(s)):
            continue
        rows.append((-f, s))
    return [s for _, s in sorted(rows)[:max(0, int(top_n))]]


def closes_from_rows(rows, now_ms=None) -> dict:
    """raw kline rows ([open_ms, o, h, l, c, …]) → {bar CLOSE ms: close} of the bars CLOSED by now_ms (all rows when now_ms is None). PURE."""
    out = {}
    for r in rows or []:
        try:
            t, c = int(r[0]) + HOUR, float(r[4])
        except (TypeError, ValueError, IndexError):
            continue
        if (now_ms is None or t <= int(now_ms)) and math.isfinite(c) and c > 0:
            out[t] = c
    return out


def up4h_from_closes(closes: dict, members, entry_ms: int):
    """closes {symbol: {bar close ms: close}} + the day's members + the entry time → (up-4h share | None, n with both closes, bar close ms | None).
    H = the newest bar close ≤ entry_ms over the members' series; NULL when none or older than STALE_MS; r = close_H / close_{H−4h} − 1 for
    every member with both; share = #(r > 0) ÷ n (strict >: flat is not up); NULL when n < MIN_MEMBERS. PURE, never raises."""
    try:
        entry_ms = int(entry_ms)
        mem = [m for m in dict.fromkeys(members or []) if m in (closes or {})]
        H = None
        for m in mem:
            ts = [t for t in closes[m] if t <= entry_ms]
            if ts:
                h = max(ts)
                H = h if H is None or h > H else H
        if H is None or entry_ms - H > STALE_MS:
            return None, 0, None
        up = n = 0
        for m in mem:
            a, b = closes[m].get(H), closes[m].get(H - LOOKBACK_MS)
            if a is None or b is None or not b:
                continue
            n += 1
            up += (a / b - 1) > 0
        return ((up / n) if n >= MIN_MEMBERS else None), n, H
    except Exception:
        return None, 0, None


def is_hot(pct):
    """True / False / None (unscored). PURE."""
    b = _num(pct)
    return None if b is None else b > HOT_CUT


# ─────────────────────────────── the hourly tracker (engine side) ───────────────────────────────
class MktUp4hTracker:
    """One value per closed 1h bar. Injected (binance_service in the engine, fakes in tests):
    fetch_klines(symbol, interval, limit, end_ms=None) → raw kline rows or None ·
    fetch_universe() → dict(vols {sym: 24 h quoteVolume}, statuses {sym: status}, bases {sym: baseAsset}, prices {sym: last}) or None ·
    load_universe(day_ms) → stored members list or None · save_universe(day_ms, members, how) → the stored list (an earlier row wins) ·
    is_limited() → True while the client is banned / backing off · used_weight() → the IP's last X-MBX-USED-WEIGHT-1M (or None) ·
    enabled() → the mkt_up4h_enabled kill switch (False → no read at all, every stamp NULL).
    🔒 ZERO-RISK CONTRACT (operator 11-06): stamp() never awaits / never does I/O; no read starts while limited / banned, while the IP weight
    is above WEIGHT_CEILING or while switched off — and a read that sees one of those mid-way stops the rest of the pass."""

    def __init__(self, fetch_klines=None, fetch_universe=None, is_limited=None, load_universe=None, save_universe=None,
                 concurrency: int = CONCURRENCY, call_timeout_s: float = CALL_TIMEOUT_S, clock=None, used_weight=None, enabled=None,
                 read_gap_s: float = 0.0, cold_gap_s: float = None):
        self.fetch_klines = fetch_klines
        self.fetch_universe = fetch_universe
        self.is_limited = is_limited
        self.load_universe = load_universe
        self.save_universe = save_universe
        self.concurrency = concurrency
        self.call_timeout_s = call_timeout_s
        self.clock = clock or (lambda: time.time() * 1000)
        self.used_weight = used_weight
        self.enabled = enabled
        self.read_gap_s = float(read_gap_s or 0.0)       # min seconds between read STARTS (0 = no pacing — the tests' fakes)
        self.cold_gap_s = float(cold_gap_s if cold_gap_s is not None else self.read_gap_s)
        self._gap = self.read_gap_s
        self._next_ok = 0.0
        self._pace_lock = asyncio.Lock()
        self.skipped_weight = 0          # reads NOT started because the IP weight was above WEIGHT_CEILING
        self.reads = 0                   # reads started (≈ weight: 1h limit 6 / 1d limit ≤ 31 are weight 1 each)
        self.value = None            # dict(bar_close_ms, pct, n, size, failed, at)
        self.universe = None         # dict(day_ms, members, how)
        self._hist = {}              # day_ms → members (the repair's lists)
        self._uni_fail_at = 0.0
        self.skipped_limited = 0
        self.last_error = None
        self.repair_error = None

    # ── read side (Order build) ──
    def stamp(self, now_ms=None):
        """(up-4h share | None, n | None) for a fill at now_ms: the value of the entry's own last closed hour, else (None, None). Pure memory
        read — never raises, never awaits."""
        try:
            if not self._on():
                return None, None
            v = self.value
            if not v:
                return None, None
            now_ms = int(now_ms if now_ms is not None else self.clock())
            if v["bar_close_ms"] != last_closed_bar_ms(now_ms):
                return None, None
            b = v.get("pct")
            return (round(float(b), 6) if b is not None else None), (int(v["n"]) if v.get("n") is not None else None)
        except Exception:
            return None, None

    # ── write side (own task) ──
    def _on(self) -> bool:
        try:
            return True if self.enabled is None else bool(self.enabled())
        except Exception:
            return False

    def _weight(self):
        try:
            w = self.used_weight() if self.used_weight else None
            return int(w) if w is not None else None
        except Exception:
            return None

    def _heavy(self) -> bool:
        w = self._weight()
        return w is not None and w > WEIGHT_CEILING

    def _limited(self) -> bool:
        """True = start no read: switched off, client limited / banned (an unreadable state counts as limited), or IP weight > ceiling."""
        try:
            if not self._on() or (self.is_limited and self.is_limited()):
                return True
        except Exception:
            return True
        return self._heavy()

    async def _pace(self):
        """(11-06 tuning) spread the reads: at most one read START per `_gap` seconds across all in-flight workers."""
        if self._gap <= 0:
            return
        async with self._pace_lock:
            wait = self._next_ok - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_ok = time.monotonic() + self._gap

    async def _klines(self, sym, interval, limit, end_ms=None):
        if self._gap > 0 and self._limited():                # (491 review) already limited → skip at once, never queue on the pace
            self.skipped_limited += 1
            self.skipped_weight += self._heavy()
            return None
        await self._pace()                                  # paced, then the limiter below sees the state right before the read
        if self._limited():
            self.skipped_limited += 1
            self.skipped_weight += self._heavy()
            return None
        self.reads += 1
        try:
            if end_ms is None:
                return await asyncio.wait_for(self.fetch_klines(sym, interval, limit), self.call_timeout_s)
            return await asyncio.wait_for(self.fetch_klines(sym, interval, limit, end_ms=end_ms), self.call_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    async def _stored(self, day):
        if not self.load_universe:
            return None
        try:
            m = await self.load_universe(int(day))
            return list(m) if m else None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[MKT_UP4H] stored list unreadable ({str(e)[:120]}) — building it")
            return None

    async def _store(self, day, members, how):
        if not self.save_universe:
            return members
        try:
            got = await self.save_universe(int(day), list(members), how)
            return list(got) if got else members
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[MKT_UP4H] list not stored ({str(e)[:120]}) — used for this process only")
            return members

    def _keep(self, day, members):
        self._hist[int(day)] = list(members)
        for d in sorted(self._hist)[:-UNIVERSE_KEEP_DAYS]:
            self._hist.pop(d, None)

    async def _build(self, day, now_ms):
        """→ (members, how) for UTC day `day` from live inputs; raises on any shortfall."""
        if self._limited():
            raise RuntimeError("limited / banned / IP weight above the ceiling / switched off")
        got = await asyncio.wait_for(self.fetch_universe(), 20.0) if self.fetch_universe else None
        if not got or not got.get("vols"):
            raise RuntimeError("universe inputs unreadable")
        vols, statuses, bases, prices = got["vols"], got.get("statuses") or {}, got.get("bases") or {}, got.get("prices") or {}
        how = "ticker-24h"
        if int(now_ms) - int(day) > FRESH_DAY_MS:      # mid-day: the 24 h ticker is not day D − 1 → exact 1d volumes for the top candidates
            cands = select_universe(vols, statuses, bases, None, EXACT_CANDIDATES)
            sem = asyncio.Semaphore(self.concurrency)
            self._gap = self.cold_gap_s                     # (11-06 tuning) the one-off cold ranking goes slower (~4 min)

            async def one(s):
                async with sem:
                    rows = await self._klines(s, "1d", 3)
                for r in rows or []:
                    try:
                        if int(r[0]) == int(day) - DAY:
                            return s, float(r[7])
                    except (TypeError, ValueError, IndexError):
                        return s, None
                return s, None
            try:
                ex = {s: v for s, v in await asyncio.gather(*[one(s) for s in cands]) if v is not None}
            finally:
                self._gap = self.read_gap_s                 # back to the normal pace, whatever happened
            if self._limited():
                raise RuntimeError("client rate-limited during the rebuild — retried later")
            if len(ex) < TOP_N:
                raise RuntimeError(f"only {len(ex)} previous-day volumes read")
            vols, how = ex, "1d-exact"
        # peg reads: near-$1 candidates down to rank TOP_N + PEG_CHECK_DEPTH (named stables are already out)
        ranked = select_universe(vols, statuses, bases, None, TOP_N + PEG_CHECK_DEPTH)
        near = [s for s in ranked if (lambda p: p is not None and PEG_PREFILTER[0] <= p <= PEG_PREFILTER[1])(_num(prices.get(s)))]
        closes = {}
        for s in near:
            if self._limited():
                raise RuntimeError("limited during the peg reads — retried later")
            rows = await self._klines(s, "1d", PEG_WINDOW + 1)
            c = [float(r[4]) for r in rows or [] if _num(r[4]) is not None and int(r[0]) < int(day)]
            if c:
                closes[s] = c
        members = select_universe(vols, statuses, bases, closes, TOP_N)
        if len(members) < TOP_N:
            raise RuntimeError(f"list has {len(members)} members")
        return members, how

    async def ensure_universe(self, now_ms):
        """the members for now_ms's UTC day (stored → built → yesterday's kept). → list or None."""
        day = universe_day_ms(now_ms)
        u = self.universe
        if u and u.get("day_ms") == day:
            return u["members"]
        st = await self._stored(day)
        if st:
            self.universe = dict(day_ms=day, members=st, how="stored")
            self._keep(day, st)
            return st
        if time.time() - self._uni_fail_at < UNIVERSE_RETRY_S or self._limited():
            return u["members"] if u else None
        r0 = self.reads
        try:
            members, how = await self._build(day, now_ms)
            members = await self._store(day, members, how)
            self.universe = dict(day_ms=day, members=members, how=how)
            self._keep(day, members)
            logger.info(f"[MKT_UP4H] universe {time.strftime('%Y-%m-%d', time.gmtime(day / 1000))}: {len(members)} members ({how}) · "
                        f"{self.reads - r0} kline reads + ticker/24hr 40 + exchangeInfo 1 (≈ weight) · IP weight {self._weight()} · "
                        f"top {', '.join(members[:5])} …")
            return members
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._uni_fail_at = time.time()
            self.last_error = f"universe: {str(e)[:120]}"
            if u:
                late = (day - u["day_ms"]) // DAY
                logger.warning(f"[MKT_UP4H] list not rebuilt ({str(e)[:120]}) — keeping the {late}-day-old list"
                               + (" ⚠ (2+ days missed)" if late >= 2 else ""))
                return u["members"]
            logger.warning(f"[MKT_UP4H] list not built ({str(e)[:120]}) — no list yet → NULL stamps")
            return None

    async def _read_closes(self, members, end_ms=None):
        """→ ({sym: rows or None}, failed list) — one retry of every empty read."""
        sem = asyncio.Semaphore(self.concurrency)

        async def one(s):
            async with sem:
                return s, await self._klines(s, "1h", KLINE_LIMIT, end_ms)
        got = dict(await asyncio.gather(*[one(s) for s in members]))
        failed = [s for s, r in got.items() if not r]
        if failed and not self._limited():
            got.update(dict(await asyncio.gather(*[one(s) for s in failed])))
        return got, [s for s, r in got.items() if not r]

    async def refresh(self, now_ms=None):
        """compute the value for the last closed hour (idempotent per hour). Never raises (except its own cancellation)."""
        try:
            now_ms = int(now_ms if now_ms is not None else self.clock())
            bar = last_closed_bar_ms(now_ms)
            if self.value and self.value.get("bar_close_ms") == bar:
                return self.value
            if not self._on():
                self.last_error = "switched off (mkt_up4h_enabled = false)"
                return None
            if self._limited():
                self.last_error = (f"IP used weight {self._weight()} > {WEIGHT_CEILING} — deferred" if self._heavy()
                                   else "client rate-limited / banned — this hour skipped")
                logger.warning(f"[MKT_UP4H] {time.strftime('%H:%M', time.gmtime(bar / 1000))} UTC refresh skipped: {self.last_error}")
                return None
            r0 = self.reads
            members = await self.ensure_universe(now_ms)
            if not members:
                return None
            if (self.universe or {}).get("day_ms") != universe_day_ms(now_ms):   # (489 safety review) yesterday's list at the day turn → this
                self.last_error = "today's list not built yet — this hour skipped (repaired later)"   # hour stays NULL; the repair uses the stored day list
                logger.warning(f"[MKT_UP4H] {time.strftime('%H:%M', time.gmtime(bar / 1000))} UTC refresh skipped: {self.last_error}")
                return None
            got, failed = await self._read_closes(members)
            if self._limited():
                self.last_error = ("IP used weight above the ceiling mid-read" if self._heavy() else "client rate-limited / banned mid-read") + \
                    " — this hour skipped"
                logger.warning(f"[MKT_UP4H] {time.strftime('%H:%M', time.gmtime(bar / 1000))} UTC: {self.last_error} "
                               f"({self.reads - r0} reads) — fills stamp NULL until the 🩹 repair")
                return None
            if len(failed) > MAX_FAILED_READS:
                self.last_error = f"{len(failed)} member reads failed after a retry ({', '.join(failed[:4])}) — this hour skipped"
                logger.warning(f"[MKT_UP4H] {time.strftime('%H:%M', time.gmtime(bar / 1000))} UTC: {self.last_error}; fills stamp NULL "
                               f"until the 🩹 repair")
                return None
            closes = {s: closes_from_rows(r, now_ms) for s, r in got.items() if r}
            b, n, H = up4h_from_closes(closes, members, now_ms)
            if H != bar:                    # the hour's own bar is not in the data yet → no value (the stamp would not match the rebuild)
                self.last_error = f"bar {time.strftime('%H:%M', time.gmtime(bar / 1000))} UTC not in the klines yet"
                return None
            self.value = dict(bar_close_ms=bar, pct=b, n=n, size=len(members), failed=len(failed), at=time.time())
            logger.info(f"[MKT_UP4H] {time.strftime('%H:%M', time.gmtime(bar / 1000))} UTC bar: up-4h share "
                        f"{('%.3f' % b) if b is not None else 'NULL'} ({'HOT' if (b is not None and b > HOT_CUT) else 'not hot'}) · "
                        f"{n} of {len(members)} members with data · {len(failed)} failed · {self.reads - r0} reads (≈ weight) · "
                        f"IP weight {self._weight()}")
            if b is None:
                self.last_error = f"{n} of {len(members)} members with both closes (< {MIN_MEMBERS}) → NULL"
            elif n < 90:
                logger.warning(f"[MKT_UP4H] ⚠ only {n} of {len(members)} members with data at "
                               f"{time.strftime('%H:%M', time.gmtime(bar / 1000))} UTC (up-4h share {b:.3f})")
            return self.value
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.last_error = str(e)[:120]
            logger.warning(f"[MKT_UP4H] refresh failed ({str(e)[:120]}) — fills stamp NULL this hour")
            return None

    # ── 🩹 stamp repair ──
    async def members_for_day(self, day):
        """the day's FROZEN list (live / kept / stored) or None — never rebuilt for a past day (its exact inputs are gone)."""
        u = self.universe
        if u and u.get("day_ms") == int(day):
            return u["members"]
        if int(day) in self._hist:
            return self._hist[int(day)]
        st = await self._stored(day)
        if st:
            self._keep(day, st)
        return st

    async def repair(self, entry_ms):
        """→ (dict(pct, n, bar_close_ms), None) or (None, why): the stamp an entry at entry_ms should carry, from the day's stored list +
        the 1h klines ending at its last closed bar — the same functions as refresh(). Never raises (except cancellation)."""
        try:
            if self._limited():
                return None, "limited / banned / IP weight above the ceiling / switched off"
            entry_ms = int(entry_ms)
            members = await self.members_for_day(universe_day_ms(entry_ms))
            if not members:
                return None, "no stored list for that day"
            bar = last_closed_bar_ms(entry_ms)
            got, failed = await self._read_closes(members, end_ms=bar - 1)   # bars opened before the entry's last close (H − 5 h … H − 1 h)
            if self._limited():
                return None, "client rate-limited"
            if len(failed) > MAX_FAILED_READS:
                return None, f"{len(failed)} member reads failed"
            closes = {s: closes_from_rows(r, entry_ms) for s, r in got.items() if r}
            b, n, H = up4h_from_closes(closes, members, entry_ms)
            if H is None:
                return None, "no closed bar ≤ 2 h before the entry"
            if H != bar:                                     # (489 safety review) one rule with the live stamp: the entry's OWN last closed bar
                return None, "the entry's own last closed bar is missing"
            return dict(pct=(round(b, 6) if b is not None else None), n=n, bar_close_ms=H), None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return None, f"error: {str(e)[:120]}"


def members_json(members) -> str:
    return json.dumps(list(members))
