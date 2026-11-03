"""📈 (DECISION_LOG 383) ALT-CORRELATION REGIME STAMP — OBSERVE-ONLY. Changes no trading decision.

WHAT  Two readings stamped on EVERY fill (orders.entry_alt_corr24 / orders.entry_alt_beta24), exactly the frozen OBS-R1 / R1b
      definitions of reports/ML_REGIME_WL_2026-10-26.md §8, as computed by scripts/study_ml_regime_wl_build.py (breadth_feats):
      · R1  (corr) = mean pairwise Pearson correlation of 5m LOG returns over the trailing 288 CLOSED bars, among the top-20 of the
                     universe list (below); a pair is scored only when its last closed bar is ≤ 30 min stale, its 288-bar window is
                     contiguous and it has ≥ 2016 bars (7 days) of history before that bar; needs ≥ 8 scored pairs, else NULL.
      · R1b (beta) = median over the same scored pairs of cov(x, btc) / var(btc) — numpy np.cov (ddof 1) over np.var (ddof 0), the
                     study's exact (slightly biased) formula — on BTC's same trailing 288 closed-bar log returns. NULL when R1 is.
      UNIVERSE: the study reads the top-50 file row DATED the previous UTC day (`universe_daily_top50.csv` date d is ranked on day d − 1's
      quote volume), so an entry on UTC day X ranks USDT perpetuals by the quote volume of UTC day X − 2 (a double lag, kept for parity:
      the 0.3164 / 1.372 thresholds were fitted on it). Top 50 by that volume → drop stables / BTCUSDT / non-[A-Z0-9]+USDT symbols →
      the first 20 (a dropped or unscored pair is never back-filled by #21 — the study does not either).
HOW   AltCorrTracker.refresh() runs once per 5m close in its own task (main.alt_corr_loop, ~1 s after the close): 21 public 5m kline
      reads (limit 290, weight 2 each → 42 weight / 5 min) on its own throttled public client + once per UTC day the universe inputs
      (ticker/24hr weight 40 + exchangeInfo weight 1 + ≤ 100 1d kline reads weight 1). The math is ~20 × 288 floats (< 1 ms).
      stamp() is a pure in-memory read used at the Order build — no I/O, never awaits, never raises: the value is used only when it was
      computed for a bar that closed ≤ STAMP_MAX_LAG_MS before the entry (else NULL). Paper and live read the same tracker (mode-agnostic).
FAIL-SAFE  an empty read (timeout / error) is re-read ONCE in the same bar (operator 11-03, AAVEUSDT #173 beta NULL); any error /
           missing data after that → NULL. Nothing in the trading path reads these values.
"""
import asyncio
import logging
import re
import time

import numpy as np

logger = logging.getLogger(__name__)

BAR = 300_000
DAY = 86_400_000
WIN = 288                    # trailing closed 5m bars (24 h)
HIST_BARS = 2016             # the study's j ≥ 2016 history requirement (7 days of 5m bars before the last closed bar)
STALE_MS = 30 * 60_000       # the study's per-pair staleness tolerance (last closed bar ≤ 30 min before t)
MIN_PAIRS = 8                # the study's okr.sum() ≥ 8
TOP_N, CUT_N = 20, 50
KLINE_LIMIT = WIN + 2        # 289 closes for 288 returns + the forming bar
STAMP_MAX_LAG_MS = 2 * BAR   # a fill may use a value computed up to 2 bars before its own last closed bar (study tolerates 30 min)
UNIVERSE_CANDIDATES = 100    # 1d-volume reads per UTC day (top by rolling 24 h volume) — the top-50 of day X − 2 lies well inside
UNIVERSE_RETRY_S = 600
UNIVERSE_KEEP_DAYS = 6       # 🩹 (DECISION_LOG 443) lists kept in memory for the stamp repair (a restart rebuilds an older day on demand)
REPAIR_CONCURRENCY = 2       # 🩹 the repair reads trickle (2 in flight), never the live refresh's burst of 5
REPAIR_TOL = 5e-5            # 🩹 a recomputed corr "reproduces" the stamp within this (real 11-03 check: right list + right bar 2e-6 … 5e-6 off —
                             # the live read lands 1 s after the close; the next bar 1.7e-4+; another coin list 9.6e-3)
STABLE = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "DAIUSDT", "USDEUSDT", "XUSDUSDT", "BTCDOMUSDT"}
_SYM_RX = re.compile(r"^[A-Z0-9]+USDT$")


# ─────────────────────────────── pure functions (parity-tested against the study) ───────────────────────────────
def universe_day_ms(now_ms: int) -> int:
    """the UTC day whose quote volume ranks the list for an entry at now_ms (= day X − 2, see the module doc)."""
    return (int(now_ms) // DAY) * DAY - 2 * DAY


def universe_top20(day_vols: dict) -> list:
    """{symbol: that day's quote volume} of every USDT perpetual → the study's top-20 (rank order). Pure."""
    rows = []
    for s, v in (day_vols or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f or not str(s).endswith("USDT"):
            continue
        rows.append((f, str(s)))
    top = sorted(rows, reverse=True)[:CUT_N]                     # backtest_phase1_universe: sorted((qv, sym), reverse=True)[:50]
    keep = [s for _, s in top if s not in STABLE and s != "BTCUSDT" and _SYM_RX.match(s)]
    return keep[:TOP_N]


def _closes(rows):
    """kline rows ([open_ms, o, h, l, c, ...], any numeric/str) → (t int64, c float64) sorted / de-duplicated, or (None, None)."""
    try:
        if not rows:
            return None, None
        d = {}
        for r in rows:
            d[int(r[0])] = float(r[4])
        t = np.array(sorted(d), dtype=np.int64)
        c = np.array([d[k] for k in t], dtype=np.float64)
        return t, c
    except Exception:
        return None, None


def _window(t, c, now_ms, listed_ms=None, check_stale=True, check_contig=True):
    """the 288 log returns ending at the last bar CLOSED by now_ms (open ≤ now − 5 min), or None (study rules: stale > 30 min,
    < 7 days of history, non-contiguous window → unscored)."""
    if t is None or len(t) < WIN + 1:
        return None
    j = int(np.searchsorted(t, int(now_ms) - BAR, side="right")) - 1
    if j < WIN:
        return None
    if check_stale and (int(now_ms) - (int(t[j]) + BAR)) > STALE_MS:
        return None
    if listed_ms is not None and int(listed_ms) > int(t[j]) - HIST_BARS * BAR:
        return None
    if check_contig and int(t[j - WIN + 1]) != int(t[j]) - (WIN - 1) * BAR:
        return None
    seg = c[j - WIN:j + 1]
    if not np.all(np.isfinite(seg)) or np.any(seg <= 0):
        return None
    return np.diff(np.log(seg))


def alt_corr_beta(pair_rows: dict, btc_rows, order: list, now_ms: int, listed: dict = None):
    """→ (corr, beta, n_scored). pair_rows {pair: kline rows}, btc_rows BTCUSDT kline rows, order = the top-20 list, listed {pair:
    onboard ms} (None / missing pair = history not checked). Exactly breadth_feats' br_corr24h_top20 / br_beta_btc_24h. Never raises."""
    try:
        X = []
        for p in list(order or [])[:TOP_N]:
            t, c = _closes(pair_rows.get(p))
            w = _window(t, c, now_ms, (listed or {}).get(p))
            if w is not None:
                X.append(w)
        n = len(X)
        if n < MIN_PAIRS:
            return None, None, n
        X = np.vstack(X)
        with np.errstate(all="ignore"):
            C = np.corrcoef(X)
            iu = np.triu_indices(n, 1)
            corr = float(np.nanmean(C[iu]))
        if corr != corr:
            corr = None
        beta = None
        bt, bc = _closes(btc_rows)
        b = _window(bt, bc, now_ms, None, check_stale=True, check_contig=False)   # the study reads BTC's window without a contiguity check
        if corr is not None and b is not None and np.var(b) > 0:
            beta = float(np.median([np.cov(x, b)[0, 1] / np.var(b) for x in X]))
            if beta != beta:
                beta = None
        return corr, beta, n
    except Exception:
        return None, None, 0


# ─────────────────────────────── the per-bar tracker (engine side) ───────────────────────────────
class AltCorrTracker:
    """One value per 5m close. fetch_klines(pair_id, interval, limit) → raw rows or None; fetch_universe() → (rolling {sym: quoteVolume},
    {sym: onboard ms}) or None. Both injected (binance_service in the engine, fakes in tests)."""

    def __init__(self, fetch_klines=None, fetch_universe=None, concurrency: int = 5, call_timeout_s: float = 8.0, is_limited=None):
        self.fetch_klines = fetch_klines
        self.fetch_universe = fetch_universe
        self.is_limited = is_limited     # () → True while the client is banned / backing off (383 review): no new call is started then
        self.skipped_limited = 0         # reads NOT started because the client was limited (observability)
        self.retried = 0                 # empty reads re-tried once in their bar (observability)
        self.concurrency = concurrency
        self.call_timeout_s = call_timeout_s
        self.value = None            # dict(bar_close_ms, corr, beta, n, at)
        self.universe = None         # dict(day_ms, pairs, listed)
        self._uni_hist = {}          # 🩹 day_ms → universe dict (the stamp repair's lists for past days)
        self._uni_repair_fail = {}   # 🩹 day_ms → time of a failed repair rebuild (retried after UNIVERSE_RETRY_S)
        self.repair_skipped = 0      # 🩹 repair reads not started because the client was limited (own counter: refresh()'s stay exact)
        self.repair_error = None     # 🩹 the repair's last error (never overwrites the live last_error)
        self._uni_fail_at = 0.0
        self.last_error = None

    # ── read side (Order build) ──
    def stamp(self, now_ms=None):
        """(corr, beta) usable for a fill at now_ms, else (None, None). Pure memory read — never raises, never awaits."""
        try:
            v = self.value
            if not v:
                return None, None
            now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
            last_close = (now_ms // BAR) * BAR
            if v["bar_close_ms"] > last_close or last_close - v["bar_close_ms"] > STAMP_MAX_LAG_MS:
                return None, None
            r = lambda x: (round(float(x), 6) if x is not None else None)
            return r(v.get("corr")), r(v.get("beta"))
        except Exception:
            return None, None

    # ── write side (own task) ──
    def _limited(self) -> bool:
        try:
            return bool(self.is_limited and self.is_limited())
        except Exception:
            return True

    async def _klines(self, pair, interval, limit):
        if self._limited():                  # (383 review) a 429 / ban mid-bar stops every remaining read of this bar — no fan-out after it
            self.skipped_limited += 1
            return None
        try:
            return await asyncio.wait_for(self.fetch_klines(pair, interval, limit), self.call_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    async def ensure_universe(self, now_ms):
        """the top-20 for now_ms's UTC day (built once per day; on failure the previous day's list stays in use — the study also falls
        back to the latest older list). → list or None."""
        day = universe_day_ms(now_ms)
        u = self.universe
        if u and u.get("day_ms") == day:
            return u["pairs"]
        if u and (day - u.get("day_ms", day)) > DAY:
            logger.warning(f"[ALT_CORR] universe list is stale: built for {time.strftime('%Y-%m-%d', time.gmtime(u['day_ms'] / 1000))} volume, "
                           f"{(day - u['day_ms']) // DAY} days older than needed (rebuild failing) — stamps keep using it")
        if time.time() - self._uni_fail_at < UNIVERSE_RETRY_S or self._limited():
            return u["pairs"] if u else None
        try:
            self.universe = await self._read_universe(day, 4)
            self._keep_universe(self.universe)
            logger.info(f"[ALT_CORR] universe for {time.strftime('%Y-%m-%d', time.gmtime(day / 1000))} volume: {', '.join(self.universe['pairs'])}")
            return self.universe["pairs"]
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._uni_fail_at = time.time()
            self.last_error = f"universe: {str(e)[:120]}"
            logger.warning(f"[ALT_CORR] universe not rebuilt ({str(e)[:120]}) — " + ("previous list kept" if u else "no list yet → NULL stamps"))
            return u["pairs"] if u else None

    def _keep_universe(self, u):
        self._uni_hist[u["day_ms"]] = u
        for d in sorted(self._uni_hist)[:-UNIVERSE_KEEP_DAYS]:
            self._uni_hist.pop(d, None)

    async def _read_universe(self, day, daily_bars, concurrency=None):
        """→ dict(day_ms, pairs, listed) for the list ranked on `day`'s quote volume; raises on any shortfall."""
        got = await asyncio.wait_for(self.fetch_universe(), 20.0) if self.fetch_universe else None
        if not got:
            raise RuntimeError("universe inputs unreadable")
        rolling, listed = got
        cands = [s for _, s in sorted(((float(v or 0), s) for s, v in (rolling or {}).items() if str(s).endswith("USDT")),
                                      reverse=True)[:UNIVERSE_CANDIDATES]]
        sem = asyncio.Semaphore(concurrency or self.concurrency)

        async def one(s):
            async with sem:
                rows = await self._klines(s, "1d", daily_bars)
            for r in rows or []:
                try:
                    if int(r[0]) == day:
                        return s, float(r[7])
                except (TypeError, ValueError, IndexError):
                    return s, None
            return s, None
        vols = {s: v for s, v in await asyncio.gather(*[one(s) for s in cands]) if v is not None}
        if self._limited():
            raise RuntimeError("client rate-limited during the rebuild — retried later")
        if len(vols) < CUT_N:
            raise RuntimeError(f"only {len(vols)} day volumes read")
        pairs = universe_top20(vols)
        if len(pairs) < TOP_N:
            raise RuntimeError(f"top list has {len(pairs)} pairs")
        return dict(day_ms=day, pairs=pairs, listed={k: v for k, v in (listed or {}).items() if v})

    async def refresh(self, now_ms=None):
        """compute the value for the last 5m close (idempotent per bar). Never raises (except its own cancellation)."""
        try:
            now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
            bar_close = (now_ms // BAR) * BAR
            if self.value and self.value.get("bar_close_ms") == bar_close:
                return self.value
            if self._limited():
                self.last_error = "client rate-limited / banned — this bar skipped"
                return None
            pairs = await self.ensure_universe(now_ms)
            if not pairs:
                return None
            sem = asyncio.Semaphore(self.concurrency)

            async def one(p):
                async with sem:
                    return p, await self._klines(p, "5m", KLINE_LIMIT)
            got = dict(await asyncio.gather(*[one(p) for p in list(pairs) + ["BTCUSDT"]]))
            # one retry of every read that came back empty (timeout / error) — operator 11-03: AAVEUSDT #173 stamped corr with a NULL beta
            # because the single BTCUSDT read failed; the same pair is re-read, never back-filled by #21 (study parity)
            failed = [p for p, rows in got.items() if not rows]
            if failed and not self._limited():
                s0 = self.skipped_limited
                got.update(dict(await asyncio.gather(*[one(p) for p in failed])))
                self.retried += len(failed) - (self.skipped_limited - s0)   # reads actually re-sent (a limited client skips them)
            if self._limited():               # limited mid-bar: the value would be built from a partial read → no value for this bar
                self.last_error = "client rate-limited mid-bar — this bar skipped"
                return None
            btc = got.pop("BTCUSDT", None)
            listed = (self.universe or {}).get("listed") or {}
            corr, beta, n = alt_corr_beta(got, btc, pairs, bar_close, {p: listed.get(p) for p in pairs})
            self.value = dict(bar_close_ms=bar_close, corr=corr, beta=beta, n=n, at=time.time())
            if corr is None:
                self.last_error = f"{n} of {len(pairs)} pairs scored (< {MIN_PAIRS}) → NULL"
            elif beta is None:
                why = "BTCUSDT 5m read empty after a retry" if not btc else "BTCUSDT 5m window unusable (short / > 30 min stale / flat)"
                self.last_error = f"{why} → beta NULL"
                logger.warning(f"[ALT_CORR] beta NULL for the {time.strftime('%H:%M', time.gmtime(bar_close / 1000))} UTC bar: {why} — "
                               f"fills stamp corr only")
            return self.value
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.last_error = str(e)[:120]
            logger.warning(f"[ALT_CORR] refresh failed ({str(e)[:120]}) — fills stamp NULL until the next bar")
            return None

    # ── 🩹 stamp repair (DECISION_LOG 443, operator 11-03: "build the self-repair") ──
    async def universe_for_day(self, day):
        """the universe dict ranked on `day`'s volume: the live one, a kept one, else rebuilt from the daily klines. None on failure."""
        for u in (self.universe, self._uni_hist.get(day)):
            if u and u.get("day_ms") == day:
                return u
        if self._limited() or time.time() - self._uni_repair_fail.get(day, 0.0) < UNIVERSE_RETRY_S:
            return None
        try:
            back = max(0, (universe_day_ms(int(time.time() * 1000)) - int(day)) // DAY)
            u = await self._read_universe(int(day), 4 + int(back), REPAIR_CONCURRENCY)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._uni_repair_fail[day] = time.time()
            self.repair_error = f"repair universe: {str(e)[:120]}"
            return None
        self._keep_universe(u)
        return u

    async def value_at(self, bar_close, u):
        """(corr, beta, n) for the bar closing at bar_close, from the 5m klines opened before it (endTime) — the same reads and math as
        refresh(), one retry of an empty read. (None, None, 0) when limited / unreadable. Never raises (except cancellation)."""
        pairs = list(u["pairs"])
        sem = asyncio.Semaphore(REPAIR_CONCURRENCY)

        async def one(p):
            async with sem:
                if self._limited():
                    self.repair_skipped += 1
                    return p, None
                try:
                    return p, await asyncio.wait_for(self.fetch_klines(p, "5m", KLINE_LIMIT, end_ms=int(bar_close) - 1), self.call_timeout_s)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return p, None
        got = dict(await asyncio.gather(*[one(p) for p in pairs + ["BTCUSDT"]]))
        failed = [p for p, rows in got.items() if not rows]
        if failed and not self._limited():
            got.update(dict(await asyncio.gather(*[one(p) for p in failed])))
        if self._limited():
            return None, None, 0
        btc = got.pop("BTCUSDT", None)
        return alt_corr_beta(got, btc, pairs, int(bar_close), {p: (u.get("listed") or {}).get(p) for p in pairs})

    async def repair(self, opened_ms, stamped_corr=None):
        """→ (dict(corr, beta, bar_close_ms), None) or (None, why). With a stamped corr, the bars the stamp could have come from (the
        entry's last close and up to STAMP_MAX_LAG_MS before it) are recomputed and only the one that REPRODUCES the stamped corr is used
        (proves the same universe + data); without one, the entry's last close is used, as the study builds it."""
        try:
            last = (int(opened_ms) // BAR) * BAR
            bars = [last - k * BAR for k in range(STAMP_MAX_LAG_MS // BAR + 1)] if stamped_corr is not None else [last]
            seen, best = [], None
            for b in bars:
                u = await self.universe_for_day(universe_day_ms(b))
                if not u:                                           # (review) one day's list unreadable never discards a match found on another bar
                    seen.append("no list")
                    continue
                c, be, n = await self.value_at(b, u)
                if c is None:
                    seen.append("NULL")
                    continue
                res = dict(corr=round(c, 6), beta=(round(be, 6) if be is not None else None), bar_close_ms=b)
                if stamped_corr is None:
                    return res, None
                seen.append(f"{c:.6f}")
                d = abs(c - float(stamped_corr))
                if d <= REPAIR_TOL and (best is None or d < best[0]):
                    best = (d, res)                                 # the CLOSEST bar within the tolerance, never just the first
            if best is not None:
                return best[1], None
            return None, f"no bar reproduces the stamped corr {stamped_corr} (recomputed {', '.join(seen)}) — another coin list or bar"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return None, f"error: {str(e)[:120]}"

