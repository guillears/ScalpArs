#!/usr/bin/env python3
"""🔭 Opportunity scout v2 (operator, 2026-09-30: "monitor to detect things we are not seeing, like this morning's SURGE").

READ-ONLY research job — public Binance market data + the operator's order exports. It never talks to the bot and never places
orders (ccxt without keys; fetch_ohlcv / fetch_tickers / load_markets only). Run hourly (local scheduled task); 26 h lookback.
Goal: find REPEATABLE moves the bot does not trade, with evidence strong enough to justify a pre-registered backtest.

UNIVERSE = the bot's own: trading_pairs_limit (50) by 24 h volume among USDT perps with underlyingType COIN, listed ≥
new_listing_filter_days, no "Alpha" sub-type (same order as binance_service.get_top_futures_pairs), minus pair_blacklist /
no_trade_pairs / BTC / ETH. Every event records whether its pair is in that universe (and its rank).

EVENTS (pre-declared, deliberately looser than the live sleeves; closed bars only; measured from the FIRST bar of a cluster):
  BTC_MOVE      BTC 30-min |r| ≥ 0.8 %. held_first = services.surge.surge_trigger's legs (live config values) on the FIRST bar.
  BREADTH_BURST ≥ 40 % of the universe (with data on that bar) moved ≥ 1 % the same way over 15 min.
  ALT_SPIKE     one pair: 15-min |r| ≥ 4 % with bar quote volume ≥ 5× its prior-288 median.
  TREND         one pair: 2 h |r| ≥ 3 %, efficiency ≥ 0.5 AND its largest single bar ≤ 35 % of the path (a grind, not a candle).
                scope = MARKET (≥ 3 same-direction TRENDs on that bar, or a same-side BREADTH_BURST within 15 min) or PAIR.
  Clusters: same type + pair + direction within 2 h (TREND 4 h) of the cluster's first bar = ONE event, stable across runs.
OUTCOMES (from the first bar's close; BTC_MOVE / BREADTH_BURST = median over the universe's top 20; pair events = the pair):
  f30/f60/f120 in the event direction (timestamp lookups) · mfe/mae over 60, 120 and 240 min + t_peak_min / mae_before_peak(_x) (pair events: minutes to the 4 h best, the worst point
  before it incl…excl the peak bar's own wick) (best / worst excursion — what a
  trailing-stop sleeve earns / must survive) · rel60 = pair f60 minus BTC's (pair events) · tbs = +1×ATR before −1×ATR within
  120 min (pair events; same-bar = stop first) · stack = the pair's EMA5/8/13/20 order at detection (UP / DOWN / NONE — a missed
  TREND with an aligned stack means the bot's signal was likely live and something blocked it).
BOT CHECK — preferred source: ~/Downloads/scalpars_decisions_paper_*.csv ("Download Decisions CSV" = the bot's decision journal:
  every fill with its strategy, every refusal per 5 min × pair × gate, 5-min scan heartbeats, every position with open/close
  times). When its heartbeats cover the window, fills + positions held at the move's start come from it (MANUAL split apart) and a
  WHY column names the gates that refused that pair/direction while the move ran (top 3), or says "not in the bot's universe" /
  "maker entry expired" / "no gate fired (no setup)"; BOOK FULL = a BOOK_FULL / NO_BALANCE refusal or ≥ max_open bot positions.
  Fallback (windows no decisions export covers):
  (all ~/Downloads/scalpars_orders_*.csv merged; coverage = union of [first opened_at, file time] per export; recomputed for
  EVERY stored row each run so late exports fill old rows): fills of that pair (pair events) or of any pair (market events) in the
  event direction, opened in [start of the measured move, last cluster bar + 60 min] OR already open at its start; MANUAL listed
  apart; book = bot positions open at the event (≥ max_open_positions → "none (BOOK FULL)" — a capacity miss, not a signal miss).
  ⭐ MISSED? = bot none ∧ book not full ∧ in universe ∧ mfe60 ≥ 1.0 % (a move a trailing exit could have captured).
TOP MOVERS: biggest 4 h move per pair/direction in 24 h (universe), merged into episodes across runs (SCOUT_MOVERS.csv).
EVIDENCE (SCOUT_EVIDENCE.json, frozen): per bucket (type × direction × held/scope), f60 NET of a 0.15 % cost. A bucket QUALIFIES at
  ≥ 20 events on ≥ 8 days with a Bonferroni-corrected t-interval over DAY means entirely beyond 0 (continuation or fade); its
  direction is then FROZEN and it must CONFIRM on the next 8 fresh days (t-bound beyond 0 in the frozen direction) before it is
  labelled "✅ BACKTEST CANDIDATE". A failed confirmation stays failed (never re-fit on the data that failed it).
Outputs (main project reports/): SCOUT_EVENTS.csv · SCOUT_MOVERS.csv · SCOUT_EVIDENCE.json · SCOUT_REPORT_latest.md ·
SCOUT_REPORT_<date>.md · SCOUT_WEEKLY_<iso-week>.md (first run of each ISO week). Usage: venv/bin/python scripts/opportunity_scout.py
"""
import glob
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
EVENTS_CSV = os.path.join(REPORTS, "SCOUT_EVENTS.csv")
MOVERS_CSV = os.path.join(REPORTS, "SCOUT_MOVERS.csv")
EVIDENCE_JSON = os.path.join(REPORTS, "SCOUT_EVIDENCE.json")
LOCK = os.path.join(REPORTS, ".scout.lock")
QUIET = False                      # --quiet: print only the new note lines (the scheduled run's single fixed command)
BAR, MIN = 300_000, 60_000
FETCH = 640
EX = ccxt.binanceusdm({"enableRateLimit": True})

BTC_MOVE_MIN, BREADTH_SHARE, BREADTH_MOVE = 0.8, 0.40, 1.0
SPIKE_MOVE, SPIKE_VOL = 4.0, 5.0
TREND_MOVE, TREND_EFF, TREND_MAX_BAR = 3.0, 0.5, 0.35
DEDUPE = {"TREND": 4 * 3600_000}
DEDUPE_DEFAULT = 2 * 3600_000
COST = 0.15                        # % round-trip fees + slippage allowance
CAND_N, CAND_DAYS, CONFIRM_DAYS = 20, 8, 8
CONFIRM_EXPIRY_DAYS = 45           # a qualified bucket that gets no 8 fresh days in 45 days expires (never re-fit)
N_TESTS = 24                       # pre-registered: 12 declared buckets × 2 directions (Bonferroni, fixed — review)
SCAN_N = 80                        # scan the top-80 eligible pairs so events OUTSIDE the bot's top-50 are recorded as such
DECISIONS_MAX_AGE_D = 50        # decisions exports read (by file age): ≥ the 45-day evidence expiry, so no stored event loses its WHY
STAMP_MAX_AGE = 40 * 3600_000     # feature stamps only while every input still reaches the event (640 × 5m = 53 h of alt bars)
MERGE_FLAG_MS = 30 * MIN           # a TREND and an ALT_SPIKE on the same pair/side within 30 min = one ⭐ flag
DETECT_COLS = ["move_first", "vol_mult", "held_first", "scope", "eff", "in_universe", "rank", "start_ts", "qvol24_event"]


def log(msg):
    print(f"[scout] {msg}", file=sys.stderr)


def _retry(fn, *a, **kw):
    err = None
    for attempt in range(3):
        try:
            return fn(*a, **kw)
        except Exception as e:
            err = e; time.sleep(1 + attempt)
    log(f"{getattr(fn, '__name__', 'call')} failed after 3 tries: {err}")
    return None


def atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _betainc(a, b, x):
    """regularized incomplete beta I_x(a, b) (Numerical-Recipes continued fraction)."""
    import math
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    def cf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300); c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300); c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
            de = d * c; h *= de
            if abs(de - 1) < 1e-12: break
        return h
    bt = math.exp(lbt)
    return bt * cf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * cf(b, a, 1 - x) / b


def tcrit(df, conf):
    """EXACT two-sided Student-t critical value (review: the table/normal-ratio shortcut was anti-conservative at 7–14 df)."""
    target = 1 - (1 - conf) / 2
    cdf = lambda t: 1 - 0.5 * _betainc(df / 2, 0.5, df / (df + t * t))
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < target else (lo, mid)
    return (lo + hi) / 2


# ─────────────────────────────── data ───────────────────────────────
def k5(sym, last_closed, limit=FETCH):
    """5m bars from the raw klines endpoint (same data as fetch_ohlcv + the taker-buy base volume `tb`, free)."""
    raw = _retry(EX.fapiPublicGetKlines, {"symbol": sym.split("/")[0] + "USDT", "interval": "5m", "limit": int(limit)})
    if not raw:
        return None
    d = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])] for r in raw],
                     columns=["t", "o", "h", "l", "c", "v", "tb"]).drop_duplicates("t").set_index("t")
    return d[d.index <= last_closed]                   # every series cut at the SAME last closed bar (no forming-bar look-ahead)


def _frame(rows):
    return pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t").set_index("t") if rows else None


def backfill_exit_shape(allv, alts, now_ms):
    """Stored pair events that left the 26 h detection window before their 4 h exit-shape fields matured (or predate them) get
    them once from the fetched candles (640 × 5m ≈ 53 h) — outcomes are facts after the move, so a late fill is exact. Never
    raises; rows whose candles are gone stay empty (the report prints n)."""
    try:
        if not len(allv):
            return allv
        cols = ("mfe240", "mae240", "t_peak_min", "mae_before_peak", "mae_before_peak_x")
        for c in cols:
            if c not in allv:
                allv[c] = np.nan
        todo = allv.index[allv.type.isin(["ALT_SPIKE", "TREND"]) & (allv.mfe240.isna() | allv.mae_before_peak_x.isna())
                          & (allv.bar_ts.astype("int64") + BAR + 240 * MIN <= now_ms)]
        for i in todo:
            r = allv.loc[i]; d = alts.get(r["pair"]); t0 = int(r["bar_ts"])
            if d is None or t0 not in d.index:
                continue
            o = _series_outcome(d, t0, 1 if r["side"] == "UP" else -1)
            if o and "mfe240" in o:
                for c in cols:
                    allv.loc[i, c] = round(float(o[c]), 2)
        return allv
    except Exception as e:
        log(f"exit-shape backfill skipped ({e})"); return allv


def stamp_features(allv, btc_full, alts, in_now, last_closed):
    """🧬 Every event not yet stamped gets, ONCE, the bot's own entry_* columns + pre_* move features at its anchor bar
    (scripts/scout_features.py). Only events ≤ STAMP_MAX_AGE old (every input — 999-bar BTC monitor, 640-bar alts/breadth, ETH —
    still reaches them); a run whose REQUIRED market inputs failed stamps nothing (retried next hour); a pair event whose 1h/1d
    reads failed stays unstamped too. Never raises: any failure leaves the rows as they were."""
    try:
        if not len(allv):
            return allv
        import scout_features as SF
        allv = allv.drop(columns=[c for c in SF.LEGACY_COLS if c in allv.columns])
        fv = pd.to_numeric(allv["feat_v"], errors="coerce") if "feat_v" in allv else pd.Series(np.nan, index=allv.index)
        unst = fv.isna() | (fv < SF.FEAT_VERSION)                        # an older stamp version is re-stamped while still reachable
        fresh = allv.bar_ts.astype("int64") >= last_closed - STAMP_MAX_AGE
        pair_ok = allv.apply(lambda r: r["pair"] in ("BTCUSDT", "UNIVERSE")
                             or (alts.get(r["pair"]) is not None and int(r["bar_ts"]) in alts[r["pair"]].index), axis=1)
        need = allv.index[unst & fresh & pair_ok]
        if not len(need) or btc_full is None:
            return allv
        h1 = _frame(_retry(EX.fetch_ohlcv, "BTC/USDT:USDT", "1h", limit=1000))
        h4 = _frame(_retry(EX.fetch_ohlcv, "BTC/USDT:USDT", "4h", limit=1000))
        d1 = _frame(_retry(EX.fetch_ohlcv, "BTC/USDT:USDT", "1d", limit=10))
        eth = k5("ETH/USDT:USDT", last_closed, FETCH)
        if any(x is None for x in (h1, h4, d1, eth)):
            log("feature stamps skipped this run: a required BTC/ETH read failed (retried next run)"); return allv
        uni = {p: alts[p] for p in in_now if alts.get(p) is not None}   # the scan's breadth universe: top-limit minus pair_blacklist,
        uni.update(BTCUSDT=btc_full, ETHUSDT=eth)                        # BTC/ETH INCLUDED (no_trade_pairs still count in breadth)
        market = (btc_full, h1, h4, d1, eth, uni)
        firsts = allv.loc[need].groupby("pair").bar_ts.min().to_dict()     # stampable rows only (review: stale OI window)
        ext = {}

        def extras_for(pair):
            if pair not in ext:
                ext[pair] = SF.fetch_pair_extras(EX, _retry, pair, int(firsts.get(pair, last_closed)))
            return ext[pair]
        cache, out = {}, {}
        for i in need:
            ev = allv.loc[i].to_dict()
            if ev["pair"] not in ("BTCUSDT", "UNIVERSE"):
                x = extras_for(ev["pair"])
                if x.get("k1h") is None or x.get("k1d") is None:
                    continue                                              # retried next run, never stamped half-empty
            try:
                out[i] = SF.event_features(ev, alts, cache, market, extras_for)
            except Exception as e:
                log(f"feature stamp failed for {ev['type']} {ev['pair']} {ev['bar_ts']}: {e}")
        if out:
            F = pd.DataFrame.from_dict(out, orient="index")
            for c in F.columns:
                if c not in allv.columns:
                    allv[c] = np.nan
                allv[c] = allv[c].astype(object)
                allv.loc[F.index, c] = F[c].values
        log(f"feature stamps: {len(out)} of {int(unst.sum())} unstamped events stamped ({len(need)} stampable)")
        return allv
    except Exception as e:
        log(f"feature stamps failed this run ({e}) — rows left as they were"); return allv


def load_cfg():
    try:
        return json.load(open(os.path.join(ROOT, "trading_config.json")))
    except Exception:
        return {}


def bot_universe(cfg):
    """The bot's eligibility filters (binance_service.get_top_futures_pairs), top SCAN_N by 24 h volume, minus its blacklists.
    Returns (scan list, current rank map, the 24 h quote-volume of the bot's #limit pair = the membership cutoff)."""
    mk = _retry(EX.load_markets) or {}
    tk = _retry(EX.fetch_tickers) or {}
    limit = int(cfg.get("trading_pairs_limit", 50) or 50)
    nl_days = int(cfg.get("new_listing_filter_days", 0) or 0)
    alpha = bool(cfg.get("alpha_subtype_filter_enabled", False))
    coin = bool(cfg.get("coin_underlying_only", False))
    now = time.time() * 1000
    rows = []
    for s, t in tk.items():
        if not s.endswith("/USDT:USDT") or t.get("last") is None:
            continue
        info = (mk.get(s, {}) or {}).get("info", {}) or {}
        if coin and info.get("underlyingType") not in (None, "COIN"):
            continue
        ob = info.get("onboardDate")
        try:
            if nl_days > 0 and ob is not None and now - int(ob) < nl_days * 86400_000:
                continue
        except (TypeError, ValueError):
            pass
        sub = info.get("underlyingSubType") or []
        if alpha and any("alpha" in str(x).lower() for x in (sub if isinstance(sub, list) else [sub])):
            continue
        rows.append((s.split("/")[0] + "USDT", float(t.get("quoteVolume") or 0)))
    rows.sort(key=lambda x: -x[1])
    rank = {p: i + 1 for i, (p, _) in enumerate(rows)}
    cutoff = rows[limit - 1][1] if len(rows) >= limit else 0.0
    excl = {x.strip() for x in (cfg.get("pair_blacklist") or "").split(",") if x.strip()}
    excl |= {x.strip() for x in (cfg.get("no_trade_pairs") or "").split(",") if x.strip()} | {"BTCUSDT", "ETHUSDT"}
    return [p for p, _ in rows[:SCAN_N] if p not in excl], rank, cutoff, limit


def load_exports():
    frames, spans = [], []
    to_ms = lambda s: (pd.to_datetime(s, utc=True, format="ISO8601") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):     # the paper bot only (live exports excluded)
        try:
            d = pd.read_csv(f, low_memory=False,
                            usecols=lambda c: c in ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status"))
            if not len(d) or not {"opened_at", "pair", "direction"} <= set(d.columns):
                continue
            if "entry_strategy" not in d:
                d["entry_strategy"] = None
            if "status" in d:                                      # aborted entries (SIGNAL_EXPIRED) are not fills (review)
                d = d[d.status.astype(str).str.upper().isin(["CLOSED", "OPEN"])]
            if not len(d):
                continue
            exp_ms = int(os.path.getmtime(f) * 1000)
            d["opened_ms"] = to_ms(d["opened_at"])
            d["closed_ms"] = to_ms(d["closed_at"]).fillna(exp_ms) if "closed_at" in d else exp_ms   # open at export → open till then
            frames.append(d); spans.append([int(d.opened_ms.min()), exp_ms])
        except Exception as e:
            log(f"export skipped ({os.path.basename(f)}): {e}")
    if not frames:
        return None, []
    o = pd.concat(frames, ignore_index=True).sort_values("closed_ms").drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    # Exports carry CLOSED rows only (main.py) — a position still OPEN at export time is invisible, so the last stretch before each
    # export cannot prove "none" (review C1). Trim every span's end by the p99 holding time seen in the exports (≤ 20 h).
    hold = (o.closed_ms - o.opened_ms)
    blind = float(np.nanpercentile(hold, 99)) if hold.notna().sum() >= 10 else 4 * 3600_000
    blind = min(max(blind, 30 * MIN), 20 * 3600_000)
    spans = [[s, e - blind] for s, e in spans if e - blind > s]
    spans.sort(); cov = []
    for s, e in spans:                                               # union of coverage spans
        if cov and s <= cov[-1][1]:
            cov[-1][1] = max(cov[-1][1], e)
        else:
            cov.append([s, e])
    return o, cov


CAPACITY_GATES = {"BOOK_FULL", "NO_BALANCE"}
HOUSEKEEPING = ("BOOK_CHANGED", "REDEPLOY_OPEN", "OPEN_FAILED", "BACKSTOP_PLACE_FAILED", "OPEN_FILTERED")


FAILS_FROM = [None]   # ms from which the decision journal carries FULL gate sets (FAILS lines); set by load_decisions


def load_decisions():
    """Decisions exports merged → (fills, blocks, positions, expired, coverage). Coverage = the 5-min SCAN heartbeats (gaps
    > 10 min split it: a missing journal day / downtime is 'unknown', never 'none'); files without heartbeats (first export
    version) fall back to [first row, file time]. Newer exports win duplicates."""
    frames = []
    to_ms = lambda s: (pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")):
        if time.time() - os.path.getmtime(f) > DECISIONS_MAX_AGE_D * 86400:   # daily auto-downloads pile up; older ones add nothing
            continue                                                      # the stored events still need (events live ≤ 45 days)
        try:
            d = pd.read_csv(f, low_memory=False)
            if not len(d) or not {"t", "e", "pair"} <= set(d.columns):
                continue
            d = d[d.e != "BOOK"]                                              # 📖 Oct-3: order-book research rows — not decisions
            d["ms"] = to_ms(d["t"]); d = d[d.ms.notna()].copy()
            d["closed_ms"] = to_ms(d["closed"]) if "closed" in d else np.nan
            d["exp_ms"] = int(os.path.getmtime(f) * 1000)
            frames.append(d)
        except Exception as e:
            log(f"decisions export skipped ({os.path.basename(f)}): {e}")
    empty = pd.DataFrame()
    if not frames:
        return empty, empty, empty, empty, [], empty
    a = pd.concat(frames, ignore_index=True).sort_values("exp_ms")      # newest export last → keep="last" wins
    fills = a[a.e == "OPEN"].drop_duplicates(["t", "pair", "dir"], keep="last")
    blocks = a[a.e == "BLOCK"].drop_duplicates(["t", "pair", "dir", "gate"], keep="last")
    blocks = blocks[~blocks.gate.astype(str).str.startswith(HOUSEKEEPING)]
    pos = a[a.e.isin(["POSITION", "POSITION_OPEN"])].drop_duplicates(["t", "pair", "dir"], keep="last")
    expired = a[a.e == "EXPIRED"]
    fails = a[a.e == "FAILS"].drop_duplicates(["t", "pair", "dir", "gate", "src"], keep="last") if "src" in a else a.iloc[0:0]
    FAILS_FROM[0] = int(fails.ms.min()) + 10 * MIN if len(fails) else None   # first FAILS line ≈ the deploy that started writing them
    beats = sorted(set(a[a.e == "SCAN"].ms.astype("int64")))
    cov = []
    if beats:
        for t in beats:
            if cov and t - cov[-1][1] <= 10 * MIN:
                cov[-1][1] = t + BAR
            else:
                cov.append([t, t + BAR])
    else:                                                            # first export version: no heartbeats
        for exp, g in a.groupby("exp_ms"):
            j = g[~g.e.isin(["POSITION", "POSITION_OPEN"])]
            if len(j):
                cov.append([int(j.ms.min()), int(exp)])
        cov.sort()
    return fills, blocks, pos, expired, cov, fails


def _held_at(pos, t):
    """positions (from the decisions export) open at time t: opened before t and closed after it (or still open at export)."""
    if pos is None or not len(pos):
        return pos
    closed = pos.closed_ms
    still_open = closed.isna() & (t <= pos.exp_ms)
    return pos[(pos.ms < t) & ((closed > t) | still_open)]


# ─────────────────────────────── detection ───────────────────────────────
def ema_stack(c):
    e = [c.ewm(span=n, adjust=False).mean().iloc[-1] for n in (5, 8, 13, 20)]
    return "UP" if e[0] > e[1] > e[2] > e[3] else ("DOWN" if e[0] < e[1] < e[2] < e[3] else "NONE")


def detect(btc, alts, now_ms, cfg):
    th = cfg.get("thresholds") or {}
    s_move = abs(float(th.get("surge_btc_move_pct", 1.0) if th.get("surge_btc_move_pct") is not None else 1.0))
    s_vol = float(th.get("surge_btc_vol_mult", 3.0) or 0.0)
    s_on = {"UP": bool(th.get("surge_long_enabled", True)), "DOWN": bool(th.get("surge_short_enabled", True))}
    s_high = bool(th.get("surge_long_require_24h_high", True)); s_low = bool(th.get("surge_short_require_24h_low", False))
    raw = []; since = now_ms - 26 * 3600_000
    r30 = (btc.c / btc.c.shift(6) - 1) * 100
    qv = btc.v * btc.c
    med = qv.shift(1).rolling(288, min_periods=288).median()
    hi24 = btc.h.shift(1).rolling(288, min_periods=288).max(); lo24 = btc.l.shift(1).rolling(288, min_periods=288).min()
    for t in btc.index[(btc.index >= since) & (r30.abs() >= BTC_MOVE_MIN)]:
        side = "UP" if r30[t] > 0 else "DOWN"
        vm = qv[t] / med[t] if pd.notna(med[t]) and med[t] > 0 else None
        held = (s_on[side] and s_move > 0 and abs(r30[t]) >= s_move and (s_vol <= 0 or (vm is not None and vm >= s_vol))
                and (not (side == "UP" and s_high) or (pd.notna(hi24[t]) and btc.c[t] >= hi24[t]))
                and (not (side == "DOWN" and s_low) or (pd.notna(lo24[t]) and btc.c[t] <= lo24[t])))
        raw.append(dict(type="BTC_MOVE", pair="BTCUSDT", side=side, bar_ts=int(t), move=round(r30[t], 2),
                        vol_mult=round(vm, 1) if vm else None, held=bool(held), start_ts=int(t) - 5 * BAR))
    r15 = {p: (d.c / d.c.shift(3) - 1) * 100 for p, d in alts.items() if d is not None and len(d) > 30}
    breadth_bars = {"UP": set(), "DOWN": set()}
    if r15:
        R = pd.DataFrame(r15); n = R.notna().sum(axis=1)
        up = (R >= BREADTH_MOVE).sum(axis=1) / n.where(n > 0); dn = (R <= -BREADTH_MOVE).sum(axis=1) / n.where(n > 0)
        for t in R.index[R.index >= since - 3 * BAR]:
            for side, sh in (("UP", up[t]), ("DOWN", dn[t])):
                if pd.notna(sh) and sh >= BREADTH_SHARE and n[t] >= 10:
                    breadth_bars[side].add(int(t))
                    if t >= since:
                        raw.append(dict(type="BREADTH_BURST", pair="UNIVERSE", side=side, bar_ts=int(t), move=round(sh * 100, 0),
                                        vol_mult=None, held=False, start_ts=int(t) - 2 * BAR))
    trend_hits = []
    for p, d in alts.items():
        if d is None or len(d) < 300:
            continue
        m15 = (d.c / d.c.shift(3) - 1) * 100
        q = d.v * d.c; qm = q.shift(1).rolling(288, min_periods=288).median()
        q24 = q.shift(1).rolling(288, min_periods=288).sum()          # 24 h quote volume BEFORE the event bar (membership then)
        for t in d.index[(d.index >= since) & (m15.abs() >= SPIKE_MOVE) & (q >= SPIKE_VOL * qm)]:
            raw.append(dict(type="ALT_SPIKE", pair=p, side="UP" if m15[t] > 0 else "DOWN", bar_ts=int(t), move=round(m15[t], 2),
                            vol_mult=round(q[t] / qm[t], 1), held=False, start_ts=int(t) - 2 * BAR, qvol24=float(q24.get(t, np.nan))))
        diffs = d.c.diff().abs()
        net = d.c - d.c.shift(24); path = diffs.rolling(24).sum(); maxbar = diffs.rolling(24).max()
        r2h = (d.c / d.c.shift(24) - 1) * 100
        eff = net.abs() / path.where(path > 0)
        ok = (d.index >= since) & (r2h.abs() >= TREND_MOVE) & (eff >= TREND_EFF) & (maxbar <= TREND_MAX_BAR * path)
        for t in d.index[ok]:
            trend_hits.append(dict(type="TREND", pair=p, side="UP" if r2h[t] > 0 else "DOWN", bar_ts=int(t), move=round(r2h[t], 2),
                                   vol_mult=None, held=False, eff=round(float(eff[t]), 2), start_ts=int(t) - 23 * BAR,
                                   qvol24=float(q24.get(t, np.nan))))
    per_bar = {}
    for h in trend_hits:
        per_bar[(h["bar_ts"], h["side"])] = per_bar.get((h["bar_ts"], h["side"]), 0) + 1
    for h in trend_hits:
        near_breadth = any(abs(h["bar_ts"] - b) <= 3 * BAR for b in breadth_bars[h["side"]])
        h["scope"] = "MARKET" if per_bar[(h["bar_ts"], h["side"])] >= 3 or near_breadth else "PAIR"
    return raw + trend_hits


def cluster(raw, known):
    """Start-anchored clusters per (type, pair, side); a bar inside a KNOWN event's window joins it (stable across runs)."""
    raw.sort(key=lambda e: e["bar_ts"])
    out = {}
    for e in raw:
        key = (e["type"], e["pair"], e["side"]); win = DEDUPE.get(e["type"], DEDUPE_DEFAULT)
        anchor = next((k for k in known.get(key, []) if 0 <= e["bar_ts"] - k < win), None)
        if anchor is None:
            anchor = next((a for (kk, a) in out if kk == key and 0 <= e["bar_ts"] - a < win), None)
        if anchor is None:
            anchor = e["bar_ts"]
        is_first = anchor == e["bar_ts"]
        ev = out.setdefault((key, anchor), dict(type=e["type"], pair=e["pair"], side=e["side"], bar_ts=anchor, start_ts=e["start_ts"],
                                                move_first=None, move_max=e["move"], vol_mult=None, held_first=None, held_any=False,
                                                scope=e.get("scope"), eff=None, end_ts=e["bar_ts"], n_bars=0))
        if is_first:                                                  # detection facts come from the FIRST bar only (no look-ahead)
            ev.update(move_first=e["move"], vol_mult=e["vol_mult"], held_first=bool(e["held"]), eff=e.get("eff"), start_ts=e["start_ts"],
                      qvol24_event=e.get("qvol24"))
            if e.get("scope"):
                ev["scope"] = e["scope"]
        ev["end_ts"] = max(ev["end_ts"], e["bar_ts"]); ev["n_bars"] += 1
        ev["held_any"] = ev["held_any"] or bool(e["held"])
        if e["type"] != "BREADTH_BURST" and abs(e["move"]) > abs(ev["move_max"] or 0):
            ev["move_max"] = e["move"]                              # descriptive only — never used for buckets / outcomes
    return list(out.values())


# ─────────────────────────────── outcomes ───────────────────────────────
def atr_pct_at(d, t):
    w = d.loc[:t].tail(40)
    if len(w) < 20:
        return None
    tr = pd.concat([w.h - w.l, (w.h - w.c.shift()).abs(), (w.l - w.c.shift()).abs()], axis=1).max(axis=1).iloc[1:]
    a = tr.iloc[:14].mean()
    for x in tr.iloc[14:]:
        a = (a * 13 + x) / 14
    return a / float(w.c.iloc[-1]) * 100


def _series_outcome(d, t0, sgn, atr_pct=None):
    if d is None or t0 not in d.index:
        return None
    ref = float(d.c.loc[t0]); out = {}
    for h in (30, 60, 120):
        t = t0 + h * MIN
        if t in d.index:
            out[f"f{h}"] = sgn * (float(d.c.loc[t]) / ref - 1) * 100
    for h in (60, 120, 240):
        w = d.loc[t0 + BAR: t0 + h * MIN]
        if len(w) >= (h // 5 if h == 240 else h // 5 - 1):   # the 4 h read needs its full 48 bars (a partial pass is never stored)
            fav = (w.h.max() / ref - 1) * 100 if sgn > 0 else (1 - w.l.min() / ref) * 100
            adv = (w.l.min() / ref - 1) * 100 if sgn > 0 else (1 - w.h.max() / ref) * 100
            out[f"mfe{h}"], out[f"mae{h}"] = fav, adv
            if h == 240:                                   # 📐 exit design: WHEN the best point came and how deep it dipped FIRST
                best = w.h if sgn > 0 else -w.l                                     # favourable extreme per bar
                pk = best.idxmax()                                                  # first bar reaching the 4 h best
                out["t_peak_min"] = (pk - t0) // MIN                                # minutes from the entry close to that bar's close
                pre = w.loc[:pk]   # bars up to and INCLUDING the peak bar: its own extreme may come after the high → CONSERVATIVE (deeper)
                # (positive when price never went against the move)
                out["mae_before_peak"] = ((pre.l.min() / ref - 1) * 100 if sgn > 0 else (1 - pre.h.max() / ref) * 100)
                ex = w.loc[:pk].iloc[:-1]            # EXCLUDING the peak bar (its wick may come after the high) → the shallow end
                out["mae_before_peak_x"] = (0.0 if not len(ex) else
                                            min(0.0, (ex.l.min() / ref - 1) * 100) if sgn > 0 else min(0.0, (1 - ex.h.max() / ref) * 100))
    if atr_pct:
        w = d.loc[t0 + BAR: t0 + 120 * MIN]
        if len(w) >= 23:
            for name, k_t in (("tbs", 1.0), ("tbs2", 2.0)):          # target k×ATR before a 1×ATR stop, within 120 min
                tgt, stp = ref * (1 + sgn * k_t * atr_pct / 100), ref * (1 - sgn * atr_pct / 100)
                res = "NEITHER"
                for r in w.itertuples():
                    hit_t = (r.h >= tgt) if sgn > 0 else (r.l <= tgt)
                    hit_s = (r.l <= stp) if sgn > 0 else (r.h >= stp)
                    if hit_s:
                        res = "STOP"; break                         # same bar as the target: the stop is assumed first
                    if hit_t:
                        res = "TARGET"; break
                out[name] = res
    return out


def outcomes(e, btc, alts, top20):
    res = {}
    t0 = e["bar_ts"]; sgn = 1 if e["side"] == "UP" else -1
    if e["type"] in ("ALT_SPIKE", "TREND"):
        d = alts.get(e["pair"])
        if d is not None and t0 in d.index:
            res["stack"] = ema_stack(d.c.loc[:t0])
            o = _series_outcome(d, t0, sgn, atr_pct_at(d, t0))
            b = _series_outcome(btc, t0, sgn)
            if o:
                for k in ("f30", "f60", "f120", "mfe60", "mae60", "mfe120", "mae120", "mfe240", "mae240", "t_peak_min", "mae_before_peak",
                          "mae_before_peak_x", "tbs", "tbs2"):
                    if k in o:
                        res[k] = round(o[k], 2) if isinstance(o[k], float) else o[k]
                if "f60" in o and b and "f60" in b:
                    res["rel60"] = round(o["f60"] - b["f60"], 2)
    else:
        outs = [o for o in (_series_outcome(alts.get(p), t0, sgn) for p in top20) if o]
        for k in ("f30", "f60", "f120", "mfe60", "mae60", "mfe120", "mae120"):
            # market events: no 4 h shape fields — independent medians across 20 pairs would not describe one path (review)
            v = [o[k] for o in outs if k in o]
            if v:
                res[k] = round(float(np.median(v)), 2)
        v = [o["f60"] for o in outs if "f60" in o]
        if v:
            res["f60_best"] = round(float(np.max(v)), 2)
    return res


def covered(cov, a, b):
    return any(s <= a and b <= e for s, e in cov)


def bot_check(row, orders, cov, max_open, dec=None):
    """(bot, manual, book, why) for one stored row — recomputed every run so late exports fill old rows."""
    st_, en_ = row.get("start_ts"), row.get("end_ts")
    if st_ is None or en_ is None or pd.isna(st_) or pd.isna(en_):
        return "unknown", "", None, "", "", "", ""
    a = int(st_); b = int(en_) + BAR + 60 * MIN                        # start_ts = the close the measured move starts from
    want = "LONG" if row["side"] == "UP" else "SHORT"
    pair_ev = row["pair"] not in ("BTCUSDT", "UNIVERSE")
    if dec is not None and len(dec) > 4 and dec[4] and covered(dec[4], a, b):          # ── the decision journal covers this window (heartbeats)
        fills, blocks, pos, expired, _, fsets = dec
        dirmask = lambda df: df.dir.astype(str).str.upper().isin([want, "ANY"])
        g = fills[(fills.ms >= a) & (fills.ms <= b) & (fills.dir.astype(str).str.upper() == want)] if len(fills) else fills
        held = _held_at(pos, a)
        if len(held):
            held = held[held.dir.astype(str).str.upper() == want]
        if pair_ev:
            g = g[g.pair == row["pair"]] if len(g) else g
            held = held[held.pair == row["pair"]] if len(held) else held
        strat = ([str(x) for x in g.strategy.fillna("MOMENTUM")] if len(g) else []) + \
                ([str(x) for x in held.strategy.fillna("MOMENTUM")] if len(held) else [])
        man = [x for x in strat if x == "MANUAL"]; botn = [x for x in strat if x != "MANUAL"]
        bot = ", ".join(f"{k}×{v}" for k, v in pd.Series(botn).value_counts().items()) if botn else "none"
        t_ev = int(row["bar_ts"]) + BAR
        book_now = _held_at(pos, t_ev)
        book = int((book_now.strategy.astype(str) != "MANUAL").sum()) if len(book_now) else 0
        why = ""; miss = "TRADED" if bot != "none" else ""; sets_txt = ""; best = ""; size = {}
        if bot == "none":
            end_close = int(en_) + BAR                                  # refusals while the move ran, not after it
            fails_era = FAILS_FROM[0] is not None and a >= FAILS_FROM[0]   # the journal wrote FULL gate sets for this window
            fs = fsets[(fsets.ms >= a - BAR) & (fsets.ms <= end_close) & dirmask(fsets)] if len(fsets) else fsets
            if len(fs):
                fs = fs[fs.pair == row["pair"]] if pair_ev else fs
            if len(fs):
                fs = fs[~fs.gate.astype(str).str.contains("_DISABLED")]  # a switched-off sleeve is not a gate to loosen
            if len(fs):                                                  # FULL gate sets (overlap-aware)
                lab = fs.gate.astype(str).where(fs.src.astype(str) == "MOMENTUM", fs.src.astype(str) + ":" + fs.gate.astype(str))
                sn = fs.assign(n=pd.to_numeric(fs.n, errors="coerce").fillna(1), lab=lab).groupby("lab").n.sum().sort_values(ascending=False)
                sets_txt = "; ".join(f"{k}×{int(v)}" for k, v in sn.head(3).items())
                size = {k: str(k).split(":", 2)[-1].count("+") + 1 if str(k).startswith("FLIP:") else str(k).count("+") + 1 for k in sn.index}
                best = min(sn.index, key=lambda k: (size[k], -sn[k]))   # the CLOSEST the move came to a trade (fewest gates)
            bl = blocks[(blocks.ms >= a - BAR) & (blocks.ms <= end_close) & dirmask(blocks)] if len(blocks) else blocks
            if len(bl):
                bl = bl[bl.pair == row["pair"]] if pair_ev else bl
            n_all = pd.to_numeric(bl.n, errors="coerce").fillna(1) if len(bl) else pd.Series(dtype=float)
            cap = bl.gate.astype(str).isin(CAPACITY_GATES) if len(bl) else pd.Series(dtype=bool)
            cap_fired = bool(len(bl) and n_all[cap].sum() > 0)
            blocked = bool(len(bl) and n_all[~cap].sum() > 0)
            refused = bool(sets_txt) or blocked
            expired_hit = bool(len(expired) and len(expired[(expired.ms >= a) & (expired.ms <= b) & ((expired.pair == row["pair"]) | (not pair_ev))]))
            if cap_fired or (book >= max_open and not refused):           # a gate refusal outranks "the book was full at the start"
                bot = "none (BOOK FULL)"
            if len(bl):
                top = bl.assign(n=n_all).groupby("gate").n.sum().sort_values(ascending=False).head(3)
                why = ", ".join(f"{k}×{int(v)}" for k, v in top.items())
            elif str(row.get("in_universe")) == "False":
                why = "not in the bot's universe"
            elif expired_hit:
                why = "maker entry expired"
            else:
                why = "no gate fired (no setup)"
            if bot == "none (BOOK FULL)" and not why.startswith("book"):
                why = f"book full ({book} open) · " + why if book >= max_open else "capacity · " + why
            if sets_txt:
                why = f"gate sets: {sets_txt} · closest: {best}" + (f" · first gates: {why}" if why and not why.startswith(("no gate", "not in")) else "")
            if not sets_txt and blocked and fails_era:
                best = "(ladder passed; refused by: " + str(bl.assign(n=n_all)[~cap].groupby("gate").n.sum().idxmax()) + ")"
            nb = 0 if not best else (1 if best.startswith("(ladder") else size.get(best, 9))
            miss = ("CAPACITY" if bot == "none (BOOK FULL)" else
                    "EXECUTION" if expired_hit else                     # the bot wanted it; the maker entry expired
                    "UNIVERSE" if str(row.get("in_universe")) == "False" else
                    "PRE_FAILS" if not fails_era else                   # no full gate sets for this window: FILTER vs SLEEVE unknowable
                    "FILTER_NEAR" if refused and 0 < nb <= 2 else       # 1–2 gates from a trade → loosen candidate
                    "FILTER_FAR" if refused else                        # ≥3 gates at best → the rules were far from it (sleeve-like)
                    "SLEEVE")                                           # in universe, never even a refused candidate
        return bot, (f"manual×{len(man)}" if man else ""), book, why, miss, sets_txt, best
    if orders is None or not covered(cov, a, b):
        return "unknown", "", None, "", "", "", ""
    want = "LONG" if row["side"] == "UP" else "SHORT"
    g = orders[((orders.opened_ms >= a) & (orders.opened_ms <= b)) | ((orders.opened_ms < a) & (orders.closed_ms > a))]
    if row["pair"] not in ("BTCUSDT", "UNIVERSE"):
        g = g[g.pair == row["pair"]]
    g = g[g.direction.astype(str).str.upper() == want]
    man = g[g.entry_strategy.astype(str) == "MANUAL"]; bot = g[g.entry_strategy.astype(str) != "MANUAL"]
    t_ev = int(row["bar_ts"]) + BAR
    book = int(((orders.opened_ms <= t_ev) & (orders.closed_ms > t_ev) & (orders.entry_strategy.astype(str) != "MANUAL")).sum())
    names = ", ".join(f"{s}×{n}" for s, n in bot.entry_strategy.fillna("MOMENTUM").astype(str).value_counts().items()) or "none"
    if names == "none" and book >= max_open:
        names = "none (BOOK FULL)"
    return names, (f"manual×{len(man)}" if len(man) else ""), book, "", ("TRADED" if not names.startswith("none") else
                                                                          "CAPACITY" if "BOOK" in names else ""), "", ""


def missed_flag(r):
    """⭐ only when a simple trade WOULD have worked (review: MFE ≥ 1 % was true for 60 % of all moves): pair events — +2×ATR reached
    before −1×ATR within 120 min; market events — the top-20 median f60 beat the cost and ran ≥ 1 % at some point."""
    if str(r.get("bot")) != "none" or str(r.get("in_universe")) == "False":
        return False
    if r["type"] in ("ALT_SPIKE", "TREND"):
        return str(r.get("tbs2")) == "TARGET"
    return pd.notna(r.get("f60")) and float(r["f60"]) > COST and pd.notna(r.get("mfe60")) and float(r["mfe60"]) >= 1.0


def dedupe_flags(df):
    """One ⭐ per move: a TREND and an ALT_SPIKE on the same pair/side within 30 min keep only the earlier flag."""
    if not len(df) or "missed" not in df:
        return df
    f = df[df.missed & df.type.isin(["ALT_SPIKE", "TREND"])].sort_values("bar_ts")
    drop, last = [], {}
    for i, r in f.iterrows():
        k = (r.pair, r.side)
        if k in last and r.bar_ts - last[k] <= MERGE_FLAG_MS:
            drop.append(i)
        else:
            last[k] = r.bar_ts
    df.loc[drop, "missed"] = False
    return df


# ─────────────────────────────── evidence ───────────────────────────────
def bucket_of(r):
    if r["type"] == "BTC_MOVE":
        return f"BTC_MOVE {r['side']} · {'trigger held' if str(r.get('held_first')) == 'True' else 'below trigger'}"
    if r["type"] == "TREND":
        return f"TREND {r['side']} · {r.get('scope') if isinstance(r.get('scope'), str) else 'PAIR'}"
    return f"{r['type']} {r['side']}"


def day_stats(g, conf):
    v = g["net60"].astype(float).values
    days = pd.to_datetime(g.bar_ts, unit="ms").dt.date.values
    dm = np.array([v[days == d].mean() for d in pd.unique(days)])
    n = len(dm); m = float(dm.mean()) if n else float("nan")
    if n < 3:
        return n, m, float("nan"), float("nan")
    se = dm.std(ddof=1) / np.sqrt(n); tc = tcrit(n - 1, conf)
    return n, m, m - tc * se, m + tc * se


def evidence(allv, now_ms):
    state = json.load(open(EVIDENCE_JSON)) if os.path.exists(EVIDENCE_JSON) else {}
    if allv is None or not len(allv) or "f60" not in allv:
        return [], state
    a = allv[allv.f60.notna() & (allv.in_universe.astype(str) != "False")].copy()
    if not len(a):
        return [], state
    a["bucket"] = a.apply(bucket_of, axis=1)
    a["net60"] = a.f60.astype(float)                                 # RAW f60; the cost is a hurdle on the interval (review C2)
    conf = 1 - 0.05 / N_TESTS                                        # Bonferroni over the pre-registered test count
    out = []
    for b, g in a.groupby("bucket"):
        days, m, lo, hi = day_stats(g, conf)
        st = state.get(b)
        verdict = f"collecting ({len(g)}/{CAND_N} events · {days}/{CAND_DAYS} days)"
        if st:
            fresh = g[g.bar_ts > st["frozen_at_ms"]]
            fdays, fm, flo, fhi = day_stats(fresh, 0.95) if len(fresh) else (0, float("nan"), float("nan"), float("nan"))
            cont = st["direction"].startswith("continuation")
            if st.get("status") in ("CONFIRMED", "FAILED", "EXPIRED"):
                pass
            elif fdays >= CONFIRM_DAYS:
                # confirmation = the fresh days reproduce it: day-mean beyond the cost AND the 95 % bound beyond 0 (one look)
                good = (cont and fm > COST and flo > 0) or ((not cont) and fm < -COST and fhi < 0)
                st["status"] = "CONFIRMED" if good else "FAILED"; st["decided_at_ms"] = now_ms
            elif now_ms - st["frozen_at_ms"] > CONFIRM_EXPIRY_DAYS * 86400_000:
                st["status"] = "EXPIRED"; st["decided_at_ms"] = now_ms
            verdict = (("✅ BACKTEST CANDIDATE — " + st["direction"]) if st.get("status") == "CONFIRMED" else
                       ("✗ failed confirmation (" + st["direction"] + ")") if st.get("status") == "FAILED" else
                       ("✗ expired unconfirmed (" + st["direction"] + ")") if st.get("status") == "EXPIRED" else
                       f"⏳ qualified ({st['direction']}) — confirming on fresh data {fdays}/{CONFIRM_DAYS} days")
        elif len(g) >= CAND_N and days >= CAND_DAYS and not np.isnan(lo) and (lo > COST or hi < -COST):
            direction = "continuation" if lo > COST else "FADE (reverses)"
            state[b] = dict(direction=direction, frozen_at_ms=int(g.bar_ts.max()), qualified_mean=round(m, 3), status="CONFIRMING")
            verdict = f"⏳ qualified ({direction}) — confirming on fresh data 0/{CONFIRM_DAYS} days"
        elif len(g) >= CAND_N and days >= CAND_DAYS:
            verdict = "no edge (range spans 0 after costs)"
        srt = g.net60.sort_values().values
        k = int(len(srt) * 0.05)
        trim = float(srt[k: len(srt) - k].mean()) if len(srt) - 2 * k > 0 else float(srt.mean())
        out.append(dict(bucket=b, n=len(g), days=days, cont=float((g.f60 > 0).mean() * 100), mean=m, median=float(np.median(srt)),
                        trim=trim, lo=lo, hi=hi, mfe60=float(g.mfe60.mean()) if "mfe60" in g and g.mfe60.notna().any() else None,
                        verdict=verdict))
    return sorted(out, key=lambda r: (r["verdict"][0] not in "✅⏳", -r["n"])), state


MISS_MEANING = {"FILTER_NEAR": "refused, but 1–2 gates from a trade (closest set) → loosen candidate",
                "FILTER_FAR": "refused, ≥3 gates even at its closest → the rules were far from it → sleeve-like",
                "SLEEVE": "in the universe, never even a refused candidate → the rules never saw a setup → new-sleeve territory",
                "CAPACITY": "book was full → capacity, not signal",
                "UNIVERSE": "outside the bot's top-50 universe → universe question, not a filter",
                "EXECUTION": "the bot wanted it, the maker entry expired → execution"}


def _exit_shape(g):
    """Exit-design shape of a WHOLE cohort (moves ≥ 4 h old only): median best point over 4 h, the dip BEFORE that peak (median and
    worst quartile) and minutes to the peak — where a stop must sit to survive and how long a trail must hold. All moves, never
    the Good ones alone: Good = +2×ATR before −1×ATR, so their dip is shallow BY CONSTRUCTION. '–' until moves are 4 h old."""
    cols = ("mfe240", "mae_before_peak", "mae_before_peak_x", "t_peak_min")
    if not len(g) or any(c not in g for c in cols):
        return "–"
    x = g[list(cols)].apply(pd.to_numeric, errors="coerce").dropna()
    if not len(x):
        return "–"
    cap = (x.t_peak_min >= 235).mean() * 100
    return (f"{x.mfe240.median():+.1f}% · {x.mae_before_peak_x.median():+.1f}…{x.mae_before_peak.median():+.1f}% / "
            f"{x.mae_before_peak_x.quantile(0.25):+.1f}…{x.mae_before_peak.quantile(0.25):+.1f}% · "
            f"{x.t_peak_min.median():.0f} min{' (floor: ' + format(cap, '.0f') + '% still running at 4 h)' if cap >= 10 else ''} (n={len(x)})")


def _diag_frame(allv):
    """Classified untraded pair moves with a known outcome, ALT_SPIKE + TREND on the same pair/side within 4 h = ONE move."""
    if allv is None or not len(allv) or "miss_class" not in allv or "tbs2" not in allv:
        return None
    d = allv[allv.type.isin(["ALT_SPIKE", "TREND"]) & allv.miss_class.isin(list(MISS_MEANING))
             & allv.tbs2.isin(["TARGET", "STOP", "NEITHER"])].copy()
    if not len(d):
        return d
    d = d.sort_values(["bar_ts", "type"], kind="stable"); keep, last = [], {}
    for i, r in zip(d.index, d.itertuples()):
        k = (r.pair, r.side)
        if k in last and int(r.bar_ts) - last[k] < 4 * 3600_000:
            continue
        last[k] = int(r.bar_ts); keep.append(i)
    d = d.loc[keep].copy()
    d["day"] = pd.to_datetime(d.bar_ts, unit="ms").dt.date
    return d


def pair_day_first(g, extra=()):
    """One row per (pair, side, UTC day[, extra keys]) = that unit's FIRST event (the first chance a bot entering once would
    have had). The same move is often stored several times — as an ALT_SPIKE and a TREND on one bar, or as repeated signals
    while a pair runs — and those must not count as independent evidence."""
    if g is None or not len(g):
        return g
    return g.sort_values(["bar_ts", "type"]).drop_duplicates(["pair", "side", "day", *extra], keep="first")


def diagnosis_lines(allv):
    """Untraded pair events (ALT_SPIKE / TREND) whose 2×ATR-vs-1×ATR outcome is known, split by WHY the bot missed them:
    FILTER (gates refused it — table per FULL gate set: good moves blocked vs bad moves blocked, overlap-aware), SLEEVE (no gate
    ever refused it / outside the universe — the current rules never had a setup), CAPACITY (book full). Only windows a decisions
    export covers are classified. A gate set is worth testing when it blocks clearly more TARGETs than STOPs over many days."""
    if allv is None or not len(allv) or "miss_class" not in allv or "tbs2" not in allv:
        return ["Needs a 'Download Decisions CSV' covering the events (and events ≥ 2 h old)."]
    d = _diag_frame(allv)
    if d is None or not len(d):
        return ["No classified untraded moves yet (needs a decisions export covering them)."]
    L = ["| Class | Moves | Good (2×ATR first) | Bad (1×ATR stop first) | Neither | Pair-days: n · good · bad | Days | All moves (n = with 4 h data): median best 4 h · worst point before the peak, excl…incl the peak bar's wick (median / worst quartile) · min to peak | Meaning |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in MISS_MEANING:
        g = d[d.miss_class == c]
        if len(g):
            u = pair_day_first(g)
            L.append(f"| {c} | {len(g)} | {(g.tbs2 == 'TARGET').sum()} | {(g.tbs2 == 'STOP').sum()} | {(g.tbs2 == 'NEITHER').sum()} | "
                     f"{len(u)} · {(u.tbs2 == 'TARGET').sum()} · {(u.tbs2 == 'STOP').sum()} | {g.day.nunique()} | "
                     f"{_exit_shape(g)} | {MISS_MEANING[c]} |")
    f = d[d.miss_class == "FILTER_NEAR"]
    if len(f):
        x = f.assign(set=(f["closest_set"] if "closest_set" in f else pd.Series("", index=f.index)).fillna("").astype(str))             # the CLOSEST full set the move came to (fewest gates)
        L += ["", "FILTER_NEAR moves by their CLOSEST full gate set. Necessary, not sufficient: loosening the set is required to free the move, "
              "but last-mile ladder gates and later engine gates were not evaluated on those scans — the engine replay decides. "
              "Pair-days = one per pair / side / UTC day (its first event; a pair-day can appear under two sets) — the count to judge by "
              "WITHIN a day; evidence across time = the Days column (market-wide gates move every pair together):", "",
              "| Gate set | Moves | Good | Bad | Pair-days | Good | Bad | Days | Good − Bad (pair-days) |", "|---|---|---|---|---|---|---|---|---|"]
        for st_, g in sorted(x.groupby("set"), key=lambda kv: -len(kv[1]))[:12]:
            good, bad = int((g.tbs2 == "TARGET").sum()), int((g.tbs2 == "STOP").sum())
            u = pair_day_first(g); ug, ub = int((u.tbs2 == "TARGET").sum()), int((u.tbs2 == "STOP").sum())
            L.append(f"| {st_} | {len(g)} | {good} | {bad} | {len(u)} | {ug} | {ub} | {g.day.nunique()} | {ug - ub:+d} |")
    pre = int((allv.miss_class == "PRE_FAILS").sum()) if "miss_class" in allv else 0
    L += ["", "Read (use the PAIR-DAY counts and the Days column — a 2-to-1 target needs more than 33 % good before fees to be worth freeing): FILTER_NEAR sets that clearly beat Bad over many days → engine-replay test of loosening exactly that set; "
          "good moves piling up in FILTER_FAR / SLEEVE → new-sleeve backtest (ALT_SPIKE / TREND). Never a live change from these counts alone."
          + (f" {pre} older events predate the full-gate-set journal (PRE_FAILS) and are left out." if pre else "")]
    return L


NOTES_MD = os.path.join(REPORTS, "SCOUT_NOTES.md")
NOTES_STATE = os.path.join(REPORTS, ".scout_notes_state.json")


def _append_notes(lines, now_ms):
    if not lines:
        return
    hdr = f"\n## {datetime.fromtimestamp(now_ms / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC\n"
    with open(NOTES_MD, "a", encoding="utf-8") as f:
        f.write(hdr + "".join(f"- {x}\n" for x in lines))


def _load_notes_state():
    try:
        with open(NOTES_STATE) as f:
            return json.load(f)
    except Exception:
        return {}


def note_failure(msg):
    """A run that cannot complete leaves ONE line in the notes (the scheduled task no longer reads or edits files itself) —
    the same message at most once per 6 h (an outage must not flood the notes)."""
    try:
        now = int(time.time() * 1000); st = _load_notes_state()
        if st.get("fail_msg") == msg and now - int(st.get("fail_ms", 0)) < 6 * 3600_000:
            return
        _append_notes([f"scout run failed: {msg}"], now)
        st.update(fail_msg=msg, fail_ms=now); atomic_write(NOTES_STATE, json.dumps(st))
    except Exception:
        pass


NOTE_MAX_LINES = 12
NOTE_ITEM_MAX_AGE = 7 * 86400_000      # (c)/(d) only consider items this young; keys live 10 days > this → never re-noted


def write_notes(allv, mv, ev, now_ms):
    """🗒 The notes the scheduled run used to write by hand (DECISION_LOG 156), now deterministic and appended by the script itself,
    so a run is ONE fixed command (no ad-hoc reads/edits → no permission prompts). Only NEW things (state in
    reports/.scout_notes_state.json), in priority order: (a) evidence verdict changes, (f) no Decisions CSV newer than 30 h (once per
    UTC day), (b) BTC moves / breadth bursts since the previous run, (c) ⭐ misses on BTC_MOVE / BREADTH_BURST / TREND with their gate
    sets (+ a gate set refusing ≥ 2 of those ⭐ moves the same day), (d) untraded 4 h movers ≥ ±10 % (one line per move episode),
    (e) diagnosis crossings (FILTER_NEAR set Good−Bad ≥ +3 over ≥ 3 days; FILTER_FAR+SLEEVE Good ≥ Bad + 5 — once each).
    The first run marks the backlog as seen and writes only the last ~4 h. Keys are recorded only for lines actually written
    (the 12-line cap never swallows an item). Returns the lines written."""
    st = _load_notes_state()
    noted = {k: v for k, v in (st.get("noted") or {}).items()
             if k.startswith("D|") or now_ms - v < NOTE_ITEM_MAX_AGE + 3 * 86400_000}   # diagnosis crossings: once, ever
    first = "last_run_ms" not in st
    since = now_ms - 250 * MIN                       # first run only: what counts as "recent" (later runs dedupe by keys)
    young = now_ms - NOTE_ITEM_MAX_AGE
    say = lambda ms: (not first) or ms > since          # the very first run marks the backlog as seen, writes only the last ~4 h
    items = []                                           # (priority, line, key or None)
    silent = []                                          # keys marked seen without a line (first-run backlog)

    def add(prio, line, key, when=None):
        if key and key in noted:
            return
        if when is not None and not say(when):
            if key:
                silent.append(key)
            return
        items.append((prio, line, key))
    t_of = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%m-%d %H:%M")
    s_ = lambda v: v if isinstance(v, str) and v and v != "nan" else ""
    # (a) evidence verdicts
    prev = st.get("verdicts") or {}; cur = {}
    for r in ev or []:
        v = r["verdict"]; cat = ("candidate" if v.startswith("✅") else "qualified" if v.startswith("⏳") else
                                 "failed" if v.startswith("✗ failed") else "expired" if v.startswith("✗ expired") else "other")
        cur[r["bucket"]] = cat
        if cat != "other" and prev.get(r["bucket"]) != cat and not first:
            items.append((0, f"EVIDENCE — {r['bucket']}: {v} ({r['n']} events · {r['days']} days · mean {fmt(r['mean'])})", None))
    # (f) stale Decisions CSV
    files = glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv"))
    newest = max((os.path.getmtime(f) for f in files), default=0) * 1000
    day = datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
    if now_ms - newest > 30 * 3600_000:
        add(1, "no Decisions CSV newer than 30 h in Downloads — the dashboard's 'Auto daily' download seems off (tab closed or "
               "downloads blocked)", f"F|{day}")
    if allv is not None and len(allv):
        a = allv
        # (b) market moves since the previous run
        for r in a[a.type.isin(["BTC_MOVE", "BREADTH_BURST"]) & (a.bar_ts.astype("int64") >= now_ms - 26 * 3600_000)].sort_values("bar_ts").itertuples():
            mvv = r.move_first if pd.notna(getattr(r, "move_first", np.nan)) else getattr(r, "move_max", np.nan)
            what = (f"{float(mvv):.0f}% of pairs" if pd.notna(mvv) else "?") if r.type == "BREADTH_BURST" else fmt(mvv)
            hf = str(getattr(r, "held_first", ""))
            held = ((" · SURGE trigger held" if hf == "True" else " · below SURGE trigger" if hf == "False" else " · trigger unknown")
                    if r.type == "BTC_MOVE" else "")
            add(2, f"{r.type} {r.side} {what} at {t_of(int(r.bar_ts) + BAR)}{held} · bot {s_(getattr(r, 'bot', '')) or 'unknown'}"
                   + (f" · why: {s_(getattr(r, 'why', ''))}" if s_(getattr(r, 'why', '')) else ""),
                f"M|{r.type}|{r.side}|{int(r.bar_ts)}", int(r.bar_ts) + BAR)    # keys dedupe; a late detection is still noted
        # (c) ⭐ misses
        if "missed" in a:
            star = a[(a.missed.astype(str) == "True") & a.type.isin(["BTC_MOVE", "BREADTH_BURST", "TREND"])
                     & (a.bar_ts.astype("int64") >= young)]
            for r in star.sort_values("bar_ts").itertuples():
                gs = s_(getattr(r, "gate_sets", "")); why = s_(getattr(r, "why", ""))
                add(3, f"⭐ missed {r.type} {r.pair} {r.side} at {t_of(int(r.bar_ts) + BAR)} · {s_(getattr(r, 'miss_class', '')) or 'class ?'}"
                       + (f" · gate sets: {gs}" if gs else (f" · why: {why}" if why else "")),
                    f"S|{r.type}|{r.pair}|{r.side}|{int(r.bar_ts)}", int(r.bar_ts) + BAR)
            if "closest_set" in star and not first:            # ⭐ flags often land a day late (the daily Decisions CSV)
                rec = star[star.bar_ts.astype("int64") >= now_ms - 3 * 86400_000].copy()
                rec["d"] = pd.to_datetime(rec.bar_ts, unit="ms").dt.strftime("%Y-%m-%d")
                rec = rec[rec.closest_set.notna() & (rec.closest_set.astype(str).str.len() > 0)]
                for (dd, gset), n in rec.groupby(["d", rec.closest_set.astype(str)]).size().items():
                    if n >= 2 and gset != "nan":
                        add(4, f"gate set {gset} refused {n} ⭐ moves on {dd}", f"G|{dd}|{gset}")
    # (d) big untraded movers — ONE line per move episode (merge_movers slides start_ts as the move grows)
    if mv is not None and len(mv):
        big = mv[(mv.move_4h.astype(float).abs() >= 10) & mv.bot.astype(str).str.startswith("none")
                 & (mv.end_ts.astype("int64") >= young)]
        seen_v = [k.split("|") for k in noted if k.startswith("V|")]
        for r in big.itertuples():
            s0 = int(r.start_ts)
            if any(p == r.pair and sd == r.side and abs(int(t) - s0) <= 8 * 3600_000 for _, p, sd, t in seen_v):
                continue                                  # the same episode, already noted (its window slid)
            cap = "capacity miss (book full)" if "BOOK" in str(r.bot) else "signal miss"
            add(5, f"untraded mover {r.pair} {r.side} {float(r.move_4h):+.1f}% ({r.start_utc[5:]}→{r.end_utc[11:]}) · {cap}"
                   + (f" · why: {s_(getattr(r, 'why', ''))}" if s_(getattr(r, 'why', '')) else ""),
                f"V|{r.pair}|{r.side}|{s0}", int(r.end_ts) + BAR)
    # (e) diagnosis crossings (once each)
    d = _diag_frame(allv)
    if d is not None and len(d):
        near = d[(d.miss_class == "FILTER_NEAR") & d.closest_set.notna() & (d.closest_set.astype(str).str.len() > 0)] \
            if "closest_set" in d else d.iloc[0:0]
        for gset, g in near.groupby(near.closest_set.astype(str)):
            u = pair_day_first(g)                                   # judged on pair-days, like the report table
            good, bad = int((u.tbs2 == "TARGET").sum()), int((u.tbs2 == "STOP").sum())
            if good - bad >= 3 and g.day.nunique() >= 3:
                add(6, f"DIAGNOSIS — FILTER_NEAR set {gset}: {good} good vs {bad} bad pair-days over {g.day.nunique()} days → engine-replay "
                       f"loosen test candidate", f"D|NEAR|{gset}", None if not first else 0)
        fs = pair_day_first(d[d.miss_class.isin(["FILTER_FAR", "SLEEVE"])])
        good, bad = int((fs.tbs2 == "TARGET").sum()), int((fs.tbs2 == "STOP").sum())
        if good >= bad + 5:
            add(6, f"DIAGNOSIS — FILTER_FAR/SLEEVE: {good} good vs {bad} bad pair-days → new-sleeve backtest territory", "D|SLEEVE",
                None if not first else 0)
    items.sort(key=lambda x: x[0])
    shown = items[:NOTE_MAX_LINES]
    lines = [x[1] for x in shown]
    if len(items) > NOTE_MAX_LINES:
        lines.append(f"… {len(items) - NOTE_MAX_LINES} more next run (see SCOUT_REPORT_latest.md)")
    _append_notes(lines, now_ms)
    for _, _, k in shown:                                  # only what was WRITTEN is marked seen (the rest comes next run)
        if k:
            noted[k] = now_ms
    for k in silent:
        noted[k] = now_ms
    st.update(last_run_ms=now_ms, noted=noted, verdicts={**prev, **cur})   # an empty evidence run never forgets verdicts
    atomic_write(NOTES_STATE, json.dumps(st))
    return lines


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return "—"


def evidence_lines(ev):
    if not ev:
        return ["No in-universe events with a 60-min read recorded yet."]
    L = ["| Bucket | Events | Days | Continued | f60 (day mean) | Median | 5% trimmed | Corrected range | Avg MFE60 | Verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ev:
        rng = "— (needs ≥3 days)" if np.isnan(r["lo"]) else f"{r['lo']:+.2f} … {r['hi']:+.2f}"
        L.append(f"| {r['bucket']} | {r['n']} | {r['days']} | {r['cont']:.0f}% | {r['mean']:+.2f}% | {r['median']:+.2f}% | {r['trim']:+.2f}% | "
                 f"{rng} | {fmt(r['mfe60'])} | {r['verdict']} |")
    L += ["", f"A bucket QUALIFIES at ≥{CAND_N} events on ≥{CAND_DAYS} days when its exact t-interval over day means (Bonferroni, "
          f"{N_TESTS} pre-registered tests) clears the {COST}% cost: entirely above +{COST}% (continuation) or below −{COST}% (fade). "
          f"Its direction is then FROZEN and it must CONFIRM on {CONFIRM_DAYS} fresh days (mean beyond the cost, 95 % bound beyond 0) within {CONFIRM_EXPIRY_DAYS} days "
          "before it becomes a backtest candidate. Failed / expired buckets are never re-fit."]
    return L


# ─────────────────────────────── movers ───────────────────────────────
def top_movers(alts, now_ms):
    rows = []; since = now_ms - 24 * 3600_000
    for p, d in alts.items():
        if d is None or len(d) < 100:
            continue
        r4 = ((d.c / d.c.shift(48) - 1) * 100)
        r4 = r4[r4.index >= since].dropna()
        if not len(r4):
            continue
        for side, t in (("UP", r4.idxmax()), ("DOWN", r4.idxmin())):
            v = float(r4[t])
            if (side == "UP" and v <= 0) or (side == "DOWN" and v >= 0):
                continue
            rows.append(dict(pair=p, side=side, move_4h=round(v, 2), start_ts=int(t) - 47 * BAR, end_ts=int(t), q24=None))
    return rows


def merge_movers(old, new):
    """Episodes: a new window overlapping an existing (pair, side) episode joins it and keeps the larger |move|."""
    eps = old.to_dict("records") if old is not None and len(old) else []
    for r in new:
        hit = next((e for e in eps if e["pair"] == r["pair"] and e["side"] == r["side"]
                    and r["start_ts"] <= int(e["end_ts"]) + BAR and int(e["start_ts"]) <= r["end_ts"]), None)
        if hit is None:
            eps.append(dict(r))
        elif abs(r["move_4h"]) > abs(float(hit["move_4h"])):
            hit.update(move_4h=r["move_4h"], start_ts=r["start_ts"], end_ts=r["end_ts"])
    return pd.DataFrame(eps)


MOVER_STAMP_MIN = 5.0     # % — movers at least this big get the feature stamps (each stamped pair costs 4 extra reads)


def mover_anchor(start_ts):
    """The mover's feature anchor = the last closed bar BEFORE its 4 h window (what the pair looked like before the move)."""
    return int(start_ts) - BAR


def mover_q24(d, anchor):
    """24 h quote volume over the 288 closed bars ending at the anchor (None when the history does not reach)."""
    if d is None:
        return None
    w = d[d.index <= anchor].tail(288)
    return float((w.c * w.v).sum()) if len(w) == 288 and int(w.index[-1]) == int(anchor) else None


def stamp_movers(mv, alts, rank, btc_full, in_now, last_closed):
    """🧬 SCOUT_MOVERS gets its 24 h volume (recomputed at the CURRENT anchor whenever the history reaches, so a slid episode never
    keeps the old one) and, for moves ≥ MOVER_STAMP_MIN %, the same entry_* / pre_* stamps as the events, anchored one bar before
    the window. An episode whose window slid (merge_movers) loses every old stamp and is stamped again at the new anchor.
    entry_pair_rank = TODAY's volume rank (as for events — not the rank at the anchor). Works on a copy: a failure returns the
    input untouched. Never raises."""
    try:
        if not len(mv):
            return mv
        out = mv.reset_index(drop=True).copy()
        anchor = out.start_ts.astype("int64").map(mover_anchor)
        q = pd.to_numeric(out.q24, errors="coerce") if "q24" in out.columns else pd.Series(np.nan, index=out.index)
        for i in out.index:
            v = mover_q24(alts.get(out.at[i, "pair"]), int(anchor[i]))
            if v is not None:
                q[i] = v
        out["q24"] = q
        if "feat_bar_ts" in out.columns:
            fb = pd.to_numeric(out.feat_bar_ts, errors="coerce")
            moved = fb.notna() & (fb != anchor)
            if moved.any():
                out.loc[moved, [c for c in out.columns if c.startswith(("feat_", "entry_", "pre_"))]] = np.nan
        big = pd.to_numeric(out.move_4h, errors="coerce").abs() >= MOVER_STAMP_MIN
        if not big.any():
            return out
        base = set(out.columns)
        view = out[big].assign(type="MOVER", bar_ts=anchor[big], start_ts=anchor[big], qvol24_event=q[big], rank=out.pair[big].map(rank))
        view = stamp_features(view, btc_full, alts, in_now, last_closed)
        new = {}
        for c in view.columns:
            if c in ("type", "bar_ts", "qvol24_event", "rank", "start_ts") or (c in base and not c.startswith(("feat_", "entry_", "pre_"))):
                continue
            col = (out[c] if c in out.columns else pd.Series(np.nan, index=out.index)).astype(object)
            col.loc[view.index] = view[c].values
            new[c] = col
        out = pd.concat([out.drop(columns=[c for c in new if c in out.columns]), pd.DataFrame(new, index=out.index)], axis=1)
        return out
    except Exception as e:
        log(f"mover stamps failed this run ({e}) — rows left as they were"); return mv


# ─────────────────────────────── main ───────────────────────────────
def main():
    os.makedirs(REPORTS, exist_ok=True)
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) >= 900:
        try:
            os.remove(LOCK)                                           # stale (a crashed run)
        except OSError:
            pass
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log("another run holds the lock — exiting"); return
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        run()
    finally:
        try:
            if open(LOCK).read().strip() == str(os.getpid()):
                os.remove(LOCK)
        except OSError:
            pass


def run():
    now_ms = int(time.time() * 1000); last_closed = (now_ms // BAR - 1) * BAR
    cfg = load_cfg(); max_open = int((cfg.get("investment") or {}).get("max_open_positions", 4) or 4)
    btc_full = k5("BTC/USDT:USDT", last_closed, 1500)                 # 1500 bars: the 72 h monitor readings need 999 before a stamp
    btc = btc_full.tail(FETCH) if btc_full is not None else None      # detection + outcomes: the last 640 CLOSED bars
    if btc is None or len(btc) < 400:
        log("BTC fetch failed or too short — no run"); note_failure("BTC 5m fetch failed or too short (network?)"); return
    scan, rank, cutoff, limit = bot_universe(cfg)
    alts = {}
    for p in scan:
        alts[p] = k5(p[:-4] + "/USDT:USDT", last_closed); time.sleep(0.05)
    got = sum(1 for d in alts.values() if d is not None)
    in_now = [p for p in scan if rank.get(p, 999) <= limit]            # the bot's CURRENT scan list (top-limit minus blacklists)
    log(f"scan {len(scan)} eligible pairs · bot universe now {len(in_now)} · fetched {got} · #{limit} cutoff ${cutoff/1e6:.0f}M")
    top20 = in_now[:20]
    orders, cov = load_exports()
    dec = load_decisions()
    old = pd.read_csv(EVENTS_CSV) if os.path.exists(EVENTS_CSV) else pd.DataFrame()
    if len(old) and "start_ts" not in old.columns:                     # v1-format file: set aside, start fresh (review I3)
        os.replace(EVENTS_CSV, EVENTS_CSV + ".v1_bak"); old = pd.DataFrame()
        log("SCOUT_EVENTS.csv was v1 format — moved to .v1_bak")
    known = {}
    for r in (old.to_dict("records") if len(old) else []):
        known.setdefault((r["type"], r["pair"], r["side"]), []).append(int(r["bar_ts"]))
    evs = cluster(detect(btc, alts, now_ms, cfg), known)
    for e in evs:
        if e["pair"] in ("BTCUSDT", "UNIVERSE"):
            e["in_universe"] = True
        else:                                                         # membership AT THE EVENT: its 24 h volume then vs today's #limit
            qv = e.get("qvol24_event")
            e["in_universe"] = bool(qv is not None and not pd.isna(qv) and qv >= cutoff) if cutoff else e["pair"] in in_now
        e["rank"] = rank.get(e["pair"])
        if now_ms - (e["bar_ts"] + BAR) >= 2 * 3600_000:
            e.update(outcomes(e, btc, alts, top20))
        e["time_utc"] = datetime.fromtimestamp((e["bar_ts"] + BAR) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
    new = pd.DataFrame(evs)
    key = ["type", "pair", "side", "bar_ts"]
    if len(new) and len(old):
        n2, o2 = new.set_index(key), old.set_index(key)
        both = n2.index.intersection(o2.index)
        allv = n2.combine_first(o2)                                    # outcomes: this run wins unless missing
        for c in DETECT_COLS:                                          # detection facts: first-seen wins
            if c in o2.columns:
                keep = o2.loc[both, c]
                allv.loc[both, c] = keep.where(keep.notna(), allv.loc[both, c])
        for c in ("n_bars", "end_ts"):                                 # a cluster only grows (early bars leave the 26 h window)
            if c in o2.columns and c in allv.columns:
                allv.loc[both, c] = np.fmax(allv.loc[both, c].astype(float), o2.loc[both, c].astype(float))
        if "held_any" in o2.columns:                                  # OR — a held bar leaving the window must not flip it back
            allv.loc[both, "held_any"] = (allv.loc[both, "held_any"].astype(str) == "True") | (o2.loc[both, "held_any"].astype(str) == "True")
        if "move_max" in o2.columns:
            nm, om = allv.loc[both, "move_max"].astype(float), o2.loc[both, "move_max"].astype(float)
            allv.loc[both, "move_max"] = np.where(om.abs() > nm.abs(), om, nm)
        allv = allv.reset_index()
    else:
        allv = new if len(new) else old
    allv = backfill_exit_shape(allv, alts, now_ms)
    allv = stamp_features(allv, btc_full, alts, in_now, last_closed)
    allv = allv.drop(columns=[c for c in ("sole_gate",) if c in allv.columns])   # renamed closest_set (DECISION_LOG 151)
    if len(allv):
        chk = [bot_check(r, orders, cov, max_open, dec) for r in allv.to_dict("records")]
        allv["bot"] = [c[0] for c in chk]; allv["manual"] = [c[1] for c in chk]; allv["book"] = [c[2] for c in chk]; allv["why"] = [c[3] for c in chk]
        allv["miss_class"] = [c[4] for c in chk]; allv["gate_sets"] = [c[5] for c in chk]; allv["closest_set"] = [c[6] for c in chk]
        allv["missed"] = [missed_flag(r) for r in allv.to_dict("records")]
        allv = dedupe_flags(allv)
        atomic_write(EVENTS_CSV, allv.sort_values("bar_ts").to_csv(index=False))
    mold = pd.read_csv(MOVERS_CSV) if os.path.exists(MOVERS_CSV) else None
    if mold is not None and len(mold) and "q24" not in mold.columns:   # v2.0 movers used the bar-open start — shift once
        mold["start_ts"] = mold["start_ts"].astype("int64") + BAR; mold["q24"] = None
    mv = merge_movers(mold, top_movers({p: d for p, d in alts.items() if p in in_now}, now_ms))
    if len(mv):
        chk = [bot_check(dict(r, bar_ts=r["start_ts"]), orders, cov, max_open, dec) for r in mv.to_dict("records")]
        mv["bot"] = [c[0] for c in chk]; mv["manual"] = [c[1] for c in chk]; mv["why"] = [c[3] for c in chk]; mv["miss_class"] = [c[4] for c in chk]
        mv["start_utc"] = pd.to_datetime(mv.start_ts.astype("int64"), unit="ms").dt.strftime("%Y-%m-%d %H:%M")
        mv["end_utc"] = pd.to_datetime(mv.end_ts.astype("int64") + BAR, unit="ms").dt.strftime("%Y-%m-%d %H:%M")
        mv["in_universe"] = True                                      # movers are scanned inside the bot's universe only
        mv = stamp_movers(mv, alts, rank, btc_full, in_now, last_closed)
        atomic_write(MOVERS_CSV, mv.sort_values("start_ts").to_csv(index=False))
    ev, state = evidence(allv, now_ms)
    atomic_write(EVIDENCE_JSON, json.dumps(state, indent=1))
    rep = allv[allv.bar_ts >= now_ms - 24 * 3600_000].copy() if len(allv) else allv
    fmt_t = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%m-%d %H:%M")
    covtxt = ("order exports cover " + " · ".join(f"{fmt_t(s)}→{fmt_t(e)}" for s, e in cov[-3:]) + " UTC") if cov else "no order export found in ~/Downloads"
    covtxt = (("decision journal covers " + " · ".join(f"{fmt_t(s)}→{fmt_t(e)}" for s, e in dec[4][-3:]) + " UTC (preferred) · ") if dec[4]
              else "no Decisions CSV in ~/Downloads (use 'Download Decisions CSV' for the WHY column) · ") + covtxt
    L = [f"# 🔭 Opportunity scout — {datetime.fromtimestamp(now_ms/1000, timezone.utc):%Y-%m-%d %H:%M} UTC (last 24 h)", "",
         "Read-only. Candidates to backtest, never trade signals. Times = the event's first bar close (UTC). Outcomes in the event "
         "direction from that close; MFE/MAE = best / worst excursion (what a trailing exit could capture / had to survive).", "",
         f"Universe = the bot's own rules: {len(in_now)} pairs now (top {limit} eligible minus blacklists); {len(scan)} eligible pairs scanned so moves "
         f"OUTSIDE it are recorded as such (membership judged by each pair's 24 h volume AT the event vs today's #{limit} cutoff ${cutoff/1e6:.0f}M). "
         f"{covtxt} — the last stretch before each export is excluded (exports hold closed trades only); outside coverage = 'unknown'.", "",
         "## Evidence so far (every recorded in-universe event)", ""] + evidence_lines(ev) + [""]
    _stair_notes = []                                  # 🪜 Oct-2 staircase watch (alert only; public data; never breaks the run)
    try:
        import scout_staircase as _stair
        _st_noted = (_load_notes_state().get("noted") or {})
        _st_res = _stair.scan(EX, last_closed, _retry, extra=_stair.followed(_st_noted, now_ms))
        _st_sec = _stair.lines(_st_res) + [""]
        _stair_notes = _stair.note_items(_st_res, _st_noted, now_ms)
    except Exception as _st_e:
        log(f"staircase watch failed: {_st_e}")
        _st_sec = ["## 🪜 Staircase watch", "", "Unavailable this run.", ""]
    _fz_notes = []                                     # 🔥 Oct-3 FRENZY watch (every pair FRENZY can be watching; never breaks the run)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        import scout_frenzy as _fz
        _fzc = {**cfg, **(cfg.get("thresholds") or {})}
        _fz_set, _fz_flag, _fz_pairs, _fz_short, _fz_fol, _fz_crash = _fz.scan(EX, _retry, _fzc, last_closed, alts, btc_full, now_ms)
        _fz_hist = _fz.save(_fz_set, _fz_pairs, now_ms)
        _fz_sec = _fz.lines(_fz_set, _fz_flag, _fz_pairs, _fz_short, _fz_fol, _fzc, _fz_hist) + [""]
        try:                                           # 🔻 the crash-short observation never takes the FRENZY section down with it
            _fz_sec += _fz.crash_lines(_fz_crash, _fz.save_crashes(_fz_crash, _fz_pairs, now_ms))
        except Exception as _cr_e:
            log(f"crash-short observation failed: {_cr_e}")
            _fz_sec += ["## 🔻 Crash-short observation", "", f"Unavailable this run ({str(_cr_e)[:120]}).", ""]
        _fz_notes = _fz.note_items(_fz_set, (_load_notes_state().get("noted") or {}), now_ms)
    except Exception as _fz_e:
        log(f"FRENZY watch failed: {_fz_e}")
        _fz_sec = ["## 🔥 FRENZY watch", "", f"Unavailable this run ({str(_fz_e)[:120]}).", ""]
    try:                                               # ⚡ Oct-4 SURGE early-trigger observation (pre-registered; never breaks the run)
        if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import scout_surge_obs as _so
        _so_bars, _so_obs = _so.scan(EX, _retry, {**cfg, **(cfg.get("thresholds") or {})}, last_closed, alts, btc_full, in_now, now_ms)
        _fz_sec += _so.lines(_so_bars, _so.save(_so_obs))
    except Exception as _so_e:
        log(f"SURGE observation failed: {_so_e}")
        _fz_sec += ["## ⚡ SURGE early-trigger observation", "", f"Unavailable this run ({str(_so_e)[:120]}).", ""]
    try:                                               # 🩹 Oct-5 BTC rebound-window observation (pre-registered; never breaks the run)
        if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import scout_rebound as _rb
        _fz_sec += _rb.run(EX, _retry, last_closed)
    except Exception as _rb_e:
        log(f"rebound observation failed: {_rb_e}")
        _fz_sec += ["## 🩹 BTC rebound-window observation", "", f"Unavailable this run ({str(_rb_e)[:120]}).", ""]
    _rg_notes = []                                     # ⏳ Oct-4 revert-gate tracker (every open pre-committed revert / arm gate; never breaks the run)
    try:
        if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import scout_revert_gates as _rg
        _rg_sec, _rg_notes = _rg.run_section(now_ms, (_load_notes_state().get("noted") or {}))
    except Exception as _rg_e:
        log(f"revert-gate tracker failed: {_rg_e}")
        _rg_sec = ["## ⏳ Revert gates", "", f"Unavailable this run ({str(_rg_e)[:120]}).", ""]
    _ev_at = next((i for i, x in enumerate(L) if x.startswith("## Evidence so far")), len(L))
    L[_ev_at:_ev_at] = _rg_sec + _fz_sec + _st_sec                         # above the evidence table, so the events / "no events" line keep their place
    if not len(rep):
        L.append("No events in the last 24 h.")
    else:
        L += ["## Events (last 24 h)", "",
              "| Time UTC | Type | Pair | Dir | Move | Note | Stack | f60 | MFE60 / MAE60 | f120 | rel60 | 2×ATR/1×ATR | Bot | Why (gates) | Manual | ⭐ |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, e in rep.sort_values("bar_ts").iterrows():
            mvv = e.get("move_first") if pd.notna(e.get("move_first")) else e.get("move_max")
            mvtxt = f"{float(mvv):.0f}% of pairs" if e["type"] == "BREADTH_BURST" else fmt(mvv)
            note = []
            if e["type"] == "BTC_MOVE":
                note.append("SURGE trigger held" if str(e.get("held_first")) == "True" else
                            ("held later in cluster" if str(e.get("held_any")) == "True" else "below trigger"))
            if e["type"] == "TREND":
                note.append(f"{e.get('scope') if isinstance(e.get('scope'), str) else 'PAIR'} · eff {e.get('eff')}")
            if pd.notna(e.get("vol_mult")):
                note.append(f"vol {e['vol_mult']}×")
            if pd.notna(e.get("n_bars")) and int(e["n_bars"]) > 1:
                note.append(f"{int(e['n_bars'])} bars")
            if str(e.get("in_universe")) == "False":
                note.append("OUTSIDE bot universe")
            stack = e.get("stack") if isinstance(e.get("stack"), str) else ""
            L.append(f"| {e['time_utc'][5:]} | {e['type']} | {e['pair']} | {e['side']} | {mvtxt} | {' · '.join(note)} | {stack} | "
                     f"{fmt(e.get('f60'))} | {fmt(e.get('mfe60'))} / {fmt(e.get('mae60'))} | {fmt(e.get('f120'))} | {fmt(e.get('rel60'))} | "
                     f"{e.get('tbs2') if isinstance(e.get('tbs2'), str) else ''} | {e['bot']} | {e.get('why') if isinstance(e.get('why'), str) else ''} | {e.get('manual') if isinstance(e.get('manual'), str) else ''} | "
                     f"{'⭐ MISSED?' if e['missed'] else ''} |")
        m = rep[rep.missed]
        L += ["", f"**{len(rep)} events · {int(rep.missed.sum())} ⭐ MISSED?** (bot did nothing, book not full, pair in the bot's universe at the "
                  f"event, and a simple trade would have worked: +2×ATR before −1×ATR for pair events; top-20 median f60 > {COST}% with ≥1% run for "
                  "market events; one flag per move)."]
        if len(m):
            L.append("Flagged: " + "; ".join(f"{r.time_utc[11:]} {r.type} {r.pair} {r.side}" for r in m.itertuples()))
    L += ["", "## Missed-move diagnosis — new SLEEVE or loosen a FILTER? (every stored pair event the bot did not trade, with an outcome)", ""]
    L += diagnosis_lines(allv)
    if "feat_v" in allv:
        L += ["", f"🧬 Feature stamps: {int(allv.feat_v.notna().sum())} of {len(allv)} stored events carry the bot's own entry_* columns + "
              "pre_* move features at their entry bar (SCOUT_EVENTS.csv) — ready for scripts/sweep_separators.py-style missed-winner vs "
              "missed-loser screens once enough days accumulate."]
    L += ["", "## Top movers (largest 4 h moves in the bot's universe, last 24 h)", ""]
    m24 = mv[mv.end_ts.astype("int64") >= now_ms - 24 * 3600_000].copy() if len(mv) else mv
    if len(m24):
        m24["abs"] = m24.move_4h.astype(float).abs()
        L += ["| Pair | Dir | 4 h move | From → to (UTC) | Bot | Why (gates) | Manual |", "|---|---|---|---|---|---|---|"]
        for r in m24.sort_values("abs", ascending=False).head(10).itertuples():
            L.append(f"| {r.pair} | {r.side} | {float(r.move_4h):+.2f}% | {r.start_utc[5:]} → {r.end_utc[11:]} | {r.bot} | "
                     f"{r.why if isinstance(getattr(r, 'why', None), str) else ''} | {r.manual if isinstance(getattr(r, 'manual', None), str) else ''} |")
    else:
        L.append("No data.")
    txt = "\n".join(L) + "\n"
    atomic_write(os.path.join(REPORTS, "SCOUT_REPORT_latest.md"), txt)
    atomic_write(os.path.join(REPORTS, f"SCOUT_REPORT_{datetime.fromtimestamp(now_ms/1000, timezone.utc):%Y-%m-%d}.md"), txt)
    iy, iw, _ = datetime.fromtimestamp(now_ms / 1000, timezone.utc).isocalendar()
    marker = os.path.join(REPORTS, f".scout_weekly_{iy}-W{iw:02d}")
    wk = os.path.join(REPORTS, f"SCOUT_WEEKLY_7d_to_{datetime.fromtimestamp(now_ms/1000, timezone.utc):%Y-%m-%d}.md")
    if not os.path.exists(marker):                                    # first run of each ISO week → the trailing 7 days
        open(marker, "w").write(wk)
        W = [f"# 🔭 Scout weekly digest — the 7 days to {datetime.fromtimestamp(now_ms/1000, timezone.utc):%Y-%m-%d %H:%M} UTC", "",
             "## Pattern evidence (all recorded in-universe events)", ""] + evidence_lines(ev)
        W += ["", "## Biggest untraded 4 h moves of the last 7 days (only windows an order export covers)", ""]
        m7 = mv[(mv.end_ts.astype("int64") >= now_ms - 7 * 86400_000) & (mv.bot.astype(str).str.startswith("none"))].copy() if len(mv) else mv
        if len(m7):
            m7["abs"] = m7.move_4h.astype(float).abs()
            W += ["| Pair | Dir | 4 h move | From (UTC) | Note |", "|---|---|---|---|---|"] + \
                 [f"| {r.pair} | {r.side} | {float(r.move_4h):+.2f}% | {r.start_utc} | {'book full' if 'BOOK' in str(r.bot) else ''} |"
                  for r in m7.sort_values("abs", ascending=False).head(15).itertuples()]
        else:
            W.append("None confirmed (export orders more often — windows without an export are 'unknown').")
        atomic_write(wk, "\n".join(W) + "\n")
    try:
        notes = write_notes(allv, mv, ev, now_ms)
    except Exception as e:
        notes = [f"(notes failed: {e})"]; log(f"notes failed: {e}")
    try:                                               # 🪜 one note per pair per spike episode (keys ST| / ST32| in the notes state)
        _stair_notes = list(_stair_notes) + list(_fz_notes) + list(_rg_notes)   # 🔥 FRENZY mismatches + ⏳ fired revert gates ride the same once-per-key notes
        if _stair_notes:
            _st = _load_notes_state(); _noted = _st.get("noted") or {}
            _new = [(k, ln) for k, ln in _stair_notes if k not in _noted]
            if _new:
                _append_notes([ln for _, ln in _new], now_ms); notes = list(notes) + [ln for _, ln in _new]
                _noted.update({k: now_ms for k, _ in _new}); _st["noted"] = _noted; atomic_write(NOTES_STATE, json.dumps(_st))
    except Exception as e:
        log(f"staircase notes failed: {e}")
    if QUIET:
        print(f"scout OK {datetime.fromtimestamp(now_ms/1000, timezone.utc):%Y-%m-%d %H:%M} UTC · {len(allv)} stored events · "
              + (f"{len(notes)} new note line(s):\n" + "\n".join("- " + x for x in notes) if notes else "nothing notable (no note written)"))
    else:
        print(txt)


if __name__ == "__main__":
    QUIET = "--quiet" in sys.argv
    try:
        main()
    except Exception as _e:
        note_failure(f"{type(_e).__name__}: {_e}")
        raise
