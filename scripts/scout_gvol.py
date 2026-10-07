#!/usr/bin/env python3
"""🌊 Scout — the market-volume reading (FRENZY / WIDE gate frenzy_gvol_max, SURGE_LONG gate surge_long_gvol_min), engine parity, FROZEN per bar.

OBSERVE-ONLY scout code: never trades, never touches the engine or the config. Shared by scripts/scout_frenzy.py (FRENZY watch + crash rows) and
scripts/scout_surge_obs.py (SURGE comparison) — every scout place that prices the market's volume goes through ensure() below.

WHY (reports/FRENZY_GVOL_GATE_REVALIDATION_2026-10-06.md §5): the scout used to rank the top-50 at ITS RUN TIME and recompute every bar of the last
24 h with that list, the latest run overwriting each row. The ratio sums BASE volume, so one low-priced coin entering the list swings it (10-06:
VTHO replaced RENDER and took API3 06:35 from the live 1.11 to 0.83 → "pass"; 3 of 21 gate-live rows sat on the wrong side of 1.0).

DEFINITION (the engine's: services/trading_engine.py _market_gvol_read / _market_gvol_bars, services/frenzy.py global_volume_ratio; 18/18 parity
in the report, MAE 0.0015):
  universe   USDT perpetuals of the markets list (ANY status: a pair delisted since the bar still traded then — PUMPBTC / 1000000BOB were
             top-50 on 10-05), coin-only (underlyingType COIN / missing), not Alpha, listed ≥ new_listing_filter_days AS OF the signal close
             (onboardDate; missing = kept, fail-open like the engine). NO blacklists, BTC / ETH and non-ASCII names included. A pair counts at a
             bar only with that exact kline bar (the engine's ticker list = the pairs trading at the read).
  ranking    top-50 by 24 h QUOTE volume at the read (≈ the signal close + 4 s) = Σ kline quote volume (field 7) of the 288 5m bars ending at the
             signal bar, PER SIGNAL BAR (a coin entering the list later cannot touch an earlier bar). A pair without that exact bar is not ranked.
  ratio      services.frenzy.global_volume_ratio: Σ BASE volume of the closed signal bar ÷ Σ each pair's mean of the 48 bars up to and including
             it; a pair counts only with the exact bar and the full lookback; None below 30 pairs.
  candidates a 1h-kline prescreen of EVERY eligible pair keeps the top-80 by 24 h quote volume at each hour boundary around the needed bars (on
             the 10-03 → 10-06 tape the top-60 already held 100 % of every 5m bar's top-50; a pumped coin can fall from #26 to #545 within 6 h,
             so the CURRENT ticker ranking cannot be used as the candidate list).
NETWORK  public klines only, 6 threads, ≤ 1,000 weight/min of our own and a pause to the next minute whenever the IP's used weight (shared with
         every local process) passes 1,500; one computation per ≤ 24 h of bars, newest first, ALL-OR-NOTHING (a 429 / failed / late fetch freezes
         nothing — retried next run; the one-off backfill of older rows completes over successive runs within the 180 s budget).
FREEZE   a bar's value is written ONCE to SCOUT_GVOL_BARS.csv (ver = VER) the first time it is computed and never overwritten; rows of an older
         method version (or none) are recomputed once. An unreadable bar (None) is not frozen — retried next run.
ACCEPTED DIFFERENCES vs the engine (review 2026-10-06, all small; parity 18/18 same side, MAE 0.0015):
  · the prescreen's CEIL hour boundary looks up to 55 min past the signal close — it only WIDENS the candidate set, never the rank (the rank
    itself uses only the 288 bars ending at the signal bar);
  · the engine ranks TICKERS, so a top-50 pair missing the exact signal bar still takes a slot (and simply does not count); the scout ranks only
    pairs that have the bar, so it promotes the #51 instead — a rare gap in a top-50 pair;
  · Alpha / coin flags (and onboard dates) are TODAY's market metadata, not as of the bar;
  · v1 rows older than MAX_AGE_MS (7 d) are never recomputed: they stay "old method" and are left out of the gate split.
LIVE     the bot's own reading wins when known (gvol_src says which): ① the stamped entry_frenzy_gvol on a FRENZY fill opened ≤ 3 min after the
         signal close (4 decimals) · ② the server log's gate line "[FRENZY_LONG|FRENZY_WIDE] PAIR: setup ON but market volume X× normal ≥ …"
         logged ≤ 90 s after the close (2 decimals; the journal's BLOCK line carries no value). Both kept in SCOUT_GVOL_LIVE.csv (never lost when
         an export or a log leaves ~/Downloads). The value is market-wide, so any row on that bar uses it.
"""
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
BAR, HOUR, DAY = 300_000, 3_600_000, 86_400_000
VER = 2                                   # 1 = run-time top-50 (pre 2026-10-06) · 2 = per-bar top-50 by Σ quote volume (this module)
TOP_N, LOOKBACK, MIN_PAIRS, RANK_BARS = 50, 48, 30, 288
PRESCREEN_K = 80
MAX_AGE_MS = 7 * DAY                      # older bars are not (re)computed (kept as 'old method' if they have a stored value)
LIVE_FILL_MAX_MS = 3 * 60_000             # a fill opened later than this after the close may be a catch-up (its stamp is another bar's)
LIVE_LOG_MAX_MS = 90_000                  # a gate log line later than this after the close may be a catch-up refusal
CACHE_CSV = os.path.join(ROOT, "reports", "SCOUT_GVOL_BARS.csv")
LIVE_CSV = os.path.join(ROOT, "reports", "SCOUT_GVOL_LIVE.csv")
LOG_GLOBS = ("~/Downloads/*.log", "~/Downloads/*/var/log/messages*", "~/Downloads/*/var/log/web.stdout.log*")
CATCHUP_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]\d+ - services\.trading_engine - \w+ - \[FRENZY_CATCHUP\] (\S+): ON bar "
                        r"\d{2}:\d{2} was not judged")
CATCHUP_SHADOW_MS = 30 * 60_000           # a pair's gate lines within ±30 min of its catch-up line may be the catch-up's (another bar's value)
FREEZE_MIN_PAIRS = 45                     # freeze only a full read: ≥ 45 of the 50 counted (else retried next run)
MAX_TRIES = 3                             # a bar that stays unreadable / short after complete fetches is retried at most this many runs
LOG_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]\d+ - services\.trading_engine - \w+ - \[(FRENZY_LONG|FRENZY_WIDE)\] (\S+): "
                    r"setup ON but market volume ([0-9.]+)× normal")
SRC_STAMP, SRC_LOG, SRC_SCOUT, SRC_OLD = "live fill stamp", "live gate log", "scout v2", "scout v1 (old method)"


def log(msg):
    print(f"[scout gvol] {msg}", file=sys.stderr)


# ─────────────────────────── pure core (tested) ───────────────────────────
def eligible(meta, close_ms, cfg):
    """pairs of meta {pair: dict(onboard, alpha, coin)} the engine's get_top_futures_pairs keeps at close_ms (no blacklists)."""
    nl = int(cfg.get("new_listing_filter_days", 0) or 0)
    use_alpha = bool(cfg.get("alpha_subtype_filter_enabled", True))
    use_coin = bool(cfg.get("coin_underlying_only", True))
    out = []
    for p, m in meta.items():
        if use_coin and not m.get("coin", True):
            continue
        if use_alpha and m.get("alpha", False):
            continue
        ob = m.get("onboard")
        if nl > 0 and ob is not None and ob == ob and int(ob) >= close_ms - nl * DAY:
            continue
        out.append(p)
    return out


class Bars:
    """{pair: (t int64[], v float[], q float[])} sorted by open time, with prefix sums for the 288-bar quote-volume rank."""

    def __init__(self, rows_by_pair):
        self.d = {}
        for p, rows in (rows_by_pair or {}).items():
            if rows is None or not len(rows):
                continue
            a = np.array([[float(r[0]), float(r[5]), float(r[6])] for r in rows if len(r) >= 7], dtype=float)
            if not len(a):
                continue
            a = a[np.argsort(a[:, 0], kind="stable")]
            _, keep = np.unique(a[:, 0], return_index=True)
            a = a[keep]
            t = a[:, 0].astype(np.int64)
            self.d[p] = (t, a[:, 1], a[:, 2], np.concatenate([[0.0], np.cumsum(a[:, 2])]))

    def q24(self, p, sig):
        """Σ quote volume of the bars opening in (sig − 288 bars, sig]; None without the exact signal bar."""
        x = self.d.get(p)
        if x is None:
            return None
        t, _, _, cq = x
        j = int(np.searchsorted(t, sig))
        if j >= len(t) or t[j] != sig:
            return None
        i = int(np.searchsorted(t, sig - (RANK_BARS - 1) * BAR))
        return float(cq[j + 1] - cq[i])

    def rows(self, p, sig, n=LOOKBACK + 2):
        x = self.d.get(p)
        if x is None:
            return []
        t, v, _, _ = x
        j = int(np.searchsorted(t, sig, side="right"))
        return [[int(t[k]), 0, 0, 0, 0, float(v[k])] for k in range(max(0, j - n), j)]


def top_pairs(bars, pairs, sig, n=TOP_N):
    """the top-n of `pairs` by Σ quote volume of the 288 bars ending at the signal bar (a pair without that bar is not ranked)."""
    sc = [(q, p) for p in pairs for q in [bars.q24(p, sig)] if q is not None]
    sc.sort(key=lambda x: (-x[0], x[1]))
    return [p for _, p in sc[:n]]


def gvol_at(bars, meta, sig, cfg):
    """→ (ratio or None, pairs counted, top list) for the signal bar opening at `sig` (services.frenzy.global_volume_ratio, the engine's)."""
    from services.frenzy import global_volume_ratio
    top = top_pairs(bars, eligible(meta, sig + BAR, cfg), sig)
    rows = {p: bars.rows(p, sig) for p in top}
    n = sum(1 for r in rows.values() if r and r[-1][0] == sig and len(r) >= LOOKBACK)
    return global_volume_ratio(rows, int(sig), lookback=LOOKBACK, min_pairs=MIN_PAIRS), n, top


def prescreen(hourly, meta, sigs, cfg, k=PRESCREEN_K):
    """hourly {pair: [[open_ms, q], …]} → the pairs that rank top-k by 24 h quote volume (hour candles opening in [h − 24 h, h)) at the hour
    boundaries floor / ceil of each needed signal close, among the pairs eligible then."""
    hs = set()
    for s in sigs:
        c = int(s) + BAR
        hs.add(c // HOUR * HOUR); hs.add(-(-c // HOUR) * HOUR)
    H = {}
    for p, rows in (hourly or {}).items():
        if rows:
            a = np.array(rows, dtype=float)
            a = a[np.argsort(a[:, 0])]
            H[p] = (a[:, 0].astype(np.int64), np.concatenate([[0.0], np.cumsum(a[:, 1])]))
    keep = set()
    for h in sorted(hs):
        el = set(eligible(meta, h, cfg))
        sc = []
        for p, (t, cq) in H.items():
            if p not in el:
                continue
            i, j = int(np.searchsorted(t, h - DAY)), int(np.searchsorted(t, h))
            if j > i:
                sc.append((cq[j] - cq[i], p))
        sc.sort(key=lambda x: (-x[0], x[1]))
        keep |= {p for _, p in sc[:k]}
    return keep


def resolve(scout_v, live):
    """(value, source) — the bot's own reading when known (a fill stamp before a log line), else the scout's frozen v2 value."""
    if live:
        return live[0], live[1]
    if scout_v is not None and scout_v == scout_v:
        return float(scout_v), SRC_SCOUT
    return None, "unread"


def parity(pairs_, gmax=1.0):
    """[(scout, live), …] on the bars where both exist → dict(n, mae, max, same_side, flips=[i…]); sides judged at the gate threshold gmax."""
    xs = [(float(a), float(b)) for a, b in pairs_ if a is not None and b is not None and a == a and b == b]
    if not xs:
        return dict(n=0, mae=None, max=None, same_side=0, flips=[])
    err = [abs(a - b) for a, b in xs]
    flips = [i for i, (a, b) in enumerate(xs) if (a >= gmax) != (b >= gmax)]
    return dict(n=len(xs), mae=float(np.mean(err)), max=float(np.max(err)), same_side=len(xs) - len(flips), flips=flips)


def parse_log_lines(lines):
    """server log text lines → [(pair, close_ms, value, sleeve)] for the gate lines logged ≤ LIVE_LOG_MAX_MS after a 5m close. A catch-up refusal
    logs the same gate text for ANOTHER (older) ON bar, so every gate line of a pair within ±CATCHUP_SHADOW_MS of a "[FRENZY_CATCHUP] PAIR: ON
    bar HH:MM was not judged" line of the same pass is dropped (cannot be attributed safely)."""
    gates, cus = [], []
    for ln in lines:
        if "FRENZY_CATCHUP" in ln:
            m = CATCHUP_RE.search(ln)
            if m:
                cus.append((m.group(2), int(pd.Timestamp(m.group(1), tz="UTC").value // 1_000_000)))
            continue
        if "market volume" not in ln:
            continue
        m = LOG_RE.search(ln)
        if not m:
            continue
        t = int(pd.Timestamp(m.group(1), tz="UTC").value // 1_000_000)
        close = t // BAR * BAR
        if t - close > LIVE_LOG_MAX_MS:
            continue
        gates.append((m.group(3), close, float(m.group(4)), "WIDE" if m.group(2) == "FRENZY_WIDE" else "FRENZY", t))
    return [g[:4] for g in gates if not any(p == g[0] and abs(tc - g[4]) <= CATCHUP_SHADOW_MS for p, tc in cus)]


def fills_live(orders):
    """orders DataFrame (opened_at, pair, entry_strategy, entry_frenzy_gvol[, entry_frenzy_catchup]) → [(pair, close_ms, value, sleeve)] of
    FRENZY fills opened ≤ LIVE_FILL_MAX_MS after a 5m close (the signal close) with a stamp. A catch-up fill (entry_frenzy_catchup truthy) is
    skipped: its stamp is the older ON bar's reading."""
    out = []
    if orders is None or not len(orders) or "entry_frenzy_gvol" not in orders:
        return out
    o = orders[orders.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & pd.to_numeric(orders.entry_frenzy_gvol, errors="coerce").notna()]
    if "entry_frenzy_catchup" in o:
        cu = o.entry_frenzy_catchup
        o = o[~(cu.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"]) | (pd.to_numeric(cu, errors="coerce").fillna(0) != 0))]
    for _, r in o.iterrows():
        try:
            t = int(pd.Timestamp(str(r.opened_at)[:19], tz="UTC").value // 1_000_000)
        except Exception:
            continue
        close = t // BAR * BAR
        if t - close > LIVE_FILL_MAX_MS:
            continue
        out.append((str(r.pair), close, float(r.entry_frenzy_gvol), "WIDE" if r.entry_strategy == "FRENZY_WIDE" else "FRENZY"))
    return out


def live_map(df):
    """registry rows → {close_ms: (value, source, pair)}: a fill stamp (4 dp) wins over a log line (2 dp) on the same bar."""
    out = {}
    if df is None or not len(df):
        return out
    rank = {"stamp": 0, "log": 1}
    d = df.assign(_r=df.source.map(rank).fillna(9), _p=df.pair.astype(str)).sort_values(["close_ms", "_r", "_p"], kind="stable")
    for _, r in d.iterrows():   # deterministic: per bar, a stamp before a log line, then the pair name
        c = int(r.close_ms)
        if c not in out:
            out[c] = (float(r.value), SRC_STAMP if r.source == "stamp" else SRC_LOG, str(r.pair))
    return out


# ─────────────────────────── state ───────────────────────────
def _save(df, path):
    tmp = f"{path}.{os.getpid()}.tmp"; df.to_csv(tmp, index=False); os.replace(tmp, path)


def load_cache(path=None):
    path = path or CACHE_CSV
    cols = ["bar_ts", "gvol", "n_pairs", "ver", "computed_ms", "top3", "tries"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        d = pd.read_csv(path)
        if len(d) and not {"bar_ts", "gvol", "ver"} <= set(d.columns):
            raise ValueError("columns missing")
        return d.reindex(columns=list(dict.fromkeys(list(d.columns) + cols)))
    except Exception as e:
        bad = f"{path}.{int(time.time())}.bad"
        os.replace(path, bad)
        log(f"⚠ WARNING: market-volume cache unreadable ({str(e)[:80]}) — moved to {bad}; every bar will be recomputed")
        return pd.DataFrame(columns=cols)


def cached(sigs=None, path=None):
    """{bar_ts: value} of the frozen v2 bars (all, or only `sigs`)."""
    c = load_cache(path)
    if not len(c):
        return {}
    c = c[pd.to_numeric(c.ver, errors="coerce") >= VER]
    m = {int(b): float(v) for b, v in zip(c.bar_ts, c.gvol) if v == v}
    return m if sigs is None else {s: m[s] for s in sigs if s in m}


def tried_out(sigs=None, path=None):
    """bars whose COMPLETE reads stayed unreadable / short MAX_TRIES times (not retried any more)."""
    c = load_cache(path)
    if not len(c):
        return set()
    t = pd.to_numeric(c.tries, errors="coerce").fillna(0)
    m = set(int(b) for b, g, k in zip(c.bar_ts, pd.to_numeric(c.gvol, errors="coerce"), t) if g != g and k >= MAX_TRIES)
    return m if sigs is None else m & set(sigs)


def freeze(new_vals, now_ms, path=None):
    """{bar_ts: (value, n, top3)} from a COMPLETE read (every fetch succeeded) → the cache. A bar already frozen at VER is NEVER overwritten
    (the first computation wins); an older version's row is replaced. A value is frozen only with ≥ FREEZE_MIN_PAIRS pairs counted; otherwise
    (None / short) the bar's `tries` goes up by one and it is retried next run, at most MAX_TRIES times."""
    path = path or CACHE_CSV
    c = load_cache(path)
    gv = pd.to_numeric(c.gvol, errors="coerce") if len(c) else pd.Series(dtype=float)
    done = set(int(b) for b, v, g in zip(c.bar_ts, c.ver, gv) if pd.notna(v) and float(v) >= VER and g == g) if len(c) else set()
    prev_tries = {int(b): (0 if k != k else int(k)) for b, k in zip(c.bar_ts, pd.to_numeric(c.tries, errors="coerce"))} if len(c) else {}
    add = []
    for b, (v, n, t3) in new_vals.items():
        b = int(b)
        if b in done:
            continue
        if v is not None and int(n) >= FREEZE_MIN_PAIRS:
            add.append(dict(bar_ts=b, gvol=round(float(v), 4), n_pairs=int(n), ver=VER, computed_ms=int(now_ms), top3=t3, tries=prev_tries.get(b, 0) + 1))
        else:
            add.append(dict(bar_ts=b, gvol=None, n_pairs=int(n), ver=VER, computed_ms=int(now_ms), top3=t3, tries=prev_tries.get(b, 0) + 1))
    if not add:
        return c
    keep = c[~c.bar_ts.astype("int64").isin([a["bar_ts"] for a in add])] if len(c) else c
    out = pd.concat([keep, pd.DataFrame(add)], ignore_index=True).sort_values("bar_ts").reset_index(drop=True)
    _save(out, path)
    return out


def load_live(path=None):
    path = path or LIVE_CSV
    if not os.path.exists(path):
        return pd.DataFrame(columns=["pair", "close_ms", "value", "sleeve", "source", "seen_ms"])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["pair", "close_ms", "value", "sleeve", "source", "seen_ms"])


def update_live(now_ms, orders=None, log_paths=None, path=None, write=True):
    """merge the fill stamps of the orders exports + the gate lines of the server logs found into the registry → the registry DataFrame."""
    path = path or LIVE_CSV
    old = load_live(path)
    rows = []
    if orders is None:
        orders = _orders()
    rows += [dict(pair=p, close_ms=c, value=v, sleeve=s, source="stamp", seen_ms=now_ms) for p, c, v, s in fills_live(orders)]
    for f in (log_paths if log_paths is not None else _log_files()):
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                rows += [dict(pair=p, close_ms=c, value=v, sleeve=s, source="log", seen_ms=now_ms) for p, c, v, s in parse_log_lines(fh)]
        except OSError:
            continue
    if not rows:
        return old
    allr = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    allr = allr.drop_duplicates(["pair", "close_ms", "source"], keep="first").sort_values(["close_ms", "pair"]).reset_index(drop=True)
    if write and len(allr) != len(old):
        _save(allr, path)
    return allr


def _orders():
    fr = []
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "pair", "entry_strategy", "entry_frenzy_gvol",
                                                                       "entry_frenzy_catchup"))
        except Exception:
            continue
        if {"opened_at", "pair", "entry_strategy", "entry_frenzy_gvol"} <= set(d.columns):
            fr.append(d)
    return pd.concat(fr, ignore_index=True).drop_duplicates(["opened_at", "pair"]) if fr else pd.DataFrame()


def _log_files():
    extra = [g for g in os.environ.get("SCOUT_SERVER_LOG_GLOBS", "").split(os.pathsep) if g]
    out = []
    for g in list(LOG_GLOBS) + extra:
        for f in glob.glob(os.path.expanduser(g)):
            try:
                if time.time() - os.path.getmtime(f) <= 14 * 86400 and os.path.getsize(f) <= 500e6:
                    out.append(f)
            except OSError:
                continue
    return sorted(set(out))


# ─────────────────────────── network (public Binance, read-only) ───────────────────────────
def universe_meta(EX, retry):
    """{pair: dict(id, onboard, alpha, coin, status, delivery)} — every USDT perpetual market, ANY status. The engine's ticker list held a pair
    that was trading at the read; a pair delisted SINCE (10-05/06: PUMPBTC, 1000000BOB — both top-50 then, SETTLING now, no ticker) must stay
    in the universe of the bars it traded on, so the universe is the markets list and a pair counts at a bar only with that exact kline bar."""
    mk = retry(EX.load_markets) or {}
    out = {}
    for s, m in mk.items():
        if not s.endswith("/USDT:USDT"):
            continue
        info = (m or {}).get("info", {}) or {}
        if info.get("contractType") not in (None, "PERPETUAL"):
            continue
        sub = info.get("underlyingSubType") or []
        try:
            ob = int(info["onboardDate"]) if info.get("onboardDate") is not None else None
        except (TypeError, ValueError):
            ob = None
        try:
            dl = int(info["deliveryDate"]) if info.get("deliveryDate") is not None else None
        except (TypeError, ValueError):
            dl = None
        out[s.split("/")[0] + "USDT"] = dict(id=m.get("id") or info.get("symbol") or s.split("/")[0] + "USDT", onboard=ob,
                                             alpha=any("alpha" in str(x).lower() for x in (sub if isinstance(sub, list) else [sub])),
                                             coin=info.get("underlyingType") in (None, "COIN"), status=info.get("status"), delivery=dl)
    return out


def _alive(m, since_ms):
    """False for a non-trading market whose delivery (delist) date is more than a week before since_ms — long dead, no klines to fetch.
    The delivery date is not a reliable cutoff by itself (PUMPBTC traded ~22 h past it), hence the week of margin."""
    return m.get("status") in (None, "TRADING") or m.get("delivery") is None or m["delivery"] >= since_ms - 7 * DAY


FAPI = "https://fapi.binance.com/fapi/v1/klines"
WORKERS, WEIGHT_PER_MIN = 6, 1000          # the scout's own share of Binance's 2,400/min IP budget (the rest of the run uses the remainder)
GROUP_BARS = 288                           # one computation per ≤ 24 h of signal bars: 1h prescreen < 100 candles (weight 1), 5m ≤ 624 (weight 5)
_W = []                                    # (monotonic time, weight) of this process's kline calls
import threading  # noqa: E402
_WL = threading.Lock()
_BAN_UNTIL = [0.0]                         # wall time until which no call is made this process (a 418 / 429: Retry-After, ≥ 60 s)
_DEADLINES = {}                            # {run now_ms: monotonic deadline} — ONE gvol time budget per scout run, shared by every ensure()
_STOP = []                                 # non-empty = a fetch of the current group failed: the other threads stop at their next call
_PAUSE = [0.0]                             # monotonic time until which every call waits: the IP's used weight (shared with every other local
IP_WEIGHT_SOFT = 1500                      #   process) passed IP_WEIGHT_SOFT of Binance's 2,400 → wait for the next minute window


class Abort(Exception):
    pass


class Banned(Abort):
    """Binance answered 418 / 429: stop every gvol read of this run (ensure breaks out; _BAN_UNTIL blocks the next call)."""


def _weight(limit):
    return 1 if limit < 100 else 2 if limit < 500 else 5 if limit <= 1000 else 10


def _check(deadline):
    if _STOP:
        raise Abort("stopped")
    if time.time() < _BAN_UNTIL[0]:
        raise Banned("Binance rate-limit back-off")
    if deadline and time.monotonic() > deadline:
        raise Abort("time budget")


def _throttle(w, deadline=None):
    """wait for our own per-minute weight share and any IP-wide pause — checking the deadline / stop / ban on every wait."""
    while time.monotonic() < _PAUSE[0]:
        _check(deadline)
        time.sleep(0.25)
    while True:
        _check(deadline)
        with _WL:
            now = time.monotonic()
            while _W and now - _W[0][0] > 60:
                _W.pop(0)
            if sum(x[1] for x in _W) + w <= WEIGHT_PER_MIN:
                _W.append((now, w)); return
        time.sleep(0.25)


def _get(mid, interval, start, limit, deadline=None):
    """one raw klines call (public, read-only) → rows; raises Abort on a ban / rate-limit answer or the deadline (the group is not frozen)."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request
    url = f"{FAPI}?{urllib.parse.urlencode(dict(symbol=mid, interval=interval, startTime=int(start), limit=int(limit)))}"
    err = None
    for attempt in range(3):
        _throttle(_weight(limit), deadline)
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                try:
                    used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
                except (TypeError, ValueError):
                    used = 0
                if used > IP_WEIGHT_SOFT:   # the window resets on the minute: wait it out (all threads)
                    _PAUSE[0] = max(_PAUSE[0], time.monotonic() + 61 - time.time() % 60)
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                try:
                    ra = float(e.headers.get("Retry-After") or 0)
                except (TypeError, ValueError, AttributeError):
                    ra = 0.0
                _BAN_UNTIL[0] = max(_BAN_UNTIL[0], time.time() + max(60.0, ra))
                raise Banned(f"Binance {e.code} (retry after {max(60.0, ra):.0f} s)")
            if e.code == 400:
                try:
                    body = e.read().decode("utf-8", "replace")
                except Exception:
                    body = ""
                if "-1121" in body or "-1122" in body:   # "Invalid symbol" (retired) / "Invalid symbol status" (PENDING_TRADING, not listed
                    return []                            # yet — 10-07 GAIBUSDT aborted every read): no klines → not ranked at that bar
                raise Abort(f"{mid} {interval}: HTTP 400 {body[:80]}")   # any other 400 = a bad request → nothing frozen
            err = e
        except Exception as e:
            err = e
        time.sleep(1 + attempt)
    raise Abort(f"{mid} {interval}: {str(err)[:80]}")


def _klines(mid, interval, start, end, step, deadline=None):
    """raw klines [open, o, h, l, c, v, q] with open in [start, end] (field 7 = quote asset volume)."""
    out = {}
    s = int(start)
    while s <= end:
        n = int(min(1500, (end - s) // step + 1))
        r = _get(mid, interval, s, n, deadline) or []
        for x in r:
            t = int(x[0])
            if start <= t <= end:
                out[t] = [t, float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[7])]
        if not r or len(r) < n:
            break
        s = int(r[-1][0]) + step
    return [out[k] for k in sorted(out)]


def _fetch_all(jobs, deadline):
    """{key: rows} for jobs {key: (mid, interval, start, end, step)} on WORKERS threads; any failure → Abort (all-or-nothing)."""
    from concurrent.futures import ThreadPoolExecutor
    _STOP.clear()
    with ThreadPoolExecutor(WORKERS) as ex:
        fut = {k: ex.submit(_klines, *j, deadline=deadline) for k, j in jobs.items()}
        out = {}
        try:
            for k, f in fut.items():
                out[k] = f.result()      # re-raises Abort
        except Exception:
            _STOP.append(1)
            for f in fut.values():
                f.cancel()
            raise
    return out


def _from_frame(d, start, end):
    """a scout frame (index t, columns … v, q) → rows when it covers [start, end] with quote volume, else None."""
    if d is None or "q" not in getattr(d, "columns", []) or not len(d):
        return None
    if int(d.index.min()) > start or int(d.index.max()) < end:
        return None
    x = d[(d.index >= start) & (d.index <= end)]
    return [[int(t), r.o, r.h, r.l, r.c, r.v, r.q] for t, r in x.iterrows()]


def compute(meta, cfg, sigs, pre=None, deadline=None):
    """{sig: (value or None, n, top3)} for ≤ GROUP_BARS of signal bars, engine definition (1h prescreen of every eligible pair → 5m klines with
    quote volume for the candidates → per-bar rank). All-or-nothing: a failed / late fetch returns {} (nothing frozen, retried next run)."""
    sigs = sorted(set(int(s) for s in sigs))
    if not sigs or not meta:
        return {}
    h0 = (sigs[0] + BAR) // HOUR * HOUR - DAY; h1 = -(-(sigs[-1] + BAR) // HOUR) * HOUR
    el = set()
    for s in sigs:
        el |= {p for p in eligible(meta, s + BAR, cfg) if _alive(meta[p], h0)}
    try:
        hr = _fetch_all({p: (meta[p]["id"], "1h", h0, h1 - HOUR, HOUR) for p in sorted(el)}, deadline)
        cand = prescreen({p: [[x[0], x[6]] for x in r] for p, r in hr.items()}, meta, sigs, cfg)
        start = sigs[0] - (RANK_BARS + LOOKBACK) * BAR; end = sigs[-1]
        rows, jobs = {}, {}
        for p in sorted(cand):
            got = _from_frame((pre or {}).get(p), start, end)
            if got is not None:
                rows[p] = got
            else:
                jobs[p] = (meta[p]["id"], "5m", start, end, BAR)
        rows.update(_fetch_all(jobs, deadline))
    except Banned:
        raise                               # ensure() stops every read of this run
    except Exception as e:                  # Abort, or any network / parse failure: nothing frozen
        log(f"bars {len(sigs)} left unread this run ({e}) — retried next run")
        return {}
    empty = [p for p in cand if not rows.get(p)]
    if empty:                               # a candidate with no klines at all: the read is not complete → nothing frozen
        log(f"bars {len(sigs)} left unread this run (no klines for {', '.join(sorted(empty)[:5])}) — retried next run")
        return {}
    B = Bars(rows)
    out = {}
    for s in sigs:
        v, n, top = gvol_at(B, meta, s, cfg)
        sv = {p: (B.rows(p, s, 1) or [[0, 0, 0, 0, 0, 0]])[-1][5] for p in top}
        tot = sum(sv.values()) or 1.0
        t3 = " ".join(f"{p}:{sv[p] / tot * 100:.0f}%" for p in sorted(sv, key=lambda k: -sv[k])[:3])
        out[s] = (v, n, t3)
    return out


def ensure(EX, retry, cfg, sigs, now_ms, pre=None, budget_s=180, path=None):
    """{sig: value} for the requested CLOSED signal bars (open ms): frozen values as they are; the missing / old-version ones (≤ MAX_AGE_MS old)
    computed once and frozen, newest ≤ 24 h group first (a one-off backfill of older rows completes over successive runs within the time
    budget). Never raises (missing bars are simply absent)."""
    try:
        sigs = sorted({int(s) for s in sigs if s is not None and int(s) + BAR <= now_ms})
        have = cached(sigs, path)
        gone = tried_out(sigs, path)
        need = [s for s in sigs if s not in have and s not in gone and s >= now_ms - MAX_AGE_MS]
        if need and time.time() < _BAN_UNTIL[0]:
            log("Binance rate-limit back-off in force — market volume not read this call")
            need = []
        if need:
            groups, cur = [], [need[-1]]
            for s in reversed(need[:-1]):           # newest first
                if cur[-1] - s < GROUP_BARS * BAR:          # the group's newest bar − this one: a span ≤ 24 h
                    cur.insert(0, s)
                else:
                    groups.append(cur); cur = [s]
            groups.append(cur)
            deadline = _DEADLINES.setdefault(int(now_ms), time.monotonic() + budget_s)   # one budget per scout run (both callers)
            if time.monotonic() > deadline:
                log(f"{len(need)} bars left for the next run (this run's market-volume time budget is spent)")
                return have
            meta = universe_meta(EX, retry)
            for g in groups:
                if time.monotonic() > deadline:
                    log(f"{sum(len(x) for x in groups[groups.index(g):])} older bars left for the next run (time budget)")
                    break
                try:
                    vals = compute(meta, cfg, g, pre=pre, deadline=deadline)
                except Banned as e:
                    log(f"{e} — every market-volume read stops for this run")
                    break
                freeze(vals, now_ms, path)
            have = cached(sigs, path)
        return have
    except Exception as e:
        log(f"market volume read failed ({str(e)[:120]})")
        try:
            return cached(sigs, path)
        except Exception:
            return {}


if __name__ == "__main__":
    # one-shot: import the gate lines of server-log files given on the command line into the live registry
    if len(sys.argv) > 2 and sys.argv[1] == "--import-logs":
        r = update_live(int(time.time() * 1000), log_paths=sys.argv[2:])
        print(f"live registry: {len(r)} rows → {LIVE_CSV}")
