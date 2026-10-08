#!/usr/bin/env python3
"""🧭 Scout — two momentum-LONG trend observe lines (pre-registered 2026-10-08; OBSERVE only — never changes config, no bot API).

Source: reports/MOMENTUM_LONG_SLEEVE_CHECKLIST_2026-10-08.md "Recommendation (a)" + scripts/study_ml_checklist_*.py /
scripts/study_negflank2d_features.py (the feature rebuild copied here so values are identical — validated by selftest).

COHORT (both)  CLOSED full-size MOMENTUM LONG bot fills (MOMENTUM / empty strategy; MANUAL and *_PROBE out), dedup (opened_at, pair,
               direction), from the ~/Downloads orders exports — the NEG_DAILYUP loader (scout_b1h_negflank._orders parity).
               Washed-out = stamped entry_btc_off30d_high_pct ≤ −15 → shown apart, never judged (a missing stamp = not washed, as
               ML_B1H_NEGFLANK).

1. PAIR_1H_DOWNTREND (block side)  zone = stamped entry_pair_1h_ema20_200_gap_pct ≤ 0 (coin 1h EMA20 ≤ EMA200; stamped live since
   Sep-30). Counted from 2026-09-30 00:00 UTC; a fill without the stamp = UNSCORED. No Binance requests.
   Frozen verdict on the FIRST crossing prefix (zone N ≥ 15 ∧ ≥ 8 days, by open time) = scout_b1h_negflank.du_verdict / du_freeze:
   FILTER CANDIDATE (operator decides) iff WR < sleeve breakeven ∧ day-clustered P(mean < 0) ≥ 0.95 (4,000 resamples, seed 7) ∧ no
   day / pair ≥ 50 % of the gross loss; RETIRE if zone mean ≥ 0; else KEEP OBSERVING; one frozen re-read at N ≥ 30.
2. TREND_ALIGNED (keep side)  keep = stamped entry_btc_1h_slope > 0 ∧ BTC 4h EMA20 slope > 0 ∧ coin 4h EMA20 > EMA50; complement =
   scored ∧ not keep. The two 4h legs are REBUILT at review time exactly like study_negflank2d_features (k_btc_4h_slope /
   k_pair_gap4h_20_50): price c = close of the last 5m bar CLOSED at/before the fill (≤ 15 min old, close time T); 4h closes = last 5m
   close per 4h bucket (BTC: btc_1h.csv hours before the 5m era + COMPLETE 5m hours); the forming bucket b = T // 4h is the partial bar
   → ema_now = α·c + (1−α)·EMA[b−1] (ewm adjust=False); slope = (ema_now − EMA[b−3]) / EMA[b−3] × 100; gap = (EMA20_now / EMA50_now − 1) × 100.
   Buckets not covered by a full 48-bar 5m bucket take this line's fetched CLOSED 4h kline (same Binance close). < 200 4h buckets
   before the fill (cold EMA seed) or any leg unreadable → UNSCORED. Counted from 2026-10-07 00:00 UTC (B18 in), DAY units.
   Frozen at the FIRST prefix where keep AND complement each reach N ≥ 15 ∧ ≥ 8 days: RETIRE if keep mean ≤ complement mean;
   ML-ONLY-IN-ZONE CANDIDATE (operator decides) iff the complement meets the filter bar above ∧ keep mean ≥ +0.05 %; else KEEP
   OBSERVING; one frozen re-read when both reach 30.
Breakeven (both) = |avg loss| / (avg win + |avg loss|) of the scored, non-washed counted sleeve fills opened ≤ the prefix's last
fill (fallback 61.8 % until 30). No freeze while a counted fill is unscored only because a cache lags. Thresholds never re-fit.

BINANCE (TREND_ALIGNED only): cache first; ≤ 2 requests per run (oldest unscored counted fill first: a 5m page or a 4h page per
symbol); refused while this line's OR NEG_DAILYUP's 418/429 cooldown is live (IP-wide ban), once the last X-MBX-USED-WEIGHT-1M ≥ 900,
or past the 25 s wall-clock budget; a 418/429 stops the run and persists a cooldown (Retry-After honoured).
"""
import glob
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (shared loader + bar maths — one definition, no drift)

H, D4, M5 = 3_600_000, 4 * 3_600_000, 300_000
WASHED = -15.0
P1_START, TA_START = "2026-09-30 00:00", "2026-10-07 00:00"
N_MIN, DAYS_MIN, REREAD_N, BE_REF, BE_MIN_FILLS = 15, 8, 30, 61.8, 30
KEEP_MIN = 0.05                                     # keep-zone mean floor (%) for the TREND_ALIGNED candidate
WARM_4H = 200                                       # 4h buckets required before the fill (EMA50 seed)
BOOT_N, BOOT_SEED = 4000, 7
MAX_REQ, WEIGHT_STOP, BUDGET_S = 2, 900, 25.0
COLS = tuple(NF.COLS) + ("entry_pair_1h_ema20_200_gap_pct",)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, "reports", "backtest_cache")
MY_CACHE = os.path.join(_CACHE, "scout_ml_trend")
COOLDOWN = os.path.join(MY_CACHE, ".ratelimited_until")
STATE_TA = os.path.join(_ROOT, "reports", "SCOUT_ML_TREND_ALIGNED.json")
STATE_P1 = os.path.join(_ROOT, "reports", "SCOUT_ML_PAIR_1H_DOWNTREND.json")
REVERT = ("Pre-committed revert if ever armed: the first 10 refused signals re-priced with the live exit → WR ≥ 61 % or Σ > 0 → "
          "switch it off")
P1_REF = ("study (on the REBUILT coin 1h EMA20/200 gap): yr5 262 · 56 % · −0.123 % PASS (yr5 = discovery set) · master ex-washed 23 · "
          "52 % · −0.136 % · P 0.78 fail; on the LIVE stamp this line uses: yr5 262 · 57 % · −0.119 % (ex-washed 158 · −0.131 %), sign "
          "agreement rebuilt vs stamp 96.8 % yr5 / 13 of 15 master · forward B15→B18 8 · 38 % · ≈ −0.03 %")
TA_REF = ("study: keep yr5 150/seed · 71 % · +0.079 % vs complement −0.116 % · master ex-washed keep +0.235 % vs complement −0.043 % "
          "(the study used the REBUILT BTC 1h slope; this line uses the live stamp — 1h legs agree in sign on the stored master rows)")
_NET_BLOCKED = False                                # selftest: any fetch attempt raises


# ─────────────────────────── orders ───────────────────────────
def load_orders():
    """NF._orders(start=None, empty_is_momentum=True) + the coin 1h EMA20/200 stamp (NF.COLS does not carry it)."""
    saved = NF.COLS
    try:
        NF.COLS = COLS
        o = NF._orders(start=None, empty_is_momentum=True)
    finally:
        NF.COLS = saved
    return _prep(o)


def _prep(o):
    o = o.copy()
    o["g1h"] = pd.to_numeric(o.get("entry_pair_1h_ema20_200_gap_pct", pd.Series(np.nan, index=o.index)), errors="coerce")
    o["washed"] = (o.off30.notna() & (o.off30 <= WASHED)).values if len(o) else np.array([], dtype=bool)
    o["t_ms"] = o.ts.values.astype("datetime64[ms]").astype("int64") if len(o) else np.array([], dtype="int64")
    return o


# ─────────────────────────── kline rebuild (study parity) ───────────────────────────
def htf_partial(closes, bucket, span, T, c):
    """faithful copy of study_negflank2d_features.htf_partial → (ema_now, ema[b−3])."""
    E = closes.ewm(span=span, adjust=False).mean()
    a = 2.0 / (span + 1)
    T = np.asarray(T, dtype="int64")
    b = (T // bucket) * bucket
    e1 = E.reindex(b - bucket).values
    e3 = E.reindex(b - 3 * bucket).values
    return a * np.asarray(c, dtype=float) + (1 - a) * e1, e3


def _m5_paths(sym):
    p = [os.path.join(_CACHE, "k5m_full", f"{sym}.csv"), os.path.join(_CACHE, "negflank2d_ext", "5m", f"{sym}.csv")]
    if sym == "BTCUSDT":
        p.append(os.path.join(_CACHE, "scout_dailyup", "BTCUSDT_5m.csv"))     # NEG_DAILYUP keeps this fresh each run
    return p + [os.path.join(MY_CACHE, "5m", f"{sym}.csv")]


def load_m5(sym):
    fr = [NF._read_k(p) for p in _m5_paths(sym) if os.path.exists(p)]
    fr = [f for f in fr if len(f)]
    m = pd.concat(fr) if fr else pd.DataFrame({"open_time": pd.Series(dtype="int64"), "c": pd.Series(dtype=float)})
    m = m.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    m["open_time"] = m.open_time.astype("int64")
    m["T"] = m.open_time + M5
    return m


def h4_closes(m5, k4, h1_base=None):
    """4h closes by bucket open. Pair (h1_base None) = study sym_grid: hour-last of every 5m bar, then 4h-last. BTC (h1_base = btc_1h
    closes) = study btc_grid: base hours before the first COMPLETE 5m hour + complete 5m hours, then 4h-last. Buckets without a full
    48-bar 5m bucket take this line's fetched closed 4h kline when one exists.
    h4.attrs["ok"] = the COMPLETE buckets (pair: 48 × 5m; BTC: 4 complete hours; or a fetched 4h kline) — the coverage guards
    (_cold / _first_missing) count a partial bucket as MISSING even though the EMA (study convention) still uses its last close."""
    if len(m5):
        g = m5.groupby(m5.open_time // H * H).c
        h1 = g.last()
        if h1_base is not None:
            h1 = h1[g.size() == 12]
    else:
        h1 = pd.Series(dtype=float)
    if h1_base is not None and len(h1_base):
        h1 = pd.concat([h1_base[h1_base.index < h1.index.min()] if len(h1) else h1_base, h1]).sort_index()
    h4 = h1.groupby((h1.index // D4) * D4).last() if len(h1) else pd.Series(dtype=float)
    if h1_base is not None:                            # BTC: every h1 row is a complete hour → complete bucket = 4 hours
        n4 = h1.groupby((h1.index // D4) * D4).size() if len(h1) else pd.Series(dtype=int)
        ok = set(n4[n4 == 4].index.tolist())
    else:
        n4 = m5.groupby(m5.open_time // D4 * D4).size() if len(m5) else pd.Series(dtype=int)
        ok = set(n4[n4 == 48].index.tolist())
    full = set(ok)
    if len(k4):
        k = k4.drop_duplicates("open_time", keep="last").set_index("open_time").c.astype(float)
        ok |= set(k.index.tolist())
        k = k[[i not in full for i in k.index]]
        h4 = pd.concat([h4[~h4.index.isin(k.index)], k]).sort_index()
    h4.index = h4.index.astype("int64")
    h4 = h4.astype(float)
    h4.attrs["ok"] = np.array(sorted(ok), dtype="int64")
    return h4


def _cold(h4, T):
    """True where the WARM_4H buckets before the fill's forming bucket are not ALL present (cold EMA seed or a hole that would
    silently bend the EMA) → UNSCORED. WARM_4H 0 = the study's no-guard convention."""
    b = (np.asarray(T, dtype="int64") // D4) * D4
    if not WARM_4H:
        return np.zeros(len(b), dtype=bool)
    ix = _ok_ix(h4)
    if not len(ix):
        return np.ones(len(b), dtype=bool)
    return (np.searchsorted(ix, b, side="left") - np.searchsorted(ix, b - WARM_4H * D4, side="left")) < WARM_4H


def _ok_ix(h4):
    return np.asarray(h4.attrs.get("ok", h4.index.values), dtype="int64")


def _first_missing(h4, b):
    """earliest missing (or partial) bucket in [b − WARM_4H·4h, b − 4h], or None."""
    have = set(_ok_ix(h4).tolist())
    for k in range(WARM_4H, 0, -1):
        if b - k * D4 not in have:
            return b - k * D4
    return None


def btc4_slope(t_ms, m5, h4):
    T, c = NF.price_at(m5, t_ms)
    if not len(h4):
        return np.full(len(T), np.nan)
    now, e3 = htf_partial(h4, D4, 20, T, c)
    s = (now - e3) / e3 * 100
    return np.where(np.isnan(c) | _cold(h4, T), np.nan, s)


def pair4_gap(t_ms, m5, h4):
    T, c = NF.price_at(m5, t_ms)
    if not len(h4):
        return np.full(len(T), np.nan)
    e20, _ = htf_partial(h4, D4, 20, T, c)
    e50, _ = htf_partial(h4, D4, 50, T, c)
    g = (e20 / e50 - 1) * 100
    return np.where(np.isnan(c) | _cold(h4, T), np.nan, g)


def _btc_h1_base():
    return NF._read_k(os.path.join(_CACHE, "btc_1h.csv")).drop_duplicates("open_time").set_index("open_time").c.sort_index()


def _k4(sym):
    return NF._read_k(os.path.join(MY_CACHE, "4h", f"{sym}.csv"))


def score_ta(o):
    """→ (b4, p4) arrays for every row of o (cache only)."""
    b4, p4 = np.full(len(o), np.nan), np.full(len(o), np.nan)
    if not len(o):
        return b4, p4
    t = o.t_ms.values
    mb = load_m5("BTCUSDT")
    b4 = btc4_slope(t, mb, h4_closes(mb, _k4("BTCUSDT"), _btc_h1_base()))
    for sym, idx in o.groupby(o.pair.astype(str)).indices.items():
        mp = load_m5(sym)
        p4[idx] = pair4_gap(t[idx], mp, h4_closes(mp, _k4(sym)))
    return b4, p4


# ─────────────────────────── Binance (TREND_ALIGNED only) ───────────────────────────
def _http_get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode()), r.headers


def _cooldown_until():
    """the ONE shared scout cooldown (NF.SHARED_COOLDOWN, the ban is per IP) + legacy per-line files (read for one release)."""
    out = NF._cooldown_until()
    try:
        with open(COOLDOWN) as f:
            out = max(out, int(f.read().strip() or 0))
    except Exception:
        pass
    return out


def _arm_cooldown(code, retry_after, now_ms):
    return NF._arm_cooldown(code, retry_after, now_ms)            # writes the shared file (never shortens a live ban)


def _fetch(sym, interval, start_ms, limit, now_ms, timeout_s=10.0):
    """ONE public USDⓈ-M klines request in a daemon thread with a hard wall-clock join → (CLOSED bars, used weight)."""
    if _NET_BLOCKED:
        raise RuntimeError("network blocked (selftest)")
    url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={interval}&startTime={int(start_ms)}"
           f"&limit={int(limit)}")
    box = {}

    def _go():
        try:
            box["ok"] = _http_get(url, max(1.0, timeout_s))
        except BaseException as e:
            box["err"] = e
    th = threading.Thread(target=_go, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError(f"{sym} {interval} fetch exceeded {timeout_s:.0f} s wall clock")
    e = box.get("err")
    if isinstance(e, urllib.error.HTTPError) and e.code in (418, 429):
        secs = _arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now_ms)
        raise NF.RateLimited(f"Binance {e.code} — stopped, cooldown {secs} s")
    if e is not None:
        raise e
    rows, headers = box["ok"]
    used = int((headers or {}).get("X-MBX-USED-WEIGHT-1M") or 0)
    step = D4 if interval == "4h" else M5
    d = pd.DataFrame([(int(x[0]), float(x[4])) for x in rows if int(x[0]) + step <= now_ms], columns=["open_time", "c"])
    return d, used


def _append(kind, sym, new):
    os.makedirs(os.path.join(MY_CACHE, kind), exist_ok=True)
    p = os.path.join(MY_CACHE, kind, f"{sym}.csv")
    d = pd.concat([NF._read_k(p), new]).drop_duplicates("open_time", keep="last").sort_values("open_time")
    tmp = f"{p}.{os.getpid()}.tmp"
    d.to_csv(tmp, index=False)
    os.replace(tmp, p)


def _plan(sym, t_ms, m5, h4):
    """→ (interval, start, limit) for the first missing input of a fill, or None. A 5m hole AT the fill (cache already past it) or a
    cache too far behind plans the fixed page t − 15 min, so a page Binance cannot fill is recognised on the next run."""
    T, c = NF.price_at(m5, [t_ms])
    if np.isnan(c[0]):
        end = int(m5.open_time.max()) if len(m5) else None
        start = end + M5 if end is not None and end < t_ms and t_ms - end <= 1500 * M5 else t_ms - 15 * 60_000
        return "5m", int(start), 1500
    b = (int(T[0]) // D4) * D4
    ok = _ok_ix(h4)
    if not len(ok) or ok.min() > b - WARM_4H * D4:            # cold / no series → one 1000-bucket page ending before the fill
        return "4h", int(b - 1000 * D4), 1000
    miss = _first_missing(h4, b)
    return ("4h", int(miss), 1000) if miss is not None else None


def _att_path():
    return os.path.join(MY_CACHE, "attempts.json")


def load_attempts():
    """{'done': {'SYM|interval|start': bars}, 'first': {SYM: first 4h open ms seen on a cold page}} — successful pages only."""
    try:
        with open(_att_path()) as f:
            a = json.load(f)
        return dict(done=dict(a.get("done") or {}), first=dict(a.get("first") or {}))
    except Exception:
        return dict(done={}, first={})


def save_attempts(a):
    os.makedirs(MY_CACHE, exist_ok=True)
    tmp = f"{_att_path()}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(a, f)
    os.replace(tmp, _att_path())


def _key(sym, pl):
    return f"{sym}|{pl[0]}|{int(pl[1])}"


class _Data:
    """per-run memo of (m5, h4) per symbol (reloaded after an append)."""
    def __init__(self):
        self.m = {}

    def get(self, sym, fresh=False):
        if fresh or sym not in self.m:
            m5 = load_m5(sym)
            self.m[sym] = (m5, h4_closes(m5, _k4(sym), _btc_h1_base() if sym == "BTCUSDT" else None))
        return self.m[sym]


def _need(sym, t_ms, data, att):
    """→ ('final', reason) | ('fetch', plan) | ('none', None) for one unreadable leg."""
    m5, h4 = data.get(sym)
    T, c = NF.price_at(m5, [t_ms])
    if sym in att["first"] and not np.isnan(c[0]):
        if int(att["first"][sym]) > (int(T[0]) // D4) * D4 - WARM_4H * D4:
            return "final", f"young listing ({sym} 4h klines start {pd.Timestamp(int(att['first'][sym]), unit='ms'):%Y-%m-%d}, < {WARM_4H} bars before the fill)"
    pl = _plan(sym, t_ms, m5, h4)
    if pl is None:
        return "final", f"{sym} unreadable with complete inputs"
    if _key(sym, pl) in att["done"]:
        return "final", f"{sym} {pl[0]} page from {pd.Timestamp(pl[1], unit='ms'):%m-%d %H:%M} already fetched — Binance has no bars there"
    return "fetch", pl


def _pending(cnt):
    """counted, non-washed fills with the 1h stamp whose 4h legs are unreadable (the only fills that can lag or be fetched for)."""
    return cnt[cnt.slope.notna() & ~cnt.washed.astype(bool) & (cnt.b4.isna() | cnt.p4.isna())].sort_values("ts", kind="stable")


def final_reasons(cnt, att=None, data=None):
    """→ Series index → reason ('' = still a cache lag) over _pending(cnt)."""
    att, data = att or load_attempts(), data or _Data()
    out = {}
    for r in _pending(cnt).itertuples():
        rs = []
        for sym in (["BTCUSDT"] if np.isnan(r.b4) else []) + ([str(r.pair)] if np.isnan(r.p4) else []):
            kind, x = _need(sym, int(r.t_ms), data, att)
            if kind == "final":
                rs.append(x)
        out[r.Index] = "; ".join(rs)
    return pd.Series(out, dtype=object)


def ensure_data(cnt, now_ms, fetch=True, budget_s=BUDGET_S, max_req=MAX_REQ):
    """cache first → ≤ max_req requests for the oldest pending fills (final-unreadable and washed-out fills never fetch). → notes."""
    notes, st = [], dict(used=0, stop=None, n=0, deadline=time.monotonic() + budget_s)
    if not fetch or not len(cnt):
        return notes
    cd = _cooldown_until()
    if cd > now_ms:
        return [f"fetching stopped: rate-limit cooldown until {pd.Timestamp(cd, unit='ms'):%m-%d %H:%M} UTC"]
    att, data, done = load_attempts(), _Data(), set()
    for r in _pending(cnt).itertuples():
        for sym in (["BTCUSDT"] if np.isnan(r.b4) else []) + ([str(r.pair)] if np.isnan(r.p4) else []):
            if st["n"] >= max_req or st["stop"]:
                break
            kind, pl = _need(sym, int(r.t_ms), data, att)
            if kind != "fetch" or _key(sym, pl) in done:
                continue
            if st["used"] >= WEIGHT_STOP:
                st["stop"] = f"used weight {st['used']} ≥ {WEIGHT_STOP}"
                break
            left = st["deadline"] - time.monotonic()
            if left < 1.0:
                st["stop"] = "wall-clock budget spent"
                break
            done.add(_key(sym, pl))
            st["n"] += 1
            try:
                d, st["used"] = _fetch(sym, pl[0], pl[1], pl[2], now_ms, timeout_s=min(10.0, left))
                if len(d):
                    _append(pl[0], sym, d)
                att["done"][_key(sym, pl)] = len(d)                  # successful page → never re-requested
                if pl[0] == "4h" and pl[2] == 1000 and len(d) and int(d.open_time.min()) > pl[1]:
                    att["first"][sym] = int(d.open_time.min())       # cold page started before the listing
                save_attempts(att)
                data.get(sym, fresh=True)
                notes.append(f"{sym} {pl[0]} +{len(d)} bars (weight used {st['used']})")
            except NF.RateLimited as e:
                st["stop"] = str(e)
            except Exception as e:
                notes.append(f"{sym} {pl[0]} fetch failed ({str(e)[:80]})")
                if isinstance(e, TimeoutError):
                    st["stop"] = "wall-clock timeout"
        if st["n"] >= max_req or st["stop"]:
            break
    if st["stop"]:
        notes.append(f"fetching stopped: {st['stop']}")
    if st["n"] >= max_req:
        notes.append(f"request cap {max_req}/run reached — the rest catches up on later runs")
    return notes


# ─────────────────────────── groups / verdicts / freeze ───────────────────────────
def p1_group(g1h):
    g = np.asarray(g1h, dtype=float)
    return np.where(np.isnan(g), "unscored", np.where(g <= 0, "zone", "rest"))


def ta_group(slope, b4, p4):
    s, b, p = (np.asarray(x, dtype=float) for x in (slope, b4, p4))
    uns = np.isnan(s) | np.isnan(b) | np.isnan(p)
    return np.where(uns, "unscored", np.where((s > 0) & (b > 0) & (p > 0), "keep", "comp"))


def _bar(c, be):
    """the locked filter bar on cohort c → (meets, detail)."""
    m, wr = float(c.pct.mean()), 100.0 * float((c.pct > 0).mean())
    p = NF.p_mean_neg(c.pct.values, c.day.values, BOOT_N, BOOT_SEED)
    sd, sp = NF.gross_loss_share(c.pct.values, c.day.values), NF.gross_loss_share(c.pct.values, c.pair.values)
    ok = wr < be and p is not None and p >= 0.95 and sd < 0.5 and sp < 0.5
    return ok, (f"N {len(c)} · {c.day.nunique()} d · WR {wr:.0f} % vs breakeven {be:.1f} % · mean {m:+.3f} % · P(mean<0) "
                f"{(p if p is not None else float('nan')):.2f} · top day {sd * 100:.0f} % / top pair {sp * 100:.0f} % of the gross loss")


def ta_verdict(pre, be, n_min=N_MIN):
    k, c = pre[pre.grp == "keep"], pre[pre.grp == "comp"]
    nk, nc, dk, dc = len(k), len(c), k.day.nunique(), c.day.nunique()
    if nk < n_min or nc < n_min or dk < DAYS_MIN or dc < DAYS_MIN:
        return "COLLECTING", f"keep N {nk}/{n_min} · {dk}/{DAYS_MIN} d · complement N {nc}/{n_min} · {dc}/{DAYS_MIN} d"
    mk, mc = float(k.pct.mean()), float(c.pct.mean())
    ok, det = _bar(c, be)
    det = f"keep {nk} · {dk} d · WR {(k.pct > 0).mean() * 100:.0f} % · mean {mk:+.3f} % | complement {det}"
    if mk <= mc:
        return "RETIRE", det
    if ok and mk >= KEEP_MIN:
        return "ML-ONLY-IN-ZONE CANDIDATE (operator decides)", det
    return "KEEP OBSERVING", det


def ta_prefix(sleeve, n_min):
    """smallest prefix (by open time) of the scored judged sleeve where keep AND complement each have N ≥ n_min ∧ ≥ DAYS_MIN days."""
    z = sleeve.sort_values(["ts", "pair"], kind="stable")
    cnt, days = {"keep": 0, "comp": 0}, {"keep": set(), "comp": set()}
    for k, (g, d) in enumerate(zip(z.grp.values, z.day.values), 1):
        cnt[g] += 1
        days[g].add(d)
        if all(cnt[x] >= n_min and len(days[x]) >= DAYS_MIN for x in ("keep", "comp")):
            return z.iloc[:k]
    return None


def _be(bs):
    be_live = NF.breakeven_wr(bs.pct) if len(bs) >= BE_MIN_FILLS else None
    return (be_live, "live") if be_live is not None else (BE_REF, "fallback")


def ta_freeze(st, sleeve, now_iso, hold=None, why=None):
    """NF.du_freeze semantics for the two-sided line: 'first' at the first crossing, 're-read' at 30 on a LATER call, never recomputed."""
    key = "reread" if "first" in st else "first"
    if key in st:
        return st, False
    pre = ta_prefix(sleeve, REREAD_N if key == "reread" else N_MIN)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if NF.freeze_hold(last, hold, why):                           # an OPEN / cache-lagging fill opened ≤ the prefix end → defer
        return st, False
    be, src = _be(sleeve[sleeve.ts <= last])
    state, det = ta_verdict(pre, be, REREAD_N if key == "reread" else N_MIN)
    st[key] = dict(state=state, detail=det, be=round(be, 2), be_src=src, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso,
                   n=len(pre), days=int(pre.day.nunique()),
                   keys=[f"{a}|{b}|{g}" for a, b, g in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str), pre.grp)])
    return st, True


def p1_freeze(st, sleeve, now_iso, hold=None, why=None):
    """= NF.du_freeze (zone crossing prefix, breakeven of the scored sleeve ≤ prefix end, fallback 61.8 %, re-read at 30)."""
    return NF.du_freeze(st, sleeve, now_iso, hold, why)


# ─────────────────────────── rendering ───────────────────────────
def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean() * 100:.0f} % | {g.pct.mean():+.3f} % | "
            f"{g.usd.sum():+,.0f} | {g.pct.min():+.2f} % |")


HDR = ["| Group | N | days | WR | avg % | Σ$ as-sized | worst |", "|---|---|---|---|---|---|---|"]
REF_HDR = ["| Reference only (before the counting start) — as traded, not comparable to the study | N | days | WR | avg % | "
           "Σ$ as-sized | worst |", "|---|---|---|---|---|---|---|"]


def _frozen_lines(st, live_state, live_det, bar_txt):
    if "first" not in st:
        return [f"{bar_txt} Now: ⏳ {live_state} ({live_det})."]
    f0 = st["first"]
    L = [f"**Frozen verdict (first crossing at fill {f0['at']}, frozen on the run of {f0.get('run_at', '?')}, N {f0['n']} · "
         f"{f0['days']} d, breakeven {f0['be']} % {f0['be_src']}): {f0['state']}** ({f0['detail']})"]
    if "reread" in st:
        r0 = st["reread"]
        L.append(f"**Frozen re-read (at fill {r0['at']}): {r0['state']}** ({r0['detail']})")
    return L + [f"Live (information only, never re-decides): {live_state} — {live_det}"]


def _freeze_io(path, now_ms, fn, sleeve, hold, why):
    st, ok = NF.du_load_state(path, now_ms)
    if ok:
        st, changed = fn(st, sleeve, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), hold, why)
        if changed:
            NF.du_save_state(st, path)
    return st, ok


def _why_line(why):
    return ["⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run."] if why else []


def run_p1(o, now_ms, state_path=None, open_ts=None):
    state_path = state_path or STATE_P1
    o = o.copy()
    o["grp"] = p1_group(o.g1h)
    cm = o.ts >= pd.Timestamp(P1_START)
    cnt, ref = o[cm], o[~cm]
    sc = cnt[(cnt.grp != "unscored") & ~cnt.washed]               # judged sleeve (zone + rest), washed-out apart
    be, src = _be(sc)
    zone = sc[sc.grp == "zone"].sort_values(["ts", "pair"], kind="stable")
    why = []
    st, ok = _freeze_io(state_path, now_ms, p1_freeze, sc, list(open_ts or []), why)
    live_state, live_det = NF.du_verdict(zone[["pct", "day", "pair"]], be)
    w = cnt[cnt.washed & (cnt.grp != "unscored")]
    L = ["## 📐 PAIR_1H_DOWNTREND — momentum longs on a coin in a 1h downtrend (pre-registered 2026-10-08, OBSERVE only, block side)", "",
         f"Zone (frozen): stamped `entry_pair_1h_ema20_200_gap_pct` ≤ 0 (coin 1h EMA20 ≤ EMA200). Counted: full-size momentum LONG fills "
         f"opened ≥ {P1_START} UTC (stamp start); washed-out (BTC ≤ {WASHED:g} % below its 30-day high) shown apart, never judged. "
         f"Reference: {P1_REF}.", "", *HDR,
         _row("**zone (1h EMA20 ≤ EMA200), judged**", zone), _row("rest (1h EMA20 > EMA200)", sc[sc.grp == "rest"]),
         _row("washed-out, zone (apart)", w[w.grp == "zone"]), _row("washed-out, rest (apart)", w[w.grp == "rest"]),
         _row("UNSCORED (no stamp)", cnt[cnt.grp == "unscored"]), "", *REF_HDR,
         _row("zone", ref[ref.grp == "zone"]), _row("rest", ref[ref.grp == "rest"]), _row("UNSCORED (no stamp)", ref[ref.grp == "unscored"]), ""]
    if len(zone):
        L += ["Counted zone fills: " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair} {r.pct:+.2f} % (1h gap {r.g1h:+.3f})"
                                                  for r in zone.itertuples()), ""]
    L += _why_line(why)
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += _frozen_lines(st, live_state, live_det,
                       f"Bar (locked expectancy, frozen at the first crossing zone N ≥ {N_MIN} ∧ ≥ {DAYS_MIN} days): WR < sleeve breakeven "
                       f"{be:.1f} % ({src}, {len(sc)} scored counted fills) ∧ day-clustered P(mean < 0) ≥ 0.95 ({BOOT_N:,} resamples, seed "
                       f"{BOOT_SEED}) ∧ no day / pair ≥ 50 % of the gross loss → FILTER CANDIDATE (operator decides); zone mean ≥ 0 → RETIRE; "
                       f"else KEEP OBSERVING (one re-read at N ≥ {REREAD_N}).")
    return L + [REVERT, ""]


def run_ta(o, now_ms, state_path=None, fetch=True, open_ts=None):
    state_path = state_path or STATE_TA
    o = o.copy()
    cm = o.ts >= pd.Timestamp(TA_START)
    o["b4"], o["p4"] = score_ta(o)
    notes = ensure_data(o[cm], now_ms, fetch=fetch)
    if any(" bars (" in x for x in notes):                        # a request landed → rescore from the cache
        o["b4"], o["p4"] = score_ta(o)
    o["grp"] = ta_group(o.slope, o.b4, o.p4)
    cnt, ref = o[cm], o[~cm]
    fin = final_reasons(cnt)
    lag = pd.Series(False, index=cnt.index)
    lag[fin[fin == ""].index] = True                              # pending (non-washed, stamped) and not final → a cache lag
    sc = cnt[(cnt.grp != "unscored") & ~cnt.washed]
    be, src = _be(sc)
    why = []
    hold = list(open_ts or []) + [(cnt.loc[i, "ts"], f"cache-lagging {cnt.loc[i, 'pair']}") for i in lag[lag].index]
    st, ok = _freeze_io(state_path, now_ms, ta_freeze, sc, hold, why)
    live_state, live_det = ta_verdict(sc, be)
    w = cnt[cnt.washed & (cnt.grp != "unscored")]
    keep, comp = sc[sc.grp == "keep"], sc[sc.grp == "comp"]
    L = ["## 🧭 TREND_ALIGNED — momentum longs only when BTC 1h ∧ BTC 4h ∧ coin 4h trend agree (pre-registered 2026-10-08, OBSERVE only, keep side)", "",
         "Keep (frozen): stamped `entry_btc_1h_slope` > 0 ∧ BTC 4h EMA20 slope > 0 ∧ coin 4h EMA20 > EMA50 (both 4h legs rebuilt from klines, "
         f"study convention); complement = scored ∧ not keep. Counted: full-size momentum LONG fills opened ≥ {TA_START} UTC; DAY units; "
         f"washed-out shown apart. Reference: {TA_REF}.", "", *HDR,
         _row("**keep (trend-aligned), judged**", keep), _row("**complement (not aligned), judged**", comp),
         _row("washed-out, keep (apart)", w[w.grp == "keep"]), _row("washed-out, complement (apart)", w[w.grp == "comp"]),
         _row("UNSCORED (a leg unreadable)", cnt[cnt.grp == "unscored"]), "", *REF_HDR,
         _row("keep", ref[ref.grp == "keep"]), _row("complement", ref[ref.grp == "comp"]), _row("UNSCORED", ref[ref.grp == "unscored"]), ""]
    if len(cnt):
        L += ["Counted fills: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair} {r.pct:+.2f} % [{r.grp}] (BTC 1h {r.slope:+.3f} · BTC 4h {r.b4:+.3f} · coin 4h gap {r.p4:+.3f})"
            for r in cnt.sort_values(["ts", "pair"], kind="stable").itertuples()), ""]
    if (fin != "").any():
        L.append("UNSCORED for good (not a lag, never blocks the freeze): " + " · ".join(
            f"{cnt.loc[i, 'ts']:%m-%d %H:%M} {cnt.loc[i, 'pair']} — {why}" for i, why in fin[fin != ""].items()))
    if lag.any():
        L.append(f"⏳ {int(lag.sum())} counted fill(s) UNSCORED only because a kline cache lags — re-scored next run (a lagging fill "
                 f"defers a freeze only if opened ≤ the crossing prefix's last fill).")
    L += _why_line(why)
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += _frozen_lines(st, live_state, live_det,
                       f"Bar (frozen at the first prefix where keep AND complement each reach N ≥ {N_MIN} ∧ ≥ {DAYS_MIN} days): keep mean ≤ "
                       f"complement mean → RETIRE; complement meets the filter bar (WR < breakeven {be:.1f} % ({src}) ∧ day-clustered "
                       f"P(mean < 0) ≥ 0.95 ∧ no day / pair ≥ 50 % of the gross loss) ∧ keep mean ≥ +{KEEP_MIN:.2f} % → ML-ONLY-IN-ZONE "
                       f"CANDIDATE (operator decides); else KEEP OBSERVING (one re-read when both reach {REREAD_N}).")
    return L + [REVERT] + ([f"Data: {' · '.join(notes)}."] if notes else []) + [""]


def run(now_ms=None, orders=None, fetch=True):
    """→ markdown lines; each line has its own guard (one can't take the other down)."""
    now_ms = now_ms or int(time.time() * 1000)
    o = load_orders() if orders is None else _prep(orders)
    try:                                               # OPEN momentum longs in the newest export defer a freeze (they may join the prefix)
        opens = NF.open_momentum_long_ts() if orders is None else []
    except Exception:
        opens = []
    L = []
    for name, fn in (("PAIR_1H_DOWNTREND", lambda: run_p1(o, now_ms, open_ts=opens)),
                     ("TREND_ALIGNED", lambda: run_ta(o, now_ms, fetch=fetch, open_ts=opens))):
        try:
            L += fn()
        except Exception as e:
            L += [f"## {name}", "", f"Unavailable this run ({str(e)[:120]}).", ""]
    return L


# ─────────────────────────── study parity ───────────────────────────
def validate_vs_study(n_yr5=60):
    """rebuild k_btc_4h_slope / k_pair_gap4h_20_50 on the checklist frame's stored rows (cache only) with the cold-seed guard OFF
    (the study had none) → (n, n_ok ≤ 1e-6, max_abs, n_cold = rows the live guard would leave UNSCORED) or None."""
    global WARM_4H, MY_CACHE
    pk = os.path.join(_ROOT, "reports", "study_ml_checklist_frame.pkl")
    if not (os.path.exists(pk) and os.path.exists(os.path.join(_CACHE, "k5m_full", "BTCUSDT.csv"))):
        return None
    F = pd.read_pickle(pk)
    F = F[F.k_btc_4h_slope.notna() & F.k_pair_gap4h_20_50.notna()]
    y = F[F.src == "yr5"]
    F = pd.concat([F[F.src == "master"], y.iloc[:: max(1, len(y) // n_yr5)]])
    F = F[[os.path.exists(os.path.join(_CACHE, "k5m_full", f"{p}.csv")) for p in F.pair]]
    o = pd.DataFrame(dict(t_ms=F.t_ms.values.astype("int64"), pair=F.pair.values))
    saved = (WARM_4H, MY_CACHE)
    with tempfile.TemporaryDirectory() as td:                    # hermetic: this line's own fetched cache is never read here
        try:
            MY_CACHE = td
            b4g, p4g = score_ta(o)
            WARM_4H = 0
            b4, p4 = score_ta(o)
        finally:
            WARM_4H, MY_CACHE = saved
    err = np.maximum(np.abs(b4 - F.k_btc_4h_slope.values), np.abs(p4 - F.k_pair_gap4h_20_50.values))   # NaN propagates
    err = np.where(np.isnan(err), np.inf, err)
    n_cold = int((np.isnan(b4g) | np.isnan(p4g)).sum())
    return len(F), int((err <= 1e-6).sum()), float(err.max()) if len(err) else 0.0, n_cold


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    global _NET_BLOCKED, _CACHE, MY_CACHE, COOLDOWN, _http_get
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    _NET_BLOCKED = True
    saved = (_CACHE, MY_CACHE, COOLDOWN, NF._CACHE, NF.DU_CACHE, NF.DU_COOLDOWN, NF.SHARED_COOLDOWN, _http_get)
    try:
        chk((N_MIN, DAYS_MIN, REREAD_N, BE_REF, BE_MIN_FILLS, BOOT_N, BOOT_SEED, WASHED) ==
            (NF.N_MIN, NF.DAYS_MIN, NF.DU_REREAD_N, NF.DU_BE_REF, NF.BE_MIN_FILLS, NF.DU_BOOT_N, NF.DU_BOOT_SEED, NF.WASHED)
            == (15, 8, 30, 61.8, 30, 4000, 7, -15.0),
            "pre-registered constants pinned: this line's rule == NF's shared du_verdict / du_freeze constants (an NF edit fails here)")
        chk(list(p1_group([0.0, -0.1, 0.0001, np.nan])) == ["zone", "zone", "rest", "unscored"], "PAIR_1H zone = gap ≤ 0, NaN unscored")
        chk(list(ta_group([0.1, 0.0, 0.1, 0.1, np.nan, 0.1], [0.1, 0.1, -0.1, 0.1, 0.1, 0.2], [0.1, 0.1, 0.1, 0.0, 0.1, np.nan]))
            == ["keep", "comp", "comp", "comp", "unscored", "unscored"], "TREND_ALIGNED keep = all three > 0 (strict); any NaN unscored")
        try:
            _fetch("BTCUSDT", "4h", 0, 1, 0)
            chk(False, "fetch must be blocked")
        except RuntimeError as e:
            chk("blocked" in str(e), "network blocked inside the selftest")
        # htf_partial = a hand-rolled EMA on [.., b−3, b−2, b−1, partial]
        idx = np.arange(300) * D4
        cl = pd.Series(100 + np.sin(np.arange(300) / 7.0) * 5, index=idx)
        T = np.array([idx[250] + 2 * H])
        now, e3 = htf_partial(cl, D4, 20, T, [101.0])
        a, e = 2 / 21, cl.iloc[0]
        es = [e]
        for v in cl.iloc[1:].values:
            e = a * v + (1 - a) * e
            es.append(e)
        chk(abs(now[0] - (a * 101.0 + (1 - a) * es[249])) < 1e-9 and abs(e3[0] - es[247]) < 1e-9, "htf_partial forming-bar EMA")
        # 4h precedence: full 48-bar 5m bucket > fetched 4h kline > partial 5m bucket
        m5 = pd.DataFrame(dict(open_time=np.r_[np.arange(48) * M5, D4 + np.arange(10) * M5], c=np.r_[np.full(48, 1.0), np.full(10, 2.0)]))
        k4 = pd.DataFrame(dict(open_time=[0, D4, 2 * D4], c=[9.0, 3.0, 4.0]))
        h = h4_closes(m5, k4)
        chk(list(h.values) == [1.0, 3.0, 4.0], f"4h precedence (full 5m > kline > partial 5m): {list(h.values)}")
        chk(list(h4_closes(m5, k4.iloc[:0]).values) == [1.0, 2.0], "no klines → study convention (partial bucket last close)")
        chk(bool(_cold(cl, [idx[150]])[0]) and not bool(_cold(cl, [idx[250]])[0]), f"< {WARM_4H} buckets before the fill → cold")
        holed = cl.drop(idx[240])
        chk(bool(_cold(holed, [idx[250]])[0]) and _first_missing(holed, idx[250]) == idx[240] and _first_missing(cl, idx[250]) is None,
            "a hole in the last 200 buckets → UNSCORED + the 4h plan starts at the hole")
        # a PARTIAL 5m bucket (< 48 bars, no 4h kline) counts as missing for the guards; a fetched 4h kline completes it
        nb = 210
        mp = pd.DataFrame(dict(open_time=np.arange(nb * 48) * M5, c=1.0))
        mp = mp[~((mp.open_time >= 205 * D4) & (mp.open_time < 205 * D4 + 8 * M5))]
        hp = h4_closes(mp, pd.DataFrame(columns=["open_time", "c"]))
        chk(205 * D4 in hp.index and bool(_cold(hp, [nb * D4])[0]) and _first_missing(hp, nb * D4) == 205 * D4,
            "partial 4h bucket from 5m → missing (UNSCORED, 4h plan starts there)")
        hk = h4_closes(mp, pd.DataFrame(dict(open_time=[205 * D4], c=[1.0])))
        chk(not bool(_cold(hk, [nb * D4])[0]) and _first_missing(hk, nb * D4) is None, "… and a fetched 4h kline completes it")
        hb = h4_closes(pd.DataFrame(dict(open_time=np.arange(4 * 12) * M5, c=1.0)), pd.DataFrame(columns=["open_time", "c"]),
                       pd.Series(1.0, index=(np.arange(3) * H + D4).astype("int64")))
        chk(list(_ok_ix(hb)) == [0], "BTC: a bucket with < 4 complete hours is not complete")
        # verdicts
        t0 = pd.Timestamp("2026-10-10 01:00")
        days = [f"2026-10-{10 + i // 2:02d}" for i in range(16)]
        z = pd.DataFrame(dict(ts=[t0 + pd.Timedelta(days=i // 2, hours=i % 2) for i in range(16)], pct=[-0.3, 0.1] * 8, day=days,
                              pair=[f"P{i}" for i in range(16)], grp="zone"))
        st, ch = p1_freeze({}, z, "t")
        chk(ch and st["first"]["state"] == "FILTER CANDIDATE (operator decides)" and st["first"]["n"] == 15, "PAIR_1H freeze = du_freeze")
        keep = z.assign(grp="keep", pct=[0.4, 0.2] * 8, ts=z.ts + pd.Timedelta(minutes=1), pair=[f"K{i}" for i in range(16)])
        comp = z.assign(grp="comp")
        sl = pd.concat([keep, comp], ignore_index=True)
        pre = ta_prefix(sl, N_MIN)
        chk(pre is not None and (pre.grp == "keep").sum() >= 15 and (pre.grp == "comp").sum() == 15, "two-sided crossing prefix")
        chk(ta_prefix(sl[sl.grp == "keep"], N_MIN) is None, "complement missing → no crossing")
        s1, ch1 = ta_freeze({}, sl, "t1")
        chk(ch1 and s1["first"]["state"] == "ML-ONLY-IN-ZONE CANDIDATE (operator decides)", f"bad complement + keep ≥ +0.05 → candidate: {s1['first']}")
        why = []
        sh, chh = ta_freeze({}, sl, "th", [(sl.ts.min(), "cache-lagging Q")], why)
        chk(not chh and sh == {} and "cache-lagging Q" in why[0], "a lagging / OPEN fill opened ≤ the prefix end defers the freeze")
        sh, chh = ta_freeze({}, sl, "th", [(pd.Timestamp("2027-01-01"), "OPEN Z")], [])
        chk(chh and sh["first"]["state"] == s1["first"]["state"], "a lagging / OPEN fill opened after the prefix end does not")
        s2, ch2 = ta_freeze(json.loads(json.dumps(s1)), sl, "t2")
        chk(not ch2 and s2 == json.loads(json.dumps(s1)), "N < 30 → no re-read; frozen first never recomputed")
        chk(ta_verdict(sl.assign(pct=np.where(sl.grp == "keep", -0.5, 0.1)), 61.8)[0] == "RETIRE", "keep ≤ complement → RETIRE")
        chk(ta_verdict(sl.assign(pct=np.where(sl.grp == "keep", 0.03, sl.pct)), 61.8)[0] == "KEEP OBSERVING", "keep < +0.05 → keep observing")
        chk(ta_verdict(sl, 30.0)[0] == "KEEP OBSERVING", "complement WR above breakeven → keep observing")
        with tempfile.TemporaryDirectory() as td:
            _CACHE = os.path.join(td, "cache")
            MY_CACHE = os.path.join(_CACHE, "scout_ml_trend")
            COOLDOWN = os.path.join(MY_CACHE, ".ratelimited_until")
            NF._CACHE, NF.DU_CACHE = _CACHE, os.path.join(_CACHE, "scout_dailyup")
            NF.DU_COOLDOWN = os.path.join(NF.DU_CACHE, ".ratelimited_until")
            # hermetic synthetic caches: BTC hours rising (btc_1h.csv) + 5m around the fills; ARB 5m rising, UNI 5m falling
            os.makedirs(os.path.join(_CACHE, "k5m_full"))
            t_f = pd.Timestamp("2026-10-07 16:31:56").value // 10**6
            h0 = t_f - 400 * 4 * H
            nh = (t_f - h0) // H - 2
            pd.DataFrame(dict(open_time=h0 + np.arange(nh) * H, c=50_000 * 1.0005 ** np.arange(nh))).to_csv(
                os.path.join(_CACHE, "btc_1h.csv"), index=False)
            m0 = t_f - 300 * 4 * H
            nm = (t_f - m0) // M5 + 6
            for sym, g in (("BTCUSDT", 1.00002), ("ARBUSDT", 1.00002), ("UNIUSDT", 0.99998)):
                pd.DataFrame(dict(open_time=m0 + np.arange(nm) * M5, c=10 * g ** np.arange(nm))).to_csv(
                    os.path.join(_CACHE, "k5m_full", f"{sym}.csv"), index=False)
            d0 = pd.Timestamp("2026-10-07 16:31:56")
            orders = pd.DataFrame(dict(ts=[d0, d0 + pd.Timedelta(minutes=5), d0 + pd.Timedelta(minutes=9), d0 + pd.Timedelta(minutes=11),
                                           d0 - pd.Timedelta(days=10)],
                                       pair=["ARBUSDT", "UNIUSDT", "LITUSDT", "WASHUSDT", "OLDUSDT"], pct=[-0.69, -0.70, 0.2, 0.1, 0.4],
                                       usd=[-155.0, -154.0, 30.0, 10.0, 50.0], slope=[0.2, 0.2, 0.2, 0.2, np.nan],
                                       off30=[-5.0, -5.0, -5.0, -20.0, np.nan],
                                       entry_pair_1h_ema20_200_gap_pct=[-0.3, 0.4, np.nan, -0.5, -1.0], gap=np.nan,
                                       day=["2026-10-07"] * 4 + ["2026-09-27"], _k=list("abcde")))
            sp1, sp2 = os.path.join(td, "p1.json"), os.path.join(td, "ta.json")
            o = _prep(orders)
            out1 = "\n".join(run_p1(o, 1791490000000, sp1))
            chk("| **zone (1h EMA20 ≤ EMA200), judged** | 1 |" in out1 and "| rest (1h EMA20 > EMA200) | 1 |" in out1
                and "| washed-out, zone (apart) | 1 |" in out1 and "| UNSCORED (no stamp) | 1 |" in out1 and not os.path.exists(sp1),
                "PAIR_1H end-to-end (ARB zone, UNI rest, WASH apart, LIT unscored)")
            out2 = "\n".join(run_ta(o, 1791490000000, sp2, fetch=False))
            chk("| **keep (trend-aligned), judged** | 1 |" in out2 and "| **complement (not aligned), judged** | 1 |" in out2
                and "| UNSCORED (a leg unreadable) | 2 |" in out2 and "1 counted fill(s) UNSCORED only because a kline cache lags" in out2
                and not os.path.exists(sp2), "TREND_ALIGNED end-to-end (ARB keep, UNI complement, LIT lag, washed WASH never a lag)")
            empty = _prep(pd.DataFrame(dict(ts=pd.to_datetime([]), pair=[], pct=[], usd=[], slope=[], off30=[], gap=[], day=[],
                                            entry_pair_1h_ema20_200_gap_pct=[])))
            out3 = "\n".join(run(1791490000000, orders=empty.drop(columns=["g1h", "washed", "t_ms"]), fetch=False))
            chk("COLLECTING" in out3 and "Unavailable" not in out3, "empty orders → collecting, never raises")
            # fetch path: LIT has no cache → plan = 5m page; 429 → ONE request, SHARED cooldown persisted, next run zero requests
            NF.SHARED_COOLDOWN = os.path.join(_CACHE, ".binance_cooldown_until")
            calls = []

            def _boom(url, timeout):
                calls.append(url)
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "120"}, None)
            _NET_BLOCKED, _http_get = False, _boom
            ta = o.assign(b4=np.nan, p4=np.nan)
            ta = ta[ta.ts >= pd.Timestamp(TA_START)]
            notes = ensure_data(ta, 1791490000000)
            chk(len(calls) == 1 and "LITUSDT" in calls[0] and abs(_cooldown_until() - (1791490000000 + 120_000)) < 2
                and any("429" in x for x in notes), f"429 → one request (LIT; washed WASH never fetched) + cooldown ({calls}, {notes})")
            chk(os.path.exists(NF.SHARED_COOLDOWN) and abs(NF._cooldown_until() - (1791490000000 + 120_000)) < 2,
                "this line's 429 lands in the SHARED file → NEG_DAILYUP sees it")
            notes = ensure_data(ta, 1791490000000 + 60_000)
            chk(len(calls) == 1 and any("cooldown" in x for x in notes), "cooldown live → zero requests")
            os.remove(NF.SHARED_COOLDOWN)
            for legacy in (NF.DU_COOLDOWN, COOLDOWN):
                os.makedirs(os.path.dirname(legacy), exist_ok=True)
                open(legacy, "w").write(str(1791490000000 + 999_999))
                ensure_data(ta, 1791490000000)
                chk(len(calls) == 1, f"legacy cooldown {legacy[len(_CACHE):]} honoured")
                os.remove(legacy)
            NF._arm_cooldown(418, None, 1791490000000)
            ensure_data(ta, 1791490000000)
            chk(len(calls) == 1, "a NEG_DAILYUP 418 (shared file) blocks this line")
            os.remove(NF.SHARED_COOLDOWN)

            def _ok(url, timeout):
                calls.append(url)
                return [], {"X-MBX-USED-WEIGHT-1M": "950"}
            _http_get = _ok
            many = pd.concat([ta.assign(pair=f"X{i}USDT") for i in range(5)], ignore_index=True)
            notes = ensure_data(many, 1791490000000)
            chk(len(calls) == 2 and any("used weight" in x for x in notes), f"used weight ≥ {WEIGHT_STOP} → no further request ({len(calls)})")

            def _ok2(url, timeout):
                calls.append(url)
                return [], {"X-MBX-USED-WEIGHT-1M": "5"}
            _http_get = _ok2
            ensure_data(many, 1791490000000)
            chk(len(calls) == 2 + MAX_REQ and "X0USDT" not in "".join(calls[2:]),
                f"request cap {MAX_REQ} per run; X0's empty page is never re-requested ({len(calls) - 2})")
            # repeated empty page → UNSCORED for good: no request, not a lag
            x0 = many[many.pair == "X0USDT"].head(1)
            fr = final_reasons(x0.assign(b4=0.1))
            chk(len(fr) == 1 and "already fetched" in fr.iloc[0], f"repeated empty page → final ({fr.to_dict()})")
            # young listing: a cold 4h page whose first bar is after its start → first-kline recorded → final, never a lag
            yo = x0.assign(pair="YOUNGUSDT", b4=0.1)
            tY = int(yo.t_ms.iloc[0])

            def _young(url, timeout):
                calls.append(url)
                q = dict(x.split("=") for x in url.split("?")[1].split("&"))
                if q["interval"] == "5m":
                    return [[tY - 10 * 60_000 + i * M5, 0, 0, 0, "1.0"] for i in range(3)], {"X-MBX-USED-WEIGHT-1M": "5"}
                b_ = (tY - 5 * 60_000) // D4 * D4
                return [[b_ - (50 - i) * D4, 0, 0, 0, "1.0"] for i in range(50)], {"X-MBX-USED-WEIGHT-1M": "5"}
            _http_get = _young
            n0 = len(calls)
            ensure_data(yo, 1791490000000)                  # run 1: the 5m page (one leg → one request per fill per run)
            ensure_data(yo, 1791490000000)                  # run 2: the cold 4h page → only 50 bars exist
            att = load_attempts()
            fr = final_reasons(yo)
            chk(len(calls) - n0 == 2 and "YOUNGUSDT" in att["first"] and "young listing" in fr.iloc[0],
                f"young listing → first kline recorded → final ({calls[n0:]}, {fr.to_dict()})")
            ensure_data(yo, 1791490000000)
            chk(len(calls) - n0 == 2, "young listing → zero further requests")

            def _kl(url, timeout):                          # synthetic Binance: LIT rising, 5m page then a 4h page
                calls.append(url)
                q = dict(x.split("=") for x in url.split("?")[1].split("&"))
                st_, lim = int(q["startTime"]), int(q["limit"])
                step = M5 if q["interval"] == "5m" else D4
                return [[st_ + i * step, 0, 0, 0, str(10 * 1.00002 ** ((st_ + i * step - m0) // M5))] for i in range(lim)], \
                    {"X-MBX-USED-WEIGHT-1M": "12"}
            _http_get = _kl
            lit = o[o.pair == "LITUSDT"].assign(b4=0.1, p4=np.nan)
            n0 = len(calls)
            ensure_data(lit, t_f + 20 * 60_000)
            ensure_data(lit, t_f + 20 * 60_000)
            gp = pair4_gap(lit.t_ms.values, load_m5("LITUSDT"), h4_closes(load_m5("LITUSDT"), _k4("LITUSDT")))
            chk(len(calls) - n0 == 2 and "interval=5m" in calls[-2] and "interval=4h" in calls[-1] and gp[0] > 0,
                f"fetch path: 5m page, then a cold 4h page → LIT scored from this line's cache ({calls[n0:]}, gap {gp})")
            ensure_data(lit, t_f + 20 * 60_000)
            chk(len(calls) - n0 == 2, "covered → zero requests")

            def _slow(url, timeout):
                calls.append(url)
                time.sleep(3)
            _http_get = _slow
            t0_ = time.monotonic()
            notes = ensure_data(many, 1791490000000, budget_s=1.5)
            chk(time.monotonic() - t0_ < 2.5 and any("wall-clock" in x for x in notes), "hung request cut by the wall-clock budget")
            _NET_BLOCKED = True
    finally:
        _CACHE, MY_CACHE, COOLDOWN, NF._CACHE, NF.DU_CACHE, NF.DU_COOLDOWN, NF.SHARED_COOLDOWN, _http_get = saved
        _NET_BLOCKED = True
    v = validate_vs_study()
    if v is None:
        print("  study parity SKIPPED (reports/study_ml_checklist_frame.pkl or the k5m_full cache missing — not a failure)")
    else:
        n, good, mx, n_cold = v
        chk(n >= 10 and good == n, f"4h rebuild reproduces the checklist frame on {good}/{n} stored rows (max abs err {mx:.2e})")
        print(f"  4h legs reproduce the checklist frame on {good}/{n} stored rows (master + yr5 sample), max abs err {mx:.1e}; "
              f"{n_cold} rows would be UNSCORED live by the coverage guard ({WARM_4H} complete 4h buckets before the fill; cache starts Dec-30)")
    _NET_BLOCKED = False
    print(f"selftest ML trend lines OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
