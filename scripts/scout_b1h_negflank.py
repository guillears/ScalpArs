#!/usr/bin/env python3
"""📉 Scout — ML_B1H_NEGFLANK observation (pre-registered 2026-10-05, operator "yes add it to scout"; OBSERVE only — never changes config).

Question: do full-size momentum LONGs opened while BTC's 1h slope is FALLING lose? yr5 says yes (falling-EMA20 fills −0.125 % vs rising
−0.001 %, 144 days), real master fills say no (44 · 77 % WR · +0.154 %, carried by the washed-out Jun-18→Jul-2 window; without it −0.040 %
vs +0.174 %). Research: reports/H1_EMA20_OVERLAP_2026-10-05.md. It is the negative flank the LONG_BTC1H_DEADBAND gate (blocks (−0.05, +0.025))
deliberately lets through — a pass would lead to a 1hPullback cell sizing verdict, then a dead-band re-scope review; never straight to arming.

COHORT (frozen)  CLOSED full-size MOMENTUM LONG fills (MANUAL and *_PROBE excluded) opened ≥ 2026-10-06 00:00 UTC with the stamped
                 entry_btc_1h_slope ≤ −0.05 (the engine's own gate input). Washed-out = entry_btc_off30d_high_pct ≤ −15 — shown both ways.
BAR (the locked expectancy filter bar, judged on the cohort EXCLUDING washed-out fills):
                 ① WR < the sleeve's breakeven WR = |avg loss| / (avg win + |avg loss|) on ALL momentum-long fills since the same start
                   (master reference 63.9 % until ≥ 30 sleeve fills) · ② P(mean < 0) ≥ 0.95 by a DAY-clustered bootstrap (market-wide
                   variable → the day is the unit) · ③ ≥ 8 distinct days ∧ no single day or pair ≥ 50 % of the cohort's loss · ④ N ≥ 15.
                 Then the 30–50 % in-sample haircut and a pre-committed revert gate. Never re-fit −0.05 / −15.
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

START = "2026-10-06 00:00"
SLOPE_MAX, WASHED, N_MIN, DAYS_MIN, BE_REF, BE_MIN_FILLS = -0.05, -15.0, 15, 8, 63.9, 30
COLS = ("opened_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "cell_multiplier_source",
        "entry_btc_1h_slope", "entry_btc_off30d_high_pct", "entry_pair_ema20_ema50_gap_pct", "pnl")
YEAR_REF = ("yr5 falling 1h EMA20: 492 · 60 % · −0.125 % (144 days) vs rising −0.001 % · master falling 44 · 77 % · +0.154 % "
            "(ex washed-out 31 · −0.040 %)")


# ─────────────────────────── pure (selftest) ───────────────────────────
def breakeven_wr(pct):
    w, l = pct[pct > 0], pct[pct <= 0]
    if not len(w) or not len(l):
        return None
    aw, al = float(w.mean()), abs(float(l.mean()))
    return 100.0 * al / (aw + al) if (aw + al) > 0 else None


def p_mean_neg(pct, day, n=4000, seed=7):
    """day-clustered bootstrap: resample DAYS with replacement, pooled mean of their fills; → P(mean < 0)."""
    g = pd.DataFrame(dict(p=pct, d=day)).groupby("d").p.agg(["sum", "count"])
    if len(g) < 2:
        return None
    s, c = g["sum"].values, g["count"].values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    return float(((s[idx].sum(1) / c[idx].sum(1)) < 0).mean())


def top_loss_share(pct, key):
    """largest single key's share of the summed losing-day / losing-pair nets (conservative: one net-negative key = 100 %)."""
    net = pd.Series(pct).groupby(pd.Series(key).values).sum()
    neg = net[net < 0]
    return float(neg.min() / neg.sum()) if len(neg) else 0.0


def decide(c, be):
    """c = DataFrame(pct, day, pair) of the judged cohort → (state, detail)."""
    n, nd = len(c), c.day.nunique() if len(c) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "collecting", f"N {n}/{N_MIN} · days {nd}/{DAYS_MIN}"
    wr = 100.0 * float((c.pct > 0).mean())
    p = p_mean_neg(c.pct.values, c.day.values)
    sd, sp = top_loss_share(c.pct.values, c.day.values), top_loss_share(c.pct.values, c.pair.values)
    ok = wr < be and p is not None and p >= 0.95 and sd < 0.5 and sp < 0.5
    return ("passes" if ok else "fails"), (f"WR {wr:.0f} % vs breakeven {be:.1f} % · P(mean<0) {p:.2f} · top day {sd * 100:.0f} % / "
                                           f"top pair {sp * 100:.0f} % of the loss")


# ─────────────────────────── data ───────────────────────────
def _orders(start=START, empty_is_momentum=False):
    """CLOSED full-size momentum LONG fills opened ≥ start (None = all). empty_is_momentum: legacy pre-column fills (Apr–Jun, no
    entry_strategy) count as MOMENTUM like study_ml_b18_common (NEG_DAILYUP_WEAKPAIR only — ML_B1H_NEGFLANK keeps its frozen filter)."""
    fr = []
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
            if len(d) and {"opened_at", "pnl_percentage", "entry_btc_1h_slope"} <= set(d.columns):
                fr.append(d.assign(_m=os.path.getmtime(f)))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["ts", "pct", "slope", "off30", "day", "gap", "usd"])   # no usable export → "collecting", never a crash (review)
    o = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable")
    o["_k"] = o.opened_at.astype(str).str[:19]
    o = o.drop_duplicates(["_k", "pair", "direction"], keep="last")      # cross-export dedup key (opened_at, pair, direction)
    o = o[(o.status.astype(str).str.upper() == "CLOSED") & (o.direction.astype(str) == "LONG")]
    if empty_is_momentum:                              # study parity (study_ml_b18_common): exact MOMENTUM / "" / nan
        o = o[o.entry_strategy.fillna("MOMENTUM").astype(str).isin(["MOMENTUM", "", "nan"])]
    else:
        o = o[o.entry_strategy.fillna("").astype(str).str.split(":").str[0] == "MOMENTUM"]
    o = o[~o.get("cell_multiplier_source", pd.Series("", index=o.index)).astype(str).str.endswith("_PROBE")]
    t = pd.to_datetime(o._k, format="ISO8601", errors="coerce")
    o = o[t.notna()].assign(ts=t[t.notna()])
    if start is not None:
        o = o[o.ts >= pd.Timestamp(start)]
    o["pct"] = pd.to_numeric(o.pnl_percentage, errors="coerce")
    o["slope"] = pd.to_numeric(o.entry_btc_1h_slope, errors="coerce")
    o["off30"] = pd.to_numeric(o.get("entry_btc_off30d_high_pct"), errors="coerce")
    o["day"] = o.ts.dt.strftime("%Y-%m-%d")
    o["gap"] = pd.to_numeric(o.get("entry_pair_ema20_ema50_gap_pct"), errors="coerce")   # holds EMA13−EMA50 (misnamed; never rename)
    o["usd"] = pd.to_numeric(o.get("pnl"), errors="coerce")
    return o[o.pct.notna()]


def _line(g):
    return f"{len(g)} · {(g.pct > 0).mean() * 100:.0f}% · {g.pct.mean():+.3f} % · {g.day.nunique()} d" if len(g) else "0"


def run():
    """→ markdown lines. Never raises past the caller's try."""
    o = _orders()
    sl = o[o.slope.notna()]
    coh = sl[sl.slope <= SLOPE_MAX]
    washed = coh.off30.notna() & (coh.off30 <= WASHED)
    judged = coh[~washed]
    be_live = breakeven_wr(o.pct) if len(o) >= BE_MIN_FILLS else None
    be = be_live if be_live is not None else BE_REF
    state, det = decide(judged[["pct", "day", "pair"]], be)
    L = ["## 📉 ML_B1H_NEGFLANK — momentum longs opened while BTC 1h slope ≤ −0.05 (pre-registered, OBSERVE only)", "",
         f"Full-size momentum LONG fills from {START} UTC (stamped `entry_btc_1h_slope`; the dead-band's open negative flank). Judged on the "
         f"cohort without washed-out fills (BTC ≤ {WASHED:g} % below its 30-day high). Bar = the expectancy filter bar: WR < sleeve breakeven ∧ "
         f"day-clustered P(mean < 0) ≥ 0.95 ∧ ≥ {DAYS_MIN} days ∧ N ≥ {N_MIN} ∧ no day / pair ≥ 50 % of the loss. Year reference: {YEAR_REF}.", "",
         "| Group | N · WR · avg % · days |", "|---|---|",
         f"| **BTC 1h ≤ −0.05, ex washed-out (judged)** | {_line(judged)} |",
         f"| BTC 1h ≤ −0.05, washed-out only | {_line(coh[washed])} |",
         f"| BTC 1h > −0.05 (rest of the sleeve) | {_line(sl[sl.slope > SLOPE_MAX])} |",
         f"| no 1h stamp | {len(o) - len(sl)} |", "",
         f"Sleeve breakeven WR {be:.1f} % ({'live, ' + str(len(o)) + ' fills' if be_live is not None else 'master reference until ' + str(BE_MIN_FILLS) + ' sleeve fills'}) · "
         + {"collecting": "⏳ collecting", "passes": "📋 PASSES the bar → review (cell sizing verdict first, never straight to arming)",
            "fails": "❌ bar not met"}[state] + f" ({det})", ""]
    try:                                               # 🧭 (253) NEG_DAILYUP_WEAKPAIR — own guard: never takes the section above down
        L += run_dailyup()
    except Exception as e:
        L += ["## 🧭 NEG_DAILYUP_WEAKPAIR", "", f"Unavailable this run ({str(e)[:120]}).", ""]
    return L


# ═══════════════════ 🧭 (253) NEG_DAILYUP_WEAKPAIR — OBSERVE only (pre-registered 2026-10-08) ═══════════════════
# Source: reports/NEGFLANK_2D_STUDY_2026-10-08.md §4, family "B1" (ex-washed scan #1; scan-null p 0.22 → NOT a survivor; observe-only).
# ZONE (frozen, never re-fit): entry_btc_1h_slope ≤ −0.05 ∧ entry_pair_ema20_ema50_gap_pct ≤ 0.238 (the column holds EMA13−EMA50) ∧ BTC 1d
#   EMA20 slope > 0.2811, the slope REBUILT from BTCUSDT futures daily klines exactly like scripts/study_negflank2d_features.py
#   (k_btc_1d_slope): price c = close of the last BTC 5m bar CLOSED at/before the fill (≤ 15 min old); the forming UTC day is the partial
#   bar → ema_now = α·c + (1−α)·EMA20[day−1], α = 2/21 (ewm adjust=False, seeded on the first cached day 2025-01-01); slope =
#   (ema_now − EMA20[day−3]) / EMA20[day−3] × 100 (the engine's ema[−4] on [.., d−3, d−2, d−1, d(partial)]). Any leg unreadable → UNSCORED.
# COUNTING: fills opened ≥ 2026-10-07 00:00 UTC (B18's ARB/UNI/LIT = the first window, operator); earlier fills = reference only.
# BAR (locked expectancy bar, judged in DAY units — the 1d leg is market-wide; computed ONCE at the first crossing N ≥ 15 ∧ ≥ 8 days and
#   frozen in reports/SCOUT_NEG_DAILYUP_WEAKPAIR.json; one frozen re-read at N ≥ 30; thresholds never re-fit).
DU_START = "2026-10-07 00:00"
DU_GAP_MAX, DU_D1_MIN, DU_BE_REF, DU_REREAD_N = 0.238, 0.2811, 61.8, 30
DU_BOOT_N, DU_BOOT_SEED = 4000, 7
DD_MS, M5_MS, STALE_MS = 86_400_000, 300_000, 900_000
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, "reports", "backtest_cache")
DU_CACHE = os.path.join(_CACHE, "scout_dailyup")                       # this line's own appends (1d + 5m BTCUSDT)
DU_STATE = os.path.join(_ROOT, "reports", "SCOUT_NEG_DAILYUP_WEAKPAIR.json")
DU_COOLDOWN = os.path.join(DU_CACHE, ".ratelimited_until")                # LEGACY per-line file (read one release, no longer written)
SHARED_COOLDOWN = os.path.join(_CACHE, ".binance_cooldown_until")        # ms epoch; ONE file for every scout Binance line (the ban is per IP)
M5_PAGES = 4                                                           # max 5m pages per run (catch-up after a long scout gap)
D1_WARM_DAYS = 60                                                      # daily EMA20 needs ≥ 60 seeded days before a fill, else UNSCORED
DU_REVERT = "Pre-committed revert if ever armed: first 10 blocked signals re-priced → WR ≥ 61 % or Σ > 0 → switch the block off"
DU_REF = ("study yr5 46/seed · 36 % · −0.385 % vs NEG-not-zone −0.065 % · master ex-B1 10 · 50 % · −0.245 % (7 · 71 % · −0.052 % ex-B18) · "
          "scan-null p 0.22 (not a survivor) · dose-response non-monotone")
_NET_BLOCKED = False                                                   # selftest sets True: any fetch attempt raises


def du_ema(s, span=20):
    return s.ewm(span=span, adjust=False).mean()


def d1_slope(d1, T, c, span=20):
    """faithful copy of study_negflank2d_features.htf_partial(d1, DD, 20, T, c) → slope %. d1 = CLOSED daily closes indexed by day
    open (ms, contiguous); T = the 5m bar close time(s) (ms); c = that bar's close. Missing day−1 / day−3 → NaN."""
    E = du_ema(d1, span)
    a = 2.0 / (span + 1)
    T = np.asarray(T, dtype="int64")
    b = (T // DD_MS) * DD_MS
    e1 = E.reindex(b - DD_MS).values
    e3 = E.reindex(b - 3 * DD_MS).values
    now = a * np.asarray(c, dtype=float) + (1 - a) * e1
    return (now - e3) / e3 * 100


def price_at(m5, t_ms):
    """study lookup(): last 5m bar with close time T ≤ t (side='right'), NaN if none or > 15 min old → (T, c)."""
    t_ms = np.asarray(t_ms, dtype="int64")
    if not len(m5):
        return np.zeros(len(t_ms), dtype="int64"), np.full(len(t_ms), np.nan)
    Tg, cg = m5["T"].values.astype("int64"), m5.c.values.astype(float)
    i = np.searchsorted(Tg, t_ms, side="right") - 1
    ii = np.clip(i, 0, None)
    ok = (i >= 0) & ((t_ms - Tg[ii]) <= STALE_MS)
    return np.where(ok, Tg[ii], 0), np.where(ok, cg[ii], np.nan)


def _read_k(path, cols=("open_time", "c")):
    """cache csv → numeric (open_time, c); unparsable rows dropped (a bad cache leaves fills UNSCORED, never 'Unavailable')."""
    try:
        d = pd.read_csv(path, usecols=list(cols))
        for k in cols:
            d[k] = pd.to_numeric(d[k], errors="coerce")
        d = d.dropna()
        d["open_time"] = d.open_time.astype("int64")
        return d
    except Exception:
        return pd.DataFrame({k: pd.Series(dtype="int64" if k == "open_time" else float) for k in cols})


def build_daily(k1d_base, k1d_scout, m5):
    """daily closes by UTC day open. Precedence = complete 5m days (the study's convention from Dec-30 on) > this line's fetched CLOSED 1d
    klines > the shared k1d cache (its LAST row dropped — it can be a forming day). Truncated at the first gap (EMA needs contiguity)."""
    base = k1d_base.drop_duplicates("open_time").sort_values("open_time").iloc[:-1] if len(k1d_base) else k1d_base
    parts = [base.set_index("open_time").c, k1d_scout.drop_duplicates("open_time", keep="last").set_index("open_time").c]
    if len(m5):
        g = m5.groupby(m5.open_time // DD_MS * DD_MS)
        parts.append(g.c.last()[g.size() == 288])
    d = pd.concat([p.astype(float) for p in parts if len(p)]) if any(len(p) for p in parts) else pd.Series(dtype=float)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    if len(d) > 1:
        brk = np.flatnonzero(np.diff(d.index.values) != DD_MS)
        if len(brk):
            d = d.iloc[: brk[0] + 1]
    return d


def _load_m5():
    fr = [_read_k(os.path.join(_CACHE, "k5m_full", "BTCUSDT.csv")),
          _read_k(os.path.join(_CACHE, "negflank2d_ext", "5m", "BTCUSDT.csv")),
          _read_k(os.path.join(DU_CACHE, "BTCUSDT_5m.csv"))]
    m = pd.concat([f for f in fr if len(f)]) if any(len(f) for f in fr) else pd.DataFrame(columns=["open_time", "c"])
    m = m.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    m["open_time"] = m.open_time.astype("int64")
    m["T"] = m.open_time + M5_MS
    return m


class RateLimited(RuntimeError):
    """Binance 418/429 — aborts every further request this run and arms a persisted cooldown."""


def _http_get(url, timeout):
    """→ (rows, headers). Isolated so the selftest can monkeypatch it (no network in tests)."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode()), r.headers


def _cooldown_paths():
    """shared file + the legacy per-line files (NEG_DAILYUP, scout_ml_trend_lines) — read for one release."""
    return [SHARED_COOLDOWN, DU_COOLDOWN, os.path.join(_CACHE, "scout_ml_trend", ".ratelimited_until")]


def _cooldown_until():
    out = 0
    for p in _cooldown_paths():
        try:
            with open(p) as f:
                out = max(out, int(f.read().strip() or 0))
        except Exception:
            pass
    return out


def _arm_cooldown(code, retry_after, now_ms):
    try:
        secs = int(float(retry_after))
    except (TypeError, ValueError):
        secs = 7200 if code == 418 else 600            # default 2 h for a 418 ban, 10 min for a 429
    os.makedirs(os.path.dirname(SHARED_COOLDOWN), exist_ok=True)
    tmp = f"{SHARED_COOLDOWN}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(str(int(max(now_ms + secs * 1000, _cooldown_until()))))     # never shortens a longer live ban
    os.replace(tmp, SHARED_COOLDOWN)
    return secs


def _fetch(interval, start_ms, limit, now_ms, timeout_s=10.0):
    """ONE public Binance USDⓈ-M klines request → (CLOSED bars, X-MBX-USED-WEIGHT-1M). The call runs in a daemon thread joined with a
    hard wall-clock timeout (DNS included). 418/429 → RateLimited + persisted cooldown (Retry-After honoured). Weight gating is done by
    the caller BEFORE each request (load_btc refuses a request once the last seen used weight ≥ 900)."""
    if _NET_BLOCKED:
        raise RuntimeError("network blocked (selftest)")
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval={interval}&startTime={int(start_ms)}&limit={int(limit)}"
    box = {}

    def _go():
        try:
            box["ok"] = _http_get(url, max(1.0, timeout_s))
        except BaseException as e:                     # handed back to the caller thread
            box["err"] = e
    th = threading.Thread(target=_go, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError(f"{interval} fetch exceeded {timeout_s:.0f} s wall clock")
    e = box.get("err")
    if isinstance(e, urllib.error.HTTPError) and e.code in (418, 429):
        secs = _arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now_ms)
        raise RateLimited(f"Binance {e.code} — stopped, cooldown {secs} s")
    if e is not None:
        raise e
    rows, headers = box["ok"]
    used = int((headers or {}).get("X-MBX-USED-WEIGHT-1M") or 0)
    step = DD_MS if interval == "1d" else M5_MS
    d = pd.DataFrame([(int(x[0]), float(x[4])) for x in rows if int(x[0]) + step <= now_ms], columns=["open_time", "c"])
    return d, used


def _append_cache(name, new):
    os.makedirs(DU_CACHE, exist_ok=True)
    p = os.path.join(DU_CACHE, name)
    d = pd.concat([_read_k(p), new]).drop_duplicates("open_time", keep="last").sort_values("open_time")
    tmp = f"{p}.{os.getpid()}.tmp"
    d.to_csv(tmp, index=False)
    os.replace(tmp, p)


def _daily(m5):
    return build_daily(_read_k(os.path.join(_CACHE, "k1d", "BTCUSDT.csv")), _read_k(os.path.join(DU_CACHE, "BTCUSDT_1d.csv")), m5)


def load_btc(need_ms, now_ms, fetch=True, budget_s=25.0):
    """→ (m5, d1, notes). Cache first. Requests (sequential, no retries): ≤ 4 × 5m pages of 1,500 bars (only while the 5m cache ends > 5 min
    before the newest counted fill) and ≤ 1 × 1d (a missing day−1 close) — a COLD/short daily series (< 60 days before the fill) instead refetches from
    2025-01-01 (limit 1000, a 2nd page only if budget allows). Every request: refused while a 418/429 cooldown is live, once the last
    seen used weight ≥ 900, or past the run's wall-clock budget (≤ 25 s total). A 418/429 aborts all further requests."""
    notes, st = [], dict(used=0, stop=None, deadline=time.monotonic() + budget_s)
    cd = _cooldown_until()
    if fetch and cd > now_ms:
        st["stop"] = f"rate-limit cooldown until {pd.Timestamp(cd, unit='ms'):%m-%d %H:%M} UTC"

    def req(interval, start, limit):
        if not fetch or st["stop"]:
            return None
        if st["used"] >= 900:
            st["stop"] = f"used weight {st['used']} ≥ 900"
            return None
        left = st["deadline"] - time.monotonic()
        if left < 1.0:
            st["stop"] = "wall-clock budget spent"
            return None
        try:
            d, st["used"] = _fetch(interval, start, limit, now_ms, timeout_s=min(10.0, left))
            return d
        except RateLimited as e:
            st["stop"] = str(e)
        except Exception as e:
            notes.append(f"{interval} fetch failed ({str(e)[:80]})")
            if isinstance(e, TimeoutError):
                st["stop"] = "wall-clock timeout"
        return None

    m5 = _load_m5()
    for _ in range(M5_PAGES):                          # paged (≤ 4 × 1,500 bars, weight 10 each) inside the same weight / time budget
        if need_ms is None or (len(m5) and m5["T"].max() >= need_ms - M5_MS):
            break
        new = req("5m", (m5.open_time.max() + M5_MS) if len(m5) else need_ms - 1500 * M5_MS, 1500)
        if new is None:
            break
        if len(new):
            _append_cache("BTCUSDT_5m.csv", new)
            m5 = _load_m5()
        notes.append(f"5m fetch +{len(new)} bars (weight used {st['used']})")
        if len(new) < 1500:
            break
    d1 = _daily(m5)
    if need_ms is not None:
        want = (need_ms // DD_MS) * DD_MS - DD_MS
        cold = not len(d1) or d1.index.min() > need_ms - D1_WARM_DAYS * DD_MS
        if cold:
            start = pd.Timestamp("2025-01-01").value // 10**6
            for _ in range(2):
                new = req("1d", start, 1000)
                if new is None or not len(new):
                    break
                _append_cache("BTCUSDT_1d.csv", new)
                notes.append(f"1d cold fetch +{len(new)} days (weight used {st['used']})")
                if len(new) < 1000:
                    break
                start = int(new.open_time.max()) + DD_MS
            d1 = _daily(m5)
        elif d1.index.max() < want:
            new = req("1d", d1.index.max() + DD_MS, 99)              # ≤ 99 → weight 1
            if new is not None:
                if len(new):
                    _append_cache("BTCUSDT_1d.csv", new)
                    d1 = _daily(m5)
                notes.append(f"1d fetch +{len(new)} days (weight used {st['used']})")
    if st["stop"]:
        notes.append(f"fetching stopped: {st['stop']}")
    return m5, d1, notes


def cache_lag(df, m5, d1):
    """counted fills whose stamps are readable but whose 1d leg is missing ONLY because the 5m / 1d cache ends before them."""
    if not len(df):
        return pd.Series([], dtype=bool)
    t = df.ts.values.astype("datetime64[ms]").astype("int64")
    m5_end = int(m5["T"].max()) if len(m5) else -1
    d1_end = int(d1.index.max()) if len(d1) else -1
    behind = (t > m5_end) | ((t // DD_MS) * DD_MS - DD_MS > d1_end)
    return pd.Series(df.d1s.isna().values & df.slope.notna().values & df.gap.notna().values & behind, index=df.index)


def score_d1(df, m5, d1):
    """per-fill BTC 1d EMA20 slope (NaN = unreadable; also NaN when the daily series starts < 60 days before the fill — EMA20 seeding)."""
    if not len(df):
        return np.array([], dtype=float)
    t = df.ts.values.astype("datetime64[ms]").astype("int64")
    T, c = price_at(m5, t)
    out = d1_slope(d1, T, c) if len(d1) else np.full(len(df), np.nan)
    if len(d1):
        out = np.where(T < d1.index.min() + D1_WARM_DAYS * DD_MS, np.nan, out)
    return np.where(np.isnan(c), np.nan, out)


def du_group(slope, gap, d1s):
    """zone / neg_not_zone / rest / unscored (any leg unreadable → unscored, never on either side)."""
    slope, gap, d1s = (np.asarray(x, dtype=float) for x in (slope, gap, d1s))
    uns = np.isnan(slope) | np.isnan(gap) | np.isnan(d1s)
    neg = slope <= SLOPE_MAX
    zone = neg & (gap <= DU_GAP_MAX) & (d1s > DU_D1_MIN)
    return np.where(uns, "unscored", np.where(zone, "zone", np.where(neg, "neg_not_zone", "rest")))


def gross_loss_share(pct, key):
    """study definition (study_negflank2d_followup.bar): the largest key's share of the summed LOSING-FILL loss (winners ignored)."""
    d = pd.DataFrame(dict(p=np.asarray(pct, dtype=float), k=np.asarray(key)))
    loss = d[d.p < 0]
    tl = -loss.p.sum()
    return float((-loss.groupby("k").p.sum()).max() / tl) if tl > 0 else 0.0


def du_verdict(c, be):
    """c = DataFrame(pct, day, pair) of the counted zone → (state, detail). RETIRE if mean ≥ 0; FILTER CANDIDATE only if WR < breakeven
    ∧ day-clustered P(mean < 0) ≥ 0.95 ∧ no day / pair ≥ 50 % of the gross loss; else KEEP OBSERVING."""
    n, nd = len(c), c.day.nunique() if len(c) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"N {n}/{N_MIN} · days {nd}/{DAYS_MIN}"
    m, wr = float(c.pct.mean()), 100.0 * float((c.pct > 0).mean())
    p = p_mean_neg(c.pct.values, c.day.values, DU_BOOT_N, DU_BOOT_SEED)
    sd, sp = gross_loss_share(c.pct.values, c.day.values), gross_loss_share(c.pct.values, c.pair.values)
    det = (f"N {n} · {nd} d · WR {wr:.0f} % vs breakeven {be:.1f} % · mean {m:+.3f} % · P(mean<0) {p:.2f} · top day {sd * 100:.0f} % / "
           f"top pair {sp * 100:.0f} % of the gross loss")
    if m >= 0:
        return "RETIRE", det
    ok = wr < be and p is not None and p >= 0.95 and sd < 0.5 and sp < 0.5
    return ("FILTER CANDIDATE (operator decides)" if ok else "KEEP OBSERVING"), det


def crossing_prefix(zone, n_min):
    """zone fills sorted by open time → the SMALLEST prefix with N ≥ n_min ∧ ≥ DAYS_MIN distinct days (None if not reached)."""
    z = zone.sort_values(["ts", "pair"], kind="stable")
    seen = set()
    for k, d in enumerate(z.day.values, 1):
        seen.add(d)
        if k >= n_min and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def freeze_hold(last, hold, why=None):
    """hold = [(Timestamp, label)] — fills that could still change the prefix (an OPEN position, a cache-lagging fill). Any opened
    ≤ the candidate prefix's last fill defers the freeze (→ True, reason appended to why); later ones cannot change it."""
    hit = [(t, lab) for t, lab in (hold or []) if pd.notna(t) and t <= last]
    if hit and why is not None:
        why.append(f"{len(hit)} fill(s) opened ≤ the crossing prefix's last fill ({last:%m-%d %H:%M}) not final yet: "
                   + ", ".join(f"{lab} {t:%m-%d %H:%M}" for t, lab in sorted(hit, key=lambda x: x[0])[:5]))
    return bool(hit)


def open_momentum_long_ts():
    """OPEN full-size momentum LONG positions in the NEWEST orders export (by mtime) → [(opened_at, 'OPEN <pair>')]. Same strategy
    filter as _orders(empty_is_momentum=True); *_PROBE out. No usable export → []."""
    fs = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")), key=os.path.getmtime)
    if not fs:
        return []
    try:
        d = pd.read_csv(fs[-1], low_memory=False, usecols=lambda c: c in COLS)
    except Exception:
        return []
    if not {"opened_at", "status", "direction"} <= set(d.columns):
        return []
    d = d[(d.status.astype(str).str.upper() == "OPEN") & (d.direction.astype(str) == "LONG")]
    if "entry_strategy" in d.columns:
        d = d[d.entry_strategy.fillna("MOMENTUM").astype(str).isin(["MOMENTUM", "", "nan"])]
    d = d[~d.get("cell_multiplier_source", pd.Series("", index=d.index)).astype(str).str.endswith("_PROBE")]
    t = pd.to_datetime(d.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    return [(a, f"OPEN {p}") for a, p in zip(t, d.pair.astype(str)) if pd.notna(a)]


def du_freeze(st, sleeve, now_iso, hold=None, why=None):
    """sleeve = the scored counted sleeve (column grp; zone + non-zone). Freezes the verdict of the CROSSING PREFIX (the first fills that
    reached N ≥ 15 ∧ ≥ 8 days, by open time) once into st['first'] — however late the run that sees it — judged with the breakeven of
    the scored sleeve fills opened ≤ the prefix's last fill (fallback 61.8 % until 30). st['reread'] = the first-30 prefix, judged the
    same way, only on a LATER call than the one that froze 'first'. Frozen entries are never recomputed. hold / why: freeze_hold.
    → (st, changed)."""
    key = "reread" if "first" in st else "first"
    if key in st:
        return st, False
    pre = crossing_prefix(sleeve[sleeve.grp == "zone"], DU_REREAD_N if key == "reread" else N_MIN)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if freeze_hold(last, hold, why):
        return st, False
    bs = sleeve[sleeve.ts <= last]
    be_live = breakeven_wr(bs.pct) if len(bs) >= BE_MIN_FILLS else None
    be = be_live if be_live is not None else DU_BE_REF
    state, det = du_verdict(pre, be)
    st[key] = dict(state=state, detail=det, be=round(be, 2), be_src="live" if be_live is not None else "fallback",
                   at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso, n=len(pre), days=int(pre.day.nunique()),
                   keys=[f"{a}|{b}" for a, b in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str))])
    return st, True


def du_load_state(path=None, now_ms=None):
    """→ (state, ok). ok False = freezing must be SKIPPED: the file is corrupt (moved to <path>.<UTC stamp>.bad now) or any .bad copy is
    newer than the state file (an operator restore is pending) — never re-freeze silently over a lost frozen verdict."""
    path = path or DU_STATE
    bads = glob.glob(f"{glob.escape(path)}.*bad")
    st_m = os.path.getmtime(path) if os.path.exists(path) else -1.0
    if os.path.exists(path):
        try:
            with open(path) as f:
                st = json.load(f)
            if not isinstance(st, dict):
                raise ValueError("state is not an object")
        except Exception:
            stamp = pd.Timestamp(now_ms or int(time.time() * 1000), unit="ms").strftime("%Y%m%dT%H%M%S")
            os.replace(path, f"{path}.{stamp}.bad")
            return {}, False
    else:
        st = {}
    if any(os.path.getmtime(b) > st_m for b in bads):
        return st, False
    return st, True


def du_save_state(st, path=None):
    path = path or DU_STATE
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, path)


def _du_row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean() * 100:.0f} % | {g.pct.mean():+.3f} % | "
            f"{g.usd.sum():+,.0f} | {g.pct.min():+.2f} % |")


def run_dailyup(now_ms=None, state_path=None, fetch=True, orders=None, open_ts=None):
    now_ms = now_ms or int(time.time() * 1000)
    o = (_orders(start=None, empty_is_momentum=True) if orders is None else orders).copy()
    cnt_mask = o.ts >= pd.Timestamp(DU_START)
    need = int(o[cnt_mask].ts.max().value // 10**6) if cnt_mask.any() else None
    m5, d1, notes = load_btc(need, now_ms, fetch=fetch)
    o["d1s"] = score_d1(o, m5, d1)
    o["grp"] = du_group(o.slope, o.gap, o.d1s)
    cnt, ref = o[cnt_mask], o[~cnt_mask]
    sc = cnt[cnt.grp != "unscored"]                    # the whole scored counted sleeve (zone + non-zone), as ML_B1H_NEGFLANK
    be_live = breakeven_wr(sc.pct) if len(sc) >= BE_MIN_FILLS else None
    be = be_live if be_live is not None else DU_BE_REF
    zone = cnt[cnt.grp == "zone"].sort_values(["ts", "pair"], kind="stable")
    lag = cache_lag(cnt, m5, d1)
    st, st_ok = du_load_state(state_path, now_ms)
    why = []
    if st_ok and not lag.any():
        hold = open_momentum_long_ts() if orders is None else list(open_ts or [])
        st, changed = du_freeze(st, sc, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), hold, why)
        if changed:
            du_save_state(st, state_path)
    live_state, live_det = du_verdict(zone[["pct", "day", "pair"]], be)
    L = ["## 🧭 NEG_DAILYUP_WEAKPAIR — NEGFLANK longs in a BTC daily uptrend on a weak-trend pair (DECISION_LOG 253, OBSERVE only)", "",
         f"Zone (frozen): BTC 1h slope ≤ {SLOPE_MAX:g} ∧ pair EMA13−EMA50 gap (`entry_pair_ema20_ema50_gap_pct`) ≤ {DU_GAP_MAX:g} ∧ BTC 1d EMA20 "
         f"slope > {DU_D1_MIN:g} (rebuilt from BTCUSDT daily klines, study convention). Counted: full-size momentum LONG fills opened ≥ {DU_START} "
         f"UTC; judged in DAY units. Reference: {DU_REF}.", "",
         "| Group (counted) | N | days | WR | avg % | Σ$ as-sized | worst |", "|---|---|---|---|---|---|---|",
         _du_row("**zone**", zone), _du_row("NEGFLANK, not zone", cnt[cnt.grp == "neg_not_zone"]),
         _du_row("rest of sleeve (1h > −0.05)", cnt[cnt.grp == "rest"]), _du_row("UNSCORED (a leg unreadable)", cnt[cnt.grp == "unscored"]), "",
         "| Reference only (before the counting start; as-traded, from exports, incl. B1 era — not comparable to the study's current-stack "
         "numbers) | N | days | WR | avg % | Σ$ as-sized | worst |", "|---|---|---|---|---|---|---|",
         _du_row("zone", ref[ref.grp == "zone"]), _du_row("NEGFLANK, not zone", ref[ref.grp == "neg_not_zone"]),
         _du_row("rest of sleeve", ref[ref.grp == "rest"]), _du_row("UNSCORED", ref[ref.grp == "unscored"]), ""]
    if len(zone):
        L += ["Counted zone fills: " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair} {r.pct:+.2f} % (1h {r.slope:+.3f} · gap {r.gap:+.3f} · "
                                                    f"1d {r.d1s:+.3f})" for r in zone.itertuples()), ""]
    be_txt = f"{be:.1f} % ({'live, ' + str(len(sc)) + ' scored counted sleeve fills' if be_live is not None else 'fallback until ' + str(BE_MIN_FILLS) + ' scored counted sleeve fills'})"
    if lag.any():
        L.append(f"⏸ freezing skipped this run: {int(lag.sum())} counted fill(s) UNSCORED only because the BTC 5m/1d cache lags "
                 f"(5m to {pd.Timestamp(int(m5['T'].max()) if len(m5) else 0, unit='ms'):%m-%d %H:%M}, 1d to "
                 f"{pd.Timestamp(int(d1.index.max()) if len(d1) else 0, unit='ms'):%m-%d}) — re-scored next run.")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not st_ok:
        L.append("⚠ frozen state corrupt — operator restore needed (reports/SCOUT_NEG_DAILYUP_WEAKPAIR.json.*.bad); freezing skipped this run.")
    if "first" in st:
        f0 = st["first"]
        L.append(f"**Frozen verdict (first crossing at fill {f0['at']}, frozen on the run of {f0.get('run_at', '?')}, N {f0['n']} · "
                 f"{f0['days']} d): {f0['state']}** ({f0['detail']})")
        if "reread" in st:
            r0 = st["reread"]
            L.append(f"**Frozen re-read (first {DU_REREAD_N}, at fill {r0['at']}): {r0['state']}** ({r0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    else:
        L.append(f"Bar (locked expectancy, frozen at the first crossing N ≥ {N_MIN} ∧ ≥ {DAYS_MIN} days): WR < sleeve breakeven {be_txt} ∧ "
                 f"day-clustered P(mean < 0) ≥ 0.95 ({DU_BOOT_N:,} resamples, seed {DU_BOOT_SEED}) ∧ no day / pair ≥ 50 % of the gross loss → "
                 f"FILTER CANDIDATE (operator decides); mean ≥ 0 → RETIRE; else KEEP OBSERVING (re-read at N ≥ {DU_REREAD_N}). "
                 f"Now: ⏳ {live_state} ({live_det}).")
    L += [DU_REVERT] + ([f"Data: {' · '.join(notes)}."] if notes else []) + [""]
    return L


def selftest_dailyup():
    global _NET_BLOCKED, DU_COOLDOWN, SHARED_COOLDOWN, _http_get, _CACHE, DU_CACHE
    _NET_BLOCKED = True
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    g = du_group([-0.05, -0.05, -0.05, -0.04, -0.3, np.nan, -0.3, -0.3],
                 [0.238, 0.238, 0.2381, 5.0, np.nan, 0.0, 0.0, 0.0],
                 [0.28111, 0.2811, 0.5, 0.5, 0.5, 0.5, np.nan, 0.6])
    chk(list(g) == ["zone", "neg_not_zone", "neg_not_zone", "rest", "unscored", "unscored", "unscored", "zone"],
        f"zone membership boundaries (≤ −0.05, ≤ 0.238, > 0.2811) + unscored on any NaN leg: {list(g)}")
    chk(_fetch_blocked(), "network blocked inside the selftest")
    # unscored never enters either side: run on synthetic orders with no BTC data → every fill unscored, zone empty
    fake = pd.DataFrame(dict(ts=pd.to_datetime(["2026-10-07 16:31:56", "2026-10-07 16:36:04"]), pair=["ARBUSDT", "UNIUSDT"],
                             pct=[-0.69, -0.70], usd=[-155.0, -154.0], slope=[-0.3, -0.3], gap=[0.02, np.nan], off30=[np.nan] * 2,
                             day=["2026-10-07"] * 2, _k=["a", "b"]))
    fake["d1s"] = [0.57, np.nan]
    fake["grp"] = du_group(fake.slope, fake.gap, fake.d1s)
    chk(list(fake.grp) == ["zone", "unscored"], "a missing gap → unscored, not 'not zone'")
    days = [f"2026-10-{i + 10:02d}" for i in range(8) for _ in range(2)]
    t0 = pd.Timestamp("2026-10-10 01:00")
    z = pd.DataFrame(dict(ts=[t0 + pd.Timedelta(days=i // 2, hours=i % 2) for i in range(16)], pct=[-0.3, 0.1] * 8, day=days,
                          pair=[f"P{i}" for i in range(16)], grp="zone"))
    st, ch = du_freeze({}, z.head(14), "t0")
    chk(not ch and "first" not in st, "N 14 → no freeze")
    late = pd.DataFrame(dict(ts=[pd.Timestamp("2026-11-01") + pd.Timedelta(hours=i) for i in range(20)], pct=5.0,
                             day=[f"2026-11-{1 + i // 3:02d}" for i in range(20)], pair="LATE", grp="zone"))
    z2 = pd.concat([late, z], ignore_index=True)                    # unsorted on purpose; 36 zone fills, run long after the crossing
    st, ch = du_freeze({}, z2, "t1")
    f0 = st["first"]
    chk(ch and f0["n"] == 15 and f0["state"] == "FILTER CANDIDATE (operator decides)" and f0["at"] == "2026-10-17 01:00 UTC"
        and "reread" not in st, f"a late run freezes the 15-fill crossing prefix only (never both keys in one call): {f0}")
    st2, ch2 = du_freeze(json.loads(json.dumps(st)), z2, "t2")
    chk(ch2 and st2["first"] == f0 and st2["reread"]["n"] == 30 and st2["reread"]["state"] == "RETIRE",
        "re-read = the first-30 prefix, on a later call; first unchanged")
    why = []
    sh, chh = du_freeze({}, z2, "th", [(pd.Timestamp("2026-10-12 00:00"), "OPEN X")], why)
    chk(not chh and sh == {} and why and "OPEN X" in why[0], "an OPEN momentum long opened ≤ the prefix end defers the freeze")
    sh, chh = du_freeze({}, z2, "th", [(pd.Timestamp("2026-10-30 00:00"), "OPEN Y")], [])
    chk(chh and sh["first"] == f0 | {"run_at": "th"}, "an OPEN position opened after the prefix end does not")
    st3, ch3 = du_freeze(st2, z2, "t3")
    chk(not ch3 and st3 == st2, "frozen entries never recomputed")
    nz = pd.DataFrame(dict(ts=[t0 - pd.Timedelta(hours=1 + i) for i in range(30)], pct=[1.0, -3.0] * 15, day="2026-10-09",
                           pair="NZ", grp="rest"))
    nz_late = nz.assign(ts=pd.Timestamp("2026-12-01"), pct=[0.1, -9.0] * 15)
    st4, _ = du_freeze({}, pd.concat([z, nz, nz_late], ignore_index=True), "t4")
    be_exp = breakeven_wr(pd.concat([z.head(15), nz]).pct)
    chk(st4["first"]["be_src"] == "live" and abs(st4["first"]["be"] - round(be_exp, 2)) < 1e-9,
        "breakeven from scored sleeve fills opened ≤ the prefix's last fill (later fills ignored)")
    chk(du_verdict(z.assign(pct=[0.3, -0.1] * 8), 61.8)[0] == "RETIRE", "mean ≥ 0 → RETIRE")
    chk(du_verdict(z, 40.0)[0] == "KEEP OBSERVING", "WR above breakeven → KEEP OBSERVING")
    conc = z.copy(); conc.loc[0, "pct"] = -10.0
    chk(du_verdict(conc, 61.8)[0] == "KEEP OBSERVING", "one day ≥ 50 % of the gross loss → KEEP OBSERVING")
    chk(abs(gross_loss_share([-1.0, -1.0, 5.0], ["a", "b", "b"]) - 0.5) < 1e-12 and gross_loss_share([1.0], ["a"]) == 0.0,
        "gross losing-fill share (winners ignored, study bar() definition)")
    lagdf = pd.DataFrame(dict(ts=pd.to_datetime(["2026-10-07 12:00", "2026-10-07 12:00", "2026-10-07 12:00"]), d1s=[np.nan, np.nan, 0.5],
                              slope=[-0.3, np.nan, -0.3], gap=[0.0, 0.0, 0.0]))
    m5l = pd.DataFrame(dict(open_time=[0], c=[1.0], T=[M5_MS]))
    chk(list(cache_lag(lagdf, m5l, pd.Series([1.0], index=[0]))) == [True, False, False],
        "cache lag = readable stamps + missing 1d leg + cache ends before the fill (a missing stamp is not lag)")
    p1 = p_mean_neg(z.pct.values, z.day.values, DU_BOOT_N, DU_BOOT_SEED)
    chk(p1 == p_mean_neg(z.pct.values, z.day.values, DU_BOOT_N, DU_BOOT_SEED), "bootstrap deterministic (seed 7)")
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "bad.csv")
        open(bad, "w").write("open_time,c\n1000,5.0\nxx,6.0\n2000,oops\n3000,7.5\n")
        rk = _read_k(bad)
        chk(list(rk.open_time) == [1000, 3000] and list(rk.c) == [5.0, 7.5], "bad cache rows dropped (numeric coerce)")
        tmp = os.path.join(td, "state.json")
        du_save_state(st3, tmp)
        chk(du_load_state(tmp) == (json.loads(json.dumps(st3)), True), "state round-trip (atomic write)")
        open(tmp, "w").write("{corrupt")
        s_, ok_ = du_load_state(tmp, 1791400000000)
        chk(s_ == {} and not ok_ and glob.glob(tmp + ".*.bad") and not os.path.exists(tmp), "corrupt state → timestamped .bad, freeze skipped")
        du_save_state({}, tmp)
        os.utime(glob.glob(tmp + ".*.bad")[0], (time.time() + 60, time.time() + 60))
        chk(du_load_state(tmp)[1] is False, "a .bad newer than the state keeps freezing skipped (operator restore pending)")
        # end-to-end, fetch off, HERMETIC: synthetic BTC 1d (2025-01-01 → 2026-10-07, +0.2 %/day → 1d EMA20 slope ≈ +0.6 > 0.2811) and
        # 5m (10-07 15:00 → 18:00) caches written to the tmp dir and injected; tmp state. No repo cache, no network.
        saved_c = (_CACHE, DU_CACHE)
        _CACHE, DU_CACHE = os.path.join(td, "cache"), os.path.join(td, "cache", "scout_dailyup")
        os.makedirs(os.path.join(_CACHE, "k1d")); os.makedirs(os.path.join(_CACHE, "k5m_full"))
        dd0 = pd.Timestamp("2025-01-01").value // 10**6
        nd_ = (pd.Timestamp("2026-10-07").value // 10**6 - dd0) // DD_MS + 1
        pd.DataFrame(dict(open_time=dd0 + np.arange(nd_) * DD_MS, c=50_000 * 1.002 ** np.arange(nd_))).to_csv(
            os.path.join(_CACHE, "k1d", "BTCUSDT.csv"), index=False)
        m0 = pd.Timestamp("2026-10-07 15:00").value // 10**6
        pd.DataFrame(dict(open_time=m0 + np.arange(36) * M5_MS, c=50_000 * 1.002 ** (nd_ - 1))).to_csv(
            os.path.join(_CACHE, "k5m_full", "BTCUSDT.csv"), index=False)
        d0 = pd.Timestamp("2026-10-07 16:31:56")
        e2e = pd.DataFrame(dict(ts=[d0, d0 + pd.Timedelta(minutes=5), d0 - pd.Timedelta(days=3)], pair=["ARBUSDT", "UNIUSDT", "OLDUSDT"],
                                pct=[-0.69, -0.70, 0.4], usd=[-155.0, -154.0, 50.0], slope=[-0.30, -0.30, -0.30], gap=[0.02, np.nan, 0.0],
                                off30=[np.nan] * 3, day=["2026-10-07", "2026-10-07", "2026-10-04"], _k=["a", "b", "c"]))
        sp = os.path.join(td, "e2e.json")
        out = "\n".join(run_dailyup(now_ms=1791490000000, state_path=sp, fetch=False, orders=e2e))
        chk("| **zone** | 1 |" in out and "| UNSCORED (a leg unreadable) | 1 |" in out and not os.path.exists(sp),
            "end-to-end: ARB in the zone, the gap-less fill UNSCORED, ref fill kept out of the count, no state written below the bar")
        empty = pd.DataFrame(columns=list(COLS) + ["ts", "pct", "slope", "off30", "day", "gap", "usd"])
        empty["ts"] = pd.to_datetime(empty.ts)
        out = "\n".join(run_dailyup(now_ms=1791490000000, state_path=sp, fetch=False, orders=empty))
        chk("COLLECTING (N 0/15" in out, "empty orders frame renders (collecting), never raises")
        # 429 → exactly one request, cooldown persisted (Retry-After honoured), next run makes ZERO requests
        saved = (DU_COOLDOWN, SHARED_COOLDOWN, _http_get)
        calls = []

        def _boom(url, timeout):
            calls.append(url)
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "120"}, None)
        try:
            DU_COOLDOWN, _http_get, _NET_BLOCKED = os.path.join(td, ".ratelimited_until"), _boom, False
            SHARED_COOLDOWN = os.path.join(td, ".binance_cooldown_until")
            far = 1791490000000 + 30 * DD_MS                   # past every cache → both a 5m and a 1d request would be wanted
            _, _, notes = load_btc(far, 1791490000000)
            chk(len(calls) == 1 and abs(_cooldown_until() - (1791490000000 + 120_000)) < 2 and any("429" in x for x in notes),
                f"429 aborts after ONE request + cooldown written ({len(calls)} calls, {notes})")
            _, _, notes = load_btc(far, 1791490000000 + 60_000)
            chk(len(calls) == 1 and any("cooldown" in x for x in notes), "cooldown live → zero requests")
            chk(os.path.exists(SHARED_COOLDOWN) and not os.path.exists(DU_COOLDOWN), "429 writes the SHARED cooldown file")
            os.remove(SHARED_COOLDOWN)
            for legacy in (DU_COOLDOWN, os.path.join(_CACHE, "scout_ml_trend", ".ratelimited_until")):
                os.makedirs(os.path.dirname(legacy), exist_ok=True)
                open(legacy, "w").write(str(1791490000000 + 999_999))
                _, _, notes = load_btc(far, 1791490000000)
                chk(len(calls) == 1 and any("cooldown" in x for x in notes), f"legacy cooldown {os.path.basename(os.path.dirname(legacy))} honoured")
                os.remove(legacy)

            def _slow(url, timeout):
                calls.append(url)
                time.sleep(3)
            _http_get = _slow
            t0 = time.monotonic()
            _, _, notes = load_btc(far, 1791490000000, budget_s=1.5)
            chk(len(calls) == 2 and time.monotonic() - t0 < 2.5 and any("wall-clock" in x for x in notes),
                "hung request cut by the wall-clock budget, no further request")
        finally:
            DU_COOLDOWN, SHARED_COOLDOWN, _http_get = saved
            _CACHE, DU_CACHE = saved_c
            _NET_BLOCKED = True
    m5w = pd.DataFrame(dict(open_time=[0], c=[1.0], T=[M5_MS]))
    d1w = pd.Series(np.arange(1.0, 11.0), index=np.arange(10) * DD_MS)
    chk(np.isnan(score_d1(pd.DataFrame(dict(ts=pd.to_datetime([M5_MS], unit="ms"))), m5w, d1w)[0]),
        "daily series < 60 days before the fill → UNSCORED (cold EMA seed)")
    # 1d slope reproduction vs the study's stored k_btc_1d_slope (cache only, no network)
    pk = os.path.join(_ROOT, "reports", "NEGFLANK_2D_features.pkl")
    if all(os.path.exists(x) for x in (pk, os.path.join(_CACHE, "k5m_full", "BTCUSDT.csv"), os.path.join(_CACHE, "k1d", "BTCUSDT.csv"))):
        F = pd.read_pickle(pk)
        F = F[F.k_btc_1d_slope.notna()]
        m5 = _load_m5()
        d1 = build_daily(_read_k(os.path.join(_CACHE, "k1d", "BTCUSDT.csv")), _read_k(os.path.join(DU_CACHE, "BTCUSDT_1d.csv")), m5)
        pick = pd.concat([F[F.src == "master"].tail(5), F[F.src == "yr5"].iloc[:: max(1, len(F[F.src == "yr5"]) // 7)]])
        T, c = price_at(m5, pick.t_ms.values)
        got = d1_slope(d1, T, c)
        ref = pick.k_btc_1d_slope.values
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-9)
        good = int((rel <= 1e-6).sum())
        chk(good >= 5 and good == len(pick), f"1d slope reproduces the study on {good}/{len(pick)} stored fills (max rel {np.nanmax(rel):.2e})")
        print(f"  1d slope reproduced on {good}/{len(pick)} stored fills, max rel err {np.nanmax(rel):.1e} · master tail: "
              + ", ".join(f"{a} {b:%m-%d} {v:+.4f}" for a, b, v in zip(pick.pair.values[:5], pick.o[:5], got[:5])))
    else:
        print("  1d slope reproduction SKIPPED (reports/NEGFLANK_2D_features.pkl or the k5m_full / k1d BTCUSDT cache missing — not a failure)")
    _NET_BLOCKED = False
    print(f"selftest NEG_DAILYUP_WEAKPAIR OK ({ok} checks)")


def _fetch_blocked():
    try:
        _fetch("1d", 0, 1, 0)
    except RuntimeError as e:
        return "blocked" in str(e)
    return False


# ─────────────────────────── self-test ───────────────────────────
def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    chk(abs(breakeven_wr(pd.Series([1.0, 1.0, -1.0])) - 50.0) < 1e-9, "breakeven = |L| / (W + |L|)")
    chk(breakeven_wr(pd.Series([1.0, 2.0])) is None, "no losers → None")
    days = [f"d{i}" for i in range(10) for _ in range(2)]
    neg = np.array([-0.2, 0.1] * 10)
    chk(p_mean_neg(neg, days) > 0.95, "consistently negative days → P ≈ 1")
    chk(p_mean_neg(-neg, days) < 0.05, "positive → P ≈ 0")
    chk(abs(top_loss_share([-1.0, -1.0, 0.5], ["a", "b", "b"]) - 1 / 1.5) < 1e-9, "top loss share on net-negative keys")
    c = pd.DataFrame(dict(pct=neg, day=days, pair=[f"P{i}" for i in range(20)]))
    chk(decide(c, 70.0)[0] == "passes", "WR 50 < 70, negative, spread → passes")
    chk(decide(c, 40.0)[0] == "fails", "WR above breakeven → fails")
    chk(decide(c.head(14), 70.0)[0] == "collecting", "N 14 → collecting")
    conc = c.copy(); conc.loc[0, "pct"] = -10.0
    chk(decide(conc, 70.0)[0] == "fails", "one day ≥ 50 % of the loss → fails")
    print(f"selftest OK ({ok} checks)")
    selftest_dailyup()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
