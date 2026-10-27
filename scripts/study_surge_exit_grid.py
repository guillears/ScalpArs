#!/usr/bin/env python3
"""⚡🎯 SURGE_LONG EXIT GRID (DECISION_LOG 397, operator 2026-10-27: "clearly the exit mechanism is not good!" → "yes run the SURGE exit
study now"; after BTW / EVAA 10-27 — peaks +0.90 / +1.32 %, both stopped −1.20 %).

  venv/bin/python scripts/study_surge_exit_grid.py --prereg   # writes reports/SURGE_EXIT_GRID_PREREG.txt (refuses overwrite)
  venv/bin/python scripts/study_surge_exit_grid.py            # walk → reports/SURGE_EXIT_GRID_fills.csv + _real.csv + _tables.md

Two cohorts, every exit walked on the SAME price path per fill:
  ① BACKTEST — SURGE_LONG CLOSED fills of reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv with EMA13−EMA50 gap ≥ 0 (today's DL 287 filter),
    3 seeds pooled; aggTrade tick paths (scripts/surge_bearrun_review.sec_path — the Oct-15 trail-cap walker, 556 / 556 parity); 1m-only
    fills are a flagged sensitivity line, never the primary read.
  ② REAL — every closed bot SURGE_LONG fill in the orders exports / master (paper + live, cohort floor 2026-09-30 15:30): path = Binance's
    own aggTrades (the public API for the last 2 days, else the public daily archive data.binance.vision), cached under
    reports/backtest_cache/agg_surge_real/; every print by default (--per-second = last trade per second). The TODAY walk must
    reproduce the booked pct (±0.15, both before slip — paper books the trigger print, e.g. EVAA −1.2027 walk vs −1.2022 booked) or the
    fill is shown but flagged ⚠.
Cache / public-endpoint only; never the bot API.
"""
import datetime
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import surge_bearrun_review as SR   # noqa: E402  (sec_path / FEE)

REP = os.path.join(ROOT, "reports")
FILLS = os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
PREREG = os.path.join(REP, "SURGE_EXIT_GRID_PREREG.txt")
OUT_BT = os.path.join(REP, "SURGE_EXIT_GRID_fills.csv")
OUT_REAL = os.path.join(REP, "SURGE_EXIT_GRID_real.csv")
OUT_MD = os.path.join(REP, "SURGE_EXIT_GRID_tables.md")
CACHE26 = SR.CACHE
CACHE25 = os.path.join(REP, "backtest_cache_2025")
MIN, DAY = 60_000, 86_400_000
FEE = SR.FEE                         # 0.09 % round trip, in every walk
SLIP_MKT, SLIP_ENTRY = -0.17, -0.03  # live: market exits −0.17, SURGE entries +0.03 → a resting TP books its line (0 exit slip)
COHORT_FLOOR = "2026-09-30T15:30:00"
SLOT_X = 8 * 0.975 / 4               # today's SURGE notional ÷ account (full slot margin × 8×) — the DCR scale
B = 5000

TH = SR.TH
ARM = float(TH.get("bullrun_be_arm_pct", 1.0))
BASE = float(TH.get("bullrun_base_sl_pct", -0.7))
SLM = float(TH.get("sl_atr_multiplier", 1.5)); SLF = float(TH.get("sl_atr_widen_floor_pct", -1.2))
HOLD = int(TH.get("surge_max_hold_minutes", 240))


def init_sl(a):
    w = -(a * SLM)
    if SLF < 0:
        w = max(w, SLF)
    return min(BASE, w)


def ladder_floor(pk):
    rungs = [(float(a), float(b)) for a, b in (x.split(":") for x in str(TH.get("bullrun_ladder", "")).split(",") if ":" in x)]
    best = None
    for trig, fl in rungs:
        if pk >= trig:
            best = fl
    return best


# ── the cells: name → (stop line from the running peak, take-profit fn(minutes) or None, hold minutes, tp_is_resting)
def cells(a):
    sl = init_sl(a)
    trail = lambda m, arm=ARM: (lambda pk: max(sl, pk - m * a) if pk >= arm else sl)
    def lock(pk, arm=ARM, lk=0.2, m=1.0, ladder=True):
        if pk < arm:
            return sl
        line = max(lk, pk - m * a)
        if ladder:
            lf = ladder_floor(pk)
            if lf is not None:
                line = max(line, lf)
        return line
    C = {
        "TODAY": (trail(1.0), None, HOLD, False),
        "LOCK_LADDER": (lock, None, HOLD, False),                                        # DL 203's alternative (+0.2 lock + ladder at +1.0)
        "LOCK_ONLY": (lambda pk: lock(pk, ladder=False), None, HOLD, False),
        "ARM05_BE": (lambda pk: (max(0.0, pk - a) if pk >= 0.5 else sl), None, HOLD, False),   # at +0.5: line ≥ break-even, then 1×ATR
        "TRAIL_0.5ATR": (trail(0.5), None, HOLD, False),
        "TRAIL_2ATR": (trail(2.0), None, HOLD, False),
        "HOLD_60": (trail(1.0), None, 60, False),
        "HOLD_120": (trail(1.0), None, 120, False),
        "WILLY_LIKE": ((lambda pk: -4.0), (lambda mins: np.where(mins >= 30, 0.25, 1.25)), 90, True),
    }
    for tp in (1, 2, 3, 5):
        for s in (1.2, 2.0):
            C[f"TP{tp}_SL{s:g}"] = ((lambda pk, s=s: -s), (lambda mins, tp=tp: np.full(len(mins), float(tp))), HOLD, True)
    return C


CELL_NAMES = list(cells(2.0).keys())
VARIANTS = [c for c in CELL_NAMES if c != "TODAY"]


def walk(ts, pnl, t0, stop, tpf, hold):
    """→ (gross-of-slip net %, reason, minutes, peak). First print with net ≤ line (filled at that print) or ≥ TP (booked AT the TP
    line — a resting order); else the last print before the hold limit."""
    m = ts <= t0 + hold * MIN
    ts, pnl = ts[m], pnl[m]
    if not len(pnl):
        return None
    peak = np.maximum.accumulate(np.maximum(pnl, 0.0))
    line = SR._lines(stop, peak)
    hit = pnl <= line
    tpl = None
    if tpf is not None:
        tpl = tpf((ts - t0) / MIN)
        hit |= pnl >= tpl
    if not hit.any():
        return float(pnl[-1]), "TIME", (ts[-1] - t0) / MIN, float(peak[-1])
    j = int(np.argmax(hit))
    if tpl is not None and pnl[j] >= tpl[j]:
        return float(tpl[j]), "TP", (ts[j] - t0) / MIN, float(peak[j])
    return float(pnl[j]), ("STOP" if line[j] < 0 else "TRAIL"), (ts[j] - t0) / MIN, float(peak[j])


def price_all(ts, px, e, t0, a):
    pnl = (px / e - 1) * 100 - FEE
    rec = {}
    for name, (stop, tpf, hold, resting) in cells(a).items():
        o = walk(ts, pnl, t0, stop, tpf, hold)
        if o is None:
            return None
        slip = SLIP_ENTRY + (0.0 if (resting and o[1] == "TP") else SLIP_MKT)
        rec[name] = o[0] + slip; rec[name + "_why"] = o[1]; rec[name + "_min"] = o[2]
        if name == "TODAY":
            rec["TODAY_peak"] = o[3]; rec["TODAY_raw"] = o[0]
    return rec


_K = {}
def k1m_path(pair, t0, t1, cache):
    key = (cache, pair)
    if key not in _K:
        if len(_K) > 30:
            _K.clear()
        fr = [pd.read_csv(os.path.join(cache, x, f"{pair}.csv"), usecols=["open_time", "o", "h", "l", "c"])
              for x in ("k1m", "k1m_ondemand", "k1m_spike") if os.path.exists(os.path.join(cache, x, f"{pair}.csv"))]
        _K[key] = pd.concat(fr).drop_duplicates("open_time").sort_values("open_time").to_numpy() if fr else None
    k = _K[key]
    if k is None:
        return None
    k = k[(k[:, 0] >= (t0 // MIN) * MIN) & (k[:, 0] <= t1)]
    if len(k) < 3 or k[0, 0] > t0 + 2 * MIN:
        return None
    ts, px = [], []
    for ot, o, h, l, c in k:
        for j, p in enumerate((o, l, h, c) if c >= o else (o, h, l, c)):
            t = ot + j * 15_000
            if t > t0:
                ts.append(t); px.append(p)
    return np.array(ts, dtype=np.int64), np.array(px, dtype=float)


_LAST = [0.0]


AGG_REAL = os.path.join(SR.CACHE, "agg_surge_real")
PER_SECOND = "--per-second" in sys.argv   # real fills: every print (default; 4 / 5 reproduce the booked exit) · --per-second = last trade per second (3 / 5)


def fetch_agg_public(pair, t0, t1):
    """Binance PUBLIC futures aggTrades on [t0 − 60 s, t1] (weight 20 / call, ≤ 1 h windows, paginated by fromId), paced ≥ 0.6 s,
    sleeps when the used weight passes 900, stops on 418 / 429. Cached as reports/backtest_cache/agg_surge_real/<pair>_<t0>.csv."""
    os.makedirs(AGG_REAL, exist_ok=True)
    f = os.path.join(AGG_REAL, f"{pair}_{t0}.csv")
    full = t1 >= t0 + HOLD * MIN          # only a COMPLETE window is cached (a fill younger than its hold would cache a truncated path)
    if os.path.exists(f):
        d = pd.read_csv(f)
        return d.T_ms.to_numpy(np.int64), d.p.to_numpy(float)
    if int(time.time() * 1000) - t0 > 40 * 3_600_000:   # the API serves only the last 2 days → Binance's public daily archive
        rows = []
        for day in sorted({pd.to_datetime(x, unit="ms").strftime("%Y-%m-%d") for x in (t0 - 60_000, t1)}):
            z = os.path.join(AGG_REAL, f"{pair}-aggTrades-{day}.zip")
            if not os.path.exists(z):
                url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{pair}/{pair}-aggTrades-{day}.zip"
                try:
                    with urllib.request.urlopen(url, timeout=120) as r, open(z + ".part", "wb") as fh:
                        fh.write(r.read())
                    os.replace(z + ".part", z)
                except Exception as ex:   # not published yet (404) / network → this fill is skipped and listed, never the whole section
                    print(f"archive {pair} {day} unavailable ({str(ex)[:60]}) — fill skipped")
                    return None
            d = pd.read_csv(z, header=None, low_memory=False)
            if not str(d.iloc[0, 0]).isdigit():
                d = d.iloc[1:]
            d = d.iloc[:, [5, 1]].astype(float)
            d = d[(d.iloc[:, 0] >= t0 - 60_000) & (d.iloc[:, 0] <= t1)]
            rows += list(zip(d.iloc[:, 0].astype(np.int64), d.iloc[:, 1]))
        d = pd.DataFrame(rows, columns=["T_ms", "p"]).sort_values("T_ms", kind="stable")
        if full:
            d.to_csv(f, index=False)
        return d.T_ms.to_numpy(np.int64), d.p.to_numpy(float)
    rows, s = [], t0 - 60_000
    while s < t1:
        e = min(s + 3_600_000 - 1, t1)
        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={pair}&startTime={s}&endTime={e}&limit=1000"
        last_t = None
        while True:
            wait = 1.0 - (time.time() - _LAST[0])   # weight 20 / call → ≤ 1,200 weight / min by construction
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
                    data = json.loads(r.read())
            except urllib.error.HTTPError as ex:
                if ex.code in (418, 429):
                    sys.exit(f"Binance {ex.code} — STOP (request budget)")
                raise
            _LAST[0] = time.time()
            if used > 900:
                time.sleep(61)
            rows += [(int(x["a"]), int(x["T"]), float(x["p"])) for x in data]
            if len(data) < 1000:
                break
            last_t = int(data[-1]["T"])
            url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={pair}&fromId={int(data[-1]['a']) + 1}&limit=1000"
            if last_t > e:
                break
        s = e + 1
    d = pd.DataFrame(rows, columns=["a", "T_ms", "p"]).drop_duplicates("a").sort_values("T_ms")[["T_ms", "p"]]
    d = d[d.T_ms <= t1]
    if full:
        d.to_csv(f, index=False)
    return d.T_ms.to_numpy(np.int64), d.p.to_numpy(float)


def check_engine_parity():
    import services.trading_engine as TE
    for a in (1.5, 1.98, 2.69, 4.0):
        f = cells(a)["TODAY"][0]; g = cells(a)["LOCK_LADDER"][0]
        for pk in (0.0, 0.5, 0.99, 1.0, 1.32, 1.7, 3.54, 4.2, 8.0, 12.5):
            assert abs(TE._bullrun_exit_for(float("inf"), pk, a, trail_mult_override=1.0, no_lock=True)[2] - f(pk)) < 1e-9, ("TODAY", a, pk)
            assert abs(TE._bullrun_exit_for(float("inf"), pk, a, trail_mult_override=1.0, no_lock=False)[2] - g(pk)) < 1e-9, ("LOCK", a, pk)
    print("engine line parity OK (TODAY = no_lock, LOCK_LADDER = the lock path)")


def check_config():
    """the grid's TODAY = the live SURGE exit (DL 203): no lock, 1 × ATR trail, 240 min — checked at run time, never at import."""
    if not (TH.get("surge_long_exit_no_lock", True) and abs(float(TH.get("surge_long_trail_atr_mult", 1.0)) - 1.0) < 1e-9 and HOLD == 240):
        sys.exit("trading_config SURGE exit changed since the pre-registration — re-register before pricing")


# ── ① backtest
def run_backtest():
    d = pd.read_csv(FILLS, low_memory=False)
    s = d[(d.entry_strategy == "SURGE_LONG") & (d.status == "CLOSED")].copy()
    s = s[pd.to_numeric(s.entry_pair_ema20_ema50_gap_pct, errors="coerce") >= 0].copy()
    s["t0"] = pd.to_datetime(s.opened_at.astype(str).str[:23], format="mixed").astype("datetime64[ms]").astype("int64")
    rows = []
    try:
        _walk_backtest(s, rows)
    finally:
        SR.CACHE = CACHE26
    W = pd.DataFrame(rows)
    return _finish_backtest(W)


def _walk_backtest(s, rows):
    for r in s.itertuples():
        a = float(r.entry_atr_pct); e = float(r.entry_price); t0 = int(r.t0); t1 = t0 + HOLD * MIN + 5000
        cache = CACHE25 if str(r.year) == "2025" else CACHE26
        SR.CACHE = cache
        sp = SR.sec_path(r.pair, t0, t1); src = "tick"
        if sp is not None:
            ts, px, _ = sp
        else:
            kp = k1m_path(r.pair, t0, t1, cache); src = "1m"
            if kp is None:
                continue
            ts, px = kp
        rec = price_all(ts, px, e, t0, a)
        if rec is None:
            continue
        rec.update(src=src, pair=r.pair, opened_at=str(r.opened_at)[:19], seed=r.seed, regime=r.regime, year=str(r.year), atr=a,
                   replay_pct=float(r.pct_new))
        rows.append(rec)


def _finish_backtest(W):
    T = W[W.src == "tick"]
    par = ((T.TODAY_raw - T.replay_pct).abs() <= 0.10).mean()
    print(f"backtest: {len(W)} walked ({len(T)} tick) · TODAY parity vs replay (tick, ±0.10) {par * 100:.1f} %")
    if not (par >= 0.85):   # NaN (no tick fill) fails too
        sys.exit("PARITY < 85 % (or no tick fill) — STOP")
    W.to_csv(OUT_BT, index=False)
    return W, par


# ── ② real fills
def real_fills():
    fs = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_*.csv")), key=os.path.getmtime)
    fr = []
    pool = os.path.join(REP, "MASTER_POOL_stacked.csv")
    for i, f in enumerate([pool] + fs):
        try:
            x = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "entry_strategy" not in x:
            continue
        x = x[x.entry_strategy.astype(str) == "SURGE_LONG"]
        if len(x):
            fr.append(x.assign(rr=i))
    if not fr:
        return pd.DataFrame()
    a = pd.concat(fr, ignore_index=True)
    a["kk"] = a.opened_at.astype(str).str.replace(" ", "T").str[:19]
    a = a.sort_values("rr").drop_duplicates(["kk", "pair"], keep="last")
    a = a[(a.kk >= COHORT_FLOOR) & (a.status.astype(str) == "CLOSED")]
    out = []
    for r in a.itertuples():
        t0 = int(pd.Timestamp(r.kk).value // 1_000_000); t1 = t0 + HOLD * MIN + 5000
        e, atr = float(r.entry_price), float(r.entry_atr_pct)
        SR.CACHE = CACHE26
        # every real fill on Binance's OWN aggTrades (operator 10-27: real fills only — the cached tick / 1m paths missed AGT / BTW)
        ag = fetch_agg_public(r.pair, t0, min(t1, int(time.time() * 1000)))
        if ag is None:
            continue
        at, ap = ag
        m = at > t0
        ts, px, src = at[m], ap[m], "aggTrades"
        if PER_SECOND:   # the live exit reads ~1 price / second (AGT 10-18: a 2-s wick to +5.0 the bot never saw; it booked +7.53 at 94.1 s)
            sec = ts // 1000
            last = np.r_[sec[1:] != sec[:-1], True]
            ts, px, src = sec[last] * 1000 + 999, px[last], "aggTrades/1s"
        if not len(ts):
            continue
        rec = price_all(ts, px, e, t0, atr)
        if rec is None:
            continue
        booked = float(r.pnl_percentage)
        rec.update(src=src, pair=r.pair, opened_at=r.kk, atr=atr, booked_pct=booked, booked_reason=r.close_reason,
                   booked_peak=float(getattr(r, "peak_pnl", np.nan)),
                   parity=abs(rec["TODAY_raw"] - booked) <= 0.15)
        out.append(rec)
    R = pd.DataFrame(out)
    R.to_csv(OUT_REAL, index=False)
    return R


# ── stats
def day_boot(day, x, rng):
    g = pd.DataFrame(dict(day=day, x=x)).groupby("day").x.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(B, len(g)))
    return s[idx].sum(1) / c[idx].sum(1)


def dcr(day_pct, days_all):
    """DCR-cal at today's size: r_d = Σ fill % × SLOT_X / 100 per OPEN day (SURGE holds ≤ 4 h — close day ≈ open day), 0 on no-fill days, over every calendar day of the span."""
    r = day_pct.reindex(days_all, fill_value=0.0) * SLOT_X / 100.0
    return (np.prod(1 + r.to_numpy()) ** (1 / len(r)) - 1) * 100 if len(r) else np.nan


def hhmm(m):
    return "–" if m is None or not np.isfinite(m) else f"{int(m // 60):02d}:{int(round(m % 60)):02d}" if round(m % 60) < 60 else f"{int(m // 60) + 1:02d}:00"


def report(W, par, R):
    rng = np.random.default_rng(20261027)
    T = W[W.src == "tick"].copy()
    T["day"] = T.opened_at.str[:10]
    T["period"] = np.where(T.year.astype(str).str[:4] == "2025", "2025", np.where(T.opened_at < "2026-07-01", "2026-H1", "2026-H2"))
    L = ["# SURGE_LONG exit grid — " + datetime.date.today().isoformat(), "",
         f"Pre-registration: reports/SURGE_EXIT_GRID_PREREG.txt (frozen before pricing). Backtest tick sample {len(T)} fills (3 seeds pooled; "
         f"TODAY parity vs the replay {par * 100:.1f} % within ±0.10). All % net of 0.09 fees, entry slip −0.03, market exits −0.17, a resting TP "
         f"books its line. DCR-cal = daily compound return at today's size (notional ≈ {SLOT_X:.2f}× the account per fill), every calendar day, "
         "per seed then averaged.", "",
         "## ① Backtest (two-year replay, today's entry stack)", "",
         "| Exit | N | WR | Avg % | Δ vs TODAY | P(Δ>0) day-boot | Δ 2025 | Δ 26-H1 | Δ 26-H2 | Avg hold | Median hold | DCR-cal %/day |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    span = {}
    for sd, g in T.groupby("seed"):
        span[sd] = pd.date_range(g.day.min(), g.day.max(), freq="D").strftime("%Y-%m-%d")
    res = {}
    for c in CELL_NAMES:
        x = T[c]; dlt = x - T.TODAY
        bt = day_boot(T.day.to_numpy(), dlt.to_numpy(), rng) if c != "TODAY" else None
        pp = float((bt > 0).mean()) if bt is not None else None
        per = {p: float((T[c] - T.TODAY)[T.period == p].mean()) for p in ("2025", "2026-H1", "2026-H2")}
        dc = np.mean([dcr(g.groupby("day")[c].sum(), span[sd]) for sd, g in T.groupby("seed")])
        res[c] = dict(avg=float(x.mean()), d=float(dlt.mean()), p=pp, per=per, dcr=dc)
        L.append(f"| {c} | {len(x)} | {(x > 0).mean() * 100:.0f} % | {x.mean():+.3f} | {'–' if c == 'TODAY' else f'{dlt.mean():+.3f}'} | "
                 f"{'–' if pp is None else f'{pp:.3f}'} | " + " | ".join('–' if c == 'TODAY' else f"{per[p]:+.3f}" for p in per)
                 + f" | {hhmm(T[c + '_min'].mean())} | {hhmm(T[c + '_min'].median())} | {dc:+.3f} |")
    # gate
    nvar = len(VARIANTS)
    L += ["", f"**Frozen gate** (multiplicity {nvar}): CANDIDATE iff Δ > 0 ∧ P(Δ>0) ≥ {1 - 0.05 / nvar:.4f} (Bonferroni 0.05 / {nvar}) ∧ Δ > 0 in "
          "EVERY period (2025 · 26-H1 · 26-H2) ∧ DCR-cal above TODAY's. Then the real fills must not contradict it (a caveat when they do, "
          "MASTER-over-backtest only when N ≥ 15).", ""]
    passed = [c for c in VARIANTS if res[c]["d"] > 0 and res[c]["p"] >= 1 - 0.05 / nvar and all(v > 0 for v in res[c]["per"].values())
              and res[c]["dcr"] > res["TODAY"]["dcr"]]
    L.append("Gate result: " + (", ".join(f"✅ {c}" for c in passed) if passed else "❌ no variant passes"))
    # armed-only / peak buckets for TODAY vs the protective cells
    L += ["", "### Where the protective exits act (backtest, TODAY's peak bucket)", "",
          "| TODAY peak | N | TODAY avg | LOCK_ONLY | ARM05_BE | TRAIL_0.5ATR | TP2_SL1.2 | TP3_SL1.2 |", "|---|---|---|---|---|---|---|---|"]
    T["pkb"] = pd.cut(T.TODAY_peak, [-1, 0.5, 1.0, 1.5, 3.0, 1e9], labels=["< 0.5", "0.5–1.0", "1.0–1.5", "1.5–3.0", "≥ 3.0"])
    for b, g in T.groupby("pkb", observed=True):
        L.append(f"| {b} | {len(g)} | {g.TODAY.mean():+.3f} | " + " | ".join(f"{g[c].mean():+.3f}" for c in ("LOCK_ONLY", "ARM05_BE", "TRAIL_0.5ATR", "TP2_SL1.2", "TP3_SL1.2")) + " |")
    M = W[W.src == "1m"]
    if len(M):
        L += ["", f"1m-fallback sensitivity ({len(M)} fills, not in the gate): " + " · ".join(f"{c} {M[c].mean():+.3f}" for c in ("TODAY", "LOCK_ONLY", "ARM05_BE", "TP2_SL1.2", "TP3_SL1.2"))]
    # real
    L += ["", "## ② Real SURGE_LONG fills (paper + live, cohort floor 09-30 15:30)", ""]
    if R is None or not len(R):
        L.append("No closed real fill.")
    else:
        L.append(f"{len(R)} closed fills · TODAY walk reproduces the booked pct (±0.15) on {int(R.parity.sum())} / {len(R)} (path: "
                 + ", ".join(f"{k} {v}" for k, v in R.src.value_counts().items()) + ").")
        L += ["", "| Fill | ATR | Booked | " + " | ".join(CELL_NAMES) + " |", "|" + "---|" * (len(CELL_NAMES) + 3)]
        for r in R.sort_values("opened_at").itertuples():
            L.append(f"| {r.pair.replace('USDT', '')} {r.opened_at[5:16].replace('T', ' ')}{'' if r.parity else ' ⚠'} | {r.atr:.2f} | {r.booked_pct:+.2f} | "
                     + " | ".join(f"{R.loc[r.Index, c]:+.2f}" for c in CELL_NAMES) + " |")
        L.append("| **mean** | | " + f"{R.booked_pct.mean():+.2f} | " + " | ".join(f"**{R[c].mean():+.2f}**" for c in CELL_NAMES) + " |")
        L.append("| avg hold | | | " + " | ".join(hhmm(R[c + '_min'].mean()) for c in CELL_NAMES) + " |")
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return res, passed


PREREG_TXT = """SURGE_LONG EXIT GRID — PRE-REGISTRATION (DECISION_LOG 397)
written {ts} UTC, before any variant was priced. Operator 2026-10-27: "clearly the exit mechanism is not good!" after BTW / EVAA
(peaks +0.90 / +1.32 %, both stopped −1.20 %; EVAA armed at +1.0 but the 1 × ATR 2.69 % line sat at −1.37 % < the −1.2 % stop).

CONTROL TODAY (DL 203): initial stop min(−0.7, max(−1.5 × ATR, −1.2)); arm at peak ≥ +1.0 net; then line = max(stop, peak − 1 × entry
ATR); no lock / ladder; 240 min hold. Engine parity asserted (_bullrun_exit_for no_lock=True; LOCK_LADDER = no_lock=False).

VARIANTS ({n}): {cells}
  LOCK_LADDER  DL 203's alternative: at +1.0 line = max(+0.2, peak − 1×ATR, bullrun_ladder floor)
  LOCK_ONLY    the same without the ladder
  ARM05_BE     at peak ≥ +0.5 line = max(0.0, peak − 1×ATR) (break-even floor), before it the initial stop
  TRAIL_0.5ATR / TRAIL_2ATR   today's arm, trail 0.5 / 2 × ATR
  HOLD_60 / HOLD_120          today's exit, max hold 60 / 120 min
  WILLY_LIKE   the WILLY V2b shape with the fixed floor target (SURGE has no spike-high stamp): resting TP +1.25, from minute 30 TP +0.25,
               stop −4, 90 min cap
  TPx_SLy      resting TP +x % (1 / 2 / 3 / 5) with a fixed stop −y % (1.2 / 2), 240 min
Costs: 0.09 fees in every walk; entry slip −0.03; market exits −0.17; a resting TP books its line (0 exit slip).
COHORT: ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv SURGE_LONG CLOSED with EMA13−EMA50 gap ≥ 0, 3 seeds pooled, tick-walked (primary);
1m-walked = sensitivity only. Parity gate: TODAY walk vs the replay pct within ±0.10 on ≥ 85 % of tick fills, else STOP.
GATE (frozen): CANDIDATE iff mean Δ vs TODAY > 0 ∧ day-clustered bootstrap P(Δ > 0) ≥ 1 − 0.05 / {n} ∧ Δ > 0 in each of 2025 / 2026-H1 /
2026-H2 ∧ DCR-cal (today's size, every calendar day, per seed averaged) above TODAY's. Real fills (paper + live, floor 09-30 15:30)
reported per fill; with N < 15 they are a caveat, not a veto. A CANDIDATE is an operator call with the 30–50 % haircut and its own
revert gate — never automatic. No re-fitting of levels after this file.
"""


if __name__ == "__main__":
    if "--prereg" in sys.argv:
        if os.path.exists(PREREG):
            sys.exit(f"REFUSED: {PREREG} exists (frozen)")
        open(PREREG, "w").write(PREREG_TXT.format(ts=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), n=len(VARIANTS),
                                                  cells=", ".join(VARIANTS)))
        print("written", PREREG); sys.exit(0)
    if not os.path.exists(PREREG):
        sys.exit("pre-register first (--prereg)")
    check_config()
    check_engine_parity()
    if "--reuse" in sys.argv and os.path.exists(OUT_BT):   # the walk is deterministic: reuse the written fills
        W = pd.read_csv(OUT_BT); T = W[W.src == "tick"]; par = ((T.TODAY_raw - T.replay_pct).abs() <= 0.10).mean()
        if not (par >= 0.85):
            sys.exit("PARITY < 85 % (or no tick fill) in the reused walk — STOP")
    else:
        W, par = run_backtest()
    R = real_fills()
    report(W, par, R)
