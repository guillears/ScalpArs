#!/usr/bin/env python3
"""FLATBASE_BASE_TREND_2026-11-26 — does a DOWNTRENDING 14-day base (or a range near the 0.444 FLAT limit) make Flatbase shorts worse / better?
Research only (operator 2026-11-26, after SLX / LAYER / BIGTIME opened on falling bases). No bot / config change.

FEATURES (known at the base bar = the 5m close 30 min before the spike bar's close — services/flatbase.py BASE_LAG_MS; the window is the
same 4,032 5m closes the live FLAT test reads, bars whose CLOSE lies in (base − 14 d, base]):
  ret14  14-day base return % (last / first close − 1)          slope  log-linear OLS slope of the closes, %/day
  ret7   7-day return % (last / the close 7 d before − 1)        pos    position of the last close in the 14-day close range (0 = low, 1 = high)
  dist_hi  % the last close sits below the 14-day max close      flat14 the FLAT range % (max / min − 1; parity-checked vs the rebuild / stamps)
DATA:
  backtest = OOS_REAL_FLATBASE rebuild (reports/OOS_REAL_FLATBASE_data/wl_features.csv): today's exit X0 (TP −5 maker / stop +10 / 72 h,
             real funding, model slippage), today's armed entry stack = HIGHER_HIGH (hh False on every row) + NEAR_BASE (dist_base > 2.09) +
             FUNDING_ABOVE_BASE (funding ≤ 0.005 % per settlement) → 545 trades (= the FB_EXIT_RERUN / FB_TP_GRID 11-21 cohort).
             The DL 607 "stop counts funding paid" is NOT re-walked (caveat; it moves a stop only when funding is paid).
  real     = master kept fills on today's stack (reports/KEPT_LIST_FLATBASE_2026-11-21.csv: 17 MASTER = stack_keep + the DL 601 funding
             cut; P&L = MASTER_POOL_stacked stack_pct) + the current-batch kept closed fills (KERNEL / MMT / PIPPIN; PHAROS is funding-
             blocked on today's stack) from the orders export (argv[1], default the 11-26 13:02 export in ~/Downloads). No probes.
  open     = SLX / LAYER / BIGTIME (opened 11-26, spike times from the decisions export) — zone membership only, no P&L.
KLINES: local 5m caches (reports/backtest_cache*/k5m_full + the Flatbase study caches); anything missing → public Binance fapi klines
  (≤ 1,000 weight / min, the used-weight header read on every call, stop on 418 / 429), cached in reports/FLATBASE_BASE_TREND_2026-11-26_data/.
  Real-fill paths use 1m bars (entry minute skipped — it holds pre-fill prints); backtest paths use 5m (entry = the bar's open).
Usage: venv/bin/python -I scripts/FLATBASE_BASE_TREND_2026-11-26.py [orders.csv] [decisions.csv]
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import path_labels as PL  # noqa: E402
import review_math as RM  # noqa: E402

TAG = "FLATBASE_BASE_TREND_2026-11-26"
OUT = os.path.join(ROOT, "reports", TAG + "_data")
os.makedirs(os.path.join(OUT, "k5m"), exist_ok=True)
os.makedirs(os.path.join(OUT, "k1m"), exist_ok=True)
EXPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-26_13-02-05.csv")
DECISIONS = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/Downloads/scalpars_decisions_paper_2026-11-26_08-51-02.csv")
BAR, MIN, HR, DAY = 300_000, 60_000, 3_600_000, 86_400_000
BASE_LAG = 6 * BAR
X_SIZE = 0.61                         # one-size notional fraction of equity (the DL 601 Flatbase study's ruler)
TP, SL, HOLD_H = 5.0, 10.0, 72
DESIGN_END = "2025-07-01"
SRC = [os.path.join(ROOT, "reports", "backtest_cache_2025", "k5m_full"), os.path.join(ROOT, "reports", "backtest_cache", "k5m_full"),
       os.path.join(ROOT, "reports", "FRENZY_FLATBASE_SHORT_2026-10-22_data", "k5m"),
       os.path.join(ROOT, "reports", "POST_FRENZY_PATHS_2026-11-17_data", "k5m"),
       os.path.join(ROOT, "reports", "OOS_REAL_FLATBASE_data", "k5m"), os.path.join(OUT, "k5m")]
OPEN3 = [("SLXUSDT", None), ("LAYERUSDT", None), ("BIGTIMEUSDT", None)]
VARS = ["ret14", "slope", "ret7", "pos", "dist_hi", "flat14"]
VLAB = {"ret14": "14-d base return %", "slope": "base slope %/day", "ret7": "7-d return %", "pos": "position in 14-d range",
        "dist_hi": "% below 14-d high", "flat14": "FLAT range %"}
# a-priori (not fitted) "sign" cuts: falling = < 0 for the return / slope variables; lower half of the range; ≥ 40 % = near the 44.4 limit;
# dist_hi has no natural sign → its DESIGN median is used at this granularity too
SIGN_CUT = {"ret14": 0.0, "slope": 0.0, "ret7": 0.0, "pos": 0.5, "flat14": 40.0}
MD = []


def P(s=""):
    print(s)
    MD.append(s)


def md(df, fl=2):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:+.{fl}f}" if abs(v) < 1000 else f"{v:,.0f}")
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


# ───────────────────────────── klines ─────────────────────────────
class Budget:
    used = 0
    calls = 0


def fetch(pair, interval, start, end):
    """public fapi klines [start, end) → DataFrame open_time,o,h,l,c,vol,qvol. ≤ 1,000 weight/min; 418 / 429 → stop the run."""
    import requests
    step = {"5m": BAR, "1m": MIN}[interval]
    rows, t = [], int(start)
    while t < end:
        if Budget.used >= 1000:
            time.sleep(62 - (time.time() % 60))
            Budget.used = 0
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params=dict(symbol=pair, interval=interval, startTime=t, endTime=int(end) - 1, limit=1000), timeout=20)
        Budget.calls += 1
        if r.status_code in (418, 429):
            raise SystemExit(f"Binance {r.status_code} — stopped (retry-after {r.headers.get('Retry-After')})")
        w = r.headers.get("X-MBX-USED-WEIGHT-1M") or r.headers.get("x-mbx-used-weight-1m")
        if w is None:
            raise SystemExit("no used-weight header — stopped")
        Budget.used = int(w)
        if r.status_code != 200:
            break
        d = r.json()
        if not d:
            break
        rows += d
        t = int(d[-1][0]) + step
        if len(d) < 1000:
            break
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
    x = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[7])] for r in rows],
                     columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
    return x[(x.open_time >= start) & (x.open_time < end)]


def cache_merge(path, x):
    if os.path.exists(path):
        x = pd.concat([pd.read_csv(path), x])
    x = x.drop_duplicates("open_time", keep="last").sort_values("open_time")
    x.to_csv(path, index=False)
    return x


_LOCAL = {}


def load_local(pair):
    if pair in _LOCAL:
        return _LOCAL[pair]
    parts = []
    for i, d in enumerate(SRC):
        f = os.path.join(d, pair + ".csv")
        if os.path.exists(f):
            x = pd.read_csv(f, usecols=lambda c: c in ("open_time", "o", "h", "l", "c"))
            if i == 0:
                x = x[x.open_time < pd.Timestamp("2026-01-10").value // 10**6]
            parts.append(x)
    d = (pd.concat(parts).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
         if parts else pd.DataFrame(columns=["open_time", "o", "h", "l", "c"]))
    if len(_LOCAL) > 40:
        _LOCAL.clear()
    _LOCAL[pair] = d
    return d


def bars5(pair, start, end, now_ms):
    """5m bars [start, end) — local caches first, the missing span fetched (once) and cached."""
    d = load_local(pair)
    w = d[(d.open_time >= start) & (d.open_time < end)]
    need = (min(end, now_ms) - start) // BAR
    if len(w) >= 0.97 * need:
        return w
    x = fetch(pair, "5m", start, min(end, now_ms))
    if len(x):
        cache_merge(os.path.join(OUT, "k5m", pair + ".csv"), x)
        _LOCAL.pop(pair, None)
        d = load_local(pair)
        w = d[(d.open_time >= start) & (d.open_time < end)]
    return w


def bars1(pair, start, end, now_ms):
    f = os.path.join(OUT, "k1m", pair + ".csv")
    d = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["open_time", "o", "h", "l", "c"])
    w = d[(d.open_time >= start) & (d.open_time < end)]
    need = (min(end, now_ms) - start) // MIN
    if len(w) < 0.97 * need:
        x = fetch(pair, "1m", start, min(end, now_ms))
        if len(x):
            d = cache_merge(f, x)
            w = d[(d.open_time >= start) & (d.open_time < end)]
    return w


# ───────────────────────────── features ─────────────────────────────
def base_feats(b, bc):
    """b = 5m bars; bc = base bar close ms → the six features (NaN when < 90 % of the 4,032 closes are present)."""
    w = b[(b.open_time >= bc - 14 * DAY) & (b.open_time <= bc - BAR)]
    out = dict(cov=len(w) / 4032.0)
    if len(w) < 0.9 * 4032 or w.open_time.iloc[-1] != bc - BAR:
        return {**out, **{v: np.nan for v in VARS}}
    c = w.c.values.astype(float)
    t = (w.open_time.values - w.open_time.values[0]) / DAY
    lo, hi, last = c.min(), c.max(), c[-1]
    k7 = w[w.open_time <= bc - 7 * DAY - BAR]
    sl = np.polyfit(t, np.log(c), 1)[0]
    out.update(ret14=(last / c[0] - 1) * 100, slope=(np.exp(sl) - 1) * 100,
               ret7=(last / k7.c.values[-1] - 1) * 100 if len(k7) else np.nan,
               pos=(last - lo) / (hi - lo) if hi > lo else np.nan, dist_hi=(1 - last / hi) * 100, flat14=(hi / lo - 1) * 100)
    return out


def walk_exit(b, E, t0):
    """today's exit on bars from t0 (SHORT, gross price levels; same-bar TP + stop → stop) → exit ms."""
    w = b[(b.open_time >= t0) & (b.open_time < t0 + HOLD_H * HR)]
    tp = np.flatnonzero(w.l.values <= E * (1 - TP / 100))
    st = np.flatnonzero(w.h.values >= E * (1 + SL / 100))
    i_tp = tp[0] if len(tp) else 10**9
    i_st = st[0] if len(st) else 10**9
    if i_tp == 10**9 and i_st == 10**9:
        return t0 + HOLD_H * HR, "TIME"
    return (int(w.open_time.values[min(i_tp, i_st)]) + (BAR if len(w) < 2 else int(w.open_time.values[1] - w.open_time.values[0])),
            "STOP" if i_st <= i_tp else "TP")


def labels(b, E, te, tc, booked, stopped, step):
    """path facts on bars that start AT/after the fill (bar 0 = post-fill) → PL.finish group. Y = today's stop gross (path_labels)."""
    Y = PL.stop_y("FLATBASE_SHORT")[0]
    w = b[(b.open_time >= te) & (b.open_time < max(te + 24 * HR, tc + 2 * HR))]
    if not len(w):
        return dict(label="PENDING", group="PENDING", reach=np.nan)
    t = w.open_time.values
    fav, adv = (1 - w.l.values / E) * 100, (1 - w.h.values / E) * 100
    hz = t < te + 24 * HR
    lab, i = PL.first_passage(fav[hz], adv[hz] - 1e-9, PL.ENTRY_X, Y)
    if stopped and tc < te + 24 * HR and (lab == "FLAT" or (lab in ("WORKED", "AMBIG") and i is not None and t[i] >= tc)):
        lab = "NEVER"
    dn = np.flatnonzero(adv[hz] <= -Y + 1e-9)
    e = dn[0] if len(dn) else (np.searchsorted(t, tc, side="right") - 1 if stopped and tc < te + 24 * HR else hz.sum() - 1)
    reach = float(np.max(fav[:e + 1])) if e >= 0 else 0.0
    m = t < tc + 45 * MIN
    best = float(np.max(fav[m])) if m.any() else None
    m2 = (t >= tc) & (t < tc + 120 * MIN)
    rec = bool(m2.any() and np.max(fav[m2]) >= PL.ENTRY_X)
    g = PL.finish(dict(label=lab, best=best, recovered=rec), booked)["group"]
    return dict(label=lab, group=g, reach=reach)


def mtm_path(b, E, te, x_ms=None, booked=None):
    """hourly short MTM % (gross, close-based) 0..72 h; with x_ms the path is frozen at the booked % after the exit."""
    w = b[(b.open_time >= te) & (b.open_time < te + HOLD_H * HR)]
    out = np.full(HOLD_H + 1, np.nan)
    out[0] = 0.0
    if not len(w):
        return out
    s = pd.Series((1 - w.c.values / E) * 100, index=(w.open_time.values + (w.open_time.values[1] - w.open_time.values[0] if len(w) > 1 else BAR) - te) / HR)
    for h in range(1, HOLD_H + 1):
        v = s[s.index <= h]
        if len(v) and v.index[-1] > h - 1:
            out[h] = v.iloc[-1]
        if x_ms is not None and te + h * HR >= x_ms:
            out[h] = booked
    return out


# ───────────────────────────── data build ─────────────────────────────
def build():
    now = int(time.time() * 1000)
    fb = os.path.join(OUT, "bt.pkl")
    if os.path.exists(fb):
        BT = pd.read_pickle(fb)
    else:
        X = pd.read_csv(os.path.join(ROOT, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
        X = X[X.X0_why.isin(["TP", "STOP", "TIME"]) & (X.dist_base > 2.09) & (X.funding_rate_pct <= 0.005) & (~X.hh.astype(bool))].copy()
        rows = []
        for pair, g in X.sort_values("pair").groupby("pair"):
            for _, r in g.iterrows():
                bc = int(r.spike_ts) - BASE_LAG
                te = int(r.cf_entry_t)
                b = bars5(pair, bc - 14 * DAY - BAR, te + (HOLD_H + 3) * HR, now)
                f = base_feats(b, bc)
                E = float(r.cf_entry_px)
                lab = labels(b, E, te, int(r.X0_x), float(r.X0_net), r.X0_why == "STOP", BAR)
                rows.append(dict(pair=pair, te=te, spike_ts=int(r.spike_ts), net=float(r.X0_net), why=r.X0_why, flat14_ref=float(r.flat14) * 100,
                                 **f, **lab, path_raw=mtm_path(b, E, te), path_exit=mtm_path(b, E, te, int(r.X0_x), float(r.X0_net))))
            print(f"  bt {len(rows)}/{len(X)} calls {Budget.calls}", end="\r")
        BT = pd.DataFrame(rows)
        BT.to_pickle(fb)
    fr = os.path.join(OUT, "real.pkl")
    if os.path.exists(fr):
        RL = pd.read_pickle(fr)
    else:
        KL = pd.read_csv(os.path.join(ROOT, "reports", "KEPT_LIST_FLATBASE_2026-11-21.csv"))
        m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
        m = m[m.entry_strategy == "FLATBASE_SHORT"]
        o = pd.read_csv(EXPORT, low_memory=False)
        o = o[(o.entry_strategy == "FLATBASE_SHORT") & (o.status == "CLOSED")]
        if "is_probe" in o:
            o = o[o.is_probe.astype(str).str.lower() != "true"]
        rows = []
        for _, k in KL.iterrows():
            src = m if k.block == "MASTER" else o
            r = src[(src.pair == k.pair) & (src.opened_at.str[:19] == k.k[:19]) & (src.direction == "SHORT")]
            if len(r) != 1:
                print("  ! real fill not found once:", k.pair, k.k, len(r))
                continue
            r = r.iloc[0]
            pct = float(r.stack_pct) if k.block == "MASTER" else float(r.pnl_percentage)
            spike = int(pd.Timestamp(r.entry_frenzy_spike_at).value // 10**6)
            bc = spike - BASE_LAG
            te = int(pd.Timestamp(r.opened_at).value // 10**6)
            b5 = bars5(r.pair, bc - 14 * DAY - BAR, bc + BAR, now)
            f = base_feats(b5, bc)
            E = float(r.entry_price)
            b1 = bars1(r.pair, (te // MIN) * MIN, te + (HOLD_H + 3) * HR, now)
            t1 = (te // MIN + 1) * MIN                       # the entry minute skipped (pre-fill prints)
            xms, why = walk_exit(b1, E, t1)
            lab = labels(b1, E, t1, xms, pct, pct < 0 and why == "STOP", MIN)
            rows.append(dict(pair=r.pair, te=te, opened_at=r.opened_at, block=k.block, net=pct, flat14_ref=float(r.entry_flatbase_flat14_pct),
                             closed_day=pd.Timestamp(xms, unit="ms").strftime("%Y-%m-%d"), exit_why=why,
                             **f, **lab, path_raw=mtm_path(b1, E, t1), path_exit=mtm_path(b1, E, t1, xms, pct)))
        RL = pd.DataFrame(rows)
        RL.to_pickle(fr)
    fo = os.path.join(OUT, "open3.pkl")
    if os.path.exists(fo):
        OP = pd.read_pickle(fo)
    else:
        d = pd.read_csv(DECISIONS, low_memory=False)
        d = d[(d.e == "FLATBASE") & d.pair.isin([p for p, _ in OPEN3]) & d.fb_spike_at.notna()]
        rows = []
        for p, _ in OPEN3:
            g = d[d.pair == p]
            if not len(g):
                continue
            g = g.iloc[-1]
            bc = int(pd.Timestamp(g.fb_spike_at).value // 10**6) - BASE_LAG
            b5 = bars5(p, bc - 14 * DAY - BAR, bc + BAR, now)
            rows.append(dict(pair=p, spike_at=g.fb_spike_at, flat14_ref=float(g.fb_flat14_pct), **base_feats(b5, bc)))
        OP = pd.DataFrame(rows)
        OP.to_pickle(fo)
    print(f"\n  klines calls {Budget.calls}")
    return BT, RL, OP


# ───────────────────────────── stats ─────────────────────────────
def boot_mean(v, win, B=4000, seed=11):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float); win = np.asarray(win)
    u, inv = np.unique(win, return_inverse=True)
    sums = np.bincount(inv, weights=v); ns = np.bincount(inv)
    ix = rng.integers(0, len(u), (B, len(u)))
    means = sums[ix].sum(1) / ns[ix].sum(1)
    return float((means < 0).mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_diff(v, z, win, B=4000, seed=5):
    """window-clustered bootstrap of mean(zone) − mean(rest) → P(diff < 0), 95 % CI."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float); z = np.asarray(z, bool); win = np.asarray(win)
    u, inv = np.unique(win, return_inverse=True)
    s1 = np.bincount(inv, weights=v * z, minlength=len(u)); n1 = np.bincount(inv, weights=z.astype(float), minlength=len(u))
    s0 = np.bincount(inv, weights=v * ~z, minlength=len(u)); n0 = np.bincount(inv, weights=(~z).astype(float), minlength=len(u))
    ix = rng.integers(0, len(u), (B, len(u)))
    d = s1[ix].sum(1) / np.maximum(n1[ix].sum(1), 1) - s0[ix].sum(1) / np.maximum(n0[ix].sum(1), 1)
    return float((d < 0).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def breakeven_wr(p):
    p = np.asarray(p, float)
    w, l_ = p[p > 0], -p[p <= 0]
    return l_.mean() / (w.mean() + l_.mean()) * 100 if len(w) and len(l_) else np.nan


def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b) + 1e-12)


def daily_path(pct, day, d0, d1, x=X_SIZE):
    s = pd.Series(np.asarray(pct, float) * x / 100).groupby(pd.to_datetime(pd.Series(np.asarray(day))).values).sum()
    r = s.reindex(pd.date_range(d0, d1, freq="D"), fill_value=0.0).values
    eq = np.cumprod(1 + r)
    dcr = (eq[-1] ** (1 / len(r)) - 1) * 100
    dd = ((eq / np.maximum.accumulate(eq)) - 1).min() * 100
    return dcr, dd, r.min() * 100


def zone_row(name, g, z):
    v = g.net.values
    a, b = v[z], v[~z]
    return dict(zone=name, N=int(z.sum()), WR=(a > 0).mean() * 100 if len(a) else np.nan, avg=a.mean() if len(a) else np.nan, sum=a.sum(),
                rest_N=int((~z).sum()), rest_WR=(b > 0).mean() * 100 if len(b) else np.nan, rest_avg=b.mean() if len(b) else np.nan)


def cuts_for(D, v):
    x = D[v].dropna()
    sign = SIGN_CUT.get(v, float(x.median()))
    return dict(sign=sign, median=float(x.median()), t1=float(x.quantile(1 / 3)), t2=float(x.quantile(2 / 3)))


def main():
    BT, RL, OP = build()
    BT["date"] = pd.to_datetime(BT.te, unit="ms").dt.strftime("%Y-%m-%d")
    BT["blk"] = np.where(BT.date < DESIGN_END, "DESIGN", "HOLDOUT")
    dt = pd.to_datetime(BT.te, unit="ms")
    BT["half"] = dt.dt.year.astype(str) + np.where(dt.dt.month < 7, "H1", "H2")
    BT["win"] = (BT.te // (4 * HR)).astype(str)
    RL["win"] = (RL.te // (4 * HR)).astype(str)
    RL["date"] = RL.closed_day
    ok = BT[VARS].notna().all(1)
    P(f"# {TAG} — does a DOWNTRENDING base (or a range near the 44.4 % FLAT limit) change Flatbase short results?\n")
    P("Research only — no bot / config change. Summaries only. Script: scripts/FLATBASE_BASE_TREND_2026-11-26.py\n")
    P("## Data + parity")
    par = (BT.flat14 - BT.flat14_ref).abs()
    P(f"- Backtest: {len(BT)} trades on today's stack (HIGHER_HIGH + NEAR_BASE + FUNDING_ABOVE_BASE; exit TP −5 / stop +10 / 72 h, real "
      f"funding; DL 607 funding-in-stop not re-walked). Base features scored on {int(ok.sum())} ({ok.mean() * 100:.0f} %); FLAT range parity vs the "
      f"rebuild: |Δ| ≤ 0.05 pts on {(par <= 0.05).sum()} / {par.notna().sum()} (median |Δ| {par.median():.3f}).")
    pr = (RL.flat14 - RL.flat14_ref).abs()
    P(f"- Real: {len(RL)} kept fills (master {int((RL.block == 'MASTER').sum())} + batch {int((RL.block == 'BATCH').sum())}); parity vs the "
      f"live stamp entry_flatbase_flat14_pct: |Δ| ≤ 0.05 pts on {(pr <= 0.05).sum()} / {pr.notna().sum()}.")
    po = (OP.flat14 - OP.flat14_ref).abs() if len(OP) else pd.Series(dtype=float)
    P(f"- Open: {', '.join(OP.pair)} (parity vs the ledger fb_flat14_pct |Δ| max {po.max():.3f}).")
    P(f"- DESIGN < {DESIGN_END} ({int((BT.blk == 'DESIGN').sum())}) · HOLDOUT ({int((BT.blk == 'HOLDOUT').sum())}). Window = one 4-h decision close. "
      f"One-size ruler X = {X_SIZE} × equity per fill.\n")
    BE = breakeven_wr(BT.net)
    P(f"Sleeve (backtest): {len(BT)} · WR {(BT.net > 0).mean() * 100:.0f} % · avg {BT.net.mean():+.2f} % · breakeven WR {BE:.0f} %. "
      f"Real: {len(RL)} · WR {(RL.net > 0).mean() * 100:.0f} % · avg {RL.net.mean():+.2f} %.\n")
    P("## Feature distribution (backtest median · IQR | real median)")
    P(md(pd.DataFrame([dict(var=VLAB[v], bt_med=BT[v].median(), bt_q1=BT[v].quantile(.25), bt_q3=BT[v].quantile(.75),
                            real_med=RL[v].median(), corr_ret14=BT[[v, 'ret14']].corr().iloc[0, 1]) for v in VARS])))
    D = BT[ok & (BT.blk == "DESIGN")]
    CUTS = {v: cuts_for(D, v) for v in VARS}
    B2 = BT[ok].copy()
    # ── 1. per variable × granularity: zone tables (backtest blocks + real) ──
    tests, rows = [], []
    for v in VARS:
        c = CUTS[v]
        specs = [("sign", f"{v} < {c['sign']:.2f}", lambda s, c=c: s < c["sign"]),
                 ("median", f"{v} < DESIGN median {c['median']:.2f}", lambda s, c=c: s < c["median"]),
                 ("T1", f"{v} ≤ {c['t1']:.2f} (DESIGN T1)", lambda s, c=c: s <= c["t1"]),
                 ("T3", f"{v} > {c['t2']:.2f} (DESIGN T3)", lambda s, c=c: s > c["t2"])]
        for gran, name, fz in specs:
            if gran == "sign" and v == "dist_hi":
                continue
            z = fz(B2[v]).values
            r = dict(var=VLAB[v], gran=gran, zone=name)
            for blk in ("DESIGN", "HOLDOUT"):
                m = (B2.blk == blk).values
                a, b = B2.net.values[m & z], B2.net.values[m & ~z]
                r[f"{blk[0]}_N"] = int(len(a)); r[f"{blk[0]}_avg"] = a.mean() if len(a) else np.nan; r[f"{blk[0]}_rest"] = b.mean() if len(b) else np.nan
            halves = sorted(B2.half.unique())
            sgn = [np.sign(B2.net[(B2.half == h).values & z].mean() - B2.net[(B2.half == h).values & ~z].mean())
                   for h in halves if (((B2.half == h).values & z).sum() >= 5 and ((B2.half == h).values & ~z).sum() >= 5)]
            r["halves_zone_worse"] = f"{sum(s < 0 for s in sgn)}/{len(sgn)}"
            pdf, lo, hi = boot_diff(B2.net.values, z, B2.win.values)
            r["Δ_all"] = B2.net.values[z].mean() - B2.net.values[~z].mean(); r["P(Δ<0)"] = pdf
            zr = fz(RL[v]).values & RL[v].notna().values
            r["real_N"] = int(zr.sum()); r["real_WR"] = (RL.net[zr] > 0).mean() * 100 if zr.any() else np.nan
            r["real_avg"] = RL.net[zr].mean() if zr.any() else np.nan; r["real_rest"] = RL.net[~zr & RL[v].notna().values].mean()
            rows.append(r)
            tests.append((v, gran, fz, welch_t(B2.net.values[z], B2.net.values[~z])))
    T = pd.DataFrame(rows)
    # ── shuffle null (family-wise over every test above) ──
    rng = np.random.default_rng(42)
    net = B2.net.values.copy()
    masks = [fz(B2[v]).values for v, g, fz, t in tests]
    mx = []
    for _ in range(2000):
        p = rng.permutation(net)
        mx.append(max(abs(welch_t(p[m], p[~m])) for m in masks))
    mx = np.array(mx)
    T["t"] = [t for *_, t in tests]
    T["p_fw"] = [(mx >= abs(t)).mean() for *_, t in tests]
    T.to_csv(os.path.join(ROOT, "reports", TAG + ".csv"), index=False)
    P("\n## 1 · Zone vs rest, every variable × granularity (backtest DESIGN / HOLDOUT; real = master + batch, same frozen cut)")
    P("Δ_all = zone avg − rest avg (backtest, all); P(Δ<0) = window-clustered bootstrap; p_fw = family-wise shuffle null over all "
      f"{len(tests)} tests (2,000 permutations, max |t|); halves = half-years where the zone is worse / half-years with ≥ 5 per side.")
    P(md(T[["var", "gran", "zone", "D_N", "D_avg", "D_rest", "H_N", "H_avg", "H_rest", "halves_zone_worse", "Δ_all", "P(Δ<0)", "t", "p_fw",
            "real_N", "real_WR", "real_avg", "real_rest"]]))
    P(f"\nNull: 95th pct of max |t| = {np.percentile(mx, 95):.2f}; best observed |t| = {T.t.abs().max():.2f} ({T.loc[T.t.abs().idxmax(), 'zone']}).")
    # ── 2. the operator's question: falling base (ret14 < 0, slope < 0) and near-limit range (flat14 ≥ 40) — full read ──
    P("\n## 2 · The operator's zones — full read vs the locked Pattern-C expectancy bar + 'no breakeven bot' (one size)")
    rows2, rows3 = [], []
    focus = [("falling base: ret14 < 0", lambda g: g.ret14 < 0), ("falling base: slope < 0", lambda g: g.slope < 0),
             ("falling 7 d: ret7 < 0", lambda g: g.ret7 < 0), ("near limit: flat14 ≥ 40", lambda g: g.flat14 >= 40),
             ("low in range: pos < 0.5", lambda g: g.pos < 0.5)]
    for name, fz in focus:
        for lab, g in (("backtest ALL", B2), ("backtest DESIGN", B2[B2.blk == "DESIGN"]), ("backtest HOLDOUT", B2[B2.blk == "HOLDOUT"]),
                       ("master (real)", RL[RL.block == "MASTER"]), ("batch (real)", RL[RL.block == "BATCH"])):
            z = fz(g).fillna(False).values
            a = g.net.values[z]
            pneg, lo, hi = boot_mean(a, g.win.values[z]) if z.sum() >= 2 else (np.nan, np.nan, np.nan)
            loss_p = pd.Series(a[a < 0]).groupby(g.pair.values[z][a < 0]).sum()
            loss_w = pd.Series(a[a < 0]).groupby(g.win.values[z][a < 0]).sum()
            d0, d1 = g.date.min(), g.date.max()
            dcr0, dd0, wd0 = daily_path(g.net, g.date, d0, d1)
            dcr1, dd1, wd1 = daily_path(g.net.values[~z], g.date.values[~z], d0, d1)
            rows2.append(dict(zone=name, block=lab, N=int(z.sum()), windows=int(pd.Series(g.win.values[z]).nunique()),
                              WR=(a > 0).mean() * 100 if len(a) else np.nan, BE_WR=breakeven_wr(g.net) if len(g) > 3 else np.nan,
                              avg=a.mean() if len(a) else np.nan, rest_avg=g.net.values[~z].mean() if (~z).any() else np.nan,
                              P_mean_lt0=pneg, ci_lo=lo, ci_hi=hi,
                              top_pair_loss=(loss_p.min() / loss_p.sum() * 100) if len(loss_p) else np.nan,
                              top_win_loss=(loss_w.min() / loss_w.sum() * 100) if len(loss_w) else np.nan))
            rows3.append(dict(zone=name, block=lab, DCR_before=dcr0, DCR_block=dcr1, maxDD_before=dd0, maxDD_block=dd1,
                              worst_day_before=wd0, worst_day_block=wd1))
    T2 = pd.DataFrame(rows2)
    T2["bar_pass"] = [("PASS" if (r.N >= 15 and r.windows >= 8 and r.WR < r.BE_WR and r.P_mean_lt0 >= 0.95
                                  and r.top_pair_loss < 50 and r.top_win_loss < 50) else "fail") if r.block.startswith("backtest") else "—"
                      for r in T2.itertuples()]
    P(md(T2))
    P("\nBlock the zone → one-size DCR %/day · max drawdown % · worst day % (before → after):")
    P(md(pd.DataFrame(rows3)))
    T2.to_csv(os.path.join(OUT, "zones_bar.csv"), index=False)
    pd.DataFrame(rows3).to_csv(os.path.join(OUT, "zones_dcr.csv"), index=False)
    # concentration + LOO on the real fills for ret14 < 0
    zr = (RL.ret14 < 0).values
    P("\nReal fills, block ret14 < 0 — leave-one-out of the Δ (SUMMED % pts over the blocked fills; the excluded fill's number is its contribution to Δ): "
      + RM.loo_text(list(np.where(zr, -RL.net.values, 0.0)), labels=list(RL.pair.str.replace("USDT", "") + " " + RL.opened_at.str[5:10]), unit="% pts"))
    # ── 3. path labels per zone (entry × exit) ──
    P("\n## 3 · Entry × exit — path labels per zone (count · Σ net %)")
    GR = ["CAPTURED", "LEFT_ON_TABLE", "NEVER_WORKED", "STOPPED_RECOVERED", "GAVE_BACK", "DRIFT_LOSS", "AMBIG", "PENDING"]
    rows4 = []
    for name, fz in focus[:2] + focus[3:4]:
        for lab, g in (("backtest", B2), ("real", RL)):
            z = fz(g).fillna(False).values
            for zn, m in (("zone", z), ("rest", ~z)):
                h = g[m]
                r = dict(zone=name, block=lab, side=zn, N=len(h), WORKED_pct=(h.label == "WORKED").mean() * 100 if len(h) else np.nan,
                         reach_med=h.reach.median())
                for gg in GR:
                    k = h.group == gg
                    if k.any():
                        r[gg] = f"{int(k.sum())} · {h.net[k].sum():+.0f}"
                rows4.append(r)
    T4 = pd.DataFrame(rows4).fillna("")
    P(md(T4))
    T4.to_csv(os.path.join(OUT, "path_labels.csv"), index=False)
    # ── 4. the three open shorts: zone membership ──
    P("\n## 4 · Open shorts (11-26) — zone membership only (no P&L)")
    rows5 = []
    for _, r in OP.iterrows():
        d = dict(pair=r.pair.replace("USDT", ""))
        for v in VARS:
            c = CUTS[v]
            tz = "T1" if r[v] <= c["t1"] else ("T3" if r[v] > c["t2"] else "T2")
            d[VLAB[v]] = f"{tz}{' · < sign cut' if v in SIGN_CUT and r[v] < SIGN_CUT[v] else ''}"
        rows5.append(d)
    P(md(pd.DataFrame(rows5)))
    P("(T1 / T2 / T3 = the DESIGN terciles; cut values in table 1.)")
    # ── chart ──
    chart(B2, RL)
    with open(os.path.join(ROOT, "reports", TAG + "_tables.md"), "w") as f:
        f.write("\n".join(MD) + "\n")


def chart(B2, RL):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    h = np.arange(HOLD_H + 1)
    col = {"falling": "#c2410c", "flat / rising": "#2563eb"}
    for k, (title, g, key) in enumerate((("Backtest — short MTM, held 72 h (no exit)", B2, "path_raw"),
                                         ("Backtest — under today's exit (frozen at booked net)", B2, "path_exit"),
                                         ("Real kept fills (master + batch) — today's exit", RL, "path_exit"))):
        for zn, m in (("falling", g.ret14 < 0), ("flat / rising", g.ret14 >= 0)):
            arr = np.vstack(g[m][key].values) if m.any() else np.full((1, HOLD_H + 1), np.nan)
            mu = np.nanmean(arr, 0)
            ax[k].plot(h, mu, color=col[zn], lw=2, label=f"{zn} base (14-d return {'<' if zn == 'falling' else '≥'} 0) · N {int(m.sum())}")
            if len(arr) > 5:
                se = np.nanstd(arr, 0) / np.sqrt(np.sum(np.isfinite(arr), 0).clip(1))
                ax[k].fill_between(h, mu - 1.96 * se, mu + 1.96 * se, color=col[zn], alpha=.12, lw=0)
        ax[k].axhline(0, color="#888", lw=.8)
        ax[k].set_title(title, fontsize=10)
        ax[k].set_xlabel("hours after entry")
        ax[k].legend(fontsize=8, loc="lower left")
        ax[k].grid(alpha=.25)
    ax[0].set_ylabel("avg short P&L % (gross MTM / booked net)")
    fig.suptitle("FLATBASE_SHORT — average P&L path after entry, falling vs flat / rising 14-day base (band = 95 % CI of the mean)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", TAG + ".png"), dpi=120)


if __name__ == "__main__":
    main()
