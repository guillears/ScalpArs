#!/usr/bin/env python3
"""FLATBASE_ENTRY_TIMING_2026-11-26 — is today's Flatbase entry (market at the FB_CONFIRM 4-h close) well timed, or does ONE later /
better-priced entry variable shorten the trade and keep it longer in profit? Research only (operator 2026-11-26). No bot / config change.
Pre-registration (frozen before any variant result): reports/FLATBASE_ENTRY_TIMING_PREREG_2026-11-26.md.

DATA
  backtest = the OOS_REAL_FLATBASE rebuild (reports/OOS_REAL_FLATBASE_data/wl_features.csv) on today's stack: HIGHER_HIGH + NEAR_BASE > 2.09
             + FUNDING ≤ 0.005 % → 545 trades (= FLATBASE_BASE_TREND_2026-11-26). 5m bars via study_frenzy_swing_short_walk.load (the rebuild's
             own sources). Funding: oos_real_flatbase_common.funding_table (missing coverage = −0.10 %/8 h, the rebuild's convention).
  real     = master 17 kept fills (reports/KEPT_LIST_FLATBASE_2026-11-21.csv, booked = MASTER_POOL_stacked stack_pct) + the current batch's 3
             kept closed fills (KERNEL / MMT / PIPPIN) from the orders export (argv[1]). All re-priced on 5m bars by the same machinery
             (LIVE-sim) so variant − LIVE is like-for-like. Missing 5m spans → public fapi klines (≤ 1,000 weight / min, used-weight header
             read on every call, stop on 418 / 429), cached in reports/FLATBASE_ENTRY_TIMING_2026-11-26_data/k5m.
VARIANTS / COSTS / DECISION RULE: see the pre-registration (10 variants + LIVE, all counted in the pick-the-best null).
Usage: venv/bin/python -I scripts/FLATBASE_ENTRY_TIMING_2026-11-26.py [orders.csv]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import oos_real_flatbase_recon as RC      # noqa: E402  (sets the walk's kline sources + T0)
import oos_real_flatbase_common as C      # noqa: E402
import review_math as RM                  # noqa: E402

W = RC.W
TAG = "FLATBASE_ENTRY_TIMING_2026-11-26"
OUT = os.path.join(ROOT, "reports", TAG + "_data")
os.makedirs(os.path.join(OUT, "k5m"), exist_ok=True)
EXPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-26_13-02-05.csv")
BAR, HR, DAY = 300_000, 3_600_000, 86_400_000
TAKER, MAKER = RC.TAKER, RC.MAKER          # % per side
TP, SL, HOLD_B = 5.0, 10.0, 864
STOP_SLIP_MIN = 0.10
TRADE_THRU = 0.0005                        # a limit fills only when the bar trades ≥ 0.05 % through it
X_SIZE = 0.61                              # one-size ruler (the DL 601 / BASE_TREND Flatbase study's)
DESIGN_END = "2025-07-01"
VARS = ["LIVE", "MK0_4h", "ATR05_12h", "ATR1_12h", "ATR2_12h", "ATR1_24h", "VWAP_24h", "C1H", "C4H", "LOW1H", "DELAY4"]
VDESC = {"LIVE": "market at the confirm close (today)", "MK0_4h": "limit at the confirm close, 4 h", "ATR05_12h": "limit +0.5×ATR1h, 12 h",
         "ATR1_12h": "limit +1×ATR1h, 12 h", "ATR2_12h": "limit +2×ATR1h, 12 h", "ATR1_24h": "limit +1×ATR1h, 24 h",
         "VWAP_24h": "limit at the spike VWAP, 24 h", "C1H": "1st 1h close < confirm close (24 h)", "C4H": "next 4h close < confirm close (24 h)",
         "LOW1H": "1st 1h close < confirm-candle low (24 h)", "DELAY4": "market 4 h after the confirm"}
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


# ───────────────────────────── klines (real fills only) ─────────────────────────────
class Budget:
    used = 0
    calls = 0


def fetch(pair, start, end):
    import requests
    rows, t = [], int(start)
    while t < end:
        if Budget.used >= 1000:
            time.sleep(62 - (time.time() % 60))
            Budget.used = 0
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params=dict(symbol=pair, interval="5m", startTime=t, endTime=int(end) - 1, limit=1000), timeout=20)
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
        t = int(d[-1][0]) + BAR
        if len(d) < 1000:
            break
        time.sleep(0.15)
    x = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows],
                     columns=["open_time", "o", "h", "l", "c", "vol"])
    return x[(x.open_time >= start) & (x.open_time < end)]


def arrays_from(d):
    d = d.drop_duplicates("open_time", keep="last").sort_values("open_time")
    grid = np.arange(d.open_time.iloc[0], d.open_time.iloc[-1] + 1, BAR, dtype="int64")
    d = d.set_index("open_time").reindex(grid)
    c = d.c.ffill().values
    o = d.o.fillna(pd.Series(c, index=d.index)).values
    h = d.h.fillna(pd.Series(c, index=d.index)).values
    l_ = d.l.fillna(pd.Series(c, index=d.index)).values
    return grid, o.astype(float), h.astype(float), l_.astype(float), c.astype(float), d.vol.fillna(0).values.astype(float)


def real_arrays(pair, a, b):
    """5m arrays covering [a, b): the rebuild's sources when they cover it, else the study cache / a fetch."""
    x = W.load(pair)
    if x is not None and x[0][0] <= a and x[0][-1] >= b - BAR:
        t, o, h, l_, c, v = x[:6]
        return t, o, h, l_, c, v
    f = os.path.join(OUT, "k5m", pair + ".csv")
    d = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["open_time", "o", "h", "l", "c", "vol"])
    now = int(time.time() * 1000) // BAR * BAR - BAR
    need = (min(b, now) - a) // BAR
    if ((d.open_time >= a) & (d.open_time < b)).sum() < 0.98 * need:
        d = pd.concat([d, fetch(pair, a, min(b, now))]).drop_duplicates("open_time", keep="last").sort_values("open_time")
        d.to_csv(f, index=False)
    return arrays_from(d[(d.open_time >= a) & (d.open_time < b)])


# ───────────────────────────── the simulator ─────────────────────────────
def atr1h_pct(t, h, l_, c, i0):
    """Wilder-14 ATR % of the 1h bars fully closed by t[i0] (200 h warm-up)."""
    a = max(0, i0 - 200 * 12)
    k = t[a:i0] // HR * HR
    df = pd.DataFrame({"k": k, "h": h[a:i0], "l": l_[a:i0], "c": c[a:i0]}).groupby("k").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), n=("c", "size"))
    df = df[df.n == 12]
    if len(df) < 20:
        return np.nan
    pc = np.r_[df.c.values[0], df.c.values[:-1]]
    tr = np.maximum(df.h.values - df.l.values, np.maximum(abs(df.h.values - pc), abs(df.l.values - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values[-1]
    return atr / df.c.values[-1] * 100


def exit_walk(t, o, h, l_, c, j0, E, sx, limit_fill):
    """SHORT at E. Market: walk from bar j0 (TP / stop both live). Limit: bar j0 = the fill bar → only the stop can hit there, then j0 + 1 ….
    → (gross %, exit fee %, why, exit bar idx). Stop first on a shared bar; TP maker at min(level, open); stop taker max(level, open) × slip."""
    n = len(t)
    S, T = E * (1 + SL / 100), E * (1 - TP / 100)
    sxs = max(sx, STOP_SLIP_MIN)
    last = min(n - 1, j0 + HOLD_B - 1)
    if limit_fill and h[j0] >= S:
        X = S * (1 + sxs / 100)
        return (1 - X / E) * 100, TAKER * X / E, "STOP", j0
    a = j0 + 1 if limit_fill else j0
    if a > last:
        return (1 - c[last] / E) * 100, TAKER, "OPEN", last
    hs = np.flatnonzero(h[a:last + 1] >= S)
    ts = np.flatnonzero(l_[a:last + 1] <= T)
    ks = hs[0] if len(hs) else 10**9
    kt = ts[0] if len(ts) else 10**9
    if ks <= kt and ks < 10**9:
        j = a + ks
        X = max(S, o[j]) * (1 + sxs / 100)
        return (1 - X / E) * 100, TAKER * X / E, "STOP", j
    if kt < 10**9:
        j = a + kt
        X = min(T, o[j])
        return (1 - X / E) * 100, MAKER * X / E, "TP", j
    X = c[last] * (1 + sx / 100)
    full = last == j0 + HOLD_B - 1
    return (1 - X / E) * 100, TAKER * X / E, ("TIME" if full else "OPEN"), last


def entry_for(var, t, o, h, l_, c, v, i0, on, se, live_E=None):
    """→ (entry bar idx, E, entry fee %, is_limit) or None (no fill / no trigger). i0 = the bar opening at the confirm close.
    live_E = a real fill's own price (confirm-entered real fills: LIVE is the real fill, not a model)."""
    k = i0 - 1
    C0 = c[k]
    n = len(t)
    if var == "LIVE":
        return i0, (live_E if live_E else o[i0] * (1 - se / 100)), TAKER, False
    if var in ("MK0_4h", "ATR05_12h", "ATR1_12h", "ATR2_12h", "ATR1_24h", "VWAP_24h"):
        if var == "VWAP_24h":
            tp_ = (h[on:k + 1] + l_[on:k + 1] + c[on:k + 1]) / 3
            q = v[on:k + 1] * tp_
            L = max(float((tp_ * q).sum() / q.sum()) if q.sum() > 0 else C0, C0)
            hrs = 24
        else:
            mult = {"MK0_4h": 0.0, "ATR05_12h": 0.5, "ATR1_12h": 1.0, "ATR2_12h": 2.0, "ATR1_24h": 1.0}[var]
            hrs = 4 if var == "MK0_4h" else (24 if var == "ATR1_24h" else 12)
            atr = atr1h_pct(t, h, l_, c, i0) if mult else 0.0
            if not np.isfinite(atr):
                return None
            L = C0 * (1 + mult * atr / 100)
        hit = np.flatnonzero(h[i0:min(n, i0 + hrs * 12)] >= L * (1 + TRADE_THRU))
        if not len(hit):
            return None
        return i0 + hit[0], L, MAKER, True
    if var == "DELAY4":
        j = i0 + 48
        return (j, o[j] * (1 - se / 100), TAKER, False) if j < n else None
    span = range(i0, min(n - 1, i0 + 24 * 12))       # 5m bars whose CLOSE is at t[j] + 5 min, from the first bar after the confirm
    if var in ("C1H", "LOW1H"):
        lvl = C0 if var == "C1H" else float(l_[k - 47:k + 1].min())
        grid = HR
    else:
        lvl, grid = C0, 4 * HR
    for j in span:
        if (t[j] + BAR) % grid == 0 and c[j] < lvl:
            return j + 1, o[j + 1] * (1 - se / 100), TAKER, False
    return None


def simulate(pair, t, o, h, l_, c, v, i0, on, se, sx, F, fund_ok=True, live_E=None):
    """every variant for one episode → list of dicts (one per variant)."""
    out = []
    for var in VARS:
        e = entry_for(var, t, o, h, l_, c, v, i0, on, se, live_E)
        if e is None:
            out.append(dict(var=var, traded=False))
            continue
        j0, E, fee_in, lim = e
        g, fee_x, why, jx = exit_walk(t, o, h, l_, c, j0, E, sx, lim)
        te = int(t[j0]) + (BAR // 2 if lim else 0)
        tx = int(t[jx]) + BAR
        fu, miss = C.fund(F, pair, te + 10_000, tx) if fund_ok else (0.0, False)
        if not fund_ok:
            fu, miss = 0.0, True
        net = g - fee_in - fee_x + fu
        mtm = (1 - c[j0:jx + 1] / E) * 100
        # descriptive path facts (from the entry bar, 72 h horizon for the first-touch reads)
        hz = slice(j0, min(len(t), j0 + HOLD_B))
        fav = np.flatnonzero(l_[hz] <= E * 0.99)
        adv = np.flatnonzero(h[hz] >= E * 1.01)
        f1 = fav[0] if len(fav) else 10**9
        a1 = adv[0] if len(adv) else 10**9
        pos = np.flatnonzero((1 - c[hz] / E) * 100 > 0)
        raw = np.full(73, np.nan)
        raw[0] = 0.0
        ex = np.full(73, np.nan)
        ex[0] = 0.0
        for hh in range(1, 73):
            b = j0 + hh * 12 - 1
            if b < len(t):
                raw[hh] = (1 - c[b] / E) * 100
                ex[hh] = net if int(t[b]) + BAR >= tx else raw[hh]
        out.append(dict(var=var, traded=True, why=why, E=E, te=te, tx=tx, hold_h=(tx - te) / HR, gross=g, fund=fu, fund_missing=miss, net=net,
                        wait_h=(te - int(t[i0])) / HR, entry_vs_C=(E / c[i0 - 1] - 1) * 100,
                        pct_time_pos=float((mtm > 0).mean() * 100), mae=float(max(0.0, (h[j0:jx + 1].max() / E - 1) * 100)),
                        first_touch=("FAV" if f1 < a1 else ("ADV" if a1 < f1 else ("SAME" if f1 < 10**9 else "NONE"))),
                        h_to_fav1=(f1 / 12 if f1 < 10**9 else np.nan),
                        mae_before_fav1=(float(max(0.0, (h[j0:j0 + f1 + 1].max() / E - 1) * 100)) if f1 < 10**9 else np.nan),
                        h_first_pos=(pos[0] / 12 if len(pos) else np.nan), path_raw=raw, path_exit=ex))
    return out


# ───────────────────────────── build ─────────────────────────────
def build():
    fb = os.path.join(OUT, "sim.pkl")
    if os.path.exists(fb):
        return pd.read_pickle(fb)
    F = C.funding_table()
    X = pd.read_csv(os.path.join(ROOT, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    X = X[X.X0_why.isin(["TP", "STOP", "TIME"]) & (X.dist_base > 2.09) & (X.funding_rate_pct <= 0.005) & (~X.hh.astype(bool))].copy()
    rows, par = [], []
    for pair, g in X.groupby("pair"):
        arr = W.load(pair)
        if arr is None:
            continue
        t, o, h, l_, c, v = arr[:6]
        for _, r in g.iterrows():
            i0 = int(np.searchsorted(t, int(r.cf_entry_t)))
            on = int(np.searchsorted(t, int(r.on_t)))
            if i0 >= len(t) or t[i0] != int(r.cf_entry_t) or t[on] != int(r.on_t):
                continue
            atr = atr1h_pct(t, h, l_, c, i0)
            par.append((atr, float(r.atr_1h)))
            # parity: LIVE re-priced with the rebuild's own stop slip (no 0.10 floor) vs X0_net
            E = o[i0] * (1 - float(r.slip_e) / 100)
            for res in simulate(pair, t, o, h, l_, c, v, i0, on, float(r.slip_e), float(r.slip_x), F):
                rows.append(dict(src="BT", ep=r.key, pair=pair, ep_t=int(r.cf_entry_t), booked=float(r.X0_net), X0_why=r.X0_why,
                                 E_parity=abs(E / float(r.cf_entry_px) - 1) * 100, atr1h=atr, **res))
        print(f"  bt rows {len(rows)}", end="\r")
    # real fills
    KL = pd.read_csv(os.path.join(ROOT, "reports", "KEPT_LIST_FLATBASE_2026-11-21.csv"))
    m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[m.entry_strategy == "FLATBASE_SHORT"]
    o_ = pd.read_csv(EXPORT, low_memory=False)
    o_ = o_[(o_.entry_strategy == "FLATBASE_SHORT") & (o_.status == "CLOSED")]
    if "is_probe" in o_:
        o_ = o_[o_.is_probe.astype(str).str.lower() != "true"]
    # master fills opened BEFORE the FB_CONFIRM ship (DL 428) were entered at the S4 signal; the master prices them at the CONFIRM entry
    # (reports/FLATBASE_CONFIRM_CF.csv, DL 527) → their T0 = the first 4-h close below the S4 close within 24 h (services/flatbase.confirm_scan)
    cfc = pd.read_csv(os.path.join(ROOT, "reports", "FLATBASE_CONFIRM_CF.csv"), comment="#")
    s4_entered = {(r.pair, r.opened_at[:19]) for r in cfc.itertuples() if r.cf_why != "CONFIRMED_LIVE"}
    for _, kk in KL.iterrows():
        src = m if kk.block == "MASTER" else o_
        r = src[(src.pair == kk.pair) & (src.opened_at.str[:19] == kk.k[:19]) & (src.direction == "SHORT")]
        if len(r) != 1:
            print("  ! real fill not found once:", kk.pair, kk.k, len(r))
            continue
        r = r.iloc[0]
        booked = float(r.stack_pct) if kk.block == "MASTER" else float(r.pnl_percentage)
        te = int(pd.Timestamp(r.opened_at).value // 10**6)
        Ts = te // BAR * BAR
        spike = int(pd.Timestamp(r.entry_frenzy_spike_at).value // 10**6)
        t, o, h, l_, c, v = real_arrays(r.pair, Ts - 210 * HR, Ts + 125 * HR)
        s4e = (r.pair, r.opened_at[:19]) in s4_entered
        if s4e:
            ks = int(np.searchsorted(t, Ts)) - 1
            T0 = None
            for j in range(ks + 1, min(len(t), ks + 1 + 24 * 12)):
                if (t[j] + BAR) % (4 * HR) == 0 and c[j] < c[ks]:
                    T0 = int(t[j]) + BAR
                    break
            if T0 is None:
                print("  ! no confirm for an S4-entered master fill:", r.pair, r.opened_at)
                continue
        else:
            T0 = Ts
        i0 = int(np.searchsorted(t, T0))
        on = int(np.searchsorted(t, spike - BAR))
        if i0 >= len(t) or t[i0] != T0:
            print("  ! no bar at the confirm close:", r.pair)
            continue
        x = F.get(r.pair)
        fok = bool(x is not None and len(x[0]) and x[0][-1] >= T0 + 100 * HR)
        for res in simulate(r.pair, t, o, h, l_, c, v, i0, on, 0.10, 0.10, F, fund_ok=fok, live_E=None if s4e else float(r.entry_price)):
            rows.append(dict(src=kk.block, ep=f"{r.pair}|{r.opened_at[:19]}", pair=r.pair, ep_t=T0, booked=booked, s4_entered=s4e,
                             X0_why=str(r.get("close_reason", "")), E_parity=abs(o[i0] / float(r.entry_price) - 1) * 100,
                             atr1h=atr1h_pct(t, h, l_, c, i0), **res))
    S = pd.DataFrame(rows)
    S.attrs["atr_parity"] = par
    S.to_pickle(fb)
    print(f"\n  klines calls {Budget.calls}")
    return S


# ───────────────────────────── stats ─────────────────────────────
def daily_path(pct, day, d0, d1, x=X_SIZE):
    s = pd.Series(np.asarray(pct, float) * x / 100).groupby(pd.to_datetime(pd.Series(np.asarray(day))).values).sum()
    r = s.reindex(pd.date_range(d0, d1, freq="D"), fill_value=0.0).values
    eq = np.cumprod(1 + r)
    return (eq[-1] ** (1 / len(r)) - 1) * 100, ((eq / np.maximum.accumulate(eq)) - 1).min() * 100, r.min() * 100, len(r)


def boot_ci(v, cl, B=4000, seed=7):
    v = np.asarray(v, float)
    u, inv = np.unique(np.asarray(cl), return_inverse=True)
    s = np.bincount(inv, weights=v)
    nn = np.bincount(inv)
    ix = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    mu = s[ix].sum(1) / nn[ix].sum(1)
    return float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5)), float((mu > 0).mean())


def wide(S):
    """episode × variant net (NaN = no trade)."""
    return S.pivot_table(index="ep", columns="var", values="net", aggfunc="first").reindex(columns=VARS)


def block_table(S, label):
    ep = S[S["var"] == "LIVE"].set_index("ep")
    Wn = wide(S)
    eps = ep.index
    Wn = Wn.loc[eps]
    live = Wn["LIVE"]
    d0 = pd.to_datetime(ep.ep_t, unit="ms").min().strftime("%Y-%m-%d")
    d1 = pd.to_datetime(S[S.traded].tx, unit="ms").max().strftime("%Y-%m-%d")
    cl = pd.to_datetime(ep.ep_t, unit="ms").dt.strftime("%Y-%m-%d").values
    rows = []
    for var in VARS:
        g = S[(S["var"] == var) & S.traded].set_index("ep").reindex(eps)
        k = g.traded.fillna(False).astype(bool).values
        net = g.net.values
        dcr, dd, wd, nd = daily_path(net[k], pd.to_datetime(g.tx[k], unit="ms").dt.strftime("%Y-%m-%d").values, d0, d1)
        dlt = np.where(k, net, 0.0) - live.values
        lo, hi, pp = boot_ci(dlt, cl)
        kept = k & np.isfinite(live.values)
        posd = g.hold_h.values[k].sum() / 24
        rows.append(dict(block=label, variant=var, N=int(k.sum()), lost=int((~k).sum()), WR=(net[k] > 0).mean() * 100, avg=net[k].mean(),
                         sum=net[k].sum(), hold_h=g.hold_h.values[k].mean(), wait_h=g.wait_h.values[k].mean(),
                         pct_time_pos=g.pct_time_pos.values[k].mean(), per_pos_day=net[k].sum() / posd if posd else np.nan,
                         conc=g.hold_h.values[k].sum() / (24 * nd), DCR=dcr, maxDD=dd, worst_day=wd,
                         d_ep=dlt.mean(), d_lo=lo, d_hi=hi, P_d_pos=pp,
                         d_kept=(net[kept] - live.values[kept]).mean() if kept.any() else np.nan,
                         lost_live_avg=live.values[~k].mean() if (~k).any() else np.nan))
    return pd.DataFrame(rows)


def pick_best_null(S, B=4000, seed=3):
    """day-block sign-flip of the paired Δ (same flips for every variant) → P(max mean Δ over the 10 variants ≥ the observed best)."""
    ep = S[S["var"] == "LIVE"].set_index("ep")
    Wn = wide(S).loc[ep.index]
    D = np.column_stack([np.nan_to_num(Wn[v].values, nan=0.0) - Wn["LIVE"].values for v in VARS[1:]])
    cl = pd.to_datetime(ep.ep_t, unit="ms").dt.strftime("%Y-%m-%d").values
    u, inv = np.unique(cl, return_inverse=True)
    Sd = np.vstack([np.bincount(inv, weights=D[:, j], minlength=len(u)) for j in range(D.shape[1])]).T
    obs = D.mean(0)
    rng = np.random.default_rng(seed)
    mx = np.empty(B)
    for b in range(B):
        f = rng.choice([-1.0, 1.0], len(u))
        mx[b] = (Sd * f[:, None]).sum(0).max() / len(D)
    best = int(np.argmax(obs))
    return VARS[1 + best], float(obs[best]), float((mx >= obs[best]).mean()), float(np.percentile(mx, 95))


def main():
    S = build()
    S["traded"] = S.traded.astype(bool)
    BT = S[S.src == "BT"].copy()
    RL = S[S.src.isin(["MASTER", "BATCH"])].copy()
    BT["blk"] = np.where(pd.to_datetime(BT.ep_t, unit="ms") < pd.Timestamp(DESIGN_END), "DESIGN", "HOLDOUT")
    dt = pd.to_datetime(BT.ep_t, unit="ms")
    BT["half"] = dt.dt.year.astype(str).str[2:] + np.where(dt.dt.month < 7, "-H1", "-H2")
    LV = BT[BT["var"] == "LIVE"]
    P(f"# {TAG} — Flatbase entry timing: is the confirm-close entry right, or can one later / better-priced entry do better?\n")
    P("Research only — no bot / config change. Summaries only. Pre-registration: reports/FLATBASE_ENTRY_TIMING_PREREG_2026-11-26.md · "
      "script: scripts/FLATBASE_ENTRY_TIMING_2026-11-26.py\n")
    P("## Data + parity")
    par = np.array(S.attrs.get("atr_parity", []), float) if S.attrs.get("atr_parity") else np.empty((0, 2))
    lp = LV[LV.why.notna()]
    agree = (lp.why == lp.X0_why).mean() * 100
    P(f"- Backtest: {LV.ep.nunique()} episodes (today's stack). LIVE re-priced here vs the rebuild's X0: exit reason agrees on {agree:.1f} %; "
      f"avg net {lp.net.mean():+.3f} vs X0 {lp.booked.mean():+.3f} (difference = the 0.10 % stop-slip floor + funding window); entry-price |Δ| "
      f"max {LV.E_parity.max():.4f} %.")
    if len(par):
        ok = np.isfinite(par).all(1)
        P(f"- ATR1h parity vs the rebuild's atr_1h: median |Δ| {np.median(abs(par[ok, 0] - par[ok, 1])):.3f} pts on {ok.sum()} (200 h warm-up vs full history).")
    rl = RL[RL["var"] == "LIVE"]
    P(f"- Real: {rl.ep.nunique()} kept fills (master {int((rl.src == 'MASTER').sum())} · batch {int((rl.src == 'BATCH').sum())}). LIVE-sim on 5m vs booked: "
      f"avg {rl.net.mean():+.2f} vs {rl.booked.mean():+.2f}; same sign on {(np.sign(rl.net) == np.sign(rl.booked)).sum()} / {len(rl)}. "
      f"Funding for real fills: {'real' if not rl.fund_missing.all() else 'not covered → 0 (all variants alike)'} "
      f"({int((~rl.fund_missing.astype(bool)).sum())} / {len(rl)} covered). {int(rl.s4_entered.fillna(False).astype(bool).sum())} master fills were "
      "entered at S4 before the FB_CONFIRM ship (DL 428): the master books them at the confirm entry (FLATBASE_CONFIRM_CF, DL 527), so their "
      "LIVE here = a market short at the rebuilt confirm close (0.10 % slip); the confirm-entered fills keep their REAL fill price as LIVE.")
    P(f"- DESIGN < {DESIGN_END} ({LV[LV.blk == 'DESIGN'].ep.nunique()}) · HOLDOUT ({LV[LV.blk == 'HOLDOUT'].ep.nunique()}). One size X = {X_SIZE} × equity. "
      "Costs: taker 0.045 % (+ row entry slip) for market entries, maker 0.018 % (no slip) for limits; TP maker; stop taker + max(slip, 0.10 %).\n")

    # ── 1 · today's entry described ──
    P("## 1 · Today's entry, described (LIVE)")
    rows = []
    for lab, g in (("backtest ALL", LV), ("backtest DESIGN", LV[LV.blk == "DESIGN"]), ("backtest HOLDOUT", LV[LV.blk == "HOLDOUT"]),
                   ("master (5m re-price)", rl[rl.src == "MASTER"]), ("batch (5m re-price)", rl[rl.src == "BATCH"])):
        g = g[g.traded]
        tp = g[g.why == "TP"]
        rows.append(dict(block=lab, N=len(g), WR=(g.net > 0).mean() * 100, avg=g.net.mean(), hold_avg_h=g.hold_h.mean(), hold_med_h=g.hold_h.median(),
                         TP_med_h=tp.hold_h.median(), TP_share=len(tp) / len(g) * 100, pct_time_pos=g.pct_time_pos.mean(),
                         MAE_TP_med=tp.mae.median(), MAE_TP_p75=tp.mae.quantile(.75),
                         against_first=(g.first_touch == "ADV").mean() * 100, fav_first=(g.first_touch == "FAV").mean() * 100,
                         h_to_fav1_med=g.h_to_fav1.median(), mae_pre_fav1_med=g.mae_before_fav1.median(), h_first_pos_med=g.h_first_pos.median()))
    T1 = pd.DataFrame(rows)
    P(md(T1))
    P("against_first / fav_first = % of trades whose first ±1 % touch went against / with us; h_to_fav1 = hours to the first +1 % in our favour; "
      "mae_pre_fav1 = the worst adverse move before that; h_first_pos = hours to the first 5m close in profit; pct_time_pos = % of the held 5m "
      "closes with gross MTM > 0.\n")
    # where does the drawdown sit: by outcome
    rows = []
    for lab, g in (("backtest", LV[LV.traded]), ("real", rl[rl.traded])):
        for o_, m in (("winners", g.net > 0), ("losers", g.net <= 0)):
            h = g[m]
            rows.append(dict(block=lab, side=o_, N=len(h), hold_med_h=h.hold_h.median(), pct_time_pos=h.pct_time_pos.mean(),
                             against_first=(h.first_touch == "ADV").mean() * 100, h_to_fav1_med=h.h_to_fav1.median(),
                             mae_pre_fav1_med=h.mae_before_fav1.median(), mae_med=h.mae.median()))
    P(md(pd.DataFrame(rows)))

    # ── 2 · variants ──
    P("\n## 2 · Variants vs LIVE (frozen grid, one size, every calendar day)")
    P("lost = episodes with no fill / no trigger (contribute 0); d_ep = paired Δ net per EPISODE vs LIVE (lost = −LIVE net) with day-block 95 % CI; "
      "d_kept = Δ on the episodes the variant still trades; lost_live_avg = what LIVE earned on the lost ones; per_pos_day = Σ net % per position-day "
      "(slot-bound turnover); conc = average open Flatbase positions.\n")
    T = []
    for lab, g in (("backtest ALL", BT), ("backtest DESIGN", BT[BT.blk == "DESIGN"]), ("backtest HOLDOUT", BT[BT.blk == "HOLDOUT"]),
                   ("master", RL[RL.src == "MASTER"]), ("batch", RL[RL.src == "BATCH"]), ("master + batch", RL)):
        T.append(block_table(g, lab))
    T = pd.concat(T, ignore_index=True)
    T.insert(2, "rule", T.variant.map(VDESC))
    T.to_csv(os.path.join(ROOT, "reports", TAG + ".csv"), index=False)
    cols = ["variant", "rule", "N", "lost", "WR", "avg", "sum", "hold_h", "wait_h", "pct_time_pos", "per_pos_day", "conc", "DCR", "maxDD", "worst_day",
            "d_ep", "d_lo", "d_hi", "d_kept", "lost_live_avg"]
    for lab in T.block.unique():
        P(f"\n### {lab}")
        P(md(T[T.block == lab][cols]))

    # ── 3 · half-years + luck test ──
    P("\n## 3 · Consistency: paired Δ per episode by half-year (backtest) + the pick-the-best null")
    rows = []
    for hf in sorted(BT.half.unique()):
        g = BT[BT.half == hf]
        Wn = wide(g)
        r = dict(half=hf, N=int(Wn.LIVE.notna().sum()), LIVE_avg=Wn.LIVE.mean())
        for v in VARS[1:]:
            r[v] = (np.nan_to_num(Wn[v].values, nan=0.0) - Wn.LIVE.values).mean()
        rows.append(r)
    H = pd.DataFrame(rows)
    pos = {v: f"{int((H[v] > 0).sum())}/{len(H)}" for v in VARS[1:]}
    P(md(H))
    P("half-years with Δ > 0: " + " · ".join(f"{v} {pos[v]}" for v in VARS[1:]))
    nb = pick_best_null(BT)
    nh = pick_best_null(BT[BT.blk == "HOLDOUT"])
    nm = pick_best_null(RL[RL.src == "MASTER"])
    P(f"\nPick-the-best null (10 variants, day-block sign flips, same flips for all variants): backtest ALL best = {nb[0]} Δ {nb[1]:+.3f}/episode · "
      f"p {nb[2]:.3f} (null 95th pct {nb[3]:+.3f}) · HOLDOUT best = {nh[0]} Δ {nh[1]:+.3f} · p {nh[2]:.3f} · master best = {nm[0]} Δ {nm[1]:+.3f} · p {nm[2]:.3f}.")

    # ── 4 · master detail for the variants: LOO + concentration ──
    P("\n## 4 · Master (real kept fills): leave-one-out + per-pair concentration of each variant's Δ")
    M = RL[RL.src == "MASTER"]
    Wm = wide(M)
    labs = [e.split("|")[0].replace("USDT", "") + " " + e.split("|")[1][5:10] for e in Wm.index]
    rows = []
    for v in VARS[1:]:
        d = np.nan_to_num(Wm[v].values, nan=0.0) - Wm.LIVE.values
        pr = pd.Series(d, index=[e.split("|")[0] for e in Wm.index]).groupby(level=0).sum()
        rows.append(dict(variant=v, d_sum=d.sum(), loo=RM.loo_text(list(d), labels=labs, unit="pts"),
                         top_pair=(pr.abs().idxmax().replace("USDT", "") if len(pr) else ""),
                         top_pair_share=(pr.abs().max() / pr.abs().sum() * 100 if pr.abs().sum() else np.nan)))
    P(md(pd.DataFrame(rows)))
    BB = BT[BT["var"] != "LIVE"].copy()
    rows = []
    Wb = wide(BT)
    for v in VARS[1:]:
        d = pd.Series(np.nan_to_num(Wb[v].values, nan=0.0) - Wb.LIVE.values, index=[e.split("|")[0] for e in Wb.index]).groupby(level=0).sum()
        neg = d[d < 0].sum()
        rows.append(dict(variant=v, d_sum=d.sum(), pairs=len(d), top_loss_pair=(d.idxmin().replace("USDT", "")), top_loss_share=(d.min() / neg * 100 if neg else np.nan),
                         top_gain_pair=d.idxmax().replace("USDT", ""), top_gain=d.max()))
    P("\nBacktest per-pair concentration of the Δ:")
    P(md(pd.DataFrame(rows)))
    del BB

    # ── 5 · verdict per frozen rule ──
    P("\n## 5 · Frozen decision rule")
    rows = []
    for v in VARS[1:]:
        a = T[(T.block == "backtest ALL") & (T.variant == v)].iloc[0]
        ho = T[(T.block == "backtest HOLDOUT") & (T.variant == v)].iloc[0]
        ms = T[(T.block == "master") & (T.variant == v)].iloc[0]
        a0 = T[(T.block == "backtest ALL") & (T.variant == "LIVE")].iloc[0]
        m0 = T[(T.block == "master") & (T.variant == "LIVE")].iloc[0]
        cand = (ms.d_ep > 0 and a.d_ep > 0 and a.d_lo > 0 and ho.d_ep > 0 and int(pos[v].split("/")[0]) >= 3 and nb[0] == v and nb[2] < 0.05
                and ms.DCR > m0.DCR and a.DCR > a0.DCR)
        obs = ms.d_ep >= 0 and a.d_ep >= 0
        rows.append(dict(variant=v, master_d=ms.d_ep, bt_d=a.d_ep, bt_lo=a.d_lo, holdout_d=ho.d_ep, halves=pos[v], DCR_bt=f"{a0.DCR:+.2f}→{a.DCR:+.2f}",
                         DCR_master=f"{m0.DCR:+.2f}→{ms.DCR:+.2f}", verdict="CANDIDATE" if cand else ("OBSERVE" if obs else "keep today")))
    V = pd.DataFrame(rows)
    P(md(V))
    V.to_csv(os.path.join(OUT, "verdict.csv"), index=False)
    best = T[(T.block == "backtest ALL") & (T.variant != "LIVE")].sort_values("d_ep").iloc[-1].variant
    chart(BT, RL, best)
    with open(os.path.join(ROOT, "reports", TAG + "_tables.md"), "w") as f:
        f.write("\n".join(MD) + "\n")


def chart(BT, RL, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hx = np.arange(73)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    def band(a, g, key, col, lab):
        g = g[g.traded]
        if not len(g):
            return
        arr = np.vstack(g[key].values)
        mu = np.nanmean(arr, 0)
        a.plot(hx, mu, color=col, lw=2, label=f"{lab} · N {len(g)}")
        if len(g) > 5:
            se = np.nanstd(arr, 0) / np.sqrt(np.sum(np.isfinite(arr), 0).clip(1))
            a.fill_between(hx, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=.13, lw=0)
    band(ax[0], BT[BT["var"] == "LIVE"], "path_exit", "#2563eb", "LIVE (confirm close, market)")
    band(ax[0], BT[BT["var"] == best], "path_exit", "#c2410c", f"best variant: {best}")
    band(ax[0], BT[BT["var"] == "DELAY4"], "path_exit", "#7c3aed", "DELAY4 (pure 4 h wait)")
    ax[0].set_title("Backtest — avg short P&L from entry, under today's exit", fontsize=10)
    lv = BT[(BT["var"] == "LIVE") & BT.traded]
    band(ax[1], lv[lv.net > 0], "path_raw", "#15803d", "LIVE winners")
    band(ax[1], lv[lv.net <= 0], "path_raw", "#b91c1c", "LIVE losers")
    ax[1].set_title("Backtest — LIVE winners vs losers, raw MTM (no exit)", fontsize=10)
    band(ax[2], RL[RL["var"] == "LIVE"], "path_exit", "#2563eb", "LIVE (5m re-price)")
    band(ax[2], RL[RL["var"] == best], "path_exit", "#c2410c", f"{best}")
    band(ax[2], RL[RL["var"] == "DELAY4"], "path_exit", "#7c3aed", "DELAY4")
    ax[2].set_title("Real kept fills (master + batch), under today's exit", fontsize=10)
    for a in ax:
        a.axhline(0, color="#888", lw=.8)
        a.set_xlabel("hours after entry")
        a.grid(alpha=.25)
        a.legend(fontsize=8, loc="lower right")
    ax[0].set_ylabel("avg short P&L % (MTM; booked net after exit)")
    fig.suptitle("FLATBASE_SHORT — event study from entry: today's entry vs the best timing variant (band = 95 % CI of the mean)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", TAG + ".png"), dpi=120)


if __name__ == "__main__":
    main()
