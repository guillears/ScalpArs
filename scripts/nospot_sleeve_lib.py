#!/usr/bin/env python3
"""NOSPOT_SLEEVE (PREREG reports/NOSPOT_SLEEVE_2026-11-18_PREREG.md) — shared: load rows, costs, rule masks (primary + plateau), one
position per pair, stats (N · /day · WR · BE · avg · @0.20 · best-5 %-out · Kelly at the 95 % Wilson floor on days · halves · concentration ·
DCR at one size · max DD · worst day), path groups."""
import glob, os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
ND = os.path.join(R, "reports", "NOSPOT_SLEEVE_data")
DAY, H1 = 86_400_000, 3_600_000
TS = lambda s: pd.Timestamp(s).value // 10**6
EXITS = ["XE", "XA", "F55", "F106", "F1012", "F1510"]
HALVES = [("23-H2", "2023-07-01", "2024-01-01"), ("24-H1", "2024-01-01", "2024-07-01"), ("24-H2", "2024-07-01", "2025-01-01"),
          ("25-H1", "2025-01-01", "2025-07-01"), ("25-H2", "2025-07-01", "2026-01-01"), ("26-H1", "2026-01-01", "2026-07-01"),
          ("26-H2", "2026-07-01", "2026-11-18")]
DES = ("2023-07-01", "2025-07-01"); HOLD = ("2025-07-01", "2026-11-18")
SIZE = 0.61
SLIP_N = 6100.0
import json  # noqa: E402
SM = json.load(open(os.path.join(R, "reports", "SPOT_SHARE_DEEP_data", "slip_model.json")))


def slip_model(q5):
    return np.maximum(SM["a"] + SM["b"] * np.sqrt(SLIP_N / np.maximum(q5, 1.0)), 0.0)


def load(rows_dir=os.path.join(ND, "rows")):
    X = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(rows_dir, "*.pkl")))], ignore_index=True)
    X = X[(X.entry_ms >= TS("2023-07-01"))].sort_values(["pair", "on_ms", "entry_ms"]).reset_index(drop=True)
    X["half"] = ""
    for h, s, e in HALVES:
        X.loc[(X.entry_ms >= TS(s)) & (X.entry_ms < TS(e)), "half"] = h
    X["blk"] = np.where(X.entry_ms < TS(DES[1]), "DESIGN", "HOLDOUT")
    X["day"] = (X.entry_ms // DAY).astype(np.int64)
    X["hour"] = ((X.entry_ms // H1) % 24).astype(np.int8)
    X["epi"] = X.groupby(["pair", "on_ms"]).ngroup().astype(np.int32)
    X["slip"] = np.maximum(0.10, slip_model(X.q5.values.astype(float)))
    reg = pd.read_pickle(os.path.join(R, "reports", "LEADS_DEEP_2026-11-18_data", "regimes.pkl"))
    for nm in ("R1", "R4"):
        tv, vv = reg[nm]; k = np.searchsorted(tv, X.entry_ms.values, side="right") - 1
        X[nm] = np.where(k >= 0, vv[np.maximum(k, 0)], np.nan)
    X["f_neg"] = np.where(np.isfinite(X.f_last), (X.f_last < 0).astype(float), np.nan)
    X["f_hourly"] = np.where(np.isfinite(X.f_int_h), (X.f_int_h == 1).astype(float), np.nan)
    for e in EXITS:
        X[f"{e}_net"] = net(X, e).astype(np.float32)
        X[f"{e}_net20"] = net(X, e, 0.20).astype(np.float32)
    return X


def net(X, e, flat=None):
    s = X.slip.values.astype(float) if flat is None else np.full(len(X), flat)
    k = X[f"{e}_k"].values; tk = k != 0
    return X[f"{e}_r"].values.astype(float) - s - np.where(tk, s, 0.0) - 0.05 - np.where(tk, 0.05, 0.02) + X[f"{e}_f"].values.astype(float)


def v(X, c):
    return np.nan_to_num(X[c].values.astype(float), nan=np.nan)


def rule_masks(X):
    """{name: (primary mask, [plateau masks])} — PREREG table, frozen."""
    dd = v(X, "dd_hi"); oi1 = v(X, "oi_chg1h"); r1 = v(X, "ret1h")
    with np.errstate(invalid="ignore"):
        M = {}
        g = lambda a: np.nan_to_num(a, nan=0).astype(bool)
        f1 = lambda o, d: g((oi1 <= o) & (dd <= d) & (r1 <= 0))
        M["R1_OI_UNWIND"] = (f1(-3, 15), [f1(o, d) for o in (-1, -2, -3, -5, -8) for d in (10, 15, 25)])
        f2 = lambda o, s: g((oi1 >= o) & (np.abs(r1) <= s) & (dd <= 10))
        M["R2_OI_TRAP"] = (f2(5, 1.5), [f2(o, s) for o in (3, 5, 8, 12) for s in (1, 1.5, 3)])
        fl, fp = v(X, "f_last8"), v(X, "f_prev8")
        f3 = lambda p, d: g((fp >= p) & (fl < fp) & (dd <= d))
        M["R3_FUND_TURN"] = (f3(0.05, 20), [f3(p, d) for p in (0.01, 0.03, 0.05, 0.10) for d in (10, 20, 30)])
        tp, ga = v(X, "topP_chg1h"), v(X, "glA_chg1h")
        f4 = lambda c, d: g((tp <= c) & (ga >= 0) & (dd <= d))
        M["R4_TOP_DISTRIB"] = (f4(-5, 20), [f4(c, d) for c in (-2, -5, -10) for d in (10, 20, 30)])
        t1, t0 = v(X, "taker1h"), v(X, "taker_prev1h")
        f5 = lambda t, d: g((t1 <= t) & (t0 >= 1.0) & (dd <= d))
        M["R5_TAKER_FADE"] = (f5(0.90, 15), [f5(t, d) for t in (0.80, 0.85, 0.90, 0.95) for d in (10, 15, 25)])
        pm, pk = v(X, "prem1h"), v(X, "prem_pk")
        f6 = lambda p, fr: g((pk >= p) & (pm <= fr * pk) & (dd <= 20))
        M["R6_PREM_COLLAPSE"] = (f6(0.25, 0.25), [f6(p, fr) for p in (0.15, 0.25, 0.50) for fr in (0.10, 0.25, 0.50)])
        age, lh, vg = v(X, "peak_age_h"), v(X, "lh2"), v(X, "vwap_gap")
        f7 = lambda l_, a: g((age >= a) & (lh <= l_) & (vg < 0) & (dd >= 8) & (dd <= 30))
        M["R7_LH_VWAP"] = (f7(0.95, 2), [f7(l_, a) for l_ in (0.97, 0.95, 0.90) for a in (1, 2, 4)])
        f8 = lambda o, r: g((oi1 <= o) & (r1 <= r))
        M["R8_CASCADE"] = (f8(-8, -5), [f8(o, r) for o in (-5, -8, -12) for r in (-3, -5, -8)])
        gl, fla = v(X, "glA"), v(X, "f_last")
        f9 = lambda L, d: g((dd >= d) & (age >= 1) & (gl >= L) & (fla >= 0) & (oi1 <= 0))
        M["R9_CROWD_LONG"] = (f9(1.5, 8), [f9(L, d) for L in (1.2, 1.5, 2.0, 2.5) for d in (5, 8, 15)])
        om = v(X, "oi_vs_max")
        f10 = lambda fr, d: g((om <= fr) & (dd <= d))
        M["R10_OI_FROM_MAX"] = (f10(0.85, 20), [f10(fr, d) for fr in (0.95, 0.90, 0.85, 0.80) for d in (10, 20, 30)])
    return M


def ns0_mask(X):
    return X.rsx15.values.astype(bool) & (np.nan_to_num(X.vd.values, nan=-1) >= 0.6366)


def one_pos(X, idx, e):
    idx = np.asarray(idx)
    if not len(idx):
        return idx
    sub = idx[np.lexsort((X.entry_ms.values[idx], X.pair.values[idx]))]
    P = X.pair.values[sub]; T = X.entry_ms.values[sub]; XX = X[f"{e}_x"].values[sub]
    keep = np.zeros(len(sub), bool); lp = None; busy = -1
    for i in range(len(sub)):
        if P[i] != lp:
            lp = P[i]; busy = -1
        if T[i] < busy:
            continue
        keep[i] = True; busy = XX[i]
    out = sub[keep]
    return out[np.argsort(X.entry_ms.values[out], kind="stable")]


def first_per_ep(X, idx):
    idx = np.asarray(idx)
    if not len(idx):
        return idx
    d = pd.DataFrame({"i": idx, "e": X.epi.values[idx], "t": X.entry_ms.values[idx]}).sort_values("t")
    return d.drop_duplicates("e").i.values


def wilson_floor(k, n, z=1.645):
    if n <= 0:
        return 0.0
    p = k / n; den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0))) / den


def kelly(vv, days):
    w = vv[vv > 0]; lo = vv[vv <= 0]
    aw = w.mean() if len(w) else 0.0; al = -lo.mean() if len(lo) else 0.0
    nd = len(np.unique(days))
    pf = wilson_floor((vv > 0).mean() * nd, nd)
    b = aw / al if al > 0 else np.inf
    return pf - (1 - pf) / b, pf * 100, (al / (aw + al) * 100 if aw + al else np.nan)


def cal_days(blk):
    if blk == "DESIGN":
        s, e = DES
    elif blk == "HOLDOUT":
        s, e = HOLD
    else:
        s, e = DES[0], HOLD[1]
    return (TS(e) - TS(s)) // DAY, TS(s) // DAY, TS(e) // DAY


def stats(X, idx, e, blk="ALL"):
    idx = np.asarray(idx)
    D, d0, d1 = cal_days(blk)
    rec = dict(N=len(idx), tpd=len(idx) / D)
    if len(idx) == 0:
        return rec
    vv = X[f"{e}_net"].values[idx].astype(float); days = X.day.values[idx]
    f, pf, be = kelly(vv, days)
    sv = np.sort(vv); tot = vv.sum()
    pp = pd.Series(vv).groupby(X.pair.values[idx]).sum(); dd_ = pd.Series(vv).groupby(days).sum()
    r = pd.Series(vv * SIZE / 100).groupby(days).sum().reindex(np.arange(d0, d1), fill_value=0.0)
    eq = np.cumprod(1 + r.values); g = eq[-1]
    mdd = (1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]).max() * 100
    rec.update(WR=(vv > 0).mean() * 100, BE=be, avg=vv.mean(), f20=X[f"{e}_net20"].values[idx].mean(),
               b5=sv[: int(np.floor(len(vv) * 0.95))].mean() if len(vv) >= 20 else np.nan, kelly=f, WRfloor=pf,
               ndays=len(np.unique(days)), pair_share=pp.max() / tot * 100 if tot > 0 else np.nan, pair_top=pp.idxmax(),
               day_share=dd_.max() / tot * 100 if tot > 0 else np.nan, npairs=len(pp),
               dcr=(g ** (1 / D) - 1) * 100 if g > 0 else np.nan, maxdd=mdd, worst_day=r.min() * 100,
               worked=X[f"{e}_worked"].values[idx].mean() * 100, tp=(X[f"{e}_k"].values[idx] == 0).mean() * 100,
               sl=(X[f"{e}_k"].values[idx] == 1).mean() * 100, fund=X[f"{e}_f"].values[idx].mean())
    for h, _, _ in HALVES:
        m = X.half.values[idx] == h
        rec[f"h_{h}_N"] = int(m.sum()); rec[f"h_{h}"] = vv[m].mean() if m.any() else np.nan
    return rec


def path_group(X, idx, e):
    """path_labels groups (5m): losers NEVER_WORKED / STOPPED_RECOVERED / GAVE_BACK / DRIFT_LOSS; winners LEFT_ON_TABLE / CAPTURED."""
    vv = X[f"{e}_net"].values[idx]; wk = X[f"{e}_worked"].values[idx] == 1; k = X[f"{e}_k"].values[idx]
    rec_ = X[f"{e}_rec"].values[idx] == 1; best = X[f"{e}_best"].values[idx]
    g = np.where(vv > 0, np.where((best - 0.09 - vv) >= np.maximum(1.0, vv), "LEFT_ON_TABLE", "CAPTURED"),
                 np.where(wk, "GAVE_BACK", np.where((k == 1) & rec_, "STOPPED_RECOVERED",
                                                     np.where(k == 1, "NEVER_WORKED", "DRIFT_LOSS"))))
    return g
