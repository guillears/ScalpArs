#!/usr/bin/env python3
"""ROUTE_B_PEAK_STUDY 2026-11-21 — evaluation (labels, layer-1 variables, candle comparators, exhaustive screen + shuffled null,
reached-X split, exit counterfactual, event-study chart). Research only. Design frozen in reports/ROUTE_B_PEAK_STUDY_2026-11-21_PREREG.md.

Input: reports/ROUTE_B_SIGNALS_data/peak_1121/panels.pkl (scripts/route_b_peak_1121_data.py).
Output: reports/ROUTE_B_PEAK_STUDY_2026-11-21_tables.csv · _screen.csv · _chart.png · peak_1121/results.pkl
Usage: venv/bin/python -I scripts/route_b_peak_1121_eval.py"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "reports", "ROUTE_B_SIGNALS_data", "peak_1121")
REP = os.path.join(ROOT, "reports")
MIN = 60_000
TAKER = 0.045
ELIG_MFE = 1.0
RNG = np.random.default_rng(1121)
L1 = ["TAKER_FADE", "AGAINST_BURST", "BOOK_FLIP", "SPREAD_WIDE", "CAPITULATION_LIQ", "OI_NO_NEW_MONEY", "BIG_AGAINST", "PREMIUM_FADE"]
CANDLE = ["RSI_ROLL", "LH2", "CLIMAX"]
SL_SHORT = {"EXHAUSTION_SHORT": "EXH", "DISTRIBUTION_SHORT": "DIST", "FLATBASE_SHORT": "FB", "FRENZY_WILLY_HT": "HT", "FRENZY_WILLY": "WILLY"}


# ─────────────────────────────── helpers (all causal: value at i uses rows ≤ i)
def rsum(x, w):
    x = np.nan_to_num(np.asarray(x, float))
    cs = np.r_[0.0, np.cumsum(x)]
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = cs[w:] - cs[:-w]
    return out


def lagged(x, n):
    y = np.full(len(x), np.nan)
    if n < len(x):
        y[n:] = x[:-n]
    return y


def run_median(x, i0, min_n=5):
    """median of x[i0 .. i−1] for every i (in-trade baseline strictly before i); NaN with < min_n non-NaN values."""
    import bisect
    out = np.full(len(x), np.nan)
    buf = []
    for i in range(i0 + 1, len(x)):
        v = x[i - 1]
        if v == v:
            bisect.insort(buf, float(v))
        k = len(buf)
        if k >= min_n:
            out[i] = buf[k // 2] if k % 2 else 0.5 * (buf[k // 2 - 1] + buf[k // 2])
    return out


def run_max(x, i0):
    out = np.full(len(x), np.nan)
    m = -np.inf
    for i in range(i0, len(x)):
        if x[i] == x[i]:
            m = max(m, x[i])
        out[i] = m if m > -np.inf else np.nan
    return out


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / n, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, 100 - 100 / (1 + up / dn))
    r[:n] = np.nan
    return r


def bars5(P):
    t = P.t.values
    bid = t // (5 * MIN)
    u, first, cnt = np.unique(bid, return_index=True, return_counts=True)
    ok = cnt == 5
    first = first[ok]
    last = first + 4
    o = P.o.values[first]; c = P.c.values[last]
    h = np.array([P.h.values[a:a + 5].max() for a in first]); l = np.array([P.l.values[a:a + 5].min() for a in first])
    qv = np.array([P.qv.values[a:a + 5].sum() for a in first])
    return dict(o=o, h=h, l=l, c=c, qv=qv, avail=last)


def to_min(n, avail, vals):
    j = np.searchsorted(avail, np.arange(n), "right") - 1
    return np.where(j >= 0, vals[np.maximum(j, 0)], np.nan) if len(avail) else np.full(n, np.nan)


# ─────────────────────────────── one trade → labels + variables
def build_trade(pn):
    m, P = pn["meta"], pn["P"].copy()
    s = int(m["side"])
    e = float(m["entry_px"])
    t = P.t.values
    e0 = m["entry_ms"] - m["entry_ms"] % MIN
    x0 = m["exit_ms"] - m["exit_ms"] % MIN
    n = len(P)
    ins = (t > e0) & (t <= x0)                   # in-trade minutes (exit minute included for MFE only)
    dec = (t > e0) & (t < x0)                    # decision minutes: their close is before the exit
    if dec.sum() < 1:
        return None
    i0 = int(np.argmax(ins))
    fav = np.where(s > 0, (P.h.values / e - 1) * 100, (1 - P.l.values / e) * 100)
    cpl = s * (P.c.values / e - 1) * 100
    tp_exit = "TP" in str(m["exit_reason"]).upper()
    di = np.where(dec)[0]
    fav_dec = np.where(dec, fav, -np.inf)
    mfe = fav_dec.max()
    p = int(np.argmax(fav_dec))
    tp = m.get("tp_pct")
    if tp_exit:
        tp_lvl = tp if tp == tp else max(mfe, 0)
        if tp_lvl >= mfe:
            mfe, p = float(tp_lvl), int(np.where(ins)[0][-1])      # the top was the TP fill in the exit minute (no decision minute)
    peak = np.zeros(n, bool)
    peak[[k for k in (p, p + 1, p + 2) if k < n and dec[k]]] = True
    cont = dec & (np.arange(n) <= p - 3) & (cpl >= max(ELIG_MFE, 0.5 * mfe)) & (mfe - cpl >= 0.5)
    # ── Route B oriented series
    buy, sell = P.buy_usd.values.astype(float), P.sell_usd.values.astype(float)
    tot = buy + sell
    favu, oppu = (buy, sell) if s > 0 else (sell, buy)
    with np.errstate(divide="ignore", invalid="ignore"):
        fs1 = favu / tot
        fs3 = rsum(favu, 3) / rsum(tot, 3)
        fs5 = rsum(favu, 5) / rsum(tot, 5)
        fs10 = rsum(favu, 10) / rsum(tot, 10)
        fs_prior = (rsum(favu, 13) - rsum(favu, 3)) / (rsum(tot, 13) - rsum(tot, 3))
        opp_sh = oppu / tot
    tot = np.where(np.isnan(P.buy_usd.values) & np.isnan(P.sell_usd.values), np.nan, tot)
    imb = s * P.imb_05.values.astype(float)
    imb3 = pd.Series(imb).rolling(3, min_periods=2).mean().values
    spread = P.spread_bps.values.astype(float)
    liq_cap = P.liq_short_usd.values if s > 0 else P.liq_long_usd.values      # the OTHER side's forced closes (push price our way)
    liq_opp = P.liq_long_usd.values if s > 0 else P.liq_short_usd.values
    with np.errstate(divide="ignore", invalid="ignore"):
        prem = s * (P["mark"].values / P["index"].values - 1) * 100
    fund = s * P.funding.values.astype(float) * 100
    mx = P.max_trade_usd.values.astype(float)
    oi = P.oi.values.astype(float)
    V = {}
    oppmed = run_median(np.where(dec | ins, oppu, np.nan), i0)
    sprmed = run_median(np.where(ins, spread, np.nan), i0)
    mxmed = run_median(np.where(ins, mx, np.nan), i0)
    V["TAKER_FADE"] = (fs3 < 0.45) & (fs_prior - fs3 >= 0.10)
    V["AGAINST_BURST"] = (oppu >= 3 * oppmed) & (opp_sh >= 0.60)
    V["BOOK_FLIP"] = imb3 <= -0.20
    V["SPREAD_WIDE"] = spread >= 2 * sprmed
    V["CAPITULATION_LIQ"] = rsum(liq_cap, 3) >= 5000
    rmx = run_max(np.where(ins, fav, np.nan), i0)
    newext = np.zeros(n, bool)
    for i in range(i0, n):
        a = max(i0, i - 14)
        w = fav[a:i + 1]
        newext[i] = len(w) and np.nanmax(w) >= rmx[i] - 1e-12
    V["OI_NO_NEW_MONEY"] = newext & ((oi - lagged(oi, 15)) <= 0)
    V["BIG_AGAINST"] = (mx >= 5 * mxmed) & (opp_sh >= 0.60)
    pmx = run_max(np.where(ins, prem, np.nan), i0)
    V["PREMIUM_FADE"] = (pmx >= 0.05) & (prem <= pmx - 0.10)
    # availability masks (NaN inputs → the variable cannot fire; coverage counted separately)
    avail = dict(TAKER_FADE=~np.isnan(fs3) & ~np.isnan(fs_prior), AGAINST_BURST=~np.isnan(oppmed) & ~np.isnan(opp_sh),
                 BOOK_FLIP=~np.isnan(imb3), SPREAD_WIDE=~np.isnan(sprmed) & ~np.isnan(spread), CAPITULATION_LIQ=~np.isnan(rsum(liq_cap, 3)),
                 OI_NO_NEW_MONEY=~np.isnan(oi - lagged(oi, 15)), BIG_AGAINST=~np.isnan(mxmed) & ~np.isnan(mx),
                 PREMIUM_FADE=~np.isnan(prem))
    # ── candle comparators (STATE_EXIT definitions, 5m clock bars known at their last minute)
    B = bars5(P)
    if len(B["c"]) > 20:
        r5 = to_min(n, B["avail"], rsi(B["c"]))
        r5in = np.where(ins, r5, np.nan)
        if s > 0:
            rx = run_max(r5in, i0)
            V["RSI_ROLL"] = (rx >= 75) & (r5 <= rx - 10)
        else:
            rn = -run_max(-r5in, i0)
            V["RSI_ROLL"] = (rn <= 25) & (r5 >= rn + 10)
        ex = B["h"] if s > 0 else -B["l"]
        lower = np.r_[False, ex[1:] < ex[:-1]]
        run = np.zeros(len(lower))
        for k in range(1, len(lower)):
            run[k] = run[k - 1] + 1 if lower[k] else 0
        V["LH2"] = to_min(n, B["avail"], run) >= 2
        v20 = lagged(rsum(B["qv"], 20) / 20, 1)
        against = (B["c"] < B["o"]) if s > 0 else (B["c"] > B["o"])
        cb = to_min(n, B["avail"], (against & (B["qv"] >= 2 * v20)).astype(float)) > 0
        cext = to_min(n, B["avail"], B["h"] if s > 0 else B["l"])
        extpx = e * (1 + s * rmx / 100)
        V["CLIMAX"] = cb & (np.abs(cext / extpx - 1) * 100 <= 1.0)
    else:
        for k in CANDLE:
            V[k] = np.zeros(n, bool)
    for k in CANDLE:
        avail[k] = np.ones(n, bool)
    # screen base series (oriented) — kept for layer 2
    with np.errstate(divide="ignore", invalid="ignore"):
        bidd = {b: P[f"bid_usd_{b}"].values.astype(float) for b in ("02", "05", "1")}
        askd = {b: P[f"ask_usd_{b}"].values.astype(float) for b in ("02", "05", "1")}
        favd = {b: (bidd[b] if s > 0 else askd[b]) for b in bidd}       # resting depth on the side that supports our move
        oppd = {b: (askd[b] if s > 0 else bidd[b]) for b in bidd}       # resting depth against our move
        exitc = {z: P[f"cost_{'sell' if s > 0 else 'buy'}_{z}"].values.astype(float) for z in ("2500", "10000")}
        entc = {z: P[f"cost_{'buy' if s > 0 else 'sell'}_{z}"].values.astype(float) for z in ("2500", "10000")}
        rng_ = (P.mid_hi.values - P.mid_lo.values) / P.mid.values * 100
        base = dict(spread=spread, imb=imb, fs1=fs1, fs3=fs3, fs5=fs5, fs10=fs10, tot_usd=tot, n_trades=P.n_trades.values.astype(float),
                    opp_usd=oppu, fav_usd=favu, max_trade=mx, liq_cap3=rsum(liq_cap, 3), liq_cap10=rsum(liq_cap, 10),
                    liq_opp3=rsum(liq_opp, 3), liq_opp10=rsum(liq_opp, 10), prem=prem, fund=fund, mid_rng=rng_,
                    **{f"favdep{b}": favd[b] for b in favd}, **{f"oppdep{b}": oppd[b] for b in oppd},
                    **{f"deplr{b}": np.log(favd[b] / oppd[b]) for b in favd},
                    **{f"exitcost{z}": exitc[z] for z in exitc}, **{f"entcost{z}": entc[z] for z in entc}, oi=oi)
    return dict(m=m, s=s, n=n, t=t, dec=dec, ins=ins, peak=peak, cont=cont, p=p, mfe=mfe, cpl=cpl, fav=fav, tp_exit=tp_exit, i0=i0,
                V={k: np.asarray(v, bool) for k, v in V.items()}, avail=avail, base=base, has_book=P.has_book.values.astype(bool),
                exitslip=np.where(np.isnan(exitc["2500"]), np.nan, np.maximum(exitc["2500"] / 100, 0.02)),
                has_oi=~np.isnan(oi))


# ─────────────────────────────── precision (exact episode definition, layer 1)
def episodes(f):
    """list of (a, b) index runs where f is True."""
    if not f.any():
        return []
    d = np.diff(np.r_[0, f.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def precision_exact(T, var, shift=None):
    hits = fa = 0
    hit_w, fa_w, hit_pairs = [], [], []
    for j, tr in enumerate(T):
        di = np.where(tr["dec"])[0]
        f = tr["V"][var][di] & tr["avail"][var][di]
        pk, ct = tr["peak"][di], tr["cont"][di]
        if shift is not None:
            f = np.roll(f, shift[j])
        tt = tr["t"][di]
        if (f & pk).any():
            hits += 1
            hit_w.append(int(tt[np.argmax(f & pk)] // (30 * MIN)))
            hit_pairs.append(tr["m"]["pair"])
        for a, b in episodes(f):
            if ct[a:b + 1].any():
                fa += 1
                fa_w.append(int(tt[a + np.argmax(ct[a:b + 1])] // (30 * MIN)))
    return hits, fa, hit_w, fa_w, hit_pairs


def layer1(T, var, nshuf=1000):
    h, fa, hw, fw, hp = precision_exact(T, var)
    rec_den = sum(1 for tr in T if tr["peak"].any())
    prec = h / (h + fa) if h + fa else np.nan
    null = []
    if h + fa >= 5:
        for _ in range(nshuf):
            sh = [int(RNG.integers(0, max(1, tr["dec"].sum()))) for tr in T]
            a, b, *_ = precision_exact(T, var, sh)
            null.append(a / (a + b) if a + b else np.nan)
    null = np.array(null, float)
    q95 = np.nanquantile(null, 0.95) if np.isfinite(null).any() else np.nan
    conc = (pd.Series(hp).value_counts().iloc[0] / len(hp)) if hp else np.nan
    dm = np.concatenate([tr["dec"][tr["dec"]] for tr in T]).size if T else 0
    fired = sum(int((tr["V"][var] & tr["avail"][var] & tr["dec"]).sum()) for tr in T)
    cov = sum(int((tr["avail"][var] & tr["dec"]).sum()) for tr in T)
    pk_rate = np.mean([x for tr in T for x in (tr["V"][var] & tr["avail"][var])[tr["peak"]]]) if any(tr["peak"].any() for tr in T) else np.nan
    ct_rate = np.mean([x for tr in T for x in (tr["V"][var] & tr["avail"][var])[tr["cont"]]]) if any(tr["cont"].any() for tr in T) else np.nan
    return dict(var=var, trades=len(T), peak_trades=rec_den, hits=h, false_alarms=fa, episodes=h + fa, precision=prec,
                recall=h / rec_den if rec_den else np.nan, null_q95=q95, p_null=float(np.nanmean(null >= prec)) if prec == prec and len(null) else np.nan,
                windows=len(set(hw) | set(fw)), hit_windows=len(set(hw)), top_pair_share=conc,
                base_rate=fired / cov if cov else np.nan, coverage=cov / dm if dm else np.nan, minute_rate_peak=pk_rate,
                minute_rate_cont=ct_rate)


# ─────────────────────────────── reached-X split + counterfactual
def split_and_cf(T, var):
    rows = []
    for tr in T:
        m = tr["m"]
        tp = m.get("tp_pct")
        if not tp == tp:
            continue
        X = tp / 2
        if tr["mfe"] < X:
            continue
        di = np.where(tr["dec"])[0]
        reach = di[tr["cpl"][di] >= X]
        if not len(reach):
            continue
        k0 = reach[0]
        end = min(tr["p"] + 2, di[-1])
        f = tr["V"][var] & tr["avail"][var]
        fired = bool(f[k0:end + 1].any()) if end >= k0 else False
        rows.append(dict(group="WENT_TP" if tr["tp_exit"] else "GAVE_BACK", fired=fired))
    d = pd.DataFrame(rows)
    out = {}
    for g in ("WENT_TP", "GAVE_BACK"):
        x = d[d.group == g] if len(d) else d
        out[g] = (int(x.fired.sum()) if len(x) else 0, len(x))
    return out


def cf(T, var):
    rows = []
    for tr in T:
        m = tr["m"]
        tp = m.get("tp_pct")
        X = tp / 2 if tp == tp else 2.5
        di = np.where(tr["dec"])[0]
        f = tr["V"][var] & tr["avail"][var]
        k = [i for i in di if tr["cpl"][i] >= X and f[i]]
        today = float(m["net_pct"])
        if not k:
            rows.append(dict(pair=m["pair"], day=pd.Timestamp(m["exit_ms"], unit="ms").date(), today=today, new=today, d=0.0, kind=m["exit_reason"]))
            continue
        i = k[0]
        slip = tr["exitslip"][i] if tr["exitslip"][i] == tr["exitslip"][i] else 0.10
        new = tr["cpl"][i] - 2 * TAKER - slip
        rows.append(dict(pair=m["pair"], day=pd.Timestamp(tr["t"][i], unit="ms").date(), today=today, new=new, d=new - today, kind=m["exit_reason"]))
    return pd.DataFrame(rows)


# ─────────────────────────────── layer 2 screen (vectorised; false alarms = runs of fire inside the CONTINUATION zone)
def screen(T, nshuf=300, label=""):
    if not T:
        return pd.DataFrame(), np.nan
    di = [np.where(tr["dec"])[0] for tr in T]
    lens = np.array([len(x) for x in di])
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    N = lens.sum()
    PK = np.concatenate([tr["peak"][d] for tr, d in zip(T, di)])
    CT = np.concatenate([tr["cont"][d] for tr, d in zip(T, di)])
    first = np.zeros(N, bool); first[starts] = True
    feats = {}
    for name in T[0]["base"]:
        raw = [tr["base"][name] for tr in T]
        lv = np.concatenate([r[d] for r, d in zip(raw, di)])
        feats[f"{name}"] = lv
        d3 = np.concatenate([(r - lagged(r, 3))[d] for r, d in zip(raw, di)])
        d10 = np.concatenate([(r - lagged(r, 10))[d] for r, d in zip(raw, di)])
        feats[f"{name}_d3"] = d3
        feats[f"{name}_d10"] = d10
        if name not in ("imb", "fs1", "fs3", "fs5", "fs10", "prem", "fund", "deplr02", "deplr05", "deplr1"):
            rel = []
            for tr, r, d in zip(T, raw, di):
                med = run_median(np.where(tr["ins"], r, np.nan), tr["i0"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel.append((r / med)[d])
            feats[f"{name}_rel"] = np.concatenate(rel)
    tests, names = [], []
    for fn, x in feats.items():
        ok = np.isfinite(x)
        if ok.sum() < 50 or np.nanstd(x) == 0:
            continue
        qs = np.nanquantile(x[ok], [0.2, 0.33, 0.67, 0.8])
        for q, op, lab in ((qs[0], "le", "q20"), (qs[1], "le", "q33"), (qs[2], "ge", "q67"), (qs[3], "ge", "q80")):
            tests.append(ok & ((x <= q) if op == "le" else (x >= q)))
            names.append((fn, f"{'≤' if op == 'le' else '≥'}{lab}", float(q)))
    n1 = len(tests)
    med = {fn: (np.isfinite(x), x >= np.nanmedian(x)) for fn, x in feats.items()
           if (fn in T[0]["base"] or fn.endswith("_d10")) and np.isfinite(x).sum() >= 50 and np.nanstd(x) > 0}   # 2D: levels + 10-min changes
    keys = list(med)
    for a, b in itertools.combinations(keys, 2):
        oka, ga = med[a]; okb, gb = med[b]
        ok = oka & okb
        for sa, sb in ((1, 1), (1, 0), (0, 1), (0, 0)):
            tests.append(ok & (ga == bool(sa)) & (gb == bool(sb)))
            names.append((f"{a} × {b}", f"{'hi' if sa else 'lo'}/{'hi' if sb else 'lo'}", np.nan))
    F = np.array(tests, dtype=bool)
    tid = np.repeat(np.arange(len(T)), lens)

    def stat(PKs, CTs):
        hits = np.zeros(len(F), int)
        for j, (a, L) in enumerate(zip(starts, lens)):
            pk = PKs[a:a + L]
            if pk.any():
                hits += F[:, a:a + L][:, pk].any(axis=1)
        x = F & CTs[None, :]
        prev = np.zeros_like(x); prev[:, 1:] = x[:, :-1]; prev[:, first] = False
        fa = (x & ~prev).sum(axis=1)
        return hits, fa

    h, fa = stat(PK, CT)
    ep = h + fa
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(ep > 0, h / ep, np.nan)
    rank = ep >= 5
    fw = []
    for _ in range(nshuf):
        sh = RNG.integers(0, np.maximum(lens, 1))
        idx = np.concatenate([a + (np.arange(L) - k) % L for a, L, k in zip(starts, lens, sh)])
        hs, fs = stat(PK[idx], CT[idx])
        e2 = hs + fs
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.where(e2 >= 5, hs / e2, np.nan)
        fw.append(np.nanmax(pr) if np.isfinite(pr).any() else np.nan)
    q95 = float(np.nanquantile(fw, 0.95)) if np.isfinite(fw).any() else np.nan
    res = pd.DataFrame(dict(feature=[x[0] for x in names], cond=[x[1] for x in names], thr=[x[2] for x in names],
                            dim=["1D"] * n1 + ["2D"] * (len(names) - n1), hits=h, false_alarms=fa, episodes=ep, precision=prec))
    res["ranked"] = rank
    res["fw_null_q95"] = q95
    res["survivor"] = rank & (res.precision > q95)
    res["cohort"] = label
    return res, q95


# ─────────────────────────────── event study
EV_VARS = [("fs3", "our-side taker share (3 min)"), ("imb", "book imbalance ±0.5 % (our side +)"), ("spread_rel", "spread ÷ in-trade median"),
           ("opp_rel", "opposite taker $ ÷ in-trade median"), ("prem", "perp premium, our side (%)"), ("liq_cap3", "other-side liquidations 3 min ($k)")]


def event_rows(T, offs=range(-20, 11)):
    rows = []
    for tr in T:
        b = tr["base"]
        sprmed = run_median(np.where(tr["ins"], b["spread"], np.nan), tr["i0"])
        oppmed = run_median(np.where(tr["ins"], b["opp_usd"], np.nan), tr["i0"])
        with np.errstate(divide="ignore", invalid="ignore"):
            ser = dict(fs3=b["fs3"], imb=b["imb"], spread_rel=b["spread"] / sprmed, opp_rel=b["opp_usd"] / oppmed, prem=b["prem"],
                       liq_cap3=b["liq_cap3"] / 1000)
        anchors = []
        pk = np.where(tr["peak"])[0]
        if tr["p"] < tr["n"] and tr["dec"][tr["p"]]:
            anchors.append(("PEAK", tr["p"]))
        ct = np.where(tr["cont"])[0]
        if len(ct):
            anchors.append(("CONTINUATION", int(RNG.choice(ct))))
        for lab, a in anchors:
            for o in offs:
                i = a + o
                if 0 <= i < tr["n"]:
                    rows.append(dict(dirn="LONG" if tr["s"] > 0 else "SHORT", anchor=lab, off=o, **{k: v[i] for k, v in ser.items()}))
    return pd.DataFrame(rows)


def chart(E, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(EV_VARS), figsize=(22, 7.5), sharex=True)
    for r, dirn in enumerate(("SHORT", "LONG")):
        for c, (v, lab) in enumerate(EV_VARS):
            a = ax[r, c]
            for anc, col in (("PEAK", "#c0392b"), ("CONTINUATION", "#2471a3")):
                x = E[(E.dirn == dirn) & (E.anchor == anc)]
                if not len(x):
                    continue
                g = x.groupby("off")[v]
                mu = g.median()
                q1, q3 = g.quantile(0.25), g.quantile(0.75)
                nn = x[x.off == 0][v].notna().sum()
                a.plot(mu.index, mu.values, color=col, lw=2, label=f"{anc} (n={nn})")
                a.fill_between(mu.index, q1.values, q3.values, color=col, alpha=0.12)
            a.axvline(0, color="k", lw=0.8, ls=":")
            a.set_title(f"{dirn}S · {lab}" if c == 0 else lab, fontsize=9)
            if r == 1:
                a.set_xlabel("min from anchor (0 = anchor close)", fontsize=8)
            if c == 0:
                a.legend(fontsize=8)
    fig.suptitle("Route B order flow around the in-trade PEAK vs a CONTINUATION moment (median, IQR band) — REAL + SHADOW fills, "
                 "pooled by direction (descriptive only)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    D = pd.read_pickle(os.path.join(DATA, "panels.pkl"))
    T = []
    for pn in D["panels"]:
        if not pn.get("ok"):
            continue
        tr = build_trade(pn)
        if tr is not None:
            T.append(tr)
    # coverage
    cov = []
    for tr in T:
        m = tr["m"]
        d = tr["dec"]
        pk = tr["peak"]
        cov.append(dict(sleeve=m["sleeve"], cohort=m["cohort"], pair=m["pair"], mfe=tr["mfe"], eligible=tr["mfe"] >= ELIG_MFE,
                        dec_min=int(d.sum()), book_dec=float(tr["has_book"][d].mean()), oi_dec=float(tr["has_oi"][d].mean()),
                        peak_in_dec=bool(pk.any()), book_at_peak=bool(tr["has_book"][pk].any()) if pk.any() else False,
                        tape_at_peak=bool(np.isfinite(tr["base"]["tot_usd"][pk]).any()) if pk.any() else False,
                        oi_at_peak=bool(tr["has_oi"][pk].any()) if pk.any() else False, has_cont=bool(tr["cont"].any()),
                        tp_exit=tr["tp_exit"], net=m["net_pct"], exit=m["exit_reason"], day=pd.Timestamp(m["entry_ms"], unit="ms").date()))
    COV = pd.DataFrame(cov)
    out_rows, scr_all = [], []
    groups = [(sl, co) for sl in ("EXHAUSTION_SHORT", "DISTRIBUTION_SHORT", "FLATBASE_SHORT", "FRENZY_WILLY_HT") for co in ("REAL", "SHADOW", "REAL+SHADOW")]
    groups += [("POOLED_SHORTS", "REAL+SHADOW"), ("POOLED_LONGS", "REAL+SHADOW")]
    cfs = {}
    for sl, co in groups:
        def keep(tr):
            m = tr["m"]
            if tr["mfe"] < ELIG_MFE or not tr["has_book"][tr["dec"]].any():
                return False
            if sl == "POOLED_SHORTS":
                ok_s = tr["s"] < 0
            elif sl == "POOLED_LONGS":
                ok_s = tr["s"] > 0
            else:
                ok_s = m["sleeve"] == sl
            return ok_s and (co == "REAL+SHADOW" or m["cohort"] == co)
        G = [tr for tr in T if keep(tr)]
        for var in L1 + CANDLE:
            r = layer1(G, var, nshuf=300 if G else 0)
            sp = split_and_cf(G, var)
            r.update(sleeve=sl, cohort=co, tp_fired=sp["WENT_TP"][0], tp_n=sp["WENT_TP"][1], gb_fired=sp["GAVE_BACK"][0], gb_n=sp["GAVE_BACK"][1],
                     layer="L1" if var in L1 else "candle")
            if r["precision"] == r["precision"] and r["precision"] >= 0.60 and r["episodes"] >= 5:
                C = cf(G, var)
                cfs[(sl, co, var)] = C
                r.update(cf_fired=int((C.d != 0).sum()), cf_mean_d=C.d.mean(), cf_saved=C.d[C.d > 0].sum(), cf_chopped=-C.d[C.d < 0].sum(),
                         cf_loo_min=min((C.d.drop(i).mean() for i in C.index), default=np.nan),
                         cf_loo_max=max((C.d.drop(i).mean() for i in C.index), default=np.nan))
            out_rows.append(r)
        if co != "REAL+SHADOW" and sl.startswith(("EXH", "DIST", "FLAT", "FRENZY")) and len(G) >= 3:
            res, q = screen(G, nshuf=200, label=f"{sl}|{co}")
            if len(res):
                res["sleeve"], res["cohort_"] = sl, co
                scr_all.append(res)
    TAB = pd.DataFrame(out_rows)
    SCR = pd.concat(scr_all, ignore_index=True) if scr_all else pd.DataFrame()
    E = event_rows([tr for tr in T if tr["mfe"] >= ELIG_MFE and tr["has_book"][tr["dec"]].any()])
    chart(E, os.path.join(REP, "ROUTE_B_PEAK_STUDY_2026-11-21_chart.png"))
    TAB.to_csv(os.path.join(REP, "ROUTE_B_PEAK_STUDY_2026-11-21_tables.csv"), index=False)
    if len(SCR):
        top = SCR[SCR.ranked].sort_values(["sleeve", "cohort_", "precision", "episodes"], ascending=[True, True, False, False])
        top.groupby(["sleeve", "cohort_"]).head(25).to_csv(os.path.join(REP, "ROUTE_B_PEAK_STUDY_2026-11-21_screen_top.csv"), index=False)
    pd.to_pickle(dict(TAB=TAB, SCR=SCR, COV=COV, E=E, cfs=cfs), os.path.join(DATA, "results.pkl"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    print(COV.groupby(["sleeve", "cohort"]).agg(n=("pair", "size"), elig=("eligible", "sum"), book=("book_dec", "mean"), peak_dec=("peak_in_dec", "sum"),
                                                 book_pk=("book_at_peak", "sum"), oi_pk=("oi_at_peak", "sum"), cont=("has_cont", "sum")).to_string())
    print(TAB[["sleeve", "cohort", "var", "trades", "peak_trades", "hits", "false_alarms", "precision", "recall", "null_q95", "windows",
               "top_pair_share", "base_rate", "minute_rate_peak", "minute_rate_cont", "tp_fired", "tp_n", "gb_fired", "gb_n"]].round(2).to_string())
    if len(SCR):
        print(SCR.groupby(["sleeve", "cohort_"]).agg(tests=("feature", "size"), ranked=("ranked", "sum"), q95=("fw_null_q95", "first"),
                                                      best=("precision", lambda x: np.nanmax(x[SCR.loc[x.index, "ranked"]]) if SCR.loc[x.index, "ranked"].any() else np.nan),
                                                      surv=("survivor", "sum")).to_string())


if __name__ == "__main__":
    main()
