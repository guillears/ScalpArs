#!/usr/bin/env python3
"""NOSPOT_SLEEVE (PREREG reports/NOSPOT_SLEEVE_2026-11-18_PREREG.md) — BUILD. Per no-spot pair: every CLOSED 15m bar inside a Frenzy window
(allspk_meta episodes, [on_ms, spike_ms + 72 h)), vol24 ≥ $20 M, ATR15 defined, spot_status(pair, month) == 0, no 1000-prefix.
Entry = next 5m open (T). Features at T (lag-safe: 5m / 15m bars closed ≤ T; metrics create_time ≤ T − 5 min; premium bars closed ≤ T;
funding settlements ≤ T). Six exits (TP maker through by 1 tick, gap-aware stop, stop first on a shared bar), real funding, path facts
(path_labels definitions on 5m: WORKED = +1 % before −Y; recovered = +1 % from entry within 120 min after a stop; best to close + 45 min).
Output reports/NOSPOT_SLEEVE_data/rows/<PAIR>.pkl  ·  Usage: venv/bin/python -I scripts/nospot_sleeve_build.py [--pairs A,B] [--procs 7]
       [--meta extra_meta.csv --k5m extra_dir --fund extra_dir --metrics extra_dir --prem extra_dir --out dir]  (live replay)"""
import argparse, glob, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import nospot_mcap_common as C   # noqa: E402
import exh_expand_build as XB   # noqa: E402  (patched K5M / funding sources)
TB = XB.TB
ND = os.path.join(R, "reports", "NOSPOT_SLEEVE_data")
BAR, M15, H1, DAY = 300_000, 900_000, 3_600_000, 86_400_000
WIN = 72 * H1
SLIP_N = 6100.0
EXITS = {  # name: (tp kind, tp mult, tp cap, sl kind, sl mult, sl cap, hold bars)
    "XE": ("atr", 20.0, 10.0, "atr", 1.5, 6.0, 288), "XA": ("atr", 3.0, 5.0, "atr", 1.5, 5.0, 288),
    "F55": ("pct", 5.0, None, "pct", 5.0, None, 288), "F106": ("pct", 10.0, None, "pct", 6.0, None, 288),
    "F1012": ("pct", 10.0, None, "pct", 12.0, None, 288), "F1510": ("pct", 15.0, None, "pct", 10.0, None, 576)}
MET_DIRS = [d for d in glob.glob(os.path.join(R, "reports", "*", "metrics*")) if os.path.isdir(d)]
PREM_DIRS = [d for d in glob.glob(os.path.join(R, "reports", "*", "prem*")) if os.path.isdir(d)]


def met_index(dirs):
    idx = {}
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(".csv") and f.count(".") == 2:
                idx.setdefault(f.split(".")[0], []).append(os.path.join(d, f))
    return idx


def load_metrics(files, days):
    xs = []
    for f in files:
        if os.path.basename(f).split(".")[1] not in days:
            continue
        try:
            x = pd.read_csv(f)
        except Exception:
            continue
        if "create_time" in x.columns and len(x):
            ct = x.create_time
            if not pd.api.types.is_numeric_dtype(ct):
                x["ms"] = pd.to_datetime(ct, format="mixed").astype("datetime64[ms]").astype("int64")
            else:
                x["ms"] = ct.astype("int64")
            x.loc[x.ms > 10**14, "ms"] //= 1000
            xs.append(x)
    if not xs:
        return None
    x = pd.concat(xs)
    return x.drop_duplicates("ms").sort_values("ms").reset_index(drop=True)


def load_prem(files):
    xs = []
    for f in files:
        try:
            x = pd.read_csv(f)
        except Exception:
            continue
        if "open_time" in x.columns and "c" in x.columns:
            xs.append(x[["open_time", "c"]])
    if not xs:
        return None
    x = pd.concat(xs); x["open_time"] = x.open_time.astype("int64")
    x.loc[x.open_time > 10**14, "open_time"] //= 1000
    return x.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def wilder_dm(h, l, c, n=14):
    up = np.r_[0, np.diff(h)]; dn = np.r_[0, -np.diff(l)]
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    a = TB.wilder(tr, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * TB.wilder(pdm, n) / a - 100 * TB.wilder(mdm, n) / a


def sim(o, h, l, c, J, atrp, tick, ex):
    tk, tm, tc, sk, sm, sc, H = EXITS[ex]
    N = len(c); out = {k: [] for k in ("r", "k", "col", "tpd", "sld", "worked", "reach", "rec", "best", "adv")}
    for a0 in range(0, len(J), 1500):
        jj = J[a0:a0 + 1500]; ap = atrp[a0:a0 + 1500]
        HH = H + 24
        idx = jj[:, None] + np.arange(HH)[None, :]; valid = idx < N; idc = np.minimum(idx, N - 1)
        Oo, Hh, Ll, Cc = o[idc], h[idc], l[idc], c[idc]; O = o[jj]; rows = np.arange(len(jj))
        vH = valid[:, :H]
        tpd = ap * tm if tk == "atr" else np.full(len(jj), tm); tpd = np.minimum(tpd, tc) if tc else tpd
        sld = ap * sm if sk == "atr" else np.full(len(jj), sm); sld = np.minimum(sld, sc) if sc else sld
        TPp = O * (1 - tpd / 100); SLp = O * (1 + sld / 100)
        ht = (Ll[:, :H] <= TPp[:, None] - tick) & vH; hs = (Hh[:, :H] >= SLp[:, None]) & vH
        fs = np.where(hs.any(1), hs.argmax(1), 99999); ft = np.where(ht.any(1), ht.argmax(1), 99999)
        last = np.minimum(H - 1, N - 1 - jj)
        stop = (fs <= ft) & (fs < 99999); tp = (ft < fs) & (ft < 99999)
        X = np.where(stop, np.maximum(Oo[rows, np.minimum(fs, H - 1)], SLp), np.where(tp, TPp, Cc[rows, last]))
        col = np.where(stop, fs, np.where(tp, ft, last))
        r = -(X / O - 1) * 100; k = np.where(stop, 1, np.where(tp, 0, 2)).astype(np.int8)
        # path facts (gross, short): fav = (O − low)/O, adv = (high − O)/O ; first passage of +1 % vs −Y (Y = stop distance)
        fav = (1 - Ll[:, :H] / O[:, None]) * 100; adv = (Hh[:, :H] / O[:, None] - 1) * 100
        fav = np.where(vH, fav, -99); adv = np.where(vH, adv, -99)
        f1 = np.where((fav >= 1.0).any(1), (fav >= 1.0).argmax(1), 99999)
        a1 = np.where((adv >= sld[:, None]).any(1), (adv >= sld[:, None]).argmax(1), 99999)
        worked = (f1 < a1).astype(np.int8)          # same bar both → not worked (conservative)
        cm = np.arange(H)[None, :] <= np.minimum(np.minimum(a1, col), H - 1)[:, None]
        reach = np.where(cm, fav, -99).max(1)
        # recovered: after a stop, +1 % from the entry within 24 bars (120 min) after the stop bar
        aft = (np.arange(HH)[None, :] > col[:, None]) & (np.arange(HH)[None, :] <= col[:, None] + 24) & valid
        recov = (stop & (np.where(aft, (1 - Ll / O[:, None]) * 100, -99).max(1) >= 1.0)).astype(np.int8)
        upto = (np.arange(HH)[None, :] <= col[:, None] + 9) & valid
        best = np.where(upto, (1 - Ll / O[:, None]) * 100, -99).max(1)
        advmax = np.where(np.arange(H)[None, :] <= col[:, None], adv, -99).max(1)
        for nm, v in (("r", r), ("k", k), ("col", col), ("tpd", tpd), ("sld", sld), ("worked", worked), ("reach", reach),
                      ("rec", recov), ("best", best), ("adv", advmax)):
            out[nm].append(v)
    return {k_: np.concatenate(v) for k_, v in out.items()}


def asof(ts, T, maxage=None):
    k = np.searchsorted(ts, T, side="right") - 1
    ok = k >= 0
    if maxage is not None:
        ok &= (T - ts[np.maximum(k, 0)]) <= maxage
    return np.where(ok, k, -1)


def work(args):
    p, eps, st, mfiles, pfiles, supply, btc, outdir = args
    fo = os.path.join(outdir, p + ".pkl")
    if os.path.exists(fo):
        return p, -1
    x = TB.load(p)
    if x is None:
        return p, 0
    t, o, h, l, c, v, qv = x
    N = len(t)
    if N < 400:
        return p, 0
    tick = np.diff(np.unique(np.r_[o, h, l, c])); tick = float(tick[tick > 1e-15].min()) if len(tick) else 0.0
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    roll1h = pd.Series(qv).rolling(12, min_periods=12).sum().values; r1n = np.nan_to_num(roll1h)
    q5 = pd.Series(qv).rolling(12, min_periods=6).median().values
    st15, en15, O15, H15, L15, C15, V15 = XB.agg(t, o, h, l, c, v, M15)
    full15 = (en15 - st15) == 2
    A15 = TB.atr(H15, L15, C15, 14); R15 = TB.rsi(C15, 14)
    rsx = np.nan_to_num((TB.sh(R15) > 70) & (R15 <= 70)).astype(bool)
    DI15 = wilder_dm(H15, L15, C15)
    s1, e1, _, H1h, L1h, C1, _ = XB.agg(t, o, h, l, c, v, H1)
    R1h = TB.rsi(C1, 14); full1 = (e1 - s1) == 11
    # candidate bars: 15m closes (index = 5m index of the 15m's last bar)
    cand = en15[full15]; ci = np.nonzero(full15)[0]
    J = cand + 1; ok = J < N
    cand, ci, J = cand[ok], ci[ok], J[ok]
    T = t[J].astype("int64")
    eps = eps.sort_values("on_ms").reset_index(drop=True)
    on = eps.on_ms.values.astype("int64"); sp = eps.spike_ms.values.astype("int64")
    k = np.searchsorted(on, T, side="right") - 1
    inwin = (k >= 0) & (T < sp[np.maximum(k, 0)] + WIN) & (T >= on[np.maximum(k, 0)])
    mo = pd.to_datetime(T, unit="ms").strftime("%Y-%m").values
    nos = np.array([st.get((p, m_), -1) == 0 for m_ in mo])
    sel = inwin & nos & (np.nan_to_num(vol24[cand]) >= 20e6) & np.isfinite(A15[ci]) & (A15[ci] > 0) & (cand >= 300)
    if not sel.any():
        return p, 0
    cand, ci, J, T, k = cand[sel], ci[sel], J[sel], T[sel], k[sel]
    n = len(J)
    E_on = on[k]; E_sp = sp[k]
    jon = np.searchsorted(t, E_on)
    # running high since on (bars closed ≤ T, i.e. up to cand), its time; anchored VWAP; vol decay; lower-high
    dd_hi = np.full(n, np.nan); pk_age = np.full(n, np.nan); vwapg = np.full(n, np.nan); vd = np.full(n, np.nan)
    lh2 = np.full(n, np.nan); maxup1h24 = np.full(n, np.nan); volmult = np.full(n, np.nan); hi_since_on = np.full(n, np.nan)
    tp_ = (h + l + c) / 3 * v; cv = np.r_[0, np.cumsum(v)]; ctp = np.r_[0, np.cumsum(tp_)]
    ret1h_s = np.r_[np.full(12, np.nan), c[12:] / c[:-12] - 1] * 100
    for i in range(n):
        a, b = jon[i], cand[i]
        if b < a:
            continue
        hh = h[a:b + 1]; im = int(np.argmax(hh)); H_ = hh[im]
        hi_since_on[i] = H_
        dd_hi[i] = (1 - c[b] / H_) * 100; pk_age[i] = (T[i] - t[a + im] - BAR) / H1
        lh2[i] = h[max(a, b - 23):b + 1].max() / H_
        vw = (ctp[b + 1] - ctp[a]) / max(cv[b + 1] - cv[a], 1e-12)
        vwapg[i] = (c[b] / vw - 1) * 100 if vw > 0 else np.nan
        pkv = r1n[a:b + 1].max(); vd[i] = roll1h[b] / pkv if pkv > 0 else np.nan
        maxup1h24[i] = np.nanmax(ret1h_s[max(0, b - 287):b + 1]) if b >= 12 else np.nan
    base = pd.Series(qv).shift(288).rolling(288 * 6, min_periods=288).median().values * 12   # 1h vol of the prior week, 1 day lagged
    volmult = roll1h[cand] / np.where(base[cand] > 0, base[cand], np.nan)
    rr = lambda m: (c[cand] / c[np.maximum(cand - m, 0)] - 1) * 100
    out = dict(pair=np.full(n, p), entry_ms=T, on_ms=E_on, spike_ms=E_sp, px=o[J], atrp=A15[ci] / o[J] * 100,
               vol24=vol24[cand], q5=q5[cand], vd=vd, dd_hi=dd_hi, peak_age_h=pk_age, vwap_gap=vwapg, lh2=lh2,
               ret15m=rr(3), ret1h=rr(12), ret4h=rr(48), ret24h=rr(288), rsi15=R15[ci], rsx15=rsx[ci], di15=DI15[ci],
               hrs_since_spike=(T - E_sp) / H1, hrs_since_on=(T - E_on) / H1, maxup1h_24h=maxup1h24, volmult=volmult,
               rise_on_to_hi=(hi_since_on / o[jon] - 1) * 100)
    k1 = np.searchsorted(t[e1], t[cand], side="right") - 1          # last closed 1h bar whose last 5m bar ≤ cand
    ok1 = (k1 >= 0) & full1[np.maximum(k1, 0)]
    out["rsi1h"] = np.where(ok1, R1h[np.maximum(k1, 0)], np.nan)
    out["mcap"] = (supply["cs"] * o[J] / supply["mult"]) if supply else np.full(n, np.nan)
    out["turnover"] = vol24[cand] / out["mcap"]
    # metrics
    days = set()
    for a_, b_ in zip(E_on, T):
        for d_ in pd.date_range(pd.Timestamp(a_ - 2 * H1, unit="ms").normalize(), pd.Timestamp(b_, unit="ms").normalize()):
            days.add(d_.strftime("%Y-%m-%d"))
    M = load_metrics(mfiles, days)
    for nm in ("oi", "oi_chg15m", "oi_chg1h", "oi_chg4h", "oi_vs_max", "oi_usd", "oi_vol", "topP", "topP_chg1h", "topP_chg4h", "topA",
               "topA_chg4h", "glA", "glA_chg1h", "glA_chg4h", "taker1h", "taker_prev1h"):
        out[nm] = np.full(n, np.nan)
    if M is not None and len(M) > 20:
        ms = M.ms.values.astype("int64")
        oi = M.sum_open_interest.values.astype(float); oiv = M.sum_open_interest_value.values.astype(float)
        tP = M.sum_toptrader_long_short_ratio.values.astype(float); tA = M.count_toptrader_long_short_ratio.values.astype(float)
        gA = M.count_long_short_ratio.values.astype(float); tk_ = M.sum_taker_long_short_vol_ratio.values.astype(float)
        kk = asof(ms, T - BAR, 30 * 60_000)
        g = kk >= 0; kg = kk[g]

        def back(m):
            kb = asof(ms, ms[kg] - m, None)
            okb = (kb >= 0) & ((ms[kg] - m) - ms[np.maximum(kb, 0)] <= 15 * 60_000)
            return np.where(okb, kb, -1)

        def chg(arr, m):
            kb = back(m); r_ = np.full(len(kg), np.nan); okb = kb >= 0
            r_[okb] = (arr[kg[okb]] / arr[kb[okb]] - 1) * 100
            return r_
        out["oi"][g] = oi[kg]; out["oi_usd"][g] = oiv[kg]
        out["oi_chg15m"][g] = chg(oi, 15 * 60_000); out["oi_chg1h"][g] = chg(oi, H1); out["oi_chg4h"][g] = chg(oi, 4 * H1)
        out["topP"][g] = tP[kg]; out["topP_chg1h"][g] = chg(tP, H1); out["topP_chg4h"][g] = chg(tP, 4 * H1)
        out["topA"][g] = tA[kg]; out["topA_chg4h"][g] = chg(tA, 4 * H1)
        out["glA"][g] = gA[kg]; out["glA_chg1h"][g] = chg(gA, H1); out["glA_chg4h"][g] = chg(gA, 4 * H1)
        cs_ = np.r_[0, np.cumsum(np.nan_to_num(tk_))]; cn = np.r_[0, np.cumsum(np.isfinite(tk_))]
        a12 = np.maximum(kg - 11, 0); a24 = np.maximum(kg - 23, 0)
        out["taker1h"][g] = (cs_[kg + 1] - cs_[a12]) / np.maximum(cn[kg + 1] - cn[a12], 1)
        out["taker_prev1h"][g] = (cs_[a12] - cs_[a24]) / np.maximum(cn[a12] - cn[a24], 1)
        mx = np.full(len(kg), np.nan)
        for ii_, (kq, a_) in enumerate(zip(kg, E_on[g])):
            a0_ = np.searchsorted(ms, a_)
            if a0_ <= kq:
                mx[ii_] = oi[a0_:kq + 1].max()
        out["oi_vs_max"][g] = oi[kg] / mx
        out["oi_vol"][g] = oiv[kg] / vol24[cand][g]
    # premium (5m premium index close, fraction → %); bars closed ≤ T: open_time ≤ T − 5 min
    for nm in ("prem1h", "prem_pk"):
        out[nm] = np.full(n, np.nan)
    P = load_prem(pfiles)
    if P is not None and len(P) > 50:
        pt = P.open_time.values.astype("int64"); pc_ = P.c.values.astype(float) * 100
        pm = pd.Series(pc_).rolling(12, min_periods=10).mean().values
        kk = asof(pt, T - BAR, 15 * 60_000)
        g = kk >= 0
        out["prem1h"][g] = pm[kk[g]]
        for ii_ in np.nonzero(g)[0]:
            a0_ = np.searchsorted(pt, E_on[ii_]); kq = kk[ii_]
            if a0_ <= kq:
                out["prem_pk"][ii_] = np.nanmax(pm[a0_:kq + 1]) if np.isfinite(pm[a0_:kq + 1]).any() else np.nan
    # funding (settled ≤ T)
    ft, fc, months = TB.funding_for(p)
    for nm in ("f_last", "f_prev", "f_int_h", "f_last8", "f_prev8"):
        out[nm] = np.full(n, np.nan)
    if len(ft) >= 2:
        ft = np.asarray(ft, "int64"); rate = np.diff(fc) * 100
        kf = np.searchsorted(ft, T, side="right") - 1
        g = kf >= 1
        out["f_last"][g] = rate[kf[g]]; out["f_prev"][g] = rate[kf[g] - 1]
        ih = (ft[kf[g]] - ft[kf[g] - 1]) / H1; ih = np.clip(np.round(ih), 1, 8)
        out["f_int_h"][g] = ih
        kp = np.maximum(kf[g] - 2, 0); ihp = np.clip(np.round((ft[kf[g] - 1] - ft[kp]) / H1), 1, 8)
        out["f_last8"][g] = rate[kf[g]] * 8 / ih; out["f_prev8"][g] = rate[kf[g] - 1] * 8 / ihp
    # BTC
    bt, bc = btc
    kb = np.searchsorted(bt, T, side="right") - 2      # last BTC bar closed ≤ T
    for nm, m in (("btc1h", 12), ("btc4h", 48), ("btc24h", 288)):
        okb = kb - m >= 0
        out[nm] = np.where(okb, (bc[np.maximum(kb, 0)] / bc[np.maximum(kb - m, 0)] - 1) * 100, np.nan)
    # exits + funding + path facts
    mo_e = mo[sel]
    for ex in EXITS:
        s = sim(o, h, l, c, J, out["atrp"], tick, ex)
        xm = T + (s["col"].astype(np.int64) + 1) * BAR
        if len(ft):
            fs = (fc[np.searchsorted(ft, xm, "right")] - fc[np.searchsorted(ft, T, "right")]) * 100
        else:
            fs = np.zeros(n)
        mo_x = pd.to_datetime(xm, unit="ms").strftime("%Y-%m").values
        miss = np.array([(a_ not in months) or (b_ not in months) for a_, b_ in zip(mo_e, mo_x)], bool)
        out[f"{ex}_f"] = np.where(miss, -(xm - T) / (8 * H1) * TB.FUND_MISS, fs)
        out[f"{ex}_fm"] = miss
        out[f"{ex}_x"] = xm
        for kq in ("r", "k", "tpd", "sld", "worked", "reach", "rec", "best", "adv"):
            out[f"{ex}_{kq}"] = s[kq]
    df = pd.DataFrame(out)
    for cc_ in df.columns:
        if df[cc_].dtype == np.float64:
            df[cc_] = df[cc_].astype(np.float32)
    df.to_pickle(fo)
    return p, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default=""); ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--meta", default=""); ap.add_argument("--k5m", default=""); ap.add_argument("--fund", default="")
    ap.add_argument("--metrics", default=""); ap.add_argument("--prem", default=""); ap.add_argument("--out", default=os.path.join(ND, "rows"))
    ap.add_argument("--spot_all_zero", action="store_true")
    A = ap.parse_args()
    if A.k5m:
        TB.K5M_SRC.insert(0, A.k5m)
    if A.fund:
        TB.FUND_SRC.append(A.fund)
    os.makedirs(A.out, exist_ok=True)
    st = C.spot_status()
    M = pd.read_csv(A.meta) if A.meta else pd.read_csv(os.path.join(R, "reports", "EXHAUSTION_EXPAND_data", "allspk_meta.csv"))
    M = M[(M.pair.str.isascii()) & (~M.pair.str.startswith("1000")) & (~M.pair.isin(["BTCUSDT", "ETHUSDT"]))]
    M = M[M.spike_ms + WIN >= pd.Timestamp("2023-07-01").value // 10**6]
    if A.spot_all_zero:
        for p_ in M.pair.unique():
            for m_ in pd.date_range("2026-10-01", "2026-12-01", freq="MS").strftime("%Y-%m"):
                st[(p_, m_)] = 0
    pairs = sorted(M.pair.unique()) if not A.pairs else A.pairs.split(",")
    # pairs with at least one no-spot month among their episodes (± next month)
    keep = []
    for p_ in pairs:
        e = M[M.pair == p_]
        ms_ = set(pd.to_datetime(e.on_ms, unit="ms").dt.strftime("%Y-%m")) | set(pd.to_datetime(e.spike_ms + WIN, unit="ms").dt.strftime("%Y-%m"))
        if any(st.get((p_, m_), -1) == 0 for m_ in ms_):
            keep.append(p_)
    mdirs = MET_DIRS + [os.path.join(ND, "metrics")] + ([A.metrics] if A.metrics else [])
    pdirs = PREM_DIRS + [os.path.join(ND, "premium")] + ([A.prem] if A.prem else [])
    MI = met_index(sorted(set(mdirs))); PI = met_index(sorted(set(pdirs)))
    import json
    SUP = json.load(open(os.path.join(R, "reports", "NOSPOT_MCAP_data", "supply_used.json")))
    import leads_deep_regimes as LR
    B = LR.btc5m(); btc = (B.open_time.values.astype("int64"), B.c.values.astype(float))
    jobs = [(p_, M[M.pair == p_], st, MI.get(p_, []), PI.get(p_, []), SUP.get(p_), btc, A.out) for p_ in keep]
    print("pairs", len(jobs), flush=True)
    t0 = time.time(); tot = 0
    with mp.get_context("fork").Pool(A.procs) as pool:
        for i, (p_, n_) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            tot += max(n_, 0)
            if i % 25 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), p_, n_, tot, f"{time.time() - t0:.0f}s", flush=True)
    print("done", tot, flush=True)


if __name__ == "__main__":
    main()
