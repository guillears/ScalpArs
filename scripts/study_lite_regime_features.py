#!/usr/bin/env python3
"""LITE MACRO-REGIME study — day-level regime features (no lookahead), per reports/LITE_REGIME_PREREG.txt section 3.
Cache only. Outputs reports/lite_regime/{spikes.csv, spike_parity.txt, day_features.csv}.
Usage: venv/bin/python scripts/study_lite_regime_features.py [--rescan]
"""
import os, sys, glob
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C26 = os.path.join(ROOT, "reports", "backtest_cache")
C25 = os.path.join(ROOT, "reports", "backtest_cache_2025")
OUT = os.path.join(ROOT, "reports", "lite_regime")
FET = os.path.join(OUT, "fetch")            # gap fetch 2026-10-11 -> (scripts/study_lite_regime_fetch.py), outside backtest_cache
os.makedirs(OUT, exist_ok=True)
DAY = 86_400_000; HOUR = 3_600_000; BAR = 300_000
STABLE = ("USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "DAIUSDT", "EURUSDT", "USDEUSDT", "XUSDUSDT", "BFUSDUSDT")
# engine spike thresholds (trading_config.json: frenzy_spike_ret_pct 5, frenzy_spike_vol_mult 20, frenzy_spike_min_hour_usd 2e6, state 100)
RET_MIN, VOL_MIN, HOUR_MIN, STATE_VOL = 5.0, 20.0, 2e6, 100.0


def load_5m(pair):
    fs = [os.path.join(d, "k5m_full", pair + ".csv") for d in (C25, C26)] + [os.path.join(FET, "k5m", pair + ".csv")]
    parts = []
    for f in fs:
        if os.path.exists(f):
            try:
                parts.append(pd.read_csv(f, usecols=[0, 1, 2, 3, 4, 5]))
            except Exception:
                pass
    if not parts:
        return None
    k = pd.concat(parts, ignore_index=True)
    k.columns = ["t", "o", "h", "l", "c", "v"]
    k = k.dropna().drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return k


def scan_pair(pair):
    """market-wide FRENZY spikes on one pair (engine spike rule; episode approx = new spike iff no lead / state-vol bar in prior 24 h)."""
    try:
        k = load_5m(pair)
        if k is None or len(k) < 3000:
            return []
        t = k.t.values.astype(np.int64); c = k.c.values.astype(float)
        q = k.v.values * (k.h.values + k.l.values + c) / 3.0
        n = len(t)
        cs = np.concatenate([[0.0], np.cumsum(q)])
        q1h = np.full(n, np.nan)
        q1h[11:] = cs[12:] - cs[:-12]
        contig = np.zeros(n, bool); contig[11:] = (t[11:] - t[:-11]) == 11 * BAR     # 12 consecutive bars
        q1h[~contig] = np.nan
        # hourly volume series (full hours only) and its rolling 720-h median
        hr = t // HOUR
        hs = pd.Series(q).groupby(hr).agg(["sum", "size"])
        hs = hs[hs["size"] == 12]["sum"]
        med = hs.rolling(720, min_periods=240).median()
        hidx = hs.index.values.astype(np.int64)            # hour index (start = idx*HOUR, end = (idx+1)*HOUR)
        # for bar i (open t_i): eligible hours end <= t_i - 24h  -> idx + 1 <= (t_i - 24h)/HOUR
        lim = (t - 24 * HOUR) // HOUR - 1
        pos = np.searchsorted(hidx, lim, side="right") - 1
        nh = np.where(pos >= 0, med.values[np.clip(pos, 0, None)], np.nan)
        volx = q1h / nh
        r30 = np.full(n, np.nan)
        ok6 = np.zeros(n, bool); ok6[6:] = (t[6:] - t[:-6]) == 6 * BAR
        r30[6:] = (c[6:] / c[:-6] - 1) * 100
        r30[~ok6] = np.nan
        with np.errstate(invalid="ignore"):
            lead = (r30 >= RET_MIN) & (volx >= VOL_MIN) & (q1h >= HOUR_MIN)
            act = lead | (volx >= STATE_VOL)
        act_t = t[act]
        out = []
        for i in np.where(lead)[0]:
            j = np.searchsorted(act_t, t[i], side="left") - 1      # previous active bar strictly before
            if j >= 0 and act_t[j] >= t[i] - 24 * HOUR:
                continue
            tc = t[i] + BAR
            def px(dt):
                jj = np.searchsorted(t, t[i] + dt, side="left")
                return c[jj] if jj < n and t[jj] == t[i] + dt else np.nan
            p2, p12 = px(2 * HOUR), px(12 * HOUR)
            out.append((pair, int(tc), float(c[i]), float(volx[i]), (p2 / c[i] - 1) * 100 if p2 == p2 else np.nan,
                        (p12 / c[i] - 1) * 100 if p12 == p12 else np.nan))
        return out
    except Exception as e:
        print("ERR", pair, e, flush=True)
        return []


def spikes(rescan=False):
    f = os.path.join(OUT, "spikes.csv")
    if os.path.exists(f) and not rescan:
        return pd.read_csv(f)
    pairs = sorted({os.path.basename(p)[:-4] for d in (C25, C26) for p in glob.glob(os.path.join(d, "k5m_full", "*.csv"))})
    print("scanning", len(pairs), "pairs", flush=True)
    rows = []
    with Pool(8) as pool:
        for i, r in enumerate(pool.imap_unordered(scan_pair, pairs, chunksize=4)):
            rows += r
            if i % 100 == 0:
                print(i, len(rows), flush=True)
    S = pd.DataFrame(rows, columns=["pair", "spike_ms", "px", "volx", "ret2h", "ret12h"]).sort_values("spike_ms")
    S.to_csv(f, index=False)
    return S


def parity(S):
    m = pd.read_csv(os.path.join(ROOT, "reports", "lite_wlsweep", "matrix.csv"), usecols=["fid", "era", "pair", "entry_frenzy_spike_at"])
    m = m[m.fid.str.startswith("BT|")]
    m["sp"] = pd.to_datetime(m.entry_frenzy_spike_at).values.astype("datetime64[ms]").astype(np.int64)
    g = {p: d.spike_ms.values for p, d in S.groupby("pair")}
    def hit(r, tol):
        a = g.get(r.pair)
        return bool(a is not None and len(a) and np.min(np.abs(a - r.sp)) <= tol)
    lines = []
    for tol, lab in ((10 * 60_000, "10 min"), (60 * 60_000, "60 min")):
        m["hit"] = m.apply(lambda r: hit(r, tol), axis=1)
        lines.append(f"BT LITE fills whose engine spike matches a rebuilt spike within {lab}: " +
                     " · ".join(f"{e} {d.hit.mean()*100:.1f}% (N {len(d)})" for e, d in m.groupby("era")))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "spike_parity.txt"), "w").write(txt + "\n")
    print(txt)


def load_daily():
    rows = []
    pairs = sorted({os.path.basename(p)[:-4] for d in (C25, C26) for p in glob.glob(os.path.join(d, "k1d", "*.csv"))})
    for p in pairs:
        parts = []
        for f in (os.path.join(C25, "k1d", p + ".csv"), os.path.join(C26, "k1d", p + ".csv"), os.path.join(FET, "k1d", p + ".csv")):
            if os.path.exists(f):
                x = pd.read_csv(f, usecols=[0, 1, 2, 3, 4, 6])
                x.columns = ["t", "o", "h", "l", "c", "qv"]
                x["t"] = (np.round(np.where(x.t < 1e10, x.t * 1_000_000, x.t) / DAY) * DAY).astype(np.int64)   # 2025 cache stores truncated kilo-seconds
                parts.append(x)
        if not parts:
            continue
        x = pd.concat(parts).drop_duplicates("t", keep="last")
        x["pair"] = p
        rows.append(x)
    D = pd.concat(rows, ignore_index=True)
    D = D[((D.t % DAY) == 0) & (D.t + DAY <= int(pd.Timestamp.utcnow().timestamp() * 1000))]   # closed daily bars only
    return D


def main():
    S = spikes("--rescan" in sys.argv)
    print("spikes", len(S), flush=True)
    parity(S)
    D = load_daily()
    C = D.pivot(index="t", columns="pair", values="c").sort_index()
    H = D.pivot(index="t", columns="pair", values="h").sort_index()
    QV = D.pivot(index="t", columns="pair", values="qv").sort_index()
    days = pd.date_range("2024-12-01", "2026-10-16", freq="D")
    dms = days.values.astype("datetime64[ms]").astype(np.int64)
    b = C["BTCUSDT"]; bh = H["BTCUSDT"]
    lr = np.log(C / C.shift(1))
    out = []
    # funding (2026 cache only)
    fund = {}
    for f in glob.glob(os.path.join(C26, "funding", "*.csv")):
        try:
            x = pd.read_csv(f); fund[os.path.basename(f)[:-4]] = x
        except Exception:
            pass
    # own rolling (BT only)
    P = pd.read_csv(os.path.join(ROOT, "reports", "LITE_PROTECT_rows.csv"), usecols=["fid", "block", "te", "base_xm", "rm_base_pct"])
    P = P[P.block == "BT"].copy(); P["cl"] = P.te + P.base_xm * 60_000; P = P.sort_values("cl")
    W = pd.read_csv(os.path.join(ROOT, "reports", "TICK_STUDY_2026-10-15_cohort.csv"))
    W = W[(W.block == "BT") & (W.sleeve == "FRENZY_WILLY")].copy(); W["cl"] = W.te + 90 * 60_000; W = W.sort_values("cl")
    St = S.spike_ms.values
    S2 = S.assign(end12=S.spike_ms + 12 * HOUR, end2=S.spike_ms + 2 * HOUR)
    for d in dms:
        prev = d - DAY                                     # last CLOSED daily bar = open time d-1
        if prev not in C.index:
            continue
        r = {"day_ms": d, "day": pd.to_datetime(d, unit="ms").strftime("%Y-%m-%d")}
        bc = b.loc[:prev].dropna()
        if len(bc) < 31:
            continue
        x0 = bc.iloc[-1]
        for k in (1, 3, 7, 30):
            r[f"r{k}"] = (x0 / bc.iloc[-1 - k] - 1) * 100
        for k in (20, 50, 200):
            r[f"dma{k}"] = (x0 / bc.iloc[-k:].mean() - 1) * 100 if len(bc) >= k else np.nan
        bl = np.log(bc).diff().dropna()
        r["rv7"] = bl.iloc[-7:].std() * 100
        r["rvtr"] = bl.iloc[-7:].std() / bl.iloc[-30:].std()
        r["off30"] = (x0 / bh.loc[:prev].dropna().iloc[-30:].max() - 1) * 100
        # universe = top-50 by day d-1 quote volume (excl BTC + stables)
        qv = QV.loc[:prev].iloc[-30:].mean().dropna().drop(labels=[p for p in ("BTCUSDT",) + STABLE if p in QV.columns], errors="ignore")   # DEV1: 30d avg
        U = qv.sort_values(ascending=False).index[:50]
        cu = C.loc[:prev, U]
        last = cu.iloc[-1]
        r["br_up3"] = float(((last / cu.iloc[-4]) > 1).mean())
        r["br_up7"] = float(((last / cu.iloc[-8]) > 1).mean())
        r["br_ma7"] = float((last > cu.iloc[-7:].mean()).mean())
        r["br_ma20"] = float((last > cu.iloc[-20:].mean()).mean())
        r["alt_rs7"] = float(((last / cu.iloc[-8] - 1) * 100).median() - r["r7"])   # DEV1: median
        tq = QV.loc[:prev].sum(axis=1)
        r["voltr"] = tq.iloc[-7:].mean() / tq.iloc[-30:].mean()
        dr = lr.loc[:prev, U].iloc[-7:]
        r["disp7"] = float(((dr.quantile(0.75, axis=1) - dr.quantile(0.25, axis=1)) / 1.349).mean() * 100)   # DEV1: robust IQR
        # funding
        fv = []
        for p in U:
            x = fund.get(p)
            if x is not None:
                fv += list(x.rate[(x.t >= d - 3 * DAY) & (x.t < d)].values)
        r["fund3"] = float(np.mean(fv)) * 100 if len(fv) >= 50 else np.nan
        # frenzy activity
        r["spk24"] = int(((St >= d - DAY) & (St < d)).sum())
        r["spk72"] = int(((St >= d - 3 * DAY) & (St < d)).sum())
        r["flags96"] = int(S.pair[(St >= d - 4 * DAY) & (St < d)].nunique())
        k12 = S2[(S2.end12 < d) & S2.ret12h.notna()].tail(20)
        k2 = S2[(S2.end2 < d) & S2.ret2h.notna()].tail(20)
        r["ft2"] = float((k2.ret2h > 0).mean()) if len(k2) == 20 else np.nan
        r["ft12"] = float((k12.ret12h > 0).mean()) if len(k12) == 20 else np.nan
        r["ftr12"] = float(k12.ret12h.mean()) if len(k12) == 20 else np.nan
        # own rolling
        for nm, X, col in (("lite", P, "rm_base_pct"), ("willy", W, "rec_pct")):
            done = X[X.cl < d]
            r[f"{nm}_n10"] = done[col].iloc[-10:].mean() if len(done) >= 10 else np.nan
            r[f"{nm}_n20"] = done[col].iloc[-20:].mean() if len(done) >= 20 else np.nan
            for k in (3, 7):
                w = done[done.cl >= d - k * DAY][col]
                r[f"{nm}_d{k}"] = w.mean() if len(w) else np.nan
        out.append(r)
    F = pd.DataFrame(out)
    F.to_csv(os.path.join(OUT, "day_features.csv"), index=False)
    print(F.describe().T[["count", "mean", "50%"]].to_string())


if __name__ == "__main__":
    main()
