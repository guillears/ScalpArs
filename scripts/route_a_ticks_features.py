#!/usr/bin/env python3
"""ROUTE A — FEATURES + PRICED OUTCOMES (PREREG reports/ROUTE_A_TICKS_2026-11-19_PREREG.md).
For every fetched event (reports/ROUTE_A_TICKS_data/ev/*.npz) and checkpoint C0 / C1 / C2 / C5 (trigger close + 0 / 60 / 120 / 300 s):
 - 14 tick features from trades strictly before the decision time (no lookahead) + t15 since-last-trigger;
 - candle features from the local 5m cache (bars closed at the decision);
 - entry = first trade ≥ decision + 8 s; path = 1 s bars from ticks → 1m bars from ticks → local 5m bars;
   exits Q3 / TR / T1 (gross, kind, exit time, funding), PATH label (+5 before −5, 24 h), REACH.
Baseline moments (same pair-day) get the C0 tick features only. Output reports/ROUTE_A_TICKS_data/features.pkl."""
import glob, os, pickle, sys
import multiprocessing as mp
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_build as B   # noqa: E402  (sets the k5m / funding sources)
TB = B.TB
D = os.path.join(R, "reports", "ROUTE_A_TICKS_data"); EV = os.path.join(D, "ev")
BAR, H1, MIN, SEC = 300_000, 3_600_000, 60_000, 1000
CHK = (0, 60, 120, 300)
LAT = 8 * SEC
TICK = ["tps_accel", "tps_burst", "buy_cnt", "buy_qv", "buy_qv_last", "big_lift", "med_trade", "top10_share", "retrace",
        "run_ratio", "last_conc", "spread_bps", "uptick_share", "vwap_gap"]
CAND = ["ret15", "volx15", "ret6h_base", "ret1h_prior", "close_pos", "logq24", "green6", "btc_1h", "since_burst_h"]
EXITS = {"Q3": (3.0, 4.0, 4 * H1, None), "TR": (None, 5.0, 24 * H1, (3.0, 3.0)), "T1": (None, None, H1, None)}
G = {}


def runs(m):
    """mean run length of taker buys (m False) and taker sells (m True)"""
    if len(m) == 0:
        return np.nan, np.nan
    ch = np.r_[True, m[1:] != m[:-1]]
    st = np.where(ch)[0]; ln = np.diff(np.r_[st, len(m)]); side = m[st]
    b = ln[~side]; s = ln[side]
    return (b.mean() if len(b) else np.nan), (s.mean() if len(s) else np.nan)


def tick_feats(t, p, q, m, sig, T):
    """features at decision T for a burst whose trigger closes at sig; uses trades with t < T only"""
    qv = p * q
    base = (t >= sig - 75 * MIN) & (t < sig - 15 * MIN)
    bur = (t >= sig - 15 * MIN) & (t < T)
    last = (t >= T - MIN) & (t < T)
    f = {}
    nb, nu, nl = base.sum(), bur.sum(), last.sum()
    if nu < 20 or nb < 5:
        return None
    dur_b = 60 * 60.0; dur_u = (T - (sig - 15 * MIN)) / 1000.0
    tps_b = nb / dur_b
    f["tps_accel"] = np.log((nl / 60.0 + 1e-3) / tps_b)
    f["tps_burst"] = np.log((nu / dur_u) / tps_b)
    mb = m[bur]; qb = qv[bur]; pb = p[bur]
    f["buy_cnt"] = (~mb).mean()
    f["buy_qv"] = qb[~mb].sum() / qb.sum()
    ql = qv[last]; ml = m[last]
    f["buy_qv_last"] = ql[~ml].sum() / ql.sum() if nl else np.nan
    p99 = np.percentile(qv[base], 99)
    # lift = prints above the previous trade's price (previous trade may sit just before the burst window)
    idx = np.where(bur)[0]
    prevp = p[np.maximum(idx - 1, 0)]
    f["big_lift"] = np.log1p(((~mb) & (qb >= p99) & (pb > prevp)).sum())
    f["med_trade"] = np.log(np.median(qb) / np.median(qv[base]))
    srt = np.sort(qb)[::-1]; k = max(1, len(srt) // 10)
    f["top10_share"] = srt[:k].sum() / srt.sum()
    hi, lo = pb.max(), pb.min()
    f["retrace"] = (hi - pb[-1]) / (hi - lo) if hi > lo else 0.0
    rb, rs = runs(mb)
    f["run_ratio"] = np.log(rb / rs) if rb == rb and rs == rs and rs > 0 else np.nan
    f["last_conc"] = np.log((ql.sum() + 1e-9) / (qb.sum() / (dur_u / 60.0)))
    opp = mb[1:] != mb[:-1]
    dl = np.abs(np.diff(np.log(pb)))[opp]
    f["spread_bps"] = np.median(dl) * 1e4 if len(dl) else np.nan
    dp = np.diff(pb); chg = dp != 0
    f["uptick_share"] = (dp[chg] > 0).mean() if chg.any() else np.nan
    f["vwap_gap"] = (pb[-1] / (qb.sum() / q[bur].sum()) - 1) * 100
    f["last_px"] = pb[-1]
    return f


def sim(o, h, l, c, te, E0, tp, sl, hz, trail):
    """long exit over bars (te = bar end times); first bar end > horizon end stops; same-bar stop first. (gross %, kind, exit ms)"""
    peak = E0; armed = False; end_ms = te[0] + hz
    last = len(c) - 1
    for i in range(len(c)):
        if te[i] > end_ms + 1:
            last = max(0, i - 1); break
        if sl is not None and l[i] <= E0 * (1 - sl / 100):
            return (min(E0 * (1 - sl / 100), o[i]) / E0 - 1) * 100, 1, te[i]
        if trail is not None and armed and l[i] <= peak * (1 - trail[1] / 100):
            return (min(peak * (1 - trail[1] / 100), o[i]) / E0 - 1) * 100, 3, te[i]
        if tp is not None and h[i] >= E0 * (1 + tp / 100) * 1.0005:
            return tp, 2, te[i]
        peak = max(peak, h[i])
        if trail is not None and peak >= E0 * (1 + trail[0] / 100):
            armed = True
    return (c[last] / E0 - 1) * 100, 0, te[last]


def path_bars(t, p, bars, k5, ent_ms, tick_end):
    """1 s bars from ticks [ent_ms, tick_end) → 1m bars from tick_end → 5m local bars after the last 1m bar"""
    sel = (t >= ent_ms) & (t < tick_end)
    ts, ps = t[sel], p[sel]
    out = []
    if len(ts):
        k = (ts - ent_ms) // SEC
        df = pd.DataFrame(dict(k=k, p=ps)).groupby("k").p
        kk = df.first().index.values
        out.append(np.c_[df.first().values, df.max().values, df.min().values, df.last().values, ent_ms + (kk + 1) * SEC])
    if len(bars):
        b = bars[bars[:, 0] >= tick_end]
        if len(b):
            out.append(np.c_[b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 0] + MIN])
    last_end = out[-1][-1, 4] if out else ent_ms
    if k5 is not None:
        g, o5, h5, l5, c5 = k5
        j = int(np.searchsorted(g, last_end))
        jj = min(len(g), j + 300)
        if jj > j:
            out.append(np.c_[o5[j:jj], h5[j:jj], l5[j:jj], c5[j:jj], g[j:jj] + BAR])
    return np.vstack(out) if out else np.zeros((0, 5))


def candle_feats(x, T, nrm, q24, btc):
    g, o, h, l, c, v, qv = x
    i = int(np.searchsorted(g, T - BAR, side="right")) - 1     # last bar with close ≤ T
    if i < 80:
        return None
    f = dict(ret15=(c[i] / c[i - 3] - 1) * 100, volx15=np.log(qv[i - 2:i + 1].sum() / (nrm[i] * 3 / 12)) if nrm[i] > 0 else np.nan,
             ret6h_base=(c[i] / c[i - 72:i + 1].min() - 1) * 100, ret1h_prior=(c[i - 3] / c[i - 15] - 1) * 100,
             close_pos=(c[i] - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 0.5, logq24=np.log10(max(q24[i], 1.0)),
             green6=float((c[i - 5:i + 1] > o[i - 5:i + 1]).sum()))
    bg, bc = btc
    j = int(np.searchsorted(bg, T - BAR, side="right")) - 1
    f["btc_1h"] = (bc[j] / bc[j - 12] - 1) * 100 if j >= 12 and abs(bg[j] - g[i]) < H1 else np.nan
    return f


def work(pair):
    evs = G["O"][G["O"].pair == pair]
    x = TB.load(pair)
    if x is None:
        return []
    g, o, h, l, c, v, qv = x
    n = len(g)
    q24 = pd.Series(qv).rolling(288, min_periods=144).sum().values
    hq = pd.Series(qv).rolling(12, min_periods=12).sum()
    hour_ix = np.where(g % H1 == H1 - BAR)[0]
    hs = pd.Series(hq.values[hour_ix], index=hour_ix)
    nrm_h = hs.rolling(240, min_periods=120).median().shift(24)
    nrm = pd.Series(np.nan, index=np.arange(n)); nrm[nrm_h.index] = nrm_h.values; nrm = nrm.ffill().values
    ft, fc, _ = TB.funding_for(pair)
    rates = np.diff(fc) * 100 if len(ft) else np.zeros(0)
    trig = G["trig"].get(pair, np.zeros(0, "int64"))
    out = []
    for e in evs.itertuples():
        f = os.path.join(EV, f"{pair}_{e.sig_ms}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        t, p, q, m, bars = z["t"], z["p"], z["q"], z["m"], z["bars"]
        sig = int(e.sig_ms)
        prev = trig[trig < sig - 6 * H1 + 1]
        since = np.log(min((sig - prev[-1]) / H1, 720.0)) if len(prev) else np.log(720.0)
        for cs in CHK:
            T = sig + cs * SEC
            tf = tick_feats(t, p, q, m, sig, T)
            cf = candle_feats(x, T, nrm, q24, G["btc"])
            if tf is None or cf is None:
                continue
            r = dict(pair=pair, sig_ms=sig, chk=cs, dec_ms=T, runner=e.runner, spike_after=e.spike_after, month=e.month, block=e.block,
                     pop_n=e.pop_n, since_burst_h=since, already_flagged=int(e.spike_after > 0 and e.spike_after <= T),
                     kind="event")
            r.update(tf); r.update(cf)
            j = int(np.searchsorted(t, T + LAT))
            if j >= len(t):
                continue
            ent_ms = int(t[j]); E0 = p[j]
            r["entry_ms"] = ent_ms; r["entry_px"] = E0; r["entry_delay_s"] = (ent_ms - T) / 1000
            r["pre_entry_move"] = (E0 / tf["last_px"] - 1) * 100
            pb = path_bars(t, p, bars, (g, o, h, l, c), ent_ms, int(t[-1]) + 1)
            if len(pb) < 10:
                continue
            bo, bh, bl, bc, be = pb.T
            for k, (tp, sl, hz, tr) in EXITS.items():
                gr, kind, xms = sim(bo, bh, bl, bc, be, E0, tp, sl, hz, tr)
                fcost = 0.0
                if len(ft):
                    a_ = int(np.searchsorted(ft, ent_ms, side="right")); b_ = int(np.searchsorted(ft, int(xms), side="right"))
                    fcost = float(rates[a_:b_].sum())
                r[k + "_g"], r[k + "_k"], r[k + "_x"], r[k + "_f"] = gr, kind, int(xms), fcost
                r[k + "_complete"] = bool(kind != 0 or be[-1] >= ent_ms + hz)
            # PATH: +5 before −5 within 24 h; REACH = max favourable before −5 (or 24 h)
            w = be <= ent_ms + 24 * H1
            up = np.where(bh[w] >= E0 * 1.05)[0]; dn = np.where(bl[w] <= E0 * 0.95)[0]
            ui = up[0] if len(up) else 10**9; di = dn[0] if len(dn) else 10**9
            r["worked"] = int(ui < di)
            stop_i = di if di < 10**9 else w.sum()
            r["reach"] = (bh[w][:max(1, stop_i + 1)].max() / E0 - 1) * 100 if w.any() else np.nan
            r["ret_path"] = [((bc[(be <= ent_ms + mm * MIN)][-1] / E0 - 1) * 100 if (be <= ent_ms + mm * MIN).any() else np.nan)
                             for mm in (1, 2, 5, 10, 15, 30, 60, 120, 240)]
            out.append(r)
        bms = int(z["bms"])
        if bms > 0 and len(z["bt"]) > 50:
            tf = tick_feats(z["bt"], z["bp"], z["bq"], z["bm"], bms, bms)
            cf = candle_feats(x, bms, nrm, q24, G["btc"])
            if tf is not None and cf is not None:
                r = dict(pair=pair, sig_ms=bms, chk=0, dec_ms=bms, runner=-1, month=e.month, block=e.block, kind="baseline", since_burst_h=np.nan)
                r.update(tf); r.update(cf); out.append(r)
    return out


def init(O, trig, btc):
    G["O"] = O; G["trig"] = trig; G["btc"] = btc


def main():
    O = pd.read_csv(os.path.join(D, "order.csv"))
    have = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(EV, "*.npz"))}
    O = O[(O.pair + "_" + O.sig_ms.astype(str)).isin(have)].reset_index(drop=True)
    X = pickle.load(open(os.path.join(R, "reports", "FRENZY_LONG_SEARCH_data", "early_rows.pkl"), "rb"))
    X = X[X.cell == "W15_R3_V10"]
    trig = {p: np.sort(g.sig_ms.values.astype("int64")) for p, g in X.groupby("pair")}
    b = TB.load("BTCUSDT"); btc = (b[0], b[4])
    rows = []
    with mp.Pool(8, initializer=init, initargs=(O, trig, btc)) as P:
        for k, r in enumerate(P.imap_unordered(work, sorted(O.pair.unique()))):
            rows += r
    F = pd.DataFrame(rows)
    pickle.dump(F, open(os.path.join(D, "features.pkl"), "wb"))
    print("events", O.shape[0], "rows", len(F), F.groupby(["kind", "chk"]).size().to_string())


if __name__ == "__main__":
    main()
