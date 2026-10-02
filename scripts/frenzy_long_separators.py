#!/usr/bin/env python3
"""🔬 FRENZY_LONG — what separates winners from losers? Operator (2026-10-02): "42 % won / 58 % stopped and nothing divides them? hard to believe,
make a deep dive". Cohort = the live rule (staircase ON, 24 h volume ≥ $20M, 5m ATR ≤ 2.5 %), live exit (stop 3 · trail 5/1.5 · 12 h), strict ruler.
Every feature is read on bars CLOSED at the entry and is scored on every trade (coverage printed).
  1D    each feature in quintiles — per-trade mean, win rate, and whether the pattern is monotonic
  RULE  single-feature keep-rules (keep above / below the 20 / 33 / 50 / 67 / 80 % cut, keeping ≥ 40 % of trades): the best kept-mean, judged
        against the SAME search on 300 shuffles of the outcomes (the luck bar for "best of ~300 rules")
  2D    every feature pair in 3 × 3 cells (≥ 50 trades per cell): best and worst cell vs the same search on 100 shuffles
  OOS   rules found on Jan–Apr are FROZEN and applied to May–Sep, and the reverse; and found on SEEN (≥ $100M) → applied to UNSEEN ($20–100M), and the reverse
Market-wide features (BTC, breadth, hour) repeat across same-day trades → their rows also show the by-day range.
Usage: venv/bin/python scripts/frenzy_long_separators.py"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_breadth as BB  # noqa: E402
import frenzy_long_followup as F  # noqa: E402
sys.argv = _a
H, B, FS, ST, M = F.H, F.B, F.FS, F.ST, BB.M; OUT = os.path.join(ST.ROOT, "reports", "FRENZY_LONG_SEPARATORS_2026-10-02.md"); RNG = np.random.default_rng(11)
MARKET = ("btc_r1h", "btc_r4h", "btc_r24h", "btc_vs_e50", "btc_vs_e200", "below50", "below200", "down1h", "down4h", "down24h", "d_below50", "hour_utc")


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean(); dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).values


def feats(pair, d, V=100.0):
    """The live entry rule (as scripts/frenzy_long_followup.triggers) with the pair's own features on the signal bar."""
    t = d.open_time.values.astype("int64"); o, h, l, c, q = d.o.values, d.h.values, d.l.values, d.c.values, d.qvol.values; n = len(c); s = pd.Series(c)
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); volx = (q1h / norm).values
    r = lambda k: np.r_[[np.nan] * k, (c[k:] / c[:-k] - 1) * 100]
    r5, r30, r1h, r4h, r24h = r(1), r(6), r(12), r(48), r(288); pc = np.r_[c[0], c[:-1]]
    tr_ = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); atr = (pd.Series(tr_).ewm(alpha=1 / 14, adjust=False).mean() / c * 100).values
    e = {k: s.ewm(span=k, adjust=False).mean().values for k in (5, 8, 20, 50, 200)}; rs = rsi(c); cq = np.concatenate([[0.0], np.cumsum(q)])
    rng1h = (pd.Series(h).rolling(12).max() / pd.Series(l).rolling(12).min() - 1).values * 100
    hi24 = pd.Series(h).rolling(288).max().values; green = pd.Series((c > o).astype(float)).rolling(12).mean().values * 100
    body = np.where(h > l, (c - o) / np.maximum(h - l, 1e-12), 0.0); upw = np.where(h > l, (h - np.maximum(c, o)) / np.maximum(h - l, 1e-12), 0.0)
    with np.errstate(invalid="ignore"):
        lead = np.nonzero((r30 >= 5) & (volx >= 20) & (q1h.values >= 2e6))[0]
    out = []; i = 0
    for on in lead:
        on = int(on)
        if on < i:
            continue
        pv = (h[on:] + l[on:] + c[on:]) / 3 * q[on:]; vw = np.cumsum(pv) / np.maximum(np.cumsum(q[on:]), 1e-12)
        above = pd.Series(c[on:] >= vw).rolling(12).min().fillna(0).values >= 1
        with np.errstate(invalid="ignore"):
            st = above & (volx[on:] >= V) & (np.arange(n - on) >= 24)
        last = on; end = n
        for j in range(on + 1, n):
            if t[j] - t[last] > 24 * 3600_000:
                end = j; break
            if st[j - on]:
                last = j
        off = 12; k_entry = 0; n_state = 0
        for j in range(on, min(end, n - 2)):
            if st[j - on]:
                n_state += 1
                if off >= 12:
                    q24 = cq[j + 1] - cq[max(j - 287, 0)]
                    if q24 >= 20e6:
                        k_entry += 1; pk = h[on:j + 1].max(); base = c[on - 6]
                        out.append(dict(pair=pair, t=int(t[j + 1]), entry=float(o[j + 1]), atr=float(atr[j]), q24_m=q24 / 1e6, hours=(t[j] - t[on]) / 3600e3,
                                        vs_vwap=(c[j] / vw[j - on] - 1) * 100, vol_mult=float(volx[j]), vol_mult_1h_ago=float(volx[j - 12]), vol_trend=float(volx[j] / volx[j - 12]) if volx[j - 12] > 0 else np.nan,
                                        run_pct=(pk / base - 1) * 100, gain_pct=(c[j] / base - 1) * 100, off_peak=(c[j] / pk - 1) * 100, spike_r30=float(r30[on]), spike_volx=float(volx[on]),
                                        r5m=float(r5[j]), r30m=float(r30[j]), r1h=float(r1h[j]), r4h=float(r4h[j]), r24h=float(r24h[j]), rsi14=float(rs[j]),
                                        gap5_20=(e[5][j] / e[20][j] - 1) * 100, gap5_8=(e[5][j] / e[8][j] - 1) * 100, vs_e20=(c[j] / e[20][j] - 1) * 100, vs_e50=(c[j] / e[50][j] - 1) * 100,
                                        vs_e200=(c[j] / e[200][j] - 1) * 100, e50_vs_e200=(e[50][j] / e[200][j] - 1) * 100, range_1h=float(rng1h[j]), off_24h_high=(c[j] / hi24[j] - 1) * 100,
                                        green_1h=float(green[j]), bar_body=float(body[j]), bar_upper_wick=float(upw[j]), stop_atr=3.0 / float(atr[j]) if atr[j] > 0 else np.nan,
                                        attempt=k_entry, state_bars_before=n_state - 1, vwap_dist_atr=((c[j] / vw[j - on] - 1) * 100) / float(atr[j]) if atr[j] > 0 else np.nan))
                off = 0
            else:
                off += 1
        i = end
    return out


def keep_rules(Xv, y, min_keep=0.40):
    """Best single-feature keep-rule on standardised arrays. Xv: dict name → values. Returns (best mean, name, side, cut, n)."""
    best = (-9, None, None, None, 0); n = len(y)
    for name, v in Xv.items():
        ok = ~np.isnan(v)
        if ok.sum() < n * 0.9:
            continue
        for qq in (0.2, 1 / 3, 0.5, 2 / 3, 0.8):
            cut = np.nanquantile(v, qq)
            for side, m in (("≤", ok & (v <= cut)), (">", ok & (v > cut))):
                k = m.sum()
                if k >= n * min_keep and k <= n * 0.85:
                    mu = y[m].mean()
                    if mu > best[0]:
                        best = (mu, name, side, cut, int(k))
    return best


if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) >= 288 * 40:
            rows += feats(pair, d)
    TR = pd.DataFrame(rows).sort_values("t").reset_index(drop=True); TR = TR[TR.atr <= 2.5].reset_index(drop=True)
    TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d"); TR["set"] = np.where(TR.q24_m >= 100, "SEEN", "UNSEEN"); TR["hour_utc"] = pd.to_datetime(TR.t, unit="ms").dt.hour.astype(float)
    yb = pd.read_csv(os.path.join(ST.K5, "BTCUSDT.csv")).drop_duplicates("open_time").sort_values("open_time"); YB = M.btc_frame(yb); grid = yb.open_time.values.astype("int64")
    frames = [pd.read_csv(f, usecols=["open_time", "c"]).drop_duplicates("open_time").sort_values("open_time") for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))) if os.path.basename(f).isascii()]
    YR = BB.breadth(frames, grid); k = TR.t.values - 300_000
    for col in YB.columns:
        TR[col] = YB[col].reindex(k).values
    for col in YR.columns:
        TR[col] = YR[col].reindex(k).values
    paths = {}
    for r in TR.itertuples():
        hh, ll, cc = B.bars1m(r.pair, r.t)
        if len(cc) >= 30:
            paths[r.Index] = (hh, ll, cc)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in TR.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None
    res = []; free = {}
    for r in TR.itertuples():
        if r.t < free.get(r.pair, 0) or r.Index not in paths:
            continue
        x, mins, how = H.walk(*paths[r.Index], r.entry, 3.0, 5.0, 1.5, side=1, slip=0.10, gap=True); free[r.pair] = r.t + mins * 60_000
        fr = FR.get(r.pair); fund = 0.0 if fr is None else -fr[(fr.t > r.t) & (fr.t <= r.t + mins * 60_000)].rate.sum() * 100
        last_f = np.nan if fr is None or not len(fr[fr.t <= r.t]) else float(fr[fr.t <= r.t].rate.iloc[-1]) * 100
        res.append((r.Index, x - B.COST + fund, how, last_f))
    E = pd.DataFrame(res, columns=["i", "net", "how", "funding_rate"]).set_index("i"); X = TR.loc[E.index].copy(); X["net"] = E.net; X["how"] = E.how; X["funding_rate"] = E.funding_rate
    FEATS = [c for c in X.columns if c not in ("pair", "t", "entry", "day", "set", "net", "how")]
    y = X.net.values; n = len(X); h1 = (X.day < B.SPLIT).values; seen = (X.set == "SEEN").values
    L = ["# 🔬 FRENZY_LONG — what separates winners from losers? (deep dive)", "",
         f"{n:,} trades (live rule, ATR ≤ 2.5 %, live exit, strict ruler): {(y > 0).mean() * 100:.0f}% won · {(X.how == 'stop').mean() * 100:.0f}% full stops · {y.mean():+.2f} per trade. "
         f"{len(FEATS)} features, all read on closed bars at entry. Coverage < 100 %: " + (", ".join(f"{c} {X[c].notna().mean() * 100:.0f}%" for c in FEATS if X[c].notna().mean() < 0.999) or "none") + ".", "",
         "## 1 · Every feature in quintiles (lowest → highest fifth): per-trade mean", "",
         "| Feature | Q1 | Q2 | Q3 | Q4 | Q5 | spread Q5−Q1 | monotonic | same sign in both halves | same sign SEEN & UNSEEN |", "|---|---|---|---|---|---|---|---|---|---|"]
    stat = []
    for c in FEATS:
        v = X[c].values; ok = ~np.isnan(v)
        if ok.sum() < n * 0.9 or len(np.unique(v[ok])) < 5:
            continue
        try:
            b = pd.qcut(v[ok], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        if len(np.unique(b)) < 4:
            continue
        mu = [y[ok][b == k].mean() for k in sorted(np.unique(b))]; sp = mu[-1] - mu[0]
        d_ = np.sign(np.diff(mu)); mono = "yes" if (abs(d_.sum()) == len(d_)) else ("mostly" if abs(d_.sum()) >= len(d_) - 2 else "no")
        def spread(mask):
            vv, yy = v[ok & mask], y[ok & mask]
            if len(vv) < 100:
                return np.nan
            lo_, hi_ = np.nanquantile(vv, 0.4), np.nanquantile(vv, 0.6)
            return yy[vv >= hi_].mean() - yy[vv <= lo_].mean()
        a, bb_, s1, s2 = spread(h1), spread(~h1), spread(seen), spread(~seen)
        stat.append((abs(sp), c, mu, sp, mono, a, bb_, s1, s2))
    stat.sort(reverse=True)
    for _, c, mu, sp, mono, a, bb_, s1, s2 in stat:
        L.append(f"| {c}{' ᵐ' if c in MARKET else ''} | " + " | ".join(f"{m:+.2f}" for m in mu) + (" | –" * (5 - len(mu))) + f" | {sp:+.2f} | {mono} | {'yes' if a * bb_ > 0 else 'NO'} ({a:+.2f} / {bb_:+.2f}) | {'yes' if s1 * s2 > 0 else 'NO'} ({s1:+.2f} / {s2:+.2f}) |")
    L += ["", "ᵐ = market-wide (repeats across same-day trades). Halves / sets columns: top 40 % minus bottom 40 % of the feature inside each part.", ""]
    # 2 · best keep-rule vs shuffles
    Xv = {c: X[c].values.astype(float) for c in FEATS}
    best = keep_rules(Xv, y); null = np.array([keep_rules(Xv, RNG.permutation(y))[0] for _ in range(300)])
    L += ["## 2 · Best single-feature keep-rule vs luck", "",
          f"Best rule keeping ≥ 40 % of trades: **{best[1]} {best[2]} {best[3]:.3g}** → {best[4]} trades at {best[0]:+.2f} per trade (all trades: {y.mean():+.2f}). "
          f"The same search on 300 shuffles of the outcomes finds a 'best rule' of {null.mean():+.2f} on average (95th percentile {np.percentile(null, 95):+.2f}). "
          f"The real best rule beats {((best[0] > null).mean() * 100):.0f}% of the shuffles.", ""]
    # 3 · out-of-sample
    L += ["## 3 · Out of sample — a rule found on one part, frozen, applied to the other", "", "| Found on | best rule there | kept there | applied to | kept · per trade | rest · per trade | all · per trade | holds? |", "|---|---|---|---|---|---|---|---|"]
    for name, m in (("Jan–Apr", h1), ("May–Sep", ~h1), ("SEEN ≥ $100M", seen), ("UNSEEN $20–100M", ~seen)):
        for rank in range(3):
            Xd = {c: v[m] for c, v in Xv.items() if rank == 0 or c not in used}
            if rank == 0:
                used = set()
            bb = keep_rules(Xd, y[m]); used.add(bb[1]); v = Xv[bb[1]]; km = (v <= bb[3]) if bb[2] == "≤" else (v > bb[3]); km = km & ~np.isnan(v); o_ = ~m
            kept, rest = y[o_ & km], y[o_ & ~km]
            L.append(f"| {name} | {bb[1]} {bb[2]} {bb[3]:.3g} | {bb[4]} at {bb[0]:+.2f} | the other part | {len(kept)} · {kept.mean():+.2f} | {len(rest)} · {rest.mean():+.2f} | {y[o_].mean():+.2f} | {'YES' if kept.mean() > y[o_].mean() + 0.05 and kept.mean() > 0 else 'no'} |")
    # 4 · 2D
    bins = {}
    for c in FEATS:
        v = Xv[c]
        if np.isnan(v).mean() < 0.1 and len(np.unique(v[~np.isnan(v)])) >= 6:
            try:
                bins[c] = pd.qcut(pd.Series(v).fillna(np.nanmedian(v)), 3, labels=False, duplicates="drop").values
            except ValueError:
                pass
    names = [c for c in bins if len(np.unique(bins[c])) == 3]

    def best2d(yy):
        hi_, lo_ = (-9, None), (9, None)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                idx = bins[names[a]] * 3 + bins[names[b]]; cnt = np.bincount(idx, minlength=9); sm = np.bincount(idx, weights=yy, minlength=9)
                for k_ in range(9):
                    if cnt[k_] >= 50:
                        mu = sm[k_] / cnt[k_]
                        if mu > hi_[0]:
                            hi_ = (mu, (names[a], k_ // 3, names[b], k_ % 3, int(cnt[k_])))
                        if mu < lo_[0]:
                            lo_ = (mu, (names[a], k_ // 3, names[b], k_ % 3, int(cnt[k_])))
        return hi_, lo_
    hi_, lo_ = best2d(y); nh = []; nl = []
    for _ in range(100):
        a_, b_ = best2d(RNG.permutation(y)); nh.append(a_[0]); nl.append(b_[0])
    T3 = ("low", "mid", "high")
    L += ["", f"## 4 · Every pair of features in 3 × 3 cells ({len(names) * (len(names) - 1) // 2} pairs, cells ≥ 50 trades)", "",
          f"- Best cell: {hi_[1][0]} {T3[hi_[1][1]]} × {hi_[1][2]} {T3[hi_[1][3]]} → {hi_[1][4]} trades at {hi_[0]:+.2f}. Shuffled outcomes give a best cell of {np.mean(nh):+.2f} on average (95th pct {np.percentile(nh, 95):+.2f}).",
          f"- Worst cell: {lo_[1][0]} {T3[lo_[1][1]]} × {lo_[1][2]} {T3[lo_[1][3]]} → {lo_[1][4]} trades at {lo_[0]:+.2f}. Shuffled: {np.mean(nl):+.2f} on average (5th pct {np.percentile(nl, 5):+.2f}).", ""]
    # 5 · path facts
    pk = []; t_pk = []
    for r in X.itertuples():
        hh, ll, cc = paths[r.Index]; e_ = r.entry; first = next((i_ for i_ in range(len(cc)) if ll[i_] <= e_ * 0.97), len(cc)); seg = hh[:max(first, 1)]
        pk.append((seg.max() / e_ - 1) * 100); t_pk.append(int(seg.argmax()) + 1)
    X["mfe"] = pk; win = X[X.net > 0]; los = X[X.how == "stop"]
    L += ["## 5 · What the trades look like after entry (not usable at entry — for the exit discussion)", "",
          f"- Winners ({len(win)}): median result {win.net.median():+.1f} %, top quarter above {win.net.quantile(0.75):+.1f} %, best {win.net.max():+.0f} %.",
          f"- Full stops ({len(los)}): median best point before the stop {los.mfe.median():+.1f} %; {((los.mfe < 1).mean() * 100):.0f}% never reached +1 %.",
          f"- Total result {y.sum():+.0f} points; the best 5 % of trades alone make {np.sort(y)[-max(1, int(n * 0.05)):].sum():+.0f} points.", "",
          "## NOT tested", "", "- Order book / spread / open interest / liquidations at entry (not in the data).", "- News, listings, sector moves.",
          "- Delisted pairs; no fresh time period (the two halves and the two volume groups are the only out-of-sample splits).",
          "- Interactions of three or more features; non-quantile thresholds."]
    X.to_csv(os.path.join(B.BC, "frenzy_long_features.csv"), index=False)
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
