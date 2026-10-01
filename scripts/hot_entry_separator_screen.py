#!/usr/bin/env python3
"""🔥 What separates the hot-state entries that hit +0.59 % first from the ones that hit −1.11 % first? (operator, 2026-10-01:
"find what separates winners vs losers" — his own 38 clicks hit the target first 79 % of the time, the blind rule 65 %.)

Sample: LONG entries at HOT seconds 30 s apart in every cached 1-second episode (hot_scalp_backtest cache). WIN = +0.59 % before
−1.11 % (30-min limit; a time-out counts by its sign). Features = only what is known at the entry second:
  seconds scale   ret_5s/15s/30s/60s/120s/300s · position in the last 60 s / 300 s range · pullback from the 5-min high and seconds
                  since it · speed (path length of the last 15 s ÷ its 5-min average) · choppiness 60 s (net ÷ path) · share of
                  up-seconds in 30 s · worst dip of the last 60 s · higher-low (15 s low vs the 15 s before) · accel (15 s vs 60 s pace)
  episode         seconds since the episode's first hot second · hot seconds so far
  5-minute        ATR · RSI · stretch above EMA5 · 24 h return · 72 h return · 24 h volume (log) · volume vs its week · hour UTC
Method (pre-declared): quintile edges cut on Jan–Apr, win rate per quintile in Jan–Apr AND May–Sep, in EPISODE-weighted terms
(each episode's entries average first, so one long episode cannot dominate); CANDIDATE = the same end quintile is best in both
halves ∧ its lift over the base rate ≥ 4 points in both ∧ the Jan–Apr spread beats a 300× shuffle null (labels shuffled by
EPISODE; best-of-all-features). Then all 2D pairs of candidate ends on May–Sep with their net % at the measured cost (0.11 %).
NOT TESTED: entries in the first 5 minutes of a window (the 300-s look-back needs them); everything is EPISODE-weighted (a trade-weighted read is ~10 points higher on target-first and less favourable to dip rules);
also not in this cache: order flow / taker-buy share, trade count, order-book depth, funding, open interest.
Usage: venv/bin/python scripts/hot_entry_separator_screen.py → reports/HOT_ENTRY_SEPARATORS_2026-10-01.md"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import hot_scalp_backtest as H  # noqa: E402
sys.argv = _a
OUT = os.path.join(H.ROOT, "reports", "HOT_ENTRY_SEPARATORS_2026-10-01.md")
COST = 0.11


def feats(pair, d5, f):
    z = np.load(f); ts, h, l, c = z["t"], z["h"], z["l"], z["c"]
    st = d5.reindex(ts // H.BAR * H.BAR)
    atr, rsi, ema5, q24, r24, q7 = (st[k].values for k in ("atr", "rsi", "ema5", "q24", "r24", "q7"))
    hot = (atr >= 2) & (rsi >= 70) & (c >= ema5 * 1.03) & (q24 >= 20e6)
    hi = np.nonzero(hot)[0]
    if not len(hi):
        return []
    step = np.abs(np.diff(c, prepend=c[0])) / c; cs = np.cumsum(step); up = np.cumsum(np.diff(c, prepend=c[0]) > 0)
    i5 = d5.index.get_indexer([int(ts[0]) // H.BAR * H.BAR])[0]
    r72 = (d5.c.values[i5 - 1] / d5.c.values[i5 - 865] - 1) * 100 if i5 > 865 else np.nan
    out = []; last = -10**9; first = hi[0]
    for n_hot, i in enumerate(hi):
        if i - last < 30 or i < 305 or i > len(c) - 120:
            continue
        last = i; e = c[i]
        r, _, how = H.walk(h, l, c, int(i), (0.59, 1.11, None), e)
        ret = lambda n: (e / c[i - n] - 1) * 100
        w60, w300 = slice(i - 60, i + 1), slice(i - 300, i + 1)
        hi60, lo60, hi300, lo300 = h[w60].max(), l[w60].min(), h[w300].max(), l[w300].min()
        path15 = cs[i] - cs[i - 15]; path300 = (cs[i] - cs[i - 300]) / 20; path60 = cs[i] - cs[i - 60]
        k_hi = i - 300 + int(np.argmax(h[w300]))
        out.append(dict(pair=pair, eid=os.path.basename(f)[:-4], t=int(ts[i]), win=bool(r > 0), r=r,
                        ret_5s=ret(5), ret_15s=ret(15), ret_30s=ret(30), ret_60s=ret(60), ret_120s=ret(120), ret_300s=ret(300),
                        pos_60s=(e - lo60) / (hi60 - lo60) * 100 if hi60 > lo60 else 50, pos_300s=(e - lo300) / (hi300 - lo300) * 100 if hi300 > lo300 else 50,
                        pullback_from_5m_high=(e / hi300 - 1) * 100, secs_since_5m_high=i - k_hi,
                        speed=path15 / path300 if path300 > 0 else np.nan, chop_60s=abs(e / c[i - 60] - 1) / path60 if path60 > 0 else np.nan,
                        up_share_30s=(up[i] - up[i - 30]) / 30 * 100, worst_dip_60s=(lo60 / hi60 - 1) * 100,
                        higher_low=(l[i - 15:i + 1].min() / l[i - 30:i - 15].min() - 1) * 100, accel=ret(15) - ret(60) / 4,
                        range_60s=(hi60 / lo60 - 1) * 100, secs_in_episode=int(i - first), hot_secs_so_far=n_hot,
                        atr_5m=atr[i], rsi_5m=rsi[i], stretch=(e / ema5[i] - 1) * 100, ret_24h=r24[i], ret_72h=r72,
                        vol24_log=np.log10(max(q24[i], 1)), vol_vs_week=q24[i] / max(q7[i], 1) if np.isfinite(q7[i]) else np.nan,
                        hour_utc=pd.Timestamp(int(ts[i]), unit="ms").hour))
    return out


def ep_rate(X, mask=None):
    g = X if mask is None else X[mask]
    return g.groupby("eid").win.mean().mean() if len(g) else np.nan


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(H.RAW, "*.npz"))); rows = []; cache = {}
    for n_, f in enumerate(files):
        pair = os.path.basename(f)[:-4].rsplit("_", 1)[0]
        if pair not in cache:
            cache = {pair: H.frame5(pair)[0]}
        rows += feats(pair, cache[pair], f)
        if n_ % 300 == 0:
            print(f"{n_}/{len(files)}", flush=True)
    T = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan); T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d")
    T.to_csv(os.path.join(H.ROOT, "reports", "backtest_cache", "hot_entry_features.csv"), index=False)
    A, B = T[T.day < H.SPLIT], T[T.day >= H.SPLIT]; F = [c for c in T.columns if c not in ("pair", "eid", "t", "win", "r", "day")]
    ba, bb = ep_rate(A), ep_rate(B)
    L = ["# 🔥 Hot-state entries — what separates target-first from stop-first?", "",
         f"{len(T):,} long entries (30 s apart) in {T.eid.nunique():,} episodes · {T.pair.nunique()} pairs · {T.day.nunique()} days. Target-first rate, episode-weighted: "
         f"**Jan–Apr {ba * 100:.1f} % · May–Sep {bb * 100:.1f} %** (needs 72 % to pay at +0.59 / −1.11 with 0.11 % costs).", ""]
    res = {}
    for f in F:
        ed = np.nanquantile(A[f], [.2, .4, .6, .8]); qa = np.searchsorted(ed, A[f].values, side="right"); qb = np.searchsorted(ed, B[f].values, side="right")
        res[f] = (ed, [ep_rate(A, qa == k) for k in range(5)], [ep_rate(B, qb == k) for k in range(5)], qa)
    rng = np.random.default_rng(3); eids = A.eid.unique(); er = A.groupby("eid").win.mean(); null = []
    for _ in range(300):                                               # shuffle whole EPISODES' labels: an episode keeps its own win rate pattern
        sh = dict(zip(eids, rng.permutation(eids))); y = A.assign(w=A.win.values)  # permute episode identities → features decoupled from outcomes
        perm = A.eid.map(sh).map(er).values; best = 0
        for f in F:
            qa = res[f][3]; v = [perm[qa == k].mean() for k in range(5) if (qa == k).sum() > 50]
            best = max(best, max(v) - min(v)) if v else best
        null.append(best)
    thr = float(np.quantile(null, 0.95))
    L += [f"Shuffle null (episodes' outcomes reassigned at random): best spread any feature shows by luck on Jan–Apr = {thr * 100:.1f} points.", "",
          "| Feature | target-first % by quintile, Jan–Apr (low → high) | May–Sep | best end same? | lift Jan–Apr / May–Sep | candidate |", "|---|---|---|---|---|---|"]
    cands = []
    for f in sorted(F, key=lambda f: -(np.nanmax(res[f][2]) - np.nanmin(res[f][2]))):
        ed, ra, rb, _ = res[f]; ka, kb = int(np.nanargmax(ra)), int(np.nanargmax(rb)); same = ka == kb and ka in (0, 4)
        la, lb = ra[ka] - ba, rb[kb] - bb; ok = same and la >= 0.04 and lb >= 0.04 and (np.nanmax(ra) - np.nanmin(ra)) > thr
        if ok:
            cands.append((f, ka, ed))
        L.append(f"| {f} | " + " · ".join(f"{x * 100:.0f}" for x in ra) + " | " + " · ".join(f"{x * 100:.0f}" for x in rb)
                 + f" | {'yes (Q' + str(ka + 1) + ')' if same else 'no'} | {la * 100:+.0f} / {lb * 100:+.0f} | {'✅' if ok else '—'} |")
    L += ["", "## Candidates, read on May–Sep (NOT unseen: selection already required the same end to win in both halves — biased upward)", "",
          "| Rule (edge from Jan–Apr) | entries | episodes | days | target first | net % / trade | by day [95 %] |", "|---|---|---|---|---|---|---|"]

    def row(name, g):
        if g.eid.nunique() < 30:
            return f"| {name} | {len(g):,} | {g.eid.nunique()} | – | – | – | too few |"
        e = g.groupby(["day", "eid"]).r.mean() - COST; dm = e.groupby(level=0).mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n); q = H.tq(0.975, n)
        return (f"| {name} | {len(g):,} | {g.eid.nunique()} | {n} | {ep_rate(g) * 100:.1f}% | {(g.groupby('eid').r.mean() - COST).mean():+.3f} | "
                f"{dm.mean():+.3f} [{dm.mean() - q * se:+.3f}, {dm.mean() + q * se:+.3f}] |")
    L.append(row("(no filter)", B))
    M = {f"{f} {'<' if k == 0 else '≥'} {ed[0] if k == 0 else ed[3]:.3g}": (lambda X, f=f, k=k, ed=ed: (X[f] < ed[0]) if k == 0 else (X[f] >= ed[3])) for f, k, ed in cands}
    for nm, m in M.items():
        L.append(row(nm, B[m(B)]))
    nms = list(M)
    for a in range(len(nms)):
        for b in range(a + 1, len(nms)):
            L.append(row(f"{nms[a]} ∧ {nms[b]}", B[M[nms[a]](B) & M[nms[b]](B)]))
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")
