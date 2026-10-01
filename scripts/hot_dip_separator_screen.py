#!/usr/bin/env python3
"""🔥 What separates the hot-state entries that run to +0.59 % from the ones that sink first? (operator, 2026-10-01: "2 % never
come back… something must filter winners vs losers".)

Sample: the FIRST hot 5m bar of every hot episode (scripts/hot_scalp_backtest.frame5: ATR ≥ 2 %, RSI ≥ 70, high ≥ 3 % above EMA5,
24 h volume ≥ $20M), entry = that bar's close, 5m cache Jan-29 → Sep-27 2026. LOSER = the price fell ≥ 10 % below the entry
before touching +0.59 % (adverse-first inside a 5m bar), or never touched it within 7 days. Everything else = winner.
Features: only what is known at that close. Method (pre-declared): quintile edges cut on Jan–Apr, loser rate per quintile in
Jan–Apr AND May–Sep; a feature is a CANDIDATE only if the same end (Q1 or Q5) is its safest in both halves, its safest-quintile
loser rate is below half the base rate in both, and the spread beats a 500× label-shuffle null (best-of-all-features, so the
multiple testing is paid). Then every 2D pair of the candidates' safe ends, confirmed on May–Sep. Units reported: episodes and days.
NOT TESTED (no data in the 5m cache): order-book depth, funding, open interest, taker-buy share, spot/futures basis, news.
Usage: venv/bin/python scripts/hot_dip_separator_screen.py → reports/HOT_DIP_SEPARATORS_2026-10-01.md"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import hot_scalp_backtest as H  # noqa: E402
sys.argv = _a
OUT = os.path.join(H.ROOT, "reports", "HOT_DIP_SEPARATORS_2026-10-01.md")
SPLIT_MS = int(pd.Timestamp("2026-05-01").timestamp() * 1000)


def build():
    b = pd.read_csv(os.path.join(H.K5, "BTCUSDT.csv")).drop_duplicates("open_time").set_index("open_time").sort_index()
    b4, b24 = (b.c / b.c.shift(48) - 1) * 100, (b.c / b.c.shift(288) - 1) * 100
    rows = []
    for f in sorted(glob.glob(os.path.join(H.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT"):
            continue
        d, eps = H.frame5(pair)
        if not eps:
            continue
        idx = d.index.values; o, h, l, c, q = d.o.values, d.h.values, d.l.values, d.c.values, d.qvol.values
        pos = {t: i for i, t in enumerate(idx)}
        hotbar = ((d.atr >= 2) & (d.rsi >= 70) & (d.h >= d.ema5 * 1.03) & (d.q24 >= 20e6)).values
        for t0, _ in eps:
            i = pos[t0]; e = c[i]; j1 = min(i + 1 + 288 * 7, len(c))
            if j1 - i < 289 or i < 288 * 8:
                continue
            Hh, Ll = h[i + 1:j1], l[i + 1:j1]
            ht = np.nonzero(Hh >= e * 1.0059)[0]; t = int(ht[0]) if len(ht) else None
            mae = (Ll[:(t + 1) if t is not None else len(Ll)].min() / e - 1) * 100
            w24 = slice(i - 287, i + 1); lo24 = int(np.argmin(l[w24])); medq = np.median(q[i - 288:i])
            rows.append(dict(pair=pair, t0=int(t0), day=pd.Timestamp(t0, unit="ms").strftime("%Y-%m-%d"),
                             loser=bool(t is None or mae <= -10), never=bool(t is None), mae=mae,
                             atr=d.atr.values[i], rsi=d.rsi.values[i], stretch=(e / d.ema5.values[i] - 1) * 100,
                             ret_1h=(e / c[i - 12] - 1) * 100, ret_4h=(e / c[i - 48] - 1) * 100, ret_24h=(e / c[i - 288] - 1) * 100,
                             ret_72h=(e / c[i - 864] - 1) * 100, ret_7d=(e / c[i - 2016] - 1) * 100,
                             vol24_usd_log=np.log10(max(d.q24.values[i], 1)), vol_vs_week=d.q24.values[i] / max(d.q7.values[i], 1),
                             bar_vol_mult=q[i] / medq if medq > 0 else np.nan, vol_1h_share=q[i - 11:i + 1].sum() / max(q[w24].sum(), 1) * 100,
                             off_24h_high=(e / h[w24].max() - 1) * 100, up_from_24h_low=(e / l[w24].min() - 1) * 100,
                             hours_since_24h_low=(287 - lo24) / 12, upper_wick=(h[i] - e) / (h[i] - l[i]) if h[i] > l[i] else 0.0,
                             bar_ret=(e / o[i] - 1) * 100, bar_range=(h[i] / l[i] - 1) * 100,
                             hot_bars_prior_24h=int(hotbar[i - 288:i].sum()), hot_bars_prior_7d=int(hotbar[i - 2016:i].sum()),
                             btc_ret_4h=float(b4.get(t0, np.nan)), btc_ret_24h=float(b24.get(t0, np.nan)),
                             hour_utc=pd.Timestamp(t0, unit="ms").hour, listed_days=i / 288))
    return pd.DataFrame(rows)


def quint(T, f, edges):
    return np.clip(np.searchsorted(edges, T[f].values, side="right"), 0, 4)


if __name__ == "__main__":
    T = build(); T = T.replace([np.inf, -np.inf], np.nan)
    A, B = T[T.t0 < SPLIT_MS], T[T.t0 >= SPLIT_MS]; feats = [c for c in T.columns if c not in ("pair", "t0", "day", "loser", "never", "mae")]
    L = ["# 🔥 Hot-state entries: what separates the ones that sink from the ones that run?", "",
         f"{len(T):,} first-entries of hot episodes · {T.pair.nunique()} pairs · {T.day.nunique()} days. LOSER = fell ≥ 10 % before +0.59 %, or never reached it in 7 days: "
         f"**{T.loser.mean() * 100:.1f} %** (Jan–Apr {A.loser.mean() * 100:.1f} % of {len(A):,} · May–Sep {B.loser.mean() * 100:.1f} % of {len(B):,}); never reached: {T.never.mean() * 100:.1f} %.",
         "Quintile edges are cut on Jan–Apr and applied unchanged to May–Sep.", ""]
    res = {}
    for f in feats:
        ed = np.nanquantile(A[f], [.2, .4, .6, .8]); qa, qb = quint(A, f, ed), quint(B, f, ed)
        ra = [A.loser.values[qa == k].mean() if (qa == k).sum() >= 30 else np.nan for k in range(5)]
        rb = [B.loser.values[qb == k].mean() if (qb == k).sum() >= 30 else np.nan for k in range(5)]
        res[f] = (ed, ra, rb, [(qb == k).sum() for k in range(5)])
    rng = np.random.default_rng(7); null = []
    for _ in range(500):                                               # best spread any feature shows on shuffled labels (Jan–Apr)
        y = rng.permutation(A.loser.values); best = 0
        for f in feats:
            qa = quint(A, f, res[f][0]); r = [y[qa == k].mean() for k in range(5) if (qa == k).sum() >= 30]
            best = max(best, max(r) - min(r)) if r else best
        null.append(best)
    thr = float(np.quantile(null, 0.95))
    L += [f"Shuffle null: the best spread (worst − safest quintile) ANY feature shows by luck on Jan–Apr = {thr * 100:.1f} points (95th pct).", "",
          "| Feature | loser % by quintile, Jan–Apr (low → high) | May–Sep | spread Jan–Apr / May–Sep | safest end same? | candidate |", "|---|---|---|---|---|---|"]
    cands = []; base_a, base_b = A.loser.mean(), B.loser.mean()
    for f in sorted(feats, key=lambda f: -(np.nanmax(res[f][2]) - np.nanmin(res[f][2]))):
        ed, ra, rb, nb = res[f]; sa, sb = np.nanmax(ra) - np.nanmin(ra), np.nanmax(rb) - np.nanmin(rb)
        ka, kb = int(np.nanargmin(ra)), int(np.nanargmin(rb)); same = ka == kb and ka in (0, 4)
        ok = same and ra[ka] < base_a / 2 and rb[kb] < base_b / 2 and sa > thr
        if ok:
            cands.append((f, ka, ed))
        L.append(f"| {f} | " + " · ".join("–" if np.isnan(x) else f"{x * 100:.0f}" for x in ra) + " | " + " · ".join("–" if np.isnan(x) else f"{x * 100:.0f}" for x in rb)
                 + f" | {sa * 100:.0f} / {sb * 100:.0f} | {'yes (Q' + str(ka + 1) + ')' if same else 'no'} | {'✅' if ok else '—'} |")
    L += ["", "## Candidates — the safe end of each, read on May–Sep (selection used both halves, so not a clean hold-out)", "",
          "| Rule (edge from Jan–Apr) | May–Sep entries | days | loser % | never reached % | 10× no-stop: average per trade (of margin) | 5× |", "|---|---|---|---|---|---|---|"]
    ev = lambda g, lev: (1 - g.loser.mean()) * 0.5 * lev - g.loser.mean() * 100
    L.append(f"| (no filter) | {len(B):,} | {B.day.nunique()} | {B.loser.mean() * 100:.1f} | {B.never.mean() * 100:.1f} | {ev(B, 10):+.2f}% | {ev(B, 5):+.2f}% |")
    masks = {}
    for f, k, ed in cands:
        m = (lambda X, f=f, k=k, ed=ed: (X[f] < ed[0]) if k == 0 else (X[f] >= ed[3]))
        masks[f"{f} {'<' if k == 0 else '≥'} {ed[0] if k == 0 else ed[3]:.3g}"] = m
    for name, m in masks.items():
        g = B[m(B)]
        L.append(f"| {name} | {len(g):,} | {g.day.nunique()} | {g.loser.mean() * 100:.1f} | {g.never.mean() * 100:.1f} | {ev(g, 10):+.2f}% | {ev(g, 5):+.2f}% |")
    names = list(masks)
    for a in range(len(names)):
        for b_ in range(a + 1, len(names)):
            g = B[masks[names[a]](B) & masks[names[b_]](B)]; ga = A[masks[names[a]](A) & masks[names[b_]](A)]
            if len(g) >= 60 and len(ga) >= 60:
                L.append(f"| {names[a]} ∧ {names[b_]} | {len(g):,} | {g.day.nunique()} | {g.loser.mean() * 100:.1f} (Jan–Apr {ga.loser.mean() * 100:.1f}) | {g.never.mean() * 100:.1f} | {ev(g, 10):+.2f}% | {ev(g, 5):+.2f}% |")
    L += ["", "A win at 10× = +5 % of the margin (+0.5 % net); a loser = the margin. Break-even loser rate: 4.8 % at 10×, 2.4 % at 5× "
          "(leverage does not change it much: lower leverage survives deeper dips, but this table's LOSER is fixed at −10 %)."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")
