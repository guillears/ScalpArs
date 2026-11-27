#!/usr/bin/env python3
"""HT_BLOCKERS_DEEP_2026-11-27 — is each armed HT blocker correct? (operator: "deep dive on these 20 … backtest to decide whether the
filter is correct or not"). Research only. Input: reports/HT_BLOCKERS_AUDIT_2026-11-27.csv (1,142 unfiltered rebuild HT signals priced at
today's exit TP +6 / SL −4 / 24 h).
CRITERIA (set after the audit's sole-blocker column was on screen — NOT pre-registered; review): each blocker is judged on the signals the OTHER two let through (= the decision it makes today):
  FAILED_SPIKE  vs_vwap ≤ 0 blocks — theory: a red bar below the spike VWAP = the spike failed → worse; deeper below = worse (monotone).
  NEAR_HIGH     dip < 1.4 blocks — theory: buying right under the high needs a breakout → worse; smaller dip = worse.
  BURST         pre_vol > 1.60 ∧ cloc15 < 0.50 blocks — theory: a volume burst that closed weak = distribution → worse.
Correct = the blocked side is below HT breakeven, worse than the kept side in BOTH blocks (DESIGN < 2025-07 ≤ HOLDOUT) and in most
half-years, P(blocked mean < 0) ≥ 0.95 by a DAY bootstrap, gap beats a within-month label shuffle at α 0.05 (one-sided), no pair / day ≥ 50 % of the blocked loss.
Buckets: fixed edges chosen before the read (no re-fit)."""
import os
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = pd.read_csv(os.path.join(R, "reports", "HT_BLOCKERS_AUDIT_2026-11-27.csv"))
D["month"] = pd.to_datetime(D.te, unit="ms").dt.strftime("%Y-%m")
w, l = D[D.net > 0].net.mean(), -D[D.net <= 0].net.mean(); BE = l / (w + l)
rng = np.random.default_rng(1127)


def pneg(x, day, n=3000):
    u = np.unique(day); idx = {k: np.nonzero(day == k)[0] for k in u}
    return float(np.mean([np.concatenate([x[idx[k]] for k in rng.choice(u, len(u))]).mean() < 0 for _ in range(n)]))


def luck(sub, flag, n=2000):
    obs = sub[~flag].net.mean() - sub[flag].net.mean()
    m = sub.month.values; f = flag.values.copy(); hits = 0
    for _ in range(n):
        g = f.copy()
        for k in np.unique(m):
            ix = np.nonzero(m == k)[0]; g[ix] = rng.permutation(g[ix])
        hits += (sub.net.values[~g].mean() - sub.net.values[g].mean()) >= obs
    return obs, hits / n


def share(x, key):
    loss = -x[x < 0]
    if not len(loss): return 0.0
    return float(pd.Series(loss.values, index=key[x < 0].values).groupby(level=0).sum().max() / loss.sum())


def one_per_coin_mask(d, c):
    """per side (blocked / kept): skip a signal while the same coin's previous counted signal on that side is still open. d sorted by te."""
    keep = pd.Series(True, index=d.index)
    for side in (True, False):
        last = {}
        for i, p, te, xt in zip(d.index[d[c] == side], d.pair[d[c] == side], d.te[d[c] == side], d.xt[d[c] == side]):
            if p in last and te < last[p]:
                keep[i] = False; continue
            last[p] = xt
    return keep


def s(g):
    return f"{len(g)} · {(g.net > 0).mean():.0%} · {g.net.mean():+.2f}" if len(g) else "0"


L = [f"HT breakeven WR at TP +6 / SL −4 (all 1,142): {BE:.1%} (avg win {w:+.2f} · avg loss −{l:.2f})", ""]
spec = [("FAILED_SPIKE", "FS", ["NH", "BU"], "vs_vwap", [-1e9, -3, -1, 0, 1, 3, 1e9]),
        ("NEAR_HIGH", "NH", ["FS", "BU"], "dip", [-1e9, 0.5, 1.0, 1.4, 2.0, 4.0, 1e9]),
        ("BURST", "BU", ["FS", "NH"], "cloc", [-1e9, 0.25, 0.5, 0.75, 1e9])]
for name, c, oth, var, edges in spec:
    sub = D[~D[oth].any(axis=1)].copy()
    sub = sub[one_per_coin_mask(sub, c)]                          # (review) one position per coin on BOTH sides
    b, k = sub[sub[c]], sub[~sub[c]]
    obs, p = luck(sub, sub[c])
    L += [f"## {name} — judged on the {len(sub)} signals the other two let through",
          f"blocked {s(b)} | kept {s(k)} · gap {obs:+.2f} · luck p {p:.3f} · P(blocked mean<0) {pneg(b.net.values, b.day.values):.2f} · "
          f"days {b.day.nunique()} · top pair {share(b.net, b.pair):.0%} / day {share(b.net, b.day):.0%} of the blocked loss · WR vs breakeven {(b.net > 0).mean():.0%} vs {BE:.0%}",
          "block: " + " · ".join(f"{bl} blocked {s(b[b.blk == bl])} vs kept {s(k[k.blk == bl])}" for bl in ("DESIGN", "HOLDOUT")),
          "half-years (blocked avg | kept avg): " + " · ".join(f"{h} {b[b.half == h].net.mean():+.2f} ({(b.half == h).sum()}) | {k[k.half == h].net.mean():+.2f}" for h in sorted(D.half.unique())),
          "2026 only: blocked " + s(b[b.month >= "2026-01"]) + " | kept " + s(k[k.month >= "2026-01"])]
    pool = sub if c != "BU" else sub[sub.pre_vol > 1.60]
    cut = pd.cut(pool[var], edges, right=False)                   # (review) [lo, hi): the edge value lands with the side the rule puts it
    L.append(f"dose-response by {var}" + (" (bursts only, pre_vol > 1.60)" if c == "BU" else "") + ": " +
             " · ".join(f"{iv}: {s(g)}" for iv, g in pool.groupby(cut, observed=True)))
    L.append("")
open(os.path.join(R, "reports", "HT_BLOCKERS_DEEP_2026-11-27_summary.txt"), "w").write("\n".join(L))
print("\n".join(L))
