# Frozen before reading: does market-wide volume (engine global volume ratio, rebuilt on closed bars) separate FRENZY / FRENZY_WIDE outcomes?
# Trades: every 100x FRENZY first candle Jan–Sep 2026, LAG 1 entry, live FRENZY exit, REAL costs. FRENZY = ATR ≤ 2.5 ∧ red/flat; WIDE = the rest.
# Reads: sign split at 1.0 (above / below normal), the live long floor 0.7, terciles; halves; day-bootstrap; DAY units (market-wide variable);
# null = random same-size subsets; luck must be < 5 %.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _a, sys.argv = sys.argv, ["x"]
import numpy as np, pandas as pd
import frenzy_scalp_pattern_search_v2 as P2, frenzy_scalp_followup as FU, frenzy_week_combos_year as W
sys.argv = _a; H = P2.H; SPLIT = "2026-05-01"; RNG = np.random.default_rng(11); S = os.environ["S"]
M = pd.read_pickle(P2.CACHE).sort_values(["pair", "t"]).reset_index(drop=True)
M["ep"] = ((M.pair != M.pair.shift()) | (M.hrs < M.hrs.shift()) | ((M.t - M.t.shift()) > 24 * 3600_000)).cumsum()
on = (M.streak >= 12) & (M.volx >= 100)
prev = on.groupby(M.ep).transform(lambda s: s.astype(int).rolling(12, min_periods=1).max().shift(1).fillna(0)).astype(bool)
G = M[on & ~prev].sort_values("t"); free = {}; rows = []; idx = FU.episode_index()
for r in G.itertuples():
    if r.t < free.get(r.pair, 0): continue
    p = FU.path(idx, r.pair, int(r.t))
    if p is None or len(p[0]) < 30: continue
    tt, hh, ll, cc = p; e = cc[0]; hh, ll, cc, tt = hh[1:], ll[1:], cc[1:], tt[1:]
    x, mins, _ = W.walk(hh, ll, cc, e, r.atr, dict(live=True), 0.02)
    rows.append(dict(pair=r.pair, t=int(r.t), day=r.day, real=x - 0.09, atr=r.atr, body=r.body))
    free[r.pair] = int(tt[min(mins, len(tt)) - 1]) + 16 * 60_000
Y = pd.DataFrame(rows); g = pd.read_pickle("reports/backtest_cache/gvr_year.pkl")
Y["gvr"] = g.gvr.reindex(Y.t - 300_000).values; Y["gvr_q"] = g.gvr_q.reindex(Y.t - 300_000).values
Y["fz"] = (Y.atr <= 2.5) & (Y.body <= 0); Y.to_csv(S + "/frenzy_gvr_trades.csv", index=False)
print(f"trades {len(Y)} · with global volume {Y.gvr.notna().sum()} · FRENZY {Y.fz.sum()} · WIDE {(~Y.fz).sum()}\n")
h1 = Y.day < SPLIT
def line(name, Z, base):
    if len(Z) < 15: return f"| {name} | {len(Z)} | – | – | – | – | – | – |"
    bd = H.boot(Z.assign(net=Z.real), "day", 1500)
    nl = np.mean([base.real.values[RNG.choice(len(base), len(Z), replace=False)].mean() >= Z.real.mean() for _ in range(2000)]) if len(Z) < len(base) else np.nan
    z1, z2 = Z[Z.day < SPLIT], Z[Z.day >= SPLIT]
    return (f"| {name} | {len(Z)} · {Z.day.nunique()} days | {(Z.real > 0).mean() * 100:.0f}% | **{Z.real.mean():+.3f}** | {z1.real.mean():+.3f} / {z2.real.mean():+.3f} | "
            f"[{bd[0]:+.3f}, {bd[1]:+.3f}] | {Z.real.sum():+.1f} | {nl * 100:.0f}% |" if len(Z) < len(base) else
            f"| {name} | {len(Z)} · {Z.day.nunique()} days | {(Z.real > 0).mean() * 100:.0f}% | **{Z.real.mean():+.3f}** | {z1.real.mean():+.3f} / {z2.real.mean():+.3f} | [{bd[0]:+.3f}, {bd[1]:+.3f}] | {Z.real.sum():+.1f} | – |")
for sname, B in (("FRENZY (ATR ≤ 2.5 ∧ red)", Y[Y.fz & Y.gvr.notna()]), ("FRENZY_WIDE (the rest)", Y[~Y.fz & Y.gvr.notna()]), ("ALL first candles", Y[Y.gvr.notna()])):
    q1, q2 = B.gvr.quantile([1 / 3, 2 / 3])
    print(f"### {sname}  (global volume terciles {q1:.2f} / {q2:.2f})\n")
    print("| cut | trades · days | won | avg %/trade | Jan–Apr / May–Sep | 95% by day | sum % | random beats it |\n|---|---|---|---|---|---|---|---|")
    for n_, m_ in (("all", B.gvr == B.gvr), ("global vol ≥ 1.0 (above normal)", B.gvr >= 1.0), ("global vol < 1.0", B.gvr < 1.0),
                   ("global vol ≥ 0.7 (live long floor)", B.gvr >= 0.7), ("global vol < 0.7", B.gvr < 0.7),
                   (f"low third (< {q1:.2f})", B.gvr < q1), (f"mid third", (B.gvr >= q1) & (B.gvr < q2)), (f"high third (≥ {q2:.2f})", B.gvr >= q2),
                   ("quote-volume version ≥ 1.0", B.gvr_q >= 1.0), ("quote-volume version < 1.0", B.gvr_q < 1.0)):
        print(line(n_, B[m_], B))
    hi = B[B.gvr >= 1.0]; lo = B[B.gvr < 1.0]
    ph = hi.groupby(hi.day.str[:7]).real.mean(); pl = lo.groupby(lo.day.str[:7]).real.mean(); jj = ph.index.intersection(pl.index)
    print(f"\nmonths where ≥ 1.0 beat < 1.0: {(ph[jj] > pl[jj]).sum()} of {len(jj)} · top-3 days' share of the ≥1.0 sum: "
          f"{hi.groupby('day').real.sum().nlargest(3).sum() / hi.real.sum() * 100 if hi.real.sum() > 0 else float('nan'):.0f}%\n")
