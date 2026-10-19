#!/usr/bin/env python3
"""Follow-up on the only OOS-confirmed P&L family (pair liquidity level): dose-response, per-regime, per-pair/day concentration,
expectancy-bar check, window-clustered bootstrap. Frozen threshold = MASTER median of log10(prior-7d avg daily volume) = 6.689."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/guillearslanian/Downloads/NOFA AI/scripts")
import study_fade_wl_screen as S
A = S.A.copy(); THR = 6.689
rows = []
def st(x):
    return dict(N=len(x), days=x.day.nunique(), pairs=x.pair.nunique(), WR=round((x.pct > 0).mean()*100, 1), stop=round(x.is_stop.mean()*100, 1),
                avg=round(x.pct.mean(), 3), sum=round(x.pct.sum(), 2))
blocks = [("MASTER", A.blk == "MASTER"), ("BT all", A.blk == "BT"), ("BT 2025", A.era == "2025 BULL"), ("BT 26H1", A.era == "2026-H1 BEAR"), ("BT 26H2", A.era == "2026-H2 BULL")]
for var, edges in [("d_log_norm7d", [0, 6.3, 6.5, 6.689, 6.9, 7.2, 99]), ("d_log_vol24h", [0, 6.5, 6.7, 6.9, 7.1, 7.4, 99]), ("d_infl", [0, 0.6, 0.8, 1.0, 1.3, 2.0, 99])]:
    print(f"\n### dose-response {var}  (bins {edges})")
    for bn, bm in blocks:
        D = A[bm]; b = pd.cut(D[var], edges)
        g = D.groupby(b, observed=False).agg(N=("pct", "size"), WR=("pct", lambda x: round((x > 0).mean()*100) if len(x) else -1), avg=("pct", lambda x: round(x.mean(), 3)), stop=("is_stop", lambda x: round(x.mean()*100) if len(x) else -1))
        print(f"  {bn:8s} " + " | ".join(f"{str(i)}: {r.N}·{r.WR}%·{r.avg:+.2f}·s{r.stop}" for i, r in g.iterrows()))
print("\n### frozen zone: prior-7d avg daily vol < $4.89M (log 6.689)")
for bn, bm in blocks:
    D = A[bm].reset_index(drop=True); z = (D.d_log_norm7d < THR).values.astype(float); s = D.d_log_norm7d.notna().values.astype(float)
    bt = S.day_boot(D, z, s, D.pct.values.astype(float), nb=4000)
    zone = D[(z*s) > 0]; rest = D[((1-z)*s) > 0]
    w = rest.pct[rest.pct > 0].mean(); l = rest.pct[rest.pct <= 0].mean()
    pp = zone.groupby("pair").pct.sum().sort_values(); dd = zone.groupby("day").pct.sum().sort_values()
    neg = zone.pct[zone.pct < 0].sum()
    print(f"  {bn:8s} zone {st(zone)} | rest {st(rest)}")
    print(f"           zone avg CI [{np.percentile(bt[:,0],2.5):+.3f},{np.percentile(bt[:,0],97.5):+.3f}] P(zone avg<0)={np.mean(bt[:,0]<0):.2f} · Δ CI [{np.percentile(bt[:,2],2.5):+.3f},{np.percentile(bt[:,2],97.5):+.3f}] P(Δ<0)={np.mean(bt[:,2]<0):.2f}"
          f" · sleeve BE-WR {abs(D.pct[D.pct<=0].mean())/(D.pct[D.pct>0].mean()+abs(D.pct[D.pct<=0].mean()))*100:.1f}% · worst pair {pp.index[0]} {pp.iloc[0]:+.2f} ({pp.iloc[0]/neg*100 if neg<0 else 0:.0f}% of zone loss) · worst day {str(dd.index[0])[:10]} {dd.iloc[0]:+.2f}")
k = A[A.blk == "LIVE"].iloc[0]
print(f"\nKMNO norm7d ${10**k.d_log_norm7d/1e6:.2f}M (log {k.d_log_norm7d:.3f}) -> {'ZONE' if k.d_log_norm7d < THR else 'rest (liquid side)'}; 24h ${10**k.d_log_vol24h/1e6:.1f}M; infl {k.d_infl:.2f}")
# master per batch for the zone
M = A[A.blk == "MASTER"]; Mz = M[M.d_log_norm7d < THR]
print("\nMASTER per batch (zone):"); print(Mz.groupby("era").agg(N=("pct","size"), W=("pct", lambda x:(x>0).sum()), sum=("pct", lambda x: round(x.sum(),2)), losers=("pair", lambda x: ",".join(x[M.loc[x.index,'pct']<0].str.replace('USDT',''))) ).to_string())
print("\nMASTER losers:"); print(M[M.pct < 0][["era","pair","opened_at","pct","d_log_norm7d","d_log_vol24h","d_infl","c_regime_HEALTHY_BEAR"]].assign(norm_M=lambda x: (10**x.d_log_norm7d/1e6).round(2)).to_string(index=False))
