#!/usr/bin/env python3
"""B18 study — the FROZEN Oct-4 watch-flag combinations (scripts/ml_watch_combo_screen.py definitions, never re-tuned) that catch
B18's fills, re-read on yr5 (trimmed; ⚠ same calendar as the yr4 discovery run → re-validation, not independent OOS) and on master
(scored subset only; forward fills after the Oct-4 screen = the only genuinely out-of-sample rows)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_b18_common as C
n = lambda d, c: pd.to_numeric(d[c], errors="coerce")
RNG = np.random.default_rng(5)


def fl(d):
    sl = n(d, "entry_btc_1h_slope")
    F = dict(NEG=(sl <= -0.05), SLOPE=(sl < 0),
             ZC=(n(d, "entry_btc_ema50_100_gap_pct") <= 0.006) & (n(d, "entry_eth_5m_ret1_pct") <= 0),
             OFF24=n(d, "entry_btc_off24h_pct") <= -2,
             HOT=(n(d, "entry_btc_adx") >= 25) & ((n(d, "entry_btc_atr_pct") >= .15) | (n(d, "entry_btc_dist_from_ema13_pct") >= .20)))
    sc = dict(NEG=sl.notna(), SLOPE=sl.notna(), ZC=n(d, "entry_btc_ema50_100_gap_pct").notna() & n(d, "entry_eth_5m_ret1_pct").notna(),
              OFF24=n(d, "entry_btc_off24h_pct").notna(), HOT=n(d, "entry_btc_adx").notna())
    return F, sc


COMBOS = [("HOT",), ("OFF24",), ("SLOPE", "OFF24"), ("ZC", "OFF24"), ("SLOPE", "ZC", "OFF24"), ("NEG", "HOT"), ("NEG", "OFF24"), ("SLOPE", "HOT")]
M = C.master(); Y = C.yr5(); NS = Y.seed.nunique()
FM, SM = fl(M); FY, SY = fl(Y)


def dnull(d, mask, k=1000):
    d = d.assign(_m=mask).sort_values(["day", "o"])
    p = d.pct.values; m = d._m.values
    ud, st = np.unique(d.day.values, return_index=True)
    bl = [p[s:e] for s, e in zip(st, list(st[1:]) + [len(p)])]
    obs = p[m].mean() - p[~m].mean(); c = 0
    for _ in range(k):
        q = np.concatenate([bl[j] for j in RNG.permutation(len(bl))]); c += (q[m].mean() - q[~m].mean()) <= obs
    return obs, c / k


print("| combo (frozen Oct-4 defs) | B18 hits | yr5 ex-washed zone N/seed · WR · avg · days · P(<0) | rest | Δ · day-null p | H1Δ / H2Δ | months Δ<0 | master scored: zone N·WR·avg (rest) · scored N | master forward ≥ Oct-4 |")
print("|---|---|---|---|---|---|---|---|---|")
for cb in COMBOS:
    my = np.logical_and.reduce([FY[c].fillna(False) for c in cb]); ye = (~Y.washed30).values
    d = Y[ye]; m = my[ye]
    st = C.stats(d[m], NS); obs, p = dnull(d, m)
    hs = [d[m & (d.half == h).values if "half" in d else m].pct.mean() for h in ("H1",)]
    med = d.o.quantile(0.5); h1 = (d.o < med).values
    H = [d[m & h1].pct.mean() - d[~m & h1].pct.mean(), d[m & ~h1].pct.mean() - d[~m & ~h1].pct.mean()]
    mo = d.o.dt.strftime("%Y-%m").values
    mons = [(d[m & (mo == x)].pct.mean() - d[~m & (mo == x)].pct.mean()) for x in np.unique(mo) if (m & (mo == x)).sum() >= 3]
    mm = np.logical_and.reduce([FM[c].fillna(False) for c in cb]); sc = np.logical_and.reduce([SM[c] for c in cb])
    me = (~M.washed30).values & sc
    z, r = M[me & mm], M[me & ~mm]
    fw = M[(M.o >= "2026-10-04") & mm]
    hits = ",".join(M[M.b18 & mm].pair.str.replace("USDT", ""))
    print(f"| {' ∧ '.join(cb)} | {hits or '–'} | {st['N']:.0f} · {st['WR']:.0f}% · {st['avg']:+.3f} · {st['days']}d · {st['p_neg']:.2f} | {d[~m].pct.mean():+.3f} | {obs:+.3f} · {p:.3f} | "
          f"{H[0]:+.3f} / {H[1]:+.3f} | {sum(x < 0 for x in mons)}/{len(mons)} | {len(z)}·{(z.pct>0).mean()*100 if len(z) else 0:.0f}%·{z.pct.mean() if len(z) else float('nan'):+.3f} ({r.pct.mean():+.3f}) · {int(me.sum())} | "
          f"{len(fw)}·{(fw.pct>0).mean()*100 if len(fw) else 0:.0f}%·{fw.pct.mean() if len(fw) else float('nan'):+.3f} |")

print("\n2x2 NEG × HOT and NEG × OFF24 (yr5 ex-washed per seed; master ex-washed scored) + bar components")
for other in ("HOT", "OFF24"):
    for a in (True, False):
        for b in (True, False):
            my = (FY["NEG"].fillna(False) == a) & (FY[other].fillna(False) == b) & ~Y.washed30
            mm = (FM["NEG"].fillna(False) == a) & (FM[other].fillna(False) == b) & ~M.washed30 & SM[other]
            sy, sm = C.stats(Y[my], NS), C.stats(M[mm])
            print(f"NEG={a!s:5} {other}={b!s:5} | yr5 {sy['N']:.0f}·{sy['WR']:.0f}%·{sy['avg']:+.3f}·{sy['days']}d·P {sy['p_neg']:.2f}·maxday {sy['maxday']:.0f}%·maxpair {sy['maxpair']:.0f}% "
                  f"| master {sm.get('N',0)}·{sm.get('WR',0):.0f}%·{sm.get('avg',float('nan')):+.3f}")
for cb in [("HOT",), ("SLOPE", "ZC", "OFF24"), ("OFF24",)]:
    my = np.logical_and.reduce([FY[c].fillna(False) for c in cb]) & ~Y.washed30
    s = C.stats(Y[my], NS)
    print(cb, f"yr5 bar: WR {s['WR']:.0f} vs BE {C.be_wr(Y[~Y.washed30]):.1f} · P {s['p_neg']:.3f} · days {s['days']} · N/seed {s['N']:.0f} · maxday {s['maxday']:.0f}% · maxpair {s['maxpair']:.0f}%")
    for sd in sorted(Y.seed.unique()):
        g = Y[my & (Y.seed == sd)]; print("   seed", sd, len(g), f"{g.pct.mean():+.3f}")
