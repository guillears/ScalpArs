#!/usr/bin/env python3
"""EXHAUSTIVE pairwise interaction scan (operator 2026-09-29: "we keep having findings" — the 1D screen + top-25 2D missed the
ADX<21 ∧ RSI-falling gate because both legs are useless alone). Every variable × every variable, both binarisations (median on
H1; sign at 0 where the variable crosses 0), all 4 quadrants, vectorised. Survivor = quadrant worse than the rest in H1 AND H2
(Δ ≤ --dmin, N ≥ --nmin fills per half) AND master direction agrees (Δ<0, N ≥ 5). Null = same scan with labels shuffled within day.
Post-gate world (the shipped LOADX gate applied to both sources). Output: reports/EXHAUSTIVE_2D_<sleeve>.csv
"""
import argparse, os, sys, itertools
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--fills", default="reports/backtest_cache/replay/year/yr3_report_fills.csv")
ap.add_argument("--dmin", type=float, default=-0.10); ap.add_argument("--nmin", type=int, default=60); ap.add_argument("--perm", type=int, default=5); ap.add_argument("--sleeve", default="MOM-long"); ap.add_argument("--top", type=int, default=25); ap.add_argument("--nm", type=int, default=5); ap.add_argument("--dmm", type=float, default=0.0)
A = ap.parse_args()
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG, entry_feature_factory as EF
M = LG.build(); sys.argv = _a
M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"), M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
def sleeve_of(strat, d):
    s = strat if isinstance(strat, str) and strat else "MOMENTUM"
    if s.startswith("FLIP"): return "FLIP-short"
    if s == "MOMENTUM": return "MOM-long" if d == "LONG" else "MOM-short"
    return {"SPIKE_FADE": "Spike-Fade", "BULLRUN_LONG": "BullRun-Long", "BEARRUN_SHORT": "BearRun-Short"}.get(s, s)
M["sleeve"] = [sleeve_of(s, d) for s, d in zip(M.entry_strategy, M.direction)]; M = M[M.sleeve == A.sleeve].reset_index(drop=True)
F = pd.read_csv(A.fills, low_memory=False)
F = F[F.cell_multiplier_source.astype(str) != "CROSS_OB_OPEN"]; _b = pd.to_numeric(F.entry_btc_atr_pct, errors="coerce")
F = F[~((F.cell_multiplier_source.astype(str) == "NONEXP_CALM3D") & (_b < 0.08))]
F = F[F.sleeve == A.sleeve]
n = lambda d, c: pd.to_numeric(d[c], errors="coerce")
if A.sleeve == "MOM-long":
    F = F[~((n(F, "entry_adx") < 21) & (n(F, "entry_rsi") < n(F, "entry_rsi_prev")))]   # post-gate world (master already is)
F = F.reset_index(drop=True).copy()
F["pct"] = F.pnl_percentage; F["day"] = F.opened_at.astype(str).str[:10]; F["h1"] = (F.opened_at.astype(str) < "2026-05-01").values
S = F.seed.nunique()
def stamped(d):
    X = {c: n(d, c) for c in d.columns if c.startswith("entry_") and pd.api.types.is_numeric_dtype(n(d, c)) and n(d, c).notna().mean() > 0.7 and d[c].nunique() > 5
         and c not in ("entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_mcap_usd", "entry_cmc_rank")}
    X["d_rsi"] = n(d, "entry_rsi") - n(d, "entry_rsi_prev"); X["d_adx"] = n(d, "entry_adx") - n(d, "entry_adx_prev"); X["d_btc_rsi"] = n(d, "entry_btc_rsi") - n(d, "entry_btc_rsi_prev")
    X["d_btc_adx"] = n(d, "entry_btc_adx") - n(d, "entry_btc_adx_prev"); X["d_btc_rsi_30m"] = n(d, "entry_btc_rsi") - n(d, "entry_btc_rsi_prev6"); X["di_spread"] = n(d, "entry_pos_di") - n(d, "entry_neg_di")
    X["log_vol"] = np.log10(n(d, "entry_pair_volume_24h_usd").where(lambda x: x > 0)); return pd.DataFrame(X, index=d.index)
print("building features…", flush=True)
XF = pd.concat([stamped(F), EF.features(F)], axis=1); XM = pd.concat([stamped(M), EF.features(M)], axis=1)
cols = [c for c in XF.columns if c in XM.columns and XF[c].notna().mean() > 0.8 and XM[c].notna().mean() > 0.8 and XF[c].nunique() > 5]
XF, XM = XF[cols].astype(float), XM[cols].astype(float)
# binarisations: median (frozen on H1) always; sign (>0) where the variable has both signs
bins = []   # (name, boolF, boolM)
for c in cols:
    med = XF[c][F.h1.values].median(); bins.append((f"{c}>{med:.3g}", (XF[c] > med).values, (XM[c] > med).values))
    if (XF[c] > 0).mean() > 0.15 and (XF[c] <= 0).mean() > 0.15 and abs(med) > 1e-9: bins.append((f"{c}>0", (XF[c] > 0).values, (XM[c] > 0).values))
names = [b[0] for b in bins]; BF = np.array([b[1] for b in bins], dtype=np.float32).T; BM = np.array([b[2] for b in bins], dtype=np.float32).T   # n × k
NF = np.nan_to_num(BF, nan=0.0); NM = np.nan_to_num(BM, nan=0.0)
k = len(names); print(f"{len(cols)} variables → {k} binary legs → {k*(k-1)//2*4:,} quadrant tests", flush=True)
def quad_stats(Bn, y, mask):
    """for every (i,j) pair and quadrant (si,sj): count and mean pct. Returns dict of 4 (count, sum) matrices."""
    out = {}
    for si in (1, 0):
        for sj in (1, 0):
            Xi = Bn[mask] if si else (1 - Bn[mask]); Xj = Bn[mask] if sj else (1 - Bn[mask])
            cnt = Xi.T @ Xj; s = (Xi * y[mask][:, None]).T @ Xj; out[(si, sj)] = (cnt, s)
    return out
def survivors(yF, yM, dmin, nmin):
    h1, h2 = F.h1.values, ~F.h1.values; mM = np.ones(len(M), bool)
    Q1, Q2, QM = quad_stats(NF, yF, h1), quad_stats(NF, yF, h2), quad_stats(NM, yM, mM)
    tot1, tot2, totM = (yF[h1].sum(), h1.sum()), (yF[h2].sum(), h2.sum()), (yM.sum(), len(yM))
    res = []
    for q in Q1:
        c1, s1 = Q1[q]; c2, s2 = Q2[q]; cM, sM = QM[q]
        with np.errstate(divide="ignore", invalid="ignore"):
            m1 = s1 / c1; r1 = (tot1[0] - s1) / (tot1[1] - c1); m2 = s2 / c2; r2 = (tot2[0] - s2) / (tot2[1] - c2); mMq = sM / cM; rM = (totM[0] - sM) / (totM[1] - cM)
        d1, d2, dM = m1 - r1, m2 - r2, mMq - rM
        ok = (c1 >= nmin) & (c2 >= nmin) & (cM >= A.nm) & (d1 <= dmin) & (d2 <= dmin) & (dM <= A.dmm) & (m1 < 0) & (m2 < 0)
        ok = np.triu(ok, 1)
        for i, j in zip(*np.where(ok)):
            res.append(dict(legA=names[i] if q[0] else "NOT " + names[i], legB=names[j] if q[1] else "NOT " + names[j], nH1=c1[i, j] / S, avgH1=m1[i, j], dH1=d1[i, j], nH2=c2[i, j] / S, avgH2=m2[i, j], dH2=d2[i, j], nM=int(cM[i, j]), avgM=mMq[i, j], dM=dM[i, j], worst=max(d1[i, j], d2[i, j], dM[i, j])))
    return pd.DataFrame(res)
yF, yM = F.pct.values.astype(np.float32), M.pct.values.astype(np.float32)
R = survivors(yF, yM, A.dmin, A.nmin)
rng = np.random.default_rng(3); null = []
for _ in range(A.perm):
    yp = yF.copy()
    for d in np.unique(F.day.values):
        idx = np.where(F.day.values == d)[0]; yp[idx] = rng.permutation(yF[idx])
    ypm = rng.permutation(yM)
    null.append(len(survivors(yp, ypm, A.dmin, A.nmin)))
print(f"\nSURVIVORS (worse than rest by ≥{-A.dmin:.2f} in H1 AND H2, N≥{A.nmin}/half, master agrees): {len(R)} · shuffled-label null: {null} (median {np.median(null):.0f})")
if len(R):
    R["family"] = R.legA.str.replace("NOT ", "").str.split(">").str[0] + " × " + R.legB.str.replace("NOT ", "").str.split(">").str[0]
    R = R.sort_values("worst"); R.to_csv(f"reports/EXHAUSTIVE_2D_{A.sleeve}_d{-A.dmin:.2f}_n{A.nmin}_m{A.nm}.csv", index=False)
    pd.set_option("display.width", 260); D1 = R.drop_duplicates("family"); print(f"distinct families: {R.family.nunique()} (survivors {len(R)})"); print(D1.head(A.top)[["legA", "legB", "nH1", "avgH1", "dH1", "nH2", "avgH2", "dH2", "nM", "avgM", "dM"]].round(3).to_string(index=False))
    print("\nfamilies among survivors (count):"); print(R.family.value_counts().head(15).to_string())
