#!/usr/bin/env python3
"""EXHAUSTIVE pairwise scan, MASTER-FIRST (operator 2026-09-29): every variable × every variable, both binarisations (median
on the master; sign at 0), all 4 quadrants, on the master+B14 momentum longs (post-gate). Survivor on the master = quadrant with
N ≥ --nmin trades, WR ≤ --wrmax, avg < 0, Δ vs rest ≤ --dmin. Shuffled-label null (within day) → how many survivors luck makes.
Then EVERY master survivor is re-tested with the SAME legs on the backtest (yr3, post-gate) in H1 and H2 → confirmed = Δ<0 and
avg<0 in both halves with N ≥ 40 fills per half. Output reports/EXHAUSTIVE_2D_MASTER_MOM-long.csv"""
import argparse, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--b14", default="/Users/guillearslanian/Downloads/scalpars_orders_paper_2026-09-29_14-06-29.csv")
ap.add_argument("--fills", default="reports/backtest_cache/replay/year/yr3_report_fills.csv")
ap.add_argument("--nmin", type=int, default=8); ap.add_argument("--wrmax", type=float, default=50.0); ap.add_argument("--dmin", type=float, default=-0.25); ap.add_argument("--perm", type=int, default=100)
A = ap.parse_args()
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG, entry_feature_factory as EF
M = LG.build(); sys.argv = _a
n = lambda d, c: pd.to_numeric(d[c], errors="coerce")
M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"), M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
M = M[(M.direction == "LONG") & (M.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")]
B = pd.read_csv(A.b14, low_memory=False); b = B[(B.status == "CLOSED") & (B.direction == "LONG") & (B.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].copy()
b["pct"] = b.pnl_percentage; b["net"] = b.pnl; b["era"] = "B14"; b = b[~((n(b, "entry_adx") < 21) & (n(b, "entry_rsi") < n(b, "entry_rsi_prev")))]
C = pd.concat([M, b], ignore_index=True); C["day"] = C.opened_at.astype(str).str[:10]
F = pd.read_csv(A.fills, low_memory=False)
F = F[F.cell_multiplier_source.astype(str) != "CROSS_OB_OPEN"]; _x = pd.to_numeric(F.entry_btc_atr_pct, errors="coerce"); F = F[~((F.cell_multiplier_source.astype(str) == "NONEXP_CALM3D") & (_x < 0.08))]
F = F[(F.direction == "LONG") & (F.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")]; F = F[~((n(F, "entry_adx") < 21) & (n(F, "entry_rsi") < n(F, "entry_rsi_prev")))].reset_index(drop=True).copy()
F["pct"] = F.pnl_percentage; F["h1"] = (F.opened_at.astype(str) < "2026-05-01").values; S = F.seed.nunique()
def stamped(d):
    X = {c: n(d, c) for c in d.columns if c.startswith("entry_") and pd.api.types.is_numeric_dtype(n(d, c)) and n(d, c).notna().mean() > 0.7 and d[c].nunique() > 5
         and c not in ("entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_mcap_usd", "entry_cmc_rank")}
    X["d_rsi"] = n(d, "entry_rsi") - n(d, "entry_rsi_prev"); X["d_adx"] = n(d, "entry_adx") - n(d, "entry_adx_prev"); X["d_btc_rsi"] = n(d, "entry_btc_rsi") - n(d, "entry_btc_rsi_prev")
    X["d_btc_adx"] = n(d, "entry_btc_adx") - n(d, "entry_btc_adx_prev"); X["d_btc_rsi_30m"] = n(d, "entry_btc_rsi") - n(d, "entry_btc_rsi_prev6"); X["di_spread"] = n(d, "entry_pos_di") - n(d, "entry_neg_di")
    X["log_vol"] = np.log10(n(d, "entry_pair_volume_24h_usd").where(lambda x: x > 0)); return pd.DataFrame(X, index=d.index)
print(f"master+B14 post-gate: {len(C)} longs, {int((C.pct<=0).sum())} losers · building features…", flush=True)
XC = pd.concat([stamped(C), EF.features(C)], axis=1); XF = pd.concat([stamped(F), EF.features(F)], axis=1)
cols = [c for c in XC.columns if c in XF.columns and XC[c].notna().mean() > 0.8 and XF[c].notna().mean() > 0.8 and XC[c].nunique() > 5]
XC, XF = XC[cols].astype(float), XF[cols].astype(float)
bins = []
for c in cols:
    med = XC[c].median(); bins.append((f"{c}>{med:.3g}", (XC[c] > med).values, (XF[c] > med).values))
    if (XC[c] > 0).mean() > 0.15 and (XC[c] <= 0).mean() > 0.15 and abs(med) > 1e-9: bins.append((f"{c}>0", (XC[c] > 0).values, (XF[c] > 0).values))
names = [x[0] for x in bins]; BC = np.nan_to_num(np.array([x[1] for x in bins], dtype=np.float32).T); BF = np.nan_to_num(np.array([x[2] for x in bins], dtype=np.float32).T)
k = len(names); print(f"{len(cols)} variables → {k} legs → {k*(k-1)//2*4:,} quadrant tests on the master", flush=True)
def scan(Bn, y, mask, nmin, wrmax, dmin):
    tot_s, tot_c, tot_w = y[mask].sum(), mask.sum(), (y[mask] > 0).sum(); res = {}
    for si in (1, 0):
        for sj in (1, 0):
            Xi = Bn[mask] if si else 1 - Bn[mask]; Xj = Bn[mask] if sj else 1 - Bn[mask]
            cnt = Xi.T @ Xj; s = (Xi * y[mask][:, None]).T @ Xj; w = (Xi * (y[mask] > 0)[:, None]).T @ Xj
            with np.errstate(divide="ignore", invalid="ignore"):
                m = s / cnt; r = (tot_s - s) / (tot_c - cnt); wr = w / cnt * 100
            ok = np.triu((cnt >= nmin) & (wr <= wrmax) & (m < 0) & ((m - r) <= dmin), 1)
            res[(si, sj)] = (ok, cnt, m, wr, m - r)
    return res
yC = C.pct.values.astype(np.float32); allC = np.ones(len(C), bool)
R = scan(BC, yC, allC, A.nmin, A.wrmax, A.dmin); nsurv = sum(int(v[0].sum()) for v in R.values())
rng = np.random.default_rng(1); null = []
for _ in range(A.perm):
    yp = yC.copy()
    for d in np.unique(C.day.values):
        idx = np.where(C.day.values == d)[0]; yp[idx] = rng.permutation(yC[idx])
    null.append(sum(int(v[0].sum()) for v in scan(BC, yp, allC, A.nmin, A.wrmax, A.dmin).values()))
print(f"\nMASTER survivors (N≥{A.nmin}, WR≤{A.wrmax:.0f}%, avg<0, Δ≤{A.dmin}): {nsurv:,} · shuffled-label null over {A.perm} runs: median {np.median(null):,.0f}, 95th {np.percentile(null,95):,.0f}, max {max(null):,}")
# re-test every master survivor on the backtest halves
yF = F.pct.values.astype(np.float32); h1, h2 = F.h1.values, ~F.h1.values; rows = []
for (si, sj), (ok, cnt, m, wr, d) in R.items():
    I, J = np.where(ok)
    for i, j in zip(I, J):
        li = BC[:, i] if si else 1 - BC[:, i]; lj = BC[:, j] if sj else 1 - BC[:, j]
        fi = BF[:, i] if si else 1 - BF[:, i]; fj = BF[:, j] if sj else 1 - BF[:, j]; z = (fi * fj) > 0
        r = dict(legA=(names[i] if si else "NOT " + names[i]), legB=(names[j] if sj else "NOT " + names[j]), nM=int(cnt[i, j]), wrM=wr[i, j], avgM=m[i, j], dM=d[i, j])
        for lab, hm in (("H1", h1), ("H2", h2)):
            zz = z & hm; r[f"n{lab}"] = zz.sum() / S; r[f"avg{lab}"] = yF[zz].mean() if zz.sum() else np.nan; r[f"d{lab}"] = (yF[zz].mean() - yF[hm & ~z].mean()) if zz.sum() else np.nan
        rows.append(r)
T = pd.DataFrame(rows)
if len(T):
    T["confirmed"] = (T.nH1 * S >= 40) & (T.nH2 * S >= 40) & (T.dH1 < 0) & (T.dH2 < 0) & (T.avgH1 < 0) & (T.avgH2 < 0)
    T["family"] = T.legA.str.replace("NOT ", "").str.split(">").str[0] + " × " + T.legB.str.replace("NOT ", "").str.split(">").str[0]
    T.to_csv("reports/EXHAUSTIVE_2D_MASTER_MOM-long.csv", index=False)
    conf = T[T.confirmed].copy(); conf["worst"] = conf[["dH1", "dH2"]].max(axis=1)
    print(f"master survivors re-tested on the backtest: confirmed in BOTH halves (Δ<0, avg<0, ≥40 fills/half): {len(conf):,} of {len(T):,} ({len(conf)/max(len(T),1)*100:.1f}%)")
    # what share of a null master survivor set would be 'confirmed' by chance? (backtest halves are independent of the master labels)
    print(f"   expected chance confirmation rate ≈ P(ΔH1<0 & avgH1<0) × P(ΔH2<0 & avgH2<0) over all quadrants — computed on the master survivors' own backtest reads: {((T.dH1<0)&(T.avgH1<0)).mean()*((T.dH2<0)&(T.avgH2<0)).mean()*100:.1f}% if independent")
    pd.set_option("display.width", 260)
    print("\nTop confirmed (sorted by the weaker backtest half), one line per family:")
    print(conf.sort_values("worst").drop_duplicates("family").head(25)[["legA", "legB", "nM", "wrM", "avgM", "nH1", "avgH1", "dH1", "nH2", "avgH2", "dH2"]].round(3).to_string(index=False))
    print("\nconfirmed families (count):"); print(conf.family.value_counts().head(12).to_string())
