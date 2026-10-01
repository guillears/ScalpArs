#!/usr/bin/env python3
"""Backtest momentum longs — what separates winners from losers? (operator 2026-09-29)
Universe = every numeric entry_* stamp + the ~800 rebuilt features of entry_feature_factory (BTC/ETH/BTCDOM/PAIR × 5 TFs) +
pair-vs-BTC relatives. Per variable: AUC(win) on all fills, in H1 (Jan–Apr) and H2 (May–Sep) separately, and a shuffled-label
null (how many variables pass by luck). Candidate = same direction in H1 and H2 with |AUC−.5| ≥ --gap in both. Each candidate:
loser-side cut frozen on H1 → H2 out-of-sample, blocked cohort at the expectancy bar (WR vs breakeven, day-clustered 95% CI,
windows, concentration), backtest per month before→after, and the MASTER per batch before→after.
Then 2D: PAIR × MACRO median quadrants (medians frozen on H1), loser zones consistent in H1, H2 and the master.
"""
import argparse, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--fills", default="reports/backtest_cache/replay/year/yr3_report_fills.csv")
ap.add_argument("--split", default="2026-05-01"); ap.add_argument("--gap", type=float, default=0.06)
ap.add_argument("--perm", type=int, default=200); ap.add_argument("--top", type=int, default=12); ap.add_argument("--min-block", type=int, default=60)
A = ap.parse_args()
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG, entry_feature_factory as EF
M = LG.build(); sys.argv = _a
M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"), M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
M = M[(M.direction == "LONG") & (M.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].reset_index(drop=True)
F = pd.read_csv(A.fills, low_memory=False)
F = F[F.cell_multiplier_source.astype(str) != "CROSS_OB_OPEN"]; b = pd.to_numeric(F.entry_btc_atr_pct, errors="coerce")
F = F[~((F.cell_multiplier_source.astype(str) == "NONEXP_CALM3D") & (b < 0.08))]
F = F[(F.direction == "LONG") & (F.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].reset_index(drop=True).copy()
F["pct"] = F.pnl_percentage; F["day"] = F.opened_at.astype(str).str[:10]; F["mon"] = F.opened_at.astype(str).str[:7]; F["h"] = np.where(F.opened_at.astype(str) < A.split, "H1", "H2")
S = F.seed.nunique()
SKIP = ("entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_mcap_usd", "entry_cmc_rank")
def stamped(d):
    n = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
    X = {("STAMP_" + c): n(c) for c in d.columns if c.startswith("entry_") and c not in SKIP}
    X["STAMP_d_rsi"] = n("entry_rsi") - n("entry_rsi_prev"); X["STAMP_d_adx"] = n("entry_adx") - n("entry_adx_prev")
    X["STAMP_d_btc_rsi"] = n("entry_btc_rsi") - n("entry_btc_rsi_prev"); X["STAMP_d_btc_rsi_30m"] = n("entry_btc_rsi") - n("entry_btc_rsi_prev6")
    X["STAMP_d_btc_adx"] = n("entry_btc_adx") - n("entry_btc_adx_prev"); X["STAMP_d_btc_rsi_1h"] = n("entry_btc_rsi_1h") - n("entry_btc_rsi_1h_prev")
    X["STAMP_log_pair_vol_usd"] = np.log10(n("entry_pair_volume_24h_usd").where(lambda x: x > 0))
    return pd.DataFrame(X, index=d.index)
print(f"building features: backtest {len(F)} ML fills ({S} seeds), master {len(M)} …", flush=True)
XF = pd.concat([stamped(F), EF.features(F)], axis=1); XM = pd.concat([stamped(M), EF.features(M)], axis=1)
cols = [c for c in XF.columns if c in XM.columns and XF[c].notna().mean() > 0.7 and XF[c].nunique() > 5 and pd.api.types.is_numeric_dtype(XF[c])]
XF, XM = XF[cols].astype(float), XM[cols].astype(float)
PAIRLVL = lambda c: c.startswith("PAIR_") or c.startswith("REL_") or (c.startswith("STAMP_") and not any(k in c for k in ("btc", "bull", "bear", "global", "br_")))
def auc_cols(X, y):
    R = X.rank(axis=0).values; ok = X.notna().values; yy = np.broadcast_to(y[:, None], ok.shape)
    n1 = (ok & yy).sum(0); n0 = (ok & ~yy).sum(0); s1 = np.where(ok & yy, R, 0.0).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"): a = (s1 - n1 * (n1 + 1) / 2) / (n1 * n0)
    return pd.Series(a, index=X.columns).where((n1 >= 20) & (n0 >= 20))
y = (F.pct > 0).values; h1 = (F.h == "H1").values
aA, a1, a2 = auc_cols(XF, y), auc_cols(XF[h1], y[h1]), auc_cols(XF[~h1], y[~h1]); aM = auc_cols(XM, (M.pct > 0).values)
def passing(x1, x2): return (np.sign(x1 - .5) == np.sign(x2 - .5)) & ((x1 - .5).abs() >= A.gap) & ((x2 - .5).abs() >= A.gap)
ok = passing(a1, a2)
rng = np.random.default_rng(11)
# null: shuffle labels WITHIN day (keeps market-window structure), recount
days = F.day.values; null = []
for _ in range(A.perm):
    yp = y.copy()
    for d in np.unique(days):
        idx = np.where(days == d)[0]; yp[idx] = rng.permutation(y[idx])
    null.append(int(passing(auc_cols(XF[h1], yp[h1]), auc_cols(XF[~h1], yp[~h1])).sum()))
W, L = F[F.pct > 0], F[F.pct <= 0]; be = abs(L.pct.mean()) / (W.pct.mean() + abs(L.pct.mean())) * 100
print(f"\n## BACKTEST MOM-long: {len(F)} fills ({len(F)/S:.0f}/seed) · WR {y.mean()*100:.0f}% · avg {F.pct.mean():+.3f} · avg win {W.pct.mean():+.3f} / avg loss {L.pct.mean():+.3f} → breakeven WR {be:.0f}%")
print(f"   variables tested {int(aA.notna().sum())} · consistent H1+H2 separators {int(ok.sum())} · day-shuffled null: median {np.median(null):.0f}, 95th {np.percentile(null, 95):.0f} → {'MORE than luck' if ok.sum() > np.percentile(null, 95) else 'NOT more than luck'}")
T = pd.DataFrame(dict(auc_all=aA, auc_H1=a1, auc_H2=a2, auc_master=aM, ok=ok)); T["level"] = ["PAIR" if PAIRLVL(c) else "MACRO" for c in T.index]
T["W_med"] = XF[y].median(); T["L_med"] = XF[~y].median(); T["strength"] = np.minimum((T.auc_H1 - .5).abs(), (T.auc_H2 - .5).abs())
T.to_csv("reports/BACKTEST_ML_SEPARATORS.csv")
print("\n### UI-style table — Winners vs Losers medians (backtest ML), strongest 25 by |AUC−.5| (all fills):")
show = T.assign(g=(T.auc_all - .5).abs()).sort_values("g", ascending=False).head(25)
print(show[["level", "W_med", "L_med", "auc_all", "auc_H1", "auc_H2", "auc_master", "ok"]].round(3).to_string())
print("\n### by level: consistent separators")
print(T[T.ok].groupby("level").size().to_dict())
C = T[T.ok].sort_values("strength", ascending=False).head(A.top)
ERAS = [e for e in LG.ERAS if e in set(M.era)]
def boot_ci(g):
    ud = g.day.unique(); bs = [g[g.day.isin(rng.choice(ud, len(ud)))].pct.mean() for _ in range(1000)]; return np.percentile(bs, [2.5, 97.5])
fm = lambda g, per=1: f"{len(g)/per:5.1f}·{(g.pct>0).mean()*100 if len(g) else 0:3.0f}%·{g.pct.mean() if len(g) else 0:+.3f}"
for v, r in C.iterrows():
    lo = r.auc_H1 > .5
    best = None; hv = XF[v][h1].dropna()
    for q in np.linspace(.1, .4, 7):
        cut = hv.quantile(q if lo else 1 - q); blk = ((XF[v] < cut) if lo else (XF[v] > cut)).values
        bb = F[h1 & blk]
        if len(bb) >= A.min_block and (best is None or bb.pct.mean() < best[1]): best = (cut, bb.pct.mean())
    if best is None: continue
    cut = best[0]; bf = ((XF[v] < cut) if lo else (XF[v] > cut)).values; bm = ((XM[v] < cut) if lo else (XM[v] > cut)).values
    B2 = F[~h1 & bf]; ci = boot_ci(B2) if len(B2) >= 10 else (np.nan, np.nan)
    conc = B2.groupby("day").pct.sum(); conc = (conc.min() / conc[conc < 0].sum()) if (conc < 0).any() else 0
    print(f"\n▶ [{r.level}] BLOCK {v} {'<' if lo else '>'} {cut:.4g}   (AUC all {r.auc_all:.3f} · H1 {r.auc_H1:.3f} · H2 {r.auc_H2:.3f} · master {r.auc_master if pd.notna(r.auc_master) else float('nan'):.3f})")
    print(f"   backtest H1 (fit): blocked {fm(F[h1&bf],S)} kept {fm(F[h1&~bf],S)} | H2 (OOS): blocked {fm(B2,S)} kept {fm(F[~h1&~bf],S)} · H2 blocked WR vs breakeven {be:.0f}% · day-CI [{ci[0]:+.3f},{ci[1]:+.3f}] · windows {B2.day.nunique()} · worst-day share {conc:.0%}")
    print("   backtest per month before→after: " + " | ".join(f"{m} {g.pct.mean():+.3f}→{g[~bf[g.index]].pct.mean() if (~bf[g.index]).any() else 0:+.3f}" for m, g in F.groupby('mon')))
    mb = M[bm]
    print(f"   MASTER blocked {len(mb)}·{(mb.pct>0).mean()*100 if len(mb) else 0:.0f}%·{mb.pct.mean() if len(mb) else 0:+.3f}·${mb.net.sum() if len(mb) else 0:+,.0f} | per batch before→after: " + " | ".join(f"{e} {len(g)}·{(g.pct>0).mean()*100:.0f}%·${g.net.sum():+,.0f}→{len(g[~bm[g.index]])}·{(g[~bm[g.index]].pct>0).mean()*100 if (~bm[g.index]).any() else 0:.0f}%·${g[~bm[g.index]].net.sum():+,.0f}" for e in ERAS for g in [M[M.era==e]] if len(g)))
# ── 2D PAIR × MACRO ──
print("\n## 2D PAIR × MACRO median quadrants (medians frozen on H1); loser zone = worse than the rest in H1, H2 AND master")
pv = [c for c in T[(T.level == "PAIR")].assign(g=(T.auc_all - .5).abs()).sort_values("g", ascending=False).index[:25]]
mv = [c for c in T[(T.level == "MACRO")].assign(g=(T.auc_all - .5).abs()).sort_values("g", ascending=False).index[:25]]
med = {c: XF[c][h1].median() for c in pv + mv}; out = []
for a in pv:
    for bb_ in mv:
        for sa in (True, False):
            for sb in (True, False):
                fz = (((XF[a] > med[a]) == sa) & ((XF[bb_] > med[bb_]) == sb) & XF[a].notna() & XF[bb_].notna()).values
                mz = (((XM[a] > med[a]) == sa) & ((XM[bb_] > med[bb_]) == sb) & XM[a].notna() & XM[bb_].notna()).values
                z1, z2, zm = F[h1 & fz], F[~h1 & fz], M[mz]
                if len(z1) < 60 or len(z2) < 60 or len(zm) < 6: continue
                d1 = z1.pct.mean() - F[h1 & ~fz].pct.mean(); d2 = z2.pct.mean() - F[~h1 & ~fz].pct.mean(); dm = zm.pct.mean() - M[~mz].pct.mean()
                if d1 < 0 and d2 < 0 and dm < 0 and z1.pct.mean() < 0 and z2.pct.mean() < 0:
                    out.append(dict(zone=f"{a}{'>' if sa else '≤'}{med[a]:.3g} & {bb_}{'>' if sb else '≤'}{med[bb_]:.3g}", nH1=len(z1)/S, avgH1=z1.pct.mean(), nH2=len(z2)/S, avgH2=z2.pct.mean(), days=F[fz].day.nunique(), nM=len(zm), wrM=(zm.pct>0).mean()*100, avgM=zm.pct.mean(), netM=zm.net.sum(), worst=max(d1, d2, dm)))
Z = pd.DataFrame(out)
print(f"   tested {len(pv)*len(mv)*4} quadrants → consistent loser zones {len(Z)} (≈{len(pv)*len(mv)*4/8:.0f} expected by luck at 1/8)")
if len(Z): print(Z.sort_values("worst").head(15).round(3).to_string(index=False))
