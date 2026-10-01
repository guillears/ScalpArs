#!/usr/bin/env python3
"""Momentum-long WINNERS vs LOSERS — deep dive on the full-year backtest (yr3) with the master alongside (2026-09-29).

Sections (→ markdown report):
 A. Population & economics — backtest vs master, per seed, per month; breakeven WR.
 B. Loser anatomy — how losers die (never green / green<0.2 / 0.2–0.4 / armed then stopped), hold time, exit reasons; winners'
    capture. Do entry variables predict the DEATH TYPE? (DOA vs armed-then-stopped)
 C. UI mirror — "Entry Conditions by Strategy — Winners vs Losers": mean of every stamped entry condition, W vs L, backtest and master.
 D. Dose-response — top variables in deciles (WR · avg) in H1 and H2: monotonic, U-shaped or noise. Sign / median / tercile
    granularity sweep for every variable, cross-half consistency (sleeve-kill checklist ①②).
 E. Context — BTC regime (stamped), BTC 24h/72h trend, BTC ATR tercile, breadth, hour (UTC), weekday, pair rank tier, pair
    age, sizing cell: N/seed · WR · avg · windows, H1 / H2, master alongside.
 F. Per-pair concentration — which pairs carry the loss; repeat losers; blacklist-vs-dimension test.
 G. Model — numpy logistic regression on rank-standardised features trained on H1 days, tested on H2 (day-grouped): OOS AUC,
    permutation importance; exhaustive depth-2 tree on the top features for interaction structure, judged OOS.
 H. Extras vs matched — what the backtest trades that live did not (attribution), and what the master's screen removes.
 I. Every finding → master per batch before→after.
Usage: venv/bin/python scripts/ml_deep_dive.py [--fills yr3_report_fills.csv] [--out reports/ML_WINNERS_LOSERS_DEEP_2026-09-29.md]
"""
import argparse, os, sys, itertools
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--fills", default="reports/backtest_cache/replay/year/yr3_report_fills.csv")
ap.add_argument("--split", default="2026-05-01"); ap.add_argument("--out", default="reports/ML_WINNERS_LOSERS_DEEP_2026-09-29.md")
A = ap.parse_args()
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG, entry_feature_factory as EF
M = LG.build(); sys.argv = _a
OUT = []
def P(*s): OUT.append(" ".join(str(x) for x in s)); print(*s, flush=True)

# ── data ──
M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"), M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
M = M[(M.direction == "LONG") & (M.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].reset_index(drop=True)
F = pd.read_csv(A.fills, low_memory=False)
F = F[F.cell_multiplier_source.astype(str) != "CROSS_OB_OPEN"]; _b = pd.to_numeric(F.entry_btc_atr_pct, errors="coerce")
F = F[~((F.cell_multiplier_source.astype(str) == "NONEXP_CALM3D") & (_b < 0.08))]
F = F[(F.direction == "LONG") & (F.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].reset_index(drop=True).copy()
for d in (F, M):
    d["pct"] = d.pnl_percentage if d is F else d.pct
    d["t"] = pd.to_datetime(d.opened_at.astype(str).str[:19]); d["day"] = d.t.dt.strftime("%Y-%m-%d"); d["mon"] = d.t.dt.strftime("%Y-%m")
    d["hour"] = d.t.dt.hour; d["wd"] = d.t.dt.dayofweek; d["win"] = d.pct > 0
    d["peak"] = pd.to_numeric(d.peak_pnl, errors="coerce"); d["hold"] = (pd.to_datetime(d.closed_at.astype(str).str[:19], errors="coerce") - d.t).dt.total_seconds() / 60
    d["exit"] = d.close_reason.astype(str).str.replace(r"\s*L\d+$", "", regex=True)
F["h"] = np.where(F.opened_at.astype(str) < A.split, "H1", "H2"); M["h"] = "M"
S = F.seed.nunique(); h1 = (F.h == "H1").values
ERAS = [e for e in LG.ERAS if e in set(M.era)]
n = lambda d, c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
fm = lambda g, per=1: (f"{len(g)/per:6.1f} · {g.win.mean()*100:3.0f}% · {g.pct.mean():+.3f}" if len(g) else "     — ")
def master_ba(mask):
    """master per batch before→after for a BLOCK mask (index-aligned bool Series on M)."""
    cells = []
    for e in ERAS:
        g = M[M.era == e]; k = g[~mask.reindex(g.index).fillna(False).astype(bool)]
        cells.append(f"{e} {len(g)}·{g.win.mean()*100:.0f}%·${g.net.sum():+,.0f}→{len(k)}·{k.win.mean()*100 if len(k) else 0:.0f}%·${k.net.sum():+,.0f}")
    b = M[mask.fillna(False).astype(bool)]
    return f"master blocked {len(b)}·{b.win.mean()*100 if len(b) else 0:.0f}%·{b.pct.mean() if len(b) else 0:+.3f}·${b.net.sum() if len(b) else 0:+,.0f} | " + " | ".join(cells)
rng = np.random.default_rng(7)
def day_ci(g, k=1000):
    ud = g.day.unique()
    if len(ud) < 3: return (np.nan, np.nan)
    bs = [g[g.day.isin(rng.choice(ud, len(ud)))].pct.mean() for _ in range(k)]; return tuple(np.percentile(bs, [2.5, 97.5]))

# ════ A ════
P("# Momentum-long winners vs losers — deep dive (backtest yr3, master alongside) — 2026-09-29\n")
P("## A. Population and economics")
W, L = F[F.win], F[~F.win]; be = abs(L.pct.mean()) / (W.pct.mean() + abs(L.pct.mean())) * 100
MW, ML_ = M[M.win], M[~M.win]; bem = abs(ML_.pct.mean()) / (MW.pct.mean() + abs(ML_.pct.mean())) * 100
P(f"- Backtest: {len(F)} fills ({len(F)/S:.0f}/seed) · WR {F.win.mean()*100:.0f}% · avg {F.pct.mean():+.3f} · avg win {W.pct.mean():+.3f} · avg loss {L.pct.mean():+.3f} · payoff {W.pct.mean()/abs(L.pct.mean()):.2f} · **breakeven WR {be:.0f}%** → the sleeve is {F.win.mean()*100-be:+.0f} pts from breakeven")
P(f"- Master:   {len(M)} fills · WR {M.win.mean()*100:.0f}% · avg {M.pct.mean():+.3f} · avg win {MW.pct.mean():+.3f} · avg loss {ML_.pct.mean():+.3f} · payoff {MW.pct.mean()/abs(ML_.pct.mean()):.2f} · breakeven {bem:.0f}%")
P("- Per seed: " + " | ".join(f"s{s} {fm(F[F.seed==s])}" for s in sorted(F.seed.unique())))
P("- Per month (bt N/seed·WR·avg | master): " + " | ".join(f"{m} {fm(g,S)} | {fm(M[M.mon==m])}" for m, g in F.groupby("mon")))
P(f"- H1 {fm(F[h1],S)} · H2 {fm(F[~h1],S)}\n")

# ════ B ════
P("## B. Loser anatomy — how trades die")
def anatomy(d, label):
    Ld = d[~d.win]; Wd = d[d.win]
    doa = (Ld.peak <= 0.05).mean()*100; g02 = ((Ld.peak > 0.05) & (Ld.peak < 0.2)).mean()*100; g24 = ((Ld.peak >= 0.2) & (Ld.peak < 0.4)).mean()*100; arm = (Ld.peak >= 0.4).mean()*100
    P(f"- {label}: losers {len(Ld)} → never green {doa:.0f}% · peaked 0.05–0.2 {g02:.0f}% · 0.2–0.4 {g24:.0f}% · armed (≥0.4) then lost {arm:.0f}% · median hold {Ld.hold.median():.0f} min (winners {Wd.hold.median():.0f}) · died ≤10 min {(Ld.hold<=10).mean()*100:.0f}% · exits {Ld.exit.value_counts().head(3).to_dict()}")
    P(f"  winners: median peak {Wd.peak.median():+.2f} · median realized {Wd.pct.median():+.2f} · captured {(Wd.pct/Wd.peak).median()*100:.0f}% of peak · exits {Wd.exit.value_counts().head(3).to_dict()}")
anatomy(F, "Backtest"); anatomy(M, "Master")
F["death"] = np.select([F.win, F.peak <= 0.05, F.peak < 0.4], ["WIN", "DOA", "GREEN_THEN_STOP"], "ARMED_THEN_LOST")
P("- Backtest death types by half: " + " | ".join(f"{h}: " + ", ".join(f"{k} {v/ (h1 if h=='H1' else ~h1).sum()*100:.0f}%" for k, v in F[F.h==h].death.value_counts().items()) for h in ("H1","H2")))

# features
P("\n(building features…)")
XF = EF.features(F); XM = EF.features(M)
STAMP = [c for c in F.columns if c.startswith("entry_") and pd.api.types.is_numeric_dtype(pd.to_numeric(F[c], errors="coerce")) and pd.to_numeric(F[c], errors="coerce").notna().mean() > 0.7 and F[c].nunique() > 5
         and c not in ("entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_mcap_usd", "entry_cmc_rank")]
for c in STAMP: XF[c] = n(F, c); XM[c] = n(M, c)
for d, X in ((F, XF), (M, XM)):
    X["d_rsi"] = n(d, "entry_rsi") - n(d, "entry_rsi_prev"); X["d_btc_rsi"] = n(d, "entry_btc_rsi") - n(d, "entry_btc_rsi_prev"); X["d_btc_adx"] = n(d, "entry_btc_adx") - n(d, "entry_btc_adx_prev")
    X["d_btc_rsi_30m"] = n(d, "entry_btc_rsi") - n(d, "entry_btc_rsi_prev6"); X["log_vol"] = np.log10(n(d, "entry_pair_volume_24h_usd").where(lambda x: x > 0))
cols = [c for c in XF.columns if c in XM.columns and XF[c].notna().mean() > 0.7 and XF[c].nunique() > 5]
XF, XM = XF[cols].astype(float), XM[cols].astype(float)
ISPAIR = lambda c: c.startswith(("PAIR_", "REL_")) or (c.startswith(("entry_", "d_", "log_")) and not any(k in c for k in ("btc", "bull", "bear", "global", "br_")))
def auc(x, y):
    ok = x.notna().values; x = x.values[ok]; yy = y[ok]
    if yy.sum() < 20 or (~yy).sum() < 20: return np.nan
    r = pd.Series(x).rank().values; return (r[yy].sum() - yy.sum()*(yy.sum()+1)/2) / (yy.sum()*(~yy).sum())
# death-type predictors
P("\n**Do entry variables predict the death type?** (AUC DOA vs green-then-stopped among backtest losers; top 8 each way)")
ld = F.death.isin(["DOA", "GREEN_THEN_STOP"]).values; ydoa = (F.death == "DOA").values[ld]
adoa = pd.Series({c: auc(XF[c][ld], ydoa) for c in cols}).dropna().sort_values()
P("- higher value → more DOA: " + ", ".join(f"{c} {v:.2f}" for c, v in adoa.tail(8)[::-1].items()))
P("- higher value → more green-then-stopped: " + ", ".join(f"{c} {v:.2f}" for c, v in adoa.head(8).items()))

# ════ C ════
P("\n## C. UI mirror — Entry Conditions: Winners vs Losers (means)")
UI = ["entry_rsi", "entry_rsi_prev", "entry_adx", "entry_adx_prev", "entry_gap", "entry_ema_gap_5_8", "entry_ema_gap_8_13", "entry_ema5_stretch", "entry_price_vs_ema5_pct", "entry_ema20_slope",
      "entry_range_position", "entry_atr_pct", "entry_pair_volume_ratio", "entry_global_volume_ratio", "entry_pair_ema20_ema50_gap_pct", "entry_dist_from_ema13_pct", "entry_pos_di", "entry_neg_di",
      "entry_quality_score", "entry_pair_rank", "entry_pair_age_days", "entry_btc_rsi", "entry_btc_rsi_prev", "entry_btc_adx", "entry_btc_adx_prev", "entry_btc_ema20_slope", "entry_btc_1h_slope",
      "entry_btc_rsi_1h", "entry_btc_trend_gap_pct", "entry_btc_dist_from_ema13_pct", "entry_btc_atr_pct", "entry_bull_pct", "entry_bear_pct"]
P("| condition | bt W mean | bt L mean | bt AUC | master W | master L | master AUC |"); P("|---|---|---|---|---|---|---|")
for c in UI:
    if c not in XF: continue
    P(f"| {c.replace('entry_','')} | {XF[c][F.win].mean():.3f} | {XF[c][~F.win].mean():.3f} | {auc(XF[c], F.win.values):.3f} | {XM[c][M.win].mean():.3f} | {XM[c][~M.win].mean():.3f} | {auc(XM[c], M.win.values):.3f} |")

# ════ D ════
P("\n## D. Dose-response and granularity sweep")
aAll = pd.Series({c: auc(XF[c], F.win.values) for c in cols}).dropna(); a1 = pd.Series({c: auc(XF[c][h1], F.win.values[h1]) for c in cols}); a2 = pd.Series({c: auc(XF[c][~h1], F.win.values[~h1]) for c in cols}); aM = pd.Series({c: auc(XM[c], M.win.values) for c in cols})
top = aAll.sub(.5).abs().sort_values(ascending=False).head(20).index
P("Deciles of the 20 strongest variables (all fills): WR per decile D1→D10 in H1 | H2 (monotone = Spearman |ρ| of WR vs decile ≥ 0.8 in BOTH halves)")
mono = []
for c in top:
    line = []; rhos = []
    for hm, lab in ((h1, "H1"), (~h1, "H2")):
        x = XF[c][hm]; y = F.win.values[hm]; q = pd.qcut(x.rank(method="first"), 10, labels=False)
        wr = pd.Series(y).groupby(q.values).mean() * 100; rho = np.corrcoef(pd.Series(wr.values).rank().values, np.arange(len(wr)))[0, 1]; rhos.append(rho)
        line.append(f"{lab} " + " ".join(f"{v:2.0f}" for v in wr.values) + f" (ρ {rho:+.2f})")
    tag = "MONOTONE" if all(abs(r) >= .8 for r in rhos) and np.sign(rhos[0]) == np.sign(rhos[1]) else ("U/hole" if any(abs(r) < .5 for r in rhos) else "weak")
    if tag == "MONOTONE": mono.append(c)
    P(f"- {c} [{'pair' if ISPAIR(c) else 'MACRO'}] AUC all {aAll[c]:.3f} · master {aM[c] if pd.notna(aM[c]) else float('nan'):.3f} → {tag}: " + " | ".join(line))
# granularity sweep
res = []
for c in cols:
    for gran in ("sign", "median", "tercile"):
        thr1 = 0 if gran == "sign" else (XF[c][h1].median() if gran == "median" else XF[c][h1].quantile(2/3))
        if gran == "sign" and not ((XF[c] > 0).any() and (XF[c] <= 0).any()): continue
        hi = (XF[c] > thr1).values; d = []
        for hm in (h1, ~h1):
            a_, b_ = F[hm & hi], F[hm & ~hi]
            if len(a_) < 30 or len(b_) < 30: d = None; break
            d.append(a_.pct.mean() - b_.pct.mean())
        if d and np.sign(d[0]) == np.sign(d[1]) and min(abs(d[0]), abs(d[1])) >= 0.05:
            mh = (XM[c] > thr1).values; dm = M[mh].pct.mean() - M[~mh].pct.mean() if mh.sum() >= 5 and (~mh).sum() >= 5 else np.nan
            res.append(dict(var=c, gran=gran, thr=thr1, dH1=d[0], dH2=d[1], dM=dm, agreeM=(np.sign(dm) == np.sign(d[0])) if pd.notna(dm) else None))
R = pd.DataFrame(res)
P(f"\nGranularity sweep: {len(cols)} vars × 3 splits → {len(R)} cross-half consistent splits (|Δavg| ≥ 0.05 in both halves); master agrees in direction on {int((R.agreeM == True).sum())} of them, disagrees on {int((R.agreeM == False).sum())}, no master read on {int(R.agreeM.isna().sum())}.")
if len(R):
    R["strength"] = np.minimum(R.dH1.abs(), R.dH2.abs()); Rm = R[R.agreeM == True].sort_values("strength", ascending=False)
    P("Top 15 where the master agrees (Δ = high-side minus low-side avg%; negative = high side loses):")
    P(Rm.head(15)[["var", "gran", "thr", "dH1", "dH2", "dM"]].round(3).to_string(index=False))

# ════ E ════
P("\n## E. Context — where the sleeve loses (N/seed · WR · avg, H1 | H2 | master)")
def ctx(name, keyF, keyM, order=None):
    P(f"\n**{name}**"); ks = order or sorted(set(keyF.dropna().unique()) | set(keyM.dropna().unique()), key=str)
    for k in ks:
        gf1 = F[(keyF == k).values & h1]; gf2 = F[(keyF == k).values & ~h1]; gm = M[(keyM == k).values]
        if len(gf1) + len(gf2) < 25 and len(gm) < 5: continue
        P(f"- {str(k):22s} H1 {fm(gf1,S)} ({gf1.day.nunique():3d}d) | H2 {fm(gf2,S)} ({gf2.day.nunique():3d}d) | master {fm(gm)} ({gm.day.nunique()}d)")
ctx("BTC regime at entry (stamped)", F.entry_btc_regime, M.entry_btc_regime)
def buck(s, edges, labels): return pd.cut(s, edges, labels=labels)
ctx("BTC 72h return (%)", buck(XF.get("BTC_1h_ret24", n(F,"entry_btc_r72_pct")), [-99,-3,0,3,6,99], ["<-3","-3..0","0..3","3..6",">6"]), buck(XM.get("BTC_1h_ret24", n(M,"entry_btc_r72_pct")), [-99,-3,0,3,6,99], ["<-3","-3..0","0..3","3..6",">6"]), ["<-3","-3..0","0..3","3..6",">6"])
ctx("BTC 24h return (%)", buck(XF["BTC_1h_ret24"], [-99,-2,0,2,4,99], ["<-2","-2..0","0..2","2..4",">4"]), buck(XM["BTC_1h_ret24"], [-99,-2,0,2,4,99], ["<-2","-2..0","0..2","2..4",">4"]), ["<-2","-2..0","0..2","2..4",">4"])
ctx("BTC 5m ATR% tercile (edges frozen on backtest)", buck(n(F,"entry_btc_atr_pct"), [0]+list(n(F,"entry_btc_atr_pct").quantile([1/3,2/3]))+[9], ["low","mid","high"]), buck(n(M,"entry_btc_atr_pct"), [0]+list(n(F,"entry_btc_atr_pct").quantile([1/3,2/3]))+[9], ["low","mid","high"]), ["low","mid","high"])
ctx("Bull breadth %", buck(n(F,"entry_bull_pct"), [-1,40,60,75,85,101], ["<40","40-60","60-75","75-85","≥85"]), buck(n(M,"entry_bull_pct"), [-1,40,60,75,85,101], ["<40","40-60","60-75","75-85","≥85"]), ["<40","40-60","60-75","75-85","≥85"])
ctx("Hour of day (UTC, 4h blocks)", (F.hour//4*4).astype(str)+"h", (M.hour//4*4).astype(str)+"h", [f"{h}h" for h in range(0,24,4)])
ctx("Weekday", F.wd.map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}), M.wd.map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
ctx("Pair volume rank tier", buck(n(F,"entry_pair_rank"), [0,10,20,30,40,60], ["1-10","11-20","21-30","31-40","41+"]), buck(n(M,"entry_pair_rank"), [0,10,20,30,40,60], ["1-10","11-20","21-30","31-40","41+"]), ["1-10","11-20","21-30","31-40","41+"])
ctx("Pair age (days)", buck(n(F,"entry_pair_age_days"), [0,90,180,365,730,9999], ["<90","90-180","180-365","1-2y",">2y"]), buck(n(M,"entry_pair_age_days"), [0,90,180,365,730,9999], ["<90","90-180","180-365","1-2y",">2y"]), ["<90","90-180","180-365","1-2y",">2y"])
ctx("Sizing cell / door", F.cell_multiplier_source.fillna("-"), M.cell_multiplier_source.fillna("-"))

# ════ F ════
P("\n## F. Per-pair concentration")
bp = F.groupby("pair").agg(n=("pct","size"), wr=("win","mean"), avg=("pct","mean"), tot=("pct","sum"), days=("day","nunique")).sort_values("tot")
tot_loss = bp[bp.tot<0].tot.sum()
P(f"- Backtest: {len(bp)} pairs; total loss carried by pairs with net<0: {tot_loss:.1f} pct-pts; top-5 loser pairs carry {bp.head(5).tot.sum()/tot_loss*100:.0f}% of it (concentration rule: ≥60% in 1–2 pairs → blacklist, not filter).")
P("  worst 12 (N all seeds · WR · avg · Σ · days): " + " | ".join(f"{p} {int(r.n)}·{r.wr*100:.0f}%·{r.avg:+.2f}·Σ{r.tot:+.1f}·{int(r.days)}d" for p, r in bp.head(12).iterrows()))
bpm = M.groupby("pair").agg(n=("pct","size"), wr=("win","mean"), tot=("net","sum")).sort_values("tot")
P("  master worst 8: " + " | ".join(f"{p} {int(r.n)}·{r.wr*100:.0f}%·${r.tot:+,.0f}" for p, r in bpm.head(8).iterrows()))
both = bp.join(bpm, lsuffix="_bt", rsuffix="_m", how="inner"); both = both[(both.n_bt >= 15) & (both.n_m >= 3)]
P(f"  pairs losing in BOTH (bt avg<0 & master net<0, N≥15/≥3): {', '.join(f'{p} (bt {r.avg:+.2f}, master ${r.tot_m:+,.0f})' for p, r in both[(both.avg<0)&(both.tot_m<0)].iterrows()) or 'none'}")

# ════ G ════
P("\n## G. Out-of-sample model (train H1 days → test H2)")
feat = list(aAll.sub(.5).abs().sort_values(ascending=False).head(40).index)
def prep(X, ref):
    Z = X[feat].copy()
    for c in feat:
        r = ref[c].dropna(); Z[c] = (Z[c].rank(pct=True) if ref is X else Z[c].map(lambda v: (r < v).mean() if pd.notna(v) else np.nan)); Z[c] = Z[c].fillna(0.5) - 0.5
    return Z.values
Ztr = prep(XF[h1], XF[h1]); Zte = prep(XF[~h1], XF[h1]); Zm = prep(XM, XF[h1]); ytr = F.win.values[h1].astype(float); yte = F.win.values[~h1]
def fit_lr(Z, y, l2=1.0, it=3000, lr=0.05):
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(it):
        p = 1/(1+np.exp(-(Z@w+b))); g = Z.T@(p-y)/len(y) + l2*w/len(y); w -= lr*g; b -= lr*(p-y).mean()
    return w, b
w, b0 = fit_lr(Ztr, ytr); pte = Zte@w+b0
def auc_np(s, y): r = pd.Series(s).rank().values; return (r[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum())
a_oos = auc_np(pte, yte); a_in = auc_np(Ztr@w+b0, F.win.values[h1])
P(f"- Logistic regression on the 40 strongest features (rank-standardised): in-sample AUC {a_in:.3f} → **out-of-sample (H2) AUC {a_oos:.3f}**; on the master {auc_np(Zm@w+b0, M.win.values):.3f}")
# OOS decile lift
q = pd.qcut(pd.Series(pte).rank(method="first"), 5, labels=False); Fh2 = F[~h1].reset_index(drop=True)
P("- H2 fills by model score quintile (worst→best): " + " | ".join(f"Q{k+1} {fm(Fh2[q.values==k],S)}" for k in range(5)))
imp = []
for j, c in enumerate(feat):
    Zp = Zte.copy(); Zp[:, j] = rng.permutation(Zp[:, j]); imp.append((c, a_oos - auc_np(Zp@w+b0, yte)))
imp = sorted(imp, key=lambda x: -x[1])
P("- Permutation importance (OOS AUC drop): " + ", ".join(f"{c} {v:+.3f}" for c, v in imp[:12]))
# depth-2 tree on top-12 features, judged OOS
P("\n**Interaction structure — best 2-variable rules (exhaustive depth-2 search on H1, judged on H2 and master):**")
tf = [c for c, _ in imp[:12]]; rules = []
for a, bb in itertools.combinations(tf, 2):
    for qa in (0.25, 0.5, 0.75):
        for qb in (0.25, 0.5, 0.75):
            ta, tb = XF[a][h1].quantile(qa), XF[bb][h1].quantile(qb)
            for sa in (1, -1):
                for sb in (1, -1):
                    z = (((XF[a] > ta) if sa > 0 else (XF[a] <= ta)) & ((XF[bb] > tb) if sb > 0 else (XF[bb] <= tb))).values
                    g1 = F[h1 & z]
                    if len(g1) < 60: continue
                    d1 = g1.pct.mean() - F[h1 & ~z].pct.mean()
                    if d1 > -0.10: continue
                    g2 = F[~h1 & z]
                    if len(g2) < 60: continue
                    d2 = g2.pct.mean() - F[~h1 & ~z].pct.mean()
                    zm = (((XM[a] > ta) if sa > 0 else (XM[a] <= ta)) & ((XM[bb] > tb) if sb > 0 else (XM[bb] <= tb))).values
                    gm = M[zm]; dm = gm.pct.mean() - M[~zm].pct.mean() if len(gm) >= 5 else np.nan
                    rules.append(dict(rule=f"{a}{'>' if sa>0 else '≤'}{ta:.3g} & {bb}{'>' if sb>0 else '≤'}{tb:.3g}", nH1=len(g1)/S, dH1=d1, avgH1=g1.pct.mean(), nH2=len(g2)/S, dH2=d2, avgH2=g2.pct.mean(), days=F[z].day.nunique(), nM=len(gm), wrM=gm.win.mean()*100 if len(gm) else np.nan, avgM=gm.pct.mean() if len(gm) else np.nan, dM=dm))
RR = pd.DataFrame(rules)
if len(RR):
    RR["score"] = RR[["dH1","dH2"]].max(axis=1); good = RR[(RR.dH2 < -0.05)].sort_values("score")
    P(f"{len(RR)} loser rules found on H1 (Δ ≤ −0.10); {len(good)} hold OOS (H2 Δ < −0.05); master agrees (Δ<0) on {int((good.dM<0).sum())}.")
    P(good.head(12)[["rule","nH1","avgH1","nH2","avgH2","days","nM","wrM","avgM","dM"]].round(3).to_string(index=False))
    for _, r in good[good.dM < 0].head(3).iterrows():
        a, rest = r.rule.split(" & ")[0], r.rule.split(" & ")[1]
        def mk(expr, X):
            v, op, thr = (expr.split(">")[0], ">", float(expr.split(">")[1])) if ">" in expr else (expr.split("≤")[0], "≤", float(expr.split("≤")[1]))
            return (X[v] > thr) if op == ">" else (X[v] <= thr)
        mask = mk(a, XM) & mk(rest, XM); gm = M[mask.values]; ci = day_ci(gm)
        P(f"   ▶ {r.rule}: {master_ba(mask)}\n     bar on master: N {len(gm)} · windows {gm.day.nunique()} · WR {gm.win.mean()*100:.0f}% vs breakeven {bem:.0f}% · day-CI [{ci[0]:+.3f},{ci[1]:+.3f}]")

# ════ H ════
P("\n## H. What the backtest trades that live did not (attribution, from the audit)")
try:
    D = pd.read_csv("reports/AUDIT_MOM-long_trades.csv"); ex = D[D.kind=="EXTRA"]; mi = D[D.kind=="MISSED"]
    P(f"- Extras (backtest-only inside live windows): {len(ex)} · WR {(ex.bt_pct>0).mean()*100:.0f}% · avg {ex.bt_pct.mean():+.3f} — by door: " + ", ".join(f"{k} {len(g)}·{g.bt_pct.mean():+.2f}" for k, g in ex.groupby(ex.bt_cell.fillna('-'))))
    P(f"- Missed (live-only): {len(mi)} · avg {mi.live_pct.mean():+.3f}; nearest-scan gate top: {mi.bt_gate_near.astype(str).str.split(',').str[0].value_counts().head(5).to_dict()}")
except Exception as e: P(f"- (audit file not available: {e})")

# ════ I ════
P("\n## I. Findings judged on the master (before→after per batch)")
for c in mono[:6]:
    lo = aAll[c] > .5; cut = XF[c][h1].quantile(0.2 if lo else 0.8); mask = (XM[c] < cut) if lo else (XM[c] > cut)
    bf = ((XF[c] < cut) if lo else (XF[c] > cut)).values
    P(f"- MONOTONE {c} block {'<' if lo else '>'} {cut:.4g}: backtest H1 blocked {fm(F[h1&bf],S)} · H2 blocked {fm(F[~h1&bf],S)} (kept H2 {fm(F[~h1&~bf],S)}) | {master_ba(mask)}")
open(A.out, "w").write("\n".join(OUT)); print(f"\n→ {A.out}")
