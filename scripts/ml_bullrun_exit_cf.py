#!/usr/bin/env python3
"""Momentum-long EXIT counterfactual: replace the momentum exit stack with the BULL-RUN exit stack (SL = min(−0.7, −1.5×ATR) floored
−1.2 · BE-arm at peak ≥ 1.0 net → lock +0.2 · trail = peak − 2.0×ATR · ladder rungs) on the REAL 1-minute path of every stack-kept
momentum LONG in the master pool. Prices with bullrun_exit_sweep.simulate (net-of-fee convention, conservative intrabar order:
stop tested before the peak updates → absolute levels are PESSIMISTIC for the BR stack). Two-sided by construction: every fill is
re-priced, winners and losers alike. Usage: venv/bin/python scripts/ml_bullrun_exit_cf.py"""
import os, sys, pickle
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bullrun_exit_sweep as BR

P = pd.read_csv(BR.POOL, low_memory=False)
L = P[(P.entry_strategy == "MOMENTUM") & (P.direction == "LONG") & P.stack_keep.astype(bool) & ~P.is_probe.astype(bool)].copy()
L["pnl_"] = pd.to_numeric(L.stack_pnl, errors="coerce").fillna(pd.to_numeric(L.pnl, errors="coerce"))
L["pct_"] = pd.to_numeric(L.pnl_percentage, errors="coerce")
cache = pickle.load(open(BR.CACHE, "rb")) if os.path.exists(BR.CACHE) else {}
rows = []
for _, r in L.iterrows():
    kl = BR.klines(r["pair"], r["opened_at"], cache)
    if not kl:
        continue
    ep, atr = float(r["entry_price"]), float(r["entry_atr_pct"] or 0.0)
    hi = [(h / ep - 1) * 100 for h, _ in kl]; lo = [(l / ep - 1) * 100 for _, l in kl]
    usd = float(r["investment"]) * float(r["leverage"]) / 100.0
    br_pct, br_peak = BR.simulate(hi, lo, atr)
    rows.append(dict(era=r["era"], pair=r["pair"], opened_at=r["opened_at"], actual_pct=r["pct_"], actual_usd=r["pnl_"], br_pct=br_pct, br_usd=br_pct * usd,
                     br_peak=br_peak, stopped=str(r["close_reason"]).startswith("STOP_LOSS"), reason=r["close_reason"], usd_per_pct=usd))
pickle.dump(cache, open(BR.CACHE, "wb"))
d = pd.DataFrame(rows); d["d_usd"] = d.br_usd - d.actual_usd; d["d_pct"] = d.br_pct - d.actual_pct
d.to_csv("reports/ML_BULLRUN_EXIT_CF.csv", index=False)
f = lambda x: f"{len(x):3d}·{(x > 0).mean() * 100:3.0f}%·{x.mean():+.3f}·${(x * 0).sum():+,.0f}"
print(f"momentum longs re-priced {len(d)}/{len(L)} (1m paths, 6 h horizon)\n")
print(f"{'':14s} {'ACTUAL (momentum stack)':>28s}   {'BULL-RUN exit stack':>26s}   Δ$")
def line(name, g):
    print(f"{name:14s} {len(g):3d}·{(g.actual_pct > 0).mean() * 100:3.0f}%·{g.actual_pct.mean():+.3f}·${g.actual_usd.sum():+7,.0f}   "
          f"{len(g):3d}·{(g.br_pct > 0).mean() * 100:3.0f}%·{g.br_pct.mean():+.3f}·${g.br_usd.sum():+7,.0f}   {g.d_usd.sum():+,.0f}")
for e, g in d.groupby("era", sort=False): line(e, g)
line("ALL", d)
print(f"\ntwo-sided: fills BR beats actual {int((d.d_pct > 0).sum())} · worse {int((d.d_pct < 0).sum())} · same {int((d.d_pct == 0).sum())}")
line("live-stopped", d[d.stopped]); line("not stopped", d[~d.stopped]); line("actual winners", d[d.actual_pct > 0]); line("actual losers", d[d.actual_pct <= 0])
print("\nby actual close reason (Δ$ = BR − actual):")
print(d.groupby(d.reason.str.split(" ").str[0]).agg(n=("d_usd", "size"), actual=("actual_usd", "sum"), br=("br_usd", "sum"), d=("d_usd", "sum")).round(0).sort_values("n", ascending=False).to_string())
print(f"\nBR stack on these fills: armed (peak ≥ 1.0 net) {int((d.br_peak >= 1.0).sum())} · stopped at base {int((d.br_pct <= -0.7).sum())} · biggest BR loss {d.br_pct.min():+.2f} · biggest BR win {d.br_pct.max():+.2f}")
print("\nlargest Δ per fill: " + ", ".join(f"{r.era} {r.pair.replace('USDT','')} {r.actual_pct:+.2f}→{r.br_pct:+.2f}" for _, r in d.reindex(d.d_pct.abs().sort_values(ascending=False).index).head(10).iterrows()))
