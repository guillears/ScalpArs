#!/usr/bin/env python3
"""YR6 WL — FRENZY question (2): what differs between WILLY 2025 (−0.2 %/fill, CI excludes 0) and WILLY 2026 (+0.10 / +0.07)?
Also the matching LITE decomposition (2025 −0.14 vs 2026 +0.23 / +0.14) and the LITE BTC-intraday gate by month.

  venv/bin/python -I scripts/study_yr6wl_frenzy_willy.py   → reports/YR6_WL_FRENZY_willy.md

Reads: ① exit mix (TP / stop / cap) × regime × top-50 and a ruler-overshoot sensitivity (stops clipped to −4.02, TPs to +1.27);
② composition: 2025 re-weighted to 2026's top-50 mix (and to 2026's mix of top-50 × BTC-1h-slope sign) → how much of the 2025 − 2026 gap
is mix vs within-cell; ③ per-pair and per-day concentration of the 2025 loss; ④ distribution shift (median per regime, KS 2025 vs 2026)
of the pair / macro stamps; ⑤ month table with the BTC-1h-slope split (LITE + WILLY); ⑥ master real fills (MASTER_POOL_stacked.csv,
stack_keep, stack_pct) for the LITE gate.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_yr6wl_frenzy_common as C  # noqa: E402


def ks_stat(a, b):
    """two-sample Kolmogorov–Smirnov D (no scipy in the venv)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, side="right") / len(a) - np.searchsorted(b, v, side="right") / len(b))))

D = C.load()
out = []
for s in ("FRENZY_WILLY", "FRENZY_LITE"):
    S = D[D.sleeve == s].copy()
    S["y_clip"] = np.where(S.exit_kind == "stop", np.maximum(S.y, -4.02), np.where(S.exit_kind == "TP", np.minimum(S.y, (1.27 if s == "FRENZY_WILLY" else 3.02)), S.y))
    out.append(f"\n## {s}\n\n### ① Exit mix × regime × top-50 (seed 1)\n")
    rows = []
    for (r, t), g in S.groupby(["regime", "top50"]):
        ek = g.exit_kind.value_counts(normalize=True)
        rows.append(dict(regime=r, top50=t, N=len(g), WR=100 * g.win.mean(), avg=g.y.mean(), avg_clipped=g.y_clip.mean(),
                         TP_share=100 * ek.get("TP", 0), stop_share=100 * ek.get("stop", 0), cap_share=100 * ek.get("cap", 0),
                         stop_avg=g.y[g.exit_kind == "stop"].mean(), TP_avg=g.y[g.exit_kind == "TP"].mean(), cap_avg=g.y[g.exit_kind == "cap"].mean(),
                         avg_win=g.y[g.y > 0].mean(), avg_loss=g.y[g.y <= 0].mean()))
    out.append(C.md_table(pd.DataFrame(rows), 3))
    # ② composition
    S["slope_pos"] = pd.to_numeric(S.entry_btc_1h_slope, errors="coerce") > 0
    a25, a26 = S[S.regime == "2025 BULL"], S[S.regime != "2025 BULL"]
    out.append("\n### ② Composition: 2025 avg re-weighted to 2026's cell mix (cells = top-50 × BTC-1h-slope sign)\n")
    rr = []
    for cells in (["top50"], ["top50", "slope_pos"], ["slope_pos"]):
        m25 = a25.groupby(cells).y.mean(); w26 = a26.groupby(cells).size() / len(a26); w25 = a25.groupby(cells).size() / len(a25)
        rw = float((m25 * w26).sum())
        rr.append(dict(cells=" × ".join(cells), avg_2025=a25.y.mean(), avg_2025_reweighted_to_2026_mix=rw, avg_2026=a26.y.mean(),
                       share_gap_explained_by_mix=(a25.y.mean() - rw) / (a25.y.mean() - a26.y.mean()),
                       cell_means_2025=" · ".join(f"{k}:{v:+.2f}({w25[k] * 100:.0f}%)" for k, v in m25.items()),
                       cell_means_2026=" · ".join(f"{k}:{v:+.2f}({w26[k] * 100:.0f}%)" for k, v in a26.groupby(cells).y.mean().items())))
    out.append(C.md_table(pd.DataFrame(rr), 3))
    # ③ concentration
    out.append("\n### ③ 2025 loss concentration\n")
    loss = -np.minimum(a25.y, 0)
    pc = loss.groupby(a25.pair).sum().sort_values(ascending=False)
    net_p = a25.groupby("pair").y.agg(["size", "sum", "mean"]).sort_values("sum").head(8)
    dc = a25.groupby("day").y.sum().sort_values().head(8)
    out.append(f"2025 fills {len(a25)} on {a25.pair.nunique()} pairs, {a25.day.nunique()} days. Sum y {a25.y.sum():+.1f} %-pts. "
               f"Top-2 pairs carry {100 * pc.head(2).sum() / loss.sum():.1f} % of the gross loss; top-10 pairs {100 * pc.head(10).sum() / loss.sum():.1f} %. "
               f"Worst 8 days sum {dc.sum():+.1f} %-pts ({100 * dc.sum() / a25.y.sum():.0f} % of the net). "
               f"Without the worst 8 days: avg {a25[~a25.day.isin(dc.index)].y.mean():+.3f}; without the worst 10 pairs (by net): "
               f"{a25[~a25.pair.isin(a25.groupby('pair').y.sum().sort_values().head(10).index)].y.mean():+.3f}.\n")
    out.append("Worst pairs (net %-pts): " + " · ".join(f"{p} {r['sum']:+.1f} ({int(r['size'])})" for p, r in net_p.iterrows()))
    out.append("\nWorst days (net %-pts): " + " · ".join(f"{d} {v:+.1f}" for d, v in dc.items()) + "\n")
    # ④ distribution shift
    out.append("\n### ④ Stamp distribution shift — median per regime; KS 2025 vs 2026 (top 12 by KS)\n")
    rows = []
    for c in C.entry_cols(S):
        x = C.num(S, c)
        if x.nunique() < 5 or c in ("entry_frenzy_vwap", "entry_desired_notional", "entry_liquidity_cap_notional"):
            continue
        x25, x26 = x[S.regime == "2025 BULL"].dropna(), x[S.regime != "2025 BULL"].dropna()
        if len(x25) < 30 or len(x26) < 30:
            continue
        ks = ks_stat(x25, x26)
        rows.append(dict(col=c, kind="MACRO" if C.is_macro(c) else "pair", med_25=x25.median(), med_26H1=x[S.regime == "2026-H1 BEAR"].median(),
                         med_26H2=x[S.regime == "2026-H2 BULL"].median(), KS=ks))
    KS = pd.DataFrame(rows).sort_values("KS", ascending=False).head(12)
    out.append(C.md_table(KS, 3))
    # ⑤ month table with the slope split
    out.append("\n### ⑤ Month × BTC-1h-slope sign at entry (avg % (N)) — does the bad-month set track the share of falling-BTC entries?\n")
    rows = []
    for mth, g in S.groupby("month"):
        p, n = g[g.slope_pos], g[~g.slope_pos]
        rows.append(dict(month=mth, N=len(g), avg=g.y.mean(), share_slope_le0=100 * (~g.slope_pos).mean(),
                         slope_pos=f"{p.y.mean():+.2f} ({len(p)})", slope_le0=f"{n.y.mean():+.2f} ({len(n)})"))
    M = pd.DataFrame(rows)
    out.append(C.md_table(M, 2))
    rho = M[["avg", "share_slope_le0"]].corr(method="spearman").iloc[0, 1]
    out.append(f"\nSpearman(month avg, share of falling-BTC entries) = {rho:+.2f} over {len(M)} months. "
               f"Months with slope>0 cohort avg > 0: {int((S[S.slope_pos].groupby('month').y.mean() > 0).sum())}/{len(M)}; "
               f"slope≤0 cohort avg > 0: {int((S[~S.slope_pos].groupby('month').y.mean() > 0).sum())}/{len(M)}.\n")

# ⑥ master real fills
out.append("\n## ⑥ Master real fills (MASTER_POOL_stacked.csv, stack_keep, stack_pct) — LITE / WILLY / LONG / WIDE BTC-1h-slope split\n")
m = pd.read_csv(os.path.join(C.REP, "MASTER_POOL_stacked.csv"), low_memory=False)
m = m[m.entry_strategy.astype(str).str.startswith("FRENZY") & m.stack_keep.astype(bool)].copy()
m["y"] = pd.to_numeric(m.stack_pct, errors="coerce"); m["sl"] = pd.to_numeric(m.entry_btc_1h_slope, errors="coerce")
m["off"] = pd.to_numeric(m.entry_btc_off24h_pct, errors="coerce")
rows = []
for s, g in m.groupby("entry_strategy"):
    for lab, mm in (("btc_1h_slope > 0", g.sl > 0), ("btc_1h_slope ≤ 0", g.sl <= 0), ("btc_off24h > −1.166", g.off > -1.166), ("btc_off24h ≤ −1.166", g.off <= -1.166)):
        h = g[mm]
        rows.append(dict(sleeve=s, zone=lab, N=len(h), days=h.opened_at.astype(str).str[:10].nunique(), WR=100 * (h.y > 0).mean() if len(h) else np.nan,
                         avg=h.y.mean() if len(h) else np.nan))
out.append(C.md_table(pd.DataFrame(rows), 3))
open(os.path.join(C.REP, "YR6_WL_FRENZY_willy.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
