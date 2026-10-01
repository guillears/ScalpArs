#!/usr/bin/env python3
"""Winners vs losers per sleeve — which entry conditions separate them, at PAIR level or MACRO (BTC/market) level?

Mirrors the UI table "Entry Conditions by Strategy — Winners vs Losers" (same stamped columns), on TWO independent sources:
  • MASTER  = live fills under today's stack (scripts/current_stack_ledger.build) — per-trade % = pnl_percentage (exit CFs re-priced)
  • BACKTEST = the full-year engine replay fills (--fills, all seeds; halves H1 = before --split, H2 = after)
For every condition: winner median vs loser median and AUC (P(winner value > loser value); 0.5 = no separation) in each
source. A condition is a CANDIDATE only when both sources point the same way with |AUC−0.5| ≥ --min-auc-gap.
Each candidate then gets ONE frozen threshold (the backtest-H1 loser-side cut that best isolates losers, N ≥ --min-block)
and is judged as a BLOCK on: backtest H1 / H2 (blocked N · WR · avg%) and the master per batch BEFORE → AFTER (N · WR · net $).
It is a SCREEN (dozens of conditions × sleeves): survivors still face the expectancy bar, window units and the haircut.

Usage: venv/bin/python scripts/winners_losers_separators.py --fills <year_report_fills.csv> [--sleeves MOM-long,Spike-Fade]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--fills", required=True)
ap.add_argument("--sleeves", default="MOM-long,MOM-short,FLIP-short,Spike-Fade")
ap.add_argument("--split", default="2026-05-01")
ap.add_argument("--min-auc-gap", type=float, default=0.08)
ap.add_argument("--min-block", type=int, default=15)
ap.add_argument("--top", type=int, default=6)
ap.add_argument("--min-side", type=int, default=5, help="min winners AND losers per source for an AUC")
ap.add_argument("--pairs", action="store_true", help="also run the 2D median-quadrant screen")
A = ap.parse_args()

PAIR = ["entry_rsi", "rsi_delta", "entry_adx", "adx_delta1", "entry_gap", "entry_ema_gap_5_8", "entry_ema_gap_8_13",
        "entry_ema5_stretch", "entry_price_vs_ema5_pct", "entry_ema20_slope", "entry_range_position", "entry_atr_pct",
        "entry_pair_volume_ratio", "entry_pair_ema20_ema50_gap_pct", "entry_dist_from_ema13_pct", "entry_pos_di",
        "entry_neg_di", "entry_quality_score", "entry_pair_rank", "entry_pair_age_days", "log_pair_vol_usd"]
MACRO = ["entry_btc_rsi", "btc_rsi_delta", "btc_rsi_30m_delta", "entry_btc_adx", "btc_adx_delta", "entry_btc_ema20_slope",
         "entry_btc_atr_pct", "entry_btc_1h_slope", "entry_btc_rsi_1h", "entry_btc_trend_gap_pct",
         "entry_btc_dist_from_ema13_pct", "entry_btc_off24h_pct", "entry_btc_off24lo_pct", "entry_btc_off30d_high_pct",
         "entry_btc_r72_pct", "entry_btc_eff72", "entry_btc_above72_pct", "entry_bull_pct", "entry_bear_pct",
         "entry_global_volume_ratio"]


def sleeve_of(strat, direction):
    s = strat if isinstance(strat, str) and strat else "MOMENTUM"
    if s.startswith("FLIP"):
        return "FLIP-short"
    if s == "MOMENTUM":
        return "MOM-long" if direction == "LONG" else "MOM-short"
    return {"SPIKE_FADE": "Spike-Fade", "BULLRUN_LONG": "BullRun-Long", "BEARRUN_SHORT": "BearRun-Short"}.get(s, s)


def derive(d):
    d = d.copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
    d["rsi_delta"] = num("entry_rsi") - num("entry_rsi_prev")
    d["adx_delta1"] = num("entry_adx") - num("entry_adx_prev")
    d["btc_rsi_delta"] = num("entry_btc_rsi") - num("entry_btc_rsi_prev")
    d["btc_rsi_30m_delta"] = num("entry_btc_rsi") - num("entry_btc_rsi_prev6")
    d["btc_adx_delta"] = num("entry_btc_adx") - num("entry_btc_adx_prev")
    d["log_pair_vol_usd"] = np.log10(num("entry_pair_volume_24h_usd").where(lambda x: x > 0))
    for c in PAIR + MACRO:
        d[c] = num(c) if c in d else np.nan
    return d


def auc(w, l):
    w, l = w.dropna().values, l.dropna().values
    if len(w) < A.min_side or len(l) < A.min_side:
        return np.nan
    allv = np.concatenate([w, l])
    ranks = pd.Series(allv).rank().values
    return (ranks[:len(w)].sum() - len(w) * (len(w) + 1) / 2) / (len(w) * len(l))


# ── sources ────────────────────────────────────────────────────────────────────────────────────────────────────────
import current_stack_ledger as LG
_argv = sys.argv
sys.argv = [sys.argv[0]]
M = LG.build()
sys.argv = _argv
M["sleeve"] = [sleeve_of(s, x) for s, x in zip(M.entry_strategy, M.direction)]
M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"),
                    M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
M = derive(M)
M["win"] = M.pct > 0

F = pd.read_csv(A.fills, low_memory=False)
F = F[~F.sleeve.astype(str).str.contains("PROBE|Probe", na=False)]
F = derive(F)
F["win"] = F.pnl_percentage > 0
F["half"] = np.where(F.opened_at.astype(str) < A.split, "H1", "H2")
ERAS = [e for e in LG.ERAS if e in set(M.era)]


def fmt(g, col):
    return f"{len(g):3d}·{(g[col] > 0).mean() * 100 if len(g) else 0:3.0f}%·{g[col].mean() if len(g) else 0:+.3f}"


for sl in [s.strip() for s in A.sleeves.split(",") if s.strip()]:
    m, f = M[M.sleeve == sl], F[F.sleeve == sl]
    if len(m) < 10 or len(f) < 50:
        print(f"\n## {sl}: too few fills (master {len(m)}, backtest {len(f)}) — skipped")
        continue
    print(f"\n## {sl} — master {fmt(m, 'pct')} · backtest {fmt(f, 'pnl_percentage')} (all seeds)")
    rows = []
    for lvl, cols in (("PAIR", PAIR), ("MACRO", MACRO)):
        for c in cols:
            am, af = auc(m[m.win][c], m[~m.win][c]), auc(f[f.win][c], f[~f.win][c])
            if np.isnan(am) or np.isnan(af):
                continue
            agree = np.sign(am - 0.5) == np.sign(af - 0.5) and min(abs(am - 0.5), abs(af - 0.5)) >= A.min_auc_gap
            rows.append(dict(level=lvl, cond=c, W_master=m[m.win][c].median(), L_master=m[~m.win][c].median(),
                             auc_master=am, W_bt=f[f.win][c].median(), L_bt=f[~f.win][c].median(), auc_bt=af, agree=agree))
    T = pd.DataFrame(rows)
    if not len(T):
        print(f"   → master has {int(m.win.sum())} winners / {int((~m.win).sum())} losers — too few on one side for any AUC (--min-side {A.min_side})")
        continue
    T["strength"] = np.minimum((T.auc_master - 0.5).abs(), (T.auc_bt - 0.5).abs())
    show = T.sort_values(["agree", "strength"], ascending=False)
    print(show.round(3).to_string(index=False))
    cands = show[show.agree].head(A.top)
    if not len(cands):
        print("   → no condition separates winners from losers in BOTH sources")
        continue
    for _, r in cands.iterrows():
        c = r.cond
        loser_low = r.auc_bt > 0.5                      # winners higher → losers sit LOW → block values below the cut
        h1 = f[(f.half == "H1") & f[c].notna()]
        best = None
        for q in np.linspace(0.1, 0.4, 7):             # loser-side tails only (never the middle)
            cut = h1[c].quantile(q if loser_low else 1 - q)
            blk = h1[h1[c] < cut] if loser_low else h1[h1[c] > cut]
            if len(blk) < A.min_block:
                continue
            score = blk.pnl_percentage.mean()
            if best is None or score < best[1]:
                best = (cut, score)
        if best is None:
            continue
        cut = best[0]
        isblk = (lambda x: x[c] < cut) if loser_low else (lambda x: x[c] > cut)
        side = f"< {cut:.4g}" if loser_low else f"> {cut:.4g}"
        fb = f[f[c].notna()]
        b1, b2 = fb[(fb.half == "H1") & isblk(fb)], fb[(fb.half == "H2") & isblk(fb)]
        k1, k2 = fb[(fb.half == "H1") & ~isblk(fb)], fb[(fb.half == "H2") & ~isblk(fb)]
        print(f"\n   ▶ [{r.level}] BLOCK {c} {side}   (frozen on backtest H1)")
        print(f"     backtest H1: blocked {fmt(b1, 'pnl_percentage')} vs kept {fmt(k1, 'pnl_percentage')} | "
              f"H2 (out of sample): blocked {fmt(b2, 'pnl_percentage')} vs kept {fmt(k2, 'pnl_percentage')}")
        mb = m[m[c].notna() & isblk(m)]
        print(f"     master blocked {len(mb)}·{(mb.pct > 0).mean() * 100 if len(mb) else 0:.0f}%·${mb.net.sum() if len(mb) else 0:+,.0f}"
              f"  (unstamped fills kept: {int(m[c].isna().sum())})")
        cells = []
        for e in ERAS:
            g = m[m.era == e]
            if not len(g):
                continue
            k = g[~(g[c].notna() & isblk(g))]
            cells.append(f"{e} {len(g)}·{(g.net > 0).mean() * 100:.0f}%·${g.net.sum():+,.0f}→{len(k)}·"
                         f"{(k.net > 0).mean() * 100 if len(k) else 0:.0f}%·${k.net.sum():+,.0f}")
        print("     per batch before→after: " + " | ".join(cells))


# ── 2D screen: every pair of conditions, median quadrants (cuts frozen on backtest H1) ─────────────────────────────
# A quadrant is a LOSER ZONE only if it is worse than the rest of the sleeve in backtest H1 AND H2 (out of sample) AND
# in the master. Counted in distinct DAYS too (window units): a market-wide (MACRO) zone lives on a handful of days.
def q2d(sl, m, f):
    cols = [c for c in PAIR + MACRO if f[c].notna().mean() > 0.8 and m[c].notna().mean() > 0.8]
    h1 = f[f.half == "H1"]
    med = {c: h1[c].median() for c in cols}
    fd = f.opened_at.astype(str).str[:10]
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            for sa in (True, False):
                for sb in (True, False):
                    fz = ((f[a] > med[a]) == sa) & ((f[b] > med[b]) == sb) & f[a].notna() & f[b].notna()
                    mz = ((m[a] > med[a]) == sa) & ((m[b] > med[b]) == sb) & m[a].notna() & m[b].notna()
                    r = {}
                    for h in ("H1", "H2"):
                        hz, hr = f[fz & (f.half == h)], f[~fz & (f.half == h)]
                        r[h] = (len(hz), hz.pnl_percentage.mean() - hr.pnl_percentage.mean(), hz.pnl_percentage.mean())
                    mzz, mr = m[mz], m[~mz]
                    if r["H1"][0] < 30 or r["H2"][0] < 30 or len(mzz) < 5:
                        continue
                    dm = mzz.pct.mean() - mr.pct.mean()
                    out.append(dict(zone=f"{a}{'>' if sa else '≤'}{med[a]:.3g} & {b}{'>' if sb else '≤'}{med[b]:.3g}",
                                    a=a, b=b, sa=sa, sb=sb,
                                    lvl=("PAIR" if a in PAIR else "MACRO") + "+" + ("PAIR" if b in PAIR else "MACRO"),
                                    nH1=r["H1"][0], dH1=r["H1"][1], avgH1=r["H1"][2], nH2=r["H2"][0], dH2=r["H2"][1],
                                    avgH2=r["H2"][2], days=fd[fz].nunique(), nM=len(mzz), wrM=(mzz.pct > 0).mean() * 100,
                                    avgM=mzz.pct.mean(), dM=dm, netM=mzz.net.sum()))
    Z = pd.DataFrame(out)
    if not len(Z):
        return
    Z["worst"] = Z[["dH1", "dH2", "dM"]].max(axis=1)          # the LEAST negative of the three Δs — all three must be < 0
    L = Z[(Z.dH1 < 0) & (Z.dH2 < 0) & (Z.dM < 0) & (Z.avgH1 < 0) & (Z.avgH2 < 0)].sort_values("worst")
    print(f"\n   2D loser zones consistent in backtest H1 + H2 + master: {len(L)} of {len(Z)} tested quadrants "
          f"(≈{len(Z) * 0.125:.0f} expected by luck at 1/8)")
    print(L.head(12)[["lvl", "zone", "nH1", "avgH1", "nH2", "avgH2", "days", "nM", "wrM", "avgM", "netM"]].round(3).to_string(index=False))
    for _, r in L.head(3).iterrows():
        isz = lambda x: ((x[r.a] > med[r.a]) == r.sa) & ((x[r.b] > med[r.b]) == r.sb) & x[r.a].notna() & x[r.b].notna()
        cells = []
        for e in ERAS:
            g = m[m.era == e]
            if not len(g):
                continue
            k = g[~isz(g)]
            cells.append(f"{e} {len(g)}·{(g.net > 0).mean() * 100:.0f}%·${g.net.sum():+,.0f}→{len(k)}·"
                         f"{(k.net > 0).mean() * 100 if len(k) else 0:.0f}%·${k.net.sum():+,.0f}")
        print(f"   ▶ BLOCK {r.zone}: per batch before→after: " + " | ".join(cells))


if A.pairs:
    for sl in [s.strip() for s in A.sleeves.split(",") if s.strip()]:
        m, f = M[M.sleeve == sl], F[F.sleeve == sl]
        if len(m) >= 10 and len(f) >= 50:
            print(f"\n## 2D {sl}")
            q2d(sl, m, f)
