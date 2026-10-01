#!/usr/bin/env python3
"""🌀 Backtest support for the BTC-chop momentum-long rules (operator 2026-10-01: "make the backtest to support this").

Data: the yr3 engine replay (the REAL engine on Jan-04 → Sep-24 2026 klines, 5 jittered-clock seeds), momentum LONG fills,
post-gate (current stack: no CROSS_OB_OPEN, CALM3D BTC-ATR floor, LOW-ADX RSI-momentum gate). eff72 is stamped by the replay on
every fill. Seeds replay the SAME year → collapsed to one row per trade (same pair, opens chained ≤ 10 min apart; pnl = seed mean; burst flags
computed within each seed first, over ALL replay fills BEFORE the gate filter — a fill the current gates block can still make the
next one a "2nd+", so R2 is slightly over-counted), and shown per seed only as a robustness line.

Rules under test (frozen 2026-10-01, CURRENT_STATE "ML BTC-CHOP OBSERVE"):
  R1  block every momentum long when eff72 ≤ 0.007
  R2  block only the 2nd+ fill of a burst (another bot fill ≤ 120 s earlier) when eff72 ≤ 0.007
Tests:
  A  cohort vs rest, per half and per month: N, WR, avg %, Δ vs rest
  B  like-for-like luck tests on the fill-level Δ = mean(chop) − mean(rest): a circular shift of the chop labels against the
     time-ordered fills, and a DAY-block bootstrap of Δ (chop lives on ~28 episodes, so even the day blocks are a little too
     narrow — the interval already spans zero). Episodes = runs of chop days with gaps ≤ 2 days.
  C  what-if in 1× units and with the size multipliers (Σ pct × inv × lev mult): actual vs R1 vs R2, per month
  D  per seed (robustness only), non-nested eff72 buckets, pre-gate fills
Limits (stated in the report): fill-level what-if — a blocked trade frees a slot the engine might have used (second-order effects
need a full replay with the rule coded); the replay finds live winners more easily than losers (memory: replay selection bias).
Usage: venv/bin/python scripts/chop_gate_backtest.py → reports/CHOP_GATE_BACKTEST_2026-10-01.md"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
OUT = os.path.join(ROOT, "reports", "CHOP_GATE_BACKTEST_2026-10-01.md")
CHOP, BURST_S, SPLIT = 0.007, 120, "2026-05-01"
n = lambda d, c: pd.to_numeric(d[c], errors="coerce")


def load(postgate=True):
    F = pd.read_csv("reports/backtest_cache/replay/year/yr3_report_fills.csv", low_memory=False)
    F["o"] = pd.to_datetime(F.opened_at, format="ISO8601")
    F["ts"] = (F.o - pd.Timestamp(0)).dt.total_seconds()
    b2 = pd.Series(False, index=F.index)                               # 2nd+ fill of a burst, within a seed, over ALL sleeves
    for _, g in F.groupby("seed"):
        g = g.sort_values("ts"); prev = g.ts.shift(1)
        b2.loc[g.index] = ((g.ts - prev) <= BURST_S).values
    F["burst2"] = b2
    F = F[(F.direction == "LONG") & (F.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")]
    if postgate:
        F = F[F.cell_multiplier_source.astype(str) != "CROSS_OB_OPEN"]
        F = F[~((F.cell_multiplier_source.astype(str) == "NONEXP_CALM3D") & (n(F, "entry_btc_atr_pct") < 0.08))]
        F = F[~((n(F, "entry_adx") < 21) & (n(F, "entry_rsi") < n(F, "entry_rsi_prev")))]
    F = F.copy()
    F["pct"] = F.pnl_percentage.astype(float); F["eff"] = n(F, "entry_btc_eff72")
    F["slope"] = n(F, "entry_btc_1h_slope"); F["ng"] = n(F, "peak_pnl").fillna(0) <= 0      # never green = the trade's peak P&L never above 0
    F["mult"] = (n(F, "cell_multiplier").fillna(1) * n(F, "cell_lev_multiplier").fillna(1)).clip(lower=0.05)
    F = F.sort_values(["pair", "ts"])                                   # one trade = same pair, opens ≤ 10 min apart (across seeds)
    new = (F.pair != F.pair.shift(1)) | ((F.ts - F.ts.shift(1)) > 600)
    F["k"] = new.cumsum()
    return F


def collapse(F):
    g = F.groupby("k")
    C = pd.DataFrame({"pct": g.pct.mean(), "eff": g.eff.mean(), "mult": g.mult.mean(), "burst2": g.burst2.mean() > 0.5,
                      "slope": g.slope.mean(), "ng": g.ng.mean() > 0.5,
                      "opened_at": g.opened_at.first(), "pair": g.pair.first(), "n_seeds": g.seed.nunique()}).reset_index(drop=True)
    C["day"] = C.opened_at.astype(str).str[:10]; C["month"] = C.opened_at.astype(str).str[:7]
    C = C.sort_values("opened_at").reset_index(drop=True)
    C["chop"] = C.eff <= CHOP
    return C


def episodes(days):
    """days (sorted unique date strings) → episode id per day (gap ≤ 2 days = same episode)."""
    d = pd.to_datetime(pd.Series(sorted(days))); ep = (d.diff().dt.days.fillna(99) > 2).cumsum()
    return dict(zip(d.dt.strftime("%Y-%m-%d"), ep))


def delta(C, mask):
    a, b = C[mask].pct, C[~mask].pct
    return (a.mean() - b.mean()) if len(a) and len(b) else np.nan


def perm_test(C, n_perm=5000, seed=11):
    """LIKE-FOR-LIKE test (review 2026-10-01: the first version compared chop FILLS with all fills on relabelled DAYS — a diluted
    null, p too small). Observed and null use the SAME statistic, fill-level Δ = mean(chop) − mean(rest):
      shift   circular shift of the chop label vector against the time-ordered fills (keeps the labels' clustering)
      boot    day-block bootstrap of Δ → 95 % CI and P(Δ ≥ 0)
    Returns (Δ, p_shift, lo, hi, p_boot, n_episodes)."""
    C = C.sort_values("opened_at").reset_index(drop=True)
    lab = C.chop.values; y = C.pct.values
    if lab.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0
    obs = y[lab].mean() - y[~lab].mean()
    rng = np.random.default_rng(seed)
    sh = []
    for k in rng.integers(1, len(C) - 1, n_perm):
        m = np.roll(lab, k); sh.append(y[m].mean() - y[~m].mean())
    days = C.day.values; ud = np.unique(days); by = {d: np.where(days == d)[0] for d in ud}
    bs = []
    for _ in range(n_perm):
        idx = np.concatenate([by[d] for d in rng.choice(ud, len(ud))])
        l2, y2 = lab[idx], y[idx]
        if l2.sum() >= 5 and (~l2).sum() >= 5:
            bs.append(y2[l2].mean() - y2[~l2].mean())
    bs = np.array(bs)
    return (obs, float((np.array(sh) <= obs).mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
            float((bs >= 0).mean()), len(set(episodes(C[C.chop].day.unique()).values())))


def cell(g):
    return f"{len(g)} · {g.day.nunique()}d · {(g.pct > 0).mean() * 100:.0f}% · {g.pct.mean():+.3f}" if len(g) else "–"


if __name__ == "__main__":
    F = load(True); C = collapse(F)
    L = ["# 🌀 BTC-chop rules — backtest support (yr3 engine replay, momentum longs, current gates)", "",
         f"{len(C)} unique trades (5 seeds collapsed), {C.day.nunique()} trading days, Jan-04 → Sep-24 2026. "
         f"Chop (eff72 ≤ {CHOP}): {int(C.chop.sum())} trades on {C[C.chop].day.nunique()} days = "
         f"{len(set(episodes(C[C.chop].day.unique()).values()))} episodes. Cells: trades · days · WR · avg %.", ""]
    L += ["## A. Chop vs the rest", "", "| Period | chop | chop ∧ 2nd+ burst | chop, other fills | not chop | Δ chop − rest |", "|---|---|---|---|---|---|"]
    for lab, m in (("Jan–Apr", C.opened_at.astype(str) < SPLIT), ("May–Sep", C.opened_at.astype(str) >= SPLIT), ("ALL", C.pct.notna())):
        g = C[m]
        L.append(f"| {lab} | {cell(g[g.chop])} | {cell(g[g.chop & g.burst2])} | {cell(g[g.chop & ~g.burst2])} | {cell(g[~g.chop])} | {delta(g, g.chop):+.3f} |")
    L += ["", "| Month | chop | not chop | Δ |", "|---|---|---|---|"]
    for mo, g in C.groupby("month"):
        L.append(f"| {mo} | {cell(g[g.chop])} | {cell(g[~g.chop])} | {delta(g, g.chop):+.3f} |" if g.chop.any() else f"| {mo} | – | {cell(g)} | – |")
    L += ["", "## B. Is the chop gap luck? (like-for-like: the same fill-level Δ in the observed and in the null)", ""]
    T = {}
    for lab, m in (("Jan–Apr", C.opened_at.astype(str) < SPLIT), ("May–Sep", C.opened_at.astype(str) >= SPLIT), ("ALL", C.pct.notna())):
        obs, p, lo, hi, pb, ne = perm_test(C[m].reset_index(drop=True)); T[lab] = (p, pb, lo, hi)
        L.append(f"- {lab}: Δ {obs:+.3f} over {ne} chop episodes · circular-shift p = **{p:.3f}** · day-block bootstrap 95 % CI [{lo:+.3f}, {hi:+.3f}], "
                 f"P(Δ ≥ 0) = **{pb:.3f}**")
    for lab, drop in (("without 2026-08", ["2026-08"]), ("without 2026-07 and 2026-08", ["2026-07", "2026-08"])):
        obs, p, lo, hi, pb, ne = perm_test(C[~C.month.isin(drop)].reset_index(drop=True)); T[lab] = (p, pb, lo, hi)
        L.append(f"- ALL {lab}: Δ {obs:+.3f} · circular-shift p = {p:.3f} · P(Δ ≥ 0) = {pb:.3f}")
    L.append("- By how many of the 5 replays took the trade (single-replay fills are the fragile ones): "
             + " · ".join(f"{k} seed{'' if k == '1' else 's'}: chop {cell(g[g.chop])} vs rest {g[~g.chop].pct.mean():+.3f}" for k, g in C.groupby(C.n_seeds.clip(upper=4).map(lambda v: f"{v}+" if v == 4 else str(v)))))
    L += ["", "## C. What-if PER SEED — one replay = one bot run (the union of five replays would triple the gain)", "",
          "| Seed | chop trades (winners) | R1 gain 1× | R1 gain ×mult | 2nd+ burst in chop (winners) | R2 gain 1× | R2 gain ×mult |", "|---|---|---|---|---|---|---|"]
    gains = []
    for sd, g in F.groupby("seed"):
        ch = g.eff <= CHOP; b = ch & g.burst2; w = g.pct * g.mult
        row = (-g.pct[ch].sum(), -w[ch].sum(), -g.pct[b].sum(), -w[b].sum()); gains.append(row)
        L.append(f"| {sd} | {int(ch.sum())} ({int((ch & (g.pct > 0)).sum())}) | {row[0]:+.1f} | {row[1]:+.1f} | {int(b.sum())} ({int((b & (g.pct > 0)).sum())}) | {row[2]:+.1f} | {row[3]:+.1f} |")
    gm = np.mean(gains, axis=0)
    L.append(f"| **mean** | | **{gm[0]:+.1f}** | **{gm[1]:+.1f}** | | **{gm[2]:+.1f}** | **{gm[3]:+.1f}** |")
    L.append(f"\nIn Σ-of-% points over ~9 months, before the 30–50 % in-sample haircut (0.007 was cut on the live master) → about "
             f"{gm[0] * 0.5:+.1f} to {gm[0] * 0.7:+.1f} at 1× for R1.")
    L += ["", "## D. Robustness", "", "Per seed (same year replayed — NOT independent): Δ chop − rest, and the chop ∧ 2nd+ burst average", ""]
    for s, g in F.groupby("seed"):
        g = g.assign(chop=g.eff <= CHOP)
        L.append(f"- seed {s}: chop {len(g[g.chop])} · {g[g.chop].pct.mean():+.3f} vs rest {g[~g.chop].pct.mean():+.3f} (Δ {g[g.chop].pct.mean() - g[~g.chop].pct.mean():+.3f}) · "
                 f"chop ∧ 2nd+ burst {len(g[g.chop & g.burst2])} · {g[g.chop & g.burst2].pct.mean():+.3f}")
    L += ["", "Non-nested eff72 buckets (trades · days · WR · avg %):", "", "| bucket | Jan–Apr | May–Sep |", "|---|---|---|"]
    edges = [(-1, 0.003), (0.003, 0.007), (0.007, 0.010), (0.010, 0.015), (0.015, 0.026), (0.026, 9)]
    for lo, hi in edges:
        m = (C.eff > lo) & (C.eff <= hi)
        L.append(f"| {max(lo, 0):.3f}–{hi if hi < 9 else '…'} | {cell(C[m & (C.opened_at.astype(str) < SPLIT)])} | {cell(C[m & (C.opened_at.astype(str) >= SPLIT)])} |")
    P = collapse(load(False))
    L += ["", f"Pre-gate (raw replay, {len(P)} trades): chop {cell(P[P.chop])} vs rest {cell(P[~P.chop])} · chop ∧ 2nd+ burst {cell(P[P.chop & P.burst2])}"]
    L += ["", "BTC 1h slope inside and outside chop (trades · days · WR · avg %):", "", "| | Jan–Apr | May–Sep |", "|---|---|---|"]
    h1 = C.opened_at.astype(str) < SPLIT
    for lab, m in (("chop ∧ slope ≥ 0", C.chop & (C.slope >= 0)), ("not chop ∧ slope ≥ 0", ~C.chop & (C.slope >= 0)),
                   ("chop ∧ slope < 0", C.chop & (C.slope < 0)), ("not chop ∧ slope < 0", ~C.chop & (C.slope < 0))):
        L.append(f"| {lab} | {cell(C[m & h1])} | {cell(C[m & ~h1])} |")
    L += ["", f"Never-green trades (peak P&L never above 0): chop {C[C.chop].ng.mean() * 100:.1f} % vs rest {C[~C.chop].ng.mean() * 100:.1f} % "
              "— no distinct loser mechanism."]
    L += ["", "## Verdict — OBSERVE, not arm (dual review 2026-10-01)", "",
          f"- Significance is borderline: full year p {T['ALL'][0]:.3f} (circular shift) / {T['ALL'][1]:.3f} (day-block bootstrap, 95 % CI "
          f"[{T['ALL'][2]:+.2f}, {T['ALL'][3]:+.2f}] still spans 0); Jan–Apr not significant ({T['Jan–Apr'][0]:.2f} / {T['Jan–Apr'][1]:.2f}); "
          f"May–Sep {T['May–Sep'][0]:.3f} / {T['May–Sep'][1]:.3f}; without Jul–Aug {T['without 2026-07 and 2026-08'][0]:.2f}.",
          "- No dose-response: the choppiest bucket (≤ 0.003) is fine in Jan–Apr; the signal sits in 0.003–0.007.",
          "- The live master's chop losers are the two days that prompted the idea (Sep-29, Oct-01); its five earlier chop trades netted +$564 (3 winners +$751; the Jul-10 ADA+LIT burst lost −$187).",
          "- Inside chop the gap appears when BTC's 1h slope is UP, not down (table in D).",
          "- The frozen CURRENT_STATE bar (fresh fills, N ≥ 15 on ≥ 8 days over ≥ 4 episodes) stays as registered.", "",
          "## Limits", "",
          "- Fill-level what-if: a blocked trade frees a slot the engine might have used — the true effect needs a replay with the rule coded.",
          "- The replay finds live winners more easily than live losers (replay selection bias) and the whole backtest sleeve is slightly negative.",
          "- One 9-month span; chop days cluster into few episodes, which is what limits the statistical power."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")
