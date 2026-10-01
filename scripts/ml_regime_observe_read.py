#!/usr/bin/env python3
"""Momentum-LONG observe reads (2026-10-01, DECISION_LOG 160/161): reproduces the BURST-CROWDING tally and the BTC-CHOP (eff72) read.

Definitions (frozen — the CURRENT_STATE bars quote these):
  burst       a momentum LONG opened ≤ 120 s from ANOTHER bot fill of any sleeve (MANUAL excluded) in the same book; in the backtest
              the neighbour must be in the SAME seed. Windows: fills ≤ 120 s apart = one window.
  btc chop    entry_btc_eff72 ≤ 0.010 (BTC 72 h efficiency = |net move| / path; stamped live since 2026-09-20).
Book: current_stack_ledger.build() (validated master, current stack) + --fresh orders exports (full-size, non-MANUAL rows after the
master's last fill). Backtest: yr3 replay, post-gate, seeds collapsed to ONE row per trade (pair × 5-min bucket; pnl = seed mean).
Usage: venv/bin/python scripts/ml_regime_observe_read.py --fresh <orders.csv> [--fresh …]"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
FRESH = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--fresh" and i + 1 < len(sys.argv)]
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG  # noqa: E402
M = LG.build(); sys.argv = _a
n = lambda d, c: pd.to_numeric(d[c], errors="coerce")
BURST_S, CHOP = 120, 0.010

M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"), M.stack_pnl / n(M, "notional_value") * 100,
                    M.pnl_percentage)
last = pd.to_datetime(M.opened_at, format="ISO8601").max()
fr = []
for f in FRESH:
    b = pd.read_csv(f, low_memory=False)
    b = b[(b.status == "CLOSED") & (pd.to_datetime(b.opened_at, format="ISO8601") > last)
          & ~b.cell_multiplier_source.astype(str).str.contains("_PROBE")].copy()
    b["pct"] = b.pnl_percentage; b["era"] = "FRESH"; fr.append(b)
A = pd.concat([M] + fr, ignore_index=True).drop_duplicates(["opened_at", "pair", "direction"])
A = A[A.entry_strategy.fillna("MOMENTUM") != "MANUAL"].copy()
A["o"] = pd.to_datetime(A.opened_at, format="ISO8601")


def label(book, by=None):
    """burst flag for each momentum LONG of `book` (neighbours searched within the same `by` group, e.g. seed)."""
    flags = {}
    for _, g in (book.groupby(by) if by else [(None, book)]):
        t = ((g.o - pd.Timestamp(0)).dt.total_seconds()).values; idx = g.index.values   # unit-safe (pandas may parse to µs)
        ml = ((g.direction == "LONG") & (g.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")).values
        for i in np.where(ml)[0]:
            d = np.abs(t - t[i]); d[i] = 10**9
            flags[idx[i]] = bool((d <= BURST_S).any())
    return pd.Series(flags)


def windows(g, by=None):
    g = g.sort_values([by, "o"] if by else "o"); w, cur, prev, pk = [], -1, None, None
    for t, k in zip(g.o, g[by] if by else [0] * len(g)):
        if prev is None or (t - prev).total_seconds() > BURST_S or k != pk:
            cur += 1
        w.append(cur); prev, pk = t, k
    return g.assign(win=w)


def line(lab, g, by=None, unit="win"):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – |"
    g = windows(g, by) if unit == "win" else g.assign(win=g.opened_at.astype(str).str[:10])
    v = g.groupby("win").pct.mean().values
    b = np.random.default_rng(1).choice(v, (10000, len(v))).mean(axis=1) if len(v) > 2 else np.array([np.nan])
    return (f"| {lab} | {len(g)} | {len(v)} | {(g.pct > 0).mean() * 100:.0f}% | {g.pct.mean():+.3f} | "
            f"[{np.nanpercentile(b, 2.5):+.2f}, {np.nanpercentile(b, 97.5):+.2f}] |")


ML = A[(A.direction == "LONG") & (A.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].copy()
ML["burst"] = label(A).reindex(ML.index).fillna(False).astype(bool)
neg = n(ML, "entry_btc_1h_slope") < 0
eff = n(ML, "entry_btc_eff72")
H = "| Cohort | N | windows | WR | avg % | 95 % window CI |\n|---|---|---|---|---|---|"
out = ["# Momentum-LONG observe reads — burst crowding & BTC chop", "", f"Master current stack + fresh: {len(ML)} momentum longs.", "",
       "## Burst crowding (master)", "", H, line("burst", ML[ML.burst]), line("alone", ML[~ML.burst]),
       line("slope<0 ∧ burst", ML[neg & ML.burst]), line("slope<0 ∧ alone", ML[neg & ~ML.burst]), "",
       f"## BTC chop eff72 ≤ {CHOP} (master; stamped on {int(eff.notna().sum())} fills — unstamped fills are NOT 'rest'; unit = DAY)", "",
       H.replace("windows", "days").replace("window CI", "day CI"),
       line(f"eff72 ≤ {CHOP}", ML[eff <= CHOP], unit="day"), line(f"eff72 > {CHOP}", ML[eff > CHOP], unit="day")]

F = pd.read_csv("reports/backtest_cache/replay/year/yr3_report_fills.csv", low_memory=False)
F = F[F.cell_multiplier_source.astype(str) != "CROSS_OB_OPEN"]
F = F[~((F.cell_multiplier_source.astype(str) == "NONEXP_CALM3D") & (n(F, "entry_btc_atr_pct") < 0.08))]
F = F[~((F.direction == "LONG") & (F.entry_strategy.fillna("MOMENTUM") == "MOMENTUM") & (n(F, "entry_adx") < 21)
        & (n(F, "entry_rsi") < n(F, "entry_rsi_prev")))].copy()
F["o"] = pd.to_datetime(F.opened_at, format="ISO8601")
F["burst"] = label(F, by="seed").reindex(F.index)
FL = F[(F.direction == "LONG") & (F.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].copy()
FL["k"] = FL.pair.astype(str) + "|" + FL.o.dt.floor("5min").astype(str)
agg = FL.groupby("k").agg(pct=("pnl_percentage", "mean"), burst=("burst", "mean"))
FL = FL.drop_duplicates("k").set_index("k"); FL["pct"] = agg.pct; FL["burst"] = agg.burst >= 0.5; FL = FL.reset_index()
fneg = n(FL, "entry_btc_1h_slope") < 0; feff = n(FL, "entry_btc_eff72"); h1 = FL.opened_at.astype(str) < "2026-05-01"
out += ["", "## yr3 backtest, post-gate, seeds collapsed (H1 Jan–Apr · H2 May–Sep)", "", H]
for lab, hm in (("H1", h1), ("H2", ~h1)):
    out += [line(f"{lab} slope<0 ∧ burst", FL[fneg & FL.burst & hm]), line(f"{lab} slope<0 ∧ alone", FL[fneg & ~FL.burst & hm])]
out += ["", H.replace("windows", "days").replace("window CI", "day CI")]
for lab, hm in (("H1", h1), ("H2", ~h1)):
    out += [line(f"{lab} eff72 ≤ {CHOP}", FL[(feff <= CHOP) & hm], unit="day"), line(f"{lab} eff72 > {CHOP}", FL[(feff > CHOP) & hm], unit="day")]
OUT = os.path.join(ROOT, "reports", "ML_REGIME_OBSERVE_READ_2026-10-01.md")
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out)); print(f"\n→ {OUT}")
