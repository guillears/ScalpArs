#!/usr/bin/env python3
"""Momentum-LONG observe reads (2026-10-01, DECISION_LOG 160/161): reproduces the BURST-CROWDING tally and the BTC-CHOP (eff72) read.

Definitions (frozen — the CURRENT_STATE bars quote these):
  burst       a momentum LONG opened ≤ 120 s from ANOTHER bot fill of any sleeve (MANUAL excluded) in the same book; in the backtest
              the neighbour must be in the SAME seed. Windows: fills ≤ 120 s apart = one window.
  btc chop    entry_btc_eff72 ≤ 0.007 (frozen, CURRENT_STATE "ML BTC-CHOP OBSERVE"; the STAMPED value = int(eff×1000)/1000, so raw
              eff < 0.008; stamped live since 2026-09-20; unstamped fills are excluded, never "rest"). Tiers: ≤0.007 · 0.007<eff≤0.026 ·
              >0.026 (comparison line only). Units: DAYS and chop EPISODES (stamped days ≥ 72 h apart = separate episodes).
  FRESH       --since (default 2026-10-01T02:00) — the observe bars count ONLY these: full-size (1×, no *_PROBE), no MANUAL.
  Bar checks  WR vs the PINNED momentum-long breakeven 61 % (Sep-25) · 95 % day-bootstrap of the mean · max day / pair share of the loss.
Book: current_stack_ledger.build() (validated master, current stack) + --fresh orders exports (full-size, non-MANUAL rows after the
master's last fill). Backtest: yr3 replay, post-gate, seeds collapsed to ONE row per trade (pair × 5-min bucket; pnl = seed mean).
Usage: venv/bin/python scripts/ml_regime_observe_read.py --fresh <orders.csv> [--fresh …] [--chop 0.007] [--since 2026-10-01T02:00]"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
FRESH = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--fresh" and i + 1 < len(sys.argv)]
_arg = lambda k, d: next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == k and i + 1 < len(sys.argv)), d)
CHOP_ARG = float(_arg("--chop", "0.007")); SINCE = pd.Timestamp(_arg("--since", "2026-10-01T02:00"))
BREAKEVEN = 61.0                                            # pinned (momentum-long sleeve economics, Sep-25)
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG  # noqa: E402
M = LG.build(); sys.argv = _a
n = lambda d, c: pd.to_numeric(d[c], errors="coerce")
BURST_S, CHOP = 120, CHOP_ARG

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
def bar(lab, g, unit):
    """The observe bar on FRESH fills: N, days (and chop episodes), WR vs the pinned breakeven, 95 % day-bootstrap, concentration."""
    if not len(g):
        return [f"- {lab}: 0 fresh fills"]
    d = g.assign(day=g.opened_at.astype(str).str[:10])
    v = d.groupby("day").pct.mean().values
    lo, hi = (np.percentile(np.random.default_rng(5).choice(v, (10000, len(v))).mean(axis=1), [2.5, 97.5]) if len(v) > 2 else (np.nan, np.nan))
    days = sorted(pd.to_datetime(d.day.unique()))
    ep = 0; last = None
    for t in days:
        if last is None or (t - last) >= pd.Timedelta(hours=72):
            ep += 1
        last = t
    loss = d[d.pct < 0]
    dshare = (loss.groupby("day").pct.sum().min() / loss.pct.sum()) if len(loss) else 0
    pshare = (loss.groupby("pair").pct.sum().min() / loss.pct.sum()) if len(loss) else 0
    meets = len(g) >= 15 and len(v) >= 8 and (unit != "chop" or ep >= 4)
    verdict = ("BLOCK CANDIDATE" if meets and (g.pct > 0).mean() * 100 < BREAKEVEN and hi < 0 and dshare < 0.5 and pshare < 0.5
               else "CLOSE (bar not met)" if meets else "keep counting")
    return [f"- {lab}: {len(g)} fills · {len(v)} days{f' · {ep} episodes' if unit == 'chop' else ''} · WR {(g.pct > 0).mean() * 100:.0f}% "
            f"(breakeven {BREAKEVEN:.0f}%) · avg {g.pct.mean():+.3f} · 95% day CI [{lo:+.2f}, {hi:+.2f}] · max day share {dshare:.0%} · "
            f"max pair share {pshare:.0%} → **{verdict}**"]


FRESH_ML = ML[pd.to_datetime(ML.opened_at, format="ISO8601") > SINCE]
fe = n(FRESH_ML, "entry_btc_eff72")
out += ["", f"## FRESH-only observe bars (fills after {SINCE:%Y-%m-%d %H:%M} UTC; full-size, non-MANUAL; unstamped excluded)", ""]
out += bar("burst crowding", FRESH_ML[FRESH_ML.burst], "burst")
out += bar(f"BTC chop eff72 ≤ {CHOP}", FRESH_ML[fe <= CHOP], "chop")
out += [f"- comparison (no bar): {CHOP} < eff72 ≤ 0.026 → {len(FRESH_ML[(fe > CHOP) & (fe <= 0.026)])} fills avg "
        f"{FRESH_ML[(fe > CHOP) & (fe <= 0.026)].pct.mean():+.3f} · eff72 > 0.026 → {len(FRESH_ML[fe > 0.026])} fills avg "
        f"{FRESH_ML[fe > 0.026].pct.mean():+.3f}"]
OUT = os.path.join(ROOT, "reports", "ML_REGIME_OBSERVE_READ_2026-10-01.md")
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out)); print(f"\n→ {OUT}")
