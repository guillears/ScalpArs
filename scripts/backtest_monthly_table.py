#!/usr/bin/env python3
"""Full-year backtest per MONTH per SLEEVE (all seeds), with the MASTER per month per sleeve alongside.

Backtest = the full-year engine replay fills (--fills; yr2 = 5 seeds, frozen config 5442bbc Sep-25). TODAY'S-RULES adjustment
(post-filter, exact rules; path effects such as freed slots are NOT re-simulated): CROSS_OB_OPEN fills removed (switched OFF
Sep-28, fd31efb) and NONEXP_CALM3D fills at entry_btc_atr_pct < nonexp_calm3d_btc_atr_min removed (Sep-27, 0e1ca92). Not
representable without a re-run: the ⚡ ADX-surge waiver admits (Sep-28; smoke 3 days = 0 trades).
Cells: N per seed · WR · avg P&L % (1×, leverage-invariant) · Σ% per seed (sum of trade %s = the month's contribution) · sw = SIZE-WEIGHTED
avg % (Σ w·pct / Σ w, w = cell multiplier × leverage multiplier: what today's cell sizing would have earned — operator-corrected Sep-29:
conditional sizing IS an edge at portfolio level, so both readings are always shown; a sw below the 1× read = the cells are pointed at
cohorts that lose in that window).
Master = current-stack ledger (real live fills), % re-priced for exit CFs (ARM040 / LATE_ARM / FADE_SL).
Integrity asserts run first (duplicates, probes, sign, seeds); any failure aborts.
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--fills", default="reports/backtest_cache/replay/year/yr2_report_fills.csv")
ap.add_argument("--raw", action="store_true", help="no today's-rules adjustment (as the frozen config ran)")
ap.add_argument("--out", default="reports/BACKTEST_MONTHLY_BY_SLEEVE.csv")
A = ap.parse_args()
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG
import config
M = LG.build(); sys.argv = _a

F = pd.read_csv(A.fills, low_memory=False)
# ── integrity (abort on failure) ──
assert F.duplicated(["seed", "pair", "direction", "opened_at"]).sum() == 0, "duplicate fills"
assert (F.status == "CLOSED").all(), "non-closed fills"
assert not F.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE").any(), "probe fills present"
assert ((np.sign(F.pnl_percentage) == np.sign(F.pnl)) | (F.pnl.abs() <= 0.5)).all(), "pct/pnl sign mismatch"
seeds = sorted(F.seed.unique()); S = len(seeds)
n0 = len(F)
if not A.raw:
    th = config.trading_config.thresholds
    xob = F.cell_multiplier_source.astype(str).eq("CROSS_OB_OPEN")
    floor = float(getattr(th, "nonexp_calm3d_btc_atr_min", 0) or 0)
    batr = pd.to_numeric(F.entry_btc_atr_pct, errors="coerce")
    c3 = F.cell_multiplier_source.astype(str).eq("NONEXP_CALM3D") & (batr < floor)
    print(f"today's-rules adjustment: removed CROSS_OB_OPEN {int(xob.sum())} · CALM3D BTC-ATR<{floor:g} {int(c3.sum())} of {n0} fills")
    F = F[~(xob | c3)]
F["mon"] = F.opened_at.astype(str).str[:7]

def sleeve_of(strat, d):
    s = strat if isinstance(strat, str) and strat else "MOMENTUM"
    if s.startswith("FLIP"): return "FLIP-short"
    if s == "MOMENTUM": return "MOM-long" if d == "LONG" else "MOM-short"
    return {"SPIKE_FADE": "Spike-Fade", "BULLRUN_LONG": "BullRun-Long", "BEARRUN_SHORT": "BearRun-Short"}.get(s, s)
M["sleeve"] = [sleeve_of(s, d) for s, d in zip(M.entry_strategy, M.direction)]
M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"),
                    M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
M["mon"] = M.opened_at.astype(str).str[:7]
# same-window rule (Sep-25 lesson): the master is cut at the backtest's last fill — never compare days one side did not cover
_bt_end = pd.to_datetime(F.opened_at.astype(str).str[:19]).max()
_mt = pd.to_datetime(M.opened_at.astype(str).str[:19])
print(f"master window cut at backtest end {_bt_end:%Y-%m-%d %H:%M}: {int((_mt > _bt_end).sum())} master fills after it excluded")
M = M[_mt <= _bt_end]

SLEEVES = ["MOM-long", "MOM-short", "FLIP-short", "Spike-Fade", "BullRun-Long", "BearRun-Short"]
rows = []
def _w(g):
    m = pd.to_numeric(g.get("cell_multiplier"), errors="coerce").fillna(1.0) if "cell_multiplier" in g else pd.Series(1.0, index=g.index)
    lv = pd.to_numeric(g.get("cell_lev_multiplier"), errors="coerce").fillna(1.0) if "cell_lev_multiplier" in g else pd.Series(1.0, index=g.index)
    return (m * lv).clip(lower=0.05)
def cell(g, col, per):
    if not len(g): return "—", dict(n=0)
    n = len(g) / per; wr = (g[col] > 0).mean() * 100; av = g[col].mean(); sm = g[col].sum() / per
    sw = float(np.average(g[col], weights=_w(g)))
    return f"{n:5.1f}·{wr:3.0f}%·{av:+.3f}·Σ{sm:+6.1f}·sw{sw:+.3f}", dict(n=n, wr=wr, avg=av, sum=sm, sw=sw)
months = sorted(F.mon.unique())
for sl in SLEEVES + ["ALL"]:
    f = F if sl == "ALL" else F[F.sleeve == sl]
    m = M if sl == "ALL" else M[M.sleeve == sl]
    print(f"\n## {sl}   (backtest: N/seed·WR·avg%·Σ%/seed·sw  |  master: N·WR·avg%·Σ%·sw   — sw = size-weighted avg %)")
    for mo in months + ["TOTAL"]:
        g = f if mo == "TOTAL" else f[f.mon == mo]
        gm = m[m.mon.isin(months)] if mo == "TOTAL" else m[m.mon == mo]
        bt, bd = cell(g, "pnl_percentage", S)
        ms, md = cell(gm, "pct", 1)
        ps = " ".join(f"{g[g.seed == s].pnl_percentage.sum():+5.1f}" for s in seeds) if len(g) else ""
        print(f"  {mo:7s} bt {bt:39s} seeds Σ% [{ps}]  | master {ms}")
        rows.append(dict(sleeve=sl, month=mo, **{f"bt_{k}": v for k, v in bd.items()}, **{f"master_{k}": v for k, v in md.items()}))
pd.DataFrame(rows).to_csv(A.out, index=False)
print(f"\n→ {A.out}  (seeds {seeds}; master months shown only where the backtest covers them)")
