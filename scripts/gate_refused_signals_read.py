#!/usr/bin/env python3
"""Price the signals a gate REFUSED with one simplified momentum exit, next to the trades the bot TOOK under the same simulator.

Source of refused signals: the replay decision journal (BLOCK events; one episode per pair-hour). Exit ("stack-lite"): SL −0.70 net
(longs on ATR ≥ 1 pairs: min(−0.7, −1.5×ATR) floored −1.2) · arm +0.40 · after the arm exit at max(0.10, 0.5×peak) · 5m bars,
adverse extreme first · fee 0.09 round trip. CALIBRATION is printed first (same trades, sim vs actual) — read the table as a
RELATIVE comparison only. The journal of a month chunk starts 3 warm-up days earlier, so a few July days appear.
Usage: venv/bin/python scripts/gate_refused_signals_read.py [--months 08 09] [--seed 1] [--out reports/GATE_REFUSED_SIGNALS.md]"""
import argparse, glob, io, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--months", nargs="+", default=["08", "09"]); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="reports/GATE_REFUSED_SIGNALS.md"); A = ap.parse_args()
GATES = (("ATR_GAP_LONG", "LONG"), ("FAN_RATIO_GATE", "LONG"), ("PAIR_ADX_DIR", "SHORT"), ("MOMENTUM_SHORT_LOATR", "SHORT"))
FEE = 0.09; K = {}


def k5(p):
    if p not in K:
        f = f"reports/backtest_cache/k5m_full/{p}.csv"; K[p] = None
        if os.path.exists(f):
            d = pd.read_csv(f); d["ot"] = pd.to_datetime(d.open_time, unit="ms"); d = d.sort_values("ot").reset_index(drop=True)
            c = d.c; tr = pd.concat([d.h - d.l, (d.h - c.shift()).abs(), (d.l - c.shift()).abs()], axis=1).max(axis=1)
            d["atrp"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100; K[p] = d
    return K[p]


def sim(p, t, direction, hours=6):
    d = k5(p)
    if d is None: return None
    i = d.ot.searchsorted(t, side="right")
    if i < 20 or i + 6 >= len(d): return None
    e = d.o.iloc[i]; atr = d.atrp.iloc[i - 1]; sl = -0.70
    if direction == "LONG" and atr >= 1.0: sl = max(min(-0.70, -1.5 * atr), -1.2)
    peak = 0.0; w = d.iloc[i:i + hours * 12]
    for h, l in zip(w.h.values, w.l.values):
        hi, lo = ((h / e - 1) * 100 - FEE, (l / e - 1) * 100 - FEE) if direction == "LONG" else ((e / l - 1) * 100 - FEE, (e / h - 1) * 100 - FEE)
        stop = sl if peak < 0.40 else max(0.10, 0.5 * peak)
        if lo <= stop: return stop
        peak = max(peak, hi)
    c = w.c.iloc[-1]; return ((c / e - 1) * 100 if direction == "LONG" else (e / c - 1) * 100) - FEE


rows = []
for mo in A.months:
    for fn in glob.glob(f"reports/backtest_cache/replay/year/journal/yr3_2026-{mo}_s{A.seed}/decisions-*.jsonl"):
        for line in open(fn):
            if '"e":"BLOCK"' not in line: continue
            for g, dr in GATES:
                if f'"gate":"{g}"' in line and f'"dir":"{dr}"' in line:
                    d = json.loads(line); rows.append((g, dr, d["pair"], pd.Timestamp(d["t"][:19])))
B = pd.DataFrame(rows, columns=["gate", "dir", "pair", "t"]).sort_values("t"); B["ep"] = B.t.dt.floor("60min"); B = B.drop_duplicates(["gate", "pair", "ep"])
B = pd.concat([g if len(g) <= 1500 else g.sample(1500, random_state=1) for _, g in B.groupby("gate")])
B["sim"] = [sim(p, t, d) for p, t, d in zip(B.pair, B.t, B.dir)]; B = B.dropna(subset=["sim"]); B["day"] = B.t.dt.date
lo, hi = f"2026-{min(A.months)}-01", f"2026-{int(max(A.months)) + 1:02d}-01"
P = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False); P = P[(P.entry_strategy == "MOMENTUM") & P.stack_keep.astype(bool) & (P.opened_at >= lo) & (P.opened_at < hi)].copy()
P["sim"] = [sim(p, pd.Timestamp(str(t)[:19]), d) for p, t, d in zip(P.pair, P.opened_at, P.direction)]; P = P.dropna(subset=["sim"])
out = io.StringIO(); w = lambda s="": print(s, file=out)
fm = lambda g: f"N {len(g):4d} · WR {(g.sim > 0).mean() * 100:3.0f}% · avg {g.sim.mean():+.3f} · full stops {(g.sim <= -0.69).mean() * 100:3.0f}%"
w(f"# Refused signals vs taken trades — months {A.months}, journal seed {A.seed}\n")
w(f"Calibration (live kept momentum fills, same trades): N {len(P)} · actual avg {P.pnl_percentage.mean():+.3f} / WR {(P.pnl_percentage > 0).mean() * 100:.0f}% · "
  f"sim avg {P.sim.mean():+.3f} / WR {(P.sim > 0).mean() * 100:.0f}% · sign agreement {(np.sign(P.sim) == np.sign(P.pnl_percentage)).mean() * 100:.0f}% · "
  f"corr {np.corrcoef(P.sim, P.pnl_percentage)[0, 1]:.2f}\n")
for d_ in ("LONG", "SHORT"): w(f"- momentum {d_}S taken (live kept): {fm(P[P.direction == d_])}")
for g, dr in GATES:
    x = B[B.gate == g]
    if len(x): w(f"- REFUSED by {g} ({dr}): {fm(x)} · days {x.day.nunique()} · positive days {(x.groupby('day').sim.mean() > 0).mean() * 100:.0f}% · by month "
                 + " / ".join(f"{m} {y.sim.mean():+.2f} (n{len(y)})" for m, y in x.groupby(x.t.dt.strftime('%m'))))
txt = out.getvalue(); print(txt); open(A.out, "w").write(txt); B.to_csv(A.out.replace(".md", "_signals.csv"), index=False); print("→", A.out)
