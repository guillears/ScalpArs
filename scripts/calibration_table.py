#!/usr/bin/env python3
"""Calibration table — master (live fills under today's stack) vs the live-phase-clock backtest (lp_* chunks, today's live
config), per batch, one sleeve. Window per batch = [first live fill of the era, last live fill + 1 h] on BOTH sides
(never compare days one side did not cover); chunk warm-up fills (before meta start_ms) are dropped.
Match = same pair + direction, open within ±--match-min. Reports: master N·WR·avg | backtest N·WR·avg | reproduced (recall)
| same-trade avg master→backtest | missed (live-only) | extras (backtest-only).
Usage: venv/bin/python scripts/calibration_table.py [--sleeve MOM-long] [--tags lp_base1 lp_base2 ...]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
REP = os.path.join(ROOT, "reports", "backtest_cache", "replay")
ap = argparse.ArgumentParser()
ap.add_argument("--sleeve", default="MOM-long")
ap.add_argument("--tags", nargs="+", default=["lp_base1", "lp_base2", "lp_b1a", "lp_b1b", "lp_b2", "lp_b3", "lp_b45", "lp_b612"])
ap.add_argument("--match-min", type=float, default=20.0)
ap.add_argument("--dump", default="reports/CALIBRATION_TABLE_trades.csv")
A = ap.parse_args()
_argv, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG
M = LG.build()
sys.argv = _argv


def sleeve_of(strat, direction):
    s = strat if isinstance(strat, str) and strat else "MOMENTUM"
    if s.startswith("FLIP"):
        return "FLIP-short"
    if s == "MOMENTUM":
        return "MOM-long" if direction == "LONG" else "MOM-short"
    return {"SPIKE_FADE": "Spike-Fade", "BULLRUN_LONG": "BullRun-Long", "BEARRUN_SHORT": "BearRun-Short"}.get(s, s)


M["t"] = pd.to_datetime(M.opened_at.astype(str).str[:19])
M["sleeve"] = [sleeve_of(s, d) for s, d in zip(M.entry_strategy, M.direction)]
M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"),
                    M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
parts = []
for t in A.tags:
    p = os.path.join(REP, f"{t}_orders.csv")
    if not os.path.exists(p):
        print(f"  (missing {t})"); continue
    o = pd.read_csv(p, low_memory=False)
    meta = json.load(open(os.path.join(REP, f"{t}_meta.json")))
    o["t"] = pd.to_datetime(o.opened_at.astype(str).str[:19])
    o = o[(o.status == "CLOSED") & (o.t >= pd.to_datetime(meta["start_ms"], unit="ms")) & (o.t < pd.to_datetime(meta["end_ms"], unit="ms"))]
    o["tag"] = t
    parts.append(o)
R = pd.concat(parts, ignore_index=True)
R["sleeve"] = [sleeve_of(s, d) for s, d in zip(R.entry_strategy, R.direction)]
R = R[(R.sleeve == A.sleeve) & ~R.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE")]
cover_lo, cover_hi = R.t.min(), R.t.max()
S = M[(M.sleeve == A.sleeve) & ~M.is_probe.fillna(0).astype(bool)]
fmt = lambda g, c: f"{len(g):3d}·{(g[c] > 0).mean() * 100 if len(g) else 0:3.0f}%·{g[c].mean() if len(g) else 0:+.3f}"
rows, dump = [], []
print(f"\n{A.sleeve} — master (current stack) vs backtest at live's scan clock (today's live config), ±{A.match_min:g} min match")
print(f"{'batch':5s} {'window':23s} | {'master':15s} | {'backtest':15s} | {'reproduced':10s} | {'same trades m→bt':17s} | {'missed':15s} | {'extras':15s}")
TOT = dict(m=[], b=[], mt=[], bt=[], miss=[], ext=[])
for e in LG.ERAS:
    eall = M[M.era == e]
    if not len(eall):
        continue
    lo, hi = eall.t.min(), eall.t.max() + pd.Timedelta(hours=1)
    if hi < cover_lo or lo > cover_hi:
        continue
    lo, hi = max(lo, cover_lo.floor("D")), min(hi, cover_hi + pd.Timedelta(hours=1))
    m = S[(S.era == e) & (S.t >= lo) & (S.t < hi)].copy()
    b = R[(R.t >= lo) & (R.t < hi)].copy()
    used, match = set(), {}
    for i, r in m.iterrows():
        c = b[(b.pair == r.pair) & ((b.t - r.t).abs() <= pd.Timedelta(minutes=A.match_min))]
        c = c[~c.index.isin(used)]
        if len(c):
            j = (c.t - r.t).abs().idxmin(); used.add(j); match[i] = j
    mt, bt = m.loc[list(match)], b.loc[list(match.values())]
    miss, ext = m[~m.index.isin(match)], b[~b.index.isin(used)]
    print(f"{e:5s} {lo:%m-%d %H:%M}→{hi:%m-%d %H:%M} | {fmt(m, 'pct')} | {fmt(b, 'pnl_percentage')} | {len(mt):3d}/{len(m):<3d}    | "
          f"{mt.pct.mean() if len(mt) else 0:+.3f}→{bt.pnl_percentage.mean() if len(bt) else 0:+.3f}   | {fmt(miss, 'pct')} | {fmt(ext, 'pnl_percentage')}")
    for k, v in (("m", m.pct), ("b", b.pnl_percentage), ("mt", mt.pct), ("bt", bt.pnl_percentage), ("miss", miss.pct), ("ext", ext.pnl_percentage)):
        TOT[k] += list(v)
    for i, j in match.items():
        dump.append(dict(batch=e, kind="MATCHED", pair=m.at[i, "pair"], live_open=m.at[i, "t"], bt_open=b.at[j, "t"],
                         live_pct=m.at[i, "pct"], bt_pct=b.at[j, "pnl_percentage"], live_exit=m.at[i, "close_reason"],
                         bt_tag=b.at[j, "tag"], bt_cell=b.at[j, "cell_multiplier_source"], bt_exit=b.at[j, "close_reason"]))
    for i, r in miss.iterrows():
        dump.append(dict(batch=e, kind="MISSED", pair=r.pair, live_open=r.t, live_pct=r.pct, live_exit=r.get("close_reason")))
    for j, r in ext.iterrows():
        dump.append(dict(batch=e, kind="EXTRA", pair=r.pair, bt_open=r.t, bt_pct=r.pnl_percentage, bt_exit=r.get("close_reason"),
                         bt_tag=r.tag, bt_cell=r.get("cell_multiplier_source")))
f2 = lambda v: f"{len(v):3d}·{(np.array(v) > 0).mean() * 100 if len(v) else 0:3.0f}%·{np.mean(v) if len(v) else 0:+.3f}"
print(f"{'ALL':5s} {'':23s} | {f2(TOT['m'])} | {f2(TOT['b'])} | {len(TOT['mt']):3d}/{len(TOT['m']):<3d}    | "
      f"{np.mean(TOT['mt']):+.3f}→{np.mean(TOT['bt']):+.3f}   | {f2(TOT['miss'])} | {f2(TOT['ext'])}")
pd.DataFrame(dump).to_csv(A.dump, index=False)
print(f"trade-level → {A.dump}")
