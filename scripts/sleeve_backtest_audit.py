#!/usr/bin/env python3
"""Per-sleeve backtest audit against LIVE (operator 2026-09-29: "deep audit of the backtest, all sleeves, no bugs").

For one sleeve, inside every live batch window (first→last live fill of ANY sleeve in that era = live demonstrably running):
  1. LIVE RAW  = every full-size fill live actually took (pool as-traded, probes excluded; BASE era from the COMBINED raw pool
                 because the pool's BASE is pre-screened) — the truth the backtest must reproduce.
  2. MASTER    = the same fills after today's filters (current-stack ledger).
  3. BACKTEST  = the live-scan-clock replay (lp_* chunks, today's config).
  Trade match = same pair + direction, open within ±--match-min. Reports per batch: raw / master / backtest N·WR·avg%,
  reproduced-of-raw, same-trade avg raw→bt, missed (raw-only), extras (bt-only), and for the matched pairs the exit-parity stats
  (entry px diff, close-time diff, peak/trough diff, exit-reason crosstab). Missed fills get the backtest's nearest-scan gate
  from the decision journal; extras are tagged by the cell/door they came through.
Usage: venv/bin/python scripts/sleeve_backtest_audit.py --sleeve MOM-short [--match-min 20] [--out reports/AUDIT_<sleeve>.csv]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
REP = "reports/backtest_cache/replay"
ap = argparse.ArgumentParser()
ap.add_argument("--sleeve", required=True)
ap.add_argument("--tags", nargs="+", default=["lp_base1", "lp_base2", "lp_b1a", "lp_b1b", "lp_b2", "lp_b3", "lp_b45", "lp_b612"])
ap.add_argument("--match-min", type=float, default=20.0)
ap.add_argument("--out", default=None)
A = ap.parse_args()
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG
M = LG.build(); sys.argv = _a


def sleeve_of(strat, d):
    s = strat if isinstance(strat, str) and strat else "MOMENTUM"
    if s.startswith("FLIP"): return "FLIP-short"
    if s == "MOMENTUM": return "MOM-long" if d == "LONG" else "MOM-short"
    return {"SPIKE_FADE": "Spike-Fade", "BULLRUN_LONG": "BullRun-Long", "BEARRUN_SHORT": "BearRun-Short"}.get(s, s)


def prep(d):
    d = d.copy(); d["t"] = pd.to_datetime(d.opened_at.astype(str).str[:19]); d["sleeve"] = [sleeve_of(s, x) for s, x in zip(d.entry_strategy, d.direction)]
    return d


# ── live raw (as traded, full size) ──
P = prep(pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)); P = P[P.status == "CLOSED"]
P = P[~P.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE") & ~P.is_probe.astype(str).str.lower().isin(["true", "1"])]
W = P.groupby("era").t.agg(["min", "max"])                                  # live running windows (any sleeve)
C = prep(pd.read_csv("reports/COMBINED_momentum_flip_2026-06-16to28_DEDUP.csv", low_memory=False)); C = C[C.status == "CLOSED"]
C = C[~C.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE")]
base = C[(C.t >= W.loc["BASE", "min"]) & (C.t <= W.loc["BASE", "max"])].assign(era="BASE")
RAW = pd.concat([base, P[P.era != "BASE"]], ignore_index=True).drop_duplicates(["opened_at", "pair", "direction"])
RAW["pct"] = pd.to_numeric(RAW.pnl_percentage, errors="coerce")
# ── master ──
M = prep(M); M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"),
                                  M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
# ── backtest at live clock ──
parts = []
for t in A.tags:
    p = f"{REP}/{t}_orders.csv"
    if not os.path.exists(p): continue
    o = pd.read_csv(p, low_memory=False); meta = json.load(open(f"{REP}/{t}_meta.json"))
    o = prep(o); o = o[(o.status == "CLOSED") & (o.t >= pd.to_datetime(meta["start_ms"], unit="ms")) & (o.t < pd.to_datetime(meta["end_ms"], unit="ms"))]
    parts.append(o.assign(tag=t))
B = pd.concat(parts, ignore_index=True); B = B[~B.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE")]
B["pct"] = pd.to_numeric(B.pnl_percentage, errors="coerce")
JC = {}
def journal(tag, day):
    k = (tag, day)
    if k not in JC:
        p = f"{REP}/year/journal/{tag}/decisions-{day}.jsonl"
        JC[k] = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    return JC[k]
tagwin = {t: (pd.to_datetime(json.load(open(f"{REP}/{t}_meta.json"))["start_ms"], unit="ms"), pd.to_datetime(json.load(open(f"{REP}/{t}_meta.json"))["end_ms"], unit="ms")) for t in A.tags if os.path.exists(f"{REP}/{t}_meta.json")}

sl = A.sleeve
raw, mas, bt = RAW[RAW.sleeve == sl], M[M.sleeve == sl], B[B.sleeve == sl]
fm = lambda g: f"{len(g):3d}·{(g.pct > 0).mean() * 100 if len(g) else 0:3.0f}%·{g.pct.mean() if len(g) else 0:+.3f}"
print(f"\n### {sl} — LIVE RAW vs MASTER vs BACKTEST(live clock) per batch window  (N·WR·avg%)")
print(f"{'batch':5s} {'window':23s} | {'live raw':15s} | {'master':15s} | {'backtest':15s} | repro | same raw→bt      | {'missed(raw only)':16s} | extras(bt only)")
tot = dict(r=[], m=[], b=[], mt=[], bt=[], miss=[], ext=[]); dump = []; pairs = []
for e in LG.ERAS:
    if e not in W.index: continue
    lo, hi = W.loc[e, "min"], W.loc[e, "max"] + pd.Timedelta(minutes=1)
    if not any(a <= lo < z or a <= hi < z or (lo <= a and z <= hi) for a, z in tagwin.values()): continue
    r = raw[(raw.era == e) & (raw.t >= lo) & (raw.t < hi)]; m = mas[(mas.era == e) & (mas.t >= lo) & (mas.t < hi)]; b = bt[(bt.t >= lo) & (bt.t < hi)]
    used, match = set(), {}
    for i, x in r.iterrows():
        c = b[(b.pair == x.pair) & (b.direction == x.direction) & ((b.t - x.t).abs() <= pd.Timedelta(minutes=A.match_min))]; c = c[~c.index.isin(used)]
        if len(c): j = (c.t - x.t).abs().idxmin(); used.add(j); match[i] = j
    mt, bb = r.loc[list(match)], b.loc[list(match.values())]; miss, ext = r[~r.index.isin(match)], b[~b.index.isin(used)]
    print(f"{e:5s} {lo:%m-%d %H:%M}→{hi:%m-%d %H:%M} | {fm(r)} | {fm(m)} | {fm(b)} | {len(mt):2d}/{len(r):<2d} | {mt.pct.mean() if len(mt) else 0:+.3f}→{bb.pct.mean() if len(bb) else 0:+.3f} | {fm(miss):16s} | {fm(ext)}")
    for k, v in (("r", r.pct), ("m", m.pct), ("b", b.pct), ("mt", mt.pct), ("bt", bb.pct), ("miss", miss.pct), ("ext", ext.pct)): tot[k] += list(v)
    for i, j in match.items():
        x, y = r.loc[i], b.loc[j]
        pairs.append(dict(batch=e, pair=x.pair, dt_open_s=(y.t - x.t).total_seconds(), entry_diff_pct=(float(y.entry_price) / float(x.entry_price) - 1) * 100,
                          dclose_s=(pd.to_datetime(str(y.closed_at)[:19]) - pd.to_datetime(str(x.closed_at)[:19])).total_seconds(),
                          dpeak=pd.to_numeric(y.peak_pnl, errors="coerce") - pd.to_numeric(x.peak_pnl, errors="coerce"),
                          dtrough=pd.to_numeric(y.trough_pnl, errors="coerce") - pd.to_numeric(x.trough_pnl, errors="coerce"),
                          live_exit=str(x.close_reason).split(" L")[0], bt_exit=str(y.close_reason).split(" L")[0], live_pct=x.pct, bt_pct=y.pct,
                          live_cell=x.cell_multiplier_source, bt_cell=y.cell_multiplier_source))
        dump.append(dict(batch=e, kind="MATCHED", pair=x.pair, live_open=x.t, bt_open=y.t, live_pct=x.pct, bt_pct=y.pct, in_master=bool(((mas.pair == x.pair) & (mas.opened_at == x.opened_at)).any())))
    for i, x in miss.iterrows():
        tag = [t for t, (a, z) in tagwin.items() if a <= x.t < z]; gate = ""
        if tag:
            ev = [q for q in journal(tag[0], x.t.strftime("%Y-%m-%d")) if q.get("pair") == x.pair and q.get("e") == "BLOCK" and q.get("dir") in (None, x.direction) and abs((pd.Timestamp(q["t"]) - x.t).total_seconds()) <= 360]
            gate = ", ".join(pd.Series([q["gate"] for q in ev]).value_counts().head(3).index) if ev else "(no event: pair not scanned / outside universe)"
        dump.append(dict(batch=e, kind="MISSED", pair=x.pair, live_open=x.t, live_pct=x.pct, live_cell=x.cell_multiplier_source, in_master=bool(((mas.pair == x.pair) & (mas.opened_at == x.opened_at)).any()), bt_gate_near=gate))
    for j, y in ext.iterrows():
        dump.append(dict(batch=e, kind="EXTRA", pair=y.pair, bt_open=y.t, bt_pct=y.pct, bt_cell=y.cell_multiplier_source, bt_exit=str(y.close_reason)))
f2 = lambda v: f"{len(v):3d}·{(np.array(v) > 0).mean() * 100 if len(v) else 0:3.0f}%·{np.mean(v) if len(v) else 0:+.3f}"
print(f"{'ALL':5s} {'':23s} | {f2(tot['r'])} | {f2(tot['m'])} | {f2(tot['b'])} | {len(tot['mt']):2d}/{len(tot['r']):<2d} | {np.mean(tot['mt']) if tot['mt'] else 0:+.3f}→{np.mean(tot['bt']) if tot['bt'] else 0:+.3f} | {f2(tot['miss']):16s} | {f2(tot['ext'])}")
D = pd.DataFrame(dump); PP = pd.DataFrame(pairs)
if len(PP):
    same = PP[PP.dclose_s.abs() <= 60]
    print(f"\nMATCHED-TRADE PARITY ({len(PP)}): entry px diff median {PP.entry_diff_pct.median():+.4f}% (|p90| {PP.entry_diff_pct.abs().quantile(.9):.3f}) · open lag median {PP.dt_open_s.median():+.0f}s · "
          f"same close ≤60s: {len(same)} → peak diff median {same.dpeak.median():+.3f}, trough diff median {same.dtrough.median():+.3f} · cell equal {(PP.live_cell == PP.bt_cell).mean() * 100:.0f}%")
    print(f"  same-trade pnl: live {PP.live_pct.mean():+.3f} vs bt {PP.bt_pct.mean():+.3f} (Δ {PP.bt_pct.mean() - PP.live_pct.mean():+.3f}); sign agrees {((PP.live_pct > 0) == (PP.bt_pct > 0)).mean() * 100:.0f}%")
    print("  exit reasons live(rows) × bt(cols):"); print(pd.crosstab(PP.live_exit, PP.bt_exit).to_string())
    big = PP.assign(gap=PP.bt_pct - PP.live_pct); big = big[big.gap.abs() >= 0.3]
    if len(big): print("  |Δ|≥0.3 trades:"); print(big[["batch", "pair", "live_exit", "bt_exit", "live_pct", "bt_pct", "dclose_s"]].round(3).to_string(index=False))
mi = D[D.kind == "MISSED"] if len(D) else D
if len(mi):
    print(f"\nMISSED live fills ({len(mi)}; in master {int(mi.in_master.sum())}): backtest nearest-scan gate")
    print(mi.bt_gate_near.str.split(",").str[0].value_counts().to_string())
ex = D[D.kind == "EXTRA"] if len(D) else D
if len(ex):
    print(f"\nEXTRA backtest fills ({len(ex)}): by cell/door → N·WR·avg")
    print(ex.groupby("bt_cell").bt_pct.agg(["size", lambda s: (s > 0).mean() * 100, "mean"]).round(3).to_string())
out = A.out or f"reports/AUDIT_{sl}_trades.csv"; D.to_csv(out, index=False); print(f"\n→ {out}")
