#!/usr/bin/env python3
"""HT before → after on the MASTER (today's filters) and the CURRENT BATCH (11-18 → 11-23), the exact recommended rule:
BEFORE = today's exit (TP +6 maker / SL −4 / 24 h). AFTER = at the +6 touch no TP; trail k × 15m ATR14 (read at the touch) from the
running best; floor +5 net; 24 h from the touch; taker 0.05 + slip 0.10; funding paid/received while held. One size X 0.61 × $10k."""
import os, sys, importlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
HM = importlib.import_module("TP_RUNNER_HT_master_2026-11-24"); DV = importlib.import_module("TP_RUNNER_DIVE_2026-11-24")
T, C, DP = HM.T, HM.C, DV.DP
MIN, HR, X, ACC, SLIP = 60_000, 3_600_000, 0.61, 10_000, 0.10
SPECS = {"4×ATR · floor +5 (recommended)": (4, 5.0), "4×ATR · no floor": (4, None), "3×ATR · no floor": (3, None)}


def walk(pair, te, E):
    tp_px = E * (100 + 6.0 + 0.05) / (100 - 0.02); sl_px = E * (100 - 4.0 + 0.05) / (100 - 0.05)
    m0 = te // MIN * MIN; a = C.k1m(pair, m0, m0 + 48 * HR + MIN)
    if a is None or len(a["t"]) < 2: return None
    g = C.grid(a, m0, min(m0 + 48 * HR + MIN, int(a["t"][-1]) + MIN)); n24 = min(len(g["t"]) - 1, 24 * 60)
    net = lambda px: (px / E - 1) * 100 - 0.10
    for i in range(1, n24 + 1):
        if g["l"][i] <= sl_px:
            b = net(min(sl_px, g["o"][i]) * (1 - SLIP / 100)); return dict(before=b, **{k: b for k in SPECS}, why="SL")
        if g["h"][i] >= tp_px:
            t_touch = int(g["t"][i]); atr = T.atr15_pct(pair, t_touch); p = DP.path(pair, t_touch)
            o = dict(before=6.0, why="TP", atr=atr)
            for nm, (k, fnet) in SPECS.items():
                fl = None if fnet is None else E * (100 + fnet + 0.05) / (100 - 0.05)
                px, j, why = DV.trail_ix(p, tp_px, 1, k * atr, fl); px *= (1 - SLIP / 100)
                fund = DV.fund_between(pair, t_touch, t_touch + (j + 1) * MIN, E, px, 1)
                o[nm] = net(px) + fund; o[nm + "_why"] = why
            return o
    b = net(g["c"][n24] * (1 - SLIP / 100)); return dict(before=b, **{k: b for k in SPECS}, why="TIME")


def main():
    old = pd.read_csv(os.path.join(T.OUT, "ht_master_batch_runner.csv"))   # the 21 kept fills (master burst-rechecked + batch)
    EX = HM.exports(); EX["k19"] = EX.k
    rows = []
    for r in old.itertuples():
        e = EX[(EX.pair == r.pair) & (EX.k19 == str(r.k)[:19])]
        if not len(e): print("no entry", r.pair, r.k); continue
        for p in [r.pair]:
            ms = sorted({r.day[:7], (pd.Timestamp(r.day) + pd.Timedelta(days=2)).strftime("%Y-%m")})
            try: C.fetch_funding(p, ms)
            except SystemExit: pass
        w = walk(r.pair, HM.MS(r.k), float(e.entry_price.iloc[0]))
        if w: rows.append(dict(block=r.block, pair=r.pair, day=r.day, **w))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(T.OUT, "ht_ba_recommended.csv"), index=False)
    cols = [("before", "BEFORE · TP +6 (today)")] + [(k, "AFTER · " + k) for k in SPECS]
    for blk, g in R.groupby("block", sort=False):
        days = (pd.date_range(g.day.min(), g.day.max()) if blk == "MASTER" else pd.date_range("2026-11-18", "2026-11-23")).strftime("%Y-%m-%d")
        tp = g[g.why == "TP"]
        print(f"\n{blk}: {len(g)} kept fills · {len(days)} calendar days · TP hits {len(tp)} · exit mix {g.why.value_counts().to_dict()}")
        for c, nm in cols:
            x = g[c]; srt = np.sort(x.values)[::-1]
            print(f"  {nm:38s} N {len(x):2d} · WR {100*(x>0).mean():3.0f}% · avg {x.mean():+5.2f}% · ${(x*X*ACC/100).sum():+7,.0f} · "
                  f"DCR {HM.dcr(g, c, days):+5.2f}%/day · w/o best fill avg {srt[1:].mean():+5.2f}%")
            if c != "before":
                d = tp[c] - tp.before
                print(f"  {'':38s} per TP hit Δ {d.mean():+.2f} pp · median {d.median():+.2f} · ran {100*(tp[c+'_why']=='trail').mean():.0f}% / floor {100*(tp[c+'_why']=='floor').mean():.0f}% / 24h {100*(tp[c+'_why']=='time').mean():.0f}% · best Δ {d.max():+.1f}")


if __name__ == "__main__":
    main()
