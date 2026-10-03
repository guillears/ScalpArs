import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _a, sys.argv = sys.argv, ["x"]
import numpy as np, pandas as pd
import frenzy_scalp_pattern_search_v2 as P2, frenzy_scalp_followup as FU, frenzy_week_combos_year as W, frenzy_exit_bullrun_test as XB
sys.argv = _a; H = P2.H; SPLIT = "2026-05-01"; RNG = np.random.default_rng(29); S = os.environ["S"]
Y = pd.read_csv(S + "/frenzy_gvr_trades.csv"); idx = FU.episode_index()
EX = {"FRENZY exit (−3 · trail +5/1.5 · 12 h)": dict(live=True), "Bull-Run TP + −3 (BE +1 → +0.2 · 2×ATR trail · ladder)": dict(be=True, atr=2.0, ladder=True),
      "Bull-Run with 1×ATR trail + −3": dict(be=True, atr=1.0, ladder=True), "FRENZY trail + ladder": dict(live=True, ladder=True),
      "fixed +2 / −3": (2.0, 3.0), "fixed +1 / −3": (1.0, 3.0)}
out = {k: [] for k in EX}
for r in Y.itertuples():
    p = FU.path(idx, r.pair, int(r.t)); tt, hh, ll, cc = p; e = cc[0]; hh, ll, cc = hh[1:], ll[1:], cc[1:]
    for k, ex in EX.items():
        out[k].append(W.walk(hh, ll, cc, e, r.atr, ex, 0.02)[0] - 0.09)
for k in EX: Y[k] = out[k]
Y.to_csv(S + "/frenzy_gvr_exits.csv", index=False)
base = "FRENZY exit (−3 · trail +5/1.5 · 12 h)"
for nm, Z in (("QUIET market (global vol < 1.0) — all", Y[Y.gvr < 1.0]), ("  FRENZY part", Y[(Y.gvr < 1.0) & Y.fz]), ("  WIDE part", Y[(Y.gvr < 1.0) & ~Y.fz]), ("BUSY market (≥ 1.0) — reference", Y[Y.gvr >= 1.0])):
    print(f"\n### {nm} · {len(Z)} trades\n\n| exit | won | avg %/trade | Jan–Apr / May–Sep | vs FRENZY exit, 95 % by day | months better |\n|---|---|---|---|---|---|")
    h1 = Z.day < SPLIT
    for k in EX:
        d = Z[k] - Z[base]
        if k == base: ci = "–"; mb = "–"
        else:
            bd = H.boot(Z.assign(net=d), "day", 1500); ci = f"{d.mean():+.3f} [{bd[0]:+.3f}, {bd[1]:+.3f}]"
            mm = d.groupby(Z.day.str[:7]).mean(); mb = f"{(mm > 0).sum()} of {len(mm)}"
        print(f"| {k} | {(Z[k] > 0).mean() * 100:.0f}% | **{Z[k].mean():+.3f}** | {Z[h1][k].mean():+.3f} / {Z[~h1][k].mean():+.3f} | {ci} | {mb} |")
