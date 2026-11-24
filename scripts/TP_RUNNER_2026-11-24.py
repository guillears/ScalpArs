#!/usr/bin/env python3
"""TP_RUNNER_2026-11-24 — SPLIT POSITION at the TP (research only; no bot code / config touched).
Operator 11-24 (PROM / ON / CYBER / EVAA ran far past the TP): instead of predicting runners, keep a RUNNER share p of every position:
the maker TP is placed for (1 − p) from the start (no cancel race); the runner share has no TP order — once price trades through the TP
level a trail starts (W × the coin's 15m ATR14 % at the touch, from the running best; gap-aware) with a 🛡 floor (none · entry price
(break-even) · half the TP · TP − 1 × ATR); 24 h cap after the touch → market. Runner exit = taker 0.05 + slip 0.10 vs the maker 0.02.
Fills that never reach the TP are unchanged. Δ per TP hit for the runner = (runner exit − TP) in pp of entry − 0.13; per fill Δ × p.
Cohorts = TP_HOLD_2026-11-24 (real fills + backtest, the same TP hits, today's exits). Pick the best (W, floor) per sleeve on DESIGN,
confirm on HOLDOUT + day-block bootstrap."""
import os, sys, importlib, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
T = importlib.import_module("TP_HOLD_2026-11-24"); C = T.C
OUT = T.OUT; MIN, HR = 60_000, 3_600_000
WS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
FLOORS = ("none", "BE", "halfTP", "TP-1atr")


def walk(g, i0, tp, d, w, fl):
    best = tp
    n = len(g["t"])
    for i in range(i0 + 1, n):
        hi, lo, op = g["h"][i], g["l"][i], g["o"][i]
        if d < 0:
            st = best * (1 + w / 100)
            if fl is not None: st = min(st, fl)
            if hi >= st: return max(st, op)
            best = min(best, lo)
        else:
            st = best * (1 - w / 100)
            if fl is not None: st = max(st, fl)
            if lo <= st: return min(st, op)
            best = max(best, hi)
    return g["c"][n - 1]


def outcomes(pair, t_touch, E, tp, d, atr):
    T0 = t_touch // MIN * MIN
    a = C.k1m(pair, T0, T0 + 24 * HR + MIN)
    if a is None or not len(a["t"]) or not np.isfinite(atr) or atr <= 0:
        return None
    g = C.grid(a, T0, min(T0 + 24 * HR + MIN, int(a["t"][-1]) + MIN))
    o = {}
    tpp = abs(tp / E - 1) * 100
    for fn in FLOORS:
        if fn == "none": fl = None
        elif fn == "BE": fl = E
        elif fn == "halfTP": fl = E * (1 + d * tpp / 200)
        else: fl = tp * (1 - d * atr / 100)
        for w in WS:
            px = walk(g, 0, tp, d, w * atr, fl)
            o[f"{fn}|{w:g}"] = d * (px / tp - 1) * 100 * (tp / E) - T.COST_PP
    return o


def run(df, label):
    rows = []
    for r in df.itertuples():
        o = outcomes(r.pair, int(r.t_touch), float(r.E), float(r.tp_px), int(r.dirn), float(r.atr15))
        if o is None: continue
        o.update(sleeve=r.sleeve, pair=r.pair, t_touch=int(r.t_touch), period=getattr(r, "period", "REAL"), src=getattr(r, "src", ""))
        rows.append(o)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, f"runner_{label}.csv"), index=False)
    return R


def ci(h, col, rng):
    x = h.assign(day=pd.to_datetime(h.t_touch, unit="ms").dt.floor("D")).groupby("day")[col].sum()
    bs = [x.sample(len(x), replace=True, random_state=int(rng.integers(1e9))).sum() / len(h) for _ in range(1000)]
    return np.quantile(bs, .025), np.quantile(bs, .975)


def main():
    B = pd.read_csv(os.path.join(OUT, "bt_tp_hits.csv"))
    Rb = run(B, "bt")
    real = pd.read_csv(os.path.join(OUT, "real_tp_hits.csv"))
    Rr = run(real, "real")
    cols = [f"{f}|{w:g}" for f in FLOORS for w in WS]
    rng = np.random.default_rng(3)
    pick = {}
    for s, g in Rb.groupby("sleeve"):
        des, hol = g[g.period == "DESIGN"], g[g.period == "HOLDOUT"]
        tab = pd.DataFrame({c: [des[c].mean(), hol[c].mean(), (des[c] > 0).mean(), (hol[c] > 0).mean()] for c in cols},
                           index=["DESIGN avg", "HOLDOUT avg", "DESIGN better", "HOLDOUT better"]).T
        tab.to_csv(os.path.join(OUT, f"runner_grid_{s}.csv"))
        best = tab["DESIGN avg"].idxmax()
        lo, hi = ci(hol, best, rng); dlo, dhi = ci(des, best, rng)
        pick[s] = dict(best=best, design=float(tab.loc[best, "DESIGN avg"]), design_ci=[dlo, dhi], holdout=float(tab.loc[best, "HOLDOUT avg"]),
                       holdout_ci=[lo, hi], n_des=len(des), n_hol=len(hol),
                       real=float(Rr[Rr.sleeve == s][best].mean()) if (Rr.sleeve == s).any() else None, n_real=int((Rr.sleeve == s).sum()))
        print(f"\n{s}: DESIGN n{len(des)} · HOLDOUT n{len(hol)}")
        print(tab.round(2).sort_values("DESIGN avg", ascending=False).head(8).to_string())
        print(f"  ▶ pick on DESIGN: {best} → DESIGN {pick[s]['design']:+.2f} (CI {dlo:+.2f}…{dhi:+.2f}) · HOLDOUT {pick[s]['holdout']:+.2f} (CI {lo:+.2f}…{hi:+.2f}) · REAL n{pick[s]['n_real']} {pick[s]['real']}")
    json.dump(pick, open(os.path.join(OUT, "runner_pick.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
