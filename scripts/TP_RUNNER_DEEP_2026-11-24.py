#!/usr/bin/env python3
"""TP_RUNNER_DEEP_2026-11-24 — runner exit after the TP, deep grid (research only).
Operator 11-24: "5 × ATR sounds too much, there must be a division depending on the ATR number" → widths that depend on the ATR level:
k × ATR (15m / 5m / 1m ATR14 at the touch) with % caps, per-ATR-tercile k (picked on DESIGN), fixed %, pure time holds (no trail), and
the WILLY-RUNNER-style floor (TP − nothing; +0.10 % net over entry = break-even floor). Vectorised walk on 1m bars after the touch
(stop = best-so-far × (1 ± w) checked on the NEXT bar; gap fills at the open); 24 h cap. Runner Δ = (exit − TP) pp of entry − 0.13."""
import os, sys, importlib, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
T = importlib.import_module("TP_HOLD_2026-11-24"); C = T.C
OUT = T.OUT; MIN, HR = 60_000, 3_600_000


def atr_tf(pair, Tm, tf):
    a = C.k1m(pair, Tm - 16 * tf * MIN - 2 * HR, Tm)
    if a is None or len(a["t"]) < 15 * tf:
        return np.nan
    b = a["t"] // (tf * MIN)
    df = pd.DataFrame(dict(b=b, h=a["h"], l=a["l"], c=a["c"])).groupby("b").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    pc = df.c.shift(1)
    tr = np.maximum(df.h - df.l, np.maximum((df.h - pc).abs(), (df.l - pc).abs())).dropna()
    return float(tr.iloc[-14:].mean() / df.c.iloc[-1] * 100) if len(tr) >= 14 else np.nan


def path(pair, t_touch):
    T0 = t_touch // MIN * MIN
    a = C.k1m(pair, T0, T0 + 24 * HR + MIN)
    if a is None or len(a["t"]) < 2:
        return None
    g = C.grid(a, T0, min(T0 + 24 * HR + MIN, int(a["t"][-1]) + MIN))
    return {k: g[k][1:] for k in ("o", "h", "l", "c")}          # bars AFTER the touch minute


def exit_trail(p, tp, d, w, floor_px=None):
    """vectorised: best = running extreme incl. the TP; stop for bar i uses best up to bar i−1."""
    if d < 0:
        best = np.minimum.accumulate(np.concatenate([[tp], p["l"]]))[:-1]
        st = best * (1 + w / 100)
        if floor_px is not None: st = np.minimum(st, floor_px)
        hit = np.nonzero(p["h"] >= st)[0]
        return max(st[hit[0]], p["o"][hit[0]]) if len(hit) else p["c"][-1]
    best = np.maximum.accumulate(np.concatenate([[tp], p["h"]]))[:-1]
    st = best * (1 - w / 100)
    if floor_px is not None: st = np.maximum(st, floor_px)
    hit = np.nonzero(p["l"] <= st)[0]
    return min(st[hit[0]], p["o"][hit[0]]) if len(hit) else p["c"][-1]


def exit_time(p, h_hours):
    k = min(len(p["c"]), int(h_hours * 60)) - 1
    return p["c"][k]


KS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
CAPS = (None, 3.0, 5.0, 8.0)
FIX = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0)
HOLDS = (1, 2, 4, 8, 24)


def rows_for(df):
    out = []
    for r in df.itertuples():
        p = path(r.pair, int(r.t_touch))
        if p is None: continue
        d, tp, E = int(r.dirn), float(r.tp_px), float(r.E)
        dd = lambda px: d * (px / tp - 1) * 100 * (tp / E) - T.COST_PP
        o = dict(sleeve=r.sleeve, pair=r.pair, t_touch=int(r.t_touch), period=getattr(r, "period", "REAL"))
        Tm = int(r.t_touch) // MIN * MIN
        atr = {15: float(r.atr15) if np.isfinite(r.atr15) else atr_tf(r.pair, Tm, 15), 5: atr_tf(r.pair, Tm, 5), 1: atr_tf(r.pair, Tm, 1)}
        o.update(atr15=atr[15], atr5=atr[5], atr1=atr[1])
        be = E * (1 + d * 0.10 / 100)                              # WILLY-runner floor: +0.10 % net over entry
        for w in FIX:
            o[f"fix{w:g}"] = dd(exit_trail(p, tp, d, w))
        for tf in (15, 5, 1):
            a = atr[tf]
            for k in KS:
                for cap in CAPS:
                    nm = f"a{tf}k{k:g}" + (f"c{cap:g}" if cap else "")
                    if not np.isfinite(a) or a <= 0: o[nm] = np.nan; continue
                    w = k * a if cap is None else min(k * a, cap)
                    o[nm] = dd(exit_trail(p, tp, d, w))
        for k in (1.0, 2.0, 3.0):                                  # WILLY-runner style (k × 15m ATR, BE floor)
            a = atr[15]
            o[f"wr{k:g}"] = dd(exit_trail(p, tp, d, k * a, be)) if np.isfinite(a) else np.nan
        for h in HOLDS:
            o[f"hold{h}h"] = dd(exit_time(p, h))
        out.append(o)
    return pd.DataFrame(out)


def half(t):
    ts = pd.to_datetime(t, unit="ms")
    return ts.dt.year.astype(str) + "H" + ((ts.dt.month > 6) + 1).astype(str)


def ci(h, col, rng, n=1000):
    x = h.assign(day=pd.to_datetime(h.t_touch, unit="ms").dt.floor("D")).groupby("day")[col].sum()
    bs = [x.sample(len(x), replace=True, random_state=int(rng.integers(1e9))).sum() / len(h) for _ in range(n)]
    return np.quantile(bs, .025), np.quantile(bs, .975)


def main():
    B = pd.read_csv(os.path.join(OUT, "bt_tp_hits.csv")); R = pd.read_csv(os.path.join(OUT, "real_tp_hits.csv"))
    Db = rows_for(B); Db.to_csv(os.path.join(OUT, "deep_bt.csv"), index=False)
    Dr = rows_for(R); Dr.to_csv(os.path.join(OUT, "deep_real.csv"), index=False)
    cols = [c for c in Db.columns if c.startswith(("fix", "a1", "a5", "a15", "wr", "hold"))]
    rng = np.random.default_rng(11)
    for s, g in Db.groupby("sleeve"):
        des, hol = g[g.period == "DESIGN"], g[g.period == "HOLDOUT"]
        tab = pd.DataFrame({c: dict(DES=des[c].mean(), HOL=hol[c].mean(), ALL=g[c].mean(), both=int(des[c].mean() > 0) + int(hol[c].mean() > 0),
                                    halves_pos=int((g.groupby(half(g.t_touch))[c].mean() > 0).sum()), n_halves=g.groupby(half(g.t_touch)).ngroups,
                                    REAL=Dr[Dr.sleeve == s][c].mean() if (Dr.sleeve == s).any() else np.nan) for c in cols}).T
        tab.to_csv(os.path.join(OUT, f"deep_grid_{s}.csv"))
        print(f"\n######## {s}  DESIGN n{len(des)} · HOLDOUT n{len(hol)} · REAL n{int((Dr.sleeve == s).sum())}")
        print("top by ALL (with DESIGN / HOLDOUT / half-years positive):")
        print(tab.sort_values("ALL", ascending=False).head(12).round(2).to_string())
        print("time holds:"); print(tab.loc[[f"hold{h}h" for h in HOLDS]].round(2).to_string())
        # per-ATR-tercile best k (15m ATR, uncapped) picked on DESIGN, read on HOLDOUT
        if len(des) >= 30:
            q = des.atr15.quantile([1 / 3, 2 / 3]).values
            lab = lambda x: np.where(x <= q[0], "lowATR", np.where(x <= q[1], "midATR", "highATR"))
            ks = [f"a15k{k:g}" for k in KS] + [f"fix{w:g}" for w in FIX]
            print(f"per-ATR-tercile pick (15m ATR cuts {q[0]:.2f} / {q[1]:.2f} %):")
            hol_sum = 0.0
            for b in ("lowATR", "midATR", "highATR"):
                dsub, hsub = des[lab(des.atr15) == b], hol[lab(hol.atr15) == b]
                best = max(ks, key=lambda c: dsub[c].mean())
                hol_sum += hsub[best].sum()
                print(f"   {b}: DESIGN n{len(dsub)} best {best} {dsub[best].mean():+.2f} → HOLDOUT n{len(hsub)} {hsub[best].mean():+.2f}")
            print(f"   tercile rule on HOLDOUT: avg {hol_sum / max(len(hol), 1):+.2f} pp per TP hit")
        best_des = tab["DES"].idxmax()
        lo, hi = ci(hol, best_des, rng)
        print(f"▶ DESIGN pick {best_des}: DESIGN {tab.loc[best_des, 'DES']:+.2f} → HOLDOUT {tab.loc[best_des, 'HOL']:+.2f} (CI {lo:+.2f}…{hi:+.2f}) · halves + {int(tab.loc[best_des, 'halves_pos'])}/{int(tab.loc[best_des, 'n_halves'])} · REAL {tab.loc[best_des, 'REAL']:+.2f}")
        hy = g.groupby(half(g.t_touch))[[best_des, "hold24h", "fix5", "a15k1"]].mean().round(2)
        print(hy.assign(n=g.groupby(half(g.t_touch)).size()).to_string())


if __name__ == "__main__":
    main()
