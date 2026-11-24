#!/usr/bin/env python3
"""TP_RUNNER_DIVE2_2026-11-24 — round 2 (research only):
 A. HT rule (TP +6 → runner 4 × 15m ATR, floor +5 net) on BIGGER entry sets: every HT setup in BASELINE_TRADES_WILLY_HT (no today's
    filters) and the plain WILLY setups (same entry family) — walked from E: stop −4 net · TP +6 net · 24 h, runner after the touch.
 B. EXH runner hold cap after the touch {8, 12, 24, 48 h} × k {4, 5} + a break-even ratchet (floor = entry once best ≥ TP + 1 × ATR);
    funding included.
 C. Revert-gate power: bootstrap the per-hit Δ (all halves) → P(Σ first N hits < 0) for N = 10 / 20 / 30 / 50, i.e. how often a gate of
    that size reverts a rule whose long-run edge is the backtest's."""
import os, sys, importlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DV = importlib.import_module("TP_RUNNER_DIVE_2026-11-24"); DP = DV.DP; T = DV.T; C = T.C
OUT = T.OUT; MIN, HR = 60_000, 3_600_000


def ht_touches(df):
    need = set()
    for r in df.itertuples():
        for ds in C.days_of(int(r.entry_ms) - 6 * HR, int(r.entry_ms) + 50 * HR):
            need.add((r.pair, ds))
    importlib.import_module("TP_HOLD_2026-11-24")._fetch_parallel(sorted(need))
    rows = []
    for r in df.itertuples():
        E = float(r.E); te = int(r.entry_ms)
        tp = E * (100 + 6.0 + 0.05) / (100 - 0.02); sl = E * (100 - 4.0 + 0.05) / (100 - 0.05)
        m0 = te // MIN * MIN
        a = C.k1m(r.pair, m0, m0 + 25 * HR)
        if a is None or len(a["t"]) < 30: continue
        g = C.grid(a, m0, min(m0 + 25 * HR, int(a["t"][-1]) + MIN))
        for i in range(1, min(len(g["t"]), 24 * 60 + 1)):
            if g["l"][i] <= sl: break
            if g["h"][i] >= tp:
                tt = int(g["t"][i])
                rows.append(dict(pair=r.pair, t_touch=tt, E=E, tp_px=tp, atr15=T.atr15_pct(r.pair, tt), blk=("DESIGN" if te < T.SPLIT else "HOLDOUT"),
                                 sleeve=r.sleeve))
                break
    return pd.DataFrame(rows)


def gate_power(x, Ns=(10, 20, 30, 50), n=20000, seed=5):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    return {N: round(float((rng.choice(x, (n, N)).sum(1) < 0).mean()), 2) for N in Ns}


def main():
    rng = np.random.default_rng(31)
    # ───── A. HT rule on the bigger entry sets ─────
    BT = pd.read_csv(os.path.join(ROOT, "reports", "BASELINE_TRADES_WILLY_HT_2026-11-18.csv"), low_memory=False)
    BT = BT.dropna(subset=["E", "entry_ms"])
    print("entry sets:", BT.sleeve.value_counts().to_dict())
    Hh = ht_touches(BT)
    Hh["blk"] = np.where(Hh.t_touch < T.SPLIT, "DESIGN", "HOLDOUT")
    specs = {"4x|+5": (4, 5.0), "4x|+4": (4, 4.0), "5x|+5": (5, 5.0), "3x|+5": (3, 5.0), "4x|stop−4": (4, -4.0)}
    for s, g in Hh.groupby("sleeve"):
        O = DV.eval_rows(g.rename(columns={}), 1, specs, tp_net=6.0)
        DV.report(f"A. {s} — every setup in the 2-year baseline (no today's filters), TP +6 hits", O, list(specs), rng)
        print("   anatomy 4x|+5:", DV.anatomy(O, "4x|+5"))
        O.to_csv(os.path.join(OUT, f"dive2_{s}.csv"), index=False)
    # ───── B. EXH hold cap + BE ratchet ─────
    X = DV.cohort("EXHAUSTION_SHORT")
    rows = []
    for r in X.itertuples():
        p = DP.path(r.pair, int(r.t_touch))
        if p is None or not np.isfinite(r.atr15): continue
        E, tp, atr = float(r.E), float(r.tp_px), float(r.atr15)
        sl = E * (1 + min(1.5 * atr, 6) / 100); tpn = -(tp / E - 1) * 100 - 0.07
        o = dict(pair=r.pair, t=int(r.t_touch), blk=r.blk)
        for k in (4, 5):
            for H in (8, 12, 24, 48):
                q = {kk: v[: H * 60] for kk, v in p.items()} if H <= 24 else None
                if H == 48:
                    a = C.k1m(r.pair, int(r.t_touch) // MIN * MIN, int(r.t_touch) // MIN * MIN + 48 * HR + MIN)
                    if a is None: continue
                    gg = C.grid(a, int(r.t_touch) // MIN * MIN, min(int(r.t_touch) // MIN * MIN + 48 * HR + MIN, int(a["t"][-1]) + MIN))
                    q = {kk: gg[kk][1:] for kk in ("o", "h", "l", "c")}
                px, i, why = DV.trail_ix(q, tp, -1, k * atr, sl)
                px = px * (1 + 0.001)
                fund = DV.fund_between(r.pair, int(r.t_touch), int(r.t_touch) + (i + 1) * MIN, E, px, -1)
                o[f"{k}x|{H}h"] = (1 - px / E) * 100 - 0.10 + fund - tpn
                o[f"{k}x|{H}h_fund"] = fund
            # BE ratchet (24 h): floor = the stop until best ≤ TP × (1 − atr %), then the entry price
            best = np.minimum.accumulate(np.concatenate([[tp], p["l"]]))[:-1]
            fl = np.where(best <= tp * (1 - atr / 100), E * (1 - 0.001), sl)
            st = np.minimum(best * (1 + k * atr / 100), fl)
            hit = np.nonzero(p["h"] >= st)[0]
            i = hit[0] if len(hit) else len(p["c"]) - 1
            px = (max(st[i], p["o"][i]) if len(hit) else p["c"][-1]) * (1 + 0.001)
            fund = DV.fund_between(r.pair, int(r.t_touch), int(r.t_touch) + (i + 1) * MIN, E, px, -1)
            o[f"{k}x|24h+BE"] = (1 - px / E) * 100 - 0.10 + fund - tpn; o[f"{k}x|24h+BE_fund"] = fund
        rows.append(o)
    OX = pd.DataFrame(rows); ts = pd.to_datetime(OX.t, unit="ms"); OX["half"] = ts.dt.year.astype(str) + "-H" + ((ts.dt.month > 6) + 1).astype(str)
    names = [c for c in OX.columns if "|" in c and not c.endswith("_fund")]
    DV.report("B. EXH runner — hold cap after the touch + break-even ratchet (stop kept, funding incl.)", OX, names, rng)
    OX.to_csv(os.path.join(OUT, "dive2_EXH.csv"), index=False)
    # ───── C. revert-gate power ─────
    OH = pd.read_csv(os.path.join(OUT, "dive_HT.csv")); OE = pd.read_csv(os.path.join(OUT, "dive_EXH.csv"))
    print("\nC. P(a gate of N TP hits shows Σ < 0 although the long-run edge = the backtest's) — the false-revert risk")
    print("   HT 4x|+5 (today's-filter backtest):", gate_power(OH["4x|+5"]), "· mean", round(OH["4x|+5"].mean(), 2))
    print("   EXH 4x|stop 24h:", gate_power(OE["4x|stop"]), "· mean", round(OE["4x|stop"].mean(), 2))
    for c in ("4x|12h", "4x|24h+BE", "5x|24h"):
        if c in OX: print(f"   EXH {c}:", gate_power(OX[c].dropna()), "· mean", round(OX[c].mean(), 2))


if __name__ == "__main__":
    main()
