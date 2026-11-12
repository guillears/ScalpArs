#!/usr/bin/env python3
"""ML RUNNER-MARKET study — MASTER (2026-11-11, operator: "yes run it on master").
QUESTION  Every winner-side exit (second chance, add-on, wider trail) wins in B17–B25 and loses in BASE–B16. Is there a variable, readable
          while the trade is open, that tells a RUNNER market (winners keep going after today's exit) from a NORMAL one?
TARGET    per fill: y = second-chance (d 0.3) − today's exit, % pts, on the 95 fills whose exit is a runner / ladder exit (the only fills
          the rule touches) — reports/ML_SECOND_CHANCE_2026-11-11_variants.csv.
VARIABLES every numeric entry_* stamp of the master row (CLAUDE.md W/L = every variable) + IN-TRADE values known at today's exit trigger:
          min_to_arm (entry → +0.40), peak_at_exit, BTC % since entry at the trigger, BTC % over the 30 min before the trigger, BTC 5m EMA20
          slope sign at the trigger, pair minutes held.
TESTS     per variable: sign split (if it crosses 0), median split, top / bottom tercile → zone vs rest; zone N ≥ 15 on ≥ 8 days. Score =
          mean y in zone − mean y outside. Kept for reading only if: day-block 95 % CI of zone mean y > 0, zone mean > 0 in ≥ 3 era groups
          (groups with ≥ 3 zone fills) and ≤ 0 overall outside. NULL: 400 day-block shuffles of y → the max score over ALL tests each time;
          the best real score's rank = the family-wise p. Coverage table + blind spots printed first.
Writes reports/ML_RUNNER_REGIME_2026-11-11.txt + _tests.csv"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S, study_ml_exit_macro_1111_report as R, scout_ml_matched_refusals as MM  # noqa
OUT = os.path.join(S.ROOT, "reports", "ML_RUNNER_REGIME_2026-11-11")
RNG = np.random.default_rng(20261111)


def intrade(C, X):
    rows = []
    with MM.replica_with_topup() as RP:
        B = S._btc_frames(RP)
        ot, c, e20 = B["ot"], B["c"], B["e20"]
        kn = ot + 300_000
        def btc_at(ms):
            j = np.searchsorted(kn, ms, side="right") - 1
            return (c[j], e20[j] - e20[j - 1]) if j > 0 else (np.nan, np.nan)
        for r in C.itertuples():
            fx = dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in), atr=None, rsi_entry=None)
            tx = int(X[r.key])
            P = RP.prep_path(fx, tx - int(r.o_ms) + 120_000)
            t, net = P["t"], P["net"]
            pk = np.maximum.accumulate(np.maximum(net, 0))
            ja = np.flatnonzero(pk >= 0.395)
            b0, _ = btc_at(int(r.o_ms)); bx, sl = btc_at(tx); b30, _ = btc_at(tx - 30 * 60_000)
            rows.append(dict(key=r.key, it_min_to_arm=(t[ja[0]] - r.o_ms) / 60_000 if len(ja) else np.nan,
                             it_peak_at_exit=float(pk[min(np.searchsorted(t, tx), len(t) - 1)]),
                             it_btc_since_entry=(bx / b0 - 1) * 100, it_btc_30m=(bx / b30 - 1) * 100, it_btc_e20_slope=np.sign(sl),
                             it_min_held=(tx - r.o_ms) / 60_000))
    return pd.DataFrame(rows).set_index("key")


def main():
    C, _ = S.cohort()
    V = pd.read_csv(os.path.join(S.ROOT, "reports", "ML_SECOND_CHANCE_2026-11-11_variants.csv"))
    W = V.pivot(index="key", columns="variant", values="pct"); RR = V.pivot(index="key", columns="variant", values="reason")
    X = V.pivot(index="key", columns="variant", values="exit_ms").BASE
    elig = RR.BASE.isin(("RUNNER_TRAIL", "HARD_TP_LADDER", "RH_RUNNER_TRAIL", "RH_HARD_TP_LADDER"))
    C = C.set_index("key").loc[W.index[elig]].reset_index()
    y = (W["SC_d0.3"] - W.BASE).loc[C.key].values
    I = intrade(C, X)
    C = C.join(I, on="key")
    C["grp"] = C.era.map(R.GROUPS); C["day"] = pd.to_datetime(C.o_ms, unit="ms").dt.strftime("%Y-%m-%d")
    cols = [c for c in C.columns if (c.startswith("entry_") or c.startswith("it_"))]
    num = {}
    for c in cols:
        v = pd.to_numeric(C[c], errors="coerce")
        if v.notna().sum() >= 30 and v.nunique() >= 3:
            num[c] = v.values
    L = [f"# ML RUNNER-MARKET study — {len(C)} fills where second-chance changes the exit · mean Δ {y.mean():+.3f} · days {C.day.nunique()}",
         f"## Coverage: {len(num)} variables testable (≥ 30 scored, ≥ 3 values) of {len(cols)} entry_* / in-trade columns",
         "blind spots (untestable): " + ", ".join(sorted(set(cols) - set(num)))[:2000], ""]
    def tests(yv):
        out = []
        for c, v in num.items():
            ok = ~np.isnan(v)
            cuts = []
            if np.nanmin(v) < 0 < np.nanmax(v):
                cuts.append(("sign", 0.0))
            q = np.nanquantile(v, [1 / 3, 0.5, 2 / 3])
            cuts += [("t1", q[0]), ("med", q[1]), ("t2", q[2])]
            for nm, cut in cuts:
                for side in (">", "<="):
                    z = ok & ((v > cut) if side == ">" else (v <= cut))
                    if z.sum() < 15:
                        continue
                    rest = ok & ~z
                    out.append((c, nm, side, cut, z, yv[z].mean() - yv[rest].mean()))
        return out
    T = tests(y)
    rows = []
    for c, nm, side, cut, z, sc in T:
        if C.day[z].nunique() < 8:
            continue
        lo, hi, _ = R.boot_ci(y[z], C.day.values[z], 0.05)
        g = pd.Series(y[z]).groupby(C.grp.values[z]).agg(["mean", "size"])
        g = g[g["size"] >= 3]
        rows.append(dict(var=c, split=nm, side=side, cut=cut, n_zone=int(z.sum()), days=int(C.day[z].nunique()), zone=y[z].mean(),
                         rest=y[~z & ~np.isnan(num[c])].mean(), score=sc, lo95=lo, hi95=hi, groups_pos=int((g["mean"] > 0).sum()), groups=len(g),
                         by_group=g["mean"].round(2).to_dict()))
    D = pd.DataFrame(rows).sort_values("score", ascending=False)
    D.to_csv(OUT + "_tests.csv", index=False)
    # null: max score over all tests under day-block shuffles of y
    days = C.day.values; ud = np.unique(days)
    mx = []
    for _ in range(400):
        perm = dict(zip(ud, RNG.permutation(ud)))
        # move each day's y block to another day's fills (day-block shuffle by re-labelling days, then re-ordering)
        ys = y.copy(); idx = {d: np.flatnonzero(days == d) for d in ud}
        pool = np.concatenate([y[idx[perm[d]]] for d in ud]); RNG.shuffle(pool) if False else None
        order = np.concatenate([idx[d] for d in ud])
        ys[order] = np.resize(pool, len(order))
        mx.append(max(s for *_, s in tests(ys)))
    mx = np.array(mx)
    best = D.score.max()
    L.append(f"## NULL: best real score {best:+.3f} · shuffled max-score median {np.median(mx):+.3f}, 95th {np.quantile(mx, .95):+.3f} · family-wise p {np.mean(mx >= best):.2f}")
    keep = D[(D.lo95 > 0) & (D.groups_pos >= 3) & (D.rest <= 0)]
    L.append(f"## Passing the reading screen (CI > 0, ≥ 3 era groups positive, ≤ 0 outside): {len(keep)}")
    L.append(keep.head(20).round(3).to_string(index=False) if len(keep) else "none")
    L.append("\n## Top 25 by score (all)")
    L.append(D.head(25).round(3).to_string(index=False))
    open(OUT + ".txt", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
