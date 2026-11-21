#!/usr/bin/env python3
"""EXH_THEORY_EXIT — EVAL (PREREG reports/EXH_THEORY_EXIT_2026-11-21_PREREG.md). Research only, BACKTEST.
Nets = EXH_EXIT2 nets (r + funding − entry slip − 0.05 − exit (slip + 0.05 taker | 0.02 maker)); flat 0.20 stress. DESIGN pick per family,
HOLDOUT read once against the 6 pre-committed gates. → reports/EXH_THEORY_EXIT_2026-11-21_data/eval.pkl + printed tables."""
import os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "EXH_THEORY_EXIT_2026-11-21_data")
SPLIT = pd.Timestamp("2025-07-01").value // 10**6
X_EQ = 0.61


def nets(slip, r, k, f, flat=None):
    s = slip[:, None] if flat is None else flat
    tk = k != 0
    return r + f - s - 0.05 - np.where(tk, s + 0.05, 0.02)


def boot(d, day, B=3000, seed=7, block=10):
    days = np.sort(np.unique(day)); bidx = np.searchsorted(days, day) // block
    u, inv = np.unique(bidx, return_inverse=True)
    s = np.bincount(inv, d, len(u)); n = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(u), (B, len(u)))
    m = s[ix].sum(1) / n[ix].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def dcr(net, xm):
    s = pd.Series(net * X_EQ / 100).groupby(pd.to_datetime(xm, unit="ms").normalize()).sum()
    days = pd.date_range(s.index.min(), s.index.max(), freq="D")
    r = s.reindex(days, fill_value=0.0).values
    return ((np.prod(1 + r)) ** (1 / len(r)) - 1) * 100


def main():
    X = pd.read_pickle(os.path.join(R, "reports", "EXH_EXIT2_data", "rows_EXH.pkl"))
    Z = dict(np.load(os.path.join(D, "sim_BT.npz"), allow_pickle=False))
    assert (Z["xi"] == X.xi.values).all()
    ok = Z["ok"]; X = X[ok].reset_index(drop=True); Z = {k: (v[ok] if v.ndim and len(v) == len(ok) else v) for k, v in Z.items()}
    names = list(Z["names"]); slip = X.slip.values.astype(float)
    N = nets(slip, Z["r"].astype(float), Z["k"], Z["f"].astype(float))
    N2 = nets(slip, Z["r"].astype(float), Z["k"], Z["f"].astype(float), flat=0.20)
    per = np.where(X.entry_ms.values < SPLIT, "DESIGN", "HOLDOUT")
    day = pd.to_datetime(X.entry_ms.values, unit="ms").normalize().values.astype("int64") // 86_400_000 // 10**0
    rows = []
    for j, nm in enumerate(names):
        for p in ("DESIGN", "HOLDOUT"):
            m = per == p
            d = N[m, j] - N[m, 0]; d2 = N2[m, j] - N2[m, 0]
            lo, hi = boot(d, (X.entry_ms.values[m] // 86_400_000))
            hv = X.half.values[m]
            hp = pd.Series(d).groupby(hv).sum()
            win = N[m, 0] > 0
            pair_s = pd.Series(d).groupby(X.pair.values[m]).sum(); day_s = pd.Series(d).groupby(X.entry_ms.values[m] // 86_400_000).sum()
            tot = d.sum()
            rows.append(dict(cell=nm, fam=nm.split("_")[0] if nm != "REF" else "REF", period=p, N=int(m.sum()), changed=int((np.abs(d) > 1e-9).sum()),
                             WR=(N[m, j] > 0).mean() * 100, avg=N[m, j].mean(), d_mean=d.mean(), ci_lo=lo, ci_hi=hi, d_sum=tot,
                             halves_pos=f"{int((hp > 0).sum())}/{len(hp)}", halves_pos_frac=(hp > 0).mean(), d_mean_020=d2.mean(),
                             top_pair=pair_s.idxmax() if tot > 0 else "", top_pair_share=(pair_s.max() / tot * 100) if tot > 0 else np.nan,
                             top_day_share=(day_s.max() / tot * 100) if tot > 0 else np.nan,
                             saved=d[~win & (d > 0)].sum(), cut=-d[win & (d < 0)].sum(),
                             dcr_ref=dcr(N[m, 0], Z["xm"][m, 0]), dcr=dcr(N[m, j], Z["xm"][m, j]),
                             tp=(Z["k"][m, j] == 0).mean() * 100, sl=(Z["k"][m, j] == 1).mean() * 100, rule=(Z["k"][m, j] == 5).mean() * 100))
    T = pd.DataFrame(rows)
    T.to_pickle(os.path.join(D, "eval.pkl"))
    pd.set_option("display.width", 260)
    cols = ["cell", "N", "changed", "WR", "avg", "d_mean", "ci_lo", "ci_hi", "halves_pos", "d_mean_020", "saved", "cut", "dcr_ref", "dcr", "tp", "sl", "rule"]
    for p in ("DESIGN", "HOLDOUT"):
        print(f"\n===== {p}"); print(T[T.period == p][cols].round(2).to_string(index=False))
    # picks + gates
    print("\n===== PICKS (DESIGN-best per family) → HOLDOUT gates")
    for fam in ("A", "N", "B", "C", "F"):
        dsg = T[(T.period == "DESIGN") & (T.fam == fam)].sort_values("d_mean", ascending=False).iloc[0]
        h = T[(T.period == "HOLDOUT") & (T.cell == dsg.cell)].iloc[0]
        g = [h.d_mean > 0 and h.ci_lo > 0, h.halves_pos_frac >= 2 / 3, h.d_mean_020 > 0,
             (h.top_pair_share < 50 and h.top_day_share < 50) if h.d_sum > 0 else False, h.saved > h.cut, h.dcr >= h.dcr_ref]
        print(f"{fam}: pick {dsg.cell} (DESIGN Δ {dsg.d_mean:+.3f}) · HOLDOUT Δ {h.d_mean:+.3f} [{h.ci_lo:+.3f}, {h.ci_hi:+.3f}] halves {h.halves_pos} "
              f"@0.20 {h.d_mean_020:+.3f} pair {h.top_pair_share:.0f}% day {h.top_day_share:.0f}% saved {h.saved:.0f} cut {h.cut:.0f} "
              f"DCR {h.dcr_ref:.2f}→{h.dcr:.2f} · gates {''.join('✓' if x else '✗' for x in g)} → {'PASS' if all(g) else 'FAIL'}")


if __name__ == "__main__":
    main()
