#!/usr/bin/env python3
"""ML_DOWNTREND_2D (2026-10-16) — the 7 pre-registered THEORY pairs (reports/ML_DOWNTREND_2D_PREREG.txt), run before the screen.
    venv/bin/python -W ignore scripts/study_ml_dt2d_theory.py      → reports/ML_DT2D_theory.txt
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_yr6wl_momentum_common import day_boot, day_boot_delta, breakeven_wr   # noqa: E402

ERAS = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
NSHUF = 2000
RNG = np.random.default_rng(7)
D = pd.read_pickle(os.path.join(ROOT, "reports/ML_DT2D_features.pkl"))
D["btc1h"] = pd.to_numeric(D.entry_btc_1h_slope, errors="coerce").fillna(D.k_btc_slope1h)
n = lambda c: pd.to_numeric(D[c], errors="coerce")
VMED = float(n("entry_pair_volume_ratio")[(D.block == ERAS[0]) & (D.DT == 1)].median())
BAD = {   # True = the pre-registered BAD side; NaN where X unscored
    "T1 DEPTH g1h ≤ −3 %": D.g1h_z <= -3.0,
    "T2 NARROWING Δg1h 4h ≤ 0 (widening)": n("k_g1h_chg4h") <= 0,
    "T3 RECLAIM price ≤ 1h EMA200": n("k_px_e200") <= 0,
    "T4 BTC 1h slope ≤ 0": D.btc1h <= 0,
    "T5 COIN 4h EMA20 ≤ EMA50": n("k_gap4h") <= 0,
    "T6 ADX falling (adx − prev ≤ 0)": n("d_adx_dir") <= 0,
    f"T7 VOLUME pair vol ratio ≤ {VMED:.3f} (2025-zone median)": n("entry_pair_volume_ratio") <= VMED,
}
XC = {"T1": "g1h_z", "T2": "k_g1h_chg4h", "T3": "k_px_e200", "T4": "btc1h", "T5": "k_gap4h", "T6": "d_adx_dir",
      "T7": "entry_pair_volume_ratio"}
L = []


def P(s=""):
    print(s); L.append(s)


def block_null_p(y, bad, day):
    """one-sided p that mean(bad) − mean(good) ≤ observed under day-block shuffles of y (time order)."""
    u, first = np.unique(day, return_index=True)
    blocks = [np.where(day == u[i])[0] for i in np.argsort(first)]
    obs = y[bad].mean() - y[~bad].mean()
    cnt = 0
    for _ in range(NSHUF):
        perm = RNG.permutation(len(blocks))
        ys = np.concatenate([y[blocks[j]] for j in perm])
        cnt += (ys[bad].mean() - ys[~bad].mean()) <= obs
    return (cnt + 1) / (NSHUF + 1)


def line(d, ycol, bad):
    z = d[(d.DT == 1)]
    b = bad.loc[z.index]
    ok = b.notna()
    z, b = z[ok], b[ok].astype(bool)
    y = z[ycol].astype(float).values
    if (b.sum() < 2) or ((~b).sum() < 2):
        return dict(nb=int(b.sum()), ng=int((~b).sum()), txt="too few")
    dl, lo, hi, pp = day_boot_delta(y, b.values, z.day.values, n=2000)
    rest = d[~d.index.isin(z.index[b.values]) & d.DT.notna()]
    return dict(nb=int(b.sum()), ng=int((~b).sum()), db=z.day[b].nunique(), mb=y[b.values].mean(), mg=y[~b.values].mean(),
                wb=(y[b.values] > 0).mean(), wg=(y[~b.values] > 0).mean(), dl=dl, lo=lo, hi=hi, pbad=1 - pp,
                mrest=rest[ycol].astype(float).mean(), unsc=int((~ok).sum()), y=y, b=b.values, day=z.day.values)


P("# ML_DOWNTREND_2D — pre-registered THEORY pairs (inside the DT zone: BAD side vs GOOD side)")
P(f"T7 cut frozen from the 2025 replay zone (label-blind): {VMED:.4f}\n")
P("| hypothesis | block | BAD n·d·WR·avg | GOOD n·WR·avg | Δ bad−good [day-boot 95 %] | P(Δ<0) | DT∧BAD vs rest avg | X unscored |")
P("|---|---|---|---|---|---|---|---|")
summ = []
for h, bad in BAD.items():
    signs = {}
    for b in ERAS + ["master as-traded", "master stack_keep", "fresh"]:
        if b in ERAS:
            d, yc = D[D.block == b], "y"
        elif b == "fresh":
            d, yc = D[D.block == "fresh"], "y"
        else:
            d = D[D.block == "master"]
            yc = "y" if "as-traded" in b else "y_stack"
            if "stack" in b:
                d = d[d.stack_keep.astype(bool)]
        r = line(d, yc, bad)
        if "dl" not in r:
            P(f"| {h} | {b} | {r['nb']} | {r['ng']} | too few | | | |"); continue
        signs[b] = np.sign(r["dl"])
        extra = ""
        if b == ERAS[0]:
            r["pnull"] = block_null_p(r["y"], r["b"], r["day"])
            extra = f" · block-null p {r['pnull']:.4f}"
            p25 = r["pnull"]
        P(f"| {h} | {b} | {r['nb']}·{r['db']}·{r['wb'] * 100:.0f} %·{r['mb']:+.3f} | {r['ng']}·{r['wg'] * 100:.0f} %·{r['mg']:+.3f} | "
          f"{r['dl']:+.3f} [{r['lo']:+.2f},{r['hi']:+.2f}]{extra} | {r['pbad']:.2f} | {r['mb']:+.3f} vs {r['mrest']:+.3f} | {r['unsc']} |")
    allrep = all(signs.get(e, 0) < 0 for e in ERAS)
    real = all(signs.get(b, 0) < 0 for b in ["master as-traded", "fresh"])
    summ.append((h, allrep, p25, real, signs))
P("\n## verdict per hypothesis (supported = Δ<0 in all 3 eras ∧ 2025 block-null p < 0.0071 (Bonferroni 7) ∧ master + fresh Δ<0)")
for h, allrep, p25, real, s in summ:
    sg = " ".join(f"{k[:9]}:{'−' if v < 0 else '+'}" for k, v in s.items())
    ok = allrep and p25 < 0.05 / 7 and real
    P(f"- {h}: eras all bad-worse {allrep} · 2025 p {p25:.4f} · master & fresh bad-worse {real} → {'SUPPORTED' if ok else 'not supported'}  [{sg}]")
with open(os.path.join(ROOT, "reports/ML_DT2D_theory.txt"), "w") as f:
    f.write("\n".join(L) + "\n")
