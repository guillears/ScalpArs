#!/usr/bin/env python3
"""ML REGIME W/L study 2026-10-26 — step 5: candidates vs the expectancy bar, before → after per era group, B25 apart, replay
(refute-only), DAY / window units. Thresholds are the pooled KEEP cuts printed in the screens and are FROZEN here.
    venv/bin/python -W ignore scripts/study_ml_regime_wl_candidates.py
"""
import numpy as np
import pandas as pd
from study_ml_regime_wl_deep import D, K, RP, bar, before_after, PFX, st, day_boot   # noqa: F401

n = lambda df, c: pd.to_numeric(df[c], errors="coerce")
CANDS = {
    "R1 HICORR regime OFF (top-20 24 h corr > 0.3164)": lambda d: n(d, "br_corr24h_top20") > 0.3164,
    "R2 LOWRV7 regime OFF (BTC 7 d rv ≤ 0.1122)": lambda d: n(d, "btc_rv_7d") <= 0.1122,
    "R3 BTC 1d EMA20 > EMA50 by > 2.494 % (macro uptrend) OFF": lambda d: n(d, "btc_e20_e50_1d") > 2.494,
    "C1 HICORR ∧ corr15 > 0.365 (regime-conditional)": lambda d: (n(d, "br_corr24h_top20") > 0.3164) & (n(d, "imp_corr15") > 0.365),
    "C2 LOWRV7 ∧ pair EMA20 slope ≤ 0.1667 (regime-conditional)": lambda d: (n(d, "btc_rv_7d") <= 0.1122) & (n(d, "entry_ema20_slope") <= 0.1667),
    "C3 LOWRV7 ∧ Ext% (dist EMA13) ≤ 0.2718 (regime-conditional)": lambda d: (n(d, "btc_rv_7d") <= 0.1122) & (n(d, "entry_dist_from_ema13_pct") <= 0.2718),
}

if __name__ == "__main__":
    B, BA, RPL = [], [], []
    for lab, f in CANDS.items():
        z = f(K).fillna(False).values
        r = bar(K, z, lab)
        rest = K.y[~z]
        r["rest"] = st(rest); r["zone"] = st(K.y[z])
        r["zone share good / bad eras %"] = f"{z[K.era_group.isin(['G0 BASE','G1 B1','G2 B2','G3 B3'])].mean()*100:.0f} / {z[K.era_group.isin(['G5 B12-B16','G6 B17-B24'])].mean()*100:.0f}"
        r["ex-washed zone"] = st(K.y[z & ~K.washed.values])
        zt = f(D[D.y_trad.notna() & (D.era_group != "B25 (prov.)")]).fillna(False).values
        r["TRAD zone"] = st(D[D.y_trad.notna() & (D.era_group != "B25 (prov.)")].y_trad[zt])
        bb = D[D.era_group == "B25 (prov.)"]
        zb = f(bb).fillna(False).values
        r["B25 zone"] = st(bb.y[zb]); r["B25 rest"] = st(bb.y[~zb])
        for rg in ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]:
            q = RP[RP.regime == rg]
            try:
                zz = f(q).fillna(False).values
                r[f"replay {rg} zone / rest"] = f"{st(q.y[zz])} / {st(q.y[~zz])}"
            except KeyError:
                r[f"replay {rg} zone / rest"] = "n/a (not in replay)"
        B.append(r)
        BA += before_after(D, lambda d, f=f: f(d).fillna(False).values, lab)
    Bdf = pd.DataFrame(B); Bdf.to_csv(PFX + "_candidates.csv", index=False)
    BAdf = pd.DataFrame(BA); BAdf.to_csv(PFX + "_before_after.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
    print(Bdf.T.to_string())
    print(BAdf.to_string(index=False))
