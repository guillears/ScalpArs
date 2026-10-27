#!/usr/bin/env python3
"""FADE REGIME W/L study 2026-10-26 — the frozen observe-line candidates in depth (bar · per batch · leave-one-batch-out · interaction
with the regime · backtest per seed and per period · concentration). Thresholds below are FROZEN in the report (cohort ≥ 2026-10-27).
Outputs reports/FADE_REGIME_WL_2026-10-26_{obs_lines,obs_per_batch,obs_rows}.csv + _deep.txt
    venv/bin/python -W ignore scripts/study_fade_regime_wl_deep.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_fade_regime_wl_common as C    # noqa: E402

PFX = C.PFX
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a); OUT.append(s); print(s, flush=True)


A = C.load(with_blocked=True)
K = A[A.blk == "MASTER"].reset_index(drop=True); B25 = A[A.blk == "B25"].reset_index(drop=True)
ALL = pd.concat([K, B25]).reset_index(drop=True); BLK = A[A.blk == "MASTER_BLK"].reset_index(drop=True)
Bt = C.load_bt()
BE = C.breakeven(K.pct.values)
n = C.num
late = lambda d: (n(d, "btc_e50_e200_1d") > 0).values
LINES = {
    "OBS-F1 late tape ∧ weak spike (+DI ≤ 48.5 ∧ Ext% ≤ 1.80)":
        lambda d: late(d) & (n(d, "entry_pos_di") <= 48.5).values & (n(d, "entry_dist_from_ema13_pct") <= 1.80).values,
    "OBS-F1b late tape ∧ +DI ≤ 43.6 (PREP-prereg cut)": lambda d: late(d) & (n(d, "entry_pos_di") <= 43.6).values,
    "(interaction check) EARLY tape ∧ weak spike (+DI ≤ 48.5 ∧ Ext% ≤ 1.80)":
        lambda d: (n(d, "btc_e50_e200_1d") <= 0).values & (n(d, "entry_pos_di") <= 48.5).values & (n(d, "entry_dist_from_ema13_pct") <= 1.80).values,
    "(no regime) weak spike (+DI ≤ 48.5 ∧ Ext% ≤ 1.80)":
        lambda d: (n(d, "entry_pos_di") <= 48.5).values & (n(d, "entry_dist_from_ema13_pct") <= 1.80).values,
    "41h HB ∧ BTC trend gap ≥ −0.13": lambda d: (n(d, "hb_gap_zone") == 1).values,
    "pair funding < 0 at entry (P10 — W side?)": lambda d: (n(d, "f_pair") < 0).values,
}
rows, pb, rr = [], [], []
for nm, f in LINES.items():
    zk, zb, za = f(K), f(B25), f(ALL)
    bk = C.bar(K.pct.values[zk], K.day.values[zk], K.pair.values[zk], BE)
    ba = C.bar(ALL.pct.values[za], ALL.day.values[za], ALL.pair.values[za], BE)
    # leave-one-batch-out on ALL: zone avg after dropping each batch
    lobo = []
    for e in ALL.era.unique():
        m = za & (ALL.era.values != e)
        if m.sum() >= 3:
            lobo.append((e, ALL.pct.values[m].mean()))
    worst = max(lobo, key=lambda x: x[1]) if lobo else ("", np.nan)
    r = {"line": nm, "master zone": C.st(K.pct.values[zk]), "master rest": C.st(K.pct.values[~zk]),
         "B25 zone (prov.)": C.st(B25.pct.values[zb]), "ALL zone": C.st(ALL.pct.values[za]), "ALL rest": C.st(ALL.pct.values[~za]),
         "ALL zone days": ba["days"], "ALL zone pairs": len(set(ALL.pair.values[za])),
         "bar master": "PASS" if bk["passes"] else f"fail ({bk['fails']})", "P_neg master": round(bk["P_neg"], 3),
         "bar ALL": "PASS" if ba["passes"] else f"fail ({ba['fails']})", "P_neg ALL": round(ba["P_neg"], 3),
         "top day / pair loss share ALL": f"{ba['top_day_share']:.0%} / {ba['top_pair_share']:.0%} (top-2 pairs {ba['top2_pair_share']:.0%})",
         "leave-one-batch-out: best zone avg (batch dropped)": f"{worst[1]:+.3f} (drop {worst[0]})",
         "Δ$ block ALL in-sample (2×)": -ALL.usd.values[za].sum(), "Δ$ after 30–50 % haircut": f"{-ALL.usd.values[za].sum() * 0.5:+.0f} … {-ALL.usd.values[za].sum() * 0.7:+.0f}",
         "stack-blocked fades in zone (as traded)": C.st(BLK.pct.values[f(BLK)])}
    for sd in (1, 2, 3):
        bb = Bt[Bt.seed == sd].reset_index(drop=True); zz = f(bb)
        r[f"BT s{sd} zone vs rest"] = f"{C.st(bb.pct.values[zz])} vs {C.st(bb.pct.values[~zz])}"
        for per in ("2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"):
            mm = bb.era.values == per
            r[f"BT s{sd} {per} zone"] = C.st(bb.pct.values[zz & mm])
    rows.append(r)
    for era, g in list(K.groupby("era", sort=False)) + [("B25 (prov.)", B25)]:
        z = f(g)
        pb.append({"line": nm, "batch": era, "before": C.st(g.pct), "before $": round(g.usd.sum(), 2), "blocked": C.st(g.pct.values[z]),
                   "blocked $": round(g.usd.values[z].sum(), 2), "after": C.st(g.pct.values[~z]), "after $": round(g.usd.values[~z].sum(), 2)})
    for _, x in ALL[za].iterrows():
        rr.append({"line": nm, "opened_at": x.opened_at, "batch": x.era, "pair": x.pair, "pct": x.pct, "usd": x.usd,
                   "pos_di": n(ALL, "entry_pos_di")[_], "ext13": n(ALL, "entry_dist_from_ema13_pct")[_], "btc_e50_e200_1d": x.btc_e50_e200_1d,
                   "regime": x.entry_btc_regime, "close_reason": x.close_reason})
O = pd.DataFrame(rows); O.to_csv(PFX + "_obs_lines.csv", index=False)
PB = pd.DataFrame(pb); PB.to_csv(PFX + "_obs_per_batch.csv", index=False)
pd.DataFrame(rr).to_csv(PFX + "_obs_rows.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 60)
P(f"breakeven WR {BE:.1f} %")
P(O.T.to_string())
P(PB[PB.line.str.startswith("OBS-F1 ")].to_string())
# late-tape fills by day (episode / window read)
L = ALL[late(ALL)].copy()
P("late-tape days:", L.day.nunique(), "· first", L.day.min(), "· last", L.day.max())
open(PFX + "_deep.txt", "w").write("\n".join(OUT) + "\n")
