#!/usr/bin/env python3
"""ML REGIME W/L 2026-10-26 — same-calendar replay check (refute-only): does today's engine replay (yr6, 3 seeds) lose in the same era
windows the master loses in? + master KEEP fills matched / not matched by a replay fill (same pair, ±10 min, any seed), fills ≤ 2026-10-11
12:00 (replay end). Out: reports/ML_REGIME_WL_2026-10-26_replay_calendar.csv, _replay_match.csv.
    venv/bin/python -W ignore scripts/study_ml_regime_wl_replaycal.py
"""
import numpy as np, pandas as pd
from study_ml_regime_wl_deep import D, RP, PFX, st

SP = [("G0 BASE", "2026-06-18", "2026-07-11"), ("G1 B1", "2026-07-11", "2026-08-01"), ("G2 B2", "2026-08-01", "2026-08-11"),
      ("G3 B3", "2026-08-11", "2026-08-24 21:00"), ("G4 B4-B11", "2026-08-24 21:00", "2026-09-21"), ("G5 B12-B16", "2026-09-21", "2026-10-05"),
      ("G6 B17-B24", "2026-10-05", "2026-10-11 12:00")]
R = RP[RP.ts >= "2026-06-18"].copy()
R["ng"] = (pd.to_numeric(R.peak_pnl, errors="coerce") < 0.10) & (R.y < 0)
K = D[D.y.notna() & (D.era_group != "B25 (prov.)") & (D.ts < "2026-10-11 12:00")].copy()
rows = []
for g, a, b in SP:
    r = R[(R.ts >= a) & (R.ts < b)]; k = K[K.era_group == g]
    rows.append({"group": g, "window": f"{a} → {b}", "master KEEP": st(k.y), "master NG-loser %": k.ng_loser_keep.mean() * 100,
                 "replay (3 seeds)": st(r.y), "replay days": r.day.nunique(), "replay NG-loser %": r.ng.mean() * 100,
                 "replay per-seed avg": " / ".join(f"{v:+.3f}" for v in r.groupby("seed").y.mean())})
C = pd.DataFrame(rows); C.to_csv(PFX + "_replay_calendar.csv", index=False)
m = []
for _, f in K.iterrows():
    q = R[(R.pair == f.pair) & ((R.ts - f.ts).abs() <= pd.Timedelta(minutes=10))]
    m.append((len(q) > 0, q.y.mean() if len(q) else np.nan))
K["matched"] = [a for a, b in m]; K["replay_y"] = [b for a, b in m]
K["side"] = np.where(K.era_group.isin(["G0 BASE", "G1 B1", "G2 B2", "G3 B3"]), "good G0–G3",
                     np.where(K.era_group.isin(["G5 B12-B16", "G6 B17-B24"]), "bad G5–G6 (to 10-11)", "G4"))
M = []
for s, k in K.groupby("side"):
    mm = k[k.matched]
    M.append({"side": s, "KEEP": st(k.y), "matched (replay also took it)": st(mm.y), "same fills priced by replay": f"{mm.replay_y.mean():+.3f}",
              "unmatched": st(k.y[~k.matched]), "NG-loser % matched / unmatched": f"{mm.ng_loser_keep.mean() * 100:.0f} / {k[~k.matched].ng_loser_keep.mean() * 100:.0f}"})
pd.DataFrame(M).to_csv(PFX + "_replay_match.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(C.to_string(index=False)); print(pd.DataFrame(M).to_string(index=False))
