#!/usr/bin/env python3
"""📊 Oct-8 research — writes reports/YR5_HALVES_TODAY_STACK_2026-10-08.csv (+ the numbers the .md quotes) from the outputs of
scripts/study_yr5_halves_today.py (scratch yr5h/res.json, F_today.pkl, L.pkl, A.pkl) and scripts/study_yr5_halves_willy.py (willy_priced.csv).
Adds: per-seed $ per half (replay sleeves), the Oct-6 → today waterfall per sleeve (FY $), WILLY backstop / liquidation sensitivity.
Usage: venv/bin/python scripts/study_yr5_halves_report.py"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_yr5_halves_today import SCR, BOOK, SLOT, T0, TS, T1, LEV, boot_day   # noqa: E402

D = os.path.join(SCR, "yr5h")
R = json.load(open(os.path.join(D, "res.json")))
F = pd.read_pickle(os.path.join(D, "F_today.pkl"))
OUT = os.path.join(ROOT, "reports", "YR5_HALVES_TODAY_STACK_2026-10-08.csv")
HALF = {"H1": (T0, TS), "H2": (TS, T1), "FY": (T0, T1)}
SRC = {
    "MOM_LONG_INCL_READMIT": "yr5 replay + heat re-admits (exit-replica priced signals)", "MOM_LONG": "yr5 replay",
    "MOM_LONG_HEAT_READMIT": "HEAT_YR5_SIGNALS_priced (exit replica)", "MOM_SHORT": "yr5 replay", "FLIP_SHORT": "yr5 replay",
    "SPIKE_FADE": "yr5 replay", "BULLRUN": "yr5 replay", "BEARRUN": "yr5 replay", "FRENZY_LONG": "yr5 replay (+ WIDE fills re-coded by the 3.0 cap)",
    "FRENZY_WIDE": "yr5 replay", "FRENZY_LITE": "study cohort (lite_streak K12, ticks), NOT engine replay",
    "FRENZY_WILLY": "study-cohort re-price on ticks (new script), NOT engine replay", "FRENZY_WILLY_A": "idem, trigger A only",
    "FRENZY_WILLY_B": "idem, trigger B only", "SURGE_LONG": "yr5 replay (trigger + exit MISMATCHED vs today)",
    "SURGE_SHORT": "yr5 replay (sleeve OFF — reference)", "XCHECK_FRENZY_TICK_COHORT": "vol/mcap study FRENZY TODAY+bearish, ticks 8 s (cross-check)",
    "XCHECK_WIDE_TICK_COHORT": "idem, WIDE (cross-check)", "TOTAL_REPLAY_ONLY": "replay sleeves + heat re-admits",
    "TOTAL_LIVE_NO_WILLY": "replay + LITE", "TOTAL_LIVE_WITH_WILLY": "replay + LITE + WILLY"}

rows = []
for k, v in R["res"].items():
    for h in ("H1", "H2", "FY"):
        x = v[h]
        rows.append(dict(strategy=k, half=h, N_mean_seeds=round(x["N"], 1), WR_pct=round(x["WR"], 1) if x["WR"] == x["WR"] else None,
                         avg_pnl_pct=round(x["avg"], 3) if x["avg"] == x["avg"] else None, ci95_lo=round(x["lo"], 3) if x["lo"] == x["lo"] else None,
                         ci95_hi=round(x["hi"], 3) if x["hi"] == x["hi"] else None, net_usd_flat3k=round(x["usd"]), days=x["days"], source=SRC.get(k, "")))
O = pd.DataFrame(rows)
# per-seed $ per half (replay sleeves)
K = F[F.keep]
for h, (a, b) in HALF.items():
    m = (K.t >= a) & (K.t < b)
    ps = K[m].groupby(["S", "seed"]).usd.sum().unstack()
    for s in ps.index:
        for sd in ps.columns:
            O.loc[(O.strategy == s) & (O.half == h), f"usd_seed{sd}"] = round(ps.at[s, sd])
O.to_csv(OUT, index=False)
print("wrote", OUT, len(O))

# ── Oct-6 → today waterfall (FY $, mean of seeds) ──
ns = F.seed.nunique(); base = BOOK * SLOT
def usd(df, pct, lev, inv=1.0, fr=None):
    return float((base * inv * lev * (df.fr if fr is None else fr) * pct / 100).sum() / ns)
wf = {}
fl0 = F[(F.sleeve == "FRENZY_LONG")]
lev_fl = np.where((pd.to_numeric(fl0.entry_frenzy_adx_delta, errors="coerce") > 0) & (pd.to_numeric(fl0.entry_frenzy_di_spread, errors="coerce") > 0), 10, 6)
mv = F[F.note != ""]
lev_mv = np.where((pd.to_numeric(mv.entry_frenzy_adx_delta, errors="coerce") > 0) & (pd.to_numeric(mv.entry_frenzy_di_spread, errors="coerce") > 0), 10, 6)
wf["FRENZY_LONG"] = [("Oct-6 table (TP +4, ATR cap 2.5, no bearish block)", usd(fl0, fl0.pct, lev_fl)),
                     ("exit fixed +3 / −3 (peak ≥ +3 → +3)", usd(fl0, fl0.pct_today, lev_fl)),
                     ("+ ATR 2.5–3.0 red setups (were WIDE fills)", usd(fl0, fl0.pct_today, lev_fl) + usd(mv, mv.pct_today, lev_mv)),
                     ("+ bearish-day block = today", R["res"]["FRENZY_LONG"]["FY"]["usd"])]
w0 = F[F.sleeve == "FRENZY_WIDE"]; w1 = w0[w0.note == ""]
hg = w1[~w1.why.isin(["WIDE_ATR_HIGH(>3.0)", "WIDE_RECLAIM(streak≤12)", "WIDE_STREAK_UNKNOWN(fail-closed)", "WIDE_OTHER"])]
wf["FRENZY_WIDE"] = [("Oct-6 table (every ATR-high / green refusal, TP +4)", usd(w0, w0.pct, 4)),
                     ("exit fixed +3 / −3", usd(w0, w0.pct_today, 4)),
                     ("− the ATR 2.5–3.0 red fills (now FRENZY_LONG)", usd(w1, w1.pct_today, 4)),
                     ("hold-green only (streak > 12, ATR ≤ 3.0; unknown streak refused)", usd(hg, hg.pct_today, 4)),
                     ("+ bearish-day block = today", R["res"]["FRENZY_WIDE"]["FY"]["usd"])]
ml = F[F.sleeve == "MOM-long"]
cm = pd.to_numeric(ml.cell_multiplier, errors="coerce").fillna(1)
src = ml.cell_multiplier_source.fillna("").astype(str)
inv6 = np.where(src.str.contains("UNMATCHED") & (cm >= 2), 1.5, cm)
ml_k1 = ml[~ml.why.isin(["LONG_HEAT_BLOCK(3-leg)"])]
wf["MOM_LONG"] = [("Oct-6 table (UNMATCHED 1.5×, CALM3D 2×)", usd(ml, ml.pct, 20, inv6)),
                  ("today's cell sizing (CALM3D 1×, sprint / pair-vol de-mux → 1×)", usd(ml, ml.pct, 20, ml.inv_today)),
                  ("− LONG_HEAT 3-leg blocks (bull 80–85 ∧ BTC hot)", usd(ml_k1, ml_k1.pct, 20, ml_k1.inv_today)),
                  ("− LONG_CHOP_BURST blocks", R["res"]["MOM_LONG"]["FY"]["usd"]),
                  ("+ heat re-admits (bull ≥ 85, BTC not hot) = today", R["res"]["MOM_LONG_INCL_READMIT"]["FY"]["usd"])]
fp = F[F.sleeve == "FLIP-short"]
wf["FLIP_SHORT"] = [("Oct-6 table (cells 1×, FAN 20×)", usd(fp, fp.pct, 20, 1.0)), ("FAN flips 10× = today", R["res"]["FLIP_SHORT"]["FY"]["usd"])]
sf = F[F.sleeve == "Spike-Fade"]
wf["SPIKE_FADE"] = [("Oct-6 table (replay's own ticket ratio)", usd(sf, sf.pct, 20, 2.0)),
                    ("today's 0.5 %-of-24 h-volume ticket on a $3k book = today", R["res"]["SPIKE_FADE"]["FY"]["usd"])]
br = F[F.sleeve == "BearRun-Short"]
wf["BEARRUN"] = [("Oct-6 table (1×)", usd(br, br.pct, 1)), ("5× = today", R["res"]["BEARRUN"]["FY"]["usd"])]
json.dump(wf, open(os.path.join(D, "waterfall.json"), "w"), indent=1)
for k, v in wf.items():
    print(k); [print(f"   {a:<70} {b:+9.0f}") for a, b in v]

# ── WILLY sensitivity: live backstop ≈ −2.2 % price (net ≈ −2.29) and the 20× liquidation zone ──
W = pd.read_csv(os.path.join(D, "willy_priced.csv")); W = W[W.st == "ok"].copy(); W["t"] = pd.to_datetime(W.te, unit="ms")
k20 = BOOK * SLOT * LEV(1.0) / 100
for lab, stop in (("paper, no stop (as specified)", None), ("live exchange backstop ≈ −2.2 % price (approx.)", -2.29)):
    p = W.pct if stop is None else np.where(W.worst <= stop, stop - 0.05, W.pct)
    for h, (a, b) in HALF.items():
        m = (W.t >= a) & (W.t < b)
        lo, hi = boot_day(np.asarray(p)[m], W.t[m].dt.floor("D").values)
        print(f"WILLY {lab:<48} {h} N {m.sum():5d} WR {(np.asarray(p)[m] > 0).mean() * 100:5.1f} avg {np.asarray(p)[m].mean():+.3f} [{lo:+.2f},{hi:+.2f}] $ {np.asarray(p)[m].sum() * k20:+8.0f}")
print(f"WILLY dips: worst ≤ −2.29 net {int((W.worst <= -2.29).sum())} · ≤ −4.5 (20× liquidation zone) {int((W.worst <= -4.5).sum())} · ≤ −10 {int((W.worst <= -10).sum())} · "
      f"worst trade {W.pct.min():+.2f} % = ${W.pct.min() * k20:+.0f} on the $3k book · time-cap exits {int((W.why == 'time_cap').sum())} avg {W[W.why == 'time_cap'].pct.mean():+.2f}")
# max single-trade $ loss per sleeve (flat book)
print(K.groupby("S").usd.min().round(0).to_string())
