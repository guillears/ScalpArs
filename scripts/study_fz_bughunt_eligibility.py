#!/usr/bin/env python3
"""🐞 Oct-10 FRENZY bug-hunt — LIVE ELIGIBILITY of every backtest fill.
The live FRENZY pass only reads pairs from binance_service.get_top_futures_pairs(new_listing_filter_days=90, alpha_subtype_filter_enabled=True,
coin_underlying_only=True) (services/trading_engine.py _update_frenzy) → a pair listed < 90 days before the signal, an "Alpha" underlyingSubType
or a non-COIN underlying can NEVER fire live. The yr5 replay (cohort a) emulates this (scripts/engine_replay.py UNI.top); the study cohorts
(b / b_off_engine / a_off_addback = FRENZY_ENGINE_COHORT, c = the LITE study cohort) and the Oct rebuild do NOT. Subtypes / onboard dates
from reports/backtest_cache/exchange_info.json (snapshot 2026-09-16 — a later Alpha → non-Alpha move is not visible; pairs absent from the
snapshot are counted 'unknown'). Writes reports/FRENZY_BACKTEST_BUGHUNT_2026-10-10_eligibility.csv.
Usage: venv/bin/python scripts/study_fz_bughunt_eligibility.py"""
import json, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = json.load(open(os.path.join(ROOT, "reports", "backtest_cache", "exchange_info.json")))
DAY = 86_400_000


def why(pair, t):
    m = EX.get(pair)
    if m is None:
        return "unknown"
    st = m.get("underlyingSubType") or []
    if any("Alpha" in x for x in st):
        return "ALPHA"
    if m.get("underlyingType") not in (None, "COIN"):
        return "NON_COIN"
    ob = m.get("onboardDate")
    if ob and t - int(ob) < 90 * DAY:
        return "NEW_LISTING"
    return "eligible"


def s(g):
    return pd.Series(dict(n=len(g), wr=round((g.fix3 > 0).mean(), 3), avg=round(g.fix3.mean(), 3)))


def main():
    F = pd.read_csv(os.path.join(ROOT, "reports", "FRENZY_TP2_VS_TP3_TICKS_2026-10-09_fills.csv")); F = F[F.st == "ok"].copy()
    F["sleeve"] = F.sleeve.replace({"FRENZY": "FRENZY_LONG", "WIDE": "FRENZY_WIDE"})
    F["elig"] = [why(p, int(t)) for p, t in zip(F.pair, F.sig)]
    O = pd.read_csv(os.path.join(ROOT, "reports", "FRENZY_BACKTEST_BUGHUNT_2026-10-10_oct_signals.csv"))
    O = O[O.how.isin(["TP", "SL", "CAP"]) & (O["stack"] == "kept")].rename(columns={"kind": "sleeve"}).assign(cohort="oct (live days)", seed=0)
    O["elig"] = [why(p, int(t)) for p, t in zip(O.pair, O.sig)]
    sets = {"a gate ON (seed 1)": F[(F.cohort == "a") & (F.seed == 1)], "a_off_addback": F[F.cohort == "a_off_addback"],
            "b_off_engine": F[F.cohort == "b_off_engine"], "c LITE (no bearish)": F[(F.cohort == "c") & (F.bear != True)], "oct (live days, kept)": O}
    out = []
    for k, g in sets.items():
        for sl, h in g.groupby("sleeve"):
            for e, hh in [("all", h), ("eligible", h[h.elig == "eligible"])] + [(x, h[h.elig == x]) for x in ["ALPHA", "NEW_LISTING", "NON_COIN", "unknown"]]:
                if len(hh):
                    out.append(dict(cohort=k, sleeve=sl, subset=e, **s(hh)))
    R = pd.DataFrame(out)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(R.to_string())
    R.to_csv(os.path.join(ROOT, "reports", "FRENZY_BACKTEST_BUGHUNT_2026-10-10_eligibility.csv"), index=False)
    # corrected gate-off LONG / WIDE = a (eligible by construction) + eligible add-back
    a = sets["a gate ON (seed 1)"]; ad = sets["a_off_addback"]
    T = pd.concat([a, ad[ad.elig == "eligible"]])
    print("\nTODAY's stack (gate off), eligible universe:"); print(T.groupby("sleeve").apply(s))
    print("a alone elig check:", a.elig.value_counts().to_dict())


if __name__ == "__main__":
    main()
