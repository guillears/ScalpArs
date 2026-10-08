#!/usr/bin/env python3
"""Oct-8 SPIKE_FADE H1 deep review — corrected copy of the yr5 halves table (scripts/study_yr5_halves_today.py is NOT edited; it is
imported read-only and its frames are re-priced here).

What this script changes vs study_yr5_halves_today.py (each one labelled in the output):
  FIX-A  sizing parity, every replay sleeve: the original multiplies each non-fade fill by `fr`, the replay's OWN fill ratio, measured
         on the replay's compounding $5k+ chunk book. 13 % of MOM_LONG and 17 % of FLIP fills were liquidity-capped at THAT book size
         (0.1 % × 24 h volume < the bigger desired notional); on the flat $3k ruler the same cap never binds (desired $14.6k × size;
         cap ≥ desired on every fill). FIX-A replaces fr by the $3k-book cap ratio min(1, 0.1 % × vol / desired_3k) on capped rows
         (uncapped rows keep fr). SPIKE_FADE already recomputes its ticket on the $3k book (0.5 % cap) — unchanged.
  FIX-B  WILLY global-hold parity (TOTAL-with-WILLY row only): fills of every other sleeve opened while a priced WILLY trade is open
         [te, xt) are refused, as the live engine does since DECISION_LOG 251. The original stated this but did not model it.
  SENS-S1 (sensitivity, NOT a correction): SPIKE_FADE stop exits booked at the −1.50 % line (live PAPER accounting) instead of the
         replay's first print past the line. Real-money fills are WORSE than the replay (scout STOP_SLIP live proxy ≈ −0.32 %).
Usage: venv/bin/python scripts/study_fade_h1_fixed_halves.py   → prints the side-by-side table, writes reports/study_fade_h1_fixed_halves.csv
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import study_yr5_halves_today as H                                # noqa: E402  (read-only import)

BOOK, SLOT = H.BOOK, H.SLOT
REPLAY = ["MOM_LONG", "MOM_SHORT", "FLIP_SHORT", "SPIKE_FADE", "BULLRUN", "BEARRUN", "FRENZY_LONG", "FRENZY_WIDE", "SURGE_LONG", "SURGE_SHORT"]
FADE_SL = -1.50


def fix_a(F):
    n = lambda c: pd.to_numeric(F[c], errors="coerce")
    capped = F.liquidity_capped.astype(str).str.lower().isin(["true", "1", "1.0"])
    des3k = BOOK * SLOT * F.inv_today * F.lev_today
    cap3k = np.fmin(0.001 * n("entry_pair_volume_24h_usd"), 500_000.0)
    r3k = np.fmin(1.0, cap3k / des3k).fillna(1.0)
    fr_new = np.where(capped & (F.S != "SPIKE_FADE"), r3k, F.fr)
    scale = np.where(F.S == "SPIKE_FADE", 1.0, fr_new / F.fr)
    F["fr_fixA"] = fr_new
    F["usd_fixA"] = F.usd * scale                               # usd is 0 for refused fills already
    return F


def willy_hold_mask(t, W):
    """True where a fill opened inside any WILLY open window [te, xt)."""
    te = W.te.values.astype("int64"); xt = W.xt.values.astype("int64")
    o = np.argsort(te); te, xt = te[o], np.maximum.accumulate(xt[o])
    tm = pd.to_datetime(t).values.astype("datetime64[ms]").astype("int64")
    j = np.searchsorted(te, tm, side="right") - 1
    return (j >= 0) & (tm < np.where(j >= 0, xt[np.maximum(j, 0)], 0))


def main():
    F = H.build_replay(False)
    ns = F.seed.nunique()
    F = fix_a(F)
    K = F[F.keep].copy()
    W = H.willy(); W = W[W.xt.notna()]
    K["willy_held"] = willy_hold_mask(K.t, W)
    rows = []

    def put(name, old_df, new_df, nseeds, label):
        o, n = H.stats(old_df, nseeds), H.stats(new_df, nseeds)
        for h in ("H1", "H2", "FY"):
            rows.append(dict(strategy=name, half=h, change=label,
                             N_old=o[h]["N"], WR_old=o[h]["WR"], avg_old=o[h]["avg"], usd_old=o[h]["usd"],
                             N_new=n[h]["N"], WR_new=n[h]["WR"], avg_new=n[h]["avg"], lo_new=n[h]["lo"], hi_new=n[h]["hi"], usd_new=n[h]["usd"]))

    for s in REPLAY:
        x = K[K.S == s].assign(pct=lambda d: d.pct_today)
        put(s, x[["t", "pct", "usd"]], x.assign(usd=x.usd_fixA)[["t", "pct", "usd"]], ns, "FIX-A" if s != "SPIKE_FADE" else "none")
    # SENS-S1 — fade stops at the line
    fd = K[K.S == "SPIKE_FADE"].copy()
    stop = fd.close_reason.astype(str).str.startswith("STOP")
    tick = fd.usd / (fd.pct / 100)
    pct_s1 = np.where(stop & (fd.pct < FADE_SL), FADE_SL, fd.pct)
    put("SPIKE_FADE·SENS-S1 stops at line", fd.assign(pct=fd.pct)[["t", "pct", "usd"]],
        fd.assign(pct=pct_s1, usd=tick * pct_s1 / 100)[["t", "pct", "usd"]], ns, "SENS-S1 (not a fix)")
    # SENS: fades refused by the WILLY hold
    put("SPIKE_FADE·under WILLY hold", fd[["t", "pct", "usd"]], fd[~fd.willy_held][["t", "pct", "usd"]], ns, "FIX-B view")

    # add-on cohorts, unchanged (no fr in their pricing)
    ml_inv = K[K.S == "MOM_LONG"].inv_today.mean()
    A, _, _ = H.heat_readmits(ml_inv)
    L = H.lite(); L = L[L.keep]
    A["willy_held"] = willy_hold_mask(A.t, W); L["willy_held"] = willy_hold_mask(L.t, W)
    adds = lambda a: a[["t", "pct", "usd", "w"]].assign(usd=a.usd * ns, w=a.w * ns)
    lit = lambda l: l[["t", "pct", "usd"]].assign(usd=l.usd * ns, w=float(ns))
    wil = W[["t", "pct", "usd"]].assign(usd=W.usd * ns, w=float(ns))
    live = K[K.S != "SURGE_SHORT"].assign(pct=K.pct_today, w=1.0)
    old_rep = live[["t", "pct", "usd", "w"]]
    new_rep = live.assign(usd=live.usd_fixA)[["t", "pct", "usd", "w"]]
    put("TOTAL replay sleeves only", old_rep, new_rep, ns, "FIX-A")
    put("TOTAL live without WILLY", pd.concat([old_rep, adds(A), lit(L)]), pd.concat([new_rep, adds(A), lit(L)]), ns, "FIX-A")
    held_rep = live[~live.willy_held].assign(usd=lambda d: d.usd_fixA)[["t", "pct", "usd", "w"]]
    put("TOTAL live with WILLY (paper)", pd.concat([old_rep, adds(A), lit(L), wil]),
        pd.concat([held_rep, adds(A[~A.willy_held]), lit(L[~L.willy_held]), wil]), ns, "FIX-A + FIX-B")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(ROOT, "reports", "study_fade_h1_fixed_halves.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    held = K.groupby("S").willy_held.mean().mul(100).round(1)
    print("\nWILLY-hold refusal share by sleeve (%):", held.to_dict(), "| heat readmits", round(A.willy_held.mean() * 100, 1), "| LITE", round(L.willy_held.mean() * 100, 1))
    capped = F.liquidity_capped.astype(str).str.lower().isin(["true", "1", "1.0"])
    print("FIX-A rows changed by sleeve:", F[capped & (F.S != "SPIKE_FADE") & F.keep].groupby("S").size().div(ns).round(1).to_dict())


if __name__ == "__main__":
    main()
