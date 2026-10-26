#!/usr/bin/env python3
"""ML impulse study — STEP 2: the impulse score / conjunction / spike / I1-I3 zones as a MOMENTUM-LONG entry filter, per SIGNAL
(refused fill = 0), P&L at 1x on today's exit stack (replica `base` of reports/ML_BTCFWD_exit_rows.csv — the momentum-exit
replica scripts/ml_exit_optimize_yr5.py BASE) and, the second way, the engine / as-traded P&L. Pre-registration:
reports/ML_IMPULSE_PREREG.txt (26 cells; only Step-1 passers in the cell's direction are gate-eligible).

    venv/bin/python -W ignore scripts/study_ml_impulse_step2.py > reports/ML_IMPULSE_step2.txt
    venv/bin/python -W ignore scripts/study_ml_impulse_step2.py --lag5 > reports/ML_IMPULSE_step2_lag5.txt   (scan-lag sensitivity:
        5m features one bar earlier; writes *_lag5 files)
Writes reports/ML_IMPULSE_step2_cells.csv and reports/ML_IMPULSE_fill_features.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_ml_impulse_features import OUT5, CONT, LEGS, REP, fill_features  # noqa: E402
from study_ml_btcfwd_common import lomo_min, max_share, breakeven_wr  # noqa: E402


def spearman(a, b):
    a = pd.Series(a).rank().values; b = pd.Series(b).rank().values
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def boot_mean(y, cl, n=2000, seed=7):
    """cluster bootstrap of the mean -> (mean, P(mean > 0))."""
    y = np.asarray(y, float); cl = np.asarray(cl)
    if len(y) == 0:
        return np.nan, np.nan
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=y); c = np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(y.mean()), float((bs > 0).mean())


def boot_sep(y, z, day, n=2000, seed=11):
    """day bootstrap of mean(zone) - mean(rest) -> (delta, P(delta < 0))."""
    y = np.asarray(y, float); z = np.asarray(z, bool); day = np.asarray(day)
    u, inv = np.unique(day, return_inverse=True)
    si = np.bincount(inv, weights=y * z, minlength=len(u)); ci = np.bincount(inv, weights=z.astype(float), minlength=len(u))
    so = np.bincount(inv, weights=y * ~z, minlength=len(u)); co = np.bincount(inv, weights=(~z).astype(float), minlength=len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = si[idx].sum(1) / ci[idx].sum(1) - so[idx].sum(1) / co[idx].sum(1)
    bs = bs[np.isfinite(bs)]
    return float(y[z].mean() - y[~z].mean()), float((bs < 0).mean())


def cells(F, frz_sign, q20):
    C = {}
    s = F.SCORE.values
    for k in range(1, 7):
        C[f"SCORE_refuse_le{k}"] = ("SCORE", +1, np.where(np.isfinite(s), s <= k, False))
    for k in range(4, 9):
        C[f"SCORE_refuse_ge{k}"] = ("SCORE", -1, np.where(np.isfinite(s), s >= k, False))
    cj = F.CONJ.values
    C["CONJ_enter_only"] = ("CONJ", +1, np.where(np.isfinite(cj), cj == 0, False))
    C["CONJ_refuse"] = ("CONJ", -1, np.where(np.isfinite(cj), cj == 1, False))
    sp = F.SPIKE.values
    C["SPIKE_refuse"] = ("SPIKE", 0, np.where(np.isfinite(sp), sp == 1, False))
    for f in CONT:
        v = frz_sign[f] * F[f].values
        C[f"Z1_{f}"] = (f, frz_sign[f], np.where(np.isfinite(v), v <= q20[f], False))
    return C


LAG = 1 if "--lag5" in sys.argv else 0
SFX = "_lag5" if LAG else ""


def main():
    S1 = pd.read_csv(os.path.join(REP, "ML_IMPULSE_step1.csv"))
    G = pd.read_csv(OUT5, index_col=0)
    g25 = G[(pd.to_datetime(G.index, unit="ms").year == 2025)]
    frz_sign, q20 = {}, {}
    for f in CONT:
        k = g25[[f, "Y_alt"]].dropna()
        frz_sign[f] = 1.0 if spearman(k[f].values, k.Y_alt.values) >= 0 else -1.0
        q20[f] = float(np.quantile(frz_sign[f] * k[f].values, 0.2))
    E = pd.read_csv(os.path.join(REP, "ML_BTCFWD_exit_rows.csv"),
                    usecols=["key", "block", "batch", "pair", "o_ms", "seed", "today_engine", "stack_pct", "stack_keep", "base", "src"])
    FF = fill_features(E.o_ms.values, lag5=LAG)
    FF.index = E.index
    F = pd.concat([E, FF], axis=1)
    F["day"] = pd.to_datetime(F.o_ms, unit="ms").dt.floor("D")
    F["month"] = F.day.dt.to_period("M").astype(str)
    F.to_csv(os.path.join(REP, f"ML_IMPULSE_fill_features{SFX}.csv"), index=False)
    # no-lookahead at fills: every 1m feature row used is the boundary <= entry, every 5m row the 5m boundary <= entry
    assert (F.t1 <= F.o_ms).all() and (F.t5 <= F.o_ms).all() and (F.o_ms - F.t5 < 300_000 * (1 + LAG)).all()
    keep = F.stack_keep.astype(str).eq("True")
    BL = {"2025": F.block == "2025 BULL", "2026": F.block.isin(["2026-H1 BEAR", "2026-H2 BULL"]),
          "2026-H1": F.block == "2026-H1 BEAR", "2026-H2": F.block == "2026-H2 BULL",
          "master": F.block == "master", "master_keep": (F.block == "master") & keep}
    pd.set_option("display.width", 260)
    print("## Coverage at fills (share with SCORE / CONJ / corr15 / er_15 defined)")
    for b, m in BL.items():
        s = F[m]
        print(f"  {b:12s} N {len(s):5d} days {s.day.nunique():4d} · SCORE {s.SCORE.notna().mean():.0%} · CONJ {s.CONJ.notna().mean():.0%}"
              f" · corr15 {s.corr15.notna().mean():.0%} · er_15 {s.er_15.notna().mean():.0%} · base avg {s.base.mean():+.3f} · engine {s.today_engine.mean():+.3f}")

    # (0) descriptive: does the score at entry separate winners / losers and BTC's next 15 min at fills?
    print("\n## (0) Fills by SCORE bucket — N · WR · avg % (replica base / engine) · BTC fwd15 at fills (bps)")
    G1 = pd.read_csv(os.path.join(REP, "ML_IMPULSE_grid1m.csv.gz"), index_col=0, usecols=["t", "Y_btc"])
    F["btc_fwd15"] = G1.Y_btc.reindex(F.t1.values).values
    F["sb"] = pd.cut(F.SCORE, [-1, 2, 3, 4, 5, 6, 9], labels=["<=2", "3", "4", "5", "6", ">=7"])
    rows = []
    for b in ("2025", "2026-H1", "2026-H2", "master"):
        s = F[BL[b]]
        for k, g in s.groupby("sb", observed=True):
            rows.append(dict(block=b, score=k, N=len(g), days=g.day.nunique(), WR=(g.base > 0).mean(), base=g.base.mean(),
                             engine=g.today_engine.mean(), btc_fwd_bps=g.btc_fwd15.mean() * 100,
                             P_btc_up10=(g.btc_fwd15 >= 0.10).mean()))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\n   Spearman(SCORE, outcome) at fills [base %, BTC fwd15]:")
    for b in ("2025", "2026-H1", "2026-H2", "master"):
        s = F[BL[b]].dropna(subset=["SCORE"])
        print(f"   {b:8s} N {len(s):5d}  ρ(score, base) {spearman(s.SCORE, s.base):+.3f}  ρ(score, btc_fwd15) "
              f"{spearman(s.SCORE, s.btc_fwd15.fillna(0)):+.3f}")

    # (1) cells
    C = cells(F, frz_sign, q20)
    passed = S1[S1.STEP1_PASS.astype(bool)]
    out = []
    for name, (item, direction, z) in C.items():
        if item == "SPIKE":
            elig = False
        else:
            p = passed[passed.item == item]
            elig = bool(len(p)) and any(np.sign(r.orient25) == direction for r in p.itertuples())
        r = dict(cell=name, item=item, eligible=elig)
        for b, m in BL.items():
            mm = m.values
            for tag, col in (("", "base"), ("_eng", "today_engine")):
                y = F.loc[mm, col].astype(float).values
                if b == "master_keep" and tag == "_eng":
                    y = F.loc[mm, "stack_pct"].astype(float).values
                zz = z[mm]; day = F.day.values[mm]; mon = F.month.values[mm]; pr = F.pair.values[mm]
                d = np.where(zz, -y, 0.0)
                dl, P = boot_mean(d, day)
                r[f"nz_{b}{tag}"] = int(zz.sum()); r[f"d_{b}{tag}"] = dl; r[f"P_{b}{tag}"] = P
                if tag == "":
                    r[f"N_{b}"] = len(y)
                    if zz.sum() >= 3 and (~zz).sum() >= 3:
                        r[f"sep_{b}"], r[f"Psep_{b}"] = boot_sep(y, zz, day)
                    if b in ("2025", "2026", "master"):
                        yz = y[zz]
                        be = breakeven_wr(y)
                        mz, Pp = boot_mean(yz, day[zz]) if len(yz) else (np.nan, np.nan)
                        _, Pm = boot_mean(yz, mon[zz]) if len(yz) else (np.nan, np.nan)
                        nd = len(np.unique(day[zz]))
                        dsh = max_share(-yz, day[zz]) if len(yz) else np.nan
                        psh = max_share(-yz, pr[zz]) if len(yz) else np.nan
                        ok = (len(yz) >= 15 and (yz > 0).mean() < be and (1 - Pp) >= 0.95 and nd >= 8 and
                              np.isfinite(dsh) and dsh < 0.5 and np.isfinite(psh) and psh < 0.5)
                        r.update({f"bar_{b}_N": len(yz), f"bar_{b}_WR": (yz > 0).mean() if len(yz) else np.nan, f"bar_{b}_BE": be,
                                  f"bar_{b}_avg": mz, f"bar_{b}_Pneg_day": 1 - Pp, f"bar_{b}_Pneg_month": 1 - Pm,
                                  f"bar_{b}_days": nd, f"bar_{b}_dayshare": dsh, f"bar_{b}_pairshare": psh, f"bar_{b}_BAR": bool(ok),
                                  f"two_{b}": f"L{int((yz < 0).sum())}:{yz[yz < 0].sum():+.1f} / W{int((yz > 0).sum())}:{yz[yz > 0].sum():+.1f}"})
                    if b in ("2025", "2026"):
                        r[f"lomo_{b}"] = lomo_min(d, mon)
                        r[f"monthshare_{b}"] = max_share(d, mon)
                        r[f"dayshare_{b}"] = max_share(d, day)
        g1 = r["d_2025"] > 0 and r["P_2025"] >= 0.9 and r["d_2026"] > 0 and r["P_2026"] >= 0.9
        g2 = r["d_master"] >= 0 and r["d_master_keep"] >= 0
        g3 = all(r.get(f"sep_{b}", 1) < 0 for b in ("2025", "2026-H1", "2026-H2", "master"))
        g4 = all(r.get(f"bar_{b}_BAR", False) for b in ("2025", "2026", "master"))
        g5 = all(r.get(f"lomo_{y}", -1) > 0 and np.isfinite(r.get(f"monthshare_{y}", np.nan)) and r[f"monthshare_{y}"] < 0.5
                 for y in ("2025", "2026"))
        allg = g1 and g2 and g3 and g4 and g5
        one = (r["d_2025"] > 0 and r["P_2025"] >= 0.9) != (r["d_2026"] > 0 and r["P_2026"] >= 0.9)
        r.update(G1=g1, G2=g2, G3=g3, G4=g4, G5=g5,
                 verdict=("CANDIDATE" if allg and elig else ("pass-but-EXPLORATORY" if allg else ("one-year-only" if one else "fail"))))
        out.append(r)
    T = pd.DataFrame(out)
    T.to_csv(os.path.join(REP, f"ML_IMPULSE_step2_cells{SFX}.csv"), index=False)
    ff = lambda v: f"{v:+.3f}"
    print(f"\n## (1) Filter cells: Δ %/signal (refused = 0) · replica base [engine] · P(Δ>0) day-bootstrap · refused N")
    for b in ("2025", "2026", "2026-H1", "2026-H2", "master", "master_keep"):
        T[f"{b}"] = [f"{x:+.3f} [{e:+.3f}] P{p:.2f} n{n}" for x, e, p, n in zip(T[f"d_{b}"], T[f"d_{b}_eng"], T[f"P_{b}"], T[f"nz_{b}"])]
    print(T[["cell", "eligible", "2025", "2026", "2026-H1", "2026-H2", "master", "master_keep", "verdict"]].to_string(index=False))
    print("\n## (2) Separation: refused − kept avg % (replica) [P(refused worse)] — a filter needs < 0 in every block")
    for b in ("2025", "2026-H1", "2026-H2", "master"):
        T[f"s{b}"] = [f"{x:+.3f} [{p:.2f}]" if np.isfinite(x) else "—" for x, p in zip(T.get(f"sep_{b}", np.nan), T.get(f"Psep_{b}", np.nan))]
    print(T[["cell", "s2025", "s2026-H1", "s2026-H2", "smaster"]].to_string(index=False))
    print("\n## (3) Expectancy bar on the refused cohort (replica, 1x): N · WR vs breakeven · avg · P(avg<0) day [month] · days · max day/pair share")
    for b in ("2025", "2026", "master"):
        T[f"e{b}"] = [f"N{n} WR{w:.0%}/BE{be:.0%} {a:+.3f} P{p:.2f}[{pm:.2f}] d{d} sh{ds:.2f}/{ps:.2f} {'PASS' if ok else 'fail'}"
                      if n else "—" for n, w, be, a, p, pm, d, ds, ps, ok in
                      zip(T[f"bar_{b}_N"], T[f"bar_{b}_WR"], T[f"bar_{b}_BE"], T[f"bar_{b}_avg"], T[f"bar_{b}_Pneg_day"],
                          T[f"bar_{b}_Pneg_month"], T[f"bar_{b}_days"], T[f"bar_{b}_dayshare"], T[f"bar_{b}_pairshare"], T[f"bar_{b}_BAR"])]
    print(T[["cell", "e2025", "e2026", "emaster"]].to_string(index=False))
    print("\n## (4) Two-sided (refused losers : sum % / refused winners : sum %) and window units (LOMO min, max month / day share of +Δ)")
    print(T[["cell", "two_2025", "two_2026", "two_master", "lomo_2025", "lomo_2026", "monthshare_2025", "monthshare_2026",
             "dayshare_2025", "dayshare_2026"]].to_string(index=False, float_format=ff))
    print("\nverdicts:", T.verdict.value_counts().to_dict())
    print("frozen 2025 signs vs Y_alt:", {k: int(v) for k, v in frz_sign.items()})

    # live fill
    lv = F[F.block == "live"]
    if len(lv):
        print("\n## Live fill(s) — score and legs at entry")
        cols = ["pair", "o_ms", "base", "today_engine", "SCORE", "CONJ", "SPIKE"] + LEGS + CONT + \
               ["btc_r15", "eth_r15", "sol_r15", "xrp_r15", "zec_r15", "top10_up", "br_trend", "br_trend_d15", "top10_tbr", "vsurge",
                "stale_list", "btc_fwd15"]
        print(lv[cols].T.to_string())
        for name, (item, direction, z) in C.items():
            if z[lv.index[0]]:
                print("   live fill is REFUSED by", name)


if __name__ == "__main__":
    main()
