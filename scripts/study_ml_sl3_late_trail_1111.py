#!/usr/bin/env python3
"""ML STOP −3 × LATE TRAIL — MASTER (2026-11-11, operator: "I don't like a wider SL and not a wider TP — why not −3 with a trailing starting
at 1 % for example? or at 1.5 %?"). PRE-DECLARED (before pricing), 123 kept master ML fills, validated replica, recovery hold off:
  L3_a<arm>_<trail>[_H]  fixed stop −3 · runner arms at +1.0 / +1.5 · trail = entry ATR × 1.0 (live k, 'atr') or a fixed 0.3 / 0.5 / 0.75
                         below the peak · lock +0.10 · ladder off · live 180-min clock, or _H = no clock (20 h max hold) → 16 variants
  refs: BASE (today) · WS_sl-3 (−3 with today's trail, arm 0.40, ladder on, clock on)
Writes reports/ML_SL3_LATE_TRAIL_2026-11-11_{variants,summary}.csv + .txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S, study_ml_exit_macro_1111_report as R, scout_ml_matched_refusals as MM, review_math as RM  # noqa
OUT = os.path.join(S.ROOT, "reports", "ML_SL3_LATE_TRAIL_2026-11-11")


def grid():
    RP = S._rp(); G = {"BASE": dict(RP.BASE), "WS_sl-3": S.V(sl_fixed=-3.0, rh=False)}
    for a in (1.0, 1.5):
        for tr in ("atr", 0.3, 0.5, 0.75):
            kw = dict(sl_fixed=-3.0, rh=False, run_arm=a, ladder=[])
            if tr != "atr":
                kw["run_g_fixed"] = tr
            n = f"L3_a{a:g}_{tr if tr == 'atr' else 'd' + format(tr, 'g')}"
            G[n] = S.V(**kw)
            G[n + "_H"] = S.V(**kw, noexp=1200)
    return G


def main():
    C, _ = S.cohort(); G = grid(); rows = []
    with MM.replica_with_topup() as RP:
        B = S._btc_frames(RP)
        for r in C.itertuples():
            rsi = RP.btc_rsi_at(int(r.o_ms))
            fx = dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in),
                      atr=float(r.atr) if (r.atr == r.atr and r.atr > 0) else None, rsi_entry=None if rsi is None else float(rsi))
            out, _ = S.price_fill(RP, fx, G, B)
            for v, (why, pct, tex, _p) in out.items():
                rows.append(dict(key=r.key, variant=v, reason=why, pct=pct, exit_ms=tex))
    V = pd.DataFrame(rows); V.to_csv(OUT + "_variants.csv", index=False)
    F = pd.read_csv(os.path.join(S.ROOT, "reports", "ML_EXIT_MACRO_2026-11-11_fills.csv"))
    W = V.pivot(index="key", columns="variant", values="pct"); X = V.pivot(index="key", columns="variant", values="exit_ms")
    RR = V.pivot(index="key", columns="variant", values="reason")
    F = F.set_index("key").loc[W.index].reset_index()
    F["day"] = pd.to_datetime(F.o_ms, unit="ms").dt.strftime("%Y-%m-%d"); F["grp"] = F.era.map(R.GROUPS)
    b = W.BASE.values; eras = sorted(F.era.unique())
    dcr_b, nd = R.dcr_group(F.assign(pp=b, ex=X.BASE.values), "pp", "ex", eras)
    out = []
    for v in [c for c in W.columns if c != "BASE"]:
        y = W[v].values; ok = ~np.isnan(y); dl = np.where(ok, y - b, 0.0)
        lo, hi, _ = R.boot_ci(dl, F.day.values, 0.05)
        dv, _ = R.dcr_group(F.assign(pp=np.where(ok, y, b), ex=np.where(ok, X[v].values, X.BASE.values)), "pp", "ex", eras)
        dd = pd.Series(dl).groupby(F.day.values).sum(); lo_ = RM.loo(list(dl), list(F.pair + " " + F.day))
        r = RR[v].astype(str)
        hold = (X[v].values - F.o_ms.values) / 3.6e6
        out.append(dict(variant=v, wr=np.mean(y[ok] > 0), avg=np.nanmean(y), d=dl.mean(), lo95=lo, hi95=hi, dcr=dv,
                        stops=int(r.str.contains("STOP").sum()), trail=int(r.str.contains("RUNNER").sum()), clock=int(r.isin(["NO_EXPANSION", "END_OF_DATA"]).sum()),
                        hold_med_h=np.median(hold), d_win=dl[b > 0].sum(), d_los=dl[b <= 0].sum(),
                        top_day=(dd.max() / dl.sum() if dl.sum() > 0 else np.nan), d_wo_top=lo_["without_top"] / len(dl),
                        **{f"d_{g}": dl[(F.grp == g).values].mean() for g in sorted(F.grp.unique())}))
    Sm = pd.DataFrame(out).sort_values("d", ascending=False); Sm.to_csv(OUT + "_summary.csv", index=False)
    txt = f"BASE avg {b.mean():+.3f} WR {(b > 0).mean():.0%} · one-size DCR {dcr_b:+.3f} %/day ({nd} d)\n" + Sm.round(3).to_string(index=False)
    open(OUT + ".txt", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
