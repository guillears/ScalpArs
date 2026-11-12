#!/usr/bin/env python3
"""ML WIDE STOP × FIXED TP — MASTER (2026-11-11, operator: "or all about exit, wider SL (−3 %? −4 %?) and higher TP 2 %? 3 %? 4 % — maybe the
entry is the same, we are looking only for TP and not SL").
PRE-DECLARED grid (before pricing), on the 123 kept master ML fills (today's stack), validated replica, recovery hold OFF for every non-live
stop (it is premised on the live stop):
  WT_sl<s>_tp<t>[_H]  fixed stop s ∈ {−1.5, −2, −3, −4} × fixed TP t ∈ {1.5, 2, 3, 4}; runner + ladder off; live 180-min clock, or _H = held
                      to the 20 h max hold. TP fills at its price (resting).
  WS_sl<s>            today's exit with ONLY the stop widened (fixed s) — a stop counterfactual: read on the LIVE-STOPPED cohort (saved vs
                      deeper); fills today did not stop must show Δ = 0 (CLAUDE.md, DL 110).
Writes reports/ML_WIDE_SL_TP_2026-11-11_{variants,summary}.csv + .txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S, study_ml_exit_macro_1111_report as R, scout_ml_matched_refusals as MM, review_math as RM  # noqa: E401,E402
OUT = os.path.join(S.ROOT, "reports", "ML_WIDE_SL_TP_2026-11-11")


def grid():
    RP = S._rp(); G = {"BASE": dict(RP.BASE)}
    for s in (-1.5, -2.0, -3.0, -4.0):
        for t in (1.5, 2.0, 3.0, 4.0):
            G[f"WT_sl{s:g}_tp{t:g}"] = S.V(run_arm=S.INF, ladder=[], tp=t, sl_fixed=s, rh=False)
            G[f"WT_sl{s:g}_tp{t:g}_H"] = S.V(run_arm=S.INF, ladder=[], tp=t, sl_fixed=s, rh=False, noexp=1200)
        G[f"WS_sl{s:g}"] = S.V(sl_fixed=s, rh=False)
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
    stopped = RR.BASE.astype(str).str.contains("STOP").values
    dcr_b, nd = R.dcr_group(F.assign(pp=b, ex=X.BASE.values), "pp", "ex", eras)
    out = []
    for v in [c for c in W.columns if c != "BASE"]:
        y = W[v].values; ok = ~np.isnan(y); dl = np.where(ok, y - b, 0.0)
        lo, hi, _ = R.boot_ci(dl, F.day.values, 0.05)
        dv, _ = R.dcr_group(F.assign(pp=np.where(ok, y, b), ex=np.where(ok, X[v].values, X.BASE.values)), "pp", "ex", eras)
        dd = pd.Series(dl).groupby(F.day.values).sum(); lo_ = RM.loo(list(dl), list(F.pair + " " + F.day))
        out.append(dict(variant=v, n=int(ok.sum()), wr=np.mean(y[ok] > 0), avg=np.nanmean(y), d=dl.mean(), lo95=lo, hi95=hi,
                        worst=np.nanmin(y), sl_hits=int(RR[v].astype(str).str.contains("STOP").sum()), tp_hits=int((RR[v] == "FIXED_TP").sum()),
                        saved=int((stopped & (dl > 1e-9)).sum()), saved_sum=dl[stopped & (dl > 1e-9)].sum(),
                        deeper=int((stopped & (dl < -1e-9)).sum()), deeper_sum=dl[stopped & (dl < -1e-9)].sum(),
                        nonstopped_d=dl[~stopped].sum(), dcr_before=dcr_b, dcr_after=dv,
                        top_day=(dd.max() / dl.sum() if dl.sum() > 0 else np.nan), d_wo_top=lo_["without_top"] / len(dl),
                        **{f"d_{g}": dl[(F.grp == g).values].mean() for g in sorted(F.grp.unique())}))
    Sm = pd.DataFrame(out).sort_values("d", ascending=False); Sm.to_csv(OUT + "_summary.csv", index=False)
    txt = f"BASE avg {b.mean():+.3f} · WR {(b > 0).mean():.0%} · one-size DCR {dcr_b:+.3f} %/day ({nd} d) · live-stopped {int(stopped.sum())}\n" + Sm.round(3).to_string(index=False)
    open(OUT + ".txt", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
