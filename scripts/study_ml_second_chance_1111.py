#!/usr/bin/env python3
"""ML "SECOND CHANCE" EXIT — MASTER (2026-11-11, operator: "this points to an exit!" on the post-exit finding: 66 % of winners trade
≥ +0.5 above our exit within 60 min and only 37 % of those dip 0.3 below the exit first → the pause is SHALLOW, a held trade could sit
through it).

RULE (pre-declared, before pricing): today's exit unchanged until today's RUNNER_TRAIL / HARD_TP_LADDER would fire at net level L. Then,
instead of closing: floor = L − d; a FRESH runner trail restarts from the post-trigger running high (giveback = entry ATR × 1.0, the live
k; no ATR → the live replica's ∞, floor only); the ladder is off in phase 2. Exit = first of floor (≤ L − d) · fresh trail · the live
NO_EXPANSION clock / max hold from the original entry · optional TIME window: no new high ≥ L + 0.20 within T min of the trigger → exit at
market. d ∈ {0.2, 0.3, 0.4, 0.5} × T ∈ {none, 60} = 8 variants (Bonferroni 8). Stop exits / recovery-hold exits / clock exits unchanged.
Read on the same 123 kept master ML fills (today's stack), Δ vs BASE per fill, per era, day-block CI, LOO, one-size DCR.
Writes reports/ML_SECOND_CHANCE_2026-11-11_{variants,summary}.csv + .txt"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S  # noqa: E402
import study_ml_exit_macro_1111_report as R  # noqa: E402
import scout_ml_matched_refusals as MM  # noqa: E402
import review_math as RM  # noqa: E402

OUT = os.path.join(S.ROOT, "reports", "ML_SECOND_CHANCE_2026-11-11")
MIN = 60_000
DS, TS = (0.2, 0.3, 0.4, 0.5), (None, 60)
RUNNERS = ("RUNNER_TRAIL", "HARD_TP_LADDER", "RH_RUNNER_TRAIL", "RH_HARD_TP_LADDER")


def second_chance(P, fx, k, d, T, clock_ms):
    t, net = P["t"], P["net"]
    L = float(net[k])
    g = fx["atr"] * 1.0 if fx["atr"] else np.inf
    m2 = L
    for i in range(k + 1, len(t)):
        v = net[i]
        if v > m2:
            m2 = v
        if v <= L - d + 0.0:
            return i, "SC_FLOOR"
        if v <= m2 - g:
            return i, "SC_TRAIL"
        if t[i] - fx["o_ms"] >= clock_ms:
            return i, "SC_CLOCK"
        if T is not None and t[i] - t[k] >= T * MIN and m2 < L + 0.20:
            return i, "SC_TIME"
    return len(t) - 1, "END_OF_DATA"


def main():
    C, _ = S.cohort()
    rows = []
    with MM.replica_with_topup() as RP:
        for r in C.itertuples():
            rsi = RP.btc_rsi_at(int(r.o_ms))
            fx = dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in),
                      atr=float(r.atr) if (r.atr == r.atr and r.atr > 0) else None, rsi_entry=None if rsi is None else float(rsi))
            P = RP.prep_path(fx, RP.HORIZON_MS)
            if P is None:
                continue
            k, why, pct = RP.evaluate(P, fx, RP.BASE)
            rows.append(dict(key=r.key, variant="BASE", reason=why, pct=pct, exit_ms=int(P["t"][k])))
            clock = min(RP.BASE["noexp"], RP.BASE["maxhold"]) * MIN
            for d in DS:
                for T in TS:
                    v = f"SC_d{d:g}" + (f"_T{T}" if T else "")
                    if why not in RUNNERS:
                        rows.append(dict(key=r.key, variant=v, reason=why, pct=pct, exit_ms=int(P["t"][k])))
                        continue
                    i, w = second_chance(P, fx, k, d, T, clock)
                    rows.append(dict(key=r.key, variant=v, reason=w, pct=float(P["net"][i]), exit_ms=int(P["t"][i])))
    V = pd.DataFrame(rows)
    V.to_csv(OUT + "_variants.csv", index=False)
    F = pd.read_csv(os.path.join(S.ROOT, "reports", "ML_EXIT_MACRO_2026-11-11_fills.csv"))
    W = V.pivot(index="key", columns="variant", values="pct")
    X = V.pivot(index="key", columns="variant", values="exit_ms")
    F = F.set_index("key").loc[W.index].reset_index()
    F["day"] = pd.to_datetime(F.o_ms, unit="ms").dt.strftime("%Y-%m-%d")
    F["grp"] = F.era.map(R.GROUPS)
    b = W.BASE.values
    eras = sorted(F.era.unique())
    dcr_b, nd = R.dcr_group(F.assign(pp=b, ex=X.BASE.values), "pp", "ex", eras)
    out = []
    for v in [c for c in W.columns if c != "BASE"]:
        y = W[v].values
        dl = y - b
        lo, hi, _ = R.boot_ci(dl, F.day.values, 0.05)
        lB, hB, _ = R.boot_ci(dl, F.day.values, 0.05 / 8)
        dv, _ = R.dcr_group(F.assign(pp=y, ex=X[v].values), "pp", "ex", eras)
        dd = pd.Series(dl).groupby(F.day.values).sum()
        lo_ = RM.loo(list(dl), list(F.pair + " " + F.day))
        ch = np.abs(dl) > 1e-9
        out.append(dict(variant=v, wr=(y > 0).mean(), avg=y.mean(), d=dl.mean(), sum_d=dl.sum(), lo95=lo, hi95=hi, loBonf=lB, hiBonf=hB,
                        changed=int(ch.sum()), better=int((dl > 1e-9).sum()), worse=int((dl < -1e-9).sum()),
                        gain=dl[dl > 0].sum(), cost=dl[dl < 0].sum(), dcr_before=dcr_b, dcr_after=dv,
                        top_day=(dd.max() / dl.sum() if dl.sum() > 0 else np.nan), d_wo_top=lo_["without_top"] / len(dl),
                        days_up=int((dd > 1e-9).sum()), days_down=int((dd < -1e-9).sum()),
                        **{f"d_{g}": dl[(F.grp == g).values].mean() for g in sorted(F.grp.unique())},
                        reasons=V[V.variant == v].reason.value_counts().to_dict()))
    Sm = pd.DataFrame(out).sort_values("d", ascending=False)
    Sm.to_csv(OUT + "_summary.csv", index=False)
    txt = f"BASE avg {b.mean():+.3f} · one-size DCR {dcr_b:+.3f} %/day over {nd} days · runner exits eligible {int(V[(V.variant == 'BASE')].reason.isin(RUNNERS).sum())}\n"
    txt += Sm.drop(columns=["reasons"]).round(3).to_string(index=False) + "\n\n" + "\n".join(f"{r.variant}: {r.reasons}" for r in Sm.itertuples())
    open(OUT + ".txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
