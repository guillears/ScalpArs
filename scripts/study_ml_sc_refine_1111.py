#!/usr/bin/env python3
"""ML SECOND-CHANCE — REFINEMENT grid on the ADX-falling zone (2026-11-11, operator: "I like the general idea, but there is opportunity
to refine it"). Zone = entry_adx_delta ≤ 0 (others keep today's exit). Phase 2 starts where today's RUNNER_TRAIL / HARD_TP_LADDER fires at L.
Pre-declared knobs (one at a time around the reference SC_d0.3 · trail ATR×1 · live clock):
  floor  d = L − 0.2 / 0.3 / 0.4 · ATR-scaled L − ATR×0.5 / ×1.0
  trail  k = ATR × 0.5 / 1.0 / 1.5 / 2.0 / none (floor only)
  guard  BE: the floor never below +0.05 (a winner never turns into a loser)
  ratchet: once the second leg reaches L + 0.50, the floor moves up to L (today's exit is locked)
  clock  live (180-min no-expansion / 20 h from entry) vs EXT (+120 min after the trigger)
Bonferroni over the grid. Read on the 123 kept master fills (all) + B25; day-block CI; one-size DCR."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S  # noqa
import study_ml_exit_macro_1111_report as R  # noqa
import scout_ml_matched_refusals as MM  # noqa
import review_math as RM  # noqa
OUT = os.path.join(S.ROOT, "reports", "ML_SC_REFINE_2026-11-11"); MIN = 60_000
RUNNERS = ("RUNNER_TRAIL", "HARD_TP_LADDER", "RH_RUNNER_TRAIL", "RH_HARD_TP_LADDER")
G = {  # name: (d_fixed, d_atr, k, be, ratchet, ext_min)
    "REF_d0.3_k1": (0.3, None, 1.0, False, False, None),
    "d0.2_k1": (0.2, None, 1.0, False, False, None), "d0.4_k1": (0.4, None, 1.0, False, False, None),
    "dATR0.5_k1": (None, 0.5, 1.0, False, False, None), "dATR1_k1": (None, 1.0, 1.0, False, False, None),
    "d0.3_k0.5": (0.3, None, 0.5, False, False, None), "d0.3_k1.5": (0.3, None, 1.5, False, False, None),
    "d0.3_k2": (0.3, None, 2.0, False, False, None), "d0.3_kNone": (0.3, None, None, False, False, None),
    "d0.3_k1_BE": (0.3, None, 1.0, True, False, None), "d0.3_k1_RATCHET": (0.3, None, 1.0, False, True, None),
    "d0.3_k1_EXT120": (0.3, None, 1.0, False, False, 120),
}


def sc(P, fx, k0, prm, clock_ms):
    d_fixed, d_atr, k, be, ratchet, ext = prm
    t, net = P["t"], P["net"]; L = float(net[k0]); atr = fx["atr"]
    d = d_fixed if d_fixed is not None else (atr * d_atr if atr else 0.3)
    floor = L - d
    if be:
        floor = max(floor, 0.05) if L > 0.05 else floor
    g = (atr * k) if (k is not None and atr) else np.inf
    m2 = L
    lim = clock_ms if ext is None else max(clock_ms, t[k0] - fx["o_ms"] + ext * MIN)
    for i in range(k0 + 1, len(t)):
        v = net[i]; m2 = max(m2, v)
        if ratchet and m2 >= L + 0.5:
            floor = max(floor, L)
        if v <= floor:
            return i, "SC_FLOOR"
        if v <= m2 - g:
            return i, "SC_TRAIL"
        if t[i] - fx["o_ms"] >= lim:
            return i, "SC_CLOCK"
    return len(t) - 1, "END_OF_DATA"


def main():
    C, _ = S.cohort(); C = C.set_index("key")
    rows = []
    with MM.replica_with_topup() as RP:
        clock = min(RP.BASE["noexp"], RP.BASE["maxhold"]) * MIN
        for key, r in C.iterrows():
            rsi = RP.btc_rsi_at(int(r.o_ms))
            fx = dict(key=key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in),
                      atr=float(r.atr) if (r.atr == r.atr and r.atr > 0) else None, rsi_entry=None if rsi is None else float(rsi))
            P = RP.prep_path(fx, RP.HORIZON_MS)
            if P is None:
                continue
            k0, why, pct = RP.evaluate(P, fx, RP.BASE)
            rows.append(dict(key=key, variant="BASE", reason=why, pct=pct, exit_ms=int(P["t"][k0])))
            zone = float(r.entry_adx_delta) <= 0
            for v, prm in G.items():
                if not zone or why not in RUNNERS:
                    rows.append(dict(key=key, variant=v, reason=why, pct=pct, exit_ms=int(P["t"][k0]))); continue
                i, w = sc(P, fx, k0, prm, clock)
                rows.append(dict(key=key, variant=v, reason=w, pct=float(P["net"][i]), exit_ms=int(P["t"][i])))
    V = pd.DataFrame(rows); V.to_csv(OUT + "_variants.csv", index=False)
    W = V.pivot(index="key", columns="variant", values="pct"); X = V.pivot(index="key", columns="variant", values="exit_ms")
    F = pd.read_csv(os.path.join(S.ROOT, "reports", "ML_EXIT_MACRO_2026-11-11_fills.csv")).set_index("key").loc[W.index].reset_index()
    F["day"] = pd.to_datetime(F.o_ms, unit="ms").dt.strftime("%Y-%m-%d"); F["grp"] = F.era.map(R.GROUPS)
    b = W.BASE.values; out = []
    for scope, m in (("MASTER", np.ones(len(F), bool)), ("B25", (F.era == "B25").values)):
        eras = sorted(F.era[m].unique())
        db, nd = R.dcr_group(F[m].assign(pp=b[m], ex=X.BASE.values[m]), "pp", "ex", eras)
        for v in G:
            y = W[v].values; dl = (y - b)[m]
            lo, hi, _ = R.boot_ci(dl, F.day.values[m], 0.05)
            lB, hB, _ = R.boot_ci(dl, F.day.values[m], 0.05 / len(G))
            da, _ = R.dcr_group(F[m].assign(pp=y[m], ex=X[v].values[m]), "pp", "ex", eras)
            dd = pd.Series(dl).groupby(F.day.values[m]).sum()
            out.append(dict(scope=scope, variant=v, n=int(m.sum()), tot_before=b[m].sum(), tot_after=y[m].sum(), d_tot=dl.sum(),
                            d_avg=dl.mean(), lo95=lo, hi95=hi, loBonf=lB, hiBonf=hB, better=int((dl > 1e-9).sum()), worse=int((dl < -1e-9).sum()),
                            worst=dl.min(), win_pts=dl[b[m] > 0].sum(), los_pts=dl[b[m] <= 0].sum(), wr_after=(y[m] > 0).mean(),
                            dcr_before=db, dcr_after=da, top_day=(dd.max() / dl.sum() if dl.sum() > 0 else np.nan),
                            **({f"d_{g}": (y - b)[(F.grp == g).values].sum() for g in sorted(F.grp.unique())} if scope == "MASTER" else {})))
    Sm = pd.DataFrame(out); Sm.to_csv(OUT + "_summary.csv", index=False)
    txt = Sm.round(3).to_string(index=False); open(OUT + ".txt", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
