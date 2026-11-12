#!/usr/bin/env python3
"""SECOND-CHANCE REFINEMENT — BACKTEST of the same 12-variant grid (study_ml_sc_refine_1111) on the 2-year today's-stack replay ML fills
(ENGINE_REPLAY_YR6W_2Y_TODAY, ML_BULL90 applied, ML_MKT_HOT not — caveat), cache paths only. Zone = entry_adx_delta ≤ 0, others keep
today's exit. Writes reports/ML_SC_REFINE_2026-11-11_bt.txt"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S, study_ml_exit_macro_1111_report as R, study_ml_sc_refine_1111 as SR, scout_ml_matched_refusals as MM  # noqa
MIN = 60_000


def _work(fxs):
    rows = []
    with MM.replica_with_topup() as RP:
        clock = min(RP.BASE["noexp"], RP.BASE["maxhold"]) * MIN
        for fx in fxs:
            try:
                rsi = RP.btc_rsi_at(int(fx["o_ms"])); fx = dict(fx, rsi_entry=None if rsi is None else float(rsi))
                P = RP.prep_path(fx, RP.HORIZON_MS)
                if P is None: continue
                k, why, pct = RP.evaluate(P, fx, RP.BASE)
                if why == "END_OF_DATA": continue
                rows.append(dict(key=fx["key"], variant="BASE", pct=pct))
                for v, prm in SR.G.items():
                    if not fx["zone"] or why not in SR.RUNNERS:
                        rows.append(dict(key=fx["key"], variant=v, pct=pct)); continue
                    i, _ = SR.sc(P, fx, k, prm, clock)
                    rows.append(dict(key=fx["key"], variant=v, pct=float(P["net"][i])))
            except Exception:
                pass
    return rows


def main():
    d = pd.read_csv(os.path.join(S.ROOT, "reports", "ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv"), low_memory=False)
    m = d[(d.entry_strategy == "MOMENTUM") & (d.direction == "LONG") & (d.is_probe != True)].copy()  # noqa: E712
    m = m[~(pd.to_numeric(m.entry_bull_pct, errors="coerce") >= 90)]
    m["o_ms"] = S._ms(m.opened_at); m["key"] = "bt|" + m.opened_at.astype(str).str[:19] + "|" + m.pair
    m["adx_d"] = pd.to_numeric(m.entry_adx_delta, errors="coerce")
    RP = S._rp()
    fx = [dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.entry_price), zone=bool(r.adx_d <= 0),
               fee_in=RP.MAKER if str(r.entry_order_type).upper().startswith("MAKER") else RP.TAKER,
               atr=float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) and r.entry_atr_pct > 0 else None) for r in m.itertuples()]
    ch = [sorted(fx[i::40], key=lambda f: (f["pair"], f["o_ms"])) for i in range(40)]
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(_work, ch): rows += r
    V = pd.DataFrame(rows); W = V.pivot(index="key", columns="variant", values="pct").dropna()
    day = pd.Series(W.index.str[3:13], index=W.index).values; half = np.where(day < "2026-07-01", "26-H1", "26-H2")
    b = W.BASE.values; K = list(SR.G)
    L = [f"BACKTEST SC-REFINE — {len(W)} replay ML fills priced (of {len(m)}) · BASE total {b.sum():+.1f} avg {b.mean():+.3f} WR {(b > 0).mean():.0%} · "
         f"zone (ADX falling) {int(m.drop_duplicates('key').set_index('key').loc[W.index].adx_d.le(0).sum())}"]
    L.append(f"{'variant':18s} {'total':>7s} {'Δtot':>6s} {'Δ/fill':>7s} {'95% CI':>17s} {'Bonf CI':>17s} better/worse  worst  WR   26-H1 Δtot  26-H2 Δtot")
    for v in K:
        y = W[v].values; dl = y - b
        lo, hi, _ = R.boot_ci(dl, day, 0.05); lB, hB, _ = R.boot_ci(dl, day, 0.05 / len(K))
        L.append(f"{v:18s} {y.sum():+7.1f} {dl.sum():+6.1f} {dl.mean():+7.3f} [{lo:+.3f}, {hi:+.3f}] [{lB:+.3f}, {hB:+.3f}] "
                 f"{int((dl > 1e-9).sum()):4d}/{int((dl < -1e-9).sum()):<4d} {dl.min():+.2f} {(y > 0).mean():.0%} "
                 f"{dl[half == '26-H1'].sum():+8.1f} {dl[half == '26-H2'].sum():+10.1f}")
    open(os.path.join(S.ROOT, "reports", "ML_SC_REFINE_2026-11-11_bt.txt"), "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
