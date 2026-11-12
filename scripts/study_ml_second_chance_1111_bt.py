#!/usr/bin/env python3
"""SECOND CHANCE exit — BACKTEST cross-check (run because SC beat today's exit on master, operator path 11-11). Same rule as
study_ml_second_chance_1111 on the 2-year today's-stack replay ML fills (ENGINE_REPLAY_YR6W_2Y_TODAY, ML_BULL90 applied, ML_MKT_HOT not —
caveat), cache paths only. Writes reports/ML_SECOND_CHANCE_2026-11-11_bt.txt"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S, study_ml_exit_macro_1111_report as R, study_ml_second_chance_1111 as SC, scout_ml_matched_refusals as MM  # noqa
MIN = 60_000


def _work(fxs):
    rows = []
    with MM.replica_with_topup() as RP:
        for fx in fxs:
            try:
                rsi = RP.btc_rsi_at(int(fx["o_ms"])); fx = dict(fx, rsi_entry=None if rsi is None else float(rsi))
                P = RP.prep_path(fx, RP.HORIZON_MS)
                if P is None: continue
                k, why, pct = RP.evaluate(P, fx, RP.BASE)
                if why == "END_OF_DATA": continue
                rows.append(dict(key=fx["key"], variant="BASE", pct=pct, src=P["src"]))
                clock = min(RP.BASE["noexp"], RP.BASE["maxhold"]) * MIN
                for d in SC.DS:
                    v = f"SC_d{d:g}"
                    if why not in SC.RUNNERS:
                        rows.append(dict(key=fx["key"], variant=v, pct=pct, src=P["src"])); continue
                    i, w = SC.second_chance(P, fx, k, d, None, clock)
                    rows.append(dict(key=fx["key"], variant=v, pct=float(P["net"][i]), src=P["src"]))
            except Exception:
                pass
    return rows


def main():
    d = pd.read_csv(os.path.join(S.ROOT, "reports", "ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv"), low_memory=False)
    m = d[(d.entry_strategy == "MOMENTUM") & (d.direction == "LONG") & (d.is_probe != True)].copy()  # noqa: E712
    m = m[~(pd.to_numeric(m.entry_bull_pct, errors="coerce") >= 90)]
    m["o_ms"] = S._ms(m.opened_at); m["key"] = "bt|" + m.opened_at.astype(str).str[:19] + "|" + m.pair
    RP = S._rp()
    fx = [dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.entry_price), fee_in=RP.MAKER if str(r.entry_order_type).upper().startswith("MAKER") else RP.TAKER,
               atr=float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) and r.entry_atr_pct > 0 else None) for r in m.itertuples()]
    ch = [sorted(fx[i::40], key=lambda f: (f["pair"], f["o_ms"])) for i in range(40)]
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(_work, ch): rows += r
    V = pd.DataFrame(rows); W = V.pivot(index="key", columns="variant", values="pct").dropna()
    day = pd.Series(W.index.str[3:13], index=W.index); half = np.where(day < "2026-01-01", "2025", np.where(day < "2026-07-01", "26-H1", "26-H2"))
    b = W.BASE.values
    L = [f"BACKTEST SC — {len(W)} replay ML fills priced (of {len(m)}; cache paths only) · BASE avg {b.mean():+.3f} WR {(b > 0).mean():.0%}"]
    for v in [c for c in W.columns if c != "BASE"]:
        dl = W[v].values - b; lo, hi, _ = R.boot_ci(dl, day.values, 0.05)
        L.append(f"{v}: Δ {dl.mean():+.3f}/fill [{lo:+.3f}, {hi:+.3f}] · " + " · ".join(f"{h} {dl[half == h].mean():+.3f} (N {(half == h).sum()})" for h in ("2025", "26-H1", "26-H2")))
    open(os.path.join(S.ROOT, "reports", "ML_SECOND_CHANCE_2026-11-11_bt.txt"), "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
