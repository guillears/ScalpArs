#!/usr/bin/env python3
"""ML EXIT × MACRO (2026-11-11) — BACKTEST cross-check (refute-only, MASTER OVER BACKTEST): the same pre-declared grid on the 2-year
today's-stack replay ML fills (reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv, 2025-01 → 2026-10-11, seed 1), ML_BULL90 applied
(entry_bull_pct ≥ 90 out); ML_MKT_HOT NOT applied (needs the rolling top-100 rebuild over 2 years — a stated caveat). Cache paths only,
no fetch. Parallel workers. Writes reports/ML_EXIT_MACRO_2026-11-11_bt_{variants,fills}.csv."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S  # noqa: E402
import scout_ml_matched_refusals as MM  # noqa: E402

OUT = os.path.join(ROOT, "reports", "ML_EXIT_MACRO_2026-11-11_bt")


def _work(fxs):
    G = S.variants()
    rows, facts = [], []
    with MM.replica_with_topup() as RP:
        B = S._btc_frames(RP)
        for fx in fxs:
            try:
                rsi = RP.btc_rsi_at(int(fx["o_ms"]))
                fx = dict(fx, rsi_entry=None if rsi is None else float(rsi))
                out, info = S.price_fill(RP, fx, G, B)
            except Exception as e:
                out, info = None, dict(err=str(e)[:80])
            if out is None:
                facts.append(dict(key=fx["key"], src="none", **(info or {})))
                continue
            for v, (why, pct, tex, pkx) in out.items():
                rows.append(dict(key=fx["key"], variant=v, reason=why, pct=pct, exit_ms=tex))
            facts.append(dict(key=fx["key"], **info))
    return rows, facts


def main():
    d = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv"), low_memory=False)
    m = d[(d.entry_strategy == "MOMENTUM") & (d.direction == "LONG") & (d.is_probe != True)].copy()  # noqa: E712
    m = m[~(pd.to_numeric(m.entry_bull_pct, errors="coerce") >= 90)]
    m["o_ms"] = S._ms(m.opened_at)
    m["key"] = "bt|" + m.opened_at.astype(str).str[:19] + "|" + m.pair
    RP = S._rp()
    fx = [dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.entry_price),
               fee_in=RP.MAKER if str(r.entry_order_type).upper().startswith("MAKER") else RP.TAKER,
               atr=float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) and r.entry_atr_pct > 0 else None) for r in m.itertuples()]
    fx.sort(key=lambda f: (f["pair"], f["o_ms"]))
    chunks = [fx[i::40] for i in range(40)]
    chunks = [sorted(c, key=lambda f: (f["pair"], f["o_ms"])) for c in chunks]
    R, F = [], []
    with ProcessPoolExecutor(8) as ex:
        for rows, facts in ex.map(_work, chunks):
            R += rows; F += facts
    pd.DataFrame(R).to_csv(OUT + "_variants.csv", index=False)
    m[["key", "opened_at", "closed_at", "pair", "pct_new", "pnl_percentage", "entry_bull_pct"]].merge(pd.DataFrame(F), on="key", how="left") \
        .to_csv(OUT + "_fills.csv", index=False)
    print("bt fills", len(m), "priced", pd.DataFrame(R).key.nunique() if R else 0)


if __name__ == "__main__":
    main()
