#!/usr/bin/env python3
"""ML EXIT × MACRO (2026-11-11) — the 10-26 BTC-REACTIVE grid (scripts/study_ml_btcexit_run: 128 pre-declared rules × IDEAL / FAST / SLOW)
RE-RUN on today's filtered cohort (operator 11-11: "the universe of study changed"). Run study_ml_btcexit_fetch.py first (BTC 1m to now).
Writes reports/ML_EXIT_MACRO_2026-11-11_btcgrid_{fills,variants}.csv."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study_ml_btcexit_run as BX  # noqa: E402
import study_ml_exit_macro_1111 as S  # noqa: E402
import scout_ml_matched_refusals as MM  # noqa: E402


def main():
    C, _ = S.cohort()
    R, O = [], []
    with MM.replica_with_topup() as RP:
        for r in C.itertuples():
            rsi = RP.btc_rsi_at(int(r.o_ms))
            fx = dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in),
                      atr=float(r.atr) if (r.atr == r.atr and r.atr > 0) else None, rsi_entry=None if rsi is None else float(rsi))
            try:
                a, b = BX.run_fill(fx)
            except Exception as e:
                a, b = dict(key=r.key, err=str(e)[:80]), []
            R.append(a); O.extend(b)
    out = os.path.join(S.ROOT, "reports", "ML_EXIT_MACRO_2026-11-11_btcgrid")
    pd.DataFrame(R).to_csv(out + "_fills.csv", index=False)
    pd.DataFrame(O, columns=["key", "variant", "scen", "changed", "reason", "base_pct", "pct", "exit_min"]).to_csv(out + "_variants.csv", index=False)
    print("btcgrid fills", len(R), "btc src", pd.DataFrame(R).btc_src.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
