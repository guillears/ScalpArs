#!/usr/bin/env python3
"""ML EXIT × MACRO (2026-11-11) — the 10-26 ML EXIT DEEP grid (scripts/study_ml_exit_b25_run.VARIANTS, 36 variants + 3 partials, its own
hooked evaluator, BASE bit-exact to RP.evaluate) RE-RUN on today's filtered cohort (operator 11-11: "the universe of study changed").
Same cohort / paths as study_ml_exit_macro_1111 (no fetch — run after it). Writes reports/ML_EXIT_MACRO_2026-11-11_oldgrid_variants.csv."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study_ml_exit_b25_run as OG  # noqa: E402  (patches RP readers with the 10-26 top-up)
import study_ml_exit_macro_1111 as S  # noqa: E402
import scout_ml_matched_refusals as MM  # noqa: E402


def main():
    C, _ = S.cohort()
    rows = []
    with MM.replica_with_topup() as RP:
        for r in C.itertuples():
            rsi = RP.btc_rsi_at(int(r.o_ms))
            fx = dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in),
                      atr=float(r.atr) if (r.atr == r.atr and r.atr > 0) else None, rsi_entry=None if rsi is None else float(rsi))
            res, _ = OG.run_fill(fx, OG.VARIANTS)
            for v, (why, pct, tex, src) in res.items():
                rows.append(dict(key=r.key, variant=v, reason=why, pct=pct, exit_ms=tex, src=src))
    V = pd.DataFrame(rows)
    W = V.pivot(index="key", columns="variant", values="pct")
    T = V.pivot(index="key", columns="variant", values="exit_ms")
    for p, h in OG.PARTIALS.items():          # half today's exit, half the wider variant
        for k in W.index:
            rows.append(dict(key=k, variant=p, reason="PARTIAL", pct=(W.at[k, "BASE"] + W.at[k, h]) / 2,
                             exit_ms=max(T.at[k, "BASE"], T.at[k, h]) if T.at[k, h] == T.at[k, h] else T.at[k, "BASE"], src=""))
    V = pd.DataFrame(rows)
    V = V[~V.variant.str.startswith("_H")]
    V.to_csv(os.path.join(S.ROOT, "reports", "ML_EXIT_MACRO_2026-11-11_oldgrid_variants.csv"), index=False)
    print("oldgrid rows", len(V), "fills", V.key.nunique())


if __name__ == "__main__":
    main()
