#!/usr/bin/env python3
"""STOP −3 × TRAIL FROM +1.5 — BACKTEST cross-check (operator 11-11: "run the backtest for stop −3 with trail from 1.5"). Variants from
study_ml_sl3_late_trail_1111 (L3_a1.5_* with / without the clock) + refs BASE and WS_sl-3, on the 2-year today's-stack replay ML fills
(ML_BULL90 applied, ML_MKT_HOT not — caveat; cache paths only). Writes reports/ML_SL3_LATE_TRAIL_2026-11-11_bt.txt"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S, study_ml_exit_macro_1111_report as R, study_ml_sl3_late_trail_1111 as LT, scout_ml_matched_refusals as MM  # noqa


def keep(k):
    return k in ("BASE", "WS_sl-3") or k.startswith("L3_a1.5_")


def _work(fxs):
    G = {k: v for k, v in LT.grid().items() if keep(k)}
    rows = []
    with MM.replica_with_topup() as RP:
        B = S._btc_frames(RP)
        for fx in fxs:
            try:
                rsi = RP.btc_rsi_at(int(fx["o_ms"])); fx = dict(fx, rsi_entry=None if rsi is None else float(rsi))
                out, _ = S.price_fill(RP, fx, G, B)
            except Exception:
                out = None
            if out:
                for v, (why, pct, tex, _p) in out.items():
                    rows.append(dict(key=fx["key"], variant=v, reason=why, pct=pct))
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
    V = pd.DataFrame(rows)
    W = V.pivot(index="key", columns="variant", values="pct"); RR = V.pivot(index="key", columns="variant", values="reason")
    ok = W.notna().all(axis=1) & (RR.BASE != "END_OF_DATA"); W, RR = W[ok], RR[ok]
    day = pd.Series(W.index.str[3:13], index=W.index); half = np.where(day < "2026-01-01", "2025", np.where(day < "2026-07-01", "26-H1", "26-H2"))
    b = W.BASE.values
    L = [f"BACKTEST STOP −3 × TRAIL FROM +1.5 — {len(W)} replay ML fills (of {len(m)}; cache paths only) · BASE avg {b.mean():+.3f} WR {(b > 0).mean():.0%}"]
    for v in [c for c in W.columns if c != "BASE"]:
        y = W[v].values; dl = y - b; lo, hi, _ = R.boot_ci(dl, day.values, 0.05); r = RR[v].astype(str)
        L.append(f"{v}: WR {(y > 0).mean():.0%} avg {y.mean():+.3f} · Δ {dl.mean():+.3f}/fill [{lo:+.3f}, {hi:+.3f}] · −3 hits {int(r.str.contains('STOP').sum())} · "
                 f"Δ winners {dl[b > 0].sum():+.1f} · Δ losers {dl[b <= 0].sum():+.1f} · "
                 + " · ".join(f"{h} {dl[half == h].mean():+.3f} (N {(half == h).sum()})" for h in ("26-H1", "26-H2")))
    open(os.path.join(S.ROOT, "reports", "ML_SL3_LATE_TRAIL_2026-11-11_bt.txt"), "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
