#!/usr/bin/env python3
"""RUNNER-MARKET (fresh-trend second chance) — BACKTEST cross-check (master found: second chance d 0.3 only when the coin was NOT in an
established uptrend at entry — EMA13 ≤ EMA50 (entry_pair_ema20_ema50_gap_pct, misnamed: it holds EMA13−EMA50) / ADX falling / EMA50 slope
≤ 0 — all 5 era groups positive, coherent-pattern null p 0.00). Same zones on the 2-year replay ML fills. Writes
reports/ML_RUNNER_REGIME_2026-11-11_bt.txt"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import study_ml_exit_macro_1111 as S, study_ml_exit_macro_1111_report as R, study_ml_second_chance_1111_bt as SB  # noqa


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
        for r in ex.map(SB._work, ch): rows += r
    V = pd.DataFrame(rows); W = V.pivot(index="key", columns="variant", values="pct").dropna()
    M = m.drop_duplicates("key").set_index("key").loc[W.index]
    day = pd.Series(W.index.str[3:13], index=W.index).values; half = np.where(day < "2026-07-01", "26-H1", "26-H2")
    b = W.BASE.values; sc = W["SC_d0.3"].values; ch_ = np.abs(sc - b) > 1e-9
    L = [f"BACKTEST {len(W)} replay ML fills · BASE avg {b.mean():+.3f} · SC everywhere Δ {(sc - b).mean():+.3f}/fill (changed {int(ch_.sum())})"]
    for name, col in (("EMA13≤EMA50", "entry_pair_ema20_ema50_gap_pct"), ("ADX falling", "entry_adx_delta"), ("EMA50 slope≤0", "entry_ema50_slope")):
        v = pd.to_numeric(M[col], errors="coerce").values; z = v <= 0
        dl = np.where(z, sc, b) - b; lo, hi, _ = R.boot_ci(dl, day, 0.05)
        zi, zo = z & ch_, (~z) & (v == v) & ch_
        L.append(f"SC only if {name}: zone {int(z.sum())} (changed {int(zi.sum())}, scored {int((v == v).sum())}) · sleeve Δ {dl.mean():+.3f}/fill [{lo:+.3f}, {hi:+.3f}] · "
                 f"Δ per changed fill: zone {(sc - b)[zi].mean():+.3f} vs outside {(sc - b)[zo].mean():+.3f} · "
                 + " · ".join(f"{h} zone {(sc - b)[zi & (half == h)].mean():+.3f} (N {int((zi & (half == h)).sum())}) / out {(sc - b)[zo & (half == h)].mean():+.3f}" for h in ("26-H1", "26-H2")))
    open(os.path.join(S.ROOT, "reports", "ML_RUNNER_REGIME_2026-11-11_bt.txt"), "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
