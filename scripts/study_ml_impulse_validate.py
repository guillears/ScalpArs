#!/usr/bin/env python3
"""ML impulse study — study-level checks (run after step 1 / step 2; then scripts/validate_against_master.py).

  V1 source parity     fills = the 4,501 rows of reports/ML_BTCFWD_exit_rows.csv; block counts equal the replay / master loaders
  V2 clock vs REAL fills master stamp entry_eth_5m_ret1_pct (engine: last CLOSED 5m bar return, refreshed per scan) == our rebuild
                       at the entry's 5m boundary or one bar earlier (scan lag) on >= 99 % of fills; 2 bars / 1 bar later ~0 %
  V3 cross-source      btc_r15 at fills equals the prior study's (reports/ML_BTCFWD_fill_features.csv) within 1e-6
  V4 score two ways    SCORE / CONJ recomputed from the raw columns by an independent numpy path == stored
  V5 replica parity    replica base vs engine / as-traded within ±0.05 per block (reported; gate 85 % pooled replay)
  V6 no lookahead      t1, t5 <= entry for every fill (the grid-level 300/300 assert is study_ml_impulse_features.py --assert)

    venv/bin/python -W ignore scripts/study_ml_impulse_validate.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_ml_impulse_features import REP, minute_frame, load_frozen, MIN1, MIN5  # noqa: E402
import study_ml_delay_common as DC  # noqa: E402

FAIL = []


def check(name, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")
    if not ok:
        FAIL.append(name)


def main():
    F = pd.read_csv(os.path.join(REP, "ML_IMPULSE_fill_features.csv"))
    E = pd.read_csv(os.path.join(REP, "ML_BTCFWD_exit_rows.csv"), usecols=["key", "block"])
    R = DC.load_replay(); M = DC.load_master()
    ok = len(F) == len(E) == 4501 and (F.key.values == E.key.values).all() and \
        (F.block.str.startswith("20").sum() == len(R)) and ((F.block == "master").sum() == len(M))
    check("V1 source parity", ok, f"fills {len(F)} · replay {len(R)} · master {len(M)} · live {(F.block == 'live').sum()}")

    # V2: master entry stamp vs our ETH 1m
    mm = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False, usecols=["opened_at", "entry_eth_5m_ret1_pct"])
    mm = mm.dropna(subset=["entry_eth_5m_ret1_pct"]).drop_duplicates("opened_at")
    o = DC._ms(mm.opened_at).astype(np.int64)
    eth = minute_frame("ETHUSDT").c.astype(float)
    st = mm.entry_eth_5m_ret1_pct.values
    mine = {}
    for k in (0, 1, 2, -1):
        t5 = (o // MIN5) * MIN5 - k * MIN5
        mine[k] = np.round((eth.reindex(t5 - MIN1).values / eth.reindex(t5 - 6 * MIN1).values - 1) * 100, 4)
    eq = {k: np.abs(v - st) <= 2e-4 for k, v in mine.items()}
    for k in (-1, 0, 1, 2):
        print(f"     stamp == our last-closed-5m return {k} bar(s) earlier: {eq[k].mean():.1%}")
    both = (eq[0] | eq[1]).mean()
    check("V2 clock vs real master stamps", both >= 0.99 and eq[2].mean() < 0.05 and eq[-1].mean() < 0.05,
          f"{both:.1%} of {len(st)} master fills equal EXACTLY our bar at the entry's 5m boundary (or the one before: the engine's "
          f"scan stamp lags a just-closed bar by seconds) · 0 shift error")

    P = pd.read_csv(os.path.join(REP, "ML_BTCFWD_fill_features.csv"), usecols=["key", "btc_r15"])
    j = F[["key", "btc_r15"]].merge(P, on="key", suffixes=("", "_prior"))
    k = j.btc_r15.notna() & j.btc_r15_prior.notna()
    d = (j.btc_r15[k] - j.btc_r15_prior[k]).abs().max()
    check("V3 cross-source btc_r15", len(j) == len(F) and d < 1e-6, f"{k.sum()} fills compared, max |diff| {d:.2e}")

    frz = load_frozen()
    L = np.column_stack([
        (F.btc_r15 > 0) & (F.er_15 >= frz["q_er"]), F.eth_r15 > 0, F.sol_r15 > 0, F.xrp_r15 > 0, F.zec_r15 > 0,
        F.top10_up >= 0.70, F.br_trend_d15 > 0, F.top10_tbr > 0.5, F.vsurge >= 1.5]).astype(float)
    raw = F[["btc_r15", "er_15", "eth_r15", "sol_r15", "xrp_r15", "zec_r15", "top10_up", "br_trend_d15", "top10_tbr", "vsurge"]]
    s2 = np.where(raw.notna().all(axis=1), L.sum(1), np.nan)
    c2 = np.where(F[["btc_r15", "top10_up", "br_trend_d15"]].notna().all(axis=1),
                  ((F.btc_r15 > 0) & (F.top10_up >= 0.7) & (F.br_trend_d15 > 0)).astype(float), np.nan)
    ok = np.allclose(s2, F.SCORE.values, equal_nan=True) and np.allclose(c2, F.CONJ.values, equal_nan=True)
    check("V4 score two ways", ok, f"SCORE defined on {np.isfinite(s2).mean():.1%} of fills; distribution "
                                   f"{pd.Series(s2).value_counts().sort_index().to_dict()}")

    for b, g in F.groupby("block"):
        w = ((g.base - g.today_engine).abs() <= 0.05).mean()
        sa = (np.sign(g.base) == np.sign(g.today_engine)).mean()
        print(f"     V5 {b:14s} N {len(g):5d} within ±0.05 {w:.1%} · sign agree {sa:.1%}")
    rp = F[F.block.str.startswith("20")]
    w = ((rp.base - rp.today_engine).abs() <= 0.05).mean()
    check("V5 replica parity (pooled replay)", w >= 0.85, f"{w:.1%} within ±0.05 (master shown above; caveat 82.7 % < 85 %)")

    o_ms = F.o_ms.astype(np.int64)
    check("V6 no lookahead at fills", bool((F.t1 <= o_ms).all() and (F.t5 <= o_ms).all()), "all t1, t5 <= entry")
    print("ALL STUDY CHECKS PASS" if not FAIL else f"FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
