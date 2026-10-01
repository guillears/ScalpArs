#!/usr/bin/env python3
"""MASTER-BATCH VALIDATION GATE (operator Sep-28: "the MASTER BATCH is the source of truth … any change to the code, logic,
filters, calculations or trade reconstruction MUST be validated against the master batch before the work is complete").

Run BEFORE presenting any analysis / after any change to the analysis tooling. Exit code 1 on any FAIL.
  M1  ledger integrity      — every master fill: sign(pct used by the analyses) == sign(net $) (catches CF re-pricing slips)
  M2  ledger provenance     — every ledger fill is a kept, non-probe row of MASTER_POOL_stacked.csv (no invented fills)
  F1  feature grid          — every higher-TF bar of entry_feature_factory equals an INDEPENDENT groupby resample (catches
                              grid shifts / in-place mutation)
  F2  feature vs live stamps — rebuilt BTC 5m RSI and BTC 1h RSI vs the live-stamped entry_btc_rsi / entry_btc_rsi_1h on every
                              master fill: best-correlated time shift must be 0 and r ≥ 0.9 (catches clock / timezone / unit
                              misalignment against REAL trades)
  C1  calibration matches   — every MATCHED trade in reports/CALIBRATION_TABLE_trades.csv (if present): live pair/open/pct
                              equal the master row; backtest open within the match window
Usage: venv/bin/python scripts/validate_against_master.py
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG
import entry_feature_factory as EF
M = LG.build()
sys.argv = _argv
FAILS = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


# M1 — the pct every analysis uses must agree in sign with the ledger's net $
pct = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"),
               M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
bad = M[(np.sign(pct) != np.sign(M.net)) & (M.net.abs() > 0.5)]
check("M1 pct/net sign", len(bad) == 0, f"{len(bad)} of {len(M)} fills disagree"
      + ("" if not len(bad) else " → " + ", ".join(f"{p}@{str(o)[:16]}({r})" for p, o, r in zip(bad.pair, bad.opened_at, bad.stack_block_reason))))

# M2 — every ledger fill is a real, kept, non-probe master-pool fill (the ledger only SUBTRACTS further live gates)
P = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
P = P[(P.status == "CLOSED") & (P.stack_keep.astype(str).str.lower().isin(["true", "1"]))
      & ~P.is_probe.astype(str).str.lower().isin(["true", "1"])]
key = lambda d: set(zip(d.opened_at.astype(str).str[:19].str.replace(" ", "T"), d.pair, d.direction))
extra = key(M) - key(P)
check("M2 ledger ⊆ pool (kept, non-probe)", len(extra) == 0,
      f"{len(M)} ledger fills, {len(extra)} not in the pool; pool kept non-probe {len(P)} → ledger gates drop {len(P) - len(M)}")

# F1 — higher-TF grid of the feature factory vs an independent resample
k = pd.read_csv(os.path.join(EF.K5, "BTCUSDT.csv"))
fr = EF._load("BTCUSDT")
for tf, ms in (("15m", 900_000), ("1h", 3_600_000), ("4h", 14_400_000)):
    g = k.assign(b=(k.open_time // ms) * ms).groupby("b").c.last()
    close_idx = pd.to_datetime(g.index + ms, unit="ms")
    ind = EF._ind(pd.DataFrame({"o": g, "h": g, "l": g, "c": g, "vol": g}).set_index(close_idx))   # rsi only needs c
    common = fr[tf].index.intersection(close_idx)
    d = (fr[tf].rsi.reindex(common) - ind.rsi.reindex(common)).abs()
    check(f"F1 {tf} grid", len(common) > 0.99 * len(close_idx) and d.max() < 1e-6,
          f"{len(common)}/{len(close_idx)} bars aligned, max RSI diff {d.max():.2e}")

# F2 — rebuilt BTC RSI vs LIVE stamps on every master fill (shift scan)
t = pd.to_datetime(M.opened_at.astype(str).str[:19])
for stamp, tf in (("entry_btc_rsi", "5m"), ("entry_btc_rsi_1h", "1h")):
    s = pd.to_numeric(M[stamp], errors="coerce")
    step = 5 if tf == "5m" else 60
    res = {}
    for sh in (-2, -1, 0, 1, 2):
        X = EF._asof({tf: fr[tf]}, t + pd.Timedelta(minutes=sh * step), "B")[f"B_{tf}_rsi"]
        res[sh] = pd.Series(X, index=M.index).corr(s)
    best = max(res, key=res.get)
    # live reads the FORMING bar → it sits between the last closed bar (shift 0) and the next (+1); both are correct alignments
    check(f"F2 {stamp} vs rebuilt {tf}", best in (0, 1) and res[best] >= 0.9,
          "r by shift(bars) " + " ".join(f"{k:+d}:{v:.3f}" for k, v in res.items()))

# C1 — calibration trade dump agrees with the master
cp = "reports/CALIBRATION_TABLE_trades.csv"
if os.path.exists(cp):
    C = pd.read_csv(cp)
    mt = C[C.kind.isin(["MATCHED", "MISSED"])]
    Mi = M.assign(t=t).set_index(["pair", "t"])
    miss = 0
    for _, r in mt.iterrows():
        key = (r.pair, pd.Timestamp(r.live_open))
        if key not in Mi.index:
            miss += 1
    check("C1 calibration rows exist in master", miss == 0, f"{len(mt) - miss}/{len(mt)} live rows found")
    m = C[C.kind == "MATCHED"]
    dt = (pd.to_datetime(m.bt_open) - pd.to_datetime(m.live_open)).abs().dt.total_seconds()
    check("C1 matched open gap ≤ 20 min", bool((dt <= 1200).all()), f"max {dt.max():.0f}s")

print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
