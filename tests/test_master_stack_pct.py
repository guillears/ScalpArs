"""stack_pct (the P&L % under today's rules) must not move on size-only re-prices — ticket CAP05, sprint de-mux, UNMATCHED 1.5×,
flip cells → 1×, sleeve sizing. Only path CFs (ARM040 / LATE_ARM / FADE_SL) and the FRENZY +TP re-price change it. The old
replay_fidelity_audit ratio (pnl% × stack_pnl / pnl) halved the pct on 79 re-sized master rows. Data-integrity test on the
committed reports/MASTER_POOL_stacked.csv (regenerate it after any builder change)."""
import json, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _master():
    return pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)


def test_stack_pct_ignores_size_reprices():
    M = _master()
    k = M[M.stack_keep == True]
    assert "stack_pct" in M and k.stack_pct.notna().all() and M[M.stack_keep != True].stack_pct.isna().all()
    path = k.stack_block_reason.fillna("").astype(str).str.contains("ARM040|LATE_ARM|FADE_SL")
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))
    tp = float((th.get("thresholds", th)).get("frenzy_tp_pct", 0) or 0)
    frenzy_tp = k.entry_strategy.astype(str).str.startswith("FRENZY") & (tp > 0) & (pd.to_numeric(k.peak_pnl, errors="coerce") >= tp)
    assert np.allclose(k.stack_pct[frenzy_tp], tp)                     # FRENZY rows that reached the TP book the TP
    plain = k[~path & ~frenzy_tp]
    assert np.allclose(plain.stack_pct, plain.pnl_percentage)          # size-only rows keep the live pct
    resized = plain[(plain.stack_pnl - plain.pnl).abs() > 0.05]        # e.g. 2× rows re-priced to 1× / 1.5×
    assert len(resized) > 0 and np.allclose(resized.stack_pct, resized.pnl_percentage)
    exp = k.stack_pnl[path] / k.stack_ticket_scale[path].fillna(1) / k.notional_value[path] * 100
    assert len(exp) > 0 and np.allclose(k.stack_pct[path], exp, atol=1e-3)


def test_audit_reads_stack_pct():
    src = open(os.path.join(ROOT, "scripts", "replay_fidelity_audit.py")).read()
    assert "LC.stack_pct" in src and "LC.stack_pnl / LC.stack_ticket_scale" not in src
