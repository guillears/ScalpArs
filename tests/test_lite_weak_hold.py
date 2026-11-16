"""🧱 (DECISION_LOG 556) LITE_WEAK_HOLD — FRENZY_LITE refused when the price spent ≤ 50 % of the episode above the spike VWAP."""
import json
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd

from services import watch_filters as WF

ROOT = pathlib.Path(__file__).resolve().parents[1]
ON = SimpleNamespace(watch_block_lite_weak_hold=True, watch_block_lite_btc_rsi_hot=False, watch_block_lite_btc_ema20_down=False)


def test_rule():
    assert WF.LITE_WEAK_HOLD_MAX == 50.0
    assert WF.lite_blocks(ON, 0.5, 60, 50.0) == ["LITE_WEAK_HOLD"] and WF.lite_blocks(ON, 0.5, 60, 8.61) == ["LITE_WEAK_HOLD"]
    assert WF.lite_blocks(ON, 0.5, 60, 50.004) == ["LITE_WEAK_HOLD"] and WF.lite_blocks(ON, 0.5, 60, 50.01) == []   # 2-dp stamp decides
    assert WF.lite_blocks(ON, 0.5, 60, 94.59) == [] and WF.lite_blocks(ON, 0.5, 60, None) == []                      # live INIT 11-15 kept
    assert WF.frenzy_blocks(ON, "FRENZY_LITE", above_share=30) == ["LITE_WEAK_HOLD"]
    for sl in ("FRENZY_LITE_RETRY", "FRENZY_WILLY", "FRENZY_WILLY_HT", "FRENZY_LONG", "FRENZY_WIDE"):
        assert "LITE_WEAK_HOLD" not in WF.frenzy_blocks(ON, sl, above_share=10)
    off = SimpleNamespace(watch_block_lite_weak_hold=False, watch_block_lite_btc_rsi_hot=False, watch_block_lite_btc_ema20_down=False)
    assert WF.lite_blocks(off, 0.5, 60, 10) == []


def test_master_cohort_matches_the_study():
    m = pd.read_csv(ROOT / "reports/MASTER_POOL_stacked.csv", low_memory=False)
    th = SimpleNamespace(**{v: True for v in WF.TOGGLES.values()})
    # (568) since STACK 2026-11-16a the lines block in the master: the study cohort = rows kept before the freeze (blocked by THESE lines only)
    kept = (m.stack_keep.astype(str) == "True") | m.stack_block_reason.astype(str).str.fullmatch(r"FRENZY_LITE_BLOCK_(LITE_WEAK_HOLD|LITE_BTC_RSI_HOT)(\+LITE_WEAK_HOLD)?")
    k = m[kept & (m.entry_strategy == "FRENZY_LITE") & (pd.to_datetime(m.opened_at, format="mixed") < "2026-11-15")]   # (review c) frozen study window
    n = lambda r, c: pd.to_numeric(r.get(c), errors="coerce")
    k = k[[not WF.lite_blocks(th, n(r, "entry_btc_ema20_slope"), n(r, "entry_btc_rsi")) for _, r in k.iterrows()]]
    z = k[[bool(WF.lite_blocks(ON, None, None, n(r, "entry_frenzy_above_share"))) for _, r in k.iterrows()]]
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_master_pool import frenzy_fixed_pct
    p = pd.Series([float(s) if pd.notna(s) else frenzy_fixed_pct("FRENZY_LITE", pd.to_numeric(pk, errors="coerce"), float(pp), 3.0)
                   for s, pk, pp in zip(pd.to_numeric(z.stack_pct), z.peak_pnl, z.pnl_percentage)])   # (568) a blocked row: its +3-TP price
    assert len(z) == 16 and (p > 0).sum() == 8


def test_wiring():
    import config
    assert config.SignalThresholds().watch_block_lite_weak_hold is False
    assert json.loads((ROOT / "trading_config.json").read_text())["thresholds"]["watch_block_lite_weak_hold"] is True
    assert (ROOT / "templates/index.html").read_text().count("config-wb-lite-weak-hold") == 3
    eng = (ROOT / "services/trading_engine.py").read_text()
    assert "above_share=(round(flag['above_share'], 2) if flag.get('above_share') is not None else None))   # 🧱 (556)" in eng
    assert 'above_share=_num(r, "entry_frenzy_above_share"))' in (ROOT / "scripts/build_master_pool.py").read_text()
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_blocked_filters as S
    assert "LITE_WEAK_HOLD" in S.LINES["FRENZY_LITE"] and S.TIGHT["LITE_WEAK_HOLD"] == (8, None)
