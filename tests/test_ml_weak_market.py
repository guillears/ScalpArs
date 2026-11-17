"""🧱 (DECISION_LOG 559) two momentum-LONG lines, both armed: ML_WEAK_MARKET (breadth bull % ≤ 64.1 ∧ BTC ADX ≤ 21.71) and
ML_FADING_QUIET (the coin's ADX falling at entry ∧ ATR % ≤ 0.38). Declared override, tight revert per line (first 8 priced blocked > 0 → off)."""
import json
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd

from services import watch_filters as WF

ROOT = pathlib.Path(__file__).resolve().parents[1]
ON = SimpleNamespace(watch_block_ml_weak_market=True, watch_block_ml_fading_quiet=True)


def test_rule_weak_market():
    assert (WF.ML_WEAK_BULL_MAX, WF.ML_WEAK_BTC_ADX_MAX) == (64.1, 21.71)
    th = SimpleNamespace(watch_block_ml_weak_market=True)
    assert WF.ml_blocks(th, None, 64.1, btc_adx=21.71) == ["ML_WEAK_MARKET"]          # both cuts inclusive
    assert WF.ml_blocks(th, None, 64.2, btc_adx=10) == []                            # breadth healthy
    assert WF.ml_blocks(th, None, 50, btc_adx=21.72) == []                           # BTC trending
    assert WF.ml_blocks(th, None, None, btc_adx=10) == []                            # unreadable never blocks
    assert WF.ml_blocks(th, None, 50, btc_adx=None) == []
    assert WF.ml_blocks(th, None, 0.0, btc_adx=10) == []                             # 0.0 = "no pairs scanned" fallback → fail open
    assert WF.ml_blocks(SimpleNamespace(watch_block_ml_weak_market=False), None, 50, btc_adx=10) == []


def test_rule_fading_quiet():
    assert (WF.ML_FADING_ADX_DELTA_MAX, WF.ML_QUIET_ATR_MAX) == (0.0, 0.38)
    th = SimpleNamespace(watch_block_ml_fading_quiet=True)
    assert WF.ml_blocks(th, None, 80, adx_delta=-0.01, atr_pct=0.38) == ["ML_FADING_QUIET"]
    assert WF.ml_blocks(th, None, 80, adx_delta=0.0, atr_pct=0.2) == []              # flat ADX is not falling
    assert WF.ml_blocks(th, None, 80, adx_delta=-1, atr_pct=0.39) == []              # coin not quiet
    assert WF.ml_blocks(th, None, 80, adx_delta=None, atr_pct=0.2) == []
    assert WF.ml_blocks(th, None, 80, adx_delta=-1, atr_pct=None) == []
    assert WF.ml_blocks(SimpleNamespace(watch_block_ml_fading_quiet=False), None, 80, adx_delta=-1, atr_pct=0.2) == []
    assert WF.ml_blocks(ON, None, 50, btc_adx=10, adx_delta=-1, atr_pct=0.2) == ["ML_WEAK_MARKET", "ML_FADING_QUIET"]
    assert WF.ml_blocks(ON, None, 70) == []                                           # old callers unchanged


def test_master_cohorts_match_the_study():
    m = pd.read_csv(ROOT / "reports/MASTER_POOL_stacked.csv", low_memory=False)
    kept = (m.stack_keep.astype(str) == "True") | m.stack_block_reason.astype(str).str.fullmatch(r"MOMENTUM_LONG_BLOCK_(ML_WEAK_RESCUE|ML_WEAK_MARKET|ML_FADING_QUIET)(\+(ML_WEAK_RESCUE|ML_WEAK_MARKET|ML_FADING_QUIET))*")   # (568) blocked by these lines only
    k = m[(m.entry_strategy == "MOMENTUM") & (m.direction == "LONG") & kept & (m.is_probe.astype(str).isin(["False", "0", "false", "nan"])) & (m.era.astype(str) != "B27")]   # (DL 576) the study = pre-B27
    # (568) LIT 07-10 17:02 was refused by LONG_CHOP_BURST at 11-14a (not in the study's 124); the 11-16a freeze judges ML_WEAK_MARKET first
    k = k[~((k.pair == "LITUSDT") & (k.opened_at.astype(str).str[:19] == "2026-07-10T17:02:16"))]
    assert len(k) == 124
    def cohort(th):
        z = k[[bool(WF.ml_blocks(th, None, b, btc_adx=a, adx_delta=d, atr_pct=t))
               for b, a, d, t in zip(k.entry_bull_pct, k.entry_btc_adx, k.entry_adx_delta, k.entry_atr_pct)]]
        p = pd.to_numeric(z.stack_pct).fillna(pd.to_numeric(z.pnl_percentage))   # (568) a blocked row: its as-traded pct (= its pre-freeze stack_pct)
        return len(z), round(p.mean(), 2)
    assert cohort(SimpleNamespace(watch_block_ml_weak_market=True)) == (8, -0.15)
    assert cohort(SimpleNamespace(watch_block_ml_fading_quiet=True)) == (11, -0.08)
    assert cohort(ON) == (17, -0.06)


def test_wiring():
    import config
    d = config.SignalThresholds()
    assert d.watch_block_ml_weak_market is False and d.watch_block_ml_fading_quiet is False
    th = json.loads((ROOT / "trading_config.json").read_text())["thresholds"]
    assert th["watch_block_ml_weak_market"] is True and th["watch_block_ml_fading_quiet"] is True
    html = (ROOT / "templates/index.html").read_text()
    assert html.count("config-wb-ml-weak-market") == 3 and html.count("config-wb-ml-fading-quiet") == 3
    assert WF.TOGGLES["ML_WEAK_MARKET"] == "watch_block_ml_weak_market" and WF.TOGGLES["ML_FADING_QUIET"] == "watch_block_ml_fading_quiet"
    eng = (ROOT / "services/trading_engine.py").read_text()
    assert "btc_adx=entry_btc_adx, adx_delta=entry_adx_delta, atr_pct=entry_atr_pct); _wf_sleeve = \"MOMENTUM_LONG\"" in eng
    assert "atr_pct=(entry_fields or {}).get('entry_atr_pct'))   # (559)" in eng      # refusal recorder
    assert 'atr_pct=_num(r, "entry_atr_pct"))   # 🧱 (559)' in (ROOT / "scripts/build_master_pool.py").read_text()
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_blocked_filters as S
    for ln in ("ML_WEAK_MARKET", "ML_FADING_QUIET"):
        assert ln in S.LINES["MOMENTUM_LONG"] and S.TIGHT[ln] == (8, None)
        K = pd.DataFrame(columns=["te", "status", "pct", "sleeve"])
        B = lambda ps: pd.DataFrame([dict(te=i, lines=ln, status="PRICED", pct=p, day=i, sleeve="MOMENTUM_LONG") for i, p in enumerate(ps)])
        assert S.gate_read(B([-0.7] * 7), K, ln)[0] == "COLLECTING"
        assert S.gate_read(B([-0.7] * 8), K, ln)[0] == "KEEP"
        assert S.gate_read(B([0.5] * 6 + [-0.7] * 2), K, ln)[0].startswith("REVERT")
