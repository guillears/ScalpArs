"""🧱 (DECISION_LOG 554) LITE_BTC_RSI_HOT — the pure rule, its config / UI wiring and its tight revert gate."""
import json
import pathlib
from types import SimpleNamespace

import pandas as pd

from services import watch_filters as WF

ROOT = pathlib.Path(__file__).resolve().parents[1]
ON = SimpleNamespace(watch_block_lite_btc_rsi_hot=True, watch_block_lite_btc_ema20_down=False)


def test_rule():
    assert WF.LITE_BTC_RSI_HOT_MIN == 70.0
    assert WF.lite_blocks(ON, 0.5, 70.0) == ["LITE_BTC_RSI_HOT"] and WF.lite_blocks(ON, 0.5, 79.3) == ["LITE_BTC_RSI_HOT"]
    assert WF.lite_blocks(ON, 0.5, 69.94) == [] and WF.lite_blocks(ON, 0.5, 69.96) == ["LITE_BTC_RSI_HOT"]   # the stamp's 1-dp value decides
    assert WF.lite_blocks(ON, 0.5, None) == [] and WF.lite_blocks(ON, 0.5, float("nan")) == []               # unreadable never blocks
    assert WF.lite_blocks(SimpleNamespace(watch_block_lite_btc_rsi_hot=False, watch_block_lite_btc_ema20_down=False), 0.5, 80) == []
    assert WF.frenzy_blocks(ON, "FRENZY_LITE", btc_rsi=75.0) == ["LITE_BTC_RSI_HOT"]
    for sl in ("FRENZY_LITE_RETRY", "FRENZY_WILLY", "FRENZY_WILLY_HT", "FRENZY_LONG", "FRENZY_WIDE"):
        assert "LITE_BTC_RSI_HOT" not in WF.frenzy_blocks(ON, sl, btc_rsi=90.0)
    assert WF.TOGGLES["LITE_BTC_RSI_HOT"] == "watch_block_lite_btc_rsi_hot"


def test_wiring():
    import config
    assert config.SignalThresholds().watch_block_lite_btc_rsi_hot is False
    j = json.loads((ROOT / "trading_config.json").read_text())["thresholds"]
    assert j["watch_block_lite_btc_rsi_hot"] is False   # DL 570: switched off (reversed on unseen backtest months, reports/FILTER_THEORY_AUDIT_2026-11-16.md)
    html = (ROOT / "templates/index.html").read_text()
    assert html.count("config-wb-lite-rsi-hot") == 3
    eng = (ROOT / "services/trading_engine.py").read_text()
    assert "btc_rsi=(_ef.get('entry_btc_rsi') if _ef.get('entry_btc_rsi') is not None else _g.get('_current_btc_rsi'))" in eng
    assert 'btc_rsi=_num(r, "entry_btc_rsi"),' in (ROOT / "scripts/build_master_pool.py").read_text()


def test_tight_gate():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_blocked_filters as S

    def B(pcts, pend=0):
        rows = [dict(te=i, lines="LITE_BTC_RSI_HOT", status="PRICED", pct=p, day=i // 2, sleeve="FRENZY_LITE") for i, p in enumerate(pcts)]
        rows += [dict(te=-1, lines="LITE_BTC_RSI_HOT", status="PENDING", pct=None, day=0, sleeve="FRENZY_LITE")] * pend
        return pd.DataFrame(rows)
    K = pd.DataFrame(columns=["te", "status", "pct", "sleeve"])
    assert S.gate_read(B([-3, -3, -3, -3]), K, "LITE_BTC_RSI_HOT")[0] == "COLLECTING"
    assert S.gate_read(B([-3, -3, -3, -3, -3]), K, "LITE_BTC_RSI_HOT")[0] == "KEEP"
    assert S.gate_read(B([3, 3, 3, -3, -3]), K, "LITE_BTC_RSI_HOT")[0].startswith("REVERT")      # avg > 0
    assert S.gate_read(B([3, 3, 1, -3, -3, -3]), K, "LITE_BTC_RSI_HOT")[0].startswith("REVERT")   # avg +0.2 > 0 on the first 5
    assert S.gate_read(B([3, 3, 3, -4, -4]), K, "LITE_BTC_RSI_HOT")[0].startswith("REVERT")      # 3 winners, avg ≤ 0
    assert S.gate_read(B([-3, -3, -3, -3, -3], pend=1), K, "LITE_BTC_RSI_HOT")[0] == "COLLECTING"
    assert "LITE_BTC_RSI_HOT" in S.LINES["FRENZY_LITE"]
