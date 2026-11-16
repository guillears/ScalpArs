"""🧱 (DECISION_LOG 561) MS_REGIME_ENDS — momentum SHORT refused while BTC's daily trend is up (1 d EMA50 > EMA200) and its quality score
is ≤ 2 or ≥ 5. Scores 3–4 and every short in a BTC daily downtrend trade as before; an unreadable leg never blocks."""
import json
import pathlib
from types import SimpleNamespace

import pytest

from services import watch_filters as WF

ROOT = pathlib.Path(__file__).resolve().parents[1]
ON = SimpleNamespace(watch_block_ms_regime_ends=True)


@pytest.mark.parametrize("gap,qs,want", [
    (1.34, 5, True), (1.34, 6, True), (1.34, 2, True), (1.34, 1, True),        # live LTC / QNT (5) and ONE (2) on 11-16
    (1.34, 3, False), (1.34, 4, False),                                         # the middle trades in any regime
    (-11.6, 2, False), (-11.6, 5, False), (0.0, 5, False),                     # BTC daily downtrend / flat: never
    (None, 5, False), (1.0, None, False), (float("nan"), 5, False),            # unreadable never blocks
])
def test_rule(gap, qs, want):
    assert (WF.ms_blocks(ON, gap, qs) == ["MS_REGIME_ENDS"]) is want
    assert WF.ms_blocks(SimpleNamespace(watch_block_ms_regime_ends=False), gap, qs) == []
    assert (WF.MS_QS_WEAK_MAX, WF.MS_QS_LATE_MIN) == (2, 5)


def test_wiring():
    import config
    assert config.SignalThresholds().watch_block_ms_regime_ends is False
    assert json.loads((ROOT / "trading_config.json").read_text())["thresholds"]["watch_block_ms_regime_ends"] is True
    assert (ROOT / "templates/index.html").read_text().count("config-wb-ms-regime-ends") == 3
    assert WF.TOGGLES["MS_REGIME_ENDS"] == "watch_block_ms_regime_ends"
    eng = (ROOT / "services/trading_engine.py").read_text()
    assert "_wf_lines = watch_ms_blocks(_wf_th, _gap1d, entry_quality_score); _wf_sleeve = \"MOMENTUM_SHORT\"" in eng
    assert "or bool(getattr(config.trading_config.thresholds, 'watch_block_ms_regime_ends', False)))" in eng   # the BTC 1 d read runs for it
    b = (ROOT / "scripts/build_master_pool.py").read_text()
    assert 'WF.ms_blocks(_WF_TH, _gap1d[i], _num(r, "entry_quality_score"))' in b and "_fdm = _es.eq(\"SPIKE_FADE\") | _msm" in b
