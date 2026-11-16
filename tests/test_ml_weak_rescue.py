"""🧱 (DECISION_LOG 555) ML_WEAK_RESCUE — momentum long refused when its trend is weak (entry_gap ≤ 0.27) in a quiet market (global volume < 0.70)."""
import json
import pathlib
from types import SimpleNamespace

import pandas as pd

from services import watch_filters as WF

ROOT = pathlib.Path(__file__).resolve().parents[1]
ON = SimpleNamespace(watch_block_ml_weak_rescue=True, watch_block_ml_mkt_hot=False, watch_block_ml_bull90=False)


def test_rule():
    assert (WF.ML_WEAK_GAP_MAX, WF.ML_QUIET_GVOL_MAX) == (0.27, 0.70)
    assert WF.ml_blocks(ON, None, 70, entry_gap=0.27, global_volume_ratio=0.69) == ["ML_WEAK_RESCUE"]
    assert WF.ml_blocks(ON, None, 70, entry_gap=0.2689, global_volume_ratio=0.7415) == []     # live ARB 11-15: market not quiet
    assert WF.ml_blocks(ON, None, 70, entry_gap=0.28, global_volume_ratio=0.40) == []         # strong enough trend: the rescue stands
    assert WF.ml_blocks(ON, None, 70, entry_gap=0.20, global_volume_ratio=0.70) == []         # exactly the gate = not quiet
    assert WF.ml_blocks(ON, None, 70, entry_gap=None, global_volume_ratio=0.4) == []          # unreadable never blocks
    assert WF.ml_blocks(ON, None, 70, entry_gap=0.2, global_volume_ratio=None) == []
    off = SimpleNamespace(watch_block_ml_weak_rescue=False, watch_block_ml_mkt_hot=False, watch_block_ml_bull90=False)
    assert WF.ml_blocks(off, None, 70, entry_gap=0.2, global_volume_ratio=0.4) == []
    assert WF.ml_blocks(ON, None, 70) == []                                                   # old callers unchanged


def test_master_cohort_matches_the_study():
    m = pd.read_csv(ROOT / "reports/MASTER_POOL_stacked.csv", low_memory=False)
    kept = (m.stack_keep.astype(str) == "True") | m.stack_block_reason.astype(str).str.contains("ML_WEAK_RESCUE")   # (review B) survives the re-freeze
    k = m[(m.entry_strategy == "MOMENTUM") & (m.direction == "LONG") & kept
          & (m.is_probe.astype(str).isin(["False", "0", "false", "nan"]))]
    z = k[[bool(WF.ml_blocks(ON, None, None, entry_gap=g, global_volume_ratio=v)) for g, v in zip(k.entry_gap, k.entry_global_volume_ratio)]]
    p = pd.to_numeric(z.stack_pct)
    assert len(z) == 20 and round(p.mean(), 2) == -0.25 and (p > 0).mean() == 0.5


def test_wiring():
    import config
    assert config.SignalThresholds().watch_block_ml_weak_rescue is False
    assert json.loads((ROOT / "trading_config.json").read_text())["thresholds"]["watch_block_ml_weak_rescue"] is True
    assert (ROOT / "templates/index.html").read_text().count("config-wb-ml-weak-rescue") == 3
    eng = (ROOT / "services/trading_engine.py").read_text()
    assert "entry_gap=entry_gap, global_volume_ratio=entry_global_volume_ratio,   # 🧱 (555) ML_WEAK_RESCUE" in eng
    assert 'global_volume_ratio=_num(r, "entry_global_volume_ratio"),   # 🧱 (555)' in (ROOT / "scripts/build_master_pool.py").read_text()
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_blocked_filters as S
    assert "ML_WEAK_RESCUE" in S.LINES["MOMENTUM_LONG"] and S.TIGHT["ML_WEAK_RESCUE"] == (8, None)
    K = pd.DataFrame(columns=["te", "status", "pct", "sleeve"])
    B = lambda ps: pd.DataFrame([dict(te=i, lines="ML_WEAK_RESCUE", status="PRICED", pct=p, day=i, sleeve="MOMENTUM_LONG") for i, p in enumerate(ps)])
    assert S.gate_read(B([-0.7] * 7), K, "ML_WEAK_RESCUE")[0] == "COLLECTING"
    assert S.gate_read(B([-0.7] * 8), K, "ML_WEAK_RESCUE")[0] == "KEEP"
    assert S.gate_read(B([0.5] * 5 + [-0.7] * 3), K, "ML_WEAK_RESCUE")[0].startswith("REVERT")
    assert S.gate_read(B([0.5] * 6 + [-0.7] * 2), K, "ML_WEAK_RESCUE")[0].startswith("REVERT")
    eng = (ROOT / "services/trading_engine.py").read_text()
    assert "global_volume_ratio=(entry_fields or {}).get('entry_global_volume_ratio'),   # (555 review A)" in eng   # refusal recorder
