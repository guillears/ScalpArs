"""🧊 Sep-25 C1 MOMENTUM-SHORT REGIME BLOCK — the pure rule the engine gate, the master-pool builder and the ledger call.

Rule: refuse a MOMENTUM short whose C1 (capitulation) signature matched while the BTC regime is in
momentum_short_c1_block_regimes (ship 'STRONG_BEAR'). Evidence: C1 in STRONG_BEAR 2·0%·−$178 (DASH Sep-23, BCH
Sep-25 — the only two ever); other C1 shorts 8W/4L; non-C1 momentum shorts in STRONG_BEAR 10/12 W. Operator-directed
ARMED override (DECISION_LOG 114).

Invariants pinned here:
  · C1 ∧ regime in list → block; non-C1 or other regimes pass (the rest of the sleeve keeps trading in STRONG_BEAR).
  · C1 flag accepted as bool or the CSV strings 'True'/'False' (pool rows are read back from CSV).
  · empty list / missing regime / unknown flag FAILS OPEN.
  · engine gate exempts flips, fades, bounces and the bear-run sleeve (source check).
  · builder freeze == live JSON; pool stamps DASH as MOM_SHORT_C1_REGIME.
"""
import json
import os
import re
from types import SimpleNamespace

import pytest

from services.trading_engine import mom_short_c1_regime_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _th(regs="STRONG_BEAR"):
    return SimpleNamespace(momentum_short_c1_block_regimes=regs)


def test_blocks_c1_in_listed_regime_only():
    assert mom_short_c1_regime_block(_th(), True, "STRONG_BEAR") is True          # DASH / BCH
    assert mom_short_c1_regime_block(_th(), "True", "STRONG_BEAR") is True        # CSV-read flag
    assert mom_short_c1_regime_block(_th(), True, "HEALTHY_BEAR") is False        # XLM/HYPE/AAVE/WLD winners
    assert mom_short_c1_regime_block(_th(), False, "STRONG_BEAR") is False        # non-C1 shorts keep trading
    assert mom_short_c1_regime_block(_th(), "False", "STRONG_BEAR") is False
    assert mom_short_c1_regime_block(_th("strong_bear, CHOPPY_FLAT"), True, "CHOPPY_FLAT") is True
    import numpy as np
    assert mom_short_c1_regime_block(_th(), np.bool_(True), "STRONG_BEAR") is True     # iterrows yields np.bool_
    assert mom_short_c1_regime_block(_th(), 1.0, "STRONG_BEAR") is True
    assert mom_short_c1_regime_block(_th(), np.bool_(False), "STRONG_BEAR") is False


def test_off_and_missing_inputs_fail_open():
    assert mom_short_c1_regime_block(_th(""), True, "STRONG_BEAR") is False
    assert mom_short_c1_regime_block(SimpleNamespace(), True, "STRONG_BEAR") is False
    assert mom_short_c1_regime_block(_th(), True, None) is False
    for flag in (None, float("nan"), "", "maybe"):
        assert mom_short_c1_regime_block(_th(), flag, "STRONG_BEAR") is False


def _src(*parts):
    with open(os.path.join(ROOT, *parts)) as fh:
        return fh.read()


def test_engine_gate_is_momentum_short_only():
    src = _src("services", "trading_engine.py")
    i = src.index('self._record_filter_block("MOM_SHORT_C1_REGIME", "SHORT")')
    guard = src[src.rindex("if (direction == \"SHORT\"", 0, i):i]
    for exempt in ("not flip_source", "not bull_long", "not bounce_long", "not spike_fade", "not bearrun_short"):
        assert exempt in guard
    assert "mom_short_c1_regime_block(config.trading_config.thresholds, _pc1_e, entry_btc_regime)" in guard


def test_builder_freeze_matches_live_json():
    with open(os.path.join(ROOT, "trading_config.json")) as fh:
        live = json.load(fh)["thresholds"]["momentum_short_c1_block_regimes"]
    m = re.search(r"_C1_TH = SimpleNamespace\(momentum_short_c1_block_regimes='([^']*)'\)", _src("scripts", "build_master_pool.py"))
    assert m, "builder C1 freeze not found"
    assert m.group(1) == live


def test_pool_stamps_exactly_the_strong_bear_c1_shorts():
    import pandas as pd
    p = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
    if not os.path.exists(p):
        pytest.skip("pool not built")
    d = pd.read_csv(p, low_memory=False, usecols=["pair", "opened_at", "stack_block_reason"])
    blk = d[d.stack_block_reason == "MOM_SHORT_C1_REGIME"]
    assert sorted(blk.pair) == ["BCHUSDT", "DASHUSDT"], sorted(blk.pair)    # DASH Sep-23 01:52 · BCH Sep-25 14:45


def test_screen_and_stack_version_carry_the_rule():
    assert "mom_short_c1_regime_block(th, r.get('entry_pattern_c1_match'), r.get('entry_btc_regime'))" in _src("scripts", "screen_pool.py")
    assert 'STACK_VERSION = "2026-09-25b"' in _src("scripts", "build_master_pool.py")
