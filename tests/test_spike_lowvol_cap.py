"""🐳 Sep-27 thin-pair fade cap 0.3 → 0.5 (DECISION_LOG 119): live JSON == pydantic default == UI fallback, and the
threshold that defines 'thin' stays $10M (pairs ≥ $10M keep the global 0.1% cap)."""
import json
import os
import re

import config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_thin_pair_cap_parity():
    inv = json.load(open(os.path.join(ROOT, "trading_config.json")))["investment"]
    assert inv["spike_lowvol_liq_cap_pct"] == 0.5
    assert config.InvestmentConfig.model_fields["spike_lowvol_liq_cap_pct"].default == 0.5
    assert inv["spike_lowvol_threshold_usd"] == 1_000_000_000_000.0   # Oct-1: every spike pair is "thin" → 0.5 % for all fades (DECISION_LOG 165)
    assert inv["max_notional_pct_of_pair_volume"] == 0.1
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert re.search(r"config-spike-lowvol-cap'\)\?\.value \?\? 0\.50\)", html)
