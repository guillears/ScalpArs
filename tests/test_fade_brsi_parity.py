"""Sep-24 — the fade BTC-RSI ceiling lives in THREE places: the live JSON (engine), the pydantic default (used on a fresh
deploy or when the JSON fails to parse) and the master-pool builder's frozen stack constant. A drift makes every fade
ledger read silently score a different cohort than the bot trades (the seventh-staleness class). Pin them together.
"""
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _live():
    with open(os.path.join(ROOT, "trading_config.json")) as fh:
        return float(json.load(fh)["thresholds"]["spike_fade_max_btc_rsi"])


def test_builder_fade_brsi_matches_live_config():
    live = _live()
    if live == 0:
        pytest.skip("gate disabled (0) — the builder must then carry no bRSI block; not expressible as a threshold")
    with open(os.path.join(ROOT, "scripts", "build_master_pool.py")) as fh:
        src = fh.read()
    m = re.search(r"elif r\.entry_btc_rsi > ([0-9.]+): k, why = False, 'FADE_BRSI(\d+)'", src)
    assert m, "builder fade bRSI gate not found"
    assert float(m.group(1)) == live, f"builder {m.group(1)} != live {live}"
    assert int(m.group(2)) == int(live), "block label must name the live ceiling"


def test_pydantic_default_matches_json():
    """Deep review: compare the DEFAULT (fresh deploy / JSON parse failure), not the JSON-loaded instance."""
    import config
    default = config.SignalThresholds.model_fields["spike_fade_max_btc_rsi"].default
    assert float(default) == _live()
