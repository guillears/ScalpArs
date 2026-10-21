"""⏱ STACK 2026-10-21b — the master's FADE LATE-ARM settings (CF_FADE_LATE_ARM) are frozen constants pinned to trading_config.json
(was a hard-coded 15.0 / 0.30 inside main()); a retune of the live setting fails here until the stack is bumped and rebuilt."""
import inspect, json, os
import pytest

from scripts import build_master_pool as B

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_frozen_late_arm_matches_trading_config():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert float(th["spike_fade_late_arm_after_min"]) == B.FADE_LATE_ARM_AFTER_MIN_FROZEN, \
        "late-arm retuned in trading_config.json — update FADE_LATE_ARM_*_FROZEN + bump STACK_VERSION, rebuild the master"
    assert float(th["spike_fade_late_arm_peak"]) == pytest.approx(B.FADE_LATE_ARM_PEAK_FROZEN)


def test_builder_uses_the_frozen_values():
    src = inspect.getsource(B.main)
    assert "spike_fade_late_arm_after_min=FADE_LATE_ARM_AFTER_MIN_FROZEN" in src
    assert "spike_fade_late_arm_peak=FADE_LATE_ARM_PEAK_FROZEN" in src
