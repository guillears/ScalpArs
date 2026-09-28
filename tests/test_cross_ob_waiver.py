"""🚪 Sep-18 narrowed overbought band — pure waiver rule. It is an UN-block, so every doubt must FAIL CLOSED
(the old band keeps blocking): stale/missing 72h reading, feature off, zeroed thresholds, non-overbought bands."""
from types import SimpleNamespace

from services.trading_engine import cross_ob_waived


def _th(**kw):
    d = dict(long_cross_ob_narrow_enabled=True, long_cross_ob_rsi_min=70.0, long_cross_ob_r72_block_min=5.0)
    d.update(kw)
    return SimpleNamespace(**d)


def test_base_breakout_is_waived():
    assert cross_ob_waived(_th(), 70.0, 1.5, True) is True      # today 13:40 UTC: BTC 72h +1.5%
    assert cross_ob_waived(_th(), 70.0, -1.6, True) is True


def test_extended_btc_keeps_the_block_inclusive():
    assert cross_ob_waived(_th(), 70.0, 5.0, True) is False
    assert cross_ob_waived(_th(), 70.0, 12.8, True) is False


def test_only_the_overbought_band_is_touched():
    assert cross_ob_waived(_th(), 50.0, 1.0, True) is False      # 50-55 total block stays
    assert cross_ob_waived(_th(), 55.0, 1.0, True) is False      # 55-60 window stays


def test_fail_closed_on_missing_or_stale_reading():
    assert cross_ob_waived(_th(), 70.0, None, True) is False
    assert cross_ob_waived(_th(), 70.0, 1.0, False) is False
    assert cross_ob_waived(_th(), None, 1.0, True) is False
    assert cross_ob_waived(_th(), 70.0, "x", True) is False


def test_disabled_or_zeroed_never_waives():
    assert cross_ob_waived(_th(long_cross_ob_narrow_enabled=False), 70.0, 1.0, True) is False
    assert cross_ob_waived(_th(long_cross_ob_r72_block_min=0.0), 70.0, 1.0, True) is False
    assert cross_ob_waived(_th(long_cross_ob_rsi_min=0.0), 70.0, 1.0, True) is False


def test_sep28_switched_off_json_and_default():
    """⛔ Sep-28 (DECISION_LOG 121): the waiver is OFF live and by default — a missing key must never re-open the band."""
    import json, os
    import config
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "trading_config.json")) as f:
        th = json.load(f)["thresholds"]
    assert th["long_cross_ob_narrow_enabled"] is False
    assert config.SignalThresholds.model_fields["long_cross_ob_narrow_enabled"].default is False
    assert config.load_trading_config().thresholds.long_cross_ob_narrow_enabled is False   # through pydantic, as the bot loads it
    live_th = SimpleNamespace(**th)
    assert cross_ob_waived(live_th, 70.0, 1.0, True) is False     # a base-breakout reading no longer waives the band
