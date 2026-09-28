"""⚡ Sep-28 BTC ADX-SURGE WAIVER — pure-rule invariants + live/default parity + engine wiring (DECISION_LOG 123).

It is an UN-block of BTC-level gates, so every doubt must FAIL CLOSED (the gate keeps blocking): feature off, non-positive
thresholds, a missing/non-finite reading, a gate it does not own (ADX HIGH, the ≥70 band, anything else)."""
import json
import os
from types import SimpleNamespace

from services.trading_engine import btc_adx_surge_waived

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _th(**kw):
    d = dict(long_btc_adx_surge_enabled=True, long_btc_adx_surge_min_delta=1.5, long_btc_adx_surge_slope_min=0.05,
             long_btc_adx_surge_band_rsi_max=60.0)
    d.update(kw)
    return SimpleNamespace(**d)


def test_adx_floor_lifted_on_surge():
    assert btc_adx_surge_waived(_th(), "BTC_ADX_GATE_LOW", None, 17.9, 16.2, 0.075) is True
    assert btc_adx_surge_waived(_th(), "BTC_ADX_GATE_LOW", None, 17.5, 16.0, 0.051) is True    # Δ exactly 1.5 → inclusive


def test_boundaries_strict_slope_inclusive_delta():
    assert btc_adx_surge_waived(_th(), "BTC_ADX_GATE_LOW", None, 17.4, 16.0, 0.075) is False   # Δ 1.4 < 1.5
    assert btc_adx_surge_waived(_th(), "BTC_ADX_GATE_LOW", None, 18.0, 16.0, 0.05) is False    # slope must be > 0.05
    assert btc_adx_surge_waived(_th(), "BTC_ADX_GATE_LOW", None, 18.0, 16.0, -0.10) is False   # falling BTC


def test_only_the_50_60_bands_never_the_overbought_band():
    assert btc_adx_surge_waived(_th(), "BTC_RSI_ADX_CROSS", 50.0, 20.0, 18.0, 0.08) is True
    assert btc_adx_surge_waived(_th(), "BTC_RSI_ADX_CROSS", 55.0, 20.0, 18.0, 0.08) is True
    assert btc_adx_surge_waived(_th(), "BTC_RSI_ADX_CROSS", 60.0, 20.0, 18.0, 0.08) is False   # floor == band_rsi_max
    assert btc_adx_surge_waived(_th(), "BTC_RSI_ADX_CROSS", 70.0, 20.0, 18.0, 0.08) is False   # CROSS_OB's band, never here
    assert btc_adx_surge_waived(_th(), "BTC_RSI_ADX_CROSS", None, 20.0, 18.0, 0.08) is False


def test_foreign_gates_never_lifted():
    for g in ("BTC_ADX_GATE_HIGH", "BTC_SLOPE_GATE", "BTC_TREND_FILTER", "", None):
        assert btc_adx_surge_waived(_th(), g, None, 20.0, 18.0, 0.08) is False


def test_fail_closed_on_bad_readings():
    for a, ap, s in ((None, 16.0, 0.08), (18.0, None, 0.08), (18.0, 16.0, None),
                     (float("nan"), 16.0, 0.08), (18.0, float("inf"), 0.08), ("x", 16.0, 0.08)):
        assert btc_adx_surge_waived(_th(), "BTC_ADX_GATE_LOW", None, a, ap, s) is False


def test_off_or_zeroed_never_waives():
    assert btc_adx_surge_waived(_th(long_btc_adx_surge_enabled=False), "BTC_ADX_GATE_LOW", None, 18.0, 16.0, 0.08) is False
    assert btc_adx_surge_waived(_th(long_btc_adx_surge_min_delta=0), "BTC_ADX_GATE_LOW", None, 18.0, 16.0, 0.08) is False
    assert btc_adx_surge_waived(_th(long_btc_adx_surge_slope_min=0), "BTC_ADX_GATE_LOW", None, 18.0, 16.0, 0.08) is False
    assert btc_adx_surge_waived(SimpleNamespace(), "BTC_ADX_GATE_LOW", None, 18.0, 16.0, 0.08) is False


def test_live_json_armed_and_default_off():
    import config
    with open(os.path.join(ROOT, "trading_config.json")) as f:
        th = json.load(f)["thresholds"]
    assert th["long_btc_adx_surge_enabled"] is True
    assert (th["long_btc_adx_surge_min_delta"], th["long_btc_adx_surge_slope_min"], th["long_btc_adx_surge_band_rsi_max"]) == (1.5, 0.05, 60.0)
    assert (th["long_btc_adx_surge_invest_mult"], th["long_btc_adx_surge_lev_mult"]) == (1.0, 1.0)
    f = config.SignalThresholds.model_fields
    assert f["long_btc_adx_surge_enabled"].default is False                 # shipped via JSON; a missing key never arms it
    assert config.load_trading_config().thresholds.long_btc_adx_surge_enabled is True
    # the live-day example (Sep-28 16:17→16:19 UTC-ish surge) is admitted by the loaded thresholds
    assert btc_adx_surge_waived(SimpleNamespace(**th), "BTC_ADX_GATE_LOW", None, 17.9, 16.3, 0.075) is True


def test_engine_wiring_and_column():
    src = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert src.count("btc_adx_surge_waived(") >= 5                        # def + 2 pre-check + 2 per-pair
    assert '"ADX_SURGE_OPEN"' in src and "adx_surge_open=_adx_surge_admit" in src      # column = sizing predicate (review #4)
    assert "if not _bl_opened and not _adx_surge_open_hit:" in src                        # no FAN flip seeded by a waiver (review #3)
    assert "_adx_surge_open_hit = False" in src
    assert "adx_surge_open=bool(signal == \"LONG\" and _adx_surge_open_hit)" in src
    from models import Order
    assert "adx_surge_open" in [c.name for c in Order.__table__.columns]
    db = open(os.path.join(ROOT, "database.py")).read()
    assert "ADD COLUMN adx_surge_open BOOLEAN" in db
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert ui.count("config-long-btc-adx-surge-enabled") == 3                # input + load + save
    assert "⚡ ADX SURGE" in ui and "o.adx_surge_open" in ui
