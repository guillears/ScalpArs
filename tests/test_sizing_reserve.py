"""Sizing/reserve invariants — the schedule tiers, 3-leg fee reserve, and the
display mirror (main._reserve_split) that MUST match the engine's math.

The v3 operating schedule is used verbatim; expected numbers are hand-derived
(and match the Sep-3 schedule-aware fee-base review's table).
"""
import config
import main
import services.trading_engine as te
from services.trading_engine import _lookup_leverage_schedule

SCHED = "10000:8000, 25000:17500, 50000:27500, 100000:40000, 150000:50000, 250000:70000, 500000:100000"


def _setup(monkeypatch, mode="schedule", pct=2.5, usd=15.0, hours=12.0,
           burn=0.0, mature=False):
    inv = config.trading_config.investment
    monkeypatch.setattr(inv, "reserve_mode", mode)
    monkeypatch.setattr(inv, "reserve_schedule", SCHED)
    monkeypatch.setattr(inv, "fee_reserve_pct", pct)
    monkeypatch.setattr(inv, "fee_reserve_usd", usd)
    monkeypatch.setattr(inv, "fee_reserve_hours", hours)
    monkeypatch.setattr(te.trading_engine, "_bnb_burn_rate", burn, raising=False)
    monkeypatch.setattr(te.trading_engine, "_bnb_data_mature", mature, raising=False)


def test_schedule_lookup_boundaries():
    assert _lookup_leverage_schedule(SCHED, 9999.99) is None      # below first tier
    assert _lookup_leverage_schedule(SCHED, 10000.0) == 8000      # inclusive boundary
    assert _lookup_leverage_schedule(SCHED, 24999.0) == 8000
    assert _lookup_leverage_schedule(SCHED, 500000.0) == 100000
    assert _lookup_leverage_schedule(SCHED, 5_000_000.0) == 100000
    assert _lookup_leverage_schedule("", 50000.0) is None          # empty = off (fail-open)
    assert _lookup_leverage_schedule("garbage", 50000.0) is None   # malformed = off


def test_reserve_split_below_first_tier(monkeypatch):
    _setup(monkeypatch)
    # sub-tier: no schedule reserve; fee reserve = max($15, 2.5% x equity)
    r, t, mode = main._reserve_split(2900.0, 0.0)
    assert mode == "schedule"
    assert abs(r - 72.50) < 0.01 and abs(t - 2827.50) < 0.01


def test_reserve_split_tier_10k_schedule_aware_fee_base(monkeypatch):
    _setup(monkeypatch)
    # $10k tier: schedule holds 2000; fee pct leg keys on the TRADEABLE TARGET
    # (Sep-3 ship): 2.5% x 8000 = 200 -> reserve 2200, tradeable 7800.
    r, t, _ = main._reserve_split(10000.0, 0.0)
    assert abs(r - 2200.0) < 0.01 and abs(t - 7800.0) < 0.01


def test_reserve_split_tier_500k(monkeypatch):
    _setup(monkeypatch)
    # 500k: schedule 400k + 2.5% x 100k target = 2.5k -> tradeable 97.5k
    r, t, _ = main._reserve_split(500000.0, 0.0)
    assert abs(r - 402500.0) < 0.5 and abs(t - 97500.0) < 0.5


def test_reserve_with_deployed_margin(monkeypatch):
    _setup(monkeypatch)
    # equity 10k with 6k in margin: tier on TOTAL equity; displayed reserve
    # capped at free balance; tradeable = free - reserve.
    r, t, _ = main._reserve_split(4000.0, 6000.0)
    assert abs(r - 2200.0) < 0.01 and abs(t - 1800.0) < 0.01


def test_burn_leg_requires_maturity(monkeypatch):
    # the B5-boot artifact: an immature $82/hr burn must NOT inflate the reserve
    _setup(monkeypatch, burn=82.0, mature=False)
    r, t, _ = main._reserve_split(2900.0, 0.0)
    assert abs(r - 72.50) < 0.01          # pct leg only
    _setup(monkeypatch, burn=5.0, mature=True)
    r, t, _ = main._reserve_split(2900.0, 0.0)
    assert abs(r - 72.50) < 0.01          # 12h x $5 = $60 < pct leg 72.50
    _setup(monkeypatch, burn=10.0, mature=True)
    r, t, _ = main._reserve_split(2900.0, 0.0)
    assert abs(r - 120.0) < 0.01          # burn leg 120 governs


def test_fee_reserve_absolute_floor(monkeypatch):
    _setup(monkeypatch, pct=0.0, hours=0.0)
    r, t, _ = main._reserve_split(2900.0, 0.0)
    assert abs(r - 15.0) < 0.01           # $15 min floor alone


def test_tradeable_never_negative(monkeypatch):
    _setup(monkeypatch)
    r, t, _ = main._reserve_split(3.24, 0.0)
    assert t == 0.0 and r <= 3.24         # reserve display capped at free balance


def test_engine_mirror_parity(monkeypatch):
    """caveman-review: the display mirror and the ENGINE must compute the same
    tradeable. mode=percentage @100% makes the engine's base investment equal its
    internal tradeable — compare it to main._reserve_split on both key tiers."""
    _setup(monkeypatch)
    inv = config.trading_config.investment
    monkeypatch.setattr(inv, "mode", "percentage")
    monkeypatch.setattr(inv, "percentage", 100.0)
    monkeypatch.setattr(inv, "max_investment_size", 10**9)
    monkeypatch.setattr(inv, "min_investment_size", 0.0)
    conf, level = next((k, v) for k, v in config.trading_config.confidence_levels.items() if v.enabled)
    monkeypatch.setattr(level, "investment_multiplier", 1.0)  # 1x so investment == tradeable exactly
    for equity in (10000.0, 500000.0):
        investment, _lev, _capped = te.trading_engine.calculate_position_size(
            equity, conf, total_portfolio=equity)
        _r, mirror_tradeable, _ = main._reserve_split(equity, 0.0)
        assert abs(investment - mirror_tradeable) < 0.51, (equity, investment, mirror_tradeable)
