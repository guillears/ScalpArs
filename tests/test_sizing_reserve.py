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
           burn=0.0, mature=False, bnb_held=0.0):
    # Sep-11: burn leg is runway-aware (hours x burn − BNB held). Tests pin BNB held explicitly
    # (paper mode → paper_bnb_balance_usd) so legacy expectations stay exact at bnb_held=0.
    monkeypatch.setattr(te.trading_engine, "is_paper_mode", True, raising=False)
    monkeypatch.setattr(te.trading_engine, "paper_bnb_balance_usd", bnb_held, raising=False)
    inv = config.trading_config.investment
    monkeypatch.setattr(inv, "reserve_mode", mode)
    monkeypatch.setattr(inv, "reserve_schedule", SCHED)
    monkeypatch.setattr(inv, "fee_reserve_pct", pct)
    monkeypatch.setattr(inv, "fee_reserve_usd", usd)
    monkeypatch.setattr(inv, "fee_reserve_hours", hours)
    monkeypatch.setattr(te.trading_engine, "_bnb_burn_rate", burn, raising=False)
    monkeypatch.setattr(te.trading_engine, "_bnb_burn_rate_bot", burn, raising=False)   # Sep-29c: the sizing leg reads the bot-only rate
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
    # $10k tier: schedule holds 2000; fee pct leg keys on the TRADEABLE TARGET (2.5% x 8000 = 200)
    # DL 424: the larger of the two, never both -> reserve 2000, tradeable 8000.
    r, t, _ = main._reserve_split(10000.0, 0.0)
    assert abs(r - 2000.0) < 0.01 and abs(t - 8000.0) < 0.01


def test_reserve_split_tier_500k(monkeypatch):
    _setup(monkeypatch)
    # 500k: schedule 400k, larger than the fee leg 2.5k (DL 424) -> tradeable 100k
    r, t, _ = main._reserve_split(500000.0, 0.0)
    assert abs(r - 400000.0) < 0.5 and abs(t - 100000.0) < 0.5


def test_reserve_with_deployed_margin(monkeypatch):
    _setup(monkeypatch)
    # equity 10k with 6k in margin: tier on TOTAL equity; displayed reserve
    # capped at free balance; tradeable = free - reserve.
    r, t, _ = main._reserve_split(4000.0, 6000.0)
    assert abs(r - 2000.0) < 0.01 and abs(t - 2000.0) < 0.01


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


def test_burn_leg_nets_out_bnb_held(monkeypatch):
    """Sep-11 (operator-caught double-provisioning): the USDT burn leg only reserves the fee
    coverage BNB does NOT already provide. 12h x $20.73 = $248.76 need."""
    _setup(monkeypatch, burn=20.73, mature=True, bnb_held=0.0)
    r, _, _ = main._reserve_split(2574.88, 0.0)
    assert abs(r - 248.76) < 0.01                       # no BNB → full leg (old behavior)
    _setup(monkeypatch, burn=20.73, mature=True, bnb_held=507.82)
    r, _, _ = main._reserve_split(2574.88, 0.0)
    assert abs(r - 2574.88 * 0.025) < 0.01              # BNB covers 24h → leg 0 → pct leg governs ($64.37)
    _setup(monkeypatch, burn=20.73, mature=True, bnb_held=150.0)
    r, _, _ = main._reserve_split(2574.88, 0.0)
    assert abs(r - (248.76 - 150.0)) < 0.01             # partial: reserve exactly the uncovered gap


def test_burn_leg_unknown_live_bnb_is_conservative(monkeypatch):
    """Live mode with no BNB probe yet (None) → full leg, never under-reserve."""
    _setup(monkeypatch, burn=20.73, mature=True)
    monkeypatch.setattr(te.trading_engine, "is_paper_mode", False, raising=False)
    monkeypatch.setattr(te.trading_engine, "_bnb_usd_last", None, raising=False)
    r, _, _ = main._reserve_split(2574.88, 0.0)
    assert abs(r - 248.76) < 0.01
    monkeypatch.setattr(te.trading_engine, "_bnb_usd_last", 507.82, raising=False)
    r, _, _ = main._reserve_split(2574.88, 0.0)
    assert abs(r - 2574.88 * 0.025) < 0.01


def test_engine_mirror_parity_with_bnb_held(monkeypatch):
    """Engine and display must net out BNB identically (mirror rule)."""
    _setup(monkeypatch, burn=20.0, mature=True, bnb_held=100.0)
    inv = config.trading_config.investment
    monkeypatch.setattr(inv, "mode", "percentage"); monkeypatch.setattr(inv, "percentage", 100.0)
    monkeypatch.setattr(inv, "max_investment_size", 10**9); monkeypatch.setattr(inv, "min_investment_size", 0.0)
    conf, level = next((k, v) for k, v in config.trading_config.confidence_levels.items() if v.enabled)
    monkeypatch.setattr(level, "investment_multiplier", 1.0)
    investment, _, _ = te.trading_engine.calculate_position_size(3000.0, conf, total_portfolio=3000.0)
    _, mirror_tradeable, _ = main._reserve_split(3000.0, 0.0)
    assert abs(investment - mirror_tradeable) < 0.51     # burn leg 240−100 = 140 on both sides


def test_burn_leg_reads_the_bot_rate_not_the_account_rate(monkeypatch):
    """Sep-29c: the dashboard burn rate counts every fee paid (manual fills included); the sizing fee leg reads the BOT-only
    rate. A manual-only batch (account rate $125.80/hr, bot rate 0, data mature) holds nothing extra back — engine and mirror."""
    _setup(monkeypatch, burn=0.0, mature=True)
    inv = config.trading_config.investment
    monkeypatch.setattr(inv, "mode", "percentage"); monkeypatch.setattr(inv, "percentage", 100.0)
    monkeypatch.setattr(inv, "max_investment_size", 10**9); monkeypatch.setattr(inv, "min_investment_size", 0.0)
    conf, level = next((k, v) for k, v in config.trading_config.confidence_levels.items() if v.enabled)
    monkeypatch.setattr(level, "investment_multiplier", 1.0)

    def both():
        investment, _, _ = te.trading_engine.calculate_position_size(2900.0, conf, total_portfolio=2900.0)
        reserve, tradeable, _ = main._reserve_split(2900.0, 0.0)
        return investment, reserve, tradeable
    quiet = both()
    monkeypatch.setattr(te.trading_engine, "_bnb_burn_rate", 125.8, raising=False)        # what the card shows
    assert both() == quiet and abs(quiet[1] - 72.50) < 0.01                               # pct leg only, on both sides
    monkeypatch.setattr(te.trading_engine, "_bnb_burn_rate", 0.0, raising=False)
    monkeypatch.setattr(te.trading_engine, "_bnb_burn_rate_bot", 10.0, raising=False)     # the bot's own fees DO switch the leg on
    investment, reserve, tradeable = both()
    assert abs(reserve - 120.0) < 0.01 and abs(investment - tradeable) < 0.51             # 12 h × $10 governs, engine == mirror



DL404 = "10000:9500, 25000:23000, 50000:40000, 100000:75000, 150000:105000, 250000:162500, 500000:300000, 1000000:500000"


def test_dl404_schedule_lookup():
    """DECISION_LOG 404 table (constant — the live json may be re-tuned from the UI without breaking the gate)."""
    assert _lookup_leverage_schedule(DL404, 9999) is None and _lookup_leverage_schedule(DL404, 10000) == 9500
    assert _lookup_leverage_schedule(DL404, 24999) == 9500 and _lookup_leverage_schedule(DL404, 25000) == 23000
    assert _lookup_leverage_schedule(DL404, 999999) == 300000 and _lookup_leverage_schedule(DL404, 1e6) == 500000
    assert _lookup_leverage_schedule(DL404, 5e6) == 500000


def test_live_reserve_schedule_is_sane():
    """whatever the live json holds: parses, ascending balances, ascending targets, a target never above its balance."""
    import json, os
    live = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_config.json")))["investment"]["reserve_schedule"]
    tiers = [(float(x.split(":")[0]), float(x.split(":")[1])) for x in live.split(",") if x.strip()]
    assert tiers and [b for b, _ in tiers] == sorted(b for b, _ in tiers) and [t for _, t in tiers] == sorted(t for _, t in tiers)
    assert all(0 < t < b for b, t in tiers)
    assert len(tiers) <= 9                                              # the UI editor has 9 rows (DL 424): a 10th tier would be dropped on Save


def test_reserve_schedule_is_a_straight_line_and_flat_above_the_last_tier():
    """DL 424: no tier makes tradeable jump; the leverage schedule stays stepped."""
    S = "10000:9750, 25000:23000, 50000:40000, 1000000:500000, 2000000:800000"
    f = te._reserve_schedule_target
    assert f(S, 9999.99) is None and f(S, 10000) == 9750 and f(S, 25000) == 23000 and f(S, 5e6) == 800000
    assert abs(f(S, 17500) - 16375) < 1e-6 and abs(f(S, 1.5e6) - 650000) < 1e-6
    assert f("", 50000) is None and f("garbage", 50000) is None and f(S, None) is None
    pts = [f(S, e) for e in range(10000, 2000001, 5000)]
    assert all(b >= a for a, b in zip(pts, pts[1:]))                         # tradeable never falls as the balance rises
    assert _lookup_leverage_schedule(S, 24999) == 9750


def test_engine_and_card_agree_on_the_larger_of_the_two(monkeypatch):
    _setup(monkeypatch)
    inv = config.trading_config.investment
    monkeypatch.setattr(inv, "reserve_schedule", "10000:9750, 25000:23000")
    monkeypatch.setattr(inv, "mode", "equal_split")
    monkeypatch.setattr(inv, "max_open_positions", 4)
    monkeypatch.setattr(inv, "max_investment_size", 10**9)
    monkeypatch.setattr(inv, "min_investment_size", 0.0)
    conf, level = next((k, v) for k, v in config.trading_config.confidence_levels.items() if v.enabled)
    monkeypatch.setattr(level, "investment_multiplier", 1.0)
    r, t, _ = main._reserve_split(15000.0, 0.0)
    assert abs(t - (9750 + 13250 / 3)) < 0.01 and abs(r - (15000 - t)) < 0.01
    i, _, _ = te.trading_engine.calculate_position_size(15000.0, conf, total_portfolio=15000.0)
    assert abs(i - t / 4) < 0.01                                             # engine and card agree between tiers
    r_m, t_m, _ = main._reserve_split(9000.0, 6000.0)                        # the same equity with margin deployed: same reserve, tradeable from free
    i_m, _, _ = te.trading_engine.calculate_position_size(9000.0, conf, total_portfolio=15000.0)
    assert abs(r_m - r) < 0.01 and abs(t_m - (9000 - r)) < 0.01 and abs(i_m - t / 4) < 0.01
    r2, t2, _ = main._reserve_split(9999.0, 0.0)
    assert abs(r2 - 249.98) < 0.01 and t2 > 9700                             # below the first line: fee reserve only, no dip at the line
    _, t3, _ = main._reserve_split(10000.0, 0.0)
    assert t3 >= t2
