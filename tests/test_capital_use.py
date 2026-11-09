"""💧 capital use / withdrawable (caps) — DECISION_LOG 507: services/capital_use.py invariants."""
from datetime import datetime, timedelta
from types import SimpleNamespace as NS

from services import capital_use as CU

NOW = datetime(2026, 11, 9, 12, 0, 0)


def _o(sl="FRENZY_LITE", want=10000.0, cap=12000.0, days_ago=1, status="CLOSED", conf=None, cell="", direction="LONG", pnl=0.0, closed_h=1):
    t = NOW - timedelta(days=days_ago)
    return NS(entry_strategy=sl, direction=direction, status=status, confidence=conf, cell_multiplier_source=cell, opened_at=t,
              closed_at=t + timedelta(hours=closed_h), entry_desired_notional=want, entry_liquidity_cap_notional=cap, pnl=pnl)


def test_no_cap_binding_means_nothing_withdrawable():
    cu = CU.compute([_o(), _o(sl="FLATBASE_SHORT")], now=NOW, equity_now=5000)
    assert cu["platform"]["kept_pct"] == 100.0 and cu["platform"]["capped"] == 0
    assert cu["withdrawable_usd"] <= 0.011 * 5000                         # only the 1 % tolerance itself


def test_one_oversized_sleeve_is_not_withdrawable_money():
    """the review's catch: Spike-Fade-like fills far above their cap must NOT make the uncapped rest withdrawable."""
    rows = [_o(sl="SPIKE_FADE", want=70000, cap=35000)] + [_o(sl="FRENZY_LITE", want=5000, cap=50000) for _ in range(10)]
    cu = CU.compute(rows, now=NOW, equity_now=7000)
    assert cu["withdrawable_usd"] < 0.05 * 7000                            # tiny, not the old ≈ 30 % "idle"
    spike = {r["sleeve"]: r for r in cu["rows"]}["SPIKE_FADE"]
    assert spike["kept_pct"] == 50.0 and spike["capped"] == 1


def test_all_capped_means_withdrawable():
    """every fill wants 2× its cap → half the money can leave with zero size lost."""
    rows = [_o(want=10000, cap=5000), _o(sl="FRENZY_LITE_RETRY", want=20000, cap=10000)]
    cu = CU.compute(rows, now=NOW, equity_now=10000)
    assert abs(cu["withdrawable_usd"] - 5000 * (1 + 0.01) ) < 60           # ~half + the 1 % tolerance band
    assert CU.size_loss(CU.fills(rows, now=NOW), 0.5) == 0.0
    cut = {x["withdraw_pct"]: x["size_cut_pct"] for x in cu["tradeoff"]}
    assert cut[10] == 0.0 and cut[50] == 0.0


def test_withdrawable_matches_tolerance_exactly():
    rows = [_o(want=10000, cap=100000)]                 # uncapped: loss = withdrawn share
    wd, k = CU.withdrawable(CU.fills(rows, now=NOW), 10000, tol=0.01)
    assert abs(wd - 100) < 0.5 and abs(k - 0.99) < 1e-4


def test_reserve_line_reprices():
    """tradeable = a schedule line: the re-price uses T(E_now) ÷ T(E_open), not the equity ratio."""
    tf = lambda e: min(e, 1000 + 0.5 * e)
    rows = [_o(want=10000, cap=100000, days_ago=2)]
    f = CU.equity_at_fn(10000, [], [(NOW - timedelta(hours=1), 10000.0)])   # a +10k deposit after the open → equity then 0 → 1 % guard → ratio clamped
    w = CU.fills(rows, now=NOW, equity_now=10000, equity_at=CU.equity_at_fn(10000, [], [(NOW - timedelta(hours=1), -10000.0)]), tradeable_fn=tf)[0][1]
    assert abs(w - 10000 * tf(10000) / tf(20000)) < 1e-6                   # 6000 / 11000, not ½
    w2 = CU.fills(rows, now=NOW, equity_now=10000, equity_at=f, tradeable_fn=tf)[0][1]
    assert w2 == 10000 * CU.RATIO_MAX                                      # broken history clamped


def test_exclusions_and_window():
    rows = [_o(want=10000, cap=1000, conf="MANUAL"), _o(want=10000, cap=1000, cell="UNMATCHED_PROBE"),
            _o(want=10000, cap=1000, status="OPEN"), _o(want=10000, cap=1000, days_ago=8), _o(sl="MANUAL", want=10000, cap=1000)]
    cu = CU.compute(rows, now=NOW, equity_now=5000)
    assert cu["platform"]["trades"] == 0 and cu["withdrawable_usd"] == 0.0 and cu["tradeoff"] == []


def test_momentum_sleeve_name():
    assert CU.sleeve_of(NS(entry_strategy="MOMENTUM", direction="SHORT")) == "MOMENTUM_SHORT"
    assert CU.sleeve_of(NS(entry_strategy=None, direction="LONG")) == "MOMENTUM_LONG"


def test_a_withdrawal_is_seen_at_once():
    """the operator's case: after half the money leaves, the same trades re-price at half the wanted size."""
    o = _o(want=10000, cap=6000, days_ago=2)
    before = CU.compute([o], now=NOW, equity_now=20000, equity_at=CU.equity_at_fn(20000, [], []))
    after = CU.compute([o], now=NOW, equity_now=10000, equity_at=CU.equity_at_fn(10000, [], [(NOW - timedelta(hours=1), -10000.0)]))
    assert before["platform"]["kept_pct"] == 60.0 and before["withdrawable_usd"] > 7000      # 10k wanted vs 6k cap: 40 % free + 1 %
    assert after["platform"]["kept_pct"] == 100.0 and after["withdrawable_usd"] <= 101         # 5k wanted ≤ 6k cap: only the tolerance


def test_equity_path_profit_and_flows():
    f = CU.equity_at_fn(11000, [(NOW - timedelta(hours=5), 1000.0)], [(NOW - timedelta(hours=2), 500.0)])
    assert f(NOW - timedelta(hours=10)) == 9500.0        # before the +1,000 close and the +500 deposit
    assert f(NOW - timedelta(hours=3)) == 10500.0        # after the close, before the deposit
    assert f(NOW) == 11000.0
    g = CU.equity_at_fn(1000, [(NOW, 5000.0)], [])
    assert g(NOW - timedelta(hours=1)) == 10.0          # guard: never below 1 % of equity now


def test_schedule_tier_jump_is_safe():
    """the live schedule jumps at its first tier (T = E below 10k, 9,750 at 10k): no negative loss, no hop over a failing band."""
    tf = lambda e: e if e < 10000 else 9750 + 0.5 * (e - 10000)
    fl = CU.fills([_o(want=10000, cap=100000)], now=NOW)
    wd, k = CU.withdrawable(fl, 10300, tf)
    for x in range(0, int(wd) + 1, 5):
        assert CU.size_loss(fl, tf(10300 - x) / tf(10300)) <= 0.01 + 1e-9    # every amount up to the answer stays inside 1 %
    assert CU.size_loss(fl, 1.2) == 0.0                                       # k > 1 never reads as a gain


def test_engine_faithful_line_has_no_free_band_at_the_tier():
    """deep review M1: with the fee-reserve pct leg mirrored (2.5 %), T is continuous at the 10k tier → crossing it is not 'free'."""
    def tf(e):   # engine mirror: tier from 10k (target 9,750 + 0.5 per $ above), fee leg 2.5 % of min(e, target)
        tgt = 9750 + 0.5 * (e - 10000) if e >= 10000 else None
        fee = 0.025 * (min(e, tgt) if tgt is not None else e)
        return max(0.0, e - (max(e - tgt, fee) if tgt is not None else fee))
    vals = [tf(9990 + i) for i in range(311)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    wd, _ = CU.withdrawable(CU.fills([_o(want=10000, cap=100000)], now=NOW), 10100, tf)
    assert abs(wd - 149) < 2      # T(10100)=9,800; 1 % = 9,702 = 0.975·E → E 9,951 → $149 (half-free above the tier, then full slope) — not a free jump
