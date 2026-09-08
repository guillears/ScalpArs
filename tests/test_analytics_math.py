"""Analytics math that feeds decisions — combined slip basis, fast-exit fee,
period stats. Display-layer numbers the operator reads at batch reviews.
"""
from types import SimpleNamespace

import config
import main


def _order(**kw):
    d = dict(direction="LONG", pnl=10.0, pnl_percentage=0.5, total_fee=1.0,
             investment=100.0, notional_value=2000.0,
             entry_slippage_pct=None, exit_slippage_pct=None)
    d.update(kw)
    return SimpleNamespace(**d)


def test_combined_slip_percent_and_dollar_share_one_basis():
    # Sep-7 ship: Slip % and Slip $ must be the SAME calculation in two units
    trades = [
        _order(entry_slippage_pct=0.01, exit_slippage_pct=-0.02),  # comb -0.01% -> -$0.20
        _order(exit_slippage_pct=0.05),                            # comb +0.05% -> +$1.00
        _order(),                                                  # no slip data -> excluded
    ]
    r = main._period_stats("T", trades)
    assert r["slippage_count"] == 2
    assert abs(r["avg_slippage_pct"] - 0.02) < 1e-9
    assert abs(r["total_slippage_usd"] - 0.80) < 1e-9


def test_slip_none_when_no_data():
    r = main._period_stats("T", [_order(), _order()])
    assert r["avg_slippage_pct"] is None and r["total_slippage_usd"] is None


def test_period_stats_core_fields():
    trades = [_order(pnl=20.0), _order(pnl=-10.0, pnl_percentage=-0.5)]
    r = main._period_stats("T", trades)
    assert r["count"] == 2 and r["win_rate"] == 50.0
    assert abs(r["total_pnl"] - 10.0) < 1e-9
    assert abs(r["profit_factor"] - 2.0) < 1e-9


def test_fast_exit_fee_follows_config(monkeypatch):
    # Sep-7 hardcode-audit fix: was a 0.063 literal; must track live rates now
    monkeypatch.setattr(config.trading_config, "taker_fee", 0.00045)
    monkeypatch.setattr(config.trading_config, "maker_fee", 0.00018)
    monkeypatch.setattr(config.trading_config, "maker_entry_enabled", True)
    assert abs(main._fast_exit_fee_pct() - 0.063) < 1e-9           # maker in + taker out
    monkeypatch.setattr(config.trading_config, "maker_entry_enabled", False)
    assert abs(main._fast_exit_fee_pct() - 0.090) < 1e-9           # taker both sides
    monkeypatch.setattr(config.trading_config, "taker_fee", 0.0005)  # discount lost
    assert abs(main._fast_exit_fee_pct() - 0.100) < 1e-9
