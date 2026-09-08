"""Fee auto-sync paths — _sync_fee_rates (Sep-3 ship + Sep-7 integration fix).

Pinned: discount math, BNB-dry fallback to full rate, fail-open on API error
with 1h retry, 24h success throttle, toggle-off inert.
"""
import asyncio
import time

import config
import services.trading_engine as te


def _run(coro):
    return asyncio.run(coro)  # caveman-review: no leaked event loops


def _arm(monkeypatch, rates, paper_bnb=100.0, auto=True):
    eng = te.trading_engine
    monkeypatch.setattr(eng, "is_paper_mode", True, raising=False)
    monkeypatch.setattr(eng, "paper_bnb_balance_usd", paper_bnb, raising=False)
    monkeypatch.setattr(eng, "_fee_sync_at", 0, raising=False)
    monkeypatch.setattr(config.trading_config, "fee_auto_fetch", auto)
    monkeypatch.setattr(config.trading_config, "taker_fee", 0.0004)   # differs from synced 0.00045 so the CHANGE branch runs
    monkeypatch.setattr(config.trading_config, "maker_fee", 0.00016)
    monkeypatch.setattr(config.trading_config, "trading_fee", 0.0004)  # legacy mirror — must not leak (caveman-review)
    if isinstance(rates, Exception):
        async def _r():
            raise rates
    else:
        async def _r():
            return rates
    monkeypatch.setattr(te.binance_service, "get_commission_rates", _r)
    return eng


def test_discount_applied(monkeypatch):
    eng = _arm(monkeypatch, {"maker": 0.0002, "taker": 0.0005, "fee_burn": True})
    _run(eng._sync_fee_rates())
    assert abs(config.trading_config.taker_fee - 0.00045) < 1e-12   # 0.0005 x 0.9
    assert abs(config.trading_config.maker_fee - 0.00018) < 1e-12   # 0.0002 x 0.9
    assert config.trading_config.trading_fee == config.trading_config.taker_fee  # legacy mirror updated


def test_bnb_dry_loses_discount(monkeypatch):
    eng = _arm(monkeypatch, {"maker": 0.0002, "taker": 0.0005, "fee_burn": True},
               paper_bnb=0.0)
    _run(eng._sync_fee_rates())
    assert abs(config.trading_config.taker_fee - 0.0005) < 1e-12   # full rate


def test_fee_burn_off_no_discount(monkeypatch):
    eng = _arm(monkeypatch, {"maker": 0.0002, "taker": 0.0005, "fee_burn": False})
    _run(eng._sync_fee_rates())
    assert abs(config.trading_config.taker_fee - 0.0005) < 1e-12


def test_fail_open_keeps_config_and_schedules_retry(monkeypatch):
    eng = _arm(monkeypatch, RuntimeError("api down"))
    _run(eng._sync_fee_rates())
    assert abs(config.trading_config.taker_fee - 0.0004) < 1e-12   # untouched (configured value)
    # retry scheduled ~1h out (throttle set to now - 23h)
    assert 22 * 3600 < time.time() - eng._fee_sync_at < 24 * 3600


def test_success_throttles_24h(monkeypatch):
    eng = _arm(monkeypatch, {"maker": 0.0002, "taker": 0.0005, "fee_burn": True})
    _run(eng._sync_fee_rates())
    assert time.time() - eng._fee_sync_at < 60
    # second call inside the throttle must not refetch. The sync swallows exceptions
    # (fail-open), so "no raise" proves nothing (caveman-review R2) — instead: a broken
    # throttle would hit the poisoned fetch, take the FAIL path, and rewind _fee_sync_at
    # by 23h. Assert the stamp is untouched.
    _stamp = eng._fee_sync_at
    async def _boom():
        raise RuntimeError("must not be called inside throttle")
    monkeypatch.setattr(te.binance_service, "get_commission_rates", _boom)
    _run(eng._sync_fee_rates())
    assert eng._fee_sync_at == _stamp                              # throttled: fail path never ran


def test_toggle_off_inert(monkeypatch):
    eng = _arm(monkeypatch, RuntimeError("must not be called"), auto=False)
    _run(eng._sync_fee_rates())
    assert abs(config.trading_config.taker_fee - 0.0004) < 1e-12
    assert eng._fee_sync_at == 0                                   # caveman-review R3: falsifiable — a broken toggle takes the fail path and rewinds the stamp
