"""Cache-rebuild key preservation — the TWICE-BITTEN omission class (May-15
phantom_be_aggr, May-20 phantom_regime, Sep-7 HARD_TP write storm).

Runs the REAL update_orders_cache against an in-memory DB with one OPEN order
and a seeded old cache generation, then asserts every state key that must
survive the 1s rebuild actually does.
"""
from datetime import datetime

import models
import config
import services.trading_engine as te

CONF = next(iter(config.trading_config.confidence_levels))  # rename-proof (caveman-review)

MUST_SURVIVE = {
    "_htp_persisted_lvl": 2,                      # HARD_TP persist dedup (write-storm guard)
    "_sp_lock_peak_persisted": True,              # spike lock-peak persist dedup
    "_trailing_pullback_first_at": datetime(2026, 9, 7, 12, 0, 0),
    "_trailing_pullback_first_pnl_pct": 0.31,
    "runner_peak_stretch": 0.42,
    "_belock_taint": True,                        # Jul-28 fix (regression guard)
}


async def test_state_keys_survive_rebuild(db, monkeypatch):
    order = models.Order(
        pair="TESTUSDT", direction="LONG", status="OPEN", is_paper=False,
        entry_price=1.0, quantity=100.0, investment=100.0, leverage=10,
        notional_value=1000.0, opened_at=datetime.utcnow(), confidence=CONF,
    )
    db.add(order)
    await db.flush()

    monkeypatch.setattr(te.trading_engine, "is_paper_mode", False, raising=False)
    old_info = {"id": order.id, "peak_pnl": 1.5, "trough_pnl": -0.2,
                "peak_ema5_gap": 0.0, "direction": "LONG",
                "high_price": 1.02, "low_price": 0.99}
    old_info.update(MUST_SURVIVE)
    monkeypatch.setattr(te, "_open_orders_cache", {"TESTUSDT": [old_info]})

    await te.trading_engine.update_orders_cache(db)

    rebuilt = te._open_orders_cache["TESTUSDT"][0]
    for key, val in MUST_SURVIVE.items():
        assert rebuilt.get(key) == val, f"{key} wiped by rebuild (write-storm class)"
    # realtime peak preserved over the (stale) DB value
    assert rebuilt["peak_pnl"] >= 1.5


async def test_fresh_boot_defaults_absent(db, monkeypatch):
    """Restart case: no old generation -> keys simply absent (consumers default)."""
    order = models.Order(
        pair="TESTUSDT", direction="LONG", status="OPEN", is_paper=False,
        entry_price=1.0, quantity=100.0, investment=100.0, leverage=10,
        notional_value=1000.0, opened_at=datetime.utcnow(), confidence=CONF,
    )
    db.add(order)
    await db.flush()
    monkeypatch.setattr(te.trading_engine, "is_paper_mode", False, raising=False)
    monkeypatch.setattr(te, "_open_orders_cache", {})
    await te.trading_engine.update_orders_cache(db)
    rebuilt = te._open_orders_cache["TESTUSDT"][0]
    assert rebuilt.get("_htp_persisted_lvl") is None
    assert rebuilt.get("_trailing_pullback_first_at") is None
