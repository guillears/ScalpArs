"""External-flow classification — _external_flow_rows against the real matcher,
in-memory BnbSwapLog rows, and monkeypatched exchange feeds.

Fixtures mirror the REAL Aug-27 verification set: bot swap legs must be
excluded, genuine deposits/withdrawals kept, small leftover returns absorbed.
"""
from datetime import datetime, timezone

import main
import models


import pytest


@pytest.fixture(autouse=True)
def _clean_recon_set():
    # caveman-review: kept rows land in the module-global log-dedup set — isolate tests
    main._FLOW_RECON_SEEN.clear()
    yield
    main._FLOW_RECON_SEEN.clear()


def _feed(monkeypatch, transfers, capital=None):
    async def _rows(start_ms):
        return [t for t in transfers if t[0] >= start_ms]
    async def _cap(start_ms):
        return capital or []
    monkeypatch.setattr(main.binance_service, "get_transfer_rows", _rows)
    monkeypatch.setattr(main.binance_service, "get_capital_flows", _cap)


def _swap(db, ts_ms, usdt, swap_type="scheduled_buy"):
    db.add(models.BnbSwapLog(
        swap_type=swap_type, amount_usdt=abs(usdt), bnb_price=700.0,
        amount_bnb=abs(usdt) / 700.0, pre_bnb_usd=0.0, post_bnb_usd=abs(usdt),
        pre_usdt=abs(usdt), post_usdt=0.0, is_paper=False,
        timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(tzinfo=None)))


T0 = 1_756_200_000_000  # arbitrary epoch-ms anchor


async def test_swap_legs_excluded_deposits_kept(db, monkeypatch):
    _swap(db, T0 + 10_000, 209.0)                       # bot buy: -209 out, +0.22 leftover back
    await db.flush()
    transfers = [
        (T0 + 12_000, -209.0),                          # swap outbound leg -> excluded
        (T0 + 14_000, +0.22),                           # leftover sweep    -> excluded
        (T0 + 500_000, +925.17),                        # real deposit      -> kept
        (T0 + 900_000, -3000.0),                        # real withdrawal   -> kept
    ]
    _feed(monkeypatch, transfers)
    rows = await main._external_flow_rows(db, T0)
    amounts = sorted(a for _, a in rows)
    assert amounts == [-3000.0, 925.17]


async def test_sell_swap_inbound_leg_excluded(db, monkeypatch):
    _swap(db, T0 + 10_000, 150.0, swap_type="manual_sell")   # sell: +150 comes BACK to futures
    await db.flush()
    _feed(monkeypatch, [(T0 + 12_000, +150.0), (T0 + 600_000, +500.0)])
    rows = await main._external_flow_rows(db, T0)
    assert [a for _, a in rows] == [500.0]


async def test_no_swaps_everything_external(db, monkeypatch):
    _feed(monkeypatch, [(T0 + 1000, +2625.75), (T0 + 2000, -50.0)])
    rows = await main._external_flow_rows(db, T0)
    assert sorted(a for _, a in rows) == [-50.0, 2625.75]


async def test_similar_but_distant_transfer_not_eaten(db, monkeypatch):
    # a transfer of swap-like SIZE far outside the +-3min window stays external
    _swap(db, T0 + 10_000, 209.0)
    await db.flush()
    _feed(monkeypatch, [(T0 + 3_600_000, -209.0)])       # 1h later
    rows = await main._external_flow_rows(db, T0)
    assert [a for _, a in rows] == [-209.0]
