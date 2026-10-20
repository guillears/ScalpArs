"""🏦 (318) Live card deposit detection: the Oct-20 +$1,045.08 Spot→UM transfer sat behind the 10-min transfer-rows cache
(booked 02:08:05 UTC, first seen 02:18:10). Pins: the card path's short cache, the per-re-read INFO line, cache invalidation,
the 'Invested' numbers. Pure — the Binance call is a fake, no network."""
import asyncio
import os
import re

import pytest

from services import binance_service as bsmod

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
T_TX = 1_792_462_085_311          # the transfer's booking ms (2026-10-20T02:08:05.311Z)


def _main_fn(name):
    src = open(os.path.join(ROOT, "main.py")).read()
    a = src.index(f"def {name}("); b = src.index("\n\n\n", a)
    ns = {}; exec(src[a:b], ns); return ns[name], src


class _FakeEx:
    """fapiPrivateGetIncome fake: ascending rows from startTime, limit-capped (Binance semantics)."""
    def __init__(self, rows):
        self.rows = rows; self.calls = 0

    async def fapiPrivateGetIncome(self, p):
        self.calls += 1
        sel = [r for r in self.rows if p["startTime"] <= r["time"] <= p["endTime"] and r["incomeType"] == p["incomeType"]]
        return sorted(sel, key=lambda r: r["time"])[: p["limit"]]


def _svc(rows):
    s = bsmod.BinanceService.__new__(bsmod.BinanceService)
    s.exchange = _FakeEx(rows); s._tx_rows_cache = {}; s._cap_flow_cache = {}
    return s


def _row(ts, amt, tid, asset="USDT"):
    return {"time": ts, "income": str(amt), "asset": asset, "incomeType": "TRANSFER", "tranId": tid}


def test_card_path_rereads_after_60s_while_default_cache_holds_10min(monkeypatch):
    clock = {"m": 1000.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["m"])
    rows = [_row(T_TX - 86_400_000, 50.0, 1)]
    s = _svc(rows); start = T_TX - 10 * 86_400_000
    assert asyncio.run(s.get_transfer_rows(start)) == [(T_TX - 86_400_000, 50.0)]
    rows.append(_row(T_TX, 1045.07724177, 2))                       # the operator's transfer lands after the read
    clock["m"] += 61
    assert len(asyncio.run(s.get_transfer_rows(start))) == 1           # default 600 s cache: still stale (the Oct-20 delay)
    got = asyncio.run(s.get_transfer_rows(start, max_age_s=bsmod.LIVE_CARD_TRANSFER_CACHE_S))
    assert (T_TX, 1045.07724177) in got and s.exchange.calls == 2       # card path: re-read after 60 s
    assert bsmod.LIVE_CARD_TRANSFER_CACHE_S <= 60 and bsmod.TRANSFER_ROWS_CACHE_S == 600


def test_invalidate_forces_a_reread(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 5.0)
    rows = []; s = _svc(rows); start = T_TX - 1000
    assert asyncio.run(s.get_transfer_rows(start)) == []
    rows.append(_row(T_TX, 1045.08, 9))
    assert asyncio.run(s.get_transfer_rows(start)) == []                # same instant: cached
    s.invalidate_flow_caches()
    assert asyncio.run(s.get_transfer_rows(start)) == [(T_TX, 1045.08)] and s.exchange.calls == 2


def test_paging_and_asset_filter():
    rows = [_row(T_TX - 5_000_000 + i, 1.0, i) for i in range(1500)] + [_row(T_TX, 1045.08, 99999), _row(T_TX + 1, 7.0, 100000, "BNB")]
    s = _svc(rows)
    got = asyncio.run(s.get_transfer_rows(T_TX - 6_000_000))
    assert len(got) == 1501 and got[-1] == (T_TX, 1045.08)             # pages forward past 1000, newest row kept, BNB dropped


def test_reread_info_line(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO, logger=bsmod.logger.name)
    s = _svc([_row(T_TX, 1045.08, 1)])
    asyncio.run(s.get_transfer_rows(T_TX - 1000))
    msgs = [r.getMessage() for r in caplog.records if "[FLOW_READ]" in r.getMessage()]
    assert len(msgs) == 1 and "2026-10-20T02:08:05Z +1,045.08" in msgs[0] and "1 rows / 1 page(s)" in msgs[0]
    asyncio.run(s.get_transfer_rows(T_TX - 1000))                     # a cache hit logs nothing
    assert len([r for r in caplog.records if "[FLOW_READ]" in r.getMessage()]) == 1
    assert "newest none" in bsmod.transfer_reread_line(0, 0, 0, [])


def test_invested_numbers_oct20_card():
    card, _ = _main_fn("live_card_numbers"); inv, _ = _main_fn("live_invested_numbers")
    init, pnl = card(2599.08, 1227.25, 1045.08)
    assert (init, pnl) == (1227.25, 326.75)                           # the deposit is not P&L
    assert inv(1227.25, 1045.08, pnl) == (2272.33, 14.38)             # Invested $2,272.33 · +14.38 % (not +111.78 %)
    assert inv(1227.25, -227.25, -100.0) == (1000.0, -10.0)           # a withdrawal reduces the invested basis
    assert inv(None, 0.0, 0.0) == (None, None) and inv(100.0, -100.0, 5.0) == (0.0, None)


def test_wiring():
    src = open(os.path.join(ROOT, "main.py")).read(); html = open(os.path.join(ROOT, "templates/index.html")).read()
    nf = src[src.index("async def _live_net_flows("):src.index("async def _live_baseline(")]
    assert "max_age_s=LIVE_CARD_TRANSFER_CACHE_S" in nf
    assert "until=_now_m + LIVE_FLOW_MEMO_S" in src and int(re.search(r"\nLIVE_FLOW_MEMO_S = (\d+)", src).group(1)) <= 60
    # the settle lag must stay above BOTH cache ages, or the cursor could pass an unseen row
    settle_s = int(re.search(r"LIVE_FLOW_SETTLE_MS = (\d+) \* 60 \* 1000", src).group(1)) * 60
    assert settle_s > bsmod.TRANSFER_ROWS_CACHE_S > bsmod.LIVE_CARD_TRANSFER_CACHE_S
    assert '"invested": _live_invested' in src and '"pnl_pct": _live_pnl_pct' in src
    assert "Invested: ${formatUSD(_inv)}" in html and "start ${_fl > 0 ? '+' : '-'}" in html
    assert "Initial Balance: ${formatUSD(_seed)}" in html              # paper card unchanged


def test_reread_log_only_on_change_or_every_10min(monkeypatch, caplog):
    """(318 review) the card re-reads every 60 s — an unchanged re-read logs nothing until 10 min have passed; a new row logs at once."""
    import logging
    caplog.set_level(logging.INFO, logger=bsmod.logger.name)
    clock = {"m": 100.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["m"])
    rows = [_row(T_TX, 1045.08, 1)]; s = _svc(rows); start = T_TX - 1000
    n = lambda: len([r for r in caplog.records if "[FLOW_READ]" in r.getMessage()])
    asyncio.run(s.get_transfer_rows(start, max_age_s=60)); assert n() == 1
    clock["m"] += 61; asyncio.run(s.get_transfer_rows(start, max_age_s=60)); assert n() == 1      # same rows → quiet
    rows.append(_row(T_TX + 5000, -20.0, 2))
    clock["m"] += 61; asyncio.run(s.get_transfer_rows(start, max_age_s=60)); assert n() == 2      # changed → logged
    clock["m"] += 601; asyncio.run(s.get_transfer_rows(start, max_age_s=60)); assert n() == 3     # 10-min heartbeat
