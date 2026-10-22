"""🏦 (DECISION_LOG 344) LIVE balance card = Binance USDT margin balance (incl. untracked positions) · USDT in Open Orders = position margin.

Operator bug: an app-opened IOSTUSDT long (not imported, so no DB row) lost ~$170 while the card stayed flat — the card was
walletBalance + the DB rows' unrealized. Pinned here (no network, stubbed binance_service, in-memory SQLite):
- get_balance parses marginBalance / unrealizedProfit / initialMargin from the raw USDT asset (real ccxt parser on a v3 payload);
- an untracked position's unrealized loss lowers total_portfolio; usdt_in_orders = initial margin and does not rise with the loss;
- a DB-tracked row is never counted twice; a failed read stays zeros / hidden lines; Multi-Assets BNB is not counted twice;
- the NAV / investor path (_portfolio_components) uses the same equity, still raises on a failed read; an existing baseline never moves.
"""
import asyncio
import types
from datetime import datetime

import ccxt
import pytest

import main
import models
from services.binance_service import BinanceService


def _raw(wallet, unrl, im, free=None, bnb=0.0, bnb_price=600.0, multi=False, drop=()):
    usdt = {"asset": "USDT", "walletBalance": str(wallet), "unrealizedProfit": str(unrl), "marginBalance": str(wallet + unrl),
            "initialMargin": str(im), "positionInitialMargin": str(im), "openOrderInitialMargin": "0",
            "maxWithdrawAmount": str(free if free is not None else wallet + min(unrl, 0) - im),
            "availableBalance": str(free if free is not None else wallet + unrl - im)}
    for k in drop:
        usdt.pop(k, None)
    assets = [usdt]
    if bnb:
        assets.append({"asset": "BNB", "walletBalance": str(bnb), "unrealizedProfit": "0", "marginBalance": str(bnb), "initialMargin": "0",
                       "maxWithdrawAmount": str(bnb), "availableBalance": str(bnb)})
    tmb = wallet + unrl + (bnb * bnb_price * 0.95 if multi else 0.0)          # Multi-Assets: the account total also values BNB
    return {"totalMaintMargin": "1.0", "totalMarginBalance": str(tmb), "assets": assets}


async def _svc(raw, info=None):
    """Run the REAL BinanceService.get_balance over the REAL ccxt parser of a /fapi/v3/account payload (no network)."""
    ex = ccxt.binanceusdm({"options": {"defaultType": "future"}})
    parsed = ex.parse_balance_custom(raw, "linear")
    parsed["info"] = raw if info is None else info                            # info: a raw payload ccxt itself could not parse

    async def _noop():
        return None

    async def _fb():
        return parsed
    fake = types.SimpleNamespace(load_markets=_noop, exchange=types.SimpleNamespace(fetch_balance=_fb))
    return await BinanceService.get_balance(fake)


async def _svc_fail():
    async def _boom():
        raise RuntimeError("down")
    return await BinanceService.get_balance(types.SimpleNamespace(load_markets=_boom, exchange=None))


def _svc_balance(raw, info=None):
    return asyncio.run(_svc(raw, info))


def _svc_failed():
    return asyncio.run(_svc_fail())


# ------------------------------------------------------------------ service parsing

def test_service_reads_margin_balance_unrealized_and_initial_margin():
    b = _svc_balance(_raw(1000.0, -170.0, 50.0))
    assert b["ok"] is True
    assert b["usdt_total"] == 1000.0 and b["usdt_unrealized"] == -170.0 and b["usdt_margin_balance"] == 830.0
    assert b["usdt_initial_margin"] == 50.0
    for k in ("usdt_free", "usdt_used", "usdt_total", "bnb_free", "bnb_total", "total_portfolio", "maint_margin", "margin_balance"):
        assert k in b                                                            # legacy keys unchanged for other callers


def test_service_fallbacks_when_fields_missing():
    b = _svc_balance(_raw(1000.0, -170.0, 50.0, drop=("marginBalance", "initialMargin")))
    assert b["usdt_margin_balance"] == 830.0                                     # walletBalance + unrealizedProfit
    assert b["usdt_initial_margin"] == 50.0                                      # positionInitialMargin + openOrderInitialMargin


def test_service_failed_read_new_keys_zero():
    b = _svc_failed()
    assert b["ok"] is False
    assert b["usdt_margin_balance"] == 0 and b["usdt_unrealized"] == 0 and b["usdt_initial_margin"] == 0 and b["usdt_total"] == 0


def test_service_multi_assets_usdt_leg_excludes_bnb():
    b = _svc_balance(_raw(1000.0, -170.0, 50.0, bnb=0.5, multi=True))
    assert b["usdt_margin_balance"] == 830.0                                     # per-USDT asset — never the BNB-inclusive account total
    assert b["margin_balance"] > 830.0                                           # (the account total does include BNB in Multi-Assets)
    assert b["bnb_total"] == 0.5


# ------------------------------------------------------------------ /api/balance live branch + NAV components

@pytest.fixture
def live(monkeypatch):
    state = {"bal": None, "bnb_price": 600.0}

    async def _init(_db):
        return None
    monkeypatch.setattr(main.trading_engine, "initialize", _init)
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", False, raising=False)

    async def _gb():
        return state["bal"]

    async def _px():
        return state["bnb_price"]

    async def _spot():
        return None

    async def _rows(start_ms, max_age_s=None):
        return []
    monkeypatch.setattr(main.binance_service, "get_balance", _gb)
    monkeypatch.setattr(main.binance_service, "get_bnb_price", _px)
    monkeypatch.setattr(main.binance_service, "get_spot_balance_usd", _spot)
    monkeypatch.setattr(main.binance_service, "get_transfer_rows", _rows, raising=False)
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)
    main._LIVE_FLOW_MEMO.clear()
    return state


async def _row(db, pair="BTCUSDT", entry=100.0, cur=90.0, qty=5.0, inv=50.0):
    o = models.Order(pair=pair, direction="LONG", status="OPEN", is_paper=False, entry_price=entry, current_price=cur, quantity=qty,
                     investment=inv, leverage=10, notional_value=inv * 10, confidence="HIGH", opened_at=datetime(2026, 10, 20, 12, 0, 0), entry_fee=0.0)
    db.add(o)
    await db.flush()
    return o


async def test_untracked_loss_lowers_total_and_margin_does_not_rise(db, live):
    live["bal"] = await _svc(_raw(1000.0, 0.0, 50.0, bnb=0.1))
    flat = await main.get_balance(db)
    live["bal"] = await _svc(_raw(1000.0, -170.0, 45.0, bnb=0.1))      # the untracked long loses $170 (IM drops with mark)
    lost = await main.get_balance(db)
    assert flat["total_portfolio"] == 1060.0 and lost["total_portfolio"] == 890.0     # 1000 − 170 + BNB 60
    assert lost["usdt_unrealized"] == -170.0
    assert flat["usdt_in_orders"] == 50.0 and lost["usdt_in_orders"] == 45.0          # the locked margin — never margin + open loss
    assert lost["usdt_in_orders"] <= flat["usdt_in_orders"]


async def test_tracked_row_not_double_counted(db, live):
    await _row(db, entry=100.0, cur=90.0, qty=5.0)                          # DB unrealized −50
    live["bal"] = await _svc(_raw(1000.0, -220.0, 95.0))                 # Binance: tracked −50 + untracked −170
    r = await main.get_balance(db)
    assert abs(r["total_portfolio"] - 780.0) < 1e-6                             # 1000 − 220 (no BNB) — the DB −50 is NOT added again
    assert r["usdt_unrealized"] == -220.0 and r["usdt_in_orders"] == 95.0


async def test_multi_assets_bnb_counted_once(db, live):
    live["bal"] = await _svc(_raw(1000.0, -170.0, 50.0, bnb=0.5, multi=True))
    r, pc = await main.get_balance(db), await main._portfolio_components(db)
    assert abs(r["total_portfolio"] - (830.0 + 0.5 * 600.0)) < 1e-6
    assert abs(pc["total_equity"] - (830.0 + 300.0)) < 1e-6


async def test_failed_read_zeros_and_hidden_lines(db, live):
    db.add(models.BotState(live_initial_total_usd=1000.0, live_baseline_at=datetime(2026, 10, 1)))
    await db.flush()
    live["bal"] = await _svc_fail()
    r = await main.get_balance(db)
    with pytest.raises(RuntimeError):
        await main._portfolio_components(db)                                # NAV never priced off a failed read
    assert r["total_portfolio"] == 0.0 and r["usdt_in_orders"] == 0.0
    assert r["usdt_unrealized"] is None                                         # the "incl. open P&L" sub-line hides on a failed read
    assert r["pnl_vs_start"] is None and r["initial_balance"] is None


async def test_nav_components_use_true_equity(db, live):
    await _row(db, entry=100.0, cur=90.0, qty=5.0)
    live["bal"] = await _svc(_raw(1000.0, -220.0, 95.0, bnb=0.1))
    pc, pv, card = await main._portfolio_components(db), await main._get_portfolio_value(db), await main.get_balance(db)
    assert abs(pc["total_equity"] - 840.0) < 1e-6 and abs(pv - 840.0) < 1e-6   # 1000 − 220 + 60 — same as the card
    assert abs(card["total_portfolio"] - pv) < 1e-6
    assert abs(pc["realized_equity"] - 1060.0) < 1e-6 and pc["unrealized"] == -220.0 and pc["margin"] == 95.0


async def test_nav_snapshot_prices_untracked_loss(db, live):
    db.add(models.Investor(name="A", shares=100.0, total_deposited=1000.0, total_withdrawn=0.0))
    await db.flush()
    live["bal"] = await _svc(_raw(1000.0, -170.0, 50.0))
    assert await main._nav_snapshot_once(db, now_utc=datetime(2026, 10, 22, 15, 0, 0))
    from sqlalchemy import select
    snap = (await db.execute(select(models.NavSnapshot))).scalar_one()
    assert abs(snap.portfolio_value - 830.0) < 1e-6 and abs(snap.nav_per_share - 8.3) < 1e-6


async def test_existing_baseline_never_moves_and_pnl_includes_untracked(db, live):
    db.add(models.BotState(live_initial_total_usd=1000.0, live_baseline_at=datetime(2026, 10, 1)))
    await db.flush()
    live["bal"] = await _svc(_raw(1000.0, -170.0, 50.0))
    r = await main.get_balance(db)
    from sqlalchemy import select
    st = (await db.execute(select(models.BotState))).scalar_one()
    assert st.live_initial_total_usd == 1000.0 and r["initial_balance"] == 1000.0
    assert r["pnl_vs_start"] == -170.0                                          # the untracked open loss now shows in P&L


async def test_one_time_baseline_backderivation_ignores_untracked_unrealized(db, live):
    db.add(models.BotState())
    await _row(db, entry=100.0, cur=90.0, qty=5.0)                          # tracked −50; untracked −170
    await db.flush()
    live["bal"] = await _svc(_raw(1000.0, -220.0, 95.0))
    await main.get_balance(db)
    from sqlalchemy import select
    st = (await db.execute(select(models.BotState))).scalar_one()
    assert abs(st.live_initial_total_usd - 1000.0) < 1e-6                       # 780 − (0 closed − 220 open) − 0 flows


def test_ui_and_wiring_pins():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, "templates", "index.html")).read()
    src = open(os.path.join(root, "main.py")).read()
    assert 'id="portfolio-open-pnl"' in html and "data.usdt_unrealized" in html and ">position margin<" in html
    assert "_open_unrealized_pnl(_live_open) + bnb_usd" not in src                # the pre-344 card formula is gone
    assert "\"usdt_in_orders\": balance['usdt_used']" not in src
    import re
    calls = [l for l in src.splitlines() if re.search(r"(?<!def )live_equity_parts\(", l) and not l.lstrip().startswith("#")]
    assert len(calls) == 2, calls                                               # /api/balance + _portfolio_components
    assert any("_usdt_eq, _usdt_unrl, _usdt_im = live_equity_parts(balance, _live_open)" in l for l in calls)
    assert any("_usdt_eq, _unrl_live, _im = live_equity_parts(bal, open_orders)" in l for l in calls)
    # the card sub-line hides on null / non-finite; both text exports print "incl. open P&L" next to "Portfolio now" (D12), omitted when null
    assert "const _has = data.usdt_unrealized != null && isFinite(Number(data.usdt_unrealized));" in html
    exp = [l for l in html.splitlines() if "lines.push(`Portfolio now:" in l]
    assert len(exp) == 2 and all("${window._lastOpenPnl != null ? ` (incl. open P&L " in l for l in exp)
    assert "window._lastOpenPnl = (data.usdt_unrealized != null && isFinite(Number(data.usdt_unrealized))) ? Number(data.usdt_unrealized) : null;" in html


def test_service_non_finite_and_empty_fields_fall_back():
    raw = _raw(1000.0, -170.0, 50.0)
    info = _raw(1000.0, -170.0, 50.0)
    u = info["assets"][0]
    u["marginBalance"] = "nan"; u["initialMargin"] = ""; u["unrealizedProfit"] = "inf"; u["maxWithdrawAmount"] = None
    b = _svc_balance(raw, info)
    assert b["ok"] is True
    assert b["usdt_unrealized"] == 0.0 and b["usdt_margin_balance"] == 1000.0    # non-finite → 0 / wallet + 0
    assert b["usdt_initial_margin"] == 50.0                                      # '' → position + open-order IM
    assert b["usdt_total"] == 1000.0 and b["usdt_free"] > 0                      # None maxWithdrawAmount → ccxt free
