"""📒 DECISION_LOG 455: the accountant export — futures trades (gross = pnl + fees, net = pnl + funding, margin × leverage = size), the
monthly summary checked against Binance's income, transfers, spot FIFO lots, and a valid .xlsx (one download)."""
import asyncio
import io
import os
import sys
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services import accounting_export as AE  # noqa: E402

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
def MS(s):
    """'YYYY-MM-DD HH:MM' UTC → epoch ms."""
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _o(opened, closed, pnl, ef=1.0, xf=1.0, fund=-0.5, size=1000.0, lev=5.0, direction="LONG"):
    return SimpleNamespace(opened_at=datetime.strptime(opened, "%Y-%m-%d %H:%M"), closed_at=datetime.strptime(closed, "%Y-%m-%d %H:%M"),
                           pair="LINKUSDT", direction=direction, quantity=70.0, entry_price=14.2857, exit_price=14.5, leverage=lev,
                           notional_value=size, pnl=pnl, entry_fee=ef, exit_fee=xf, funding_fee_usd=fund)


def test_trade_row_math():
    r = AE.trade_row(_o("2026-10-20 10:00", "2026-10-20 11:00", 10.0))
    assert r["Gross P&L (USDT)"] == 12.0 and r["Trading fees (USDT)"] == -2.0 and r["Funding (USDT)"] == -0.5
    assert r["Net P&L (USDT)"] == 9.5 and r["Result"] == "Win"
    assert r["Margin (USDT)"] == 200.0 and r["Leverage"] == 5.0 and r["Position size (USDT)"] == 1000.0
    assert list(AE.TRADE_COLS).index("Margin (USDT)") < AE.TRADE_COLS.index("Leverage") < AE.TRADE_COLS.index("Position size (USDT)")
    assert not any("source" in c.lower() or "bot" in c.lower() or "manual" in c.lower() for c in AE.TRADE_COLS)   # never disclosed
    loss = AE.trade_row(_o("2026-10-20 10:00", "2026-10-20 11:00", 0.3, fund=-0.5))
    assert loss["Net P&L (USDT)"] == -0.2 and loss["Result"] == "Loss"                      # funding turns a small win into a loss
    nofund = AE.trade_row(SimpleNamespace(**dict(vars(_o("2026-10-20 10:00", "2026-10-20 11:00", 1.0)), funding_fee_usd=None, exit_fee=None)))
    assert nofund["Funding (USDT)"] == 0.0 and nofund["Gross P&L (USDT)"] == 2.0


def test_monthly_rows_check_balance_and_flows():
    trades = AE.trade_rows([_o("2026-10-20 10:00", "2026-10-20 11:00", 10.0), _o("2026-11-02 10:00", "2026-11-02 11:00", -5.0)])
    inc = [dict(incomeType="REALIZED_PNL", tranId=1, asset="USDT", income=12.0, time=MS("2026-10-20 11:00")),
           dict(incomeType="COMMISSION", tranId=2, asset="BNB", income=-0.004, time=MS("2026-10-20 11:00")),   # 0.004 BNB × 500 = −2
           dict(incomeType="FUNDING_FEE", tranId=3, asset="USDT", income=-0.5, time=MS("2026-10-20 10:30")),
           dict(incomeType="TRANSFER", tranId=4, asset="USDT", income=1000.0, time=MS("2026-10-15 09:00")),
           dict(incomeType="REALIZED_PNL", tranId=5, asset="USDT", income=-3.0, time=MS("2026-11-02 11:00")),
           dict(incomeType="COMMISSION", tranId=6, asset="USDT", income=-2.0, time=MS("2026-11-02 11:00")),
           dict(incomeType="FUNDING_FEE", tranId=7, asset="USDT", income=-0.5, time=MS("2026-11-02 10:30"))]
    flows = [(MS("2026-10-15 08:00"), 1000.0, "USDT"), (MS("2026-11-03 08:00"), -200.0, "USDT"), (MS("2026-11-03 09:00"), 0.5, "BNB")]
    usd = lambda a, ms: 1.0 if a in AE.STABLES else (500.0 if a == "BNB" else None)   # noqa: E731
    m = {r["Month"]: r for r in AE.monthly_rows(trades, inc, flows, usd, 1500.0, True)}
    oc, nv, tot = m["2026-10"], m["2026-11"], m["TOTAL"]
    assert (oc["Trades"], oc["Wins"], nv["Losses"]) == (1, 1, 1)
    assert oc["Net P&L (USDT)"] == 9.5 and oc["Binance net P&L (check)"] == 9.5 and oc["Difference"] == 0.0
    assert nv["Net P&L (USDT)"] == -5.5 and nv["Binance net P&L (check)"] == -5.5
    assert oc["Deposits (USDT)"] == 1000.0 and nv["Withdrawals (USDT)"] == 200.0                          # the BNB deposit is not USDT
    assert oc["Net transfers into futures (USDT)"] == 1000.0
    assert oc["End-of-month futures balance (USDT)"] == 1500.0 - (-3.0 - 2.0 - 0.5)                      # wallet less later USDT income
    assert nv["End-of-month futures balance (USDT)"] == 1500.0 and tot["End-of-month futures balance (USDT)"] == 1500.0
    assert tot["Trades"] == 2 and tot["Net P&L (USDT)"] == 4.0


def test_transfer_rows_types():
    inc = [dict(incomeType="TRANSFER", tranId=1, asset="USDT", income=-50.0, time=MS("2026-10-20 10:00"))]
    rows = AE.transfer_rows(inc, [(MS("2026-10-19 10:00"), 100.0, "USDT"), (MS("2026-10-21 10:00"), -20.0, "BNB")], 0, MS("2026-12-01 00:00"))
    assert [r["Type"] for r in rows] == ["Deposit", "Futures → Spot", "Withdrawal"] and rows[2]["Amount"] == 20.0


def test_spot_fifo_lots_fees_and_unknown_cost():
    usd = lambda a, ms: 1.0 if a in AE.STABLES else (500.0 if a == "BNB" else None)   # noqa: E731
    f = [dict(symbol="SOLUSDT", id=1, time=MS("2026-10-01 10:00"), price=100.0, qty=2.0, quoteQty=200.0, commission=0.002,
              commissionAsset="BNB", isBuyer=True),                                               # cost 200 + 1 = 201 → 100.5 / SOL
         dict(symbol="SOLUSDT", id=2, time=MS("2026-10-20 10:00"), price=110.0, qty=1.0, quoteQty=110.0, commission=0.01,
              commissionAsset="SOL", isBuyer=True),                                               # fee in SOL → 0.99 SOL at 110 / 0.99
         dict(symbol="SOLUSDT", id=3, time=MS("2026-10-25 10:00"), price=120.0, qty=2.5, quoteQty=300.0, commission=0.3,
              commissionAsset="USDT", isBuyer=False),                                             # proceeds 299.7 / 2.5
         dict(symbol="ETHUSDT", id=4, time=MS("2026-10-26 10:00"), price=2000.0, qty=0.1, quoteQty=200.0, commission=0.0,
              commissionAsset="USDT", isBuyer=False)]                                             # no earlier ETH buy → cost unknown
    tr, hold, lots = AE.spot_fifo(f, {"SOLUSDT": "SOL", "ETHUSDT": "ETH"}, usd, {"SOL": 0.49, "USDT": 50.0}, {"SOL": 130.0},
                                  MS("2026-10-15 00:00"), MS("2026-12-01 00:00"))
    assert [t["Side"] for t in tr] == ["Buy", "Sell", "Sell"]                                     # the 10-01 buy is before the period
    sol = [l for l in lots if l["Asset"] == "SOL"]
    assert len(sol) == 2 and sol[0]["Buy date (UTC-3)"] == "2026-10-01 07:00:00" and sol[0]["Quantity"] == 2.0
    assert sol[0]["Cost (USDT)"] == 201.0 and abs(sol[0]["Realized P&L (USDT)"] - (2.0 * 299.7 / 2.5 - 201.0)) < 1e-3
    assert abs(sol[1]["Quantity"] - 0.5) < 1e-9 and abs(sol[1]["Buy price (USDT)"] - 110 / 0.99) < 1e-4
    eth = [l for l in lots if l["Asset"] == "ETH"][0]
    assert eth["Cost (USDT)"] is None and eth["Realized P&L (USDT)"] is None and eth["Buy date (UTC-3)"].startswith("unknown")
    h = {x["Asset"]: x for x in hold}
    assert abs(h["SOL"]["Cost (USDT)"] - 0.49 * 110 / 0.99) < 1e-3 and h["SOL"]["Current value (USDT)"] == round(0.49 * 130.0, 4)
    assert h["USDT"]["Cost (USDT)"] is None and h["USDT"]["Current value (USDT)"] == 50.0


def _sheets(xlsx):
    z = zipfile.ZipFile(io.BytesIO(xlsx))
    names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", NS)]
    data = []
    for i in range(len(names)):
        root = ET.fromstring(z.read(f"xl/worksheets/sheet{i + 1}.xml"))
        rows = []
        for r in root.iter(f"{{{NS['m']}}}row"):
            rows.append([(c.findtext("m:v", namespaces=NS) or "".join(t.text or "" for t in c.iter(f"{{{NS['m']}}}t"))) for c in r])
        data.append(rows)
    return names, data


def test_xlsx_bytes_valid_and_escaped():
    x = AE.xlsx_bytes([("A & B", ["Col <1>", "N"], [{"Col <1>": "x & y", "N": 1.5}, {"Col <1>": "TOTAL", "N": None}], ["note"])])
    names, data = _sheets(x)
    assert names == ["A & B"] and data[0][0] == ["note"] and data[0][1] == ["Col <1>", "N"] and data[0][2] == ["x & y", "1.5"]


class _FakeBS:
    """the spot / capital / balance side of binance_service."""
    def __init__(self):
        self.spot_exchange = SimpleNamespace(
            markets={"SOL/USDT": dict(spot=True, base="SOL", quote="USDT", id="SOLUSDT"),
                     "BNB/USDT": dict(spot=True, base="BNB", quote="USDT", id="BNBUSDT")},
            fetch_balance=self._bal, privateGetMyTrades=self._trades, fetch_tickers=self._tickers, fetch_ohlcv=self._ohlcv,
            sapiGetCapitalDepositHisrec=self._deps, sapiGetCapitalWithdrawHistory=self._wds)
        self.calls = []

    async def _deps(self, q):
        self.calls.append(("dep", q["startTime"], q["endTime"]))
        return [dict(status=1, insertTime=MS("2026-10-15 08:00"), amount="1000", coin="USDT")] \
            if q["startTime"] <= MS("2026-10-15 08:00") <= q["endTime"] else []

    async def _wds(self, q):
        return []

    async def _load_spot_markets(self):
        return None

    async def get_balance(self):
        return {"ok": True, "usdt_total": 1500.0}

    async def _bal(self):
        return {"info": {"balances": [{"asset": "SOL", "free": "1", "locked": "0"}, {"asset": "USDT", "free": "10", "locked": "0"}]}}

    async def _trades(self, q):
        return [dict(id=1, time=MS("2026-10-16 10:00"), price="100", qty="1", quoteQty="100", commission="0.1",
                     commissionAsset="USDT", isBuyer=True)] if q["symbol"] == "SOLUSDT" else []

    async def _tickers(self, syms):
        return {"SOL/USDT": {"last": 120.0}, "BNB/USDT": {"last": 600.0}}

    async def _ohlcv(self, sym, tf, since=None, limit=None):
        return [[MS("2026-10-20 11:00"), 0, 0, 0, 500.0, 0]] if since <= MS("2026-10-20 11:00") else []


def _fill(i, t, side, price, qty, rpnl=0.0, comm=0.0, asset="USDT"):
    return dict(id=i, orderId=1000 + i, time=MS(t), side=side, price=str(price), qty=str(qty), realizedPnl=str(rpnl),
                commission=str(comm), commissionAsset=asset, maker=False)


class _FakeRec:
    """services.app_trades.AppTradeRecorder's read surface (budget / ban handled there): LINK long 10 @ 100 → 101 on 10-20 (fees 0.04 USDT
    + 0.002 BNB at 500, funding −0.5) · ARB short opened 10-03 (before the fetch window) closed 10-16 · an open BTC position."""
    INCOME = [dict(incomeType="REALIZED_PNL", tranId=1, symbol="LINKUSDT", asset="USDT", income=10.0, time=MS("2026-10-20 11:00")),
              dict(incomeType="COMMISSION", tranId=2, symbol="LINKUSDT", asset="USDT", income=-0.04, time=MS("2026-10-20 10:00")),
              dict(incomeType="COMMISSION", tranId=3, symbol="LINKUSDT", asset="BNB", income=-0.002, time=MS("2026-10-20 11:00")),
              dict(incomeType="FUNDING_FEE", tranId=4, symbol="LINKUSDT", asset="USDT", income=-0.5, time=MS("2026-10-20 10:30")),
              dict(incomeType="REALIZED_PNL", tranId=5, symbol="ARBUSDT", asset="USDT", income=4.0, time=MS("2026-10-16 12:00")),
              dict(incomeType="COMMISSION", tranId=6, symbol="ARBUSDT", asset="USDT", income=-0.2, time=MS("2026-10-16 12:00")),
              dict(incomeType="COMMISSION", tranId=7, symbol="BTCUSDT", asset="USDT", income=-0.1, time=MS("2026-10-25 09:00")),
              dict(incomeType="TRANSFER", tranId=8, asset="USDT", income=1000.0, time=MS("2026-10-15 09:00"))]
    FILLS = {"LINKUSDT": [_fill(1, "2026-10-20 10:00", "BUY", 100, 10, comm=0.04), _fill(2, "2026-10-20 11:00", "SELL", 101, 10, 10.0, 0.002, "BNB")],
             "ARBUSDT": [_fill(3, "2026-09-28 09:00", "SELL", 1.0, 100, comm=0.0), _fill(4, "2026-10-16 12:00", "BUY", 0.96, 100, 4.0, 0.2)],
             "BTCUSDT": [_fill(5, "2026-10-25 09:00", "BUY", 60000, 0.01, comm=0.1)]}

    def __init__(self):
        self.svc = SimpleNamespace(get_net_positions=self._pos)
        self.fill_calls = []

    def now_ms(self):
        return MS("2026-11-02 12:00")

    EARLY_FUNDING = [dict(incomeType="FUNDING_FEE", tranId=9, symbol="ARBUSDT", asset="USDT", income=0.3, time=MS("2026-09-29 08:00"))]

    def ban_active(self):
        return False

    async def _pos(self):
        return {"BTCUSDT": 0.01}

    async def _call(self, w, fn, *a):
        return await fn(*a)

    async def _income(self, lo, hi, symbol=None, income_type=None):
        rows = [r for r in self.INCOME + self.EARLY_FUNDING if lo <= r["time"] <= hi]
        return [r for r in rows if (symbol is None or r.get("symbol") == symbol) and (income_type is None or r["incomeType"] == income_type)]

    async def _fills(self, sym, lo, hi):
        from services.app_trades import parse_fill
        self.fill_calls.append((sym, lo, hi))
        return [parse_fill(r) for r in self.FILLS.get(sym, []) if lo <= r["time"] <= hi]

    async def prepare_rates(self, times):
        return None

    def rate(self, asset, t):
        return {"USDT": (1.0, False), "BNB": (500.0, False)}.get(asset)

    async def _leverage(self, sym):
        return 10


def test_futures_from_fills_binance_numbers_and_backward_read():
    rec = _FakeRec()
    lo, hi = AE.month_bounds("2026-10")
    rows = asyncio.run(AE.futures_from_fills(rec, rec.INCOME, lo, rec.now_ms(), lo, hi,
                                             lambda sym, t: 5.0 if sym == "LINKUSDT" else None, rec.now_ms() - AE.INCOME_HORIZON_MS))
    r = {x["Symbol"]: x for x in rows}
    assert set(r) == {"LINKUSDT", "ARBUSDT"}                                   # the open BTC position is not a closed trade
    ln = r["LINKUSDT"]
    assert ln["Gross P&L (USDT)"] == 10.0 and ln["Trading fees (USDT)"] == -1.04 and ln["Funding (USDT)"] == -0.5
    assert ln["Net P&L (USDT)"] == 8.46 and ln["Leverage"] == 5.0 and ln["Margin (USDT)"] == 200.0 and ln["Position size (USDT)"] == 1000.0
    assert ln["Close date (UTC-3)"] == "2026-10-20 08:00:00" and ln["Side"] == "Long"     # Argentina time
    ar = r["ARBUSDT"]
    assert ar["Side"] == "Short" and ar["Open date (UTC-3)"] == "2026-09-28 06:00:00" and ar["Leverage"] == 10.0   # read back past the window
    assert any(c[0] == "ARBUSDT" and c[1] <= MS("2026-09-28 09:00") <= c[2] for c in rec.fill_calls)   # a week with no fee row, read back
    assert ar["Funding (USDT)"] == 0.3 and ar["Net P&L (USDT)"] == 4.1        # its funding before the read window is counted too


def test_futures_unreadable_entry_raises():
    rec = _FakeRec()
    lo, hi = AE.month_bounds("2026-10")
    with pytest.raises(RuntimeError):                                             # the entry is older than Binance's history
        asyncio.run(AE.futures_from_fills(rec, rec.INCOME, lo, rec.now_ms(), lo, hi, None, MS("2026-10-01 05:00")))


def test_build_live_end_to_end_one_file():
    lo, hi = AE.month_bounds("2026-10")
    xlsx, st = asyncio.run(AE.build_live(lo, hi, None, rec=_FakeRec(), bs=_FakeBS()))
    names, data = _sheets(xlsx)
    assert names == ["Futures trades", "Monthly summary", "Transfers", "Spot trades", "Spot holdings", "Spot sold lots"]
    assert st["trades"] == 2 and st["activity"] and abs(st["net"] - (8.46 + 4.1)) < 1e-9
    ms = data[1][[i for i, r in enumerate(data[1]) if r[:1] == ["Month"]][0] + 1]
    assert ms[0] == "2026-10" and abs(float(ms[MS_COL("Difference")]) - 0.4) < 1e-9   # the open BTC fee (0.1) + ARB's September funding (0.3)
    assert any(r[:1] == ["SOL"] and r[2] for r in data[4])                                     # holding with a known average price


def MS_COL(c):
    return AE.MONTH_COLS.index(c)


def test_build_live_raises_on_failed_wallet_and_old_period():
    bs = _FakeBS()

    async def bad():
        return {"ok": False}
    bs.get_balance = bad
    lo, hi = AE.month_bounds("2026-10")
    with pytest.raises(RuntimeError):
        asyncio.run(AE.build_live(lo, hi, None, rec=_FakeRec(), bs=bs))
    lo, hi = AE.month_bounds("2026-08")
    with pytest.raises(RuntimeError):                                              # starts beyond Binance's ~3 months: use the stored statement
        asyncio.run(AE.build_live(lo, hi, None, rec=_FakeRec(), bs=_FakeBS()))


def test_monthly_income_horizon_blanks_old_months():
    trades = AE.trade_rows([_o("2026-08-20 10:00", "2026-08-20 11:00", 10.0), _o("2026-10-20 10:00", "2026-10-20 11:00", 5.0)])
    usd = lambda a, ms: 1.0   # noqa: E731
    m = {r["Month"]: r for r in AE.monthly_rows(trades, [], [], usd, 100.0, True, income_from_ms=MS("2026-09-10 00:00"))}
    assert m["2026-08"]["Binance net P&L (check)"] is None and m["2026-08"]["End-of-month futures balance (USDT)"] is None
    assert m["2026-10"]["Binance net P&L (check)"] == 0.0 and m["2026-10"]["End-of-month futures balance (USDT)"] == 100.0
    assert m["TOTAL"]["Binance net P&L (check)"] is None and m["2026-08"]["Net P&L (USDT)"] == 9.5     # own numbers stay


def test_capital_flows_windows():
    bs = _FakeBS()
    rows = asyncio.run(AE.read_capital_flows(bs, MS("2026-06-01 00:00"), MS("2026-11-01 00:00")))
    assert rows == [(MS("2026-10-15 08:00"), 1000.0, "USDT")]
    assert len([c for c in bs.calls if c[0] == "dep"]) == 2 and all(b - a < 90 * 86_400_000 for _k, a, b in bs.calls)


def test_argentina_months_and_dates():
    a, b = AE.month_bounds("2026-11")
    assert a == MS("2026-11-01 03:00") and b == MS("2026-12-01 03:00") - 1      # 00:00 ART = 03:00 UTC
    assert AE.fmt_dt(MS("2026-11-01 02:30")) == "2026-10-31 23:30:00" and AE.month_of_ms(MS("2026-11-01 02:30")) == "2026-10"
    assert AE.fmt_dt(datetime(2026, 11, 1, 2, 30)) == "2026-10-31 23:30:00"     # naive DB datetimes are UTC
    assert AE.month_bounds("2026-12")[1] == MS("2027-01-01 03:00") - 1


def test_due_months_day2_art():
    assert AE.due_months(datetime(2026, 12, 2, 2, 59)) == ["2026-10"]          # Dec 1 23:59 ART: November not due yet
    assert AE.due_months(datetime(2026, 12, 2, 3, 0)) == ["2026-11", "2026-10"]  # Dec 2 00:00 ART
    assert AE.due_months(datetime(2027, 1, 5, 12, 0)) == ["2026-12", "2026-11"]


def test_statement_pass_stores_and_retries():
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import select
    import models

    calls = []

    async def fake_build(lo, hi, look):
        calls.append(AE.fmt_dt(lo)[:7])
        if len(calls) == 1:
            raise RuntimeError("Binance busy")
        return b"PK-xlsx", dict(trades=3 if AE.fmt_dt(lo)[:7] == "2026-11" else 0, net=12.5, activity=AE.fmt_dt(lo)[:7] == "2026-11")

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        sf = async_sessionmaker(eng, expire_on_commit=False)

        async def commit(db):
            await db.commit()
        now = datetime(2026, 12, 3, 12, 0)
        d1 = await AE.statement_pass(sf, commit, now, fake_build, keys=lambda: True)
        d2 = await AE.statement_pass(sf, commit, now, fake_build, keys=lambda: True)
        d3 = await AE.statement_pass(sf, commit, now, fake_build, keys=lambda: True)
        async with sf() as db:
            rows = {r.month: r for r in (await db.execute(select(models.AccountingStatement))).scalars().all()}
        await eng.dispose()
        return d1, d2, d3, rows
    d1, d2, d3, rows = asyncio.run(run())
    assert d1 == ["2026-10"] and d2 == ["2026-11"] and d3 == []                    # Nov failed once → retried; nothing rebuilt after
    assert rows["2026-11"].status == "READY" and rows["2026-11"].attempts == 2 and rows["2026-11"].xlsx == b"PK-xlsx" and rows["2026-11"].trades == 3
    assert rows["2026-10"].status == "EMPTY" and rows["2026-10"].error is None
    assert AE.statement_name("2026-11") == "scalpars_accounting_2026-11.xlsx"


def test_statement_pass_no_keys_does_nothing():
    assert asyncio.run(AE.statement_pass(object(), None, datetime(2026, 12, 3), None, keys=lambda: False)) == []


def test_archive_sync_routes_statements(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import export_glob as EG
    dl, ar = tmp_path / "dl", tmp_path / "ar"
    dl.mkdir()
    (dl / "scalpars_accounting_2027-01.xlsx").write_bytes(b"x")
    (dl / "scalpars_accounting_2027-01 (1).xlsx").write_bytes(b"y")                   # a browser duplicate is not a statement name
    assert EG.sync_archive(str(dl), str(ar), force=True) == 1
    assert (ar / "accounting" / "scalpars_accounting_2027-01.xlsx").read_bytes() == b"x"


def test_xlsx_strips_control_chars_and_nan():
    names, data = _sheets(AE.xlsx_bytes([("S", ["A", "B"], [{"A": "x\x01y", "B": float("nan")}], [])]))
    assert data[0][1] == ["xy", ""]


def test_build_paper_marked_and_binance_free():
    names, data = _sheets(AE.build_paper([_o("2026-10-20 10:00", "2026-10-20 11:00", 10.0)], MS("2026-10-14 00:00"), MS("2026-10-31 23:59")))
    assert names == ["Futures trades (PAPER)", "Monthly summary (PAPER)"]
    assert data[0][0][0].startswith("PAPER TRADING") and data[1][0][0].startswith("PAPER TRADING")
    hdr = data[1].index(AE.PAPER_MONTH_COLS)
    assert data[1][hdr + 1][0] == "2026-10" and float(data[1][hdr + 1][AE.PAPER_MONTH_COLS.index("Net P&L (USDT)")]) == 9.5
    assert not any("Binance" in c or "balance" in c.lower() for c in AE.PAPER_MONTH_COLS)


def test_dashboard_controls():
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert 'id="acct-mode"' not in html and "Paper (check)" not in html and "q.push('mode=" not in html   # the bot's mode decides (DL 458)
    assert "onclick=\"_acctDownload(this)\"" in html and "isPaperMode ? 'Accounting (paper)' : 'Accounting'" in html
    assert html.count("try { _acctLabel(); } catch (e) {}") == 3                            # poll + mode switch + after an export
    assert "fetch(_acctUrl('/api/accounting/prepare'), {method: 'POST'})" in html          # prepare → poll → save
    assert 'id="acct-statements"' in html and 'id="auto-acct-toggle"' in html
    assert "if (last === null) {" in html and "for (const x of ready.filter(x => x.month > last))" in html   # first visit remembers; each new month once


def test_prepare_follows_bot_mode():
    import inspect
    import main
    src = inspect.getsource(main.accounting_prepare)
    assert "mode: Optional[str] = None" in src and "bool(trading_engine.is_paper_mode)" in src and 'mode must be live or paper' in src


def test_statements_box_hidden_until_a_month_exists():
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert '<span id="acct-statements-group" class="flex items-center gap-2 whitespace-nowrap" style="display:none">' in html
    assert "grp.style.display = list.length ? '' : 'none';" in html and "Monthly statements…" not in html


def test_number_formats_per_column():
    assert [AE.col_kind(c) for c in ("Margin (USDT)", "Entry price", "Quantity", "Leverage", "Trades", "Difference", "Symbol",
                                      "Average buy price (USDT)", "Amount", "Win rate %", "Return %", "Avg leverage")] == \
        ["money", "price", "qty", None, "int", "money", None, "price", "price", "pct", "pct", None]
    x = AE.xlsx_bytes([("S", ["Month", "Net P&L (USDT)", "Entry price", "Trades"], [{"Month": "2026-10", "Net P&L (USDT)": 1767.19,
                                                                                       "Entry price": 0.004863, "Trades": 3},
                                                                                      {"Month": "TOTAL", "Net P&L (USDT)": 1767.19, "Trades": 3}], [])])
    z = zipfile.ZipFile(io.BytesIO(x))
    st, sh = z.read("xl/styles.xml").decode(), z.read("xl/worksheets/sheet1.xml").decode()
    assert 'formatCode="#,##0.00"' in st and 'formatCode="#,##0.00######"' in st and 'formatCode="#,##0.0000"' in st
    assert 'cellXfs count="13"' in st and 'numFmts count="3"' in st and st.count("<xf ") == 13       # every cellXfs entry declared (count matches)
    assert '<c r="B2" s="3"><v>1767.19</v></c>' in sh and '<c r="C2" s="5"><v>0.004863</v></c>' in sh and '<c r="D2" s="7"><v>3</v></c>' in sh
    assert '<c r="B3" s="4"><v>1767.19</v></c>' in sh and '<c r="D3" s="8"><v>3</v></c>' in sh      # TOTAL row: bold, same formats
    assert '<c r="A2" t="inlineStr"><is><t>2026-10</t></is></c>' in sh                              # text keeps the default style


def test_live_rows_carry_binance_order_ids():
    rec = _FakeRec()
    lo, hi = AE.month_bounds("2026-10")
    rows = asyncio.run(AE.futures_from_fills(rec, rec.INCOME, lo, rec.now_ms(), lo, hi, None, rec.now_ms() - AE.INCOME_HORIZON_MS))
    r = {x["Symbol"]: x for x in rows}
    assert (r["LINKUSDT"]["Open order ID"], r["LINKUSDT"]["Close order ID"]) == ("1001", "1002")
    assert (r["ARBUSDT"]["Open order ID"], r["ARBUSDT"]["Close order ID"]) == ("1003", "1004")
    assert AE.TRADE_COLS_LIVE[-2:] == ["Open order ID", "Close order ID"] and "Open order ID" not in AE.TRADE_COLS   # paper has no real orders
    assert AE.col_kind("Open order ID") is None                                                    # an ID is text, never 1,001.00


def test_column_width_fits_the_formatted_number():
    assert AE._shown(12345678.0, "price") == "12,345,678.00" and AE._shown(0.004863, "price") == "0.004863"
    assert AE._shown(22902.89657854, "qty") == "22,902.8966" and AE._shown(0.1234, "pct") == "12.34%"
    assert AE._shown(1767.19, "money") == "1,767.19" and AE._shown(3, "int") == "3" and AE._shown("x", None) == "x"
    sh = zipfile.ZipFile(io.BytesIO(AE.xlsx_bytes([("S", ["Entry price"], [{"Entry price": 12345678.0}], [])]))).read("xl/worksheets/sheet1.xml").decode()
    assert 'width="15"' in sh                                                                     # 13 shown characters + 2


def test_monthly_new_columns():
    t = [dict(_month="2026-10", _gross=12.0, _fees=-2.0, _funding=0.0, _net=10.0, _size=1000.0, _margin=100.0),
         dict(_month="2026-10", _gross=-3.0, _fees=-1.0, _funding=-1.0, _net=-5.0, _size=3000.0, _margin=600.0),
         dict(_month="2026-10", _gross=31.0, _fees=-1.0, _funding=0.0, _net=30.0, _size=1000.0, _margin=100.0)]
    inc = [dict(incomeType="REALIZED_PNL", tranId=1, asset="USDT", income=40.0, time=MS("2026-10-20 11:00")),
           dict(incomeType="COMMISSION", tranId=2, asset="USDT", income=-4.0, time=MS("2026-10-20 11:00")),
           dict(incomeType="FUNDING_FEE", tranId=3, asset="USDT", income=-1.0, time=MS("2026-10-20 11:00"))]
    m = {r["Month"]: r for r in AE.monthly_rows(t, inc, [], lambda a, ms: 1.0, 1035.0, True)}["2026-10"]
    assert abs(m["Win rate %"] - 2 / 3) < 1e-9 and m["Volume traded (USDT)"] == 5000.0 and m["Avg margin per trade (USDT)"] == 266.67
    assert m["Avg leverage"] == 6.2 and abs(m["Fees % of volume"] - 4 / 5000) < 1e-12                 # 5000 / 800 size-weighted
    assert (m["Avg win (USDT)"], m["Avg loss (USDT)"], m["Largest win (USDT)"], m["Largest loss (USDT)"]) == (20.0, -5.0, 30.0, -5.0)
    assert m["Start-of-month futures balance (USDT)"] == 1000.0 and m["End-of-month futures balance (USDT)"] == 1035.0
    assert abs(m["Return %"] - 35 / 1000) < 1e-12
    p = {r["Month"]: r for r in AE.monthly_rows(t, [], [], lambda a, ms: 1.0, None, False)}["2026-10"]
    assert p["Return %"] is None and p["Start-of-month futures balance (USDT)"] is None                # paper: no account balance
    assert "Return %" not in AE.PAPER_MONTH_COLS and "Win rate %" in AE.PAPER_MONTH_COLS


def test_spot_quantity_keeps_small_decimals():
    x = AE.xlsx_bytes([("Spot holdings", AE.HOLDING_COLS, [{"Asset": "BTC", "Quantity": 0.00012345}], [], AE.SPOT_QTY),
                       ("Futures trades", ["Quantity"], [{"Quantity": 22902.89657854}], [])])
    z = zipfile.ZipFile(io.BytesIO(x))
    assert '<c r="B2" s="5">' in z.read("xl/worksheets/sheet1.xml").decode()                     # spot: price format (≤ 8 decimals)
    assert '<c r="A2" s="11">' in z.read("xl/worksheets/sheet2.xml").decode()                     # futures: 4 decimals
