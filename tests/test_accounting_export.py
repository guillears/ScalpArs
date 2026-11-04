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
    assert len(sol) == 2 and sol[0]["Buy date (UTC)"] == "2026-10-01 10:00:00" and sol[0]["Quantity"] == 2.0
    assert sol[0]["Cost (USDT)"] == 201.0 and abs(sol[0]["Realized P&L (USDT)"] - (2.0 * 299.7 / 2.5 - 201.0)) < 1e-3
    assert abs(sol[1]["Quantity"] - 0.5) < 1e-9 and abs(sol[1]["Buy price (USDT)"] - 110 / 0.99) < 1e-4
    eth = [l for l in lots if l["Asset"] == "ETH"][0]
    assert eth["Cost (USDT)"] is None and eth["Realized P&L (USDT)"] is None and eth["Buy date (UTC)"].startswith("unknown")
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
    def __init__(self):
        self.spot_exchange = SimpleNamespace(
            markets={"SOL/USDT": dict(spot=True, base="SOL", quote="USDT", id="SOLUSDT"),
                     "BNB/USDT": dict(spot=True, base="BNB", quote="USDT", id="BNBUSDT")},
            fetch_balance=self._bal, privateGetMyTrades=self._trades, fetch_tickers=self._tickers, fetch_ohlcv=self._ohlcv,
            sapiGetCapitalDepositHisrec=self._deps, sapiGetCapitalWithdrawHistory=self._wds)
        self.calls = []

    def used_weight_1m(self):
        return 100

    async def _deps(self, q):
        self.calls.append(("dep", q["startTime"], q["endTime"]))
        return [dict(status=1, insertTime=MS("2026-10-15 08:00"), amount="1000", coin="USDT")] \
            if q["startTime"] <= MS("2026-10-15 08:00") <= q["endTime"] else []

    async def _wds(self, q):
        return []

    async def _load_spot_markets(self):
        return None

    async def fetch_income_page(self, lo, hi, **_k):
        return [dict(incomeType="REALIZED_PNL", tranId=1, asset="USDT", income=12.0, time=MS("2026-10-20 11:00"))] \
            if lo <= MS("2026-10-20 11:00") <= hi else []

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
        return [[MS("2026-10-20 11:00"), 0, 0, 0, 600.0, 0]] if since <= MS("2026-10-20 11:00") else []


def test_build_end_to_end_one_file():
    xlsx = asyncio.run(AE.build([_o("2026-10-20 10:00", "2026-10-20 11:00", 10.0, fund=0.0)], _FakeBS(),
                                MS("2026-10-14 00:00"), MS("2026-10-31 23:59")))
    names, data = _sheets(xlsx)
    assert names == ["Futures trades", "Monthly summary", "Transfers", "Spot trades", "Spot holdings", "Spot sold lots"]
    hdr = data[0].index(AE.TRADE_COLS)
    assert data[0][hdr + 1][2] == "LINKUSDT" and float(data[0][hdr + 1][13]) == 10.0
    ms = data[1][[i for i, r in enumerate(data[1]) if r[:1] == ["Month"]][0] + 1]
    assert ms[0] == "2026-10" and float(ms[MS_COL("Binance net P&L (check)")]) == 12.0 and float(ms[MS_COL("Difference")]) == -2.0
    assert any(r[:1] == ["SOL"] and r[2] for r in data[4])                                     # holding with a known average price


def MS_COL(c):
    return AE.MONTH_COLS.index(c)


def test_build_raises_on_failed_wallet():
    bs = _FakeBS()

    async def bad():
        return {"ok": False}
    bs.get_balance = bad
    with pytest.raises(RuntimeError):
        asyncio.run(AE.build([], bs, MS("2026-10-14 00:00"), MS("2026-10-31 23:59")))


def test_monthly_income_horizon_blanks_old_months():
    trades = AE.trade_rows([_o("2026-08-20 10:00", "2026-08-20 11:00", 10.0), _o("2026-10-20 10:00", "2026-10-20 11:00", 5.0)])
    usd = lambda a, ms: 1.0   # noqa: E731
    m = {r["Month"]: r for r in AE.monthly_rows(trades, [], [], usd, 100.0, True, income_from_ms=MS("2026-09-10 00:00"))}
    assert m["2026-08"]["Binance net P&L (check)"] is None and m["2026-08"]["End-of-month futures balance (USDT)"] is None
    assert m["2026-10"]["Binance net P&L (check)"] == 0.0 and m["2026-10"]["End-of-month futures balance (USDT)"] == 100.0
    assert m["TOTAL"]["Binance net P&L (check)"] is None and m["2026-08"]["Net P&L (USDT)"] == 9.5     # own numbers stay


def test_capital_flows_windows_and_guard():
    bs = _FakeBS()
    rows = asyncio.run(AE.read_capital_flows(bs, MS("2026-06-01 00:00"), MS("2026-11-01 00:00")))
    assert rows == [(MS("2026-10-15 08:00"), 1000.0, "USDT")]
    assert len([c for c in bs.calls if c[0] == "dep"]) == 2 and all(b - a < 90 * 86_400_000 for _k, a, b in bs.calls)
    bs.used_weight_1m = lambda: 1500
    with pytest.raises(RuntimeError):
        asyncio.run(AE.read_income(bs, MS("2026-10-14 00:00"), MS("2026-10-20 00:00")))       # the live bot's budget comes first


def test_xlsx_strips_control_chars_and_nan():
    names, data = _sheets(AE.xlsx_bytes([("S", ["A", "B"], [{"A": "x\x01y", "B": float("nan")}], [])]))
    assert data[0][1] == ["xy", ""]


def test_build_paper_marked_and_binance_free():
    names, data = _sheets(AE.build_paper([_o("2026-10-20 10:00", "2026-10-20 11:00", 10.0)], MS("2026-10-14 00:00"), MS("2026-10-31 23:59")))
    assert names == ["Futures trades (PAPER)", "Monthly summary (PAPER)"]
    assert data[0][0][0].startswith("PAPER TRADING") and data[1][0][0].startswith("PAPER TRADING")
    hdr = data[1].index(AE.PAPER_MONTH_COLS)
    assert data[1][hdr + 1][0] == "2026-10" and float(data[1][hdr + 1][7]) == 9.5
    assert not any("Binance" in c or "balance" in c.lower() for c in AE.PAPER_MONTH_COLS)


def test_dashboard_mode_select_and_error_message():
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert 'id="acct-mode"' in html and '<option value="paper">Paper (check)</option>' in html
    assert "onclick=\"_acctDownload(this)\"" in html and "q.push('mode=' + encodeURIComponent(m));" in html
    assert "if (!r.ok)" in html                                            # an error is a message, never a downloaded .json
