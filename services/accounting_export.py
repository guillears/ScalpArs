"""📒 Accountant export (DECISION_LOG 455, operator 2026-11-04: "an excel / csv to download for my accountant … simple, but contain all
the necessary information. Similar to what i would download from binance").

SCOPE      LIVE money (mode=paper → build_paper: the paper trades + monthly totals, marked NOT for accounting — operator 11-04), USDT, UTC, from the first live trade (or the period asked). Bot trades, dashboard-opened
           trades and Binance-app trades (auto-recorded, DECISION_LOG 329) are ONE list — nothing in the file says which is which
           (operator: "dont want to disclose bot or manual").
SHEETS     ① Futures trades — one row per closed position from the live Order rows (the dashboard's own numbers): close / open date,
             symbol, side, quantity, entry / exit price, margin, leverage, position size, gross P&L, trading fees, funding, net P&L, result.
             Order convention: pnl is NET of the entry + exit fees and EXCLUDES funding (funding_fee_usd, negative = paid) → gross = pnl
             + fees · net = pnl + funding · margin = position size / leverage (so margin × leverage = position size on every row).
           ② Monthly summary — per UTC month: trades, wins / losses, gross, fees, funding, net (from ①) + the CHECK against Binance's own
             income history (REALIZED_PNL + COMMISSION + FUNDING_FEE, BNB commission × the BNB hourly close) and the difference ·
             deposits / withdrawals (capital history, stablecoins) · net transfers into futures · end-of-month futures USDT wallet
             balance (= today's wallet − every later USDT income row). A total row.
           ③ Transfers — deposits, withdrawals (every coin) and spot ↔ futures moves (TRANSFER income), date · type · asset · amount.
           ④ Spot trades · ⑤ Spot holdings · ⑥ Spot sold lots — spot fills (GET /api/v3/myTrades, per symbol) for the coins this account
             is known to hold or have moved (balances + deposit / withdrawal history + BNB) against USDT / USDC / FDUSD; FIFO lots (fees in
             the cost / proceeds; a sale with no earlier buy in the history = cost unknown, never invented). Coins bought AND sold to
             zero that never sat in the wallet are not discovered (Binance has no all-symbols trade list) — stated on the sheet.
FORMAT     ONE .xlsx (operator: one download), written directly (stdlib zip + XML, no extra package).
Every read is Binance READ-ONLY (income, capital history, balances, myTrades, klines, tickers); nothing here places or changes anything.
"""
import io
import logging
import math
import re
import zipfile
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

STABLES = ("USDT", "USDC", "FDUSD")
WINDOW_MS = 7 * 86_400_000                      # Binance income: ≤ 7-day windows
DAY_MS = 86_400_000
INCOME_HORIZON_MS = 89 * DAY_MS                 # Binance keeps ~3 months of futures income: older months get no check / balance (noted)
CAPITAL_WINDOW_MS = 89 * DAY_MS                 # deposit / withdrawal history: ≤ 90-day windows per call
WEIGHT_STOP = 1200                              # X-MBX-USED-WEIGHT-1M above this → stop (the live bot shares the IP's 2,400 / min)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
TRADE_COLS = ["Close date (UTC)", "Open date (UTC)", "Symbol", "Side", "Quantity", "Entry price", "Exit price", "Margin (USDT)",
              "Leverage", "Position size (USDT)", "Gross P&L (USDT)", "Trading fees (USDT)", "Funding (USDT)", "Net P&L (USDT)", "Result"]
MONTH_COLS = ["Month", "Trades", "Wins", "Losses", "Gross P&L (USDT)", "Trading fees (USDT)", "Funding (USDT)", "Net P&L (USDT)",
              "Binance net P&L (check)", "Difference", "Deposits (USDT)", "Withdrawals (USDT)", "Net transfers into futures (USDT)",
              "End-of-month futures balance (USDT)"]
TRANSFER_COLS = ["Date (UTC)", "Type", "Asset", "Amount"]
SPOT_TRADE_COLS = ["Date (UTC)", "Symbol", "Side", "Quantity", "Price", "Total (USDT)", "Fee (USDT)"]
HOLDING_COLS = ["Asset", "Quantity", "Average buy price (USDT)", "Cost (USDT)", "Current price (USDT)", "Current value (USDT)",
                "Unrealized P&L (USDT)"]
LOT_COLS = ["Asset", "Buy date (UTC)", "Sell date (UTC)", "Quantity", "Buy price (USDT)", "Sell price (USDT)", "Cost (USDT)",
            "Proceeds (USDT)", "Realized P&L (USDT)"]


def fmt_dt(d) -> str:
    if d is None:
        return ""
    if isinstance(d, (int, float)):
        d = datetime.fromtimestamp(d / 1000, tz=timezone.utc)
    if d.tzinfo is not None:
        d = d.astimezone(timezone.utc).replace(tzinfo=None)
    return d.strftime("%Y-%m-%d %H:%M:%S")


def _f(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _r(x, n=4):
    return None if x is None else round(float(x), n)


# ── ① futures trades (pure) ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def trade_row(o) -> dict:
    fees = _f(getattr(o, "entry_fee", 0)) + _f(getattr(o, "exit_fee", 0))
    pnl = _f(getattr(o, "pnl", 0))
    funding = _f(getattr(o, "funding_fee_usd", 0))
    size = _f(getattr(o, "notional_value", 0)) or _f(getattr(o, "entry_price", 0)) * _f(getattr(o, "quantity", 0))
    lev = _f(getattr(o, "leverage", 0)) or 1.0
    net = pnl + funding
    return {
        "Close date (UTC)": fmt_dt(o.closed_at), "Open date (UTC)": fmt_dt(o.opened_at), "Symbol": str(o.pair).replace("/", ""),
        "Side": "Long" if str(o.direction).upper() == "LONG" else "Short", "Quantity": _r(_f(o.quantity), 8),
        "Entry price": _r(_f(o.entry_price), 8), "Exit price": _r(_f(o.exit_price), 8), "Margin (USDT)": _r(size / lev, 2),
        "Leverage": _r(lev, 2), "Position size (USDT)": _r(size, 2), "Gross P&L (USDT)": _r(pnl + fees, 4),
        "Trading fees (USDT)": _r(-fees, 4), "Funding (USDT)": _r(funding, 4), "Net P&L (USDT)": _r(net, 4),
        "Result": "Win" if net > 0 else "Loss",
        "_month": fmt_dt(o.closed_at)[:7], "_gross": pnl + fees, "_fees": -fees, "_funding": funding, "_net": net}


def trade_rows(orders: Iterable) -> List[dict]:
    """CLOSED live rows → sheet ① rows, oldest close first (caller filters is_paper False ∧ CLOSED ∧ the period)."""
    rows = [trade_row(o) for o in orders if getattr(o, "closed_at", None) is not None]
    return sorted(rows, key=lambda r: r["Close date (UTC)"])


# ── Binance income → monthly check (pure) ───────────────────────────────────────────────────────────────────────────────────────────
def income_key(r: dict) -> tuple:
    return (str(r.get("incomeType")), str(r.get("tranId")), str(r.get("symbol") or ""), str(r.get("asset") or ""))


def month_of_ms(ms: int) -> str:
    return fmt_dt(int(ms))[:7]


def income_month_net(income: Iterable[dict], usd: Callable[[str, int], Optional[float]]) -> Dict[str, float]:
    """{month: Σ REALIZED_PNL + COMMISSION + FUNDING_FEE in USDT} — usd(asset, ms) → the asset's USDT price (stables 1)."""
    out: Dict[str, float] = {}
    for r in income:
        if str(r.get("incomeType")) not in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE"):
            continue
        px = usd(str(r.get("asset") or "USDT"), int(r.get("time") or 0))
        if px is None:
            continue
        m = month_of_ms(int(r.get("time") or 0))
        out[m] = out.get(m, 0.0) + _f(r.get("income")) * px
    return out


def month_end_balances(months: List[str], income: Iterable[dict], wallet_now: Optional[float]) -> Dict[str, Optional[float]]:
    """futures USDT wallet at each month's end = today's wallet − Σ USDT income rows after that month (every income type moves it)."""
    if wallet_now is None:
        return {m: None for m in months}
    rows = [(month_of_ms(int(r.get("time") or 0)), _f(r.get("income"))) for r in income if str(r.get("asset") or "") == "USDT"]
    return {m: wallet_now - sum(v for mm, v in rows if mm > m) for m in months}


def month_bounds(m: str) -> Tuple[int, int]:
    """'YYYY-MM' → (first ms, last ms) UTC."""
    y, mo = int(m[:4]), int(m[5:7])
    a = int(datetime(y, mo, 1, tzinfo=timezone.utc).timestamp() * 1000)
    b = int(datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1
    return a, b


def monthly_rows(trades: List[dict], income: List[dict], flows: List[Tuple[int, float, str]], usd, wallet_now: Optional[float],
                 binance_ok: bool, income_from_ms: int = 0) -> List[dict]:
    """sheet ② — one row per UTC month with a trade, an income row or a flow, + TOTAL. income_from_ms = the oldest instant Binance's
    income history still holds: a month starting before it gets no check (blank), a month ending before it no balance (blank)."""
    tin = [r for r in income if str(r.get("incomeType")) == "TRANSFER" and str(r.get("asset") or "") in STABLES]
    months = sorted({t["_month"] for t in trades} | {month_of_ms(int(r.get("time") or 0)) for r in income}
                    | {month_of_ms(int(t)) for t, _, _ in flows})
    bn = income_month_net(income, usd) if binance_ok else {}
    bal = month_end_balances(months, income, wallet_now) if binance_ok else {m: None for m in months}
    checked = {m for m in months if month_bounds(m)[0] >= income_from_ms}
    bal = {m: (v if month_bounds(m)[1] >= income_from_ms else None) for m, v in bal.items()}
    out, tot = [], dict.fromkeys(["n", "w", "l", "g", "f", "fu", "net", "bn", "dep", "wd", "tr"], 0.0)
    for m in months:
        g = [t for t in trades if t["_month"] == m]
        dep = sum(a for t, a, c in flows if month_of_ms(int(t)) == m and a > 0 and str(c) in STABLES)
        wd = sum(-a for t, a, c in flows if month_of_ms(int(t)) == m and a < 0 and str(c) in STABLES)
        tr = sum(_f(r.get("income")) for r in tin if month_of_ms(int(r.get("time") or 0)) == m)
        s = dict(n=len(g), w=sum(t["_net"] > 0 for t in g), l=sum(t["_net"] <= 0 for t in g), g=sum(t["_gross"] for t in g),
                 f=sum(t["_fees"] for t in g), fu=sum(t["_funding"] for t in g), net=sum(t["_net"] for t in g), bn=bn.get(m, 0.0),
                 dep=dep, wd=wd, tr=tr)
        for k in tot:
            tot[k] += s[k]
        out.append(_month_row(m, s, binance_ok and m in checked, bal.get(m)))
    out.append(_month_row("TOTAL", tot, binance_ok and set(months) <= checked, bal.get(months[-1]) if months else None))
    return out


def _month_row(m, s, ok, bal) -> dict:
    return {"Month": m, "Trades": int(s["n"]), "Wins": int(s["w"]), "Losses": int(s["l"]), "Gross P&L (USDT)": _r(s["g"], 2),
            "Trading fees (USDT)": _r(s["f"], 2), "Funding (USDT)": _r(s["fu"], 2), "Net P&L (USDT)": _r(s["net"], 2),
            "Binance net P&L (check)": _r(s["bn"], 2) if ok else None, "Difference": _r(s["net"] - s["bn"], 2) if ok else None,
            "Deposits (USDT)": _r(s["dep"], 2), "Withdrawals (USDT)": _r(s["wd"], 2), "Net transfers into futures (USDT)": _r(s["tr"], 2),
            "End-of-month futures balance (USDT)": _r(bal, 2)}


def transfer_rows(income: List[dict], flows: List[Tuple[int, float, str]], lo_ms: int, hi_ms: int) -> List[dict]:
    """sheet ③ — capital deposits / withdrawals (every coin) + futures TRANSFER income (+ = into futures), oldest first."""
    rows = [(int(t), "Deposit" if a > 0 else "Withdrawal", str(c), abs(a)) for t, a, c in flows if lo_ms <= int(t) <= hi_ms]
    rows += [(int(r.get("time") or 0), "Spot → Futures" if _f(r.get("income")) > 0 else "Futures → Spot", str(r.get("asset") or ""),
              abs(_f(r.get("income")))) for r in income if str(r.get("incomeType")) == "TRANSFER"
             and lo_ms <= int(r.get("time") or 0) <= hi_ms]
    return [{"Date (UTC)": fmt_dt(t), "Type": k, "Asset": a, "Amount": _r(v, 8)} for t, k, a, v in sorted(rows)]


# ── ④–⑥ spot, FIFO (pure) ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def spot_fifo(fills: List[dict], base_of: Dict[str, str], usd: Callable[[str, int], Optional[float]],
              balances: Dict[str, float], prices: Dict[str, float], lo_ms: int, hi_ms: int):
    """fills = raw myTrades rows (symbol, time, price, qty, quoteQty, commission, commissionAsset, isBuyer) of STABLE-quoted symbols;
    base_of = {symbol: base asset}. → (spot trade rows in the period, holdings rows, sold-lot rows with a sell date in the period).
    Cost basis includes the buy fee; proceeds are net of the sell fee; a fee paid in the coin itself changes the quantity instead.
    A sale beyond the known buys → a lot with cost unknown (blank, no P&L) — never invented."""
    trades_out, lots_out = [], []
    lots: Dict[str, List[list]] = {}              # asset → [[t, qty, unit_cost]]
    for r in sorted(fills, key=lambda x: (int(x.get("time") or 0), str(x.get("id")))):
        sym, t = str(r.get("symbol")), int(r.get("time") or 0)
        base = base_of.get(sym)
        if not base:
            continue
        qty, price, quote = _f(r.get("qty")), _f(r.get("price")), _f(r.get("quoteQty"))
        ca, cm = str(r.get("commissionAsset") or ""), _f(r.get("commission"))
        fee_in_base = ca == base
        px = usd(ca, t) if (ca and not fee_in_base) else None
        fee_usd = cm * price if fee_in_base else (cm * px if px is not None else 0.0)
        buy = bool(r.get("isBuyer"))
        if lo_ms <= t <= hi_ms:
            trades_out.append({"Date (UTC)": fmt_dt(t), "Symbol": sym, "Side": "Buy" if buy else "Sell", "Quantity": _r(qty, 8),
                               "Price": _r(price, 8), "Total (USDT)": _r(quote, 4), "Fee (USDT)": _r(-fee_usd, 4)})
        book = lots.setdefault(base, [])
        if buy:
            got = qty - (cm if fee_in_base else 0.0)
            if got > 0:
                book.append([t, got, (quote + (0.0 if fee_in_base else fee_usd)) / got])
            continue
        sold = qty + (cm if fee_in_base else 0.0)
        unit_proceeds = (quote - (0.0 if fee_in_base else fee_usd)) / sold if sold > 0 else 0.0
        left = sold
        while left > 1e-12:
            if book:
                lt, lq, lc = book[0]
                take = min(lq, left)
                book[0][1] -= take
                if book[0][1] <= 1e-12:
                    book.pop(0)
                cost, proc = take * lc, take * unit_proceeds
                if lo_ms <= t <= hi_ms:
                    lots_out.append({"Asset": base, "Buy date (UTC)": fmt_dt(lt), "Sell date (UTC)": fmt_dt(t), "Quantity": _r(take, 8),
                                     "Buy price (USDT)": _r(lc, 8), "Sell price (USDT)": _r(unit_proceeds, 8), "Cost (USDT)": _r(cost, 4),
                                     "Proceeds (USDT)": _r(proc, 4), "Realized P&L (USDT)": _r(proc - cost, 4)})
                left -= take
            else:
                if lo_ms <= t <= hi_ms:
                    lots_out.append({"Asset": base, "Buy date (UTC)": "unknown (no earlier buy in this account's history)",
                                     "Sell date (UTC)": fmt_dt(t), "Quantity": _r(left, 8), "Buy price (USDT)": None,
                                     "Sell price (USDT)": _r(unit_proceeds, 8), "Cost (USDT)": None,
                                     "Proceeds (USDT)": _r(left * unit_proceeds, 4), "Realized P&L (USDT)": None})
                left = 0.0
    hold = []
    for a in sorted(balances):
        q = balances[a]
        if q <= 0:
            continue
        book = lots.get(a, [])
        lq = sum(x[1] for x in book)
        cost = sum(x[1] * x[2] for x in book)
        px = 1.0 if a in STABLES else prices.get(a)
        val = q * px if px is not None else None
        known = lq > 0 and abs(lq - q) <= max(1e-8, 0.001 * q)      # the bought lots explain the balance (else cost unknown)
        hold.append({"Asset": a, "Quantity": _r(q, 8), "Average buy price (USDT)": _r(cost / lq, 8) if lq > 0 and a not in STABLES else None,
                     "Cost (USDT)": _r(cost, 4) if known and a not in STABLES else None, "Current price (USDT)": _r(px, 8),
                     "Current value (USDT)": _r(val, 4),
                     "Unrealized P&L (USDT)": _r(val - cost, 4) if (known and val is not None and a not in STABLES) else None})
    return trades_out, hold, lots_out


# ── writers (pure) ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _col(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        s = chr(65 + rem) + s
    return s


def _cell(ref: str, v, style: int = 0) -> str:
    st = f' s="{style}"' if style else ""
    if v is None or v == "":
        return f'<c r="{ref}"{st}/>'
    if isinstance(v, bool):
        v = "Yes" if v else "No"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return f'<c r="{ref}"{st}/>'                     # NaN / inf → blank (an invalid number would break the file)
        return f'<c r="{ref}"{st}><v>{repr(float(v)) if isinstance(v, float) else v}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{st}><is><t>{escape(_CTRL.sub("", str(v)))}</t></is></c>'


def _sheet_xml(cols: List[str], rows: List[dict], notes: List[str]) -> str:
    widths = [max([len(c)] + [len(str(r.get(c) if r.get(c) is not None else "")) for r in rows]) for c in cols]
    cx = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{min(60, w + 2)}" customWidth="1"/>' for i, w in enumerate(widths))
    out, n = [], 1
    for t in notes:
        out.append(f'<row r="{n}">{_cell(f"A{n}", t, 2)}</row>')
        n += 1
    if notes:
        n += 1
    hdr = n
    out.append(f'<row r="{n}">' + "".join(_cell(f"{_col(i)}{n}", c, 1) for i, c in enumerate(cols)) + "</row>")
    for r in rows:
        n += 1
        bold = 1 if str(r.get(cols[0])) == "TOTAL" else 0
        out.append(f'<row r="{n}">' + "".join(_cell(f"{_col(i)}{n}", r.get(c), bold) for i, c in enumerate(cols)) + "</row>")
    pane = (f'<sheetViews><sheetView workbookViewId="0"><pane ySplit="{hdr}" topLeftCell="A{hdr + 1}" activePane="bottomLeft" '
            f'state="frozen"/></sheetView></sheetViews>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'{pane}<cols>{cx}</cols><sheetData>{"".join(out)}</sheetData></worksheet>')


def xlsx_bytes(sheets: List[Tuple[str, List[str], List[dict], List[str]]]) -> bytes:
    """[(name ≤ 31 chars, columns, rows, note lines above the table)] → an .xlsx file (inline strings, bold header + TOTAL rows,
    frozen header, column widths)."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                   + "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" '
                             'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(len(sheets)))
                   + "</Types>")
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                   'Target="xl/workbook.xml"/></Relationships>')
        z.writestr("xl/workbook.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>'
                   + "".join(f'<sheet name="{escape(n[:31])}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, (n, *_r) in enumerate(sheets))
                   + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(f'<Relationship Id="rId{i + 1}" Type="{rel}/worksheet" Target="worksheets/sheet{i + 1}.xml"/>'
                             for i in range(len(sheets)))
                   + f'<Relationship Id="rId{len(sheets) + 1}" Type="{rel}/styles" Target="styles.xml"/></Relationships>')
        z.writestr("xl/styles.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="{ns}">'
                   '<fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font>'
                   '<font><i/><sz val="10"/><name val="Calibri"/></font></fonts>'
                   '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
                   '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
                   '<cellXfs count="3"><xf fontId="0"/><xf fontId="1" applyFont="1"/><xf fontId="2" applyFont="1"/></cellXfs></styleSheet>')
        for i, (_n, cols, rows, notes) in enumerate(sheets):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", _sheet_xml(cols, rows, notes))
    return b.getvalue()


def strip_private(rows: List[dict], cols: List[str]) -> List[dict]:
    return [{c: r.get(c) for c in cols} for r in rows]


# ── Binance reads (read-only) ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _guard(bs):
    """before every futures-API call: an active ban, or the IP's used weight above WEIGHT_STOP → stop (the live bot shares the budget)."""
    from services.binance_service import get_ban_status
    if get_ban_status().get("banned"):
        raise RuntimeError("Binance rate-limit ban active")
    w = bs.used_weight_1m() if hasattr(bs, "used_weight_1m") else None
    if w is not None and w > WEIGHT_STOP:
        raise RuntimeError(f"Binance request budget busy ({w} / min used)")


async def read_income(bs, lo_ms: int, hi_ms: int) -> List[dict]:
    """every income row in [lo, hi] — ≤ 7-day windows, paged on the last row's time, deduped (income_key)."""
    seen, out, w = set(), [], lo_ms
    while w <= hi_ms:
        we, cur = min(w + WINDOW_MS - 1, hi_ms), w
        for _ in range(50):
            _guard(bs)
            rows = await bs.fetch_income_page(cur, we)
            for r in rows:
                k = income_key(r)
                if k not in seen:
                    seen.add(k)
                    out.append(r)
            if len(rows) < 1000:
                break
            nxt = int(rows[-1].get("time") or 0)
            if nxt <= cur:
                raise RuntimeError("income paging made no progress")
            cur = nxt
        w = we + 1
    return sorted(out, key=lambda r: int(r.get("time") or 0))


async def read_capital_flows(bs, lo_ms: int, hi_ms: int) -> List[Tuple[int, float, str]]:
    """completed deposits (+) and withdrawals (−), every coin, [lo, hi] — ≤ 89-day windows, a full page (1,000) splits its window, no cache
    (binance_service.get_capital_flows keeps only the last 89 days). Statuses as the portfolio card: deposit 1 / 6, withdrawal 6."""
    ex = bs.spot_exchange
    out: List[Tuple[int, float, str]] = []

    async def win(a: int, b: int, depth: int = 0):
        deps = await ex.sapiGetCapitalDepositHisrec({"startTime": a, "endTime": b, "limit": 1000})
        wds = await ex.sapiGetCapitalWithdrawHistory({"startTime": a, "endTime": b, "limit": 1000})
        if (len(deps or []) >= 1000 or len(wds or []) >= 1000) and b - a > 60_000 and depth < 20:
            mid = (a + b) // 2
            await win(a, mid, depth + 1)
            await win(mid + 1, b, depth + 1)
            return
        for r in deps or []:
            if str(r.get("status")) in ("1", "6"):
                out.append((int(r.get("insertTime") or 0), abs(_f(r.get("amount"))), str(r.get("coin"))))
        for r in wds or []:
            if str(r.get("status")) == "6":
                t = r.get("completeTime") or r.get("applyTime")
                ms = int(t) if str(t).isdigit() else int(datetime.fromisoformat(str(t)).replace(tzinfo=timezone.utc).timestamp() * 1000)
                out.append((ms, -abs(_f(r.get("amount"))), str(r.get("coin"))))
    a = lo_ms
    while a <= hi_ms:
        b = min(a + CAPITAL_WINDOW_MS - 1, hi_ms)
        await win(a, b)
        a = b + 1
    return sorted(set(out))


async def read_bnb_hourly(bs, lo_ms: int, hi_ms: int) -> Dict[int, float]:
    """{hour start ms: BNB/USDT close} from the public spot klines."""
    await bs._load_spot_markets()
    out, since = {}, lo_ms - 3_600_000
    while since <= hi_ms:
        k = await bs.spot_exchange.fetch_ohlcv("BNB/USDT", "1h", since=since, limit=1000)
        if not k:
            break
        for t, _o, _h, _l, c, _v in k:
            out[int(t)] = float(c)
        if int(k[-1][0]) <= since:
            break
        since = int(k[-1][0]) + 3_600_000
    return out


def usd_rate(bnb_hourly: Dict[int, float], spot_px: Dict[str, float]):
    hours = sorted(bnb_hourly)

    def usd(asset: str, ms: int) -> Optional[float]:
        if asset in STABLES:
            return 1.0
        if asset == "BNB" and hours:
            h = ms - ms % 3_600_000
            if h in bnb_hourly:
                return bnb_hourly[h]
            import bisect
            i = max(0, min(len(hours) - 1, bisect.bisect_right(hours, h) - 1))
            return bnb_hourly[hours[i]]
        return spot_px.get(asset)
    return usd


async def read_spot(bs, extra_assets: Iterable[str]):
    """→ (fills of every stable-quoted symbol of the known coins, {symbol: base}, {asset: total balance}, {asset: USDT price})."""
    await bs._load_spot_markets()
    ex = bs.spot_exchange
    bal = await ex.fetch_balance()
    balances = {}
    for r in (bal.get("info") or {}).get("balances", []):
        q = _f(r.get("free")) + _f(r.get("locked"))
        if q > 0:
            balances[str(r.get("asset"))] = q
    assets = {a for a in balances if a not in STABLES} | {str(a) for a in extra_assets if a and str(a) not in STABLES} | {"BNB"}
    base_of = {}
    for m in ex.markets.values():
        if m.get("spot") and m.get("base") in assets and m.get("quote") in STABLES and m.get("id"):
            base_of[str(m["id"])] = str(m["base"])
    fills = []
    for sym in sorted(base_of):
        frm = 0
        for _ in range(100):
            rows = await ex.privateGetMyTrades({"symbol": sym, "fromId": frm, "limit": 1000})
            fills += [dict(r, symbol=sym) for r in (rows or [])]
            if not rows or len(rows) < 1000:
                break
            frm = int(rows[-1]["id"]) + 1
    prices = {}
    want = [f"{a}/USDT" for a in assets if f"{a}/USDT" in ex.markets]
    if want:
        tk = await ex.fetch_tickers(want)
        for s, t in (tk or {}).items():
            if t and t.get("last"):
                prices[s.split("/")[0]] = float(t["last"])
    return fills, base_of, balances, prices


PAPER_MONTH_COLS = ["Month", "Trades", "Wins", "Losses", "Gross P&L (USDT)", "Trading fees (USDT)", "Funding (USDT)", "Net P&L (USDT)"]
PAPER_NOTE = "PAPER TRADING — simulated, NOT for accounting (a check of the platform's own numbers only)."


def build_paper(orders: List, lo_ms: int, hi_ms: int) -> bytes:
    """(operator 11-04: "make it available too, as to check only") PAPER rows → the trades sheet + the monthly totals, marked as paper;
    no Binance read (the account's transfers / spot / Binance check describe the REAL account, not the simulation)."""
    trades = trade_rows(orders)
    months = monthly_rows(trades, [], [], lambda a, ms: 1.0, None, False)
    period = f"Period {fmt_dt(lo_ms)} → {fmt_dt(hi_ms)} UTC · USDT"
    return xlsx_bytes([
        ("Futures trades (PAPER)", TRADE_COLS, strip_private(trades, TRADE_COLS),
         [PAPER_NOTE, period, "One row per closed futures position. Net P&L = gross P&L + trading fees + funding."]),
        ("Monthly summary (PAPER)", PAPER_MONTH_COLS, strip_private(months, PAPER_MONTH_COLS), [PAPER_NOTE, period]),
    ])


async def build(orders: List, bs, lo_ms: int, hi_ms: int) -> bytes:
    """→ the xlsx bytes. Raises when a Binance read fails: an accountant file is never shipped incomplete."""
    trades = trade_rows(orders)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    inc_lo = max(lo_ms, now_ms - INCOME_HORIZON_MS)                      # Binance keeps ~3 months of income: older months noted, never guessed
    income = await read_income(bs, inc_lo, hi_ms) if inc_lo <= hi_ms else []
    after = await read_income(bs, max(hi_ms + 1, inc_lo), now_ms) if hi_ms < now_ms - 60_000 else []
    _guard(bs)
    flows = await read_capital_flows(bs, lo_ms, hi_ms)
    flows_all = flows
    wb = await bs.get_balance()
    if not wb.get("ok"):
        raise RuntimeError("futures wallet balance read failed")
    fills, base_of, balances, prices = await read_spot(bs, [c for _t, _a, c in flows_all])
    first = min([lo_ms] + [int(f.get("time") or lo_ms) for f in fills])   # BNB fees of older spot fills (their FIFO cost) priced at their hour too
    bnb = await read_bnb_hourly(bs, first, hi_ms)
    usd = usd_rate(bnb, prices)
    # the end-of-month balances walk back from TODAY's wallet: income after the period end is included in the walk
    wallet_at_end = _f(wb.get("usdt_total")) - sum(_f(r.get("income")) for r in after if str(r.get("asset") or "") == "USDT")
    months = monthly_rows(trades, income, flows, usd, wallet_at_end, True, inc_lo if inc_lo > lo_ms else 0)   # only Binance's horizon blanks a month, never the period start
    old = (f"Binance keeps about 3 months of futures history: months before {fmt_dt(inc_lo)[:10]} have no Binance check, no end-of-month "
           "balance and no spot ↔ futures moves (blank)." if inc_lo > lo_ms else "")
    s_trades, holds, lots = spot_fifo(fills, base_of, usd, balances, prices, lo_ms, hi_ms)
    period = f"Period {fmt_dt(lo_ms)} → {fmt_dt(hi_ms)} UTC · USDT"
    sheets = [
        ("Futures trades", TRADE_COLS, strip_private(trades, TRADE_COLS),
         [period, "One row per closed futures position. Net P&L = gross P&L + trading fees + funding.",
          "Per-trade fees and funding are the platform's bookings (some fees are estimated at the exchange fee rate, a missing funding read "
          "counts as 0; positions taken over from the Binance app show the take-over time as open date) — the monthly sheet's Binance "
          "check is the authoritative total."]),
        ("Monthly summary", MONTH_COLS, months,
         [period, "Binance net P&L (check) = Binance's own income history (realized P&L + commissions + funding; BNB fees at the hourly "
                  "BNB price). Difference = this file's net P&L − Binance's.",
          "End-of-month futures balance = the USDT futures wallet at the month's last second (today's wallet less every later income row)."]
         + ([old] if old else [])),
        ("Transfers", TRANSFER_COLS, transfer_rows(income, flows, lo_ms, hi_ms),
         [period, "Deposits and withdrawals of the Binance account, and moves between the spot and futures wallets."] + ([old] if old else [])),
        ("Spot trades", SPOT_TRADE_COLS, s_trades,
         [period, "Spot buys and sells against USDT / USDC / FDUSD (stablecoins counted 1:1 with USDT).",
          "Coins found from the current balances, the deposit / withdrawal history and BNB; a coin bought and sold to zero that never "
          "sat in the wallet and was never deposited is not listed."]),
        ("Spot holdings", HOLDING_COLS, holds,
         [f"Holdings at {fmt_dt(datetime.now(timezone.utc))} UTC. Average buy price / cost from the FIFO lots still held; blank = the "
          "quantity is not fully explained by spot buys (deposited coins, coins moved to futures, or BNB spent on trading fees)."]),
        ("Spot sold lots", LOT_COLS, lots,
         [period, "Each sale matched to the earliest buys first (FIFO). Cost includes the buy fee; proceeds are net of the sell fee. "
                  "Confirm FIFO with your tax rules."]),
    ]
    return xlsx_bytes(sheets)
