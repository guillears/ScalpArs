"""📒 Accountant export (DECISION_LOG 455, operator 2026-11-04: "an excel / csv to download for my accountant … simple, but contain all
the necessary information. Similar to what i would download from binance").

SCOPE      LIVE money (mode=paper → build_paper: the paper trades + monthly totals, marked NOT for accounting — operator 11-04), USDT,
           ARGENTINA TIME UTC-3 for every date and month boundary (DECISION_LOG 457), the period asked. Bot trades, dashboard-opened
           trades and Binance-app trades (auto-recorded, DECISION_LOG 329) are ONE list — nothing in the file says which is which
           (operator: "dont want to disclose bot or manual").
SHEETS     ① Futures trades — one row per closed position from the live Order rows (the dashboard's own numbers): close / open date,
             symbol, side, quantity, entry / exit price, margin, leverage, position size, gross P&L, trading fees, funding, net P&L, result.
             Order convention: pnl is NET of the entry + exit fees and EXCLUDES funding (funding_fee_usd, negative = paid) → gross = pnl
             + fees · net = pnl + funding · margin = position size / leverage (so margin × leverage = position size on every row).
           ② Monthly summary — per month (Argentina time): trades, wins / losses, gross, fees, funding, net (from ①) + the CHECK against Binance's own
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
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

STABLES = ("USDT", "USDC", "FDUSD")
EXPORT_WEIGHT_PER_MIN = 300                     # the export's own futures-API budget (the app recorder and track record use 600 each; the bot first)
WINDOW_MS = 7 * 86_400_000                      # Binance income: ≤ 7-day windows
DAY_MS = 86_400_000
INCOME_HORIZON_MS = 89 * DAY_MS                 # Binance keeps ~3 months of futures income: older months get no check / balance (noted)
CAPITAL_WINDOW_MS = 89 * DAY_MS                 # deposit / withdrawal history: ≤ 90-day windows per call
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
ART = timezone(timedelta(hours=-3))             # 🇦🇷 (DECISION_LOG 457, operator) months and every date in the file: Argentina time, UTC-3
TRADE_COLS = ["Close date (UTC-3)", "Open date (UTC-3)", "Symbol", "Side", "Quantity", "Entry price", "Exit price", "Margin (USDT)",
              "Leverage", "Position size (USDT)", "Gross P&L (USDT)", "Trading fees (USDT)", "Funding (USDT)", "Net P&L (USDT)", "Result"]
# 📊 (DECISION_LOG 461 / 462, operator) + win rate, volume, average margin / leverage, fees % of volume, and on the live file the
# start-of-month balance and the month's return (average / largest win and loss removed 11-04: no value for the accountant)
_MONTH_TRADING = ["Month", "Trades", "Wins", "Losses", "Win rate %", "Volume traded (USDT)", "Avg margin per trade (USDT)", "Avg leverage",
                  "Gross P&L (USDT)", "Trading fees (USDT)", "Fees % of volume", "Funding (USDT)", "Net P&L (USDT)"]
MONTH_COLS = _MONTH_TRADING + ["Binance net P&L (check)", "Difference", "Deposits (USDT)", "Withdrawals (USDT)",
                               "Net transfers into futures (USDT)", "Start-of-month futures balance (USDT)",
                               "End-of-month futures balance (USDT)", "Return %"]
SPOT_QTY = {"Quantity": "price"}                # spot quantities keep up to 8 decimals (a 0.00012345 BTC balance stays readable)
TRADE_COLS_LIVE = TRADE_COLS + ["Open order ID", "Close order ID"]   # 🧾 (DECISION_LOG 460) Binance orders — the same kind of reference on every row
TRANSFER_COLS = ["Date (UTC-3)", "Type", "Asset", "Amount"]
SPOT_TRADE_COLS = ["Date (UTC-3)", "Symbol", "Side", "Quantity", "Price", "Total (USDT)", "Fee (USDT)"]
HOLDING_COLS = ["Asset", "Quantity", "Average buy price (USDT)", "Cost (USDT)", "Current price (USDT)", "Current value (USDT)",
                "Unrealized P&L (USDT)"]
LOT_COLS = ["Asset", "Buy date (UTC-3)", "Sell date (UTC-3)", "Quantity", "Buy price (USDT)", "Sell price (USDT)", "Cost (USDT)",
            "Proceeds (USDT)", "Realized P&L (USDT)"]


def fmt_dt(d) -> str:
    """epoch ms or a datetime (naive = UTC, the database convention) → 'YYYY-MM-DD HH:MM:SS' in Argentina time (UTC-3)."""
    if d is None:
        return ""
    if isinstance(d, (int, float)):
        d = datetime.fromtimestamp(d / 1000, tz=timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(ART).strftime("%Y-%m-%d %H:%M:%S")


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
        "Close date (UTC-3)": fmt_dt(o.closed_at), "Open date (UTC-3)": fmt_dt(o.opened_at), "Symbol": str(o.pair).replace("/", ""),
        "Side": "Long" if str(o.direction).upper() == "LONG" else "Short", "Quantity": _r(_f(o.quantity), 8),
        "Entry price": _r(_f(o.entry_price), 8), "Exit price": _r(_f(o.exit_price), 8), "Margin (USDT)": _r(size / lev, 2),
        "Leverage": _r(lev, 2), "Position size (USDT)": _r(size, 2), "Gross P&L (USDT)": _r(pnl + fees, 4),
        "Trading fees (USDT)": _r(-fees, 4), "Funding (USDT)": _r(funding, 4), "Net P&L (USDT)": _r(net, 4),
        "Result": "Win" if net > 0 else "Loss",
        "_month": fmt_dt(o.closed_at)[:7], "_gross": pnl + fees, "_fees": -fees, "_funding": funding, "_net": net, "_size": size, "_margin": size / lev}


def trade_rows(orders: Iterable) -> List[dict]:
    """CLOSED live rows → sheet ① rows, oldest close first (caller filters is_paper False ∧ CLOSED ∧ the period)."""
    rows = [trade_row(o) for o in orders if getattr(o, "closed_at", None) is not None]
    return sorted(rows, key=lambda r: r["Close date (UTC-3)"])


# ── Binance income → monthly check (pure) ───────────────────────────────────────────────────────────────────────────────────────────

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
    """'YYYY-MM' → (first ms, last ms) of that month in Argentina time (UTC-3)."""
    y, mo = int(m[:4]), int(m[5:7])
    a = int(datetime(y, mo, 1, tzinfo=ART).timestamp() * 1000)
    b = int(datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=ART).timestamp() * 1000) - 1
    return a, b


def dietz_base(start_bal: Optional[float], transfers: Iterable[Tuple[int, float]], lo_ms: int, hi_ms: int) -> Optional[float]:
    """📒 (DECISION_LOG 508, operator 11-09: "make the Return % adjust for withdrawals") — the Modified Dietz capital base: the start
    balance + every transfer into (+) / out of (−) the futures wallet weighted by the share of the span it was there,
    (hi − t) ÷ (hi − lo). A mid-month withdrawal then lowers the base only for the days after it. None when there is no start
    balance or the base is not positive. PURE."""
    if start_bal is None:
        return None
    span = max(1, int(hi_ms) - int(lo_ms))
    base = float(start_bal) + sum(float(a) * min(1.0, max(0.0, (int(hi_ms) - int(t)) / span)) for t, a in transfers)
    return base if base > 0 else None


def monthly_rows(trades: List[dict], income: List[dict], flows: List[Tuple[int, float, str]], usd, wallet_now: Optional[float],
                 binance_ok: bool, income_from_ms: int = 0, period_lo_ms: int = 0, period_hi_ms: Optional[int] = None) -> List[dict]:
    """sheet ② — one row per month (Argentina time) with a trade, an income row or a flow, + TOTAL. income_from_ms = the oldest instant Binance's
    income history still holds: a month starting before it gets no check (blank), a month ending before it no balance (blank).
    period_lo_ms / period_hi_ms = the period the income was filtered to — a caller that filters income MUST pass both, or the Return %
    transfer weights use whole months (DECISION_LOG 508)."""
    tin = [r for r in income if str(r.get("incomeType")) == "TRANSFER" and str(r.get("asset") or "") in STABLES]
    months = sorted({t["_month"] for t in trades} | {month_of_ms(int(r.get("time") or 0)) for r in income}
                    | {month_of_ms(int(t)) for t, _, _ in flows})
    bn = income_month_net(income, usd) if binance_ok else {}
    bal = month_end_balances(months, income, wallet_now) if binance_ok else {m: None for m in months}
    checked = {m for m in months if month_bounds(m)[0] >= income_from_ms}
    bal = {m: (v if month_bounds(m)[1] >= income_from_ms else None) for m, v in bal.items()}
    usdt_in = {m: sum(_f(r.get("income")) for r in income if str(r.get("asset") or "") == "USDT"
                      and month_of_ms(int(r.get("time") or 0)) == m) for m in months}
    start = {m: (None if bal.get(m) is None or m not in checked else bal[m] - usdt_in[m]) for m in months}   # the wallet before the month's income
    # 📒 (508) USDT transfers into (+) / out of (−) the futures wallet — the Return % base is time-weighted for them (Modified Dietz)
    xfer = [(int(r.get("time") or 0), _f(r.get("income"))) for r in income
            if str(r.get("incomeType")) == "TRANSFER" and str(r.get("asset") or "") == "USDT"]

    def _span(m_first, m_last):
        lo, hi = month_bounds(m_first)[0], month_bounds(m_last)[1]
        return max(lo, int(period_lo_ms or 0)), (min(hi, int(period_hi_ms)) if period_hi_ms is not None else hi)

    def _base(m_first, m_last):
        lo, hi = _span(m_first, m_last)
        if lo > hi:
            return None
        return dietz_base(start.get(m_first), [(t, a) for t, a in xfer if lo <= t <= hi], lo, hi)
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
        out.append(_month_row(m, s, g, binance_ok and m in checked, start.get(m), bal.get(m), _base(m, m)))
    out.append(_month_row("TOTAL", tot, trades, binance_ok and set(months) <= checked, start.get(months[0]) if months else None,
                          bal.get(months[-1]) if months else None, _base(months[0], months[-1]) if months else None))
    return out


def _month_row(m, s, trades, ok, start_bal, end_bal, ret_base=None) -> dict:
    """one summary row. Avg leverage = Σ size ÷ Σ margin (size-weighted); fees % of volume; return % = net P&L ÷ the time-weighted capital
    base (DECISION_LOG 508: the start-of-month futures balance + every transfer in / out weighted by the share of the month it was there —
    Modified Dietz; TOTAL: the same over the whole period, not compounded)."""
    vol = sum(t.get("_size", 0.0) for t in trades)
    mar = sum(t.get("_margin", 0.0) for t in trades)
    n = int(s["n"])
    return {"Month": m, "Trades": n, "Wins": int(s["w"]), "Losses": int(s["l"]), "Win rate %": (s["w"] / n) if n else None,
            "Volume traded (USDT)": _r(vol, 2), "Avg margin per trade (USDT)": _r(mar / n, 2) if n else None,
            "Avg leverage": _r(vol / mar, 1) if mar > 0 else None, "Gross P&L (USDT)": _r(s["g"], 2),
            "Trading fees (USDT)": _r(s["f"], 2), "Fees % of volume": (-s["f"] / vol) if vol > 0 else None,
            "Funding (USDT)": _r(s["fu"], 2), "Net P&L (USDT)": _r(s["net"], 2),
            "Binance net P&L (check)": _r(s["bn"], 2) if ok else None, "Difference": _r(s["net"] - s["bn"], 2) if ok else None,
            "Deposits (USDT)": _r(s["dep"], 2), "Withdrawals (USDT)": _r(s["wd"], 2), "Net transfers into futures (USDT)": _r(s["tr"], 2),
            "Start-of-month futures balance (USDT)": _r(start_bal, 2), "End-of-month futures balance (USDT)": _r(end_bal, 2),
            "Return %": (s["net"] / ret_base) if (ret_base and ret_base > 0) else None}


def transfer_rows(income: List[dict], flows: List[Tuple[int, float, str]], lo_ms: int, hi_ms: int) -> List[dict]:
    """sheet ③ — capital deposits / withdrawals (every coin) + futures TRANSFER income (+ = into futures), oldest first."""
    rows = [(int(t), "Deposit" if a > 0 else "Withdrawal", str(c), abs(a)) for t, a, c in flows if lo_ms <= int(t) <= hi_ms]
    rows += [(int(r.get("time") or 0), "Spot → Futures" if _f(r.get("income")) > 0 else "Futures → Spot", str(r.get("asset") or ""),
              abs(_f(r.get("income")))) for r in income if str(r.get("incomeType")) == "TRANSFER"
             and lo_ms <= int(r.get("time") or 0) <= hi_ms]
    return [{"Date (UTC-3)": fmt_dt(t), "Type": k, "Asset": a, "Amount": _r(v, 8)} for t, k, a, v in sorted(rows)]


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
            trades_out.append({"Date (UTC-3)": fmt_dt(t), "Symbol": sym, "Side": "Buy" if buy else "Sell", "Quantity": _r(qty, 8),
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
                    lots_out.append({"Asset": base, "Buy date (UTC-3)": fmt_dt(lt), "Sell date (UTC-3)": fmt_dt(t), "Quantity": _r(take, 8),
                                     "Buy price (USDT)": _r(lc, 8), "Sell price (USDT)": _r(unit_proceeds, 8), "Cost (USDT)": _r(cost, 4),
                                     "Proceeds (USDT)": _r(proc, 4), "Realized P&L (USDT)": _r(proc - cost, 4)})
                left -= take
            else:
                if lo_ms <= t <= hi_ms:
                    lots_out.append({"Asset": base, "Buy date (UTC-3)": "unknown (no earlier buy in this account's history)",
                                     "Sell date (UTC-3)": fmt_dt(t), "Quantity": _r(left, 8), "Buy price (USDT)": None,
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
# 🔢 (DECISION_LOG 460, operator: "numbers … with 2 decimals") display formats only — the cell keeps the full value
_FMT_MONEY, _FMT_PRICE, _FMT_INT, _FMT_PCT, _FMT_QTY = "#,##0.00", "#,##0.00######", "#,##0", "0.00%", "#,##0.0000"
STYLE = {None: (0, 1), "money": (3, 4), "price": (5, 6), "int": (7, 8), "pct": (9, 10), "qty": (11, 12)}   # (plain, bold TOTAL) cellXfs per kind
_PRICE_WORDS = ("price", "amount")


def col_kind(name: str) -> Optional[str]:
    """a column header → money (USDT, 2 decimals) · price (prices / quantities / coin amounts: ≥ 2, ≤ 8 decimals) · int · None (text)."""
    n = name.lower()
    if "%" in n:
        return "pct"                                       # a fraction shown as 12.34 %
    if n == "quantity":
        return "qty"                                       # operator 11-04: 4 decimals are enough for a quantity
    if any(w in n for w in _PRICE_WORDS):
        return "price"
    if n in ("trades", "wins", "losses"):
        return "int"                                       # leverage stays General: a 2.5× bracket leverage never shows as 3
    if "(usdt)" in n or n in ("binance net p&l (check)", "difference"):
        return "money"
    return None


def _shown(v, kind) -> str:
    """the text a cell displays under its format (column widths: a formatted number never shows as ####)."""
    if v is None:
        return ""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
        if kind == "money":
            return f"{v:,.2f}"
        if kind == "price":
            t = f"{v:,.8f}".rstrip("0")
            return t + "0" * max(0, 2 - len(t.split(".")[1])) if "." in t else t
        if kind == "int":
            return f"{v:,.0f}"
        if kind == "pct":
            return f"{v * 100:.2f}%"
        if kind == "qty":
            return f"{v:,.4f}"
    return str(v)


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


def _sheet_xml(cols: List[str], rows: List[dict], notes: List[str], kind_over: Optional[Dict[str, str]] = None) -> str:
    kinds = [(kind_over or {}).get(c, col_kind(c)) for c in cols]
    widths = [max([len(c)] + [len(_shown(r.get(c), k)) for r in rows]) for c, k in zip(cols, kinds)]
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
        out.append(f'<row r="{n}">' + "".join(_cell(f"{_col(i)}{n}", r.get(c), STYLE[kinds[i]][bold]) for i, c in enumerate(cols)) + "</row>")
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
                   f'<numFmts count="3"><numFmt numFmtId="164" formatCode="{_FMT_MONEY}"/><numFmt numFmtId="165" formatCode="{_FMT_PRICE}"/>'
                   f'<numFmt numFmtId="166" formatCode="{_FMT_QTY}"/></numFmts>'
                   '<fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font>'
                   '<font><i/><sz val="10"/><name val="Calibri"/></font></fonts>'
                   '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
                   '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
                   '<cellXfs count="13"><xf fontId="0"/><xf fontId="1" applyFont="1"/><xf fontId="2" applyFont="1"/>'
                   '<xf numFmtId="164" fontId="0" applyNumberFormat="1"/><xf numFmtId="164" fontId="1" applyNumberFormat="1" applyFont="1"/>'
                   '<xf numFmtId="165" fontId="0" applyNumberFormat="1"/><xf numFmtId="165" fontId="1" applyNumberFormat="1" applyFont="1"/>'
                   '<xf numFmtId="3" fontId="0" applyNumberFormat="1"/><xf numFmtId="3" fontId="1" applyNumberFormat="1" applyFont="1"/>'
                   '<xf numFmtId="10" fontId="0" applyNumberFormat="1"/><xf numFmtId="10" fontId="1" applyNumberFormat="1" applyFont="1"/>'
                   '<xf numFmtId="166" fontId="0" applyNumberFormat="1"/><xf numFmtId="166" fontId="1" applyNumberFormat="1" applyFont="1"/>'
                   '</cellXfs></styleSheet>')
        for i, (_n, cols, rows, notes, *over) in enumerate(sheets):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", _sheet_xml(cols, rows, notes, over[0] if over else None))
    return b.getvalue()


def strip_private(rows: List[dict], cols: List[str]) -> List[dict]:
    return [{c: r.get(c) for c in cols} for r in rows]


# ── Binance reads (read-only) ───────────────────────────────────────────────────────────────────────────────────────────────────────
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


PAPER_MONTH_COLS = list(_MONTH_TRADING)
PAPER_NOTE = "PAPER TRADING — simulated, NOT for accounting (a check of the platform's own numbers only)."


def build_paper(orders: List, lo_ms: int, hi_ms: int) -> bytes:
    """(operator 11-04: "make it available too, as to check only") PAPER rows → the trades sheet + the monthly totals, marked as paper;
    no Binance read (the account's transfers / spot / Binance check describe the REAL account, not the simulation)."""
    trades = trade_rows(orders)
    months = monthly_rows(trades, [], [], lambda a, ms: 1.0, None, False)
    period = f"Period {fmt_dt(lo_ms)} → {fmt_dt(hi_ms)} (UTC-3) · USDT"
    return xlsx_bytes([
        ("Futures trades (PAPER)", TRADE_COLS, strip_private(trades, TRADE_COLS),
         [PAPER_NOTE, period, "One row per closed futures position. Net P&L = gross P&L + trading fees + funding."]),
        ("Monthly summary (PAPER)", PAPER_MONTH_COLS, strip_private(months, PAPER_MONTH_COLS),
         [PAPER_NOTE, period, "Volume = Σ position size (entry side) — fees % of volume therefore covers the round trip. Avg leverage = "
                              "volume ÷ Σ margin (size-weighted)."]),
    ])


# ── live futures trades from Binance's own fills (DECISION_LOG 457) ──────────────────────────────────────────────────────────────────
def first_flat_ms(fills: List, start_net: float) -> Optional[int]:
    """the time the replay first reaches a flat book from start_net (None = it never does inside these fills) — pure."""
    from services.app_trades import qty_eps
    eps, net = qty_eps(fills), float(start_net)
    if abs(net) <= eps:
        return None
    for f in sorted(fills, key=lambda x: (x.t, x.trade_id)):
        net += f.signed
        if abs(net) <= eps or (net > 0) != (start_net > 0):
            return f.t
    return None


def episode_row(ep, comm_usd: float, funding: float, leverage: Optional[float]) -> dict:
    """one closed Binance position (services.app_trades.Episode) → sheet ① row: gross = Σ Binance realized P&L (before fees), fees = Σ
    commission actually charged (USDT), funding = Σ funding paid / received while open, net = gross − fees + funding; size = entry VWAP
    × peak quantity; margin = size / leverage — pure."""
    from services.app_trades import vwap
    qty = ep.max_abs_net
    entry, exitp = vwap(ep.opens()), vwap(ep.closes())
    gross = sum(l.rpnl for l in ep.legs)
    size = (entry or 0.0) * qty
    lev = float(leverage) if leverage and float(leverage) >= 1 else 1.0
    net = gross - comm_usd + funding
    return {"Close date (UTC-3)": fmt_dt(ep.end_ms), "Open date (UTC-3)": fmt_dt(ep.start_ms), "Symbol": ep.symbol,
            "Side": "Long" if ep.direction == "LONG" else "Short", "Quantity": _r(qty, 8), "Entry price": _r(entry, 8),
            "Exit price": _r(exitp, 8), "Margin (USDT)": _r(size / lev, 2), "Leverage": _r(lev, 2), "Position size (USDT)": _r(size, 2),
            "Gross P&L (USDT)": _r(gross, 4), "Trading fees (USDT)": _r(-comm_usd, 4), "Funding (USDT)": _r(funding, 4),
            "Net P&L (USDT)": _r(net, 4), "Result": "Win" if net > 0 else "Loss",
            "Open order ID": ", ".join(dict.fromkeys(str(l.fill.order_id) for l in ep.opens() if l.fill.order_id not in (None, "None", ""))),
            "Close order ID": ", ".join(dict.fromkeys(str(l.fill.order_id) for l in ep.closes() if l.fill.order_id not in (None, "None", ""))),
            "_month": fmt_dt(ep.end_ms)[:7], "_gross": gross, "_fees": -comm_usd, "_funding": funding, "_net": net, "_close_ms": ep.end_ms,
            "_size": size, "_margin": size / lev}


async def futures_from_fills(rec, income: List[dict], fetch_lo: int, now_ms: int, lo_ms: int, hi_ms: int,
                             lev_lookup: Callable[[str, int], Optional[float]], horizon_ms: int) -> List[dict]:
    """every position that CLOSED in [lo, hi], rebuilt from Binance's fills (rec = services.app_trades.AppTradeRecorder: its budgeted,
    ban-aware reads). Symbols and fill windows from the account's income (REALIZED_PNL / COMMISSION rows); each symbol's replay is anchored
    on its live position now; a position already open before fetch_lo whose close lands in the period reads earlier 7-day windows until it
    is flat (never before horizon_ms → RuntimeError: never an invented entry)."""
    from services.app_trades import build_episodes, comm_usdt, W_POS
    active: Dict[str, set] = {}
    for r in income:
        if str(r.get("incomeType")) in ("REALIZED_PNL", "COMMISSION") and r.get("symbol"):
            active.setdefault(str(r["symbol"]), set()).add((int(r.get("time") or 0) - fetch_lo) // WINDOW_MS)
    positions = await rec._call(W_POS, rec.svc.get_net_positions) if active else {}
    if not isinstance(positions, dict):
        raise RuntimeError("Binance positions unreadable")
    eps = []
    for sym in sorted(active):
        fills, seen, fetched = [], set(), set()

        async def read(k):
            w0 = max(fetch_lo + k * WINDOW_MS, horizon_ms)    # the oldest week is clamped to what Binance still holds
            fetched.add(k)
            for f in await rec._fills(sym, w0, min(w0 + WINDOW_MS - 1, now_ms)):
                if f.trade_id not in seen:
                    seen.add(f.trade_id)
                    fills.append(f)
        for k in sorted(active[sym]):
            await read(k)
        while True:
            start_net = float(positions.get(sym, 0.0)) - sum(f.signed for f in fills)
            flat = first_flat_ms(fills, start_net)
            if flat is None or flat < lo_ms:
                break                                         # flat at the start, or the leading position closed before the period
            k = min(fetched) - 1                              # every week back from the earliest one read (an opening fill may carry no fee row)
            if fetch_lo + (k + 1) * WINDOW_MS - 1 < horizon_ms:
                raise RuntimeError(f"{sym}: a position open before {fmt_dt(horizon_ms)} closes in the period — Binance no longer has its entry")
            await read(k)
        done, _open, _lead = build_episodes(sym, fills, float(positions.get(sym, 0.0)) - sum(f.signed for f in fills))
        eps += [e for e in done if lo_ms <= e.end_ms <= hi_ms]
    await rec.prepare_rates({"BNB": [l.fill.t for e in eps for l in e.legs if l.fill.comm_asset == "BNB"]})
    rows = []
    for e in eps:
        comm, _est = comm_usdt(e.legs, rec.rate)              # raises on a fee asset it cannot price (never guessed)
        fund = 0.0
        early = await rec._income(e.start_ms, fetch_lo - 1, e.symbol, "FUNDING_FEE") if e.start_ms < fetch_lo else []
        for r in list(income) + list(early):                 # a position read back past the window: its earlier funding too
            if str(r.get("incomeType")) == "FUNDING_FEE" and str(r.get("symbol")) == e.symbol and e.start_ms <= int(r.get("time") or 0) <= e.end_ms:
                px = rec.rate(str(r.get("asset") or "USDT"), int(r.get("time") or 0))
                if px is None:
                    raise RuntimeError(f"funding paid in {r.get('asset')} cannot be converted to USDT")
                fund += _f(r.get("income")) * px[0]
        lev = lev_lookup(e.symbol, e.start_ms) if lev_lookup else None
        if not lev:
            lev = await rec._leverage(e.symbol)
        rows.append(episode_row(e, comm, fund, lev))
    return sorted(rows, key=lambda r: r["_close_ms"])


async def build_live(lo_ms: int, hi_ms: int, lev_lookup=None, rec=None, bs=None) -> Tuple[bytes, dict]:
    """→ (xlsx bytes, {trades, net, activity}). Every number from Binance (read-only); raises when a read fails — an accountant file is
    never shipped incomplete."""
    if rec is None:
        from services.app_trades import AppTradeRecorder, WeightBudget
        rec = AppTradeRecorder(budget=WeightBudget(per_min=EXPORT_WEIGHT_PER_MIN))
    if bs is None:
        from services.binance_service import binance_service as bs
    now_ms = rec.now_ms()
    horizon = now_ms - INCOME_HORIZON_MS
    if lo_ms < horizon:                                   # trades before it cannot be read: never a file with silent zeros
        raise RuntimeError(f"Binance keeps about 3 months of futures history — the period starts before {fmt_dt(horizon)[:10]}: "
                           "download those months from the stored monthly statements")
    fetch_lo = max(lo_ms - WINDOW_MS, horizon)            # a week before the period: positions that opened earlier and close inside it
    income_all = await rec._income(fetch_lo, now_ms)
    trades = await futures_from_fills(rec, income_all, fetch_lo, now_ms, lo_ms, hi_ms, lev_lookup, horizon)
    income = [r for r in income_all if lo_ms <= int(r.get("time") or 0) <= hi_ms]
    after = [r for r in income_all if int(r.get("time") or 0) > hi_ms]
    if rec.ban_active():                                  # the spot / capital reads below go through the spot client: same IP, same ban
        raise RuntimeError("Binance rate-limit ban active")
    flows = await read_capital_flows(bs, lo_ms, hi_ms)
    wb = await rec._call(5, bs.get_balance)
    if not wb or not wb.get("ok"):
        raise RuntimeError("futures wallet balance read failed")
    fills, base_of, balances, prices = await read_spot(bs, [c for _t, _a, c in flows])
    first = min([lo_ms] + [int(f.get("time") or lo_ms) for f in fills])
    bnb = await read_bnb_hourly(bs, first, hi_ms)
    usd = usd_rate(bnb, prices)
    wallet_at_end = _f(wb.get("usdt_total")) - sum(_f(r.get("income")) for r in after if str(r.get("asset") or "") == "USDT")
    months = monthly_rows(trades, income, flows, usd, wallet_at_end, True, period_lo_ms=lo_ms, period_hi_ms=hi_ms)
    old = ""
    s_trades, holds, lots = spot_fifo(fills, base_of, usd, balances, prices, lo_ms, hi_ms)
    xfers = transfer_rows(income, flows, lo_ms, hi_ms)
    period = f"Period {fmt_dt(lo_ms)} → {fmt_dt(hi_ms)} (UTC-3) · USDT"
    sheets = [
        ("Futures trades", TRADE_COLS_LIVE, strip_private(trades, TRADE_COLS_LIVE),
         [period, "One row per futures position closed in the period, rebuilt from Binance's own fills: gross P&L = Binance's realized "
                  "P&L, trading fees = the commission Binance charged, funding = paid / received while open. Net = gross + fees + funding.",
          "Margin and leverage: the leverage the position was opened with when the platform recorded it, else the coin's leverage setting "
          "at the time of this file (Binance does not keep the leverage of past trades). Open / Close order ID = the Binance orders that opened and "
          "closed the position (several → comma-separated), to find each row in Binance's trade history."]),
        ("Monthly summary", MONTH_COLS, months,
         [period, "Binance net P&L (check) = Binance's income history (realized P&L + commissions + funding; BNB fees at the hourly BNB "
                  "price). Difference = this file's net P&L − Binance's (funding outside a closed position, rounding).",
          "End-of-month futures balance = the USDT futures wallet at the month's last second (today's wallet less every later income row); "
          "start-of-month = that less the month's own income. Return % = net P&L ÷ the month's average capital: the start-of-month balance "
          "plus each transfer into / out of futures counted for the share of the month it was there (a mid-month withdrawal lowers it only "
          "after that day — Modified Dietz; USDT transfers only, like the balance). TOTAL: the same over the whole period, not compounded; a period starting mid-month: from the "
          "balance at the period start. Volume = Σ position size (entry side) — "
          "fees % of volume therefore covers the round trip. Avg leverage = volume ÷ Σ margin (size-weighted)."]
         + ([old] if old else [])),
        ("Transfers", TRANSFER_COLS, xfers,
         [period, "Deposits and withdrawals of the Binance account, and moves between the spot and futures wallets."] + ([old] if old else [])),
        ("Spot trades", SPOT_TRADE_COLS, s_trades,
         [period, "Spot buys and sells against USDT / USDC / FDUSD (stablecoins counted 1:1 with USDT).",
          "Coins found from the current balances, the deposit / withdrawal history and BNB; a coin bought and sold to zero that never "
          "sat in the wallet and was never deposited is not listed."], SPOT_QTY),
        ("Spot holdings", HOLDING_COLS, holds,
         [f"Holdings at {fmt_dt(datetime.now(timezone.utc))} (UTC-3). Average buy price / cost from the FIFO lots still held; blank = the "
          "quantity is not fully explained by spot buys (deposited coins, coins moved to futures, or BNB spent on trading fees)."], SPOT_QTY),
        ("Spot sold lots", LOT_COLS, lots,
         [period, "Each sale matched to the earliest buys first (FIFO). Cost includes the buy fee; proceeds are net of the sell fee. "
                  "Confirm FIFO with your tax rules."], SPOT_QTY),
    ]
    stats = dict(trades=len(trades), net=sum(t["_net"] for t in trades), activity=bool(trades or xfers or s_trades))
    return xlsx_bytes(sheets), stats


# ── monthly statements (DECISION_LOG 457): the previous Argentina month, built on day 2, stored for good ──────────────────────────────
STATEMENT_DAY = 2                       # operator 11-04: "maybe day 2" — late bookings / the last funding are in
STATEMENT_INTERVAL_S = 3600             # hourly: a failed build (Binance busy) retries the next hour
STATEMENT_FIRST_DELAY_S = 300


def statement_name(month: str) -> str:
    return f"scalpars_accounting_{month}.xlsx"


def due_months(now_utc: datetime, n_back: int = 2) -> List[str]:
    """the months whose statement is due at now (Argentina time): the previous month from day STATEMENT_DAY 00:00 on, and the one before
    it (a missed run while Binance still holds it) — newest first — pure."""
    a = now_utc.replace(tzinfo=timezone.utc).astimezone(ART) if now_utc.tzinfo is None else now_utc.astimezone(ART)
    y, m = a.year, a.month
    out = []
    for k in range(1, n_back + 1):
        mm, yy = m - k, y
        while mm <= 0:
            mm, yy = mm + 12, yy - 1
        out.append(f"{yy:04d}-{mm:02d}")
    return out if a.day >= STATEMENT_DAY else out[1:]


async def leverage_lookup_for(db, lo_ms: int, hi_ms: int):
    """{(symbol) → [(open ms, leverage)]} of the live rows the platform recorded around the period → lookup(sym, open_ms): the leverage of a
    row opened within ±120 s of the Binance position's first fill, else None."""
    from sqlalchemy import select, and_
    from models import Order
    lo = datetime.fromtimestamp((lo_ms - 30 * DAY_MS) / 1000, tz=timezone.utc).replace(tzinfo=None)
    hi = datetime.fromtimestamp(hi_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
    rows = (await db.execute(select(Order.pair, Order.opened_at, Order.leverage).where(and_(
        Order.is_paper == False, Order.opened_at >= lo, Order.opened_at <= hi)))).all()  # noqa: E712
    idx: Dict[str, List[Tuple[int, float]]] = {}
    for p, o, lv in rows:
        if o is not None and lv:
            idx.setdefault(str(p).replace("/", ""), []).append((int(o.replace(tzinfo=timezone.utc).timestamp() * 1000), float(lv)))

    def look(sym: str, t: int) -> Optional[float]:
        best = min(((abs(ot - t), lv) for ot, lv in idx.get(sym, [])), default=None)
        return best[1] if best and best[0] <= 120_000 else None
    return look


async def statement_pass(sf=None, commit=None, now_utc: Optional[datetime] = None, build=None, keys=None) -> List[str]:
    """build every due month without a READY / EMPTY statement → stored; a failure is stored as ERROR (retried next pass)."""
    from sqlalchemy import select
    from models import AccountingStatement
    if sf is None:
        from database import AsyncSessionLocal as sf
    if commit is None:
        from database import locked_commit as commit
    if keys is None:
        from services.track_record import _keys_configured as keys
    if not keys():
        return []
    now_utc = now_utc or datetime.now(timezone.utc).replace(tzinfo=None)
    done = []
    for month in due_months(now_utc):
        lo, hi = month_bounds(month)
        async with sf() as db:                                # short read session — never held open across the Binance reads
            row = (await db.execute(select(AccountingStatement).where(AccountingStatement.month == month))).scalars().first()
            if row is not None and row.status in ("READY", "EMPTY"):
                continue
            look = await leverage_lookup_for(db, lo, hi)
        try:
            xlsx, st = await (build or build_live)(lo, hi, look)
            status, err = ("READY" if st["activity"] else "EMPTY"), None
        except Exception as e:
            xlsx, st, status, err = None, {}, "ERROR", f"{type(e).__name__}: {e}"[:300]
            logger.error(f"[ACCOUNTING_STATEMENT] {month}: not built — {err} (retried next hour)")
        async with sf() as db:
            row = (await db.execute(select(AccountingStatement).where(AccountingStatement.month == month))).scalars().first()
            if row is None:
                row = AccountingStatement(month=month, status=status, attempts=0)
                db.add(row)
            row.attempts = int(row.attempts or 0) + 1
            row.status, row.error = status, err
            if xlsx is not None:
                row.xlsx, row.size, row.built_at = xlsx, len(xlsx), now_utc
                row.trades, row.net_pnl = int(st.get("trades") or 0), float(st.get("net") or 0.0)
                logger.info(f"[ACCOUNTING_STATEMENT] {month}: {status} · {row.trades} trades · net {row.net_pnl:+.2f} USDT · {len(xlsx)} bytes")
            await commit(db)
            if status != "ERROR":
                done.append(month)
    return done


async def statement_loop(should_stop: Callable[[], bool], lock=None):
    """background task: every hour, statement_pass (under the export lock — one Binance-reading export at a time). Never raises out."""
    import asyncio
    await asyncio.sleep(STATEMENT_FIRST_DELAY_S)
    while not should_stop():
        try:
            if lock is not None:
                async with lock:
                    await statement_pass()
            else:
                await statement_pass()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[ACCOUNTING_STATEMENT] pass failed: {type(e).__name__}: {e}", exc_info=True)
        for _ in range(STATEMENT_INTERVAL_S):
            if should_stop():
                return
            await asyncio.sleep(1)
