"""🧾 Investor monthly statement — one-page A4 PDF per investor per month, built ON CLICK (DECISION_LOG 476).

Design agreed with the operator 11-02 → 11-05 (memory: project_investor_statement): the investor's OWN numbers only — no ownership %,
no NAV per share, no platform $, no trade sizes (nothing an investor could use to work out the size of the whole account). Every
per-trade / platform figure is a ratio (count, win rate, average % per trade); every $ on the page is the investor's own money.

Time: days and months close at 00:00 UTC-3 — the NAV history (nav_snapshots, one row per UTC-3 day), the dashboard days and the
accountant Excel already use it. Nothing is stored: a statement is recomputed from the ledger + NAV history each time it is opened
(the footer prints when it was generated); the current month is marked PRELIMINARY.

Money math (all from the investor's ledger + the daily NAV closes):
  balance(d)  = shares held at the end of day d × NAV/share close of day d (a day without a close carries the last known one)
  day P&L(d)  = balance(d) − balance(d−1) − net flows(d)                (the investor's own trading result that day, $)
  day return  = NAV(d) ÷ NAV(d−1) − 1  (time-weighted: deposits never move it; the join day is measured from the NAV the shares were
                bought at)
  month result = ending − starting balance − net flows · month return % = result ÷ (starting balance + deposits in the month)
  daily compound return = (Π (1 + day return))^(1/days) − 1 over every day of the investor's period
  largest drop = the deepest fall of the NAV path from its running peak in the period ($ = the investor's day P&L from the peak to the
                 trough — deposits never count) · best / worst day by day return, with that day's $.
PURE core (compute_statement) + renderer (render_pdf); main.py reads the DB and calls them."""
from __future__ import annotations

import calendar
import io
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

UTC3 = timezone(timedelta(hours=-3))
IN_TYPES = ("DEPOSIT", "FOUNDING", "OPENING", "TRANSFER_IN")
OUT_TYPES = ("WITHDRAW", "CASHOUT", "TRANSFER_OUT")
SHORT_PERIOD_DAYS = 15          # below this the page notes the period is too short to be representative
MAX_MONTH_ROWS = 6              # "Since you joined" keeps one A4 page: the last 5 months + one row folding every earlier month (12 overflowed)

ABOUT = ("SCALPARS is an automated system that trades crypto futures 24/7, built on statistical expectancy. Every rule is "
         "forward-tested and checked against randomised results before it trades capital, then reviewed regularly against real "
         "trading results and updated only when the evidence supports it. Every position carries predefined exit and risk "
         "parameters. The operator may occasionally place or close trades manually.")
DISCLAIMER_PAPER = "Results shown are simulated (paper trading) and involve no real money. "
DISCLAIMER = ("This statement is confidential and intended only for the named investor. It is provided for information only, is "
              "unaudited, and does not constitute investment advice or an offer. Crypto futures are high risk: leverage can magnify "
              "losses, and you can lose all invested capital. Funds are held on a third-party exchange and are subject to its "
              "operational and counterparty risks. Past results do not guarantee future results. If you have any questions, or "
              "believe anything in this statement is incorrect, please let us know within 10 days. Figures in USDT, shown as $; "
              "days and months close at 00:00 UTC-3.")


# ── time helpers (pure) ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def day_utc3(t: datetime) -> date:
    """UTC-3 calendar day of a UTC-naive (or aware) moment — the NavSnapshot.date key."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(UTC3).date()


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_days(month: str):
    y, m = int(month[:4]), int(month[5:7])
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def next_month(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"{y + (m == 12):04d}-{(m % 12) + 1:02d}"


def utc_bounds(first: date, last: date):
    """[first 00:00 UTC-3, (last + 1) 00:00 UTC-3) as UTC-naive datetimes — the SQL window for a span of UTC-3 days."""
    lo = datetime(first.year, first.month, first.day, tzinfo=UTC3).astimezone(timezone.utc).replace(tzinfo=None)
    hi_d = last + timedelta(days=1)
    hi = datetime(hi_d.year, hi_d.month, hi_d.day, tzinfo=UTC3).astimezone(timezone.utc).replace(tzinfo=None)
    return lo, hi


def fmt_day(d: date) -> str:
    return f"{calendar.month_abbr[d.month]} {d.day}, {d.year}"


def fmt_day_short(d: date) -> str:
    return f"{calendar.month_abbr[d.month]} {d.day}"


def month_label(month: str) -> str:
    return f"{calendar.month_abbr[int(month[5:7])]} {month[:4]}"


def month_name(month: str) -> str:
    return calendar.month_name[int(month[5:7])]


def signed_flow(row: dict) -> float:
    """+ money into the investor's balance, − out. ADJUST (a deposit-total correction) takes the sign of its share change."""
    t, amt = str(row.get("type") or "").upper(), abs(float(row.get("amount") or 0.0))
    if t in IN_TYPES:
        return amt
    if t in OUT_TYPES:
        return -amt
    if t == "ADJUST":
        return amt if float(row.get("shares_delta") or 0.0) >= 0 else -amt
    return 0.0


def dep_wd(rows: List[dict]):
    """(deposits, withdrawals) of ledger rows: an ADJUST (a deposit-total correction — the NAV rebuild, a deposit edit, a paper top-up
    mint) moves DEPOSITS by its sign and never counts as a withdrawal (deep review 476 — models.InvestorLedger: "must NEVER touch
    total_withdrawn"). PURE."""
    dep = wd = 0.0
    for r in rows:
        f = signed_flow(r)
        if str(r.get("type") or "").upper() in OUT_TYPES:
            wd -= f
        else:
            dep += f
    return dep, wd


REBUILD_MARK = "DL 422 NAV rebuild"      # services/nav_flows.REBUILD_MARK (pinned by a test)


def normalize_ledger(ledger: List[dict]) -> List[dict]:
    """the one-off NAV rebuild (DECISION_LOG 422) wrote its ADJUST rows when it RAN, but restated the NAV history back to the batch start:
    its "opening capital" row belongs to the opening row's moment and each "paper top-up MM-DD HH:MM UTC" row to the top-up's own time
    (deep review 476: left at the run time, the days in between were mis-valued and the run day booked a fake P&L). Other rows unchanged.
    PURE — returns new dicts."""
    import re as _re
    rows = [dict(r) for r in ledger]
    plain = [r["created_at"] for r in rows if REBUILD_MARK not in str(r.get("note") or "")]
    opening = min(plain) if plain else None
    for r in rows:
        note = str(r.get("note") or "")
        if REBUILD_MARK not in note:
            continue
        if "opening capital" in note and opening is not None:
            r["created_at"] = opening
            continue
        m = _re.search(r"paper top-up (\d{2})-(\d{2}) (\d{2}):(\d{2}) UTC", note)
        if m:
            ca = r["created_at"]
            mo, dy, hh, mi = (int(x) for x in m.groups())
            yr = ca.year - (1 if mo > ca.month else 0)
            try:
                r["created_at"] = datetime(yr, mo, dy, hh, mi)
            except ValueError:
                pass
    return rows


def join_day(ledger: List[dict]) -> Optional[date]:
    ins = [r["created_at"] for r in ledger if signed_flow(r) > 0]
    return day_utc3(min(ins)) if ins else None


def statement_months(ledger: List[dict], now_utc: datetime) -> List[str]:
    """every month from the investor's first deposit to the current UTC-3 month (oldest first)."""
    j = join_day(ledger)
    if j is None:
        return []
    out, m, cur = [], month_key(j), month_key(day_utc3(now_utc))
    while m <= cur:
        out.append(m)
        m = next_month(m)
    return out


# ── the numbers (pure) ───────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class DayRow:
    day: date
    balance: float
    flow: float
    pnl: float
    ret: Optional[float]
    deposited_net: float


@dataclass
class Statement:
    name: str
    month: str
    first: date
    last: date
    join: date
    preliminary: bool
    simulated: bool
    generated_at: datetime
    start_balance: float
    end_balance: float
    deposits: float
    withdrawals: float
    result: float
    ret_pct: Optional[float]
    total_deposited: float          # gross, all-time to the period end
    total_withdrawn: float
    total_profit: float             # all-time result to the period end
    total_profit_pct: Optional[float]
    days: List[DayRow]
    trades: int
    win_rate: Optional[float]
    avg_pnl_pct: Optional[float]
    dcr_pct: Optional[float]
    max_dd_pct: float
    max_dd_usd: float
    best: Optional[DayRow]
    worst: Optional[DayRow]
    positive_days: int
    mixed_until: Optional[datetime] = None
    month_rows: List[dict] = field(default_factory=list)
    month_total: Optional[dict] = None


class StatementError(ValueError):
    """the statement cannot be built (no deposit, a month outside the investor's life, no NAV history)."""


def _nav_lookup(snaps: Dict[date, float]):
    keys = sorted(snaps)

    def nav_at(d: date) -> Optional[float]:
        lo, hi, best = 0, len(keys) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if keys[mid] <= d:
                best, lo = keys[mid], mid + 1
            else:
                hi = mid - 1
        return snaps[best] if best is not None else None
    return nav_at


def _walk(ledger: List[dict], snaps: Dict[date, float], first: date, last: date):
    """→ (start balance (end of first − 1), [DayRow per day first..last], net deposited at first − 1). PURE."""
    nav_at = _nav_lookup(snaps)
    rows = sorted(ledger, key=lambda r: r["created_at"])
    by_day: Dict[date, List[dict]] = {}
    for r in rows:
        by_day.setdefault(day_utc3(r["created_at"]), []).append(r)
    shares = sum(float(r.get("shares_delta") or 0.0) for r in rows if day_utc3(r["created_at"]) < first)
    dep_net = sum(signed_flow(r) for r in rows if day_utc3(r["created_at"]) < first)
    prev_nav = nav_at(first - timedelta(days=1))
    if prev_nav is None and shares > 1e-9:          # holding shares from before the NAV history starts (an epoch reset): the
        prev_nav = nav_at(first) or next((float(r["nav_at_time"]) for r in reversed(rows)   # first known value is the base
                                          if r.get("nav_at_time") and day_utc3(r["created_at"]) < first), None)
    start_bal = shares * prev_nav if (prev_nav is not None and shares > 1e-9) else 0.0
    prev_bal, out, d = start_bal, [], first
    while d <= last:
        todays = by_day.get(d, [])
        flow = sum(signed_flow(r) for r in todays)
        base = prev_nav if shares > 1e-9 else None           # held shares overnight → yesterday's close is the base
        if base is None:                                       # (re)entered today → the NAV the shares were bought at
            buys = [r for r in todays if float(r.get("shares_delta") or 0.0) > 0 and r.get("nav_at_time")]
            base = float(buys[0]["nav_at_time"]) if buys else None
        shares += sum(float(r.get("shares_delta") or 0.0) for r in todays)
        dep_net += flow
        nav = nav_at(d)
        if nav is None:
            nav = base
        if nav is None:
            raise StatementError(f"no NAV history on or before {d.isoformat()}")
        bal = max(shares, 0.0) * nav
        held = shares > 1e-9 or abs(flow) > 1e-9
        ret = (nav / base - 1.0) if (held and base) else None
        out.append(DayRow(day=d, balance=bal, flow=flow, pnl=bal - prev_bal - flow, ret=ret, deposited_net=dep_net))
        prev_bal, prev_nav = bal, nav
        d += timedelta(days=1)
    return start_bal, out


def _month_row(ledger, snaps, month, join, today):
    first, last = month_days(month)
    first, last = max(first, join), min(last, today)
    start, days = _walk(ledger, snaps, first, last)
    flows = sum(r.flow for r in days)
    dep_in, _ = dep_wd([r for r in ledger if first <= day_utc3(r["created_at"]) <= last])
    end = days[-1].balance if days else start
    res = end - start - flows
    den = start + dep_in
    return dict(month=month, label=month_label(month) + (f" (from {fmt_day_short(join)})" if first == join and join.day != 1 else ""),
                flows=flows, result=res, ret_pct=(res / den * 100.0) if den > 1e-9 else None, end=end, dep_in=dep_in)


def compute_statement(name: str, ledger: List[dict], snaps: Dict[date, float], orders: List[dict], month: str,
                      now_utc: datetime, paper_now: bool) -> Statement:
    """name · ledger [{type, amount, shares_delta, nav_at_time, created_at UTC}] · snaps {UTC-3 day: NAV close} · orders [{closed_at UTC,
    pnl, pnl_percentage, is_paper}] (CLOSED, any mode) · month 'YYYY-MM' → Statement. PURE."""
    ledger = normalize_ledger(ledger)
    join = join_day(ledger)
    if join is None:
        raise StatementError("this investor has no deposit yet")
    today = day_utc3(now_utc)
    m_first, m_last = month_days(month)
    if m_last < join or m_first > today:
        raise StatementError(f"{month_label(month)} is outside this investor's history ({month_key(join)} → {month_key(today)})")
    first, last = max(m_first, join), min(m_last, today)
    preliminary = last >= today and month_key(today) == month
    start_bal, days = _walk(ledger, snaps, first, last)
    dep, wd = dep_wd([r for r in ledger if first <= day_utc3(r["created_at"]) <= last])
    end_bal = days[-1].balance
    result = end_bal - start_bal - (dep - wd)
    den = start_bal + dep
    upto = [r for r in ledger if day_utc3(r["created_at"]) <= last]
    tot_dep, tot_wd = dep_wd(upto)
    tot_profit = end_bal - (tot_dep - tot_wd)

    rets = [r for r in days if r.ret is not None]
    dcr = None
    if days and rets:
        g = 1.0
        for r in rets:
            g *= (1.0 + r.ret)
        dcr = (g ** (1.0 / len(days)) - 1.0) * 100.0 if g > 0 else -100.0
    idx = peak = 1.0
    peak_i, dd, dd_lo, dd_hi = -1, 0.0, None, None
    for i, r in enumerate(days):
        if r.ret is not None:
            idx *= (1.0 + r.ret)
        if idx > peak:
            peak, peak_i = idx, i
        if idx / peak - 1.0 < dd:
            dd, dd_lo, dd_hi = idx / peak - 1.0, peak_i, i
    dd_usd = sum(r.pnl for r in days[dd_lo + 1:dd_hi + 1]) if dd_hi is not None else 0.0
    best = max(rets, key=lambda r: r.ret) if rets else None
    worst = min(rets, key=lambda r: r.ret) if rets else None

    lo, hi = utc_bounds(first, last)
    lo = max(lo, min(r["created_at"] for r in ledger if signed_flow(r) > 0))      # trades before the investor's money arrived: out
    ts = [o for o in orders if o.get("closed_at") is not None and lo <= o["closed_at"] < hi]
    n = len(ts)
    paper = [o for o in ts if o.get("is_paper")]
    simulated = bool(paper) or (paper_now and (n == 0 or preliminary))
    mixed_until = max(o["closed_at"] for o in paper) if (paper and len(paper) < n) else None   # a paper → live switch inside the month

    months = [m for m in statement_months(ledger, now_utc) if m <= month]
    rows = []
    for m in months:
        try:
            rows.append(_month_row(ledger, snaps, m, join, today))
        except StatementError:          # a month before the NAV history (epoch reset) cannot be valued: left out, never fatal
            continue
    if len(rows) > MAX_MONTH_ROWS:
        old, rows = rows[:len(rows) - MAX_MONTH_ROWS + 1], rows[len(rows) - MAX_MONTH_ROWS + 1:]
        o_dep = sum(r["dep_in"] for r in old)
        o_res = sum(r["result"] for r in old)
        rows.insert(0, dict(month=old[0]["month"], label=f"{month_label(old[0]['month'])} – {month_label(old[-1]['month'])}",
                            flows=sum(r["flows"] for r in old), result=o_res, ret_pct=None, end=old[-1]["end"], dep_in=o_dep))
    total = dict(flows=tot_dep - tot_wd, result=tot_profit, ret_pct=(tot_profit / tot_dep * 100.0) if tot_dep > 1e-9 else None,
                 end=end_bal)
    return Statement(
        name=name, month=month, first=first, last=last, join=join, preliminary=preliminary, simulated=simulated,
        generated_at=now_utc, start_balance=start_bal, end_balance=end_bal, deposits=dep, withdrawals=wd, result=result,
        ret_pct=(result / den * 100.0) if den > 1e-9 else None, total_deposited=tot_dep, total_withdrawn=tot_wd,
        total_profit=tot_profit, total_profit_pct=(tot_profit / tot_dep * 100.0) if tot_dep > 1e-9 else None, days=days,
        mixed_until=mixed_until, trades=n, win_rate=(sum(1 for o in ts if float(o.get("pnl") or 0.0) > 0) / n * 100.0) if n else None,
        avg_pnl_pct=(sum(float(o.get("pnl_percentage") or 0.0) for o in ts) / n) if n else None, dcr_pct=dcr,
        max_dd_pct=dd * 100.0, max_dd_usd=dd_usd, best=best, worst=worst, positive_days=sum(1 for r in rets if r.ret > 0),
        month_rows=rows if len(rows) >= 2 else [], month_total=total if len(rows) >= 2 else None)


# ── text (pure) ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
MINUS = "−"


def money(x: float, sign: bool = False, cents: bool = True) -> str:
    v = round(x, 2 if cents else 0)
    if abs(v) < 0.005:
        v = 0.0
    s = f"${abs(v):,.{2 if cents else 0}f}"
    if v < 0:
        return MINUS + s
    return ("+" + s) if (sign and v > 0) else s


def pct(x: Optional[float], digits: int = 1, sign: bool = True) -> str:
    if x is None:
        return "—"
    v = round(x, digits)
    if abs(v) < 0.5 * 10 ** -digits:
        v = 0.0
    s = f"{abs(v):.{digits}f} %"
    if v < 0:
        return MINUS + s
    return ("+" + s) if (sign and v > 0) else s


def period_text(st: Statement) -> str:
    s = f"{fmt_day_short(st.first)} – {fmt_day(st.last)}" if st.first.year == st.last.year else f"{fmt_day(st.first)} – {fmt_day(st.last)}"
    return s + (" (to date)" if st.preliminary else "")


def summary_text(st: Statement) -> str:
    mn = month_name(st.month)
    s = f"Your balance is {money(st.end_balance)}: {money(st.result, sign=True)} ({pct(st.ret_pct)}) in {mn}"
    if month_key(st.join) != st.month:
        s += f", and {money(st.total_profit, sign=True)} ({pct(st.total_profit_pct)}) since your first deposit on {fmt_day(st.join)}"
    s += "."
    if st.trades:
        s += f" In {mn}, {st.trades:,} trades closed; {st.win_rate:.0f} % closed in profit."
    else:
        s += f" No trades closed in {mn}."
    return s


def disclaimer_text(st: Statement) -> str:
    if st.mixed_until is not None:
        return (f"Results up to {fmt_day(day_utc3(st.mixed_until))} are simulated (paper trading) and involve no real money; later "
                f"results are from live trading. ") + DISCLAIMER
    return (DISCLAIMER_PAPER if st.simulated else "") + DISCLAIMER


def banner_bits(st: Statement) -> List[str]:
    bits = []
    if st.mixed_until is not None:
        bits.append(f"Simulated (paper trading) up to {fmt_day_short(day_utc3(st.mixed_until))}, live after")
    elif st.simulated:
        bits.append("Simulated: paper trading, no real money")
    if st.preliminary:
        bits.append("Preliminary: month in progress")
    return bits


def file_name(name: str, month: str) -> str:
    safe = "".join(ch if (ch.isascii() and ch.isalnum()) else "_" for ch in name).strip("_") or "investor"   # ASCII: a header value
    return f"scalpars_statement_{safe}_{month}.pdf"


# ── PDF (reportlab) ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "fonts")
_FONTS = {"R": ("PlexSans", "IBMPlexSans-Regular.ttf"), "M": ("PlexSans-Medium", "IBMPlexSans-Medium.ttf"),
          "B": ("PlexSans-Bold", "IBMPlexSans-Bold.ttf")}
_registered = False

C_TEXT, C_SEC, C_MUTED = "#111827", "#5b6472", "#8a919e"
C_POS, C_NEG = "#15803d", "#b42318"
C_LINE, C_LINE_STRONG, C_TILE = "#e5e7eb", "#c9ced6", "#f3f4f6"
C_WARN_BG, C_WARN = "#fdf3d7", "#8a5a00"
C_HEAD_BG, C_HEAD_LINE, C_HEAD_TXT, C_HEAD_MUTED, C_HEAD_ACC = "#151922", "#262c3c", "#e6e9f0", "#8b94a7", "#3fcf90"
C_CHART = "#0F6E56"


def _fonts():
    global _registered
    if _registered:
        return
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for name, fn in _FONTS.values():
        pdfmetrics.registerFont(TTFont(name, os.path.join(FONT_DIR, fn)))
    _registered = True


def _f(k):
    return _FONTS[k][0]


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * p >= v:
            return m * p
    return 10 * p


def _k(v: float) -> str:
    if v >= 1000:
        s = f"{v / 1000:.1f}".rstrip("0").rstrip(".")
        return f"${s}k"
    return f"${v:,.0f}"


def render_pdf(st: Statement) -> bytes:
    """one A4 page. Raises RuntimeError if the content would run past the bottom margin (never a silent 2nd page)."""
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas
    _fonts()
    W, H = A4
    L, R = 40.0, W - 40.0
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, pageCompression=1)
    c.setTitle(f"SCALPARS statement · {month_label(st.month)} · {st.name}")
    c.setAuthor("SCALPARS")
    col = HexColor

    def text(x, y, s, font="R", size=9.5, color=C_TEXT, align="l", spacing=0.0):
        c.setFillColor(col(color))
        if spacing:                       # letter-spaced (the logo): saved / restored — a text object's char space leaks into the canvas
            c.saveState()
            t = c.beginText()
            t.setTextOrigin(x, y)
            t.setFont(_f(font), size)
            t.setCharSpace(spacing)
            t.textOut(s)
            c.drawText(t)
            c.restoreState()
            return
        c.setFont(_f(font), size)
        (c.drawRightString if align == "r" else c.drawCentredString if align == "c" else c.drawString)(x, y, s)

    def width(s, font="R", size=9.5):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        return stringWidth(s, _f(font), size)

    def hline(y, x0=L, x1=R, color=C_LINE, w=0.6):
        c.setStrokeColor(col(color))
        c.setLineWidth(w)
        c.line(x0, y, x1, y)

    def para(s, y, size, color, lead, font="R", x0=L, x1=R):
        for ln in simpleSplit(s, _f(font), size, x1 - x0):
            text(x0, y, ln, font, size, color)
            y -= lead
        return y

    # ① header band (the site's logo, colours and type)
    hb = 62
    c.setFillColor(col(C_HEAD_BG))
    c.rect(0, H - hb, W, hb, stroke=0, fill=1)
    hline(H - hb, 0, W, C_HEAD_LINE, 1)
    text(L, H - 30, "SCALP", "B", 17, C_HEAD_TXT, spacing=2.4)
    text(L + width("SCALP", "B", 17) + 5 * 2.4, H - 30, "ARS", "B", 17, C_HEAD_ACC, spacing=2.4)
    text(L, H - 46, "Neural Decision Engine", "R", 9, C_HEAD_MUTED)
    text(R, H - 30, f"Monthly statement · {month_name(st.month)} {st.month[:4]}", "M", 12, C_HEAD_TXT, "r")
    text(R, H - 46, f"Statement period: {period_text(st)}", "R", 9, C_HEAD_MUTED, "r")
    y = H - hb
    bits = banner_bits(st)
    if bits:
        c.setFillColor(col(C_WARN_BG))
        c.rect(0, y - 18, W, 18, stroke=0, fill=1)
        text(W / 2, y - 12.5, "  ·  ".join(bits), "M", 8.5, C_WARN, "c")
        y -= 18
    y -= 22

    # ② prepared for / investor since
    text(L, y, "Prepared for", "R", 8, C_SEC)
    text(W / 2, y, "Investor since", "R", 8, C_SEC)
    y -= 13
    nm = st.name
    while width(nm, "M", 10.5) > W / 2 - L - 12 and len(nm) > 4:
        nm = nm[:-2].rstrip() + "\u2026" if not nm.endswith("\u2026") else nm[:-2].rstrip() + "\u2026"
    text(L, y, nm, "M", 10.5)
    text(W / 2, y, fmt_day(st.join), "M", 10.5)
    y -= 10
    hline(y)
    y -= 15

    # ③ summary line
    y = para(summary_text(st), y, 10, C_TEXT, 13.5) - 4

    # ④ all-time tiles
    tw, th, gap = (R - L - 2 * 10) / 3, 42, 10
    tiles = [("Your balance", money(st.end_balance), C_TEXT, None),
             ("Total deposited", money(st.total_deposited), C_TEXT, None),
             ("Total profit", money(st.total_profit, sign=True, cents=False), C_POS if st.total_profit >= 0.005 else (C_NEG if st.total_profit <= -0.005 else C_TEXT),
              pct(st.total_profit_pct))]
    for i, (lab, val, colr, sub) in enumerate(tiles):
        x = L + i * (tw + gap)
        c.setFillColor(col(C_TILE))
        c.roundRect(x, y - th, tw, th, 5, stroke=0, fill=1)
        text(x + 10, y - 14, lab, "R", 8, C_SEC)
        text(x + 10, y - 33, val, "M", 15, colr)
        if sub:
            text(x + 14 + width(val, "M", 15), y - 33, sub, "R", 9, colr)
    y -= th + 20

    # ⑤ month tables
    mn = month_name(st.month)
    cw = (R - L - 24) / 2
    xa, xb = L, L + cw + 24

    def head(x, yy, s, x1):
        text(x, yy, s, "M", 10.5)
        hline(yy - 5, x, x1, C_LINE_STRONG, 0.7)

    def row(x, x1, yy, lab, val, vcol=C_TEXT, bold=False, sub=None, subcol=None):
        text(x, yy, lab, "M" if bold else "R", 9, C_TEXT if bold else C_SEC)
        if sub:
            text(x1, yy, sub, "R", 9, subcol, "r")
            text(x1 - width(sub, "R", 9) - 4, yy, val, "M" if bold else "R", 9, vcol, "r")
        else:
            text(x1, yy, val, "M" if bold else "R", 9, vcol, "r")

    head(xa, y, f"{mn}: balance summary", xa + cw)
    head(xb, y, f"{mn}: trading performance", xb + cw)
    ya = yb = y - 18
    lead = 13
    rcol = C_POS if st.result >= 0.005 else (C_NEG if st.result <= -0.005 else C_TEXT)
    row(xa, xa + cw, ya, f"Starting balance, {fmt_day_short(st.first)}", money(st.start_balance)); ya -= lead
    row(xa, xa + cw, ya, "Deposits", money(st.deposits)); ya -= lead
    row(xa, xa + cw, ya, "Withdrawals", money(st.withdrawals)); ya -= lead
    row(xa, xa + cw, ya, "Trading result", money(st.result, sign=True), rcol); ya -= 6
    hline(ya, xa, xa + cw)
    ya -= 11
    row(xa, xa + cw, ya, f"Ending balance, {fmt_day_short(st.last)}", money(st.end_balance), bold=True)

    nd = len(st.days)
    perf = [("Trades (per day)", f"{st.trades:,} ({st.trades / nd:.1f})" if nd else "—", None, None),
            ("Win rate", pct(st.win_rate, 0, sign=False), None, None),
            ("Average P&L per trade", pct(st.avg_pnl_pct, 2), None, None),
            ("Daily compound return", pct(st.dcr_pct, 2), None, None),
            ("Largest drop from peak", pct(st.max_dd_pct), f"({money(-abs(st.max_dd_usd), cents=False)})" if st.max_dd_pct < -0.05 else None, C_NEG)]
    if st.best is not None:
        perf.append((f"Best day · {fmt_day_short(st.best.day)}", pct(st.best.ret * 100), f"({money(st.best.pnl, sign=True, cents=False)})",
                     C_POS if st.best.pnl >= 0.5 else (C_NEG if st.best.pnl <= -0.5 else C_SEC)))
        perf.append((f"Worst day · {fmt_day_short(st.worst.day)}", pct(st.worst.ret * 100), f"({money(st.worst.pnl, sign=True, cents=False)})",
                     C_POS if st.worst.pnl >= 0.5 else (C_NEG if st.worst.pnl <= -0.5 else C_SEC)))
    perf.append(("Positive days", f"{st.positive_days} of {nd}", None, None))
    for lab, val, sub, subcol in perf:
        row(xb, xb + cw, yb, lab, val, sub=sub, subcol=subcol)
        yb -= lead
    y = min(ya, yb + lead) - 14
    if nd < SHORT_PERIOD_DAYS:
        text(L, y, f"Short period ({nd} day{'s' if nd != 1 else ''}): not representative of long-term performance.", "R", 8.5, C_WARN)
        y -= 16

    # ⑥ since you joined
    if st.month_rows:
        y -= 4
        head(L, y, "Since you joined", R)
        y -= 17
        cols = [(L, "Month", "l"), (R - 290, "Deposits / withdrawals", "r"), (R - 190, "Result", "r"), (R - 110, "Return", "r"),
                (R, "Ending balance", "r")]
        for x, lab, a in cols:
            text(x, y, lab, "R", 8.5, C_SEC, a)
        y -= 5
        hline(y)
        y -= 12

        def mrow(yy, r, bold=False):
            f = "M" if bold else "R"
            rc = C_POS if r["result"] >= 0.005 else (C_NEG if r["result"] <= -0.005 else C_TEXT)
            text(L, yy, r.get("label", "Total"), f, 9)
            text(cols[1][0], yy, money(r["flows"], sign=True), f, 9, align="r")
            text(cols[2][0], yy, money(r["result"], sign=True), f, 9, rc, "r")
            text(cols[3][0], yy, pct(r["ret_pct"]), f, 9, rc, "r")
            text(cols[4][0], yy, money(r["end"]), f, 9, align="r")
        for r in st.month_rows:
            mrow(y, r)
            y -= 12.5
        y += 4
        hline(y)
        y -= 12
        mrow(y, dict(st.month_total, label="Total"), bold=True)
        y -= 20

    # ⑦ chart: the balance each day vs the money put in
    text(L, y, f"Your balance in {mn}", "M", 10.5)
    y -= 8
    ch_h, x0, x1 = 112.0, L + 34, R - 6
    yb0 = y - ch_h + 14
    vals = [st.start_balance + st.days[0].flow] + [d.balance for d in st.days]      # a join-day deposit starts the line at its amount
    deps = [max(st.days[0].deposited_net, 0.0)] + [max(d.deposited_net, 0.0) for d in st.days]   # never drawn below $0
    top = _nice_max(max(vals + deps) * 1.05)

    def py(v):
        return yb0 + (ch_h - 22) * (v / top)
    n = len(vals)

    def px(i):
        return x0 + (x1 - x0) * (i / (n - 1 if n > 1 else 1))
    for g in (0.0, 0.5, 1.0):
        hline(py(top * g), x0, x1, C_LINE_STRONG if g == 0 else C_LINE, 0.5)
        text(x0 - 6, py(top * g) - 3, _k(top * g), "R", 7.5, C_SEC, "r")
    c.setStrokeColor(col(C_MUTED))
    c.setLineWidth(1)
    c.setDash(4, 3)
    p = c.beginPath()
    p.moveTo(px(0), py(deps[0]))
    for i in range(1, n):
        p.lineTo(px(i), py(deps[i - 1]))
        p.lineTo(px(i), py(deps[i]))
    c.drawPath(p, stroke=1, fill=0)
    c.setDash()
    c.setStrokeColor(col(C_CHART))
    c.setLineWidth(1.8)
    p = c.beginPath()
    p.moveTo(px(0), py(vals[0]))
    for i in range(1, n):
        p.lineTo(px(i), py(vals[i]))
    c.drawPath(p, stroke=1, fill=0)
    c.setFillColor(col(C_CHART))
    c.circle(px(n - 1), py(vals[-1]), 2.6, stroke=0, fill=1)
    labelled = 0
    for i, d in enumerate(st.days, start=1):
        if abs(d.flow) > 0.005 and labelled < 3 and i > 1 and d.day != st.join:
            lab = f"{'Deposit' if d.flow > 0 else 'Withdrawal'} {money(d.flow, sign=True, cents=False)}"
            right = px(i) + 3 + width(lab, "R", 7) > x1
            text(px(i) - 3 if right else px(i) + 3, py(max(vals[i], deps[i])) + 5, lab, "R", 7, C_SEC, "r" if right else "l")
            labelled += 1
    ticks = sorted({0, n // 2, n - 1})
    for i in ticks:
        dd = st.first + timedelta(days=max(i - 1, 0)) if i else st.first
        text(px(i), yb0 - 11, fmt_day_short(dd), "R", 7.5, C_SEC, "l" if i == 0 else ("r" if i == n - 1 else "c"))
    y = yb0 - 24
    lx = L
    c.setStrokeColor(col(C_CHART)); c.setLineWidth(2); c.line(lx, y + 3, lx + 14, y + 3)
    text(lx + 18, y, "Your balance", "R", 8, C_SEC)
    lx += 18 + width("Your balance", "R", 8) + 16
    c.setStrokeColor(col(C_MUTED)); c.setLineWidth(1); c.setDash(4, 3); c.line(lx, y + 3, lx + 14, y + 3); c.setDash()
    text(lx + 18, y, "Total deposited" if st.total_withdrawn < 0.005 else "Deposits minus withdrawals", "R", 8, C_SEC)
    y -= 32                                                                        # one blank line between the chart legend and About (operator 11-06)

    # ⑧ about + disclaimer + generated stamp
    text(L, y, "About SCALPARS", "M", 10.5)
    y -= 13
    y = para(ABOUT, y, 8.5, C_SEC, 11.5)
    y -= 2
    hline(y)
    y -= 11
    y = para(disclaimer_text(st), y, 7.3, C_MUTED, 9.4)
    gen = st.generated_at.replace(tzinfo=timezone.utc).astimezone(UTC3)
    text(R, y - 3, f"Generated on {fmt_day(gen.date())} {gen:%H:%M} UTC-3", "R", 7, C_MUTED, "r")
    if y - 3 < 22:
        raise RuntimeError(f"statement overflows one A4 page (bottom at {y - 3:.0f} pt)")
    c.showPage()
    c.save()
    return buf.getvalue()
