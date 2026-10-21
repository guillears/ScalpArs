"""🧾 Auto-record Binance-app trades (DECISION_LOG 329, operator 2026-10-20).

WHY: the operator trades directly in the Binance app (Oct-20: BTW, IOST ×6+, COMP). Those trades sit in the Binance balance and the live
P&L card but in NO table / calendar / win rate. This module books every COMPLETED app trade as a CLOSED MANUAL row with Binance's REAL
numbers (its fills, its realized P&L, its commissions), from the first live trade on — including periods when the bot was stopped.

DATA SOURCE (server side, a DEDICATED authenticated ccxt client with its own rate limiter — review I5: never the trading client's queue —
binance_service.fetch_income_page / fetch_user_trades_page / get_net_positions / get_symbol_leverages / fetch_bnb_1m_closes /
fetch_bnb_price_raw): /fapi/v1/income (all types, ≤ 7-day windows, lossless overlap paging deduped on (incomeType, tranId, symbol, asset))
says which symbols traded when and feeds the reconcile sums; /fapi/v1/userTrades per symbol per ≤ 7-day window (a full page continues with
fromId) gives the fills. Per symbol and pass window the fills must reproduce the income (every REALIZED_PNL / COMMISSION tradeId read,
Σ realizedPnl and Σ commission to $0.01) or the symbol is skipped and retried (review I4: a missing fill fakes a flat). A weight budget
(APP_WEIGHT_PER_MIN) is spent before every call, X-MBX-USED-WEIGHT-1M is read after it, an active ban skips the pass, a 418 / 429 / ban
answer STOPS the pass (cursor not advanced → the next cycle resumes; rows already written stay — idempotent).

EPISODES: per symbol the fills are replayed in time order as one SIGNED net position (one-way mode). The replay is ANCHORED on the live
position (positionRisk read at the pass start): start net = live net − Σ signed fills; a non-zero start net extends the read backwards one
7-day window at a time (never before the era start). An episode runs flat → flat; a fill that crosses zero is split (its closing part ends
the episode — it carries the fill's whole realizedPnl — and the rest opens the next one; commission split pro rata). A position still open
at the end is NOT recorded (the 📥 import / untracked flow covers open ones); an episode already open at the era start is skipped. A symbol
with a fill in the last APP_BUSY_S before the position read is skipped this pass (the anchor could miss that fill) — retried next pass.

ATTRIBUTION vs DB rows (live, any status, same pair, not our own app rows) — CONSERVATIVE: when the numbers cannot be explained exactly,
nothing is written and [APP_TRADE_AMBIGUOUS] is logged with the episode (the operator must never see invented numbers).
  * membership: a row whose Binance order ids (binance_order_id + its Transactions' ids — in production the ENTRY id and ➕ ADD ids only:
    a CLOSE Transaction carries the entry id or None, the bot's close order id is never stored) appear among the symbol's fills belongs to
    the episode holding them (exact); a row without such ids belongs when its window [opened_at − 90 s, (closed_at or now) + 90 s] overlaps
    the episode. A 📥 IMPORTED row (DL 328, import_source BINANCE_APP) adopted the WHOLE position at the import → its window starts at the
    episode start. A member's window is widened to its own fills (matched by order id), so a booking time minutes off its fill (maker
    entry, slow close booking) never makes the bot's own fill look like an app fill.
  * no member → a pure APP trade (close_reason APP_CLOSE).
  * members: any OPEN → deferred (re-read every pass; > APP_DEFER_MAX_DAYS → ambiguous). Opposite direction, or a non-imported row opened
    before the episode (spans two episodes) → ambiguous. Every fill inside some member window AND |net| ≤ the members' concurrent
    quantity × 1.01 at every fill → fully the bot's → nothing written — UNLESS a non-resting-TP fill took |net| below the bot's quantity
    > 30 s before the close of every member open then (review I3: the app sold bot coins) → ambiguous. A member booked on an unverified
    exit (notes RTP_FILL_UNVERIFIED / RTP_SETTLE_BLOCKED) or an EXTERNAL_CLOSE whose exit matches no closing fill (a fallback price) →
    ambiguous for a share (review I2). Otherwise ONE "app share" line: qty = max over fills of
    (|net| − members' concurrent qty); P&L = episode Σ(realizedPnl − commission) − Σ members' booked pnl; entry = VWAP of the opening fills
    NOT attributable to a member (order id not a member's; for a member without a known entry id, fills near its open — for an imported row
    every fill before the import — are its own); exit = VWAP of closing fills not in a member's order ids when they cover the share, else
    of ALL closing fills (APPROXIMATION: the bot's own exit fills can carry unknown ids — the residual sweep's — so the display exit price
    may blend the bot's; the P&L does not depend on it). close_reason APP_ADD_SOLD_BY_BOT when the final reducing fill came within 5 s of
    a member's closed_at (the bot's close / residual sweep sold it, DL 322), else APP_CLOSE. App opening fills that do not cover the share
    → ambiguous.

ROW (Order, CLOSED, is_paper False): entry_strategy MANUAL, confidence STRONG_BUY, import_source BINANCE_APP_HISTORY, app_trade_key
"{symbol}:{first fill trade id}" (unique), direction = the first fill's, entry / exit = VWAP of opening / closing fills, quantity = max |net|
(a scale-in / scale-out episode is ONE row at its peak size), opened_at / closed_at = first / last fill, pnl = Σ realizedPnl − Σ commission
in USDT (BNB commission × the BNB 1m close at the fill minute; unavailable → the current BNB price and app_fee_estimated = True),
entry_fee / exit_fee = the opening / closing fills' commission, funding_fee_usd = Σ FUNDING_FEE over [opened_at, closed_at] — NOT in pnl
(the bot's convention), pnl_percentage = pnl / (entry × qty) × 100 (net of fees on the entry notional, like every row), leverage = the
symbol's CURRENT setting (symbolConfig, approximate — noted; unreadable → 1), investment = notional / leverage, exit / entry order type
MAKER / TAKER / MIXED from the fills' maker flag, peak / trough 0 (no intratrade path observed).
"""
import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

IMPORT_SOURCE_APP_HISTORY = "BINANCE_APP_HISTORY"   # Order.import_source of an auto-recorded app trade (DL 329)
IMPORT_SOURCE_IMPORTED = "BINANCE_APP"              # 📥 DL 328 (services.position_import.IMPORT_SOURCE_BINANCE_APP)
ROW_PAD_MS = 90_000                                 # a DB row's window: [opened_at − 90 s, (closed_at or now) + 90 s]
QTY_TOL = 0.01                                      # members explain |net| when |net| ≤ their quantity × 1.01
SOLD_BY_BOT_MS = 5_000                              # the final reducing fill within 5 s of a member's closed_at → the bot sold it
WINDOW_MS = 7 * 86_400_000                          # Binance: income / userTrades windows ≤ 7 days
DAY_MS = 86_400_000
APP_BUSY_S = 180                                    # a fill this close to the position read → the symbol is retried next pass
APP_DEFER_MAX_DAYS = 3                              # a member row still OPEN this long after Binance went flat → ambiguous
APP_MAX_EXTEND = 13                                 # backward 7-day extensions per symbol (≈ 3 months, Binance's income horizon)
APP_WEIGHT_PER_MIN = 600                            # this job's own budget (Binance's IP limit is 2,400 / min)
APP_HEADER_PAUSE = 1800                             # X-MBX-USED-WEIGHT-1M above this → pause 60 s
APP_INTERVAL_S = 300                                # incremental pass every 5 min
APP_FIRST_DELAY_S = 90                              # first pass ~90 s after startup (after the first reconcile)
W_INCOME, W_TRADES, W_POS, W_SYMCFG, W_KLINES = 30, 5, 5, 5, 5
PAGE = 1000
EARLY_REDUCE_MS = 30_000                            # (review I3) a bot-position reduction this long before the bot's close = the app's
RTP_PX_TOL = 0.0005                                 # a maker fill within 0.05 % of a member's resting-TP limit = that TP
EXT_PX_TOL = 0.001                                  # (review I2) an EXTERNAL_CLOSE row's exit must match a closing fill within 0.1 %
SHARE_XCHECK_USD, SHARE_XCHECK_SHARE = 0.50, 0.02   # (review I2) share P&L: fills vs episode − booked, max($0.50, 2 %)
FILL_CHECK_USD = 0.01                               # (review I4) fills vs income per symbol / pass window
FILL_PAD_MS = 120_000                               # (round-2 F) fills read 2 min before lo: an income row just inside lo finds its fill
EXACT_USD = 0.01                                    # (review I6) a 📱 APP row vs its income to the cent
STATE_AMBIG_CAP = 1000


RETRY_MAX_DAYS = 14                                 # a symbol retried this long (since it was FIRST put on the list) is dropped → ambiguous


def retry_point(v) -> int:
    """A retry entry's re-read point (round-3 3: {point, first_seen_ms}; a bare int = the round-2 format)."""
    return int(v.get("point")) if isinstance(v, dict) else int(v)


def retry_add(state: dict, sym: str, point: int, now_ms: int, first_seen_ms=None):
    """Put sym on the retry list from `point` (the earliest point wins); first_seen_ms is kept from its FIRST entry — the 14-day drop ages
    on that, never on the point (a backfill point is older than 14 days from the start). first_seen_ms (final gate): the age of an entry
    popped earlier in this pass, so a symbol that keeps failing never restarts its 14-day clock."""
    cur = state["retry"].get(sym)
    if cur is None:
        state["retry"][sym] = {"point": int(point), "first_seen_ms": int(first_seen_ms if first_seen_ms is not None else now_ms)}
    else:
        first = int(cur.get("first_seen_ms", now_ms)) if isinstance(cur, dict) else int(now_ms)
        state["retry"][sym] = {"point": min(retry_point(cur), int(point)), "first_seen_ms": first}


class AppTradeStop(Exception):
    """418 / 429 / ban (or an active ban) — the pass stops; the cursor does not move; the next cycle resumes."""


# ── pure: fills → episodes ──────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Fill:
    trade_id: int
    order_id: str
    t: int                  # ms
    side: str               # BUY / SELL
    price: float
    qty: float
    rpnl: float
    comm: float             # positive = paid (userTrades convention)
    comm_asset: str
    maker: bool

    @property
    def signed(self) -> float:
        return self.qty if self.side == "BUY" else -self.qty


def income_key(r: dict) -> tuple:
    """(329 review C1) an income row's identity: tranId is unique only WITHIN an income type (a TRANSFER and a COMMISSION can share one)."""
    return (str(r.get("incomeType") or ""), str(r.get("tranId") if r.get("tranId") is not None else (r.get("time"), r.get("income"))),
            str(r.get("symbol") or ""), str(r.get("asset") or ""))


def parse_fill(r: dict) -> Fill:
    """A raw /fapi/v1/userTrades row → Fill (raises on a malformed row — the caller treats the symbol as unreadable)."""
    side = str(r.get("side") or "").upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"fill side {side!r}")
    q, p = float(r["qty"]), float(r["price"])
    if not (q > 0 and p > 0 and math.isfinite(q) and math.isfinite(p)):
        raise ValueError(f"fill qty/price {q}/{p}")
    return Fill(trade_id=int(r["id"]), order_id=str(r.get("orderId")), t=int(r["time"]), side=side, price=p, qty=q,
                rpnl=float(r.get("realizedPnl") or 0.0), comm=float(r.get("commission") or 0.0),
                comm_asset=str(r.get("commissionAsset") or "USDT").upper(), maker=bool(r.get("maker")))


@dataclass
class Leg:
    fill: Fill
    qty: float              # the part of the fill in this episode
    kind: str               # 'open' (|net| grows) / 'close' (|net| shrinks)
    frac: float             # qty / fill.qty (commission share)
    net_after: float        # |net| after this leg

    @property
    def rpnl(self) -> float:
        return self.fill.rpnl if self.kind == "close" else 0.0   # Binance books a fill's realizedPnl on its closing part


@dataclass
class Episode:
    symbol: str
    direction: str
    legs: List[Leg] = field(default_factory=list)
    partial: bool = False   # started before the replay's start (never recorded)

    @property
    def start_ms(self) -> int:
        return self.legs[0].fill.t

    @property
    def end_ms(self) -> int:
        return self.legs[-1].fill.t

    @property
    def first_trade_id(self) -> int:
        return self.legs[0].fill.trade_id

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.first_trade_id}"

    @property
    def max_abs_net(self) -> float:
        return max((l.net_after for l in self.legs), default=0.0)

    def opens(self) -> List[Leg]:
        return [l for l in self.legs if l.kind == "open"]

    def closes(self) -> List[Leg]:
        return [l for l in self.legs if l.kind == "close"]

    @property
    def n_fills(self) -> int:
        return len({l.fill.trade_id for l in self.legs})

    def order_ids(self) -> set:
        return {l.fill.order_id for l in self.legs}

    def describe(self) -> str:
        return " ".join(f"{l.fill.side}{'+' if l.kind == 'open' else '-'}{l.qty:g}@{l.fill.price:g}[{l.fill.order_id}]" for l in self.legs)


def qty_eps(fills: Iterable[Fill]) -> float:
    m = max((f.qty for f in fills), default=0.0)
    return max(1e-12, 1e-7 * m)


def build_episodes(symbol: str, fills: List[Fill], start_net: float = 0.0, eps: Optional[float] = None):
    """Replay fills (sorted by time, trade id) from start_net → (complete episodes, the open tail or None, n leading partials skipped).
    Flat → flat; a zero-crossing fill is split; an episode that was already open at the start is marked partial and dropped."""
    fills = sorted(fills, key=lambda f: (f.t, f.trade_id))
    eps = qty_eps(fills) if eps is None else eps
    net = 0.0 if abs(start_net) <= eps else float(start_net)
    cur = Episode(symbol, "LONG" if net > 0 else "SHORT", partial=True) if net else None
    done, n_lead = [], 0
    for f in fills:
        rem = f.signed
        if net and (rem > 0) != (net > 0):                     # reducing
            cq = min(abs(rem), abs(net))
            net += math.copysign(cq, rem)
            rem -= math.copysign(cq, rem)
            if abs(net) <= eps:
                net = 0.0
            cur.legs.append(Leg(f, cq, "close", cq / f.qty, abs(net)))
            if net == 0.0:
                if cur.partial:
                    n_lead += 1
                else:
                    done.append(cur)
                cur = None
            if abs(rem) <= eps:
                rem = 0.0
        if rem:                                                # opening / adding
            if cur is None:
                cur = Episode(symbol, "LONG" if rem > 0 else "SHORT")
            net += rem
            cur.legs.append(Leg(f, abs(rem), "open", abs(rem) / f.qty, abs(net)))
    return done, cur, n_lead


def fill_income_bad_windows(sym: str, fills: List[Fill], income: Iterable[dict], lo: int, hi: int, rate,
                            bucket_ms: int = DAY_MS) -> List[Tuple[int, int, str]]:
    """(329 review I4, round-2 F) the UTC-day windows of [lo, hi) where the fills read do NOT reproduce the symbol's income → [(start, end,
    why)], [] when all agree. Matched on tradeId, so a fill and its income row on opposite sides of a window edge (or of lo — the fills
    are read with FILL_PAD_MS before lo) still pair up: every REALIZED_PNL / COMMISSION row's tradeId must be a fill read; Σ realizedPnl and
    Σ commission (USDT) of the rows vs of THOSE fills within FILL_CHECK_USD; a fill with P&L or commission at least FILL_PAD_MS after lo
    must have an income row (income lagging). An income row without a tradeId cannot be verified → that window is bad (conservative)."""
    inc = [r for r in income if r.get("symbol") == sym and r.get("incomeType") in ("REALIZED_PNL", "COMMISSION")
           and lo <= int(r.get("time") or 0) < hi]
    by_id = {f.trade_id: f for f in fills}
    all_ids = set()
    for r in inc:
        try:
            all_ids.add(int(r.get("tradeId")))
        except (TypeError, ValueError):
            pass

    def conv(x, asset, t):
        rr = rate(str(asset or "USDT").upper(), int(t))
        if rr is None:
            raise ValueError(f"commission asset {asset} cannot be converted")
        return x * rr[0]
    out = []
    b = lo
    while b < hi:
        e = min(hi, b - (b % bucket_ms) + bucket_ms)
        rows = [r for r in inc if b <= int(r.get("time") or 0) < e]
        why = None
        if any(r.get("tradeId") in (None, "") for r in rows):
            why = "income row(s) without a tradeId (cannot be matched to fills)"
        else:
            tids = {int(r.get("tradeId")) for r in rows}
            missing = sorted(tids - set(by_id))
            if missing:
                why = f"{len(missing)} fill(s) in the income but not in the fills read (trade ids {', '.join(map(str, missing[:5]))})"
            else:
                rp_i = sum(float(r.get("income") or 0) for r in rows if r.get("incomeType") == "REALIZED_PNL")
                c_i = -sum(conv(float(r.get("income") or 0), r.get("asset"), r.get("time") or 0) for r in rows if r.get("incomeType") == "COMMISSION")
                fs = [by_id[t] for t in tids]
                rp_f = sum(f.rpnl for f in fs)
                c_f = sum(conv(f.comm, f.comm_asset, f.t) for f in fs if f.comm)
                if abs(rp_i - rp_f) > FILL_CHECK_USD:
                    why = f"realized P&L: fills {rp_f:+.4f} vs income {rp_i:+.4f}"
                elif abs(c_i - c_f) > FILL_CHECK_USD:
                    why = f"commission: fills {c_f:.4f} vs income {c_i:.4f} USDT"
        if why is None:
            lag = [f.trade_id for f in fills if max(b, lo + FILL_PAD_MS) <= f.t < e and (f.rpnl or f.comm) and f.trade_id not in all_ids]
            if lag:
                why = f"{len(lag)} fill(s) with no income row yet (trade ids {', '.join(map(str, lag[:5]))})"
        if why:
            out.append((b, e, why))
        b = e
    return out


def close_price_candidates(ep: "Episode") -> List[float]:
    """(round-2 E) prices a real bot exit could have been booked at: every closing fill, each closing ORDER's VWAP (a multi-fill market
    exit on a thin pair) and the VWAP of all closing fills."""
    cl = ep.closes()
    out = [l.fill.price for l in cl]
    by_order: Dict[str, List[Leg]] = {}
    for l in cl:
        by_order.setdefault(l.fill.order_id, []).append(l)
    out += [vwap(v) for v in by_order.values()]
    if cl:
        out.append(vwap(cl))
    return [p for p in out if p]


def episode_income_pnl(ep: "Episode", income_rows: Iterable[dict], rate) -> float:
    """(round-2 A) Σ REALIZED_PNL + Σ COMMISSION (USDT) of the episode's OWN fills — income rows whose tradeId is one of its fills — with a
    split (zero-crossing) fill counted exactly as the row counts it: its realizedPnl only where the episode holds the CLOSING part, its
    commission × the episode's share of the fill (Σ leg frac). Rows of another episode at the boundary millisecond never leak in."""
    frac: Dict[int, float] = {}
    closes: set = set()
    for l in ep.legs:
        frac[l.fill.trade_id] = frac.get(l.fill.trade_id, 0.0) + l.frac
        if l.kind == "close":
            closes.add(l.fill.trade_id)
    tot = 0.0
    for r in income_rows:
        tid = r.get("tradeId")
        if tid in (None, ""):
            continue
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        if tid not in frac:
            continue
        typ, x, t = r.get("incomeType"), float(r.get("income") or 0), int(r.get("time") or 0)
        asset = str(r.get("asset") or "USDT").upper()
        if typ not in ("REALIZED_PNL", "COMMISSION"):
            continue
        if asset != "USDT":
            rr = rate(asset, t)
            if rr is None:
                raise ValueError(f"income asset {asset} cannot be converted")
            x *= rr[0]
        if typ == "REALIZED_PNL":
            tot += x if tid in closes else 0.0
        else:
            tot += x * frac[tid]
    return tot


def vwap(legs: Iterable[Leg]) -> Optional[float]:
    legs = list(legs)
    q = sum(l.qty for l in legs)
    return (sum(l.qty * l.fill.price for l in legs) / q) if q > 0 else None


def order_type(legs: Iterable[Leg]) -> Optional[str]:
    flags = {bool(l.fill.maker) for l in legs}
    if not flags:
        return None
    return "MIXED" if len(flags) > 1 else ("MAKER" if True in flags else "TAKER")


def comm_usdt(legs: Iterable[Leg], rate: Callable[[str, int], Optional[Tuple[float, bool]]]) -> Tuple[float, bool]:
    """Σ commission of the legs in USDT (pro rata on split fills) and whether any conversion was an estimate. rate(asset, t_ms) →
    (USDT per unit, estimated) or None (unknown asset → ValueError: the caller records nothing)."""
    tot, est = 0.0, False
    for l in legs:
        c = l.fill.comm * l.frac
        if not c:
            continue
        r = rate(l.fill.comm_asset, l.fill.t)
        if r is None:
            raise ValueError(f"commission asset {l.fill.comm_asset} cannot be converted to USDT")
        tot += c * r[0]
        est = est or bool(r[1])
    return tot, est


# ── pure: attribution ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class RowView:
    id: int
    direction: str
    status: str
    qty: float
    open_ms: int
    close_ms: Optional[int]
    pnl: Optional[float]
    imported: bool = False
    order_ids: frozenset = frozenset()
    entry_id_known: bool = False
    funding: Optional[float] = None
    close_reason: str = ""
    notes: str = ""
    exit_price: Optional[float] = None
    rtp_price: Optional[float] = None


def _row_hi(r: RowView, now_ms: int) -> int:
    return (r.close_ms if r.close_ms is not None else now_ms) + ROW_PAD_MS


def episode_members(ep: Episode, rows: List[RowView], now_ms: int, symbol_order_ids: set) -> List[RowView]:
    out = []
    ep_ids = ep.order_ids()
    for r in rows:
        if r.order_ids and (r.order_ids & symbol_order_ids):
            if r.order_ids & ep_ids:                           # exact: the row's own orders are in this episode
                out.append(r)
            continue
        if r.open_ms - ROW_PAD_MS <= ep.end_ms and _row_hi(r, now_ms) >= ep.start_ms:
            out.append(r)
    return out


def _own_times(r: RowView, ep: Episode) -> List[int]:
    return [l.fill.t for l in ep.legs if l.fill.order_id in r.order_ids]


def _member_lo(r: RowView, ep: Episode) -> int:
    """A member's window start: the episode start for a 📥 imported row; else opened_at − 90 s, widened to its own fills (a booking time
    later than the fill — e.g. a maker entry — never pushes the row's own fill outside its window)."""
    if r.imported:
        return ep.start_ms
    own = _own_times(r, ep)
    return min([r.open_ms - ROW_PAD_MS] + own)


def _member_hi(r: RowView, ep: Episode, now_ms: int) -> int:
    own = _own_times(r, ep)
    return max([_row_hi(r, now_ms)] + own)


@dataclass
class Verdict:
    kind: str               # APP / SHARE / BOT / DEFER / AMBIGUOUS
    reason: str = ""
    members: List[RowView] = field(default_factory=list)
    share_qty: float = 0.0
    entry_legs: List[Leg] = field(default_factory=list)
    exit_legs: List[Leg] = field(default_factory=list)
    exit_scale: float = 1.0          # share of the exit legs' commission charged to the app line (fallback = all closing fills)
    exit_approx: bool = False
    close_reason: str = "APP_CLOSE"


def attribute_episode(ep: Episode, rows: List[RowView], now_ms: int, symbol_order_ids: set) -> Verdict:
    members = episode_members(ep, rows, now_ms, symbol_order_ids)
    if not members:
        return Verdict("APP")
    if any(r.direction != ep.direction for r in members):
        return Verdict("AMBIGUOUS", "a bot row on the pair has the opposite direction", members)
    if any((not r.imported) and not (r.order_ids & ep.order_ids()) and r.open_ms + ROW_PAD_MS < ep.start_ms for r in members):
        return Verdict("AMBIGUOUS", "a bot row opened before this Binance episode began (it spans two episodes)", members)
    if any(str(r.status).upper() != "CLOSED" for r in members):
        return Verdict("DEFER", "a bot row in this episode is still OPEN", members)
    if any(r.pnl is None for r in members):
        return Verdict("AMBIGUOUS", "a bot row in this episode has no booked pnl", members)
    eps = qty_eps(l.fill for l in ep.legs)

    def concurrent(t):
        return sum(r.qty for r in members if _member_lo(r, ep) <= t <= _member_hi(r, ep, now_ms))
    outside, over, share = False, False, 0.0
    for l in ep.legs:
        c = concurrent(l.fill.t)
        outside = outside or c <= 0
        over = over or l.net_after > c * (1 + QTY_TOL) + eps
        share = max(share, l.net_after - c)
    # (329 review I3) the APP reduced a BOT position: a closing fill that is not a resting-TP fill of a member, takes |net| BELOW the
    # members' concurrent quantity, more than 30 s before the close of every member open at that moment → ambiguous (the app's P&L on
    # the bot's coins cannot be split from the rows; never silently "the bot's")
    flat_orders = {l.fill.order_id for l in ep.closes() if l.net_after <= eps}   # (round-3 4) every fill of the flattening ORDER
    for l in ep.closes():
        if l.net_after <= eps or l.fill.order_id in flat_orders:
            continue                                           # (round-2 B) a FLATTEN carries no app share — a bot stop / backstop that
                                                               # filled minutes before the bot booked it (bot down) stays the bot's
        live = [r for r in members if _member_lo(r, ep) <= l.fill.t <= _member_hi(r, ep, now_ms)]
        c = sum(r.qty for r in live)
        if not live or l.net_after >= c * (1 - QTY_TOL) - eps:
            continue
        if any(r.rtp_price and l.fill.maker and abs(l.fill.price - r.rtp_price) <= RTP_PX_TOL * r.rtp_price for r in live):
            continue                                           # the bot's own resting TP (a partial fill before its final close)
        if all(r.close_ms is not None and l.fill.t < r.close_ms - EARLY_REDUCE_MS for r in live):
            return Verdict("AMBIGUOUS", f"a fill reduced the bot's position {(min(r.close_ms for r in live) - l.fill.t) / 1000:.0f} s before "
                                        f"the bot closed it (the app sold bot coins?)", members)
    if not outside and not over:
        return Verdict("BOT", "fully explained by the bot's rows", members)
    if share <= max(eps, QTY_TOL * ep.max_abs_net):
        return Verdict("AMBIGUOUS", "fills outside every bot row's window but no quantity above the bot's", members)
    ids = frozenset().union(*[r.order_ids for r in members])

    def own_entry(l):                                          # an opening leg that is a member's own
        if l.fill.order_id in ids:
            return True
        for r in members:
            if r.imported and l.fill.t <= r.open_ms + ROW_PAD_MS:
                return True                                    # 📥 the import adopted everything held up to it
            if not r.imported and not r.entry_id_known and abs(l.fill.t - r.open_ms) <= ROW_PAD_MS:
                return True
        return False
    app_open = [l for l in ep.opens() if not own_entry(l)]
    if sum(l.qty for l in app_open) < share * (1 - QTY_TOL):
        return Verdict("AMBIGUOUS", "the app's opening fills cannot be told apart from the bot's", members)
    app_close = [l for l in ep.closes() if l.fill.order_id not in ids]
    exit_legs, scale, approx = app_close, 1.0, False
    if sum(l.qty for l in app_close) < share * (1 - QTY_TOL) or sum(l.qty for l in app_close) > share * (1 + QTY_TOL):
        exit_legs = ep.closes()
        tq = sum(l.qty for l in exit_legs)
        scale, approx = (share / tq if tq > 0 else 0.0), True
    # (329 review I2) a member booked on an unverified exit → its pnl cannot anchor "episode − booked"
    for r in members:
        if any(f in (r.notes or "") for f in ("RTP_FILL_UNVERIFIED", "RTP_SETTLE_BLOCKED")):
            return Verdict("AMBIGUOUS", f"bot row #{r.id} was booked on an unverified exit ({r.notes})", members)
        if str(r.close_reason or "").upper() == "EXTERNAL_CLOSE" and not (
                r.exit_price and any(abs(px - r.exit_price) <= EXT_PX_TOL * r.exit_price for px in close_price_candidates(ep))):
            return Verdict("AMBIGUOUS", f"bot row #{r.id} (EXTERNAL_CLOSE) was booked at {r.exit_price}, no Binance fill at that price "
                                        f"(fallback price)", members)
    last_close = ep.closes()[-1].fill.t if ep.closes() else ep.end_ms
    sold_by_bot = any(r.close_ms is not None and abs(last_close - r.close_ms) <= SOLD_BY_BOT_MS for r in members)
    return Verdict("SHARE", "app quantity above the bot's rows", members, share, app_open, exit_legs, scale, approx,
                   "APP_ADD_SOLD_BY_BOT" if sold_by_bot else "APP_CLOSE")


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def dt_to_ms(d: Optional[datetime]) -> Optional[int]:
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def build_row_fields(ep: Episode, v: Verdict, rate, funding: Optional[float], leverage: Optional[int], now: datetime) -> dict:
    """The Order fields for a pure APP episode or the app share of a mixed one (raises ValueError when a number cannot be computed —
    the caller then records nothing and logs it as ambiguous)."""
    if v.kind == "APP":
        open_legs, close_legs, scale = ep.opens(), ep.closes(), 1.0
        qty = ep.max_abs_net
        e_fee, e_est = comm_usdt(open_legs, rate)
        x_fee, x_est = comm_usdt(close_legs, rate)
        pnl = sum(l.rpnl for l in ep.legs) - e_fee - x_fee
        opened_ms = ep.start_ms
        notes_extra = ""
    elif v.kind == "SHARE":
        open_legs, close_legs, scale = v.entry_legs, v.exit_legs, v.exit_scale
        qty = v.share_qty
        all_fee, est_all = comm_usdt(ep.legs, rate)
        ep_pnl = sum(l.rpnl for l in ep.legs) - all_fee
        pnl = ep_pnl - sum(float(r.pnl) for r in v.members)
        e_fee, e_est = comm_usdt(open_legs, rate)
        x_fee, x_est = comm_usdt(close_legs, rate)
        x_fee *= scale
        e_est, x_est = e_est or est_all, x_est or est_all
        opened_ms = min(l.fill.t for l in open_legs)
        notes_extra = (f" · APP SHARE of a mixed episode with bot row(s) #{', #'.join(str(r.id) for r in v.members)}: P&L = episode "
                       f"{ep_pnl:+.4f} − booked {sum(float(r.pnl) for r in v.members):+.4f}; qty = peak |net| above the bot's"
                       + (" · exit price = VWAP of ALL closing fills (approximate: the bot's own exit fills carry unknown ids)" if v.exit_approx else ""))
        if funding is not None:
            funding = funding - sum(float(r.funding or 0.0) for r in v.members)
        # (329 review I2) cross-check: the share P&L from its own fills must agree with "episode − booked" — a gap means a booking error
        # of the bot row would be absorbed into the 📱 line → nothing recorded
        # (round-2 D) when the exit is the BLENDED VWAP (v.exit_approx — always in production, I1) the app's true exit is unknown but
        # it IS one of the episode's closing prices: the residual must fall inside the P&L band spanned by the lowest … highest closing
        # price (entry side + qty exact), widened by the same tolerance. A bot-row booking error moves the residual out of that band;
        # the blend alone cannot. With identified exit fills the point comparison stays.
        _e, _x = vwap(open_legs), vwap(close_legs)
        if _e and _x:
            sg = 1.0 if ep.direction == "LONG" else -1.0
            tol = max(SHARE_XCHECK_USD, SHARE_XCHECK_SHARE * abs(pnl))
            if v.exit_approx:
                pxs = [l.fill.price for l in ep.closes()]
                band = sorted(sg * (p - _e) * qty - e_fee - x_fee for p in (min(pxs), max(pxs)))
                if not (band[0] - tol <= pnl <= band[1] + tol):
                    raise ValueError(f"app share P&L {pnl:+.4f} (episode − booked) is outside what its fills allow "
                                     f"[{band[0]:+.4f}, {band[1]:+.4f}] (a bot row's booking differs from Binance)")
            else:
                fills_pnl = sg * (_x - _e) * qty - e_fee - x_fee
                if abs(fills_pnl - pnl) > tol:
                    raise ValueError(f"app share P&L from its fills {fills_pnl:+.4f} vs episode − booked {pnl:+.4f} (a bot row's booking "
                                     f"differs from Binance)")
    else:
        raise ValueError(f"nothing to record for a {v.kind} episode")
    entry, exitp = vwap(open_legs), vwap(close_legs)
    if not (entry and exitp and qty > 0):
        raise ValueError("no entry / exit price")
    notional = entry * qty
    lev = int(leverage) if leverage and int(leverage) >= 1 else 1
    est = bool(e_est or x_est)
    notes = (f"🧾 Binance-app trade auto-recorded from Binance fills (DECISION_LOG 329): {ep.n_fills} fill(s), "
             f"{'scale-in/out — quantity = peak size, ' if len(open_legs) > 1 or len(close_legs) > 1 else ''}"
             f"leverage = the symbol's setting at record time ({'approximate' if leverage else 'unreadable → 1'})"
             f"{' · a BNB fee converted at the CURRENT BNB price (estimate)' if est else ''}{notes_extra}")
    return dict(
        pair=ep.symbol, direction=ep.direction, status="CLOSED", entry_price=entry, exit_price=exitp, current_price=exitp,
        investment=notional / lev, leverage=float(lev), notional_value=notional, quantity=qty, confidence="STRONG_BUY",
        entry_strategy="MANUAL", import_source=IMPORT_SOURCE_APP_HISTORY, imported_at=now, app_trade_key=ep.key,
        app_fee_estimated=est, app_fills=ep.n_fills, entry_fee=e_fee, exit_fee=x_fee, total_fee=e_fee + x_fee,
        pnl=pnl, pnl_percentage=(pnl / notional * 100.0), peak_pnl=0.0, trough_pnl=0.0,
        opened_at=ms_to_dt(opened_ms), closed_at=ms_to_dt(ep.end_ms), close_reason=v.close_reason,
        entry_order_type=order_type(open_legs), exit_order_type=order_type(close_legs), is_paper=False,
        funding_fee_usd=(round(funding, 6) if funding is not None else None), notes=notes[:2000])


# ── pure: reconcile ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def income_day_sums(rows: Iterable[dict], rate, lo_ms: int, hi_ms: int) -> Dict[str, list]:
    """{UTC day: [Σ REALIZED_PNL, Σ COMMISSION in USDT (negative = paid), Σ FUNDING_FEE, n commissions converted at an estimated price]}
    over income rows with lo_ms ≤ time ≤ hi_ms. Unknown commission asset → ValueError."""
    out: Dict[str, list] = {}
    for r in rows:
        t = int(r.get("time") or 0)
        if t < lo_ms or t > hi_ms:
            continue
        typ = str(r.get("incomeType") or "")
        if typ not in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE"):
            continue
        d = ms_to_dt(t).strftime("%Y-%m-%d")
        acc = out.setdefault(d, [0.0, 0.0, 0.0, 0])
        x = float(r.get("income") or 0.0)
        asset = str(r.get("asset") or "USDT").upper()
        if typ == "COMMISSION":
            rr = rate(asset, t)
            if rr is None:
                raise ValueError(f"commission asset {asset} cannot be converted to USDT")
            acc[1] += x * rr[0]
            acc[3] += 1 if rr[1] else 0
        else:
            if asset != "USDT":
                rr = rate(asset, t)
                if rr is None:
                    raise ValueError(f"income asset {asset} cannot be converted to USDT")
                x *= rr[0]
            acc[0 if typ == "REALIZED_PNL" else 2] += x
    return out


def exact_summary(state: dict, key_to_id: Dict[str, int]) -> dict:
    """(329 review I6) the per-row acceptance check: every pure APP 📱 row vs Σ REALIZED_PNL + Σ COMMISSION income of its symbol over its
    [opened_at, closed_at] (to the cent); SHARE rows listed apart (their pnl = episode − booked by design, not comparable to income)."""
    rows = []
    for k, v in sorted(((state or {}).get("exact") or {}).items()):
        if k not in key_to_id:
            continue                                           # a row deleted by a reset
        rows.append({"id": key_to_id[k], "key": k, **v})
    app = [r for r in rows if r.get("kind") == "APP"]
    return {"n_exact": sum(1 for r in app if r.get("ok") is True),
            "n_mismatched": sum(1 for r in app if r.get("ok") is False),
            "mismatched_ids": [r["id"] for r in app if r.get("ok") is False],
            "n_pending": sum(1 for r in app if r.get("ok") is None) + sum(1 for k in key_to_id if k not in ((state or {}).get("exact") or {})),
            "row_pnl_total": round(sum(float(r.get("row_pnl") or 0) for r in app), 2),
            "income_pnl_total": round(sum(float(r.get("income_pnl") or 0) for r in app if r.get("income_pnl") is not None), 2),
            "rows": app, "share_rows": [r for r in rows if r.get("kind") == "SHARE"]}


def reconcile_report(state: dict, era_start: Optional[datetime], db_pnl: float, db_funding: float, db_rows: int,
                     app_est_rows: int, key_to_id: Optional[Dict[str, int]] = None) -> dict:
    """The apples-to-apples check (DECISION_LOG 329). Binance: REALIZED_PNL is GROSS of commission, COMMISSION and FUNDING_FEE are their own
    income rows → Binance trading P&L = Σ REALIZED_PNL + Σ COMMISSION (USDT); funding = Σ FUNDING_FEE. DB: every row's pnl is NET of its
    fees with funding NOT in it (funding_fee_usd) → DB trading P&L = Σ pnl, funding = Σ funding_fee_usd, over ALL live CLOSED rows (bot +
    🖐 + 📥 + 📱) closed since the era start. gap = Binance − DB, per part and in total; what is known to sit in it is listed."""
    days = (state or {}).get("days") or {}
    rp = sum(float(v[0]) for v in days.values())
    cm = sum(float(v[1]) for v in days.values())
    fu = sum(float(v[2]) for v in days.values())
    est_comm = sum(int(v[3]) for v in days.values())
    op = (state or {}).get("open_partials") or {}
    amb = (state or {}).get("ambiguous") or {}
    dfr = (state or {}).get("deferred") or {}
    open_trading = sum(float(v.get("trading", 0.0)) for v in op.values())
    open_funding = sum(float(v.get("funding", 0.0)) for v in op.values())
    amb_pnl = sum(float(v.get("pnl") or 0.0) for v in amb.values())
    dfr_pnl = sum(float(v.get("pnl") or 0.0) for v in dfr.values())
    b_trade, b_total = rp + cm, rp + cm + fu
    g_trade, g_fund = b_trade - db_pnl, fu - db_funding
    gap = b_total - (db_pnl + db_funding)
    known = open_trading + open_funding + amb_pnl + dfr_pnl
    return {
        "since": era_start.isoformat() if era_start else None,
        "binance": round(b_total, 2),
        "binance_parts": {"realized_pnl_gross": round(rp, 2), "commission_usdt": round(cm, 2), "funding": round(fu, 2),
                          "trading_net": round(b_trade, 2), "commissions_converted_at_estimated_bnb_price": est_comm},
        "db": round(db_pnl + db_funding, 2),
        "db_parts": {"pnl_net_of_fees": round(db_pnl, 2), "funding_fee_usd": round(db_funding, 2), "closed_rows": int(db_rows)},
        "gap": round(gap, 2), "gap_trading": round(g_trade, 2), "gap_funding": round(g_fund, 2),
        "gap_explained_by": [
            {"what": "open positions' realized partials + their opening commissions (Binance has them, no CLOSED row yet)",
             "usd": round(open_trading, 2), "symbols": sorted(op)},
            {"what": "funding paid / received on positions still open", "usd": round(open_funding, 2)},
            {"what": "ambiguous app episodes NOT recorded (manual look needed)", "usd": round(amb_pnl, 2), "n": len(amb)},
            {"what": "episodes deferred (a bot row still OPEN)", "usd": round(dfr_pnl, 2), "n": len(dfr)},
            {"what": "app rows whose BNB fee used the current BNB price (estimate)", "n": int(app_est_rows)},
        ],
        "unexplained": round(gap - known, 2),
        "unexplained_note": ("the rest = the bot rows' booked fees / fill prices vs Binance's real ones (fee-rate estimates, BNB discount), "
                             "📥 imported rows' estimated entry fee and import-time entry, funding the bot did not stamp on a row, rounding"),
        "account_gap_noise_sources": [
            "bot rows' fees are booked from the configured fee rates (BNB discount, maker / taker), not Binance's charged commission",
            "bot rows' exit prices: a fallback / blended exit price on an EXTERNAL_CLOSE or resting-TP settle",
            "📥 imported rows (DL 328): estimated entry fee, entry = Binance's average at the import",
            "funding: only stamped on rows whose funding read succeeded (funding_fee_usd NULL counts 0)",
            "BNB commission converted at the 1m close of its minute (an estimate when that kline was missing)",
            "income history: Binance serves ~3 months — older days keep the sums stored when they were read"],
        "app_rows_exact": exact_summary(state, key_to_id or {}),
        "last_pass": (state or {}).get("last_pass"),
    }


def app_trades_summary(orders: Iterable, ambiguous_n: int = 0) -> dict:
    """{n, wins, wr, pnl, avg_pct, ambiguous_skipped} over the 📱 rows (import_source BINANCE_APP_HISTORY) in `orders` — both text exports."""
    rows = [o for o in (orders or ()) if str(getattr(o, "import_source", None) or "") == IMPORT_SOURCE_APP_HISTORY
            and getattr(o, "pnl_percentage", None) is not None]
    n = len(rows)
    wins = sum(1 for o in rows if (o.pnl_percentage or 0) > 0)
    return {"n": n, "wins": wins, "wr": (round(100.0 * wins / n, 1) if n else None),
            "pnl": round(sum(float(o.pnl or 0) for o in rows), 2),
            "avg_pct": (round(sum(float(o.pnl_percentage or 0) for o in rows) / n, 3) if n else None),
            "ambiguous_skipped": int(ambiguous_n or 0)}


def is_app_trade(o) -> bool:
    return str(getattr(o, "import_source", None) or "") == IMPORT_SOURCE_APP_HISTORY


def load_state(raw: Optional[str]) -> dict:
    try:
        s = json.loads(raw) if raw else {}
        return s if isinstance(s, dict) else {}
    except (TypeError, ValueError):
        return {}


# ── the recorder (I/O) ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
class WeightBudget:
    """≤ per_min request weight in any rolling 60 s (this job only)."""

    def __init__(self, per_min: int = APP_WEIGHT_PER_MIN, sleep=asyncio.sleep, clock=time.monotonic):
        self.per_min, self._sleep, self._clock, self._log = per_min, sleep, clock, []

    async def spend(self, w: int):
        while True:
            now = self._clock()
            self._log = [(t, x) for t, x in self._log if now - t < 60.0]
            if sum(x for _, x in self._log) + w <= self.per_min or not self._log:
                self._log.append((now, w))
                return
            await self._sleep(max(0.05, 60.0 - (now - self._log[0][0]) + 0.05))


def _is_limit_error(e: Exception) -> bool:
    try:
        from ccxt.base.errors import RateLimitExceeded, DDoSProtection
        if isinstance(e, (RateLimitExceeded, DDoSProtection)):
            return True
    except Exception:
        pass
    s = str(e)
    return any(k in s for k in ("418", "429", "-1003", "Too many requests", "banned", "Way too many"))


class AppTradeRecorder:
    """One pass = read → replay → attribute → write (own sessions; never touches OPEN rows or the trading loops)."""

    def __init__(self, svc=None, session_factory=None, commit=None, is_paper: Callable[[], bool] = None, enabled: Callable[[], bool] = None,
                 now_ms: Callable[[], int] = None, sleep=asyncio.sleep, budget: Optional[WeightBudget] = None, ban_active: Callable[[], bool] = None):
        if svc is None:
            from services.binance_service import binance_service as svc
        if session_factory is None:
            from database import AsyncSessionLocal as session_factory
        if commit is None:
            from database import locked_commit as commit
        self.svc, self.sf, self.commit = svc, session_factory, commit
        self.is_paper = is_paper or (lambda: bool(getattr(__import__("services.trading_engine", fromlist=["trading_engine"]).trading_engine, "is_paper_mode", True)))
        self.enabled = enabled or (lambda: bool(getattr(__import__("config").trading_config, "app_trade_record_enabled", True)))
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._sleep = sleep
        self.budget = budget or WeightBudget(sleep=sleep)
        if ban_active is None:
            def ban_active():
                from services.binance_service import get_ban_status
                return bool(get_ban_status().get("banned"))
        self.ban_active = ban_active
        self._bnb: Dict[int, float] = {}          # minute open ms → BNB close
        self._bnb_missing: set = set()            # blocks that could not be read this pass
        self._bnb_done: set = set()               # blocks read this pass
        self._bnb_now: Optional[float] = None
        self._lev: Optional[Dict[str, int]] = None
        self.last_summary: dict = {}
        self._rec_logged = -1e18

    # ── exchange calls (budget + ban + header) ──
    async def _call(self, weight: int, fn, *a):
        if self.ban_active():
            raise AppTradeStop("an IP ban is active")
        await self.budget.spend(weight)
        try:
            res = await fn(*a)
        except Exception as e:
            try:
                from services.binance_service import BinanceService
                BinanceService._detect_ban(e)
            except Exception:
                pass
            if _is_limit_error(e):
                raise AppTradeStop(f"Binance rate limit / ban: {e}")
            raise
        try:
            uw = self.svc.used_weight_1m()
        except Exception:
            uw = None
        if uw is not None and uw > APP_HEADER_PAUSE:
            logger.warning(f"[APP_TRADE] X-MBX-USED-WEIGHT-1M = {uw} > {APP_HEADER_PAUSE} — pausing 60 s")
            await self._sleep(60)
        return res

    async def _income(self, lo, hi, symbol=None, income_type=None) -> List[dict]:
        """All income rows in [lo, hi] (windows ≤ 7 days). Lossless paging: the next page starts AT the last row's time (overlap) and rows
        are deduped on (incomeType, tranId, symbol, asset) — (329 review C1) tranId alone repeats across income types. A full page inside
        ONE millisecond cannot be paged losslessly → RuntimeError (the pass ends, logged; never a silent gap)."""
        out, seen = [], set()
        w0 = lo
        while w0 <= hi:
            w1 = min(hi, w0 + WINDOW_MS - 1)
            cur = w0
            for _ in range(500):
                rows = await self._call(W_INCOME, self.svc.fetch_income_page, cur, w1, PAGE, symbol, income_type)
                for r in rows:
                    k = income_key(r)
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(r)
                if len(rows) < PAGE:
                    break
                nxt = int(rows[-1].get("time") or 0)
                if nxt <= cur:
                    raise RuntimeError(f"income page of {PAGE} rows inside one millisecond at {cur} — cannot page losslessly")
                cur = nxt
            else:
                raise RuntimeError("income paging did not finish")
            w0 = w1 + 1
        return out

    async def _fills(self, sym, lo, hi) -> List[Fill]:
        """All fills of sym in [lo, hi]: one ≤ 7-day window at a time; a full page continues with fromId = last id + 1 (lossless — the
        (329 review I4) same-millisecond skip is gone) until a trade past the window shows."""
        out, seen = [], set()
        w0 = lo
        while w0 <= hi:
            w1 = min(hi, w0 + WINDOW_MS - 1)
            rows = await self._call(W_TRADES, self.svc.fetch_user_trades_page, sym, w0, w1, PAGE)
            batch = list(rows)
            for _ in range(500):
                if len(rows) < PAGE or not rows or any(int(r["time"]) > w1 for r in rows):
                    break
                rows = await self._call(W_TRADES, self.svc.fetch_user_trades_page, sym, None, None, PAGE, int(rows[-1]["id"]) + 1)
                batch.extend(rows)
            else:
                raise RuntimeError("userTrades paging did not finish")
            for r in batch:
                if w0 <= int(r["time"]) <= w1 and int(r["id"]) not in seen:
                    seen.add(int(r["id"]))
                    out.append(r)
            w0 = w1 + 1
        return sorted((parse_fill(r) for r in out), key=lambda f: (f.t, f.trade_id))

    async def _bnb_block(self, minute_ms):
        blk = minute_ms - (minute_ms % (PAGE * 60_000))
        if blk in self._bnb_missing or blk in self._bnb_done:
            return                                         # one read per block per pass (a gap minute never re-reads it)
        self._bnb_done.add(blk)
        if len(self._bnb) > 200_000:                       # ~140 days of minutes — keep memory bounded
            self._bnb.clear()
        try:
            for t, c in await self._call(W_KLINES, self.svc.fetch_bnb_1m_closes, blk, PAGE):
                if c > 0:
                    self._bnb[t] = c
        except AppTradeStop:
            raise
        except Exception as e:
            logger.warning(f"[APP_TRADE] BNB 1m klines unreadable from {blk}: {e} — current BNB price used (estimate)")
            self._bnb_missing.add(blk)

    async def prepare_rates(self, times_by_asset: Dict[str, Iterable[int]]):
        """Load the BNB 1m closes every BNB-paid commission needs (blocks of 1000 minutes)."""
        for t in sorted(set(times_by_asset.get("BNB") or ())):
            m = t - (t % 60_000)
            if m not in self._bnb:
                await self._bnb_block(m)
        if any(t - (t % 60_000) not in self._bnb for t in (times_by_asset.get("BNB") or ())) and self._bnb_now is None:
            try:
                self._bnb_now = float(await self._call(1, self.svc.fetch_bnb_price_raw))
            except AppTradeStop:
                raise
            except Exception as e:
                logger.warning(f"[APP_TRADE] current BNB price unreadable ({e}) — a BNB fee without its minute kline cannot be converted")
                self._bnb_now = None

    def rate(self, asset: str, t: int) -> Optional[Tuple[float, bool]]:
        a = str(asset or "").upper()
        if a == "USDT":
            return 1.0, False
        if a in ("USDC", "FDUSD", "BUSD"):
            return 1.0, True
        if a == "BNB":
            c = self._bnb.get(t - (t % 60_000))
            if c:
                return c, False
            return (self._bnb_now, True) if self._bnb_now else None
        return None

    # ── DB ──
    async def _era_start(self, db):
        from sqlalchemy import select, func, and_, or_
        from models import Order
        return (await db.execute(select(func.min(Order.opened_at)).where(and_(
            Order.is_paper == False, or_(Order.import_source.is_(None), Order.import_source != IMPORT_SOURCE_APP_HISTORY))))).scalar()  # noqa: E712

    async def _rows(self, db, sym, lo_ms, now_ms) -> List[RowView]:
        from sqlalchemy import select, and_, or_
        from models import Order, Transaction
        lo = ms_to_dt(lo_ms - ROW_PAD_MS - WINDOW_MS)
        orders = (await db.execute(select(Order).where(and_(
            Order.is_paper == False, Order.pair == sym, Order.status.in_(("OPEN", "CLOSED")),  # noqa: E712
            or_(Order.import_source.is_(None), Order.import_source != IMPORT_SOURCE_APP_HISTORY),
            or_(Order.closed_at.is_(None), Order.closed_at >= lo))))).scalars().all()
        if not orders:
            return []
        tx = (await db.execute(select(Transaction.order_id, Transaction.binance_order_id).where(and_(
            Transaction.order_id.in_([o.id for o in orders]), Transaction.binance_order_id.isnot(None))))).all()
        ids: Dict[int, set] = {}
        for oid, bid in tx:
            ids.setdefault(oid, set()).add(str(bid))
        out = []
        for o in orders:
            s = set(ids.get(o.id, set()))
            if o.binance_order_id:
                s.add(str(o.binance_order_id))
            imported = str(o.import_source or "") == IMPORT_SOURCE_IMPORTED
            out.append(RowView(id=o.id, direction=str(o.direction or "").upper(), status=str(o.status or ""), qty=float(o.quantity or 0),
                               open_ms=dt_to_ms(o.imported_at if imported and o.imported_at else o.opened_at), close_ms=dt_to_ms(o.closed_at),
                               pnl=(float(o.pnl) if o.pnl is not None else None), imported=imported, order_ids=frozenset(s),
                               entry_id_known=bool(o.binance_order_id), funding=o.funding_fee_usd, close_reason=str(o.close_reason or ""),
                               notes=str(o.notes or ""), exit_price=o.exit_price, rtp_price=o.resting_tp_price))
        return out

    async def _existing_keys(self, db, sym) -> Dict[str, Optional[float]]:
        from sqlalchemy import select, and_
        from models import Order
        return {k: p for (k, p) in (await db.execute(select(Order.app_trade_key, Order.pnl).where(and_(
            Order.pair == sym, Order.app_trade_key.isnot(None))))).all()}

    # ── one pass ──
    async def run_pass(self) -> dict:
        if not self.enabled():
            return {"status": "disabled"}
        if self.is_paper():
            return {"status": "paper"}
        if self.ban_active():
            return {"status": "ban"}
        from sqlalchemy import select, update
        from models import BotState
        async with self.sf() as db:
            era = await self._era_start(db)
            bs = (await db.execute(select(BotState).limit(1))).scalar_one_or_none()
            if era is None or bs is None:
                return {"status": "no_live_trades" if era is None else "no_state"}
            bs_id, cursor, state = bs.id, bs.app_trades_cursor_ms, load_state(bs.app_trades_state_json)
        era_ms = dt_to_ms(era)
        if state.get("era_ms") != era_ms:                      # a live reset moved the era → drop what lay before it
            state["days"] = {d: v for d, v in (state.get("days") or {}).items() if d >= ms_to_dt(era_ms).strftime("%Y-%m-%d")}
            if state.get("era_ms") is not None:                # the old era's episodes / partials are re-judged from scratch
                for _k in ("ambiguous", "deferred", "open_partials", "retry"):
                    state[_k] = {}
            if cursor is not None and cursor < era_ms:
                cursor = None
            state["era_ms"] = era_ms
        for k in ("ambiguous", "deferred", "open_partials", "days", "retry", "exact"):
            state.setdefault(k, {})
        self._bnb_missing, self._bnb_done, self._bnb_now, self._lev = set(), set(), None, None
        summary = {"status": "ok", "recorded": 0, "shares": 0, "bot": 0, "ambiguous_new": 0, "deferred": 0, "busy": [], "symbols": 0}
        try:
            tp = self.now_ms()
            positions = await self._call(W_POS, self.svc.get_net_positions)
            if not isinstance(positions, dict):
                raise RuntimeError("positions unreadable")
            base = era_ms if cursor is None else cursor - DAY_MS
            _now = self.now_ms()
            for _s, _r in list(state["retry"].items()):        # (round-3 3) retried 14 days since FIRST listed → dropped, kept as ambiguous
                if not isinstance(_r, dict):
                    _r = state["retry"][_s] = {"point": int(_r), "first_seen_ms": _now}
                if _now - int(_r.get("first_seen_ms", _now)) > RETRY_MAX_DAYS * DAY_MS:
                    logger.error(f"[APP_TRADE_AMBIGUOUS] {_s}: still unreadable / mismatched from {ms_to_dt(retry_point(_r)):%Y-%m-%d %H:%M} after "
                                 f"{RETRY_MAX_DAYS} days of retries — dropped from the retry list, NOTHING recorded there (manual look)")
                    state["ambiguous"][f"{_s}:retry:{retry_point(_r)}"] = {"symbol": _s, "start_ms": retry_point(_r), "end_ms": None,
                                                                           "reason": f"dropped after {RETRY_MAX_DAYS} days of retries", "pnl": None, "rows": []}
                    state["retry"].pop(_s, None)
            lo = max(era_ms, base)
            if cursor is not None:
                lo = max(era_ms, lo - (lo % DAY_MS))           # midnight-aligned (or the era start): whole income days
            # (round-2 G) a deferred / retried symbol re-reads from its OWN point (its own symbol income read) — the global read stays at
            # cursor − 1 day, never dragged back for every symbol
            sym_from: Dict[str, int] = {}
            for v in state["deferred"].values():
                if v.get("symbol"):
                    sym_from[str(v["symbol"])] = min(sym_from.get(str(v["symbol"]), lo), int(v.get("first_ms") or lo))
            for _s, _r in state["retry"].items():
                sym_from[_s] = min(sym_from.get(_s, lo), retry_point(_r))
            sym_from = {k: max(era_ms, x - (x % DAY_MS)) for k, x in sym_from.items()}
            income = await self._income(lo, tp)
            recent = (cursor - 600_000) if cursor is not None else lo
            syms = {str(r.get("symbol")) for r in income if r.get("symbol") and r.get("incomeType") in ("REALIZED_PNL", "COMMISSION")
                    and int(r.get("time") or 0) >= recent}
            syms |= {s for s, q in positions.items() if q}
            syms |= {str(v.get("symbol")) for v in state["deferred"].values() if v.get("symbol")}
            syms |= set(state["retry"])
            for s in list(state["open_partials"]):
                if not positions.get(s):
                    state["open_partials"].pop(s, None)
            summary["symbols"] = len(syms)
            for sym in sorted(syms):
                if not self.enabled():
                    summary["status"] = "disabled_mid_pass"
                    break
                # (round-3 1) a stop inside the symbol must not lose its retry / deferred points (popped before its last exchange calls)
                _pre_r = state["retry"].get(sym)
                _pre_d = {k: v for k, v in state["deferred"].items() if v.get("symbol") == sym}
                try:
                    await self._symbol(sym, lo, min(lo, sym_from.get(sym, lo)), tp, era_ms, float(positions.get(sym) or 0.0), income, state, summary)
                except AppTradeStop:
                    if _pre_r is not None:
                        state["retry"][sym] = _pre_r if sym not in state["retry"] else state["retry"][sym]
                    for _k, _v in _pre_d.items():
                        state["deferred"].setdefault(_k, _v)
                    raise
            # reconcile income sums (whole days from lo; the era day from the era start)
            await self.prepare_rates({"BNB": [int(r.get("time") or 0) for r in income
                                              if r.get("incomeType") == "COMMISSION" and str(r.get("asset") or "").upper() == "BNB"]})
            try:   # every day from lo is WHOLE in this read (lo = midnight or the era start) → overwritten, never added twice
                for d, v in income_day_sums(income, self.rate, lo, tp).items():
                    state["days"][d] = [round(v[0], 8), round(v[1], 8), round(v[2], 8), int(v[3])]
            except ValueError as e:
                logger.warning(f"[APP_TRADE_RECONCILE] income sums not updated this pass: {e}")
            state["last_pass"] = {"at_ms": tp, "from_ms": lo, **{k: summary[k] for k in ("recorded", "shares", "bot", "ambiguous_new", "deferred")}}
            if len(state["ambiguous"]) > STATE_AMBIG_CAP:
                for k in sorted(state["ambiguous"], key=lambda k: state["ambiguous"][k].get("end_ms") or 0)[:-STATE_AMBIG_CAP]:
                    state["ambiguous"].pop(k, None)
            new_cursor = tp if summary["status"] == "ok" else cursor   # disabled mid-pass → the same window again next time
            async with self.sf() as db:
                await db.execute(update(BotState).where(BotState.id == bs_id).values(
                    app_trades_cursor_ms=new_cursor, app_trades_state_json=json.dumps(state, separators=(",", ":"))))
                await self.commit(db)
            if cursor is None or summary["recorded"] or time.monotonic() - self._rec_logged >= 3600:   # after the backfill, a write, or hourly
                await self._log_reconcile(state, era)
                self._rec_logged = time.monotonic()
        except AppTradeStop as e:
            logger.warning(f"[APP_TRADE] pass stopped: {e} — cursor kept, resumes next cycle (rows already written stay)")
            summary["status"] = "stopped"
            summary["error"] = str(e)
            try:   # (round-2 H) keep what this pass learned (exact entries of rows it wrote, ambiguous / deferred / retry) — cursor NOT moved
                async with self.sf() as db:
                    await db.execute(update(BotState).where(BotState.id == bs_id).values(
                        app_trades_state_json=json.dumps(state, separators=(",", ":"))))
                    await self.commit(db)
            except Exception as se:
                logger.error(f"[APP_TRADE] state not saved after the stop: {se}")
        self.last_summary = summary
        if summary["recorded"] or summary["ambiguous_new"] or cursor is None:
            logger.info(f"[APP_TRADE] pass {summary['status']}: {summary['recorded']} recorded ({summary['shares']} app share) · "
                        f"{summary['bot']} bot-only · {summary['ambiguous_new']} new ambiguous · {summary['deferred']} deferred · "
                        f"{summary['symbols']} symbol(s){' · busy ' + ','.join(summary['busy']) if summary['busy'] else ''}")
        return summary

    async def _exact(self, ep, kind, row_pnl, ep_inc, ep_net) -> dict:
        """(review I6, round-2 A) the row vs the income of the episode's OWN tradeIds (split fills pro-rated exactly as the row)."""
        ex = {"symbol": ep.symbol, "kind": kind, "row_pnl": round(row_pnl, 6), "income_pnl": None, "ok": None}
        if ep_inc is None:
            return ex
        try:
            await self.prepare_rates({"BNB": [int(r.get("time") or 0) for r in ep_inc if str(r.get("asset") or "").upper() == "BNB"]})
            ipnl = episode_income_pnl(ep, ep_inc, self.rate)
        except ValueError:
            return ex
        ex["income_pnl"] = round(ipnl, 6)
        if kind == "SHARE":
            ex["episode_vs_income"] = round(ep_net - ipnl, 6) if ep_net is not None else None
        else:
            ex["ok"] = abs(ipnl - row_pnl) <= EXACT_USD
        return ex

    async def _symbol(self, sym, lo, sym_lo, tp, era_ms, end_net, income, state, summary):
        _retry_first = None   # (final gate) first_seen_ms of a retry entry popped this pass
        try:
            fills = await self._fills(sym, max(era_ms, sym_lo - FILL_PAD_MS), self.now_ms())
            busy = any(f.t >= tp - APP_BUSY_S * 1000 for f in fills)
            fills = [f for f in fills if f.t < tp]
            sym_income = [r for r in income if r.get("symbol") == sym]
            if not busy and sym_lo < lo:                       # (round-2 G) this symbol's own older income
                sym_income = await self._income(sym_lo, lo - 1, symbol=sym) + sym_income
        except AppTradeStop:
            raise
        except Exception as e:
            logger.warning(f"[APP_TRADE] {sym}: fills unreadable ({e}) — retried next pass from {ms_to_dt(sym_lo):%Y-%m-%d %H:%M}")
            retry_add(state, sym, sym_lo, tp)
            summary.setdefault("retry", []).append(sym)
            return
        if busy:
            summary["busy"].append(sym)                        # the position anchor could miss a fill this fresh — next pass
            return
        # (329 review I4, round-2 F) completeness guard per UTC day of [sym_lo, tp): the fills must reproduce the income (tradeIds + sums). A bad
        # day cuts the replay: only fills AFTER the last bad day are replayed (anchored on the live position from the end, so they are exact);
        # the episodes before it wait (retry from the first bad day) — one bad window never blocks the symbol's later windows.
        try:
            await self.prepare_rates({"BNB": [f.t for f in fills if f.comm_asset == "BNB"] +
                                      [int(r.get("time") or 0) for r in sym_income if str(r.get("asset") or "").upper() == "BNB"]})
            bad = fill_income_bad_windows(sym, fills, sym_income, sym_lo, tp, self.rate)
        except ValueError as e:
            bad = [(sym_lo, tp, str(e))]
        cut = None
        if bad:
            cut = max(e for _, e, _ in bad)
            for b0, b1, why in bad:
                logger.warning(f"[APP_TRADE_FILL_MISMATCH] {sym} {ms_to_dt(b0):%Y-%m-%d %H:%M} → {ms_to_dt(b1):%Y-%m-%d %H:%M}: {why} — "
                               f"episodes before {ms_to_dt(cut):%Y-%m-%d %H:%M} skipped, retried next pass")
            _first = state["retry"].get(sym)                   # the point moves to the first bad day (earlier days now verify); age kept
            state["retry"][sym] = {"point": min(b for b, _, _ in bad),
                                   "first_seen_ms": int(_first.get("first_seen_ms", tp)) if isinstance(_first, dict) else tp}
            summary.setdefault("retry", []).append(sym)
            fills = [f for f in fills if f.t >= cut]
        else:
            _popped = state["retry"].pop(sym, None)
            if isinstance(_popped, dict) and _popped.get("first_seen_ms") is not None:
                _retry_first = int(_popped["first_seen_ms"])   # (final gate) a later failure this pass keeps the original age
        start = sym_lo if cut is None else cut
        eps = qty_eps(fills)
        net0 = end_net - sum(f.signed for f in fills)
        ext = 0
        try:
            while cut is None and abs(net0) > eps and start > era_ms and ext < APP_MAX_EXTEND:   # anchor: read back until the replay starts flat
                nlo = max(era_ms, start - WINDOW_MS)
                more = await self._fills(sym, nlo, start - 1)
                have = {f.trade_id for f in fills}
                fills = [f for f in more if f.trade_id not in have] + fills
                start, ext = nlo, ext + 1
                eps = qty_eps(fills)
                net0 = end_net - sum(f.signed for f in fills)
        except AppTradeStop:
            raise
        except Exception as e:
            logger.warning(f"[APP_TRADE] {sym}: older fills unreadable ({e}) — retried next pass")
            retry_add(state, sym, sym_lo, tp, _retry_first)
            return
        if not fills:
            return
        episodes, tail, n_lead = build_episodes(sym, fills, net0, eps)
        if n_lead:
            logger.info(f"[APP_TRADE] {sym}: {n_lead} episode(s) already open at "
                        f"{'the era start' if start <= era_ms else ('a fill-mismatch cut' if cut else 'the read horizon')} — not recorded")
        # open-position partials (reconcile): Σ(realizedPnl − commission) of the open tail + its funding seen in this pass's income
        if tail is not None and not tail.partial:
            await self.prepare_rates({"BNB": [l.fill.t for l in tail.legs if l.fill.comm_asset == "BNB"]})
            try:
                c, _ = comm_usdt(tail.legs, self.rate)
                fund = sum(float(r.get("income") or 0) for r in sym_income if r.get("incomeType") == "FUNDING_FEE"
                           and int(r.get("time") or 0) >= tail.start_ms)
                state["open_partials"][sym] = {"trading": round(sum(l.rpnl for l in tail.legs) - c, 6), "funding": round(fund, 6),
                                               "since_ms": tail.start_ms}
            except ValueError:
                pass
        elif not end_net:
            state["open_partials"].pop(sym, None)
        if not episodes:
            return
        async with self.sf() as db:
            rows = await self._rows(db, sym, min(f.t for f in fills), tp)
            existing = await self._existing_keys(db, sym)
        sym_ids = {f.order_id for f in fills}
        to_add = []
        for ep in episodes:
            if ep.key in existing:
                state["ambiguous"].pop(ep.key, None); state["deferred"].pop(ep.key, None)
                if ep.key not in state["exact"]:               # (round-2 H) a row written by a pass that stopped before saving its state
                    v0 = attribute_episode(ep, rows, tp, sym_ids)
                    if v0.kind not in ("APP", "SHARE"):        # (round-3 5) the episode reads differently now → no exact entry invented
                        continue
                    await self.prepare_rates({"BNB": [l.fill.t for l in ep.legs if l.fill.comm_asset == "BNB"]})
                    try:
                        ep_net0 = sum(l.rpnl for l in ep.legs) - comm_usdt(ep.legs, self.rate)[0]
                    except ValueError:
                        ep_net0 = None
                    state["exact"][ep.key] = await self._exact(ep, v0.kind, float(existing[ep.key] or 0),
                                                               await self._ep_income(sym, ep, sym_lo, sym_income), ep_net0)
                continue
            v = attribute_episode(ep, rows, tp, sym_ids)
            if v.kind == "DEFER" and tp - ep.end_ms > APP_DEFER_MAX_DAYS * DAY_MS:
                v = Verdict("AMBIGUOUS", f"a bot row is still OPEN {APP_DEFER_MAX_DAYS} days after Binance went flat", v.members)
            await self.prepare_rates({"BNB": [l.fill.t for l in ep.legs if l.fill.comm_asset == "BNB"]})
            try:
                ep_net = sum(l.rpnl for l in ep.legs) - comm_usdt(ep.legs, self.rate)[0]
            except ValueError:
                ep_net = None
            if v.kind == "BOT":
                summary["bot"] += 1
                state["ambiguous"].pop(ep.key, None); state["deferred"].pop(ep.key, None)
                continue
            if v.kind == "DEFER":
                summary["deferred"] += 1
                state["deferred"][ep.key] = {"symbol": sym, "first_ms": ep.start_ms, "end_ms": ep.end_ms,
                                             "pnl": (round(ep_net - sum(float(r.pnl or 0) for r in v.members), 6) if ep_net is not None else None)}
                continue
            state["deferred"].pop(ep.key, None)
            fields, ex = None, None
            if v.kind in ("APP", "SHARE"):
                try:
                    ep_inc = await self._ep_income(sym, ep, sym_lo, sym_income)
                    fund = (self._sum_type([r for r in ep_inc if ep.start_ms <= int(r.get("time") or 0) <= ep.end_ms], "FUNDING_FEE")
                            if ep_inc is not None else None)   # funding: the tight window (round-3 2)
                    lev = await self._leverage(sym)
                    fields = build_row_fields(ep, v, self.rate, fund, lev, ms_to_dt(tp))
                    ex = await self._exact(ep, v.kind, fields["pnl"], ep_inc, ep_net)
                except AppTradeStop:
                    raise
                except ValueError as e:
                    v = Verdict("AMBIGUOUS", f"numbers not computable: {e}", v.members)
                # (round-2 C) a pure APP row is written ONLY when its income agrees to the cent; unreadable income → pending (retried)
                if fields is not None and v.kind == "APP" and ex["ok"] is None:
                    logger.info(f"[APP_TRADE] {sym} {ep.key}: income check pending (income unreadable) — not written, retried next pass")
                    retry_add(state, sym, ep.start_ms, tp, _retry_first)
                    continue
                if fields is not None and v.kind == "APP" and ex["ok"] is False:
                    v = Verdict("AMBIGUOUS", f"exact check failed: row pnl {ex['row_pnl']:+.4f} vs Binance income {ex['income_pnl']:+.4f} "
                                             f"(Δ {ex['row_pnl'] - ex['income_pnl']:+.4f})", v.members)
                    fields = None
            if fields is None:
                self._ambiguous(state, summary, ep, v, ep_net)
                continue
            state["ambiguous"].pop(ep.key, None)
            to_add.append((ep, v, fields, ex))
        if not to_add:
            return
        from models import Order
        from sqlalchemy.exc import IntegrityError
        async with self.sf() as db:
            for ep, v, fields, ex in to_add:
                db.add(Order(**fields))
            try:
                await self.commit(db)
            except IntegrityError:
                await db.rollback()                            # another pass wrote one of them first — insert one by one
                for ep, v, fields, ex in to_add:
                    async with self.sf() as db1:
                        db1.add(Order(**fields))
                        try:
                            await self.commit(db1)
                        except IntegrityError as ie:
                            await db1.rollback()
                            if "app_trade_key" not in str(ie):     # (round-3 6) not "already written" — a real write error: kept, never repeated silently
                                logger.warning(f"[APP_TRADE] {ep.symbol} {ep.key}: row NOT written — {ie}")
                                self._ambiguous(state, summary, ep, Verdict("AMBIGUOUS", f"write failed: {str(ie)[:200]}", v.members),
                                                fields["pnl"] + sum(float(r.pnl or 0) for r in v.members))   # back to the episode net
                            continue
                        self._logged(ep, v, fields, summary)
                        state["exact"][ep.key] = ex
                return
        for ep, v, fields, ex in to_add:
            self._logged(ep, v, fields, summary)
            state["exact"][ep.key] = ex

    def _logged(self, ep, v, f, summary):
        summary["recorded"] += 1
        summary["shares"] += 1 if v.kind == "SHARE" else 0
        logger.info(f"[APP_TRADE] 📱 {f['pair']} {f['direction']} recorded ({v.kind}{' with row(s) #' + ',#'.join(str(r.id) for r in v.members) if v.members else ''}): "
                    f"qty {f['quantity']:g} · entry {f['entry_price']:.6g} → exit {f['exit_price']:.6g} · pnl {f['pnl']:+.4f} USDT "
                    f"({f['pnl_percentage']:+.3f}%) · fees {f['total_fee']:.4f}{' (BNB price estimated)' if f['app_fee_estimated'] else ''} · "
                    f"funding {f['funding_fee_usd'] if f['funding_fee_usd'] is not None else 'n/a'} · {f['close_reason']} · "
                    f"{f['opened_at']:%Y-%m-%d %H:%M:%S} → {f['closed_at']:%H:%M:%S} UTC · key {f['app_trade_key']} · {f['app_fills']} fill(s)")

    def _ambiguous(self, state, summary, ep, v, ep_net):
        prev = state["ambiguous"].get(ep.key)
        state["ambiguous"][ep.key] = {"symbol": ep.symbol, "start_ms": ep.start_ms, "end_ms": ep.end_ms, "reason": v.reason,
                                      "pnl": (round(ep_net - sum(float(r.pnl or 0) for r in v.members if r.pnl is not None), 6)
                                              if ep_net is not None else None),
                                      "rows": [r.id for r in v.members]}
        if not prev or prev.get("reason") != v.reason:
            summary["ambiguous_new"] += 1
            logger.warning(f"[APP_TRADE_AMBIGUOUS] {ep.symbol} {ep.direction} {ms_to_dt(ep.start_ms):%Y-%m-%d %H:%M:%S} → "
                           f"{ms_to_dt(ep.end_ms):%H:%M:%S} UTC key {ep.key}: {v.reason} — NOTHING recorded (manual look). "
                           f"rows {[r.id for r in v.members]} · episode net {ep_net if ep_net is not None else 'n/a'} · fills: {ep.describe()}")

    async def _ep_income(self, sym, ep, lo, income) -> Optional[List[dict]]:
        """The symbol's income rows over [episode start, episode end]: from this pass's read when it covers the episode, else one paged,
        budgeted, ban-checked symbol read (review: no unbudgeted side read). None when unreadable (a 418 / 429 still stops the pass)."""
        a, b = ep.start_ms - FILL_PAD_MS, ep.end_ms + FILL_PAD_MS   # (round-3 2) ±2 min: an income row may land just after its fill;
        if a >= lo:                                                    # the exact sum scopes by tradeId, funding re-filters to [start, end]
            return [r for r in income if r.get("symbol") == sym and a <= int(r.get("time") or 0) <= b]
        try:
            return await self._income(max(0, a), b, symbol=sym)
        except AppTradeStop:
            raise
        except Exception as e:
            logger.warning(f"[APP_TRADE] {sym}: income over {ep.key} unreadable ({e}) — funding left empty, exact check pending")
            return None

    @staticmethod
    def _sum_type(rows, typ, rate=None) -> float:
        tot = 0.0
        for r in rows:
            if r.get("incomeType") != typ:
                continue
            x = float(r.get("income") or 0)
            if rate is not None and str(r.get("asset") or "USDT").upper() != "USDT":
                rr = rate(str(r.get("asset")).upper(), int(r.get("time") or 0))
                if rr is None:
                    raise ValueError(f"income asset {r.get('asset')} cannot be converted")
                x *= rr[0]
            tot += x
        return tot

    async def _leverage(self, sym) -> Optional[int]:
        if self._lev is None:
            try:
                self._lev = await self._call(W_SYMCFG, self.svc.get_symbol_leverages) or {}
            except AppTradeStop:
                raise
            except Exception as e:
                logger.warning(f"[APP_TRADE] symbol leverage unreadable ({e}) — recorded with leverage 1")
                self._lev = {}
        return self._lev.get(sym)

    async def _log_reconcile(self, state, era):
        try:
            async with self.sf() as db:
                rep = await reconcile_from_db(db, state, era)
            logger.info(f"[APP_TRADE_RECONCILE] since {rep['since']}: Binance {rep['binance']:+.2f} (trading {rep['binance_parts']['trading_net']:+.2f} "
                        f"+ funding {rep['binance_parts']['funding']:+.2f}) vs DB {rep['db']:+.2f} ({rep['db_parts']['closed_rows']} closed rows) "
                        f"→ gap {rep['gap']:+.2f} (trading {rep['gap_trading']:+.2f} · funding {rep['gap_funding']:+.2f}) · "
                        f"open partials {rep['gap_explained_by'][0]['usd']:+.2f} · ambiguous {rep['gap_explained_by'][2]['n']} "
                        f"({rep['gap_explained_by'][2]['usd']:+.2f}) · unexplained {rep['unexplained']:+.2f} (noise: bot rows' booked fees / exits) · "
                        f"📱 rows: {rep['app_rows_exact']['n_exact']} exact / {rep['app_rows_exact']['n_mismatched']} mismatched "
                        f"({', '.join('#' + str(i) for i in rep['app_rows_exact']['mismatched_ids']) or 'none'}) / "
                        f"{rep['app_rows_exact']['n_pending']} pending · {len(rep['app_rows_exact']['share_rows'])} app-share row(s) reported apart")
        except Exception as e:
            logger.warning(f"[APP_TRADE_RECONCILE] not computed: {e}")


async def reconcile_from_db(db, state: dict, era: Optional[datetime]) -> dict:
    """reconcile_report with the DB side read: Σ pnl / Σ funding_fee_usd of every live CLOSED row closed since the era start."""
    from sqlalchemy import select, func, and_
    from models import Order
    if era is None:
        return reconcile_report(state, None, 0.0, 0.0, 0, 0)
    pnl, fund, n = (await db.execute(select(func.coalesce(func.sum(Order.pnl), 0.0), func.coalesce(func.sum(Order.funding_fee_usd), 0.0),
                                            func.count(Order.id)).where(and_(Order.is_paper == False, Order.status == "CLOSED",  # noqa: E712
                                                                             Order.closed_at >= era)))).one()
    est = (await db.execute(select(func.count(Order.id)).where(and_(Order.is_paper == False, Order.app_fee_estimated == True,  # noqa: E712
                                                                    Order.import_source == IMPORT_SOURCE_APP_HISTORY)))).scalar() or 0
    k2i = {k: i for k, i in (await db.execute(select(Order.app_trade_key, Order.id).where(and_(
        Order.is_paper == False, Order.import_source == IMPORT_SOURCE_APP_HISTORY, Order.app_trade_key.isnot(None))))).all()}  # noqa: E712
    return reconcile_report(state, era, float(pnl or 0), float(fund or 0), int(n or 0), int(est), k2i)


async def app_trade_loop(should_stop: Callable[[], bool], recorder: Optional[AppTradeRecorder] = None):
    """Background task (started with the app's background tasks; LIVE only — each pass checks the mode and the kill switch). Never blocks the
    monitor / scan loops: its own task, its own sessions; any exception is logged and the job continues next cycle."""
    rec = recorder or AppTradeRecorder()
    await asyncio.sleep(APP_FIRST_DELAY_S)
    while not should_stop():
        try:
            await rec.run_pass()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[APP_TRADE] pass failed: {type(e).__name__}: {e} — next cycle in {APP_INTERVAL_S} s", exc_info=True)
        for _ in range(APP_INTERVAL_S):
            if should_stop():
                return
            await asyncio.sleep(1)
