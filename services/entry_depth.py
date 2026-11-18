"""📖 (DECISION_LOG 551) ENTRY DEPTH GUARD — size a taker entry to the order book it will meet.

Operator 2026-11-15 (first live trade, SHELL FRENZY_LITE $16,575): the market order filled 0.32 % above the best ask on a thin book
(11 earlier small live LITE fills: never above 0.11 %). The guard reads the book in the same call the dislocation guard already makes
and keeps the position to what fills within `max_sweep_pct` of the best price: full size → unchanged · part fits (≥ min_frac) → shrink
to what fits · less than min_frac fits → skip. It only ever shrinks or skips; a missing / unreadable book is fail-open (unchanged).
Pure math here; the engine wiring lives in TradingEngine.open_position (the FRENZY / SURGE / FLATBASE taker path).
📖↔ (DECISION_LOG 565) every bot entry (momentum, flips and every other open_position sleeve too, entry_depth_guard_all_enabled) and the
EXIT side of the book (entry_depth_guard_exit_side_enabled): the notional is kept to the THINNER side — a thin exit side is where the
INIT 11-16 flip lost 0.37 % on its market close."""

FULL, SHRINK, SKIP = "FULL", "SHRINK", "SKIP"


def depth_fillable_usd(levels, direction, max_sweep_pct):
    """$ that fills within max_sweep_pct of the best price on the side a taker entry hits: LONG buys the asks (best first, rising),
    SHORT sells into the bids (best first, falling). levels = [[price, qty], …] best first. None when the book is unusable or the read
    never reached a level beyond the limit (a truncated read cannot tell a dense book from a thin one)."""
    try:
        if not levels or max_sweep_pct is None or float(max_sweep_pct) < 0:
            return None
        best = float(levels[0][0])
        if best <= 0:
            return None
        lim = best * (1 + float(max_sweep_pct) / 100.0) if direction == "LONG" else best * (1 - float(max_sweep_pct) / 100.0)
        tot = 0.0
        for lv in levels:
            p, q = float(lv[0]), float(lv[1])
            if (direction == "LONG" and p > lim + 1e-15) or (direction != "LONG" and p < lim - 1e-15):
                return tot
            if p > 0 and q > 0:
                tot += p * q
        return None   # every level read sits inside the push limit: the read was truncated, the depth is unmeasured (caveman review) → fail-open
    except (TypeError, ValueError, IndexError):
        return None


def depth_guard_size(planned_notional, fillable_usd, min_frac):
    """(action, notional): FULL (planned fits, or no usable book → fail-open) · SHRINK (min_frac ≤ fits < planned → what fits) ·
    SKIP (fits < min_frac × planned). Never returns more than planned."""
    n = float(planned_notional or 0.0)
    if n <= 0 or fillable_usd is None:
        return FULL, n
    f = max(0.0, float(fillable_usd))
    if f >= n:
        return FULL, n
    if f < max(0.0, float(min_frac or 0.0)) * n:
        return SKIP, f
    return SHRINK, f


def exit_side_fillable_usd(book, direction, max_sweep_pct):
    """📖↔ (DECISION_LOG 565) $ that the EXIT of this trade would fill within max_sweep_pct of the best price on the book read at entry: a LONG
    exits by selling into the bids, a SHORT by buying the asks. book = {'bids': [...], 'asks': [...]} best first. None when unusable."""
    try:
        if not book:
            return None
        return depth_fillable_usd(book.get('bids') if direction == "LONG" else book.get('asks'), "SHORT" if direction == "LONG" else "LONG", max_sweep_pct)
    except (AttributeError, TypeError):
        return None


def thinner_side_usd(entry_fill, exit_fill):
    """the depth the guard sizes to: the thinner of the two readable sides; None when neither side is measured (fail-open)."""
    v = [float(x) for x in (entry_fill, exit_fill) if x is not None]
    return min(v) if v else None


# ── 📖🧢 (DECISION_LOG 553) CAPPED ENTRY — the guarded order can never fill above the price the guard measured ─────────────────────────────
# Operator 2026-11-15 (INIT LITE: fill 0.32 % above the book the guard read, ~1 s later): the entry is a LIMIT order at the guard's ceiling
# (best price ± max_sweep_pct, rounded to the tick on the safe side); what fills at once fills, the rest rests up to wait_s seconds (fills
# only if the price comes back to the ceiling), then is cancelled — the trade keeps what filled. No fill at all = no trade.

import math as _math


def cap_price(best, direction, max_sweep_pct, tick):
    """the guard's ceiling as an order price: LONG best × (1 + p) rounded DOWN to the tick (never above the ceiling) · SHORT best × (1 − p)
    rounded UP. None when the inputs are unusable."""
    try:
        b, p, t = float(best), float(max_sweep_pct), float(tick or 0)
        if b <= 0 or p < 0:
            return None
        raw = b * (1 + p / 100.0) if direction == "LONG" else b * (1 - p / 100.0)
        if t <= 0:
            return raw
        steps = raw / t
        n = _math.floor(steps + 1e-9) if direction == "LONG" else _math.ceil(steps - 1e-9)
        px = n * t
        if direction == "LONG" and px < b:          # a ceiling below the best ask could never fill: at least the best price
            px = b
        if direction != "LONG" and px > b:
            px = b
        return px
    except (TypeError, ValueError):
        return None


def capped_fill_fee(immediate_qty, total_qty, avg_price, taker_rate, maker_rate):
    """entry fee of a capped entry: the part that filled at once crossed the book (taker), the part that filled while resting was maker."""
    q, q0, px = max(0.0, float(total_qty or 0)), max(0.0, float(immediate_qty or 0)), float(avg_price or 0)
    q0 = min(q0, q)
    return q0 * px * float(taker_rate or 0) + (q - q0) * px * float(maker_rate or 0)


# ── 📖📐 (DECISION_LOG 584) IMPACT-CAPPED DEPTH SIZING — size to an AVERAGE walk cap, not a fixed sweep band ───────────────────────────────
# reports/CAPACITY_2026-11-18.md: on 123 live fills the real slippage ≈ −0.023 + 0.89 × walk, walk = the VWAP of the book levels a taker
# order consumes vs the best price. Counting only levels inside a ±0.15 % band throws away the depth just beyond it that a large order
# would still fill at an acceptable AVERAGE price. Mode "impact" keeps the notional to the largest $ whose average walk ≤ the cap.
# Pure math: never raises, a bad book → None (the engine stays fail-open). LONG buys the asks (rising), SHORT sells into the bids (falling).

def _impact_scan(levels, direction, max_avg_impact_pct):
    """(usd, deepest_price, exhausted): the largest $ whose average fill sits ≤ max_avg_impact_pct from the best price, the deepest level
    price it touches, and whether every level read fit under the cap (the read ran out first → the $ is what the book SHOWS, a floor).
    (None, None, False) on an unusable book / cap."""
    try:
        if not levels or max_avg_impact_pct is None:
            return None, None, False
        cap = float(max_avg_impact_pct)
        if not (cap >= 0) or cap != cap:
            return None, None, False
        best = float(levels[0][0])
        if not (best > 0):
            return None, None, False
        long_ = direction == "LONG"
        tgt = best * (1 + cap / 100.0) if long_ else best * (1 - cap / 100.0)   # the average fill price the cap allows
        usd = qty = 0.0
        deepest = None
        for lv in levels:
            p, q = float(lv[0]), float(lv[1])
            if not (p > 0 and q > 0):
                continue
            n_all, q_all = usd + p * q, qty + q
            if (n_all / q_all <= tgt * (1 + 1e-12)) if long_ else (n_all / q_all >= tgt * (1 - 1e-12)):
                usd, qty, deepest = n_all, q_all, p                              # the whole level keeps the average inside the cap
                continue
            # part of this level: (usd + p·x) / (qty + x) = tgt → x = (tgt·qty − usd) / (p − tgt)
            x = (tgt * qty - usd) / (p - tgt) if p != tgt else 0.0
            if x > 0:
                usd, qty, deepest = usd + p * min(x, q), qty + min(x, q), p
            return usd, deepest, False
        return usd, deepest, True
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None, None, False


def impact_fillable_usd(levels, direction, max_avg_impact_pct):
    """📖📐 (584) (usd, deepest_price): the largest notional whose AVERAGE fill sits ≤ max_avg_impact_pct from the best price on the side a
    taker entry hits, and the deepest price level that notional touches. A book that runs out before the cap → everything it shows
    (conservative). (None, None) on an unusable book."""
    u, d, _ = _impact_scan(levels, direction, max_avg_impact_pct)
    return u, d


def impact_fillable_detail(levels, direction, max_avg_impact_pct):
    """impact_fillable_usd + `exhausted` (every level read fit under the cap: the engine reads deeper once, then keeps what the book shows)."""
    return _impact_scan(levels, direction, max_avg_impact_pct)


def exit_side_impact_fillable_usd(book, direction, max_avg_impact_pct):
    """📖📐↔ (584) impact_fillable_usd on the EXIT side of the book read at entry (a LONG exits into the bids, a SHORT into the asks).
    (None, None) when unusable."""
    try:
        if not book:
            return None, None
        return impact_fillable_usd(book.get('bids') if direction == "LONG" else book.get('asks'), "SHORT" if direction == "LONG" else "LONG",
                                   max_avg_impact_pct)
    except (AttributeError, TypeError):
        return None, None


def walk_detail(levels, direction, notional_usd):
    """(walk_pct, vwap, deepest_price) of sweeping the book for notional_usd: walk_pct = the average fill's distance from the best price in
    % (≥ 0, adverse), vwap = that average fill price, deepest = the last level touched. (None, None, None) when the book is unusable or
    too shallow to fill the notional. notional ≤ 0 → (0.0, best, best)."""
    try:
        if not levels:
            return None, None, None
        best = float(levels[0][0])
        if not (best > 0):
            return None, None, None
        n = float(notional_usd)
        if n != n:
            return None, None, None
        if n <= 0:
            return 0.0, best, best
        rem, qty = n, 0.0
        for lv in levels:
            p, q = float(lv[0]), float(lv[1])
            if not (p > 0 and q > 0):
                continue
            take = min(rem, p * q)
            qty += take / p
            rem -= take
            if rem <= max(n * 1e-9, 0.01):   # (584 review M2) float dust after the last level never reads as "too shallow"
                vwap = n / qty
                w = (vwap / best - 1) * 100.0 if direction == "LONG" else (1 - vwap / best) * 100.0
                return max(0.0, w), vwap, p
        return None, None, None
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None, None, None


def walk_vwap_pct(levels, direction, notional_usd):
    """📖📐 (584) the average fill price's distance from the best price (%, adverse ≥ 0) when a taker sweeps the book for notional_usd;
    None when the book is unusable or too shallow."""
    return walk_detail(levels, direction, notional_usd)[0]


def _tick_decimals(t):
    """decimals of a tick size (0.0001 → 4, 0.5 → 1, 1 → 0, 1e-07 → 7)."""
    try:
        from decimal import Decimal
        return max(0, -Decimal(repr(float(t))).normalize().as_tuple().exponent)
    except Exception:
        return 10


def impact_limit_price(deepest_price, direction, tick):
    """📖📐🧢 (584) the capped-entry LIMIT price in impact mode: the deepest level the walk touches for the final notional plus one tick
    toward the fill (LONG +tick, SHORT −tick), so that level fills in full even if it shifts by a tick; rounded to the tick's decimals.
    None when unusable."""
    try:
        d, t = float(deepest_price), float(tick or 0)
        if not (d > 0):
            return None
        if t <= 0:
            return d
        n = round(d / t)
        px = round((n + 1) * t if direction == "LONG" else (n - 1) * t, _tick_decimals(t))
        return px if px > 0 else d
    except (TypeError, ValueError):
        return None


def impact_cap_limit(levels, direction, deepest_price, tick, max_avg_impact_pct, sweep_cap_px=None):
    """📖📐🧢 (584, deep review C1) the live capped LIMIT in impact mode: impact_limit_price(deepest) CLAMPED to the marginal bound (best ±
    3 × the average cap, cap_price rounding — on a gapped book the deepest level can sit far beyond any acceptable price) and never TIGHTER
    than the DL 553 sweep ceiling (sweep_cap_px). The engine caps the impact fillable at the $ inside that bound, so the clamp never cuts the
    notional the walk assumed. None when unusable."""
    try:
        best = float(levels[0][0])
        imp = impact_limit_price(deepest_price, direction, tick)
        if imp is None or not (best > 0):
            return None
        ceil = cap_price(best, direction, 3.0 * float(max_avg_impact_pct), tick)
        if ceil is not None:
            imp = min(imp, ceil) if direction == "LONG" else max(imp, ceil)
        if sweep_cap_px:
            imp = max(float(sweep_cap_px), imp) if direction == "LONG" else min(float(sweep_cap_px), imp)
        return round(imp, _tick_decimals(tick)) if tick and float(tick) > 0 else imp
    except (TypeError, ValueError, IndexError):
        return None


# ── 📖📊 (DECISION_LOG 584, operator addition) CAPACITY STAMPS — what the book could take at every guarded entry, for batch reviews ───────────
BOOK_BANDS = (("02", 0.2), ("05", 0.5), ("1", 1.0))   # entry_book_usd_<k> / entry_book_exit_usd_<k>: cumulative $ within p % of the best price


def book_band_usd(levels, direction, pct):
    """cumulative $ of the levels within pct % of the best price on the side a `direction` taker hits (LONG = asks rising, SHORT = bids
    falling) — what the read SHOWS (a floor when the read ends inside the band). None on an unusable book."""
    try:
        if not levels:
            return None
        best = float(levels[0][0])
        if not (best > 0) or pct is None or float(pct) < 0:
            return None
        lim = best * (1 + float(pct) / 100.0) if direction == "LONG" else best * (1 - float(pct) / 100.0)
        tot = 0.0
        for lv in levels:
            p, q = float(lv[0]), float(lv[1])
            if (direction == "LONG" and p > lim + 1e-15) or (direction != "LONG" and p < lim - 1e-15):
                break
            if p > 0 and q > 0:
                tot += p * q
        return tot
    except (TypeError, ValueError, IndexError):
        return None


def book_bands(book, direction):
    """{entry_book_usd_02/_05/_1, entry_book_exit_usd_02/_05/_1} from ONE book read (entry side = what the entry hits, exit side = the
    other side). Values None when a side is unusable. Never raises."""
    out = {}
    try:
        es = (book or {}).get('asks' if direction == "LONG" else 'bids')
        xs = (book or {}).get('bids' if direction == "LONG" else 'asks')
    except AttributeError:
        es = xs = None
    xdir = "SHORT" if direction == "LONG" else "LONG"
    for k, p in BOOK_BANDS:
        e, x = book_band_usd(es, direction, p), book_band_usd(xs, xdir, p)
        out[f"entry_book_usd_{k}"] = round(e, 2) if e is not None else None
        out[f"entry_book_exit_usd_{k}"] = round(x, 2) if x is not None else None
    return out


def binding_side(planned, entry_usd, exit_usd):
    """'ENTRY' / 'EXIT' — the side whose fillable $ is below the plan and the thinner one (a tie → ENTRY) · 'NONE' — both sides hold the plan
    or neither was measured. Pure."""
    try:
        n = float(planned or 0)
    except (TypeError, ValueError):
        return "NONE"
    e = float(entry_usd) if entry_usd is not None else None
    x = float(exit_usd) if exit_usd is not None else None
    v = [s for s in (e, x) if s is not None]
    if not v or min(v) >= n:
        return "NONE"
    if x is None or (e is not None and e <= x):
        return "ENTRY"
    return "EXIT"


def _median(v):
    v = sorted(float(x) for x in v if x is not None and x == x)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


CAPACITY_FOCUS = ("EXHAUSTION_SHORT", "FLATBASE_SHORT")


def capacity_rows(fills, skips=None, focus=CAPACITY_FOCUS):
    """📖📊 (584) per-sleeve capacity table. fills = [dict(sleeve, planned, final, binding, walk, slip)] — guarded bot fills (planned
    stamped); skips = {sleeve: n depth-guard skips}. → [dict(sleeve, n, skipped, med_planned, med_final, shrunk_pct, skipped_pct,
    bind_entry, bind_exit, bind_none, med_walk, med_slip)], the focus sleeves first (listed even when empty), then the rest by N. Pure."""
    skips = dict(skips or {})
    by = {}
    for f in fills or []:
        try:
            if f.get('planned') is None:
                continue
            by.setdefault(str(f.get('sleeve') or '?'), []).append(f)
        except AttributeError:
            continue
    names = list(focus) + sorted((s for s in set(by) | set(skips) if s not in focus), key=lambda s: -len(by.get(s, [])))
    out = []
    for s in names:
        g = by.get(s, [])
        n, k = len(g), int(skips.get(s, 0) or 0)
        shr = sum(1 for f in g if f.get('final') is not None and f.get('planned') and float(f['final']) < float(f['planned']) * (1 - 1e-6))
        bm = [str(f.get('binding') or 'NONE').upper() for f in g]
        out.append(dict(sleeve=s, n=n, skipped=k, med_planned=_median([f.get('planned') for f in g]), med_final=_median([f.get('final') for f in g]),
                        shrunk_pct=(round(100.0 * shr / n, 1) if n else None),
                        skipped_pct=(round(100.0 * k / (n + k), 1) if (n + k) else None),
                        bind_entry=bm.count('ENTRY'), bind_exit=bm.count('EXIT'), bind_none=bm.count('NONE'),
                        med_walk=_median([f.get('walk') for f in g]), med_slip=_median([f.get('slip') for f in g])))
    return out
