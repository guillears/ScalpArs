#!/usr/bin/env python3
"""🧺 Shared-book capacity for review / study simulations (DECISION_LOG 542 follow-up, operator 2026-11-13: "fix the review scripts now").

Since DL 542 no bot sleeve has its own open-position limit that binds (every *_max_slots = 10 = the book count ceiling). What still stops
a trade is the SHARED book, exactly as services/trading_engine.open_position sizes it:
  ① COUNT  — open bot positions ≥ the count ceiling (max_open_positions_hard while redeploy_leftover_enabled, else max_open_positions)
             → refused BOOK_FULL ("Max open positions (10) reached"). MANUAL rows live in their own lane and are not counted.
  ② MARGIN — every fill's margin = the equal-split ticket ((equity − reserves) ÷ max_open_positions) × its sleeve's invest multiplier,
             clamped to what is left of the tradeable balance. In TICKET units the book holds max_open_positions (4) tickets: a fill gets
             min(its ticket, the free margin); nothing free → refused NO_BALANCE. A partial leftover fill is real (the engine invests "all
             available"); its size is returned so a $ figure can be scaled.
Not modelled (stated, never guessed): each row's OWN opening multiplier (today's is used), FLIP / probe / premium-cell sizing on MOMENTUM
rows, the 25× gross-notional cap, the min_investment_size clamp ($100 floor — pass min_frac when the ticket
in $ is known), open P&L moving the free balance, MANUAL positions' margin.
PURE — no I/O except limits(), which reads trading_config.json when no dict is passed."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-9

# sleeve (entry_strategy) → the threshold key of its invest multiplier (the margin share of one ticket), read with the ENGINE's own bounds
# (services/trading_engine open_position): SLEEVE_KEY rows (the _surge sleeve block: SURGE / FRENZY / FLATBASE) = max(0.1, min(raw, the
# rsi_adx_multiplier_hard_cap)), blank → 1.0, an explicit 0 → the 0.1 floor; OWN_KEY rows (Bull / Bear-Run, Spike) = max(0.1, raw or default).
SLEEVE_KEY = {
    "FRENZY_LONG": "frenzy_long_invest_mult", "FRENZY_WIDE": "frenzy_wide_invest_mult", "FRENZY_LITE": "frenzy_lite_invest_mult",
    "FRENZY_LITE_RETRY": "frenzy_lite_retry_invest_mult", "FRENZY_WILLY": "frenzy_willy_invest_mult", "FRENZY_WILLY_HT": "frenzy_willy_invest_mult",
    "FLATBASE_SHORT": "flatbase_invest_mult",
}
OWN_KEY = {
    "BULLRUN_LONG": ("bullrun_invest_mult", 1.0), "BULLRUN": ("bullrun_invest_mult", 1.0),
    "BEARRUN_SHORT": ("bearrun_invest_mult", 1.0), "BEARRUN": ("bearrun_invest_mult", 1.0),
    "SPIKE_FADE": ("spike_fade_invest_mult", 1.0), "SPIKE_BOUNCE": ("spike_bounce_invest_mult", 1.0), "SPIKE_CHASE": ("spike_invest_mult", 2.0),
}
HT_FRAC_DEF = 0.25   # services.frenzy.WILLY_HT_FRAC_DEF — a blank / out-of-range frenzy_willy_ht_size_frac

def _f(v, d):
    try:
        x = float(v)
        return x if x == x else d
    except (TypeError, ValueError):
        return d


def _cfg(cfg):
    if cfg is None:
        with open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    return cfg


def limits(cfg=None) -> dict:
    """→ {count, margin}: the book's count ceiling and its margin in tickets, from trading_config.json (or the dict passed)."""
    inv = _cfg(cfg).get("investment") or {}
    m = max(1, int(_f(inv.get("max_open_positions"), 4) or 4))
    hard = max(m, int(_f(inv.get("max_open_positions_hard"), m) or m))
    return dict(count=hard if inv.get("redeploy_leftover_enabled") else m, margin=float(m))


def ticket(sleeve, th=None, direction=None) -> float:
    """one fill's margin in tickets, as the engine sizes it (see SLEEVE_KEY / OWN_KEY). th = the thresholds dict or object. A blank / NaN
    entry_strategy = a legacy MOMENTUM row. MOMENTUM LONG × momentum_long_invest_mult; FLIP / probes / premium cells on MOMENTUM rows are NOT
    modelled (1.0) — and every row is read at TODAY's multipliers, never the one it opened with (stated limit)."""
    raw = (lambda k: (th or {}).get(k)) if isinstance(th, dict) or th is None else (lambda k: getattr(th, k, None))
    s = str(sleeve).upper() if sleeve is not None and str(sleeve).strip().lower() not in ("", "nan", "none") else "MOMENTUM"
    cap = _f(raw("rsi_adx_multiplier_hard_cap"), 2.0)

    def sleeve_mult(key):
        v = raw(key)
        return max(0.1, min(1.0 if v is None or v == "" else _f(v, 1.0), cap))
    if s.startswith("SURGE"):
        side = "short" if ("SHORT" in s or str(direction or "").upper() == "SHORT") else "long"
        return sleeve_mult(f"surge_{side}_invest_mult")
    if s in SLEEVE_KEY:
        v = sleeve_mult(SLEEVE_KEY[s])
        if s == "FRENZY_WILLY_HT":
            f = _f(raw("frenzy_willy_ht_size_frac"), HT_FRAC_DEF)
            v *= f if 0 < f <= 1.0 else HT_FRAC_DEF
        return v
    if s in OWN_KEY:
        k, d = OWN_KEY[s]
        return max(0.1, _f(raw(k), d) or d)
    if s == "MOMENTUM" and str(direction or "").upper() == "LONG":
        v = _f(raw("momentum_long_invest_mult"), 1.0)            # services.trading_engine.momentum_long_inv_factor: ≤ 0 → 1.0, [0.05, 1]
        return 1.0 if v <= 0 else min(1.0, max(0.05, v))
    return 1.0

def check(open_tickets, want, lim, min_frac=0.0):
    """open_tickets = the margins (tickets) of the bot positions open right now (MANUAL excluded); want = this fill's ticket.
    → (size in tickets, None) when it opens (size ≤ want: a leftover fill is partial), else (0.0, "BOOK_FULL" | "NO_BALANCE")."""
    op = [float(x) for x in (open_tickets or [])]
    if len(op) >= int(lim["count"]):
        return 0.0, "BOOK_FULL"
    free = float(lim["margin"]) - sum(op)
    size = min(float(want), free)
    if size <= EPS or size + EPS < float(min_frac):
        return 0.0, "NO_BALANCE"
    return round(size, 9), None


def fills_on_empty_book(want, lim, cap=None) -> int:
    """how many fills of one ticket size one event can open on an EMPTY book (count ceiling, margin, an optional sleeve cap) — a partial
    leftover fill counts as one."""
    op, n = [], 0
    while cap is None or n < int(cap):
        s, why = check(op, want, lim)
        if why:
            break
        op.append(s)
        n += 1
    return n
