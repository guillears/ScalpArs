"""💧 CAPITAL USE — "Withdrawable (caps)" (DECISION_LOG 507, operator 2026-11-09: "we need to find an indicator of 'unused money' for when
the caps start acting per sleeve, so I can remove that money and send it elsewhere instead of keeping it in the bot").

Every bot fill records the size it WANTED (entry_desired_notional, which follows the account) and the most the coin's volume allowed
(entry_liquidity_cap_notional — the ① per-pair cap, 0.1 % of 24 h volume, spike trades 0.5 %, the $500k hard ceiling folded in).
Over a rolling window (default 7 days, by open time) of CLOSED bot fills (MANUAL and probes out):

  RE-PRICED AT TODAY'S BALANCE — each wanted size × T(equity now) ÷ T(equity when it opened), T = the balance → tradeable line
  (reserve schedule), ratio clamped to [0.2, 5]. A coin's cap is a fixed share of ITS volume; only the wanted size follows the account,
  so a withdrawal is seen at once (no waiting 7 days for new trades).

  traded size        = Σ min(wanted, cap)        (what the bot actually puts on)
  WITHDRAWABLE (caps)= the most $ that can leave the account while the traded size falls ≤ 1 % (TOLERANCE). Every wanted size scales
                       with the shared pool, so a withdrawal shrinks the UNCAPPED fills one for one; it is free only on fills whose
                       wanted size stays above their cap. ≈ $0 while most fills are uncapped; it grows only once most trading is capped.
  trade-off          = withdraw 10 / 25 / 50 % of equity → the traded size falls by A / B / C %.
  per sleeve         = trades, capped, size kept (Σ min(want, cap) ÷ Σ want — information: a sleeve far below 100 % wants more
                       than its coins allow, a sizing question for that sleeve, not money to take out).
PURE — no I/O.
"""
from datetime import datetime, timedelta

WINDOW_DAYS = 7
EPS_USD = 1.0
TOLERANCE = 0.01                    # operator 11-09: "go with 1 %"
RATIO_MIN, RATIO_MAX = 0.2, 5.0     # a broken balance history can never blow a wanted size up / down more than this
TRADEOFF_STEPS = (0.10, 0.25, 0.50)


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _naive(t):
    return t.replace(tzinfo=None) if getattr(t, 'tzinfo', None) is not None else t


def sleeve_of(o) -> str:
    s = getattr(o, 'entry_strategy', None) or 'MOMENTUM'
    if s == 'MOMENTUM':
        return f"MOMENTUM_{getattr(o, 'direction', '') or ''}".rstrip('_')
    return str(s)


def is_counted(o) -> bool:
    """CLOSED bot fill — not MANUAL, not a probe."""
    if str(getattr(o, 'status', '') or '').upper() != 'CLOSED':
        return False
    if str(getattr(o, 'entry_strategy', '') or '').upper() == 'MANUAL' or str(getattr(o, 'confidence', '') or '').upper() == 'MANUAL':
        return False
    if str(getattr(o, 'cell_multiplier_source', '') or '').endswith('_PROBE'):
        return False
    return True


def equity_at_fn(equity_now, closed, flows=()):
    """→ f(t) = the account equity at time t: equity_now − Σ realised $ of the closed fills whose close is after t − Σ net flows after t.
    closed = [(closed_at, pnl $)], flows = [(time, +deposit / −withdrawal)]. Never below 1 % of equity_now (a guard, not a model). PURE."""
    e0 = _f(equity_now) or 0.0
    cl = [(_naive(c), _f(p) or 0.0) for c, p in closed if c is not None]
    fl = [(_naive(t), _f(a) or 0.0) for t, a in flows if t is not None]

    def f(t):
        t = _naive(t)
        e = e0 - sum(p for c, p in cl if c > t) - sum(a for ft, a in fl if ft > t)
        return max(e, 0.01 * e0) if e0 > 0 else e
    return f


def fills(orders, now=None, window_days=WINDOW_DAYS, equity_now=None, equity_at=None, tradeable_fn=None):
    """→ [(sleeve, wanted $ at today's balance, cap $ or None)] for the counted fills opened in the window. PURE."""
    now = _naive(now or datetime.utcnow())
    since = now - timedelta(days=window_days)
    tf = tradeable_fn or (lambda e: e)
    t_now = _f(tf(equity_now)) if equity_now else None
    out = []
    for o in orders:
        if not is_counted(o):
            continue
        t = getattr(o, 'opened_at', None)
        if t is None:
            continue
        t = _naive(t)
        if t < since:
            continue
        want = _f(getattr(o, 'entry_desired_notional', None))
        if not want or want <= 0:
            continue
        if t_now and t_now > 0 and equity_at is not None:
            t_open = _f(tf(equity_at(t)))
            if t_open and t_open > 0:
                want *= min(RATIO_MAX, max(RATIO_MIN, t_now / t_open))      # today's wanted size (the coin's cap does not move)
        cap = _f(getattr(o, 'entry_liquidity_cap_notional', None))
        out.append((sleeve_of(o), want, cap if (cap is not None and cap > 0) else None))
    return out


def _traded(fl, k=1.0):
    return sum(min(w * k, c) if c is not None else w * k for _, w, c in fl)


def size_loss(fl, k) -> float:
    """share of the traded size lost when every wanted size is scaled by k (clamped to 0..1 — a schedule tier edge can make T(E−X) > T(E))."""
    k = min(1.0, max(0.0, k))
    base = _traded(fl)
    return (base - _traded(fl, k)) / base if base > 0 else 0.0


def withdrawable(fl, equity_now, tradeable_fn=None, tol=TOLERANCE, steps=400, iters=50):
    """→ (withdrawable $, k) — the largest X such that EVERY withdrawal up to X keeps the traded-size loss ≤ tol (a scan, then bisection
    inside the first failing step: the reserve line can jump at a schedule tier, so the loss is not always monotone in X). PURE."""
    e = _f(equity_now) or 0.0
    tf = tradeable_fn or (lambda x: x)
    t_now = _f(tf(e)) or 0.0
    if e <= 0 or t_now <= 0 or not fl:
        return 0.0, 1.0
    loss = lambda x: size_loss(fl, max(0.0, tf(e - x)) / t_now)
    lo, hi = 0.0, None
    for i in range(1, steps + 1):
        x = e * i / steps
        if loss(x) > tol:
            hi = x
            break
        lo = x
    if hi is not None:
        for _ in range(iters):
            mid = (lo + hi) / 2
            if loss(mid) <= tol:
                lo = mid
            else:
                hi = mid
    return round(lo, 2), min(1.0, max(0.0, tf(e - lo)) / t_now)


def compute(orders, now=None, window_days=WINDOW_DAYS, equity_now=None, equity_at=None, tradeable_fn=None, tol=TOLERANCE) -> dict:
    """orders = Order rows (any) → {'window_days', 'tolerance_pct', 'rows', 'platform', 'withdrawable_usd', 'withdrawable_pct',
    'tradeoff': [{withdraw_pct, withdraw_usd, size_cut_pct}]}. Without equity_now nothing is re-priced and the $ fields are 0. PURE."""
    fl = fills(orders, now=now, window_days=window_days, equity_now=equity_now, equity_at=equity_at, tradeable_fn=tradeable_fn)
    agg = {}
    for sl, w, c in fl:
        a = agg.setdefault(sl, dict(trades=0, capped=0, want=0.0, traded=0.0))
        capped = c is not None and c < w - EPS_USD
        a['trades'] += 1
        a['capped'] += 1 if capped else 0
        a['want'] += w
        a['traded'] += min(w, c) if c is not None else w
    tot_traded = sum(a['traded'] for a in agg.values())
    rows = [dict(sleeve=sl, trades=a['trades'], capped=a['capped'],
                 capped_pct=round(100.0 * a['capped'] / a['trades'], 1) if a['trades'] else 0.0,
                 kept_pct=round(100.0 * a['traded'] / a['want'], 1) if a['want'] > 0 else 100.0,
                 traded_share_pct=round(100.0 * a['traded'] / tot_traded, 1) if tot_traded > 0 else 0.0)
            for sl, a in agg.items()]
    rows.sort(key=lambda r: (r['kept_pct'], -r['trades'], r['sleeve']))
    tot_want = sum(a['want'] for a in agg.values())
    e = _f(equity_now) or 0.0
    tf = tradeable_fn or (lambda x: x)
    wd, _k = withdrawable(fl, e, tf, tol) if e > 0 else (0.0, 1.0)
    t_now = (_f(tf(e)) or 0.0) if e > 0 else 0.0
    tradeoff = []
    if e > 0 and t_now > 0 and fl:
        for q in TRADEOFF_STEPS:
            k = max(0.0, tf(e * (1 - q))) / t_now
            tradeoff.append(dict(withdraw_pct=round(100 * q), withdraw_usd=round(e * q, 2), size_cut_pct=round(100.0 * size_loss(fl, k), 1)))
    return dict(window_days=window_days, tolerance_pct=round(100 * tol, 2), rows=rows, tradeoff=tradeoff,
                withdrawable_usd=wd, withdrawable_pct=round(100.0 * wd / e, 1) if e > 0 else 0.0, equity=round(e, 2),
                withdrawable_pct_tradeable=round(100.0 * wd / t_now, 1) if t_now > 0 else 0.0, tradeable_line=round(t_now, 2),
                platform=dict(trades=sum(a['trades'] for a in agg.values()), capped=sum(a['capped'] for a in agg.values()),
                              kept_pct=round(100.0 * tot_traded / tot_want, 1) if tot_want > 0 else 100.0))
