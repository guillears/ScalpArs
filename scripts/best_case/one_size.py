"""📏 ONE-SIZE daily compound for the current batch (DECISION_LOG 503, operator 2026-11-08: "this is a terrible mistake … normalize it, to
one size"). The account-path DCR (review_math.dcr_account: each fill's ACTUAL $ ÷ that morning's equity) let a mid-batch top-up and
mid-batch size changes flip a sleeve's sign — WILLY with WILLY_GAP lost −$798 on 13 kept trades (Σ −7.8 %) yet read +0.10 %/day,
because +$267 on a $2,900 morning outweighed −$760 on a $10k one.

ONE SIZE = every fill at today's size X (sleeve_x: median notional ÷ the account equity when it opened — the same X master uses), so a day's
return is r_d = Σ (fill % × X ÷ 100) over the fills CLOSED that UTC day, and DCR = (Π_d (1 + r_d))^(1/D) − 1 over EVERY calendar day D
from the batch's first open to its last close (0 on no-fill days) — the master method (best_dcr.m_dcr), so the two blocks compare
directly. The account path stays beside it as "as traded" (what the account really did), never as the line-picking number. PURE.
"""
import numpy as np
import pandas as pd


def frac(pct, x):
    """a fill's account fraction at one size: % × X ÷ 100 (NaN % → 0, an unpriced fill never adds)."""
    v = pd.to_numeric(pd.Series(pct), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return v * np.asarray(x, dtype=float) / 100.0


def dcr(closed_at, f, mask, t0, t1):
    """closed_at (per fill), f (per-fill account fraction), mask (bool per fill), batch span t0 → t1 → DCR %/day. PURE."""
    c = pd.to_datetime(pd.Series(closed_at).reset_index(drop=True), format="mixed", utc=True).dt.tz_convert(None).dt.normalize()
    m = np.asarray(mask, dtype=bool)
    if c[m].isna().any():   # (503 deep review) like dcr_account: a fill with no close time would silently drop out of the path
        raise ValueError(f"one_size.dcr: {int(c[m].isna().sum())} fill(s) without a closed_at")
    r = pd.Series(np.asarray(f, dtype=float)[m]).groupby(c[m].values).sum()
    def _naive(t):
        t = pd.Timestamp(t)
        return t.tz_convert(None) if t.tzinfo else t
    days = pd.date_range(_naive(t0).normalize(), _naive(t1).normalize(), freq="D")
    if not len(days):
        return float("nan")
    r = r.reindex(days).fillna(0.0)
    return float(((1.0 + r).prod() ** (1.0 / len(days)) - 1.0) * 100.0)



def _naive_series(v):
    return pd.to_datetime(pd.Series(v).reset_index(drop=True), format="mixed", utc=True).dt.tz_convert(None)


def equity_at_open(opened, all_closed_at, all_pnl, start, flows=()):
    """account equity when each fill opened, on the batch's account path: start + every fill's realised $ closed before the open +
    every flow before it (review_math.dcr_account's path). PURE."""
    o = _naive_series(opened).to_numpy()
    c = _naive_series(all_closed_at).to_numpy()
    u = pd.to_numeric(pd.Series(all_pnl).reset_index(drop=True), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fl = [(pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t).tz_convert(None), float(a)) for t, a in flows]
    order = np.argsort(c)
    cs, us = c[order], np.cumsum(u[order])
    out = []
    for t in o:
        k = int(np.searchsorted(cs, t, side="left"))
        out.append(float(start) + (us[k - 1] if k else 0.0) + sum(a for ft, a in fl if ft.to_datetime64() <= t))
    return np.asarray(out, dtype=float)


def sleeve_x(notional, equity, sleeve, leverage=None, cell_mult=None, willy_inv=1.0):
    """today's size of a sleeve as a fraction of the account (DECISION_LOG 503 + deep review): the median of each fill's notional ÷ the
    account equity when it opened (equity_at_open — the bot sizes on current equity, so this matches the config: LITE ≈ 2.0, RETRY ≈ 1.0,
    ML 0.25), WILLY / WILLY_HT re-scaled to today's 5× where a fill ran at 10×, WILLY also to today's margin (DL 487: × willy_inv ÷ the
    cell multiplier it traded at). One X for best_dcr.py, best_dcr_noflip.py (current batch AND master) and scout_frenzy_duration.py.
    NaN when no usable fill. PURE."""
    nv = pd.to_numeric(pd.Series(notional).reset_index(drop=True), errors="coerce")
    eq = pd.Series(np.asarray(equity, dtype=float))
    x = nv / eq.where(eq > 0)
    if sleeve in ("FRENZY_WILLY", "FRENZY_WILLY_HT"):
        if leverage is None:   # (503 review) a missing column must fail clearly, never on a shape mismatch
            raise ValueError(f"sleeve_x({sleeve}): the leverage column is required to re-scale 10× fills to today's 5×")
        lev = pd.to_numeric(pd.Series(leverage).reset_index(drop=True), errors="coerce")
        x = x * np.where(lev >= 10, 0.5, 1.0)   # today's 5× (WILLY DL 487 · HT 11-01)
    if sleeve == "FRENZY_WILLY":
        cm = pd.to_numeric(pd.Series(cell_mult).reset_index(drop=True), errors="coerce").fillna(1.0).replace(0, 1.0)
        x = x * float(willy_inv) / cm
    x = x.to_numpy(dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


def cap_factor(x, stop_pct, cap_pct=5.0):
    """the size factor that makes ONE stop cost cap_pct % of the account (DL 502 / 503): cap ÷ (X × stop %). The stop is the price line —
    a real stop books a little worse (fees + slip, e.g. LITE −3.02), so "one stop = 5 %" is slightly optimistic. NaN on a bad input. PURE."""
    try:
        x, st = float(x), float(stop_pct)
    except (TypeError, ValueError):
        return float("nan")
    return float(cap_pct) / (x * st) if x > 0 and st > 0 else float("nan")
