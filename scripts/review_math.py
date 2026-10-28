#!/usr/bin/env python3
"""Shared review math (DECISION_LOG 406 — the analysis methodology, operator 2026-10-28). PURE, no I/O.

  loo(deltas, labels)        — 🧮 LEAVE-ONE-OUT: every before → after Δ is shown with and without its single biggest contributor
                               (the CFX lesson: WILLY no-late master +3.33 pts, −0.56 without CFX alone).
  loo_text(...)              — the one-line form printed under every before → after table.
  dcr_account(...)           — 📈 DCR-cal (CLAUDE.md "How to evaluate" step 3, DL 386 definition) on the REAL account path: r_d = the
                               sleeve's realised day $ ÷ start-of-day account equity (batch start + every sleeve's realised $ so far),
                               a flow day split at the flow, 0 % on no-fill days, every calendar day of the span. $ per fill = % × the
                               fill's ACTUAL notional (the leverage it really got after the exchange's bracket cap — never the intended one).
  fill_usd(pct, notional)    — that $ conversion.
"""
import math

import pandas as pd


def loo(deltas, labels=None):
    """→ dict(total, n, top_label, top_delta, without_top, top_share) — top = the largest |Δ| contributor in the direction of the total
    (the one a reader would worry carries the result). share = top Δ ÷ total (> 0.5 = one trade carries the result)."""
    d = pd.Series(list(deltas), dtype=float)
    lab = list(labels) if labels is not None else [str(i) for i in range(len(d))]
    if len(lab) != len(d):
        raise ValueError(f"loo: {len(d)} deltas vs {len(lab)} labels")
    nan = int(d.isna().sum())
    d = d.fillna(0.0)
    tot = float(d.sum())
    if not len(d) or not d.abs().sum():
        return dict(total=tot, n=len(d), top_label=None, top_delta=0.0, without_top=tot, top_share=0.0, nan=nan)
    sgn = 1.0 if tot >= 0 else -1.0
    i = int((d * sgn).values.argmax())
    top = float(d.iloc[i])
    share = (top / tot) if tot else float("inf")
    return dict(total=tot, n=len(d), top_label=lab[i], top_delta=top, without_top=tot - top, top_share=share, nan=nan)


def loo_text(deltas, labels=None, unit="pts"):
    r = loo(deltas, labels)
    if r["top_label"] is None:
        return f"Δ {r['total']:+.2f} {unit} (no changed fill)"
    sh = "∞" if not math.isfinite(r["top_share"]) else f"{r['top_share']:.0%}"
    flag = " ⚠ one fill carries ≥ 50 % of the result" if (not math.isfinite(r["top_share"]) or r["top_share"] >= 0.5) else ""
    return (f"Δ {r['total']:+.2f} {unit} · without the biggest ({r['top_label']} {r['top_delta']:+.2f}, {sh} of Δ): "
            f"{r['without_top']:+.2f} {unit}{flag}" + (f" · {r['nan']} unpriced (counted 0)" if r.get("nan") else ""))


def fill_usd(pct, notional):
    try:
        return float(pct) / 100.0 * float(notional)
    except (TypeError, ValueError):
        return float("nan")


def dcr_account(account_fills, sleeve_mask, usd_col, start_equity, span_start, span_end, flows=()):
    """DCR-cal %/day of ONE sleeve on the account path.
    account_fills: every closed fill of the batch (all sleeves) with closed_at + `usd_col` ($, the after-path for the sleeve when
    re-priced, as traded for the rest); sleeve_mask: bool Series selecting the sleeve; flows: [(timestamp, amount)] deposits / top-ups.
    Equity starts at start_equity, moves with EVERY fill's $ (booked on its close) and every flow; r_d = the sleeve's $ that day ÷ the
    equity at the start of that day (a flow day: (1 + pre / E_start)(1 + post / (E_at_flow + flow)) − 1). → %/day."""
    def _naive(x):   # review: '+00:00' exports → tz-aware; everything is compared naive UTC
        return pd.to_datetime(x, format="mixed", utc=True).dt.tz_convert(None)
    f = account_fills.assign(_t=_naive(account_fills.closed_at).values, _s=sleeve_mask.values,
                             _u=pd.to_numeric(account_fills[usd_col], errors="coerce").fillna(0.0))
    n_nat = int(f._t.isna().sum())
    if n_nat:
        raise ValueError(f"dcr_account: {n_nat} fill(s) without a closed_at — the account path would be wrong")
    def _ts(v):
        v = pd.Timestamp(v)
        return v.tz_convert(None) if v.tzinfo else v
    fl = sorted((_ts(t), float(a)) for t, a in flows)
    span_start, span_end = _ts(span_start), _ts(span_end)
    d0, d1 = pd.Timestamp(span_start).normalize(), pd.Timestamp(span_end).normalize()
    days = pd.date_range(d0, d1, freq="D")
    eq, g = float(start_equity), 1.0
    for day in days:
        nxt = day + pd.Timedelta(days=1)
        ev = [(t, "fill", u, s) for t, u, s in zip(f._t, f._u, f._s) if day <= t < nxt]
        ev += [(t, "flow", a, False) for t, a in fl if day <= t < nxt]
        ev.sort(key=lambda x: x[0])
        seg_start, seg_sleeve, r = eq, 0.0, 1.0
        for t, kind, amt, is_s in ev:
            if kind == "flow":
                r *= 1 + (seg_sleeve / seg_start if seg_start > 0 else 0.0)
                eq += amt
                seg_start, seg_sleeve = eq, 0.0
            else:
                eq += amt
                if is_s:
                    seg_sleeve += amt
        r *= 1 + (seg_sleeve / seg_start if seg_start > 0 else 0.0)
        g *= r
    return ((g ** (1.0 / len(days)) - 1.0) * 100.0) if (len(days) and g > 0) else -100.0
