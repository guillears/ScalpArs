"""Route B signal-level loader — exit walkers on a 1-minute path (pure, no I/O; unit-tested in tests/test_route_b_signals.py).

Path = minute rows AFTER the entry: open_ms, o, h, l, c (numpy arrays). Conventions (conservative, the scouts' sim order):
  · a minute that touches both the stop and the TP → the STOP (intraminute order unknown);
  · stop = market order: a minute that OPENS beyond the level fills at the open; otherwise at the WORSE of the level and that minute's
    close (STOP_FILL, review 7) — a fast candle that closes through the stop is filled at the close. PESSIMISTIC bias on purpose: the bot
    polls every few seconds and fills near the level + slip, so real stops land between the level and this price (BLESS HT 11-20:
    level −4.0 · booked −4.9 · close-of-minute fill nearer the booked one); + taker fee + exit slip;
  · TP = resting maker limit at the level, filled only when price trades TP_THROUGH_PCT through it, at the level (no gap bonus), maker fee;
  · time cap = the close of the minute that ends at entry + hold (taker + slip);
  · funding settlements (fundingTime ms, rate, mark) with entry < fundingTime ≤ minute open are booked at that minute: P&L += −side ×
    rate × mark / entry × 100. FLATBASE's stop counts the funding PAID so far (DL 607): stop when move − paid ≤ −stop.
  · path ends before an exit → reason CENSORED (final False), marked at the last close.
"""
import math

import numpy as np

from common import MAKER_PCT, STOP_FILL, TAKER_PCT, TP_THROUGH_PCT


def walk(o, h, l, c, t, entry_px, side, tp_pct, sl_pct, hold_min, funding=None, stop_counts_funding=False, tp_counts_funding=False,
         entry_slip_pct=0.0, exit_slip_pct=0.0, taker_pct=None, maker_pct=None, through_pct=TP_THROUGH_PCT, entry_ms=None,
         stop_fill=STOP_FILL):
    """→ dict(reason, exit_px, exit_ms, minutes, final, gross_pct, fees_pct, slip_pct, funding_pct, net_pct, mfe_pct, mae_pct,
    min_to_mfe, min_to_mae). side +1 long / −1 short; tp_pct / sl_pct = PRICE distances from entry_px (> 0)."""
    o, h, l, c, t = (np.asarray(x, dtype=float) for x in (o, h, l, c, t))
    taker_pct = TAKER_PCT if taker_pct is None else taker_pct
    maker_pct = MAKER_PCT if maker_pct is None else maker_pct
    e = float(entry_px)
    out = dict(reason=None, exit_px=math.nan, exit_ms=math.nan, minutes=0, final=False, gross_pct=math.nan, fees_pct=math.nan,
               slip_pct=math.nan, funding_pct=0.0, net_pct=math.nan, mfe_pct=math.nan, mae_pct=math.nan, min_to_mfe=math.nan,
               min_to_mae=math.nan)
    if not (e > 0) or len(o) == 0 or not (tp_pct and tp_pct > 0 and sl_pct and sl_pct > 0):
        out["reason"] = "NO_PATH" if len(o) == 0 else "NO_LEVELS"
        return out
    tp_px = e * (1 + side * tp_pct / 100.0)
    tp_trig = tp_px * (1 + side * through_pct / 100.0)
    fund = sorted(funding or [], key=lambda x: x[0])
    fi, fcum, paid = 0, 0.0, 0.0
    t0 = entry_ms if entry_ms is not None else t[0]
    while fi < len(fund) and fund[fi][0] <= t0:          # settlements at or before the entry are not ours
        fi += 1
    mfe, mae, kmfe, kmae = -math.inf, math.inf, 0, 0
    n = min(len(o), int(hold_min))
    reason, xpx, k_exit = None, None, None
    for k in range(n):
        while fi < len(fund) and fund[fi][0] <= t[k]:
            ft, rate, mark = fund[fi]
            pay = -side * float(rate) * (float(mark) if mark and mark > 0 else e) / e * 100.0
            fcum += pay
            fi += 1
        paid = max(0.0, -fcum) if stop_counts_funding else 0.0
        if tp_counts_funding:   # 💸 (DECISION_LOG 653) the FLATBASE TP moves out by the funding paid (received never moves it)
            tp_px = e * (1 + side * (tp_pct + max(0.0, -fcum)) / 100.0)
            tp_trig = tp_px * (1 + side * through_pct / 100.0)
        sl_eff = sl_pct - paid
        sl_px = e * (1 - side * sl_eff / 100.0)
        fav = (h[k] / e - 1) * 100 * side if side > 0 else (l[k] / e - 1) * 100 * side
        adv = (l[k] / e - 1) * 100 * side if side > 0 else (h[k] / e - 1) * 100 * side
        if fav > mfe:
            mfe, kmfe = fav, k + 1
        if adv < mae:
            mae, kmae = adv, k + 1
        hit_sl = (l[k] <= sl_px) if side > 0 else (h[k] >= sl_px)
        hit_tp = (h[k] >= tp_trig) if side > 0 else (l[k] <= tp_trig)
        if hit_sl:
            gapped = (o[k] <= sl_px) if side > 0 else (o[k] >= sl_px)
            if gapped:
                px = o[k]
            elif stop_fill == "worse_of_level_and_close":
                px = min(sl_px, c[k]) if side > 0 else max(sl_px, c[k])
            else:
                px = sl_px
            reason, xpx, k_exit = "STOP_LOSS", px, k
            break
        if hit_tp:
            reason, xpx, k_exit = "TP", tp_px, k
            break
        if k + 1 >= hold_min:
            reason, xpx, k_exit = "MAX_HOLD_TIME", c[k], k
            break
    if reason is None:
        k_exit = n - 1
        reason, xpx = "CENSORED", c[k_exit]
    gross = (xpx / e - 1) * 100 * side
    maker = reason == "TP"
    fees = taker_pct + (maker_pct if maker else taker_pct)
    slip = float(entry_slip_pct or 0) + (0.0 if maker else float(exit_slip_pct or 0))
    out.update(reason=reason, exit_px=float(xpx), exit_ms=float(t[k_exit]), minutes=k_exit + 1, final=reason != "CENSORED",
               gross_pct=gross, fees_pct=fees, slip_pct=slip, funding_pct=fcum, net_pct=gross - fees - slip + fcum,
               mfe_pct=mfe, mae_pct=mae, min_to_mfe=kmfe, min_to_mae=kmae)
    return out


def funding_pct(funding, side, entry_px, a_ms, b_ms):
    """funding booked by a position of `side` opened at entry_px over settlements with a_ms < fundingTime ≤ b_ms (% of notional)."""
    tot = 0.0
    for ft, rate, mark in funding or []:
        if a_ms < ft <= b_ms:
            tot += -side * float(rate) * (float(mark) if mark and mark > 0 else entry_px) / entry_px * 100.0
    return tot


def forward_returns(o, c, side, horizons=(5, 15, 60, 240), t=None, funding=None):
    """direction-signed GROSS % from the entry minute's open to the close of minute h−1 after it; NaN when the path is shorter. With t
    (minute opens) and funding: also fwd_{h}m_funding_pct = the settlements booked inside that horizon (kept apart from the gross)."""
    o, c = np.asarray(o, dtype=float), np.asarray(c, dtype=float)
    res = {}
    for hz in horizons:
        ok = len(o) >= hz and o[0] > 0
        res[f"fwd_{hz}m_pct"] = ((c[hz - 1] / o[0] - 1) * 100 * side) if ok else math.nan
        if t is not None:
            res[f"fwd_{hz}m_funding_pct"] = funding_pct(funding, side, o[0], t[0], t[0] + hz * 60_000) if ok else math.nan
    return res


def grid(o, h, l, c, t, side, tps=(1, 2, 3, 5), sls=(1, 2, 3, 5), hold_min=240, entry_slip_pct=0.0, exit_slip_pct=0.0, funding=None):
    """fixed TP / SL grid (price %) from the entry minute's OPEN → {f'g_tp{a}_sl{b}_net': net %, …} (NaN when censored)."""
    res = {}
    if len(o) == 0:
        return res
    e = float(o[0])
    for a in tps:
        for b in sls:
            w = walk(o, h, l, c, t, e, side, a, b, hold_min, entry_slip_pct=entry_slip_pct, exit_slip_pct=exit_slip_pct, entry_ms=t[0],
                     funding=funding)
            res[f"g_tp{a}_sl{b}_net"] = w["net_pct"] if w["final"] else math.nan
    return res
