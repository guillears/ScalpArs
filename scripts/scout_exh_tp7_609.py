#!/usr/bin/env python3
"""🎯🔒 Scout — EXH_TP7 revert gate (DECISION_LOG 609, operator 2026-11-21 "build 2 with TP 7", operator-declared; ALERT-only, never
changes config, public 1m klines only).

CHANGE     EXHAUSTION_SHORT take profit cap 10 → 7 % (exhaustion_tp_cap_pct 7: TP min(20 × ATR, 7 %); SL min(1.5 × ATR, 6 %) and 24 h
           unchanged). Evidence reports/TP_GRID_2026-11-20_EXH.md (FRENZY ∪ WIDE 4,307 fills, today's filters): TP 7 Δ −0.06 %/fill vs 10
           (CI −0.16 … +0.03), daily compound 0.95 vs 0.97 %/day, max drawdown −58 vs −72 %, worst day −28 vs −40 %, ≈ 40 % shorter holds.
           A risk / hold-time choice at ≈ equal return — NOT an edge.
QUESTION   is TP 7 shown WORSE than TP 10 on real fills?
COHORT     CLOSED EXHAUSTION_SHORT fills (both universes) opened ≥ the DECISION_LOG 609 commit + 10 min (orders exports, dedup, CLOSED wins,
           operator-closed out). Each is walked on its own 1m path from the REAL fill price (scout_dist_tp_shadow.walk_x: from the minute after
           the fill, the fill's own stop first, maker TP ≥ 0.05 % through, 24 h, modelled slip on taker exits, entry leg per DL 584) at
           TP 7 = the row's own level (services.exhaustion.row_levels) and TP 10 = min(20 × the stamped ATR, 10 %). No funding on either
           side (stated: TP 10 holds longer → funding would favour TP 7). Δ = net(TP 7) − net(TP 10). Parity = the TP 7 walk vs the booked %.
🔒 GATE     frozen ONCE (reports/SCOUT_EXH_TP7_609.json, gitignored) at the first 20 priced fills (open-time prefix) spanning ≥ 5 UTC days:
           the one-sided 90 % upper bound of mean Δ (day-block bootstrap, 2,000, seed 609) < 0 → REVERT (alert: exhaustion_tp_cap_pct back
           to 10 — operator decides) · else KEEP. A plain "Σ Δ < 0" would flag ≈ 50 % of the time when the two TPs truly tie (per-fill Δ SD
           ≈ 4 pts vs a ≈ 0 true gap); the upper bound flags only when TP 7 is shown worse. Freezing waits while a kline fetch stopped this run
           or the newest orders export is < the 20th fill + 24 h. A path still incomplete 48 h after entry → NOT_EVALUABLE (out of the read).
💸 FUNDING   (DECISION_LOG 613, information only — never part of the verdict): per fill the BOOKED funding paid (funding_fee_usd as % of the
           entry notional; negative = paid) and the hold (closed_at − opened_at), and their averages over the counted fills — to check on
           real fills that the shorter TP-7 holds cut the funding cost (the walks above carry no funding; TP 10 holds longer → would pay more).
Usage: venv/bin/python -I scripts/scout_exh_tp7_609.py"""
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import scout_gatekit as _GK          # noqa: E402
import scout_exh_wide_585 as _W      # noqa: E402  (shared orders loader)
import scout_long_stop4 as LS        # noqa: E402  (budgeted, cached public fapi klines)
import scout_dist_tp_shadow as _DT   # noqa: E402  (walk_x, fetch_bars, newest_export)
import scout_exh_exit_override as XO  # noqa: E402  (upper_bound: day-block one-sided bound)
from services.exhaustion import EXHAUSTION_SHORT, row_levels, entry_leg_slip, exit_era_609   # noqa: E402
from services.paper_costs import funding_pct_of   # noqa: E402  💸 (613) the Closed Orders ruler (entry × (qty + partial_closed_qty))

DL = 609
KEY = "EXH_TP7"
STATE = os.path.join(_R, "reports", "SCOUT_EXH_TP7_609.json")
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "pnl_percentage", "close_reason", "entry_price", "entry_exh_tp_pct",
        "entry_exh_sl_pct", "entry_exh_atr_pct", "entry_exh_model_slip_pct", "entry_depth_mode", "entry_depth_walk_pct", "entry_exh_universe",
        "funding_fee_usd", "quantity", "partial_closed_qty", "closed_at")   # 💸 (613) the funding read
TP_OLD_ATR, TP_OLD_CAP = 20.0, 10.0
N_GATE, DAYS_MIN, UPPER_Q, BOOT_N, SEED = 20, 5, 0.90, 2000, 609
HOLD_H, STALE_H, DAY, HOUR = 24, 48, 86_400_000, 3_600_000


def funding_read(funding_usd, entry_px, qty, opened, closed, partial_qty=None):
    """💸 (613) → (funding % on the SAME base as pnl_percentage / the Closed Orders Funding % (negative = paid) or None,
    hold hours or None). PURE, never raises."""
    pct = funding_pct_of(funding_usd, entry_px, qty, partial_qty)
    try:
        h = (pd.Timestamp(str(closed)[:19]) - pd.Timestamp(str(opened)[:19])).total_seconds() / 3600.0
        h = h if h >= 0 else None      # NaT → the comparison is False → None
    except Exception:
        h = None
    return pct, h


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def tp_old(atr_pct):
    """TP 10's level for this fill: min(20 × ATR, 10 %); None on a bad ATR stamp. PURE."""
    a = _f(atr_pct)
    return round(min(TP_OLD_ATR * a, TP_OLD_CAP), 6) if a and a > 0 else None


def gate(rows):
    """rows = [(day, Δ)] priced prefix in open-time order → (state, text). PURE. (gate_k gives the cut length.)"""
    return gate_k(rows)[:2]


def gate_k(rows):
    """→ (state, text, k): k = the length of the deciding prefix (≥ N_GATE fills spanning ≥ DAYS_MIN days), None while COLLECTING. PURE."""
    for k in range(N_GATE, len(rows) + 1):
        if len({d for d, _ in rows[:k]}) >= DAYS_MIN:
            cut = rows[:k]                  # the first prefix with ≥ N_GATE fills spanning ≥ DAYS_MIN days
            m = float(np.mean([x for _, x in cut]))
            ub = XO.upper_bound(cut, q=UPPER_Q, n_boot=BOOT_N, seed=SEED)
            info = f"first {len(cut)} fills on {len({d for d, _ in cut})} days: mean Δ (TP 7 − TP 10) {m:+.2f} pts · {UPPER_Q:.0%} upper bound {ub:+.2f}"
            return (("REVERT", info + " — TP 7 shown worse: flag cap back to 10 (operator decides)", k) if ub < 0 else
                    ("KEEP", info + " — TP 7 not shown worse: keep", k))
    n = len(rows)
    return "COLLECTING", (f"⏳ {n}/{N_GATE} fills on {len({d for d, _ in rows})}/{DAYS_MIN} days" +
                          (f" · running mean Δ {np.mean([x for _, x in rows]):+.2f} (not judged)" if n else "")), None


def report(frame=None, since_ms=None, state_path=STATE, now_ms=None, bars_fn=None, newest=None, th=None):
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🎯🔒 EXH_TP7 revert gate (DECISION_LOG {DL}, operator-declared) — EXHAUSTION TP 7 vs TP 10 on real fills "
         "(1m walk from the real fill price, own stop first, TP ≥ 0.05 % through, 24 h, modelled slip, no funding)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="EXH_TP7")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    # services.exhaustion reads thresholds by attribute (getattr) — a plain dict would fall back to the study defaults (cap 5)
    th = th if th is not None else SimpleNamespace(**_DT._thresholds())
    a = _W.load(sleeve=EXHAUSTION_SHORT, cols=COLS) if frame is None else frame
    a = a[a.o >= pd.Timestamp(int(since_ms), unit="ms")] if len(a) else a
    try:
        with open(state_path) as fh:
            st = json.load(fh)
    except FileNotFoundError:
        st = {}
    except (OSError, ValueError):
        st = None
    rec = st.get(KEY) if isinstance(st, dict) else None
    bud, rows, opens, par, stopped, wick, fund = LS.Budget(), [], [], [], False, [], []
    for r in a.itertuples():
        if not exit_era_609(getattr(r, "entry_exh_tp_pct", None), getattr(r, "entry_exh_atr_pct", None), r.o):
            L.append(f"  · {r.pair} {r.o:%m-%d %H:%M} → not TP-7 era (stamp {getattr(r, 'entry_exh_tp_pct', None)}; opened inside the deploy lag) — out")
            continue
        E, slip = _f(r.entry_price), _f(getattr(r, "entry_exh_model_slip_pct", None)) or 0.0
        t7, sl = row_levels(r, th)
        t10 = tp_old(getattr(r, "entry_exh_atr_pct", None))
        es = entry_leg_slip(slip, getattr(r, "entry_depth_mode", None), _f(getattr(r, "entry_depth_walk_pct", None)))
        te = int(r.o.value // 1_000_000)
        tag = f"{r.pair} {r.o:%m-%d %H:%M}"
        if t10 is None:
            L.append(f"  · {tag} → NOT_EVALUABLE (no ATR stamp for the TP 10 level) — out of the read")
            continue
        bars = bars_fn(r.pair, te, now_ms) if bars_fn else _DT.fetch_bars(r.pair, te, now_ms, bud)
        stopped |= bars is None
        w7, v7, _ = _DT.walk_x(bars, E, te, t7, sl, slip, es) if bars else ("UNPRICED", np.nan, None)
        w10, v10, _ = _DT.walk_x(bars, E, te, t10, sl, slip, es) if bars else ("UNPRICED", np.nan, None)
        if v7 != v7 or v10 != v10:
            if bars is not None and now_ms > te + STALE_H * HOUR:
                L.append(f"  · {tag} → NOT_EVALUABLE (no complete price path) — out of the read")
                continue
            L.append(f"  · {tag} booked {_f(r.pnl_percentage) or 0:+.2f} → pending / unpriced (holds the read)")
            break                           # open-time prefix: the read waits at the first unpriced fill
        if _f(r.pnl_percentage) is not None:
            par.append(abs(v7 - _f(r.pnl_percentage)))
        rows.append((te // DAY, v7 - v10))
        opens.append(r.o)
        fpct, hold_h = funding_read(getattr(r, "funding_fee_usd", None), E, getattr(r, "quantity", None), r.opened_at, getattr(r, "closed_at", None),
                                    getattr(r, "partial_closed_qty", None))
        fund.append((fpct, hold_h))
        wick.append(w7 == "TP" and w10 == "STOP")   # (deep review #4) TP 7 booked, the TP-10 replica stopped — may be a wick live rode through
        L.append(f"  · {tag} booked {_f(r.pnl_percentage) or 0:+.2f} · TP {t7:g} {w7} {v7:+.2f} · TP {t10:g} {w10} {v10:+.2f} · Δ {v7 - v10:+.2f}"
                 + (f" · funding {fpct:+.2f} %" if fpct is not None else "") + (f" · held {hold_h:.1f} h" if hold_h is not None else ""))
    v, txt, k = gate_k(rows)
    last_o = opens[k - 1] if k else None
    L.append(f"  cohort: CLOSED fills opened ≥ {pd.Timestamp(int(since_ms), unit='ms'):%Y-%m-%d %H:%M} UTC · priced {len(rows)} on "
             f"{len({d for d, _ in rows})} days" + (f" · parity TP 7 walk vs booked: median |diff| {np.median(par):.2f}" if par else ""))
    hold = []
    if stopped:
        hold.append("a kline fetch stopped this run")
    newest = _DT.newest_export() if newest is None else newest
    if v != "COLLECTING" and (newest is None or last_o is None or newest < last_o + pd.Timedelta(hours=HOLD_H)):
        hold.append("newest export < the deciding prefix's last fill + 24 h")
    if st is None:
        L.append("⚠ state file corrupt — operator restore needed; not frozen this run.")
    elif rec is None and v != "COLLECTING" and not hold:
        rec = dict(verdict=v, detail=txt, run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
        st[KEY] = rec
        try:
            tmp = state_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(st, fh, indent=1)
            os.replace(tmp, state_path)
        except OSError as e:
            L.append(f"⚠ state not saved ({e})")
    if rec:
        L.append(f"**FROZEN ({rec.get('run_at')}): {rec.get('verdict')}** — {rec.get('detail')}")
    else:
        L.append(f"Now: {v} — {txt}" + (" · ⏸ freezing deferred: " + " · ".join(hold) if hold else ""))
    fset = fund[:k] if k else fund        # the deciding prefix once a verdict is reached, else every priced fill
    fp = [x for x, _ in fset if x is not None]
    hh = [h for _, h in fset if h is not None]
    if fp or hh:   # 💸 (613) information only — never part of the verdict
        who = f"the {k} deciding fills" if k else f"the {len(fset)} priced fills"
        L.append(f"  💸 funding on {who} (booked, information only): avg {np.mean(fp):+.2f} % of notional per fill ({len(fp)} read; negative = paid)"
                 if fp else f"  💸 funding: no funding stamps on {who}")
        if hh:
            L[-1] += f" · avg hold {np.mean(hh):.1f} h (median {np.median(hh):.1f} h)"
    if any(wick):   # sensitivity (information only, never deciding): the replica TP-10 arm can stop on a 1m wick live polling rode through
        nw = sum(wick)
        rest = [x for x, w in zip(rows, wick) if not w]
        L.append(f"  ⚠ replica sensitivity: {nw} fill(s) booked TP 7 while the TP-10 replica STOPPED (Σ Δ {sum(x[1] for x, w in zip(rows, wick) if w):+.2f}) — "
                 f"the read without them: {gate(rest)[0]} ({len(rest)} fills; information only)")
    return L + [""]


if __name__ == "__main__":
    print("\n".join(report()))
