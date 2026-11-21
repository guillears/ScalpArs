#!/usr/bin/env python3
"""🎯🔒 Scout — HT_TP6 revert gate (DECISION_LOG 608, operator 2026-11-21 "build 1", operator-declared; ALERT-only, never changes config).

CHANGE     FRENZY_WILLY_HT take profit +5 → +6 % net (frenzy_willy_ht_tp_pct 6.0; stop −4 net, 24 h, flat target, no late TP unchanged).
           Evidence reports/HT_EXIT_RERUN_2026-11-21.md: +6 beats +5 only if P(reach +6 | reached +5) > ≈ 90 % (a hit gains ≈ +1 pt, a miss
           costs ≈ −9: a +5 winner ends at the −4 stop) — real kept fills 16 / 16, backtest 118 / 131 = 90 % (= breakeven).
QUESTION   on the trades where +5 would have paid, does waiting for +6 lose more than it gains?
COHORT     CLOSED FRENZY_WILLY_HT fills opened ≥ the DECISION_LOG 608 commit + 10 min (orders exports, dedup, CLOSED wins, operator-closed
           out) whose STORED target is the +6 one (frenzy_willy_tp_target_pct ≥ 6 — a fill that opened inside the deploy lag kept +5 and is out)
           and that REACHED +5 net before closing: the recorded peak_pnl ≥ +5 − FEE_GAP OR a FRENZY_TP booked ≥ that bar. peak_pnl = the
           realtime net P&L with a TAKER exit fee while the +5 target is a MAKER limit → at the +5 limit price the peak reads ≈ 4.97; the bar is
           lowered by (taker − maker fee) × 100 (trading_config.json fees; deep review: a +5 limit fill showed peak 4.974). peak_pnl is never
           NULL on new rows (default 0.0); UNSCORED only covers exports without the column. ⚠ On paper the booked % includes the funding
           accrued since DL 539 — the Δ carries it (≈ 0 over HT's short holds; stated, not removed). Δ = booked pnl_percentage − 5.00 (what the +5 resting maker TP would have booked).
           A fill with no peak_pnl stamp and no TP is UNSCORED (listed, out of the read).
🔒 GATE     frozen ONCE (reports/SCOUT_HT_TP6_608.json, gitignored) at the first 10 reached fills in open-time order:
           Σ Δ < 0 → REVERT (alert: frenzy_willy_ht_tp_pct back to 5.0 — operator decides) · else KEEP. In practice two misses fire it.
           Priced (binomial, 10 reached fills): P(REVERT) at a true hit rate 80 % ≈ 62 % · 85 % ≈ 46 % · 90 % ≈ 26 % · 95 % ≈ 9 % · 100 % 0 %.
           Freezing waits while the newest orders export is < the 10th fill + 24 h (an earlier-opened HT fill could still be open).
Usage: venv/bin/python -I scripts/scout_ht_tp6_608.py"""
import json
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import scout_gatekit as _GK          # noqa: E402
import scout_exh_wide_585 as _W      # noqa: E402  (shared orders loader)
import scout_dist_tp_shadow as _DT   # noqa: E402  (newest_export)

DL = 608
KEY = "HT_TP6"
SLEEVE = "FRENZY_WILLY_HT"
STATE = os.path.join(_R, "reports", "SCOUT_HT_TP6_608.json")
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "pnl_percentage", "close_reason", "peak_pnl", "frenzy_willy_tp_target_pct")
TP_OLD, TP_NEW, N_GATE, HOLD_H = 5.0, 6.0, 10, 24
TP_REASONS = ("FRENZY_TP",)


def _fee_gap():
    """(taker − maker) exit fee in % pts from trading_config.json (peak_pnl uses the taker fee, the TP limit the maker fee); 0.027 default."""
    try:
        with open(os.path.join(_R, "trading_config.json")) as fh:
            c = json.load(fh)
        g = (float(c.get("taker_fee", 0.00045)) - float(c.get("maker_fee", 0.00018))) * 100.0
        return g if 0.0 <= g < 0.2 else 0.027
    except Exception:
        return 0.027


FEE_GAP = _fee_gap()


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def classify(pnl_pct, close_reason, peak_pnl, tp_old=TP_OLD, fee_gap=None):
    """→ ('REACHED', Δ) when +tp_old net was reached (peak ≥ tp_old − fee_gap, or a TP booked at / above that bar), ('NOT_REACHED', 0.0), or
    ('UNSCORED', None). PURE."""
    p, pk = _f(pnl_pct), _f(peak_pnl)
    bar = tp_old - (FEE_GAP if fee_gap is None else fee_gap)
    if p is None:
        return "UNSCORED", None
    reason = str(close_reason or "").upper().split(" ")[0]
    # reached = the recorded net peak got to +tp_old, or the TP booked at / above it (an older era's lower TP never counts as reached)
    if (pk is not None and pk >= bar - 1e-9) or (reason in TP_REASONS and p >= bar - 1e-9):
        return "REACHED", p - tp_old
    if pk is None:
        return "UNSCORED", None
    return "NOT_REACHED", 0.0


def verdict(deltas):
    """first N_GATE reached Δ → (state, text). Σ Δ < 0 → REVERT · else KEEP · fewer → COLLECTING. PURE."""
    d = list(deltas)[:N_GATE]
    if len(d) < N_GATE:
        return "COLLECTING", f"{len(d)}/{N_GATE} fills reached +{TP_OLD:g} net · Σ Δ so far {sum(d):+.2f} pts"
    hits = sum(1 for x in d if x > 0)
    s = sum(d)
    return ("REVERT" if s < 0 else "KEEP"), (f"first {N_GATE} fills that reached +{TP_OLD:g}: {hits} went on to the +6 TP · Σ Δ {s:+.2f} pts "
                                             f"(< 0 → REVERT to +5)")


def report(frame=None, since_ms=None, state_path=STATE, now_ms=None, newest=None):
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🎯🔒 HT_TP6 revert gate (DECISION_LOG {DL}, operator-declared) — WILLY_HT TP +6 vs +5 on the fills that reached +5 net", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="HT_TP6")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    a = _W.load(sleeve=SLEEVE, cols=COLS) if frame is None else frame
    a = a[a.o >= pd.Timestamp(int(since_ms), unit="ms")] if len(a) else a
    n_old = 0
    if len(a) and "frenzy_willy_tp_target_pct" in a.columns:   # (deep review #5) the row's stored target must be the +6 one
        tgt = pd.to_numeric(a.frenzy_willy_tp_target_pct, errors="coerce")
        n_old = int((tgt < TP_NEW - 1e-9).sum())
        a = a[~(tgt < TP_NEW - 1e-9)]
    try:
        with open(state_path) as fh:
            st = json.load(fh)
    except FileNotFoundError:
        st = {}
    except (OSError, ValueError):
        st = None
    if st is not None and not isinstance(st, dict):
        st = None
    rec = st.get(KEY) if isinstance(st, dict) else None
    deltas, last_o, n_not, uns = [], None, 0, []
    for r in a.itertuples():
        k, d = classify(r.pnl_percentage, r.close_reason, getattr(r, "peak_pnl", None))
        tag = f"{r.pair} {r.o:%m-%d %H:%M} booked {(_f(r.pnl_percentage) or 0):+.2f} ({r.close_reason}) peak {getattr(r, 'peak_pnl', None)}"
        if k == "REACHED":
            deltas.append(d)
            if len(deltas) <= N_GATE:
                last_o = r.o
            L.append(f"  · {tag} → reached +5 · Δ {d:+.2f}")
        elif k == "UNSCORED":
            uns.append(tag)
        else:
            n_not += 1
    v, txt = verdict(deltas)
    L.append(f"  cohort: CLOSED HT fills opened ≥ {pd.Timestamp(int(since_ms), unit='ms'):%Y-%m-%d %H:%M} UTC · {len(a)} fills · "
             f"{len(deltas)} reached +5 · {n_not} never reached +5 (Δ 0, not counted) · {len(uns)} unscored"
             + (f" · {n_old} opened inside the deploy lag with the old +5 target (out)" if n_old else ""))
    for u in uns:
        L.append(f"  · UNSCORED (no peak stamp): {u}")
    newest = _DT.newest_export() if newest is None else newest
    hold = v != "COLLECTING" and (newest is None or last_o is None or newest < last_o + pd.Timedelta(hours=HOLD_H))
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
        L.append(f"Now: {v} — {txt}" + (" · ⏸ freezing deferred: newest export < the 10th fill + 24 h" if hold else ""))
    L.append("  Priced: P(REVERT) at a true P(+6 | +5) of 80 % ≈ 62 % · 85 % ≈ 46 % · 90 % ≈ 26 % · 95 % ≈ 9 % (break-even ≈ 90 %).")
    return L + [""]


if __name__ == "__main__":
    print("\n".join(report()))
