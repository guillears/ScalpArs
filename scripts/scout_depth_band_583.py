#!/usr/bin/env python3
"""📖 Scout — DL 583 depth-guard BAND revert gate (DECISION_LOG 583, 2026-11-18, operator-approved; ALERT-only, never changes config).

CHANGE      trading_config.json investment.entry_depth_guard_min_frac 0.5 → 0.25: a guarded entry whose thinner book side holds 25–50 % of
            the plan now trades SHRUNK to that side instead of being skipped (ZKPUSDT 07:30 / ZILUSDT 15:30 EXHAUSTION signals on 11-18).
COHORT      the BAND fills = CLOSED bot fills (any sleeve) with entry_depth_shrunk_from set and notional_value ÷ entry_depth_shrunk_from < 0.5
            (they would have been SKIPPED at 0.5 → their counterfactual is 0), opened ≥ the DL 583 commit + 10 min; dedup (opened_at, pair,
            direction), CLOSED beats OPEN; operator-closed (MANUAL / EXTERNAL_CLOSE) out.
GATE        decided ONCE at the first 10 band fills (by open time): Σ booked pnl $ < 0 → flag "revert min_frac to 0.5" · or mean |entry
            slippage| > 0.15 % · or mean |exit slippage| > 0.15 % (the thin side is the exit — a squeeze stop buys into it; paper books no
            slippage inside 0.15 %, so the slippage legs are LIVE-only evidence) → flag · else KEEP. Before 10: COLLECTING. The old DL 551 / 565
            guard gate has no pricer — it is a by-hand read at the batch review (stated in DL 583).
STATE       reports/SCOUT_DEPTH_BAND_583.json (gitignored) — the verdict frozen once.
Usage: venv/bin/python -I scripts/scout_depth_band_583.py"""
import json
import math
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402

DL = 583
GATE_N, SLIP_MAX, BAND = 10, 0.15, 0.5
PATTERNS = (os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv"), os.path.expanduser("~/Downloads/scalpars_orders_live_*.csv"))
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "notional_value",
        "entry_depth_shrunk_from", "entry_slippage_pct", "exit_slippage_pct", "is_paper")
STATE = os.path.join(_R, "reports", "SCOUT_DEPTH_BAND_583.json")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def in_band(notional, shrunk_from) -> bool:
    """True when the booked notional is < BAND × the planned (pre-guard) notional. Pure."""
    n, p = _f(notional), _f(shrunk_from)
    return n is not None and p is not None and p > 0 and n / p < BAND - 1e-9


def load(paths=None):
    paths = paths if paths is not None else sorted({p for pat in PATTERNS for p in _EG.glob_exports(pat)}, key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o"])
    a = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS))
    a = a.assign(_c=(a.status.astype(str) == "CLOSED").astype(int), _i=range(len(a))).sort_values(["_c", "_i"])
    a = a.drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a = a[(a.status.astype(str) == "CLOSED") & (a.entry_strategy.astype(str).str.upper() != "MANUAL")]   # bot fills only
    a = a[~a.apply(lambda r: _GK.is_operator_closed(r.entry_strategy, r.close_reason), axis=1).astype(bool)] if len(a) else a
    a["o"] = pd.to_datetime(a.opened_at.astype(str).str[:19], errors="coerce")
    return a[a.o.notna()].sort_values("o").reset_index(drop=True)


def gate(rows):
    """rows = [(pnl $, entry slip %, exit slip %)] in open order → (verdict, text). Pure."""
    if len(rows) < GATE_N:
        return "COLLECTING", f"⏳ {len(rows)}/{GATE_N} band fills"
    cut = rows[:GATE_N]
    s = sum(p for p, _, _ in cut if p is not None)
    es = [abs(x) for _, x, _ in cut if x is not None]
    xs = [abs(x) for _, _, x in cut if x is not None]
    me = sum(es) / len(es) if es else None
    mx = sum(xs) / len(xs) if xs else None
    info = (f"first {GATE_N} band fills: Σ ${s:+,.2f} · mean |entry slip| {me if me is not None else float('nan'):.3f} % · "
            f"mean |exit slip| {mx if mx is not None else float('nan'):.3f} %")
    if s < 0 or (me is not None and me > SLIP_MAX) or (mx is not None and mx > SLIP_MAX):
        return "REVERT", f"🔴 {info} → flag: revert entry_depth_guard_min_frac to 0.5 (operator decides)"
    return "KEEP", f"🟢 {info} → keep 0.25"


def report(frame=None, since_ms=None, state_path=STATE, now_ms=None):
    out = [f"## 📖 DL {DL} depth-guard band (min_frac 0.5 → 0.25) — revert gate (alert only)"]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="DL 583 depth band")
    if since_ms is None:
        return "\n".join(out + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted."])
    a = load() if frame is None else frame
    a = a[a.o >= pd.Timestamp(int(since_ms), unit="ms")] if len(a) else a
    band = a[[in_band(n, p) for n, p in zip(a.notional_value, a.entry_depth_shrunk_from)]] if len(a) else a
    rows = [(_f(r.pnl), _f(r.entry_slippage_pct), _f(r.exit_slippage_pct)) for r in band.itertuples()]
    st = {}
    try:
        with open(state_path) as f:
            st = json.load(f)
    except FileNotFoundError:
        st = {}
    except Exception:
        st = None
    if st is not None and "verdict" not in st and len(rows) >= GATE_N:
        v, txt = gate(rows)
        st = dict(verdict=v, detail=txt, run_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime((now_ms or time.time() * 1000) / 1000)))
        try:
            with open(state_path, "w") as f:
                json.dump(st, f)
        except Exception:
            pass
    if st and st.get("verdict"):
        out.append(f"FROZEN ({st.get('run_at')}): {st['verdict']} — {st.get('detail')}")
    v, txt = gate(rows)
    out.append(f"now: {v} — {txt}" + (" (information; the frozen read decides)" if st and st.get("verdict") else ""))
    for r in band.itertuples():
        out.append(f"  {str(r.o)[:16]} {r.pair} {r.entry_strategy} {r.direction}: ${_f(r.notional_value) or 0:,.0f} of ${_f(r.entry_depth_shrunk_from) or 0:,.0f} "
                   f"planned · pnl ${_f(r.pnl) or 0:+,.2f} · slip in {r.entry_slippage_pct} / out {r.exit_slippage_pct}")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
