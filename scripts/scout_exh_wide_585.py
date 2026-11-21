#!/usr/bin/env python3
"""🌐 Scout — EXHAUSTION WIDE universe revert gate (DECISION_LOG 585, 2026-11-18, operator-approved PAPER run, DECLARED OVERRIDE — ALERT-only,
never changes config).

CHANGE      exhaustion_wide_enabled: EXHAUSTION's own detector also tracks spikes by the FRENZY bar rule at ≥ 15× the normal hour (FRENZY: 20×)
            on coins with 24 h quote volume ≥ $5M (FRENZY: $20M); fills stamped entry_exh_universe = WIDE (a FRENZY-flagged episode → FRENZY).
COHORT      CLOSED EXHAUSTION_SHORT fills with entry_exh_universe == WIDE opened ≥ the DL 585 commit + 10 min, from the orders exports (paper +
            live via export_glob; dedup (opened_at, pair, direction), CLOSED beats OPEN); operator-closed (MANUAL / MANUAL_* / EXTERNAL_CLOSE) out.
NET         each fill re-priced with the study's modelled slippage (services.exhaustion.gate_net_pct — entry leg via entry_leg_slip, taker exit
            leg unless the maker TP booked it), the same rule as the EXHAUSTION revert gate.
GATE        decided ONCE (services.distribution.upper_bound_gate): the first 30 WIDE fills by OPEN time, extended until they span ≥ 10 days →
            the one-sided 80 % UPPER bound of their mean net (day-block bootstrap, seeded) < 0 → flag "switch exhaustion_wide_enabled off" ·
            else KEEP. Before that: COLLECTING. Also printed: WIDE vs FRENZY N · WR · avg (running, information only).
STATE       reports/SCOUT_EXH_WIDE_585_609.json (gitignored) — the verdict frozen once.
🔁 RE-BASED 2026-11-21 (DECISION_LOG 609, EXH TP cap 10 → 7): the count starts at the DL 609 commit + 10 min (fresh state file; the TP-10
            era had not frozen).
NETWORK     none. Usage: venv/bin/python -I scripts/scout_exh_wide_585.py"""
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
from services.distribution import upper_bound_gate          # noqa: E402
from services.exhaustion import gate_net_pct, entry_leg_slip, model_slip_pct, exit_era_609   # noqa: E402

DL = 585
PATTERNS = (os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv"), os.path.expanduser("~/Downloads/scalpars_orders_live_*.csv"))
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "notional_value",
        "entry_exh_universe", "entry_exh_model_slip_pct", "entry_exh_q5_usd", "entry_depth_mode", "entry_depth_walk_pct",
        "entry_exh_tp_pct", "entry_exh_atr_pct")   # 🔁 (609) the era rule reads the stamps
STATE = os.path.join(_R, "reports", "SCOUT_EXH_WIDE_585_609.json")
DL_EXIT = 609   # 🔁 the exit era the gate counts
SLEEVE = "EXHAUSTION_SHORT"


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load(paths=None, sleeve=SLEEVE, cols=COLS):
    """closed bot fills of `sleeve` from the orders exports (dedup, CLOSED wins, operator-closed out), with `o` = open time. Pure over files."""
    paths = paths if paths is not None else sorted({p for pat in PATTERNS for p in _EG.glob_exports(pat)}, key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(cols) + ["o"])
    a = pd.concat(fr, ignore_index=True).reindex(columns=list(cols))
    a = a.assign(_c=(a.status.astype(str) == "CLOSED").astype(int), _i=range(len(a))).sort_values(["_c", "_i"])
    a = a.drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a = a[(a.status.astype(str) == "CLOSED") & (a.entry_strategy.astype(str) == sleeve)]
    a = a[~a.apply(lambda r: _GK.is_operator_closed(r.entry_strategy, r.close_reason), axis=1).astype(bool)] if len(a) else a
    a["o"] = pd.to_datetime(a.opened_at.astype(str).str[:19], errors="coerce")
    return a[a.o.notna()].sort_values("o").reset_index(drop=True)


def net_of(r, slip_col="entry_exh_model_slip_pct", q5_col="entry_exh_q5_usd"):
    """a fill's net % re-priced with the modelled slippage (a missing stamp → re-priced from its notional and Q5; none → raw). Pure."""
    s = _f(getattr(r, slip_col, None))
    if s is None:
        s = model_slip_pct(_f(getattr(r, "notional_value", None)), _f(getattr(r, q5_col, None)))
    return gate_net_pct(_f(r.pnl_percentage), r.close_reason, s, entry_leg_slip(s, getattr(r, "entry_depth_mode", None), _f(getattr(r, "entry_depth_walk_pct", None))))


def frozen(rows, state_path, gate_fn, now_ms=None):
    """→ (state dict or None, live verdict, live text): the verdict is written ONCE, the first time the gate decides (not COLLECTING)."""
    v, txt = gate_fn(rows)
    st = {}
    try:
        with open(state_path) as fh:
            st = json.load(fh)
    except FileNotFoundError:
        st = {}
    except Exception:
        st = None
    if st is not None and "verdict" not in st and v != "COLLECTING":
        st = dict(verdict=v, detail=txt, run_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime((now_ms or time.time() * 1000) / 1000)))
        try:
            with open(state_path, "w") as fh:
                json.dump(st, fh)
        except Exception:
            pass
    return st, v, txt


def _row(lbl, nets):
    n = len(nets)
    if not n:
        return f"  {lbl}: N 0"
    return f"  {lbl}: N {n} · WR {sum(1 for x in nets if x > 0) / n * 100:.1f} % · avg {sum(nets) / n:+.3f} % (re-priced)"


def report(frame=None, since_ms=None, state_path=STATE, now_ms=None):
    out = [f"## 🌐 DL {DL} EXHAUSTION WIDE universe (15× / $5M) — revert gate (alert only)"]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL_EXIT, lag_min=10, label="DL 585 EXHAUSTION WIDE (re-based on the DL 609 exit)")
    if since_ms is None:
        return "\n".join(out + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL_EXIT} commit is not in git yet; nothing is counted."])
    a = load() if frame is None else frame
    a = a[a.o >= pd.Timestamp(int(since_ms), unit="ms")] if len(a) else a
    uni = a.entry_exh_universe.fillna("FRENZY").astype(str) if len(a) else pd.Series(dtype=str)
    if len(a) and "entry_exh_tp_pct" in a.columns:   # 🔁 (609) the dashboard's rule too: TP-7 stamps only (a row OPEN at the deploy kept its TP-10 stamp)
        a = a[a.apply(lambda r: exit_era_609(r.entry_exh_tp_pct, r.entry_exh_atr_pct, r.o), axis=1).astype(bool)]
        uni = a.entry_exh_universe.fillna("FRENZY").astype(str) if len(a) else pd.Series(dtype=str)
    wide = a[uni == "WIDE"] if len(a) else a
    fz = a[uni != "WIDE"] if len(a) else a
    rows = [(r.o.to_pydatetime(), net_of(r)) for r in wide.itertuples() if net_of(r) is not None]
    st, v, txt = frozen(rows, state_path, lambda rr: upper_bound_gate(rr, "WIDE universe", "exhaustion_wide_enabled", DL), now_ms)
    if st and st.get("verdict"):
        out.append(f"FROZEN ({st.get('run_at')}): {st['verdict']} — {st.get('detail')}")
    out.append(f"now: {v} — {txt}" + (" (information; the frozen read decides)" if st and st.get("verdict") else ""))
    out.append(_row("WIDE  ", [x for _, x in rows]))
    out.append(_row("FRENZY", [x for x in (net_of(r) for r in fz.itertuples()) if x is not None]))
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
