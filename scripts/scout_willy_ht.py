#!/usr/bin/env python3
"""🎲¼ Scout — FRENZY_WILLY_HT gate tracker (DECISION_LOG 341, operator option B 2026-10-21, ALERT-only — never changes config).

RULE (engine)   a WILLY setup refused SOLELY by the turnover filter (measured 24 h volume / mcap ≥ frenzy_willy_max_vol_mcap_ratio) opens
                FRENZY_WILLY_HT at frenzy_willy_ht_size_frac (¼) of WILLY's size, WILLY's exit, own slot. The turnover shadow keeps running.
COHORT          CLOSED FRENZY_WILLY_HT fills from the orders exports (paper + live via export_glob; dedup (opened_at, pair, direction), CLOSED
                wins), opened ≥ the DL 341 deploy (git commit + 10 min; not in git → NOT EVALUABLE, nothing counted); operator-closed
                (MANUAL / MANUAL_* / EXTERNAL_CLOSE) excluded — scout_gatekit. Comparator: full-size FRENZY_WILLY fills, same filters.
GATES           services.frenzy.frenzy_willy_ht_gate (the SAME function the dashboard row uses): Σ ≤ −8 pts → ALERT review · first 10 avg < 0
                → ALERT review · MAIN READ at the frozen first 30 fills on ≥ 10 UTC days vs WILLY fills opened in the same period: HT avg ≥ WILLY
                avg → recommend turnover ratio 0 + HT off · HT ≤ WILLY − 0.20 → filter confirmed, HT off · else re-read at 60.
Usage           venv/bin/python scripts/scout_willy_ht.py           (batch review; no network, no bot API)
"""
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import scout_gatekit as _GK                                  # noqa: E402
import scout_willy_hold_off as _HO                           # noqa: E402  (the shared export loader: dedup, CLOSED wins, paper + live)
from services.frenzy import frenzy_willy_ht_gate             # noqa: E402

DL = 341


def cohorts(a, since):
    """→ (ht kept, willy kept, ht operator-closed) — CLOSED, opened ≥ since, operator-closed out."""
    c = a[(a.status.astype(str) == "CLOSED") & (a.o >= since)]
    ht, ox = _GK.split_operator_closed(c[c.es == "FRENZY_WILLY_HT"]) if len(c) else (c, c)
    w, _ = _GK.split_operator_closed(c[c.es == "FRENZY_WILLY"]) if len(c) else (c, c)
    return ht.sort_values("o"), w.sort_values("o"), ox


def run(now_ms=None, paths=None, since_ms=None):
    L = [f"## 🎲¼ WILLY_HT — quarter-size WILLY on turnover-only refusals (DECISION_LOG {DL}, ALERT-only gates)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="WILLY_HT")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} deploy commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    a = _HO.load(paths)
    ht, w, ox = cohorts(a, since)
    L.append(f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC (deploy + 10 min) · {_GK.operator_closed_line(ox)}")
    for name, g in (("FRENZY_WILLY_HT (¼)", ht), ("FRENZY_WILLY (full)", w)):
        L.append(f"- {name}: " + (f"N {len(g)} · {g.o.dt.date.nunique()} days · WR {(g.pct > 0).mean():.0%} · avg {g.pct.mean():+.3f} % · "
                                  f"Σ {g.pct.sum():+.2f} pts · ${g.usd.sum():+.2f}" if len(g) else "N 0"))
    v, txt = frenzy_willy_ht_gate([(r.o.to_pydatetime(), r.pct) for r in ht.itertuples()], [(r.o.to_pydatetime(), r.pct) for r in w.itertuples()])
    L += [f"Gate verdict: {v} — {txt}", ""]
    return L


def main():
    print("\n".join(run(int(time.time() * 1000))))


if __name__ == "__main__":
    main()
