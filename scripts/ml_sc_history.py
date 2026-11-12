#!/usr/bin/env python3
"""🔂 MOMENTUM-LONG HISTORY RE-BASED ON TODAY'S EXIT (DECISION_LOG 534) — the shared helper the momentum-long watch lines use to read
"today's exit" = the runner trail / ladder / recovery hold + the SECOND CHANCE (DL 532) on fills booked BEFORE the second chance went live.

SOURCE   reports/ML_SC_HISTORY_DELTA_2026-11-12.csv — frozen from the DL 532 study replica (reports/ML_SC_REFINE_2026-11-11_variants.csv,
         BASE vs REF_d0.3_k1 = the shipped knobs: floor 0.30 · trail 1×entry ATR · clock 180 min · zone entry_adx_delta ≤ 0), one row per master
         momentum long (123): delta = second chance − today's exit at the same fill (0 out of zone / not a runner or ladder exit).
RULE     a fill opened BEFORE the DL 532 commit + 10 min → its booked % (master: stack_pct) + delta ONLY when the replica reproduced the
         booked exit (|booked − base_pct| ≤ AGREE_TOL; review 534: on 12 of the 38 changed fills the replica disagrees with the booked exit — e.g. a
         fill booked at its stop the replica ran to a runner — and adding the delta there credits a second chance that could never have run) →
         else the booked % unchanged (no credit), counted as "replica disagrees"; no delta row → unchanged when the second chance could not touch it
         (ADX rising at entry / no ADX stamp / not a runner trail or TP-ladder exit), else NaN = UNPRICED (shown, never silently booked). A fill
         opened after → its booked % (the engine ran the second chance itself). Ship time unknown (DL 532 not in git) → NO re-base (the caller
         shows it as not evaluable) — never "every fill is pre-ship".
The knobs are pinned: a change to any ml_second_chance_* key makes the lines that use this helper ⚠ STALE (observe_registry).
"""
import os
import sys

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(_D)
for _p in (_D, _R):
    _p in sys.path or sys.path.insert(0, _p)

DELTA = os.path.join(_R, "reports", "ML_SC_HISTORY_DELTA_2026-11-12.csv")
SC_DL = 532
AGREE_TOL = 0.10      # % — the replica's BASE exit must match the booked exit to carry its second-chance delta
KNOBS = {"ml_second_chance_enabled": True, "ml_second_chance_adx_delta_max": 0.0, "ml_second_chance_floor_pct": 0.3,
         "ml_second_chance_trail_atr_mult": 1.0, "ml_second_chance_clock_min": 180.0}


def key(opened_at, pair):
    return pd.Series(opened_at).astype(str).str.replace(" ", "T").str[:19].to_numpy() + "|" + pd.Series(pair).astype(str).to_numpy()


def load_delta(path=DELTA):
    """→ {key: (delta, base_pct)} (empty on a missing file — every pre-ship in-zone runner fill then reads UNPRICED, never silently booked)."""
    try:
        d = pd.read_csv(path)
        return dict(zip(d.key.astype(str), zip(pd.to_numeric(d.delta, errors="coerce"), pd.to_numeric(d.base_pct, errors="coerce"))))
    except Exception:
        return {}


def can_change(adx_delta, close_reason, adx_max=0.0):
    """True where the second chance could have changed the exit (zone + a runner / ladder exit). PURE."""
    from services.ml_second_chance import sc_base_reason
    a = pd.to_numeric(pd.Series(adx_delta), errors="coerce").to_numpy(float)
    r = np.array([sc_base_reason(x if isinstance(x, str) else "") is not None for x in pd.Series(close_reason)], dtype=bool)
    return (a <= adx_max) & r      # NaN ADX → False (fail closed, as the engine)


def rebase(pct, opened_at, pair, adx_delta, close_reason, ship_ms, delta=None):
    """→ (re-based %, n changed, n unpriced, n replica-disagrees). pct / opened_at / … aligned 1-D; delta = {key: (delta, base_pct)}.
    ship_ms None → returned unchanged (not evaluable). PURE given delta."""
    delta = load_delta() if delta is None else delta
    p = pd.to_numeric(pd.Series(pct), errors="coerce").to_numpy(float).copy()
    adx_delta = [np.nan] * len(p) if adx_delta is None else adx_delta
    close_reason = [""] * len(p) if close_reason is None else close_reason
    o = pd.to_datetime(pd.Series(opened_at).astype(str).str.replace(" ", "T").str[:19], format="ISO8601", errors="coerce")
    if ship_ms is None:
        return p, 0, 0, 0
    pre = (o < pd.Timestamp(int(ship_ms), unit="ms")).to_numpy()
    k = key(opened_at, pair)
    cc = can_change(adx_delta, close_reason)
    n_ch = n_un = n_dis = 0
    for i in range(len(p)):
        if not pre[i]:
            continue
        row = delta.get(k[i])
        dv, base = row if row is not None else (None, None)
        if dv is not None and dv == dv:
            if abs(dv) <= 1e-12:
                continue
            if base is not None and base == base and p[i] == p[i] and abs(p[i] - base) <= AGREE_TOL:
                p[i] += dv
                n_ch += 1
            else:
                n_dis += 1                                   # replica ≠ booked exit → no second-chance credit (review 534)
        elif cc[i]:
            p[i], n_un = np.nan, n_un + 1
    return p, n_ch, n_un, n_dis


def ship_ms():
    import scout_gatekit as _GK
    return _GK.dl_deploy_ms(SC_DL, lag_min=10, label="ML_SECOND_CHANCE history re-base")


def build(variants=os.path.join(_R, "reports", "ML_SC_REFINE_2026-11-11_variants.csv"), out=DELTA):
    """one-off: freeze the per-fill delta from the DL 532 study replica (BASE vs REF_d0.3_k1)."""
    v = pd.read_csv(variants)
    b = v[v.variant == "BASE"].set_index("key")
    s = v[v.variant == "REF_d0.3_k1"].set_index("key")
    d = pd.DataFrame({"base_pct": b.pct, "base_reason": b.reason, "sc_pct": s.pct, "sc_reason": s.reason}).dropna(subset=["base_pct", "sc_pct"])
    d["delta"] = d.sc_pct - d.base_pct
    d.index.name = "key"
    d.reset_index().to_csv(out, index=False, float_format="%.6f")
    return d


if __name__ == "__main__":
    d = build()
    print(f"{len(d)} fills · {int((d.delta.abs() > 1e-12).sum())} changed · Σ delta {d.delta.sum():+.3f}")
