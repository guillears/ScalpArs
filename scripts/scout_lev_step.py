#!/usr/bin/env python3
"""📈 Scout — B26 LEVERAGE STEP revert gates (DECISION_LOG 533, operator-directed declared override in PAPER; reads the orders exports, no fetches).

STEP       Momentum Long 10× → 15× · LITE RETRY 7× → 10× · WILLY 5× → 8× · WILLY HT 5× → 8× (follows WILLY) · Flatbase 5× → 8× (money in per trade
           unchanged) + the ML risk cap (ml_risk_cap_pct 10: Σ open momentum-long risk + the new trade ≤ 10 % of the account → shrink / skip).
COHORT     per sleeve, CLOSED full-size fills (*_PROBE out, operator closes out) opened ≥ the DL 533 commit + 10 min AT the new leverage (a fill the
           exchange brackets lowered is not counted — DL 535 kept positions too: the 11-11 cohort is pre-committed, they are read by
           scripts/scout_bracket_keep.py), dedup (opened_at[:19], pair, entry_strategy), newest export wins. P&L = booked pnl_percentage.
🔒 GATE    (pre-committed 2026-11-11, never re-fit) per sleeve, decided once on its first 15 counted fills: WR < its breakeven (frozen: Momentum
           Long 61.1 % · LITE RETRY 50.0 % · WILLY 69.0 % · WILLY HT 69.0 % · Flatbase 66.7 %) OR avg % < 0 → REVERT to the old leverage (operator
           applies: ML 0.5 · RETRY 0.35 · WILLY 0.25 (HT follows WILLY's leverage → back to 5× too) · HT alone: HT frac 0.625 · Flatbase 0.25) · else KEEP.
           Frozen once decided (reports/SCOUT_LEV_STEP.json, gitignored) — a reset / missing export never flips a verdict.
ALSO       the ML risk cap: momentum longs opened with the cap shrinking them (notional below the paired 15× size) and the open risk stamped at entry
           (entry_ml_risk_open_pct); its skips are priced by scout_blocked_filters (line ML_RISK_CAP).
Usage: venv/bin/python -I scripts/scout_lev_step.py [export.csv …]
"""
import os
import sys

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402

DL = 533
N_GATE = 15
# sleeve label: (entry_strategy, direction, new leverage, frozen breakeven WR, revert instruction)
SLEEVES = {"Momentum Long": ("MOMENTUM", "LONG", 15, 0.611, "momentum_long_lev_mult 0.75 → 0.5"),
           "LITE RETRY": ("FRENZY_LITE_RETRY", "LONG", 10, 0.500, "frenzy_lite_retry_lev_mult 0.5 → 0.35"),
           "WILLY": ("FRENZY_WILLY", "LONG", 8, 0.690, "frenzy_willy_lev_mult 0.4 → 0.25 — HT runs at WILLY's leverage, so HT goes back to 5× with it "
                     "(an HT that passed its own gate can only be kept bigger via frenzy_willy_ht_size_frac ≤ 1)"),
           "WILLY HT": ("FRENZY_WILLY_HT", "LONG", 8, 0.690, "frenzy_willy_ht_size_frac 1.0 → 0.625 (HT back to the 5× notional)"),
           "Flatbase": ("FLATBASE_SHORT", "SHORT", 8, 0.667, "flatbase_lev_mult 0.4 → 0.25")}
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "close_reason", "cell_multiplier_source", "leverage",
        "entry_ml_risk_open_pct", "entry_ml_risk_shrunk_from", "notional_value")
STATE = os.path.join(os.path.dirname(_D), "reports", "SCOUT_LEV_STEP.json")


def load(paths=None):
    fr = []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "pct"])
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "direction", "entry_strategy"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    if len(c):
        c = c[~np.array([_GK.is_operator_closed(s, r) for s, r in zip(c.entry_strategy, c.close_reason)], dtype=bool)]
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    return c[c.pct.notna()].sort_values(["o", "pair"]).reset_index(drop=True)


def gate(pcts, breakeven):
    """first N_GATE booked % (open-time order) → (verdict, detail). Pure."""
    x = np.asarray(pcts, float)
    if len(x) < N_GATE:
        return "OBSERVING", f"{len(x)} of {N_GATE}" + (f" · WR {np.mean(x > 0):.0%} · avg {x.mean():+.2f}" if len(x) else "")
    y = x[:N_GATE]
    wr, avg = float(np.mean(y > 0)), float(y.mean())
    bad = wr < breakeven or avg < 0
    return ("REVERT" if bad else "KEEP"), f"first {N_GATE}: WR {wr:.0%} (breakeven {breakeven:.1%}) · avg {avg:+.2f} %"


def _state(path):
    import json
    try:
        st = json.load(open(path)) if (path and os.path.exists(path)) else {}
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def run(now_ms=None, paths=None, floor_ms=None, state_path=STATE):
    L = ["## 📈 B26 LEVERAGE STEP — revert gates (DECISION_LOG 533, paper)", ""]
    floor = floor_ms if floor_ms is not None else _GK.dl_deploy_ms(DL, label="LEV_STEP")
    if floor is None:
        return L + ["Not evaluable yet: the DECISION_LOG 533 commit is not in git history (counted from its commit + 10 min).", ""]
    c = load(paths)
    c = c[c.o >= pd.Timestamp(int(floor), unit="ms")]
    L += ["| Sleeve | New lev | Counted | Verdict | Detail |", "|---|---|---|---|---|"]
    fired = []
    st, changed = _state(state_path), False
    for name, (es, di, lev, be, how) in SLEEVES.items():
        g = c[(c.entry_strategy.astype(str) == es) & (c.direction.astype(str) == di)]
        g = g[pd.to_numeric(g.leverage, errors="coerce") >= lev - 1e-9]
        if name in st:                                    # decided once — frozen
            v, why = st[name]["verdict"], st[name]["detail"] + f" (frozen {st[name]['at']})"
        else:
            v, why = gate(g.pct.tolist(), be)
            if v != "OBSERVING":
                st[name] = dict(verdict=v, detail=why, at=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M"))
                changed = True
        L.append(f"| {name} | {lev}× | {len(g)} | **{v}** | {why} |")
        if v == "REVERT":
            fired.append(f"⛔ {name} → REVERT: {how} (operator applies; log it in DECISION_LOG).")
    if changed and state_path:
        try:
            import json
            json.dump(st, open(state_path, "w"), indent=1)
        except Exception:
            L.append("⚠ gate state not saved this run.")
    ml = c[(c.entry_strategy.astype(str) == "MOMENTUM") & (c.direction.astype(str) == "LONG")]
    if len(ml):
        rk = pd.to_numeric(ml.entry_ml_risk_open_pct, errors="coerce")
        sh = pd.to_numeric(ml.entry_ml_risk_shrunk_from, errors="coerce")
        L.append("")
        L.append(f"ML risk cap: {int(rk.notna().sum())} momentum longs stamped · open risk at entry median {rk.median():.1f} % · max {rk.max():.1f} % · "
                 f"{int(sh.notna().sum())} shrunk by the cap (avg to {(pd.to_numeric(ml.notional_value, errors='coerce')[sh.notna()] / sh[sh.notna()]).mean():.0%} of size)"
                 if sh.notna().any() else
                 f"ML risk cap: {int(rk.notna().sum())} momentum longs stamped · open risk at entry median {rk.median():.1f} % · max {rk.max():.1f} % · none shrunk")
        L.append("Skips: blocked-trades scout line ML_RISK_CAP (a risk limiter — its price is what the cap cost).")
    if fired:
        L += [""] + fired
    return L + ["", "Gate per sleeve (frozen 2026-11-11): first 15 fills at the new leverage, WR < breakeven or avg < 0 → old leverage.", ""]


if __name__ == "__main__":
    print("\n".join(run(paths=sys.argv[1:] or None)))
