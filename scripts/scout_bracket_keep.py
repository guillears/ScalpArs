#!/usr/bin/env python3
"""🪜📈 Scout — KEEP THE PLANNED POSITION ON CAPPED COINS, revert gate (DECISION_LOG 535, operator "build it" 2026-11-12, paper; reads the
orders exports, no fetches, changes nothing).

RULE       services/trading_engine.bracket_keep_position: when the exchange brackets lower a position (same margin, lower leverage) the listed
           sleeves (bracket_keep_position_sleeves: FRENZY_LITE · FRENZY_LITE_RETRY · FRENZY_WILLY · FRENZY_WILLY_HT · FLATBASE_SHORT) keep the
           planned position at a lower leverage with more margin (≤ 1.6 × planned, ≤ free balance). Stamps bracket_keep_margin_from /
           bracket_keep_notional_from = what the plain resize gave.
COHORT     CLOSED full-size fills (*_PROBE out, operator closes out) with bracket_keep_notional_from stamped, opened ≥ the DL 535 commit + 10 min,
           dedup (opened_at[:19], pair, entry_strategy), newest export wins. Extra $ = booked % × (notional_value − bracket_keep_notional_from) / 100
           = what the kept size added over today's resize (the same trade, the same %). By design this is the size-weighted sign of the kept
           cohort's booked % — the extra size can only pay when those trades win (review 535: it can fire on a sleeve's bad run; that is intended).
🔒 GATE    (pre-committed 2026-11-12, never re-fit) on the first 15 kept fills (open-time order): Σ extra $ < 0 (the kept size lost money vs the plain
           resize — equivalently the cohort's size-weighted booked % < 0) → REVERT (bracket_keep_position_enabled false; operator applies) · else KEEP.
           Frozen once decided (reports/SCOUT_BRACKET_KEEP.json, gitignored).
BLIND SPOT the free-balance cost (a later trade opened smaller / not at all because a kept fill held more margin) is not in the booked rows —
           watch BOOK_CHANGED / CELL_MULT_CAPPED in the logs; the B25 study saw 1 squeeze in 111 fills.
Usage: venv/bin/python -I scripts/scout_bracket_keep.py [export.csv …]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402

DL = 535
N_GATE = 15
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason", "cell_multiplier_source",
        "leverage", "investment", "notional_value", "bracket_keep_margin_from", "bracket_keep_notional_from")
STATE = os.path.join(os.path.dirname(_D), "reports", "SCOUT_BRACKET_KEEP.json")


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
        return pd.DataFrame(columns=list(COLS) + ["o", "pct", "extra"])
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
    c["n_from"] = pd.to_numeric(c.bracket_keep_notional_from, errors="coerce")
    c["n_now"] = pd.to_numeric(c.notional_value, errors="coerce")
    c = c[c.pct.notna() & c.n_from.notna() & c.n_now.notna()]
    c["extra"] = c.pct * (c.n_now - c.n_from) / 100.0
    return c.sort_values(["o", "pair"]).reset_index(drop=True)


def gate(extras):
    """first N_GATE extra $ (open-time order) → (verdict, detail). Pure."""
    x = np.asarray(extras, float)
    if len(x) < N_GATE:
        return "OBSERVING", f"{len(x)} of {N_GATE} kept fills · Σ extra ${x.sum():+,.0f}" if len(x) else f"0 of {N_GATE} kept fills"
    s = float(x[:N_GATE].sum())
    return ("REVERT" if s < 0 else "KEEP"), f"first {N_GATE} kept fills: Σ extra ${s:+,.0f} ({'< 0' if s < 0 else '≥ 0'})"


def _state(path):
    try:
        st = json.load(open(path)) if (path and os.path.exists(path)) else {}
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def run(now_ms=None, paths=None, floor_ms=None, state_path=STATE):
    L = ["## 🪜📈 KEEP THE PLANNED POSITION ON CAPPED COINS — revert gate (DECISION_LOG 535, paper)", ""]
    floor = floor_ms if floor_ms is not None else _GK.dl_deploy_ms(DL, label="BRACKET_KEEP")
    if floor is None:
        return L + ["Not evaluable yet: the DECISION_LOG 535 commit is not in git history (counted from its commit + 10 min).", ""]
    c = load(paths)
    c = c[c.o >= pd.Timestamp(int(floor), unit="ms")]
    st = _state(state_path)
    if "verdict" in st:
        v, why = st["verdict"], st["detail"] + f" (frozen {st.get('at', '')})"
    else:
        v, why = gate(c.extra.tolist())
        if v != "OBSERVING" and state_path:
            try:
                json.dump(dict(verdict=v, detail=why, at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")), open(state_path, "w"), indent=1)
            except Exception:
                L.append("⚠ gate state not saved this run.")
    L.append(f"Kept fills since {pd.Timestamp(int(floor), unit='ms'):%Y-%m-%d %H:%M} UTC: {len(c)} · **{v}** — {why}.")
    if len(c):
        L += ["", "| Opened (UTC) | Pair | Sleeve | position: resize → kept | margin: resize → kept | lev | booked % | extra $ |", "|---|---|---|---|---|---|---|---|"]
        for r in c.itertuples():
            L.append(f"| {r.o:%m-%d %H:%M} | {r.pair} | {r.entry_strategy} | ${r.n_from:,.0f} → ${r.n_now:,.0f} | "
                     f"${float(r.bracket_keep_margin_from):,.0f} → ${float(r.investment):,.0f} | {float(r.leverage):g}× | {r.pct:+.2f} | {r.extra:+,.0f} |")
        by = c.groupby("entry_strategy").agg(n=("extra", "size"), extra=("extra", "sum"), avg=("pct", "mean"))
        L.append("")
        L.append("By sleeve: " + " · ".join(f"{s} {int(r.n)} fills, avg {r.avg:+.2f} %, extra ${r.extra:+,.0f}" for s, r in by.iterrows()))
    if v == "REVERT":
        L += ["", "⛔ GATE FIRED → REVERT: set bracket_keep_position_enabled false (dashboard 🪜📈 box) and log it in DECISION_LOG — operator applies."]
    return L + ["", "Gate (frozen 2026-11-12): first 15 kept fills, Σ extra $ (booked % × added position) < 0 → off. Blind spot: the free-balance "
                    "cost on LATER trades is not in these rows (logs: BOOK_CHANGED / CELL_MULT_CAPPED).", ""]


if __name__ == "__main__":
    print("\n".join(run(paths=sys.argv[1:] or None)))
