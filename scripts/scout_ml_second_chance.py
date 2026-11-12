#!/usr/bin/env python3
"""🔂 Scout — MOMENTUM LONG SECOND CHANCE revert gate (ARMED in paper 2026-11-11, DECISION_LOG 532; reads stamped columns only — no fetches, changes nothing).

RULE       services/ml_second_chance.py: a MOMENTUM LONG with ADX falling at entry (entry_adx_delta ≤ 0) whose RUNNER_TRAIL / HARD_TP_LADDER would close
           at L is held — SC_FLOOR at L − 0.30 · SC_TRAIL 1×entry ATR below the post-trigger high · SC_CLOCK at 180 min from entry.
SHADOW     every held trade stamps what today's exit would have booked (sc_base_pnl = L, sc_base_reason, sc_armed_at); a changed exit's Δ = the booked
           pnl_percentage − sc_base_pnl (one size: both at the same position, %).
COHORT     CLOSED MOMENTUM LONG fills, full size (*_PROBE out — probes run the same exit, paper = live), operator-closed held trades listed apart (scout_gatekit), from the orders exports (export_glob), dedup
           (opened_at[:19], pair) — CLOSED beats OPEN, newest export wins. CHANGED = sc_armed_at stamped. Counted from the DL 532 commit + 10 min.
EVIDENCE   (in-sample, NEVER counted) master 123 kept ML fills +33.1 → +44.9 pts (one-size DCR 0.57 → 0.77 %/day, +11.0 from winners, 25 better / 13
           worse); B25 +10.7 → +17.8; 2-year backtest +12.1 pts, +0.022/fill CI [−0.003, +0.049] (mostly 26-H2).
🔒 GATE    (pre-committed 2026-11-11, operator-directed armed ship, DECISION_LOG 532 — never re-fit):
           ① EARLY STOP: the first 8 changed exits (by open time) Σ Δ < −1.5 pts → REVERT (ml_second_chance_enabled → false; operator applies).
           ② at 15 changed exits: Σ Δ ≤ 0 → REVERT · Σ Δ > 0 → PASS (stays on; a go-live decision is separate).
           Decided once at the first prefix that reaches the count; the read shows it every run.
Usage: venv/bin/python -I scripts/scout_ml_second_chance.py [export.csv …]
"""
import os
import sys

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK        # noqa: E402  (operator-closed fills out · the DL commit time)

DL = 532
EARLY_N, EARLY_SUM = 8, -1.5
GATE_N = 15
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "close_reason", "cell_multiplier_source",
        "entry_adx_delta", "peak_pnl", "sc_armed_at", "sc_base_pnl", "sc_base_reason", "sc_high_pct")


def load(paths=None):
    """→ closed full-size MOMENTUM LONG fills (dedup, CLOSED beats OPEN, newest export wins), operator closes out."""
    fr = []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "pct", "base", "changed", "delta"])
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[(d.entry_strategy.astype(str) == "MOMENTUM") & (d.direction.astype(str) == "LONG")
          & ~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["op_closed"] = np.array([_GK.is_operator_closed(s, r) for s, r in zip(c.entry_strategy, c.close_reason)], dtype=bool) if len(c) else False
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c["base"] = pd.to_numeric(c.sc_base_pnl, errors="coerce")
    c["changed"] = c.sc_armed_at.notna() & c.base.notna()
    c["delta"] = np.where(c.changed, c.pct - c.base, 0.0)
    c = c[c.pct.notna()]
    return c.sort_values(["o", "pair"]).reset_index(drop=True)


def gate(deltas):
    """deltas = Δ of the changed exits in open-time order → (verdict, detail). Pure. Verdicts: REVERT (early) · REVERT · PASS · OBSERVING."""
    x = [float(v) for v in deltas]
    if len(x) >= EARLY_N and sum(x[:EARLY_N]) < EARLY_SUM:
        return "REVERT", f"early stop: the first {EARLY_N} changed exits Σ {sum(x[:EARLY_N]):+.2f} < {EARLY_SUM:+.1f} pts"
    if len(x) >= GATE_N:
        s = sum(x[:GATE_N])
        return ("PASS" if s > 0 else "REVERT"), f"the first {GATE_N} changed exits Σ {s:+.2f} pts ({'> 0' if s > 0 else '≤ 0'})"
    return "OBSERVING", f"{len(x)} of {GATE_N} changed exits · Σ {sum(x):+.2f} pts" + (f" (early-stop check at {EARLY_N})" if len(x) < EARLY_N else "")


def _zone_max():
    """the live zone bound (ml_second_chance_adx_delta_max) — review: never a hard-coded 0."""
    try:
        import json
        with open(os.path.join(os.path.dirname(_D), "trading_config.json")) as fh:
            return float(json.load(fh)["thresholds"].get("ml_second_chance_adx_delta_max", 0.0))
    except Exception:
        return 0.0


def run(now_ms=None, paths=None, floor_ms=None):
    L = ["## 🔂 ML SECOND CHANCE — revert gate (DECISION_LOG 532, ARMED in paper)", ""]
    floor = floor_ms if floor_ms is not None else _GK.dl_deploy_ms(DL, label="ML_SECOND_CHANCE")
    if floor is None:
        return L + ["Not evaluable yet: the DECISION_LOG 532 commit is not in git history (counted from its commit + 10 min).", ""]
    c = load(paths)
    c = c[c.o >= pd.Timestamp(int(floor), unit="ms")]
    opc = c[c.op_closed & c.changed]       # operator closes of held trades: listed, not counted (the gate reads bot exits only)
    c = c[~c.op_closed]
    zone = pd.to_numeric(c.entry_adx_delta, errors="coerce") <= _zone_max()
    ch = c[c.changed]
    v, why = gate(ch.delta.tolist())
    L.append(f"Counted since {pd.Timestamp(int(floor), unit='ms'):%Y-%m-%d %H:%M} UTC: {len(c)} momentum longs · {int(zone.sum())} with ADX falling at entry · "
             f"{len(ch)} exits changed · **{v}** — {why}.")
    if len(ch):
        b, a = ch.base.sum(), ch.pct.sum()
        L.append(f"Changed exits: today's exit {b:+.2f} → second chance {a:+.2f} pts (Δ {a - b:+.2f}; {int((ch.delta > 1e-9).sum())} better / "
                 f"{int((ch.delta < -1e-9).sum())} worse; avg Δ {ch.delta.mean():+.3f}/fill) · win rate {(ch.base > 0).mean():.0%} → {(ch.pct > 0).mean():.0%}.")
        L += ["", "| Opened (UTC) | Pair | ADX Δ | Today's exit | Second chance | Δ | Reason | High after trigger |", "|---|---|---|---|---|---|---|---|"]
        for r in ch.itertuples():
            L.append(f"| {r.o:%m-%d %H:%M} | {r.pair} | {float(r.entry_adx_delta):+.2f} | {r.base:+.2f} ({r.sc_base_reason}) | {r.pct:+.2f} | {r.delta:+.2f} | "
                     f"{r.close_reason} | {float(r.sc_high_pct):+.2f} |" if pd.notna(r.sc_high_pct) else
                     f"| {r.o:%m-%d %H:%M} | {r.pair} | {float(r.entry_adx_delta):+.2f} | {r.base:+.2f} ({r.sc_base_reason}) | {r.pct:+.2f} | {r.delta:+.2f} | {r.close_reason} | – |")
    if len(opc):
        L.append("")
        L.append(f"⚠ {len(opc)} held trade(s) closed by the operator — NOT counted (Δ " + ", ".join(f"{r.pair} {r.pct - r.base:+.2f}" for r in opc.itertuples())
                 + "); if this grows, read the gate with them too.")
    if v == "REVERT":
        L.append("")
        L.append("⛔ GATE FIRED → REVERT: set ml_second_chance_enabled false (dashboard 🔂 box) and log the verdict in DECISION_LOG — operator applies.")
    L.append("")
    L.append("Rule: early stop if the first 8 changed exits Σ < −1.5 pts; at 15 changed exits revert if Σ ≤ 0. Δ = booked % − today's-exit level at the "
             "trigger (sc_base_pnl). In-sample at ship (never counted): master +11.8 pts over 38 changed exits, B25 +7.0 over 13.")
    return L + [""]


if __name__ == "__main__":
    print("\n".join(run(paths=sys.argv[1:] or None)))
