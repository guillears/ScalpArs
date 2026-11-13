#!/usr/bin/env python3
"""🏃 Scout — WILLY RUNNER (SHIPPED in paper, DECISION_LOG 541 declared override) revert gate (observe only — never changes config, no bot API;
public 1m klines only, budgeted through the DL 537 scout's fetcher).

WHY        Operator 11-13: "it's a finding of a better exit". On FRENZY_WILLY fills whose coin's own 1h EMA20 is below its EMA200 (falling +
           mild coins) a short squeeze tends to keep going after the target (63 % follow-through vs 37 % on rising coins). Study
           reports/WILLY_RUNNER_FLOOR_1113_rows.csv: trail 2 × entry ATR % past the target with a +0.10 floor — backtest 115 fills Δ +88.7 pts
           vs today's exit (0 target-hit trades turned into losses; +134 without the floor but 16 did), master 7 fills Δ +33.2.
LIVE RULE  services/frenzy.py frenzy_willy_exit_for (runner branch): the row's frenzy_willy_runner_k set at the open → no resting TP; at the
           target T the trail arms: line = max(peak − k × ATR, +floor, −4) → WILLY_RUNNER_TRAIL; before T the −4 stop / late TP / 90-min cap.
PRICED     every runner-cohort fill on its OWN 1m path with the study's walker (scripts/study_willy_ht_exit_1112.leg — the same ruler):
             TODAY   the V2b exit at the row's stored target T (what today's exit would have booked: maker TP at T / −4 / late +0.25 / 90)
             REPLICA the runner rule at T, k, ATR and the floor (parity vs booked only)
           Δ = booked − TODAY, summed over RUNNER fills only (a fill that never touched T runs today's rule; its booked − walker gap is
           only costs — the walker's −4 stop has no slip — and must never move the gate).
COHORT     CLOSED full-size FRENZY_WILLY fills with frenzy_willy_runner_k set (*_PROBE out, operator closes out) opened ≥ the DL 541 commit
           + 10 min, from the orders exports (dedup newest wins). A "runner fill" = one whose path touched its target (TODAY booked a TP).
🔒 GATE    (pre-committed 2026-11-13, never re-fit; one size %, open-time order, decided ONCE, frozen in reports/SCOUT_WILLY_RUNNER.json)
           EARLY — Σ Δ over the runner fills so far < −5 pts at any read → REVERT (frenzy_willy_runner_enabled false, operator applies;
                   rows already open finish on the runner — the k is stored on the row)
           FINAL — at the 10th runner fill: Σ Δ over the first 10 runner fills < 0 → REVERT · else KEEP. Waits while a cohort fill opened
           before that 10th fill is still pending (a path still unpriced UNRES_DAYS after the open is dropped as "no path").
Usage: venv/bin/python -I scripts/scout_willy_runner_watch.py [--no-fetch]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(_D)
for _p in (_D, _R):
    _p in sys.path or sys.path.insert(0, _p)
import export_glob as _EG                  # noqa: E402
import scout_gatekit as _GK                # noqa: E402
import scout_willy_ht_exit_lines as _HT    # noqa: E402  (its 1m loader / budgeted fetch / atomic state — and the study import guard)

S = _HT.S                                  # the study's walker (study_willy_ht_exit_1112), import-time patch already undone by _HT
DL = 541
FINAL_N = 10
EARLY_SUM = -5.0
K_SHIPPED, FLOOR_SHIPPED = 2.0, 0.10       # the SHIPPED rule (registry-pinned) — the replica reads the row's own k, these are fallbacks
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
STATE = os.path.join(_R, "reports", "SCOUT_WILLY_RUNNER.json")
UNRES_DAYS = 3


def load(paths=None):
    fr = []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame()
    d = pd.concat(fr, ignore_index=True)
    d = d.reindex(columns=sorted(set(d.columns) | {"cell_multiplier_source", "close_reason", "frenzy_willy_runner_k"}))
    d = d[(d.entry_strategy.astype(str) == "FRENZY_WILLY")]
    d = d[~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")]
    if "is_probe" in d.columns:                                                                    # full-size fills only
        d = d[~d.is_probe.astype(str).str.lower().isin(("true", "1", "1.0"))]
    d = d.sort_values("_m").drop_duplicates(["opened_at", "pair", "entry_strategy"], keep="last")     # newest export wins (an open row closes later)
    d = d[pd.to_numeric(d.frenzy_willy_runner_k, errors="coerce") > 0]
    if len(d):
        d = d[~np.array([_GK.is_operator_closed(s, r) for s, r in zip(d.entry_strategy, d.close_reason)], dtype=bool)]
    return d.reset_index(drop=True)


def gate(deltas, runner_flags, held=False):
    """deltas = Δ per PRICED cohort fill in open-time order; runner_flags = touched its target (same order) — only those count; held = a
    cohort fill opened before the 10th runner fill still pending → (verdict, detail). Pure."""
    x = np.asarray(deltas, dtype=float)
    rf = np.asarray(runner_flags, dtype=bool)
    xr = x[rf]
    if len(xr) and xr.sum() < EARLY_SUM:
        return "REVERT", f"early stop: Σ Δ {xr.sum():+.2f} pts over {len(xr)} runner fills < {EARLY_SUM:+.0f} (today's exit would have made more)"
    if len(xr) >= FINAL_N and not held:
        s_ = float(xr[:FINAL_N].sum())
        return (("KEEP" if s_ >= 0 else "REVERT"), f"the first {FINAL_N} runner fills: Σ Δ {s_:+.2f} pts ({'≥ 0' if s_ >= 0 else '< 0'})")
    return "OBSERVING", (("a fill opened before the 10th runner fill is still pending · " if held and len(xr) >= FINAL_N else "")
                         + f"{len(xr)} of {FINAL_N} runner fills · {len(x)} cohort fills · Σ Δ (runner fills) {xr.sum():+.2f} pts")


def _ex_funding(r):
    """booked net % WITHOUT funding (DL 539 folds it into pnl; the walker has none — review 541 M2: compare like with like)."""
    p = S.CF._num(r.get("pnl_percentage"))
    if p is None:
        return None
    f, n = S.CF._num(r.get("funding_fee_usd")), S.CF._num(r.get("notional_value"))
    fin = str(r.get("funding_in_pnl")).strip().lower() in ("true", "1", "1.0")
    return p - f / n * 100.0 if (fin and f is not None and n) else p


def price(use, now_ms=None, floor_pct=FLOOR_SHIPPED):
    """→ DataFrame per fill: T, k, atr, TODAY / REPLICA (net %, NaN = unpriced) + why, booked, status. The study's walker."""
    rows, cache = [], {}
    for r in use.to_dict("records"):
        T, k, atr = S.CF._num(r.get("frenzy_willy_tp_target_pct")), S.CF._num(r.get("frenzy_willy_runner_k")), S.CF._num(r.get("entry_atr_pct"))
        base = dict(key=S._key(r.get("opened_at"), r.get("pair")), opened_at=str(r.get("opened_at"))[:19], pair=r.get("pair"),
                    booked=_ex_funding(r), status_row=str(r.get("status")).upper(), why_live=r.get("close_reason"),
                    T=T, k=k, atr=atr)
        te, E = S.CF._ms(r.get("opened_at")), S.CF._num(r.get("entry_price"))
        if base["status_row"] != "CLOSED":
            rows.append(dict(base, status="open")); continue
        if te is None or not E or not T or not k or not atr or atr <= 0:
            rows.append(dict(base, status="no stamps")); continue
        old = now_ms is not None and te + UNRES_DAYS * 86_400_000 < now_ms
        P = S.FX.load_path(r.get("pair"), te, E, S.CAP, cache)
        if not len(P["t"]):
            rows.append(dict(base, status=("no path" if old else "UNRES"))); continue
        tod = S.leg(P, T)
        rep = S.leg(P, T, run_k=k, atr=atr, be_floor=floor_pct)
        st = "UNRES" if (tod[1] == "UNRES" or rep[1] == "UNRES") else "priced"
        st = "no path" if (st == "UNRES" and old) else st
        rows.append(dict(base, status=st, TODAY=tod[0], TODAY_why=tod[1], REPLICA=rep[0], REPLICA_why=rep[1]))
    return pd.DataFrame(rows)


def run(now_ms=None, paths=None, floor_ms=None, fetch_ok=True, state_path=STATE):
    """→ report lines. The DL 537 scout's 1m loader is installed only for this call and the shared loader / its cache restored after."""
    k1_snap = dict(S.FX.WT._K1)
    S.FX.WT.load_1m = _HT._load_1m
    try:
        return _run(now_ms, paths, floor_ms, fetch_ok, state_path)
    finally:
        S.FX.WT.load_1m = _HT._TRUE_LOAD_1M
        S.FX.WT._K1.clear(); S.FX.WT._K1.update(k1_snap)


def _run(now_ms, paths, floor_ms, fetch_ok, state_path):
    now_ms = int(now_ms or time.time() * 1000)
    L = ["## 🏃 WILLY RUNNER — revert gate (shipped in paper, DECISION_LOG 541)", ""]
    try:
        S.TH = S.CF.load_th()
        floor = floor_ms if floor_ms is not None else _GK.dl_deploy_ms(DL, label="WILLY_RUNNER")
        if floor is None:
            return L + ["Not evaluable yet: the DECISION_LOG 541 commit is not in git history (counted from its commit + 10 min).", ""]
        d = load(paths)
        if len(d):
            d = d[[(t is not None and t >= floor) for t in (S.CF._ms(x) for x in d.opened_at)]]
        if not len(d):
            return L + ["No runner-cohort WILLY fill in the exports yet (FRENZY_WILLY on a coin with 1h EMA20 < EMA200, opened after the ship).", ""]
        note = _HT.fetch(d[d.status.astype(str).str.upper().eq("CLOSED")].to_dict("records"), now_ms) if fetch_ok else "no fetch"
        try:
            fl = float(getattr(S.TH, "frenzy_willy_runner_floor_pct", FLOOR_SHIPPED))
        except (TypeError, ValueError):
            fl = FLOOR_SHIPPED
        R = price(d, now_ms, fl).sort_values("opened_at").reset_index(drop=True)
        for c in ("TODAY", "REPLICA", "TODAY_why"):
            if c not in R:
                R[c] = np.nan
        ok = R.status.eq("priced").to_numpy()
        pend = R[R.status.isin(["open", "UNRES"])]
        F = R[ok]
        dl = (F.booked - F.TODAY).to_numpy()
        rf = F.TODAY_why.astype(str).eq("TP").to_numpy()
        tenth = F.opened_at.iloc[int(np.flatnonzero(rf)[FINAL_N - 1])] if rf.sum() >= FINAL_N else None
        held = bool(tenth is not None and len(pend) and pend.opened_at.min() < tenth)
        st = _HT._state(state_path)
        v = gate(dl, rf, held)
        if "RUNNER" in st:
            v = (st["RUNNER"]["verdict"], st["RUNNER"]["detail"] + f" (frozen {st['RUNNER']['at']})")
        elif v[0] != "OBSERVING":
            st["RUNNER"] = dict(verdict=v[0], detail=v[1], at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"))
            if state_path:
                try:
                    _HT._save_state(st, state_path)
                except Exception:
                    L.append("⚠ gate state not saved this run.")
        L.append(f"Forward since {pd.Timestamp(int(floor), unit='ms'):%Y-%m-%d %H:%M} UTC: {len(F)} priced cohort fills ({int(rf.sum())} reached "
                 f"their target) · pending {len(pend)} · no stamps {int(R.status.eq('no stamps').sum())} · no path {int(R.status.eq('no path').sum())} ({note}).")
        if len(F):
            Fr, dr = F[rf], dl[rf]
            L.append(f"Runner fills, live vs today's exit: Σ {Fr.booked.sum():+.2f} vs {Fr.TODAY.sum():+.2f} pts (Δ {dr.sum():+.2f}; "
                     f"{int((dr > 1e-9).sum())} better / {int((dr < -1e-9).sum())} worse) · trades that touched their target and ended a "
                     f"loss: {int(((F.TODAY_why.astype(str) == 'TP') & (F.booked < 0)).sum())}.")
        L.append(f"- **WILLY runner (shipped)**: **{v[0]}** — {v[1]}")
        if len(R):
            f = lambda val, w: "–" if val != val else f"{val:+.2f} ({w})"   # noqa: E731
            L += ["", "| Opened (UTC) | Pair | T | k × ATR | today's exit | replica | booked (reason) |", "|---|---|---|---|---|---|---|"]
            for r in R.itertuples():
                kk = f"{r.k:g} × {r.atr:.2f}" if (r.k == r.k and r.atr == r.atr and r.k and r.atr) else "–"
                L.append(f"| {str(r.opened_at)[5:16].replace('T', ' ')} | {r.pair} | {('+%.2f' % r.T) if r.T == r.T and r.T else '–'} | {kk} | "
                         f"{f(r.TODAY, r.TODAY_why)} | {f(r.REPLICA, r.REPLICA_why)} | "
                         f"{('%+.2f' % r.booked) if r.booked == r.booked and r.booked is not None else r.status} ({r.why_live}) |")
        bad = F[(F.REPLICA - F.booked).abs() > 0.15]
        if len(bad):
            L.append(f"⚠ replica ≠ booked on {len(bad)} fill(s) by > 0.15 pt ({', '.join(bad.pair)}) — read those with care.")
        if v[0] == "REVERT":
            L.append("⛔ GATE FIRED → REVERT: switch OFF the 🏃 WILLY runner (frenzy_willy_runner_enabled false) and log it in DECISION_LOG — operator applies (runner rows already open finish on the runner).")
    except Exception as e:
        L.append(f"Unavailable this run ({type(e).__name__}: {str(e)[:120]}).")
    return L + ["", f"Gate (frozen 2026-11-13, runner fills = touched their target): Σ (live − today's exit) < {EARLY_SUM:+.0f} pts at any read → revert · "
                    f"the first {FINAL_N} runner fills Σ < 0 → revert, else keep. Study: reports/WILLY_RUNNER_FLOOR_1113_rows.csv.", ""]


if __name__ == "__main__":
    print("\n".join(run(fetch_ok="--no-fetch" not in sys.argv)))
