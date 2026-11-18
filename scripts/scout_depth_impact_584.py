#!/usr/bin/env python3
"""📖📐 Scout — DL 584 impact-capped depth sizing revert gate (DECISION_LOG 584, 2026-11-18, operator "build it now"; ALERT-only, never
changes config).

CHANGE      trading_config.json investment.entry_depth_guard_mode "sweep" → "impact": the depth guard keeps the notional to the largest $
            whose AVERAGE walk (VWAP of the book levels consumed vs the best price) ≤ entry_depth_guard_max_avg_impact_pct (0.15 %; no level
            beyond 3 × it) on the thinner side, instead of only the book within ±max_sweep_pct (0.15 %). Each guarded entry stamps the modelled walk of its
            final notional (entry_depth_walk_pct) and the mode (entry_depth_mode).
REFERENCE   (deep review I4) both numbers are measured from the SAME book read, against its best price on the side the entry hits:
            modelled = entry_depth_walk_pct (VWAP of the levels the final notional consumes vs that best price); realised =
            entry_fill_vs_book_pct (the booked fill vs that best far-side price: LONG fill / best ask − 1, SHORT 1 − fill / best bid; + = worse).
            NOT entry_slippage_pct (vs the decision price — it adds the spread and the drift since the signal). Only the book-read sleeves
            (FRENZY / SURGE / FLATBASE / EXHAUSTION) stamp entry_fill_vs_book_pct; other sleeves are excluded from the gate (no same-read ruler).
COHORT      LIVE bot fills (is_paper false / live export) with entry_depth_mode == "impact", a stamped walk and entry_fill_vs_book_pct,
            entry_order_type TAKER or TAKER+MAKER (a capped order's resting part fills at or better than its limit), opened ≥ the DL 584
            commit + 10 min; dedup (opened_at, pair, direction), CLOSED beats OPEN; any status (the entry cost is known at the open). Paper
            rows are listed as information only (paper books the modelled walk itself — no real book impact).
GATE        decided ONCE at the first 20 live impact fills (by open time): mean (realised fill vs book − modelled walk) > +0.05 % (the model
            under-states the cost) · or mean realised fill vs book > max_avg_impact + 0.05 % → flag "revert entry_depth_guard_mode to sweep"
            (operator decides) · else KEEP. Before 20: COLLECTING. Also shown (information, deep review I1): the live capped orders' fill share
            (entry_cap_fill_frac) and the capped NO-FILL count (DEPTH_CAP_NOFILL blocked records of the decisions export, live).
            Exit slippage is NOT read here — it is read by hand at the batch review (stops stay market orders).
CAPACITY    (operator addition) a per-sleeve capacity table for EXHAUSTION_SHORT and FLATBASE_SHORT (every guarded fill since the DL 584
            commit, both modes, paper + live): N · median planned / final $ · % shrunk · skipped (the journals' cap_action == SKIP rows of the
            decisions exports: e=EXHAUSTION exh_cap_action / e=FLATBASE fb_cap_action) · binding side mix · median modelled walk vs realised
            entry slippage (services.entry_depth.capacity_rows — the same table the dashboard / both text exports show). Information only.
STATE       reports/SCOUT_DEPTH_IMPACT_584.json (gitignored) — the verdict frozen once.
Usage: venv/bin/python -I scripts/scout_depth_impact_584.py"""
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

DL = 584
GATE_N, GAP_MAX, OVER_CAP = 20, 0.05, 0.05
IMPACT_CAP_DEFAULT = 0.15
PATTERNS = (os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv"), os.path.expanduser("~/Downloads/scalpars_orders_live_*.csv"))
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "notional_value", "entry_slippage_pct", "entry_depth_walk_pct",
        "entry_depth_mode", "is_paper", "trading_mode", "entry_cap_planned_usd", "entry_cap_final_usd", "entry_cap_binding_side",
        "entry_fill_vs_book_pct", "entry_order_type", "entry_cap_fill_frac")
TAKER_TYPES = ("TAKER", "TAKER+MAKER")
DEC_PATTERNS = (os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"),)
DEC_COLS = ("t", "e", "pair", "exh_signal_at", "exh_cap_action", "fb_spike_at", "fb_cap_action", "bl_lines", "bl_is_paper")
FOCUS = ("EXHAUSTION_SHORT", "FLATBASE_SHORT")
STATE = os.path.join(_R, "reports", "SCOUT_DEPTH_IMPACT_584.json")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def is_live(is_paper, trading_mode=None) -> bool:
    """True for a LIVE fill: trading_mode says live, else is_paper is false. Unknown → False (never counted as live). Pure."""
    tm = str(trading_mode).strip().lower() if trading_mode is not None else ""
    if tm in ("live", "paper"):
        return tm == "live"
    s = str(is_paper).strip().lower()
    if s in ("false", "0", "0.0", "no"):
        return True
    return False


def max_impact():
    """today's entry_depth_guard_max_avg_impact_pct (the gate's second leg reads it); unreadable → the 0.15 default."""
    try:
        with open(os.path.join(_R, "trading_config.json")) as f:
            v = _f(json.load(f).get("investment", {}).get("entry_depth_guard_max_avg_impact_pct"))
        return v if v is not None and 0 < v <= 1 else IMPACT_CAP_DEFAULT
    except Exception:
        return IMPACT_CAP_DEFAULT


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
    a = a[a.entry_strategy.astype(str).str.upper() != "MANUAL"]   # bot fills only
    a["o"] = pd.to_datetime(a.opened_at.astype(str).str[:19], errors="coerce")
    return a[a.o.notna()].sort_values("o").reset_index(drop=True)


def load_skips(paths=None, since_ms=None):
    """{sleeve: n} of the EXHAUSTION / FLATBASE journal rows whose depth guard SKIPPED (cap_action == SKIP), deduped per decision
    (EXHAUSTION: pair + signal time · FLATBASE: pair + spike), at / after since_ms. Missing exports → {}."""
    paths = paths if paths is not None else sorted({p for pat in DEC_PATTERNS for p in _EG.glob_exports(pat)}, key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in DEC_COLS))
        except Exception:
            continue
    if not fr:
        return {}
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(DEC_COLS))
    d["o"] = pd.to_datetime(d.t.astype(str).str[:19], errors="coerce")
    if since_ms is not None:
        d = d[d.o >= pd.Timestamp(int(since_ms), unit="ms")]
    out = {}
    ex = d[(d.e.astype(str) == "EXHAUSTION") & (d.exh_cap_action.astype(str).str.upper() == "SKIP")].drop_duplicates(["pair", "exh_signal_at"])
    fb = d[(d.e.astype(str) == "FLATBASE") & (d.fb_cap_action.astype(str).str.upper() == "SKIP")].drop_duplicates(["pair", "fb_spike_at"])
    if len(ex):
        out["EXHAUSTION_SHORT"] = int(len(ex))
    if len(fb):
        out["FLATBASE_SHORT"] = int(len(fb))
    return out


def load_nofills(paths=None, since_ms=None) -> int:
    """the LIVE capped-order NO-FILL count since since_ms: decisions-export e=BLOCKED rows whose lines carry DEPTH_CAP_NOFILL and bl_is_paper
    false (one row per pair / sleeve / 5m bar — a lower bound). Missing exports → 0."""
    paths = paths if paths is not None else sorted({p for pat in DEC_PATTERNS for p in _EG.glob_exports(pat)}, key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in DEC_COLS))
        except Exception:
            continue
    if not fr:
        return 0
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(DEC_COLS))
    d["o"] = pd.to_datetime(d.t.astype(str).str[:19], errors="coerce")
    if since_ms is not None:
        d = d[d.o >= pd.Timestamp(int(since_ms), unit="ms")]
    d = d[(d.e.astype(str) == "BLOCKED") & d.bl_lines.astype(str).str.contains("DEPTH_CAP_NOFILL", regex=False)
          & d.bl_is_paper.astype(str).str.lower().isin(("false", "0", "0.0"))]
    return int(len(d.drop_duplicates(["t", "pair"])))


def capacity_table(frame, skips=None):
    """📖📊 the per-sleeve capacity lines for EXHAUSTION_SHORT and FLATBASE_SHORT (guarded fills = entry_cap_planned_usd stamped). Pure."""
    from services.entry_depth import capacity_rows
    fills = []
    if frame is not None and len(frame):
        for r in frame.itertuples():
            if str(getattr(r, "entry_strategy", "")) in FOCUS and _f(getattr(r, "entry_cap_planned_usd", None)) is not None:
                fills.append(dict(sleeve=str(r.entry_strategy), planned=_f(r.entry_cap_planned_usd), final=_f(getattr(r, "entry_cap_final_usd", None)),
                                  binding=(None if pd.isna(getattr(r, "entry_cap_binding_side", None)) else str(r.entry_cap_binding_side)),
                                  walk=_f(r.entry_depth_walk_pct), slip=_f(r.entry_slippage_pct)))
    rows = [r for r in capacity_rows(fills, {k: v for k, v in (skips or {}).items() if k in FOCUS}) if r["sleeve"] in FOCUS]
    u = lambda v: f"${v:,.0f}" if v is not None else "-"
    p = lambda v, d: f"{v:.{d}f}" if v is not None else "-"
    out = ["### 📖📊 Capacity at entry — EXHAUSTION_SHORT · FLATBASE_SHORT (information; paper + live, both modes)",
           f"{'Sleeve':<18} {'N':>4} {'MedPlan$':>10} {'MedFinal$':>10} {'%shrunk':>8} {'Skip(%)':>12} {'Bound E/X/none':>15} {'MedWalk%':>9} {'MedSlip%':>9}"]
    for r in rows:
        out.append(f"{r['sleeve']:<18} {r['n']:>4} {u(r['med_planned']):>10} {u(r['med_final']):>10} {p(r['shrunk_pct'], 1):>8} "
                   f"{str(r['skipped']) + ' (' + p(r['skipped_pct'], 1) + ')':>12} {str(r['bind_entry']) + '/' + str(r['bind_exit']) + '/' + str(r['bind_none']):>15} "
                   f"{p(r['med_walk'], 3):>9} {p(r['med_slip'], 3):>9}")
    return out


def gate(rows, cap=IMPACT_CAP_DEFAULT):
    """rows = [(realised fill vs book %, modelled walk %)] of LIVE impact fills in open order → (verdict, text). Pure."""
    rows = [(s, w) for s, w in rows if s is not None and w is not None]
    if len(rows) < GATE_N:
        return "COLLECTING", f"⏳ {len(rows)}/{GATE_N} live impact-mode fills"
    cut = rows[:GATE_N]
    gap = sum(s - w for s, w in cut) / GATE_N
    ms = sum(s for s, _ in cut) / GATE_N
    mw = sum(w for _, w in cut) / GATE_N
    info = (f"first {GATE_N} live impact fills: mean realised fill vs book {ms:+.3f} % · mean modelled walk {mw:.3f} % · "
            f"realised − modelled {gap:+.3f} % (bar +{GAP_MAX}) · slip bar {cap + OVER_CAP:.2f} %")
    if gap > GAP_MAX + 1e-12 or ms > cap + OVER_CAP + 1e-12:
        return "REVERT", f"🔴 {info} → flag: revert entry_depth_guard_mode to sweep (operator decides)"
    return "KEEP", f"🟢 {info} → keep impact mode"


def report(frame=None, since_ms=None, state_path=STATE, now_ms=None, cap=None, skips=None, nofills=None):
    out = [f"## 📖📐 DL {DL} impact-capped depth sizing (mode sweep → impact) — revert gate (alert only)"]
    cap = max_impact() if cap is None else cap
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="DL 584 depth impact")
    if since_ms is None:
        return "\n".join(out + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted."])
    a = load() if frame is None else frame
    for c in COLS:   # an export from before a column existed reads as unstamped
        if c not in a.columns:
            a = a.assign(**{c: None})
    a = a[a.o >= pd.Timestamp(int(since_ms), unit="ms")] if len(a) else a
    imp = a[a.entry_depth_mode.astype(str).str.lower() == "impact"] if len(a) else a
    tm = imp.trading_mode if "trading_mode" in imp.columns else [None] * len(imp)
    live_mask = [is_live(p, t) for p, t in zip(imp.is_paper, tm)] if len(imp) else []
    live = imp[live_mask] if len(imp) else imp
    paper = imp[[not x for x in live_mask]] if len(imp) else imp
    _tk = (live.entry_order_type.astype(str).str.upper().isin(TAKER_TYPES)) if len(live) else []
    live_g = live[_tk] if len(live) else live   # (I4) TAKER / TAKER+MAKER only; rows without fill-vs-book drop out in gate()
    rows = [(_f(r.entry_fill_vs_book_pct), _f(r.entry_depth_walk_pct)) for r in live_g.itertuples()]
    st = {}
    try:
        with open(state_path) as f:
            st = json.load(f)
    except FileNotFoundError:
        st = {}
    except Exception:
        st = None
    n_ok = sum(1 for s, w in rows if s is not None and w is not None)
    if st is not None and "verdict" not in st and n_ok >= GATE_N:
        v, txt = gate(rows, cap)
        st = dict(verdict=v, detail=txt, run_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime((now_ms or time.time() * 1000) / 1000)))
        try:
            with open(state_path, "w") as f:
                json.dump(st, f)
        except Exception:
            pass
    if st and st.get("verdict"):
        out.append(f"FROZEN ({st.get('run_at')}): {st['verdict']} — {st.get('detail')}")
    v, txt = gate(rows, cap)
    out.append(f"now: {v} — {txt}" + (" (information; the frozen read decides)" if st and st.get("verdict") else ""))
    _ff = [x for x in (_f(v) for v in (live.entry_cap_fill_frac if len(live) else [])) if x is not None]
    try:
        _nf = load_nofills(since_ms=since_ms) if nofills is None else int(nofills)
    except Exception:
        _nf = None
    out.append(f"  live capped orders (I1): fill share mean {(sum(_ff) / len(_ff)) if _ff else float('nan'):.2f} over {len(_ff)} · "
               f"partial (< 1) {sum(1 for x in _ff if x < 0.999)} · NO-FILL {_nf if _nf is not None else '?'} (DEPTH_CAP_NOFILL, lower bound)")
    for r in live.itertuples():
        out.append(f"  LIVE  {str(r.o)[:16]} {r.pair} {r.entry_strategy} {r.direction} {r.entry_order_type}: ${_f(r.notional_value) or 0:,.0f} · "
                   f"fill vs book {r.entry_fill_vs_book_pct} % vs modelled walk {r.entry_depth_walk_pct} % · decision slip {r.entry_slippage_pct} % · "
                   f"cap fill {r.entry_cap_fill_frac}")
    try:
        out += capacity_table(a, load_skips(since_ms=since_ms) if skips is None else skips)
    except Exception as e:   # the gate above never depends on the table
        out.append(f"  capacity table unavailable ({str(e)[:100]})")
    if len(paper):
        out.append(f"  paper impact-mode fills (information only — paper books the modelled walk, no real book impact): {len(paper)}")
        for r in paper.tail(10).itertuples():
            out.append(f"  paper {str(r.o)[:16]} {r.pair} {r.entry_strategy} {r.direction}: ${_f(r.notional_value) or 0:,.0f} · "
                       f"slip {r.entry_slippage_pct} % · modelled walk {r.entry_depth_walk_pct} %")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
