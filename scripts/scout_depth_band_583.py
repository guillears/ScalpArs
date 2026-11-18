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
584 NOTE    (deep review I3) the slippage legs read LIVE rows only (is_paper false) — from DL 584 an impact-mode PAPER fill books its modelled
            walk in its entry price, so a paper slip is the model, not evidence. The cohort is split by entry_depth_mode in the output: from
            the DL 584 deploy the band mixes sweep-sized (pre-584) and impact-sized fills (confound — impact sizes on the average walk, so a
            0.25–0.5 band fill is a different trade than under sweep); the frozen verdict reads all band fills as before.
STATE       reports/SCOUT_DEPTH_BAND_583.json (gitignored) — the verdict frozen once.
588 NOTE    (DECISION_LOG 588) with entry_depth_guard_min_usd > 0 a thin book shrinks below 0.25 of the plan instead of skipping → the band above
            also holds those fills from the DL 588 deploy. Its OWN line (frozen once, reports/SCOUT_DEPTH_FLOOR_588.json, gitignored): the
            first 20 CLOSED bot fills with entry_cap_frac < 0.25 opened ≥ the DL 588 commit + 10 min → mean booked net % < 0 OR mean LIVE
            |entry slippage| > 0.15 % → flag "revert entry_depth_guard_min_usd to 0" (alert only) · else KEEP.
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
        "entry_depth_shrunk_from", "entry_slippage_pct", "exit_slippage_pct", "is_paper", "entry_depth_mode", "entry_cap_frac",   # (588)
        "is_probe", "cell_multiplier_source", "pattern_cell_source")   # (588 re-review 12) probe fills out of the 588 line
STATE = os.path.join(_R, "reports", "SCOUT_DEPTH_BAND_583.json")
STATE_588 = os.path.join(_R, "reports", "SCOUT_DEPTH_FLOOR_588.json")
DL_588, GATE_N_588, FRAC_588 = 588, 20, 0.25


_TRUE = ("true", "1", "1.0", "yes")


def is_probe_row(r) -> bool:
    """(588 re-review 12) a test-size fill: is_probe true, or a *_PROBE strategy / cell source (the full-size-only rule; FRENZY_WILLY_HT is
    its own reduced-size sleeve, DL 341, so its is_probe flag is ignored). Pure."""
    es = str(getattr(r, "entry_strategy", "") or "").upper()
    if es != "FRENZY_WILLY_HT" and str(getattr(r, "is_probe", "") or "").strip().lower() in _TRUE:
        return True
    return any("PROBE" in str(getattr(r, c, "") or "").upper() for c in ("entry_strategy", "cell_multiplier_source", "pattern_cell_source"))


# UNIT (588 re-review 12): Order.pnl_percentage = net P&L (fees + funding in) ÷ NOTIONAL × 100 — a PRICE %, unleveraged (NOT the ROE on
# margin); entry_slippage_pct is a price % too, so both halves of the gate read the same unit.
def gate_588(rows):
    """rows = [(booked net % of notional, live entry slip % or None)] of the deep-shrunk fills in open order → (verdict, text). Pure."""
    if len(rows) < GATE_N_588:
        return "COLLECTING", f"⏳ {len(rows)}/{GATE_N_588} fills with entry_cap_frac < {FRAC_588}"
    cut = rows[:GATE_N_588]
    ps = [p for p, _ in cut if p is not None]
    m = sum(ps) / len(ps) if ps else 0.0
    sl = [abs(x) for _, x in cut if x is not None]
    ms = sum(sl) / len(sl) if sl else None
    info = (f"first {GATE_N_588} full-size fills < {FRAC_588} of the plan: mean booked net {m:+.3f} % of notional (unleveraged price %, fees + "
            f"funding in) · mean live |entry slip| "
            + (f"{ms:.3f} %" if ms is not None else "n/a (no live fill)"))
    if m < 0 or (ms is not None and ms > SLIP_MAX):
        return "REVERT", f"🔴 {info} → flag: revert entry_depth_guard_min_usd to 0 (operator decides)"
    return "KEEP", f"🟢 {info} → keep the $ floor mode"


def report_588(frame=None, since_ms=None, state_path=STATE_588, now_ms=None):
    """📖💵 (DECISION_LOG 588) the deep-shrink line (frozen once, alert only)."""
    out = [f"## 📖💵 DL {DL_588} depth guard $ floor — fills shrunk below {FRAC_588} of the plan (alert only)"]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL_588, lag_min=10, label="DL 588 depth $ floor")
    if since_ms is None:
        return "\n".join(out + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL_588} commit is not in git yet; nothing is counted."])
    a = load() if frame is None else frame
    if "entry_cap_frac" not in a.columns:
        a = a.assign(entry_cap_frac=None)
    a = a[a.o >= pd.Timestamp(int(since_ms), unit="ms")] if len(a) else a
    deep = a[[(_f(x) is not None and _f(x) < FRAC_588) for x in a.entry_cap_frac]] if len(a) else a
    deep = deep[[not is_probe_row(r) for r in deep.itertuples()]] if len(deep) else deep   # (re-review 12) full-size fills only
    _live = lambda r: str(getattr(r, "is_paper", "")).strip().lower() in ("false", "0", "0.0", "no")
    rows = [(_f(r.pnl_percentage), (_f(r.entry_slippage_pct) if _live(r) else None)) for r in deep.itertuples()]
    st = {}
    try:
        with open(state_path) as f:
            st = json.load(f)
    except FileNotFoundError:
        st = {}
    except Exception:
        st = None
    if st is not None and "verdict" not in st and len(rows) >= GATE_N_588:
        v, txt = gate_588(rows)
        st = dict(verdict=v, detail=txt, run_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime((now_ms or time.time() * 1000) / 1000)))
        try:
            with open(state_path, "w") as f:
                json.dump(st, f)
        except Exception:
            pass
    if st and st.get("verdict"):
        out.append(f"FROZEN ({st.get('run_at')}): {st['verdict']} — {st.get('detail')}")
    v, txt = gate_588(rows)
    out.append(f"now: {v} — {txt}")
    return "\n".join(out)


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
    if "entry_depth_mode" not in a.columns:   # (584) older frames / exports carry no mode stamp
        a = a.assign(entry_depth_mode=None)
    a = a[a.o >= pd.Timestamp(int(since_ms), unit="ms")] if len(a) else a
    band = a[[in_band(n, p) for n, p in zip(a.notional_value, a.entry_depth_shrunk_from)]] if len(a) else a
    _live = lambda r: str(getattr(r, "is_paper", "")).strip().lower() in ("false", "0", "0.0", "no")   # (584 I3) slip legs: LIVE only
    rows = [(_f(r.pnl), (_f(r.entry_slippage_pct) if _live(r) else None), (_f(r.exit_slippage_pct) if _live(r) else None)) for r in band.itertuples()]
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
    if len(band):   # (584 I3) the cohort by depth-guard mode (the impact deploy is a confound inside the band)
        _m = band.entry_depth_mode.fillna("pre-584").astype(str).value_counts()
        out.append("  by depth mode: " + " · ".join(f"{k} {v}" for k, v in _m.items()) + " (slippage legs: live rows only)")
    for r in band.itertuples():
        out.append(f"  {str(r.o)[:16]} {r.pair} {r.entry_strategy} {r.direction}: ${_f(r.notional_value) or 0:,.0f} of ${_f(r.entry_depth_shrunk_from) or 0:,.0f} "
                   f"planned · pnl ${_f(r.pnl) or 0:+,.2f} · slip in {r.entry_slippage_pct} / out {r.exit_slippage_pct}")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
    print()
    print(report_588())   # (DL 588 review 16)
