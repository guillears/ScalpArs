#!/usr/bin/env python3
"""📏 Scout — FRENZY_LONG RE-SIZE read across batches (DECISION_LOG 351, operator 2026-10-23: "yes add the tracker"; OBSERVE-only, never
changes config, never calls the bot API, no network).

WHY             FRENZY_LONG runs at BLOCKER size (invest 0.2 · lev 0.05 ≈ $100, DL 289) with the −4 stop live since DL 317. On the 32 master
                fills (B16–B23, re-walked at today's +1.5 TP) −4 would have booked +0.22 %/fill vs −0.27 at −3 (5 of 12 stops saved); the two-year backtest is flat at
                −4 (623 · 73 % · −0.04). If the live −4 fills keep the master result, LONG earns money in its own right → size it up.
LOCKED GATE     the DL 317 revert gate + RE-SIZING read, frozen 2026-10-19 — services.frenzy.frenzy_long_stop4_gate, called VERBATIM (the
                dashboard's FRENZY Fills row runs the same function on the live DB only; a reset censors that tally, this scout counts
                every export): REVERT to −3 if < 3 of the first 10 −3 touches recover / the main touch sample is net negative vs −3 ·
                RE-SIZE at ≥ 20 live −4 fills on ≥ 8 UTC days: avg > 0 at day-bootstrap P ≥ 0.80 ∧ WR ≥ 73 % ∧ no day / pair ≥ 50 % of
                the gain → PROPOSE one step (blocker → invest 1.0 at lev 0.05; operator decides; own revert: the next 15 at the new size
                average < 0 or a UTC day's LONG fills lose > 10 % of the book). The locked bars never move (CLAUDE.md).
CONFIDENCE      INFO line, never a gate (DL 351): the stricter bars proposed 2026-10-23 before the locked read was found — N ≥ 30 on ≥ 10
                UTC days · WR > the cohort's own breakeven (|avg loss| / (avg win + |avg loss|); +1.5 / −4 with fees ≈ 74 %) · avg ≥ +0.10 %
                with day-bootstrap P(mean > 0) ≥ 0.95 · Σ $ > 0. Shown next to the locked verdict + the 30–50 % in-sample haircut of the
                avg, so the operator sees how strong a RESIZE is.
COHORT          CLOSED FRENZY_LONG LONG fills with entry_frenzy_stop_pct ≥ 3.5 (= ran the live −4 stop) from the orders exports (paper + live
                via export_glob; dedup (opened_at, pair, direction), CLOSED wins, never id); operator-closed (MANUAL / MANUAL_* /
                EXTERNAL_CLOSE) excluded — scout_gatekit.
Usage           venv/bin/python scripts/scout_long_resize.py           (batch review)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import export_glob as EG                                     # noqa: E402
import scout_gatekit as _GK                                  # noqa: E402
from services.frenzy import frenzy_long_stop4_gate           # noqa: E402

DL = 351
STRATEGY = "FRENZY_LONG"
LIVE4_MIN_STOP = 3.5                                         # entry_frenzy_stop_pct ≥ this = the live −4 stop (scout_long_stop4.LIVE4_MIN_STOP)
CONF_N, CONF_DAYS, CONF_AVG, CONF_P = 30, 10, 0.10, 0.95     # info bars (DL 351) — never a gate
BE_FALLBACK = 74.1                                           # +1.44 win / −4.12 loss net of fees → 4.12 / 5.56
BOOT_N, BOOT_SEED = 4000, 7
PATTERN = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "is_paper",
        "entry_frenzy_stop_pct", "touch3_at", "touch3_recovered_tp")


def _ts(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:19].str.replace(" ", "T"), errors="coerce", format="%Y-%m-%dT%H:%M:%S")


def _tri(v):
    """touch3_recovered_tp → True / False / None (unknown)."""
    s = str(v).strip().lower()
    return True if s in ("true", "1", "1.0") else False if s in ("false", "0", "0.0") else None


def load(paths=None):
    """every export row (paper + live), dedup (opened_at, pair, direction) — a CLOSED row wins, then the newest export."""
    paths = paths if paths is not None else sorted(EG.glob_exports(PATTERN), key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "pct", "usd", "stop"])
    a = pd.concat(fr, ignore_index=True)
    for c in COLS:
        if c not in a:
            a[c] = None
    a = a.assign(_c=(a.status.astype(str) == "CLOSED").astype(int),
                 _k=a.opened_at.astype(str).str[:19].str.replace(" ", "T")).reset_index(drop=True)   # one key across 19 / 26-char exports
    a = a.assign(_i=range(len(a))).sort_values(["_c", "_i"]).drop_duplicates(["_k", "pair", "direction"], keep="last")
    a["o"] = _ts(a.opened_at).values
    a["pct"] = pd.to_numeric(a.pnl_percentage, errors="coerce")
    a["usd"] = pd.to_numeric(a.pnl, errors="coerce")
    a["stop"] = pd.to_numeric(a.entry_frenzy_stop_pct, errors="coerce")
    return a[a.o.notna()].drop(columns=["_c", "_i", "_k"]).sort_values("o").reset_index(drop=True)


def unstamped(a):
    """CLOSED FRENZY_LONG LONG fills without a readable stop stamp (pre-317 exports / the −3 era) — shown, never counted."""
    if not len(a):
        return 0
    return int(((a.status.astype(str) == "CLOSED") & (a.entry_strategy.astype(str) == STRATEGY)
                & (a.direction.astype(str).str.upper() == "LONG") & ~(a.stop >= LIVE4_MIN_STOP)).sum())


def cohort(a):
    """→ (live −4 FRENZY_LONG kept, operator-closed) — CLOSED LONG fills, stamp ≥ 3.5, pnl known."""
    if not len(a):
        return a, a
    c = a[(a.status.astype(str) == "CLOSED") & (a.entry_strategy.astype(str) == STRATEGY) & (a.direction.astype(str).str.upper() == "LONG")
          & (a.stop >= LIVE4_MIN_STOP)]
    kept, ox = _GK.split_operator_closed(c)
    return kept[kept.pct.notna()].sort_values("o").reset_index(drop=True), ox


def gate_rows(c):
    """cohort → the (opened_at, pair, pnl %, stop, touch3_at, recovered) tuples frenzy_long_stop4_gate reads (dashboard parity)."""
    t3 = _ts(c.touch3_at) if len(c) else pd.Series([], dtype="datetime64[ns]")
    return [(o.to_pydatetime(), p, float(x), float(s), (t.to_pydatetime() if pd.notna(t) else None), _tri(r))
            for o, p, x, s, t, r in zip(c.o, c.pair, c.pct, c.stop, t3, c.touch3_recovered_tp)]


def day_boot_p(vals, days, nb=BOOT_N, seed=BOOT_SEED):
    """P(per-fill mean > 0) under a UTC-day block bootstrap (scout_long_stop4.day_boot's resampling)."""
    v, dd = np.asarray(vals, float), np.asarray(days)
    groups = [v[dd == x] for x in pd.unique(dd)]
    if not groups:
        return np.nan
    s = np.array([g.sum() for g in groups]); n = np.array([len(g) for g in groups])
    idx = np.random.default_rng(seed).integers(0, len(groups), size=(nb, len(groups)))
    return float(((s[idx].sum(1) / n[idx].sum(1)) > 0).mean())


def confidence(c):
    """the DL 351 info bars → (all_pass, text). Never raises; never a verdict."""
    try:
        n = len(c)
        if n == 0:
            return False, "no live −4 fills yet"
        days = c.o.dt.date
        nd = days.nunique()
        avg = float(c.pct.mean()); wr = 100.0 * float((c.pct > 0).mean())
        w, l = c.pct[c.pct > 0], c.pct[c.pct <= 0]
        be = 100.0 * abs(l.mean()) / (w.mean() + abs(l.mean())) if len(w) and len(l) else BE_FALLBACK
        p = day_boot_p(c.pct.values, days.values)
        usd = float(c.usd.sum(skipna=True))
        chk = [(n >= CONF_N and nd >= CONF_DAYS, f"N {n}/{CONF_N} on {nd}/{CONF_DAYS} days"),
               (wr > be, f"WR {wr:.0f} % vs breakeven {be:.0f} %"),
               (avg >= CONF_AVG, f"avg {avg:+.3f} % (≥ {CONF_AVG:+.2f}; haircut 30–50 % → {min(avg * 0.5, avg * 0.7):+.3f} … {max(avg * 0.5, avg * 0.7):+.3f})"),
               (p >= CONF_P, f"day-boot P {p:.2f} (≥ {CONF_P:.2f})"),
               (usd > 0, f"Σ ${usd:+.2f}")]
        ok = all(k for k, _ in chk)
        return ok, " · ".join(("✓ " if k else "✗ ") + t for k, t in chk)
    except Exception as e:
        return False, f"unreadable ({str(e)[:80]})"


def run(now_ms=None, paths=None):
    L = [f"## 📏 FRENZY_LONG RE-SIZE read (locked DECISION_LOG 317 · scout DECISION_LOG {DL}; observe-only, operator decides)", ""]
    a = load(paths)
    c, ox = cohort(a)
    L.append(f"Cohort: CLOSED FRENZY_LONG fills on the live −4 stop (entry_frenzy_stop_pct ≥ {LIVE4_MIN_STOP}), every export · "
             f"{_GK.operator_closed_line(ox)} · not counted: {unstamped(a)} −3-era / unstamped FRENZY_LONG fills")
    if len(c):
        paper = int(c.is_paper.astype(str).str.strip().str.lower().isin(("true", "1", "1.0")).sum())
        L.append(f"- −4 stop fills: N {len(c)} (live {len(c) - paper} · paper {paper}) · {c.o.dt.date.nunique()} days · WR {(c.pct > 0).mean():.0%} · "
                 f"avg {c.pct.mean():+.3f} % · Σ {c.pct.sum():+.2f} pts · ${c.usd.sum(skipna=True):+.2f}")
    v, txt = frenzy_long_stop4_gate(gate_rows(c))
    L.append(f"Locked gate (DL 317, same function as the dashboard; N can exceed the dashboard's — every export, resets included, "
             f"operator-closed out): {v} — {txt}")
    ok, ctxt = confidence(c)
    L.append(f"Confidence (DL {DL}, info only — never moves the locked gate): {'STRONG' if ok else 'not yet'} — {ctxt}")
    L.append("Size note: the blocker role is size-independent (LITE is refused on any held pair); a step up changes only LONG's own $.")
    return L + [""]


def main():
    print("\n".join(run(int(time.time() * 1000))))


if __name__ == "__main__":
    main()
