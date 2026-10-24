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
RE-SIZE LADDER  📈 (DECISION_LOG 365, operator 2026-10-24; frozen, operator decides — nothing automatic): frenzy_willy_ht_size_frac
                0.25 → 0.5 → 1.0 of WILLY's margin. Judged under TODAY's exit (V2b: fills opened ≥ the DL 358 deploy + 10 min, runners
                included) on the CURRENT run at this size (entry_willy_ht_size_frac; unstamped HT fills opened before the DL 365 deploy = 0.25,
                the only value since DL 341) vs full-size WILLY fills of the same period, with the SAME frenzy_willy_ht_gate. Frozen sample =
                30 on ≥ 10 days with ≥ 10 WILLY comparators. Pattern-W (WR ≥ 70 % ∧ avg ≥ +0.10 ∧ Σ > 0) ∧ HT ≥ WILLY − 0.10 → STEP-UP
                candidate (0.5 → 1.0 also re-decides frenzy_willy_ht_max_slots) · REMOVE_FILTER (HT ≥ WILLY) + Pattern-W → full size by the
                DL 341 rule, ADVISORY (needs the DL 341 gate or an operator declaration) · CONFIRMED / KILL / FIRST / frozen avg < 0 / first-60
                avg < 0 → STEP-DOWN candidate (at 0.25 the DL 341 rule = HT off). Off the ladder → no verdict.
Usage           venv/bin/python scripts/scout_willy_ht.py           (batch review; no network, no bot API)
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
import scout_gatekit as _GK                                  # noqa: E402
import scout_willy_hold_off as _HO                           # noqa: E402  (the shared export loader: dedup, CLOSED wins, paper + live)
import willy_exit_era as WEra                                # noqa: E402  🎲² (DECISION_LOG 356) exit-era split / header
from services.frenzy import (frenzy_willy_ht_gate, frenzy_willy_ht_frac, WILLY_HT_MAIN_N, WILLY_HT_MAIN_DAYS,   # noqa: E402
                             WILLY_HT_MIN_WILLY, WILLY_HT_REREAD_N)

DL = 341
DL_V2B, DL_STAMP = 358, 365
LADDER = (0.25, 0.5, 1.0)                    # ¼ (DL 341) → ½ → full WILLY margin
LADDER_TOL = 0.005
STEP_GAP = 0.10                              # step up iff HT ≥ full-size WILLY avg − 0.10 on the frozen sample …
STEP_MIN_AVG, STEP_MIN_WR = 0.10, 0.70       # … AND the locked Pattern-W multiplier gate (CLAUDE.md: WR ≥ 70 % ∧ avg ≥ +0.10 % ∧ Σ > 0)
PRE_STAMP_FRAC = 0.25                        # every HT fill before the DL 365 stamp opened at 0.25 (the only value since DL 341)


def _frozen_sample(f):
    """the gate's frozen main-read sample: the shortest prefix with ≥ WILLY_HT_MAIN_N fills on ≥ WILLY_HT_MAIN_DAYS UTC days, else None."""
    ds = set()
    for k, d in enumerate(f.o.dt.date):
        ds.add(d)
        if k + 1 >= WILLY_HT_MAIN_N and len(ds) >= WILLY_HT_MAIN_DAYS:
            return f.iloc[:k + 1]
    return None


def resize_lines(a, v2b_ms, stamp_ms, frac_now=None):
    """📈 (DECISION_LOG 365) the frozen WILLY_HT re-size ladder → printed lines. a = the shared loader frame; v2b_ms / stamp_ms = the DL 358 /
    DL 365 deploy floors (ms, None = not deployed). Never raises."""
    try:
        if v2b_ms is None:
            return ["📈 HT re-size ladder: NOT evaluable — the DECISION_LOG 358 deploy is not in git"]
        if a is None or not len(a) or "o" not in a:
            return ["📈 HT re-size ladder: no exports"]
        if frac_now is None:
            import json as _j
            from types import SimpleNamespace as _NS
            with open(os.path.join(_R, "trading_config.json"), encoding="utf-8") as fh:
                frac_now = frenzy_willy_ht_frac(_NS(**_j.load(fh)["thresholds"]))   # the engine's own reading
        frac_now = float(frac_now)
        lvl = min(range(len(LADDER)), key=lambda i: abs(LADDER[i] - frac_now))
        nxt = LADDER[lvl + 1] if lvl + 1 < len(LADDER) else None
        since = pd.Timestamp(int(v2b_ms), unit="ms")
        c = a[(a.status.astype(str) == "CLOSED") & (a.o >= since)] if len(a) else a
        ht, _ = _GK.split_operator_closed(c[c.es == "FRENZY_WILLY_HT"]) if len(c) else (c, c)
        w, _ = _GK.split_operator_closed(c[c.es == "FRENZY_WILLY"]) if len(c) else (c, c)
        ht = ht.sort_values("o")
        sz = pd.to_numeric(ht.get("entry_willy_ht_size_frac"), errors="coerce") if len(ht) else pd.Series(dtype=float)
        if len(ht):
            pre = ht.o < pd.Timestamp(int(stamp_ms), unit="ms") if stamp_ms is not None else pd.Series(True, index=ht.index)
            sz = sz.where(sz.notna() | ~pre, PRE_STAMP_FRAC)
        ht = ht.assign(_sz=sz.values if len(ht) else [])
        n_unk = int(ht._sz.isna().sum()) if len(ht) else 0
        known = ht[ht._sz.notna()] if len(ht) else ht
        other = known[(known._sz - LADDER[lvl]).abs() > LADDER_TOL]
        at = known[(known._sz - LADDER[lvl]).abs() <= LADDER_TOL]
        if len(other):
            at = at[at.o > other.o.max()]
        at = at[np.isfinite(pd.to_numeric(at.pct, errors="coerce"))]                     # the gate drops non-finite pct — the frozen sample must be the gate's own prefix
        days = at.o.dt.date.nunique() if len(at) else 0
        L = [f"📈 HT re-size ladder (DECISION_LOG 365, frozen, under today's V2b exit): {' → '.join(f'{x:g}' for x in LADDER)} of WILLY's margin · "
             f"now {frac_now:g} · current run at this size: {len(at)} fills on {days} days" + (f" · avg {at.pct.mean():+.3f} %" if len(at) else "")
             + (f" · ⚠ {n_unk} HT fill(s) with unknown size not counted" if n_unk else "")]
        if len(known) and abs(float(known._sz.iloc[-1]) - frac_now) > LADDER_TOL:
            L.append(f"  ⚠ the newest HT fill opened at {float(known._sz.iloc[-1]):g} but trading_config.json says {frac_now:g} — sync the config")
        if abs(LADDER[lvl] - frac_now) > LADDER_TOL:
            L.append(f"  ⚠ {frac_now:g} is OFF the ladder — no step verdict (set frenzy_willy_ht_size_frac to a rung)")
            return L
        wl = [(r.o.to_pydatetime(), r.pct) for r in w.itertuples()]
        v, txt = frenzy_willy_ht_gate([(r.o.to_pydatetime(), r.pct) for r in at.itertuples()], wl)
        L.append(f"  gate on this run: {v} — {txt}")
        smp = _frozen_sample(at) if len(at) else None
        h_avg = w_avg = None
        wv = w.pct.iloc[0:0]
        # the step bars read the FROZEN first-30 sample even after the gate moves to its 60 re-read (frozen = never re-fit)
        if smp is not None:
            wv = w[(w.o >= smp.o.iloc[0]) & (w.o <= smp.o.iloc[-1])].pct.dropna()
            h_avg = float(smp.pct.mean())
            w_avg = float(wv.mean()) if len(wv) >= WILLY_HT_MIN_WILLY else None
        # a losing HT run steps down even when WILLY did worse (gate REMOVE_FILTER with HT avg < 0): the size follows HT's own expectancy
        # … and once the gate re-reads on 60, a negative first-60 average also steps down (the step-UP bar stays on the frozen 30)
        down = (v in ("CONFIRMED", "KILL", "FIRST") or (h_avg is not None and h_avg < 0)
                or (len(at) >= WILLY_HT_REREAD_N and float(at.pct.iloc[:WILLY_HT_REREAD_N].mean()) < 0))
        if down:
            L.append(f"  🛑 STEP-DOWN candidate to {LADDER[lvl - 1]:g} (operator decides)" if lvl > 0
                     else "  🛑 at 0.25 the DL 341 rule applies: review / switch HT off (operator decides)")
            return L
        h_wr = float((smp.pct > 0).mean()) if smp is not None else None
        bar = f"needs ≥ WILLY − {STEP_GAP:.2f} ∧ the Pattern-W gate: WR ≥ {STEP_MIN_WR:.0%} ∧ avg ≥ +{STEP_MIN_AVG:.2f} % ∧ Σ > 0"
        pat_w = h_avg is not None and h_avg >= STEP_MIN_AVG and h_wr >= STEP_MIN_WR and float(smp.pct.sum()) > 0
        if v == "REMOVE_FILTER" and not pat_w:
            L.append(f"  HT ≥ full-size WILLY (the DL 341 filter question) but the size bar fails: frozen HT avg "
                     + (f"{h_avg:+.3f} % · WR {h_wr:.0%}" if h_avg is not None else "n/a") + f" ({bar}) — no size step")
            return L
        if v == "REMOVE_FILTER":
            L.append("  ✅ HT ≥ full-size WILLY under today's exit → full size by the DL 341 rule (turnover ratio 0 + HT off) — ADVISORY: the filter "
                     "change needs the registered DL 341 gate to agree, or the operator to declare this V2b read the governing one")
            return L
        if nxt is None:
            L.append("  top of the ladder (full WILLY margin) — no further step")
            return L
        if pat_w and w_avg is not None and h_avg >= w_avg - STEP_GAP:
            L.append(f"  ✅ STEP-UP candidate {LADDER[lvl]:g} → {nxt:g} (operator decides): frozen {len(smp)} HT avg {h_avg:+.3f} % · WR {h_wr:.0%} "
                     f"vs WILLY {w_avg:+.3f} %" + (" · ALSO re-decide frenzy_willy_ht_max_slots (at 1.0 HT = a second full WILLY ticket; the "
                                                   "Σ ≤ −8 pts kill costs 4× the dollars of 0.25)" if nxt >= 1.0 else ""))
        elif h_avg is not None and w_avg is not None:
            L.append(f"  no step: frozen {len(smp)} HT avg {h_avg:+.3f} % · WR {h_wr:.0%} vs WILLY {w_avg:+.3f} % ({bar})")
        else:
            L.append(f"  next step {LADDER[lvl]:g} → {nxt:g} at the frozen {WILLY_HT_MAIN_N} on ≥ {WILLY_HT_MAIN_DAYS} days with ≥ "
                     f"{WILLY_HT_MIN_WILLY} same-period WILLY comparators (now {len(at)} HT"
                     + (f" · {len(wv)} same-period WILLY)" if smp is not None else f" · {len(w)} WILLY in the V2b era)"))
        return L
    except Exception as e:
        return [f"📈 HT re-size ladder: unreadable ({str(e)[:80]})"]


def cohorts(a, since):
    """→ (ht kept, willy kept, ht operator-closed) — CLOSED, opened ≥ since, operator-closed out. DL 356: the gate was registered under the
    OLD WILLY exit (+1.25 fixed) → V2 runner / floor fills (either sleeve) are kept OUT of both sides (v2_out lists them)."""
    c = a[(a.status.astype(str) == "CLOSED") & (a.o >= since)]
    ht, ox = _GK.split_operator_closed(c[c.es == "FRENZY_WILLY_HT"]) if len(c) else (c, c)
    w, _ = _GK.split_operator_closed(c[c.es == "FRENZY_WILLY"]) if len(c) else (c, c)
    ht, _ = WEra.split(ht)
    w, _ = WEra.split(w)
    return ht.sort_values("o"), w.sort_values("o"), ox


def v2_out(a, since):
    """DL 356: the closed, non-operator-closed WILLY / HT fills opened ≥ since → (all of them, the V2 runner / floor ones kept out)."""
    c = a[(a.status.astype(str) == "CLOSED") & (a.o >= since) & a.es.isin(("FRENZY_WILLY", "FRENZY_WILLY_HT"))] if len(a) else a
    k, _ = _GK.split_operator_closed(c) if len(c) else (c, c)
    return k, WEra.split(k)[1] if len(k) else k


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
    allw, vx = v2_out(a, since)
    L.append(WEra.era_line(allw, "WILLY_HT"))                                 # DL 356 (c)
    if len(vx):
        L.append(f"V2 runner / floor fills kept OUT of both sides (DECISION_LOG 356 — the gate was registered on the old +{WEra.OLD_TP:g} exit): "
                 + " · ".join(f"{r.o:%m-%d %H:%M} {r.pair} {r.es} {r.pct:+.2f} %" for r in vx.itertuples()))
    for name, g in (("FRENZY_WILLY_HT (¼)", ht), ("FRENZY_WILLY (full)", w)):
        L.append(f"- {name}: " + (f"N {len(g)} · {g.o.dt.date.nunique()} days · WR {(g.pct > 0).mean():.0%} · avg {g.pct.mean():+.3f} % · "
                                  f"Σ {g.pct.sum():+.2f} pts · ${g.usd.sum():+.2f}" if len(g) else "N 0"))
    v, txt = frenzy_willy_ht_gate([(r.o.to_pydatetime(), r.pct) for r in ht.itertuples()], [(r.o.to_pydatetime(), r.pct) for r in w.itertuples()])
    L.append(f"Gate verdict: {v} — {txt}")
    L += resize_lines(a, _GK.dl_deploy_ms(DL_V2B, lag_min=10, label="WILLY_HT ladder"), _GK.dl_deploy_ms(DL_STAMP, lag_min=0, label="WILLY_HT size stamp"))
    return L + [""]


def main():
    print("\n".join(run(int(time.time() * 1000))))


if __name__ == "__main__":
    main()
