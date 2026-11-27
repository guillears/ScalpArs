#!/usr/bin/env python3
"""🌡 Scout — EXH_BEAR80 watch line (registered 2026-11-25, DECISION_LOG 644; OBSERVE ONLY — never changes config, no bot API, no network).

WHY        operator 11-25 (the two BIGTIME + YB losses, 14-fill batch W/L): "why don't we add the watchline of possible block of 80%+ breath for
           EXH?". Study reports/EXH_BREADTH_2026-11-25.md (DL 643, 2,793 EXH fills 2024-06 → 2026-11 at TODAY's exit TP 7):
             bearish % > 80:  337 · WR 45 % · −0.02 %/trade (CI −0.60 … +0.64) vs the rest +0.75 — breakeven overall, but by half-year
             2024-H2 −0.39 · 2025-H1 −0.51 · 2025-H2 −0.16 · 2026-H1 +0.09 · 2026-H2 +1.43 (two eras). Batch 11-18 → 11-25: 3 · 1 W · −3.04.
           Not a filter (fails the Pattern-C bar; the 65–80 / > 80 split was a bucket read, post-hoc). The forward line answers which era
           the live tape is in.
LINE       zone = entry_bear_pct > 80 (the bot's own breadth stamp: share of the top-50 perps whose 5m EMA20 fell > 0.02 % over 3 bars,
           read at the open) · rest = every other scored fill · NULL stamp → UNSCORED, never counted.
COHORT     CLOSED EXHAUSTION_SHORT fills, FRENZY universe (scout_exhaustion_short.load_fills: dedup, CLOSED wins, operator-closed out; WIDE
           fills are a different universe — the study read FRENZY rows only), opened ≥ the DECISION_LOG 644 commit + 10 min (forward only —
           YB 11-19 · ENSO 11-21 · BIGTIME 11-25, the motivating fills, are in-sample and never counted). Booked pnl_percentage.
FROZEN BAR read ONCE at the first prefix where the ZONE reaches N ≥ 15 on ≥ 8 UTC days; if that read is KEEP OBSERVING, ONE more read at
           N ≥ 30 on ≥ 10 days (same bar), then the line stays informational:
           zone avg ≥ 0 or ≥ the rest avg → RETIRED · FILTER CANDIDATE (operator decides; never automatic) iff zone WR < 44.1 % (the sleeve's
           breakeven at TP 7 in the DL 643 backtest, FROZEN) ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ 0.95 ∧ zone avg < rest avg ∧ no
           day / pair ≥ 50 % of the zone's gross loss · else KEEP OBSERVING. Market-wide variable → DAY units (window-units rule).
           Booked P&L → pinned to the EXH exit keys in observe_registry (an exit change → ⚠ STALE the same day).
STATE      reports/SCOUT_EXH_BEAR80_644.json (gitignored).
Usage: venv/bin/python -I scripts/scout_exh_bear80_644.py"""
import math
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import scout_gatekit as _GK                     # noqa: E402
import scout_b1h_negflank as NF                 # noqa: E402
import scout_exhaustion_short as XS             # noqa: E402  (the EXH fill loader — one definition)

DL = 644
LINE = "EXH_BEAR80"
ZONE_MIN = 80.0                     # FROZEN: zone iff entry_bear_pct > 80 (strict)
BE_WR = 0.441                       # FROZEN breakeven at TP 7 (reports/EXH_BREADTH_2026-11-25.csv, ALL, BE 44.06 %)
READS = ((15, 8), (30, 10))         # (zone N, distinct UTC days) — first read; a second only if the first is KEEP OBSERVING
P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED = 0.95, 0.50, 4000, 644
HOLD_MIN = 24 * 60                  # freeze once a LATER EXH fill opened ≥ the crossing fill + the 24 h cap (every counted fill has closed)
COL = "entry_bear_pct"
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_EXH_BEAR80_644.json")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def zone_of(bear_pct):
    """> 80 → zone · other finite → rest · missing → unscored. Pure."""
    v = _f(bear_pct)
    if v is None:
        return "unscored"
    return "zone" if v > ZONE_MIN else "rest"


def load(paths=None):
    a = XS.load_fills(paths, extra=(COL,))
    if not len(a):
        return a.assign(pct=[], day=[], z=[])
    a = a.copy()
    a["pct"] = pd.to_numeric(a.pnl_percentage, errors="coerce")
    a = a[a.pct.notna()]
    a["day"] = a.o.dt.strftime("%Y-%m-%d")
    a["z"] = [zone_of(v) for v in a[COL]]
    return a.reset_index(drop=True)


def verdict(zone, rest, n_min=READS[0][0], days_min=READS[0][1]):
    """the frozen bar. PURE."""
    n, nd = len(zone), (zone.day.nunique() if len(zone) else 0)
    if n < n_min or nd < days_min:
        return "COLLECTING", f"zone N {n}/{n_min} · {nd}/{days_min} days"
    if not len(rest):                   # the bar compares zone vs rest — no rest fills, no judgement
        return "KEEP OBSERVING", f"zone N {n} · {nd} d · no rest fills to compare"
    mz, mr = float(zone.pct.mean()), float(rest.pct.mean())
    wr = float((zone.pct > 0).mean())
    p = NF.p_mean_neg(zone.pct.values, zone.day.values, BOOT_N, BOOT_SEED)
    sd, sp = NF.gross_loss_share(zone.pct.values, zone.day.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {wr:.0%} (breakeven {BE_WR:.1%}) · avg {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · "
           f"P(mean<0) {(p if p is not None else float('nan')):.2f} · top day {sd:.0%} / top pair {sp:.0%} of the zone's gross loss")
    if mz >= 0 or mz >= mr:
        return "RETIRED", det
    if wr < BE_WR and mz < 0 and mz < mr and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def crossing_prefix(sc, n_min, days_min):
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(sc.z.values, sc.day.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(d)
        if nz >= n_min and len(seen) >= days_min:
            return sc.iloc[:k]
    return None


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – |"
    return f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % |"


def _key(i):
    return LINE if i == 0 else f"{LINE}@{READS[i][0]}"


def _valid(rec):
    return isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 644 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🌡 {LINE} — EXHAUSTION shorts with > 80 % of coins bearish (registered 2026-11-25, DECISION_LOG {DL}, OBSERVE only)", "🔒 ARMED as an entry block since DECISION_LOG 650 (2026-11-27, declared override): new zone fills ≈ 0 — the verdict now lives in the shadow revert gate (scripts/scout_exh_regime_650.py); this line keeps showing the fills the block let through / pre-arm.", ""]
    given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label=f"{LINE} (DL {DL})")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    a = load(paths)
    newest = a.o.max() if len(a) else None
    s = a[a.o >= since] if len(a) else a
    sc = s[s.z != "unscored"] if len(s) else s
    zone = sc[sc.z == "zone"] if len(sc) else sc
    rest = sc[sc.z == "rest"] if len(sc) else sc
    uns = s[s.z == "unscored"] if len(s) else s
    L += [f"Cohort: CLOSED EXHAUSTION_SHORT fills (FRENZY universe), opened ≥ {since:%Y-%m-%d %H:%M} UTC "
          f"({'DL ' + str(DL) + ' commit + 10 min' if given is None else 'floor given'}) · booked P&L %.", "",
          "| Group | N | days | WR | avg % |", "|---|---|---|---|---|",
          _row("zone (would block: > 80 % bearish)", zone), _row("rest", rest), _row("UNSCORED (no stamp) — never counted", uns), ""]
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    recs, bad = [], []
    for i, (n_min, d_min) in enumerate(READS):
        rec = st.get(_key(i)) if ok else None
        if rec is not None and not _valid(rec):
            bad.append(_key(i))
            rec = None
            break                                                    # never re-judge over a damaged record
        if rec is None and ok and len(sc) and (i == 0 or (recs and recs[-1] and recs[-1]["state"] == "KEEP OBSERVING")):
            pre = crossing_prefix(sc, n_min, d_min)
            if pre is not None:
                last = pre.o.max()
                if newest is not None and newest >= last + pd.Timedelta(minutes=HOLD_MIN):
                    state, det = verdict(pre[pre.z == "zone"], pre[pre.z == "rest"], n_min, d_min)
                    rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC",
                               run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
                    if state_path:
                        st[_key(i)] = rec
                        NF.du_save_state(st, state_path)
        recs.append(rec)
        if not rec:
            break
    frozen = [r for r in recs if r]
    # the live bar = the next read still pending (the 30-fill read only after a frozen KEEP OBSERVING), else the last frozen read's bar
    if not frozen:
        bar = READS[0]
    elif frozen[-1]["state"] == "KEEP OBSERVING" and len(frozen) < len(READS):
        bar = READS[len(frozen)]
    else:
        bar = READS[len(frozen) - 1]
    live_state, live_det = verdict(zone, rest, *bar)
    shown = False
    for i, rec in enumerate(recs):
        if rec and state_path:
            L.append(f"Frozen verdict @ {READS[i][0]} fills (crossing {rec['at']}, frozen {rec['run_at']}): **{rec['state']}** ({rec['detail']})")
            shown = True
    L += ([f"Live (information only): {live_state} — {live_det}", ""] if shown else [f"Now: ⏳ {live_state} ({live_det}).", ""])
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    if bad:
        L.append(f"⚠ frozen record(s) {', '.join(bad)} malformed in {os.path.basename(state_path)} — operator restore needed; no further read this run.")
    L += [f"FROZEN: zone {COL} > {ZONE_MIN:g} · read once at zone N ≥ {READS[0][0]} on ≥ {READS[0][1]} days, a second read at "
          f"{READS[1][0]} on ≥ {READS[1][1]} days only after KEEP OBSERVING (RETIRED / {CANDIDATE} / KEEP OBSERVING) · breakeven {BE_WR:.1%}. "
          "Backtest prior (never counted): > 80 % 337 · −0.02 %/trade, negative 2024-H2 → 2025-H2, positive 2026 (reports/EXH_BREADTH_2026-11-25.md) · "
          "in-sample batch fills YB 11-19 · ENSO 11-21 · BIGTIME 11-25.", ""]
    return L


def main():
    print("\n".join(run()))


if __name__ == "__main__":
    main()
