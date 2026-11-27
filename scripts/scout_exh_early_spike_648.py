#!/usr/bin/env python3
"""⏱ Scout — EXH_EARLY_SPIKE watch line (registered 2026-11-26, DECISION_LOG 648; OBSERVE ONLY — never changes config, no bot API, no network).

WHY        operator 11-26 (INIT −5.38 %, stopped 10 min after an entry 0.25 h after the spike): "register the EXH < 0.5 h after spike observe
           line". Study reports/EXH_SPIKE_AGE_2026-11-26.md (3,358 EXH fills 2023-07 → 2026-11 at TODAY's exit TP 7, today's filters):
             0–0.5 h after the spike: 216 · WR 38 % · −0.23 %/trade, 61 % stopped / 32 % TP (≈ 50 / 47 in every later bucket); DESIGN −0.46 ·
             HOLDOUT −0.04; negative in 5 / 7 half-years; difference vs ≥ 0.5 h 95 % CI −1.72 … −0.20 — but the 0.5 h cut was picked AFTER
             seeing the buckets (in-sample), the shape is non-monotonic (0.5–1 h is one of the best buckets, +0.90) and the zone fails the
             Pattern-C bar on confidence (day-bootstrap upper bound +0.27). Blocking < 0.5 h: DCR 0.821 → 0.856 %/day (marginal); blocking
             < 1 h LOWERS it (0.715). Batch 11-18 → 11-26: 1 fill (INIT, in-sample). Theory: an RSI 70 → 70 turn inside 30 min of the spike is
             often a pause before the pump's second leg (late buyers + fresh shorts → squeeze).
LINE       zone = entry_frenzy_hours < 0.5 (the bot's own stamp: hours from the Frenzy spike bar to the entry; FROZEN at 0.5, never refit) ·
           rest = every other scored fill · NULL stamp → UNSCORED, never counted.
           (review) the stamp is hours from the spike bar to the SIGNAL bar's close (the fill follows ≈ 8 s later) and the engine rounds it to
           2 decimals → a true 0.495–0.4999 h entry is stored as 0.50 and lands in rest (zone = stored stamp < 0.50, i.e. ≲ 0.495 h); the
           backtest bucketed the unrounded hours — a negligible edge effect, the frozen cut is not moved.
COHORT     CLOSED EXHAUSTION_SHORT fills, FRENZY universe (scout_exhaustion_short.load_fills: dedup, CLOSED wins, operator-closed out),
           opened ≥ the DECISION_LOG 648 commit + 10 min (forward only — INIT 11-26, the motivating fill, is in-sample, never counted).
           Booked pnl_percentage.
FROZEN BAR read ONCE at the first prefix where the ZONE reaches N ≥ 15 on ≥ 8 UTC days; if that read is KEEP OBSERVING, ONE more read at
           N ≥ 30 on ≥ 10 days (same bar), then the line stays informational:
           zone avg ≥ 0 or ≥ the rest avg → RETIRED · FILTER CANDIDATE (operator decides; never automatic) iff zone WR < 43.3 % (the sleeve's
           breakeven at TP 7 in the DL 648 backtest, FROZEN) ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ 0.95 ∧ zone avg < rest avg ∧ no
           day / pair ≥ 50 % of the zone's gross loss · else KEEP OBSERVING. Day units (Frenzy spikes cluster by day).
           Booked P&L → pinned to the EXH exit keys in observe_registry (an exit change → ⚠ STALE the same day).
           ⚠ Rare zone: the backtest has ≈ 216 zone fills in 3.4 years (≈ 1 / 6 days) — the first read may take months.
STATE      reports/SCOUT_EXH_EARLY_SPIKE_648.json (gitignored).
Usage: venv/bin/python -I scripts/scout_exh_early_spike_648.py"""
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

DL = 648
LINE = "EXH_EARLY_SPIKE"
ZONE_MAX = 0.5                      # FROZEN: zone iff entry_frenzy_hours < 0.5 (strict; picked post-hoc in DL 648 → never refit)
BE_WR = 0.4328                      # FROZEN breakeven at TP 7 (reports/EXH_SPIKE_AGE_2026-11-26.md, kept fills, BE 43.3 %)
READS = ((15, 8), (30, 10))         # (zone N, distinct UTC days) — first read; a second only if the first is KEEP OBSERVING
P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED = 0.95, 0.50, 4000, 648
HOLD_MIN = 24 * 60                  # freeze once a LATER EXH fill opened ≥ the crossing fill + the 24 h cap (every counted fill has closed)
COL = "entry_frenzy_hours"
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_EXH_EARLY_SPIKE_648.json")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def zone_of(hours):
    """< 0.5 h → zone · other finite ≥ 0 → rest · missing / negative → unscored. Pure."""
    v = _f(hours)
    if v is None or v < 0:
        return "unscored"
    return "zone" if v < ZONE_MAX else "rest"


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
    """→ markdown lines. since_ms None → the DECISION_LOG 648 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## ⏱ {LINE} — EXHAUSTION shorts entered < 0.5 h after the Frenzy spike (registered 2026-11-26, DECISION_LOG {DL}, OBSERVE only)", "🔒 ARMED as an entry block since DECISION_LOG 650 (2026-11-27, declared override): new zone fills ≈ 0 — the verdict now lives in the shadow revert gate (scripts/scout_exh_regime_650.py); this line keeps showing the fills the block let through / pre-arm.", ""]
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
          _row("zone (would block: < 0.5 h after the spike)", zone), _row("rest", rest), _row("UNSCORED (no stamp) — never counted", uns), ""]
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
    L += [f"FROZEN: zone {COL} < {ZONE_MAX:g} · read once at zone N ≥ {READS[0][0]} on ≥ {READS[0][1]} days, a second read at "
          f"{READS[1][0]} on ≥ {READS[1][1]} days only after KEEP OBSERVING (RETIRED / {CANDIDATE} / KEEP OBSERVING) · breakeven {BE_WR:.1%}. "
          "Backtest prior (never counted): < 0.5 h 216 · WR 38 % · −0.23 %/trade (DESIGN −0.46 · HOLDOUT −0.04), 61 % stopped; 0.5–1 h +0.90 "
          "(non-monotonic; cut post-hoc) (reports/EXH_SPIKE_AGE_2026-11-26.md) · in-sample batch fill INIT 11-26.", ""]
    return L


def main():
    print("\n".join(run()))


if __name__ == "__main__":
    main()
