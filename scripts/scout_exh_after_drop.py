#!/usr/bin/env python3
"""📉 Scout — EXH_AFTER_DROP watch line (registered 2026-11-19, DECISION_LOG 596; OBSERVE ONLY — never changes config, no bot API, no network).

WHY        operator 11-19 (after the YB short): "what do we have in exhaustion … shorting after the drop already happened?". The 11-18 entry
           study (reports/EXH_BADDAY_2026-11-18.md, the live exit) cut the coin's move over the hour before the short:
             2023-07 → 2025-06   −5 … −10 %:  61 · 34 % · −0.48   (all other buckets but one tiny one positive)
             2025-07 → 2026-11   −5 … −10 %: 144 · 35 % · −0.71
           negative in BOTH periods while most shorts win — but NON-MONOTONIC (−10 … −15 % wins: 7 · +5.00 / 32 · +2.12), a hole with winning
           flanks = a confound by our rules → NOT a filter. Watch it on real fills; never re-fit the band.
LINE       zone = entry_exh_ret_1h_pct in [−10, −5] — the signal 15m close vs the 5m close 1 h earlier (services.exhaustion.features
           ret_1h, stamped at the open since DL 596) · rest = every other scored fill · NULL stamp → UNSCORED, never counted.
COHORT     CLOSED EXHAUSTION_SHORT fills, FRENZY universe (scout_exhaustion_short.load_fills: dedup, CLOSED wins, operator-closed out; WIDE
           fills are a different universe and are not read here), opened ≥ the DECISION_LOG 596 commit + 10 min (forward only — YB 11-19,
           the motivating fill, is in-sample and never counted). Booked pnl_percentage.
FROZEN BAR at the first prefix where the ZONE reaches N ≥ 15 on ≥ 8 UTC days, decided ONCE (the shared watch-line bar, DL 414 / 509):
           zone avg ≥ 0 or ≥ the rest avg → RETIRED · FILTER CANDIDATE (operator decides; never automatic) iff zone WR < 35.0 % (the sleeve's
           breakeven, FROZEN from the study's DESIGN set: avg win +8.39 / avg loss −4.51) ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ 0.95
           ∧ no day / pair ≥ 50 % of the zone's gross loss · else KEEP OBSERVING. Booked P&L → pinned to the EXH exit keys in observe_registry.
STATE      reports/SCOUT_EXH_AFTER_DROP.json (gitignored).
Usage: venv/bin/python -I scripts/scout_exh_after_drop.py"""
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

DL = 596
LINE = "EXH_AFTER_DROP"
ZONE_LO, ZONE_HI = -10.0, -5.0      # FROZEN band (inclusive)
BE_WR = 0.350                       # FROZEN breakeven (EXH_BADDAY DESIGN: 8.39 / 4.51 → 4.51 ÷ 12.90)
N_MIN, DAYS_MIN, P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED = 15, 8, 0.95, 0.50, 4000, 596
HOLD_MIN = 24 * 60                  # the verdict freezes once a LATER EXH fill opened ≥ the crossing fill + the 24 h cap (conservative: every counted fill has closed)
COL = "entry_exh_ret_1h_pct"
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_EXH_AFTER_DROP.json")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def zone_of(ret_1h):
    """[−10, −5] → zone · other finite → rest · missing → unscored. Pure."""
    v = _f(ret_1h)
    if v is None:
        return "unscored"
    return "zone" if ZONE_LO <= v <= ZONE_HI else "rest"


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


def verdict(zone, rest):
    """the frozen bar. PURE."""
    n, nd = len(zone), (zone.day.nunique() if len(zone) else 0)
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz, mr = float(zone.pct.mean()), (float(rest.pct.mean()) if len(rest) else float("nan"))
    wr = float((zone.pct > 0).mean())
    p = NF.p_mean_neg(zone.pct.values, zone.day.values, BOOT_N, BOOT_SEED)
    sd, sp = NF.gross_loss_share(zone.pct.values, zone.day.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {wr:.0%} (breakeven {BE_WR:.1%}) · avg {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · "
           f"P(mean<0) {(p if p is not None else float('nan')):.2f} · top day {sd:.0%} / top pair {sp:.0%} of the zone's gross loss")
    if mz >= 0 or (len(rest) and mz >= mr):
        return "RETIRED", det
    if wr < BE_WR and mz < 0 and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def crossing_prefix(sc):
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(sc.z.values, sc.day.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(d)
        if nz >= N_MIN and len(seen) >= DAYS_MIN:
            return sc.iloc[:k]
    return None


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – |"
    return f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % |"


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 596 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 📉 {LINE} — EXHAUSTION shorts after a −5 … −10 % hour (registered 2026-11-19, DECISION_LOG {DL}, OBSERVE only)", ""]
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
          _row("zone (would block: −10 … −5 % hour)", zone), _row("rest", rest), _row("UNSCORED (no stamp) — never counted", uns), ""]
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    rec = st.get(LINE) if ok else None
    if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
        rec, ok = None, False
    if not rec and ok and len(sc):
        pre = crossing_prefix(sc)
        if pre is not None:
            last = pre.o.max()
            if newest is not None and newest >= last + pd.Timedelta(minutes=HOLD_MIN):
                state, det = verdict(pre[pre.z == "zone"], pre[pre.z == "rest"])
                rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC",
                           run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
                if state_path:
                    st[LINE] = rec
                    NF.du_save_state(st, state_path)
    live_state, live_det = verdict(zone, rest)
    if rec and ok and state_path:
        L += [f"Frozen verdict (crossing {rec['at']}, frozen {rec['run_at']}): **{rec['state']}** ({rec['detail']})",
              f"Live (information only): {live_state} — {live_det}", ""]
    else:
        L += [f"Now: ⏳ {live_state} ({live_det}).", ""]
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += [f"FROZEN: zone {ZONE_LO:g} ≤ {COL} ≤ {ZONE_HI:g} · bar zone N ≥ {N_MIN} on ≥ {DAYS_MIN} days, decided once (RETIRED / {CANDIDATE} / "
          f"KEEP OBSERVING) · breakeven {BE_WR:.1%}. Non-monotonic in the study (−10 … −15 % wins) — expected to need the full bar before any "
          "action. In-sample evidence (never counted): reports/EXH_BADDAY_2026-11-18.md · the YB 11-19 short.", ""]
    return L


def main():
    print("\n".join(run()))


if __name__ == "__main__":
    main()
