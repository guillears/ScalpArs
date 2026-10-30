#!/usr/bin/env python3
"""👁 Scout — RETRY_BTC1H watch line (pre-registered 2026-10-30, DECISION_LOG 415; OBSERVE ONLY — never changes config, no bot API, no network).

QUESTION   Does FRENZY_LITE_RETRY lose when BTC's 1 h trend is falling? LITE itself does the opposite, so RETRY gets its own read. Theory: a retry re-enters after
           a failed first attempt; with BTC falling the second attempt has no market behind it.
EVIDENCE   (in-sample, NEVER counted) master on today's stack 2 + B25 16: zone 12 · 42 % · −0.92 %/fill vs 6 · 100 % · +2.51 (day-block p ≈ 0.04). B25 16 · 56 % · 0.00 → 6 · 100 % · +2.51,
           daily compound −0.06 → +2.11 (each trade = account), real account −0.13 → +1.68. AGAINST: seen after the fact, 18 fills, nearly all one batch; the backtest has no separate RETRY → observe only. CAVEAT: all 12 zone fills ran TP +2, 3 of the 6 rest fills TP +3 (LITE C), so "zone ≥ rest" starts with a mechanical edge for rest.
COHORT     CLOSED FRENZY_LITE_RETRY LONG fills, full size (*_PROBE out), from the orders exports (export_glob: Downloads + exports_archive/), dedup (opened_at[:19], pair) —
           CLOSED beats OPEN, newest export wins; operator-closed out (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 415 commit + 10 min (forward only). zone =
           entry_btc_1h_slope ≤ 0 · rest = > 0 · stamp missing → UNSCORED (listed, never counted). P&L = the booked pnl_percentage.
FROZEN BAR (CLAUDE.md expectancy bar, never re-fit): at the first prefix (by open time) where the ZONE reaches N ≥ 15 on ≥ 8 distinct UTC days, decided ONCE —
           ① zone avg ≥ rest avg, OR zone avg ≥ 0 → RETIRED; ② else FILTER CANDIDATE (operator decides; never automatic) iff zone WR < the frozen breakeven WR 60.0 % (the zone is always TP +2 / −3 by
           LITE C construction) ∧ zone mean < 0 at a day-clustered bootstrap P(mean < 0) ≥ 0.95 (4,000 resamples, seed 7) ∧ no single day and no single pair carries
           ≥ 50 % of the zone's gross loss; ③ else KEEP OBSERVING. Deferred until the newest orders export was downloaded ≥ the prefix end + 780 min (exports carry CLOSED rows only; RETRY's
           12 h hold + an hour). observe_registry pins the LITE exit keys (an exit change → ⚠ STALE). A candidate ships only after the 30–50 % in-sample haircut, a before → after per batch
           with the daily compound return and a pre-committed revert.
STATE      reports/SCOUT_RETRY_BTC1H.json (gitignored).
Usage: venv/bin/python -I scripts/scout_retry_btc1h_watch.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK        # noqa: E402  (operator-closed fills out · the DL commit time)
import scout_b1h_negflank as NF    # noqa: E402  (day-bootstrap, gross-loss share, state I/O)

DL = 415
SLEEVE = "FRENZY_LITE_RETRY"
SLOPE_CUT = 0.0
N_MIN, DAYS_MIN = 15, 8
BE_WR = 0.600                      # FROZEN 2026-10-30: the zone is always TP +2 / −3 (LITE C never applies at slope ≤ 0) → 3 / (2 + 3)
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
WAIT_MIN = 780
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_RETRY_BTC1H.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
STAMPS = ("entry_btc_1h_slope",)
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason",
        "cell_multiplier_source") + STAMPS
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
REGISTRATION = "in-sample at registration (NEVER counted): zone 12 · 42 % · −0.92 vs 6 · 100 % · +2.51 (master 2 + B25 16), day-block p ≈ 0.04; no separate RETRY in the backtest"


def load(paths=None):
    """→ (closed full-size RETRY fills — dedup, CLOSED beats OPEN, newest export wins —, the newest read export's download time or None)."""
    fr, read = [], []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
            read.append(f)
    newest = None
    for f in read:
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "day", "zone", "pct"]), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[(d.entry_strategy.astype(str) == SLEEVE) & (d.direction.astype(str) == "LONG")
          & ~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    sl = pd.to_numeric(c[STAMPS[0]], errors="coerce")
    c["zone"] = np.where(sl.isna(), "unscored", np.where(sl <= SLOPE_CUT, "zone", "rest"))
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


def verdict(zone, rest):
    """zone / rest = counted fills (pct, day, pair) → (state, detail). PURE."""
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
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
    if wr < BE_WR and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def crossing_prefix(sc):
    """scored counted fills by open time → the prefix ending at the zone fill where the zone first reaches N ≥ 15 on ≥ 8 days."""
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(sc.zone.values, sc.day.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(d)
        if nz >= N_MIN and len(seen) >= DAYS_MIN:
            return sc.iloc[:k]
    return None


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % | "
            f"{pd.to_numeric(g.pnl, errors='coerce').sum():+,.0f} |")


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 415 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 RETRY_BTC1H — FRENZY_LITE_RETRY with BTC 1 h slope ≤ {SLOPE_CUT:g} "
         f"(pre-registered 2026-10-30, DECISION_LOG {DL}, OBSERVE only)", ""]
    since_given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="RETRY_BTC1H watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    sc = keep[keep.zone != "unscored"] if len(keep) else keep
    zone, rest = sc[sc.zone == "zone"], sc[sc.zone == "rest"]
    uns = keep[keep.zone == "unscored"] if len(keep) else keep
    L += [f"Cohort: CLOSED full-size RETRY fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if since_given is None else 'floor given'}) · "
          f"{_GK.operator_closed_line(ox)}.", "", "| Group | N | days | WR | avg % | Σ$ as traded |", "|---|---|---|---|---|---|",
          _row("**zone (BTC 1 h slope ≤ 0)**", zone), _row("rest", rest), _row("UNSCORED (no slope stamp) — never counted", uns), ""]
    why = []
    rec = st.get(SLEEVE) if ok else None
    if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
        rec, ok = None, False          # a damaged record never re-freezes silently
    if not rec and ok:
        pre = crossing_prefix(sc) if len(sc) else None
        if pre is not None:
            last = pre.o.max()
            if newest is None or newest < last + pd.Timedelta(minutes=WAIT_MIN):
                why.append(f"newest export < the crossing fill + {WAIT_MIN} min (a fill could still be open)")
            else:
                state, det = verdict(pre[pre.zone == "zone"], pre[pre.zone == "rest"])
                rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC",
                           run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                           keys=[f"{a:%Y-%m-%dT%H:%M:%S}|{b}|{g}" for a, b, g in zip(pre.o, pre.pair, pre.zone)])
                if state_path:
                    st[SLEEVE] = rec
                    NF.du_save_state(st, state_path)
    live_state, live_det = verdict(zone, rest)
    if rec and ok and state_path:
        L.append(f"**Frozen verdict (crossing at fill {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    else:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    if why:
        L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += ["", f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}): at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} UTC days, decided ONCE — zone avg ≥ rest "
          f"avg or ≥ 0 → RETIRED; {CANDIDATE} iff zone WR < {BE_WR:.1%} ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ {P_MIN:.2f} "
          f"({BOOT_N:,}, seed {BOOT_SEED}) ∧ no day / pair ≥ {SHARE_MAX:.0%} of the zone's gross loss; else KEEP OBSERVING. "
          "Haircut, before → after per batch with the daily compound return and a pre-committed revert before any ship.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
