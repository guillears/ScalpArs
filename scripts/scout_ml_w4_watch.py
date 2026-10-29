#!/usr/bin/env python3
"""👁 Scout — ML_W4 watch line (pre-registered 2026-10-29, DECISION_LOG 410; OBSERVE ONLY — never changes config, no bot API, no network).

QUESTION   Do W4-matched momentum longs lose? The normal route blocks every pattern-matched long (long_unmatched_only); W4 longs still
           reach the book through the routes exempt from that block — almost all through NONEXP_CALM3D (19 of the 25 W4 fills at
           registration), the rest through *_PROBE fires. If the forward W4 fills clear the locked EXPECTANCY bar, the candidate rule is
           "the NONEXP_CALM3D route obeys the W4 block too" (operator decides; never automatic).
EVIDENCE   (in-sample, NEVER counted) full-size momentum longs, master on today's stack + B25 (2026-10-29 sweep, probes out): W4 19 ·
           14 d · 47 % · −0.068 %/fill (all through NONEXP_CALM3D) vs the rest of NONEXP 15 · 87 % · +0.30 and the normal route 143 · 72 %
           · +0.114. Separator sweep (sweep_separators MOMENTUM:LONG, full-size, ENTRY label): W4 the strongest single split — WORKED
           rate −29 pts in BOTH eras — but family-wise p 0.060 vs the 0.017 bar → not above luck → observe first (CLAUDE.md OBSERVE-FIRST).
COHORT     CLOSED MOMENTUM LONG fills from the orders exports (export_glob: Downloads + exports_archive/), dedup (opened_at[:19], pair,
           direction) — CLOSED beats OPEN; FULL-SIZE only (cell_multiplier_source not *_PROBE); operator-closed out (scout_gatekit).
           COUNTED: opened ≥ the DECISION_LOG 410 commit + 10 min (forward only). zone = entry_pattern_w4_match True · rest = False ·
           missing stamp → UNSCORED (listed, never counted). P&L = the booked pnl_percentage (% is leverage-invariant; momentum longs 1×).
FROZEN BAR (CLAUDE.md expectancy bar, Sep-25; never re-fit): at the first prefix (by open time) where the ZONE reaches N ≥ 15 on ≥ 8
           distinct UTC days, decided ONCE —
           ① zone avg ≥ rest avg, OR zone avg ≥ 0 (a cohort that does not lose is never a filter) → RETIRED;
           ② else FILTER CANDIDATE (operator decides; never automatic) iff zone WR < the frozen breakeven WR 62.0 % (full-size momentum
              longs at registration: avg win +0.487 / avg loss −0.796, N 177) ∧ zone mean < 0 at day-clustered bootstrap P(mean < 0) ≥ 0.95
              (4,000 resamples, seed 7) ∧ no single day and no single pair carries ≥ 50 % of the zone's gross loss;
           ③ else KEEP OBSERVING.
           Pre-registration choices beyond the CLAUDE.md expectancy bar: the relative retire (zone ≥ rest) and the absolute one
           (zone ≥ 0). The prefix = every scored fill up to AND including the crossing zone fill (sorted by open time, then pair). An empty rest
           group skips ① (the zone is judged on ② alone).
           Deferred until the newest orders export was downloaded ≥ the prefix end + the momentum time cap (max_holding_time_minutes,
           1200 min) — the exports carry CLOSED rows only (main.py export), so a long opened before the crossing fill and still open
           at download time would otherwise be missing from the frozen prefix; an OPEN row in an export (never today) also holds. A candidate ships only after the 30–50 % in-sample
           haircut and with a pre-committed revert (operator), per CLAUDE.md.
STATE      reports/SCOUT_ML_W4_WATCH.json (gitignored).
Usage: venv/bin/python -I scripts/scout_ml_w4_watch.py
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
import scout_b1h_negflank as NF    # noqa: E402  (day-bootstrap, gross-loss share, freeze hold, state I/O)

DL = 410
N_MIN, DAYS_MIN = 15, 8
BE_WR = 0.620                      # FROZEN 2026-10-29: full-size momentum-long breakeven WR (avg win +0.487 / loss −0.796, N 177)
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
ZONE_COL = "entry_pattern_w4_match"
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_ML_W4_WATCH.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason",
        "cell_multiplier_source", ZONE_COL)
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
REGISTRATION = ("in-sample at registration (NEVER counted): W4 19 · 14 d · 47 % · −0.068 % (all NONEXP_CALM3D) vs the rest of NONEXP "
                "15 · 87 % · +0.30 and the normal route 143 · 72 % · +0.114; sweep W4 WORKED −29 pts both eras, family-wise p 0.060")
_TRUE = ("true", "1", "1.0", "yes")


def load(paths=None):
    """→ (closed full-size momentum longs, [(opened_at, 'OPEN <pair>')] of open momentum longs, n probes dropped)."""
    fr = []
    files = list(_EG.glob_exports(EXPORT_GLOB) if paths is None else paths)
    for f in files:
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f), _f=f))
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "day", "zone"]), [], 0
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m", "_f"])
    # strict "MOMENTUM": the counted cohort is forward-only (DL 410+), and every bot fill since Sep stamps entry_strategy
    d = d[(d.entry_strategy.astype(str) == "MOMENTUM") & (d.direction.astype(str) == "LONG")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    probe = d.cell_multiplier_source.astype(str).str.endswith("_PROBE")
    n_probe = int((probe & d._c).sum())
    d = d[~probe]
    # review: an OPEN row holds the freeze only if the NEWEST export of its mode still shows it (a position wiped by a reset never
    # gets a CLOSED row — it must not defer the freeze forever)
    newest = set(_EG.newest_per_mode([f for f in files if os.path.exists(f)]))
    op = (d.status.astype(str).str.upper() == "OPEN") & d._f.isin(newest)
    opens = [(t, f"OPEN {p}") for t, p in zip(d.o[op], d.pair[op])]
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    z = c[ZONE_COL].astype(str).str.strip().str.lower()
    c["zone"] = np.where(c[ZONE_COL].isna() | z.isin(("", "nan", "none")), "unscored", np.where(z.isin(_TRUE), "zone", "rest"))
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), opens, n_probe


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
    if wr < BE_WR and mz < 0 and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def newest_export_ts(files):
    """the newest orders export's download instant (filename stamp, else mtime; naive UTC) or None."""
    best = None
    for f in files or ():
        st = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(st) if st else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        best = t if best is None or t > best else best
    return best


def cap_minutes(path=None):
    """the momentum time cap (trading_config.json max_holding_time_minutes; unreadable → 1200)."""
    import json
    try:
        with open(path or os.path.join(_ROOT, "trading_config.json")) as f:
            c = json.load(f)
        for blk in (c, c.get("investment", {}), c.get("thresholds", {})):
            if isinstance(blk, dict) and "max_holding_time_minutes" in blk:
                return float(blk["max_holding_time_minutes"])
    except Exception:
        pass
    return 1200.0


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


def freeze(st, sc, now_iso, hold=None, why=None):
    """ONCE at the first crossing; never recomputed. → (st, changed)."""
    if "first" in st:
        return st, False
    pre = crossing_prefix(sc)
    if pre is None:
        return st, False
    last = pre.o.max()
    if NF.freeze_hold(last, hold, why):
        return st, False
    state, det = verdict(pre[pre.zone == "zone"], pre[pre.zone == "rest"])
    st["first"] = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso,
                       keys=[f"{a:%Y-%m-%dT%H:%M:%S}|{b}|{g}" for a, b, g in zip(pre.o, pre.pair, pre.zone)])
    return st, True


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % | "
            f"{pd.to_numeric(g.pnl, errors='coerce').sum():+,.0f} |")


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, open_ts=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 410 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 ML_W4 — W4-matched momentum longs vs the rest (pre-registered 2026-10-29, DECISION_LOG {DL}, OBSERVE only)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="ML_W4 watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    files = list(_EG.glob_exports(EXPORT_GLOB) if paths is None else paths)
    c, opens, n_probe = load(files)
    if open_ts is not None:
        opens = list(open_ts)
    newest, cap = newest_export_ts(files), cap_minutes()
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    sc = keep[keep.zone != "unscored"]
    zone, rest, uns = sc[sc.zone == "zone"], sc[sc.zone == "rest"], keep[keep.zone == "unscored"]
    L += [f"Cohort: CLOSED full-size MOMENTUM LONG fills opened ≥ {since:%Y-%m-%d %H:%M} UTC (DL {DL} commit + 10 min) · zone = "
          f"{ZONE_COL} True (W4 longs reach the book only through routes exempt from long_unmatched_only — NONEXP_CALM3D) · "
          f"{_GK.operator_closed_line(ox)} · probe fires out (all exports: {n_probe}).", "",
          "| Group | N | days | WR | avg % | Σ$ as traded |", "|---|---|---|---|---|---|",
          _row("**zone (W4-matched)**", zone), _row("rest (not W4)", rest), _row("UNSCORED (no W4 stamp) — never counted", uns), ""]
    why = []
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    if ok:
        hold = [(t, lab) for t, lab in opens if pd.notna(t) and t >= since]
        pre = crossing_prefix(sc)
        if pre is not None and (newest is None or newest < pre.o.max() + pd.Timedelta(minutes=cap)):
            # a long opened ≤ the crossing fill can still be open at download time (exports hold CLOSED rows only) → wait
            hold.append((pre.o.max(), f"newest export {('none' if newest is None else f'{newest:%m-%d %H:%M}')} < crossing + "
                                      f"{cap:.0f}-min momentum cap — wait for a later export"))
        st, ch = freeze(st, sc, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), hold, why)
        if ch and state_path:
            NF.du_save_state(st, state_path)
    live_state, live_det = verdict(zone, rest)
    if ok and "first" in st and state_path:
        f0 = st["first"]
        L.append(f"**Frozen verdict (crossing at fill {f0['at']}, frozen {f0['run_at']}): {f0['state']}** ({f0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    else:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += [f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}): at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} UTC days, decided ONCE — "
          f"zone avg ≥ rest avg → RETIRED; {CANDIDATE} iff zone WR < {BE_WR:.1%} ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ {P_MIN:.2f} "
          f"({BOOT_N:,}, seed {BOOT_SEED}) ∧ no day / pair ≥ {SHARE_MAX:.0%} of the zone's gross loss; else KEEP OBSERVING. Candidate rule: "
          "the NONEXP_CALM3D route obeys the W4 block; haircut + pre-committed revert before any ship.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
