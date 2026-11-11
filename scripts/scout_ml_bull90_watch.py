#!/usr/bin/env python3
"""⛔ RETIRED 2026-11-11 (DECISION_LOG 530): ARMED (DL 527) — verdict in scout_blocked_filters. Unhooked from the opportunity scout; still runnable standalone.
👁 Scout — ML_BULL90 watch line (pre-registered 2026-11-11, DECISION_LOG 523, operator "yes build it and update best case"; OBSERVE ONLY — never
changes config, no bot API, no network).

QUESTION   Do MOMENTUM LONGs lose when the market is already stretched — ≥ 90 % of the scanned coins bullish at entry (entry_bull_pct ≥ 90)?
           Complementary to ML_MKT_HOT (DL 489: the rolling top-100 breadth zone) — a different breadth read; on 11-09 01:08–01:11 ML_MKT_HOT kept
           three longs (BCH / TAO / LTC, bull 93.5 %) bought at BTC's local top that all stopped within 17 min (the best case's worst day).
           Theory: when almost every coin is already up, a fresh momentum long buys the end of the move — no buyers left to carry it.
EVIDENCE   (in-sample, NEVER counted) 211 real momentum-long fills (master on today's stack 158 + B25 53): breadth ≥ 90 % 8 · 12 % · −0.63 on 5 days
           (master 5 · 20 % · −0.56 · B25 3 · 0 % · −0.75, the 11-09 burst) vs 85–90 % 15 · 73 % · +0.52 (the best bucket — a cliff at 90 is suspect);
           on top of ML_MKT_HOT it blocks 4 more fills (master SHIB 08-22 −0.88 · B25 the 11-09 three): master 87 · 75 % · +0.22 · Σ +18.91 · DCR
           +0.011 → 86 · 76 % · +0.23 · Σ +19.80 · +0.012 %/day · B25 32 · 72 % · +0.23 · Σ +7.21 · +0.089 → 29 · 79 % · +0.33 · Σ +9.45 · +0.117
           (one size). Found ON the day it fixes (operator call: complementary, saves the worst day, ≈ no master cost) → observe, never armed on this.
           Breadth is MARKET-WIDE (window-units rule): the bar counts days and bootstraps by day; one day / pair ≥ 50 % of the loss blocks a candidate.
COHORT     CLOSED MOMENTUM LONG fills, full size (*_PROBE out), from the orders exports (export_glob: Downloads + exports_archive/), dedup (opened_at[:19],
           pair) — CLOSED beats OPEN, newest export wins; operator-closed out (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 523 commit + 10 min
           (forward only). zone = entry_bull_pct ≥ 90 · rest = < 90 · stamp missing → UNSCORED (listed, never counted). P&L = the booked pnl_percentage.
           Every fill counts (ML_MKT_HOT-blocked ones too: it is observe-only, so the bot still trades them).
FROZEN BAR (CLAUDE.md expectancy bar, never re-fit): at the first prefix (by open time) where the ZONE reaches N ≥ 15 on ≥ 8 distinct UTC days, decided ONCE —
           ① zone avg ≥ rest avg, OR zone avg ≥ 0 → RETIRED; ② else FILTER CANDIDATE (operator decides; never automatic) iff zone WR < the frozen breakeven WR 61.1 %
           (momentum long, 211 fills at registration: avg win +0.505 / avg loss −0.793) ∧ zone mean < 0 at a day-clustered bootstrap P(mean < 0) ≥ 0.95
           (4,000 resamples, seed 7) ∧ no single day and no single pair carries ≥ 50 % of the zone's gross loss; ③ else KEEP OBSERVING. Deferred until the
           newest orders export was downloaded ≥ the prefix end + 1,200 min (exports carry CLOSED rows only). observe_registry pins the momentum-long exit
           keys (an exit change → ⚠ STALE). A candidate ships only after the 30–50 % in-sample haircut, a before → after per batch with the daily compound
           return and a pre-committed revert (the first 15 blocked signals re-priced with the live exit average better than the momentum longs kept over
           the same period → switch it off).
STATE      reports/SCOUT_ML_BULL90.json (gitignored).
Usage: venv/bin/python -I scripts/scout_ml_bull90_watch.py
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

DL = 523
SLEEVE = "MOMENTUM"
BULL_CUT = 90.0                    # zone = entry_bull_pct ≥ 90 (≥ 90 % of the scanned coins bullish)
N_MIN, DAYS_MIN = 15, 8
FLOOR_DL = 523                     # forward only: counted from this line's own commit
BE_WR = 0.611                      # FROZEN 2026-11-11: momentum long, 211 real fills — avg loss 0.793 / (avg win 0.505 + 0.793)
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
WAIT_MIN = 1200
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_ML_BULL90.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
STAMPS = ("entry_bull_pct",)
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason",
        "cell_multiplier_source") + STAMPS
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
REGISTRATION = ("in-sample at registration (NEVER counted): 211 real momentum-long fills — bull % > 90 8 · 12 % · −0.63 on 5 days (master 5 · 20 % · −0.56, "
                "B25 3 · 0 % · −0.75 = the 11-09 01:08 burst) vs 85–90 % 15 · 73 % · +0.52; on top of ML_MKT_HOT 4 more blocks: master 87 · +0.22 · +0.011 → "
                "86 · +0.23 · +0.012 %/day, B25 32 · +0.23 · +0.089 → 29 · +0.33 · +0.117 %/day; found on the day it fixes (operator call)")


def load(paths=None):
    """→ (closed full-size MOMENTUM LONG fills — dedup, CLOSED beats OPEN, newest export wins —, the newest read export's download time or None)."""
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
    c["zone"] = np.where(sl.isna(), "unscored", np.where(sl >= BULL_CUT, "zone", "rest"))
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
    """→ markdown lines. since_ms None → the DECISION_LOG 523 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 ML_BULL90 — MOMENTUM LONG with ≥ {BULL_CUT:g} % of coins bullish at entry "
         f"(pre-registered 2026-11-11, DECISION_LOG {DL}, OBSERVE only)", ""]
    since_given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(FLOOR_DL, lag_min=10, label="ML_BULL90 watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {FLOOR_DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    sc = keep[keep.zone != "unscored"] if len(keep) else keep
    zone, rest = sc[sc.zone == "zone"], sc[sc.zone == "rest"]
    uns = keep[keep.zone == "unscored"] if len(keep) else keep
    L += [f"Cohort: CLOSED full-size MOMENTUM LONG fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(FLOOR_DL) + ' commit + 10 min' if since_given is None else 'floor given'}) · "
          f"{_GK.operator_closed_line(ox)}.", "", "| Group | N | days | WR | avg % | Σ$ as traded |", "|---|---|---|---|---|---|",
          _row(f"**zone (bull % ≥ {BULL_CUT:g})**", zone), _row(f"rest (bull % < {BULL_CUT:g})", rest), _row("UNSCORED (no bull % stamp) — never counted", uns), ""]
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
          "Haircut, before → after per batch with the daily compound return and a pre-committed revert before any ship.",
          "Pre-committed revert if ever armed: the first 15 blocked signals re-priced with the live exit average better than the momentum longs kept over the same period → switch it off.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
