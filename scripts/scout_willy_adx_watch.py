#!/usr/bin/env python3
"""👁 Scout — WILLY_ADX55 watch line (pre-registered 2026-10-29, DECISION_LOG 412; OBSERVE ONLY — never changes config, no bot API, no network).

QUESTION   Do WILLY / WILLY_HT entries on a coin already in an EXTREME trend lose? Theory: WILLY buys the first red 5m candle after a
           spike, betting on a bounce back toward the spike high; when the coin's ADX is already > 55 the move is mature — the red candle is
           the trend running out, not a dip to buy.
EVIDENCE   (in-sample, NEVER counted; operator 10-29 after LSK −$485 + 1000PEPE −$905 the same afternoon) entry_adx > 55: B25 4 fills (ORDER,
           T, LSK, 1000PEPE) — all −4 % stops, 3 days; master 6 · 67 % · −0.34 %/fill vs ≤ 55 32 · 84 % · +0.91 (WILLY + HT, today's
           stack). 10 fills on a handful of days → below every gate → observe first (CLAUDE.md OBSERVE-FIRST). BTC STRONG_BULL was the other
           shared trait of the B25 losers but the master contradicts it (2 · 2 W) — not registered.
COHORT     CLOSED FRENZY_WILLY and FRENZY_WILLY_HT fills (each read and decided SEPARATELY — operator 10-29), from the orders exports
           (export_glob: Downloads + exports_archive/), dedup (opened_at[:19], pair, entry_strategy) — CLOSED beats OPEN, newest export
           wins; operator-closed out (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 412 commit + 10 min (forward only). zone =
           entry_adx > 55 · rest = ≤ 55 · missing stamp → UNSCORED (listed, never counted). P&L = the booked pnl_percentage (leverage-
           invariant; HT's ¼ size does not change it).
FROZEN BAR (CLAUDE.md expectancy bar, never re-fit), per sleeve: at the first prefix (by open time) where the ZONE reaches N ≥ 15 on ≥ 8
           distinct UTC days, decided ONCE —
           ① zone avg ≥ rest avg, OR zone avg ≥ 0 → RETIRED;
           ② else FILTER CANDIDATE (operator decides; never automatic) iff zone WR < the frozen breakeven WR 69.0 % (WILLY's V2b exit at
              registration: avg win +1.806 / avg loss −4.009, N 50; HT runs the same exit → the same bar) ∧ zone mean < 0 at a
              day-clustered bootstrap P(mean < 0) ≥ 0.95 (4,000 resamples, seed 7) ∧ no single day and no single pair carries ≥ 50 % of
              the zone's gross loss;
           ③ else KEEP OBSERVING.
           The prefix = every scored fill up to AND including the crossing zone fill (by open time, then pair). Deferred until the newest
           orders export was downloaded ≥ the prefix end + WILLY's max hold (frenzy_willy_max_hold_minutes, 90; exports carry CLOSED
           rows only — an OPEN row is therefore never used to hold, unlike ML_W4's newest-export OPEN check). The 69.0 % breakeven belongs
           to WILLY's V2b exit: observe_registry pins those keys, so a WILLY exit change flags this line ⚠ STALE (re-base or retire).
           WILLY_HT had 0 zone fills at registration — it may never reach the gate; it is read at every batch review regardless. A candidate ships only
           after the 30–50 % in-sample haircut, a before → after per batch with the daily compound return, and a pre-committed revert.
STATE      reports/SCOUT_WILLY_ADX55.json (gitignored).
Usage: venv/bin/python -I scripts/scout_willy_adx_watch.py
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

DL = 412
SLEEVES = ("FRENZY_WILLY", "FRENZY_WILLY_HT")
ADX_CUT = 55.0
N_MIN, DAYS_MIN = 15, 8
BE_WR = 0.690                      # FROZEN 2026-10-29: WILLY V2b breakeven WR (avg win +1.806 / loss −4.009, N 50) — HT runs the same exit
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
CAP_DEFAULT = 90                   # WILLY's live max hold (frenzy_willy_max_hold_minutes) — read from the config at run time


def cap_minutes(path=None):
    """frenzy_willy_max_hold_minutes from trading_config.json (unreadable → 90)."""
    import json
    try:
        with open(path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_config.json")) as f:
            c = json.load(f)
        th = c.get("thresholds", c)
        return float(th.get("frenzy_willy_max_hold_minutes", CAP_DEFAULT))
    except Exception:
        return float(CAP_DEFAULT)
ZONE_COL = "entry_adx"
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_ADX55.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason", ZONE_COL)
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
REGISTRATION = ("in-sample at registration (NEVER counted): entry_adx > 55 — B25 4 fills, all −4 % stops (ORDER 10-24, T 10-25, LSK / "
                "1000PEPE 10-29); master 6 · 67 % · −0.34 vs ≤ 55 32 · 84 % · +0.91 (WILLY + HT)")


def load(paths=None):
    """→ (closed WILLY / HT fills — dedup, CLOSED beats OPEN, newest export wins —, the newest read export's download time or None)."""
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
    d = d[d.entry_strategy.astype(str).isin(SLEEVES) & (d.direction.astype(str) == "LONG")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    adx = pd.to_numeric(c[ZONE_COL], errors="coerce")
    c["zone"] = np.where(adx.isna(), "unscored", np.where(adx > ADX_CUT, "zone", "rest"))
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
    if wr < BE_WR and mz < 0 and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
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
    """→ markdown lines. since_ms None → the DECISION_LOG 412 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 WILLY_ADX55 — WILLY / WILLY_HT entries on a coin with ADX > {ADX_CUT:g} (pre-registered 2026-10-29, DECISION_LOG {DL}, "
         "OBSERVE only)", ""]
    since_given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="WILLY_ADX55 watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    changed, cap = False, cap_minutes()
    L += [f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if since_given is None else 'floor given'}) · zone = {ZONE_COL} > {ADX_CUT:g} · "
          f"each sleeve decided separately · {_GK.operator_closed_line(ox)}.", ""]
    for es in SLEEVES:
        s = keep[keep.entry_strategy.astype(str) == es] if len(keep) else keep
        sc = s[s.zone != "unscored"] if len(s) else s
        zone, rest = sc[sc.zone == "zone"], sc[sc.zone == "rest"]
        uns = s[s.zone == "unscored"] if len(s) else s
        L += [f"### {es}", "", "| Group | N | days | WR | avg % | Σ$ as traded |", "|---|---|---|---|---|---|",
              _row(f"**zone (ADX > {ADX_CUT:g})**", zone), _row(f"rest (ADX ≤ {ADX_CUT:g})", rest),
              _row("UNSCORED (no ADX stamp) — never counted", uns), ""]
        why = []
        rec = st.get(es) if ok else None
        rec = rec if isinstance(rec, dict) else None
        if not rec and ok:
            pre = crossing_prefix(sc) if len(sc) else None
            if pre is not None:
                last = pre.o.max()
                if newest is None or newest < last + pd.Timedelta(minutes=cap):
                    why.append(f"newest export < the crossing fill + {cap:.0f} min (a fill could still be open)")
                else:
                    state, det = verdict(pre[pre.zone == "zone"], pre[pre.zone == "rest"])
                    rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC",
                               run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                               keys=[f"{a:%Y-%m-%dT%H:%M:%S}|{b}|{g}" for a, b, g in zip(pre.o, pre.pair, pre.zone)])
                    if state_path:
                        st[es] = rec
                        changed = True
        live_state, live_det = verdict(zone, rest)
        if rec and ok and state_path:
            L.append(f"**Frozen verdict (crossing at fill {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
            L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
        else:
            L.append(f"Now: ⏳ {live_state} ({live_det}).")
        if why:
            L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
        L.append("")
    if changed and ok and state_path:
        NF.du_save_state(st, state_path)
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += [f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}, per sleeve): at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} UTC days, decided ONCE — "
          f"zone avg ≥ rest avg or ≥ 0 → RETIRED; {CANDIDATE} iff zone WR < {BE_WR:.1%} ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ "
          f"{P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no day / pair ≥ {SHARE_MAX:.0%} of the zone's gross loss; else KEEP OBSERVING. "
          "Haircut, before → after per batch with the daily compound return and a pre-committed revert before any ship.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
