#!/usr/bin/env python3
"""👁 Scout — LONG_ADX35 watch line (pre-registered 2026-10-30, DECISION_LOG 418; OBSERVE ONLY — never changes config, no bot API, no network).

QUESTION   Do FRENZY_LONG entries with coin ADX > 35 lose where the rest earn? zone = entry_adx > 35. Theory: FRENZY_LONG buys a bounce; a high ADX means the coin's trend is already old and stretched, so less is left.
EVIDENCE   (in-sample, NEVER counted) master on today's stack re-walked at the −4 stop (1m): zone 11 on 7 days · 64 % · −0.50 %/fill vs 18 · 100 % · +1.50, day-bootstrap P(zone < rest) 0.99; daily compound
           +0.993 → +1.275 (each trade = account). B25: zone 0 vs 4 · 75 % · +0.13.
           AGAINST: the cut was picked after seeing the data; 5 losers only; best 1D split vs shuffled results p ≈ 0.65; the 18 of 18 is too clean; backtest yr6 seed 1 re-walked at +1.5 / −4: zone 311 · 75 % · +0.10 vs 309 · 72 % · −0.05 (reversed). → observe only.
COHORT     CLOSED FRENZY_LONG LONG fills, from the orders exports (export_glob: Downloads + exports_archive/), dedup (opened_at[:19], pair) — CLOSED beats OPEN, newest export wins; operator-closed out
           (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 418 commit + 10 min (forward only). Stamp missing → UNSCORED (listed, never counted). P&L = the booked pnl_percentage.
FROZEN BAR (CLAUDE.md expectancy bar, never re-fit): at the first prefix (by open time) where the ZONE reaches N ≥ 15 on ≥ 8 distinct UTC days, decided ONCE — ① zone avg ≥ rest avg, OR zone avg ≥ 0 → RETIRED;
           ② else FILTER CANDIDATE (operator decides; never automatic) iff zone WR < the frozen breakeven WR 72.7 % (+1.5 / −4 = 4 / 5.5) ∧ zone mean < 0 at a day-clustered bootstrap P(mean < 0) ≥ 0.95 (4,000 resamples, seed 7)
           ∧ no single day and no single pair carries ≥ 50 % of the zone's gross loss; ③ else KEEP OBSERVING. Deferred until the newest orders export was downloaded ≥ the prefix end + the sleeve's max hold.
           Pinned to the +1.5 / −4 / 12 h exit keys in observe_registry (an exit change → ⚠ STALE). A candidate ships only after the 30–50 % in-sample haircut, a before → after per batch with the daily compound
           return and a pre-committed revert.
STATE      reports/SCOUT_LONG_ADX35.json (gitignored).
Usage: venv/bin/python -I scripts/scout_long_adx35_watch.py
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

DL = 418
SLEEVE = "FRENZY_LONG"
ADX_CUT = 35.0
N_MIN, DAYS_MIN = 15, 8
BE_WR = 0.727                      # FROZEN 2026-10-30: FRENZY_LONG breakeven WR on +1.5 / −4 (4 / 5.5)
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
CAP_DEFAULT = 720                  # the sleeve's live max hold (frenzy_max_hold_minutes) — read from the config at run time


def cap_minutes(path=None):
    """frenzy_max_hold_minutes from trading_config.json (unreadable → 720)."""
    import json
    try:
        with open(path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_config.json")) as f:
            c = json.load(f)
        th = c.get("thresholds", c)
        return float(th.get("frenzy_max_hold_minutes", CAP_DEFAULT))
    except Exception:
        return float(CAP_DEFAULT)
ZONE_COL = "entry_adx"
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_LONG_ADX35.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason", ZONE_COL)
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
REGISTRATION = "in-sample at registration (NEVER counted): master at −4 zone 11 · 64 % · −0.50 vs 18 · 100 % · +1.50; B25 zone 0; shuffled-results p ≈ 0.65; backtest reversed (+0.10 vs −0.05)"


def load(paths=None):
    """→ (closed FRENZY_LONG fills — dedup, CLOSED beats OPEN, newest export wins —, the newest read export's download time or None)."""
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
    d = d[(d.entry_strategy.astype(str) == SLEEVE) & (d.direction.astype(str) == "LONG")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair"], keep="last")
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
    """→ markdown lines. since_ms None → the DECISION_LOG 418 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 LONG_ADX35 — FRENZY_LONG entries with ADX > {ADX_CUT:g} (pre-registered 2026-10-30, DECISION_LOG {DL}, "
         "OBSERVE only)", ""]
    since_given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="LONG_ADX35 watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    changed, cap, damaged = False, cap_minutes(), False
    L += [f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if since_given is None else 'floor given'}) · zone = {ZONE_COL} > {ADX_CUT:g} · "
          f"{_GK.operator_closed_line(ox)}.", ""]
    for es in (SLEEVE,):
        s = keep
        sc = s[s.zone != "unscored"] if len(s) else s
        zone, rest = sc[sc.zone == "zone"], sc[sc.zone == "rest"]
        uns = s[s.zone == "unscored"] if len(s) else s
        L += ["| Group | N | days | WR | avg % | Σ$ as traded |", "|---|---|---|---|---|---|",
              _row("**zone**", zone), _row("rest", rest),
              _row("UNSCORED (stamp missing) — never counted", uns), ""]
        why = []
        rec = st.get(es) if ok else None
        if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
            rec, ok, damaged = None, False, True      # a damaged record never re-freezes silently
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
    if damaged:
        L.append(f"⚠ frozen record damaged — fix {os.path.basename(state_path)} by hand; freezing skipped this run.")
    elif not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += [f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}): at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} UTC days, decided ONCE — "
          f"zone avg ≥ rest avg or ≥ 0 → RETIRED; {CANDIDATE} iff zone WR < {BE_WR:.1%} ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ "
          f"{P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no day / pair ≥ {SHARE_MAX:.0%} of the zone's gross loss; else KEEP OBSERVING. "
          "Haircut, before → after per batch with the daily compound return and a pre-committed revert before any ship.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
