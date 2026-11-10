#!/usr/bin/env python3
"""🎯 Scout — BULLRUN REARM_TP revert gate (shipped 2026-11-10, DECISION_LOG 519, operator declared override "yes build it"). ALERT-only — never
changes config, no bot API. Public Binance 1m klines only (fapi /klines, cached in reports/backtest_cache/bullrun_exit_1m.pkl, a handful of calls).

WHAT       `bullrun_rearm_tp_pct` (1.0) closes a REARM-door Bull-Run trade at +1.0 % net (reason BR_REARM_TP) instead of trailing it (today's REARM
           trail 1.0 × entry ATR, BE arm +1.0 / lock +0.2, ATR-widened stop −0.7 … −1.2). GREEN never uses it.
EVIDENCE   (in-sample, NEVER counted) the 14 REARM fills today's gates keep (DL 518): TP +1.0 vs the 1.0× trail +$338 (9 better / 4 worse;
           Sep-20 +$204 · Sep-21 +$135) on 1m paths (sim corr 0.99) — ONE REARM episode. Below the 57i bar → declared override.
COHORT     CLOSED BULLRUN_LONG fills with entry_br_door = REARM from the orders exports (Downloads + exports_archive/), dedup (opened_at[:19], pair),
           CLOSED beats OPEN, newest export wins; opened ≥ the DECISION_LOG 519 commit + 10 min (forward only); operator-closed out.
           EPISODE = the REARM door period: start = opened_at − entry_br_door_age_min, starts ≤ 30 min apart merged (the engine's flap merge);
           an episode is COMPLETE once every fill in it is CLOSED and priced and the newest export is ≥ max(door start + 24 h — the door's
           bullrun_rearm_max_hours —, last open + 12 h). The gate reads episodes in order and stops at the first incomplete one.
PRICING    only fills the TP really closed (BR_REARM_TP) move: Δ $ = (booked − TRAIL counterfactual) × notional ÷ 100, the trail = the same fill
           walked on its 1m path with the TP off (today's REARM exit, stop tested on the bar low first — biases toward KEEP, stated). Every other
           fill is Δ 0 (deep review 519: a stop's real slippage vs the sim's exact line must never vote). Paths are fetched only once the full
           HORIZON (the engine's 1,200-min max hold) has passed, never cached short / empty, in this scout's own cache (not the sweep's); a walk
           still open at the horizon is marked censored. A sim-with-TP column checks the walk against the booking.
🔒 GATE    (frozen, never re-fit) the first 2 COMPLETE episodes: either episode Δ $ < 0 → REVERT (bullrun_rearm_tp_pct → 0, no further vote); both
           ≥ 0 → KEEP (gate passed, the line closes). Fewer than 2 complete episodes → COLLECTING.
Usage: venv/bin/python -I scripts/scout_bullrun_rearm_tp.py
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_ROOT = os.path.dirname(_D)
_ROOT in sys.path or sys.path.insert(0, _ROOT)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402

DL = 519
TP_PCT = 1.0                        # FROZEN: the shipped value
REARM_TRAIL, ARM, LOCK = 1.0, 1.0, 0.2
FEE = 0.09                          # round-trip taker toll, net-% convention (bullrun_exit_sweep parity)
LADDER = [(4, 3.5), (5, 4.5), (6, 5.5), (8, 7.0), (10, 9.0), (12, 11.0)]
HORIZON = 1200                      # the engine's global max hold (min)
FLAP_MIN, WAIT_MIN, DOOR_MAX_MIN, EPISODES = 30, 720, 1440, 2
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "bullrun_rearm_tp_1m.pkl")   # own cache (review: never the sweep's 360-bar one)
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason", "entry_price",
        "entry_atr_pct", "entry_br_door", "entry_br_door_age_min", "investment", "leverage")


def walk(hi, lo, atr_pct, tp=None, trail=REARM_TRAIL, censored=None):
    """today's REARM Bull-Run exit on a gross % path (hi / lo per 1m bar from the entry price) → exit % net. tp = net % take-profit or None
    (the trail counterfactual). Stop tested on the bar low before the peak updates (conservative, identical for both arms). `censored` (a list)
    gets True appended when the path ends with the trade still open (valued at the last bar's mid). PURE."""
    hi = [h - FEE for h in hi]
    lo = [x - FEE for x in lo]
    a = float(atr_pct or 0.0)
    stop0 = max(min(-0.7, -1.5 * a), -1.2)
    peak = 0.0
    for h, x in zip(hi, lo):
        stop = stop0
        if peak >= ARM:
            rung = max([f for t, f in LADDER if peak >= t], default=-99.0)
            stop = max(max(LOCK, peak - trail * a), rung)
        if x <= stop:
            return stop
        if tp is not None and h >= tp:
            return float(tp)
        peak = max(peak, h)
    if censored is not None:
        censored.append(True)
    return (hi[-1] + lo[-1]) / 2 if hi else 0.0


def episodes(open_ts, age_min, flap_min=FLAP_MIN):
    """→ an episode id per fill: door start = open − age, starts ≤ flap_min apart share one id (in start order). PURE."""
    st = [o - pd.Timedelta(minutes=float(a)) if (a is not None and a == a) else o for o, a in zip(open_ts, age_min)]
    order = np.argsort(np.array(st, dtype="datetime64[ns]"))
    ids, cur, last = [0] * len(st), -1, None
    for i in order:
        if last is None or (st[i] - last).total_seconds() / 60.0 > flap_min:
            cur += 1
        ids[i] = cur
        last = st[i]
    return ids


def gate(ep_deltas):
    """ep_deltas = Δ $ per episode in episode order, None = not complete yet → (state, detail). Reads in order and stops at the first
    incomplete episode (review 519: a later episode never jumps the queue). PURE."""
    done = []
    for v in ep_deltas:
        if v is None:
            break
        done.append(v)
    done = done[:EPISODES]
    if len(done) < EPISODES:
        return "COLLECTING", f"{len(done)}/{EPISODES} complete REARM episodes"
    det = " · ".join(f"episode {i + 1} Δ {d:+,.0f} $" for i, d in enumerate(done))
    if any(d < 0 for d in done):
        return "REVERT → bullrun_rearm_tp_pct 0 (no further vote)", det
    return "KEEP — gate passed, line closes", det


def load(paths=None):
    fr, newest = [], None
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if not {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            continue
        fr.append(d.assign(_m=os.path.getmtime(f)))
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    if not fr:
        return pd.DataFrame(columns=list(COLS)), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[(d.entry_strategy.astype(str) == "BULLRUN_LONG") & d.entry_br_door.astype(str).str.upper().eq("REARM")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    return d[d.o.notna()].sort_values("o").reset_index(drop=True), newest


def _klines(pair, opened_at, cache, now_ms):
    """the fill's 1m path (HORIZON bars) — fetched only once open + HORIZON has passed; a short / failed answer is returned as None and
    NEVER cached (review 519)."""
    key = (pair, str(opened_at))
    if cache.get(key):
        return cache[key]
    import requests
    t0 = pd.Timestamp(opened_at)
    start = int((t0.tz_localize("UTC") if t0.tzinfo is None else t0).timestamp() * 1000)
    if now_ms < start + HORIZON * 60_000:
        return None
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol=pair, interval="1m", startTime=start, limit=HORIZON), timeout=10).json()
        time.sleep(0.12)
    except Exception:
        return None
    if not (isinstance(r, list) and len(r) >= HORIZON):
        return None
    cache[key] = [(float(k[2]), float(k[3])) for k in r]
    return cache[key]


def run(now_ms=None, paths=None, since_ms=None, fetch=True):
    """→ markdown lines. since_ms None → the DECISION_LOG 519 commit + 10 min (not in git → NOT EVALUABLE)."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🎯 BULLRUN REARM_TP — REARM-door take-profit +{TP_PCT:g} % vs the trail (shipped 2026-11-10, DECISION_LOG {DL}, declared override)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="BULLRUN REARM_TP gate")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    d, newest = load(paths)
    d = d[d.o >= since] if len(d) else d
    if len(d):
        d, ox = _GK.split_operator_closed(d)
    if not len(d):
        return L + [f"No REARM fill since {since:%Y-%m-%d %H:%M} UTC — {gate([])[1]}. (Bull-Run last fired 2026-09-21.)", ""]
    d = d.reset_index(drop=True)
    d["ep"] = episodes(list(d.o), list(pd.to_numeric(d.entry_br_door_age_min, errors="coerce")))
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    rows = []
    for _, r in d.iterrows():
        closed = str(r.status).upper() == "CLOSED"
        kl = _klines(r.pair, r.opened_at, cache, now_ms) if (closed and fetch) else None
        notional = float(pd.to_numeric(r.investment, errors="coerce") or 0) * float(pd.to_numeric(r.leverage, errors="coerce") or 0)
        booked = float(pd.to_numeric(r.pnl_percentage, errors="coerce")) if closed else float("nan")
        trail = sim_tp = float("nan")
        cens = []
        if kl:
            ep = float(r.entry_price)
            hi = [(h / ep - 1) * 100 for h, _ in kl]
            lo = [(x / ep - 1) * 100 for _, x in kl]
            trail, sim_tp = walk(hi, lo, r.entry_atr_pct, censored=cens), walk(hi, lo, r.entry_atr_pct, tp=TP_PCT)
        tp_closed = "REARM_TP" in str(r.close_reason)
        d_usd = (booked - trail) * notional / 100 if (tp_closed and trail == trail) else (0.0 if closed and not tp_closed else float("nan"))
        rows.append(dict(ep=r.ep, o=r.o, start=r.o - pd.Timedelta(minutes=float(pd.to_numeric(r.entry_br_door_age_min, errors="coerce") or 0.0)),
                         pair=r.pair, closed=closed, tp=tp_closed, reason=str(r.close_reason), booked=booked, trail=trail, sim_tp=sim_tp,
                         censored=bool(cens), d_usd=d_usd))
    if fetch and os.path.isdir(os.path.dirname(CACHE)):
        pickle.dump(cache, open(CACHE, "wb"))
    t = pd.DataFrame(rows)
    deltas, lines = [], ["| Episode (door start) | fills | closed | TP closes | TP booked Σ % | their trail Σ % | Δ $ | complete |", "|---|---|---|---|---|---|---|---|"]
    for ep, g in t.groupby("ep", sort=True):
        due = max(g.start.min() + pd.Timedelta(minutes=DOOR_MAX_MIN), g.o.max() + pd.Timedelta(minutes=WAIT_MIN))
        complete = bool(g.closed.all() and g.d_usd.notna().all() and newest is not None and newest >= due)
        dv = float(g.d_usd.fillna(0).sum())
        deltas.append(dv if complete else None)
        lines.append(f"| {ep + 1} ({g.start.min():%m-%d %H:%M}) | {len(g)} | {int(g.closed.sum())} | {int(g.tp.sum())} | "
                     f"{g[g.tp].booked.sum():+.2f} | {g[g.tp].trail.sum():+.2f} | {dv:+,.0f} | {'✅' if complete else '⏳'} "
                     f"{'· ' + str(int(g[g.tp].censored.sum())) + ' censored' if g[g.tp].censored.any() else ''}{'· path missing' if (g.tp & g.trail.isna()).any() else ''} |")
    state, det = gate(deltas)
    cal = t[t.tp & t.sim_tp.notna()]
    L += [f"Cohort: CLOSED REARM Bull-Run fills opened ≥ {since:%Y-%m-%d %H:%M} UTC (DL {DL} commit + 10 min).", ""] + lines + ["",
          f"**Gate (first {EPISODES} complete episodes): {state}** — {det}.",
          f"Walk check: sim-with-TP vs booked, mean |Δ| {(cal.sim_tp - cal.booked).abs().mean():.2f} pts on {len(cal)} fills" if len(cal) else "Walk check: no closed fill yet.",
          "Bias on record: the trail walk tests the stop on the bar low first (stops some trades live rode) → leans toward KEEP; bar 0 holds "
          "pre-entry prints (both arms alike).",
          "Evidence at ship (never counted): 14 REARM fills, one episode — TP +1.0 vs the 1.0× trail +$338 (9 better / 4 worse).", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
