#!/usr/bin/env python3
"""🧭📊 Scout — two FRENZY entry observe lines from the two-year winners-vs-losers study (reports/YR6_WL_FRENZY.md §6; operator-approved
2026-10-13, DECISION_LOG 288). OBSERVE only — never changes config, never calls the bot API.

COHORT (both)  CLOSED non-MANUAL fills of the line's sleeve in the ~/Downloads orders exports, dedup (opened_at[:19], pair, entry_strategy) —
               never id; a CLOSED row beats an OPEN one whatever the export order, else the newest export (mtime) wins — opened ≥ the
               "(DECISION_LOG 288)" commit time + 10 min (git %ct; commit not found → the PIN DEPLOY_PIN + 10 min). y = pnl_percentage
               (leverage-invariant); Σ$ = the fill's own pnl. These are ENTRY filters: zone fills traded live, so ZONE vs REST compares the
               fills' actual results (no re-walk). UNSCORED fills are listed apart and never counted in zone or rest.

LINE 1 LITE_BTC1H_FALLING  sleeve FRENZY_LITE · zone = entry_btc_1h_slope ≤ 0 (the engine's _current_btc_1h_slope = (BTC 1h EMA20 − EMA20 3
               bars ago) / EMA20 3 bars ago × 100, stamped on every FRENZY fill; the study's exact column, sign and 0 cut — never re-fit);
               stamp missing → UNSCORED (final). Study: 2025 −0.674 %/fill (250, CI [−1.07, −0.28]), 2026-H2 −0.495 (P 0.91), day-unit p_adj
               0.004 — REVERSES in the 2026-H1 bear (+0.321, 143) and weak inside the top-50 (2025 −0.336, P 0.85).
LINE 2 LONG_NONTOP50       sleeve FRENZY_LONG · zone = the fill's pair-day is OUTSIDE top-50(day) ∪ top-50(day − 1), where top-50(D) = the 50
               USDT perpetuals with the highest quote volume on the CLOSED UTC day D − 1 (the replay's rule: scripts/backtest_phase1_universe.py
               / backtest_extend_cache.py write reports/backtest_cache/universe_daily_top50.csv; the study's top50_pairdays() takes that day's
               OR the prior day's list). FRENZY fills stamp no entry_pair_rank (None on the FRENZY path), so membership is rebuilt: the
               universe csv where it covers D, else daily quote volumes fetched once per closed day (1d klines of every USDT perpetual alive
               that day — scripts/scout_gvol.py's budgeted fetcher: ≤ 1,000 weight/min of ours, used-weight header pause, 418 / 429 →
               stop + the shared cooldown reports/backtest_cache/.binance_cooldown_until; all-or-nothing per refresh) cached in
               reports/backtest_cache/scout_frenzy_wl/daily_qvol.json. Pair in either known list → top-50; a needed list unknown → UNSCORED
               (pending — holds the freeze — until 7 days after the fill's day, then final). Study: non-top-50 −0.273 / −0.351 / −0.516 vs
               top-50 +0.154 / −0.002 / +0.022 %/fill in 2025 / 2026-H1 / 2026-H2; post-hoc, failed the family-wise null.

🔒 FROZEN BAR (pre-registered 2026-10-13, DECISION_LOG 288) — per line, decided ONCE at the first prefix by (open time, pair) of the line's
               scored fills where the zone reaches N ≥ 20 on ≥ 8 UTC days:
                 RETIRE                                 if zone avg ≥ rest avg (rest = the prefix's scored non-zone fills);
                 FILTER CANDIDATE (operator decides)   iff zone WR < the sleeve's breakeven WR = stop / (TP + stop) of TODAY's config exit
                                                        (LITE +3 / −3 → 50 %; LONG +1.5 / −3 → 66.7 %) ∧ day-clustered bootstrap
                                                        P(zone avg < 0) ≥ 0.95 (4,000, seed 7) ∧ no single pair or day ≥ 50 % of the zone's
                                                        gross loss;
                 else KEEP OBSERVING.
               Deferred while a fill of the sleeve opened ≤ the prefix end is OPEN in the newest export or UNSCORED-pending. Never re-fit.
Usage: venv/bin/python scripts/scout_frenzy_wl_lines.py [--selftest]
"""
import glob
import os as _os_eg, sys as _sys_eg; _d_eg = _os_eg.path.dirname(_os_eg.path.abspath(__file__)); _d_eg in _sys_eg.path or _sys_eg.path.insert(0, _d_eg)
import export_glob as _EG   # 📦 (DECISION_LOG 294) paper + post-go-live LIVE exports
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (bootstrap, gross-loss share, freeze hold, state I/O, shared cooldown)

LINES = ("LITE_BTC1H_FALLING", "LONG_NONTOP50")
SLEEVE = {"LITE_BTC1H_FALLING": "FRENZY_LITE", "LONG_NONTOP50": "FRENZY_LONG"}
SLOPE_CUT = 0.0                                      # zone = entry_btc_1h_slope ≤ 0 (frozen; the study's sign cut)
TOP_N = 50
N_MIN, DAYS_MIN = 20, 8
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
DEPLOY_GREP = "(DECISION_LOG 288)"
DEPLOY_PIN = "2026-10-13 12:00:00"                  # PIN (UTC) — used only if git cannot find the DL 288 commit
LAG_MIN = 10
PENDING_DAYS = 7                                     # an unknown top-50 membership stops holding the freeze 7 days after the fill's day
FETCH_BUDGET_S = 120.0
DAY = 86_400_000
RETIRE, CAND, KEEP = "RETIRE", "FILTER CANDIDATE (operator decides)", "KEEP OBSERVING"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(_ROOT, "reports", "SCOUT_FRENZY_WL_LINES.json")
UNIVERSE_CSV = os.path.join(_ROOT, "reports", "backtest_cache", "universe_daily_top50.csv")
VOL_CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_frenzy_wl", "daily_qvol.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason",
        "entry_btc_1h_slope")
STUDY_REF = {
    "LITE_BTC1H_FALLING": "yr6 seed 1: zone 2025 250 · 38.8 % · −0.674 % (kept +0.088) · 2026-H1 143 · 51.7 % · **+0.321 %** (kept +0.166) · "
                          "2026-H2 86 · 38.4 % · −0.495 % (kept +0.457) · 2025 top-50 81 · −0.336 % (P 0.85)",
    "LONG_NONTOP50": "yr6 seed 1: zone 2025 136 · −0.273 % (top-50 +0.154) · 2026-H1 116 · −0.351 % (−0.002) · 2026-H2 68 · −0.516 % (+0.022); "
                     "confirmation P 0.93 / 0.88 / 0.88; seeds 2+3 agree",
}
REVERT = ("Pre-committed revert if ever armed: the first 10 blocked signals re-priced with the live exit → WR ≥ the sleeve breakeven or "
          "Σ > 0 → switch it off")


# ─────────────────────────── deploy floor ───────────────────────────
def _git_ct(grep):
    try:
        out = subprocess.run(["git", "-C", _ROOT, "log", "--format=%ct", "--grep=" + grep, "--fixed-strings", "--reverse"],
                             capture_output=True, text=True, timeout=10)
        return pd.Timestamp(int(out.stdout.strip().splitlines()[0]) * 1000, unit="ms")
    except Exception:
        return None


def deploy_ts():
    """the counted floor = the "(DECISION_LOG 288)" commit time + 10 min; not found → DEPLOY_PIN + 10 min."""
    t = _git_ct(DEPLOY_GREP)
    return (pd.Timestamp(DEPLOY_PIN) if t is None else t) + pd.Timedelta(minutes=LAG_MIN)


# ─────────────────────────── orders ───────────────────────────
def _read_exports():
    fr = []
    for f in _EG.glob_exports(EXPORT_GLOB):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    return fr


def load(orders=None):
    """→ (CLOSED FRENZY_LITE / FRENZY_LONG LONG fills, prepared; [(opened_at, 'OPEN <sleeve> <pair>')] from the newest export)."""
    if orders is None:
        fr = _read_exports()
        raw = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable") if fr else pd.DataFrame(columns=list(COLS))
        newest = fr[int(np.argmax([x._m.iloc[0] for x in fr]))] if fr else raw.iloc[:0]
    else:
        raw, newest = orders, orders
    o = raw.reindex(columns=list(COLS)).copy()
    if len(o):
        o = o.assign(_k=o.opened_at.astype(str).str[:19], _c=o.status.astype(str).str.upper().eq("CLOSED"))
        o = o.sort_values("_c", kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
        o = o[o._c & (o.direction.astype(str).isin(["LONG", "nan"])) & o.entry_strategy.astype(str).isin(set(SLEEVE.values()))]
    o = o.copy()
    o["ts"] = pd.to_datetime(o.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    o["pct"] = pd.to_numeric(o.pnl_percentage, errors="coerce")
    o["usd"] = pd.to_numeric(o.pnl, errors="coerce")
    o = o[o.ts.notna() & o.pct.notna()].copy()
    o["pair"] = o.pair.astype(str)
    o["strat"] = o.entry_strategy.astype(str)
    o["why"] = o.close_reason.astype(str)
    o["day"] = o.ts.dt.strftime("%Y-%m-%d")
    o["slope"] = pd.to_numeric(o.entry_btc_1h_slope, errors="coerce")
    o = o[["pair", "strat", "ts", "day", "pct", "usd", "why", "slope"]].sort_values(["ts", "pair"], kind="stable").reset_index(drop=True)
    n = newest.reindex(columns=list(COLS))
    n = n[(n.status.astype(str).str.upper() == "OPEN") & n.entry_strategy.astype(str).isin(set(SLEEVE.values()))]
    t = pd.to_datetime(n.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    opens = [(a, f"OPEN {s} {p}") for a, s, p in zip(t, n.entry_strategy.astype(str), n.pair.astype(str)) if pd.notna(a)]
    return o, opens


# ─────────────────────────── top-50 universe ───────────────────────────
def _d(ms):
    return pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%d")


def _ms(day):
    return int(pd.Timestamp(day).value // 1_000_000)


def universe_csv_sets(path=None):
    """{D: set(pairs)} from universe_daily_top50.csv (row date D = ranked by day D − 1's quote volume)."""
    path = path or UNIVERSE_CSV
    if not os.path.exists(path):
        return {}
    u = pd.read_csv(path)
    u = u[u["rank"] <= TOP_N]
    return {_d(k): set(g.pair.astype(str)) for k, g in u.groupby("date_ms")}


def vol_cache_load(path=None):
    """{volume day V: {pair: quote volume}} — only COMPLETE days (written all-or-nothing)."""
    path = path or VOL_CACHE
    try:
        with open(path) as f:
            d = json.load(f)
        return {k: {p: float(q) for p, q in v.items()} for k, v in (d.get("complete") or {}).items()}
    except Exception:
        return {}


def vol_cache_save(vols, path=None):
    path = path or VOL_CACHE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(dict(complete=vols), f)
    os.replace(tmp, path)


def top_from_vols(v):
    return {p for p, _ in sorted(v.items(), key=lambda x: (-x[1], x[0]))[:TOP_N]}


def u_set(D, csv_sets, vols):
    """top-50(D) = ranked by the closed day D − 1 → set, or None when unknown."""
    if D in csv_sets:
        return csv_sets[D]
    V = (pd.Timestamp(D) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return top_from_vols(vols[V]) if V in vols else None


def top50_status(pair, day, csv_sets, vols):
    """True / False / None (unknown) — pair ∈ top-50(day) ∪ top-50(day − 1)."""
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    a, b = u_set(day, csv_sets, vols), u_set(prev, csv_sets, vols)
    if (a is not None and pair in a) or (b is not None and pair in b):
        return True
    if a is None or b is None:
        return None
    return False


def needed_vol_days(fills, csv_sets, vols):
    """volume days V (closed) whose daily ranking would decide an unknown fill."""
    need = set()
    for r in fills.itertuples():
        if top50_status(r.pair, r.day, csv_sets, vols) is not None:
            continue
        for k in (1, 2):
            D = (pd.Timestamp(r.day) - pd.Timedelta(days=k - 1)).strftime("%Y-%m-%d")
            if D not in csv_sets:
                need.add((pd.Timestamp(r.day) - pd.Timedelta(days=k)).strftime("%Y-%m-%d"))
    return sorted(need - set(vols))


def _gvol_fetch(EX, retry, vdays, deadline):
    """→ {V: {pair: qvol}} for every USDT perpetual alive on V (scout_gvol's markets list + budgeted kline fetcher). Raises on any failure."""
    import scout_gvol as SG
    meta = SG.universe_meta(EX, retry)
    if not meta:
        raise RuntimeError("no markets list")
    v0, v1 = _ms(min(vdays)), _ms(max(vdays))
    jobs = {p: (m["id"], "1d", v0, v1, DAY) for p, m in meta.items() if SG._alive(m, v0)}
    rows = SG._fetch_all(jobs, deadline)
    out = {V: {} for V in vdays}
    for p, rr in rows.items():
        for x in rr:
            V = _d(x[0])
            if V in out:
                out[V][p] = float(x[6])
    return out


def ensure_volumes(vdays, now_ms, EX=None, retry=None, fetch=True, fetcher=None, path=None):
    """fetch the missing closed volume days (all-or-nothing) into the cache → notes."""
    if not vdays:
        return []
    vdays = [V for V in vdays if _ms(V) + DAY <= now_ms]
    if not vdays:
        return []
    if not fetch:
        return [f"top-50 volumes for {', '.join(vdays)} not cached (fetch off)"]
    if NF._cooldown_until() > now_ms:
        return [f"Binance cooldown live until {_d(NF._cooldown_until())} — top-50 volumes for {', '.join(vdays)} not fetched"]
    if fetcher is None:
        if EX is None:
            return [f"no exchange handle — top-50 volumes for {', '.join(vdays)} not fetched"]
        fetcher = lambda vd, dl: _gvol_fetch(EX, retry or (lambda f, *a, **k: f(*a, **k)), vd, dl)   # noqa: E731
    try:
        got = fetcher(vdays, time.monotonic() + FETCH_BUDGET_S)
    except Exception as e:
        if type(e).__name__ == "Banned":                    # review fix: a 418 IP ban arms its own (2 h / Retry-After) cooldown, not 429's 10 min
            _m = re.search(r"Binance (\d{3}) \(retry after (\d+)", str(e))
            NF._arm_cooldown(int(_m.group(1)) if _m else 429, _m.group(2) if _m else None, now_ms)
        return [f"top-50 volume fetch failed ({str(e)[:80]}) — nothing cached, retried next run"]
    vols = vol_cache_load(path)
    ok = []
    for V in vdays:
        v = got.get(V) or {}
        if "BTCUSDT" in v and len(v) >= 100:                 # sanity: a complete market day
            vols[V] = v
            ok.append(V)
    if ok:
        vol_cache_save(vols, path)
    bad = sorted(set(vdays) - set(ok))
    return [f"top-50 volumes cached for {', '.join(ok)}" if ok else ""] + ([f"incomplete volume day(s) {', '.join(bad)} — not cached"] if bad else [])


# ─────────────────────────── zone rules ───────────────────────────
def lite_group(slope):
    s = np.asarray(slope, dtype=float)
    return np.where(np.isnan(s), "unscored", np.where(s <= SLOPE_CUT, "zone", "rest"))


def long_group(fills, csv_sets, vols, now_ms):
    """→ (grp array, pending bool array): non-top-50 = zone; unknown = unscored (pending until PENDING_DAYS after the fill's day)."""
    g, pend = [], []
    for r in fills.itertuples():
        s = top50_status(r.pair, r.day, csv_sets, vols)
        g.append("unscored" if s is None else "rest" if s else "zone")
        pend.append(s is None and now_ms < _ms(r.day) + (1 + PENDING_DAYS) * DAY)
    return np.array(g, dtype=object), np.array(pend, dtype=bool)


# ─────────────────────────── verdict / freeze ───────────────────────────
def breakeven(tp, stop):
    return 100.0 * float(stop) / (float(tp) + float(stop)) if stop and (tp + stop) > 0 else None


def verdict(zone, rest, be):
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz, wr = float(zone.pct.mean()), 100.0 * float((zone.pct > 0).mean())
    mr = float(rest.pct.mean()) if len(rest) else float("nan")
    p = NF.p_mean_neg(zone.pct.values, zone.day.values, BOOT_N, BOOT_SEED)
    sd, sp = NF.gross_loss_share(zone.pct.values, zone.day.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {wr:.0f} % vs breakeven {('%.1f %%' % be) if be is not None else 'n/a'} · avg {mz:+.3f} % vs rest "
           f"{mr:+.3f} % (N {len(rest)}) · P(avg<0) {(p if p is not None else float('nan')):.2f} · top day {sd * 100:.0f} % / top pair "
           f"{sp * 100:.0f} % of the zone loss")
    if len(rest) and mz >= mr:
        return RETIRE, det
    if be is not None and wr < be and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return CAND, det
    return KEEP, det


def crossing_prefix(sc):
    """scored fills sorted by (open time, pair) → the prefix ending at the zone fill where the zone first reaches N ≥ 20 on ≥ 8 days."""
    z = sc.sort_values(["ts", "pair"], kind="stable")
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(z.grp.values, z.day.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(d)
        if nz >= N_MIN and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def freeze(st, line, sc, be, now_iso, cfg, hold=None, why=None):
    """decide `line` ONCE at its crossing prefix; deferred by hold. → (st, changed)."""
    if line in st:
        return st, False
    pre = crossing_prefix(sc)
    if pre is None:
        return st, False
    last = pre.ts.max()
    w = []
    if NF.freeze_hold(last, hold, w):
        if why is not None:
            why += [f"{line} — {s}" for s in w]
        return st, False
    zone, rest = pre[pre.grp == "zone"], pre[pre.grp == "rest"]
    state, det = verdict(zone, rest, be)
    st[line] = dict(state=state, detail=det, be=None if be is None else round(be, 2), config=cfg, at=f"{last:%Y-%m-%d %H:%M} UTC",
                    run_at=now_iso, n=len(zone), days=int(zone.day.nunique()),
                    keys=[f"{a}|{b}|{g}" for a, b, g in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair, pre.grp)])
    return st, True


# ─────────────────────────── rendering ───────────────────────────
HDR = ["| Group | N | days | WR | avg % | Σ$ as-sized | worst |", "|---|---|---|---|---|---|---|"]


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean() * 100:.0f} % | {g.pct.mean():+.3f} % | "
            f"{g.usd.sum():+,.0f} | {g.pct.min():+.2f} % |")


RULES = {"LITE_BTC1H_FALLING": ("🧭", "FRENZY_LITE fill with entry_btc_1h_slope ≤ 0 (BTC 1h EMA20 falling at entry)",
                                "zone (BTC 1h slope ≤ 0)", "rest (slope > 0)", "UNSCORED (no slope stamp)"),
         "LONG_NONTOP50": ("📊", "FRENZY_LONG fill whose pair is outside top-50(day) ∪ top-50(day − 1) by prior-closed-day quote volume",
                           "zone (not top-50)", "rest (top-50)", "UNSCORED (top-50 membership unknown)")}


def section(line, c, st, ok, floor, be, lv, why):
    icon, rule, zl, rl, ul = RULES[line]
    sc = c[c.grp != "unscored"]
    zone, rest = sc[sc.grp == "zone"], sc[sc.grp == "rest"]
    L = [f"## {icon} {line} — {SLEEVE[line]} entry observe line (pre-registered 2026-10-13, DECISION_LOG 288, OBSERVE only)", "",
         f"Zone (frozen): {rule}. Counted: CLOSED non-MANUAL {SLEEVE[line]} fills opened ≥ {floor:%Y-%m-%d %H:%M} UTC (the \"{DEPLOY_GREP}\" "
         f"commit + {LAG_MIN} min" + ("" if _git_ct(DEPLOY_GREP) is not None else f"; commit not found → PIN {DEPLOY_PIN} UTC") +
         "); actual pnl % (leverage-invariant), no re-walk.", "",
         *HDR, _row(f"**{zl}**", zone), _row(rl, rest), _row(ul, c[c.grp == "unscored"]), ""]
    if len(c):
        L.append("Fills: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair} {r.pct:+.2f} % [{r.grp}{', pending' if getattr(r, 'pend', False) else ''}]"
            + (f" (slope {r.slope:+.4f})" if line == "LITE_BTC1H_FALLING" and r.slope == r.slope else "") for r in c.itertuples()))
    L.append(f"Backtest reference (not counted): {STUDY_REF[line]}.")
    be_txt = f"{be:.1f} % = stop / (TP + stop) of today's exit +{lv[0]:g} / −{lv[1]:g}" if be is not None else "n/a (no stop in today's exit)"
    L.append(f"Bar (🔒 frozen 2026-10-13, DECISION_LOG 288; decided ONCE at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} "
             f"UTC days, never re-fit): zone avg ≥ rest avg → {RETIRE}; zone WR < breakeven {be_txt} ∧ day-clustered P(zone avg < 0) ≥ "
             f"{P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no pair / day ≥ {SHARE_MAX * 100:.0f} % of the zone loss → {CAND}; else {KEEP}. "
             "Open / pending-unscored fills ≤ the prefix end hold the freeze.")
    s0 = st.get(line) if ok else None
    live_state, live_det = verdict(zone, rest, be)
    if s0:
        L.append(f"**Frozen (crossing at {s0['at']}, run {s0.get('run_at', '?')}, zone N {s0['n']} · {s0['days']} d, breakeven {s0['be']} %): "
                 f"{s0['state']}** ({s0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    else:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    w = [x for x in why if x.startswith(line)]
    if w:
        L.append("⏸ freezing deferred this run: " + " · ".join(w) + " — re-checked next run.")
    return L + [REVERT, ""]


def run(now_ms=None, orders=None, state_path=None, open_ts=None, EX=None, retry=None, fetch=True, deploy=None, th=None,
        csv_sets=None, vol_path=None, fetcher=None):
    """→ markdown lines for both lines (the caller wraps it in its own try)."""
    import scout_frenzy_willy_exit as FX
    now_ms = now_ms or int(time.time() * 1000)
    state_path = state_path or STATE
    o, opens = load(orders)
    if open_ts is not None:
        opens = list(open_ts)
    floor = deploy_ts() if deploy is None else pd.Timestamp(deploy)
    o = o[(o.ts >= floor)].reset_index(drop=True)
    man = o[o.why.str.upper().str.startswith("MANUAL")]
    o = o[~o.why.str.upper().str.startswith("MANUAL")].reset_index(drop=True)
    csv_sets = universe_csv_sets() if csv_sets is None else csv_sets
    notes = []
    lg = o[o.strat == "FRENZY_LONG"]
    vols = vol_cache_load(vol_path)
    need = needed_vol_days(lg, csv_sets, vols)
    if need:
        notes += [x for x in ensure_volumes(need, now_ms, EX, retry, fetch, fetcher, vol_path) if x]
        vols = vol_cache_load(vol_path)
    st, ok = NF.du_load_state(state_path, now_ms)
    now_iso = pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC")
    out, why, changed = [], [], False
    for line in LINES:
        sl = SLEEVE[line]
        c = o[o.strat == sl].copy()
        if line == "LITE_BTC1H_FALLING":
            c["grp"] = lite_group(c.slope) if len(c) else np.array([], dtype=object)
            c["pend"] = False
        else:
            c["grp"], c["pend"] = long_group(c, csv_sets, vols, now_ms) if len(c) else (np.array([], dtype=object), np.array([], dtype=bool))
        lv = FX.frenzy_levels(th, sl)
        be = breakeven(lv[0], lv[1])
        sc = c[c.grp != "unscored"]
        hold = [x for x in opens if str(x[1]).split(" ")[1:2] == [sl]] + [(r.ts, f"unscored-pending {r.pair}") for r in c[c.pend].itertuples()]
        if ok:
            st, ch = freeze(st, line, sc, be, now_iso, dict(tp=lv[0], stop=lv[1], hold=lv[2]), hold, why)
            changed |= ch
        out += section(line, c, st, ok, floor, be, lv, why)
    if changed:
        NF.du_save_state(st, state_path)
    if not ok:
        out.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    if len(man):
        out.append(f"Excluded: {len(man)} manually closed fill(s): " + ", ".join(f"{r.ts:%m-%d %H:%M} {r.pair} {r.strat}" for r in man.itertuples()) + ".")
    if notes:
        out.append(f"Data: {' · '.join(notes)}.")
    return out + [""]


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    import socket
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved_sock, saved_glob, saved_cd, saved_cp = socket.socket, globals()["EXPORT_GLOB"], NF.SHARED_COOLDOWN, NF._cooldown_paths

    class _NoNet(socket.socket):
        def connect(self, *a, **k):
            raise RuntimeError("network blocked (selftest)")
    socket.socket = _NoNet
    td = tempfile.mkdtemp()
    globals()["EXPORT_GLOB"] = os.path.join(td, "none_*.csv")
    NF.SHARED_COOLDOWN = os.path.join(td, ".binance_cooldown_until")
    NF._cooldown_paths = lambda: [NF.SHARED_COOLDOWN]          # never read / write the repo's live cooldown files
    try:
        chk((SLOPE_CUT, TOP_N, N_MIN, DAYS_MIN, P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED, DEPLOY_GREP, LAG_MIN, PENDING_DAYS) ==
            (0.0, 50, 20, 8, 0.95, 0.50, 4000, 7, "(DECISION_LOG 288)", 10, 7), "pre-registered constants pinned")
        chk(SLEEVE == {"LITE_BTC1H_FALLING": "FRENZY_LITE", "LONG_NONTOP50": "FRENZY_LONG"}, "sleeves pinned")
        chk(deploy_ts() >= pd.Timestamp("2026-10-13") + pd.Timedelta(minutes=LAG_MIN), "floor = DL 288 commit (or the pin) + 10 min")
        chk(abs(breakeven(3.0, 3.0) - 50.0) < 1e-9 and abs(breakeven(1.5, 3.0) - 66.6667) < 1e-3 and breakeven(1.5, None) is None,
            "breakeven = stop / (TP + stop): LITE 50 %, LONG 66.7 %")
        # ── LITE zone: ≤ 0 inclusive, NaN unscored
        chk(list(lite_group([0.0, -0.1449, 0.1527, np.nan, 1e-9])) == ["zone", "zone", "rest", "unscored", "rest"], "LITE zone = slope ≤ 0")
        # ── top-50 membership: csv day, prior-day union, vol-cache fallback, unknown
        csv = {"2026-10-11": {"AUSDT", "BUSDT"}, "2026-10-12": {"BUSDT", "CUSDT"}}
        vols = {"2026-10-12": dict({f"X{i}USDT": 1e9 - i for i in range(60)}, DUSDT=2e9, BTCUSDT=3e9)}
        chk(top50_status("AUSDT", "2026-10-12", csv, vols) is True, "prior-day top-50 counts")
        chk(top50_status("CUSDT", "2026-10-12", csv, vols) is True, "same-day top-50 counts")
        chk(top50_status("ZUSDT", "2026-10-12", csv, vols) is False, "in neither list → not top-50")
        chk(top50_status("DUSDT", "2026-10-13", csv, vols) is True and top50_status("X55USDT", "2026-10-13", csv, vols) is False,
            "top-50(10-13) from the cached 10-12 volumes (#51+ out)")
        chk(top50_status("ZUSDT", "2026-10-14", csv, vols) is None and top50_status("DUSDT", "2026-10-14", csv, vols) is True,
            "unknown list → None unless the pair is in the known one")
        lg = pd.DataFrame(dict(pair=["ZUSDT"], day=["2026-10-14"]))
        chk(needed_vol_days(lg, csv, vols) == ["2026-10-13"], "needs only the missing closed volume day")
        # ── fetch: all-or-nothing into the cache; network blocked otherwise
        vp = os.path.join(td, "vol.json")
        now = _ms("2026-10-15") + 3600_000
        n1 = ensure_volumes(["2026-10-13"], now, fetcher=lambda vd, dl: (_ for _ in ()).throw(RuntimeError("boom")), path=vp)
        chk("failed" in n1[0] and vol_cache_load(vp) == {}, "a failed fetch caches nothing")
        n2 = ensure_volumes(["2026-10-13"], now, fetcher=lambda vd, dl: {"2026-10-13": {"BTCUSDT": 1.0}}, path=vp)
        chk(vol_cache_load(vp) == {} and "incomplete" in " ".join(n2), "an incomplete market day is not cached")
        full = {"2026-10-13": dict({f"Y{i}USDT": 1e8 + i for i in range(120)}, BTCUSDT=5e9, ZUSDT=4e9)}
        ensure_volumes(["2026-10-13"], now, fetcher=lambda vd, dl: full, path=vp)
        chk(top50_status("ZUSDT", "2026-10-14", csv, {**vols, **vol_cache_load(vp)}) is True, "fetched day decides membership")
        chk("no exchange handle" in ensure_volumes(["2026-10-13"], now, path=os.path.join(td, "v2.json"))[0], "no EX → no fetch, noted")
        chk(ensure_volumes(["2026-10-15"], now, fetcher=lambda vd, dl: full, path=vp) == [], "an unclosed day is never fetched")
        with open(NF.SHARED_COOLDOWN, "w") as f:
            f.write(str(now + 600_000))
        chk("cooldown" in ensure_volumes(["2026-10-13"], now, fetcher=lambda vd, dl: full, path=os.path.join(td, "v3.json"))[0],
            "shared cooldown respected")
        os.remove(NF.SHARED_COOLDOWN)
        n5 = ensure_volumes(["2026-10-13"], now, fetcher=lambda vd, dl: (_ for _ in ()).throw(type("Banned", (Exception,), {})("418")),
                            path=os.path.join(td, "v5.json"))
        chk("failed" in n5[0] and NF._cooldown_until() > now, "a 418 / 429 ban arms the shared cooldown")
        os.remove(NF.SHARED_COOLDOWN)
        # ── LONG grouping + pending window
        fl = pd.DataFrame(dict(pair=["AUSDT", "ZUSDT", "QUSDT"], day=["2026-10-12", "2026-10-12", "2026-10-20"]))
        g, p = long_group(fl, csv, vols, _ms("2026-10-21"))
        chk(list(g) == ["rest", "zone", "unscored"] and list(p) == [False, False, True], "LONG groups + pending unknown")
        chk(not long_group(fl, csv, vols, _ms("2026-10-29"))[1][2], "unknown stops holding 7 days after its day")

        # ── verdict paths
        def coh(zp, rp, nz=20, days=8, nr=20, pairs=None):
            dz = [f"2026-10-{14 + i % days:02d}" for i in range(nz)]
            dr = [f"2026-10-{14 + i % days:02d}" for i in range(nr)]
            z = pd.DataFrame(dict(pair=pairs or [f"P{i}" for i in range(nz)], day=dz, pct=zp, grp="zone", usd=0.0,
                                  ts=pd.to_datetime(dz) + pd.to_timedelta(np.arange(nz) * 2 + 1, "min")))
            r = pd.DataFrame(dict(pair=[f"R{i}" for i in range(nr)], day=dr, pct=rp, grp="rest", usd=0.0,
                                  ts=pd.to_datetime(dr) + pd.to_timedelta(np.arange(nr) * 2, "min")))
            return pd.concat([z, r], ignore_index=True)

        def v(c, be=50.0):
            return verdict(c[c.grp == "zone"], c[c.grp == "rest"], be)[0]
        zl = [-3.0] * 14 + [3.0] * 6                                  # WR 30 %, avg −1.2
        chk(v(coh(zl, [0.5] * 20)) == CAND, "zone loses, WR < breakeven, P high, spread → candidate")
        chk(v(coh([0.6] * 20, [0.5] * 20)) == RETIRE, "zone avg ≥ rest avg → retire")
        chk(v(coh(zl, [-2.0] * 20)) == RETIRE, "zone better than a worse rest → retire (checked first)")
        chk(v(coh(zl, [0.5] * 20), be=25.0) == KEEP, "WR above breakeven → keep observing")
        chk(v(coh(zl, [0.5] * 20, pairs=["P0"] * 20)) == KEEP, "one pair carries the loss → keep observing")
        chk(v(coh([-0.1, 0.1] * 10, [0.5] * 20)) == KEEP, "P(avg<0) < 0.95 → keep observing")
        chk(v(coh(zl[:19], [0.5] * 20, nz=19)) == "COLLECTING" and v(coh(zl, [0.5] * 20, days=7)) == "COLLECTING", "N < 20 or < 8 days")
        chk(verdict(coh(zl, [0.5] * 20).query("grp=='zone'"), coh(zl, [0.5] * 20).iloc[:0], None)[0] == KEEP, "no breakeven → never candidate")
        # ── freeze once + hold
        c0 = coh(zl, [0.5] * 20)
        st, ch = freeze({}, "LONG_NONTOP50", c0, 66.67, "now", {})
        chk(ch and st["LONG_NONTOP50"]["state"] == CAND and st["LONG_NONTOP50"]["n"] == 20, "frozen at the crossing")
        st2, ch2 = freeze(st, "LONG_NONTOP50", coh([0.6] * 25, [0.5] * 20, nz=25), 66.67, "x", {})
        chk(not ch2 and st2["LONG_NONTOP50"]["state"] == CAND, "decided once, never re-fit")
        w = []
        chk(not freeze({}, "LITE_BTC1H_FALLING", c0, 50.0, "x", {}, [(pd.Timestamp("2026-10-14"), "OPEN FRENZY_LITE X")], w)[1]
            and w and w[0].startswith("LITE_BTC1H_FALLING"), "an open fill ≤ the prefix end holds the freeze")
        chk(freeze({}, "LITE_BTC1H_FALLING", c0, 50.0, "x", {}, [(pd.Timestamp("2026-11-30"), "OPEN FRENZY_LITE X")])[1],
            "a later open fill does not hold")
        # ── end to end: loader dedup, floor, MANUAL out, unscored, state file
        rows = []
        for i in range(24):
            d = f"2026-10-{14 + i % 8:02d}T{10 + i // 8:02d}:00:0{i % 3}"
            rows.append(dict(opened_at=d, closed_at=d, pair=f"L{i}USDT", direction="LONG", entry_strategy="FRENZY_LITE", status="CLOSED",
                             pnl_percentage=-3.0 if i % 6 else 3.0, pnl=-60.0, close_reason="STOP_LOSS", entry_btc_1h_slope=-0.1))
            rows.append(dict(opened_at=d, closed_at=d, pair=f"K{i}USDT", direction="LONG", entry_strategy="FRENZY_LITE", status="CLOSED",
                             pnl_percentage=3.0, pnl=60.0, close_reason="FRENZY_TP", entry_btc_1h_slope=0.2))
        rows.append(dict(rows[0], status="OPEN", pnl_percentage=None, closed_at=None))                       # CLOSED beats OPEN
        rows.append(dict(rows[0], opened_at="2026-10-14T09:00:00", entry_btc_1h_slope=None))                # unscored
        rows.append(dict(rows[0], opened_at="2026-10-14T09:30:00", close_reason="MANUAL_CLOSE"))            # manual out
        rows.append(dict(rows[0], opened_at="2026-10-10T09:30:00"))                                         # before the floor
        rows.append(dict(rows[0], opened_at="2026-10-15T09:00:00", pair="ZUSDT", entry_strategy="FRENZY_LONG", pnl_percentage=-3.0))
        od = pd.DataFrame(rows)
        sp = os.path.join(td, "state.json")
        th = dict(frenzy_tp_pct=1.5, frenzy_lite_tp_pct=3.0, frenzy_stop_pct=3.0, frenzy_max_hold_minutes=720)
        L = run(now_ms=_ms("2026-10-23"), orders=od, state_path=sp, open_ts=[], deploy="2026-10-13 12:10", th=th, csv_sets=csv,
                vol_path=os.path.join(td, "v4.json"), fetch=False)
        txt = "\n".join(L)
        stt = json.load(open(sp))
        chk("LITE_BTC1H_FALLING" in stt and stt["LITE_BTC1H_FALLING"]["state"] == CAND and stt["LITE_BTC1H_FALLING"]["be"] == 50.0,
            f"end to end: LITE frozen candidate at breakeven 50 % ({stt.get('LITE_BTC1H_FALLING')})")
        chk("LONG_NONTOP50" not in stt and "| UNSCORED (top-50 membership unknown) | 1 |" in txt, "LONG unknown membership listed apart")
        chk("| UNSCORED (no slope stamp) | 1 |" in txt and "manually closed" in txt and "66.7 %" in txt, "unscored / manual / LONG breakeven shown")
        chk("not cached (fetch off)" in txt, "fetch off noted")
        L2 = run(now_ms=_ms("2026-10-24"), orders=od.assign(pnl_percentage=3.0), state_path=sp, open_ts=[], deploy="2026-10-13 12:10", th=th,
                 csv_sets=csv, vol_path=os.path.join(td, "v4.json"), fetch=False)
        chk(json.load(open(sp))["LITE_BTC1H_FALLING"]["state"] == CAND and "Live (information only" in "\n".join(L2), "frozen survives a rerun")
    finally:
        socket.socket = saved_sock
        globals()["EXPORT_GLOB"] = saved_glob
        NF.SHARED_COOLDOWN, NF._cooldown_paths = saved_cd, saved_cp
        import shutil
        shutil.rmtree(td, ignore_errors=True)
    print(f"selftest FRENZY WL lines OK ({ok} checks)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        try:
            import ccxt
            _EX = ccxt.binanceusdm({"enableRateLimit": True})
        except Exception:
            _EX = None
        print("\n".join(run(EX=_EX)))
