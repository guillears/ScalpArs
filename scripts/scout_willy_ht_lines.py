#!/usr/bin/env python3
"""👁 Scout — three WILLY / WILLY_HT watch lines (registered 2026-11-18, DECISION_LOG 580; OBSERVE ONLY — never changes config, no bot API).

WHY        operator 11-18: "Build the 3 WILLY / HT observe lines" after reports/HT_TP6_WL_2026-11-18.md (no filter passes: best 1D/2D luck
           p 0.08, Kelly floor ≤ 0) and reports/WILLY_BACKTEST_AUDIT_2026-11-18.md (the rebuild matches the real fills; HT re-anchored
           entries −0.39 over 114 trades). Each line is frozen here, read on REAL forward fills only, decided ONCE, never re-fit.
LINES      (zone = the entries the line would block; rest = the scored others; a missing input → UNSCORED, never counted)
  CROWDED_LONGS    top-trader ACCOUNT long/short ratio > 1.19 OR global ACCOUNT long/short ratio > 1.0624 at entry (crowded longs —
                   theory: late buyers are already in; reports/HT_TP6_WL: rebuild kept 210 · +0.39 vs blocked 207 · −0.88, holdout agrees,
                   real fills DISAGREE: the 9 blocked were 89 % winners). Values: Binance /futures/data/topLongShortAccountRatio and
                   /futures/data/globalLongShortAccountRatio, 5m; the API's `timestamp` is the END of the 5m period (Binance docs), so the
                   latest row with timestamp ≤ the fill − 1 min is the last period CLOSED before the fill (1 min publication guard).
                   The API keeps ~30 days → older fills are UNSCORED; an empty answer is never cached (retried next run); a run whose
                   fetches failed never freezes a CROWDED_LONGS verdict; ≤ 300 calls per run (weight 0, 1,000 / 5 min IP cap).
                   Cached in reports/.willy_ht_lines_cache.json (gitignored).
  NEAR_SPIKE_HIGH  off_peak = (entry_price ÷ entry_frenzy_willy_spike_high − 1) × 100 > −1.4 (bought within 1.4 % of the spike high —
                   theory: no pull-back to buy; WL_DEEP_WILLY_HT top-tercile cut). Measured on the REAL fill (the rebuild's off_peak differs
                   by up to ~1 pt on the same trade — this line uses the fill only).
  REANCHORED       entry A (entry_frenzy_willy_trigger == "A") > 1 h after the spike (entry_frenzy_hours > 1.0 — measured signal-bar
                   OPEN − spike-bar OPEN, services/frenzy.py, ≈ 5 min more than "since the spike close"): entry A fired on a spike
                   that re-anchored an episode already running (audit: HT 114 · −0.39; today it fires only when the pair is warm, restarts
                   suppress it). Non-A entries → UNSCORED.
COHORT     CLOSED FRENZY_WILLY and FRENZY_WILLY_HT LONG fills from the orders exports (scout_willy_failed_spike_watch.load: dedup, CLOSED beats
           OPEN, newest export wins), operator-closed out, opened ≥ the DECISION_LOG 580 commit + 10 min (forward only). Each sleeve × line
           read and decided SEPARATELY (never pooled).
FROZEN BAR the WILLY_FAILED_SPIKE / WILLY_GAP bar (DL 414 / 509), per sleeve: at the first prefix where a line's ZONE reaches N ≥ 15 on ≥ 8 UTC
           days, decided ONCE — zone avg ≥ rest avg or ≥ 0 → RETIRED · FILTER CANDIDATE (operator decides; never automatic) iff zone WR <
           the sleeve's breakeven (WILLY 69.0 % · HT 66.7 % → 44.4 % re-based at DL 587, TP +5 / SL −4; HT cohort from the 587 commit) ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ 0.95 ∧ no day / pair ≥ 50 % of
           the zone's gross loss · else KEEP OBSERVING. Booked P&L → pinned to the WILLY / HT exit keys in observe_registry.
STATE      reports/SCOUT_WILLY_HT_LINES.json (gitignored).
Usage: venv/bin/python -I scripts/scout_willy_ht_lines.py"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402,F401  (Downloads + exports_archive/, DL 409 — used through WF.load)
import scout_gatekit as _GK        # noqa: E402
import scout_b1h_negflank as NF    # noqa: E402
import scout_willy_failed_spike_watch as WF   # noqa: E402  (the loader + the frozen bar — one definition)

DL = 580
DL_HT = 587   # 🔁 (DECISION_LOG 587) RE-BASED 2026-11-18: HT's exit changed (TP +2 → +5 net, stop −4) → HT counts only fills ≥ the DL 587 commit + 10 min
SLEEVES = ("FRENZY_WILLY", "FRENZY_WILLY_HT")
# FROZEN breakevens: WILLY V2b (DL 509) · HT re-based at DL 587 = the +5 / −4 exit: |loss| ÷ (win + |loss|) = 4 ÷ 9 (was 0.667 at +2 / −4)
BE_WR = {"FRENZY_WILLY": 0.690, "FRENZY_WILLY_HT": 0.444}
STATE_KEY = {"FRENZY_WILLY": "FRENZY_WILLY", "FRENZY_WILLY_HT": "FRENZY_WILLY_HT@587"}   # the HT verdicts of the +2 era are never read again
TOP_ACC_CUT, GLOB_ACC_CUT, OFF_PEAK_CUT, REANCHOR_H = 1.19, 1.0624, -1.4, 1.0
LINES = ("CROWDED_LONGS", "NEAR_SPIKE_HIGH", "REANCHORED")
N_MIN, DAYS_MIN, P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED = 15, 8, 0.95, 0.50, 4000, 7
LAG_MS = 60_000
MAX_CALLS = 300
API_DAYS = 29
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_HT_LINES.json")
CACHE = os.path.join(_ROOT, "reports", ".willy_ht_lines_cache.json")
EXTRA = ("entry_price", "entry_frenzy_willy_spike_high", "entry_frenzy_hours", "entry_frenzy_willy_trigger")
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
UA = "Mozilla/5.0"


class NetStop(Exception):
    """418 / 429 — stop fetching for this run."""


# ── the three zones (pure) ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def zone_crowded(top_acc, glob_acc):
    t, g = _f(top_acc), _f(glob_acc)
    if t is None and g is None:
        return "unscored"
    if (t is not None and t > TOP_ACC_CUT) or (g is not None and g > GLOB_ACC_CUT):
        return "zone"
    if t is None or g is None:
        return "unscored"            # one side missing and the other not crowded → cannot say "rest"
    return "rest"


def off_peak(entry_price, spike_high):
    e, h = _f(entry_price), _f(spike_high)
    if e is None or h is None or e <= 0 or h <= 0:
        return None
    return (e / h - 1.0) * 100.0


def zone_near_high(entry_price, spike_high):
    op = off_peak(entry_price, spike_high)
    if op is None:
        return "unscored"
    return "zone" if op > OFF_PEAK_CUT else "rest"


def zone_reanchored(trigger, hours):
    if str(trigger or "").strip().upper() != "A":
        return "unscored"
    h = _f(hours)
    if h is None:
        return "unscored"
    return "zone" if h > REANCHOR_H else "rest"


# ── the long/short ratios (network, lag-safe) ───────────────────────────────────────────────────────────────────────────────────────

_last = [0.0]
_calls = [0]


def _get(path, params):
    if _calls[0] >= MAX_CALLS:
        raise NetStop(f"per-run call cap {MAX_CALLS}")
    _calls[0] += 1
    wait = 0.6 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    url = "https://fapi.binance.com" + path + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
            _last[0] = time.time()
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        _last[0] = time.time()
        if e.code in (418, 429):
            raise NetStop(f"HTTP {e.code}")
        raise


def pick_ratio(rows, entry_ms):
    """the latest row with timestamp (period END) ≤ entry − 1 min → its longShortRatio (None if none). Pure."""
    best = None
    for r in rows or []:
        t, v = _f(r.get("timestamp")), _f(r.get("longShortRatio"))
        if t is None or v is None or t > entry_ms - LAG_MS:
            continue
        if best is None or t > best[0]:
            best = (t, v)
    return best[1] if best else None


def ratios(pair, entry_ms, cache, now_ms):
    """→ (top_acc, glob_acc) or (None, None). Cached per fill; fills older than the API window → (None, None)."""
    key = f"{pair}|{int(entry_ms)}"
    if key in cache:
        v = cache[key]
        return (v[0], v[1]) if isinstance(v, list) else (None, None)
    if now_ms - entry_ms > API_DAYS * 86_400_000:
        cache[key] = "OLD"
        return None, None
    q = dict(symbol=pair, period="5m", startTime=int(entry_ms - 40 * 60_000), endTime=int(entry_ms), limit=30)
    top = pick_ratio(_get("/futures/data/topLongShortAccountRatio", q), entry_ms)
    glob = pick_ratio(_get("/futures/data/globalLongShortAccountRatio", q), entry_ms)
    if top is not None and glob is not None:
        cache[key] = [top, glob]                                  # only a complete answer is cached; partial / empty → retried next run
    else:
        raise ValueError(f"incomplete ratio answer for {pair}")   # → this run's CROWDED_LONGS verdicts are not frozen (score() stop)
    return top, glob


def _load_cache():
    try:
        with open(CACHE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(c):
    try:
        with open(CACHE, "w") as f:
            json.dump(c, f)
    except Exception:
        pass


# ── read ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def load(paths=None):
    """the WILLY loader (one definition) + the columns these lines need."""
    return WF.load(paths, extra=EXTRA)


def score(c, now_ms, fetch=True):
    """add zone columns for the three lines. fetch False → CROWDED_LONGS from the cache only."""
    cache = _load_cache()
    _calls[0] = 0
    stop = None
    failed = []
    tops, globs = [], []
    for r in c.itertuples():
        t = g = None
        v = cache.get(f"{r.pair}|{int(r.o.value // 1_000_000)}")
        if isinstance(v, list):
            t, g = v[0], v[1]                                     # cached → always read, also after a stop
        elif fetch and stop is None:
            try:
                t, g = ratios(r.pair, int(r.o.value // 1_000_000), cache, now_ms)
            except NetStop as e:
                stop = str(e)
            except (urllib.error.URLError, OSError, ValueError) as e:
                failed.append(f"{type(e).__name__} on {r.pair}")     # this fill stays UNSCORED this run; the others are still fetched
        tops.append(t)
        globs.append(g)
    _save_cache(cache)
    if failed and not stop:
        stop = f"{len(failed)} fill(s) not readable ({failed[0]}{' …' if len(failed) > 1 else ''})"   # any gap → no CROWDED_LONGS freeze this run
    c = c.copy()
    c["top_acc"], c["glob_acc"] = tops, globs
    c["z_CROWDED_LONGS"] = [zone_crowded(a, b) for a, b in zip(tops, globs)]
    c["z_NEAR_SPIKE_HIGH"] = [zone_near_high(a, b) for a, b in zip(c.get("entry_price", [None] * len(c)), c.get("entry_frenzy_willy_spike_high", [None] * len(c)))]
    c["z_REANCHORED"] = [zone_reanchored(a, b) for a, b in zip(c.get("entry_frenzy_willy_trigger", [None] * len(c)), c.get("entry_frenzy_hours", [None] * len(c)))]
    return c, stop


def verdict(zone, rest, be):
    """the frozen bar (WF.verdict with the sleeve's own breakeven). PURE."""
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz, mr = float(zone.pct.mean()), (float(rest.pct.mean()) if len(rest) else float("nan"))
    wr = float((zone.pct > 0).mean())
    p = NF.p_mean_neg(zone.pct.values, zone.day.values, BOOT_N, BOOT_SEED)
    sd, sp = NF.gross_loss_share(zone.pct.values, zone.day.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {wr:.0%} (breakeven {be:.1%}) · avg {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · "
           f"P(mean<0) {(p if p is not None else float('nan')):.2f} · top day {sd:.0%} / top pair {sp:.0%} of the zone's gross loss")
    if mz >= 0 or (len(rest) and mz >= mr):
        return "RETIRED", det
    if wr < be and mz < 0 and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def crossing_prefix(sc, col):
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(sc[col].values, sc.day.values), 1):
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


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, fetch=True):
    """→ markdown lines. since_ms None → the DECISION_LOG 580 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 WILLY / WILLY_HT lines — CROWDED_LONGS · NEAR_SPIKE_HIGH · REANCHORED (registered 2026-11-18, DECISION_LOG {DL}, OBSERVE only)", ""]
    given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="WILLY / HT lines")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    keep, stop = score(keep, now_ms, fetch) if len(keep) else (keep, None)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    changed, cap = False, WF.cap_minutes()
    L += [f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if given is None else 'floor given'}) · "
          f"each sleeve × line decided separately · {_GK.operator_closed_line(ox)}.", ""]
    if stop:
        L += [f"⚠ ratio fetching stopped this run ({stop}) — later fills stay UNSCORED for CROWDED_LONGS and no CROWDED_LONGS verdict is frozen this run.", ""]
    ht_floor = (_GK.dl_deploy_ms(DL_HT, lag_min=10, label="WILLY_HT lines (re-based, DL 587)") if given is None else since_ms)
    for es in SLEEVES:
        s = keep[keep.entry_strategy.astype(str) == es] if len(keep) else keep
        L += [f"### {es}", ""]
        if es == "FRENZY_WILLY_HT":   # 🔁 (587) re-based on the +5 / −4 exit
            if ht_floor is None:
                L += [f"⚠ HT re-based on the DECISION_LOG {DL_HT} exit (TP +5 / SL −4) — NOT EVALUABLE until that commit is in git.", ""]
                continue
            s = s[s.o >= pd.Timestamp(int(ht_floor), unit="ms")] if len(s) else s
            L += [f"(re-based: HT fills opened ≥ {pd.Timestamp(int(ht_floor), unit='ms'):%Y-%m-%d %H:%M} UTC — the DL {DL_HT} TP +5 / SL −4 exit; breakeven 44.4 %)", ""]
        elif es == "FRENZY_WILLY":
            L += ["(plain WILLY entries are OFF since DECISION_LOG 587 — this sleeve collects no new fills while they stay off)", ""]
        for ln in LINES:
            col = f"z_{ln}"
            if not len(s) or col not in s:
                L += [f"**{ln}** — no counted fills yet.", ""]
                continue
            sc = s[s[col] != "unscored"]
            zone, rest, uns = sc[sc[col] == "zone"], sc[sc[col] == "rest"], s[s[col] == "unscored"]
            L += [f"**{ln}**", "", "| Group | N | days | WR | avg % |", "|---|---|---|---|---|",
                  _row("zone (would block)", zone), _row("rest", rest), _row("UNSCORED — never counted", uns), ""]
            key = f"{STATE_KEY[es]}|{ln}"
            rec = st.get(key) if ok else None
            if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
                rec, ok = None, False
            if not rec and ok and not (ln == "CROWDED_LONGS" and stop):
                pre = crossing_prefix(sc, col)
                if pre is not None:
                    last = pre.o.max()
                    if newest is not None and newest >= last + pd.Timedelta(minutes=cap):
                        state, det = verdict(pre[pre[col] == "zone"], pre[pre[col] == "rest"], BE_WR[es])
                        rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC",
                                   run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
                        if state_path:
                            st[key] = rec
                            changed = True
            live_state, live_det = verdict(zone, rest, BE_WR[es])
            if rec and ok and state_path:
                L += [f"Frozen verdict (crossing {rec['at']}, frozen {rec['run_at']}): **{rec['state']}** ({rec['detail']})",
                      f"Live (information only): {live_state} — {live_det}", ""]
            else:
                L += [f"Now: ⏳ {live_state} ({live_det}).", ""]
    if changed and ok and state_path:
        NF.du_save_state(st, state_path)
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += [f"FROZEN cuts: CROWDED_LONGS top-trader account L/S > {TOP_ACC_CUT} or global account L/S > {GLOB_ACC_CUT} (5m period ended ≤ fill − 1 min) · "
          f"NEAR_SPIKE_HIGH off_peak > {OFF_PEAK_CUT} % on the real fill · REANCHORED entry A > {REANCHOR_H:g} h after the spike. "
          f"Bar: zone N ≥ {N_MIN} on ≥ {DAYS_MIN} days, decided once (RETIRED / {CANDIDATE} / KEEP OBSERVING). In-sample evidence at "
          "registration (never counted): reports/HT_TP6_WL_2026-11-18.md, reports/WILLY_BACKTEST_AUDIT_2026-11-18.md.", ""]
    return L


def main():
    print("\n".join(run()))


if __name__ == "__main__":
    main()
