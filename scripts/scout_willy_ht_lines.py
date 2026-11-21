#!/usr/bin/env python3
"""👁 Scout — WILLY / WILLY_HT watch lines (three registered 2026-11-18, DECISION_LOG 580 · two HT lines 2026-11-19, DL 596; OBSERVE ONLY — never changes config, no bot API).

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
                   ⚠ ARMED for WILLY_HT since DECISION_LOG 594 (2026-11-19, operator-declared; frenzy_willy_ht_min_dip_pct 1.4 — the engine
                   judges the DECISION price vs the same H, the fill sits a few bps away) → forward HT zone fills should be ~0 and the
                   HT row here mostly shows rest-side fills; the filter's forward verdict is the blocked-trade pricer (scout_blocked_filters.py,
                   FRENZY_WILLY_HT · WILLY_HT_NEAR_HIGH, absolute gate: first 15 priced on ≥ 8 days, avg ≥ 0 → REVERT).
  REANCHORED       entry A (entry_frenzy_willy_trigger == "A") > 1 h after the spike (entry_frenzy_hours > 1.0 — measured signal-bar
                   OPEN − spike-bar OPEN, services/frenzy.py, ≈ 5 min more than "since the spike close"): entry A fired on a spike
                   that re-anchored an episode already running (audit: HT 114 · −0.39; today it fires only when the pair is warm, restarts
                   suppress it). Non-A entries → UNSCORED.
  PRE_VOL_SURGE    (WILLY_HT only · registered 2026-11-19, DECISION_LOG 596) entry_frenzy_pre_vol15_vs60 > 1.60 — the 15 minutes before the
                   fill traded > 1.6× the hour before (theory: buying into a fresh volume burst = buying the chase; reports/WILLY_HT_WL594_2026-11-19:
                   master 22 · 32 % · −1.21 vs 22 · 77 % · +2.93, family-wise luck p 0.33 — NOT a filter; the backtest has no volume → unchecked).
                   The stamp is written by the post-fill backfill (services.frenzy.pre_vol15_vs60, 1m base volume); NULL → UNSCORED.
  BTC5M_UP         (WILLY_HT only · DL 596) entry_btc_trend_gap_pct > 0 — BTC 5m EMA13 above EMA50 at entry (master 20 · 30 % · −1.37 vs
                   24 · 75 % · +2.73; the BACKTEST FLIPPED it in both periods → a REFUTE line: it is expected to retire). MARKET-WIDE → counted in
                   WINDOW units: its zone needs ≥ 15 distinct UTC days (LINE_DAYS_MIN), not just 15 fills.
                   Both 596 lines count only HT fills opened ≥ the DECISION_LOG 596 commit + 10 min (the NEAR_HIGH-filtered era, forward only).
COHORT     CLOSED FRENZY_WILLY and FRENZY_WILLY_HT LONG fills from the orders exports (scout_willy_failed_spike_watch.load: dedup, CLOSED beats
           OPEN, newest export wins), operator-closed out, opened ≥ the DECISION_LOG 580 commit + 10 min (forward only). Each sleeve × line
           read and decided SEPARATELY (never pooled).
FROZEN BAR the WILLY_FAILED_SPIKE / WILLY_GAP bar (DL 414 / 509), per sleeve: at the first prefix where a line's ZONE reaches N ≥ 15 on ≥ 8 UTC
           days, decided ONCE — zone avg ≥ rest avg or ≥ 0 → RETIRED · FILTER CANDIDATE (operator decides; never automatic) iff zone WR <
           the sleeve's breakeven (WILLY 69.0 % · HT 66.7 % → 44.4 % at DL 587 → 40.0 % re-based at DL 608, TP +6 / SL −4; HT cohort from the 608 commit) ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ 0.95 ∧ no day / pair ≥ 50 % of
           the zone's gross loss · else KEEP OBSERVING. ⚠ HT's 40.0 % is frozen from THEORY (+6 / −4 net = 4 ÷ 10 — no stop slippage, no
           24 h time exits), WILLY's 69.0 % from real fills (V2b, N 50). Booked P&L → pinned to the WILLY / HT exit keys in observe_registry.
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
DL_ARMED_NEAR_HIGH = 594   # 🧱 (DECISION_LOG 594) NEAR_SPIKE_HIGH ARMED for WILLY_HT (the engine refuses the zone; the blocked-trade pricer decides)
DL_596 = 596   # 📊 (DECISION_LOG 596) PRE_VOL_SURGE + BTC5M_UP registered 2026-11-19 — HT only, forward from that commit
DL_HT = 608   # 🔁 (DECISION_LOG 608) RE-BASED 2026-11-21: HT's TP +5 → +6 net (stop −4; was 587: +2 → +5) → HT counts only fills ≥ the DL 608 commit + 10 min
SLEEVES = ("FRENZY_WILLY", "FRENZY_WILLY_HT")
# FROZEN breakevens: WILLY V2b (DL 509) · HT re-based at DL 608 = the +6 / −4 exit: |loss| ÷ (win + |loss|) = 4 ÷ 10 (0.444 at +5 / −4, 0.667 at +2 / −4)
BE_WR = {"FRENZY_WILLY": 0.690, "FRENZY_WILLY_HT": 0.400}
BE_NOTE = WF.BE_NOTE   # HT's 40.0 % is frozen from theory (+6 / −4 net), WILLY's 69.0 % from real fills
STATE_KEY = {"FRENZY_WILLY": "FRENZY_WILLY", "FRENZY_WILLY_HT": "FRENZY_WILLY_HT@608"}   # the HT verdicts of the +2 / +5 eras are never read again
TOP_ACC_CUT, GLOB_ACC_CUT, OFF_PEAK_CUT, REANCHOR_H = 1.19, 1.0624, -1.4, 1.0
PRE_VOL_CUT, BTC_GAP_CUT = 1.60, 0.0   # 📊 (596) FROZEN — the WL594 cuts (never re-fit)
LINES = ("CROWDED_LONGS", "NEAR_SPIKE_HIGH", "REANCHORED", "PRE_VOL_SURGE", "BTC5M_UP")
LINES_596 = ("PRE_VOL_SURGE", "BTC5M_UP")   # HT only, floor = the DL 596 commit + 10 min
LINE_DAYS_MIN = {"BTC5M_UP": 15}   # 📊 (596 review) market-wide → WINDOW units: the zone needs ≥ 15 distinct UTC days (not 15 fills)
N_MIN, DAYS_MIN, P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED = 15, 8, 0.95, 0.50, 4000, 7
LAG_MS = 60_000
MAX_CALLS = 300
API_DAYS = 29
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_HT_LINES.json")
CACHE = os.path.join(_ROOT, "reports", ".willy_ht_lines_cache.json")
EXTRA = ("entry_price", "entry_frenzy_willy_spike_high", "entry_frenzy_hours", "entry_frenzy_willy_trigger",
         "entry_frenzy_pre_vol15_vs60", "entry_btc_trend_gap_pct")   # 📊 (596)
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


def zone_above(v, cut):
    """📊 (596) value > cut → zone · ≤ cut → rest · missing → unscored. Pure."""
    x = _f(v)
    if x is None:
        return "unscored"
    return "zone" if x > cut else "rest"


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
    """add zone columns for every line. fetch False → CROWDED_LONGS from the cache only."""
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
    c["z_PRE_VOL_SURGE"] = [zone_above(v, PRE_VOL_CUT) for v in c.get("entry_frenzy_pre_vol15_vs60", [None] * len(c))]   # 📊 (596)
    c["z_BTC5M_UP"] = [zone_above(v, BTC_GAP_CUT) for v in c.get("entry_btc_trend_gap_pct", [None] * len(c))]
    return c, stop


def verdict(zone, rest, be, days_min=DAYS_MIN):
    """the frozen bar (WF.verdict with the sleeve's own breakeven). days_min = the line's day floor (BTC5M_UP: 15, window units). PURE."""
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < days_min:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{days_min} days"
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


def crossing_prefix(sc, col, days_min=DAYS_MIN):
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(sc[col].values, sc.day.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(d)
        if nz >= N_MIN and len(seen) >= days_min:
            return sc.iloc[:k]
    return None


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – |"
    return f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % |"


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, fetch=True):
    """→ markdown lines. since_ms None → the DECISION_LOG 580 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 WILLY / WILLY_HT lines — CROWDED_LONGS · NEAR_SPIKE_HIGH · REANCHORED (registered 2026-11-18, DECISION_LOG {DL}) · "
         f"HT PRE_VOL_SURGE · BTC5M_UP (2026-11-19, DECISION_LOG {DL_596}) — OBSERVE only", ""]
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
    changed = False
    L += [f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if given is None else 'floor given'}) · "
          f"each sleeve × line decided separately · {_GK.operator_closed_line(ox)}.", ""]
    if stop:
        L += [f"⚠ ratio fetching stopped this run ({stop}) — later fills stay UNSCORED for CROWDED_LONGS and no CROWDED_LONGS verdict is frozen this run.", ""]
    ht_floor = (_GK.dl_deploy_ms(DL_HT, lag_min=10, label="WILLY_HT lines (re-based, DL 608)") if given is None else since_ms)
    floor_596 = (_GK.dl_deploy_ms(DL_596, lag_min=10, label="WILLY_HT PRE_VOL_SURGE / BTC5M_UP (DL 596)") if given is None else since_ms)
    for es in SLEEVES:
        s = keep[keep.entry_strategy.astype(str) == es] if len(keep) else keep
        L += [f"### {es}", ""]
        if es == "FRENZY_WILLY_HT":   # 🔁 (608) re-based on the +6 / −4 exit
            if ht_floor is None:
                L += [f"⚠ HT re-based on the DECISION_LOG {DL_HT} exit (TP +6 / SL −4) — NOT EVALUABLE until that commit is in git.", ""]
                continue
            s = s[s.o >= pd.Timestamp(int(ht_floor), unit="ms")] if len(s) else s
            L += [f"(re-based: HT fills opened ≥ {pd.Timestamp(int(ht_floor), unit='ms'):%Y-%m-%d %H:%M} UTC — the DL {DL_HT} TP +6 / SL −4 exit; breakeven 40.0 %)", f"⚠ {BE_NOTE}.",
                  f"⚠ NEAR_SPIKE_HIGH is ARMED for WILLY_HT since DECISION_LOG {DL_ARMED_NEAR_HIGH} (frenzy_willy_ht_min_dip_pct) — zone entries are REFUSED, "
                  "so forward HT zone fills should be ~0 and its table mostly shows rest-side fills. The filter's forward verdict is the blocked-trade "
                  "pricer (scout_blocked_filters: FRENZY_WILLY_HT · WILLY_HT_NEAR_HIGH — the refused zone trades priced on HT's live exit).", ""]
        elif es == "FRENZY_WILLY":
            L += ["(plain WILLY entries back ON since DECISION_LOG 595 (2026-11-19) — OFF 11-18 → 11-19 (DL 587), no fills in that gap)", ""]
        for ln in LINES:
            col = f"z_{ln}"
            s_ln = s
            if ln in LINES_596:   # 📊 (596) HT only, forward from the DL 596 commit
                if es != "FRENZY_WILLY_HT":
                    continue
                if floor_596 is None:
                    L += [f"**{ln}** — ⚠ NOT EVALUABLE until the DECISION_LOG {DL_596} commit is in git.", ""]
                    continue
                s_ln = s[s.o >= pd.Timestamp(int(floor_596), unit="ms")] if len(s) else s
            if not len(s_ln) or col not in s_ln:
                L += [f"**{ln}** — no counted fills yet.", ""]
                continue
            sc = s_ln[s_ln[col] != "unscored"]
            zone, rest, uns = sc[sc[col] == "zone"], sc[sc[col] == "rest"], s_ln[s_ln[col] == "unscored"]
            L += [f"**{ln}**", "", "| Group | N | days | WR | avg % |", "|---|---|---|---|---|",
                  _row("zone (would block)", zone), _row("rest", rest), _row("UNSCORED — never counted", uns), ""]
            key = f"{STATE_KEY[es]}|{ln}"
            rec = st.get(key) if ok else None
            if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
                rec, ok = None, False
            if not rec and ok and not (ln == "CROWDED_LONGS" and stop):
                dmin = LINE_DAYS_MIN.get(ln, DAYS_MIN)
                pre = crossing_prefix(sc, col, dmin)
                if pre is not None:
                    last = pre.o.max()
                    if newest is not None and newest >= last + pd.Timedelta(minutes=WF.cap_minutes(es=es)):   # 🔁 (587) HT: its own 24 h cap
                        state, det = verdict(pre[pre[col] == "zone"], pre[pre[col] == "rest"], BE_WR[es], dmin)
                        rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC",
                                   run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
                        if state_path:
                            st[key] = rec
                            changed = True
            live_state, live_det = verdict(zone, rest, BE_WR[es], LINE_DAYS_MIN.get(ln, DAYS_MIN))
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
          f"NEAR_SPIKE_HIGH off_peak > {OFF_PEAK_CUT} % on the real fill · REANCHORED entry A > {REANCHOR_H:g} h after the spike · "
          f"PRE_VOL_SURGE pre_vol15_vs60 > {PRE_VOL_CUT:g} · BTC5M_UP BTC 5m EMA13−EMA50 gap > {BTC_GAP_CUT:g} (both HT-only, from the DL {DL_596} commit; "
          "BTC5M_UP is market-wide and already refuted on the backtest — a refute line). "
          f"Bar: zone N ≥ {N_MIN} on ≥ {DAYS_MIN} days (BTC5M_UP: ≥ {LINE_DAYS_MIN['BTC5M_UP']} days — window units), decided once (RETIRED / {CANDIDATE} / KEEP OBSERVING). In-sample evidence at "
          "registration (never counted): reports/HT_TP6_WL_2026-11-18.md, reports/WILLY_BACKTEST_AUDIT_2026-11-18.md.", ""]
    return L


def main():
    print("\n".join(run()))


if __name__ == "__main__":
    main()
