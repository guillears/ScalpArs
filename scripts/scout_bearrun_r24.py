#!/usr/bin/env python3
"""🐻⏬ Scout — BEARRUN_R24_25 shadow (DECISION_LOG 369, operator 2026-10-24: "I think we should change bear to −2.5 %"). OBSERVE ONLY —
never changes config, never calls the bot API, never reads server logs. Public Binance fapi only.

WHY             Bear-Run turns ON at BTC 24 h return ≤ −bearrun_r24_on (4.0, stays ON while ≤ −bearrun_r24_off 3.0) with the other monitor
                legs. Earlier dd24 / r12 / r6 triggers were tested and ADD losing shorts (reports/BEARRUN_TRIGGER_TIMING_2026-10-23_FINAL.md:
                T1a / T1b / T1c Δ Σ −11.7 / −10.3 / −3.8 % vs T0, worse than the shuffled-timing null), but r24 itself at −2.5 % was never
                tested. Operator chose a tracking line, no config change.
DEFINITIONS     the 5m monitor replica of scripts/study_bearrun_timing_triggers.py (compute_readings: r24 / below24 / eff24 / r6 / 1 h EMA50
                on CLOSED BTCUSDT 5m bars) + the REAL engine state machine services.trading_engine._bearrun_next_state (Schmitt band +
                squeeze latch), every leg at the LIVE trading_config.json value (the engine's own getattr defaults when a key is missing):
                  T0  = today's Bear-Run (r24 ON ≤ −bearrun_r24_on, stay ≤ −bearrun_r24_off)
                  V25 = T0 with the r24 leg at ON ≤ −2.5 / stay ≤ −1.5 (the same 1.0 hysteresis as 4 / 3); every other leg identical.
                V25 ⊇ T0 on every bar while the live r24 legs are stricter (on > 2.5 ∧ off > 1.5) — GUARD: otherwise a warning prints and
                nothing is evaluated or read. The windows that matter are the EXTRA time: V25 ON ∧ T0 OFF. Extra stretches ≤ 180 min apart
                (bearrun_window_merge_minutes) are ONE window when T0 stays OFF in the gap. A window ends at the first reading where V25 is
                OFF or T0 is ON (end reason). Kind: EARLIER = T0 turned ON at the end or ≤ 180 min after it (V25 was simply earlier in the
                same move) · TAIL = it starts right where a T0 stretch ended (V25's lower stay band) · TAIL→EARLIER = both · EXTRA = T0
                never ON in the move. The kind is settled (and persisted) at finality.
                Per-entry refusals (bearrun_btc_off24lo_max, bearrun_breadth_min, blacklist, spacing) are NOT monitor legs → not modelled.
CLASS (24 h)    scripts/scout_bearrun_v2.py classify() on the 5m BTC bars (bars_per_h 12) from the window start: clean_slide /
                false_start (F1, F1_unresolved) / true_other.
BASKET (proxy)  top-20 alt USDT perps by quote volume over the 24 closed 1 h bars before the start (universe = the V2 scout's top-60,
                snapshotted per window at first pricing), SHORT, priced ON THE 5m GRID FROM THE WINDOW START: entry = each coin's close of
                the 5m bar closing at the start (the first closed 5m bar ≥ the start — always inside the window):
                  a' : +0.5 % TP / −1 % stop, first touch on 5m bars (both in one bar → stop), 6 h cap, 0.10 % fees   ← the gate's value
                  d  : hold while V25 is ON — exit at the close of the 5m bar where the window ends (EXTRA / TAIL), or, for EARLIER windows,
                       where V25 turns OFF (the full V25 stretch, labelled 'd to V25 OFF'), +2 % stop on 5m highs, 24 h cap, 0.10 % fees
                PROXIES OF THE SLEEVE, NOT THE ENGINE: no engine gates / signals / slippage, the momentum-short exit (incl. the EMA13
                cross) is not modelled, and earlier entries would take slots / per-pair spacing and can displace T0's own fills.
PARITY          CONTAINMENT of today's live Bear-Run fills (decisions journal OPEN rows with strategy BEARRUN_SHORT + the orders exports'
                BEARRUN_SHORT fills, dedup per pair within ±2 min) in the replica's T0 ON (±15 min); also per live window (fills ≤ 180 min
                apart). A replica T0 window without a fill is not a mismatch (per-entry refusals / no signal).
COHORT          extra windows starting ≥ the DECISION_LOG 369 commit + 10 min (scout_gatekit.dl_deploy_ms). Until the commit exists: NOT
                evaluable, nothing persisted, the last 7 days of extra windows print as context (informational; priced once final).
FINALITY        a cohort window is final when its end is known ∧ 24 h have passed since its start (class known) ∧ 180 min have passed since
                its end (kind settled) ∧ it is fully priced. Only final windows are priced. Persisted APPEND-ONLY (keyed on the start) in
                reports/.scout_bearrun_r24_windows.json, flock-guarded (the V2 scout's _locked). Never silently dropped: a time-final window
                whose basket cannot be priced (every coin 4xx) → an 'unpriceable' record; a seen cohort window that scrolls out of the 30-day
                5m fetch before it is final → a 'lost' record; stretches that would merge into an already-persisted window → a 'lost'
                counter line. Unpriceable / lost records are listed and counted, never in the read.
🔒 FROZEN READ  decided ONCE (persisted), at the shortest prefix of ≥ 20 final priced extra windows spanning ≥ 6 distinct ISO weeks →
                CANDIDATE (operator decides lowering bearrun_r24_on to 2.5 / off 1.5) iff ALL: ① Σ a'-basket window means > 0 ∧
                ② window-bootstrap P(Σ > 0) ≥ 0.90 (4,000, seed 7) ∧ ③ no window ≥ 50 % of Σ ∧ ④ false-start windows lose ≤ 60 % of what
                clean-slide windows earn (max(0, −Σ false) ≤ 0.6 × Σ clean, Σ clean > 0 — ④ FAILS automatically when the prefix holds no
                clean slide); else KEEP_4 (failed conditions listed). A read WAITS while any cohort extra window started by its prefix end
                is not final. INTERIM line at ≥ 8 final windows — printed only, non-terminal, never frozen.
NETWORK         the V2 scout's Net: public fapi only, ≤ 900 used weight (X-MBX-USED-WEIGHT-1M) per run, paced, the shared cooldown
                (scripts/binance_cooldown.py) honoured, 418 / 429 arm it and stop, 403 stops. BTC 5m klines in pages of 1,000 bars
                (weight 5) on fixed boundaries and the alt 5m pricing pages (290 bars, weight 2, one per basket coin per final window) are
                cached when complete under reports/backtest_cache/scout_bearrun_r24/ (with the universe snapshots); the alt 1 h ranking
                klines and the hourly universe list ride the V2 scout's cache (reports/backtest_cache/scout_bearrun_v2/).
Usage           venv/bin/python scripts/scout_bearrun_r24.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import export_glob as EG                                     # noqa: E402
import scout_bearrun_v2 as V2                                # noqa: E402
import scout_gatekit as _GK                                  # noqa: E402
import study_bearrun_timing_triggers as TT                   # noqa: E402

DL = 369
HOUR = V2.HOUR
FIVE = 300_000
PAGE_BARS = 1000                                             # limit 1000 → weight 5
PAGE_MS = PAGE_BARS * FIVE
HIST_DAYS = 30
WARM_MS = 8 * 24 * HOUR                                      # first 8 days of the fetch = warm-up (r24 288 bars; 1 h EMA50 converges ≈ 8 days)
CACHE = os.path.join(_R, "reports", "backtest_cache", "scout_bearrun_r24")
STATE = os.path.join(_R, "reports", ".scout_bearrun_r24_windows.json")
CFG = os.path.join(_R, "trading_config.json")
DEC_PATTERN = os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")

V25_ON, V25_OFF = 2.5, 1.5                                   # 🔒 the shadow's r24 leg (same 1.0 hysteresis as 4 / 3)
MERGE_MIN_DEFAULT = 180.0
PARITY_TOL_MIN = 15
DEDUP_MS = 120_000                                           # fills of one pair within ±2 min = one fill (journal vs orders stamps)
CONTEXT_D = 7
SHOW_N = 20
BPH = 12                                                     # 5m bars per hour
A_CAP_BARS = V2.A_CAP_H * BPH                                # a' 6 h cap
D_CAP_BARS = V2.D_CAP_H * BPH                                # d 24 h cap
PX_BARS = D_CAP_BARS + 2                                     # entry bar + 24 h + 1 spare → limit 290 (weight 2)
RANK_H = 24
# 🔒 frozen read (pre-committed 2026-10-24, DECISION_LOG 369)
GATE_N, GATE_WEEKS, GATE_P, SHARE_MAX, FS_RATIO_MAX = 20, 6, 0.90, 0.50, 0.60
INTERIM_N = 8
GATE_KEY = "a_mean"

# the engine's own reading: _update_bearrun_monitor passes getattr(th, key, default) → _bearrun_next_state kwargs
_ENGINE_KEYS = (("r24_on", "bearrun_r24_on", 4.0), ("r24_off", "bearrun_r24_off", 3.0), ("below_on", "bearrun_below_on", 50.0),
                ("below_off", "bearrun_below_off", 47.0), ("eff_on", "bearrun_eff_on", 0.15), ("eff_off", "bearrun_eff_off", 0.15),
                ("latch_r6h", "bearrun_latch_r6h", 3.0), ("latch_ema50", "bearrun_latch_ema50", True))


def live_engine_kw(path=None):
    """→ (T0 kwargs for _bearrun_next_state, merge minutes) from trading_config.json thresholds (engine defaults when missing)."""
    th = {}
    try:
        with open(path or CFG, encoding="utf-8") as f:
            th = json.load(f).get("thresholds", {}) or {}
    except Exception:
        th = {}
    kw = {}
    for arg, key, dflt in _ENGINE_KEYS:
        v = th.get(key, dflt)
        kw[arg] = bool(v) if arg == "latch_ema50" else v
    try:
        merge = float(th.get("bearrun_window_merge_minutes", MERGE_MIN_DEFAULT) or 0)
    except (TypeError, ValueError):
        merge = MERGE_MIN_DEFAULT
    return kw, merge


def v25_kw(kw):
    return dict(kw, r24_on=V25_ON, r24_off=V25_OFF)


def superset_ok(kw):
    """V25 ⊇ T0 only while the live r24 legs are stricter than the shadow's (|on| > 2.5 ∧ |off| > 1.5)."""
    try:
        return abs(float(kw["r24_on"] or 0)) > V25_ON and abs(float(kw["r24_off"] or 0)) > V25_OFF
    except (TypeError, ValueError):
        return False


# ═══════════════════════════════ klines (budgeted, cached when complete) ═══════════════════════════════
def btc_5m(net, now_ms, days=HIST_DAYS, cache_dir=None):
    """CLOSED BTCUSDT 5m bars of the last `days` → DataFrame open_time, o, h, l, c, vol (the study's btc_5m.csv format), or None when
    any page is unreadable this run (a gap would corrupt r24 / below24)."""
    cache_dir = cache_dir or CACHE
    k0 = (now_ms - days * 24 * HOUR) // PAGE_MS
    k1 = (now_ms - 1) // PAGE_MS
    rows = []
    for k in range(int(k0), int(k1) + 1):
        start = k * PAGE_MS
        name = f"btc5m_{start}.json"
        complete = start + PAGE_MS <= now_ms
        hit = V2._cache_get(name, cache_dir) if complete else None
        if hit is None:
            raw = net.json("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "5m", "startTime": int(start), "limit": PAGE_BARS}, now_ms)
            if raw is None:
                return None
            hit = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in raw
                   if start <= int(x[0]) < start + PAGE_MS and int(x[0]) + FIVE <= now_ms]
            if complete and len(hit) == PAGE_BARS:
                V2._cache_put(name, hit, cache_dir)
        rows += hit
    if not rows:
        return None
    b = pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c", "vol"]).drop_duplicates("open_time").sort_values("open_time")
    return b.reset_index(drop=True)


def klines_5m(sym, open_ms, limit, net, now_ms, cache_dir=None):
    """CLOSED 5m bars [open_ms, o, h, l, c, quote_vol] of one alt from open_ms; cached when all `limit` came back. None = not readable this
    run (retried) · V2.DROPPED = a per-symbol 4xx (not 403 / 418 / 429)."""
    name = f"a5_{sym}_{int(open_ms)}_{int(limit)}.json"
    hit = V2._cache_get(name, cache_dir or CACHE)
    if hit is not None:
        return hit
    rows = net.json("/fapi/v1/klines", {"symbol": sym, "interval": "5m", "startTime": int(open_ms), "limit": int(limit)}, now_ms)
    if rows is None:
        if not net.stop and net.last_code is not None and 400 <= int(net.last_code) < 500:
            return V2.DROPPED
        return None
    out = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7])] for x in rows if int(x[0]) + FIVE <= now_ms]
    if len(out) == int(limit):
        V2._cache_put(name, out, cache_dir or CACHE)
    return out


# ═══════════════════════════════ T0 / V25 states + the extra windows ═══════════════════════════════
def states(R, kw):
    """→ (T0 bool array, V25 bool array) on the readings R (the study's run_variant with the REAL engine state machine)."""
    return TT.run_variant(R, "T0", engine_kw=kw), TT.run_variant(R, "T0", engine_kw=v25_kw(kw))


def _ms(t):
    return int(pd.Timestamp(t).value // 1_000_000)


def _tms(R):
    return np.array([_ms(x) for x in R.t.values], dtype=np.int64)


def extra_windows(R, t0, v25, merge_min=MERGE_MIN_DEFAULT):
    """V25 ON ∧ T0 OFF stretches, merged across gaps ≤ merge_min where T0 stays OFF → [dict(start_ms, end_ms | None (still extra at the
    last reading), end_reason, kind, t0_on_ms, v25_end_ms (first V25-OFF reading after the start, None = still ON), stretches, i0, i1)].
    i1 = the first reading after the window (len(R) if open)."""
    t = _tms(R)
    t0 = np.asarray(t0, bool)
    v = np.asarray(v25, bool)
    ex = v & ~t0
    n = len(ex)
    st, i = [], 0
    while i < n:
        if ex[i]:
            j = i
            while j < n and ex[j]:
                j += 1
            st.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for s, e in st:
        if merged:
            ps, pe, k = merged[-1]
            if pe < n and t[s] - t[pe] <= merge_min * 60_000 and not t0[pe:s].any():
                merged[-1] = [ps, e, k + 1]
                continue
        merged.append([s, e, 1])
    out = []
    for s, e, k in merged:
        tail = s > 0 and bool(t0[s - 1])
        t0_on = None
        if e < n:
            reason = "T0 ON" if t0[e] else "V25 OFF"
            lim = t[e] + merge_min * 60_000
            jj = np.nonzero(t0[e:] & (t[e:] <= lim))[0]
            if len(jj):
                t0_on = int(t[e + int(jj[0])])
        else:
            reason = "open"
        # V25 OFF = the first V25-off reading at / after the window END (a merged window's gaps are V25-off too — never stop there)
        off = np.nonzero(~v[e:])[0] if e < n else np.array([], dtype=int)
        v_end = int(t[e + int(off[0])]) if len(off) else None
        earlier = t0_on is not None
        kind = "TAIL→EARLIER" if (tail and earlier) else ("TAIL" if tail else ("EARLIER" if earlier else "EXTRA"))
        out.append(dict(start_ms=int(t[s]), end_ms=(int(t[e]) if e < n else None), end_reason=reason, kind=kind, t0_on_ms=t0_on,
                        v25_end_ms=v_end, stretches=int(k), i0=int(s), i1=int(e)))
    return out


def t0_windows(R, t0, merge_min=MERGE_MIN_DEFAULT):
    """the replica's T0 ON stretches merged like the engine ledger → [(start_ms, end_ms | None)]."""
    S = TT.stretches(R, np.asarray(t0, bool))
    if not len(S):
        return []
    open_last = bool(np.asarray(t0, bool)[-1])
    W = TT.merge(S, gap_min=merge_min)
    out = [(_ms(r.w_start), _ms(r.w_end)) for r in W.itertuples()]
    if open_last:
        out[-1] = (out[-1][0], None)
    return out


# ═══════════════════════════════ parity (containment) vs today's live Bear-Run fills ═══════════════════════════════
def dedup_fills(F, ms=DEDUP_MS):
    """one fill per pair within ±ms (the journal and the orders export stamp the same fill seconds apart)."""
    if F is None or not len(F):
        return pd.DataFrame(columns=["o_ms", "pair", "src"])
    keep, last = [], {}
    for r in F.sort_values(["pair", "o_ms"]).itertuples():
        if r.pair in last and r.o_ms - last[r.pair] <= ms:
            continue
        last[r.pair] = r.o_ms
        keep.append(r.Index)
    return F.loc[keep].sort_values("o_ms").reset_index(drop=True)


def load_fills(dec_paths=None, order_paths=None):
    """live Bear-Run fills → DataFrame o_ms, pair, src (journal OPEN rows strategy BEARRUN_SHORT + orders BEARRUN_SHORT fills), dedup per
    pair within ±2 min."""
    rows = []
    dec_paths = dec_paths if dec_paths is not None else sorted(EG.glob_exports(DEC_PATTERN))
    for p in dec_paths:
        try:
            d = pd.read_csv(p, low_memory=False, usecols=lambda c: c in ("t", "e", "pair", "strategy"))
        except Exception:
            continue
        if not {"t", "e", "pair", "strategy"} <= set(d.columns):
            continue
        d = d[(d.e.astype(str) == "OPEN") & (d.strategy.fillna("").astype(str).str.upper() == "BEARRUN_SHORT")]
        for r in d.itertuples():
            ts = pd.to_datetime(str(r.t)[:19].replace(" ", "T"), errors="coerce", utc=True)
            if pd.notna(ts):
                rows.append((int(ts.value // 1_000_000), str(r.pair), "journal"))
    try:
        T = V2.load_t0(order_paths)
        for r in T.itertuples():
            rows.append((int(r.o_ms), str(r.pair), "orders"))
    except Exception:
        pass
    if not rows:
        return pd.DataFrame(columns=["o_ms", "pair", "src"])
    return dedup_fills(pd.DataFrame(rows, columns=["o_ms", "pair", "src"]))


def parity(R, t0, fills, merge_min=MERGE_MIN_DEFAULT, tol_min=PARITY_TOL_MIN):
    """containment → dict(n_fills, matched, n_live_w, live_w_matched, replica_w, replica_w_with_fill, span, unmatched[]) on fills inside
    the replica's post-warm-up range."""
    t = _tms(R)
    t0 = np.asarray(t0, bool)
    lo, hi = (int(t[0]) + WARM_MS, int(t[-1])) if len(t) else (0, -1)
    F = dedup_fills(fills)
    F = F[(F.o_ms >= lo) & (F.o_ms <= hi)] if len(F) else F
    tol = tol_min * 60_000
    hits = []
    for r in F.itertuples():
        m = (t >= r.o_ms - tol) & (t <= r.o_ms + tol)
        hits.append(bool(t0[m].any()))
    F = F.assign(hit=hits)
    lw, cur = [], None                                       # live windows = fills ≤ merge_min apart
    for r in F.itertuples():
        if cur is not None and r.o_ms - cur[1] <= merge_min * 60_000:
            cur[1] = r.o_ms
            cur[2] = cur[2] or r.hit
        else:
            cur = [r.o_ms, r.o_ms, r.hit]
            lw.append(cur)
    RW = [w for w in t0_windows(R, t0, merge_min) if w[0] >= lo]
    with_fill = 0
    for s, e in RW:
        e2 = e if e is not None else hi
        if len(F) and ((F.o_ms >= s - tol) & (F.o_ms <= e2 + tol)).any():
            with_fill += 1
    return dict(n_fills=int(len(F)), matched=int(sum(hits)), n_live_w=len(lw), live_w_matched=int(sum(1 for w in lw if w[2])),
                replica_w=RW, replica_w_with_fill=with_fill, span=(lo, hi),
                unmatched=[f"{pd.Timestamp(int(r.o_ms), unit='ms'):%m-%d %H:%M} {r.pair}" for r in F.itertuples() if not r.hit])


def _parity_line(P):
    if P["n_fills"] == 0:
        return (f"Parity (containment): no live Bear-Run fill in the replica range {_fmt(P['span'][0])} → {_fmt(P['span'][1])} UTC "
                f"(replica T0 windows there: {len(P['replica_w'])}) — not measurable this run.")
    rate = P["matched"] / P["n_fills"]
    txt = (f"Parity (containment of live Bear-Run fills in the replica's T0 ON, {_fmt(P['span'][0])} → {_fmt(P['span'][1])} UTC): "
           f"{P['matched']}/{P['n_fills']} fills contained (±{PARITY_TOL_MIN} min) = {rate:.0%} · live windows {P['live_w_matched']}/"
           f"{P['n_live_w']} · replica T0 windows {len(P['replica_w'])} ({P['replica_w_with_fill']} with a live fill; one without = "
           f"per-entry refusals / no signal, not a mismatch — containment cannot see replica ON time the engine missed)")
    if P["unmatched"]:
        txt += " · not contained: " + ", ".join(P["unmatched"][:6])
    if P["replica_w"]:
        txt += " · replica T0 windows: " + ", ".join(f"{_fmt(s)}→{_fmt(e)}" for s, e in P["replica_w"][-6:])
    return txt


# ═══════════════════════════════ pricing (5m grid from the window start) + records ═══════════════════════════════
def _fmt(ms):
    return V2._fmt_ms(ms)


def _num(x, f="{:+.2f}"):
    return V2._num(x, f)


def _window_universe(start_ms, uni_fn, now_ms, cache_dir=None):
    """the universe snapshot for a window — taken at the first run that prices it, reused afterwards (own cache folder)."""
    name = f"uni_w_{int(start_ms)}.json"
    hit = V2._cache_get(name, cache_dir or CACHE)
    if isinstance(hit, dict) and hit.get("syms"):
        return hit["syms"], hit.get("at")
    sy = uni_fn()
    if sy is None:
        return None, None
    V2._cache_put(name, {"at": int(now_ms), "syms": list(sy)}, cache_dir or CACHE)
    return list(sy), int(now_ms)


def d_exit_ms(w):
    """d exit reading: EARLIER kinds hold to V25 OFF (the full V25 stretch); else the window end; capped at start + 24 h."""
    cap = int(w["start_ms"]) + V2.D_CAP_H * HOUR
    if "EARLIER" in w["kind"]:
        x = w.get("v25_end_ms")
        return (min(int(x), cap), "V25 OFF") if x is not None else (cap, "24 h cap")
    x = w.get("end_ms")
    return (min(int(x), cap), "window end") if x is not None else (cap, "24 h cap")


def price_5m(w, net, now_ms, syms, cache_dir=None):
    """→ dict(a / d stats, basket, rows, dropped, unread, done) for one time-final window. Basket = top-20 by quote volume of the 24 closed
    1 h bars before the start (V2's cached 1 h klines); pricing on 5m bars, entry = the close of the 5m bar closing at the start."""
    start = int(w["start_ms"])
    rh = (start // HOUR) * HOUR                               # the last 1 h close ≤ start
    rank, unread, bad = [], 0, []
    for s in syms:
        rows = V2.klines_1h(s, rh - RANK_H * HOUR, RANK_H, net, now_ms, cache=True)
        if rows is None:
            unread += 1
            continue
        if rows == V2.DROPPED:
            bad.append(s)
            continue
        if len(rows) == RANK_H and all(int(r[0]) == rh - (RANK_H - k) * HOUR for k, r in enumerate(rows)):
            rank.append((sum(float(r[5]) for r in rows), s))
    rank.sort(reverse=True)
    basket = [s for _, s in rank[:V2.BASKET_N]]
    xd_ms, xd_lab = d_exit_ms(w)
    xd = max(1, int((xd_ms - start) // FIVE))
    rows_a, rows_d, dropped = [], [], []
    for s in basket:
        k5 = klines_5m(s, start - FIVE, PX_BARS, net, now_ms, cache_dir)
        if k5 is None:
            unread += 1
            continue
        if k5 == V2.DROPPED:
            bad.append(s)
            continue
        bars = []
        for k, r in enumerate(k5):
            if int(r[0]) != start - FIVE + k * FIVE:
                break                                         # contiguous bars only
            bars.append(r)
        if len(bars) - 1 < max(xd, A_CAP_BARS):
            dropped.append(s)                                 # a gap / halt / delisting inside the window → out of the basket (listed)
            continue
        rows_a.append((s,) + V2.walk_a(bars, 0, len(bars) - 1, cap=A_CAP_BARS))
        rows_d.append((s,) + V2.walk_d(bars, 0, xd))
    return dict(basket=basket, a=V2._stats(rows_a), d=V2._stats(rows_d), d_exit=xd_lab, d_exit_ms=int(xd_ms),
                a_rows=[[s, round(p, 3), r] for s, p, r in rows_a], dropped=dropped, dropped_4xx=bad, unread=unread,
                done=bool(unread == 0 and not net.stop))


def time_final(w, last_ms, now_ms, merge_min):
    """end known ∧ 24 h since the start (class) ∧ merge_min since the end (kind settled)."""
    if w.get("end_ms") is None:
        return False
    need = max(int(w["start_ms"]) + 24 * HOUR, int(w["end_ms"]) + int(merge_min * 60_000))
    return last_ms >= need and now_ms >= need


def base_record(w, R):
    s, e = w["i0"], w["i1"]
    px = R.px.values
    p0 = float(px[s])
    tms = _tms(R)
    cap = int(np.searchsorted(tms, int(w["start_ms"]) + 24 * HOUR, side="right"))
    seg = R.lo.values[s + 1:min(e, cap)]                       # after the entry bar (its low printed before the ON close) to the window
    #                                                            end (merged gaps included; the first non-extra bar excluded), ≤ 24 h
    lo = float(np.min(seg)) if len(seg) else p0
    p_end = float(px[e]) if e < len(px) else float(px[-1])
    cls, sub, cls_final = V2.classify(pd.DataFrame({"c": px, "h": R.hi.values, "l": R.lo.values}), s, bars_per_h=BPH)
    return dict(on_ms=int(w["start_ms"]), start_utc=_fmt(w["start_ms"]), end_ms=w["end_ms"], end_utc=_fmt(w["end_ms"]),
                end_reason=w["end_reason"], kind=w["kind"], t0_on_ms=w["t0_on_ms"], t0_on_utc=_fmt(w["t0_on_ms"]),
                v25_end_ms=w.get("v25_end_ms"), stretches=w["stretches"],
                dur_min=(None if w["end_ms"] is None else round((w["end_ms"] - w["start_ms"]) / 60_000)),
                btc_start=round(p0, 2), btc_end=round(p_end, 2), move_pct=round((p_end / p0 - 1) * 100, 3), btc_low=round(lo, 2),
                drop_pct=round((lo / p0 - 1) * 100, 3), r24=round(float(R.r24.values[s]), 3), below24=round(float(R.below24.values[s]), 1),
                eff24=round(float(R.eff24.values[s]), 3), cls=cls, cls_sub=sub, _cls_final=bool(cls_final), status="open",
                a_n=0, a_mean=None, a_wr=None, a_stops=0, d_n=0, d_mean=None, d_wr=None, d_stops=0, d_exit=None, basket=None,
                dropped=[], dropped_4xx=[], universe_at=None)


def evaluate(w, R, net, now_ms, uni_fn, merge_min, price=True, cache_dir=None):
    """one extra window → (record, status): 'open' (not time-final) · 'retry' (time-final, pricing incomplete this run) · 'final' ·
    'unpriceable' (time-final, every basket coin unpriceable)."""
    rec = base_record(w, R)
    if not time_final(w, _ms(R.t.values[-1]), now_ms, merge_min):
        rec.pop("_cls_final", None)
        return rec, "open"
    rec["status"] = "retry"
    if not rec.pop("_cls_final", True):                        # a BTC 5m gap left the 24 h class unresolved → never final as "pending"
        return rec, "retry"
    if not price:
        return rec, "retry"
    sy, at = _window_universe(w["start_ms"], uni_fn, now_ms, cache_dir)
    if sy is None:
        return rec, "retry"
    P = price_5m(w, net, now_ms, sy, cache_dir)
    rec.update(basket=P["basket"], a_n=P["a"]["n"], a_mean=P["a"]["mean"], a_wr=P["a"]["wr"], a_stops=P["a"]["stops"],
               d_n=P["d"]["n"], d_mean=P["d"]["mean"], d_wr=P["d"]["wr"], d_stops=P["d"]["stops"], d_exit=P["d_exit"],
               d_exit_utc=_fmt(P["d_exit_ms"]), dropped=P["dropped"], dropped_4xx=P["dropped_4xx"], a_rows=P["a_rows"],
               universe_at=None if at is None else pd.Timestamp(int(at), unit="ms").isoformat())
    if not P["done"]:
        return rec, "retry"
    if rec["a_mean"] is None:
        rec["status"] = "unpriceable"
        return rec, "unpriceable"
    rec["status"] = "final"
    return rec, "final"


def _line(r):
    st = r.get("status")
    mark = {"final": "", "unpriceable": "✗ UNPRICEABLE ", "lost": "✗ LOST "}.get(st, "⏳ ")
    if st == "lost":
        return f"- {mark}{r['start_utc']} · {r.get('note', '')}"
    t0 = (f" (T0 ON {r['t0_on_utc']})" if r.get("t0_on_ms") else "")
    frag = f" [{r['stretches']} stretches]" if int(r.get("stretches") or 1) > 1 else ""
    dur = f"{r['dur_min']} min" if r.get("dur_min") is not None else "open"
    drop = (f" · dropped {len(r.get('dropped') or []) + len(r.get('dropped_4xx') or [])}" if (r.get("dropped") or r.get("dropped_4xx")) else "")
    dx = (f" (d to {r['d_exit']} {r.get('d_exit_utc', '')})" if r.get("d_exit") else "")
    return (f"- {mark}{r['start_utc']} → {r['end_utc']} ({dur}, end: {r['end_reason']}){frag} · {r['kind']}{t0} · "
            f"r24 {r['r24']:+.2f} · BTC {r['btc_start']:,.0f} → {r['btc_end']:,.0f} ({r['move_pct']:+.2f} %, low {r['drop_pct']:+.2f} %) · "
            f"{r['cls']}{(' ' + r['cls_sub']) if r.get('cls_sub') else ''} · basket a' {_num(r['a_mean'], '{:+.3f}')} % "
            f"(WR {_num(r['a_wr'], '{:.0f}')} %, {r['a_n']} coins) · d {_num(r['d_mean'], '{:+.3f}')} %{dx}{drop}")


# ═══════════════════════════════ persistence + the frozen read ═══════════════════════════════
def _fresh():
    return {"version": 1, "dl": DL, "windows": [], "seen": {}, "gate_read": None}


def load_state(path=None):
    st = V2.load_state(path or STATE)
    if st.get("dl") != DL:                                   # V2's default skeleton (missing / unreadable file) → this scout's
        st = _fresh() if not st.get("windows") else st
    st.setdefault("gate_read", None)
    st.setdefault("seen", {})
    return st


def _final(windows):
    return sorted([w for w in windows if w.get(GATE_KEY) is not None and w.get("status", "final") == "final"], key=lambda w: w["on_ms"])


def _sumtxt(m):
    return (f"{m['n']} windows · {m['days']} days · {m['weeks']} ISO weeks (span {m['span_w']:.1f} wk) · a' Σ {m['S']:+.2f} pts · "
            f"P(Σ > 0) {_num(m['p'], '{:.2f}')} · largest window " + (f"{m['share']:.0%} of Σ" if m["S"] > 0 else "n/a (Σ ≤ 0)")
            + f" · false starts lose {m['fs_loss']:.2f} vs clean slides earn {m['clean_sum']:+.2f} (FS-loss ratio "
            + ("∞" if m["fs_ratio"] == float("inf") else f"{m['fs_ratio']:.0%}") + ")")


def gate_read(windows, pending=()):
    """🔒 the frozen read (DECISION_LOG 369) → (verdict, text, prefix_n); COLLECTING / WAITING are not decisions."""
    W = _final(windows)
    end, wk = None, set()
    for k, w in enumerate(W):
        wk.add(V2._week(w))
        if k + 1 >= GATE_N and len(wk) >= GATE_WEEKS:
            end = k
            break
    if end is None:
        return "COLLECTING", f"⏳ {len(W)}/{GATE_N} final extra windows over {len(wk)}/{GATE_WEEKS} ISO weeks", len(W)
    P = W[:end + 1]
    early = V2._blocked_by_pending(P[-1]["on_ms"], pending)
    if early:
        return "WAITING", (f"prefix reached, but {len(early)} cohort extra window(s) started by its end are not final yet "
                           f"({', '.join(_fmt(x) for x in early[:4])}) — decided once they are persisted"), len(P)
    m = V2.summary(P, key=GATE_KEY)
    cond = [("① Σ a' > 0", m["S"] > 0),
            (f"② window-bootstrap P(Σ > 0) ≥ {GATE_P:.2f}", m["S"] > 0 and m["p"] >= GATE_P),
            ("③ no window ≥ 50 % of Σ", m["S"] > 0 and m["share"] < SHARE_MAX),
            ("④ false starts lose ≤ 60 % of what clean slides earn" + ("" if m["clean_sum"] > 0 else " (no clean-slide gain in the prefix)"),
             m["clean_sum"] > 0 and m["fs_ratio"] <= FS_RATIO_MAX)]
    failed = [n for n, ok in cond if not ok]
    head = f"read at {_sumtxt(m)} (to {_fmt(P[-1]['on_ms'])})"
    if not failed:
        return "CANDIDATE", head + (" → ALL four hold: CANDIDATE to lower bearrun_r24_on 4 → 2.5 / off 3 → 1.5 — operator decides "
                                    "(proxy basket, not the engine: re-check on an engine replay first); nothing changed by the scout"), len(P)
    return "KEEP_4", head + " → KEEP bearrun_r24_on 4.0; failed: " + " · ".join(failed), len(P)


def interim(windows):
    W = _final(windows)
    if len(W) < INTERIM_N:
        return f"interim line at {INTERIM_N} final extra windows — not reached ({len(W)})"
    m = V2.summary(W, key=GATE_KEY)
    return f"INTERIM (NON-TERMINAL, informational, never frozen): {_sumtxt(m)}"


def _tally(W):
    F = _final(W)
    bad = {k: sum(1 for w in W if w.get("status") == k) for k in ("unpriceable", "lost")}
    tail = f" · unpriceable {bad['unpriceable']} · lost {bad['lost']} (listed, never in the read)"
    if not F:
        return "Tally: 0 priced final windows" + tail
    parts = []
    for key, lab in (("cls", "class"), ("kind", "kind")):
        g = {}
        for w in F:
            g.setdefault(w.get(key) or "?", []).append(w)
        parts.append(f"by {lab}: " + " · ".join(
            f"{k} {len(v)} (a' Σ {sum(x['a_mean'] for x in v):+.2f}, d Σ {sum((x.get('d_mean') or 0) for x in v):+.2f})"
            for k, v in sorted(g.items())))
    return (f"Tally ({len(F)} final): a' Σ {sum(w['a_mean'] for w in F):+.2f} · d Σ {sum((w.get('d_mean') or 0) for w in F):+.2f} · "
            + " | ".join(parts) + tail)


def _lost_record(start_ms, note, now_ms):
    return dict(on_ms=int(start_ms), start_utc=_fmt(start_ms), status="lost", note=note, a_mean=None, d_mean=None, cls="lost",
                kind="?", final_at=pd.Timestamp(int(now_ms), unit="ms").isoformat())


# ═══════════════════════════════ run ═══════════════════════════════
def run(now_ms=None, net=None, state_path=None, deploy_ms="auto", btc5=None, syms=None, fills=None, cfg_path=None, cache_dir=None):
    """→ report lines. Never raises on data (the opportunity-scout wrapper also catches)."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    state_path = state_path or STATE
    net = net or V2.Net()
    kw, merge_min = live_engine_kw(cfg_path)
    L = [f"## 🐻⏬ BEARRUN_R24_25 shadow — Bear-Run r24 −2.5 % (DECISION_LOG {DL}, observe-only, config unchanged)", "",
         f"T0 = today's Bear-Run monitor (5m replica of the engine: r24 ON ≤ −{abs(float(kw['r24_on'])):g} / stay ≤ −{abs(float(kw['r24_off'])):g}, "
         f"below24 ≥ {kw['below_on']:g} / {kw['below_off']:g} %, eff24 ≥ {kw['eff_on']:g} / {kw['eff_off']:g}, latch r6 ≥ +{kw['latch_r6h']:g} % "
         f"∨ price > 1 h EMA50 [{'on' if kw['latch_ema50'] else 'off'}] — live trading_config.json) · V25 = the same with r24 ON ≤ −{V25_ON} / "
         f"stay ≤ −{V25_OFF} · EXTRA = V25 ON ∧ T0 OFF, merged ≤ {merge_min:.0f} min · basket (top-20 alt perps, SHORT) priced on 5m bars "
         "from the window start: a' (+0.5 / −1 / 6 h) and d (hold while V25 ON: to the window end, EARLIER windows to V25 OFF; +2 % stop, "
         "24 h cap), 0.10 % fees · PROXIES OF THE SLEEVE, NOT THE ENGINE."]
    dep = _GK.dl_deploy_ms(DL, lag_min=10, label="BEARRUN_R24_25") if deploy_ms == "auto" else deploy_ms
    if not superset_ok(kw):
        L += [f"⚠ GUARD: live bearrun_r24_on {kw['r24_on']} / off {kw['r24_off']} is not stricter than the shadow's {V25_ON} / {V25_OFF} — "
              "V25 is no longer a superset of T0, the extra-time unit is undefined: nothing evaluated, the frozen read is SKIPPED. "
              "Re-scope this line (DECISION_LOG 369) before reading it.", ""]
        if dep is not None:
            L += _persisted_lines(load_state(state_path), None, now_ms, read=False)
        return L + [""]
    if btc5 is None:
        btc5 = btc_5m(net, now_ms, cache_dir=cache_dir)
    if btc5 is None or len(btc5) < 288 + 24:
        L += [f"BTC 5m klines unreadable this run ({net.why or 'no data'}) — nothing evaluated.", ""]
        if dep is not None:
            L += _persisted_lines(load_state(state_path), None, now_ms)
        L.append(f"network: {net.requests} requests · used weight {net.used}" + (f" · stopped: {net.why}" if net.stop else ""))
        return L + [""]
    R = TT.compute_readings(btc5)
    t0, v25 = states(R, kw)
    X = extra_windows(R, t0, v25, merge_min)
    last_ms = _ms(R.t.values[-1])
    uni = {"syms": syms, "tried": syms is not None}

    def _syms():
        if not uni["tried"]:
            uni["syms"], uni["tried"] = V2.universe(net, now_ms), True
        return uni["syms"]
    F = fills if fills is not None else load_fills()
    L.append(_parity_line(parity(R, t0, F, merge_min)))
    t0_now, v25_now = bool(t0[-1]), bool(v25[-1])
    L.append(f"Now ({_fmt(last_ms)} UTC): r24 {_num(float(R.r24.values[-1]))} % · below24 {_num(float(R.below24.values[-1]), '{:.1f}')} % · "
             f"eff24 {_num(float(R.eff24.values[-1]), '{:.3f}')} · r6 {_num(float(R.r6.values[-1]))} % · T0 {'ON' if t0_now else 'OFF'} · "
             f"V25 {'ON' if v25_now else 'OFF'}" + (" · ⚡ EXTRA time now" if (v25_now and not t0_now) else ""))
    st = load_state(state_path)
    persisted = {int(w["on_ms"]): w for w in st["windows"]}
    ctx_from = last_ms - CONTEXT_D * 24 * HOUR
    warm_end = _ms(R.t.values[0]) + WARM_MS
    show, new, pending, seen_now, merged_lost = [], [], set(), {}, []
    for w in X:
        if w["start_ms"] < warm_end:
            continue
        in_cohort = dep is not None and w["start_ms"] >= dep
        in_ctx = (not in_cohort) and w["start_ms"] >= ctx_from
        if not (in_cohort or in_ctx):
            continue
        if in_cohort and int(w["start_ms"]) in persisted:
            p = persisted[int(w["start_ms"])]
            if p.get("status") != "lost" and p.get("end_ms") is not None and w.get("end_ms") != p.get("end_ms"):
                merged_lost.append(f"{_fmt(w['start_ms'])} (persisted end {_fmt(p.get('end_ms'))}, now {_fmt(w.get('end_ms'))})")
            continue
        rec, status = evaluate(w, R, net, now_ms, _syms, merge_min, price=(in_cohort or dep is None), cache_dir=cache_dir)
        if in_cohort:
            seen_now[str(int(w["start_ms"]))] = rec["start_utc"]
            if status in ("final", "unpriceable"):
                rec["final_at"] = pd.Timestamp(now_ms, unit="ms").isoformat()
                new.append(rec)
            else:
                pending.add(int(rec["on_ms"]))
        show.append((rec, in_cohort))
    if dep is None:
        L.append(f"⚠ NOT evaluable — the DECISION_LOG {DL} commit is not in git yet; nothing is counted or persisted. "
                 f"Context (informational): extra windows of the last {CONTEXT_D} days (priced once final).")
        body = []
    else:
        L.append(f"Cohort: extra windows starting ≥ {_fmt(dep)} UTC (DL {DL} commit + 10 min); final when ended + 24 h since the start + "
                 f"{merge_min:.0f} min since the end + fully priced; persisted append-only.")
        with V2._locked(state_path):
            st = load_state(state_path)
            have = {int(x["on_ms"]) for x in st["windows"]}
            lost = []
            for k, utc in list(st["seen"].items()):          # a seen cohort window no longer extracted (scrolled out) → a lost record
                if int(k) not in have and k not in seen_now and not any(int(r["on_ms"]) == int(k) for r in new):
                    lost.append(_lost_record(int(k), f"seen {utc}, scrolled out of the {HIST_DAYS}-day 5m fetch before it was final", now_ms))
            n = V2.append_windows(st, new + lost) if (new or lost) else 0
            done = {str(int(r["on_ms"])) for r in new + lost}
            seen_before = dict(st["seen"])
            st["seen"] = {k: v for k, v in {**st["seen"], **seen_now}.items() if k not in done and int(k) not in have}
            body = _persisted_lines(st, pending, now_ms)
            dirty = bool(st.pop("_dirty", False)) or st["seen"] != seen_before
            ok = V2.save_state(st, state_path) if (n or dirty) else True
        if n:
            L.append(f"Persisted {n} newly final / unpriceable / lost extra window record(s)." if ok
                     else "⚠ state file not writable — nothing persisted.")
    if merged_lost:
        L.append(f"✗ LOST stretches (would merge into an already-persisted window — not counted): {len(merged_lost)} · "
                 + ", ".join(merged_lost[:4]))
    live = [r for r, c in show if c]
    ctx = [r for r, c in show if not c]
    if live:
        L.append("Open / not yet final cohort extra windows:")
        L += ["  " + _line(r) for r in live]
    if ctx:
        L.append(f"Context (last {CONTEXT_D} days, before the cohort — informational, never counted):")
        L += ["  " + _line(r) for r in ctx]
        cf = [r for r in ctx if r.get("a_mean") is not None]
        if cf:
            L.append(f"  context totals (priced): {len(cf)} windows · a' Σ {sum(r['a_mean'] for r in cf):+.2f} · d Σ "
                     f"{sum((r['d_mean'] or 0) for r in cf):+.2f}")
    if not show:
        L.append(f"No extra window in the last {CONTEXT_D} days.")
    L += body
    L.append(f"network: {net.requests} requests · used weight {net.used}" + (f" · stopped: {net.why}" if net.stop else "")
             + (" · universe unreadable — baskets not priced" if uni["tried"] and uni["syms"] is None else ""))
    L.append("Prior evidence (reports/BEARRUN_TRIGGER_TIMING_2026-10-23_FINAL.md, engine replay, today's momentum-short exit, 3 seeds): T0 48 "
             "fills/seed · 57 % WR · −0.096 %/fill · Σ −4.66 %; earlier triggers ADD losing shorts — T1a / T1b / T1c (dd24 3 / 3.5 / 4) Δ Σ "
             "−11.7 / −10.3 / −3.8 % vs T0, all worse than the shuffled-timing null; T2 / T3 'beat' T0 only by not trading. r24 at −2.5 % "
             "itself was never tested. reports/BEARRUN_V2_REPLAY_s12.md: the earlier V2 trigger loses vs T0 under every exit.")
    L.append("Blind spots: proxy basket ≠ engine fills (no engine gates / signals / per-entry refusals off24lo / breadth / spacing, no "
             "slippage) · the momentum-short exit and its EMA13-cross exit are not modelled · earlier entries take slots / per-pair "
             "spacing and can displace T0's own later fills (not modelled) · 5m closed-bar replica vs the engine's ~110 s poll "
             "(containment parity above) · universe = top-60 snapshot at first pricing (survivorship).")
    return L + [""]


def _persisted_lines(st, pending, now_ms, read=True):
    """final / unpriceable / lost records (last 20) + tally + interim + the frozen read (decided once). pending None = this run could not
    see the cohort; read False = the r24 guard is tripped (no read)."""
    L = []
    W = sorted(st["windows"], key=lambda w: w["on_ms"])
    if W:
        L.append(f"Persisted cohort extra windows ({len(W)}; last {min(SHOW_N, len(W))}):")
        L += ["  " + _line(w) for w in W[-SHOW_N:]]
        L.append(_tally(W))
    L.append("🟡 " + interim(W))
    lab = "🔒 Frozen read (r24 −2.5 vs 4, pre-committed, terminal)"
    if not read:
        L.append(f"{lab}: SKIPPED (r24 guard)" + (f" — frozen earlier: {st['gate_read']['verdict']}" if st.get("gate_read") else ""))
        return L
    if pending is None and not st.get("gate_read"):
        L.append(f"{lab}: not read this run (BTC unreadable)")
        return L
    v, txt, now_ = V2.decide_once(st, "gate_read", gate_read, W, now_ms, pending or ())
    if now_:
        st["_dirty"] = True
    L.append(f"{lab}: {v} — {txt}")
    return L


def main():
    print("\n".join(run(int(time.time() * 1000))))


if __name__ == "__main__":
    main()
