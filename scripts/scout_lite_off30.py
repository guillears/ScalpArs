#!/usr/bin/env python3
"""🪶 Scout — LITE_OFF30H3 observe line (pre-registered 2026-10-08, DECISION_LOG 259; OBSERVE only — never changes config, no bot API).

Source: reports/LITE_ENTRY_SIGNS_STUDY_2026-10-08.md ("Watch line proposal") + scripts/study_lite_signs_main.py (S1 dd30 — the formula is
copied here so values are identical; the selftest re-derives the study's stored dd30 on its 724 rows, ≤ 1e-6, when the CSV + cache exist).

ZONE (frozen 3 %)  a FRENZY_LITE fill whose SIGNAL close is ≥ 3 % below the max 5m HIGH of the 6 5m bars ending at the signal bar
                   (signal bar included = study bar_feats: hi30 = h[i−5 : i+1].max(), dd30 = (1 − c[i] / hi30) × 100).
                   Signal bar = the last 5m bar CLOSED at / before the fill (study entry_ts = bar_open + 5 min): open = ⌊t / 5m⌋·5m − 5m.
                   All 6 bars must be present (study indexing assumed contiguity) — else the fill is UNSCORED.
COHORT             CLOSED FRENZY_LITE LONG fills in the ~/Downloads orders exports, dedup (opened_at, pair, direction), newest export wins,
                   opened ≥ 2026-10-07 00:00 UTC (SKL 10-08 in); DAY units; Σ$ as sized = the fill's own pnl. Screened by today's
                   stack: fills the FRENZY bearish-day block (DECISION_LOG 250) would refuse (stamped BTC 1d ret < 0 ∧ BTC trend gap < 0)
                   are shown apart, never counted; a missing stamp counts (fail-open, like the engine). A fill opened > 60 s after its
                   signal bar's close is flagged (the engine may have judged another bar) but still counted.
FROZEN VERDICT     computed ONCE on the first prefix (by open time) where the zone reaches N ≥ 30 on ≥ 15 days, persisted in
                   reports/SCOUT_LITE_OFF30H3.json, never re-fit: RETIRE if zone mean ≥ rest mean (rest = scored non-zone fills ≤ the
                   prefix end); FILTER CANDIDATE (operator decides) iff zone WR < LITE breakeven WR (scored LITE fills ≤ prefix end;
                   fallback 49.9 % until 30) ∧ day-clustered bootstrap P(mean < 0) ≥ 0.95 (4,000, seed 7) ∧ no day / pair ≥ 50 % of
                   the gross loss; else KEEP OBSERVING. No freeze while an OPEN LITE position or a cache-lagging counted fill opened ≤
                   the prefix end could still change it (NF.freeze_hold).
BINANCE            cache first (reports/backtest_cache/k5m_full + this line's own scout_lite_off30/5m); ≤ 2 requests per run (oldest
                   unscored fill first: one 6-bar 5m page ending at its signal bar, weight 1); refused while the SHARED scout cooldown
                   (NF.SHARED_COOLDOWN, the ban is per IP) is live, once the last X-MBX-USED-WEIGHT-1M ≥ 900, or past the 25 s
                   wall-clock budget; a 418 / 429 stops the run and persists the shared cooldown (Retry-After honoured). A page fetched
                   with ≥ 1 bar that still leaves the window incomplete, a 4xx other than 418/429 (final at once, its request slot
                   refunded), or 3 empty / failed attempts on one page = UNSCORED for good (never re-requested, never blocks a freeze).
                   Dedup prefers a CLOSED row over an OPEN one for the same fill whatever the export order.
"""
import glob
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (shared cooldown, bootstrap, loss share, freeze state I/O)

M5 = 300_000
START = "2026-10-07 00:00"
OFF_PCT, WIN_BARS = 3.0, 6                          # FROZEN: dd30 ≥ 3 % over the 6 bars ending at the signal bar
N_MIN, DAYS_MIN, BE_REF, BE_MIN_FILLS = 30, 15, 49.9, 30
BOOT_N, BOOT_SEED = 4000, 7
MAX_REQ, WEIGHT_STOP, BUDGET_S = 2, 900, 25.0
STRATEGY = "FRENZY_LITE"
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "entry_btc_1d_ret_pct", "entry_btc_trend_gap_pct")
LATE_S = 60                                         # a fill opened > 60 s after its signal bar's close is flagged (still counted)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, "reports", "backtest_cache")
MY_CACHE = os.path.join(_CACHE, "scout_lite_off30")
STATE = os.path.join(_ROOT, "reports", "SCOUT_LITE_OFF30H3.json")
STUDY_CSV = os.path.join(_ROOT, "reports", "LITE_ENTRY_SIGNS_STUDY_2026-10-08.csv")
REF = ("study MAIN 548 (bearish days out, today's exit on ticks): zone 120 · 48 % · −0.10 % vs rest +0.38 % (Δ −0.48), day-clustered "
       "P(mean<0) 0.64 — fails the confidence leg; negative in both halves and every leave-one-month-out; ROBUST 724 (bearish days in) "
       "zone +0.04 %; flush-stop share 52 % of zone stops vs 24 % of the rest")
REVERT = ("Pre-committed revert if ever armed: the first 10 blocked signals re-priced with the live exit → WR ≥ LITE breakeven or Σ > 0 → "
          "switch it off")
_NET_BLOCKED = False                                # selftest: any fetch attempt raises


# ─────────────────────────── orders ───────────────────────────
def _read_exports():
    fr = []
    for f in glob.glob(EXPORT_GLOB):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    return fr


def load_orders():
    """CLOSED FRENZY_LITE LONG fills, dedup (opened_at, pair, direction) across exports (newest export by mtime wins) → prepared frame."""
    fr = _read_exports()
    if not fr:
        return _prep(pd.DataFrame(columns=list(COLS)))
    return _prep(_cohort(pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable")))


def _cohort(o):
    """rows in precedence order (last wins; a CLOSED row always beats an OPEN one) → dedup (opened_at[:19], pair, direction) → CLOSED FRENZY_LITE LONG (run(orders=…) too)."""
    o = o.reindex(columns=list(dict.fromkeys(list(COLS) + list(o.columns))))
    if not len(o):
        return o
    o["_k"] = o.opened_at.astype(str).str[:19]
    o["_c"] = o.status.astype(str).str.upper().eq("CLOSED")
    o = o.sort_values("_c", kind="stable")                         # a CLOSED row beats an OPEN one whatever the file order; else last wins
    o = o.drop_duplicates(["_k", "pair", "direction"], keep="last")
    return o[(o.status.astype(str).str.upper() == "CLOSED") & (o.direction.astype(str) == "LONG")
             & (o.entry_strategy.astype(str) == STRATEGY)]


def _prep(o):
    o = o.copy()
    k = o.opened_at.astype(str).str[:19] if len(o) else pd.Series(dtype=str)
    t = pd.to_datetime(k, format="ISO8601", errors="coerce") if len(o) else pd.Series(dtype="datetime64[ns]")
    o = o[t.notna()].assign(ts=t[t.notna()]) if len(o) else o.assign(ts=pd.Series(dtype="datetime64[ns]"))
    o["pct"] = pd.to_numeric(o.get("pnl_percentage"), errors="coerce") if len(o) else pd.Series(dtype=float)
    o["usd"] = pd.to_numeric(o.get("pnl"), errors="coerce") if len(o) else pd.Series(dtype=float)
    o = o[o.pct.notna()].copy()
    o["pair"] = o.pair.astype(str) if len(o) else pd.Series(dtype=str)
    o["day"] = o.ts.dt.strftime("%Y-%m-%d") if len(o) else pd.Series(dtype=str)
    o["t_ms"] = o.ts.values.astype("datetime64[ms]").astype("int64") if len(o) else pd.Series(dtype="int64")
    o["bo"] = sig_bar(o.t_ms.values) if len(o) else pd.Series(dtype="int64")
    o["late_s"] = (o.t_ms - o.bo - M5) / 1000.0 if len(o) else pd.Series(dtype=float)
    o["bear"] = bearish_day(o.get("entry_btc_1d_ret_pct"), o.get("entry_btc_trend_gap_pct"), len(o))
    return o.reset_index(drop=True)


def bearish_day(r, g, n):
    """today's stack: the engine's frenzy_bearish_day_block (services/frenzy.frenzy_bearish_day, DECISION_LOG 250) refuses a LITE entry
    when BTC 1d ret < 0 ∧ BTC trend gap < 0 (both stamped). A missing stamp = FAIL-OPEN (the engine lets it through) → counted."""
    r = pd.to_numeric(r, errors="coerce") if r is not None else pd.Series(np.nan, index=range(n))
    g = pd.to_numeric(g, errors="coerce") if g is not None else pd.Series(np.nan, index=range(n))
    return ((np.asarray(r, dtype=float) < 0) & (np.asarray(g, dtype=float) < 0)) if n else np.array([], dtype=bool)


def open_lite_ts():
    """OPEN FRENZY_LITE LONG positions in the NEWEST export (by mtime) → [(opened_at, 'OPEN <pair>')]; no usable export → []."""
    fs = sorted(glob.glob(EXPORT_GLOB), key=os.path.getmtime)
    if not fs:
        return []
    try:
        d = pd.read_csv(fs[-1], low_memory=False, usecols=lambda c: c in COLS)
    except Exception:
        return []
    if not {"opened_at", "status", "direction", "entry_strategy", "pair"} <= set(d.columns):
        return []
    d = d[(d.status.astype(str).str.upper() == "OPEN") & (d.direction.astype(str) == "LONG") & (d.entry_strategy.astype(str) == STRATEGY)]
    t = pd.to_datetime(d.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    return [(a, f"OPEN {p}") for a, p in zip(t, d.pair.astype(str)) if pd.notna(a)]


# ─────────────────────────── S1 dd30 (study parity) ───────────────────────────
def sig_bar(t_ms):
    """open time of the last 5m bar CLOSED at / before t (study: entry_ts = bar_open + 5 min)."""
    return (np.asarray(t_ms, dtype="int64") // M5) * M5 - M5


def dd30_at(k, bo):
    """k = 5m bars (open_time, h, c), unique + sorted; bo = signal-bar opens → dd30 % (NaN unless all WIN_BARS bars are present).
    = study bar_feats: (1 − c[i] / h[i−5 : i+1].max()) × 100."""
    bo = np.asarray(bo, dtype="int64")
    out = np.full(len(bo), np.nan)
    if not len(k) or not len(bo):
        return out
    t = k.open_time.values.astype("int64")
    h, c = k.h.values.astype(float), k.c.values.astype(float)
    i = np.searchsorted(t, bo)
    for j, (ii, b) in enumerate(zip(i, bo)):
        if ii >= len(t) or t[ii] != b or ii < WIN_BARS - 1 or t[ii - WIN_BARS + 1] != b - (WIN_BARS - 1) * M5:
            continue                                   # signal bar missing, or a hole inside the 30-min window
        hi = h[ii - WIN_BARS + 1:ii + 1].max()
        out[j] = (1 - c[ii] / hi) * 100
    return out


def _read_hc(path):
    try:
        d = pd.read_csv(path, usecols=["open_time", "h", "c"])
        for x in ("open_time", "h", "c"):
            d[x] = pd.to_numeric(d[x], errors="coerce")
        d = d.dropna()
        d["open_time"] = d.open_time.astype("int64")
        return d
    except Exception:
        return pd.DataFrame({"open_time": pd.Series(dtype="int64"), "h": pd.Series(dtype=float), "c": pd.Series(dtype=float)})


def load_k(sym):
    fr = [_read_hc(p) for p in (os.path.join(_CACHE, "k5m_full", f"{sym}.csv"), os.path.join(MY_CACHE, "5m", f"{sym}.csv"))
          if os.path.exists(p)]
    fr = [f for f in fr if len(f)]
    if not fr:
        return _read_hc("")
    return pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)


def score(o):
    dd = np.full(len(o), np.nan)
    for sym, idx in o.groupby(o.pair.astype(str)).indices.items():
        dd[idx] = dd30_at(load_k(sym), o.bo.values[idx])
    return dd


def group(dd):
    d = np.asarray(dd, dtype=float)
    return np.where(np.isnan(d), "unscored", np.where(d >= OFF_PCT, "zone", "rest"))


# ─────────────────────────── Binance ───────────────────────────
def _http_get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode()), r.headers


def _fetch(sym, start_ms, limit, now_ms, timeout_s=10.0):
    """ONE public USDⓈ-M 5m klines request in a daemon thread with a hard wall-clock join → (CLOSED bars (open_time, h, c), used weight)."""
    if _NET_BLOCKED:
        raise RuntimeError("network blocked (selftest)")
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=5m&startTime={int(start_ms)}&limit={int(limit)}"
    box = {}

    def _go():
        try:
            box["ok"] = _http_get(url, max(1.0, timeout_s))
        except BaseException as e:
            box["err"] = e
    th = threading.Thread(target=_go, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError(f"{sym} 5m fetch exceeded {timeout_s:.0f} s wall clock")
    e = box.get("err")
    if isinstance(e, urllib.error.HTTPError) and e.code in (418, 429):
        secs = NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now_ms)     # the SHARED file (never shortens a live ban)
        raise NF.RateLimited(f"Binance {e.code} — stopped, cooldown {secs} s")
    if e is not None:
        raise e
    rows, headers = box["ok"]
    used = int((headers or {}).get("X-MBX-USED-WEIGHT-1M") or 0)
    d = pd.DataFrame([(int(x[0]), float(x[2]), float(x[4])) for x in rows if int(x[0]) + M5 <= now_ms], columns=["open_time", "h", "c"])
    return d, used


def _append(sym, new):
    os.makedirs(os.path.join(MY_CACHE, "5m"), exist_ok=True)
    p = os.path.join(MY_CACHE, "5m", f"{sym}.csv")
    d = pd.concat([_read_hc(p), new]).drop_duplicates("open_time", keep="last").sort_values("open_time")
    tmp = f"{p}.{os.getpid()}.tmp"
    d.to_csv(tmp, index=False)
    os.replace(tmp, p)


def _att_path():
    return os.path.join(MY_CACHE, "attempts.json")


FAIL_MAX = 3                                        # failed / empty attempts on one page before the fill is UNSCORED for good


def load_attempts():
    """{'done': {'SYM|start': final marker}, 'fail': {'SYM|start': failed-or-empty attempts}}. done markers: int ≥ 1 = bars returned
    (window still incomplete), 'HTTP <code>' = a 4xx other than 418/429, 'empty×3' / 'failed×3' = FAIL_MAX attempts. Never re-requested."""
    try:
        with open(_att_path()) as f:
            a = json.load(f)
        return dict(done=dict(a.get("done") or {}), fail=dict(a.get("fail") or {}))
    except Exception:
        return dict(done={}, fail={})


def save_attempts(att):
    os.makedirs(MY_CACHE, exist_ok=True)
    tmp = f"{_att_path()}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(att, f)
    os.replace(tmp, _att_path())


def _fail(att, key, what):
    """one failed / empty attempt on a page → final after FAIL_MAX."""
    att["fail"][key] = int(att["fail"].get(key, 0)) + 1
    if att["fail"][key] >= FAIL_MAX:
        att["done"][key] = f"{what}×{FAIL_MAX}"


def _final_txt(v):
    if isinstance(v, str) and v.startswith("HTTP"):
        return f"Binance answered {v} (permanent client error)"
    if isinstance(v, str):
        return f"{v.split('×')[0]} {FAIL_MAX} times"
    return "already fetched — Binance has no complete 30-min window there"


def _page(bo):
    return int(bo) - (WIN_BARS - 1) * M5, WIN_BARS


def _key(sym, start):
    return f"{sym}|{int(start)}"


def final_reasons(cnt, done=None):
    """→ Series index → reason ('' = still a cache lag) over the counted unscored fills."""
    done = load_attempts()["done"] if done is None else done
    out = {}
    for r in cnt[np.isnan(cnt.dd30.values.astype(float))].itertuples():
        st, _ = _page(r.bo)
        k = _key(r.pair, st)
        out[r.Index] = f"{r.pair} 5m page from {pd.Timestamp(st, unit='ms'):%m-%d %H:%M}: {_final_txt(done[k])}" if k in done else ""
    return pd.Series(out, dtype=object)


def ensure_data(cnt, now_ms, fetch=True, budget_s=BUDGET_S, max_req=MAX_REQ):
    """cache first → ≤ max_req requests for the oldest unscored, not-final counted fills → notes."""
    notes, used, stop, n = [], 0, None, 0
    pend = cnt[np.isnan(cnt.dd30.values.astype(float))].sort_values("ts", kind="stable") if len(cnt) else cnt
    if not fetch or not len(pend):
        return notes
    cd = NF._cooldown_until()
    if cd > now_ms:
        return [f"fetching stopped: rate-limit cooldown until {pd.Timestamp(cd, unit='ms'):%m-%d %H:%M} UTC"]
    att, deadline = load_attempts(), time.monotonic() + budget_s
    done = att["done"]
    for r in pend.itertuples():
        st, lim = _page(r.bo)
        if _key(r.pair, st) in done:
            continue
        if n >= max_req:
            notes.append(f"request cap {max_req}/run reached — the rest catches up on later runs")
            break
        if used >= WEIGHT_STOP:
            stop = f"used weight {used} ≥ {WEIGHT_STOP}"
            break
        left = deadline - time.monotonic()
        if left < 1.0:
            stop = "wall-clock budget spent"
            break
        n += 1
        k = _key(r.pair, st)
        try:
            d, used = _fetch(r.pair, st, lim, now_ms, timeout_s=min(10.0, left))
            if len(d):                                         # a page with ≥ 1 bar is final; an empty one retries (FAIL_MAX → final)
                _append(r.pair, d)
                done[k] = len(d)
            else:
                _fail(att, k, "empty")
            save_attempts(att)
            notes.append(f"{r.pair} 5m +{len(d)} bars (weight used {used})")
        except NF.RateLimited as e:
            stop = str(e)
            break
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:                            # permanent client error (bad / delisted symbol) → final now, slot refunded
                done[k] = f"HTTP {e.code}"
                n -= 1
            else:
                _fail(att, k, "failed")
            save_attempts(att)
            notes.append(f"{r.pair} 5m fetch failed (HTTP {e.code})")
        except Exception as e:
            notes.append(f"{r.pair} 5m fetch failed ({str(e)[:80]})")
            if isinstance(e, TimeoutError):                    # a hang is the run's budget, not the page's fault → not counted
                stop = "wall-clock timeout"
                break
            _fail(att, k, "failed")
            save_attempts(att)
    if stop:
        notes.append(f"fetching stopped: {stop}")
    return notes


# ─────────────────────────── verdict / freeze ───────────────────────────
def _be(sc):
    be = NF.breakeven_wr(sc.pct) if len(sc) >= BE_MIN_FILLS else None
    return (be, "live") if be is not None else (BE_REF, "fallback")


def verdict(zone, rest, be):
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz, wr = float(zone.pct.mean()), 100.0 * float((zone.pct > 0).mean())
    mr = float(rest.pct.mean()) if len(rest) else float("nan")
    p = NF.p_mean_neg(zone.pct.values, zone.day.values, BOOT_N, BOOT_SEED)
    sd, sp = NF.gross_loss_share(zone.pct.values, zone.day.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {wr:.0f} % vs breakeven {be:.1f} % · mean {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · "
           f"P(mean<0) {(p if p is not None else float('nan')):.2f} · top day {sd * 100:.0f} % / top pair {sp * 100:.0f} % of the gross loss")
    if len(rest) and mz >= mr:
        return "RETIRE", det
    if wr < be and p is not None and p >= 0.95 and sd < 0.5 and sp < 0.5:
        return "FILTER CANDIDATE (operator decides)", det
    return "KEEP OBSERVING", det


def crossing_prefix(sc):
    """scored sleeve sorted by (open time, pair) → the POSITIONAL prefix ending at the zone fill where the zone first reaches N ≥ N_MIN
    on ≥ DAYS_MIN days (iloc[:k], like NF.crossing_prefix — exactly N_MIN zone fills; ties after it in the sort order are out) or None."""
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


def freeze(st, sc, now_iso, hold=None, why=None):
    """sc = scored counted fills (grp zone / rest). Freezes the first-crossing verdict ONCE into st['first']; never recomputed."""
    if "first" in st:
        return st, False
    pre = crossing_prefix(sc)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if NF.freeze_hold(last, hold, why):
        return st, False
    be, src = _be(pre)
    zone, rest = pre[pre.grp == "zone"], pre[pre.grp == "rest"]
    state, det = verdict(zone, rest, be)
    st["first"] = dict(state=state, detail=det, be=round(be, 2), be_src=src, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso,
                       n=len(zone), days=int(zone.day.nunique()),
                       keys=[f"{a}|{b}|{g}" for a, b, g in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str), pre.grp)])
    return st, True


# ─────────────────────────── rendering ───────────────────────────
HDR = ["| Group | N | days | WR | avg % | Σ$ as-sized | worst |", "|---|---|---|---|---|---|---|"]


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean() * 100:.0f} % | {g.pct.mean():+.3f} % | "
            f"{g.usd.sum():+,.0f} | {g.pct.min():+.2f} % |")


def run(now_ms=None, orders=None, fetch=True, state_path=None, open_ts=None):
    """→ markdown lines (the caller wraps it in its own try)."""
    now_ms = now_ms or int(time.time() * 1000)
    state_path = state_path or STATE
    o = load_orders() if orders is None else _prep(_cohort(orders))
    if open_ts is None:
        try:
            open_ts = open_lite_ts() if orders is None else []
        except Exception:
            open_ts = []
    win = o[o.ts >= pd.Timestamp(START)]
    bear = win[win.bear.astype(bool)].copy()                     # refused by today's bearish-day block → apart, never counted
    cnt = win[~win.bear.astype(bool)].copy()
    cnt["dd30"] = score(cnt)
    bear["dd30"] = score(bear)                                    # cache only (no request is ever spent on a non-counted fill)
    notes = ensure_data(cnt, now_ms, fetch=fetch)
    if any(" bars (" in x for x in notes):                        # a request landed → rescore from the cache
        cnt["dd30"] = score(cnt)
    cnt["grp"] = group(cnt.dd30)
    fin = final_reasons(cnt)
    lag = fin[fin == ""].index
    sc = cnt[cnt.grp != "unscored"]
    be, src = _be(sc)
    zone, rest = sc[sc.grp == "zone"], sc[sc.grp == "rest"]
    why = []
    hold = list(open_ts) + [(cnt.loc[i, "ts"], f"cache-lagging {cnt.loc[i, 'pair']}") for i in lag]
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok:
        st, changed = freeze(st, sc, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), hold, why)
        if changed:
            NF.du_save_state(st, state_path)
    live_state, live_det = verdict(zone, rest, be)
    L = ["## 🪶 LITE_OFF30H3 — FRENZY_LITE bought ≥ 3 % below its 30-min high (pre-registered 2026-10-08, DECISION_LOG 259, OBSERVE only, block side)", "",
         f"Zone (frozen): signal close ≥ {OFF_PCT:g} % below the max 5m HIGH of the {WIN_BARS} bars ending at the signal bar (signal bar = last "
         f"5m bar closed at/before the fill; study S1 dd30). Counted: CLOSED {STRATEGY} fills opened ≥ {START} UTC that today's bearish-day "
         f"block would admit (missing stamp = admitted); DAY units; a fill whose "
         f"30-min window can't be read = UNSCORED. Reference: {REF}.", "", *HDR,
         _row(f"**zone (dd30 ≥ {OFF_PCT:g} %)**", zone), _row(f"rest (dd30 < {OFF_PCT:g} %)", rest),
         _row("UNSCORED (30-min high unreadable)", cnt[cnt.grp == "unscored"]),
         _row("pre-block bearish fills, not counted (BTC 1d ret < 0 ∧ trend gap < 0)", bear), ""]
    fmt = lambda r: (f"{r.ts:%m-%d %H:%M} {r.pair} {r.pct:+.2f} %" + (f" [{r.grp}]" if hasattr(r, "grp") else "")
                     + (f" (dd30 {r.dd30:.2f} %)" if not np.isnan(r.dd30) else "")
                     + (f" ⚠ late fill +{r.late_s:.0f} s after the bar close" if r.late_s > LATE_S else ""))
    if len(cnt):
        L += ["Counted fills: " + " · ".join(fmt(r) for r in cnt.sort_values(["ts", "pair"], kind="stable").itertuples()), ""]
    if len(bear):
        L += ["Not counted — today's FRENZY bearish-day block (DECISION_LOG 250) would refuse them: "
              + " · ".join(fmt(r) for r in bear.sort_values(["ts", "pair"], kind="stable").itertuples()), ""]
    if len(cnt) and (cnt.late_s > LATE_S).any():
        L += [f"⚠ {int((cnt.late_s > LATE_S).sum())} counted fill(s) opened > {LATE_S} s after their 5m bar close — the engine may have judged "
              "a different bar than the study convention (still counted).", ""]
    if (fin != "").any():
        L.append("UNSCORED for good (not a lag, never blocks the freeze): " + " · ".join(
            f"{cnt.loc[i, 'ts']:%m-%d %H:%M} {cnt.loc[i, 'pair']} — {w}" for i, w in fin[fin != ""].items()))
    if len(lag):
        L.append(f"⏳ {len(lag)} counted fill(s) UNSCORED only because the kline cache lags — re-scored next run (a lagging fill defers a "
                 f"freeze only if opened ≤ the crossing prefix's last zone fill).")
    L.append(f"Live so far: {len(cnt)} counted fills, {len(zone)} in the zone — far below the review bar (N ≥ {N_MIN} zone fills on ≥ "
             f"{DAYS_MIN} days); information only, no read at this N.")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    bar = (f"Bar (frozen once at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} days, never re-fit): zone mean ≥ rest "
           f"mean → RETIRE; zone WR < LITE breakeven {be:.1f} % ({src}, {len(sc)} scored fills) ∧ day-clustered P(mean < 0) ≥ 0.95 "
           f"({BOOT_N:,} resamples, seed {BOOT_SEED}) ∧ no day / pair ≥ 50 % of the gross loss → FILTER CANDIDATE (operator decides); else "
           f"KEEP OBSERVING.")
    if "first" not in st:
        L.append(f"{bar} Now: ⏳ {live_state} ({live_det}).")
    else:
        f0 = st["first"]
        L += [f"**Frozen verdict (first crossing at fill {f0['at']}, frozen on the run of {f0.get('run_at', '?')}, zone N {f0['n']} · "
              f"{f0['days']} d, breakeven {f0['be']} % {f0['be_src']}): {f0['state']}** ({f0['detail']})",
              f"Live (information only, never re-decides): {live_state} — {live_det}"]
    return L + [REVERT] + ([f"Data: {' · '.join(notes)}."] if notes else []) + [""]


# ─────────────────────────── study parity ───────────────────────────
def validate_vs_study():
    """recompute dd30 on the study CSV's stored rows from k5m_full only (this line's fetched cache NOT read) →
    (n, n_ok ≤ 1e-6, max_abs, n_unscored) or None when the CSV / cache is absent."""
    global MY_CACHE
    if not (os.path.exists(STUDY_CSV) and os.path.isdir(os.path.join(_CACHE, "k5m_full"))):
        return None
    S = pd.read_csv(STUDY_CSV, usecols=["pair", "bar_open", "dd30"])
    S = S[[os.path.exists(os.path.join(_CACHE, "k5m_full", f"{p}.csv")) for p in S.pair]]
    if not len(S):
        return None
    got = np.full(len(S), np.nan)
    saved = MY_CACHE
    with tempfile.TemporaryDirectory() as td:
        try:
            MY_CACHE = td
            for sym, idx in S.groupby("pair").indices.items():
                got[idx] = dd30_at(load_k(sym), S.bar_open.values[idx].astype("int64"))
        finally:
            MY_CACHE = saved
    err = np.abs(got - S.dd30.values)
    n_uns = int(np.isnan(got).sum())
    err = np.where(np.isnan(err), np.inf, err)
    return len(S), int((err <= 1e-6).sum()), float(err.max()), n_uns


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    global _NET_BLOCKED, _CACHE, MY_CACHE, STATE, EXPORT_GLOB, _http_get
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved = (_CACHE, MY_CACHE, STATE, EXPORT_GLOB, _http_get, NF.SHARED_COOLDOWN, NF.DU_COOLDOWN, NF._CACHE)
    _NET_BLOCKED = True
    try:
        chk((OFF_PCT, WIN_BARS, N_MIN, DAYS_MIN, BE_REF, BE_MIN_FILLS, BOOT_N, BOOT_SEED, MAX_REQ, WEIGHT_STOP, BUDGET_S, START)
            == (3.0, 6, 30, 15, 49.9, 30, 4000, 7, 2, 900, 25.0, "2026-10-07 00:00"), "pre-registered constants pinned")
        try:
            _fetch("BTCUSDT", 0, 1, 0)
            chk(False, "fetch must be blocked")
        except RuntimeError as e:
            chk("blocked" in str(e), "network blocked inside the selftest")
        t = pd.Timestamp("2026-10-08 21:40:08").value // 10**6
        chk(int(sig_bar([t])[0]) == pd.Timestamp("2026-10-08 21:35").value // 10**6, "signal bar of a 21:40:08 fill = the 21:35 bar")
        chk(int(sig_bar([pd.Timestamp("2026-10-08 21:40:00").value // 10**6])[0]) == pd.Timestamp("2026-10-08 21:35").value // 10**6,
            "a fill exactly at a bar close judges the bar that just closed")
        # dd30 = window of 6 bars INCLUDING the signal bar; the 7th bar back is outside
        b0 = 1_000 * M5
        k = pd.DataFrame(dict(open_time=b0 + np.arange(10) * M5, h=[200.0, 100, 100, 110, 103, 100, 100, 100, 100, 101],
                              c=[100.0] * 9 + [97.0]))
        bo = b0 + 9 * M5
        chk(abs(dd30_at(k, [bo])[0] - (1 - 97 / 103) * 100) < 1e-12, "dd30 = (1 − c / max h of bars i−5..i) × 100 (bar i−6 = 110 excluded)")
        k2 = k.copy()
        k2.loc[4, "h"] = 100.0
        chk(abs(dd30_at(k2, [bo])[0] - (1 - 97 / 101) * 100) < 1e-12, "the signal bar's own high counts")
        chk(np.isnan(dd30_at(k.drop(index=6), [bo])[0]) and np.isnan(dd30_at(k, [bo + M5])[0]) and np.isnan(dd30_at(k, [b0 + 4 * M5])[0]),
            "a hole in the window / a missing signal bar / < 6 bars of history → UNSCORED")
        chk(list(group([3.0, 2.9999, np.nan, 5.0])) == ["zone", "rest", "unscored", "zone"], "zone = dd30 ≥ 3 (inclusive), NaN unscored")
        # verdicts
        days = [f"2026-10-{1 + i // 2:02d}" if i < 30 else "2026-11-01" for i in range(40)]
        z = pd.DataFrame(dict(ts=[pd.Timestamp("2026-10-01 01:00") + pd.Timedelta(hours=12 * i) for i in range(40)],
                              pct=[-3.0, 3.0, -3.0] * 13 + [-3.0], day=[f"D{i}" for i in range(40)], pair=[f"P{i}" for i in range(40)],
                              usd=0.0, grp="zone"))
        r = z.assign(grp="rest", pct=[3.0, 3.0, -3.0] * 13 + [3.0], ts=z.ts + pd.Timedelta(minutes=1), pair=[f"R{i}" for i in range(40)])
        chk(verdict(z.iloc[:29], r, 49.9)[0] == "COLLECTING" and verdict(z.iloc[:30].assign(day="D1"), r, 49.9)[0] == "COLLECTING",
            "N < 30 or < 15 days → collecting")
        bad = z.assign(pct=[-3.0, -3.0, 3.0, -3.0] * 10)
        chk(verdict(bad, r, 49.9)[0] == "FILTER CANDIDATE (operator decides)", f"bad zone → candidate ({verdict(bad, r, 49.9)})")
        chk(verdict(bad, r.assign(pct=-3.0), 49.9)[0] == "RETIRE", "zone mean ≥ rest mean → RETIRE (outranks the bar)")
        chk(verdict(bad, r, 20.0)[0] == "KEEP OBSERVING", "WR above breakeven → keep observing")
        one = bad.assign(day=np.where(np.arange(40) == 0, "D0", np.array([f"D{i}" for i in range(40)])),
                         pct=np.where(np.arange(40) == 0, -500.0, bad.pct))
        chk(verdict(one, r, 49.9)[0] == "KEEP OBSERVING", "one day / pair ≥ 50 % of the gross loss → no candidate")
        # freeze: once, at the first crossing, breakeven of the scored fills ≤ the prefix end, hold defers
        sc = pd.concat([bad, r], ignore_index=True)
        st, ch = freeze({}, sc, "t1")
        last = bad.sort_values("ts").ts.iloc[29]
        chk(ch and st["first"]["n"] == 30 and st["first"]["at"] == f"{last:%Y-%m-%d %H:%M} UTC" and st["first"]["be_src"] == "live"
            and len(st["first"]["keys"]) == len(sc[sc.ts <= last]), f"first crossing frozen on its prefix ({st['first']['n']}, {st['first']['at']})")
        chk(abs(st["first"]["be"] - round(NF.breakeven_wr(sc[sc.ts <= last].pct), 2)) < 1e-9, "breakeven from scored fills ≤ prefix end")
        st2, ch2 = freeze(json.loads(json.dumps(st)), sc.assign(pct=-sc.pct), "t2")
        chk(not ch2 and st2 == json.loads(json.dumps(st)), "a frozen verdict is never recomputed")
        why = []
        sh, chh = freeze({}, sc, "th", [(sc.ts.min(), "OPEN Q")], why)
        chk(not chh and sh == {} and "OPEN Q" in why[0], "an OPEN / lagging fill opened ≤ the prefix end defers the freeze")
        sh, chh = freeze({}, sc, "th", [(pd.Timestamp("2027-01-01"), "OPEN Z")], [])
        chk(chh, "… one opened after the prefix end does not")
        zs = bad.sort_values("ts", kind="stable").reset_index(drop=True)
        tie = pd.concat([zs.assign(ts=np.where(np.arange(40) == 30, zs.ts.iloc[29], zs.ts)),
                         r.iloc[:1].assign(ts=zs.ts.iloc[29], pair="ZZREST")], ignore_index=True)
        stt, _ = freeze({}, tie, "tt")
        chk(stt["first"]["n"] == 30 and len(stt["first"]["keys"]) == 30 and not any("ZZREST" in x or "|P30|" in x for x in stt["first"]["keys"]),
            f"positional prefix: exactly 30 zone fills, ties at the crossing second sorted after it are out ({stt['first']['n']}, {len(stt['first']['keys'])})")
        small = pd.concat([bad, r.iloc[:0]], ignore_index=True).iloc[:30]
        chk(_be(small)[1] == "live" and _be(small.iloc[:29])[1] == "fallback" and _be(small.iloc[:29])[0] == 49.9,
            "breakeven fallback 49.9 % until 30 scored fills")
        with tempfile.TemporaryDirectory() as td:
            _CACHE = os.path.join(td, "cache")
            MY_CACHE = os.path.join(_CACHE, "scout_lite_off30")
            STATE = os.path.join(td, "SCOUT_LITE_OFF30H3.json")
            NF._CACHE = _CACHE
            NF.SHARED_COOLDOWN = os.path.join(_CACHE, ".binance_cooldown_until")
            NF.DU_COOLDOWN = os.path.join(_CACHE, "scout_dailyup", ".ratelimited_until")
            os.makedirs(os.path.join(_CACHE, "k5m_full"))
            dl = os.path.join(td, "dl")
            os.makedirs(dl)
            EXPORT_GLOB = os.path.join(dl, "scalpars_orders_paper_*.csv")
            # exports: older file has SKL OPEN, newer has it CLOSED (newest wins); a non-LITE row, a pre-START row, a SHORT row
            base = dict(direction="LONG", entry_strategy="FRENZY_LITE", status="CLOSED")
            old = pd.DataFrame([dict(base, opened_at="2026-10-08T21:40:08", pair="SKLUSDT", status="OPEN", pnl_percentage=0.0, pnl=0.0)])
            new = pd.DataFrame([dict(base, opened_at="2026-10-08T21:40:08.123", pair="SKLUSDT", pnl_percentage=-3.0, pnl=-127.0,
                                     entry_btc_1d_ret_pct=-2.6, entry_btc_trend_gap_pct=0.12),
                                dict(base, opened_at="2026-10-07T08:00:08", pair="BEARUSDT", pnl_percentage=-3.0, pnl=-100.0,
                                     entry_btc_1d_ret_pct=-0.24, entry_btc_trend_gap_pct=-0.18),
                                dict(base, opened_at="2026-10-07T19:56:30", pair="METUSDT", pnl_percentage=3.0, pnl=76.0),
                                dict(base, opened_at="2026-10-07T05:45:12", pair="NOKUSDT", pnl_percentage=2.0, pnl=61.0),
                                dict(base, opened_at="2026-10-06T05:45:12", pair="OLDUSDT", pnl_percentage=2.0, pnl=61.0),
                                dict(base, opened_at="2026-10-07T06:00:00", pair="MOMUSDT", entry_strategy="MOMENTUM", pnl_percentage=1.0, pnl=9.0),
                                dict(base, opened_at="2026-10-07T07:00:00", pair="SHTUSDT", direction="SHORT", pnl_percentage=1.0, pnl=9.0),
                                dict(base, opened_at="2026-10-08T23:00:00", pair="OPNUSDT", status="OPEN", pnl_percentage=np.nan, pnl=np.nan)])
            fo, fn = os.path.join(dl, "scalpars_orders_paper_a.csv"), os.path.join(dl, "scalpars_orders_paper_b.csv")
            old.to_csv(fo, index=False)
            new.to_csv(fn, index=False)
            os.utime(fo, (1, 1))
            o = load_orders()
            chk(sorted(o.pair) == ["BEARUSDT", "METUSDT", "NOKUSDT", "OLDUSDT", "SKLUSDT"] and list(o.bear) == [False, True, False, False, False]
                and float(o[o.pair == "SKLUSDT"].pct.iloc[0]) == -3.0,
                f"loader: CLOSED FRENZY_LITE LONG, dedup newest export wins ({sorted(o.pair)})")
            chk([lab for _, lab in open_lite_ts()] == ["OPEN OPNUSDT"], "open LITE positions read from the newest export")
            fc = os.path.join(dl, "scalpars_orders_paper_c.csv")          # a NEWER export still showing MET OPEN (stale row)
            pd.DataFrame([dict(base, opened_at="2026-10-07T19:56:30", pair="METUSDT", status="OPEN", pnl_percentage=0.0, pnl=0.0)]).to_csv(fc, index=False)
            os.utime(fc, (os.path.getmtime(fn) + 10,) * 2)
            o = load_orders()
            chk(len(o[o.pair == "METUSDT"]) == 1 and float(o[o.pair == "METUSDT"].pct.iloc[0]) == 3.0,
                "a CLOSED row beats an OPEN one for the same fill whatever the export mtime")
            os.remove(fc)
            # caches: SKL in the zone (3.93 % below), MET in the rest; NOK has no cache
            def bars(sym, t_fill, hs, c_last):
                b = int(sig_bar([pd.Timestamp(t_fill).value // 10**6])[0])
                pd.DataFrame(dict(open_time=b - np.arange(len(hs))[::-1] * M5, o=1.0, h=hs, l=1.0, c=[1.0] * (len(hs) - 1) + [c_last],
                                  vol=1.0, qvol=1.0)).to_csv(os.path.join(_CACHE, "k5m_full", f"{sym}.csv"), index=False)
            bars("SKLUSDT", "2026-10-08 21:40:08", [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1.0 - 0.0393)
            bars("METUSDT", "2026-10-07 19:56:30", [1.0] * 7, 0.99)
            out = "\n".join(run(1791490000000, orders=pd.concat([old, new]), fetch=False, state_path=STATE, open_ts=[]))
            chk("| **zone (dd30 ≥ 3 %)** | 1 |" in out and "| rest (dd30 < 3 %) | 1 |" in out and "| UNSCORED (30-min high unreadable) | 1 |" in out
                and "| pre-block bearish fills, not counted (BTC 1d ret < 0 ∧ trend gap < 0) | 1 |" in out
                and "Not counted — today's FRENZY bearish-day block (DECISION_LOG 250) would refuse them: 10-07 08:00 BEARUSDT" in out
                and "SKLUSDT -3.00 % [zone] (dd30 3.93 %)" in out and "1 counted fill(s) UNSCORED only because" in out
                and "METUSDT +3.00 % [rest] (dd30 1.00 %) ⚠ late fill +90 s after the bar close" in out and "SKLUSDT -3.00 % [zone] (dd30 3.93 %) ⚠" not in out
                and "1 counted fill(s) opened > 60 s" in out and "MOMUSDT" not in out and "SHTUSDT" not in out and "OPNUSDT" not in out
                and "COLLECTING" in out and not os.path.exists(STATE),
                f"end-to-end via run(orders=raw exports): SKL zone, MET rest + late flag, NOK lag, BEAR apart, OLD pre-start, non-LITE / SHORT / OPEN filtered\n{out}")
            chk(list(bearish_day(pd.Series([-1.0, -1.0, 1.0, np.nan, -1.0, 0.0]), pd.Series([-1.0, 1.0, -1.0, -1.0, np.nan, -1.0]), 6))
                == [True, False, False, False, False, False], "bearish day = 1d ret < 0 ∧ trend gap < 0 (strict); a missing stamp fails open")
            empty = pd.DataFrame(columns=list(COLS))
            out = "\n".join(run(1791490000000, orders=empty, fetch=False, state_path=STATE, open_ts=[]))
            chk("COLLECTING" in out and "| **zone (dd30 ≥ 3 %)** | 0 |" in out, "empty orders → collecting, never raises")
            # fetch path: NOK has no cache → one 6-bar page; 429 → shared cooldown, next run zero requests
            calls = []

            def _boom(url, timeout):
                calls.append(url)
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "120"}, None)
            _NET_BLOCKED, _http_get = False, _boom
            cnt = load_orders()
            cnt = cnt[(cnt.ts >= pd.Timestamp(START)) & ~cnt.bear.astype(bool)].copy()
            cnt["dd30"] = score(cnt)
            notes = ensure_data(cnt, 1791490000000)
            chk(len(calls) == 1 and "NOKUSDT" in calls[0] and "limit=6" in calls[0] and any("429" in x for x in notes)
                and abs(NF._cooldown_until() - (1791490000000 + 120_000)) < 2, f"429 → one request + SHARED cooldown ({calls}, {notes})")
            notes = ensure_data(cnt, 1791490000000 + 60_000)
            chk(len(calls) == 1 and any("cooldown" in x for x in notes), "cooldown live → zero requests")
            os.remove(NF.SHARED_COOLDOWN)
            os.makedirs(os.path.dirname(NF.DU_COOLDOWN), exist_ok=True)
            open(NF.DU_COOLDOWN, "w").write(str(1791490000000 + 999_999))
            ensure_data(cnt, 1791490000000)
            chk(len(calls) == 1, "another scout line's (legacy) cooldown is honoured")
            os.remove(NF.DU_COOLDOWN)

            def _w(url, timeout):
                calls.append(url)
                return [], {"X-MBX-USED-WEIGHT-1M": "950"}
            _http_get = _w
            many = pd.concat([cnt[cnt.pair == "NOKUSDT"].assign(pair=f"X{i}USDT") for i in range(5)], ignore_index=True)
            notes = ensure_data(many, 1791490000000)
            chk(len(calls) == 2 and any("used weight" in x for x in notes), f"used weight ≥ {WEIGHT_STOP} → no further request")

            def _ok2(url, timeout):
                calls.append(url)
                s_ = int(dict(x.split("=") for x in url.split("?")[1].split("&"))["startTime"])
                return [[s_, "1", "1.0", "1.0", "1.0"]], {"X-MBX-USED-WEIGHT-1M": "5"}
            chk("X0USDT|" + str(int(many.bo.iloc[0]) - 5 * M5) not in load_attempts()["done"], "an EMPTY page is not recorded as done")
            _http_get = _ok2
            notes = ensure_data(many, 1791490000000)
            chk(len(calls) == 2 + MAX_REQ and "X0USDT" in calls[2] and any("request cap" in x for x in notes),
                f"request cap {MAX_REQ}/run; X0's empty page is retried ({calls[2:]})")
            notes = ensure_data(many, 1791490000000)
            chk(len(calls) == 2 + 2 * MAX_REQ and "X0USDT" not in "".join(calls[4:]) and "X1USDT" not in "".join(calls[4:]),
                f"a page with ≥ 1 bar is never re-requested ({calls[4:]})")
            fr = final_reasons(many.assign(dd30=np.nan).iloc[:1])
            chk(len(fr) == 1 and "already fetched" in fr.iloc[0], "a non-empty page that leaves the window incomplete → UNSCORED for good")
            # a real page: NOK priced from this line's own cache
            tN = int(cnt[cnt.pair == "NOKUSDT"].bo.iloc[0])

            def _kl(url, timeout):
                calls.append(url)
                q = dict(x.split("=") for x in url.split("?")[1].split("&"))
                s = int(q["startTime"])
                return [[s + i * M5, "1", "1.10" if i == 2 else "1.0", "0.9", "1.0"] for i in range(int(q["limit"]))], {"X-MBX-USED-WEIGHT-1M": "3"}
            _http_get = _kl
            n0 = len(calls)
            notes = ensure_data(cnt, tN + 10 * M5)
            got = dd30_at(load_k("NOKUSDT"), [tN])[0]
            chk(len(calls) - n0 == 1 and f"startTime={tN - 5 * M5}" in calls[-1] and abs(got - (1 - 1 / 1.1) * 100) < 1e-9,
                f"fetch: one 6-bar page ending at the signal bar → scored from this line's cache ({calls[n0:]}, {got})")
            ensure_data(cnt.assign(dd30=score(cnt)), tN + 10 * M5)
            chk(len(calls) - n0 == 1, "covered → zero requests")
            # the forming bar is never stored
            d, _ = _fetch("NOKUSDT", tN - 5 * M5, 6, tN + M5 - 1)
            chk(int(d.open_time.max()) == tN - M5, "only CLOSED bars are kept")

            # permanently failing pages: a 4xx is final at once (slot refunded, the queue moves on); 3 empty pages → final
            tmpl = cnt[cnt.pair == "NOKUSDT"].iloc[:1]
            q = pd.concat([tmpl.assign(pair="Q0USDT", dd30=np.nan),
                           tmpl.assign(pair="Q1USDT", dd30=np.nan, ts=tmpl.ts + pd.Timedelta(hours=1), bo=tmpl.bo + 12 * M5)], ignore_index=True)

            def _q(url, timeout):
                calls.append(url)
                if "Q0USDT" in url:
                    raise urllib.error.HTTPError(url, 400, "Invalid symbol", {}, None)
                s_ = int(dict(x.split("=") for x in url.split("?")[1].split("&"))["startTime"])
                return [[s_ + i * M5, "1", "1.0", "1.0", "1.0"] for i in range(6)], {"X-MBX-USED-WEIGHT-1M": "5"}
            _http_get = _q
            n0 = len(calls)
            ensure_data(q, 1791490000000, max_req=1)
            fr = final_reasons(q)
            chk(len(calls) - n0 == 2 and "Q1USDT" in calls[-1] and "HTTP 400" in fr.iloc[0] and not np.isnan(dd30_at(load_k("Q1USDT"), q.bo.values[1:])[0]),
                f"a 400 on the oldest fill → final, its slot refunded, the later fill is fetched ({calls[n0:]}, {fr.to_dict()})")
            ensure_data(q, 1791490000000)
            chk(len(calls) - n0 == 2, "the 400 page is never re-requested")

            def _q5(url, timeout):
                calls.append(url)
                raise urllib.error.HTTPError(url, 503, "busy", {}, None)
            _http_get = _q5
            q2 = tmpl.assign(pair="Q2USDT", dd30=np.nan)
            ensure_data(q2, 1791490000000)
            chk(final_reasons(q2).iloc[0] == "" and load_attempts()["fail"].get(_key("Q2USDT", _page(tmpl.bo.iloc[0])[0])) == 1,
                "a 5xx counts one failed attempt, not final")
            _http_get = lambda url, timeout: (calls.append(url), ([], {"X-MBX-USED-WEIGHT-1M": "5"}))[1]
            e0 = tmpl.assign(pair="E0USDT", dd30=np.nan)
            n0 = len(calls)
            for _ in range(FAIL_MAX + 1):
                ensure_data(e0, 1791490000000)
            fr = final_reasons(e0)
            chk(len(calls) - n0 == FAIL_MAX and "empty 3 times" in fr.iloc[0], f"{FAIL_MAX} empty pages → UNSCORED for good ({fr.to_dict()})")

            def _slow(url, timeout):
                calls.append(url)
                time.sleep(3)
            _http_get = _slow
            t0 = time.monotonic()
            notes = ensure_data(pd.concat([cnt.assign(pair=f"Y{i}USDT", dd30=np.nan) for i in range(3)]), 1791490000000, budget_s=1.5)
            chk(time.monotonic() - t0 < 2.5 and any("wall-clock" in x for x in notes), "hung request cut by the wall-clock budget")
            # end-to-end freeze through run(): 30 zone fills on 15 days → frozen once, persisted, a later run keeps it
            rows = []
            for i in range(32):
                ts = pd.Timestamp("2026-10-10 00:00:08") + pd.Timedelta(hours=13 * i)
                rows.append(dict(base, opened_at=f"{ts:%Y-%m-%dT%H:%M:%S}", pair=f"Z{i}USDT", pnl_percentage=[-3.0, -3.0, 3.0][i % 3], pnl=-10.0))
                bars(f"Z{i}USDT", f"{ts}", [1.0] * 6, 0.95)
            _http_get = None
            _NET_BLOCKED = True
            zo = pd.DataFrame(rows)
            out = "\n".join(run(1791490000000, orders=zo, fetch=False, state_path=STATE, open_ts=[]))
            s1 = json.load(open(STATE))
            chk("Frozen verdict" in out and s1["first"]["n"] == 30 and s1["first"]["be_src"] == "live" and s1["first"]["be"] == 50.0,
                f"run() freezes at the 30th zone fill on ≥ 15 days ({s1['first']['state']})")
            out = "\n".join(run(1791490000000, orders=zo.assign(pnl_percentage=3.0), fetch=False, state_path=STATE, open_ts=[]))
            chk(json.load(open(STATE)) == s1 and s1["first"]["state"] in out, "persisted verdict survives a later run unchanged")
    finally:
        _CACHE, MY_CACHE, STATE, EXPORT_GLOB, _http_get, NF.SHARED_COOLDOWN, NF.DU_COOLDOWN, NF._CACHE = saved
        _NET_BLOCKED = True
    v = validate_vs_study()
    if v is None:
        print("  study parity SKIPPED (reports/LITE_ENTRY_SIGNS_STUDY_2026-10-08.csv or the k5m_full cache missing — not a failure)")
    else:
        n, good, mx, nu = v
        chk(n >= 100 and good == n, f"dd30 reproduces the study on {good}/{n} stored rows (max abs err {mx:.2e}, {nu} unscored)")
        print(f"  dd30 reproduces the study CSV on {good}/{n} stored rows, max abs err {mx:.1e}")
    _NET_BLOCKED = False
    print(f"selftest LITE_OFF30H3 OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
