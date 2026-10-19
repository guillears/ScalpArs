#!/usr/bin/env python3
"""👁 Scout — LONG_STOP4 observe line (DECISION_LOG 315, operator 2026-10-19: "add LONG_STOP4 and LITE_GREEN_RETRY watch lines").
OBSERVE ONLY — never changes config, never calls the bot API.

Origin: reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19.md §A3 / §G. On the real fills 5 of the 9 FRENZY_LONG fills stopped at −3 came
back to +1.5 without touching −4 (+0.67 vs +0.02 %/fill, LODO picks −4 in 13/13 folds, Δ +0.65 [+0.08, +1.36]); the 2-year backtest
refutes it (18 % saved, Δ −0.003, 2026-H2 −0.25). At blocker size ($100) ≈ $1/day either way → observe line, not a ship.

FROZEN DEFINITION (pre-registered 2026-10-19 — never re-fit, never re-fit the −4)
  cohort   CLOSED FRENZY_LONG LONG fills opened ≥ COHORT_START (orders exports, paper + LIVE via export_glob), dedup (opened_at[:19], pair,
           direction), CLOSED beats OPEN, never id. MANUAL_* closes are excluded (listed). The 9 motivating stops (before the start) never count.
  path     the study's gross path (study_frenzy_realfills_common.gross_path, rebuilt here): from the fill (opened_at, entry_price) to +12 h,
           TICKS FIRST — tick days from the shared caches, else the data.binance.vision daily aggTrades archive (CDN, no API weight) — as
           1-SECOND bars (hi / lo / first / last print, ticks strictly after the fill second); the uncovered tail (archive not published
           yet) from public fapi 1m klines (from the first whole minute after the covered time) → PROVISIONAL (flagged; re-run later).
  walker   the study's generic first-touch walker (study_frenzy_realfills_common.walk, copied VERBATIM below so this tracked scout never
           imports an untracked study; tests/test_scout_long_stop4.py checks parity when the study file exists): gross price lines, the
           stop judged BEFORE the take-profit on the same bar (same-bar stop + TP = STOP), a bar opening through the stop fills at its open.
           Fees as the study: entry taker 0.045 %; the resting TP (DL 310) exits MAKER 0.018 % at the line; stop / cap exits taker 0.045 %
           + 0.03 % slippage. TP +1.5, 12 h cap — unchanged; only the stop moves: shadow −4 vs the same walker at today's −3.
  Δ        shadow(−4) − walker(−3), % of notional per fill; $ at the fill's notional_value (fallback $100 = LONG blocker size).
           Parity: walker(−3) vs booked pnl_percentage, sign disagreements printed.
  live-stopped = the fill's booked close_reason is a stop (contains "STOP"; no reason → pnl_percentage ≤ −2).
GATE (frozen, pre-registered 2026-10-19 — the report's §G)
  READ at ≥ 15 live-stopped fills on ≥ 8 distinct UTC days. Below → "collecting (n/15 stops on d/8 days)".
  CANDIDATE (operator decides; nothing armed) only if ALL: saved share (live-stopped fills the −4 shadow turns into TP: walker −3 STOP
    → −4 TP, over the live stops the −3 walker confirms = the study's own stopped cohort; a live stop the −3 walker misses is a replica
    artefact — printed as REVERSE PARITY, in neither numerator nor denominator) ≥ 50 %
    ∧ mean Δ over ALL cohort LONG fills > 0 with day-bootstrap P(mean > 0) ≥ 0.90 (4,000 resamples, seed 7)
    ∧ Δ = 0 on every fill the live −3 did NOT stop (walker integrity — violations printed)
    ∧ no UTC day carries > 50 % of ΣΔ.
    Next step on CANDIDATE: re-price the LITE entries a longer LONG hold would block (pair held longer) + the 30–50 % in-sample haircut.
  RETIRE if saved share ≤ 25 % (the backtest base rate is 18 %) or mean Δ ≤ 0 at the read. Otherwise "hold — keep observing".
BINANCE  aggTrades archives: ≤ MAX_ARCHIVES per run, asked only ≥ 6 h after the UTC day ended, a 404 retried ≥ 6 h later, > 600 MB skipped.
  fapi 1m: one request (weight 5) per fill tail, paced, stop once X-MBX-USED-WEIGHT-1M ≥ 700, shared scout cooldown honoured / armed on
  418-429 (binance_cooldown + scout_b1h_negflank._arm_cooldown), a 403 stops the run. ≤ MAX_FILLS uncached fills per run; a fill whose 12 h
  window is not complete (or whose data is pending) stays PENDING. Cache reports/backtest_cache/scout_long_stop4/ (ticks/, klines/).
Usage  venv/bin/python scripts/scout_long_stop4.py                 (batch review)
       venv/bin/python scripts/scout_long_stop4.py --since ISO     (DIAGNOSTIC only — an earlier cohort start, never the frozen read)
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as EG                                     # noqa: E402
import binance_cooldown as BC                                # noqa: E402
import scout_b1h_negflank as NF                              # noqa: E402  (shared cooldown writer)

COHORT_START = "2026-10-19T17:00:00"
STRATEGY = "FRENZY_LONG"
TP, SL_LIVE, SL_SHADOW, CAP = 1.5, 3.0, 4.0, 720              # frozen (TP / cap / fees unchanged; only the stop moves)
GATE_STOPS, GATE_DAYS = 15, 8
SAVED_MIN, SAVED_RETIRE, P_MIN, DAY_MAX = 0.50, 0.25, 0.90, 0.50
BOOT_N, BOOT_SEED = 4000, 7
NOTIONAL_FALLBACK = 100.0
TAKER, MAKER, SLIP, ENTRY = 0.045, 0.018, 0.03, 0.045         # the study's fee model
MIN, SEC, HOUR, DAY = 60_000, 1_000, 3_600_000, 86_400_000
MAX_FILLS, MAX_ARCHIVES, WEIGHT_STOP, PACE_S = 12, 4, 700, 0.35
ARCH_LAG_MS, ARCH_RETRY_MS, ARCH_MAX_BYTES = 6 * HOUR, 6 * HOUR, 600 * 1024 * 1024
_ROOT = os.path.dirname(_D)
_BCACHE = os.path.join(_ROOT, "reports", "backtest_cache")
CACHE = os.path.join(_BCACHE, "scout_long_stop4")
SHARED_TICK_DIRS = [os.path.join(_BCACHE, d) for d in ("ticks", "ticks_q", os.path.join("scout_willy_timecap", "ticks"),
                                                       os.path.join("scout_long_stop4", "ticks"),
                                                       os.path.join("scout_lite_green_retry", "ticks"))]
ARCHIVE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{p}/{p}-aggTrades-{d}.zip"
FAPI_URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&limit={n}"


def _ms(s):
    return int(pd.Timestamp(str(s)[:19]).value // 1_000_000)


def _num(v):
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) else np.nan


# ─────────────────────────── the study's walker (VERBATIM copy of study_frenzy_realfills_common.walk) ───────────────────────────
def walk(P, tp=np.inf, sl=3.0, b=np.inf, be=0.1, A=np.inf, G=np.inf, cap=720, tp_late=None, maker_tp=True):
    """→ (net %, why, exit minute). tp_late = (minute, tp2): from that order age the TP is tp2 (WILLY late TP)."""
    te = P["te"]; t = P["t"]
    n = int(np.searchsorted(t, te + cap * MIN, side="right"))
    if n == 0:
        return np.nan, "UNRES", np.nan
    hi, lo, op, pk = P["hi"][:n], P["lo"][:n], P["op"][:n], P["pk_prev"][:n]
    line = np.full(n, -float(sl))
    if np.isfinite(b):
        line = np.where(pk >= b, np.maximum(line, be), line)
    if np.isfinite(A):
        line = np.where(pk >= A, np.maximum(line, pk - G), line)
    if tp_late is not None:
        TPa = np.where(t[:n] - te < tp_late[0] * MIN, tp, tp_late[1])
    else:
        TPa = tp
    hl = lo <= line
    ht = hi >= TPa
    il = int(np.argmax(hl)) if hl.any() else n
    it = int(np.argmax(ht)) if ht.any() else n
    if il < n and il <= it:
        f = line[il]
        if np.isfinite(op[il]) and op[il] < f:
            f = op[il]
        why = "STOP" if line[il] <= -sl + 1e-9 else "LINE"
        return f - ENTRY - TAKER * (1 + f / 100) - SLIP, why, (t[il] - te) / MIN
    if it < n:
        f = TPa[it] if np.ndim(TPa) else TPa
        if maker_tp:
            return f - ENTRY - MAKER * (1 + f / 100), "TP", (t[it] - te) / MIN
        if np.isfinite(op[it]) and op[it] > f:
            f = op[it]
        return f - ENTRY - TAKER * (1 + f / 100), "TP", (t[it] - te) / MIN
    f = P["cl"][n - 1]
    why = "CAP" if P["end_ms"] >= te + cap * MIN - MIN else "UNRES"
    return f - ENTRY - TAKER * (1 + f / 100) - SLIP, why, (t[n - 1] - te) / MIN


def day_boot(vals, days, nb=BOOT_N, seed=BOOT_SEED):
    """(mean, lo95, hi95, P(mean > 0)) of the per-FILL mean under a day-block bootstrap (study_frenzy_realfills_common.day_boot)."""
    v, dd = np.asarray(vals, float), np.asarray(days)
    u = [x for x in pd.unique(dd)]
    groups = [v[dd == x] for x in u]
    groups = [g for g in groups if len(g)]
    k = len(groups)
    if k == 0:
        return np.nan, np.nan, np.nan, np.nan
    s = np.array([g.sum() for g in groups]); c = np.array([len(g) for g in groups])
    idx = np.random.default_rng(seed).integers(0, k, size=(nb, k))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


# ─────────────────────────── path (the study's gross_path, data sources made scout-safe) ───────────────────────────
def assemble_path(te, E, horizon, ticks=None, k1m=None):
    """PURE. ticks = (t ms, p, cov ms) — prints covering [te, cov) — or None; k1m = rows [open_ms, o, h, l, c] for the uncovered tail (only bars with
    open ≥ the first whole minute at / after the covered time and open + 1 m ≤ te + (horizon + 1) min, contiguous). Gross % vs E.
    → dict(t bar END ms, hi, lo, op, cl, pk_prev, end_ms, n_tick_s, n_bar, te, tick_end) or None (no event at all)."""
    te, E = int(te), float(E)
    lim = te + (horizon + 1) * MIN
    T, HI, LO, OP, CL = [], [], [], [], []
    cov = te
    nt = 0
    if ticks is not None:
        t_all, p_all = np.asarray(ticks[0], "int64"), np.asarray(ticks[1], float)
        cov = min(int(ticks[2]), lim)
        i, j = np.searchsorted(t_all, te, side="right"), np.searchsorted(t_all, cov, side="right")
        t, x = t_all[i:j], (p_all[i:j] / E - 1) * 100
        nt = len(t)
        if nt:
            s = (t - 1) // SEC
            brk = np.flatnonzero(np.diff(s)) + 1
            st = np.r_[0, brk]; en = np.r_[brk, nt]
            T.append((s[st] + 1) * SEC); HI.append(np.maximum.reduceat(x, st)); LO.append(np.minimum.reduceat(x, st))
            OP.append(x[st]); CL.append(x[en - 1])
    n_bar = 0
    if cov < lim and k1m is not None and len(k1m):
        k = np.asarray(k1m, float)
        first = -(-int(cov) // MIN) * MIN
        o = k[:, 0].astype("int64")
        m = (o >= first) & (o + MIN <= lim)
        k, o = k[m], o[m]
        if len(o) and o[0] == first:
            gaps = np.flatnonzero(np.diff(o) != MIN)
            if len(gaps):
                k, o = k[:gaps[0] + 1], o[:gaps[0] + 1]
            T.append(o + MIN); OP.append((k[:, 1] / E - 1) * 100); HI.append((k[:, 2] / E - 1) * 100)
            LO.append((k[:, 3] / E - 1) * 100); CL.append((k[:, 4] / E - 1) * 100)
            n_bar = len(o)
    if not T:
        return None
    P = dict(t=np.concatenate(T).astype("int64"), hi=np.concatenate(HI), lo=np.concatenate(LO), op=np.concatenate(OP),
             cl=np.concatenate(CL), te=te, n_tick_s=int(len(T[0]) if nt else 0), n_bar=int(n_bar), tick_end=int(cov))
    P["end_ms"] = int(P["t"][-1]) if n_bar else int(cov)        # the study: the 1m tail's last bar end, else the tick coverage end
    P["pk_prev"] = np.r_[-np.inf, np.maximum.accumulate(P["hi"])[:-1]]
    return P


def rebase(P, f):
    """the study's re-pricing (price_one): a gross path vs the signal close → the same path vs an entry at close × f."""
    Q = dict(P)
    for x in ("hi", "lo", "op", "cl"):
        Q[x] = ((1 + np.asarray(P[x], float) / 100) / f - 1) * 100
    Q["pk_prev"] = np.r_[-np.inf, np.maximum.accumulate(Q["hi"])[:-1]]
    return Q


class Budget:
    def __init__(self):
        self.used, self.stop, self.why, self.archives, self.requests = 0, False, "", 0, 0

    def halt(self, why):
        self.stop, self.why = True, self.why or why


def _cooling(now):
    return max(BC.read_until_ms(), NF._cooldown_until()) > now


def fapi_klines(sym, interval, start, limit, bud, cache_dir, now=None, fetch=True):
    """public fapi klines → rows [open, o, h, l, c, v] (CLOSED bars only) · None = budget / cooldown / 403 stop · [] = no data.
    Cached (JSON) only when all `limit` bars came back closed."""
    now = int(time.time() * 1000) if now is None else int(now)
    step = {"1m": MIN, "5m": 5 * MIN, "1h": HOUR, "1d": DAY}[interval]
    p = os.path.join(cache_dir, "klines", f"{sym}_{interval}_{int(start)}_{int(limit)}.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    if bud.stop or not fetch:
        return None
    if _cooling(now):
        bud.halt("shared Binance cooldown active")
        return None
    if bud.used >= WEIGHT_STOP:
        bud.halt(f"weight {bud.used} ≥ {WEIGHT_STOP}")
        return None
    time.sleep(PACE_S)
    bud.requests += 1
    try:
        with urllib.request.urlopen(FAPI_URL.format(s=sym, i=interval, a=int(start), n=int(limit)), timeout=15) as r:
            rows = json.loads(r.read().decode())
            bud.used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or bud.used)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now)
            bud.halt(f"HTTP {e.code} — shared cooldown armed")
            return None
        if e.code == 403:
            bud.halt("HTTP 403 (WAF / geo) — run stopped")
            return None
        return []
    except Exception:
        return []
    out = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in rows if int(x[0]) + step <= now]
    if len(out) == int(limit):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f)
    return out


def parse_agg_zip(path):
    """an aggTrades daily zip (agg_id, price, qty, first_id, last_id, transact_time, is_buyer_maker; header optional) → (t int64, p
    float64) sorted by time. Parsed as plain CSV only (untrusted download)."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError("no csv in archive")
        with z.open(names[0]) as fh:
            d = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), header=None, usecols=[1, 5], dtype=str)
    t = pd.to_numeric(d[5], errors="coerce"); p = pd.to_numeric(d[1], errors="coerce")
    ok = t.notna() & p.notna() & (p > 0)
    t, p = t[ok].astype("int64").values, p[ok].astype(float).values
    o = np.argsort(t, kind="stable")
    return t[o], p[o]


def _local_tick_file(pair, day, dirs=None):
    for d in (dirs or SHARED_TICK_DIRS):
        f = os.path.join(d, pair, f"{day}.npz")
        if os.path.exists(f):
            return f
    return None


def tick_day(pair, day, bud, cache_dir, now=None, fetch=True):
    """(t, p) for one full UTC day — shared caches first, else the data.binance.vision archive (cached under cache_dir/ticks).
    None = not available (yet)."""
    now = int(time.time() * 1000) if now is None else int(now)
    f = _local_tick_file(pair, day)
    if f is not None:
        try:
            with np.load(f) as z:
                if "w0" in z.files:                              # a window slice, not a full day → not used here
                    raise ValueError("slice")
                t = z["t"].astype("int64"); p = z["p"].astype(float)
            o = np.argsort(t, kind="stable")
            return t[o], p[o]
        except Exception:
            pass
    if not fetch:
        return None
    d0 = int(pd.Timestamp(day).value // 1_000_000)
    if now < d0 + DAY + ARCH_LAG_MS:
        return None                                              # archive not published yet → 1m tail (provisional)
    tdir = os.path.join(cache_dir, "ticks", pair)
    mark = os.path.join(tdir, f"{day}.miss")
    if os.path.exists(mark):
        try:
            with open(mark) as fh:
                info = fh.read().strip()
        except Exception:
            info = ""
        if info.startswith("toobig") or now - os.path.getmtime(mark) * 1000 < ARCH_RETRY_MS:
            return None
    if bud.stop or bud.archives >= MAX_ARCHIVES or _cooling(now):
        return None
    bud.archives += 1
    os.makedirs(tdir, exist_ok=True)
    tmp = os.path.join(tdir, f".{day}.{os.getpid()}.zip")
    try:
        with urllib.request.urlopen(ARCHIVE_URL.format(p=pair, d=day), timeout=180) as r:
            size = int(r.headers.get("Content-Length") or 0)
            if size > ARCH_MAX_BYTES:
                with open(mark, "w") as fh:
                    fh.write(f"toobig {size}")
                return None
            with open(tmp, "wb") as fo:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fo.write(b)
        t, p = parse_agg_zip(tmp)
        np.savez_compressed(os.path.join(tdir, f"{day}.npz"), t=t, p=p)
        return t, p
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now)
            bud.halt(f"archive HTTP {e.code} — shared cooldown armed")
        elif e.code == 403:
            bud.halt("archive HTTP 403 — run stopped")
        else:
            with open(mark, "w") as fh:
                fh.write(f"http {e.code}")
        return None
    except Exception as e:
        with open(mark, "w") as fh:
            fh.write(f"err {e!r}"[:200])
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def build_path(pair, te, E, horizon, bud, cache_dir, now=None, fetch=True):
    """ticks while consecutive full UTC days are available, then the fapi 1m tail. → (P or None, status) with status 'ok' / 'budget'."""
    now = int(time.time() * 1000) if now is None else int(now)
    te = int(te)
    lim = te + (horizon + 1) * MIN
    days = [f"{d:%Y-%m-%d}" for d in pd.date_range(pd.Timestamp(te, unit="ms").normalize(), pd.Timestamp(lim, unit="ms").normalize())]
    ts, ps, cov = [], [], te
    for d in days:
        z = tick_day(pair, d, bud, cache_dir, now, fetch)
        if z is None:
            break
        ts.append(z[0]); ps.append(z[1])
        cov = int(pd.Timestamp(d).value // 1_000_000) + DAY
    ticks = (np.concatenate(ts), np.concatenate(ps), min(cov, lim)) if ts else None
    cov = min(cov, lim) if ts else te
    k1 = None
    if cov < lim:
        first = -(-cov // MIN) * MIN
        n = int((min(lim, now) - first) // MIN)
        if n > 0:
            k1 = fapi_klines(pair, "1m", first, min(n, 1000), bud, cache_dir, now, fetch=fetch)
            if k1 is None:
                return None, "budget"
    P = assemble_path(te, E, horizon, ticks, k1)
    return P, "ok"


# ─────────────────────────── cohort, shadow, verdict ───────────────────────────
def load_orders(pattern="~/Downloads/scalpars_orders_paper_*.csv", cols=None):
    fr = []
    for f in sorted(EG.glob_exports(os.path.expanduser(pattern)), key=os.path.getmtime):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=(lambda c: c in cols) if cols else None))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame()
    a = pd.concat(fr, ignore_index=True)
    a = a.assign(_k=a.opened_at.astype(str).str[:19].str.replace(" ", "T"), _c=(a.status.astype(str) == "CLOSED").astype(int))
    return a.sort_values("_c", kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last").reset_index(drop=True)


COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "entry_price", "pnl_percentage", "pnl",
        "notional_value", "close_reason", "is_paper")


def fills(start=COHORT_START):
    a = load_orders(cols=COLS)
    if not len(a):
        return a, a
    a = a[(a.entry_strategy.astype(str) == STRATEGY) & (a.direction.astype(str).str.upper() == "LONG") & (a._k >= start)
          & (a.status.astype(str) == "CLOSED")]
    man = a.close_reason.astype(str).str.upper().str.startswith("MANUAL")
    return a[~man].sort_values("_k").reset_index(drop=True), a[man].reset_index(drop=True)


def live_stopped(close_reason, pnl_pct):
    cr = str(close_reason or "").upper()
    if cr and cr != "NAN":
        return "STOP" in cr
    return bool(np.isfinite(pnl_pct) and pnl_pct <= -2.0)


def shadow(P):
    """→ dict(base, base_why, base_min, shadow, shadow_why, shadow_min, delta) — the study walker at −3 (today) and −4 (shadow)."""
    b, bw, bm = walk(P, tp=TP, sl=SL_LIVE, cap=CAP)
    s, sw, sm = walk(P, tp=TP, sl=SL_SHADOW, cap=CAP)
    return dict(base=b, base_why=bw, base_min=bm, shadow=s, shadow_why=sw, shadow_min=sm,
                delta=(s - b) if (np.isfinite(s) and np.isfinite(b)) else np.nan)


def verdict(d):
    """d = scored fills: day, pair, live_stopped (bool), delta, shadow_why → (verdict string, detail dict)."""
    if not len(d):
        return f"collecting (0/{GATE_STOPS} stops on 0/{GATE_DAYS} days)", {}
    st = d[d.live_stopped.astype(bool)]
    n, nd = len(st), st.day.nunique()
    viol = d[~d.live_stopped.astype(bool) & (d.delta.abs() > 1e-9)]
    if n < GATE_STOPS or nd < GATE_DAYS:
        return f"collecting ({n}/{GATE_STOPS} stops on {nd}/{GATE_DAYS} days)", dict(violations=len(viol))
    sw = st[st.base_why == "STOP"]                                   # live stops the −3 walker confirms (the study's rg < −2 cohort)
    saved = float((sw.shadow_why == "TP").mean()) if len(sw) else np.nan   # the −4 TURNS a −3 stop into a TP
    mu, lo, hi, p = day_boot(d.delta.values, d.day.values)
    tot = float(d.delta.sum())
    by_day = d.groupby("day").delta.sum()
    day_share = float(by_day.max() / tot) if tot > 0 else np.nan
    det = dict(saved=saved, n_stopped=n, days=nd, mean=mu, lo=lo, hi=hi, p=p, total=tot, day=str(by_day.idxmax()), day_share=day_share,
               violations=len(viol))
    if not np.isfinite(saved):
        return "hold — keep observing (no live stop confirmed by the −3 walker)", det
    if saved <= SAVED_RETIRE or mu <= 0:
        return "RETIRE the line (frozen gate: saved share ≤ 25 % or mean Δ ≤ 0 at the read)", det
    ok = saved >= SAVED_MIN and p >= P_MIN and len(viol) == 0 and np.isfinite(day_share) and day_share <= DAY_MAX
    if ok:
        return ("CANDIDATE (operator decides) — next: re-price the LITE entries a longer LONG hold would block + 30–50 % in-sample "
                "haircut"), det
    return "hold — keep observing (read reached, CANDIDATE bar not met, RETIRE bar not hit)", det


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    start, diag = COHORT_START, False
    if "--since" in argv:
        start, diag = str(argv[argv.index("--since") + 1])[:19], True
    a, man = fills(start)
    tag = "DIAGNOSTIC (cohort start overridden — NOT the frozen read) · " if diag else ""
    print(f"LONG_STOP4 (DECISION_LOG 315, observe) · {tag}cohort CLOSED {STRATEGY} LONG fills opened ≥ {start} UTC → {len(a)}"
          f"  (MANUAL closes excluded: {len(man)})")
    print(f"  shadow: stop −{SL_SHADOW:g} vs today's −{SL_LIVE:g} · TP +{TP} resting maker · {CAP // 60} h cap · study walker + fees · "
          f"ticks first (aggTrades archive), else fapi 1m = provisional")
    if not len(a):
        print(f"  verdict: collecting (0/{GATE_STOPS} stops on 0/{GATE_DAYS} days)")
        return
    bud, rows, pend, unres, fetched = Budget(), [], [], 0, 0
    now = int(time.time() * 1000)
    for r in a.to_dict("records"):
        te, E = _ms(r["opened_at"]), _num(r["entry_price"])
        if not np.isfinite(E) or E <= 0:
            unres += 1
            continue
        n0 = bud.requests + bud.archives
        P, stt = build_path(r["pair"], te, E, CAP, bud, CACHE, now, fetch=(fetched < MAX_FILLS and not bud.stop))
        fetched += int(bud.requests + bud.archives > n0)          # MAX_FILLS counts fills that needed the network
        if P is None:
            pend.append(f"{r['_k']} {r['pair']} ({stt if stt != 'ok' else 'no data yet'})")
            continue
        s = shadow(P)
        if not (np.isfinite(s["base"]) and np.isfinite(s["shadow"])) or "UNRES" in (s["base_why"], s["shadow_why"]):
            if now < te + (CAP + 1) * MIN or P["end_ms"] < te + CAP * MIN - MIN:
                pend.append(f"{r['_k']} {r['pair']} (window not complete)")
            else:
                unres += 1
            continue
        bk = _num(r.get("pnl_percentage"))
        notl = _num(r.get("notional_value"))
        notl = notl if np.isfinite(notl) and notl > 0 else NOTIONAL_FALLBACK
        rows.append(dict(day=r["_k"][:10], opened_at=r["_k"], pair=r["pair"], booked=bk, close_reason=r.get("close_reason"),
                         live_stopped=live_stopped(r.get("close_reason"), bk), notional=notl, prov=bool(P["n_bar"] > 0),
                         usd=s["delta"] * notl / 100, sign_flag=bool(np.isfinite(bk) and np.sign(bk) != np.sign(s["base"])), **s))
    d = pd.DataFrame(rows)
    print(f"  scored {len(d)} · pending {len(pend)}{(' (' + bud.why + ')') if bud.why else ''} · unresolved {unres} · "
          f"fapi requests {bud.requests} (weight {bud.used}) · archives fetched {bud.archives}")
    for x in pend[:20]:
        print(f"    ⏳ {x}")
    if not len(d):
        print(f"  verdict: collecting (0/{GATE_STOPS} stops on 0/{GATE_DAYS} days)")
        return
    print(f"  parity walker(−3) vs booked: mean |Δ| {(d.base - d.booked).abs().mean():.2f} pp · sign disagreements "
          f"{int(d.sign_flag.sum())}/{len(d)} · 1m-provisional {int(d.prov.sum())}")
    for x in d.itertuples():
        mark = "■" if x.live_stopped else "·"
        print(f"    {mark} {x.opened_at} {x.pair:12s} booked {x.booked:+.2f} ({x.close_reason}) · −3 {x.base:+.2f} {x.base_why:4s} · "
              f"−4 {x.shadow:+.2f} {x.shadow_why:4s} · Δ {x.delta:+.2f} % / ${x.usd:+.2f}{'  ᵖ' if x.prov else ''}"
              f"{'  ⚠ sign' if x.sign_flag else ''}")
    st = d[d.live_stopped]
    sv = int(((st.base_why == "STOP") & (st.shadow_why == "TP")).sum())
    dp = int(((st.base_why == "STOP") & (st.shadow_why == "STOP")).sum())
    rp = st[st.base_why != "STOP"]
    print(f"  live-stopped {len(st)} on {st.day.nunique()} days · saved by −4 (−3 STOP → −4 TP) {sv} · deeper {dp} · other {len(st) - sv - dp}"
          f" · mean Δ all {d.delta.mean():+.3f} % · ΣΔ ${d.usd.sum():+.2f}")
    if len(rp):
        print(f"  ⚠ reverse parity: {len(rp)} live-stopped fill(s) the walker at −3 did NOT stop (replica artefact — outside the saved "
              f"share): " + ", ".join(f"{x.opened_at} {x.pair}" for x in rp.itertuples()))
    viol = d[~d.live_stopped & (d.delta.abs() > 1e-9)]
    print(f"  integrity (Δ = 0 on fills live did NOT stop): {'✓' if not len(viol) else f'✗ {len(viol)} violation(s)'}")
    for x in viol.itertuples():
        print(f"    ✗ {x.opened_at} {x.pair}: live {x.close_reason} but walker −3 {x.base_why} → Δ {x.delta:+.2f}")
    v, det = verdict(d)
    if "saved" in det:
        print(f"  gate: saved {det['saved']:.0%} (≥ 50 %; retire ≤ 25 %) · mean Δ {det['mean']:+.3f} [{det['lo']:+.2f}, {det['hi']:+.2f}] "
              f"day-boot P {det['p']:.3f} (≥ {P_MIN}) · top day {det['day']} {det['day_share']:.0%} of ΣΔ (≤ 50 %) · violations "
              f"{det['violations']} (= 0)")
    print(f"  verdict: {v}")


if __name__ == "__main__":
    main()
