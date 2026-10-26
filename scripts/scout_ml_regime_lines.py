#!/usr/bin/env python3
"""🌐 Scout — ML_REGIME_LINES: the frozen momentum-long REGIME observe lines OBS-R1 / R1b / R2 (DECISION_LOG 382, operator 2026-10-26;
OBSERVE-ONLY — never changes config, never calls the bot API, never writes the DB; writes only under reports/).

SOURCE  reports/ML_REGIME_WL_2026-10-26.md §8 (thresholds frozen 2026-10-26, never re-tuned). The regime study found the B12–B24 ML loss
  is ONE regime episode (from 2026-09-21 alts move as high-beta copies of BTC), uniform across 31 / 31 sub-cohorts, with no within-regime
  entry separator beyond chance; the operator decided NO switch / filter / sizing change and registered these watch lines.
  (The study's 13-fill "replay did not take it" lead was tested twice — closed-bar re-check and live-vs-replay tag — and RETIRED: replay
  per-trade selection is path-dependent; evidence reports/ML_CLOSEDBAR_REFUTED_PARITY_2026-10-26.json + ML_LIVE_VS_REPLAY_REFUTED_PARITY_2026-10-26.json.)
COHORT  MOMENTUM LONG, CLOSED, non-probe, non-MANUAL, operator-closed out (scout_gatekit), opened ≥ 2026-10-27 00:00 UTC; orders exports
  ~/Downloads/scalpars_orders_*.csv (paper + live via export_glob), dedup (opened_at[:19], pair, direction), CLOSED beats OPEN, newest export
  wins; master rows = today's-stack KEEP; % at 1×; DAY units (a day = one window). The 9 B25 fills (exports after the master's last fill,
  before the floor) are SEEN and shown apart, never counted. The closed-bar and live-vs-replay parity generators were local, unversioned
  and are removed; their results are the two REFUTED_PARITY files.
LINES (computed per fill from CLOSED 5m bars; within the backtest cache's list span (lists dated ≤ 2026-10-11) the formulas reproduce
  the study's per-fill table exactly — corr / beta / rv7 max |Δ| < 1e-13 on its 185 fills ≤ 2026-10-11 12:00)
  UNIVERSE  a fill on UTC day D reads the list DATED D − 1 = every USDT perpetual ranked by its day D − 2 quote volume (1d klines, full
          exchange universe; scripts/backtest_phase1_universe.py / backtest_extend_cache.py convention, reports/backtest_cache/
          universe_daily_top50.csv). Later dates are rebuilt here from the 1d volumes of EVERY current perpetual (cache daily/ ∪ this
          scout's own fetch) — only when the daily fetch is complete (an early stop leaves reports/backtest_cache/scout_ml_regime_lines/
          universe_incomplete.json and no list is built until a run resumes and finishes); no stale-list fallback (missing list →
          unscored). The rebuild equals the cached list on 2026-10-10 / 10-11 (test). When an order row carries the DL 383 stamps
          entry_alt_corr24 / entry_alt_beta24 (same D − 2 definition) the stamp is used and the stamp-vs-rebuild agreement printed.
  OBS-R1 ML_HICORR_REGIME  mean pairwise Pearson corr of 5m log returns over the trailing 288 closed bars among the top-20 of that list
          (ex BTC / stables) > 0.3164; alias R1b = median alt beta to BTC over the same bars > 1.372 (read with R1; its episodes are R1's)
  OBS-R2 ML_LOWRV7         BTC 7-day realised vol = std of 2016 closed 5m log returns × 100 ≤ 0.1122 (descriptive)
🔒 BAR (R1, R1b, R2 the same; read ONCE at zone N = 15 and ONCE at N = 30, each persisted decide-once in reports/.ml_regime_lines_gate.json,
  atomic, an unreadable file is never overwritten): SWITCH-CANDIDATE → OPERATOR CALL iff zone N ≥ 15 on ≥ 8 days ∧ WR < 61.5 % ∧ day-block
  bootstrap P(avg < 0) ≥ 0.95 ∧ no day / pair ≥ 50 % of the zone loss ∧ ≥ 3 separate regime episodes (episodes split by ≥ 5 consecutive
  outside-zone noon readings between two zone-fill days) · RETIRE if zone avg ≥ 0 · else KEEP OBSERVING. NEVER decided while a cohort fill
  up to the n-th zone fill is unscored (NaN) or an ML fill opened before it is still OPEN → HOLD ("n unscored / n open"). R1 is read
  jointly with the DL 289 ML kill gate (evidence not double-counted). OBS-C1 (R1 zone ∧ corr15 > 0.365) needs the 1m top-10 grid → NOT
  computed here: read with study_ml_impulse_fetch.py + study_ml_impulse_features.py (shares fills with PREP P6, each fill counts once).
Binance (only past the caches): /fapi/v1/exchangeInfo (1) + /fapi/v1/klines 1d per perpetual (limit 60 = weight 1) for the universe lists,
  /fapi/v1/klines 5m (limit 1000 = weight 5) for BTC + the top-20 members of every needed list; ≤ MAX_REQ requests per run (resumes next
  run; --max-req N for a catch-up) paced ≤ 250 weight / min; X-MBX-USED-WEIGHT-1M ≥ 350 → wait for the next minute, at most MAX_WAITS waits
  per RUN, then stop; 418 / 429 arm the shared cooldown (scripts/binance_cooldown.py) and stop; an active cooldown means no fetch.
Usage: venv/bin/python scripts/scout_ml_regime_lines.py [--no-fetch] [--max-req N]
"""
import json
import os
import sys
import time
from urllib.parse import quote

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import binance_cooldown as _CD       # noqa: E402
import export_glob as _EG            # noqa: E402
import scout_gatekit as _GK          # noqa: E402

DL = 382
OBS_FLOOR = "2026-10-27T00:00:00"
BE_WR = 61.5
P_MIN = 0.95
R1_THR, R1B_THR, R2_THR, C1_THR = 0.3164, 1.372, 0.1122, 0.365
OBS_N1, OBS_N2, OBS_DAYS, EPISODES_MIN, EPISODE_GAP_DAYS = 15, 30, 8, 3, 5
BOOT_N, BOOT_SEED = 4000, 382
BAR5, HOUR, DAY = 300_000, 3_600_000, 86_400_000
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
REPORTS = os.path.join(_ROOT, "reports")
MASTER = os.path.join(REPORTS, "MASTER_POOL_stacked.csv")
CACHE = os.path.join(REPORTS, "backtest_cache")
WORK = os.path.join(CACHE, "scout_ml_regime_lines")
FETCH5 = os.path.join(WORK, "k5m")                              # this scout's own budgeted 5m fetch
# read-only 5m caches other studies fetched past the backtest cache's alt end (2026-10-12 04:05) — caches first; FETCH5 last (wins)
EXTRA5 = [os.path.join(REPORTS, "PREP_B25_ML_WL_fetch", "k5m_full"), os.path.join(REPORTS, "ml_regime", "fetch", "k5m"),
          os.path.join(REPORTS, "ML_EXIT_DEEP_2026-10-26_fetch", "k5m"), FETCH5]
TOP50_CACHE = os.path.join(CACHE, "universe_daily_top50.csv")
TOP50_OWN = os.path.join(WORK, "universe_daily_top50.csv")     # this scout's lists for dates after the cache (same convention)
DAILY_CACHE = os.path.join(CACHE, "daily")                       # 1d quote volume per USDT perp (backtest_phase1_universe.py / extend)
DAILY_OWN = os.path.join(WORK, "daily")                          # this scout's 1d volumes after the cache's end
SYMBOLS = os.path.join(WORK, "exchange_symbols.json")            # the USDT perpetuals of the last complete exchangeInfo read
CHECKED = os.path.join(WORK, "daily_checked.json")               # {sym: last day asked} — no bar that day = not listed yet
INCOMPLETE = os.path.join(WORK, "universe_incomplete.json")      # set while a daily fetch stopped early → no own list, resume next run
CSV_OUT = os.path.join(REPORTS, "SCOUT_ML_REGIME_LINES.csv")
STATE = os.path.join(REPORTS, ".ml_regime_lines_gate.json")
MAX_REQ, PACE_S, WEIGHT_WAIT, KLINES_LIMIT, MAX_WAITS = 150, 1.25, 350, 1000, 2
STABLE = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "DAIUSDT", "USDEUSDT", "XUSDUSDT", "BTCDOMUSDT"}
COLS = ["opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "cell_multiplier_source", "pnl_percentage",
        "peak_pnl", "close_reason", "era", "stack_keep", "stack_pct", "is_probe", "entry_alt_corr24", "entry_alt_beta24"]


def _key(v):
    return str(v)[:19].replace(" ", "T")


def _ms(k):
    try:
        return int(pd.Timestamp(str(k)[:19]).value // 1_000_000)
    except Exception:
        return None


def _f(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def is_probe(src):
    return "PROBE" in str(src or "").upper()


def stat(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if not len(y):
        return "0"
    return f"{len(y)}·{(y > 0).mean() * 100:.0f}%·{y.mean():+.3f}"


def day_boot_neg(y, day, n=BOOT_N, seed=BOOT_SEED):
    """day-block bootstrap P(avg < 0)."""
    y, day = np.asarray(y, float), np.asarray(day)
    u = np.unique(day)
    if len(u) < 2:
        return float("nan")
    s = np.array([y[day == d].sum() for d in u]); c = np.array([(day == d).sum() for d in u])
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    return float((s[idx].sum(1) / c[idx].sum(1) < 0).mean())


def loss_share(y, key):
    """largest share of the cohort's gross LOSS carried by one key (day / pair); nan when no loss."""
    y = pd.Series(np.asarray(y, float)); k = pd.Series(np.asarray(key))
    neg = y.clip(upper=0)
    tot = -neg.sum()
    return float((-neg.groupby(k).sum()).max() / tot) if tot > 0 else float("nan")


def read_exports(export_glob=EXPORT_GLOB):
    fr = []
    for f in _EG.glob_exports(export_glob):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        fr.append(d.assign(_m=os.path.getmtime(f), _src=os.path.basename(f)))
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=COLS + ["_m", "_src"])


def read_master(path=MASTER):
    try:
        return pd.read_csv(path, low_memory=False, usecols=lambda c: c in COLS)
    except Exception:
        return pd.DataFrame(columns=COLS)


def _batch_label(src):
    """'scalpars_orders_paper_2026-10-26_15-11-28.csv' → 'export 2026-10-26 (paper)' — the earliest export holding a fill = its batch."""
    import re
    m = re.search(r"_(paper|live)_(\d{4}-\d{2}-\d{2})", str(src))
    return f"export {m.group(2)} ({m.group(1)})" if m else f"export {str(src)[:40]}"


def _bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def ml_rows(raw, master):
    """→ (fills, open_rows, operator_closed): ML LONG non-probe non-MANUAL rows. master rows = today's-stack KEEP (src 'master'),
    export rows not in the master = as traded (src 'export'). Columns: _k, pair, t_ms, day, y, ng, src, batch, status + stamps."""
    out = []
    mk, m_end = set(), ""
    if master is not None and len(master):
        m = master[(master.entry_strategy.astype(str) == "MOMENTUM") & (master.direction.astype(str) == "LONG")
                   & (master.status.astype(str) == "CLOSED")].copy()
        m = m[~_bool(m.is_probe) & ~m.cell_multiplier_source.map(is_probe)]
        mk = set(zip(master.opened_at.map(_key), master.pair))
        m_end = master.opened_at.map(_key).max()                 # export rows count only AFTER the master's last fill (B25 provisional
        m = m[_bool(m.stack_keep)].copy()                         # rule; older export rows not in the master were never archived)
        m["y"] = pd.to_numeric(m.stack_pct, errors="coerce")
        m["src"] = "master"
        m["batch"] = m.era.astype(str)
        out.append(m)
    op = ox = pd.DataFrame()
    if raw is not None and len(raw):
        e = raw[(raw.entry_strategy.astype(str) == "MOMENTUM") & (raw.direction.astype(str) == "LONG")].copy()
        e = e[~e.cell_multiplier_source.map(is_probe)]
        e["_k"] = e.opened_at.map(_key)
        first = e.groupby(["_k", "pair"])._src.agg(lambda s: sorted(s)[0])            # the earliest export holding the fill = its batch
        e["_c"] = e.status.astype(str).str.upper().eq("CLOSED")
        e = e.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last")
        e = e[[(k, p) not in mk and k > m_end for k, p in zip(e._k, e.pair)]]
        e["batch"] = [_batch_label(first.get((k, p), "")) for k, p in zip(e._k, e.pair)]
        op = e[~e._c & e.status.astype(str).str.upper().eq("OPEN")].copy()
        c = e[e._c].copy()
        c, ox = _GK.split_operator_closed(c) if len(c) else (c, c.iloc[0:0])
        c = c.copy()
        c["y"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
        c["src"] = "export"
        out.append(c)
    f = pd.concat(out, ignore_index=True, sort=False) if out else pd.DataFrame(columns=COLS)
    if len(f):
        f["_k"] = f.opened_at.map(_key)
        f["t_ms"] = f._k.map(_ms)
        f = f[f.t_ms.notna() & f.y.notna()].copy()
        f["t_ms"] = f.t_ms.astype("int64")
        f["day"] = f._k.str[:10]
        f["ng"] = (pd.to_numeric(f.peak_pnl, errors="coerce") < 0.10) & (f.y < 0)
        f = f.sort_values("t_ms").reset_index(drop=True)
    for x in (op,):
        if len(x):
            x["t_ms"] = x._k.map(_ms)
    return f, op, ox


def _tail_ms(fp):
    """open_time of the last row of a 5m csv (reads the file tail only); None when unreadable."""
    try:
        with open(fp, "rb") as fh:
            fh.seek(0, 2)
            n = fh.tell()
            fh.seek(max(0, n - 4096))
            last = fh.read().decode("utf-8", "ignore").strip().splitlines()[-1]
        return int(float(last.split(",")[0]))
    except Exception:
        return None


def cover_end(pair):
    ends = [_tail_ms(os.path.join(d, f"{pair}.csv")) for d in [os.path.join(CACHE, "k5m_full"), os.path.join(CACHE, "k5m")] + EXTRA5
            if os.path.exists(os.path.join(d, f"{pair}.csv"))]
    if pair == "BTCUSDT" and os.path.exists(os.path.join(CACHE, "btc_5m.csv")):
        ends.append(_tail_ms(os.path.join(CACHE, "btc_5m.csv")))
    ends = [e for e in ends if e is not None]
    return max(ends) if ends else None


class Budget:
    """one run's Binance budget: requests, stop reason, notes, the last X-MBX-USED-WEIGHT-1M and the TOTAL weight waits of the run
    (≤ MAX_WAITS — a wait never resets the counter, so a busy IP stops the run instead of blocking it)."""
    def __init__(self, max_req=MAX_REQ):
        self.n, self.max, self.stop, self.notes, self.weight, self.waits = 0, max_req, None, [], None, 0


def _arm_cooldown(sec):
    try:
        until = max(_CD.read_until_ms(), int(time.time() * 1000) + int(max(60, sec) * 1000))
        tmp = f"{_CD.SHARED}.{os.getpid()}.tmp"
        os.makedirs(os.path.dirname(_CD.SHARED), exist_ok=True)
        with open(tmp, "w") as fh:
            fh.write(str(until))
        os.replace(tmp, _CD.SHARED)
    except Exception:
        pass


def _get(url, bud, sleep=time.sleep, opener=None):
    """one public GET with the weight / cooldown discipline → parsed JSON or None (bud.stop set on a stop condition)."""
    import urllib.error
    import urllib.request
    if bud.stop:
        return None
    if _CD.active():
        bud.stop = "shared Binance cooldown active"
        return None
    if bud.n >= bud.max:
        bud.stop = f"request budget {bud.max} reached"
        return None
    if bud.weight is not None and bud.weight >= WEIGHT_WAIT:
        if bud.waits >= MAX_WAITS:
            bud.stop = f"X-MBX-USED-WEIGHT-1M {bud.weight} ≥ {WEIGHT_WAIT} after {bud.waits} waits this run"
            return None
        sleep(61 - (time.time() % 60))
        bud.waits += 1
    bud.n += 1
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scalpars-scout/382"})
        with (opener or urllib.request.urlopen)(req, timeout=30) as r:
            w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            bud.weight = int(w) if w and str(w).isdigit() else None
            data = json.loads(r.read())
        sleep(PACE_S)
        return data
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            ra = e.headers.get("Retry-After") if e.headers else None
            _arm_cooldown(float(ra) if ra and str(ra).replace(".", "").isdigit() else 300)
            bud.stop = f"HTTP {e.code} — shared cooldown armed"
        else:
            bud.notes.append(f"HTTP {e.code} on {url[:90]}")
        return None
    except Exception as e:
        bud.notes.append(f"fetch error {str(e)[:60]}")
        return None


# ─────────────────────────── universe = the study's daily top-50 lists (backtest_phase1_universe.py convention) ───────────────────────────
# A list DATED L ranks every USDT perpetual by its day L − 1 quote volume (1d klines, full exchange universe, survivorship-free). A fill
# on UTC day X reads the list dated X − 1, i.e. the D − 2 volume. The backtest cache holds the lists through its last date; later dates
# are rebuilt here from 1d klines of EVERY USDT perpetual (cache daily/ ∪ this scout's DAILY_OWN), only when the daily fetch is complete.
def _read_daily(dirs=None):
    """{day_ms: {sym: quote_vol}} from <dir>/<SYM>.csv rows 'SYM,open_ms,quote_vol' (later dirs win)."""
    by = {}
    for d in dirs or (DAILY_CACHE, DAILY_OWN):
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".csv"):
                continue
            try:
                for line in open(os.path.join(d, fn)):
                    p = line.strip().split(",")
                    if len(p) == 3:
                        by.setdefault(int(p[1]), {})[p[0]] = float(p[2])
            except Exception:
                continue
    return by


def rank_list(by_day, list_date, n=50):
    """the study's list dated list_date: top-n USDT perps by day list_date − 1 quote volume → [(rank, sym, qv)] or None."""
    src = by_day.get(int(list_date) - DAY)
    if not src:
        return None
    top = sorted(((qv, s) for s, qv in src.items()), reverse=True)[:n]
    return [(r, s, qv) for r, (qv, s) in enumerate(top, 1)]


def _lists():
    """{list date ms: [pairs by rank, ex BTC / stables]} — the cache's lists, then this scout's own for later dates."""
    fr = [pd.read_csv(f) for f in (TOP50_CACHE, TOP50_OWN) if os.path.exists(f)]
    if not fr:
        return {}
    u = pd.concat(fr).drop_duplicates(["date_ms", "pair"], keep="first")
    u = u[~u.pair.isin(STABLE) & (u.pair != "BTCUSDT") & u.pair.astype(str).str.match(r"^[A-Z0-9]+USDT$")]
    return {int(d): g.sort_values("rank").pair.tolist() for d, g in u.groupby("date_ms")}


def _symbols():
    try:
        return json.load(open(SYMBOLS))
    except Exception:
        return None


def _checked():
    """{sym: the last day the daily fetch asked Binance about} — a perpetual with no bar that day was not listed yet."""
    try:
        return {k: int(v) for k, v in json.load(open(CHECKED)).items()}
    except Exception:
        return {}


def universe_incomplete():
    return os.path.exists(INCOMPLETE)


def ensure_daily(through_list_date, bud, now_ms):
    """1d quote volumes of EVERY USDT perpetual through day through_list_date − 1 (what the list dated through_list_date ranks on).
    A run that stops early leaves the INCOMPLETE marker (no own list is built until a later run resumes and finishes). → True when
    complete."""
    need_day = int(through_list_date) - DAY
    info = _get("https://fapi.binance.com/fapi/v1/exchangeInfo", bud)
    if not info:
        _atomic_json({"through": need_day, "why": bud.stop or "exchangeInfo unavailable"}, INCOMPLETE)
        return False
    syms = sorted(s["symbol"] for s in info.get("symbols", []) if str(s.get("symbol", "")).endswith("USDT")
                  and s.get("contractType") == "PERPETUAL" and s.get("status") in ("TRADING", "SETTLING"))
    _atomic_json(syms, SYMBOLS)
    have = _read_daily()
    last = {}
    for d, m in have.items():
        for s in m:
            last[s] = max(last.get(s, 0), d)
    os.makedirs(DAILY_OWN, exist_ok=True)
    checked = _checked()
    pending = [s for s in syms if last.get(s, 0) < need_day and checked.get(s, 0) < need_day]
    for k, s in enumerate(pending):
        start = (last[s] + DAY) if s in last else need_day - 40 * DAY
        kl = _get(f"https://fapi.binance.com/fapi/v1/klines?symbol={quote(s)}&interval=1d&startTime={start}&limit=60", bud)
        if kl is None:
            if bud.stop:
                _atomic_json(checked, CHECKED)
                _atomic_json({"through": need_day, "pending": pending[k:], "why": bud.stop}, INCOMPLETE)
                return False
            bud.notes.append(f"daily {s}: no answer — NOT counted (its days block the lists until a later run gets it)")
            continue
        fp = os.path.join(DAILY_OWN, f"{s}.csv")
        rows = {}
        if os.path.exists(fp):
            for line in open(fp):
                p = line.strip().split(",")
                if len(p) == 3:
                    rows[int(p[1])] = p[2]
        for x in kl:
            if int(x[6]) < now_ms:                                         # closed days only
                rows[int(x[0])] = str(x[7])
        if rows:
            tmp = f"{fp}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                fh.writelines(f"{s},{d},{rows[d]}\n" for d in sorted(rows))
            os.replace(tmp, fp)
        checked[s] = need_day                                             # asked through need_day (no bar = not listed yet)
        if k % 50 == 49:
            _atomic_json(checked, CHECKED)
    _atomic_json(checked, CHECKED)
    if os.path.exists(INCOMPLETE):
        os.remove(INCOMPLETE)
    return True


def build_own_lists(list_dates):
    """the study's list for every needed date the cache lacks, from the FULL universe's 1d volumes → TOP50_OWN. Built only when the
    daily fetch is complete (no INCOMPLETE marker) and every current perpetual has day L − 1 or was asked through it. → list dates."""
    have = set(_lists())
    need = sorted(d for d in set(int(x) for x in list_dates) if d not in have)
    if not need or universe_incomplete():
        return have
    syms = _symbols()
    if not syms:
        return have
    by = _read_daily()
    chk = _checked()
    rows = []
    for d in need:
        day = by.get(d - DAY, {})
        if not day or any(s not in day and chk.get(s, 0) < d - DAY for s in syms):
            continue                                                       # day L − 1 not covered for the whole universe → no list
        rows += [(d, r, s, round(qv)) for r, s, qv in rank_list(by, d)]
    if rows:
        new = pd.DataFrame(rows, columns=["date_ms", "rank", "pair", "quote_vol"])
        if os.path.exists(TOP50_OWN):
            new = pd.concat([pd.read_csv(TOP50_OWN), new]).drop_duplicates(["date_ms", "pair"], keep="last")
        _atomic_csv(new, TOP50_OWN)
    return set(_lists())


def list_pairs(list_dates, top=20):
    """the top-`top` eligible pairs (ex BTC / stables) of each list date — the corr / beta members to keep in 5m data."""
    U = _lists()
    out = set()
    for d in list_dates:
        out |= set(U.get(int(d), [])[:top])
    return out


def ensure_data(pairs, t_lo, t_hi, bud, now_ms):
    """extend the 5m coverage of the given pairs over [t_lo, t_hi] (budgeted). → notes."""
    need_end = min(int(t_hi) // BAR5 * BAR5, int(now_ms) - BAR5)
    os.makedirs(FETCH5, exist_ok=True)
    done = 0
    for p in sorted(set(pairs)):
        if bud.stop:
            break
        have = cover_end(p)
        if have is not None and have >= need_end - BAR5:
            continue
        cur = (have + BAR5) if have is not None and have >= t_lo else int(t_lo)
        rows = []
        while cur <= need_end and not bud.stop:
            kl = _get(f"https://fapi.binance.com/fapi/v1/klines?symbol={quote(p)}&interval=5m&startTime={cur}&endTime={need_end + BAR5 - 1}"
                      f"&limit={KLINES_LIMIT}", bud)
            if not kl:
                break
            rows += [[int(k[0]), k[1], k[2], k[3], k[4], k[5], k[7]] for k in kl if int(k[6]) < now_ms]
            nxt = int(kl[-1][0]) + BAR5
            if nxt <= cur or len(kl) < KLINES_LIMIT:
                break
            cur = nxt
        if rows:
            fp = os.path.join(FETCH5, f"{p}.csv")
            new = pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
            if os.path.exists(fp):
                try:
                    new = pd.concat([pd.read_csv(fp), new])
                except Exception:
                    pass
            _atomic_csv(new.drop_duplicates("open_time", keep="last").sort_values("open_time"), fp)
            _K5.pop(p, None)
            done += 1
    return [f"5m fetch: {done} pairs extended"]


_K5 = {}


def load5(pair):
    if pair not in _K5:
        fr = []
        for d in [os.path.join(CACHE, "k5m_full"), os.path.join(CACHE, "k5m")] + EXTRA5:
            fp = os.path.join(d, f"{pair}.csv")
            if os.path.exists(fp):
                try:
                    x = pd.read_csv(fp, usecols=[0, 2, 3, 4, 6]); x.columns = ["t", "h", "l", "c", "qv"]; fr.append(x)
                except Exception:
                    pass
        if pair == "BTCUSDT" and os.path.exists(os.path.join(CACHE, "btc_5m.csv")):
            try:
                x = pd.read_csv(os.path.join(CACHE, "btc_5m.csv"), usecols=[0, 2, 3, 4, 6]); x.columns = ["t", "h", "l", "c", "qv"]
                fr.insert(0, x)
            except Exception:
                pass
        k = None
        if fr:
            k = pd.concat(fr).dropna(subset=["c"]).drop_duplicates("t", keep="last").sort_values("t").reset_index(drop=True)
            k["t"] = k.t.astype(np.int64)
        _K5[pair] = k
    return _K5[pair]


def regime_at(T):
    """per time T (ms): corr24 (top-20 of the list dated day(T) − 1, 288 closed bars), beta (median), BTC rv7 — the study's frozen
    formulas. The list must be the exact date (no stale fallback): a missing list leaves corr / beta unscored."""
    T = np.asarray(T, dtype=np.int64)
    out = pd.DataFrame({"t": T, "corr24": np.nan, "beta24": np.nan, "rv7": np.nan, "n_corr": 0})
    btc = load5("BTCUSDT")
    if btc is None or not len(T):
        return out
    bt = btc.t.values; bc = btc.c.values.astype(float)
    blr = np.r_[0.0, np.diff(np.log(bc))]
    rv = pd.Series(blr).rolling(2016).std().values * 100
    U = _lists()
    for i, t in enumerate(T):
        j = np.searchsorted(bt, t - BAR5, side="right") - 1
        if j >= 2016 and (t - (bt[j] + BAR5)) <= 30 * 60_000:
            out.loc[i, "rv7"] = rv[j]
        L = U.get(int(t) // DAY * DAY - DAY, [])
        X = []
        for p in L[:20]:
            k = load5(p)
            if k is None or len(k) < 300:
                continue
            tt = k.t.values
            jj = np.searchsorted(tt, t - BAR5, side="right") - 1
            if jj < 2016 or (t - (tt[jj] + BAR5)) > 30 * 60_000 or tt[jj - 287] != tt[jj] - 287 * BAR5:
                continue
            c = k.c.values.astype(float)
            X.append(np.diff(np.log(c[jj - 288:jj + 1])))
        if len(X) >= 8 and j >= 288:
            X = np.array(X)
            C = np.corrcoef(X); iu = np.triu_indices(len(X), 1)
            out.loc[i, "corr24"] = float(np.nanmean(C[iu])); out.loc[i, "n_corr"] = len(X)
            b = blr[j - 287:j + 1]
            if np.var(b) > 0:
                out.loc[i, "beta24"] = float(np.median([np.cov(x, b)[0, 1] / np.var(b) for x in X]))
    return out


def _atomic_csv(df, path):
    _guard(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_json(o, path):
    _guard(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(o, fh)
    os.replace(tmp, path)


def episodes(zone_days, daily, thr, gt=True, gap=EPISODE_GAP_DAYS):
    """number of separate regime episodes among the zone-fill days: a new episode starts when ≥ gap consecutive UTC days with a noon
    reading OUTSIDE the zone (corr ≤ thr for R1, rv7 > thr for R2) lie between two zone-fill days. daily = {day 'YYYY-MM-DD': value}."""
    zd = sorted(set(zone_days))
    if not zd:
        return 0
    ep = 1
    for a, b in zip(zd[:-1], zd[1:]):
        run = best = 0
        for d in pd.date_range(a, b, inclusive="neither").strftime("%Y-%m-%d"):
            v = daily.get(d, np.nan)
            out_ = np.isfinite(v) and ((v <= thr) if gt else (v > thr))
            run = run + 1 if out_ else 0
            best = max(best, run)
        if best >= gap:
            ep += 1
    return ep


def frozen(state_path, name, v, txt, decided):
    """🔒 decided ONCE (atomic temp + os.replace); an unreadable state file is never overwritten. → (verdict, text)."""
    st = {}
    if state_path and os.path.exists(state_path):
        try:
            with open(state_path) as fh:
                st = json.load(fh)
            if not isinstance(st, dict):
                raise ValueError("not a JSON object")
        except Exception as e:
            print(f"⚠ ML_REGIME_LINES: state file {state_path} unreadable ({str(e)[:60]}) — left untouched", file=sys.stderr)
            return v, txt + " (⚠ state file unreadable — not frozen)"
    if name in st:
        return st[name]["verdict"], st[name]["text"] + f" (frozen {str(st[name].get('at', ''))[:16]} UTC)"
    if state_path and decided:
        if not os.path.abspath(state_path).startswith(REPORTS + os.sep):
            raise ValueError("ML_REGIME_LINES writes only under reports/")
        try:
            st[name] = {"verdict": v, "text": txt, "at": pd.Timestamp.now("UTC").isoformat()}
            tmp = f"{state_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(st, fh, indent=1)
            os.replace(tmp, state_path)
        except Exception as e:
            print(f"⚠ ML_REGIME_LINES: could not persist {name} ({str(e)[:60]})", file=sys.stderr)
    return v, txt

def obs_read(c, open_t, zone_col, var_col, thr_txt, daily, thr, gt, n_read):
    """one OBS line at its frozen read size n_read → (verdict, text, decided). c = EVERY closed cohort fill (scored or not), open_t = the
    open times of OPEN cohort ML fills. Never decided while a fill up to the n-th zone fill is unscored (NaN var_col) or still OPEN."""
    z = c[c[zone_col]].sort_values("t_ms")
    unsc_all, open_all = int(c[var_col].isna().sum()), int(len(open_t))
    if len(z) < n_read:
        return "WAIT", f"zone {stat(z.y)} on {z.day.nunique()} days ({len(z)}/{n_read}) · {unsc_all} unscored / {open_all} open", False
    tn = int(z.t_ms.iloc[n_read - 1])
    unsc = int(((c.t_ms <= tn) & c[var_col].isna()).sum())
    opn = int((np.asarray(open_t, dtype=np.int64) < tn).sum())
    if unsc or opn:
        return "HOLD", (f"{n_read} zone fills reached {pd.Timestamp(tn, unit='ms'):%m-%d %H:%M} but {unsc} unscored / {opn} open before it — "
                        f"read when complete"), False
    z = z.iloc[:n_read]
    wr = (z.y > 0).mean() * 100
    pneg = day_boot_neg(z.y, z.day)
    dshare, pshare = loss_share(z.y, z.day), loss_share(z.y, z.pair)
    ep = episodes(z.day.tolist(), daily, thr, gt)
    txt = (f"first {n_read} zone fills {stat(z.y)} on {z.day.nunique()} days · WR {wr:.0f} % · P(avg<0) {pneg:.2f} · top day / pair "
           f"{dshare * 100:.0f} / {pshare * 100:.0f} % of the loss · {ep} episode(s) [{thr_txt}]")
    if z.y.mean() >= 0:
        return "RETIRE", "zone avg ≥ 0 · " + txt, True
    if (z.day.nunique() >= OBS_DAYS and wr < BE_WR and pneg >= P_MIN and not (dshare >= 0.5) and not (pshare >= 0.5)
            and ep >= EPISODES_MIN):
        return "SWITCH-CANDIDATE", "OPERATOR CALL (read with the DL 289 kill gate) · " + txt, True
    return "KEEP OBSERVING", txt, True


def obs_lines(f, state_path, regime_fn=None, open_t=()):
    L = ["### OBS-R1 / R1b / R2 (DECISION_LOG 382 · frozen in ML_REGIME_WL_2026-10-26.md §8 · cohort ML KEEP ≥ 2026-10-27 00:00 UTC · DAY units)"]
    floor = _ms(OBS_FLOOR)
    c = f[(f.t_ms >= floor)].copy()
    seen = f[(f.src == "export") & (f.t_ms < floor)].copy()
    open_t = [int(t) for t in open_t if int(t) >= floor]
    for nm, x in (("cohort", c), ("B25 SEEN (not counted)", seen)):
        if not len(x):
            L.append(f"  {nm}: 0 fills")
            continue
        L.append(f"  {nm}: {stat(x.y)} · R1 zone {stat(x.y[x.r1])} / out {stat(x.y[~x.r1 & x.corr24.notna()])} · R1b zone {stat(x.y[x.r1b])} · "
                 f"R2 zone {stat(x.y[x.r2])} / out {stat(x.y[~x.r2 & x.rv7.notna()])} · unscored corr {int(x.corr24.isna().sum())} / rv7 {int(x.rv7.isna().sum())}")
        if nm.startswith("B25") and x.corr24.notna().any():
            L.append("    B25 per fill corr24 / beta24 / rv7: " + " · ".join(
                f"{p[:-4]} {k[5:16]} {a:.3f}/{b:.2f}/{r:.4f}" for p, k, a, b, r in zip(x.pair, x._k, x.corr24, x.beta24, x.rv7)))
    if "corr24_stamp" in f and f.corr24_stamp.notna().any():
        b = f[f.corr24_stamp.notna() & f.corr24_rebuild.notna()]
        L.append(f"  DL 383 stamp vs rebuild: {len(b)} fills with both · max |Δ corr| "
                 f"{(b.corr24_stamp - b.corr24_rebuild).abs().max() if len(b) else float('nan'):.4f} · max |Δ beta| "
                 f"{(b.beta24_stamp - b.beta24_rebuild).abs().max() if len(b) else float('nan'):.4f} (the stamp is used when present)")
    if len(c):
        lo, hi = c.day.min(), c.day.max()
        days = pd.date_range(lo, hi).strftime("%Y-%m-%d").tolist()
        noon = [_ms(d + "T12:00:00") for d in days]
        rg = (regime_fn or regime_at)(noon)
        dc = dict(zip(days, rg.corr24.values)); dr = dict(zip(days, rg.rv7.values))
    else:
        dc = dr = {}
    for name, col, var, thr, gt, txt in (("OBS-R1", "r1", "corr24", R1_THR, True, f"corr24 > {R1_THR}"),
                                         ("OBS-R1b", "r1b", "beta24", R1_THR, True, f"beta > {R1B_THR} (alias, read with R1; R1's episodes)"),
                                         ("OBS-R2", "r2", "rv7", R2_THR, False, f"rv7 ≤ {R2_THR}")):
        daily = dr if name == "OBS-R2" else dc
        for n in (OBS_N1, OBS_N2):
            v, t, dec = (obs_read(c, open_t, col, var, txt, daily, thr, gt, n) if len(c)
                         else ("WAIT", f"0 fills · {len(open_t)} open", False))
            fv, ft = frozen(state_path, f"{name}@{n}", v, t, dec)
            L.append(f"  {name} @ N={n}: {fv} — {ft}")
    L.append(f"  OBS-C1 (R1 zone ∧ corr15 > {C1_THR}): not computed here — needs the 1m top-10 grid (scripts/study_ml_impulse_fetch.py + "
             f"study_ml_impulse_features.py); shares fills with PREP P6 (each fill counts once). R1-zone cohort fills to score: "
             f"{int(c.r1.sum()) if len(c) else 0}.")
    return L


def add_regime(f, regime_fn=None):
    """corr24 / beta24 / rv7 per fill: the DL 383 stamps entry_alt_corr24 / entry_alt_beta24 when the row carries them (same D − 2 list
    definition), else the rebuild; rv7 always rebuilt. Both kept for the agreement line."""
    f = f.copy()
    if not len(f):
        for c in ("corr24", "beta24", "rv7"):
            f[c] = np.nan
        f["r1"] = f["r1b"] = f["r2"] = False
        return f
    rg = (regime_fn or regime_at)(f.t_ms.values)
    f["corr24_rebuild"], f["beta24_rebuild"], f["rv7"] = rg.corr24.values, rg.beta24.values, rg.rv7.values
    f["corr24_stamp"] = pd.to_numeric(f["entry_alt_corr24"], errors="coerce") if "entry_alt_corr24" in f else np.nan
    f["beta24_stamp"] = pd.to_numeric(f["entry_alt_beta24"], errors="coerce") if "entry_alt_beta24" in f else np.nan
    f["corr24"] = f.corr24_stamp.where(f.corr24_stamp.notna(), f.corr24_rebuild)
    f["beta24"] = f.beta24_stamp.where(f.beta24_stamp.notna(), f.beta24_rebuild)
    f["r1"], f["r1b"], f["r2"] = f.corr24 > R1_THR, f.beta24 > R1B_THR, f.rv7 <= R2_THR
    return f


def _guard(path):
    if not os.path.abspath(path).startswith(os.path.abspath(REPORTS) + os.sep):
        raise ValueError("ML_REGIME_LINES writes only under reports/")


def write_csv(f, path=CSV_OUT):
    if not path or not len(f):
        return
    keep = ["_k", "pair", "src", "batch", "day", "y", "ng", "close_reason", "corr24", "beta24", "rv7", "corr24_rebuild", "beta24_rebuild",
            "corr24_stamp", "beta24_stamp", "r1", "r1b", "r2"]
    _atomic_csv(f.reindex(columns=keep), path)


def run(now_ms=None, raw=None, master=None, fetch=True, state_path=STATE, csv_path=CSV_OUT, regime_fn=None, max_req=MAX_REQ):
    """the Scout section (list of lines). Tests inject raw / master / regime_fn and temp paths."""
    now_ms = int(now_ms or pd.Timestamp.now("UTC").value // 1_000_000)
    L = [f"## 🌐 ML_REGIME_LINES — momentum-long regime watch OBS-R1 / R1b / R2 (DECISION_LOG {DL}, observe-only; frozen reads at N "
         f"{OBS_N1} / {OBS_N2} → OPERATOR CALL)", ""]
    raw = read_exports() if raw is None else raw
    master = read_master() if master is None else master
    f, op, ox = ml_rows(raw, master)
    floor = _ms(OBS_FLOOR)
    f = f[(f.t_ms >= floor) | (f.src == "export")].copy() if len(f) else f   # the cohort (master + exports) + the SEEN B25 fills
    open_t = [int(t) for t in (op.t_ms if len(op) else []) if t == t]
    notes = []
    if len(f) and regime_fn is None:
        try:
            T = list(f.t_ms.astype("int64")) + [t for t in open_t if t >= floor]
            c = f[f.t_ms >= floor]
            if len(c):
                T += [_ms(d + "T12:00:00") for d in pd.date_range(c.day.min(), c.day.max()).strftime("%Y-%m-%d")]
            list_dates = sorted({int(t) // DAY * DAY - DAY for t in T})
            bud = Budget(max_req)
            if fetch and set(list_dates) - set(_lists()):
                ok = ensure_daily(max(list_dates), bud, now_ms)
                notes.append(f"universe 1d volumes: {'complete' if ok else 'INCOMPLETE — resumes next run, no own list built'}")
            if universe_incomplete():
                notes.append("universe incomplete marker set (a daily fetch stopped early) — fills on uncovered days stay unscored")
            have = build_own_lists(list_dates)
            miss = [d for d in list_dates if d not in have]
            if miss:
                notes.append(f"no universe list for {len(miss)} date(s) ({pd.Timestamp(miss[0], unit='ms'):%m-%d}…) — those fills unscored")
            if fetch:
                pairs = {"BTCUSDT"} | list_pairs(list_dates)
                notes += ensure_data(pairs, min(T) - 9 * DAY, max(T), bud, now_ms)
            if bud.n or bud.stop:
                notes.append(f"Binance: {bud.n} requests · {bud.waits} weight waits · last weight {bud.weight}"
                             + (f" · stop: {bud.stop}" if bud.stop else ""))
            notes += bud.notes[:5]
        except Exception as e:
            notes.append(f"universe / data step failed: {str(e)[:120]}")
    if len(f):
        try:
            f = add_regime(f, regime_fn)
        except Exception as e:
            notes.append(f"regime features failed: {str(e)[:120]}")
            for col in ("corr24", "beta24", "rv7"):
                f[col] = np.nan
            f["r1"] = f["r1b"] = f["r2"] = False
    for n in notes:
        L.append(f"  (data) {n}")
    L.append(f"Operator-closed (excluded): {len(ox)} · OPEN ML fills: {len(op)}")
    L += obs_lines(f, state_path, regime_fn, open_t) if len(f) else ["### OBS-R1 / R1b / R2: no fills"]
    try:
        write_csv(f, csv_path)
    except Exception as e:
        L.append(f"  (csv not written: {str(e)[:80]})")
    L.append("")
    return L


if __name__ == "__main__":
    _mr = int(sys.argv[sys.argv.index("--max-req") + 1]) if "--max-req" in sys.argv else MAX_REQ
    print("\n".join(run(fetch="--no-fetch" not in sys.argv, max_req=_mr)))
