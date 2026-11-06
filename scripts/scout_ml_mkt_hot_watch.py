#!/usr/bin/env python3
"""👁 Scout — ML_MKT_HOT watch line (pre-registered 2026-10-30, DECISION_LOG 416; OBSERVE ONLY — never changes config, no bot API; public 1h klines only).

QUESTION   Are full-size momentum longs bought when most of the market has already risen over 4 hours flat or worse, while those bought in a soft market earn? Theory (post-hoc:
           the frozen pre-study hypothesis H5 said breadth 50–80 % would be BETTER): with most coins already up we buy late, with few buyers left.
EVIDENCE   (in-sample, NEVER counted) 180 full-size fills, master on today's stack + B25: zone 83 on 41 days · 64 % · −0.03 %/fill vs 97 · 78 % · +0.30, day-bootstrap P(zone < rest) 0.99. Master 157 · 71 % ·
           +0.13 → 84 · 76 % · +0.24, daily compound +0.155 → +0.156 (each trade = account); B25 23 · 74 % · +0.30 → 13 · 92 % · +0.71, +0.95 → +1.29. Backtest yr6 2026 at the 0.48 cut: seeds 1 / 2 / 3 zone −0.11 / −0.10 / −0.10 vs
           −0.03 / −0.01 / −0.04, daily compound improves in all 3. AGAINST: the 322-split family test does not beat shuffled data (p ≈ 0.65); clearer in the later half of the data; only about half the batches lean this way.
OTHER      (operator 10-30) the same split is also read on FRENZY_WILLY and FRENZY_WILLY_HT, each decided SEPARATELY on the same bar, and shown for FRENZY_LITE (DISPLAY ONLY, never decided: its master, B25 and backtest disagree). In-sample, hot vs not: WILLY 26 · 65 % · −0.30 vs 27 · 78 % · +0.54
           (backtest REVERSED: +0.28 vs −0.09) · WILLY_HT 6 · 50 % · −0.49 vs 6 · 100 % · +1.92 (B25 only) · LITE 24 · 54 % · +0.02 vs 17 · 76 % · +1.02 (backtest reversed: +0.28 vs −0.03). NOT read: FRENZY_LONG (master re-priced at today's −4 stop, hot 17 · 82 % · +0.53 vs 16 · 88 % · +0.81, still earning, blocking lowers compound; at the old −3 it was −0.32 vs +0.60) and LITE_RETRY (hot is the better side).
COHORT     CLOSED LONG fills of the read sleeves, full size (*_PROBE out), from the orders exports (export_glob: Downloads + exports_archive/), dedup (opened_at[:19], pair, entry_strategy) — CLOSED beats OPEN, newest export wins; operator-closed out
           (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 416 commit + 10 min (forward only). Market-hot = % of the rolling top-100 up over 4 h — 🌡 V2 (DECISION_LOG 489), replaced the frozen 108 on
           2026-11-06: the share of the day's top-100 USDT perpetuals by the PREVIOUS UTC day's quote volume (BTC / ETH / stables out — services/mkt_up4h.py,
           the engine's own rule) whose last closed 1h close at the entry is above the close 4 h earlier, ≥ 30 members with data, bar ≤ 2 h old. A fill reads the engine's
           stamp entry_mkt_up4h_pct when it carries one; fills without it (all fills before the 489 deploy) are rebuilt with the same functions from the 11-06 study
           cache (reports/backtest_cache/mkt_hot_rolling/) + budgeted public klines for later days. zone = share > 0.50 · rest = ≤ 0.50 · no reading → UNSCORED (never counted). P&L = the booked pnl_percentage.
FROZEN BAR (🔒 OPERATOR 10-30, DECLARED: "we block negative or breakeven … we should target much higher than breakeven" — deliberately NOT the CLAUDE.md expectancy bar, whose WR-below-breakeven and P(mean < 0) legs this line does not use): at the first prefix (by open
           time) where the ZONE reaches N ≥ 15 on ≥ 8 distinct UTC days, decided ONCE — ① zone avg ≥ rest avg → RETIRED; ② FILTER CANDIDATE (operator decides; never automatic) iff zone avg ≤ +0.05 %/fill ∧ day-clustered bootstrap P(zone avg < rest avg) ≥ 0.90 (4,000 resamples,
           seed 7) ∧ no single day and no single pair carries ≥ 50 % of the zone's gross loss; ③ else KEEP OBSERVING. Deferred until the newest export was downloaded ≥ the prefix end + 1,200 min, no unscored fill sits inside the prefix and the breadth fetch completed. The read shows before → after
           with the daily compound return. A live rule also needs the 30–50 % in-sample haircut and a pre-committed revert (the engine stamps Market-hot on every fill since DL 489).
MEASURE    the gate (zone N ≥ 15 on ≥ 8 days, the bars above) and the freeze time (DL 416 commit + 10 min) are unchanged by DL 489 — only the measure moved
           from the frozen 108 to V2 (forward zone count on V2 at the switch: 5 of 15).
STATE      reports/SCOUT_ML_MKT_HOT.json (gitignored).
Usage: venv/bin/python -I scripts/scout_ml_mkt_hot_watch.py
"""
import glob
import json
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
import export_glob as _EG          # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK        # noqa: E402  (operator-closed fills out · the DL commit time)
import scout_b1h_negflank as NF    # noqa: E402  (gross-loss share, state I/O)
import scout_long_stop4 as LS      # noqa: E402  (budgeted, cached public fapi klines)
os.path.dirname(_D) in sys.path or sys.path.insert(0, os.path.dirname(_D))
from services import mkt_up4h as MB   # noqa: E402  (🌡 DL 489: THE V2 rule — the engine stamp and every rebuild call the same functions)

DL = 416
HOT_CUT = 0.50
DECIDED = (("ML", "MOMENTUM"), ("FRENZY_WILLY", "FRENZY_WILLY"), ("FRENZY_WILLY_HT", "FRENZY_WILLY_HT"))   # (state key, entry_strategy), each decided on its own
DISPLAY = (("FRENZY_LITE", "FRENZY_LITE"),)                                                                # shown, never frozen
MIN_COINS = 30
N_MIN, DAYS_MIN = 15, 8
AVG_MAX, P_MIN, SHARE_MAX = 0.05, 0.90, 0.50
BOOT_N, BOOT_SEED = 4000, 7
WAIT_MIN = 1200
STAMP = "entry_mkt_up4h_pct"   # 🌡 (DL 489) the engine's own V2 stamp — preferred over any rebuild when present
HOUR = 3_600_000
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_ML_MKT_HOT.json")
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_ml_mkt_hot")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason", "cell_multiplier_source",
        "entry_mkt_up4h_pct")
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
REGISTRATION = ("in-sample at registration (NEVER counted): momentum longs zone 83 on 41 days · 64 % · −0.03 vs 97 · 78 % · +0.30, P(zone < rest) 0.99; master compound +0.155 → +0.156, B25 +0.95 → +1.29; backtest 3 seeds zone worse; "
                "family test not above luck; WILLY −0.30 vs +0.54 and LITE +0.02 vs +1.02 with the backtest REVERSED; WILLY_HT −0.49 vs +1.92 on 12 fills")
UNIVERSE_F108_RETIRED = tuple((   # ⛔ RETIRED 2026-11-06 (DECISION_LOG 489) — kept ONLY so scripts/study_ml_mkt_hot_rolling.py can re-run its F108 comparison; no read uses it

    "1000BONKUSDT 1000PEPEUSDT 1000SHIBUSDT AAVEUSDT ACEUSDT ADAUSDT AKEUSDT ALLOUSDT APTUSDT ARBUSDT ARIAUSDT ASTERUSDT AVAXUSDT AXSUSDT BANKUSDT BCHUSDT "
    "BEATUSDT BERAUSDT BILLUSDT BNBUSDT BRUSDT BSBUSDT BTRUSDT BTWUSDT BULLAUSDT BUSDT CHIPUSDT CLOUSDT CRVUSDT CYSUSDT DASHUSDT DEXEUSDT DOGEUSDT DOTUSDT "
    "EDGEUSDT ENAUSDT ENSOUSDT ESPORTSUSDT FARTCOINUSDT FETUSDT FILUSDT GIGGLEUSDT HBARUSDT HEIUSDT HOMEUSDT HUSDT HYPEUSDT ICPUSDT INJUSDT IPUSDT JTOUSDT "
    "KAITOUSDT KITEUSDT LABUSDT LINKUSDT LITUSDT LTCUSDT LYNUSDT MAGMAUSDT NEARUSDT NOMUSDT ONDOUSDT ONUSDT OPNUSDT OPUSDT PAXGUSDT PENGUUSDT PIPPINUSDT "
    "PLAYUSDT PONSUSDT POWERUSDT PUMPUSDT RAVEUSDT RENDERUSDT REUSDT RIVERUSDT SAHARAUSDT SIRENUSDT SKYAIUSDT SOLUSDT SUIUSDT SYNUSDT TAOUSDT TONUSDT "
    "TRUMPUSDT TRXUSDT TUTUSDT UAIUSDT UBUSDT UNIUSDT USELESSUSDT USUSDT VELVETUSDT VIRTUALUSDT VVVUSDT WIFUSDT WLDUSDT WLFIUSDT XAUTUSDT XLMUSDT XMRUSDT "
    "XPLUSDT XRPUSDT ZAMAUSDT ZECUSDT ZROUSDT 币安人生USDT 龙虾USDT "
).split())


def load(paths=None):
    """→ (closed full-size fills of the read sleeves — dedup, CLOSED beats OPEN, newest export wins —, the newest read export's download time or None)."""
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
        return pd.DataFrame(columns=list(COLS) + ["o", "day", "pct"]), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[d.entry_strategy.astype(str).isin([e for _, e in DECIDED + DISPLAY]) & (d.direction.astype(str) == "LONG")
          & ~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


# ───────────────────── 🌡 V2 rolling top-100 (DECISION_LOG 489) — the engine's own rule (services/mkt_up4h.py), rebuilt for unstamped fills ─────────────────────
STUDY_CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "mkt_hot_rolling")   # the 11-06 study's 1d / 1h cache (history to 2026-11-06 15:00 UTC)
OLD_DAILY = os.path.join(_ROOT, "reports", "backtest_cache", "daily")                # delisted coins' daily quote volume (the study's fallback)
OLD_EI = os.path.join(_ROOT, "reports", "backtest_cache", "exchange_info.json")
DAY = 86_400_000
DAY_PAGE = 29                     # 1h bars per (member, UTC day) read beyond the study cache: opened D − 5 h … D + 23 h
_MEMO = {}


def _jload(p):
    with open(p) as f:
        return json.load(f)


def _raw_klines(sym, interval, start, limit, bud, now_ms, fetch=True):
    """public fapi klines → RAW rows (quote volume kept) of the bars CLOSED by now · cached under CACHE/v2/ only when all `limit` bars are
    closed · None = budget / cooldown / 403 stop · [] = no data (400) or a transient error (bud.errors). Same budget rules as LS.fapi_klines."""
    step = {"1h": HOUR, "1d": DAY}[interval]
    p = os.path.join(CACHE, "v2", f"{sym}_{interval}_{int(start)}_{int(limit)}.json")
    if os.path.exists(p):
        try:
            return _jload(p)
        except Exception:
            pass
    if bud.stop or not fetch:
        return None
    if LS._cooling(now_ms):
        bud.halt("shared Binance cooldown active")
        return None
    if bud.used >= LS.WEIGHT_STOP:
        bud.halt(f"weight {bud.used} ≥ {LS.WEIGHT_STOP}")
        return None
    time.sleep(LS.PACE_S)
    bud.requests += 1
    try:
        with urllib.request.urlopen(LS.FAPI_URL.format(s=urllib.parse.quote(sym), i=interval, a=int(start), n=int(limit)), timeout=15) as r:
            rows = json.loads(r.read().decode())
            bud.used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or bud.used)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now_ms)
            bud.halt(f"HTTP {e.code} — shared cooldown armed")
            return None
        if e.code == 403:
            bud.halt("HTTP 403 (WAF / geo) — run stopped")
            return None
        bud.errors += e.code != 400
        return []
    except Exception:
        bud.errors += 1
        return []
    out = [list(x) for x in rows if int(x[0]) + step <= now_ms]
    if len(out) == int(limit):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f)
    return out


def _study_daily():
    """the study's daily inputs → ({sym: {day ms: quote vol}}, {sym: {day ms: close}}, {sym: baseAsset}, last covered (closed) day ms). Memoized."""
    if "daily" in _MEMO:
        return _MEMO["daily"]
    newest = {}
    for f in glob.glob(os.path.join(STUDY_CACHE, "1d", "*.json")):
        sym, stamp = os.path.basename(f)[:-5].rsplit("_", 1)
        if stamp > newest.get(sym, ("",))[0]:
            newest[sym] = (stamp, f)
    vol, cls, cover = {}, {}, 0
    for sym, (stamp, f) in newest.items():
        rows = _jload(f)
        if not rows and os.path.exists(os.path.join(OLD_DAILY, f"{sym}.csv")):   # fapi no longer serves it: the repo cache (as the study)
            d = pd.read_csv(os.path.join(OLD_DAILY, f"{sym}.csv"), header=None, names=["s", "o", "qv"])
            rows = [[int(o), None, float(q)] for o, q in zip(d.o, d.qv)]
        if rows:
            vol[sym] = {int(r[0]): float(r[2]) for r in rows}
            cls[sym] = {int(r[0]): float(r[1]) for r in rows if r[1] is not None}
        cover = max(cover, int(pd.Timestamp(stamp).value // 10**6) - DAY)      # the fetch day's previous day = its last CLOSED 1d bar
    bases = {}
    if os.path.exists(OLD_EI):
        bases.update({s: m.get("baseAsset") for s, m in _jload(OLD_EI).items() if isinstance(m, dict) and m.get("baseAsset")})
    ei = os.path.join(STUDY_CACHE, "exchangeInfo_live.json")
    if os.path.exists(ei):
        bases.update({m["symbol"]: m.get("baseAsset") for m in _jload(ei).get("symbols", []) if m.get("baseAsset")})
    _MEMO["daily"] = (vol, cls, bases, cover)
    return _MEMO["daily"]


def _live_perps():
    """{sym: status} of every USDT PERPETUAL in the study's cached exchangeInfo (the candidate list of days past the study cache)."""
    ei = os.path.join(STUDY_CACHE, "exchangeInfo_live.json")
    if not os.path.exists(ei):
        return {}
    return {m["symbol"]: m.get("status") for m in _jload(ei).get("symbols", [])
            if m.get("quoteAsset") == "USDT" and m.get("contractType") == "PERPETUAL"}


def v2_universes(days, bud=None, now_ms=None, fetch=True):
    """UTC days → ({day: [members]} by services.mkt_up4h.select_universe on the PREVIOUS day's quote volume, complete?).
    History (≤ the study cache) = the study's daily inputs, no status filter (as-of status unknown — the study's rule); later days = 1d klines
    of every USDT perpetual in the cached exchangeInfo, status TRADING (the engine's rule; cached per symbol-day, ≈ 550 reads of weight 1
    per uncached day — only for days whose fills carry no stamp)."""
    bud = bud or LS.Budget()
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    vol, cls, bases, cover = _study_daily()
    out, complete = {}, True
    for d in sorted({pd.Timestamp(x).normalize() for x in days}):
        dm = int(d.value // 10**6)
        prev = dm - DAY
        if ("uni", dm) in _MEMO:
            out[d] = _MEMO[("uni", dm)]
            continue
        if dm > now_ms:
            continue
        if prev <= cover:
            dv = {s: v[prev] for s, v in vol.items() if prev in v}
            closes = {s: [c[t] for t in range(prev - (MB.PEG_WINDOW - 1) * DAY, prev + 1, DAY) if t in c] for s, c in cls.items()}
            mem = MB.select_universe(dv, None, bases, closes)
        else:
            st = _live_perps()
            dv, closes = {}, {}
            for s in sorted(x for x, v in st.items() if v == "TRADING"):
                rows = _raw_klines(s, "1d", prev, 1, bud, now_ms, fetch)
                if rows is None or getattr(bud, "errors", 0):
                    complete = False
                    break
                r = [x for x in rows if int(x[0]) == prev]
                if r:
                    dv[s] = float(r[0][7])
                hist = [cls.get(s, {}).get(t) for t in range(prev - (MB.PEG_WINDOW - 1) * DAY, prev, DAY)]
                closes[s] = [x for x in hist if x is not None] + [float(x[4]) for x in r]
            if not complete:
                break
            mem = MB.select_universe(dv, st, bases, closes)
        if len(mem) < MB.TOP_N:
            complete = False
            continue
        _MEMO[("uni", dm)] = mem
        out[d] = mem
    return out, complete


def _study_1h(sym):
    """the study's 1h closes of `sym` → {bar close ms: close} (full pages + the newest tail of each page)."""
    pages, tails = {}, {}
    for f in glob.glob(os.path.join(STUDY_CACHE, "1h", glob.escape(sym) + "_*.json")):
        parts = os.path.basename(f)[len(sym) + 1:-5].split("_")
        if not parts[0].isdigit():
            continue
        if len(parts) == 1:
            pages[int(parts[0])] = f
        elif len(parts) == 3 and parts[1] == "tail" and int(parts[2]) > tails.get(int(parts[0]), (0, ""))[0]:
            tails[int(parts[0])] = (int(parts[2]), f)
    out = {}
    for t in sorted(set(pages) | set(tails)):
        for r in _jload(pages.get(t) or tails[t][1]):
            out[int(r[0]) + HOUR] = float(r[1])
    return out


def _study_1h_cover():
    """the bar close (ms) up to which the study's 1h cache is complete (its newest tail's run hour)."""
    if "cover1h" not in _MEMO:
        hs = [int(os.path.basename(f)[:-5].rsplit("_", 1)[1]) for f in glob.glob(os.path.join(STUDY_CACHE, "1h", "*_tail_*.json"))]
        _MEMO["cover1h"] = (max(hs) * HOUR) if hs else 0
    return _MEMO["cover1h"]


def fetch_closes(t_min, now_ms, bud=None, days=None, fetch=True):
    """🌡 V2: 1h closes of every member of the rolling top-100 (services.mkt_up4h rule) for the UTC days from t_min's day to now (or only
    `days` — pass the days of fills that carry no stamp) → (closes indexed by bar CLOSE, complete?), closes.attrs['universe'] = {day: members}.
    History comes from the 11-06 study cache; later hours are read per (member, UTC day) — /fapi/v1/klines 1h limit 29, weight 1, cached once
    the day has closed."""
    bud = bud or LS.Budget()
    now_ms = int(now_ms)
    if days is None:
        d0 = pd.Timestamp(int(t_min), unit="ms").normalize()
        days = list(pd.date_range(d0, pd.Timestamp(now_ms, unit="ms").normalize(), freq="D"))
    U, complete = v2_universes(days, bud, now_ms, fetch)
    cover = _study_1h_cover()
    data, reading, short = {}, True, False
    for d in sorted(U):
        dm = int(d.value // 10**6)
        late = dm + DAY > cover                     # the day is not complete in the study cache → read it per member
        for s in U[d]:
            if s not in data:
                if ("1h", s) not in _MEMO:
                    _MEMO[("1h", s)] = _study_1h(s)
                data[s] = dict(_MEMO[("1h", s)])
            if late:
                rows = _raw_klines(s, "1h", dm - 5 * HOUR, DAY_PAGE, bud, now_ms, fetch) if reading else None
                if rows is None or getattr(bud, "errors", 0):
                    reading, short, complete = False, True, False   # stop asking; hours past the study cache are left unscored (never a stale bar)
                    continue
                data[s].update(MB.closes_from_rows(rows, now_ms))
    lo = (int(min(U).value // 10**6) - 5 * HOUR) if U else 0
    ser = {s: pd.Series({pd.Timestamp(t, unit="ms"): c for t, c in v.items() if t >= lo}, dtype=float) for s, v in data.items() if v}
    closes = pd.DataFrame(ser).sort_index() if ser else pd.DataFrame()
    closes.attrs["universe"] = dict(U)
    closes.attrs["valid_until"] = pd.Timestamp(cover, unit="ms") if short else None
    return closes, complete


def breadth_series(closes, universe=None):
    """closes (coin × 1h bar CLOSE) → the Market-hot share (% up over 4 h) per bar close: services.mkt_up4h's formula (close_H / close_{H−4h} − 1 > 0 over the members
    with both closes; NaN when < MIN_COINS) with the members of the bar's UTC day (universe {day: members}, default closes.attrs['universe'];
    none → every column). Every bar where any member has a close is kept (NaN = the rule's NULL, never a fall-back to an older bar). PURE."""
    if closes is None or not len(closes):
        return pd.Series(dtype=float)
    U = universe if universe is not None else closes.attrs.get("universe")
    idx = closes.index
    prev = closes.reindex(idx - pd.Timedelta(hours=4))
    prev.index = idx
    r4 = closes / prev - 1
    out = []
    groups = [(None, list(closes.columns))] if not U else [(d, [m for m in U[d] if m in closes.columns]) for d in sorted(U)]
    for d, mem in groups:
        rows = idx if d is None else idx[(idx >= d) & (idx < d + pd.Timedelta(days=1))]
        if not len(rows) or not mem:
            continue
        sub = r4.loc[rows, mem]
        n = sub.notna().sum(axis=1)
        b = ((sub > 0).sum(axis=1) / n.where(n > 0)).where(n >= MIN_COINS)
        out.append(b[closes.loc[rows, mem].notna().any(axis=1)])
    res = pd.concat(out).sort_index() if out else pd.Series(dtype=float)
    vu = closes.attrs.get("valid_until")
    if vu is not None and U:                        # a read that stopped short: every later hour of the read days is NaN (unscored), never an older bar
        late = pd.date_range(vu + pd.Timedelta(hours=1), max(U) + pd.Timedelta(hours=23), freq="1h")
        res = pd.concat([res[res.index <= vu], pd.Series(np.nan, index=late)]).sort_index()
    return res


def needs_rebuild(c):
    """boolean mask of the fills WITHOUT the engine's own V2 stamp (they need the rebuild). PURE."""
    if STAMP not in c:
        return pd.Series(True, index=c.index)
    return pd.to_numeric(c[STAMP], errors="coerce").isna()


def score(c, breadth):
    """fills + the breadth series → zone / rest / unscored. A fill carrying the engine's V2 stamp (entry_mkt_up4h_pct) reads its stamp;
    the others read the rebuilt series on the last closed bar at the entry (a bar more than 2 h old = no reading). PURE."""
    c = c.copy()
    c["breadth"] = np.nan
    if len(c) and breadth is not None and len(breadth):
        idx = breadth.index.values.astype("datetime64[ns]")
        o = c.o.values.astype("datetime64[ns]")
        i = np.searchsorted(idx, o, side="right") - 1
        ok = (i >= 0) & ((o - idx[np.clip(i, 0, len(idx) - 1)]) <= np.timedelta64(2, "h"))
        c.loc[ok, "breadth"] = breadth.values[i[ok]]
    if STAMP in c and len(c):
        st = pd.to_numeric(c[STAMP], errors="coerce")
        c.loc[st.notna(), "breadth"] = st[st.notna()]
    c["zone"] = np.where(c.breadth.isna(), "unscored", np.where(c.breadth > HOT_CUT, "zone", "rest"))
    return c


def p_zone_below(zone, rest, n=BOOT_N, seed=BOOT_SEED):
    """day-clustered bootstrap P(zone avg < rest avg), resampling UTC days across both groups."""
    days = sorted(set(zone.day) | set(rest.day))
    z = {d: zone.pct.values[zone.day.values == d] for d in days}
    r = {d: rest.pct.values[rest.day.values == d] for d in days}
    rng, hit, tot = np.random.default_rng(seed), 0, 0
    for _ in range(n):
        pick = [days[i] for i in rng.integers(0, len(days), len(days))]
        a, b = np.concatenate([z[d] for d in pick]), np.concatenate([r[d] for d in pick])
        if len(a) and len(b):
            tot += 1
            hit += a.mean() < b.mean()
    return hit / tot if tot else None


def verdict(zone, rest):
    """zone / rest = counted fills (pct, day, pair) → (state, detail). PURE."""
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz, mr = float(zone.pct.mean()), (float(rest.pct.mean()) if len(rest) else float("nan"))
    p = p_zone_below(zone, rest) if len(rest) else None
    sd, sp = NF.gross_loss_share(zone.pct.values, zone.day.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {(zone.pct > 0).mean():.0%} · avg {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · P(zone < rest) "
           f"{(p if p is not None else float('nan')):.2f} · top day {sd:.0%} / top pair {sp:.0%} of the zone's gross loss")
    if len(rest) and mz >= mr:
        return "RETIRED", det
    if mz <= AVG_MAX and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def before_after(sc):
    """before → after blocking the zone: N · WR · avg · the daily compound return with each trade = the account (comparison only). PURE."""
    if not len(sc):
        return "no scored fills"
    span = pd.date_range(sc.o.min().normalize(), sc.o.max().normalize(), freq="D").strftime("%Y-%m-%d")

    def dcr(g):
        return (((1 + (g.groupby("day").pct.sum() / 100).reindex(span).fillna(0)).prod()) ** (1 / len(span)) - 1) * 100
    k = sc[sc.zone == "rest"]
    a = f"{len(k)} · {(k.pct > 0).mean():.0%} · {k.pct.mean():+.3f} % · {dcr(k):+.3f} %/day" if len(k) else "0 fills"
    return f"before {len(sc)} · {(sc.pct > 0).mean():.0%} · {sc.pct.mean():+.3f} % · {dcr(sc):+.3f} %/day → after {a}"


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


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, breadth_fn=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 416 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write.
    breadth_fn(t_min_ms, now_ms) → (breadth series, complete?); None → public klines."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 ML_MKT_HOT — bought with > {HOT_CUT:.0%} of the market up over 4 h: momentum longs, WILLY, WILLY_HT (LITE shown) (pre-registered 2026-10-30, DECISION_LOG {DL}, OBSERVE only)", ""]
    since_given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="ML_MKT_HOT watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    complete = True
    if len(keep):
        miss = keep[needs_rebuild(keep)]            # 🌡 (489) stamped fills read their own V2 stamp; only the others are rebuilt
        br = pd.Series(dtype=float)
        if len(miss):
            t_min = int(miss.o.min().value // 10**6)
            if breadth_fn is None:
                closes, complete = fetch_closes(t_min, now_ms, days=sorted(set(miss.o.dt.normalize())))
                br = breadth_series(closes) if len(closes) and complete else pd.Series(dtype=float)   # a partial read is not the measure → unscored
            else:
                br, complete = breadth_fn(t_min, now_ms)
        keep = score(keep, br)
    else:
        keep = keep.assign(zone=pd.Series(dtype=object), breadth=pd.Series(dtype=float))
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    L += [f"Cohort: CLOSED full-size fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if since_given is None else 'floor given'}) · each sleeve read separately · "
          f"{_GK.operator_closed_line(ox)}.", ""]
    if not complete:
        L += ["⚠ Market-hot read incomplete this run (budget, cooldown or too few coins): fills are unscored until a later run; nothing freezes.", ""]
    changed = False
    for key, es in DECIDED + DISPLAY:
        s = keep[keep.entry_strategy.astype(str) == es] if len(keep) else keep
        sc = s[s.zone != "unscored"] if len(s) else s
        zone, rest, uns = sc[sc.zone == "zone"], sc[sc.zone == "rest"], s[s.zone == "unscored"]
        shown = (key, es) in DISPLAY
        L += [f"### {'MOMENTUM LONG' if es == 'MOMENTUM' else es}" + (" (display only, never decided)" if shown else ""), "", "| Group | N | days | WR | avg % | Σ$ as traded |", "|---|---|---|---|---|---|",
              _row(f"**zone (market hot, > {HOT_CUT:.0%} up)**", zone), _row("rest", rest), _row("UNSCORED (no Market-hot reading) — never counted", uns), ""]
        why = []
        rec = st.get(key) if (ok and not shown) else None
        damaged = rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec))
        rec = None if damaged else rec     # a damaged record never re-freezes silently and blocks only its own sleeve
        if damaged:
            L.append(f"⚠ {key}: frozen record damaged — operator restore needed; this sleeve does not freeze until then.")
        if not rec and ok and not shown and not damaged:
            pre = crossing_prefix(sc) if len(sc) else None
            if pre is not None:
                last = pre.o.max()
                if not complete:
                    why.append("the Market-hot read stopped early this run")
                elif (uns.o <= last).any():
                    why.append("an unscored fill sits inside the prefix")
                elif newest is None or newest < last + pd.Timedelta(minutes=WAIT_MIN):
                    why.append(f"newest export < the crossing fill + {WAIT_MIN} min (a fill could still be open)")
                else:
                    state, det = verdict(pre[pre.zone == "zone"], pre[pre.zone == "rest"])
                    rec = dict(state=state, detail=det, before_after=before_after(pre), at=f"{last:%Y-%m-%d %H:%M} UTC",
                               run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                               keys=[f"{a:%Y-%m-%dT%H:%M:%S}|{b}|{g}" for a, b, g in zip(pre.o, pre.pair, pre.zone)])
                    if state_path:
                        st[key] = rec
                        changed = True
        live_state, live_det = verdict(zone, rest)
        if rec and ok and state_path:
            L.append(f"**Frozen verdict (crossing at fill {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
            L.append(f"Before → after at the crossing: {rec.get('before_after', '–')}")
            L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
        else:
            L.append(f"Now: ⏳ {live_state} ({live_det})." if not shown else f"Now: {live_det}.")
        L.append("Before → after now: " + before_after(sc) + ".")
        if why:
            L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
        L.append("")
    if changed and ok and state_path:
        NF.du_save_state(st, state_path)
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += ["", f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}, operator's, not the expectancy bar): at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} UTC days, decided ONCE — zone avg ≥ rest avg → RETIRED; {CANDIDATE} iff zone avg ≤ {AVG_MAX:+.2f} % ∧ "
          f"P(zone < rest) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no day / pair ≥ {SHARE_MAX:.0%} of the zone's gross loss; else KEEP OBSERVING.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
