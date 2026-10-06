#!/usr/bin/env python3
"""🎯 Scout — FRENZY / WIDE per-fill EXIT SHADOWS (pre-registered 2026-10-05, operator after RLC 10-05; DECISION_LOG 214). OBSERVE only —
never changes config, never a trade. Called by scripts/opportunity_scout.py every run (never breaks it). Public Binance 1m / 5m / 1h klines.

For every live FRENZY_LONG / FRENZY_WIDE fill in the orders exports (~/Downloads, MANUAL excluded), re-priced from the ACTUAL entry price on
1m bars (low before high inside a minute, a line set by a minute never closes in that minute, fill at the line or at the open when a minute
gaps through it), net of 0.09 % fees, 12 h cap — the same accounting as scripts/frenzy_fill_table (Oct-5 table):
  LOCK2   the live exit: −3 until +3, then max(+2, peak − 2)              LOCK3   the same with a 3-pt trail
  EMA20 / EMA50   −3 until +3, then a +2 floor and out at the first 5m close below the 5m EMA20 / EMA50
  FIX3    the old fixed +3 / −3                                             ACTUAL  the bot's own result (the exit live at the time)
⛔ SHADOWS 1 and 2 RETIRED 2026-10-06 (operator; DECISION_LOG 218): on the engine cohort (reports/FRENZY_REDO_EXITS_2026-10-05.md) ATRE is
  −0.333 vs the lock (CI −0.54…−0.13, both halves negative) and NOTSTRETCHED's first-entry half is negative (−0.21 / −0.14); re-entry of
  every kind is refuted (FRENZY_REDO_REENTRY_2026-10-05.md). No longer computed; their old columns stay in the CSV for the record.
SHADOW 1 (retired) — FRENZY_ATRE_SHADOW (reports/FRENZY_ATR_AND_PURE_EMA_EXIT_2026-10-05.md, frozen): FRENZY_LONG first entries only; −3 until +3, then a
  +2 floor plus an exit at the first 5m close whose ATR% (Wilder 14 on the last 300 closed bars ÷ close, the engine's own ruler) is below the
  signal bar's (the stamped entry_atr_pct); 12 h cap. BAR (read at N ≥ 60 entries on ≥ 30 days): Δ(shadow − LOCK2) mean > 0 ∧ day-block 95 %
  CI low > 0 ∧ mean > 0 without the top 5 and the top 10 ∧ no pair > 35 % of the gain ∧ both halves > 0. Otherwise retired; never re-fit.
SHADOW 2 (retired) — TYPE_III_NOTSTRETCHED (reports/FRENZY_EXIT_SELECTOR_2026-10-05.md, frozen): a FRENZY or WIDE fill whose stamped
  entry_frenzy_vs_vwap_pct ≤ +5.0 → its EMA50 shadow exit, then RE-ENTRY #1: the first 5m bar of the same UTC day, closing ≥ 15 min after
  the shadow exit, on which the FRENZY state is ON (services.frenzy.frenzy_walk on the 1500-bar window) and the bar is red / flat; entered at
  the next minute's open, EMA50 shadow exit. BAR (read at N ≥ 60 re-entry #1 fills on ≥ 20 days): re-entry mean > 0 ∧ day-block CI low > 0 ∧
  mean > 0 without the top 5 and the top 10 ∧ no pair > 35 % of the gain ∧ both halves > 0. Otherwise retired; never re-fit the +5.0 %.
TRACKER 5 — ATR_FAST_LOCK3 (reports/FRENZY_REDO_EXITS_2026-10-05.md §5a, frozen, observe-only): every FRENZY / WIDE first entry from
  SOFT_FROM tagged by the 30-min ATR% change at the signal bar — (ATR% now ÷ ATR% 6 bars earlier − 1) × 100, ATR = EWM(α 1/14) of the true
  range ÷ close on 5m bars (the study's atr_series) — "fast" when > +14.3 %. Year (engine cohort, pooled): the 3-pt trail beat the lock by
  +0.139 %/trade in that top tercile (CI +0.00…+0.29, both halves +, 2D null p 0.01) but fails drop-top-10 (−0.065) and walk-forward picked
  nothing. BAR (read on LOCK3 − LOCK2 for fast fills): ≥ 30 fills on ≥ 15 days → review candidate only if mean > 0 ∧ day CI low > 0 ∧ > 0
  without the top 5 and top 10 ∧ top pair < 50 % of the net; RETIRE if the mean ≤ 0 at ≥ 30 or no verdict by 60.
TRACKER 4 — WIDE_CHOPPY_OBS (DECISION_LOG 215, shipped OBSERVE-ONLY): every WIDE first entry from SOFT_FROM tagged by the engine's own stamp
  entry_frenzy_above_share (% of the episode's 5m closes at / above the spike VWAP at the signal) ≤ 67.8 = "would have been blocked". BAR:
  at ≥ 15 would-block fills → ARM REVIEW if their mean < 0 ∧ day CI high < 0 ∧ WR < 52 %; RETIRE if their mean ≥ 0; at ≥ 30 without an
  arm verdict → RETIRE. Frozen at 67.8, never re-fit (re-validation of the study on the engine's fresh bars pending).
TRACKER 3 — WIDE_BTC_SOFT (reports/FRENZY_REGIME_2026-10-05.md, frozen): every WIDE first entry tagged by BTC's RSI(12) on CLOSED 5m bars at
  the signal bar (ta.RSIIndicator(close, 12) — the study's ruler, NOT the bot's entry_btc_rsi stamp) ≤ 45 = "soft"; BTC 5m EMA20 slope shown
  alongside. Year: soft +1.00 / +0.64 %/day (Jan–Apr / May–Sep) vs not soft +0.09 / −0.34 — at the luck level (p 0.11), not a filter.
  BAR (review candidate only): ≥ 40 fills on ≥ 15 days in EACH state ∧ soft mean of day means > 0 ∧ day-block 95 % CI of the gap
  (soft − not soft) excludes 0 ∧ the not-soft cohort meets the expectancy bar (WR < 51.7 % breakeven ∧ day CI high < 0). RETIRE if the gap CI
  still includes 0 at ≥ 60 fills per state. Never armed from this tracker.
Rows are stored in reports/SCOUT_FRENZY_EXITS.csv (keyed opened_at + pair) so fills survive their export leaving ~/Downloads; a row is FINAL
once its 12 h (and the re-entry's) have passed. 1m bars are coarser than the year studies' ticks (stated on the table).
"""
import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from services.frenzy import frenzy_walk, normal_hour_usd  # noqa: E402
from services.surge import wilder_atr_pct  # noqa: E402

CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_EXITS.csv")
MIN, BAR, H = 60_000, 300_000, 3_600_000
FEE, CAP_MIN, NOTSTRETCHED_MAX, REENTRY_WAIT_MIN = 0.09, 720, 5.0, 15
ATRE_N, ATRE_DAYS, NS_N, NS_DAYS, PAIR_MAX = 60, 30, 60, 20, 35.0
EXITS = ["LOCK2", "LOCK3", "EMA20", "EMA50", "FIX3"]
VER = 5   # row schema / rules version: rows priced by an older version are recomputed (review) · 3 = + BTC RSI(12) · 4 = + above_share · 5 = + atr_chg30
SOFT_RSI, SOFT_N, SOFT_DAYS, SOFT_RETIRE_N = 45.0, 40, 15, 60
SOFT_FROM = "2026-10-06T00:00:00"   # the trackers' cohort floor: WIDE fills opened from their registration (215 / 216)
CHOPPY_MAX, CHOPPY_N, CHOPPY_RETIRE_N = 67.8, 15, 30   # 🌀 215 observe-only: the frozen choppy-pump candidate on live WIDE fills
ATRF_MIN, ATRF_N, ATRF_DAYS, ATRF_RETIRE_N = 14.3, 30, 15, 60   # ⚡ ATR_FAST_LOCK3 watch line (FRENZY_REDO_EXITS_2026-10-05.md, frozen)
_BTC5 = {}   # per-run cache: BTC 5m klines by window end


# ─────────────────────────── pure exit walkers (selftest) ───────────────────────────
def walk(m1, e, kind, b5=None, atr0=None):
    """net % (fees in) and exit ms of one exit on 1m rows [open_ms, o, h, l, c] from the entry minute. b5 = DataFrame(t, c, e20, e50, atr)
    of 5m bars for the structure / ATR exits. Returns (pnl, exit_ms, how) or (None, None, 'no data')."""
    if not m1:
        return None, None, "no data"
    net = lambda p: (p / e - 1) * 100 - FEE
    trail = 3.0 if kind == "LOCK3" else 2.0
    pk = -1e9
    b5c = {int(t): (c, e20, e50, a) for t, c, e20, e50, a in b5.itertuples(index=False)} if b5 is not None else {}
    t_end = int(m1[0][0]) + CAP_MIN * MIN           # 12 h of CLOCK time, not of rows (a kline gap never stretches it — review)
    prev = None
    for b in m1:
        t, o, h, l, c = b[0], b[1], b[2], b[3], b[4]
        if t >= t_end:
            return net(prev[4]), t_end, "12 h cap"
        prev = b
        if kind == "FIX3":
            if net(l) <= -3:
                return -3.0, t + MIN, "stop"
            if net(h) >= 3:
                return 3.0, t + MIN, "take profit"
            continue
        armed = pk >= 3
        line = (max(2.0, pk - trail) if kind.startswith("LOCK") else 2.0) if armed else -3.0
        lpx = e * (1 + (line + FEE) / 100)
        if l <= lpx:
            return net(min(o, lpx)), t + MIN, ("floor / trail" if armed else "stop")
        pk = max(pk, net(h))
        if pk >= 3 and kind in ("EMA20", "EMA50", "ATRE") and (t + MIN) % BAR == 0:
            r = b5c.get(t + MIN - BAR)
            if r is not None:
                cc, e20, e50, a = r
                if (kind == "EMA20" and cc < e20) or (kind == "EMA50" and cc < e50) or (kind == "ATRE" and atr0 is not None and a is not None and a < atr0):
                    return net(cc), t + MIN, ("5m close < EMA" if kind != "ATRE" else "ATR < entry")
    return net(prev[4]), prev[0] + MIN, ("12 h cap" if prev[0] + MIN >= t_end else "open")


def day_ci(x, days, n=3000, seed=7):
    g = pd.DataFrame(dict(x=x, d=days)).groupby("d").x.agg(["sum", "count"])
    if len(g) < 3:
        return None
    s, c = g["sum"].values, g["count"].values
    r = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    return tuple(np.percentile(s[r].sum(1) / c[r].sum(1), [2.5, 97.5]))


def bar_check(df, col, n_min, d_min):
    """the frozen review bar on a column of per-fill values (Δ or re-entry %). → (state, text)."""
    x = df[col].dropna()
    g = df.loc[x.index]
    n, nd = len(x), g.day.nunique()
    if n < n_min or nd < d_min:
        return "collecting", f"{n}/{n_min} fills · {nd}/{d_min} days"
    ci = day_ci(x.values, g.day.values)
    top = x.sort_values(ascending=False)
    gain = g.assign(v=x).groupby("pair").v.sum()
    pshare = (gain.max() / x.sum() * 100) if x.sum() > 0 else float("inf")   # the studies' ruler: top pair ÷ the NET total (review)
    h1 = g.day < g.day.sort_values().iloc[len(g) // 2]
    ok = (x.mean() > 0 and ci and ci[0] > 0 and top.iloc[5:].mean() > 0 and top.iloc[10:].mean() > 0 and pshare <= PAIR_MAX
          and x[h1].mean() > 0 and x[~h1].mean() > 0)
    return ("review" if ok else "retire"), (f"mean {x.mean():+.3f} · CI [{ci[0]:+.2f}, {ci[1]:+.2f}] · w/o top5 {top.iloc[5:].mean():+.3f} · "
                                            f"w/o top10 {top.iloc[10:].mean():+.3f} · top pair {pshare:.0f} % · halves {x[h1].mean():+.2f} / {x[~h1].mean():+.2f}")


# ─────────────────────────── data ───────────────────────────
def _kl(sym, tf, start, end):
    step = {"1m": MIN, "5m": BAR, "1h": H}[tf]
    out, s = {}, int(start)
    while s < end:
        q = urllib.parse.urlencode(dict(symbol=sym, interval=tf, startTime=s, endTime=int(end), limit=1500))
        r = json.loads(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?{q}", timeout=20).read())
        if not r:
            break
        for x in r:
            out[int(x[0])] = [int(x[0])] + [float(v) for v in x[1:6]]
        nxt = int(r[-1][0]) + step
        if nxt <= s:
            break
        s = nxt
        time.sleep(0.05)
    return [out[k] for k in sorted(out)]


def _fills():
    fr = []
    cols = ("opened_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "entry_atr_pct", "entry_frenzy_vs_vwap_pct",
            "entry_frenzy_above_share")
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols)
        except Exception:
            continue
        if {"opened_at", "entry_strategy", "entry_price"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame(columns=list(cols) + ["k"])
    o = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable").reindex(columns=list(cols) + ["_m"])   # an export missing a column never breaks the section
    o["k"] = o.opened_at.astype(str).str[:19]
    o = o.drop_duplicates(["k", "pair", "direction"], keep="last")
    o = o[o.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (o.direction.astype(str) == "LONG")]
    return o


def btc_soft_reading(t_in):
    """(RSI(12), EMA20 slope %) of BTC on the CLOSED 5m bars up to the signal bar (the last bar closed at or before the entry). Wilder RSI as
    ta.RSIIndicator (ewm alpha 1/12, adjust=False). None, None when unreadable."""
    end = (int(t_in) // BAR) * BAR                     # bars with open < end are closed by the entry
    if end not in _BTC5:
        _BTC5[end] = [b for b in _kl("BTCUSDT", "5m", end - 400 * BAR, end) if b[0] + BAR <= end]
    rows = _BTC5[end]
    if len(rows) < 100 or rows[-1][0] != end - BAR:   # the SIGNAL bar itself must be there — never read the bar before it (review)
        return None, None
    c = pd.Series([r[4] for r in rows], dtype=float)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 12, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / 12, adjust=False).mean()
    rsi = float((100 - 100 / (1 + up / dn)).iloc[-1]) if float(dn.iloc[-1]) > 0 else 100.0
    e20 = c.ewm(span=20, adjust=False).mean()
    return rsi, float((e20.iloc[-1] / e20.iloc[-2] - 1) * 100)


def _cfg():
    c = json.load(open(os.path.join(ROOT, "trading_config.json")))
    return SimpleNamespace(**{**c, **(c.get("thresholds") or {})})


def _price(r, th, now_ms, first):
    """one fill → dict of every exit + the shadows (first = the earliest fill of its sleeve / pair / UTC day: the frozen shadows read first
    entries only — review). Raises on data trouble (the caller keeps the old row)."""
    sym = str(r.pair); e = float(r.entry_price)
    t_in = int(pd.Timestamp(r.k, tz="UTC").value // 1_000_000)
    m0 = t_in // MIN * MIN
    day_end = (t_in // 86_400_000 + 1) * 86_400_000
    horizon = min(now_ms, day_end + CAP_MIN * MIN)
    m1 = [b for b in _kl(sym, "1m", m0, horizon) if b[0] + MIN <= now_ms]
    b5raw = [b for b in _kl(sym, "5m", m0 - 1800 * BAR, horizon) if b[0] + BAR <= now_ms]
    if len(m1) < 2 or len(b5raw) < 400:
        raise ValueError("klines unavailable")
    b5 = pd.DataFrame(b5raw, columns=["t", "o", "h", "l", "c", "v"])
    b5["e20"] = b5.c.ewm(span=20, adjust=False).mean(); b5["e50"] = b5.c.ewm(span=50, adjust=False).mean()
    atr = [None] * len(b5raw)
    for i in range(len(b5raw)):
        if b5raw[i][0] >= m0 - BAR:               # only bars from the signal bar on are ever read
            atr[i] = wilder_atr_pct(b5raw[max(0, i - 299):i + 1])
    b5["atr"] = atr
    # ⚡ ATR_FAST_LOCK3: the study's ATR ruler (EWM α 1/14 of the true range ÷ close, over the whole fetched history), signal bar = the last
    # bar closed by the entry; change vs 6 bars earlier
    _h = b5.h.values; _l = b5.l.values; _c = b5.c.values
    _pc = np.r_[_c[0], _c[:-1]]; _tr = np.maximum(_h - _l, np.maximum(abs(_h - _pc), abs(_l - _pc)))
    _atrp = pd.Series(_tr).ewm(alpha=1 / 14, adjust=False).mean().values / _c * 100
    _js = int(np.searchsorted(b5.t.values, (t_in // BAR) * BAR - BAR))
    atr_chg30 = (float(_atrp[_js] / _atrp[_js - 6] - 1) * 100
                 if _js < len(b5) and int(b5.t.values[_js]) == (t_in // BAR) * BAR - BAR and _js >= 30 else None)
    b5k = b5[["t", "c", "e20", "e50", "atr"]]
    m1in = [list(b) for b in m1 if b[0] >= m0]
    if m1in and m1in[0][0] == m0:   # the entry minute: only its prints AFTER the fill count → flatten it to entry → its close (review)
        m1in[0] = [m0, e, max(e, m1in[0][4]), min(e, m1in[0][4]), m1in[0][4], m1in[0][5]]
    out = dict(k=r.k, pair=sym, sleeve=str(r.entry_strategy).replace("FRENZY_", ""), day=r.k[:10], entry=e, first=bool(first), ver=VER,
               closed=str(r.status).upper() == "CLOSED",
               actual=float(r.pnl_percentage) if pd.notna(r.pnl_percentage) and str(r.status).upper() == "CLOSED" else None,
               atr_entry=float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) else None,
               vs_vwap=float(r.entry_frenzy_vs_vwap_pct) if pd.notna(r.entry_frenzy_vs_vwap_pct) else None,
               above_share=float(r.entry_frenzy_above_share) if pd.notna(r.entry_frenzy_above_share) else None)
    fin = [out["closed"]]
    ema50_exit = None
    for kind in EXITS:
        p, x, how = walk(m1in, e, kind, b5k)
        out[kind] = p; fin.append(how != "open")
        if kind == "EMA50" and how != "open":
            ema50_exit = x                          # the re-entry search starts only from a REAL shadow exit (review)
    # ⛔ ATRE shadow retired (218) — no longer priced
    out["atr_chg30"] = atr_chg30
    out["btc_rsi12"], out["btc_e20_slope"] = btc_soft_reading(t_in) if out["sleeve"] == "WIDE" else (None, None)
    if out["sleeve"] == "WIDE" and out["btc_rsi12"] is None:
        fin.append(False)                                 # a WIDE row without its BTC reading stays provisional → retried next run (review)
    out["re1"] = None; out["re1_at"] = None
    atr_max = float(getattr(th, "frenzy_max_atr_pct", 2.5) or 2.5)
    if False:   # ⛔ NOTSTRETCHED re-entry shadow retired (218) — the search below is kept only for the record
        h1 = _kl(sym, "1h", m0 - 800 * H, m0)
        cand = [i for i, b in enumerate(b5raw) if b[0] + BAR >= ema50_exit + REENTRY_WAIT_MIN * MIN and b[0] + BAR < day_end]
        for i in cand:
            bar = b5raw[i]; close_ms = bar[0] + BAR
            nh = normal_hour_usd(h1, bar[0])
            ep = frenzy_walk(b5raw[max(0, i - 1499):i + 1], nh, th) if nh else None
            if not (ep and ep.get("in_state")):
                continue
            a_i = wilder_atr_pct(b5raw[max(0, i - 299):i + 1])
            red = bar[4] <= bar[1]
            # the study's chain rules (frenzy_exit_combo_test PART 2): FRENZY = ON ∧ red / flat ∧ ATR ≤ the cap; WIDE = its fresh bars, or
            # ON ∧ red / flat ∧ ATR above the cap. (The live market-volume gate is NOT applied here — stated on the table.)
            ok = (red and a_i is not None and a_i <= atr_max) if out["sleeve"] == "LONG" else \
                 (bool(ep.get("fresh_on")) or (red and a_i is not None and a_i > atr_max))
            if not ok:
                continue
            m1r = [b for b in m1 if b[0] >= close_ms]
            if not m1r:
                break
            p, x, how = walk(m1r, m1r[0][1], "EMA50", b5k)
            out["re1"] = p; out["re1_at"] = pd.Timestamp(close_ms, unit="ms").strftime("%m-%d %H:%M"); fin.append(how != "open")
            break
        fin.append(now_ms >= day_end + CAP_MIN * MIN or out["re1"] is not None)
    out["final"] = all(fin)
    return out


def run(now_ms=None):
    now_ms = int(now_ms or time.time() * 1000)
    old = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
    if len(old) and "ver" not in old:
        old["ver"] = 1
    done = {(a, b) for a, b, f, v in zip(old.get("k", []), old.get("pair", []), old.get("final", []), old.get("ver", []))
            if str(f) in ("True", "1", "1.0") and str(v) in (str(VER), f"{VER}.0")}
    th = _cfg(); rows = []; err = 0
    F = _fills()
    # first entry = the earliest fill of its sleeve / pair / UTC day across the stored rows AND the exports (an old export can surface late)
    keys = pd.concat([F[["k", "pair", "entry_strategy"]].assign(sl=F.entry_strategy.astype(str).str.replace("FRENZY_", "")),
                      old[["k", "pair", "sleeve"]].rename(columns={"sleeve": "sl"}) if len(old) else pd.DataFrame(columns=["k", "pair", "sl"])],
                     ignore_index=True)
    keys["day"] = keys.k.astype(str).str[:10]
    firsts = set(keys.sort_values("k").drop_duplicates(["sl", "pair", "day"]).apply(lambda x: (x.k, x.pair), axis=1)) if len(keys) else set()
    for r in F.itertuples():
        if (r.k, r.pair) in done:
            continue
        try:
            rows.append(_price(r, th, now_ms, (r.k, r.pair) in firsts))
        except Exception:
            err += 1
    new = pd.DataFrame(rows)
    if len(new) and len(old):   # the retired shadows' recorded values survive a reprice (review): copy them forward
        keep_cols = [c for c in ("ATRE", "re1", "re1_at") if c in old]
        if keep_cols:
            prev = old.drop_duplicates(["k", "pair"], keep="last").set_index(["k", "pair"])[keep_cols]
            idx = pd.MultiIndex.from_arrays([new.k, new.pair])
            for c in keep_cols:
                new[c] = prev[c].reindex(idx).values
    allr = pd.concat([old, new], ignore_index=True) if rows else old
    if len(allr):
        allr = allr.drop_duplicates(["k", "pair"], keep="last").sort_values("k")
        allr["first"] = [(k, p) in firsts for k, p in zip(allr.k, allr.pair)]   # every stored row's key is in `keys`, so this is complete
        tmp = CSV + ".tmp"; allr.to_csv(tmp, index=False); os.replace(tmp, CSV)
    L = ["## 🎯 FRENZY / WIDE exit shadows per fill (pre-registered, OBSERVE only — the live exit stays the lock)", "",
         "Every live FRENZY / WIDE fill re-priced from its ACTUAL entry on 1m bars (fees in, 12 h cap; coarser than the year studies' ticks). "
         "LOCK2 = live exit · LOCK3 = 3-pt trail · EMA20/EMA50 = +2 floor then the first 5m close below the EMA · FIX3 = old fixed +3/−3 · "
         "ATR Δ30m = the 30-min ATR change at the signal (ATR_FAST_LOCK3 watch) · "
         "Trackers read FIRST entries only (² = a later fill of its pair-day). Actual = the exit live at the time (the fixed +3 TP shipped 10-04 19:23, the lock 10-05 ~16:00); "
         "FIX3 fills at exactly ±3 (reference column). ᵖ = not final yet.", ""]
    if not len(allr):
        return L + ["No FRENZY / WIDE fill in the exports yet.", ""]
    show = allr.tail(15)
    L += ["| Opened UTC | Pair | Sleeve | ATR | ATR Δ30m | vs avg | Actual | LOCK2 | LOCK3 | EMA20 | EMA50 | FIX3 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):+.2f}"
    for r in show.itertuples():
        fl = ("" if str(r.final) in ("True", "1", "1.0") else "ᵖ") + ("" if r.first else "²")
        L.append(f"| {str(r.k)[5:16].replace('T', ' ')}{fl} | {str(r.pair).replace('USDT', '')} | {r.sleeve} | {f(r.atr_entry)} | {f(getattr(r, 'atr_chg30', None))} | "
                 f"{f(r.vs_vwap)} | {f(r.actual)} | {f(r.LOCK2)} | {f(r.LOCK3)} | {f(r.EMA20)} | {f(r.EMA50)} | {f(r.FIX3)} |")
    fin = allr[allr.final.astype(str).isin(["True", "1", "1.0"])].copy()
    L += ["", "| Final fills | N | " + " | ".join(EXITS) + " |", "|---|---|" + "---|" * len(EXITS)]
    for nm, g in (("all", fin), ("FRENZY_LONG", fin[fin.sleeve == "LONG"]), ("WIDE", fin[fin.sleeve == "WIDE"])):
        if len(g):
            L.append(f"| {nm} | {len(g)} | " + " | ".join(f"{g[c].mean():+.2f}" for c in EXITS) + " |")
    L += ["", "_ATRE and NOTSTRETCHED shadows retired 2026-10-06 (DECISION_LOG 218: evidence reversed on the engine cohort)._"]
    if "atr_chg30" in fin:
        st, tx = atrfast_check(fin[fin["first"] & fin.atr_chg30.notna() & (fin.k.astype(str) >= SOFT_FROM)])
        L += [f"**ATR_FAST_LOCK3 watch (30-min ATR change > +{ATRF_MIN:g} % at the signal → 3-pt trail vs the lock):** "
              + {"review": "📋 REVIEW CANDIDATE", "retire": "❌ retire", "collecting": "⏳ collecting"}.get(st, st) + f" ({tx})"]
    if "above_share" in fin:
        st, tx = choppy_check(fin[(fin.sleeve == "WIDE") & fin["first"] & fin.above_share.notna() & (fin.k.astype(str) >= SOFT_FROM)])
        L += [f"**WIDE_CHOPPY_OBS (215, observe-only — the bot still takes these):** "
              + {"arm_review": "📋 ARM REVIEW (would-block fills clearly losing)", "retire": "❌ retire the candidate", "collecting": "⏳ collecting"}.get(st, st) + f" ({tx})"]
    if "btc_rsi12" in fin:
        st, tx = soft_check(fin[(fin.sleeve == "WIDE") & fin["first"] & fin.btc_rsi12.notna() & (fin.k.astype(str) >= SOFT_FROM)])
        L += [f"**WIDE_BTC_SOFT tracker (BTC 5m RSI(12) ≤ {SOFT_RSI:g} at the signal; WIDE first entries from {SOFT_FROM[:10]}):** "
              + {"review": "📋 REVIEW CANDIDATE", "retire": "❌ gap unclear at ≥ 60 per state → retire", "collecting": "⏳ collecting"}.get(st, st) + f" ({tx})"]
    if err:
        L.append(f"_{err} fill(s) not priced this run (klines unavailable) — retried next run._")
    return L + [""]


def atrfast_check(w):
    """frozen ATR_FAST_LOCK3 bar on first entries with an ATR-change reading: Δ = LOCK3 − LOCK2 on the 'fast' fills."""
    f = w[(w.atr_chg30 > ATRF_MIN) & w.LOCK2.notna() & w.LOCK3.notna()].assign(d=lambda x: x.LOCK3 - x.LOCK2)
    n, nd = len(f), f.day.nunique()
    txt = f"fast fills {n} / {nd} d" + (f" · Δ(3-pt − lock) {f.d.mean():+.2f} %" if n else "") + f" · other fills {len(w) - n} (bar ≥ {ATRF_N} fast fills on ≥ {ATRF_DAYS} days)"
    if n >= ATRF_RETIRE_N or (n >= ATRF_N and f.d.mean() <= 0):   # retirement is checked BEFORE the days gate (review)
        ok_days = nd >= ATRF_DAYS
    elif n < ATRF_N or nd < ATRF_DAYS:
        return "collecting", txt
    else:
        ok_days = True
    ci = day_ci(f.d.values, f.day.values) if nd >= 3 else None
    top = f.d.sort_values(ascending=False)
    pshare = (f.groupby("pair").d.sum().max() / f.d.sum() * 100) if f.d.sum() > 0 else float("inf")
    txt += (f" · CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "") + f" · w/o top5 {top.iloc[5:].mean():+.2f} · w/o top10 {top.iloc[10:].mean():+.2f} · top pair {pshare:.0f} %"
    if ok_days and f.d.mean() > 0 and ci and ci[0] > 0 and top.iloc[5:].mean() > 0 and top.iloc[10:].mean() > 0 and pshare < 50:
        return "review", txt
    if f.d.mean() <= 0 or n >= ATRF_RETIRE_N:
        return "retire", txt
    return "collecting", txt


def choppy_check(w):
    """215 observe-only bar on live WIDE first entries: would-block = stamped above_share ≤ 67.8 (LOCK2 = the live exit's result)."""
    w = w[w.LOCK2.notna()]
    wb, kp = w[w.above_share <= CHOPPY_MAX], w[w.above_share > CHOPPY_MAX]
    mf = lambda g: f"{g.LOCK2.mean():+.2f} %" if len(g) else "–"
    txt = f"would-block {len(wb)} fills / {wb.day.nunique()} d · {mf(wb)}  vs  kept {len(kp)} · {mf(kp)} (bar: {CHOPPY_N} would-block fills)"
    if len(wb) < CHOPPY_N:
        return "collecting", txt
    ci = day_ci(wb.LOCK2.values, wb.day.values)
    if wb.LOCK2.mean() < 0 and ci and ci[1] < 0 and (wb.LOCK2 > 0).mean() * 100 < 52:
        return "arm_review", txt + f" · CI [{ci[0]:+.2f}, {ci[1]:+.2f}]"
    if wb.LOCK2.mean() >= 0 or len(wb) >= CHOPPY_RETIRE_N:
        return "retire", txt
    return "collecting", txt


def soft_check(w):
    """the frozen WIDE_BTC_SOFT bar on WIDE first entries with a BTC RSI(12) reading (LOCK2 = the live exit's result)."""
    soft = w.btc_rsi12 <= SOFT_RSI
    a, b = w[soft], w[~soft]
    na, nb, da, db_ = len(a), len(b), a.day.nunique(), b.day.nunique()
    mf = lambda g: f"{g.LOCK2.mean():+.2f} %" if len(g) else "–"
    txt = f"soft {na} fills / {da} d · {mf(a)}  vs  not soft {nb} / {db_} d · {mf(b)}"
    if min(na, nb) < SOFT_N or min(da, db_) < SOFT_DAYS:
        return "collecting", txt + f" (bar ≥ {SOFT_N} fills on ≥ {SOFT_DAYS} days per state)"
    am, bm = a.groupby("day").LOCK2.mean(), b.groupby("day").LOCK2.mean()
    days = np.array(sorted(set(am.index) | set(bm.index)))
    rng = np.random.default_rng(7); gaps = []
    for _ in range(3000):                       # JOINT day-block bootstrap: one draw of days serves both states (they share days — review)
        dr = rng.choice(days, len(days))
        ga, gb = am.reindex(dr).dropna(), bm.reindex(dr).dropna()
        if len(ga) and len(gb):
            gaps.append(ga.mean() - gb.mean())
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    nci = day_ci(b.LOCK2.values, b.day.values)
    ok = am.mean() > 0 and lo > 0 and (b.LOCK2 > 0).mean() * 100 < 51.7 and nci and nci[1] < 0
    txt += f" · gap CI [{lo:+.2f}, {hi:+.2f}]"
    if ok:
        return "review", txt
    if min(na, nb) >= SOFT_RETIRE_N and lo <= 0:   # gap unclear OR the wrong way (review) — at 60 per state either retires it
        return "retire", txt
    if min(na, nb) >= SOFT_RETIRE_N:
        return "retire", txt + " · gap clear but the soft / not-soft expectancy legs failed"
    return "collecting", txt


def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    t0 = 1_790_000_000_000 // BAR * BAR
    m = lambda ps: [[t0 + i * MIN, p, p, p, p] for i, p in enumerate(ps)]
    chk(abs(walk([[t0, 100, 100, 96, 96]], 100, "LOCK2")[0] + 3.0) < 1e-9, "−3 stop fills at the line inside a minute")
    chk(abs(walk(m([100, 96]), 100, "LOCK2")[0] + 4.09) < 1e-9, "a minute that opens through the stop fills at its open (gap)")
    up = m([100, 104, 110]) + [[t0 + 3 * MIN, 110, 110, 105, 105]]   # trades down through the lines inside one minute
    chk(abs(walk(up, 100, "LOCK2")[0] - 7.91) < 0.02, "lock trails 2 pts below the prior peak (fill at the line)")
    chk(abs(walk(up, 100, "LOCK3")[0] - 6.91) < 0.02, "3-pt trail")
    chk(walk(up, 100, "FIX3")[0] == 3.0, "fixed +3")
    chk(abs(walk(m([100, 104, 110, 100]), 100, "LOCK2")[0] + 0.09) < 1e-9, "a minute opening through the trail line fills at its open")
    chk(walk(m([100, 101]), 100, "LOCK2")[2] == "open", "still open")
    b5 = pd.DataFrame(dict(t=[t0], c=[105.0], e20=[106.0], e50=[104.0], atr=[1.0]))
    p = [100, 104] + [105] * 4
    r20 = walk(m(p), 100, "EMA20", b5); r50 = walk(m(p), 100, "EMA50", b5)
    chk(r20[2] == "5m close < EMA" and abs(r20[0] - 4.91) < 0.02, "EMA20 exit at the 5m close below the EMA")
    chk(r50[2] == "open", "EMA50 holds while the close is above it")
    chk(walk(m(p), 100, "ATRE", b5, atr0=2.0)[2] == "ATR < entry", "ATRE exits when the 5m ATR falls below the entry ATR")
    chk(walk(m(p), 100, "ATRE", b5, atr0=0.5)[2] == "open", "ATRE holds while ATR ≥ entry")
    df = pd.DataFrame(dict(day=[f"d{i}" for i in range(70)], pair=[f"P{i}" for i in range(70)], v=[0.3] * 70))
    chk(bar_check(df, "v", 60, 30)[0] == "review", "all bars met → review")
    chk(bar_check(df.head(10), "v", 60, 30)[0] == "collecting", "small N → collecting")
    df2 = df.copy(); df2.loc[0, "v"] = 500.0; df2.loc[1:, "v"] = -0.1
    chk(bar_check(df2, "v", 60, 30)[0] == "retire", "one-trade lottery → retire")
    gap = [[t0, 100, 100, 100, 100], [t0 + 800 * MIN, 101, 101, 101, 101]]
    chk(walk(gap, 100, "LOCK2")[2] == "12 h cap", "the 12 h cap is clock time — a kline gap cannot stretch it")
    conc = pd.DataFrame(dict(day=[f"d{i}" for i in range(70)], pair=["A"] * 35 + [f"P{i}" for i in range(35)], v=[0.3] * 70))
    chk(bar_check(conc, "v", 60, 30)[0] == "retire", "one pair carrying half the net gain fails the 35 % pair bar")
    aw = pd.DataFrame(dict(day=[f"d{i}" for i in range(40)], pair=[f"P{i}" for i in range(40)], atr_chg30=[20.0] * 35 + [5.0] * 5,
                           LOCK2=[2.0] * 40, LOCK3=[2.6, 2.4, 2.8, 2.5] * 10))
    chk(atrfast_check(aw)[0] == "review", "fast fills where the 3-pt trail beats the lock everywhere → review")
    chk(atrfast_check(aw.assign(LOCK3=1.5))[0] == "retire", "3-pt trail worse → retire")
    chk(atrfast_check(aw.head(20))[0] == "collecting", "< 30 fast fills → collecting")
    big = pd.concat([aw.assign(day=aw.day + s_) for s_ in "ab"], ignore_index=True).assign(LOCK3=[2.4, 2.3] * 40, pair="A")
    chk(atrfast_check(big)[0] == "retire", "≥ 60 fast fills, one pair carrying everything → retire")
    cw = pd.DataFrame(dict(day=[f"d{i}" for i in range(40)], above_share=[50.0] * 20 + [90.0] * 20, LOCK2=[-1.0, -1.2, 0.4, -1.5] * 5 + [0.5] * 20))
    chk(choppy_check(cw)[0] == "arm_review", "would-block fills clearly losing → arm review")
    chk(choppy_check(cw.assign(LOCK2=0.3))[0] == "retire", "would-block fills not losing → retire")
    chk(choppy_check(cw.tail(25))[0] == "collecting", "< 15 would-block fills → collecting")
    w = pd.DataFrame(dict(day=[f"d{i % 30}" for i in range(100)], btc_rsi12=[40.0] * 50 + [60.0] * 50, LOCK2=[0.8] * 50 + [-0.6] * 50))
    chk(soft_check(w)[0] == "review", "soft clearly better, not-soft clearly losing → review candidate")
    chk(soft_check(w.head(60))[0] == "collecting", "< 40 fills in one state → collecting")
    w2 = w.assign(LOCK2=[0.1, 0.1, -0.1, -0.1] * 25, btc_rsi12=[40.0, 60.0] * 50)   # both states the same mix → no gap
    chk(soft_check(pd.concat([w2, w2.assign(day=w2.day + "b")]))[0] == "retire", "no gap at ≥ 60 per state → retire")
    print(f"selftest OK ({ok} checks)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
