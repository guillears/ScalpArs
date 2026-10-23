#!/usr/bin/env python3
"""🐻⏩ Scout — BEARRUN_V2_SHADOW (DECISION_LOG 359, operator 2026-10-23). OBSERVE ONLY — never changes config, never calls the bot API,
never reads server logs. Public Binance fapi only.

WHY             Bear-Run V2 (the earlier BTC 1 h slope + DI trigger) FAILED the forced replay (reports/BEARRUN_V2_REPLAY_s1.md: V2-C1 / C2 lose
                vs T0 under every exit). Clean slides earn under 'hold while ON' (exit d) but false starts cancel it, and nothing at the ON
                bar separates them (reports/BEARRUN_V2_2026-10-22.md). The history cannot give more windows → run V2 as a LIVE SHADOW and
                collect forward windows (10-22-type days count for free).
DEFINITIONS     🔒 frozen (reports/BEARRUN_V2_2026-10-22.md; state machine = scripts/study_bearrun_early_trigger.py v2_state, PRIMARY V2-C1,
                Wilder copied verbatim), evaluated at each CLOSED BTCUSDT 1 h bar close (public fapi klines):
                  ON  : EMA20 slope3 = (EMA20 / EMA20[−3] − 1) × 100 < 0 ∧ −DI > +DI (Wilder 14) ∧ C1: BTC 6 h return ≤ −1.0 %
                  OFF : slope3 ≥ 0 ∨ +DI > −DI on 2 consecutive closed bars ∨ BTC +1.5 % above the window's running low
                  a new ON only on a later hour than the last OFF; a re-ON ≤ 180 min after the OFF continues the SAME window (the study's
                  merge(), MERGE_MIN 180 — keyed on the first ON; the hold-while-ON basket exits at the FIRST OFF).
                ⚠ TWO 1 h ADAPTATIONS (the study's v2_state ran on the 5-MIN grid: r6 from 5-min prices, the 1 h legs read at every 5-min
                step, so ON / OFF could flip intra-hour): ① here everything is read at the 1 h close — an ON can come later than the
                study's, or be MISSED on an intra-hour dip that recovers by the close; ② the bounce leg: the running low starts at the ON
                close; a bar's HIGH is tested against the low of EARLIER bars, its CLOSE against the low including itself (the intrabar
                order of a 1 h bar is unknown). The backtest reference numbers come from the 5-min machine.
C2 label = ≥ 50 % of the top-50 alt USDT perps (24 h quote volume at the ON hour) below their price 24 h earlier — computed
                from the same public 1 h klines (n/a if < 40 of them are readable). A label only — the shadow trigger is C1.
CLASS (24 h)    clean_slide: BTC ≥ 3 % below the ON price within 24 h without a +1.5 % bounce off the running low ·
                false_start (F1): within 24 h BTC above its 24 h high (pre-ON) or +2 % above the ON price before reaching −3 % from that
                high (the study's not-true-start: also 'neither reached in 24 h' → F1_unresolved, as in study false_starts()) ·
                true_other: the rest. Ties inside one 1 h bar count as the bounce / up move first (conservative, the study's strict <).
BASKET (proxy)  equal-weight SHORT of the top-20 USDT perps by 24 h quote volume at the ON hour (alts: BTC, ETH, stablecoins and non-COIN
                underlyings out; universe = today's top-60 by 24 h quote volume, re-ranked on the klines' quote volume at the ON hour).
                Entry = each coin's ON-hour close. These are PROXIES OF THE SLEEVE, NOT THE ENGINE (no engine gates, signals or slippage):
                  d  : hold while ON — exit at the OFF-hour close, +2 % per-coin stop on 1 h highs, 24 h cap, 0.10 % round-trip fees
                  a' : scalp proxy — +0.5 % TP / −1 % stop per coin, first touch on 1 h bars (both in one bar → stop), 6 h cap, 0.10 % fees
T0 COMPARISON   the LIVE Bear-Run (today's trigger) from the orders exports (export_glob, paper + live): BEARRUN_SHORT fills (entry_strategy, or
                a stamped entry_bear_bypass) dedup (opened_at, pair, direction), CLOSED wins — listed per V2 window: inside (opened ≤ OFF) /
                after (≤ 24 h after the OFF or the cap), how late vs the V2 ON, P&L. Operator-closed fills are flagged (scout_gatekit).
COHORT          windows with ON ≥ the DECISION_LOG 359 commit + 10 min (scout_gatekit.dl_deploy_ms). Until the commit exists: NOT evaluable,
                nothing is persisted, the last 72 h of V2 windows print as context (informational).
PERSISTENCE     reports/.scout_bearrun_v2_windows.json — APPEND-ONLY: a cohort window is final 24 h after ON (BTC bars to ON + 24 h and a
                fully priced basket); its record is written once and never rewritten.
🔒 GATE          = the PRE-COMMITTED gate of reports/BEARRUN_V2_2026-10-22.md ('Pre-committed gate (frozen now; window units)'), the only
                terminal decision; each read is decided ONCE and persisted:
                  ARM read at the shortest prefix of ≥ 20 final windows spanning ≥ 6 distinct ISO weeks → ARM_CANDIDATE (probe size
                  bearrun_lev_mult 0.05, revert if the first 10 armed windows Σ ≤ 0 — operator decides, nothing armed by the scout) iff ALL:
                  ① ≥ 20 windows on ≥ 6 weeks · ② Σ (V2-d basket window means) > 0 ∧ window-bootstrap P(Σ > 0) ≥ 0.90 (4,000, seed 7) ·
                  ③ no window ≥ 50 % of Σ · ④ false starts lose ≤ 60 % of what clean slides earn (max(0, −Σ false) ≤ 0.6 × Σ clean, Σ clean
                  > 0) · ⑤ the replay walk-forward held-out total is not negative (frozen ref −253, so ⑤ FAILS until a new replay) → else NO_ARM
                  (failed conditions listed).
                  DROP read at the 30th final window → DROP_SHADOW iff Σ ≤ 0 ∧ false starts cost more than clean slides earn; else KEEP_SHADOW.
                  A read waits (WAITING) while any cohort window opened by its prefix end is not yet final.
                INTERIM checkpoint (≥ 8 windows on ≥ 8 UTC days): printed only — NON-TERMINAL, never frozen, never a 'stays off'.
NETWORK         public fapi only, ≤ 900 used weight (X-MBX-USED-WEIGHT-1M) per run (Binance cap 1,200 / min), paced, the shared cooldown
                (scripts/binance_cooldown.py) honoured; a 418 / 429 arms it and stops the run, 403 stops the run. Every final window's
                klines are cached under reports/backtest_cache/scout_bearrun_v2/. A per-symbol 4xx (not 403 / 418 / 429) drops that coin
                (listed in the record) — only budget / cooldown / network halts leave a window un-final for a retry. The universe is
                snapshotted per window at the first run that prices it (uni_on_<ON>.json, recorded as universe_at). The state JSON is
                read / appended / saved under an exclusive flock.
Usage           venv/bin/python scripts/scout_bearrun_v2.py
"""
import contextlib
import fcntl
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
_R = os.path.dirname(_D)
import binance_cooldown as BC                                # noqa: E402
import export_glob as EG                                     # noqa: E402
import scout_gatekit as _GK                                  # noqa: E402

DL = 359
HOUR = 3_600_000
FAPI = "https://fapi.binance.com"
CACHE = os.path.join(_R, "reports", "backtest_cache", "scout_bearrun_v2")
STATE = os.path.join(_R, "reports", ".scout_bearrun_v2_windows.json")
PATTERN = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")

# 🔒 frozen definitions (PRIMARY V2-C1 of the study)
SLOPE_MAX, R6_MAX, BOUNCE_PCT, RECLAIM_BARS, WARM_BARS = 0.0, -1.0, 1.5, 2, 50
MERGE_MIN = 180                      # study merge(): an ON ≤ 180 min after the previous OFF is the same window
SLIDE_PCT, FALSE_HIGH_PCT, FALSE_UP_PCT, HORIZON_H = 3.0, 3.0, 2.0, 24
C2_TOP, C2_MIN, C2_PCT = 50, 40, 50.0
BASKET_N, UNIVERSE_N = 20, 60
D_STOP, D_CAP_H = 2.0, 24
A_TP, A_STOP, A_CAP_H = 0.5, 1.0, 6
FEE_RT = 0.10
# 🔒 the PRE-COMMITTED gate (reports/BEARRUN_V2_2026-10-22.md 'Pre-committed gate (frozen now; window units)') — the terminal decision
ARM_N, ARM_WEEKS, ARM_P, SHARE_MAX, FS_RATIO_MAX, DROP_N = 20, 6, 0.90, 0.50, 0.60, 30
WF_HELDOUT_SUM = -253.0              # ⑤ the queued replay walk-forward, held-out total (V2 picks −23 / −50 / −58 / −123 vs T0 −9)
WF_SRC = "reports/BEARRUN_V2_2026-10-22.md §Verdict 5"
# interim checkpoint — printed only, NON-TERMINAL, never frozen
INTERIM_N, INTERIM_DAYS = 8, 8
BOOT_N, BOOT_SEED = 4000, 7
CONTEXT_H = 72
BTC_BARS = 499                       # limit < 500 → weight 2
# network
WEIGHT_STOP, PACE_S, MAX_REQ = 900, 0.25, 400
EXCLUDE_BASES = {"BTC", "ETH", "USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USDE", "USD1", "PYUSD", "EUR", "EURI", "AEUR", "RLUSD",
                 "XUSD", "USDS", "BFUSD", "FRAX", "BTCDOM", "USDX", "GUSD", "LUSD", "SUSD"}


# ═══════════════════════════════ network (public, budgeted, cooldown-aware) ═══════════════════════════════
def _default_get(url, timeout=15):
    """→ (status, headers dict, body text). HTTP errors come back as their status (never raise)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, dict(r.headers or {}), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), ""


def _arm_cooldown(code, retry_after, now_ms, path=None):
    """write the SHARED cooldown (ms) — never shortens a longer ban already armed."""
    path = path or BC.SHARED
    try:
        secs = int(float(retry_after))
    except (TypeError, ValueError):
        secs = 7200 if code == 418 else 600
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(str(int(max(now_ms + secs * 1000, BC.read_until_ms(path)))))
        os.replace(tmp, path)
    except Exception:
        pass
    return secs


class Net:
    """one per run: budget (used weight from the header), pacing, the shared cooldown, stop on 418 / 429 / 403."""

    def __init__(self, http_get=None, fetch=True, cooldown_path=None, pace_s=PACE_S, weight_stop=WEIGHT_STOP, max_req=MAX_REQ):
        self.get = http_get or _default_get
        self.fetch, self.cooldown_path, self.pace_s = fetch, cooldown_path, pace_s
        self.weight_stop, self.max_req = weight_stop, max_req
        self.used, self.requests, self.stop, self.why, self.last_code = 0, 0, False, "", None

    def halt(self, why):
        self.stop, self.why = True, self.why or why

    def json(self, path, params, now_ms):
        self.last_code = None
        if self.stop or not self.fetch:
            return None
        if BC.active(self.cooldown_path, now_ms):
            self.halt("shared Binance cooldown active")
            return None
        if self.used >= self.weight_stop:
            self.halt(f"used weight {self.used} ≥ {self.weight_stop}")
            return None
        if self.requests >= self.max_req:
            self.halt(f"{self.requests} requests ≥ {self.max_req} per run")
            return None
        if self.pace_s:
            time.sleep(self.pace_s)
        self.requests += 1
        url = FAPI + path + ("?" + urllib.parse.urlencode(params) if params else "")
        try:
            code, hdr, body = self.get(url)
        except Exception as e:
            self.halt(f"network error {str(e)[:60]}")
            return None
        h = {str(k).lower(): v for k, v in (hdr or {}).items()}
        self.last_code = code
        try:
            self.used = int(h.get("x-mbx-used-weight-1m") or self.used)
        except (TypeError, ValueError):
            pass
        if code in (418, 429):
            _arm_cooldown(code, h.get("retry-after"), now_ms, self.cooldown_path)
            self.halt(f"HTTP {code} — shared cooldown armed")
            return None
        if code == 403:
            self.halt("HTTP 403 (WAF / geo) — run stopped")
            return None
        if code != 200:
            return None
        try:
            return json.loads(body)
        except Exception:
            return None


def _cache_path(name):
    return os.path.join(CACHE, name)


def _cache_get(name):
    p = _cache_path(name)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _cache_put(name, obj):
    try:
        os.makedirs(CACHE, exist_ok=True)
        p = _cache_path(name)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, p)
    except Exception:
        pass


DROPPED = "DROPPED"                  # a per-symbol 4xx (bad / delisted symbol): the coin is dropped and listed, never retried as 'unread'


def klines_1h(sym, start_ms, limit, net, now_ms, cache=True):
    """CLOSED 1 h bars [open_ms, o, h, l, c, quote_vol] from start_ms; cached only when all `limit` bars came back closed.
    None = not readable this run (budget / cooldown / network / 5xx → retried next run) · DROPPED = a per-symbol 4xx other than
    403 / 418 / 429 (those halt the run)."""
    name = f"k_{sym}_{int(start_ms)}_{int(limit)}.json"
    if cache:
        hit = _cache_get(name)
        if hit is not None:
            return hit
    rows = net.json("/fapi/v1/klines", {"symbol": sym, "interval": "1h", "startTime": int(start_ms), "limit": int(limit)}, now_ms)
    if rows is None:
        if not net.stop and net.last_code is not None and 400 <= int(net.last_code) < 500:
            return DROPPED
        return None
    out = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7])] for x in rows if int(x[0]) + HOUR <= now_ms]
    if cache and len(out) == int(limit):
        _cache_put(name, out)
    return out


def universe(net, now_ms):
    """today's top-UNIVERSE_N alt USDT perps by 24 h quote volume (cached per UTC hour). None = not readable."""
    name = f"universe_{now_ms // HOUR}.json"
    hit = _cache_get(name)
    if hit is not None:
        return hit
    info = net.json("/fapi/v1/exchangeInfo", None, now_ms)
    tick = net.json("/fapi/v1/ticker/24hr", None, now_ms) if info is not None else None
    if info is None or tick is None:
        return None
    ok = set()
    for s in info.get("symbols", []):
        if (s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"
                and str(s.get("baseAsset", "")).upper() not in EXCLUDE_BASES
                and str(s.get("underlyingType", "COIN")).upper() == "COIN"):
            ok.add(s["symbol"])
    vol = sorted(((float(t.get("quoteVolume") or 0), t["symbol"]) for t in tick if t.get("symbol") in ok), reverse=True)
    out = [s for _, s in vol[:UNIVERSE_N]]
    _cache_put(name, out)
    return out


# ═══════════════════════════════ BTC indicators + the V2 state machine (1 h closed bars) ═══════════════════════════════
def wilder(h, l, c, n=14):
    """Wilder ±DI / ADX — a verbatim copy of scripts/study_bearrun_early_trigger.py wilder() (that study file is untracked in git, so it is
    copied, not imported; tests pin the copy against the study when the study file is present)."""
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return pdi, ndi, adx


def btc_frame(rows):
    """klines rows → DataFrame t (bar CLOSE ms), o h l c, slope3, pdi, ndi, pdi_run, r6."""
    a = np.asarray(rows, dtype=float) if len(rows) else np.zeros((0, 6))
    D = pd.DataFrame({"open": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4]})
    D["t"] = D.open + HOUR
    e20 = D.c.ewm(span=20, adjust=False).mean()
    D["slope3"] = (e20 / e20.shift(3) - 1) * 100
    pdi, ndi, _ = wilder(D.h, D.l, D.c)
    D["pdi"], D["ndi"] = pdi.values, ndi.values
    run, prev = [], 0
    for g in (D.pdi > D.ndi).values:
        prev = prev + 1 if g else 0
        run.append(prev)
    D["pdi_run"] = run
    D["r6"] = (D.c / D.c.shift(6) - 1) * 100
    return D


def v2_windows(D):
    """the frozen state machine → [dict(on_i, on_ms, off_i, off_ms, off_reason, x_off_i, x_off_ms, fragments)] (off None = still ON at the
    last bar). Fragments are merged like the study (MERGE_MIN 180): a re-ON ≤ 180 min after the previous OFF continues the SAME window
    (keyed on its first ON; off_* = the last OFF). x_off_* = the FIRST OFF = where the hold-while-ON basket (exit d) actually exits."""
    c, h, s, pdi, ndi, prun, r6 = (D[k].values for k in ("c", "h", "slope3", "pdi", "ndi", "pdi_run", "r6"))
    out, on, runlo, off_i, cur = [], False, np.inf, -1, None
    for i in range(len(D)):
        if i < WARM_BARS or np.isnan(s[i]) or np.isnan(pdi[i]) or np.isnan(ndi[i]):
            continue
        if on:
            why = []
            if s[i] >= 0:
                why.append("slope≥0")
            if prun[i] >= RECLAIM_BARS:
                why.append(f"+DI>−DI ×{RECLAIM_BARS}")
            bounce = h[i] > runlo * (1 + BOUNCE_PCT / 100)          # the bar's high vs the low of EARLIER bars
            runlo = min(runlo, float(D.l.values[i]))
            bounce = bounce or c[i] > runlo * (1 + BOUNCE_PCT / 100)  # its close vs the low including itself
            if bounce:
                why.append(f"+{BOUNCE_PCT} % off the low")
            if why:
                on, off_i = False, i
                cur.update(off_i=i, off_ms=int(D.t.values[i]), off_reason=" ∧ ".join(why))
                if cur.get("x_off_i") is None:
                    cur.update(x_off_i=i, x_off_ms=int(D.t.values[i]))
                out.append(cur)
                cur = None
        else:
            if (i > off_i and s[i] < SLOPE_MAX and ndi[i] > pdi[i] and not np.isnan(r6[i]) and r6[i] <= R6_MAX):
                on, runlo = True, float(c[i])
                t_i = int(D.t.values[i])
                if out and out[-1].get("off_ms") is not None and t_i - int(out[-1]["off_ms"]) <= MERGE_MIN * 60_000:
                    cur = out.pop()                                 # a fragment: the same window continues (study MERGE_MIN)
                    cur.update(off_i=None, off_ms=None, off_reason=None, fragments=cur.get("fragments", 1) + 1)
                else:
                    cur = dict(on_i=i, on_ms=t_i, off_i=None, off_ms=None, off_reason=None, x_off_i=None, x_off_ms=None, fragments=1,
                               r6=float(r6[i]), slope3=float(s[i]), di_gap=float(ndi[i] - pdi[i]))
    if cur is not None:
        out.append(cur)
    return out


def classify(D, i0):
    """window class over the 24 h after the ON close (needs bars i0+1 … i0+24) → (class, sub, final?)."""
    c, h, l = D.c.values, D.h.values, D.l.values
    p0 = float(c[i0])
    H = float(np.max(h[max(0, i0 - 23):i0 + 1]))                   # BTC's 24 h high at ON (incl. the ON bar)
    seg = range(i0 + 1, min(len(D), i0 + 1 + HORIZON_H))
    final = (i0 + HORIZON_H) < len(D)
    runlo, slide = p0, False
    for j in seg:
        if h[j] > runlo * (1 + BOUNCE_PCT / 100):
            break
        runlo = min(runlo, l[j])
        if runlo <= p0 * (1 - SLIDE_PCT / 100):
            slide = True
            break
    dn = up = None
    for k, j in enumerate(seg):
        if up is None and (h[j] > H or h[j] >= p0 * (1 + FALSE_UP_PCT / 100)):
            up = k
        if dn is None and l[j] <= H * (1 - FALSE_HIGH_PCT / 100):
            dn = k
    true = dn is not None and (up is None or dn < up)
    if slide:
        return "clean_slide", "", final
    if true:
        return "true_other", "", final
    if up is not None:
        return "false_start", "F1", final
    return ("false_start", "F1_unresolved", final) if final else ("pending", "", final)


# ═══════════════════════════════ basket (proxy of the sleeve, NOT the engine) ═══════════════════════════════
def walk_d(bars, e, x_bar):
    """exit d on one coin: entry = close[e]; +2 % stop on 1 h highs (bars e+1 … x); else the close of bar x. → (pnl %, reason)."""
    en = bars[e][4]
    sl = en * (1 + D_STOP / 100)
    for j in range(e + 1, x_bar + 1):
        if bars[j][1] >= sl:                                        # the bar OPENS above the stop → filled at the gapped open
            return (1 - bars[j][1] / en) * 100 - FEE_RT, "STOP_GAP"
        if bars[j][2] >= sl:
            return -D_STOP - FEE_RT, "STOP"
    return (1 - bars[x_bar][4] / en) * 100 - FEE_RT, "EXIT"


def walk_a(bars, e, last):
    """exit a': +0.5 % TP / −1 % stop, first touch on 1 h bars (both in one bar → stop), 6 h cap at the close."""
    en = bars[e][4]
    x = min(e + A_CAP_H, last)
    sl = en * (1 + A_STOP / 100)
    for j in range(e + 1, x + 1):
        if bars[j][1] >= sl:
            return (1 - bars[j][1] / en) * 100 - FEE_RT, "STOP_GAP"
        if bars[j][2] >= sl:
            return -A_STOP - FEE_RT, "STOP"
        if bars[j][3] <= en * (1 - A_TP / 100):
            return A_TP - FEE_RT, "TP"
    return (1 - bars[x][4] / en) * 100 - FEE_RT, ("CAP" if x == e + A_CAP_H else "OPEN")


def _contig(rows, start_ms):
    """rows indexed by hour from start_ms (None where a bar is missing)."""
    out = {}
    for r in rows or []:
        k = (int(r[0]) - int(start_ms)) // HOUR
        if (int(r[0]) - int(start_ms)) % HOUR == 0 and k >= 0:
            out[k] = r
    return out


def price_window(w, D, net, now_ms, syms, final):
    """→ dict(basket d / a' stats, C2 label, coins) for one window. Needs the window's klines per coin (cached once final)."""
    t_on = int(w["on_ms"])
    start = t_on - 25 * HOUR                                        # bar 0 closes 24 h before ON; bar 24 = the ON bar
    nbars = 25 + HORIZON_H
    got, unread, bad = {}, 0, []
    for s in syms:
        rows = klines_1h(s, start, nbars, net, now_ms, cache=True)
        if rows is None:
            unread += 1
            continue
        if rows == DROPPED:
            bad.append(s)                                           # per-symbol 4xx → out of the universe (listed), never blocks finality
            continue
        got[s] = _contig(rows, start)
    out = dict(coins_read=len(got), coins_unread=unread, dropped_4xx=bad)
    # ranking + C2 on bars 0 … 24 (quote volume of the 24 bars closing ≤ ON)
    rank = []
    for s, B in got.items():
        if all(k in B for k in range(25)):
            rank.append((sum(B[k][5] for k in range(1, 25)), s))
    rank.sort(reverse=True)
    top50 = [s for _, s in rank[:C2_TOP]]
    if len(top50) >= C2_MIN:
        below = sum(1 for s in top50 if got[s][24][4] < got[s][0][4])
        out["c2_pct"] = round(100.0 * below / len(top50), 1)
        out["c2"] = bool(out["c2_pct"] >= C2_PCT)
    else:
        out["c2_pct"], out["c2"] = None, None
    basket = [s for _, s in rank[:BASKET_N]]
    on_i = int(w["on_i"])
    if w.get("x_off_i") is not None:                                 # exit d leaves at the FIRST OFF (a later fragment re-ON is not held)
        x_rel = min(int(w["x_off_i"]) - on_i, D_CAP_H)
    else:
        x_rel = D_CAP_H
    rows_d, rows_a, dropped = [], [], []
    for s in basket:
        B = got[s]
        bars = [B.get(k) for k in range(nbars)]
        last = 24
        for k in range(25, nbars):                                  # contiguous bars after ON only
            if k not in B:
                break
            last = k
        xd = 24 + x_rel
        if last >= xd:
            rows_d.append((s,) + walk_d(bars, 24, xd))
        elif final:
            dropped.append(s)                                       # a gap / halt inside a final window → out of the basket (listed)
            continue
        elif last > 24:
            p_, r_ = walk_d(bars, 24, last)
            rows_d.append((s, p_, r_ if r_.startswith("STOP") else "OPEN"))
        if last > 24:
            rows_a.append((s,) + walk_a(bars, 24, last))
    out["basket"] = basket
    out["d"] = _stats(rows_d)
    out["a"] = _stats(rows_a)
    out["d_rows"] = [[s, round(p, 3), r] for s, p, r in rows_d]
    out["dropped"] = dropped
    out["priced_full"] = bool(final and not unread and len(basket) == min(BASKET_N, len(rank)) and len(rows_d) > 0
                              and all(r != "OPEN" for _, _, r in rows_d))
    return out


def _stats(rows):
    if not rows:
        return dict(n=0, mean=None, wr=None, stops=0)
    v = np.array([p for _, p, _ in rows], float)
    return dict(n=int(len(v)), mean=round(float(v.mean()), 4), wr=round(float((v > 0).mean() * 100), 1),
                stops=int(sum(1 for _, _, r in rows if str(r).startswith("STOP"))))


# ═══════════════════════════════ T0 (live Bear-Run) from the orders exports ═══════════════════════════════
T0_COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason",
           "entry_bear_bypass", "is_paper")


def load_t0(paths=None):
    paths = paths if paths is not None else sorted(EG.glob_exports(PATTERN), key=os.path.getmtime)
    fr = []
    for p in paths:
        try:
            fr.append(pd.read_csv(p, low_memory=False, usecols=lambda c: c in T0_COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(T0_COLS) + ["o_ms", "oc"])
    a = pd.concat(fr, ignore_index=True)
    for c in T0_COLS:
        if c not in a:
            a[c] = None
    byp = a.entry_bear_bypass.fillna("").astype(str).str.strip()               # fillna first: pandas ≥ 3 keeps NaN through astype(str)
    es = a.entry_strategy.fillna("").astype(str).str.strip().str.upper()
    m = (es == "BEARRUN_SHORT") | (byp.ne("") & ~byp.str.lower().isin(("nan", "none")))
    m = m & es.ne("MANUAL") & (a.direction.fillna("").astype(str).str.upper() == "SHORT")
    a = a[m.fillna(False).astype(bool)].copy()
    if not len(a):
        return pd.DataFrame(columns=list(T0_COLS) + ["o_ms", "oc"])
    a["_k"] = a.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + a.pair.astype(str) + "|SHORT"
    a["_c"] = (a.status.astype(str) == "CLOSED").astype(int)
    a = a.assign(_i=range(len(a))).sort_values(["_c", "_i"]).drop_duplicates("_k", keep="last")
    o = pd.to_datetime(a.opened_at.astype(str).str[:19].str.replace(" ", "T"), errors="coerce", utc=True)
    a = a[o.notna()].copy()
    a["o_ms"] = ((o[o.notna()] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64").values   # unit-safe (s / ns)
    a["oc"] = _GK.operator_closed_mask(a).values
    return a.sort_values("o_ms").reset_index(drop=True)


def t0_for(w, T, end_ms=None):
    """T0 fills inside this V2 window (opened ON … OFF / cap) or after it (≤ 24 h after the OFF / cap)."""
    if T is None or not len(T):
        return []
    on = int(w["on_ms"])
    off = int(w["off_ms"]) if w.get("off_ms") else on + D_CAP_H * HOUR
    sel = T[(T.o_ms >= on) & (T.o_ms <= off + 24 * HOUR)]
    out = []
    for r in sel.itertuples():
        out.append(dict(pair=str(r.pair), opened=pd.Timestamp(int(r.o_ms), unit="ms").strftime("%m-%d %H:%M"),
                        where="inside" if r.o_ms <= off else "after", late_h=round((r.o_ms - on) / HOUR, 1),
                        status=str(r.status), pct=None if pd.isna(r.pnl_percentage) else round(float(r.pnl_percentage), 3),
                        usd=None if pd.isna(r.pnl) else round(float(r.pnl), 2), operator_closed=bool(r.oc)))
    return out


def _t0_text(fl):
    if not fl:
        return "T0: no Bear-Run fill"
    return "T0: " + " · ".join(
        f"{f['pair']} {f['opened']} [{f['where']} +{f['late_h']:.1f} h] "
        + (f"{f['pct']:+.2f} % (${f['usd']:+.2f})" if f["pct"] is not None and f["status"] == "CLOSED" else f["status"])
        + (" (operator-closed)" if f["operator_closed"] else "") for f in fl)


# ═══════════════════════════════ persistence (append-only) + the frozen read ═══════════════════════════════
def load_state(path=STATE):
    try:
        with open(path) as f:
            st = json.load(f)
        if isinstance(st, dict) and isinstance(st.get("windows"), list):
            return st
    except Exception:
        pass
    return {"version": 2, "dl": DL, "windows": [], "arm_read": None, "drop_read": None}


def save_state(st, path=STATE):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def append_windows(st, recs):
    """append-only: a window (keyed by on_ms) is written once and never rewritten. → number appended."""
    have = {int(w["on_ms"]) for w in st["windows"]}
    n = 0
    for r in sorted(recs, key=lambda x: x["on_ms"]):
        if int(r["on_ms"]) not in have:
            st["windows"].append(r)
            have.add(int(r["on_ms"]))
            n += 1
    return n


def p_sum_pos(vals, nb=BOOT_N, seed=BOOT_SEED):
    v = np.asarray(vals, float)
    if not len(v):
        return float("nan")
    idx = np.random.default_rng(seed).integers(0, len(v), size=(nb, len(v)))
    return float((v[idx].sum(1) > 0).mean())


def _day(w):
    return pd.Timestamp(int(w["on_ms"]), unit="ms").date()


def _week(w):
    iso = pd.Timestamp(int(w["on_ms"]), unit="ms").isocalendar()
    return (int(iso[0]), int(iso[1]))


def summary(P):
    """the window-unit numbers every read prints: Σ, P(Σ > 0), clean share, largest-window share, FS-loss ratio, days / weeks / span."""
    v = np.array([w["d_mean"] for w in P], float)
    S = float(v.sum()) if len(v) else 0.0
    cl = float(sum(w["d_mean"] for w in P if w["cls"] == "clean_slide"))
    fs = float(sum(w["d_mean"] for w in P if w["cls"] == "false_start"))
    fs_loss = max(0.0, -fs)
    span_w = ((int(P[-1]["on_ms"]) - int(P[0]["on_ms"])) / (7 * 24 * HOUR)) if len(P) > 1 else 0.0
    return dict(n=len(P), S=S, p=p_sum_pos(v) if len(v) else float("nan"),
                clean=float(np.mean([w["cls"] == "clean_slide" for w in P])) if P else float("nan"),
                share=float(v.max() / S) if S > 0 else float("nan"), clean_sum=cl, fs_sum=fs, fs_loss=fs_loss,
                fs_ratio=(fs_loss / cl) if cl > 0 else float("inf"), days=len({_day(w) for w in P}), weeks=len({_week(w) for w in P}),
                span_w=span_w)


def _sumtxt(m):
    return (f"{m['n']} windows · {m['days']} days · {m['weeks']} ISO weeks (span {m['span_w']:.1f} wk) · V2-d Σ {m['S']:+.2f} pts · "
            f"P(Σ > 0) {_num(m['p'], '{:.2f}')} · clean slides {_num(m['clean'] * 100 if m['n'] else None, '{:.0f}')} % · largest window "
            + (f"{m['share']:.0%} of Σ" if m["S"] > 0 else "n/a (Σ ≤ 0)")
            + f" · false starts lose {m['fs_loss']:.2f} vs clean slides earn {m['clean_sum']:+.2f} (FS-loss ratio "
            + ("∞" if m["fs_ratio"] == float("inf") else f"{m['fs_ratio']:.0%}") + ")")


def _final(windows):
    return sorted([w for w in windows if w.get("d_mean") is not None], key=lambda w: w["on_ms"])


def interim_read(windows):
    """INTERIM checkpoint (non-terminal, never frozen, never a stop): printed once ≥ 8 windows on ≥ 8 UTC days exist."""
    W = _final(windows)
    if len(W) < INTERIM_N or len({_day(w) for w in W}) < INTERIM_DAYS:
        return f"interim checkpoint at {INTERIM_N} windows on {INTERIM_DAYS} days — not reached ({len(W)} windows, {len({_day(w) for w in W})} days)"
    m = summary(W)
    tone = "on track" if (m["S"] > 0 and m["p"] >= 0.80 and m["clean"] >= 0.40 and m["share"] < SHARE_MAX) else "not on track"
    return f"interim checkpoint (NON-TERMINAL, informational, never frozen): {_sumtxt(m)} → {tone} for the pre-committed ARM gate"


def _blocked_by_pending(prefix_end_ms, pending):
    early = sorted(int(x) for x in (pending or ()) if int(x) <= int(prefix_end_ms))
    return early


def arm_read(windows, pending=()):
    """🔒 the PRE-COMMITTED ARM gate (reports/BEARRUN_V2_2026-10-22.md, 'Pre-committed gate'), read ONCE at the shortest prefix of
    ≥ 20 final windows spanning ≥ 6 distinct ISO weeks. → (verdict, text, prefix_n) · COLLECTING / WAITING are not decisions."""
    W = _final(windows)
    end = None
    wk = set()
    for k, w in enumerate(W):
        wk.add(_week(w))
        if k + 1 >= ARM_N and len(wk) >= ARM_WEEKS:
            end = k
            break
    if end is None:
        return "COLLECTING", f"⏳ {len(W)}/{ARM_N} final windows over {len(wk)}/{ARM_WEEKS} ISO weeks", len(W)
    P = W[:end + 1]
    early = _blocked_by_pending(P[-1]["on_ms"], pending)
    if early:
        return "WAITING", (f"prefix reached, but {len(early)} cohort window(s) opened by its end are not final yet "
                           f"({', '.join(_fmt_ms(x) for x in early[:4])}) — decided once they are persisted"), len(P)
    m = summary(P)
    cond = [("① ≥ 20 windows on ≥ 6 weeks", True),
            ("② Σ > 0 ∧ P(Σ > 0) ≥ 0.90", m["S"] > 0 and m["p"] >= ARM_P),
            ("③ no window ≥ 50 % of the gain", m["S"] > 0 and m["share"] < SHARE_MAX),
            ("④ false starts lose ≤ 60 % of what clean slides earn", m["clean_sum"] > 0 and m["fs_ratio"] <= FS_RATIO_MAX),
            (f"⑤ replay walk-forward held-out not negative (frozen ref {WF_HELDOUT_SUM:+.0f}, {WF_SRC})", WF_HELDOUT_SUM >= 0)]
    failed = [n for n, ok in cond if not ok]
    head = f"ARM read at {_sumtxt(m)} (to {_fmt_ms(P[-1]['on_ms'])})"
    if not failed:
        return "ARM_CANDIDATE", head + (" → ALL five hold: ARM CANDIDATE at probe size (bearrun_lev_mult 0.05) with the revert 'first 10 "
                                        "armed windows Σ ≤ 0' — operator decides; nothing armed by the scout"), len(P)
    return "NO_ARM", head + " → NOT armable; failed: " + " · ".join(failed), len(P)


def drop_read(windows, pending=()):
    """🔒 the pre-committed DROP rule, read ONCE at the 30th final window: DROP_SHADOW iff Σ ≤ 0 ∧ false starts cost more than clean
    slides earn; else KEEP_SHADOW."""
    W = _final(windows)
    if len(W) < DROP_N:
        return "COLLECTING", f"⏳ {len(W)}/{DROP_N} final windows", len(W)
    P = W[:DROP_N]
    early = _blocked_by_pending(P[-1]["on_ms"], pending)
    if early:
        return "WAITING", f"{len(early)} earlier cohort window(s) not final yet — decided once they are persisted", len(P)
    m = summary(P)
    head = f"DROP read at {_sumtxt(m)}"
    if m["S"] <= 0 and m["fs_loss"] > m["clean_sum"]:
        return "DROP_SHADOW", head + " → Σ ≤ 0 and false starts cost more than clean slides earn: DROP the shadow", len(P)
    return "KEEP_SHADOW", head + " → keep the shadow running", len(P)


def decide_once(st, key, fn, windows, now_ms, pending=()):
    """a read stored under st[key] the first time it is a decision (not COLLECTING / WAITING); returned as frozen afterwards."""
    if st.get(key):
        v = st[key]
        return v["verdict"], v["text"] + f" (frozen {v['at'][:16]} UTC)", False
    v, txt, n = fn(windows, pending)
    if v not in ("COLLECTING", "WAITING"):
        st[key] = {"verdict": v, "text": txt, "prefix_n": n, "at": pd.Timestamp(int(now_ms), unit="ms").isoformat()}
        return v, txt, True
    return v, txt, False


@contextlib.contextmanager
def _locked(path):
    """exclusive flock around load + append + decide + save of the state JSON (scout + manual run never interleave)."""
    fh = None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fh = open(path + ".lock", "w")
        fcntl.flock(fh, fcntl.LOCK_EX)
    except Exception:
        fh = None
    try:
        yield
    finally:
        if fh is not None:
            try:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            except Exception:
                pass


# ═══════════════════════════════ run ═══════════════════════════════
def _fmt_ms(ms):
    return "—" if ms is None else pd.Timestamp(int(ms), unit="ms").strftime("%m-%d %H:%M")


def _num(x, f="{:+.2f}"):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


def build_record(w, D, P, cls, sub):
    on_i = int(w["on_i"])
    p0 = float(D.c.values[on_i])
    end_i = min(int(w["off_i"]) if w.get("off_i") is not None else on_i + HORIZON_H, on_i + HORIZON_H, len(D) - 1)   # capped at ON + 24 h like d
    lo = float(np.min(D.l.values[on_i + 1:end_i + 1])) if end_i > on_i else p0
    return dict(on_ms=int(w["on_ms"]), on_utc=_fmt_ms(w["on_ms"]), off_ms=w.get("off_ms"), off_utc=_fmt_ms(w.get("off_ms")),
                off_reason=w.get("off_reason") or ("still ON at +24 h" if w.get("off_ms") is None else ""),
                btc_on=round(p0, 2), btc_low=round(lo, 2), drop_pct=round((lo / p0 - 1) * 100, 3), cls=cls, cls_sub=sub,
                r6=round(w["r6"], 3), slope3=round(w["slope3"], 4), di_gap=round(w["di_gap"], 2),
                c2_pct=P.get("c2_pct"), c2=P.get("c2"), basket=P.get("basket"),
                d_n=P["d"]["n"], d_mean=P["d"]["mean"], d_wr=P["d"]["wr"], d_stops=P["d"]["stops"],
                a_n=P["a"]["n"], a_mean=P["a"]["mean"], a_wr=P["a"]["wr"], a_stops=P["a"]["stops"], d_rows=P.get("d_rows"),
                dropped=P.get("dropped") or [], dropped_4xx=P.get("dropped_4xx") or [], coins_unread=P.get("coins_unread"),
                fragments=int(w.get("fragments", 1)), first_off_utc=_fmt_ms(w.get("x_off_ms")), universe_at=P.get("universe_at"))


def _line(rec, t0, final):
    c2 = "C2 n/a" if rec.get("c2_pct") is None else f"C2 {'✓' if rec['c2'] else '✗'} {rec['c2_pct']:.0f} %"
    frag = (f" [{rec['fragments']} fragments, basket exit at the first OFF {rec.get('first_off_utc', '—')}]"
            if int(rec.get("fragments") or 1) > 1 else "")
    drop = (f" · dropped {len(rec.get('dropped') or []) + len(rec.get('dropped_4xx') or [])}"
            if (rec.get("dropped") or rec.get("dropped_4xx")) else "")
    return (f"- {'' if final else '⏳ '}ON {rec['on_utc']} → OFF {rec['off_utc']} ({rec['off_reason']}){frag} · BTC {rec['btc_on']:,.0f} → low "
            f"{rec['btc_low']:,.0f} ({rec['drop_pct']:+.2f} %) · {rec['cls']}{(' ' + rec['cls_sub']) if rec['cls_sub'] else ''} · "
            f"r6 {rec['r6']:+.2f} · {c2} · basket d {rec['d_n']} coins avg {_num(rec['d_mean'], '{:+.3f}')} % "
            f"(WR {_num(rec['d_wr'], '{:.0f}')} %, stops {rec['d_stops']}){drop} · a' avg {_num(rec['a_mean'], '{:+.3f}')} % · "
            + _t0_text(t0))


def _window_universe(w, uni_fn, now_ms):
    """the universe SNAPSHOT for a window: taken at the first run that prices it after ON, cached, reused on every later run."""
    name = f"uni_on_{int(w['on_ms'])}.json"
    hit = _cache_get(name)
    if isinstance(hit, dict) and hit.get("syms"):
        return hit["syms"], hit.get("at")
    sy = uni_fn()
    if sy is None:
        return None, None
    _cache_put(name, {"at": int(now_ms), "syms": list(sy)})
    return list(sy), int(now_ms)


def _fragment_of(w, persisted):
    """True when w starts inside / ≤ 180 min after a persisted window (a fragment whose first ON scrolled out of the BTC fetch)."""
    for p in persisted:
        end = int(p["off_ms"]) if p.get("off_ms") else int(p["on_ms"]) + D_CAP_H * HOUR
        if int(p["on_ms"]) < int(w["on_ms"]) <= end + MERGE_MIN * 60_000:
            return True
    return False


def run(now_ms=None, net=None, paths=None, state_path=STATE, deploy_ms="auto", btc_rows=None, syms=None):
    """→ report lines. Never raises on data (the opportunity-scout wrapper also catches)."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    net = net or Net()
    L = [f"## 🐻⏩ BEARRUN_V2_SHADOW — Bear-Run V2-C1 live shadow (DECISION_LOG {DL}, observe-only, pre-committed gate)", "",
         "V2 ON = BTC 1 h EMA20 slope3 < 0 ∧ −DI > +DI ∧ BTC 6 h ≤ −1 % on CLOSED 1 h bars (the study ran on the 5-min grid — ON can be "
         "later / missed on intra-hour dips) · OFF = slope ≥ 0 ∨ +DI > −DI ×2 ∨ +1.5 % off the low · fragments ≤ 180 min apart = one "
         "window · basket = top-20 alt perps by 24 h quote volume, SHORT · d = hold while ON (first OFF, +2 % stop, 24 h cap) · "
         "a' = +0.5 / −1 / 6 h · 0.10 % fees · PROXIES OF THE SLEEVE, NOT THE ENGINE."]
    dep = _GK.dl_deploy_ms(DL, lag_min=10, label="BEARRUN_V2_SHADOW") if deploy_ms == "auto" else deploy_ms
    st = load_state(state_path)
    if btc_rows is None:
        start = (now_ms // HOUR - BTC_BARS) * HOUR
        btc_rows = klines_1h("BTCUSDT", start, BTC_BARS, net, now_ms, cache=False)
    if not btc_rows or btc_rows == DROPPED or len(btc_rows) <= WARM_BARS + 6:
        L += [f"BTC 1 h klines unreadable this run ({net.why or 'no data'}) — nothing evaluated.", ""]
        L += _persisted_lines(st, dep, now_ms, None, state_path, pending=None)
        return L + [""]
    D = btc_frame(btc_rows)
    W = v2_windows(D)
    last_close = int(D.t.values[-1])
    uni = {"syms": syms, "tried": syms is not None}

    def _syms():                                                    # lazy: the universe is read only when a window needs pricing
        if not uni["tried"]:
            uni["syms"], uni["tried"] = universe(net, now_ms), True
        return uni["syms"]
    T = load_t0(paths)
    persisted = list(st["windows"])
    have = {int(w["on_ms"]) for w in persisted}
    ctx_from = last_close - CONTEXT_H * HOUR
    show, new, pending = [], [], set()
    for w in W:
        in_cohort = dep is not None and w["on_ms"] >= dep
        in_ctx = w["on_ms"] >= ctx_from or (w.get("off_ms") is None)
        if not (in_cohort or in_ctx):
            continue
        if in_cohort and (int(w["on_ms"]) in have or _fragment_of(w, persisted)):
            continue                                                # already final + persisted (or its fragment) — printed from the state
        cls, sub, final = classify(D, int(w["on_i"]))
        sy, at = _window_universe(w, _syms, now_ms)
        if sy is None:
            P = dict(d=_stats([]), a=_stats([]), basket=None, c2_pct=None, c2=None, priced_full=False, d_rows=None, dropped=[])
        else:
            P = price_window(w, D, net, now_ms, sy, final)
            P["universe_at"] = None if at is None else pd.Timestamp(int(at), unit="ms").isoformat()
        rec = build_record(w, D, P, cls, sub)
        ok_final = bool(final and P.get("priced_full"))
        if in_cohort:
            if ok_final:
                rec["final_at"] = pd.Timestamp(now_ms, unit="ms").isoformat()
                new.append(rec)
            else:
                pending.add(int(rec["on_ms"]))
        show.append((rec, t0_for(rec, T), ok_final, in_cohort))
    if dep is None:
        L.append(f"⚠ NOT evaluable — the DECISION_LOG {DL} commit is not in git yet; nothing is counted or persisted. "
                 f"Context (informational): V2 windows of the last {CONTEXT_H} h.")
    else:
        L.append(f"Cohort: V2 windows with ON ≥ {_fmt_ms(dep)} UTC (DL {DL} commit + 10 min); final 24 h after ON, persisted append-only.")
    if dep is not None:
        with _locked(state_path):                                   # re-load under the lock: another run may have appended meanwhile
            st = load_state(state_path)
            n = append_windows(st, new) if new else 0
            body = _persisted_lines(st, dep, now_ms, T, None, pending=pending)
            dirty = bool(st.pop("_dirty", False))
            ok = save_state(st, state_path) if (n or dirty) else True
        if n:
            L.append(f"Persisted {n} newly final window(s)." if ok else "⚠ state file not writable — nothing persisted.")
    else:
        body = []
    live = [x for x in show if x[3]]
    ctx = [x for x in show if not x[3]]
    if live:
        L.append("Open / not yet final cohort windows:")
        L += ["  " + _line(r, t, f) for r, t, f, _ in live]
    if ctx:
        L.append(f"Context (last {CONTEXT_H} h, before the cohort — informational, never counted):")
        L += ["  " + _line(r, t, f) for r, t, f, _ in ctx]
    if not show:
        L.append(f"No V2 window in the last {CONTEXT_H} h.")
    L += body
    L.append(f"network: {net.requests} requests · used weight {net.used}" + (f" · stopped: {net.why}" if net.stop else "")
             + (" · universe unreadable — baskets not priced" if uni["tried"] and uni["syms"] is None else ""))
    L.append("Backtest ref (reports/BEARRUN_V2_2026-10-22.md, 5-MIN state machine, proxy exit d): V2-C1 +0.02 %/trade (Σ +55, CI −310 … +455); "
             "clean slides +415 %, false starts −331 %; 36 % of windows false starts; forced replay: V2 loses vs T0 under every exit; "
             f"walk-forward held-out {WF_HELDOUT_SUM:+.0f} (gate ⑤ fails until a new replay says otherwise).")
    return L + [""]


def _persisted_lines(st, dep, now_ms, T, state_path=None, pending=None):
    """final windows + tally + interim checkpoint + the two pre-committed reads (decided once). pending None = this run could not see the
    cohort (BTC unreadable) → no decision is taken."""
    L = []
    if dep is None:
        return L
    W = sorted(st["windows"], key=lambda w: w["on_ms"])
    if W:
        L.append(f"Final cohort windows ({len(W)}):")
        for w in W:
            L.append("  " + _line(w, t0_for(w, T) if T is not None else [], True))
        F = _final(W)
        a = np.array([w["a_mean"] for w in W if w.get("a_mean") is not None], float)
        cls = pd.Series([w["cls"] for w in W]).value_counts().to_dict()
        L.append("Tally: " + (_sumtxt(summary(F)) if F else "0 priced windows") + f" · a' Σ {a.sum() if len(a) else 0:+.2f} · classes "
                 + ", ".join(f"{k} {n}" for k, n in sorted(cls.items())))
    L.append("🟡 " + interim_read(W))
    for key, fn, lab in (("arm_read", arm_read, "🔒 ARM gate (pre-committed, terminal)"),
                         ("drop_read", drop_read, "🔒 DROP rule (pre-committed, at 30 windows)")):
        if pending is None and not st.get(key):
            L.append(f"{lab}: not read this run (BTC unreadable)")
            continue
        v, txt, now_ = decide_once(st, key, fn, W, now_ms, pending or ())
        if now_:
            st["_dirty"] = True
        L.append(f"{lab}: {v} — {txt}")
    if st.get("_dirty") and state_path:
        st.pop("_dirty", None)
        save_state(st, state_path)
    return L


def main():
    print("\n".join(run(int(time.time() * 1000))))


if __name__ == "__main__":
    main()
