#!/usr/bin/env python3
"""🩳 Scout — FLATBASE_SHORT gate tracker (DECISION_LOG 348, 2026-10-22, operator DISCIPLINE OVERRIDE — ALERT-only, never changes config).

RULE (engine)   services/flatbase.py: a FRENZY episode on a FLAT 14-day base (range ≤ 44.4 %) is shorted at the study's S4 bar (first 4-h
                close below the spike VWAP ≥ 4 h after the spike, 24 h volume ≥ $20M, ≤ 96 h) when turnover (24 h volume / mcap) > 1 — else
                recorded OBSERVE-ONLY (flatbase_episodes ledger, status SHADOW, no order); unknown mcap → traded. Exit −8 % / +15 % / 72 h.
COHORT (real)   CLOSED FLATBASE_SHORT fills from the orders exports (paper + live via export_glob; dedup (opened_at, pair, direction), CLOSED
                wins), opened ≥ the DL 348 deploy (git commit + 10 min; not in git → NOT EVALUABLE, nothing counted); operator-closed
                (MANUAL / MANUAL_* / EXTERNAL_CLOSE) excluded — scout_gatekit.
COHORT (shadow) e=FLATBASE rows with fb_status SHADOW from the decisions exports (~/Downloads + reports/BASELINE*_decisions.csv), dedup
                (pair, fb_spike_at), S4 bar ≥ the deploy; each PRICED from public 5m klines AFTER the S4 bar with the SAME exit (the
                sim's order: stop first on a bar that touches both, gap-aware; TP at the limit; else the close at 72 h) and the bot's fee
                model (entry taker, TP maker, stop / time taker) — services.flatbase.price_path_exit. Unfinished windows wait.
NET OF FUNDING  real fills: pnl % + funding_fee_usd / (entry price × qty) × 100 (services.flatbase.net_of_funding_pct; paper rows carry
                no funding → shown as unknown, pnl % only). Shadows: the real fills' median entry slippage (≥ 5 stamped, else 0.03 %) on the
                entry and on a taker exit, plus the funding settlements over the hold (public /fapi/v1/fundingRate; unreadable → counted
                unknown).
GATES           services.flatbase.flatbase_gate (the SAME function as the dashboard row): 3 stop-outs in a row or Σ ≤ −30 pts → review ·
                first 10 avg < 0 → review · main read at 30 fills on ≥ 10 UTC days: avg ≥ +0.5 % → step the size up · < 0 → switch off ·
                else re-read at 60. Observe cohort: after 30 real fills, shadow avg > real avg (same period) → review adding them
                (services.flatbase.flatbase_observe_gate).
NETWORK         public fapi klines only (no key), ≤ 700 used weight / run, paced 0.35 s, the shared Binance cooldown honoured, stop on
                418 / 429 / 403 (scout_long_stop4.fapi_klines); each priced setup cached under reports/backtest_cache/scout_flatbase_short.

OBSERVE LINES (DL 348 addendum, 2026-10-22 — observe-only, alert-only, never change config; cohort from the DL 348 deploy + 10 min; each
its own section, each crash-isolated; one shared Budget for the whole run; final results cached per setup under CACHE/setups):
  ① NON-FLAT  ledger rows NOT_FLAT (the engine's own 14-day judgement). The S4 bar is recomputed from public 5m klines with the ENGINE's
              function (services.flatbase.s4_scan — spike-anchored VWAP, first 4-h close below it ≥ 4 h after the spike, 24 h vol ≥ $20M,
              ≤ 96 h, not before the ledger's first-seen bar) on a window 20 h before the spike bar → 96 h after it (read once the window
              has closed), then priced exactly like the shadows (price_shadow: −8 / +15 / 72 h, fee model, slippage, funding). Turnover =
              ledger fb_turnover when stamped, else INFERRED: circulating supply from the nearest stamped mcap of the same pair (ledger
              fb_mcap_usd / price or an order's entry_mcap_usd / entry_price, ≤ 30 days) × the S4 close; none → unscored (shown apart).
              Compared with the FLAT real cohort and the FLAT shadows over the same period. Purpose: "flat coins fall harder" on fresh data.
  ② MISSED    ledger rows MISSED (the first qualifying 4-h close passed unjudged), REFUSED (by reason: FLATBASE_MAX_OPEN, pair held,
              late, other) and UNREADABLE — the S4 bar from the row when stamped, else recomputed like ①; priced like the shadows; per reason.
  ③ BLOCKED   FRENZY longs refused PAIR_HELD while a FLATBASE_SHORT held the pair: decisions BLOCK rows (dir LONG) with the sleeve-named
              gates FRENZY_LITE_PAIR_HELD / FRENZY_LITE_RETRY_PAIR_HELD / FRENZY_WILLY_PAIR_HELD / FRENZY_WILLY_HT_PAIR_HELD (and
              FRENZY_PAIR_HELD / FRENZY_WIDE_PAIR_HELD should the engine ever name them); the holder is INFERRED from the orders exports (a
              FLATBASE_SHORT on that pair open over the 5-min bucket). Priced from the first 1m open AFTER the bucket floor (bucket + 1 min — no
              look-ahead inside the bucket) with the sleeve's CURRENT exit:
              WILLY / WILLY_HT → scout_frenzy_exits' DL 336 walker (tb_walker_rule / tb_walker_eval: +1.25 / −4 / 90 min, late TP, resting
              TP); FRENZY_LONG / WIDE → scout_frenzy_exits.walk_live with the engine's per-sleeve stop (services.frenzy.frenzy_stop_for);
              LITE / LITE_RETRY → faithful only with LITE C off (its TP needs the unstamped entry vol-mult / BTC 1h slope) — else UNPRICED,
              shown with the [+C TP, +LITE TP] bracket. −0.10 % slippage on every long (the FRENZY scouts' live ruler). One position per
              (pair, sleeve, holder): a later block inside a counted trade's life (its cap when unpriced) is not a second trade; the $ net
              counts ONE blocked position per (pair, holder) — precedence WILLY > WILLY_HT > LITE_RETRY > LITE > LONG > WIDE — on CLOSED holders. The generic PAIR_HELD counter (FRENZY_LONG /
              WIDE and momentum share it, no sleeve) is listed as UNATTRIBUTED, never priced. Compared with the holding short's own result.
Usage           venv/bin/python scripts/scout_flatbase_short.py [--no-fetch]
"""
import glob
import json
import math
import os
import re
import sys
import time

import pandas as pd
from types import SimpleNamespace

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import scout_gatekit as _GK                                  # noqa: E402
import scout_willy_hold_off as _HO                           # noqa: E402  (the shared orders-export loader: dedup, CLOSED wins, paper + live)
from services.flatbase import (FLATBASE_SHORT, flatbase_gate, flatbase_observe_gate, price_path_exit, flatbase_params,   # noqa: E402
                               BAR_MS, BOUNDARY_MS, net_of_funding_pct, s4_scan, ST_NOT_FLAT, ST_MISSED, ST_REFUSED, ST_UNREADABLE,
                               ACT_ENTER, ACT_MISSED, ACT_EXPIRED, ACT_WAIT, STOP_REASONS)

DL = 348
DEC_PATTERNS = (os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"), os.path.join(_R, "reports", "BASELINE*_decisions.csv"))
CACHE = os.path.join(_R, "reports", "backtest_cache", "scout_flatbase_short")
SLIP_DEF = 0.03                                              # % per taker side when the real fills carry too few slippage stamps
ORDER_COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "is_paper",
              "funding_fee_usd", "entry_price", "quantity", "entry_slippage_pct", "entry_mcap_usd", "cell_multiplier",
              "entry_frenzy_run_pct", "entry_flatbase_off_peak_pct",
              "entry_frenzy_spike_at", "entry_frenzy_hours", "entry_flatbase_peak_age_h")   # ⏱ (DECISION_LOG 401) hours since the peak
FUND_URL = "https://fapi.binance.com/fapi/v1/fundingRate?symbol={s}&startTime={a}&endTime={b}&limit=1000"
FB_COLS = ("e", "pair", "price", "fb_spike_at", "fb_status", "fb_entry_bar_at", "fb_turnover", "fb_mcap_usd", "fb_vol24_usd", "fb_flat14_pct")
LEDGER_COLS = FB_COLS + ("t", "fb_reason", "fb_first_seen_at")
BLOCK_COLS = ("t", "e", "pair", "dir", "gate", "n", "reason")
S4_PRE_BARS = 240                                            # 20 h of 5m bars before the spike bar: the first S4 candidate (spike + 48) has 288 bars of volume
SUPPLY_MAX_DAYS = 30                                         # an inferred circulating supply is read from a stamped mcap ≤ 30 days away
LONG_SLIP = 0.10                                             # % on a blocked FRENZY long (the FRENZY scouts' live ruler)
# ③ the sleeve-named PAIR_HELD counters (services/trading_engine._frenzy_open records f"{_bk}_PAIR_HELD" for LITE / LITE_RETRY / WILLY /
#   WILLY_HT only; FRENZY_LONG / WIDE reach open_position and land on the generic PAIR_HELD today — FRENZY_PAIR_HELD / FRENZY_WIDE_PAIR_HELD
#   are mapped only so the scout prices them if the engine ever names them)
HELD_GATES = {"FRENZY_PAIR_HELD": "FRENZY_LONG", "FRENZY_WIDE_PAIR_HELD": "FRENZY_WIDE", "FRENZY_LITE_PAIR_HELD": "FRENZY_LITE",
              "FRENZY_LITE_RETRY_PAIR_HELD": "FRENZY_LITE_RETRY", "FRENZY_WILLY_PAIR_HELD": "FRENZY_WILLY",
              "FRENZY_WILLY_HT_PAIR_HELD": "FRENZY_WILLY_HT"}
GENERIC_HELD = "PAIR_HELD"                                   # open_position's counter: FRENZY_LONG / WIDE / momentum — no sleeve on it
MIN_MS = 60_000
LONG_PRECEDENCE = ("FRENZY_WILLY", "FRENZY_WILLY_HT", "FRENZY_LITE_RETRY", "FRENZY_LITE", "FRENZY_LONG", "FRENZY_WIDE")   # ③ $ net: one
#   blocked position per (pair, holder) — the engine opens one position per pair, and on a FRENZY episode WILLY A arms first (the flag bar)
FINAL_AFTER_MS = 86_400_000                                  # "no klines" on a window closed ≥ 1 day ago = final (delisted / invalid symbol)


def _frozen_th():
    """the sleeve's exit as shipped (DL 348) — the scout prices shadows with the RULE, never with a later settings change."""
    from types import SimpleNamespace
    return SimpleNamespace(flatbase_tp_pct=8.0, flatbase_stop_pct=15.0, flatbase_max_hold_hours=72.0)


def load_shadows(paths=None):
    """every SHADOW ledger row from the decisions exports, dedup (pair, fb_spike_at) — the newest file wins."""
    if paths is None:
        paths = sorted({p for pat in DEC_PATTERNS for p in glob.glob(pat)}, key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in FB_COLS)
        except Exception:
            continue
        if "e" not in x or "fb_status" not in x:
            continue
        fr.append(x[(x.e.astype(str) == "FLATBASE")])
    if not fr:
        return pd.DataFrame(columns=list(FB_COLS))
    a = pd.concat(fr, ignore_index=True)
    for c in FB_COLS:
        if c not in a:
            a[c] = None
    a = a[a.fb_status.astype(str) == "SHADOW"].drop_duplicates(["pair", "fb_spike_at"], keep="last")
    a["t"] = pd.to_datetime(a.fb_entry_bar_at.astype(str).str[:19], errors="coerce")
    a["px"] = pd.to_numeric(a.price, errors="coerce")
    return a[a.t.notna() & (a.px > 0)].sort_values("t").reset_index(drop=True)


def load_orders(paths=None):
    """the orders exports (paper + live), dedup (opened_at, pair, direction) — a CLOSED row wins, then the newest export; + the funding /
    size / slippage columns the net-of-funding gate reads."""
    paths = paths if paths is not None else sorted(_HO.EG.glob_exports(_HO.PATTERN), key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in ORDER_COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(ORDER_COLS) + ["o", "c", "pct", "usd", "es", "net", "fund_known"])
    a = pd.concat(fr, ignore_index=True)
    for c in ORDER_COLS:
        if c not in a:
            a[c] = None
    a = a.assign(_c=(a.status.astype(str) == "CLOSED").astype(int)).reset_index(drop=True)
    a = a.assign(_i=range(len(a))).sort_values(["_c", "_i"]).drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a["o"] = _HO._ts(a.opened_at).values
    a["c"] = _HO._ts(a.closed_at).values                    # ③ the holder's life (NaT = still open in that export)
    a["pct"] = pd.to_numeric(a.pnl_percentage, errors="coerce")
    a["usd"] = pd.to_numeric(a.pnl, errors="coerce")
    a["es"] = a.entry_strategy.astype(str)
    nf = [net_of_funding_pct(p, f, e, q) for p, f, e, q in zip(a.pct, a.funding_fee_usd, a.entry_price, a.quantity)]
    a["net"] = [x[0] for x in nf]
    a["fund_known"] = [x[1] for x in nf]
    return a[a.o.notna()].drop(columns=["_c", "_i"]).sort_values("o").reset_index(drop=True)


def funding_sum_pct(pair, t0_ms, t1_ms, bud, now_ms, fetch=True):
    """% of notional a SHORT receives from the funding settlements in (t0, t1] (positive rate = longs pay shorts) — public fundingRate,
    paced / budgeted / cooldown-aware like the klines; None = unreadable."""
    import json, urllib.request, urllib.error
    import scout_long_stop4 as _LS
    p = os.path.join(CACHE, "funding", f"{pair}_{int(t0_ms)}_{int(t1_ms)}.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return float(json.load(f))
        except Exception:
            pass
    if bud.stop or not fetch or _LS._cooling(now_ms) or bud.used >= _LS.WEIGHT_STOP:
        return None
    time.sleep(_LS.PACE_S)
    try:
        with urllib.request.urlopen(FUND_URL.format(s=pair, a=int(t0_ms) + 1, b=int(t1_ms)), timeout=15) as r:
            rows = json.loads(r.read().decode())
            bud.used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or bud.used)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429, 403):
            bud.halt(f"HTTP {e.code} on fundingRate")
        return None
    except Exception:
        return None
    v = sum(float(x.get("fundingRate") or 0) for x in rows if int(t0_ms) < int(x.get("fundingTime") or 0) <= int(t1_ms)) * 100.0
    if int(t1_ms) <= now_ms:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(v, f)
    return v


def price_shadow(row, bud, now_ms, fetch=True, th=None, klines=None, slip_pct=SLIP_DEF, funding=None):
    """→ (net % incl. funding, funding known) of a shadow setup under the sleeve's exit, or None (window unfinished / no data / budget stop).
    klines = an injectable fetcher (sym, start_ms, limit) → rows · funding = (sym, t0, t1) → % or None (tests); defaults = the public
    fapi fetchers (cached, paced, cooldown-aware). The same taker slippage as the real fills is charged on the entry and a taker exit."""
    th = th or _frozen_th()
    hb = int(round(flatbase_params(th)['max_hold_h'] * 12))
    start = int(pd.Timestamp(row.t).value // 1_000_000) + BAR_MS          # the bar AFTER the S4 bar (the sim's path)
    if klines is None:
        import scout_long_stop4 as _LS
        rows = _LS.fapi_klines(str(row.pair), "5m", start, hb, bud, CACHE, now=now_ms, fetch=fetch)
    else:
        rows = klines(str(row.pair), start, hb)
    if not rows:
        return None
    r = price_path_exit(float(row.px), rows, th, slip_pct=slip_pct)
    if r is None or r["reason"] == "OPEN":
        return None                                                     # the 72 h window is not over and nothing hit yet
    t1 = int(r["exit_open_ms"]) + BAR_MS
    f = funding(str(row.pair), start, t1) if funding is not None else funding_sum_pct(str(row.pair), start, t1, bud, now_ms, fetch=fetch)
    return (r["net_pct"] + (f or 0.0), f is not None)


# ═══════════════════════════ DL 348 addendum — the three observe lines (observe-only, alert-only) ═══════════════════════════

def _ms(v):
    """an export timestamp → epoch ms (naive UTC), or None."""
    try:
        if v is None or (isinstance(v, float) and v != v):
            return None
        t = pd.Timestamp(str(v)[:19].replace(" ", "T"))
        return None if pd.isna(t) else int(t.value // 1_000_000)
    except Exception:
        return None


def _fnum(v):
    try:
        x = float(v)
        return x if x == x and abs(x) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _dec_paths():
    return sorted({p for pat in DEC_PATTERNS for p in glob.glob(pat)}, key=os.path.getmtime)


def read_decisions(paths=None, cols=None):
    """the decisions exports (oldest → newest) as one frame with only the columns the observe lines read."""
    cols = set(cols or (LEDGER_COLS + BLOCK_COLS))
    fr = []
    for f in (paths if paths is not None else _dec_paths()):
        try:
            x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols)
        except Exception:
            continue
        if "e" in x:
            fr.append(x)
    a = pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=sorted(cols))
    for c in cols:
        if c not in a:
            a[c] = None
    return a


def load_ledger(frame):
    """every e=FLATBASE ledger row, dedup (pair, fb_spike_at): a decided row beats TRACKING, then the newest export. + spike_ms / fs_ms
    (first-seen bar open) / s4_ms (the S4 bar open when stamped) / px / vol24 / turnover / flat14."""
    a = frame[frame.e.astype(str) == "FLATBASE"].copy() if len(frame) else frame.iloc[0:0].copy()
    if not len(a):
        return pd.DataFrame(columns=list(LEDGER_COLS) + ["spike_ms", "fs_ms", "s4_ms", "px", "vol24", "turnover", "flat14", "mcap"])
    a["_f"] = (a.fb_status.astype(str) != "TRACKING").astype(int)
    a = a.assign(_i=range(len(a))).sort_values(["_f", "_i"]).drop_duplicates(["pair", "fb_spike_at"], keep="last")
    a["spike_ms"] = [_ms(x) for x in a.fb_spike_at]
    a["fs_ms"] = [_ms(x) for x in a.fb_first_seen_at]
    a["s4_ms"] = [_ms(x) for x in a.fb_entry_bar_at]
    a["px"] = [_fnum(x) for x in a.price]
    a["vol24"] = [_fnum(x) for x in a.fb_vol24_usd]
    a["turnover"] = [_fnum(x) for x in a.fb_turnover]
    a["flat14"] = [_fnum(x) for x in a.fb_flat14_pct]
    a["mcap"] = [_fnum(x) for x in a.fb_mcap_usd]
    a = a[a.spike_ms.notna()].drop(columns=["_f", "_i"])
    return a.sort_values("spike_ms").reset_index(drop=True)


def _floor(led):
    """the cohort clock of a ledger row: its first-seen bar (the engine tracks an episode from the first pass after the deploy that sees it
    flagged — its spike may be up to 96 h older), else its spike."""
    return led.fs_ms.where(led.fs_ms.notna(), led.spike_ms).astype(float)


def _cache_path(key):
    return os.path.join(CACHE, "setups", re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")


def _cache_get(key):
    p = _cache_path(key)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _cache_put(key, d):
    p = _cache_path(key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(d, f)


def s4_from_klines(pair, spike_ms, first_seen_ms, bud, now_ms, fetch=True, klines=None):
    """the engine's S4 (services.flatbase.s4_scan, the frozen DL 348 rule) on public 5m klines from 20 h before the spike bar to the end of
    its 96 h window — read only once that window has closed. → dict(state S4 / NO_S4 / WAIT / UNPRICED, s4_ms, px, vol24, why)."""
    th = _frozen_th()
    p = flatbase_params(th)
    start = int(spike_ms) - BAR_MS - S4_PRE_BARS * BAR_MS
    limit = S4_PRE_BARS + int(round(p['max_h'] * 12)) + 1     # the spike bar at row 240, its last S4 candidate at the last row
    if now_ms < start + limit * BAR_MS:
        return dict(state="WAIT", why="S4 window still open")
    if klines is not None:
        rows = klines(str(pair), start, limit)
    else:
        import scout_long_stop4 as _LS
        rows = _LS.fapi_klines(str(pair), "5m", start, limit, bud, CACHE, now=now_ms, fetch=fetch)
    if rows is None:
        return dict(state="UNPRICED", why="fetch stopped / --no-fetch", final=False)
    if not rows:
        return dict(state="UNPRICED", why="no klines", final=(now_ms >= start + limit * BAR_MS + FINAL_AFTER_MS))
    sc = s4_scan(rows, int(spike_ms), th, first_seen_ms)
    st = sc.get("status")
    if st in (ACT_ENTER, ACT_MISSED):
        return dict(state="S4", s4_ms=int(sc["bar_open"]), px=float(sc["close"]), vol24=sc.get("vol24"))
    if st in (ACT_EXPIRED, ACT_WAIT):        # WAIT on a closed window = the pair's data ends early (delisted / halted) — no S4 in the data
        return dict(state="NO_S4", why=("no 4-h close below the spike VWAP within 96 h" if st == ACT_EXPIRED else "data ends inside the window"))
    return dict(state="UNPRICED", why="spike bar not in the klines", final=True)


def price_setup(line, r, bud, now_ms, fetch=True, klines=None, funding=None, slip=SLIP_DEF):
    """one ledger episode (row with pair / spike_ms / fs_ms / s4_ms / px / vol24) priced like a shadow: the S4 bar from the row when
    stamped, else recomputed (s4_from_klines); then price_shadow (−8 / +15 / 72 h, fees, slippage, funding). Final results (priced with
    funding known, no S4, unreadable spike) are cached per setup. → dict(state PRICED / WAIT / NO_S4 / UNPRICED, s4_ms, px, vol24, net,
    fund_known, why)."""
    key = f"{line}_{r.pair}_{int(r.spike_ms)}_s{float(slip):.3f}"      # the slippage is part of the price → part of the key
    c = _cache_get(key)
    if c is not None:
        return c
    s4, px, vol = _fnum(getattr(r, "s4_ms", None)), _fnum(getattr(r, "px", None)), _fnum(getattr(r, "vol24", None))
    if s4 is None or not (px and px > 0):
        fs = _fnum(getattr(r, "fs_ms", None))
        s = s4_from_klines(r.pair, int(r.spike_ms), (int(fs) if fs is not None else None), bud, now_ms, fetch=fetch, klines=klines)
        if s["state"] != "S4":
            if s["state"] == "NO_S4" or s.get("final"):
                _cache_put(key, s)
            return s
        s4, px, vol = s["s4_ms"], s["px"], s.get("vol24")
    out = dict(s4_ms=int(s4), px=float(px), vol24=vol)
    row = SimpleNamespace(pair=str(r.pair), t=pd.Timestamp(int(s4), unit="ms"), px=float(px))
    pr = price_shadow(row, bud, now_ms, fetch=fetch, klines=klines, slip_pct=slip, funding=funding)
    if pr is None:
        hold_end = int(s4) + BAR_MS + int(round(flatbase_params(_frozen_th())['max_hold_h'] * 12)) * BAR_MS
        stopped = (bud is not None and bud.stop) or not fetch
        res = dict(out, state=("WAIT" if now_ms < hold_end else "UNPRICED"),
                   why=("72 h hold unfinished" if now_ms < hold_end else "no klines after the S4 bar / fetch stopped"))
        if not stopped and now_ms >= hold_end + FINAL_AFTER_MS:
            _cache_put(key, res)                                       # delisted / invalid: stop spending weight on it every run
        return res
    out.update(state="PRICED", net=float(pr[0]), fund_known=bool(pr[1]))
    if out["fund_known"]:
        _cache_put(key, out)
    return out


def supply_points(ledger, orders):
    """pair → [(ms, circulating supply)] from every stamped market cap: ledger fb_mcap_usd / price and orders entry_mcap_usd / entry_price."""
    pts = {}
    if len(ledger):
        for r in ledger.itertuples():
            m, p_ = _fnum(r.mcap), _fnum(r.px)
            t = _fnum(r.s4_ms) or _fnum(r.spike_ms)
            if m and m > 0 and p_ and p_ > 0 and t:
                pts.setdefault(str(r.pair), []).append((int(t), m / p_))
    if orders is not None and len(orders) and "entry_mcap_usd" in orders:
        for pr_, t, m, p_ in zip(orders.pair, orders.o, orders.entry_mcap_usd, orders.entry_price):
            m, p_ = _fnum(m), _fnum(p_)
            if m and m > 0 and p_ and p_ > 0 and pd.notna(t):
                pts.setdefault(str(pr_), []).append((int(pd.Timestamp(t).value // 1_000_000), m / p_))
    return pts


def turnover_of(stamped, pair, t_ms, px, vol24, pts):
    """(turnover, how): the ledger's own stamp when present, else 24 h volume / (the nearest stamped supply ≤ 30 days × the S4 close);
    (None, None) = unscored (no mcap source) — shown apart, never folded into either side."""
    s_ = _fnum(stamped)
    if s_ is not None:
        return s_, "stamped"
    v, p_ = _fnum(vol24), _fnum(px)
    if not (v and p_ and p_ > 0 and t_ms):
        return None, None
    c = [(abs(int(t) - int(t_ms)), s) for t, s in pts.get(str(pair), []) if abs(int(t) - int(t_ms)) <= SUPPLY_MAX_DAYS * 86_400_000]
    if not c:
        return None, None
    return v / (min(c)[1] * p_), "inferred"


def _st(x):
    """'N 3 · WR 67 % · avg +1.234 % · Σ +3.70 pts' (N 0 when empty)."""
    x = [float(v) for v in x if _fnum(v) is not None]
    if not x:
        return "N 0"
    return f"N {len(x)} · WR {sum(v > 0 for v in x) / len(x):.0%} · avg {sum(x) / len(x):+.3f} % · Σ {sum(x):+.2f} pts"


def _tally(res):
    """coverage words for a list of price_setup results."""
    k = lambda s: sum(1 for r in res if r.get("state") == s)
    return f"priced {k('PRICED')} · waiting {k('WAIT')} (S4 window / 72 h hold unfinished) · no S4 bar {k('NO_S4')} · unpriced {k('UNPRICED')}"


def unledgered_flags(frame, ledger, since_ms):
    """FRENZY flags the ledger never saw: WILLY 'ARMED_A' events (A arms on the bar a pair becomes FRENZY-flagged) ≥ the deploy with no
    ledger episode of that pair whose spike lies within 3 h before → 10 min after the arm. A coverage count only (FLAT unknown, not scored)."""
    if not len(frame) or "gate" not in frame:
        return 0
    w = frame[(frame.e.astype(str) == "WILLY") & frame.gate.astype(str).str.startswith("FRENZY_WILLY_ARMED_A")]
    n = 0
    for r in w.itertuples():
        t = _ms(r.t)
        if t is None or t < since_ms:
            continue
        sp = ledger[ledger.pair.astype(str) == str(r.pair)].spike_ms if len(ledger) else []
        if not any(t - 3 * 3_600_000 <= int(s) <= t + 600_000 for s in sp):
            n += 1
    return n


def nonflat_lines(ctx):
    """① NON-FLAT pumps priced the same way vs the FLAT cohorts (same period)."""
    L = [f"## 🩳① FLATBASE observe — NON-FLAT pumps priced like the sleeve (DECISION_LOG {DL} addendum, observe-only, no gate)", ""]
    led, since = ctx["ledger"], ctx["since_ms"]
    nf = led[(led.fb_status.astype(str) == ST_NOT_FLAT) & (_floor(led) >= since)] if len(led) else led
    res, items = [], []
    for r in nf.itertuples():
        x = price_setup("nonflat", r, ctx["bud"], ctx["now_ms"], fetch=ctx["fetch"], klines=ctx["klines"], funding=ctx["funding"], slip=ctx["slip"])
        res.append(x)
        if x.get("state") == "PRICED":
            tv, how = turnover_of(r.turnover, r.pair, x["s4_ms"], x["px"], x.get("vol24"), ctx["pts"])
            items.append(dict(t=pd.Timestamp(x["s4_ms"], unit="ms"), net=x["net"], tv=tv, how=how, fk=x.get("fund_known")))
    hi = [i["net"] for i in items if i["tv"] is not None and i["tv"] > 1.0]
    lo = [i["net"] for i in items if i["tv"] is not None and i["tv"] <= 1.0]
    un = [i["net"] for i in items if i["tv"] is None]
    inf = sum(1 for i in items if i["how"] == "inferred")
    L.append(f"Cohort: {len(nf)} NOT_FLAT episodes (ledger, first seen ≥ deploy) · {_tally(res)} · FRENZY flags with no ledger episode: "
             f"≥ {unledgered_flags(ctx['frame'], led, since)} (a LOWER BOUND: only WILLY ARMED_A events are visible; FLAT unknown — unscored ≠ rest)")
    L.append(f"- non-flat, all priced: {_st([i['net'] for i in items])}"
             + (f" · funding unknown on {sum(1 for i in items if not i['fk'])}" if any(not i["fk"] for i in items) else ""))
    L.append(f"- by turnover (24 h vol / mcap at the S4 bar; {inf} inferred from a stamped supply): > 1 {_st(hi)} · ≤ 1 {_st(lo)} · "
             f"unscored (no mcap source) {_st(un)}")
    real, sh = ctx["real"], ctx["shadows"]
    if len(real):
        t0, t1 = real.o.min(), real.o.max()
        win = lambda t: t0 <= t <= t1
        lab = f"same period ({t0:%m-%d %H:%M} → {t1:%m-%d %H:%M}, the real fills' span)"
    else:
        win = lambda t: True
        lab = "whole period (no real fill yet)"
    s_hi = [i["net"] for i in items if win(i["t"]) and i["tv"] is not None and i["tv"] > 1.0]
    s_lo = [i["net"] for i in items if win(i["t"]) and i["tv"] is not None and i["tv"] <= 1.0]
    r_net = list(real.net) if len(real) else []
    s_sh = [p for t, p in sh if win(t)]
    L.append(f"- {lab}: turnover > 1 — FLAT real {_st(r_net)} vs non-flat {_st(s_hi)} · turnover ≤ 1 — FLAT shadows {_st(s_sh)} vs non-flat {_st(s_lo)}")
    if r_net and s_hi:
        L.append(f"- reading: FLAT − non-flat (turnover > 1) avg Δ {sum(r_net) / len(r_net) - sum(s_hi) / len(s_hi):+.3f} pts "
                 f"(positive = flat coins fell harder; observe-only — small N is noise)")
    if ctx["bud"].stop:
        L.append(f"- ⚠ fetch stopped: {ctx['bud'].why} — the rest waits for the next run")
    return L + [""]


def _reason_class(status, reason):
    s = str(status or "")
    if s == ST_MISSED:
        return "MISSED"
    if s == ST_UNREADABLE:
        return "FLAT_UNREADABLE"
    t = str(reason or "").lower()
    if "flatbase_max_open" in t:
        return "FLATBASE_MAX_OPEN"
    if "pair held" in t:
        return "REFUSED_PAIR_HELD"
    if "after the 4-h bar closed" in t:
        return "REFUSED_LATE"
    return "REFUSED_OTHER"


MISSED_ORDER = ("MISSED", "FLATBASE_MAX_OPEN", "FLAT_UNREADABLE", "REFUSED_PAIR_HELD", "REFUSED_LATE", "REFUSED_OTHER")


def missed_lines(ctx):
    """② MISSED setups (+ FLATBASE_MAX_OPEN refusals, FLAT_UNREADABLE and the other refusals) priced like the shadows, per reason."""
    L = [f"## 🩳② FLATBASE observe — MISSED / refused setups priced like the shadows (DECISION_LOG {DL} addendum, observe-only, no gate)", ""]
    led, since = ctx["ledger"], ctx["since_ms"]
    c = led[led.fb_status.astype(str).isin([ST_MISSED, ST_REFUSED, ST_UNREADABLE]) & (_floor(led) >= since)] if len(led) else led
    by = {}
    for r in c.itertuples():
        k = _reason_class(r.fb_status, r.fb_reason)
        x = price_setup("missed", r, ctx["bud"], ctx["now_ms"], fetch=ctx["fetch"], klines=ctx["klines"], funding=ctx["funding"], slip=ctx["slip"])
        by.setdefault(k, []).append((r, x))
    L.append(f"Cohort: {len(c)} ledger episodes first seen ≥ deploy (MISSED · REFUSED · UNREADABLE) — "
             + (" · ".join(f"{k} {len(by[k])}" for k in MISSED_ORDER if k in by) or "none yet"))
    for k in MISSED_ORDER:
        if k not in by:
            continue
        res = [x for _, x in by[k]]
        net = [x["net"] for x in res if x.get("state") == "PRICED"]
        extra = ""
        if k == "MISSED":
            fl = [_fnum(r.flat14) for r, _ in by[k]]
            cap = flatbase_params(_frozen_th())['flat_max'] * 100
            extra = f" · 14-day base: FLAT {sum(1 for v in fl if v is not None and v <= cap + 1e-9)} · not yet read {sum(1 for v in fl if v is None)}"
        L.append(f"- {k}: {len(res)} · {_st(net)} · {_tally(res)}{extra}")
    if ctx["bud"].stop:
        L.append(f"- ⚠ fetch stopped: {ctx['bud'].why} — the rest waits for the next run")
    return L + [""]


def load_blocks(frame, since_ms):
    """the PAIR_HELD BLOCK rows (dir LONG, ≥ the deploy): sleeve-named gates + the generic counter. dedup (t, pair, gate) across exports."""
    if not len(frame) or "gate" not in frame:
        return pd.DataFrame(columns=["t_ms", "pair", "gate", "sleeve"])
    b = frame[(frame.e.astype(str) == "BLOCK") & (frame.dir.astype(str) == "LONG")
              & frame.gate.astype(str).isin(list(HELD_GATES) + [GENERIC_HELD])].copy()
    if not len(b):
        return pd.DataFrame(columns=["t_ms", "pair", "gate", "sleeve"])
    b = b.drop_duplicates(["t", "pair", "gate"])
    b["t_ms"] = [_ms(x) for x in b.t]
    b = b[b.t_ms.notna() & (b.t_ms >= since_ms)]
    b["sleeve"] = [HELD_GATES.get(str(g)) for g in b.gate]
    return b.sort_values("t_ms").reset_index(drop=True)


def holder_of(orders, pair, t_ms):
    """the FLATBASE_SHORT order holding `pair` over the 5-min bucket starting at t_ms (opened ≤ bucket end, not closed before the bucket),
    from the orders exports — or None (another holder / not in the exports)."""
    if orders is None or not len(orders):
        return None
    t0, t1 = pd.Timestamp(int(t_ms), unit="ms"), pd.Timestamp(int(t_ms) + BAR_MS, unit="ms")
    g = orders[(orders.es == FLATBASE_SHORT) & (orders.pair.astype(str) == str(pair)) & (orders.o <= t1)
               & (orders.c.isna() | (orders.c >= t0))]
    return None if not len(g) else g.iloc[-1]


def _live_th():
    import scout_willy_hold as WH
    return WH._live_th()


def price_long(sleeve, pair, t_ms, bud, now_ms, fetch=True, m1=None, th=None):
    """③ one blocked FRENZY long priced from the first 1m open AFTER the 5-min bucket floor with the sleeve's CURRENT exit (see the module docstring). m1 = an
    injectable (sym, start, limit) → 1m rows (tests). → dict(state PRICED / BRACKET / WAIT / UNPRICED, pct, lo, hi, exit_ms, how)."""
    import scout_frenzy_exits as FX
    from services.frenzy import frenzy_stop_for, frenzy_lite_c_enabled, frenzy_lite_c_tp, FRENZY_LITE_FAMILY, WILLY_EXIT_CLASS
    th = _live_th() if th is None else th
    # entry = the first 1m open AFTER the 5-min bucket floor (bucket + 1 min): the refusal lies somewhere inside the bucket, so its first
    # minute would be an up-to-4-min optimistic look-ahead; the FRENZY refusals fall seconds after the 5m close → ≤ 1 min late, conservative
    start = (int(t_ms) // MIN_MS) * MIN_MS + MIN_MS
    ckey = f"long_{sleeve}_{pair}_{start}_s{LONG_SLIP:.2f}"
    if m1 is None:
        c = _cache_get(ckey)
        if c is not None:
            return c
    willy = sleeve in WILLY_EXIT_CLASS
    if willy:
        wr = FX.tb_walker_rule(th)
        cap = int(wr["cap"])
    else:
        ex = dict(FX.live_exit(th, sleeve))
        ex["sl"] = float(frenzy_stop_for(th, sleeve))          # the engine's per-sleeve stop (FRENZY_LONG: frenzy_long_stop_pct, DL 317)
        cap = int(ex["cap"])
    if now_ms < start + (cap + 1) * MIN_MS:
        return dict(state="WAIT", how="exit window unfinished", cap_ms=cap * MIN_MS)
    res = _price_long_path(sleeve, pair, start, cap, bud, now_ms, fetch, m1, th, willy, (wr if willy else ex), FX)
    res["cap_ms"] = cap * MIN_MS
    final = res["state"] in ("PRICED", "BRACKET") or (res.get("how") == "no 1m klines at the refusal"
                                                       and now_ms >= start + (cap + 1) * MIN_MS + FINAL_AFTER_MS)
    if final and m1 is None:
        _cache_put(ckey, res)
    return res


def _price_long_path(sleeve, pair, start, cap, bud, now_ms, fetch, m1, th, willy, rule, FX):
    from services.frenzy import frenzy_lite_c_enabled, frenzy_lite_c_tp, FRENZY_LITE_FAMILY
    wr = ex = rule
    if m1 is not None:
        rows = m1(str(pair), start, cap + 1)
    else:
        import scout_long_stop4 as _LS
        rows = _LS.fapi_klines(str(pair), "1m", start, cap + 1, bud, CACHE, now=now_ms, fetch=fetch)
    if rows is None:
        return dict(state="UNPRICED", how="fetch stopped / --no-fetch")
    if not rows or int(rows[0][0]) != start:
        return dict(state="UNPRICED", how="no 1m klines at the refusal")
    k = [list(r[:5]) for r in rows]
    E = float(k[0][1])
    if willy:
        import scout_frenzy_willy_exit as FWX
        P = FWX.load_path(str(pair), start, E, cap, {}, k1m=FX._tb_k1m(k))
        ev = FX.tb_walker_eval(P, wr)
        if ev["why"] == "UNRES":
            return dict(state="UNPRICED", how="walker undecided")
        xm = _fnum(ev.get("xmin"))
        return dict(state="PRICED", pct=float(ev["pct"]) - LONG_SLIP, exit_ms=start + int((xm if xm is not None else cap) * MIN_MS),
                    how=FX.TB_WHOW.get(ev["why"], ev["why"]))

    def _w(e_):
        p_, xt, how = FX.walk_live(k, E, e_)
        return (None, None, how) if (p_ is None or how == "open") else (float(p_) - LONG_SLIP, xt, how)
    if sleeve in FRENZY_LITE_FAMILY and frenzy_lite_c_enabled(th) and ex.get("mode") == "fixed":
        ctp = float(frenzy_lite_c_tp(th, None, None)[0])       # LITE C out of zone (the lower TP); in zone = the LITE TP
        a, b = _w(dict(ex, tp=ctp)), _w(ex)
        if a[0] is None or b[0] is None:
            return dict(state="UNPRICED", how="path short")
        return dict(state="BRACKET", lo=min(a[0], b[0]), hi=max(a[0], b[0]), exit_ms=max(a[1], b[1]),
                    how=f"LITE C on: TP +{ctp:g} … +{ex['tp']:g} by the unstamped entry vol-mult / BTC 1h slope")
    p_, xt, how = _w(ex)
    if p_ is None:
        return dict(state="UNPRICED", how="path short")
    return dict(state="PRICED", pct=p_, exit_ms=xt, how=how)


SLEEVE_ORDER = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_LITE_RETRY", "FRENZY_WILLY", "FRENZY_WILLY_HT")


def blocked_lines(ctx):
    """③ FRENZY longs refused PAIR_HELD while a FLATBASE_SHORT held the pair, priced with each sleeve's current exit, vs the short."""
    L = [f"## 🩳③ FLATBASE observe — FRENZY longs blocked by an open FLATBASE short (DECISION_LOG {DL} addendum, observe-only, no gate)", ""]
    b = load_blocks(ctx["frame"], ctx["since_ms"])
    o = ctx["orders"]
    rows, gen, other = [], 0, 0
    for r in b.itertuples():
        named = isinstance(r.sleeve, str) and bool(r.sleeve)          # the generic counter's sleeve is None / NaN
        h = holder_of(o, r.pair, int(r.t_ms))
        if h is None:
            other += 1 if named else 0
            continue
        if not named:
            gen += 1
            continue
        rows.append((r, h))
    th = (ctx.get("th_long") or _live_th()) if rows else None      # no block → no config read
    last_exit, by, fold, holders, per_hold = {}, {}, 0, {}, {}
    for r, h in rows:
        hk = (str(h.opened_at), str(h.pair))
        k = (str(r.pair), r.sleeve, hk)                               # one position per (pair, sleeve, holder episode)
        if k in last_exit and int(r.t_ms) < last_exit[k]:
            fold += 1                                             # inside the life (or the cap, when unpriced) of a trade already counted
            continue
        x = price_long(r.sleeve, r.pair, int(r.t_ms), ctx["bud"], ctx["now_ms"], fetch=ctx["fetch"], m1=ctx.get("klines1m"), th=th)
        if x.get("exit_ms") is not None:
            last_exit[k] = int(x["exit_ms"])
        elif x.get("cap_ms") is not None:
            last_exit[k] = int(r.t_ms) + int(x["cap_ms"])         # WAIT / UNPRICED: the exit is unknown — fold for the cap
        by.setdefault(r.sleeve, []).append(x)
        holders[hk] = h
        if x.get("state") == "PRICED":
            per_hold.setdefault(hk, []).append((r.sleeve, x["pct"]))
    L.append(f"Cohort: {len(b)} PAIR_HELD LONG blocks ≥ deploy (any holder) · held by a FLATBASE_SHORT (inferred from the orders exports): "
             f"{len(rows)} sleeve-named · other / unknown holder: {other} · repeats inside a priced trade folded: {fold} · generic PAIR_HELD on a "
             f"FLATBASE-held pair: {gen} UNATTRIBUTED (FRENZY_LONG / WIDE / momentum share that counter — not priced)")
    med = {}
    if o is not None and len(o):
        for s in SLEEVE_ORDER:
            g = o[o.es == s]
            n_ = pd.to_numeric(g.entry_price, errors="coerce") * pd.to_numeric(g.quantity, errors="coerce")
            n_ = n_.dropna()
            med[s] = float(n_.median()) if len(n_) else None
    for s in SLEEVE_ORDER:
        if s not in by:
            continue
        xs = by[s]
        p = [x["pct"] for x in xs if x.get("state") == "PRICED"]
        br = [x for x in xs if x.get("state") == "BRACKET"]
        usd = (sum(p) / 100.0 * med[s]) if (p and med.get(s)) else None
        k_ = lambda st: sum(1 for x in xs if x.get("state") == st)
        line = f"- {s}: {len(xs)} · {_st(p)}" + (f" · ≈ ${usd:+.2f} (× median {s} notional ${med[s]:.0f})" if usd is not None else "")
        if br:
            line += (f" · unpriced {len(br)} (LITE C: TP depends on the unstamped entry vol-mult / BTC 1h slope) — bracket avg "
                     f"[{sum(x['lo'] for x in br) / len(br):+.3f}, {sum(x['hi'] for x in br) / len(br):+.3f}] %")
        line += f" · waiting {k_('WAIT')} · unpriced (no data / fetch) {k_('UNPRICED')}"
        if s in ("FRENZY_WILLY", "FRENZY_WILLY_HT"):   # 🎲² (DECISION_LOG 356) the V2 per-order runner target is not reconstructible here
            line += (" · 🎲² priced under the fixed WILLY exit (+1.25 / −4 / 90, late TP, resting TP) — the DL 356 V2 runner target / "
                     "profit floor needs the episode spike high, not on a BLOCK row (a runner could book above or below this)")
        L.append(line)
    if holders:
        hs = list(holders.values())
        cl = [h for h in hs if str(h.status) == "CLOSED"]
        nets = [float(h.net) for h in cl if _fnum(h.net) is not None]
        usd_f = sum((_fnum(h.usd) or 0.0) + (_fnum(h.funding_fee_usd) or 0.0) for h in cl)
        L.append(f"- the holding shorts: {len(hs)} FLATBASE_SHORT holds · closed {len(cl)} {_st(nets)} · ${usd_f:+.2f} (pnl + funding) · "
                 f"open {len(hs) - len(cl)}")
        blk_usd, n_pos = 0.0, 0
        for hk, lst in per_hold.items():                          # CLOSED holders only (the same set as usd_f); ONE blocked position each
            if str(holders[hk].status) != "CLOSED":
                continue
            lst = sorted(lst, key=lambda z: LONG_PRECEDENCE.index(z[0]) if z[0] in LONG_PRECEDENCE else 99)
            s_, p_ = lst[0]
            if med.get(s_):
                blk_usd += p_ / 100.0 * med[s_]; n_pos += 1
        L.append(f"- net of the 72 h hold (closed holds; one blocked position per hold, precedence {' > '.join(x.replace('FRENZY_', '') for x in LONG_PRECEDENCE)}): "
                 f"FLATBASE ${usd_f:+.2f} − blocked longs ≈ ${blk_usd:+.2f} ({n_pos} positions) = ${usd_f - blk_usd:+.2f} "
                 f"(positive = the hold paid for what it blocked; bracketed / unpriced blocks not in it)")
    elif not len(b):
        L.append("- no PAIR_HELD LONG block yet")
    if ctx["bud"].stop:
        L.append(f"- ⚠ fetch stopped: {ctx['bud'].why} — the rest waits for the next run")
    return L + [""]


# 📈 (DECISION_LOG 363, operator 2026-10-23: "add the re-sizing of FLATBASE if it ends up being good") — the frozen size ladder.
# Size = flatbase_invest_mult (lev 0.25 = 5× unchanged); each fill carries its invest multiplier in cell_multiplier. Every rung is judged by
# the SAME frozen gate as the sleeve (services.flatbase.flatbase_gate — first-30-on-10-days main read, net of funding, STEP_UP / OFF / REVIEW /
# REREAD) run on the CURRENT run of fills at that size (deep review: no parallel, unfrozen bar; no one-stop day trigger).
LADDER = (0.2, 0.5, 1.0)                     # momentum-long size (DL 348) → ½ slot → full slot
LADDER_TOL = 0.005                           # the engine stamps the exact config float — 0.25 is OFF the ladder, not "0.2"
FB_SLOT_MARGIN_REF = 456.0                   # ≈ one equal-split slot at ≈ $1,870 equity (for the cap note only)


def resize_lines(real, inv_now=None):
    """🔒 the frozen FLATBASE re-size ladder → printed lines (operator decides; nothing automatic). → STEP-UP candidate when flatbase_gate on
    the current run at this size says STEP_UP; STEP-DOWN candidate (rungs above 0.2) when it says OFF or REVIEW. Every step up also needs the
    operator to re-decide flatbase_max_open (FLATBASE holds no bot slot — the cap was sized for 0.2). Above 0.2 the ladder line — not the
    sleeve gate line above it (all fills, frozen first-30 sample) — is the size verdict. Off the ladder → no step verdict. Never raises."""
    try:
        cap = 0
        if inv_now is None:
            from types import SimpleNamespace as _NS
            with open(os.path.join(_R, "trading_config.json"), encoding="utf-8") as _fh:
                _fp = flatbase_params(_NS(**json.load(_fh)["thresholds"]))   # the engine's own reading (invalid → its fallback)
            inv_now, cap = float(_fp["invest"]), int(_fp.get("max_open") or 0)
        inv_now = float(inv_now)
        lvl = min(range(len(LADDER)), key=lambda i: abs(LADDER[i] - inv_now))
        off_ladder = abs(LADDER[lvl] - inv_now) > LADDER_TOL
        nxt = LADDER[lvl + 1] if lvl + 1 < len(LADDER) else None
        if len(real):
            srt = real.assign(_cm=pd.to_numeric(real.get("cell_multiplier"), errors="coerce")).sort_values("o")
        else:
            srt = real.assign(_cm=pd.Series(dtype=float))
        n_nan = int(srt._cm.isna().sum())
        known = srt[srt._cm.notna()]
        # only the CURRENT run at this size: fills since the last fill at a different size (0.5 → 0.2 → 0.5 never reuses old 0.5 fills)
        other = known[(known._cm - LADDER[lvl]).abs() > LADDER_TOL]
        at = known[(known._cm - LADDER[lvl]).abs() <= LADDER_TOL]
        if len(other):
            at = at[at.o > other.o.max()]
        days = at.o.dt.date.nunique() if len(at) else 0
        L = [f"📈 Re-size ladder (DECISION_LOG 363, frozen): {' → '.join(f'{x:g}' for x in LADDER)} invest × 5× · now {inv_now:g} · current run "
             f"at this size: {len(at)} fills on {days} days" + (f" · avg {at.net.mean():+.3f} %" if len(at) else "")
             + (f" · ⚠ {n_nan} fill(s) with unknown size not counted" if n_nan else "")]
        if len(known) and abs(float(known._cm.iloc[-1]) - inv_now) > LADDER_TOL:
            L.append(f"  ⚠ the newest fill was sized {float(known._cm.iloc[-1]):g} but trading_config.json says {inv_now:g} — sync the config")
        if off_ladder:
            L.append(f"  ⚠ {inv_now:g} is OFF the ladder — no step verdict (set flatbase_invest_mult to a rung)")
            return L
        v, txt = flatbase_gate([(r.o.to_pydatetime(), r.net, r.close_reason) for r in at.itertuples()])
        L.append(f"  gate on this run: {v} — {txt}")
        if lvl > 0 and v in ("OFF", "REVIEW"):
            L.append(f"  🛑 STEP-DOWN candidate to {LADDER[lvl - 1]:g} (operator decides)")
            return L
        if nxt is None:
            L.append("  top of the ladder (full slot) — no further step")
            return L
        if v == "STEP_UP":
            m = (f" — with the cap at {cap} that is up to ≈ ${cap * FB_SLOT_MARGIN_REF * nxt:,.0f} margin outside the bot slots "
                 f"(≈ {cap * FB_SLOT_MARGIN_REF * nxt / (FB_SLOT_MARGIN_REF * 4):.1f}× the 4-slot book at ≈ $1,870 equity)" if cap else "")
            L.append(f"  ✅ STEP-UP candidate {LADDER[lvl]:g} → {nxt:g} (operator decides) · ALSO re-decide flatbase_max_open{m}")
        else:
            L.append(f"  next step {LADDER[lvl]:g} → {nxt:g} when this gate reads STEP_UP (now {v})")
        return L
    except Exception as e:
        return [f"📈 Re-size ladder: unreadable ({str(e)[:80]})"]


# 🎯 (DECISION_LOG 367, operator 2026-10-24: "something to consider, lowering the TP of FLATBASE trades" — after KAT 10-24 peaked +6 % and
# stopped −15) — exit shadows on every REAL fill: (take-profit %, stop %), the 72 h hold unchanged. TP5_SL10 / TP5_SL5 (operator 2026-10-24:
# "a 5 % TP with a different SL?"); TP8_SL10 = the isolating cell (today's TP, the 10 % stop alone). Prior = the exit study's S4 · 72 h cells
# (reports/FRENZY_FLATBASE_SHORT_2026-10-22_EXITS_cells.csv, the NO-FUNDING columns — the shadows leave funding out too): today's −8 / 15 wins
# every half against every TP +5 cell; TP +3 and a 5 % stop were never in its grid (tp_backtest_note prints the per-half Δ at read time).
# TP1 (operator 2026-10-24: "almost all FLATBASE trades reached 1 % — that would give ~100 % WR"): a scratch run of the DL 368 study machinery
# (S4 · 72 h, reports/FRENZY_FLATBASE_TP1_2026-10-24_*) gave TP +1 / SL −15 WR 92 % but avg −0.05 %/trade in 2026 (with funding;
# +0.21 / −0.20 / +0.02 / −0.17 per half) vs live +0.93 — break-even ≈ 95 % if every loss were a full stop; realised 2026 break-even 92.3 %
# vs WR 91.8 % (time exits / funding count as losses too) → about break-even (±0.1 by cost model), ≈ 0.9 %/trade below the live exit.
# Tracked like the others. TP1_SL10 / TP1_SL5 (operator: "the SL should be narrower too"): same run
# — stop 10 WR 90 % avg −0.07, stop 5 WR 82 % avg −0.19 (2026); a narrower stop cuts more +1 % winners than it saves.
TP_SHADOWS = {"TP5": (5.0, 15.0), "TP3": (3.0, 15.0), "TP5_SL10": (5.0, 10.0), "TP5_SL5": (5.0, 5.0), "TP8_SL10": (8.0, 10.0),
              "TP1": (1.0, 15.0), "TP1_SL10": (1.0, 10.0), "TP1_SL5": (1.0, 5.0),
              # 🎯 (DECISION_LOG 390, operator 2026-10-27) tighter stops under the small TP, registered AFTER the 12-fill read (in-sample):
              # on those 12, TP +2 / SL −5 +0.18 · TP +1 / SL −4 +0.09 · +1 / −3 −0.49 · +1 / −2 −0.75 %/fill vs live −3.57
              "TP1_SL4": (1.0, 4.0), "TP1_SL3": (1.0, 3.0), "TP1_SL2": (1.0, 2.0), "TP2_SL5": (2.0, 5.0)}
TP_SH_LATE = ("TP1_SL4", "TP1_SL3", "TP1_SL2", "TP2_SL5")      # the DL 390 additions
# Bonferroni per shadow: the DL 367 / 370 eight keep their frozen ×8 (pre-committed gates never move); the DL 390 four are judged at ×12
# (the whole family — the stricter bar for the late additions)
TP_SH_M = {k: (12 if k in TP_SH_LATE else 8) for k in TP_SHADOWS}   # literals: a later addition never moves either bar
TP_SH_DL = 367
TP_SH_PARITY_TOL, TP_SH_PARITY_MIN = 0.50, 0.90                # the live re-walk vs the booked pnl_percentage (+ 100 % on the live-stopped fills)
TP_SH_N, TP_SH_DAYS, TP_SH_MIN_GAIN, TP_SH_SHARE = 30, 10, 0.10, 0.50
TP_SH_BOOT_N, TP_SH_BOOT_SEED, TP_SH_ALPHA = 4000, 367, 0.05   # UTC-day-block bootstrap; one-sided α / TP_SH_M[k] (Bonferroni: ×8 the original eight, ×12 the DL 390 four)
TP_SH_ENTRY_TOL_MS = 60_000                                   # a fill ≤ 60 s into its 5m bar starts the path AT that bar (like price_shadow)
TP_SH_EPS = 0.005                                             # |Δ| ≤ this = unchanged in the split lines
TP_SH_STATE = os.path.join(_R, "reports", ".flatbase_tp_shadow_gate.json")
TP_SH_FINAL = "cache"                                         # → CACHE/tp_shadow_unpriceable.json at call time (UNPRICEABLE_FINAL fills, never re-fetched)
TP_SH_FINAL_NAME = "tp_shadow_unpriceable.json"
TP_SH_CELLS = os.path.join(_R, "reports", "FRENZY_FLATBASE_SHORT_2026-10-22_EXITS_cells.csv")
TP_SH_CELLS_TP1 = os.path.join(_R, "reports", "FRENZY_FLATBASE_TP1_2026-10-24_cells.csv")   # (370) the TP +1 / +2 × stop 5 / 10 / 15 run


def _th_tp(tp, stop=15.0):
    from types import SimpleNamespace
    th = _frozen_th()
    return SimpleNamespace(**dict(vars(th), flatbase_tp_pct=float(tp), flatbase_stop_pct=float(stop)))


def _tp_live():
    """(TP %, stop %) of the shipped exit — flatbase_params on the frozen DL 348 rule (never a later settings change)."""
    p = flatbase_params(_frozen_th())
    return float(p['tp']), float(p['stop'])


def tp_path_start(t_ms):
    """→ (path start ms, entry bar skipped). Real fills open ~HH:00:09–13: a fill ≤ 60 s into its 5m bar starts AT that bar (the bar opening
    at the boundary — price_shadow's and the exit study's convention; a few seconds of pre-fill prints); later in the bar → the next bar."""
    b = int(t_ms) // BAR_MS * BAR_MS
    return (b, False) if int(t_ms) - b <= TP_SH_ENTRY_TOL_MS else (b + BAR_MS, True)


def tp_shadow_walk(entry_price, rows, slip=SLIP_DEF):
    """→ {"LIVE" | shadow: price_path_exit dict} on one path, or None (no path / a variant still OPEN). entry_price = the REAL fill (already
    slipped): price_path_exit slips its decision price by (1 − slip), so it gets entry / (1 − slip) and its entry slip cancels — the entry is
    priced at slip 0, the taker slip stays on a stop / time exit only (a resting TP has none)."""
    s = float(slip or 0)
    dec = float(entry_price) / (1.0 - s / 100.0)
    out = {}
    for k, (tp, sl) in (("LIVE", _tp_live()),) + tuple(TP_SHADOWS.items()):
        x = price_path_exit(dec, rows, _th_tp(tp, sl), slip_pct=s)
        if x is None or x["reason"] == "OPEN":
            return None
        out[k] = x
    return out


def _tp_rows(r, bud, now_ms, fetch=True, klines=None):
    start, _ = tp_path_start(int(pd.Timestamp(r.o).value // 1_000_000))
    hb = int(round(flatbase_params(_frozen_th())['max_hold_h'] * 12))
    if klines is None:
        import scout_long_stop4 as _LS
        return _LS.fapi_klines(str(r.pair), "5m", start, hb, bud, CACHE, now=now_ms, fetch=fetch), start, hb
    return klines(str(r.pair), start, hb), start, hb


def tp_shadow_price(r, bud, now_ms, fetch=True, klines=None, slip=SLIP_DEF):
    """one real fill → {"LIVE": re-walk at the shipped exit, <shadow>: …} (net %, price + fees, NO funding — the same hold-independent
    convention for every variant) or None when the path is unreadable / a variant's 72 h window is not over. Never raises."""
    try:
        rows, _, _ = _tp_rows(r, bud, now_ms, fetch=fetch, klines=klines)
        w = tp_shadow_walk(float(r.entry_price), rows, slip) if rows else None
        return None if w is None else {k: float(x["net_pct"]) for k, x in w.items()}
    except Exception:
        return None


def tp_shadow_eval(r, bud, now_ms, fetch=True, klines=None, slip=SLIP_DEF, finals=None):
    """→ (status, walk). PRICED (walk) · HOLD (not fetched / budget stop / window not over) · UNPRICEABLE_FINAL (the 72 h window closed
    > 1 day ago and the path came back empty or OPEN on TWO separate runs — a single timeout also returns [], so one miss is never final;
    recorded in `finals` {key: …} and never re-fetched once final). Never raises."""
    try:
        start, _ = tp_path_start(int(pd.Timestamp(r.o).value // 1_000_000))
        key = f"{r.pair}|{start}"
        _fe = finals.get(key) if finals is not None else None
        if finals is not None and key in finals and (not isinstance(_fe, dict) or _fe.get("final", True) is True):   # malformed → final
            return "UNPRICEABLE_FINAL", None
        rows, start, hb = _tp_rows(r, bud, now_ms, fetch=fetch, klines=klines)
        if rows is None:                                                       # budget / cooldown / no fetch — retry next run
            return "HOLD", None
        w = tp_shadow_walk(float(r.entry_price), rows, slip) if rows else None
        if w is not None:
            return "PRICED", w
        if int(now_ms) >= start + hb * BAR_MS + FINAL_AFTER_MS and finals is not None:
            e = finals.setdefault(key, dict(pair=str(r.pair), opened=str(r.o), misses=0, final=False))
            e.update(misses=int(e.get("misses") or 0) + 1, bars=len(rows), at=pd.Timestamp.now("UTC").isoformat())
            if e["misses"] >= 2:                                               # a second run's miss → final
                e["final"] = True
                return "UNPRICEABLE_FINAL", None
        return "HOLD", None
    except Exception:
        return "HOLD", None


def _live_kind(cr):
    s = str(cr or "").strip().upper().split(" ")[0]
    return "STOP" if s in STOP_REASONS else "TIME" if s.startswith("MAX_HOLD") else "TP" if "TP" in s else "OTHER"


def tp_split_line(k, ok):
    """the two-sided read of one shadow on every priced fill: live-STOPPED → saved / unchanged / deeper · live-TP → give-back (the shadow's
    lower TP) / cut winners (the shadow STOPPED a live winner: Δ ≈ −(TP_live + stop)) · live-TIME / other. n · ΣΔ each."""
    b = {}
    for r, w in ok:
        d = w[k]["net_pct"] - w["LIVE"]["net_pct"]
        lk, sr = _live_kind(r.close_reason), str(w[k]["reason"])
        if lk == "STOP":
            c = "stopped→saved" if d > TP_SH_EPS else "stopped→deeper" if d < -TP_SH_EPS else "stopped→unchanged"
        elif lk == "TP":
            c = ("TP→cut winner" if sr in STOP_REASONS else "TP→give-back" if d < -TP_SH_EPS else "TP→unchanged")
        else:
            c = f"{lk.lower()}→" + ("cut" if sr in STOP_REASONS else "TP" if "TP" in sr else "time")
        n, s = b.get(c, (0, 0.0))
        b[c] = (n + 1, s + d)
    order = ("stopped→saved", "stopped→unchanged", "stopped→deeper", "TP→give-back", "TP→cut winner", "TP→unchanged")
    keys = [c for c in order if c in b] + sorted(c for c in b if c not in order)
    return f"  split {k}: " + (" · ".join(f"{c} {b[c][0]} (Δ Σ {b[c][1]:+.2f})" for c in keys) if keys else "none priced")


def tp_backtest_note(k, path=TP_SH_CELLS):
    """the exit study's per-half Δ (shadow − live, NO-funding columns, S4 · 72 h) for the same cell, or why there is none. Never raises."""
    try:
        tp, sl = TP_SHADOWS[k]
        ltp, lsl = _tp_live()
        h = flatbase_params(_frozen_th())['max_hold_h']
        d = pd.read_csv(path, usecols=lambda c: c == "cell" or c.endswith("_nofund"))
        d = d[d.cell.astype(str).str.startswith(f"S4|") & d.cell.astype(str).str.contains(f"|{h:g}h|", regex=False)]
        grid_tp = {c.split("|")[1] for c in d.cell}
        grid_sl = {c.split("|")[3] for c in d.cell}
        miss = ([f"TP +{tp:g} %"] if f"tp{tp:g}" not in grid_tp else []) + ([f"SL −{sl:g} %"] if f"stop{sl:g}" not in grid_sl else [])
        if miss:
            _n1 = tp1_backtest_note(tp, sl, ltp, lsl, h)
            if _n1:
                return _n1
            if tp in (1.0, 2.0):   # the DL 370 run has TP +1 / +2 — only the stop is missing there
                miss = [m_ for m_ in miss if not m_.startswith("TP")]
            return f"backtest: {' and '.join(miss)} never in the exit-study grid — no prior"
        cell, live = d[d.cell == f"S4|tp{tp:g}|{h:g}h|stop{sl:g}"], d[d.cell == f"S4|tp{ltp:g}|{h:g}h|stop{lsl:g}"]
        if not len(cell) or not len(live):
            return "backtest: cell missing in the exit study"
        hs = (("2025", "2025"), ("26-H1", "2026H1"), ("26-H2", "2026H2"))
        return "backtest Δ vs live (no funding, S4 · 72 h) " + " / ".join(
            f"{lab} {float(cell[f'{c}_nofund'].iloc[0]) - float(live[f'{c}_nofund'].iloc[0]):+.2f}" for lab, c in hs)
    except Exception as e:
        return f"backtest: unreadable ({str(e)[:60]})"


def tp1_backtest_note(tp, sl, ltp, lsl, h, path=None):
    """(370) the TP +1 / +2 run's per-half Δ (shadow − live, NO-funding columns, S4 · 72 h; cells named tp{tp}|{h}h|stop{sl}) or None."""
    try:
        d = pd.read_csv(path or TP_SH_CELLS_TP1, usecols=lambda c: c == "cell" or c.endswith("_nofund"))
        cell, live = d[d.cell == f"tp{tp:g}|{h:g}h|stop{sl:g}"], d[d.cell == f"tp{ltp:g}|{h:g}h|stop{lsl:g}"]
        if not len(cell) or not len(live):
            return None
        hs = (("25-H1", "2025H1"), ("25-H2", "2025H2"), ("26-H1", "2026H1"), ("26-H2", "2026H2"))
        return "backtest (DL 370 run) Δ vs live (no funding, S4 · 72 h) " + " / ".join(
            f"{lab} {float(cell[f'{c}_nofund'].iloc[0]) - float(live[f'{c}_nofund'].iloc[0]):+.2f}" for lab, c in hs)
    except Exception:
        return None


def tp_boot(d, days, n=TP_SH_BOOT_N, seed=TP_SH_BOOT_SEED, m=None):
    """UTC-day-block bootstrap of the pooled avg Δ → (one-sided lower bound, upper bound) at α / m each side (Bonferroni over the m
    shadows); None (< 2 days)."""
    import numpy as np
    m = 8 if m is None else m   # the original eight's frozen ×8 (DL 390: the late four pass m=12 via TP_SH_M)
    g = pd.DataFrame(dict(p=np.asarray(d, float), d=[str(x) for x in days])).groupby("d").p.agg(["sum", "count"])
    if len(g) < 2:
        return None
    s, c = g["sum"].values, g["count"].values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    mu = s[idx].sum(1) / c[idx].sum(1)
    q = 100.0 * TP_SH_ALPHA / m
    return float(np.percentile(mu, q)), float(np.percentile(mu, 100.0 - q))


def tp_gate_read(k, smp):
    """🔒 one shadow's frozen read on the frozen sample [(row, walk)] → (verdict, text). Parity is checked by the caller."""
    ltp, lsl = _tp_live()
    tp, sl = TP_SHADOWS[k]
    d = [w[k]["net_pct"] - w["LIVE"]["net_pct"] for _, w in smp]
    days = [r.o.date() for r, _ in smp]
    avg = sum(d) / len(d)
    m = TP_SH_M.get(k, len(TP_SHADOWS))
    ci = tp_boot(d, days, m=m)
    lvl = 1.0 - TP_SH_ALPHA / m
    byday = {}
    for x, dd in zip(d, days):
        byday[dd] = byday.get(dd, 0.0) + x
    pos_d = [v for v in byday.values() if v > 0]
    day_sh = max(pos_d) / sum(pos_d) if pos_d else 1.0
    pos_f = [x for x in d if x > 0]
    fill_sh = max(pos_f) / sum(pos_f) if pos_f else 1.0
    fail = ([f"avg < +{TP_SH_MIN_GAIN:g}"] if avg < TP_SH_MIN_GAIN else []) + (["CI lower bound ≤ 0"] if ci is None or ci[0] <= 0 else []) \
        + ([f"top day ≥ {TP_SH_SHARE:.0%}"] if day_sh >= TP_SH_SHARE else []) + ([f"top fill ≥ {TP_SH_SHARE:.0%}"] if fill_sh >= TP_SH_SHARE else [])
    v = "KEEP" if fail else "CANDIDATE"
    ci_t = (f"day-block bootstrap {TP_SH_BOOT_N} (seed {TP_SH_BOOT_SEED}) one-sided {lvl:.2%} LB {ci[0]:+.3f} (Bonferroni ×{m}; "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}])" if ci else "bootstrap n/a (< 2 days)")
    txt = (f"first {len(smp)} real fills on {len(set(days))} UTC days (incl. the 10-24 motivating batch — operator override, in-sample): avg Δ ({k} − live TP +{ltp:g} % / SL −{lsl:g} %) "
           f"{avg:+.3f} %/fill · {ci_t} · top day {day_sh:.0%} / top fill {fill_sh:.0%} of the gain → "
           + (f"CANDIDATE: operator decides TP +{tp:g} % / SL −{sl:g} % · if adopted: revert to TP +{ltp:g} % / SL −{lsl:g} % if the first 15 fills "
              f"after the switch average < the live-exit re-walk on the same fills" if v == "CANDIDATE"
              else f"keep TP +{ltp:g} % / SL −{lsl:g} % ({'; '.join(fail)})")
           + (f" · in-sample haircut 30–50 % (levels chosen after the 10-27 12-fill read — DL 390 — which is in the sample) → expect {avg * 0.5:+.3f}…{avg * 0.7:+.3f}"
              if k in TP_SH_LATE else
              f" · in-sample haircut 30–50 % (levels chosen after the 10-24 batch, which is in the sample) → expect {avg * 0.5:+.3f}…{avg * 0.7:+.3f}")
           + f" · {tp_backtest_note(k)}")
    return v, txt


# 🩳📋 (DECISION_LOG 389) OBS-FB1 "small pump, shallow off-peak" — FROZEN observe line, registered 2026-10-27 from the 12 real fills
# (operator: "record this … for next batch review of this sleeve"). On those 12, the TP +1 / SL −5 shadow lost only on BMT (run 70.4 %)
# and ORDER (run 195.8 %, off-peak −34.8 %); every winner had run ≤ 67.8 % and off-peak > −27.2 %. In-sample + chance-level (shuffled-label
# null: as clean a split in 49 % of random labelings), and the 21-month study refutes it (zone + TP1/SL5 −0.06 %/fill vs live +1.42) → NO
# verdict here: the zone is split on every priced fill under the live exit AND the TP1_SL5 shadow; fresh fills (opened ≥ the pin) are the
# evidence, the 12 registration fills are shown apart. Read at the FLATBASE deep review / the 30-fill main read with the expectancy bar.
OBS_PK_RUN_MAX, OBS_PK_OFFPK_MIN = 67.82, -27.21                 # frozen: entry_frenzy_run_pct ≤ · entry_flatbase_off_peak_pct >
OBS_PK_PIN = pd.Timestamp("2026-10-27T19:30:00")                 # the registration export (scalpars_orders_paper_2026-10-27_16-29-11, UTC)
OBS_PK_SHADOW = "TP1_SL5"


def obs_pk_zone(run_pct, off_peak_pct):
    """zone / out / unscored for one fill (frozen thresholds). Pure."""
    try:
        rp, op = float(run_pct), float(off_peak_pct)
    except (TypeError, ValueError):
        return "unscored"
    if rp != rp or op != op:
        return "unscored"
    return "zone" if (rp <= OBS_PK_RUN_MAX and op > OBS_PK_OFFPK_MIN) else "out"


def obs_pk_lines(ok):
    """OBS-FB1 lines from the priced exit-shadow fills [(row, walks)] — never raises."""
    L = [f"🩳📋 OBS-FB1 (DECISION_LOG 389, frozen observe line, no verdict): zone = run ≤ {OBS_PK_RUN_MAX:g} % ∧ off-peak > {OBS_PK_OFFPK_MIN:g} % · "
         f"live exit vs {OBS_PK_SHADOW} shadow · fresh = opened ≥ {OBS_PK_PIN:%Y-%m-%d %H:%M} UTC"]
    try:
        def blk(rows, label):
            parts = []
            for z in ("zone", "out", "unscored"):
                g = [(r, w) for r, w in rows if obs_pk_zone(getattr(r, 'entry_frenzy_run_pct', None), getattr(r, 'entry_flatbase_off_peak_pct', None)) == z]
                if not g:
                    parts.append(f"{z} N 0")
                    continue
                lv = [w['LIVE']['net_pct'] for _, w in g]; sh = [w[OBS_PK_SHADOW]['net_pct'] for _, w in g]
                parts.append(f"{z} N {len(g)} · live WR {sum(x > 0 for x in lv) / len(g):.0%} avg {sum(lv) / len(g):+.2f} · "
                             f"{OBS_PK_SHADOW} WR {sum(x > 0 for x in sh) / len(g):.0%} avg {sum(sh) / len(g):+.2f}")
            return f"  {label}: " + " | ".join(parts)
        reg = [(r, w) for r, w in ok if r.o < OBS_PK_PIN]
        fresh = [(r, w) for r, w in ok if r.o >= OBS_PK_PIN]
        L.append(blk(fresh, f"FRESH (the evidence) {len(fresh)} fill(s)"))
        L.append(blk(reg, f"registration (in-sample: opened before the pin — the rule was read from these) {len(reg)} fill(s)"))
        return L
    except Exception as e:
        return L + [f"  unreadable ({str(e)[:80]})"]


def hhmm(minutes):
    """minutes → 'HH:MM' (the dashboard's duration format; hours past 99 keep growing). Pure."""
    m = max(0, int(math.floor(float(minutes) + 0.5)))   # half-up (round() is half-to-even and every mid-bar value ends in .5)
    return f"{m // 60:02d}:{m % 60:02d}"


def tp_hold_lines(ok):
    """⏱ (DECISION_LOG 395, operator 10-27) average hold per exit on the priced fills: the real booked hold (opened → closed), the live
    re-walk and every shadow ((bars − ½) × 5 min from the path start: the exit placed mid-bar, ± 2.5 min; a fill > 60 s into its bar starts
    the walk at the next bar) + the median. Shorter holds free the money / the pair sooner (an open FLATBASE short blocks FRENZY longs on its coin). Never raises."""
    try:
        import statistics as _stt

        def _h(xs):
            return (sum(xs) / len(xs), _stt.median(xs)) if xs else (None, None)
        booked = [(pd.Timestamp(r.c) - pd.Timestamp(r.o)).total_seconds() / 60.0 for r, _ in ok
                  if getattr(r, "c", None) is not None and not pd.isna(r.c) and pd.Timestamp(r.c) >= pd.Timestamp(r.o)]
        def lead(r):   # minutes from the fill to the walk's first bar (a fill > 60 s into its bar starts the walk at the next bar)
            o_ms = int(pd.Timestamp(r.o).value // 1_000_000)
            return (tp_path_start(o_ms)[0] - o_ms) / 60_000.0
        parts = []
        a, m = _h(booked)
        if a is not None:
            parts.append(f"booked {hhmm(a)} (median {hhmm(m)})")
        for k in ("LIVE",) + tuple(TP_SHADOWS):
            a, m = _h([lead(r) + (w[k]["bars"] - 0.5) * 5.0 for r, w in ok if w.get(k, {}).get("bars") is not None])   # exit mid-bar (± 2.5 min)
            if a is not None:
                parts.append(f"{'live re-walk' if k == 'LIVE' else k} {hhmm(a)} (median {hhmm(m)})")
        return [f"  ⏱ avg hold per exit, HH:MM (DL 395; walks ± 2.5 min, {len(ok)} priced fills): " + " · ".join(parts)] if parts else []
    except Exception as e:
        return [f"  ⏱ avg hold unreadable ({str(e)[:60]})"]


def tp_shadow_lines(real, bud, now_ms, fetch=True, klines=None, slip=SLIP_DEF, state_path=TP_SH_STATE, final_path=TP_SH_FINAL):
    """🔒 (DECISION_LOG 367) the frozen exit-shadow reads → printed lines. Δ = shadow − the live re-walk (same path, same fees / slippage).
    Split lines on every priced fill. Frozen sample = the first 30 real fills on ≥ 10 UTC days of the FLATBASE cohort run() builds (DL 348
    deploy + 10 min, every paper + live export, operator-closed out) — operator override 2026-10-24 ("COUNT THEM!"): the 6 motivating 10-24
    fills are counted (in-sample), the verdict prints the 30–50 % haircut; UNPRICEABLE_FINAL fills are out (counted).
    CANDIDATE (operator decides) iff avg Δ ≥ +0.10 ∧ day-block bootstrap Bonferroni LB > 0 ∧ no UTC day / fill ≥ 50 % of the positive Δ —
    read only when parity on the sample ≥ 90 % ∧ 100 % on its live-stopped fills; else KEEP. A sample fill still HOLD holds. Never raises."""
    import json as _j
    ltp, lsl = _tp_live()
    L = [f"🎯 Exit shadows (DECISION_LOG {TP_SH_DL} / 370 / 390 — {', '.join(TP_SH_LATE)} added by DL 390, frozen, observe-only): "
         f"{' · '.join(f'{k} TP +{tp:g} % / SL −{sl:g} %' for k, (tp, sl) in TP_SHADOWS.items())} vs the live TP +{ltp:g} % / SL −{lsl:g} % on the "
         f"same path (72 h · fees · entry at the real fill, taker slip {slip:.3f} % on stop / time exits · funding left out)"]
    try:
        if final_path == TP_SH_FINAL:
            final_path = os.path.join(CACHE, TP_SH_FINAL_NAME)          # resolved per call: a test's CACHE monkeypatch keeps it in tmp
        finals = {}
        try:
            if final_path and os.path.exists(final_path):
                with open(final_path, encoding="utf-8") as fh:
                    finals = _j.load(fh) or {}
        except Exception:
            finals = {}
        n_fin0 = _j.dumps(finals, sort_keys=True)                      # a miss count or a new final both rewrite the file
        rs = real.sort_values("o") if len(real) else real
        rows_ = list(rs.itertuples())
        ev = [tp_shadow_eval(r, bud, now_ms, fetch=fetch, klines=klines, slip=slip, finals=finals) for r in rows_]
        if final_path and _j.dumps(finals, sort_keys=True) != n_fin0:
            try:
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                with open(final_path, "w", encoding="utf-8") as fh:
                    _j.dump(finals, fh, indent=1)
            except Exception:
                pass
        ok = [(r, w) for r, (s, w) in zip(rows_, ev) if s == "PRICED"]
        n_fin = sum(1 for s, _ in ev if s == "UNPRICEABLE_FINAL")
        n_skip = sum(1 for r in rows_ if tp_path_start(int(pd.Timestamp(r.o).value // 1_000_000))[1])
        L.append(f"  priced {len(ok)} / {len(rows_)} real fills · hold {sum(1 for s, _ in ev if s == 'HOLD')} · UNPRICEABLE_FINAL {n_fin} (out, "
                 f"never re-fetched)" + (f" · {n_skip} fill(s) > 60 s into the 5m bar → path from the next bar" if n_skip else ""))
        if not ok:
            L.append(f"  0 / {len(rows_)} real fills priced yet")
        else:
            par = sum(1 for r, w in ok if abs(w["LIVE"]["net_pct"] - float(r.pct)) <= TP_SH_PARITY_TOL) / len(ok)
            L.append(f"  parity (live re-walk vs booked, ±{TP_SH_PARITY_TOL:g}) {par:.0%} on all priced · " + " · ".join(
                [f"live re-walk avg {sum(w['LIVE']['net_pct'] for _, w in ok) / len(ok):+.3f}"]
                + [f"{k} avg {sum(w[k]['net_pct'] for _, w in ok) / len(ok):+.3f} (Δ Σ {sum(w[k]['net_pct'] - w['LIVE']['net_pct'] for _, w in ok):+.2f})"
                   for k in TP_SHADOWS]))
            L += tp_hold_lines(ok)
            L.append("  per fill: " + " · ".join(f"{r.pair} {r.o:%m-%d %H:%M} booked {float(r.pct):+.2f} | live {w['LIVE']['net_pct']:+.2f} "
                                                 + " ".join(f"{k} {w[k]['net_pct']:+.2f}" for k in TP_SHADOWS) for r, w in ok[-20:]))
            L += [tp_split_line(k, ok) for k in TP_SHADOWS]
            L += obs_pk_lines(ok)
        st = {}
        try:
            if state_path and os.path.exists(state_path):
                with open(state_path, encoding="utf-8") as fh:
                    st = _j.load(fh) or {}
        except Exception:
            st = {}
        smp, ds, n_post = None, set(), 0
        post = [(r, s, w) for r, (s, w) in zip(rows_, ev) if s != "UNPRICEABLE_FINAL"]
        n_post = len(post)
        for i, (r, _, _) in enumerate(post):
            ds.add(r.o.date())
            if i + 1 >= TP_SH_N and len(ds) >= TP_SH_DAYS:
                smp = post[:i + 1]
                break
        par_txt, par_ok = "", False
        if smp is not None and not any(s != "PRICED" for _, s, _ in smp):
            par_s = sum(1 for r, _, w in smp if abs(w["LIVE"]["net_pct"] - float(r.pct)) <= TP_SH_PARITY_TOL) / len(smp)
            stp = [(r, w) for r, _, w in smp if _live_kind(r.close_reason) == "STOP"]
            # an exchange-BACKSTOP fill books at the backstop (≈ stop + 0.5) — the re-walk prices the bot stop: within 1.0 for those
            _tol = lambda r: TP_SH_PARITY_TOL + (0.5 if "BACKSTOP" in str(r.close_reason).upper() else 0.0)
            stp_ok = sum(1 for r, w in stp if abs(w["LIVE"]["net_pct"] - float(r.pct)) <= _tol(r))
            par_ok = par_s >= TP_SH_PARITY_MIN and stp_ok == len(stp)
            par_txt = f"parity on the sample {par_s:.0%} (≥ {TP_SH_PARITY_MIN:.0%}) · live-stopped {stp_ok}/{len(stp)} (100 % required)"
            L.append(f"  frozen sample: {par_txt}" + ("" if par_ok else " → no verdict (parity)"))
        for k in TP_SHADOWS:
            name = f"FB_TP_{k}"
            if name in st:
                L.append(f"  {name}: {st[name]['verdict']} — {st[name]['text']} (frozen {str(st[name].get('at', ''))[:16]} UTC)")
                continue
            if smp is None:
                L.append(f"  {name}: COLLECTING — {n_post}/{TP_SH_N} real fills on {len(ds)}/{TP_SH_DAYS} days · {tp_backtest_note(k)}")
                continue
            if any(s != "PRICED" for _, s, _ in smp):
                L.append(f"  {name}: HOLD — {sum(s != 'PRICED' for _, s, _ in smp)} fill(s) in the frozen {len(smp)} not priced yet")
                continue
            if not par_ok:
                L.append(f"  {name}: no verdict (parity)")
                continue
            v, txt = tp_gate_read(k, [(r, w) for r, _, w in smp])
            txt += f" · {par_txt}"
            L.append(f"  {name}: {v} — {txt}")
            if state_path:
                try:
                    st[name] = {"verdict": v, "text": txt, "at": pd.Timestamp.now("UTC").isoformat()}
                    with open(state_path, "w", encoding="utf-8") as fh:
                        _j.dump(st, fh, indent=1)
                except Exception:
                    pass
        return L
    except Exception as e:
        return L + [f"  unreadable ({str(e)[:80]})"]


# ═══════════════════════════ 🩳🔒 OBS-FB2 — LOW-TURNOVER SWITCH (DECISION_LOG 407, operator 2026-10-28, frozen, observe-only) ═══════════════════════════
# "yes register it" · "+5 −10 is interesting too, mainly due to the Avg Hold time" · "register it in the same line". The setups the sleeve
# REFUSES today (turnover ≤ 1 → ledger SHADOW, no order) priced under two exits — LOW×TODAY (+8 / −15) and LOW×TP5_SL10 (+5 / −10) —
# each with funding, against the real traded cohort (turnover > 1) at today's exit over the same period. The joint read that motivated it
# (B25, in-sample — registration only, never counted): 7 shadows · today +4.83 (86 %, hold 50:48) · +5/−10 +4.02 (86 %, hold 19:32) vs
# the 13 traded fills −2.71 % · the small exits (+2/−5, +1/−4) helped the traded fills but destroyed the low-turnover edge.
OBS_FB2_DL = 407
OBS_FB2_CELLS = {"LOW×TODAY": None, "LOW×TP5_SL10": (5.0, 10.0)}   # None = the live exit (_frozen_th)
OBS_FB2_N, OBS_FB2_DAYS, OBS_FB2_P, OBS_FB2_MARGIN = 30, 10, 0.95, 1.0
OBS_FB2_STATE = os.path.join(_R, "reports", ".flatbase_obs_fb2_gate.json")


def price_shadow_cell(row, cell, bud, now_ms, fetch=True, klines=None, slip_pct=SLIP_DEF, funding=None):
    """one shadow setup under an OBS-FB2 cell → (net % incl. funding, hold minutes, funding known) or None (window open / no data)."""
    th = _frozen_th() if OBS_FB2_CELLS[cell] is None else _th_tp(*OBS_FB2_CELLS[cell])
    hb = int(round(flatbase_params(_frozen_th())['max_hold_h'] * 12))
    start = int(pd.Timestamp(row.t).value // 1_000_000) + BAR_MS
    if klines is None:
        import scout_long_stop4 as _LS
        rows = _LS.fapi_klines(str(row.pair), "5m", start, hb, bud, CACHE, now=now_ms, fetch=fetch)
    else:
        rows = klines(str(row.pair), start, hb)
    if not rows:
        return None
    r = price_path_exit(float(row.px), rows, th, slip_pct=slip_pct)
    if r is None or r["reason"] == "OPEN":
        return None
    t1 = int(r["exit_open_ms"]) + BAR_MS
    f = funding(str(row.pair), start, t1) if funding is not None else funding_sum_pct(str(row.pair), start, t1, bud, now_ms, fetch=fetch)
    return (r["net_pct"] + (f or 0.0), (t1 - start) / 60_000.0, f is not None)


def _boot_p_pos(vals, days, n=4000, seed=7):
    """day-block bootstrap P(mean > 0). Pure."""
    import numpy as _np
    g = pd.Series(list(vals)).groupby(list(days)).apply(list).tolist()
    if len(g) < 2:
        return float("nan")
    rng = _np.random.default_rng(seed)
    hit = 0
    for _ in range(n):
        pick = [x for i in rng.integers(0, len(g), len(g)) for x in g[i]]
        hit += (sum(pick) / len(pick)) > 0
    return hit / n


OBS_FB2_MIN_COMPARATOR = 5


def obs_fb2_verdict(cell_rows, real_same, unpriced=0, unpriced_days=()):
    """the frozen read on the FIRST fresh setups by entry time (≥ OBS_FB2_N on ≥ OBS_FB2_DAYS days), decided only when every one of them
    is priced (review 2: a still-open 72 h window or a budget skip BLOCKS the read — early TP / SL exits never over-represented).
    cell_rows = [(t, pair, net)] of the first N fresh setups, by time; unpriced = how many of those N are not priced yet; real_same = the
    traded fills' net % over the same period (≥ OBS_FB2_MIN_COMPARATOR needed). → (verdict, text). Pure."""
    days = {pd.Timestamp(t).date() for t, _, _ in cell_rows} | set(unpriced_days or ())
    if len(cell_rows) + unpriced < OBS_FB2_N or len(days) < OBS_FB2_DAYS:
        return "COLLECTING", f"⏳ {len(cell_rows) + unpriced}/{OBS_FB2_N} fresh setups on {len(days)}/{OBS_FB2_DAYS} days"
    if unpriced:
        return "COLLECTING", f"⏳ {unpriced} of the first {OBS_FB2_N} fresh setups not priced yet (window open / data) — the read waits"
    if len(real_same) < OBS_FB2_MIN_COMPARATOR:
        return "COLLECTING", f"⏳ traded comparator N {len(real_same)} < {OBS_FB2_MIN_COMPARATOR} over the same period"
    v = [x for _, _, x in cell_rows]
    avg = sum(v) / len(v)
    ra = (sum(real_same) / len(real_same)) if real_same else float("nan")
    p = _boot_p_pos(v, [pd.Timestamp(t).date() for t, _, _ in cell_rows])
    pos = pd.Series([max(x, 0.0) for x in v])
    by_day = pos.groupby([pd.Timestamp(t).date() for t, _, _ in cell_rows]).sum()
    by_pair = pos.groupby([pr for _, pr, _ in cell_rows]).sum()
    conc = max(by_day.max(), by_pair.max()) / pos.sum() if pos.sum() > 0 else 1.0
    txt = (f"N {len(v)} on {len(days)} days · avg {avg:+.3f} % (funding incl.) · P(mean > 0) {p:.2f} · traded same period "
           f"{'–' if ra != ra else f'{ra:+.3f} %'} (N {len(real_same)}) · top day / pair share {conc:.0%}")
    if avg <= ra:
        return "RETIRE", txt + " → the refused setups do NOT beat the traded ones"
    if avg > 0 and p >= OBS_FB2_P and avg - ra >= OBS_FB2_MARGIN and conc < 0.5:
        return "CANDIDATE", txt + " → CANDIDATE (operator decides the switch; then the backtest split by turnover + the 30–50 % haircut)"
    return "KEEP_OBSERVING", txt + " → bar not met (frozen; never re-fit)"


def obs_fb2_lines(real, bud, now_ms, fetch=True, klines=None, slip=SLIP_DEF, funding=None, state_path=OBS_FB2_STATE, shadows=None):
    import json as _j
    L = [f"🩳🔒 OBS-FB2 LOW-TURNOVER SWITCH (DECISION_LOG {OBS_FB2_DL}, frozen, observe-only): the REFUSED turnover ≤ 1 setups under "
         "LOW×TODAY (+8 / −15) and LOW×TP5_SL10 (+5 / −10), funding incl., vs the traded (turnover > 1) fills at today's exit. Bar per cell "
         f"(fresh setups only, decided once at ≥ {OBS_FB2_N} on ≥ {OBS_FB2_DAYS} UTC days): CANDIDATE iff avg > 0 ∧ day-block P(mean > 0) ≥ "
         f"{OBS_FB2_P} ∧ avg − traded same-period avg ≥ +{OBS_FB2_MARGIN:g} pt ∧ no day / pair ≥ 50 % of the gain · RETIRE iff avg ≤ traded · "
         "else KEEP OBSERVING. Hold time + return per hour held shown (a shorter hold frees the slot and the coin)."]
    try:
        reg_ms = _GK.dl_deploy_ms(OBS_FB2_DL, lag_min=10, label="OBS-FB2")
        sh = load_shadows() if shadows is None else shadows
        _tmin = float(flatbase_params(_frozen_th())["turnover_min"])
        if len(sh) and "fb_turnover" in sh:   # review 2: only MEASURED turnover ≤ the floor (an unreadable-volume SHADOW is not this cohort)
            _tv = pd.to_numeric(sh.fb_turnover, errors="coerce")
            sh = sh[_tv.notna() & (_tv <= _tmin)]
        state = {}
        try:
            if state_path and os.path.exists(state_path):
                with open(state_path, encoding="utf-8") as fh:
                    state = _j.load(fh) or {}
        except Exception:
            state = {}
        for cell in OBS_FB2_CELLS:
            rows, f_unk, fresh_all = [], 0, []
            for r in sh.sort_values("t").itertuples() if len(sh) else []:
                is_fresh = reg_ms is not None and int(pd.Timestamp(r.t).value // 1_000_000) >= reg_ms
                p_ = price_shadow_cell(r, cell, bud, now_ms, fetch=fetch, klines=klines, slip_pct=slip, funding=funding)
                if is_fresh:
                    fresh_all.append((r.t, str(r.pair), None if p_ is None else p_[0], None if p_ is None else p_[1]))
                if p_ is not None:
                    rows.append((r.t, str(r.pair), p_[0], p_[1], is_fresh))
                    f_unk += 0 if p_[2] else 1
            first = []                                                       # the frozen sample = the shortest prefix (by entry time) of
            for x in fresh_all:                                              # ≥ N fresh setups on ≥ DAYS UTC days, priced or not (review 2 F1:
                first.append(x)                                              # a fixed first-30 on < 10 days would never decide)
                if len(first) >= OBS_FB2_N and len({pd.Timestamp(y[0]).date() for y in first}) >= OBS_FB2_DAYS:
                    break
            fresh = [x for x in first if x[2] is not None]
            unpriced = sum(1 for x in first if x[2] is None)
            reg = [x[:4] for x in rows if not x[4]]

            def _st(xs):
                if not xs:
                    return "N 0"
                v = [x[2] for x in xs]
                h = sorted(x[3] for x in xs)
                avg_h = sum(h) / len(h)
                return (f"N {len(v)} · WR {sum(1 for x in v if x > 0) / len(v):.0%} · avg {sum(v) / len(v):+.3f} % · hold {hhmm(avg_h)} "
                        f"(median {hhmm(h[len(h) // 2])}) · {sum(v) / len(v) / max(avg_h / 60.0, 1e-9):+.3f} %/hour held")
            if fresh:
                t0, t1 = min(x[0] for x in fresh), max(x[0] for x in fresh)
                real_same = [float(x) for o, x in zip(real.o, real.net) if t0 <= o <= t1] if len(real) else []
            else:
                real_same = []
            if cell in state:
                v, txt = state[cell]["verdict"], state[cell]["text"] + f" (frozen {state[cell]['at']})"
            else:
                v, txt = obs_fb2_verdict([(x[0], x[1], x[2]) for x in fresh], real_same, unpriced,
                                         [pd.Timestamp(x[0]).date() for x in first if x[2] is None])
                if v in ("CANDIDATE", "RETIRE", "KEEP_OBSERVING") and state_path:
                    state[cell] = dict(verdict=v, text=txt, at=pd.Timestamp.now("UTC").isoformat()[:16])
                    try:
                        with open(state_path, "w", encoding="utf-8") as fh:
                            _j.dump(state, fh, indent=1)
                    except Exception:
                        pass
            L.append(f"  {cell}: FRESH {_st(fresh)} · registration (in-sample, never counted) {_st(reg)} → {v}: {txt}"
                     + (f" · funding unknown on {f_unk} (counted 0)" if f_unk else ""))
        if reg_ms is None:
            L.append("  ⚠ the DECISION_LOG 407 commit is not in git yet — every setup is registration, nothing counted")
        if len(real):
            rh = [(pd.Timestamp(c) - pd.Timestamp(o)).total_seconds() / 60.0 for o, c in zip(real.o, real.c) if c is not None and not pd.isna(c)]
            L.append(f"  traded comparator (turnover > 1, today's exit; funding where known — paper fills carry none, the shadows are charged it, a slight tilt against the shadows): N {len(real)} · avg {real.net.mean():+.3f} % · hold "
                     f"{hhmm(sum(rh) / len(rh)) if rh else '–'}")
    except Exception as e:
        L.append(f"  OBS-FB2 unreadable ({type(e).__name__}: {str(e)[:80]})")
    return L


def run(now_ms=None, paths=None, dec_paths=None, since_ms=None, fetch=True, klines=None, funding=None, klines1m=None, th_long=None,
        tp_state_path=TP_SH_STATE, tp_klines=None, tp_final_path=TP_SH_FINAL, obs_fb2_state_path=OBS_FB2_STATE):
    L = [f"## 🩳 FLATBASE_SHORT — flat-base FRENZY short (DECISION_LOG {DL}, discipline override, ALERT-only gates)", ""]
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="FLATBASE_SHORT")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} deploy commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    a = load_orders(paths)
    c = a[(a.status.astype(str) == "CLOSED") & (a.o >= since) & (a.es == FLATBASE_SHORT)] if len(a) else a
    real, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    real = real.sort_values("o")
    L.append(f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC (deploy + 10 min) · {_GK.operator_closed_line(ox)}")
    _unk = int((~real.fund_known.astype(bool)).sum()) if len(real) else 0
    L.append("- real (net of funding): " + (f"N {len(real)} · {real.o.dt.date.nunique()} days · WR {(real.net > 0).mean():.0%} · avg {real.net.mean():+.3f} % · "
                                            f"Σ {real.net.sum():+.2f} pts · ${real.usd.sum():+.2f}" if len(real) else "N 0")
             + (f" · funding unknown on {_unk} fill(s) (paper / unread — pnl % only)" if _unk else ""))
    v, txt = flatbase_gate([(r.o.to_pydatetime(), r.net, r.close_reason) for r in real.itertuples()])
    L.append(f"Gate verdict: {v} — {txt}")
    L += resize_lines(real)
    sh = load_shadows(dec_paths)
    sh = sh[sh.t >= since] if len(sh) else sh
    from scout_long_stop4 import Budget
    bud = Budget()
    _sl = pd.to_numeric(real.entry_slippage_pct, errors="coerce").dropna() if len(real) else pd.Series(dtype=float)
    slip = float(_sl.median()) if len(_sl) >= 5 else SLIP_DEF
    priced, f_unk = [], 0
    for r in sh.itertuples():
        p_ = price_shadow(r, bud, now_ms, fetch=fetch, klines=klines, slip_pct=slip, funding=funding)
        if p_ is not None:
            priced.append((r.t, p_[0])); f_unk += 0 if p_[1] else 1
    if len(real) and priced:   # the same period as the real cohort: shadows from the first to the last real fill
        t0, t1 = real.o.min(), real.o.max()
        same = [p for t, p in priced if t0 <= t <= t1]
    else:
        same = [p for _, p in priced]
    L.append(f"- observe-only (turnover ≤ 1, no order): {len(sh)} setups · {len(priced)} priced (slippage {slip:.3f} %/taker side, funding incl."
             + (f"; unknown on {f_unk}" if f_unk else "") + ")"
             + (f" · avg {sum(p for _, p in priced) / len(priced):+.3f} %" if priced else "")
             + (f" · ⚠ fetch stopped: {bud.why}" if bud.stop else ""))
    ov, otxt = flatbase_observe_gate(list(real.net), same)
    L.append(f"Observe-cohort verdict: {ov} — {otxt}")
    L += tp_shadow_lines(real, bud, now_ms, fetch=fetch, klines=(tp_klines or klines), slip=slip, state_path=tp_state_path,
                         final_path=tp_final_path) + [""]
    L += obs_fb2_lines(real, bud, now_ms, fetch=fetch, klines=klines, slip=slip, funding=funding, shadows=sh,
                       state_path=obs_fb2_state_path) + [""]   # 🩳🔒 DL 407 (run()'s own shadow cohort)
    L += peak_age_lines(real, bud, now_ms, fetch=fetch, klines=(tp_klines or klines))
    # ── DL 348 addendum: the three observe lines (each its own section, crash-isolated; the same Budget) ──
    try:
        frame = read_decisions(dec_paths)
        ledger = load_ledger(frame)
        ctx = dict(since_ms=int(since_ms), now_ms=now_ms, bud=bud, fetch=fetch, klines=klines, funding=funding, klines1m=klines1m, slip=slip,
                   real=real, shadows=priced, frame=frame, ledger=ledger, orders=a, pts=supply_points(ledger, a), th_long=th_long)
    except Exception as e:                                         # the gate output above stays; one block says why the lines are missing
        return L + ["## 🩳①②③ FLATBASE observe lines", "", f"observe lines unavailable ({str(e)[:120]}).", ""]
    for fn, title in ((nonflat_lines, "🩳① FLATBASE observe — NON-FLAT pumps"), (missed_lines, "🩳② FLATBASE observe — MISSED setups"),
                      (blocked_lines, "🩳③ FLATBASE observe — FRENZY longs blocked by FLATBASE")):
        try:
            L += fn(ctx)
        except Exception as e:
            L += [f"## {title}", "", f"Unavailable this run ({str(e)[:120]}).", ""]
    return L


# ⏱ (DECISION_LOG 401, operator 2026-10-28: "are we considering how long ago the peak was for each trade?") — the hours from the episode
# peak bar's close to the S4 bar's close. The engine stamps it from 401 on (entry_flatbase_peak_age_h, services.flatbase.s4_scan); older fills
# are REBUILT here from public 5m klines (spike bar → the S4 bar, the first row printing the highest high — the engine's exact rule), checked
# against the stamped off-peak %. Observe line, no verdict: the buckets are fixed in advance and N < 10 per bucket is noise.
PK_AGE_BUCKETS = ((0, 6, "< 6 h"), (6, 24, "6–24 h"), (24, 48, "24–48 h"), (48, 1e9, "≥ 48 h"))
PK_AGE_OFFPK_TOL = 0.5                                         # pts: rebuilt off-peak vs the stamped one — beyond this the rebuild is flagged
PK_AGE_MAX_LAG_MS = 15 * 60_000                                # a fill > 15 min after its S4 close cannot be pinned to the bar → unscored


def peak_age_from_rows(rows, spike_ms, s4_open_ms):
    """pure: closed 5m rows [open, o, h, l, c, v] → (hours from the peak bar to the S4 bar, off-peak % at the S4 close) on the rows from the
    spike bar (open = spike_ms − 5 min) to the S4 bar; the FIRST row printing the highest high = the engine's peak bar. None when the S4 row or
    the spike bar is missing."""
    try:
        a0, s4 = int(spike_ms) - BAR_MS, int(s4_open_ms)
        w = [r for r in rows if a0 <= int(r[0]) <= s4]
        if not w or int(w[0][0]) != a0 or int(w[-1][0]) != s4:
            return None
        pk_t, pk = None, None
        for r in w:
            if pk is None or float(r[2]) > pk:
                pk, pk_t = float(r[2]), int(r[0])
        return (s4 - pk_t) / 3_600_000, ((float(w[-1][4]) / pk - 1) * 100 if pk > 0 else None)
    except (TypeError, ValueError, IndexError):
        return None


def peak_age_of(r, bud, now_ms, fetch=True, klines=None):
    """one fill → (hours, source 'stamped' / 'rebuilt' / 'rebuilt ⚠ off-peak Δ', why-unscored). The S4 bar = the 5m bar closing on the last
    4-h boundary at or before the fill (S4 always closes on the 4-h grid; opened_at is the insert time, up to the 300 s leash + the fill
    after the close — review: snapping to the 5m grid could pick the bar after S4)."""
    v = _fnum(getattr(r, "entry_flatbase_peak_age_h", None))
    if v is not None:
        return v, "stamped", None
    sp = _ms(getattr(r, "entry_frenzy_spike_at", None))
    if sp is None:
        return None, None, "no spike stamp"
    o_ms = int(pd.Timestamp(r.o).value // 1_000_000)
    s4 = (o_ms // BOUNDARY_MS) * BOUNDARY_MS - BAR_MS
    if o_ms - (s4 + BAR_MS) > PK_AGE_MAX_LAG_MS:
        return None, None, "fill > 15 min after its 4-h close"
    n = (s4 - (int(sp) - BAR_MS)) // BAR_MS + 1
    if n < 1 or n > 1500:
        return None, None, f"window {n} bars"
    if klines is not None:
        rows = klines(str(r.pair), int(sp) - BAR_MS, int(n))
    else:
        import scout_long_stop4 as _LS
        rows = _LS.fapi_klines(str(r.pair), "5m", int(sp) - BAR_MS, int(n), bud, CACHE, now=now_ms, fetch=fetch)
    if rows is None:
        return None, None, "fetch stopped / --no-fetch"
    got = peak_age_from_rows(rows, sp, s4)
    if got is None:
        return None, None, "klines miss the spike / S4 bar"
    st_op = _fnum(getattr(r, "entry_flatbase_off_peak_pct", None))
    flag = (st_op is not None and got[1] is not None and abs(got[1] - st_op) > PK_AGE_OFFPK_TOL)
    return got[0], ("rebuilt ⚠ off-peak Δ" if flag else "rebuilt"), None


def peak_age_lines(real, bud, now_ms, fetch=True, klines=None):
    """the ⏱ section on the real closed fills (net of funding) — never raises."""
    L = ["## ⏱ FLATBASE — hours since the episode peak at entry (DECISION_LOG 401, observe — no verdict)", ""]
    try:
        rows = list(real.sort_values("o").itertuples()) if len(real) else []
        def one(r):
            try:
                return peak_age_of(r, bud, now_ms, fetch=fetch, klines=klines)
            except Exception as e:                                   # one bad row is unscored, never the whole section
                return None, None, f"error {str(e)[:40]}"
        ev = [(r,) + one(r) for r in rows]
        sc = [(r, h, src) for r, h, src, _ in ev if h is not None]
        un = [(r, why) for r, h, _, why in ev if h is None]
        n_st = sum(1 for _, _, s in sc if s == "stamped")
        n_flag = sum(1 for _, _, s in sc if "⚠" in s)
        L.append(f"Coverage: {len(rows)} closed fills · scored {len(sc)} (stamped {n_st} · rebuilt from public 5m klines {len(sc) - n_st}"
                 + (f", {n_flag} with the rebuilt off-peak > ±{PK_AGE_OFFPK_TOL:g} pts from the stamp" if n_flag else "") + ")"
                 + (" · BLIND SPOT — unscored: " + ", ".join(f"{r.pair} {r.o:%m-%d %H:%M} ({w})" for r, w in un) if un else ""))
        if not sc:
            return L + [""]
        W = [h for r, h, _ in sc if r.net > 0]; Lo = [h for r, h, _ in sc if not r.net > 0]
        med = lambda x: f"{sorted(x)[len(x) // 2] if len(x) % 2 else (sorted(x)[len(x) // 2 - 1] + sorted(x)[len(x) // 2]) / 2:.1f} h" if x else "—"
        L.append(f"- winners N {len(W)} median {med(W)} · losers N {len(Lo)} median {med(Lo)}")
        parts = []
        for lo, hi, lab in PK_AGE_BUCKETS:
            g = [r.net for r, h, _ in sc if lo <= h < hi]
            parts.append(f"{lab} N {len(g)}" + (f" WR {sum(x > 0 for x in g) / len(g):.0%} avg {sum(g) / len(g):+.2f} %" if g else ""))
        L.append("- buckets (fixed in advance; N < 10 = noise): " + " · ".join(parts))
        L.append("- per fill: " + " · ".join(
            f"{r.pair} {r.o:%m-%d %H:%M} peak {h:.1f} h ago" + (" (rebuilt)" if src != "stamped" else "") + ("⚠" if "⚠" in src else "")
            + (f" / spike {float(r.entry_frenzy_hours):.0f} h" if _fnum(getattr(r, 'entry_frenzy_hours', None)) is not None else "")
            + f" → {r.net:+.2f} %" for r, h, src in sc))
        return L + [""]
    except Exception as e:
        return L + [f"Unavailable this run ({str(e)[:120]}).", ""]


def main():
    print("\n".join(run(int(time.time() * 1000), fetch=("--no-fetch" not in sys.argv))))


if __name__ == "__main__":
    main()
