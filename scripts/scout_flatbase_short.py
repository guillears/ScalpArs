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
                               BAR_MS, net_of_funding_pct, s4_scan, ST_NOT_FLAT, ST_MISSED, ST_REFUSED, ST_UNREADABLE,
                               ACT_ENTER, ACT_MISSED, ACT_EXPIRED, ACT_WAIT)

DL = 348
DEC_PATTERNS = (os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"), os.path.join(_R, "reports", "BASELINE*_decisions.csv"))
CACHE = os.path.join(_R, "reports", "backtest_cache", "scout_flatbase_short")
SLIP_DEF = 0.03                                              # % per taker side when the real fills carry too few slippage stamps
ORDER_COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "is_paper",
              "funding_fee_usd", "entry_price", "quantity", "entry_slippage_pct", "entry_mcap_usd")
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


def run(now_ms=None, paths=None, dec_paths=None, since_ms=None, fetch=True, klines=None, funding=None, klines1m=None, th_long=None):
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
    L += [f"Observe-cohort verdict: {ov} — {otxt}", ""]
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


def main():
    print("\n".join(run(int(time.time() * 1000), fetch=("--no-fetch" not in sys.argv))))


if __name__ == "__main__":
    main()
