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
Usage           venv/bin/python scripts/scout_flatbase_short.py [--no-fetch]
"""
import glob
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import scout_gatekit as _GK                                  # noqa: E402
import scout_willy_hold_off as _HO                           # noqa: E402  (the shared orders-export loader: dedup, CLOSED wins, paper + live)
from services.flatbase import (FLATBASE_SHORT, flatbase_gate, flatbase_observe_gate, price_path_exit, flatbase_params,   # noqa: E402
                               BAR_MS, net_of_funding_pct)

DL = 348
DEC_PATTERNS = (os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"), os.path.join(_R, "reports", "BASELINE*_decisions.csv"))
CACHE = os.path.join(_R, "reports", "backtest_cache", "scout_flatbase_short")
SLIP_DEF = 0.03                                              # % per taker side when the real fills carry too few slippage stamps
ORDER_COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "is_paper",
              "funding_fee_usd", "entry_price", "quantity", "entry_slippage_pct")
FUND_URL = "https://fapi.binance.com/fapi/v1/fundingRate?symbol={s}&startTime={a}&endTime={b}&limit=1000"
FB_COLS = ("e", "pair", "price", "fb_spike_at", "fb_status", "fb_entry_bar_at", "fb_turnover", "fb_mcap_usd", "fb_vol24_usd", "fb_flat14_pct")


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
        return pd.DataFrame(columns=list(ORDER_COLS) + ["o", "pct", "usd", "es"])
    a = pd.concat(fr, ignore_index=True)
    for c in ORDER_COLS:
        if c not in a:
            a[c] = None
    a = a.assign(_c=(a.status.astype(str) == "CLOSED").astype(int)).reset_index(drop=True)
    a = a.assign(_i=range(len(a))).sort_values(["_c", "_i"]).drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a["o"] = _HO._ts(a.opened_at).values
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


def run(now_ms=None, paths=None, dec_paths=None, since_ms=None, fetch=True, klines=None, funding=None):
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
    return L


def main():
    print("\n".join(run(int(time.time() * 1000), fetch=("--no-fetch" not in sys.argv))))


if __name__ == "__main__":
    main()
