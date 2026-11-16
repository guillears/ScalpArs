#!/usr/bin/env python3
"""🎯 FAN flip TP counterfactual (DECISION_LOG 563 → 568 master re-freeze) — every master FLIP:FAN_RATIO_GATE row (kept or not, every era)
re-walked on its own 1m path at TODAY's live FAN exit: a resting maker TP at NET +0.5 % (books exactly +0.5), today's stop
−max(0.70, min(1.5 × ATR %, 1.20)), no trail / no ladder, 180-min no-expansion close (books the reading there). The walker is the one validated
in reports/FAN_FLIP_EXIT_STUDY_2026-11-16.md (scripts/flip_exit_replica.py fees / stop; 1m bars read O → H → L → C, the adverse high first for a
short, so a stop wins every tie). Also writes the replica of the OLD exit (FX.simulate) for the agreement check.

RULES (never credit a stopped trade with a TP the 1m bars cannot confirm; never charge a 1m gap print as a stop fill):
  · BOOKED_NO_TOUCH — (568 review) the TP walk books +TP but the live fill's recorded peak_pnl stayed below +TP → booked pct (a resting
    maker TP may catch a wick polling missed, but the master never credits an unconfirmed touch).
  · BOOKED_STOP_KEPT — the live fill was STOPPED and either the replica of the old exit misses the booked result by > 0.25 pt (1m bars missed
    the wick that stopped it, e.g. 1000PEPE 11-06) or the TP walk also ends on the stop (the same stop → its REAL fill): booked pct.
  · SL_AT_LINE — the TP walk stops a fill the live trail closed earlier: max(walk, stop line − 0.05) (the line, not a 1m gap print).
  · TP05_WALK — every other row: the walk (TP +0.5 / 180-min close).
No path → no row (the builder keeps it as traded and prints it).

Output reports/FAN_FLIP_TP05_CF.csv: opened_at, pair, era, booked, rep, rep_why, tp05, tp05_why, livestop, cf_pct, cf_rule, exit_rule.
Klines cache reports/.fan_tp_cf_1m/{pair}_{open_ms}.csv (721 × 1m from the fill minute). Binance public klines, weight 2 / call, ≤ 1,200 / min.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import flip_exit_replica as FX  # noqa: E402

ES = "FLIP:FAN_RATIO_GATE"
EXIT_RULE = "TP+0.5net/rtp/sl=-max(0.70,min(1.5ATR,1.20))/180m/no-trail"   # = build_master_pool.FAN_EXIT_FROZEN
TP, HORIZON_MIN, MISMATCH = 0.5, 180, 0.25
SL_SLIP = 0.05   # 1m bars vs ticks (the study's measured walker pessimism); a stop books its line − this
CACHE = os.path.join(ROOT, "reports", ".fan_tp_cf_1m")
OUT = os.path.join(ROOT, "reports", "FAN_FLIP_TP05_CF.csv")


def _klines(pair, o_ms):
    fn = os.path.join(CACHE, f"{pair}_{o_ms}.csv")
    if os.path.exists(fn):
        return pd.read_csv(fn)
    st = (o_ms // 60000) * 60000
    r = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol=pair, interval="1m", startTime=st, limit=721), timeout=20)
    if r.status_code in (418, 429):
        sys.exit(f"Binance rate limit {r.status_code} — stop, re-run later")
    if int(r.headers.get("x-mbx-used-weight-1m", "0") or 0) > 900:
        time.sleep(30)
    js = r.json()
    time.sleep(0.3)
    if not isinstance(js, list) or not js:
        return None
    d = pd.DataFrame(js).iloc[:, :5].astype(float)
    d.columns = ["t", "o", "h", "l", "c"]
    os.makedirs(CACHE, exist_ok=True)
    d.to_csv(fn, index=False)
    return d


def path(d):
    t, p = [], []
    for j, b in enumerate(d.itertuples()):
        seq = (b.c,) if j == 0 else (b.o, b.h, b.l, b.c)   # (568 review) the entry minute: only its close — its high / low may predate the fill
        for k, x in enumerate(seq):
            t.append(int(b.t) + k); p.append(x)
    return np.array(t, np.int64), np.array(p, float)


def walk_tp(t, p, E, atr, fee, tp=TP, horizon_min=HORIZON_MIN):
    """today's FAN exit: stop (FX.sl_pct), fixed TP booked AT the line, 180-min close at the reading there; no trail, no ladder."""
    if not len(t):
        return None, "NO_PATH"
    end = t[0] + horizon_min * 60000
    sl = FX.sl_pct(atr)
    pnl = ((E - p) / E - fee - FX.TAKER * p / E) * 100
    for i in range(len(p)):
        if t[i] > end:
            return float(pnl[i - 1]), "TIME"
        x = pnl[i]
        if x <= sl:
            return float(x), "SL"
        if x >= tp:
            return float(tp), "TP"
    return float(pnl[-1]), "END"


def main():
    m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    f = m[m.entry_strategy.astype(str).eq(ES)].copy()
    f["o_ms"] = pd.to_datetime(f.opened_at.astype(str).str[:19].str.replace("T", " ")).values.astype("datetime64[ms]").astype("int64")
    rows, miss = [], []
    for r in f.itertuples():
        d = _klines(r.pair, int(r.o_ms))
        if d is None or not len(d):
            miss.append(f"{r.pair} {str(r.opened_at)[:16]}"); continue
        t, p = path(d)
        nv, efee = pd.to_numeric(r.notional_value, errors="coerce"), pd.to_numeric(r.entry_fee, errors="coerce")
        fee = float(np.clip(efee / nv, 0.00015, 0.0005)) if (nv and nv > 0 and pd.notna(efee)) else FX.TAKER
        E, atr, booked = float(r.entry_price), pd.to_numeric(r.entry_atr_pct, errors="coerce"), float(r.pnl_percentage)
        rep = FX.simulate(t, p, E, atr, entry_fee_rate=fee)
        tp5, why = walk_tp(t, p, E, atr, fee)
        livestop = "STOP" in str(r.close_reason).upper()
        rep_pnl = rep["pnl"] if rep else np.nan
        mism = (not np.isfinite(rep_pnl)) or abs(rep_pnl - booked) > MISMATCH
        if livestop and (mism or why == "SL"):
            cf, rule = booked, "BOOKED_STOP_KEPT"            # the same stop really fired live: its real fill, never a 1m-bar print
        elif why == "SL":
            cf, rule = max(tp5, FX.sl_pct(atr) - SL_SLIP), "SL_AT_LINE"   # a stop the live trail never reached: the line − slip, not a 1m gap print
        else:
            _pk = pd.to_numeric(getattr(r, 'peak_pnl', None), errors="coerce")
            if why == "TP" and (not np.isfinite(_pk) or _pk < TP):
                cf, rule = booked, "BOOKED_NO_TOUCH"   # (568 review) the live fill never recorded +TP → never credit a TP the trade did not confirm
            else:
                cf, rule = tp5, "TP05_WALK"
        rows.append(dict(opened_at=str(r.opened_at)[:19], pair=r.pair, era=r.era, stack_keep=r.stack_keep, booked=round(booked, 6),
                         rep=round(rep_pnl, 6) if np.isfinite(rep_pnl) else None, rep_why=(rep or {}).get("reason"),
                         tp05=round(tp5, 6), tp05_why=why, livestop=livestop, mismatch=bool(mism), cf_pct=round(cf, 6), cf_rule=rule,
                         exit_rule=EXIT_RULE))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    k = out[out.stack_keep.astype(str) == "True"]
    print(f"{len(out)} FAN rows walked ({len(k)} kept) · no path: {len(miss)} {miss[:5]} · BOOKED_STOP_KEPT: "
          f"{(out.cf_rule == 'BOOKED_STOP_KEPT').sum()} · replica mismatch > {MISMATCH}: {out.mismatch.sum()}")
    print(f"kept: booked {(k.booked > 0).mean():.0%} · {k.booked.mean():+.3f}  →  cf {(k.cf_pct > 0).mean():.0%} · {k.cf_pct.mean():+.3f}")


if __name__ == "__main__":
    main()
