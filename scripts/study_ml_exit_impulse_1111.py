#!/usr/bin/env python3
"""ML EXIT — IMPULSE-END theory study (2026-11-11, operator: "this is obvious that is late exit 'BTC 5m closes below its EMA20' — you are
not thinking on the theory of what we are analysing").

THEORY  A momentum long is opened on an ACCELERATING short-term impulse (EMA5 > EMA8 with the gap expanding, price > EMA20, BTC allowed by
        the regime gates). Its edge lives while the acceleration lives (winners peak ~19 min after entry). The premise is gone when the
        impulse DECELERATES — which happens before any slow trend line (5m EMA20, 1h slope) turns. So the exit candidates are FAST,
        1-minute, CLOSED-bar signals of the impulse ending, read first as an EVENT STUDY (when does each fire vs the pair's real peak; how
        often does it fire and the pair makes a new high after = false alarm), then priced through the validated replica.
SIGNALS (1m closed bars, known at close + LAT 10 s; 'against' state):
  pair : P_GAPSHRINK2 (1m EMA5−EMA8 gap fell on 2 consecutive closes) · P_C_E5 (close < 1m EMA5) · P_C_E8 (close < 1m EMA8) ·
         P_ROC3 (3-bar return ≤ 0) · P_LOWERLOW (close < the previous bar's low)
  rel  : R_ROC3 (pair 3-bar return − BTC 3-bar return ≤ 0, the coin's excess impulse gone)
  BTC  : B_ROC3 (BTC 3-bar return ≤ −0.05 %) · B_DD10 / B_DD15 (BTC close ≥ 0.10 / 0.15 % below its highest 1m close since entry) ·
         B_C_E8 (BTC close < BTC 1m EMA8)
PRICING (pre-declared, two semantics, each at arm a ∈ {0.40, 0.80} = the running peak the trade must have reached first):
  ADD<a>_<s>  today's exit UNCHANGED + exit when the signal is against us at a known bar while armed (can only exit earlier)
  REP<a>_<s>  today's exit until +a; then runner trail + ladder replaced by the signal (lock +0.10 kept), the 180-min clock only pre-arm
Cohort / paths = study_ml_exit_macro_1111 (123 kept master ML fills, today's stack). Writes reports/ML_EXIT_IMPULSE_2026-11-11_*.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study_ml_btcexit_run as BX  # noqa: E402,F401  (BTC 1m reader with the 10-26 + 11-11 top-up)
import study_ml_exit_macro_1111 as S  # noqa: E402
import scout_ml_matched_refusals as MM  # noqa: E402

OUT = os.path.join(S.ROOT, "reports", "ML_EXIT_IMPULSE_2026-11-11")
MIN, LAT = 60_000, 10_000
SIGS = ("P_GAPSHRINK2", "P_C_E5", "P_C_E8", "P_ROC3", "P_LOWERLOW", "R_ROC3", "B_ROC3", "B_DD10", "B_DD15", "B_C_E8")


def _bars(k1, o_ms, t1, pre=60):
    ot = k1.open_time.values.astype(np.int64)
    a, b = np.searchsorted(ot, o_ms - pre * MIN), np.searchsorted(ot, t1)
    d = k1.iloc[a:b]
    return d.open_time.values.astype(np.int64), d.l.values.astype(float), d.c.values.astype(float)


def signals(RP, pair, o_ms, t1):
    """→ {name: (known_ms, against bool)} on the 1m grid."""
    pk1, bk1 = RP._k1(pair), RP._k1("BTCUSDT")
    if pk1 is None or bk1 is None:
        return {}
    ot, lo, c = _bars(pk1, o_ms, t1)
    bot, _, bc = _bars(bk1, o_ms, t1)
    if len(ot) < 70 or len(bot) < 70:
        return {}
    bcs = pd.Series(bc, index=bot).reindex(ot).ffill().values
    e5 = pd.Series(c).ewm(span=5, adjust=False).mean().values
    e8 = pd.Series(c).ewm(span=8, adjust=False).mean().values
    gap = e5 - e8
    dg = np.r_[0, np.diff(gap)]
    roc = np.r_[np.full(3, np.nan), c[3:] / c[:-3] - 1] * 100
    broc = np.r_[np.full(3, np.nan), bcs[3:] / bcs[:-3] - 1] * 100
    be8 = pd.Series(bcs).ewm(span=8, adjust=False).mean().values
    kn = ot + MIN + LAT
    j0 = max(int(np.searchsorted(kn, o_ms, side="right")) - 1, 0)
    bmax = np.full(len(bcs), np.nan)
    bmax[j0:] = np.maximum.accumulate(bcs[j0:])
    dd = (1 - bcs / bmax) * 100
    out = {
        "P_GAPSHRINK2": (dg < 0) & (np.r_[0, dg[:-1]] < 0),
        "P_C_E5": c < e5, "P_C_E8": c < e8, "P_ROC3": roc <= 0,
        "P_LOWERLOW": np.r_[False, c[1:] < lo[:-1]],
        "R_ROC3": (roc - broc) <= 0, "B_ROC3": broc <= -0.05,
        "B_DD10": dd >= 0.10, "B_DD15": dd >= 0.15, "B_C_E8": bcs < be8,
    }
    return {k: (kn, np.nan_to_num(v.astype(float), nan=0).astype(bool)) for k, v in out.items()}


def variants():
    RP = S._rp()
    G = {"BASE": dict(RP.BASE)}
    for a in (0.40, 0.80):
        for s in SIGS:
            G[f"ADD{a:g}_{s}"] = dict(RP.BASE, ema13=True, _mx=s, _arm=a)
            G[f"REP{a:g}_{s}"] = dict(RP.BASE, run_arm=a, run_g_fixed=S.INF, ladder=[], noexp=1200, ema13=True, _mx=s, _arm=a, _clock_pre=a)
    return G


def price(RP, fx, G):
    P = RP.prep_path(fx, RP.HORIZON_MS)
    if P is None:
        return None, None
    t, net = P["t"], P["net"]
    pk = np.maximum.accumulate(np.maximum(net, 0.0))
    rel = t - fx["o_ms"]
    sg = signals(RP, fx["pair"], fx["o_ms"], int(t[-1]))
    if not sg:
        return None, None
    st = {}
    for s, (kn, ag) in sg.items():
        j = np.searchsorted(kn, t, side="right") - 1
        j0 = max(int(np.searchsorted(kn, fx["o_ms"], side="right")) - 1, 0)
        st[s] = (j >= j0) & ag[np.clip(j, 0, len(ag) - 1)]
    out = {}
    for name, pr in G.items():
        prm = {k: v for k, v in pr.items() if not k.startswith("_")}
        Q = P
        if "_mx" in pr:
            m = st[pr["_mx"]] & (pk >= pr["_arm"] - 0.005)
            if pr.get("_clock_pre") is not None:
                m = m | ((rel >= 180 * MIN) & (pk < pr["_clock_pre"] - 0.005))
            Q = dict(P, e13=np.where(m, np.inf, -np.inf), flip=m)
        k, why, pct = RP.evaluate(Q, fx, prm)
        if why in ("EMA13_CROSS", "RH_EMA13_CROSS"):
            why = "SIGNAL" if not (pr.get("_clock_pre") and pk[k] < pr["_clock_pre"] - 0.005) else "NO_EXPANSION"
        out[name] = (why, pct, int(t[k]))
    # ── event study: the pair's SWING peak = highest net before it gives back 0.5 (or before BASE exit + 60 min) after arming +0.40
    ev = dict(src=P["src"])
    arm = np.flatnonzero(pk >= 0.395)
    if len(arm):
        lim = np.searchsorted(t, out["BASE"][2] + 60 * MIN)
        dd = np.flatnonzero((pk - net >= 0.5) & (np.arange(len(t)) >= arm[0]))
        end = min(lim, dd[0] if len(dd) else len(t)) + 1
        ip = int(np.argmax(net[:end]))
        ev.update(armed=True, t_arm=(t[arm[0]] - fx["o_ms"]) / MIN, t_peak=(t[ip] - fx["o_ms"]) / MIN, peak=float(net[ip]))
        for s in SIGS:
            f = np.flatnonzero(st[s] & (np.arange(len(t)) >= arm[0]))
            if not len(f):
                ev[f"{s}_lag"] = np.nan
                continue
            i = f[0]
            ev[f"{s}_lag"] = (t[i] - t[ip]) / MIN                         # minutes AFTER the swing peak (negative = before)
            ev[f"{s}_net"] = float(net[i])
            ev[f"{s}_left"] = float(net[ip] - net[i])                     # what the first fire leaves vs the swing peak
            ev[f"{s}_false"] = bool(i < ip and net[ip] - net[i] >= 0.30)  # fired, then the pair went ≥ 0.30 higher
    else:
        ev["armed"] = False
    return out, ev


def main():
    C, _ = S.cohort()
    G = variants()
    rows, evs = [], []
    with MM.replica_with_topup() as RP:
        for r in C.itertuples():
            rsi = RP.btc_rsi_at(int(r.o_ms))
            fx = dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in),
                      atr=float(r.atr) if (r.atr == r.atr and r.atr > 0) else None, rsi_entry=None if rsi is None else float(rsi))
            try:
                out, ev = price(RP, fx, G)
            except Exception as e:
                print("ERR", r.key, e, flush=True)
                out, ev = None, None
            if out is None:
                evs.append(dict(key=r.key, src="none"))
                continue
            for v, (why, pct, tex) in out.items():
                rows.append(dict(key=r.key, variant=v, reason=why, pct=pct, exit_ms=tex))
            evs.append(dict(key=r.key, **ev))
    pd.DataFrame(rows).to_csv(OUT + "_variants.csv", index=False)
    pd.DataFrame(evs).to_csv(OUT + "_events.csv", index=False)
    print("fills", len(C), "priced", len({x["key"] for x in rows}))


if __name__ == "__main__":
    main()
