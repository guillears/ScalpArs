#!/usr/bin/env python3
"""ML BTC-REACTIVE EXIT study (2026-10-26) — per-fill engine.

Hypothesis (operator brief): an ML long whose BTC turns down in the first minutes after entry becomes a loser; exiting when BTC reacts
(earlier than the pair's own stop) saves more on losers than it costs on winners.

Cohort = the ML EXIT DEEP cohort unchanged (scripts/study_ml_exit_b25_data.cohort: 158 KEEP master + 9 B25 provisional + 166 PROBE,
outcomes at 1×). Pair paths + exits = the VALIDATED replica of that study (scripts/study_ml_exit_b25_run.evaluate, BASE = today's stack,
100 % exit-reason match on master tick fills). BTC path = BTCUSDT built by the SAME path builder (real ticks from
reports/backtest_cache/ticks_q/BTCUSDT where the day archive exists, else fully CLOSED 1m bars from backtest_cache/k1m + this study's
top-up reports/ML_BTC_EXIT_2026-10-26_fetch/k1m) — source recorded per fill.

PRE-DECLARED GRID (written 2026-10-26 before any variant was read):
  X   exit at minute N if BTC return since entry ≤ −x                       N ∈ {3,5,10,15} × x ∈ {0.05,0.10,0.15,0.20,0.30}  (20)
  XG  … AND pair not yet green (running net peak < +0.10 by N)               (20)
  XN  … AND pair net P&L < 0 at N                                            (20)
  XGN … AND both                                                             (20)
  TR  exit when BTC falls x from its post-entry high (incl. the entry price) within the first 30 min, x ∈ {0.10,0.15,0.20,0.30} (4)
  TRG … AND pair not yet green at the trigger                               (4)
  TS  tighten instead of exit: BTC ≤ −x at N → stop moved to L for the rest of the trade (unarmed fills; armed fills keep the runner
      floor ≥ +0.10), L ∈ {−0.4, 0.0 (net break-even)}                     (40)
  K = 128 variants (Bonferroni K for the primary latency scenario).
Latency scenarios (all three priced; PRIMARY = FAST, pre-declared):
  IDEAL  exit at the trigger time, no slip (upper bound).
  FAST   BTC ticks → exit at trigger + 8 s; BTC 1m → exit at the close of the 1m bar containing the trigger ((T//60 s + 1)·60 s);
         slip 0.02.
  SLOW   exit ≥ 60 s after the trigger: BTC ticks → T + 60 s, then the next pair print at or after it; BTC 1m → the next 1m close at
         or after T + 60 s; slip 0.10.
  Exit price: tick pair path → last print ≤ exit time (IDEAL / FAST), first print ≥ it (SLOW); 1m pair path → the close of the first
  bar closing at or after the exit time.
  Slip is charged only on fills whose exit the rule changes (BASE exits carry no extra slip — conservative against the rule).
A minute-N check sees BTC at o+N: ticks → last print ≤ o+N vs the first print ≥ the fill time; 1m → the close of the last bar CLOSED
≤ o+N vs the close of the last bar closed ≤ the fill time (fully closed bars only — no intrabar look-ahead). The pair's own net P&L /
running peak at N follow the same rule on 1m pair paths (closes + highs of bars opened after the fill); ticks unchanged. The BTC trail
(TR) runs on the same series (ticks, or 1m closes). The tighten variants (TS) keep the replica's stop evaluation on the full path.

Usage:  venv/bin/python -W ignore scripts/study_ml_btcexit_run.py [--jobs 10] [--yr5]
Outputs: reports/ML_BTC_EXIT_2026-10-26_fills.csv (per fill: BTC + pair path at 1/3/5/10/15/30 min, BASE exit),
         reports/ML_BTC_EXIT_2026-10-26_variants.csv.gz (fill × variant × scenario); with --yr5: _yr5_fills.csv + _yr5_variants.csv.gz.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_exit_b25_run as ER  # noqa: E402  (imports study_ml_exit_b25_data → patches the replica readers)
DD = ER.DD
RP = ER.RP

OUT = os.path.join(DD.ROOT, "reports", "ML_BTC_EXIT_2026-10-26")
BTC_FETCH = os.path.join(DD.ROOT, "reports", "ML_BTC_EXIT_2026-10-26_fetch", "k1m", "BTCUSDT.csv")
NS = (1, 3, 5, 10, 15, 30)
GRID_N = (3, 5, 10, 15)
GRID_X = (0.05, 0.10, 0.15, 0.20, 0.30)
TR_X = (0.10, 0.15, 0.20, 0.30)
TS_L = (-0.4, 0.0)
GREEN = 0.10
SCEN = {"IDEAL": (0, 0.0), "FAST": (8_000, 0.02), "SLOW": (None, 0.10)}

_prev_k1 = RP._k1


_BTC1 = None


def _k1_btc(pair):
    """the study readers + this study's BTC 1m top-up (BTC kept in its own cache: RP._K1 evicts after 6 pairs)."""
    global _BTC1
    if pair != "BTCUSDT":
        return _prev_k1(pair)
    if _BTC1 is None:
        df = _prev_k1(pair)
        RP._K1.pop("BTCUSDT", None)
        if os.path.exists(BTC_FETCH):
            f = pd.read_csv(BTC_FETCH)
            df = f if df is None else pd.concat([df, f[~f.open_time.isin(df.open_time)]])
        _BTC1 = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return _BTC1


RP._k1 = _k1_btc


def variants():
    out = {}
    for N in GRID_N:
        for x in GRID_X:
            for tag, g, neg in (("X", False, False), ("XG", True, False), ("XN", False, True), ("XGN", True, True)):
                out[f"{tag}_N{N}_x{x:.2f}"] = dict(kind="X", N=N, x=x, green=g, neg=neg)
            for L in TS_L:
                out[f"TS_N{N}_x{x:.2f}_L{L:+.1f}"] = dict(kind="TS", N=N, x=x, L=L)
    for x in TR_X:
        out[f"TR_x{x:.2f}"] = dict(kind="TR", x=x, green=False)
        out[f"TRG_x{x:.2f}"] = dict(kind="TR", x=x, green=True)
    return out


VARIANTS = variants()


def _closed_1m(k1, o_ms, t1):
    """fully CLOSED 1m bars after the fill: (close time = open + 60 s, close, high of bars opened ≥ o) for close time in (o, t1];
    plus the reference close = the last bar CLOSED at or before o (no intrabar look-ahead)."""
    ot = k1.open_time.values.astype(np.int64)
    ct = ot + 60_000
    jr = int(np.searchsorted(ct, o_ms, side="right")) - 1
    a, b = jr + 1, int(np.searchsorted(ct, t1, side="right"))
    if jr < 0 or b <= a:
        return None
    c = k1.c.values[a:b].astype(float)
    h = np.where(ot[a:b] >= o_ms, k1.h.values[a:b].astype(float), c)
    return ct[a:b], c, h, float(k1.c.values[jr])


def btc_path(o_ms):
    """BTC return since entry (%) as a step series (t, r): ticks where the day archive exists (ref = first print ≥ entry), else
    CLOSED 1m bars only (ref = close of the last bar closed ≤ entry, values at each bar close). A (o, 0) point is prepended."""
    bp = RP.build_path("BTCUSDT", o_ms - 5 * 60_000, 41 * 60_000)
    if bp is None:
        return None
    t, p, src = bp
    if src == "tick":
        i0 = int(np.searchsorted(t, o_ms, side="left"))
        if i0 >= len(t) or t[-1] < o_ms + 30 * 60_000:
            return None
        ref = float(p[i0]); tt, pp = t[i0:], p[i0:]
    else:
        cb = _closed_1m(RP._k1("BTCUSDT"), o_ms, o_ms + 41 * 60_000)
        if cb is None or cb[0][-1] < o_ms + 30 * 60_000:
            return None
        tt, pp, _, ref = cb
        src = "1m"
    r = (pp / ref - 1.0) * 100.0
    return dict(t=np.r_[o_ms, tt], r=np.r_[0.0, r], src=src)


def pair_state(fx, P):
    """(t, net, pk) the bot could know: the tick path as is; on 1m pair paths CLOSED bars only (net at each close, running peak from
    closes and the highs of bars opened after the fill), with a (o, net at the reference close) point prepended."""
    if P["src"] == "tick":
        return P["t"], P["net"], np.maximum.accumulate(np.maximum(P["net"], 0.0))
    o = fx["o_ms"]
    cb = _closed_1m(RP._k1(fx["pair"]), o, o + 9 * 3600_000)
    E = fx["E"]
    f = lambda p: (p / E - 1.0) * 100.0 - fx["fee_in"] - RP.TAKER * p / E
    if cb is None:
        return P["t"], P["net"], np.maximum.accumulate(np.maximum(P["net"], 0.0))
    ct, c, h, rc = cb
    net = np.r_[f(rc), f(c)]
    pk = np.maximum.accumulate(np.maximum(np.r_[0.0, np.maximum(f(c), f(h))], 0.0))
    return np.r_[o, ct], net, pk


def _at(t, v, tm):
    j = int(np.searchsorted(t, tm, side="right")) - 1
    return float(v[max(j, 0)])


def exit_price(P, Ps, Te, scen):
    """net P&L of a market exit at Te. Tick pair path: last print ≤ Te (IDEAL / FAST), first print ≥ Te (SLOW). 1m pair path: the
    close of the first bar closing at or after Te."""
    if P["src"] == "tick":
        t, net = P["t"], P["net"]
        if scen == "SLOW":
            j = min(int(np.searchsorted(t, Te, side="left")), len(t) - 1)
            return float(net[j])
        return _at(t, net, Te)
    ct, net = Ps[0], Ps[1]
    j = min(int(np.searchsorted(ct, Te, side="left")), len(ct) - 1)
    return float(net[j])


def base_eval(fx):
    for hz in (2 * 3600_000, 6 * 3600_000, ER.HZ):
        P = RP.prep_path(fx, hz)
        if P is None:
            return None, None
        k, why, pct = ER.evaluate(P, fx, ER.V())
        if why != "END_OF_DATA" or hz == ER.HZ:
            return P, (k, why, pct, int(P["t"][k]))
    return None, None


def exit_time(T, scen, btc_src):
    """T = trigger time (ms) → the time the exit order is assumed to fill (see module docstring)."""
    if scen == "IDEAL":
        return T
    if scen == "FAST":
        return T + SCEN["FAST"][0] if btc_src == "tick" else ((T // 60_000) + 1) * 60_000
    # SLOW: at least 60 s after the trigger, then the next print (ticks) / the next 1m close at or after T + 60 s (1m)
    T1 = T + 60_000
    return T1 if btc_src == "tick" else -(-T1 // 60_000) * 60_000


def run_fill(fx):
    P, base = base_eval(fx)
    B = btc_path(fx["o_ms"])
    row = dict(key=fx["key"], pair_src=None if P is None else P["src"], btc_src=None if B is None else B["src"])
    if P is None or base is None:
        return row, []
    k0, why0, pct0, t0 = base
    row.update(base_reason=why0, base_pct=pct0, base_exit_min=(t0 - fx["o_ms"]) / 60_000)
    o = fx["o_ms"]
    Ps = pair_state(fx, P)
    t, net, pk = Ps
    for N in NS:
        tm = o + N * 60_000
        row[f"pair_net{N}"] = _at(t, net, tm)
        row[f"pair_pk{N}"] = _at(t, pk, tm)
        row[f"open{N}"] = t0 > tm
    if B is None:
        return row, []
    bt, br = B["t"], B["r"]
    for N in NS:
        tm = o + N * 60_000
        m = bt <= tm
        row[f"btc{N}"] = _at(bt, br, tm)
        row[f"btc_min{N}"] = float(br[m].min()) if m.any() else np.nan
        row[f"btc_max{N}"] = float(br[m].max()) if m.any() else np.nan
    w30 = bt <= o + 30 * 60_000
    bhi = np.maximum.accumulate(np.maximum(br[w30], 0.0))
    dd = bhi - br[w30]
    out = []
    for name, v in VARIANTS.items():
        trig = None
        if v["kind"] in ("X", "TS"):
            tm = o + v["N"] * 60_000
            if t0 > tm and row[f"btc{v['N']}"] <= -v["x"] + 1e-12:
                ok = True
                if v.get("green") and row[f"pair_pk{v['N']}"] >= GREEN:
                    ok = False
                if v.get("neg") and row[f"pair_net{v['N']}"] >= 0:
                    ok = False
                if ok:
                    trig = tm
        else:  # TR
            idx = np.flatnonzero(dd >= v["x"] - 1e-12)
            for j in idx:
                Tj = int(bt[w30][j])
                if Tj >= t0:
                    break
                if v["green"] and _at(t, pk, Tj) >= GREEN:
                    continue
                trig = Tj
                break
        for sc, (_, slip) in SCEN.items():
            if trig is None:
                out.append((fx["key"], name, sc, 0, why0, pct0, pct0, None))
                continue
            Te = exit_time(trig, sc, B["src"])
            if v["kind"] == "TS":
                pr = ER.V(td=((Te - o) / 60_000, (Te - o) / 60_000, v["L"]))
                k, why, pct = ER.evaluate(P, fx, pr)
                if why == "END_OF_DATA":
                    P2 = RP.prep_path(fx, ER.HZ)
                    k, why, pct = ER.evaluate(P2, fx, pr)
                    tex = int(P2["t"][k])
                else:
                    tex = int(P["t"][k])
                changed = abs(pct - pct0) > 1e-9 or why != why0
                pct_v = pct - (slip if changed else 0.0)
                out.append((fx["key"], name, sc, int(changed), why, pct0, pct_v, (tex - o) / 60_000))
            else:
                if Te >= t0:
                    out.append((fx["key"], name, sc, 0, why0, pct0, pct0, None))
                else:
                    pv = exit_price(P, Ps, Te, sc) - slip
                    out.append((fx["key"], name, sc, 1, "BTC_EXIT", pct0, pv, (Te - o) / 60_000))
    return row, out


def _work(batch):
    R, O = [], []
    for fx in batch:
        try:
            r, o = run_fill(fx)
        except Exception as ex:
            r, o = dict(key=fx["key"], err=str(ex)), []
        R.append(r); O.extend(o)
    return R, O


def _pool(fxs, jobs):
    fxs = sorted(fxs, key=lambda f: (f["o_ms"]))
    n = max(1, len(fxs) // (jobs * 6))
    batches = [fxs[i:i + n] for i in range(0, len(fxs), n)]
    R, O = [], []
    with ProcessPoolExecutor(jobs) as ex:
        for a, b in ex.map(_work, batches):
            R.extend(a); O.extend(b)
    V = pd.DataFrame(O, columns=["key", "variant", "scen", "changed", "reason", "base_pct", "pct", "exit_min"])
    return pd.DataFrame(R), V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--yr5", action="store_true")
    a = ap.parse_args()
    if a.yr5:
        Y = RP.load_yr()
        Y = Y[~Y.cb]
        fxs = RP.fills_of(Y)
        R, V = _pool(fxs, a.jobs)
        R.to_csv(OUT + "_yr5_fills.csv", index=False)
        V.to_csv(OUT + "_yr5_variants.csv.gz", index=False)
        print(f"yr5 {len(fxs)} fills; with BTC path {R.btc_src.notna().sum()}; rows {len(V)}")
        return
    C = DD.cohort()
    fxs = [DD.fill_dict(r) for r in C.itertuples()]
    R, V = _pool(fxs, a.jobs)
    R.to_csv(OUT + "_fills.csv", index=False)
    V.to_csv(OUT + "_variants.csv.gz", index=False)
    print(f"{len(C)} fills; BTC src {R.btc_src.value_counts(dropna=False).to_dict()}; pair src {R.pair_src.value_counts(dropna=False).to_dict()}; "
          f"errors {R.get('err', pd.Series(dtype=str)).notna().sum()}; rows {len(V)}")


if __name__ == "__main__":
    main()
