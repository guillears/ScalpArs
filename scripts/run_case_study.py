#!/usr/bin/env python3
"""📒 Case study — one pair's spike run traded both ways with a tight stop and a trailing exit (operator, 2026-10-02: "build the MOVR case
since Sep-30 and the SAND case since this morning, LONGs and SHORTs, with SL and trailing TP"). IN-SAMPLE illustration on the two pairs
that prompted the idea; the year tests decide whether any of it generalises.
  LONG    the staircase state turns on (≥ 2 h after the spike ∧ every 5m close of the last hour ≥ the spike-anchored VWAP ∧ last-hour volume
          ≥ 100× normal) after being off for the previous hour → buy at the next open
  SHORT   a 5m close below the EMA50 / the EMA200 with the previous 12 closes above it, after the run reached +50 % → sell at the next open
  exits   hard stop S · trailing: once A % in profit, close when price gives back T % from its best point · 12 h cap · 1-minute bars, stop first
  books   longs and shorts are separate books, one position per side at a time; cost 0.11 % per trade (funding not included)
Usage: venv/bin/python scripts/run_case_study.py PAIR "2026-09-29 20:00" [...]"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_review as B  # noqa: E402
sys.argv = _a
ST = B.ST; COST = 0.11; CONFIGS = ((2.0, 2.0, 1.5), (3.0, 3.0, 2.0), (3.0, 5.0, 3.0))


def m1_all(pair, start_ms):
    out = []; cur = start_ms
    while True:
        r = json.load(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=1m&startTime={cur}&limit=1500", timeout=25))
        if not r:
            break
        out += r; cur = int(r[-1][0]) + 1; time.sleep(0.05)
        if len(r) < 1500:
            break
    a = np.array([[float(x[0]), float(x[2]), float(x[3]), float(x[4])] for x in out[:-1]])
    return a[:, 0].astype("int64"), a[:, 1], a[:, 2], a[:, 3]


def long_triggers(d, V=100.0):
    t = d.open_time.values.astype("int64"); o, h, l, c, q = d.o.values, d.h.values, d.l.values, d.c.values, d.qvol.values; n = len(c)
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); volx = (q1h / norm).values
    r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]
    with np.errstate(invalid="ignore"):
        lead = np.nonzero((r30 >= 5) & (volx >= 20) & (q1h.values >= 2e6))[0]
    out = []; i = 0
    for on in lead:
        on = int(on)
        if on < i:
            continue
        pv = (h[on:] + l[on:] + c[on:]) / 3 * q[on:]; vw = np.cumsum(pv) / np.maximum(np.cumsum(q[on:]), 1e-12)
        above = pd.Series(c[on:] >= vw).rolling(12).min().fillna(0).values >= 1
        with np.errstate(invalid="ignore"):
            st = above & (volx[on:] >= V) & (np.arange(n - on) >= 24)
        last = on; end = n
        for j in range(on + 1, n):
            if t[j] - t[last] > 24 * 3600_000:
                end = j; break
            if st[j - on]:
                last = j
        off = 12
        for j in range(on, min(end, n - 1)):
            if st[j - on]:
                if off >= 12:
                    out.append(dict(t=int(t[j + 1]), entry=float(o[j + 1]), why="staircase ON", vwap=float(vw[j - on]), base=float(c[on - 6])))
                off = 0
            else:
                off += 1
        i = end
    return out


def walk(side, h, l, c, e, S, A, T):
    best = e
    for i in range(min(len(c), 720)):
        if side == "LONG":
            if l[i] <= e * (1 - S / 100):
                return -S, i + 1, "stop"
            if (best / e - 1) * 100 >= A and l[i] <= best * (1 - T / 100):
                return (best * (1 - T / 100) / e - 1) * 100, i + 1, "trail"
            best = max(best, h[i])
        else:
            if h[i] >= e * (1 + S / 100):
                return -S, i + 1, "stop"
            if (1 - best / e) * 100 >= A and h[i] >= best * (1 + T / 100):
                return (1 - best * (1 + T / 100) / e) * 100, i + 1, "trail"
            best = min(best, l[i])
    k = min(len(c), 720) - 1
    return ((c[k] / e - 1) if side == "LONG" else (1 - c[k] / e)) * 100, k + 1, "open" if len(c) < 720 else "12 h cap"


if __name__ == "__main__":
    args = sys.argv[1:]; L = ["# 📒 Case study — MOVR and SAND traded both ways with a tight stop and a trailing exit (in-sample)", ""]
    for pair, start in zip(args[0::2], args[1::2]):
        s_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000); d = ST.api5m(pair)
        T1, H1, L1, C1 = m1_all(pair, s_ms)
        sig = [dict(side="LONG", **x) for x in long_triggers(d) if x["t"] >= s_ms]
        for span in (50, 200):
            sig += [dict(side="SHORT", t=x["t"], entry=x["entry"], why=f"close below EMA{span}") for x in B.triggers(pair, d, span, min_q24=0) if x["t"] >= s_ms]
        sig.sort(key=lambda x: x["t"])
        L += [f"## {pair} since {start} UTC — price {C1[0]:.5g} → high {H1.max():.5g} → now {C1[-1]:.5g}", ""]
        for S, A, T in CONFIGS:
            free = {"LONG": 0, "SHORT": 0}; rows = []
            for x in sig:
                if x["t"] < free[x["side"]]:
                    continue
                i = int(np.searchsorted(T1, x["t"]))
                if i >= len(C1) - 2:
                    continue
                r, mins, how = walk(x["side"], H1[i:], L1[i:], C1[i:], x["entry"], S, A, T); free[x["side"]] = x["t"] + mins * 60_000
                rows.append(dict(t=x["t"], side=x["side"], why=x["why"], entry=x["entry"], net=r - COST, mins=mins, how=how))
            R = pd.DataFrame(rows)
            L += [f"### Stop {S:g} % · trail starts at +{A:g} %, gives back {T:g} %", "", "| Time (UTC) | Side | Signal | Entry | Result | Exit | Held |", "|---|---|---|---|---|---|---|"]
            L += [f"| {pd.Timestamp(r.t, unit='ms'):%m-%d %H:%M} | {r.side} | {r.why} | {r.entry:.5g} | {r.net:+.2f}% | {r.how} | {r.mins} min |" for r in R.itertuples()]
            for side in ("LONG", "SHORT"):
                g = R[R.side == side]
                L.append(f"\n**{side}: {len(g)} trades · {int((g.net > 0).sum())} won · total {g.net.sum():+.1f} % on price = {g.net.sum() * 20:+.0f} % of margin at 20× · stops {int((g.how == 'stop').sum())}**" if len(g) else f"\n**{side}: no trade**")
            L += [f"\n**Both sides: {len(R)} trades · {R.net.sum():+.1f} % on price = {R.net.sum() * 20:+.0f} % of margin at 20×**", ""]
    out = os.path.join(ST.ROOT, "reports", "RUN_CASE_STUDY_MOVR_SAND_2026-10-02.md"); open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
