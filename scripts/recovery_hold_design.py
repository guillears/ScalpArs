#!/usr/bin/env python3
"""🩹 RECOVERY HOLD — full exit design for the operator's "we exited wrong" case (2026-10-02), read on the LIVE master batch only.
TRIGGER   a momentum LONG reaches its stop  ∧  BTC RSI(14, closed 5m) ≥ its value at entry  ∧  60 ≤ that RSI ≤ 66
          (band chosen AFTER seeing the master stops → every number here is in-sample; 13 qualifying stops)
ON TRIGGER the position is NOT closed; it enters recovery mode:
  hard stop     0.5 % below the original stop price (the most the hold can add to the loss)
  premise exit  at any 5m close where BTC RSI < 60 or < its entry value → close at market (the reason to hold is gone)
  time exit     30 min after the trigger, if the trade is still below entry → close at market
  V1 BREAKEVEN  close when price is back at the entry price
  V2 PREMISE    V1 + the premise exit
  V3 RESUME     hard stop + premise exit + time exit; no break-even close — once back above entry the normal runner logic resumes:
                arm at +0.40 %, then floor = max(peak − 1×ATR, +0.10 %); 240 min cap
1m futures bars from the minute after the stop; inside a bar the hard stop is checked first. Δ = result − the actual stop (gross; same single exit fee).
Usage: venv/bin/python scripts/recovery_hold_design.py <today's orders csv>"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BC = os.path.join(ROOT, "reports", "backtest_cache"); C1 = os.path.join(BC, "k1m_stop240")
OUT = os.path.join(ROOT, "reports", "RECOVERY_HOLD_DESIGN_2026-10-02.md"); LO, HI, ROOM = 60.0, 66.0, 0.5


def get(u):
    for k in range(4):
        try:
            return json.load(urllib.request.urlopen(u, timeout=20))
        except Exception:
            time.sleep(2 + 2 * k)
    return []


def path(pair, ms):
    f = os.path.join(C1, f"{pair}_{ms}.json")
    if not os.path.exists(f):
        r = get(f"https://fapi.binance.com/fapi/v1/klines?symbol={urllib.request.quote(pair)}&interval=1m&startTime={ms}&limit=241")
        os.makedirs(C1, exist_ok=True); json.dump([[int(x[0]), float(x[2]), float(x[3]), float(x[4])] for x in r], open(f, "w")); time.sleep(0.05)
    return json.load(open(f))


def walk(bars, entry, stop_px, atr, rin, rsi_at, variant):
    hard = stop_px * (1 - ROOM / 100); peak = -9.0; t0 = bars[0][0]
    for t, h, l, c in bars:
        if l <= hard:
            return (hard / entry - 1) * 100, "hard stop", (t - t0) // 60000 + 1
        ph, pc = (h / entry - 1) * 100, (c / entry - 1) * 100
        if variant in ("V1", "V2") and ph >= 0:
            return 0.0, "break-even", (t - t0) // 60000 + 1
        if variant == "V3":
            peak = max(peak, ph)
            if peak >= 0.40 and pc <= max(peak - atr, 0.10):
                return max(peak - atr, 0.10), "runner trail", (t - t0) // 60000 + 1
        if variant != "V1" and (t + 60000) % 300000 == 0:                # a 5m bar just closed
            r = rsi_at(t + 60000)
            if (r < LO or r < rin) and pc < 0.40:
                return pc, "premise gone", (t - t0) // 60000 + 1
        if (t - t0) >= 29 * 60000 and pc < 0 and (variant != "V3" or peak < 0.40):
            return pc, "time", (t - t0) // 60000 + 1
        if variant != "V3" and (t - t0) >= 29 * 60000:
            return pc, "time", 30
    return (bars[-1][3] / entry - 1) * 100, "cap", len(bars)


if __name__ == "__main__":
    new = pd.read_csv(sys.argv[1]); new = new[(new.entry_strategy == "MOMENTUM") & (new.direction == "LONG")].assign(era="B16")
    m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    A = pd.concat([m[(m.entry_strategy == "MOMENTUM") & (m.direction == "LONG")], new], ignore_index=True).drop_duplicates(["opened_at", "pair"])
    b = pd.read_csv(os.path.join(BC, "btc_5m.csv")); b = b.rename(columns={b.columns[0]: "t"})[["t", "c"]]; rows = []; cur = int(b.t.max()) + 1
    while True:
        r = get(f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&startTime={cur}&limit=1500")
        if not r:
            break
        rows += [(int(x[0]), float(x[4])) for x in r]; cur = int(r[-1][0]) + 1
        if len(r) < 1500:
            break
    b = pd.concat([b, pd.DataFrame(rows, columns=["t", "c"])]).astype(float).drop_duplicates("t").sort_values("t"); b = b[b.t < b.t.max()]
    d = b.c.diff(); u = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); v = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    TT, RR = b.t.values.astype("int64"), (100 - 100 / (1 + u / v)).values
    rsi_ms = lambda ms: RR[np.searchsorted(TT, ms - 300_000, side="right") - 1]                  # last 5m bar CLOSED at/before ms
    rsi = lambda ts: rsi_ms(int(pd.Timestamp(ts).timestamp() * 1000))
    s = A[A.close_reason.astype(str).str.contains("STOP_LOSS") & ~A.close_reason.astype(str).str.contains("BR_|FLIP")].copy()
    s["rin"] = [rsi(x) for x in s.opened_at]; s["rout"] = [rsi(x) for x in s.closed_at]
    q = s[(s.rout >= s.rin) & (s.rout >= LO) & (s.rout <= HI)].sort_values("opened_at").copy(); res = []
    for r in q.itertuples():
        bars = path(r.pair, (int(pd.Timestamp(r.closed_at).timestamp() * 1000) // 60000 + 1) * 60000); act = (r.exit_price / r.entry_price - 1) * 100
        row = dict(era=r.era, opened=str(r.opened_at)[:16], pair=r.pair.replace("USDT", ""), rsi=f"{r.rin:.0f}→{r.rout:.0f}", stop=round(act, 2), notional=r.investment * r.leverage)
        for V in ("V1", "V2", "V3"):
            x, why, mn = walk(bars, r.entry_price, r.exit_price, float(r.entry_atr_pct), r.rin, rsi_ms, V)
            row[V] = round(x - act, 2); row[V + "_how"] = f"{why} {int(mn)}m"; row[V + "_usd"] = (x - act) / 100 * row["notional"]
        res.append(row)
    R = pd.DataFrame(res); pd.set_option("display.width", 250)
    L = ["# 🩹 RECOVERY HOLD — exit design on the live master batch (in-sample, 13 stops)", "", __doc__.split("Usage")[0].strip(), "",
         "| Batch | Opened | Pair | BTC RSI entry→stop | Stop | V1 Δ (how) | V2 Δ (how) | V3 Δ (how) |", "|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.era} | {r.opened} | {r.pair} | {r.rsi} | {r.stop:+.2f}% | {r.V1:+.2f} ({r.V1_how}) | {r.V2:+.2f} ({r.V2_how}) | {r.V3:+.2f} ({r.V3_how}) |")
    L += ["", "| Batch | qualifying stops | of all ML stops in the batch | V1 Δ % / $ | V2 Δ % / $ | V3 Δ % / $ |", "|---|---|---|---|---|---|"]
    tot = s.groupby("era").size()
    for e, g in R.groupby("era", sort=False):
        L.append(f"| {e} | {len(g)} | {int(tot.get(e, 0))} | {g.V1.sum():+.2f} / {g.V1_usd.sum():+.0f} | {g.V2.sum():+.2f} / {g.V2_usd.sum():+.0f} | {g.V3.sum():+.2f} / {g.V3_usd.sum():+.0f} |")
    L.append(f"| **TOTAL** | {len(R)} | {len(s)} | **{R.V1.sum():+.2f} / {R.V1_usd.sum():+.0f}** | **{R.V2.sum():+.2f} / {R.V2_usd.sum():+.0f}** | **{R.V3.sum():+.2f} / {R.V3_usd.sum():+.0f}** |")
    L.append(""); L.append("Better than the stop: " + " · ".join(f"{V} {(R[V] > 0).sum()} of {len(R)} (worse {(R[V] < 0).sum()})" for V in ("V1", "V2", "V3")))
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[4:]))
