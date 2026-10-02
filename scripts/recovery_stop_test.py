#!/usr/bin/env python3
"""🩹 RECOVERY STOP — operator hypothesis (2026-10-02, the ZRO stop): "when a momentum long hits its stop while BTC's RSI is
ABOVE its entry value and above 62, BTC has not weakened — don't close, the pair comes back". Master evidence = 2 of 2 (ZRO, 0G);
4 trades meet 'RSI at stop ≥ entry' (2 recover). Tested here on the year engine replay's stopped momentum longs.

PRE-DECLARED before any result:
  cohort     every MOM-long fill of reports/ENGINE_REPLAY_YEAR_2026-09-26_fills.csv closed by STOP_LOSS / STOP_LOSS_WIDE
  ruler      BTC RSI(14) on CLOSED 5m bars (cache btc_5m.csv) at opened_at and at closed_at
  condition  C = RSI at stop ≥ RSI at entry ∧ RSI at stop ≥ 62      (also shown: each half alone, and NOT-C as the control)
  recovery   at the stop, do NOT close: hard stop a further E below the stop price · exit when P&L vs ENTRY returns to T ·
             else close at the 1m close after 30 min. 1m futures bars from the stop minute + 1; hard stop first inside a bar.
  grid       E ∈ {0.5, 1.0} × T ∈ {−0.20, 0.00, +0.40} = 6 cells.  Δ = recovery result − the actual stop result (same single exit fee)
  PASS       a cell passes if, on C: Δ mean > 0 in BOTH halves (Jan–Apr / May–Sep) ∧ day-clustered 95 % interval above 0
             ∧ Δ on C > Δ on NOT-C in both halves. ~6 cells → a lone pass is weak; the dose across cells matters.
  caveat     replay stops ≠ live stops (the replica stops fills live rode through); the 23 live master stops are listed beside.
Usage: venv/bin/python scripts/recovery_stop_test.py → reports/RECOVERY_STOP_TEST_2026-10-02.md"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); C1 = os.path.join(ROOT, "reports", "backtest_cache", "k1m_stop")
OUT = os.path.join(ROOT, "reports", "RECOVERY_STOP_TEST_2026-10-02.md"); SPLIT = "2026-05-01"; GRID = [(e, t) for e in (0.5, 1.0) for t in (-0.20, 0.0, 0.40)]


def tq(p, n):
    z = 1.959964; d = max(n - 1, 1); return z + (z**3 + z) / (4 * d) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * d * d)


def path(pair, ms):
    f = os.path.join(C1, f"{pair}_{ms}.json")
    if os.path.exists(f):
        return json.load(open(f))
    for k in range(4):
        try:
            r = json.load(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=1m&startTime={ms}&limit=31", timeout=20)); break
        except Exception:
            r = None; time.sleep(2 + 2 * k)
    r = [[float(x[2]), float(x[3]), float(x[4])] for x in (r or [])]
    os.makedirs(C1, exist_ok=True); json.dump(r, open(f, "w")); time.sleep(0.04)
    return r


def recover(bars, entry, stop_px, E, T):
    hard = stop_px * (1 - E / 100); tgt = entry * (1 + T / 100)
    for h, l, c in bars:
        if l <= hard:
            return (hard / entry - 1) * 100
        if h >= tgt:
            return T
    return (bars[-1][2] / entry - 1) * 100


if __name__ == "__main__":
    f = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YEAR_2026-09-26_fills.csv"), low_memory=False)
    s = f[(f.sleeve == "MOM-long") & f.close_reason.astype(str).str.contains("STOP_LOSS")].copy()
    b = pd.read_csv(os.path.join(ROOT, "reports", "backtest_cache", "btc_5m.csv")); b = b.rename(columns={b.columns[0]: "t"}).drop_duplicates("t").sort_values("t")
    d = b.c.diff(); u = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); v = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    TT, RR = b.t.values, (100 - 100 / (1 + u / v)).values
    rsi = lambda ts: RR[np.searchsorted(TT, int(pd.Timestamp(ts).timestamp() * 1000) - 300_000, side="right") - 1]
    s["rin"] = [rsi(x) for x in s.opened_at]; s["rout"] = [rsi(x) for x in s.closed_at]; s["day"] = s.closed_at.astype(str).str[:10]
    s["C"] = (s.rout >= s.rin) & (s.rout >= 62); s["ge"] = s.rout >= s.rin; s["hi"] = s.rout >= 62
    rows = []
    for n_, r in enumerate(s.itertuples()):
        ms = (int(pd.Timestamp(r.closed_at).timestamp() * 1000) // 60_000 + 1) * 60_000
        bars = path(r.pair, ms)
        if len(bars) < 25:
            rows.append([np.nan] * len(GRID)); continue
        rows.append([recover(bars, r.entry_price, r.exit_price, E, T) - (r.exit_price / r.entry_price - 1) * 100 for E, T in GRID])
        if n_ % 200 == 0:
            print(n_, flush=True)
    for j, (E, T) in enumerate(GRID):
        s[f"d_{E}_{T}"] = [x[j] for x in rows]
    s = s.dropna(subset=[f"d_{GRID[0][0]}_{GRID[0][1]}"]); s.to_csv(os.path.join(ROOT, "reports", "backtest_cache", "recovery_stop_rows.csv"), index=False)
    L = ["# 🩹 RECOVERY STOP — hold a stopped momentum long when BTC's RSI has not weakened?", "",
         f"{len(s)} stopped momentum longs of the year replay with a 30-min 1m path. Condition C (BTC RSI at stop ≥ entry ∧ ≥ 62): {int(s.C.sum())} "
         f"on {s[s.C].day.nunique()} days. Δ = % of position gained (+) or lost (−) versus taking the stop.", "",
         f"Did the pair get back above its ENTRY within ~45 min (recorded post-exit peak > 0)? C {(s[s.C].post_exit_peak_pnl > 0).mean() * 100:.0f} % · "
         f"RSI ≥ entry only {(s[s.ge].post_exit_peak_pnl > 0).mean() * 100:.0f} % · RSI ≥ 62 only {(s[s.hi].post_exit_peak_pnl > 0).mean() * 100:.0f} % · "
         f"NOT-C {(s[~s.C].post_exit_peak_pnl > 0).mean() * 100:.0f} % · all {(s.post_exit_peak_pnl > 0).mean() * 100:.0f} %", "",
         "| Extra room E | Exit target T | C: N | C: Δ Jan–Apr / May–Sep | C: Δ by day [95 %] | better than stop | NOT-C: Δ Jan–Apr / May–Sep | PASS |", "|---|---|---|---|---|---|---|---|"]
    for E, T in GRID:
        c = f"d_{E}_{T}"; g = s[s.C]; o = s[~s.C]
        a1, a2 = g[g.day < SPLIT][c].mean(), g[g.day >= SPLIT][c].mean(); o1, o2 = o[o.day < SPLIT][c].mean(), o[o.day >= SPLIT][c].mean()
        dm = g.groupby("day")[c].mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n); lo, hi = dm.mean() - tq(.975, n) * se, dm.mean() + tq(.975, n) * se
        ok = a1 > 0 and a2 > 0 and lo > 0 and a1 > o1 and a2 > o2
        L.append(f"| −{E} | {T:+.2f} | {len(g)} | {a1:+.3f} / {a2:+.3f} | {dm.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {(g[c] > 0).mean() * 100:.0f}% | {o1:+.3f} / {o2:+.3f} | {'✅' if ok else '—'} |")
    L += ["", "Dose — Δ of the −1.0 / 0.00 cell by BTC RSI change (stop − entry) and by RSI level at the stop:", "", "| Cut | N | Δ | back above entry |", "|---|---|---|---|"]
    c = "d_1.0_0.0"; s["chg"] = s.rout - s.rin
    for lab, g in [("RSI change ≥ +5", s[s.chg >= 5]), ("0 … +5", s[(s.chg >= 0) & (s.chg < 5)]), ("−5 … 0", s[(s.chg >= -5) & (s.chg < 0)]), ("−10 … −5", s[(s.chg >= -10) & (s.chg < -5)]), ("< −10", s[s.chg < -10]),
                   ("RSI at stop ≥ 65", s[s.rout >= 65]), ("60–65", s[(s.rout >= 60) & (s.rout < 65)]), ("50–60", s[(s.rout >= 50) & (s.rout < 60)]), ("< 50", s[s.rout < 50])]:
        L.append(f"| {lab} | {len(g)} | {g[c].mean():+.3f} | {(g.post_exit_peak_pnl > 0).mean() * 100:.0f}% |")
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
