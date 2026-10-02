#!/usr/bin/env python3
"""✂️ BTC-RSI early cut — operator (2026-10-02): "if BTC RSI is X points down since entry, exit … cut losers earlier".
PRE-DECLARED before any result. TWO-SIDED: applied to EVERY momentum long (winners pay for false cuts), not only to losers.
  ruler   BTC RSI(14) on CLOSED 5m bars; reference = its value at entry
  rule    at each 5m close while the trade is open: if RSI ≤ entry RSI − X → close at the pair's 5m close of that bar
          variant LOSING: only when the trade is below entry at that close
  grid    X ∈ {5, 10, 15, 20} × {ALWAYS, LOSING} = 8 cells.  Δ = cut result − actual result (gross, % of position)
  pools   (a) year engine replay MOM-long fills (3,591)  (b) the 108 current-stack master momentum longs + today (live)
  PASS    (a) Δ mean > 0 in both halves ∧ day-clustered 95 % interval above 0;  (b) is shown beside, never fitted
Usage: venv/bin/python scripts/btc_rsi_cut_test.py → reports/BTC_RSI_CUT_TEST_2026-10-02.md"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BC = os.path.join(ROOT, "reports", "backtest_cache")
OUT = os.path.join(ROOT, "reports", "BTC_RSI_CUT_TEST_2026-10-02.md"); SPLIT = "2026-05-01"; BAR = 300_000
GRID = [(x, m) for m in ("ALWAYS", "LOSING") for x in (5, 10, 15, 20)]
_P = {}


def tq(n):
    z = 1.959964; d = max(n - 1, 1); return z + (z**3 + z) / (4 * d) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * d * d)


def pair5(p):
    if p not in _P:
        fs = [os.path.join(BC, d_, p + ".csv") for d_ in ("k5m_full", "k5m_ext")]; fr = [pd.read_csv(f)[["open_time", "c"]] for f in fs if os.path.exists(f) and os.path.getsize(f) > 30]
        _P[p] = None if not fr else pd.concat(fr).astype(float).drop_duplicates("open_time").sort_values("open_time")
        if _P[p] is not None:
            _P[p] = (_P[p].open_time.values.astype("int64"), _P[p].c.values)
    return _P[p]


def run(F, TT, RR, label):
    rows = []
    for r in F.itertuples():
        a, z = int(pd.Timestamp(r.opened_at).timestamp() * 1000), int(pd.Timestamp(r.closed_at).timestamp() * 1000); pp = pair5(r.pair)
        if pp is None:
            continue
        i0 = np.searchsorted(TT, a - BAR, side="right") - 1; rin = RR[i0]; act = (r.exit_price / r.entry_price - 1) * 100
        j0, j1 = i0 + 1, np.searchsorted(TT, z - BAR, side="right") - 1          # bars that CLOSE inside the trade
        out = dict(pair=r.pair, day=str(r.closed_at)[:10], act=act, win=act > 0, rin=rin, bars=max(j1 - j0 + 1, 0))
        for X, m in GRID:
            d = 0.0
            for j in range(j0, j1 + 1):
                if RR[j] <= rin - X:
                    k = np.searchsorted(pp[0], TT[j])
                    if k < len(pp[0]) and pp[0][k] == TT[j]:
                        v = (pp[1][k] / r.entry_price - 1) * 100
                        if m == "ALWAYS" or v < 0:
                            d = v - act; break
            out[f"{m}_{X}"] = d
        rows.append(out)
    D = pd.DataFrame(rows); L = [f"## {label} — {len(D)} fills, {D.day.nunique()} days · win rate {D.win.mean() * 100:.0f} % · median hold {int(D.bars.median())} closed 5m bars", "",
                                 "| Rule | cuts fired | on eventual losers: N · Δ each | on eventual winners: N · Δ each | Δ per fill Jan–Apr / May–Sep | Δ by day [95 %] | PASS |", "|---|---|---|---|---|---|---|"]
    for X, m in GRID:
        c = f"{m}_{X}"; f_ = D[D[c] != 0]; lo_, wi_ = f_[~f_.win], f_[f_.win]; a1, a2 = D[D.day < SPLIT][c].mean(), D[D.day >= SPLIT][c].mean()
        dm = D.groupby("day")[c].mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n); lo, hi = dm.mean() - tq(n) * se, dm.mean() + tq(n) * se
        L.append(f"| RSI −{X}, {m.lower()} | {len(f_)} ({len(f_) / len(D) * 100:.0f}%) | {len(lo_)} · {lo_[c].mean():+.3f} | {len(wi_)} · {wi_[c].mean():+.3f} | {a1:+.4f} / {a2:+.4f} | {dm.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {'✅' if a1 > 0 and a2 > 0 and lo > 0 else '—'} |")
    return L + [""]


if __name__ == "__main__":
    b = pd.read_csv(os.path.join(BC, "btc_5m.csv")); b = b.rename(columns={b.columns[0]: "t"})[["t", "c"]]
    e = os.path.join(BC, "k5m_ext", "BTCUSDT.csv")
    if os.path.exists(e):
        b = pd.concat([b, pd.read_csv(e).rename(columns={"open_time": "t"})[["t", "c"]]])
    b = b.astype(float).drop_duplicates("t").sort_values("t"); d = b.c.diff()
    u = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); v = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    TT, RR = b.t.values.astype("int64"), (100 - 100 / (1 + u / v)).values
    Y = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YEAR_2026-09-26_fills.csv"), low_memory=False); Y = Y[Y.sleeve == "MOM-long"]
    L = ["# ✂️ BTC-RSI early cut on momentum longs — close when BTC's RSI has dropped X points since entry", "",
         "Δ = % of position versus what the trade actually did (positive = the cut helped). Two-sided: winners cut too early count against the rule.", ""]
    L += run(Y, TT, RR, "Year engine replay")
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        M = pd.read_pickle(sys.argv[1]); M = M[pd.to_datetime(M.closed_at).astype("int64") // 10**6 <= TT[-1]]
        L += run(M, TT, RR, "Live master momentum longs (current stack)")
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
