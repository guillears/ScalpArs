#!/usr/bin/env python3
"""🧮 How many attempts? — operator (2026-10-02): "the cap may not be 2 but 3 or 4 … and maybe after the stops there is a win. Think smart."
For each leg of the sleeve, with the exit held fixed, on the year's 1-minute paths (already cached by the leg tests):
  A  result by ATTEMPT NUMBER on the same pair the same UTC day (1st, 2nd, 3rd, 4th, 5th+)
  B  result of a trade given how many STOPS IN A ROW came just before it on that pair (0, 1, 2, 3, 4+) within 24 h — is the next one a win?
  C  the cap grid: at most N trades per pair per day (N = 1 … 6, none) × pause for the day after K stops in a row (K = 1 … 4, none):
     trades · total · per trade in each half · longest losing run
Reads are after-the-fact (no bar was set first): a cap is only credible if neighbouring N / K agree and both halves agree.
Usage: venv/bin/python scripts/sleeve_limits_analysis.py [STOP A T]   (default 3 3 1)"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import run_case_study as C  # noqa: E402
import ema200_confirmed_short_test as E2  # noqa: E402
sys.argv = _a
B, ST = C.B, C.ST; COST = 0.11; SPLIT = "2026-05-01"; OUT = os.path.join(ST.ROOT, "reports", "SLEEVE_LIMITS_ANALYSIS_2026-10-02.md")
S, A, T = (float(x) for x in sys.argv[1:4]) if len(sys.argv) >= 4 else (3.0, 3.0, 1.0)


def legs():
    L, S50, S200 = [], [], []
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) < 288 * 40:
            continue
        t = d.open_time.values.astype("int64"); cq = np.concatenate([[0.0], np.cumsum(d.qvol.values)])
        for x in C.long_triggers(d):
            i = int(np.searchsorted(t, x["t"]))
            if cq[i] - cq[max(i - 288, 0)] >= 100e6:
                L.append(dict(pair=pair, t=x["t"], entry=x["entry"], near=(x["entry"] / x["vwap"] - 1) * 100 <= 5))
        S50 += [dict(pair=pair, t=x["t"], entry=x["entry"]) for x in B.triggers(pair, d, 50)]
        S200 += [dict(pair=pair, t=x["t"], entry=x["entry"], confirmed=x["confirmed"]) for x in E2.breaks(pair, d)]
    L, S50, S200 = (pd.DataFrame(x).sort_values("t").reset_index(drop=True) for x in (L, S50, S200))
    return {"LONG · staircase": ("LONG", L), "LONG · staircase, near the line": ("LONG", L[L.near].reset_index(drop=True)), "SHORT · EMA50 break": ("SHORT", S50),
            "SHORT · EMA200 break (all)": ("SHORT", S200), "SHORT · EMA200 break, confirmed": ("SHORT", S200[S200.confirmed].reset_index(drop=True))}


def load(TR):
    paths = {}
    for r in TR.itertuples():
        f = os.path.join(B.C1, f"{r.pair}_{r.t}.npz")
        if os.path.exists(f):
            z = np.load(f)
            if len(z["c"]) >= 30:
                paths[r.Index] = (z["h"], z["l"], z["c"])
    return paths


def run(side, TR, paths, max_day=None, pause=None):
    rows = []; free = {}; cnt = {}; hist = {}; paused = {}
    for r in TR.itertuples():
        day = r.t // 86_400_000
        if r.t < free.get(r.pair, 0) or r.Index not in paths or (max_day and cnt.get((r.pair, day), 0) >= max_day) or (pause and paused.get(r.pair) == day):
            continue
        h, l, c = paths[r.Index]; res, mins, how = C.walk(side, h, l, c, r.entry, S, A, T)
        hh = [x for x in hist.get(r.pair, []) if r.t - x[0] <= 86_400_000]; k = 0
        for _, st in reversed(hh):
            if not st:
                break
            k += 1
        cnt[(r.pair, day)] = cnt.get((r.pair, day), 0) + 1; free[r.pair] = r.t + mins * 60_000
        hist[r.pair] = hh + [(r.t, how == "stop")]
        if pause and k + (how == "stop") >= pause and how == "stop":
            paused[r.pair] = day
        rows.append(dict(t=r.t, net=res - COST, how=how, attempt=cnt[(r.pair, day)], prior_stops=k))
    X = pd.DataFrame(rows)
    if len(X):
        X["h"] = np.where(pd.to_datetime(X.t, unit="ms") < SPLIT, "A", "B")
    return X


def line(g):
    if len(g) < 15:
        return f"{len(g)} · –"
    return f"{len(g):,} · won {(g.net > 0).mean() * 100:.0f}% · {g.net.mean():+.2f} ({g[g.h == 'A'].net.mean():+.2f} / {g[g.h == 'B'].net.mean():+.2f})"


if __name__ == "__main__":
    R = [f"# 🧮 How many attempts per pair? — every leg, exit fixed at stop {S:g} % · trail from +{A:g} % giving back {T:g} % (before funding)", "",
         "Each cell: trades · win rate · per trade (Jan–Apr / May–Sep), % of position at 1×.", ""]
    for name, (side, TR) in legs().items():
        paths = load(TR); X = run(side, TR, paths)
        R += [f"## {name} — {len(X):,} trades ({len(paths):,} of {len(TR):,} signals have minute data)", "", "**A · by attempt number on the pair that day**", "", "| Attempt | result |", "|---|---|"]
        R += [f"| {lab} | {line(X[m])} |" for lab, m in (("1st", X.attempt == 1), ("2nd", X.attempt == 2), ("3rd", X.attempt == 3), ("4th", X.attempt == 4), ("5th or later", X.attempt >= 5))]
        R += ["", "**B · by the stops in a row just before it (same pair, last 24 h)**", "", "| Before this trade | result |", "|---|---|"]
        R += [f"| {lab} | {line(X[m])} |" for lab, m in (("no stop just before", X.prior_stops == 0), ("1 stop", X.prior_stops == 1), ("2 stops in a row", X.prior_stops == 2), ("3 in a row", X.prior_stops == 3), ("4 or more", X.prior_stops >= 4))]
        R += ["", "**C · caps** (total points · trades · per trade Jan–Apr / May–Sep · longest losing run)", "", "| Max per pair per day ↓ / pause after K stops → | K = 1 | K = 2 | K = 3 | K = 4 | no pause |", "|---|---|---|---|---|---|"]
        for N in (1, 2, 3, 4, 6, None):
            row = f"| {N if N else 'no cap'} |"
            for K in (1, 2, 3, 4, None):
                Y = run(side, TR, paths, N, K)
                if len(Y) < 15:
                    row += " – |"; continue
                st = mx = 0
                for v in Y.net:
                    st = st + 1 if v < 0 else 0; mx = max(mx, st)
                row += f" {Y.net.sum():+.0f} · {len(Y):,} · {Y[Y.h == 'A'].net.mean():+.2f} / {Y[Y.h == 'B'].net.mean():+.2f} · run {mx} |"
            R.append(row)
        R.append("")
    open(OUT, "w").write("\n".join(R) + "\n"); print("\n".join(R))
