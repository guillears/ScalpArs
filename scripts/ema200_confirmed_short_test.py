#!/usr/bin/env python3
"""📉 CONFIRMED EMA200 BREAK SHORT — frozen 2026-10-02 after the operator's MOVR read ("the EMA200 short that paid came after a confirmed trend
— find what defines it"). Screen that led here (25 cuts of 2,470 breaks, so this is a candidate, NOT a result): the short is positive in both
halves when the EMA50 has already come down to the EMA200, and negative when it is still far above (a sharp dip inside an uptrend).
FROZEN RULE (before any tight-stop result):
  episode   onset = 5m close with 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× the pair's normal hour ∧ ≥ $2M (none in the prior 24 h);
            the run reached ≥ +50 % above the pre-spike price; 24 h volume at the break ≥ $50M
  trigger   a 5m CLOSE below the EMA200 with the previous 12 closes above it, 4–96 h after the onset
  CONFIRMED the EMA50 is 0 … 2 % above the EMA200 at that close  ∧  the run's peak was ≥ 4 h earlier
  trade     SHORT at the next open, one position per pair, 1-minute bars, stop first inside a bar, 12 h cap
  exits     stop S ∈ {1, 1.5, 2, 3} % × trailing (starts at A %, gives back T %) ∈ {(2, 1), (3, 1), (3, 1.5), (5, 1.5), (5, 2)} = 20 cells
  limits    none · at most 2 trades per pair per UTC day · pause the pair for the rest of the day after 2 stops in a row
  cost      0.11 % + real funding while open
  control   the same exits on the EMA200 breaks that are NOT confirmed (EMA50 > 2 % above, or peak < 4 h ago)
  PASS      per cell: mean > 0 in both halves ∧ day-clustered 95 % interval above 0. 20 cells → the pattern across cells matters, not one cell.
Usage: venv/bin/python scripts/ema200_confirmed_short_test.py"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_review as B  # noqa: E402  (bars1m cache, walk, funding via FS)
sys.argv = _a
ST, FS = B.ST, B.FS; OUT = os.path.join(ST.ROOT, "reports", "EMA200_CONFIRMED_SHORT_TEST_2026-10-02.md"); COST = 0.11; SPLIT = "2026-05-01"
STOPS = (1.0, 1.5, 2.0, 3.0); TRAILS = ((2.0, 1.0), (3.0, 1.0), (3.0, 1.5), (5.0, 1.5), (5.0, 2.0))


def breaks(pair, d, G=50.0, min_q24=50e6):
    t = d.open_time.values.astype("int64"); o, h, c, q = d.o.values, d.h.values, d.c.values, d.qvol.values; n = len(c)
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); volx = (q1h / norm).values
    r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]; e200 = pd.Series(c).ewm(span=200, adjust=False).mean().values; e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    with np.errstate(invalid="ignore"):
        lead = np.nonzero((r30 >= 5) & (volx >= 20) & (q1h.values >= 2e6))[0]
    ab = pd.Series(c >= e200).rolling(12).min().shift(1).fillna(0).values >= 1; cq = np.concatenate([[0.0], np.cumsum(q)]); out = []; last = -10**18; seen = set()
    for on in lead:
        on = int(on)
        if t[on] - last < 24 * 3600_000:
            last = t[on]; continue
        last = t[on]; base = c[on - 6]; pk = on + int(np.argmax(h[on:min(on + 48, n)]))
        for j in range(on + 48, min(n - 2, on + 96 * 12)):
            if h[j] > h[pk]:
                pk = j
            if c[j] < e200[j] and ab[j] and (h[pk] / base - 1) * 100 >= G and j not in seen and cq[j + 1] - cq[max(j - 287, 0)] >= min_q24:
                seen.add(j); gap = (e50[j] / e200[j] - 1) * 100; hp = (t[j] - t[pk]) / 3600e3
                out.append(dict(pair=pair, t=int(t[j + 1]), entry=float(o[j + 1]), gap=gap, hrs_peak=hp, gain=(h[pk] / base - 1) * 100, confirmed=bool(0 <= gap < 2 and hp >= 4)))
    return out


def evaluate(G, paths, S, A, T, limit):
    rows = []; free = {}; cnt = {}; streak = {}; paused = {}
    for r in G.itertuples():
        day = r.t // 86_400_000
        if r.t < free.get(r.pair, 0) or r.Index not in paths:
            continue
        if limit == "max2" and cnt.get((r.pair, day), 0) >= 2:
            continue
        if limit == "pause2" and paused.get(r.pair) == day:
            continue
        h, l, c = paths[r.Index]; res, mins, how = B.walk(h, l, c, r.entry, S, A, T)
        free[r.pair] = r.t + mins * 60_000; cnt[(r.pair, day)] = cnt.get((r.pair, day), 0) + 1
        streak[r.pair] = streak.get(r.pair, 0) + 1 if how == "stop" else 0
        if streak[r.pair] >= 2:
            paused[r.pair] = day; streak[r.pair] = 0
        rows.append((r.Index, res, mins, how))
    return pd.DataFrame(rows, columns=["i", "r", "mins", "how"]).set_index("i")


if __name__ == "__main__":
    tr = []
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) >= 288 * 40:
            tr += breaks(pair, d)
    TR = pd.DataFrame(tr).sort_values("t").reset_index(drop=True); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d")
    print(len(TR), "EMA200 breaks ·", int(TR.confirmed.sum()), "confirmed", flush=True)
    paths = {}
    for n_, r in enumerate(TR.itertuples()):
        h, l, c = B.bars1m(r.pair, r.t)
        if len(c) >= 30:
            paths[r.Index] = (h, l, c)
        if n_ % 400 == 0:
            print(n_, flush=True)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in TR.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None

    def cell(G, S, A, T, limit):
        E = evaluate(G, paths, S, A, T, limit)
        if len(E) < 20:
            return None
        X = G.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how
        fund = [0.0 if FR.get(r.pair) is None else FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()]
        X["net"] = E.r - COST + np.array(fund); a1, a2 = X[X.day < SPLIT].net.mean(), X[X.day >= SPLIT].net.mean()
        dm = X.groupby("day").net.mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n); l_, h_ = dm.mean() - ST.tq(n) * se, dm.mean() + ST.tq(n) * se
        s = X.net.sort_values(); st = mx = 0
        for v in X.sort_values("t").net:
            st = st + 1 if v < 0 else 0; mx = max(mx, st)
        return dict(n=len(X), won=(X.net > 0).mean() * 100, stopped=(X.how == "stop").mean() * 100, win=X[X.net > 0].net.mean(), loss=X[X.net <= 0].net.mean(), a1=a1, a2=a2, dm=dm.mean(), lo=l_, hi=h_,
                    trim=s.iloc[:-max(1, int(len(s) * 0.05))].mean(), run=mx, ok=a1 > 0 and a2 > 0 and l_ > 0, mins=X.mins.median())
    C_, N_ = TR[TR.confirmed], TR[~TR.confirmed]
    L = ["# 📉 CONFIRMED EMA200 BREAK SHORT — tight stop + trailing exit, year test (1-minute bars)", "",
         f"{len(TR):,} EMA200 breaks after a +50 % run on ≥ $50M pairs, Jan–Sep 2026; **{len(C_)} confirmed** (EMA50 0–2 % above the EMA200 ∧ peak ≥ 4 h earlier) on {C_.pair.nunique()} pairs / {C_.day.nunique()} days. "
         "% of position at 1× after 0.11 % and funding. Per-stop cost at 20× = 20 × (stop + 0.11) % of the margin.", ""]
    for limit, lab in (("none", "no trade limit"), ("max2", "at most 2 trades per pair per day"), ("pause2", "pause the pair for the day after 2 stops in a row")):
        L += [f"## Confirmed breaks · {lab}", "", "| Stop | Trail (start / give-back) | trades | won | stopped | avg win / loss | per trade Jan–Apr / May–Sep | by day [95 %] | best 5 % removed | longest losing run | PASS |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for S in STOPS:
            for A, T in TRAILS:
                x = cell(C_, S, A, T, limit)
                if x:
                    L.append(f"| −{S:g} % | {A:g} / {T:g} | {x['n']} | {x['won']:.0f}% | {x['stopped']:.0f}% | {x['win']:+.1f} / {x['loss']:+.1f} | {x['a1']:+.2f} / {x['a2']:+.2f} | {x['dm']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] | {x['trim']:+.2f} | {x['run']} | {'✅' if x['ok'] else '—'} |")
        L.append("")
    L += ["## Control — the EMA200 breaks that are NOT confirmed (no trade limit)", "", "| Stop | Trail | trades | won | per trade Jan–Apr / May–Sep | by day [95 %] |", "|---|---|---|---|---|---|"]
    for S, A, T in ((1.0, 3.0, 1.0), (2.0, 3.0, 1.0), (3.0, 3.0, 1.0), (2.0, 5.0, 2.0)):
        x = cell(N_, S, A, T, "none")
        if x:
            L.append(f"| −{S:g} % | {A:g} / {T:g} | {x['n']} | {x['won']:.0f}% | {x['a1']:+.2f} / {x['a2']:+.2f} | {x['dm']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] |")
    TR.to_csv(os.path.join(ST.ROOT, "reports", "backtest_cache", "ema200_confirmed_breaks.csv"), index=False)
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
