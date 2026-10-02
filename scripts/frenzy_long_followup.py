#!/usr/bin/env python3
"""🔥 FRENZY_LONG follow-up — two tests FROZEN 2026-10-02 before reading (the designed cell fell to +0.05 %/trade on the strict ruler).
  entry (both)  the staircase state turns ON after ≥ 1 h off (≥ 2 h after the spike ∧ an hour of 5m closes ≥ the spike-anchored VWAP ∧ last-hour
                volume ≥ 100× normal); one position per pair
  sets          SEEN = 24 h volume ≥ $100M (the set every earlier cut was read on) · UNSEEN = $20–100M (never tested) — a rule must hold on both
  TEST 1  ATR   (1-minute bars) a) gate: enter only when 5m ATR(14) ≤ 2 % (the 3 % stop ≥ 1.5 ATR), exit stop 3 · trail 5/1.5
                b) scaled: stop = k×ATR (k 1.5 → 2–6 %, k 2 → 2–8 %) · trail arms at 2.5×ATR (3–10 %) and gives back 1×ATR (1–4 %) · 12 h cap
  TEST 2  RIDE  (5-minute bars) sell at the open after a 5m close below the anchored VWAP; stop = 8 % (the earlier year test) / 3 % / 1.5×ATR (2–8 %)
                / 2×ATR (2–8 %) / 3×ATR (3–10 %); cap 96 h after the spike (the flag ends)
  ruler         gap-aware fills, 0.10 % slippage on every exit, 0.11 % costs, real funding (the long pays)
  PASS          per trade > 0 in both halves ∧ 95 % range by day above 0 — on SEEN and on UNSEEN
  size          'R' = result ÷ the trade's own stop; account path with every stop costing 2 % of the account (compounding, time order)
Usage: venv/bin/python scripts/frenzy_long_followup.py"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_hostile_review as H  # noqa: E402
sys.argv = _a
B, FS, ST = H.B, H.FS, H.ST; OUT = os.path.join(ST.ROOT, "reports", "FRENZY_LONG_FOLLOWUP_2026-10-02.md"); SLIP = 0.10; clip = lambda x, a, b: float(min(max(x, a), b))


def triggers(pair, d, V=100.0):
    t = d.open_time.values.astype("int64"); o, h, l, c, q = d.o.values, d.h.values, d.l.values, d.c.values, d.qvol.values; n = len(c)
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); volx = (q1h / norm).values
    r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]; pc = np.r_[c[0], c[:-1]]
    atr = (pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).ewm(alpha=1 / 14, adjust=False).mean() / c * 100).values; cq = np.concatenate([[0.0], np.cumsum(q)])
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
        for j in range(on, min(end, n - 2)):
            if st[j - on]:
                if off >= 12:
                    q24 = cq[j + 1] - cq[max(j - 287, 0)]
                    if q24 >= 20e6:
                        # ride-it walk on 5m bars from the entry bar: (stop-independent pieces) the first close below the VWAP, or the 96 h cap
                        k_end = min(n - 2, on + 96 * 12); x = None
                        for k in range(j + 1, k_end):
                            if c[k] < vw[k - on]:
                                x = k; break
                        x = k_end if x is None else x
                        out.append(dict(pair=pair, t=int(t[j + 1]), entry=float(o[j + 1]), atr=float(atr[j]), q24=float(q24), j=j + 1, x=x, exit_open=float(o[x + 1])))
                off = 0
            else:
                off += 1
        i = end
    return out, (t, o, l)


def ride(o, l, r, S):
    """5m bars j..x: stop first (a bar opening below the stop fills at its open); else out at the open after the close below the VWAP."""
    sp = r.entry * (1 - S / 100)
    for k in range(r.j, r.x + 1):
        if l[k] <= sp:
            return (min(sp, o[k]) / r.entry - 1) * 100 - SLIP, (k - r.j + 1) * 5, "stop"
    return (r.exit_open / r.entry - 1) * 100 - SLIP, (r.x + 1 - r.j) * 5, "line"


def stats(X, scol="S"):
    if len(X) < 30:
        return f"| {len(X)} | – | | | | | | | |"
    bd = H.boot(X, "day", 1500); s = X.net.sort_values(); a1, a2 = X[X.day < B.SPLIT].net.mean(), X[X.day >= B.SPLIT].net.mean(); R = (X.net / X[scol]).values
    eq = 1.0; pk = 1.0; dd = 0.0
    for v in (X.sort_values("t").net / X.sort_values("t")[scol]).values:
        eq *= max(0.0, 1 + 0.02 * v); pk = max(pk, eq); dd = max(dd, 1 - eq / pk)
    run_ = max((len(z) for z in "".join("L" if v <= 0 else "W" for v in X.sort_values("t").net).split("W")), default=0)
    ok = a1 > 0 and a2 > 0 and bd[0] > 0
    return (f"| {len(X)} | {(X.net > 0).mean() * 100:.0f}% | {X.net.mean():+.2f} | {a1:+.2f} / {a2:+.2f} | [{bd[0]:+.2f}, {bd[1]:+.2f}] | {s.iloc[:-max(1, int(len(s) * 0.05))].mean():+.2f} | {R.mean():+.3f} R | "
            f"×{eq:.2f} (−{dd * 100:.0f}%) | {run_} | {'✅' if ok else '—'} |")


if __name__ == "__main__":
    tr = []; K5 = {}
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) < 288 * 40:
            continue
        got, arr = triggers(pair, d)
        if got:
            tr += got; K5[pair] = arr
    TR = pd.DataFrame(tr).sort_values("t").reset_index(drop=True); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d"); TR["set"] = np.where(TR.q24 >= 100e6, "SEEN", "UNSEEN")
    print(len(TR), "entries", TR.set.value_counts().to_dict(), flush=True)
    paths = {}
    for n_, r in enumerate(TR.itertuples()):
        h, l, c = B.bars1m(r.pair, r.t)
        if len(c) >= 30:
            paths[r.Index] = (h, l, c)
        if n_ % 300 == 0:
            print(n_, flush=True)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in TR.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None
    fund = lambda X: np.array([0.0 if FR.get(r.pair) is None else -FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()])

    def t1(G, sfn, afn, tfn):
        rows = []; free = {}
        for r in G.itertuples():
            if r.t < free.get(r.pair, 0) or r.Index not in paths:
                continue
            S, A, T = sfn(r.atr), afn(r.atr), tfn(r.atr); res, mins, how = H.walk(*paths[r.Index], r.entry, S, A, T, side=1, slip=SLIP, gap=True); free[r.pair] = r.t + mins * 60_000
            rows.append((r.Index, res, mins, how, S))
        E = pd.DataFrame(rows, columns=["i", "r", "mins", "how", "S"]).set_index("i"); X = G.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how; X["S"] = E.S + 0.21
        X["net"] = E.r - B.COST + fund(X); return X

    def t2(G, sfn):
        rows = []; free = {}
        for r in G.itertuples():
            if r.t < free.get(r.pair, 0):
                continue
            _, o, l = K5[r.pair]; S = sfn(r.atr); res, mins, how = ride(o, l, r, S); free[r.pair] = r.t + mins * 60_000; rows.append((r.Index, res, mins, how, S))
        E = pd.DataFrame(rows, columns=["i", "r", "mins", "how", "S"]).set_index("i"); X = G.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how; X["S"] = E.S + 0.21
        X["net"] = E.r - B.COST + fund(X); return X

    f3, f5, f15 = (lambda a: 3.0), (lambda a: 5.0), (lambda a: 1.5)
    HEAD = ["| Rule | trades | won | per trade | Jan–Apr / May–Sep | 95 % by day | best 5 % removed | per trade in stops | account at 2 % per stop (deepest drawdown) | longest losing run | PASS |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    L = ["# 🔥 FRENZY_LONG follow-up — ATR at entry, and the ride-it exit (frozen before reading)", "",
         f"{len(TR):,} entries: {(TR.set == 'SEEN').sum():,} SEEN (24 h volume ≥ $100M) · {(TR.set == 'UNSEEN').sum():,} UNSEEN ($20–100M). Gap-aware fills, 0.10 % slippage, 0.11 % costs, funding. "
         f"Median 5m ATR at entry: {TR.atr.median():.1f} % · ATR ≤ 2 % on {(TR.atr <= 2).mean() * 100:.0f}% of entries.", ""]
    for sname in ("SEEN", "UNSEEN"):
        G = TR[TR.set == sname]
        L += [f"## TEST 1 · ATR — {sname} ({len(G):,} entries, {G.pair.nunique()} pairs) · 1-minute bars, 12 h cap", ""] + HEAD
        for name, g, fns in (("All entries · stop 3 · trail 5/1.5 (the design)", G, (f3, f5, f15)), ("GATE ATR ≤ 2 % · stop 3 · trail 5/1.5", G[G.atr <= 2.0], (f3, f5, f15)), ("(the rest: ATR > 2 %)", G[G.atr > 2.0], (f3, f5, f15)),
                             ("SCALED stop 1.5×ATR · trail arms 2.5×ATR, gives back 1×ATR", G, (lambda a: clip(1.5 * a, 2, 6), lambda a: clip(2.5 * a, 3, 10), lambda a: clip(a, 1, 4))),
                             ("SCALED stop 2×ATR · same trail", G, (lambda a: clip(2 * a, 2, 8), lambda a: clip(2.5 * a, 3, 10), lambda a: clip(a, 1, 4)))):
            L.append(f"| {name} " + stats(t1(g, *fns)))
        L += ["", f"## TEST 2 · RIDE IT — {sname} · 5-minute bars, sell after a close below the spike's average price", ""] + HEAD
        for name, fn in (("stop 8 % (the earlier year test)", lambda a: 8.0), ("stop 3 %", f3), ("stop 1.5×ATR (2–8 %)", lambda a: clip(1.5 * a, 2, 8)), ("stop 2×ATR (2–8 %)", lambda a: clip(2 * a, 2, 8)), ("stop 3×ATR (3–10 %)", lambda a: clip(3 * a, 3, 10))):
            X = t2(G, fn); L.append(f"| {name} " + stats(X))
            if "2×ATR" in name:
                keep = X
        L += ["", f"Ride-it with the 2×ATR stop: stopped {(keep.how == 'stop').mean() * 100:.0f}% · median hold {keep.mins.median() / 60:.1f} h · average win {keep[keep.net > 0].net.mean():+.1f} % / loss {keep[keep.net <= 0].net.mean():+.1f} % · "
              f"biggest win {keep.net.max():+.0f} % · trades per day (median on trading days) {keep.groupby('day').size().median():.0f}.", ""]
    L += ["## NOT tested", "", "- Delisted pairs; slippage beyond 0.10 % (thin UNSEEN pairs will be worse); exchange position limits; the live scan delay.",
          "- TEST 2 runs on 5-minute bars (stop first inside a bar), TEST 1 on 1-minute bars — the two tests are not on one ruler.", "- No fresh time period: UNSEEN means thinner pairs, not later dates."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
