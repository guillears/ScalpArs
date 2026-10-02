#!/usr/bin/env python3
"""📉 EMA200 short on frenzy pairs, second look (operator 2026-10-02, MOVR chart: "this is a massive move"). The first break of the EMA200
failed the year test; MOVR's fall came AFTER the break, with the price living below a falling EMA200 and being rejected at it.
FROZEN before reading — two entries the earlier tests did not have, on the same episodes (spike → run ≥ +50 %, 4–96 h after the spike,
24 h volume ≥ $50M):
  RETEST   the price has closed below the 5m EMA200 for ≥ 1 h, then a bar's HIGH comes within 1 % of the EMA200 and it CLOSES below the line
           and below its open → SHORT at the next open (one per hour per pair). RETEST2 = the same within 2 % (added before the year run:
           MOVR's 10-02 rejection came within ~1.5 % and the 1 % version did not fire on it)
  CROSS    the 5m EMA50 crosses below the EMA200 → SHORT at the next open
  exits    stop 2 % / 3 % / "above the line" (EMA200 + 1 %, held to 1–4 %) · trail arms at +3 % or +5 %, gives back 2 % or 3 % · 12 h cap
  ruler    1-minute bars, gap-aware fills, 0.10 % slippage, 0.11 % costs, funding. One position per pair.
  PASS     per trade > 0 in both halves ∧ 95 % range by day above 0. 2 entries × 2 run sizes × 9 exits = 36 cells — a lone pass means little.
Usage: venv/bin/python scripts/ema200_retest_short_test.py [--cases]"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_hostile_review as H  # noqa: E402
import run_case_study as RC  # noqa: E402
sys.argv = _a
B, FS, ST = H.B, H.FS, H.ST; OUT = os.path.join(ST.ROOT, "reports", "EMA200_RETEST_SHORT_TEST_2026-10-02.md")
EXITS = [(S, A, T) for S in (2.0, 3.0, "line") for A, T in ((3.0, 2.0), (5.0, 3.0), (5.0, 2.0))]


def triggers(pair, d, G=50.0, min_q24=50e6):
    t = d.open_time.values.astype("int64"); o, h, c, q = d.o.values, d.h.values, d.c.values, d.qvol.values; n = len(c)
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); volx = (q1h / norm).values
    r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]
    e200 = pd.Series(c).ewm(span=200, adjust=False).mean().values; e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    with np.errstate(invalid="ignore"):
        lead = np.nonzero((r30 >= 5) & (volx >= 20) & (q1h.values >= 2e6))[0]
    below1h = pd.Series(c < e200).rolling(12).min().shift(1).fillna(0).values >= 1; cq = np.concatenate([[0.0], np.cumsum(q)])
    out = []; last = -10**18; seen = set(); last_rt = -10**9
    for on in lead:
        on = int(on)
        if t[on] - last < 24 * 3600_000:
            last = t[on]; continue
        last = t[on]; base = c[on - 6]; peak = h[on:min(on + 48, n)].max()
        for j in range(on + 48, min(n - 2, on + 96 * 12)):
            peak = max(peak, h[j])
            if (peak / base - 1) * 100 < G or j in seen or cq[j + 1] - cq[max(j - 287, 0)] < min_q24:
                continue
            kind = None
            if below1h[j] and h[j] >= e200[j] * 0.98 and c[j] < e200[j] and c[j] < o[j] and j - last_rt >= 12:
                kind = "RETEST" if h[j] >= e200[j] * 0.99 else "RETEST2"; last_rt = j
            elif e50[j] < e200[j] and e50[j - 1] >= e200[j - 1]:
                kind = "CROSS"
            if kind:
                seen.add(j)
                out.append(dict(pair=pair, kind=kind, t=int(t[j + 1]), entry=float(o[j + 1]), gain=(peak / base - 1) * 100, off_peak=(o[j + 1] / peak - 1) * 100,
                                line=float(e200[j]), hrs=(t[j + 1] - t[on]) / 3600e3, slope=float((e200[j] / e200[j - 12] - 1) * 100)))
    return out


def stop_of(r, S):
    return float(min(max((r.line * 1.01 / r.entry - 1) * 100, 1.0), 4.0)) if S == "line" else float(S)


if __name__ == "__main__":
    if "--cases" in sys.argv:
        s_ms = int(pd.Timestamp("2026-09-29 20:00", tz="UTC").timestamp() * 1000)
        for p in ("MOVRUSDT", "SANDUSDT", "GTCUSDT"):
            d = ST.api5m(p); T1, H1, L1, C1 = RC.m1_all(p, s_ms)
            for kind in ("RETEST", "RETEST2", "CROSS"):
                for S, A, T in ((2.0, 5.0, 3.0), (3.0, 5.0, 3.0), ("line", 5.0, 3.0)):
                    rows = []; free = 0
                    for x in triggers(p, d, min_q24=0):
                        if x["kind"] != kind or x["t"] < max(s_ms, free):
                            continue
                        i = int(np.searchsorted(T1, x["t"]))
                        if i >= len(C1) - 2:
                            continue
                        r_ = pd.Series(x); st = stop_of(r_, S); res, mins, how = H.walk(H1[i:i + 720], L1[i:i + 720], C1[i:i + 720], x["entry"], st, A, T, slip=0.10, gap=True)
                        free = x["t"] + mins * 60_000; rows.append(f"{pd.Timestamp(x['t'], unit='ms'):%m-%d %H:%M} {res - 0.11:+.1f}% {how} {mins}m")
                    tot = sum(float(r.split()[2].rstrip('%')) for r in rows)
                    print(f"{p[:-4]} {kind} stop {S} trail {A:g}/{T:g}: N={len(rows)} total {tot:+.1f}% | " + " | ".join(rows))
        sys.exit(0)
    tr = []
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) >= 288 * 40:
            tr += triggers(pair, d)
    TR = pd.DataFrame(tr).sort_values("t").reset_index(drop=True); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d")
    print(len(TR), "triggers", TR.kind.value_counts().to_dict(), flush=True)
    paths = {}
    for n_, r in enumerate(TR.itertuples()):
        h, l, c = B.bars1m(r.pair, r.t)
        if len(c) >= 30:
            paths[r.Index] = (h, l, c)
        if n_ % 500 == 0:
            print(n_, flush=True)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in TR.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None

    def net(G, S, A, T):
        rows = []; free = {}
        for r in G.itertuples():
            if r.t < free.get(r.pair, 0) or r.Index not in paths:
                continue
            res, mins, how = H.walk(*paths[r.Index], r.entry, stop_of(r, S), A, T, slip=0.10, gap=True); free[r.pair] = r.t + mins * 60_000; rows.append((r.Index, res, mins, how))
        E = pd.DataFrame(rows, columns=["i", "r", "mins", "how"]).set_index("i"); X = G.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how
        X["net"] = E.r - B.COST + np.array([0.0 if FR.get(r.pair) is None else FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()])
        return X

    L = ["# 📉 EMA200 short on frenzy pairs — retest rejection and EMA50/EMA200 cross (frozen before reading)", "",
         f"{len(TR):,} triggers ({TR.kind.isin(('RETEST', 'RETEST2')).sum():,} retests · {(TR.kind == 'CROSS').sum():,} crosses) on {TR.pair.nunique()} pairs, Jan–Sep 2026. % of position at 1×, strict ruler.", ""]
    passed = 0; cells = 0
    for kind in ("RETEST", "RETEST2", "CROSS"):
        for lab, lo_, hi_ in (("run +50–200 %", 50, 200), ("run > +200 %", 200, 1e9)):
            G = TR[(TR.kind == kind) & (TR.gain >= lo_) & (TR.gain < hi_)]
            L += [f"## {kind} · {lab} — {len(G)} triggers on {G.pair.nunique()} pairs", "",
                  "| Stop | Trail | trades | won | stopped | avg win / loss | per trade | Jan–Apr / May–Sep | 95 % by day | best 5 % removed | longest losing run | PASS |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for S, A, T in EXITS:
                X = net(G, S, A, T); cells += 1
                if len(X) < 30:
                    L.append(f"| {S} | {A:g}/{T:g} | {len(X)} | – | | | | | | | | |"); continue
                bd = H.boot(X, "day", 1500); a1, a2 = X[X.day < B.SPLIT].net.mean(), X[X.day >= B.SPLIT].net.mean(); s = X.net.sort_values(); ok = a1 > 0 and a2 > 0 and bd[0] > 0; passed += ok
                run_ = max((len(z) for z in "".join("L" if v <= 0 else "W" for v in X.sort_values("t").net).split("W")), default=0)
                L.append(f"| {'above the line' if S == 'line' else f'−{S:g} %'} | {A:g} / {T:g} | {len(X)} | {(X.net > 0).mean() * 100:.0f}% | {(X.how == 'stop').mean() * 100:.0f}% | {X[X.net > 0].net.mean():+.1f} / {X[X.net <= 0].net.mean():+.1f} | "
                         f"{X.net.mean():+.2f} | {a1:+.2f} / {a2:+.2f} | [{bd[0]:+.2f}, {bd[1]:+.2f}] | {s.iloc[:-max(1, int(len(s) * 0.05))].mean():+.2f} | {run_} | {'✅' if ok else '—'} |")
            L.append("")
    X = net(TR[TR.kind.isin(("RETEST", "RETEST2"))], 3.0, 5.0, 3.0)
    L += [f"## Cells passing: {passed} of {cells}", "", "**RETEST cuts (stop 3 %, trail 5/3 — read after the fact, descriptive)**", "", "| Cut | trades · per trade (Jan–Apr / May–Sep) |", "|---|---|"]
    hv = lambda v: f"{v[v.day < B.SPLIT].net.mean():+.2f} / {v[v.day >= B.SPLIT].net.mean():+.2f}"
    for name, col, edges in (("EMA200 slope over the last hour %", "slope", (-99, -0.5, -0.2, 0, 99)), ("Hours since the spike", "hrs", (4, 24, 48, 72, 96.01)), ("Entry vs the run's peak %", "off_peak", (-100, -50, -35, -20, 0.01))):
        for a, b in zip(edges, edges[1:]):
            v = X[(X[col] >= a) & (X[col] < b)]; L.append(f"| {name} {a:g} to {b:g} | " + (f"{len(v)} · {v.net.mean():+.2f} ({hv(v)}) |" if len(v) >= 30 else f"{len(v)} · – |"))
    L += ["", "## NOT tested", "", "- Delisted pairs; slippage beyond 0.10 %; exchange position limits; no fresh time period.", "- Retests on one pair in one episode are not independent — the by-day range is the read."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
