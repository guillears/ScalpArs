#!/usr/bin/env python3
"""🔥 SECOND PUSH on frenzy pairs — operator (2026-10-02), from two hand trades on SCR (LONG 18:11 / 18:12 UTC, +1.0 % and +2.0 %): the pair
was flagged ~20 h earlier, had faded 12 % below its spike's average price on 12× volume, then pushed +10 % in 20 minutes on rising volume
(52×, then 74×), dipped 2 %, and he bought the dip. FRENZY's own long was off (below the average price, volume < 100×, ATR 2.08 %).
FROZEN before reading:
  episode   a spike (5m close, 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× normal ∧ ≥ $2M) starts it; it ends after 24 h without a staircase
            state bar — exactly the FRENZY flag (scripts/frenzy_long_followup.py)
  PUSH      inside a live episode, ≥ 4 h after its spike: a 5m close with 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× normal, with no such bar
            in the previous 2 h (a fresh burst, not the middle of a run); 24 h volume ≥ $20M
  entries   NOW  = buy the next 5m open after the push bar
            DIP  = after the push bar, the first 5m close ≥ 2 % below the highest close since the push (within 60 min) → buy the next open
  exits     take-profit +1 / +2 / +3 % with stop 3 % · trailing 3/1.5 and 5/1.5 with stop 3 % · 12 h cap (1-minute bars)
  ruler     gap-aware fills, 0.10 % slippage on stops and trails (a take-profit is a resting limit: filled at its price, no slippage),
            0.11 % costs, funding. One position per pair.
  PASS      per trade > 0 in both halves ∧ 95 % range by day above 0. 2 entries × 5 exits = 10 cells (+ the below / above average-price split).
Usage: venv/bin/python scripts/frenzy_second_push_test.py [--case PAIR]"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_hostile_review as H  # noqa: E402
sys.argv = _a
B, FS, ST = H.B, H.FS, H.ST; OUT = os.path.join(ST.ROOT, "reports", "FRENZY_SECOND_PUSH_TEST_2026-10-02.md")
EXITS = (("TP +1 % · stop 3", "tp", 1.0), ("TP +2 % · stop 3", "tp", 2.0), ("TP +3 % · stop 3", "tp", 3.0), ("trail 3/1.5 · stop 3", "tr", (3.0, 1.5)), ("trail 5/1.5 · stop 3", "tr", (5.0, 1.5)))


def pushes(pair, d, V=100.0):
    t = d.open_time.values.astype("int64"); o, h, l, c, q = d.o.values, d.h.values, d.l.values, d.c.values, d.qvol.values; n = len(c)
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); volx = (q1h / norm).values
    r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]; pc = np.r_[c[0], c[:-1]]
    atr = (pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).ewm(alpha=1 / 14, adjust=False).mean() / c * 100).values; cq = np.concatenate([[0.0], np.cumsum(q)])
    with np.errstate(invalid="ignore"):
        is_lead = (r30 >= 5) & (volx >= 20) & (q1h.values >= 2e6)
    lead = np.nonzero(is_lead)[0]; out = []; i = 0
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
        for j in range(on + 48, min(end, n - 14)):
            if not is_lead[j] or is_lead[j - 24:j].any() or cq[j + 1] - cq[max(j - 287, 0)] < 20e6:
                continue
            base = dict(pair=pair, hrs=(t[j] - t[on]) / 3600e3, vs_vwap=(c[j] / vw[j - on] - 1) * 100, volx=float(volx[j]), atr=float(atr[j]), r30=float(r30[j]),
                        run=(h[on:j + 1].max() / c[on - 6] - 1) * 100, state=bool(st[j - on]), q24=float(cq[j + 1] - cq[max(j - 287, 0)]))
            out.append(dict(base, kind="NOW", t=int(t[j + 1]), entry=float(o[j + 1])))
            hi = c[j]
            for k in range(j + 1, min(j + 13, n - 2)):
                hi = max(hi, c[k])
                if c[k] <= hi * 0.98:
                    out.append(dict(base, kind="DIP", t=int(t[k + 1]), entry=float(o[k + 1]))); break
        i = end
    return out


def walk_tp(h, l, c, e, S, TP):
    """LONG, 1m bars: stop first inside a bar (gap-aware, 0.10 % slippage); the take-profit is a resting limit (filled at its price)."""
    sp, tp = e * (1 - S / 100), e * (1 + TP / 100)
    for i in range(len(c)):
        prev = c[i - 1] if i else e
        if l[i] <= sp:
            return (min(sp, prev) / e - 1) * 100 - 0.10, i + 1, "stop"
        if h[i] >= tp:
            return TP, i + 1, "tp"
    return (c[-1] / e - 1) * 100, len(c), "cap"


if __name__ == "__main__":
    if "--case" in sys.argv:
        p = sys.argv[sys.argv.index("--case") + 1]; d = ST.api5m(p)
        for x in pushes(p, d)[-12:]:
            print(f"{p} {x['kind']} {pd.Timestamp(x['t'], unit='ms'):%m-%d %H:%M} entry {x['entry']:.5g} · {x['hrs']:.1f} h after the spike · {x['vs_vwap']:+.1f}% vs avg · vol {x['volx']:.0f}× · ATR {x['atr']:.2f} · 30m {x['r30']:+.1f}%")
        sys.exit(0)
    tr = []
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) >= 288 * 40:
            tr += pushes(pair, d)
    TR = pd.DataFrame(tr).sort_values("t").reset_index(drop=True); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d")
    print(len(TR), "entries", TR.kind.value_counts().to_dict(), flush=True)
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

    def net(G, mode, arg):
        rows = []; free = {}
        for r in G.itertuples():
            if r.t < free.get(r.pair, 0) or r.Index not in paths:
                continue
            hh, ll, cc = paths[r.Index]
            res, mins, how = walk_tp(hh, ll, cc, r.entry, 3.0, arg) if mode == "tp" else H.walk(hh, ll, cc, r.entry, 3.0, arg[0], arg[1], side=1, slip=0.10, gap=True)
            free[r.pair] = r.t + mins * 60_000; rows.append((r.Index, res, mins, how))
        E = pd.DataFrame(rows, columns=["i", "r", "mins", "how"]).set_index("i"); X = G.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how
        X["net"] = E.r - B.COST + np.array([0.0 if FR.get(r.pair) is None else -FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()])
        return X

    def row(X):
        if len(X) < 30:
            return f"{len(X)} | – | – | – | – | – | – | –", False
        bd = H.boot(X, "day", 1500); a1, a2 = X[X.day < B.SPLIT].net.mean(), X[X.day >= B.SPLIT].net.mean(); ok = a1 > 0 and a2 > 0 and bd[0] > 0
        run_ = max((len(z) for z in "".join("L" if v <= 0 else "W" for v in X.sort_values("t").net).split("W")), default=0)
        return (f"{len(X)} | {(X.net > 0).mean() * 100:.0f}% | {(X.how == 'stop').mean() * 100:.0f}% | {X.net.mean():+.2f} | {a1:+.2f} / {a2:+.2f} | [{bd[0]:+.2f}, {bd[1]:+.2f}] | "
                f"{X.mins.median():.0f} min | {run_}"), ok

    HEAD = ["| Exit | trades | won | stopped | per trade | Jan–Apr / May–Sep | 95 % by day | median hold | longest losing run | PASS |", "|---|---|---|---|---|---|---|---|---|---|"]
    L = ["# 🔥 SECOND PUSH on frenzy pairs — buy a fresh burst inside a live episode (frozen before reading)", "",
         f"{(TR.kind == 'NOW').sum():,} pushes on {TR.pair.nunique()} pairs; {(TR.kind == 'DIP').sum():,} of them gave a 2 % dip within the hour. Jan–Sep 2026, % of position at 1×, strict ruler.", ""]
    passed = cells = 0
    for kind, lab in (("NOW", "buy the push at once"), ("DIP", "buy the first 2 % dip after the push")):
        for sub, G in (("all", TR[TR.kind == kind]), ("price BELOW the spike's average price (the SCR case)", TR[(TR.kind == kind) & (TR.vs_vwap < 0)]),
                       ("price ABOVE the spike's average price", TR[(TR.kind == kind) & (TR.vs_vwap >= 0)])):
            L += [f"## {kind} ({lab}) · {sub} — {len(G)} entries on {G.pair.nunique()} pairs", ""] + HEAD
            for name, mode, arg in EXITS:
                txt, ok = row(net(G, mode, arg)); cells += 1; passed += ok
                L.append(f"| {name} | {txt} | {'✅' if ok else '—'} |")
            L.append("")
    X = net(TR[TR.kind == "DIP"], "tp", 2.0)
    hv = lambda v: f"{v[v.day < B.SPLIT].net.mean():+.2f} / {v[v.day >= B.SPLIT].net.mean():+.2f}"
    L += [f"## Cells passing: {passed} of {cells}", "", "**DIP entry, TP +2 % · stop 3 — cuts read after the fact (descriptive)**", "", "| Cut | trades · won · per trade (Jan–Apr / May–Sep) |", "|---|---|"]
    for name, col, edges in (("Hours since the spike", "hrs", (4, 12, 24, 48, 1e9)), ("Volume × normal at the push", "volx", (20, 40, 80, 160, 1e9)), ("5m ATR % at the push", "atr", (0, 1.5, 2.5, 4, 1e9)),
                             ("Price vs the spike's average price %", "vs_vwap", (-100, -10, 0, 10, 1e9)), ("Run so far %", "run", (0, 50, 100, 200, 1e9))):
        for a, b in zip(edges, edges[1:]):
            v = X[(X[col] >= a) & (X[col] < b)]
            L.append(f"| {name} {a:g} to {'+' if b >= 1e9 else f'{b:g}'} | " + (f"{len(v)} · {(v.net > 0).mean() * 100:.0f}% · {v.net.mean():+.2f} ({hv(v)}) |" if len(v) >= 30 else f"{len(v)} · – |"))
    L += ["", "## NOT tested", "", "- Delisted pairs; slippage beyond 0.10 % (a burst bar is where the book is thinnest); exchange position limits; no fresh time period.",
          "- A take-profit is assumed filled at its price whenever a 1-minute bar's high reaches it.", "- The operator's own timing inside the minute (he bought a dip by eye) is not reproducible from 5-minute closes."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
