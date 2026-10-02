#!/usr/bin/env python3
"""🔪 HOSTILE REVIEW of the one passing run-sleeve cell: SHORT the first 5m close below EMA50 after a run > +200 % (scripts/break_short_review.py).
Fixed before the run — the cell is judged on the PROPOSED exit (stop 2 %, trail from +5 % giving back 3 %) and three neighbours; nothing is re-fit.
  1 costs      stop / entry slippage 0.02, 0.10, 0.30 % and a gap-aware stop (a 1m bar that opens beyond the stop fills at its open)
  2 clusters   bootstrap by DAY, by EPISODE (pair + spike) and by PAIR; leave-one-month-out; leave-one-pair-out; share carried by the top pairs
  3 dose       run size in finer buckets (the three coarse buckets were non-monotonic: ~0 / negative / positive)
  4 mirror     LONG at the same triggers with the mirrored exit — is the break a direction or only volatility?
  5 null       same exit on RANDOM 5m bars of the same pairs in the same condition (4–96 h after the spike, run > +200 %, ≥ $50M), matched
               per pair, 500 draws — all on 5m bars so real and null use one ruler. Also: any bar already below EMA50 (no fresh break).
Usage: venv/bin/python scripts/break_short_hostile_review.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_review as B  # noqa: E402
sys.argv = _a
ST, FS = B.ST, B.FS
OUT = os.path.join(ST.ROOT, "reports", "BREAK_SHORT_HOSTILE_REVIEW_2026-10-02.md")
CELLS = ((2.0, 5.0, 3.0), (2.0, 3.0, 2.0), (3.0, 5.0, 3.0), (3.0, 3.0, 2.0)); MAIN = CELLS[0]; RNG = np.random.default_rng(7)


def walk(h, l, c, e, S, A, T, side=-1, slip=0.0, gap=False):
    """side −1 = SHORT, +1 = LONG (mirror). Stop first; trail from the best price of PRIOR bars. gap=True: a bar whose previous close is
    already beyond the level fills there (worse than the level). slip is charged on every stop / trail fill."""
    best = e
    for i in range(len(c)):
        prev = c[i - 1] if i else e
        if side < 0:
            sp = e * (1 + S / 100)
            if h[i] >= sp:
                px = max(sp, prev) if gap else sp; return (1 - px / e) * 100 - slip, i + 1, "stop"
            if (1 - best / e) * 100 >= A and h[i] >= best * (1 + T / 100):
                tp = best * (1 + T / 100); px = max(tp, prev) if gap else tp; return (1 - px / e) * 100 - slip, i + 1, "trail"
            best = min(best, l[i])
        else:
            sp = e * (1 - S / 100)
            if l[i] <= sp:
                px = min(sp, prev) if gap else sp; return (px / e - 1) * 100 - slip, i + 1, "stop"
            if (best / e - 1) * 100 >= A and l[i] <= best * (1 - T / 100):
                tp = best * (1 - T / 100); px = min(tp, prev) if gap else tp; return (px / e - 1) * 100 - slip, i + 1, "trail"
            best = max(best, h[i])
    return side * (c[-1] / e - 1) * 100, len(c), "cap"


def run(G, paths, S, A, T, **kw):
    rows = []; free = {}
    for r in G.itertuples():
        if r.t < free.get(r.pair, 0) or r.Index not in paths:
            continue
        res, mins, how = walk(*paths[r.Index], r.entry, S, A, T, **kw); free[r.pair] = r.t + mins * 60_000; rows.append((r.Index, res, mins, how))
    return pd.DataFrame(rows, columns=["i", "r", "mins", "how"]).set_index("i")


def boot(X, key, n=2000):
    g = [v.values for _, v in X.groupby(key).net]; k = len(g); m = []
    for _ in range(n):
        s = np.concatenate([g[i] for i in RNG.integers(0, k, k)]); m.append(s.mean())
    return np.percentile(m, 2.5), np.percentile(m, 97.5), k


def eligible(pair, d):
    """Every 5m bar in the cell's CONDITION (no EMA rule): 4–96 h after a spike, run so far > +200 %, 24 h volume ≥ $50M."""
    t = d.open_time.values.astype("int64"); o, h, c, q = d.o.values, d.h.values, d.c.values, d.qvol.values; n = len(c)
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); volx = (q1h / norm).values
    r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]; ema = pd.Series(c).ewm(span=50, adjust=False).mean().values
    with np.errstate(invalid="ignore"):
        lead = np.nonzero((r30 >= 5) & (volx >= 20) & (q1h.values >= 2e6))[0]
    cq = np.concatenate([[0.0], np.cumsum(q)]); out = {}; last = -10**18
    for on in lead:
        on = int(on)
        if t[on] - last < 24 * 3600_000:
            last = t[on]; continue
        last = t[on]; base = c[on - 6]; peak = h[on:min(on + 48, n)].max()
        for j in range(on + 48, min(n - 146, on + 96 * 12)):
            peak = max(peak, h[j])
            if (peak / base - 1) * 100 >= 200 and cq[j + 1] - cq[max(j - 287, 0)] >= 50e6 and j not in out:
                out[j] = bool(c[j] < ema[j])
    return t, o, h, d.l.values, c, out


if __name__ == "__main__":
    TR = pd.read_csv(os.path.join(B.BC, "break_short_triggers.csv")); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d")
    TR["month"] = TR.day.str[:7]; TR["ep"] = TR.pair + "|" + ((TR.t - TR.hrs * 3600e3) // 3600e3).astype("int64").astype(str)
    G = TR[(TR.line == 50) & (TR.gain >= 200)].copy(); ALL50 = TR[TR.line == 50].copy()
    paths = {}
    for r in ALL50.itertuples():
        h, l, c = B.bars1m(r.pair, r.t)
        if len(c) >= 30:
            paths[r.Index] = (h, l, c)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in G.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None

    def net(Gx, S, A, T, fund=True, **kw):
        E = run(Gx, paths, S, A, T, **kw); X = Gx.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how
        f = [0.0 if (not fund or FR.get(r.pair) is None) else kw.get("side", -1) * -1 * FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()]
        X["net"] = E.r - B.COST + np.array(f); return X

    hv = lambda X: f"{X[X.day < B.SPLIT].net.mean():+.2f} / {X[X.day >= B.SPLIT].net.mean():+.2f}"
    cell = lambda c: f"stop {c[0]:g} · trail {c[1]:g}/{c[2]:g}"
    L = ["# 🔪 HOSTILE REVIEW — short the first 5m close below EMA50 after a run > +200 %", "",
         f"{len(G)} triggers on {G.pair.nunique()} pairs, {G.ep.nunique()} spike episodes, {G.day.nunique()} days, Jan–Sep 2026. % of position at 1×, after 0.11 % and funding unless stated.", "",
         "## 1 · Costs — slippage on every stop / trail fill, and a gap-aware stop", "",
         "| Exit | as tested | +0.02 % | +0.10 % | +0.30 % | gap-aware + 0.10 % | halves at gap-aware + 0.10 % |", "|---|---|---|---|---|---|---|"]
    for c in CELLS:
        row = [f"{net(G, *c, slip=s).net.mean():+.2f}" for s in (0.0, 0.02, 0.10, 0.30)]; Xg = net(G, *c, slip=0.10, gap=True)
        L.append(f"| {cell(c)} | " + " | ".join(row) + f" | {Xg.net.mean():+.2f} | {hv(Xg)} |")
    L += ["", "From here on every number uses the gap-aware stop with 0.10 % slippage (the honest ruler).", "", "## 2 · Who carries it — clusters, months, pairs", "",
          "| Exit | trades | won | per trade | 95 % by day | 95 % by episode | 95 % by pair | worst if one month removed | worst if one pair removed | top 3 pairs' share of the total |", "|---|---|---|---|---|---|---|---|---|---|"]
    KEEP = {}
    for c in CELLS:
        X = net(G, *c, slip=0.10, gap=True); KEEP[c] = X; tot = X.net.sum()
        bd, be, bp = boot(X, "day"), boot(X, "ep"), boot(X, "pair")
        lom = min((X[X.month != m].net.mean(), m) for m in X.month.unique()); lop = min((X[X.pair != p].net.mean(), p) for p in X.pair.unique())
        top = X.groupby("pair").net.sum().sort_values(ascending=False)
        L.append(f"| {cell(c)} | {len(X)} | {(X.net > 0).mean() * 100:.0f}% | {X.net.mean():+.2f} | [{bd[0]:+.2f}, {bd[1]:+.2f}] ({bd[2]}) | [{be[0]:+.2f}, {be[1]:+.2f}] ({be[2]}) | [{bp[0]:+.2f}, {bp[1]:+.2f}] ({bp[2]}) | "
                 f"{lom[0]:+.2f} (−{lom[1]}) | {lop[0]:+.2f} (−{lop[1][:-4]}) | {top.head(3).sum() / tot * 100 if tot > 0 else float('nan'):.0f}% ({', '.join(p[:-4] for p in top.head(3).index)}) |")
    X = KEEP[MAIN]
    L += ["", f"**By month — {cell(MAIN)}** (trades · pairs · per trade · total)", "", "| Month | trades | pairs | per trade | total |", "|---|---|---|---|---|"]
    for m, v in X.groupby("month"):
        L.append(f"| {m} | {len(v)} | {v.pair.nunique()} | {v.net.mean():+.2f} | {v.net.sum():+.0f} |")
    pp = X.groupby("pair").net.agg(["size", "sum", "mean"]).sort_values("sum", ascending=False)
    L += ["", f"Pairs: {len(pp)} · positive total on {(pp['sum'] > 0).sum()} · best 5: " + ", ".join(f"{p[:-4]} {r['sum']:+.0f} ({int(r['size'])})" for p, r in pp.head(5).iterrows())
          + " · worst 5: " + ", ".join(f"{p[:-4]} {r['sum']:+.0f} ({int(r['size'])})" for p, r in pp.tail(5).iterrows()),
          f"Longest losing run (time order): {max((len(s) for s in ''.join('L' if v <= 0 else 'W' for v in X.sort_values('t').net).split('W')), default=0)} trades.", "",
          "## 3 · Dose — does a bigger run give a better short? (all EMA50 triggers, same ruler)", "",
          "| Run so far | " + " | ".join(cell(c) for c in CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    for a, b in ((50, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 1e9)):
        Gb = ALL50[(ALL50.gain >= a) & (ALL50.gain < b)]; row = []
        for c in CELLS:
            Xb = net(Gb, *c, fund=False, slip=0.10, gap=True); row.append(f"{len(Xb)} · {Xb.net.mean():+.2f} ({hv(Xb)}) · {Xb.pair.nunique()} pairs" if len(Xb) >= 20 else f"{len(Xb)} · –")
        L.append(f"| +{a:g}–{b:g} % | ".replace("–1e+09", "+") + " | ".join(row) + " |")
    L += ["", "(Before funding, so all six rows use one ruler.)", "", "**Other cuts of the > +200 % cell (read after the fact — descriptive only)**", "", f"| Cut ({cell(MAIN)}) | trades · per trade (Jan–Apr / May–Sep) |", "|---|---|"]
    for name, col, edges in (("Hours since the spike", "hrs", (4, 12, 24, 48, 96.01)), ("Entry vs the run's peak", "off_peak", (-100, -40, -25, -15, 0.01)), ("24 h volume $M", "q24", (50e6, 150e6, 500e6, 1e13))):
        for a, b in zip(edges, edges[1:]):
            v = X[(X[col] >= a) & (X[col] < b)]
            L.append(f"| {name} {a / (1e6 if col == 'q24' else 1):g} to {b / (1e6 if col == 'q24' else 1):g} | {len(v)} · {v.net.mean():+.2f} ({hv(v)}) |" if len(v) >= 15 else f"| {name} {a:g} to {b:g} | {len(v)} · – |")
    L += ["", "## 4 · Mirror — LONG at the same triggers with the mirrored exit", "", "| Exit | SHORT per trade (halves) | LONG per trade (halves) | LONG won |", "|---|---|---|---|"]
    for c in CELLS:
        Xs, Xl = KEEP[c], net(G, *c, slip=0.10, gap=True, side=1)
        L.append(f"| {cell(c)} | {Xs.net.mean():+.2f} ({hv(Xs)}) | {Xl.net.mean():+.2f} ({hv(Xl)}) | {(Xl.net > 0).mean() * 100:.0f}% |")
    # 5 · null on 5m bars
    EL = {}; real5 = {c: [] for c in CELLS}; nullres = {c: {} for c in CELLS}
    for p in sorted(G.pair.unique()):
        f = os.path.join(ST.K5, p + ".csv")
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time"); t, o, h, l, c5, el = eligible(p, d); EL[p] = (t, el); pos = {int(x): i for i, x in enumerate(t)}
        for cc in CELLS:
            res = {}
            for j in el:
                r_, m_, _ = walk(h[j + 1:j + 145], l[j + 1:j + 145], c5[j + 1:j + 145], o[j + 1], *cc, slip=0.10, gap=True); res[j] = (r_ - B.COST, int(t[j + 1]), m_ * 300_000)
            nullres[cc][p] = res
            for r in G[G.pair == p].itertuples():
                j = pos.get(int(r.t))
                if j is not None and j - 1 in res:
                    real5[cc].append((p, j - 1))
    L += ["", "## 5 · Null — the same exit on random bars of the same pairs in the same condition (5m bars, before funding)", "",
          f"Eligible bars: {sum(len(v[1]) for v in EL.values()):,} on {len(EL)} pairs; {np.mean([np.mean(list(v[1].values())) for v in EL.values() if v[1]]) * 100:.0f}% of them are below EMA50.", "",
          "| Exit | real triggers (5m ruler) | random bars: mean [5–95 %] | real beats random in | bars already below EMA50: mean [5–95 %] | real beats those in |", "|---|---|---|---|---|---|"]

    def one(res_by_pair, picks):
        tot = []; 
        for p, js in picks.items():
            free = 0
            for j in sorted(js):
                r_, t0, dur = res_by_pair[p][j]
                if t0 < free:
                    continue
                free = t0 + dur; tot.append(r_)
        return float(np.mean(tot)) if tot else np.nan

    for cc in CELLS:
        rp = {}
        for p, j in real5[cc]:
            rp.setdefault(p, []).append(j)
        real = one(nullres[cc], rp); outs = []
        for only_below in (False, True):
            ms = []
            for _ in range(500):
                picks = {}
                for p, js in rp.items():
                    pool = [j for j, b in EL[p][1].items() if (b or not only_below)]
                    if pool:
                        picks[p] = list(RNG.choice(pool, size=min(len(js), len(pool)), replace=False))
                ms.append(one(nullres[cc], picks))
            ms = np.array(ms); outs.append((ms.mean(), np.percentile(ms, 5), np.percentile(ms, 95), (real > ms).mean() * 100))
        L.append(f"| {cell(cc)} | {real:+.2f} ({sum(len(v) for v in rp.values())}) | {outs[0][0]:+.2f} [{outs[0][1]:+.2f}, {outs[0][2]:+.2f}] | {outs[0][3]:.0f}% of draws | {outs[1][0]:+.2f} [{outs[1][1]:+.2f}, {outs[1][2]:+.2f}] | {outs[1][3]:.0f}% of draws |")
    L += ["", "## NOT tested", "",
          "- Pairs delisted during the year are not in the 5m cache (for a short this probably hides winners, but it is unmeasured).",
          "- Exchange position limits: small pairs cap the position per leverage tier (MOVR $5k at 25×) — the sleeve's size per trade is bounded by that, not by the account.",
          "- Order-book slippage beyond the 0.10–0.30 % assumed; borrow / funding spikes are included only at the recorded funding rate.",
          "- Several pairs breaking on the same day are separate trades here; the by-day interval is the read that treats them as one observation.",
          "- The > +200 % threshold and the exit grid were chosen on this same data (72 cells × 2 lines): apply the 30–50 % haircut to any figure above."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
