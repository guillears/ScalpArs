#!/usr/bin/env python3
"""📉 BREAK SHORT review — operator (2026-10-02): "review the short, see if it is EMA200 or EMA50 or both … build the case with MOVR and SAND.
We will use 20× or more, the stop cannot be 8 %, the take-profit can be trailing." FROZEN before the run:
  episode   onset = 5m close with 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× the pair's normal hour ∧ ≥ $2M (none in the prior 24 h)
  trigger   a 5m CLOSE below the line (EMA50 or EMA200 of 5m closes) with the previous 12 closes all above it, 4–96 h after the onset,
            after the run reached ≥ +50 % above the pre-spike price; 24 h volume at entry ≥ $50M (the liquidity the long review needed)
  trade     SHORT at the next open; one position per pair; walked on futures 1-MINUTE bars for up to 12 h, stop first inside a bar
  exits     hard stop S ∈ {1.5, 2, 3} % · trailing: once the trade is A % in profit, close when price rises T % above its lowest
            point since entry — (A, T) ∈ {(1, 1), (2, 1.5), (3, 2), (5, 3)} · 12 h cap. 12 exit cells per line.
  cost      0.11 % + funding (short pays / receives the real rate while open)
  buckets   run size +50–100 % / +100–200 % / > +200 % · PASS (per line × bucket × exit): mean > 0 in both halves ∧ day-clustered
            95 % interval above 0. 2 lines × 3 buckets × 12 exits = 72 cells → a lone pass means little; the pattern across cells matters.
Usage: venv/bin/python scripts/break_short_review.py [--cases]"""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import frenzy_swing_review as FS  # noqa: E402
import staircase_swing_test as ST  # noqa: E402
sys.argv = _a
BC = os.path.join(ST.ROOT, "reports", "backtest_cache"); C1 = os.path.join(BC, "k1m_break"); OUT = os.path.join(ST.ROOT, "reports", "BREAK_SHORT_REVIEW_2026-10-02.md")
COST = 0.11; SPLIT = "2026-05-01"; STOPS = (1.5, 2.0, 3.0); TRAILS = ((1.0, 0.5), (1.0, 1.0), (2.0, 1.0), (2.0, 1.5), (3.0, 1.0), (3.0, 1.5), (3.0, 2.0), (5.0, 1.5), (5.0, 2.0), (5.0, 3.0)); HOLD = 720


def triggers(pair, d, span, G=50.0, min_q24=50e6):
    t = d.open_time.values.astype("int64"); o, h, c, q = d.o.values, d.h.values, d.c.values, d.qvol.values; n = len(c)
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); volx = (q1h / norm).values
    r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]; ema = pd.Series(c).ewm(span=span, adjust=False).mean().values
    with np.errstate(invalid="ignore"):
        lead = np.nonzero((r30 >= 5) & (volx >= 20) & (q1h.values >= 2e6))[0]
    ab = pd.Series(c >= ema).rolling(12).min().shift(1).fillna(0).values >= 1; cq = np.concatenate([[0.0], np.cumsum(q)])
    out = []; last = -10**18; seen = set()
    for on in lead:
        on = int(on)
        if t[on] - last < 24 * 3600_000:
            last = t[on]; continue
        last = t[on]; base = c[on - 6]; peak = h[on:min(on + 48, n)].max()
        for j in range(on + 48, min(n - 2, on + 96 * 12)):
            peak = max(peak, h[j])
            if c[j] < ema[j] and ab[j] and (peak / base - 1) * 100 >= G and j not in seen:
                q24 = cq[j + 1] - cq[max(j - 287, 0)]
                if q24 >= min_q24:
                    seen.add(j); out.append(dict(pair=pair, line=span, t=int(t[j + 1]), entry=float(o[j + 1]), gain=(peak / base - 1) * 100, off_peak=(o[j + 1] / peak - 1) * 100, q24=q24, hrs=(t[j + 1] - t[on]) / 3600e3))
    return out


def bars1m(pair, ms):
    f = os.path.join(C1, f"{pair}_{ms}.npz")
    if os.path.exists(f):
        z = np.load(f); return z["h"], z["l"], z["c"]
    r = None
    for k in range(4):
        try:
            r = json.load(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?symbol={urllib.request.quote(pair)}&interval=1m&startTime={ms}&limit={HOLD}", timeout=25)); break
        except Exception:
            time.sleep(2 + 2 * k)
    r = r or []; h, l, c = (np.array([float(x[i]) for x in r]) for i in (2, 3, 4))
    os.makedirs(C1, exist_ok=True); np.savez_compressed(f, h=h, l=l, c=c); time.sleep(0.07)
    return h, l, c


def walk(h, l, c, e, S, A, T):
    """SHORT from e on 1m bars → (result %, minutes, how). Stop first; trail uses the lowest low of PRIOR bars (no look-ahead)."""
    low = e
    for i in range(len(c)):
        if h[i] >= e * (1 + S / 100):
            return -S, i + 1, "stop"
        if (1 - low / e) * 100 >= A and h[i] >= low * (1 + T / 100):
            return (1 - low * (1 + T / 100) / e) * 100, i + 1, "trail"
        low = min(low, l[i])
    return (1 - c[-1] / e) * 100, len(c), "cap"


def evaluate(TR, paths, S, A, T):
    """One position per pair: a trigger inside an open trade is skipped."""
    rows = []; free = {}
    for r in TR.itertuples():
        if r.t < free.get(r.pair, 0) or r.Index not in paths:
            continue
        h, l, c = paths[r.Index]; res, mins, how = walk(h, l, c, r.entry, S, A, T)
        free[r.pair] = r.t + mins * 60_000; rows.append((r.Index, res, mins, how))
    return pd.DataFrame(rows, columns=["i", "r", "mins", "how"]).set_index("i")


if __name__ == "__main__":
    if "--cases" in sys.argv:
        for p in ("MOVRUSDT", "SANDUSDT", "GTCUSDT", "NIGHTUSDT"):
            d = ST.api5m(p)
            for span in (50, 200):
                tg = [x for x in triggers(p, d, span) if x["t"] >= (time.time() - 5 * 86400) * 1000]
                for S, A, T in ((2.0, 2.0, 1.5), (3.0, 3.0, 2.0), (2.0, 5.0, 3.0)):
                    out = []; free = 0
                    for x in tg:
                        if x["t"] < free:
                            continue
                        r = json.load(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?symbol={p}&interval=1m&startTime={x['t']}&limit={HOLD}", timeout=25))
                        h, l, c = (np.array([float(y[i]) for y in r]) for i in (2, 3, 4))
                        if len(c) < 5:
                            continue
                        res, mins, how = walk(h, l, c, x["entry"], S, A, T); free = x["t"] + mins * 60_000
                        out.append(f"{pd.Timestamp(x['t'], unit='ms'):%m-%d %H:%M} @{x['entry']:.5g} ({x['off_peak']:+.0f}% off peak) → {res:+.1f}% {how} {mins}m")
                    print(f"{p} EMA{span} · stop {S}% · trail {A}/{T}: " + (" | ".join(out) or "no trigger") + f"  || sum {sum(float(o.split('→ ')[1].split('%')[0]) for o in out):+.1f}%")
        sys.exit(0)
    tr = []
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) < 288 * 40:
            continue
        for span in (50, 200):
            tr += triggers(pair, d, span)
    TR = pd.DataFrame(tr).sort_values("t").reset_index(drop=True); print(len(TR), "triggers", TR.line.value_counts().to_dict(), flush=True)
    paths = {}
    for r in TR.itertuples():
        h, l, c = bars1m(r.pair, r.t)
        if len(c) >= 30:
            paths[r.Index] = (h, l, c)
        if r.Index % 500 == 0:
            print(r.Index, flush=True)
    TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d"); TR["bucket"] = pd.cut(TR.gain, [50, 100, 200, 1e9], labels=["+50–100 %", "+100–200 %", "> +200 %"], right=False)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in TR.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None
    L = ["# 📉 BREAK SHORT review — short the close below the EMA50 / EMA200 after a +50 % spike run, tight stop + trailing exit (1-minute bars)", "",
         f"{len(TR):,} triggers ({(TR.line == 50).sum():,} EMA50 · {(TR.line == 200).sum():,} EMA200) on {TR.pair.nunique()} pairs, ≥ $50M traded, Jan–Sep 2026. % of position at 1×, after 0.11 % and funding. "
         "At 20× multiply by 20: a 2 % stop is −42 % of the margin.", ""]
    best = []
    for span in (50, 200):
        for b in ("+50–100 %", "+100–200 %", "> +200 %"):
            G = TR[(TR.line == span) & (TR.bucket == b)]
            L += [f"## EMA{span} · run {b} · {len(G)} triggers on {G.pair.nunique()} pairs", "",
                  "| Stop | Trail (arm / give-back) | trades | won | stopped | avg win / loss | per trade Jan–Apr / May–Sep | by day [95 %] | best 5 % removed | PASS |", "|---|---|---|---|---|---|---|---|---|---|"]
            for S in STOPS:
                for A, T in TRAILS:
                    E = evaluate(G, paths, S, A, T)
                    if len(E) < 20:
                        continue
                    X = G.loc[E.index].copy(); X["mins"] = E.mins
                    fund = [0.0 if FR.get(r.pair) is None else FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()]
                    X["net"] = E.r - COST + np.array(fund); a1, a2 = X[X.day < SPLIT].net.mean(), X[X.day >= SPLIT].net.mean()
                    dm = X.groupby("day").net.mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n); l_, h_ = dm.mean() - ST.tq(n) * se, dm.mean() + ST.tq(n) * se
                    s = X.net.sort_values(); trm = s.iloc[:-max(1, int(len(s) * 0.05))].mean(); ok = a1 > 0 and a2 > 0 and l_ > 0
                    L.append(f"| −{S:g} % | {A:g} / {T:g} | {len(X)} | {(X.net > 0).mean() * 100:.0f}% | {(E.how == 'stop').mean() * 100:.0f}% | {X[X.net > 0].net.mean():+.1f} / {X[X.net <= 0].net.mean():+.1f} | {a1:+.2f} / {a2:+.2f} | {dm.mean():+.2f} [{l_:+.2f}, {h_:+.2f}] | {trm:+.2f} | {'✅' if ok else '—'} |")
                    best.append((min(a1, a2), span, b, S, A, T, len(X), ok))
            L.append("")
    best.sort(reverse=True)
    L += ["## Summary", "", f"Cells passing: {sum(1 for x in best if x[-1])} of {len(best)}. Positive in both halves: {sum(1 for x in best if x[0] > 0)} of {len(best)}.",
          "Best cells by their weaker half: " + " · ".join(f"EMA{x[1]} {x[2]} stop {x[3]:g} trail {x[4]:g}/{x[5]:g} ({x[0]:+.2f}, N={x[6]})" for x in best[:6])]
    TR.to_csv(os.path.join(BC, "break_short_triggers.csv"), index=False)
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[-4:]))
