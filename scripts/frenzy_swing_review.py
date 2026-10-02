#!/usr/bin/env python3
"""🔪 Hostile review of the FRENZY swing short (frenzy_short_test.py: SHORT at the first 5m bar the frenzy flag is off, hold 24 h,
10 % stop; passed the pre-declared bar at +2.35 %/trade by day). Every check below tries to break it.
  1 funding      real Binance funding history (public), short receives +rate / pays −rate while open
  2 fills        entry at the bar OPEN, stop filled at max(stop, the open of the breaching bar) + extra stop slippage stress
  3 nulls        the same exit on (a) every eligible pair at the same timestamps (market drift), (b) the same pair at random
                 non-frenzy times, (c) every frenzy bar (does waiting for the switch-off matter?), (d) fixed delays after onset
  4 robustness   leave-one-month-out, week-clustered bootstrap, top-tail removal, which condition switched off, flicker re-entries
  5 portfolio    concurrent positions, losing streak, drawdown at 1× per trade
Usage: venv/bin/python scripts/frenzy_swing_review.py → reports/FRENZY_SWING_REVIEW_2026-10-02.md"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import frenzy_dip_test as F  # noqa: E402
sys.argv = _a
H = F.H; COST = 0.11; N = 288; STOP = 10.0
FUND = os.path.join(H.ROOT, "reports", "backtest_cache", "funding"); OUT = os.path.join(H.ROOT, "reports", "FRENZY_SWING_REVIEW_2026-10-02.md")
rng = np.random.default_rng(7)


def funding(pair, s, e):
    f = os.path.join(FUND, pair + ".csv")
    if not os.path.exists(f):
        rows = []; cur = s
        while cur < e:
            r = H.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={pair}&startTime={cur}&endTime={e}&limit=1000") or []
            rows += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in r]; time.sleep(0.7)
            if len(r) < 1000:
                break
            cur = int(r[-1]["fundingTime"]) + 1
        os.makedirs(FUND, exist_ok=True); pd.DataFrame(rows, columns=["t", "rate"]).to_csv(f, index=False)
    return pd.read_csv(f)


def outcome(o, h, c):
    """per bar i: SHORT at o[i], 24 h, 10 % stop (gap-aware) → (result %, bars held)."""
    n = len(c); res = np.full(n, np.nan); held = np.full(n, N)
    hv = np.lib.stride_tricks.sliding_window_view(h, N) if n >= N else np.empty((0, N))
    m = len(hv); e = o[:m]; breach = hv >= (e * (1 + STOP / 100))[:, None]; anyb = breach.any(1); k = breach.argmax(1)
    res[:m] = (1 - c[np.arange(m) + N - 1] / e) * 100
    fill = np.maximum(e * (1 + STOP / 100), o[np.minimum(np.arange(m) + k, n - 1)])
    res[:m][anyb] = ((1 - fill / e) * 100)[anyb]; held[:m][anyb] = k[anyb] + 1
    return res, held


if __name__ == "__main__":
    tr, null_same, null_frz, delay, mkt = [], [], [], [], {}; frames = {}
    for f in sorted(glob.glob(os.path.join(H.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT"):
            continue
        try:
            d5, eps = F.frame(pair)
        except Exception:
            continue
        if d5 is None:
            continue
        o, h, c, t = d5.o.values, d5.h.values, d5.c.values, d5.index.values; res, held = outcome(o, h, c)
        q24 = d5.qvol.rolling(N).sum().shift(1).values
        frames[pair] = (t, res, q24)
        if not eps:
            continue
        fz = d5.frenzy.values; free = 0
        q = d5.qvol.rolling(N).sum(); norm = q.shift(N * 2).rolling(N * 30).median(); pc = d5.c.shift(1)
        trr = np.maximum(d5.h - d5.l, np.maximum((d5.h - pc).abs(), (d5.l - pc).abs()))
        volx = (q / norm).shift(1).values; r24 = ((d5.c / d5.c.shift(N) - 1) * 100).shift(1).values; atr = (trr.ewm(alpha=1 / 14, adjust=False).mean() / d5.c * 100).shift(1).values
        ep_start = {a: None for a, _ in eps}
        for i in np.nonzero(fz[:-1] & ~fz[1:])[0] + 1:
            if i < free or np.isnan(res[i]):
                continue
            why = "ATR<2" if atr[i] < 2 else ("24h<30%" if r24[i] < 30 else ("vol<100x" if volx[i] < 100 else "vol$"))
            j = i - 1
            while j > 0 and fz[j]:
                j -= 1
            tr.append(dict(pair=pair, t=int(t[i]), r=res[i], held=int(held[i]), why=why, run_bars=i - 1 - j, volx=volx[i], r24=r24[i], atr=atr[i], q24=q24[i],
                           back6=bool(fz[i + 1:i + 73].any()))); free = i + N
        ok = ~np.isnan(res); nf = ok.copy()
        for a, b in eps:
            nf &= ~((t >= a - 3 * 86400_000) & (t <= b + 3 * 86400_000))
        idx = np.nonzero(nf & (q24 >= 20e6))[0]
        null_same += [(pair, int(t[i]), res[i]) for i in (rng.choice(idx, min(40, len(idx)), replace=False) if len(idx) else [])]
        null_frz += [(pair, int(t[i]), res[i]) for i in np.nonzero(fz & ok)[0]]
        for a, _ in eps:
            i0 = int(np.searchsorted(t, a))
            for hrs in (0, 6, 12, 24, 48):
                i = i0 + hrs * 12
                if i < len(res) and not np.isnan(res[i]):
                    delay.append((pair, hrs, int(t[i]), res[i]))
    T = pd.DataFrame(tr); T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d"); T["mon"] = T.day.str[:7]; T["wk"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%G-%V")
    # 1 funding
    fund = []
    for r in T.itertuples():
        try:
            fr = funding(r.pair, int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6))
            fund.append(fr[(fr.t > r.t) & (fr.t <= r.t + r.held * 300_000)].rate.sum() * 100)
        except SystemExit:
            fund.append(np.nan)
    T["fund"] = fund; T["net"] = T.r - COST; T["netf"] = T.net + T.fund.fillna(0)
    # 3a market null: same exit on every eligible pair at the trade timestamps
    for ts in T.t.unique():
        v = []
        for p, (t, res, q24) in frames.items():
            i = np.searchsorted(t, ts)
            if i < len(t) and t[i] == ts and not np.isnan(res[i]) and q24[i] >= 20e6:
                v.append(res[i])
        mkt[ts] = (np.mean(v) if v else np.nan, len(v))
    T["mkt"] = T.t.map(lambda x: mkt[x][0]) - COST; T["excess"] = T.netf - T.mkt
    T.to_csv(os.path.join(H.ROOT, "reports", "backtest_cache", "frenzy_swing_trades.csv"), index=False)

    def ci(g, col, key="day"):
        dm = g.groupby(key)[col].mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n); q = H.tq(0.975, n)
        return f"{g[col].mean():+.2f} (by {key} {dm.mean():+.2f} [{dm.mean() - q * se:+.2f}, {dm.mean() + q * se:+.2f}], {n} {key}s)"

    def boot(g, col, key):
        ks = g[key].unique(); grp = {k: v[col].values for k, v in g.groupby(key)}
        m = [np.concatenate([grp[k] for k in rng.choice(ks, len(ks))]).mean() for _ in range(4000)]
        return f"[{np.percentile(m, 2.5):+.2f}, {np.percentile(m, 97.5):+.2f}]"
    L = ["# 🔪 Hostile review — FRENZY swing short (short when the frenzy flag switches off, 24 h, 10 % stop)", "",
         f"{len(T)} trades on {T.pair.nunique()} pairs, {T.day.nunique()} days, Jan–Sep 2026. Entry at the bar open, stop filled at the worse of the stop and the breaching bar's open. % of position at 1×.", "",
         "## 1 · Costs", "", "| Line | per trade |", "|---|---|",
         f"| before funding (after 0.11 %) | {ci(T, 'net')} |", f"| funding while open (short receives +) | mean {T.fund.mean():+.3f} · median {T.fund.median():+.3f} · worst {T.fund.min():+.2f} · best {T.fund.max():+.2f} · missing {int(T.fund.isna().sum())} |",
         f"| **after funding** | **{ci(T, 'netf')}** · week-clustered bootstrap {boot(T, 'netf', 'wk')} |"]
    for s in (0.5, 1.0, 2.0):
        x = T.netf - np.where(T.r <= -STOP + 1e-9, s, 0); L.append(f"| + {s} % extra slippage on every stop | {x.mean():+.2f} |")
    L += ["", "## 2 · Is it the switch-off, or just shorting?", "", "| Comparison (same exit, same costs, no funding) | N | per trade |", "|---|---|---|"]
    A = pd.DataFrame(null_same, columns=["pair", "t", "r"]); B = pd.DataFrame(null_frz, columns=["pair", "t", "r"]); D = pd.DataFrame(delay, columns=["pair", "hrs", "t", "r"])
    for nm, g in (("the swing short", T.assign(r=T.r)), ("every eligible pair at the same timestamps (market drift)", None), ("same pairs, random non-frenzy times (≥ 3 days away)", A), ("every bar INSIDE a frenzy", B)):
        if g is None:
            L.append(f"| {nm} | {int(np.mean([v[1] for v in mkt.values()]))} pairs / time | {T.mkt.mean():+.2f} |"); continue
        g = g.assign(day=pd.to_datetime(g.t, unit="ms").dt.strftime("%Y-%m-%d")); pm = g.groupby("pair").r.mean().mean() - COST
        L.append(f"| {nm} | {len(g):,} | {g.r.mean() - COST:+.2f} (pair-weighted {pm:+.2f}, day-weighted {g.groupby('day').r.mean().mean() - COST:+.2f}) |")
    for hrs, g in D.groupby("hrs"):
        L.append(f"| fixed delay: {hrs} h after the frenzy starts | {len(g)} | {g.r.mean() - COST:+.2f} · won {(g.r > COST).mean() * 100:.0f}% |")
    L += [f"| **excess over the market, after funding** | {len(T)} | **{ci(T, 'excess')}** · week bootstrap {boot(T, 'excess', 'wk')} |", "",
          "## 3 · Robustness", "", "| Cut | N | won | per trade after funding |", "|---|---|---|---|"]
    for m, g in T.groupby("mon"):
        L.append(f"| {m} | {len(g)} | {(g.netf > 0).mean() * 100:.0f}% | {g.netf.mean():+.2f} · without this month: {T[T.mon != m].netf.mean():+.2f} |")
    s = T.netf.sort_values(); k = max(1, int(len(s) * 0.05))
    L += [f"| best 5 % of trades removed | {len(s) - k} | | {s.iloc[:-k].mean():+.2f} |", f"| best 10 % removed | {len(s) - 2 * k} | | {s.iloc[:-2 * k].mean():+.2f} |"]
    for w, g in T.groupby("why"):
        L.append(f"| switched off because {w} | {len(g)} | {(g.netf > 0).mean() * 100:.0f}% | {g.netf.mean():+.2f} |")
    for lab, g in (("frenzy came BACK within 6 h (flicker)", T[T.back6]), ("frenzy did not come back within 6 h", T[~T.back6]), ("frenzy had lasted < 1 h", T[T.run_bars < 12]), ("1–6 h", T[(T.run_bars >= 12) & (T.run_bars < 72)]), ("≥ 6 h", T[T.run_bars >= 72])):
        L.append(f"| {lab} | {len(g)} | {(g.netf > 0).mean() * 100:.0f}% | {g.netf.mean():+.2f} |")
    for a, b in ((20e6, 100e6), (100e6, 400e6), (400e6, 1e13)):
        g = T[(T.q24 >= a) & (T.q24 < b)]; L.append(f"| 24 h volume ${a / 1e6:.0f}M–{'∞' if b > 1e12 else f'${b / 1e6:.0f}M'} | {len(g)} | {(g.netf > 0).mean() * 100:.0f}% | {g.netf.mean():+.2f} |")
    p = T.groupby("pair").netf.sum(); dd = T.groupby("day").netf.sum()
    L += ["", "## 4 · Portfolio", ""]
    ev = sorted([(r.t, 1) for r in T.itertuples()] + [(r.t + r.held * 300_000, -1) for r in T.itertuples()]); cur = mx = 0; over3 = 0
    for _, x in ev:
        cur += x; mx = max(mx, cur)
    S = T.sort_values("t"); eq = S.netf.cumsum(); streak = mx_s = 0
    for v in S.netf:
        streak = streak + 1 if v < 0 else 0; mx_s = max(mx_s, streak)
    L += [f"- stopped out: {(T.r <= -STOP + 1e-9).mean() * 100:.0f} % of trades (avg stop fill {T[T.r <= -STOP + 1e-9].r.mean():.2f} %); the rest average {T[T.r > -STOP + 1e-9].netf.mean():+.1f} %.",
          f"- max positions open at once: {mx}; longest losing streak: {mx_s} trades; deepest drawdown of the running sum: {(eq - eq.cummax()).min():.0f} points (total {eq.iloc[-1]:+.0f}).",
          f"- pairs positive {(p > 0).mean() * 100:.0f} % of {len(p)}; best 3 pairs {p.nlargest(3).sum():+.0f} of {p.sum():+.0f}; worst day {dd.min():+.0f}, best day {dd.max():+.0f}; days positive {(dd > 0).mean() * 100:.0f} %."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
