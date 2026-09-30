#!/usr/bin/env python3
"""BTC-spike follow study v2 (operator, 2026-09-30: "when BTC spikes, the alts follow — a Bull-Run-like sleeve triggered by a
BTC spike"). Built to match what the operator actually traded this morning, not a generic event study.

PRE-REGISTERED PRIMARY (fixed before any result was read):
  event    = BTC 30-min return ≥ +1.0 % AND BTC 5m close ≥ its prior 24 h high (breakout) AND the event bar's BTC quote volume
             ≥ 3 × the median of the prior 288 bars; ≥ 4 h after the previous event
  universe = top 10 tradeable alts by trailing 24 h quote volume at the event (the Bull-Run universe: BTC/ETH (no-trade) and the
             pair blacklist excluded)
  entry    = next 5m open after detection (plus LATE entries +15 / +30 min, as the operator entered)
  exit     = the bot's live LONG momentum exit: stop −0.70 net; the runner arms at peak ≥ +0.40 and then exits at
             max(peak − 1.0 × entry ATR %, +0.10); 4 h max hold; fee 0.09 round trip; 5m bars, adverse extreme first
  control  = the same universe at the same clock time 1, 2 and 3 days earlier (no event)
  units    = EVENTS (one BTC spike = one observation); 95 % CI by bootstrap over events
Everything else printed (spike size 1.5 %, no volume / no breakout condition, top-20 / top-60 universe) is ROBUSTNESS.
Usage: venv/bin/python scripts/btc_spike_follow_study.py"""
import glob, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = os.path.join(ROOT, "reports", "backtest_cache", "k5m_full")
FEE, BAR = 0.09, 300_000
cfg = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))
EXCL = set((cfg.get("pair_blacklist") or "").split(",")) | set((cfg.get("no_trade_pairs") or "").split(",")) | {"BTCUSDT", "ETHUSDT"}
STABLE = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "DAIUSDT", "USDEUSDT"}


def load(p):
    d = pd.read_csv(os.path.join(K, p + ".csv"), usecols=["open_time", "o", "h", "l", "c", "qvol"])
    d = d.drop_duplicates("open_time").set_index("open_time").sort_index()
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    d["atrp"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / d.c * 100
    d["v24"] = d.qvol.rolling(288, min_periods=200).sum()
    return d


def runner_exit(d, i, hold_bars=48):
    """Enter at bar i+1 open; the live LONG momentum exit. Returns net % or None."""
    if i + 2 >= len(d):
        return None
    e = d.o.iloc[i + 1]; atr = float(d.atrp.iloc[i]); peak = 0.0
    end = min(len(d), i + 1 + hold_bars)
    for j in range(i + 1, end):
        hi = (d.h.iloc[j] / e - 1) * 100 - FEE; lo = (d.l.iloc[j] / e - 1) * 100 - FEE
        stop = -0.70 if peak < 0.40 else max(peak - 1.0 * atr, 0.10)
        if lo <= stop:
            return stop
        peak = max(peak, hi)
    return (d.c.iloc[end - 1] / e - 1) * 100 - FEE


btc = load("BTCUSDT")
r30 = btc.c / btc.c.shift(6) - 1
hi24 = btc.h.shift(1).rolling(288, min_periods=200).max()
vmed = btc.qvol.shift(1).rolling(288, min_periods=200).median()


def events(th, need_breakout, need_vol):
    out, last = [], -10**18
    ok = (r30 * 100 >= th)
    if need_breakout:
        ok &= btc.c >= hi24
    if need_vol:
        ok &= btc.qvol >= 3 * vmed
    for t in btc.index[ok.fillna(False).values]:
        if t - last >= 4 * 3600_000 and t >= 1767225600000:
            out.append(t); last = t
    return out


pairs = [os.path.basename(f)[:-4] for f in glob.glob(os.path.join(K, "*.csv"))]
pairs = [p for p in pairs if p not in EXCL and p not in STABLE and p.endswith("USDT")]
D = {}
for p in pairs:
    try:
        D[p] = load(p)
    except Exception:
        pass


def universe(t, n):
    vols = [(p, d.v24.get(t)) for p, d in D.items() if t in d.index]
    vols = [(p, v) for p, v in vols if v is not None and not np.isnan(v)]
    return [p for p, _ in sorted(vols, key=lambda x: -x[1])[:n]]


def run(evs, n, delay):
    rows = []
    for t0 in evs:
        t = t0 + delay * BAR
        for p in universe(t0, n):
            d = D[p]
            if t not in d.index:
                continue
            r = runner_exit(d, d.index.get_loc(t))
            if r is not None:
                rows.append((t0, p, r))
    return pd.DataFrame(rows, columns=["ev", "pair", "r"])


def ci(x, n=4000, seed=1):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    if len(x) < 3:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def line(name, R):
    if R.empty:
        print(f"{name:<58} no fills"); return None
    e = R.groupby("ev").r.mean(); lo, hi = ci(e.values)
    first = e[e.index < 1777593600000]; second = e[e.index >= 1777593600000]   # split at 2026-05-01
    print(f"{name:<58} events={len(e):>3} fills={len(R):>4} | per event {e.mean():+.3f} [95% {lo:+.3f},{hi:+.3f}] "
          f"positive {(e > 0).mean()*100:3.0f}% | fill WR {(R.r > 0).mean()*100:3.0f}% | Jan-Apr {first.mean():+.3f} (n{len(first)}) "
          f"May-Sep {second.mean():+.3f} (n{len(second)})")
    return e


def control(evs, n):
    rows = []
    for t0 in evs:
        for back in (1, 2, 3):
            t = t0 - back * 288 * BAR
            for p in universe(t, n):
                d = D[p]
                if t in d.index:
                    r = runner_exit(d, d.index.get_loc(t))
                    if r is not None:
                        rows.append((t0, p, r))
    return pd.DataFrame(rows, columns=["ev", "pair", "r"])


if __name__ == "__main__":
    P = events(1.0, True, True)
    print(f"PRIMARY events (BTC +1.0 %/30 min, 24 h-high breakout, volume ≥ 3× median): {len(P)}  "
          f"({pd.to_datetime(P[0], unit='ms').date() if P else '-'} → {pd.to_datetime(P[-1], unit='ms').date() if P else '-'})\n")
    print("== PRIMARY: top-10 tradeable alts, live LONG momentum exit ==")
    line("entry at detection", run(P, 10, 0))
    line("entry +15 min (late)", run(P, 10, 3))
    line("entry +30 min (late)", run(P, 10, 6))
    line("CONTROL: same universe, same clock time 1-3 days earlier", control(P, 10))
    print("\n== ROBUSTNESS (not the decision) ==")
    for name, evs in (("BTC +1.5 %, breakout + volume", events(1.5, True, True)),
                      ("BTC +1.0 %, volume only", events(1.0, False, True)),
                      ("BTC +1.0 %, breakout only", events(1.0, True, False)),
                      ("BTC +1.0 %, no condition", events(1.0, False, False))):
        line(f"{name} · top-10", run(evs, 10, 0))
    line("PRIMARY events · top-20", run(P, 20, 0))
    line("PRIMARY events · top-60", run(P, 60, 0))
    R = run(P, 10, 0)
    if not R.empty:
        R["date"] = pd.to_datetime(R.ev, unit="ms")
        print("\nPRIMARY per event (top-10, entry at detection):")
        print(R.groupby("date").r.agg(["count", "mean", lambda s: (s > 0).mean()]).rename(columns={"<lambda_0>": "WR"}).round(3).to_string())
