"""BTC spike → do alts follow? Event study on 5m klines (Jan–Sep 2026). Event = BTC 30-min return crosses ≥ +TH % (first bar
of the move; ≥ 4 h since the previous event). For each alt at the event bar: its own 30-min return → LAGGARD (< 0.3× BTC's),
FOLLOWER, LEADER (> BTC's). Entry = next bar open. Outcomes: forward return at 15/30/60 min, and a stack-lite exit over 2 h
(SL −0.70 net, arm +0.40, then exit at max(0.10, 0.5×peak), fee 0.09). Units: EVENTS (one BTC spike = one observation)."""
import glob, os, sys
import numpy as np, pandas as pd
K = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "backtest_cache", "k5m_full")
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
FEE = 0.09
def load(p):
    d = pd.read_csv(f"{K}/{p}.csv", usecols=["open_time", "o", "h", "l", "c", "qvol"]); d = d.drop_duplicates("open_time").set_index("open_time").sort_index()
    return d
btc = load("BTCUSDT"); r30 = btc.c / btc.c.shift(6) - 1
ev, last = [], -10**18
for t, v in r30.items():
    if v * 100 >= TH and (r30.shift(1).get(t, 0) or 0) * 100 < TH and t - last >= 4 * 3600_000:
        ev.append(t); last = t
ev = [t for t in ev if t >= 1767225600000]   # from 2026-01-01
print(f"BTC spike events (30-min ≥ +{TH}%): {len(ev)}")
# universe: liquid alts = top 60 by median quote volume
pairs = [os.path.basename(f)[:-4] for f in glob.glob(f"{K}/*.csv")]
vol = {}
for p in pairs:
    if p == "BTCUSDT": continue
    try:
        vol[p] = pd.read_csv(f"{K}/{p}.csv", usecols=["qvol"]).qvol.median()
    except Exception: pass
univ = [p for p, _ in sorted(vol.items(), key=lambda x: -x[1])[:60]]
rows = []
for p in univ:
    d = load(p); idx = d.index
    for t in ev:
        if t not in idx or t - 6 * 300_000 not in idx: continue
        i = idx.get_loc(t)
        if i + 25 >= len(d): continue
        own30 = d.c.iloc[i] / d.c.iloc[i - 6] - 1; b30 = r30.loc[t]
        e = d.o.iloc[i + 1]
        fw = {h: (d.c.iloc[i + h // 5] / e - 1) * 100 - FEE for h in (15, 30, 60)}
        peak, res = 0.0, None
        for h_, l_ in zip(d.h.iloc[i + 1:i + 25].values, d.l.iloc[i + 1:i + 25].values):
            hi, lo = (h_ / e - 1) * 100 - FEE, (l_ / e - 1) * 100 - FEE
            stop = -0.70 if peak < 0.40 else max(0.10, 0.5 * peak)
            if lo <= stop: res = stop; break
            peak = max(peak, hi)
        if res is None: res = (d.c.iloc[i + 24] / e - 1) * 100 - FEE
        cls = "LAGGARD" if own30 < 0.3 * b30 else ("LEADER" if own30 > b30 else "FOLLOWER")
        rows.append(dict(t=t, pair=p, cls=cls, own30=own30 * 100, btc30=b30 * 100, f15=fw[15], f30=fw[30], f60=fw[60], stk=res))
R = pd.DataFrame(rows); R["m"] = pd.to_datetime(R.t, unit="ms").dt.month
def rep(name, x):
    if len(x) == 0: print(f"{name:<34} N=0"); return
    e = x.groupby("t")[["f15", "f30", "f60", "stk"]].mean()
    print(f"{name:<34} fills={len(x):>5} events={e.shape[0]:>3} | per-event avg: 15m {e.f15.mean():+.3f} 30m {e.f30.mean():+.3f} "
          f"60m {e.f60.mean():+.3f} stack {e.stk.mean():+.3f} | events positive (stack) {(e.stk > 0).mean()*100:3.0f}% | fill WR {(x.stk > 0).mean()*100:3.0f}%")
rep("ALL alts at a BTC spike", R)
for c in ("LAGGARD", "FOLLOWER", "LEADER"): rep(c, R[R.cls == c])
lag = R[R.cls == "LAGGARD"]
for m, g in lag.groupby("m"): rep(f"  LAGGARD month {m:02d}", g)
# control: same alts, same clock time, 1 day earlier (no event)
# control: the SAME alts at the same clock time 24 h and 48 h before each event (no spike), same exits
ctl = []
evset = set(ev)
for p in univ:
    d = load(p); idx = d.index
    for t0 in ev:
        for back in (288, 576):
            t = t0 - back * 300_000
            if t not in idx or any(abs(t - x) < 4 * 3600_000 for x in (t0,)): continue
            i = idx.get_loc(t)
            if i + 25 >= len(d) or i < 6: continue
            e = d.o.iloc[i + 1]; peak, res = 0.0, None
            for h_, l_ in zip(d.h.iloc[i + 1:i + 25].values, d.l.iloc[i + 1:i + 25].values):
                hi, lo = (h_ / e - 1) * 100 - FEE, (l_ / e - 1) * 100 - FEE
                stop = -0.70 if peak < 0.40 else max(0.10, 0.5 * peak)
                if lo <= stop: res = stop; break
                peak = max(peak, hi)
            if res is None: res = (d.c.iloc[i + 24] / e - 1) * 100 - FEE
            ctl.append(dict(t=t, f15=(d.c.iloc[i + 3] / e - 1) * 100 - FEE, f30=(d.c.iloc[i + 6] / e - 1) * 100 - FEE,
                            f60=(d.c.iloc[i + 12] / e - 1) * 100 - FEE, stk=res))
rep("CONTROL: same alts, 24/48 h before", pd.DataFrame(ctl))
R.to_csv(os.path.join(os.path.dirname(K), f"btc_spike_events_{TH}.csv"), index=False)
