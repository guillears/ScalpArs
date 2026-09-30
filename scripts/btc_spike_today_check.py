import json, time, os
import ccxt, numpy as np, pandas as pd
ex = ccxt.binanceusdm({"enableRateLimit": True})
cfg = json.load(open("trading_config.json"))
EXCL = set((cfg.get("pair_blacklist") or "").split(",")) | {"BTCUSDT", "ETHUSDT", "USDCUSDT"}
since = int(pd.Timestamp("2026-09-29 00:00", tz="UTC").timestamp() * 1000)
def k(sym, n=1000):
    rows = ex.fetch_ohlcv(sym, "5m", since=since, limit=n)
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).set_index("t")
    d["qv"] = d.v * d.c
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    d["atrp"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / d.c * 100
    return d
b = k("BTC/USDT:USDT")
r30 = b.c / b.c.shift(6) - 1; hi24 = b.h.shift(1).rolling(288, min_periods=100).max(); vmed = b.qv.shift(1).rolling(288, min_periods=100).median()
cand = b[(r30 * 100 >= 1.0) & (b.c >= hi24) & (b.qv >= 3 * vmed)]
print("BTC events today (primary rule):", [pd.to_datetime(t, unit="ms").strftime("%H:%M") for t in cand.index if t >= int(pd.Timestamp("2026-09-30", tz="UTC").timestamp()*1000)])
loose = b[(r30 * 100 >= 1.0)]
print("BTC bars with +1% in 30 min today:", [pd.to_datetime(t, unit="ms").strftime("%H:%M") for t in loose.index if t >= int(pd.Timestamp("2026-09-30", tz="UTC").timestamp()*1000)])
tk = ex.fetch_tickers()
vols = sorted(((s.split("/")[0] + "USDT", v.get("quoteVolume") or 0) for s, v in tk.items() if s.endswith("/USDT:USDT")), key=lambda x: -x[1])
top = [p for p, _ in vols if p not in EXCL][:10]
print("top-10 tradeable by 24h volume now:", top)
def runner(d, i, hold=48):
    e = d.o.iloc[i + 1]; atr = float(d.atrp.iloc[i]); peak = 0.0
    for j in range(i + 1, min(len(d), i + 1 + hold)):
        hi = (d.h.iloc[j] / e - 1) * 100 - 0.09; lo = (d.l.iloc[j] / e - 1) * 100 - 0.09
        stop = -0.70 if peak < 0.40 else max(peak - atr, 0.10)
        if lo <= stop: return round(stop, 3), round(peak, 3), "stop" if stop < 0 else "runner"
        peak = max(peak, hi)
    return round((d.c.iloc[min(len(d), i + hold) - 1] / e - 1) * 100 - 0.09, 3), round(peak, 3), "open/4h"
evs = [t for t in (cand.index if len(cand) else loose.index) if t >= int(pd.Timestamp("2026-09-30", tz="UTC").timestamp()*1000)]
if not evs:
    print("no event today"); raise SystemExit
t0 = evs[0]; print("using event", pd.to_datetime(t0, unit="ms"))
extra = ["QNTUSDT", "NEARUSDT", "MOVRUSDT"]
res = []
for p in top + [x for x in extra if x not in top]:
    d = k(p[:-4] + "/USDT:USDT")
    for delay in (0, 3, 6):
        t = t0 + delay * 300_000
        if t not in d.index: continue
        i = d.index.get_loc(t)
        r, pk, how = runner(d, i)
        res.append(dict(pair=p, in_top10=p in top, delay_min=delay * 5, result=r, peak=pk, how=how, atr=round(float(d.atrp.iloc[i]), 2)))
    time.sleep(0.2)
R = pd.DataFrame(res); pd.set_option("display.width", 200)
print(R.pivot_table(index=["pair", "in_top10"], columns="delay_min", values="result").round(3).to_string())
for dly in (0, 15, 30):
    x = R[(R.delay_min == dly) & R.in_top10]
    print(f"top-10, entry +{dly} min: avg {x.result.mean():+.3f}  WR {(x.result > 0).mean()*100:.0f}%  (n{len(x)})")
