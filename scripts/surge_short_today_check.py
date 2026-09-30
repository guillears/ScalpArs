#!/usr/bin/env python3
"""Today's check of the SURGE-SHORT candidate (DUMP trigger · top-20 · HI_ATR · entry +20 min · QUICK exit) on live Binance prices."""
import json, os, sys, time
import ccxt, numpy as np, pandas as pd
ex = ccxt.binanceusdm({"enableRateLimit": True}); FEE = 0.09; MIN = 60_000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
since = int(pd.Timestamp("2026-09-29 00:00", tz="UTC").timestamp() * 1000); today = int(pd.Timestamp("2026-09-30", tz="UTC").timestamp() * 1000)
def k5(sym):
    d = pd.DataFrame(ex.fetch_ohlcv(sym, "5m", since=since, limit=1000), columns=["t", "o", "h", "l", "c", "v"]).set_index("t"); d["qv"] = d.v * d.c
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    d["atrp"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / d.c * 100; return d
b = k5("BTC/USDT:USDT"); r30 = b.c / b.c.shift(6) - 1; vmed = b.qv.shift(1).rolling(288, min_periods=100).median()
ev = [t for t in b.index[((r30 * 100 <= -1.0) & (b.qv >= 3 * vmed)).values] if t >= today]
print("DUMP events today:", [pd.to_datetime(t, unit="ms").strftime("%H:%M") for t in ev])
if not ev: sys.exit()
t0 = ev[0]; start = t0 + 300_000 + 20 * MIN
cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))
EXCL = set((cfg.get("pair_blacklist") or "").split(",")) | {"BTCUSDT", "ETHUSDT", "USDCUSDT"}
tk = ex.fetch_tickers()
top = [p for p, _ in sorted(((s.split("/")[0] + "USDT", v.get("quoteVolume") or 0) for s, v in tk.items() if s.endswith("/USDT:USDT")), key=lambda x: -x[1]) if p not in EXCL][:20]
rows = []
for p in top:
    d = k5(p[:-4] + "/USDT:USDT")
    if t0 not in d.index: continue
    atr = float(d.atrp.loc[t0])
    w = pd.DataFrame(ex.fetch_ohlcv(p[:-4] + "/USDT:USDT", "1m", since=start, limit=40), columns=["t", "o", "h", "l", "c", "v"]).set_index("t")
    if len(w) < 5:
        rows.append(dict(pair=p, atr=round(atr, 2), selected=atr >= 1.5, result=None, peak=None)); continue
    e = w.o.iloc[0]; peak = 0.0; res = None
    for o, h, l, c in zip(w.o, w.h, w.l, w.c):
        for px in ((o, h, l, c) if c < o else (o, l, h, c)):
            v = (e / px - 1) * 100 - FEE; stop = -0.50 if peak < 0.30 else max(0.5 * peak, 0.10)
            if v <= stop: res = stop; break
            peak = max(peak, v)
        if res is not None: break
    if res is None: res = (e / w.c.iloc[min(29, len(w) - 1)] - 1) * 100 - FEE
    rows.append(dict(pair=p, atr=round(atr, 2), selected=atr >= 1.5, result=round(res, 3), peak=round(peak, 2)))
    time.sleep(0.1)
R = pd.DataFrame(rows); print(f"event {pd.to_datetime(t0, unit='ms'):%H:%M} UTC, entry {pd.to_datetime(start, unit='ms'):%H:%M} UTC"); print(R.to_string(index=False))
s = R[R.selected & R.result.notna()]
print("selected now:", list(R[R.selected].pair))
if len(s): print(f"SELECTED (ATR ≥ 1.5): n={len(s)} avg {s.result.mean():+.3f} WR {(s.result > 0).mean()*100:.0f}%  |  all top-20: avg {R.result.mean():+.3f}")
