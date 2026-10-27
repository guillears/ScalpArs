#!/usr/bin/env python3
"""FADE REGIME W/L study 2026-10-26 — the only network step (public Binance USDⓈ-M endpoints, ≤ 500 weight / min).

Fetches ONLY what the caches lack, into reports/FADE_REGIME_WL_2026-10-26_fetch/:
  k5m/<PAIR>.csv   5m klines for fade pairs whose 7 d history is not cached, and for the top-50 lists of 10-22 … 10-25 up to 10-26 14:00
  k1m/<PAIR>_<openms>.csv   1m klines [t − 60 min, t) per fade fill (spike geometry: closed 1m bars before the fill only)
  funding/<PAIR>.csv        funding-rate history around the fade fills
Reads X-MBX-USED-WEIGHT-1M on every response (pauses above 400), stops on 418 / 429 (arms the shared cooldown file in ms) and refuses to
start while scripts/binance_cooldown.py says a cooldown is active.
    venv/bin/python scripts/study_fade_regime_wl_fetch.py
"""
import os, sys, json, time, urllib.request, urllib.error, urllib.parse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import binance_cooldown as CD                      # noqa: E402
import study_fade_regime_wl_data as D              # noqa: E402
import study_ml_regime_wl_build as B               # noqa: E402

R = D.R
OUT = os.path.join(R, "reports/FADE_REGIME_WL_2026-10-26_fetch")
BASE = "https://fapi.binance.com"
W_PAUSE = 400
B.SRC += [os.path.join(R, "reports/backtest_cache/scout_ml_regime_lines/k5m"), os.path.join(R, "reports/ML_MATCHED_SHADOW_2026-10-26_fetch/k5m"),
          os.path.join(OUT, "k5m")]
UNI = [os.path.join(R, "reports/backtest_cache/scout_ml_regime_lines/universe_daily_top50.csv")]
END = int(pd.Timestamp("2026-10-26 14:00").value // 10**6)
DAY, BAR = 86_400_000, 300_000


class Stop(Exception):
    pass


def get(path, params, weight):
    if CD.active():
        raise Stop("shared cooldown active")
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{BASE}{path}?{q}", headers={"User-Agent": "scalpars-research"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            ra = int(e.headers.get("Retry-After", "120") or 120)
            with open(CD.SHARED, "w") as f:
                f.write(str(int(time.time() * 1000) + ra * 1000))
            raise Stop(f"HTTP {e.code}, Retry-After {ra}s — cooldown armed")
        raise
    if used > W_PAUSE:
        print(f"  used weight {used} > {W_PAUSE} → pause 60 s", flush=True); time.sleep(60)
    return body


def klines(pair, iv, start, end, step_ms):
    out = []
    s = start
    while s < end:
        n = min(1000, int((end - s) // step_ms) + 1)
        w = 1 if n <= 100 else 2 if n <= 500 else 5
        rows = get("/fapi/v1/klines", dict(symbol=pair, interval=iv, startTime=s, endTime=end - 1, limit=n), w)
        if not rows:
            break
        out += rows
        s = int(rows[-1][0]) + step_ms
        if len(rows) < n:
            break
    return pd.DataFrame([r[:11] for r in out], columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbv", "tbqv"]) if out else None


def cached_ok(pair, lo, hi):
    k = B.load5(pair)
    if k is None:
        return False
    t = k.t.values
    return ((t >= lo) & (t < hi)).sum() >= 0.97 * (hi - lo) / BAR


def main():
    os.makedirs(os.path.join(OUT, "k5m"), exist_ok=True); os.makedirs(os.path.join(OUT, "k1m"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "funding"), exist_ok=True)
    A = D.load(with_bt=False)
    A["T"] = A.t.values.astype("datetime64[ms]").astype(np.int64)
    try:
        # (a) 5m for the fade pairs: [t − 8 d, t + 2 h)
        need = {}
        for _, r in A.iterrows():
            lo, hi = int(r["T"]) - 8 * DAY, min(int(r["T"]) + 2 * 3_600_000, END)
            if not cached_ok(r.pair, lo, hi):
                a, b = need.get(r.pair, (lo, hi)); need[r.pair] = (min(a, lo), max(b, hi))
        # (b) top-50 lists dated 10-22 … 10-25 (they serve fills of 10-23 … 10-26), up to 10-26 14:00
        u = pd.concat([pd.read_csv(f) for f in UNI])
        u = u[u.date_ms >= int(pd.Timestamp("2026-10-22").value // 10**6)]
        for p in sorted(p for p in set(u.pair) - B.STABLE - {"BTCUSDT"} if __import__("re").fullmatch(r"[A-Z0-9]+USDT", p)):
            lo = int(pd.Timestamp("2026-10-15").value // 10**6)
            if not cached_ok(p, lo, END):
                a, b = need.get(p, (lo, END)); need[p] = (min(a, lo), max(b, END))
        print(f"5m pairs to fetch: {len(need)}", flush=True)
        for p, (lo, hi) in sorted(need.items()):
            f = os.path.join(OUT, "k5m", p + ".csv")
            if os.path.exists(f):
                continue
            k = klines(p, "5m", lo, hi, BAR)
            if k is not None:      # the shared cache format read by study_ml_regime_wl_build.load5 (column 6 = quote volume)
                k = k[["t", "o", "h", "l", "c", "v", "qv"]]; k.columns = ["open_time", "o", "h", "l", "c", "vol", "qvol"]
                k.to_csv(f, index=False)
            print(f"  5m {p} {0 if k is None else len(k)}", flush=True)
        # (c) 1m [t − 60 min, t) per fill
        for _, r in A.iterrows():
            T = int(r["T"]); f = os.path.join(OUT, "k1m", f"{r.pair}_{T}.csv")
            if os.path.exists(f):
                continue
            k = klines(r.pair, "1m", T - 3_600_000 - (T % 60000), T - (T % 60000) + 60000, 60000)
            if k is not None:
                k.to_csv(f, index=False)
        print("1m done", flush=True)
        # (d) funding history per pair (fills ± 4 d)
        for p, g in A.groupby("pair"):
            f = os.path.join(OUT, "funding", p + ".csv")
            if os.path.exists(f):
                continue
            lo, hi = int(g["T"].min()) - 4 * DAY, int(g["T"].max()) + 3_600_000
            rows, s = [], lo
            while s < hi:
                x = get("/fapi/v1/fundingRate", dict(symbol=p, startTime=s, endTime=hi, limit=1000), 1)
                if not x:
                    break
                rows += x; s = int(x[-1]["fundingTime"]) + 1
                if len(x) < 1000:
                    break
            pd.DataFrame([dict(t=int(x["fundingTime"]), rate=float(x["fundingRate"])) for x in rows]).to_csv(f, index=False)
        print("funding done", flush=True)
    except Stop as e:
        print("STOPPED:", e); sys.exit(2)


if __name__ == "__main__":
    main()
