"""B25-prep ML W/L study — minimal Binance top-up for the momentum-long fills the 10-15 impulse-feature file does not cover
(master B22/B23 fills after 2026-10-15 14:36 UTC + the provisional B25 export fills).

Fetches ONLY: BTC + fill pair 1m over [entry − 70 min, entry + 20 min] (per fill), and — for fills after the 5m cache end
(the universe pairs in reports/backtest_cache/k5m_full end 2026-10-12) — BTC + fill pair + the cached top-50 universe 5m over
[first entry − 26 h, last entry + 20 min]. Writes to reports/PREP_B25_ML_WL_fetch/{k1m,k5m_full}/ (never into backtest_cache).
Budget: public endpoint, ≤ 1,200 weight/min, reads x-mbx-used-weight-1m (pause > 700), stops on 418/429 and writes the shared
cooldown file, honours scripts/binance_cooldown.py before every call.

    venv/bin/python -W ignore scripts/study_ml_wl_b25prep_fetch.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import binance_cooldown as BC  # noqa: E402
from study_ml_wl_b25prep_data import ROOT, load_all  # noqa: E402
from study_ml_wl_1015_common import load_universe, universe_for  # noqa: E402

OUT = os.path.join(ROOT, "reports/PREP_B25_ML_WL_fetch")
URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&endTime={b}&limit={n}"
K5_END = int(pd.Timestamp("2026-10-12 04:00").value // 10**6)   # universe pairs' 5m cache ends 2026-10-12 (BTC alone to 10-25)
LAST_W = [0]


def get(sym, iv, a, b, n):
    if BC.active():
        sys.exit("shared Binance cooldown active — stopping")
    req = urllib.request.Request(URL.format(s=sym, i=iv, a=a, b=b, n=n), headers={"User-Agent": "scalpars-study"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            w = int(r.headers.get("x-mbx-used-weight-1m", "0"))
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            open(BC.SHARED, "w").write(str(int(time.time() * 1000) + 15 * 60_000))
            sys.exit(f"HTTP {e.code} — cooldown written, stopping")
        print(f"  {sym} {iv} HTTP {e.code} — skipped"); return []
    LAST_W[0] = w
    if w > 700:
        print(f"  used weight {w} — pausing 60 s", flush=True); time.sleep(60)
    time.sleep(0.25)
    return data


def save(sub, sym, rows):
    d = os.path.join(OUT, sub); os.makedirs(d, exist_ok=True)
    f = os.path.join(d, sym + ".csv")
    df = pd.DataFrame([[int(r[0]), r[1], r[2], r[3], r[4], r[5], r[7]] for r in rows],
                      columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
    if os.path.exists(f):
        df = pd.concat([pd.read_csv(f), df])
    df.drop_duplicates("open_time").sort_values("open_time").to_csv(f, index=False)


def main():
    D = load_all(with_features=True, use_new=False)
    need = D[D.btc_r15.isna() & ~D.is_probe]
    print(f"fills lacking impulse features: {len(need)}")
    days, keys = load_universe()
    n1 = 0
    for r in need.itertuples():
        t = int(r.ts.value // 10**6)
        a, b = t - 70 * 60_000, t + 20 * 60_000
        for s in ("BTCUSDT", r.pair):
            rows = get(s, "1m", a, b, 100); n1 += 1
            if rows:
                save("k1m", s, rows)
    late = need[need.ts.map(lambda x: x.value // 10**6) + 20 * 60_000 > K5_END]
    n5 = 0
    if len(late):
        a = int(late.ts.min().value // 10**6) - 26 * 3600_000
        b = int(late.ts.max().value // 10**6) + 20 * 60_000
        syms = {"BTCUSDT"} | set(late.pair)
        for r in late.itertuples():
            u, _ = universe_for(days, keys, int(r.ts.value // 10**6))
            syms |= set(u)
        for s in sorted(syms):
            rows, cur = [], a
            while cur < b:
                part = get(s, "5m", cur, b, 1000); n5 += 1
                if not part:
                    break
                rows += part
                cur = int(part[-1][0]) + 300_000
                if len(part) < 1000:
                    break
            if rows:
                save("k5m_full", s, rows)
    print(f"done: {n1} 1m calls, {n5} 5m calls, last used weight {LAST_W[0]}")


if __name__ == "__main__":
    main()
