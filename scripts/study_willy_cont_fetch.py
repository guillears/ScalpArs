#!/usr/bin/env python3
"""WILLY CONTINUATION — coin 1m taker klines around every WILLY fill (engine replay seeds 1–3 + master + live).
Window [entry minute − 9, entry minute + 90) = 99 bars → one /fapi/v1/klines call, limit 99, weight 1.
Budget (SHARED with another study): ≤ 1,200 weight/min; pause while X-MBX-USED-WEIGHT-1M ≥ 700; stop on 418 / 429 and arm the shared
cooldown file reports/backtest_cache/.binance_cooldown_until (honoured before starting and before each call).
Cache reports/willy_cont/k1m/<pair>_<startms>.csv (re-runs never re-fetch). Pre-registration: reports/WILLY_CONT_PREREG.txt.
"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import os, sys, time
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
CD = os.path.join(REP, "willy_cont", "k1m")
os.makedirs(CD, exist_ok=True)
COOL = os.path.join(REP, "backtest_cache", ".binance_cooldown_until")
URL = "https://fapi.binance.com/fapi/v1/klines"
MIN = 60_000
FILLS = os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
MASTER = os.path.join(REP, "MASTER_POOL_stacked.csv")
LIVE = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-16_15-19-54 (1).csv")
PACE_S = 60.0 / 400                       # ≤ 400 req/min of our own (shared budget)
assert os.path.exists(os.path.join(REP, "WILLY_CONT_PREREG.txt")), "pre-registration missing"


def ms(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:23], format="mixed").values.astype("datetime64[ms]").astype("int64")


def cooldown_until():
    try:
        return _BC.read_until_ms(COOL)   # DL 306: unit-robust (ms)
    except Exception:  # noqa: BLE001
        return 0


def arm_cooldown(code, ra):
    try:
        secs = int(float(ra))
    except (TypeError, ValueError):
        secs = 7200 if code == 418 else 600
    v = max(int(time.time() * 1000) + secs * 1000, cooldown_until())
    tmp = f"{COOL}.{os.getpid()}.tmp"
    open(tmp, "w").write(str(v)); os.replace(tmp, COOL)


def windows():
    D = pd.read_csv(FILLS, low_memory=False, usecols=["sleeve", "status", "is_probe", "pair", "opened_at"])
    D = D[(D.sleeve == "FRENZY_WILLY") & (D.status == "CLOSED") & (~D.is_probe.astype(bool))]
    out = list(zip(D.pair, ms(D.opened_at)))
    for f in (MASTER, LIVE):
        d = pd.read_csv(f, low_memory=False)
        d = d[d.entry_strategy == "FRENZY_WILLY"]
        out += list(zip(d.pair, ms(d.opened_at.astype(str).str[:19])))
    w = sorted({(p, int(te // MIN) * MIN - 9 * MIN) for p, te in out})
    return w


def main():
    W = windows()
    todo = [(p, s) for p, s in W if not os.path.exists(os.path.join(CD, f"{p}_{s}.csv"))]
    print(f"windows {len(W)} · to fetch {len(todo)}", flush=True)
    s = requests.Session()
    n = 0
    for p, st in todo:
        cu = cooldown_until()
        if cu > time.time() * 1000:
            print(f"cooldown active until {cu} → stop"); return
        t0 = time.time()
        try:
            r = s.get(URL, params=dict(symbol=p, interval="1m", startTime=st, limit=99), timeout=15)
        except Exception as e:  # noqa: BLE001
            print("net error", p, e); time.sleep(2); continue
        if r.status_code in (418, 429):
            arm_cooldown(r.status_code, r.headers.get("Retry-After")); print("STOP", r.status_code); return
        used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
        if r.status_code == 200:
            rows = r.json()
            df = pd.DataFrame([x[:11] for x in rows], columns=["open_time", "o", "h", "l", "c", "vol", "ct", "qvol", "n", "tbv", "tbq"])
            df = df.drop(columns="ct")
            df.to_csv(os.path.join(CD, f"{p}_{st}.csv"), index=False)
        else:
            open(os.path.join(CD, f"{p}_{st}.csv"), "w").write("open_time,o,h,l,c,vol,qvol,n,tbv,tbq\n")   # delisted / invalid → empty
            print("http", r.status_code, p, r.text[:80])
        n += 1
        if n % 200 == 0:
            print(f"{n}/{len(todo)} used {used}", flush=True)
        while used >= 700:
            time.sleep(5)
            try:
                used = int(s.get("https://fapi.binance.com/fapi/v1/time", timeout=10).headers.get("X-MBX-USED-WEIGHT-1M", "0"))
            except Exception:  # noqa: BLE001
                pass
        dt = time.time() - t0
        if dt < PACE_S:
            time.sleep(PACE_S - dt)
    print("done", n)


if __name__ == "__main__":
    main()
