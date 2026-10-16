#!/usr/bin/env python3
"""WILLY THEORY — W2 data: public Binance USDⓈ-M 1m klines (with taker-buy quote volume) covering each WILLY signal's spike bar → entry
red bar (one request, limit ≤ 99 → weight 1). Budget: ≤ 800 weight/min self-paced, X-MBX-USED-WEIGHT-1M read after every call (pause
when ≥ 900), 418/429 → stop + the shared cooldown file reports/backtest_cache/.binance_cooldown_until (honoured before starting).
Cache: reports/willy_theory/k1m_taker/<pair>_<startms>.csv (re-runs never re-fetch). Input: signals.pkl entries + master + live fills.
"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import os, sys, time, json
import numpy as np, pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_theory_common as W  # noqa: E402

CD = os.path.join(W.OUT, "k1m_taker")
os.makedirs(CD, exist_ok=True)
COOL = os.path.join(W.REP, "backtest_cache", ".binance_cooldown_until")
URL = "https://fapi.binance.com/fapi/v1/klines"
PACE_S = 60.0 / 800


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


def jobs():
    S = pd.read_pickle(W.SIG)
    S = S[S.T_ent.notna()]
    out = [(r.pair, int(r.spike_ts) - W.BAR_MS, int(r.T_ent) + W.BAR_MS) for r in S.itertuples()]
    for f, strat in ((W.MASTER, "entry_strategy"), (W.LIVE, "entry_strategy")):
        d = pd.read_csv(f, low_memory=False)
        d = d[d[strat] == "FRENZY_WILLY"]
        for r in d.itertuples():
            te = int(W.ms(pd.Series([str(r.opened_at)[:19]]))[0])
            sp = int(W.ms(pd.Series([str(r.entry_frenzy_spike_at)[:19]]))[0])
            out.append((r.pair, sp - W.BAR_MS, (te // W.BAR_MS) * W.BAR_MS))
    seg = []
    for pair, a, b in out:                      # one window ≤ 99 min; a far entry gets its own 25-min window ending at the red bar's close
        if b - a <= 99 * W.MIN:
            seg.append((pair, a, b))
        else:
            seg.append((pair, a, a + 30 * W.MIN)); seg.append((pair, b - 25 * W.MIN, b))
    return sorted(set(seg))


def main():
    J = jobs()
    if cooldown_until() > time.time() * 1000:
        sys.exit(f"cooldown active until {cooldown_until()} — not fetching")
    n_req = n_skip = n_bad = 0
    last = 0.0
    for k, (pair, a, b) in enumerate(J):
        f = os.path.join(CD, f"{pair}_{a}.csv")
        if os.path.exists(f):
            n_skip += 1; continue
        lim = int(min(99, max(1, (b - a) // W.MIN)))
        w = PACE_S - (time.time() - last)
        if w > 0:
            time.sleep(w)
        last = time.time()
        try:
            r = requests.get(URL, params=dict(symbol=pair, interval="1m", startTime=a, limit=lim), timeout=15)
        except Exception as e:  # noqa: BLE001
            print("net error", pair, str(e)[:80], flush=True); time.sleep(5); continue
        n_req += 1
        used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
        if r.status_code in (418, 429):
            arm_cooldown(r.status_code, r.headers.get("Retry-After"))
            sys.exit(f"STOP {r.status_code} at {k}/{len(J)} — cooldown armed")
        if r.status_code != 200:
            n_bad += 1
            open(f, "w").write("open_time,o,h,l,c,vol,qvol,n,tbv,tbq\n")      # empty marker (delisted / invalid symbol)
            continue
        rows = [[x[0], x[1], x[2], x[3], x[4], x[5], x[7], x[8], x[9], x[10]] for x in r.json()]
        pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c", "vol", "qvol", "n", "tbv", "tbq"]).to_csv(f, index=False)
        if used >= 900:
            print(f"used weight {used} → pause 30 s", flush=True); time.sleep(30)
        if k % 250 == 0:
            print(f"{k}/{len(J)} · req {n_req} · skip {n_skip} · bad {n_bad} · used {used}", flush=True)
    print(f"done: {len(J)} jobs · req {n_req} · cached {n_skip} · bad {n_bad}")


if __name__ == "__main__":
    main()
