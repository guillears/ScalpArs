#!/usr/bin/env python3
"""🐞 Oct-10 FRENZY backtest bug-hunt — fetch the LIVE window's 5m klines (2026-10-03 23:25 UTC → now) for every pair of the backtest
universe (reports/backtest_cache/k5m_full) into a SCRATCH folder (the shared k5m_full cache is not touched), so the backtest's own
cohort builder (frenzy_engine_cohort_build logic, real services.frenzy functions) can be run on the live days.

Budget (project rule): sequential, one call at a time, limit 1000 (weight 5), X-MBX-USED-WEIGHT-1M read on every reply (pause 60 s at ≥ 900),
stop on 418/429 and arm the shared cooldown reports/backtest_cache/.binance_cooldown_until; refuse to start while it is live. Resumable.
Usage: OUT=<dir> venv/bin/python scripts/study_fz_bughunt_fetch5m.py"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import json, os, sys, time, urllib.request, urllib.error
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "reports", "backtest_cache")
COOL = os.path.join(CACHE, ".binance_cooldown_until")
OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
START = 1791072300000                      # 2026-10-03 23:25 UTC = the bar after k5m_full's last bar
END = (int(time.time() * 1000) // 300_000) * 300_000 - 1   # closed bars only


def cooldown_left():
    try:
        return _BC.read_until_ms(COOL) - time.time() * 1000
    except Exception:
        return 0


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            w = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
            return json.loads(r.read()), w
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            ra = int(e.headers.get("Retry-After") or 120)
            with open(COOL, "w") as f:
                f.write(str(int(time.time() * 1000 + ra * 1000)))
            print(f"STOP {e.code} retry-after {ra}s — cooldown armed", flush=True); sys.exit(2)
        raise


def main():
    if cooldown_left() > 0:
        print("shared cooldown live — not starting"); sys.exit(1)
    pairs = sorted(f[:-4] for f in os.listdir(os.path.join(CACHE, "k5m_full")) if f.endswith(".csv"))
    tot_calls = 0
    for i, p in enumerate(pairs):
        fp = os.path.join(OUT, f"{p}.json")
        if os.path.exists(fp):
            continue
        rows, cur = [], START
        while cur < END:
            if cooldown_left() > 0:
                print("cooldown appeared — stop"); sys.exit(1)
            try:
                kl, w = get(f"https://fapi.binance.com/fapi/v1/klines?symbol={quote(p)}&interval=5m&startTime={cur}&endTime={END}&limit=1000")
            except urllib.error.HTTPError as e:
                print("HTTP", e.code, p, flush=True); kl, w = [], 0
            except Exception as e:
                print("ERR", p, str(e)[:80], flush=True); time.sleep(5); kl, w = [], 0
            tot_calls += 1
            if w >= 900:
                print(f"used weight {w} — pause 60 s", flush=True); time.sleep(60)
            else:
                time.sleep(0.4)
            if not kl:
                break
            rows += [[int(k[0]), k[1], k[2], k[3], k[4], k[5], k[7]] for k in kl]
            cur = int(kl[-1][0]) + 300_000
            if len(kl) < 1000:
                break
        with open(fp + ".tmp", "w") as f:
            json.dump(rows, f)
        os.replace(fp + ".tmp", fp)
        if i % 50 == 0:
            print(i, p, len(rows), "calls", tot_calls, "w", w, flush=True)
    print("done calls", tot_calls)


if __name__ == "__main__":
    main()
