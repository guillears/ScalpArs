#!/usr/bin/env python3
"""ML BTC-forward follow-up (2026-10-15) — fetch the ONE field the cache does not hold: taker-buy volume.

The kline cache (reports/backtest_cache*/k1m, k5m_full) stores open_time,o,h,l,c,vol,qvol only; the aggTrade cache
(ticks_q/*.npz) stores t,p,q only (no buyer-maker flag). Taker-buy volume is therefore a real gap. This script fetches
Binance USD-M klines (fields incl. taker-buy base/quote volume) for:
  phase 1: BTCUSDT, ETHUSDT, SOLUSDT 1m, 2024-12-29 -> now
  phase 2: every top-50 member (previous-day universe lists, both caches) 5m over its membership spans (± 1 day)
Writes to reports/ml_btcfwd_fetch/{k1m_taker,k5m_taker}/SYM.csv (never into reports/backtest_cache). Resumable.
Budget rule: <= 1,200 weight/min (pauses at used-weight 900), reads x-mbx-used-weight-1m, stops on 418/429 and writes
the shared cooldown file reports/backtest_cache/.binance_cooldown_until; refuses to start while it is active.

    venv/bin/python -W ignore scripts/study_ml_btcfwd_fetch.py phase1 BTCUSDT      (one process per symbol)
    venv/bin/python -W ignore scripts/study_ml_btcfwd_fetch.py phase2 <i> <n>       (worker i of n over the spans)
Parallel workers share the IP budget: every worker pauses to the next minute boundary when the shared used-weight header exceeds 950 (< 1,200).
"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports/ml_btcfwd_fetch")
COOL = os.path.join(ROOT, "reports/backtest_cache/.binance_cooldown_until")
URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&endTime={b}&limit=499"
START = int(pd.Timestamp("2024-12-29").value // 10**6)
DAY = 86_400_000
NREQ, LASTW = 0, 0
COLS = ["open_time", "o", "h", "l", "c", "vol", "qvol", "n", "tbv", "tbq"]


def cooldown_ok():
    try:
        return not _BC.active(COOL)
    except Exception:
        return True


def get(sym, iv, a, b):
    global NREQ, LASTW
    if not cooldown_ok():
        sys.exit("shared Binance cooldown active — stopping")
    req = urllib.request.Request(URL.format(s=sym, i=iv, a=a, b=b), headers={"User-Agent": "scalpars-study"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                w = int(r.headers.get("x-mbx-used-weight-1m", "0"))
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                open(COOL, "w").write(str(time.time() + 900))
                sys.exit(f"HTTP {e.code} — cooldown written, stopping")
            if e.code == 400:
                return []
            time.sleep(5)
        except Exception:
            time.sleep(5)
    else:
        return None
    NREQ += 1; LASTW = w
    if w > 950:                     # the 1m weight window resets on the minute: sleep to the next boundary (+1 s)
        nap = 61 - time.time() % 60
        print(f"  used weight {w} — pausing {nap:.0f} s", flush=True); time.sleep(nap)
    time.sleep(0.12)
    return data


def fetch_span(sym, iv, a, b, sub):
    step = 60_000 if iv == "1m" else 300_000
    d = os.path.join(OUT, sub); os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, sym + ".csv")
    done_f = os.path.join(OUT, "done.txt")
    tag = f"{sub} {sym} {a} {b}"
    if os.path.exists(done_f) and tag in open(done_f).read().splitlines():
        return
    if os.path.exists(fp):
        old = pd.read_csv(fp, usecols=["open_time"]).open_time
        inside = old[(old >= a) & (old < b)]
        if len(inside):
            a = int(inside.max()) + step          # resume an interrupted span
    rows, cur = [], a
    while cur < b:
        dd = get(sym, iv, cur, b)
        if dd is None:
            print("  request failed repeatedly — skipping rest of span", sym); break
        if not dd:
            break
        rows += dd
        nxt = int(dd[-1][0]) + step
        if nxt <= cur:
            break
        cur = nxt
        if len(rows) > 50_000:
            _save(fp, rows); rows = []
    _save(fp, rows)
    if b < int(time.time() * 1000) - 3600_000:
        open(done_f, "a").write(tag + "\n")


def _save(fp, rows):
    if not rows:
        return
    df = pd.DataFrame([[int(r[0]), r[1], r[2], r[3], r[4], r[5], r[7], int(r[8]), r[9], r[10]] for r in rows],
                      columns=COLS)
    if os.path.exists(fp):
        df = pd.concat([pd.read_csv(fp), df]).drop_duplicates("open_time")
    df.sort_values("open_time").to_csv(fp, index=False)


def spans():
    u = pd.concat([pd.read_csv(os.path.join(ROOT, f, "universe_daily_top50.csv")) for f in
                   ("reports/backtest_cache_2025", "reports/backtest_cache")]).drop_duplicates(["date_ms", "pair"])
    u = u[u.date_ms >= START]
    out = []
    for p, g in u.groupby("pair"):
        ds = sorted(g.date_ms.astype("int64"))
        # a list dated D is used on day D+1 -> need bars of D+1 (and the day before for 24 h context is not needed: 5m feats)
        s = ds[0]; prev = ds[0]
        for x in ds[1:] + [None]:
            if x is None or x - prev > 3 * DAY:
                out.append((p, s + DAY - 3600_000, prev + 2 * DAY))
                if x is not None:
                    s = x
            if x is not None:
                prev = x
    return out


def main():
    ph = sys.argv[1] if len(sys.argv) > 1 else "all"
    now = int(time.time() * 1000)
    if ph in ("phase1", "all"):
        for s in (sys.argv[2:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
            print("phase1", s, flush=True)
            fetch_span(s, "1m", START, now, "k1m_taker")
            print(f"  requests {NREQ} · last used weight {LASTW}", flush=True)
    if ph in ("phase2", "all"):
        sp = spans()
        if len(sys.argv) > 3:
            i0, nw = int(sys.argv[2]), int(sys.argv[3])
            prs = sorted({p for p, _, _ in sp}); own = set(prs[i0::nw])
            sp = [x for x in sorted(sp) if x[0] in own]
        print("phase2 spans", len(sp), "pairs", len({p for p, _, _ in sp}), flush=True)
        for i, (p, a, b) in enumerate(sp):
            fetch_span(p, "5m", a, min(b, now), "k5m_taker")
            if i % 25 == 0:
                print(f"  {i}/{len(sp)} {p} requests {NREQ} · last used weight {LASTW}", flush=True)
    print(f"DONE requests {NREQ} · last used weight {LASTW}", flush=True)


if __name__ == "__main__":
    main()
