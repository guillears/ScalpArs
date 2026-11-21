#!/usr/bin/env python3
"""STATE_EXIT_2026-11-21 — fetch every pair-day the study needs (research only).
klines 1m (with volume + taker-buy) + metrics 5m from data.binance.vision daily zips (0 Binance weight, 8 threads, stop on 403 / 418 / 429);
days not yet published → fapi klines (used-weight header read, pause > 900). Funding: vision monthly zips + fapi for the current month.
Window per fill: [min(te − 26 h, spike − 1 h), te + hold] for klines (warm-up + the spike VWAP), [te − 2 h, te + hold] for metrics; BTCUSDT
klines on every day any fill is open.
Usage: venv/bin/python -I scripts/STATE_EXIT_2026-11-21_fetch.py"""
import os, sys, time
import concurrent.futures as cf
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_sp = importlib.util.spec_from_file_location("SEC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "STATE_EXIT_2026-11-21_common.py"))
C = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C)
HR = C.HR


def main():
    Z = pd.read_csv(os.path.join(C.D, "fills.csv"))
    kd, md, months = set(), set(), {}
    for r in Z.itertuples():
        a = r.te - 26 * HR
        if r.spike_ms == r.spike_ms:
            a = min(a, int(r.spike_ms) - HR)
        b = r.te + int(r.hold_h) * HR + 10 * C.MIN
        for ds in C.days_of(a, b):
            kd.add((r.pair, ds)); kd.add(("BTCUSDT", ds))
        for ds in C.days_of(r.te - 2 * HR, b):
            md.add((r.pair, ds))
        months.setdefault(r.pair, set()).update(pd.date_range(pd.Timestamp(a, unit="ms").normalize().replace(day=1),
                                                               pd.Timestamp(b, unit="ms"), freq="MS").strftime("%Y-%m"))
    kd = sorted(x for x in kd if not os.path.exists(os.path.join(C.K1, x[0], x[1] + ".npz")))
    md = sorted(x for x in md if not os.path.exists(os.path.join(C.MET, x[0], x[1] + ".npz")))
    print(time.strftime("%H:%M:%S"), "kline days", len(kd), "metric days", len(md), "pairs", len(months), flush=True)
    for name, fn, need in (("klines", C.fetch_kday, kd), ("metrics", C.fetch_mday, md)):
        n = 0
        with cf.ThreadPoolExecutor(8) as ex:
            for i, v in enumerate(ex.map(lambda x: fn(*x), need)):
                n += v
                if i % 500 == 0:
                    print(time.strftime("%H:%M:%S"), name, i, len(need), "new", n, "fapi weight", C._W["used"], flush=True)
        print(time.strftime("%H:%M:%S"), name, "done", n, flush=True)
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda kv: C.fetch_funding(kv[0], kv[1]), months.items()))
    print(time.strftime("%H:%M:%S"), "funding done", flush=True)


if __name__ == "__main__":
    main()
