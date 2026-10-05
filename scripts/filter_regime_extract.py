#!/usr/bin/env python3
"""FILTER x BTC-REGIME MATRIX (2026-10-05) step 1 — extract every momentum-LONG refusal + every SCAN line from the yr5 journals
(reports/backtest_cache/replay/year/journal/yr5_<chunk>_s<seed>/decisions-*.jsonl), counting only lines inside each chunk's
[start_ms, end_ms) window (meta json). Read-only research.
Out (scratch dir $S): frm_block.pkl (LONG BLOCK lines), frm_fails.pkl (LONG FAILS lines: gates, n_gates, src), frm_scan.pkl (SCAN, seed 1).
"""
import glob, json, os, sys, pickle, collections
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(ROOT, "reports/backtest_cache/replay/year/journal"); R = os.path.join(ROOT, "reports/backtest_cache/replay")
S = os.environ.get("S", "/tmp")
SCAN_KEYS = ("btc_rsi", "btc_adx", "btc_adx_prev", "btc_slope", "veto_long", "br_state", "r72", "above", "eff", "off24h", "off30d", "bull", "bear")


def _ms(s):
    return int(pd.Timestamp(s).value // 1_000_000)


def work(arg):
    f, seed, a, z = arg
    B, F, Sc = [], [], []
    C = collections.Counter()   # gate -> appearances in ANY LONG MOMENTUM fail set (inventory)
    for line in open(f):
        if '"SCAN"' in line[:60]:
            if seed != 1:
                continue
            r = json.loads(line); t = _ms(r["t"])
            if a <= t < z:
                Sc.append((t,) + tuple(r.get(k) for k in SCAN_KEYS))
            continue
        if '"dir":"LONG"' not in line:
            continue
        r = json.loads(line)
        e = r.get("e")
        if e not in ("BLOCK", "FAILS"):
            continue
        t = _ms(r["t"])
        if not (a <= t < z):
            continue
        if e == "BLOCK":
            B.append((seed, t, r.get("gate"), r.get("pair"), r.get("room")))
        else:
            for g in str(r.get("gates")).split("+"):
                C[(g, r.get("n_gates") == 1)] += 1
            if r.get("n_gates") == 1:
                F.append((seed, t, r.get("pair"), r.get("gates"), r.get("n_gates"), r.get("src")))
    return B, F, Sc, C


def main():
    tasks = []
    for d in sorted(glob.glob(f"{J}/yr5_*_s*")):
        tag = os.path.basename(d); mf = f"{R}/{tag}_meta.json"
        if not os.path.exists(mf):
            print("no meta", tag); continue
        m = json.load(open(mf)); seed = int(tag.rsplit("_s", 1)[1])
        for f in sorted(glob.glob(f"{d}/decisions-*.jsonl")):
            tasks.append((f, seed, m["start_ms"], m["end_ms"]))
    print(len(tasks), "files", flush=True)
    B, F, Sc = [], [], []; C = collections.Counter()
    with ProcessPoolExecutor(9) as ex:
        for i, (b, f, s, c) in enumerate(ex.map(work, tasks, chunksize=4)):
            B += b; F += f; Sc += s; C.update(c)
            if i % 200 == 0:
                print(i, len(B), len(F), len(Sc), flush=True)
    pd.DataFrame(B, columns=["seed", "t", "gate", "pair", "room"]).to_pickle(f"{S}/frm_block.pkl")
    pd.DataFrame(F, columns=["seed", "t", "pair", "gates", "n_gates", "src"]).to_pickle(f"{S}/frm_fails.pkl")
    pd.DataFrame(Sc, columns=["t"] + list(SCAN_KEYS)).to_pickle(f"{S}/frm_scan.pkl")
    pickle.dump(C, open(f"{S}/frm_failcount.pkl", "wb"))
    print("done", len(B), len(F), len(Sc))


if __name__ == "__main__":
    main()
