#!/usr/bin/env python3
"""WILLY late joiners (2026-10-22) — PRICE step: every entries.csv row with a red bar walked under today's WILLY exit (common.walk:
1m path, fill = 1m open at the red close × 1.0010, evaluate_late book live). One walk per distinct (pair, red_close).
Output: reports/WILLY_LATE_JOIN_2026-10-22_entries_priced.csv
Usage: venv/bin/python scripts/study_willy_latejoin_price.py [--procs 8]
"""
import argparse, os, sys
import multiprocessing as mp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_latejoin_common as C  # noqa: E402


def work(keys):
    return [dict(pair=p, red_close=rc, **C.walk(p, int(rc))) for p, rc in keys]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); A = ap.parse_args()
    en = pd.read_csv(os.path.join(C.D, "entries.csv"))
    ok = en[en.st == "ok"].copy(); ok["red_close"] = ok.red_close.astype("int64")
    keys = sorted(set(zip(ok.pair, ok.red_close)))
    pairs = sorted({k[0] for k in keys}); g = {p: i % (A.procs * 4) for i, p in enumerate(pairs)}
    chunks = [[k for k in keys if g[k[0]] == n] for n in range(A.procs * 4)]
    with mp.Pool(A.procs) as P:
        res = sum(P.map(work, chunks), [])
    W = pd.DataFrame(res)
    en = en.merge(W, on=["pair", "red_close"], how="left")
    en["rule_exit"] = C.RULE
    en["block"] = [C.block(x) for x in en.spike_close]; en["era"] = [C.era(x) for x in en.spike_close]
    en["day"] = pd.to_datetime(en.red_close.fillna(en.spike_close), unit="ms").dt.strftime("%Y-%m-%d")
    en.to_csv(C.OUTP + "_entries_priced.csv", index=False)
    print(W.why.value_counts(dropna=False).to_dict())
