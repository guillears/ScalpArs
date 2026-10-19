#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — build gross paths (1 s tick bars, then 1m) for a list of entries.
Input csv: key, pair, te (ms), E. Output: <out>.pkl = {key: path dict} (study_frenzy_realfills_common.gross_path).
Usage: venv/bin/python scripts/study_frenzy_realfills_paths.py <entries.csv> <out.pkl> [workers] [horizon_min]"""
import os, sys
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402

H = int(sys.argv[4]) if len(sys.argv) > 4 else 720


def one(a):
    k, pair, te, E = a
    try:
        return k, RC.gross_path(pair, int(te), float(E), H)
    except Exception as e:  # noqa: BLE001
        return k, {"err": repr(e)}


if __name__ == "__main__":
    X = pd.read_csv(sys.argv[1])
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    with Pool(nw) as p:
        res = dict(p.map(one, [(r.key, r.pair, r.te, r.E) for r in X.itertuples()], chunksize=2))
    pd.to_pickle(res, sys.argv[2])
    bad = [k for k, v in res.items() if v is None or "err" in v]
    print("done", len(res), "bad", len(bad), bad[:5])
