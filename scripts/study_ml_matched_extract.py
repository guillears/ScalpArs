#!/usr/bin/env python3
"""ML MATCHED-LONG SHADOW study (2026-10-26) — step 1b: extract every LONG_UNMATCHED_ONLY refusal from the engine-replay
decision journals of the TODAY's-config year replays yr6 (2026-01-04 → 2026-10-11) and yr6a (2025-01-01 → 2026-01-04), 3 seeds.

Read-only on the journals. For every chunk dir yr6[a]_<chunk>_s<seed> the BLOCK lines are TRIMMED to the chunk's own window
[start_ms, end_ms) from replay/<tag>_meta.json (warm-up days repeat the previous chunk — feedback_yr5_warmup_dupes).
Each refusal carries: t, pair, room flag, the pair indicator snapshot (ctx) the engine journalled, and the BTC / market state of the
SCAN header that preceded it in the same file (btc_rsi, btc_adx, btc_adx_prev, btc_slope, off30d, bull, bear, eff, r72, above,
off24h, br_state, veto_long).

    venv/bin/python -W ignore scripts/study_ml_matched_extract.py [--jobs 8]
Output: reports/ML_MATCHED_SHADOW_2026-10-26_replay_blocks.csv.gz
"""
import argparse
import glob
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports", "backtest_cache", "replay")
JDIR = os.path.join(REP, "year", "journal")
OUT = os.path.join(ROOT, "reports", "ML_MATCHED_SHADOW_2026-10-26_replay_blocks.csv.gz")
RX = re.compile(r"^(yr6a?)_(\d{4}-\d{2}-\d{2})_s(\d+)$")
SCAN_KEYS = ("btc_rsi", "btc_adx", "btc_adx_prev", "btc_slope", "off30d", "bull", "bear", "eff", "r72", "above", "off24h",
             "br_state", "veto_long")


def one(d):
    tag = os.path.basename(d)
    m = RX.match(tag)
    meta = os.path.join(REP, f"{tag}_meta.json")
    if not m or not os.path.exists(meta):
        return []
    mj = json.load(open(meta))
    s_ms, e_ms = int(mj["start_ms"]), int(mj["end_ms"])
    files = sorted(glob.glob(os.path.join(d, "decisions-*.jsonl")))
    rows = []
    for f in files:
        p = subprocess.run(["grep", "-E", '"LONG_UNMATCHED_ONLY"|"e":"SCAN"', f], capture_output=True, text=True)
        scan = {}
        for line in p.stdout.splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("e") == "SCAN":
                scan = ev
                continue
            if ev.get("e") != "BLOCK" or ev.get("gate") != "LONG_UNMATCHED_ONLY":
                continue
            t = pd.Timestamp(ev["t"])
            t_ms = int(t.value // 10**6)
            if not (s_ms <= t_ms < e_ms):
                continue
            r = dict(prefix=m.group(1), chunk=m.group(2), seed=int(m.group(3)), tag=tag, t=ev["t"], t_ms=t_ms,
                     pair=ev.get("pair"), room=ev.get("room"), scan_t=scan.get("t"))
            for k in SCAN_KEYS:
                r["scan_" + k] = scan.get(k)
            for k, v in (ev.get("ctx") or {}).items():
                r["ctx_" + k] = v
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(JDIR, "yr6*_*")) if RX.match(os.path.basename(d)))
    print(f"{len(dirs)} chunk dirs", flush=True)
    out = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, rows in enumerate(ex.map(one, dirs)):
            out.extend(rows)
    df = pd.DataFrame(out).sort_values(["seed", "t_ms", "pair"])
    df.to_csv(OUT, index=False, compression="gzip")
    print(df.groupby(["prefix", "seed"]).size(), "\n→", OUT)


if __name__ == "__main__":
    main()
