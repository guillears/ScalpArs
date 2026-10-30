#!/usr/bin/env python3
"""📐 Stepped vs straight-line reserve table (DECISION_LOG 424). Bootstrap: 365 days from $10,000, 2,000 runs, days resampled from the master's daily stack P&L ÷ its $3,000 base, each day's P&L scaled to
the tradeable money of each table. In-sample and a simple proxy (no slots, leverage steps, or liquidity limits): compare the two columns, the levels are not a forecast. Read-only, no network.
Usage: venv/bin/python scripts/study_reserve_table_compare.py"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.trading_engine import _reserve_schedule_target as line  # noqa: E402

OLD = "10000:9500, 25000:23000, 50000:40000, 100000:75000, 150000:105000, 250000:162500, 500000:300000, 1000000:500000"
NEW = json.load(open(os.path.join(ROOT, "trading_config.json")))["investment"]["reserve_schedule"]
OLD_T = sorted(tuple(map(float, p.split(":"))) for p in OLD.split(","))


def stepped(e):
    """the DL 404 table: tier target frozen, fee reserve (2.5 %) on top."""
    t = max([v for b, v in OLD_T if e >= b], default=None)
    return t * 0.975 if t else e * 0.975


def gradual(e):
    """this table: a straight line, the larger of schedule or fee reserve."""
    t = line(NEW, e)
    return min(t, e * 0.975) if t else e * 0.975


def run(days, f, start=10000.0):
    e, p, dd = start, start, 0.0
    for x in days:
        e = max(e + x * f(e), 1.0)
        p = max(p, e)
        dd = max(dd, 1 - e / p)
    return e, dd


def main():
    m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[(m.stack_keep.astype(str) == "True") & (m.is_probe.astype(str) != "True")]
    r = m.groupby(m.closed_at.astype(str).str[:10]).stack_pnl.sum() / 3000.0
    r = r.reindex(pd.date_range(r.index.min(), r.index.max()).strftime("%Y-%m-%d")).fillna(0).values
    rng = np.random.default_rng(1)
    out = {"stepped": [], "gradual": []}
    for _ in range(2000):
        seq = rng.choice(r, 365)
        out["stepped"].append(run(seq, stepped))
        out["gradual"].append(run(seq, gradual))
    print(f"{len(r)} days · daily mean {r.mean() * 100:+.2f} %")
    for k, v in out.items():
        a = np.array(v)
        print(f"{k}: median end ${np.median(a[:, 0]):,.0f} · daily compound {(np.median(a[:, 0]) / 10000) ** (1 / 365) * 100 - 100:+.3f} % · largest drop median {np.median(a[:, 1]) * 100:.1f} % · p90 {np.percentile(a[:, 1], 90) * 100:.1f} %")


if __name__ == "__main__":
    main()
