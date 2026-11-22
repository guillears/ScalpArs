#!/usr/bin/env python3
"""ROUTE_B_PEAK_STUDY 2026-11-21 — look-ahead check: every variable at decision minute k must be identical when the panel is TRUNCATED
after minute k (no row after k may influence it). 3 random decision minutes × every trade. Usage: venv/bin/python -I scripts/route_b_peak_1121_lagcheck.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_b_peak_1121_eval as EV  # noqa: E402


def main():
    D = pd.read_pickle(os.path.join(EV.DATA, "panels.pkl"))
    rng = np.random.default_rng(7)
    checked = bad = 0
    badv = {}
    for pn in D["panels"]:
        if not pn.get("ok"):
            continue
        full = EV.build_trade(pn)
        if full is None:
            continue
        di = np.where(full["dec"])[0]
        for k in rng.choice(di, size=min(3, len(di)), replace=False):
            p2 = dict(pn)
            p2["P"] = pn["P"].iloc[: k + 1].copy()
            tr = EV.build_trade(p2)
            if tr is None:
                continue
            for v in full["V"]:
                a = full["V"][v][k] & full["avail"][v][k]
                b = tr["V"][v][k] & tr["avail"][v][k]
                checked += 1
                if a != b:
                    bad += 1
                    badv[v] = badv.get(v, 0) + 1
    print(f"lag check: {checked} (trade, minute, variable) values · mismatches {bad} {badv}")


if __name__ == "__main__":
    main()
