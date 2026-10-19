#!/usr/bin/env python3
"""Scored-only screen for the <90 %-coverage BT family (rebuilt mcap, turnover): fills without a value are dropped, never 'rest'."""
import sys, os
sys.path.insert(0, "/Users/guillearslanian/Downloads/NOFA AI/scripts")
import study_fade_wl_screen as S, pandas as pd
A = S.A
for blk in ["BT", "MASTER"]:
    D = A[(A.blk == blk) & A.d_log_mcap.notna()].reset_index(drop=True)
    for y, src in [("y_pct", "pct"), ("y_stop", "is_stop"), ("y_loser", "loser")]: D[y] = D[src].astype(float)
    res = S.run_screen(blk + "_scored_mcap", D, ["d_log_mcap", "d_turnover"], ["y_pct", "y_stop", "y_loser"], two_d=False)
    for yn, (Mi, s) in res.items():
        print(Mi[["test", "n_zone", "zone_mean", "rest_mean", "z", "fw_p"]].head(4).to_string(index=False))
        Mi.to_csv(f"{S.OUT}_screen_{blk}_scoredmcap_{yn}.csv", index=False)
