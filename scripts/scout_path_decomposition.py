#!/usr/bin/env python3
"""🧭 Scout — LOSER / WINNER DECOMPOSITION per sleeve (DECISION_LOG 406, operator 2026-10-28: entries and exits are read together).
Alert-only; changes nothing.

Two blocks (the three-blocks rule — the backtest block is the review's job):
  ① the CURRENT batch — every closed bot fill since sweep_separators.BATCH_START (orders exports, as traded)
  ② the MASTER — reports/MASTER_POOL_stacked.csv rows today's stack keeps (stack_keep), read AS TRADED (the path, close and stop are the
     traded ones; a re-priced % would mix two exits inside one group)
Each fill labelled from its own 1m price path by scripts/path_labels.py (entry label WORKED / NEVER + the decomposition group); the
missing 1m windows are fetched within path_labels' per-run request / weight budget (current batch first, then the master backlog
newest first) — the master table fills over a few runs. The path-label cache also feeds the separator sweep's ENTRY-label table.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import path_labels as PL  # noqa: E402
import sweep_separators as SW  # noqa: E402

DL = 406
_TRUE = ("True", "true", "1", "1.0")


def batch_fills(exports=None, since=None):
    since = since or SW.BATCH_START
    fr = []
    for f in (SW._exports() if exports is None else exports):
        try:
            d = SW._read(f)
        except Exception:
            continue
        if "entry_strategy" not in d or "opened_at" not in d:
            continue
        k = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
        d = d[(d.status.astype(str) == "CLOSED") & (d.entry_strategy.astype(str) != "MANUAL") & (k >= since)]
        if len(d):
            fr.append(d)
    if not fr:
        return pd.DataFrame()
    return pd.concat(fr, ignore_index=True).drop_duplicates(["opened_at", "pair", "direction"], keep="last").reset_index(drop=True)


def master_fills(master=SW.MASTER):
    if not master or not os.path.exists(master):
        return pd.DataFrame()
    m = SW._read(master)
    m = m[m.stack_keep.astype(str).isin(_TRUE) & (m.entry_strategy.astype(str) != "MANUAL")].copy()
    # AS TRADED (review M1 / M4): the path, the close and the stop are the traded ones — a re-priced % would mix two exits in one group
    return m.sort_values("opened_at", ascending=False).reset_index(drop=True)   # newest first: the backlog fills the recent eras first


def run(now_ms=None, exports=None, master=SW.MASTER, fetch=True):
    notes, bud = [], {}
    b = batch_fills(exports)
    L = []
    lab_b = PL.label_frame(b, fetch=fetch, now_ms=now_ms, notes=notes, budget=bud) if len(b) else b
    L += PL.report_lines(lab_b, f"① current batch (fills opened ≥ {SW.BATCH_START[:16]} UTC, as traded)") if len(b) else \
        [f"## 🧭 LOSER / WINNER DECOMPOSITION (DECISION_LOG {DL})", "", "_no closed bot fills in the current batch yet_"]
    m = master_fills(master)
    if len(m):
        lab_m = PL.label_frame(m, fetch=fetch, now_ms=now_ms, notes=notes, budget=bud)
        pend = int((lab_m.group == "PENDING").sum())
        L += [""] + PL.report_lines(lab_m, f"② master — the fills today's stack keeps ({len(m)}), % / $ / close AS TRADED"
                                           + (f"; {pend} still PENDING — 1m bars fill over the next runs" if pend else "") + ")")
    L += [""] + [f"  · {n}" for n in notes] + [""]
    return L


if __name__ == "__main__":
    print("\n".join(run(fetch="--no-fetch" not in sys.argv)))
