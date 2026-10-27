#!/usr/bin/env python3
"""FADE REGIME W/L study 2026-10-26 — cohort loader (research only, no network, no writes except via callers).

Blocks (dedup key = (opened_at[:19], pair, direction), CLOSED only, non-probe):
  MASTER      reports/MASTER_POOL_stacked.csv  entry_strategy == SPIKE_FADE ∧ stack_keep  → outcome stack_pct (price %, leverage 1×;
              the 2× invest multiplier does not move %), stack_pnl ($ at today's ticket)
  MASTER_BLK  same file, stack_keep False (FRESHBREAK / LAGGARD / MAXVOL / BD13 / blacklist / floor) — cohort-completeness only, outcome =
              as-traded pnl_percentage (mechanism of their own era; refute-only, never in a kept-cohort statistic)
  B25         ~/Downloads/scalpars_orders_paper_2026-10-26_15-11-28.csv SPIKE_FADE CLOSED (PROVISIONAL; size boundary = paper top-up
              2026-10-26 13:50 UTC → flag post_topup; compare in %)
  BT          reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv  sleeve Spike-Fade, seeds 1/2/3 (caveat only)
Era groups follow ML_REGIME_WL_2026-10-26: G1 B1 · G2 B2 · G3 B3 · G4 B4–B11 · G5 B12–B16 · G6 B17–B24 · B25.
"""
import os
import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B25_FILE = os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-10-26_15-11-28.csv")
TOPUP = pd.Timestamp("2026-10-26 13:50:00")


def era_group(e):
    if e == "B25":
        return "B25"
    try:
        n = int(str(e).lstrip("B"))
    except ValueError:
        return "G0"
    if n == 1: return "G1 B1"
    if n == 2: return "G2 B2"
    if n == 3: return "G3 B3"
    if n <= 11: return "G4 B4-B11"
    if n <= 16: return "G5 B12-B16"
    return "G6 B17-B24"


def load(with_bt=False, with_blocked=True):
    m = pd.read_csv(os.path.join(R, "reports/MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[(m.entry_strategy == "SPIKE_FADE") & (m.status == "CLOSED") & (~m.is_probe.astype(bool))].copy()
    m["blk"] = np.where(m.stack_keep == True, "MASTER", "MASTER_BLK")
    m["pct"] = np.where(m.stack_keep == True, m.stack_pct, m.pnl_percentage).astype(float)
    m["usd"] = np.where(m.stack_keep == True, m.stack_pnl, m.pnl).astype(float)
    if not with_blocked:
        m = m[m.blk == "MASTER"]
    b = pd.read_csv(B25_FILE, low_memory=False)
    b = b[(b.entry_strategy == "SPIKE_FADE") & (b.status == "CLOSED")].copy()
    b["blk"] = "B25"; b["era"] = "B25"; b["pct"] = b.pnl_percentage.astype(float); b["usd"] = b.pnl.astype(float)
    b["stack_block_reason"] = "B25"
    A = pd.concat([m, b], ignore_index=True, sort=False)
    A["key"] = A.opened_at.astype(str).str[:19] + "|" + A.pair + "|" + A.direction.astype(str)
    A = A.drop_duplicates("key", keep="first")             # master first
    A["seed"] = 0
    if with_bt:
        d = pd.read_csv(os.path.join(R, "reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv"), low_memory=False)
        d = d[(d.sleeve == "Spike-Fade") & (~d.is_probe.astype(bool))].copy()
        d["blk"] = "BT"; d["era"] = d.regime; d["pct"] = d.pct.astype(float)
        d["usd"] = d.usd_new.astype(float) / d.cell_multiplier.astype(float)   # 1× $ on the $2,900 replay book
        d["key"] = d.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + d.pair + "|SHORT|s" + d.seed.astype(str)
        A = pd.concat([A, d], ignore_index=True, sort=False)
    A["t"] = pd.to_datetime(A.opened_at.astype(str).str[:19].str.replace(" ", "T"), format="%Y-%m-%dT%H:%M:%S")
    A["day"] = A.t.dt.strftime("%Y-%m-%d")
    A["eg"] = [era_group(e) if bk != "BT" else str(e) for e, bk in zip(A.era, A.blk)]
    A["win"] = A.pct > 0
    A["post_topup"] = (A.blk == "B25") & (A.t >= TOPUP)
    A["is_stop"] = A.close_reason.astype(str).str.startswith("STOP_LOSS") | (A.pct <= -1.45)
    return A.sort_values("t").reset_index(drop=True)


if __name__ == "__main__":
    A = load(with_bt=True)
    print(A.groupby(["blk", "eg"]).agg(n=("pct", "size"), wr=("win", "mean"), avg=("pct", "mean"), days=("day", "nunique")).to_string())
