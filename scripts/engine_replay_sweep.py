#!/usr/bin/env python3
"""Per-sleeve winner/loser separator sweep on ENGINE-REPLAY fills (year-to-date).

For every sleeve and every numeric entry_* dimension, test one-sided cuts at deciles.
A candidate BLOCK cohort (the side we would refuse) is scored by:
  * blocked cohort net < 0 overall AND in the OUT-OF-SAMPLE half (Jan-01 → Jun-16) on its own
  * month consistency: share of months (with ≥3 blocked fills) where the blocked cohort is net-negative
  * cut sensitivity: neighbouring deciles (±1) must also give a net-negative blocked cohort
  * window units: distinct days in the blocked cohort; pair concentration: top-2 pairs' share of its loss
  * cost: winners deleted and the kept sleeve's avg % / WR after the cut
Ranked by (consistency, OOS net). Also prints the EXIT anatomy per sleeve.

  venv/bin/python scripts/engine_replay_sweep.py reports/backtest_cache/replay/ALL_fills.csv
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
fp = sys.argv[1] if len(sys.argv) > 1 else "reports/backtest_cache/replay/ALL_fills.csv"
SPLIT = pd.Timestamp("2026-06-17")
o = pd.read_csv(fp, low_memory=False)
o["opened_at"] = pd.to_datetime(o.opened_at); o["closed_at"] = pd.to_datetime(o.closed_at)
o["month"] = o.opened_at.dt.strftime("%Y-%m"); o["day"] = o.opened_at.dt.date
o["hold"] = (o.closed_at - o.opened_at).dt.total_seconds() / 60
o["win"] = o.pnl > 0
o["exit"] = o.close_reason.fillna("").str.replace(r" L\d", "", regex=True)
SKIP = {"entry_price", "entry_fee", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_slippage_pct",
        "entry_pair_age_days", "entry_funding_rate"}

def agg(x):
    return f"{len(x)}·{round(x.win.mean()*100) if len(x) else 0}%·{x.pnl.sum():+.0f}$·{x.pnl_percentage.mean():+.2f}" if len(x) else "-"

def sweep(s, name):
    dims = [c for c in s.columns if c.startswith("entry_") and c not in SKIP and s[c].dtype != object
            and s[c].notna().sum() >= 0.8 * len(s) and s[c].nunique() > 8]
    tot_net = s.pnl.sum(); rows = []
    for c in dims:
        qs = s[c].quantile(np.linspace(0.1, 0.9, 9)).values
        for qi, thr in enumerate(qs):
            for side in ("<", ">="):
                blk = (s[c] < thr) if side == "<" else (s[c] >= thr)
                b = s[blk]; k = s[~blk]
                if len(b) < 20 or len(k) < 20 or b.pnl.sum() >= 0 or len(b) > 0.5 * len(s):
                    continue                       # a "filter" that blocks most of the sleeve is a sleeve verdict, not a filter
                oos = b[b.opened_at < SPLIT]
                if len(oos) < 8 or oos.pnl.sum() >= 0:
                    continue
                k_oos = k[k.opened_at < SPLIT]; k_is = k[k.opened_at >= SPLIT]
                if len(k_oos) and k_oos.pnl_percentage.mean() <= s[s.opened_at < SPLIT].pnl_percentage.mean():
                    continue                       # the KEPT side must be better out of sample, not just the blocked side worse
                mo = b.groupby("month").agg(n=("pnl", "size"), net=("pnl", "sum")); mo = mo[mo.n >= 3]
                cons = (mo.net < 0).mean() if len(mo) else 0
                # neighbours
                ok = 0
                for dq in (-1, 1):
                    j = qi + dq
                    if 0 <= j < 9:
                        t2 = qs[j]; b2 = s[(s[c] < t2) if side == "<" else (s[c] >= t2)]
                        ok += int(len(b2) >= 20 and b2.pnl.sum() < 0)
                    else:
                        ok += 1
                lossers = b[b.pnl < 0]
                top2 = lossers.groupby("pair").pnl.sum().sort_values().head(2).sum() / lossers.pnl.sum() if len(lossers) else 0
                rows.append(dict(dim=c, cut=f"{side}{thr:.3g}", blk=agg(b), blk_oos=agg(oos), months=f"{int((mo.net<0).sum())}/{len(mo)}",
                                 cons=round(cons, 2), neigh=ok, days=b.day.nunique(), top2=round(top2, 2),
                                 wins_deleted=int(b.win.sum()), kept=agg(k), kept_oos=agg(k[k.opened_at < SPLIT]),
                                 kept_is=agg(k[k.opened_at >= SPLIT]), gain=round(-b.pnl.sum())))
    r = pd.DataFrame(rows)
    if not len(r):
        print(f"\n=== {name}: no candidate survives (blocked cohort must be net-negative overall AND out-of-sample)"); return
    r = r[(r.cons >= 0.6) & (r.neigh == 2) & (r.top2 < 0.6)].sort_values(["cons", "gain"], ascending=False)
    print(f"\n=== {name}: {len(r)} candidates pass (consistency ≥0.6, neighbours agree, no pair concentration) — top 12")
    if len(r):
        print(r.head(12).to_string(index=False))

for sl, s in o.groupby("sleeve"):
    if len(s) < 40:
        continue
    print(f"\n\n##### {sl}: {agg(s)} | OOS {agg(s[s.opened_at < SPLIT])} | IS {agg(s[s.opened_at >= SPLIT])}")
    L = s[~s.win]; W = s[s.win]
    print(f"exit anatomy: winners avg {W.pnl_percentage.mean():+.3f} (peak med {W.peak_pnl.median():.2f}, given back {(1 - W.pnl_percentage / W.peak_pnl.replace(0, np.nan)).median()*100:.0f}% of peak) | "
          f"losers avg {L.pnl_percentage.mean():+.3f}, never-positive {(L.peak_pnl <= 0.05).mean()*100:.0f}%, stopped {(L.exit.str.contains('STOP')).mean()*100:.0f}% | hold med W {W.hold.median():.0f}m L {L.hold.median():.0f}m")
    print("exits:", s.groupby("exit").agg(N=("pnl", "size"), WR=("win", lambda x: round(x.mean()*100)), net=("pnl", "sum")).round(0).sort_values("N", ascending=False).head(6).to_dict("index"))
    sweep(s, sl)
