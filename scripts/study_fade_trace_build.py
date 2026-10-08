#!/usr/bin/env python3
"""SPIKE_FADE recall trace (2026-10-08) — step 1: the matched set.

Read-only. Live = every SPIKE_FADE fill in reports/MASTER_POOL_stacked.csv (the master builder already dedups (opened_at, pair,
direction) over the master + archived batches; cross-checked here against every batch CSV and the latest Downloads export),
CLOSED, opened inside the yr5 replay window [2026-07-28, 2026-10-04). Replay = yr5 (code 181131e) SPIKE_FADE fills, 3 seeds,
warm-up trimmed (scripts/yr5_fills_trimmed.py).
Match: same pair, SHORT, |open Δ| ≤ 10 min, any seed. Live-up periods = union of the live batch spans [first open, last close]
(BATCH1 + BASELINE2..18); the two real offline stretches are Aug-10 11:30 → Aug-11 20:00 and Aug-27 16:10 → Sep-11 15:26.
Writes reports/study_fade_trace_matchset.csv (one row per live fade and per replay fade) and
reports/study_fade_trace_pairdays.csv (pair-days whose real trades the per-signal trace needs).
Usage: venv/bin/python scripts/study_fade_trace_build.py
"""
import glob, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import yr5_fills_trimmed as YT                                          # noqa: E402

REP = os.path.join(ROOT, "reports")
W0, W1 = pd.Timestamp("2026-07-28"), pd.Timestamp("2026-10-04")
TOL = pd.Timedelta(minutes=10)


def ts(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:23].str.replace("T", " "), format="mixed", errors="coerce")


def batch_files():
    fs = [os.path.join(REP, "BATCH1_2026-07-11to31_orders_FINAL.csv")] + sorted(glob.glob(os.path.join(REP, "BASELINE*_*.csv")))
    return [f for f in fs if "ANCHOR" not in f and "superseded" not in f]


def live_up_intervals():
    iv = []
    for f in batch_files():
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "closed_at"))
        a = ts(d.opened_at).min(); z = max(ts(d.opened_at).max(), ts(d.closed_at).max())
        iv.append((a, z))
    iv.sort()
    out = []
    for a, z in iv:                                    # merge overlaps / sub-hour seams (reset gaps of minutes)
        if out and a <= out[-1][1] + pd.Timedelta(hours=1):
            out[-1] = (out[-1][0], max(out[-1][1], z))
        else:
            out.append((a, z))
    return out


def is_up(t, iv):
    t = pd.Series(t)
    m = np.zeros(len(t), bool)
    for a, z in iv:
        m |= ((t >= a) & (t <= z)).values
    return m


def load_live():
    M = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
    L = M[(M.entry_strategy == "SPIKE_FADE") & (M.status == "CLOSED")].copy()
    L["t"] = ts(L.opened_at).values
    L = L[(L.t >= W0) & (L.t < W1)]
    L["pct_raw"] = pd.to_numeric(L.pnl_percentage, errors="coerce")
    nv = pd.to_numeric(L.notional_value, errors="coerce")
    cf = pd.to_numeric(L.cf_pnl_current_stack, errors="coerce")
    # master convention (ENGINE_REPLAY_YEAR_PLAN 2026-09-28 fix): CF rows' stack % = CF $ / notional
    L["pct_stack"] = np.where(cf.notna() & (nv > 0), cf / nv * 100.0, L.pct_raw)
    # cross-check: every SPIKE_FADE row of every batch CSV + the newest Downloads export is in the master
    seen = set(zip(L.opened_at.astype(str).str[:19], L.pair))
    extra = []
    dl = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")), key=os.path.getmtime)[-1:]
    for f in batch_files() + dl:
        d = pd.read_csv(f, low_memory=False)
        if "entry_strategy" not in d:
            continue
        d = d[(d.entry_strategy == "SPIKE_FADE") & (d.status == "CLOSED")]
        for oa, p in zip(d.opened_at.astype(str).str[:19].str.replace(" ", "T"), d.pair):
            t = pd.Timestamp(oa)
            if W0 <= t < W1 and (oa, p) not in seen:
                extra.append((os.path.basename(f), oa, p))
    return L, extra


def main():
    iv = live_up_intervals()
    L, extra = load_live()
    R = YT.load(sleeves=["Spike-Fade"])
    R = R[(R.t >= W0) & (R.t < W1)].copy()
    R["pct"] = pd.to_numeric(R.pnl_percentage, errors="coerce")
    # match
    L["m_seeds"] = ""; L["m_dt_s"] = np.nan; L["m_rep_pct"] = np.nan; L["m_rep_reason"] = ""
    R["m_live_idx"] = -1
    for i, r in L.iterrows():
        c = R[(R.pair == r.pair) & ((R.t - r.t).abs() <= TOL)]
        if len(c):
            L.at[i, "m_seeds"] = ",".join(str(s) for s in sorted(c.seed.unique()))
            L.at[i, "m_dt_s"] = (c.t - r.t).dt.total_seconds().median()
            L.at[i, "m_rep_pct"] = c.pct.mean()
            L.at[i, "m_rep_reason"] = "|".join(sorted(set(c.close_reason.astype(str))))
            R.loc[c.index, "m_live_idx"] = i
    L["n_seeds"] = L.m_seeds.str.count(",") + (L.m_seeds != "").astype(int)
    R["live_up"] = is_up(R.t, iv)
    # nearest replay fade on the same pair (any distance) for live-only rows — "fired at a different time"
    L["near_rep_min"] = [((R[R.pair == p].t - t).dt.total_seconds() / 60).abs().min() if (R.pair == p).any() else np.nan
                         for p, t in zip(L.pair, L.t)]
    out_l = pd.DataFrame({"side": "LIVE", "pair": L.pair.values, "t": L.t.values, "seed": 0, "era": L.era.values,
                          "stack_keep": L.stack_keep.values, "pct_live_raw": L.pct_raw.values, "pct_live_stack": L.pct_stack.values,
                          "close_reason": L.close_reason.values, "stack_reason": L.get("stack_reason", pd.Series([""] * len(L))).values,
                          "n_seeds": L.n_seeds.values, "m_seeds": L.m_seeds.values, "m_dt_s": L.m_dt_s.values,
                          "pct_rep": L.m_rep_pct.values, "rep_reason": L.m_rep_reason.values, "near_rep_min": L.near_rep_min.values,
                          "live_up": True})
    for c in [c for c in L.columns if c.startswith("entry_")]:
        out_l[c] = L[c].values
    out_r = pd.DataFrame({"side": "REPLAY", "pair": R.pair.values, "t": R.t.values, "seed": R.seed.values, "era": "",
                          "stack_keep": True, "pct_rep": R.pct.values, "rep_reason": R.close_reason.values,
                          "matched_live": R.m_live_idx.values >= 0, "live_up": R.live_up.values})
    for c in [c for c in R.columns if c.startswith("entry_")]:
        out_r[c] = R[c].values
    out = pd.concat([out_l, out_r], ignore_index=True)
    out.to_csv(os.path.join(REP, "study_fade_trace_matchset.csv"), index=False)
    # pair-days needed (the signal day, plus the previous day when the signal is < 30 min after midnight)
    pdays = set()
    for p, t in zip(out.pair, pd.to_datetime(out.t)):
        pdays.add((p, t.strftime("%Y-%m-%d")))
        if t.hour == 0 and t.minute < 30:
            pdays.add((p, (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
    PD = pd.DataFrame(sorted(pdays), columns=["pair", "date"])
    PD["in_cache_q"] = [os.path.exists(os.path.join(REP, "backtest_cache", "ticks_q", p, f"{d}.npz")) for p, d in zip(PD.pair, PD.date)]
    PD.to_csv(os.path.join(REP, "study_fade_trace_pairdays.csv"), index=False)
    # console summary
    print("live-up intervals:", [(str(a)[:16], str(z)[:16]) for a, z in iv])
    print(f"live fades {len(L)} (kept {int(L.stack_keep.sum())}) · reproduced ≥1 seed {int((L.n_seeds > 0).sum())} "
          f"(kept {int(((L.n_seeds > 0) & L.stack_keep).sum())}) · extra batch rows not in master: {len(extra)} {extra[:5]}")
    print(f"replay fades {len(R)} ({R.seed.value_counts().to_dict()}) · live-up {int(R.live_up.sum())} · matched {int((R.m_live_idx >= 0).sum())}")
    print(f"pair-days needed {len(PD)} · in ticks_q cache {int(PD.in_cache_q.sum())}")


if __name__ == "__main__":
    main()
