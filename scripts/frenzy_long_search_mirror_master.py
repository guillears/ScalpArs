#!/usr/bin/env python3
"""FRENZY LONG SEARCH — (1) pre-registered MIRROR composite on the UNIFIED_LONG Frenzy rows (0–24 h, universe F; no re-fit) and
(2) master check: real Frenzy long fills (MASTER_POOL_stacked, kept on today's stack) by regime sign, in DAY units.
Writes reports/FRENZY_LONG_SEARCH_2026-11-19_mirror.csv and _master.csv."""
import os, pickle, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_eval as EV   # noqa: E402
OUT = EV.OUT; DAY = EV.DAY
T = pd.read_csv(os.path.join(EV.D, "tape_daily.csv"), index_col=0)


def mirror():
    rows = []
    for blk in ("NR", "H23", "H24", "DESIGN", "B"):
        x = pickle.load(open(os.path.join(R, "reports", "UNIFIED_LONG_data", f"all_{blk}.pkl"), "rb"))
        x = x[x.u == "F"]
        cov = x[["taker1h", "top_acc_chg24", "oi_chg24h", "fund_last", "vd"]].notna().all(1)
        m = cov & (x.taker1h >= 1.05) & (x.top_acc_chg24 > 0) & (x.oi_chg24h > 0) & (x.fund_last <= 0) & (x.vd >= 0.5)
        h24 = x["ev288"] - 0.10 - 0.10 - 0.10 - (x["fund_cost"].fillna(0) * 0)   # 24 h no-stop hold, gross − fees − slip (funding not in ev288)
        for name, sel in (("MIRROR", m), ("not MIRROR (scored)", cov & ~m), ("unscored", ~cov)):
            d = x[sel]
            if not len(d):
                continue
            r = dict(block=blk, set=name, N=len(d), days=d.day.nunique() if "day" in d else np.nan, coverage=cov.mean() * 100)
            for ex in ("T3S4H4h", "T5S6H24h", "T2S3H90m"):
                r[ex] = d[ex + "_n"].mean()
            r["hold24_gross_minus_costs"] = h24[sel].mean()
            rows.append(r)
        del x
    M = pd.DataFrame(rows); M.to_csv(OUT + "_mirror.csv", index=False)
    return M


def master():
    m = pd.read_csv(os.path.join(R, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    S = ["FRENZY_LONG", "FRENZY_LITE", "FRENZY_LITE_RETRY", "FRENZY_WIDE", "FRENZY_WILLY", "SURGE_LONG"]
    f = m[m.entry_strategy.isin(S) & (m.is_probe != True)].copy()
    f["day"] = pd.to_datetime(f.opened_at).values.astype("datetime64[ms]").astype("int64") // DAY
    for k in ("R1", "R6", "R5", "RF", "spikes7_prev", "breadth_prev", "btc_r7_prev"):
        f[k] = f.day.map(T[k])
    f["kept"] = f.stack_keep == True
    f["pct"] = np.where(f.kept, f.stack_pct, np.nan)
    f["week"] = pd.to_datetime(f.opened_at).dt.strftime("%G-W%V")
    rows = []
    for sl in S + ["ALL"]:
        d = f if sl == "ALL" else f[f.entry_strategy == sl]
        dk = d[d.kept]
        r = dict(sleeve=sl, fills=len(d), kept=len(dk), kept_avg=dk.pct.mean(), kept_days=dk.day.nunique(),
                 asTraded_avg=d.pnl_percentage.mean())
        for rv in ("R6", "RF"):
            for sg in (1, 0):
                s_ = dk[dk[rv] == sg]
                r[f"{rv}={sg}"] = f"{len(s_)}·{s_.pct.mean():+.2f}·{s_.day.nunique()}d" if len(s_) else "0"
        r["R1 share ON"] = dk.R1.mean() * 100 if len(dk) else np.nan
        r["R5 share ON"] = dk.R5.mean() * 100 if len(dk) else np.nan
        rows.append(r)
    M = pd.DataFrame(rows); M.to_csv(OUT + "_master.csv", index=False)
    W = f[f.kept].groupby("week").agg(N=("pct", "size"), avg=("pct", "mean"), spikes7=("spikes7_prev", "mean"),
                                       breadth=("breadth_prev", "mean"), btc_r7=("btc_r7_prev", "mean"))
    W.to_csv(OUT + "_master_weeks.csv")
    return M, W


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(master()[0].round(2).to_string()); print(master()[1].round(2).to_string())
    if len(sys.argv) > 1: print(mirror().round(2).to_string())
