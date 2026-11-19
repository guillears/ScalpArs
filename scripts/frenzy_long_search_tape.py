#!/usr/bin/env python3
"""FRENZY LONG SEARCH — daily tape / regime table + current-tape analogs (PREREG §Regime splits, §Tape-analog).
Inputs (local, 0 API weight): reports/LONG_LEADS_DEEP_data/btc_daily.csv (BTC daily closes), reports/REGIME_LONGS_data/daily_closes.csv
(914 USDT perps, delisted incl.), reports/UNIFIED_LONG_data/eps_meta.csv (validated-walk Frenzy spikes 2023-01 → 2026-11-17).
Output: reports/FRENZY_LONG_SEARCH_data/tape_daily.csv (one row per UTC day; every regime column is the value of the PRIOR closed day,
i.e. what a trade on that day could know) and reports/FRENZY_LONG_SEARCH_2026-11-19_analogs.csv."""
import os
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "FRENZY_LONG_SEARCH_data"); os.makedirs(D, exist_ok=True)
DAY = 86_400_000


def main():
    b = pd.read_csv(os.path.join(R, "reports", "LONG_LEADS_DEEP_data", "btc_daily.csv")).sort_values("d").set_index("d")
    c = b.c
    e50 = c.ewm(span=50, adjust=False).mean(); e200 = c.ewm(span=200, adjust=False).mean()
    T = pd.DataFrame({"btc_c": c, "btc_gap": (e50 / e200 - 1) * 100, "btc_r1": c.pct_change() * 100,
                      "btc_r7": c.pct_change(7) * 100, "btc_r14": c.pct_change(14) * 100, "btc_r30": c.pct_change(30) * 100,
                      "btc_off30h": (c / c.rolling(30).max() - 1) * 100})
    dc = pd.read_csv(os.path.join(R, "reports", "REGIME_LONGS_data", "daily_closes.csv"))
    P = dc.pivot_table(index="d", columns="sym", values="c").sort_index()
    P = P.drop(columns=[x for x in ("BTCUSDT", "ETHUSDT") if x in P.columns])
    cnt = P.notna().cumsum()
    E = P.ewm(span=50, adjust=False, min_periods=50).mean()
    above = (P > E) & (cnt >= 50)
    valid = (cnt >= 50) & P.notna()
    T["breadth"] = (above.sum(1) / valid.sum(1).replace(0, np.nan) * 100).reindex(T.index)
    r1 = P.pct_change(fill_method=None) * 100; r7 = P.pct_change(7, fill_method=None) * 100
    T["alt_med_r1"] = r1.median(1).reindex(T.index); T["alt_med_r7"] = r7.median(1).reindex(T.index)
    T["alt_up15"] = (r1 >= 15).sum(1).reindex(T.index)
    T["alt_up_share"] = ((r1 > 0).sum(1) / r1.notna().sum(1).replace(0, np.nan) * 100).reindex(T.index)
    m = pd.read_csv(os.path.join(R, "reports", "UNIFIED_LONG_data", "eps_meta.csv"))
    sd = (m.spike_ms // DAY).value_counts().sort_index()
    T["spikes"] = sd.reindex(T.index).fillna(0)
    T.loc[T.index < 19358, "spikes"] = np.nan                     # episode lists start 2023-01-01 (day 19358)
    T["spikes7"] = T.spikes.rolling(7).mean()
    # every regime column shifted by one day: the row for day d = what was known at 00:00 UTC of day d
    S = T.shift(1)
    S.columns = [k + "_prev" for k in S.columns]
    out = pd.concat([T, S], axis=1)
    out["date"] = pd.to_datetime(out.index * DAY, unit="ms").strftime("%Y-%m-%d")
    out["R1"] = (out.btc_gap_prev > 0).astype(int); out["R6"] = (out.btc_r1_prev > 0).astype(int)
    out["R5"] = (out.breadth_prev > 50).astype(int)
    med = out.loc[out.index >= 19358 + 7, "spikes7_prev"].median()
    out["RF"] = (out.spikes7_prev >= med).astype(int)
    out.attrs["rf_median"] = med
    out.to_csv(os.path.join(D, "tape_daily.csv"))
    print("RF median spikes/day (7-d avg):", med)
    # current tape (14 days ending 2026-11-17 = day 20774) and analogs
    feats = ["btc_gap", "btc_r14", "breadth", "spikes7", "alt_med_r7"]
    W = out[feats].rolling(14).mean()
    W["btc_r14"] = out["btc_r14"]
    cur_d = 20774
    cur = W.loc[cur_d]
    hist = W.loc[(W.index >= 19358 + 21) & (W.index <= cur_d - 60)].dropna()
    z = (hist - hist.mean()) / hist.std()
    zc = (cur - hist.mean()) / hist.std()
    dist = np.sqrt(((z - zc) ** 2).sum(1))
    thr = dist.quantile(0.10)
    A = pd.DataFrame({"date": out.loc[hist.index, "date"], "dist": dist, "analog": dist <= thr})
    A = pd.concat([A, hist], axis=1)
    A.to_csv(os.path.join(R, "reports", "FRENZY_LONG_SEARCH_2026-11-19_analogs.csv"))
    print("current tape (14-d means to 2026-11-17):"); print(cur.round(2).to_string())
    print("history mean / sd:"); print(pd.DataFrame({"mean": hist.mean(), "sd": hist.std()}).round(2).to_string())
    an = A[A.analog]
    # analog windows = runs of consecutive analog days
    run = (an.index.to_series().diff() != 1).cumsum()
    print("analog days", len(an), "windows", run.nunique())
    for k, g in an.groupby(run.values):
        print(f"  {g.date.iloc[0]} → {g.date.iloc[-1]} ({len(g)} d) dist {g.dist.min():.2f}")
    # last 21 days of the tape
    print(out.loc[out.index >= cur_d - 20, ["date", "btc_c", "btc_gap", "btc_r1", "btc_r7", "breadth", "alt_med_r1", "alt_med_r7", "alt_up15", "spikes", "spikes7"]].round(2).to_string())


if __name__ == "__main__":
    main()
