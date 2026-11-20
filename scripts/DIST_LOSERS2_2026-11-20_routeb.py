"""DIST_LOSERS2 (2026-11-20): Route B recorded book / flow (5-min buckets) around the DIST + EXH fills inside the recording (from 2026-11-19 23 UTC)."""
import sys, io, os, pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_b")); import daily_report as R  # noqa: E402
pd.set_option("display.width", 250)
T = [("CYBERUSDT", "DIST win", "2026-11-20 09:00:26", "2026-11-20 11:17:57"),
     ("KERNELUSDT", "DIST loss", "2026-11-20 12:30:27", "2026-11-20 13:07:21"),
     ("KERNELUSDT", "EXH win", "2026-11-20 14:15:30", "2026-11-20 15:45:49")]
hours = sorted({f"features/{t:%Y-%m-%d}/{t:%H}.csv.gz" for _, _, a, b in T
                for t in pd.date_range(pd.Timestamp(a) - pd.Timedelta("90min"), pd.Timestamp(b) + pd.Timedelta("60min"), freq="h")})
fr = []
for k in hours:
    b = R.fetch(k)
    if b: fr.append(pd.read_csv(io.BytesIO(b), compression="gzip"))
    else: print("missing", k)
F = pd.concat(fr).sort_values("n_book").drop_duplicates(["sym", "minute"], keep="last")
for p, tag, a, b in T:
    e = pd.Timestamp(a, tz="UTC").value // 10**6 // 60000 * 60000; x = pd.Timestamp(b, tz="UTC").value // 10**6
    g = F[(F.sym == p) & (F.minute >= e - 60 * 60000) & (F.minute <= x + 30 * 60000)].sort_values("minute").copy()
    g["t"] = (g.minute - e) // 60000
    g["px%"] = (g.mid / g.loc[g.t == 0, "mid"].iloc[0] - 1) * 100 if (g.t == 0).any() else np.nan
    g["net_taker_k"] = (g.buy_usd - g.sell_usd) / 1e3
    g["vol_k"] = (g.buy_usd + g.sell_usd) / 1e3
    g["ask05_k"] = g.ask_usd_05 / 1e3; g["bid05_k"] = g.bid_usd_05 / 1e3
    g["liqS_k"] = g.liq_short_usd / 1e3; g["liqL_k"] = g.liq_long_usd / 1e3
    g["fund_bp"] = g.funding * 1e4
    cols = ["t", "px%", "vol_k", "net_taker_k", "bid05_k", "ask05_k", "imb_05", "spread_bps", "liqS_k", "liqL_k", "fund_bp"]
    print(f"\n=== {p} {tag}  entry {a}  exit {b}  rows {len(g)}")
    # 5-minute buckets
    g["b5"] = (g.t // 5) * 5
    agg = g.groupby("b5").agg({"px%": "last", "vol_k": "sum", "net_taker_k": "sum", "bid05_k": "mean", "ask05_k": "mean",
                               "imb_05": "mean", "spread_bps": "mean", "liqS_k": "sum", "liqL_k": "sum", "fund_bp": "last"}).round(2)
    print(agg.to_string())
