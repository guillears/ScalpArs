#!/usr/bin/env python3
"""FRENZY W/L screen 2026-10-19 — build the REAL-fill pool (master STACK 2026-10-19a kept non-probe + live export B23).
Read-only on every input. Outcome y = master stack_pct (today's exit) / live booked pnl_percentage (live = today's config).
Adds derived + journal (order-book) features joinable at entry. Output: scratch pickle passed as argv[1]; journal pickle argv[2]."""
import sys
import numpy as np, pandas as pd
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"
LIVE = "/Users/guillearslanian/Downloads/scalpars_orders_live_2026-10-19_14-54-01.csv"
OUT, JPK = sys.argv[1], sys.argv[2]
SL = ["FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY"]

M = pd.read_csv(f"{ROOT}/reports/MASTER_POOL_stacked.csv", low_memory=False)
L = pd.read_csv(LIVE, low_memory=False)
allfz = pd.concat([M[M.entry_strategy.astype(str).str.startswith("FRENZY")].assign(src="master"),
                   L[L.entry_strategy.astype(str).str.startswith("FRENZY")].assign(src="live", era="B23")], ignore_index=True)
allpos = pd.concat([M.assign(src="master"), L.assign(src="live")], ignore_index=True)   # every real position (for concurrency)

Mk = M[M.entry_strategy.isin(SL) & (M.stack_keep.astype(str).str.lower() == "true") & ~M.is_probe.astype(str).str.lower().isin(["true", "1"])
       & (M.status == "CLOSED")].copy()
Mk["y"] = pd.to_numeric(Mk.stack_pct, errors="coerce"); Mk["src"] = "master"
Lk = L[L.entry_strategy.isin(SL) & (L.status == "CLOSED")].copy()
Lk["y"] = pd.to_numeric(Lk.pnl_percentage, errors="coerce"); Lk["src"] = "live"; Lk["era"] = "B23"
D = pd.concat([Mk, Lk], ignore_index=True)
key = D.opened_at.astype(str).str[:19] + D.pair + D.direction
assert not key.duplicated().any(), "duplicate fills master/live"
D["ts"] = pd.to_datetime(D.opened_at.astype(str).str[:19])
D["tc"] = pd.to_datetime(D.closed_at.astype(str).str[:19])
D["day"] = D.ts.dt.strftime("%Y-%m-%d")
D["sleeve"] = D.entry_strategy
D["win"] = D.y > 0
D["stopped"] = D.close_reason.astype(str).str.contains("STOP") & (D.y < -2)
D["is_live_money"] = ~D.is_paper.astype(str).str.lower().isin(["true", "1"])

# ── derived features (all known at entry) ─────────────────────────────────────────────────────────────────────────
D["x_hour_utc"] = D.ts.dt.hour + D.ts.dt.minute / 60
D["x_weekday"] = D.ts.dt.weekday
D["x_weekend"] = (D.ts.dt.weekday >= 5).astype(float)
D["x_log_mcap"] = np.log10(pd.to_numeric(D.entry_mcap_usd, errors="coerce").where(lambda s: s > 0))
D["x_log_vol24"] = np.log10(pd.to_numeric(D.entry_pair_volume_24h_usd, errors="coerce").where(lambda s: s > 0))
allfz["ts"] = pd.to_datetime(allfz.opened_at.astype(str).str[:19]); allfz["tc"] = pd.to_datetime(allfz.closed_at.astype(str).str[:19], errors="coerce")
allpos["ts"] = pd.to_datetime(allpos.opened_at.astype(str).str[:19], errors="coerce"); allpos["tc"] = pd.to_datetime(allpos.closed_at.astype(str).str[:19], errors="coerce")
kth, since_prev, prev_res, n_open, n_fz_open, pair_fills_7d = [], [], [], [], [], []
for _, r in D.iterrows():
    same = allfz[(allfz.pair == r.pair) & (allfz.ts < r.ts)]
    ep = same[same.entry_frenzy_spike_at.astype(str) == str(r.entry_frenzy_spike_at)] if pd.notna(r.entry_frenzy_spike_at) else same.iloc[0:0]
    kth.append(len(ep) + 1)
    since_prev.append((r.ts - same.ts.max()).total_seconds() / 3600 if len(same) else np.nan)
    pv = same.sort_values("ts").tail(1)
    prev_res.append(float(pd.to_numeric(pv.pnl_percentage, errors="coerce").iloc[0]) if len(pv) and pv.tc.iloc[0] <= r.ts else np.nan)
    op = allpos[(allpos.ts < r.ts) & ((allpos.tc > r.ts) | allpos.tc.isna()) & (allpos.status.isin(["CLOSED", "OPEN"]))]
    n_open.append(len(op)); n_fz_open.append(int(op.entry_strategy.astype(str).str.startswith("FRENZY").sum()))
    pair_fills_7d.append(int(((allfz.pair == r.pair) & (allfz.ts < r.ts) & (allfz.ts >= r.ts - pd.Timedelta(days=7))).sum()))
D["x_kth_in_episode"] = kth; D["x_hours_since_prev_fz_fill_pair"] = since_prev; D["x_prev_fz_fill_pct_pair"] = prev_res
D["x_open_positions_at_entry"] = n_open; D["x_frenzy_open_at_entry"] = n_fz_open; D["x_pair_fz_fills_prev7d"] = pair_fills_7d
sp = pd.to_datetime(D.entry_frenzy_spike_at.astype(str).str[:19], errors="coerce")
D["x_minutes_since_spike"] = (D.ts - sp).dt.total_seconds() / 60

# ── journal: order book at entry (BOOK minute rows, same mode) ──────────────────────────────────────────────────────
J = pd.read_pickle(JPK)
B = J[J.e == "BOOK"].copy()
B["tm"] = pd.to_datetime(B.t.astype(str).str[:16])
for c in [c for c in B.columns if c.startswith("ob_")]:
    B[c] = pd.to_numeric(B[c], errors="coerce")
B["d1"] = B.ob_bid_usd_1 + B.ob_ask_usd_1
B["d05"] = B.ob_bid_usd_05 + B.ob_ask_usd_05
nbook = B.groupby(["mode", "tm"]).pair.nunique()
Bi = {k: g.set_index("tm").sort_index() for k, g in B.groupby(["mode", "pair"])}
feat = []
for _, r in D.iterrows():
    mode = "live" if r.is_live_money else "paper"
    g = Bi.get((mode, r.pair))
    out = {}
    if g is not None:
        t0 = r.ts.floor("min")
        pre = g.loc[:t0]
        if len(pre) and (t0 - pre.index[-1]) <= pd.Timedelta(minutes=2):
            s = pre.iloc[-1]
            out = {"j_spread_pct": s.ob_spread_pct, "j_imb_025": s.ob_imb_025, "j_imb_05": s.ob_imb_05, "j_imb_1": s.ob_imb_1, "j_imb_2": s.ob_imb_2,
                   "j_log_depth1": np.log10(s.d1) if s.d1 > 0 else np.nan, "j_log_depth05": np.log10(s.d05) if s.d05 > 0 else np.nan,
                   "j_bidask_ratio_1": s.ob_bid_usd_1 / s.ob_ask_usd_1 if s.ob_ask_usd_1 > 0 else np.nan,
                   "j_wall_bid_dist": s.ob_wall_bid_dist_pct, "j_wall_bid_share": s.ob_wall_bid_share, "j_wall_ask_dist": s.ob_wall_ask_dist_pct,
                   "j_wall_ask_share": s.ob_wall_ask_share, "j_reach_bid": s.ob_reach_bid_pct, "j_reach_ask": s.ob_reach_ask_pct,
                   "j_depth1_to_notional": s.d1 / float(r.notional_value) if float(r.notional_value or 0) > 0 else np.nan}
            p15 = g.loc[t0 - pd.Timedelta(minutes=16): t0 - pd.Timedelta(minutes=14)]
            if len(p15):
                q = p15.iloc[-1]
                out["j_depth1_chg15_pct"] = (s.d1 / q.d1 - 1) * 100 if q.d1 > 0 else np.nan
                out["j_mid_chg15_pct"] = (s.ob_mid / q.ob_mid - 1) * 100 if q.ob_mid > 0 else np.nan
                out["j_imb1_chg15"] = s.ob_imb_1 - q.ob_imb_1
        out["j_n_frenzy_pairs_in_book"] = nbook.get((mode, r.ts.floor("min")), np.nan) if out else np.nan
    feat.append(out)
F = pd.DataFrame(feat, index=D.index)
D = pd.concat([D, F], axis=1)
D.to_pickle(OUT)
print(D.groupby(["sleeve", "src"]).size())
print(D.groupby("sleeve")[["j_spread_pct", "j_depth1_chg15_pct"]].apply(lambda x: x.notna().mean()))
