#!/usr/bin/env python3
"""Theory-first pre-registered SPIKE_FADE hypotheses (written BEFORE reading the exhaustive screen), plus the open 41h gate tally,
scored in every block (LIVE KMNO · MASTER per batch · BT per regime) with day-cluster bootstrap. Research only.
Output: reports/FADE_WL_SCREEN_2026-10-19_prereg.csv + _prereg_perbatch.csv
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/guillearslanian/Downloads/NOFA AI/scripts")
import study_fade_wl_screen as S

A = S.A.copy()
num = lambda c: pd.to_numeric(A[c], errors="coerce")
H = {  # name: (mask, scored, expected sign of zone-minus-rest, theory)
 "H1 infl>=2 (24h vol >= 2x prior-7d avg)": (A.d_infl >= 2, A.d_infl.notna(), -1, "attention surge = new information -> spike continues"),
 "H2 24h vol in low tercile (BT T1)": (A.d_log_vol24h <= A.loc[A.blk == 'BT', 'd_log_vol24h'].quantile(1/3), A.d_log_vol24h.notna(), -1, "thin pair: spike is one buyer, but also easier to squeeze"),
 "H3 turnover V/mcap >= BT median": (A.d_turnover >= A.loc[A.blk == 'BT', 'd_turnover'].median(), A.d_turnover.notna(), -1, "volume large vs size = speculative attention"),
 "H4 pair 1h EMA20>EMA200 (1h uptrend)": (num("entry_pair_1h_ema20_200_gap_pct") > 0, num("entry_pair_1h_ema20_200_gap_pct").notna(), -1, "fading WITH the higher-TF trend against you"),
 "H5 DI spread (+DI - -DI) >= BT T2": (A.d_di_spread >= A.loc[A.blk == 'BT', 'd_di_spread'].quantile(2/3), A.d_di_spread.notna(), -1, "one-sided buying pressure"),
 "H6 ADX<20 and rising (fresh ignition)": ((num("entry_adx") < 20) & (num("entry_adx_delta") > 0), num("entry_adx").notna() & num("entry_adx_delta").notna(), -1, "the original leg-6 logic: low ADX spike = ignition, not exhaustion"),
 "H7 range position >= 95": (num("entry_range_position") >= 95, num("entry_range_position").notna(), -1, "breakout to a new range high"),
 "H8 BTC 5m RSI < 40 (weak BTC)": (num("entry_btc_rsi") < 40, num("entry_btc_rsi").notna(), 0, "two-sided: weak tape sells spikes, OR alt spike is idiosyncratic"),
 "H9 = gate 41h HEALTHY_BEAR & btc_trend_gap>=-0.13": ((A.c_regime_HEALTHY_BEAR == 1) & (num("entry_btc_trend_gap_pct") >= -0.13), A.c_regime_HEALTHY_BEAR.notna() & num("entry_btc_trend_gap_pct").notna(), -1, "registered observe gate 41h"),
 "H10 hot tape BTC 1h RSI >= 70": (num("entry_btc_rsi_1h") >= 70, num("entry_btc_rsi_1h").notna(), -1, "registered hot-tape sub-line"),
 "H11 mcap < $50M": (A.d_log_mcap < np.log10(50e6), A.d_log_mcap.notna(), -1, "size/mcap study: weakest edge"),
}

out, perb = [], []
blocks = [("MASTER", A.blk == "MASTER"), ("BT all", A.blk == "BT"), ("BT 2025 BULL", A.era == "2025 BULL"),
          ("BT 2026-H1 BEAR", A.era == "2026-H1 BEAR"), ("BT 2026-H2 BULL", A.era == "2026-H2 BULL")]
km = A[A.blk == "LIVE"].iloc[0]
for hn, (mask, sc, sign, th) in H.items():
    row = dict(hypothesis=hn, theory=th, expected=sign, KMNO_in_zone=bool(mask[A.blk == "LIVE"].iloc[0]))
    for bn, bm in blocks:
        D = A[bm].reset_index(drop=True); z = mask[bm].values.astype(float); s = sc[bm].values.astype(float)
        zs, rs = describe = S.describe(D, z, s)
        y = D.pct.values.astype(float)
        if zs.get("N", 0) >= 3 and rs.get("N", 0) >= 3:
            bt = S.day_boot(D, z, s, y, nb=4000)
            lo, hi = np.percentile(bt[:, 2], [2.5, 97.5]); p_neg = (bt[:, 2] < 0).mean()
            zlo, zhi = np.percentile(bt[:, 0], [2.5, 97.5]); p_zone_neg = (bt[:, 0] < 0).mean()
        else:
            lo = hi = p_neg = zlo = zhi = p_zone_neg = np.nan
        row[f"{bn}|zone"] = f"{zs.get('N',0)}·{zs.get('days','-')}d·WR {zs.get('WR','-')}·stop {zs.get('stop','-')}·avg {zs.get('avg','-')}"
        row[f"{bn}|rest"] = f"{rs.get('N',0)}·WR {rs.get('WR','-')}·stop {rs.get('stop','-')}·avg {rs.get('avg','-')}"
        row[f"{bn}|delta_CI"] = f"{(zs.get('avg',np.nan) or 0) - (rs.get('avg',np.nan) or 0):+.3f} [{lo:+.3f},{hi:+.3f}] P(d<0)={p_neg:.2f}"
        row[f"{bn}|zone_avg_CI"] = f"[{zlo:+.3f},{zhi:+.3f}] P(zone<0)={p_zone_neg:.2f}"
        row[f"{bn}|top_pair"] = f"{zs.get('top_pair','-')}x{zs.get('top_pair_n','-')}"
    out.append(row)
    # master per batch
    D = A[A.blk == "MASTER"]; z = mask[A.blk == "MASTER"] & sc[A.blk == "MASTER"]
    for era, g in D[z].groupby("era"):
        perb.append(dict(hypothesis=hn, batch=era, N=len(g), W=int((g.pct > 0).sum()), L=int((g.pct <= 0).sum()), sum_pct=round(g.pct.sum(), 2),
                         pairs=",".join(g.pair.str.replace("USDT", "")), losers=",".join(g[g.pct < 0].pair.str.replace("USDT", ""))))
P = pd.DataFrame(out)
P.to_csv(S.OUT + "_prereg.csv", index=False)
pd.DataFrame(perb).to_csv(S.OUT + "_prereg_perbatch.csv", index=False)
for _, r in P.iterrows():
    print("\n##", r.hypothesis, "| KMNO in zone:", r.KMNO_in_zone)
    for bn, _ in blocks:
        print(f"  {bn:16s} zone {r[bn+'|zone']:45s} rest {r[bn+'|rest']:35s} d {r[bn+'|delta_CI']}  top {r[bn+'|top_pair']}")
# sleeve breakeven WR per block (today's exit)
for bn, bm in blocks:
    D = A[bm]; w = D.pct[D.pct > 0].mean(); lo = D.pct[D.pct <= 0].mean()
    print(f"BE-WR {bn}: N {len(D)} WR {(D.pct>0).mean()*100:.1f}% avgW {w:+.3f} avgL {lo:+.3f} BE-WR {abs(lo)/(w+abs(lo))*100:.1f}% avg {D.pct.mean():+.3f} stop {D.is_stop.mean()*100:.1f}%")
print("KMNO", km[["pair", "opened_at", "pct", "d_infl", "d_log_vol24h", "d_turnover", "d_log_mcap", "entry_adx", "entry_rsi", "entry_range_position", "entry_btc_rsi", "entry_bear_pct"]].to_dict())
