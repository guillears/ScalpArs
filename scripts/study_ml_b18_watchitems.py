#!/usr/bin/env python3
"""B18 ML losers — which registered watch items catch the 3 fills, + cohort tallies of the ones that do (read-only)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_b18_common as C
n = lambda d, c: pd.to_numeric(d[c], errors="coerce")


def flags(d):
    F = {}
    F["NEGFLANK (210): BTC 1h slope ≤ −0.05"] = n(d, "entry_btc_1h_slope") <= -0.05
    F["SLOPE<0 observe tally (pin)"] = n(d, "entry_btc_1h_slope") < 0
    F["ML_STOP_COOLDOWN (248) <30 min after ML stop"] = d.cooldown
    F["CLUSTER2_120 (exploratory)"] = d.cluster2
    F["BTC-CHOP eff72 ≤ 0.007 (161)"] = n(d, "entry_btc_eff72") <= 0.007
    F["BURST ≤ 2 min from another bot fill (149)"] = d.burst
    F["HEAT 3-leg (208): b20 ≥ .07 ∧ RSIprev ≥ 64 ∧ bull ≥ 80"] = (n(d, "entry_btc_ema20_slope") >= .07) & (n(d, "entry_btc_rsi_prev") >= 64) & (n(d, "entry_bull_pct") >= 80)
    F["LOADX (126, armed): RSI < RSI_prev2 ∧ ADX < 21"] = (n(d, "entry_rsi") < n(d, "entry_rsi_prev")) & (n(d, "entry_adx") < 21)
    F["3-LEG late-entry observe (39): b20 ≥ .06 ∧ +DI < 28 ∧ EMA50 slope ≥ .15"] = (n(d, "entry_btc_ema20_slope") >= .06) & (n(d, "entry_pos_di") < 28) & (n(d, "entry_ema50_slope") >= .15)
    F["PVR band ③ 0.68–0.90"] = n(d, "entry_pair_volume_ratio").between(0.68, 0.90, inclusive="left")
    F["ATR×GAP zone (133): ATR ≥ 1 ∧ gap ≥ 0.5"] = (n(d, "entry_atr_pct") >= 1) & (n(d, "entry_pair_ema20_ema50_gap_pct") >= 0.5)
    F["BTC_HOT_MATURE (Jul-16 macro watch): BTC ADX ≥ 25 ∧ (ATR ≥ .15 ∨ ext13 ≥ .20)"] = (n(d, "entry_btc_adx") >= 25) & ((n(d, "entry_btc_atr_pct") >= .15) | (n(d, "entry_btc_dist_from_ema13_pct") >= .20))
    F["OFF24: BTC ≥ 2 % below 24 h high (combo screen flag)"] = n(d, "entry_btc_off24h_pct") <= -2
    F["ROLL: BTC RSI drop ≥ 5 ∧ pair RSI falling"] = (n(d, "entry_btc_rsi_prev") - n(d, "entry_btc_rsi") >= 5) & (n(d, "entry_rsi") < n(d, "entry_rsi_prev"))
    F["ZC: BTC EMA50/100 gap ≤ .006 ∧ ETH 5m ≤ 0"] = (n(d, "entry_btc_ema50_100_gap_pct") <= .006) & (n(d, "entry_eth_5m_ret1_pct") <= 0)
    F["SPRINT de-mux: gvol > .74 ∧ b20 > .07"] = (n(d, "entry_global_volume_ratio") > .74) & (n(d, "entry_btc_ema20_slope") > .07)
    F["MEGACAP rank ≤ 10 (armed)"] = n(d, "entry_pair_rank") <= 10
    F["REBOUND watch: off30 ≤ −15"] = d.off30 <= -15
    F["FRENZY bearish-day def (1d < 0 ∧ trend gap < 0)"] = (d.btc1d < 0) & (n(d, "entry_btc_trend_gap_pct") < 0)
    F["BTC 1d < 0"] = d.btc1d < 0
    return pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in F.items()}, index=d.index)


M = C.master(); Y = C.yr5(); NS = Y.seed.nunique()


def burst(d, allf, key):
    out = pd.Series(False, index=d.index)
    for g, sub in d.groupby(key):
        t = np.sort(allf[allf[key] == g].o.values)
        for i, r in sub.iterrows():
            dt = np.abs((t - np.datetime64(r.o)) / np.timedelta64(1, "s"))
            out[i] = ((dt > 0.5) & (dt <= 120)).any()
    return out


P = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
P = P[(P.entry_strategy.fillna("MOMENTUM") != "MANUAL")].copy(); P["o"] = pd.to_datetime(P.opened_at.astype(str).str[:19])
M["burst"] = burst(M, P, "era")
A = C.YT.load(); A["o"] = A.t
Y["burst"] = burst(Y, A, "seed")
FM, FY = flags(M), flags(Y)
b = M[M.b18]
out = ["| watch item | ARB 16:31 | UNI 16:36 | LIT 17:13 | master ex-B1 ex-washed: zone N · WR · avg % (rest avg) | yr5 ex-washed: zone N/seed · WR · avg % (rest) | stamp coverage master |", "|---|---|---|---|---|---|---|"]
for k in FM.columns:
    hit = ["✔" if FM.loc[i, k] else "·" for i in b.sort_values("o").index]
    me, ye = ~M.washed30, ~Y.washed30
    req = {"CHOP": "entry_btc_eff72", "OFF24": "entry_btc_off24h_pct", "ZC": "entry_btc_ema50_100_gap_pct"}
    sc = pd.Series(True, index=M.index)
    for kk, cc in req.items():
        if k.startswith(kk) or ("CHOP" in k and kk == "CHOP"):
            sc = M[cc].notna()
    me = me & sc   # coverage first: unscored master fills are dropped, never counted as 'rest'
    z, r = M[me & FM[k]], M[me & ~FM[k]]; zy, ry = Y[ye & FY[k]], Y[ye & ~FY[k]]
    cov = "—"
    out.append(f"| {k} | {' | '.join(hit)} | {len(z)} · {(z.pct>0).mean()*100 if len(z) else 0:.0f}% · {z.pct.mean() if len(z) else float('nan'):+.3f} ({r.pct.mean():+.3f}) | "
               f"{len(zy)/NS:.0f} · {(zy.pct>0).mean()*100 if len(zy) else 0:.0f}% · {zy.pct.mean() if len(zy) else float('nan'):+.3f} ({ry.pct.mean():+.3f}) | |")
print("\n".join(out))
# coverage notes
for c in ["entry_btc_eff72", "entry_chop_burst_prior_fill_s", "entry_btc_off24h_pct", "entry_btc_ema50_100_gap_pct", "entry_eth_5m_ret1_pct"]:
    print(c, f"master coverage {M[c].notna().mean()*100:.0f}% (CHOP/OFF24/ZC master rows above use the scored subset only)")
# as-traded NEGFLANK (DECISION_LOG 204 reading rule): every non-probe ML fill as traded, B1 excluded, ex-washed
H = C.history_master(); H = H[H.era != "B1"].copy()
H["pct"] = pd.to_numeric(H.pnl_percentage, errors="coerce"); H["slope"] = n(H, "entry_btc_1h_slope")
H["off30"] = n(H, "entry_btc_off30d_high_pct")
H["wash"] = (H.o >= C.WASH[0]) & (H.o < C.WASH[1])
H["unit"] = H.o.dt.strftime("%Y-%m-%d"); H["usd"] = n(H, "pnl")
for lab, g in [("as-traded ex-B1", H), ("as-traded ex-B1 ex-washed(date)", H[~H.wash])]:
    a, r = g[g.slope <= -0.05], g[g.slope > -0.05]
    sa = C.stats(a)
    print(f"{lab}: NEG {sa['N']}·{sa['WR']:.0f}%·{sa['avg']:+.3f}·{sa['days']}d·P(<0) {sa['p_neg']:.2f} vs rest {len(r)}·{(r.pct>0).mean()*100:.0f}%·{r.pct.mean():+.3f}")
M1 = C.master(include_b1=True)
M1["slope"] = n(M1, "entry_btc_1h_slope")
for lab, g in [("master WITH B1", M1), ("master WITH B1 ex-washed", M1[~M1.washed30])]:
    a, r = g[g.slope <= -0.05], g[g.slope > -0.05]; sa = C.stats(a)
    print(f"{lab}: NEG {sa['N']}·{sa['WR']:.0f}%·{sa['avg']:+.3f}·{sa['days']}d·P(<0) {sa['p_neg']:.2f} vs rest {len(r)}·{r.pct.mean():+.3f}")
