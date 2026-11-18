#!/usr/bin/env python3
"""DIST_WL stage 3 (PREREG reports/DIST_WL_2026-11-18_PREREG.md): the STRENGTH theme (every HOLDOUT passer is one correlated factor);
representative picked on DESIGN only (highest DESIGN |t_net| 1D null-passer of the theme = rsi1h T3/3, keep only RSI1h ≥ 74.07).
HOLDOUT day-block CI + random-cut null for the representative · sibling table · before → after (one position per pair re-run, LIVE
exit) · blocked side priced vs the locked EXPECTANCY bar · Kelly (trade / day) · concurrency · T5A cross-check · charts · trade CSV.
Usage: venv/bin/python -I scripts/dist_wl_final.py"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import dist_wl_analyze as A  # noqa: E402
import dist_wl_survivors as SV  # noqa: E402
import unified_short_lib as L  # noqa: E402
OUT = A.OUT; RP = os.path.join(L.R, "reports")
REP = dict(kind="1D", var="rsi1h", lo=74.07103, hi=np.inf, mode="keep_only", var2="", lo2=np.nan, hi2=np.nan)
SIB = [("rsi1h", "T3/3"), ("rsi1h", "Q5/5"), ("di_diff1h", "T3/3"), ("di_diff1h", "Q5/5"), ("gap15", "T3/3"), ("ret4h_pct", "T3/3"),
       ("ret24h_pct", "T3/3"), ("ret1h_pct", "Q5/5"), ("rsi15", "T3/3"), ("q5", "T3/3"), ("q5", "Q5/5"), ("hrs_since_peak", "Q2/5"),
       ("top_acct_chg4h", "T1/3"), ("top_acct_chg4h_rel", "T1/3"), ("pos_minus_acct_rel", "Q5/5"), ("glob_chg4h", "Q1/5")]


def dayboot_mean(v, day, B=4000, seed=3):
    return A.block_boot_mean(v, day, B, seed)


def expectancy_bar(v, day, pair, be_wr):
    lo, hi, q95 = dayboot_mean(v, day)
    neg = -v[v < 0].sum() if (v < 0).any() else 0
    loss_day = (-pd.Series(np.minimum(v, 0)).groupby(day).sum()).max() / neg if neg else np.nan
    loss_pair = (-pd.Series(np.minimum(v, 0)).groupby(pair).sum()).max() / neg if neg else np.nan
    return dict(N=len(v), WR=(v > 0).mean() * 100, be_wr=be_wr, avg=v.mean(), ci_hi=hi, ci95_one_sided=q95, windows=len(np.unique(day)),
                top_day_loss_share=loss_day, top_pair_loss_share=loss_pair,
                passes=bool((v > 0).mean() * 100 < be_wr and q95 < 0 and len(np.unique(day)) >= 8 and loss_day < 0.5 and loss_pair < 0.5 and len(v) >= 15))


if __name__ == "__main__":
    B_ = pd.read_pickle(os.path.join(OUT, "base.pkl")); S = pd.read_pickle(os.path.join(OUT, "rows.pkl"))
    T = pd.read_pickle(os.path.join(OUT, "screen.pkl")); P = pd.read_pickle(os.path.join(OUT, "design_passers.pkl"))
    R1 = json.load(open(os.path.join(OUT, "stage1.json")))
    des = B_.entry_ms.values < A.HOLD_START; hol = ~des
    v = B_.net_LIVE.values; day = B_.day.values
    rr = T[(T.kind == "1D") & (T["var"] == "rsi1h") & (T.bin == "T3/3")].iloc[0]
    REP.update(lo=float(rr.lo), hi=float(rr.hi))
    Rz = {"rep_edge": float(rr.lo)}
    # ── siblings (all already read on HOLDOUT in stage 2)
    sib = []
    for var, b in SIB:
        r = T[(T.kind == "1D") & (T["var"] == var) & (T.bin == b)].iloc[0].to_dict()
        r["mode"] = "keep_only" if r["m1_net"] > r["m0_net"] else "block"
        k = SV.keep_mask(B_, r)
        row = dict(var=var, bin=b, lo=r["lo"], hi=r["hi"], mode=r["mode"], t_net=r["t_net"], t_reach=r["t_reach"],
                   null_pass=bool(abs(r["t_net"]) > R1["null_net"]["q95"] or abs(r["t_reach"]) > R1["null_reach"]["q95"] or abs(r["t_worked"]) > R1["null_worked"]["q95"]
                                  or abs(r["t_reach"]) > R1["null_reach"]["q95_theory"] and var in sum(A.THEORY.values(), [])
                                  or abs(r["t_worked"]) > R1["null_worked"]["q95_theory"] and var in sum(A.THEORY.values(), [])))
        for nm, m in (("D", des), ("H", hol)):
            row[f"{nm}_N_keep"] = int((m & k).sum()); row[f"{nm}_keep"] = v[m & k].mean(); row[f"{nm}_block"] = v[m & ~k].mean()
            row[f"{nm}_base"] = v[m].mean()
        hv = B_.half.values
        row["halves_keep_pos"] = f"{sum(v[(hv == h) & k].mean() > 0 for h, _, _ in L.HALVES if ((hv == h) & k).sum() >= 20)}/" \
                                 f"{sum(((hv == h) & k).sum() >= 20 for h, _, _ in L.HALVES)}"
        sib.append(row)
    SIBT = pd.DataFrame(sib); SIBT.to_csv(os.path.join(OUT, "siblings.csv"), index=False)
    print(SIBT.round(3).to_string(), flush=True)
    # ── representative: holdout CI, random-cut null
    k = SV.keep_mask(B_, REP)
    lo_, hi_ = A.block_boot(v[hol], day[hol], k[hol], B=4000)
    rng = np.random.default_rng(5); nk = int((hol & k).sum()); hv_ = v[hol]
    rnd = np.array([hv_[rng.choice(len(hv_), nk, replace=False)].mean() for _ in range(5000)]) - hv_.mean()
    obs = v[hol & k].mean() - hv_.mean()
    # day-block random null: keep whole random days' trades until ≈ nk
    Rz["rep_holdout"] = dict(delta=obs, ci=[lo_, hi_], p_random_cut=float((rnd >= obs).mean()))
    lo_d, hi_d = A.block_boot(v[des], day[des], k[des], B=4000)
    Rz["rep_design"] = dict(delta=v[des & k].mean() - v[des].mean(), ci=[lo_d, hi_d])
    # how many of the 538 HOLDOUT reads one would pass by chance: random keep-only / block masks of the same sizes
    hp = []
    for _, r in P.sample(min(120, len(P)), random_state=1).iterrows():
        kk = SV.keep_mask(B_, r); n_ = int((hol & kk).sum())
        idx = np.nonzero(hol)[0]
        ok = 0
        for _ in range(40):
            rk = np.zeros(len(v), bool); rk[rng.choice(idx, n_, replace=False)] = True; rk |= des & kk
            ok += bool(v[hol & rk].mean() > v[hol].mean() and v[hol & rk].mean() > 0 and B_.net20_LIVE.values[hol & rk].mean() > 0)
        hp.append(ok / 40)
    Rz["random_mask_holdout_pass_rate"] = float(np.mean(hp)); Rz["observed_holdout_pass_rate"] = float(P.h_pass.mean())
    Rz["n_design_passers"] = int(len(P)); Rz["n_holdout_pass"] = int(P.h_pass.sum())
    # per half-year + regime table for the representative
    rows = []
    for h, _, _ in L.HALVES:
        m = B_.half.values == h
        rows.append(dict(period=h, N=int(m.sum()), base=v[m].mean(), N_keep=int((m & k).sum()), keep=v[m & k].mean(), block=v[m & ~k].mean()))
    for g, nm in ((-1, "BTC 1d bear"), (1, "BTC 1d bull")):
        for blk, mm in (("DESIGN", des), ("HOLDOUT", hol)):
            m = (B_.btc_d1_regime.values == g) & mm
            if m.sum():
                rows.append(dict(period=f"{nm} · {blk}", N=int(m.sum()), base=v[m].mean(), N_keep=int((m & k).sum()),
                                 keep=v[m & k].mean() if (m & k).any() else np.nan, block=v[m & ~k].mean()))
    PER = pd.DataFrame(rows); PER.to_csv(os.path.join(OUT, "rep_periods.csv"), index=False); print(PER.round(3).to_string(), flush=True)
    # concentration / b5 / slip on the representative (both blocks)
    conc = {}
    for nm, mm in (("DESIGN", des), ("HOLDOUT", hol)):
        ex = v[mm & k] - v[mm].mean(); tot = ex.sum()
        conc[nm] = dict(top_pair=pd.Series(ex).groupby(B_.pair.values[mm & k]).sum().max() / tot,
                        top_day=pd.Series(ex).groupby(day[mm & k]).sum().max() / tot,
                        b5_keep=A.b5(v[mm & k]), s20_keep=B_.net20_LIVE.values[mm & k].mean(), s20_base=B_.net20_LIVE.values[mm].mean(),
                        b5_base=A.b5(v[mm]))
    Rz["rep_conc"] = conc
    # ── before → after cards (full row frame, one position per pair re-run)
    span = (int(S.entry_ms.min() // A.DAY), int(S.LIVE_x.max() // A.DAY))
    Sk = SV.keep_mask(S, REP); allm = np.ones(len(S), bool)
    Sdes = S.entry_ms.values < A.HOLD_START
    cards = []
    for blk, mm, sp in (("ALL", allm, span), ("DESIGN", Sdes, (span[0], int(A.HOLD_START // A.DAY) - 1)),
                        ("HOLDOUT", ~Sdes, (int(A.HOLD_START // A.DAY), span[1]))):
        for lab, msk in (("before (L1)", mm), ("after (keep RSI1h ≥ 74.07)", mm & Sk), ("blocked side", mm & ~Sk)):
            c = SV.card(S, msk, lab, sp); c["block"] = blk; cards.append(c)
    CARD = pd.DataFrame(cards); CARD.to_csv(os.path.join(OUT, "before_after.csv"), index=False)
    print(CARD.round(3).to_string(), flush=True)
    # blocked side vs the locked expectancy bar (sleeve breakeven WR from the before card of the same block)
    eb = {}
    for blk, mm in (("DESIGN", des), ("HOLDOUT", hol), ("ALL", np.ones(len(v), bool))):
        al = -v[mm & (v <= 0)].mean(); aw = v[mm & (v > 0)].mean()
        x = mm & ~k
        eb[blk] = expectancy_bar(v[x], day[x], B_.pair.values[x], al / (al + aw) * 100)
    Rz["blocked_expectancy_bar"] = eb
    # concurrency (clustering): open positions at each entry, same-day count
    for nm, msk in (("before", allm), ("after", Sk)):
        m = A.one_pos(S, msk); e = S.entry_ms.values[m]; x = S.LIVE_x.values[m]
        o = np.sort(e); xs = np.sort(x)
        openn = np.searchsorted(o, e, side="right") - np.searchsorted(xs, e, side="right")
        dd = pd.Series(1, index=S.day.values[m]).groupby(level=0).sum()
        Rz[f"conc_{nm}"] = dict(open_mean=float(openn.mean()), open_p95=float(np.percentile(openn, 95)), open_max=int(openn.max()),
                               per_active_day_mean=float(dd.mean()), per_active_day_p95=float(dd.quantile(0.95)))
    # T5A cross-check of the same cut
    S5 = S.copy()
    for c_ in ("k", "x"):
        S5[f"LIVE_{c_}"] = S[f"T5A_{c_}"]
    S5["net_LIVE"] = S.net_T5A; S5["net20_LIVE"] = S.net20_T5A
    Rz["t5a"] = [SV.card(S5, allm, "T5A before", span), SV.card(S5, Sk, "T5A after", span)]
    json.dump(Rz, open(os.path.join(OUT, "stage3.json"), "w"), indent=1, default=float)
    print(json.dumps(Rz, indent=1, default=float), flush=True)
    # ── charts: anchored path (short-favourable %, + = price below the entry)
    Pth = np.load(os.path.join(OUT, "paths.npy")); Pb = Pth[S.in_LIVE.values]
    xh = (np.arange(Pb.shape[1]) - A.__dict__.get("PRE", 24)) * 5 / 60
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a_, groups, ttl in ((ax[0], [("winners", v > 0, "#2a9d8f"), ("losers", v <= 0, "#e76f51")], "Winners vs losers (LIVE exit), all L1 trades"),
                            (ax[1], [("kept: RSI1h ≥ 74", k, "#264653"), ("blocked: RSI1h < 74", ~k, "#e9c46a")], "Kept vs blocked (STRENGTH cut)")):
        for nm, m, col in groups:
            med = np.nanmedian(Pb[m], 0); q1 = np.nanpercentile(Pb[m], 25, 0); q3 = np.nanpercentile(Pb[m], 75, 0)
            a_.plot(xh, med, color=col, lw=2, label=f"{nm} (N {m.sum():,}, avg {v[m].mean():+.2f} %)")
            a_.fill_between(xh, q1, q3, color=col, alpha=0.15)
        a_.axvline(0, color="grey", lw=0.8); a_.axhline(0, color="grey", lw=0.8)
        a_.set_title(ttl, fontsize=10); a_.set_xlabel("hours from entry (red VSW candle close)"); a_.legend(fontsize=8, loc="upper left")
        a_.grid(alpha=0.25)
    ax[0].set_ylabel("short-side move from the entry, % (median, IQR band)")
    fig.tight_layout(); fig.savefig(os.path.join(RP, "DIST_WL_2026-11-18_paths.png"), dpi=130)
    # ── trade CSV
    TR = B_.copy(); TR["keep_strength"] = k
    TR["block_name"] = np.where(des, "DESIGN", "HOLDOUT")
    TR["entry_utc"] = pd.to_datetime(TR.entry_ms, unit="ms")
    fo = os.path.join(RP, "DIST_WL_2026-11-18_trades.csv")
    TR.to_csv(fo, index=False)
    if os.path.getsize(fo) > 3e6:
        TR.to_csv(fo + ".gz", index=False, compression="gzip"); os.remove(fo)
    print("done", flush=True)
