#!/usr/bin/env python3
"""Report tables for study_reentry_after_loss_avgdown.py (averaging down / sequence martingale / post-exit re-entry appendix).
Reads reports/AVERAGE_DOWN_2026-10-19_rows.csv → writes _summary.csv, _margin.csv, _seqmart.csv, _reentry_appendix.csv and prints
markdown tables (pasted into reports/AVERAGE_DOWN_2026-10-19.md)."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_reentry_after_loss_avgdown as S  # noqa: E402
OUT = S.OUT
rng = np.random.default_rng(11)
NB = 4000


def day_ci(vals, days, stat="sum"):
    """day-block bootstrap 95 % CI of the per-fill mean (stat='mean') or of the total (stat='sum')."""
    g = pd.DataFrame(dict(v=vals, d=days)).groupby("d").v.agg(["sum", "size"])
    if len(g) < 3:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(g), size=(NB, len(g)))
    s = g["sum"].values[idx].sum(1)
    if stat == "mean":
        s = s / g["size"].values[idx].sum(1)
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), float((s > 0).mean()))


def summarize(R):
    out = []
    for (blk, sl), g in R.groupby(["block", "sleeve"]):
        g = g[np.isfinite(g.ref)]
        if not len(g):
            continue
        N = S.SL[sl]["N"]
        for v, m, (a, b) in S.VARIANTS:
            k = f"{v}_{a}{b}"
            var = g[k].astype(float)
            ok = np.isfinite(var)
            gg, var = g[ok], var[ok]
            trig = gg[k + "_trig"].astype(bool)
            dl = var - gg.ref
            ref = gg.ref
            res = dict(block=blk, sleeve=sl, variant=k, X=round(S.SL[sl]["S"] * a / b, 2), n=len(gg), n_trig=int(trig.sum()),
                       trig_orig_losers=int((trig & (ref < 0)).sum()), trig_orig_winners=int((trig & (ref >= 0)).sum()),
                       losers_rescued=int((trig & (ref < 0) & (var >= 0)).sum()),
                       losers_bigger=int((trig & (ref < 0) & (var < ref)).sum()),
                       winners_bigger=int((trig & (ref >= 0) & (var > ref)).sum()),
                       winners_smaller_or_lost=int((trig & (ref >= 0) & (var < ref)).sum()),
                       base_avg=ref.mean(), var_avg=var.mean(), base_wr=(ref > 0).mean() * 100, var_wr=(var > 0).mean() * 100,
                       d_sum_pct=dl.sum(), d_mean_trig=dl[trig].mean() if trig.any() else np.nan,
                       d_usd=dl.sum() / 100 * N, base_usd=ref.sum() / 100 * N)
            lo, hi, p = day_ci((dl / 100 * N).values, gg.day.values, "sum")
            res.update(d_usd_lo=lo, d_usd_hi=hi, p_pos=p)
            pp = (dl / 100 * N).groupby(gg.pair).sum()
            res["top_pair"] = pp.abs().idxmax() if len(pp) and pp.abs().sum() > 0 else ""
            res["top_pair_share"] = float(pp.abs().max() / pp.abs().sum()) if pp.abs().sum() > 0 else np.nan
            res["worst_trade_pct_acct"] = float((var / 100 * N).min() / S.ACCOUNT * 100)
            res["base_worst_pct_acct"] = float((ref / 100 * N).min() / S.ACCOUNT * 100)
            e = (dl / 100 * N).groupby(gg.batch).sum()
            et = trig.groupby(gg.batch).sum()
            e = e[et > 0]
            res["batches_trig"] = len(e); res["batches_pos"] = int((e > 0).sum())
            out.append(res)
    return pd.DataFrame(out)


def margin_table():
    rows = []
    for sl, c in S.SL.items():
        for m in (1, 2):
            for a, b in S.FRACS:
                X = c["S"] * a / b
                N, Lv = c["N"], c["L"]
                eq = S.ACCOUNT - X / 100 * N
                ntot = N * (1 + m)
                liq = (eq / ntot - S.MMR / 100) * 100
                loss_stop = (c["S"] * N + m * (c["S"] - X + 2 * S.T + S.SLIP) * N) / 100
                own = N / Lv
                add_m = m * N / Lv
                free = eq - own
                k_ok = int(np.floor((free - add_m) / S.SLOT_MARGIN)) if free >= add_m else -1
                rows.append(dict(sleeve=sl, m=m, X=round(X, 2), notional_after=ntot, lev=Lv, add_margin=add_m,
                                 liq_dist_from_add_pct=liq, stop_dist_from_add_pct=c["S"] - X,
                                 loss_at_stop_usd=loss_stop, loss_at_stop_pct_acct=loss_stop / S.ACCOUNT * 100,
                                 base_loss_at_stop_pct_acct=c["S"] * N / 100 / S.ACCOUNT * 100,
                                 max_other_slots_open=k_ok))
    return pd.DataFrame(rows)


def concurrency(R):
    """master: other kept master fills open at each add time (lower bound — refused / probe fills not in the pool)."""
    M = S.load_master()
    o = M.opened_at.map(S.ms).values; c = M.closed_at.map(S.ms).values
    out = []
    for k in [f"{v}_{a}{b}" for v, m, (a, b) in S.VARIANTS if v == "AD1"]:
        g = R[(R.block == "MASTER") & R[k + "_trig"].astype(bool)]
        for r in g.itertuples():
            t = getattr(r, k + "_tadd")
            if not np.isfinite(t):
                continue
            n_open = int(((o < t) & (c > t)).sum()) - 1
            out.append(dict(variant=k, sleeve=r.sleeve, others_open=max(n_open, 0)))
    return pd.DataFrame(out)


def seqmart():
    blocks = {"MASTER": S.load_master()}
    B = S.load_bt()
    for s, g in B.groupby("block"):
        blocks[s] = g
    rows = []
    for blk, D in blocks.items():
        for sl, g in D.groupby("sleeve"):
            g = g.sort_values("opened_at")
            p = g.base.astype(float).values
            if len(p) < 6:
                continue
            day = g.opened_at.astype(str).str[:10].values
            prev = np.r_[np.nan, p[:-1]]
            nxt_l = p[1:][prev[1:] < 0]; nxt_w = p[1:][prev[1:] >= 0]
            # bootstrap the difference by day blocks
            dd = pd.DataFrame(dict(v=p[1:], pl=prev[1:] < 0, d=day[1:]))
            days = dd.d.unique()
            diffs = []
            for _ in range(2000):
                pick = rng.choice(days, len(days))
                x = pd.concat([dd[dd.d == q] for q in pick]) if len(days) < 60 else dd.set_index("d").loc[pick].reset_index()
                if x.pl.any() and (~x.pl).any():
                    diffs.append(x.v[x.pl].mean() - x.v[~x.pl].mean())
            lo, hi = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)) if diffs else (np.nan, np.nan)
            # streaks + doubling (cap 2 doublings: 1, 2, 4)
            st = mx = 0
            mult, mults = 1, []
            for v in p:
                mults.append(mult)
                if v < 0:
                    st += 1; mx = max(mx, st); mult = min(mult * 2, 4)
                else:
                    st = 0; mult = 1
            N = S.SL[sl]["N"]
            flat = p / 100 * N; mart = p * np.array(mults) / 100 * N
            dd_f = float((np.cumsum(flat) - np.maximum.accumulate(np.r_[0, np.cumsum(flat)])[1:]).min())
            dd_m = float((np.cumsum(mart) - np.maximum.accumulate(np.r_[0, np.cumsum(mart)])[1:]).min())
            rows.append(dict(block=blk, sleeve=sl, n=len(p), avg=p.mean(), wr=(p > 0).mean() * 100,
                             n_after_loss=len(nxt_l), avg_after_loss=nxt_l.mean() if len(nxt_l) else np.nan,
                             wr_after_loss=(nxt_l > 0).mean() * 100 if len(nxt_l) else np.nan,
                             n_after_win=len(nxt_w), avg_after_win=nxt_w.mean() if len(nxt_w) else np.nan,
                             diff=(nxt_l.mean() - nxt_w.mean()) if len(nxt_l) and len(nxt_w) else np.nan, diff_lo=lo, diff_hi=hi,
                             max_loss_streak=mx, flat_usd=flat.sum(), mart_usd=mart.sum(), flat_maxdd_usd=dd_f, mart_maxdd_usd=dd_m,
                             mart_maxdd_pct_acct=dd_m / S.ACCOUNT * 100, flat_maxdd_pct_acct=dd_f / S.ACCOUNT * 100))
    return pd.DataFrame(rows)


def reentry_appendix():
    """post-exit drift from the stamped post_exit_* columns (net % from the ORIGINAL entry as if still held; tracked ≤ 45 min):
    re-buy at the exit ≈ post_exit_pnl_at_t − pnl_percentage − 0.09 (one more round trip of taker fees)."""
    M = S.load_master()
    rows = []
    for sl, g in M.groupby("sleeve"):
        for lab, h in (("losers", g[g.base < 0]), ("winners", g[g.base >= 0])):
            for t in (5, 15, 30):
                c = f"post_exit_pnl_at_{t}min"
                v = (h[c] - h.pnl_percentage - 0.09).dropna()
                if len(v) < 1:
                    continue
                lo, hi, p = day_ci(v.values, h.loc[v.index, "opened_at"].astype(str).str[:10].values, "mean")
                rows.append(dict(sleeve=sl, cohort=lab, minutes=t, n=len(v), reentry_avg=v.mean(), wr=(v > 0).mean() * 100, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def md(df, cols, fmt=None):
    fmt = fmt or {}
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        L.append("| " + " | ".join(fmt.get(c, "{}").format(x) if not (isinstance(x, float) and np.isnan(x)) else "—" for c, x in zip(cols, r)) + " |")
    return "\n".join(L)


def main():
    R = pd.read_csv(OUT + "_rows.csv", low_memory=False)
    R = R[R.get("unres", False) != True]  # noqa: E712
    Sm = summarize(R); Sm.to_csv(OUT + "_summary.csv", index=False)
    Mg = margin_table(); Mg.to_csv(OUT + "_margin.csv", index=False)
    Sq = seqmart(); Sq.to_csv(OUT + "_seqmart.csv", index=False)
    Ra = reentry_appendix(); Ra.to_csv(OUT + "_reentry_appendix.csv", index=False)
    Cc = concurrency(R); Cc.to_csv(OUT + "_concurrency.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
    print(Sm.round(3).to_string())
    print(Mg.round(2).to_string())
    print(Sq.round(3).to_string())
    print(Ra.round(3).to_string())
    print(Cc.groupby(["variant", "sleeve"]).others_open.describe().round(2).to_string())


if __name__ == "__main__":
    main()
