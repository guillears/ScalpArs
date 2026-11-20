#!/usr/bin/env python3
"""DIST_FB_LOSERS_2026-11-20 — cohort winners vs losers, the luck read, and before → after for the leads (each sleeve apart).

DIST:  real = the 5 current-batch fills (master holds 0 DIST fills; sweep_separators refuses < 6) · backtest = the 1,547 DL 586 fills
       (reports/ENTRY_WL_2026-11-18_data/dist.pkl, Exit B net with model costs + real funding; DESIGN < 2025-07 ≤ HOLDOUT).
FB:    real = master kept fills (reports/MASTER_POOL_stacked.csv stack_keep, stack_pct) + the 3 current-batch fills · backtest = the
       OOS_REAL_FLATBASE rebuild (wl_features.csv, today's exit X0, real funding) kept by NEAR_BASE (dist_base > 2.09) = 614.
Reads: ① luck — how often the backtest produces a run as bad as the batch · ② every real-stamped entry variable that maps onto a backtest
column: W/L medians, AUC per block, DESIGN-tercile net per block, where each live fill sits (percentile) · ③ BTC daily regime at sign
granularity (DIST, HOLDOUT) · ④ before → after (three blocks where they exist) for the frozen leads, one-size DCR (X = 0.61 × equity),
expectancy bar (blocked WR vs breakeven, window-clustered bootstrap, ≥ 8 windows, N ≥ 15), LOO, per-pair / per-window concentration.
Output: reports/DIST_FB_LOSERS_2026-11-20_data/wl_*.csv + wl_tables.md.
Usage: venv/bin/python -I scripts/DIST_FB_LOSERS_2026-11-20_wl.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "best_case"))
import review_math as RM  # noqa: E402
import one_size as OS  # noqa: E402

D = os.path.join(ROOT, "reports", "DIST_FB_LOSERS_2026-11-20_data")
EXPORT = os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-11-20_16-01-39.csv")
X_SIZE = 0.61
OUTMD = []


def P(s=""):
    print(s)
    OUTMD.append(s)


def md(df, fl=2):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:+.{fl}f}" if abs(v) < 1000 else f"{v:,.0f}")
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def auc(x, y):
    """P(value of a y=1 fill > value of a y=0 fill), ties 0.5."""
    x = np.asarray(x, float); y = np.asarray(y, int)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    r = pd.Series(x).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def window_boot_mean(v, win, B=4000, seed=11):
    """window-clustered bootstrap of the mean: resample windows (market moments), not fills → P(mean < 0), 95 % CI."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float); win = np.asarray(win)
    u = np.unique(win)
    groups = [v[win == w] for w in u]
    sums = np.array([g.sum() for g in groups]); ns = np.array([len(g) for g in groups])
    ix = rng.integers(0, len(u), (B, len(u)))
    means = sums[ix].sum(1) / ns[ix].sum(1)
    return float((means < 0).mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def dcr_one_size(pct, day, d0, d1, x=X_SIZE):
    """one-size DCR %/day over every calendar day d0..d1 (day = 'YYYY-MM-DD' of the close / entry)."""
    s = pd.Series(np.asarray(pct, float) * x / 100).groupby(pd.to_datetime(pd.Series(day)).values).sum()
    days = pd.date_range(d0, d1, freq="D")
    r = s.reindex(days, fill_value=0.0).values
    return (np.prod(1 + r) ** (1 / len(r)) - 1) * 100


def breakeven_wr(p):
    p = np.asarray(p, float)
    w, l_ = p[p > 0], -p[p <= 0]
    if not len(w) or not len(l_):
        return np.nan
    return l_.mean() / (w.mean() + l_.mean()) * 100


# ───────────────────────────── DIST ─────────────────────────────
def dist_part(o):
    S = pd.read_pickle(os.path.join(ROOT, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl"))
    S["W"] = (S.net > 0).astype(int)
    S["WK"] = S.worked.astype(float)
    S["date"] = pd.to_datetime(S.entry_ms, unit="ms").dt.strftime("%Y-%m-%d")
    real = o[o.entry_strategy == "DISTRIBUTION_SHORT"].copy()
    P("## DIST — ① luck: how often does the backtest give a batch this bad?")
    obs_avg, obs_w = real.pnl_percentage.mean(), int((real.pnl_percentage > 0).sum())
    rng = np.random.default_rng(5)
    rows = []
    for nm, pool in (("ALL 2023-07 → 2026-11", S), ("HOLDOUT 2025-07 → 2026-11", S[S.block == "HOLDOUT"]),
                     ("2026 only", S[S.date >= "2026-01-01"])):
        v = pool.net.values
        draw = v[rng.integers(0, len(v), (100_000, len(real)))]
        rows.append(dict(pool=nm, N=len(v), WR=(v > 0).mean() * 100, avg=v.mean(),
                         p_avg_le_obs=(draw.mean(1) <= obs_avg).mean() * 100, p_wins_le_obs=((draw > 0).sum(1) <= obs_w).mean() * 100))
    P(f"batch: {len(real)} fills · {obs_w} win · avg {obs_avg:+.2f} % (model-free, as booked)")
    P(md(pd.DataFrame(rows)))
    # ② variables that map real stamp → backtest column
    MAP = [("entry_dist_rsi1h", "rsi1h", 1), ("entry_dist_pos_chg_4h", "top_pos_chg4h", 1), ("entry_dist_pos_ratio", "top_pos_ls", 1),
           ("entry_dist_vsw_mult", "vmult15", 1), ("entry_dist_bar_ret_pct", "bar_ret_pct", 1), ("entry_dist_atr_pct", "atr15_pct", 1),
           ("entry_dist_di_diff1h", "di_diff1h", 1), ("entry_dist_gap15", "gap15", 1), ("entry_frenzy_hours", "hrs_since_spike", 1),
           ("entry_pair_volume_24h_usd", "vol24", 1), ("entry_dist_q5_usd", "q5", 1), ("entry_funding_rate", "fund_last", 100),
           ("entry_btc_1d_ret_pct", "btc_ret1d", 1)]
    P("\n## DIST — ② winners vs losers on every real-stamped variable that the backtest also carries")
    P("AUC = P(a winner's value > a loser's) on booked net > 0; AUC_wk = same on the ENTRY label (WORKED = +1 % before the stop). "
      "Net by DESIGN tercile (edges frozen on DESIGN). Live fills: percentile of each fill's value in the backtest (L = loser, W = winner).")
    tab = []
    for rc, bc, sc in MAP:
        if bc not in S:
            continue
        x = pd.to_numeric(S[bc], errors="coerce")
        q1, q2 = x[S.block == "DESIGN"].quantile([1 / 3, 2 / 3])
        t = np.select([x <= q1, x <= q2], ["T1", "T2"], "T3")
        rv = pd.to_numeric(real[rc], errors="coerce") * sc
        pct = [f"{'W' if p > 0 else 'L'}{(x < v).mean() * 100:.0f}" if pd.notna(v) else "–" for v, p in zip(rv, real.pnl_percentage)]
        row = dict(var=bc, real_stamp=rc.replace("entry_", ""),
                   AUC_D=auc(x[S.block == "DESIGN"], S.W[S.block == "DESIGN"]), AUC_H=auc(x[S.block == "HOLDOUT"], S.W[S.block == "HOLDOUT"]),
                   AUCwk_H=auc(x[S.block == "HOLDOUT"], S.WK[S.block == "HOLDOUT"]))
        for blk in ("DESIGN", "HOLDOUT"):
            for tt in ("T1", "T3"):
                m = (S.block == blk) & (t == tt) & x.notna()
                row[f"{blk[0]}_{tt}"] = S.net[m].mean()
        row["live (pctile)"] = " ".join(pct)
        tab.append(row)
    T = pd.DataFrame(tab)
    T.to_csv(os.path.join(D, "wl_dist_vars.csv"), index=False)
    P(md(T))
    # ③ BTC daily regime, sign granularity, HOLDOUT (DESIGN has 1 bear-daily fill)
    P("\n## DIST — ③ BTC daily regime (1d EMA50 vs EMA200 sign), DAY units beside fills")
    rr = []
    for blk in ("DESIGN", "HOLDOUT"):
        for reg in (1.0, -1.0):
            m = (S.block == blk) & (S.btc_d1_regime == reg)
            g = S[m]
            if not len(g):
                continue
            dd = g.groupby("date").net.mean()
            rr.append(dict(block=blk, btc_daily="bull" if reg > 0 else "bear", N=len(g), days=g.date.nunique(), WR=(g.net > 0).mean() * 100,
                           avg=g.net.mean(), day_avg=dd.mean(), worked=g.WK.mean() * 100, stop_share=(g.k == 1).mean() * 100 if "k" in g else np.nan))
    P(md(pd.DataFrame(rr)))
    P("Live batch: every DIST fill opened with BTC's 4h EMA50 ≈ 10 % below its EMA200 and BTC −1.7 … −2.9 % on the day (bear tape).")
    # ④ the fresh-spike lead (hypothesis raised by these losers: 4 of 5 fills ≤ 1.2 h after the spike) — frozen DESIGN tercile T1
    x = S.hrs_since_spike
    q1 = x[S.block == "DESIGN"].quantile(1 / 3)
    P(f"\n## DIST — ④ lead check: 'fresh spike' = hours since the spike ≤ {q1:.2f} h (DESIGN lowest tercile, frozen before reading HOLDOUT)")
    rows = []
    for blk, m0 in (("DESIGN", S.block == "DESIGN"), ("HOLDOUT", S.block == "HOLDOUT"), ("2026", S.date >= "2026-01-01")):
        for side, m in (("fresh (would be blocked)", m0 & (x <= q1)), ("rest (kept)", m0 & (x > q1))):
            g = S[m]
            rows.append(dict(block=blk, side=side, N=len(g), days=g.date.nunique(), WR=(g.net > 0).mean() * 100, avg=g.net.mean(),
                             worked=g.WK.mean() * 100, reach=g.reach.mean()))
    rl = real.entry_frenzy_hours <= q1
    rows.append(dict(block="batch (live)", side="fresh", N=int(rl.sum()), days=real[rl].opened_at.str[:10].nunique(),
                     WR=(real[rl].pnl_percentage > 0).mean() * 100, avg=real[rl].pnl_percentage.mean(), worked=np.nan, reach=np.nan))
    rows.append(dict(block="batch (live)", side="rest", N=int((~rl).sum()), days=real[~rl].opened_at.str[:10].nunique(),
                     WR=(real[~rl].pnl_percentage > 0).mean() * 100, avg=real[~rl].pnl_percentage.mean(), worked=np.nan, reach=np.nan))
    P(md(pd.DataFrame(rows)))
    # ⑤ the stop-hit cohort: how fast, how far after
    P("\n## DIST — ⑤ the backtest's stopped fills: how fast and how far (context for the four live stops)")
    st = S[S.label.isin(["GAVE_BACK", "NEVER_WORKED"]) | (S.net < 0)]
    P(f"backtest losers {len(st)} of {len(S)} ({len(st) / len(S) * 100:.0f} %) · label split: " +
      " · ".join(f"{k} {v}" for k, v in S[S.net < 0].label.value_counts().items()))
    return S


# ───────────────────────────── FLATBASE ─────────────────────────────
def fb_part(o):
    X = pd.read_csv(os.path.join(ROOT, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    X = X[X.X0_why.isin(["TP", "STOP", "TIME"]) & (X.dist_base > 2.09)].copy()
    X["date"] = pd.to_datetime(X.cf_entry_t, unit="ms").dt.strftime("%Y-%m-%d")
    X["blk"] = np.where(X.date < "2025-07-01", "DESIGN", "HOLDOUT")
    X["W"] = (X.X0_net > 0).astype(int)
    X["win"] = pd.to_datetime(X.cf_entry_t, unit="ms").dt.floor("4h")          # one 4-h close = one market window
    m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    mk = m[(m.entry_strategy == "FLATBASE_SHORT") & (m.stack_keep.astype(str).str.lower() == "true")].copy()
    mk["pct"] = mk.stack_pct
    mk["batch"] = np.where(mk.opened_at < "2026-10-23T18:30", "B24", np.where(mk.opened_at < "2026-11-11T20:49", "B25",
                           np.where(mk.opened_at < "2026-11-14T20:30", "B26", "B27")))
    cb = o[o.entry_strategy == "FLATBASE_SHORT"].copy()
    cb["pct"] = cb.pnl_percentage; cb["batch"] = "B28 (current)"
    R = pd.concat([mk, cb], ignore_index=True)
    R["win"] = pd.to_datetime(R.opened_at).dt.floor("4h")
    R["fund_pos"] = R.entry_funding_rate > 0
    P("\n## FLATBASE — ① luck: how often does the backtest give 2 stops in 3 fills?")
    v = X.X0_net.values
    rng = np.random.default_rng(6)
    draw = v[rng.integers(0, len(v), (100_000, 3))]
    P(f"backtest (NEAR_BASE-kept) {len(X)} · WR {(v > 0).mean() * 100:.0f} % · avg {v.mean():+.2f} · P(≤ 1 win in 3) "
      f"{((draw > 0).sum(1) <= 1).mean() * 100:.0f} % · P(avg ≤ batch avg {cb.pct.mean():+.2f}) {(draw.mean(1) <= cb.pct.mean()).mean() * 100:.0f} %")
    wins_per = X.groupby("win").X0_why.apply(lambda s: (s == "STOP").sum())
    P(f"backtest 4-h windows with ≥ 2 Flatbase entries: {(X.groupby('win').size() >= 2).sum()} · of those, both/all stopped: "
      f"{((X.groupby('win').size() >= 2) & (wins_per == X.groupby('win').size())).sum()} · windows with ≥ 2 stops: {(wins_per >= 2).sum()}")
    P(f"real kept fills {len(R)} (master {len(mk)} + batch {len(cb)}) · {int((R.pct > 0).sum())} W / {int((R.pct <= 0).sum())} L · avg {R.pct.mean():+.2f}")
    # ② variables
    MAP = [("entry_frenzy_hours", "hours", 1), ("entry_frenzy_run_pct", "run_pct", 1), ("entry_frenzy_vs_vwap_pct", "vs_vwap", 1),
           ("entry_flatbase_off_peak_pct", "off_peak_pct", 1), ("entry_flatbase_peak_age_h", "peak_age_h", 1),
           ("entry_flatbase_flat14_pct", "flat14_pct", 1), ("entry_funding_rate", "funding_rate_pct", 100)]
    X["flat14_pct"] = X.flat14 * 100
    P("\n## FLATBASE — ② winners vs losers (backtest AUC per block, DESIGN-tercile net; real fills: W/L medians; live = percentile)")
    tab = []
    for rc, bc, sc in MAP:
        x = pd.to_numeric(X[bc], errors="coerce")
        q1, q2 = x[X.blk == "DESIGN"].quantile([1 / 3, 2 / 3])
        t = np.select([x <= q1, x <= q2], ["T1", "T2"], "T3")
        rv = pd.to_numeric(R[rc], errors="coerce") * sc
        cbv = pd.to_numeric(cb[rc], errors="coerce") * sc
        row = dict(var=bc, AUC_D=auc(x[X.blk == "DESIGN"], X.W[X.blk == "DESIGN"]), AUC_H=auc(x[X.blk == "HOLDOUT"], X.W[X.blk == "HOLDOUT"]),
                   D_T1=X.X0_net[(X.blk == "DESIGN") & (t == "T1")].mean(), D_T3=X.X0_net[(X.blk == "DESIGN") & (t == "T3")].mean(),
                   H_T1=X.X0_net[(X.blk == "HOLDOUT") & (t == "T1")].mean(), H_T3=X.X0_net[(X.blk == "HOLDOUT") & (t == "T3")].mean(),
                   real_W_med=rv[R.pct > 0].median(), real_L_med=rv[R.pct <= 0].median(), real_cov=f"{rv.notna().mean() * 100:.0f} %",
                   live=" ".join(f"{'W' if p > 0 else 'L'}{(x < v_).mean() * 100:.0f}" if pd.notna(v_) else "–" for v_, p in zip(cbv, cb.pct)))
        tab.append(row)
    T = pd.DataFrame(tab)
    T.to_csv(os.path.join(D, "wl_fb_vars.csv"), index=False)
    P(md(T))
    # ③ before → after: the funding-sign lead (block a short when the entry funding rate > 0) — three blocks
    P("\n## FLATBASE — ③ lead check, three blocks: block the short when the entry funding rate is POSITIVE (sign; frozen, no threshold fitted)")
    rows = []

    def block_row(name, pct, fund_pos, day, win, pair, d0, d1):
        pct = np.asarray(pct, float); fp = np.asarray(fund_pos, bool)
        kept, blk = pct[~fp], pct[fp]
        be = breakeven_wr(pct)
        pneg, lo, hi = window_boot_mean(blk, np.asarray(win)[fp]) if fp.sum() >= 2 else (np.nan, np.nan, np.nan)
        loss = pd.Series(blk[blk < 0]).groupby(np.asarray(pair)[fp][blk < 0]).sum() if (blk < 0).any() else pd.Series(dtype=float)
        top_pair = (loss.min() / loss.sum() * 100) if len(loss) else np.nan
        return dict(block=name, N=len(pct), before_avg=pct.mean(), before_DCR=dcr_one_size(pct, day, d0, d1),
                    after_N=len(kept), after_avg=kept.mean() if len(kept) else np.nan,
                    after_DCR=dcr_one_size(kept, np.asarray(day)[~fp], d0, d1),
                    blocked_N=int(fp.sum()), blocked_windows=int(pd.Series(np.asarray(win)[fp]).nunique()), blocked_WR=(blk > 0).mean() * 100 if len(blk) else np.nan,
                    breakeven_WR=be, blocked_avg=blk.mean() if len(blk) else np.nan, P_blocked_mean_lt0=pneg, top_pair_loss_share=top_pair)

    for blk_name in ("DESIGN", "HOLDOUT"):
        g = X[X.blk == blk_name]
        rows.append(block_row(f"backtest {blk_name}", g.X0_net, g.funding_rate_pct > 0, g.date, g.win, g.pair, g.date.min(), g.date.max()))
    for b in sorted(R.batch.unique()):
        g = R[R.batch == b]
        cd = pd.to_datetime(g.closed_at, format="mixed").dt.strftime("%Y-%m-%d")
        rows.append(block_row(f"master {b}" if "current" not in b else b, g.pct, g.fund_pos, cd, g.win, g.pair,
                              str(pd.to_datetime(g.opened_at).min())[:10], str(pd.to_datetime(g.closed_at, format='mixed').max())[:10]))
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(D, "wl_fb_funding_beforeafter.csv"), index=False)
    P(md(B))
    d = np.where(R.fund_pos, -R.pct, 0.0)          # Δ of blocking = − the blocked fill's %
    P("real fills LOO: " + RM.loo_text(list(d), labels=list(R.pair + " " + R.opened_at.str[5:10]), unit="% pts"))
    P(f"real fills: positive funding {int(R.fund_pos.sum())} · {int((R[R.fund_pos].pct > 0).sum())} W / {int((R[R.fund_pos].pct <= 0).sum())} L · "
      f"zero / negative {int((~R.fund_pos).sum())} · {int((R[~R.fund_pos].pct > 0).sum())} W / {int((R[~R.fund_pos].pct <= 0).sum())} L · "
      f"value 0.00005 (= +0.005 %, the floor many coins print) on {int((R.entry_funding_rate.round(6) == 0.00005).sum())} fills")
    return X, R


def main():
    o = pd.read_csv(EXPORT, low_memory=False)
    o = o[(o.status == "CLOSED")]
    if "is_probe" in o:
        o = o[o.is_probe.astype(str).str.lower() != "true"]
    pd.set_option("display.width", 250)
    dist_part(o)
    fb_part(o)
    with open(os.path.join(D, "wl_tables.md"), "w") as f:
        f.write("\n".join(OUTMD) + "\n")


if __name__ == "__main__":
    main()


# ───────────── FB lead 2: funding ABOVE the base rate (> +0.005 % per settlement = DESIGN top-tercile edge, frozen) ─────────────
def fb_funding_above_base():
    THR = 0.005           # % per settlement; = the DESIGN 2/3 quantile of funding_rate_pct (and Binance's base rate) — frozen
    X = pd.read_csv(os.path.join(ROOT, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    X = X[X.X0_why.isin(["TP", "STOP", "TIME"]) & (X.dist_base > 2.09)].copy()
    X["date"] = pd.to_datetime(X.cf_entry_t, unit="ms").dt.strftime("%Y-%m-%d")
    X["win"] = pd.to_datetime(X.cf_entry_t, unit="ms").dt.floor("4h").astype(str)
    X["blk"] = np.where(X.date < "2025-07-01", "DESIGN", "HOLDOUT")
    o = pd.read_csv(EXPORT, low_memory=False)
    cb = o[(o.status == "CLOSED") & (o.entry_strategy == "FLATBASE_SHORT")].copy()
    cb["pct"] = cb.pnl_percentage; cb["batch"] = "B28 (current)"
    m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    mk = m[(m.entry_strategy == "FLATBASE_SHORT") & (m.stack_keep.astype(str).str.lower() == "true")].copy()
    mk["pct"] = mk.stack_pct
    mk["batch"] = np.where(mk.opened_at < "2026-10-23T18:30", "B24", np.where(mk.opened_at < "2026-11-11T20:49", "B25",
                           np.where(mk.opened_at < "2026-11-14T20:30", "B26", "B27")))
    R = pd.concat([mk, cb], ignore_index=True)
    R["blkd"] = R.entry_funding_rate * 100 > THR
    R["win"] = pd.to_datetime(R.opened_at).dt.floor("4h").astype(str)
    P(f"\n## FLATBASE — ④ lead check: block when the entry funding rate > +{THR} % per settlement (longs paying above the base rate)")
    P("Threshold = the DESIGN top-tercile edge of the rebuild (it equals Binance's base rate) — not fitted on the losers. Raised by: "
      "the real-fill sweep's top separator (funding sign, null p 0.14) and WL_DEEP_FLATBASE (AUC 0.42, 4 of 5 periods).")
    rows = []
    for name, g, pct, blkd, day, d0, d1 in (
            [(f"backtest {b}", X[X.blk == b], X[X.blk == b].X0_net, X[X.blk == b].funding_rate_pct > THR, X[X.blk == b].date,
              X[X.blk == b].date.min(), X[X.blk == b].date.max()) for b in ("DESIGN", "HOLDOUT")] +
            [(f"backtest ALL", X, X.X0_net, X.funding_rate_pct > THR, X.date, X.date.min(), X.date.max())] +
            [(f"master {b}" if "current" not in b else b, R[R.batch == b], R[R.batch == b].pct, R[R.batch == b].blkd,
              pd.to_datetime(R[R.batch == b].closed_at, format="mixed").dt.strftime("%Y-%m-%d"),
              str(pd.to_datetime(R[R.batch == b].opened_at).min())[:10], str(pd.to_datetime(R[R.batch == b].closed_at, format="mixed").max())[:10])
             for b in sorted(R.batch.unique())] +
            [("real ALL (master + batch)", R, R.pct, R.blkd, pd.to_datetime(R.closed_at, format="mixed").dt.strftime("%Y-%m-%d"),
              str(pd.to_datetime(R.opened_at).min())[:10], str(pd.to_datetime(R.closed_at, format="mixed").max())[:10])]):
        pct = np.asarray(pct, float); b_ = np.asarray(blkd, bool); day = np.asarray(day)
        blk = pct[b_]
        win = np.asarray(g.win)[b_]
        pneg, lo, hi = window_boot_mean(blk, win) if b_.sum() >= 2 else (np.nan, np.nan, np.nan)
        pr = np.asarray(g.pair)[b_]
        loss = pd.Series(blk[blk < 0]).groupby(pr[blk < 0]).sum()
        wl = pd.Series(blk[blk < 0]).groupby(win[blk < 0]).sum()
        rows.append(dict(block=name, N=len(pct), before_avg=pct.mean(), before_DCR=dcr_one_size(pct, day, d0, d1),
                         after_avg=pct[~b_].mean() if (~b_).any() else np.nan, after_DCR=dcr_one_size(pct[~b_], day[~b_], d0, d1),
                         blocked_N=int(b_.sum()), windows=len(np.unique(win)), blocked_WR=(blk > 0).mean() * 100 if len(blk) else np.nan,
                         breakeven_WR=breakeven_wr(pct), blocked_avg=blk.mean() if len(blk) else np.nan, P_mean_lt0=pneg, ci_lo=lo, ci_hi=hi,
                         top_pair_share=(loss.min() / loss.sum() * 100) if len(loss) else np.nan,
                         top_window_share=(wl.min() / wl.sum() * 100) if len(wl) else np.nan))
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(D, "wl_fb_funding_above_base.csv"), index=False)
    P(md(B))
    d = np.where(R.blkd, -R.pct, 0.0)
    P("real fills LOO (Δ of blocking, % pts): " + RM.loo_text(list(d), labels=list(R.pair + " " + R.opened_at.str[5:10]), unit="% pts"))
    X["half"] = pd.to_datetime(X.cf_entry_t, unit="ms").dt.year.astype(str) + np.where(pd.to_datetime(X.cf_entry_t, unit="ms").dt.month < 7, "H1", "H2")
    hv = X.assign(b=X.funding_rate_pct > THR).groupby(["half", "b"]).X0_net.agg(["count", "mean"]).unstack("b")
    P("by half-year (backtest): blocked side count / mean vs kept")
    P(md(hv.reset_index().set_axis(["half", "kept N", "blocked N", "kept avg", "blocked avg"], axis=1)))
    # family size of this study's FB cuts (7 variables × 2 outer terciles) → the Bonferroni bar on the blocked-mean confidence
    P("Multiple looks: this study read 7 FB variables × 2 outer terciles = 14 cuts → a fair confidence bar is 1 − 0.05/14 ≈ 99.6 %.")


if __name__ == "__main__":
    fb_funding_above_base()
    with open(os.path.join(D, "wl_tables.md"), "a") as f:
        f.write("\n".join(OUTMD[OUTMD.index(next(s for s in OUTMD if s.startswith('\n## FLATBASE — ④'))):]) + "\n")
