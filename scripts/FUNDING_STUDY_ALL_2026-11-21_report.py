#!/usr/bin/env python3
"""FUNDING_STUDY_ALL_2026-11-21 — tables + chart (research only). Reads the outputs of _cohort / _bt / _real and writes
reports/FUNDING_STUDY_ALL_2026-11-21_tables.md + reports/FUNDING_STUDY_ALL_2026-11-21_chart.png.
One size X = 0.61 × equity for EXH / DIST / FB (the TP_GRID 11-20 convention); "$" = Σ % × 61 = dollars on a $10,000 account at one size.
Statistics: day-block bootstrap (entry day) 4,000 resamples; family-wise null per sleeve per family (exit: day-cluster sign flips of Δ,
max over cells; entry: joint permutation of the entry funding facts across fills, max over cells), 2,000 draws."""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "FUNDING_STUDY_ALL_2026-11-21_data")
X1 = 0.61
SPLIT = pd.Timestamp("2025-07-01")


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


BT = _imp("FSB", "FUNDING_STUDY_ALL_2026-11-21_bt.py")
C = BT.C
EXIT_CELLS = [f"B{f:g}_{g}" for f in BT.F_GRID for g in BT.G_GRID] + [f"T{h}" for h in BT.T_GRID]
NET_SHORT = ["NTP", "NSL", "NTPSL"]          # EXH / DIST: today = price TP + price SL
NET_FB = ["NTP", "PSL", "NTP_PSL"]           # FB: today = price TP + NET SL (DL 607) → NTP = both net · PSL = DL 607 off · NTP_PSL = net TP only
FB_CELLS = [c for c in EXIT_CELLS if c.startswith("B")] + NET_FB
ENTRY_CELLS = [(f"skip paid≥{x:.2f}%", "x", x, 1.0) for x in BT.X_GRID] + [("skip interval≤1h", "h", 1, 1.0)] + \
              [(f"½ paid≥{x:.2f}%", "x", x, 0.5) for x in BT.X_GRID] + [("½ interval≤1h", "h", 1, 0.5)]
L = []


def md(df, fl=2):
    return C.md(df, fl)


def dcr(net, day):
    if not len(net):
        return np.nan, np.nan
    d = pd.to_datetime(pd.Series(list(day)))
    return C.day_dcr(net, d.dt.normalize(), d.min().normalize(), d.max().normalize(), X1)


def stats(net, day):
    net = np.asarray(net, float)
    if not len(net):
        return dict(N=0)
    w, l_ = net[net > 0], net[net <= 0]
    be = abs(l_.mean()) / (w.mean() + abs(l_.mean())) * 100 if len(w) and len(l_) else np.nan
    dc, dd = dcr(net, day)
    return dict(N=len(net), WR=(net > 0).mean() * 100, avg=net.mean(), sum_pts=net.sum(), usd_1size=net.sum() * X1 * 100,
                BE_WR=be, DCR=dc, maxDD=dd)


def blocks_bt(R):
    te = pd.to_datetime(R.te, unit="ms")
    out = [("BT DESIGN <2025-07", te < SPLIT), ("BT HOLDOUT ≥2025-07", te >= SPLIT), ("BT ALL", te.notna())]
    if "univ" in R.columns and R.univ.nunique() > 1:
        out += [(f"BT ALL · {u}", R.univ.eq(u)) for u in sorted(R.univ.unique())]
    return out


def entry_mask(R, kind, x):
    if kind == "x":
        return (R.paid_rate >= x - 1e-12).fillna(False).values
    return (R.interval_h <= x).fillna(False).values


def exit_table(R, cells, blocks, closeday, title, slip30=True):
    rows, det = [], []
    for bn, m in blocks:
        r = R[m.values if hasattr(m, "values") else m]
        if not len(r):
            rows.append(dict(block=bn, rule="(no fills)")); continue
        day = closeday(r, "base")
        b = stats(r.base_net, day)
        rows.append(dict(block=bn, rule="today (with funding)", **b))
        rows.append(dict(block=bn, rule="today, NO funding (blind)", **stats(r.base_net_nofund, day)))
        for c in cells:
            d = (r[c + "_net"] - r.base_net).values
            ch = np.abs(d) > 1e-9
            mu, lo, hi, p = C.cluster_boot(d, pd.to_datetime(r.te, unit="ms").dt.date.values) if len(d) > 1 else (d.mean(), np.nan, np.nan, np.nan)
            s = stats(r[c + "_net"], closeday(r, c))
            bw = r.base_why.values
            row = dict(block=bn, rule=c, **s, d_avg=mu, d_lo=lo, d_hi=hi, P_gain=p, changed=int(ch.sum()),
                       saved=f"{int((ch & (bw == 'STOP') & (d > 0)).sum())} · {d[ch & (bw == 'STOP') & (d > 0)].sum():+.1f}",
                       deeper=f"{int((ch & (bw == 'STOP') & (d < 0)).sum())} · {d[ch & (bw == 'STOP') & (d < 0)].sum():+.1f}",
                       cut_TP=f"{int((ch & (bw == 'TP')).sum())} · {d[ch & (bw == 'TP')].sum():+.1f}",
                       vs_TIME=f"{int((ch & (bw == 'TIME')).sum())} · {d[ch & (bw == 'TIME')].sum():+.1f}",
                       unchanged_max=float(np.abs(d[~ch]).max()) if (~ch).any() else 0.0)
            if slip30:
                forced = (r[c + "_why"] == "FORCED").values
                d30 = d - np.where(forced, 0.30 - r.slip.values.astype(float), 0.0)
                row["d_avg_slip0.30"] = d30.mean()
            rows.append(row)
    L.append(f"\n**{title}**\n")
    df = pd.DataFrame(rows)
    keep = [c for c in ["block", "rule", "N", "WR", "avg", "sum_pts", "usd_1size", "DCR", "maxDD", "d_avg", "d_lo", "d_hi", "P_gain",
                        "d_avg_slip0.30", "changed", "saved", "deeper", "cut_TP", "vs_TIME", "unchanged_max"] if c in df.columns]
    L.append(md(df[keep]))
    return df


def entry_table(R, blocks, closeday, title):
    rows = []
    for bn, m in blocks:
        r = R[m.values if hasattr(m, "values") else m]
        if not len(r):
            rows.append(dict(block=bn, rule="(no fills)")); continue
        day = closeday(r, "base")
        base = stats(r.base_net, day)
        rows.append(dict(block=bn, rule="today", **{k: base[k] for k in ("N", "WR", "avg", "sum_pts", "DCR")}, BE_WR=base["BE_WR"]))
        for name, kind, x, frac in ENTRY_CELLS:
            mk = entry_mask(r, kind, x)
            new = np.where(mk, r.base_net.values * (1 - frac), r.base_net.values)
            s = stats(new[~mk] if frac == 1 else new, np.asarray(day)[~mk] if frac == 1 else day)
            dc = dcr(new, day)[0]
            bl = r.base_net.values[mk]
            bday = pd.to_datetime(r.te.values[mk], unit="ms").date if mk.any() else []
            if mk.sum() > 1:
                mu, lo, hi, pg = C.cluster_boot(bl, bday)
                p_neg = 1 - pg
            else:
                mu = bl.mean() if len(bl) else np.nan; hi = np.nan; p_neg = np.nan
            pairs = pd.Series(bl, index=r.pair.values[mk]).groupby(level=0).sum() if mk.any() else pd.Series(dtype=float)
            dsum = pd.Series(bl, index=list(bday)).groupby(level=0).sum() if mk.any() else pd.Series(dtype=float)
            loss = -bl[bl < 0].sum()
            top_pair = (-pairs[pairs < 0]).max() / loss * 100 if loss > 0 and (pairs < 0).any() else 0
            top_day = (-dsum[dsum < 0]).max() / loss * 100 if loss > 0 and (dsum < 0).any() else 0
            rows.append(dict(block=bn, rule=name, N=s["N"], WR=s.get("WR"), avg=s.get("avg"), sum_pts=new.sum(), DCR=dc,
                             BE_WR=base["BE_WR"], blk_N=int(mk.sum()), blk_WR=(bl > 0).mean() * 100 if len(bl) else np.nan,
                             blk_avg=mu, blk_hi95=hi, P_blk_lt0=p_neg, blk_days=len(set(bday)), top_pair_loss_pct=top_pair,
                             top_day_loss_pct=top_day))
    L.append(f"\n**{title}**\n")
    L.append(md(pd.DataFrame(rows)))
    return pd.DataFrame(rows)


def null_exit(R, cells, B=2000, seed=11):
    d = np.stack([(R[c + "_net"] - R.base_net).values for c in cells], 1)
    day = pd.to_datetime(R.te, unit="ms").dt.date.values
    u, inv = np.unique(day, return_inverse=True)
    S = np.zeros((len(u), len(cells))); np.add.at(S, inv, d)
    obs = d.mean(0)
    rng = np.random.default_rng(seed)
    sg = rng.choice([-1.0, 1.0], (B, len(u)))
    mx = (sg @ S / len(d)).max(1)
    best = int(np.argmax(obs))
    return cells[best], obs[best], (mx >= obs[best]).mean()


def null_entry(R, B=2000, seed=12):
    base = R.base_net.values
    def cellstats(pr, ih):
        out = []
        for name, kind, x, frac in ENTRY_CELLS:
            mk = (pr >= x - 1e-12) if kind == "x" else (ih <= x)
            out.append((-base[mk] * frac).sum() / len(base))
        return np.array(out)
    pr = np.nan_to_num(R.paid_rate.values, nan=-9); ih = np.nan_to_num(R.interval_h.values, nan=99)
    obs = cellstats(pr, ih)
    rng = np.random.default_rng(seed)
    mx = np.empty(B)
    for b in range(B):
        p = rng.permutation(len(base)); mx[b] = cellstats(pr[p], ih[p]).max()
    best = int(np.argmax(obs))
    return ENTRY_CELLS[best][0], obs[best], (mx >= obs[best]).mean()


def loo_best(R, cell):
    d = (R[cell + "_net"] - R.base_net).values
    pr = pd.Series(d, index=R.pair.values).groupby(level=0).sum()
    dy = pd.Series(d, index=pd.to_datetime(R.te, unit="ms").dt.date.values).groupby(level=0).sum()
    n = len(d)
    return dict(mean=d.mean(), loo_pair_min=((d.sum() - pr) / n).min(), loo_pair_max=((d.sum() - pr) / n).max(),
                top_pair=pr.abs().idxmax(), top_pair_share=pr.abs().max() / max(np.abs(pr).sum(), 1e-12) * 100,
                loo_day_min=((d.sum() - dy) / n).min(), top_day_share=dy.abs().max() / max(np.abs(dy).sum(), 1e-12) * 100)


def bt_close(r, c):
    hold = r["base_hold"] if c == "base" else r[c + "_hold"]
    return (pd.to_datetime(r.te, unit="ms") + pd.to_timedelta(hold, unit="h")).dt.date.values


def exposure(R, title, hours=(1, 4, 8, 24, 1e9)):
    R = R.copy()
    R["hb"] = pd.cut(R.base_hold, [-1, *hours], labels=[f"≤{h:g} h" if h < 1e8 else ">24 h" for h in hours])
    rows = []
    for (bn, m) in blocks_bt(R):
        r = R[m.values]
        for hb, g in list(r.groupby("hb", observed=True)) + [("ALL", r)]:
            rows.append(dict(block=bn, hold=hb, N=len(g), paying_pct=(g.base_fund < -1e-9).mean() * 100,
                             fund_pts=g.base_fund.sum(), fund_avg=g.base_fund.mean(), fund_usd_1size=g.base_fund.sum() * X1 * 100,
                             net_nofund_pts=g.base_net_nofund.sum(), net_pts=g.base_net.sum(),
                             fund_share_of_net_pct=g.base_fund.sum() / abs(g.base_net_nofund.sum()) * 100 if g.base_net_nofund.sum() else np.nan,
                             hourly_at_entry_pct=(g.interval_h <= 1).mean() * 100, fund_p90_paid=-g.base_fund.quantile(0.1),
                             fund_worst=g.base_fund.min()))
    L.append(f"\n**{title}**\n")
    L.append(md(pd.DataFrame(rows)))


def path_table(R, title):
    rows = []
    for bn, m in blocks_bt(R):
        r = R[m.values]
        for lab, g in r.groupby("label"):
            w = g[g.base_net > 0]; lo = g[g.base_net <= 0]
            rows.append(dict(block=bn, entry_label=lab, N=len(g), win_N=len(w), lose_N=len(lo), lose_pts=lo.base_net.sum(),
                             lose_fund_pts=lo.base_fund.sum(), win_fund_pts=w.base_fund.sum(), reach_med=g.reach.median()))
    L.append(f"\n**{title}** (WORKED = +1 % our way before −SL; losers by entry label; funding pts paid by each side)\n")
    L.append(md(pd.DataFrame(rows)))


def sleeve_bt(tag, name, cells):
    R = pd.read_pickle(os.path.join(D, f"bt_{tag}.pkl"))
    R = R[~R.miss.astype(bool)].reset_index(drop=True)
    if "slip" not in R.columns:      # FB rebuild: the per-trade exit slip lives in slip_x
        R = R.merge(BT.fb_rows()[["rid", "slip_x"]].rename(columns={"slip_x": "slip"}), on="rid", how="left")
    L.append(f"\n### {name} — backtest ({len(R)} fills with funding data)\n")
    exposure(R, f"{name} backtest — funding paid under today's exit, by hold bucket (pts = Σ % of notional; + = received)")
    path_table(R, f"{name} backtest — path decomposition")
    blocks = blocks_bt(R)
    ex = exit_table(R, cells, blocks, bt_close, f"{name} backtest — EXIT rules vs today (Δ = rule − today per fill, % pts; CI = day-block 95 %)")
    en = entry_table(R, blocks, bt_close, f"{name} backtest — ENTRY rules (paid rate = last settled rate ≤ entry, against the position)")
    c, o, p = null_exit(R, cells)
    ce, oe, pe = null_entry(R)
    L.append(f"\n**{name} family-wise null (ALL):** exit family {len(cells)} cells — best {c} mean Δ {o:+.3f} %/fill, p = {p:.3f} · "
             f"entry family {len(ENTRY_CELLS)} cells — best {ce} mean Δ {oe:+.3f} %/fill, p = {pe:.3f}\n")
    lb = loo_best(R, c)
    L.append(f"LOO for the best exit cell {c}: mean Δ {lb['mean']:+.3f} · leave-one-pair-out {lb['loo_pair_min']:+.3f} … "
             f"{lb['loo_pair_max']:+.3f} (top pair {lb['top_pair']} = {lb['top_pair_share']:.0f} % of Σ|Δ|) · leave-one-day-out min "
             f"{lb['loo_day_min']:+.3f} (top day {lb['top_day_share']:.0f} % of Σ|Δ|)\n")
    hy = pd.to_datetime(R.te, unit="ms")
    R["half"] = hy.dt.year.astype(str) + "-H" + np.where(hy.dt.month <= 6, "1", "2")
    hh = R.groupby("half").apply(lambda g: pd.Series({"N": len(g), "today": g.base_net.mean(), "fund_avg": g.base_fund.mean(),
                                                       **{f"Δ {k}": (g[k + "_net"] - g.base_net).mean() for k in cells[:0] + [c]}}))
    L.append(f"\n**{name} per half-year — today vs the best exit cell {c}**\n")
    L.append(md(hh.reset_index()))
    return R, ex, en


def real_exh():
    W = pd.read_pickle(os.path.join(D, "real_walk.pkl"))
    E = W[W.sleeve == "EXHAUSTION_SHORT"].copy()
    L.append("\n### EXHAUSTION_SHORT — real fills\n")
    L.append("\n**Parity — the re-walk under each fill's OWN era exit (TP 10 / 5, RARE #32 + ENSO clamped to 7 at the DL 609 deploy "
             "2026-11-21 20:59:49 UTC) vs booked**\n")
    E["gap"] = E.par_net - E.booked
    L.append(md(E[["id", "pair", "opened_at", "booked_reason", "booked", "par_why", "par_net", "gap", "booked_fund_usd", "notional",
                   "par_fund", "par_hold"]]))
    E["paid_stamp"] = -E.stamp_rate * 100
    E["paid_last"] = -E.last_rate * 100; E["paid_next"] = -E.next_rate * 100
    L.append("\n**Entry funding at the 9 real fills (% per settlement, paid by the short): live stamp (premiumIndex estimate) vs the last "
             "SETTLED rate (the backtest's lag-safe proxy) vs the next settlement**\n")
    L.append(md(E[["id", "pair", "paid_stamp", "paid_last", "paid_next", "interval_h"]]))
    L.append("\n**Re-priced under TODAY's exit (TP min(20×ATR, 7) / SL min(1.5×ATR, 6) / 24 h) with real funding, + the exit cells**\n")
    cols = ["id", "pair", "booked", "base_why", "base_net", "base_fund", "base_hold", "label", "reach"] + \
           [c + "_net" for c in ("B0.5_0", "B1_0", "B1_2", "T6", "T12", "NTP", "NSL", "NTPSL")]
    L.append(md(E[cols]))
    C8 = E[E.booked_reason != "OPEN"]
    rows = [dict(rule="as booked (era exits, booked funding)", N=len(C8), WR=(C8.booked > 0).mean() * 100, avg=C8.booked.mean(),
                 fund_pts=(C8.booked_fund_usd / C8.notional * 100).sum())]
    rows.append(dict(rule="today's exit (TP 7), with funding", N=len(C8), WR=(C8.base_net > 0).mean() * 100, avg=C8.base_net.mean(),
                     fund_pts=C8.base_fund.sum()))
    rows.append(dict(rule="today's exit, NO funding", N=len(C8), WR=(C8.base_net_nofund > 0).mean() * 100, avg=C8.base_net_nofund.mean(), fund_pts=0))
    for c in EXIT_CELLS + NET_SHORT:
        d = C8[c + "_net"] - C8.base_net
        rows.append(dict(rule=c, N=len(C8), WR=(C8[c + "_net"] > 0).mean() * 100, avg=C8[c + "_net"].mean(), d_avg=d.mean(),
                         changed=int((d.abs() > 1e-9).sum())))
    for name, kind, x, frac in ENTRY_CELLS:
        mk = (C8.paid_last >= x) if kind == "x" else (C8.interval_h <= x)
        mks = (C8.paid_stamp >= x) if kind == "x" else mk
        rows.append(dict(rule=name + " (settled / stamp)", N=f"{int(mk.sum())} / {int(mks.sum())} blocked",
                         d_avg=(-(C8.base_net * mk).sum() * frac) / len(C8), d_avg_stamp=(-(C8.base_net * mks).sum() * frac) / len(C8)))
    L.append("\n**① current batch, 8 closed EXH fills (ENSO open, shown above only) — before → after**\n")
    L.append(md(pd.DataFrame(rows)))
    return E


def real_fb():
    W = pd.read_pickle(os.path.join(D, "real_walk.pkl"))
    F = W[W.sleeve == "FLATBASE_SHORT"].copy()
    F["gap"] = F.base_net - F.ref_Bp
    L.append("\n### FLATBASE_SHORT — real fills (KEPT_LIST_FLATBASE 20 = master 17 + batch 3; today's-stack entries from FUNDING_STOP real_rows)\n")
    L.append(f"Parity vs the DL 607 study's Bp arm (today's exit incl. funding): mean |gap| {F.gap.abs().mean():.4f} pts, max {F.gap.abs().max():.4f}; "
             f"funding |gap| max {(F.base_fund - F.ref_Bp_fund).abs().max():.4f}\n")
    rows = []
    for bn, g in (("① BATCH", F[F.block == "BATCH"]), ("② MASTER", F[F.block == "MASTER"]), ("①+②", F)):
        rows.append(dict(block=bn, rule="today (TP 5 price · SL 10 net-of-paid)", N=len(g), WR=(g.base_net > 0).mean() * 100, avg=g.base_net.mean(),
                         fund_pts=g.base_fund.sum()))
        for c in FB_CELLS:
            d = g[c + "_net"] - g.base_net
            ch = d.abs() > 1e-9
            rows.append(dict(block=bn, rule=c, N=len(g), WR=(g[c + "_net"] > 0).mean() * 100, avg=g[c + "_net"].mean(), d_avg=d.mean(),
                             changed=int(ch.sum()), changed_fills=", ".join(f"{i} {v:+.2f}" for i, v in zip(g.id[ch], d[ch]))[:120]))
    L.append(md(pd.DataFrame(rows)))
    return F


def real_other():
    X = pd.read_csv(os.path.join(D, "real_cohort.csv"), low_memory=False)
    fb = pd.read_csv(os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data", "real_rows.csv"))
    kl = pd.read_csv(os.path.join(ROOT, "reports", "KEPT_LIST_FLATBASE_2026-11-21.csv"))
    fb = fb.merge(kl, left_on=["pair", "k"], right_on=["pair", "k"], how="inner")
    X["net"] = X.price_pct + X.mod_fund_pct
    # FLATBASE: today's stack entry (MODEL_CONFIRM / REAL) + today's exit (DL 607 paid-only stop) from FUNDING_STOP real_rows (Bp arm)
    m = X.sleeve.eq("FLATBASE_SHORT")
    Y = X[~m].copy()
    F = pd.DataFrame(dict(block=np.where(fb.block.eq("BATCH"), "BATCH", "MASTER"), sleeve="FLATBASE_SHORT", pair=fb.pair, opened_at=fb.k,
                          net=fb.Bp_net, mod_fund_pct=fb.Bp_fund, price_pct=fb.Bp_net - fb.Bp_fund,
                          hold_h=(fb.Bp_xt - fb.te) / 3.6e6, interval_h=fb.interval_h, last_rate=-fb.entry_rate.fillna(0) * 0 + fb.rate_at_entry,
                          entry_kind=fb.kind, notional=fb.notional))
    Y = pd.concat([Y, F], ignore_index=True)
    Y["paid_last"] = -Y.last_rate * 100 * np.where(Y.sleeve.isin(["FLATBASE_SHORT"]), 1, Y.get("sgn", 1))
    Y.loc[~Y.sleeve.eq("FLATBASE_SHORT"), "paid_last"] = -Y.sgn * Y.last_rate * 100
    Y.to_csv(os.path.join(D, "real_cohort_final.csv"), index=False)
    rows = []
    for (s, b), g in Y.groupby(["sleeve", "block"]):
        rows.append(dict(sleeve=s, block=b, N=len(g), WR=(g.net > 0).mean() * 100, price_pts=g.price_pct.sum(), fund_pts=g.mod_fund_pct.sum(),
                         net_pts=g.net.sum(), fund_share_pct=g.mod_fund_pct.sum() / abs(g.price_pct.sum()) * 100 if g.price_pct.sum() else np.nan,
                         paying_pct=(g.mod_fund_pct < -1e-9).mean() * 100, worst=g.mod_fund_pct.min(),
                         hold_med=g.hold_h.median(), hold_max=g.hold_h.max(), held_gt6h=int((g.hold_h > 6).sum()),
                         paid_ge0_5=int((g.mod_fund_pct <= -0.5).sum()), hourly_at_entry=int((g.interval_h <= 1).sum())))
    L.append("\n**Real fills, every sleeve — funding priced from the real settlements (master today's stack + current batch)**\n")
    L.append(md(pd.DataFrame(rows)))
    Y["hb"] = pd.cut(Y.hold_h, [-1, 1, 4, 8, 24, 1e9], labels=["≤1 h", "1–4 h", "4–8 h", "8–24 h", ">24 h"])
    hb = Y.groupby(["sleeve", "block", "hb"], observed=True).agg(N=("net", "size"), fund_pts=("mod_fund_pct", "sum"),
                                                                  price_pts=("price_pct", "sum")).reset_index()
    L.append("\n**Real fills — funding by hold bucket**\n")
    L.append(md(hb))
    rows = []
    for (s, b), g in Y.groupby(["sleeve", "block"]):
        for name, kind, x, frac in ENTRY_CELLS:
            mk = (g.paid_last >= x) if kind == "x" else (g.interval_h <= x)
            mk = mk.fillna(False).values
            if not mk.any():
                continue
            bl = g.net.values[mk]
            rows.append(dict(sleeve=s, block=b, rule=name, blk_N=int(mk.sum()), blk_WR=(bl > 0).mean() * 100, blk_avg=bl.mean(),
                             d_avg=-(bl.sum() * frac) / len(g), blk_pairs=", ".join(sorted(set(g.pair.values[mk])))[:80]))
    L.append("\n**Real fills — ENTRY rules (paid rate = last settled ≤ open, against the position); only cells that block anything**\n")
    L.append(md(pd.DataFrame(rows)) if rows else "none block anything\n")
    return Y


def chart(R, fn, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SF = np.stack(R.snap_fund.values); SP = np.stack(R.snap_px.values)
    win = R.base_net.values > 0
    hrs = np.arange(SF.shape[1])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="#fcfcfb")
    cols = {"winners": "#2a78d6", "losers": "#eb6834"}
    for lab, mk in (("winners", win), ("losers", ~win)):
        n_open = np.isfinite(SP[mk]).sum(0)
        ax[0].plot(hrs, np.nanmean(SP[mk], 0), color=cols[lab], lw=2, label=f"{lab} (N {mk.sum()})")
        ax[1].plot(hrs, -np.nanmean(SF[mk], 0), color=cols[lab], lw=2, label=lab)
        ax[2].plot(hrs, n_open / mk.sum() * 100, color=cols[lab], lw=2, label=lab)
    hourly = (R.interval_h <= 1).values
    ax[1].plot(hrs, -np.nanmean(SF[hourly], 0), color="#52514e", lw=1.5, ls="--", label=f"hourly coins at entry (N {hourly.sum()})")
    ax[0].set_title("Price P&L of fills still open (% , + = our way)", fontsize=10)
    ax[1].set_title("Cumulative funding PAID by fills still open (% of notional)", fontsize=10)
    ax[2].set_title("Share of fills still open (%)", fontsize=10)
    for a in ax:
        a.set_xlabel("hours since entry"); a.grid(color="#e6e5e0", lw=0.6); a.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(fn, dpi=130)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    L.append("# FUNDING_STUDY_ALL_2026-11-21 — generated tables\n")
    L.append("## Real fills\n")
    real_exh()
    real_fb()
    real_other()
    L.append("\n## Backtests\n")
    Rx, exx, enx = sleeve_bt("EXH", "EXHAUSTION_SHORT (FRENZY ∪ WIDE, TP 7)", EXIT_CELLS + NET_SHORT)
    chart(Rx, os.path.join(ROOT, "reports", "FUNDING_STUDY_ALL_2026-11-21_chart.png"),
          f"EXHAUSTION_SHORT backtest ({len(Rx)} fills, today's exit TP 7 / ATR stop / 24 h): price vs funding over the hold")
    Rd, exd, end_ = sleeve_bt("DIST", "DISTRIBUTION_SHORT (TP 10)", EXIT_CELLS + NET_SHORT)
    Rf, exf, enf = sleeve_bt("FB", "FLATBASE_SHORT (stop counts funding, DL 607)", FB_CELLS)
    open(os.path.join(ROOT, "reports", "FUNDING_STUDY_ALL_2026-11-21_tables.md"), "w").write("\n".join(L))
    for n, df in (("EXH_exit", exx), ("EXH_entry", enx), ("DIST_exit", exd), ("DIST_entry", end_), ("FB_exit", exf), ("FB_entry", enf)):
        df.to_csv(os.path.join(D, f"{n}.csv"), index=False)
    print("ok")
