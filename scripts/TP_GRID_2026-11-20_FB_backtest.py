#!/usr/bin/env python3
"""TP_GRID_2026-11-20_FB — STEP 2: the FLATBASE_SHORT TP × SL grid on the OOS_REAL_FLATBASE rebuild (backtest).

Cohort: reports/OOS_REAL_FLATBASE_data/wl_features.csv, today's NEAR_BASE keep (dist_base > 2.09, X0_why in TP / STOP / TIME) = 614,
then today's DL 601 funding filter (FLATBASE_FUNDING_ABOVE_BASE: funding_rate_pct > 0.005 blocked) = the "today's stack" cohort.
Paths: the rebuild's own 5m paths (study_frenzy_swing_short_walk.load with the oos_real_flatbase_recon sources), entry = cf_entry_px at the
cf_entry_t bar, per-trade exit slip draw slip_x (the rebuild's real-slip distribution), taker 0.045 / maker 0.018 % (rebuild convention),
real funding over each variant's own hold (oos_real_flatbase_common.fund).
  parity mode  : TP = touch (low ≤ line), fill at min(line, open) — the rebuild's walk; the (5, 10, 72) cell must equal X0_net exactly.
  primary mode : TP needs a trade THROUGH the line by ≥ 0.05 % (live resting-maker reality), fills at the line.
Same-bar TP + stop → stop first (conservative); the count is reported.
Output: reports/TP_GRID_2026-11-20_FB_data/bt_*.csv + bt_tables.md
Usage: venv/bin/python -I scripts/TP_GRID_2026-11-20_FB_backtest.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import oos_real_flatbase_common as C  # noqa: E402
import oos_real_flatbase_recon as RC  # noqa: E402  (sets the 5m sources + T0)

_spec = importlib.util.spec_from_file_location("G", os.path.join(ROOT, "scripts", "TP_GRID_2026-11-20_FB_common.py"))
G = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(G)
W = RC.W
OUT = os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_FB_data")
BAR = 300_000
THROUGH = 0.05
FUND_THR = 0.005
OUTMD = []


def P(s=""):
    print(s)
    OUTMD.append(s)


def walk(o, h, l, c, i, n, E, tp, sl, hold_h, sx, mode):
    """→ (gross %, exit fee %, reason, exit idx, same-bar flag)."""
    last = min(n - 1, i + hold_h * 12 - 1)
    S = E * (1 + sl / 100); T = E * (1 - tp / 100)
    Tt = T if mode == "parity" else T * (1 - THROUGH / 100)
    hh, ll = h[i:last + 1], l[i:last + 1]
    ks, kt = G.first_hits(hh, ll, S, Tt)
    m = len(hh)
    if ks < m and ks <= kt:
        j = i + ks; X = max(S, o[j]) * (1 + sx / 100)
        return (1 - X / E) * 100, RC.TAKER * X / E, "STOP", j, ks == kt
    if kt < m:
        j = i + kt; X = min(T, o[j]) if mode == "parity" else T
        return (1 - X / E) * 100, RC.MAKER * X / E, "TP", j, False
    X = c[last] * (1 + sx / 100)
    return (1 - X / E) * 100, RC.TAKER * X / E, ("TIME" if last == i + hold_h * 12 - 1 else "OPEN"), last, False


def build():
    X = pd.read_csv(os.path.join(ROOT, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    X = X[X.X0_why.isin(["TP", "STOP", "TIME"]) & (X.dist_base > 2.09)].copy().reset_index(drop=True)
    F = C.funding_table()
    holds = (24, 48, 72)
    recs = []
    for pair, g in X.groupby("pair"):
        x = W.load(pair)
        if x is None:
            print("NO PATH", pair, len(g)); continue
        t, o, h, l, c = x[:5]
        n = len(t)
        tidx = {int(tt): k for k, tt in enumerate(t)}
        for r in g.itertuples():
            i = tidx.get(int(r.cf_entry_t))
            if i is None:
                print("NO BAR", pair, r.cf_entry_t); continue
            E = float(r.cf_entry_px); sx = float(r.slip_x)
            d = dict(idx=r.Index)
            for mode in ("parity", "primary"):
                for hd in holds:
                    for tp, sl in G.CELLS:
                        if mode == "parity" and not ((tp, sl, hd) == G.TODAY):
                            continue
                        if hd != 72 and mode == "parity":
                            continue
                        gr, fee, why, j, sb = walk(o, h, l, c, i, n, E, tp, sl, hd, sx, mode)
                        xt = int(t[j]) + BAR
                        fu = C.fund(F, pair, int(r.cf_entry_t) + 10_000, xt)[0]
                        k = f"{mode}_{tp}_{sl}_{hd}"
                        d[k] = gr - RC.TAKER - fee + fu; d[k + "_why"] = why; d[k + "_x"] = xt; d[k + "_sb"] = sb
            recs.append(d)
    Rr = pd.DataFrame(recs).set_index("idx")
    X = X.join(Rr, how="inner")
    X.to_pickle(os.path.join(OUT, "bt_grid.pkl"))
    return X


def main():
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "bt_grid.pkl")
    X = pd.read_pickle(f) if os.path.exists(f) and "--rebuild" not in sys.argv else build()
    ent = pd.to_datetime(X.cf_entry_t, unit="ms")
    X["day"] = ent.dt.strftime("%Y-%m-%d")
    X["win"] = ent.dt.floor("4h").astype(str)
    X["blk"] = np.where(X.day < "2025-07-01", "DESIGN", "HOLDOUT")
    X["half"] = ent.dt.year.astype(str).str[2:] + np.where(ent.dt.month <= 6, "-H1", "-H2")
    X["fund_blk"] = X.funding_rate_pct > FUND_THR
    pk = "parity_5_10_72"
    dif = (X[pk] - X.X0_net).abs()
    P("# TP_GRID FB — backtest (OOS_REAL_FLATBASE rebuild)")
    P(f"\nparity: (5, 10, 72) touch-mode vs X0_net — {len(X)} trades · max |Δ| {dif.max():.4f} · mean |Δ| {dif.mean():.4f} · "
      f"exact (<0.001) {int((dif < 1e-3).sum())}")
    P(f"primary-mode today vs parity today: mean {(X['primary_5_10_72'] - X[pk]).mean():+.3f} pts/trade "
      f"(TP needs ≥ {THROUGH} % through, fills at the line) · TP→other {int(((X[pk + '_why'] == 'TP') & (X['primary_5_10_72_why'] != 'TP')).sum())}")
    P(f"mean exit-slip draw slip_x {X.slip_x.mean():.3f} % (median {X.slip_x.median():.3f}) · funding filter blocks {int(X.fund_blk.sum())} of {len(X)}")
    for stack_name, S in (("TODAY'S STACK (NEAR_BASE + DL 601 funding filter)", X[~X.fund_blk]), ("NEAR_BASE only (pre-DL 601)", X)):
        report(S.copy(), stack_name, primary=stack_name.startswith("TODAY"))
    with open(os.path.join(OUT, "bt_tables.md"), "w") as fh:
        fh.write("\n".join(OUTMD) + "\n")


def col(tp, sl, hd=72):
    return f"primary_{tp}_{sl}_{hd}"


def report(S, name, primary):
    P(f"\n## {name}: {len(S)} trades · {S.day.min()} → {S.day.max()}")
    blocks = [("DESIGN", S.blk == "DESIGN"), ("HOLDOUT", S.blk == "HOLDOUT"), ("ALL", S.blk.notna())]
    halves = sorted(S.half.unique())
    blocks_h = blocks + [(h, S.half == h) for h in halves]
    sb = {c_: int(S[col(*c_) + "_sb"].sum()) for c_ in G.CELLS}
    op = {c_: int((S[col(*c_) + "_why"] == "OPEN").sum()) for c_ in G.CELLS}
    P(f"same-bar TP + stop (stop first) per cell: total {sum(sb.values())} over 49 cells (max {max(sb.values())} in one cell; today's "
      f"{sb[(5, 10)]}) · OPEN at data end (marked to the last close): max {max(op.values())} per cell")
    rows = []
    for bn, m in blocks_h:
        g = S[m]
        if not len(g):
            continue
        d0, d1 = g.day.min(), max(g.day.max(), pd.to_datetime(g[col(5, 10)+"_x"], unit="ms").max().strftime("%Y-%m-%d"))
        base = g[col(5, 10)].values
        for tp, sl in G.CELLS:
            v = g[col(tp, sl)].values
            cd = pd.to_datetime(g[col(tp, sl) + "_x"], unit="ms").dt.strftime("%Y-%m-%d")
            rr = G.cell_row(v, cd, d0, d1)
            dm, lo, hi, pg = G.cluster_boot(v - base, g.day.values)
            rows.append(dict(block=bn, tp=tp, sl=sl, **rr, d_avg=dm, d_lo=lo, d_hi=hi, P_gain=pg))
    T = pd.DataFrame(rows)
    tag = "today" if primary else "nb"
    T.to_csv(os.path.join(OUT, f"bt_cells_{tag}.csv"), index=False)
    for bn in ("DESIGN", "HOLDOUT", "ALL"):
        g = T[T.block == bn]
        P(f"\n### {bn} — avg net % per trade (TP rows × stop columns; today bold)")
        P(G.heat({(r.tp, r.sl): r.avg for r in g.itertuples()}))
        P(f"\n### {bn} — WR %")
        P(G.heat({(r.tp, r.sl): r.WR for r in g.itertuples()}, fl=0))
        P(f"\n### {bn} — one-size DCR %/day (X = 0.61, every calendar day)")
        P(G.heat({(r.tp, r.sl): r.DCR for r in g.itertuples()}, fl=3))
    # family-wise null
    P("\n### Family-wise (pick-the-best) null over the 49 cells — paired Δ vs today, random sign per entry DAY, 4,000 draws")
    fw = {}
    for bn, m in blocks:
        g = S[m]
        D = np.column_stack([g[col(*c_)].values - g[col(5, 10)].values for c_ in G.CELLS])
        mx, p = G.fw_null(D, g.day.values)
        obs = D.mean(0)
        b = int(np.argmax(obs))
        fw[bn] = (obs, p, np.percentile(mx, 95))
        P(f"- {bn}: best cell −{G.CELLS[b][0]} / +{G.CELLS[b][1]} Δ {obs[b]:+.3f} %/trade · null 95th pct of the best-of-49 {np.percentile(mx, 95):+.3f} · "
          f"family-wise p {p[b]:.3f} · cells beating the null at 5 %: {[f'-{c_[0]}/+{c_[1]}' for c_, pp in zip(G.CELLS, p) if pp < 0.05]}")
    obsD = fw["DESIGN"][0]
    bD = G.CELLS[int(np.argmax(obsD))]
    hv = T[(T.block == "HOLDOUT") & (T.tp == bD[0]) & (T.sl == bD[1])].iloc[0]
    P(f"- DESIGN-picked cell −{bD[0]} / +{bD[1]} in HOLDOUT: Δ {hv.d_avg:+.3f} (95 % CI {hv.d_lo:+.3f} … {hv.d_hi:+.3f}, P(gain) {hv.P_gain:.2f})")
    # top cells (ALL avg) + today
    allT = T[T.block == "ALL"].sort_values("avg", ascending=False)
    top = [tuple(x) for x in allT[["tp", "sl"]].values if tuple(x) != (5, 10)][:5]
    P("\n### Today + the top 5 cells (by ALL avg), per block")
    sel = T[[(a, b) in ([(5, 10)] + top) for a, b in zip(T.tp, T.sl)]].copy()
    sel["cell"] = "−" + sel.tp.astype(str) + " / +" + sel.sl.astype(str)
    sel = sel[sel.block.isin(["DESIGN", "HOLDOUT", "ALL"] + halves)]
    P(G.md(sel[["block", "cell", "N", "WR", "avg", "sum", "DCR", "maxDD", "worst_day", "BE_WR", "d_avg", "d_lo", "d_hi", "P_gain"]].round(3), 2))
    # robustness for the top cells
    P("\n### Robustness of the top cells (ALL): leave-best-5 %-out Δ · top pair / window share of Σ Δ · LOO range")
    rr = []
    for tp, sl in top:
        d = (S[col(tp, sl)] - S[col(5, 10)]).values
        lo, hi, _ = G.loo_range(d)
        rr.append(dict(cell=f"−{tp} / +{sl}", d_avg=d.mean(), d_drop_best5=G.drop_best_share(d), top_pair_pct=G.top_share(d, S.pair.values),
                       top_window_pct=G.top_share(d, S.win.values), loo_min=lo, loo_max=hi, p_fw_ALL=fw["ALL"][1][G.CELLS.index((tp, sl))]))
    P(G.md(pd.DataFrame(rr).round(3), 3))
    # two-sided stop split (SL-only changes at TP −5)
    P("\n### Two-sided stop split — stop-only changes at TP −5 (vs today +10): Σ Δ pts by group; unchanged fills must show Δ = 0")
    base_why = S[col(5, 10) + "_why"]
    sp = []
    for sl in G.SLS:
        if sl == 10:
            continue
        for bn, m in blocks:
            g = S[m]; bw = base_why[m]
            d = g[col(5, sl)] - g[col(5, 10)]
            w2 = g[col(5, sl) + "_why"]
            if sl > 10:
                grp = {"saved (today STOP → not)": (bw == "STOP") & (w2 != "STOP"), "deeper (STOP both)": (bw == "STOP") & (w2 == "STOP"),
                       "unchanged (not stopped today)": bw != "STOP"}
            else:
                grp = {"new stops (today TP/TIME → STOP)": (bw != "STOP") & (w2 == "STOP"), "cheaper (STOP both)": (bw == "STOP") & (w2 == "STOP"),
                       "unchanged (no stop either)": (bw != "STOP") & (w2 != "STOP")}
            r = dict(stop=f"+{sl}", block=bn)
            for k_, mm in grp.items():
                r[k_.split(" (")[0] + " N"] = int(mm.sum()); r[k_.split(" (")[0] + " ΣΔ"] = d[mm].sum()
            r["total ΣΔ"] = d.sum()
            lo_, hi_ = G.cluster_boot(d.values, g.day.values)[1:3]
            r["Δ/trade CI"] = f"{d.mean():+.2f} ({lo_:+.2f} … {hi_:+.2f})"
            sp.append(r)
    P(G.md(pd.DataFrame(sp).round(1), 1))

    # one-dimensional families (move ONE lever, keep the other at today's value) — smaller families, own pick-the-best null
    P("\n### One lever at a time — TP only (stop +10) and stop only (TP −5): paired Δ vs today per block + half-year, family null within the 7")
    for fam, cells in (("TP only @ +10", [(tp, 10) for tp in G.TPS]), ("stop only @ −5", [(5, sl) for sl in G.SLS])):
        rows_ = []
        for bn, m in blocks_h:
            g = S[m]
            if len(g) < 5:
                continue
            Dm = np.column_stack([g[col(*c_)].values - g[col(5, 10)].values for c_ in cells])
            _, pf = G.fw_null(Dm, g.day.values, B=2000) if bn in ("DESIGN", "HOLDOUT", "ALL") else (None, [np.nan] * len(cells))
            for j, c_ in enumerate(cells):
                if c_ == (5, 10):
                    continue
                d = Dm[:, j]
                dm, lo, hi, pg = G.cluster_boot(d, g.day.values)
                rows_.append(dict(block=bn, cell=f"−{c_[0]} / +{c_[1]}", N=len(d), d_avg=dm, ci=f"{lo:+.2f} … {hi:+.2f}", P_gain=pg,
                                  drop_best5=G.drop_best_share(d), p_family=pf[j]))
        P(f"\n**{fam}**")
        P(G.md(pd.DataFrame(rows_).round(3), 2))
    # hold variants for the top 3
    P("\n### Hold 24 h / 48 h / 72 h for today + the top 3 cells (ALL / DESIGN / HOLDOUT avg net %, WR)")
    hr_ = []
    for tp, sl in [(5, 10)] + top[:3]:
        for hd in (24, 48, 72):
            r = dict(cell=f"−{tp} / +{sl}", hold=hd)
            for bn, m in blocks:
                v = S[m][col(tp, sl, hd)]
                r[f"{bn} avg"] = v.mean(); r[f"{bn} WR"] = (v > 0).mean() * 100
            hr_.append(r)
    P(G.md(pd.DataFrame(hr_).round(2), 2))


if __name__ == "__main__":
    main()
