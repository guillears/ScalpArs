#!/usr/bin/env python3
"""HT_WILLY 2026-11-20 — PART 2 timing rules + PART 3 joint (frozen in the PREREG ADDENDUM before this ran). Research only.
HT_R1 DELAY_DIP4 · HT_R2 FILTER_DIP4 · HT_R3 DELAY_DIP8 on the HT cohort (dip ≥ 1.4, one per coin per holding period) and on the current-batch
real HT fills (post-587). Exit = REF (TP 5 / SL 4 / 24 h) and the Part-1 pick (T8_S3_H12) for the joint read.
Output reports/HT_WILLY_2026-11-20_rules.csv (+ _rules_rows.csv) and printed tables.
Usage: venv/bin/python -I scripts/ht_willy_rules.py"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ht_willy_sim as W  # noqa: E402
import ht_willy_eval as EV  # noqa: E402
import fast_tp_data as FD  # noqa: E402
MIN, H1 = 60_000, 3_600_000
RULES = {"HT_R1_DELAY_DIP4": ("delay", 4.0), "HT_R2_FILTER_DIP4": ("filter", 4.0), "HT_R3_DELAY_DIP8": ("delay", 8.0)}
PICK = "T8_S3_H12"
OUT = EV.OUT


def rule_entry(t, c, i0, T, live, H, kind, dip):
    lvl = H * (1 - dip / 100.0)
    if live <= lvl:
        return i0, T, live
    if kind == "filter":
        return None
    j1 = int(np.searchsorted(t, T + 6 * H1))
    for j in range(i0 + 1, min(j1, len(t))):
        if t[j - 1] != t[j] - MIN:
            continue
        if c[j - 1] <= lvl:
            return j, int(t[j]), float(c[j - 1])
    return None


def run(rows, getter):
    out = []
    for _, r in rows.iterrows():
        T = int(r.ts); a = getter(r.pair, T - 2 * MIN, T + 31 * H1)
        if a is None:
            continue
        t, o, h, l, c = a
        i0 = int(np.searchsorted(t, T))
        if i0 >= len(t) or t[i0] != T:
            continue
        W.ref_exit.pair = r.pair
        tick = W.S.tick_for(r.pair, r.live)
        rec = dict(row=r.row, pair=r.pair, ts=T, blk=r.blk, half=r.half, day=r.day, ref_net=r.ref_net, ref_x=r.ref_x)
        y = W.ref_exit("HT", t, o, h, l, c, i0, T, r.live, r.H, None, tick, cell=PICK)
        rec["pick_net"] = y[0] if y else np.nan; rec["pick_x"] = int(t[y[1]]) + MIN if y else 0
        for nm, (kind, dip) in RULES.items():
            e = rule_entry(t, c, i0, T, float(r.live), float(r.H), kind, dip)
            if e is None:
                rec[nm + "_ts"] = 0; rec[nm + "_net"] = np.nan; rec[nm + "_pnet"] = np.nan; rec[nm + "_x"] = 0; continue
            j, tj, Ej = e
            y = W.ref_exit("HT", t, o, h, l, c, j, tj, Ej, r.H, None, tick)
            z = W.ref_exit("HT", t, o, h, l, c, j, tj, Ej, r.H, None, tick, cell=PICK)
            rec[nm + "_ts"] = tj; rec[nm + "_net"] = y[0] if y else np.nan; rec[nm + "_x"] = int(t[y[1]]) + MIN if y else 0
            rec[nm + "_pnet"] = z[0] if z else np.nan; rec[nm + "_wait"] = (tj - T) / MIN
        out.append(rec)
    R = pd.DataFrame(out)
    for nm in RULES:   # one trade per coin per holding period under the rule's own exit
        m = R[nm + "_ts"] > 0
        k = np.zeros(len(R), bool)
        k[np.flatnonzero(m)] = EV.deoverlap_x(R.loc[m, nm + "_ts"].values, R.loc[m, "pair"].values, R.loc[m, nm + "_x"].values)
        R[nm + "_take"] = k & np.isfinite(R[nm + "_net"].values)
    return R


def summarize(R, label, null=True):
    rows = []
    ref = R.ref_net.values
    tvals = {}
    for nm in RULES:
        tk = R[nm + "_take"].values
        rn = np.where(tk, R[nm + "_net"].values, 0.0)
        pn = np.where(tk, R[nm + "_pnet"].values, 0.0)
        reads = {"rule vs real (REF exit)": rn - ref, "rule vs real (pick exit)": pn - R.pick_net.values,
                 "pick vs REF on rule entries": np.where(tk, pn - rn, 0.0)}
        for rd, d in reads.items():
            blocks = [("ALL", np.ones(len(R), bool))] + ([("DESIGN", (R.blk == "DESIGN").values), ("HOLDOUT", (R.blk == "HOLDOUT").values)]
                                                          + [(h, (R.half == h).values) for h in sorted(R.half.unique())] if label == "BACKTEST" else [])
            for b, m in blocks:
                if m.sum() < 2:
                    continue
                ci = EV.boot_ci(d[m], R.day.values[m])
                ps = pd.Series(d[m]).groupby(R.pair.values[m]).sum().sort_values(ascending=False)
                r = dict(set=label, rule=nm, read=rd, block=b, signals=int(m.sum()), taken=int(tk[m].sum()),
                         real_avg=round(ref[m].mean(), 3), rule_avg_per_fill=round(R[nm + "_net"].values[m & tk].mean(), 3) if (m & tk).any() else np.nan,
                         rule_WR=round(100 * (R[nm + "_net"].values[m & tk] > 0).mean(), 1) if (m & tk).any() else np.nan,
                         d_per_signal=round(d[m].mean(), 3), lo=round(ci[0], 3), hi=round(ci[1], 3),
                         usd_before=round(ref[m].sum() * 61), usd_after=round(rn[m].sum() * 61) if rd.startswith("rule vs real (REF") else np.nan,
                         top_pair=ps.index[0] if len(ps) else "", top_share=round(ps.iloc[0] / d[m].sum(), 2) if len(ps) and d[m].sum() > 0 else np.nan)
                if rd.startswith("rule vs real (REF") and b in ("DESIGN", "HOLDOUT", "ALL"):
                    t0 = R.ts[m].min(); xa = np.where(tk, R[nm + "_x"].values, 0)
                    t1 = max(R.ref_x[m].max(), xa[m].max())
                    r["DCR"] = f"{EV.dcr(ref[m], R.ref_x[m], t0, t1):+.3f} → {EV.dcr(R[nm + '_net'].values[m & tk], xa[m & tk], t0, t1):+.3f}"
                    r["wait_med_min"] = round(np.nanmedian(R[nm + "_wait"].values[m & tk]), 0) if (nm + "_wait") in R and (m & tk).any() else np.nan
                rows.append(r)
            if rd.startswith("rule vs real (REF") and label == "BACKTEST":
                m = (R.blk == "HOLDOUT").values
                tvals[nm] = d[m]
    T = pd.DataFrame(rows)
    if null and tvals:
        m = (R.blk == "HOLDOUT").values; u, inv = np.unique(R.day.values[m], return_inverse=True)
        DS = np.vstack([np.bincount(inv, tvals[nm], len(u)) for nm in RULES]).T
        def ts(M):
            return M.mean(0) / (M.std(0, ddof=1) / np.sqrt(len(M)) + 1e-12)
        obs = ts(DS); rng = np.random.default_rng(5)
        mx = [np.max(ts(DS * rng.choice([-1.0, 1.0], len(u))[:, None])) for _ in range(2000)]
        n95 = float(np.percentile(mx, 95))
        print("HOLDOUT family null 95th (max t over 3 rules):", round(n95, 2), "| observed t:", dict(zip(RULES, np.round(obs, 2))))
        T.attrs["null95"] = n95; T.attrs["t"] = dict(zip(RULES, obs))
    return T


def main():
    import exh_exit2_sim  # noqa: F401
    C = pd.read_pickle(os.path.join(W.D, "cohorts.pkl"))
    Z = C["HT"].copy()
    R = run(Z, FD.get1m)
    print("backtest signals", len(R))
    T = summarize(R, "BACKTEST")
    import ht_willy_real as HR
    W.MARK_OPEN = True
    REAL = pd.read_pickle(os.path.join(W.D, "real.pkl"))
    Rh = REAL[(REAL.sleeve == "HT") & (pd.to_datetime(REAL.opened_at) >= pd.Timestamp("2026-11-18T22:56"))].copy()
    Rh["blk"] = "REAL"; Rh["half"] = "REAL"; Rh["day"] = EV.day(Rh.ts)
    RR = run(Rh, HR.get1m)
    T2 = summarize(RR, "CURRENT (post-587)", null=False)
    A = pd.concat([T, T2], ignore_index=True)
    A.to_csv(OUT + "_rules.csv", index=False)
    pd.concat([R.assign(set="BACKTEST"), RR.assign(set="CURRENT")]).to_csv(OUT + "_rules_rows.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(A.to_string(index=False))
        print(RR[["pair", "ts", "ref_net", "pick_net"] + [c for c in RR.columns if c.endswith(("_net", "_take", "_wait")) and c.startswith("HT_R")]].to_string(index=False))


if __name__ == "__main__":
    main()
