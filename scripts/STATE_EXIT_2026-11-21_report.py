#!/usr/bin/env python3
"""STATE_EXIT_2026-11-21 — render the tables (research only) → reports/STATE_EXIT_2026-11-21_tables.md + the per-sleeve event-study PNGs
(reports/STATE_EXIT_2026-11-21_data/chart_<SLEEVE>.png) + the case timelines. The narrative report (STATE_EXIT_2026-11-21.md) quotes these."""
import os, sys, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_sp = importlib.util.spec_from_file_location("SEC", os.path.join(HERE, "STATE_EXIT_2026-11-21_common.py"))
C = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C)
_sp = importlib.util.spec_from_file_location("SEV", os.path.join(HERE, "STATE_EXIT_2026-11-21_eval.py"))
EV = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(EV)
NOT = EV.NOT
CHART_VAR = {"FB": ("vd_now", "1h volume ÷ episode-peak hourly volume"), "HT": ("rsi_5m", "5m RSI14"), "EXH": ("rsi_15m", "15m RSI14")}


def label(o):
    return np.where(o.base_why == "TP", "REACHED_TP", np.where((o.peak >= 2) & (o.base_net < 0), "GAVE_BACK",
                    np.where(o.base_net >= 0, "TIME_WIN", "NEVER_+2_LOSS")))


def cohort_tables(O):
    rows, par, gb = [], [], []
    for per in EV.PERIODS:
        o = O[O.period == per]
        if not len(o):
            continue
        rows.append(dict(period=per, N=len(o), days=o.day.nunique(), WR=(o.base_net > 0).mean() * 100, avg=o.base_net.mean(),
                         TP=int((o.base_why == "TP").sum()), STOP=int((o.base_why == "STOP").sum()), TIME=int((o.base_why == "TIME").sum()),
                         booked_avg=o.booked.mean() if "booked" in o and o.booked.notna().any() else np.nan,
                         dcr=EV.dcr(o.base_net, o.base_xt, pd.to_datetime(o.te.min(), unit="ms"), pd.to_datetime(o.base_xt.max(), unit="ms"))))
        lab = label(o)
        for L in ("REACHED_TP", "GAVE_BACK", "NEVER_+2_LOSS", "TIME_WIN"):
            m = lab == L
            gb.append(dict(period=per, label=L, n=int(m.sum()), pts=o.base_net[m].sum(), usd=o.base_net[m].sum() * NOT / 100,
                           peak_med=o.peak[m].median() if m.any() else np.nan))
        m = (o.peak >= 4) & (o.base_net < 0)
        gb.append(dict(period=per, label="  of which peak ≥ +4", n=int(m.sum()), pts=o.base_net[m].sum(), usd=o.base_net[m].sum() * NOT / 100,
                       peak_med=o.peak[m].median() if m.any() else np.nan))
        if o.kind.iloc[0] == "REAL":
            q = o[o.par_ok == True]
            if len(q):
                par.append(dict(period=per, N=len(q), mean_abs_gap=(q.par_net - q.booked).abs().mean(), median_abs_gap=(q.par_net - q.booked).abs().median(),
                                exit_kind_match=f"{int(((q.par_why == 'TP') == q.booked_reason.isin(['FRENZY_TP', 'TP'])).sum())} / {len(q)}"))
        else:
            if o.rec_net.notna().any():
                par.append(dict(period=per, N=len(o), mean_abs_gap=(o.base_net - o.rec_net).abs().mean(), median_abs_gap=(o.base_net - o.rec_net).abs().median(),
                                exit_kind_match=f"{int((o.base_why == o.rec_why.replace({'SL': 'STOP'})).sum())} / {len(o)}"))
            else:
                par.append(dict(period=per, N=len(o), mean_abs_gap=np.nan, median_abs_gap=np.nan,
                                exit_kind_match=f"{int((o.base_why == o.rec_why.replace({'SL': 'STOP'})).sum())} / {len(o)}"))
    return pd.DataFrame(rows), pd.DataFrame(par), pd.DataFrame(gb)


def timing(O, dec, TH, per):
    """per thesis rule (ungated beyond the stored rows): on GAVE_BACK fills, first firing vs the final in-trade peak."""
    o = O[O.period.isin(per)].copy(); o["lab"] = label(o)
    gbf = o[o.lab == "GAVE_BACK"].set_index("fid"); tpf = o[o.lab == "REACHED_TP"].set_index("fid")
    out = []
    for nm, (_, fn) in TH.items():
        try:
            m = fn(dec).fillna(False).values
        except AttributeError:
            continue
        _, kk = EV.first_fire(dec, m & (dec.peak >= 2.0).values)
        g = kk[kk.index.isin(gbf.index)]
        rel = g.values - gbf.loc[g.index, "peak_min"].values
        t_ = kk[kk.index.isin(tpf.index)]
        t_ = t_[t_.values < tpf.loc[t_.index, "hold_min"].values]
        pnl_at = dec.set_index(["fid", "k"]).pnl.reindex(list(zip(g.index, g.values))).values if len(g) else np.array([])
        out.append(dict(rule=nm, gave_back_n=len(gbf), fired_on_gave_back=len(g), before_final_peak=int((rel < 0).sum()),
                        median_min_after_peak=float(np.median(rel)) if len(rel) else np.nan,
                        median_pnl_at_fire=float(np.nanmedian(pnl_at)) if len(pnl_at) else np.nan,
                        tp_n=len(tpf), fired_on_tp_before_tp=len(t_)))
    return pd.DataFrame(out)


def chart(s, O):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P = pd.read_pickle(os.path.join(C.D, "paths.pkl"))
    o = O[O.kind.isin(["REAL", "BT"])].copy(); o["lab"] = label(o)
    lab = o.set_index("fid").lab
    v, vname = CHART_VAR[s]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    span = 72 * 60 if s == "FB" else 6 * 60
    for L, col in (("GAVE_BACK", "#c0392b"), ("REACHED_TP", "#2471a3")):
        ps = [p for p in P if p["fid"] in lab.index and lab[p["fid"]] == L]
        if not ps:
            continue
        rel = ps[0]["rel"]; m = (rel >= -360) & (rel <= span)
        A = np.vstack([p["pnl"] for p in ps])[:, m]
        ax[0].plot(rel[m] / 60, np.nanmean(A, 0), color=col, label=f"{L} (n={len(ps)})")
        ax[0].fill_between(rel[m] / 60, np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0), color=col, alpha=0.12)
        if all(v in p for p in ps):
            B = np.vstack([p[v] for p in ps])[:, m]
            ax[1].plot(rel[m] / 60, np.nanmean(B, 0), color=col, label=L)
    ax[0].axvline(0, color="grey", lw=0.8); ax[1].axvline(0, color="grey", lw=0.8); ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_ylabel("gross P&L % from entry (our direction)"); ax[1].set_ylabel(vname)
    ax[1].set_xlabel("hours from the in-trade peak (trades that reached ≥ +2 %)")
    ax[0].set_title(f"{s}: average path aligned at the in-trade peak — real + backtest fills (mean, IQR band)")
    ax[0].legend(); ax[1].legend()
    if s == "HT":
        ax[1].axhline(75, color="grey", ls=":", lw=0.8)
    fig.tight_layout(); f = os.path.join(C.D, f"chart_{s}.png"); fig.savefig(f, dpi=110); plt.close(fig)
    return f


def cases(s, O, dec, TH):
    o = O[O.kind == "CASE"]
    out = []
    for r in o.itertuples():
        d = dec[dec.fid == r.fid]
        row = dict(pair=r.pair, entry_utc=str(pd.Timestamp(r.te, unit="ms"))[:16], E=r.E, today_exit=f"{r.base_why} {r.base_net:+.2f}",
                   peak=f"{r.peak:+.2f} at +{r.peak_min} min", exit_min=r.hold_min)
        for nm, (_, fn) in TH.items():
            try:
                m = fn(d).fillna(False).values & (d.peak >= 2.0).values
            except AttributeError:
                continue
            if m.any():
                i = np.flatnonzero(m)[0]
                row[nm] = f"+{int(d.k.values[i])}m @ {d.pnl.values[i]:+.2f}"
            else:
                row[nm] = "—"
        out.append(row)
    return pd.DataFrame(out)


def main():
    L = ["# STATE_EXIT 2026-11-21 — auto tables (rendered by scripts/STATE_EXIT_2026-11-21_report.py)\n"]
    for s in ("FB", "HT", "EXH"):
        f = os.path.join(C.D, f"eval_{s}.pkl")
        if not os.path.exists(f):
            continue
        R = pd.read_pickle(f); O = R["O"]
        dec = pd.read_pickle(os.path.join(C.D, f"dec_{s}.pkl")).sort_values(["fid", "k"]).reset_index(drop=True)
        TH = EV.thesis(s)
        L.append(f"\n## {s}\n")
        a, p, g = cohort_tables(O)
        L += ["### Cohort under today's exit\n", C.md(a), "\n### Parity (replay vs booked / recorded)\n", C.md(p, 3),
              "\n### Give-back decomposition (today's exit; $ at $3,050 notional per fill)\n", C.md(g)]
        R1 = R["thesis"]
        cols = ["rule", "fired", "fired_tp", "fired_loss", "WR_before", "WR_after", "avg_before", "avg_after", "d_mean", "ci_lo", "ci_hi", "p_pos",
                "saved_usd", "chopped_usd", "sum_usd", "loo_min", "loo_max", "top_pair", "top_pair_share", "dcr_before", "dcr_after"]
        for per in EV.PERIODS:
            x = R1[R1.period == per]
            if not len(x):
                continue
            x = x.copy()
            nl = R["thesis_null"].get(per)
            if nl:
                x["p_familywise"] = x.rule.map(nl["p_fw"])
            L += [f"\n### Layer 1 thesis rules + fixed TP alternatives — {per} (N = {int(x.N.iloc[0])})" +
                  (f" · family-wise null q95 of per-fill mean Δ = {nl['q95']:+.3f}" if nl else "") + "\n",
                  C.md(x[cols + (["p_familywise"] if nl else [])].sort_values("d_mean", ascending=False), 2)]
        for per in (("MASTER", "BATCH"), ("DESIGN", "HOLDOUT")):
            if O.period.isin(per).any():
                L += [f"\n### Thesis-variable timing vs the final in-trade peak — {' + '.join(per)} (fires only once peak ≥ +2 %)\n",
                      C.md(timing(O, dec, TH, per), 1)]
        S = R["screen"]; dcd = R["screen_decide"]
        if dcd in S:
            top = S[dcd][S[dcd].fired >= 5].sort_values("t", ascending=False).head(15).copy()
            for per in EV.PERIODS:
                if per != dcd and per in S:
                    mm = S[per].set_index("test")
                    top[f"t_{per}"] = top.test.map(mm.t); top[f"d_{per}"] = top.test.map(mm.d_mean)
            L += [f"\n### Layer 2 broad screen — deciding cohort {dcd}: {len(S[dcd])} tests ({int((S[dcd].dim == 1).sum())} 1D + "
                  f"{int((S[dcd].dim == 2).sum())} 2D), gate G1 · family-wise null q95 of max t = {R.get('screen_null_q95', np.nan):.2f}\n",
                  C.md(top, 2)]
        cv = R["coverage"]
        L += ["\n### Coverage (share of decision minutes with the variable defined)\n", C.md(cv, 2)]
        cs = cases(s, O, dec, TH)
        if len(cs):
            L += ["\n### Named live cases (illustration only — first firing after the peak reached ≥ +2 %: minutes after entry @ gross P&L)\n", C.md(cs)]
        L.append(f"\n![{s}](STATE_EXIT_2026-11-21_data/chart_{s}.png)\n")
        chart(s, O)
    open(os.path.join(C.ROOT, "reports", "STATE_EXIT_2026-11-21_tables.md"), "w").write("\n".join(L))
    print("written")


if __name__ == "__main__":
    main()
