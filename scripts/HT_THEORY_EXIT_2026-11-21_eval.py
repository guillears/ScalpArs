#!/usr/bin/env python3
"""HT_THEORY_EXIT 2026-11-21 — EVAL (PREREG reports/HT_THEORY_EXIT_2026-11-21_PREREG.md). Reads HT_THEORY_EXIT_2026-11-21_data/sim.pkl.
Parity gate · per cell × cohort: N, WR, avg, Δ, saved / cut $ ($3,050 per fill), day-block bootstrap P(Δ>0) + 95 % CI, one-size DCR
before → after (0.61 × equity per fill), slip 0.20 Δ, pair / day concentration, leave-one-pair-out, avg loss, worst day.
→ reports/HT_THEORY_EXIT_2026-11-21_cells.csv + reports/HT_THEORY_EXIT_2026-11-21_tables.md"""
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
_sp = importlib.util.spec_from_file_location("HTS", os.path.join(HERE, "HT_THEORY_EXIT_2026-11-21_sim.py"))
HTS = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(HTS)
D = HTS.D
NOT = 3050.0
X = 0.61
DAYMS = 86_400_000


def dcr(net, xms, t0, t1):
    if not len(net):
        return np.nan
    d = (np.asarray(xms, "int64") - 60_000) // DAYMS
    days = np.arange(int(t0) // DAYMS, int(t1) // DAYMS + 1)
    r = pd.Series(np.asarray(net) * X / 100).groupby(d).sum().reindex(days, fill_value=0.0).values
    return (np.prod(1 + r) ** (1 / len(days)) - 1) * 100


def boot(delta, day, B=3000, seed=7):
    u = np.unique(day)
    if len(u) < 2:
        return np.nan, np.nan, np.nan
    s = pd.DataFrame({"d": day, "x": delta}).groupby("d").x.agg(["sum", "count"]).loc[u]
    S_, N_ = s["sum"].values, s["count"].values
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(u), (B, len(u)))
    m = S_[ix].sum(1) / N_[ix].sum(1)
    return float((m > 0).mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def read(G, cell):
    g = G[G[cell].notna() & G["REF"].notna()]
    n = len(g)
    if not n:
        return None
    net, ref = g[cell].values, g["REF"].values
    dl = net - ref
    day = (g.te // DAYMS).values
    P, lo, hi = boot(dl, day)
    pos = dl[dl > 0]
    t0, t1 = g.te.min(), max(g[cell + "_x"].max(), g["REF_x"].max())
    by_pair = pd.Series(dl, index=g.pair.values).groupby(level=0).sum()
    by_day = pd.Series(dl, index=day).groupby(level=0).sum()
    tot = dl.sum()
    conc_p = by_pair.max() / pos.sum() if pos.sum() > 0 else np.nan
    conc_d = by_day.max() / pos.sum() if pos.sum() > 0 else np.nan
    loo = min(((tot - by_pair[p]) / (n - (g.pair == p).sum())) if n > (g.pair == p).sum() else np.nan for p in by_pair.index)
    s20 = (g[cell + "_s20"] - g["REF_s20"]).mean() if cell + "_s20" in g else np.nan
    dd = pd.Series(net * NOT / 100, index=day).groupby(level=0).sum()
    fired = int((np.abs(dl) > 1e-9).sum())
    return dict(N=n, fired=fired, WR=(net > 0).mean() * 100, avg=net.mean(), ref_avg=ref.mean(), d=dl.mean(), P=P, lo=lo, hi=hi,
                saved=pos.sum() * NOT / 100, cut=-dl[dl < 0].sum() * NOT / 100, usd=net.sum() * NOT / 100,
                dcr_b=dcr(ref, g["REF_x"].values, t0, t1), dcr_a=dcr(net, g[cell + "_x"].values, t0, t1), s20=s20,
                conc_pair=conc_p, top_pair=(by_pair.idxmax() if len(by_pair) else ""), conc_day=conc_d, loo=loo,
                avg_loss=net[net < 0].mean() if (net < 0).any() else 0.0, worst_day=dd.min())


def cohorts(O, sl):
    s = O[O.sleeve == sl]
    return {"MASTER_TF": s[(s.kind == "REAL") & (s.period == "MASTER") & s.today_keep.astype(bool)],
            "MASTER_ALL": s[(s.kind == "REAL") & (s.period == "MASTER")],
            "BATCH": s[(s.kind == "REAL") & (s.period == "BATCH")],
            "DESIGN": s[(s.kind == "BT") & (s.period == "DESIGN")],
            "HOLDOUT": s[(s.kind == "BT") & (s.period == "HOLDOUT")]}


def parity(O):
    L = []
    h = O[(O.sleeve == "HT") & (O.kind == "REAL") & (O.tp_target.astype(float).round(6) == 5.0) & O.REF.notna()]
    mp_ = {"FRENZY_TP": "TP", "STOP_LOSS": "STOP", "MAX_HOLD_TIME": "TIME"}
    same = (h.booked_reason.map(mp_) == h.REF_k)
    L.append(("HT real booked +5/−4", len(h), int(same.sum()), float((h.par_net - h.booked).abs().median()), float((h.REF - h.booked).abs().mean()),
              ", ".join(f"{r.pair} {r.k[5:16]} booked {r.booked_reason} {r.booked:+.2f} / REF {r.REF_k} {r.REF:+.2f}" for r in h[~same].itertuples())))
    b = O[(O.sleeve == "HT") & (O.kind == "BT") & O.REF.notna()]
    mb = {"TP": "TP", "SL": "STOP", "TIME": "TIME"}
    same = b.booked_reason.map(mb) == b.REF_k
    L.append(("HT backtest ref_kind", len(b), int(same.sum()), np.nan, np.nan,
              ", ".join(f"{r.pair} {r.k[:16]} {r.booked_reason}/{r.REF_k}" for r in b[~same].head(8).itertuples())))
    w = O[(O.sleeve == "WILLY") & (O.kind == "BT") & O.REF.notna()]
    same = w.booked_reason == w.REF_kk
    L.append(("WILLY backtest ref_kind", len(w), int(same.sum()), np.nan, np.nan,
              ", ".join(f"{r.pair} {r.k[:16]} {r.booked_reason}/{r.REF_kk}" for r in w[~same].head(8).itertuples())))
    wr = O[(O.sleeve == "WILLY") & (O.kind == "REAL") & O.REF.notna() & (O.k >= "2026-10-23T12")]
    mw = {"FRENZY_TP": "TP", "STOP_LOSS": "SL", "MAX_HOLD_TIME": "TIME"}
    same = wr.booked_reason.map(mw).fillna("DYN") == wr.REF_kk
    L.append(("WILLY real booked (V2b era, info only)", len(wr), int(same.sum()), np.nan, float((wr.REF - wr.booked).abs().mean()), ""))
    return pd.DataFrame(L, columns=["check", "N", "same kind", "median |paper-mode − booked|", "mean |REF − booked|", "mismatches"])


def main():
    O = pd.read_pickle(os.path.join(D, "sim.pkl"))
    O = O[O.REF.notna()].copy()
    P = parity(O)
    rows = []
    for sl, cells, fixed in (("HT", HTS.HT_CELLS, HTS.HT_FIXED), ("WILLY", HTS.W_CELLS, HTS.W_FIXED)):
        C = cohorts(O, sl)
        for cname, G in C.items():
            base = read(G, "REF")
            for cell in ["REF"] + cells + fixed:
                r = read(G, cell) if cell != "REF" else base
                if r is None:
                    continue
                rows.append(dict(sleeve=sl, cohort=cname, cell=cell, fam=("FIXED" if cell in fixed else ("REF" if cell == "REF" else "RULE")), **r))
    T = pd.DataFrame(rows)
    # pre-committed candidate test on MASTER_TF
    for sl in ("HT", "WILLY"):
        m = T[(T.sleeve == sl) & (T.cohort == "MASTER_TF")]
        bestfix = m[m.fam == "FIXED"].d.max()
        T.loc[(T.sleeve == sl), "best_fixed_master"] = bestfix
        ok = (m.d >= 0) & (m.saved >= m.cut) & (m.P >= 0.80) & (m.d > bestfix) & (m.fam == "RULE")
        T.loc[m.index, "CANDIDATE"] = ok.values
    T.to_csv(os.path.join(ROOT, "reports", "HT_THEORY_EXIT_2026-11-21_cells.csv"), index=False)
    md = ["# HT_THEORY_EXIT 2026-11-21 — auto tables", "", "## Parity", "", HTS.SEC.md(P), ""]
    show = ["cell", "N", "fired", "WR", "avg", "d", "lo", "hi", "P", "saved", "cut", "usd", "dcr_b", "dcr_a", "s20", "conc_pair", "top_pair",
            "conc_day", "loo", "avg_loss", "worst_day"]
    for sl in ("HT", "WILLY"):
        for cn in ("MASTER_TF", "MASTER_ALL", "BATCH", "DESIGN", "HOLDOUT"):
            g = T[(T.sleeve == sl) & (T.cohort == cn)]
            md += [f"## {sl} · {cn}", "", HTS.SEC.md(g[show].round(3)), ""]
        m = T[(T.sleeve == sl) & (T.cohort == "MASTER_TF") & (T.CANDIDATE == True)]  # noqa: E712
        md += [f"**{sl} candidates on MASTER_TF (pre-committed rule):** " + (", ".join(m.cell) if len(m) else "none"), ""]
    open(os.path.join(ROOT, "reports", "HT_THEORY_EXIT_2026-11-21_tables.md"), "w").write("\n".join(md))
    print(P.to_string())
    for sl in ("HT", "WILLY"):
        print(T[(T.sleeve == sl) & (T.cohort == "MASTER_TF")][["cell", "N", "fired", "WR", "avg", "d", "P", "saved", "cut", "dcr_b", "dcr_a", "CANDIDATE"]].round(3).to_string())


if __name__ == "__main__":
    main()
