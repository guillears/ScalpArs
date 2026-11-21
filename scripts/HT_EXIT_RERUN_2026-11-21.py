#!/usr/bin/env python3
"""HT_EXIT_RERUN 2026-11-21 — research only (no engine / config change). Re-reads the FRENZY_WILLY_HT exit study on the CORRECT
today's-filter cohorts (reports/KEPT_LIST_WILLY_HT_2026-11-21.csv: dip ≥ 1.4 AND WILLY_HT_FAILED_SPIKE, master 12 / batch 12).
Imports (never edits) scripts/HT_THEORY_EXIT_2026-11-21_sim.py / _eval.py; reads HT_THEORY_EXIT_2026-11-21_data/sim.pkl + its 1m caches.
New here: TP grid 4.5 / 5 / 5.5 / 6 / 7 / 8 re-walked with the study's own engine (run_ht, same costs, slip 0.10 and 0.20),
P(reach +X before the stop | reached +5), durations, before → after tables, gate set, revert-gate pricing, chart.
→ reports/HT_EXIT_RERUN_2026-11-21_{grid.pkl, perfill.csv, tables.md, chart.png}
Usage: venv/bin/python -I scripts/HT_EXIT_RERUN_2026-11-21.py [--procs 8]"""
import argparse, importlib.util, math, os
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
_sp = importlib.util.spec_from_file_location("HTE", os.path.join(HERE, "HT_THEORY_EXIT_2026-11-21_eval.py"))
HTE = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(HTE)
HTS = HTE.HTS
RP = os.path.join(ROOT, "reports")
OUT = os.path.join(RP, "HT_EXIT_RERUN_2026-11-21")
TPS = (4.5, 5.0, 5.5, 6.0, 7.0, 8.0)
NOT, X, MIN, HR = 3050.0, 0.61, 60_000, 3_600_000
HALF = lambda ms: (lambda t: f"{t.year % 100}-H{1 if t.month <= 6 else 2}")(pd.Timestamp(int(ms), unit="ms"))


def name(tp):
    return f"G{tp:g}"


def sim_grid(fl, slip):
    """the TP grid on one fill — same setup as HTS.sim_ht (entry bar, 24 h end, tick, costs, stop −4 net, funding)."""
    p, te, E = fl["pair"], int(fl["te"]), float(fl["E"])
    sp = int(fl["spike_ms"]) if fl["spike_ms"] == fl["spike_ms"] else te - HR
    a = (min(te, sp) - 60 * MIN) // MIN * MIN
    g = HTS.bars(p, a, te + 25 * HR, fresh=(fl["kind"] == "REAL"))
    if g is None:
        return None
    t = g["t"]
    T0 = (te // MIN) * MIN + (MIN if fl["start"] == "next" else 0)
    i0 = int(np.searchsorted(t, T0))
    if i0 >= len(t) or t[i0] - T0 > 5 * MIN:
        return None
    t_end = T0 + 24 * HR
    jend = int(np.searchsorted(t, t_end, "left"))
    has_end = jend < len(t) and t[jend] - t_end < 30 * MIN
    tick = HTS.S.tick_for(p, E)
    sl = HTS.stop_px(E, 4.0)
    out = {}
    for tp in TPS:
        r = HTS.run_ht(g, i0, jend, E, tick, HTS.lp_net(E, tp, HTS.MAKER, tick), sl, slip, HTS.TAKER, has_end=has_end)
        if r is None:
            out[name(tp)] = np.nan; out[name(tp) + "_k"] = "NA"; out[name(tp) + "_x"] = 0; continue
        net0, j, kind = r
        xm = int(t[j]) + MIN
        out[name(tp)] = net0 + HTS.fund_long(fl, xm); out[name(tp) + "_k"] = kind; out[name(tp) + "_x"] = xm
    # max gross reach before the stop (within 24 h) — descriptive
    hs = np.flatnonzero(g["l"][i0:jend] <= sl)
    jr = i0 + (hs[0] if len(hs) else jend - i0)
    out["reach_pre_stop"] = float((g["h"][i0:jr + (1 if len(hs) else 0)].max() / E - 1) * 100) if jr > i0 else np.nan
    return out


def work(fl):
    try:
        a = sim_grid(fl, 0.10)
        b = sim_grid(fl, 0.20)
        if a is None:
            return fl["fid"], None
        for tp in TPS:
            a[name(tp) + "_s20"] = b[name(tp)] if b else np.nan
        return fl["fid"], a
    except Exception as e:  # noqa: BLE001
        return fl["fid"], {"err": f"{type(e).__name__}: {e}"}


def build(procs):
    f = OUT + "_grid.pkl"
    if os.path.exists(f):
        return pd.read_pickle(f)
    O = pd.read_pickle(os.path.join(HTE.D, "sim.pkl"))
    O = O[(O.sleeve == "HT") & O.REF.notna()].copy()
    recs = O[["fid", "pair", "te", "E", "spike_ms", "start", "kind"]].to_dict("records")
    with mp.Pool(procs) as pool:
        res = dict(pool.map(work, recs, chunksize=2))
    G = pd.DataFrame([dict(fid=k, **v) for k, v in res.items() if v is not None])
    O = O.merge(G, on="fid", how="left")
    O.to_pickle(f)
    return O


# ───────────────────────────── reads ─────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def summary(G, cell):
    g = G[G[cell].notna()]
    net = g[cell].values
    dur = (g[cell + "_x"].values - g.te.values) / MIN
    k = g[cell + "_k"].values
    t0, t1 = g.te.min(), max(g[cell + "_x"].max(), g["G5_x"].max())
    return dict(N=len(g), WR=(net > 0).mean() * 100, avg=net.mean(), usd=net.sum() * NOT / 100,
                avg_win=net[net > 0].mean() if (net > 0).any() else np.nan, avg_loss=net[net < 0].mean() if (net < 0).any() else np.nan,
                worst=net.min(), dcr=HTE.dcr(net, g[cell + "_x"].values, t0, t1),
                dur_avg_min=dur.mean(), dur_med_min=np.median(dur),
                n_TP=int((k == "TP").sum()), n_STOP=int((k == "STOP").sum()), n_TIME=int(np.isin(k, ["TIME", "OPEN"]).sum()))


def delta_read(G, cell, ref="G5"):
    g = G[G[cell].notna() & G[ref].notna()]
    dl = g[cell].values - g[ref].values
    day = (g.te // HTE.DAYMS).values
    P, lo, hi = HTE.boot(dl, day)
    s20 = (g[cell + "_s20"] - g[ref + "_s20"]).mean()
    loo_fill = min((dl.sum() - x) / (len(dl) - 1) for x in dl) if len(dl) > 1 else np.nan
    bp = pd.Series(dl, index=g.pair.values).groupby(level=0).sum()
    loo_pair = min(((dl.sum() - bp[p]) / (len(dl) - (g.pair == p).sum())) if len(dl) > (g.pair == p).sum() else np.nan for p in bp.index)
    pos = dl[dl > 0].sum()
    return dict(N=len(g), d=dl.mean(), d_usd=dl.sum() * NOT / 100, better=int((dl > 1e-9).sum()), worse=int((dl < -1e-9).sum()),
                P=P, lo=lo, hi=hi, s20=s20, loo_fill=loo_fill, loo_pair=loo_pair,
                conc_pair=(bp.max() / pos if pos > 0 else np.nan), top_pair=bp.idxmax() if len(bp) else "")


def cond(G, tp):
    """P(reach +tp before the stop | reached +5 before the stop), both inside 24 h; and the realised cost of the misses."""
    g = G[G["G5_k"] == "TP"]
    hit = g[name(tp) + "_k"] == "TP"
    k, n = int(hit.sum()), len(g)
    lo, hi = wilson(k, n)
    gain = (g.loc[hit, name(tp)] - g.loc[hit, "G5"]).mean() if k else tp - 5.0
    miss = g.loc[~hit, name(tp)] - g.loc[~hit, "G5"]
    cost = -miss.mean() if len(miss) else np.nan
    cost_be = cost if cost == cost else 9.1   # no miss observed → a −4 stop after a +5 touch ≈ 5 − (−4.1)
    be = cost_be / (gain + cost_be)
    return dict(n5=n, hit=k, P=k / n if n else np.nan, lo=lo, hi=hi, gain=gain, miss_cost=cost, be=be,
                miss_kinds=", ".join(g.loc[~hit, name(tp) + "_k"].astype(str)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); A = ap.parse_args()
    O = build(A.procs)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    # parity: grid G5 == REF, G6 == TP6, G4.5 == TP4.5 (same engine)
    par = {c: float((O[g_] - O[c]).abs().max()) for g_, c in (("G5", "REF"), ("G6", "TP6"), ("G4.5", "TP4.5"))}
    print("parity max|Δ|", par, "err rows", int(O.get("err", pd.Series(dtype=object)).notna().sum()))
    K = pd.read_csv(os.path.join(RP, "KEPT_LIST_WILLY_HT_2026-11-21.csv"))
    R = O[O.kind == "REAL"]
    M = R.merge(K, on=["pair", "k"]); M = M[M.block == "MASTER"]; B = R.merge(K, on=["pair", "k"]); B = B[B.block == "BATCH"]
    assert len(M) == 12 and len(B) == 12, (len(M), len(B))
    BT = O[O.kind == "BT"].copy(); BT["half_y"] = BT.te.map(HALF)
    C = {"MASTER (12, today's filters)": M, "BATCH (12, today's filters)": B, "ALL REAL HT ever (any filter)": R,
         "BACKTEST DESIGN (<2025-07)": BT[BT.period == "DESIGN"], "BACKTEST HOLDOUT (≥2025-07)": BT[BT.period == "HOLDOUT"],
         "BACKTEST ALL": BT}
    for h in sorted(BT.half_y.unique()):
        C[f"BT {h}"] = BT[BT.half_y == h]
    L = ["# HT_EXIT_RERUN 2026-11-21 — auto tables", "", f"Parity (grid vs sim.pkl, max |Δ| net pts): {par}", ""]
    # ① conditional reach
    rows = []
    for cn, G in C.items():
        for tp in (5.5, 6.0, 7.0, 8.0):
            rows.append(dict(cohort=cn, tp=tp, **cond(G, tp)))
    CR = pd.DataFrame(rows); CR.to_csv(OUT + "_cond.csv", index=False)
    L += ["## P(reach +X before the stop | reached +5), 24 h", "", HTS.SEC.md(CR.round(3)), ""]
    print(CR.round(3).to_string(index=False))
    # ② before → after per cohort
    rows = []
    for cn, G in C.items():
        for tp in TPS:
            rows.append(dict(cohort=cn, cell=f"TP {tp:g}" + (" (today)" if tp == 5 else ""), **summary(G, name(tp))))
    SM = pd.DataFrame(rows); SM.to_csv(OUT + "_summary.csv", index=False)
    L += ["## Before → after (one size $3,050 · DCR at 0.61 × equity per fill)", "", HTS.SEC.md(SM.round(2)), ""]
    print(SM.round(2).to_string(index=False))
    # ③ Δ gate reads vs TP5
    rows = []
    for cn, G in C.items():
        for tp in (4.5, 5.5, 6.0, 7.0, 8.0):
            rows.append(dict(cohort=cn, cell=f"TP {tp:g}", **delta_read(G, name(tp))))
    DR = pd.DataFrame(rows); DR.to_csv(OUT + "_delta.csv", index=False)
    L += ["## Δ vs today's TP +5 (per fill, net pts) · day-block bootstrap · slip 0.20 · LOO", "", HTS.SEC.md(DR.round(3)), ""]
    print(DR.round(3).to_string(index=False))
    # ④ other leading cells (sim.pkl, vs REF) on the correct cohorts + PREREG rule
    rows = []
    for cn in ("MASTER (12, today's filters)", "BATCH (12, today's filters)", "BACKTEST DESIGN (<2025-07)", "BACKTEST HOLDOUT (≥2025-07)"):
        G = C[cn]
        for cell in ["REF"] + HTS.HT_CELLS + HTS.HT_FIXED:
            r = HTE.read(G, cell)
            if r is not None:
                rows.append(dict(cohort=cn, cell=cell, fam=("FIXED" if cell in HTS.HT_FIXED else ("REF" if cell == "REF" else "RULE")), **r))
    OC = pd.DataFrame(rows)
    m = OC[OC.cohort.str.startswith("MASTER")]
    bestfix = m[m.fam == "FIXED"].d.max()
    OC["prereg_candidate"] = False
    ok = (m.d >= 0) & (m.saved >= m.cut) & (m.P >= 0.80) & (m.d > bestfix) & (m.fam == "RULE")
    OC.loc[m.index, "prereg_candidate"] = ok.values
    OC.to_csv(OUT + "_cells.csv", index=False)
    show = ["cohort", "cell", "N", "fired", "WR", "avg", "d", "P", "lo", "hi", "saved", "cut", "usd", "dcr_b", "dcr_a", "s20", "loo", "conc_pair",
            "avg_loss", "prereg_candidate"]
    L += [f"## Other cells vs REF (best fixed-TP Δ on master = {bestfix:+.3f})", "", HTS.SEC.md(OC[show].round(3)), ""]
    print("best fixed master", bestfix)
    print(OC[OC.cell.isin(["REF", "C_BASE", "A_H0", "A_Hp05", "A_Hm05", "N_H0", "TP4.5", "TP6", "E_H0_5", "D_HLOCK0"])][show].round(2).to_string(index=False))
    print("prereg candidates:", list(OC[OC.prereg_candidate].cell))
    # ⑤ per-fill tables
    pf = []
    for blk, G in (("MASTER", M), ("BATCH", B)):
        for r in G.sort_values("te").itertuples():
            row = dict(block=blk, pair=r.pair, opened=r.k[:16].replace("T", " "), dip=round(r.dip, 2), booked=round(r.booked, 2),
                       booked_reason=r.booked_reason, reach_pre_stop=round(r.reach_pre_stop, 2))
            for tp in TPS:
                c = name(tp)
                row[f"TP{tp:g}"] = round(G.loc[r.Index, c], 2); row[f"TP{tp:g}_k"] = G.loc[r.Index, c + "_k"]
                row[f"TP{tp:g}_min"] = round((G.loc[r.Index, c + "_x"] - r.te) / MIN)
            for c in ("C_BASE", "A_H0", "A_Hp05"):
                row[c] = round(G.loc[r.Index, c], 2)
            pf.append(row)
    PF = pd.DataFrame(pf); PF.to_csv(OUT + "_perfill.csv", index=False)
    L += ["## Per fill (master + batch)", "", HTS.SEC.md(PF), ""]
    print(PF[["block", "pair", "opened", "booked", "reach_pre_stop", "TP5", "TP5_k", "TP5_min", "TP6", "TP6_k", "TP6_min", "TP7", "TP7_k", "TP8", "TP8_k",
              "C_BASE", "A_H0"]].to_string(index=False))
    # ⑥ revert-gate pricing: first N fills that reach +5; revert if Σ(TP6 − TP5) < 0. Monte Carlo on hit prob p, gain +1.0, miss cost from data.
    allmiss = pd.concat([(G.loc[(G.G5_k == "TP") & (G.G6_k != "TP"), "G6"] - G.loc[(G.G5_k == "TP") & (G.G6_k != "TP"), "G5"]) for G in (BT, R)])
    cost = -allmiss.mean() if len(allmiss) else 9.1
    rng = np.random.default_rng(11)
    gp = []
    for N in (8, 10, 15):
        for p in (0.80, 0.85, 0.90, 0.95, 1.00):
            hits = rng.binomial(N, p, 200_000)
            s = hits * 1.0 - (N - hits) * cost
            gp.append(dict(N=N, p=p, P_revert=float((s < 0).mean()), E_sum=float(N * (p * 1.0 - (1 - p) * cost))))
    GP = pd.DataFrame(gp)
    L += [f"## Revert-gate pricing (gain +1.0 per hit, miss cost {cost:.2f} pts = mean realised miss, real + backtest)", "", HTS.SEC.md(GP.round(3)), ""]
    print("miss cost", cost); print(GP.round(3).to_string(index=False))
    open(OUT + "_tables.md", "w").write("\n".join(L))
    chart(CR, M, B)


def chart(CR, M, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, mute, blue, gray, red = "#1f2328", "#6e7781", "#2f6fd6", "#b8bec6", "#c4432b"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1, 1.35]))
    c6 = CR[(CR.tp == 6.0) & ~CR.cohort.str.startswith("BT 2")].reset_index(drop=True)
    y = np.arange(len(c6))[::-1]
    for yi, r in zip(y, c6.itertuples()):
        ax[0].plot([r.lo * 100, r.hi * 100], [yi, yi], color=gray, lw=2, solid_capstyle="round")
        ax[0].plot(r.P * 100, yi, "o", ms=9, color=blue, mec="white", mew=2)
        ax[0].text(101, yi, f"{r.hit}/{r.n5}", va="center", fontsize=9, color=mute)
    be = c6.be.median() * 100
    ax[0].axvline(be, color=red, lw=1.5, ls="--")
    ax[0].text(be - 1, y.max() + 0.6, f"breakeven ≈ {be:.0f} %", color=red, fontsize=9, ha="right")
    ax[0].set_yticks(y); ax[0].set_yticklabels([s.split(" (")[0] for s in c6.cohort], fontsize=9, color=ink)
    ax[0].set_xlim(40, 108); ax[0].set_xlabel("P(reach +6 before the stop | reached +5), % · 95 % Wilson", fontsize=9, color=mute)
    ax[0].set_title("Does a +5 winner go on to +6?", fontsize=11, color=ink, loc="left")
    D = pd.concat([M.assign(b="M"), B.assign(b="B")]).sort_values("te")
    dl = (D.G6 - D.G5).values
    xs = np.arange(len(D))
    ax[1].bar(xs, dl, color=[blue if b == "M" else "#8fb0e6" for b in D.b], width=0.7)
    ax[1].axhline(0, color=mute, lw=1)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{p.replace('USDT', '')} {k[5:10]}" for p, k in zip(D.pair, D.k)], fontsize=7.5, rotation=90, color=ink)
    ax[1].text(len(D) - 0.5, -0.12, "0 = fill stopped under both (no fill is worse under +6 in either block)", fontsize=8.5, color=mute, ha="right")
    ax[1].set_ylim(-0.2, 1.2)
    ax[1].set_ylabel("TP +6 − TP +5, net pts per fill", fontsize=9, color=mute)
    ax[1].set_title("Per fill: TP +6 vs today's +5 (dark = master 12, light = batch 12)", fontsize=11, color=ink, loc="left")
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=mute)
    fig.tight_layout(); fig.savefig(OUT + "_chart.png", dpi=130); print("chart", OUT + "_chart.png")


if __name__ == "__main__":
    main()
