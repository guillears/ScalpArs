#!/usr/bin/env python3
"""TP_GRID 2026-11-20 (EXHAUSTION_SHORT take-profit level) — STEP 2 BACKTEST: evaluation.

Research only. Reads reports/TP_GRID_2026-11-20_EXH_data/sim_{FRENZY,WIDE,E}.npz (scripts/TP_GRID_2026-11-20_EXH_sim.py).
Nets (same as EXH_EXIT2): gross + real funding − entry (0.05 % taker + slip) − exit (maker 0.02 % on a TP; taker 0.05 % + slip on a
stop / time exit); slip = max(0.10 %, size model at 0.61 × $10k) per taker leg; stress = flat 0.20 % per taker leg.
Per cohort (FRENZY · WIDE · TODAY = FRENZY ∪ WIDE, today's two universes) × block (DESIGN 2023-07 → 2025-06 · HOLDOUT 2025-07 → 2026-11-18 ·
2026 · ALL · each half-year) × cell: N · WR · breakeven WR · avg win / loss · avg net · Σ · TP / SL / time share · paired Δ vs today
(day-block bootstrap 95 % CI, 10-day blocks) · @0.20 · best-5 %-out · one-size DCR 0.61 (every calendar day of the block, 0 on no-fill
days, P&L on the exit day) · max DD · worst day · top pair share of Δ + leave-top-pair-out Δ · Kelly at the Wilson floor.
Family-wise null: day-level sign flip of the paired Δ, same flips for every cell, 4,000 draws, max-T (Δ > 0) and min-T (Δ < 0),
TODAY ALL and FRENZY DESIGN. Decomposition (who gains / who pays) by today's exit outcome. E (2023-H1): DL 582 block + one position.
Writes reports/TP_GRID_2026-11-20_EXH_{grid,null,decomp,E}.csv + _data/eval.pkl.
Usage: venv/bin/python -I scripts/TP_GRID_2026-11-20_EXH_eval.py"""
import importlib.util, os, pickle, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("tpsim", os.path.join(HERE, "TP_GRID_2026-11-20_EXH_sim.py"))
TS = importlib.util.module_from_spec(spec); spec.loader.exec_module(TS)
import exh_exit2_eval as XE   # noqa: E402  (e_block, one_pos, wilson_floor)
R = TS.R; RP = TS.RP; D = TS.D
OUT = os.path.join(RP, "TP_GRID_2026-11-20_EXH")
DAY = 86_400_000; H1 = 3_600_000
SIZE = 0.61
CL = TS.cells(); NAMES = [c["name"] for c in CL]; REF = 0
GRID = [i for i, n in enumerate(NAMES) if n != "REF"]
ts = lambda s: int(pd.Timestamp(s).value // 10**6) // DAY
D0, DH, D26, D1 = ts("2023-07-01"), ts("2025-07-01"), ts("2026-01-01"), ts("2026-11-19")
HALVES = ["2023-H2", "2024-H1", "2024-H2", "2025-H1", "2025-H2", "2026-H1", "2026-H2"]
RNG = np.random.default_rng(20261120)


def load(tag):
    X = pd.read_pickle(os.path.join(D, f"rows_{tag}.pkl")).reset_index(drop=True)
    Z = dict(np.load(os.path.join(D, f"sim_{tag}.npz")))
    assert (Z["xi"] == X.xi.values).all()
    ok = Z["ok"]
    return X[ok].reset_index(drop=True), {k: (v[ok] if k != "names" and len(v) == len(ok) else v) for k, v in Z.items()}


def nets(X, Z, flat=None):
    s = np.maximum(X.slip.values.astype(float), 0.10)[:, None] if flat is None else flat
    tk = Z["k"] != 0
    return Z["r"].astype(float) + Z["f"].astype(float) - s - 0.05 - np.where(tk, s + 0.05, 0.02)


def half_of(ms):
    d = pd.to_datetime(ms, unit="ms")
    return np.array([f"{a}-H{1 if b <= 6 else 2}" for a, b in zip(d.year, d.month)])


def onesize(v, xd, d0, d1):
    r = pd.Series(v * SIZE / 100).groupby(xd).sum().reindex(np.arange(d0, d1), fill_value=0.0)
    eq = np.cumprod(1 + r.values)
    return dict(dcr=(eq[-1] ** (1 / len(r)) - 1) * 100 if eq[-1] > 0 else np.nan,
                maxdd=(1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]).max() * 100, worst_day=r.min() * 100)


def boot_ci(v, days, nb=2000, block=10):
    s = pd.Series(v).groupby(days).agg(["sum", "count"])
    dd = np.arange(days.min(), days.max() + 1); s = s.reindex(dd, fill_value=0)
    n = len(dd); nbk = int(np.ceil(n / block)); st = RNG.integers(0, max(n - block, 1), (nb, nbk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :n]
    a = s["sum"].values[idx].sum(1) / np.maximum(s["count"].values[idx].sum(1), 1)
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


def kelly(v):
    w = v[v > 0]; lo = v[v <= 0]
    if not len(w) or not len(lo):
        return np.nan
    pf = XE.wilson_floor(len(w), len(v)); b = w.mean() / -lo.mean()
    return pf - (1 - pf) / b


def stats(C, j, idx, d0, d1):
    v = C["NET"][idx, j]; ref = C["NET"][idx, REF]; dl = v - ref; ed = C["ed"][idx]; xd = C["xm"][idx, j] // DAY
    k = C["k"][idx, j]
    w = v[v > 0]; lo = v[v <= 0]
    sv = np.sort(v); n95 = int(np.floor(len(v) * 0.95))
    ci = boot_ci(dl, ed) if j != REF else (np.nan, np.nan)
    pp = pd.Series(dl).groupby(C["pair"][idx]).sum()
    tot = dl.sum()
    top = pp.abs().idxmax() if len(pp) else ""
    rec = dict(N=len(idx), WR=(v > 0).mean() * 100, BE=(-lo.mean() / (w.mean() - lo.mean()) * 100) if len(w) and len(lo) else np.nan,
               avg_win=w.mean() if len(w) else np.nan, avg_loss=lo.mean() if len(lo) else np.nan, avg=v.mean(), sum=v.sum(),
               tp=(k == 0).mean() * 100, sl=(k == 1).mean() * 100, time=(k == 2).mean() * 100,
               hold_h=((C["xm"][idx, j] - C["T"][idx]) / H1).mean(),
               delta=dl.mean(), d_lo=ci[0], d_hi=ci[1], f20=C["N20"][idx, j].mean(), d20=(C["N20"][idx, j] - C["N20"][idx, REF]).mean(),
               b5=sv[:n95].mean(), kelly=kelly(v),
               top_pair=top, top_pair_share=pp[top] / tot * 100 if tot != 0 and top else np.nan,
               loo_pair=dl[C["pair"][idx] != top].mean() if top else np.nan, **onesize(v, xd, d0, d1))
    return rec


def build(tags):
    parts = []
    for tg in tags:
        X, Z = load(tg)
        parts.append((tg, X, Z))
    X = pd.concat([p[1] for p in parts], ignore_index=True)
    C = dict(NET=np.vstack([nets(p[1], p[2]) for p in parts]), N20=np.vstack([nets(p[1], p[2], 0.20) for p in parts]),
             k=np.vstack([p[2]["k"] for p in parts]), xm=np.vstack([p[2]["xm"] for p in parts]),
             r=np.vstack([p[2]["r"] for p in parts]), mfe=np.concatenate([p[2]["best"][:, 0] for p in parts]),
             rch=np.concatenate([p[2]["reach"][:, 0] for p in parts]),
             T=X.entry_ms.values.astype("int64"), pair=X.pair.values, univ=np.concatenate([[p[0]] * len(p[1]) for p in parts]))
    C["ed"] = C["T"] // DAY; C["half"] = half_of(C["T"])
    return X, C


def blocks(C):
    ed = C["ed"]; out = {"DESIGN": (ed < DH, D0, DH), "HOLDOUT": (ed >= DH, DH, D1), "2026": (ed >= D26, D26, D1),
                         "ALL": (np.ones(len(ed), bool), D0, D1)}
    for h in HALVES:
        y, hh = int(h[:4]), int(h[-1])
        a = ts(f"{y}-{'01' if hh == 1 else '07'}-01"); b = min(ts(f"{y + (hh == 2)}-{'07' if hh == 1 else '01'}-01"), D1)
        out[h] = (C["half"] == h, a, b)
    return out


def null(C, mask, ndraw=4000):
    ed = C["ed"][mask]; ud, inv = np.unique(ed, return_inverse=True); nd = len(ud)
    Dd = np.zeros((len(GRID), nd))
    for q, j in enumerate(GRID):
        np.add.at(Dd[q], inv, C["NET"][mask, j] - C["NET"][mask, REF])
    t = lambda M: M.mean(-1) / (M.std(-1, ddof=1) / np.sqrt(nd) + 1e-12)
    tobs = t(Dd); mx = np.zeros(ndraw); mn = np.zeros(ndraw)
    for b in range(ndraw):
        tb = t(Dd * RNG.choice((-1.0, 1.0), nd)[None, :]); mx[b] = tb.max(); mn[b] = tb.min()
    return pd.DataFrame(dict(cell=[NAMES[j] for j in GRID], t_obs=tobs, p_fw_better=[(mx >= x).mean() for x in tobs],
                             p_fw_worse=[(mn <= x).mean() for x in tobs]))


def decomp(C, mask, j):
    kr = C["k"][mask, REF]; kc = C["k"][mask, j]; d = C["NET"][mask, j] - C["NET"][mask, REF]
    rr = C["r"][mask, REF]
    grp = np.where((kr == 0) & (rr >= 9.99) & (kc == 0), "REF +10 % TP → cell's smaller TP (pays)",
          np.where((kr == 0) & (kc == 0), "REF TP < 10 % (ATR < 0.5 %) → same / smaller TP",
          np.where((kr == 1) & (kc == 0), "REF stopped → cell hits its TP first (gains)",
          np.where((kr == 2) & (kc == 0), "REF time exit → cell hits its TP (gains or pays)",
          np.where((kr == 1) & (kc == 1), "stopped in both (Δ = 0)",
          np.where((kr == 2) & (kc == 2), "time exit in both (Δ = 0)", "other"))))))
    out = []
    for g in pd.unique(grp):
        m = grp == g
        out.append(dict(group=g, N=int(m.sum()), share=m.mean() * 100, sum_delta=d[m].sum(), avg_delta=d[m].mean(),
                        usd_at_6100=d[m].sum() * 61.0))
    return pd.DataFrame(out).sort_values("sum_delta")


def main():
    pd.set_option("display.width", 280); pd.set_option("display.max_columns", 40)
    # parity: REF vs EXH_EXIT2 REF on the same FRENZY rows
    _, Zf = load("FRENZY")
    Z2 = dict(np.load(os.path.join(RP, "EXH_EXIT2_data", "sim_EXH.npz")))
    same_k = (Zf["k"][:, REF] == Z2["k"][:, 0]).mean(); dr = np.abs(Zf["r"][:, REF] - Z2["r"][:, 0]).max()
    B = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv")); B = B[~B.dl582_block.astype(bool)].reset_index(drop=True)
    bk = B.exit_kind.map({"TP": 0, "SL": 1, "TIME": 2}).values
    parity = dict(same_kind_vs_EXH_EXIT2=same_k, max_abs_r_diff=float(dr), same_kind_vs_baseline=(Zf["k"][:, REF] == bk).mean(),
                  max_abs_r_vs_baseline=float(np.abs(Zf["r"][:, REF] - B.r.values).max()))
    Xw, Zw = load("WIDE")
    Ww = pd.read_pickle(os.path.join(D, "rows_WIDE.pkl")).set_index("xi").loc[Xw.xi]
    parity.update(WIDE_ok=len(Xw), WIDE_same_kind_vs_L4=(Zw["k"][:, REF] == Ww.base_kind.values).mean(),
                  WIDE_r_med_absdiff=float(np.median(np.abs(Zw["r"][:, REF] - Ww.base_r.values))))
    print("parity", parity, flush=True)
    res = []; nulls = []; dec = []; pk = {}
    for coh, tags in (("FRENZY", ["FRENZY"]), ("WIDE", ["WIDE"]), ("TODAY", ["FRENZY", "WIDE"])):
        X, C = build(tags); pk[coh] = C
        for bn, (m, d0, d1) in blocks(C).items():
            idx = np.nonzero(m)[0]
            if not len(idx):
                continue
            for j in range(len(CL)):
                res.append(dict(cohort=coh, block=bn, cell=NAMES[j], **stats(C, j, idx, d0, d1)))
        for bn in ("ALL", "DESIGN", "HOLDOUT"):
            m = blocks(C)[bn][0]
            nu = null(C, m); nu.insert(0, "block", bn); nu.insert(0, "cohort", coh); nulls.append(nu)
        for nm in ("F6", "F7", "F8", "F9", "K3_C10"):
            dd = decomp(C, np.ones(len(C["T"]), bool), NAMES.index(nm)); dd.insert(0, "cell", nm); dd.insert(0, "cohort", coh); dec.append(dd)
        print(coh, "done", flush=True)
    G = pd.DataFrame(res); NU = pd.concat(nulls); DE = pd.concat(dec)
    # E 2023-H1 (DL 582 block + one position under today's exit)
    XE_, ZE = load("E")
    NE = nets(XE_, ZE)
    oiv, blkE = XE.e_block(XE_)
    keep = np.nonzero(~blkE)[0]
    keep = keep[XE.one_pos(XE_.pair.values[keep], XE_.entry_ms.values[keep], ZE["xm"][keep, REF])]
    erow = []
    e0, e1 = ts("2023-01-01"), ts("2023-07-01")
    for j in range(len(CL)):
        v = NE[keep, j]; dl = v - NE[keep, REF]; ci = boot_ci(dl, XE_.entry_ms.values[keep] // DAY) if j else (np.nan, np.nan)
        erow.append(dict(cell=NAMES[j], N=len(v), WR=(v > 0).mean() * 100, avg=v.mean(), sum=v.sum(), delta=dl.mean(), d_lo=ci[0], d_hi=ci[1],
                         **onesize(v, ZE["xm"][keep, j] // DAY, e0, e1)))
    EE = pd.DataFrame(erow)
    for nm, df in (("grid", G), ("null", NU), ("decomp", DE), ("E", EE)):
        df.to_csv(f"{OUT}_{nm}.csv", index=False)
    # extra facts: MFE of the stopped fills, MFE of the +10 % winners in the first hours not needed — reach before the stop
    C = pk["TODAY"]; st = C["k"][:, REF] == 1
    reach = -C["rch"][st]
    facts = dict(parity=parity, E_info=dict(cands=len(XE_), blocked=int(blkE.sum()), kept=len(keep)),
                 stopped=int(st.sum()), **{f"stopped_reach_ge{a}": float((reach >= a).mean() * 100) for a in (4, 5, 6, 7, 8, 9, 10)},
                 tp10_n=int(((C["k"][:, REF] == 0) & (C["r"][:, REF] >= 9.99)).sum()),
                 tp10_mfe_med=float(np.median(C["mfe"][(C["k"][:, REF] == 0) & (C["r"][:, REF] >= 9.99)])))
    pickle.dump(facts, open(os.path.join(D, "eval.pkl"), "wb"))
    print(facts)
    show = ["N", "WR", "BE", "avg_win", "avg_loss", "avg", "sum", "tp", "sl", "time", "delta", "d_lo", "d_hi", "d20", "b5", "dcr", "maxdd", "worst_day"]
    main_cells = ["REF", "F4", "F5", "F6", "F7", "F8", "F9", "K1.5_C10", "K2_C10", "K3_C10", "K5_C10", "K2_C8", "K3_C8", "F8_T05", "REF_T05"]
    for coh in ("TODAY", "FRENZY", "WIDE"):
        for bn in ("ALL", "DESIGN", "HOLDOUT", "2026"):
            g = G[(G.cohort == coh) & (G.block == bn) & G.cell.isin(main_cells)].set_index("cell").loc[main_cells, show]
            print(f"\n== {coh} {bn}"); print(g.round(2).to_string())
    print("\n== halves TODAY avg"); print(G[(G.cohort == "TODAY") & G.block.isin(HALVES)].pivot(index="cell", columns="block", values="avg").loc[main_cells].round(2).to_string())
    print("\n== halves TODAY delta"); print(G[(G.cohort == "TODAY") & G.block.isin(HALVES)].pivot(index="cell", columns="block", values="delta").loc[main_cells].round(2).to_string())
    print("\n== null"); print(NU.round(3).to_string())
    print("\n== decomp"); print(DE.round(2).to_string())
    print("\n== E"); print(EE.round(2).to_string())
    print(G[(G.cohort == "TODAY") & (G.block == "ALL")][["cell", "top_pair", "top_pair_share", "loo_pair", "kelly", "hold_h"]].round(3).to_string())


if __name__ == "__main__":
    main()
