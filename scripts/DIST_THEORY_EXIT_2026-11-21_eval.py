#!/usr/bin/env python3
"""DIST_THEORY_EXIT — EVAL (PREREG reports/DIST_THEORY_EXIT_2026-11-21_PREREG.md). Research only, BACKTEST.
1) parity vs DIST_EXIT REF_B (reports/DIST_EXIT_data/sim_L1.npz col 0, row zi) — stop if < 99.9 %.
2) nets = DIST_EXIT net_mat (r − slip − slip·taker − 0.05 − (0.05 taker | 0.02 maker) + funding); flat 0.20 stress.
3) DESIGN pick per family → HOLDOUT gates ①–⑦; risk at one size; leave-one-out; every cell both blocks.
→ reports/DIST_THEORY_EXIT_2026-11-21_data/{eval.csv, picks.csv, perfill.csv.gz, halves.csv}"""
import os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "DIST_THEORY_EXIT_2026-11-21_data")
DAY = 86_400_000
TS = lambda s: pd.Timestamp(s).value // 10**6
SPAN = dict(DESIGN=(TS("2023-07-01"), TS("2025-07-01")), HOLDOUT=(TS("2025-07-01"), TS("2026-11-18")))
HALVES = [("23-H2", "2023-07-01", "2024-01-01"), ("24-H1", "2024-01-01", "2024-07-01"), ("24-H2", "2024-07-01", "2025-01-01"),
          ("25-H1", "2025-01-01", "2025-07-01"), ("25-H2", "2025-07-01", "2026-01-01"), ("26-H1", "2026-01-01", "2026-07-01"),
          ("26-H2", "2026-07-01", "2026-11-19")]
X_EQ = 0.61; USD = 3050.0
FAMS = ["A", "N", "M", "B", "C", "R", "G", "F", "W"]


def nets(slip, r, k, f, flat=None):
    s = (slip if flat is None else np.full(len(slip), flat))[:, None]
    tk = k != 0
    v = r - s - np.where(tk, s, 0.0) - 0.05 - np.where(tk, 0.05, 0.02) + f
    return np.where((k == 9) | (k < 0), np.nan, v)


def boot(d, dayn, span, B=3000, block=10, seed=7):
    lo = span[0] // DAY; nd = span[1] // DAY - lo
    di = (dayn - lo).astype(int)
    s = np.bincount(di, weights=d, minlength=nd); c = np.bincount(di, minlength=nd).astype(float)
    rng = np.random.default_rng(seed); nblk = int(np.ceil(nd / block))
    st = rng.integers(0, nd - block + 1, (B, nblk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :nd]
    m = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def risk(v, xm, span):
    lo, hi = span[0] // DAY, span[1] // DAY
    s = pd.Series(v * X_EQ / 100).groupby(xm // DAY).sum()
    full = s.reindex(np.arange(lo, max(hi, int(s.index.max()) + 1)), fill_value=0.0).values
    eq = np.cumprod(1 + full)
    if (eq <= 0).any():
        return -100.0, -100.0, full.min() * 100
    return (eq[-1] ** (1 / len(full)) - 1) * 100, (eq / np.maximum.accumulate(eq) - 1).min() * 100, full.min() * 100


def half_of(T):
    out = np.empty(len(T), object)
    for h, a, b in HALVES:
        out[(T >= TS(a)) & (T < TS(b))] = h
    return out


def main():
    S = pd.read_pickle(os.path.join(R, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl"))
    Z = dict(np.load(os.path.join(D, "sim_BT.npz"), allow_pickle=False))
    assert (Z["ix"] == np.arange(len(S))).all()
    names = [str(x) for x in Z["names"]]; nC = len(names)
    L = np.load(os.path.join(R, "reports", "DIST_EXIT_data", "sim_L1.npz"))
    zi = S.zi.values
    # ---- parity
    pr = np.abs(Z["r"][:, 0] - L["r"][zi, 0]) <= 0.01; pk = Z["k"][:, 0] == L["k"][zi, 0]
    par = pr & pk
    slip = S.slip.values.astype(float)
    Nn = nets(slip, Z["r"].astype(float), Z["k"], Z["f"].astype(float))
    ref_dist = nets(slip, L["r"][zi, :1].astype(float), L["k"][zi, :1], L["f"][zi, :1].astype(float))[:, 0]
    print(f"PARITY: kind+gross match {par.sum()}/{len(par)} = {par.mean() * 100:.2f} % · net |Δ|>0.01 on "
          f"{int((np.abs(Nn[:, 0] - ref_dist) > 0.01).sum())} fills · REF mean net {np.nanmean(Nn[:, 0]):+.3f} vs DIST_EXIT {np.nanmean(ref_dist):+.3f}")
    if par.mean() < 0.999:
        print("PARITY FAIL — stop"); sys.exit(1)
    bad = np.nonzero(~par)[0]
    for i in bad:
        print("  mismatch", S.pair.iloc[i], pd.Timestamp(int(S.entry_ms.iloc[i]), unit="ms"), Z["r"][i, 0], L["r"][zi[i], 0], Z["k"][i, 0], L["k"][zi[i], 0])
    N2 = nets(slip, Z["r"].astype(float), Z["k"], Z["f"].astype(float), flat=0.20)
    T = S.entry_ms.values.astype(np.int64); dayn = T // DAY
    per = np.where(T < SPAN["DESIGN"][1], "DESIGN", "HOLDOUT")
    hv = half_of(T); pair = S.pair.astype(str).values
    ok_all = np.isfinite(Nn).all(1)
    print("fills with a finite net in every cell:", int(ok_all.sum()), "of", len(S), "(censored dropped from every cell)")
    rows = []; halves = []
    for j, nm in enumerate(names):
        fam = "REF" if nm == "REF" else nm.split("_")[0]
        for p in ("DESIGN", "HOLDOUT", "ALL"):
            m = ok_all & ((per == p) if p != "ALL" else True)
            sp = SPAN[p] if p != "ALL" else (SPAN["DESIGN"][0], SPAN["HOLDOUT"][1])
            v = Nn[m, j]; v0 = Nn[m, 0]; d = v - v0; d2 = N2[m, j] - N2[m, 0]
            lo, hi = boot(d, dayn[m], sp)
            hp = pd.Series(d).groupby(hv[m]).sum()
            win = v0 > 0
            ps = pd.Series(d).groupby(pair[m]).sum(); ds = pd.Series(d).groupby(dayn[m]).sum(); tot = d.sum()
            dc0, dd0, wd0 = risk(v0, Z["xm"][m, 0], sp); dc, dd, wd = risk(v, Z["xm"][m, j], sp)
            ch = np.abs(d) > 1e-9
            # leave-one-pair-out: drop the pair carrying the largest positive share
            lopo = (tot - ps.max()) / max(m.sum() - (pair[m] == ps.idxmax()).sum(), 1) if len(ps) else np.nan
            rows.append(dict(cell=nm, fam=fam, period=p, N=int(m.sum()), changed=int(ch.sum()), WR=(v > 0).mean() * 100, avg=v.mean(),
                             avg_loss=v[v <= 0].mean(), d_mean=d.mean(), ci_lo=lo, ci_hi=hi, d_sum=tot,
                             halves_pos=f"{int((hp > 0).sum())}/{len(hp)}", halves_pos_frac=(hp > 0).mean(), d_mean_020=d2.mean(),
                             top_pair=ps.idxmax() if tot > 0 else "", top_pair_share=(ps.max() / tot * 100) if tot > 0 else np.nan,
                             top_day_share=(ds.max() / tot * 100) if tot > 0 else np.nan, lopo_d=lopo,
                             saved_usd=d[~win & (d > 0)].sum() * USD / 100, cut_usd=-d[win & (d < 0)].sum() * USD / 100,
                             usd_sum=v.sum() * USD / 100, dcr_ref=dc0, dcr=dc, maxdd_ref=dd0, maxdd=dd, worst_ref=wd0, worst=wd,
                             med_stop_changed=float(np.nanmedian(Z["slp"][m, j][ch])) if ch.any() else np.nan,
                             tp=(Z["k"][m, j] == 0).mean() * 100, sl=(Z["k"][m, j] == 1).mean() * 100, time=(Z["k"][m, j] == 2).mean() * 100,
                             rule=(Z["k"][m, j] == 5).mean() * 100))
        for h_, _, _ in HALVES:
            m = ok_all & (hv == h_)
            halves.append(dict(cell=nm, half=h_, N=int(m.sum()), ref_avg=Nn[m, 0].mean(), d_mean=(Nn[m, j] - Nn[m, 0]).mean()))
    E = pd.DataFrame(rows); E.to_csv(os.path.join(D, "eval.csv"), index=False)
    Hh = pd.DataFrame(halves); Hh.to_csv(os.path.join(D, "halves.csv"), index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 200)
    cols = ["cell", "N", "changed", "WR", "avg", "d_mean", "ci_lo", "ci_hi", "halves_pos", "d_mean_020", "saved_usd", "cut_usd", "dcr_ref", "dcr",
            "maxdd", "worst", "med_stop_changed", "tp", "sl", "rule"]
    for p in ("DESIGN", "HOLDOUT", "ALL"):
        print(f"\n===== {p}"); print(E[E.period == p][cols].round(2).to_string(index=False))
    print("\n===== PICKS (DESIGN-best per family) → HOLDOUT gates ①–⑦")
    W = E[(E.period == "HOLDOUT") & (E.fam == "W")].set_index("cell")
    wcap = {"W_SL8": 8, "W_SL10": 10, "W_SL12": 12, "W_SL15": 15}
    P = []
    for fam in FAMS:
        dsg = E[(E.period == "DESIGN") & (E.fam == fam)].sort_values("d_mean", ascending=False).iloc[0]
        h = E[(E.period == "HOLDOUT") & (E.cell == dsg.cell)].iloc[0]
        g = [h.d_mean > 0 and h.ci_lo > 0, h.halves_pos_frac >= 2 / 3, h.d_mean_020 > 0,
             (h.top_pair_share < 50 and h.top_day_share < 50) if h.d_sum > 0 else False, h.saved_usd > h.cut_usd, h.dcr >= h.dcr_ref]
        ctrl = ""
        if fam in ("C", "R"):
            ms = h.med_stop_changed if dsg.cell != "R_DH" else E[(E.period == "HOLDOUT") & (E.cell == "C_DH0")].med_stop_changed.iloc[0]
            wc = min(wcap, key=lambda c: abs(wcap[c] - ms)) if np.isfinite(ms) else "W_SL10"
            g.append(h.d_mean > W.loc[wc, "d_mean"]); ctrl = f" · ⑦ vs {wc} (Δ {W.loc[wc, 'd_mean']:+.3f}, med stop {ms:.1f} %)"
        verdict = "PASS" if all(g) else "FAIL"
        P.append(dict(fam=fam, pick=dsg.cell, design_d=dsg.d_mean, hold_d=h.d_mean, ci_lo=h.ci_lo, ci_hi=h.ci_hi, halves=h.halves_pos,
                      d020=h.d_mean_020, pair_share=h.top_pair_share, day_share=h.top_day_share, saved=h.saved_usd, cut=h.cut_usd,
                      dcr_ref=h.dcr_ref, dcr=h.dcr, maxdd_ref=h.maxdd_ref, maxdd=h.maxdd, worst_ref=h.worst_ref, worst=h.worst,
                      avg_loss_ref=E[(E.period == "HOLDOUT") & (E.cell == "REF")].avg_loss.iloc[0], avg_loss=h.avg_loss, lopo=h.lopo_d,
                      gates="".join("✓" if x else "✗" for x in g), verdict=verdict))
        print(f"{fam}: pick {dsg.cell} (DESIGN Δ {dsg.d_mean:+.3f}) · HOLDOUT Δ {h.d_mean:+.3f} [{h.ci_lo:+.3f}, {h.ci_hi:+.3f}] halves {h.halves_pos} "
              f"@0.20 {h.d_mean_020:+.3f} pair {h.top_pair_share:.0f}% day {h.top_day_share:.0f}% saved ${h.saved_usd:,.0f} cut ${h.cut_usd:,.0f} "
              f"DCR {h.dcr_ref:.2f}→{h.dcr:.2f}{ctrl} · gates {''.join('✓' if x else '✗' for x in g)} → {verdict}")
    pd.DataFrame(P).to_csv(os.path.join(D, "picks.csv"), index=False)
    # per-fill
    PF = pd.DataFrame(dict(pair=pair, entry=pd.to_datetime(T, unit="ms"), period=per, half=hv, slip=slip))
    for j, nm in enumerate(names):
        PF[nm] = np.round(Nn[:, j], 3); PF[nm + "_k"] = Z["k"][:, j]
    for nm in ("DH", "DO", "DL", "PK", "base", "vwap", "E0"):
        PF[nm] = Z[nm]
    PF.to_csv(os.path.join(D, "perfill.csv.gz"), index=False)


if __name__ == "__main__":
    main()
