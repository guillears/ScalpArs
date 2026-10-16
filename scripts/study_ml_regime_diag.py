#!/usr/bin/env python3
"""ML REGIME study 2026-10-16 — POST-HOC diagnostics (declared post-hoc in the report; none of these can promote anything):
D1 era-level variable shifts vs within-era slopes ("could this variable explain the era gap?")
D2 fill-composition regression (does the coin / entry mix explain the era gap?)
D3 BTC-after-entry anatomy: y by btc_fwd15 bucket per era; unconditional BTC 15-min drift per era vs after ML entries
D4 Q3 effects unconditional (all fills) vs inside the splits, 2,000 shuffles
D5 dose-response quintiles of the leading day variables per era + master
Output appended to reports/ml_regime/diag.md
    venv/bin/python -W ignore scripts/study_ml_regime_diag.py
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_ml_regime_main as S   # noqa: E402

OUT = os.path.join(ROOT, "reports", "ml_regime")
L = []


def P(*a):
    s = " ".join(str(x) for x in a); L.append(s); print(s, flush=True)


def main():
    R, M, Fz = S.load(); F = S.feats()
    ERAS, EA = S.ERAS, S.EA
    V = [c for c in F.columns if c != "day_ms" and F[c].dtype != object]
    # ── D1
    P("## D1 · Could a day variable explain the ERA gap? (era means on fill days vs the within-era slope)\n")
    P("Predicted H2−2025 gap = pooled within-era OLS slope of the day outcome on the variable × (H2 mean − 2025 mean). Actual gap +0.172 %/fill.\n")
    rows = []
    g = R.groupby(["era", "day"]).y.mean().reset_index()
    g["yd"] = g.y - g.groupby("era").y.transform("mean")
    for v in V:
        x = F.reindex(pd.DatetimeIndex(g.day))[v].values
        m = np.isfinite(x)
        if m.sum() < 50:
            continue
        xd = x - pd.Series(x).groupby(g.era.values).transform("mean").values
        mm = m & np.isfinite(xd)
        b = np.polyfit(xd[mm], g.yd.values[mm], 1)[0]
        mu = {e: np.nanmean(x[(g.era == e).values]) for e in ERAS}
        sd = np.nanstd(x)
        pred = b * (mu[ERAS[2]] - mu[ERAS[0]]); pred1 = b * (mu[ERAS[1]] - mu[ERAS[0]])
        rows.append(dict(var=v, m25=mu[ERAS[0]], mH1=mu[ERAS[1]], mH2=mu[ERAS[2]], shift_sd=(mu[ERAS[2]] - mu[ERAS[0]]) / sd,
                         slope=b, pred_H2=pred, pred_H1=pred1))
    T = pd.DataFrame(rows).sort_values("pred_H2", ascending=False)
    P("| var | mean 2025 / H1 / H2 | H2−2025 shift (sd) | predicted H1−2025 / H2−2025 gap (actual +0.102 / +0.172) |")
    P("|---|---|---|---|")
    for r in pd.concat([T.head(8), T.tail(3)]).itertuples():
        P(f"| {r.var} | {r.m25:.3g} / {r.mH1:.3g} / {r.mH2:.3g} | {r.shift_sd:+.2f} | {r.pred_H1:+.3f} / {r.pred_H2:+.3f} |")
    P(f"\n- Largest predicted share of the H2−2025 gap from any single variable: {T.pred_H2.max() / 0.172 * 100:.0f} % ({T.iloc[0]['var']}).")
    T.to_csv(os.path.join(OUT, "diag_era_shift.csv"), index=False)
    # ── D2
    P("\n## D2 · Does the fill mix explain the era gap? (OLS y ~ era + entry/coin covariates)\n")
    X = pd.DataFrame(index=R.index)
    num = {"atr": "entry_atr_pct", "adx": "entry_adx", "rsi": "entry_rsi", "age": "entry_pair_age_days", "volr": "entry_pair_volume_ratio",
           "gvol": "entry_global_volume_ratio", "rank": "entry_pair_rank", "gap": "entry_gap", "slope": "entry_ema20_slope",
           "btcadx": "entry_btc_adx", "btcrsi": "entry_btc_rsi", "fund": "entry_funding_rate", "rng": "entry_range_position"}
    for k, c in num.items():
        x = pd.to_numeric(R[c], errors="coerce")
        X[k] = x.fillna(x.median()); X[k + "_na"] = x.isna().astype(float)
    X["logvol24"] = np.log1p(pd.to_numeric(R.entry_pair_volume_24h_usd, errors="coerce")).fillna(0)
    X["DT"] = R.DT.fillna(False).astype(float); X["btc24dn"] = (R.k_btc_ret24h < 0).astype(float)
    X["tick"] = R.tick.astype(float); X["top50"] = R.top50.astype(float)
    for h in range(0, 24, 4):
        X[f"h{h}"] = ((R.hour // 4 * 4) == h).astype(float)
    X = X.loc[:, X.std() > 0]
    Xs = (X - X.mean()) / X.std()
    E1 = (R.era == ERAS[1]).astype(float); E2 = (R.era == ERAS[2]).astype(float)
    def ols(A):
        A = np.column_stack([np.ones(len(R))] + A)
        beta, *_ = np.linalg.lstsq(A, R.y.values, rcond=None)
        return beta
    b0 = ols([E1.values, E2.values]); b1 = ols([E1.values, E2.values, Xs.values])
    P(f"- Era gaps vs 2025, raw: H1 {b0[1]:+.3f} · H2 {b0[2]:+.3f}. After controlling {Xs.shape[1]} entry/coin covariates "
      f"(ATR, ADX, RSI, coin age, volume ratios, rank, 24h volume, gap, slopes, BTC ADX/RSI, funding, range position, DT, BTC24H, pricing, top-50, hour block): "
      f"H1 {b1[1]:+.3f} · H2 {b1[2]:+.3f}.")
    P(f"- Share of the H2−2025 gap explained by the fill mix: {100 * (1 - b1[2] / b0[2]):.0f} %; of H1−2025: {100 * (1 - b1[1] / b0[1]):.0f} %.")
    comp = pd.DataFrame({e: X[R.era == e].mean() for e in ERAS})
    comp["d_sd"] = (comp[ERAS[2]] - comp[ERAS[0]]) / X.std()
    P("\nBiggest mix shifts (H2 vs 2025, in sd): " + ", ".join(f"{k} {v:+.2f}" for k, v in comp.d_sd.abs().sort_values(ascending=False).head(6).items()
                                                                 if True))
    # ── D3
    P("\n## D3 · The BTC 15 minutes after entry, by era\n")
    q = R.loc[R.era == ERAS[0], "btc_fwd15"].quantile([1 / 3, 2 / 3]).values
    R["bf"] = np.where(R.btc_fwd15 <= q[0], "BTC down", np.where(R.btc_fwd15 <= q[1], "BTC flat", "BTC up"))
    P(f"Buckets frozen on 2025 terciles of btc_fwd15 (≤ {q[0]:+.3f} % · ≤ {q[1]:+.3f} % · above).\n")
    P("| bucket | 2025 share · avg · WR | H1 share · avg · WR | H2 share · avg · WR | master share · avg |")
    P("|---|---|---|---|---|")
    import pandas as _pd
    Mf = _pd.read_csv(os.path.join(ROOT, "reports/ML_BTCFWD_fill_features.csv"), usecols=["blk", "y", "btc_fwd15"])
    Mf = Mf[Mf.blk == "master"]
    Mf["bf"] = np.where(Mf.btc_fwd15 <= q[0], "BTC down", np.where(Mf.btc_fwd15 <= q[1], "BTC flat", "BTC up"))
    for b in ("BTC down", "BTC flat", "BTC up"):
        cells = []
        for e in ERAS:
            d = R[R.era == e]; s = d[d.bf == b]
            cells.append(f"{100 * len(s) / len(d):.0f} % · {s.y.mean():+.3f} · {100 * s.win.mean():.0f} %")
        s = Mf[Mf.bf == b]
        cells.append(f"{100 * len(s) / len(Mf):.0f} % · {s.y.mean():+.3f}")
        P(f"| {b} | " + " | ".join(cells) + " |")
    # unconditional BTC 15-min drift per era (all 5-min moments) vs after ML entries
    b5 = S_btc5()
    P("\n| era | BTC 15-min forward drift, all 5-min moments (bps, share up) | after ML entries (bps, share up) | BTC 15-min AC (lag 1, era) |")
    P("|---|---|---|---|")
    for e in ERAS:
        d = R[R.era == e]
        t0, t1 = d.ts.min().floor("D"), d.ts.max().ceil("D")
        s = b5[(b5.index >= t0) & (b5.index < t1)]
        f15 = (s.shift(-3) / s - 1).dropna() * 1e4
        r15 = np.log(s.iloc[::3]).diff().dropna()
        ac = np.corrcoef(r15.values[1:], r15.values[:-1])[0, 1]
        P(f"| {EA[e]} | {f15.mean():+.2f} · {100 * (f15 > 0).mean():.1f} % | {1e2 * d.btc_fwd15.mean():+.2f} · {100 * (d.btc_fwd15 > 0).mean():.1f} % | {ac:+.3f} |")
    # ── D4
    P("\n## D4 · Q3 entry effects: unconditional vs inside the splits (2,000 within-7-day-block shuffles)\n")
    rng = np.random.default_rng(77)
    adx = pd.to_numeric(R.entry_adx, errors="coerce"); adxp = pd.to_numeric(R.entry_adx_prev, errors="coerce")
    DT = R.DT.fillna(False).astype(bool)
    E = {"E1 ADX 20–30 inside DT": (adx.between(20, 30, inclusive="left"), DT),
         "E2 PAIR_1H_DOWNTREND": (DT, pd.Series(True, index=R.index)),
         "E3 BTC24H_DOWN": (R.k_btc_ret24h < 0, R.k_btc_ret24h.notna()),
         "E4 ADX falling": (adx < adxp, adx.notna() & adxp.notna())}
    splits = {"all fills": pd.Series(True, index=R.index)}
    for v, cut, hi in (("spk24", 11.0, True), ("btc_vr3_7", 1.0, True)):
        lab = S.day_lab(F, v, cut, hi, R.day)
        splits[f"{v} good"] = pd.Series(lab == 1, index=R.index); splits[f"{v} bad"] = pd.Series(lab == 0, index=R.index)
    P("| effect | subset | 2025 Δ (N in) | H1 Δ | H2 Δ | pooled Δ · p (2,000) |"); P("|---|---|---|---|---|---|")
    for nm, (flag, base) in E.items():
        for sn, sub in splits.items():
            ds, ns, nls = [], [], []
            for e in ERAS:
                s = R[(R.era == e) & base & sub]
                f = flag[s.index].values.astype(bool); y = s.y.values
                if f.sum() < 5 or (~f).sum() < 5:
                    ds.append(np.nan); ns.append(int(f.sum())); nls.append(None); continue
                ds.append(y[f].mean() - y[~f].mean()); ns.append(int(f.sum()))
                blk = ((s.day - s.day.min()).dt.days // 7).values
                groups = [np.where(blk == b)[0] for b in np.unique(blk)]
                nl = np.empty(2000)
                for k in range(2000):
                    yy = y.copy()
                    for ii in groups:
                        yy[ii] = yy[rng.permutation(ii)]
                    nl[k] = yy[f].mean() - yy[~f].mean()
                nls.append(nl)
            if all(x is not None for x in nls):
                w = np.array(ns, float) / sum(ns); pob = float(np.dot(w, ds)); pnl = sum(w[i] * nls[i] for i in range(3))
                pp = (1 + np.sum(np.abs(pnl) >= abs(pob))) / 2001
            else:
                pob = pp = np.nan
            P(f"| {nm} | {sn} | {S.f3(ds[0])} ({ns[0]}) | {S.f3(ds[1])} ({ns[1]}) | {S.f3(ds[2])} ({ns[2]}) | {S.f3(pob)} · {pp:.4f} |")
    # ── D5
    P("\n## D5 · Dose-response (day-variable quintiles, cuts frozen on 2025 fill days)\n")
    for v in ("btc_vr3_7", "btc_ft_7", "off30", "spk24", "btc_vr12_7", "disp7"):
        x25 = F.reindex(pd.DatetimeIndex(R[R.era == ERAS[0]].day))[v].dropna()
        qs = x25.quantile([.2, .4, .6, .8]).values
        P(f"**{v}** (quintile cuts {', '.join(f'{c:.3g}' for c in qs)}) — avg %/fill (N):\n")
        P("| quintile | 2025 | H1 | H2 | master |"); P("|---|---|---|---|---|")
        blocks = [(e, R[R.era == e]) for e in ERAS] + [("master", M)]
        cols = []
        for nm, d in blocks:
            x = F.reindex(pd.DatetimeIndex(d.day))[v].values
            qi = np.searchsorted(qs, x, side="right"); qi = np.where(np.isfinite(x), qi, -1)
            cols.append([(d.y.values[qi == k].mean() if (qi == k).any() else np.nan, int((qi == k).sum())) for k in range(5)])
        for k in range(5):
            P(f"| Q{k + 1} | " + " | ".join(f"{S.f3(c[k][0])} ({c[k][1]})" for c in cols) + " |")
        P("")
    open(os.path.join(OUT, "diag.md"), "w").write("\n".join(L) + "\n")


def S_btc5():
    import study_ml_regime_features as FE
    k = FE.load5("BTCUSDT")
    s = pd.Series(k.c.values.astype(float), index=pd.to_datetime(k.t.values, unit="ms"))
    return s.asfreq("5min").ffill()


if __name__ == "__main__":
    main()
