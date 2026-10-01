#!/usr/bin/env python3
"""Winners vs losers — ALL variables (operator Sep-28: "what about BTC EMA50/EMA200 and its gap? all the hundreds of variables").

Universe per sleeve = every numeric entry_* stamp shared by master + backtest (+ derived deltas) + the ~800 rebuilt features of
scripts/entry_feature_factory.py (BTC / ETH / BTCDOM / PAIR × 5m-15m-1h-4h-1d × EMA9-200 levels & gaps & slopes, RSI, ADX/DI,
ATR, returns, range, vol, volume + pair-vs-BTC relative).
CANDIDATE = winners-vs-losers points the SAME way in master, backtest H1 and backtest H2 (out of sample), |AUC−0.5| ≥ --gap-m in
master and ≥ --gap-b in each backtest half. The count is repeated with master labels SHUFFLED (--perm) → what luck produces.
Each candidate: one loser-side cut frozen on backtest H1 → H2 OOS + master per batch before→after (N·WR·net $).
Usage: venv/bin/python scripts/winners_losers_allfeatures.py --fills <year fills csv> [--sleeves MOM-long] [--split 2026-05-01]
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--fills", required=True)
ap.add_argument("--sleeves", default="MOM-long,MOM-short,FLIP-short,Spike-Fade")
ap.add_argument("--split", default="2026-05-01")
ap.add_argument("--gap-m", type=float, default=0.10)
ap.add_argument("--gap-b", type=float, default=0.04)
ap.add_argument("--perm", type=int, default=100)
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--min-block", type=int, default=30)
ap.add_argument("--out", default="reports/WINNERS_LOSERS_ALLFEATURES.csv")
A = ap.parse_args()
_argv, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG
import entry_feature_factory as EF
M = LG.build()
sys.argv = _argv


def sleeve_of(strat, direction):
    s = strat if isinstance(strat, str) and strat else "MOMENTUM"
    if s.startswith("FLIP"):
        return "FLIP-short"
    if s == "MOMENTUM":
        return "MOM-long" if direction == "LONG" else "MOM-short"
    return {"SPIKE_FADE": "Spike-Fade", "BULLRUN_LONG": "BullRun-Long", "BEARRUN_SHORT": "BearRun-Short"}.get(s, s)


SKIP = ("entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional", "entry_liquidity_cap_notional")


def stamped(d):
    n = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
    S = {c: n(c) for c in d.columns if c.startswith("entry_") and c not in SKIP}
    S["d_rsi"] = n("entry_rsi") - n("entry_rsi_prev")
    S["d_adx"] = n("entry_adx") - n("entry_adx_prev")
    S["d_btc_rsi"] = n("entry_btc_rsi") - n("entry_btc_rsi_prev")
    S["d_btc_rsi_30m"] = n("entry_btc_rsi") - n("entry_btc_rsi_prev6")
    S["d_btc_adx"] = n("entry_btc_adx") - n("entry_btc_adx_prev")
    S["d_btc_rsi_1h"] = n("entry_btc_rsi_1h") - n("entry_btc_rsi_1h_prev")
    S["log_pair_vol_usd"] = np.log10(n("entry_pair_volume_24h_usd").where(lambda x: x > 0))
    return pd.DataFrame({("STAMP_" + k): v for k, v in S.items()}, index=d.index)


def auc_cols(X, y):
    """Vectorised AUC per column (NaN-aware). y bool. Returns Series (NaN if <4 on either side)."""
    R = X.rank(axis=0).values
    ok = X.notna().values
    yy = np.broadcast_to(y.values[:, None], ok.shape)
    n1 = (ok & yy).sum(axis=0)
    n0 = (ok & ~yy).sum(axis=0)
    s1 = np.where(ok & yy, R, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = pd.Series((s1 - n1 * (n1 + 1) / 2) / (n1 * n0), index=X.columns)
    n1, n0 = pd.Series(n1, index=X.columns), pd.Series(n0, index=X.columns)
    return a.where((n1 >= 4) & (n0 >= 4))


M["sleeve"] = [sleeve_of(s, x) for s, x in zip(M.entry_strategy, M.direction)]
M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"),
                    M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
F = pd.read_csv(A.fills, low_memory=False)
F = F[~F.sleeve.astype(str).str.contains("PROBE|Probe", na=False)].reset_index(drop=True)
F = F.copy()
F["half"] = np.where(F.opened_at.astype(str) < A.split, "H1", "H2")
sleeves = [s.strip() for s in A.sleeves.split(",") if s.strip()]
M = M[M.sleeve.isin(sleeves)].reset_index(drop=True)
F = F[F.sleeve.isin(sleeves)].reset_index(drop=True)
print(f"building features: master {len(M)} fills, backtest {len(F)} fills …", flush=True)
XM = pd.concat([stamped(M), EF.features(M)], axis=1)
XF = pd.concat([stamped(F), EF.features(F)], axis=1)
cols = [c for c in XM.columns if c in XF.columns and XM[c].notna().mean() > 0.7 and XF[c].notna().mean() > 0.7
        and XF[c].nunique() > 5]
XM, XF = XM[cols].astype(float), XF[cols].astype(float)
ERAS = [e for e in LG.ERAS if e in set(M.era)]
rng = np.random.default_rng(7)
allrows = []


def fmt(g, col):
    return f"{len(g):4d}·{(g[col] > 0).mean() * 100 if len(g) else 0:3.0f}%·{g[col].mean() if len(g) else 0:+.3f}"


for sl in sleeves:
    mi, fi = M.index[M.sleeve == sl], F.index[F.sleeve == sl]
    m, f = M.loc[mi], F.loc[fi]
    if len(m) < 10 or len(f) < 100:
        print(f"\n## {sl}: too few (master {len(m)}, backtest {len(f)})")
        continue
    ym, yf = (m.pct > 0), (f.pnl_percentage > 0)
    h1, h2 = (f.half == "H1").values, (f.half == "H2").values
    am = auc_cols(XM.loc[mi], ym)
    a1 = auc_cols(XF.loc[fi][h1], yf[h1])
    a2 = auc_cols(XF.loc[fi][h2], yf[h2])

    def passing(am_):
        s = np.sign(am_ - .5)
        return ((s == np.sign(a1 - .5)) & (s == np.sign(a2 - .5)) & ((am_ - .5).abs() >= A.gap_m)
                & ((a1 - .5).abs() >= A.gap_b) & ((a2 - .5).abs() >= A.gap_b))

    ok = passing(am)
    null = [int(passing(auc_cols(XM.loc[mi], pd.Series(rng.permutation(ym.values), index=ym.index))).sum())
            for _ in range(A.perm)]
    nl = len(m) - int(ym.sum())
    print(f"\n## {sl} — master {fmt(m, 'pct')} ({nl} losers) · backtest H1 {fmt(f[h1], 'pnl_percentage')} · "
          f"H2 {fmt(f[h2], 'pnl_percentage')} · variables tested {am.notna().sum()}")
    print(f"   candidates (same direction master + bt H1 + bt H2): {int(ok.sum())}  |  shuffled-label null: "
          f"median {np.median(null):.0f}, 95th pct {np.percentile(null, 95):.0f}  → "
          f"{'MORE than luck' if ok.sum() > np.percentile(null, 95) else 'NOT more than luck'}")
    T = pd.DataFrame(dict(auc_master=am, auc_btH1=a1, auc_btH2=a2, pass_=ok))
    T["W_master"] = XM.loc[mi][ym.values].median()
    T["L_master"] = XM.loc[mi][~ym.values].median()
    T["W_bt"], T["L_bt"] = XF.loc[fi][yf.values].median(), XF.loc[fi][~yf.values].median()
    T["strength"] = pd.concat([(T.auc_master - .5).abs(), (T.auc_btH1 - .5).abs(), (T.auc_btH2 - .5).abs()], axis=1).min(axis=1)
    T["sleeve"] = sl
    allrows.append(T.reset_index().rename(columns={"index": "variable"}))
    C = T[T.pass_].sort_values("strength", ascending=False)
    print("   top master-only separators (for reference, backtest AUCs alongside):")
    print(T.assign(g=(T.auc_master - .5).abs()).sort_values("g", ascending=False).head(10)[
        ["auc_master", "auc_btH1", "auc_btH2", "W_master", "L_master"]].round(3).to_string())
    if not len(C):
        continue
    print("   CANDIDATES:")
    print(C.head(A.top)[["auc_master", "auc_btH1", "auc_btH2", "W_master", "L_master", "W_bt", "L_bt"]].round(3).to_string())
    xf, xm = XF.loc[fi], XM.loc[mi]
    for v in C.head(A.top).index:
        lo = T.loc[v, "auc_btH1"] > .5          # winners higher → losers low
        best = None
        hv = xf[v][h1].dropna()
        for q in np.linspace(.1, .4, 7):
            cut = hv.quantile(q if lo else 1 - q)
            blk = (xf[v] < cut) if lo else (xf[v] > cut)
            b = f[h1 & blk.values]
            if len(b) >= A.min_block and (best is None or b.pnl_percentage.mean() < best[1]):
                best = (cut, b.pnl_percentage.mean())
        if best is None:
            continue
        cut = best[0]
        bf = ((xf[v] < cut) if lo else (xf[v] > cut)).values
        bm = ((xm[v] < cut) if lo else (xm[v] > cut)).values
        print(f"\n   ▶ BLOCK {v} {'<' if lo else '>'} {cut:.4g}  (frozen on backtest H1)")
        print(f"     bt H1 blocked {fmt(f[h1 & bf], 'pnl_percentage')} kept {fmt(f[h1 & ~bf], 'pnl_percentage')} | "
              f"H2 OOS blocked {fmt(f[h2 & bf], 'pnl_percentage')} kept {fmt(f[h2 & ~bf], 'pnl_percentage')} | "
              f"days blocked {f[bf].opened_at.astype(str).str[:10].nunique()}")
        mb = m[bm]
        print(f"     master blocked {len(mb)}·{(mb.pct > 0).mean() * 100 if len(mb) else 0:.0f}%·avg {mb.pct.mean() if len(mb) else 0:+.3f}·${mb.net.sum() if len(mb) else 0:+,.0f}")
        cells = []
        for e in ERAS:
            g, gb = m[m.era.values == e], bm[m.era.values == e]
            if len(g):
                k = g[~gb]
                cells.append(f"{e} {len(g)}·{(g.net > 0).mean() * 100:.0f}%·${g.net.sum():+,.0f}→{len(k)}·"
                             f"{(k.net > 0).mean() * 100 if len(k) else 0:.0f}%·${k.net.sum():+,.0f}")
        print("     per batch before→after: " + " | ".join(cells))

if allrows:
    pd.concat(allrows).to_csv(A.out, index=False)
    print(f"\nfull table → {A.out}")
