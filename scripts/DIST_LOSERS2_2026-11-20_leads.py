"""DIST_LOSERS2 (2026-11-20): leads from the screen, priced against the locked filter EXPECTANCY bar,
plus the 1D screen split by BTC daily regime (HOLDOUT bull vs bear, sign/tercile, shuffled null)."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "reports", "DIST_LOSERS2_2026-11-20_data")
d = pd.read_pickle(os.path.join(ROOT, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl")).reset_index(drop=True)
d["win"] = d.net > 0
d["w4h"] = (d.entry_ms // (4 * 3600_000))
des = d.block == "DESIGN"
e_vol = np.quantile(d.loc[des, "vol24"], 1 / 3)
LEADS = {
    "thin coin (vol24 DESIGN T1 < %.1fM)" % (e_vol / 1e6): d.vol24 < e_vol,
    "top-trader 1h pos change >= 0": d.top_pos_chg1h >= 0,
    "top-trader 1h pos change < 0": d.top_pos_chg1h < 0,
    "oi_chg30m DESIGN T2 [9.1, 24.6)": (d.oi_chg30m >= 9.13) & (d.oi_chg30m < 24.56),
    "small 15m range (DESIGN T1 < 5.0 %)": d.range15_pct < 5.001,
}
rng = np.random.default_rng(1)

def be_wr(x):
    w, l = x[x > 0], x[x <= 0]
    return abs(l.mean()) / (w.mean() + abs(l.mean())) if len(w) and len(l) else np.nan

def boot_hi(x, win, B=4000):
    u = np.unique(win); g = {k: x[win == k] for k in u}; m = []
    for _ in range(B):
        s = rng.choice(u, len(u)); v = np.concatenate([g[k] for k in s]); m.append(v.mean())
    return np.quantile(m, 0.95)

rows = []
for name, m in LEADS.items():
    for blk, bm in (("DESIGN", des), ("HOLDOUT", ~des), ("2026", d.half.str.startswith("26")), ("ALL", np.ones(len(d), bool))):
        c = d[m & bm]; r = d[~m & bm]
        if len(c) < 5: continue
        x = c.net.values
        byp = c[c.net < 0].groupby("pair").net.sum()   # each pair's GROSS loss
        loss = -x[x < 0].sum()
        rows.append(dict(lead=name, block=blk, N=len(c), windows=c.w4h.nunique(), WR=round(100 * c.win.mean()),
                         BE_WR_sleeve=round(100 * be_wr(d[bm].net.values)), avg=round(x.mean(), 2),
                         rest_avg=round(r.net.mean(), 2), avg_95hi=round(boot_hi(x, c.w4h.values), 2),
                         top_pair_loss_share=round(100 * (-byp.min() / loss) if loss > 0 and byp.min() < 0 else 0)))
    for h, g in d[m].groupby("half"):
        rows.append(dict(lead=name, block="half " + h, N=len(g), avg=round(g.net.mean(), 2),
                         rest_avg=round(d[~m & (d.half == h)].net.mean(), 2), WR=round(100 * g.win.mean())))
L = pd.DataFrame(rows); L.to_csv(os.path.join(OUT, "leads.csv"), index=False)
pd.set_option("display.width", 250); print(L.to_string(index=False))

# regime-split 1D screen (HOLDOUT; DESIGN has 1 bear fill)
import json
vars_ = json.load(open(os.path.join(OUT, "screen_summary.json")))["vars"]
for reg in ("bull", "bear"):
    sub = d[(~des) & (d.btc_d1_regime == (1 if reg == "bull" else -1))] if d.btc_d1_regime.dtype != object else None
    if sub is None or len(sub) == 0:
        print("regime col values", d.btc_d1_regime.unique()[:5]); break
    y = sub.net.values; cells = {}
    for v in vars_:
        s = pd.to_numeric(sub[v], errors="coerce").values
        if np.isnan(s).mean() > 0.4 or len(np.unique(s[~np.isnan(s)])) < 3: continue
        if np.nanmin(s) < 0 < np.nanmax(s): cells[f"{v}<0"] = s < 0
        e = np.nanquantile(s, [1 / 3, 2 / 3])
        for k, mm in enumerate((s < e[0], (s >= e[0]) & (s < e[1]), s >= e[1])): cells[f"{v}:T{k+1}"] = mm
    def tmax(yy):
        out = {}
        for n_, mm in cells.items():
            a, b = yy[mm], yy[~mm]
            if len(a) < 15 or len(b) < 15: continue
            out[n_] = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return out
    t = tmax(y); null = [max(abs(v) for v in tmax(rng.permutation(y)).values()) for _ in range(200)]
    top = sorted(t.items(), key=lambda kv: -abs(kv[1]))[:8]
    print(f"\nHOLDOUT BTC-daily {reg}: N={len(sub)} tests={len(t)} null95={np.quantile(null, .95):.2f}")
    for n_, tv in top:
        mm = cells[n_]; print(f"  {n_:40s} t={tv:+.2f} n={mm.sum():4d} avg={y[mm].mean():+.2f} rest={y[~mm].mean():+.2f}")

# 15m RSI x BTC daily regime on FROZEN DESIGN terciles (the report's §5 table)
e = np.quantile(d.loc[des, "rsi15"].dropna(), [1 / 3, 2 / 3])
d["rsi15_T"] = np.digitize(d.rsi15, e) + 1
d["reg"] = np.where(des, "DESIGN", np.where(d.btc_d1_regime > 0, "HOLDOUT bull", "HOLDOUT bear"))
tab = d.groupby(["reg", "rsi15_T"]).net.agg(["size", "mean"]).round(2).unstack("rsi15_T")
print("\nrsi15 DESIGN edges", e.round(2)); print(tab.to_string())
tab.to_csv(os.path.join(OUT, "rsi15_regime.csv"))
