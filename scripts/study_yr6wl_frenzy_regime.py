#!/usr/bin/env python3
"""YR6 WL — FRENZY regime-switch read (question 1): is each sleeve's 2025 loss a regime effect (a DAY-granularity on/off switch) or an
entry-level separator?

  venv/bin/python -I scripts/study_yr6wl_frenzy_regime.py   → reports/YR6_WL_FRENZY_regime.md (+ _regime_daily_panel.csv, _regime_tests.csv)

A. EX-ANTE DAILY PANEL (known at 00:00 UTC — every value uses closes up to the PRIOR day; cache-only k1d):
   BTC: ret1d, ret3d, ret7d, ret30d, off30d-high %, realized vol 7d / 30d (daily-return SD), close vs SMA20 / SMA50 / SMA200 (%), SMA20 5d slope.
   ALT breadth (all USDT pairs in reports/backtest_cache/k1d, ≥ 30 days of history that day): share above own SMA20, median 1d / 7d return,
   median 7d alt − BTC 7d ("alt season"), cross-sectional SD of 1d returns, share of pairs +20 % in 1d (pump count).
   SLEEVE trailing state (ex-ante): avg y of the sleeve's fills CLOSED in the prior 7 / 30 days (equity-curve switch).
   Unit = UTC day with ≥ 1 fill of the sleeve (seed 1); outcome = day-mean y. Each variable at sign / median / outer terciles (cuts from the
   discovery regime, frozen), Welch t over days, null = day-label shuffle (1,000), max |t| over the panel family → adjusted p.
   Discover 2025 BULL → confirm 2026-H2 (and reverse); bear H1 shown. Confirm = same sign, day-bootstrap P ≥ 0.90.
B. FILL-LEVEL re-read of the screen's confirmed macro STAMP separators (cut frozen at discovery, applied to each fill's own stamp = an
   implementable entry gate), DAY-clustered: zone vs rest per regime, top-50 split, per-pair / per-day loss concentration, expectancy bar.
C. MONTHLY: the 22 months' sleeve avg vs month-mean of each panel variable (Spearman), listing LITE's / WILLY's bad months.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_yr6wl_frenzy_common as C  # noqa: E402

K1D = os.path.join(C.REP, "backtest_cache/k1d")
K1D25 = os.path.join(C.REP, "backtest_cache_2025/k1d")
NPERM = 1000
out = []


def read_k1d(path, unit_scale=1):
    x = pd.read_csv(path)
    t = pd.to_numeric(x.open_time, errors="coerce") * unit_scale
    x["date"] = pd.to_datetime(t, unit="ms").dt.strftime("%Y-%m-%d")
    return x.set_index("date")[["o", "h", "l", "c", "qvol"]].astype(float)


def build_panel():
    b = pd.concat([read_k1d(os.path.join(K1D25, "BTCUSDT.csv"), 1_000_000), read_k1d(os.path.join(K1D, "BTCUSDT.csv"))])
    b = b[~b.index.duplicated(keep="last")].sort_index()
    c = b.c; r = c.pct_change() * 100
    P = pd.DataFrame(index=b.index)
    P["btc_ret1d"] = r
    P["btc_ret3d"] = c.pct_change(3) * 100
    P["btc_ret7d"] = c.pct_change(7) * 100
    P["btc_ret30d"] = c.pct_change(30) * 100
    P["btc_off30d_hi"] = (c / b.h.rolling(30).max() - 1) * 100
    P["btc_rv7"] = r.rolling(7).std()
    P["btc_rv30"] = r.rolling(30).std()
    for n in (20, 50, 200):
        P[f"btc_vs_sma{n}"] = (c / c.rolling(n).mean() - 1) * 100
    s20 = c.rolling(20).mean(); P["btc_sma20_slope5"] = (s20 / s20.shift(5) - 1) * 100
    # alt breadth
    closes = {}
    for f in os.listdir(K1D):
        if not f.endswith("USDT.csv") or f.startswith(("BTCUSDT", "USDC", "FDUSD")):
            continue
        try:
            x = read_k1d(os.path.join(K1D, f))
        except Exception:
            continue
        closes[f[:-4]] = x.c
    M = pd.DataFrame(closes).sort_index()
    M = M[~M.index.duplicated()]
    R1 = M.pct_change(fill_method=None) * 100
    R7 = M.pct_change(7, fill_method=None) * 100
    age = M.notna().cumsum()
    ok = age >= 30
    above = (M > M.rolling(20, min_periods=20).mean()).where(ok)
    P2 = pd.DataFrame(index=M.index)
    P2["alt_above_sma20"] = above.mean(1) * 100
    P2["alt_med_ret1d"] = R1.where(ok).median(1)
    P2["alt_med_ret7d"] = R7.where(ok).median(1)
    P2["alt_n"] = ok.sum(1)
    P2["alt_xsd_ret1d"] = R1.where(ok).std(1)
    P2["alt_pump20_share"] = (R1.where(ok) > 20).mean(1) * 100
    P = P.join(P2, how="left")
    P["alt_minus_btc_7d"] = P.alt_med_ret7d - P.btc_ret7d
    # EX-ANTE: the value usable on day d is the one computed from closes up to d-1
    P = P.shift(1)
    return P


P = build_panel()
P = P.reindex(sorted(set(P.index) | set(pd.date_range("2025-01-01", "2026-10-12", freq="D").strftime("%Y-%m-%d"))))
P.to_csv(os.path.join(C.REP, "YR6_WL_FRENZY_regime_daily_panel.csv"))
D = C.load()
D = D.sort_values("ts").reset_index(drop=True)
D["close_ts"] = pd.to_datetime(D.closed_at)

# sleeve trailing state (ex-ante): fills CLOSED before 00:00 of day d within the prior 7 / 30 days
for s in C.SLEEVES:
    S = D[D.sleeve == s]
    days = pd.date_range("2025-01-01", "2026-10-12", freq="D")
    cs = S.set_index("close_ts").sort_index().y
    for w in (7, 30):
        vals = []
        for d in days:
            seg = cs[(cs.index < d) & (cs.index >= d - pd.Timedelta(days=w))]
            vals.append(seg.mean() if len(seg) >= 5 else np.nan)
        P.loc[[d.strftime("%Y-%m-%d") for d in days], f"trail{w}_{s}"] = vals

PANEL_VARS = [c for c in P.columns if not c.startswith("trail") and c != "alt_n"]


def cuts(x):
    x = x.dropna(); o = []
    if len(x) < 10:
        return o
    if (x > 0).mean() > 0.05 and (x < 0).mean() > 0.05:
        o.append(("sign", ">", 0.0))
    o.append(("med", ">", float(x.median())))
    q1, q2 = x.quantile([1 / 3, 2 / 3]); o += [("lo3", "<=", float(q1)), ("hi3", ">", float(q2))]
    return o


def mk(x, op, v):
    x = np.asarray(x, float)
    return (x > v) if op == ">" else (x <= v)


def day_frame(S, cols):
    g = S.groupby("day").agg(y=("y", "mean"), n=("y", "size"))
    return g.join(P[cols], how="left")


def welch(m, y):
    return C.tstats(m.astype(float)[:, None], y[:, None])[0, 0]


def boot_p(m, y, sign, seed=3):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); d = []
    for _ in range(4000):
        b = rng.choice(idx, len(idx)); mb = m[b]
        if mb.sum() and (~mb).sum():
            d.append(y[b][mb].mean() - y[b][~mb].mean())
    return float((sign * np.array(d) > 0).mean())


rows = []
for s in C.SLEEVES:
    cols = PANEL_VARS + [f"trail7_{s}", f"trail30_{s}"]
    for disc, conf in (("2025 BULL", "2026-H2 BULL"), ("2026-H2 BULL", "2025 BULL")):
        G = day_frame(D[(D.sleeve == s) & (D.regime == disc)], cols)
        Gc = day_frame(D[(D.sleeve == s) & (D.regime == conf)], cols)
        Gb = day_frame(D[(D.sleeve == s) & (D.regime == "2026-H1 BEAR")], cols)
        y = G.y.values
        tests = []
        for c in cols:
            for g, op, v in cuts(G[c]):
                m = mk(G[c], op, v) & G[c].notna().values
                if m.sum() >= 8 and (G[c].notna().values & ~m).sum() >= 8:
                    tests.append((c, g, op, v, m))
        rng = np.random.default_rng(42)
        perms = np.column_stack([rng.permutation(len(y)) for _ in range(NPERM)])
        Yp = y[perms]
        mx = np.zeros(NPERM); T0 = []
        for c, g, op, v, m in tests:
            ok = G[c].notna().values
            Z = m[ok].astype(float)[:, None]
            T0.append(C.tstats(Z, y[ok][:, None])[0, 0])
            mx = np.maximum(mx, np.nan_to_num(np.abs(C.tstats(Z, Yp[ok, :])[0])))
        for (c, g, op, v, m), t in zip(tests, T0):
            ok = G[c].notna().values
            sign = np.sign(t)
            rec = dict(sleeve=s, disc=disc, var=c, gran=g, zone=f"{c} {op} {v:.4g}", days_zone=int(m.sum()), avg_zone=y[m].mean(),
                       avg_rest=y[ok & ~m].mean(), t=t, p_adj=float((mx >= abs(t)).mean()), null_q95=float(np.percentile(mx, 95)))
            for tag, H in (("conf", Gc), ("bear", Gb)):
                okh = H[c].notna().values; mh = mk(H[c], op, v) & okh
                if mh.sum() >= 3 and (okh & ~mh).sum() >= 3:
                    yh = H.y.values[okh]; mm = mh[okh]
                    rec[f"{tag}_days"] = int(mm.sum()); rec[f"{tag}_diff"] = yh[mm].mean() - yh[~mm].mean()
                    rec[f"{tag}_P"] = boot_p(mm, yh, sign) if tag == "conf" or True else np.nan
            rec["CONFIRMED"] = bool(rec.get("conf_diff") is not None and np.sign(rec.get("conf_diff", 0)) == sign and rec.get("conf_P", 0) >= 0.90)
            rows.append(rec)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(C.REP, "YR6_WL_FRENZY_regime_tests.csv"), index=False)

out.append("## A. Ex-ante daily panel (DAY units; prior-day closes) — per sleeve, discovery regime: survivors (p_adj < 0.10) and best 6\n")
for (s, disc), g in T.groupby(["sleeve", "disc"]):
    g = g.reindex(g.t.abs().sort_values(ascending=False).index)
    sv = g[g.p_adj < 0.10]
    show = pd.concat([sv, g.head(6)]).drop_duplicates("zone").head(10)
    out.append(f"\n**{s} · discover {disc}** — {len(g)} tests, null 95th max|t| {g.null_q95.iloc[0]:.2f}, survivors {len(sv)}, confirmed {int(g[g.p_adj < 0.10].CONFIRMED.sum())}\n")
    out.append(C.md_table(show[["zone", "days_zone", "avg_zone", "avg_rest", "t", "p_adj", "conf_days", "conf_diff", "conf_P", "bear_days", "bear_diff", "CONFIRMED"]], 3))

# ── B. fill-level stamp gates
out.append("\n\n## B. Fill-level read of the confirmed macro STAMP separators (cut frozen at discovery; applied per fill; DAY-clustered)\n")
GATES = [("FRENZY_LITE", "entry_btc_1h_slope", ">", 0.0, "keep"),        # discovered 2025 (sign), confirmed H2
         ("FRENZY_LITE", "entry_btc_off24h_pct", "<=", -1.166, "block"),   # discovered 2025 lo3 (and independently H2 lo3 −1.217)
         ("FRENZY_WILLY", "entry_btc_1h_slope", ">", 0.0, "keep"),
         ("FRENZY_WILLY", "entry_btc_off24h_pct", "<=", -1.166, "block"),
         ("FRENZY_LONG", "entry_btc_1h_slope", ">", 0.0, "keep"),
         ("FRENZY_LONG", "entry_btc_off24h_pct", "<=", -1.166, "block"),
         ("FRENZY_WIDE", "entry_btc_1h_slope", ">", 0.0, "keep"),
         ("FRENZY_WIDE", "entry_btc_off24h_pct", "<=", -1.166, "block")]
brow = []
for s, c, op, v, kind in GATES:
    for reg in C.REGIMES + ["ALL"]:
        S = D[(D.sleeve == s) & ((D.regime == reg) if reg != "ALL" else True)]
        x = pd.to_numeric(S[c], errors="coerce")
        cov = float(x.notna().mean())
        blk = ~mk(x, op, v) if kind == "keep" else mk(x, op, v)
        blk = blk & x.notna().values
        y = S.y.values
        for sub, mm in (("all", np.ones(len(S), bool)), ("top50", S.top50.values)):
            b = blk & mm; k = ~blk & mm & x.notna().values
            if b.sum() < 3:
                continue
            yb = y[b]
            lo, hi, ppos = C.day_boot(yb, S.day.values[b])
            loss = -np.minimum(yb, 0)
            pc = pd.Series(loss, index=S.pair.values[b]).groupby(level=0).sum().sort_values(ascending=False)
            dc = pd.Series(loss, index=S.day.values[b]).groupby(level=0).sum().sort_values(ascending=False)
            brow.append(dict(sleeve=s, gate=f"block {c} {'≤' if (kind == 'block') else '≤'} {v:g}" if kind == "block" else f"block {c} ≤ {v:g}",
                             regime=reg, sub=sub, cov=round(cov, 2), N_blocked=int(b.sum()), days=len(set(S.day.values[b])), WR_blk=100 * (yb > 0).mean(),
                             avg_blk=yb.mean(), ci_blk=f"[{lo:+.2f}, {hi:+.2f}]", P_blk_lt0=1 - ppos, N_kept=int(k.sum()), avg_kept=y[k].mean(),
                             be_wr_kept=100 * C.breakeven_wr(y[k]), top2pair_loss_share=pc.head(2).sum() / max(loss.sum(), 1e-9),
                             top1day_loss_share=dc.head(1).sum() / max(loss.sum(), 1e-9), usd_blk=S.usd_new.values[b].sum()))
B = pd.DataFrame(brow)
out.append(C.md_table(B, 3))
out.append("\nExpectancy bar (CLAUDE.md, filter): WR_blk < sleeve breakeven WR (be_wr_kept) · P(avg_blk < 0) ≥ 0.95 day-bootstrap · ≥ 8 days · "
           "no pair / day ≥ 50 % of the blocked loss · N ≥ 15.\n")

# ── C. monthly
out.append("\n## C. Monthly — sleeve avg % vs month-mean of each panel variable (Spearman ρ over 22 months; |ρ| ≥ 0.43 ≈ p < 0.05)\n")
mrows = []
Pm = P.copy(); Pm.index = pd.to_datetime(Pm.index); Pm = Pm.resample("MS").mean(); Pm.index = Pm.index.strftime("%Y-%m")
for s in C.SLEEVES:
    ms = D[D.sleeve == s].groupby("month").y.mean()
    J = Pm.join(ms.rename("y"), how="inner")
    for c in PANEL_VARS:
        rho = J[[c, "y"]].corr(method="spearman").iloc[0, 1]
        j25 = J[J.index < "2026-01"]; j26 = J[J.index >= "2026-01"]
        mrows.append(dict(sleeve=s, var=c, rho_all=rho, rho_2025=j25[[c, "y"]].corr(method="spearman").iloc[0, 1],
                          rho_2026=j26[[c, "y"]].corr(method="spearman").iloc[0, 1]))
MR = pd.DataFrame(mrows)
for s in C.SLEEVES:
    g = MR[MR.sleeve == s].reindex(MR[MR.sleeve == s].rho_all.abs().sort_values(ascending=False).index).head(6)
    out.append(f"\n**{s}**\n"); out.append(C.md_table(g, 2))
mon = Pm[["btc_ret30d", "btc_off30d_hi", "btc_rv30", "alt_above_sma20", "alt_minus_btc_7d", "alt_pump20_share"]].copy()
for s in C.SLEEVES:
    mon[s.split("_")[1]] = D[D.sleeve == s].groupby("month").y.mean()
out.append("\n**Month table (panel month-means; sleeve avg % seed 1)**\n")
out.append(C.md_table(mon.reset_index().rename(columns={"index": "month"}), 2))
open(os.path.join(C.REP, "YR6_WL_FRENZY_regime.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
