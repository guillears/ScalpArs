#!/usr/bin/env python3
"""FRENZY RUNNER-ID 2026-10-19 — can a RUNNER be identified at entry? Screen, null, OOS, model AUC, sizing price.

Research only. Input: <scratch>/cohort.pkl (study_frenzy_runner_id_build.py). Writes reports/FRENZY_RUNNER_ID_2026-10-19_*.csv.

PRE-REGISTERED (frozen before the screen was read; the only outcome numbers seen before were the 200-fill smoke-test base rates):
  outcome   runner = mfe_720 >= +5 (net, before the −3 stop, stop-first); sensitivity +3 / +8 and the 2 h / 6 h horizons
  blocks    BT = yr6 2-year replay seed 1 (today's stack), split 2025 (discovery) · 2026-H1 · 2026-H2 (< 2026-10-03, real era excluded);
            REAL = master kept non-probe + live B23
  screen    per sleeve on BT 2025: every feature with >= 50 % coverage × {sign, median, low tercile, high tercile} (zone vs rest,
            two-proportion z) + exhaustive 2D median quadrants over features with >= 70 % coverage
  null      day-block: each UTC day's outcome vector replaced by a random donor day's (resampled to size) — 400 draws 1D, 100 draws 2D;
            familywise p of max z (runner-RICH side) and the count of z >= 3 (1D) / >= 3.5 (2D)
  OOS       top 15 1D + top 15 2D runner-rich cells (zone N >= 30) + the theory cells, frozen 2025 cut → 2026-H1 and 2026-H2 → real.
            SURVIVOR = runner rate higher in BOTH 2026 halves and the pooled-2026 day-bootstrap P(zone > rest) >= 0.95
  model     ridge logistic on every >= 70 %-covered feature (median-imputed, z-scored on 2025), trained 2025 → AUC on 2026-H1 / H2 / real
  sizing    Pattern-W bar (N >= 30 ∧ WR >= 70 % ∧ avg >= +0.10 ∧ Σ > 0) on the zone at today's exit and the runner exits, BE-compat
            (>= 60 % of zone losers peak >= +0.20), $ at today's notionals (LONG blocker $100 → a sized LONG priced at $2,000; LITE $2,000;
            WILLY $2,250 ≈ live 10-17/19 rows) at 1× / 2× / 3×, worst day and max drawdown vs today, 30–50 % haircut
THEORY (written before the screen; direction = MORE runners in the zone):
  T1 late-ON multi-day trend   entry_frenzy_hours > 24 ∧ h_4h_align == 1                  (AGT 10-19: 94.9 h, 4h stacked)
  T2 HTF alignment             h_1h_align == 1 ∧ h_4h_align == 1
  T3 liquid mid-cap            1e8 <= entry_mcap_usd <= 1e9 ∧ entry_pair_volume_24h_usd >= 1e8
  T4 big volume multiple       entry_frenzy_vol_mult >= 200
  T5 BTC tailwind              entry_btc_1h_slope > 0 ∧ b_4h_gap20_50 > 0
  T6 fresh pump (FEWER)        h_hrs_since_7d_low <= 24                                   (pumps mean-revert, Oct-1)
  T7 multi-day staircase       h_up_days_streak >= 2
  T8 stretched (FEWER)         entry_frenzy_vs_vwap_pct >= 13                             (STRETCHED line's frozen 75th pct)
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"; os.chdir(ROOT)
S = sys.argv[1]
OUT = "reports/FRENZY_RUNNER_ID_2026-10-19"
RNG = np.random.default_rng(19)
SL = ["FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY"]
NOT = {"FRENZY_LONG": 2000.0, "FRENZY_LITE": 2000.0, "FRENZY_WILLY": 2250.0}
TABLES = {}


def tbl(name, df):
    TABLES[name] = df
    df.to_csv(f"{OUT}_{name}.csv", index=False)
    return df


# ───────────────────────── load ─────────────────────────
D = pd.read_pickle(f"{S}/cohort.pkl")
D = D[~((D.blk == "BT") & (D.ts >= "2026-10-03"))].copy()
D["day"] = D.ts.dt.strftime("%Y-%m-%d")
D["per"] = np.where(D.blk != "BT", "REAL", np.where(D.ts < "2026-01-01", "2025", np.where(D.ts < "2026-07-01", "26H1", "26H2")))
for th in (3, 5, 8):
    D[f"run{th}"] = (D.mfe_720 >= th).astype(float)
D["run5_120"] = (D.mfe_120 >= 5).astype(float); D["run5_360"] = (D.mfe_360 >= 5).astype(float)
D["x_hour"] = D.ts.dt.hour + D.ts.dt.minute / 60
D["x_weekday"] = D.ts.dt.weekday.astype(float)
mc = pd.to_numeric(D.entry_mcap_usd, errors="coerce"); v24 = pd.to_numeric(D.entry_pair_volume_24h_usd, errors="coerce")
D["x_log_mcap"] = np.log10(mc.where(mc > 0)); D["x_log_vol24"] = np.log10(v24.where(v24 > 0))
D["x_turnover"] = v24 / mc.where(mc > 0)
D["x_vs24high_after"] = D.h_off24h_high

EXCL = {"entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional", "entry_liquidity_cap_notional",
        "entry_bracket_cap_notional", "entry_bracket_max_leverage", "entry_order_type", "entry_strategy", "entry_frenzy_spike_at",
        "entry_btc_regime_started_at", "entry_surge_trigger_at", "entry_frenzy_vwap", "entry_order_lag_s"}
feats = []
for c in D.columns:
    if c in EXCL or c.startswith("entry_book") or c.startswith("entry_pattern"):
        continue
    if c.startswith(("entry_", "h_", "b_", "x_")):
        s = pd.to_numeric(D[c], errors="coerce")
        if s.notna().sum() > 50 and s.nunique() > 1:
            D[c] = s; feats.append(c)
feats = sorted(set(feats))


# ───────────────────────── helpers ─────────────────────────
def dayboot_diff(z, y, day, n=2000):
    """P(rate_zone > rate_rest) by day-block bootstrap."""
    df = pd.DataFrame(dict(z=z, y=y, d=day))
    g = df.groupby(["d", "z"]).y.agg(["sum", "count"]).unstack(fill_value=0)
    days = g.index.values; S1, C1 = g[("sum", True)].values if ("sum", True) in g else 0, g[("count", True)].values if ("count", True) in g else 0
    S0, C0 = g[("sum", False)].values, g[("count", False)].values
    idx = RNG.integers(0, len(days), (n, len(days)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = S1[idx].sum(1) / C1[idx].sum(1) - S0[idx].sum(1) / C0[idx].sum(1)
    d = d[np.isfinite(d)]
    return float((d > 0).mean()) if len(d) else np.nan, (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (np.nan, np.nan)


def dayboot_mean(x, day, n=2000):
    df = pd.DataFrame(dict(x=x, d=day)).dropna()
    if not len(df):
        return np.nan, (np.nan, np.nan)
    g = df.groupby("d").x.agg(["sum", "count"]); s, c = g["sum"].values, g["count"].values
    idx = RNG.integers(0, len(g), (n, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float((m > 0).mean()), (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def ztest(Z, y):
    """Z: n×k bool zone matrix, y: n float {0,1} → z (zone rate − rest rate), n_zone, rate_zone, rate_rest."""
    n = len(y); nz = Z.sum(0).astype(float); sz = y @ Z
    tot = y.sum(); nr = n - nz; sr = tot - sz
    with np.errstate(invalid="ignore", divide="ignore"):
        pz, pr = sz / nz, sr / nr; p = tot / n
        se = np.sqrt(p * (1 - p) * (1 / nz + 1 / nr))
        z = (pz - pr) / se
    z[(nz < 30) | (nr < 30)] = np.nan
    return z, nz, pz, pr


def null_y(y, day, draws):
    """day-block null: each day's outcomes replaced by a random donor day's (resampled to size)."""
    days = pd.Series(day)
    groups = {d: np.flatnonzero(days.values == d) for d in days.unique()}
    keys = list(groups); yv = {d: y[ix] for d, ix in groups.items()}
    out = []
    for _ in range(draws):
        yy = np.empty_like(y)
        for d, ix in groups.items():
            dn = keys[RNG.integers(len(keys))]
            yy[ix] = RNG.choice(yv[dn], size=len(ix), replace=len(yv[dn]) < len(ix)) if len(yv[dn]) else 0
        out.append(yy)
    return out


def cells_1d(X, cols, cuts):
    zs, names = [], []
    for c in cols:
        x = X[c].values
        for kind, cut in cuts[c]:
            if kind in ("sign", "med", "hi"):
                z = x > cut
            else:
                z = x <= cut
            z = z & np.isfinite(x)
            zs.append(z); names.append((c, kind, cut))
    return np.array(zs).T, names


def make_cuts(X, cols):
    cuts = {}
    for c in cols:
        x = X[c].dropna()
        cs = []
        if x.min() < 0 < x.max():
            cs.append(("sign", 0.0))
        q = x.quantile([1 / 3, 0.5, 2 / 3]).values
        cs += [("med", float(q[1])), ("lo", float(q[0])), ("hi", float(q[2]))]
        cuts[c] = cs
    return cuts


def apply_cell(X, cell):
    if cell[0] == "2D":
        _, a, ca, sa, b, cb, sb = cell
        xa, xb = X[a].values, X[b].values
        za = (xa > ca) if sa else (xa <= ca); zb = (xb > cb) if sb else (xb <= cb)
        return za & zb & np.isfinite(xa) & np.isfinite(xb), np.isfinite(xa) & np.isfinite(xb)
    if cell[0] == "TH":
        sc = X[cell[4]].notna().all(axis=1).values
        return cell[2](X) & sc, sc
    c, kind, cut = cell
    x = X[c].values
    z = (x > cut) if kind in ("sign", "med", "hi") else (x <= cut)
    return z & np.isfinite(x), np.isfinite(x)


THEORY = [
    ("TH", "T1 late-ON multi-day (hours>24 ∧ 4h aligned)", lambda X: (X.entry_frenzy_hours > 24).values & (X.h_4h_align == 1).values, +1, ["entry_frenzy_hours", "h_4h_align"]),
    ("TH", "T2 1h ∧ 4h EMA aligned", lambda X: (X.h_1h_align == 1).values & (X.h_4h_align == 1).values, +1, ["h_1h_align", "h_4h_align"]),
    ("TH", "T3 liquid mid-cap ($100M–1B mcap ∧ vol24 ≥ $100M)",
     lambda X: X.entry_mcap_usd.between(1e8, 1e9).values & (X.entry_pair_volume_24h_usd >= 1e8).values, +1, ["entry_mcap_usd", "entry_pair_volume_24h_usd"]),
    ("TH", "T4 volume multiple ≥ 200×", lambda X: (X.entry_frenzy_vol_mult >= 200).values, +1, ["entry_frenzy_vol_mult"]),
    ("TH", "T5 BTC tailwind (1h slope>0 ∧ 4h EMA20>50)", lambda X: (X.entry_btc_1h_slope > 0).values & (X.b_4h_gap20_50 > 0).values, +1, ["entry_btc_1h_slope", "b_4h_gap20_50"]),
    ("TH", "T6 fresh pump (7d low ≤ 24 h ago) [fewer]", lambda X: (X.h_hrs_since_7d_low <= 24).values, -1, ["h_hrs_since_7d_low"]),
    ("TH", "T7 multi-day staircase (≥ 2 up days)", lambda X: (X.h_up_days_streak >= 2).values, +1, ["h_up_days_streak"]),
    ("TH", "T8 stretched vs spike VWAP ≥ 13 % [fewer]", lambda X: (X.entry_frenzy_vs_vwap_pct >= 13).values, -1, ["entry_frenzy_vs_vwap_pct"]),
]


def cell_name(cell):
    if cell[0] == "2D":
        _, a, ca, sa, b, cb, sb = cell
        return f"{a} {'>' if sa else '≤'} {ca:.4g} ∧ {b} {'>' if sb else '≤'} {cb:.4g}"
    if cell[0] == "TH":
        return cell[1]
    c, kind, cut = cell
    return f"{c} {'>' if kind in ('sign', 'med', 'hi') else '≤'} {cut:.4g} ({kind})"


# ───────────────────────── 1. base rates ─────────────────────────
rows = []
for sv in SL:
    for per in ("2025", "26H1", "26H2", "BT_ALL", "REAL"):
        x = D[(D.sleeve == sv) & ((D.per == per) | ((per == "BT_ALL") & (D.blk == "BT")))]
        if not len(x):
            continue
        rr = x[x.run5 == 1]
        tp = x[x.x0 > 0]
        rows.append(dict(sleeve=sv, block=per, N=len(x), days=x.day.nunique(), run3=x.run3.mean() * 100, run5=x.run5.mean() * 100,
                         run8=x.run8.mean() * 100, run5_2h=x.run5_120.mean() * 100, run5_6h=x.run5_360.mean() * 100,
                         stop_12h=x.stop_720.mean() * 100, med_mfe=x.mfe_720.median(), med_tmfe_runners_min=rr.tmfe_720.median(),
                         share_X0_winners_that_ran5=(tp.run5.mean() * 100 if len(tp) else np.nan),
                         X0=x.x0.mean(), R1=x.r1.mean(), R2=x.r2.mean(), censored=int(x.cens_720.sum())))
tbl("base_rates", pd.DataFrame(rows).round(3))

# parity of the walker vs the replay / books
par = []
for sv, col in (("FRENZY_LONG", "x15"), ("FRENZY_LITE", "x3"), ("FRENZY_WILLY", "x0")):
    x = D[(D.blk == "BT") & (D.sleeve == sv)]
    d = (x[col] - x.pct_new).abs()
    par.append(dict(block="BT vs pct_new", sleeve=sv, walker=col, N=len(x), within_005=(d < 0.05).mean(), same_sign=(np.sign(x[col]) == np.sign(x.pct_new)).mean(),
                    walker_avg=x[col].mean(), book_avg=x.pct_new.mean()))
    r = D[(D.blk != "BT") & (D.sleeve == sv)]
    d = (r.x0 - r.y_book).abs()
    par.append(dict(block="REAL x0 vs booked/stack", sleeve=sv, walker="x0", N=len(r), within_005=(d < 0.1).mean(),
                    same_sign=(np.sign(r.x0) == np.sign(r.y_book)).mean(), walker_avg=r.x0.mean(), book_avg=r.y_book.mean()))
tbl("parity", pd.DataFrame(par).round(3))

# ───────────────────────── 2. coverage ─────────────────────────
cov = []
for sv in SL:
    for c in feats:
        e = {"sleeve": sv, "feature": c}
        for per in ("2025", "26H1", "26H2", "REAL"):
            x = D[(D.sleeve == sv) & (D.per == per)]
            e[per] = round(float(x[c].notna().mean()) * 100, 1) if len(x) else np.nan
        cov.append(e)
COV = tbl("coverage", pd.DataFrame(cov))

# ───────────────────────── 3. screen + null + OOS per sleeve ─────────────────────────
screen_rows, null_rows, oos_rows, model_rows = [], [], [], []
SURV = {}
for sv in SL + ["POOLED"]:
    X = D[(D.sleeve == sv) | (sv == "POOLED")].copy()
    X25 = X[X.per == "2025"].reset_index(drop=True)
    cv = COV[COV.sleeve == (sv if sv != "POOLED" else "FRENZY_LITE")].set_index("feature")
    c1 = [c for c in feats if X25[c].notna().mean() >= 0.5 and X25[c].nunique() > 2 or (X25[c].notna().mean() >= 0.5 and X25[c].nunique() == 2)]
    c2 = [c for c in c1 if X25[c].notna().mean() >= 0.7 and X25[c].nunique() > 2]
    cuts = make_cuts(X25, c1)
    y = X25.run5.values.astype(float)
    Z1, n1 = cells_1d(X25, c1, cuts)
    z1, nz1, pz1, pr1 = ztest(Z1, y)
    # 2D median quadrants
    med = {c: float(X25[c].median()) for c in c2}
    M = {c: X25[c].values for c in c2}
    Z2, n2 = [], []
    for a_i in range(len(c2)):
        a = c2[a_i]; xa = M[a]
        for b in c2[a_i + 1:]:
            xb = M[b]; ok = np.isfinite(xa) & np.isfinite(xb)
            for sa in (True, False):
                za = (xa > med[a]) if sa else (xa <= med[a])
                for sb in (True, False):
                    zb = (xb > med[b]) if sb else (xb <= med[b])
                    Z2.append(za & zb & ok); n2.append(("2D", a, med[a], sa, b, med[b], sb))
    Z2 = np.array(Z2).T
    z2, nz2, pz2, pr2 = ztest(Z2, y)
    # null
    ny = null_y(y, X25.day.values, 400)
    mx1, h1 = [], []
    for yy in ny:
        zz = ztest(Z1, yy)[0]; mx1.append(np.nanmax(zz)); h1.append(int(np.nansum(zz >= 3)))
    mx2, h2 = [], []
    for yy in ny[:100]:
        zz = ztest(Z2, yy)[0]; mx2.append(np.nanmax(zz)); h2.append(int(np.nansum(zz >= 3.5)))
    for lay, zz, mxn, hn, thr, k in (("1D", z1, mx1, h1, 3.0, len(n1)), ("2D", z2, mx2, h2, 3.5, len(n2))):
        null_rows.append(dict(sleeve=sv, layer=lay, tests=k, N2025=len(X25), base_run5_2025=y.mean() * 100, max_z_obs=np.nanmax(zz),
                              null_max_z_95=np.percentile(mxn, 95), fw_p=float(np.mean(np.array(mxn) >= np.nanmax(zz))),
                              hits_obs=int(np.nansum(zz >= thr)), hits_null_mean=float(np.mean(hn)), hits_null_95=float(np.percentile(hn, 95)),
                              hits_p=float(np.mean(np.array(hn) >= np.nansum(zz >= thr)))))
    # top cells
    o1 = np.argsort(-np.nan_to_num(z1, nan=-99))[:15]; o2 = np.argsort(-np.nan_to_num(z2, nan=-99))[:15]
    for lay, oo, nn, zz, nz, pz, pr in (("1D", o1, n1, z1, nz1, pz1, pr1), ("2D", o2, n2, z2, nz2, pz2, pr2)):
        for j in oo:
            screen_rows.append(dict(sleeve=sv, layer=lay, cell=cell_name(nn[j]), z=zz[j], n_zone=nz[j], run5_zone=pz[j] * 100, run5_rest=pr[j] * 100))
    cand = [("1D", n1[j]) for j in o1] + [("2D", n2[j]) for j in o2] + [("TH", t) for t in THEORY]
    for lay, cell in cand:
        e = dict(sleeve=sv, layer=lay, cell=cell_name(cell), pred=(cell[3] if cell[0] == "TH" else +1))
        sgn = e["pred"]
        ok_dir = []
        for per in ("2025", "26H1", "26H2", "REAL"):
            Xp = X[X.per == per]
            if not len(Xp):
                continue
            z, sc = apply_cell(Xp, cell)
            zz_, rr_ = Xp[z], Xp[sc & ~z]
            e[f"{per}_n"] = int(z.sum()); e[f"{per}_days"] = zz_.day.nunique()
            e[f"{per}_run5_zone"] = zz_.run5.mean() * 100 if len(zz_) else np.nan
            e[f"{per}_run5_rest"] = rr_.run5.mean() * 100 if len(rr_) else np.nan
            e[f"{per}_X0_zone"] = zz_.x0.mean() if len(zz_) else np.nan
            e[f"{per}_X0_rest"] = rr_.x0.mean() if len(rr_) else np.nan
            e[f"{per}_R1_zone"] = zz_.r1.mean() if len(zz_) else np.nan
            if per in ("26H1", "26H2"):
                ok_dir.append(np.isfinite(e[f"{per}_run5_zone"]) and sgn * (e[f"{per}_run5_zone"] - e[f"{per}_run5_rest"]) > 0)
        X26 = X[X.per.isin(["26H1", "26H2"])]
        z, sc = apply_cell(X26, cell)
        m = sc
        if z.sum() >= 10 and (m & ~z).sum() >= 10:
            P, ci = dayboot_diff(z[m], X26.run5.values[m], X26.day.values[m])
            e["P26_zone_gt_rest"] = P if sgn > 0 else 1 - P
        e["survivor"] = bool(all(ok_dir) and len(ok_dir) == 2 and e.get("P26_zone_gt_rest", 0) >= 0.95)
        if e["survivor"]:
            SURV.setdefault(sv, []).append(cell)
        oos_rows.append(e)
    # model AUC (ridge logistic, numpy)
    mc_ = [c for c in c2]
    mu = X25[mc_].median(); sd = X25[mc_].fillna(mu).std().replace(0, 1)

    def prep(Z):
        return ((Z[mc_].fillna(mu) - mu) / sd).clip(-5, 5).values

    A = np.c_[np.ones(len(X25)), prep(X25)]
    w = np.zeros(A.shape[1]); lam = 5.0
    for _ in range(400):
        p = 1 / (1 + np.exp(-A @ w)); g = A.T @ (p - y) / len(y) + lam * np.r_[0, w[1:]] / len(y)
        Wd = p * (1 - p); Hm = (A.T * Wd) @ A / len(y) + lam * np.diag(np.r_[0, np.ones(len(w) - 1)]) / len(y)
        w -= np.linalg.solve(Hm + 1e-9 * np.eye(len(w)), g)

    def auc(s, t):
        t = np.asarray(t); r = pd.Series(s).rank().values
        n1_, n0_ = t.sum(), len(t) - t.sum()
        return (r[t == 1].sum() - n1_ * (n1_ + 1) / 2) / (n1_ * n0_) if n1_ and n0_ else np.nan

    for per in ("2025", "26H1", "26H2", "REAL"):
        Xp = X[X.per == per]
        if len(Xp) < 10:
            continue
        sc_ = np.c_[np.ones(len(Xp)), prep(Xp)] @ w
        q = pd.Series(sc_).rank(pct=True).values
        top = Xp[q >= 0.8]
        model_rows.append(dict(sleeve=sv, block=per, N=len(Xp), features=len(mc_), AUC=auc(sc_, Xp.run5.values),
                               run5_all=Xp.run5.mean() * 100, run5_top20pct=top.run5.mean() * 100, X0_top20=top.x0.mean(), R1_top20=top.r1.mean(),
                               X0_all=Xp.x0.mean(), R1_all=Xp.r1.mean()))

tbl("screen_top", pd.DataFrame(screen_rows).round(3))
tbl("null", pd.DataFrame(null_rows).round(3))
OOS = tbl("oos", pd.DataFrame(oos_rows).round(3))
tbl("model_auc", pd.DataFrame(model_rows).round(3))


# ───────────────────────── 4. sizing price (theory + survivors + the best 2025 cells) ─────────────────────────
def book(x, col, mult, base):
    """daily $ path at the sleeve notional; mult applies to zone fills."""
    usd = x[col] / 100 * NOT[base] * np.where(x.zone, mult, 1.0)
    d = usd.groupby(x.day).sum().sort_index()
    eq = d.cumsum(); dd = (eq - eq.cummax()).min()
    return usd.sum(), d.min(), dd


price_rows = []
cells_to_price = []
for sv in SL:
    for cell in SURV.get(sv, []):
        cells_to_price.append((sv, "SURVIVOR", cell))
    for t in THEORY:
        cells_to_price.append((sv, "THEORY", t))
for sv, why, cell in cells_to_price:
    X = D[(D.sleeve == sv) & (D.blk == "BT")].copy()
    z, sc = apply_cell(X, cell); X["zone"] = z
    for per in ("2025", "2026", "ALL"):
        x = X if per == "ALL" else X[X.per.isin(["2025"] if per == "2025" else ["26H1", "26H2"])]
        zz = x[x.zone]
        if len(zz) < 5:
            continue
        for col in ("x0", "r1", "r2"):
            P_, ci = dayboot_mean(zz[col].values, zz.day.values)
            losers = zz[zz[col] <= 0]
            e = dict(sleeve=sv, kind=why, cell=cell_name(cell), period=per, exit=col, N=len(zz), days=zz.day.nunique(),
                     WR=(zz[col] > 0).mean() * 100, avg=zz[col].mean(), ci_lo=ci[0], ci_hi=ci[1], P_pos=P_, sum_pts=zz[col].sum(),
                     run5=zz.run5.mean() * 100, rest_avg_x0=x[~x.zone].x0.mean(),
                     top_pair_share=(zz.groupby("pair")[col].sum().abs().max() / max(zz[col].abs().sum(), 1e-9)),
                     be_compat=(losers.mfe_720 >= 0.2).mean() * 100 if len(losers) else np.nan)
            e["patternW"] = bool(e["N"] >= 30 and e["WR"] >= 70 and e["avg"] >= 0.10 and e["sum_pts"] > 0)
            base_usd, base_wd, base_dd = book(x.assign(zone=False), "x0", 1, sv)
            for k in (1, 2, 3):
                xx = x.copy()
                xx["mix"] = np.where(xx.zone, xx[col], xx.x0)
                u, wd, dd = book(xx, "mix", k, sv)
                e[f"usd_{k}x"] = u; e[f"worstday_{k}x"] = wd; e[f"maxdd_{k}x"] = dd
            e["usd_today"] = base_usd; e["worstday_today"] = base_wd; e["maxdd_today"] = base_dd
            price_rows.append(e)
tbl("sizing", pd.DataFrame(price_rows).round(3))

# ───────────────────────── 5. real fills + AGT ─────────────────────────
R = D[D.blk != "BT"].copy()
keep = ["blk", "sleeve", "opened_at", "pair", "y_book", "x0", "r1", "r2", "mfe_120", "mfe_720", "tmfe_720", "stop_720", "run5",
        "entry_frenzy_hours", "entry_frenzy_vol_mult", "entry_frenzy_vs_vwap_pct", "entry_mcap_usd", "entry_pair_volume_24h_usd",
        "h_ret24", "h_ret72", "h_hrs_since_7d_low", "h_1h_align", "h_4h_align", "h_up_days_streak", "entry_btc_1h_slope", "b_4h_gap20_50"]
tbl("real_fills", R[keep].sort_values(["sleeve", "opened_at"]).round(3))
th_real = []
for sv in SL:
    x = R[R.sleeve == sv]
    for t in THEORY:
        z, sc = apply_cell(x, t)
        th_real.append(dict(sleeve=sv, cell=t[1], n_zone=int(z.sum()), days=x[z].day.nunique(), run5_zone=x[z].run5.mean() * 100,
                            run5_rest=x[~z & sc].run5.mean() * 100, X0_zone=x[z].x0.mean(), R1_zone=x[z].r1.mean(),
                            agt_in_zone=int((x[z].pair == "AGTUSDT").sum())))
tbl("theory_real", pd.DataFrame(th_real).round(3))
json.dump({k: [cell_name(c) for c in v] for k, v in SURV.items()}, open(f"{OUT}_survivors.json", "w"), indent=1)
print("survivors", {k: [cell_name(c) for c in v] for k, v in SURV.items()})
for k, v in TABLES.items():
    if k in ("coverage", "real_fills"):
        continue
    print(f"\n=== {k}\n", v.to_string(max_rows=80, max_colwidth=70))
