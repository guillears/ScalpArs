#!/usr/bin/env python3
"""H1_EMA20 OVERLAP (2026-10-05) — is "BTC 1h EMA20 falling" (FILTER_REGIME_MATRIX side lead) a NEW signal for momentum-LONG fills, or
already covered by the engine's BTC 1h-slope gates (LONG_BTC1H_DEADBAND, 1hPullback_L cell, CALM3D b1h leg)? Read-only research.

Inputs
  yr5 admitted ML fills, de-duplicated across the 3 seeds by (pair, 5-min bucket) and priced with the live exit replica
  (ml_exit_optimize_yr5 BASE, entry +60 s) = $S/calib.csv (built by the FILTER_REGIME_MATRIX run; columns pair,bucket,t,n,pnl,sim60);
  per-seed engine stamps from reports/ENGINE_REPLAY_YR5_ML_fills.csv (entry_btc_1h_slope, entry_btc_adx, cell_multiplier_source).
  Real fills: reports/MASTER_POOL_stacked.csv (MOMENTUM LONG, CLOSED, stack_keep, non-probe, non-MANUAL) — refute-only.
Tag  H1_EMA20_UP = EMA20(span 20, adjust=False) of CLOSED BTC 1h closes (reports/backtest_cache/btc_1h.csv), last bar closed ≤ t,
     slope = e20[k] − e20[k−1] > 0 — identical to filter_regime_matrix.raw_tags (asserted).
Out  reports/H1_EMA20_OVERLAP_2026-10-05_tables.md (+ _fills.csv); the .md report is written by hand from these tables.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path[:0] = [ROOT, "scripts"]
S = os.environ["S"]
import filter_regime_matrix as FM   # noqa: E402
DAY, H = 86_400_000, 3_600_000
RNG = np.random.default_rng(20261005)
OUT = "reports/H1_EMA20_OVERLAP_2026-10-05_tables.md"
L = []
W = L.append

H1 = FM.H1.copy()
E20, SL, C1 = H1.e20.values, H1.sl.values, H1.c.values
OT = H1.open_time.values


def h1(t):
    t = np.asarray(t, dtype=np.int64)
    k = np.searchsorted(OT, t - H, side="right") - 1
    return dict(up=SL[k] > 0, s1=SL[k] / C1[k] * 100, s3c=(E20[k] - E20[k - 3]) / E20[k - 3] * 100)


# ───────────── yr5 admitted fills ─────────────
u = pd.read_csv(f"{S}/calib.csv")
u = u[(u.t >= FM.Y0) & (u.t < FM.Y1) & u.sim60.notna()].copy()
f = pd.read_csv("reports/ENGINE_REPLAY_YR5_ML_fills.csv", low_memory=False)
f["t"] = (pd.to_datetime(f.opened_at.astype(str).str[:23]) - pd.Timestamp(0)).dt.total_seconds().mul(1000).astype("int64")
f["bucket"] = f.t // 300_000 * 300_000
st = f.groupby(["pair", "bucket"]).agg(b1h=("entry_btc_1h_slope", "mean"), badx=("entry_btc_adx", "mean"),
                                       src=("cell_multiplier_source", "first")).reset_index()
u = u.merge(st, on=["pair", "bucket"], how="left")
u["w"] = u.n / 3.0
u["day"] = u.t // DAY
g = h1(u.t.values)
u["up"], u["s1"], u["s3c"] = g["up"], g["s1"], g["s3c"]
assert (FM.raw_tags(u.t.values).H1_EMA20_UP.values == u.up.values).all(), "tag mismatch vs filter_regime_matrix"
u["fall"] = ~u.up
u["win"] = u.sim60 > 0
u["month"] = pd.to_datetime(u.t, unit="ms").dt.month
u["half"] = np.where(u.t < FM.HALF, "Jan–Apr", "May–Oct")
u["pullcell"] = (u.b1h >= -0.2) & (u.b1h < -0.1) & (u.badx >= 18) & (u.badx < 25) & (u.src == "UNMATCHED")
NDAYS = int((FM.Y1 - FM.Y0) // DAY)


def zone(x):
    b = [-np.inf, -0.2, -0.1, -0.05, 0.025, 0.05, 0.1, 0.2, np.inf]
    lab = ["≤−0.20", "(−0.20,−0.10]", "(−0.10,−0.05]", "(−0.05,+0.025) dead-band", "[+0.025,+0.05)", "[+0.05,+0.10)", "[+0.10,+0.20)", "≥+0.20"]
    return pd.cut(x, b, labels=lab, right=False)


u["zone"] = zone(u.b1h)


def stat(d, col="sim60"):
    if not len(d):
        return dict(n=0, days=0, wr=np.nan, avg=np.nan, sum=0.0)
    return dict(n=len(d), days=d.day.nunique(), wr=np.average(d[col] > 0, weights=d.w) * 100, avg=np.average(d[col], weights=d.w),
                sum=float((d[col] * d.w).sum()))


def dayboot(d, reps=2000, col="sim60"):
    gg = d.assign(s=d[col] * d.w).groupby("day")[["s", "w"]].sum()
    PW = RNG.poisson(1.0, (reps, len(gg)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (PW @ gg.s.values) / (PW @ gg.w.values)
    return np.nanquantile(b, [.025, .975]), float(np.nanmean(b < 0))


def fs(d, col="sim60", ci=True):
    s = stat(d, col)
    if not s["n"]:
        return "0 · –"
    txt = f"{s['n']} · {s['days']} d · {s['wr']:.0f}% · {s['avg']:+.3f}"
    if ci and s["days"] >= 3:
        (lo, hi), _ = dayboot(d, col=col)
        txt += f" [{lo:+.3f}, {hi:+.3f}]"
    return txt


def gap_boot(d, reps=2000, col="sim60"):
    days = np.sort(d.day.unique()); ix = {x: i for i, x in enumerate(days)}; nd = len(days)
    di = d.day.map(ix).values
    sums = {}
    for k, m in (("F", d.fall.values), ("R", ~d.fall.values)):
        sums[k] = (np.bincount(di[m], weights=(d[col] * d.w).values[m], minlength=nd), np.bincount(di[m], weights=d.w.values[m], minlength=nd))
    PW = RNG.poisson(1.0, (reps, nd))
    with np.errstate(invalid="ignore", divide="ignore"):
        bf = (PW @ sums["F"][0]) / (PW @ sums["F"][1]); br = (PW @ sums["R"][0]) / (PW @ sums["R"][1])
    return np.nanquantile(bf - br, [.025, .975])


def gap(d, col="sim60"):
    a, b = d[d.fall], d[~d.fall]
    return np.average(a[col], weights=a.w) - np.average(b[col], weights=b.w)


# 0. headline
W("## 0. Headline (yr5 admitted ML fills, replica +60 s, 1×, n_seeds/3-weighted)")
W("")
W("| cohort | N · days · WR · avg % [day-block 95 % CI] |"); W("|---|---|")
W(f"| all | {fs(u)} |"); W(f"| H1 EMA20 rising | {fs(u[u.up])} |"); W(f"| H1 EMA20 falling | {fs(u[u.fall])} |")
lo, hi = gap_boot(u)
W(f"| gap falling − rising | {gap(u):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
W("")

# 1. overlap with engine 1h slope zones
W("## 1. Overlap: H1_EMA20 tag × engine stamp `entry_btc_1h_slope` (3-bar EMA20 % change incl. FORMING 1h bar)")
W("")
W("Live gate geometry: LONG_BTC1H_DEADBAND blocks (−0.05, +0.025); 1hPullback_L 2× cell = [−0.20, −0.10) ∧ BTC 5m ADX [18, 25).")
W("")
W("| engine 1h-slope zone | rising: N · days · WR · avg [CI] | falling: N · days · WR · avg [CI] |"); W("|---|---|---|")
for z in u.zone.cat.categories:
    d = u[u.zone == z]
    W(f"| {z} | {fs(d[d.up])} | {fs(d[d.fall])} |")
d = u[u.b1h.isna()]
if len(d):
    W(f"| (no stamp) | {fs(d[d.up])} | {fs(d[d.fall])} |")
W("")
agree = ((u.b1h < 0) == u.fall).mean() * 100
W(f"Sign agreement between tag (falling) and engine stamp (<0): {agree:.1f} % of fills. "
  f"Falling-tag fills with engine stamp ≥ +0.025 (gate sees RISING): {int(((u.fall) & (u.b1h >= 0.025)).sum())}; "
  f"rising-tag fills with stamp ≤ −0.05: {int(((u.up) & (u.b1h <= -0.05)).sum())}.")
W(f"Falling-tag fills 'near' the dead-band edges (stamp in (−0.10, −0.05] or [+0.025, +0.05)): "
  f"{int((u.fall & (((u.b1h > -0.10) & (u.b1h <= -0.05)) | ((u.b1h >= 0.025) & (u.b1h < 0.05)))).sum())} of {int(u.fall.sum())}.")
W("")
W("| 1hPullback_L 2× cell (UNMATCHED, stamp [−0.20,−0.10), BTC ADX [18,25)) | N · days · WR · avg [CI] |"); W("|---|---|")
W(f"| cell, all | {fs(u[u.pullcell])} |"); W(f"| cell ∧ falling | {fs(u[u.pullcell & u.fall])} |"); W(f"| cell ∧ rising | {fs(u[u.pullcell & u.up])} |")
W("")
W("| sub-sleeve | rising | falling |"); W("|---|---|---|")
for s_ in ("UNMATCHED", "NONEXP_CALM3D", "ADX_SURGE_OPEN"):
    d = u[u.src == s_]
    W(f"| {s_} | {fs(d[d.up])} | {fs(d[d.fall])} |")
W("")

# 2. dose response
W("## 2. Dose-response")
W("")
for col, name in (("s1", "tag's continuous form: 1-bar EMA20 change, % of price (closed bars)"), ("b1h", "engine stamp entry_btc_1h_slope")):
    W(f"### deciles of {name}")
    W(""); W("| decile | range | N · days · WR · avg [CI] |"); W("|---|---|---|")
    q = pd.qcut(u[col], 10, labels=False, duplicates="drop")
    for i in sorted(q.dropna().unique()):
        d = u[q == i]
        W(f"| D{int(i) + 1} | {d[col].min():+.3f} … {d[col].max():+.3f} | {fs(d)} |")
    W("")
W("### falling-tag cohort only, by engine-stamp zone (is the loss in a sub-range the current gates nearly cut?)")
W(""); W("| engine zone | falling N · days · WR · avg [CI] | share of falling-cohort net % |"); W("|---|---|---|")
tot = (u[u.fall].sim60 * u[u.fall].w).sum()
for z in u.zone.cat.categories:
    d = u[u.fall & (u.zone == z)]
    if len(d):
        W(f"| {z} | {fs(d)} | {(d.sim60 * d.w).sum() / tot * 100:.0f} % |")
W("")

# 3. day units
W("## 3. Day units (market-wide variable → a day is one observation)")
W("")
def daymeans(d):
    gg = d.assign(s=d.sim60 * d.w).groupby("day")[["s", "w"]].sum()
    return gg.s / gg.w
dmF, dmR = daymeans(u[u.fall]), daymeans(u[u.up])
def bmean(x, reps=4000):
    b = RNG.choice(x.values, (reps, len(x))).mean(1)
    return np.quantile(b, [.025, .975])
W("| state | days | mean of per-day means [95 % CI] | share of days negative |"); W("|---|---|---|---|")
for nm, x in (("falling", dmF), ("rising", dmR)):
    lo, hi = bmean(x); W(f"| {nm} | {len(x)} | {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {(x < 0).mean() * 100:.0f} % |")
both = dmF.index.intersection(dmR.index)
pdiff = (dmF[both] - dmR[both])
lo, hi = bmean(pdiff)
W(f"| paired (days with both states) falling − rising | {len(both)} | {pdiff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | falling worse on {(pdiff < 0).mean() * 100:.0f} % |")
W("")
W("### halves"); W(""); W("| half | rising | falling | gap [CI] |"); W("|---|---|---|---|")
for hname in ("Jan–Apr", "May–Oct"):
    d = u[u.half == hname]; lo, hi = gap_boot(d)
    W(f"| {hname} | {fs(d[d.up])} | {fs(d[d.fall])} | {gap(d):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
W("")
W("### leave-one-month-out"); W(""); W("| month left out | falling avg | rising avg | gap |"); W("|---|---|---|---|")
for mo in sorted(u.month.unique()):
    d = u[u.month != mo]
    W(f"| {mo:02d} | {stat(d[d.fall])['avg']:+.3f} | {stat(d[d.up])['avg']:+.3f} | {gap(d):+.3f} |")
W("")
W("### per month (context)"); W(""); W("| month | rising | falling |"); W("|---|---|---|")
for mo in sorted(u.month.unique()):
    d = u[u.month == mo]; W(f"| {mo:02d} | {fs(d[d.up], ci=False)} | {fs(d[d.fall], ci=False)} |")
W("")
# per seed: each seed's own fills, priced by the de-duplicated replica result of its bucket
px = dict(zip(zip(u.pair, u.bucket), u.sim60))
f["sim60"] = [px.get(k) for k in zip(f.pair, f.bucket)]
fs_ = f[(f.t >= FM.Y0) & (f.t < FM.Y1) & f.sim60.notna()].copy()
fs_["fall"] = ~h1(fs_.t.values)["up"]; fs_["day"] = fs_.t // DAY; fs_["w"] = 1.0
W("### per seed (each seed's own fills; replica price; also the as-replayed pnl)"); W("")
W("| seed | rising (replica) | falling (replica) | gap [CI] | as-replayed rising / falling avg |"); W("|---|---|---|---|---|")
for s_ in sorted(fs_.seed.unique()):
    d = fs_[fs_.seed == s_]; lo, hi = gap_boot(d)
    pr = pd.to_numeric(d.pnl_percentage, errors="coerce")
    W(f"| {s_} | {fs(d[~d.fall])} | {fs(d[d.fall])} | {gap(d):+.3f} [{lo:+.3f}, {hi:+.3f}] | {pr[~d.fall].mean():+.3f} / {pr[d.fall].mean():+.3f} |")
W("")
# as-replayed (de-dup) pricing cross-check
W(f"As-replayed pricing (de-dup, engine's own exits): rising {fs(u[u.up], 'pnl')} · falling {fs(u[u.fall], 'pnl')} · "
  f"gap {gap(u, 'pnl'):+.3f}")
W("")

# shuffled-day null
NR = int(os.environ.get("NULL_ROUNDS", 1000))
days_all = np.sort(u.day.unique()); dpos = {x: i for i, x in enumerate(days_all)}; dix = u.day.map(dpos).values
month = pd.to_datetime(days_all * DAY, unit="ms").month.values
obs = gap(u)
nul = []
for r in range(NR):
    perm = np.arange(len(days_all))
    for mo in np.unique(month):
        ii = np.flatnonzero(month == mo); perm[ii] = RNG.permutation(ii)
    shift = (days_all[perm] - days_all) * DAY
    fl = ~h1(u.t.values + shift[dix])["up"]
    a = fl; b = ~fl
    nul.append(np.average(u.sim60[a], weights=u.w[a]) - np.average(u.sim60[b], weights=u.w[b]))
nul = np.array(nul)
W(f"### shuffled-day null ({NR} rounds; each day's tag replaced by another same-month day's tag at the same time of day)")
W("")
W(f"observed gap {obs:+.3f}; null mean {nul.mean():+.3f}, sd {nul.std():.3f}, 2.5–97.5 % [{np.quantile(nul, .025):+.3f}, {np.quantile(nul, .975):+.3f}]; "
  f"one-sided p(null ≤ obs) = {(nul <= obs).mean():.3f}; two-sided p(|null| ≥ |obs|) = {(np.abs(nul) >= abs(obs)).mean():.3f}")
W("")

# 4. expectancy bar on the would-be-blocked cohort + Δ
W("## 4. Sleeve-level 'skip ML while H1 EMA20 falling' — expectancy bar + Δ (yr5)")
W("")
wins, loss = u[u.sim60 > 0], u[u.sim60 <= 0]
aw, al = np.average(wins.sim60, weights=wins.w), abs(np.average(loss.sim60, weights=loss.w))
be = al / (aw + al) * 100
F_ = u[u.fall]
(lo, hi), pneg = dayboot(F_, reps=4000)
fs_d = F_.assign(s=F_.sim60 * F_.w).groupby("day").s.sum(); fs_p = F_.assign(s=F_.sim60 * F_.w).groupby("pair").s.sum()
net = fs_d.sum()
W(f"- Sleeve breakeven WR (all admitted, replica): avg win {aw:+.3f} / avg loss −{al:.3f} → **{be:.1f} %**")
W(f"- Blocked cohort: {fs(F_)}; WR {stat(F_)['wr']:.1f} % vs breakeven {be:.1f} %")
W(f"- Window(day)-clustered bootstrap P(avg<0) = {pneg:.3f}; distinct days {F_.day.nunique()}; N {len(F_)}")
W(f"- Largest single-day share of cohort net loss: {fs_d.min() / net * 100:.0f} % ({pd.to_datetime(fs_d.idxmin() * DAY, unit='ms').date()}); "
  f"largest single-pair share: {fs_p.min() / net * 100:.0f} % ({fs_p.idxmin()})")
W(f"- Winners removed: {int((F_.sim60 > 0).sum())} of {len(F_)} (of all winners {int((u.sim60 > 0).sum())}); "
  f"gross winner % removed {float((F_.sim60.clip(lower=0) * F_.w).sum()):+.1f} pts, gross loser % removed {float((F_.sim60.clip(upper=0) * F_.w).sum()):+.1f} pts")
dlt = -net
W(f"- Δ (1×, %-points summed over fills): {dlt:+.1f} pts over {NDAYS} calendar days = {dlt / NDAYS:+.3f} pts/day; "
  f"after 30–50 % haircut {dlt * .5 / NDAYS:+.3f} … {dlt * .7 / NDAYS:+.3f} pts/day. Fills removed {stat(F_)['n']} of {len(u)} ({len(F_) / len(u) * 100:.0f} %).")
W(f"- Avg of KEPT (rising) fills {stat(u[u.up])['avg']:+.3f} vs all {stat(u)['avg']:+.3f} (the sleeve stays ~flat on yr5 even after the skip).")
W("")
# deadband-anchored alternative: only the falling cohort beyond the existing negative edge (≤ −0.05)
W("Alternative scopes (same bar, for orientation only — NOT pre-registered):"); W("")
W("| scope | N · days · WR · avg [CI] | P(avg<0) | Δ pts/day (raw) |"); W("|---|---|---|---|")
for nm, m in (("falling ∧ stamp ≤ −0.05 (engine also sees falling)", u.fall & (u.b1h <= -0.05)),
              ("falling ∧ stamp ≥ +0.025 (engine sees rising)", u.fall & (u.b1h >= 0.025)),
              ("falling ∧ NOT 1hPullback 2× cell", u.fall & ~u.pullcell),
              ("stamp ≤ −0.05 (engine-only definition, any tag)", u.b1h <= -0.05)):
    d = u[m]; (_, _), pn = dayboot(d)
    W(f"| {nm} | {fs(d)} | {pn:.3f} | {-(d.sim60 * d.w).sum() / NDAYS:+.3f} |")
W("")

# 5. master real fills (refute-only)
M = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
M = M[(M.direction == "LONG") & (M.entry_strategy.astype(str) == "MOMENTUM") & (M.status.astype(str) == "CLOSED")
      & M.stack_keep.astype(str).str.lower().isin(["true", "1"]) & ~M.is_probe.astype(str).str.lower().isin(["true", "1"])
      & ~M.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE")].copy()
M = M.drop_duplicates(["opened_at", "pair", "direction"])
M["t"] = (pd.to_datetime(M.opened_at.astype(str).str[:19]) - pd.Timestamp(0)).dt.total_seconds().mul(1000).astype("int64")
M["pct"] = pd.to_numeric(M.pnl_percentage, errors="coerce"); M = M[M.pct.notna()]
M["w"] = 1.0; M["day"] = M.t // DAY
g = h1(M.t.values); M["up"] = g["up"]; M["fall"] = ~M.up; M["b1h"] = pd.to_numeric(M.entry_btc_1h_slope, errors="coerce")
M["zone"] = zone(M.b1h)
wo = (M.t >= pd.Timestamp("2026-06-18").value // 10**6) & (M.t < pd.Timestamp("2026-07-03").value // 10**6)
W("## 5. Real fills (refute-only): master pool, MOMENTUM LONG, CLOSED, kept, non-probe, non-MANUAL, as traded (pnl_percentage)")
W("")
W(f"Span {str(M.opened_at.min())[:10]} → {str(M.opened_at.max())[:10]}.")
W("")
W("| cohort | rising: N · days · WR · avg [CI] | falling: N · days · WR · avg [CI] | gap |"); W("|---|---|---|---|")
for nm, d in (("all", M), ("without washed-out Jun-18→Jul-2", M[~wo]), ("washed-out window only", M[wo])):
    gp = (d[d.fall].pct.mean() - d[d.up].pct.mean()) if d.fall.any() and d.up.any() else np.nan
    W(f"| {nm} | {fs(d[d.up], 'pct')} | {fs(d[d.fall], 'pct')} | {gp:+.3f} |")
W("")
W("| engine zone (master) | rising | falling |"); W("|---|---|---|")
for z in M.zone.cat.categories:
    d = M[M.zone == z]
    if len(d):
        W(f"| {z} | {fs(d[d.up], 'pct', ci=False)} | {fs(d[d.fall], 'pct', ci=False)} |")
W("")
W("| era (master) | rising | falling |"); W("|---|---|---|")
for e in M.era.value_counts().index:
    d = M[M.era == e]; W(f"| {e} | {fs(d[d.up], 'pct', ci=False)} | {fs(d[d.fall], 'pct', ci=False)} |")
W("")
mw, ml = M[M.pct > 0], M[M.pct <= 0]
be_m = abs(ml.pct.mean()) / (mw.pct.mean() + abs(ml.pct.mean())) * 100
Fm = M[M.fall]
W(f"Master skip-rule read: breakeven WR {be_m:.1f} %; blocked cohort {fs(Fm, 'pct')}; winners removed {int((Fm.pct > 0).sum())}; "
  f"Δ {-Fm.pct.sum():+.2f} pts over {M.day.nunique()} trading days with ML fills ({-Fm.pct.sum() / M.day.nunique():+.3f} pts/day); "
  f"without washed-out: Δ {-Fm[~wo[Fm.index]].pct.sum():+.2f} pts.")
W("")

open(OUT, "w").write("\n".join(L) + "\n")
u.to_csv("reports/H1_EMA20_OVERLAP_2026-10-05_fills.csv", index=False)
print("\n".join(L))
