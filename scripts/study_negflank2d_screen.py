#!/usr/bin/env python3
"""NEGFLANK 2D study (2026-10-08) — conditional screen INSIDE NEGFLANK (BTC 1h EMA20 slope ≤ −0.05, momentum LONG). Read-only.

Input : reports/NEGFLANK_2D_features.pkl (scripts/study_negflank2d_features.py; cohorts from study_ml_b18_common).
Output: reports/NEGFLANK_2D_sweep_<scan>.csv (every mask, every test) and reports/NEGFLANK_2D_STUDY_tables.md (raw tables).

Design
  * Discovery = yr5 NEGFLANK fills, 3 seeds pooled as REPLICATES; inference is clustered on CALENDAR DAY (all seeds' fills on a
    day form one cluster), so replication never inflates N. N is quoted per seed.
  * Masks: every variable × {sign (if both sides ≥ 10 %), median, low/high tercile, low/high quintile} + every PAIR of variables
    × 4 median quadrants (+ 4 sign quadrants when both are signed). Zone and rest must each hold ≥ 30 fills (≈ 10/seed).
    Unscored fills (NaN) are in NEITHER side (coverage first).
  * Statistic: Δ = mean pct(zone) − mean pct(rest), z = Δ / day-clustered SE (cluster-robust variance of a difference in means).
  * Scan null: whole-day outcome blocks permuted (1,000×) and re-laid onto the fixed features, z recomputed with the moved blocks
    as clusters, max |z| over the SAME full scan → family-wise (scan-corrected) p per mask.
  * Survivor = scan p < 0.05 on yr5 ∧ same sign in both yr5 halves ∧ same sign with every month left out ∧ master (ex-B1)
    NEGFLANK points the same way with ≥ 5 fills in the zone.
  * Pure OOS: a separate scan on yr5 H1 only (cuts frozen from H1) → its top 25 are read on yr5 H2 and on master.
"""
import os, sys, itertools
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
NSH = int(os.environ.get("NSH", 1000))
RNG = np.random.default_rng(20261008)
OUTMD = "reports/NEGFLANK_2D_STUDY_tables.md"
LINES = []


def say(s=""):
    print(s); LINES.append(s)


F = pd.read_pickle("reports/NEGFLANK_2D_features.pkl")
F["slope"] = pd.to_numeric(F.entry_btc_1h_slope, errors="coerce")
F["neg"] = F.slope <= -0.05
F["month"] = F.o.dt.strftime("%Y-%m")

STAMPS = ["entry_adx", "entry_adx_delta", "entry_atr_pct", "entry_bear_pct", "entry_btc_1h_slope", "entry_btc_adx", "entry_btc_atr_pct",
          "entry_btc_dist_from_ema13_pct", "entry_btc_ema20_slope", "entry_btc_rsi", "entry_btc_rsi_1h", "entry_btc_rsi_prev6",
          "entry_btc_trend_gap_pct", "entry_bull_pct", "entry_dist_from_ema13_pct", "entry_ema20_slope", "entry_ema50_slope",
          "entry_ema5_stretch", "entry_ema_gap_5_8", "entry_ema_gap_8_13", "entry_gap", "entry_gap_expand_marginal",
          "entry_global_volume_ratio", "entry_neg_di", "entry_pos_di", "entry_pair_ema20_ema50_gap_pct", "entry_pair_rank",
          "entry_pair_volume_24h_usd", "entry_pair_volume_ratio", "entry_price_vs_ema5_pct", "entry_quality_score",
          "entry_range_position", "entry_rsi", "entry_rsi_prev",
          "d_btc_rsi1h_chg", "d_btc_rsi5m_chg6", "d_btc_adx_chg", "d_rsi_chg", "d_di_spread"]
FLAGS = [c for c in F.columns if c.startswith("entry_pattern_") and c.endswith("_match")]
YR5ONLY = ["entry_btc_above72_pct", "entry_btc_ema50_100_gap_pct", "entry_eth_5m_ret1_pct", "entry_btc_rsi_closed",
           "entry_gap_5_8_signed_pct", "entry_gap_5_20_signed_pct", "entry_gap_5_20_prev_signed_pct", "entry_pair_age_days"]
REBUILT = [c for c in F.columns if c.startswith("k_")]
for c in STAMPS + YR5ONLY + REBUILT:
    F[c] = pd.to_numeric(F[c], errors="coerce")
for c in FLAGS:
    F[c] = F[c].astype(str).str.lower().map({"true": 1.0, "1": 1.0, "1.0": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0})
F["k_weekend"] = (F.k_dow >= 5).astype(float)
MARKET = {c for c in STAMPS + YR5ONLY + REBUILT if ("btc" in c or "eth" in c or "alt" in c or "global" in c or "bull_pct" in c
                                                    or "bear_pct" in c or c in ("k_hour_utc", "k_dow", "k_weekend"))}

M_ALL = F[(F.src == "master")].copy()
M = M_ALL[M_ALL.era != "B1"].copy()
Y = F[F.src == "yr5"].copy()
NS = 3
YN = Y[Y.neg].sort_values(["day", "o"]).reset_index(drop=True)
MN = M[M.neg].reset_index(drop=True)
ydays = np.sort(YN.day.unique())
HALF = ydays[len(ydays) // 2]
YN["half"] = np.where(YN.day < HALF, "H1", "H2")
say(f"# NEGFLANK 2D — raw tables ({pd.Timestamp.now():%Y-%m-%d %H:%M})\n")
say(f"yr5 NEGFLANK: {len(YN)} fills = {len(YN)/NS:.0f}/seed · {YN.day.nunique()} days · halves split at {HALF} · "
    f"master ex-B1 NEGFLANK {len(MN)} fills · {MN.day.nunique()} days")


# ───────────────────────────── stats helpers
def cstat(p, mask, day):
    """Δ and day-clustered z for one mask (p: pct, mask bool, day labels)."""
    a, b = p[mask], p[~mask]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    ma, mb = a.mean(), b.mean()
    e = np.where(mask, (p - ma) / len(a), -(p - mb) / len(b))
    v = pd.Series(e).groupby(day).sum().pow(2).sum()
    return ma - mb, (ma - mb) / np.sqrt(v) if v > 0 else np.nan


def zmat(B, nA, nB, y, bounds):
    """day-clustered z for all masks at once. B (K×n float32), y (n), bounds = start index of each cluster (contiguous)."""
    S = y.sum(); SA = B @ y; mA = SA / nA; mB = (S - SA) / nB
    nAd = np.add.reduceat(B, bounds, axis=1)
    SAd = np.add.reduceat(B * y[None, :], bounds, axis=1)
    nd = np.add.reduceat(np.ones_like(y), bounds); Sd = np.add.reduceat(y, bounds)
    eA = SAd - mA[:, None] * nAd
    eB = (Sd[None, :] - SAd) - mB[:, None] * (nd[None, :] - nAd)
    v = ((eA / nA[:, None] - eB / nB[:, None]) ** 2).sum(1)
    return (mA - mB) / np.sqrt(v), mA - mB


def boot_day(p, day, n=4000, seed=7):
    g = pd.DataFrame({"p": p, "d": day}).groupby("d").p.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n, len(s)))
    mm = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(mm, 2.5), np.percentile(mm, 97.5), (mm < 0).mean()


def line(d, ns=1):
    if len(d) == 0:
        return "0"
    lo, hi, pn = boot_day(d.pct.values, d.day.values)
    return f"{len(d)/ns:.0f} · {(d.pct>0).mean()*100:.0f}% · {d.pct.mean():+.3f} · ${d.usd.sum()/ns:+,.0f} · {d.day.nunique()}d · P(<0) {pn:.2f}"


# ───────────────────────────── variable / mask construction
def usable(c, d):
    x = d[c]
    return x.notna().mean() >= 0.8 and x.nunique() >= 2


def build_masks(D, VARS):
    """returns list of (kind, label, fn(df)->bool array with NaN→False, nan_fn(df)->bool scored, vars, cutinfo)."""
    out = []
    one = {}
    for c in VARS:
        x = D[c].dropna()
        if c in FLAGS or c == "k_weekend":
            sh = (x > 0.5).mean()
            if 0.1 <= sh <= 0.9:
                one[c] = [("=1", 0.5, 1)]
            continue
        cuts = []
        sp = (x > 0).mean()
        if 0.1 <= sp <= 0.9 and x.min() < 0:
            cuts.append(("sign>0", 0.0, 1))
        q = x.quantile([0.2, 1 / 3, 0.5, 2 / 3, 0.8]).values
        cuts += [("med>", q[2], 1), ("T1≤", q[1], 0), ("T3>", q[3], 1), ("Q1≤", q[0], 0), ("Q5>", q[4], 1)]
        one[c] = cuts
    for c, cuts in one.items():
        for lab, v, s in cuts:
            out.append(("1D", f"{c} {lab}{v:.4g}" if lab != "=1" else f"{c}=1", ((c, v, s),)))
    # 2D: median quadrants (+ sign quadrants when both signed)
    cols = list(one)
    for c1, c2 in itertools.combinations(cols, 2):
        def med(c):
            if c in FLAGS or c == "k_weekend":
                return 0.5
            return D[c].median()
        m1, m2 = med(c1), med(c2)
        for s1, s2 in itertools.product((0, 1), (0, 1)):
            out.append(("2D", f"{c1}{'>' if s1 else '≤'}{m1:.4g} ∧ {c2}{'>' if s2 else '≤'}{m2:.4g}", ((c1, m1, s1), (c2, m2, s2))))
        sg1 = any(l == "sign>0" for l, _, _ in one[c1]); sg2 = any(l == "sign>0" for l, _, _ in one[c2])
        if sg1 and sg2 and not (abs(m1) < 1e-12 and abs(m2) < 1e-12):
            for s1, s2 in itertools.product((0, 1), (0, 1)):
                out.append(("2Ds", f"{c1}{'>' if s1 else '≤'}0 ∧ {c2}{'>' if s2 else '≤'}0", ((c1, 0.0, s1), (c2, 0.0, s2))))
    return out


def apply(spec, D):
    zone = np.ones(len(D), bool); scored = np.ones(len(D), bool)
    for c, v, s in spec:
        x = D[c].values
        scored &= ~np.isnan(x)
        zone &= (x > v) if s else (x <= v)
    return zone & scored, scored


def screen(D, VARS, tag, nsh=NSH, mz_min=30):
    """full scan on D (sorted by day) → DataFrame with z, Δ, scan p."""
    specs = build_masks(D, VARS)
    rows, Bl, Sl = [], [], []
    for k, lab, spec in specs:
        z, s = apply(spec, D)
        if s.mean() < 0.8:
            continue
        nz, nr = z.sum(), (s & ~z).sum()
        if nz < mz_min or nr < mz_min:
            continue
        if len(np.unique(D.day.values[z])) < 10 or len(np.unique(D.day.values[s & ~z])) < 10:   # ≥ 10 days each side
            continue
        rows.append((k, lab, spec)); Bl.append(z); Sl.append(s)
    B = np.array(Bl, dtype=np.float32)
    Sm = np.array(Sl, dtype=bool)
    # masks with partial coverage: treat unscored as excluded — handled by restricting y via weights: set B row to zone,
    # and compute rest = scored & ~zone. For the vectorised null we require full coverage (coverage ≥ 0.8 → drop NaN rows
    # by using per-mask "rest" = scored & ~zone): implemented as a second indicator matrix R.
    R = (Sm & ~B.astype(bool)).astype(np.float32)
    y = D.pct.values.astype(np.float32)
    day = D.day.values
    _, bounds = np.unique(day, return_index=True)
    blocks = [np.arange(s, e) for s, e in zip(bounds, list(bounds[1:]) + [len(y)])]

    def zfull(yv, bnd):
        nA = B.sum(1); nB = R.sum(1)
        SA = B @ yv; SB = R @ yv
        mA, mB = SA / nA, SB / nB
        nAd = np.add.reduceat(B, bnd, axis=1); nBd = np.add.reduceat(R, bnd, axis=1)
        SAd = np.add.reduceat(B * yv[None, :], bnd, axis=1); SBd = np.add.reduceat(R * yv[None, :], bnd, axis=1)
        eA = SAd - mA[:, None] * nAd; eB = SBd - mB[:, None] * nBd
        v = ((eA / nA[:, None] - eB / nB[:, None]) ** 2).sum(1)
        QA = B @ (yv * yv); QB = R @ (yv * yv)
        vw = (QA / nA - mA ** 2) / nA + (QB / nB - mB ** 2) / nB
        return (mA - mB) / np.sqrt(v), mA - mB, (mA - mB) / np.sqrt(vw)

    z, dlt, zw = zfull(y, bounds)
    nullmax = np.empty(nsh); nullmax_w = np.empty(nsh); exc = np.empty(nsh); exc_w = np.empty(nsh)
    sizes = np.array([len(b) for b in blocks])
    for i in range(nsh):
        perm = RNG.permutation(len(blocks))
        yp = np.concatenate([y[blocks[j]] for j in perm])
        bp = np.concatenate([[0], np.cumsum(sizes[perm])[:-1]])
        zp, _, zwp = zfull(yp, bp)
        nullmax[i] = np.nanmax(np.abs(zp)); nullmax_w[i] = np.nanmax(np.abs(zwp))
        exc[i] = np.sum(np.abs(zp) >= 3); exc_w[i] = np.sum(np.abs(zwp) >= 3)
    # reference only: trade-level shuffle (anti-conservative under day clustering), clusters = real days
    nullmax_t = np.empty(nsh // 4)
    for i in range(len(nullmax_t)):
        zt, _, _ = zfull(RNG.permutation(y), bounds)
        nullmax_t[i] = np.nanmax(np.abs(zt))
    TRADE_NULL[tag] = nullmax_t
    R_ = pd.DataFrame({"kind": [r[0] for r in rows], "mask": [r[1] for r in rows], "z": z, "delta": dlt,
                       "n_zone": B.sum(1) / NS if tag.startswith("yr5") else B.sum(1), "n_rest": R.sum(1) / NS if tag.startswith("yr5") else R.sum(1)})
    R_["scan_p"] = [(nullmax >= abs(v)).mean() for v in z]
    R_["z_welch"] = zw
    R_["scan_p_welch"] = [(nullmax_w >= abs(v)).mean() for v in zw]
    DIAG[tag] = dict(obs_max_cl=np.nanmax(np.abs(z)), obs_max_w=np.nanmax(np.abs(zw)), null95_cl=np.percentile(nullmax, 95),
                     null95_w=np.percentile(nullmax_w, 95), p_cl=np.mean(nullmax >= np.nanmax(np.abs(z))),
                     p_w=np.mean(nullmax_w >= np.nanmax(np.abs(zw))), obs_exc_cl=int(np.sum(np.abs(z) >= 3)),
                     obs_exc_w=int(np.sum(np.abs(zw) >= 3)), null_exc_cl_med=np.median(exc), null_exc_cl_95=np.percentile(exc, 95),
                     null_exc_w_med=np.median(exc_w), null_exc_w_95=np.percentile(exc_w, 95),
                     p_exc_cl=np.mean(exc >= np.sum(np.abs(z) >= 3)), p_exc_w=np.mean(exc_w >= np.sum(np.abs(zw) >= 3)))
    R_["spec"] = [r[2] for r in rows]
    R_["market_wide"] = [all(c in MARKET for c, _, _ in r[2]) for r in rows]
    return R_, nullmax


def evaluate(R_, D, Dm, top=None):
    """add halves / LOMO / master columns."""
    recs = []
    it = R_ if top is None else R_.head(top)
    for _, r in it.iterrows():
        z, s = apply(r.spec, D)
        p = D.pct.values
        def dd(sel):
            a, b = p[z & sel], p[s & ~z & sel]
            return a.mean() - b.mean() if len(a) >= 3 and len(b) >= 3 else np.nan
        h1 = (D.half == "H1").values if "half" in D else np.ones(len(D), bool)
        mons = D.month.values
        lomo = [dd(mons != mo) for mo in np.unique(mons)]
        mon = [dd(mons == mo) for mo in np.unique(mons)]
        sg = np.sign(r.delta)
        zm, sm = apply(r.spec, Dm)
        pm = Dm.pct.values
        ma, mb = pm[zm], pm[sm & ~zm]
        dm, zmz = cstat(pm[sm], zm[sm], Dm.day.values[sm]) if sm.sum() > 4 else (np.nan, np.nan)
        recs.append(dict(H1=dd(h1), H2=dd(~h1), lomo_same=int(np.nansum(np.sign(lomo) == sg)), lomo_n=int(np.sum(~np.isnan(lomo))),
                         months_same=int(np.nansum(np.sign(mon) == sg)), months_n=int(np.sum(~np.isnan(mon))),
                         zone_avg=p[z].mean(), rest_avg=p[s & ~z].mean(), zone_days=D.day[z].nunique(),
                         m_n=int(zm.sum()), m_rest_n=int((sm & ~zm).sum()), m_avg=ma.mean() if len(ma) else np.nan,
                         m_rest=mb.mean() if len(mb) else np.nan, m_delta=dm, m_z=zmz, m_days=Dm.day[zm].nunique()))
    E = pd.DataFrame(recs, index=it.index)
    out = it.join(E)
    sg = np.sign(out.delta)
    out["halves_ok"] = (np.sign(out.H1) == sg) & (np.sign(out.H2) == sg)
    out["lomo_ok"] = out.lomo_same == out.lomo_n
    out["master_ok"] = (np.sign(out.m_delta) == sg) & (out.m_n >= 5)
    out["survivor"] = ((out.scan_p < 0.05) | (out.scan_p_welch < 0.05)) & out.halves_ok & out.lomo_ok & out.master_ok
    return out


VARS = [c for c in STAMPS + FLAGS + YR5ONLY + REBUILT + ["k_weekend"] if c in YN and usable(c, YN) and c != "k_dow"]
say(f"variables screened: {len(VARS)} (stamped {len([v for v in VARS if v.startswith('entry_') or v.startswith('d_')])} · "
    f"rebuilt {len([v for v in VARS if v.startswith('k_')])}); yr5-only stamps (master scored only): {[v for v in YR5ONLY if v in VARS]}")
MNs = MN.copy(); MNs["half"] = "H1"
RESULTS = {}
TRADE_NULL = {}
DIAG = {}
for tag, D, Dm in [("yr5_all", YN, MN), ("yr5_exwash", YN[~YN.washed30].reset_index(drop=True), MN[~MN.wash].reset_index(drop=True))]:
    D = D.sort_values(["day", "o"]).reset_index(drop=True)
    R_, nullmax = screen(D, VARS, tag)
    R_ = R_.reindex(R_.z.abs().sort_values(ascending=False).index)
    zc = np.percentile(nullmax, 95)
    say(f"\n## scan {tag}: {len(R_)} masks (1D {int((R_.kind=='1D').sum())} · 2D {int((R_.kind!='1D').sum())}) on {len(D)} fills "
        f"({len(D)/NS:.0f}/seed, {D.day.nunique()} days) · observed max |z| {R_.z.abs().max():.2f} · null 95th pct of max |z| {zc:.2f} · "
        f"P(null max ≥ observed max) {np.mean(nullmax >= R_.z.abs().max()):.3f} · reference trade-shuffle null ({len(TRADE_NULL[tag])}×): "
        f"95th {np.percentile(TRADE_NULL[tag], 95):.2f}, P {np.mean(TRADE_NULL[tag] >= R_.z.abs().max()):.3f}")
    dg = DIAG[tag]
    say(f"- day-clustered z: obs max {dg['obs_max_cl']:.2f} vs null95 {dg['null95_cl']:.2f} (p {dg['p_cl']:.3f}); masks |z|≥3: obs {dg['obs_exc_cl']} vs null median "
        f"{dg['null_exc_cl_med']:.0f} / 95th {dg['null_exc_cl_95']:.0f} (p {dg['p_exc_cl']:.3f})")
    say(f"- Welch z (trade SE) under the same day-block null: obs max {dg['obs_max_w']:.2f} vs null95 {dg['null95_w']:.2f} (p {dg['p_w']:.3f}); masks |z|≥3: obs "
        f"{dg['obs_exc_w']} vs null median {dg['null_exc_w_med']:.0f} / 95th {dg['null_exc_w_95']:.0f} (p {dg['p_exc_w']:.3f})")
    E = evaluate(R_, D, Dm, top=None)
    E.drop(columns=["spec"]).to_csv(f"reports/NEGFLANK_2D_sweep_{tag}.csv", index=False)
    RESULTS[tag] = (E, D, Dm)
    say(f"survivors (either statistic): scan p<0.05 cl {int((E.scan_p<0.05).sum())} / welch {int((E.scan_p_welch<0.05).sum())} · +halves {int((((E.scan_p<0.05)|(E.scan_p_welch<0.05))&E.halves_ok).sum())} · +LOMO "
        f"{int((((E.scan_p<0.05)|(E.scan_p_welch<0.05))&E.halves_ok&E.lomo_ok).sum())} · +master {int(E.survivor.sum())}")
    say(f"masks passing halves ∧ LOMO ∧ master WITHOUT the scan correction: {int((E.halves_ok&E.lomo_ok&E.master_ok).sum())} of {len(E)}")
    say("\n(top 25 by day-clustered |z|; scan p = cl / welch)\n")
    say("\n| # | kind | mask | yr5 zone N/seed · avg | rest avg | Δ · z | scan p | H1Δ / H2Δ | LOMO | master zone N · avg · Δ · z | flags |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(E.head(25).iterrows()):
        say(f"| {i+1} | {r.kind} | {r['mask']} | {r.n_zone:.0f} · {r.zone_avg:+.3f} | {r.rest_avg:+.3f} | {r.delta:+.3f} · {r.z:+.2f} | {r.scan_p:.3f} | "
            f"{r.H1:+.3f} / {r.H2:+.3f} | {r.lomo_same}/{r.lomo_n} | {r.m_n} · {r.m_avg:+.3f} · {r.m_delta:+.3f} · {r.m_z:+.2f} | "
            f"{'H' if r.halves_ok else ''}{'L' if r.lomo_ok else ''}{'M' if r.master_ok else ''}{' ★' if r.survivor else ''} |")
    W = E.reindex(E.z_welch.abs().sort_values(ascending=False).index).head(15)
    say("\nTop 15 by Welch |z| (day-block null on Welch):\n\n| mask | yr5 N/seed · avg vs rest · z_w · z_cl | scan p welch | H1/H2 | LOMO | master N · avg vs rest · z | flags |\n|---|---|---|---|---|---|---|")
    for _, r in W.iterrows():
        say(f"| {r['mask']} | {r.n_zone:.0f} · {r.zone_avg:+.3f} vs {r.rest_avg:+.3f} · {r.z_welch:+.2f} · {r.z:+.2f} | {r.scan_p_welch:.3f} | {r.H1:+.3f}/{r.H2:+.3f} | "
            f"{r.lomo_same}/{r.lomo_n} | {r.m_n} · {r.m_avg:+.3f} vs {r.m_rest:+.3f} · {r.m_z:+.2f} | {'H' if r.halves_ok else ''}{'L' if r.lomo_ok else ''}{'M' if r.master_ok else ''} |")
    G = E[E.halves_ok & E.lomo_ok & E.master_ok].head(15)
    say(f"\nTop 15 that pass halves ∧ LOMO ∧ master direction (ignoring scan p):\n\n| mask | yr5 N/seed · avg vs rest · z | scan p | H1/H2 | master N · avg vs rest · z |\n|---|---|---|---|---|")
    for _, r in G.iterrows():
        say(f"| {r['mask']} | {r.n_zone:.0f} · {r.zone_avg:+.3f} vs {r.rest_avg:+.3f} · {r.z:+.2f} | {r.scan_p:.3f} | {r.H1:+.3f}/{r.H2:+.3f} | "
            f"{r.m_n} · {r.m_avg:+.3f} vs {r.m_rest:+.3f} · {r.m_z:+.2f} |")

# ───────────────────────────── pure OOS: discover on H1, test on H2 + master
H1D = YN[YN.half == "H1"].sort_values(["day", "o"]).reset_index(drop=True)
H2D = YN[YN.half == "H2"].sort_values(["day", "o"]).reset_index(drop=True)
TRADE_NULL["yr5_H1"] = None
R1, null1 = screen(H1D, VARS, "yr5_H1")
R1 = R1.reindex(R1.z.abs().sort_values(ascending=False).index)
say(f"\n## pure OOS — scan on yr5 H1 only ({len(H1D)/NS:.0f}/seed, {H1D.day.nunique()} days): max |z| {R1.z.abs().max():.2f} · "
    f"null 95th {np.percentile(null1, 95):.2f} · P {np.mean(null1 >= R1.z.abs().max()):.3f}")
rec = []
for _, r in R1.head(25).iterrows():
    z2, s2 = apply(r.spec, H2D); zm, sm = apply(r.spec, MN)
    d2, zz2 = cstat(H2D.pct.values[s2], z2[s2], H2D.day.values[s2]) if z2.sum() >= 6 and (s2 & ~z2).sum() >= 6 else (np.nan, np.nan)
    dm, zzm = cstat(MN.pct.values[sm], zm[sm], MN.day.values[sm]) if zm.sum() >= 3 and (sm & ~zm).sum() >= 3 else (np.nan, np.nan)
    rec.append((r['mask'], r.n_zone, r.delta, r.z, r.scan_p, z2.sum() / NS, d2, zz2, int(zm.sum()), dm, zzm))
O = pd.DataFrame(rec, columns=["mask", "h1_n", "h1_d", "h1_z", "h1_scan_p", "h2_n", "h2_d", "h2_z", "m_n", "m_d", "m_z"])
O.to_csv("reports/NEGFLANK_2D_oos_H1top25.csv", index=False)
same2 = (np.sign(O.h2_d) == np.sign(O.h1_d)); samem = (np.sign(O.m_d) == np.sign(O.h1_d))
say(f"H1 top-25 → H2 same sign {int(same2.sum())}/{int(O.h2_d.notna().sum())} (|z2| ≥ 1.96 same sign: {int((same2 & (O.h2_z.abs() >= 1.96)).sum())}) · "
    f"master same sign {int(samem.sum())}/{int(O.m_d.notna().sum())}")
say("\n| H1 mask | H1 N/seed · Δ · z · scan p | H2 N/seed · Δ · z | master N · Δ · z |\n|---|---|---|---|")
for _, r in O.iterrows():
    say(f"| {r['mask']} | {r.h1_n:.0f} · {r.h1_d:+.3f} · {r.h1_z:+.2f} · {r.h1_scan_p:.2f} | {r.h2_n:.0f} · {r.h2_d:+.3f} · {r.h2_z:+.2f} | {r.m_n} · {r.m_d:+.3f} · {r.m_z:+.2f} |")

pd.to_pickle({k: (v[0],) for k, v in RESULTS.items()}, "reports/NEGFLANK_2D_screen_results.pkl")
open(OUTMD, "w").write("\n".join(LINES) + "\n")
