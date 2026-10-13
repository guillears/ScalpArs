#!/usr/bin/env python3
"""YR6 WL — FRENZY exhaustive winners-vs-losers screen, per sleeve, discover in one BULL regime → confirm in the other; bear H1 reported.

  venv/bin/python -I scripts/study_yr6wl_frenzy_screen.py [--nperm 500]
  → reports/YR6_WL_FRENZY_screen_tests.csv (every test, every direction) · YR6_WL_FRENZY_screen_oos.csv (survivors + confirmation)
    · YR6_WL_FRENZY_screen.md (tables pasted into reports/YR6_WL_FRENZY.md)

Cohort: seed 1 (FRENZY seeds ≈ one path), y = pct_new. Variables: every entry_* column scored on ≥ 80 % of the sleeve × regime's fills
(< 80 % → own scored-only read in the summary, never zone-vs-rest), minus price-level / sizing / execution columns (entry_frenzy_vwap,
entry_desired_notional, entry_liquidity_cap_notional, entry_slippage_pct), plus derived fill-level vars: hour (UTC), k_ep (1 + earlier
FRENZY-family fills of the same (pair, spike) episode), top50 (0/1), lvol24 = log10 entry_pair_volume_24h_usd, fz_age_min (open − spike).
PAIR family (fill units): each var at sign (x > 0, only when both signs occur), median (x > med), outer terciles (x ≤ q33, x > q67) — cuts
from the DISCOVERY regime, frozen for confirmation; + every 2D median-quadrant of the 40 strongest 1D pair vars (n ≥ max(15, 5 % N)).
Statistic Welch t zone vs rest. Null: DAY-block resampling (each day's outcomes replaced by a permuted donor day's), max |t| over the
whole family → adjusted p and the observed |t|'s rank vs the null distribution.
MACRO family (DAY units — window-units rule): fills aggregated per UTC day (day-mean x, day-mean y, one observation per day); same cut
granularities on day values; Welch t over days; null = day-label shuffle (max |t| over the macro family).
Survivor = adjusted p < 0.10 in discovery. CONFIRMED = same sign in the other bull regime with one-sided day-bootstrap P ≥ 0.90; for 2025
discovery the zone effect must also keep its sign inside the top-50 subset. Bear H1 Δ printed for every survivor.
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_yr6wl_frenzy_common as C  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--nperm", type=int, default=500); A = ap.parse_args()
EXCL = {"entry_frenzy_vwap", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_slippage_pct", "entry_frenzy_catchup_bars",
        "entry_frenzy_catchup_move_pct", "entry_frenzy_catchup", "entry_btc_regime_started_at", "entry_frenzy_spike_at", "entry_surge_trigger_at"}
DISC = {"2025 BULL": "2026-H2 BULL", "2026-H2 BULL": "2025 BULL"}
MINZ = 15

D = C.load()
D = D.sort_values("ts").reset_index(drop=True)
# derived fill-level vars
D["hour"] = D.ts.dt.hour.astype(float)
D["top50f"] = D.top50.astype(float)
D["lvol24"] = np.log10(pd.to_numeric(D.entry_pair_volume_24h_usd, errors="coerce").clip(lower=1))
D["fz_age_min"] = (D.ts - pd.to_datetime(D.entry_frenzy_spike_at, errors="coerce")).dt.total_seconds() / 60
D["k_ep"] = D.groupby("ep").cumcount() + 1.0
COLS = [c for c in C.entry_cols(D) if c not in EXCL] + ["hour", "top50f", "lvol24", "fz_age_min", "k_ep"]
for c in COLS:
    D[c] = C.num(D, c) if c.startswith("entry_") else D[c]


def cuts_for(x):
    x = x.dropna()
    out = []
    if len(x) == 0 or x.nunique() < 2:
        return out
    if x.nunique() == 2:
        return [("flag", ">", float(x.min()))]
    if (x > 0).mean() > 0.05 and (x < 0).mean() > 0.05:
        out.append(("sign", ">", 0.0))
    out.append(("med", ">", float(x.median())))
    q1, q2 = x.quantile([1 / 3, 2 / 3])
    out.append(("lo3", "<=", float(q1))); out.append(("hi3", ">", float(q2)))
    return out


def mask(x, op, v):
    x = np.asarray(x, float)
    return (x > v) if op == ">" else (x <= v)


def build_pair_tests(S, cols, frozen=None):
    tests = []
    for c in cols:
        cs = frozen[c] if frozen is not None else cuts_for(S[c])
        for g, op, v in cs:
            m = mask(S[c], op, v)
            m = m & S[c].notna().values
            if m.sum() >= MINZ and (~m).sum() >= MINZ:
                tests.append((c, g, op, v, None, None, None, None, m))
    return tests


def quad_tests(S, cols, frozen_med=None):
    tests = []
    N = len(S); nmin = max(MINZ, int(0.05 * N))
    meds = frozen_med or {c: float(S[c].median()) for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            xa, xb = S[a].values.astype(float), S[b].values.astype(float)
            ok = ~np.isnan(xa) & ~np.isnan(xb)
            for sa in (">", "<="):
                for sb in (">", "<="):
                    m = mask(xa, sa, meds[a]) & mask(xb, sb, meds[b]) & ok
                    if m.sum() >= nmin and (~m).sum() >= nmin:
                        tests.append((a, "quad", sa, meds[a], b, sb, meds[b], None, m))
    return tests, meds


def dedup(tests):
    seen, out = set(), []
    for t in tests:
        k = np.packbits(t[-1]).tobytes()
        if k in seen:
            continue
        seen.add(k); out.append(t)
    return out


def run_family(y, blocks, tests, nperm, seed):
    Z = np.column_stack([t[-1] for t in tests]).astype(float)
    t0 = C.tstats(Z, y[:, None])[:, 0]
    Yp = C.block_draws(y, blocks, nperm, seed)
    mx = np.empty(nperm)
    for s in range(0, nperm, 100):
        mx[s:s + 100] = np.nanmax(np.abs(C.tstats(Z, Yp[:, s:s + 100])), 0)
    padj = np.array([(mx >= abs(t)).mean() for t in t0])
    return t0, padj, mx


def zone_stats(S, m):
    y = S.y.values
    return dict(n_zone=int(m.sum()), avg_zone=y[m].mean(), wr_zone=100 * (y[m] > 0).mean(), avg_rest=y[~m].mean(), diff=y[m].mean() - y[~m].mean())


def day_agg(S, c):
    g = S.groupby("day").agg(x=(c, "mean"), y=("y", "mean"), n=("y", "size")).dropna()
    return g


rows, oos_rows, md = [], [], []
rng_seed = 100
for s in C.SLEEVES:
    for disc, conf in DISC.items():
        Sd = D[(D.sleeve == s) & (D.regime == disc)].reset_index(drop=True)
        Sc = D[(D.sleeve == s) & (D.regime == conf)].reset_index(drop=True)
        Sb = D[(D.sleeve == s) & (D.regime == "2026-H1 BEAR")].reset_index(drop=True)
        cov_ok = [c for c in COLS if Sd[c].notna().mean() >= 0.8 and Sc[c].notna().mean() >= 0.8 and Sd[c].nunique() >= 2]
        pair_cols = [c for c in cov_ok if not C.is_macro(c)]
        mac_cols = [c for c in cov_ok if C.is_macro(c)]
        y = Sd.y.values.astype(float)
        # ── PAIR family (fill units, day-block null)
        frozen = {c: cuts_for(Sd[c]) for c in pair_cols}
        t1 = dedup(build_pair_tests(Sd, pair_cols, frozen))
        Z1 = np.column_stack([t[-1] for t in t1]).astype(float)
        tt = C.tstats(Z1, y[:, None])[:, 0]
        best = {}
        for t, v in zip(t1, tt):
            best[t[0]] = max(best.get(t[0], 0), abs(v) if np.isfinite(v) else 0)
        top40 = [c for c, _ in sorted(best.items(), key=lambda kv: -kv[1])[:40]]
        t2, meds = quad_tests(Sd, top40)
        fam = dedup(t1 + t2)
        rng_seed += 1
        t0, padj, mx = run_family(y, Sd.day.values, fam, A.nperm, rng_seed)
        q95 = float(np.percentile(mx, 95))
        for t, tv, pa in zip(fam, t0, padj):
            zs = zone_stats(Sd, t[-1])
            rows.append(dict(sleeve=s, disc=disc, family="PAIR", var=t[0], gran=t[1], op=t[2], cut=t[3], var2=t[4], op2=t[5], cut2=t[6],
                             t=tv, p_adj=pa, null_q95=q95, n_tests=len(fam), **zs))
        # ── MACRO family (DAY units, day-label shuffle)
        mtests = []
        dd = {c: day_agg(Sd, c) for c in mac_cols}
        for c in mac_cols:
            g = dd[c]
            for gname, op, v in cuts_for(g.x):
                m = mask(g.x, op, v)
                if m.sum() >= 8 and (~m).sum() >= 8:
                    mtests.append((c, gname, op, v, g, m))
        if mtests:
            ref_days = sorted(set().union(*[set(t[4].index) for t in mtests]))
            Yday = Sd.groupby("day").y.mean().reindex(ref_days).values
            rngm = np.random.default_rng(rng_seed + 500)
            perms = np.column_stack([rngm.permutation(len(ref_days)) for _ in range(A.nperm)])
            T0, MX = [], np.zeros(A.nperm)
            pos = {d: i for i, d in enumerate(ref_days)}
            for c, gname, op, v, g, m in mtests:
                ii = np.array([pos[d] for d in g.index])
                Z = m.astype(float)[:, None]
                T0.append(C.tstats(Z, Yday[ii][:, None])[0, 0])
                Tp = C.tstats(Z, Yday[perms][ii, :])[0]
                MX = np.maximum(MX, np.nan_to_num(np.abs(Tp)))
            qm = float(np.percentile(MX, 95))
            for (c, gname, op, v, g, m), tv in zip(mtests, T0):
                yd = g.y.values
                rows.append(dict(sleeve=s, disc=disc, family="MACRO(day)", var=c, gran=gname, op=op, cut=v, t=tv, p_adj=float((MX >= abs(tv)).mean()),
                                 null_q95=qm, n_tests=len(mtests), n_zone=int(m.sum()), avg_zone=yd[m].mean(), wr_zone=100 * (yd[m] > 0).mean(),
                                 avg_rest=yd[~m].mean(), diff=yd[m].mean() - yd[~m].mean()))

R = pd.DataFrame(rows)
R["null_rank"] = R.groupby(["sleeve", "disc", "family"]).t.transform(lambda t: t.abs().rank(ascending=False))
R.to_csv(os.path.join(C.REP, "YR6_WL_FRENZY_screen_tests.csv"), index=False)


def apply_zone(S, r):
    """Re-apply a discovered zone (frozen cuts) on another regime's fills (fill units) → boolean mask."""
    m = mask(S[r["var"]], r["op"], r["cut"]) & S[r["var"]].notna().values
    if isinstance(r.get("var2"), str):
        m = m & mask(S[r["var2"]], r["op2"], r["cut2"]) & S[r["var2"]].notna().values
    return m


def conf_read(S, r, sign):
    if len(S) == 0:
        return dict(n=0)
    if r["family"].startswith("MACRO"):
        g = day_agg(S, r["var"])
        m = mask(g.x, r["op"], r["cut"])
        if m.sum() < 3 or (~m).sum() < 3:
            return dict(n=int(m.sum()))
        yd = g.y.values
        rng = np.random.default_rng(3); dlt = []
        idx = np.arange(len(yd))
        for _ in range(4000):
            b = rng.choice(idx, len(idx)); mb = m[b]
            if mb.sum() and (~mb).sum():
                dlt.append(yd[b][mb].mean() - yd[b][~mb].mean())
        dlt = np.array(dlt)
        return dict(n=int(m.sum()), avg_zone=yd[m].mean(), avg_rest=yd[~m].mean(), diff=yd[m].mean() - yd[~m].mean(), P=float((sign * dlt > 0).mean()))
    m = apply_zone(S, r)
    if m.sum() < 3 or (~m).sum() < 3:
        return dict(n=int(m.sum()))
    y = S.y.values
    return dict(n=int(m.sum()), avg_zone=y[m].mean(), avg_rest=y[~m].mean(), diff=y[m].mean() - y[~m].mean(),
                P=C.boot_diff_p(y, m, S.day.values, sign))


# ── survivors → OOS
for (s, disc, fam), g in R.groupby(["sleeve", "disc", "family"]):
    sv = g[g.p_adj < 0.10].sort_values("p_adj")
    conf = DISC[disc]
    for _, r in sv.iterrows():
        sign = np.sign(r["diff"])
        Sc = D[(D.sleeve == s) & (D.regime == conf)]
        Sb = D[(D.sleeve == s) & (D.regime == "2026-H1 BEAR")]
        c1, cb = conf_read(Sc, r, sign), conf_read(Sb, r, sign)
        rec = dict(sleeve=s, disc=disc, family=fam, zone=f"{r['var']} {r['op']} {r['cut']:.4g}" + (f" & {r['var2']} {r['op2']} {r['cut2']:.4g}" if isinstance(r.get('var2'), str) else ""),
                   gran=r["gran"], n_disc=r["n_zone"], avg_zone_disc=r["avg_zone"], avg_rest_disc=r["avg_rest"], t=r["t"], p_adj=r["p_adj"],
                   conf_n=c1.get("n"), conf_diff=c1.get("diff"), conf_P=c1.get("P"), bear_n=cb.get("n"), bear_diff=cb.get("diff"), bear_P=cb.get("P"))
        if disc == "2025 BULL" and not fam.startswith("MACRO"):
            S25 = D[(D.sleeve == s) & (D.regime == disc) & D.top50]
            ct = conf_read(S25, r, sign); rec.update(top50_n=ct.get("n"), top50_diff=ct.get("diff"), top50_P=ct.get("P"))
        rec["CONFIRMED"] = bool(rec["conf_diff"] is not None and pd.notna(rec["conf_diff"]) and np.sign(rec["conf_diff"]) == sign
                                and (rec["conf_P"] or 0) >= 0.90 and (disc != "2025 BULL" or fam.startswith("MACRO") or
                                                                      (pd.notna(rec.get("top50_diff")) and np.sign(rec.get("top50_diff")) == sign)))
        oos_rows.append(rec)
O = pd.DataFrame(oos_rows)
O.to_csv(os.path.join(C.REP, "YR6_WL_FRENZY_screen_oos.csv"), index=False)

# ── markdown
md.append("### Screen size and null per sleeve × discovery regime\n")
z = R.groupby(["sleeve", "disc", "family"]).agg(tests=("t", "size"), max_abs_t=("t", lambda t: np.nanmax(np.abs(t))), null_q95=("null_q95", "first"),
                                                 survivors_p10=("p_adj", lambda p: int((p < 0.10).sum())), survivors_p05=("p_adj", lambda p: int((p < 0.05).sum()))).reset_index()
md.append(C.md_table(z, 2))
md.append("\n\n### Top 8 per sleeve × discovery regime × family (by |t|), with null-adjusted p\n")
for (s, disc, fam), g in R.groupby(["sleeve", "disc", "family"]):
    g = g.reindex(g.t.abs().sort_values(ascending=False).index).head(8)
    g = g.assign(zone=[f"{r['var']} {r['op']} {r['cut']:.4g}" + (f" & {r['var2']} {r['op2']} {r['cut2']:.4g}" if isinstance(r['var2'], str) else "") for _, r in g.iterrows()])
    md.append(f"\n**{s} · discover {disc} · {fam}**\n")
    md.append(C.md_table(g[["zone", "gran", "n_zone", "wr_zone", "avg_zone", "avg_rest", "t", "p_adj"]], 3))
md.append("\n\n### Survivors (p_adj < 0.10) → OOS confirmation in the other bull regime; bear H1 shown\n")
if len(O):
    md.append(C.md_table(O, 3))
else:
    md.append("(none)")
open(os.path.join(C.REP, "YR6_WL_FRENZY_screen.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
