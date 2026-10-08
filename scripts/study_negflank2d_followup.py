#!/usr/bin/env python3
"""NEGFLANK 2D study (2026-10-08) — follow-ups on the operator's candidate (washed-out / BTC distance from 30d high) and on the
closest-to-surviving scan families. Read-only. Appends to reports/NEGFLANK_2D_STUDY_tables.md.
Requires reports/NEGFLANK_2D_features.pkl and reports/NEGFLANK_2D_sweep_*.csv (study_negflank2d_screen.py)."""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUTMD = "reports/NEGFLANK_2D_STUDY_tables.md"
L = []
NS = 3


_md = open(OUTMD).read() if os.path.exists(OUTMD) else ""
if "\n# Follow-ups" in _md:                       # idempotent: drop a previous follow-up block before appending
    open(OUTMD, "w").write(_md[:_md.index("\n# Follow-ups")] + "\n")


def say(s=""):
    print(s); L.append(s)


F = pd.read_pickle("reports/NEGFLANK_2D_features.pkl")
for c in [c for c in F.columns if c.startswith("k_") or c.startswith("entry_")]:
    if F[c].dtype == object:
        F[c] = pd.to_numeric(F[c], errors="coerce") if F[c].astype(str).str.match(r"^-?[\d.eE+-]+$|^nan$|^None$").mean() > 0.9 else F[c]
F["slope"] = pd.to_numeric(F.entry_btc_1h_slope, errors="coerce")
F["neg"] = F.slope <= -0.05
F["off30"] = F.k_btc_off30d_high
F["washk"] = F.off30 <= -15
M = F[(F.src == "master") & (F.era != "B1")].copy()
Y = F[F.src == "yr5"].copy()


def boot(p, day, n=4000, seed=7):
    g = pd.DataFrame({"p": p, "d": day}).groupby("d").p.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n, len(s)))
    mm = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(mm, 2.5), np.percentile(mm, 97.5), (mm < 0).mean()


def ln(d, ns=1):
    if len(d) == 0:
        return "0"
    lo, hi, pn = boot(d.pct.values, d.day.values)
    return f"{len(d)/ns:.0f} · {(d.pct>0).mean()*100:.0f}% · {d.pct.mean():+.3f} · ${d.usd.sum()/ns:+,.0f} · {d.day.nunique()}d · P(<0) {pn:.2f}"


def interaction(D, zone, n=2000, seed=11):
    """Δ(zone − rest) inside NEG, inside non-NEG, and their difference; day-bootstrap CI of the difference."""
    p, dy, ng = D.pct.values, D.day.values, D.neg.values
    def dd(sel):
        a, b = p[sel & zone], p[sel & ~zone]
        return a.mean() - b.mean() if len(a) >= 3 and len(b) >= 3 else np.nan
    dn, dr = dd(ng), dd(~ng)
    ud = np.unique(dy); idx = {d: np.where(dy == d)[0] for d in ud}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        ii = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
        pp, zz, gg = p[ii], zone[ii], ng[ii]
        def d2(sel):
            a, b = pp[sel & zz], pp[sel & ~zz]
            return a.mean() - b.mean() if len(a) >= 2 and len(b) >= 2 else np.nan
        out.append(d2(gg) - d2(~gg))
    out = np.array(out)
    return dn, dr, dn - dr, np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)


def be_wr(d):
    w = d[d.pct > 0].pct.mean(); l = -d[d.pct <= 0].pct.mean()
    return l / (w + l) * 100


def bar(d, sleeve, ns=1, label=""):
    """locked expectancy bar on a BLOCKED cohort d (1×: pct is leverage-invariant)."""
    if len(d) == 0:
        return f"{label}: empty"
    be = be_wr(sleeve)
    lo, hi, pn = boot(d.pct.values, d.day.values)
    loss = d[d.pct < 0]; tl = -loss.pct.sum()
    mday = (-loss.groupby("day").pct.sum()).max() / tl * 100 if tl > 0 else 0
    mpair = (-loss.groupby("pair").pct.sum()).max() / tl * 100 if tl > 0 else 0
    wr = (d.pct > 0).mean() * 100
    chk = [wr < be, pn >= 0.95, d.day.nunique() >= 8, len(d) / ns >= 15, mday < 50 and mpair < 50]
    return (f"| {label} | {len(d)/ns:.0f} | {wr:.0f}% vs BE {be:.1f}% {'✔' if chk[0] else '✗'} | {d.pct.mean():+.3f} · P(<0) {pn:.2f} {'✔' if chk[1] else '✗'} | "
            f"{d.day.nunique()} {'✔' if chk[2] else '✗'} | {'✔' if chk[3] else '✗'} | day {mday:.0f}% / pair {mpair:.0f}% {'✔' if chk[4] else '✗'} | "
            f"**{'PASS' if all(chk) else 'FAIL'}** |")


# ═════════════ A. the operator's candidate: BTC distance from 30-day high, inside NEGFLANK
say("\n# Follow-ups\n\n## A. BTC distance from its 30-day high (continuous) inside NEGFLANK\n")
say(f"rebuild k_btc_off30d_high vs live stamp: see NEGFLANK_2D_rebuild_validation.csv (r 0.997 master / 0.9999 yr5). "
    f"washed (k ≤ −15) on master = date window exactly: {int((M.washk == M.wash).all())} (1 = identical)")
BK = [(-99, -15), (-15, -10), (-10, -6), (-6, -3), (-3, 0.01)]
say("\n| BTC off 30d high | yr5 NEG N/seed · WR · avg · $/seed · days | yr5 non-NEG | master NEG (ex-B1) | master non-NEG |\n|---|---|---|---|---|")
for lo, hi in BK:
    f = lambda d: d[(d.off30 > lo) & (d.off30 <= hi)]
    say(f"| ({lo}, {hi}] | {ln(f(Y[Y.neg]), NS)} | {ln(f(Y[~Y.neg]), NS)} | {ln(f(M[M.neg]))} | {ln(f(M[~M.neg]))} |")
# washed episodes (window units)
for lab, D in [("yr5", Y), ("master", M)]:
    w = D[D.neg & D.washk].sort_values("o")
    days = pd.to_datetime(sorted(w.day.unique()))
    ep = (pd.Series(days).diff().dt.days.fillna(99) > 3).cumsum()
    eps = pd.Series(days).groupby(ep.values).agg(["min", "max", "count"])
    say(f"\n{lab}: NEG ∧ washed fills {len(w)/(NS if lab=='yr5' else 1):.0f} on {len(days)} days in **{len(eps)} episodes** (gap > 3 d splits): " +
        "; ".join(f"{a:%m-%d}→{b:%m-%d} ({c}d, {w[(w.o>=a)&(w.o<b+pd.Timedelta(days=1))].pct.mean():+.3f})" for a, b, c in eps.values))
say("\n### A2. washed vs not, inside NEG (and the same split in non-NEG longs = interaction)\n")
say("| cohort | NEG ∧ washed | NEG ∧ not washed | Δ in NEG | Δ in non-NEG | interaction (NEG − non-NEG) [95% day-CI] |\n|---|---|---|---|---|---|")
for lab, D, ns in [("yr5", Y, NS), ("master ex-B1", M, 1), ("master ex-B1 ex-B18", M[~M.b18.astype(bool)], 1)]:
    z = D.washk.values
    dn, dr, it, lo, hi = interaction(D, z)
    say(f"| {lab} | {ln(D[D.neg & D.washk], ns)} | {ln(D[D.neg & ~D.washk], ns)} | {dn:+.3f} | {dr:+.3f} | {it:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
# continuous: rank correlation within NEG
for lab, D in [("yr5", Y[Y.neg]), ("yr5 ex-washed", Y[Y.neg & ~Y.washk]), ("master", M[M.neg]), ("master ex-washed", M[M.neg & ~M.washk])]:
    r = D[["off30", "pct"]].corr("spearman").iloc[0, 1]
    dm = D.groupby("day").agg(o=("off30", "mean"), p=("pct", "mean"))
    rd = dm.corr("spearman").iloc[0, 1]
    say(f"- Spearman(off30, pct) {lab}: fills {r:+.3f} · day-means {rd:+.3f} ({len(dm)} days)")
S = pd.concat([pd.read_csv("reports/NEGFLANK_2D_sweep_yr5_all.csv").assign(scan="all"), pd.read_csv("reports/NEGFLANK_2D_sweep_yr5_exwash.csv").assign(scan="exwash")])
o30 = S[S["mask"].str.startswith("k_btc_off30d_high") & (S.kind == "1D")]
say("\nwhere the 1D off30 masks sit in the scans:\n\n| scan | mask | yr5 N/seed · zone avg vs rest · z_cl · z_w | scan p (cl / welch) | H1/H2 | master N · Δ |\n|---|---|---|---|---|---|")
for _, r in o30.iterrows():
    say(f"| {r.scan} | {r['mask']} | {r.n_zone:.0f} · {r.zone_avg:+.3f} vs {r.rest_avg:+.3f} · {r.z:+.2f} · {r.z_welch:+.2f} | {r.scan_p:.2f} / {r.scan_p_welch:.2f} | {r.H1:+.3f}/{r.H2:+.3f} | {r.m_n} · {r.m_delta:+.3f} |")

say("\n### A3. candidate rule \"block NEGFLANK long unless washed-out (BTC ≤ −15 % below 30d high)\"\n")
for lab, D, ns in [("master ex-B1", M, 1), ("master ex-B1 ex-B18", M[~M.b18.astype(bool)], 1), ("B18", M[M.b18.astype(bool)], 1), ("yr5", Y, NS)]:
    blk = D[D.neg & ~D.washk]
    say(f"- {lab}: sleeve before {len(D)/ns:.0f} fills · ${D.usd.sum()/ns:+,.0f} · {D.pct.mean():+.3f}%/fill → after {(len(D)-len(blk))/ns:.0f} · "
        f"${(D.usd.sum()-blk.usd.sum())/ns:+,.0f} · {D.drop(blk.index).pct.mean():+.3f}%/fill · blocked {len(blk)/ns:.0f} fills worth ${blk.usd.sum()/ns:+,.0f} "
        f"→ Δ ${-blk.usd.sum()/ns:+,.0f} (30–50 % haircut: ${-blk.usd.sum()/ns*0.5:+,.0f} … ${-blk.usd.sum()/ns*0.7:+,.0f})")
say("\nlocked expectancy bar on the BLOCKED side (NEG ∧ not washed), BE WR from the whole sleeve's kept fills:\n")
say("| cohort | N | WR vs BE | avg · day-clustered P(mean<0) | days ≥ 8 | N ≥ 15 | concentration < 50 % | verdict |\n|---|---|---|---|---|---|---|---|")
say(bar(Y[Y.neg & ~Y.washk], Y, NS, "yr5"))
say(bar(M[M.neg & ~M.washk], M, 1, "master ex-B1"))
say(bar(M[M.neg & ~M.washk & ~M.b18.astype(bool)], M[~M.b18.astype(bool)], 1, "master ex-B1 ex-B18"))

# ═════════════ B. closest-to-surviving families
say("\n## B. closest-to-surviving families (none passes the scan null)\n")
FAM = {
    "B1 pair EMA20/50 gap ≤ 0.238 ∧ BTC 1d EMA20 slope > 0.281 (ex-wash scan #1 by Welch)":
        lambda d: (d.entry_pair_ema20_ema50_gap_pct <= 0.238) & (d.k_btc_1d_slope > 0.2811),
    "B2 pair EMA20/50 gap ≤ 0.238 ∧ BTC above its 1d EMA20":
        lambda d: (d.entry_pair_ema20_ema50_gap_pct <= 0.238) & (d.k_btc_vs_1d_ema20 > -0.0114),
    "B3 BTC 1h slope falling further over 3 h (chg3h ≤ −0.014) ∧ pair EMA50 slope > 0.1685 [winner side]":
        lambda d: (d.entry_ema50_slope > 0.1685) & (d.k_btc_slope1h_chg3h <= -0.01405),
    "B4 BTC 1h slope negative > 9 h (hrs_since_slope_neg > 9)":
        lambda d: d.k_btc_hrs_since_slope_neg > 9,
    "B5 BTC dominance proxy > 0.61 (BTC 24h − median alt 24h) ∧ ETH 1h slope > −0.246 [winner side]":
        lambda d: (d.k_btc_dom24 > 0.6118) & (d.k_eth_slope1h > -0.2463),
    "B6 global volume ratio high (> 0.95, post-hoc from the B18 study)":
        lambda d: d.entry_global_volume_ratio > 0.95,
}
for name, fn in FAM.items():
    say(f"\n### {name}\n")
    say("| cohort | zone (NEG ∧ X) | NEG ∧ ¬X | Δ in NEG | Δ in non-NEG | interaction [95% day-CI] |\n|---|---|---|---|---|---|")
    for lab, D, ns in [("yr5 all", Y, NS), ("yr5 ex-washed", Y[~Y.washk], NS), ("master ex-B1", M, 1), ("master ex-B18", M[~M.b18.astype(bool)], 1),
                       ("master ex-washed", M[~M.washk], 1), ("master ex-washed ex-B18", M[~M.washk & ~M.b18.astype(bool)], 1)]:
        z = fn(D).fillna(False).values
        dn, dr, it, lo, hi = interaction(D, z)
        say(f"| {lab} | {ln(D[D.neg & z], ns)} | {ln(D[D.neg & ~z], ns)} | {dn:+.3f} | {dr:+.3f} | {it:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    YN = Y[Y.neg]; z = fn(YN).fillna(False)
    zl = YN[z]; loss = zl[zl.pct < 0]; tl = -loss.pct.sum()
    say(f"- yr5 NEG zone concentration: largest day {(-loss.groupby('day').pct.sum()).max()/tl*100:.0f}% · largest pair {(-loss.groupby('pair').pct.sum()).max()/tl*100:.0f}% of zone loss; "
        f"overlap with washed: {zl.washk.mean()*100:.0f}% of zone vs {YN[~z].washk.mean()*100:.0f}% of rest")
    b18 = M[M.b18.astype(bool)]
    say(f"- B18 fills in zone: {', '.join(f'{r.pair[:-4]}={bool(fn(b18.loc[[i]]).fillna(False).iloc[0])}' for i, r in b18.iterrows())}")

# dose-response for B1 (2D grid of terciles, yr5 NEG ex-washed)
say("\n### B1 dose-response — yr5 NEG ex-washed, avg pct by tercile grid (N/seed)\n")
D = Y[Y.neg & ~Y.washk].copy()
D["pg"] = pd.qcut(D.entry_pair_ema20_ema50_gap_pct, 3, labels=["pair gap low", "mid", "high"])
D["bs"] = pd.qcut(D.k_btc_1d_slope, 3, labels=["BTC 1d slope low", "mid", "high"])
T = D.groupby(["pg", "bs"], observed=True).pct.agg(["mean", "count"])
say("| | " + " | ".join(D.bs.cat.categories) + " |\n|---|---|---|---|")
for a in D.pg.cat.categories:
    say(f"| {a} | " + " | ".join(f"{T.loc[(a, b), 'mean']:+.3f} ({T.loc[(a, b), 'count']/NS:.0f})" for b in D.bs.cat.categories) + " |")
D = M[M.neg & ~M.washk].copy()
say(f"\nmaster NEG ex-washed, same cuts frozen from yr5 (pair gap ≤ 0.238 / BTC 1d slope > 0.281):")
for a, fa in [("pair gap ≤ 0.238", D.entry_pair_ema20_ema50_gap_pct <= 0.238), ("pair gap > 0.238", D.entry_pair_ema20_ema50_gap_pct > 0.238)]:
    for b, fb in [("1d slope > 0.281", D.k_btc_1d_slope > 0.2811), ("1d slope ≤ 0.281", D.k_btc_1d_slope <= 0.2811)]:
        say(f"- {a} ∧ {b}: {ln(D[fa & fb])}")

# B18 values on the key variables
say("\n## C. B18's three fills on the key variables\n")
KEYV = ["slope", "off30", "k_btc_1d_slope", "k_btc_vs_1d_ema20", "entry_pair_ema20_ema50_gap_pct", "entry_ema50_slope", "k_btc_slope1h_chg3h",
        "k_btc_hrs_since_slope_neg", "k_btc_dom24", "k_eth_slope1h", "entry_global_volume_ratio", "k_btc_4h_gap20_50", "k_btc_day_ret",
        "k_pair_gap1h_20_200", "k_alt_up_share24", "pct"]
b18 = M[M.b18.astype(bool)][["pair"] + KEYV]
say("| var | " + " | ".join(b18.pair.str[:-4]) + " | yr5 NEG median | master NEG median |\n|---|---|---|---|---|---|")
for v in KEYV:
    say(f"| {v} | " + " | ".join(f"{x:+.3f}" for x in b18[v]) + f" | {Y[Y.neg][v].median():+.3f} | {M[M.neg][v].median():+.3f} |")

open(OUTMD, "a").write("\n".join(L) + "\n")

# ═════════════ D. extra reads on the B1/B2 family (one-variable legs, per seed, rule arithmetic, bar)
L.clear()
say(f"\n## D. B1/B2 family — legs, seeds, rule arithmetic, bar\n")
say(f"washed flag rebuilt (off30 ≤ −15) vs Jun-18→Jul-2 date window on master: {int((M.washk != M.wash).sum())} fill(s) differ")
for v in ["entry_pair_ema20_ema50_gap_pct", "k_btc_1d_slope", "k_btc_vs_1d_ema20"]:
    r = S[(S.kind == "1D") & S["mask"].str.startswith(v + " ")]
    say(f"\n1D masks of {v}:\n\n| scan | mask | yr5 N/seed · zone vs rest · z_w | scan p welch | H1/H2 | master N · Δ |\n|---|---|---|---|---|---|")
    for _, x in r.iterrows():
        say(f"| {x.scan} | {x['mask']} | {x.n_zone:.0f} · {x.zone_avg:+.3f} vs {x.rest_avg:+.3f} · {x.z_welch:+.2f} | {x.scan_p_welch:.2f} | {x.H1:+.3f}/{x.H2:+.3f} | {x.m_n} · {x.m_delta:+.3f} |")
say("\nBTC 1d EMA20 slope quintiles inside NEG (yr5 cuts):\n\n| bucket | yr5 NEG | master NEG ex-B1 |\n|---|---|---|")
q = Y[Y.neg].k_btc_1d_slope.quantile([0.2, 0.4, 0.6, 0.8]).values
edges = [-99] + list(q) + [99]
for a, b in zip(edges[:-1], edges[1:]):
    f = lambda d: d[d.neg & (d.k_btc_1d_slope > a) & (d.k_btc_1d_slope <= b)]
    say(f"| ({a:+.2f}, {b:+.2f}] | {ln(f(Y), NS)} | {ln(f(M))} |")
B1f = FAM["B1 pair EMA20/50 gap ≤ 0.238 ∧ BTC 1d EMA20 slope > 0.281 (ex-wash scan #1 by Welch)"]
B2f = FAM["B2 pair EMA20/50 gap ≤ 0.238 ∧ BTC above its 1d EMA20"]
for nm, fn in [("B1", B1f), ("B2", B2f)]:
    YN = Y[Y.neg]; z = fn(YN).fillna(False)
    say(f"\n{nm} per seed (zone avg vs rest): " + " · ".join(f"s{s}: {YN[z & (YN.seed == s)].pct.mean():+.3f} vs {YN[~z & (YN.seed == s)].pct.mean():+.3f}" for s in (1, 2, 3)))
    mons = YN.o.dt.strftime("%m")
    say(f"{nm} per month Δ (zone − rest, yr5 NEG): " + " · ".join(f"{m}: {YN[z & (mons == m)].pct.mean() - YN[~z & (mons == m)].pct.mean():+.2f} (n{(z & (mons == m)).sum()//NS})"
                                                    for m in sorted(mons.unique())))
    say(f"\nrule \"block NEGFLANK long when {nm}\":")
    for lab, D, ns in [("master ex-B1", M, 1), ("master ex-B1 ex-B18", M[~M.b18.astype(bool)], 1), ("B18", M[M.b18.astype(bool)], 1), ("yr5", Y, NS)]:
        blk = D[D.neg & fn(D).fillna(False)]
        say(f"- {lab}: before {len(D)/ns:.0f} · ${D.usd.sum()/ns:+,.0f} · {D.pct.mean():+.3f}%/fill → after {(len(D)-len(blk))/ns:.0f} · ${(D.usd.sum()-blk.usd.sum())/ns:+,.0f} · "
            f"{D.drop(blk.index).pct.mean():+.3f}%/fill · blocked {len(blk)/ns:.0f} worth ${blk.usd.sum()/ns:+,.0f} → Δ ${-blk.usd.sum()/ns:+,.0f} "
            f"(haircut ${-blk.usd.sum()/ns*0.5:+,.0f} … ${-blk.usd.sum()/ns*0.7:+,.0f})")
    say("\n| cohort | N | WR vs BE | avg · P(mean<0) | days ≥ 8 | N ≥ 15 | concentration | verdict |\n|---|---|---|---|---|---|---|---|")
    say(bar(Y[Y.neg & fn(Y).fillna(False)], Y, NS, f"{nm} yr5"))
    say(bar(M[M.neg & fn(M).fillna(False)], M, 1, f"{nm} master ex-B1"))
    say(bar(M[M.neg & fn(M).fillna(False) & ~M.b18.astype(bool)], M[~M.b18.astype(bool)], 1, f"{nm} master ex-B18"))
open(OUTMD, "a").write("\n".join(L) + "\n")
