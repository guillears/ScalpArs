#!/usr/bin/env python3
"""📊 2026-10-19 research (WILLY turnover filter re-price) — statistics on the re-priced cohorts (no network).
Inputs: reports/WILLY_TURNOVER_REPRICE_2026-10-19_willyA.csv (study_willy_turnover_reprice.py) and _live.csv (…_live.py).
Writes reports/WILLY_TURNOVER_REPRICE_2026-10-19_grid.csv and prints every table of the report.
Usage: venv/bin/python -I scripts/study_willy_turnover_reprice_analyze.py"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RP = os.path.join(ROOT, "reports")
RNG = np.random.default_rng(20261019)
NB = 4000
BUCKETS = ((0, 50e6, "< $50M"), (50e6, 100e6, "$50–100M"), (100e6, 200e6, "$100–200M"), (200e6, np.inf, "≥ $200M"))


def bucket(m):
    if not np.isfinite(m):
        return "unknown"
    return next(l for a, b, l in BUCKETS if a <= m < b)


def dayboot(df, col="pct", nb=NB):
    """day-block bootstrap of the mean → (lo, hi, P(mean<0))."""
    if len(df) < 2:
        return np.nan, np.nan, np.nan
    g = df.groupby("day")[col].agg(["sum", "size"])
    s, n = g["sum"].values, g["size"].values
    k = len(g)
    idx = RNG.integers(0, k, size=(nb, k))
    m = s[idx].sum(1) / n[idx].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), float((m < 0).mean())


def stats(df, col="pct"):
    p = df[col]
    if not len(df):
        return dict(N=0)
    lo, hi, pneg = dayboot(df, col)
    loss = df[p < 0]
    tl = -loss[col].sum()
    tp_share = (loss.groupby("pair")[col].sum().min() * -1 / tl) if tl > 0 else np.nan
    td_share = (loss.groupby("day")[col].sum().min() * -1 / tl) if tl > 0 else np.nan
    top_pair_n = df.pair.value_counts().iloc[0] / len(df)
    return dict(N=len(df), days=df.day.nunique(), WR=(p > 0).mean() * 100, avg=p.mean(), lo=lo, hi=hi, Pneg=pneg,
                top_pair_loss=tp_share * 100, top_day_loss=td_share * 100, top_pair_n=top_pair_n * 100, top_pair=df.pair.value_counts().index[0])


def fmt(lab, s):
    if not s.get("N"):
        return f"| {lab} | 0 | | | | | | | |"
    return (f"| {lab} | {s['N']} | {s['days']} | {s['WR']:.0f} % | {s['avg']:+.3f} | [{s['lo']:+.2f}, {s['hi']:+.2f}] | {s['Pneg']:.2f} | "
            f"{s['top_pair_loss']:.0f} % / {s['top_day_loss']:.0f} % | {s['top_pair']} {s['top_pair_n']:.0f} % |")


HDR = ("| cohort | N | days | WR | avg % | day-boot 95 % CI | P(avg<0) | top pair / day share of loss | top pair share of N |\n"
       "|---|---|---|---|---|---|---|---|---|")


def be_wr(p):
    w, l = p[p > 0].mean(), -p[p < 0].mean()
    return l / (w + l) * 100, w, l


def main():
    A = pd.read_csv(os.path.join(RP, "WILLY_TURNOVER_REPRICE_2026-10-19_willyA.csv"))
    A["month"] = A.day.str[:7]
    A["half"] = np.where(A.day < "2026-07-01", "2026-H1", "2026-H2")
    print("rows", len(A), A.st.value_counts().to_dict())
    ok = A[A.st == "ok"].copy()
    print("path", ok.path.value_counts().to_dict(), "entry_src", ok.entry_src.value_counts().to_dict(),
          "decided on a 1m bar", int(ok.dec_on_bar.fillna(False).astype(bool).sum()))
    print("wait bars", ok.wait_bars.describe().round(2).to_dict())
    ok["R"] = ok.R_red
    ok["mcap"] = ok.mcap_red
    sc = ok[ok.R.notna() & ok.mcap.notna()].copy()
    print(f"scored {len(sc)} of {len(ok)} priced; unscored avg {ok[ok.R.isna() | ok.mcap.isna()].pct.mean():+.3f}")
    print("R_red vs R_t0: spearman", round(sc[["R", "R_t0"]].corr("spearman").iloc[0, 1], 3), "side flips (R>=1)",
          int(((sc.R >= 1) != (sc.R_t0 >= 1)).sum()))
    be, w, l = be_wr(ok.pct)
    print(f"\nWHOLE priced cohort: {stats(ok)}\nbreakeven WR {be:.1f} % (avg win {w:+.3f}, avg loss {-l:+.3f})")
    print("exit mix", ok.why.value_counts().to_dict())
    sc["blk"] = sc.R >= 1.0
    sc["bk"] = sc.mcap.apply(bucket)

    print("\n### A. blocked (R ≥ 1) vs kept, today's entry + exit\n" + HDR)
    for lab, g in (("ALL scored", sc), ("kept R < 1", sc[~sc.blk]), ("BLOCKED R ≥ 1", sc[sc.blk])):
        print(fmt(lab, stats(g)))
    print("\n### B. per mcap bucket\n" + HDR)
    for _, _, b in BUCKETS:
        g = sc[sc.bk == b]
        print(fmt(f"{b} kept", stats(g[~g.blk])))
        print(fmt(f"{b} BLOCKED", stats(g[g.blk])))
    print("\n### C. per half / month — kept vs blocked avg % (N)")
    for key in ("half", "month"):
        t = sc.groupby([key, "blk"]).pct.agg(["size", "mean"]).unstack()
        print(t.round(3).to_string())
    for _, _, b in BUCKETS[:1]:
        g = sc[sc.bk == b]
        print("\n< $50M by half:\n", g.groupby(["half", "blk"]).pct.agg(["size", "mean"]).unstack().round(3).to_string())

    # immediate-entry variant + old exit, same rows
    print("\n### D. decomposition (same scored rows; R ≥ 1 vs < 1 avg %)")
    for col, lab in (("old_pct", "OLD study: immediate entry, +1/−3/60"), ("imm_pct", "immediate entry, TODAY's exit"),
                     ("pct", "red-bar entry, TODAY's exit (= live)")):
        g = sc[sc[col].notna()]
        k, b = g[~g.blk][col], g[g.blk][col]
        lo, hi, pn = dayboot(g[g.blk].assign(pct=g[g.blk][col]))
        print(f"| {lab} | {len(k)} · {k.mean():+.3f} | {len(b)} · {b.mean():+.3f} [{lo:+.2f}, {hi:+.2f}] P<0 {pn:.2f} | Δ {k.mean() - b.mean():+.3f} |")

    # grid
    rows = []
    Ntot = len(sc)
    for rc in (1.0, 1.5, 2.0, 3.0):
        for mc_lab, mc in (("none", np.inf), ("< $50M", 50e6), ("< $100M", 100e6)):
            blk = (sc.R >= rc) & (sc.mcap < mc)
            g = sc[blk]
            s = stats(g)
            gain = -g.pct.sum() / Ntot
            # day-boot CI of gain per fill (whole cohort)
            gs = sc.assign(x=np.where(blk, -sc.pct, 0.0))
            lo, hi, pneg = dayboot(gs, "x")
            beb = be_wr(sc[~blk].pct)[0]
            h = {hh: (sc[blk & (sc.half == hh)].pct.mean(), int((blk & (sc.half == hh)).sum())) for hh in ("2026-H1", "2026-H2")}
            mo = sc[blk].groupby("month").pct.mean()
            rows.append(dict(R_cut=rc, mcap_scope=mc_lab, **{k: s.get(k) for k in ("N", "days", "WR", "avg", "lo", "hi", "Pneg",
                                                                                   "top_pair_loss", "top_day_loss", "top_pair_n")},
                             breakeven_WR=be_wr(sc.pct)[0], gain_per_fill=gain, gain_lo=lo, gain_hi=hi, P_gain_le0=pneg,
                             H1_avg=h["2026-H1"][0], H1_N=h["2026-H1"][1], H2_avg=h["2026-H2"][0], H2_N=h["2026-H2"][1],
                             months_neg=int((mo < 0).sum()), months=int(len(mo))))
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(RP, "WILLY_TURNOVER_REPRICE_2026-10-19_grid.csv"), index=False)
    print("\n### E. grid (blocked side; gain = −Σ blocked / N all scored)")
    print(G.round(3).to_string(index=False))

    # shuffled-label null (within-day shuffle of pct, R / mcap fixed): max gain over the 12 cells
    def maxgain(pct):
        best = -np.inf
        for rc in (1.0, 1.5, 2.0, 3.0):
            for mc in (np.inf, 50e6, 100e6):
                blk = ((sc.R.values >= rc) & (sc.mcap.values < mc))
                best = max(best, -pct[blk].sum() / Ntot)
        return best
    obs = maxgain(sc.pct.values)
    days = sc.day.values
    grp = [np.flatnonzero(days == d) for d in np.unique(days)]
    nul_day, nul_all = [], []
    base = sc.pct.values
    for _ in range(2000):
        x = base.copy()
        for ix in grp:
            if len(ix) > 1:
                x[ix] = x[RNG.permutation(ix)]
        nul_day.append(maxgain(x))
        nul_all.append(maxgain(RNG.permutation(base)))
    nul_day, nul_all = np.array(nul_day), np.array(nul_all)
    print(f"\n### F. null: observed best gain {obs:+.4f}/fill · within-day shuffle p {(nul_day >= obs).mean():.3f} (95th {np.percentile(nul_day, 95):+.4f}) · "
          f"global shuffle p {(nul_all >= obs).mean():.3f} (95th {np.percentile(nul_all, 95):+.4f})")
    # same null for the fixed pre-registered cell R>=1/all (single test)
    blk1 = sc.R.values >= 1.0
    obs1 = -base[blk1].sum() / Ntot
    n1 = []
    for _ in range(2000):
        x = base.copy()
        for ix in grp:
            if len(ix) > 1:
                x[ix] = x[RNG.permutation(ix)]
        n1.append(-x[blk1].sum() / Ntot)
    print(f"pre-registered R ≥ 1 / all: gain {obs1:+.4f} · within-day null p {(np.array(n1) >= obs1).mean():.3f}")

    # LOMO for R>=1 & <50M and R>=1 all
    print("\n### G. leave-one-month-out: blocked avg without each month")
    for lab, m in (("R≥1 all", sc.R >= 1), ("R≥1 <$50M", (sc.R >= 1) & (sc.mcap < 50e6))):
        out = {mo: round(sc[m & (sc.month != mo)].pct.mean(), 3) for mo in sorted(sc.month.unique())}
        print(lab, out)

    # haircut on best cell
    # LIVE
    L = pd.read_csv(os.path.join(RP, "WILLY_TURNOVER_REPRICE_2026-10-19_live.csv"))
    L["day"] = L.t.str[:10]
    L["pct"] = L.cf_pct
    L["bk"] = L.mcap.apply(lambda v: bucket(float(v)) if pd.notna(v) else "unknown")
    b = L[L.kind == "blocked"].head(15)
    f = L[L.kind == "fill"]
    print("\n### H. LIVE (today's exit re-walk)\n" + HDR)
    print(fmt("live BLOCKED first 15 (gate cohort)", stats(b)))
    print(fmt("live BLOCKED, scout's price", stats(b.assign(pct=b.scout_pct))))
    print(fmt("live BLOCKED +0.035 slip", stats(b.assign(pct=b.cf_pct_slip))))
    print(fmt("live let-through fills (walker)", stats(f)))
    print(fmt("live let-through fills (booked)", stats(f.assign(pct=f.booked_pct))))
    for _, _, bb in BUCKETS:
        print(fmt(f"blocked {bb}", stats(b[b.bk == bb])))
        print(fmt(f"let-through {bb}", stats(f[f.bk == bb])))
    # parity
    cls = lambda w: {"FRENZY_TP": "TP", "WILLY_LATE_TP": "LATE_TP", "STOP_LOSS": "STOP", "MAX_HOLD_TIME": "CAP"}.get(w, w)
    f = f.assign(bcls=f.booked_why.map(cls))
    post = f[f.t >= "2026-10-15T16:15"]
    print("\nparity post-297 (exit identical to today):", len(post), "same class", int((post.bcls == post.cf_why).sum()),
          "mean |Δ|", round((post.cf_pct - post.booked_pct).abs().mean(), 3), "Σ walker", round(post.cf_pct.sum(), 2), "Σ booked", round(post.booked_pct.sum(), 2))
    st = f[f.bcls.isin(["STOP"])]
    print("stops (all eras, −4 stop since DL 271):", len(st), "walker STOP", int((st.cf_why == "STOP").sum()), "mean walker", round(st.cf_pct.mean(), 3), "booked", round(st.booked_pct.mean(), 3))
    cf = pd.read_csv(os.path.join(RP, "WILLY_EXIT_CF_PRE356.csv"))   # the pre-356 table (WILLY_EXIT_CF.csv is V2b since DL 377)
    m = f.merge(cf, left_on=["t", "pair"], right_on=["opened_at", "pair"], how="inner")
    print("vs master WILLY_EXIT_CF_PRE356.csv:", len(m), "rows, max |walker − cf|", round((m.cf_pct_x - m.cf_pct_y).abs().max(), 4))
    # tick share
    print("live paths", L.cf_path.value_counts().to_dict())


if __name__ == "__main__":
    main()
