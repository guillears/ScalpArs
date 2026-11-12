#!/usr/bin/env python3
"""ML EXIT × MACRO study (2026-11-11) — the read: fidelity, decomposition, every variant before → after (Δ %/fill, day-block bootstrap with
Bonferroni, winners cut / losers changed, per-era groups, leave-one-out, one-size DCR). Reads reports/ML_EXIT_MACRO_2026-11-11_{variants,fills}.csv.
X (one size, DL 503) = Momentum Long at today's C5 size 1 × 10× ≈ 2.43 × the account per 1 % move (≈ 1.7 % per −0.70 stop, DL 528)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import review_math as RM  # noqa: E402

IN = os.path.join(ROOT, "reports", "ML_EXIT_MACRO_2026-11-11")
X = 2.43
GROUPS = {"BASE": "G0 BASE", "B1": "G1 B1-B3", "B2": "G1 B1-B3", "B3": "G1 B1-B3", "B25": "G4 B25"}
for i in range(4, 17):
    GROUPS[f"B{i}"] = "G2 B4-B16"
for i in range(17, 25):
    GROUPS[f"B{i}"] = "G3 B17-B24"
REPS, SEED = 4000, 20261111


def boot_ci(delta, day, alpha):
    rng = np.random.default_rng(SEED)
    days = np.unique(day)
    idx = {d: np.flatnonzero(day == d) for d in days}
    n = len(delta)
    out = np.empty(REPS)
    for r in range(REPS):
        pick = rng.choice(days, len(days))
        ii = np.concatenate([idx[d] for d in pick])
        out[r] = delta[ii].sum() / n
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return lo, hi, float((out > 0).mean())


def dcr_group(F, pct_col, exit_col, eras):
    """one-size DCR over every calendar day of each era in `eras` (concatenated)."""
    rs = []
    for e in eras:
        g = F[F.era == e]
        if not len(g):
            continue
        d0 = pd.Timestamp(g.t0.iloc[0]).tz_convert(None).normalize()
        d1 = pd.Timestamp(g.t1.iloc[0]).tz_convert(None).normalize()
        days = pd.date_range(d0, d1, freq="D")
        cd = pd.to_datetime(g[exit_col], unit="ms").dt.normalize()
        r = (g[pct_col].fillna(0) * X / 100.0).groupby(cd.values).sum().reindex(days).fillna(0.0)
        rs.append(r.values)
    if not rs:
        return np.nan, 0
    r = np.concatenate(rs)
    return float(((1 + r).prod() ** (1 / len(r)) - 1) * 100), len(r)


def main():
    V = pd.read_csv(IN + "_variants.csv")
    F = pd.read_csv(IN + "_fills.csv")
    F["grp"] = F.era.map(GROUPS)
    F["day"] = pd.to_datetime(F.o_ms, unit="ms").dt.strftime("%Y-%m-%d")
    W = V.pivot(index="key", columns="variant", values="pct")
    R = V.pivot(index="key", columns="variant", values="reason")
    T = V.pivot(index="key", columns="variant", values="exit_ms")
    F = F.set_index("key").loc[W.index].reset_index()
    base = W["BASE"].values
    L = []
    p = L.append
    # ── fidelity
    d = base - F.y.values
    p(f"# ML EXIT × MACRO — {len(F)} kept master fills (today's stack, % at 1×)\n")
    p("## Replica fidelity (BASE replica vs the master's booked stack_pct)")
    p(f"mean Δ {np.nanmean(d):+.3f} · median |Δ| {np.nanmedian(np.abs(d)):.3f} · within ±0.05: {np.mean(np.abs(d) <= .05):.0%} · corr "
      f"{np.corrcoef(base[~np.isnan(d)], F.y.values[~np.isnan(d)])[0, 1]:.3f} · paths {F.src.value_counts().to_dict()}")
    big = F.assign(rep=base, d=d).loc[np.abs(d) > 0.15, ["era", "pair", "src", "y", "rep", "close_reason"]]
    big = big.assign(rr=R.loc[F.key[np.abs(d) > 0.15], "BASE"].values)
    p(f"|Δ| > 0.15: {len(big)} fills\n" + big.round(2).to_string(index=False) + "\n")
    # ── decomposition on the replica BASE
    win = base > 0
    p("## Where the money is (BASE replica)")
    gb = F.mfe_to_exit.values - base
    p(f"N {len(base)} · WR {win.mean():.0%} · avg {base.mean():+.3f} · Σ {base.sum():+.1f} pts · avg win {base[win].mean():+.3f} · avg loss "
      f"{base[~win].mean():+.3f} · breakeven WR {abs(base[~win].mean()) / (base[win].mean() + abs(base[~win].mean())):.0%}")
    p(f"winners: peak before exit median {np.median(F.mfe_to_exit[win]):+.2f} · giveback median {np.median(gb[win]):.2f} (Σ {gb[win].sum():.1f} pts) "
      f"· best in 20 h median {np.median(F.mfe_20h[win]):+.2f} · post-exit 60 min max median {np.nanmedian(F.post60_max[win] - base[win]):+.2f} vs exit, "
      f"4 h {np.nanmedian(F.post4h_max[win] - base[win]):+.2f}")
    p(f"losers: peak before exit median {np.median(F.mfe_to_exit[~win]):+.2f} · {np.mean(F.mfe_to_exit[~win] < 0.10):.0%} never reached +0.10 · "
      f"{np.mean(F.mfe_to_exit[~win] >= 0.40):.0%} reached +0.40 (gave back) · post-exit 4 h max ≥ +0.40 from entry: "
      f"{np.mean(F.post4h_max[~win] >= 0.40):.0%}")
    p("oracle (perfect exit at each fill's 20 h peak, losers at their stop): " f"{np.where(win, F.mfe_20h, base).mean():+.3f}/fill — the ceiling, not a target")
    for g, gg in F.assign(b=base).groupby("grp"):
        p(f"  {g}: N {len(gg)} · WR {(gg.b > 0).mean():.0%} · avg {gg.b.mean():+.3f} · booked {gg.y.mean():+.3f}")
    p("")
    # ── variants
    K = [c for c in W.columns if c != "BASE"]
    alpha = 0.05 / len(K)
    rows = []
    eras_all = sorted(F.era.unique())
    dcr_b, nd = dcr_group(F.assign(pp=base, ex=T["BASE"].values), "pp", "ex", eras_all)
    for v in K:
        y = W[v].values
        ok = ~np.isnan(y)
        dl = np.where(ok, y - base, 0.0)
        lo, hi, pp = boot_ci(dl, F.day.values, alpha)
        lo95, hi95, _ = boot_ci(dl, F.day.values, 0.05)
        ch = np.abs(dl) > 1e-9
        wc = (base > 0) & (dl < -1e-9)
        lb = (base <= 0) & (dl > 1e-9)
        gtxt = {g: dl[(F.grp == g).values].mean() for g in sorted(F.grp.unique())}
        dv, _ = dcr_group(F.assign(pp=np.where(ok, y, base), ex=np.where(ok, T[v].values, T["BASE"].values)), "pp", "ex", eras_all)
        dayd = pd.Series(dl).groupby(F.day.values).sum()
        top_day = dayd.max() / dl.sum() if dl.sum() > 0 else np.nan
        rows.append(dict(variant=v, n=int(ok.sum()), wr=np.mean(y[ok] > 0), avg=np.nanmean(y), d=dl.mean(), lo95=lo95, hi95=hi95, loB=lo, hiB=hi,
                         p_pos=pp, changed=int(ch.sum()), win_cut=int(wc.sum()), win_cut_sum=dl[wc].sum(), los_better=int(lb.sum()),
                         gain_sum=dl[dl > 0].sum(), dcr=dv, top_day=top_day, **{f"d_{g}": x for g, x in gtxt.items()},
                         loo=RM.loo_text(list(dl), list(F.pair + " " + F.day), unit="pts")))
    S = pd.DataFrame(rows).sort_values("d", ascending=False)
    S.to_csv(IN + "_summary.csv", index=False)
    p(f"## Every variant — before → after (Δ %/fill vs BASE {base.mean():+.3f}; one-size DCR BASE {dcr_b:+.3f} %/day over {nd} days, X {X})")
    p(f"Bonferroni over {len(K)} variants: two-sided α {alpha:.5f}. 'sig' = the Bonferroni day-block CI excludes 0.\n")
    cols = ["variant", "n", "wr", "avg", "d", "lo95", "hi95", "loB", "hiB", "changed", "win_cut", "win_cut_sum", "los_better", "dcr"] + \
        [c for c in S.columns if c.startswith("d_G")]
    p(S[cols].round(3).to_string(index=False))
    p("\n## Leave-one-out (Δ in pts, top 12 by Δ)")
    for r in S.head(12).itertuples():
        p(f"{r.variant}: {r.loo} · top day share {r.top_day:.0%}" if r.top_day == r.top_day else f"{r.variant}: {r.loo}")
    # ── macro: signal against us at entry (how often the 'macro that marked the entry' was already against us)
    p("\n## Macro signal state AT ENTRY (share of fills where the signal was already against the long)")
    for c in [c for c in F.columns if c.startswith("against_at_entry_")]:
        v = F[c].dropna().astype(bool)
        bw = pd.Series(base, index=F.index)[v.index]
        p(f"{c[17:]}: against {v.mean():.0%} (N {len(v)}) · BASE avg when against {bw[v].mean():+.3f} vs with {bw[~v].mean():+.3f}")
    # exit-reason mix for the macro family
    p("\n## Exit reasons, macro variants")
    for v in [k for k in K if k.startswith(("M_", "H"))]:
        p(f"{v}: {R[v].value_counts().to_dict()}")
    open(IN + ".txt", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
