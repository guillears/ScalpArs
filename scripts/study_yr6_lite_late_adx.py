#!/usr/bin/env python3
"""LITE late-in-run × falling ADX (operator 2026-10-13, after the SKL 10-13 08:10 LITE stop): yr6 two-year replay re-priced to today
(reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv), FRENZY_LITE seed 1, y = pct_new (+3 / −3 / 12 h, 10×). Pre-declared cuts (no fitting):
LATE = entry_frenzy_hours ≥ 12 (LITE window 2 → 16.75 h; also the outer tercile), ADX_FALL = entry_frenzy_adx_delta < 0 (sign; and
entry_adx_delta < 0), plus the SKL signature (both). Per regime: zone vs rest, day-bootstrap P(zone − rest < 0), 2025 top-50 split,
day-shift null on the zone-minus-rest gap. Output reports/YR6_LITE_LATE_ADX.txt."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(13)
F = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv"), low_memory=False)
F = F[(F.sleeve == "FRENZY_LITE") & (F.seed == 1)].copy()
F["t"] = pd.to_datetime(F.t, format="mixed"); F["day"] = F.t.dt.strftime("%Y-%m-%d"); F["y"] = F.pct_new.astype(float)
U = pd.concat([pd.read_csv(os.path.join(ROOT, p)) for p in ("reports/backtest_cache/universe_daily_top50.csv", "reports/backtest_cache_2025/universe_daily_top50.csv")])
U["d"] = pd.to_datetime(U.date_ms, unit="ms").dt.strftime("%Y-%m-%d")
top = set(zip(U.d, U.pair)) | {(str((pd.Timestamp(d) + pd.Timedelta(days=1)).date()), p) for d, p in zip(U.d, U.pair)}
F["top50"] = [(d, p) in top for d, p in zip(F.day, F.pair)]
h = pd.to_numeric(F.entry_frenzy_hours, errors="coerce"); fa = pd.to_numeric(F.entry_frenzy_adx_delta, errors="coerce"); pa = pd.to_numeric(F.entry_adx_delta, errors="coerce")
T3 = h.quantile(2 / 3)
Z = {"LATE (hours ≥ 12)": h >= 12, f"LATE tercile (hours ≥ {T3:.1f})": h >= T3, "FRENZY ADXΔ < 0": fa < 0, "pair ADXΔ < 0": pa < 0,
     "LATE ∧ FRENZY ADXΔ < 0 (SKL)": (h >= 12) & (fa < 0), "LATE ∧ pair ADXΔ < 0": (h >= 12) & (pa < 0)}
cov = {"hours": h.notna().mean(), "frenzy_adx_delta": fa.notna().mean(), "adx_delta": pa.notna().mean()}


def gap_stats(g, z):
    a, b = g[z], g[~z]
    if len(a) < 5 or len(b) < 5:
        return None
    days = g.day.unique(); dz = g.assign(z=z.values)
    agg = dz.groupby(["day", "z"]).y.agg(["sum", "size"]).unstack(fill_value=0)
    idx = rng.integers(0, len(agg), size=(4000, len(agg)))
    S1, N1 = agg["sum"][True].values, agg["size"][True].values; S0, N0 = agg["sum"][False].values, agg["size"][False].values
    with np.errstate(invalid="ignore", divide="ignore"):
        d = S1[idx].sum(1) / N1[idx].sum(1) - S0[idx].sum(1) / N0[idx].sum(1)
    # null: shuffle the zone label within each day-block of the regime (keeps day clustering of outcomes)
    obs = a.y.mean() - b.y.mean(); nul = []
    zz = z.values.copy()
    for _ in range(300):
        p = rng.permutation(zz); nul.append(g.y[p].mean() - g.y[~p].mean())
    return dict(nz=len(a), wz=(a.y > 0).mean() * 100, az=a.y.mean(), nr=len(b), ar=b.y.mean(), gap=obs,
                p_worse=float(np.nanmean(d < 0)), null_p=float(np.mean(np.array(nul) <= obs)), days=a.day.nunique())


L = [__doc__.splitlines()[0], f"coverage: {', '.join(f'{k} {v * 100:.0f} %' for k, v in cov.items())} · LITE seed-1 fills {len(F)} · late cut tercile {T3:.2f} h", ""]
for name, z in Z.items():
    L.append(f"■ {name}")
    for rg, g in [("2025 BULL", F[F.regime == "2025 BULL"]), ("2025 BULL top-50", F[(F.regime == "2025 BULL") & F.top50]),
                  ("2026-H1 BEAR", F[F.regime == "2026-H1 BEAR"]), ("2026-H2 BULL", F[F.regime == "2026-H2 BULL"]), ("ALL", F)]:
        s = gap_stats(g.reset_index(drop=True), z.loc[g.index].reset_index(drop=True).fillna(False).astype(bool))
        if s is None:
            L.append(f"   {rg:17} n/a"); continue
        L.append(f"   {rg:17} zone {s['nz']:4} ({s['days']:3} d) WR {s['wz']:4.1f} % avg {s['az']:+.3f} │ rest {s['nr']:4} avg {s['ar']:+.3f} │ gap "
                 f"{s['gap']:+.3f} · P(zone worse) {s['p_worse']:.2f} · label-shuffle p {s['null_p']:.3f}")
    L.append("")
txt = "\n".join(L); open(os.path.join(ROOT, "reports", "YR6_LITE_LATE_ADX.txt"), "w").write(txt + "\n"); print(txt)
