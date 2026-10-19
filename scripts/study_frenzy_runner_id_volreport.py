#!/usr/bin/env python3
"""FRENZY RUNNER-ID 2026-10-19 · Volume threshold (operator) — tables from <scratch>/vol_priced.pkl + the AGT 10-17→10-19 case.

Frozen before reading (see study_frenzy_runner_id_volprice.py for entry/outcome rules):
  universe   READY bars, not dislocated, priced, bearish-day bars removed (approx. live block), opened < 2026-10-12 (cache end)
  A FIRST    the first READY bar of each T-episode (T = 50 / 70 / 85), split by what the 100x engine does in that window:
             SAME = a 100x READY bar on the same bar (lower T changes nothing) · LATER = a 100x READY bar later (≤ spike + 96 h) ·
             NEVER = no 100x READY bar → a brand-new trade
  B PAIRED   LATER episodes: enter at the T bar vs at the first later 100x bar (paired per episode): x0 / r1 / runner, price gap, hours gap
  C POLICY   the whole sleeve at T vs at 100 (every READY bar, ≤ 3 per pair per UTC day): N, avg, runner rate, Σ, daily Δ day-bootstrap
Usage: venv/bin/python scripts/study_frenzy_runner_id_volreport.py <scratch>
"""
import os, sys
import numpy as np, pandas as pd
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
S = sys.argv[1]
OUT = "reports/FRENZY_RUNNER_ID_2026-10-19"
RNG = np.random.default_rng(70)
V = pd.read_pickle(f"{S}/vol_priced.pkl")
V = V[V.ready & (V.disloc == False) & V.x0.notna()].copy()  # noqa: E712
V["ts"] = pd.to_datetime(V.t_signal_close, unit="ms")
V = V[V.ts < "2026-10-12"]
V["day"] = V.ts.dt.strftime("%Y-%m-%d")
V["per"] = np.where(V.ts < "2026-01-01", "2025", np.where(V.ts < "2026-07-01", "26H1", "26H2"))
V["run5"] = (V.mfe_720 >= 5).astype(float)
VB = V.copy()                         # incl. bearish (sensitivity)
V = V[~V.bear]


def boot(x, day, n=3000):
    df = pd.DataFrame(dict(x=x, d=day)).dropna()
    if len(df) < 3:
        return np.nan, np.nan, np.nan
    g = df.groupby("d").x.agg(["sum", "count"]); s, c = g["sum"].values, g["count"].values
    idx = RNG.integers(0, len(g), (n, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float((m > 0).mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summ(x, label, extra=None):
    P, lo, hi = boot(x.x0.values, x.day.values)
    P1, lo1, hi1 = boot(x.r1.values, x.day.values)
    e = dict(group=label, N=len(x), days=x.day.nunique(), run5=x.run5.mean() * 100, med_mfe=x.mfe_720.median(),
             x0_WR=(x.x0 > 0).mean() * 100, x0_avg=x.x0.mean(), x0_ci=f"[{lo:+.2f}, {hi:+.2f}]", x0_P=P,
             r1_avg=x.r1.mean(), r1_ci=f"[{lo1:+.2f}, {hi1:+.2f}]", r2_avg=x.r2.mean(), sum_x0=x.x0.sum())
    if extra:
        e.update(extra)
    return e


H100 = V[V["T"] == 100][["pair", "t_signal_close", "E", "x0", "r1", "r2", "run5", "mfe_720"]].sort_values("t_signal_close")
rowsA, rowsB, pairedB = [], [], []
for T in (50.0, 70.0, 85.0):
    X = V[V["T"] == T].sort_values("t_signal_close")
    F = X.groupby(["pair", "spike_ts"]).head(1).copy()
    cls, j100 = [], []
    for _, r in F.iterrows():
        h = H100[(H100.pair == r.pair) & (H100.t_signal_close >= r.t_signal_close) & (H100.t_signal_close <= r.spike_ts + 96 * 3_600_000)]
        if not len(h):
            cls.append("NEVER"); j100.append(None)
        elif int(h.t_signal_close.iloc[0]) == int(r.t_signal_close):
            cls.append("SAME"); j100.append(h.index[0])
        else:
            cls.append("LATER"); j100.append(h.index[0])
    F["cls"] = cls; F["j100"] = j100
    for per in ("2025", "26H1", "26H2", "ALL"):
        f = F if per == "ALL" else F[F.per == per]
        for c in ("SAME", "LATER", "NEVER", "ALL"):
            x = f if c == "ALL" else f[f.cls == c]
            if len(x) >= 3:
                rowsA.append(summ(x, f"T{int(T)} first-ON · {c}", dict(T=T, period=per, cls=c, share=len(x) / max(len(f), 1) * 100)))
    L = F[F.cls == "LATER"]
    for _, r in L.iterrows():
        h = H100.loc[r.j100]
        pairedB.append(dict(T=T, pair=r.pair, per=r.per, day=r.day, t_T=r.ts, hours_earlier=(h.t_signal_close - r.t_signal_close) / 3.6e6,
                            price_gap_pct=(h.E / r.E - 1) * 100, x0_T=r.x0, x0_100=h.x0, r1_T=r.r1, r1_100=h.r1, run5_T=r.run5, run5_100=h.run5,
                            vol_mult_T=r.vol_mult))
PB = pd.DataFrame(pairedB)
for T in (50.0, 70.0, 85.0):
    for per in ("2025", "26H1", "26H2", "ALL"):
        x = PB[(PB["T"] == T) & ((PB.per == per) | (per == "ALL"))]
        if len(x) < 3:
            continue
        d0 = x.x0_T - x.x0_100; d1 = x.r1_T - x.r1_100
        P0, lo0, hi0 = boot(d0.values, x.day.values); P1, lo1, hi1 = boot(d1.values, x.day.values)
        rowsB.append(dict(T=T, period=per, N=len(x), days=x.day.nunique(), med_hours_earlier=x.hours_earlier.median(),
                          med_price_gap_pct=x.price_gap_pct.median(), share_100bar_higher=(x.price_gap_pct > 0).mean() * 100,
                          run5_T=x.run5_T.mean() * 100, run5_100=x.run5_100.mean() * 100,
                          x0_T=x.x0_T.mean(), x0_100=x.x0_100.mean(), d_x0=d0.mean(), d_x0_ci=f"[{lo0:+.2f}, {hi0:+.2f}]", P_d_x0_pos=P0,
                          r1_T=x.r1_T.mean(), r1_100=x.r1_100.mean(), d_r1=d1.mean(), d_r1_ci=f"[{lo1:+.2f}, {hi1:+.2f}]"))


# C: policy (every READY bar, ≤ 3 per pair per UTC day)
def cap3(X):
    X = X.sort_values("t_signal_close")
    return X.groupby(["pair", "day"]).head(3)


rowsC = []
for src, VV in (("bearish removed", V), ("incl. bearish days", VB)):
    P100 = cap3(VV[VV["T"] == 100])
    for T in (50.0, 70.0, 85.0, 100.0):
        PT = cap3(VV[VV["T"] == T])
        for per in ("2025", "26H1", "26H2", "ALL"):
            a = PT if per == "ALL" else PT[PT.per == per]
            b = P100 if per == "ALL" else P100[P100.per == per]
            da = a.groupby("day").x0.sum(); db = b.groupby("day").x0.sum()
            dd = da.sub(db, fill_value=0)
            idx = RNG.integers(0, len(dd), (3000, len(dd))); bs = dd.values[idx].sum(1)
            e = summ(a, f"policy T{int(T)}", dict(T=T, period=per, universe=src))
            e.update(N_100=len(b), avg_100=b.x0.mean(), sum_x0_100=b.x0.sum(), d_sum_x0=a.x0.sum() - b.x0.sum(),
                     d_sum_ci=f"[{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]", P_d_sum_pos=float((bs > 0).mean()),
                     d_sum_r1=a.r1.sum() - b.r1.sum())
            rowsC.append(e)

A = pd.DataFrame(rowsA).round(3); B = pd.DataFrame(rowsB).round(3); Cc = pd.DataFrame(rowsC).round(3)
A.to_csv(f"{OUT}_volume_first_on.csv", index=False); B.to_csv(f"{OUT}_volume_paired.csv", index=False)
Cc.to_csv(f"{OUT}_volume_policy.csv", index=False); PB.round(3).to_csv(f"{OUT}_volume_paired_rows.csv", index=False)

# parity: 100x READY priced bars vs yr6 FRENZY_LONG fills (same pair, same signal bar)
D = pd.read_pickle(f"{S}/cohort.pkl"); Y = D[(D.blk == "BT") & (D.sleeve == "FRENZY_LONG")].copy()
Y["sc"] = (Y.te - 8_000) // 300_000 * 300_000
m = Y.merge(V[V["T"] == 100][["pair", "t_signal_close", "x0", "E"]], left_on=["pair", "sc"], right_on=["pair", "t_signal_close"], how="left",
            suffixes=("", "_v"))
par = dict(yr6_fills=len(Y), matched=int(m.x0_v.notna().sum()), same_sign_x0=float((np.sign(m.x0) == np.sign(m.x0_v))[m.x0_v.notna()].mean()),
           med_abs_entry_gap_pct=float(((m.E_v / m.E - 1).abs() * 100)[m.x0_v.notna()].median()))
pd.DataFrame([par]).to_csv(f"{OUT}_volume_parity.csv", index=False)
pd.set_option("display.width", 320)
print("PARITY", par)
print(A[A.period == "ALL"].to_string()); print(A[(A.period != "ALL") & A.cls.isin(["LATER", "NEVER"])].to_string())
print(B.to_string()); print(Cc.to_string())
