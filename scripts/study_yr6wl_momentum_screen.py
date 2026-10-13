"""YR6 WL MOMENTUM — step 2: exhaustive separator screen with a shuffled null, discover -> confirm.

For a sleeve (ML | MS) and a discovery regime D:
  1D: every signal column (entry_* + categorical dummies, coverage >= 80 % in D) at
      sign (x>0) / median (x>med_D) / top-tercile (x>q67_D) / bottom-tercile (x<=q33_D); binaries: ==1.
  2D: every pair of the top-40 columns (by 1D null p in D) at median x median quadrants.
  Statistic per test: delta = mean(in-zone) - mean(out-zone), outcome pct_new.
      Pair columns: fill level.  Macro columns (BTC/breadth/market-wide): DAY level — each day's in-zone
      mean and out-zone mean are one observation each (window-units rule).
  Null: 200 circular shifts of the time-sorted outcome vector (keeps the day clustering of outcomes and
      features, breaks their alignment); per-test p = share of shifts with |delta| >= observed; plus the
      family-wise max-|delta| null rank. Zone thresholds are fixed in D and applied unchanged to C.
  Confirm: in the other same-direction regime C, day-bootstrap P(delta has the D sign) >= 0.90.
  Bear (2026-H1) is reported separately, plus the top-50 subset, concentration, master real fills.

    venv/bin/python scripts/study_yr6wl_momentum_screen.py ML "2025 BULL"    "2026-H2 BULL"
    venv/bin/python scripts/study_yr6wl_momentum_screen.py ML "2026-H2 BULL" "2025 BULL"
    venv/bin/python scripts/study_yr6wl_momentum_screen.py MS "2025 BULL"    "2026-H2 BULL"
    venv/bin/python scripts/study_yr6wl_momentum_screen.py ML "2026-H1 BEAR" ""      (bear: discovery only)
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_momentum_common import (ROOT, breakeven_wr, day_boot, day_boot_delta, is_macro, load_master_mom,
                                         load_replay, signal_columns)

SLV = {"ML": "MOM-long", "MS": "MOM-short"}[sys.argv[1] if len(sys.argv) > 1 else "ML"]
DISC = sys.argv[2] if len(sys.argv) > 2 else "2025 BULL"
CONF = sys.argv[3] if len(sys.argv) > 3 else "2026-H2 BULL"
BEAR = "2026-H1 BEAR"
NSHUF = 200
TOP2D = 40
RNG = np.random.default_rng(20261013)
tag = f"{sys.argv[1] if len(sys.argv) > 1 else 'ML'}_{DISC[:4]}{DISC[-4:].strip()}".replace(" ", "")

d_all = load_replay([SLV])
cols_all = signal_columns(d_all)
D = d_all[d_all.regime == DISC].sort_values("ts").reset_index(drop=True)
cols = [c for c in cols_all if pd.to_numeric(D[c], errors="coerce").notna().mean() >= 0.8
        and pd.to_numeric(D[c], errors="coerce").nunique() >= 2]
lowcov = [c for c in cols_all if c not in cols]
N = len(D)
day_codes, day_idx = np.unique(D.day.values, return_inverse=True)
ND = len(day_codes)
DM = np.zeros((N, ND), np.float32); DM[np.arange(N), day_idx] = 1


def X(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float).values


# ---------------- build the test list (zone definitions frozen in D) ----------------
tests = []   # dict(name, cols, kind, defs)  defs = list of (col, op, thr)
for c in cols:
    x = X(D, c); xs = x[np.isfinite(x)]
    if len(np.unique(xs)) == 2:
        tests.append(dict(name=f"{c}==hi", defs=[(c, "ge", xs.max())], macro=is_macro(c), dim=1))
        continue
    if (xs > 0).any() and (xs <= 0).any():
        tests.append(dict(name=f"{c}>0", defs=[(c, "gt", 0.0)], macro=is_macro(c), dim=1))
    med, q33, q67 = np.median(xs), np.quantile(xs, 1 / 3), np.quantile(xs, 2 / 3)
    tests.append(dict(name=f"{c}>med({med:.4g})", defs=[(c, "gt", med)], macro=is_macro(c), dim=1))
    tests.append(dict(name=f"{c}>q67({q67:.4g})", defs=[(c, "gt", q67)], macro=is_macro(c), dim=1))
    tests.append(dict(name=f"{c}<=q33({q33:.4g})", defs=[(c, "le", q33)], macro=is_macro(c), dim=1))


def zone(df, defs):
    """returns (in_mask, scored_mask) for the conjunction of defs."""
    inz = np.ones(len(df), bool); sc = np.ones(len(df), bool)
    for c, op, thr in defs:
        x = X(df, c); ok = np.isfinite(x); sc &= ok
        if op == "gt": inz &= x > thr
        elif op == "ge": inz &= x >= thr
        elif op == "le": inz &= x <= thr
        elif op == "notgt": inz &= ~(x > thr)
    return inz & sc, sc


def run_null(tests, D):
    y = D.y.values.astype(np.float32)
    Y = np.empty((N, NSHUF + 1), np.float32); Y[:, 0] = y
    for k in range(1, NSHUF + 1):
        Y[:, k] = np.roll(y, RNG.integers(int(0.1 * N), int(0.9 * N)))
    T = len(tests)
    IN = np.zeros((T, N), np.float32); OUT = np.zeros((T, N), np.float32)
    for i, t in enumerate(tests):
        a, s = zone(D, t["defs"]); IN[i] = a; OUT[i] = s & ~a
    nin, nout = IN.sum(1), OUT.sum(1)
    din, dout = (IN @ DM > 0).sum(1), (OUT @ DM > 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fill = (IN @ Y) / nin[:, None] - (OUT @ Y) / nout[:, None]                      # T x K
        macro = np.array([t["macro"] for t in tests])
        dayd = np.full_like(fill, np.nan)
        mi = np.where(macro)[0]
        if len(mi):
            cin, cout = IN[mi] @ DM, OUT[mi] @ DM                                         # Tm x ND
            for k in range(NSHUF + 1):
                yk = Y[:, k]
                sin, sout = (IN[mi] * yk) @ DM, (OUT[mi] * yk) @ DM
                min_ = np.where(cin > 0, sin / np.where(cin > 0, cin, 1), np.nan)
                mout = np.where(cout > 0, sout / np.where(cout > 0, cout, 1), np.nan)
                dayd[mi, k] = np.nanmean(min_, 1) - np.nanmean(mout, 1)
    stat = np.where(macro[:, None], dayd, fill)
    obs = stat[:, 0]
    valid = (nin >= 0.05 * N) & (nout >= 0.05 * N) & (din >= 8) & (dout >= 8) & np.isfinite(obs)
    p = ((np.abs(stat[:, 1:]) >= np.abs(obs)[:, None]).sum(1) + 1) / (NSHUF + 1)
    fwmax = np.nanmax(np.where(valid[:, None], np.abs(stat[:, 1:]), np.nan), 0)
    fwrank = np.array([(fwmax >= abs(o)).mean() if v else np.nan for o, v in zip(obs, valid)])
    res = pd.DataFrame(dict(test=[t["name"] for t in tests], dim=[t["dim"] for t in tests], macro=macro,
                            n_in=nin.astype(int), n_out=nout.astype(int), days_in=din, days_out=dout,
                            delta_fill=fill[:, 0], delta_stat=obs, p_null=p, fw_p=fwrank, valid=valid))
    res["avg_in"] = (IN @ Y[:, 0]) / nin
    res["avg_out"] = (OUT @ Y[:, 0]) / nout
    return res, fwmax


print(f"# {SLV} · discovery {DISC} (N {N}, {ND} days, avg {D.y.mean():+.3f}) → confirm {CONF or '—'} · bear {BEAR}")
print(f"# {len(cols)} columns tested at >= 80 % coverage; low-coverage columns read separately: {lowcov}")
r1, fw1 = run_null(tests, D)
allt1 = {t["name"]: t for t in tests}
v1 = r1[r1.valid]
print(f"\n## 1D: {len(v1)} valid tests · per-test p<0.05: {(v1.p_null < 0.05).sum()} (expected by luck {0.05 * len(v1):.0f}) · "
      f"p<0.01: {(v1.p_null < 0.01).sum()} (exp {0.01 * len(v1):.1f}) · family-wise p<0.05: {(v1.fw_p < 0.05).sum()} · "
      f"null 95th pct of max|Δ| = {np.percentile(fw1, 95):.3f}")

# ---------------- 2D on top-40 columns ----------------
best = (v1.assign(col=[allt1[t]["defs"][0][0] for t in v1.test]).groupby("col").p_null.min().sort_values())
top = best.index[:TOP2D].tolist()
tests2 = []
for a, b in itertools.combinations(top, 2):
    xa, xb = X(D, a), X(D, b)
    ma, mb = np.nanmedian(xa), np.nanmedian(xb)
    if len(np.unique(xa[np.isfinite(xa)])) == 2: ma = (np.nanmax(xa) + np.nanmin(xa)) / 2
    if len(np.unique(xb[np.isfinite(xb)])) == 2: mb = (np.nanmax(xb) + np.nanmin(xb)) / 2
    for oa, ob in [("gt", "gt"), ("gt", "notgt"), ("notgt", "gt"), ("notgt", "notgt")]:
        nm = f"{a}{'>' if oa == 'gt' else '<='}{ma:.4g} & {b}{'>' if ob == 'gt' else '<='}{mb:.4g}"
        tests2.append(dict(name=nm, defs=[(a, oa, ma), (b, ob, mb)], macro=is_macro(a) or is_macro(b), dim=2))
r2, fw2 = run_null(tests2, D)
v2 = r2[r2.valid]
print(f"## 2D: top-{len(top)} cols → {len(tests2)} quadrant tests, {len(v2)} valid · per-test p<0.05: {(v2.p_null < 0.05).sum()} "
      f"(exp {0.05 * len(v2):.0f}) · p<0.01: {(v2.p_null < 0.01).sum()} (exp {0.01 * len(v2):.0f}) · family-wise p<0.05: "
      f"{(v2.fw_p < 0.05).sum()} · null 95th pct max|Δ| = {np.percentile(fw2, 95):.3f}")

allt = {t["name"]: t for t in tests + tests2}
R = pd.concat([v1, v2]).reset_index(drop=True)


# ---------------- confirmation + bear + master ----------------
def evaluate(df, defs, sign):
    """zone stats in another dataset with the D-frozen thresholds. sign = sign of D delta."""
    if df is None or not len(df):
        return {}
    a, s = zone(df, defs)
    if a.sum() < 3 or (s & ~a).sum() < 3:
        return dict(n_in=int(a.sum()), n_out=int((s & ~a).sum()))
    sub = df[s]; inz = a[s]
    dl, lo, hi, p = day_boot_delta(sub.y.values, inz, sub.day.values, n=1000)
    return dict(n_in=int(inz.sum()), n_out=int((~inz).sum()), days_in=sub[inz].day.nunique(),
                avg_in=sub.y[inz].mean(), avg_out=sub.y[~inz].mean(), delta=dl, lo=lo, hi=hi,
                p_same=p if sign > 0 else 1 - p)


C = d_all[d_all.regime == CONF] if CONF else None
B = d_all[d_all.regime == BEAR]
M = load_master_mom(); M = M[M.sleeve == SLV]
Mk = M[M.stack_keep == True].assign(y=lambda x: x.y_stack)

cand = R[(R.p_null < 0.05) & (R.delta_stat.abs() >= 0.05)].copy()
rows = []
for _, r in cand.iterrows():
    t = allt[r.test]; sg = np.sign(r.delta_stat)
    row = dict(test=r.test, dim=r.dim, macro=r.macro, D_n_in=r.n_in, D_days_in=r.days_in, D_avg_in=r.avg_in,
               D_avg_out=r.avg_out, D_delta=r.delta_stat, D_p=r.p_null, D_fw=r.fw_p)
    for lab, df in [("C", C), ("B", B), ("Mall", M), ("Mkeep", Mk)]:
        for k, v in evaluate(df, t["defs"], sg).items():
            row[f"{lab}_{k}"] = v
    rows.append(row)
cf = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, f"reports/YR6_WL_MOMENTUM_screen_{tag}.csv")
pd.concat([R.assign(stage="all")]).to_csv(out_csv.replace(".csv", "_alltests.csv"), index=False)
cf.to_csv(out_csv, index=False)

print(f"\n## Discovery survivors (p_null<0.05 & |Δ|>=0.05): {len(cf)}")
if len(cf) and CONF:
    cf["confirmed"] = (cf.C_p_same >= 0.90)
    print(f"   confirmed in {CONF} (same sign, day-boot P>=0.90): {int(cf.confirmed.sum())} of {len(cf)}  "
          f"[luck baseline: a random sign would confirm ~10 % → {0.10 * len(cf):.0f}]")
    sh = cf.sort_values(["confirmed", "D_p", "C_p_same"], ascending=[False, True, False])
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 70)
    show = ["test", "macro", "D_n_in", "D_days_in", "D_avg_in", "D_avg_out", "D_delta", "D_p", "D_fw",
            "C_n_in", "C_avg_in", "C_avg_out", "C_delta", "C_p_same", "B_avg_in", "B_avg_out", "B_delta",
            "Mall_n_in", "Mall_avg_in", "Mall_avg_out", "Mkeep_n_in", "Mkeep_avg_in", "Mkeep_avg_out"]
    print(sh[sh.confirmed][show].head(40).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\n   top-15 discovery survivors that did NOT confirm:")
    print(sh[~sh.confirmed][show[:15]].head(15).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
elif len(cf):
    print(cf.sort_values("D_p").head(30)[["test", "macro", "D_n_in", "D_days_in", "D_avg_in", "D_avg_out", "D_delta", "D_p", "D_fw",
                                           "Mall_n_in", "Mall_avg_in", "Mall_avg_out"]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

# ---------------- positive pockets: can ANY zone keep the sleeve positive in BOTH bulls? ----------------
if CONF:
    print(f"\n## Positive-pocket search: zones (in OR out side of every valid test) holding >= 5 % of {DISC} fills and >= 15 days,"
          f" mean > 0 in {DISC}; then the same frozen zone in {CONF} / bear / master")
    pk = []
    for _, r in R.iterrows():
        t = allt[r.test]
        for side in ("in", "out"):
            nn = r.n_in if side == "in" else r.n_out; dd = r.days_in if side == "in" else r.days_out
            av = r.avg_in if side == "in" else r.avg_out
            if nn >= 0.05 * N and dd >= 15 and av > 0:
                pk.append((r.test, side, nn, dd, av, t))
    out = []
    for name, side, nn, dd, av, t in pk:
        def side_stats(df):
            if df is None or not len(df): return (0, np.nan, np.nan, np.nan)
            a, s = zone(df, t["defs"]); m = a if side == "in" else (s & ~a)
            g = df[m]
            if len(g) < 5: return (len(g), np.nan, np.nan, np.nan)
            mu, lo, hi, p = day_boot(g.y, g.day, n=1000)
            return (len(g), mu, lo, p)
        dn, dmu, dlo, dp = side_stats(D)
        cn, cmu, clo, cp = side_stats(C)
        bn, bmu, _, bp = side_stats(B)
        mn, mmu, _, _ = side_stats(M)
        out.append(dict(zone=("" if side == "in" else "NOT ") + name, D_n=dn, D_share=dn / N, D_avg=dmu, D_P=dp,
                        C_n=cn, C_avg=cmu, C_lo=clo, C_P=cp, B_n=bn, B_avg=bmu, M_n=mn, M_avg=mmu))
    po = pd.DataFrame(out)
    if len(po):
        po["both_pos"] = (po.D_avg > 0) & (po.C_avg > 0)
        print(f"   {len(po)} pockets positive in {DISC}; of those positive in {CONF} too: {int(po.both_pos.sum())}; "
              f"with D P>0 >= 0.95: {int((po.D_P >= 0.95).sum())}; both P>=0.90: {int(((po.D_P >= 0.90) & (po.C_P >= 0.90)).sum())}")
        print(po.sort_values(["D_P", "C_P"], ascending=False).head(20).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        po.to_csv(out_csv.replace(".csv", "_pockets.csv"), index=False)
    else:
        print("   none.")

# ---------------- low-coverage columns, own read (scored rows only) ----------------
print("\n## Low-coverage columns — own read on scored rows only (never zone-vs-rest)")
for c in lowcov:
    for lab, df in [(DISC, D), (CONF, C), (BEAR, B)]:
        if df is None or not len(df): continue
        x = X(df, c); ok = np.isfinite(x)
        if ok.sum() < 20: print(f"   {c:<36}{lab:<14} scored {ok.sum()} — too few"); continue
        g = df[ok]; med = np.nanmedian(X(D, c))
        hi = X(g, c) > med
        print(f"   {c:<36}{lab:<14} scored {ok.mean()*100:4.0f}% (N {ok.sum()}) avg {g.y.mean():+.3f} · "
              f">med({med:.3g}) {g.y[hi].mean():+.3f} ({hi.sum()}) vs <= {g.y[~hi].mean():+.3f} ({(~hi).sum()}) · "
              f"unscored avg {df[~ok].y.mean():+.3f}")
print(f"\nCSV: {out_csv} (+ _alltests / _pockets)")
