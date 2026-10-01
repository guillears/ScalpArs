#!/usr/bin/env python3
"""BTC 1h slope < 0 × second variable — exhaustive 2D screen for momentum-LONG losers (operator 2026-10-01: "what if BTC 1h slope
< 0 is a 2D variable to filter losers?"). The 1D read is settled (NOT a filter: removes 33/89 winners; observe tally since Jul-3).

Method (CLAUDE.md quant discipline; exhaustive_2d / separator-coverage / window-units rules):
  pool      current-stack momentum LONGs = current_stack_ledger.build() (validated: validate_against_master.py) + the fresh
            batch's momentum longs (live gates already applied), pct = the ledger's pct convention
  cohort    entry_btc_1h_slope < 0
  legs      every stamped entry_* variable scored on ≥ 80 % of the cohort (+ deltas), split at the cohort MEDIAN and at 0, both sides
  survivor  quadrant (slope<0 ∧ leg): N ≥ 8, WR ≤ 50 %, avg < 0, Δ vs rest-of-cohort ≤ −0.25 %
  null      outcome shuffled WITHIN DAY inside the cohort, 200 permutations → survivors luck produces
  interaction  the same leg on slope ≥ 0: a leg that is just as bad there is a 1D effect, not a 2D one
  OOS       the yr3 backtest fills (H1 < 2026-05-01 ≤ H2), same legs/thresholds: confirmed = avg < 0 and Δ < 0 in BOTH halves (≥ 20 each)
  units     windows (fills ≤ 2 min apart = one window) + per-era (batch) before/after table
  bar       a candidate is only a FILTER candidate under the EXPECTANCY bar (WR < sleeve breakeven ≈ 61 % ∧ avg < 0 at 95 % window-
            bootstrap ∧ ≥ 8 windows ∧ N ≥ 15 ∧ no window/pair ≥ 50 % of the loss) — else observe-only.
Usage: venv/bin/python scripts/slope_neg_2d_screen.py [--fresh <orders csv>]…  → reports/SLOPE_NEG_2D_SCREEN_2026-10-01.md"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
FRESH = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--fresh" and i + 1 < len(sys.argv)]
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG  # noqa: E402
M = LG.build(); sys.argv = _a
OUT = os.path.join(ROOT, "reports", "SLOPE_NEG_2D_SCREEN_2026-10-01.md")
n = lambda d, c: pd.to_numeric(d[c], errors="coerce")

M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"),
                    M.stack_pnl / n(M, "notional_value") * 100, M.pnl_percentage)
M = M[(M.direction == "LONG") & (M.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].copy()
last = pd.to_datetime(M.opened_at, format="ISO8601").max()
fr = []
for f in FRESH:
    b = pd.read_csv(f, low_memory=False)
    b = b[(b.status == "CLOSED") & (b.direction == "LONG") & (b.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")
          & (pd.to_datetime(b.opened_at, format="ISO8601") > last)].copy()
    b = b[~b.cell_multiplier_source.astype(str).str.contains("_PROBE")]   # full-size only
    b["pct"] = b.pnl_percentage; b["net"] = b.pnl; b["era"] = "FRESH"; fr.append(b)
C = pd.concat([M] + fr, ignore_index=True)
C = C.drop_duplicates(["opened_at", "pair", "direction"], keep="first").reset_index(drop=True)
C["day"] = C.opened_at.astype(str).str[:10]
ts = pd.to_datetime(C.opened_at, format="ISO8601").astype("int64") // 10**6
order = np.argsort(ts.values); win = np.empty(len(C), int); cur, prev = -1, None
for i in order:
    if prev is None or ts.values[i] - prev > 120_000:
        cur += 1
    win[i] = cur; prev = ts.values[i]
C["win"] = win
slope = n(C, "entry_btc_1h_slope")
NEG = (slope < 0).values


PREGATE = "--pregate" in sys.argv


def load_bt(postgate=True):
    """yr3 backtest momentum longs. postgate = the current stack (as exhaustive_2d_master.py). The 5 seeds replay the SAME year
    (correlated, not independent samples — review 2026-10-01): collapsed to ONE row per trade (pair × 5-min open bucket), pnl =
    the mean across the seeds that took it, features from the first."""
    F = pd.read_csv("reports/backtest_cache/replay/year/yr3_report_fills.csv", low_memory=False)
    F = F[(F.direction == "LONG") & (F.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")]
    if postgate:
        F = F[F.cell_multiplier_source.astype(str) != "CROSS_OB_OPEN"]
        F = F[~((F.cell_multiplier_source.astype(str) == "NONEXP_CALM3D") & (n(F, "entry_btc_atr_pct") < 0.08))]
        F = F[~((n(F, "entry_adx") < 21) & (n(F, "entry_rsi") < n(F, "entry_rsi_prev")))]
    F = F.copy(); F["k"] = F.pair.astype(str) + "|" + pd.to_datetime(F.opened_at, format="ISO8601").dt.floor("5min").astype(str)
    pnl = F.groupby("k").pnl_percentage.mean()
    F = F.drop_duplicates("k").set_index("k"); F["pnl_percentage"] = pnl
    return F.reset_index(drop=True)


def stamped(d):
    X = {c: n(d, c) for c in d.columns if c.startswith("entry_") and pd.api.types.is_numeric_dtype(n(d, c))
         and c not in ("entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional", "entry_liquidity_cap_notional",
                       "entry_mcap_usd", "entry_cmc_rank", "entry_btc_1h_slope")}
    X["d_rsi"] = n(d, "entry_rsi") - n(d, "entry_rsi_prev"); X["d_adx"] = n(d, "entry_adx") - n(d, "entry_adx_prev")
    X["d_btc_rsi"] = n(d, "entry_btc_rsi") - n(d, "entry_btc_rsi_prev"); X["d_btc_adx"] = n(d, "entry_btc_adx") - n(d, "entry_btc_adx_prev")
    X["d_btc_rsi_30m"] = n(d, "entry_btc_rsi") - n(d, "entry_btc_rsi_prev6"); X["di_spread"] = n(d, "entry_pos_di") - n(d, "entry_neg_di")
    X["d_btc_rsi_1h"] = n(d, "entry_btc_rsi_1h") - n(d, "entry_btc_rsi_1h_prev")
    return pd.DataFrame(X, index=d.index)


X = stamped(C)
cov = X[NEG].notna().mean()
cols = [c for c in X.columns if cov.get(c, 0) >= 0.8 and X.loc[NEG, c].nunique() > 4]
y = C.pct.values.astype(float)


def legs():
    out = []
    for c in cols:
        med = float(X.loc[NEG, c].median())
        for thr, lab in ((med, f"{med:.3g}"), (0.0, "0")):
            if lab == "0" and (abs(med) < 1e-9 or (X.loc[NEG, c] > 0).mean() < 0.15 or (X.loc[NEG, c] <= 0).mean() < 0.15):
                continue
            hi = (X[c] > thr).values; lo = (X[c] <= thr).values & X[c].notna().values
            out += [(f"{c}>{lab}", c, ">", thr, hi), (f"{c}≤{lab}", c, "≤", thr, lo)]
    return out


L = legs()


def stats(mask, yy=y):
    v = yy[mask]
    return len(v), (float((v > 0).mean() * 100) if len(v) else np.nan), (float(v.mean()) if len(v) else np.nan)


def scan(yy):
    res = []
    for name, c, op, thr, m in L:
        q = NEG & m; rest = NEG & ~m & X[c].notna().values
        N, wr, av = stats(q, yy)
        if N < 8 or rest.sum() < 5:
            continue
        d = av - yy[rest].mean()
        if wr <= 50 and av < 0 and d <= -0.25:
            res.append((name, c, op, thr, N, wr, av, d))
    return res


S = scan(y)
rng = np.random.default_rng(7); null = []
days = C.day.values
for _ in range(200):
    yp = y.copy()
    for dd in np.unique(days[NEG]):
        idx = np.where(NEG & (days == dd))[0]
        yp[idx] = rng.permutation(y[idx])
    null.append(len(scan(yp)))
null = np.array(null)


def win_boot(mask, n_boot=10000, seed=3):
    g = pd.DataFrame({"w": C.win.values[mask], "y": y[mask]}).groupby("w").y.mean().values
    if len(g) < 3:
        return len(g), np.nan, np.nan
    r = np.random.default_rng(seed).choice(g, (n_boot, len(g))).mean(axis=1)
    return len(g), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


# OOS on the yr3 backtest fills (same thresholds)
OOS = {}
fp = "reports/backtest_cache/replay/year/yr3_report_fills.csv"
if os.path.exists(fp):
    F = load_bt(postgate=True)
    XF = stamped(F); fneg = (n(F, "entry_btc_1h_slope") < 0).values; fy = F.pnl_percentage.values.astype(float)
    h1 = (F.opened_at.astype(str) < "2026-05-01").values
    for name, c, op, thr, *_ in S:
        if c not in XF:
            OOS[name] = "not stamped in backtest"; continue
        m = (XF[c] > thr).values if op == ">" else ((XF[c] <= thr).values & XF[c].notna().values)
        parts = []; ok = True
        for lab, h in (("H1", h1), ("H2", ~h1)):
            q = fneg & m & h; rest = fneg & ~m & h & XF[c].notna().values
            if q.sum() < 20 or rest.sum() < 20:
                parts.append(f"{lab} N={int(q.sum())} (too few)"); ok = False; continue
            av, d = fy[q].mean(), fy[q].mean() - fy[rest].mean()
            parts.append(f"{lab} {int(q.sum())}·{(fy[q] > 0).mean()*100:.0f}%·{av:+.3f} (Δ{d:+.3f})"); ok &= (av < 0 and d < 0)
        OOS[name] = ("✅ confirmed · " if ok else "✗ not confirmed · ") + " · ".join(parts)

# ── report
Lr = ["# BTC 1h slope < 0 × second variable — 2D screen for momentum-LONG losers (2026-10-01)", "",
      f"Pool: current-stack momentum longs from the master ledger (validated) + fresh batch → **{len(C)} fills**, "
      f"**{int(NEG.sum())} with BTC 1h slope < 0**. Legs tested: {len(L)} ({len(cols)} variables × median/zero × both sides).", ""]
N0, wr0, av0 = stats(NEG); N1, wr1, av1 = stats(~NEG & slope.notna().values)
Lr += ["## The cohort itself", "", "| | N | WR | avg % |", "|---|---|---|---|",
       f"| slope < 0 | {N0} | {wr0:.0f}% | {av0:+.3f} |", f"| slope ≥ 0 | {N1} | {wr1:.0f}% | {av1:+.3f} |", ""]
Lr += ["## Per batch (era): the slope<0 cohort", "", "| Era | N (slope<0) | WR | avg % | N (slope≥0) | WR | avg % |", "|---|---|---|---|---|---|---|"]
for era, g in C.groupby("era", sort=False):
    a = g.index[NEG[g.index]]; b = g.index[(~NEG[g.index]) & slope.loc[g.index].notna().values]
    sa, sb = stats(np.isin(np.arange(len(C)), a)), stats(np.isin(np.arange(len(C)), b))
    Lr.append(f"| {era} | {sa[0]} | {sa[1]:.0f}% | {sa[2]:+.3f} | {sb[0]} | {sb[1]:.0f}% | {sb[2]:+.3f} |" if sa[0] else
              f"| {era} | 0 | – | – | {sb[0]} | {sb[1]:.0f}% | {sb[2]:+.3f} |")
Lr += ["", f"## Survivors (slope<0 ∧ leg: N≥8, WR≤50 %, avg<0, Δ≤−0.25) — **{len(S)} on the real data vs null median {np.median(null):.0f} "
       f"(95th pct {np.percentile(null, 95):.0f}, P(null ≥ real) = {(null >= len(S)).mean():.2f})**", ""]
if S:
    Lr += ["| Leg (with slope<0) | N | WR | avg % | Δ vs rest of cohort | windows · 95 % window CI | same leg on slope ≥ 0 | yr3 backtest OOS |",
           "|---|---|---|---|---|---|---|---|"]
    for name, c, op, thr, N, wr, av, d in sorted(S, key=lambda r: r[7]):
        m = (X[c] > thr).values if op == ">" else ((X[c] <= thr).values & X[c].notna().values)
        nw, lo, hi = win_boot(NEG & m)
        po = (~NEG) & m & slope.notna().values; Np, wrp, avp = stats(po)
        Lr.append(f"| {name} | {N} | {wr:.0f}% | {av:+.3f} | {d:+.3f} | {nw} · [{lo:+.2f}, {hi:+.2f}] | {Np}·{wrp:.0f}%·{avp:+.3f} | {OOS.get(name, '–')} |")
open(OUT, "w").write("\n".join(Lr) + "\n")
print("\n".join(Lr)); print(f"\n→ {OUT}")


# ═══════════════════ PART 2 — discovery on the yr3 backtest (H1), confirmation on H2 + seeds + master ═══════════════════
def backtest_2d(nperm=100, postgate=True):
    """Master N is too small (39 slope<0 fills, 9 losers) for a 2D survivor. The backtest (seeds collapsed to one row per trade):
    discover on H1 (< 2026-05-01) inside slope<0, legs split at the H1-cohort median / 0; a candidate = H1 quadrant N ≥ 30, avg < 0,
    Δ vs rest-of-cohort ≤ −0.20. CONFIRMED = in H2 the same quadrant has avg < 0 and Δ ≤ −0.10 (N ≥ 30) AND the leg is weaker on
    slope ≥ 0 IN H2 (interaction: Δ(slope<0) − Δ(slope≥0) ≤ −0.10, out of sample). Null: the same pipeline on outcomes shuffled
    within day (all days, one permutation per run)."""
    F = load_bt(postgate=postgate)
    XF = stamped(F); fy = F.pnl_percentage.values.astype(float); fs = n(F, "entry_btc_1h_slope").values
    neg, pos = fs < 0, fs >= 0; h1 = (F.opened_at.astype(str) < "2026-05-01").values
    fday = F.opened_at.astype(str).str[:10].values
    fcols = [c for c in XF.columns if XF.loc[neg, c].notna().mean() >= 0.8 and XF.loc[neg, c].nunique() > 4 and c in X.columns]
    legsF = []
    for c in fcols:
        med = float(XF.loc[neg & h1, c].median())
        for thr in {med, 0.0}:
            if thr == 0.0 and ((XF.loc[neg, c] > 0).mean() < 0.15 or (XF.loc[neg, c] <= 0).mean() < 0.15):
                continue
            v = XF[c].values; ok = ~np.isnan(v)
            legsF += [(f"{c}>{thr:.3g}", c, ">", thr, (v > thr) & ok), (f"{c}≤{thr:.3g}", c, "≤", thr, (v <= thr) & ok)]

    def run(yy):
        out = []
        for name, c, op, thr, m in legsF:
            ok = ~np.isnan(XF[c].values)
            def q(mask, part):
                a = neg & m & part; r = neg & ~m & ok & part
                return a.sum(), (yy[a].mean() if a.sum() else np.nan), (yy[a].mean() - yy[r].mean() if a.sum() and r.sum() else np.nan)
            n1, a1, d1 = q(m, h1)
            if not (n1 >= 30 and a1 < 0 and d1 <= -0.20):
                continue
            n2, a2, d2 = q(m, ~h1)
            if not (n2 >= 30 and a2 < 0 and d2 <= -0.10):
                continue
            pa = pos & m & ~h1; pr = pos & ~m & ok & ~h1                     # interaction read on H2 only (out of sample)
            dpos = yy[pa].mean() - yy[pr].mean() if pa.sum() and pr.sum() else np.nan
            dneg = d2
            if not (dneg - dpos <= -0.10):
                continue
            out.append((name, c, op, thr, n1, a1, d1, n2, a2, d2, "–", dneg, dpos))
        return out

    real = run(fy)
    rng = np.random.default_rng(11); nul = []
    for _ in range(nperm):
        yp = fy.copy()
        for dd in np.unique(fday):
            idx = np.where(neg & (fday == dd))[0]; yp[idx] = rng.permutation(fy[idx])
        nul.append(len(run(yp)))
    return real, np.array(nul), len(legsF), int(neg.sum()), F


for _pg in ([True, False] if PREGATE else [True]):
  real, nul, nlegs, nneg, F = backtest_2d(postgate=_pg)
  L2 = ["", f"## Part 2 — discovery on the year backtest, {'POST-GATE (current stack)' if _pg else 'PRE-GATE (raw replay — for the record only)'}, "
        "seeds collapsed to one row per trade; H1 Jan–Apr discover, H2 May–Sep confirm + H2 interaction", "",
        f"Backtest momentum longs with slope < 0: **{nneg}** unique trades. Legs tested: {nlegs}. **Confirmed candidates: {len(real)} on the real "
        f"data vs null median {np.median(nul):.0f} (95th pct {np.percentile(nul, 95):.0f}, P(null ≥ real) = {(nul >= len(real)).mean():.2f})**", ""]
  if real:
      L2 += ["| Leg (with slope<0) | H1 N·avg·Δ | H2 N·avg·Δ | seeds | H2 Δ on slope<0 vs on slope≥0 | MASTER slope<0 ∧ leg | MASTER per era (N·avg) |",
             "|---|---|---|---|---|---|---|"]
      for (name, c, op, thr, n1, a1, d1, n2, a2, d2, ns, dneg, dpos) in sorted(real, key=lambda r: r[9]):
          if c in X.columns:
              m = (X[c] > thr).values if op == ">" else ((X[c] <= thr).values & X[c].notna().values)
              q = NEG & m; rest = NEG & ~m & X[c].notna().values
              Nm, wrm, avm = stats(q); dm = (y[q].mean() - y[rest].mean()) if q.sum() and rest.sum() else np.nan
              per = " · ".join(f"{e}:{int((q & (C.era.values == e)).sum())}·{y[q & (C.era.values == e)].mean():+.2f}"
                               for e in pd.unique(C.era) if (q & (C.era.values == e)).sum())
              mtxt = f"{Nm}·{wrm:.0f}%·{avm:+.3f} (Δ{dm:+.3f})"
          else:
              mtxt, per = "not stamped live", ""
          L2.append(f"| {name} | {n1}·{a1:+.3f}·{d1:+.3f} | {n2}·{a2:+.3f}·{d2:+.3f} | {ns} | {dneg:+.3f} vs {dpos:+.3f} | {mtxt} | {per} |")
  open(OUT, "a").write("\n".join(L2) + "\n")
  print("\n".join(L2))
