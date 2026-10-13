#!/usr/bin/env python3
"""YR6 winners-vs-losers, AREA = OTHERS — follow-up checks on the screen (scripts/study_yr6wl_others_screen.py must have run).

  venv/bin/python scripts/study_yr6wl_others_checks.py

  A. headline per-regime table computed a SECOND way (straight from reports/ENGINE_REPLAY_YR6_2Y_TODAY_regimes.csv) vs the fills
  B. regime-switch read per sleeve: uniform-degradation test (frozen whole-sleeve tercile cohorts of 6 pair variables × regime),
     month sign table, per-day concentration of the sleeve's P&L
  C. confirmation-count NULL: the confirmed-separator count vs the count obtained after shuffling the confirmation regime's outcomes
     across pair-day units (300 shuffles; same criterion: same sign ∧ one-sided z ≥ 1.28 on pair-day units)
  D. SURGE_LONG candidate "pair EMA13−EMA50 gap < 0" (sign split, threshold-free; column entry_pair_ema20_ema50_gap_pct is
     EMA13−EMA50 — misnamed, see memory): per regime, dose-response, per-seed, top-50 split, concentration, locked EXPECTANCY bar
     (breakeven WR, 95 % DAY-bootstrap, ≥ 8 windows, pair/day ≤ 50 % of loss, N ≥ 15), computed two independent ways
  E. Spike-Fade bear-discovered macro cell (breadth weak ∧ BTC ATR high) in DAY units across regimes
  F. FLIP-short: replay flips in the washed-out state (BTC ≤ −15 % off its 30-day high) vs the rest; replay exit-stop share by
     top-50 membership; Spike-Fade / BullRun top-50 shares 2026 (entry-fidelity context for the master comparison)
Output: reports/YR6_WL_OTHERS_checks.txt
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_others_common import REP, SLEEVES, REGIMES, load, top50_flag, day_ci, bootstrap_diff, is_macro, apply_zone

rng = np.random.default_rng(20261014)
D = load(); D["top50"] = top50_flag(D)
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)


def L(g, ns=3):
    if not len(g):
        return "N 0"
    m, lo, hi = day_ci(g)
    return f"{len(g) / ns:6.1f}/seed · {g.day.nunique():3d} d · WR {(g.y > 0).mean() * 100:4.0f}% · avg {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


# ── A ──
P("A. headline per-regime numbers — fills (this study) vs reports/ENGINE_REPLAY_YR6_2Y_TODAY_regimes.csv (reprice script)")
R = pd.read_csv(os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_regimes.csv"))
for sl in SLEEVES:
    for rg in REGIMES:
        g = D[(D.sleeve == sl) & (D.regime == rg)]
        r = R[(R.sleeve == sl) & (R.regime == rg)]
        if not len(g) and not len(r):
            continue
        a = (len(g) / 3, (g.y > 0).mean() * 100, g.y.mean(), g.usd_new.sum() / 3)
        b = (r.n_seed.iloc[0], r.wr.iloc[0], r.avg.iloc[0], r.usd.iloc[0]) if len(r) else (np.nan,) * 4
        ok = all(abs(x - y) < 1e-6 for x, y in zip(a, b))
        P(f"  {sl:14} {rg:13} fills: {a[0]:6.1f}/seed WR {a[1]:5.1f}% avg {a[2]:+.4f} ${a[3]:+,.0f} | regimes.csv: {b[0]:6.1f} {b[1]:5.1f}% {b[2]:+.4f} ${b[3]:+,.0f} | {'MATCH' if ok else 'DIFF'}")

# ── B ──
P("\nB. regime-switch read (uniform-degradation test: frozen whole-sleeve tercile cohorts × regime; avg %/fill, N/seed)")
COH = ["entry_atr_pct", "entry_rsi", "entry_adx", "entry_pair_volume_24h_usd", "entry_pair_ema20_ema50_gap_pct", "entry_dist_from_ema13_pct"]
for sl in SLEEVES:
    G = D[D.sleeve == sl]
    P(f"  {sl}")
    rows = []
    for c in COH:
        x = pd.to_numeric(G[c], errors="coerce")
        q1, q2 = x.quantile([1 / 3, 2 / 3])
        for lab, m in (("lo", x <= q1), ("mid", (x > q1) & (x < q2)), ("hi", x >= q2)):
            row = {"cohort": f"{c.replace('entry_', '')}:{lab}"}
            for rg in REGIMES:
                h = G[m & (G.regime == rg)]
                row[rg] = f"{h.y.mean():+.3f} ({len(h) / 3:.0f})" if len(h) >= 6 else "–"
            rows.append(row)
    T = pd.DataFrame(rows)
    P(T.to_string(index=False))
    # sign agreement: per regime, share of cohorts with the same sign as the regime mean
    for rg in REGIMES:
        g = G[G.regime == rg]
        if not len(g):
            continue
        sgn = np.sign(g.y.mean()); vals = []
        for c in COH:
            x = pd.to_numeric(G[c], errors="coerce"); q1, q2 = x.quantile([1 / 3, 2 / 3])
            for m in (x <= q1, (x > q1) & (x < q2), x >= q2):
                h = G[m & (G.regime == rg)]
                if len(h) >= 6: vals.append(np.sign(h.y.mean()) == sgn)
        dd = g.groupby("day").y.sum().sort_values()
        tot_neg = dd[dd < 0].sum(); tot_pos = dd[dd > 0].sum()
        P(f"    {rg}: regime avg {g.y.mean():+.3f} · {np.mean(vals) * 100:.0f}% of {len(vals)} cohorts share its sign · "
          f"worst-5 days carry {dd.iloc[:5].sum() / tot_neg * 100 if tot_neg else 0:.0f}% of losing-day Σ · best-5 days {dd.iloc[-5:].sum() / tot_pos * 100 if tot_pos else 0:.0f}% of winning-day Σ · "
          f"Σ ex best-5 days {(dd.sum() - dd.iloc[-5:].sum()) / 3:+.2f}/seed vs Σ {dd.sum() / 3:+.2f}/seed")
    mo = G.groupby(G.t.dt.strftime("%Y-%m")).y.agg(["mean", "size"])
    P("    month avg (N/seed): " + " · ".join(f"{k[2:]} {v['mean']:+.2f}({v['size'] / 3:.0f})" for k, v in mo.iterrows()))
    P(f"    months positive: {(mo['mean'] > 0).sum()}/{len(mo)}")

# ── C ──
P("\nC. confirmation-count null (pair-day units; criterion same sign ∧ z_conf ≥ 1.28 one-sided; 300 shuffles of the confirmation regime)")
C = pd.read_csv(os.path.join(REP, "YR6_WL_OTHERS_confirm.csv"))


def masks_for(g, r):
    if r.family == "2D":
        return None
    col, op, thr = r.col, r.op, r.thr
    x = pd.to_numeric(g[col], errors="coerce")
    if r.unit == "day":
        x = x.groupby(g.day).transform("mean")
    return apply_zone(x.values, op, thr), ~np.isnan(x.values)


S_ALL = {}
for sl in SLEEVES:
    T = os.path.join(REP, f"YR6_WL_OTHERS_screen_{sl}.csv")
    if os.path.exists(T):
        S_ALL[sl] = pd.read_csv(T)
for sl, T in S_ALL.items():
    for disc, conf in (("2025 BULL", "2026-H2 BULL"), ("2026-H2 BULL", "2025 BULL")):
        S = T[(T.regime == disc) & (T.family != "2D") & (T.p < 0.05)]
        gc = D[(D.sleeve == sl) & (D.regime == conf)]
        if not len(S) or not len(gc):
            continue
        key = gc.pd_key.values
        uk, inv = np.unique(key, return_inverse=True)
        yu = np.bincount(inv, weights=gc.y.values) / np.bincount(inv)
        Ms = []
        for _, r in S.iterrows():
            m, s = masks_for(gc, r)
            mu = np.bincount(inv, weights=(m & s).astype(float)) / np.bincount(inv) >= 0.5
            su = np.bincount(inv, weights=s.astype(float)) > 0
            Ms.append((mu & su, su, np.sign(r.z)))
        def count(y):
            k = 0
            for mu, su, sg in Ms:
                n1, n0 = (mu).sum(), (su & ~mu).sum()
                if n1 < 5 or n0 < 5:
                    continue
                ys = y[su]; sd = ys.std() or 1e-9
                z = (y[mu].mean() - y[su & ~mu].mean()) / (sd * np.sqrt(1 / n1 + 1 / n0))
                k += (np.sign(z) == sg) and abs(z) >= 1.28
            return k
        real = count(yu)
        null = np.array([count(rng.permutation(yu)) for _ in range(300)])
        P(f"  {sl:14} {disc} → {conf}: {len(Ms)} 1D survivors · confirmed {real} vs null mean {null.mean():.1f} (95th {np.percentile(null, 95):.0f}) · P(null ≥ real) {(null >= real).mean():.3f}")

# ── D ──
P("\nD. SURGE_LONG candidate: pair EMA13−EMA50 gap < 0 (column entry_pair_ema20_ema50_gap_pct; sign split, no fitted threshold)")
G = D[D.sleeve == "SURGE_LONG"].copy()
G["gap"] = pd.to_numeric(G.entry_pair_ema20_ema50_gap_pct, errors="coerce")
G["zone"] = G.gap < 0
w = G.y[G.y > 0].mean(); l = G.y[G.y <= 0].mean()
be = abs(l) / (w + abs(l)) * 100
P(f"  sleeve breakeven WR (all kept SURGE fills, 1×): avg win {w:+.3f} / avg loss {l:+.3f} → BE-WR {be:.1f}%")
for rg in REGIMES + ["ALL"]:
    g = G if rg == "ALL" else G[G.regime == rg]
    z, r = g[g.zone], g[~g.zone & g.gap.notna()]
    diff, ppos = bootstrap_diff(g[g.gap.notna()], g.zone[g.gap.notna()].values)
    P(f"  {rg:13} gap<0: {L(z)} | gap≥0: {L(r)} | Δ {diff:+.3f} P(Δ<0) {1 - ppos:.2f}")
    if rg == "2025 BULL":
        for f in (True, False):
            P(f"      2025 {'top-50' if f else 'non-top-50'}: gap<0 {L(z[z.top50 == f])} | gap≥0 {L(r[r.top50 == f])}")
P("  dose-response (all regimes, gap bins):")
bins = [-99, -1, -0.5, 0, 0.5, 1, 2, 4, 99]
for iv, h in G.groupby(pd.cut(G.gap, bins), observed=True):
    P(f"    gap {str(iv):14}: {len(h) / 3:5.1f}/seed · WR {(h.y > 0).mean() * 100:4.0f}% · avg {h.y.mean():+.3f} · "
      + " · ".join(f"{rg[:7]} {h[h.regime == rg].y.mean():+.2f}({(h.regime == rg).sum() / 3:.0f})" for rg in REGIMES))
P("  per-seed avg of the gap<0 zone: " + " · ".join(f"seed {s} {G[G.zone & (G.seed == s)].y.mean():+.3f} vs rest {G[~G.zone & (G.seed == s)].y.mean():+.3f}" for s in (1, 2, 3)))
Z = G[G.zone]
# expectancy bar, way 1 (masks above) and way 2 (independent: rebuild from raw columns with a loop)
def bar(Zf, label):
    n_seed = len(Zf) / 3
    wr = (Zf.y > 0).mean() * 100
    dsum = Zf.groupby("day").y.agg(["sum", "size"])
    idx = rng.integers(0, len(dsum), size=(5000, len(dsum)))
    bs = dsum["sum"].values[idx].sum(1) / dsum["size"].values[idx].sum(1)
    loss = Zf[Zf.y < 0]
    pair_share = (loss.groupby("pair").y.sum() / loss.y.sum()).max() * 100
    day_share = (loss.groupby("day").y.sum() / loss.y.sum()).max() * 100
    P(f"  [{label}] N {n_seed:.1f}/seed ({len(Zf)} fills) · windows (days) {Zf.day.nunique()} · WR {wr:.1f}% vs BE {be:.1f}% → {'✓' if wr < be else '✗'} · "
      f"avg {Zf.y.mean():+.3f}, 95% day-bootstrap upper {np.percentile(bs, 95):+.3f} → {'✓' if np.percentile(bs, 95) < 0 else '✗'} (one-sided 95 %) · "
      f"max pair share of loss {pair_share:.0f}% · max day share {day_share:.0f}% → {'✓' if max(pair_share, day_share) < 50 else '✗'} · N≥15 {'✓' if n_seed >= 15 else '✗'}")
bar(Z, "way 1 · all regimes")
raw = pd.read_csv(os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv"), low_memory=False,
                  usecols=["sleeve", "pair", "t", "pct_new", "entry_pair_ema20_ema50_gap_pct", "seed", "regime"])
raw = raw[raw.sleeve == "SURGE_LONG"]
zz = []
for row in raw.itertuples():
    if float(row.entry_pair_ema20_ema50_gap_pct) < 0:
        zz.append(dict(y=float(row.pct_new), pair=row.pair, day=pd.Timestamp(row.t).floor("D"), regime=row.regime))
Z2 = pd.DataFrame(zz)
bar(Z2, "way 2 · raw loop")
for rg in REGIMES:
    bar(Z[Z.regime == rg], f"way 1 · {rg}")
# the effect of blocking it, per regime, with the 30–50 % haircut
for rg in REGIMES + ["ALL"]:
    g = G if rg == "ALL" else G[G.regime == rg]
    keep = g[~g.zone]
    d = (keep.y.sum() - g.y.sum()) / 3
    P(f"  block gap<0 · {rg:13}: sleeve {g.y.mean():+.3f}/fill Σ {g.y.sum() / 3:+.2f}/seed → kept {keep.y.mean():+.3f}/fill Σ {keep.y.sum() / 3:+.2f}/seed · ΔΣ {d:+.2f} %-pts/seed "
      f"(haircut 30–50 % → {d * 0.5:+.2f} … {d * 0.7:+.2f})")
# concentration of the zone's losses per regime
for rg in REGIMES:
    z = Z[(Z.regime == rg) & (Z.y < 0)]
    if len(z):
        pr = (z.groupby("pair").y.sum() / z.y.sum()).sort_values(ascending=False)
        P(f"  {rg} zone-loss concentration: top pairs {', '.join(f'{p} {v * 100:.0f}%' for p, v in pr.head(3).items())}")

# D2: is gap<0 a PAIR effect or a trigger/window (macro) effect? Within-window paired difference: windows = (seed, 30-min bucket of
# open time) holding BOTH a gap<0 and a gap≥0 fill; per-window mean(zone) − mean(rest), day-bootstrap over windows.
G["win"] = G.seed.astype(str) + "|" + G.t.dt.floor("30min").astype(str)
W = G[G.gap.notna()].groupby(["win", "zone"]).y.mean().unstack()
W = W.dropna()
W["day"] = [pd.Timestamp(k.split("|")[1]).floor("D") for k in W.index]
W["d"] = W[True] - W[False]
dd = W.groupby("day").d.mean()
bs = [dd.values[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
P(f"  D2 within-trigger-window test: {len(W)} seed-windows on {len(dd)} days hold both zones · mean(gap<0) − mean(gap≥0) = {W.d.mean():+.3f} "
  f"(day-mean {dd.mean():+.3f}, 95 % day CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]) · by regime "
  + " · ".join(f"{rg[:7]} {W[[G.loc[G.win == k, 'regime'].iloc[0] == rg for k in W.index]].d.mean():+.3f} (n {sum(G.loc[G.win == k, 'regime'].iloc[0] == rg for k in W.index)})" for rg in REGIMES))
# trigger-level macro share: fraction of a window's picks with gap<0 vs window outcome
WW = G[G.gap.notna()].groupby("win").agg(share=("zone", "mean"), y=("y", "mean"), day=("day", "first"))
for lab, m in (("windows with no gap<0 pick", WW.share == 0), ("mixed", (WW.share > 0) & (WW.share < 1)), ("all picks gap<0", WW.share == 1)):
    P(f"     {lab:28}: {m.sum():4d} seed-windows · mean window outcome {WW.y[m].mean():+.3f}")

# ── E ──
P("\nE. Spike-Fade bear-discovered macro cell in DAY units: entry_bull_pct < 21.72 ∧ entry_btc_atr_pct ≥ 0.1445 (thresholds frozen from the bear screen)")
F = D[D.sleeve == "Spike-Fade"].copy()
F["cell"] = (pd.to_numeric(F.entry_bull_pct, errors="coerce") < 21.72) & (pd.to_numeric(F.entry_btc_atr_pct, errors="coerce") >= 0.1445)
for rg in REGIMES:
    g = F[F.regime == rg]
    dd = g.groupby("day").agg(y=("y", "mean"), c=("cell", "mean"))
    cd, rd = dd[dd.c >= 0.5], dd[dd.c < 0.5]
    diff, ppos = bootstrap_diff(g, g.cell.values)
    P(f"  {rg:13}: cell {L(g[g.cell])} | rest {L(g[~g.cell])} | DAY units: cell-days {len(cd)} mean {cd.y.mean():+.3f} vs other days {len(rd)} mean {rd.y.mean():+.3f} · fill Δ {diff:+.3f} P(Δ<0) {1 - ppos:.2f}")
w = F.y[F.y > 0].mean(); l = F.y[F.y <= 0].mean()
P(f"  Spike-Fade BE-WR {abs(l) / (w + abs(l)) * 100:.1f}% (avg win {w:+.3f} / loss {l:+.3f}); 2025 fades are {100 - F[F.regime == '2025 BULL'].top50.mean() * 100:.0f}% non-top-50 → exits on 1m bars")

# ── F ──
P("\nF. FLIP-short context")
Fl = D[D.sleeve == "FLIP-short"].copy()
off = pd.to_numeric(Fl.entry_btc_off30d_high_pct, errors="coerce")
for lab, m in (("BTC ≤ −15 % off 30d high (washed-out state)", off <= -15), ("−15 … −8", (off > -15) & (off <= -8)), ("> −8", off > -8)):
    P(f"  {lab:44}: {L(Fl[m])} · by regime " + " · ".join(f"{rg[:7]} {Fl[m & (Fl.regime == rg)].y.mean():+.3f}({(m & (Fl.regime == rg)).sum() / 3:.0f})" for rg in REGIMES))
wo = Fl[(Fl.t >= "2026-06-17") & (Fl.t < "2026-07-03")]
P(f"  replay flips inside the real washed-out window Jun-17→Jul-2: {L(wo)}")
for sl in ["FLIP-short", "Spike-Fade", "BullRun-Long", "SURGE_LONG", "BearRun-Short"]:
    g = D[D.sleeve == sl]
    P(f"  {sl:14} top-50 share by regime: " + " · ".join(f"{rg} {g[g.regime == rg].top50.mean() * 100:.0f}%" for rg in REGIMES if (g.regime == rg).any())
      + f" · stop-loss exit share {g.close_reason.astype(str).str.contains('STOP').mean() * 100:.0f}%")
st = Fl.close_reason.astype(str).str.contains("STOP_LOSS")
P("  FLIP stop share by regime × top-50: " + " · ".join(f"{rg[:7]} top50 {st[(Fl.regime == rg) & Fl.top50].mean() * 100:.0f}% / non {st[(Fl.regime == rg) & ~Fl.top50].mean() * 100:.0f}%" for rg in REGIMES))

# ── G ── power: minimum detectable Δ (80 % power, two-sided 5 %) for a median split, in the screen's units
P("\nG. power — minimum detectable zone-vs-rest Δ (%/fill) for a 50/50 split at 80 % power, α 5 % two-sided (units as screened)")
for sl in SLEEVES:
    for rg in REGIMES:
        g = D[(D.sleeve == sl) & (D.regime == rg)]
        if not len(g):
            continue
        u_pd = g.groupby("pd_key").y.mean(); u_d = g.groupby("day").y.mean()
        mde = lambda u: 2.8 * u.std() * np.sqrt(4 / len(u)) if len(u) > 3 else np.nan
        P(f"  {sl:14} {rg:13}: pair-day units {len(u_pd):4d} → MDE {mde(u_pd):.2f} · day units {len(u_d):4d} → MDE {mde(u_d):.2f} · sleeve |avg| {abs(g.y.mean()):.2f}")

# ── H ── SURGE gap<0: leave-one-month-out (block effect per fill must stay > 0 with every month removed) + seed-1-only bar
P("\nH. SURGE_LONG gap<0 robustness: leave-one-month-out (Δ = avg kept − avg sleeve, %/fill) and seed-by-seed bar")
mos = sorted(G.t.dt.strftime("%Y-%m").unique()); deltas = []
for mo in mos:
    g = G[G.t.dt.strftime("%Y-%m") != mo]
    deltas.append((mo, g[~g.zone].y.mean() - g.y.mean(), g[g.zone].y.mean()))
dmin = min(deltas, key=lambda x: x[1]); zmax = max(deltas, key=lambda x: x[2])
P(f"  {len(mos)} months · Δ range {min(d[1] for d in deltas):+.3f} … {max(d[1] for d in deltas):+.3f} (weakest without {dmin[0]}) · zone avg range "
  f"{min(d[2] for d in deltas):+.3f} … {max(d[2] for d in deltas):+.3f} (highest without {zmax[0]}) · zone avg < 0 in {sum(d[2] < 0 for d in deltas)}/{len(mos)}")
zm = G[G.zone].groupby(G.t.dt.strftime("%Y-%m")).y.agg(["mean", "size"])
P(f"  zone months negative: {(zm['mean'] < 0).sum()}/{len(zm)} · " + " · ".join(f"{k[2:]} {v['mean']:+.2f}({v['size'] / 3:.0f})" for k, v in zm.iterrows()))
for s in (1, 2, 3):
    bar(Z[Z.seed == s].assign(), f"seed {s} only (N shown /3)")

# ── I ── Spike-Fade bear cell on REAL fills (master, today's stack) + concentration in the bear replay
P("\nI. Spike-Fade cell (entry_bull_pct < 21.72 ∧ entry_btc_atr_pct ≥ 0.1445) on master kept fades + bear concentration")
M = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
mf = M[(M.entry_strategy == "SPIKE_FADE") & (M.stack_keep == True) & (M.is_probe != True)].copy()
mf["cell"] = (pd.to_numeric(mf.entry_bull_pct, errors="coerce") < 21.72) & (pd.to_numeric(mf.entry_btc_atr_pct, errors="coerce") >= 0.1445)
mf["y"] = pd.to_numeric(mf.stack_pct, errors="coerce"); mf["day"] = pd.to_datetime(mf.opened_at, format="ISO8601").dt.floor("D")
for lab, h in (("cell", mf[mf.cell]), ("rest", mf[~mf.cell])):
    P(f"  master kept fades {lab}: N {len(h)} · {h.day.nunique()} d · WR {(h.y > 0).mean() * 100:.0f}% · avg {h.y.mean():+.3f}")
zb = F[(F.regime == "2026-H1 BEAR") & F.cell & (F.y < 0)]
pr = (zb.groupby("pair").y.sum() / zb.y.sum()).sort_values(ascending=False); dy = (zb.groupby("day").y.sum() / zb.y.sum()).sort_values(ascending=False)
P(f"  bear cell loss concentration: top pair {pr.index[0]} {pr.iloc[0] * 100:.0f}% · top-2 pairs {pr.iloc[:2].sum() * 100:.0f}% · top day {dy.index[0].date()} {dy.iloc[0] * 100:.0f}%")
P(f"  FLIP hi-ATR (entry_atr_pct ≥ 0.698, whole-sleeve T2) on master kept flips: see report (washed-out vs after)")

open(os.path.join(REP, "YR6_WL_OTHERS_checks.txt"), "w").write("\n".join(OUT) + "\n")
