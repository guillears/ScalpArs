#!/usr/bin/env python3
"""💪📉 Oct-4 (operator: "Run the pair strength vs BTC entry analysis for momentum longs").

Momentum LONG, full-size, no probes, no MANUAL. Lever under test = ENTRY selection by the pair's strength relative to BTC.

PRE-DECLARED GRID (written before any result was read — printed verbatim at the top of the report):
  Family (16 variables, every one scored from reports/backtest_cache/k5m_full CLOSED 5m bars before entry, unscored ≠ rest):
    REL  = pair return − BTC return over 15m (3 bars) · 1h (12) · 4h (48) · 24h (288)          rel15 rel1h rel4h rel24h
    PAIR = pair return alone over the same windows                                              p15 p1h p4h p24h
    RANK = entry_pair_rank (24 h volume rank, live/engine stamp; bands 1–10 / 11–20 / 21–35 / 36–50)   rank
    BTC extension = BTC close vs 5m EMA13 / EMA50 (%), BTC 1h / 4h / 24h return                   btc_d13 btc_d50 btc1h btc4h btc24h
    Deep-dive lead continuity (entry_feature_factory): REL_1d_ret24 (24 DAILY bars ≈ 24 days — the 09-29 "REL_1d_ret24" is NOT 24 h)
      and REL_4h_px_vs_ema200 (unscored before 2026-02-02: 200×4h EMA warm-up)                    L_rel24d L_relE200
  Cuts: quintiles (and deciles for the dose-response) fixed on the pooled yr4 distribution; master binned with the SAME edges.
  Halves: H1 = Jan-04…Apr-30, H2 = May-01…Oct-03. CIs: calendar-day block bootstrap (seeds pooled), 4,000 draws.
  2D screen: every 1D quintile cell (16×5) + every 2D quintile cell (120 pairs × 25) = 3,080 cells; cell must hold ≥ 15 fills/seed.
    Statistic T = (cell avg − population avg) / (σ·√(n_c(N−n_c)/N)) (fill-level t). Null = outcomes shuffled WITHIN seed×day,
    1,000 draws, family-wise 95th pct of max T (multiplier side) and 5th pct of min T (block side), same min-size rule.
    Survivor = beats the family-wise null AND both halves agree in sign (cell − rest).
  Pre-declared candidates (tested regardless of the screen): C1 rel24h bottom quintile (BLOCK) · C2 rel24h bottom decile (BLOCK) ·
    C3 laggard × BTC-extended = rel24h Q1 ∧ btc_d50 Q5 (BLOCK) · C4 rank 36–50 (BLOCK) · C5 rel24h top quintile (MULTIPLIER 1.5×) ·
    C6 L_rel24d bottom quintile (BLOCK, deep-dive lead) · each also with CHOP∧BURST (eff72 ≤ 0.007 ∧ burst, registered) pre-removed.
Data: yr4 = reports/ENGINE_REPLAY_YR4_ML_fills.csv (3 seeds, today's config); CHOP/BURST flags from
reports/ML_CHOP_BURST_DEEP_2026-10-04_yr4_flags.csv; master = reports/ML_WATCHLIST_master_features.csv (113 current-stack ML fills).
Out: reports/ML_PAIR_STRENGTH_2026-10-04.md (+ _yr4_features.csv, _master_features.csv, _screen.csv, _dose.csv)
  venv/bin/python scripts/ml_pair_strength.py
"""
import os, sys, itertools
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import entry_feature_factory as EF

R = "reports"; K5 = f"{R}/backtest_cache/k5m_full"; NB = 4000; NNULL = 1000; MINN = 15
SPLIT = pd.Timestamp("2026-05-01"); WARM200 = pd.Timestamp("2026-02-02")
rng = np.random.default_rng(20261004)
OUT = []
W = lambda s="": OUT.append(s)
n_ = lambda d, c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)

# ═════════════════════ data ═════════════════════
Y = pd.read_csv(f"{R}/ENGINE_REPLAY_YR4_ML_fills.csv", low_memory=False)
FL = pd.read_csv(f"{R}/ML_CHOP_BURST_DEEP_2026-10-04_yr4_flags.csv", low_memory=False, usecols=["seed", "pair", "opened_at", "CHOP", "BURST", "B2"])
Y = Y.merge(FL, on=["seed", "pair", "opened_at"], how="left", validate="1:1")
assert Y.CHOP.notna().all(), "chop/burst flags missing for some yr4 fills"
Y["o"] = pd.to_datetime(Y.opened_at.astype(str).str[:19])
Y["pct"] = n_(Y, "pct"); Y["usd"] = n_(Y, "pnl")
M = pd.read_csv(f"{R}/ML_WATCHLIST_master_features.csv", low_memory=False)
M["o"] = pd.to_datetime(M.opened_at.astype(str).str[:19])
M["CHOP"] = n_(M, "eff72") <= 0.007; M["BURST"] = M.burst.astype(str) == "True"
for d in (Y, M):
    d["day"] = d.o.dt.strftime("%Y-%m-%d"); d["mon"] = d.o.dt.strftime("%Y-%m"); d["h1"] = d.o < SPLIT
    d["CB"] = (d.CHOP.astype(bool) & d.BURST.astype(bool))
SEEDS = sorted(Y.seed.unique()); NS = len(SEEDS)

# ═════════════════════ features from closed 5m bars ═════════════════════
_k = {}


def k5(sym):
    if sym not in _k:
        k = pd.read_csv(f"{K5}/{sym}.csv", usecols=["open_time", "c"])
        t = (pd.to_datetime(k.open_time, unit="ms") + pd.Timedelta("5min")).values      # bar CLOSE time
        c = k.c.astype(float).values
        e13 = pd.Series(c).ewm(span=13, adjust=False).mean().values; e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
        _k[sym] = (t, c, e13, e50)
    return _k[sym]


WINS = {"15": 3, "1h": 12, "4h": 48, "24h": 288}


def build(d):
    f = pd.DataFrame(index=d.index)
    tb, cb, e13, e50 = k5("BTCUSDT")
    ib = np.searchsorted(tb, d.o.values, side="right") - 1
    for w, k in WINS.items():
        ok = ib - k >= 0
        f[f"btc{w}"] = np.where(ok, (cb[ib] / cb[np.maximum(ib - k, 0)] - 1) * 100, np.nan)
    f["btc_d13"] = (cb[ib] / e13[ib] - 1) * 100; f["btc_d50"] = (cb[ib] / e50[ib] - 1) * 100
    f["bar_lag_s"] = (d.o.values - tb[ib]).astype("timedelta64[s]").astype(float)
    for pair, g in d.groupby("pair"):
        tp, cp, _, _ = k5(pair)
        ip = np.searchsorted(tp, g.o.values, side="right") - 1
        stale = (g.o.values - tp[np.maximum(ip, 0)]).astype("timedelta64[s]").astype(float) > 600   # pair bar older than 2 bars → unscored
        for w, k in WINS.items():
            ok = (ip - k >= 0) & (ip >= 0) & ~stale
            f.loc[g.index, f"p{w}"] = np.where(ok, (cp[np.maximum(ip, 0)] / cp[np.maximum(ip - k, 0)] - 1) * 100, np.nan)
    for w in WINS:
        f[f"rel{w}"] = f[f"p{w}"] - f[f"btc{w}"]
    f["rank"] = n_(d, "entry_pair_rank")
    X = EF.features(d[["opened_at", "pair"]])
    f["L_rel24d"] = X["REL_1d_ret24"]; f["L_relE200"] = X["REL_4h_px_vs_ema200"].where(d.o >= WARM200)
    f["chk_p1h_EF"] = X["PAIR_5m_ret12"]; f["chk_b1h_EF"] = X["BTC_5m_ret12"]
    f["zoneD"] = (X["BTC_1d_ret1"] > 0.092) & (X["PAIR_1h_gap_ema20_200"] <= 0)
    return f.rename(columns={"p15": "p15", "btc15": "btc15", "rel15": "rel15"})


FY = build(Y); FM = build(M)
VARS = ["rel15", "rel1h", "rel4h", "rel24h", "p15", "p1h", "p4h", "p24h", "rank", "btc_d13", "btc_d50", "btc1h", "btc4h", "btc24h", "L_rel24d", "L_relE200"]
LAB = {"rel15": "REL 15m", "rel1h": "REL 1h", "rel4h": "REL 4h", "rel24h": "REL 24h", "p15": "pair 15m", "p1h": "pair 1h", "p4h": "pair 4h",
       "p24h": "pair 24h", "rank": "24h-volume rank", "btc_d13": "BTC vs EMA13 (5m)", "btc_d50": "BTC vs EMA50 (5m)", "btc1h": "BTC 1h ret",
       "btc4h": "BTC 4h ret", "btc24h": "BTC 24h ret", "L_rel24d": "REL 24-day (deep-dive lead)", "L_relE200": "REL vs 4h EMA200 (lead)"}
for c in VARS + ["zoneD"]:
    Y[c] = FY[c].values; M[c] = FM[c].values

# internal consistency: own 1h returns must equal the factory's 5m ret12 (same bars, independent code)
d1 = (FY.p1h - FY.chk_p1h_EF).abs().max(); d2 = (FY.btc1h - FY.chk_b1h_EF).abs().max()
assert d1 < 1e-6 and d2 < 1e-6, f"own vs factory 1h return mismatch {d1} {d2}"
# vs live stamps on the master: BTC dist from EMA13
st = n_(M, "entry_btc_dist_from_ema13_pct"); okst = st.notna() & M.btc_d13.notna()
r_d13 = np.corrcoef(st[okst], M.btc_d13[okst])[0, 1]
sty = n_(Y, "entry_btc_dist_from_ema13_pct"); oky = sty.notna() & Y.btc_d13.notna()
r_d13y = np.corrcoef(sty[oky], Y.btc_d13[oky])[0, 1]

# ═════════════════════ helpers ═════════════════════
aw, al = Y.pct[Y.pct > 0].mean(), Y.pct[Y.pct <= 0].mean(); BE = abs(al) / (aw + abs(al))
DAYS = np.array(sorted(Y.day.unique())); Y["di"] = Y.day.map({d: i for i, d in enumerate(DAYS)}).values
BW = np.stack([np.bincount(rng.integers(0, len(DAYS), len(DAYS)), minlength=len(DAYS)) for _ in range(NB)]).astype(float)
PCT = Y.pct.values; H1 = Y.h1.values


def dsum(mask):
    return (np.bincount(Y.di[mask], weights=PCT[mask], minlength=len(DAYS)), np.bincount(Y.di[mask], minlength=len(DAYS)).astype(float))


def diff_ci(a, b):
    """avg(a) − avg(b) with a day-block bootstrap CI."""
    sa, ca = dsum(a); sb, cb = dsum(b); x, y = BW @ ca, BW @ cb; ok = (x > 0) & (y > 0)
    d = (BW @ sa)[ok] / x[ok] - (BW @ sb)[ok] / y[ok]
    return PCT[a].mean() - PCT[b].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def avg_ci(m):
    s, c = dsum(m); cc = BW @ c; mm = (BW @ s)[cc > 0] / cc[cc > 0]
    return np.percentile(mm, 2.5), np.percentile(mm, 97.5), np.percentile(mm, 95)


def share(g, key):
    l = -g.pct[g.pct < 0].groupby(g[key][g.pct < 0]).sum()
    return (l.max() / l.sum(), l.idxmax()) if len(l) and l.sum() > 0 else (0.0, "-")


def edges(v, q):
    x = Y[v].dropna()
    return np.unique(np.quantile(x, np.linspace(0, 1, q + 1)[1:-1]))


def binv(d, v, e):
    x = d[v].values.astype(float); b = np.digitize(x, e, right=True).astype(float); b[np.isnan(x)] = np.nan
    return b


def fmt(m, sel=None):
    sel = np.ones(len(Y), bool) if sel is None else sel; mm = m & sel
    if mm.sum() == 0:
        return "—"
    return f"{mm.sum() / NS:.1f} · {(PCT[mm] > 0).mean() * 100:.0f}% · {PCT[mm].mean():+.3f}"


# ═════════════════════ report header ═════════════════════
W("# 💪📉 Momentum LONG — pair strength vs BTC at entry (2026-10-04)")
W()
W("## 0 · Pre-declared grid (fixed before reading any result)")
W()
W(__doc__.split("PRE-DECLARED GRID (written before any result was read — printed verbatim at the top of the report):")[1].split("Data:")[0].strip("\n"))
W()
cov = {v: (Y[v].notna().sum(), M[v].notna().sum()) for v in VARS}
W(f"**Coverage (scored N / N — unscored fills are EXCLUDED from that variable's tables, never 'rest'):** yr4 {len(Y)} fills ({len(Y) / NS:.0f}/seed, "
  f"{Y.day.nunique()} days) · master {len(M)} fills ({M.day.nunique()} days). k5m_full cache ends "
  f"{pd.Timestamp(k5('BTCUSDT')[0][-1])} UTC (covers both). " + " · ".join(f"{v} {a}/{len(Y)} | {b}/{len(M)}" for v, (a, b) in cov.items()))
W()
W(f"**Feature checks:** own 1h pair/BTC returns = entry_feature_factory 5m ret12 on every yr4 fill (max |Δ| {d1:.1e} / {d2:.1e}); rebuilt BTC "
  f"distance from 5m EMA13 vs the stamp entry_btc_dist_from_ema13_pct: master r = {r_d13:.3f} (N {okst.sum()}), yr4 r = {r_d13y:.3f} (N {oky.sum()}). "
  f"Median lag entry − last closed BTC bar {Y.bar_lag_s.median() if 'bar_lag_s' in Y else FY.bar_lag_s.median():.0f} s.")
W()
W(f"**ML breakeven WR (yr4)** = {abs(al):.3f} / ({aw:.3f} + {abs(al):.3f}) = **{BE * 100:.1f} %**. Sleeve yr4: {(PCT > 0).mean() * 100:.0f} % WR · "
  f"{PCT.mean():+.3f} %/trade · H1 {fmt(H1)} · H2 {fmt(~H1)} (N/seed · WR · avg). Because the whole replay sleeve is below breakeven, "
  f"absolute WR/avg levels are not informative there — only Δ vs rest carries over (replay selection bias, DECISION_LOG 163).")
W()

# ═════════════════════ §1 dose-response ═════════════════════
W("## 1 · Dose-response per variable (quintiles on the pooled yr4 distribution)")
W()
W("Cells: N/seed · WR · avg % ; Δ = bucket avg − rest avg (same half, scored fills) [95 % day-block CI]. ρ = Spearman(bucket index, bucket avg) per half; "
  "'monotone' = |ρ| ≥ 0.9 in both halves with the same sign.")
W()
DR = []
for v in VARS:
    e = edges(v, 5); b = binv(Y, v, e); sc = ~np.isnan(b)
    W(f"### {LAB[v]} (`{v}`) — scored {sc.sum()}/{len(Y)}; edges {', '.join(f'{x:.3g}' for x in e)}")
    W()
    W("| quintile | range | YEAR N/seed · WR · avg | Δ vs rest [CI] | H1 N/seed · WR · avg | H1 Δ | H2 N/seed · WR · avg | H2 Δ |")
    W("|---|---|---|---|---|---|---|---|")
    rh = {"H1": [], "H2": []}
    for q in range(len(e) + 1):
        m = b == q; lo = Y[v][m].min(); hi = Y[v][m].max(); cells = []
        dd, l_, h_ = diff_ci(m, sc & ~m)
        for nm, hs in (("H1", H1), ("H2", ~H1)):
            dh, lh, hh = diff_ci(m & hs, sc & ~m & hs); rh[nm].append(PCT[m & hs].mean())
            cells.append((fmt(m, hs), f"{dh:+.3f} [{lh:+.2f}, {hh:+.2f}]"))
        W(f"| Q{q + 1} | {lo:.3g} … {hi:.3g} | {fmt(m)} | {dd:+.3f} [{l_:+.2f}, {h_:+.2f}] | {cells[0][0]} | {cells[0][1]} | {cells[1][0]} | {cells[1][1]} |")
        DR.append(dict(var=v, q=q + 1, n_seed=m.sum() / NS, wr=(PCT[m] > 0).mean(), avg=PCT[m].mean(), d=dd, d_lo=l_, d_hi=h_))
    rho = {k: pd.Series(x).rank().corr(pd.Series(range(len(x)), dtype=float)) for k, x in rh.items()}
    mono = abs(rho["H1"]) >= 0.9 and abs(rho["H2"]) >= 0.9 and np.sign(rho["H1"]) == np.sign(rho["H2"])
    W()
    W(f"ρ H1 {rho['H1']:+.2f} · H2 {rho['H2']:+.2f} → **{'monotone' if mono else 'not monotone'}**")
    # deciles (compact)
    e10 = edges(v, 10); b10 = binv(Y, v, e10)
    W(f"Deciles avg % H1: {' '.join(f'{PCT[(b10 == q) & H1].mean():+.2f}' for q in range(len(e10) + 1))} | "
      f"H2: {' '.join(f'{PCT[(b10 == q) & ~H1].mean():+.2f}' for q in range(len(e10) + 1))}")
    W()
    DR[-1]["rho_h1"], DR[-1]["rho_h2"], DR[-1]["mono"] = rho["H1"], rho["H2"], mono

# rank bands
W("### 24h-volume rank — operator bands")
W()
W("| band | yr4 YEAR N/seed · WR · avg | Δ vs rest [CI] | H1 | H2 | master N · WR · avg % · net $ |")
W("|---|---|---|---|---|---|")
BANDS = [(1, 10), (11, 20), (21, 35), (36, 50)]
for a, z in BANDS:
    m = ((Y["rank"] >= a) & (Y["rank"] <= z)).values; mm = (M["rank"] >= a) & (M["rank"] <= z); g = M[mm]
    dd, l_, h_ = diff_ci(m, ~m) if m.any() else (np.nan, np.nan, np.nan)
    W(f"| {a}–{z} | {fmt(m)} | {dd:+.3f} [{l_:+.2f}, {h_:+.2f}] | {fmt(m, H1)} | {fmt(m, ~H1)} | "
      f"{len(g)} · {(g.pct > 0).mean() * 100 if len(g) else 0:.0f}% · {g.pct.mean() if len(g) else 0:+.3f} · ${g.net.sum():+,.0f} |")
W()
W("(Ranks 1–10 are absent by construction: the mega-cap exclusion keeps the momentum sleeve off the top-10 volume pairs.)")
W()

# laggard × BTC-extended grid
W("### Pre-declared interaction: REL 24h quintile × BTC vs EMA50 quintile (avg % · N/seed; H1 / H2)")
W()
eR, eB = edges("rel24h", 5), edges("btc_d50", 5); bR, bB = binv(Y, "rel24h", eR), binv(Y, "btc_d50", eB)
W("| REL24h ↓ / BTC-EMA50 → | " + " | ".join(f"B{j + 1}" for j in range(5)) + " |"); W("|---" * 6 + "|")
for i in range(5):
    cells = []
    for j in range(5):
        m = (bR == i) & (bB == j)
        cells.append(f"{PCT[m].mean():+.2f} ({m.sum() / NS:.0f}) · {PCT[m & H1].mean():+.2f} / {PCT[m & ~H1].mean():+.2f}" if m.any() else "—")
    W(f"| R{i + 1} | " + " | ".join(cells) + " |")
W()

# ═════════════════════ §2 exhaustive 2D with within-day null ═════════════════════
# The 3 replay seeds share ~55 % of their fills (same pair, ±10 min, outcome corr ≈ 0.85). A fill-level t on pooled seeds with an
# independent per-seed shuffle would make the null too NARROW (it breaks the cross-seed duplication the observed data keeps). The
# screen therefore runs on UNIQUE TRADES: fills of different seeds on the same pair within 10 min = one trade (pct = member mean).
Ys = Y.sort_values(["pair", "o"]); cid = np.zeros(len(Y), int); cur = -1; lastp, lasto, seeds_in = None, None, set()
for i, p_, o_, sd_ in zip(Ys.index, Ys.pair, Ys.o, Ys.seed):
    if p_ != lastp or (o_ - lasto) > pd.Timedelta("10min") or sd_ in seeds_in:
        cur += 1; seeds_in = set(); lasto = o_
    cid[i] = cur; seeds_in.add(sd_); lastp = p_
Y["tid"] = cid
U = Y.groupby("tid").agg(pct=("pct", "mean"), day=("day", "first"), h1=("h1", "first"), nseed=("seed", "nunique"),
                          **{v: (v, "first") for v in VARS}).reset_index()
NU = len(U); MINU = int(np.ceil(MINN * NU / (len(Y) / NS)))
W("## 2 · Exhaustive 1D+2D quintile screen vs a within-day shuffled-outcome null (unique trades)")
W()
W(f"Unit = unique trade: {len(Y)} seed-fills collapse to **{NU} unique trades** (in 1/2/3 seeds: "
  f"{(U.nseed == 1).sum()}/{(U.nseed == 2).sum()}/{(U.nseed == 3).sum()}). Why: the seeds share ~55 % of fills with outcome corr ≈ 0.85, so a "
  f"per-seed shuffle would give a null that is too narrow. Minimum cell = {MINU} unique trades (= the 15/seed share of the sleeve). "
  f"Quintile edges are the pooled-yr4 edges of §1.")
W()
UP = U.pct.values; UH1 = U.h1.values
QU = {v: binv(U, v, edges(v, 5)) for v in VARS}
QB = {v: binv(Y, v, edges(v, 5)) for v in VARS}
cells, cmask, cpop = [], [], []
POPS = {}
for v in VARS:
    pk = (v,); POPS.setdefault(pk, ~np.isnan(QU[v]))
    for q in range(5):
        cells.append((v, q, None, None)); cmask.append(QU[v] == q); cpop.append(pk)
for a, b in itertools.combinations(VARS, 2):
    pk = (a, b); POPS[pk] = ~np.isnan(QU[a]) & ~np.isnan(QU[b])
    for qa in range(5):
        for qb in range(5):
            cells.append((a, qa, b, qb)); cmask.append((QU[a] == qa) & (QU[b] == qb)); cpop.append(pk)
C = np.array(cmask, dtype=float)                                  # K × NU
nc = C.sum(1); keep = nc >= MINU
pk_list = list(POPS); pidx = np.array([pk_list.index(p) for p in cpop]); PM = np.array([POPS[p] for p in pk_list], dtype=float)
NP = PM.sum(1)


def tstat(P):                                                     # P: NU × D
    s = C @ P; S = PM @ P; Q = PM @ (P ** 2)
    mu = S / NP[:, None]; sd = np.sqrt(np.maximum(Q / NP[:, None] - mu ** 2, 1e-12))
    mu_c, sd_c, n_p = mu[pidx], sd[pidx], NP[pidx][:, None]
    n = np.maximum(nc[:, None], 1)
    return (s / n - mu_c) / (sd_c * np.sqrt(np.maximum(n * (n_p - n) / n_p, 1e-9)) / n)


T = tstat(UP[:, None])[:, 0]
grp = pd.factorize(U.day)[0]; order = np.argsort(grp, kind="stable"); bnd = np.r_[0, np.cumsum(np.bincount(grp))]
REL_ONLY = {"rel15", "rel1h", "rel4h", "rel24h", "rank"}
relm = np.array([(a in REL_ONLY) and (b is None or b in REL_ONLY) for a, _, b, _ in cells])
NULLMAX, NULLMIN, NULLCNT, RNMAX, RNMIN = [], [], [], [], []
for it in range(0, NNULL, 100):
    Pm = np.empty((NU, 100))
    for j in range(100):
        p = UP[order].copy()
        for a_, b_ in zip(bnd[:-1], bnd[1:]):
            if b_ - a_ > 1:
                p[a_:b_] = rng.permutation(p[a_:b_])
        Pm[order, j] = p
    TT = tstat(Pm); Tn = TT[keep]; Tr = TT[keep & relm]
    RNMAX += list(Tr.max(0)); RNMIN += list(Tr.min(0))
    NULLMAX += list(Tn.max(0)); NULLMIN += list(Tn.min(0)); NULLCNT += list((np.abs(Tn) >= 1.96).sum(0))
thrH, thrL = np.percentile(NULLMAX, 95), np.percentile(NULLMIN, 5)
SC = pd.DataFrame(cells, columns=["v1", "q1", "v2", "q2"]); SC["n_trades"] = nc; SC["T"] = T; SC["eligible"] = keep
SC["avg"] = (C @ UP) / np.maximum(nc, 1)
for nm, hs in (("h1", UH1), ("h2", ~UH1)):
    cn = C @ hs.astype(float); cs = C @ (UP * hs); pn = PM @ hs.astype(float); ps = PM @ (UP * hs)
    SC[f"avg_{nm}"] = cs / np.maximum(cn, 1); SC[f"rest_{nm}"] = (ps[pidx] - cs) / np.maximum(pn[pidx] - cn, 1)
    SC[f"d_{nm}"] = SC[f"avg_{nm}"] - SC[f"rest_{nm}"]
SC["agree"] = np.sign(SC.d_h1) == np.sign(SC.d_h2)
SC["label"] = [f"{LAB[a]} Q{qa + 1}" + (f" ∧ {LAB[b]} Q{qb + 1}" if b else "") for a, qa, b, qb in cells]
E = SC[SC.eligible]
surv_lo = E[(E["T"] <= thrL) & E.agree]; surv_hi = E[(E["T"] >= thrH) & E.agree]
SC.to_csv(f"{R}/ML_PAIR_STRENGTH_2026-10-04_screen.csv", index=False)
W(f"Cells: {len(SC)} built, {keep.sum()} eligible (≥ {MINU} unique trades). Family-wise null (outcomes shuffled within calendar day, {NNULL} draws): "
  f"5th pct of min T = **{thrL:+.2f}**, 95th pct of max T = **{thrH:+.2f}** (null median min/max {np.median(NULLMIN):+.2f} / {np.median(NULLMAX):+.2f}). "
  f"Observed min T {E['T'].min():+.2f}, max T {E['T'].max():+.2f}.")
W()
W(f"Eligible cells beyond the family-wise threshold: block side {int((E['T'] <= thrL).sum())} (with both-halves agreement {len(surv_lo)}), "
  f"multiplier side {int((E['T'] >= thrH).sum())} (agreeing {len(surv_hi)}). Cells with |T| ≥ 1.96 (pointwise): observed "
  f"{int((E['T'].abs() >= 1.96).sum())} of {len(E)} vs null median {np.median(NULLCNT):.0f} (95th pct {np.percentile(NULLCNT, 95):.0f}) — "
  f"P(null count ≥ observed) = {(np.array(NULLCNT) >= (E['T'].abs() >= 1.96).sum()).mean():.3f}.")
W()
W("Top 12 block-side and multiplier-side cells (eligible), whether or not they beat the null:")
W()
W("| side | cell | unique trades | avg % | T | H1 cell / rest | H2 cell / rest | halves agree | beats null |")
W("|---|---|---|---|---|---|---|---|---|")
for side, g in (("block", E.nsmallest(12, "T")), ("mult", E.nlargest(12, "T"))):
    for r in g.itertuples():
        beats = (r.T <= thrL) if side == "block" else (r.T >= thrH)
        W(f"| {side} | {r.label} | {r.n_trades:.0f} | {r.avg:+.3f} | {r.T:+.2f} | {r.avg_h1:+.3f} / {r.rest_h1:+.3f} | {r.avg_h2:+.3f} / {r.rest_h2:+.3f} | "
          f"{'Y' if r.agree else '·'} | {'**YES**' if beats and r.agree else ('yes (halves disagree)' if beats else '·')} |")
W()
# family-only view: the relative-strength variables alone (REL + PAIR + RANK, no BTC extension, no leads)
E_rel = SC[keep & relm]
W(f"Core relative-strength sub-family only (REL 15m/1h/4h/24h + rank, 1D and 2D; {len(E_rel)} eligible cells): min T {E_rel['T'].min():+.2f} "
  f"({E_rel.loc[E_rel['T'].idxmin(), 'label']}), max T {E_rel['T'].max():+.2f} ({E_rel.loc[E_rel['T'].idxmax(), 'label']}) — vs this sub-family's OWN null "
  f"thresholds {np.percentile(RNMIN, 5):+.2f} / {np.percentile(RNMAX, 95):+.2f} (same within-day shuffle, max/min over these cells only).")
W()
fam = lambda v: "REL" if v.startswith(("rel", "L_")) else ("PAIR" if v.startswith("p") else ("RANK" if v == "rank" else "BTC"))
if len(surv_lo) + len(surv_hi):
    S2 = pd.concat([surv_lo, surv_hi])
    W("Survivor family composition: " + ", ".join(f"{k} {v}" for k, v in pd.Series(
        [fam(a) + ("×" + fam(b) if isinstance(b, str) else "") for a, b in zip(S2.v1, S2.v2)]).value_counts().items()))
    W()

# ═════════════════════ §3 candidates: block / multiplier ═════════════════════
W("## 3 · Candidates as BLOCK (and the strong end as MULTIPLIER) — yr4")
W()
qb = lambda v, q, k=5: binv(Y, v, edges(v, k)) == q
qbm = lambda v, q, k=5: binv(M, v, edges(v, k)) == q
CANDS = {
    "C1 REL24h bottom quintile": (qb("rel24h", 0), qbm("rel24h", 0), "block"),
    "C2 REL24h bottom decile": (qb("rel24h", 0, 10), qbm("rel24h", 0, 10), "block"),
    "C3 laggard × BTC-extended (REL24h Q1 ∧ BTC-EMA50 Q5)": (qb("rel24h", 0) & qb("btc_d50", 4), qbm("rel24h", 0) & qbm("btc_d50", 4), "block"),
    "C4 rank 36–50": (((Y["rank"] >= 36) & (Y["rank"] <= 50)).values, ((M["rank"] >= 36) & (M["rank"] <= 50)).values, "block"),
    "C5 REL24h top quintile (1.5×)": (qb("rel24h", 4), qbm("rel24h", 4), "mult"),
    "C6 REL 24-day bottom quintile (lead)": (qb("L_rel24d", 0), qbm("L_rel24d", 0), "block"),
    "ref Zone D (O3, registered)": (Y.zoneD.values.astype(bool), M.zoneD.values.astype(bool), "block"),
    "P7 REL4h top quintile (1.5×, POST-HOC from §1)": (qb("rel4h", 4), qbm("rel4h", 4), "mult"),
}
for i, r in enumerate(pd.concat([surv_lo.nsmallest(3, "T"), surv_hi.nlargest(2, "T")]).itertuples()):
    my = (QB[r.v1] == r.q1) & ((QB[r.v2] == r.q2) if isinstance(r.v2, str) else True)
    mm = (binv(M, r.v1, edges(r.v1, 5)) == r.q1) & ((binv(M, r.v2, edges(r.v2, 5)) == r.q2) if isinstance(r.v2, str) else True)
    CANDS[f"S{i + 1} {r.label}"] = (np.asarray(my, bool), np.asarray(mm, bool), "block" if r.T < 0 else "mult")
CB = Y.CB.values.astype(bool)


def cand_row(m, sel, kind):
    mm = m & sel; g = Y[mm]
    if not len(g):
        return None
    lo, hi, up = avg_ci(mm); dd, dl, dh = diff_ci(sel & ~mm, sel) if kind == "block" else (np.nan,) * 3
    tday, kd = share(g, "day"); tpair, kp = share(g, "pair")
    ps = [(Y.usd[mm & (Y.seed == s).values].sum()) for s in SEEDS]; pp = [PCT[mm & (Y.seed == s).values].sum() for s in SEEDS]
    bar = (g.pct > 0).mean() < BE and up < 0 and g.day.nunique() >= 8 and len(g) / NS >= MINN and tday < 0.5 and tpair < 0.5
    return dict(n=len(g) / NS, days=g.day.nunique(), wr=(g.pct > 0).mean(), avg=g.pct.mean(), lo=lo, hi=hi, up=up, d=dd, dl=dl, dh=dh,
                tday=tday, kd=kd, tpair=tpair, kp=kp, usd=ps, pp=pp, bar=bar)


W("Block Δ = avg(kept) − avg(all) per trade [day CI]; Σ %/seed and $/seed = what blocking REMOVES (negative = blocking gains, positive = blocking "
  "throws away profit; $ at today's yr4 sizing). For the multiplier: +0.5 × the cell's Σ (1.5× step). Bar = yr4 equivalent of the locked "
  "expectancy bar (WR < BE ∧ one-sided 95 % upper bound < 0 ∧ ≥ 8 days ∧ ≥ 15/seed ∧ top day/pair < 50 %) — informative only for Δ, see §0.")
W()
W("| candidate | pool | period | N/seed | days | WR | avg % [CI] | Δ kept−all [CI] | Σ % cell / seed (s1/s2/s3) | $ cell / seed | top day / top pair | bar |")
W("|---|---|---|---|---|---|---|---|---|---|---|---|")
CROWS = {}
for nm, (my, mm, kind) in CANDS.items():
    for pool, base in (("all", np.ones(len(Y), bool)), ("ex-CHOP∧BURST", ~CB)):
        for per, hs in (("YEAR", np.ones(len(Y), bool)), ("H1", H1), ("H2", ~H1)):
            r = cand_row(my, base & hs, kind)
            if r is None:
                continue
            CROWS[(nm, pool, per)] = r
            dtxt = "" if kind == "mult" else f"{r['d']:+.4f} [{r['dl']:+.4f}, {r['dh']:+.4f}]"
            W(f"| {nm} | {pool} | {per} | {r['n']:.1f} | {r['days']} | {r['wr'] * 100:.0f}% | {r['avg']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | "
              f"{dtxt} | "
              f"{sum(r['pp']) / NS:+.1f} ({' / '.join(f'{x:+.0f}' for x in r['pp'])}) | ${sum(r['usd']) / NS:+,.0f} | "
              f"{r['tday'] * 100:.0f}% ({r['kd']}) / {r['tpair'] * 100:.0f}% ({r['kp']}) | {'PASS' if r['bar'] else '—'} |")
W()

# leave-one-month-out + per-seed
W("### 3b · Leave-one-month-out block Δ (kept − all, per trade, all pool) and multiplier Σ%/seed")
W()
mons = sorted(Y.mon.unique()); kk = list(CANDS)
W("| month left out | " + " | ".join(k.split(" (")[0] if len(k) < 30 else k[:28] for k in kk) + " |"); W("|---" * (len(kk) + 1) + "|")
for mo in ["(none)"] + mons:
    sel = (Y.mon != mo).values; cells = []
    for k in kk:
        my, _, kind = CANDS[k]
        cells.append(f"{PCT[sel & ~my].mean() - PCT[sel].mean():+.4f}" if kind == "block" else f"{0.5 * PCT[sel & my].sum() / NS:+.1f}")
    W(f"| {mo} | " + " | ".join(cells) + " |")
W()
W("Per month (cell N/seed · avg %) for C1 / C3 / C5:")
W()
W("| month | all ML | C1 | C3 | C5 |"); W("|---|---|---|---|---|")
for mo in mons:
    s = (Y.mon == mo).values
    W(f"| {mo} | {fmt(s)} | " + " | ".join(fmt(CANDS[k][0], s) for k in (kk[0], kk[2], kk[4])) + " |")
W()

# ═════════════════════ §4 master per batch ═════════════════════
W("## 4 · Master per batch — BEFORE / AFTER (live $ as traded = ledger net; avg % = fee-net 1×)")
W()
ERAS = [e for e in ["BASE"] + [f"B{i}" for i in range(1, 17)] if e in set(M.era)]


def mfmt(g):
    return f"{len(g)} · {(g.pct > 0).mean() * 100 if len(g) else 0:.0f}% · ${g.net.sum():+,.0f} · {g.pct.mean() if len(g) else 0:+.3f}"


def master_table(nm, mm, kind):
    mm = pd.Series(mm, index=M.index).fillna(False).astype(bool)
    W(f"### {nm} — master cohort: {mm.sum()} fills · {(M.pct[mm] > 0).mean() * 100 if mm.any() else 0:.0f}% WR · avg {M.pct[mm].mean() if mm.any() else 0:+.3f} · "
      f"${M.net[mm].sum():+,.0f} live · {M.day[mm].nunique()} days")
    W()
    W(f"| batch | before N · WR · $ · avg | after ({'block' if kind == 'block' else '1.5× on cell'}) | Δ$ | before, CHOP∧BURST blocked | after, CHOP∧BURST blocked | Δ$ |")
    W("|---|---|---|---|---|---|---|")
    for e in ERAS + ["TOTAL"]:
        sel = (M.era == e) if e != "TOTAL" else pd.Series(True, index=M.index)
        g = M[sel]; cbm = M.CB & sel
        row = []
        for base in (sel, sel & ~M.CB):
            gb = M[base]
            if kind == "block":
                ga = M[base & ~mm]; dlt = ga.net.sum() - gb.net.sum(); after = mfmt(ga)
            else:
                dlt = 0.5 * M.net[base & mm].sum(); ga = gb
                after = f"{len(gb)} · {(gb.pct > 0).mean() * 100 if len(gb) else 0:.0f}% · ${gb.net.sum() + dlt:+,.0f} · {gb.pct.mean() if len(gb) else 0:+.3f}"
            row += [mfmt(gb), after, f"{dlt:+,.0f}"]
        W(f"| {e} | " + " | ".join(row) + " |")
    W()
    if mm.any():
        W("Cohort trades (master):")
        W()
        W("| opened (UTC) | pair | batch | REL 15m | REL 1h | REL 4h | REL 24h | BTC-EMA50 | rank | pct | $ live | CHOP∧BURST |"); W("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in M[mm].sort_values("o").itertuples():
            W(f"| {r.o} | {r.pair} | {r.era} | {r.rel15:+.2f} | {r.rel1h:+.2f} | {r.rel4h:+.2f} | {r.rel24h:+.2f} | {r.btc_d50:+.2f} | {r.rank:.0f} | {r.pct:+.3f} | {r.net:+,.0f} | {'Y' if r.CB else '·'} |")
        W()


for nm, (my, mm, kind) in CANDS.items():
    master_table(nm, mm, kind)

# ═════════════════════ outputs ═════════════════════
keepc = ["seed", "pair", "opened_at", "pct", "pnl", "day", "CHOP", "BURST", "CB"] + VARS + ["zoneD"]
Y[keepc].to_csv(f"{R}/ML_PAIR_STRENGTH_2026-10-04_yr4_features.csv", index=False)
M[["pair", "opened_at", "era", "pct", "net", "p1x", "CHOP", "BURST", "CB"] + VARS + ["zoneD"]].to_csv(f"{R}/ML_PAIR_STRENGTH_2026-10-04_master_features.csv", index=False)
pd.DataFrame(DR).to_csv(f"{R}/ML_PAIR_STRENGTH_2026-10-04_dose.csv", index=False)
VERDICT = """
## 5 · Verdict (written after reading §1–§4; UNREVIEWED backtest — no arm recommendation without the caveman + deep reviews)

**Headline — dose-response.** None of the 16 variables is monotone in both halves. The core relative-strength variables (REL 1h / 4h / 24h, pair
return alone, 24 h-volume rank) are flat: every REL 24h quintile sits within ±0.04 %/trade of the rest, and the bottom quintile flips sign
between halves (H1 Δ −0.07, H2 +0.13). The 09-29 'laggard' lead was the 24-DAY variable (`REL_1d_ret24` = 24 daily bars, not 24 h): its bottom
quintile reproduces the low WR (53 % vs 61 %) and a small block Δ (+0.018 %/trade [−0.003, +0.038], + in both halves, + in every LOMO fold),
but is not monotone (ρ +0.2 / +0.2). Laggard × BTC-extended (REL 24h Q1 ∧ BTC 5m EMA50 Q5) is NOT a loser on yr4 (27/seed, 67 % WR, +0.02)
and 3/3 winners on the master. Rank bands 11–20 / 21–35 / 36–50 are within 0.03 %/trade of each other (yr4) and all profitable on the master.

**Screen vs null.** 2,849 eligible 1D+2D quintile cells on 1,038 unique trades; within-day shuffled null family-wise thresholds T −3.99 / +4.34.
Two block-side cells beat it with both halves agreeing — `REL 15m Q2` (pair's last-15-min move within −0.11…+0.04 pp of BTC's; 120/seed,
48 % WR, −0.21 %, block Δ +0.034 [+0.009, +0.058], every LOMO fold +0.027…+0.038) and its sub-cell `REL 15m Q2 ∧ pair 4h Q2`. Nothing on the
multiplier side beats the null. The pointwise excess (218 cells at |T| ≥ 1.96 vs null median 157, p 0.02) is real but diffuse.

**Why the survivor is still not a ship.** ① NON-MONOTONE: it is a mid-band hole (REL 15m Q1 −0.09 ≈ sleeve, Q2 −0.21, Q3 −0.02) — the locked
rule reads that as a confound, not a separator. ② MASTER REFUTES (the master can only refute): the same band on the live master is 24 fills ·
83 % WR · +0.296 %/trade · +$1,268 live (sleeve +0.227); blocking it costs −$1,268 live (−$1,626 with CHOP∧BURST already blocked) and
removes winners in BASE, B1, B2, B3, B12, B13. ③ Same tape as yr3 (the lead's source) → not out-of-sample.

**Strong end as a multiplier.** REL 24h top quintile is a LOSER on yr4 (68 % WR but −0.110 %; 1.5× = −$4,434/seed/yr) — fails the W gate.
REL 4h top quintile (post-hoc, picked from §1, so in-sample) is the only positive 1D tail (Δ +0.14 in both halves) but its yr4 avg is +0.034
(< the +0.10 W gate), H1 1.5× loses (−$1,039/seed), H2 wins (+$2,086/seed). Master: 28 · 82 % · +0.329 · +$1,935 live → 1.5× ≈ +$967 live
(+$1,305 with CHOP∧BURST blocked). Interesting, but N 28 < 30 on the master and it is post-hoc.

**CHOP∧BURST:** removing it first changes no verdict (every candidate's sign and pass/fail is the same with and without it; table §3, §4).

**Recommendation:**
- **DROP** the pair-strength BLOCK family as a filter (REL 24h laggard, bottom decile, laggard × BTC-extended, rank 36–50, 24-day laggard):
  refuted on yr4 (24 h variants) and/or by the master (24-day: 16 fills · 81 % · +$963 live). Zone D (O3) stays the family's registered
  representative with its existing fresh-only bar — unchanged.
- **DROP** `REL 15m Q2` as a block (beats the null, but non-monotone and refuted by 24 master fills).
- **PRE-REGISTER OBSERVE-ONLY (no code, no sizing change):** W-candidate *REL 4h top quintile* — frozen definition: pair 4 h return − BTC 4 h
  return > **+1.53 pp** on closed 5m bars at entry (this script's rebuild). Fresh fills only: momentum LONG, full-size, non-probe, opened
  ≥ **2026-10-05 00:00 UTC**. Promotion gate = locked Pattern-W bar on fresh fills (N ≥ 30 ∧ WR ≥ 70 % ∧ avg ≥ +0.10 % ∧ Σ$ > 0, ≥ 8 days,
  no day/pair ≥ 50 %), first step 1.5×. Kill: fresh N ≥ 15 with avg < 0 → drop. Re-run `scripts/ml_pair_strength.py` (extend the master
  input with fresh fills) at each batch review.
- **No override** recommended: nothing here clears both the null and the master.

**Fidelity / honesty:** yr4 reproduces 64 % of live ML fills (winners 66 % / losers 62 %), but its sleeve is negative (−0.079) vs master
+0.227 → only Δ vs rest is read, never levels. yr3 and yr4 share the Jan–Oct tape the 09-29 lead came from → this is a re-test on the same
tape, NOT out-of-sample; the only OOS evidence will be fresh live fills after 2026-10-05. Replay seeds share ~55 % of fills (outcome corr
0.85) → the screen ran on unique trades; dose-response tables pool seeds with day-block CIs. The within-day null cannot test pure
day-level (BTC-regime) effects — those are absorbed by the shuffle (conservative for BTC-extension cells). REL 24-day and REL vs 4h EMA200
are unscored before ~Jan 24 / Feb 2 (excluded, not 'rest'). Rebuilt features use the last CLOSED 5m bar (median 130 s before entry);
the live stamp uses the forming price (BTC-EMA13 r 0.84, best shift 0 — validate_against_master PS1 PASS).
"""
W(VERDICT)
open(f"{R}/ML_PAIR_STRENGTH_2026-10-04.md", "w").write("\n".join(OUT) + "\n")
print("\n".join(OUT))
