"""B25-prep ML W/L — writes reports/PREP_B25_ML_WL.md from the analyse outputs (run study_ml_wl_b25prep_analyse.py first).
Adds: dashboard map, coverage table, headline before → after per era, DAY-unit reads, bars, frozen B25 pre-registration and the
B25 tally of the stamp-computable existing ML observe lines. Read-only on inputs; no network (kline cache + this study's top-up only).

    venv/bin/python -W ignore scripts/study_ml_wl_b25prep_report.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_ml_wl_b25prep_data import DASHBOARD_MAP, ROOT, load_all  # noqa: E402
import study_ml_wl_1015_common as C  # noqa: E402

C.TOPUP = os.path.join(ROOT, "reports/PREP_B25_ML_WL_fetch")
OUT = os.path.join(ROOT, "reports/PREP_B25_ML_WL")
ENG = open(OUT + "_engine.txt").read()
COV = pd.read_csv(OUT + "_coverage.csv")
CR = pd.read_csv(OUT + "_candidates.csv")
R1 = pd.read_csv(OUT + "_screen1d.csv")
DV = pd.read_csv(OUT + "_dashboard_vars.csv")
PE = pd.read_csv(OUT + "_post_entry.csv")
Q = pd.read_csv(OUT + "_quintiles.csv"); Q["lo"] = pd.to_numeric(Q.lo, errors="coerce"); Q["hi"] = pd.to_numeric(Q.hi, errors="coerce")
BR = pd.read_csv(OUT + "_replay_baserate.csv").iloc[0]
D = load_all()
M = D[D.src == "master"]
K = M[~M.is_probe & M.stack_keep & M.y_keep.notna()].copy(); K["y"] = K.y_keep.astype(float)
T = M[~M.is_probe].copy(); T["y"] = T.y_trad.astype(float)
B = D[(D.era == "B25") & ~D.is_probe].copy(); B["y"] = B.y_trad.astype(float)
ERA = ["G0 BASE", "G1 B1", "G2 B2", "G3 B3", "G4 B4-B11", "G5 B12-B16", "G6 B17-B24"]


def be(y):
    y = np.asarray(y, float); w, l = y[y > 0], y[y <= 0]
    return abs(l.mean()) / (w.mean() + abs(l.mean()))


BE = be(K.y)


def dboot(y, day, n=4000, seed=3):
    y = np.asarray(y, float); u, inv = np.unique(np.asarray(day), return_inverse=True)
    s = np.bincount(inv, weights=y); c = np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs < 0).mean()


def st(d):
    if not len(d):
        return "0"
    return f"{len(d)}·{(d.y > 0).mean()*100:.0f}%·{d.y.mean():+.3f}"


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)


# ------------------------------------------------------------------ headline zones (frozen thresholds = the KEEP-pool cuts printed)
HEAD = {
    "H1 STRONG_BULL regime (winner side)": lambda d: (d.entry_btc_regime.astype(str) == "STRONG_BULL").where(d.entry_btc_regime.notna()),
    "H1b BTC ADX > 27.99 (alias of H1)": lambda d: (num(d, "entry_btc_adx") > 27.99).where(num(d, "entry_btc_adx").notna()),
    "H2 pair ADXΔ ≤ 0.309 ∧ regime ≠ STRONG_BULL (loser)": lambda d: ((num(d, "entry_adx_delta") <= 0.3094)
                                                                     & (d.entry_btc_regime.astype(str) != "STRONG_BULL")).where(
        num(d, "entry_adx_delta").notna() & d.entry_btc_regime.notna()),
    "H3 EMA5−8 gap ≤ 0.082 (loser; dashboard Gap5-8 — the signed stamp holds the same values)": lambda d: (num(d, "entry_ema_gap_5_8") <= 0.0819).where(num(d, "entry_ema_gap_5_8").notna()),
    "H4 BTC 15-min return ≤ 0 (loser)": lambda d: (num(d, "btc_r15") <= 0).where(num(d, "btc_r15").notna()),
    "H5 top-10 corr15 > 0.365 (loser)": lambda d: (num(d, "imp_corr15") > 0.3654).where(num(d, "imp_corr15").notna()),
    "H6 breadth ret15 ≤ med ∧ pair vol15 ≤ med (winner)": None,   # filled below with the KEEP medians
    "H7 imp spike_30 ≤ med ∧ corr15 ≤ med (winner)": None,
}
med = {c: num(K, c).median() for c in ["br_ret15", "pair_vol15_x24h", "imp_spike_30", "imp_corr15"]}
HEAD["H6 breadth ret15 ≤ med ∧ pair vol15 ≤ med (winner)"] = lambda d: ((num(d, "br_ret15") <= med["br_ret15"])
                                                                      & (num(d, "pair_vol15_x24h") <= med["pair_vol15_x24h"])).where(
    num(d, "br_ret15").notna() & num(d, "pair_vol15_x24h").notna())
HEAD["H7 imp spike_30 ≤ med ∧ corr15 ≤ med (winner)"] = lambda d: ((num(d, "imp_spike_30") <= med["imp_spike_30"])
                                                                 & (num(d, "imp_corr15") <= med["imp_corr15"])).where(
    num(d, "imp_spike_30").notna() & num(d, "imp_corr15").notna())


def head_block(name, f):
    z = f(K); v = z.notna(); z = z.fillna(False).astype(bool)
    zk, rk = K[z], K[v & ~z]
    lo, hi, pneg = dboot(zk.y, zk.day)
    dm = zk.groupby("day").y.mean()
    dm_rest = rk.groupby("day").y.mean()
    zt = f(T); vt = zt.notna(); zt = zt.fillna(False).astype(bool)
    zb = f(B); vb = zb.notna(); zb = zb.fillna(False).astype(bool)
    after = K[~z]
    nw = ~K.washed
    s = []
    s.append(f"**{name}** — KEEP scored {int(v.sum())}/{len(K)} · zone {st(zk)} (95% day-CI [{lo:+.2f}, {hi:+.2f}], P(avg<0) {pneg:.2f}) "
             f"vs rest {st(rk)} · Δ {zk.y.mean() - rk.y.mean():+.3f}")
    s.append(f"- DAY units: zone on {zk.day.nunique()} days · mean of day-means {dm.mean():+.3f} (rest {dm_rest.mean():+.3f}) · "
             f"days with zone mean < 0: {(dm < 0).mean():.0%} · 2-min scan windows {zk.win_id.nunique()}")
    neg = zk[zk.y <= 0]
    if len(neg):
        dl = zk.groupby("day").y.sum(); pl = zk.groupby("pair").y.sum()
        ld, lp = dl[dl < 0], pl[pl < 0]
        s.append(f"- concentration: worst day {ld.idxmin().date() if len(ld) else '—'} {ld.min() / ld.sum() if len(ld) else 0:.0%} of the zone's net loss · "
                 f"worst pair {lp.idxmin() if len(lp) else '—'} {lp.min() / lp.sum() if len(lp) else 0:.0%} · gain side: best day "
                 f"{(dl.max() / dl[dl > 0].sum()) if (dl > 0).any() else 0:.0%} / best pair {(pl.max() / pl[pl > 0].sum()) if (pl > 0).any() else 0:.0%} of the net gain")
    s.append(f"- without the washed-out window: zone {st(zk[nw[z.values]] if False else K[z & nw])} vs rest {st(K[v & ~z & nw])} · washed-out only: zone {st(K[z & ~nw])}")
    s.append(f"- TRAD (as traded, all full-size): zone {st(T[zt])} vs rest {st(T[vt & ~zt])} · PROBES (separate): "
             f"{st(M[M.is_probe & f(M).fillna(False).astype(bool)].assign(y=lambda x: x.y_trad))}")
    s.append(f"- sleeve before → after removing the zone (KEEP): {st(K)} → {st(after)} · Δ avg {after.y.mean() - K.y.mean():+.3f} "
             f"→ with the 30–50 % in-sample haircut {0.5 * (after.y.mean() - K.y.mean()):+.3f} … {0.7 * (after.y.mean() - K.y.mean()):+.3f}")
    rows = []
    for g in ERA:
        m = K.era_group == g
        rows.append(f"| {g} | {st(K[m])} | {st(K[m & ~z])} | {st(K[m & z])} | {st(K[m & v & ~z])} |")
    s.append("\n| era group | before (all) | after (zone removed) | zone | rest (scored) |\n|---|---|---|---|---|\n" + "\n".join(rows))
    s.append(f"\n- B25 provisional: scored {int(vb.sum())}/{len(B)} · zone {st(B[zb])} ({', '.join(f'{p} {q:+.2f}' for p, q in zip(B[zb].pair, B[zb].y)) or '—'}) vs rest {st(B[vb & ~zb])}")
    return "\n".join(s)


# ------------------------------------------------------------------ B25 tally of existing frozen ML lines (stamp / kline computable)
def btc24h(ts):
    t = int(pd.Timestamp(ts).value // 10**6)
    K_ = C.Klines()
    a = K_.px("BTCUSDT", t, C.MIN5); b = K_.px("BTCUSDT", t - 86_400_000, C.MIN5)
    return (a / b - 1) * 100 if np.isfinite(a) and np.isfinite(b) else np.nan


B["btc24h"] = [btc24h(t) for t in B.ts]
EX = [
    ("DL 125 BTC-drop + pair-rollover", (num(B, "entry_btc_rsi_prev") - num(B, "entry_btc_rsi") >= 5) & (num(B, "entry_rsi") < num(B, "entry_rsi_prev"))),
    ("DL 160 burst ≤ 2 min after another bot fill", num(B, "d_burst_prior_min") <= 2),
    ("DL 161 BTC-chop eff72 ≤ 0.007 (live stamp)", num(B, "entry_btc_eff72") <= 0.007),
    ("DL 258 PAIR_1H_DOWNTREND (pair 1h EMA20 ≤ EMA200)", num(B, "entry_pair_1h_ema20_200_gap_pct") <= 0),
    ("DL 273 ① BTC24H_DOWN (BTC 5m close vs 288 bars earlier ≤ 0)", B.btc24h <= 0),
    ("DL 273 ② ADX Δ ≤ 0 (live forming-bar stamp — the line uses the closed-bar rebuild)", num(B, "entry_adx_delta") <= 0),
    ("DL 126 LOW-ADX RSI-momentum (should be 0 — live gate)", (num(B, "entry_adx") < 21) & (num(B, "entry_rsi") < num(B, "entry_rsi_prev"))),
    ("DL 110 mega-cap rank ≤ 10 (should be 0 — live gate)", num(B, "entry_pair_rank") <= 10),
]

# ------------------------------------------------------------------ compose
L = []
P = L.append
P("# PREP B25 — Momentum-LONG winners vs losers (2026-10-26)\n")
P("Prepared for the B25 batch review (sleeve order: … → LITE / FRENZY_LONG → **momentum long** → …). Study only — nothing shipped, "
  "no engine / config / template edits, no git, no bot API. Binance: ≈140 public 1m + ≈370 public 5m kline calls over three runs (peak used weight 281/min of 1,200, "
  "cooldown file honoured) for the 32 fills the earlier feature files did not cover.\n")
P("**Validation:** `scripts/validate_against_master.py` → ALL CHECKS PASS (run before the study; master STACK 2026-10-26a). "
  "Rebuilt impulse features reuse `study_ml_wl_1015_features.feats` unchanged. ⚠ Caught during the build: the Oct-15 impulse file's "
  "`Y_btc` / `Y_alt` are FORWARD 15-min returns (the study's targets) — they were excluded from the entry screen and appear only as "
  "post-entry description; including them had produced fake 'top separators' (Δ −0.5, p 0.001).\n")
P("## 1 · TL;DR\n")
P(f"- **Sleeve now (master, today's stack, full-size):** {st(K)} over {K.day.nunique()} days; ML breakeven WR = **{BE*100:.1f}%** "
  f"(avg win {K.y[K.y>0].mean():+.3f} / avg loss {K.y[K.y<=0].mean():+.3f}). Without the washed-out window: {st(K[~K.washed])}. "
  f"As traded: {st(T)}. Recent eras are the problem: B12–B16 {st(K[K.era_group=='G5 B12-B16'])}, B17–B24 {st(K[K.era_group=='G6 B17-B24'])}. "
  f"B25 provisional: {st(B)} (3 days).")
P("- **Every variable was tested** (225 candidates: all 31 dashboard columns, every entry_* stamp, the Oct-4 pair-strength, Oct-15 impulse "
  "and 10-15 market-impulse features, cooldown/burst, hour/day): 155 screenable, 70 not (listed in §11).")
P("- **No separator beats chance.** 1D: 21 tests at p<0.05 vs ≈33 expected by chance; one family-wise hit (a recent-era-only stamp, refuted "
  "below). Exhaustive 2D (24,280 quadrant tests): 172 at p<0.01 vs a shuffled-label null of median 104 / 95th pct 426 — inside the null; "
  f"0 family-wise. Replay agreement of the 2D candidates ({BR.cand_all3:.0%} same sign in all three regimes) equals the random-test base "
  f"rate ({BR.base_all3:.0%}).")
P("- **Survivors vs the bar:** no loser zone passes the full expectancy bar with out-of-sample support. Closest: **H2 pair ADXΔ ≤ 0.31 ∧ "
  "regime ≠ STRONG_BULL** (58·59%·−0.110; same sign in 6/6 held-out era groups, 2nd half, as-traded and all three replay regimes) — "
  "**fails only on confidence** (P(avg<0) 0.84 < 0.95) → pre-registered observe line (§12). It is the third mining of ADX direction "
  "(DL 273 ②) — fresh fills only.")
P("- **Winner signature:** **STRONG_BULL regime / BTC ADX > 28** (53·77%·+0.279, Pattern-W bar met in-sample, replay positive in all "
  "three regimes) — but the newest era group is OPPOSITE (B17–B24 zone 15·40%·−0.20) and B25 is flat (5 fills 3W/2L ±0). Not a multiplier: "
  "observe line with a frozen bar; cell-registry history (NONEXP·SBULL / CALM3D at 1× since 10-06) must be read before any sizing talk.")
P("- **Refuted this round:** EMA5−8 gap ≤ 0.082 (the only family-wise hit) is a recent-era artefact — on the 115 older fills the same cut "
  "gives +0.217 vs +0.217, and B25 already contradicts (7 fills · 71% · +0.07). BTC 15-min return ≤ 0 is regime-dependent (replay 2025 and "
  "2026-H1 OPPOSITE sign).")
P("- **Read:** where the loser zones separate at all, the effect sits in the losing recent eras (B12–B24); in BASE–B3 the same zones are "
  "flat or positive. That is what an unmeasured regime variable (sleeve-kill checklist ③) would produce just as well as a new entry "
  "edge — the forward fills decide. B25 so far contradicts every loser line (P2 zone 2·100 %, P3 7·71 %, P4 4·75 %, DL 273 BTC24H_DOWN "
  "8·88 %·+0.21). **No ship and no sizing change recommended from this study.**\n")

P("## 2 · Blocks and sleeve baseline\n")
P("```\n" + ENG.split("## S1")[0].split("## S0")[1].strip() + "\n```\n")
P(f"PROBE fills (B1/B2 probe era, separate line, never pooled): {st(M[M.is_probe].assign(y=lambda x: x.y_trad))}.\n")

P("## 3 · Dashboard columns → master columns\n")
P("Source: `main.py` `_ec_row` (the builder behind *Entry Conditions by Outcome* and *Entry Conditions by Strategy — Winners vs Losers*, "
  "`templates/index.html` L1510–1620; text exports ~L10110 / ~L11969).\n")
P("| dashboard column | column tested |\n|---|---|")
for a, b in DASHBOARD_MAP:
    P(f"| {a} | `{b}` |")
P("")

P("## 4 · Coverage (variable × era group: scored / fills) — unscored is never 'rest'\n")
P("Full-size momentum longs. Screened = ≥30 scored KEEP fills and ≥2 values; ≥80 % KEEP coverage → exhaustive 2D; lower coverage → 1D on "
  "scored rows only (flagged). Full CSV: `reports/PREP_B25_ML_WL_coverage.csv`.\n")
gcols = ["G0 BASE", "G1 B1", "G2 B2", "G3 B3", "G4 B4-B11", "G5 B12-B16", "G6 B17-B24", "B25 (prov.)"]
P("| variable | " + " | ".join(g.split()[0] if g != "B25 (prov.)" else "B25" for g in gcols) + " | KEEP cov | status |")
P("|---|" + "---|" * (len(gcols) + 2))
for r in COV.itertuples(index=False):
    rr = r._asdict() if hasattr(r, "_asdict") else dict(zip(COV.columns, r))
    rr = dict(zip(COV.columns, r))
    if rr["KEEP scored"] == 0 and sum(rr[f"{g} scored"] for g in gcols) == 0:
        continue
    status = "2D+1D" if rr["KEEP cov"] >= 0.8 and rr["KEEP scored"] >= 30 and rr["KEEP distinct"] >= 2 else \
        ("1D scored-only" if rr["KEEP scored"] >= 30 and rr["KEEP distinct"] >= 2 else "NOT testable")
    P(f"| `{rr['variable']}` | " + " | ".join(f"{rr[f'{g} scored']}/{rr[f'{g} N']}" for g in gcols)
      + f" | {rr['KEEP cov']:.0%} | {status} |")
zero = COV[(COV["KEEP scored"] == 0)].variable.tolist()
P(f"\nNever stamped on a momentum long (0 fills in every era; listed for completeness): {', '.join('`'+z+'`' for z in zero)}.\n")

P("## 5 · Method and null calibration\n")
P("- Outcome: % at 1× (leverage-invariant). Discovery block KEEP = master full-size fills kept by today's stack, y = `stack_pct` "
  "(in-sample to the stack's own filters — DL 204 reading rule); robustness block TRAD = as traded; B25 descriptive; replay "
  "(`ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv`, MOM-long, 3 seeds, 2025 BULL / 2026-H1 BEAR / 2026-H2 BULL) REFUTE-ONLY.")
P("- 1D zones per variable: sign (when both signs ≥5 %), binary ==1/==0, > / ≤ median, ≤ q33, > q67; quintile tables for every variable "
  "(`_quintiles.csv`). 2D: exhaustive all pairs of the full-coverage variables, median × median, all four quadrants (zone ≥10 fills).")
P("- Null: circular shifts of the time-sorted outcome (keeps day clustering): 1,000 for 1D, 200 for 2D; per-test p and family-wise p "
  "(max-|Δ| over the whole family). Second null for 2D: the count of p<0.01 tests under 20 shifted outcomes.")
P("- OOS: leave-one-era-group-out with thresholds re-fit on the training groups; chronological halves (fit on the first, test the second); "
  "TRAD same sign; replay sign per regime with day-bootstrap P.")
P("- Bars: expectancy bar (filters) on KEEP — WR < breakeven, P(avg<0) ≥ 0.95 by DAY-clustered bootstrap, ≥ 8 days, N ≥ 15, no day / "
  "pair ≥ 50 % of the zone's net loss; Pattern-W bar (multipliers) N ≥ 30, WR ≥ 70 %, avg ≥ +0.10 %, $ > 0. Market-wide variables "
  "read in DAY units (and 2-min scan windows reported).\n")
P("```\n" + "\n".join(l for l in ENG.split("\n") if l.startswith("## S1") or l.startswith("## S2") or l.startswith("## S3")
                      or l.strip().startswith("2D tests") or l.strip().startswith("shuffled") or l.strip().startswith("replay loaded")) + "\n```\n")

P("## 6 · Dashboard variables, 1D (KEEP)\n")
P("Best zone per variable by p_null (sign / median / tercile); p_fw = family-wise over all 668 1D tests. None reaches p_fw < 0.05.\n")
P("| variable | scored | median W / L | best zone | zone N·WR·avg | Δ vs rest | p | p_fw | macro |\n|---|---|---|---|---|---|---|---|---|")
for r in DV.itertuples():
    P(f"| `{r.variable}` | {r.scored} | {r.med_W:+.4g} / {r.med_L:+.4g} | {r.best_zone} | {int(r.best_N)}·{r.best_WR*100:.0f}%·{r.best_avg:+.3f} "
      f"| {r.best_delta:+.3f} | {r.best_p:.3f} | {r.best_pfw:.2f} | {'DAY' if r.macro else ''} |")
P("\nQuintile shape (KEEP N·WR·avg per quintile) for the dashboard variables:\n")
P("| variable | Q1 | Q2 | Q3 | Q4 | Q5 |\n|---|---|---|---|---|---|")
for c in DV.variable:
    q = Q[Q.variable == c]
    if len(q) == 5:
        P(f"| `{c}` | " + " | ".join(f"≤{r.hi:.3g}: {r.N}·{r.WR*100:.0f}%·{r.avg:+.2f}" if i < 4 else f">{r.lo:.3g}: {r.N}·{r.WR*100:.0f}%·{r.avg:+.2f}"
                                     for i, r in enumerate(q.itertuples())) + " |")
P("\nNo dashboard variable reaches family-wise significance; the BTC ADX / regime pair is the only one whose sign is confirmed in all three "
  "replay regimes (H1).\n")

P("## 7 · Top separators — before → after per era (KEEP, today's stack)\n")
P("Thresholds are the KEEP-pool cuts and are FROZEN as printed. Full candidate list with every test's OOS / replay / B25 column: "
  "`reports/PREP_B25_ML_WL_candidates.csv`; all 1D tests `_screen1d.csv`; all 2D tests `_screen2d.csv`.\n")
for name, f in HEAD.items():
    P(head_block(name, f))
    r = None
    P("")

P("## 8 · Survivors vs the bars\n")
P("Candidates = 1D p<0.05 ∧ |Δ|≥0.10, 2D p<0.01 ∧ |Δ|≥0.20 (top 150). OOS✓ = LOEO same sign in ≥75 % of ≥4 held-out era groups ∧ "
  "2nd-half same sign ∧ TRAD same sign. Replay columns are refute-only.\n")
s = CR.copy()
P(f"- 1D candidates {int((s.kind==1).sum())} (OOS✓ {int(((s.kind==1)&s.oos_ok).sum())}); 2D candidates {int((s.kind==2).sum())} "
  f"(OOS✓ {int(((s.kind==2)&s.oos_ok).sum())}). Expectancy-bar passes {int(s.FILTER_bar.sum())} (with OOS✓ {int((s.FILTER_bar&s.oos_ok).sum())}); "
  f"Pattern-W passes {int(s.W_bar.sum())} (with OOS✓ {int((s.W_bar&s.oos_ok).sum())}; with OOS✓ ∧ replay same sign in all 3 regimes "
  f"{int((s.W_bar&s.oos_ok&(s.replay_same==3)).sum())}).")
P("- Every family-wise p ≥ 0.045 (1D) / ≥ 0.5 (2D): **no candidate survives the multiple-testing correction.** The bar passes below are "
  "therefore in-sample patterns, not evidence.\n")
P("### 8a · Loser zones that meet the expectancy bar (in-sample)\n")
P("| zone | N | days | WR | avg | P(avg<0) | LOEO | 2nd half | TRAD Δ | replay (refute-only) | B25 | verdict |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in s[s.FILTER_bar].itertuples():
    verdict = "REFUTED (older eras Δ 0.00; B25 opposite)" if "gap_5_8" in r.zone else (
        "no OOS / replay null → not a candidate" if not r.oos_ok or r.replay_same < 3 else "observe")
    P(f"| {r.zone} | {r.N} | {r.days} | {r.WR*100:.0f}% | {r.avg:+.3f} | {r.P_avg_neg:.2f} | {r.oos_same}/{r.oos_tot} | {r.half_test:+.2f} | "
      f"{r.trad_delta:+.2f} | {r.replay} | {r.b25} | {verdict} |")
P("\n### 8b · Loser zones with OOS✓ and replay same sign in all three regimes (fail the bar only on confidence)\n")
P("| zone | N | days | WR | avg | P(avg<0) | fails | LOEO | 2nd half | replay |\n|---|---|---|---|---|---|---|---|---|---|")
for r in s[(s.side == "LOSER") & s.oos_ok & (s.replay_same == 3)].sort_values("p_null").itertuples():
    P(f"| {r.zone} | {r.N} | {r.days} | {r.WR*100:.0f}% | {r.avg:+.3f} | {r.P_avg_neg:.2f} | {r.filter_fails} | {r.oos_same}/{r.oos_tot} | "
      f"{r.half_test:+.2f} | {r.replay} |")
P("\nThe ADXΔ ≤ med ∧ (≠ STRONG_BULL / = HEALTHY_BULL) rows are one zone (two labels of the same fills); the corr15 rows need the "
  "Oct-15 impulse grid (not scored after 10-15).\n")

P("## 9 · Winner signatures (Pattern-W bar — multiplier candidates)\n")
w = s[s.W_bar & s.oos_ok & (s.replay_same == 3)].sort_values("p_null")
P("Pattern-W passes with OOS✓ and replay same sign in all three regimes (base rate of 'all three agree' for random 2D tests "
  f"= {BR.base_all3:.0%}, so ~1 in 6 of these is expected by luck):\n")
P("| zone | N | days | WR | avg | $ (stack) | LOEO | 2nd half | ex-washed | replay | B25 |\n|---|---|---|---|---|---|---|---|---|---|---|")
for r in w.itertuples():
    P(f"| {r.zone} | {r.N} | {r.days} | {r.WR*100:.0f}% | {r.avg:+.3f} | {r.usd:+.0f} | {r.oos_same}/{r.oos_tot} | {r.half_test:+.2f} | "
      f"{r.exwash_N}·{r.exwash_avg:+.2f} | {r.replay} | {r.b25} |")
P("\nRead: the recurring theme is a **quiet, idiosyncratic tape** (top-10 coins NOT moving together, no BTC spike, low pair 15-min volume "
  "burst, flat 15-min breadth) and **BTC in a strong trend (STRONG_BULL / ADX > 28)**. None clears the family-wise null and each is "
  "in-sample; the newest era group (B17–B24) is the weakest block for most of them. Multipliers stay off the table (operator DL 289: "
  "momentum longs at 1×; memory: filters, not sizing). Registered as observe lines in §11.\n")

P("## 10 · Post-entry (descriptive only — not entry separators)\n")
P("| variable | scored | median winners | median losers |\n|---|---|---|---|")
for r in PE.itertuples():
    P(f"| `{r.variable}` | {r.scored} | {r.med_W:+.3f} | {r.med_L:+.3f} |")
P("\n`post_Y_btc` / `post_Y_alt` = BTC / top-50 return over the 15 min AFTER entry: winners see BTC rise, losers see it fall — the same "
  "finding as the 10-15 study (the trade's fate is decided by the next 15 min of BTC, which no entry variable predicts). Duration and "
  "peak timing: losers are stopped within ~the first half hour; winners peak late.\n")

P("## 11 · What we could NOT test (blind spots)\n")
P("1. **Never stamped on momentum longs** (0 scored in every era): Bull-Run door stamps `entry_br_*`, Bear-Run `entry_bear_*`, FRENZY / "
  "FLATBASE / SURGE / WILLY stamps, fade-only `entry_btc_4h_ema50_200_gap_pct` and `entry_pair_1d_ndi`, order-book `entry_book_*` "
  "(new, 0 ML fills). `entry_rsi_prev2` (the RSI-momentum filter's own leg) is not stamped — RSIDir uses `rsi_prev`.")
P("2. **Low coverage, 1D scored-only, no 2D, no LOEO (all recent-era):** BTC 24h / 30-day / 72h stamps (`entry_btc_off24h_pct`, "
  "`off24lo`, `off30d_high`, `r72`, `eff72`, `above72`, heat flags; 66–72 KEEP fills since mid-Sep), signed gaps and EMA5−20 change "
  "(43), `entry_pair_1h_ema20_200_gap_pct`, `entry_btc_1d_ret_pct`, `entry_btc_ema50_100_gap_pct`, `entry_eth_5m_ret1_pct` (43), "
  "`entry_cmc_rank` / `entry_mcap_usd` (47), `entry_btc_rsi_closed` (37), `entry_vol_mcap_ratio` (28), `entry_pair_age_days` (121), "
  "`entry_chop_burst_prior_fill_s` (9), cooldown minutes since a stop (8). Any 'effect' here is confounded with the recent-era loss.")
P("3. **Prior-study features that stop before the newest fills:** Oct-15 impulse legs / score / corr15 / spike / volume shape (`imp_*`, "
  "fills ≤ 10-15; 0 of the B22–B25 fills) and Oct-4 pair-strength (`ps_*`, ≤ 10-04). Rebuilding them needs the impulse-grid fetch "
  "(`study_ml_impulse_fetch.py` + `_features.py`, the ML_MARKET_TURN scout step) — not run here (shared Binance budget).")
P("4. **Not available anywhere:** open interest, funding history beyond the entry stamp, long/short ratio, liquidations, per-pair "
  "taker-flow / tick data at entry, order-book depth for bot fills, news / listing events, cross-exchange premium.")
P("5. **Universe list lag:** the top-50 breadth for fills after 2026-10-11 uses the 10-11 list (cache end; up to 15 days stale).")
P("6. **Not tested exhaustively:** 3-way interactions; non-monotonic mid-bands beyond the quintile tables; thresholds other than "
  "sign / median / terciles; regime-conditional exits (exit side is a separate study).")
P("7. **Replay coverage:** only engine-stamped columns exist in the replay — `confidence`, burst / scan, pair-strength and cooldown "
  "features have no replay check (n/a in the tables). The replay refutes, never proves.")
P("8. **Statistical power:** KEEP N = 158 over 71 days; the most recent era group (where the losses are) has 37 fills over 13 days. A "
  "real +0.2 % separator in a 30-fill zone is below the detectable size with this N (family-wise 95 % |Δ| ≈ 0.82 in 1D).\n")

P("## 12 · FROZEN pre-registration for B25 (thresholds fixed 2026-10-26, never re-tuned)\n")
P(f"Cohort for every line: MOMENTUM LONG, CLOSED, non-probe, non-MANUAL fills opened **≥ 2026-10-23 18:00 UTC** (B25 onward; the master "
  f"fills that produced the hypotheses never count). % at 1×. DAY units for market-wide lines. ML breakeven WR frozen at **{BE*100:.1f}%** "
  "(KEEP, 2026-10-26). Filter verdict (expectancy bar): N ≥ 15 on ≥ 8 days ∧ WR < breakeven ∧ day-bootstrap P(avg<0) ≥ 0.95 ∧ no day / "
  "pair ≥ 50 % of the zone's net loss → FILTER CANDIDATE (then 30–50 % haircut + tight revert gate at promotion); RETIRE at N ≥ 15 if zone "
  "avg ≥ rest avg. Winner verdict (Pattern-W): N ≥ 30 ∧ WR ≥ 70 % ∧ avg ≥ +0.10 ∧ $ > 0 ∧ zone − rest > 0 at day-bootstrap P ≥ 0.90 → "
  "SIZING CANDIDATE (operator call, cell-registry read first); RETIRE at N ≥ 15 if zone avg ≤ rest avg. Verdict frozen at the first "
  "crossing; one re-read at N = 30.\n")
P("| # | line | zone definition (frozen) | side | data needed | B25 so far |\n|---|---|---|---|---|---|")
reg = [
    ("P1", "ML_SBULL_W", "`entry_btc_regime == STRONG_BULL`", "winner", "stamp", HEAD["H1 STRONG_BULL regime (winner side)"]),
    ("P1b", "ML_BTCADX_W (alias, read together)", "`entry_btc_adx > 27.99`", "winner", "stamp", HEAD["H1b BTC ADX > 27.99 (alias of H1)"]),
    ("P2", "ML_ADXFLAT_NOTSBULL", "`entry_adx_delta ≤ 0.309` ∧ `entry_btc_regime ≠ STRONG_BULL`", "loser", "stamp",
     HEAD["H2 pair ADXΔ ≤ 0.309 ∧ regime ≠ STRONG_BULL (loser)"]),
    ("P3", "ML_GAP58_LOW (refute-tracking)", "`entry_ema_gap_5_8 ≤ 0.082`", "loser", "stamp", HEAD["H3 EMA5−8 gap ≤ 0.082 (loser; dashboard Gap5-8 — the signed stamp holds the same values)"]),
    ("P4", "ML_BTC15_DOWN", "BTC 1m close at entry ≤ close 15 min earlier (`btc_r15 ≤ 0`, kline rebuild)", "loser", "BTC 1m klines",
     HEAD["H4 BTC 15-min return ≤ 0 (loser)"]),
    ("P5", "ML_QUIET_TAPE_W", f"top-50 15-min mean return ≤ {med['br_ret15']:+.4f} % ∧ pair 15-min quote vol / 24h mean ≤ {med['pair_vol15_x24h']:.3f}",
     "winner", "5m klines (top-50 + pair)", HEAD["H6 breadth ret15 ≤ med ∧ pair vol15 ≤ med (winner)"]),
    ("P6", "ML_CORR15_HIGH", "top-10 15-min return correlation `corr15 > 0.365` (Oct-15 grid definition)", "loser", "impulse grid rebuild", HEAD["H5 top-10 corr15 > 0.365 (loser)"]),
    ("P7", "ML_LOWSPIKE_LOWCORR_W", f"BTC `spike_30 ≤ {med['imp_spike_30']:.4f}` ∧ `corr15 ≤ {med['imp_corr15']:.4f}`", "winner", "impulse grid rebuild",
     HEAD["H7 imp spike_30 ≤ med ∧ corr15 ≤ med (winner)"]),
]
for a, nm, zdef, side, need, f in reg:
    z = f(B); v = z.notna(); z = z.fillna(False).astype(bool)
    P(f"| {a} | {nm} | {zdef} | {side} | {need} | scored {int(v.sum())}/{len(B)} · zone {st(B[z])} · rest {st(B[v & ~z])} |")
P("\nP3 is registered only to close it: it is already refuted on the 115 older master fills (Δ 0.000) — retire at N ≥ 15 unless B25+ "
  "show WR < breakeven at P ≥ 0.95. P2 is the third read of ADX direction (DL 273 ② uses ADXΔ ≤ 0 on a closed-bar rebuild); it does "
  "not replace that line — both are read, and evidence counts once per fill.\n")

P("### 12b · Existing frozen ML lines — B25 tally from stamps (definitions unchanged; full reads by their scouts)\n")
P("| line | B25 zone (N·WR·avg) | fills |\n|---|---|---|")
for nm, z in EX:
    z = z.fillna(False).astype(bool)
    P(f"| {nm} | {st(B[z])} | {', '.join(f'{p} {q:+.2f}' for p, q in zip(B[z].pair, B[z].y)) or '—'} |")
P("\nRead by their own scouts at the review (not recomputed here): ML_MARKET_TURN (DL 299, `scout_ml_turn.py`, needs the impulse grid "
  "extended), TREND_ALIGNED (DL 258, `scout_ml_trend_lines.py`), NEG_DAILYUP_WEAKPAIR (DL 253), ML_STOP_COOLDOWN (DL 248), C∪D "
  "(DL 127), REBOUND watch, the LONG_CHOP_BURST / mega-cap / heat revert tallies.\n")

P("## 13 · Files\n")
P("- Scripts: `scripts/study_ml_wl_b25prep_data.py` (loader) · `_fetch.py` (Binance top-up) · `_features.py` (impulse rebuild) · "
  "`_analyse.py` (screen) · `_report.py` (this file)")
P("- Outputs: `reports/PREP_B25_ML_WL.md` · `_engine.txt` · `_coverage.csv` · `_screen1d.csv` · `_screen2d.csv` · `_candidates.csv` · "
  "`_quintiles.csv` · `_dashboard_vars.csv` · `_post_entry.csv` · `_replay_baserate.csv` · `_features_new.csv` · kline top-up "
  "`reports/PREP_B25_ML_WL_fetch/`")
open(OUT + ".md", "w").write("\n".join(L))
print(f"wrote {OUT}.md ({len(L)} lines)")
