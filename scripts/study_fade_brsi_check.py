#!/usr/bin/env python3
"""SPIKE_FADE BTC-RSI band check (2026-10-09) — is the (45, 50] band (admitted since the Sep-24 45→50 raise) costing money?

Read-only. No network. Cohorts, never mixed:
  (a) yr5 replay SPIKE_FADE fills (code 181131e, ceiling 50), 3 seeds, warm-up trimmed (scripts/yr5_fills_trimmed.py)
  (b) live master fades (reports/MASTER_POOL_stacked.csv, STACK 2026-10-08c), kept (stack_keep), CLOSED, today's-stack pct
  (c) forward cohort of the 50 rule = live fades opened after the raise deploy (2026-09-24 21:36 UTC), master + batch files +
      Downloads exports, dedup (opened_at, pair, direction)
  (d) Sep-24 tick-priced evidence — not saved as rows; quoted from DECISION_LOG 112 in the report.
Engine semantics (services/trading_engine.py 9340 / 17266): block iff _current_btc_rsi > spike_fade_max_btc_rsi (strict) where
_current_btc_rsi = RSI of the BTC 5m series INCLUDING the forming candle (get_ohlcv 5m 100, last row forming). So the 45 rule
admits bRSI <= 45 and the 50 rule admits bRSI <= 50: the band the raise added is (45, 50]. Stamp = entry_btc_rsi (same reading).
$ ruler: today's fade ticket on a $3k book = min(3000 x 0.24375 x 2 x 20, 0.5 % x 24h vol, $500k)  (H1 review section 1).
Writes reports/SPIKE_FADE_BTC_RSI_CHECK_2026-10-09.csv (all tables, long form) and prints them.
Usage: venv/bin/python scripts/study_fade_brsi_check.py
"""
import glob, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import yr5_fills_trimmed as YT                                          # noqa: E402

REP = os.path.join(ROOT, "reports")
OUT = os.path.join(REP, "SPIKE_FADE_BTC_RSI_CHECK_2026-10-09.csv")
RAISE_UTC = pd.Timestamp("2026-09-24 21:36:51")      # commit c0d7fedf 18:36:51 -03 (deploy follows within minutes)
TIGHTEN_UTC = pd.Timestamp("2026-08-05 16:16:40")    # commit 249ce1fa 13:16:40 -03
CEIL50_UTC = pd.Timestamp("2026-07-31 00:13:56")     # commit 93201381 21:13:56 -03 Jul-30
H1_END = pd.Timestamp("2026-05-20")
DESIRED = 3000 * 0.24375 * 2 * 20
rng = np.random.default_rng(20261009)
ROWS = []

BANDS = [("<=45", -1, 45.0), ("(45,50]", 45.0, 50.0)]
FINE = [("<=40", -1, 40.0), ("(40,45]", 40.0, 45.0), ("(45,47.5]", 45.0, 47.5), ("(47.5,50]", 47.5, 50.0)]


def ts(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:19].str.replace("T", " "), format="mixed", errors="coerce")


def ticket(vol):
    return np.fmin(np.fmin(DESIRED, 0.005 * pd.to_numeric(vol, errors="coerce").fillna(0)), 500_000.0)


def windows(t, gap_min=60):
    """Market-window id: fills (pooled over seeds) chained while consecutive opens are <= gap_min apart."""
    order = np.argsort(t.values)
    st = t.values[order]
    wid = np.zeros(len(st), int)
    for i in range(1, len(st)):
        wid[i] = wid[i - 1] + (1 if (st[i] - st[i - 1]) > np.timedelta64(gap_min, "m") else 0)
    out = np.empty(len(st), int); out[order] = wid
    return out


def boot(df, key, n=10000):
    """Cluster bootstrap of the mean pct, resampling clusters `key`. Returns (lo95, hi95, P(mean<0))."""
    g = df.groupby(key).pct.agg(["sum", "size"])
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    s, c = g["sum"].values, g["size"].values
    idx = rng.integers(0, len(g), (n, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m < 0).mean()


def max_loss_share(df, key):
    """Largest single cluster's share of the cohort's GROSS loss (sum of the negative cluster sums)."""
    g = df.groupby(key).pct.sum()
    neg = g[g < 0]
    if not len(neg):
        return 0.0, ""
    return neg.min() / neg.sum(), str(neg.idxmin())


def stats(df, cohort, label, nseeds=1):
    if not len(df):
        ROWS.append(dict(cohort=cohort, cut=label, N=0)); return
    df = df.copy()
    df["day"] = df.t.dt.floor("D")
    df["win"] = windows(df.t)
    lo, hi, pneg = boot(df, "day")
    wlo, whi, wpneg = boot(df, "win")
    sw, kw = max_loss_share(df, "win"); sp, kp = max_loss_share(df, "pair")
    r = dict(cohort=cohort, cut=label, N=len(df), N_per_seed=len(df) / nseeds, WR=(df.pct > 0).mean() * 100,
             avg_pct=df.pct.mean(), med_pct=df.pct.median(), sum_pct_per_seed=df.pct.sum() / nseeds,
             usd_per_seed=df.usd.sum() / nseeds, days=df.day.nunique(), windows=df.win.nunique(),
             day_ci_lo=lo, day_ci_hi=hi, P_mean_lt0_day=pneg, win_ci_lo=wlo, win_ci_hi=whi, P_mean_lt0_win=wpneg,
             top_window_loss_share=sw, top_pair_loss_share=sp, top_pair=kp, pairs=df.pair.nunique())
    ROWS.append(r)


def band_of(x, bands):
    for name, a, b in bands:
        if (x > a) and (x <= b):
            return name
    return ">50" if x > 50 else "nan"


# ---------------------------------------------------------------- (a) yr5 replay
F = YT.load(sleeves=["Spike-Fade"])
F["pct"] = F.pct.astype(float)
F["usd"] = ticket(F.entry_pair_volume_24h_usd) * F.pct / 100.0
F["brsi"] = pd.to_numeric(F.entry_btc_rsi, errors="coerce")
F["band"] = [band_of(x, BANDS) for x in F.brsi]
F["fine"] = [band_of(x, FINE) for x in F.brsi]
NS = F.seed.nunique()
assert F.brsi.notna().all() and F.brsi.max() <= 50.0 + 1e-9, "replay admitted a fade above the 50 ceiling?"
print(f"yr5 fades {len(F)} ({NS} seeds) {F.t.min()} -> {F.t.max()}  bRSI max {F.brsi.max():.2f}")

SEG = {"FULL": F, "H1 Jan04-May20": F[F.t < H1_END], "H2 May20-Oct04": F[F.t >= H1_END],
       "45-RULE ERA Aug05-Sep24": F[(F.t >= TIGHTEN_UTC) & (F.t < RAISE_UTC)],
       "POST-RAISE Sep24-Oct04": F[F.t >= RAISE_UTC]}
for seg, d in SEG.items():
    for b, _, _ in BANDS:
        stats(d[d.band == b], f"a_yr5|{seg}", b, NS)
    for b, _, _ in FINE:
        stats(d[d.fine == b], f"a_yr5|{seg}|fine", b, NS)
# per month x band
F["month"] = F.t.dt.to_period("M").astype(str)
for m, d in F.groupby("month"):
    for b, _, _ in BANDS:
        x = d[d.band == b]
        ROWS.append(dict(cohort="a_yr5|month", cut=f"{m} {b}", N=len(x), N_per_seed=len(x) / NS,
                         WR=(x.pct > 0).mean() * 100 if len(x) else np.nan, avg_pct=x.pct.mean() if len(x) else np.nan,
                         usd_per_seed=x.usd.sum() / NS, days=x.t.dt.floor("D").nunique()))
# per seed (seed robustness)
for s, d in F.groupby("seed"):
    for b, _, _ in BANDS:
        x = d[d.band == b]
        ROWS.append(dict(cohort="a_yr5|seed", cut=f"s{s} {b}", N=len(x), WR=(x.pct > 0).mean() * 100,
                         avg_pct=x.pct.mean(), usd_per_seed=x.usd.sum()))
# band difference (45,50] minus <=45, day-paired bootstrap over the full year
def diff_boot(d, n=10000):
    d = d.assign(day=d.t.dt.floor("D"))
    days = d.day.unique()
    gi = {k: g for k, g in d.groupby("day")}
    hs, hc, ls, lc = [], [], [], []
    for k in days:
        g = gi[k]; h = g[g.band == "(45,50]"]; l = g[g.band == "<=45"]
        hs.append(h.pct.sum()); hc.append(len(h)); ls.append(l.pct.sum()); lc.append(len(l))
    hs, hc, ls, lc = map(np.array, (hs, hc, ls, lc))
    idx = rng.integers(0, len(days), (n, len(days)))
    m = hs[idx].sum(1) / np.maximum(hc[idx].sum(1), 1) - ls[idx].sum(1) / np.maximum(lc[idx].sum(1), 1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m < 0).mean()
for seg, d in SEG.items():
    lo, hi, p = diff_boot(d)
    dd = d[d.band == "(45,50]"].pct.mean() - d[d.band == "<=45"].pct.mean()
    ROWS.append(dict(cohort=f"a_yr5|{seg}|diff", cut="(45,50] minus <=45", avg_pct=dd, day_ci_lo=lo, day_ci_hi=hi, P_mean_lt0_day=p))

# stale-candle bias by band (recall-trace signal file, window Jul-28 -> Oct-4)
S = pd.read_csv(os.path.join(REP, "study_fade_trace_signals.csv"), low_memory=False)
R = S[S.side == "REPLAY"].copy()
R["brsi"] = pd.to_numeric(R.stamp_btc_rsi, errors="coerce")
R["band"] = [band_of(x, BANDS) for x in R.brsi]
for b, d in R.groupby("band"):
    tt = d.truth_trig.astype(str) == "True"
    ROWS.append(dict(cohort="a_trace|replay window fades", cut=b, N=len(d), N_per_seed=len(d) / 3,
                     avg_pct=d.pct_rep.mean(), stale_1m_share=(d.src0 == "1M").mean(),
                     artefact_share=(~tt).mean(), avg_tick_true=d.pct_rep[tt].mean(), avg_tick_false=d.pct_rep[~tt].mean(),
                     n_class_era_brsi=(d["class"] == "ERA_CONFIG_BRSI").sum(), n_stale_artefact=(d["class"] == "REPLAY_STALE_ARTEFACT").sum()))
L = S[S.side == "LIVE"].copy()
miss = L[L.class_family.astype(str).str.startswith(("A data", "B data"))]
M0 = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
M0["t"] = ts(M0.opened_at)
mm = M0[M0.entry_strategy == "SPIKE_FADE"][["t", "pair", "entry_btc_rsi"]]
miss = miss.assign(t=ts(miss.t)).merge(mm, on=["t", "pair"], how="left")
miss["band"] = [band_of(x, BANDS) for x in pd.to_numeric(miss.entry_btc_rsi, errors="coerce")]
for b, d in miss.groupby("band"):
    ROWS.append(dict(cohort="a_trace|live fades the replay missed (A+B, stale data)", cut=b, N=len(d),
                     avg_pct=d.pct_live_stack.mean(), WR=(d.pct_live_stack > 0).mean() * 100))

# bRSI reading parity: live stamp vs replay stamp on matched signals
LV = M0[M0.entry_strategy == "SPIKE_FADE"].copy()
pairs = []
for _, r in LV.iterrows():
    c = F[(F.pair == r.pair) & ((F.t - r.t).abs() <= pd.Timedelta(minutes=10))]
    for _, q in c.iterrows():
        pairs.append((r.entry_btc_rsi, q.brsi))
P = pd.DataFrame(pairs, columns=["live", "rep"]).dropna()
if len(P):
    dlt = P.rep - P.live
    flips = ((P.live > 45) != (P.rep > 45)).sum()
    ROWS.append(dict(cohort="a_parity|bRSI live stamp vs replay stamp (matched +-10 min)", cut="all", N=len(P),
                     avg_pct=dlt.mean(), med_pct=dlt.abs().median(), band_flips_at_45=flips))
    print(f"bRSI parity: {len(P)} matched pairs, mean replay-live {dlt.mean():+.2f}, median |d| {dlt.abs().median():.2f}, 45-side flips {flips}")

# ---------------------------------------------------------------- (b) live master
M = M0[(M0.entry_strategy == "SPIKE_FADE") & (M0.status == "CLOSED")].copy()
M["pct"] = pd.to_numeric(M.stack_pct, errors="coerce")
M["usd"] = ticket(M.entry_pair_volume_24h_usd) * M.pct / 100.0
M["brsi"] = pd.to_numeric(M.entry_btc_rsi, errors="coerce")
M["band"] = [band_of(x, BANDS) for x in M.brsi]
M["fine"] = [band_of(x, FINE) for x in M.brsi]
M["rule"] = np.select([M.t < CEIL50_UTC, M.t < TIGHTEN_UTC, M.t < RAISE_UTC], ["none(pre-Jul31)", "50", "45"], "50")
K = M[M.stack_keep.astype(str) == "True"]
# breakeven WR of the fade sleeve on current-stack kept fills
aw, al = K.pct[K.pct > 0].mean(), K.pct[K.pct <= 0].mean()
BE = abs(al) / (aw + abs(al)) * 100
ROWS.append(dict(cohort="b_master|breakeven", cut="kept fades today's stack", N=len(K), WR=(K.pct > 0).mean() * 100,
                 avg_pct=K.pct.mean(), avg_win=aw, avg_loss=al, breakeven_WR=BE))
print(f"fade breakeven WR (master kept, N={len(K)}): avg win {aw:+.3f} / avg loss {al:+.3f} -> {BE:.1f}%")
Fw, Fl = F.pct[F.pct > 0].mean(), F.pct[F.pct <= 0].mean()
ROWS.append(dict(cohort="a_yr5|breakeven", cut="all yr5 fades", N=len(F), avg_win=Fw, avg_loss=Fl,
                 breakeven_WR=abs(Fl) / (Fw + abs(Fl)) * 100, WR=(F.pct > 0).mean() * 100))
for b, _, _ in BANDS + [(">50", 0, 0)]:
    stats(K[K.band == b], "b_master|kept|all eras", b)
for b, _, _ in FINE:
    stats(K[K.fine == b], "b_master|kept|all eras|fine", b)
for rule, d in K.groupby("rule"):
    for b, _, _ in BANDS + [(">50", 0, 0)]:
        stats(d[d.band == b], f"b_master|kept|rule {rule}", b)
MA = M.assign(pct=pd.to_numeric(M.pnl_percentage, errors="coerce"))
MA["usd"] = ticket(MA.entry_pair_volume_24h_usd) * MA.pct / 100.0
for b, _, _ in BANDS + [(">50", 0, 0)]:
    stats(MA[MA.band == b], "b_master|ALL incl stack-blocked (as-traded pct)", b)
    stats(MA[(MA.band == b) & (MA.stack_keep.astype(str) != "True")], "b_master|stack-BLOCKED only (as-traded pct)", b)
print(MA[MA.band == "(45,50]"][["opened_at", "pair", "era", "rule", "brsi", "pct", "stack_pct", "stack_keep", "stack_block_reason"]].to_string())

# ---------------------------------------------------------------- (c) forward cohort of the 50 rule
files = [f for f in sorted(glob.glob(os.path.join(REP, "BASELINE1[2-9]_*orders*.csv"))) if "superseded" not in f]
files += sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-*.csv")))
parts = []
for f in files:
    try:
        d = pd.read_csv(f, low_memory=False)
    except Exception:
        continue
    if "entry_strategy" not in d or "opened_at" not in d:
        continue
    d = d[d.entry_strategy == "SPIKE_FADE"].copy()
    if not len(d):
        continue
    d["src"] = os.path.basename(f)
    parts.append(d)
B = pd.concat(parts, ignore_index=True)
B["t"] = ts(B.opened_at)
B = B[(B.t >= RAISE_UTC) & (B.status == "CLOSED")]
B = B.sort_values("src").drop_duplicates(["opened_at", "pair", "direction"], keep="last")
# attach master's today-stack pct where present (else as-traded)
B = B.merge(M[["t", "pair", "pct", "stack_keep", "stack_block_reason"]], on=["t", "pair"], how="left")
B["pct_traded"] = pd.to_numeric(B.pnl_percentage, errors="coerce")
B["in_master"] = B.pct.notna()
B["pct"] = B.pct.fillna(B.pct_traded)
B["usd"] = ticket(B.entry_pair_volume_24h_usd) * B.pct / 100.0
B["brsi"] = pd.to_numeric(B.entry_btc_rsi, errors="coerce")
B["band"] = [band_of(x, BANDS) for x in B.brsi]
B["fine"] = [band_of(x, FINE) for x in B.brsi]
B["minute"] = B.t.dt.floor("min")
print(f"forward fades since raise: {len(B)} (in master {B.in_master.sum()}) last open {B.t.max()}")
for b, _, _ in BANDS:
    stats(B[B.band == b], "c_forward|since raise (stack pct)", b)
    x = B[B.band == b].assign(pct=lambda z: z.pct_traded)
    stats(x, "c_forward|since raise (as-traded pct)", b)
for b, _, _ in FINE:
    stats(B[B.fine == b], "c_forward|since raise|fine", b)
G = B[B.band == "(45,50]"].groupby("minute").agg(pct=("pct", "mean"), pct_traded=("pct_traded", "mean"), n=("pct", "size"))
ROWS.append(dict(cohort="c_forward|REVERT GATE (DECISION_LOG 112: fresh (45,50] fades, same-minute once; N>=10 -> WR<55 or sum<0 -> 45)",
                 cut="(45,50]", N=len(G), WR=(G.pct > 0).mean() * 100 if len(G) else np.nan,
                 sum_pct_per_seed=G.pct.sum(), avg_pct=G.pct.mean() if len(G) else np.nan,
                 sum_pct_traded=G.pct_traded.sum()))
cols = ["opened_at", "pair", "brsi", "band", "pct_traded", "pct", "usd", "stack_block_reason", "close_reason", "src"]
cols = [c for c in cols if c in B.columns]
print(B.sort_values("t")[cols].to_string())
B.sort_values("t")[cols].to_csv(OUT.replace(".csv", "_forward_fills.csv"), index=False)

T = pd.DataFrame(ROWS)
T.to_csv(OUT, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
show = ["cohort", "cut", "N", "N_per_seed", "WR", "avg_pct", "usd_per_seed", "days", "windows", "day_ci_lo", "day_ci_hi",
        "P_mean_lt0_day", "P_mean_lt0_win", "top_window_loss_share", "top_pair_loss_share", "top_pair"]
print(T[[c for c in show if c in T.columns]].round(3).to_string())
extra = [c for c in T.columns if c not in show]
print(T[["cohort", "cut"] + extra].dropna(how="all", subset=extra).round(3).to_string())
print("wrote", OUT)
