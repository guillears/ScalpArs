#!/usr/bin/env python3
"""🫧🔁 Oct-5 (operator: "the 9/10 heat-block winners were in a BULL market — BTC daily EMA50 above EMA200 — the earlier ones in BEAR; or some
macro BTC variable decides which rule makes sense when"). Read-only research.

Cohort = momentum-LONG signals inside the heat zone, three sources, every one tagged with BTC macro state at the signal (last CLOSED bar):
  yr5   every LONG_HEAT_BLOCK signal of the yr5 replay (breadth-only re-scope: bull ≥ 85, not washed out) — re-priced with the live
        momentum-LONG exit replica (scout price_ml), one per pair per 30 min, de-duplicated across seeds (5-min bucket)
        + yr5 FILLS in the original rule's extra zone (bull 80–85 ∧ BTC slope ≥ 0.07 ∧ BTC RSI prev ≥ 64), outcomes as replayed
  master the real fills the heat rules touch (re-scope blocks bull ≥ 85; the 4 the 3-leg rule re-blocks), as traded
  fwd   the scout's re-priced re-scope blocks after Sep-25 (SCOUT_REVERT_GATES.json, HEAT items, +1 min entry)
Macro tags: D1 = BTC daily EMA50 > EMA200 (golden-cross regime) · P200 = BTC close > daily EMA200 · H4 = BTC 4h EMA50 > EMA200 ·
R30 = BTC 30-day return > 0 · R7 = 7-day return > 0. Units: signals AND distinct days (market-wide state → days are the observations).
Out: reports/HEAT_REGIME_REVIEW_2026-10-05.md
"""
import json, os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path[:0] = [ROOT, "scripts"]
C = "reports/backtest_cache"; S = os.environ.get("S", "/tmp")
D = pd.read_csv(f"{C}/k1d/BTCUSDT.csv").drop_duplicates("open_time").sort_values("open_time")
D["e50"] = D.c.ewm(span=50, adjust=False).mean(); D["e200"] = D.c.ewm(span=200, adjust=False).mean()
H4 = pd.read_csv(f"{C}/k4h/BTCUSDT.csv").drop_duplicates("open_time").sort_values("open_time")
H4["e50"] = H4.c.ewm(span=50, adjust=False).mean(); H4["e200"] = H4.c.ewm(span=200, adjust=False).mean()
DAY, H4MS = 86_400_000, 14_400_000


def macro(t):
    i = D.open_time.searchsorted(t - DAY, side="right") - 1          # last daily bar CLOSED before t
    j = H4.open_time.searchsorted(t - H4MS, side="right") - 1
    if i < 210 or j < 210:
        return {}
    d, h = D.iloc[i], H4.iloc[j]
    c30 = D.c.iloc[i - 30]; c7 = D.c.iloc[i - 7]
    return dict(D1=bool(d.e50 > d.e200), P200=bool(d.c > d.e200), H4=bool(h.e50 > h.e200), R30=bool(d.c > c30), R7=bool(d.c > c7),
                gap_d=round((d.e50 / d.e200 - 1) * 100, 2))


rows = []
# yr5 blocks — the already-priced re-admitted ones + price the rest
import heat_yr5_signals as HY   # noqa: E402  (scratch helper copied next to this script at run time)
for r in HY.signals():
    rows.append(dict(src="yr5 block (bull ≥85)", pair=r["pair"], t=r["t"], pct=r["pct"], n=r["n_seeds"]))
# yr5 fills in the original rule's extra zone
Y = pd.read_pickle(f"{S}/y5.pkl")
nn = lambda c: pd.to_numeric(Y[c], errors="coerce")
z = (nn("entry_bull_pct") >= 80) & (nn("entry_bull_pct") < 85) & (nn("entry_btc_ema20_slope") >= 0.07) & (nn("entry_btc_rsi_prev") >= 64) & ~(nn("entry_btc_off30d_high_pct") <= -10)
for r in Y[z].itertuples():
    rows.append(dict(src="yr5 fill (bull 80–85 ∧ BTC hot)", pair=r.pair, t=int((r.o - pd.Timestamp(0)).total_seconds() * 1000), pct=float(r.pct), n=1 / 3))
# master
P = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
P = P[(P.direction == "LONG") & (P.entry_strategy.fillna("MOMENTUM").astype(str) == "MOMENTUM") & (P.status.astype(str) == "CLOSED")]
P = P[~P.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE")].drop_duplicates(["opened_at", "pair"])
b = pd.to_numeric(P.entry_bull_pct, errors="coerce"); sl = pd.to_numeric(P.entry_btc_ema20_slope, errors="coerce"); rp = pd.to_numeric(P.entry_btc_rsi_prev, errors="coerce")
for r, bb, s_, rr in zip(P.itertuples(), b, sl, rp):
    if not (bb >= 80):
        continue
    hot = (s_ >= 0.07) and (rr >= 64)
    if bb >= 85 or hot:
        rows.append(dict(src="master (bull ≥85 or 80–85 hot)" if True else "", pair=r.pair, t=int((pd.Timestamp(str(r.opened_at)[:19]) - pd.Timestamp(0)).total_seconds() * 1000),
                         pct=float(r.pnl_percentage), n=1, zone=("≥85" if bb >= 85 else "80–85 hot")))
# forward (scout)
st = json.load(open("reports/SCOUT_REVERT_GATES.json"))
for k, it in (st.get("gates", {}).get("HEAT", {}).get("items", {}) or {}).items():
    if it.get("sim1") is not None:
        rows.append(dict(src="fwd re-scope block", pair=it["pair"], t=int(it["t"]), pct=float(it["sim1"]), n=1))
T = pd.DataFrame(rows)
M = pd.DataFrame([macro(t) for t in T.t]); T = pd.concat([T.reset_index(drop=True), M], axis=1)
T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d")
T.to_csv("reports/HEAT_REGIME_REVIEW_2026-10-05_signals.csv", index=False)


def cell(g):
    if not len(g):
        return "–"
    w = g.n
    return f"{w.sum():.0f} · {(g.pct > 0).mean() * 100:.0f}% · {np.average(g.pct, weights=w):+.3f} % · {g.day.nunique()} d"


L = ["# 🫧🔁 Heat zone × BTC macro regime — 2026-10-05", "", "Each cell: signals (per-seed weighted for yr5) · WR · avg % per signal · distinct days. Positive avg = the heat block "
     "would have REMOVED winners (block hurts); negative = it removed losers (block helps).", ""]
for var, lab in (("D1", "BTC daily EMA50 > EMA200"), ("P200", "BTC above daily EMA200"), ("H4", "BTC 4h EMA50 > EMA200"), ("R30", "BTC 30-day return > 0"), ("R7", "BTC 7-day return > 0")):
    L += [f"## {lab}", "", "| source | ALL | yes | no |", "|---|---|---|---|"]
    for s_, g in T.groupby("src", sort=False):
        L.append(f"| {s_} | {cell(g)} | {cell(g[g[var] == True])} | {cell(g[g[var] == False])} |")
    L.append("")
L += ["## When was BTC in each daily regime (EMA50 vs EMA200)?", ""]
D["t"] = pd.to_datetime(D.open_time, unit="ms"); D["bull"] = D.e50 > D.e200
seg = D[D.t >= "2026-01-01"].assign(ch=lambda x: x.bull.ne(x.bull.shift()).cumsum()).groupby("ch").agg(a=("t", "min"), b=("t", "max"), bull=("bull", "first"))
L += [f"- {r.a:%Y-%m-%d} → {r.b:%Y-%m-%d}: {'EMA50 ABOVE EMA200 (bull)' if r.bull else 'below (bear)'}" for r in seg.itertuples()]
open("reports/HEAT_REGIME_REVIEW_2026-10-05.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
