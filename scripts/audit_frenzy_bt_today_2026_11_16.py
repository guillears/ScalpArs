#!/usr/bin/env python3
"""FRENZY family + Flatbase 2-year backtest on TODAY's rules (operator 2026-11-16 "rebuild the backtest for the Frenzy sleeves first").
Read-only, no network. Assembles the validated per-sleeve walkers already priced on the yr6 / yr6a real-engine replay (Oct-13 stack,
seeds 1-3, warm-up trimmed) and the Flatbase S4 simulator, re-priced to today's exits, then applies today's FITTED armed filters as a
second read. Sources:
  FRENZY_LITE   reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv — replay exit +3 / −3 / 720 min == today's (frenzy_tp_pct 3.0 then = LITE TP today)
  FRENZY_LONG   same fills, re-priced to +1.5 / −3: peak_pnl ≥ +1.5 → +1.5 (first touch; exact for a lower TP, same stop / cap), else replay pct
  FRENZY_WILLY  reports/PREP_B25_BT_WILLY_rows.csv arm A (the yr6 / yr6a WILLY fills re-walked on their own paths at the LIVE V2b exit:
                target max(1.25, min(8, net at spike high × 0.99)), stop −4, late +0.25 @ 30, cap 90, resting maker TP) — runner NOT modelled
  FRENZY_WILLY_HT  PREP arm B (study replica, R ≥ 1, 2026-01-10 → 10-07, one realisation, V2b exit — today's HT TP floor +2 not modelled)
  FLATBASE_SHORT   reports/FLATBASE_SL_GRID_1113_bt.pkl (S4 simulator, TP 5 / SL 10 = today's '5/10', higher-high flag hh; costs 0.20 %)
  FRENZY_LITE_RETRY  not in any replay (sleeve shipped after the snapshot) → untestable here.
Filters (fitted on the master, Jun-17 → Nov): services.watch_filters.frenzy_blocks with today's toggles on the replay stamps."""
import os, sys, json, warnings
import numpy as np, pandas as pd
from types import SimpleNamespace
warnings.simplefilter("ignore")
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, R); os.chdir(R)
from services import watch_filters as WF  # noqa: E402

TH = SimpleNamespace(**{v: True for v in WF.TOGGLES.values()})
rng = np.random.default_rng(1116)
OUT = []


def regime(ts):
    ts = pd.Timestamp(ts)
    return "2025" if ts.year == 2025 else ("26-H1" if ts.month <= 6 else "26-H2")


def stats(df, p="p"):
    if not len(df):
        return dict(N=0)
    x = df[p].values
    g = df.groupby("day")[p].apply(np.array); keys = list(g.index)
    bs = [np.concatenate([g.iloc[i] for i in rng.integers(0, len(keys), len(keys))]).mean() for _ in range(1500)]
    days = pd.date_range(df.day.min(), df.day.max()).strftime("%Y-%m-%d")
    r = (df.groupby("day")[p].sum() / 100).reindex(days, fill_value=0)
    dcr = ((1 + r).prod() ** (1 / len(days)) - 1) * 100
    return dict(N=len(df), WR=round((x > 0).mean() * 100), avg=round(x.mean(), 3), lo=round(np.percentile(bs, 2.5), 3),
                hi=round(np.percentile(bs, 97.5), 3), P=round((np.array(bs) > 0).mean(), 2), days=df.day.nunique(), dcr=round(dcr, 3))


def fmt(s):
    return "—" if not s.get("N") else f"{s['N']} · {s['WR']}% · {s['avg']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] · P {s['P']:.2f} · {s['days']}d · DCR {s['dcr']:+.3f}"


d = pd.read_csv("reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv", low_memory=False)
d = d[d.status.eq("CLOSED") & ~d.is_probe.astype(bool)]
n = lambda df, c: pd.to_numeric(df[c], errors="coerce")
rows = []
# LITE
L = d[d.entry_strategy.eq("FRENZY_LITE")].copy(); L["sleeve"] = "FRENZY_LITE"; L["p"] = n(L, "pnl_percentage"); rows.append(L)
# LONG re-priced
G = d[d.entry_strategy.eq("FRENZY_LONG")].copy(); G["sleeve"] = "FRENZY_LONG"
G["p"] = np.where(n(G, "peak_pnl") >= 1.5, 1.5, n(G, "pnl_percentage")); rows.append(G)
# WILLY: replay rows for stamps, priced from PREP arm A
W = d[d.entry_strategy.eq("FRENZY_WILLY")].copy(); W["sleeve"] = "FRENZY_WILLY"
W["te"] = pd.to_datetime(W.opened_at, format="mixed").values.astype("datetime64[ms]").astype("int64")
A = pd.read_csv("reports/PREP_B25_BT_WILLY_rows.csv", low_memory=False); A = A[A.arm.eq("A_replay")][["seed", "pair", "te", "V2b", "V2b_why"]].copy()
A["te"] = pd.to_numeric(A.te).astype("int64")
W = W.sort_values("te"); A = A.sort_values("te")
W = pd.merge_asof(W, A, on="te", by=["seed", "pair"], tolerance=180_000, direction="nearest")
W_unm = W.V2b.isna().sum(); W = W[W.V2b.notna()].copy(); W["p"] = W.V2b; rows.append(W)
F = pd.concat(rows, ignore_index=True)
F["day"] = pd.to_datetime(F.opened_at, format="mixed").dt.strftime("%Y-%m-%d"); F["reg"] = F.opened_at.map(regime)
F["pre_master"] = pd.to_datetime(F.opened_at, format="mixed") < pd.Timestamp("2026-06-17")
blk = []
for r in F.itertuples():
    b = WF.frenzy_blocks(TH, r.sleeve, adx=n(F.loc[[r.Index]], "entry_adx").iloc[0] if False else getattr(r, "entry_adx"),
                         ema_gap_8_13=getattr(r, "entry_ema_gap_8_13"), vs_vwap_pct=getattr(r, "entry_frenzy_vs_vwap_pct"),
                         btc_1h_slope=getattr(r, "entry_btc_1h_slope"), btc_ema20_slope_raw=getattr(r, "entry_btc_ema20_slope"),
                         frenzy_vol_mult=getattr(r, "entry_frenzy_vol_mult"), btc_rsi=getattr(r, "entry_btc_rsi"),
                         above_share=getattr(r, "entry_frenzy_above_share"))
    blk.append("+".join(b))
F["blk"] = blk
# HT (study replica, one realisation)
B = pd.read_csv("reports/PREP_B25_BT_WILLY_rows.csv", low_memory=False); H = B[B.arm.eq("B_study") & B.sleeve.eq("FRENZY_WILLY_HT")].copy()
H["p"] = H.V2b; H["day"] = pd.to_datetime(pd.to_numeric(H.te), unit="ms").dt.strftime("%Y-%m-%d"); H["seed"] = 1
H["reg"] = pd.to_datetime(H.day).map(regime); H["pre_master"] = pd.to_datetime(H.day) < pd.Timestamp("2026-06-17")
# Flatbase
P = pd.read_pickle("reports/FLATBASE_SL_GRID_1113_bt.pkl").copy()
P["p"] = pd.to_numeric(P["5/10"], errors="coerce"); P = P[P.p.notna()]
P["day"] = pd.to_datetime(P.spike, format="mixed").dt.strftime("%Y-%m-%d"); P["seed"] = 1
P["reg"] = pd.to_datetime(P.day).map(regime); P["pre_master"] = pd.to_datetime(P.day) < pd.Timestamp("2026-06-17")
P_today = P[~P.hh.astype(bool)]

lines = []
def block(title, df, filt=None):
    lines.append(f"\n### {title}")
    lines.append("| read | seed | ALL | 2025 | 26-H1 | 26-H2 | pre-June-2026 (clean OOS) |")
    lines.append("|---|---|---|---|---|---|---|")
    for lab, X in (("no fitted filters", df),) + ((("today's filters", df[df.blk.eq("")]),) if filt else ()):
        for sd in sorted(X.seed.unique()):
            S = X[X.seed.eq(sd)]
            cells = [fmt(stats(S))] + [fmt(stats(S[S.reg.eq(g)])) for g in ("2025", "26-H1", "26-H2")] + [fmt(stats(S[S.pre_master]))]
            lines.append(f"| {lab} | {sd} | " + " | ".join(cells) + " |")
    return

for sl in ("FRENZY_WILLY", "FRENZY_LITE", "FRENZY_LONG"):
    block(sl, F[F.sleeve.eq(sl)], filt=True)
block("FRENZY_WILLY_HT (study replica, 2026-01 → 10, one realisation; HT +2 TP floor not modelled)", H)
block("FLATBASE_SHORT (S4 simulator, TP 5 / SL 10, costs 0.20 %, one realisation) — all signals", P)
block("FLATBASE_SHORT — today's higher-high filter applied (fitted on the master: DL 540)", P_today)
hdr = [f"WILLY replay rows without a V2b price (dropped): {W_unm}",
       "Filter blocks on the replay (today's toggles): " + str({f"{a}:{b}": int(c) for (a, b), c in F[F.blk.ne("")].groupby(["sleeve", "blk"]).size().items()})]
open(os.path.join(R, "scripts", ".audit_frenzy_bt_today.out.md"), "w").write("\n".join(hdr + lines))
print("\n".join(hdr + lines))
