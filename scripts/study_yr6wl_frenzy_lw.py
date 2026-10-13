#!/usr/bin/env python3
"""YR6 WL — FRENZY question (3): FRENZY_LONG / FRENZY_WIDE (negative in every regime at +1.5) — any separator worth keeping them for, and
the LONG-blocks-LITE side effect from what the stamps / replay counters allow.

  venv/bin/python -I scripts/study_yr6wl_frenzy_lw.py   → reports/YR6_WL_FRENZY_lw.md

① Pair-universe split (top-50 by prior/same-day quote volume) and 24 h-volume terciles per regime — the one split that kept its sign in
   all three regimes for LONG in the summary (NOT a pre-registered hypothesis: seen in the summary table → priced here, observe-only at best).
   Expectancy bar on the would-be-blocked side; seeds 2+3 as the second computation; 2026 (full ticks) is the clean read.
② Peak-based read: share of fills whose replayed peak reached the TP (+1.5) vs the −3 stop, per regime (what +1.5 does to the cohort).
③ LONG-blocks-LITE: (a) replay counters FRENZY_LITE_PAIR_HELD (any holder; reports/backtest_cache/replay/yr6*_s1_meta.json, READ-ONLY;
   counters include each chunk's warm-up → upper bound); (b) proxy value of a LITE entry that comes AFTER a LONG / WIDE fill in the same
   (pair, spike) episode (what the blocked entries resemble — they would have entered while LONG held the pair) vs LITE first-in-episode;
   (c) $ bound = blocked count × proxy avg × LITE $ per 1 % (usd_new / pct_new median) vs LONG / WIDE own $ (usd_new, 2× / 1×).
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_yr6wl_frenzy_common as C  # noqa: E402

A = C.load(all_seeds=True)
out = []


def blk_read(S, b):
    y = S.y.values; yb = y[b]
    if b.sum() < 3:
        return dict(N_blk=int(b.sum()))
    lo, hi, pp = C.day_boot(yb, S.day.values[b])
    loss = -np.minimum(yb, 0)
    pc = pd.Series(loss, index=S.pair.values[b]).groupby(level=0).sum().sort_values(ascending=False)
    dc = pd.Series(loss, index=S.day.values[b]).groupby(level=0).sum().sort_values(ascending=False)
    r = dict(N_blk=int(b.sum()), days=len(set(S.day.values[b])), WR_blk=100 * (yb > 0).mean(), avg_blk=yb.mean(), ci=f"[{lo:+.2f}, {hi:+.2f}]",
             P_lt0=1 - pp, N_kept=int((~b).sum()), avg_kept=y[~b].mean(), be_wr=100 * C.breakeven_wr(y[~b]),
             top2pair=pc.head(2).sum() / max(loss.sum(), 1e-9), top1day=dc.head(1).sum() / max(loss.sum(), 1e-9), usd_blk=S.usd_new.values[b].sum())
    r["BAR"] = bool(r["WR_blk"] < r["be_wr"] and r["P_lt0"] >= 0.95 and r["days"] >= 8 and r["top2pair"] < 0.5 and r["top1day"] < 0.5 and r["N_blk"] >= 15)
    return r


out.append("## ① Universe split — block = pair-day NOT in top-50 (prior / same day)\n")
for way, seeds in (("seed 1", [1]), ("seeds 2+3", [2, 3])):
    rows = []
    for s in ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY"):
        for reg in C.REGIMES + ["2026 both", "ALL"]:
            S = A[(A.sleeve == s) & A.seed.isin(seeds)]
            S = S[S.regime == reg] if reg in C.REGIMES else (S[S.regime != "2025 BULL"] if reg == "2026 both" else S)
            rows.append(dict(sleeve=s, regime=reg, **blk_read(S, ~S.top50.values)))
    out.append(f"\n**{way}**\n"); out.append(C.md_table(pd.DataFrame(rows), 3))

out.append("\n\n## ① b. 24 h quote-volume terciles (cuts per sleeve on the 2-year seed-1 fills) — monotone?\n")
S1 = A[A.seed == 1].copy(); S1["v"] = pd.to_numeric(S1.entry_pair_volume_24h_usd, errors="coerce")
rows = []
for s in ("FRENZY_LONG", "FRENZY_WIDE"):
    g = S1[S1.sleeve == s]; q = g.v.quantile([1 / 3, 2 / 3]).values
    g = g.assign(t=np.where(g.v <= q[0], "T1 low", np.where(g.v <= q[1], "T2 mid", "T3 high")))
    for reg in C.REGIMES:
        h = g[g.regime == reg]
        rows.append(dict(sleeve=s, regime=reg, cuts_musd=f"{q[0] / 1e6:.0f} / {q[1] / 1e6:.0f}",
                         **{t: f"{h.y[h.t == t].mean():+.2f} ({(h.t == t).sum()})" for t in ("T1 low", "T2 mid", "T3 high")}))
out.append(C.md_table(pd.DataFrame(rows), 2))

out.append("\n\n## ② What +1.5 does — replayed peak reached +1.5 / +3 before exit, stop share (seed 1)\n")
rows = []
for s in ("FRENZY_LONG", "FRENZY_WIDE"):
    for reg in C.REGIMES:
        g = S1[(S1.sleeve == s) & (S1.regime == reg)]
        pk = pd.to_numeric(g.peak_pnl, errors="coerce")
        rows.append(dict(sleeve=s, regime=reg, N=len(g), peak_ge_1p5=100 * (pk >= 1.5).mean(), peak_ge_3=100 * (pk >= 3).mean(),
                         stop_share=100 * (g.y <= -2.9).mean(), avg_new=g.y.mean(), avg_old_plus3=pd.to_numeric(g.pct, errors="coerce").mean(),
                         be_wr_new=100 * C.breakeven_wr(g.y.values), WR_new=100 * g.win.mean()))
out.append(C.md_table(pd.DataFrame(rows), 2))

# ③ LONG-blocks-LITE
out.append("\n\n## ③ LONG-blocks-LITE side effect\n")
cnt = {}
for f in sorted(glob.glob(os.path.join(C.REP, "backtest_cache/replay/yr6*_s1_meta.json"))):
    m = json.load(open(f))
    era = "2025" if os.path.basename(f).startswith("yr6a") else "2026"
    for r in (m.get("filter_blocks") or {}).get("rows", []):
        if r["filter"] in ("FRENZY_LITE_PAIR_HELD", "FRENZY_PAIR_HELD_BY_LITE", "FRENZY_WIDE_PAIR_HELD_BY_LITE", "FRENZY_WILLY_PAIR_HELD", "FRENZY_LITE_MAX_SLOTS"):
            cnt[(era, r["filter"])] = cnt.get((era, r["filter"]), 0) + int(r["total"])
out.append("(a) Replay refusal counters, seed 1 (upper bound — each chunk's warm-up is counted too):\n")
out.append(C.md_table(pd.DataFrame([dict(era=k[0], counter=k[1], refusals=v) for k, v in sorted(cnt.items())]), 0))

S1 = S1.sort_values("ts")
L = S1[S1.sleeve == "FRENZY_LITE"].copy()
prev_lw = S1[S1.sleeve.isin(["FRENZY_LONG", "FRENZY_WIDE"])].groupby("ep").ts.min()
L["after_lw"] = L.ep.map(prev_lw) < L.ts
fz = S1[S1.sleeve.isin(["FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE"])]
first_any = fz.groupby("ep").ts.min()
L["first_in_ep"] = L.ts <= L.ep.map(first_any)
rows = []
for reg in C.REGIMES + ["ALL"]:
    g = L if reg == "ALL" else L[L.regime == reg]
    for lab, mm in (("LITE after a LONG/WIDE fill, same episode", g.after_lw), ("LITE not after LONG/WIDE", ~g.after_lw),
                    ("LITE first FRENZY entry of its episode", g.first_in_ep)):
        h = g[mm]
        if len(h) == 0:
            continue
        lo, hi, pp = C.day_boot(h.y.values, h.day.values)
        rows.append(dict(regime=reg, cohort=lab, N=len(h), days=h.day.nunique(), WR=100 * h.win.mean(), avg=h.y.mean(), ci=f"[{lo:+.2f}, {hi:+.2f}]"))
out.append("\n(b) Proxy — LITE fills that entered AFTER a LONG / WIDE fill in the same episode (seed 1):\n")
out.append(C.md_table(pd.DataFrame(rows), 3))
lite_usd_per_pct = float((L.usd_new / L.y).replace([np.inf, -np.inf], np.nan).median())
proxy = L[L.after_lw].y.mean()
held = cnt.get(("2025", "FRENZY_LITE_PAIR_HELD"), 0) + cnt.get(("2026", "FRENZY_LITE_PAIR_HELD"), 0)
lw_usd = S1[S1.sleeve == "FRENZY_LONG"].usd_new.sum(), S1[S1.sleeve == "FRENZY_WIDE"].usd_new.sum()
out.append(f"\n(c) Bound: LITE $ per 1 % (today's 10×, $2,900 book) ≈ ${lite_usd_per_pct:,.0f}. Refused LITE stretches (any holder, both years, seed 1) "
           f"≤ {held}. If EVERY one were held by LONG / WIDE and priced at the proxy avg {proxy:+.3f} %, LONG / WIDE saved ≤ "
           f"${-held * proxy * lite_usd_per_pct:,.0f} over 2 years; at the DL 280 yr5 rate (8 of 10 lost ≈ −1.8 %/fill) ≤ ${held * 1.8 * lite_usd_per_pct:,.0f}. "
           f"LONG / WIDE's own 2-year $ at 2× / 1× (seed 1): LONG ${lw_usd[0]:,.0f} · WIDE ${lw_usd[1]:,.0f} · sum ${sum(lw_usd):,.0f}.\n")
open(os.path.join(C.REP, "YR6_WL_FRENZY_lw.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
