#!/usr/bin/env python3
"""B25 prep — block ③ BACKTEST for the six SPIKE_FADE watch lines (DECISION_LOG 374, scripts/scout_fade_watch.py thresholds, frozen).

Source: reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv (yr6a 2025 + yr6 2026, seeds 1/2/3, warm-up TRIMMED, CLOSED, probes out), sleeve
Spike-Fade. Fade entry + exit rules of the replay config (frozen_config_yr6_01419d4) == today's trading_config.json on every spike_fade_*
key (checked 2026-10-26). Known stack gap: the replay ran the WILLY global hold ON (today OFF, DL 340) → fades the hold refused are missing.
Extra arm "s1tick" = seed 1 with the 12 worst stops re-priced on aggTrades (reports/FADE_SIZE_MCAP_2026-10-19_rows.csv src BT) — the
pricing the DL 374 lines ⑤ ⑥ were discovered on (IN-SAMPLE for ⑤ ⑥).
$ = fixed $2,900 book, engine sizing (usd_new, cell 2×); 1× $ = usd_new / cell_multiplier. Read-only, no network.
Outputs: reports/PREP_B25_BT_FADE_summary.csv · _monthly.csv · _rows.csv
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_b25_bt_common as S  # noqa: E402
import scout_fade_watch as FW   # noqa: E402  (frozen thresholds only)

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "reports", "PREP_B25_BT_FADE")
BOOK = 2900.0

d = pd.read_csv(os.path.join(R, "reports", "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv"), low_memory=False)
f = d[(d.sleeve == "Spike-Fade") & (~d.is_probe.astype(bool))].copy()
f["day"] = f.opened_at.astype(str).str[:10]
f["k"] = f.pair + "|" + f.opened_at.astype(str).str[:19]
f["arm"] = "s" + f.seed.astype(int).astype(str)
tk = pd.read_csv(os.path.join(R, "reports", "FADE_SIZE_MCAP_2026-10-19_rows.csv"))
tk = tk[tk.src == "BT"].copy(); tk["k"] = tk.pair + "|" + tk.opened_at.astype(str).str[:19]
s1 = f[f.seed == 1].merge(tk[["k", "pct"]].rename(columns={"pct": "pct_tick"}), on="k", how="left")
assert s1.pct_tick.notna().all(), "s1tick: unmatched rows"
s1["usd_new"] = s1.usd_new * np.where(s1.pct != 0, s1.pct_tick / s1.pct.replace(0, np.nan), 1.0)
s1["pct"] = s1.pct_tick; s1["arm"] = "s1tick"
F = pd.concat([f, s1.drop(columns="pct_tick")], ignore_index=True)
F["cm"] = F.cell_multiplier.astype(float)
F["usd2x"] = F.usd_new.astype(float)
F["usd1x"] = F.usd2x / F.cm
F["period"] = F.regime
F["month"] = F.opened_at.astype(str).str[:7]

adxd = pd.to_numeric(F.entry_adx_delta, errors="coerce"); adx = pd.to_numeric(F.entry_adx, errors="coerce")
badx = pd.to_numeric(F.entry_btc_adx_prev, errors="coerce"); mc = pd.to_numeric(F.entry_mcap_usd, errors="coerce")
hb = F.entry_btc_regime.astype(str) == FW.HB_REGIME
F["L1"] = np.where(adxd.isna(), "unscored", np.where(adxd < FW.ADX_DELTA_MIN, "blocked", "kept"))
F["L2"] = np.where(adx.isna(), "unscored", np.where(adx <= FW.PAIR_ADX_MAX, "blocked", "kept"))
F["L5"] = np.where(F.entry_btc_regime.isna() | adxd.isna(), "unscored", np.where(hb & (adxd <= FW.HB_ADXD_MAX), "blocked", "kept"))
F["L6"] = np.where(F.entry_btc_regime.isna() | badx.isna(), "unscored", np.where(hb & (badx > FW.HB_BTCADX_MIN), "blocked", "kept"))
F["L3"] = np.where(mc.isna(), "unscored", np.where(mc >= FW.MCAP_MIN, "zone", "rest"))
F["L4"] = np.where(F.entry_btc_regime.isna(), "unscored", np.where(hb, "zone", "rest"))
F["L4sub"] = np.where(hb & (pd.to_numeric(F.entry_btc_trend_gap_pct, errors="coerce") >= FW.HB_GAP_MIN), "zone", "rest")
ARMS = ["s1", "s2", "s3", "s1tick"]
PERIODS = ["ALL", "2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
NAMES = dict(L1="① ADX Δ < 2.359 (block weak side)", L2="② pair ADX ≤ 19.2 (block)", L5="⑤ HB ∧ ADX Δ ≤ 2.4341 (block)",
             L6="⑥ HB ∧ BTC ADX prev > 24.5874 (block)", L3="③ mcap ≥ $200M: 2× → 1.5× (size)", L4="④ HEALTHY_BEAR: 2× → 1× (size)",
             L4sub="④b HB ∧ BTC trend gap ≥ −0.13 (41h sub-zone): 2× → 1× (size)")


def max_dd(usd, t):
    s = pd.Series(np.asarray(usd, float), index=pd.to_datetime(t)).sort_index().cumsum()
    eq = BOOK + s
    return float((eq - eq.cummax()).min())


rows, mrows = [], []
for arm in ARMS:
    A = F[F.arm == arm]
    for per in PERIODS:
        P = A if per == "ALL" else A[A.period == per]
        if not len(P):
            continue
        base = S.summ(P, usd={"usd1x": "usd1x", "usd2x": "usd2x"})
        rows.append(dict(arm=arm, period=per, line="BASE", side="all", **base))
        for L in ("L1", "L2", "L5", "L6"):
            kept = P[P[L] == "kept"]; blk = P[P[L] == "blocked"]; uns = (P[L] == "unscored").sum()
            be = S.breakeven_wr(kept.pct.values)
            ds, dk = S.loss_share(blk, "day"); ps, pk = S.loss_share(blk, "pair")
            for side, g in (("blocked", blk), ("kept", kept)):
                if not len(g):
                    continue
                r = S.summ(g, usd={"usd1x": "usd1x", "usd2x": "usd2x"})
                extra = {}
                if side == "blocked":
                    bar = dict(wr_below_be=bool(r["WR"] < be), p95=bool((r["p_neg"] or 0) >= 0.95), days8=bool(r["days"] >= 8),
                               conc_ok=bool((ds if np.isfinite(ds) else 0) < 0.5 and (ps if np.isfinite(ps) else 0) < 0.5), n15=bool(r["N"] >= 15))
                    extra = dict(breakeven_wr=round(be, 1), top_day_loss_share=round(ds, 2) if np.isfinite(ds) else np.nan, top_day=dk,
                                 top_pair_loss_share=round(ps, 2) if np.isfinite(ps) else np.nan, top_pair=pk, unscored=int(uns),
                                 expectancy_bar=("MET" if all(bar.values()) else "fails: " + ",".join(k for k, v in bar.items() if not v)),
                                 after_usd2x=round(float(kept.usd2x.sum()), 0), before_usd2x=round(float(P[P[L] != "unscored"].usd2x.sum()), 0),
                                 after_avg=round(float(kept.pct.mean()), 4), before_avg=round(float(P[P[L] != "unscored"].pct.mean()), 4))
                rows.append(dict(arm=arm, period=per, line=L, side=side, **r, **extra))
        for L, mult in (("L3", 1.5), ("L3", 1.0), ("L4", 1.0), ("L4sub", 1.0)):
            z = P[P[L] == "zone"]
            if not len(z):
                continue
            r = S.summ(z, usd={"usd1x": "usd1x", "usd2x": "usd2x"})
            after = np.where(P[L] == "zone", P.usd2x * np.minimum(mult / P.cm, 1.0), P.usd2x)
            zu = z.usd2x * np.minimum(mult / z.cm, 1.0)
            dd0, dd1 = max_dd(P.usd2x, P.opened_at), max_dd(after, P.opened_at)
            wd0 = P.groupby("day").usd2x.sum().min(); wd1 = pd.Series(after, index=P.index).groupby(P.day).sum().min()
            ds, dk = S.loss_share(z, "day"); ps, pk = S.loss_share(z, "pair")
            rows.append(dict(arm=arm, period=per, line=f"{L}@{mult:g}x", side="zone", **r, size_to=mult,
                             zone_usd_after=round(float(zu.sum()), 0), delta_usd=round(float(zu.sum() - z.usd2x.sum()), 0),
                             sleeve_usd_before=round(float(P.usd2x.sum()), 0), sleeve_usd_after=round(float(after.sum()), 0),
                             worst_fill_pct_book_before=round(100 * z.usd2x.min() / BOOK, 1), worst_fill_pct_book_after=round(100 * zu.min() / BOOK, 1),
                             fills_over4pct_before=int((z.usd2x < -0.04 * BOOK).sum()), fills_over4pct_after=int((zu < -0.04 * BOOK).sum()),
                             worst_day_before=round(float(wd0), 0), worst_day_after=round(float(wd1), 0),
                             maxdd_before=round(dd0, 0), maxdd_after=round(dd1, 0),
                             top_day_loss_share=round(ds, 2) if np.isfinite(ds) else np.nan, top_pair_loss_share=round(ps, 2) if np.isfinite(ps) else np.nan,
                             top_pair=pk, unscored=int((P[L] == "unscored").sum())))
    for L in ("L1", "L2", "L5", "L6", "L3", "L4"):
        for (m, side), g in A[A[L] != "unscored"].groupby(["month", L]):
            mrows.append(dict(arm=arm, line=L, month=m, side=side, N=len(g), WR=round(100 * (g.pct > 0).mean(), 1), avg=round(g.pct.mean(), 4),
                              usd1x=round(g.usd1x.sum(), 0), usd2x=round(g.usd2x.sum(), 0)))

Sm = pd.DataFrame(rows); Mm = pd.DataFrame(mrows)
Sm.to_csv(OUT + "_summary.csv", index=False); Mm.to_csv(OUT + "_monthly.csv", index=False)
F[["arm", "seed", "pair", "opened_at", "day", "period", "pct", "usd1x", "usd2x", "close_reason", "entry_adx_delta", "entry_adx",
   "entry_btc_regime", "entry_btc_adx_prev", "entry_btc_trend_gap_pct", "entry_mcap_usd", "L1", "L2", "L3", "L4", "L4sub", "L5", "L6"]].to_csv(OUT + "_rows.csv", index=False)

# month-consistency: months where blocked avg < kept avg (N ≥ 3 both sides)
cons = []
for (arm, L), g in Mm[Mm.line.isin(["L1", "L2", "L5", "L6"])].groupby(["arm", "line"]):
    p = g.pivot(index="month", columns="side", values="avg"); n = g.pivot(index="month", columns="side", values="N")
    ok = (n.get("blocked", 0) >= 3) & (n.get("kept", 0) >= 3)
    if "blocked" in p and "kept" in p:
        cons.append(dict(arm=arm, line=L, months=int(ok.sum()), blocked_worse=int(((p.blocked < p.kept) & ok).sum()),
                         blocked_neg=int(((p.blocked < 0) & ok).sum())))
pd.DataFrame(cons).to_csv(OUT + "_monthcons.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
print(Sm[Sm.period == "ALL"][["arm", "line", "side", "N", "WR", "avg", "ci_lo", "ci_hi", "p_neg", "days", "usd1x", "usd2x", "breakeven_wr",
                              "expectancy_bar", "top_day_loss_share", "top_pair_loss_share", "unscored"]].to_string())
print(pd.DataFrame(cons).to_string())
