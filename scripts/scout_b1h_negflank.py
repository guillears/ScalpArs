#!/usr/bin/env python3
"""📉 Scout — ML_B1H_NEGFLANK observation (pre-registered 2026-10-05, operator "yes add it to scout"; OBSERVE only — never changes config).

Question: do full-size momentum LONGs opened while BTC's 1h slope is FALLING lose? yr5 says yes (falling-EMA20 fills −0.125 % vs rising
−0.001 %, 144 days), real master fills say no (44 · 77 % WR · +0.154 %, carried by the washed-out Jun-18→Jul-2 window; without it −0.040 %
vs +0.174 %). Research: reports/H1_EMA20_OVERLAP_2026-10-05.md. It is the negative flank the LONG_BTC1H_DEADBAND gate (blocks (−0.05, +0.025))
deliberately lets through — a pass would lead to a 1hPullback cell sizing verdict, then a dead-band re-scope review; never straight to arming.

COHORT (frozen)  CLOSED full-size MOMENTUM LONG fills (MANUAL and *_PROBE excluded) opened ≥ 2026-10-06 00:00 UTC with the stamped
                 entry_btc_1h_slope ≤ −0.05 (the engine's own gate input). Washed-out = entry_btc_off30d_high_pct ≤ −15 — shown both ways.
BAR (the locked expectancy filter bar, judged on the cohort EXCLUDING washed-out fills):
                 ① WR < the sleeve's breakeven WR = |avg loss| / (avg win + |avg loss|) on ALL momentum-long fills since the same start
                   (master reference 63.9 % until ≥ 30 sleeve fills) · ② P(mean < 0) ≥ 0.95 by a DAY-clustered bootstrap (market-wide
                   variable → the day is the unit) · ③ ≥ 8 distinct days ∧ no single day or pair ≥ 50 % of the cohort's loss · ④ N ≥ 15.
                 Then the 30–50 % in-sample haircut and a pre-committed revert gate. Never re-fit −0.05 / −15.
"""
import glob
import os

import numpy as np
import pandas as pd

START = "2026-10-06 00:00"
SLOPE_MAX, WASHED, N_MIN, DAYS_MIN, BE_REF, BE_MIN_FILLS = -0.05, -15.0, 15, 8, 63.9, 30
COLS = ("opened_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "cell_multiplier_source",
        "entry_btc_1h_slope", "entry_btc_off30d_high_pct")
YEAR_REF = ("yr5 falling 1h EMA20: 492 · 60 % · −0.125 % (144 days) vs rising −0.001 % · master falling 44 · 77 % · +0.154 % "
            "(ex washed-out 31 · −0.040 %)")


# ─────────────────────────── pure (selftest) ───────────────────────────
def breakeven_wr(pct):
    w, l = pct[pct > 0], pct[pct <= 0]
    if not len(w) or not len(l):
        return None
    aw, al = float(w.mean()), abs(float(l.mean()))
    return 100.0 * al / (aw + al) if (aw + al) > 0 else None


def p_mean_neg(pct, day, n=4000, seed=7):
    """day-clustered bootstrap: resample DAYS with replacement, pooled mean of their fills; → P(mean < 0)."""
    g = pd.DataFrame(dict(p=pct, d=day)).groupby("d").p.agg(["sum", "count"])
    if len(g) < 2:
        return None
    s, c = g["sum"].values, g["count"].values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    return float(((s[idx].sum(1) / c[idx].sum(1)) < 0).mean())


def top_loss_share(pct, key):
    """largest single key's share of the summed losing-day / losing-pair nets (conservative: one net-negative key = 100 %)."""
    net = pd.Series(pct).groupby(pd.Series(key).values).sum()
    neg = net[net < 0]
    return float(neg.min() / neg.sum()) if len(neg) else 0.0


def decide(c, be):
    """c = DataFrame(pct, day, pair) of the judged cohort → (state, detail)."""
    n, nd = len(c), c.day.nunique() if len(c) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "collecting", f"N {n}/{N_MIN} · days {nd}/{DAYS_MIN}"
    wr = 100.0 * float((c.pct > 0).mean())
    p = p_mean_neg(c.pct.values, c.day.values)
    sd, sp = top_loss_share(c.pct.values, c.day.values), top_loss_share(c.pct.values, c.pair.values)
    ok = wr < be and p is not None and p >= 0.95 and sd < 0.5 and sp < 0.5
    return ("passes" if ok else "fails"), (f"WR {wr:.0f} % vs breakeven {be:.1f} % · P(mean<0) {p:.2f} · top day {sd * 100:.0f} % / "
                                           f"top pair {sp * 100:.0f} % of the loss")


# ─────────────────────────── data ───────────────────────────
def _orders():
    fr = []
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
            if len(d) and {"opened_at", "pnl_percentage", "entry_btc_1h_slope"} <= set(d.columns):
                fr.append(d.assign(_m=os.path.getmtime(f)))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["ts", "pct", "slope", "off30", "day"])   # no usable export → "collecting", never a crash (review)
    o = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable")
    o["_k"] = o.opened_at.astype(str).str[:19]
    o = o.drop_duplicates(["_k", "pair", "direction"], keep="last")      # cross-export dedup key (opened_at, pair, direction)
    o = o[(o.status.astype(str).str.upper() == "CLOSED") & (o.direction.astype(str) == "LONG")]
    o = o[o.entry_strategy.fillna("").astype(str).str.split(":").str[0] == "MOMENTUM"]
    o = o[~o.get("cell_multiplier_source", pd.Series("", index=o.index)).astype(str).str.endswith("_PROBE")]
    t = pd.to_datetime(o._k, format="ISO8601", errors="coerce")
    o = o[t.notna()].assign(ts=t[t.notna()])
    o = o[o.ts >= pd.Timestamp(START)]
    o["pct"] = pd.to_numeric(o.pnl_percentage, errors="coerce")
    o["slope"] = pd.to_numeric(o.entry_btc_1h_slope, errors="coerce")
    o["off30"] = pd.to_numeric(o.get("entry_btc_off30d_high_pct"), errors="coerce")
    o["day"] = o.ts.dt.strftime("%Y-%m-%d")
    return o[o.pct.notna()]


def _line(g):
    return f"{len(g)} · {(g.pct > 0).mean() * 100:.0f}% · {g.pct.mean():+.3f} % · {g.day.nunique()} d" if len(g) else "0"


def run():
    """→ markdown lines. Never raises past the caller's try."""
    o = _orders()
    sl = o[o.slope.notna()]
    coh = sl[sl.slope <= SLOPE_MAX]
    washed = coh.off30.notna() & (coh.off30 <= WASHED)
    judged = coh[~washed]
    be_live = breakeven_wr(o.pct) if len(o) >= BE_MIN_FILLS else None
    be = be_live if be_live is not None else BE_REF
    state, det = decide(judged[["pct", "day", "pair"]], be)
    L = ["## 📉 ML_B1H_NEGFLANK — momentum longs opened while BTC 1h slope ≤ −0.05 (pre-registered, OBSERVE only)", "",
         f"Full-size momentum LONG fills from {START} UTC (stamped `entry_btc_1h_slope`; the dead-band's open negative flank). Judged on the "
         f"cohort without washed-out fills (BTC ≤ {WASHED:g} % below its 30-day high). Bar = the expectancy filter bar: WR < sleeve breakeven ∧ "
         f"day-clustered P(mean < 0) ≥ 0.95 ∧ ≥ {DAYS_MIN} days ∧ N ≥ {N_MIN} ∧ no day / pair ≥ 50 % of the loss. Year reference: {YEAR_REF}.", "",
         "| Group | N · WR · avg % · days |", "|---|---|",
         f"| **BTC 1h ≤ −0.05, ex washed-out (judged)** | {_line(judged)} |",
         f"| BTC 1h ≤ −0.05, washed-out only | {_line(coh[washed])} |",
         f"| BTC 1h > −0.05 (rest of the sleeve) | {_line(sl[sl.slope > SLOPE_MAX])} |",
         f"| no 1h stamp | {len(o) - len(sl)} |", "",
         f"Sleeve breakeven WR {be:.1f} % ({'live, ' + str(len(o)) + ' fills' if be_live is not None else 'master reference until ' + str(BE_MIN_FILLS) + ' sleeve fills'}) · "
         + {"collecting": "⏳ collecting", "passes": "📋 PASSES the bar → review (cell sizing verdict first, never straight to arming)",
            "fails": "❌ bar not met"}[state] + f" ({det})", ""]
    return L


# ─────────────────────────── self-test ───────────────────────────
def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    chk(abs(breakeven_wr(pd.Series([1.0, 1.0, -1.0])) - 50.0) < 1e-9, "breakeven = |L| / (W + |L|)")
    chk(breakeven_wr(pd.Series([1.0, 2.0])) is None, "no losers → None")
    days = [f"d{i}" for i in range(10) for _ in range(2)]
    neg = np.array([-0.2, 0.1] * 10)
    chk(p_mean_neg(neg, days) > 0.95, "consistently negative days → P ≈ 1")
    chk(p_mean_neg(-neg, days) < 0.05, "positive → P ≈ 0")
    chk(abs(top_loss_share([-1.0, -1.0, 0.5], ["a", "b", "b"]) - 1 / 1.5) < 1e-9, "top loss share on net-negative keys")
    c = pd.DataFrame(dict(pct=neg, day=days, pair=[f"P{i}" for i in range(20)]))
    chk(decide(c, 70.0)[0] == "passes", "WR 50 < 70, negative, spread → passes")
    chk(decide(c, 40.0)[0] == "fails", "WR above breakeven → fails")
    chk(decide(c.head(14), 70.0)[0] == "collecting", "N 14 → collecting")
    conc = c.copy(); conc.loc[0, "pct"] = -10.0
    chk(decide(conc, 70.0)[0] == "fails", "one day ≥ 50 % of the loss → fails")
    print(f"selftest OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
