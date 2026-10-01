#!/usr/bin/env python3
"""Momentum-LONG observe reads (2026-10-01, DECISION_LOG 160/161): reproduces the BURST-CROWDING tally and the BTC-CHOP (eff72) read.

Definitions (frozen — the CURRENT_STATE bars quote these):
  burst       a momentum LONG opened ≤ 120 s from ANOTHER bot fill of any sleeve (MANUAL excluded) in the same book; in the backtest
              the neighbour must be in the SAME seed. Windows: fills ≤ 120 s apart = one window.
  btc chop    eff72 ≤ 0.007 (frozen, CURRENT_STATE "ML BTC-CHOP OBSERVE"): the LIVE stamp entry_btc_eff72 (= int(eff×1000)/1000, so raw
              eff < 0.008; stamped since 2026-09-20), else the validated REBUILD from BTC 5m klines (engine formula; the live stamp
              can lag the rebuild by one bar — monitor throttle); fills with neither are excluded, never "rest". Tiers: ≤0.007 · 0.007<eff≤0.026 ·
              >0.026 (comparison line only). Units: DAYS and chop EPISODES (stamped days ≥ 72 h apart = separate episodes).
  FRESH       --since (default 2026-10-01T02:00) — the observe bars count ONLY these: full-size (1×, no *_PROBE), no MANUAL.
  Bar checks  WR vs the PINNED momentum-long breakeven 61 % (Sep-25) · 95 % day-bootstrap of the mean · max day / pair share of the loss.
Book: current_stack_ledger.build() (validated master, current stack) + --fresh orders exports (full-size, non-MANUAL rows after the
master's last fill). Backtest: yr3 replay, post-gate, seeds collapsed to ONE row per trade (pair × 5-min bucket; pnl = seed mean).
Usage: venv/bin/python scripts/ml_regime_observe_read.py --fresh <orders.csv> [--fresh …] [--chop 0.007] [--since 2026-10-01T02:00]"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
FRESH = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--fresh" and i + 1 < len(sys.argv)]
_arg = lambda k, d: next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == k and i + 1 < len(sys.argv)), d)
CHOP_ARG = float(_arg("--chop", "0.007")); SINCE = pd.Timestamp(_arg("--since", "2026-10-01T02:00"))
BREAKEVEN = 61.0                                            # pinned (momentum-long sleeve economics, Sep-25)
_a, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG  # noqa: E402
M = LG.build(); sys.argv = _a
n = lambda d, c: pd.to_numeric(d[c], errors="coerce")
BURST_S, CHOP = 120, CHOP_ARG

M["pct"] = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"), M.stack_pnl / n(M, "notional_value") * 100,
                    M.pnl_percentage)
last = pd.to_datetime(M.opened_at, format="ISO8601").max()
fr = []
for f in FRESH:
    b = pd.read_csv(f, low_memory=False)
    b = b[(b.status == "CLOSED") & (pd.to_datetime(b.opened_at, format="ISO8601") > last)
          & ~b.cell_multiplier_source.astype(str).str.contains("_PROBE")].copy()
    b["pct"] = b.pnl_percentage; b["era"] = "FRESH"; fr.append(b)
A = pd.concat([M] + fr, ignore_index=True).drop_duplicates(["opened_at", "pair", "direction"])
A = A[A.entry_strategy.fillna("MOMENTUM") != "MANUAL"].copy()
A["o"] = pd.to_datetime(A.opened_at, format="ISO8601")


def label(book, by=None, second=False):
    """burst flag for each momentum LONG of `book` (neighbours searched within the same `by` group, e.g. seed). second=True: only
    the 2nd+ fill of a burst (another bot fill opened ≤ 120 s BEFORE it) — the only fills a live rule could ever block."""
    flags = {}
    for _, g in (book.groupby(by) if by else [(None, book)]):
        t = ((g.o - pd.Timestamp(0)).dt.total_seconds()).values; idx = g.index.values   # unit-safe (pandas may parse to µs)
        ml = ((g.direction == "LONG") & (g.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")).values
        for i in np.where(ml)[0]:
            d = (t[i] - t) if second else np.abs(t - t[i]); d[i] = 10**9
            flags[idx[i]] = bool(((d >= 0) & (d <= BURST_S)).any())
    return pd.Series(flags)


def windows(g, by=None):
    g = g.sort_values([by, "o"] if by else "o"); w, cur, prev, pk = [], -1, None, None
    for t, k in zip(g.o, g[by] if by else [0] * len(g)):
        if prev is None or (t - prev).total_seconds() > BURST_S or k != pk:
            cur += 1
        w.append(cur); prev, pk = t, k
    return g.assign(win=w)


def line(lab, g, by=None, unit="win"):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – |"
    g = windows(g, by) if unit == "win" else g.assign(win=g.opened_at.astype(str).str[:10])
    v = g.groupby("win").pct.mean().values
    b = np.random.default_rng(1).choice(v, (10000, len(v))).mean(axis=1) if len(v) > 2 else np.array([np.nan])
    return (f"| {lab} | {len(g)} | {len(v)} | {(g.pct > 0).mean() * 100:.0f}% | {g.pct.mean():+.3f} | "
            f"[{np.nanpercentile(b, 2.5):+.2f}, {np.nanpercentile(b, 97.5):+.2f}] |")


ML = A[(A.direction == "LONG") & (A.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].copy()
ML["burst"] = label(A).reindex(ML.index).fillna(False).astype(bool)
ML["burst2"] = label(A, second=True).reindex(ML.index).fillna(False).astype(bool)   # 2nd+ fill of a burst
neg = n(ML, "entry_btc_1h_slope") < 0


NOTES = []


def _btc5_closes():
    """BTC 5m closes: the backtest cache + public klines since its end (read-only). A missing cache never crashes the read."""
    try:
        k = pd.read_csv(os.path.join(ROOT, "reports", "backtest_cache", "k5m_full", "BTCUSDT.csv"), usecols=["open_time", "c"])
        k = k.drop_duplicates("open_time").set_index("open_time").sort_index()
    except Exception as e:
        NOTES.append(f"BTC 5m cache unreadable ({e}) — eff72 = live stamps only"); return np.array([], dtype="int64"), np.array([])
    try:
        import ccxt
        ex = ccxt.binanceusdm({"enableRateLimit": True}); since = int(k.index[-1]) + 1; rows = []
        while True:
            r = ex.fetch_ohlcv("BTC/USDT:USDT", "5m", since=since, limit=1500)
            if not r:
                break
            rows += r; since = r[-1][0] + 1
            if len(r) < 1500:
                break
        if rows:
            e = pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c", "v"]).set_index("open_time")[["c"]]
            k = pd.concat([k, e]); k = k[~k.index.duplicated()].sort_index()
    except Exception as e:
        NOTES.append(f"BTC kline top-up failed ({e}) — fills after the cache end keep only their live stamp")
    return k.index.values, k.c.values


def eff72_rebuild(open_ms, T, C):
    """The engine's monitor eff (_update_bullrun_monitor: 864 closed 5m bars, |net| / path, truncated to 3 dp) at a fill time —
    validated 2026-10-01 against 24 live stamps: corr 0.999, same tier 96 %. NaN when the history is short / missing."""
    if not len(T):
        return np.nan
    i = np.searchsorted(T, open_ms, side="right") - 1
    if i < 0:
        return np.nan
    lc = i - 1 if T[i] + 300_000 > open_ms else i
    if lc < 999 or open_ms - T[lc] > 15 * 60_000:
        return np.nan
    w = C[lc - 863: lc + 1]; d = np.abs(np.diff(w)).sum()
    return int(abs(w[-1] - w[0]) / d * 1000) / 1000.0 if d > 0 else 0.0


_T, _C = _btc5_closes()
_om = ((pd.to_datetime(ML.opened_at, format="ISO8601") - pd.Timestamp(0)).dt.total_seconds() * 1000).astype("int64").values
ML["eff72_rb"] = [eff72_rebuild(x, _T, _C) for x in _om]
ML["eff72_src"] = np.where(n(ML, "entry_btc_eff72").notna(), "live", np.where(pd.notna(ML.eff72_rb), "rebuilt", "none"))
ML["eff72"] = n(ML, "entry_btc_eff72").where(n(ML, "entry_btc_eff72").notna(), ML.eff72_rb)   # live stamp wins
eff = ML["eff72"]
_v = ML[(ML.eff72_src == "live") & pd.notna(ML.eff72_rb)]
_tier = lambda x: np.digitize(x, [0.0071, 0.0261])
VALID = (f"rebuild vs live stamp on {len(_v)} fills: corr {np.corrcoef(n(_v, 'entry_btc_eff72'), _v.eff72_rb)[0, 1]:.3f} · same tier "
         f"{(_tier(n(_v, 'entry_btc_eff72')) == _tier(_v.eff72_rb)).mean():.0%}" if len(_v) > 2 else "rebuild validation: too few live stamps")
H = "| Cohort | N | windows | WR | avg % | 95 % window CI |\n|---|---|---|---|---|---|"
out = ["# Momentum-LONG observe reads — burst crowding & BTC chop", "", f"Master current stack + fresh: {len(ML)} momentum longs.", "",
       "## Burst crowding (master)", "", H, line("burst", ML[ML.burst]), line("alone", ML[~ML.burst]),
       line("slope<0 ∧ burst", ML[neg & ML.burst]), line("slope<0 ∧ alone", ML[neg & ~ML.burst]), "",
       f"## BTC chop eff72 ≤ {CHOP} (master; eff72 = live stamp on {int((ML.eff72_src == 'live').sum())} fills, validated rebuild on "
       f"{int((ML.eff72_src == 'rebuilt').sum())}, missing on {int((ML.eff72_src == 'none').sum())}; unit = DAY)", "",
       H.replace("windows", "days").replace("window CI", "day CI"),
       line(f"eff72 ≤ {CHOP}", ML[eff <= CHOP], unit="day"), line(f"eff72 > {CHOP}", ML[eff > CHOP], unit="day")]

F = pd.read_csv("reports/backtest_cache/replay/year/yr3_report_fills.csv", low_memory=False)
F = F[F.cell_multiplier_source.astype(str) != "CROSS_OB_OPEN"]
F = F[~((F.cell_multiplier_source.astype(str) == "NONEXP_CALM3D") & (n(F, "entry_btc_atr_pct") < 0.08))]
F = F[~((F.direction == "LONG") & (F.entry_strategy.fillna("MOMENTUM") == "MOMENTUM") & (n(F, "entry_adx") < 21)
        & (n(F, "entry_rsi") < n(F, "entry_rsi_prev")))].copy()
F["o"] = pd.to_datetime(F.opened_at, format="ISO8601")
F["burst"] = label(F, by="seed").reindex(F.index)
FL = F[(F.direction == "LONG") & (F.entry_strategy.fillna("MOMENTUM") == "MOMENTUM")].copy()
FL["k"] = FL.pair.astype(str) + "|" + FL.o.dt.floor("5min").astype(str)
agg = FL.groupby("k").agg(pct=("pnl_percentage", "mean"), burst=("burst", "mean"))
FL = FL.drop_duplicates("k").set_index("k"); FL["pct"] = agg.pct; FL["burst"] = agg.burst >= 0.5; FL = FL.reset_index()
fneg = n(FL, "entry_btc_1h_slope") < 0; feff = n(FL, "entry_btc_eff72"); h1 = FL.opened_at.astype(str) < "2026-05-01"
out += ["", "## yr3 backtest, post-gate, seeds collapsed (H1 Jan–Apr · H2 May–Sep)", "", H]
for lab, hm in (("H1", h1), ("H2", ~h1)):
    out += [line(f"{lab} slope<0 ∧ burst", FL[fneg & FL.burst & hm]), line(f"{lab} slope<0 ∧ alone", FL[fneg & ~FL.burst & hm])]
out += ["", H.replace("windows", "days").replace("window CI", "day CI")]
for lab, hm in (("H1", h1), ("H2", ~h1)):
    out += [line(f"{lab} eff72 ≤ {CHOP}", FL[(feff <= CHOP) & hm], unit="day"), line(f"{lab} eff72 > {CHOP}", FL[(feff > CHOP) & hm], unit="day")]
def bar(lab, g, unit):
    """The observe bar on FRESH fills: N, days (and chop episodes), WR vs the pinned breakeven, 95 % day-bootstrap, concentration."""
    if not len(g):
        return [f"- {lab}: 0 fresh fills"]
    d = g.assign(day=g.opened_at.astype(str).str[:10])
    v = d.groupby("day").pct.mean().values
    lo, hi = (np.percentile(np.random.default_rng(5).choice(v, (10000, len(v))).mean(axis=1), [2.5, 97.5]) if len(v) > 2 else (np.nan, np.nan))
    days = sorted(pd.to_datetime(d.day.unique()))
    ep = 0; last = None
    for t in days:
        if last is None or (t - last) >= pd.Timedelta(hours=72):
            ep += 1
        last = t
    loss = d[d.pct < 0]
    dshare = (loss.groupby("day").pct.sum().min() / loss.pct.sum()) if len(loss) else 0
    pshare = (loss.groupby("pair").pct.sum().min() / loss.pct.sum()) if len(loss) else 0
    meets = len(g) >= 15 and len(v) >= 8 and (unit != "chop" or ep >= 4)
    verdict = ("BLOCK CANDIDATE" if meets and (g.pct > 0).mean() * 100 < BREAKEVEN and hi < 0 and dshare < 0.5 and pshare < 0.5
               else "CLOSE (bar not met)" if meets else "keep counting")
    return [f"- {lab}: {len(g)} fills · {len(v)} days{f' · {ep} episodes' if unit == 'chop' else ''} · WR {(g.pct > 0).mean() * 100:.0f}% "
            f"(breakeven {BREAKEVEN:.0f}%) · avg {g.pct.mean():+.3f} · 95% day CI [{lo:+.2f}, {hi:+.2f}] · max day share {dshare:.0%} · "
            f"max pair share {pshare:.0%} → **{verdict}**"]


FRESH_ML = ML[pd.to_datetime(ML.opened_at, format="ISO8601") > SINCE]
fe = FRESH_ML["eff72"]                                    # live stamp, else the validated rebuild
out += ["", f"## FRESH-only observe bars (fills after {SINCE:%Y-%m-%d %H:%M} UTC; full-size, non-MANUAL; eff72 = live stamp, else the validated rebuild)", ""]
out += [f"- eff72 source on fresh fills: {int((FRESH_ML.eff72_src == 'live').sum())} live · {int((FRESH_ML.eff72_src == 'rebuilt').sum())} "
        f"rebuilt · {int((FRESH_ML.eff72_src == 'none').sum())} none (excluded) · {VALID}"]
out += bar("burst crowding", FRESH_ML[FRESH_ML.burst], "burst")
out += bar(f"BTC chop eff72 ≤ {CHOP}", FRESH_ML[fe <= CHOP], "chop")
_sl = n(FRESH_ML, "entry_btc_1h_slope"); fs = _sl < 0; fsok = _sl.notna()
_c = lambda g: f"{len(g)} fills · {g.opened_at.astype(str).str[:10].nunique()} days · WR {(g.pct > 0).mean() * 100:.0f}% · avg {g.pct.mean():+.3f}" if len(g) else "0 fills"
_A = FRESH_ML[(fe <= CHOP) & FRESH_ML.burst2]
_Aw = (_A.assign(o2=pd.to_datetime(_A.opened_at, format="ISO8601")).sort_values("o2").o2.diff().dt.total_seconds().fillna(1e9) > BURST_S).sum() if len(_A) else 0
_tl = FRESH_ML[(fe <= CHOP) & (FRESH_ML.pct < 0)].pct.sum()
_share = (_A[_A.pct < 0].pct.sum() / _tl) if _tl < 0 else np.nan
out += [f"- sub-line A (pre-declared: the 2nd+ fill of a burst in chop is the WORST cell): chop ∧ 2nd+-burst → {_c(_A)} · {_Aw} windows · "
        f"chop ∧ not-2nd → {_c(FRESH_ML[(fe <= CHOP) & ~FRESH_ML.burst2])} · its share of the chop tier's loss (1× pct): "
        + (f"{_share:.0%}" if _share == _share else "n/a") + " — narrow-rule bar: chop bar passes ∧ cell N ≥ 6 on ≥ 3 windows ∧ share ≥ 50 % ∧ no window ≥ 50 % of the cell loss",
        f"- sub-line B (comparison only: middle ∧ slope<0 expected ≤ break-even while middle ∧ slope≥0 stays strong): middle ∧ slope<0 → "
        f"{_c(FRESH_ML[(fe > CHOP) & (fe <= 0.026) & fs])} · middle ∧ slope≥0 → {_c(FRESH_ML[(fe > CHOP) & (fe <= 0.026) & ~fs & fsok])}"]
out += [f"- comparison (no bar): {CHOP} < eff72 ≤ 0.026 → {_c(FRESH_ML[(fe > CHOP) & (fe <= 0.026)])} · eff72 > 0.026 → {_c(FRESH_ML[fe > 0.026])}"]
out += [f"- note: {x}" for x in NOTES]
OUT = os.path.join(ROOT, "reports", "ML_REGIME_OBSERVE_READ_2026-10-01.md")
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out)); print(f"\n→ {OUT}")
