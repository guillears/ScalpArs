"""🌊 Gate 57 / watch 57i — bull-run EXIT parameter sweep on REAL 1-minute paths.

Answers "should the BE-arm / ATR-trail move?" by re-simulating the live BR exit stack over the
actual 1m candles of every fill the CURRENT entry stack would take, at a grid of arm and trail
values, split by door (GREEN vs REARM).

WHY THIS SCRIPT EXISTS (2026-09-21, DECISION_LOG 94): the same questions were answered twice from
post-exit CHECKPOINT stamps and both times the checkpoint CF lied — the Sep-20 fade-exit widening
showed +$3,084 on checkpoints and −$213..−$468 on real paths. Method rule: checkpoint stamps may
NOMINATE an exit candidate; only 1m re-simulation may PRICE one. This script is the pricing
instrument, and it CALIBRATES itself first (sim at the live settings vs what actually happened) —
if the calibration block is not tight, no number below it may be quoted.

Usage:
    venv/bin/python scripts/bullrun_exit_sweep.py [--batch reports/BASELINE<n>_....csv] [--refresh]

    --batch   append a live batch export (the era column is taken as B<next>; repeatable)
    --refresh ignore the kline cache for the fills in --batch (use after a batch is still running)

Cache: reports/backtest_cache/bullrun_exit_1m.pkl, keyed (pair, opened_at) — klines for a closed
fill never change, so the cache is append-only and safe to keep.
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import requests

POOL = "reports/MASTER_POOL_stacked.csv"
CACHE = "reports/backtest_cache/bullrun_exit_1m.pkl"
V2_FLOOR = "2026-08-21T19:16"          # gate-57 v2 cohort floor (locked)
SLEEVE_BLACKLIST = ("ONGUSDT", "ETHUSDT")

# live BR exit stack (config.py: bullrun_be_arm_pct / _be_lock_pct / _trail_atr_mult / _ladder)
LIVE_ARM, LIVE_TRAIL, LOCK = 1.0, 2.0, 0.2
LADDER = [(4, 3.5), (5, 4.5), (6, 5.5), (8, 7.0), (10, 9.0), (12, 11.0)]
FEE = 0.09                              # round-trip taker toll in pnl% terms (replay-harness parity)
ARMS = [0.6, 0.7, 0.85, 1.0, 1.2, 1.5]
TRAILS = [1.0, 1.25, 1.5, 2.0, 2.5]
HORIZON = 360                           # minutes of path to simulate


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce")


def load_cohort(batches):
    frames = [pd.read_csv(POOL, low_memory=False)]
    for i, b in enumerate(batches):
        d = pd.read_csv(b, low_memory=False)
        d["stack_pnl"] = d["pnl"]
        d["era"] = f"LIVE{i + 1}"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.status == "CLOSED"].drop_duplicates(subset=["opened_at", "pair", "direction"])
    br = df[(df.entry_strategy == "BULLRUN_LONG") & (df.opened_at >= V2_FLOOR)].copy()
    br["pnl_"] = _num(br, "stack_pnl").fillna(_num(br, "pnl"))
    rearm = br["entry_br_door"].eq("REARM") | br["era"].eq("B4")   # B4 = the Aug-25 REARM window
    # the CURRENT entry stack: sleeve blacklist · r72 10/8 on the GREEN door · BTC ≥ EMA13
    keep = (~br["pair"].isin(SLEEVE_BLACKLIST)
            & ((_num(br, "entry_br_r72").fillna(99) >= 10) | rearm)
            & (_num(br, "entry_btc_dist_from_ema13_pct").fillna(1) >= 0))
    cc = br[keep].copy()
    cc["door"] = np.where(rearm[keep], "REARM", "GREEN")
    return cc


def klines(pair, opened_at, cache, refresh=False):
    key = (pair, str(opened_at))
    if key in cache and not refresh:
        return cache[key]
    t0 = pd.Timestamp(opened_at)
    start = int((t0.tz_localize("UTC") if t0.tzinfo is None else t0).timestamp() * 1000)
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params=dict(symbol=pair, interval="1m", startTime=start, limit=HORIZON),
                         timeout=10).json()
        time.sleep(0.12)                                   # be polite: a burst here earned an IP ban once
        cache[key] = [(float(k[2]), float(k[3])) for k in r] if isinstance(r, list) and len(r) >= 10 else None
    except Exception as e:                                  # noqa: BLE001 — a fetch miss must not kill the sweep
        print(f"  ! {pair} {opened_at}: {e}")
        cache[key] = None
    return cache[key]


def simulate(hi, lo, atr_pct, arm, trail):
    """Live BR exit stack with (arm, trail) swapped in. Returns realized pnl% net of fees.

    Conservative intra-bar ordering: the stop is tested against the bar LOW before the peak is
    updated from the bar HIGH, i.e. the adverse move is assumed to come first. Identical for every
    variant, so the RELATIVE comparison is valid even though absolute levels are pessimistic.
    """
    stop_base = max(min(-0.7, -1.5 * atr_pct), -1.2)        # SL: min(−0.7, ATR-widened), floor −1.2
    peak = 0.0
    for h, l in zip(hi, lo):
        stop = stop_base
        if peak >= arm:
            rung = max([f for t, f in LADDER if peak >= t], default=-99)
            stop = max(max(LOCK, peak - trail * atr_pct), rung)
        if l <= stop:
            return stop - FEE, peak
        peak = max(peak, h)
    return (hi[-1] + lo[-1]) / 2 - FEE, peak


def main():
    batches = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--batch"]
    refresh = "--refresh" in sys.argv
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}

    cc = load_cohort(batches)
    rows = []
    for _, r in cc.iterrows():
        kl = klines(r["pair"], r["opened_at"], cache, refresh and str(r["era"]).startswith("LIVE"))
        if not kl:
            continue
        ep, atr_pct = float(r["entry_price"]), float(r["entry_atr_pct"])
        hi = [(h / ep - 1) * 100 for h, _ in kl]
        lo = [(l / ep - 1) * 100 for _, l in kl]
        usd = float(r["investment"]) * float(r["leverage"]) / 100.0
        rec = dict(era=r["era"], door=r["door"], pair=r["pair"], atr=atr_pct,
                   actual=float(r["pnl_"]), usd_per_pct=usd)
        for a in ARMS:
            p, pk = simulate(hi, lo, atr_pct, a, LIVE_TRAIL)
            rec[f"arm{a}"] = p * usd
            if a == LIVE_ARM:
                rec["live"], rec["peak"] = p * usd, pk
        for t in TRAILS:
            p, _ = simulate(hi, lo, atr_pct, LIVE_ARM, t)
            rec[f"trail{t}"] = p * usd
        rec["pkg"] = simulate(hi, lo, atr_pct, 0.7, 1.0)[0] * usd    # the 57i REARM candidate profile
        rows.append(rec)
    pickle.dump(cache, open(CACHE, "wb"))
    t = pd.DataFrame(rows)
    if t.empty:
        sys.exit("no fills simulated — check the pool / batch paths")

    print(f"\nfills {len(t)}  (GREEN {int((t.door == 'GREEN').sum())} / REARM {int((t.door == 'REARM').sum())})"
          f"  eras {dict(t.era.value_counts())}")
    print("\n=== CALIBRATION (must be tight or nothing below is quotable) ===")
    print(f"  sim @ live arm {LIVE_ARM} / trail {LIVE_TRAIL}: ${t.live.sum():+.0f}   actual: ${t.actual.sum():+.0f}"
          f"   per-trade corr {t.actual.corr(t.live):.2f}   sign agreement {(np.sign(t.actual) == np.sign(t.live)).mean() * 100:.0f}%")

    armed = t[t.peak >= LIVE_ARM]
    for label, grid, col in (("ARM (trail fixed at live)", ARMS, "arm"), ("TRAIL (arm fixed at live)", TRAILS, "trail")):
        print(f"\n=== {label} ===")
        for v in grid:
            c = t[f"{col}{v}"]
            star = "  <- LIVE" if (col == "arm" and v == LIVE_ARM) or (col == "trail" and v == LIVE_TRAIL) else ""
            print(f"  {col} {v:<5}: all ${c.sum():+7.0f}  (Δ ${c.sum() - t.live.sum():+7.0f})"
                  f"   armed-only ${armed[f'{col}{v}'].sum():+7.0f}{star}")

    print("\n=== DOOR SPLIT (Δ vs live) — the 57i axis ===")
    for door, g in t.groupby("door"):
        arm_d = " ".join(f"{a}:{g[f'arm{a}'].sum() - g.live.sum():+.0f}" for a in ARMS if a != LIVE_ARM)
        tr_d = " ".join(f"{v}:{g[f'trail{v}'].sum() - g.live.sum():+.0f}" for v in TRAILS if v != LIVE_TRAIL)
        print(f"  {door} N={len(g):2d} base ${g.live.sum():+7.0f}\n     arm   {arm_d}\n     trail {tr_d}")

    print("\n=== 57i CANDIDATE PACKAGE (REARM: arm 0.7 + trail 1.0; GREEN untouched) ===")
    green = t[t.door == "GREEN"].live.sum()
    rearm_t = t[t.door == "REARM"]
    print(f"  live            total ${t.live.sum():+.0f}")
    print(f"  package         total ${green + rearm_t.pkg.sum():+.0f}   Δ ${green + rearm_t.pkg.sum() - t.live.sum():+.0f}")
    print("  by window (REARM fills only — the ship bar counts WINDOWS, not fills):")
    for e, g in rearm_t.groupby("era"):
        print(f"    {e}: N={len(g)} live ${g.live.sum():+7.0f} -> package ${g.pkg.sum():+7.0f}  Δ ${g.pkg.sum() - g.live.sum():+7.0f}")
    print("\n  🔒 57i SHIP BAR: ≥2 more REARM windows AND ≥10 armed REARM fills AND direction-consistent.")
    print(f"     armed REARM fills so far: {int((rearm_t.peak >= LIVE_ARM).sum())}   REARM windows so far: {rearm_t.era.nunique()}")


if __name__ == "__main__":
    main()
