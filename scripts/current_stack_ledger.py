"""📌 Per-era ledger under the FULL current stack — the pinned table in CLAUDE_CURRENT_STATE.

WHY THIS SCRIPT EXISTS (2026-09-21, DECISION_LOG 100): the pinned ledger was hand-rolled four
times and corrected four times, EVERY time for the same reason — a gate shipped after the ledger
was built, so the hand-written predicate list was stale (FRESHBREAK ruler artifact → 龙虾
blacklist → B6 band cohort → the breadth floor). The gate list now lives in ONE place, here, and
reads the live blacklist straight from config so it cannot drift.

    venv/bin/python scripts/current_stack_ledger.py [--no-rearm-trail]

Population: stack_keep ∧ non-probe ∧ pair_blacklist ∧ gate-51 bands (momentum longs) ∧ the full
bull-run gate list ∧ fade-cap reprice ∧ live-passed boundary fills restored. DCR compounds each
era's net on a flat $3,000 over its ACTIVE days (days with fills), so it is a per-trading-day rate.
"""
import sys

import os

import numpy as np
import pandas as pd

# run-as-script fix (2026-09-21): sys.path[0] is scripts/ when invoked as
# `python scripts/current_stack_ledger.py`, so `import config` failed and the blacklist
# silently fell back to the PRE-ship list — the ledger then showed 33 B10 fills instead of 28.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POOL = "reports/MASTER_POOL_stacked.csv"
EQ = 3000.0
ERAS = ["BASE", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]
# fills that PASSED the live gate but the stack ruler blocks on stamped boundary values
# (ADA-weakcap divergence class — live governs for an as-lived ledger)
RESTORE = [("QTUMUSDT", "2026-09-19T00:00:51"), ("API3USDT", "2026-09-19T06:23:54")]
FADE_CAP_REPRICE = {("SANDUSDT", "2026-09-19T00:04:12"): -185.0}   # gate 50b cap restored


def _n(d, c):
    return pd.to_numeric(d[c], errors="coerce")


def _blacklist(key, fallback):
    try:
        import config
        return {p.strip().upper() for p in
                str(getattr(config.trading_config.thresholds, key, "") or fallback).split(",") if p.strip()}
    except Exception as e:                                        # noqa: BLE001
        # LOUD, never silent: a quiet fallback here scores a different cohort than the bot trades
        print(f"  ⚠ could not read live '{key}' ({e}) — falling back to '{fallback}'. NUMBERS BELOW ARE SUSPECT.")
        return {p.strip().upper() for p in fallback.split(",") if p.strip()}


def build(rearm_trail=True):
    df = pd.read_csv(POOL, low_memory=False)
    df = df[df.status == "CLOSED"].copy()
    keep = (df["stack_keep"].astype(str).fillna("nan").str.lower().isin(["true", "1"])
            & ~df["is_probe"].astype(str).fillna("nan").str.lower().isin(["true", "1"]))
    d = df[keep].copy()
    d["pnl_"] = _n(d, "stack_pnl").fillna(_n(d, "pnl"))

    d = d[~d.pair.str.upper().isin(_blacklist("pair_blacklist", ""))]              # global blacklist

    # gate 51 bands — momentum LONGs only
    ml = (d.entry_strategy == "MOMENTUM") & (d.direction == "LONG")
    rsi, adx, r72 = _n(d, "entry_btc_rsi"), _n(d, "entry_btc_adx"), _n(d, "entry_btc_r72_pct")
    band = ((adx < 18) | (adx > 40) | ((rsi >= 50) & (rsi < 55))
            | ((rsi >= 55) & (rsi < 60) & ~((adx >= 20) & (adx <= 25)))
            | ((rsi >= 70) & ((r72 >= 5) | r72.isna())))
    d = d[~(ml & band.fillna(False))]

    # bull-run sleeve — the FULL live gate list
    br = d.entry_strategy == "BULLRUN_LONG"
    rearm = d["entry_br_door"].eq("REARM") | ((d.era == "B4") & br)
    bull, bear = _n(d, "entry_bull_pct"), _n(d, "entry_bear_pct")
    ok = (~br) | ((d.opened_at >= "2026-08-21T19:16")
                  & ~d.pair.str.upper().isin(_blacklist("bullrun_pair_blacklist", "ONGUSDT,ETHUSDT"))
                  & (_n(d, "entry_btc_dist_from_ema13_pct").fillna(1) >= 0)
                  & (_n(d, "entry_btc_1h_slope").fillna(1) > 0)
                  & ((_n(d, "entry_br_eff").fillna(1) >= 0.095) | rearm)
                  & ((_n(d, "entry_br_off24h").fillna(0) >= -2.0) | rearm)
                  & ((_n(d, "entry_br_r72").fillna(99) >= 10) | rearm)
                  & (bull >= 40)
                  & ~(rearm & (bull.fillna(0) <= 0) & (bear.fillna(0) <= 0))
                  & (_n(d, "entry_slippage_pct").abs().fillna(0) < 0.3)
                  & (_n(d, "entry_pair_volume_ratio").fillna(0) <= 1.2))
    d = d[ok].copy()

    for (pair, ts), v in FADE_CAP_REPRICE.items():
        d.loc[(d.pair == pair) & (d.opened_at == ts), "pnl_"] = v
    hive = (d.pair == "HIVEUSDT") & (d.era == "B7") & (d.pnl_ > 100)
    d.loc[hive, "pnl_"] = 48.0

    # restore live-passed boundary fills the ruler blocks
    raw = pd.read_csv(POOL, low_memory=False)
    for pair, ts in RESTORE:
        if not ((d.pair == pair) & (d.opened_at == ts)).any():
            r = raw[(raw.pair == pair) & (raw.opened_at == ts)]
            if len(r):
                r = r.copy(); r["pnl_"] = _n(r, "pnl")   # raw: stack_pnl is 0 for ruler-blocked rows
                d = pd.concat([d, r], ignore_index=True)

    # 57i: REARM-door fills exit on the 1.0× trail — priced per fill by bullrun_exit_sweep
    d["delta"] = 0.0
    if rearm_trail:
        try:
            deltas = _rearm_deltas(d)
            d["delta"] = [deltas.get((r.pair, r.opened_at), 0.0) for r in d.itertuples()]
        except Exception as e:                                        # noqa: BLE001
            print(f"  ! REARM trail deltas unavailable ({e}) — showing live-trail numbers")
    d["net"] = d.pnl_ + d.delta
    return d


def _rearm_deltas(d):
    """Per-fill Δ of the REARM 1.0× trail vs the 2.0× trail, from cached 1m paths."""
    import pickle
    sys.path.insert(0, "scripts")
    import bullrun_exit_sweep as B
    cache = pickle.load(open(B.CACHE, "rb"))
    out = {}
    sub = d[(d.entry_strategy == "BULLRUN_LONG") & (d["entry_br_door"] == "REARM")]
    for r in sub.itertuples():
        kl = cache.get((r.pair, str(r.opened_at)))
        if not kl:
            continue
        ep, atr = float(r.entry_price), float(r.entry_atr_pct)
        hi = [(h / ep - 1) * 100 for h, _ in kl]
        lo = [(l / ep - 1) * 100 for _, l in kl]
        usd = float(r.investment) * float(r.leverage) / 100.0
        out[(r.pair, r.opened_at)] = (B.simulate(hi, lo, atr, trail=1.0)[0]
                                      - B.simulate(hi, lo, atr, trail=2.0)[0]) * usd
    return out


def main():
    d = build(rearm_trail="--no-rearm-trail" not in sys.argv)
    print(f"\n{'Era':5} {'Dates':22} {'N':>4} {'WR':>5} {'Net $':>10} {'Days':>5} {'DCR/day':>9}")
    tot_n = tot_w = 0
    tot_net = 0.0
    for e in ERAS:
        g = d[d.era == e]
        if not len(g):
            continue
        days = max(1, g.opened_at.str[:10].nunique())
        net = g.net.sum()
        dcr = ((EQ + net) / EQ) ** (1 / days) - 1
        dates = f"{g.opened_at.min()[:10]} → {g.opened_at.max()[:10]}"
        print(f"{e:5} {dates:22} {len(g):>4} {(g.net > 0).mean() * 100:>4.0f}% {net:>+10,.0f} {days:>5} {dcr * 100:>+8.2f}%")
        tot_n += len(g); tot_w += int((g.net > 0).sum()); tot_net += net
    days = d.opened_at.str[:10].nunique()
    print(f"{'TOTAL':5} {d.opened_at.min()[:10] + ' → ' + d.opened_at.max()[:10]:22} "
          f"{tot_n:>4} {tot_w / tot_n * 100:>4.0f}% {tot_net:>+10,.0f} {days:>5} "
          f"{(((EQ + tot_net) / EQ) ** (1 / days) - 1) * 100:>+8.2f}%")
    print("\n⚠ in-sample: eras whose rules postdate them are partly self-graded (30-50% haircut).")
    print("⚠ these rows count ALL SLEEVES — never compare a sleeve-level N against them.")


if __name__ == "__main__":
    main()
