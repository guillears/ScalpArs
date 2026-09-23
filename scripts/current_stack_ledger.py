"""📌 Per-era ledger under the FULL current stack — the pinned table in CLAUDE_CURRENT_STATE.

WHY THIS SCRIPT EXISTS (2026-09-21, DECISION_LOG 100): the pinned ledger was hand-rolled four
times and corrected four times, EVERY time for the same reason — a gate shipped after the ledger
was built, so the hand-written predicate list was stale (FRESHBREAK ruler artifact → 龙虾
blacklist → B6 band cohort → the breadth floor). The gate list now lives in ONE place, here, and
reads the live blacklist straight from config so it cannot drift.

    venv/bin/python scripts/current_stack_ledger.py [--batch <live batch csv>] [--no-age-cap] [--no-rearm-trail]

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
ERAS = ["BASE", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"] + [f"B{n}" for n in range(11, 30)]
# fills that PASSED the live gate but the stack ruler blocks on stamped boundary values
# (ADA-weakcap divergence class — live governs for an as-lived ledger)
RESTORE = [("QTUMUSDT", "2026-09-19T00:00:51"), ("API3USDT", "2026-09-19T06:23:54")]
FADE_CAP_REPRICE = {("SANDUSDT", "2026-09-19T00:04:12"): -185.0}   # gate 50b cap restored

# 🕐 57l REARM entry-age cap. The door clock is NOT in the trade data for historical fills —
# `entry_br_door_age_min` only stamps from the 2026-09-21 ship onward — so the episode boundaries
# below are transcribed from the EB log bundle (`web.stdout.log`, [BULLRUN_MONITOR] state
# transitions). They are the ONLY record: the logs rotate, and monitor_periods is wiped by a paper
# reset.
# ⚠ SCOPE OF THE "verified" CLAIM (deep review): every REARM fill that SURVIVES THE GATE LIST above
# falls inside one of these. The pool ALSO holds 7 B4 REARM fills (2026-08-25 02:28→03:53) with NO
# covering episode — they are invisible here only because the bull-run gate drops them first, so the
# fail-open warning never fires for them. Do not read this list as covering the whole pool.
# ⚠ #44's END IS AN ASSUMPTION, NOT A LOG LINE — the door was still open at the log dump and this
# constant assumes it never re-armed for 15h. That single assumption produces the 346-398 min ages
# that delete all 10 B11 fills and the 187-222 min ages that delete 5 B10 fills, i.e. it carries the
# ENTIRE headline swing. The list also skips #41/#43 (states between these REARM stretches), so the
# transcription is selective by design, not exhaustive.
# ⚠ FRAGILE BY SECONDS: ENAUSDT 2026-09-20T17:13:14 lands at 60.33 min — a 21-second error in #40's
# transcribed start flips that fill in or out of the ledger.
REARM_EPISODES = [
    ("#40", "2026-09-20T16:12:54", "2026-09-20T18:17:11"),
    ("#42", "2026-09-21T00:42:59", "2026-09-21T01:36:55"),
    ("#44", "2026-09-21T08:42:50", "2026-09-22T00:00:00"),   # still open at the log dump
]


# 🔌 sleeve MASTER TOGGLES — a "full current stack" ledger must not contain fills the live stack
# would refuse to take at all. Operator-caught 2026-09-21: SPIKE_CHASE (dormant since the Aug-10
# tripwire — CORRECTION per deep review: the real kill is 2026-08-21 via `spike_chase_enabled`;
# Aug-10 was the accidental probe-flag retirement, reverted the same day) and SPIKE_BOUNCE
# ("OFF FOR GOOD 2026-08-10 PM") were still in the table. This is the
# SIXTH time the ledger went stale, and the same root cause every time — something shipped after
# the gate list was written. Entry filters were checked; the on/off switch never was.
# Each sleeve maps to ALL the flags that must be ON — the spike species have TWO switches, and
# `spike_chase_probe_enabled` killing every fade AND chase at once is exactly what happened on
# Aug-10. Deep review: mapping only the per-species flag reproduces that failure.
SLEEVE_TOGGLE = {
    "SPIKE_CHASE":   ["spike_chase_probe_enabled", "spike_chase_enabled"],
    "SPIKE_BOUNCE":  ["spike_bounce_enabled"],
    "SPIKE_FADE":    ["spike_chase_probe_enabled", "spike_fade_enabled"],
    "BULLRUN_LONG":  ["bullrun_sleeve_enabled"],
    "BEARRUN_SHORT": ["bearrun_sleeve_enabled"],
    "FLIP":          ["flip_entry_enabled"],
    "BOUNCE_LONG":   ["bounce_long_enabled"],     # OFF today — the engine can still stamp the label
    "BULL_LONG":     ["bull_long_enabled"],       # OFF today
}


def _validate_toggles():
    """A flag name that no longer exists returns True with NO exception and NO warning — the bug
    back, silently. Validate the map against the real field list at import (deep review)."""
    try:
        import config
        fields = set(getattr(config.trading_config.thresholds, "model_fields", {}) or {})
        if not fields:
            return
        unknown = sorted({k for ks in SLEEVE_TOGGLE.values() for k in ks} - fields)
        if unknown:
            print(f"  ⚠ SLEEVE_TOGGLE names not found on SignalThresholds: {unknown} — those sleeves "
                  f"are treated as ON. NUMBERS BELOW ARE SUSPECT.")
    except Exception:                                             # noqa: BLE001
        pass


def _sleeve_on(strategy):
    """True when EVERY master toggle for the sleeve is ON (or it has none, e.g. MOMENTUM)."""
    # engine labels can be namespaced (`FLIP:FAN_RATIO_GATE`) — key on the family
    keys = SLEEVE_TOGGLE.get(str(strategy)) or SLEEVE_TOGGLE.get(str(strategy).split(":")[0])
    if not keys:
        return True
    try:
        import config
        th = config.trading_config.thresholds
        for key in keys:
            val = getattr(th, key, None)
            if val is not None and not bool(val):
                return False
        return True
    except Exception as e:                                        # noqa: BLE001
        print(f"  ⚠ could not read {keys} ({e}) — KEEPING {strategy} fills. NUMBERS SUSPECT.")
        return True


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


def _rearm_age_min(opened_at):
    """Minutes into its door episode from the TRANSCRIBED boundaries, or None when none covers it."""
    t = pd.Timestamp(opened_at)
    for _lab, a, b in REARM_EPISODES:
        if pd.Timestamp(a) <= t < pd.Timestamp(b):
            return (t - pd.Timestamp(a)).total_seconds() / 60.0
    return None


def build(rearm_trail=True, entry_age_cap=60.0):
    frames = [pd.read_csv(POOL, low_memory=False)]
    _batches = []
    for i, a in enumerate(sys.argv):
        if a == "--batch":
            if i + 1 >= len(sys.argv):
                sys.exit("usage: --batch <live batch csv>  (no path given)")
            _batches.append(sys.argv[i + 1])
    for i, b in enumerate(_batches):
        _b = pd.read_csv(b, low_memory=False)
        _b["stack_keep"] = True; _b["is_probe"] = False; _b["stack_pnl"] = _b["pnl"]
        _b["era"] = f"B{11 + i}"        # each --batch is its OWN era; they used to collide into B11
        frames.append(_b)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.status == "CLOSED"]
    _n0 = len(df)
    # keep="last": a --batch export is FRESHER than the pool, so a corrected row must win
    df = df.drop_duplicates(subset=["opened_at", "pair", "direction"], keep="last").copy()
    if len(df) != _n0:
        print(f"  ℹ dedup dropped {_n0 - len(df)} duplicate row(s) on (opened_at, pair, direction)")
    keep = (df["stack_keep"].astype(str).fillna("nan").str.lower().isin(["true", "1"])
            & ~df["is_probe"].astype(str).fillna("nan").str.lower().isin(["true", "1"]))
    d = df[keep].copy()
    d["pnl_"] = _n(d, "stack_pnl").fillna(_n(d, "pnl"))

    d = d[~d.pair.str.upper().isin(_blacklist("pair_blacklist", ""))]              # global blacklist

    # drop fills from sleeves that are switched OFF today
    _validate_toggles()
    _off = sorted({s_ for s_ in d.entry_strategy.astype(str).unique() if not _sleeve_on(s_)})
    if _off:
        _drop = d.entry_strategy.astype(str).isin(_off)
        print(f"  ℹ sleeve(s) OFF, excluded: {', '.join(_off)} "
              f"({int(_drop.sum())} fills, ${d.loc[_drop, 'pnl_'].sum():+,.0f})")
        d = d[~_drop]

    # gate 51 bands — momentum LONGs only
    ml = (d.entry_strategy == "MOMENTUM") & (d.direction == "LONG")
    rsi, adx, r72 = _n(d, "entry_btc_rsi"), _n(d, "entry_btc_adx"), _n(d, "entry_btc_r72_pct")
    band = ((adx < 18) | (adx > 40) | ((rsi >= 50) & (rsi < 55))
            | ((rsi >= 55) & (rsi < 60) & ~((adx >= 20) & (adx <= 25)))
            | ((rsi >= 70) & ((r72 >= 5) | r72.isna())))
    d = d[~(ml & band.fillna(False))]

    # 🏦 Sep-23 mega-cap exclusion — momentum LONGs only, RAW eligible-universe rank (entry_pair_rank), same
    # pure rule the engine gate calls. Reads the live threshold so the ledger cannot go stale a 7th time.
    try:
        from services.trading_engine import long_megacap_block
        import config as _cfg
        _th = _cfg.trading_config.thresholds
        if int(float(getattr(_th, 'long_megacap_rank_max', 0) or 0)) != 10:
            print(f"  ⚠ live long_megacap_rank_max={getattr(_th, 'long_megacap_rank_max', 0)} but build_master_pool.py freezes 10 — "
                  "stacked pool and this ledger DISAGREE until the builder constant is updated and the pool rebuilt.")
        # element-wise on the column (not row-apply): an empty frame yields an empty Series, never a DataFrame
        mega = _n(d, "entry_pair_rank").map(lambda v: long_megacap_block(_th, v)).astype(bool)
        _mega_drop = ml.reindex(d.index).fillna(False) & mega
        if int(_mega_drop.sum()):
            print(f"  ℹ mega-cap exclusion (rank <= {getattr(_th, 'long_megacap_rank_max', 0)}): "
                  f"{int(_mega_drop.sum())} momentum-long fill(s) dropped, ${d.loc[_mega_drop, 'pnl_'].sum():+,.0f}")
        d = d[~_mega_drop]
    except Exception as e:                                        # noqa: BLE001
        print(f"  ⚠ could not apply the mega-cap exclusion ({e}) — NUMBERS BELOW INCLUDE rank<=10 momentum longs.")

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

    # 🕐 57l: drop REARM-door fills taken past the entry-age cap. GREEN/null-door fills are never
    # touched (the decay evidence is REARM-only). A REARM fill with NO known episode is KEPT and
    # reported LOUDLY — an unknown clock must never silently delete rows from the ledger.
    if entry_age_cap and entry_age_cap > 0:
        _r = (d.entry_strategy == "BULLRUN_LONG") & d["entry_br_door"].eq("REARM")
        if _r.any():
            # 🕐 PREFER THE STAMPED CLOCK. From the 2026-09-21 ship every fill carries the engine's
            # own `entry_br_door_age_min`; without this the ledger would keep scoring B12+ off the
            # hand-transcribed constant forever, silently (deep review, DECISION_LOG 106).
            _age = d.loc[_r, "opened_at"].map(_rearm_age_min)
            if "entry_br_door_age_min" in d.columns:
                _stamped = pd.to_numeric(d.loc[_r, "entry_br_door_age_min"], errors="coerce")
                _both = _stamped.notna() & _age.notna()
                _bad = _both & ((_stamped - _age).abs() > 1.0)
                if _bad.any():
                    print(f"  ⚠ {int(_bad.sum())} fill(s): the STAMPED door age disagrees with the "
                          f"transcribed episode by >1 min — REARM_EPISODES is wrong, fix it. "
                          f"Using the stamped value.")
                _age = _stamped.where(_stamped.notna(), _age)
            _unknown = int(_age.isna().sum())
            if _unknown:
                print(f"  ⚠ {_unknown} REARM fill(s) have no known door episode — KEPT (fail-open). "
                      f"Add the episode to REARM_EPISODES or the 57l column is understated.")
            _drop = _age.notna() & (_age > float(entry_age_cap))
            d = d.drop(index=_age.index[_drop])

    for (pair, ts), v in FADE_CAP_REPRICE.items():
        d.loc[(d.pair == pair) & (d.opened_at == ts), "pnl_"] = v
    hive = (d.pair == "HIVEUSDT") & (d.era == "B7") & (d.pnl_ > 100)
    d.loc[hive, "pnl_"] = 48.0

    # restore live-passed boundary fills the ruler blocks
    # ⚠ these are concatenated AFTER the gate chain, so they bypass the blacklist, the gate-51
    #   bands, the SLEEVE TOGGLES and the 57l entry-age cap. Harmless today (both are SPIKE_FADE,
    #   which is ON, and never REARM) but
    #   a REARM row added here would silently dodge 57l. Deep review, DECISION_LOG 106.
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
            # LOUD + typed: this used to swallow anything, so a batch CSV missing `closed_at`
            # would silently revert the WHOLE table to live-trail numbers.
            print(f"  ⚠ REARM trail deltas unavailable ({type(e).__name__}: {e}) — the trail\n"
                  f"    counterfactual is NOT applied; every era below is on live-trail numbers.")
    d["net"] = d.pnl_ + d.delta
    return d


def _rearm_deltas(d):
    """Per-fill Δ of the REARM 1.0× trail vs the 2.0× trail, from cached 1m paths."""
    import pickle
    sys.path.insert(0, "scripts")
    import bullrun_exit_sweep as B
    cache = pickle.load(open(B.CACHE, "rb"))
    out = {}
    # ⚠ ONLY fills that closed BEFORE the 57i ship get a counterfactual. A fill that already
    # EXITED on the 1.0× trail has lived it — re-pricing 1.0×-vs-2.0× on top double-counts the
    # change and understates the era. (B11 read −$933 against an actual −$643 before this guard;
    # same class as the sweep's `_trail_as_lived` fix, DECISION_LOG 103.)
    sub = d[(d.entry_strategy == "BULLRUN_LONG") & (d["entry_br_door"] == "REARM")
            & (d.closed_at.astype(str) < B.REARM_TRAIL_SHIP_UTC)]
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
    d = build(rearm_trail="--no-rearm-trail" not in sys.argv,
              entry_age_cap=(0.0 if "--no-age-cap" in sys.argv else 60.0))
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
