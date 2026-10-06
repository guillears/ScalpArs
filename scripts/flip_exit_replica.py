#!/usr/bin/env python3
"""FAN flip-SHORT exit replica — pure functions over a price path (READ-ONLY; never talks to the bot, never changes config).

WHAT IT REPRODUCES (the live exit of a FLIP:FAN_RATIO_GATE short, `exitm = "strpk"`):
  · ATR-widened hard stop:   sl % = −max(0.70, min(sl_atr_multiplier × ATR %, −sl_atr_widen_floor_pct)) = −max(0.70, min(1.5·ATR, 1.20))
                             (ATR % = Wilder ATR(14) of the pair's CLOSED 5m bars at the signal, as stamped in entry_atr_pct).
  · SHORT runner trail:      armed once the running peak ≥ runner_trail_short_arm_peak (0.40; 0.005 tolerance);
                             give-back = runner_trail_short_atr_mult × ATR (0.5·ATR), capped at runner_trail_short_giveback_frac × peak
                             (0.35·peak, the Aug-14 cap); floor = peak − give-back; a NEGATIVE floor is ridden (no exit) — the
                             negative-floor ride.
  · HARD_TP ladder (short):  hard_tp_ladder_short "1.0:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80" — once peak ≥ a trigger, the lock is
                             trigger − offset (the highest such lock); exit when P&L ≤ the lock.
  · Fees:                    P&L % = ((E − p) / E − entry_fee − taker · p / E) · 100 (entry fee: taker 0.045 % or maker 0.018 %; exit taker).
  · The first print that crosses a line closes there AT that print (real per-tick exit path, like the live @trade WebSocket).

VALIDATION (reports/FLIP_OVERNIGHT_FOLLOWUPS_2026-10-06.md §2a): with each era's settings, live entry, real aggTrades ticks (every
trade, no sampling) and a 6 h horizon, it reproduced all 49 live FAN flips to a mean |Δ| of 0.019 pp (98 % within 0.10; the one outlier,
BICO 08-12, closed on a negative-floor trail before the negative-floor ride existed) and all 411 yr5 replay flips to 0.013 pp (99 %
within 0.05). The flip exit fires in the per-tick path, so the validated mode is EVERY tick; `sample_1hz` (what a 1 Hz monitor sees) is
kept for sensitivity reads only — the follow-up found tick vs 1 Hz explains none of the live/replay gap. No 6 h path in either
validation set ran out without an exit (the 'OPEN_END' branch is a safety net).

SETTINGS: `live_settings()` reads the CURRENT values from trading_config.json (config.py defaults as fallback) and reports any field that
differs from the frozen 'NOW' era (the Aug-14 → today stack the replica was validated on). At 2026-10-06 they are identical:
arm 0.40 · 0.5 × ATR · cap 0.35 × peak · short ladder as above · stop 1.5 × ATR in [0.70, 1.20].

Usage:  from flip_exit_replica import live_settings, simulate
        cfg = live_settings()["cfg"]
        res = simulate(t, p, entry, atr_pct, entry_fee_rate=TAKER, **cfg)     # → dict(pnl, reason, i, t, peak, trough) or None
        venv/bin/python scripts/flip_exit_replica.py --selftest
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAKER, MAKER = 0.00045, 0.00018              # fee RATES (fractions)
HORIZON_MS = 6 * 3_600_000                   # the validated horizon (no validated flip ran past it). NOT modelled: the 180-min NO_EXPANSION close (never bound on the 49 live flips, max hold 84 min; could bind on blocked signals)
SL_BASE = 0.70                               # the flip's base stop (confidence-level stop_loss −0.70) — widened, never tightened
LADDER_NOW = [(1.0, 0.25), (1.5, 0.30), (2.0, 0.40), (3.0, 0.60), (4.0, 0.80)]

# exit-config eras (git history of trading_config.json) — the 'NOW' era is the stack in force since 2026-08-14 13:35 UTC
ERAS = {
    "E1_jun17_aug5":  dict(arm=0.45, n=0.5, frac=0.0, ladder=None, negride=False),
    "E1b_jul22_aug5": dict(arm=0.45, n=0.5, frac=0.0, ladder=LADDER_NOW, negride=False),
    "E2_aug5_aug14":  dict(arm=0.40, n=0.5, frac=0.0, ladder=LADDER_NOW, negride=True),
    "NOW":            dict(arm=0.40, n=0.5, frac=0.35, ladder=LADDER_NOW, negride=True),
}
SL_NOW = dict(sl_mult=1.5, sl_cap=1.20)


def parse_ladder(s):
    """'1.0:0.25,1.5:0.30' → [(1.0, 0.25), (1.5, 0.30)] (sorted by trigger); '' / None → None."""
    if not s:
        return None
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            a, b = part.split(":")
            out.append((float(a), float(b)))
    return sorted(out) or None


def sl_pct(atr, sl_mult=1.5, sl_cap=1.20, base=SL_BASE):
    """the ATR-widened stop as a NEGATIVE P&L % (fees included in the P&L it is compared with). atr None/NaN → the base stop."""
    try:
        a = float(atr)
    except (TypeError, ValueError):
        a = float("nan")
    if not np.isfinite(a) or sl_mult <= 0:
        return -base
    return -max(base, min(sl_mult * a, sl_cap))


def sample_1hz(t, p):
    """last price of each second (what a 1 Hz monitor sees), stamped at the second's end. Sensitivity reads only."""
    t = np.asarray(t, np.int64)
    p = np.asarray(p, float)
    if len(t) == 0:
        return t, p
    s = t // 1000
    last = np.r_[s[1:] != s[:-1], True]
    return (s[last] + 1) * 1000 - 1, p[last]


def simulate(t, p, entry, atr, entry_fee_rate=TAKER, arm=0.40, n=0.5, frac=0.35, ladder=LADDER_NOW, negride=True,
             taker=TAKER, sl=None, sl_mult=1.5, sl_cap=1.20):
    """one FAN flip short from the first print of the path. t (ms), p (prices) from the fill onward; entry = fill price; atr = ATR %.
    → dict(pnl %, reason SL | TRAIL | LADDER | OPEN_END, i, t, peak, trough) · None on an empty path. Order per print = the live
    order: stop, then the runner trail, then the HARD_TP ladder."""
    t = np.asarray(t, np.int64)
    p = np.asarray(p, float)
    if len(t) == 0 or not np.isfinite(entry) or entry <= 0:
        return None
    if sl is None:
        sl = sl_pct(atr, sl_mult, sl_cap)
    try:
        a = float(atr)
    except (TypeError, ValueError):
        a = float("nan")
    pnl = ((entry - p) / entry - entry_fee_rate - taker * p / entry) * 100
    peak = trough = 0.0
    for i in range(len(p)):
        x = float(pnl[i])
        if x > peak:
            peak = x
        if x < trough:
            trough = x
        if x <= sl:
            return dict(pnl=x, reason="SL", i=i, t=int(t[i]), peak=peak, trough=trough)
        if peak >= arm - 0.005 and np.isfinite(a):
            gb = n * a
            if frac > 0 and frac * peak < gb:
                gb = frac * peak
            fl = peak - gb
            if not (fl < 0 and negride) and x <= fl:
                return dict(pnl=x, reason="TRAIL", i=i, t=int(t[i]), peak=peak, trough=trough)
        if ladder and peak >= ladder[0][0]:
            f = max(tr - o for tr, o in ladder if peak >= tr)
            if x <= f:
                return dict(pnl=x, reason="LADDER", i=i, t=int(t[i]), peak=peak, trough=trough)
    return dict(pnl=float(pnl[-1]), reason="OPEN_END", i=len(p) - 1, t=int(t[-1]), peak=peak, trough=trough)


def _cfg_get(cfg, key):
    stack = [cfg]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            if key in d:
                return d[key]
            stack.extend(v for v in d.values() if isinstance(v, dict))
    return None


def live_settings(path=None):
    """the CURRENT live flip-short exit from trading_config.json (config.py ThresholdConfig defaults where a key is missing).
    → dict(cfg=simulate kwargs, diffs=[human-readable differences vs the validated 'NOW' era], src=where each value came from)."""
    fp = path or os.path.join(ROOT, "trading_config.json")
    try:
        cfg = json.load(open(fp))
    except Exception:
        cfg = {}
    dflt = {}
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        import config as _c                                   # noqa: E402 — read-only defaults
        for cls in vars(_c).values():                          # pydantic models: defaults live in model_fields
            mf = getattr(cls, "model_fields", None) if isinstance(cls, type) else None
            if isinstance(mf, dict) and "runner_trail_short_arm_peak" in mf:
                dflt.update({k: f.default for k, f in mf.items()})
    except Exception:
        dflt = {}

    def g(k, fb):
        v = _cfg_get(cfg, k)
        if v is None:
            v = dflt.get(k, fb)
        return v

    now = ERAS["NOW"]
    on = bool(g("runner_trail_short_enabled", True))
    use_atr = bool(g("runner_trail_short_use_atr", True))
    out = dict(arm=float(g("runner_trail_short_arm_peak", now["arm"])),
               n=float(g("runner_trail_short_atr_mult", now["n"])) if use_atr else 0.0,
               frac=float(g("runner_trail_short_giveback_frac", now["frac"]) or 0.0),
               ladder=(parse_ladder(g("hard_tp_ladder_short", "")) if bool(g("hard_tp_enabled", True)) else None),
               negride=now["negride"],
               sl_mult=float(g("sl_atr_multiplier", SL_NOW["sl_mult"]) or 0.0),
               sl_cap=abs(float(g("sl_atr_widen_floor_pct", -SL_NOW["sl_cap"]) or SL_NOW["sl_cap"])))
    diffs = []
    if not on:
        diffs.append("runner_trail_short_enabled is OFF (the replica still trails — re-validate before use)")
    if not use_atr:
        diffs.append("runner_trail_short_use_atr is OFF (give-back would not be ATR-based — re-validate)")
    if not bool(g("flip_fan_runner_strpk", True)):
        diffs.append("flip_fan_runner_strpk is OFF (FAN flips would not use the short runner trail — re-validate)")
    for k, ref in (("arm", now["arm"]), ("n", now["n"]), ("frac", now["frac"]), ("sl_mult", SL_NOW["sl_mult"]), ("sl_cap", SL_NOW["sl_cap"])):
        if abs(out[k] - ref) > 1e-9:
            diffs.append(f"{k} live {out[k]:g} ≠ validated {ref:g}")
    if out["ladder"] != now["ladder"]:
        diffs.append(f"short ladder live {out['ladder']} ≠ validated {now['ladder']}")
    return dict(cfg=out, diffs=diffs)


def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1

    chk(abs(sl_pct(0.3) + 0.70) < 1e-12 and abs(sl_pct(0.6) + 0.90) < 1e-12 and abs(sl_pct(2.0) + 1.20) < 1e-12, "stop: floor/ATR/cap")
    chk(abs(sl_pct(None) + 0.70) < 1e-12, "stop: unknown ATR → base")
    chk(parse_ladder("1.0:0.25, 1.5:0.30") == [(1.0, 0.25), (1.5, 0.30)] and parse_ladder("") is None, "ladder parse")
    E = 100.0
    t = np.arange(10, dtype=np.int64) * 1000
    # straight to the stop: price +0.7 % → short P&L ≈ −0.79 ≤ the −0.75 stop (ATR 0.5)
    r = simulate(t[:3], np.array([100.0, 100.7, 101.0]), E, 0.5, TAKER)
    chk(r["reason"] == "SL" and r["i"] == 1, "stop hit at −0.75 on the 100.7 print")
    # runner: peak +0.80 (price 99.15), ATR 1.0 → gb = min(0.5, 0.35·0.8 = 0.28) → floor ≈ 0.52
    px = np.array([100.0, 99.5, 99.15, 99.3, 99.45])
    r = simulate(t[:5], px, E, 1.0, TAKER)
    chk(r["reason"] == "TRAIL" and r["i"] == 4 and 0.4 < r["pnl"] < 0.52, "trail: capped give-back floor")
    # negative floor ride: arm at +0.40 with ATR 2.0 and no cap → floor −0.6 → ridden (no trail exit), then the stop
    r = simulate(t[:4], np.array([100.0, 99.55, 100.3, 101.5]), E, 2.0, TAKER, frac=0.0)
    chk(r["reason"] == "SL", "negative floor is ridden, the stop still applies")
    # ladder: peak +1.6 → lock 1.5 − 0.30 = 1.20
    r = simulate(t[:4], np.array([100.0, 98.3, 98.7, 98.85]), E, 10.0, TAKER, frac=0.0)
    chk(r["reason"] == "LADDER" and r["i"] == 3, "ladder lock at 1.2 after a +1.6 peak")
    chk(simulate(t[:2], np.array([100.0, 99.99]), E, 0.5)["reason"] == "OPEN_END", "open end")
    tt, pp = sample_1hz(np.array([100, 900, 1100, 2500]), np.array([1.0, 2.0, 3.0, 4.0]))
    chk(list(pp) == [2.0, 3.0, 4.0] and list(tt) == [999, 1999, 2999], "1 Hz sampler keeps each second's last print")
    s = live_settings()
    chk(set(s["cfg"]) == {"arm", "n", "frac", "ladder", "negride", "sl_mult", "sl_cap"}, "live settings keys")
    print(f"flip_exit_replica selftest OK — {ok} checks · live vs validated 'NOW': {s['diffs'] or 'identical'}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(json.dumps(live_settings(), indent=1, default=str))
