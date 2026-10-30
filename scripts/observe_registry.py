#!/usr/bin/env python3
"""🧭 OBSERVATION-LINE EXIT-ERA REGISTRY (DECISION_LOG 406, operator 2026-10-28). Alert-only; changes nothing.

Every scripts/scout_*.py (+ the scout's forward reads) is classified here — a test fails on an unclassified scout file, so a new line
cannot ship without deciding:
  EXIT   — the line prices / walks fills under specific exit values; `walks` = the trading_config.json thresholds keys it effectively
           assumes and the value it assumes (e.g. a WILLY line walking the flat +1.25 exit assumes frenzy_willy_high_target_enabled
           False; a FRENZY_LONG walk at frenzy_stop_pct assumes frenzy_long_stop_pct 3.0). When the live config differs → ⚠ STALE:
           the line measures against an exit that no longer runs → the SAME DAY the exit changes: re-base it (fresh count) or retire
           it (operator decides). Its verdict is not read while STALE.
  PROXY  — prices a designed proxy exit that was never the live engine exit (Bear-Run baskets) — shown, never STALE.
  BOOKED — booked P&L only (exit-agnostic). `era` = the line still drops fills by exit era (registered on an old exit) → STALE too.
  NONE   — no trade outcome (infra, coverage, tooling).
Classification source: the 2026-10-28 audit of all 53 scout files (DECISION_LOG 406). Update an entry whenever its line is re-based.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "trading_config.json")

_OLD_WILLY = {"frenzy_willy_high_target_enabled": False}          # the flat +1.25 / −4 / 90 exit (pre-V2b, DL 356 / 358)
_FZ_STOP3 = {"frenzy_long_stop_pct": 3.0}                         # FRENZY_LONG walked at frenzy_stop_pct (live −4 since DL 317)
_NO_LITE_C = {"frenzy_lite_c_enabled": False}                     # LITE walked at +3 for every fill (live LITE C +2 out of zone, DL 301)

EXIT = {
    "scout_willy_timecap.py": dict(dl="264 · 278 · 281 · 285 · 290 · 297", sleeves="FRENZY_WILLY", walks=_OLD_WILLY),
    "scout_willy_tp125.py": dict(dl="265", sleeves="FRENZY_WILLY", walks=_OLD_WILLY),
    "scout_willy_stop.py": dict(dl="268 · 269", sleeves="FRENZY_WILLY", walks=_OLD_WILLY),
    # 🔁 (409) re-based 2026-10-29: every fill walked under today's V2b exit (the old +1.25 / −4 / 90 ruler printed as context only)
    "scout_willy_btc1h.py": dict(dl="275 · 279 · 409", sleeves="FRENZY_WILLY", walks={"frenzy_willy_high_target_enabled": True,
                                 "frenzy_willy_high_k_pct": 1.0, "frenzy_willy_high_tp_max_pct": 8.0, "frenzy_willy_stop_pct": 4.0,
                                 "frenzy_willy_late_tp_minutes": 30.0, "frenzy_willy_late_tp_pct": 0.25, "frenzy_willy_max_hold_minutes": 90}),
    "scout_willy_mktdip.py": dict(dl="305", sleeves="FRENZY_WILLY", walks=_OLD_WILLY),
    "scout_willy_avgdown.py": dict(dl="313", sleeves="FRENZY_WILLY", walks=_OLD_WILLY),
    "scout_willy_hold.py": dict(dl="251", sleeves="FRENZY_WILLY · FRENZY family", walks={**_OLD_WILLY, **_FZ_STOP3, **_NO_LITE_C}),
    "scout_willy_hold_frenzy.py": dict(dl="266", sleeves="FRENZY_LONG / WIDE / LITE", walks={**_FZ_STOP3, **_NO_LITE_C}),
    "scout_willy_exit_v2.py": dict(dl="356 · 357 · 358", sleeves="FRENZY_WILLY · HT", walks={"frenzy_willy_high_target_enabled": True,
                                   "frenzy_willy_high_k_pct": 1.0, "frenzy_willy_high_tp_max_pct": 8.0, "frenzy_willy_stop_pct": 4.0,
                                   "frenzy_willy_late_tp_minutes": 30.0, "frenzy_willy_late_tp_pct": 0.25}),
    "scout_willy_latejoin.py": dict(dl="366", sleeves="FRENZY_WILLY", walks={"frenzy_willy_high_target_enabled": True,
                                    "frenzy_willy_stop_pct": 4.0, "frenzy_willy_late_tp_minutes": 30.0, "frenzy_willy_late_tp_pct": 0.25}),
    # 🔁 (408) re-based the day the exit changed: the shadows stay priced at the frozen DL 348 rule (+8 / −15) and every comparison against
    # traded fills reads them under the SAME exit (real_by_exit_fn); the FBX switch-back gate walks both. The exit-shadow reads (DL 367 / 370
    # / 390) and OBS-FB1 stay FROZEN to the +8 / −15 rule by design (labelled in their lines; parity is era-aware) — not stale, not the live exit.
    "scout_flatbase_short.py": dict(dl="348 · 367 · 370 · 390 · 407 · 408", sleeves="FLATBASE_SHORT (+ blocked WILLY leg)",
                                    walks={"flatbase_tp_pct": 5.0, "flatbase_stop_pct": 10.0, "flatbase_max_hold_hours": 72.0, **_OLD_WILLY}),
    # 🚪 (411) exit shadows on FLIPS only (fades / momentum answered by history, operator 10-29) — the walks use each fill's LIVE stop
    # (−0.7 widened to 1.5 × entry ATR, capped −1.2) and Δ reads the BOOKED flip exit (FAN flips' short runner trail, flip_fan_runner_strpk):
    # a change to any of these keys makes the line STALE (the −0.7 base is a confidence-tier stop_loss outside `thresholds` — not pinned)
    "scout_exit_shadows.py": dict(dl="411", sleeves="FLIP", walks={
                                  "sl_atr_multiplier": 1.5, "sl_atr_widen_floor_pct": -1.2, "flip_fan_runner_strpk": True,
                                  "runner_trail_short_enabled": True, "runner_trail_short_arm_peak": 0.4, "runner_trail_short_k": 0.5,
                                  "runner_trail_short_use_atr": True, "runner_trail_short_atr_mult": 0.5,
                                  "runner_trail_short_giveback_frac": 0.35, "flip_runner_strpk_shorts": True}),
    "scout_frenzy.py": dict(dl="195 · 196 · 205", sleeves="FRENZY_LONG / WIDE", walks=_FZ_STOP3),
    "scout_frenzy_exits.py": dict(dl="214 · 243 · 250 · 251 · 263 · 272 · 284 · 333 · 336", sleeves="FRENZY family + WILLY gate",
                                  walks={**_FZ_STOP3, **_NO_LITE_C, **_OLD_WILLY}),
    "scout_frenzy_tp_revert.py": dict(dl="283 · 284", sleeves="FRENZY_LONG / WIDE", walks=_FZ_STOP3),
    "scout_frenzy_willy_exit.py": dict(dl="267", sleeves="FRENZY_LONG / WIDE / LITE", walks={"frenzy_willy_tp_pct": 1.0, **_FZ_STOP3}),
    "scout_frenzy_v2b_shadow.py": dict(dl="379", sleeves="FRENZY_LITE · LONG", walks={"frenzy_long_stop_pct": 4.0, "frenzy_stop_pct": 3.0,
                                       "frenzy_lite_c_enabled": True, "frenzy_lite_c_tp_pct": 2.0, "frenzy_tp_pct": 1.5}),
    "scout_lite_c.py": dict(dl="301", sleeves="FRENZY_LITE", walks={"frenzy_lite_c_enabled": True, "frenzy_lite_c_tp_pct": 2.0,
                                                                    "frenzy_lite_tp_pct": 3.0, "frenzy_stop_pct": 3.0}),
    "scout_lite_green_retry.py": dict(dl="315 · 316 · 355", sleeves="FRENZY_LITE · RETRY", walks={"frenzy_lite_c_tp_pct": 2.0,
                                      "frenzy_lite_tp_pct": 3.0, "frenzy_stop_pct": 3.0}),
    "scout_lite_stretch_reentry.py": dict(dl="272 · 284", sleeves="FRENZY_LITE", walks=_NO_LITE_C),
    "scout_long_stop4.py": dict(dl="315 · 317 · 355", sleeves="FRENZY_LONG", walks={"frenzy_tp_pct": 1.5, "frenzy_long_stop_pct": 4.0}),
    "scout_revert_gates.py": dict(dl="110 … 252 (TP3_VS_LOCK · BEARISH_BLOCKED stale; ML / flip / MS replicas live)",
                                  sleeves="FRENZY family · ML · flips · MS", walks={**_FZ_STOP3, **_NO_LITE_C}),
    "scout_stop_slip.py": dict(dl="244", sleeves="every stopped sleeve", walks=_FZ_STOP3),
    "scout_heat_regime.py": dict(dl="208", sleeves="MOMENTUM", walks={"ema13_cross_exit_long_enabled": False}),
    "scout_ml_matched_refusals.py": dict(dl="380 · 383", sleeves="MOMENTUM", walks={"ema13_cross_exit_long_enabled": False}),
    "scout_surge_obs.py": dict(dl="202 · 203", sleeves="SURGE_LONG", walks={"surge_long_exit_no_lock": True}),
    "study_surge_exit_grid.py": dict(dl="397", sleeves="SURGE_LONG", walks={"surge_long_exit_no_lock": True}),
}
PROXY = {"scout_bearrun_v2.py": "359", "scout_bearrun_r24.py": "369"}
BOOKED = {   # booked P&L — `era` = drops fills by WILLY exit era (registered on the old +1.25 exit)
    "scout_b1h_negflank.py": None, "scout_fade_late_weak.py": None, "scout_fade_watch.py": None,
    # review M5: booked WR judged against TODAY's breakeven over FRENZY fills spanning exit eras (LONG −3 → −4, LITE pre-C)
    "scout_frenzy_entry_lines.py": {**_FZ_STOP3}, "scout_frenzy_wl_lines.py": {**_FZ_STOP3, **_NO_LITE_C}, "scout_lite_late_btcfall.py": None, "scout_lite_ob_thin.py": None, "scout_lite_off30.py": None,
    "scout_lite_z3.py": None, "scout_long_resize.py": None, "scout_ml_cooldown.py": None, "scout_ml_regime_lines.py": None,
    "scout_ml_trend_lines.py": None, "scout_ml_w4_watch.py": None,
    # (413) FADE_NOSTRETCH judges booked WR against the frozen breakeven of today's fade exit (72.1 %) → pinned to its keys
    "scout_fade_nostretch_watch.py": {"hard_tp_enabled": True, "runner_trail_short_enabled": True, "runner_trail_short_use_atr": True, "runner_trail_short_be_ratchet_enabled": False,
                                      "runner_trail_short_be_lock_pct": 0.1, "runner_trail_short_atr_min": 0.0, "spike_lock_exempt_fade": True, "spike_fade_sl_pct": -1.5, "runner_trail_short_arm_peak": 0.4, "runner_trail_short_atr_mult": 0.5,
                                      "runner_trail_short_giveback_frac": 0.35, "spike_fade_late_arm_after_min": 15.0, "spike_fade_late_arm_peak": 0.3,
                                      "hard_tp_ladder_short": "1.0:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80"},
    # (412) WILLY_ADX55 judges booked WR against the frozen V2b breakeven (69.0 %) → pinned to the V2b exit keys
    "scout_willy_adx_watch.py": {"frenzy_willy_high_target_enabled": True, "frenzy_willy_stop_pct": 4.0, "frenzy_willy_high_k_pct": 1.0,
                                 "frenzy_willy_high_tp_max_pct": 8.0, "frenzy_willy_max_hold_minutes": 90,
                                 "frenzy_willy_late_tp_minutes": 30.0, "frenzy_willy_late_tp_pct": 0.25}, "scout_ml_turn.py": None, "scout_rebound.py": None, "scout_willy_hold_off.py": None,
    "scout_willy_ht.py": _OLD_WILLY, "scout_willy_nomiss.py": _OLD_WILLY,
}
NONE = {"scout_features.py", "scout_features_parity.py", "scout_gatekit.py", "scout_gvol.py", "scout_staircase.py",
        "scout_stamp_coverage.py", "scout_sweep_all.py", "scout_path_decomposition.py"}


def thresholds(path=CONFIG):
    with open(path) as f:
        c = json.load(f)
    return c.get("thresholds", c)


def diffs(walks, th):
    """→ [(key, assumed, live)] where the line's assumption differs from the live config. Pure."""
    out = []
    for k, v in (walks or {}).items():
        if k not in th:
            out.append((k, v, "MISSING from config"))
            continue
        live = th.get(k)
        try:
            same = (live == v) if isinstance(v, bool) or isinstance(live, bool) else (abs(float(live) - float(v)) < 1e-9)
        except (TypeError, ValueError):
            same = str(live) == str(v)
        if not same:
            out.append((k, v, live))
    return out


def status(th=None):
    """→ rows (script, kind, dl, sleeves, state, diffs) for every registered line. Pure on th."""
    th = th if th is not None else thresholds()
    rows = []
    for s, e in sorted(EXIT.items()):
        d = diffs(e["walks"], th)
        rows.append((s, "EXIT", e["dl"], e["sleeves"], "STALE" if d else "OK", d))
    for s, dl in sorted(PROXY.items()):
        rows.append((s, "PROXY", dl, "BEARRUN_SHORT", "PROXY", []))
    for s, era in sorted(BOOKED.items()):
        d = diffs(era, th) if era else []
        rows.append((s, "BOOKED", "", "", "STALE" if d else "OK", d))
    return rows


def classified():
    return set(EXIT) | set(PROXY) | set(BOOKED) | NONE


def run(now_ms=None, th=None):
    rows = status(th)
    stale = [r for r in rows if r[4] == "STALE"]
    L = ["## 🧭 OBSERVATION LINES vs the LIVE EXIT (DECISION_LOG 406, alert-only)", "",
         f"{len(rows) + len(NONE)} scout files classified ({sum(r[1] == 'EXIT' for r in rows)} exit-priced · {len(PROXY)} proxy · "
         f"{len(BOOKED)} booked · {len(NONE)} no outcome). A line whose assumed exit differs from today's config is ⚠ STALE: its "
         "verdict measures an exit that no longer runs → re-base it (fresh count) or retire it — operator decides; never read a STALE "
         "line's verdict.", ""]
    if not stale:
        return L + ["✅ every exit-priced line matches today's exits.", ""]
    L += ["| Line | Sleeves | DL | Assumes → live |", "|---|---|---|---|"]
    for s, kind, dl, sl, st, d in stale:
        L.append(f"| ⚠ {s} ({kind}) | {sl or '–'} | {dl or '–'} | " + " · ".join(f"{k} {a} → {b}" for k, a, b in d) + " |")
    return L + [""]


if __name__ == "__main__":
    print("\n".join(run()))
