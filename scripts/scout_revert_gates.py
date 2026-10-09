#!/usr/bin/env python3
"""⏳ Scout — pre-committed REVERT / ARM gate tracker (operator, 2026-10-04: "the hourly scout tracks every open pre-committed revert
gate automatically and flags the moment one fires").

READ-ONLY. Public Binance market data (1m / 5m klines over REST, aggTrades daily archives via scripts/backtest_fetch_ticks.py) + the
operator's exports in ~/Downloads (orders: scalpars_orders_paper_*.csv, decisions journal: scalpars_decisions_paper_*.csv) +
reports/MASTER_POOL_stacked.csv. Never talks to the bot, never changes config. Called by scripts/opportunity_scout.py every run
(own try/except there; every gate here has its own try/except too). State: reports/SCOUT_REVERT_GATES.json (atomic writes).

GATES (frozen definitions — quoted from CLAUDE_CURRENT_STATE.md / DECISION_LOG; never re-tuned here):
  CHOP_BURST (201) momentum LONG refused (BLOCK LONG_CHOP_BURST): first 6 refused signals re-priced → WR ≥ 50 % ∨ Σ > 0 → FIRES
                   (long_chop_burst_block_enabled false).
  FRENZY_LOCK (205, SUPERSEDED Oct-8 by TP3_VS_LOCK — the lock is off; kept running for the record) first 20 FRENZY_LONG + FRENZY_WIDE fills after the lock-then-trail deploy: live exit vs the old fixed +3/−3 re-priced on
                   ticks (bot accounting) — +3/−3 averages better → FIRES (frenzy_lock_arm_pct 0); +6/−3 and +4/−3 shown for the record.
  ── Oct-8 (DECISION_LOG 250, operator declared overrides; fills / refusals counted from the deploy = push + 10 min, key FRENZY_OCT8) ──
  ATR_RAISE (250)  frenzy_max_atr_pct 2.5 → 3.0 (vs reports/FRENZY_ATR_CAP_STUDY_2026-10-08.md, which said keep 2.5): the first 15 FRENZY_LONG /
                   FRENZY_WIDE fills with entry_atr_pct in (2.5, 3.0] (= what the raise re-admitted; a closed prefix, live pnl %) average < 0 →
                   FIRES ("REVERT (operator decision): frenzy_max_atr_pct back to 2.5"). Contrast: the ≤ 2.5 fills of the same period. Blocking
                   reasons: journal FRENZY_ATR_HIGH / FRENZY_WIDE_ATR_HIGH refusals (ATR > 3.0 or unreadable) per UTC day — COUNT ONLY (not priced).
  TP3_VS_LOCK (250) the exit is back on the fixed +3 / −3 (frenzy_lock_arm_pct 0): the first 20 FRENZY_LONG / WIDE / LITE fills from the
                   deploy, both exits re-priced on ticks with the bot's accounting (price_fixed: fixed +3/−3 and the lock −3 → +2 at +3 → peak − 2,
                   12 h) — paired Δ per fill; Σ lock − Σ fixed > +3 % points (the runners the lock keeps outweigh the +1 it gives back on every
                   +3 touch) → FIRES ("REVIEW (operator decision): consider the lock again"). Decided on the SAME accounting both sides (the
                   205 review rule); the live result is the fidelity line. Mirror of FRENZY_LOCK (205), which it SUPERSEDES.
  BEARISH_BLOCKED (250) frenzy_bearish_day_block: reads scripts/scout_frenzy_exits.py's FRENZY_BEARISH_BLOCKED store (refusals priced as if
                   opened: first print ≥ close + 8 s, fixed +3/−3, 0.09 % fees + 0.10 % slip, 12 h; one per pair-episode; DAY units): at ≥ 15
                   counted signals on ≥ 8 days, blocked mean > 0 → FIRES ("REVERT (operator decision): turn frenzy_bearish_day_block off").
  FRENZY_TP3 (199, RETIRED Oct-5 — superseded by FRENZY_LOCK) first 20 FRENZY_LONG + FRENZY_WIDE fills opened after the +3 deploy re-priced with fixed +4/−3 on ticks (bot accounting:
                   net levels, 0.09 % fees, fill at the crossing print, 12 h cap) → +4/−3 beats the actual average → FIRES (frenzy_tp_pct 4).
  FRENZY_STRONG (197) first 10 sized-up FRENZY_LONG fills (entry_frenzy_adx_delta > 0 ∧ entry_frenzy_di_spread > 0) average below the
                   other FRENZY_LONG fills of the same period, or below 0 → FIRES (frenzy_long_lev_mult_strong 0).
  FRENZY_GVOL (194) first 20 FRENZY + WIDE fills under the market-volume gate average < 0 → FIRES (frenzy_gvol_max 0).
  SURGE_LONG (202) option B at full size (operator override): the first 15 SURGE_LONG triggers that FILLED after the deploy (a closed
                   prefix; one trigger = the mean of its fills) mean ≤ 0 → FIRES (surge_long_lev_mult 0.05). Supersedes the 200 probe gate.
  BEARRUN (200)    windows (fills ≤ 180 min apart = one window) started after 2026-10-04 22:00 UTC: ≥ 5 windows, ≥ 3 positive ∧ Σ > 0 →
                   ARM bar met (bearrun_lev_mult 1.0) — a positive event, not a revert. From 252 (2026-10-08) its fills are at 5× (lev 0.25).
  ── Oct-8 (DECISION_LOG 252, sizing; fills counted from the deploy = push + 10 min, key SIZING_252) ──
  BEARRUN_5X (252) bearrun_lev_mult 0.05 → 0.25 (operator declared override at 1 live window): ALL BEARRUN_SHORT fills chained into windows
                   exactly like the (200) row (fills ≤ 180 min apart); counted = windows from the deploy whose fills are all at 5 ≤ leverage < 20;
                   FROZEN at the first 3 complete windows (all fills closed and the window can no longer grow: a later fill exists or the newest
                   export is > 180 min past its last fill): Σ of the window-mean pnl % < 0 → FIRES
                   ("ROLLBACK: bearrun_lev_mult 0.05"); else holds. The (200) gate still decides full size.
  FAN_10X (252)    FAN flips 20× → 10× (flip_entry_sources FAN_RATIO_GATE:1.0:0.5): FLIP:FAN_RATIO_GATE fills from the deploy at leverage ≤ 10
                   (closed prefix, FROZEN at 15): WR ≥ 63 % ∧ avg pnl % ≥ +0.20 → FIRES ("RESTORE 20×"); avg < 0 → REVIEW (sleeve-kill checklist
                   first, no auto-off); else stays 10×. pnl % is leverage-invariant.
  FADE_BRSI50 (112) spike_fade_max_btc_rsi 45 → 50 (Sep-24): fresh SPIKE_FADE fills opened from the deploy (commit c0d7fed + 10 min) with
                   entry_btc_rsi in (45, 50] (the band the raise added — the engine blocks iff the BTC 5m RSI incl. the forming candle is strictly
                   > the ceiling; the stamp is that reading; DECISION_LOG wrote it "[45,50)"), same-minute fires counted once (one minute = the mean
                   pnl % of its fills), CLOSED prefix by open time, FROZEN at the 10th counted fire: WR < 55 % ∨ Σ pnl % < 0 → FIRES
                   ("REVERT: spike_fade_max_btc_rsi back to 45"); else holds.
  LOADX (126)      first 30 (extended from 8 on 2026-10-04, operator) PAIR_RSI_MOMENTUM_LOADX-blocked LONG signals (journal FAILS lines whose COMPLETE fail set is LOADX alone,
                   rank ≤ 10 pairs excluded = mega-cap gate), WINDOW units (one 5-min journal bucket = one scan = one window, value =
                   mean) → WR ≥ 60 % ∨ net > 0 → long_rsi_momentum_adx_max 0.
  FLIP_EMA13_BLOCKED / FLIP_PADX_BLOCKED (221) FAN flip-short entry filters judged on the signals they BLOCK (not the trades they kept):
                   journal FAILS lines (dir SHORT, src FLIP:FAN_RATIO_GATE) whose COMPLETE fail set is FLIP_FAN_BTC_EMA13 alone / FLIP_FAN_PAIR_ADX
                   alone (sole blockers), one signal per pair-episode, WINDOW units (5-min buckets — market-wide inputs), first 10 windows,
                   re-priced with the live FAN flip-short exit (scripts/flip_exit_replica.py: ATR-widened stop, short runner trail with the
                   0.35 × peak cap, HARD_TP short ladder; taker fee both sides; ATR = Wilder ATR(14) % of the closed 5m bars at the signal;
                   6 h horizon) → mean of window means > 0 → FIRED (review flip_fan_btc_ema13_max → off / flip_fan_pair_adx_min → 0); ≤ 0 → holds.
  MS_PVR_BLOCKED (226) momentum_short_pair_vol_max 0.86 stays — the Sep-18 kept-side revert (< 70 % on 15 → 1.0) is OVERRIDDEN (operator,
                   DECISION_LOG 226). Frozen gates: ① REVERT to 1.0 if Cohort A (0.86 ≤ PVR < 1.0 refusals from 10-06 18:00 UTC, journal BLOCK
                   MOMENTUM_SHORT_PAIRVOL, one per pair-episode, final prices, refusals a live fill of the pair followed ≤ 65 min excluded) reaches
                   15 signals with mean ≥ the kept side's mean over the same period (kept since 10-06 18:00; < 5 fills → since 09-18 12:00) ·
                   ② SLEEVE REVIEW (full sleeve-kill checklist, no auto-kill) if the first 20 kept fills since 09-18 12:00 (keys frozen in the state)
                   are below breakeven WR 59 %. Refusal PVR rebuilt on a 5-s grid of the bucket; re-priced with the live momentum-SHORT exit
                   replica (scripts/ms_pvr_shadow.py), entry = first print ≥ the earliest consistent refusal second + 35 s (bracket: latest + 35 s).
  HEAT (116)       first 30 (extended from 6 on 2026-10-04, operator) LONG_HEAT_BLOCK fires re-priced → WR ≥ 60 % → legs back to 0.07 / 64 / 80 (second leg — the Jan–Jun
                   engine replay failing the expectancy bar — is manual).
  MEGACAP (110)    LONG_MEGACAP_BLOCK refusals re-priced → ≥ 60 % WR ∧ Σ > 0 on N ≥ 8 across ≥ 3 windows → long_megacap_rank_max 0.

REFUSED-SIGNAL PRICING (CHOP_BURST / LOADX / HEAT / MEGACAP; the 220 flip gates use the same timing / final rule with the flip replica): journal bucket t = the 5-min START of the refusal → entries at t+1 min
(primary) and t+5 min (sensitivity); one signal per pair-episode (same pair refused again ≤ 60 min later = the same signal). Price = the
live momentum-LONG exit REPLICA of scripts/ml_exit_optimize.py (evaluate/run_fill, BASE params = today's live stack incl. recovery hold;
reproduces live exits 104/104 on ticks), taker entry fee, ATR = 14-bar ATR % of the closed 5m bars before entry, recovery-hold RSI ruler =
BTC closed-5m RSI at entry. Path = aggTrades ticks where the daily archive exists (fetched on demand), 1m klines otherwise →
'provisional' until every day of the path is on ticks (accepted as final on 1m after 4 days without an archive). A gate only FIRES on
final prices; a provisional verdict is shown as such. The journal hides the exact refusal second / price → the two timings bracket it.

Usage:  venv/bin/python scripts/scout_revert_gates.py            # print the section (and update the state)
        venv/bin/python scripts/scout_revert_gates.py --selftest # synthetic checks of every gate's fire logic
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
REPORTS = os.path.join(ROOT, "reports")
CACHE = os.path.join(REPORTS, "backtest_cache")
STATE_JSON = os.path.join(REPORTS, "SCOUT_REVERT_GATES.json")
DL = os.path.expanduser("~/Downloads")
MIN, H, DAY = 60_000, 3_600_000, 86_400_000
EPISODE_MIN = 60                     # same pair refused again ≤ 60 min later = the same signal
FINAL_AFTER_MS = 4 * DAY             # no tick archive after 4 days → the 1m price is accepted as final
TICK_RETRY_MS = 3 * H                # an archive that is not published yet is retried at most every 3 h
FETCH_TIMEOUT_S = 240                # tick-archive fetch budget per run
PRICE_BUDGET_S = 240                 # pricing budget per run (the rest waits for the next run)
FRENZY_FEES, FRENZY_HOLD = 0.09, 12 * H
PROBE_START = "2026-10-04 22:00"     # DECISION_LOG 200: SURGE_LONG / BEARRUN probe windows count from here
SURGE_R72_MAX = 2.7
# (commit, fallback UTC push time) — deploy = push + 10 min
DEPLOYS = {"FRENZY_TP3": ("2e36c26", "2026-10-04 19:23:15"),
           "FADE_BRSI50": ("c0d7fed", "2026-09-24 21:36:51"),   # 🔓 Sep-24 (112) fade BTC-RSI ceiling 45 → 50 — commit c0d7fedf 18:36:51 -03
           "FRENZY_STRONG": ("181131e", "2026-10-04 14:06:09"),
           "FRENZY_GVOL": ("0d79904", "2026-10-03 22:14:13"),
           "SURGE_B": ("grep:(DECISION_LOG 202)", "2026-10-05 01:30:00"),
           "FRENZY_LOCK": ("grep:(DECISION_LOG 205)", "2026-10-05 22:00:00"),
           "HEAT_REVERT": ("grep:(DECISION_LOG 208)", "2026-10-05 22:00:00"),
           "FRENZY_WILLY": ("grep:(DECISION_LOG 251)", None),   # 🎲 Oct-8 FRENZY_WILLY (declared exception) — its commit message must carry "(DECISION_LOG 251)"; not found → NOW
           "SIZING_252": ("grep:(DECISION_LOG 252)", None),   # 🐻🔄 Oct-8 sizing: BEARRUN 5× + FAN flips 10× (one commit; its message must carry "(DECISION_LOG 252)"); not found → NOW
           "FRENZY_OCT8": ("grep:(DECISION_LOG 250)", None)}   # 🐻⬆🎯 Oct-8: bearish-day block + ATR 3.0 + fixed TP (one commit; its message must carry "(DECISION_LOG 250)")   # 🔁 Oct-5 heat re-scope reverted   # 🎯 Oct-5 lock-then-trail exit (commit message carries the exact string)   # ⚡ Oct-4 option B (found by its commit message)
SHIPS = {"HEAT": "2026-09-25", "LOADX": "2026-09-29", "MEGACAP": "2026-09-23"}
# Oct-4 operator: "keep collecting" → trackers extended to 30; (new N, frozen N, frozen verdict) — the frozen first-N verdict stays on record
EXT_N = {"LOADX": (30, 8, "FIRED (fragile at t+5m)"), "HEAT": (30, 6, "FIRED (6/6 won)")}   # ship dates (journal coverage notes)
# 🔄 (221) FAN flip-short entry filters judged on the signals they BLOCK (gate code → the engine's _flip_filters fail name)
FLIP_SRC = "FLIP:FAN_RATIO_GATE"
FLIP_GATES = {"FLIP_EMA13_BLOCKED": "FLIP_FAN_BTC_EMA13", "FLIP_PADX_BLOCKED": "FLIP_FAN_PAIR_ADX"}
FLIP_N = 10                          # first 10 WINDOWS (5-min journal buckets) per gate
FLIP_REG_MS = 1791244800000          # 2026-10-06 00:00 UTC registration: earlier windows were seen in the 10-06 flip review (in-sample) → excluded
# 📊 (operator 2026-10-06) momentum shorts refused by the pair-volume ceiling (momentum_short_pair_vol_max 0.86), shadow-priced
MS_PVR_GATE = "MOMENTUM_SHORT_PAIRVOL"
MS_PVR_REG_MS = 1791309600000        # 2026-10-06 18:00 UTC registration floor: earlier refusals are shown as reference only
# ── Oct-8 (DECISION_LOG 250) — FROZEN bars ──
ATR_OLD, ATR_NEW, ATR_RAISE_N = 2.5, 3.0, 15          # ATR_RAISE: fills with entry_atr_pct in (2.5, 3.0]; first 15 (closed prefix) average < 0 → fires
ATR_BLOCK_GATES = {"FRENZY_ATR_HIGH": "LONG", "FRENZY_WIDE_ATR_HIGH": "WIDE"}
TP3_N, TP3_MARGIN = 20, 3.0                           # TP3_VS_LOCK: first 20 fills, Σ lock − Σ fixed > +3.0 % points → review
BB_N, BB_DAYS = 15, 8                                 # BEARISH_BLOCKED: ≥ 15 counted signals on ≥ 8 days, blocked mean > 0 → revert
BB_CSV = os.path.join(REPORTS, "SCOUT_FRENZY_BEARISH_BLOCKED.csv")
FRENZY3 = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
# ── Oct-8 (DECISION_LOG 252) — FROZEN bars ──
BR5_LEV_MIN, BR5_LEV_MAX, BR5_WINDOWS, BR_GAP_MIN = 5, 20, 3, 180   # BEARRUN_5X: windows whose fills are all at 5 ≤ lev < 20, first 3 complete, Σ window-mean pnl % < 0 → rollback
FAN10_N, FAN10_WR, FAN10_AVG, FAN10_LEV_MAX = 15, 63.0, 0.20, 10   # FAN_10X: first 15 closed at lev ≤ 10 → restore / review / stay
# ── Sep-24 (DECISION_LOG 112) — FROZEN bar: fresh SPIKE_FADE fills in the band the 45 → 50 raise added, same-minute fires once ──
FB_LO, FB_HI, FB_N, FB_WR = 45.0, 50.0, 10, 55.0       # band (45, 50] (engine blocks iff bRSI > ceiling, strict) · first 10 · WR < 55 ∨ Σ < 0 → back to 45
FB_STOP = -1.5                                          # the fade's full stop (pnl %) — caution text only, not part of the rule


def log(msg):
    print(f"[revert-gates] {msg}", file=sys.stderr, flush=True)


def _ms(ts):
    """UTC epoch ms of a string / datetime / Timestamp (naive = UTC)."""
    t = pd.Timestamp(ts)
    return int((t.tz_localize("UTC") if t.tzinfo is None else t).value // 1_000_000)


def _ms_series(s):
    t = pd.to_datetime(pd.Series(s).astype(str).str[:23].str.replace("T", " ", regex=False), errors="coerce", format="mixed")
    return ((t - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("float")


def _fmt_t(ms, full=False):
    if ms is None or (isinstance(ms, float) and not np.isfinite(ms)):
        return "–"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M" if full else "%m-%d %H:%M")


def _f(v, fmt="+.2f"):
    try:
        v = float(v)
        return format(v, fmt) if np.isfinite(v) else "–"
    except (TypeError, ValueError):
        return "–"


def atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_state():
    try:
        with open(STATE_JSON) as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:                                # a corrupt file must be visible: the frozen first-N sets live in it
        log(f"state unreadable ({e}) — starting fresh; old file kept as .bad")
        try:
            os.replace(STATE_JSON, STATE_JSON + ".bad")
        except OSError:
            pass
        return {}


def deploy_ms(name):
    """push time of the gate's commit (git) + 10 min; the pinned fallback when git is unavailable."""
    h, fb = DEPLOYS[name]
    try:
        args = ["--grep=" + h[5:], "--fixed-strings", "--reverse"] if h.startswith("grep:") else [h]   # (no --since: it hid every commit here)   # grep: the FIRST commit naming it
        out = subprocess.run(["git", "-C", ROOT, "log", "--format=%ct"] + args, capture_output=True, text=True, timeout=10)
        out.stdout = (out.stdout.strip().splitlines() or [""])[0]
        ct = int(out.stdout.strip())
        return ct * 1000 + 10 * MIN
    except Exception:
        if fb is None:
            # 🐻 (250, review): no pinned guess — until git shows the commit the floor is NOW (nothing before it can be counted; a pinned
            # time earlier than the real deploy would count lock-era / pre-raise fills as the new rules'). Logged every run it applies.
            log(f"deploy {name}: commit {h} not found in git — counting from NOW (no fill / refusal counted until the commit exists)")
            return int(time.time() * 1000)
        return _ms(fb) + 10 * MIN


def cfg_value(key):
    """current value of a config key in trading_config.json (top level or any nested dict), else None."""
    try:
        cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))
    except Exception:
        return None
    stack = [cfg]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            if key in d:
                return d[key]
            stack.extend(v for v in d.values() if isinstance(v, dict))
    return None


# ═══════════════════════════════ pure decision rules (self-tested) ═══════════════════════════════
def _wr(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return (100.0 * sum(1 for x in v if x > 0) / len(v)) if v else float("nan")


def decide_first_n(vals, n, wr_min, need_sum_pos=False, or_sum_pos=True):
    """first-N rule on re-priced refused signals (vals in order). Returns (state, wr, sum) with state 'collecting' | 'fired' | 'holds'.
    or_sum_pos: fires when WR ≥ wr_min OR Σ > 0 · need_sum_pos: fires when WR ≥ wr_min AND Σ > 0 · neither: WR alone."""
    v = list(vals)[:n]
    if len(v) < n:
        return "collecting", _wr(v), float(np.sum(v)) if v else 0.0
    wr, s = _wr(v), float(np.sum(v))
    if need_sum_pos:
        fired = wr >= wr_min and s > 0
    elif or_sum_pos:
        fired = wr >= wr_min or s > 0
    else:
        fired = wr >= wr_min
    return ("fired" if fired else "holds"), wr, s


def decide_megacap(vals, windows, n_min=8, w_min=3, wr_min=60.0):
    """cumulative re-admit bar: ≥ 60 % WR ∧ Σ > 0 on N ≥ 8 across ≥ 3 windows (fires; otherwise collecting — no frozen N)."""
    v = list(vals)
    wr, s = _wr(v), float(np.sum(v)) if v else 0.0
    if len(v) < n_min or len(set(windows)) < w_min:
        return "collecting", wr, s
    return ("fired" if (wr >= wr_min and s > 0) else "collecting"), wr, s


def decide_tp(actual, alt, n=20):   # RETIRED Oct-5 with gate_frenzy_tp (kept for the record / selftest)
    """FRENZY TP: on the first n fills, the alternative exit's average beats the actual average → fired."""
    a, b = list(actual)[:n], list(alt)[:n]
    if len(a) < n or len(b) < n:
        return "collecting", float(np.mean(a)) if a else float("nan"), float(np.mean(b)) if b else float("nan")
    ma, mb = float(np.mean(a)), float(np.mean(b))
    return ("fired" if mb > ma else "holds"), ma, mb


def decide_lock(actual, fixed3, n=20):
    """🎯 (205) FRENZY lock-then-trail revert gate: on the first n FRENZY + WIDE fills after the deploy, the old fixed +3/−3 (tick re-price)
    averages better than the live lock exit → 'fired' (frenzy_lock_arm_pct 0 → the fixed TP returns); else 'holds'."""
    a, b = list(actual)[:n], list(fixed3)[:n]
    if len(a) < n or len(b) < n:
        return "collecting", float(np.mean(a)) if a else float("nan"), float(np.mean(b)) if b else float("nan")
    ma, mb = float(np.mean(a)), float(np.mean(b))
    return ("fired" if mb > ma else "holds"), ma, mb


def decide_strong(sized, normal, n=10):
    """FRENZY strong leverage: first n sized-up fills average below the normal fills' average, or below 0 → fired."""
    s = list(sized)[:n]
    ms = float(np.mean(s)) if s else float("nan")
    mn = float(np.mean(normal)) if len(normal) else float("nan")
    if len(s) < n:
        return "collecting", ms, mn
    fired = ms < 0 or (np.isfinite(mn) and ms < mn)
    return ("fired" if fired else "holds"), ms, mn


def decide_mean_neg(vals, n=20):
    v = list(vals)[:n]
    m = float(np.mean(v)) if v else float("nan")
    if len(v) < n:
        return "collecting", m
    return ("fired" if m < 0 else "holds"), m


def decide_atr_raise(vals, n=ATR_RAISE_N):
    """⬆ (250) the first n fills the ATR raise re-admitted (entry ATR in (2.5, 3.0], closed prefix, live pnl %): mean < 0 → 'fired'."""
    v = list(vals)[:n]
    m = float(np.mean(v)) if v else float("nan")
    if len(v) < n:
        return "collecting", m
    return ("fired" if m < 0 else "holds"), m


def atr_band(v):
    """entry ATR % → 'raise' (2.5 < ATR ≤ 3.0 — re-admitted by the raise) · 'old' (≤ 2.5) · 'above' (> 3.0) · None (unreadable)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return "raise" if ATR_OLD < x <= ATR_NEW else ("old" if x <= ATR_OLD else "above")


def decide_tp3_lock(lock, fixed, n=TP3_N, margin=TP3_MARGIN):
    """🎯 (250) paired, first n fills (same order both lists): Σ lock − Σ fixed > margin % points → 'fired' (review the lock); else 'holds'.
    → (state, Σ lock − Σ fixed)."""
    a, b = list(lock)[:n], list(fixed)[:n]
    d = float(np.sum(a) - np.sum(b)) if a and len(a) == len(b) else float("nan")
    if len(a) < n or len(b) < n:
        return "collecting", d
    return ("fired" if d > margin else "holds"), d


def decide_bearish_blocked(x, days, n=BB_N, d_min=BB_DAYS):
    """🐻 (250) counted, final blocked bearish-day signals priced as if opened (x, in order) and their UTC days: ≥ n signals on ≥ d_min days →
    mean > 0 → 'fired' (revert: block off) · ≤ 0 → 'holds'; else 'collecting'. → (state, n, days, mean)."""
    v = [(float(a), d) for a, d in zip(x, days) if a is not None and np.isfinite(float(a))]
    nd = len({d for _, d in v})
    m = float(np.mean([a for a, _ in v])) if v else float("nan")
    if len(v) < n or nd < d_min:
        return "collecting", len(v), nd, m
    return ("fired" if m > 0 else "holds"), len(v), nd, m


def bb_first_crossing(rows, n=BB_N, d_min=BB_DAYS):
    """🐻 (250, deep review) the FROZEN verdict cohort: rows = [(key, day, final)] of the counted blocked signals in time order → the keys of
    the SHORTEST prefix holding ≥ n signals on ≥ d_min distinct days, all of them final (a provisional row inside the prefix holds the
    freeze back — like a first-N closed prefix); None while not reached. Decided ONCE on that set, never re-decided on a growing one."""
    days = set()
    for i, (k, d, fin) in enumerate(rows):
        if not fin:
            return None
        days.add(d)
        if i + 1 >= n and len(days) >= d_min:
            return [r[0] for r in rows[:i + 1]]
    return None


def boot_ci(x, n=5000, seed=7):
    """95 % bootstrap CI of the mean (window units) — same ruler as scripts/surge_bearrun_review.py."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan")
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def decide_surge(win, min_windows=8, r72_max=SURGE_R72_MAX):   # RETIRED Oct-5 (superseded by decide_surge_b, DECISION_LOG 202) — kept for the record
    """win = [(btc_r72, window_mean)]. Frozen rule (DECISION_LOG 200) after ≥ 8 windows with fills:
    low group (r72 ≤ +2.7) mean > 0 with CI > 0 ∧ rest ≤ 0 → 'arm_group' · low group ≤ 0 → 'fired' (LONG off) · else 'open'."""
    lo = [m for r, m in win if r is not None and np.isfinite(r) and r <= r72_max]
    hi = [m for r, m in win if r is not None and np.isfinite(r) and r > r72_max]
    info = dict(n=len(win), n_lo=len(lo), n_hi=len(hi), m_lo=float(np.mean(lo)) if lo else float("nan"),
                m_hi=float(np.mean(hi)) if hi else float("nan"), ci_lo=boot_ci(lo))
    if len(win) < min_windows:
        return "collecting", info
    if any(r is None or not np.isfinite(r) for r, _ in win):
        return "collecting", info                    # an unreadable BTC 3-day return blocks the split — never guess a group
    if not lo:
        return "open", info
    if info["m_lo"] <= 0:
        return "fired", info
    if info["m_lo"] > 0 and np.isfinite(info["ci_lo"][0]) and info["ci_lo"][0] > 0 and (not hi or info["m_hi"] <= 0):
        return "arm_group", info
    return "open", info


def decide_surge_b(trigger_means, n=15):
    """⚡ Oct-4 option B (DECISION_LOG 202) revert gate, pre-committed: the first 15 SURGE_LONG triggers that FILLED since the B deploy
    (one trigger = the mean of its fills' pnl %) — mean ≤ 0 → 'fired' (leverage back to the 0.05 probe); mean > 0 → 'holds'."""
    v = list(trigger_means)[:n]
    if len(v) < n:
        return "collecting"
    return "fired" if float(np.mean(v)) <= 0 else "holds"


def decide_bearrun(win_means, total_sum, min_windows=5, min_pos=3):
    """ARM bar (DECISION_LOG 200): ≥ 5 windows, ≥ 3 positive ∧ Σ > 0 → 'armbar'."""
    if len(win_means) < min_windows:
        return "collecting"
    return "armbar" if (sum(1 for m in win_means if m > 0) >= min_pos and total_sum > 0) else "open"


def decide_bearrun_5x(window_sums, n=BR5_WINDOWS):
    """🐻 (252) first n complete 5× windows (window-mean pnl %, in order): Σ of the window means < 0 → 'fired' (rollback to 0.05), else 'holds'."""
    v = list(window_sums)[:n]
    tot = float(np.sum(v)) if v else 0.0
    if len(v) < n:
        return "collecting", tot
    return ("fired" if tot < 0 else "holds"), tot


def decide_fan_10x(vals, n=FAN10_N, wr_min=FAN10_WR, avg_min=FAN10_AVG):
    """🔄 (252) first n closed FAN flips at 10×: WR ≥ 63 ∧ avg ≥ +0.20 → 'fired' (restore 20×) · avg < 0 → 'review' · else 'holds' (stay 10×)."""
    v = list(vals)[:n]
    wr, m = _wr(v), (float(np.mean(v)) if v else float("nan"))
    if len(v) < n:
        return "collecting", wr, m
    if wr >= wr_min and m >= avg_min:
        return "fired", wr, m
    return ("review" if m < 0 else "holds"), wr, m


def fade_brsi_band(v):
    """🔓 (112) entry BTC RSI → 'band' (45 < x ≤ 50 — what the raise re-admitted) · 'low' (≤ 45) · 'above' (> 50) · None (unreadable)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return "band" if FB_LO < x <= FB_HI else ("low" if x <= FB_LO else "above")


def fade_brsi_fires(f):
    """🔓 (112) band fills (any order) → counted fires in open order: same-minute fills = ONE fire (value = mean pnl % of its fills).
    → [dict(m=minute_ms, n, closed, val, rows=[(o_ms, pair, brsi, pnl)])]; closed = every fill of the minute CLOSED with a pnl %."""
    out = {}
    for r in f.sort_values("o_ms").itertuples():
        m = int(r.o_ms) // MIN * MIN
        g = out.setdefault(m, dict(m=m, rows=[], closed=True))
        pn = pd.to_numeric(r.pnl_percentage, errors="coerce")
        c = str(r.status) == "CLOSED" and pn is not None and np.isfinite(pn)
        g["closed"] = g["closed"] and c
        g["rows"].append((int(r.o_ms), str(r.pair), float(r.entry_btc_rsi), float(pn) if c else float("nan")))
    fires = []
    for m in sorted(out):
        g = out[m]
        g["n"] = len(g["rows"])
        g["val"] = float(np.mean([x[3] for x in g["rows"]])) if g["closed"] else float("nan")
        fires.append(g)
    return fires


def fade_brsi_prefix(fires):
    """the leading CLOSED fires (open order) — a later closed fire waits behind an earlier open one."""
    out = []
    for g in fires:
        if not g["closed"]:
            break
        out.append(g)
    return out


def decide_fade_brsi(vals, n=FB_N, wr_min=FB_WR):
    """🔓 (112) first n counted fires (closed prefix, values in open order): WR < 55 % ∨ Σ < 0 → 'fired' (back to 45); else 'holds'.
    → (state, wr, Σ)."""
    v = list(vals)[:n]
    wr, s = _wr(v), (float(np.sum(v)) if v else 0.0)
    if len(v) < n:
        return "collecting", wr, s
    return ("fired" if (wr < wr_min or s < 0) else "holds"), wr, s


def window_chain(ts_ms, gap_ms):
    """window index per (sorted) timestamp: a new window when the gap to the previous one exceeds gap_ms."""
    out, cur, prev = [], -1, None
    for t in ts_ms:
        if prev is None or t - prev > gap_ms:
            cur += 1
        out.append(cur)
        prev = t
    return out


def episodes(rows, gap_min=EPISODE_MIN):
    """first journal line of each pair-episode (same pair refused again ≤ gap_min later = the same signal). rows: DataFrame(ms, pair)."""
    rows = rows.sort_values(["ms", "pair"])
    keep, last = [], {}
    for i, r in zip(rows.index, rows.itertuples()):
        if r.pair not in last or r.ms - last[r.pair] > gap_min * MIN:
            keep.append(i)
        last[r.pair] = r.ms
    return rows.loc[keep]


# ═══════════════════════════════ data: exports ═══════════════════════════════
ORDER_COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "entry_price", "exit_price", "pnl_percentage",
              "leverage", "entry_frenzy_adx_delta", "entry_frenzy_di_spread", "entry_surge_trigger_at", "close_reason", "entry_pair_rank",
              "entry_bull_pct", "entry_btc_ema20_slope", "entry_btc_rsi_prev", "entry_btc_off30d_high_pct", "cell_multiplier_source", "entry_atr_pct",
              "entry_btc_rsi")


def _export_ms(path):
    try:
        b = os.path.basename(path).rsplit("_paper_", 1)[1][:19]
        return _ms(datetime.strptime(b, "%Y-%m-%d_%H-%M-%S"))
    except Exception:
        return int(os.path.getmtime(path) * 1000)


def load_orders():
    """bot fills: every orders export (dedupe opened_at·pair·direction, newest export wins) + the master pool (lowest priority);
    MANUAL excluded. Returns (df, newest_export_ms, n_files)."""
    fs = sorted(glob.glob(os.path.join(DL, "scalpars_orders_paper_*.csv")), key=_export_ms)
    fr = []
    pool = os.path.join(REPORTS, "MASTER_POOL_stacked.csv")
    if os.path.exists(pool):
        try:
            fr.append(pd.read_csv(pool, low_memory=False, usecols=lambda c: c in ORDER_COLS).assign(_rank=-1))
        except Exception as e:
            log(f"master pool unreadable: {e}")
    newest = None
    for i, f in enumerate(fs):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in ORDER_COLS).assign(_rank=i))
            newest = max(newest or 0, _export_ms(f))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(ORDER_COLS) + ["o_ms"]), None, 0
    A = pd.concat(fr, ignore_index=True)
    for c in ORDER_COLS:
        if c not in A:
            A[c] = np.nan
    A = A.dropna(subset=["opened_at", "pair"])
    A["_k"] = A.opened_at.astype(str).str.replace(" ", "T", regex=False).str[:19]   # pool and exports may differ in the separator
    A = A.sort_values("_rank").drop_duplicates(["_k", "pair", "direction"], keep="last")
    A = A[A.entry_strategy.astype(str) != "MANUAL"].copy()
    A["o_ms"] = _ms_series(A.opened_at).values
    A = A[np.isfinite(A.o_ms)].sort_values("o_ms").reset_index(drop=True)
    A["o_ms"] = A.o_ms.astype("int64")
    A["pnl_percentage"] = pd.to_numeric(A.pnl_percentage, errors="coerce")
    return A, newest, len(fs)


def load_journal():
    """decision-journal lines needed by the refused-signal gates + coverage. Returns (J, (t_min, t_max, n_files, gaps_h))."""
    fs = sorted(glob.glob(os.path.join(DL, "scalpars_decisions_paper_*.csv")))
    fr, beats = [], []
    for f in fs:
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("t", "e", "pair", "dir", "gate", "src"))
        except Exception:
            continue
        if not {"t", "e"} <= set(d.columns):
            continue
        for c in ("pair", "dir", "gate", "src"):
            if c not in d:
                d[c] = None
        beats.append(d.loc[d.e == "SCAN", "t"])
        g = d.gate.astype(str)
        keep = (d.dir.astype(str) == "LONG") & (((d.e == "BLOCK") & g.isin(["LONG_CHOP_BURST", "LONG_HEAT_BLOCK", "LONG_MEGACAP_BLOCK", "FRENZY_WIDE_CHOPPY"]))
                                               | ((d.e == "FAILS") & g.str.contains("PAIR_RSI_MOMENTUM_LOADX", regex=False)))
        # 📊 momentum-short pair-volume refusals (an open_position gate → BLOCK line, first gate = the only one evaluated so far)
        keep |= (d.e == "BLOCK") & (d.dir.astype(str) == "SHORT") & (g == MS_PVR_GATE)
        # ⬆ (250) ATR_RAISE blocking-reason tally: FRENZY / WIDE ATR refusals (count only)
        keep |= (d.e == "BLOCK") & g.isin(list(ATR_BLOCK_GATES))
        # 🔄 (221) FAN flip-short refusals that carry one of the two tracked gates (the sole-blocker cut happens in _signals_for)
        keep |= ((d.e == "FAILS") & (d.dir.astype(str) == "SHORT") & (d.src.astype(str) == FLIP_SRC)
                 & g.str.contains("|".join(FLIP_GATES.values()), regex=True))
        fr.append(d[keep])
    if not fr:
        return pd.DataFrame(columns=["t", "e", "pair", "dir", "gate", "src", "ms"]), None
    J = pd.concat(fr, ignore_index=True).drop_duplicates(["t", "e", "pair", "dir", "gate"])
    J["ms"] = _ms_series(J.t).values
    J = J[np.isfinite(J.ms)].copy()
    J["ms"] = J.ms.astype("int64")
    b = pd.concat(beats) if beats else pd.Series(dtype=str)
    bm = np.sort(_ms_series(b.drop_duplicates()).dropna().values.astype("int64")) if len(b) else np.array([], dtype="int64")
    if not len(bm):
        return J, None
    gaps = np.diff(bm)
    return J, (int(bm[0]), int(bm[-1]) + 5 * MIN, len(fs), int((gaps > H).sum()))


def rank_map(orders):
    """latest stamped entry_pair_rank per pair (any sleeve) — the mega-cap exclusion for the LOADX cohort."""
    r = orders.dropna(subset=["entry_pair_rank"]).sort_values("o_ms") if "entry_pair_rank" in orders else orders.iloc[0:0]
    return r.groupby("pair").entry_pair_rank.last().astype(float).to_dict() if len(r) else {}


# ═══════════════════════════════ data: market ═══════════════════════════════
_K1_MEM = {}


def _get_json(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def klines(pair, tf, t0, t1):
    """CLOSED klines [t0, t1) from Binance USDⓈ-M public REST → DataFrame(open_time, o, h, l, c). In-memory cache per run."""
    key = (pair, tf, t0, t1)
    if key in _K1_MEM:
        return _K1_MEM[key]
    step = {"1m": MIN, "5m": 5 * MIN}[tf]
    now = int(time.time() * 1000)
    rows, since = [], t0
    end = min(t1, now)
    while since < end:
        q = urllib.parse.urlencode(dict(symbol=pair, interval=tf, startTime=since, endTime=end - 1, limit=1500))
        r = _get_json(f"https://fapi.binance.com/fapi/v1/klines?{q}")
        if not r:
            break
        rows += [x for x in r if int(x[0]) + step <= now]        # forming bar dropped
        nxt = int(r[-1][0]) + step
        if nxt <= since or len(r) < 1500:
            break
        since = nxt
        time.sleep(0.05)
    d = pd.DataFrame([[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in rows],
                     columns=["open_time", "o", "h", "l", "c"]).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    _K1_MEM[key] = d
    return d


def btc5m_array(t_from):
    """BTC 5m [open_time, o, h, l, c] — reports/backtest_cache/btc_5m.csv merged with REST bars up to now (closed only)."""
    fr = []
    fp = os.path.join(CACHE, "btc_5m.csv")
    if os.path.exists(fp):
        try:
            fr.append(pd.read_csv(fp, usecols=["open_time", "o", "h", "l", "c"]))   # whole cache (~90k rows): any stored signal's RSI ruler
        except Exception:
            pass
    last = int(fr[0].open_time.max()) if fr and len(fr[0]) else t_from - 4 * DAY
    now = int(time.time() * 1000)
    try:
        fr.append(klines("BTCUSDT", "5m", max(last - 2 * H, t_from - 4 * DAY), now))
    except Exception as e:
        log(f"BTC 5m fetch failed: {e}")
    if not fr:
        return None
    b = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time")
    return b[["open_time", "o", "h", "l", "c"]].values.astype(float)


def btc_r72_at(btc, trig_close_ms):
    """BTC 3-day return at the SURGE trigger bar (entry_surge_trigger_at = the bar's CLOSE): close(bar) / close(864 bars earlier) − 1."""
    if btc is None or trig_close_ms is None:
        return None
    ot = btc[:, 0]
    i = np.searchsorted(ot, trig_close_ms - 5 * MIN)
    j = np.searchsorted(ot, trig_close_ms - 5 * MIN - 864 * 5 * MIN)
    if i >= len(ot) or ot[i] != trig_close_ms - 5 * MIN or j >= len(ot) or ot[j] != trig_close_ms - 5 * MIN - 864 * 5 * MIN:
        return None
    return float((btc[i, 4] / btc[j, 4] - 1) * 100)


def _tick_path(pair, day_ms):
    ds = time.strftime("%Y-%m-%d", time.gmtime(day_ms / 1000))
    return [os.path.join(CACHE, sub, pair, f"{ds}.npz") for sub in ("ticks_q", "ticks")]


def has_ticks(pair, day_ms):
    return any(os.path.exists(p) for p in _tick_path(pair, day_ms))


def days_of(t0, t1):
    return list(range((t0 // DAY) * DAY, t1, DAY))


def ensure_ticks(pairdays, st, now_ms):
    """fetch missing aggTrades daily archives (completed UTC days only) through scripts/backtest_fetch_ticks.py; a missing archive is
    retried at most every 3 h (state 'tick_tries')."""
    tries = st.setdefault("tick_tries", {})
    want = []
    for pair, dms in sorted(set(pairdays)):
        if dms + DAY > now_ms or has_ticks(pair, dms):
            continue
        k = f"{pair}|{dms}"
        if now_ms - int(tries.get(k, 0)) < TICK_RETRY_MS:
            continue
        tries[k] = now_ms
        want.append((pair, time.strftime("%Y-%m-%d", time.gmtime(dms / 1000))))
    for k in [k for k, v in tries.items() if now_ms - int(v) > 10 * DAY]:
        tries.pop(k, None)
    if not want:
        return 0
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=REPORTS)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("pair,date\n" + "".join(f"{p},{d}\n" for p, d in want))
        subprocess.run([sys.executable, os.path.join(SCRIPTS, "backtest_fetch_ticks.py"), tmp], cwd=ROOT,
                       capture_output=True, text=True, timeout=FETCH_TIMEOUT_S)
    except Exception as e:
        log(f"tick fetch: {e}")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if _M is not None:
        _M._DAYS.clear()                                  # a day cached as 'no archive' must be re-read
    return sum(1 for p, d in want if has_ticks(p, _ms(d)))


# ═══════════════════════════════ pricing ═══════════════════════════════
_M = None


def _replica(btc):
    """scripts/ml_exit_optimize.py (the live momentum-long exit replica) with this run's BTC 5m injected (recovery-hold ruler)."""
    global _M
    if _M is None:
        for p in (ROOT, SCRIPTS):
            if p not in sys.path:
                sys.path.insert(0, p)
        import ml_exit_optimize as M
        _M = M
    if btc is not None:
        _M._BTC = btc
    return _M


def _inject_k1(M, pair, k1):
    M._K1[pair] = k1
    M._K5.pop(pair, None)


def _atr_pct(k1, entry_ms):
    """14-bar ATR % (ta / Wilder, as services.indicators) on the CLOSED 5m bars before entry, built from the 1m klines."""
    from ta.volatility import AverageTrueRange
    d = k1[k1.open_time < entry_ms].copy()
    d["b"] = (d.open_time // (5 * MIN)) * (5 * MIN)
    g = d.groupby("b").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), n=("o", "size"))
    g = g[(g.index + 5 * MIN <= entry_ms) & (g.n == 5)]
    if len(g) < 30:
        return None
    a = AverageTrueRange(high=g.h, low=g.l, close=g.c, window=14).average_true_range()
    v = float(a.iloc[-1] / g.c.iloc[-1] * 100)
    return v if np.isfinite(v) and v > 0 else None


def price_ml(pair, entry_ms, btc, kwin=None):
    """live momentum-LONG exit replica from entry_ms → dict(pct, how, src) or dict(pending=reason). kwin = the 1m kline span to fetch
    (one span shared by both entry timings of a signal); it must cover ≥ 1 day before and ≥ 7 h after the entry."""
    M = _replica(btc)
    a, b = kwin or (entry_ms - DAY, entry_ms + 8 * H)
    try:
        k1 = klines(pair, "1m", a, b)
    except Exception as e:
        return dict(pending=f"klines: {str(e)[:60]}")
    if k1 is None or not len(k1) or k1.open_time.max() < entry_ms:
        return dict(pending="no klines yet")
    _inject_k1(M, pair, k1)
    bp = M.build_path(pair, entry_ms, 2 * H)
    if bp is None or not len(bp[0]):
        return dict(pending="no path")
    atr = _atr_pct(k1, entry_ms)
    rsi = M.btc_rsi_at(entry_ms)
    fx = dict(key=f"{pair}|{entry_ms}", pair=pair, o_ms=int(entry_ms), E=float(bp[1][0]), fee_in=M.TAKER, atr=atr, rsi_entry=rsi)
    k, why, pct, tex, src, pk = M.run_fill(fx, {"BASE": M.BASE})["BASE"]
    if why in ("END_OF_DATA", "NO_DATA") or str(why).startswith("ERR") or not np.isfinite(pct):
        return dict(pending=("still running" if why == "END_OF_DATA" else str(why)[:60]))
    return dict(pct=round(float(pct), 3), how=f"{why} {int((tex - entry_ms) / MIN)}m", src=src)


_FX = None


def _flip_replica():
    """scripts/flip_exit_replica.py + the CURRENT live flip-short exit settings (trading_config.json), loaded once per run."""
    global _FX
    if _FX is None:
        if SCRIPTS not in sys.path:
            sys.path.insert(0, SCRIPTS)
        import flip_exit_replica as F
        s = F.live_settings()
        if s["diffs"]:
            log(f"flip exit: live settings differ from the validated 'NOW' era — {'; '.join(s['diffs'])} (live values used)")
        _FX = (F, s["cfg"])
    return _FX


def _atr5_pct(pair, sig_ms):
    """Wilder ATR(14) % (ta, as services.indicators) on the pair's CLOSED 5m bars at the signal (bars that closed ≤ the journal bucket
    start) — the entry_atr_pct the engine stamps at the signal scan."""
    from ta.volatility import AverageTrueRange
    k5 = klines(pair, "5m", sig_ms - DAY, sig_ms)
    if k5 is None or len(k5) < 30:
        return None
    a = AverageTrueRange(high=k5.h, low=k5.l, close=k5.c, window=14).average_true_range()
    v = float(a.iloc[-1] / k5.c.iloc[-1] * 100)
    return v if np.isfinite(v) and v > 0 else None


def price_flip(pair, entry_ms, sig_ms=None, kwin=None):
    """🔄 (221) the live FAN flip-SHORT exit (scripts/flip_exit_replica.py, live settings) from the first print at entry_ms → dict(pct, how,
    src) or dict(pending=reason). Entry = that first print (ticks where the day archive exists, else the 1m bar open → src '1m'/'mixed');
    taker fee BOTH sides (a blocked signal has no maker fill to copy; the replica's timing-jitter validation used the taker basis);
    stop / trail ATR = Wilder ATR(14) % of the closed 5m bars at the signal (sig_ms = journal bucket start); 6 h horizon (validated)."""
    F, cfg = _flip_replica()
    M = _replica(None)
    sig_ms = int(sig_ms if sig_ms is not None else entry_ms)
    a, b = kwin or (entry_ms - H, entry_ms + F.HORIZON_MS + 10 * MIN)
    try:
        k1 = klines(pair, "1m", a, b)
        atr = _atr5_pct(pair, sig_ms)
    except Exception as e:
        return dict(pending=f"klines: {str(e)[:60]}")
    if k1 is None or not len(k1) or k1.open_time.max() < entry_ms:
        return dict(pending="no klines yet")
    if atr is None:
        return dict(pending="no 5m ATR")
    try:                                                      # a corrupt tick archive leaves THIS item pending, never the whole gate
        _inject_k1(M, pair, k1)
        bp = M.build_path(pair, entry_ms, F.HORIZON_MS)
        if bp is None or not len(bp[0]):
            return dict(pending="no path")
        t, p, src = bp
        r = F.simulate(t, p, float(p[0]), atr, F.TAKER, **cfg)
    except Exception as e:
        return dict(pending=f"path: {str(e)[:60]}")
    if r is None:
        return dict(pending="no path")
    if r["reason"] == "OPEN_END":
        if t[-1] < entry_ms + F.HORIZON_MS - 2 * MIN:
            return dict(pending="still running")
        r["reason"] = "6h cap"
    return dict(pct=round(float(r["pnl"]), 3), how=f"{r['reason']} {int((r['t'] - entry_ms) / MIN)}m", src=src, atr=round(atr, 3))


def lock_exit(net, A, Lk, g, sl, tick=True, complete=True):
    """🎯 (205) the lock-then-trail exit on one NET P&L path (pure, self-tested): −sl stop until the peak of PRIOR prints ≥ A, then
    max(Lk, peak − g). → (pct, why) · None while the 12 h path is incomplete and no line was hit. tick=False (1m path) fills AT the line."""
    pkp = np.concatenate([[-np.inf], np.maximum.accumulate(net)[:-1]])
    line = np.where(pkp >= A, np.maximum(Lk, pkp - g), -float(sl))
    hit = np.flatnonzero(net <= line)
    if len(hit):
        i = int(hit[0]); v = float(net[i]) if tick else float(line[i])
        return round(v, 3), ("TRAIL" if line[i] > 0 else "SL")
    if complete:
        return round(float(net[-1]), 3), "12h cap"
    return None


def price_fixed(pair, entry_ms, E, levels):
    """FRENZY bot-exact fixed exits on one path: levels = [(tp, sl)] → {f'{tp}/{sl}': (pct, why)} + src, or dict(pending=…).
    Net P&L at every print = (p / E − 1) · 100 − 0.09; first print with net ≤ −sl or ≥ +tp closes there; else the last print ≤ 12 h."""
    M = _replica(None)
    try:
        k1 = klines(pair, "1m", entry_ms - H, entry_ms + FRENZY_HOLD + 5 * MIN)
    except Exception as e:
        return dict(pending=f"klines: {str(e)[:60]}")
    if k1 is None or not len(k1):
        return dict(pending="no klines yet")
    _inject_k1(M, pair, k1)
    bp = M.build_path(pair, entry_ms, FRENZY_HOLD)
    if bp is None or not len(bp[0]):
        return dict(pending="no path")
    t, p, src = bp
    net = (p / float(E) - 1) * 100 - FRENZY_FEES
    out = dict(src=src)
    complete = t[-1] >= entry_ms + FRENZY_HOLD - 2 * MIN
    for spec in levels:
        if spec[0] == "lock":   # 🎯 (205) lock-then-trail: −sl stop until the peak of PRIOR prints ≥ A, then max(L, peak − g)
            _, A, Lk, g, sl = spec
            r_ = lock_exit(net, A, Lk, g, sl, src == "tick", complete)
            if r_ is None:
                return dict(pending="still running")
            out[f"lock{A:g}/{Lk:g}/{g:g}"] = r_
            continue
        tp, sl = spec
        hit = np.flatnonzero((net <= -sl) | (net >= tp))
        if len(hit):
            i = int(hit[0])
            v = float(net[i])
            if src != "tick":   # a 1m o→l/h→c path jumps to the bar extreme — the provisional price fills AT the level instead
                v = float(tp) if v >= tp else -float(sl)
            out[f"{tp:g}/{sl:g}"] = (round(v, 3), "TP" if v >= tp else "SL")
        elif complete:
            out[f"{tp:g}/{sl:g}"] = (round(float(net[-1]), 3), "12h cap")
        else:
            return dict(pending="still running")
    return out


def _is_final(src_list, t_ms, now_ms):
    return all(s == "tick" for s in src_list) or now_ms - t_ms > FINAL_AFTER_MS


class Budget:
    def __init__(self, s):
        self.end = time.time() + s

    def ok(self):
        return time.time() < self.end


# ═══════════════════════════════ gates ═══════════════════════════════
def flip_sole_blocker(J, fail_name):
    """🔄 (221) FAN flip-short refusals whose COMPLETE fail set is exactly `fail_name` (journal FAILS · dir SHORT · src FLIP:FAN_RATIO_GATE
    · gate == fail_name, no other gate joined by '+') → DataFrame(ms, pair). Same honest-cohort rule as LOADX: a signal that also failed
    another gate would have been refused anyway, so it says nothing about this filter."""
    if J is None or not len(J):
        return pd.DataFrame(columns=["ms", "pair"])
    f = J[(J.e == "FAILS") & (J.dir.astype(str) == "SHORT") & (J.src.astype(str) == FLIP_SRC)]
    f = f[f.gate.astype(str).str.split("+").map(lambda s: [x.strip() for x in s if x.strip()] == [fail_name])]
    return f[["ms", "pair"]]


def _signals_for(J, gate_name, ranks=None, store=None):
    """refused LONG signals of one gate → DataFrame(ms, pair, key) — pair-episodes, in time order. Signals already stored in the state
    are kept even when their export has left ~/Downloads (a frozen first-N set never shrinks)."""
    old = pd.DataFrame([dict(ms=int(v["t"]), pair=v["pair"]) for v in (store or {}).values()], columns=["ms", "pair"])
    if J is None or not len(J):
        ep = old
    elif gate_name == "LOADX":
        f = J[J.e == "FAILS"].copy()
        f = f[f.gate.astype(str).str.replace("MACRO:", "", regex=False) == "PAIR_RSI_MOMENTUM_LOADX"]
        if "src" in f:
            f = f[f.src.astype(str).isin(["MOMENTUM", "nan", "None"])]
        ep = episodes(pd.concat([old, f[["ms", "pair"]]], ignore_index=True))
    elif gate_name in FLIP_GATES:
        ep = episodes(pd.concat([old, flip_sole_blocker(J, FLIP_GATES[gate_name])], ignore_index=True))
    else:
        g = {"CHOP_BURST": "LONG_CHOP_BURST", "HEAT": "LONG_HEAT_BLOCK", "HEAT_ORIG": "LONG_HEAT_BLOCK", "MEGACAP": "LONG_MEGACAP_BLOCK"}[gate_name]
        raw = pd.concat([old, J[(J.e == "BLOCK") & (J.gate.astype(str) == g)][["ms", "pair"]]], ignore_index=True)
        if gate_name in ("HEAT", "HEAT_ORIG"):   # 🔁 (208) split BEFORE the episode fold: the re-scope's blocks end at the commit, the
            _t = deploy_ms("HEAT_REVERT") - 10 * MIN   # original rule's start 30 min after it (EB deploy margin — review)
            raw = raw[raw.ms < _t] if gate_name == "HEAT" else raw[raw.ms >= _t + 30 * MIN]
        ep = episodes(raw)
    if ranks and gate_name in ("LOADX", "HEAT", "HEAT_ORIG"):    # these gates run BEFORE the mega-cap block: a rank ≤ 10 pair is refused there anyway
        ep = ep[~ep.pair.map(lambda p: ranks.get(p, 999) <= 10)]
    if not len(ep):   # e.g. HEAT_ORIG right after the revert: no blocked fire yet
        return pd.DataFrame(columns=["ms", "pair", "key"]).astype({"ms": "int64"})
    ep = ep.astype({"ms": "int64"}).sort_values(["ms", "pair"]).reset_index(drop=True)
    ep["key"] = ep.pair.astype(str) + "|" + ep.ms.astype(str)
    return ep


def _price_signals(sig, store, n_needed, btc, budget, now_ms, need_days, pricer=None):
    """price (or reuse stored prices for) the first n_needed signals (None = all). Both entry timings. Returns the list of items.
    pricer(pair, entry_ms, sig_ms) → price dict; default = the momentum-LONG replica (price_ml)."""
    live = set(sig.key)
    for k in [k for k in store if k not in live]:      # e.g. a pair whose rank now marks it mega-cap — no longer in the cohort
        store.pop(k, None)
    out = []
    todo = sig if n_needed is None else sig.head(n_needed)
    for r in todo.itertuples():
        it = store.get(r.key) or dict(pair=r.pair, t=int(r.ms))
        if not it.get("final"):
            if need_days is not None:                           # phase 1: collect the tick days this signal needs
                for off in (1, 5):
                    need_days.update((r.pair, d) for d in days_of(int(r.ms) + off * MIN, int(r.ms) + off * MIN + 7 * H))
            elif budget.ok():
                if pricer is None:
                    res = {off: price_ml(r.pair, int(r.ms) + off * MIN, btc, (int(r.ms) - DAY, int(r.ms) + 8 * H)) for off in (1, 5)}
                else:
                    res = {off: pricer(r.pair, int(r.ms) + off * MIN, int(r.ms)) for off in (1, 5)}
                for off, x in res.items():
                    if "pending" in x:
                        it[f"p{off}"] = x["pending"]
                        it.pop(f"sim{off}", None)
                    else:
                        it.update({f"sim{off}": x["pct"], f"how{off}": x["how"], f"src{off}": x["src"]})
                        if x.get("atr") is not None:                  # 🔄 (221) flip pricer: the 5m ATR % the stop / trail used
                            it["atr"] = x["atr"]
                        it.pop(f"p{off}", None)
                srcs = [it.get("src1"), it.get("src5")]
                # final = both timings on ticks, or the primary priced and 4 days passed (no archive / a timing that never prices)
                it["final"] = bool((all(s is not None for s in srcs) and all(s == "tick" for s in srcs))
                                   or (srcs[0] is not None and now_ms - int(r.ms) > FINAL_AFTER_MS))
            store[r.key] = it
        out.append(it)
    return out


def _sig_progress(items, n, unit="signals"):
    pr = [x for x in items if x.get("sim1") is not None]
    fin = [x for x in pr if x.get("final")]
    s1 = [x["sim1"] for x in pr]
    s5 = [x["sim5"] for x in pr if x.get("sim5") is not None]
    txt = (f"{len(pr)}/{n} {unit} re-priced" + (f" ({len(pr) - len(fin)} provisional)" if len(pr) > len(fin) else "")
           + (f" · {sum(1 for v in s1 if v > 0)} won · Σ {sum(s1):+.2f} %" if s1 else "")
           + (f" (t+5m: {sum(1 for v in s5 if v > 0)} won · Σ {sum(s5):+.2f})" if s5 else ""))
    return txt, pr, fin


def _priced_txt(items):
    pr = [x for x in items if x.get("sim1") is not None]
    return f"priced {sum(1 for x in pr if x.get('final'))} final / {sum(1 for x in pr if not x.get('final'))} on 1m" if pr else ""


def _sig_detail(items, k=8):
    return " · ".join(f"{_fmt_t(x['t'])} {str(x['pair']).replace('USDT', '')} "
                      + (f"{x['sim1']:+.2f}/{_f(x.get('sim5'))}" if x.get("sim1") is not None else f"pending ({x.get('p1', '?')})")
                      + ("" if x.get("final") or x.get("sim1") is None else "ᵖ")
                      for x in items[:k])


def gate_first_n_signals(code, J, st, n, wr_min, mode, btc, budget, now_ms, need_days, ranks=None, windows=False):
    """CHOP_BURST / HEAT (signal units) and LOADX (window units: one 5-min bucket = one window, value = mean of its signals)."""
    G = st.setdefault("gates", {}).setdefault(code, {})
    store = G.setdefault("items", {})
    sig = _signals_for(J, code, ranks, store)
    if windows:   # the first n windows = the signals of the first n distinct buckets
        bk = sorted(sig.ms.unique())[:n]
        sig = sig[sig.ms.isin(bk)]
        n_sig = None
    else:
        n_sig = n
    items = _price_signals(sig, store, n_sig, btc, budget, now_ms, need_days)
    if need_days is not None:
        return None
    if windows:
        byb = {}
        for x in items:
            byb.setdefault(x["t"], []).append(x)
        wins = [byb[b] for b in sorted(byb)]
        ready = [w for w in wins if all(x.get("sim1") is not None for x in w)]
        v1 = [float(np.mean([x["sim1"] for x in w])) for w in ready]
        v5 = [float(np.mean([x["sim5"] for x in w])) for w in ready if all(x.get("sim5") is not None for x in w)]
        all_final = len(ready) == len(wins) and all(x.get("final") for w in wins for x in w)
        npr = len(ready)
        prog = (f"{npr}/{n} windows re-priced ({len(items)} signals)" + ("" if all_final or not npr else " (provisional)")
                + (f" · {sum(1 for v in v1 if v > 0)} won · Σ {sum(v1):+.2f} %" if v1 else "")
                + (f" (t+5m: {sum(1 for v in v5 if v > 0)} won · Σ {sum(v5):+.2f})" if v5 else ""))
    else:
        prog, pr, fin = _sig_progress(items, n)
        v1 = [x["sim1"] for x in pr]
        v5 = [x["sim5"] for x in pr if x.get("sim5") is not None]
        all_final = len(fin) == len(pr)
    kw = dict(or_sum_pos=(mode == "or"), need_sum_pos=(mode == "and"))
    state, wr, s = decide_first_n(v1, n, wr_min, **kw)
    state5 = decide_first_n(v5, n, wr_min, **kw)[0] if len(v5) >= n else None
    G.update(progress=prog, detail=_sig_detail(items, len(items) if windows else max(8, n)), priced=_priced_txt(items))
    if state != "collecting" and not all_final:
        G["provisional"] = state
        state = "collecting"
    else:
        G.pop("provisional", None)
    G["fragile"] = bool(state in ("fired", "holds") and state5 is not None and state5 != state)
    return state


def decide_flip_blocked(window_means, n=FLIP_N):
    """🔄 (221) frozen bar for a FAN flip-short entry filter, judged on the signals it BLOCKED: the first n WINDOWS (5-min journal buckets;
    value = the mean of the window's sole-blocker signals re-priced with the live flip exit) — mean of the window means > 0 → 'fired'
    (the filter blocked winners → review switching it off); ≤ 0 → 'holds'; fewer than n → 'collecting'."""
    v = list(window_means)[:n]
    if len(v) < n:
        return "collecting"
    return "fired" if float(np.mean(v)) > 0 else "holds"


def gate_flip_blocked(code, J, st, budget, now_ms, need_days, n=FLIP_N):
    """🔄 (221) FLIP_EMA13_BLOCKED / FLIP_PADX_BLOCKED: sole-blocker refusals of one FAN flip filter (journal FAILS whose complete set is that
    gate alone; one signal per pair-episode), WINDOW units (EMA13 is market-wide; pair ADX is pair-level, so window clustering is merely
    conservative there → same-bucket signals = ONE observation). Sole blocker within _flip_filters only = necessary, not sufficient
    (open_position's slot / existing-position / cooldown refusals are not replayed). EMA13 almost always co-fails with other gates
    (0 sole-blocker lines in the first ~8 days of journals) → that gate may never reach 10 windows. Windows before FLIP_REG_MS excluded. re-priced as if the flip had opened at bucket +1 min (primary) and +5 min (bracket) with the live FAN
    flip-short exit (price_flip). A verdict counts on final (tick) prices only; on 1m it is shown as provisional."""
    G = st.setdefault("gates", {}).setdefault(code, {})
    store = G.setdefault("items", {})
    sig = _signals_for(J, code, None, store)
    sig = sig[sig.ms >= FLIP_REG_MS]                          # forward windows only (registration anchor)
    bk = sorted(sig.ms.unique())[:n]
    sig = sig[sig.ms.isin(bk)]
    items = _price_signals(sig, store, None, None, budget, now_ms, need_days,
                           pricer=lambda pair, e_ms, s_ms: price_flip(pair, e_ms, s_ms, (s_ms - H, s_ms + 5 * MIN + 6 * H + 10 * MIN)))
    if need_days is not None:
        return None
    byb = {}
    for x in items:
        byb.setdefault(x["t"], []).append(x)
    wins = [byb[b] for b in sorted(byb)]
    ready = [w for w in wins if all(x.get("sim1") is not None for x in w)]
    v1 = [float(np.mean([x["sim1"] for x in w])) for w in ready]
    v5 = [float(np.mean([x["sim5"] for x in w])) for w in ready if all(x.get("sim5") is not None for x in w)]
    all_final = len(ready) == len(wins) and all(x.get("final") for w in wins for x in w)
    G["progress"] = (f"{len(ready)}/{n} windows re-priced ({len(items)} sole-blocker signals)" + ("" if all_final or not ready else " (provisional)")
                     + (f" · {sum(1 for v in v1 if v > 0)} positive · mean of window means {np.mean(v1):+.3f} %" if v1 else "")
                     + (f" (t+5m: {np.mean(v5):+.3f})" if v5 else ""))
    G["detail"] = _sig_detail(items, len(items))
    G["priced"] = _priced_txt(items)
    state = decide_flip_blocked(v1, n)
    state5 = decide_flip_blocked(v5, n) if len(v5) >= n else None
    if state != "collecting" and not all_final:
        G["provisional"] = state
        state = "collecting"
    else:
        G.pop("provisional", None)
    G["fragile"] = bool(state in ("fired", "holds") and state5 is not None and state5 != state)
    return state


# ═══════════════════════════════ 📊 MS_PVR_BLOCKED (operator 2026-10-06) ═══════════════════════════════
_MSP = None


def _msp():
    """scripts/ms_pvr_shadow.py (PVR rebuild, the momentum-short exit replica, the kept-side tally) — loaded once."""
    global _MSP
    if _MSP is None:
        if SCRIPTS not in sys.path:
            sys.path.insert(0, SCRIPTS)
        import ms_pvr_shadow as S
        _MSP = S
    return _MSP


def ms_pvr_signals(J, store=None):
    """MOMENTUM_SHORT_PAIRVOL refusals → DataFrame(ms, pair, key): journal BLOCK lines (dir SHORT), one signal per pair-episode, in time
    order. The gate sits in open_position after every momentum-short entry gate and before only sizing / balance / book checks, so a
    BLOCK line = the pair-volume ceiling was the sole blocker of a signal that passed the ladder (sizing refusals not replayed). Stored
    signals are kept when their export leaves ~/Downloads."""
    old = pd.DataFrame([dict(ms=int(v["t"]), pair=v["pair"]) for v in (store or {}).values()], columns=["ms", "pair"])
    new = (J[(J.e == "BLOCK") & (J.dir.astype(str) == "SHORT") & (J.gate.astype(str) == MS_PVR_GATE)][["ms", "pair"]]
           if J is not None and len(J) else pd.DataFrame(columns=["ms", "pair"]))
    raw = pd.concat([old, new], ignore_index=True)
    if not len(raw):
        return pd.DataFrame(columns=["ms", "pair", "key"]).astype({"ms": "int64"})
    ep = episodes(raw.astype({"ms": "int64"})).sort_values(["ms", "pair"]).reset_index(drop=True)
    ep["key"] = ep.pair.astype(str) + "|" + ep.ms.astype(str)
    return ep


def ms_cohort_stats(items, cohorts, final_only=False):
    """(n, days, wr, mean) of the primary re-price over the items whose PVR class is in `cohorts` (pure, self-tested).
    Double-count episodes (a live fill of the pair followed the refusal) are excluded."""
    v = [(x["t"], x["sim1"]) for x in items if x.get("cls") in cohorts and x.get("sim1") is not None and not x.get("dup")
         and (x.get("final") or not final_only)]
    if not v:
        return 0, 0, float("nan"), float("nan")
    vals = [b for _, b in v]
    return len(v), len({_fmt_t(a)[:5] for a, _ in v}), _wr(vals), float(np.mean(vals))


def ms_followed_by_fill(sig_ms, pair, fills_ms_by_pair):
    """True when a live momentum-short fill of the same pair opened within the refusal's bucket + EPISODE_MIN (the refused signal and
    the fill are the same move → pricing both double-counts it). fills_ms_by_pair = {pair: [o_ms, …]} (pure, self-tested)."""
    lo, hi = int(sig_ms), int(sig_ms) + 5 * MIN + EPISODE_MIN * MIN
    return any(lo <= int(t) <= hi for t in fills_ms_by_pair.get(pair, ()))


def ms_review_first(rows, frozen, n):
    """② the first n kept fills since 09-18 12:00 as [(key, pnl %)] — a CLOSED prefix in open order (a still-open earlier fill holds the
    set back); `frozen` (from the state) wins once set, so later-closing fills never rewrite the read (pure, self-tested).
    rows = iterable of (key, status, pnl %) sorted by open time."""
    if frozen:
        return [tuple(x) for x in frozen][:n], True
    out = []
    for k, stt, pnl in rows:
        if str(stt) != "CLOSED" or pnl is None or not np.isfinite(pnl):
            break
        out.append((k, float(pnl)))
        if len(out) >= n:
            return out, True
    return out, False


def gate_ms_pvr(J, st, budget, now_ms, need_days, orders_fn=None):
    """📊 MS_PVR_BLOCKED — the momentum shorts the pair-volume ceiling (0.86) REFUSED, re-priced as if opened (observe-only), judged by
    the frozen DECISION_LOG 226 gates (the Sep-18 kept-side 70 %/15 revert is OVERRIDDEN):
      ① REVERT to 1.0 if Cohort A (0.86 ≤ PVR < 1.0 refusals from 2026-10-06 18:00 UTC, final prices, double-count episodes excluded)
        reaches 15 signals with mean ≥ the kept side's mean over the same period (kept fills since 10-06 18:00; < 5 → since 09-18 12:00)
      ② SLEEVE REVIEW (full sleeve-kill checklist, no auto-kill) if the first 20 kept fills since 09-18 12:00 (keys frozen into the state
        when N first reaches 20) are below the breakeven WR 59 %.
    Per signal: the refusal bucket's PVR rebuilt on a 5-s grid (ms_pvr_shadow.refusal_profile: A = every second consistent with the
    refusal is < 1.0; B ≥ 1.0 context; A~/B~ straddle 1.0); entry = the first print ≥ the earliest consistent second + the live 35-s
    scan→fill delay (primary) / the latest + 35 s (bracket), taker fee; ATR % and the C1 legs rebuilt from bars CLOSED by the scan + the
    last print; exit = the live momentum-short replica (ms_pvr_shadow.simulate_ms). Earlier refusals = reference only.
    Returns 'fired' (①) · 'review' (②) · 'holds' · 'collecting'."""
    S = _msp()
    G = st.setdefault("gates", {}).setdefault("MS_PVR_BLOCKED", {})
    store = G.setdefault("items", {})
    cfg, diffs = S.live_ms_settings()
    A_ = (orders_fn or S.load_momentum_shorts)() if need_days is None else None
    fills = {}
    if A_ is not None:
        for p_, t_ in zip(A_.pair, A_.o_ms):
            fills.setdefault(p_, []).append(int(t_))
    sig = ms_pvr_signals(J, store)
    items = []
    for r in sig.itertuples():
        it = store.get(r.key) or dict(pair=r.pair, t=int(r.ms), ref=bool(int(r.ms) < MS_PVR_REG_MS))
        if A_ is not None:
            it["dup"] = ms_followed_by_fill(int(r.ms), r.pair, fills)
        if not it.get("final"):
            if "cls" not in it and budget.ok():
                pf = S.refusal_profile(r.pair, int(r.ms))
                if "pending" in pf:
                    it["p1"] = pf["pending"]
                else:
                    it.update(cls=pf["cls"], pvr_lo=pf["lo"], pvr_med=pf["med"], pvr_hi=pf["hi"], scan1=pf["first_ok_ms"], scan5=pf["last_ok_ms"])
                    it.pop("p1", None)
            if it.get("scan1") is not None and "atr" not in it and budget.ok():
                try:
                    stp = S.entry_stamps(r.pair, int(it["scan1"]))
                except Exception as e:
                    stp = None
                    log(f"MS_PVR stamps {r.pair}: {str(e)[:80]}")
                if stp:
                    it.update(atr=stp["atr"], c1=bool(stp["c1"]))
            if need_days is not None:
                for k in ("scan1", "scan5"):
                    if it.get(k):
                        e_ms = int(it[k]) + S.ENTRY_DELAY_S * 1000
                        need_days.update((r.pair, d) for d in days_of(e_ms, e_ms + int(cfg["maxhold_min"]) * MIN))
            elif it.get("atr") is not None and it.get("cls") != "unk" and budget.ok():
                for off, k in ((1, "scan1"), (5, "scan5")):
                    if not it.get(k):
                        continue
                    e_ms = int(it[k]) + S.ENTRY_DELAY_S * 1000
                    x = S.price_ms(r.pair, e_ms, int(it[k]), it["atr"], it.get("c1", False), cfg=cfg)
                    if "pending" in x:
                        it[f"p{off}"] = x["pending"]
                        it.pop(f"sim{off}", None)
                    else:
                        it.update({f"sim{off}": x["pct"], f"how{off}": x["how"], f"src{off}": x["src"]})
                        it.pop(f"p{off}", None)
                srcs = [it.get("src1"), it.get("src5")]
                it["final"] = bool((all(s is not None for s in srcs) and all(s == "tick" for s in srcs))
                                   or (srcs[0] is not None and now_ms - int(r.ms) > FINAL_AFTER_MS))
        store[r.key] = it
        items.append(it)
    if need_days is not None:
        return None
    cnt = [x for x in items if not x.get("ref")]
    ref = [x for x in items if x.get("ref")]
    dups = [x for x in cnt if x.get("dup")]
    nA, dA, wA, mA = ms_cohort_stats(cnt, ("A", "A~"), final_only=True)
    nAp, _, _, mAp = ms_cohort_stats(cnt, ("A", "A~"))
    nAs = ms_cohort_stats(cnt, ("A",), final_only=True)[0]
    nB, dB, wB, mB = ms_cohort_stats(cnt, ("B", "B~"), final_only=True)
    nBp = ms_cohort_stats(cnt, ("B", "B~"))[0]
    # ① revert gate — Cohort A (final, non-duplicate, signal order) vs the kept side over the same period
    a_vals = [x["sim1"] for x in cnt if x.get("cls") in ("A", "A~") and x.get("sim1") is not None and x.get("final") and not x.get("dup")]
    kref, klab = S.kept_ref(A_)
    g1 = S.revert_gate(a_vals, kref["mean"])
    # ② sleeve review — the first 20 kept fills since 09-18 12:00, frozen into the state at N = 20
    t0 = int(pd.Timestamp(S.KEPT_FROM).value // 1_000_000)
    kk = A_[(A_.o_ms >= t0) & (A_.pvr < S.PVR_CEIL) & A_.stack_keep.astype(bool)].sort_values("o_ms")
    rows = [(f"{str(o)[:19].replace(' ', 'T')}|{p_}", stt, (float(v) if pd.notna(v) else None))
            for o, p_, stt, v in zip(kk.opened_at, kk.pair, kk.status, kk.pnl_percentage)]
    first, full = ms_review_first(rows, G.get("review_first20"), S.REVIEW_N)
    if full and not G.get("review_first20"):
        G["review_first20"] = [list(x) for x in first]
        G["review_frozen_at"] = _fmt_t(now_ms, True)
    g2 = S.review_gate([v for _, v in first])
    kf, ks = S.kept_tally(A_)
    state = "fired" if g1 == "fired" else "review" if g2 == "review" else ("holds" if g1 == "holds" else "collecting")
    G["kept"] = dict(n=ks["n"], wins=ks["wins"], wr=round(ks["wr"], 1) if np.isfinite(ks["wr"]) else None,
                     mean=round(ks["mean"], 3) if np.isfinite(ks["mean"]) else None, usd=round(ks["sum_usd"], 2),
                     ref=klab, ref_n=kref["n"], ref_mean=(round(kref["mean"], 3) if np.isfinite(kref["mean"]) else None), g1=g1, g2=g2)
    G["revalidate"] = ("; ".join(diffs) if diffs else None)
    wr20 = (100.0 * sum(1 for _, v in first if v > 0) / len(first)) if first else float("nan")
    g1txt = {"insufficient": f"① insufficient (A {nA}/{S.REVERT_N})", "fired": "① 🔔 FIRED → revert momentum_short_pair_vol_max to 1.0",
             "holds": "① holds — Cohort A below the kept mean, 0.86 stays"}[g1]
    g2txt = {"collecting": f"② collecting ({len(first)}/{S.REVIEW_N} kept fills)",
             "review": f"② 🔔 SLEEVE REVIEW — first {S.REVIEW_N} kept fills {wr20:.0f} % < breakeven {S.BREAKEVEN_WR:.0f} % (full sleeve-kill checklist, no auto-kill)",
             "holds": f"② holds — first {S.REVIEW_N} kept fills {wr20:.0f} % ≥ {S.BREAKEVEN_WR:.0f} %"}[g2]
    G["status_txt"] = (f"{g1txt} · {g2txt} · Sep-18 70 %/15 kept-side revert: overridden (DECISION_LOG 226)"
                       + (" · ⚠ replica re-validate needed" if diffs else ""))
    G["progress"] = (f"Cohort A (0.86 ≤ PVR < 1.0, what a revert re-admits; final prices): {nA}/{S.REVERT_N} signals"
                     + (f" ({nA - nAs} straddle 1.0)" if nA > nAs else "")
                     + (f" · {dA} days · WR {wA:.0f} % · mean {mA:+.3f} %" if nA else "")
                     + (f" [incl. {nAp - nA} provisional: mean {mAp:+.3f} %]" if nAp > nA else "")
                     + f" · Cohort B (PVR ≥ 1.0, context): {nB}" + (f" · WR {wB:.0f} % · mean {mB:+.3f} %" if nB else "")
                     + (f" [+{nBp - nB} provisional]" if nBp > nB else "")
                     + (f" · {len(dups)} double-count episode(s) excluded (live fill followed)" if dups else "")
                     + f" ‖ ① kept reference ({klab}): N {kref['n']}" + (f" · mean {kref['mean']:+.3f} %" if kref["n"] else "")
                     + f" ‖ ② KEPT-SIDE TALLY (momentum shorts PVR < 0.86, stack-kept, from 09-18 12:00, closed): {ks['n']} fills · "
                     + (f"{ks['wins']} won · WR {ks['wr']:.0f} % · mean {ks['mean']:+.3f} % · ${ks['sum_usd']:+.0f}" if ks["n"] else "–")
                     + f" · first-{S.REVIEW_N} set {len(first)}/{S.REVIEW_N}" + (f" (frozen {G.get('review_frozen_at')})" if G.get("review_first20") else "")
                     + (f" · ⚠ replica re-validate needed: {G['revalidate']}" if diffs else ""))
    fmt = lambda x: (f"{_fmt_t(x['t'])} {str(x['pair']).replace('USDT', '')} [{x.get('cls', '?')} PVR {_f(x.get('pvr_lo'), '.2f')}–{_f(x.get('pvr_hi'), '.2f')}"
                     + (" C1" if x.get("c1") else "") + "] "
                     + (f"{x['sim1']:+.2f}/{_f(x.get('sim5'))} ({x.get('how1', '')})" if x.get("sim1") is not None else f"pending ({x.get('p1', '?')})")
                     + ("" if x.get("final") or x.get("sim1") is None else "ᵖ"))
    live = [x for x in cnt if not x.get("dup")]
    G["detail"] = ((" · ".join(fmt(x) for x in live) if live else "no counted refusal yet (floor 10-06 18:00 UTC)")
                   + (" ‖ double-count (a live fill of the pair followed ≤ 65 min; NOT in the gate count): " + " · ".join(fmt(x) for x in dups) if dups else "")
                   + (" ‖ pre-floor reference: " + " · ".join(fmt(x) + (" (fill followed)" if x.get("dup") else "") for x in ref) if ref else "")
                   + " ‖ kept fills: " + " · ".join(f"{str(r_.opened_at)[5:16].replace('T', ' ')} {str(r_.pair).replace('USDT', '')} "
                                                     f"PVR {r_.pvr:.2f} {r_.pnl_percentage:+.2f}" for r_ in kf.itertuples()))
    G["priced"] = _priced_txt(cnt)
    return state


def gate_megacap(J, st, btc, budget, now_ms, need_days):
    G = st.setdefault("gates", {}).setdefault("MEGACAP", {})
    store = G.setdefault("items", {})
    sig = _signals_for(J, "MEGACAP", None, store)
    items = _price_signals(sig, store, None, btc, budget, now_ms, need_days)
    if need_days is not None:
        return None
    prog, pr, fin = _sig_progress(items, 8)
    fin_items = [x for x in pr if x.get("final")]
    state, wr, s = decide_megacap([x["sim1"] for x in fin_items], [x["t"] for x in fin_items])
    G.update(progress=prog + f" · {len({x['t'] for x in pr})} windows", detail=_sig_detail(items), priced=_priced_txt(items))
    return state


def gate_frenzy_tp(orders, st, budget, now_ms, need_days, n=20):   # RETIRED Oct-5 (superseded by gate_frenzy_lock, DECISION_LOG 205)
    G = st.setdefault("gates", {}).setdefault("FRENZY_TP3", {})
    store = G.setdefault("items", {})
    t0 = deploy_ms("FRENZY_TP3")
    f = orders[orders.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (orders.o_ms >= t0)].head(n)
    items = []
    for r in f.itertuples():
        key = f"{str(r.opened_at)[:19]}|{r.pair}"
        it = store.get(key) or dict(pair=r.pair, t=int(r.o_ms), sleeve=r.entry_strategy)
        closed = str(r.status) == "CLOSED" and np.isfinite(r.pnl_percentage)
        it["actual"] = float(r.pnl_percentage) if closed else None
        if not it.get("final") and np.isfinite(pd.to_numeric(r.entry_price, errors="coerce")):
            if need_days is not None:
                need_days.update((r.pair, d) for d in days_of(int(r.o_ms), int(r.o_ms) + FRENZY_HOLD))
            elif budget.ok():
                x = price_fixed(r.pair, int(r.o_ms), float(r.entry_price), [(4, 3), (3, 3)])
                if "pending" in x:
                    it["p"] = x["pending"]
                else:
                    it.update(alt4=x["4/3"][0], alt4_why=x["4/3"][1], rep3=x["3/3"][0], src=x["src"])
                    it.pop("p", None)
                    it["final"] = _is_final([x["src"]], int(r.o_ms), now_ms)
        store[key] = it
        items.append(it)
    if need_days is not None:
        return None
    done = [x for x in items if x.get("actual") is not None and x.get("alt4") is not None]
    fin = [x for x in done if x.get("final")]
    state, ma, mb = decide_tp([x["actual"] for x in fin], [x["alt4"] for x in fin], n)
    pa = [x["actual"] for x in done]
    pb = [x["alt4"] for x in done]
    pr3 = [x["rep3"] for x in done if x.get("rep3") is not None]
    G["progress"] = (f"{len(done)}/{n} fills re-priced" + (f" ({len(done) - len(fin)} provisional)" if len(done) > len(fin) else "")
                     + (f" · actual avg {np.mean(pa):+.2f} % vs +4/−3 {np.mean(pb):+.2f} %" if done else "")
                     + (f" (tick +3/−3 replica {np.mean(pr3):+.2f} — fidelity check)" if pr3 else "")
                     + (f" · {len(items) - len(done)} open/pending" if len(items) > len(done) else ""))
    G["detail"] = " · ".join(f"{_fmt_t(x['t'])} {x['pair'].replace('USDT', '')} {_f(x.get('actual'))}→{_f(x.get('alt4'))}"
                             + ("" if x.get("final") or x.get("alt4") is None else "ᵖ") for x in items[:10])
    if state == "collecting" and len(done) >= n and len(fin) < n:
        G["provisional"] = decide_tp(pa, pb, n)[0]
    else:
        G.pop("provisional", None)
    return state, t0


def gate_frenzy_lock(orders, st, budget, now_ms, need_days, n=20):
    """🎯 (205) first n FRENZY + WIDE fills after the lock deploy: live (lock +2 at +3, trail 2) vs fixed +3/−3 (the revert line), plus
    fixed +6/−3 and +4/−3 for the record, all re-priced on ticks with the bot's accounting; a tick lock replica is the fidelity check."""
    G = st.setdefault("gates", {}).setdefault("FRENZY_LOCK", {})
    store = G.setdefault("items", {})
    t0 = deploy_ms("FRENZY_LOCK")
    f = orders[orders.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (orders.o_ms >= t0)].head(n)
    items = []
    for r in f.itertuples():
        key = f"{str(r.opened_at)[:19]}|{r.pair}"
        it = store.get(key) or dict(pair=r.pair, t=int(r.o_ms), sleeve=r.entry_strategy)
        closed = str(r.status) == "CLOSED" and np.isfinite(r.pnl_percentage)
        it["actual"] = float(r.pnl_percentage) if closed else None
        if not it.get("final") and np.isfinite(pd.to_numeric(r.entry_price, errors="coerce")):
            if need_days is not None:
                need_days.update((r.pair, d) for d in days_of(int(r.o_ms), int(r.o_ms) + FRENZY_HOLD))
            elif budget.ok():
                x = price_fixed(r.pair, int(r.o_ms), float(r.entry_price), [(3, 3), (6, 3), (4, 3), ("lock", 3, 2, 2, 3)])
                if "pending" in x:
                    it["p"] = x["pending"]
                else:
                    it.update(f3=x["3/3"][0], f6=x["6/3"][0], f4=x["4/3"][0], lockrep=x["lock3/2/2"][0], src=x["src"])
                    it.pop("p", None)
                    it["final"] = _is_final([x["src"]], int(r.o_ms), now_ms)
        store[key] = it
        items.append(it)
    if need_days is not None:
        return None
    done = [x for x in items if x.get("actual") is not None and x.get("f3") is not None]
    fin = [x for x in done if x.get("final")]
    # review (Oct-5): decide on the SAME accounting both sides — the tick lock replica vs the tick fixed +3/−3 (the live P&L carries real
    # slippage / polling the replica does not; it is shown as the fidelity line, never the decision)
    fin = [x for x in fin if x.get("lockrep") is not None]
    state, ma, mb = decide_lock([x["lockrep"] for x in fin], [x["f3"] for x in fin], n)
    if done:
        m = lambda k: float(np.mean([x[k] for x in done]))
        best6 = m("f6") > max(m("lockrep"), m("f3"))
        G["progress"] = (f"{len(done)}/{n} fills re-priced" + (f" ({len(done) - len(fin)} provisional)" if len(done) > len(fin) else "")
                         + f" · lock (tick) {m('lockrep'):+.2f} % vs fixed +3/−3 (tick) {m('f3'):+.2f} · +6/−3 {m('f6'):+.2f} · +4/−3 {m('f4'):+.2f}"
                         + f" · live as traded {m('actual'):+.2f} (fidelity)" + (" · ⚑ +6/−3 leads — review" if best6 else "")
                         + (f" · {len(items) - len(done)} open/pending" if len(items) > len(done) else ""))
    else:
        G["progress"] = f"0/{n} fills re-priced" + (f" · {len(items)} open/pending" if items else "")
    G["detail"] = " · ".join(f"{_fmt_t(x['t'])} {x['pair'].replace('USDT', '')} live {_f(x.get('actual'))} · +3 {_f(x.get('f3'))} · +6 {_f(x.get('f6'))}"
                             + ("" if x.get("final") or x.get("f3") is None else "ᵖ") for x in items[:10])
    if state == "collecting" and len(done) >= n and len(fin) < n:
        G["provisional"] = decide_lock([x["lockrep"] for x in done if x.get("lockrep") is not None], [x["f3"] for x in done if x.get("lockrep") is not None], n)[0]
    else:
        G.pop("provisional", None)
    return state, t0


def gate_tp3_vs_lock(orders, st, budget, now_ms, need_days, n=TP3_N):
    """🎯 (250) mirror of FRENZY_LOCK (205): the first n FRENZY_LONG / WIDE / LITE fills from the Oct-8 deploy (live exit = fixed +3/−3), both
    exits re-priced on ticks with the bot's accounting (price_fixed) — the decision reads the tick lock replica vs the tick fixed +3/−3 (same
    accounting, paired); the live result is the fidelity line."""
    G = st.setdefault("gates", {}).setdefault("TP3_VS_LOCK", {})
    store = G.setdefault("items", {})
    t0 = deploy_ms("FRENZY_OCT8")
    f = orders[orders.entry_strategy.astype(str).isin(list(FRENZY3)) & (orders.o_ms >= t0)].head(n)
    items = []
    for r in f.itertuples():
        key = f"{str(r.opened_at)[:19]}|{r.pair}"
        it = store.get(key) or dict(pair=r.pair, t=int(r.o_ms), sleeve=r.entry_strategy)
        closed = str(r.status) == "CLOSED" and np.isfinite(r.pnl_percentage)
        it["actual"] = float(r.pnl_percentage) if closed else None
        if not it.get("final") and np.isfinite(pd.to_numeric(r.entry_price, errors="coerce")):
            if need_days is not None:
                need_days.update((r.pair, d) for d in days_of(int(r.o_ms), int(r.o_ms) + FRENZY_HOLD))
            elif budget.ok():
                x = price_fixed(r.pair, int(r.o_ms), float(r.entry_price), [(3, 3), ("lock", 3, 2, 2, 3)])
                if "pending" in x:
                    it["p"] = x["pending"]
                else:
                    it.update(f3=x["3/3"][0], lockrep=x["lock3/2/2"][0], src=x["src"])
                    it.pop("p", None)
                    it["final"] = _is_final([x["src"]], int(r.o_ms), now_ms)
        store[key] = it
        items.append(it)
    if need_days is not None:
        return None
    done = [x for x in items if x.get("f3") is not None and x.get("lockrep") is not None]
    fin = [x for x in done if x.get("final")]
    state, d = decide_tp3_lock([x["lockrep"] for x in fin], [x["f3"] for x in fin], n)
    if done:
        dd = float(np.sum([x["lockrep"] - x["f3"] for x in done]))
        la = [x for x in done if x.get("actual") is not None]
        G["progress"] = (f"{len(done)}/{n} fills re-priced" + (f" ({len(done) - len(fin)} provisional)" if len(done) > len(fin) else "")
                         + f" · Σ lock {np.sum([x['lockrep'] for x in done]):+.2f} vs Σ fixed +3/−3 {np.sum([x['f3'] for x in done]):+.2f} (ticks) → Δ {dd:+.2f} pts (bar > +{TP3_MARGIN:g})"
                         + (f" · live as traded Σ {np.sum([x['actual'] for x in la]):+.2f} on {len(la)} closed (fidelity)" if la else "")
                         + (f" · {len(items) - len(done)} open/pending" if len(items) > len(done) else ""))
    else:
        G["progress"] = f"0/{n} fills re-priced" + (f" · {len(items)} open/pending" if items else "")
    G["detail"] = " · ".join(f"{_fmt_t(x['t'])} {x['pair'].replace('USDT', '')} {str(x.get('sleeve', '')).replace('FRENZY_', '')[:4]} live {_f(x.get('actual'))} · "
                             f"+3 {_f(x.get('f3'))} · lock {_f(x.get('lockrep'))} · Δ {_f((x['lockrep'] - x['f3']) if x.get('f3') is not None and x.get('lockrep') is not None else None)}"
                             + ("" if x.get("final") or x.get("f3") is None else "ᵖ") for x in items[:10])
    if state == "collecting" and len(done) >= n and len(fin) < n:
        G["provisional"] = decide_tp3_lock([x["lockrep"] for x in done], [x["f3"] for x in done], n)[0]
    else:
        G.pop("provisional", None)
    return state, t0


def gate_atr_raise(orders, J, st, n=ATR_RAISE_N):
    """⬆ (250) the first n FRENZY_LONG / WIDE fills from the Oct-8 deploy with entry ATR in (2.5, 3.0] (live pnl %, closed prefix): mean < 0 →
    fired. Contrast: the ≤ 2.5 fills of the same period. Blocking reasons: journal ATR_HIGH refusals per UTC day (count only, not priced)."""
    G = st.setdefault("gates", {}).setdefault("ATR_RAISE", {})
    t0 = deploy_ms("FRENZY_OCT8")
    f = orders[orders.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (orders.o_ms >= t0)].copy()
    f["band"] = [atr_band(v) for v in f.entry_atr_pct]
    rz = f[f.band == "raise"].head(n)
    vals = _closed_prefix(rz)
    state, m = decide_atr_raise(vals, n)
    old = f[(f.band == "old") & (f.status.astype(str) == "CLOSED") & f.pnl_percentage.notna()]
    if len(rz) == n:                                                  # "the same period" = deploy → the n-th re-admitted fill
        old = old[old.o_ms <= int(rz.o_ms.iloc[-1])]
    tally = ""
    if J is not None and len(J) and "gate" in J:
        b = J[(J.e == "BLOCK") & J.gate.isin(list(ATR_BLOCK_GATES)) & (J.ms >= t0)].drop_duplicates(["ms", "pair", "gate"])
        if len(b):
            b = b.assign(day=[_fmt_t(x, True)[5:10] for x in b.ms])
            tally = " · ATR > 3.0 refusals (count only): " + ", ".join(
                f"{d_} {int((g_.gate == 'FRENZY_ATR_HIGH').sum())} LONG / {int((g_.gate == 'FRENZY_WIDE_ATR_HIGH').sum())} WIDE" for d_, g_ in b.groupby("day"))
        else:
            tally = " · ATR > 3.0 refusals: none in the journal yet"
    nu = int(f.band.isna().sum())
    G["progress"] = (f"{len(vals)}/{n} re-admitted (ATR 2.5–3.0) fills closed" + (f" · avg {m:+.2f} % · {sum(1 for v in vals if v > 0)} won" if vals else "")
                     + f" vs ≤ 2.5 same period {len(old)}" + (f" · avg {old.pnl_percentage.mean():+.2f} %" if len(old) else "")
                     + (f" · {len(rz) - len(vals)} later/open" if len(rz) > len(vals) else "") + (f" · {nu} fills without an ATR stamp" if nu else "") + tally)
    G["detail"] = " · ".join(f"{_fmt_t(r.o_ms)} {r.pair.replace('USDT', '')} {str(r.entry_strategy).replace('FRENZY_', '')[:4]} ATR {_f(r.entry_atr_pct, '.2f')} {_f(r.pnl_percentage)}"
                             for r in rz.head(10).itertuples())
    return state, t0


def gate_bearish_blocked(st):
    """🐻 (250) reads the FRENZY_BEARISH_BLOCKED store written by scripts/scout_frenzy_exits.py (counted ∧ final rows) → the frozen bar."""
    G = st.setdefault("gates", {}).setdefault("BEARISH_BLOCKED", {})
    t0 = deploy_ms("FRENZY_OCT8")
    if not os.path.exists(BB_CSV):
        G["progress"] = "no FRENZY_BEARISH_BLOCKED rows yet (scout_frenzy_exits writes them)"
        return "collecting", t0
    d = pd.read_csv(BB_CSV)
    T = lambda c: d[c].astype(str).isin(("True", "1", "1.0")) if c in d else pd.Series(False, index=d.index)
    c = d[T("counted") & (d.kind.astype(str) == "BLOCKED")] if "kind" in d else d.iloc[0:0]
    c = c.sort_values("k", kind="stable")
    v = c[T("verdict_set")[c.index]] if "verdict_set" in c else c.iloc[0:0]   # the frozen first-crossing cohort (written by scout_frenzy_exits)
    if len(v):
        state, n_, nd, m = decide_bearish_blocked(pd.to_numeric(v.PNL, errors="coerce").tolist(), v.day.tolist(), n=len(v))
    else:
        cf = c[T("final")[c.index]]
        _, n_, nd, m = decide_bearish_blocked(pd.to_numeric(cf.PNL, errors="coerce").tolist(), cf.day.tolist())
        state = "collecting"
    kp = d[T("counted") & T("final") & (d.kind.astype(str) == "KEPT")] if "kind" in d else d.iloc[0:0]
    km = pd.to_numeric(kp.PNL, errors="coerce").dropna() if len(kp) else pd.Series(dtype=float)
    prov = int((T("counted") & ~T("final")).sum())
    G["progress"] = (("VERDICT SET FROZEN (first crossing): " if len(v) else "") + f"{n_}/{BB_N} counted blocked signals · {nd}/{BB_DAYS} days" + (f" · mean {m:+.2f} % (as if opened, fixed +3/−3)" if n_ else "")
                     + f" · kept side {len(km)}" + (f" · mean {km.mean():+.2f} %" if len(km) else "") + (f" · {prov} provisional" if prov else ""))
    return state, t0


def _closed_prefix(f):
    """pnl % of the leading CLOSED fills (in open order) — a first-N set waits for an earlier fill that is still open."""
    out = []
    for r in f.itertuples():
        if str(r.status) != "CLOSED" or not np.isfinite(r.pnl_percentage):
            break
        out.append(float(r.pnl_percentage))
    return out


def gate_frenzy_strong(orders, st, n=10):
    G = st.setdefault("gates", {}).setdefault("FRENZY_STRONG", {})
    t0 = deploy_ms("FRENZY_STRONG")
    f = orders[(orders.entry_strategy.astype(str) == "FRENZY_LONG") & (orders.o_ms >= t0)].copy()
    f["sized"] = (pd.to_numeric(f.entry_frenzy_adx_delta, errors="coerce") > 0) & (pd.to_numeric(f.entry_frenzy_di_spread, errors="coerce") > 0)
    sz = f[f.sized].head(n)
    vals = _closed_prefix(sz)
    nm = f[~f.sized & (f.status.astype(str) == "CLOSED") & f.pnl_percentage.notna()]
    if len(sz) == n:                                                  # "the same period" = deploy → the 10th sized-up fill
        nm = nm[nm.o_ms <= int(sz.o_ms.iloc[-1])]
    state, ms, mn = decide_strong(vals, nm.pnl_percentage.tolist(), n)
    lev = pd.to_numeric(sz.leverage, errors="coerce")
    G["progress"] = (f"{len(vals)}/{n} sized-up fills closed" + (f" · avg {ms:+.2f} %" if vals else "")
                     + f" vs normal {len(nm)} · " + (f"avg {mn:+.2f} %" if len(nm) else "none yet")
                     + (f" · sized-up leverage seen {', '.join(sorted({f'{v:g}×' for v in lev.dropna()}))}" if lev.notna().any() else ""))
    G["detail"] = " · ".join(f"{_fmt_t(r.o_ms)} {r.pair.replace('USDT', '')} {'⬆' if r.sized else ''}{_f(r.pnl_percentage)}"
                             for r in f.head(12).itertuples())
    return state, t0


def gate_frenzy_gvol(orders, st, n=20):
    G = st.setdefault("gates", {}).setdefault("FRENZY_GVOL", {})
    t0 = deploy_ms("FRENZY_GVOL")
    f = orders[orders.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (orders.o_ms >= t0)].head(n)
    vals = _closed_prefix(f)
    state, m = decide_mean_neg(vals, n)
    G["progress"] = (f"{len(vals)}/{n} fills closed" + (f" · avg {m:+.2f} % · {sum(1 for v in vals if v > 0)} won" if vals else "")
                     + (f" · {len(f) - len(vals)} later/open" if len(f) > len(vals) else ""))
    G["detail"] = " · ".join(f"{_fmt_t(r.o_ms)} {r.pair.replace('USDT', '')} {str(r.entry_strategy).replace('FRENZY_', '')[:4]} {_f(r.pnl_percentage)}"
                             for r in f.head(10).itertuples())
    return state, t0


def decide_heat_admit(vals, n=10):
    """🔁 (208) revert-of-the-revert: the first n WINDOWS of momentum longs the Sep-25 re-scope WOULD have blocked — mean of the window means < 0 → 'fired'."""
    v = list(vals)[:n]
    if len(v) < n:
        return "collecting"
    return "fired" if float(np.mean(v)) < 0 else "holds"


def gate_heat_admit(orders, st, n=10):
    """momentum-LONG fills after the revert that the breadth-only re-scope would have refused: bull ≥ 85 ∧ NOT (BTC slope ≥ 0.07 ∧ BTC
    RSI prev ≥ 64) ∧ not washed out (BTC > −10 % vs its 30d high; unknown counts). Probes / MANUAL excluded."""
    G = st.setdefault("gates", {}).setdefault("HEAT_ADMIT", {})
    t0 = deploy_ms("HEAT_REVERT")
    nn = lambda c: pd.to_numeric(orders[c], errors="coerce") if c in orders else pd.Series(np.nan, index=orders.index)
    bull, slope, rsi, off = nn("entry_bull_pct"), nn("entry_btc_ema20_slope"), nn("entry_btc_rsi_prev"), nn("entry_btc_off30d_high_pct")
    src = orders.get("cell_multiplier_source", pd.Series("", index=orders.index)).fillna("").astype(str)
    m = ((orders.entry_strategy.fillna("MOMENTUM").astype(str) == "MOMENTUM") & (orders.direction.astype(str) == "LONG")
         & (orders.o_ms >= t0) & ~src.str.endswith("_PROBE") & (bull >= 85) & ~((slope >= 0.07) & (rsi >= 64)) & off.notna() & (off > -10))   # the re-scope failed OPEN on an unknown 30d reading
    f = orders[m.fillna(False)].sort_values("o_ms")
    # WINDOW units (review: every input is market-wide → fills in the same 5-min bucket are ONE observation); a closed prefix of windows
    wins, n_open = [], 0
    for b_, g in f.groupby(f.o_ms // (5 * MIN) * (5 * MIN), sort=True):
        if not ((g.status.astype(str) == "CLOSED") & g.pnl_percentage.notna()).all():
            n_open += 1
            break
        wins.append((int(b_), float(g.pnl_percentage.mean()), len(g)))
        if len(wins) >= n:
            break
    vals = [w[1] for w in wins]
    state = decide_heat_admit(vals, n)
    G["progress"] = (f"{len(vals)}/{n} windows closed ({sum(w[2] for w in wins)} fills)" + (f" · {sum(1 for v in vals if v > 0)} positive · mean of window means {np.mean(vals):+.3f} %" if vals else "")
                     + (" · a window still open" if n_open else ""))
    G["detail"] = " · ".join(f"{_fmt_t(t)} n{k} {_f(v)}" for t, v, k in wins)
    return state, t0


def decide_wide_choppy(vals, n=15):
    """🌀 (215) the frozen revert bar: the first n blocked WIDE signals re-priced with the live FRENZY exit average ≥ 0 → FIRED (switch the
    block off); < 0 → holds. Fewer than n priced → collecting."""
    if len(vals) < n:
        return "collecting"
    return "fired" if float(np.mean(vals[:n])) >= 0 else "holds"


def gate_wide_choppy(J, st, now_ms, n=15):
    """🌀 (215) FRENZY_WIDE_CHOPPY refusals (journal BLOCK lines; one per pair-episode) re-priced as if WIDE had opened: entry at the open of the
    first full minute after the refusal, the LIVE FRENZY exit (lock: −3 until +3, then max(+2, peak − 2)) on public 1m klines, net of fees, 12 h
    (scout_frenzy_exits.walk — the same walker as the exit-shadow table). FINAL once 12 h have passed."""
    import scout_frenzy_exits as FX
    G = st.setdefault("gates", {}).setdefault("WIDE_CHOPPY", {})
    store = G.setdefault("items", {})
    rows = J[(J.e == "BLOCK") & (J.gate.astype(str) == "FRENZY_WIDE_CHOPPY")][["ms", "pair"]] if J is not None and len(J) else pd.DataFrame(columns=["ms", "pair"])
    old = pd.DataFrame([dict(ms=int(v["t"]), pair=v["pair"]) for v in store.values()], columns=["ms", "pair"])
    ep = episodes(pd.concat([old, rows], ignore_index=True)) if (len(rows) or len(old)) else rows
    items = []
    for r in ep.sort_values("ms").head(n).itertuples():
        k = f"{r.pair}|{int(r.ms)}"
        it = store.get(k) or dict(pair=r.pair, t=int(r.ms))
        if not it.get("final"):
            try:
                t_in = (int(r.ms) // MIN + 1) * MIN
                m1 = [b for b in FX._kl(str(r.pair), "1m", t_in, min(now_ms, t_in + 721 * MIN)) if b[0] + MIN <= now_ms]
                if m1:
                    p, _x, how = FX.walk(m1, float(m1[0][1]), "LOCK2")
                    it.update(sim=p, how=how, final=bool(how != "open"))
            except Exception as e:
                log(f"WIDE_CHOPPY price {r.pair}: {str(e)[:80]}")
            store[k] = it
        items.append(it)
    vals = [x["sim"] for x in items if x.get("final") and x.get("sim") is not None]
    state = decide_wide_choppy(vals, n)
    pr = [x for x in items if x.get("sim") is not None]
    G["progress"] = (f"{len(vals)}/{n} blocked signals re-priced (final)" + (f" · {len(pr) - len(vals)} provisional" if len(pr) > len(vals) else "")
                     + (f" · {sum(1 for v in vals if v > 0)} would have won · mean {np.mean(vals):+.2f} %" if vals else ""))
    G["detail"] = " · ".join(f"{_fmt_t(x['t'])} {str(x['pair']).replace('USDT', '')} " + (_f(x.get("sim")) if x.get("sim") is not None else "pending")
                             + ("" if x.get("final") else "ᵖ") for x in items)
    return state


def gate_surge(orders, st, btc):   # RETIRED Oct-5 (DECISION_LOG 202) — the SURGE_LONG row is gate_surge_b
    G = st.setdefault("gates", {}).setdefault("SURGE_LONG", {})
    t0 = _ms(PROBE_START)
    f = orders[(orders.entry_strategy.astype(str) == "SURGE_LONG") & (orders.o_ms >= t0)].copy()
    f["trig"] = _ms_series(f.entry_surge_trigger_at).values
    f["trig"] = f.trig.where(np.isfinite(f.trig), f.o_ms)            # unstamped → the fill time (window = the fill)
    win, n_open = [], 0
    for trig, g in f.groupby("trig"):
        if not ((g.status.astype(str) == "CLOSED") & g.pnl_percentage.notna()).all():
            n_open += 1                                               # a window counts once every fill of it has closed
            continue
        r72 = btc_r72_at(btc, int(trig))
        win.append((r72, float(g.pnl_percentage.mean()), int(trig), len(g)))
    state, info = decide_surge([(r, m) for r, m, _, _ in win])
    G["progress"] = (f"{len(win)}/8 windows · BTC 3d ≤ +2.7 %: {info['n_lo']} (avg {_f(info['m_lo'])}, CI {_f(info['ci_lo'][0])}…{_f(info['ci_lo'][1])})"
                     f" · rest {info['n_hi']} (avg {_f(info['m_hi'])})"
                     + (f" · {sum(1 for r, *_ in win if r is None)} unreadable 3d return" if any(r is None for r, *_ in win) else "")
                     + (f" · {n_open} window(s) still open" if n_open else ""))
    G["detail"] = " · ".join(f"{_fmt_t(t)} 3d {_f(r, '+.1f')} n{k} {m:+.2f}" for r, m, t, k in win[:10])
    return state


def gate_surge_b(orders, st):
    """⚡ option B: SURGE_LONG triggers (fills grouped by entry_surge_trigger_at) opened after the B deploy; a trigger counts once all its fills closed."""
    G = st.setdefault("gates", {}).setdefault("SURGE_LONG", {})
    t0 = deploy_ms("SURGE_B")
    f = orders[(orders.entry_strategy.astype(str) == "SURGE_LONG") & (orders.o_ms >= t0)].copy()
    f["trig"] = _ms_series(f.entry_surge_trigger_at).values
    f["trig"] = f.trig.where(np.isfinite(f.trig), f.o_ms)
    done, n_open = [], 0
    for trig, g in f.sort_values("o_ms").groupby("trig", sort=True):
        if not ((g.status.astype(str) == "CLOSED") & g.pnl_percentage.notna()).all():
            n_open += 1
            break   # review: the FIRST 15 = a closed prefix — never let a later trigger take the place of one still open
        done.append((int(trig), float(g.pnl_percentage.mean()), len(g)))
    vals = [m for _, m, _ in done]
    state = decide_surge_b(vals)
    G["progress"] = (f"{min(len(vals), 15)}/15 filled triggers closed" + (f" · mean {np.mean(vals[:15]):+.3f} %/trigger" if vals else "")
                     + (f" · {sum(1 for v in vals[:15] if v > 0)} positive" if vals else "") + (f" · {n_open} still open" if n_open else ""))
    G["detail"] = " · ".join(f"{_fmt_t(t)} n{k} {m:+.2f}" for t, m, k in done[:15])
    return state, t0


def gate_bearrun(orders, st):
    G = st.setdefault("gates", {}).setdefault("BEARRUN", {})
    t0 = _ms(PROBE_START)
    f = orders[(orders.entry_strategy.astype(str) == "BEARRUN_SHORT") & (orders.o_ms >= t0)].copy()
    f = f.sort_values("o_ms")
    f["win"] = window_chain(f.o_ms.tolist(), 180 * MIN)
    wins = []
    for w, g in f.groupby("win"):
        done = (g.status.astype(str) == "CLOSED").all() and g.pnl_percentage.notna().all()
        wins.append((int(g.o_ms.min()), len(g), float(g.pnl_percentage.mean()) if done else None, float(g.pnl_percentage.sum()) if done else None))
    closed = [w for w in wins if w[2] is not None]                    # a window with a fill still open is not counted yet
    state = decide_bearrun([w[2] for w in closed], sum(w[3] for w in closed))
    G["progress"] = (f"{len(closed)}/5 windows closed · {sum(1 for w in closed if w[2] > 0)} positive · Σ {sum(w[3] for w in closed):+.2f} %"
                     + (f" · {len(wins) - len(closed)} window(s) still open" if len(wins) > len(closed) else ""))
    G["detail"] = " · ".join(f"{_fmt_t(t)} n{k} {_f(m)}" for t, k, m, _ in wins[:10])
    return state


def bearrun_5x_windows(f, newest_ms, gap_min=BR_GAP_MIN, lev_min=BR5_LEV_MIN, lev_max=BR5_LEV_MAX):
    """[(start_ms, n_fills, window-mean pnl % or None, complete, at_5x)] — the (200) row's 180-min chain on ALL BEARRUN fills (o_ms
    sorted); at_5x = every fill of the window has lev_min ≤ leverage < lev_max (a 1×-probe or a later 20× arm fill disqualifies it).
    complete = every fill closed AND the window can no longer grow (a later window exists, or the newest export is > gap past its last fill)."""
    f = f.sort_values("o_ms")
    if not len(f):
        return []
    f = f.assign(win=window_chain(f.o_ms.tolist(), gap_min * MIN))
    out, wins = [], list(f.groupby("win"))
    for j, (w, g) in enumerate(wins):
        closed = (g.status.astype(str) == "CLOSED").all() and g.pnl_percentage.notna().all()
        sealed = j < len(wins) - 1 or (newest_ms is not None and newest_ms - int(g.o_ms.max()) > gap_min * MIN)
        lv = pd.to_numeric(g.leverage, errors="coerce")
        at5 = bool(((lv >= lev_min) & (lv < lev_max)).all())
        out.append((int(g.o_ms.min()), len(g), float(g.pnl_percentage.mean()) if closed else None, bool(closed and sealed), at5))
    return out


def gate_bearrun_5x(orders, st, newest_ms):
    """🐻 (252) the tight rollback of the 5× override — first 3 complete 5× windows (start ≥ the deploy), Σ of window means, verdict frozen."""
    G = st.setdefault("gates", {}).setdefault("BEARRUN_5X", {})
    t0 = deploy_ms("SIZING_252")
    if G.get("frozen"):
        z = G["frozen"]
        G["progress"] = f"FROZEN at {BR5_WINDOWS} windows: Σ window means {z['sum']:+.2f} % · windows {', '.join(z['windows'])}"
        return z["state"], t0
    f = orders[(orders.entry_strategy.astype(str) == "BEARRUN_SHORT") & (orders.o_ms >= _ms(PROBE_START))].copy()   # chain = the (200) row's
    allw = bearrun_5x_windows(f, newest_ms)
    wins = [w for w in allw if w[0] >= t0 and w[4]]
    mixed = sum(1 for w in allw if w[0] >= t0 and not w[4])
    done = []
    for w in wins:                                                    # closed prefix — a later window never takes the place of an open one
        if not w[3]:
            break
        done.append(w)
    state, tot = decide_bearrun_5x([w[2] for w in done])
    if state != "collecting":
        G["frozen"] = {"state": state, "sum": tot, "windows": [f"{_fmt_t(w[0])} n{w[1]} {w[2]:+.2f}" for w in done[:BR5_WINDOWS]]}
    G["progress"] = (f"{min(len(done), BR5_WINDOWS)}/{BR5_WINDOWS} complete 5× windows · Σ window means {tot:+.2f} %"
                     + (f" · {sum(1 for w in done[:BR5_WINDOWS] if w[2] > 0)} positive" if done else "")
                     + (f" · {len(wins) - len(done)} window(s) still open / growing" if len(wins) > len(done) else "")
                     + (f" · ⚠ {mixed} window(s) from the deploy not all at 5× (excluded)" if mixed else ""))
    G["detail"] = " · ".join(f"{_fmt_t(t)} n{k} {_f(m_)}{'' if c else ' (open)'}{'' if a5 else ' (not 5×)'}" for t, k, m_, c, a5 in allw[-8:])
    return state, t0


def gate_fan_10x(orders, st):
    """🔄 (252) FAN flips at 10× from the deploy — first 15 closed (prefix), verdict frozen in the state."""
    G = st.setdefault("gates", {}).setdefault("FAN_10X", {})
    t0 = deploy_ms("SIZING_252")
    if G.get("frozen"):
        z = G["frozen"]
        G["progress"] = f"FROZEN at {FAN10_N}: WR {z['wr']:.0f} % · avg {z['avg']:+.3f} %"
        return z["state"], t0
    lev = pd.to_numeric(orders.leverage, errors="coerce")
    a = orders[(orders.entry_strategy.astype(str) == FLIP_SRC) & (orders.o_ms >= t0)]
    _al = pd.to_numeric(a.leverage, errors="coerce")
    hi = int(((_al > FAN10_LEV_MAX) | _al.isna()).sum())          # above 10× or leverage unreadable = not counted at 10×
    f = orders[(orders.entry_strategy.astype(str) == FLIP_SRC) & (orders.o_ms >= t0) & (lev <= FAN10_LEV_MAX)].head(FAN10_N)
    vals = _closed_prefix(f)
    state, wr, m = decide_fan_10x(vals)
    if state != "collecting":
        G["frozen"] = {"state": state, "wr": wr, "avg": m}
    G["progress"] = (f"{len(vals)}/{FAN10_N} FAN flips closed at ≤ 10×" + (f" · WR {wr:.0f} % · avg {m:+.3f} %" if vals else "")
                     + (f" · {len(f) - len(vals)} later/open" if len(f) > len(vals) else "") + (f" · ⚠ {hi} fill(s) not at ≤ 10× after the deploy (above 10× or leverage unreadable; excluded)" if hi else ""))
    G["detail"] = " · ".join(f"{_fmt_t(r.o_ms)} {r.pair.replace('USDT', '')} {_f(r.leverage, 'g')}× {_f(r.pnl_percentage)}" for r in f.head(15).itertuples())
    return state, t0


def _fb_fire_txt(g):
    """one counted fire for the detail list: '09-26 14:05 2Z 48.0 +0.52' (same-minute fills joined, value = their mean)."""
    names = "+".join(f"{p.replace('USDT', '')} {b:.1f}" for _, p, b, _ in g["rows"])
    val = _f(g["val"]) if g["closed"] else "open"
    return f"{_fmt_t(g['m'])} {names} {val}" + (f" (×{g['n']} same minute, mean)" if g["n"] > 1 else "")


def gate_fade_brsi(orders, st, n=FB_N):
    """🔓 (112) SPIKE_FADE fills from the 45 → 50 deploy with entry_btc_rsi in (45, 50], same-minute fires once, closed prefix by open time;
    verdict FROZEN in the state at the n-th counted fire (later fills / closes never rewrite it)."""
    G = st.setdefault("gates", {}).setdefault("FADE_BRSI50", {})
    t0 = deploy_ms("FADE_BRSI50")
    a = orders[(orders.entry_strategy.astype(str) == "SPIKE_FADE") & (orders.o_ms >= t0)].copy()
    a["band"] = [fade_brsi_band(v) for v in a.entry_btc_rsi]
    f = a[a.band == "band"]
    fires = fade_brsi_fires(f)
    caution = (f"⚠ caution (not part of the rule): Σ is thin — one more {str(format(FB_STOP, '+.1f')).replace('-', '−')} % stop can turn Σ < 0 and fire the gate even at ≥ 80 % WR")
    if G.get("frozen"):
        z = G["frozen"]
        later = [g for g in fires if g["m"] > z.get("last_m", 0)]
        G["progress"] = (f"FROZEN at {n}/{n} (on {z.get('frozen_at', '?')}): WR {z['wr']:.0f} % · Σ {z['sum']:+.2f} %"
                         + (f" · {len(later)} later fire(s) not counted" if later else ""))
        G["detail"] = f"frozen {n}: " + " · ".join(z["fires"]) + (f" · later (not counted): " + " · ".join(_fb_fire_txt(g) for g in later[-5:]) if later else "")
        return z["state"], t0
    pre = fade_brsi_prefix(fires)[:n]
    vals = [g["val"] for g in pre]
    state, wr, sm = decide_fade_brsi(vals, n)
    if state != "collecting":
        G["frozen"] = {"state": state, "wr": wr, "sum": sm, "fires": [_fb_fire_txt(g) for g in pre], "last_m": pre[-1]["m"],
                       "frozen_at": _fmt_t(int(time.time() * 1000), True)}
    nu = int(a.band.isna().sum())
    oth = a.band.value_counts()
    tail = len(fires) - len(pre)
    G["progress"] = (f"{len(pre)}/{n} fires counted (closed prefix, same-minute once)" + (f" · WR {wr:.0f} % · Σ {sm:+.2f} %" if pre else "")
                     + (f" · {tail} later/open" if tail > 0 else "")
                     + f" · other fades since the deploy: {int(oth.get('low', 0))} at ≤ 45 · {int(oth.get('above', 0))} above 50"
                     + (f" · ⚠ {nu} fade(s) with unreadable entry_btc_rsi (excluded)" if nu else "")
                     + (f" · {caution} (Σ headroom {sm:+.2f} %)" if state == "collecting" and pre else ""))
    G["detail"] = " · ".join(_fb_fire_txt(g) for g in fires[:15]) + f" · {caution}"
    return state, t0


# ═══════════════════════════════ assembly ═══════════════════════════════
DEFS = {
    "CHOP_BURST": ("🌀👥 Chop∧burst block (201)", "first 6 refused momentum-LONG signals (LONG_CHOP_BURST) re-priced with the live exit replica: "
                   "WR ≥ 50 % ∨ Σ > 0", "set long_chop_burst_block_enabled false", "long_chop_burst_block_enabled"),
    "FRENZY_TP3": ("🎯 FRENZY TP +3 (199)", "first 20 FRENZY + WIDE fills after the +3 deploy re-priced with fixed +4/−3 on ticks (bot accounting): "
                   "+4/−3 avg > actual avg", "set frenzy_tp_pct 4", "frenzy_tp_pct"),
    "FRENZY_STRONG": ("💪 FRENZY strong leverage (197)", "first 10 sized-up FRENZY_LONG fills (ADX Δ > 0 ∧ DI spread > 0) avg < the normal "
                      "FRENZY_LONG fills of the same period, or < 0 · ⚠ from 2026-10-08 the bearish-day block also filters entries — read the split at review", "set frenzy_long_lev_mult_strong 0", "frenzy_long_lev_mult_strong"),
    "FRENZY_GVOL": ("🌊 FRENZY market-volume gate (194)", "first 20 FRENZY + WIDE fills under the gate: avg pnl % < 0 · ⚠ from 2026-10-08 the bearish-day block also filters entries — read the split at review", "set frenzy_gvol_max 0",
                    "frenzy_gvol_max"),
    "FRENZY_LOCK": ("🔒 FRENZY lock exit (205 — SUPERSEDED Oct-8 by TP3_VS_LOCK; the lock is off, the row runs for the record)", "first 20 FRENZY + WIDE fills after the lock deploy, both exits re-priced on ticks with the bot's accounting: "
                    "the old fixed +3/−3 averages better than the lock (+2 at +3, trail 2 pts) → revert (also shown: +6/−3, +4/−3, live as traded)", "set frenzy_lock_arm_pct 0 (fixed +3 returns)",
                    "frenzy_lock_arm_pct"),
    "TP3_VS_LOCK": ("🎯 FRENZY fixed +3 vs the lock (250)", "first 20 FRENZY_LONG / WIDE / LITE fills from the Oct-8 deploy (live exit fixed +3/−3), both exits "
                    "re-priced on ticks with the bot's accounting (12 h): Σ lock (−3 → +2 at +3 → peak − 2) − Σ fixed +3/−3 > +3 % points → the runners the "
                    "lock keeps outweigh the +1 it gives back on each +3 touch", "REVIEW (operator decision): consider the lock again (frenzy_lock_arm_pct 3)",
                    "frenzy_lock_arm_pct"),
    "ATR_RAISE": ("⬆ FRENZY ATR cap 3.0 (250)", "first 15 FRENZY_LONG / WIDE fills from the Oct-8 deploy with entry ATR in (2.5, 3.0] (what the raise re-admitted; "
                  "live pnl %, closed prefix): average < 0 · contrast ≤ 2.5 fills same period · ATR > 3.0 refusals per day (count only)",
                  "REVERT (operator decision): frenzy_max_atr_pct back to 2.5", "frenzy_max_atr_pct"),
    "BEARISH_BLOCKED": ("🐻 FRENZY bearish-day block (250)", "FRENZY / WIDE / LITE bearish-day refusals (journal *_BEARISH_DAY) from the Oct-8 deploy, priced as if "
                        "opened (first print ≥ close + 8 s, fixed +3/−3, 0.09 % fees + 0.10 % slip, 12 h; one per pair-episode; DAY units) — store "
                        "reports/SCOUT_FRENZY_BEARISH_BLOCKED.csv (scout_frenzy_exits): verdict cohort FROZEN at the first crossing — the shortest final prefix "
                        "with ≥ 15 counted signals on ≥ 8 days (column verdict_set) — decided once: blocked mean > 0",
                        "REVERT (operator decision): turn frenzy_bearish_day_block off", "frenzy_bearish_day_block"),
    "SURGE_LONG": ("⚡ SURGE_LONG option B (202)", "trigger 0.3 % · 5× · market vol ≥ 1 · spacing after a fill, FULL size (operator override, "
                   "unproven: year +0.01 %/trigger): the first 15 triggers that filled, mean pnl %/trigger ≤ 0 → revert (supersedes the 200 probe gate)",
                   "set surge_long_lev_mult 0.05 (back to the probe)", "surge_long_lev_mult"),
    "BEARRUN": ("🐻 BEARRUN probe arm bar (200)", "≥ 5 windows started after 10-04 22:00 with fills, ≥ 3 positive ∧ Σ > 0 (positive event) · "
                "from 2026-10-08 (252) fills are at 5× (lev 0.25), not 1× — this gate still decides FULL size (1.0)",
                "set bearrun_lev_mult 1.0", "bearrun_lev_mult"),
    "BEARRUN_5X": ("🐻 BEARRUN 5× rollback (252)", "operator declared override 0.05 → 0.25 (5×) at 1 live window (replay 7 windows −0.24 %/window, 3/7 positive): "
                   "windows chained on ALL BEARRUN_SHORT fills as the (200) row (fills ≤ 180 min apart); counted = windows starting from the 252 deploy whose "
                   "fills are ALL at 5 ≤ leverage < 20 (a 1× probe or a later 20× arm never pollutes it); FROZEN at the first 3 complete windows: "
                   "Σ of the window-mean pnl % < 0 → rollback; else holds (the (200) gate still decides full size; kill bar unchanged)",
                   "ROLLBACK: bearrun_lev_mult 0.05", "bearrun_lev_mult"),
    "FAN_10X": ("🔄 FAN flips 10× (252)", "FLIP:FAN_RATIO_GATE fills from the 252 deploy at leverage ≤ 10 (closed prefix, pnl % = leverage-invariant), FROZEN at 15: "
                "WR ≥ 63 % ∧ avg ≥ +0.20 % → restore 20× · avg < 0 → REVIEW the sleeve (sleeve-kill checklist first; no auto-off) · else stay 10×",
                "RESTORE 20× (registry lev 1.0: flip_entry_sources FAN_RATIO_GATE:1.0)", "flip_entry_sources"),
    "FADE_BRSI50": ("🔓 Fade BTC-RSI ceiling 45 → 50 (112)", "TIGHT RE-REVERT, verbatim: \"fresh fades opened at BTC RSI [45,50) (entry_btc_rsi stamp) "
                    "at N≥10 — WR<55% ∨ Σ<0 → back to 45; same-minute fires count once.\" Read as the engine's band (45, 50] (fade blocked iff BTC 5m RSI "
                    "incl. the forming candle > spike_fade_max_btc_rsi, strict; the stamp is that reading): SPIKE_FADE fills from the c0d7fed deploy (+10 min), "
                    "one fire per open minute (value = mean pnl % of its fills), CLOSED prefix by open time, FROZEN at the 10th fire · ⚠ caution (not part "
                    "of the rule): Σ is thin — one more −1.5 % stop can flip Σ < 0 even at ≥ 80 % WR",
                    "REVERT: spike_fade_max_btc_rsi back to 45", "spike_fade_max_btc_rsi"),
    "LOADX": ("🧭 LOADX gate (126)", "first 30 LOADX-only refused LONG signals (journal FAILS, rank ≤ 10 excluded), WINDOW units: WR ≥ 60 % ∨ net > 0 · extended from 8 on 10-04 (first 8 had FIRED, fragile at t+5m)",
              "set long_rsi_momentum_adx_max 0", "long_rsi_momentum_adx_max"),
    "FLIP_EMA13_BLOCKED": ("🔄 FAN flip BTC-EMA13 filter (221)", "first 10 WINDOWS (5-min journal buckets) of FAN flip-short refusals whose COMPLETE fail set is "
                           "FLIP_FAN_BTC_EMA13 alone (BTC > its 5m EMA13 by more than flip_fan_btc_ema13_max; one signal per pair-episode), re-priced as if "
                           "the flip had opened at bucket +1 min (t+5m bracket) with the live FAN flip-short exit (ATR stop, short runner trail, HARD_TP "
                           "ladder, taker fees both sides): mean of window means > 0 → the filter blocked winners · forward windows from 2026-10-06 only; may never fill (EMA13 almost always co-fails)", "review flip_fan_btc_ema13_max → off",
                           "flip_fan_btc_ema13_max"),
    "FLIP_PADX_BLOCKED": ("🔄 FAN flip pair-ADX floor (221)", "first 10 WINDOWS (5-min journal buckets) of FAN flip-short refusals whose COMPLETE fail set is "
                          "FLIP_FAN_PAIR_ADX alone (pair ADX < flip_fan_pair_adx_min outside the exempt regimes; one signal per pair-episode), re-priced as "
                          "if the flip had opened at bucket +1 min (t+5m bracket) with the live FAN flip-short exit (ATR stop, short runner trail, HARD_TP "
                          "ladder, taker fees both sides): mean of window means > 0 → the filter blocked winners · forward windows from 2026-10-06 only", "review flip_fan_pair_adx_min → 0",
                          "flip_fan_pair_adx_min"),
    "MS_PVR_BLOCKED": ("📊 Mom-short pair-vol ceiling 0.86 (226)", "Sep-18 kept-side revert (< 70 % WR on 15 → 1.0) OVERRIDDEN (DECISION_LOG 226), "
                       "0.86 stays. ① REVERT to 1.0 if Cohort A — MOMENTUM_SHORT_PAIRVOL refusals (journal BLOCK, one per pair-episode) from 2026-10-06 "
                       "18:00 UTC with refusal PVR in [0.86, 1.0) (= what a revert re-admits), re-priced with the live momentum-short exit replica, final "
                       "prices, episodes a live fill of the pair followed ≤ 65 min excluded (double count) — reaches 15 signals with mean ≥ the kept side's "
                       "mean over the same period (kept fills since 10-06 18:00; < 5 → since 09-18 12:00). ② SLEEVE REVIEW (full sleeve-kill checklist, "
                       "no auto-kill) if the first 20 kept fills (PVR < 0.86, stack-kept) since 09-18 12:00 — keys frozen at N = 20 — are below "
                       "breakeven WR 59 %. Cohort B (PVR ≥ 1.0) = context. Scope caveat: the journal BLOCK line cannot tell a BEARRUN_SHORT refusal "
                       "(it rides the momentum ladder and this gate) from a momentum one — no stamp or log line separates them, so they are counted "
                       "together (BEARRUN uses the momentum exits). Logging note: PAIRVOL BLOCK lines appear only 09-28 → 09-30 in the exported journals "
                       "(through 10-06 04:00) while the adjacent open_position gate MOM_SHORT_C1_REGIME logged on 10-05 and kept fills on 10-05/10-06 "
                       "had PVR 0.43/0.61 — consistent with no ≥ 0.86 candidate reaching the gate, not a logging loss (no engine change since; worth a "
                       "server-log grep for [MOMENTUM_SHORT_PAIRVOL] to confirm)",
                       "① set momentum_short_pair_vol_max 1.0 · ② run the sleeve-kill checklist", "momentum_short_pair_vol_max"),
    "HEAT_ADMIT": ("🔁 Heat revert check (208)", "first 10 WINDOWS (5-min buckets) of momentum longs the breadth-only re-scope WOULD have blocked (bull ≥ 85, BTC slope/RSI not "
                   "both hot, not washed out), now admitted by the original 3-leg rule: mean pnl % < 0 → re-scope back", "set long_heat_btc_slope_min 0 · "
                   "long_heat_btc_rsi_prev_min 0 · long_heat_bull_pct_min 85", "long_heat_bull_pct_min"),
    "HEAT_ORIG": ("🔥 Heat original rule (208)", "first 15 WINDOWS of LONG_HEAT_BLOCK fires of the ORIGINAL rule (slope ≥ 0.07 ∧ RSI ≥ 64 ∧ bull ≥ 80) after the "
                  "revert, re-priced with the live exit: WR ≥ 60 % → the block itself is a review candidate", "review switching long_heat_block_enabled off",
                  "long_heat_block_enabled"),
    "HEAT": ("🫧 Heat re-scope (116)", "first 30 LONG_HEAT_BLOCK fires re-priced: WR ≥ 60 % (2nd leg — Jan–Jun replay expectancy — manual) · extended from 6 on 10-04 (first 6 had FIRED, 6/6 won)",
             "legs back to long_heat_btc_slope_min 0.07 · long_heat_btc_rsi_prev_min 64 · long_heat_bull_pct_min 80", "long_heat_bull_pct_min"),
    "WIDE_CHOPPY": ("🌀 WIDE choppy-pump block (215)", "first 15 FRENZY_WIDE_CHOPPY refusals (one per pair-episode) re-priced as if WIDE had opened — "
                    "the live FRENZY exit on 1m klines, entry at the next minute's open: mean ≥ 0 → the block removed winners", "set frenzy_wide_above_share_min 0",
                    "frenzy_wide_above_share_min"),
    "MEGACAP": ("🏦 Mega-cap exclusion (110)", "LONG_MEGACAP_BLOCK refusals re-priced: ≥ 60 % WR ∧ Σ > 0 on N ≥ 8 across ≥ 3 windows",
                "set long_megacap_rank_max 0", "long_megacap_rank_max"),
}
ORDER = ["CHOP_BURST", "BEARISH_BLOCKED", "ATR_RAISE", "TP3_VS_LOCK", "FRENZY_LOCK", "FRENZY_STRONG", "FRENZY_GVOL", "WIDE_CHOPPY", "SURGE_LONG", "BEARRUN", "BEARRUN_5X", "FAN_10X", "FADE_BRSI50", "LOADX", "FLIP_EMA13_BLOCKED", "FLIP_PADX_BLOCKED", "MS_PVR_BLOCKED", "HEAT", "HEAT_ADMIT", "HEAT_ORIG", "MEGACAP"]


def _status_text(code, state, G):
    action = DEFS[code][2]
    if state == "fired":
        return f"🔔 FIRED → {action}" + (" (t+5m entry disagrees — fragile)" if G.get("fragile") else "")
    if state == "arm_group":
        return "🔔 ARM BAR MET → normal size for the BTC 3d ≤ +2.7 % group only (needs a group-scoped size switch)"
    if state == "armbar":
        return f"🔔 ARM BAR MET → {action}"
    if state == "review" and code == "FAN_10X":
        return "🔔 REVIEW → the FAN flip sleeve (run the sleeve-kill checklist first; no auto-off)"
    if state == "holds" and code == "FAN_10X":
        return "✅ bar resolved — stay 10×"
    if state == "holds":
        return "✅ holds — bar resolved, keep" + (" (t+5m entry disagrees — fragile)" if G.get("fragile") else "")
    if state == "open":
        return "⏳ bar not met — keep probing (neither branch)"
    if state == "nodata":
        return "⚠ no data"
    if state == "error":
        return "⚠ error"
    prov = G.get("provisional")
    return "⏳ collecting" + (f" (provisional on 1m prices: would {'FIRE' if prov == 'fired' else 'hold'})" if prov else "")


def run_section(now_ms=None, noted=None, record_notes=True):
    """→ (markdown lines, [(note_key, note_line)]). Never raises (each gate guarded); persists reports/SCOUT_REVERT_GATES.json.
    Notes are once per gate-state for good: the key is also kept in this state file (the scout's own note keys expire after
    10 days). record_notes=False (the standalone CLI preview) leaves them for the scout run to deliver."""
    now_ms = int(now_ms or time.time() * 1000)
    noted = noted or {}
    st = load_state()
    st.setdefault("gates", {})
    res, cov = {}, {}
    try:
        orders, newest, n_ord = load_orders()
    except Exception as e:
        log(f"orders: {e}")
        orders, newest, n_ord = pd.DataFrame(columns=list(ORDER_COLS) + ["o_ms"]), None, 0
    try:
        J, jcov = load_journal()
    except Exception as e:
        log(f"journal: {e}")
        J, jcov = None, None
    try:
        ranks = rank_map(orders) if len(orders) else {}
    except Exception as e:
        log(f"ranks: {e}")
        ranks = {}
    jtxt = (f"journal {_fmt_t(jcov[0])}→{_fmt_t(jcov[1])} ({jcov[2]} exports{', ' + str(jcov[3]) + ' gaps > 1 h' if jcov[3] else ''})"
            if jcov else "no decisions export in ~/Downloads")
    otxt = f"orders to {_fmt_t(newest)} ({n_ord} exports)" if newest else "no orders export in ~/Downloads"
    btc = None
    try:
        btc = btc5m_array(now_ms - 12 * DAY)
    except Exception as e:
        log(f"btc: {e}")
    budget = Budget(PRICE_BUDGET_S)
    sig_specs = [("CHOP_BURST", 6, 50.0, "or", None, False), ("LOADX", EXT_N["LOADX"][0], 60.0, "or", ranks, True),
                 ("HEAT", EXT_N["HEAT"][0], 60.0, "wr", ranks, False), ("HEAT_ORIG", 15, 60.0, "wr", ranks, True)]   # window units (market-wide rule)
    # phase 1 — which tick days do the unpriced items need? fetch them once
    need = set()
    for code, n, wr, mode, rk, win in sig_specs:
        try:
            gate_first_n_signals(code, J, st, n, wr, mode, btc, budget, now_ms, need, rk, win)
        except Exception as e:
            log(f"{code} phase 1: {e}")
    for fcode in FLIP_GATES:                                  # 🔄 (221) own try each — a flip gate must never break the run
        try:
            gate_flip_blocked(fcode, J, st, budget, now_ms, need)
        except Exception as e:
            log(f"{fcode} phase 1: {e}")
    try:                                                      # 📊 own try — the momentum-short shadow must never break the run
        gate_ms_pvr(J, st, budget, now_ms, need)
    except Exception as e:
        log(f"MS_PVR_BLOCKED phase 1: {e}")
    for fn in (lambda: gate_megacap(J, st, btc, budget, now_ms, need), lambda: gate_frenzy_lock(orders, st, budget, now_ms, need),
               lambda: gate_tp3_vs_lock(orders, st, budget, now_ms, need)):
        try:
            fn()
        except Exception as e:
            log(f"phase 1: {e}")
    fetched = 0
    try:
        fetched = ensure_ticks(need, st, now_ms)
    except Exception as e:
        log(f"ticks: {e}")
    # phase 2 — price + decide
    for code, n, wr, mode, rk, win in sig_specs:
        try:
            if (J is None or not len(J)) and not st["gates"].get(code, {}).get("items"):
                res[code] = "nodata"
                st["gates"].setdefault(code, {})["progress"] = "no decisions export covers it"
            else:
                res[code] = gate_first_n_signals(code, J, st, n, wr, mode, btc, budget, now_ms, None, rk, win)
            cov[code] = jtxt + (f" · shipped {SHIPS[code]}: earlier refusals only in the EB logs" if code in SHIPS and jcov and
                                _ms(SHIPS[code]) < jcov[0] - DAY else "")
            if code == "LOADX":
                cov[code] += " · FAILS sets start ≈ 09-30 19:00"
        except Exception as e:
            log(f"{code}: {e}")
            res[code] = "error"
            st["gates"].setdefault(code, {})["progress"] = f"error: {str(e)[:100]}"
    for fcode in FLIP_GATES:                                  # 🔄 (221) FAN flip filters judged on what they BLOCK
        try:
            if (J is None or not len(J)) and not st["gates"].get(fcode, {}).get("items"):
                res[fcode] = "nodata"
                st["gates"].setdefault(fcode, {})["progress"] = "no decisions export covers it"
            else:
                res[fcode] = gate_flip_blocked(fcode, J, st, budget, now_ms, None)
            cov[fcode] = jtxt + " · FAILS sets start ≈ 09-30 19:00 · sole-blocker sets only"
        except Exception as e:
            log(f"{fcode}: {e}")
            res[fcode] = "error"
            st["gates"].setdefault(fcode, {})["progress"] = f"error: {str(e)[:100]}"
    try:                                                      # 📊 momentum-short pair-vol ceiling: kept-side revert + refused-signal shadow
        res["MS_PVR_BLOCKED"] = gate_ms_pvr(J, st, budget, now_ms, None)
        cov["MS_PVR_BLOCKED"] = (jtxt + " · refusals counted from 10-06 18:00 UTC (journal BLOCK lines exist from ≈ 09-28) · kept side: "
                                 + otxt + " + reports/MASTER_POOL_stacked.csv · " + _msp().PARITY
                                 + (" · ⚠ replica re-validate needed" if st["gates"].get("MS_PVR_BLOCKED", {}).get("revalidate") else ""))
    except Exception as e:
        log(f"MS_PVR_BLOCKED: {e}")
        res["MS_PVR_BLOCKED"] = "error"
        st["gates"].setdefault("MS_PVR_BLOCKED", {})["progress"] = f"error: {str(e)[:100]}"
    try:
        if SCRIPTS not in sys.path:
            sys.path.insert(0, SCRIPTS)
        res["WIDE_CHOPPY"] = gate_wide_choppy(J, st, now_ms)
        cov["WIDE_CHOPPY"] = jtxt + " · counts refusals from the 215 deploy"
    except Exception as e:
        log(f"WIDE_CHOPPY: {e}")
        res["WIDE_CHOPPY"] = "error"
        st["gates"].setdefault("WIDE_CHOPPY", {})["progress"] = f"error: {str(e)[:100]}"
    try:
        res["MEGACAP"] = gate_megacap(J, st, btc, budget, now_ms, None)
        cov["MEGACAP"] = jtxt + " · shipped 09-23: earlier refusals only in the EB logs"
    except Exception as e:
        log(f"MEGACAP: {e}")
        res["MEGACAP"] = "error"
        st["gates"].setdefault("MEGACAP", {})["progress"] = f"error: {str(e)[:100]}"
    for code, fn in (("FRENZY_LOCK", lambda: gate_frenzy_lock(orders, st, budget, now_ms, None)),
                     ("FRENZY_STRONG", lambda: gate_frenzy_strong(orders, st)), ("FRENZY_GVOL", lambda: gate_frenzy_gvol(orders, st)),
                     ("HEAT_ADMIT", lambda: gate_heat_admit(orders, st)),
                     ("TP3_VS_LOCK", lambda: gate_tp3_vs_lock(orders, st, budget, now_ms, None)),   # 🎯 (250)
                     ("ATR_RAISE", lambda: gate_atr_raise(orders, J, st)),                         # ⬆ (250)
                     ("BEARISH_BLOCKED", lambda: gate_bearish_blocked(st)),                        # 🐻 (250)
                     ("BEARRUN_5X", lambda: gate_bearrun_5x(orders, st, newest)),                 # 🐻 (252)
                     ("FAN_10X", lambda: gate_fan_10x(orders, st)),                                # 🔄 (252)
                     ("FADE_BRSI50", lambda: gate_fade_brsi(orders, st))):                         # 🔓 (112)
        try:
            state, t0 = fn()
            res[code] = state
            cov[code] = otxt + f" · counts fills opened ≥ {_fmt_t(t0)} (deploy = push + 10 min)"
            if newest is None or newest < t0:
                res[code] = "collecting"
                st["gates"][code]["progress"] = "no export covers it yet"
        except Exception as e:
            log(f"{code}: {e}")
            res[code] = "error"
            st["gates"].setdefault(code, {})["progress"] = f"error: {str(e)[:100]}"
    try:
        res["SURGE_LONG"], _t0 = gate_surge_b(orders, st)
        cov["SURGE_LONG"] = otxt + f" · counts triggers from {_fmt_t(_t0)} (deploy = push + 10 min)"
        if newest is None or newest < _t0:
            res["SURGE_LONG"] = "collecting"
            st["gates"]["SURGE_LONG"]["progress"] = "no export covers it yet"
    except Exception as e:
        log(f"SURGE_LONG: {e}")
        res["SURGE_LONG"] = "error"
        st["gates"].setdefault("SURGE_LONG", {})["progress"] = f"error: {str(e)[:100]}"
    for code, fn in (("BEARRUN", lambda: gate_bearrun(orders, st)),):
        try:
            res[code] = fn()
            cov[code] = otxt + " · windows from 10-04 22:00 UTC"
            if newest is None or newest < _ms(PROBE_START):
                st["gates"][code]["progress"] = "no export covers it yet (probe windows count from 10-04 22:00 UTC)"
        except Exception as e:
            log(f"{code}: {e}")
            res[code] = "error"
            st["gates"].setdefault(code, {})["progress"] = f"error: {str(e)[:100]}"
    ftp = st["gates"].get("FRENZY_LOCK", {}).get("items", {})
    if ftp and "FRENZY_LOCK" in cov:
        srcs = [x.get("src") for x in ftp.values() if x.get("src")]
        cov["FRENZY_LOCK"] += f" · paths: {sum(1 for s in srcs if s == 'tick')} tick / {sum(1 for s in srcs if s != 'tick')} 1m"
    for code in ("CHOP_BURST", "LOADX", "HEAT", "HEAT_ORIG", "MEGACAP", "MS_PVR_BLOCKED") + tuple(FLIP_GATES):
        if st["gates"].get(code, {}).get("priced"):
            cov[code] = cov.get(code, "") + " · " + st["gates"][code]["priced"]
    # render
    L = ["## ⏳ Revert gates (every open pre-committed revert / arm gate, tracked each run)", "",
         "Refused signals are re-priced with the live momentum-LONG exit replica (scripts/ml_exit_optimize.py) at the journal bucket "
         "+1 min (primary) and +5 min (in brackets); FRENZY fills on aggTrades ticks with the bot's accounting. A gate fires only on final "
         "(tick) prices; ᵖ = provisional 1m price. FAN flip-short refusals (🔄) are re-priced the same way with the live flip-short exit "
         "replica (scripts/flip_exit_replica.py); refused momentum shorts (📊) with the live momentum-short exit replica (scripts/ms_pvr_shadow.py). "
         "Decisions are the operator's — this table never changes config.", "",
         "| Gate | Definition (frozen) | Progress | Status | Config now | Data coverage |", "|---|---|---|---|---|---|"]
    notes = []
    for code in ORDER:
        G = st["gates"].setdefault(code, {})
        state = res.get(code, "error")
        prev = G.get("state")
        if state in ("fired", "armbar", "arm_group", "review") and prev != state:
            G["state_at"] = now_ms
        G["state"] = state
        title, d, action, ck = DEFS[code]
        stt = _status_text(code, state, G)
        cv = cfg_value(ck)
        shown = (f"first {EXT_N[code][1]} (frozen gate): {EXT_N[code][2]} · first {EXT_N[code][0]} (extension): {stt}"
                 if code in EXT_N and state not in ("nodata", "error") else stt)
        if code == "MS_PVR_BLOCKED" and state not in ("nodata", "error") and G.get("status_txt"):   # 📊 (226) both frozen gates
            shown = G["status_txt"]
        if code == "HEAT" and state not in ("nodata", "error"):   # 🔁 (208) resolved — the extension's blocked set is frozen at the revert
            shown = f"first {EXT_N['HEAT'][1]} (frozen gate): {EXT_N['HEAT'][2]} → ✅ REVERTED Oct-5 (DECISION_LOG 208) · tally frozen at the revert"
        L.append(f"| {title} | {d} | {G.get('progress', '–')} | {shown} | {ck} = {cv if cv is not None else '–'} | {cov.get(code, '–')} |")
        if state in ("fired", "armbar", "arm_group") + (("review",) if code in ("MS_PVR_BLOCKED", "FAN_10X") else ()) and code != "HEAT":   # HEAT resolved (reverted, 208): no further alerts
            k = f"RG|{code}|n{EXT_N[code][0]}|{state}" if code in EXT_N else f"RG|{code}|{state}"   # N in the key: the frozen-N alert must not mute the extension
            if k not in noted and k not in G.get("noted", []):
                notes.append((k, f"🔔 Revert gate {title}: {(shown if code == 'MS_PVR_BLOCKED' else stt[2:]).strip()} — {G.get('progress', '')}"))
                if record_notes:
                    G.setdefault("noted", []).append(k)
    L += [""]
    for code in ORDER:
        dt = st["gates"].get(code, {}).get("detail")
        if dt:
            L.append(f"- {DEFS[code][0]}: {dt}")
    L += ["", f"_Tick archives fetched this run: {fetched}. State: reports/SCOUT_REVERT_GATES.json._", ""]
    st["updated_utc"] = _fmt_t(now_ms, True)
    try:
        atomic_write(STATE_JSON, json.dumps(st, indent=1, default=str))
    except Exception as e:
        log(f"state write failed: {e}")
    return L, notes


# ═══════════════════════════════ self-test ═══════════════════════════════
def selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    # every deploy key the gates pass to deploy_ms must be a DEPLOYS key (10-09 review: a stray comment swallowed FRENZY_STRONG → KeyError → silent "error" row)
    import re as _re
    _src = open(os.path.abspath(__file__), encoding="utf-8").read()
    _used = set(_re.findall(r'deploy_ms\("([A-Z0-9_]+)"', _src))
    chk(_used and _used <= set(DEPLOYS), f"deploy keys missing from DEPLOYS: {sorted(_used - set(DEPLOYS))}")

    # CHOP_BURST — first 6, WR ≥ 50 ∨ Σ > 0
    chk(decide_first_n([0.3, -0.5, -0.5], 6, 50)[0] == "collecting", "chop: < 6 collects")
    chk(decide_first_n([0.3, 0.2, 0.1, -0.5, -0.5, -0.5], 6, 50)[0] == "fired", "chop: 3/6 = 50 % fires")
    chk(decide_first_n([0.9, -0.1, -0.1, -0.1, -0.1, -0.1], 6, 50)[0] == "fired", "chop: Σ > 0 fires at WR 17 %")
    chk(decide_first_n([0.1, 0.1, -0.5, -0.5, -0.5, -0.5], 6, 50)[0] == "holds", "chop: 2/6 Σ<0 holds")
    chk(decide_first_n([0.1, 0.1, -0.5, -0.5, -0.5, -0.5, 9, 9, 9], 6, 50)[0] == "holds", "chop: only the FIRST 6 count")
    # HEAT — WR ≥ 60 alone
    chk(decide_first_n([1, 1, 1, 1, -9, -9], 6, 60, or_sum_pos=False)[0] == "fired", "heat: 67 % fires even with Σ<0")
    chk(decide_first_n([1, 1, 1, -0.1, -0.1, -0.1], 6, 60, or_sum_pos=False)[0] == "holds", "heat: 50 % with Σ>0 holds")
    # LOADX — windows, WR ≥ 60 ∨ net > 0
    chk(decide_first_n([-0.1] * 7 + [1.0], 8, 60)[0] == "fired", "loadx: net > 0 fires")
    chk(decide_first_n([-0.1] * 8, 8, 60)[0] == "holds", "loadx: all losers holds")
    # MEGACAP — cumulative N ≥ 8, ≥ 3 windows, WR ≥ 60 ∧ Σ > 0
    chk(decide_megacap([1] * 8, [1, 1, 1, 1, 2, 2, 2, 2])[0] == "collecting", "mega: 2 windows not enough")
    chk(decide_megacap([1] * 5 + [-0.1] * 3, [1, 2, 3, 1, 2, 3, 1, 2])[0] == "fired", "mega: 62.5 % Σ>0 3 windows fires")
    chk(decide_megacap([5] * 4 + [-0.1] * 4, [1, 2, 3, 4] * 2)[0] == "collecting", "mega: 50 % does not fire")
    chk(decide_megacap([1] * 7, [1, 2, 3, 4, 5, 6, 7])[0] == "collecting", "mega: N 7 collects")
    # FRENZY TP — +4/−3 beats actual on average over the first 20
    chk(decide_tp([1.0] * 19, [2.0] * 19)[0] == "collecting", "tp: 19 collects")
    chk(decide_tp([0.5] * 20, [0.6] * 20)[0] == "fired", "tp: alt better fires")
    chk(decide_tp([0.5] * 20, [0.5] * 20)[0] == "holds", "tp: tie holds (must BEAT)")
    chk(decide_tp([0.5] * 20 + [-9] * 5, [0.4] * 20 + [9] * 5)[0] == "holds", "tp: only the first 20")
    # FRENZY strong — first 10 sized-up < normal or < 0
    chk(decide_strong([1.0] * 9, [0.5])[0] == "collecting", "strong: 9 collects")
    chk(decide_strong([0.4] * 10, [0.5] * 5)[0] == "fired", "strong: below normal fires")
    chk(decide_strong([-0.1] * 10, [])[0] == "fired", "strong: below 0 fires with no normal fills")
    chk(decide_strong([0.6] * 10, [0.5] * 5)[0] == "holds", "strong: above normal and > 0 holds")
    chk(decide_strong([0.6] * 10, [])[0] == "holds", "strong: > 0, no normal fills holds")
    # FRENZY gvol — first 20 avg < 0
    chk(decide_mean_neg([-0.1] * 19)[0] == "collecting", "gvol: 19 collects")
    chk(decide_mean_neg([-0.1] * 20)[0] == "fired", "gvol: negative fires")
    chk(decide_mean_neg([0.0] * 20)[0] == "holds", "gvol: zero holds (bar is < 0)")
    # SURGE_LONG — ≥ 8 windows, split at BTC 3d ≤ +2.7
    w = [(1.0, 0.5), (2.0, 0.6), (0.5, 0.4), (2.7, 0.7), (3.5, -0.2), (4.0, -0.1), (5.0, -0.3), (1.5, 0.55)]
    chk(decide_surge(w[:7])[0] == "collecting", "surge: 7 windows collects")
    chk(decide_surge(w)[0] == "arm_group", "surge: low group + CI > 0, rest ≤ 0 → arm group (2.7 inclusive)")
    chk(decide_surge([(1.0, -0.2), (2.0, 0.1), (0.5, -0.3)] + [(4.0, 0.5)] * 5)[0] == "fired", "surge: low group ≤ 0 → LONG off")
    chk(decide_surge([(1.0, 0.5)] * 5 + [(4.0, 0.5)] * 3)[0] == "open", "surge: rest > 0 → neither branch")
    chk(decide_surge([(None, 0.5)] + w[1:])[0] == "collecting", "surge: unreadable 3d return never guesses a group")
    chk(decide_surge([(4.0, 0.5)] * 8)[0] == "open", "surge: no low-group windows → open")
    # BEARRUN — ≥ 5 windows, ≥ 3 positive ∧ Σ > 0
    chk(decide_bearrun([0.1, 0.2, 0.3, -0.1], 0.5) == "collecting", "bear: 4 windows collects")
    chk(decide_bearrun([0.1, 0.2, 0.3, -0.1, -0.2], 0.3) == "armbar", "bear: 3/5 positive Σ>0 → arm bar")
    chk(decide_bearrun([0.1, 0.2, 0.3, -1.0, -1.0], -1.4) == "open", "bear: Σ<0 → not met")
    chk(decide_bearrun([0.1, 0.2, -0.3, -0.1, -0.2], 0.1) == "open", "bear: 2 positive → not met")
    # helpers
    chk(window_chain([0, 60, 400, 401], 180) == [0, 0, 1, 1], "window chain")
    ep = episodes(pd.DataFrame(dict(ms=[0, 10 * MIN, 70 * MIN, 135 * MIN, 0], pair=["A", "A", "A", "A", "B"])))
    chk(list(zip(ep.pair, ep.ms)) == [("A", 0), ("B", 0), ("A", 135 * MIN)], "episodes: chained ≤ 60 min = one signal")
    btc = np.array([[i * 5 * MIN, 1, 1, 1, 100.0 + (3 if i == 900 else 0)] for i in range(901)])
    chk(abs(btc_r72_at(btc, 900 * 5 * MIN + 5 * MIN) - 3.0) < 1e-9, "btc r72 at the trigger bar (close-stamped)")
    chk(btc_r72_at(btc, 10 * 5 * MIN) is None, "btc r72 unreadable without 3 days of history")
    of = pd.DataFrame(dict(status=["CLOSED", "CLOSED", "OPEN", "CLOSED"], pnl_percentage=[1.0, -2.0, np.nan, 3.0]))
    chk(_closed_prefix(of) == [1.0, -2.0], "first-N waits behind an earlier open fill")
    chk(_is_final(["tick"], 0, 1) and not _is_final(["1m"], 0, 1) and _is_final(["1m"], 0, FINAL_AFTER_MS + 1), "final rule")
    # status text / notes are once per gate-state
    chk(_status_text("CHOP_BURST", "fired", {}).startswith("🔔 FIRED → set long_chop_burst_block_enabled false"), "status text")
    chk(EXT_N["LOADX"][0] == 30 and EXT_N["HEAT"][0] == 30, "extension: trackers at 30")
    chk(f"RG|HEAT|n{EXT_N['HEAT'][0]}|fired" != "RG|HEAT|fired", "extension: note key differs from the frozen-N key")
    chk(decide_surge_b([0.5] * 14) == "collecting", "surge B: < 15 collects")
    chk(decide_surge_b([-0.1] * 15) == "fired", "surge B: mean ≤ 0 fires")
    chk(decide_surge_b([1.0] + [-0.05] * 14) == "holds", "surge B: mean > 0 holds")
    chk(decide_surge_b([-1.0] * 15 + [9.0] * 5) == "fired", "surge B: only the FIRST 15 count")
    chk(decide_lock([0.5] * 19, [0.1] * 19)[0] == "collecting", "lock: < 20 collects")
    chk(lock_exit(np.array([-0.09, 1.0, 3.2, 5.0, 4.0, 2.9]), 3, 2, 2, 3) == (2.9, "TRAIL"), "lock path: +5 peak → trail line +3 → exit 2.9")
    chk(lock_exit(np.array([-0.09, 2.9, 1.0, -3.05]), 3, 2, 2, 3) == (-3.05, "SL"), "lock path: never armed → −3 stop")
    chk(lock_exit(np.array([-0.09, 3.0, 2.5, 1.9]), 3, 2, 2, 3) == (1.9, "TRAIL"), "lock path: armed at +3 → floor +2 → exit 1.9")
    chk(lock_exit(np.array([-0.09, 3.0, 2.5, 1.9]), 3, 2, 2, 3, tick=False) == (2.0, "TRAIL"), "lock path: 1m fallback fills at the line")
    chk(lock_exit(np.array([-0.09, 1.0]), 3, 2, 2, 3, complete=False) is None, "lock path: running → None")
    chk(decide_lock([0.2] * 20, [0.3] * 20)[0] == "fired", "lock: fixed +3 better fires")
    chk(decide_lock([0.4] * 20, [0.3] * 20)[0] == "holds", "lock: lock better holds")
    chk(decide_heat_admit([0.3] * 9) == "collecting", "heat admit: < 10 collects")
    chk(decide_heat_admit([0.3] * 5 + [-0.9] * 5) == "fired", "heat admit: mean < 0 fires")
    chk(decide_heat_admit([0.3] * 6 + [-0.2] * 4) == "holds", "heat admit: mean > 0 holds")
    chk(decide_wide_choppy([0.5] * 14) == "collecting", "WIDE_CHOPPY: 14 signals → collecting")
    chk(decide_wide_choppy([-3.0] * 10 + [2.0] * 5) == "holds", "WIDE_CHOPPY: blocked set losing → the block holds")
    chk(decide_wide_choppy([2.0] * 8 + [-3.0] * 5 + [0.5] * 2) == "fired", "WIDE_CHOPPY: blocked set ≥ 0 → FIRED (switch off)")
    chk(decide_wide_choppy([0.0] * 15) == "fired", "WIDE_CHOPPY: exactly 0 fires (the bar is ≥ 0)")
    # 🔄 (221) FAN flip filters judged on the BLOCKED signals — first 10 windows, mean of window means > 0 fires
    chk(decide_flip_blocked([0.5] * 9) == "collecting", "flip blocked: 9 windows → collecting")
    chk(decide_flip_blocked([0.3] * 5 + [-0.2] * 5) == "fired", "flip blocked: mean > 0 → FIRED (blocked winners)")
    chk(decide_flip_blocked([0.0] * 10) == "holds", "flip blocked: exactly 0 holds (the bar is > 0)")
    chk(decide_flip_blocked([-0.9] * 10 + [5.0] * 5) == "holds", "flip blocked: only the FIRST 10 windows count")
    Jt = pd.DataFrame(dict(
        t=["x"] * 7, ms=[0, 0, 5 * MIN, 10 * MIN, 15 * MIN, 20 * MIN, 25 * MIN],
        e=["FAILS", "FAILS", "FAILS", "FAILS", "BLOCK", "FAILS", "FAILS"],
        pair=["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT", "FUSDT", "GUSDT"],
        dir=["SHORT", "SHORT", "SHORT", "LONG", "SHORT", "SHORT", "SHORT"],
        gate=["FLIP_FAN_BTC_EMA13", "FLIP_FAN_BTC_EMA13+FLIP_FAN_PAIR_ADX", "FLIP_FAN_PAIR_ADX", "FLIP_FAN_BTC_EMA13",
              "FLIP_FAN_BTC_EMA13", "FLIP_FAN_BTC_EMA13", "FLIP_FAN_BTC_EMA13_X"],
        src=[FLIP_SRC, FLIP_SRC, FLIP_SRC, FLIP_SRC, FLIP_SRC, "FLIP:PAIR_RSI_OB", FLIP_SRC]))
    chk(list(flip_sole_blocker(Jt, "FLIP_FAN_BTC_EMA13").pair) == ["AUSDT"], "flip sole-blocker: exact set · SHORT · FAILS · FAN src only")
    chk(list(flip_sole_blocker(Jt, "FLIP_FAN_PAIR_ADX").pair) == ["CUSDT"], "flip sole-blocker: pADX alone (a joint EMA13+pADX set is excluded)")
    chk(list(_signals_for(Jt, "FLIP_EMA13_BLOCKED").pair) == ["AUSDT"], "flip signals: routed through the sole-blocker cut")
    chk(ORDER.index("FLIP_EMA13_BLOCKED") == ORDER.index("LOADX") + 1 and all(c in DEFS for c in FLIP_GATES), "flip gates wired (ORDER / DEFS)")
    # 📊 MS_PVR_BLOCKED — BLOCK lines of the pair-volume gate only, SHORT only, one per pair-episode; cohort stats by PVR class
    Jm = pd.DataFrame(dict(t=["x"] * 5, ms=[0, 20 * MIN, 0, 0, 90 * MIN], e=["BLOCK", "BLOCK", "BLOCK", "FAILS", "BLOCK"],
                           pair=["AUSDT", "AUSDT", "BUSDT", "CUSDT", "AUSDT"], dir=["SHORT", "SHORT", "LONG", "SHORT", "SHORT"],
                           gate=[MS_PVR_GATE, MS_PVR_GATE, MS_PVR_GATE, MS_PVR_GATE, MS_PVR_GATE], src=[None] * 5))
    chk(list(zip(ms_pvr_signals(Jm).pair, ms_pvr_signals(Jm).ms)) == [("AUSDT", 0), ("AUSDT", 90 * MIN)],
        "ms pvr signals: BLOCK · SHORT · pair-episodes (20-min repeat folded, 90-min repeat new)")
    its = [dict(t=0, cls="A", sim1=0.4), dict(t=DAY, cls="A~", sim1=-0.6), dict(t=DAY, cls="B", sim1=-0.2), dict(t=0, cls="unk", sim1=1.0),
           dict(t=0, cls="A", sim1=None)]
    n_, d_, w_, m_ = ms_cohort_stats(its, ("A", "A~"))
    chk((n_, d_, round(w_), round(m_, 3)) == (2, 2, 50, -0.1), "ms cohort A: A + A~ priced, days, WR, mean (unk / unpriced excluded)")
    chk(ms_cohort_stats(its, ("B", "B~"))[0] == 1 and ms_cohort_stats([], ("A",))[0] == 0, "ms cohort B / empty")
    chk(ms_cohort_stats(its + [dict(t=0, cls="A", sim1=5.0, dup=True)], ("A", "A~"))[0] == 2, "ms cohort: double-count episodes excluded")
    chk(ms_cohort_stats([dict(t=0, cls="A", sim1=1.0, final=False), dict(t=0, cls="A", sim1=-1.0, final=True)], ("A",), final_only=True)[3] == -1.0,
        "ms cohort: final_only drops provisional prices")
    fl = {"AUSDT": [10 * MIN, 500 * MIN]}
    chk(ms_followed_by_fill(0, "AUSDT", fl) and not ms_followed_by_fill(100 * MIN, "AUSDT", fl) and not ms_followed_by_fill(0, "BUSDT", fl)
        and not ms_followed_by_fill(11 * MIN, "AUSDT", fl), "ms dup: a fill of the same pair within bucket + 60 min (after the refusal only)")
    rw = [("k1", "CLOSED", 0.1), ("k2", "OPEN", None), ("k3", "CLOSED", -0.2)]
    chk(ms_review_first(rw, None, 2) == ([("k1", 0.1)], False), "ms review: a still-open fill holds the first-N set back")
    chk(ms_review_first([("k1", "CLOSED", 0.1), ("k2", "CLOSED", -0.2), ("k3", "CLOSED", 0.3)], None, 2) == ([("k1", 0.1), ("k2", -0.2)], True),
        "ms review: first N closed in open order, frozen flag set")
    chk(ms_review_first([("k9", "CLOSED", 9.0)], [["k1", 0.1], ["k2", -0.2]], 2) == ([("k1", 0.1), ("k2", -0.2)], True),
        "ms review: a frozen set wins over later data")
    chk(MS_PVR_REG_MS == _ms("2026-10-06 18:00") and "MS_PVR_BLOCKED" in ORDER and "MS_PVR_BLOCKED" in DEFS, "ms pvr: floor + wiring")
    # ── Oct-8 (250) ──
    chk(decide_atr_raise([0.5] * 14)[0] == "collecting", "atr raise: 14 collects")
    chk(decide_atr_raise([-0.1] * 15)[0] == "fired" and decide_atr_raise([0.0] * 15)[0] == "holds", "atr raise: mean < 0 fires, 0 holds")
    chk(decide_atr_raise([1.0] * 15 + [-9.0] * 5)[0] == "holds", "atr raise: only the FIRST 15 count")
    chk([atr_band(v) for v in (2.5, 2.5001, 3.0, 3.0001, None, "x", float("nan"), 1.0)] == ["old", "raise", "raise", "above", None, None, None, "old"],
        "atr band: (2.5, 3.0] = re-admitted, edges exact")
    chk(decide_tp3_lock([1.0] * 19, [0.0] * 19)[0] == "collecting", "tp3 vs lock: 19 collects")
    chk(decide_tp3_lock([3.0] * 20, [2.8] * 20) == ("fired", decide_tp3_lock([3.0] * 20, [2.8] * 20)[1]) and decide_tp3_lock([3.0] * 20, [2.8] * 20)[1] > 3,
        "tp3 vs lock: Σ Δ +4 > +3 fires")
    chk(decide_tp3_lock([3.0] * 20, [2.85] * 20)[0] == "holds", "tp3 vs lock: Σ Δ +3.0 holds (bar is > +3)")
    chk(decide_tp3_lock([0.0] * 20 + [99] * 5, [0.0] * 25)[0] == "holds", "tp3 vs lock: only the first 20")
    dd = [f"d{i}" for i in range(8)] * 2
    chk(decide_bearish_blocked([0.1] * 14, dd[:14])[0] == "collecting", "bearish: 14 collects")
    chk(decide_bearish_blocked([0.1] * 15, ["d0"] * 15)[0] == "collecting", "bearish: 15 on 1 day collects (≥ 8 days)")
    chk(decide_bearish_blocked([0.1] * 16, dd)[0] == "fired", "bearish: mean > 0 on 16/8 → revert")
    chk(decide_bearish_blocked([0.0] * 16, dd)[0] == "holds" and decide_bearish_blocked([-1.0] * 16, dd)[0] == "holds", "bearish: mean ≤ 0 holds")
    chk(decide_bearish_blocked([None, float("nan")] + [0.1] * 16, ["x", "y"] + dd)[1] == 16, "bearish: unpriced values ignored")
    rw = [(f"k{i}", f"d{i % 8}", True) for i in range(20)]
    chk(bb_first_crossing(rw[:14]) is None and bb_first_crossing(rw) == [f"k{i}" for i in range(15)], "bearish freeze: the first 15 once ≥ 8 days are covered")
    rw2 = [(f"k{i}", "d0" if i < 14 else f"d{i}", True) for i in range(25)]
    chk(len(bb_first_crossing(rw2)) == 21, "bearish freeze: 14 on one day → the prefix grows until the 8th day (21 signals)")
    chk(bb_first_crossing([(k, d, (k != "k3")) for k, d, _ in rw]) is None, "bearish freeze: a provisional row inside the prefix holds the freeze")
    chk(decide_bearish_blocked([0.1] * 21, [r[1] for r in rw2[:21]], n=21)[0] == "fired", "bearish: the frozen set is decided as a whole")
    chk(all(c in ORDER and c in DEFS for c in ("BEARISH_BLOCKED", "ATR_RAISE", "TP3_VS_LOCK")) and "FRENZY_OCT8" in DEPLOYS and "SUPERSEDED" in DEFS["FRENZY_LOCK"][0],
        "Oct-8 gates wired (ORDER / DEFS / deploy key) · 205 marked superseded")
    _orig = DEPLOYS["FRENZY_OCT8"]
    DEPLOYS["FRENZY_OCT8"] = ("grep:(NO SUCH COMMIT 0xdeadbeef)", None)
    _t = time.time() * 1000
    chk(abs(deploy_ms("FRENZY_OCT8") - _t) < 60_000, "Oct-8 deploy: no commit found → the floor is NOW (no pinned guess)")
    DEPLOYS["FRENZY_OCT8"] = _orig
    chk(_status_text("ATR_RAISE", "fired", {}).startswith("🔔 FIRED → REVERT (operator decision): frenzy_max_atr_pct back to 2.5"), "atr raise status text")
    # ── Oct-8 (252) ──
    chk(decide_bearrun_5x([0.5, -0.2])[0] == "collecting", "br5x: 2 windows collect")
    chk(decide_bearrun_5x([0.5, -0.2, -0.4]) == ("fired", decide_bearrun_5x([0.5, -0.2, -0.4])[1]) and decide_bearrun_5x([0.5, -0.2, -0.4])[1] < 0, "br5x: Σ < 0 → rollback")
    chk(decide_bearrun_5x([0.5, -0.2, -0.3])[0] == "holds" and decide_bearrun_5x([0.0, 0.0, 0.0])[0] == "holds", "br5x: Σ ≥ 0 holds (bar is < 0)")
    chk(decide_bearrun_5x([0.1, 0.1, 0.1, -9, -9])[0] == "holds", "br5x: only the FIRST 3 windows count")
    _o = pd.DataFrame({"o_ms": [0, 60 * MIN, 400 * MIN, 900 * MIN], "status": ["CLOSED"] * 3 + ["OPEN"], "pnl_percentage": [0.4, 0.6, -0.3, np.nan],
                       "leverage": [5, 5, 1, 5]})
    _w = bearrun_5x_windows(_o, 900 * MIN + 1)
    chk([(w[1], w[3], w[4]) for w in _w] == [(2, True, True), (1, True, False), (1, False, True)] and abs(_w[0][2] - 0.5) < 1e-9,
        "br5x windows: 180-min chain on all fills, window MEAN pnl %, 1× window flagged not-5×, open window incomplete")
    _w2 = bearrun_5x_windows(_o.assign(leverage=[5, 20, 5, 5]), 900 * MIN + 1)
    chk(_w2[0][4] is False, "br5x windows: a 20× (armed) fill disqualifies its window")
    _bo = pd.DataFrame({"o_ms": [DAY + k * 400 * MIN for k in range(5)], "entry_strategy": ["BEARRUN_SHORT"] * 5, "leverage": [1, 5, 20, 5, 5],
                        "status": ["CLOSED"] * 5, "pnl_percentage": [5.0, 0.4, 9.0, -0.3, -0.2]})
    _orig2 = (DEPLOYS["SIZING_252"], PROBE_START)
    DEPLOYS["SIZING_252"] = ("grep:(NO SUCH COMMIT 0xdeadbeef)", "1970-01-01 00:00:00")
    try:
        globals()["PROBE_START"] = "1970-01-01 00:00"
        _st2 = {}
        _r = gate_bearrun_5x(_bo, _st2, DAY + 3000 * MIN)
        _pg = _st2["gates"]["BEARRUN_5X"]["progress"]
        chk(_r[0] == "fired" and abs(_st2["gates"]["BEARRUN_5X"]["frozen"]["sum"] + 0.1) < 1e-9 and "2 window(s) from the deploy not all at 5×" in _pg,
            "br5x gate: the 1× and 20× windows are skipped; 3 × 5× windows Σ means −0.1 → rollback, frozen")
        chk(gate_bearrun_5x(_bo.iloc[:0], _st2, DAY)[0] == "fired", "br5x gate: the frozen verdict is reused")
    finally:
        DEPLOYS["SIZING_252"], globals()["PROBE_START"] = _orig2
    chk(bearrun_5x_windows(_o.iloc[:1], 100 * MIN)[0][3] is False and bearrun_5x_windows(_o.iloc[:1], 200 * MIN)[0][3] is True,
        "br5x windows: the LAST window is complete only once it can no longer grow (> 180 min past its last fill)")
    chk(decide_fan_10x([0.5] * 14)[0] == "collecting", "fan10: 14 collects")
    chk(decide_fan_10x([0.5] * 10 + [-0.2] * 5)[0] == "fired", "fan10: WR 67 % ∧ avg +0.27 → restore 20×")
    chk(decide_fan_10x([0.5] * 9 + [-0.2] * 6)[0] == "holds", "fan10: WR 60 % < 63 → stay 10×")
    chk(decide_fan_10x([0.3] * 15)[0] == "fired" and decide_fan_10x([0.19] * 15)[0] == "holds", "fan10: avg +0.20 is the line (0.19 stays)")
    chk(decide_fan_10x([0.2] * 10 + [-1.0] * 5)[0] == "review", "fan10: avg < 0 → review (even at WR 67 %)")
    chk(decide_fan_10x([0.2] * 15 + [-9.0] * 5)[0] == "fired", "fan10: only the FIRST 15 count")
    _fo = pd.DataFrame({"o_ms": [DAY, DAY + 1, DAY + 2], "entry_strategy": [FLIP_SRC] * 3, "leverage": [10, 20, 10], "status": ["CLOSED"] * 3,
                        "pnl_percentage": [0.5, -1.0, 0.3], "pair": ["AUSDT", "BUSDT", "CUSDT"]})
    _orig = DEPLOYS["SIZING_252"]
    DEPLOYS["SIZING_252"] = ("grep:(NO SUCH COMMIT 0xdeadbeef)", "1970-01-01 00:00:00")
    try:
        _st = {}
        chk(gate_fan_10x(_fo, _st)[0] == "collecting" and _st["gates"]["FAN_10X"]["progress"].startswith("2/15")
            and "1 fill(s) not at ≤ 10×" in _st["gates"]["FAN_10X"]["progress"], "fan10 gate: the 20× fill is excluded and flagged")
        _st = {}
        gate_fan_10x(_fo.assign(leverage=[10, np.nan, 10]), _st)
        chk("1 fill(s) not at ≤ 10×" in _st["gates"]["FAN_10X"]["progress"], "fan10 gate: a NaN-leverage fill is flagged too")
    finally:
        DEPLOYS["SIZING_252"] = _orig
    chk(all(c in ORDER and c in DEFS for c in ("BEARRUN_5X", "FAN_10X")) and "SIZING_252" in DEPLOYS and "5×" in DEFS["BEARRUN"][1],
        "252 gates wired (ORDER / DEFS / deploy key) · the (200) row notes the 5× fills")
    chk(_status_text("BEARRUN_5X", "fired", {}).startswith("🔔 FIRED → ROLLBACK: bearrun_lev_mult 0.05"), "br5x status text")
    chk(_status_text("FAN_10X", "fired", {}).startswith("🔔 FIRED → RESTORE 20×") and "sleeve-kill" in _status_text("FAN_10X", "review", {})
        and "stay 10×" in _status_text("FAN_10X", "holds", {}), "fan10 status texts")
    # ── Sep-24 (112) fade BTC-RSI ceiling 45 → 50 ──
    chk(fade_brsi_band(45.0) == "low" and fade_brsi_band(45.01) == "band" and fade_brsi_band(50.0) == "band" and fade_brsi_band(50.01) == "above"
        and fade_brsi_band(None) is None and fade_brsi_band("x") is None and fade_brsi_band(np.nan) is None,
        "fade bRSI band edges: 45 excluded · 50 included · > 50 above · unreadable None")
    chk(decide_fade_brsi([0.5] * 9)[0] == "collecting", "fade bRSI: 9 fires collect")
    chk(decide_fade_brsi([0.5] * 6 + [-0.1] * 4)[0] == "holds", "fade bRSI: WR 60 % ∧ Σ +2.6 → holds")
    chk(decide_fade_brsi([0.5] * 5 + [-0.1] * 5)[0] == "fired", "fade bRSI: WR 50 % < 55 → fires (WR leg, Σ > 0)")
    chk(decide_fade_brsi([0.2] * 8 + [-1.5] * 2)[0] == "fired" and decide_fade_brsi([0.2] * 8 + [-1.5] * 2)[1] == 80.0,
        "fade bRSI: Σ −1.4 < 0 → fires at 80 % WR (Σ leg)")
    chk(decide_fade_brsi([0.15] * 8 + [-0.6] * 2)[0] == "holds", "fade bRSI: Σ exactly 0 holds (bar is < 0)")
    chk(decide_fade_brsi([0.5] * 10 + [-9.0] * 5)[0] == "holds", "fade bRSI: only the FIRST 10 fires count")
    _ff = pd.DataFrame({"o_ms": [DAY, DAY + 20_000, DAY + 5 * MIN, DAY + 9 * MIN], "pair": ["AUSDT", "BUSDT", "CUSDT", "DUSDT"],
                        "entry_btc_rsi": [46.0, 49.0, 47.0, 48.0], "status": ["CLOSED", "CLOSED", "OPEN", "CLOSED"],
                        "pnl_percentage": [0.5, -1.5, np.nan, 0.3]})
    _fi = fade_brsi_fires(_ff)
    chk(len(_fi) == 3 and _fi[0]["n"] == 2 and abs(_fi[0]["val"] + 0.5) < 1e-9, "fade bRSI fires: two fills in one minute = ONE fire (mean pnl %)")
    chk(len(fade_brsi_prefix(_fi)) == 1, "fade bRSI fires: a later closed fire waits behind an earlier open one (closed prefix)")
    _ff2 = _ff.assign(status=["CLOSED", "OPEN", "CLOSED", "CLOSED"], pnl_percentage=[0.5, np.nan, 0.2, 0.3])
    chk(len(fade_brsi_prefix(fade_brsi_fires(_ff2))) == 0, "fade bRSI fires: an open fill inside a minute holds that whole fire")
    _orig = DEPLOYS["FADE_BRSI50"]
    DEPLOYS["FADE_BRSI50"] = ("grep:(NO SUCH COMMIT 0xdeadbeef)", "1970-01-01 00:00:00")
    try:
        _rs = [44.0, 45.0, 50.5, 46.0, 50.0] + [47.0] * 9          # 44 / 45 / 50.5 out · 46 + 50 in · then 9 more in-band
        _go = pd.DataFrame({"o_ms": [DAY + k * 10 * MIN for k in range(14)], "pair": [f"P{k}USDT" for k in range(14)],
                            "entry_strategy": ["SPIKE_FADE"] * 13 + ["FRENZY_LONG"], "entry_btc_rsi": _rs, "status": ["CLOSED"] * 14,
                            "pnl_percentage": [-9, -9, -9] + [0.3] * 7 + [-1.5] * 2 + [0.3, 0.3]})
        _st = {}
        _r = gate_fade_brsi(_go.iloc[:8], _st)
        chk(_r[0] == "collecting" and _st["gates"]["FADE_BRSI50"]["progress"].startswith("5/10") and "2 at ≤ 45 · 1 above 50" in _st["gates"]["FADE_BRSI50"]["progress"]
            and "Σ headroom" in _st["gates"]["FADE_BRSI50"]["progress"],
            "fade bRSI gate: 44 + 45 counted as ≤ 45, 50.5 above, 50 in the band · caution shown while collecting")
        _st = {}
        _r = gate_fade_brsi(_go, _st)
        _z = _st["gates"]["FADE_BRSI50"]["frozen"]
        chk(_r[0] == "fired" and abs(_z["sum"] - (8 * 0.3 - 3.0)) < 1e-9 and abs(_z["wr"] - 80.0) < 1e-9 and len(_z["fires"]) == 10,
            "fade bRSI gate: 10th fire freezes · 8W/2 stops Σ −0.6 → FIRES at 80 % WR")
        _go2 = _go.copy(); _go2["pnl_percentage"] = [0.3] * 14
        chk(gate_fade_brsi(_go2, _st)[0] == "fired" and "FROZEN at 10/10" in _st["gates"]["FADE_BRSI50"]["progress"],
            "fade bRSI gate: the frozen verdict is reused (new data never rewrites it)")
        _go3 = pd.concat([_go, _go.iloc[[12]].assign(o_ms=DAY + 500 * MIN, pair="LATEUSDT")])
        gate_fade_brsi(_go3, _st)
        chk("1 later fire(s) not counted" in _st["gates"]["FADE_BRSI50"]["progress"], "fade bRSI gate: later fires listed, not counted")
        _st = {}
        _gh = _go.copy(); _gh["pnl_percentage"] = [-9, -9, -9] + [0.3] * 9 + [-1.0, 0.3]   # 9W / 1L Σ +1.7
        chk(gate_fade_brsi(_gh, _st)[0] == "holds", "fade bRSI gate: 90 % WR ∧ Σ > 0 → holds")
    finally:
        DEPLOYS["FADE_BRSI50"] = _orig
    chk("FADE_BRSI50" in ORDER and "FADE_BRSI50" in DEFS and "entry_btc_rsi" in ORDER_COLS and DEFS["FADE_BRSI50"][3] == "spike_fade_max_btc_rsi",
        "112 gate wired (ORDER / DEFS / deploy key / ORDER_COLS / config key)")
    chk(_status_text("FADE_BRSI50", "fired", {}).startswith("🔔 FIRED → REVERT: spike_fade_max_btc_rsi back to 45"), "fade bRSI status text")
    print(f"selftest OK — {ok} checks")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        lines, notes = run_section(record_notes=False)
        print("\n".join(lines))
        for k, ln in notes:
            print("NOTE", k, ln)
