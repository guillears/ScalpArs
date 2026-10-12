#!/usr/bin/env python3
"""⏱ Scout — WILLY_TIMECAP exit shadow (operator hypothesis 2026-10-09, DECISION_LOG 264; OBSERVE only — never changes config, no bot API).

HYPOTHESIS  FRENZY_WILLY should take its TP (+1 % net) within the first 15 or 20 minutes, else close as it is at that minute — instead of
            today's live exit (TP +frenzy_willy_tp_pct net · NO stop in paper · frenzy_willy_max_hold_minutes 120 cap, read from
            trading_config.json exactly like services/frenzy.frenzy_willy_levels).
COHORT      CLOSED FRENZY_WILLY LONG fills in the ~/Downloads orders exports, dedup (opened_at[:19], pair, direction) — never id; a CLOSED row
            beats an OPEN one whatever the export order, else the newest export (mtime) wins — opened ≥ the commit carrying "(DECISION_LOG 251)"
            (the sleeve's ship = every WILLY fill ever; git unavailable → the pinned 2026-10-08 15:13 UTC). A / B = entry_frenzy_willy_trigger.
WALKER      from the fill's OWN entry (entry_price, opened_at — sub-second when the export carries it; a whole-second stamp is truncated, so
            prints up to < 1 s BEFORE the real fill can be walked = a ≤ 1 s look-ahead): every aggTrade print strictly after opened_at, net % =
            (p/E − 1)·100 − 0.045 − 0.045·p/E (taker 0.045 % each side, exit leg on the exit price — the bot's pnl / notional; verified on
            the live exit: KAIA 1.0103, W 1.0014). Variants: LIVE (today's config) · CAP15 / CAP20 (TP within X min, else close at the last
            print ≤ opened_at + X min) · CAP30 / CAP60 (context, not part of the verdict). TP = the first print with net ≥ TP (that print's
            net is booked); a configured stop (none today) = the first print with net ≤ −stop. 1m klines fallback = PROVISIONAL: bars fully
            after the entry second; TP booked at exactly +TP on the first bar whose HIGH reaches it (bar close ≤ the cap); cap close = the
            close of the last bar closed ≤ opened_at + X min. Parity: the LIVE walk must reproduce each fill's actual exit (reason + % within
            0.05 pts) — the match rate is printed.
VERDICT     (FROZEN, pre-registered — computed ONCE on the first prefix by (open time, pair) of the scored fills reaching N ≥ 20 on ≥ 8 days;
            deferred while a WILLY fill opened ≤ the prefix end is OPEN, unscored-but-not-final or 1m-provisional with its tick archive still
            pending; persisted in reports/SCOUT_WILLY_TIMECAP.json; never re-fit; one re-read at N ≥ 40 on a later run). Per CAPx (15, 20):
            Δ_i = CAPx_i − LIVE_i; CAPx CANDIDATE (operator decides) iff mean Δ > 0 ∧ day-clustered bootstrap P(mean Δ > 0) ≥ 0.90 (4,000,
            seed 7) ∧ no single fill > 50 % of Σ Δ ∧ Σ Δ > 0 on the fills WITHOUT a TP inside the cap (fills with a TP inside the cap must
            show Δ = 0 — internal sanity, else Δ=0 SANITY FAILED, nothing frozen). No freeze unless walker parity holds on the prefix:
            every fill's LIVE walk matches its actual exit (reason + % within 0.05 pts), or ≥ 90 % match AND every fill without a TP inside
            15 or 20 min matches — else WALKER PARITY FAILED, not frozen. Both qualify → the larger mean Δ (tie → CAP15);
            both mean Δ ≤ 0 → KEEP 120; else KEEP OBSERVING.
S4 CELLS    (DECISION_LOG 278, operator-approved 2026-10-12, OBSERVE only — the cells above, their cohort and their frozen state are untouched.)
            CAP60 / CAP90: TP +1 · stop −frenzy_willy_stop_pct (4) · close at market at 60 / 90 min if neither line was hit. T125C60 /
            T125C90: the same with TP +1.25. (281) CAP60 / CAP90 are PINNED at +1 (they read today's TP before; under today's +1.25 that
            would have made CAP60 a copy of T125C60 and CAP90 a copy of today). Each cell is compared with TODAY = trading_config.json's
            live exit read by levels() (+1.25 / −4 / 90 since DECISION_LOG 281; +1 / −4 / 120 before). A cell whose cap is > today's max
            hold ("cap > hold"), or identical to today ("= today", T125C90 under +1.25 / 90), is n/a: not walked, judged or frozen. A cap
            EQUAL to the hold with another TP is judged (CAP90 +1 vs today +1.25, both 90 min). Walker = the DECISION_LOG 272 hybrid
            (scout_frenzy_willy_exit.load_path / evaluate on ONE path of max(hold, 120) min: ticks per covered UTC day, then the 1m tail =
            provisional; tick ruler = the first print through a line; a cell undecided at the path end is not counted). NOTE: the CAP15 /
            CAP20 cells above still use this file's single-source score_fill — left as is so their frozen bar keeps the scoring it was
            registered with. COHORT (281): CLOSED non-MANUAL WILLY fills opened ≥ the "(DECISION_LOG 281)" commit + 10 min (git %ct; commit
            not found → the PLACEHOLDER pin S4_DEPLOY_PIN 2026-10-12 12:00 UTC + 10 min — replace with the real commit time once known), so
            the counted cohort ran ONE live exit and parity is read against the exit the cells are compared with (it started at the DL 271
            deploy before; 0 counted fills then, nothing frozen). Earlier fills are printed as context (walked with today's exit, never
            counted). Same loader / dedup as above; MANUAL_* closes dropped and listed.
            Evidence (reports/B19_WILLY_TIMECAP4.txt, scripts/study_b19_willy_timecap4.py — yr5 live-reachable cohort, re-sequenced, +1 / −4):
              cap 120  avg +0.054  H1 +0.051 / H2 +0.058  shared book 10× $9,035
              cap  60  avg +0.069  H1 +0.071 / H2 +0.068  shared book 10× $11,507
              cap  90  avg +0.075  H1 +0.085 / H2 +0.064  shared book 10× $12,148
            B19 (14 live fills, own path): 120 +0.404 · 60 +0.402 · 90 +0.473. 15 / 20-min caps lose (+0.040 / +0.036 on the year).
            Caveat: the best of 7 caps, a small Δ from ~9 changed fills → observe only.
            T125 evidence (reports/B19_WILLY_TP075.txt / B19_WILLY_TIMECAP4.txt, yr5 reachable, re-sequenced):
              +1.25 / −4 / 120  avg +0.105  H1 +0.123 / H2 +0.085  book 10× $13,834
              +1.25 / −4 /  90  avg +0.124  H1 +0.150 / H2 +0.094  book 10× $18,040
              +1.25 / −4 /  60  avg +0.097  H1 +0.132 / H2 +0.059  book 10× $13,091  (reports/B19_WILLY_T125C60_LEVERAGE.txt, shared book
                                                                                     at WILLY 10×)
              +1 / −4 / 120     avg +0.054                         book 10× $9,035
            B19 (14 fills): +1 / −4 / 120 +0.404 · +1.25 / −4 / 120 +0.435 · +1.25 / −4 / 60 +0.462 (· / 90 +0.447). ZK 10-09 is the one
            fill that loses its +1 (it reaches ≈ +1.0 only): −1.34 at cap 120, −0.93 at cap 60. Caveat: chosen among many combinations on the
            same year → observe only. T125C60 vs today (+1.25 / −4 / 90) = the operator's "then review 60 min" question.
            Review fixes (DL 278 change set): a prefix failing walker parity (< 90 %) or the untouched-integrity leg is NEVER frozen (any
            verdict); final fills a cell leaves undecided are counted and recorded next to dropped_no_data.
            UNTOUCHED (the "fills the cap never touches show Δ = 0" integrity leg, s4_untouched — read from TODAY's walk only): a fill whose
            outcome the cell cannot change = today's exit (TP or −4) fired at ≤ the cell's cap, NO print before it reached the cell's TP
            (s4_peak_LIVE = the highest print / bar high strictly before today's exit < the cell's TP), and — for a TP exit — today's TP
            print itself is ≥ the cell's TP (the same first print through both lines; a 1m-bar TP is booked at exactly the line). Works for a
            cell TP above or below today's (281: CAP60 +1 vs today +1.25). Everything else (today's cap close, an exit after the cell's cap,
            an earlier print through the cell's TP) is touched and may move.
            FROZEN BARS (pre-registered 2026-10-12, DECISION_LOG 278) — S4_BAR (CAP60 / CAP90) and S4_BAR_T125 (T125 cells), with the 281
            cohort / n/a sentences; "KEEP 120" in S4_BAR = keep today's exit (verdict label KEEP TODAY). State under S4_CAP60 / S4_CAP90 /
            S4_T125C60 / S4_T125C90 of reports/SCOUT_WILLY_TIMECAP.json.
OLD_EXIT    (DECISION_LOG 281, operator-approved 2026-10-12, OBSERVE only) — the REVERT GATE of the WILLY exit change (+1 / −4 / 120 →
            +1.25 / −4 / 90, lev 10×). Same cohort (from the DL 281 deploy), same path: every counted fill walked under the NEW exit (today,
            from config) and the OLD exit (OLD_EXIT = +1 / −4 / 120); Δ = old − new; parity = the new walk vs the actual exit; untouched as
            above with the old TP. FROZEN REVERT BAR (OLD_BAR, pre-registered 2026-10-12, DECISION_LOG 281): first prefix N ≥ 20 counted on
            ≥ 8 UTC days, decided once — REVERT CANDIDATE (operator decides) if EITHER mean Δ > 0 with day-bootstrap P(Δ > 0) ≥ 0.90 (4,000,
            seed 7), no fill > 50 % of ΣΔ and parity ≥ 90 %, OR the new exit's walked average < 0; else KEEP NEW. Never frozen on failed
            parity / integrity; provisional / unpriced fills hold it while data is due; n/a while today's exit IS the old one. State key
            OLD_EXIT. Own report section "WILLY exit change revert gate (DL 281)".
BINANCE     cache first: ticks reports/backtest_cache/{ticks_q,ticks}/<PAIR>/<day>.npz (+ legacy scout_willy_timecap/ticks), else 1m klines
            (k1m / k1m_ondemand + this line's scout_willy_timecap/1m). Requests, ≤ 2 per run in all:
            · data.binance.vision daily aggTrades archive — ≤ 1 per run, its own 120 s timeout (outside the 25 s fapi budget), asked only ≥ 24 h
              after the UTC day ended; streamed in the caller's thread to a temp zip (1 MB reads, deadline between reads, temp removed on any
              failure — no background thread), parsed in 500k-row chunks with numeric dtypes, stored in the SHARED ticks/<PAIR>/<day>.npz
              (backtest_fetch_ticks price-only format: t int64, p float32; temp + rename). A day above 4,000,000 rows keeps only this line's
              fill windows [te − 1 min, te + max(hold, 120) + 2 min] in scout_willy_timecap/ticks_slice (shared cache not written). A 404 = not
              published yet → retried ≥ 3 h later, final only after 3 × 404 AND ≥ 3 days after the day; another 4xx final; timeout / 5xx /
              empty count as failed attempts (3 → final). Once final, the 1m walk is final. A too-big day's slice is MERGED with any slice
              already on disk (prints + windows) and is never final for a fill whose window it does not cover (lines share this ledger:
              WILLY_TIMECAP / WILLY_TP125 / WILLY_STOP / FRENZY_WILLY_TP, Oct-10).
            · fapi 1m klines — ONE page of max(hold, 120) + 3 bars per fill, once its window closed, inside the 25 s budget; another 4xx is
              final at once (slot refunded); 3 failed / empty / timed-out attempts = final.
            Both refused while the SHARED scout cooldown (NF.SHARED_COOLDOWN) is live; fapi refused once the last X-MBX-USED-WEIGHT-1M ≥ 900;
            a 418 / 429 stops the run and arms the shared cooldown (Retry-After honoured).
Usage: venv/bin/python scripts/scout_willy_timecap.py [--selftest | --build-study-ref]
"""
import glob
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (shared cooldown, bootstrap, freeze hold, state I/O)

MIN = 60_000
DAY_MS = 86_400_000
TAKER = 0.045
CAPS = (15, 20, 30, 60)                              # CAP30 / CAP60 = context only
VERDICT_CAPS = (15, 20)
MARKS = (15, 20, 30, 60, 120)
N_MIN, DAYS_MIN, REREAD_N = 20, 8, 40
P_MIN, SHARE_MAX = 0.90, 0.50
BOOT_N, BOOT_SEED = 4000, 7
PARITY_TOL = 0.05
MAX_REQ, WEIGHT_STOP, BUDGET_S = 2, 900, 25.0
ARCHIVE_LAG_MS = DAY_MS                              # an aggTrades daily archive is requested only ≥ 24 h after its UTC day ended
ARCH_TIMEOUT_S, ARCH_PER_RUN = 120.0, 1              # archive: its own 120 s timeout (backtest_fetch_ticks.one), ≤ 1 per run (outside BUDGET_S)
ARCH_CHUNK = 1 << 20                                 # archive streamed to a temp file in 1 MB reads, deadline checked between reads
ARCH_FULL_MAX_ROWS = 4_000_000                       # a day above this keeps only this line's fill windows (no shared-cache write)
ARCH_RETRY_MS = 3 * 3_600_000                        # a 404 (archive not published yet) is retried ≥ 3 h later …
ARCH_GIVEUP_MS = 3 * DAY_MS                          # … and final only after FAIL_MAX 404s AND ≥ 3 days after the UTC day ended
FAIL_MAX = 3
DEPLOY_GREP = "(DECISION_LOG 251)"
DEPLOY_PIN = "2026-10-08 15:13:12"                   # 84318b3 commit time (UTC) — used only if git is unavailable
STRATEGY = "FRENZY_WILLY"
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "entry_price", "exit_price", "notional_value",
        "pnl_percentage", "pnl", "close_reason", "entry_frenzy_willy_trigger")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, "reports", "backtest_cache")
MY_CACHE = os.path.join(_CACHE, "scout_willy_timecap")
STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_TIMECAP.json")
CONFIG = os.path.join(_ROOT, "trading_config.json")
STUDY_CSV = os.path.join(_ROOT, "reports", "FRENZY_TP3_VS_TP4_TICKS_2026-10-08_willy.csv")
REVERT = ("Pre-committed revert if a cap is ever adopted: the first 15 WILLY fills after the switch — Σ % below what the 120-min exit "
          "would have given on the same fills (this walker) → back to 120.")
_NET_BLOCKED = False                                 # selftest: any fetch attempt raises
# S4 cells (DECISION_LOG 278) — CAP60 / CAP90 on today's TP / stop, own cohort + own frozen bar
S4_CELLS = (("CAP60", 1.0, 60), ("CAP90", 1.0, 90),          # (name, TP, cap minutes) — (281) CAP60 / CAP90 pinned at the DL 278 +1
#                                                     (they read today's TP before; under today +1.25 that would turn CAP90 into today)
            ("T125C60", 1.25, 60), ("T125C90", 1.25, 90))
S4_KEY = {n: f"S4_{n}" for n, _, _ in S4_CELLS}
S4_KEEP = {"CAP60": "KEEP TODAY", "CAP90": "KEEP TODAY", "T125C60": "KEEP TODAY", "T125C90": "KEEP TODAY"}   # (281) "KEEP 120" in
#                                                     S4_BAR = keep today's exit, which is +1.25 / −4 / 90 since DECISION_LOG 281
S4_DEPLOY_GREP = "(DECISION_LOG 281)"                # (281) the S4 counted cohort runs ONE live exit: from the DL 281 deploy
S4_DEPLOY_PIN = "2026-10-12 12:00:00"                # PLACEHOLDER commit time (UTC) of the DL 281 commit — used only if git cannot find it
S4_LAG_MIN = 10                                      # the cohort starts 10 min after the commit (deploy lag)
# OLD_EXIT (DECISION_LOG 281) — the revert gate of the WILLY exit change: the old live exit re-walked on every fill since the change
OLD_EXIT = (1.0, 4.0, 120)                           # (TP %, stop %, cap min) = WILLY's live exit before DECISION_LOG 281
OLD_KEY = "OLD_EXIT"
OLD_LABEL = "pre-registered 2026-10-12, DECISION_LOG 281"
OLD_REVERT, OLD_KEEP = "REVERT CANDIDATE (operator decides)", "KEEP NEW"
OLD_BAR = ("FROZEN REVERT BAR (pre-registered 2026-10-12, DECISION_LOG 281). At the first prefix with N ≥ 20 counted fills on ≥ 8 distinct "
           "UTC days, decided once: REVERT CANDIDATE (operator decides) if EITHER the mean paired Δ (old − new) > 0 with day-bootstrap "
           "P(Δ > 0) ≥ 0.90 (4,000, seed 7), with no fill > 50 % of ΣΔ and walker parity ≥ 90 %; OR the new exit's average (walked) < 0. "
           "Else KEEP NEW. Not frozen on failed parity or integrity (WALKER PARITY / INTEGRITY FAILED — re-checked next run); provisional "
           "(1m-part) or unpriced fills in the prefix hold the freeze while their tick archive / 1m page is still due. Counted from the "
           "DECISION_LOG 281 deploy; n/a while today's exit equals the old one.")
S4_PARITY_MIN = 0.90
S4_LABEL = "pre-registered 2026-10-12, DECISION_LOG 278"
S4_BAR = ("FROZEN BAR (pre-registered 2026-10-12, DECISION_LOG 278). For each cell separately, at the first prefix with N ≥ 20 counted fills "
          "on ≥ 8 distinct UTC days, decided ONCE: CANDIDATE (operator decides; never automatic) iff all of: the mean paired Δ (cell − today) "
          "> 0 with day-bootstrap P(Δ > 0) ≥ 0.90 (4,000 resamples, seed 7); no single fill carries > 50 % of Σ Δ; fills the cap never "
          "touches show Δ = 0 (walker integrity); walker parity vs live ≥ 90 %; else, mean Δ ≤ 0 → KEEP 120; else KEEP OBSERVING. "
          "Provisional (1m-part) or unpriced fills in the prefix hold the freeze while their tick archive / 1m page is still due. A prefix "
          "whose walker parity is < 90 % or whose untouched-integrity check fails is never frozen (WALKER PARITY / INTEGRITY FAILED — "
          "re-checked next run). A cell whose cap is > today's max hold, or that is identical to today's exit, is n/a (not judged, not "
          "frozen). Counted from the DECISION_LOG 281 deploy (today = +1.25 / −4 / 90).")
S4_BAR_T125 = ("FROZEN BAR (pre-registered 2026-10-12, DECISION_LOG 278). Per cell, N ≥ 20 counted fills on ≥ 8 days: CANDIDATE iff mean "
               "paired Δ > 0 with day-bootstrap P ≥ 0.90, no fill > 50 % of Σ Δ, untouched fills Δ = 0, walker parity ≥ 90 %; mean Δ ≤ 0 "
               "→ keep today; else keep observing. Compared against today's exit (trading_config.json; +1.25 / −4 / 90 from DECISION_LOG 281, "
               "+1 / −4 / 120 before). Untouched (these cells change the TP as well as the cap) = fills where neither the TP difference nor "
               "the cap changes the outcome: today's exit (TP or −4) fired at ≤ the cell's cap, no print before it reached the cell's TP, "
               "and — for a TP exit — today's TP print itself is ≥ the cell's TP. Provisional (1m-part) or unpriced fills "
               "in the prefix hold the freeze while their tick archive / 1m page is still due. A prefix whose walker parity is < 90 % or "
               "whose untouched-integrity check fails is never frozen (WALKER PARITY / INTEGRITY FAILED — re-checked next run). A cell whose "
               "cap is > today's max hold, or that is identical to today's exit, is n/a (not judged, not frozen). Counted from the "
               "DECISION_LOG 281 deploy (today = +1.25 / −4 / 90).")


# ─────────────────────────── config / accounting ───────────────────────────
def levels(th=None):
    """(tp %, stop % as a positive number or None, max hold minutes) — services/frenzy.frenzy_willy_levels semantics."""
    if th is None:
        try:
            with open(CONFIG) as f:
                th = json.load(f)
        except Exception:
            th = {}
    if isinstance(th, dict) and isinstance(th.get("thresholds"), dict):
        th = th["thresholds"]                                        # trading_config.json keeps the sleeve keys under "thresholds"

    def _f(k, d):
        try:
            v = th.get(k)
            return float(v) if v not in (None, "") else d
        except (TypeError, ValueError):
            return d
    tp, st, mh = _f("frenzy_willy_tp_pct", 1.0), abs(_f("frenzy_willy_stop_pct", 0.0)), _f("frenzy_willy_max_hold_minutes", 120)
    st = st if st > 0 else None                                      # DECISION_LOG 271: WILLY's own stop, no hidden floor, paper = live
    return (tp if tp > 0 else 1.0), st, (int(round(mh)) if mh >= 1 else 120)


def net_pct(p, e):
    """the bot's paper net % of notional: gross − 0.045 entry taker − 0.045 × exit/entry exit taker."""
    r = np.asarray(p, dtype=float) / float(e)
    return (r - 1) * 100 - TAKER - TAKER * r


# ─────────────────────────── walkers ───────────────────────────
def _variants(hold):
    return [("LIVE", hold)] + [(f"CAP{x}", x) for x in CAPS]


def walk_ticks(t, p, te, e, tp, stop, hold):
    """t, p = prints strictly AFTER the entry (sorted). → dict(var → dict(pct, why, xmin, worst), marks, tp_min)."""
    t = np.asarray(t, dtype="int64")
    net = net_pct(p, e) if len(t) else np.array([], dtype=float)
    out = {}
    for v, x in _variants(hold):
        n = int(np.searchsorted(t, te + x * MIN, side="right"))
        w = net[:n]
        if not n:
            out[v] = dict(pct=float(net_pct(e, e)), why="CAP", xmin=float(x), worst=float(net_pct(e, e)))
            continue
        itp = np.flatnonzero(w >= tp)
        ist = np.flatnonzero(w <= -stop) if stop else np.array([], dtype=int)
        a = int(itp[0]) if len(itp) else None
        b = int(ist[0]) if len(ist) else None
        if b is not None and (a is None or b <= a):
            out[v] = dict(pct=float(w[b]), why="STOP", xmin=(t[b] - te) / MIN, worst=float(w[:b + 1].min()))
        elif a is not None:
            out[v] = dict(pct=float(w[a]), why="TP", xmin=(t[a] - te) / MIN, worst=float(w[:a + 1].min()))
        else:
            out[v] = dict(pct=float(w[-1]), why="CAP", xmin=float(x), worst=float(w.min()))
    marks = {}
    for x in MARKS:
        n = int(np.searchsorted(t, te + x * MIN, side="right"))
        marks[x] = float(net[n - 1]) if n else float(net_pct(e, e))
    n = int(np.searchsorted(t, te + max(hold, max(MARKS)) * MIN, side="right"))
    i = np.flatnonzero(net[:n] >= tp)
    return dict(v=out, marks=marks, tp_min=float((t[i[0]] - te) / MIN) if len(i) else float("nan"))


def walk_1m(k, te, e, tp, stop, hold):
    """k = 1m bars (open_time, h, l, c) sorted; bars fully after the entry second only. PROVISIONAL (see module doc)."""
    first = -(-int(te) // MIN) * MIN
    k = k[k.open_time.values >= first]
    o = k.open_time.values.astype("int64")
    cl = o + MIN
    nh, nl, nc = net_pct(k.h.values, e), net_pct(k.l.values, e), net_pct(k.c.values, e)
    out = {}
    for v, x in _variants(hold):
        n = int(np.searchsorted(cl, te + x * MIN, side="right"))
        if not n:
            out[v] = dict(pct=float(net_pct(e, e)), why="CAP", xmin=float(x), worst=float(net_pct(e, e)))
            continue
        itp = np.flatnonzero(nh[:n] >= tp)
        ist = np.flatnonzero(nl[:n] <= -stop) if stop else np.array([], dtype=int)
        a = int(itp[0]) if len(itp) else None
        b = int(ist[0]) if len(ist) else None
        if b is not None and (a is None or b <= a):                 # same bar → stop (conservative)
            ob = float(k.o.values[b]) if "o" in k and np.isfinite(k.o.values[b]) else np.nan
            gap = float(net_pct(ob, e)) if np.isfinite(ob) else -float(stop)   # a bar that OPENS through the stop fills at its open
            out[v] = dict(pct=min(-float(stop), gap), why="STOP", xmin=(cl[b] - te) / MIN, worst=float(nl[:b + 1].min()))
        elif a is not None:
            out[v] = dict(pct=float(tp), why="TP", xmin=(cl[a] - te) / MIN, worst=float(nl[:a + 1].min()))
        else:
            out[v] = dict(pct=float(nc[n - 1]), why="CAP", xmin=float(x), worst=float(nl[:n].min()))
    marks = {}
    for x in MARKS:
        n = int(np.searchsorted(cl, te + x * MIN, side="right"))
        marks[x] = float(nc[n - 1]) if n else float(net_pct(e, e))
    n = int(np.searchsorted(cl, te + max(hold, max(MARKS)) * MIN, side="right"))
    i = np.flatnonzero(nh[:n] >= tp)
    return dict(v=out, marks=marks, tp_min=float((cl[i[0]] - te) / MIN) if len(i) else float("nan"))


LIVE_WHY = {"FRENZY_TP": "TP", "FRENZY_TP_LATE": "TP", "MAX_HOLD_TIME": "CAP", "STOP_LOSS": "STOP"}


def parity(why_live, pct_live, w):
    """walker LIVE vs the fill's actual exit → (ok, text)."""
    r = LIVE_WHY.get(str(why_live), str(why_live))
    d = w["v"]["LIVE"]["pct"] - float(pct_live)
    ok = (r == w["v"]["LIVE"]["why"]) and abs(d) <= PARITY_TOL and str(why_live) != "FRENZY_TP_LATE"
    return ok, f"live {why_live} {float(pct_live):+.3f} vs walk {w['v']['LIVE']['why']} {w['v']['LIVE']['pct']:+.3f} (Δ {d:+.3f})"


# ─────────────────────────── orders ───────────────────────────
def deploy_ts():
    try:
        out = subprocess.run(["git", "-C", _ROOT, "log", "--format=%ct", "--grep=" + DEPLOY_GREP, "--fixed-strings", "--reverse"],
                             capture_output=True, text=True, timeout=10)
        return pd.Timestamp(int(out.stdout.strip().splitlines()[0]) * 1000, unit="ms")
    except Exception:
        return pd.Timestamp(DEPLOY_PIN)


def _read_exports():
    fr = []
    for f in glob.glob(EXPORT_GLOB):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    return fr


def _dedup(o):
    """rows in precedence order (last wins; a CLOSED row always beats an OPEN one) → dedup (opened_at[:19], pair, direction), WILLY LONG."""
    o = o.reindex(columns=list(dict.fromkeys(list(COLS) + list(o.columns))))
    if not len(o):
        return o
    o["_k"] = o.opened_at.astype(str).str[:19]
    o["_c"] = o.status.astype(str).str.upper().eq("CLOSED")
    o = o.sort_values("_c", kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last")
    return o[(o.direction.astype(str) == "LONG") & (o.entry_strategy.astype(str) == STRATEGY)]


def load_orders(orders=None):
    """→ (closed WILLY fills prepared, [(opened_at, 'OPEN <pair>')] for OPEN WILLY rows after dedup)."""
    if orders is None:
        fr = _read_exports()
        raw = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable") if fr else pd.DataFrame(columns=list(COLS))
    else:
        raw = orders
    o = _dedup(raw)
    if not len(o):
        return _prep(o), []
    t = pd.to_datetime(o.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    op = o.status.astype(str).str.upper().eq("OPEN")
    opens = [(a, f"OPEN {p}") for a, p in zip(t[op], o.pair[op].astype(str)) if pd.notna(a)]
    return _prep(o[o.status.astype(str).str.upper() == "CLOSED"]), opens


def _prep(o):
    cols = ["pair", "ts", "day", "te", "E", "pct", "usd", "notional", "why", "trig", "closed_at"]
    if not len(o):
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
    t = pd.to_datetime(o.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    d = pd.DataFrame(dict(pair=o.pair.astype(str).values, ts=t.values,
                          E=pd.to_numeric(o.entry_price, errors="coerce").values,
                          pct=pd.to_numeric(o.pnl_percentage, errors="coerce").values,
                          usd=pd.to_numeric(o.pnl, errors="coerce").values,
                          notional=pd.to_numeric(o.notional_value, errors="coerce").values,
                          why=o.close_reason.astype(str).values,
                          trig=o.entry_frenzy_willy_trigger.fillna("?").astype(str).str.strip().str.upper().values,
                          closed_at=o.closed_at.astype(str).values))
    tf = pd.to_datetime(o.opened_at.astype(str), format="ISO8601", errors="coerce").values   # sub-second when the export carries it
    d["tf"] = np.where(pd.isna(tf), d.ts.values, tf)
    d = d[d.ts.notna() & d.E.notna() & (d.E > 0) & d.pct.notna()].copy()
    d["te"] = pd.to_datetime(d.tf).values.astype("datetime64[ms]").astype("int64")
    d = d.drop(columns=["tf"])
    d["day"] = d.ts.dt.strftime("%Y-%m-%d")
    return d.sort_values(["ts", "pair"], kind="stable").reset_index(drop=True)


# ─────────────────────────── cache ───────────────────────────
def _span(hold):
    """minutes of path needed after the entry: the live hold and every mark (120)."""
    return max(int(hold), max(MARKS))


def _days(te, hold):
    a = pd.Timestamp(int(te), unit="ms").normalize()
    b = pd.Timestamp(int(te) + (_span(hold) + 1) * MIN, unit="ms").normalize()
    return [f"{d:%Y-%m-%d}" for d in pd.date_range(a, b)]


def _tick_file(pair, day):
    for p in (os.path.join(MY_CACHE, "ticks", pair, f"{day}.npz"), os.path.join(_CACHE, "ticks_q", pair, f"{day}.npz"),
              os.path.join(_CACHE, "ticks", pair, f"{day}.npz")):
        if os.path.exists(p):
            return p
    return None


def _slice_file(pair, day):
    return os.path.join(MY_CACHE, "ticks_slice", pair, f"{day}.npz")


def _covers(w, need):
    """w = None (a full day) or (w0, w1) window arrays; need = (a, b) or None."""
    return w is None or need is None or bool(np.any((w[0] <= need[0]) & (w[1] >= need[1])))


def _load_day(pair, d, cache):
    """→ (t, p, windows or None for a full day) or None. Full-day files first, then this line's window slice (too-big days)."""
    k = (pair, d)
    if cache is not None and k in cache:
        return cache[k]
    z = None
    f = _tick_file(pair, d)
    if f is None and os.path.exists(_slice_file(pair, d)):
        f = _slice_file(pair, d)
    if f is not None:
        with np.load(f) as zz:
            t = zz["t"].astype("int64")
            o = np.argsort(t, kind="stable")
            w = (zz["w0"].astype("int64"), zz["w1"].astype("int64")) if "w0" in zz.files else None
            z = (t[o], zz["p"][o].astype(np.float64), w)
        if cache is not None:
            if len(cache) > 12:
                cache.clear()
            cache[k] = z
    return z


def _have_day(pair, d, need=None, cache=None):
    z = _load_day(pair, d, cache) if (_tick_file(pair, d) is None) else (None, None, None)
    return _tick_file(pair, d) is not None or (z is not None and _covers(z[2], need))


def load_ticks(pair, days, cache=None, need=None):
    ts, ps = [], []
    for d in days:
        z = _load_day(pair, d, cache)
        if z is None or not _covers(z[2], need):
            return None, None
        ts.append(z[0])
        ps.append(z[1])
    return np.concatenate(ts), np.concatenate(ps)


def _need(te, hold):
    """the tick window a fill needs: [te − 1 min, te + max(hold, 120) + 2 min]."""
    return int(te) - MIN, int(te) + (_span(hold) + 2) * MIN


def _read_k1(path):
    try:
        d = pd.read_csv(path, usecols=lambda c: c in ("open_time", "o", "h", "l", "c"))
        d = d.reindex(columns=["open_time", "o", "h", "l", "c"]).apply(pd.to_numeric, errors="coerce").dropna(subset=["open_time", "h", "l", "c"])
        d["open_time"] = d.open_time.astype("int64")
        return d                                                       # o = the bar's open (NaN in caches written before Oct-10)
    except Exception:
        return pd.DataFrame({"open_time": pd.Series(dtype="int64"), "o": pd.Series(dtype=float), "h": pd.Series(dtype=float),
                             "l": pd.Series(dtype=float), "c": pd.Series(dtype=float)})


_K1 = {}                                             # per-run 1m cache (cleared by run(); a pair is dropped when this line appends to it)


def load_1m(pair):
    if pair in _K1:
        return _K1[pair]
    fr = [_read_k1(p) for p in (os.path.join(_CACHE, "k1m", f"{pair}.csv"), os.path.join(_CACHE, "k1m_ondemand", f"{pair}.csv"),
                                os.path.join(MY_CACHE, "1m", f"{pair}.csv")) if os.path.exists(p)]
    fr = [f for f in fr if len(f)]
    k = (pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)) if fr else _read_k1("")
    _K1[pair] = k
    return k


def _need_1m(te, hold):
    first = -(-int(te) // MIN) * MIN
    last = ((int(te) + _span(hold) * MIN) // MIN) * MIN - MIN           # last bar whose close ≤ te + max(hold, 120)
    return np.arange(first, last + 1, MIN, dtype="int64")


def score_fill(r, hold, tp, stop, tcache=None):
    """→ (walk dict or None, source 'ticks' / '1m' / None)."""
    days = _days(r.te, hold)
    t, p = load_ticks(r.pair, days, tcache, _need(r.te, hold))
    if t is not None:
        i, j = np.searchsorted(t, r.te, side="right"), np.searchsorted(t, r.te + (_span(hold) + 1) * MIN, side="right")
        return walk_ticks(t[i:j], p[i:j], r.te, r.E, tp, stop, hold), "ticks"
    k = load_1m(r.pair)
    need = _need_1m(r.te, hold)
    if len(k) and np.isin(need, k.open_time.values).all():
        return walk_1m(k[(k.open_time >= need[0]) & (k.open_time <= need[-1])], r.te, r.E, tp, stop, hold), "1m"
    return None, None


# ─────────────────────────── Binance ───────────────────────────
def _http_get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode()), r.headers


def _threaded(fn, url, timeout_s, now_ms, what):
    if _NET_BLOCKED:
        raise RuntimeError("network blocked (selftest)")
    box = {}

    def _go():
        try:
            box["ok"] = fn(url, max(1.0, timeout_s))
        except BaseException as e:
            box["err"] = e
    th = threading.Thread(target=_go, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError(f"{what} exceeded {timeout_s:.0f} s wall clock")
    e = box.get("err")
    if isinstance(e, urllib.error.HTTPError) and e.code in (418, 429):
        secs = NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now_ms)
        raise NF.RateLimited(f"Binance {e.code} — stopped, cooldown {secs} s")
    if e is not None:
        raise e
    return box["ok"]


def _fetch_1m(sym, start_ms, limit, now_ms, timeout_s=10.0):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1m&startTime={int(start_ms)}&limit={int(limit)}"
    rows, headers = _threaded(_http_get, url, timeout_s, now_ms, f"{sym} 1m fetch")
    used = int((headers or {}).get("X-MBX-USED-WEIGHT-1M") or 0)
    d = pd.DataFrame([(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])) for x in rows if int(x[0]) + MIN <= now_ms],
                     columns=["open_time", "o", "h", "l", "c"])
    return d, used


def _open_url(url, timeout):
    """→ a readable response (context manager, .read(n)). Isolated so the selftest can monkeypatch it."""
    return urllib.request.urlopen(url, timeout=timeout)


def _download_archive(sym, day, now_ms, timeout_s=ARCH_TIMEOUT_S):
    """ONE data.binance.vision daily aggTrades zip streamed in the CALLER's thread to a temp file (1 MB reads, each bounded by the socket
    timeout, the deadline checked between reads) → temp path. Any failure removes the temp file; there is no background thread."""
    if _NET_BLOCKED:
        raise RuntimeError("network blocked (selftest)")
    from urllib.parse import quote
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{quote(sym)}/{quote(sym)}-aggTrades-{day}.zip"
    end = time.monotonic() + timeout_s
    try:
        resp = _open_url(url, max(1.0, min(20.0, timeout_s)))
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            secs = NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now_ms)
            raise NF.RateLimited(f"Binance {e.code} — stopped, cooldown {secs} s")
        raise
    os.makedirs(MY_CACHE, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".zip", prefix="arch_", dir=MY_CACHE)
    try:
        with resp, os.fdopen(fd, "wb") as out:
            while True:
                if time.monotonic() > end:
                    raise TimeoutError(f"{sym} {day} archive exceeded {timeout_s:.0f} s")
                ch = resp.read(ARCH_CHUNK)
                if not ch:
                    break
                out.write(ch)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return tmp


def _fetch_archive(sym, day, now_ms, timeout_s=ARCH_TIMEOUT_S, windows=None):
    """download + parse → (t, p, full). full False = the day was too big: only `windows` kept."""
    tmp = _download_archive(sym, day, now_ms, timeout_s)
    try:
        return parse_archive(tmp, windows)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def parse_archive(src, windows=None, max_rows=None):
    """an aggTrades zip (path / file object / bytes) → (t int64, p float64 sorted, full). Streamed in 500k-row chunks with numeric dtypes
    (a header line is sniffed). The full day is kept while ≤ max_rows (ARCH_FULL_MAX_ROWS); past that only the rows inside `windows`
    [(a, b), …] are kept and full = False."""
    max_rows = ARCH_FULL_MAX_ROWS if max_rows is None else max_rows
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    full, sl, n = [], [], 0
    with zipfile.ZipFile(src) as z, z.open(z.namelist()[0]) as fh:
        first = fh.peek(200)[:200].split(b"\n", 1)[0]
        hdr = 0 if first[:1] and not first[:1].isdigit() else None
        rd = pd.read_csv(fh, header=hdr, names=["a", "p", "q", "f", "l", "t", "m"], usecols=["p", "t"],
                         dtype={"p": np.float64, "t": np.int64}, chunksize=500_000)
        for ch in rd:
            n += len(ch)
            if full is not None:
                if n <= max_rows:
                    full.append(ch)
                else:
                    full = None
            if windows:
                m = np.zeros(len(ch), dtype=bool)
                tv = ch.t.values
                for a, b in windows:
                    m |= (tv >= a) & (tv <= b)
                if m.any():
                    sl.append(ch[m])
    keep = full if full is not None else sl
    if not keep:
        return np.array([], dtype=np.int64), np.array([], dtype=float), full is not None
    d = pd.concat(keep)
    t, p = d.t.values.astype(np.int64), d.p.values.astype(float)
    o = np.argsort(t, kind="stable")
    return t[o], p[o], full is not None


def _arch_final(done, key):
    """an archive day is final for every line only when fully cached or given up — a SLICED day (too big for the shared cache) keeps only
    the windows asked so far, so another line's fill outside them must still be able to request it (Oct-10 review: the FRENZY_WILLY_TP
    line shares this ledger)."""
    return key in done and not str(done[key]).startswith("sliced")


def _save_slice(sym, day, t, p, windows):
    """this line's window slice of a too-big day — MERGED with any slice already on disk (prints and windows), never overwritten."""
    fp = _slice_file(sym, day)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    if os.path.exists(fp):
        try:
            with np.load(fp) as z:
                t0, p0 = z["t"].astype(np.int64), z["p"].astype(np.float64)
                w2 = list(zip(z["w0"].tolist(), z["w1"].tolist())) + list(windows)
            if len(t0) != len(p0):
                raise ValueError("old slice t / p length mismatch")
            t2 = np.concatenate([t0, np.asarray(t, dtype=np.int64)])
            p2 = np.concatenate([p0, np.asarray(p, dtype=np.float64)])
            o = np.argsort(t2, kind="stable")                          # trade order kept inside a millisecond (old prints first)
            t2, p2 = t2[o], p2[o]
            keep = np.ones(len(t2), dtype=bool)
            keep[1:] = (t2[1:] != t2[:-1]) | (p2[1:] != p2[:-1])       # exact repeats (the overlap of two slices) collapse
            t, p, windows = t2[keep], p2[keep], sorted(set((int(a), int(b)) for a, b in w2))
        except Exception:
            pass                                                       # unreadable old slice → replaced by this one (locals untouched)
    tmp = fp[:-4] + f".{os.getpid()}.part.npz"
    np.savez_compressed(tmp, t=np.asarray(t, dtype=np.int64), p=np.asarray(p, dtype=np.float64),
                        w0=np.array([a for a, _ in windows], dtype=np.int64), w1=np.array([b for _, b in windows], dtype=np.int64))
    os.replace(tmp, fp)


def _save_ticks(sym, day, t, p):
    """the SHARED tick cache in scripts/backtest_fetch_ticks.py's price-only format (t int64 ms, p float32) so every line reuses it.
    Only after a complete download + parse (all in the caller's thread); temp file + os.replace, never a half file
    (backtest_fetch_ticks 'skip' trusts any existing file)."""
    fp = os.path.join(_CACHE, "ticks", sym, f"{day}.npz")
    if os.path.exists(fp):
        return
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    tmp = fp[:-4] + f".{os.getpid()}.part.npz"
    np.savez_compressed(tmp, t=np.asarray(t, dtype=np.int64), p=np.asarray(p, dtype=np.float32))
    os.replace(tmp, fp)


def _append_1m(sym, new):
    os.makedirs(os.path.join(MY_CACHE, "1m"), exist_ok=True)
    p = os.path.join(MY_CACHE, "1m", f"{sym}.csv")
    d = pd.concat([_read_k1(p), new]).drop_duplicates("open_time", keep="last").sort_values("open_time")
    tmp = f"{p}.{os.getpid()}.tmp"
    d.to_csv(tmp, index=False)
    os.replace(tmp, p)
    _K1.pop(sym, None)


def _att_path():
    return os.path.join(MY_CACHE, "attempts.json")


def load_attempts():
    try:
        with open(_att_path()) as f:
            a = json.load(f)
        return dict(done=dict(a.get("done") or {}), fail=dict(a.get("fail") or {}), last=dict(a.get("last") or {}))
    except Exception:
        return dict(done={}, fail={}, last={})


def save_attempts(att):
    os.makedirs(MY_CACHE, exist_ok=True)
    tmp = f"{_att_path()}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(att, f)
    os.replace(tmp, _att_path())


def _fail(att, key, what):
    att["fail"][key] = int(att["fail"].get(key, 0)) + 1
    if att["fail"][key] >= FAIL_MAX:
        att["done"][key] = f"{what}×{FAIL_MAX}"


def _ak(sym, day):
    return f"ARCH|{sym}|{day}"


def _kk(sym, te, span=120):
    """the 1m-page attempts key; a longer window (another line's 12 h) gets its own key so a 123-bar page never stands in for it."""
    return f"K1M|{sym}|{(int(te) // MIN) * MIN}" + ("" if int(span) == 120 else f"|{int(span)}")


def ticks_pending(r, hold, done, now_ms, cache=None):
    """True while a tick path could still arrive (a needed day not on disk for this fill's window and its archive not final)."""
    return any(not _have_day(r.pair, d, _need(r.te, hold), cache) and not _arch_final(done, _ak(r.pair, d)) for d in _days(r.te, hold))


def _day_end(day):
    return int(pd.Timestamp(day).value // 10**6) + DAY_MS


def _arch_eligible(day, now_ms, att=None, key=None):
    """due ≥ 24 h after the UTC day ended, and ≥ 3 h after this archive's last 404."""
    if now_ms < _day_end(day) + ARCHIVE_LAG_MS:
        return False
    return att is None or now_ms >= int(att.get("last", {}).get(key, 0)) + ARCH_RETRY_MS


def _arch_404(att, key, day, now_ms):
    """a 404 = not published yet: counted + time-stamped; final only after FAIL_MAX 404s AND ≥ 3 days after the UTC day ended."""
    att["fail"][key] = int(att["fail"].get(key, 0)) + 1
    att.setdefault("last", {})[key] = int(now_ms)
    _arch_giveup(att, key, day, now_ms)


def _arch_giveup(att, key, day, now_ms):
    if int(att["fail"].get(key, 0)) >= FAIL_MAX and now_ms >= _day_end(day) + ARCH_GIVEUP_MS and key not in att["done"]:
        att["done"][key] = f"404×{att['fail'][key]} (no archive 3 days after the day)"


def ensure_data(fills, hold, now_ms, fetch=True, budget_s=BUDGET_S, max_req=MAX_REQ):
    """fills = rows with src column ('ticks' / '1m' / None). Oldest first: a needed tick day whose archive is due → archive request;
    else, with no path at all and the hold window closed → one 1m page. → notes."""
    notes, used, stop, n = [], 0, None, 0
    if not fetch or not len(fills):
        return notes
    att = load_attempts()
    done = att["done"]
    jobs, gave_up = [], False
    for r in fills.sort_values(["ts", "pair"], kind="stable").itertuples():
        if r.src == "ticks":
            continue
        for d in _days(r.te, hold):                                    # a long-404 archive past its give-up time → final, no request
            if _ak(r.pair, d) not in done and int(att["fail"].get(_ak(r.pair, d), 0)) >= FAIL_MAX:
                _arch_giveup(att, _ak(r.pair, d), d, now_ms)
                gave_up = gave_up or _ak(r.pair, d) in done
        days = [d for d in _days(r.te, hold) if not _have_day(r.pair, d, _need(r.te, hold)) and not _arch_final(done, _ak(r.pair, d))]
        due = [d for d in days if _arch_eligible(d, now_ms, att, _ak(r.pair, d))]
        if due:
            jobs.append(("arch", r, due[0]))
        if ((not isinstance(r.src, str) or bool(getattr(r, "need_1m", False)))     # (272 review I2) also a scored fill with an undecided
                and _kk(r.pair, r.te, _span(hold)) not in done and now_ms >= r.te + (_span(hold) + 1) * MIN):   # cell (FRENZY_WILLY_TP)
            jobs.append(("1m", r, None))
    if gave_up:
        save_attempts(att)
    if not jobs:
        return notes
    cd = NF._cooldown_until()
    if cd > now_ms:
        return [f"fetching stopped: rate-limit cooldown until {pd.Timestamp(cd, unit='ms'):%m-%d %H:%M} UTC"]
    deadline, seen, n_arch = time.monotonic() + budget_s, set(), 0
    for kind, r, day in jobs:
        key = _ak(r.pair, day) if kind == "arch" else _kk(r.pair, r.te, _span(hold))
        if key in seen or (_arch_final(done, key) if kind == "arch" else key in done) or (kind == "arch" and n_arch >= ARCH_PER_RUN):
            continue
        seen.add(key)
        if n >= max_req:
            notes.append(f"request cap {max_req}/run reached — the rest catches up on later runs")
            break
        if used >= WEIGHT_STOP:
            stop = f"used weight {used} ≥ {WEIGHT_STOP}"
            break
        left = deadline - time.monotonic()
        if kind != "arch" and left < 1.0:                              # the archive has its own timeout, outside the 25 s fapi budget
            stop = "wall-clock budget spent"
            break
        n += 1
        n_arch += kind == "arch"
        lab = f"{r.pair} {day} aggTrades archive" if kind == "arch" else f"{r.pair} 1m page"
        try:
            if kind == "arch":
                wins = sorted({_need(x.te, hold) for x in fills.itertuples() if x.pair == r.pair and day in _days(x.te, hold)})
                t, p, full = _fetch_archive(r.pair, day, now_ms, timeout_s=ARCH_TIMEOUT_S, windows=wins)
                if len(t) and full:
                    _save_ticks(r.pair, day, t, p)
                    done[key] = int(len(t))
                    att["fail"].pop(key, None)
                elif len(t):
                    _save_slice(r.pair, day, t, p, wins)
                    done[key] = f"sliced {len(t)}"
                    att["fail"].pop(key, None)                         # a later line's re-request starts its own failure count
                else:
                    _fail(att, key, "empty")
                notes.append(f"{lab} +{len(t):,} prints" + ("" if full else f" (day > {ARCH_FULL_MAX_ROWS:,} rows: only the "
                                                                              f"{len(wins)} fill window(s) kept here, shared tick cache not written)"))
            else:
                d, used = _fetch_1m(r.pair, (int(r.te) // MIN) * MIN, _span(hold) + 3, now_ms, timeout_s=min(10.0, left))
                if len(d):
                    _append_1m(r.pair, d)
                    done[key] = int(len(d))
                else:
                    _fail(att, key, "empty")
                notes.append(f"{lab} +{len(d)} bars (weight used {used})")
            save_attempts(att)
        except NF.RateLimited as e:
            stop = str(e)
            break
        except urllib.error.HTTPError as e:
            if kind == "arch" and e.code == 404:                         # not published yet — retried ≥ 3 h later (not final)
                _arch_404(att, key, day, now_ms)
            elif 400 <= e.code < 500:
                done[key] = f"HTTP {e.code}"
                n -= 1
            else:
                _fail(att, key, "failed")
            save_attempts(att)
            notes.append(f"{lab} failed (HTTP {e.code})")
        except Exception as e:
            notes.append(f"{lab} failed ({str(e)[:80]})")
            if isinstance(e, TimeoutError):                            # counted (FAIL_MAX → final) and the run stops fetching
                _fail(att, key, "timeout")
                save_attempts(att)
                stop = "wall-clock timeout"
                break
            _fail(att, key, "failed")
            save_attempts(att)
    if stop:
        notes.append(f"fetching stopped: {stop}")
    return notes


# ─────────────────────────── verdict / freeze ───────────────────────────
def p_delta_pos(delta, day):
    """day-clustered bootstrap P(mean Δ > 0) (4,000, seed 7) = NF.p_mean_neg on −Δ."""
    return NF.p_mean_neg(-np.asarray(delta, dtype=float), np.asarray(day), BOOT_N, BOOT_SEED)


def judge_cap(sc, x):
    """sc = scored fills with columns LIVE, CAPx, whyCAPx, day → dict(meanΔ, p, share, d_notp, n_notp, walker_ok, qualifies)."""
    d = sc[f"CAP{x}"].values - sc["LIVE"].values
    tp_in = sc[f"why_CAP{x}"].values == "TP"
    s = float(d.sum())
    p = p_delta_pos(d, sc.day.values) if len(sc) else None
    share = float(d.max() / s) if s > 0 else float("nan")
    walker_ok = bool(np.all(np.abs(d[tp_in]) < 1e-9))
    d_notp = float(d[~tp_in].sum())
    m = float(d.mean()) if len(d) else float("nan")
    q = walker_ok and m > 0 and p is not None and p >= P_MIN and s > 0 and share <= SHARE_MAX and d_notp > 0
    return dict(mean=m, p=p, share=share, d_notp=d_notp, n_notp=int((~tp_in).sum()), walker_ok=walker_ok, q=bool(q))


def verdict(sc):
    n, nd = len(sc), sc.day.nunique() if len(sc) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"N {n}/{N_MIN} · {nd}/{DAYS_MIN} days", {}
    J = {x: judge_cap(sc, x) for x in VERDICT_CAPS}
    det = " · ".join(f"CAP{x}: mean Δ {j['mean']:+.3f} pts · P(Δ>0) {(j['p'] if j['p'] is not None else float('nan')):.2f} · top fill "
                     f"{(j['share'] * 100 if j['share'] == j['share'] else float('nan')):.0f} % of ΣΔ · ΣΔ on the {j['n_notp']} no-TP-in-cap "
                     f"fills {j['d_notp']:+.2f} · TP-in-cap Δ = 0 {'✓' if j['walker_ok'] else '✗'}" for x, j in J.items())
    det = f"N {n} · {nd} d · " + det
    if not all(j["walker_ok"] for j in J.values()):
        return "Δ=0 SANITY FAILED", det, J
    qs = [x for x in VERDICT_CAPS if J[x]["q"]]
    if qs:
        best = max(qs, key=lambda x: (round(J[x]["mean"], 12), -x))   # tie → CAP15
        return f"CAP{best} CANDIDATE (operator decides)", det, J
    if all(J[x]["mean"] <= 0 for x in VERDICT_CAPS):
        return "KEEP 120", det, J
    return "KEEP OBSERVING", det, J


def prefix(sc, n_min):
    z = sc.sort_values(["ts", "pair"], kind="stable")
    seen = set()
    for k, d in enumerate(z.day.values, 1):
        seen.add(d)
        if k >= n_min and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def parity_gate(pre):
    """→ (ok, text). ok iff every fill's LIVE walk matches its actual exit, or ≥ 90 % match AND every fill without a TP inside 15 or
    20 min (the fills where the variants differ) matches."""
    po = pre.par_ok.astype(bool).values
    notp = ((pre.why_CAP15 != "TP") | (pre.why_CAP20 != "TP")).values
    txt = f"walker parity {int(po.sum())}/{len(po)} · {int(po[notp].sum())}/{int(notp.sum())} on the no-TP-in-cap fills"
    return bool(po.all() or (po.mean() >= 0.90 and po[notp].all())), txt


def freeze(st, sc, now_iso, cfg, hold=None, why=None, dropped=None):
    """freeze 'first' (N ≥ 20 on ≥ 8 days) ONCE; 'reread' (N ≥ 40) only on a later call. Never frozen: a failed walker-parity gate or a
    Δ=0 SANITY FAILED verdict. dropped = open times of fills final with no data (excluded; counted ≤ the prefix end in the record)."""
    key = "reread" if "first" in st else "first"
    if key in st:
        return st, False
    pre = prefix(sc, REREAD_N if key == "reread" else N_MIN)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if NF.freeze_hold(last, hold, why):
        return st, False
    pok, ptxt = parity_gate(pre)
    if not pok:
        if why is not None:
            why.append(f"WALKER PARITY FAILED — not frozen ({ptxt})")
        return st, False
    state, det, _ = verdict(pre)
    if state == "Δ=0 SANITY FAILED":
        if why is not None:
            why.append("Δ=0 sanity failed (TP-inside-cap fills show Δ ≠ 0) — fix the walker, nothing frozen")
        return st, False
    nd = int(sum(1 for t in (dropped or []) if pd.notna(t) and t <= last))
    st[key] = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso, n=len(pre), days=int(pre.day.nunique()),
                   parity=ptxt, dropped_no_data=nd,
                   config=dict(tp=cfg[0], stop=cfg[1], hold=cfg[2]),
                   keys=[f"{a}|{b}" for a, b in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str))])
    return st, True


# ─────────────────────────── scoring the cohort ───────────────────────────
def scored_frame(o, hold, tp, stop, tcache=None):
    rows = []
    for r in o.itertuples():
        w, src = score_fill(r, hold, tp, stop, tcache)
        rec = dict(src=src)
        if w is not None:
            for v, res in w["v"].items():
                rec[v], rec[f"why_{v}"], rec[f"min_{v}"], rec[f"worst_{v}"] = res["pct"], res["why"], res["xmin"], res["worst"]
            for x, m in w["marks"].items():
                rec[f"at{x}"] = m
            rec["tp_min"] = w["tp_min"]
            rec["par_ok"], rec["par_txt"] = parity(r.why, r.pct, w)
        rows.append(rec)
    if not rows:
        return o.reset_index(drop=True).assign(src=pd.Series(dtype=object))
    return pd.concat([o.reset_index(drop=True), pd.DataFrame(rows, index=range(len(o)))], axis=1)


# ─────────────────────────── S4 cells: CAP60 / CAP90 / T125C60 / T125C90 on today's −4 stop (DECISION_LOG 278) ───────────────────────────
def _fx():
    """the DECISION_LOG 272 hybrid walker (load_path / evaluate) — imported lazily: scout_frenzy_willy_exit imports this module."""
    import scout_frenzy_willy_exit as FX
    return FX


def _git_ct(grep):
    """the earliest commit whose message carries `grep` (fixed string) → its commit time (UTC), else None."""
    try:
        out = subprocess.run(["git", "-C", _ROOT, "log", "--format=%ct", "--grep=" + grep, "--fixed-strings", "--reverse"],
                             capture_output=True, text=True, timeout=10)
        return pd.Timestamp(int(out.stdout.strip().splitlines()[0]) * 1000, unit="ms")
    except Exception:
        return None


def s4_deploy_ts():
    """the S4 / OLD_EXIT counted floor = the "(DECISION_LOG 281)" commit time (WILLY's exit → +1.25 / −4 / 90) + 10 min; git cannot find it →
    the PLACEHOLDER pin S4_DEPLOY_PIN + 10 min (replace the pin with the real commit time once known). Shared by WILLY_BTC1H."""
    t = _git_ct(S4_DEPLOY_GREP)
    return (pd.Timestamp(S4_DEPLOY_PIN) if t is None else t) + pd.Timedelta(minutes=S4_LAG_MIN)


dl281_floor = s4_deploy_ts


def s4_spec(name, tp):
    """→ (cell TP, cap minutes); a None TP in S4_CELLS = today's TP (trading_config.json). OLD_EXIT → the old exit's (TP, cap)."""
    if name == OLD_KEY:
        return float(OLD_EXIT[0]), int(OLD_EXIT[2])
    _, t, x = next(c for c in S4_CELLS if c[0] == name)
    return (float(tp) if t is None else float(t)), int(x)


def s4_na(name, tp, hold, stop=None):
    """→ '' when the cell is judged, else the n/a reason (not judged, not frozen): 'cap > hold' (a cap beyond today's max hold), or
    '= today' (identical to today's exit: Δ ≡ 0). OLD_EXIT is n/a only while today's exit IS the old one."""
    ctp, x = s4_spec(name, tp)
    if name == OLD_KEY:
        same = abs(float(tp) - ctp) < 1e-12 and int(hold) == x and (stop is None or abs(float(stop) - OLD_EXIT[1]) < 1e-12)
        return "= today" if same else ""
    if x > int(hold):
        return "cap > hold"
    if x == int(hold) and abs(ctp - float(tp)) < 1e-12:
        return "= today"
    return ""


def _peak_before_exit(P, tp, stop, cap):
    """the highest net print / bar high strictly BEFORE today's exit event (first-touch, as evaluate) — −inf when the exit is the first
    event; today's CAP exit → the peak over the whole cap. Feeds the untouched-integrity leg only."""
    n = int(np.searchsorted(P["t"], P["te"] + cap * MIN, side="right"))
    hi, lo = P["hi"][:n], P["lo"][:n]
    a = np.flatnonzero(hi >= tp)
    b = np.flatnonzero(lo <= -stop) if stop else np.array([], dtype=int)
    k = n
    if len(b) and (not len(a) or b[0] <= a[0]):
        k = int(b[0])
    elif len(a):
        k = int(a[0])
    return float(hi[:k].max()) if k > 0 else float("-inf")


def s4_cells(P, tp, stop, hold):
    """one hybrid path → dict: today (TP / stop / hold cap), every S4 cell (its TP, the same stop, its cap) and OLD_EXIT (+1 / −4 / 120);
    an n/a cell (cap > hold, or = today) is NaN with why 'N/A'. None if today is UNRES."""
    FX = _fx()
    la = FX.evaluate(P, tp, stop, hold)
    if la["why"] == "UNRES":
        return None
    rec, evs = dict(s4_LIVE=la["pct"], s4_why_LIVE=la["why"], s4_min_LIVE=la["xmin"],
                    s4_peak_LIVE=_peak_before_exit(P, tp, stop, hold), s4_onbar_LIVE=bool(la.get("on_bar", False))), [la]
    for name in [c[0] for c in S4_CELLS] + [OLD_KEY]:
        ctp, x = s4_spec(name, tp)
        if s4_na(name, tp, hold, stop):                              # never folded into today: n/a
            rec[f"s4_{name}"], rec[f"s4_why_{name}"], rec[f"s4_min_{name}"] = np.nan, "N/A", np.nan
            continue
        b = FX.evaluate(P, ctp, OLD_EXIT[1] if name == OLD_KEY else stop, x)
        evs.append(b)
        rec[f"s4_{name}"], rec[f"s4_why_{name}"], rec[f"s4_min_{name}"] = b["pct"], b["why"], b["xmin"]
    rec["s4_n_unres"] = sum(e["why"] == "UNRES" for e in evs)
    rec["_on_ticks"] = bool(P["n_tick"] > 0 and all(e["why"] != "UNRES" and not e["on_bar"] for e in evs))
    return rec


def s4_score_fill(r, tp, stop, hold, tcache=None):
    """→ (rec or None, source 'ticks' / 'ticks+1m' / '1m' / None) on ONE hybrid path (DECISION_LOG 272)."""
    FX = _fx()
    P = FX.load_path(r.pair, int(r.te), float(r.E), _span(hold), tcache)    # max(hold, 120): the old 120-min exit walks the same path
    rec = s4_cells(P, tp, stop, hold)
    if rec is None:
        return None, None
    rec["s4_par_ok"], rec["s4_par_txt"] = parity(r.why, r.pct, {"v": {"LIVE": {"why": rec["s4_why_LIVE"], "pct": rec["s4_LIVE"]}}})
    return rec, ("ticks" if rec.pop("_on_ticks") else FX.path_src(P))


def s4_frame(o, tp, stop, hold, tcache=None):
    rows = []
    for r in o.itertuples():
        w, src = s4_score_fill(r, tp, stop, hold, tcache)
        rows.append(dict(s4_src=src, **(w or {})))
    base = o.reset_index(drop=True)
    if not rows:
        return base.assign(s4_src=pd.Series(dtype=object))
    return pd.concat([base, pd.DataFrame(rows, index=range(len(o)))], axis=1)


def s4_counted(g, name):
    """the fills that count for a cell: today scored AND the cell decided."""
    if not len(g) or "s4_LIVE" not in g:
        return g.iloc[:0]
    return g[g.s4_src.notna() & pd.to_numeric(g[f"s4_{name}"], errors="coerce").notna()]


def s4_untouched(g, name, tp):
    """fills whose outcome the cell cannot change, read from TODAY's walk only (see the module doc): today's exit (TP or stop) fired at ≤ the
    cell's cap, NO print before it reached the cell's TP (s4_peak_LIVE < cell TP), and — for a TP exit — today's TP print itself is ≥ the
    cell's TP (the same first print through both lines). Valid for a cell TP above OR below today's (DECISION_LOG 281: CAP60 +1 vs
    today +1.25) and for OLD_EXIT (same −4 stop)."""
    ctp, x = s4_spec(name, tp)
    m = pd.to_numeric(g.s4_min_LIVE, errors="coerce").values <= x + 1e-9
    w, live = g.s4_why_LIVE.values, g.s4_LIVE.values.astype(float)
    pk = pd.to_numeric(g.s4_peak_LIVE, errors="coerce").values.astype(float) < ctp - 1e-12
    # deep review (DL 281): a TP booked on a 1m BAR is booked AT the line (today +1.25, a +1 cell +1.00) → never untouched unless the TPs match
    ob = g.s4_onbar_LIVE.astype(bool).values if "s4_onbar_LIVE" in g else np.zeros(len(g), bool)
    tp_ok = (live >= ctp - 1e-12) & (~ob | (abs(ctp - float(tp)) < 1e-12))
    return m & pk & ((w == "STOP") | ((w == "TP") & tp_ok))


def s4_judge(g, name, tp):
    """g = counted fills → dict(n, days, mean, sum, p, share, integ, n_untouched, par, d, q)."""
    d = g[f"s4_{name}"].values.astype(float) - g.s4_LIVE.values.astype(float)
    s = float(d.sum()) if len(d) else 0.0
    m = float(d.mean()) if len(d) else float("nan")
    p = NF.p_mean_neg(-d, g.day.values, BOOT_N, BOOT_SEED) if len(d) else None
    share = float(d.max() / s) if s > 0 else float("nan")
    untouched = s4_untouched(g, name, tp) if len(g) else np.zeros(0, bool)
    integ = bool(np.all(np.abs(d[untouched]) < 1e-9))
    par = float(g.s4_par_ok.astype(bool).mean()) if len(g) else 0.0
    q = bool(m > 0 and p is not None and p >= P_MIN and s > 0 and share <= SHARE_MAX and integ and par >= S4_PARITY_MIN)
    return dict(n=int(len(g)), days=int(g.day.nunique()) if len(g) else 0, mean=m, sum=s, p=p, share=share, integ=integ,
                n_untouched=int(untouched.sum()), par=par, d=d, q=q)


def s4_verdict(g, name, tp):
    """the DECISION_LOG 278 bar on the counted fills g → (state, detail, judge dict)."""
    n, nd = len(g), g.day.nunique() if len(g) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"N {n}/{N_MIN} · {nd}/{DAYS_MIN} days", {}
    j = s4_judge(g, name, tp)
    det = (f"N {n} · {nd} d · mean Δ {j['mean']:+.3f} pts · Σ Δ {j['sum']:+.2f} · P(Δ>0) {(j['p'] if j['p'] is not None else float('nan')):.2f} · "
           f"top fill {(j['share'] * 100 if j['share'] == j['share'] else float('nan')):.0f} % of ΣΔ · untouched Δ = 0 on "
           f"{j['n_untouched']} fill(s) {'✓' if j['integ'] else '✗'} · parity {j['par'] * 100:.0f} %")
    if j["q"]:
        return f"{name} CANDIDATE (operator decides)", det, j
    if j["mean"] <= 0:
        return S4_KEEP[name], det, j
    return "KEEP OBSERVING", det, j


def old_verdict(g, tp):
    """the DECISION_LOG 281 revert bar on the counted fills g (today = the NEW exit, OLD_EXIT = +1 / −4 / 120) → (state, detail, j)."""
    n, nd = len(g), g.day.nunique() if len(g) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"N {n}/{N_MIN} · {nd}/{DAYS_MIN} days", {}
    j = s4_judge(g, OLD_KEY, tp)
    new_avg, old_avg = float(g.s4_LIVE.astype(float).mean()), float(g[f"s4_{OLD_KEY}"].astype(float).mean())
    j.update(new_avg=new_avg, old_avg=old_avg)
    better = bool(j["mean"] > 0 and j["p"] is not None and j["p"] >= P_MIN and j["share"] <= SHARE_MAX and j["par"] >= S4_PARITY_MIN)
    det = (f"N {n} · {nd} d · new avg {new_avg:+.3f} % vs old {old_avg:+.3f} % · mean Δ (old − new) {j['mean']:+.3f} pts · Σ Δ "
           f"{j['sum']:+.2f} · P(Δ>0) {(j['p'] if j['p'] is not None else float('nan')):.2f} · top fill "
           f"{(j['share'] * 100 if j['share'] == j['share'] else float('nan')):.0f} % of ΣΔ · untouched Δ = 0 on {j['n_untouched']} fill(s) "
           f"{'✓' if j['integ'] else '✗'} · parity {j['par'] * 100:.0f} %")
    if better or new_avg < 0:
        return OLD_REVERT, det + (" · old better" if better else "") + (" · new avg < 0" if new_avg < 0 else ""), j
    return OLD_KEEP, det, j


def s4_freeze(st, name, g, now_iso, cfg, hold=None, why=None, dropped=None, undecided=None):
    """freeze a cell ONCE under st[S4_KEY[name]] at the first prefix (N ≥ 20 counted fills on ≥ 8 UTC days); deferred while a fill opened ≤
    the prefix end is OPEN, unpriced-but-not-final or 1m-provisional (hold); NEVER frozen while the prefix fails walker parity (< 90 %) or
    the untouched-integrity check (any verdict, KEEP included). dropped / undecided = open times of final fills with no path / with this
    cell undecided (recorded ≤ the prefix end). Never touches any other key (the CAP15 / CAP20 frozen verdicts, the other S4 cells)."""
    key = OLD_KEY if name == OLD_KEY else S4_KEY[name]
    if key in st:
        return st, False
    pre = prefix(g, N_MIN)
    if pre is None:
        return st, False
    last = pre.ts.max()
    w = []
    if NF.freeze_hold(last, hold, w):
        if why is not None:
            why += [f"{name} — {s}" for s in w]
        return st, False
    state, det, j = old_verdict(pre, cfg[0]) if name == OLD_KEY else s4_verdict(pre, name, cfg[0])
    if j and (j["par"] < S4_PARITY_MIN or not j["integ"]):
        if why is not None:
            why.append(f"{name} — WALKER PARITY / INTEGRITY FAILED — not frozen (parity {j['par'] * 100:.0f} %, untouched Δ = 0 "
                       f"{'✓' if j['integ'] else '✗'})")
        return st, False
    ctp, x = s4_spec(name, cfg[0])
    st[key] = dict(state=state, detail=det, bar=OLD_LABEL if name == OLD_KEY else S4_LABEL, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso, n=len(pre),
                   days=int(pre.day.nunique()), dropped_no_data=int(sum(1 for t in (dropped or []) if pd.notna(t) and t <= last)),
                   undecided_final=int(sum(1 for t in (undecided or []) if pd.notna(t) and t <= last)),
                   config=dict(tp=cfg[0], stop=cfg[1], hold=cfg[2], cell_tp=ctp, cap=x),
                   keys=[f"{a}|{b}" for a, b in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str))])
    return st, True


def _s4_val(r, n):
    v = getattr(r, f"s4_{n}")
    if v != v:
        return f"{n} n/a" if getattr(r, f"s4_why_{n}") == "N/A" else f"{n} undecided"
    return f"{n} {v:+.2f} ({getattr(r, f's4_why_{n}')}, Δ {v - r.s4_LIVE:+.2f})"


def _s4_fill_txt(r):
    return (f"{r.ts:%m-%d %H:%M} {r.pair} [{r.s4_src}{'' if r.s4_src == 'ticks' else ', provisional'}] today {r.s4_LIVE:+.2f} ({r.s4_why_LIVE} @ "
            f"{r.s4_min_LIVE:.0f} min) → " + " / ".join(_s4_val(r, n) for n, _, _ in S4_CELLS))


def s4_section(o, opens, now_ms, levels_, tcache, st, ok):
    """→ (markdown lines, state, changed). o = the line's closed WILLY fills (≥ the DL 251 floor) with te / E / why / pct."""
    tp, stop, hold = levels_
    FX = _fx()
    floor = s4_deploy_ts()
    manual = o.why.astype(str).str.upper().str.startswith("MANUAL")          # operator closes: not the walked exit (each = a parity miss)
    man = o[manual & (o.ts >= floor)]
    o = o[~manual].reset_index(drop=True)
    s4 = s4_frame(o, tp, stop, hold, tcache)
    done = load_attempts()["done"]
    s4["s4_pend"] = [FX._pending(s, getattr(r, "s4_n_unres", 0), ticks_pending(r, hold, done, now_ms, tcache),
                                 _kk(r.pair, r.te, _span(hold)) not in done) for s, r in zip(s4.s4_src, s4.itertuples())]
    coh = s4[s4.ts >= floor]
    ctx = s4[(s4.ts < floor) & s4.s4_src.notna()]
    today = f"TP +{tp:g} · {'no stop' if not stop else f'stop −{stop:g}'} · {hold} min"

    def lab(n):
        ctp, x = s4_spec(n, tp)
        return f"{n} (+{ctp:g} / −{stop if stop else 0:g} / {x} min)"
    L = [f"### ⏱ CAP60 / CAP90 / T125C60 / T125C90 on today's stop (DECISION_LOG 278 — OBSERVE only; own cohort + own frozen bar, CAP15 / "
         f"CAP20 above untouched)", "",
         f"Cohort: CLOSED non-MANUAL {STRATEGY} fills counted from the DECISION_LOG 281 deploy (today = {today}) — opened ≥ "
         f"{floor:%Y-%m-%d %H:%M:%S} UTC (the \"{S4_DEPLOY_GREP}\" commit + {S4_LAG_MIN} min"
         + ("" if _git_ct(S4_DEPLOY_GREP) is not None else f"; commit not found → PLACEHOLDER pin {S4_DEPLOY_PIN} UTC") + "), same loader / "
         f"dedup as above; one live exit, so parity is read against the exit the cells are compared with. Cells: the same stop; CAP60 / "
         f"CAP90 today's TP, T125C60 / T125C90 TP +1.25; closed at market at the cap if neither line was hit; a cell with cap > today's hold, "
         f"or identical to today, is n/a. Walker: the DECISION_LOG 272 hybrid (ticks per covered UTC day, then the 1m tail = provisional; "
         f"tick ruler = the first print through a line); a cell undecided at the path end is not counted. Untouched = today's exit (TP or "
         f"−4) fired ≤ the cell's cap with no earlier print through the cell's TP, and a TP print already through it (these must show "
         f"Δ = 0). \"KEEP 120\" in the CAP60 / CAP90 bar text = keep today's exit.", "",
         "| Cell | N counted | days | mean Δ vs today (pts) | Σ Δ | P(Δ>0) day-bootstrap | top fill share of Σ Δ | untouched fills Δ = 0 | parity |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, _, _ in S4_CELLS:
        g = s4_counted(coh, n)
        if s4_na(n, tp, hold):
            L.append(f"| {lab(n)} | n/a ({s4_na(n, tp, hold)}{f' {hold} min' if s4_na(n, tp, hold) == 'cap > hold' else ''}) | – | – | – | – | – | – | – |")
            continue
        if not len(g):
            L.append(f"| {lab(n)} | 0 | 0 | – | – | – | – | – | – |")
            continue
        j = s4_judge(g, n, tp)
        ptxt = "–" if j["p"] is None else f"{j['p']:.2f}"
        stxt = "–" if j["share"] != j["share"] else f"{j['share'] * 100:.0f} %"
        L.append(f"| {lab(n)} | {j['n']} | {j['days']} | {j['mean']:+.3f} | {j['sum']:+.2f} | "
                 f"{ptxt} | {stxt} | {'✓' if j['integ'] else '✗'} ({j['n_untouched']}) | {j['par'] * 100:.0f} % |")
    L.append("")
    sc = coh[coh.s4_src.notna()]
    if len(sc):
        L.append("Per fill (counted cohort): " + " · ".join(_s4_fill_txt(r) for r in sc.itertuples()))
        par = sc.s4_par_ok.astype(bool)
        L.append(f"Walker parity (today's walk vs the actual exit, reason + % within {PARITY_TOL} pts): {int(par.sum())}/{len(sc)}"
                 + ("" if par.all() else " — off: " + " · ".join(f"{r.pair} {r.s4_par_txt}" for r in sc[~par].itertuples())))
    else:
        L.append("Per fill (counted cohort): none yet — no WILLY fill closed since the DECISION_LOG 281 deploy.")
    uns = coh[coh.s4_src.isna()]
    if len(man):
        L.append(f"Excluded: {len(man)} manually closed fill(s) (not the walked exit): "
                 + ", ".join(f"{r.ts:%m-%d %H:%M} {r.pair} ({r.why})" for r in man.itertuples()) + ".")

    def undecided(n):
        """final fills the cell leaves undecided: today decided, the cell UNRES, no data still due."""
        if not len(coh) or f"s4_{n}" not in coh:
            return []
        m = coh.s4_src.notna() & pd.to_numeric(coh[f"s4_{n}"], errors="coerce").isna() & ~coh.s4_pend.astype(bool)
        return list(coh[m].ts)
    if len(uns):
        L.append("UNPRICED (no path yet): " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair}" + ("" if r.s4_pend else " (final — no data)")
                                                      for r in uns.itertuples()))
    if len(ctx):
        L.append(f"Context, NOT counted ({len(ctx)} fill(s) opened before the DECISION_LOG 281 deploy, walked with today's {today}; parity is vs "
                 f"what they really did): " + " · ".join(_s4_fill_txt(r) for r in ctx.itertuples()) + " · mean Δ "
                 + " / ".join(f"{n} {(pd.to_numeric(ctx[f's4_{n}'], errors='coerce') - ctx.s4_LIVE).mean():+.3f}" for n, _, _ in S4_CELLS
                              if not s4_na(n, tp, hold))
                 + f" · today avg {ctx.s4_LIVE.mean():+.3f} vs " + " / ".join(f"{n} {pd.to_numeric(ctx[f's4_{n}'], errors='coerce').mean():+.3f}"
                                                                           for n, _, _ in S4_CELLS if not s4_na(n, tp, hold)) + ".")
    L += [f"CAP60 / CAP90 — {S4_BAR}", f"T125C60 / T125C90 — {S4_BAR_T125}"]
    why, changed = [], False
    hold_l = [x for x in opens if pd.notna(x[0]) and x[0] >= floor] + [
        (r.ts, f"{'unpriced' if not isinstance(r.s4_src, str) else 'undecided cell' if (getattr(r, 's4_n_unres', 0) or 0) > 0 else '1m-provisional'} {r.pair}")
        for r in coh.itertuples() if r.s4_pend]
    if ok and stop:
        for n, _, _ in S4_CELLS:
            if s4_na(n, tp, hold):
                continue
            st, ch = s4_freeze(st, n, s4_counted(coh, n), pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), (tp, stop, hold),
                               hold_l, why, dropped=list(uns[~uns.s4_pend.astype(bool)].ts) if len(uns) else [], undecided=undecided(n))
            changed |= ch
    elif ok:
        why.append("today's WILLY exit has no stop in trading_config.json — the cells are defined on the −4 stop, nothing frozen")
    for n, _, _ in S4_CELLS:
        f0 = st.get(S4_KEY[n]) if ok else None
        nu = len(undecided(n))
        if s4_na(n, tp, hold) and not f0:
            L.append(f"{n}: n/a ({s4_na(n, tp, hold)}) — today's exit is {today}; not judged, not frozen.")
            continue
        v = s4_verdict(s4_counted(coh, n), n, tp)
        if f0:
            L.append(f"{n}: **Frozen verdict ({f0['bar']}; prefix to {f0['at']}, frozen on the run of {f0['run_at']}, N {f0['n']} · "
                     f"{f0['days']} d, {f0.get('dropped_no_data', 0)} final-no-data fill(s) excluded, {f0.get('undecided_final', 0)} final "
                     f"fill(s) left undecided by this cell, today then TP +{f0['config']['tp']:g} / "
                     f"stop −{f0['config']['stop']:g} / {f0['config']['hold']} min): {f0['state']}** ({f0['detail']}) · live (information only, "
                     f"never re-decides): {v[0]} — {v[1]}")
        else:
            L.append(f"{n}: ⏳ {v[0]} ({v[1]}){f' · {nu} final fill(s) left undecided by this cell (not counted)' if nu else ''}.")
    if why:
        L.append("⏸ S4 freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    L.append("Δ is per fill on the SAME entries: a shorter cap / different TP changes when the one WILLY slot (and the global hold) frees — "
             "the extra trades are not priced here (the year study's re-sequenced book is the reference for that).")
    # ── ↩ OLD_EXIT revert gate (DECISION_LOG 281) — own section, own key
    otp, ost, ocap = OLD_EXIT
    L += ["", f"### ↩ WILLY exit change revert gate (DL 281): new +{tp:g} / −{stop if stop else 0:g} / {hold} vs old +{otp:g} / −{ost:g} / "
              f"{ocap}", "",
          f"Every counted fill (same cohort as above: opened ≥ {floor:%Y-%m-%d %H:%M:%S} UTC, non-MANUAL) re-walked from its own entry on ONE "
          f"hybrid path (ticks first) under the NEW exit (today, trading_config.json: {today}) and the OLD exit (+{otp:g} / −{ost:g} / "
          f"{ocap} min). Δ = old − new per fill; parity = the new walk vs each fill's actual exit.", ""]
    na_old = s4_na(OLD_KEY, tp, hold, stop)
    go = s4_counted(coh, OLD_KEY)
    if na_old:
        L.append(f"OLD_EXIT: n/a (= today) — trading_config.json still runs the old exit ({today}); nothing to compare, nothing frozen.")
    else:
        if len(go):
            jo = s4_judge(go, OLD_KEY, tp)
            L += ["| N counted | days | new avg % (walked) | old avg % | mean Δ old − new (pts) | Σ Δ | P(Δ>0) | top fill share | untouched Δ = 0 | parity |",
                  "|---|---|---|---|---|---|---|---|---|---|",
                  f"| {jo['n']} | {jo['days']} | {go.s4_LIVE.astype(float).mean():+.3f} | {go[f's4_{OLD_KEY}'].astype(float).mean():+.3f} | "
                  f"{jo['mean']:+.3f} | {jo['sum']:+.2f} | {'–' if jo['p'] is None else format(jo['p'], '.2f')} | "
                  f"{'–' if jo['share'] != jo['share'] else format(jo['share'] * 100, '.0f') + ' %'} | {'✓' if jo['integ'] else '✗'} "
                  f"({jo['n_untouched']}) | {jo['par'] * 100:.0f} % |", ""]
            L.append("Per fill: " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair} [{r.s4_src}{'' if r.s4_src == 'ticks' else ', provisional'}] new "
                                               f"{r.s4_LIVE:+.2f} ({r.s4_why_LIVE} @ {r.s4_min_LIVE:.0f} min) vs " + _s4_val(r, OLD_KEY).replace(OLD_KEY, "old", 1)
                                               for r in go.itertuples()))
        else:
            L.append("Counted: none yet — no WILLY fill closed since the DECISION_LOG 281 deploy.")
        nuo = len(undecided(OLD_KEY))
        if nuo:
            L.append(f"{nuo} final fill(s) left undecided by the old exit (not counted).")
    if len(ctx) and f"s4_{OLD_KEY}" in ctx and pd.to_numeric(ctx[f"s4_{OLD_KEY}"], errors="coerce").notna().any():
        cx = ctx[pd.to_numeric(ctx[f"s4_{OLD_KEY}"], errors="coerce").notna()]
        L.append(f"Context, NOT counted ({len(cx)} fill(s) before the deploy; they ran the old exit live): new walk avg "
                 f"{cx.s4_LIVE.astype(float).mean():+.3f} vs old walk avg {cx[f's4_{OLD_KEY}'].astype(float).mean():+.3f} — per fill Δ (old − new): "
                 + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair} {getattr(r, f's4_{OLD_KEY}') - r.s4_LIVE:+.2f}" for r in cx.itertuples()) + ".")
    L.append(OLD_BAR)
    whyo = []
    if ok and stop and not na_old:
        st, ch = s4_freeze(st, OLD_KEY, go, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), (tp, stop, hold), hold_l, whyo,
                           dropped=list(uns[~uns.s4_pend.astype(bool)].ts) if len(uns) else [], undecided=undecided(OLD_KEY))
        changed |= ch
    f0 = st.get(OLD_KEY) if ok else None
    vo = old_verdict(go, tp) if not na_old else ("n/a (= today)", "", {})
    if f0:
        L.append(f"**Frozen revert verdict ({f0['bar']}; prefix to {f0['at']}, frozen on the run of {f0['run_at']}, N {f0['n']} · {f0['days']} d, "
                 f"{f0.get('dropped_no_data', 0)} final-no-data fill(s) excluded, {f0.get('undecided_final', 0)} left undecided by the old exit, "
                 f"new exit then +{f0['config']['tp']:g} / −{f0['config']['stop']:g} / {f0['config']['hold']} min): {f0['state']}** "
                 f"({f0['detail']}) · live (information only, never re-decides): {vo[0]} — {vo[1]}")
    elif not na_old:
        L.append(f"Revert gate now: ⏳ {vo[0]} ({vo[1]}).")
    if whyo:
        L.append("⏸ revert-gate freezing deferred this run: " + " · ".join(whyo) + " — re-checked next run.")
    return L, st, changed


# ─────────────────────────── rendering ───────────────────────────
def _vrow(lab, g, v):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – | – | – |"
    p = g[v]
    usd = (p * g.notional / 100).sum()
    tp = (g[f"why_{v}"] == "TP").mean() * 100
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(p > 0).mean() * 100:.0f} % | {p.mean():+.3f} % | {p.sum():+.2f} % | "
            f"{usd:+,.0f} | {g[f'worst_{v}'].min():+.2f} % | {tp:.0f} % |")


VHDR = ["| Variant | N | days | WR | avg % | Σ % | Σ$ at the fill's size | worst point before close | closed at TP |",
        "|---|---|---|---|---|---|---|---|---|"]


def summary_block(sc, hold, label_live):
    L = [*VHDR]
    for v, x in _variants(hold):
        lab = (f"**{label_live}**" if v == "LIVE" else (f"**{v}**" if x in VERDICT_CAPS else f"{v} (context)"))
        L.append(_vrow(lab, sc, v))
    return L


def run(now_ms=None, orders=None, fetch=True, state_path=None, open_ts=None, th=None, study=True):
    now_ms = now_ms or int(time.time() * 1000)
    state_path = state_path or STATE
    _K1.clear()
    tp, stop, hold = levels(th)
    o, opens = load_orders(orders)
    if open_ts is not None:
        opens = list(open_ts)
    floor = deploy_ts()
    o = o[o.ts >= floor].reset_index(drop=True)
    tc = {}                                                        # one tick cache for the source pass, the walk and the lag checks
    o["src"] = [score_fill(r, hold, tp, stop, tc)[1] for r in o.itertuples()]
    notes = ensure_data(o, hold, now_ms, fetch=fetch)
    if notes:
        tc.clear()                                                 # a slice may have landed for a day already cached
    sc_all = scored_frame(o.drop(columns=["src"]), hold, tp, stop, tc)
    done = load_attempts()["done"]
    # lagging = could still change: unscored unless BOTH paths are final (1m page final ∧ every missing tick day's archive final);
    # 1m-provisional while a tick archive is still pending
    sc_all["pend"] = [(not isinstance(s, str) and not (_kk(r.pair, r.te, _span(hold)) in done and not ticks_pending(r, hold, done, now_ms, tc)))
                      or (s == "1m" and ticks_pending(r, hold, done, now_ms, tc)) for s, r in zip(sc_all.src, sc_all.itertuples())]
    sc = sc_all[sc_all.src.notna()].copy()
    uns = sc_all[sc_all.src.isna()]
    live_lab = f"LIVE (TP +{tp:g} · {'no stop' if not stop else f'stop −{stop:g}'} · {hold} min)"
    L = [f"## ⏱ WILLY_TIMECAP — FRENZY_WILLY: TP within 15 / 20 min else close, vs today's {hold}-min cap (operator hypothesis, "
         f"DECISION_LOG 264, OBSERVE only, exit shadow)", "",
         f"Cohort: CLOSED {STRATEGY} fills opened ≥ {floor:%Y-%m-%d %H:%M} UTC (the sleeve's ship), dedup (opened_at, pair, direction). "
         f"Walked from each fill's own entry on aggTrades (1m klines = provisional) with the bot's accounting (net = gross − 0.045 − "
         f"0.045 × exit/entry). Live exit read from trading_config.json: {live_lab}. Variants differ only on fills WITHOUT a TP inside "
         f"the cap.", ""]
    L += summary_block(sc, hold, live_lab) + [""]
    if len(sc):
        L.append("By trigger: " + " · ".join(
            f"{t}: " + ", ".join(f"{v} {len(g)} · {g[v].mean():+.3f} %" for v in ("LIVE", "CAP15", "CAP20"))
            for t, g in sc.groupby("trig")))
        for x in VERDICT_CAPS:
            L.append(f"TP reached within {x} min: {(sc[f'why_CAP{x}'] == 'TP').sum()}/{len(sc)} ({(sc[f'why_CAP{x}'] == 'TP').mean() * 100:.0f} %).")
        L.append("")
        for x in VERDICT_CAPS:
            nt = sc[sc[f"why_CAP{x}"] != "TP"]
            if len(nt):
                L.append(f"**No TP within {x} min ({len(nt)} fills) — % at minute {x} vs the live outcome:** " + " · ".join(
                    f"{r.ts:%m-%d %H:%M} {r.pair} {r.trig}: at {x} min {getattr(r, f'CAP{x}'):+.2f} % vs live-walk {r.LIVE:+.2f} % "
                    f"({r.why_LIVE}{'' if r.why_LIVE != 'TP' else f' @ {r.min_LIVE:.0f} min'}) → Δ {getattr(r, f'CAP{x}') - r.LIVE:+.2f}"
                    for r in nt.itertuples()) + f" · Σ Δ {(nt[f'CAP{x}'] - nt.LIVE).sum():+.2f} pts")
            else:
                L.append(f"No TP within {x} min: none yet (every fill took its TP inside {x} min → CAP{x} ≡ LIVE so far).")
        L.append("")
        L.append("Per fill: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair} {r.trig} [{r.src}{', provisional' if r.src == '1m' else ''}] TP@"
            f"{'–' if r.tp_min != r.tp_min else f'{r.tp_min:.1f} min'} · path if still held at 15/20/30/60/120 min: "
            + "/".join(f"{getattr(r, f'at{x}'):+.2f}" for x in MARKS) + f" · worst before the live close {r.worst_LIVE:+.2f}"
            for r in sc.itertuples()))
        dd = sc.assign(d15=sc.CAP15 - sc.LIVE, d20=sc.CAP20 - sc.LIVE).groupby("day")[["d15", "d20"]].agg(["count", "sum"])
        L.append("Day units (Σ Δ per day, pts): " + " · ".join(
            f"{d} n{int(r[('d15', 'count')])} Δ15 {r[('d15', 'sum')]:+.2f} / Δ20 {r[('d20', 'sum')]:+.2f}" for d, r in dd.iterrows()))
        par = sc.par_ok.astype(bool)
        L.append(f"Walker parity (LIVE walk vs the actual exit, reason + % within {PARITY_TOL} pts): {int(par.sum())}/{len(sc)} match — "
                 + " · ".join(f"{r.pair} {r.par_txt}{'' if r.par_ok else ' ✗'}" for r in sc.itertuples()))
        if (sc.src == "1m").any():
            L.append(f"⏳ {int((sc.src == '1m').sum())} fill(s) walked on 1m klines (PROVISIONAL: TP booked at exactly +{tp:g}; the entry "
                     "minute's bar is skipped, so a TP in the first ≤ 59 s after the fill is not seen there; the cap closes at the close of "
                     "the last bar closed ≤ the cap, i.e. up to 59 s early) — re-walked on aggTrades once the daily archive is published "
                     "(≥ 24 h after the UTC day ends; a 404 is retried every ≥ 3 h).")
        L.append("")
    if len(uns):
        L.append("UNSCORED (no path yet): " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair}" + ("" if r.pend else " (final — no data)")
                                                       for r in uns.itertuples()))
    hold_list = list(opens) + [(r.ts, f"{'1m-provisional' if r.src == '1m' else 'unscored'} {r.pair}") for r in sc_all.itertuples() if r.pend]
    why = []
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok:
        changed_any = False
        for _ in range(2):                                     # 'first' then (a LATER run only) 'reread'
            had = set(st)
            st, ch = freeze(st, sc, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), (tp, stop, hold), hold_list, why,
                            dropped=list(uns[~uns.pend.astype(bool)].ts))
            changed_any |= ch
            if not ch or "first" not in had:
                break
        if changed_any:
            NF.du_save_state(st, state_path)
    try:                                                       # S4 cells (DECISION_LOG 278): own cohort, own keys in the same state file
        L4, st, ch4 = s4_section(o.drop(columns=["src"]), opens, now_ms, (tp, stop, hold), tc, st, ok)
        if ok and ch4:
            NF.du_save_state(st, state_path)
    except Exception as e:
        L4 = [f"### ⏱ CAP60 / CAP90 / T125C60 / T125C90 on today's stop + the DL 281 revert gate (DECISION_LOG 278 / 281) — unavailable this run ({type(e).__name__}: {str(e)[:160]})"]
    live_state, live_det, _ = verdict(sc)
    bar = (f"Bar (pre-registered, frozen once at the first prefix of N ≥ {N_MIN} closed WILLY fills on ≥ {DAYS_MIN} days, never re-fit; "
           f"one re-read at N ≥ {REREAD_N}): per CAP15 / CAP20, Δ = CAPx − LIVE per fill; CAPx CANDIDATE (operator decides) iff mean Δ > 0 ∧ "
           f"day-clustered P(mean Δ > 0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no fill > {SHARE_MAX * 100:.0f} % of Σ Δ ∧ Σ Δ > 0 on "
           f"the no-TP-in-cap fills (TP-in-cap fills must show Δ = 0); both → larger Δ (tie → CAP15); both ≤ 0 → KEEP 120; else keep observing.")
    L.append("Δ is per fill on the SAME entries: it excludes the extra trades a shorter cap would let through (the single WILLY slot "
             "and the global hold — no other automated trade while a WILLY is open — free up sooner; see the WILLY_HOLD tracker).")
    n_drop = int((~uns.pend.astype(bool)).sum()) if len(uns) else 0
    L.append(f"Live so far: {len(sc)} scored WILLY fills on {sc.day.nunique() if len(sc) else 0} day(s)"
             + (f" ({parity_gate(sc)[1]})" if len(sc) else "") + f" · {n_drop} fill(s) final — no data, excluded from the cohort / prefix"
             + f" — information only, no read below N ≥ {N_MIN} on ≥ {DAYS_MIN} days.")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    if "first" not in st:
        L.append(f"{bar} Now: ⏳ {live_state} ({live_det}).")
    else:
        for k in ("first", "reread"):
            if k in st:
                f0 = st[k]
                L.append(f"**Frozen {'verdict' if k == 'first' else 're-read'} (prefix to {f0['at']}, frozen on the run of {f0['run_at']}, "
                         f"N {f0['n']} · {f0['days']} d, {f0.get('parity', '')}, {f0.get('dropped_no_data', 0)} final-no-data fill(s) "
                         f"excluded, live exit then TP +{f0['config']['tp']:g} / hold {f0['config']['hold']}): "
                         f"{f0['state']}** ({f0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    L.append(REVERT)
    if notes:
        L.append(f"Data: {' · '.join(notes)}.")
    L += [""] + L4
    if study:
        try:
            L += [""] + study_block()
        except Exception as e:
            L.append(f"Study reference unavailable ({str(e)[:100]}).")
    return L + [""]


# ─────────────────────────── study reference (cached ticks only) ───────────────────────────
def _study_key():
    s = os.stat(STUDY_CSV)
    return f"{int(s.st_mtime)}|{s.st_size}"


def _study_rows_path():
    return os.path.join(MY_CACHE, "study_ref_rows.csv")


def build_study_ref(tp=1.0, hold=120):
    """re-walk the study's FRENZY_WILLY A cohort (NOSTOP variant = today's live design, its own sequencing, st ok) from CACHED ticks only
    (ticks_q then ticks, no fetch) with this walker → per-row CSV in this line's cache (keyed by the study CSV's mtime + size)."""
    D = pd.read_csv(STUDY_CSV)
    S = D[(D.variant == "NOSTOP") & (D.st == "ok") & (D.trigger == "A")].reset_index(drop=True)
    cache, rows = {}, []
    for r in S.sort_values(["pair", "te"]).itertuples():
        te = int(r.te)
        t, p = load_ticks(r.pair, _days(te, hold), cache)
        if len(cache) > 6:
            cache.clear()
        if t is None:
            rows.append(dict(pair=r.pair, te=te, day=r.day, study_pct=r.pct, ok=False))
            continue
        k = int(np.searchsorted(t, te, side="left"))                # the study's entry print t[k] == te; it walks t[k+1:]
        j = int(np.searchsorted(t, te + hold * MIN, side="right"))
        w = walk_ticks(t[k + 1:j], p[k + 1:j], te, float(r.E), tp, None, hold)
        rec = dict(pair=r.pair, te=te, day=r.day, study_pct=r.pct, ok=True, tp_min=w["tp_min"])
        for v, res in w["v"].items():
            rec[v], rec[f"why_{v}"], rec[f"worst_{v}"] = res["pct"], res["why"], res["worst"]
        rows.append(rec)
    R = pd.DataFrame(rows)
    os.makedirs(MY_CACHE, exist_ok=True)
    R.assign(_key=_study_key()).to_csv(_study_rows_path(), index=False)
    return R


def study_block():
    head = "**Study reference, not part of the verdict** — FRENZY_WILLY A, yr5 tick study (reports/FRENZY_TP3_VS_TP4_TICKS_2026-10-08_willy.csv, "
    if not os.path.exists(STUDY_CSV):
        return [head + "file absent)."]
    p = _study_rows_path()
    try:
        R = pd.read_csv(p) if os.path.exists(p) else None
    except Exception:
        R = None
    if R is None or not len(R) or str(R["_key"].iloc[0]) != _study_key():
        R = build_study_ref()                                       # ~10 s, cached ticks only (no fetch); keyed by the study file
    miss = int((~R.ok.astype(bool)).sum())
    R = R[R.ok.astype(bool)].copy()
    par = (np.abs(R.LIVE - R.study_pct) <= 1e-6).mean() * 100
    L = [head + f"NOSTOP-sequenced entries re-walked from cached ticks; the study's own 120-min sequencing kept — shorter caps would free "
         f"the one WILLY slot earlier, not modelled). N {len(R)} on {R.day.nunique()} days ({miss} rows without cached ticks); LIVE walk "
         f"reproduces the study's NOSTOP % on {par:.1f} % of rows.", "", *VHDR]
    for v, x in _variants(120):
        g = R
        pp = g[v]
        L.append(f"| {v if v != 'LIVE' else 'LIVE (TP +1 · no stop · 120)'}{' (context)' if x not in VERDICT_CAPS and v != 'LIVE' else ''} | "
                 f"{len(g)} | {g.day.nunique()} | {(pp > 0).mean() * 100:.0f} % | {pp.mean():+.3f} % | {pp.sum():+.1f} % | – | "
                 f"{g[f'worst_{v}'].min():+.2f} % | {(g[f'why_{v}'] == 'TP').mean() * 100:.0f} % |")
    for x in VERDICT_CAPS:
        nt = R[R[f"why_CAP{x}"] != "TP"]
        d = R[f"CAP{x}"] - R.LIVE
        dt = d[R[f"why_CAP{x}"] == "TP"]
        pr = p_delta_pos(d.values, R.day.values)
        L.append(f"CAP{x}: TP within {x} min {(R[f'why_CAP{x}'] == 'TP').mean() * 100:.0f} %; the {len(nt)} without: at {x} min "
                 f"{nt[f'CAP{x}'].mean():+.3f} % vs live {nt.LIVE.mean():+.3f} % (live: {(nt.why_LIVE == 'TP').mean() * 100:.0f} % still "
                 f"took the TP later); mean Δ per fill {d.mean():+.3f} pts, Σ Δ {d.sum():+.1f}, day-clustered P(Δ>0) {pr:.2f}; "
                 f"TP-in-cap Δ = 0 on {int((dt.abs() < 1e-9).sum())}/{len(dt)}.")
    return L


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    global _NET_BLOCKED, _CACHE, MY_CACHE, STATE, EXPORT_GLOB, _http_get, _open_url, STUDY_CSV, S4_DEPLOY_GREP, S4_DEPLOY_PIN
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved = (_CACHE, MY_CACHE, STATE, EXPORT_GLOB, _http_get, _open_url, STUDY_CSV, NF.SHARED_COOLDOWN, NF.DU_COOLDOWN, NF._CACHE,
             S4_DEPLOY_GREP, S4_DEPLOY_PIN)
    _NET_BLOCKED = True
    try:
        chk((CAPS, VERDICT_CAPS, N_MIN, DAYS_MIN, REREAD_N, P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED, PARITY_TOL, MAX_REQ, WEIGHT_STOP, BUDGET_S, TAKER)
            == ((15, 20, 30, 60), (15, 20), 20, 8, 40, 0.90, 0.50, 4000, 7, 0.05, 2, 900, 25.0, 0.045), "pre-registered constants pinned")
        for fn in (lambda: _fetch_1m("X", 0, 1, 0), lambda: _fetch_archive("X", "2026-01-01", 0)):  # both refuse while blocked
            try:
                fn()
                chk(False, "fetch must be blocked")
            except RuntimeError as e:
                chk("blocked" in str(e), "network blocked inside the selftest")
        # accounting = the live exits (KAIA, W 2026-10-09)
        chk(abs(float(net_pct(0.04776, 0.04724)) - 1.0102667) < 1e-5 and abs(float(net_pct(0.018516, 0.018316)) - 1.0014501) < 1e-5,
            "net % reproduces the live KAIA / W pnl_percentage")
        with open(CONFIG) as f:
            real = json.load(f)
        rt = real.get("thresholds", {})
        chk("frenzy_willy_tp_pct" in rt and levels() == levels(real) == levels(rt) and levels()[0] == float(rt["frenzy_willy_tp_pct"])
            and levels()[2] == int(round(float(rt["frenzy_willy_max_hold_minutes"]))),
            f"levels() reads trading_config.json['thresholds'] on the real file ({levels()})")
        chk(levels({"thresholds": {"frenzy_willy_tp_pct": 2.0, "frenzy_willy_max_hold_minutes": 60}, "frenzy_willy_tp_pct": 9.0}) == (2.0, None, 60),
            "the thresholds block wins over a top-level key")
        chk(levels({}) == (1.0, None, 120) and levels({"frenzy_willy_tp_pct": 0, "frenzy_willy_stop_pct": 0, "frenzy_willy_max_hold_minutes": 0})
            == (1.0, None, 120) and levels({"frenzy_willy_stop_pct": -2.5, "frenzy_willy_max_hold_minutes": 60}) == (1.0, 2.5, 60),
            "config levels = services/frenzy.frenzy_willy_levels semantics")
        E, te = 100.0, 1_800_000_000_000
        tp_px = 101.2                                            # net ≈ +1.11
        # synthetic ticks: TP at minute 5 → identical across variants
        t = te + np.array([30_000, 2 * MIN, 5 * MIN, 30 * MIN])
        w = walk_ticks(t, [99.0, 98.0, tp_px, 90.0], te, E, 1.0, None, 120)
        chk(all(w["v"][v]["why"] == "TP" and abs(w["v"][v]["pct"] - w["v"]["LIVE"]["pct"]) < 1e-12 for v in w["v"])
            and abs(w["tp_min"] - 5) < 1e-9 and abs(w["v"]["LIVE"]["worst"] - float(net_pct(98.0, E))) < 1e-12,
            "TP before 15 min → identical across every variant, worst = the dip before it")
        # no TP: close at the last print ≤ 15 / 20 vs the 120 outcome
        t = te + np.array([1 * MIN, 14 * MIN, 15 * MIN, 19 * MIN, 21 * MIN, 119 * MIN, 121 * MIN])
        p = [99.5, 99.0, 98.5, 99.7, 97.0, 96.0, 105.0]
        w = walk_ticks(t, p, te, E, 1.0, None, 120)
        chk(w["v"]["CAP15"]["why"] == "CAP" and abs(w["v"]["CAP15"]["pct"] - float(net_pct(98.5, E))) < 1e-12
            and abs(w["v"]["CAP20"]["pct"] - float(net_pct(99.7, E))) < 1e-12 and abs(w["v"]["LIVE"]["pct"] - float(net_pct(96.0, E))) < 1e-12
            and w["v"]["LIVE"]["why"] == "CAP" and w["tp_min"] != w["tp_min"] and abs(w["marks"][60] - float(net_pct(97.0, E))) < 1e-12,
            "no TP → CAPx closes at the last print ≤ x min, LIVE at the last ≤ 120 (a print after 120 ignored)")
        # TP at minute 18: CAP15 closes at 15, CAP20 / LIVE take the TP
        t = te + np.array([10 * MIN, 15 * MIN, 18 * MIN])
        w = walk_ticks(t, [99.0, 98.0, tp_px], te, E, 1.0, None, 120)
        chk(w["v"]["CAP15"]["why"] == "CAP" and abs(w["v"]["CAP15"]["pct"] - float(net_pct(98.0, E))) < 1e-12
            and w["v"]["CAP20"]["why"] == "TP" and w["v"]["CAP20"]["pct"] == w["v"]["LIVE"]["pct"], "TP at 18 min: CAP15 cap, CAP20 = LIVE")
        ws = walk_ticks(t, [97.0, 98.0, tp_px], te, E, 1.0, 2.0, 120)
        chk(ws["v"]["LIVE"]["why"] == "STOP" and abs(ws["v"]["LIVE"]["pct"] - float(net_pct(97.0, E))) < 1e-12, "a configured stop fires first")
        we = walk_ticks([], [], te, E, 1.0, None, 120)
        chk(abs(we["v"]["LIVE"]["pct"] + 0.09) < 1e-12, "no print → flat −0.09 (fees only)")
        # 1m walker
        te1 = te + 9_000                                       # entry 9 s into a minute
        o = (te // MIN) * MIN + np.arange(0, 125) * MIN
        k = pd.DataFrame(dict(open_time=o, h=99.8, l=99.0, c=99.5))
        k.loc[k.open_time == o[0], "h"] = 200.0                # the entry minute's bar is ignored (prices before the fill)
        k.loc[k.open_time == o[17], "h"] = tp_px               # bar closing 18 min after o[0]
        k.loc[k.open_time == o[14], "c"] = 98.0                # bar closing at o[0]+15 min ≤ te1+15 min
        w1 = walk_1m(k, te1, E, 1.0, None, 120)
        chk(w1["v"]["CAP15"]["why"] == "CAP" and abs(w1["v"]["CAP15"]["pct"] - float(net_pct(98.0, E))) < 1e-12
            and w1["v"]["CAP20"]["why"] == "TP" and w1["v"]["CAP20"]["pct"] == 1.0 and w1["v"]["LIVE"]["pct"] == 1.0
            and abs(w1["tp_min"] - (o[17] + MIN - te1) / MIN) < 1e-9, f"1m walker: entry bar skipped, cap close, TP booked at +1 ({w1['v']})")
        k2 = k.copy()
        k2["h"] = 99.8
        w2 = walk_1m(k2, te1, E, 1.0, None, 120)
        chk(w2["v"]["LIVE"]["why"] == "CAP" and abs(w2["v"]["LIVE"]["pct"] - float(net_pct(99.5, E))) < 1e-12, "1m: no TP → close of the last bar ≤ 120")
        # walker parity vs the actual exit
        wt = walk_ticks(te + np.array([MIN]), [101.15], te, E, 1.0, None, 120)
        chk(parity("FRENZY_TP", float(net_pct(101.15, E)), wt)[0] and not parity("MAX_HOLD_TIME", -1.0, wt)[0]
            and not parity("FRENZY_TP", float(net_pct(101.15, E)) + 0.06, wt)[0] and not parity("FRENZY_TP_LATE", -0.5, wt)[0],
            "parity: reason + % within 0.05 (a TP_LATE is never a match)")
        # verdict machinery
        def fills(dl, tp15, days=None, n=None):
            n = n or len(dl)
            return pd.DataFrame(dict(ts=[pd.Timestamp("2026-10-10") + pd.Timedelta(hours=7 * i) for i in range(n)],
                                     pair=[f"P{i}" for i in range(n)], day=days if days is not None else [f"D{i % 10}" for i in range(n)],
                                     LIVE=0.0, CAP15=dl, CAP20=dl, why_CAP15=tp15, why_CAP20=tp15, par_ok=True))
        n = 24
        tp15 = np.array(["TP"] * 12 + ["CAP"] * 12)
        good = np.where(tp15 == "TP", 0.0, 1.0)
        chk(verdict(fills(good, tp15).iloc[:19])[0] == "COLLECTING" and verdict(fills(good, tp15, days=["D1"] * n))[0] == "COLLECTING",
            "N < 20 or < 8 days → collecting")
        v = verdict(fills(good, tp15))
        chk(v[0] == "CAP15 CANDIDATE (operator decides)", f"positive Δ on the no-TP fills, both caps equal → CAP15 (tie) ({v[0]})")
        f20 = fills(good, tp15).assign(CAP20=np.where(tp15 == "TP", 0.0, 2.0))
        chk(verdict(f20)[0] == "CAP20 CANDIDATE (operator decides)", "both qualify → the larger mean Δ")
        chk(verdict(fills(-good, tp15))[0] == "KEEP 120", "both Δ ≤ 0 → KEEP 120")
        one = good.copy()
        one[-1] = 100.0
        chk(verdict(fills(one, tp15))[0] == "KEEP OBSERVING", "one fill > 50 % of Σ Δ → not a candidate")
        mixed = np.concatenate([np.zeros(12), np.tile([1.0, -0.8], 6)])
        vm = verdict(fills(mixed, tp15))
        chk(vm[0] == "KEEP OBSERVING", f"weak / noisy Δ (P < 0.90) → keep observing ({vm[1]})")
        bad = good.copy()
        bad[0] = 0.3
        chk(verdict(fills(bad, tp15))[0] == "Δ=0 SANITY FAILED", "a TP-in-cap fill with Δ ≠ 0 → sanity flagged")
        d = np.concatenate([np.zeros(12), np.tile([1.0, -0.2, 0.5], 4)])
        days = [f"D{i % 10}" for i in range(n)]
        chk(p_delta_pos(d, days) == p_delta_pos(d, days) == NF.p_mean_neg(-d, np.asarray(days), 4000, 7), "bootstrap deterministic (seed 7)")
        # freeze / deferral / reread
        F = fills(good, tp15)
        why = []
        st, ch = freeze({}, F, "t0", (1.0, None, 120), [(F.ts.iloc[3], "OPEN Q")], why)
        chk(not ch and st == {} and "OPEN Q" in why[0], "an OPEN WILLY opened ≤ the prefix end defers the freeze")
        st, ch = freeze({}, F, "t1", (1.0, None, 120), [(pd.Timestamp("2030-01-01"), "OPEN Z")])
        chk(ch and st["first"]["n"] == 20 and st["first"]["state"].startswith("CAP15") and len(st["first"]["keys"]) == 20,
            f"frozen on the first 20-fill prefix ({st['first']['state']})")
        snap = json.loads(json.dumps(st))
        st2, ch2 = freeze(json.loads(json.dumps(st)), F.assign(CAP15=-F.CAP15), "t2", (1.0, None, 120))
        chk(not ch2 and st2 == snap, "first frozen verdict never recomputed; no re-read below N 40")
        F40 = fills(np.tile(good, 2), np.tile(tp15, 2))
        st3, ch3 = freeze(json.loads(json.dumps(st)), F40, "t3", (1.0, None, 120))
        chk(ch3 and st3["first"] == snap["first"] and st3["reread"]["n"] == 40, "one re-read at N ≥ 40 on a later call")
        stw, chw = freeze({}, fills(bad, tp15), "tw", (1.0, None, 120), [], [])
        chk(not chw and stw == {}, "Δ=0 SANITY FAILED is never frozen")
        # walker-parity gate: all match → ok; ≥ 90 % with every no-TP fill matching → ok; a no-TP fill off → never frozen
        Fp = fills(good, tp15)
        chk(parity_gate(Fp)[0] and parity_gate(Fp.assign(par_ok=np.arange(n) != 0))[0], "parity: all / ≥ 90 % with the no-TP fills matching")
        why = []
        stp, chp = freeze({}, Fp.assign(par_ok=np.arange(n) != 15), "tp", (1.0, None, 120), [], why)
        chk(not chp and stp == {} and "WALKER PARITY FAILED" in why[0], "a no-TP-in-cap fill off parity → WALKER PARITY FAILED, not frozen")
        chk(not freeze({}, Fp.assign(par_ok=np.arange(n) >= 3), "tq", (1.0, None, 120))[1], "< 90 % parity → not frozen")
        std, chd = freeze({}, Fp, "td", (1.0, None, 120), [], [], dropped=[Fp.ts.iloc[2], pd.Timestamp("2030-01-01")])
        chk(chd and std["first"]["dropped_no_data"] == 1 and "20/20" in std["first"]["parity"], "final-no-data fills ≤ the prefix end recorded")
        # ── S4 cells (DECISION_LOG 278): CAP60 / CAP90 on the −4 stop, hybrid walker ──
        chk((S4_CELLS, S4_KEY, S4_PARITY_MIN, S4_LAG_MIN, S4_DEPLOY_GREP)
            == ((("CAP60", 1.0, 60), ("CAP90", 1.0, 90), ("T125C60", 1.25, 60), ("T125C90", 1.25, 90)),
                {"CAP60": "S4_CAP60", "CAP90": "S4_CAP90", "T125C60": "S4_T125C60", "T125C90": "S4_T125C90"}, 0.90, 10, "(DECISION_LOG 281)")
            and (OLD_EXIT, OLD_KEY, OLD_LABEL, S4_DEPLOY_PIN) == ((1.0, 4.0, 120), "OLD_EXIT", "pre-registered 2026-10-12, DECISION_LOG 281",
                                                                "2026-10-12 12:00:00")
            and OLD_BAR.startswith("FROZEN REVERT BAR (pre-registered 2026-10-12, DECISION_LOG 281).")
            and "Counted from the DECISION_LOG 281 deploy (today = +1.25 / −4 / 90)" in S4_BAR and "DECISION_LOG 281 deploy" in S4_BAR_T125
            and S4_BAR.startswith("FROZEN BAR (pre-registered 2026-10-12, DECISION_LOG 278).") and "KEEP 120" in S4_BAR
            and S4_BAR_T125.startswith("FROZEN BAR (pre-registered 2026-10-12, DECISION_LOG 278).") and "keep today" in S4_BAR_T125
            and s4_spec("CAP90", 1.25) == (1.0, 90) and s4_spec("T125C60", 1.0) == (1.25, 60),
            "S4 constants + bar pinned")
        g281 = _git_ct("(DECISION_LOG 281)")
        chk(s4_deploy_ts() == (pd.Timestamp("2026-10-12 12:00:00") if g281 is None else g281) + pd.Timedelta(minutes=10)
            and dl281_floor is s4_deploy_ts, f"S4 / OLD_EXIT floor = the DL 281 commit (or the placeholder pin) + 10 min ({s4_deploy_ts()})")
        S4_DEPLOY_GREP = "(no such commit — selftest)"
        chk(s4_deploy_ts() == pd.Timestamp("2026-10-12 12:10:00"), "git cannot find the DL 281 commit → the placeholder pin + 10 min")
        S4_DEPLOY_GREP = saved[-2]
        if os.path.isdir(os.path.join(_ROOT, ".git")):                  # a checkout without git history (staged-tree test copy) skips this one
            chk(_git_ct("(DECISION_LOG 271)") == pd.Timestamp("2026-10-12 01:49:51"), "the git %ct lookup (DL 271 = 99b9048)")
        # n/a: identical to today / cap beyond the hold; OLD_EXIT n/a only while today IS the old exit
        chk(s4_na("T125C90", 1.25, 90) == "= today" and s4_na("CAP90", 1.25, 90) == "" and s4_na("CAP60", 1.25, 90) == ""
            and s4_na("T125C60", 1.25, 90) == "" and s4_na("CAP90", 1.0, 75) == "cap > hold" and s4_na("CAP90", 1.0, 90) == "= today"
            and s4_na("CAP90", 1.0, 120) == "" and s4_na(OLD_KEY, 1.0, 120, 4.0) == "= today" and s4_na(OLD_KEY, 1.25, 90, 4.0) == ""
            and s4_na(OLD_KEY, 1.0, 120, 3.0) == "" and s4_spec(OLD_KEY, 1.25) == (1.0, 120),
            "n/a: a cell identical to today (T125C90 under +1.25 / 90) or with cap > hold; OLD_EXIT n/a only while today = the old exit")
        chk(levels()[1] == float(rt["frenzy_willy_stop_pct"]) == 4.0, f"today's ruler reads the −4 stop from trading_config.json ({levels()})")

        def _P(tt, pp):                                           # a ticks-only hybrid path (load_path's shape)
            tt = np.asarray(tt, dtype="int64")
            nn = net_pct(pp, E) if len(tt) else np.array([], dtype=float)
            lim = te + 121 * MIN
            return dict(t=tt, hi=nn, lo=nn, cl=nn, op=np.full(len(tt), np.nan), bar=np.zeros(len(tt), bool), end_ms=lim,
                        n_tick=len(tt), n_bar=0, tick_end=lim, lim=lim, te=te)
        nx = lambda px: float(net_pct(px, E))                     # noqa: E731
        c = s4_cells(_P(te + np.array([10 * MIN, 60 * MIN, 60 * MIN + 1, 90 * MIN, 90 * MIN + 1, 119 * MIN]),
                        [99.5, 99.0, 98.0, 97.5, 97.0, 96.5]), 1.0, 4.0, 120)
        chk(c["s4_why_CAP60"] == c["s4_why_CAP90"] == c["s4_why_LIVE"] == "CAP" and abs(c["s4_CAP60"] - nx(99.0)) < 1e-12
            and abs(c["s4_CAP90"] - nx(97.5)) < 1e-12 and abs(c["s4_LIVE"] - nx(96.5)) < 1e-12 and c["s4_min_CAP60"] == 60.0
            and c["s4_min_CAP90"] == 90.0 and c["s4_n_unres"] == 0 and c["_on_ticks"],
            f"S4: the cap fires at exactly 60 / 90 min (a print AT the cap counts, one 1 ms later does not) ({c})")
        c = s4_cells(_P(te + np.array([20 * MIN, 30 * MIN, 100 * MIN]), [99.0, 95.8, 105.0]), 1.0, 4.0, 120)
        chk(all(c[f"s4_why_{v}"] == "STOP" and abs(c[f"s4_{v}"] - nx(95.8)) < 1e-12 for v in ("LIVE", "CAP60", "CAP90", "T125C60", "T125C90"))
            and abs(c["s4_min_LIVE"] - 30) < 1e-9, "S4: −4 before the cap → every cell stops at the same print (Δ = 0)")
        c = s4_cells(_P(te + np.array([10 * MIN, 40 * MIN, 100 * MIN]), [99.0, tp_px, 90.0]), 1.0, 4.0, 120)
        chk(all(c[f"s4_why_{v}"] == "TP" and abs(c[f"s4_{v}"] - nx(tp_px)) < 1e-12 for v in ("LIVE", "CAP60", "CAP90")),
            "S4: TP before the cap → every cell takes the same TP (Δ = 0)")
        chk(all(c[f"s4_why_{v}"] == "CAP" and abs(c[f"s4_{v}"] - nx(tp_px)) < 1e-12 for v in ("T125C60", "T125C90")),
            "S4: +1.11 net is a TP for today, not for +1.25 (T125 cells close at their cap on the last print ≤ it; the −4 at 100 min is after)")
        c = s4_cells(_P(te + np.array([10 * MIN, 40 * MIN, 100 * MIN]), [99.0, 101.5, 90.0]), 1.0, 4.0, 120)
        chk(all(c[f"s4_why_{v}"] == "TP" and abs(c[f"s4_{v}"] - nx(101.5)) < 1e-12 for v in ("LIVE", "T125C60", "T125C90")),
            "S4: a print through +1.25 → today and the T125 cells take the same print (Δ = 0)")
        c = s4_cells(_P(te + np.array([10 * MIN, 30 * MIN, 50 * MIN, 61 * MIN, 95 * MIN]), [99.0, 101.15, 101.18, 98.8, 98.6]), 1.0, 4.0, 120)
        chk(c["s4_why_LIVE"] == "TP" and c["s4_why_T125C60"] == "CAP" and abs(c["s4_T125C60"] - nx(101.18)) < 1e-12
            and c["s4_why_T125C90"] == "CAP" and abs(c["s4_T125C90"] - nx(98.8)) < 1e-12 and c["s4_T125C90"] < c["s4_LIVE"],
            "S4: a ZK-like path (+1.0 reached, +1.25 never) → T125 cells lose today's +1 and close at their cap")
        c = s4_cells(_P(te + np.array([10 * MIN, 55 * MIN, 75 * MIN]), [99.0, 98.5, tp_px]), 1.0, 4.0, 120)
        chk(c["s4_why_CAP60"] == "CAP" and abs(c["s4_CAP60"] - nx(98.5)) < 1e-12 and c["s4_why_CAP90"] == c["s4_why_LIVE"] == "TP"
            and c["s4_CAP90"] == c["s4_LIVE"], "S4: TP at 75 min → CAP60 closes at 60, CAP90 = today")
        c = s4_cells(_P(te + np.array([59 * MIN, 89 * MIN, 100 * MIN]), [99.8, 99.9, 95.8]), 1.0, 4.0, 120)
        chk(c["s4_why_LIVE"] == "STOP" and c["s4_why_CAP60"] == c["s4_why_CAP90"] == "CAP" and c["s4_CAP60"] - c["s4_LIVE"] > 3,
            "S4: a stop after both caps → the caps close earlier (Δ > 0)")
        c = s4_cells(_P(te + np.array([59 * MIN]), [99.8]), 1.0, 4.0, 60)
        chk(all(c[f"s4_{v}"] != c[f"s4_{v}"] and c[f"s4_why_{v}"] == "N/A" for v in ("CAP60", "CAP90", "T125C90"))
            and c["s4_why_T125C60"] == "CAP" and c["s4_n_unres"] == 0 and s4_na("CAP60", 1.0, 60) == "= today"
            and s4_na("CAP90", 1.0, 60) == "cap > hold" and not s4_na("CAP60", 1.0, 61) and not s4_na("T125C90", 1.0, 90),
            "S4: cap > today's hold or = today → n/a (never folded into today with min()), never a pending undecided cell; "
            "cap = hold with another TP is judged")
        Pu = _P(te + np.array([10 * MIN]), [99.0])
        Pu.update(end_ms=te + 70 * MIN, tick_end=te + 70 * MIN)
        cu = s4_cells(Pu, 1.0, 4.0, 120)
        chk(cu is None, "S4: today undecided at the path end → unpriced")
        Pu.update(end_ms=te + 121 * MIN)
        chk(s4_cells(Pu, 1.0, 4.0, 120)["s4_n_unres"] == 0, "S4: a full path decides every cell")

        def s4f(d, wl, ml, days=None, par=True, n=None):
            n = n or len(d)
            lv = np.where(np.asarray(wl) == "TP", 1.0, 0.0)          # today's TP booked at +1.00 (below +1.25); cells = today + d
            d = lv + np.asarray(d, dtype=float)
            return pd.DataFrame(dict(ts=[pd.Timestamp("2026-10-13") + pd.Timedelta(hours=7 * i) for i in range(n)],
                                     pair=[f"P{i}" for i in range(n)], day=days if days is not None else [f"D{i % 10}" for i in range(n)],
                                     s4_src="ticks", s4_LIVE=lv, s4_peak_LIVE=-np.inf, s4_CAP60=d, s4_CAP90=d, s4_T125C60=d, s4_T125C90=d,
                                     s4_OLD_EXIT=d, s4_why_LIVE=wl,
                                     s4_min_LIVE=ml, s4_par_ok=par))
        wl = np.array(["TP"] * 12 + ["CAP"] * 12)
        ml = np.where(wl == "TP", 5.0, 120.0)
        g4 = np.where(wl == "TP", 0.0, 1.0)
        chk(s4_verdict(s4f(g4, wl, ml).iloc[:19], "CAP60", 1.0)[0] == "COLLECTING" and s4_verdict(s4f(g4, wl, ml, days=["D1"] * 24), "CAP60", 1.0)[0] == "COLLECTING",
            "S4: N < 20 or < 8 days → collecting")
        v4 = s4_verdict(s4f(g4, wl, ml), "CAP60", 1.0)
        chk(v4[0] == "CAP60 CANDIDATE (operator decides)" and v4[2]["integ"] and v4[2]["n_untouched"] == 12, f"S4: candidate ({v4[1]})")
        chk(s4_verdict(s4f(-g4, wl, ml), "CAP90", 1.0)[0] == "KEEP TODAY", "S4: mean Δ ≤ 0 → KEEP TODAY (the bar text's KEEP 120 = keep today's exit)")
        chk(s4_verdict(s4f(np.concatenate([np.zeros(12), np.tile([1.0, -0.8], 6)]), wl, ml), "CAP60", 1.0)[0] == "KEEP OBSERVING",
            "S4: P(Δ>0) < 0.90 → keep observing")
        o4 = g4.copy()
        o4[-1] = 100.0
        chk(s4_verdict(s4f(o4, wl, ml), "CAP60", 1.0)[0] == "KEEP OBSERVING", "S4: one fill > 50 % of Σ Δ → keep observing")
        b4 = g4.copy()
        b4[0] = 0.3
        vb = s4_verdict(s4f(b4, wl, ml), "CAP60", 1.0)
        chk(vb[0] == "KEEP OBSERVING" and not vb[2]["integ"], "S4: a cap-untouched fill with Δ ≠ 0 (walker integrity) → never a candidate")
        chk(s4_verdict(s4f(g4, wl, np.where(wl == "TP", 75.0, 120.0)), "CAP60", 1.0)[2]["n_untouched"] == 0
            and s4_verdict(s4f(g4, wl, np.where(wl == "TP", 75.0, 120.0)), "CAP90", 1.0)[2]["n_untouched"] == 12,
            "S4: a TP at 75 min is cap-touched for CAP60, untouched for CAP90")
        chk(s4_verdict(s4f(g4, wl, ml, par=np.arange(24) >= 3), "CAP60", 1.0)[0] == "KEEP OBSERVING", "S4: walker parity < 90 % → never a candidate")
        chk(s4_verdict(s4f(g4, wl, ml, par=np.arange(24) >= 2), "CAP60", 1.0)[0].startswith("CAP60 CANDIDATE"), "S4: parity 22/24 ≥ 90 % passes")
        chk(s4_verdict(s4f(-g4, wl, ml), "T125C90", 1.0)[0] == "KEEP TODAY" and s4_verdict(s4f(g4, wl, ml), "T125C60", 1.0)[2]["n_untouched"] == 0,
            "S4: T125 mean Δ ≤ 0 → KEEP TODAY; a today-TP print below +1.25 net is touched for T125")
        ut = pd.DataFrame(dict(s4_why_LIVE=["TP", "TP", "STOP", "STOP", "CAP"], s4_min_LIVE=[30.0, 30.0, 30.0, 70.0, 120.0],
                               s4_LIVE=[1.30, 1.05, -4.1, -4.1, -1.0], s4_peak_LIVE=[0.5, 0.5, 0.9, 0.9, 0.9]))
        chk(list(s4_untouched(ut, "T125C60", 1.0)) == [True, False, True, False, False]
            and list(s4_untouched(ut, "CAP90", 1.0)) == [True, True, True, True, False],
            "S4 untouched: today's −4 ≤ the cap, or today's TP ≤ the cap on a print through the cell's TP")
        ut2 = pd.DataFrame(dict(s4_why_LIVE=["TP", "TP", "STOP", "STOP"], s4_min_LIVE=[30.0, 30.0, 30.0, 30.0],
                                s4_LIVE=[1.30, 1.30, -4.1, -4.1], s4_peak_LIVE=[0.8, 1.10, 0.8, 1.10]))
        _g = pd.DataFrame(dict(s4_min_LIVE=[20.0], s4_why_LIVE=["TP"], s4_LIVE=[1.25], s4_peak_LIVE=[0.5], s4_onbar_LIVE=[True]))
        chk(not s4_untouched(_g, OLD_KEY, 1.25)[0] and not s4_untouched(_g, "CAP60", 1.25)[0]
            and s4_untouched(_g.assign(s4_onbar_LIVE=[False]), "CAP60", 1.25)[0] and s4_untouched(_g, "T125C60", 1.25)[0],
            "deep review: a 1m-BAR TP (booked at the line) with the cell TP below today's is touched; a tick TP ≥ cell TP stays untouched")
        chk(list(s4_untouched(ut2, "CAP60", 1.25)) == [True, False, True, False] and list(s4_untouched(ut2, OLD_KEY, 1.25)) == [True, False, True, False],
            "untouched with a cell TP BELOW today's (+1 vs +1.25): an earlier print through +1 makes the fill touched")
        F4 = s4f(g4, wl, ml)
        sentinel = {"first": {"state": "KEEP OBSERVING", "n": 20}}
        why = []
        s4s, ch = s4_freeze(json.loads(json.dumps(sentinel)), "CAP60", F4, "t0", (1.0, 4.0, 120), [(F4.ts.iloc[4], "1m-provisional Q")], why)
        chk(not ch and s4s == sentinel and "1m-provisional Q" in why[0] and why[0].startswith("CAP60"),
            "S4: a 1m-provisional fill in the prefix holds the freeze")
        s4s, ch = s4_freeze(json.loads(json.dumps(sentinel)), "CAP60", F4, "t1", (1.0, 4.0, 120), [(F4.ts.iloc[22], "unpriced late")])
        chk(ch and s4s["first"] == sentinel["first"] and s4s["S4_CAP60"]["state"].startswith("CAP60 CANDIDATE") and s4s["S4_CAP60"]["n"] == 20
            and s4s["S4_CAP60"]["bar"] == S4_LABEL and set(s4s) == {"first", "S4_CAP60"}, "S4: frozen once on the first prefix, CAP15 / CAP20 state untouched")
        snap4 = json.loads(json.dumps(s4s))
        s4s, ch = s4_freeze(s4s, "CAP60", F4.assign(s4_CAP60=F4.s4_LIVE - 1.0), "t2", (1.0, 4.0, 120))
        chk(not ch and s4s == snap4, "S4: a frozen cell is never recomputed")
        so, cho = freeze({"S4_CAP60": snap4["S4_CAP60"]}, fills(good, tp15), "t3", (1.0, None, 120), [])
        chk(cho and so["S4_CAP60"] == snap4["S4_CAP60"] and so["first"]["state"].startswith("CAP15"),
            "S4: the CAP15 / CAP20 freeze never touches the S4 keys")
        why = []
        sp, chp = s4_freeze({}, "CAP60", s4f(g4, wl, ml, par=np.arange(24) >= 3), "tp", (1.0, 4.0, 120), [], why)
        chk(not chp and sp == {} and "WALKER PARITY / INTEGRITY FAILED — not frozen" in why[0] and "parity 85 %" in why[0],
            f"S4 freeze: parity < 90 % on the prefix → not frozen ({why})")
        nb = -g4
        nb[0] = 0.3                                              # KEEP-side verdict (mean ≤ 0) with a broken untouched fill
        why = []
        sk, chk_ = s4_freeze({}, "CAP90", s4f(nb, wl, ml), "tk", (1.0, 4.0, 120), [], why)
        chk(not chk_ and sk == {} and "INTEGRITY FAILED" in why[0] and "✗" in why[0], "S4 freeze: a KEEP verdict with integrity ✗ → not frozen")
        su, chu = s4_freeze({}, "T125C60", F4, "tu", (1.0, 4.0, 120), [], [], dropped=[F4.ts.iloc[1]],
                            undecided=[F4.ts.iloc[3], F4.ts.iloc[5], pd.Timestamp("2030-01-01")])
        chk(chu and su["S4_T125C60"]["undecided_final"] == 2 and su["S4_T125C60"]["dropped_no_data"] == 1,
            "S4 freeze: final fills left undecided by the cell are recorded (≤ the prefix end) next to dropped_no_data")
        # ── OLD_EXIT revert gate (DECISION_LOG 281): today = the NEW exit (+1.25 / −4 / 90), old = +1 / −4 / 120
        def ofr(dold, live, why_l, par=True, days=None):
            n = len(dold)
            lv = np.asarray(live, dtype=float)
            return pd.DataFrame(dict(ts=[pd.Timestamp("2026-10-13") + pd.Timedelta(hours=7 * i) for i in range(n)], pair=[f"O{i}" for i in range(n)],
                                     day=days if days is not None else [f"D{i % 10}" for i in range(n)], s4_src="ticks", s4_LIVE=lv,
                                     s4_peak_LIVE=0.5, s4_why_LIVE=why_l, s4_min_LIVE=np.where(np.asarray(why_l) == "CAP", 90.0, 10.0),
                                     s4_OLD_EXIT=lv + np.asarray(dold, dtype=float), s4_par_ok=par))
        wl_o = np.array(["TP"] * 12 + ["CAP"] * 12)
        lv_o = np.where(wl_o == "TP", 1.3, 0.2)                   # new avg > 0; TP prints ≥ +1 (untouched for the old +1 → Δ 0)
        d_up = np.where(wl_o == "TP", 0.0, 0.5)
        v = old_verdict(ofr(d_up, lv_o, wl_o).iloc[:19], 1.25)
        chk(v[0] == "COLLECTING", "revert gate: N < 20 → collecting")
        v = old_verdict(ofr(d_up, lv_o, wl_o), 1.25)
        chk(v[0] == OLD_REVERT and "old better" in v[1] and v[2]["integ"] and v[2]["n_untouched"] == 12, f"revert gate: old better, P ≥ 0.90 → REVERT ({v[1]})")
        v = old_verdict(ofr(-d_up, lv_o, wl_o), 1.25)
        chk(v[0] == OLD_KEEP, f"revert gate: old worse, new avg > 0 → KEEP NEW ({v[1]})")
        noisy = np.concatenate([np.zeros(12), np.tile([0.5, -0.45], 6)])
        v = old_verdict(ofr(noisy, lv_o, wl_o), 1.25)
        chk(v[0] == OLD_KEEP and v[2]["p"] < 0.90 and v[2]["mean"] > 0, f"revert gate: mean Δ > 0 but P < 0.90 → KEEP NEW ({v[1]})")
        top = d_up.copy()
        top[-1] = 50.0
        chk(old_verdict(ofr(top, lv_o, wl_o), 1.25)[0] == OLD_KEEP, "revert gate: one fill > 50 % of ΣΔ (and new avg > 0) → KEEP NEW")
        lv_neg = np.where(wl_o == "TP", 1.3, -2.0)
        v = old_verdict(ofr(-d_up, lv_neg, wl_o), 1.25)
        chk(v[0] == OLD_REVERT and "new avg < 0" in v[1] and "old better" not in v[1], f"revert gate: new avg < 0 → REVERT even with old worse ({v[1]})")
        why = []
        so_, cho_ = s4_freeze({}, OLD_KEY, ofr(d_up, lv_o, wl_o, par=np.arange(24) >= 3), "tp", (1.25, 4.0, 90), [], why)
        chk(not cho_ and so_ == {} and "OLD_EXIT — WALKER PARITY / INTEGRITY FAILED — not frozen" in why[0], "revert gate: parity < 90 % → not frozen")
        bi = d_up.copy()
        bi[0] = 0.3
        why = []
        chk(not s4_freeze({}, OLD_KEY, ofr(bi, lv_o, wl_o), "ti", (1.25, 4.0, 90), [], why)[1] and "✗" in why[0],
            "revert gate: integrity ✗ (an untouched fill with Δ ≠ 0) → not frozen")
        why = []
        chk(not s4_freeze({}, OLD_KEY, ofr(d_up, lv_o, wl_o), "th", (1.25, 4.0, 90), [(pd.Timestamp("2026-10-14"), "1m-provisional Q")], why)[1]
            and why[0].startswith("OLD_EXIT — "), "revert gate: a provisional fill in the prefix holds the freeze")
        so_, cho_ = s4_freeze({"first": {"x": 1}, "S4_CAP60": {"y": 2}}, OLD_KEY, ofr(d_up, lv_o, wl_o), "tf", (1.25, 4.0, 90), [])
        chk(cho_ and so_["OLD_EXIT"]["state"] == OLD_REVERT and so_["OLD_EXIT"]["bar"] == OLD_LABEL and so_["first"] == {"x": 1}
            and so_["S4_CAP60"] == {"y": 2} and so_["OLD_EXIT"]["config"] == dict(tp=1.25, stop=4.0, hold=90, cell_tp=1.0, cap=120)
            and not s4_freeze(so_, OLD_KEY, ofr(-d_up, lv_o, wl_o), "tg", (1.25, 4.0, 90), [])[1],
            "revert gate: frozen once under its own key (other keys untouched), never recomputed")
        # the hybrid walk: today +1.25 / −4 / 90 vs old +1 / −4 / 120 on one path; peak-before-exit feeds the untouched leg
        c = s4_cells(_P(te + np.array([5 * MIN, 80 * MIN, 89 * MIN, 100 * MIN]), [101.15, 99.0, 98.5, 97.0]), 1.25, 4.0, 90)
        chk(c["s4_why_LIVE"] == "CAP" and abs(c["s4_LIVE"] - nx(98.5)) < 1e-12 and c["s4_why_OLD_EXIT"] == "TP"
            and abs(c["s4_OLD_EXIT"] - nx(101.15)) < 1e-12 and c["s4_why_T125C90"] == "N/A" and c["s4_why_CAP90"] == "TP"
            and c["s4_why_T125C60"] == "CAP" and abs(c["s4_T125C60"] - nx(101.15)) < 1e-12,
            f"walk: +1.06 net is the old TP, not the new; new closes at 90; T125C90 n/a (= today) ({c})")
        c = s4_cells(_P(te + np.array([5 * MIN, 20 * MIN, 30 * MIN]), [100.8, 101.15, 101.5]), 1.25, 4.0, 90)
        chk(c["s4_why_LIVE"] == "TP" and abs(c["s4_peak_LIVE"] - nx(101.15)) < 1e-12 and c["s4_why_OLD_EXIT"] == "TP"
            and c["s4_min_OLD_EXIT"] == 20.0 and c["s4_OLD_EXIT"] < c["s4_LIVE"], "walk: an earlier +1 print → the old exit takes it first (touched)")
        c = s4_cells(_P(te + np.array([5 * MIN, 20 * MIN]), [99.0, 101.5]), 1.25, 4.0, 90)
        chk(c["s4_OLD_EXIT"] == c["s4_LIVE"] and list(s4_untouched(pd.DataFrame([c]), OLD_KEY, 1.25)) == [True],
            "walk: a print straight through +1.25 → old = new, untouched")
        # caches + fetch path in a temp tree
        S4_DEPLOY_GREP, S4_DEPLOY_PIN = "(no such commit — selftest)", "2026-10-12 01:49:51"   # fixtures' floor: 01:59:51 UTC
        with tempfile.TemporaryDirectory() as td:
            _CACHE = os.path.join(td, "cache")
            MY_CACHE = os.path.join(_CACHE, "scout_willy_timecap")
            STATE = os.path.join(td, "SCOUT_WILLY_TIMECAP.json")
            STUDY_CSV = os.path.join(td, "absent.csv")
            NF._CACHE = _CACHE
            NF.SHARED_COOLDOWN = os.path.join(_CACHE, ".binance_cooldown_until")
            NF.DU_COOLDOWN = os.path.join(_CACHE, "scout_dailyup", ".ratelimited_until")
            dl = os.path.join(td, "dl")
            os.makedirs(dl)
            EXPORT_GLOB = os.path.join(dl, "scalpars_orders_paper_*.csv")
            base = dict(direction="LONG", entry_strategy=STRATEGY, status="CLOSED", notional_value=10_000.0, entry_frenzy_willy_trigger="A")
            rows = [dict(base, opened_at="2026-10-09T08:45:09", pair="KAIAUSDT", entry_price=100.0, pnl_percentage=float(net_pct(tp_px, 100.0)),
                         pnl=1.0, close_reason="FRENZY_TP", closed_at="2026-10-09T08:50:00"),
                    dict(base, opened_at="2026-10-09T09:05:08", pair="WUSDT", entry_price=100.0, pnl_percentage=float(net_pct(97.0, 100.0)),
                         pnl=-1.0, close_reason="MAX_HOLD_TIME", closed_at="2026-10-09T11:05:10", entry_frenzy_willy_trigger="B"),
                    dict(base, opened_at="2026-10-09T12:00:00", pair="NOKUSDT", entry_price=1.0, pnl_percentage=1.0, pnl=1.0,
                         close_reason="FRENZY_TP", closed_at="2026-10-09T12:01:00"),
                    dict(base, opened_at="2026-10-07T12:00:00", pair="OLDUSDT", entry_price=1.0, pnl_percentage=1.0, pnl=1.0,
                         close_reason="FRENZY_TP", closed_at="2026-10-07T12:01:00"),
                    dict(base, opened_at="2026-10-09T10:00:00", pair="LITEUSDT", entry_strategy="FRENZY_LITE", entry_price=1.0,
                         pnl_percentage=1.0, pnl=1.0, close_reason="FRENZY_TP", closed_at="x"),
                    dict(base, opened_at="2026-10-09T13:00:00", pair="OPNUSDT", status="OPEN", entry_price=1.0, pnl_percentage=np.nan,
                         pnl=np.nan, close_reason="", closed_at="")]
            old = pd.DataFrame([dict(rows[0], status="OPEN", pnl_percentage=0.0)])
            fo, fn = os.path.join(dl, "scalpars_orders_paper_a.csv"), os.path.join(dl, "scalpars_orders_paper_b.csv")
            pd.DataFrame(rows).to_csv(fo, index=False)
            old.to_csv(fn, index=False)                         # NEWER export shows KAIA still OPEN → the CLOSED row must win
            os.utime(fo, (1, 1))
            o, opens = load_orders()
            chk(sorted(o.pair) == ["KAIAUSDT", "NOKUSDT", "OLDUSDT", "WUSDT"] and [x[1] for x in opens] == ["OPEN OPNUSDT"]
                and float(o[o.pair == "KAIAUSDT"].pct.iloc[0]) > 1.0, f"loader: CLOSED beats OPEN, WILLY only, OPEN listed ({sorted(o.pair)}, {opens})")
            # ticks for KAIA (cached archive day), 1m for W, nothing for NOK
            tk = pd.Timestamp("2026-10-09 08:45:09").value // 10**6
            os.makedirs(os.path.join(_CACHE, "ticks_q", "KAIAUSDT"))
            np.savez_compressed(os.path.join(_CACHE, "ticks_q", "KAIAUSDT", "2026-10-09.npz"),
                                t=np.array([tk - 5000, tk, tk + 70_000, tk + 75_000], dtype=np.int64), p=np.array([200.0, 100.0, 99.0, tp_px], dtype=np.float32))
            tw = pd.Timestamp("2026-10-09 09:05:08").value // 10**6
            ow = (tw // MIN) * MIN + np.arange(0, 125) * MIN
            os.makedirs(os.path.join(_CACHE, "k1m_ondemand"))
            kw = pd.DataFrame(dict(open_time=ow, o=99.0, h=99.5, l=96.5, c=97.0, vol=1, qvol=1))
            kw.loc[14, "c"] = 98.0
            kw.loc[19, "c"] = 98.5
            kw.to_csv(os.path.join(_CACHE, "k1m_ondemand", "WUSDT.csv"), index=False)
            now = pd.Timestamp("2026-10-09 16:00").value // 10**6
            out = "\n".join(run(now, fetch=False, state_path=STATE, th={}, study=True))
            chk("| **CAP15** | 2 |" in out and "KAIAUSDT A [ticks] TP@1.2 min" in out and "WUSDT B [1m, provisional]" in out
                and "UNSCORED (no path yet): 10-09 12:00 NOKUSDT" in out and "OLDUSDT" not in out and "LITEUSDT" not in out
                and "Walker parity" in out and "2/2 match" in out and "COLLECTING" in out and "file absent" in out
                and "at 15 min -2.09 %" in out and not os.path.exists(STATE),
                f"end-to-end run(): ticks KAIA, 1m W (provisional), NOK unscored, pre-ship / LITE out, parity 2/2\n{out}")
            # fetch: NOK → archive not yet due (day 10-09, now 10-09 16:00) → one 1m page; 429 arms the SHARED cooldown
            calls = []

            def _boom(url, timeout):
                calls.append(url)
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "120"}, None)
            _NET_BLOCKED, _http_get, _open_url = False, _boom, _boom
            tp_, st_, hd_ = levels({})
            o["src"] = [score_fill(r, hd_, tp_, st_)[1] for r in o.itertuples()]
            o = o[o.ts >= deploy_ts()].reset_index(drop=True)
            notes = ensure_data(o, hd_, now)
            chk(len(calls) == 1 and "NOKUSDT" in calls[0] and "interval=1m" in calls[0] and "limit=123" in calls[0] and any("429" in x for x in notes)
                and abs(NF._cooldown_until() - (now + 120_000)) < 2, f"429 → one request + SHARED cooldown ({calls}, {notes})")
            chk(len(calls) == 1 and any("cooldown" in x for x in ensure_data(o, hd_, now + 1000)), "cooldown live → zero requests")
            os.remove(NF.SHARED_COOLDOWN)

            def _w(url, timeout):
                calls.append(url)
                return [], {"X-MBX-USED-WEIGHT-1M": "950"}
            _http_get = _w
            many = pd.concat([o[o.pair == "NOKUSDT"].assign(pair=f"X{i}USDT") for i in range(4)], ignore_index=True)
            notes = ensure_data(many, hd_, now)
            chk(len(calls) == 2 and any("used weight" in x for x in notes), f"used weight ≥ 900 → no further request ({notes})")
            calls.clear()
            _http_get = lambda url, timeout: (calls.append(url), ([], {"X-MBX-USED-WEIGHT-1M": "5"}))[1]
            notes = ensure_data(many, hd_, now)
            chk(len(calls) == MAX_REQ and any("request cap" in x for x in notes), f"request cap {MAX_REQ}/run")
            calls.clear()
            e0 = many.iloc[:1].assign(pair="E0USDT")
            for _ in range(FAIL_MAX + 1):
                ensure_data(e0, hd_, now)
            chk(len(calls) == FAIL_MAX and load_attempts()["done"][_kk("E0USDT", e0.te.iloc[0])] == f"empty×{FAIL_MAX}", "3 empty pages → final")

            def _q(url, timeout):
                calls.append(url)
                if "Q0USDT" in url:
                    raise urllib.error.HTTPError(url, 400, "Invalid symbol", {}, None)
                q = dict(x.split("=") for x in url.split("?")[1].split("&"))
                s = int(q["startTime"])
                return [[s + i * MIN, "1", "1.02", "0.99", "1.0"] for i in range(int(q["limit"]))], {"X-MBX-USED-WEIGHT-1M": "3"}
            _http_get = _q
            calls.clear()
            qq = pd.concat([many.iloc[:1].assign(pair="Q0USDT"), many.iloc[:1].assign(pair="Q1USDT", ts=many.ts.iloc[0] + pd.Timedelta(minutes=1))],
                           ignore_index=True)
            ensure_data(qq, hd_, now, max_req=1)
            chk(len(calls) == 2 and "Q1USDT" in calls[1] and load_attempts()["done"][_kk("Q0USDT", qq.te.iloc[0])] == "HTTP 400"
                and score_fill(next(qq.iloc[1:].itertuples()), hd_, tp_, st_)[1] == "1m",
                "a 4xx is final at once (slot refunded) and the next fill is fetched + walked on 1m")
            ensure_data(qq, hd_, now)
            chk(len(calls) == 2, "final / covered pages are never re-requested")
            # archive path: the day is due 24 h after it ended → one zip, cached, the fill becomes a tick fill
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("NOKUSDT-aggTrades-2026-10-09.csv", "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
                           + "\n".join(f"{i},{px},1,1,1,{tm},false" for i, (px, tm) in
                                       enumerate([(1.0, pd.Timestamp('2026-10-09 12:00:30').value // 10**6),
                                                  (1.0111, pd.Timestamp('2026-10-09 12:00:40').value // 10**6)])))
            _open_url = lambda url, timeout: (calls.append(url), io.BytesIO(buf.getvalue()))[1]
            calls.clear()
            later = pd.Timestamp("2026-10-11 00:00").value // 10**6
            nok = o[o.pair == "NOKUSDT"].assign(src="1m")
            ensure_data(nok, hd_, later)
            chk(len(calls) == 1 and "data.binance.vision" in calls[0] and "NOKUSDT-aggTrades-2026-10-09.zip" in calls[0]
                and score_fill(next(nok.itertuples()), hd_, tp_, st_)[1] == "ticks", f"archive due → one zip → tick fill ({calls})")
            chk(not ensure_data(nok.assign(src="ticks"), hd_, later) and len(calls) == 1, "a tick fill never fetches")
            t_, p_, f_ = parse_archive(buf.getvalue())
            chk(len(t_) == 2 and abs(p_[1] - 1.0111) < 1e-12 and f_ and t_.dtype == np.int64, "archive parser: header sniffed, numeric dtypes")
            with np.load(os.path.join(_CACHE, "ticks", "NOKUSDT", "2026-10-09.npz")) as zz:
                chk(zz["t"].dtype == np.int64 and zz["p"].dtype == np.float32 and set(zz.files) == {"t", "p"},
                    "archive written to the SHARED ticks/<PAIR>/<day>.npz in backtest_fetch_ticks' format")
            # a too-big day: only this line's fill windows kept (own slice file), the shared cache untouched; chunked streaming
            big = io.BytesIO()
            t12 = pd.Timestamp("2026-10-09 12:00:00").value // 10**6
            rows_big = [(1.0, t12 - 3 * 3_600_000 + i * 1000) for i in range(30)] + [(1.0, t12 + 30_000), (1.0111, t12 + 40_000)]
            with zipfile.ZipFile(big, "w") as z:
                z.writestr("BIGUSDT-aggTrades-2026-10-09.csv", "\n".join(f"{i},{px},1,1,1,{tm},false" for i, (px, tm) in enumerate(rows_big)))
            t_, p_, f_ = parse_archive(big.getvalue(), [(t12 - MIN, t12 + 122 * MIN)], max_rows=10)
            chk(not f_ and len(t_) == 2, "past max_rows → only the window rows kept, full = False")
            saved_mx = ARCH_FULL_MAX_ROWS
            globals()["ARCH_FULL_MAX_ROWS"] = 10

            class _Slow(io.BytesIO):
                def read(self, n=-1):
                    calls.append("chunk")
                    return super().read(min(n, 64) if n and n > 0 else n)
            try:
                _open_url = lambda url, timeout: (calls.append(url), _Slow(big.getvalue()))[1]
                calls.clear()
                bg = nok.assign(pair="BIGUSDT")
                nb = ensure_data(bg, hd_, later)
            finally:
                globals()["ARCH_FULL_MAX_ROWS"] = saved_mx
            chk(_tick_file("BIGUSDT", "2026-10-09") is None and os.path.exists(_slice_file("BIGUSDT", "2026-10-09"))
                and score_fill(next(bg.itertuples()), hd_, tp_, st_)[1] == "ticks" and calls.count("chunk") > 2
                and any("shared tick cache not written" in x for x in nb) and not ticks_pending(next(bg.itertuples()), hd_, load_attempts()["done"], later),
                f"too-big day → slice cached for this line, fill walked on it, shared cache untouched ({nb})")
            chk(not glob.glob(os.path.join(MY_CACHE, "arch_*")), "temp archive files removed")

            # archive 404 = not published yet: retried ≥ 3 h later, final only after 3 × 404 AND ≥ 3 days after the day; ≤ 1 archive / run
            def _404(url, timeout):
                calls.append(url)
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            _open_url = _404
            calls.clear()
            a1 = nok.assign(pair="A1USDT")
            ka = _ak("A1USDT", "2026-10-09")
            ensure_data(a1, hd_, later)
            att = load_attempts()
            chk(len(calls) == 1 and ka not in att["done"] and att["fail"][ka] == 1 and att["last"][ka] == later, "a 404 is counted, not final")
            ensure_data(a1, hd_, later + ARCH_RETRY_MS - 1)
            chk(len(calls) == 1, "no retry within 3 h of a 404")
            ensure_data(a1, hd_, later + ARCH_RETRY_MS)
            ensure_data(a1, hd_, later + 2 * ARCH_RETRY_MS)
            att = load_attempts()
            chk(len(calls) == 3 and att["fail"][ka] == 3 and ka not in att["done"], "3 × 404 before the 3-day mark → still not final")
            gu = _day_end("2026-10-09") + ARCH_GIVEUP_MS
            ensure_data(a1, hd_, gu)
            chk(len(calls) == 3 and "404×3" in str(load_attempts()["done"].get(ka)), "≥ 3 × 404 and ≥ 3 days after the day → final, no request")
            calls.clear()
            two = pd.concat([nok.assign(pair="B1USDT"), nok.assign(pair="B2USDT")], ignore_index=True)
            ensure_data(two, hd_, later)
            chk(len(calls) == 1 and "B1USDT" in calls[0], f"≤ {ARCH_PER_RUN} archive per run ({calls})")

            class _Drip(io.RawIOBase):
                def read(self, n=-1):
                    time.sleep(0.2)
                    return b"x" * 16

            def _hang(url, timeout):
                calls.append(url)
                return _Drip()
            saved_to = ARCH_TIMEOUT_S
            globals()["ARCH_TIMEOUT_S"] = 0.5
            try:
                _open_url = _hang
                calls.clear()
                nt = ensure_data(nok.assign(pair="T1USDT"), hd_, later)
            finally:
                globals()["ARCH_TIMEOUT_S"] = saved_to
            chk(load_attempts()["fail"].get(_ak("T1USDT", "2026-10-09")) == 1 and any("timeout" in x for x in nt)
                and _tick_file("T1USDT", "2026-10-09") is None and not glob.glob(os.path.join(MY_CACHE, "arch_*")),
                "an archive stream past its deadline stops in the caller's thread: one failed attempt, temp file removed, nothing cached")
            # sub-second opened_at → te carries it; the dedup key / day stay whole-second
            ps = _prep(_dedup(pd.DataFrame([dict(base, opened_at="2026-10-09T09:05:08.750", pair="SUBUSDT", entry_price=1.0, pnl_percentage=1.0,
                                                  pnl=1.0, close_reason="FRENZY_TP", closed_at="x")])))
            chk(int(ps.te.iloc[0]) == pd.Timestamp("2026-10-09 09:05:08.750").value // 10**6 and f"{ps.ts.iloc[0]}" == "2026-10-09 09:05:08",
                "sub-second opened_at used for the walk")
            # per-run 1m cache: reused, dropped on append
            k_a = load_1m("Q1USDT")
            chk(load_1m("Q1USDT") is k_a, "1m bars cached per pair within a run")
            _append_1m("Q1USDT", k_a.iloc[:1])
            chk(load_1m("Q1USDT") is not k_a, "an append drops the pair from the run cache")

            def _slow(url, timeout):
                calls.append(url)
                time.sleep(3)
            _http_get = _slow
            t0 = time.monotonic()
            notes = ensure_data(many.assign(pair=[f"Y{i}USDT" for i in range(len(many))]), hd_, now, budget_s=1.5)
            chk(time.monotonic() - t0 < 2.5 and any("wall-clock" in x for x in notes), "hung request cut by the wall-clock budget")
            _NET_BLOCKED = True
            # freeze end-to-end through run(): 20 tick fills on 10 days, deferral by an OPEN fill, then frozen + persisted
            rows2 = []
            for i in range(20):
                ts = pd.Timestamp("2026-10-12 01:00:00") + pd.Timedelta(hours=13 * i)
                day = f"{ts:%Y-%m-%d}"
                tms = ts.value // 10**6
                path = [(tms + 2 * MIN, 99.0), (tms + 16 * MIN, 99.6), (tms + 30 * MIN, 97.0), (tms + 119 * MIN, 96.0)] if i % 2 else \
                       [(tms + 3 * MIN, tp_px)]
                os.makedirs(os.path.join(_CACHE, "ticks", f"Z{i}USDT"), exist_ok=True)
                for dd in _days(tms, 120):
                    pts = [(a, b) for a, b in path if f"{pd.Timestamp(a, unit='ms'):%Y-%m-%d}" == dd]
                    np.savez_compressed(os.path.join(_CACHE, "ticks", f"Z{i}USDT", f"{dd}.npz"),
                                        t=np.array([a for a, _ in pts], dtype=np.int64), p=np.array([b for _, b in pts], dtype=np.float32))
                live = float(net_pct(path[-1][1], 100.0))
                rows2.append(dict(base, opened_at=f"{ts:%Y-%m-%dT%H:%M:%S}", pair=f"Z{i}USDT", entry_price=100.0, pnl_percentage=live, pnl=0.0,
                                  close_reason="MAX_HOLD_TIME" if i % 2 else "FRENZY_TP", closed_at=day, entry_frenzy_willy_trigger="A"))
            zo = pd.DataFrame(rows2)
            nowz = pd.Timestamp("2026-11-01").value // 10**6
            out = "\n".join(run(nowz, orders=zo, fetch=False, state_path=STATE, th={}, open_ts=[(pd.Timestamp("2026-10-12 02:00"), "OPEN Q")], study=False))
            chk("freezing deferred" in out and not os.path.exists(STATE), "run(): an OPEN WILLY ≤ the prefix end defers the freeze")
            out = "\n".join(run(nowz, orders=zo, fetch=False, state_path=STATE, th={}, open_ts=[], study=False))
            s1 = json.load(open(STATE))
            chk("Frozen verdict" in out and s1["first"]["n"] == 20 and s1["first"]["state"].startswith("CAP20 CANDIDATE") and "20/20 match" in out,
                f"run() freezes at N 20 on ≥ 8 days: no-TP fills at 20 min (−0.49) beat 15 min (−1.09) and the 120 close (−4.09) ({s1['first']['state']})")
            out = "\n".join(run(nowz, orders=zo.assign(pnl_percentage=0.0), fetch=False, state_path=STATE, th={}, open_ts=[], study=False))
            chk(json.load(open(STATE)) == s1 and s1["first"]["state"] in out, "persisted verdict survives a later run unchanged")
            chk(set(s1) == {"first"} and "CAP60: ⏳ COLLECTING (N 19/20" in out and "Context, NOT counted (1 fill(s)" in out,
                "existing cells unchanged: the S4 cells add no key below their own N, the pre-floor fill is context only")
            # S4 end-to-end: 22 fills after the −4 stop went live; one walked on 1m (provisional) holds the freeze until its archive is final
            rows4, STATE4 = [], os.path.join(td, "S4.json")
            for i in range(22):
                ts = pd.Timestamp("2026-10-12 03:00:00") + pd.Timedelta(hours=13 * i)
                tms = ts.value // 10**6
                path = [(tms + 2 * MIN, 99.0), (tms + 16 * MIN, 99.6), (tms + 59 * MIN, 99.8), (tms + 89 * MIN, 99.9),
                        (tms + 119 * MIN, 96.5)] if i % 2 else [(tms + 3 * MIN, tp_px)]
                pr = f"S{i}USDT"
                if i == 2:                                        # 1m klines only (no tick day) → provisional; TP booked at +1
                    ok1 = (tms // MIN) * MIN + np.arange(0, 125) * MIN
                    k1 = pd.DataFrame(dict(open_time=ok1, o=99.9, h=99.95, l=99.85, c=99.9))
                    k1.loc[2, "h"] = tp_px
                    k1.to_csv(os.path.join(_CACHE, "k1m_ondemand", f"{pr}.csv"), index=False)
                    live = 1.0
                else:
                    os.makedirs(os.path.join(_CACHE, "ticks", pr), exist_ok=True)
                    for dd in _days(tms, 120):
                        pts = [(a, b) for a, b in path if f"{pd.Timestamp(a, unit='ms'):%Y-%m-%d}" == dd]
                        np.savez_compressed(os.path.join(_CACHE, "ticks", pr, f"{dd}.npz"),
                                            t=np.array([a for a, _ in pts], dtype=np.int64), p=np.array([b for _, b in pts], dtype=np.float32))
                    live = float(net_pct(path[-1][1], 100.0))
                rows4.append(dict(base, opened_at=f"{ts:%Y-%m-%dT%H:%M:%S}", pair=pr, entry_price=100.0, pnl_percentage=live, pnl=0.0,
                                  close_reason="MAX_HOLD_TIME" if i % 2 else "FRENZY_TP", closed_at="x", entry_frenzy_willy_trigger="A"))
            so4 = pd.DataFrame(rows4)
            out = "\n".join(run(nowz, orders=so4, fetch=False, state_path=STATE4, th={"frenzy_willy_stop_pct": 4.0}, open_ts=[], study=False))
            s4j = json.load(open(STATE4)) if os.path.exists(STATE4) else {}
            chk("S4_CAP60" not in s4j and "S4_CAP90" not in s4j and "CAP60 — 1 fill(s) opened" in out and "T125C90 — 1 fill(s) opened" in out and "1m-provisional S2USDT" in out
                and "S2USDT [1m, provisional]" in out, f"run(): a 1m-provisional fill in the prefix holds the S4 freeze\n{out}")
            att = load_attempts()
            t2 = pd.Timestamp("2026-10-13 05:00").value // 10**6
            for dd in _days(t2, 120):
                att["done"][_ak("S2USDT", dd)] = "404×3 (selftest)"
            save_attempts(att)
            out = "\n".join(run(nowz, orders=so4, fetch=False, state_path=STATE4, th={"frenzy_willy_stop_pct": 4.0}, open_ts=[], study=False))
            s4j = json.load(open(STATE4))
            chk(all(s4j[S4_KEY[n]]["state"] == f"{n} CANDIDATE (operator decides)" and s4j[S4_KEY[n]]["n"] == 20
                    and s4j[S4_KEY[n]]["config"] == dict(tp=1.0, stop=4.0, hold=120, cell_tp=s4_spec(n, 1.0)[0], cap=x) for n, _, x in S4_CELLS)
                and "CAP60: **Frozen verdict (pre-registered 2026-10-12, DECISION_LOG 278" in out and "| CAP90 (+1 / −4 / 90 min) | 22 |" in out,
                f"run(): archive final → all four S4 cells frozen (no-TP fills close at 60 / 90 min near flat vs −3.59 at 120) ({s4j})\n{out}")
            out = "\n".join(run(nowz, orders=so4.assign(pnl_percentage=0.0), fetch=False, state_path=STATE4, th={"frenzy_willy_stop_pct": 4.0},
                                open_ts=[], study=False))
            chk(json.load(open(STATE4)) == s4j, "run(): the S4 frozen verdicts survive a later run unchanged")
            man4 = pd.concat([so4, pd.DataFrame([dict(rows4[1], opened_at="2026-10-14T09:00:00", close_reason="MANUAL_CLOSE", pnl_percentage=0.3)])],
                             ignore_index=True)
            STATE5 = os.path.join(td, "S5.json")
            out = "\n".join(run(nowz, orders=man4, fetch=False, state_path=STATE5, th={"frenzy_willy_stop_pct": 4.0}, open_ts=[], study=False))
            chk("Excluded: 1 manually closed fill(s) (not the walked exit): 10-14 09:00 S1USDT (MANUAL_CLOSE)" in out
                and "| CAP90 (+1 / −4 / 90 min) | 22 |" in out
                and {k: v for k, v in json.load(open(STATE5)).items() if k.startswith("S4_")} == {k: v for k, v in s4j.items() if k.startswith("S4_")},
                "run(): a MANUAL close is dropped from the S4 cohort and listed (no parity miss, same S4 frozen verdicts)")
            STATE6 = os.path.join(td, "S6.json")
            out = "\n".join(run(nowz, orders=so4, fetch=False, state_path=STATE6,
                                th={"frenzy_willy_stop_pct": 4.0, "frenzy_willy_max_hold_minutes": 75}, open_ts=[], study=False))
            s6 = json.load(open(STATE6)) if os.path.exists(STATE6) else {}
            chk("| CAP90 (+1 / −4 / 90 min) | n/a (cap > hold 75 min) |" in out and "T125C90: n/a (cap > hold)" in out
                and not any(k.startswith("S4_") or k == OLD_KEY for k in s6) and "CAP60 — WALKER PARITY / INTEGRITY FAILED — not frozen" in out
                and "T125C90 — WALKER" not in out and "CAP90 — WALKER" not in out,
                f"run(): hold 75 → CAP90 / T125C90 n/a (never judged / frozen); CAP60 still judged, and refused on parity (the fills really "
                f"ran 120 min) ({sorted(s6)})")
            # ── DL 281 end-to-end: today = +1.25 / −4 / 90; every fill prints +1.06 at 5 min then drifts → new closes at 90 (−1.59), old TP
            rows8, STATE8 = [], os.path.join(td, "S8.json")
            th8 = {"frenzy_willy_tp_pct": 1.25, "frenzy_willy_stop_pct": 4.0, "frenzy_willy_max_hold_minutes": 90}
            for i in range(22):
                ts = pd.Timestamp("2026-10-12 04:00:00") + pd.Timedelta(hours=13 * i)
                tms = ts.value // 10**6
                path = [(tms + 5 * MIN, 101.15), (tms + 80 * MIN, 99.0), (tms + 89 * MIN, 98.5), (tms + 100 * MIN, 97.0)]
                pr = f"N{i}USDT"
                os.makedirs(os.path.join(_CACHE, "ticks", pr), exist_ok=True)
                for dd in _days(tms, 120):
                    pts = [(a, b) for a, b in path if f"{pd.Timestamp(a, unit='ms'):%Y-%m-%d}" == dd]
                    np.savez_compressed(os.path.join(_CACHE, "ticks", pr, f"{dd}.npz"),
                                        t=np.array([a for a, _ in pts], dtype=np.int64), p=np.array([b for _, b in pts], dtype=np.float32))
                rows8.append(dict(base, opened_at=f"{ts:%Y-%m-%dT%H:%M:%S}", pair=pr, entry_price=100.0, pnl_percentage=float(net_pct(98.5, 100.0)),
                                  pnl=0.0, close_reason="MAX_HOLD_TIME", closed_at="x", entry_frenzy_willy_trigger="A"))
            out = "\n".join(run(nowz, orders=pd.DataFrame(rows8), fetch=False, state_path=STATE8, th=th8, open_ts=[], study=False))
            s8 = json.load(open(STATE8))
            chk(s8[OLD_KEY]["state"] == OLD_REVERT and s8[OLD_KEY]["n"] == 20 and "new avg < 0" in s8[OLD_KEY]["detail"]
                and "old better" in s8[OLD_KEY]["detail"] and "S4_T125C90" not in s8 and {"S4_CAP60", "S4_CAP90", "S4_T125C60"} <= set(s8)
                and "### ↩ WILLY exit change revert gate (DL 281): new +1.25 / −4 / 90 vs old +1 / −4 / 120" in out
                and "| T125C90 (+1.25 / −4 / 90 min) | n/a (= today) |" in out and "**Frozen revert verdict (pre-registered 2026-10-12, DECISION_LOG 281" in out,
                f"run(): DL 281 cohort → OLD_EXIT REVERT CANDIDATE frozen, T125C90 n/a (= today) never frozen ({sorted(s8)})\n{out}")
            out = "\n".join(run(nowz, orders=so4, fetch=False, state_path=os.path.join(td, "S9.json"), th={"frenzy_willy_stop_pct": 4.0},
                                open_ts=[], study=False))
            chk("OLD_EXIT: n/a (= today)" in out and OLD_KEY not in json.load(open(os.path.join(td, "S9.json"))),
                "run(): today = the old exit → the revert gate is n/a, nothing frozen")
            g_s4 = globals()["s4_section"]

            def _boom4(*a, **k):
                raise ValueError("selftest")
            globals()["s4_section"] = _boom4
            try:
                out = "\n".join(run(nowz, orders=so4, fetch=False, state_path=os.path.join(td, "S7.json"), th={}, open_ts=[], study=False))
            finally:
                globals()["s4_section"] = g_s4
            chk("### ⏱ CAP60 / CAP90 / T125C60 / T125C90 on today's stop + the DL 281 revert gate (DECISION_LOG 278 / 281) — unavailable this run "
                "(ValueError: selftest)" in out,
                "run(): an S4 failure prints the fallback header naming all four cells, the line still renders")
    finally:
        (_CACHE, MY_CACHE, STATE, EXPORT_GLOB, _http_get, _open_url, STUDY_CSV, NF.SHARED_COOLDOWN, NF.DU_COOLDOWN, NF._CACHE,
         S4_DEPLOY_GREP, S4_DEPLOY_PIN) = saved
        _NET_BLOCKED = False
    print(f"selftest WILLY_TIMECAP OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    sys.modules.setdefault("scout_willy_timecap", sys.modules[__name__])   # the lazily imported hybrid walker shares THIS module (caches)
    if "--selftest" in sys.argv:
        selftest()
    elif "--build-study-ref" in sys.argv:
        t0 = time.time()
        R = build_study_ref(*levels()[::2])
        print(f"study reference rows: {len(R)} ({int(R.ok.sum())} walked) in {time.time() - t0:.0f} s → {_study_rows_path()}")
        print("\n".join(study_block()))
    else:
        print("\n".join(run()))
