#!/usr/bin/env python3
"""🩸 SPIKE_FADE watch lines — DECISION_LOG 374, observe-only (operator 2026-10-26: "add them to the watchlist"). Never changes config,
never calls the bot API, no network — stamped export columns only.

REGISTRATION    winners-vs-losers screen on 83 SPIKE_FADE fades (76 master stack-kept non-probe, reports/MASTER_POOL_stacked.csv stack_pct
                + 7 closed fades of scalpars_orders_paper_2026-10-26_12-39-59.csv; dedup opened_at[:19] + pair): 83 · 86 % WR · +0.265 %/fill ·
                avg win +0.546 / loss −1.394 → breakeven WR ≈ 72 %. NO separator beyond chance (0 / 38 entry columns at p < 0.05, median
                split vs 2,000 permutations; reports/FADE_WL_SCREEN_2026-10-19.md agrees) — the six lines are operator watch items.
LINE 1 ADX_RISING   entry_adx_delta ≥ +2.359 (pool median, 🔒 FROZEN): weak (< 2.359) 41 · 80 % · +0.104 vs strong 42 · 90 % · +0.423
                    at the frozen threshold (screen median split: hi +0.420 / lo +0.114), permutation p 0.10.
                    Filter candidate = block the WEAK side (entry_adx_delta < 2.359). In-sample block = −$832 (41 fades · 80 % WR).
LINE 2 PAIR_ADX_LOW entry_adx ≤ 19.2 (pool low quarter, 🔒 FROZEN): weak 20 · 80 % · +0.096 vs rest 63 · 87 % · +0.319, p 0.35.
                    Filter candidate = block the WEAK side (entry_adx ≤ 19.2). B25: RAYSOL 14.8 in the zone; 1000LUNC 19.37 just above the
                    frozen 19.2 (not blocked).
                    In-sample block = −$550 (20 fades · 80 % WR). Both weak sides sit ABOVE the ≈ 72 % breakeven; the registration Δ is
                    in-sample (30–50 % haircut) and NOT a ship case (per-batch table: the 2026-10-26 session report, no file).
LINE 3 MCAP_CAP15   size, not filter: SPIKE_FADE runs spike_fade_invest_mult 2.0 × lev_mult 1.0 (cell_multiplier 2.0). Every stamped fade
                    with entry_mcap_usd ≥ $200M (the DL 311 threshold) since the DL 311 mcap-stamp deploy (4353aa9, 2026-10-19 05:08 UTC):
                    $ at its traded multiplier vs the shadows at 1.5× and 1× = pnl × min(X / cell_multiplier, 1); a fill sized below its
                    desired notional (liquidity_capped true, or notional < desired) re-tickets at
                    min(desired × X / cell_multiplier, entry_liquidity_cap_notional, notional) → shadow $ = pnl × ticket / notional
                    (liquidity_capped but desired unknown → no saving, flagged). ⚠ DL 311 / 41i gate TRIGGERED NOW (1000LUNC 10-25, $249M,
                    −$308 ≈ −10.6 % of the $2,900 paper account) → the mcap / size question is REOPENED at the B25 review. 1.5× prices it
                    −$308 → −$236 (still ≈ −8.1 %; 1× ≈ −$157, −5.4 %): no size cap alone meets the 4 % bar at today's −1.5 % stop — the decision is
                    stop width × size (NEXT_BATCH_REVIEW_PLAN §5). DL 312's "1× if ≥ $300M" did not fire ($249M < $300M).
LINE 4 HB_SIZE      HEALTHY_BEAR size shadow (operator 2026-10-26 "add HEALTHY_BEAR smaller size to watchlist"): fades with
                    entry_btc_regime == HEALTHY_BEAR (unstamped → unscored) priced at 1× invest instead of the traded 2× (shadow $ = pnl ×
                    min(1.0 / cell_multiplier, 1), same sized-below-desired handling as line 3). Registration (pool 83): HEALTHY_BEAR 24 · 71 %
                    WR · +0.07 %/fill · +$711 · 7 of the 13 pool stops (master 22 · 73 % · +0.13 · +$978 · 6 stops; B25 2 · 50 % · −0.62 ·
                    −$267), permutation p 0.17 vs the rest; STRONG_BEAR 10 · 100 % · +0.52 · BEAR_EXHAUSTED 4 · 100 % · CHOPPY_WEAK 27 · 93 % ·
                    +0.39 · CHOPPY_FLAT 18 · 83 % · +0.12. ⚠ the 10-19 study's today's-stack BACKTEST contradicts it (zone 121 · 77 % · +0.23
                    vs +0.02). The SIZE version of the 41h observe gate (HEALTHY_BEAR ∧ entry_btc_trend_gap_pct ≥ −0.13, sub-zone tallied
                    here) — sizing on a still-winning cohort (CLAUDE.md "high WR but net-losing is NOT a filter candidate when the loss is
                    sizing"), not a block.
LINE 5 HB_ADXFLAT  HEALTHY_BEAR ∧ entry_adx_delta ≤ 2.4341 (🔒 FROZEN: the backtest HB median) — entry-FILTER candidate (block the zone).
LINE 6 HB_BTCADX   HEALTHY_BEAR ∧ entry_btc_adx_prev > 24.5874 (🔒 FROZEN: the backtest HB 75th percentile) — entry-FILTER candidate.
                    Discovery (lines 5 / 6): HB × one-variable screen (38 stamped cols × q25/q50/q75 × lo/hi = 226 cells) on the 180
                    HEALTHY_BEAR fades of the 2-year backtest (reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv, tick-priced pct from
                    reports/FADE_SIZE_MCAP_2026-10-19_rows.csv), confirmed on the 24 HB master + B25 fades: family max |gap| 0.539 vs null p95
                    0.775 (P 0.72); confirmations 37 / 113 vs null mean 36.4 (P 0.43) → NO separator beyond chance; the most consistent pair,
                    watch lines only. Zone = HB rows at the threshold; rest = every other scored fade (non-HB included); an HB fade without the
                    variable, or a fade without a regime stamp → unscored. Line 5 overlaps lines 1 and 4.
FORWARD COHORT  lines 1 / 2 / 4 / 5 / 6: CLOSED SPIKE_FADE fills (paper + live via export_glob) opened ≥ the DL 374 commit + 10 min
                (scout_gatekit.dl_deploy_ms; not in git → NOT EVALUABLE, nothing counted, nothing persisted) · line 3: opened ≥ the DL 311
                deploy (dl_deploy_ms(311, lag 0), else pinned 2026-10-19 05:08 UTC); dedup (opened_at[:19], pair, direction), CLOSED wins, newest export;
                operator-closed (MANUAL / MANUAL_* / EXTERNAL_CLOSE) excluded — scout_gatekit. A missing stamp → UNSCORED (never "rest").
🔒 READS (decided ONCE, persisted in reports/.scout_fade_watch.json; COLLECTING / HOLD never persisted)
  Lines 1 / 2 / 5 / 6 — CLAUDE.md EXPECTANCY filter bar on the forward WEAK-side cohort, read at the first prefix by open time where it reaches
                N ≥ 15 on ≥ 8 UTC days (window = UTC day): WR < the sleeve breakeven WR (|avg loss| / (avg win + |avg loss|) of the prefix's
                forward kept fades; 72 % if < 5 losses) ∧ P(avg < 0) ≥ 0.95 by a day-clustered bootstrap (4,000 · seed 7) ∧ ≥ 8 days ∧ no
                single day / pair ≥ 50 % of the cohort loss ∧ N ≥ 15 → FILTER_CANDIDATE (operator decides; 30–50 % in-sample haircut; revert
                if shipped: the first 10 blocked fades re-priced average > 0); else KEEP_OBSERVING (frozen threshold, never re-fit).
  Line 4      — at the shortest prefix of ≥ 15 HEALTHY_BEAR fades on ≥ 8 UTC days: SIZE_CUT_CANDIDATE ("HEALTHY_BEAR fades at 1×" —
                operator decides) iff avg P&L % < 0 ∧ day-clustered bootstrap P(avg < 0) ≥ 0.80 (DELIBERATELY below the 0.95 filter bar: a
                size cut is the lighter action) ∧ no single day ≥ 50 % of the cohort loss; else KEEP_2X.
  Line 3      — confirmation tracker at the first 10 high-mcap fades: CAP_CANDIDATE (operator decides "high-mcap fades max 1.5×") iff
                their avg P&L % < 0 (DL 311 wording; % never raw $ — CLAUDE.md principle 3); else KEEP_2X. Informational, never in the
                read: Σ$ traded vs 1.5× vs 1× per book (paper / live never summed) and the > 4 %-of-account test at each size (reference
                equity = the newest export's balance column for the fill's book, else ASSUMED $2,900 paper / $1,225 live).
Usage           venv/bin/python scripts/scout_fade_watch.py      (batch review)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
import export_glob as EG                                     # noqa: E402
import scout_gatekit as _GK                                  # noqa: E402
import scout_lite_late_btcfall as _LLB                       # noqa: E402  day-clustered bootstrap + loss-share helpers (DL 353)

DL = 374
SLEEVE = "SPIKE_FADE"
ADX_DELTA_MIN = 2.359                                        # 🔒 line 1: strong side ≥, weak side <
PAIR_ADX_MAX = 19.2                                          # 🔒 line 2: weak side ≤
MCAP_MIN = 200e6                                             # 🔒 line 3: the DL 311 threshold
HB_ADXD_MAX, HB_BTCADX_MIN = 2.4341, 24.5874              # 🔒 lines 5 / 6 (backtest HB median / 75th percentile)
HB_REGIME, HB_MULT, HB_GAP_MIN, HB_P_MIN = "HEALTHY_BEAR", 1.0, -0.13, 0.80   # 🔒 line 4 (41h sub-zone: trend gap ≥ −0.13)
MCAP_DL, MCAP_FLOOR_PIN = 311, "2026-10-19T05:08:00"         # line 3 cohort floor = the DL 311 mcap-stamp deploy (4353aa9)
CAP_MULT = 1.5                                               # the operator's candidate invest cap on high-mcap fades
GATE_N, GATE_DAYS, P_MIN, SHARE_MAX = 15, 8, 0.95, 0.50
BE_FALLBACK, BE_MIN_LOSSES = 72.0, 5
BOOT_N, BOOT_SEED = 4000, 7
MCAP_READ_N, ACCT_LOSS_PCT = 10, 4.0
ACCOUNT_PAPER, ACCOUNT_LIVE = 2900.0, 1225.0                 # stated assumptions when no export carries a balance column
BALANCE_COLS = ("balance", "account_balance", "wallet_balance", "total_balance", "equity", "account_equity")
STATE = os.path.join(_R, "reports", ".scout_fade_watch.json")
PATTERN = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "is_paper",
        "entry_adx", "entry_adx_delta", "entry_mcap_usd", "cell_multiplier", "notional_value", "entry_desired_notional",
        "entry_liquidity_cap_notional", "liquidity_capped", "entry_btc_regime", "entry_btc_trend_gap_pct", "entry_btc_adx_prev") + BALANCE_COLS
INSAMPLE = ("the weak side sits above the ≈ 72 % breakeven; registration Δ is in-sample (30–50 % haircut) and NOT a ship case "
            "(per-batch table: the 2026-10-26 session report)")
REG = dict(ADX_RISING="registration (pool 83 · 86 % · +0.265 %/fill · breakeven ≈ 72 %): at the frozen +2.359 (median) weak (ADX Δ < 2.359) "
                      "41 · 80 % · +0.104 vs strong 42 · 90 % · +0.423 · permutation p 0.10 — no separator beyond chance (0 / 38 columns) · in-sample block = "
                      "−$832 (41 fades · 80 % WR) — " + INSAMPLE,
           PAIR_ADX_LOW="registration (pool 83): at the frozen 19.2 (low quarter) weak (pair ADX ≤ 19.2) 20 · 80 % · +0.096 vs rest 63 · 87 % · +0.319 · "
                        "p 0.35 · B25: RAYSOL 14.8 in the zone; 1000LUNC 19.37 just above the frozen 19.2 (not blocked) · in-sample block = −$550 (20 fades · 80 % WR) — " + INSAMPLE,
           MCAP_CAP15="⚠ DL 311 / 41i gate TRIGGERED NOW (1000LUNC 10-25, $249M, −$308 ≈ −10.6 % of the $2,900 paper account) → the mcap / size "
                      "question is REOPENED at the B25 review. Line ③ prices the operator's 1.5× candidate on every stamped ≥ $200M fade since "
                      "10-19 (−$308 → −$236, still ≈ −8.1 %; 1× ≈ −$157, −5.4 %). No size cap alone meets the 4 % bar at today's stop width — the "
                      "decision is stop width × size (plan §5). The forward read (first 10, avg P&L % < 0 per DL 311) continues as the "
                      "confirmation tracker. DL 312's '1× if ≥ $300M' did not fire ($249M < $300M).",
           HB_SIZE="registration (pool 83, master stack_pct + B25): HEALTHY_BEAR 24 · 71 % WR · +0.07 %/fill · +$711 · 7 of the 13 pool stops "
                   "(master 22 · 73 % · +0.13 · +$978 · 6 stops; B25 2 · 50 % · −0.62 · −$267) · permutation p 0.17 vs the rest · STRONG_BEAR 10 · "
                   "100 % · +0.52 · BEAR_EXHAUSTED 4 · 100 % · CHOPPY_WEAK 27 · 93 % · +0.39 · CHOPPY_FLAT 18 · 83 % · +0.12 · ⚠ the 10-19 study's "
                   "today's-stack BACKTEST contradicts it: zone 121 · 77 % · +0.23 vs +0.02 · ④ is the SIZE version of the 41h observe gate "
                   "(HEALTHY_BEAR ∧ BTC trend gap ≥ −0.13): sizing on a still-winning cohort, not a block")
_HB_DISC = ("discovery: HB × one-variable screen (226 cells) on 180 backtest HB fades, confirmed on 24 HB master + B25 fades — family max "
            "|gap| 0.539 vs null p95 0.775 (P 0.72), confirmations 37 / 113 vs null 36.4 (P 0.43) → NO separator beyond chance; in-sample "
            "hindsight (30–50 % haircut), watch line only")
REG.update(
    HB_ADXFLAT="⑤ HEALTHY_BEAR ∧ entry_adx_delta ≤ 2.4341 (frozen, backtest HB median): backtest 90 · −0.06 vs rest +0.40 · master + B25 15 · "
               "67 % · −0.14 · 5 of HB's 7 stops (rest 9 · +0.42) · block-zone before → after: B25 7 · −$396 → 6 · −$92 (blocks RAYSOL −$304) · "
               "master 76 · +$4,237 → 62 · +$4,095 (blocked 14 · 71 % · −0.05 % · +$141) · ALL 83 · +$3,841 → 68 · +$4,004 · overlaps ① and ④ · "
               + _HB_DISC,
    HB_BTCADX="⑥ HEALTHY_BEAR ∧ entry_btc_adx_prev > 24.5874 (frozen, backtest HB 75th pct): backtest 45 · −0.17 vs rest +0.29 · master + B25 "
              "7 · 43 % · −0.47 · 4 stops · block-zone before → after: B25 unchanged (RAYSOL BTC ADX 22.2) · master 76 · +$4,237 → 69 · +$4,652 "
              "(blocked 7 · 43 % · −$415) · ALL 83 → 76 · +$4,256 · " + _HB_DISC)
LINES = (("ADX_RISING", "① ADX rising at entry", "entry_adx_delta", "entry_adx_delta < 2.359", "entry_adx_delta ≥ 2.359"),
         ("PAIR_ADX_LOW", "② Pair ADX ≤ 19.2", "entry_adx", "entry_adx ≤ 19.2", "entry_adx > 19.2"),
         ("HB_ADXFLAT", "⑤ HEALTHY_BEAR + ADX not rising", "entry_btc_regime + entry_adx_delta",
          "zone: HEALTHY_BEAR ∧ entry_adx_delta ≤ 2.4341", "rest — every other scored fade"),
         ("HB_BTCADX", "⑥ HEALTHY_BEAR + BTC ADX high", "entry_btc_regime + entry_btc_adx_prev",
          "zone: HEALTHY_BEAR ∧ entry_btc_adx_prev > 24.5874", "rest — every other scored fade"))
REVERT = "if shipped: revert if the first 10 blocked fades re-priced average > 0"
BLIND = ("Blind spots: the registration pool is master (mixed exit eras re-priced to today's stack) + 7 B25 export fills; the mcap stamp exists only "
         "since 10-19 (the master has none — reports/FADE_SIZE_MCAP_2026-10-19.md scored it separately); windows = UTC days (a multi-day "
         "regime is several windows); the > 4 %-of-account test uses a FIXED reference equity (the paper balance drifts with P&L); no backtest block yet — re-read the fade backtest (yr5 / yr6) for each line at the B25 review.")


def _ts(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:19].str.replace(" ", "T"), errors="coerce", format="%Y-%m-%dT%H:%M:%S")


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def _paths(paths):
    return paths if paths is not None else sorted(EG.glob_exports(PATTERN), key=os.path.getmtime)


def load(paths=None):
    """every export row (paper + live), dedup (opened_at[:19], pair, direction) — a CLOSED row wins, then the newest export."""
    fr = []
    for f in _paths(paths):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "y", "usd", "paper"])
    a = pd.concat(fr, ignore_index=True)
    for c in COLS:
        if c not in a:
            a[c] = None
    a["_k"] = a.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + a.pair.astype(str) + "|" + a.direction.astype(str).str.upper()
    a = a.assign(_c=(a.status.astype(str) == "CLOSED").astype(int)).reset_index(drop=True)
    a = a.assign(_i=range(len(a))).sort_values(["_c", "_i"]).drop_duplicates("_k", keep="last")
    a["o"] = _ts(a.opened_at).values
    a["y"] = pd.to_numeric(a.pnl_percentage, errors="coerce")
    a["usd"] = pd.to_numeric(a.pnl, errors="coerce")
    a["paper"] = [_truthy(v) for v in a.is_paper]
    return a[a.o.notna()].drop(columns=["_c", "_i", "_k"]).sort_values("o").reset_index(drop=True)


def forward(a, since):
    """→ (kept closed SPIKE_FADE fills opened ≥ since, operator-closed rows)."""
    if not len(a):
        return a, a
    f = a[(a.entry_strategy.astype(str) == SLEEVE) & (a.status.astype(str) == "CLOSED") & (a.o >= since) & a.y.notna()]
    kept, ox = _GK.split_operator_closed(f) if len(f) else (f, f)
    return kept.sort_values("o").reset_index(drop=True), ox


def side(df, key):
    """'weak' / 'strong' / 'unscored' per row at the frozen threshold of line key."""
    if key == "ADX_RISING":
        v = pd.to_numeric(df.entry_adx_delta, errors="coerce")
        return np.where(v.isna(), "unscored", np.where(v < ADX_DELTA_MIN, "weak", "strong"))
    if key == "PAIR_ADX_LOW":
        v = pd.to_numeric(df.entry_adx, errors="coerce")
        return np.where(v.isna(), "unscored", np.where(v <= PAIR_ADX_MAX, "weak", "strong"))
    # ⑤ / ⑥ — zone = HEALTHY_BEAR at the threshold; a non-HB scored fade is rest; HB without the variable / no regime stamp → unscored
    col, inzone = (("entry_adx_delta", lambda x: x <= HB_ADXD_MAX) if key == "HB_ADXFLAT"
                   else ("entry_btc_adx_prev", lambda x: x > HB_BTCADX_MIN))
    v = pd.to_numeric(df[col], errors="coerce").values
    hb = hb_split(df)
    return np.array(["unscored" if h == "unscored" else ("strong" if h == "other" else
                     ("unscored" if pd.isna(x) else ("weak" if inzone(x) else "strong"))) for h, x in zip(hb, v)], dtype=object)


def breakeven(y):
    """sleeve breakeven WR % = |avg loss| / (avg win + |avg loss|); < 5 losses (or no win) → the registration 72 %."""
    y = pd.Series(y, dtype=float)
    w, l = y[y > 0], y[y < 0]
    if len(l) < BE_MIN_LOSSES or not len(w):
        return BE_FALLBACK, True
    return 100.0 * abs(l.mean()) / (w.mean() + abs(l.mean())), False


def pending_opens(a, since):
    """open times of every non-CLOSED SPIKE_FADE row in the loaded exports opened ≥ since — a fade that may still join a prefix."""
    if a is None or not len(a):
        return pd.Series([], dtype="datetime64[ns]")
    m = (a.entry_strategy.astype(str) == SLEEVE) & (a.status.astype(str) != "CLOSED") & (a.o >= since)
    return a.o[m].reset_index(drop=True)


def _hold(pending, t_end):
    """→ HOLD text when a still-open fade opened ≤ the crossing fill (its outcome belongs in the prefix), else None. HOLD is never persisted."""
    if pending is None or not len(pending):
        return None
    n = int((pd.Series(pending) <= t_end).sum())
    if not n:
        return None
    return (f"⏸ HOLD — the read crossed at {t_end:%m-%d %H:%M} but {n} SPIKE_FADE fill(s) opened ≤ then are still OPEN in the exports; "
            "decided once they close (never persisted while held)")


def filter_read(fw, key, pending=None):
    """🔒 the expectancy bar on the forward weak side → (verdict, text). COLLECTING until the weak side reaches N ≥ 15 on ≥ 8 UTC days; then
    decided on that prefix (every forward kept fade opened ≤ the crossing fill). Never raises."""
    try:
        if not len(fw):
            return "COLLECTING", f"⏳ weak side 0/{GATE_N} fills on 0/{GATE_DAYS} days"
        g = fw.assign(side=side(fw, key)).sort_values(["o", "pair"], kind="mergesort").reset_index(drop=True)
        wk_idx = g.index[g.side == "weak"]
        end = None
        for k, i in enumerate(wk_idx):
            if k + 1 >= GATE_N and g.loc[wk_idx[:k + 1], "o"].dt.date.nunique() >= GATE_DAYS:
                end = i
                break
        wk_all = g[g.side == "weak"]
        if end is None:
            return "COLLECTING", (f"⏳ weak side {len(wk_all)}/{GATE_N} fills on {wk_all.o.dt.date.nunique()}/{GATE_DAYS} days — decided once at "
                                  "the read")
        held = _hold(pending, g.o[end])
        if held:
            return "HOLD", held
        pre = g[g.o <= g.o[end]]
        wk = pre[pre.side == "weak"]
        be, fb = breakeven(pre.y)
        wr = 100.0 * float((wk.y > 0).mean())
        avg = float(wk.y.mean())
        p = _LLB.p_neg(wk.y.values, wk.o.dt.date.values, nb=BOOT_N, seed=BOOT_SEED)
        days = wk.o.dt.date.nunique()
        dsh, psh = _LLB.loss_share(wk, wk.o.dt.date), _LLB.loss_share(wk, wk.pair)
        ok = wr < be and avg < 0 and p >= P_MIN and days >= GATE_DAYS and dsh < SHARE_MAX and psh < SHARE_MAX and len(wk) >= GATE_N
        txt = (f"read at {len(wk)} weak fills on {days} days (to {pre.o.iloc[-1]:%m-%d %H:%M}): WR {wr:.0f} % vs breakeven {be:.0f} %"
               f"{' (fallback)' if fb else ''} · avg {avg:+.3f} % · P(avg < 0) {p:.2f} · worst day {dsh:.0%} / pair {psh:.0%} of the loss")
        if ok:
            return "FILTER_CANDIDATE", (txt + " → 🚨 FILTER CANDIDATE (operator decides; apply the 30–50 % in-sample haircut to any Δ; "
                                        + REVERT + ")")
        return "KEEP_OBSERVING", txt + " → keep observing (bar not all met; frozen threshold, never re-fit)"
    except Exception as e:
        return "ERROR", f"read unreadable ({str(e)[:80]})"


def shadow_usd(r, mult=CAP_MULT):
    """the shadow $ of one fill at an invest cap of mult → ($, label). pnl × min(mult / cell_multiplier, 1); a fill sized below its desired
    notional re-tickets at min(desired × mult / cell_multiplier, entry_liquidity_cap_notional, notional) → pnl × ticket / notional.
    label: 'liquidity-capped' (liquidity_capped true) · 'sized below desired' (notional < desired, flag false) · 'liquidity-capped,
    desired unknown — no saving' (flag true, desired NaN → pnl unchanged, never × ratio) · 'liquidity-capped, notional unknown — no
    saving' (flag true, notional NaN / ≤ 0 → pnl unchanged) · '' otherwise. Never above the traded size."""
    pnl = float(r["usd"]) if pd.notna(r.get("usd")) else np.nan
    cm = pd.to_numeric(r.get("cell_multiplier"), errors="coerce")
    ratio = min(mult / cm, 1.0) if pd.notna(cm) and cm > 0 else 1.0           # unknown multiplier → no shadow change
    nv = pd.to_numeric(r.get("notional_value"), errors="coerce")
    des = pd.to_numeric(r.get("entry_desired_notional"), errors="coerce")
    cap = pd.to_numeric(r.get("entry_liquidity_cap_notional"), errors="coerce")
    flag = _truthy(r.get("liquidity_capped"))
    below = pd.notna(nv) and pd.notna(des) and nv < des * 0.999
    if flag and pd.isna(des):
        return pnl, "liquidity-capped, desired unknown — no saving"
    if flag and (pd.isna(nv) or nv <= 0):
        return pnl, "liquidity-capped, notional unknown — no saving"
    if (flag or below) and pd.notna(nv) and nv > 0:
        tk = min(des * ratio, nv)
        if pd.notna(cap) and cap > 0:
            tk = min(tk, cap)
        return pnl * tk / nv, ("liquidity-capped" if flag else "sized below desired")
    return pnl * ratio, ""


def ref_equity(paths=None):
    """→ ({'paper': $, 'live': $}, source text). Per book: the newest export whose balance column has a value on a row of THAT book
    (is_paper) wins; a book with none falls back to the stated assumption."""
    eq, src = {}, {}
    for f in reversed(list(_paths(paths))):
        if len(eq) == 2:
            break
        try:
            h = pd.read_csv(f, nrows=0).columns
            bc = next((c for c in BALANCE_COLS if c in h), None)
            if not bc:
                continue
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "is_paper", bc))
            d = d.assign(v=pd.to_numeric(d[bc], errors="coerce"),
                         paper=[_truthy(x) for x in (d.is_paper if "is_paper" in d else ["true"] * len(d))])
            d = d[d.v.notna() & (d.v > 0)].sort_values("opened_at")
            for book, pp in (("paper", True), ("live", False)):
                g = d[d.paper == pp]
                if book not in eq and len(g):
                    eq[book] = float(g.v.iloc[-1])
                    src[book] = f"{book} ${eq[book]:,.0f} (column {bc} of {os.path.basename(f)})"
        except Exception:
            continue
    for book, v in (("paper", ACCOUNT_PAPER), ("live", ACCOUNT_LIVE)):
        if book not in eq:
            eq[book] = v
            src[book] = f"{book} ASSUMED ${v:,.0f} (no balance column)"
    return eq, "reference equity " + " / ".join(src[b] for b in ("paper", "live"))


def mcap_floor_ms():
    """line ③ cohort floor = the DL 311 mcap-stamp deploy (lag 0) → (ms, text); not in git → the pinned 2026-10-19 05:08 UTC."""
    ms = _GK.dl_deploy_ms(MCAP_DL, lag_min=0, label="SPIKE_FADE watch ③ (DL 311 floor)")
    if ms is not None:
        return ms, f"DL {MCAP_DL} deploy"
    return int(pd.Timestamp(MCAP_FLOOR_PIN).value // 1_000_000), f"DL {MCAP_DL} deploy, pinned"


def mcap_rows(fw, eq):
    """high-mcap fades with traded / 1.5× / 1× $ and % of the book's reference equity → (rows frame, unscored count)."""
    m = pd.to_numeric(fw.entry_mcap_usd, errors="coerce") if len(fw) else pd.Series(dtype=float)
    uns = int(m.isna().sum())
    h = fw[(m >= MCAP_MIN).values].copy() if len(fw) else fw
    if not len(h):
        return h, uns
    s15 = [shadow_usd(r, CAP_MULT) for _, r in h.iterrows()]
    s10 = [shadow_usd(r, 1.0) for _, r in h.iterrows()]
    h["shadow"] = [x for x, _ in s15]
    h["shadow1"] = [x for x, _ in s10]
    h["szlab"] = [lab for _, lab in s15]
    h["acct"] = [eq["paper"] if p else eq["live"] for p in h.paper]
    h["loss_pct"] = -h.usd / h.acct * 100.0
    h["loss_pct15"] = -h.shadow / h.acct * 100.0
    h["loss_pct1"] = -h.shadow1 / h.acct * 100.0
    h["mcap"] = pd.to_numeric(h.entry_mcap_usd, errors="coerce")
    return h.sort_values("o").reset_index(drop=True), uns


def mcap_read(h, pending=None):
    """🔒 confirmation tracker at the first 10 high-mcap fades (DL 311 wording) → (verdict, text): avg P&L % < 0 → CAP_CANDIDATE."""
    h = h[h.y.notna()] if len(h) else h
    if len(h) < MCAP_READ_N:
        return "COLLECTING", f"⏳ {len(h)}/{MCAP_READ_N} high-mcap fades — decided once at the read"
    f = h.head(MCAP_READ_N)
    held = _hold(pending, f.o.iloc[-1])
    if held:
        return "HOLD", held
    avg, wr = float(f.y.mean()), float((f.y > 0).mean())
    txt = (f"read at the first {MCAP_READ_N} (to {f.o.iloc[-1]:%m-%d %H:%M}): avg P&L {avg:+.3f} % · WR {wr:.0%} · "
           f"{int((f.loss_pct > ACCT_LOSS_PCT).sum())} lost > {ACCT_LOSS_PCT:g} % of the account at the traded size (informational)")
    if avg < 0:
        return "CAP_CANDIDATE", txt + " → 🚨 CAP CANDIDATE (operator decides \"high-mcap fades max 1.5×\"; with the stop-width question)"
    return "KEEP_2X", txt + " → keep 2× (frozen)"


def hb_split(df):
    """'hb' / 'other' / 'unscored' per row (entry_btc_regime; blank → unscored)."""
    r = ["" if pd.isna(x) else str(x).strip().upper() for x in df.entry_btc_regime] if len(df) else []
    return np.array(["unscored" if x in ("", "NAN", "NONE") else ("hb" if x == HB_REGIME else "other") for x in r], dtype=object)


def hb_read(hb, pending=None):
    """🔒 line ④ on the HEALTHY_BEAR fades (by open time) → (verdict, text). Decided at the shortest prefix of ≥ 15 on ≥ 8 UTC days."""
    try:
        g = hb[hb.y.notna()].sort_values(["o", "pair"], kind="mergesort").reset_index(drop=True) if len(hb) else hb
        end = None
        for k in range(len(g)):
            if k + 1 >= GATE_N and g.o.iloc[:k + 1].dt.date.nunique() >= GATE_DAYS:
                end = k
                break
        if end is None:
            nd = g.o.dt.date.nunique() if len(g) else 0
            return "COLLECTING", f"⏳ {len(g)}/{GATE_N} HEALTHY_BEAR fades on {nd}/{GATE_DAYS} days — decided once at the read"
        c = g.iloc[:end + 1]
        held = _hold(pending, c.o.iloc[-1])
        if held:
            return "HOLD", held
        avg = float(c.y.mean())
        p = _LLB.p_neg(c.y.values, c.o.dt.date.values, nb=BOOT_N, seed=BOOT_SEED)
        dsh = _LLB.loss_share(c, c.o.dt.date)
        txt = (f"read at {len(c)} fades on {c.o.dt.date.nunique()} days (to {c.o.iloc[-1]:%m-%d %H:%M}): WR {(c.y > 0).mean():.0%} · avg "
               f"{avg:+.3f} % · P(avg < 0) {p:.2f} (bar {HB_P_MIN}, deliberately below the 0.95 filter bar — a size cut is the lighter "
               f"action) · worst day {dsh:.0%} of the loss")
        if avg < 0 and p >= HB_P_MIN and dsh < SHARE_MAX:
            return "SIZE_CUT_CANDIDATE", txt + " → 🚨 SIZE-CUT CANDIDATE (\"HEALTHY_BEAR fades at 1×\" — operator decides)"
        return "KEEP_2X", txt + " → keep 2× (frozen)"
    except Exception as e:
        return "ERROR", f"read unreadable ({str(e)[:80]})"


def hb_lines(fw, state_path, pending=None):
    """line ④ — HEALTHY_BEAR fades: traded $ vs the 1× shadow, per book, the 41h sub-zone, the frozen read."""
    g = fw.assign(hb=hb_split(fw)) if len(fw) else fw.assign(hb=[])
    hb = g[g.hb == "hb"].copy() if len(g) else g
    L = [f"**④ HEALTHY_BEAR size shadow** (entry_btc_regime == {HB_REGIME}; traded vs 1× invest, shadow = pnl × min(1 / cell_multiplier, 1), "
         "sized-below-desired fills re-ticketed as in ③) — SIZE, not a block:"]
    for lab, x in (("HEALTHY_BEAR", hb), ("other regimes", g[g.hb == "other"] if len(g) else g),
                   ("unscored (no regime stamp)", g[g.hb == "unscored"] if len(g) else g)):
        L.append(f"  - {lab}: {_stat(x)}")
    if len(hb):
        sh = [shadow_usd(r, HB_MULT) for _, r in hb.iterrows()]
        hb["shadow1"] = [x for x, _ in sh]
        hb["szlab"] = [lab for _, lab in sh]
        gap = pd.to_numeric(hb.entry_btc_trend_gap_pct, errors="coerce")
        sub = hb[(gap >= HB_GAP_MIN).values]
        L.append(f"  - 41h sub-zone (HEALTHY_BEAR ∧ entry_btc_trend_gap_pct ≥ {HB_GAP_MIN}): {_stat(sub)} · gap unstamped {int(gap.isna().sum())}")
        for r in hb.itertuples():
            L.append(f"    · {r.pair} {r.o:%m-%d %H:%M} {'paper' if r.paper else 'live'} · mult {r.cell_multiplier} · {r.y:+.2f} % · traded "
                     f"${r.usd:+.2f} vs 1× ${r.shadow1:+.2f}" + (f" ({r.szlab})" if r.szlab else ""))
        hv = hb[hb.usd.notna()]
        for book, pp in (("paper", True), ("live", False)):                 # $ per book, never summed across books
            b = hv[hv.paper == pp]
            if len(b):
                L.append(f"  {book}: Σ traded ${b.usd.sum():+.2f} vs 1× ${b.shadow1.sum():+.2f} (Δ ${b.shadow1.sum() - b.usd.sum():+.2f})")
    v, txt = decide_once(state_path, "HB_SIZE", *hb_read(hb, pending))
    L.append(f"  Read (≥ {GATE_N} fades on ≥ {GATE_DAYS} days: avg P&L % < 0 ∧ P(avg < 0) ≥ {HB_P_MIN} ∧ no day ≥ 50 % of the loss → "
             f"SIZE_CUT_CANDIDATE, else KEEP_2X): {v} — {txt}")
    return L


def _load_state(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def decide_once(state_path, key, v, txt):
    """🔒 the first verdict that is not COLLECTING / HOLD / ERROR is written under key and returned on every later run → (verdict, text)."""
    st = _load_state(state_path) if state_path else {}
    if key in st:
        return st[key]["verdict"], st[key]["text"] + f" (frozen {st[key]['at'][:16]} UTC)"
    if state_path and v not in ("COLLECTING", "HOLD", "ERROR"):
        st[key] = {"verdict": v, "text": txt, "at": pd.Timestamp.now("UTC").isoformat()}
        try:
            tmp = f"{state_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(st, fh, indent=1)
            os.replace(tmp, state_path)
        except Exception:
            pass
    return v, txt


def _stat(g):
    if not len(g):
        return "N 0"
    return (f"N {len(g)} · {g.o.dt.date.nunique()} days · WR {(g.y > 0).mean():.0%} · avg {g.y.mean():+.3f} % · Σ {g.y.sum():+.2f} pts · "
            f"${g.usd.sum(skipna=True):+.2f}")


def _floor_text(since, how):
    return f"{since:%Y-%m-%d %H:%M} UTC ({how})"


def mcap_lines(a, state_path, paths, mcap_since_ms=None):
    """line ③ — independent of the DL 374 commit (its cohort starts at the DL 311 mcap-stamp floor)."""
    if mcap_since_ms is None:
        ms, how = mcap_floor_ms()
    else:
        ms, how = int(mcap_since_ms), "override"
    since = pd.Timestamp(int(ms), unit="ms")
    fw, ox = forward(a, since)
    pend = pending_opens(a, since)
    eq, eq_src = ref_equity(paths)
    h, uns = mcap_rows(fw, eq)
    L = [f"**③ High-mcap 1.5× cap** (entry_mcap_usd ≥ ${MCAP_MIN / 1e6:.0f}M · closed SPIKE_FADE fills opened ≥ {_floor_text(since, how)} · "
         f"{_GK.operator_closed_line(ox)}; shadow X = pnl × min(X / cell_multiplier, 1), sized-below-desired fills re-ticketed at "
         f"min(desired × X / mult, liquidity cap, notional)) · {eq_src}"]
    for r in h.itertuples():
        L.append(f"  - {r.pair} {r.o:%m-%d %H:%M} {'paper' if r.paper else 'live'} · mcap ${r.mcap / 1e6:,.0f}M · mult "
                 f"{r.cell_multiplier} · {r.y:+.2f} % · traded ${r.usd:+.2f} ({-r.loss_pct:+.1f} % of acct) · 1.5× ${r.shadow:+.2f} "
                 f"({-r.loss_pct15:+.1f} %) · 1× ${r.shadow1:+.2f} ({-r.loss_pct1:+.1f} %)" + (f" ({r.szlab})" if r.szlab else "")
                 + "".join(f" · 🔴 {lab} > {ACCT_LOSS_PCT:g} %" for lab, v in (("traded", r.loss_pct), ("1.5×", r.loss_pct15), ("1×", r.loss_pct1))
                           if v > ACCT_LOSS_PCT))
    hv = h[h.usd.notna()] if len(h) else h
    if len(hv):
        for book, pp in (("paper", True), ("live", False)):                 # $ per book, never summed across books
            g = hv[hv.paper == pp]
            if len(g):
                L.append(f"  {book}: N {len(g)} · avg {g.y.mean():+.3f} % · Σ traded ${g.usd.sum():+.2f} vs 1.5× ${g.shadow.sum():+.2f} "
                         f"(Δ ${g.shadow.sum() - g.usd.sum():+.2f}) vs 1× ${g.shadow1.sum():+.2f} (informational)")
        w = hv.loc[hv.usd.idxmin()]
        L.append(f"  largest single loss: {w.pair} {w.o:%m-%d %H:%M} traded ${w.usd:+.2f} ({-w.loss_pct:+.1f} %) → 1.5× ${w.shadow:+.2f} "
                 f"({-w.loss_pct15:+.1f} %) → 1× ${w.shadow1:+.2f} ({-w.loss_pct1:+.1f} %) · > {ACCT_LOSS_PCT:g} %-of-account test: traded "
                 f"{int((hv.loss_pct > ACCT_LOSS_PCT).sum())} · 1.5× {int((hv.loss_pct15 > ACCT_LOSS_PCT).sum())} · 1× "
                 f"{int((hv.loss_pct1 > ACCT_LOSS_PCT).sum())} fill(s)")
    else:
        L.append("  - no high-mcap fade yet")
    L.append(f"  unscored (no mcap stamp): {uns} fade(s) — never counted as low-mcap")
    v, txt = decide_once(state_path, "MCAP_CAP15", *mcap_read(h, pend))
    L.append(f"  Read (confirmation tracker, DL 311 wording — first {MCAP_READ_N}: avg P&L % < 0 → CAP_CANDIDATE, else KEEP_2X): {v} — {txt}")
    return L


def run(now_ms=None, paths=None, since_ms=None, state_path=STATE, mcap_since_ms=None):
    """→ markdown lines (opportunity_scout embeds them; its caller's try catches). since_ms / mcap_since_ms override the floors."""
    L = [f"## 🩸 SPIKE_FADE watch lines — DECISION_LOG {DL}, observe-only (ADX rising · pair ADX ≤ 19.2 · high-mcap 1.5× cap · HEALTHY_BEAR 1× · HB + ADX flat · HB + BTC ADX high)", ""]
    how = "override"
    if since_ms is None:
        since_ms, how = _GK.dl_deploy_ms(DL, lag_min=10, label="SPIKE_FADE watch"), f"DL {DL} commit + 10 min"
    for key, *_ in LINES:
        L.append(f"- {REG[key]}")
    L.append(f"- {REG['MCAP_CAP15']}")
    L.append(f"- ④ {REG['HB_SIZE']}")
    ps = _paths(paths)
    a = load(ps)
    if since_ms is None:
        L.append(f"⚠ Lines ① ② ④ ⑤ ⑥ NOT EVALUABLE — the DECISION_LOG {DL} deploy commit is not in git yet; nothing is counted or persisted for them.")
    else:
        since = pd.Timestamp(int(since_ms), unit="ms")
        fw, ox = forward(a, since)
        pend = pending_opens(a, since)
        L.append(f"Forward cohort ① ② ④ ⑤ ⑥: closed SPIKE_FADE fills opened ≥ {_floor_text(since, how)} · paper + live · {_stat(fw)} · "
                 f"{_GK.operator_closed_line(ox)}")
        L.append("")
        for key, title, col, weak, strong in LINES:
            g = fw.assign(side=side(fw, key)) if len(fw) else fw.assign(side=[])
            L.append(f"**{title}** ({col}, frozen) — weak side = the block candidate:")
            for s_, lab in (("weak", f"weak ({weak})"), ("strong", f"strong ({strong})"), ("unscored", "unscored (stamp missing)")):
                L.append(f"  - {lab}: {_stat(g[g.side == s_] if len(g) else g)}")
            v, txt = decide_once(state_path, key, *filter_read(fw, key, pend))
            L.append(f"  Read (expectancy bar: WR < breakeven ∧ P(avg < 0) ≥ {P_MIN} day-bootstrap ∧ ≥ {GATE_DAYS} days ∧ no day / pair ≥ 50 % "
                     f"of the loss ∧ N ≥ {GATE_N}): {v} — {txt}")
        L += hb_lines(fw, state_path, pend)
    L.append("")
    L += mcap_lines(a, state_path, ps, mcap_since_ms)
    L.append(BLIND)
    return L + [""]


def main():
    print("\n".join(run(int(time.time() * 1000))))


if __name__ == "__main__":
    main()
