#!/usr/bin/env python3
"""LITE THEORY — writes the pre-registration (reports/LITE_THEORY_PREREG.txt). Refuses to overwrite."""
import os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_lite_theory_common as T  # noqa: E402

if os.path.exists(T.PREREG):
    sys.exit(f"REFUSED: {T.PREREG} exists (pre-registration is frozen)")

FROZEN = [T.BARS, os.path.join(T.OUT, "signals_today.pkl"), os.path.join(T.OUT, "parity_rows.csv"), T.FILLS, T.COHORT,
          os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-15_19-55-15.csv")]
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
hashes = "\n".join(f"   {T.sha(p)}  {os.path.relpath(p, T.ROOT) if p.startswith(T.ROOT) else p}" for p in FROZEN)

TXT = f"""LITE THEORY STUDY — PRE-REGISTRATION (frozen {now}, BEFORE any bar / signal / variant was priced)
Operator 2026-10-15: "go with theory — what is a LITE trade — and from there start testing everything in detail."
Today's LITE: verified FRENZY flag · 2 h ≤ hours ≤ 16.75 · above_streak ≥ 12 · vol_mult < 100 · not in FRENZY state · 24 h volume ≥ $20M ·
the stretch's FIRST signal bar only, entered only if RED · exit +3 resting maker TP / −3 stop / 720 min, 10×.
Theory under test: after a pump, buyers defend the spike-anchored VWAP for an hour → the pump is "accepted" → a pullback buy catches the next leg.
Code: scripts/study_lite_theory_*.py. Research only: no engine / config / trading_config.json / template / test edit, no bot API, no commit.
Cache-only (no Binance call planned). Not reviewed → supports no arm / ship by itself.

DISCLOSURE (read before this freeze; no outcome of any cell below): the summaries of LITE_WL_FULLSWEEP (no separator in 187 vars) and the
services/frenzy.py header comment (DECISION_LOG 243: retrying later bars of a stretch turned +0.163 %/trade into −0.005 on 1,992 fills;
6-vs-12 closes study exists, its tables NOT read here). Looked at outcome-free: the rebuild parity table (fills reproduced, stamps), the
replay fills' entry stamps (above_streak is 12 on ≥ 75 % of fills), the 4 live fills' entry stamps and outcomes (DEEP TP, ZK / DUSK / EPIC stops).

1. FROZEN INPUTS (sha256)
{hashes}

2. EPISODES / SIGNALS (built before this freeze, outcome-free)
   scripts/study_lite_theory_episodes.py → reports/lite_theory/bars.pkl: every 5m bar ≤ 30 h after a (loosely screened) spike on the 398
   cohort pairs (2025-01-01 → last cached 5m bar), judged by services.frenzy.frenzy_walk on the 1,499 closed bars ending there with
   services.frenzy.normal_hour_usd (743 completed 1h bars, refreshed each hour; engine: every 6–8 h — declared). Kept: flagged (verified,
   ≤ 96 h) bars ≤ 30 h old. 1,852,059 bars · 6,841 episodes.
   scripts/study_lite_theory_signals.py: emulator of frenzy_lite_status (asserted equal to the engine function on every bar for today's
   params) + bearish-day (frenzy_bearish_block on BTC last-closed-day return + BTC 5m EMA13−EMA50 gap, rebuilt).
   PARITY: replay LITE fills (yr6 2Y_TODAY seed 1) reproduced by a rebuilt READY signal on the same pair & bar: 1,405 / 1,454 = 96.6 %
   (2025 97.9 %, 2026 94.9 %) ≥ 85 % gate → PASS. Rebuilt READY signals 2,765, 2,106 not on a bearish day (the replay also refuses slots,
   pair-day cap, WILLY hold, chunk warm-up — signal-level study ignores those; declared).

3. UNIVERSE + RULERS
   U = rebuilt bars NOT on a bearish day (the engine refuses LITE there for every variant). Signal unit = one entry at the signal bar's close.
   ENTRY: te = bar close + 12 s; E = open of the first 1m bar opening ≥ te, × (1 + 0.036 %) (taker slip T1); same rule for every set.
   EXIT (today): TP +3 net as a RESTING maker limit (plim = ceil(E·r(3, taker .045, maker .018)/tick)·tick, filled by a print ≥ plim + ½ tick,
   booked at the maker net) · stop −3 net (tick: the print through; 1m bar: min(line, bar open); a bar touching both = stop) · cap 720 min at
   the last close. LS ruler (PRIMARY, the exit-opt convention): −0.20 on every non-TP exit (stop, cap, VWAP exit). UNRES (data ends) excluded.
   R1 = 1m-bar ruler (all bars; used for A0 / A1 / distribution tables so every set is priced on the SAME source).
   RH = hybrid tick ruler (FX.load_path: ticks while the days are cached, then 1m) — every signal set (today + every variant cell) and every
   A6 / A7 cell. GATES are read on RH (R1 shown as sensitivity). Walker parity: on the replay fills with the replay's own E, RH reproduces the
   LITE_PROTECT rm_base_pct (asserted, |Δ| < 1e-9) before anything else is read.
   Blocks: 2025 · 2026 (H1 / H2 shown) · MASTER (20 paper fills: filter cells via their entry stamps + cached tick paths; replacement cells via
   the rebuilt signals inside the master window 2026-10-06 → last 5m bar — labelled REBUILT) · LIVE 4 (stamps; listed, never gated).

4. ASSUMPTIONS / CELLS (every cell counted; nothing added after pricing without an addendum)
   A0 DIAGNOSTIC (not gated): within the SAME episodes that hold a today-LITE signal, window 2–16.75 h, not in state, vol < 100, U:
     paired per signal: Y(LITE) − mean Y over that episode's (i) every bar · (ii) every bar with close ≥ VWAP (streak ≥ 1) · (iii) every RED bar
     with close ≥ VWAP. Day-clustered bootstrap. Also unpaired means over ALL flagged episodes (R1). "No edge" = LITE not above (i)–(iii).
   A1 DIAGNOSTIC: over U, window 2–16.75 h, not in state, vol < 100: mean Y, P(TP before stop), forward close return +60 / +240 / +720 min by
     above_streak bucket {{0, 1–5, 6–11, 12–23, 24–35, 36+}} and by above_share bucket {{< 50, 50–70, 70–85, ≥ 85 %}}; per year.
   A2 (replacement, emulator): A2_GREEN (enter the first signal bar only if GREEN) · A2_ANY (enter it whatever its colour) · A2_WAITRED
     (today + when the first signal bar is green, enter the next RED bar of the same stretch within 6 bars, need / window still met) ·
     stretch ordinal filters on today's signals: A2_ORD1 (keep only the episode's 1st LITE-judged stretch) · A2_ORD2P (keep only 2nd +).
   A3 (replacement): need ∈ {{6, 24, 36}} (A3_N6 / A3_N24 / A3_N36), everything else today.
   A4: window cells (emulator): A4_H2_8, A4_H2_12, A4_H4_1675, A4_H8_1675, A4_H2_24 ; bucket table 2–4 / 4–8 / 8–12 / 12–16.75 / 16.75–24.
   A5: vmax cells A5_V20 (vol < 20), A5_V50 (vol < 50); filter A5_DROPLT20 (drop vol < 20); bucket table < 20 / 20–50 / 50–100.
   A6: stretch filters on today's signals (vs_vwap = entry_frenzy_vs_vwap_pct): A6_CAP2 / A6_CAP3 / A6_CAP5 (skip vs_vwap > 2 / 3 / 5 %) ·
     A6_FLOOR1 (skip < 1 %); bucket table 0–1 / 1–2 / 2–3 / 3–5 / 5+.
     LIMIT entry A6_L{{k}}_W{{w}}: k ∈ {{0.5, 1.0, 1.5}}, w ∈ {{30, 60, 120}} min (9 cells): limit at VWAP_signal × (1 + k %); if the market E ≤
     the limit → today's market entry; else a resting maker BUY filled by a print ≤ limit − ½ tick within w minutes (E = limit, entry fee
     maker .018); unfilled → no trade (0). Exit today's from the fill.
   A7 (exit, two-sided on today's signals): A7_C5_b{{0, 0.5, 1}} = + exit at the first 5m CLOSE < VWAP × (1 − b) (VWAP = the spike-anchored
     VWAP of that bar, rebuilt) — market at the next print, −0.20 · A7_C1_b{{0, 0.5, 1}} = same on 1m closes vs the last closed 5m VWAP ·
     A7_R5 = 5m-close b=0 exit REPLACING the −3 stop (disaster stop −10) · A7_VS_b{{0, 0.5, 1}} = stop placed AT VWAP × (1 − b) at entry,
     replacing −3 (net distance clipped to [0.5, 10] %). TP +3 resting and 720 min stay. (10 cells)
   A8 filters ("drop bucket"): spike run_pct {{< 10, 10–20, ≥ 20}} · spike volume multiple spike_volx {{< 50, 50–100, ≥ 100}} · distance
     from the spike-bar high at entry (close / spike high − 1) {{≥ −3, −3…−8, < −8 %}} · prior LITE-judged stretches in the episode {{0, 1, ≥ 2}}
     (12 cells, A8_<feat>_<bucket>).
   A9 filters: A9_NEWHI (keep only signals where a high > the spike-bar high printed after the spike, up to the signal bar) · A9_NONEWHI.
   TOTAL gated cells: A2 5 + A3 3 + A4 5 + A5 3 + A6 4 + 9 + A7 10 + A8 12 + A9 2 = 53.

5. GATES (a cell is a CANDIDATE only if ALL hold; RH ruler, LS slip)
   Δ = Σ Y(variant signals) − Σ Y(today signals), per day (filters: Δ = −Σ Y of the dropped signals; exits / limit: per-signal paired).
   ① day-clustered bootstrap P(Δ > 0) ≥ 0.90 in 2025 AND in 2026 separately · ② MASTER Δ ≥ 0 (n/a declared where untestable) ·
   ③ leave-one-month-out sign-stable (each year pooled) · ④ no single day and no single pair ≥ 50 % of ΣΔ · ⑤ neighbours (adjacent
   parameter / bucket cells of the same family) Δ > 0 · ⑥ projected Δ shown after a 30–50 % in-sample haircut.
   Filters additionally state the CLAUDE.md expectancy read of the dropped cohort (1×: WR vs breakeven, day-bootstrap P(avg < 0)).
   Multiple testing: 53 cells → at P ≥ 0.90 per year, ~0.5 cells pass both years by luck (independent approx.); a pass is a SCREEN.
   Outcome of a pass: observe-only Scout line, or — only if strong in EVERY block — stated plainly with its revert gate. A0 / A1 reported
   plainly whatever they say.
"""
open(T.PREREG, "w").write(TXT)
print(f"written {T.PREREG}")
