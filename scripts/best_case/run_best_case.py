#!/usr/bin/env python3
"""📊 BEST-CASE VIEW (operator 2026-11-03: "how do we name them … to ask again this quick view").

Two tables, in-sample upper bound — every current watch line switched on from the batch start:
  ① each sleeve with its best line ("helps both": never worse on the current batch NOR on master — or, when the line
     changes fewer than 5 master trades, never worse on master + current batch POOLED, DECISION_LOG 501), daily compound before → best
  ② total, all sleeves, with the re-size ideas on top (DECISION_LOG 430), with and without the FAN flip TP +2 exit
     (TP +2's only master gain is the late-June crash window — the "no flip TP+2" columns are the corrected view), and CLEAN ONLY
     (only lines past their checkpoint on new trades, DECISION_LOG 447 — the number without the in-sample cut; DECISION_LOG 453).
     🎲 LUCK-CHECKED (DECISION_LOG 473): every line whose built-on history does not beat the same search on day-shuffled results
     (scripts/best_case/luck_test.py, run inside best_dcr.py → luck.csv) switched off; ③ shows each line's luck test + when its theory was written.

Pipeline (outputs in reports/best_case/): fb_combo.py (FB_CONFIRM per Flatbase fill) → flip_exits.py (each flip on its own
1m path under TP 1.5 / 2 / 3) → best_dcr.py (with flip exits; also best_hold.csv = ①b hold time + trades/day) → best_dcr_noflip.py → confidence.py (③ how much to trust each line). Writes reports/best_case/BEST_CASE_<export stamp>.md.
Every fill is priced on TODAY's exit (master_exits.py, DECISION_LOG 448): master FRENZY_LONG at −4, current-batch LITE C +2 closes at +3,
current-batch Flatbase at +5 / −10. A run STOPS on an unpriced fill or an incomplete market-breadth read (Binance per-minute weight: re-run).
⚠ Batch constants are hard-coded — update ALL of them for a new batch, or the cohort is silently wrong: best_dcr*.py (B0 = batch start,
  FLOWS = top-ups, START = 2,900, the 2,900 → 10,000 equity step date) · fb_combo.py (the previous batch's BASELINE archive file and the
  batch cutoff date) · flip_exits.py (the batch start date).

Usage: venv/bin/python -I scripts/best_case/run_best_case.py ~/Downloads/scalpars_orders_paper_<stamp>.csv [review date, e.g. 2026-11-19]
④ Checkpoint timeline (DECISION_LOG 485): ④a where each line stands by the review date (default confidence.REVIEW_AT) · ④b what each needs to be decided.
"""
import os
import re
import subprocess
import sys

import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(D))
OUT = os.path.join(ROOT, "reports", "best_case")
PY = os.path.join(ROOT, "venv", "bin", "python")
NAMES = {"FRENZY_WILLY": "WILLY", "SPIKE_FADE": "Spike Fade", "FLATBASE_SHORT": "Flatbase", "FLIP:FAN_RATIO_GATE": "FAN flips",
         "FRENZY_LITE_RETRY": "LITE RETRY", "FRENZY_WILLY_HT": "WILLY_HT", "SURGE_LONG": "SURGE Long", "MOMENTUM_LONG": "Momentum Long",
         "FRENZY_LONG": "FRENZY_LONG", "FRENZY_LITE": "LITE", "MOMENTUM_SHORT": "Momentum Short", "BEARRUN_SHORT": "Bear-Run Short"}
LABELS = {"EXIT TP1.5": "Exit TP +1.5", "EXIT TP2": "Exit TP +2", "EXIT TP3": "Exit TP +3", "SURGE_BTC_HOT": "BTC RSI ≥ 78",
          "MKT_HOT": "Market-hot", "FB_CONFIRM": "FB_CONFIRM (live)"}
NUM = r"([-+]?(?:\d+\.?\d*|nan|inf))"
# 📏 (DECISION_LOG 503, operator 11-08) re-size rows = the DL 502 5 % cap (one stop = 5 % of the account): RETRY / Flatbase up (above
# breakeven), LITE down; the old WILLY 8× / 10×, FRENZY_LONG ×5, Flatbase 0.5 examples are retired (Momentum Long ×5 / ×10 back as C3 / C4, DL 504) (WILLY and
# FRENZY_LONG lose below breakeven — sizing them up can only lose more; Flatbase 0.5 broke the 5 % limit)
SCEN = [("TOTAL B", "Watch lines only"), ("C1", "+ 5 % cap sizes: RETRY ↑ · Flatbase ↑ · LITE ↓"), ("C2", "+ 5 % cap, step-ups only: RETRY ↑ · Flatbase ↑"),
        ("C3", "+ C1 + Momentum Long ×5"), ("C4", "+ C3 with Momentum Long ×10")]   # 📏 DL 504


def step(script, exp, *extra):
    r = subprocess.run([PY, "-I", "-W", "ignore", os.path.join(D, script), exp, *extra], capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        sys.exit(f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def totals(txt):
    m = re.search(rf"TOTAL B: B25 one size {NUM} → {NUM} %/day · master intended size {NUM} → {NUM}", txt)
    if not m:
        sys.exit("best_dcr output changed: the 'TOTAL B:' line was not found — update run_best_case.totals()")
    t = {"today": (float(m[1]), float(m[3])), "TOTAL B": (float(m[2]), float(m[4]))}
    for k in [k for k, _ in SCEN if k != "TOTAL B"]:
        c = re.search(rf"^{k} .*?B25 total {NUM} %/day.*master total {NUM}", txt, re.M)
        if not c:
            sys.exit(f"best_dcr output changed: the '{k}' re-size line was not found — update run_best_case.totals()")
        t[k] = (float(c[1]), float(c[2]))
    return t


def clean(txt, tag="CLEAN"):
    """CLEAN-ONLY totals (best_dcr.py): only lines past their checkpoint with the clean 95 % range on their side (DECISION_LOG 447).
    tag LUCK → the LUCK-CHECKED totals (DECISION_LOG 473): every line whose built-on history is NOT ABOVE LUCK switched off."""
    m = re.search(rf"{tag} TOTAL B: B25 one size {NUM} → {NUM} %/day · master intended size {NUM} → {NUM}", txt)
    calls = re.search(rf"^{tag} CALLS: (.*)$", txt, re.M)
    if not m or not calls:
        sys.exit(f"best_dcr output changed: the {tag} lines were not found — update run_best_case.clean()")
    t = {"today": (float(m[1]), float(m[3])), "TOTAL B": (float(m[2]), float(m[4])), "calls": calls[1]}
    for k in [k for k, _ in SCEN if k != "TOTAL B"]:
        c = re.search(rf"^{tag} {k}: B25 total {NUM} %/day \| master total {NUM}", txt, re.M)
        if not c:
            sys.exit(f"best_dcr output changed: the '{tag} {k}' line was not found — update run_best_case.clean()")
        t[k] = (float(c[1]), float(c[2]))
    return t


def f(x):
    return "–" if x != x else f"{x:+.2f}".replace("-", "−")


def dur(m):
    """minutes → '45 min' / '3.2 h' / '–'."""
    if m != m:
        return "–"
    return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"


def caps_section():
    """①c ⏱ time cap on the best line (DECISION_LOG 497) — from best_dcr.py's best_caps.csv; shown only, never in the totals."""
    p = os.path.join(OUT, "best_caps.csv")
    if not os.path.exists(p):
        sys.exit("best_caps.csv missing — best_dcr.py did not write ①c")
    C = pd.read_csv(p)
    L = ["", "## ①c ⏱ Shorter time cap, on top of each FRENZY sleeve's best line", "",
         "Does a shorter cap still pay on the trades the sleeve's best line KEEPS, or only on trades a filter already blocks? Daily compound "
         "%/day: ALL = the cap on every trade · KEPT = best line → best line + cap. Cut = kept trades the cap changes: winners (n, Δ points) "
         "vs losers. Current-batch trades on today's exit use their real path (closed before the cap = unchanged; open = a market close at "
         "the cap, taker + 0.10 slip); master and re-priced trades a paired 1m walk of today's exit (only trades still open at the cap move). "
         "P = day-block bootstrap P(Δ > 0) on the kept trades (current batch + master when read, by close day). ✅ helps both = better on the "
         "current batch AND never worse on master, with at least one master trade changed (a read whose walk does not reproduce ≥ 80 % of "
         "the booked results is not read). 🎲 Luck = the current-batch kept trades (where the cap is chosen), every cap tried vs "
         "day-shuffled results; only a cap that gains is a candidate. Walked stops book the stop line exactly while a cap close pays "
         "taker + 0.10 slip (a small bias against a cap that saves a loser). Not modelled: a capped trade's Δ is booked on its ORIGINAL close "
         "day (a few move a day), and the slot a shorter cap frees is not refilled (WILLY / WILLY_HT hold 1 slot). Luck shows the sleeve's "
         "chosen cap (one test per sleeve). In-sample (30–50 % cut) · shown only, not in ② · a cap is "
         "an exit change: before any ship, the entry × exit joint check + a pre-committed revert gate.", "",
         "| Sleeve | Best line | Cap | Current batch ALL | Current batch KEPT | Master ALL | Master KEPT | Cut: winners · losers (current batch) "
         "| Cut: winners · losers (master) | P | Helps both | 🎲 Luck (sleeve's chosen cap) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in C.itertuples():
        ln = " + ".join(LABELS.get(x, x) for x in r.line.split(" + ")) if isinstance(r.line, str) and r.line else "no line"
        mall = f"{f(r.m_all0)} → {f(r.m_all1)}" if r.m_read == "ok" else r.m_read
        mk = f"{f(r.m_kept0)} → {f(r.m_kept1)}" if r.m_read == "ok" else "–"
        cut = lambda wn, wd, ln_, ld: f"{wn:.0f} ({f(wd)}) · {ln_:.0f} ({f(ld)})"
        cur_all = f"{f(r.b25_all0)} → {f(r.b25_all1)}" if r.b_read == "ok" else r.b_read
        cur_k = f"{f(r.b25_kept0)} → **{f(r.b25_kept1)}**" if r.b_read == "ok" else "–"
        L.append(f"| {NAMES.get(r.sleeve, r.sleeve)} | {ln} | {dur(r.cap)} (now {dur(r.live_cap)}) | {cur_all} | "
                 f"{cur_k} | {mall} | {mk} | {cut(r.b25_w_cut, r.b25_w_d, r.b25_l_cut, r.b25_l_d)} | "
                 f"{cut(r.m_w_cut, r.m_w_d, r.m_l_cut, r.m_l_d) if r.m_read == 'ok' else '–'} | {'–' if r.p_boot != r.p_boot else f'{r.p_boot:.2f}'} | "
                 f"{'✅' if r.helps_both else '—'} | {r.luck} |")
    return L


def late_section():
    """①d ⏬ late TP on the best line (operator 11-07) — from best_dcr.py's best_late.csv; shown only, never in the totals."""
    p = os.path.join(OUT, "best_late.csv")
    if not os.path.exists(p):
        sys.exit("best_late.csv missing — best_dcr.py did not write ①d")
    try:
        C = pd.read_csv(p)
    except pd.errors.EmptyDataError:   # no sleeve on the long FRENZY cap (review)
        C = pd.DataFrame()
    L = ["", "## ①d ⏬ Lower the TP after X minutes, on top of each 12-hour FRENZY sleeve's best line", "",
         "The sleeves on the 12-hour cap (LITE · RETRY · LONG): after X minutes the resting take-profit drops from today's target to the "
         "lower one (maker fill); stop and 12-hour cap unchanged. Same reading as ①c: ALL = every trade · KEPT = best line → best line + "
         "late TP; cut = kept trades it changes, winners (n, Δ points) vs losers (a trade that later hit the full TP gives the difference "
         "back; a trade that later stopped or capped low is rescued). Current-batch trades on today's exit use their real path (the lower "
         "line arms on minute bars that start after X and closed before the real exit — a bar touching it and the stop books the lower of the "
         "two; a real exit after X above the lower TP books the lower TP, since price crossed it on the way); master and re-priced trades a paired 1m walk (a bar "
         "touching both lines = the stop). 🎲 Luck = every X × TP tried for the sleeve on the current-batch kept "
         "trades vs day-shuffled results. In-sample (30–50 % cut) · shown only · an exit change: entry × exit joint check + a pre-committed "
         "revert gate before any ship.", "",
         "| Sleeve | Best line | After | TP → | Current batch ALL | Current batch KEPT | Master ALL | Master KEPT | Cut: winners · losers "
         "(current batch) | Cut: winners · losers (master) | P | Helps both | 🎲 Luck (sleeve's chosen variant) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if not len(C):
        return L + ["", "No FRENZY sleeve on the long cap — nothing to read."]
    for r in C.itertuples():
        if r.after != r.after:   # this sleeve's late-TP pricing has a hole → not read
            L.append(f"| {NAMES.get(r.sleeve, r.sleeve)} | – | – | – | {r.b_read} | – | – | – | – | – | – | – | – |")
            continue
        ln = " + ".join(LABELS.get(x, x) for x in r.line.split(" + ")) if isinstance(r.line, str) and r.line else "no line"
        mall = f"{f(r.m_all0)} → {f(r.m_all1)}" if r.m_read == "ok" else r.m_read
        mk = f"{f(r.m_kept0)} → {f(r.m_kept1)}" if r.m_read == "ok" else "–"
        cut = lambda wn, wd, ln_, ld: f"{wn:.0f} ({f(wd)}) · {ln_:.0f} ({f(ld)})"
        cur_all = f"{f(r.b25_all0)} → {f(r.b25_all1)}" if r.b_read == "ok" else r.b_read
        cur_k = f"{f(r.b25_kept0)} → **{f(r.b25_kept1)}**" if r.b_read == "ok" else "–"
        L.append(f"| {NAMES.get(r.sleeve, r.sleeve)} | {ln} | {dur(r.after)} | +{r.tp:g} → +{r.late_tp:g} | {cur_all} | {cur_k} | {mall} | {mk} | "
                 f"{cut(r.b25_w_cut, r.b25_w_d, r.b25_l_cut, r.b25_l_d)} | "
                 f"{cut(r.m_w_cut, r.m_w_d, r.m_l_cut, r.m_l_d) if r.m_read == 'ok' else '–'} | {'–' if r.p_boot != r.p_boot else f'{r.p_boot:.2f}'} | "
                 f"{f'✅ ({r.b25_w_cut + r.b25_l_cut:.0f} + {r.m_w_cut + r.m_l_cut:.0f} trades)' if r.helps_both else '—'} | {r.luck} |")
    return L


def main(exp, review=None):
    os.makedirs(OUT, exist_ok=True)
    step("fb_combo.py", exp)
    step("flip_exits.py", exp)
    step("frenzy_caps.py", exp, "--fetch")   # ⏱ (DECISION_LOG 497) every FRENZY fill under a shorter time cap → ①c · ⏬ late TP → ①d
    a, b = step("best_dcr.py", exp), step("best_dcr_noflip.py", exp)
    conf = step("confidence.py", exp, review or "")
    conf, _, tline = conf.partition("<!-- ④ checkpoint timeline -->")   # (485) ③ table · ④ checkpoint timeline
    open(os.path.join(OUT, "best_dcr_out.txt"), "w").write(a)
    open(os.path.join(OUT, "best_dcr_noflip_out.txt"), "w").write(b)
    t = pd.read_csv(os.path.join(OUT, "best_dcr.csv"))
    t = t[t.scen == "B"].copy()
    t["gain"] = t.b25_best - t.b25_before
    stamp = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})", os.path.basename(exp))
    stamp = stamp[1] if stamp else "latest"
    L = [f"# 📊 Best-case view — export {stamp}", "",
         "In-sample upper bound: every current watch line switched on from the batch start. Daily compound return, %/day.", "",
         "## ① Each sleeve with its best watch line", "",
         "📏 Daily compound at ONE size — every trade at today's size, current batch and master alike (DECISION_LOG 503: the account path let "
         "the 10-26 top-up and mid-batch size changes flip WILLY's sign). Kept Σ % = the money check (sum of per-trade % on the kept trades); "
         "As traded = the account path, what the account really did — shown, never used to pick a line.", "",
         "| Sleeve | Line switched on | Current batch, before → best | Master, before → best | Kept Σ %, current batch (N) | As traded, current batch |",
         "|---|---|---|---|---|---|"]
    for r in t.sort_values("gain", ascending=False).itertuples():
        nm = NAMES.get(r.sleeve, r.sleeve)
        ln = r.lines if isinstance(r.lines, str) and r.lines else "no line helps"
        ln = " + ".join(LABELS.get(x, x) for x in ln.split(" + "))
        cur = f"{f(r.b25_before)} → **{f(r.b25_best)}**" if ln != "no line helps" else f(r.b25_before)
        mas = "no master trades" if r.m_before != r.m_before else (f"{f(r.m_before)} → {f(r.m_best)}" if ln != "no line helps" else f(r.m_before))
        _ks = (f"{f(r.b25_sum_before)} → {f(r.b25_sum_best)} ({int(r.b25_n_best)})" if ln != "no line helps" else f"{f(r.b25_sum_before)} ({int(r.b25_n_best)})") \
            if hasattr(r, "b25_sum_best") else "–"
        _rt = (f"{f(r.b25_before_real)} → {f(r.b25_best_real)}" if ln != "no line helps" else f(r.b25_before_real)) if hasattr(r, "b25_best_real") else "–"
        L.append(f"| {nm} | {ln}{' ¹' if getattr(r, 'basis', '') == 'pooled' else ''} | {cur} | {mas} | {_ks} | {_rt} |")
    # 🤝 (DECISION_LOG 501, operator 11-08) the pooled rule + the runner-up whenever "best on the current batch" (A) and "helps both" (B) pick different lines
    TA = pd.read_csv(os.path.join(OUT, "best_dcr.csv"))
    TA = TA[TA.scen == "A"].set_index("sleeve")
    if "basis" in t and (t.basis == "pooled").any():
        _mm = re.search(r"master changes only \d+ fill\(s\) < (\d+)", a)
        _pl = " · ".join(f"{NAMES.get(r.sleeve, r.sleeve)}: master {f(r.m_before)} → {f(r.m_best)} on {int(r.m_changed)} trade(s)"
                         for r in t[t.basis == "pooled"].itertuples())
        L += ["", f"¹ Picked on master + current batch POOLED: the line changes fewer than {_mm[1] if _mm else '?'} master trades, so master "
                  f"alone has no veto (DECISION_LOG 501: a handful of master trades must not outvote the current batch) — {_pl}. "
                  "In-sample view only: this rule never feeds a ship or arm decision."]
    alt = []
    for r in t.itertuples():
        if r.sleeve in TA.index:
            ra = TA.loc[r.sleeve]   # (never `a` / `b`: main() keeps best_dcr's printed output in those)
            la, lb = (ra.lines if isinstance(ra.lines, str) else ""), (r.lines if isinstance(r.lines, str) else "")
            if la != lb and la:
                mas = "no master trades" if ra.m_before != ra.m_before else f"{f(ra.m_before)} → {f(ra.m_best)}"
                alt.append(f"| {NAMES.get(r.sleeve, r.sleeve)} | {' + '.join(LABELS.get(x, x) for x in la.split(' + '))} | "
                           f"{f(ra.b25_before)} → {f(ra.b25_best)} | {mas} |")
    if alt:
        L += ["", "Runner-up — the line that is best on the current batch alone, where it differs from ① (it fails ①'s master / pooled check):", "",
              "| Sleeve | Line | Current batch, before → best | Master, before → best |", "|---|---|---|---|"] + alt
    H = pd.read_csv(os.path.join(OUT, "best_hold.csv")).set_index("sleeve")
    L += ["", "## ①b Hold time and trades per day, before → best", "",
          "Same best line as ①. Hold = average time from entry to exit; an exit line holds to its own exit (FAN flips TP +2: first touch of "
          "+2 or the stop on the 1m path), FB_CONFIRM from its confirmed entry. Trades per day: current batch = trades ÷ "
          f"{H.b25_days.iloc[0]:.1f} days since the batch start · master = trades ÷ the calendar days of the batches the sleeve traded in "
          "(its own live days) — the two bases are not directly comparable. Trade counts in brackets. Master hold = as traded: re-priced "
          "master exits (most WILLY, FRENZY_LONG −4, some LITE and fades) and current-batch LITE C re-walks keep their traded hold.", "",
          "| Sleeve | Line switched on | Avg hold, current batch | Avg hold, master | Trades/day, current batch | Trades/day, master |",
          "|---|---|---|---|---|---|"]
    for r in t.sort_values("gain", ascending=False).itertuples():
        h = H.loc[r.sleeve]
        nm = NAMES.get(r.sleeve, r.sleeve)
        ln = r.lines if isinstance(r.lines, str) and r.lines else "no line helps"
        on = ln != "no line helps"
        ln = " + ".join(LABELS.get(x, x) for x in ln.split(" + "))
        def pair(a0, a1):
            return (f"{a0} → **{a1}**" if on and a0 != a1 else a0)
        hc = pair(dur(h.b25_h0), dur(h.b25_h1))
        hm = "no master trades" if not h.m_n0 else pair(dur(h.m_h0), dur(h.m_h1))
        tc_ = pair(f"{h.b25_n0 / h.b25_days:.2f} ({h.b25_n0:.0f})", f"{h.b25_n1 / h.b25_days:.2f} ({h.b25_n1:.0f})")
        tm = "no master trades" if not h.m_n0 else pair(f"{h.m_n0 / h.m_days:.2f} ({h.m_n0:.0f})", f"{h.m_n1 / h.m_days:.2f} ({h.m_n1:.0f})")
        L.append(f"| {nm} | {ln} | {hc} | {hm} | {tc_} | {tm} |")
    L += ["", f"All sleeves, trades/day: current batch {H.b25_n0.sum() / H.b25_days.iloc[0]:.2f} → **{H.b25_n1.sum() / H.b25_days.iloc[0]:.2f}** "
          f"({H.b25_n0.sum():.0f} → {H.b25_n1.sum():.0f} trades). Master per sleeve only (each sleeve traded a different set of batches)."]
    L += caps_section()
    L += late_section()
    ta, tb, tc, tl = totals(a), totals(b), clean(a), clean(a, "LUCK")
    npass = tc["calls"].count(" PASS")
    L += ["", "## ② Total, all sleeves, with re-sizing on top", "",
          "Columns 3–4 leave out the FAN flips TP +2 exit (corrected view: its only master gain is the late-June crash). "
          "CLEAN ONLY = the same scenario with only the lines that passed their checkpoint on NEW trades (95 % range on their side, "
          "DECISION_LOG 447) — no in-sample cut needed; a step UP counts only when its line passed, a step DOWN (the 5 % cap) always counts. It grows toward the full number as lines pass. "
          "LUCK-CHECKED = the same scenario with every line whose built-on history is NOT ABOVE LUCK switched off (the 🎲 luck test, "
          "DECISION_LOG 473: the same search re-run on day-shuffled results) — the others still carry the 30–50 % in-sample cut; a re-size is off "
          "when any of its sleeve's lines is NOT ABOVE LUCK.", "",
          "| Scenario | Current batch | Master | Current batch, no FAN flips TP +2 | Master, no FAN flips TP +2 | **Clean only, current batch** | **Clean only, master** "
          "| **Luck-checked, current batch** | **Luck-checked, master** |",
          "|---|---|---|---|---|---|---|---|---|",
          f"| Today | {f(ta['today'][0])} | {f(ta['today'][1])} | {f(tb['today'][0])} | {f(tb['today'][1])} | {f(tc['today'][0])} | {f(tc['today'][1])} "
          f"| {f(tl['today'][0])} | {f(tl['today'][1])} |"]
    _at = re.search(rf"AS TRADED TODAY: B25 {NUM}", a)
    if _at:   # 📏 (503 deep review) the account path beside the one-size totals — what the account really did
        L.append(f"| Today, as traded (account path) | {f(float(_at[1]))} | – | – | – | – | – | – | – |")
    for k, lab in SCEN:
        L.append(f"| {lab} | {f(ta[k][0])} | {f(ta[k][1])} | {f(tb[k][0])} | {f(tb[k][1])} | {f(tc[k][0])} | {f(tc[k][1])} | {f(tl[k][0])} | {f(tl[k][1])} |")
    _rf = re.search(r"^RESIZE FACTORS \(.*?: (.*)$", a, re.M)
    _rf2 = re.search(r"^RESIZE FACTORS 2 \(DL 504\): (.*)$", a, re.M)
    if _rf:
        L += ["", "Re-size rows (DECISION_LOG 502 / 503): each sleeve sized so ONE stop = 5 % of the account at today's size — " + _rf[1]
              + ". Only sleeves above their breakeven WR step up; WILLY and FRENZY_LONG lose below breakeven, so sizing them up can only lose more "
              "(the old WILLY 8× / 10×, FRENZY_LONG ×5 and Flatbase 0.5 rows are retired). Every factor is all sleeves at one size."]
        if _rf2:
            L += ["", "C3 / C4 (DECISION_LOG 504): " + _rf2[1] + ". Momentum Long is limited by its thin edge (master +0.12 %/trade, 71 % WR vs 61 % "
                  "breakeven) and its clusters (the all-stopped ones were 3 longs — 11-06 at ×10 ≈ 6.8 % of the account), not by the 5 % cap; "
                  "a sleeve re-size (part-reversal of DL 289), not a Pattern W cell. Its step waits for ML_MKT_HOT to PASS (deciding test ≈ Dec 26) and "
                  "leave NOT ABOVE LUCK; 🔒 revert: first 15 ×5 fills WR < 61 % or avg < 0. Spike Fade (×1.4 at its cap), Momentum Short and FAN flips "
                  "are size watch lines only (CURRENT_STATE), not rows here."]
    L += ["", f"Clean-only lines: {npass} of {len(tc['calls'].split(' · '))} passed — {tc['calls']} "
          "(PASS = in · STRADDLE = past checkpoint, range crosses 0 · AGAINST = retire · PRE = before its checkpoint).",
          "", f"🎲 Luck test (built-on trades vs the same search on day-shuffled results; REAL p ≤ 0.05 · UNCLEAR ≤ 0.20 · NOT ABOVE LUCK > 0.20 · "
          f"TOO FEW = cannot be tested): {tl['calls']}."]
    L += ["", "## ③ How much to trust each line", "",
          "Built on = the trades the line was found on (master + current batch before it was frozen: in-sample). Clean = current-batch "
          "trades since it was frozen. The 30–50 % cut applies until the checkpoint; after it, the clean result stands with its 95 % range. "
          "Two trade counts (80 % chance of showing it at 95 % confidence): 'adds money' = the blocked trades really lose (or the change really gains); "
          "'blocked < kept' = the blocked trades really do worse than the ones kept, per side. "
          "The DECIDING TEST is the one the line's own gate uses (DECISION_LOG 484): 'blocked < kept' for the Market-hot lines (their blocked trades "
          "average ≈ 0, so 'adds money' never proves), 'adds money' for the rest. Pace = the line's trades per day over the WHOLE current batch; the "
          "≈ dates project the checkpoint (N trades and D days) and the deciding test from it — zone trades come in bursts, dates move by days. "
          "N to prove it adds money (from the built-on spread, never below the "
          "checkpoint; still optimistic, because in-sample averages are inflated). 95 % range shown from 3 clean days. "
          "Flatbase's built-on = the current batch before the freeze (its history is short). Backtest = a cross-check, never a veto.", "", conf.strip()]
    L += ["", "FAN flips row: its best line is the TP +2 exit; table ② shows the totals with and without it."]
    if tline.strip():   # (DECISION_LOG 485, operator: "add this tables to the best case view, so its only one call")
        L += ["", "## ④ Checkpoint timeline", "", tline.strip()]
    L += ["", "Caveats: in-sample (30–50 % haircut minimum) · a step UP waits for its line's frozen read (RETRY ← RETRY_BTC1H, Flatbase ← the FB_CONFIRM revert gate, Momentum Long ← ML_MKT_HOT); "
          "the LITE step DOWN is a risk cap (operator call at the review); WILLY and FRENZY_LONG are never sized up while below breakeven (DL 503) · "
          "the 5 % factors use the price stop (real stops book a little worse) and assume no Binance bracket / notional cap — RETRY at 7× would "
          "meet the caps that already cut ≈ 35 % of LITE fills, so its step-up is an upper bound · both blocks use today's intended sizes (one size, DL 503), not the sizes traded — ① 'As traded' = the account path.", ""]
    p = os.path.join(OUT, f"BEST_CASE_{stamp}.md")
    open(p, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"saved → {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(os.path.expanduser(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)   # optional review date, e.g. 2026-11-19
