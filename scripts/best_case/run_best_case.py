#!/usr/bin/env python3
"""📊 BEST-CASE VIEW (operator 2026-11-03: "how do we name them … to ask again this quick view").

Two tables, in-sample upper bound — every current watch line switched on from the batch start:
  ① each sleeve with its best line ("helps both": never worse on the current batch NOR on master), daily compound before → best
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

Usage: venv/bin/python -I scripts/best_case/run_best_case.py ~/Downloads/scalpars_orders_paper_<stamp>.csv
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
SCEN = [("TOTAL B", "Watch lines only"), ("C1", "+ WILLY at 8×"), ("C2", "+ WILLY at 10×"),
        ("C3", "+ WILLY 10×, FRENZY_LONG ×5, Flatbase 0.5"), ("C4", "+ WILLY 8×, FRENZY_LONG ×5, Flatbase 0.5"),
        ("C5", "+ WILLY 10×, FRENZY_LONG ×5, Flatbase 0.5, Momentum Long ×5"), ("C6", "+ WILLY 10×, FRENZY_LONG ×5, Flatbase 0.5, Momentum Long ×10")]


def step(script, exp):
    r = subprocess.run([PY, "-I", "-W", "ignore", os.path.join(D, script), exp], capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        sys.exit(f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def totals(txt):
    m = re.search(rf"TOTAL B: B25 real path {NUM} → {NUM} %/day · master intended size {NUM} → {NUM}", txt)
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
    m = re.search(rf"{tag} TOTAL B: B25 real path {NUM} → {NUM} %/day · master intended size {NUM} → {NUM}", txt)
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


def main(exp):
    os.makedirs(OUT, exist_ok=True)
    step("fb_combo.py", exp)
    step("flip_exits.py", exp)
    a, b = step("best_dcr.py", exp), step("best_dcr_noflip.py", exp)
    conf = step("confidence.py", exp)
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
         "| Sleeve | Line switched on | Current batch, before → best | Master, before → best |", "|---|---|---|---|"]
    for r in t.sort_values("gain", ascending=False).itertuples():
        nm = NAMES.get(r.sleeve, r.sleeve)
        ln = r.lines if isinstance(r.lines, str) and r.lines else "no line helps"
        ln = " + ".join(LABELS.get(x, x) for x in ln.split(" + "))
        cur = f"{f(r.b25_before)} → **{f(r.b25_best)}**" if ln != "no line helps" else f(r.b25_before)
        mas = "no master trades" if r.m_before != r.m_before else (f"{f(r.m_before)} → {f(r.m_best)}" if ln != "no line helps" else f(r.m_before))
        L.append(f"| {nm} | {ln} | {cur} | {mas} |")
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
    ta, tb, tc, tl = totals(a), totals(b), clean(a), clean(a, "LUCK")
    npass = tc["calls"].count(" PASS")
    L += ["", "## ② Total, all sleeves, with re-sizing on top", "",
          "Columns 3–4 leave out the FAN flips TP +2 exit (corrected view: its only master gain is the late-June crash). "
          "CLEAN ONLY = the same scenario with only the lines that passed their checkpoint on NEW trades (95 % range on their side, "
          "DECISION_LOG 447) — no in-sample cut needed; a re-size counts only when its line passed. It grows toward the full number as lines pass. "
          "LUCK-CHECKED = the same scenario with every line whose built-on history is NOT ABOVE LUCK switched off (the 🎲 luck test, "
          "DECISION_LOG 473: the same search re-run on day-shuffled results) — the others still carry the 30–50 % in-sample cut; a re-size is off "
          "when any of its sleeve's lines is NOT ABOVE LUCK.", "",
          "| Scenario | Current batch | Master | Current batch, no FAN flips TP +2 | Master, no FAN flips TP +2 | **Clean only, current batch** | **Clean only, master** "
          "| **Luck-checked, current batch** | **Luck-checked, master** |",
          "|---|---|---|---|---|---|---|---|---|",
          f"| Today | {f(ta['today'][0])} | {f(ta['today'][1])} | {f(tb['today'][0])} | {f(tb['today'][1])} | {f(tc['today'][0])} | {f(tc['today'][1])} "
          f"| {f(tl['today'][0])} | {f(tl['today'][1])} |"]
    for k, lab in SCEN:
        L.append(f"| {lab} | {f(ta[k][0])} | {f(ta[k][1])} | {f(tb[k][0])} | {f(tb[k][1])} | {f(tc[k][0])} | {f(tc[k][1])} | {f(tl[k][0])} | {f(tl[k][1])} |")
    L += ["", f"Clean-only lines: {npass} of {len(tc['calls'].split(' · '))} passed — {tc['calls']} "
          "(PASS = in · STRADDLE = past checkpoint, range crosses 0 · AGAINST = retire · PRE = before its checkpoint).",
          "", f"🎲 Luck test (built-on trades vs the same search on day-shuffled results; REAL p ≤ 0.05 · UNCLEAR ≤ 0.20 · NOT ABOVE LUCK > 0.20 · "
          f"TOO FEW = cannot be tested): {tl['calls']}."]
    L += ["", "## ③ How much to trust each line", "",
          "Built on = the trades the line was found on (master + current batch before it was frozen: in-sample). Clean = current-batch "
          "trades since it was frozen. The 30–50 % cut applies until the checkpoint; after it, the clean result stands with its 95 % range. "
          "Two trade counts (80 % chance of showing it at 95 % confidence): 'adds money' = the blocked trades really lose (or the change really gains); "
          "'blocked < kept' = the blocked trades really do worse than the ones kept, per side. "
          "N to prove it adds money (from the built-on spread, never below the "
          "checkpoint; still optimistic, because in-sample averages are inflated). 95 % range shown from 3 clean days. "
          "Flatbase's built-on = the current batch before the freeze (its history is short). Backtest = a cross-check, never a veto.", "", conf.strip()]
    L += ["", "FAN flips row: its best line is the TP +2 exit; table ② shows the totals with and without it."]
    L += ["", "Caveats: in-sample (30–50 % haircut minimum) · each re-size waits for its line's frozen read (WILLY_GAP 15 fills, "
          "Flatbase revert gate, FRENZY_LONG DL 317, Momentum Long ML_MKT_HOT 15 fills on 8 days and a clean edge above 0 — DL 450) · master uses today's intended sizes, not the sizes traded.", ""]
    p = os.path.join(OUT, f"BEST_CASE_{stamp}.md")
    open(p, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"saved → {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(os.path.expanduser(sys.argv[1]))
