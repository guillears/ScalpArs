#!/usr/bin/env python3
"""📋 Batch-review CHECKLIST generator (operator 2026-10-23: "when I send the next batch, pass through ALL items, nothing missing").

Builds reports/BATCH_REVIEW_CHECKLIST.md from the LIVE sources — nothing is keyword-filtered away, so a new scout / gate / plan item / state
line can never be forgotten:
  D.  the standing method steps (save, validate, master, three blocks, era tags, bug gate, separators, logs …)
  A.  the operator's ⭐ priority items in reports/NEXT_BATCH_REVIEW_PLAN.md (① … ②x, ③)
  A2. EVERY other plan section (## / ###) and EVERY "- [ ]" checkbox, tagged with its section (yr6 / two-year sections marked conditional)
  B.  EVERY scripts/scout_*.py — labelled: opportunity scout · via <importer> · standalone · library · parity
  C.  EVERY CLAUDE_CURRENT_STATE.md line that opens an item ("> **", "> N.", "> - **", "- **", "N."), tagged with its section;
      struck / closed / retired lines go to a separate "confirm still closed" bucket
  E.  EVERY scripts/*.py (not scout_*) named in the plan or CURRENT_STATE — with the source line and whether the file exists
  F.  EVERY persisted gate verdict: the keys of reports/SCOUT_REVERT_GATES.json + every reports/SCOUT_*.json / reports/.*gate*.json /
      reports/.scout_*.json state file
  G.  EVERY section of reports/SCOUT_REPORT_latest.md (plan 0b: every Scout line read with a recommendation)
Each row starts ☐; the review ticks every one (☑ result · ⏳ n / bar · ✗ not runnable / obsolete + why). A missing source becomes a visible
"✗ source missing" row, never a crash. No network, no bot API.
Usage: venv/bin/python scripts/batch_review_checklist.py [--batch NAME]
"""
import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "reports", "NEXT_BATCH_REVIEW_PLAN.md")
STATE = os.path.join(ROOT, "CLAUDE_CURRENT_STATE.md")
OPP = os.path.join(ROOT, "scripts", "opportunity_scout.py")
SCOUT_REPORT = os.path.join(ROOT, "reports", "SCOUT_REPORT_latest.md")
REVERT_GATES = os.path.join(ROOT, "reports", "SCOUT_REVERT_GATES.json")
OUT = os.path.join(ROOT, "reports", "BATCH_REVIEW_CHECKLIST.md")
HELPERS = {"scout_gatekit"}                               # pure helpers — not listed at all
LIBRARIES = {"scout_features", "scout_gvol"}              # shared libraries (listed, labelled — no tracker of their own)
PARITY = {"scout_features_parity"}
CONDITIONAL = re.compile(r"yr6|TWO-YEAR|two-year", re.I)
# closed only when the ITEM HEAD says so (text before the first ':' / '—', ≤ 80 chars; case-sensitive, word-bounded) — a later
# "refuted / superseded" in the prose never closes an open item (review 2)
CLOSED = re.compile(r"✅ (?:CLOSED|RESOLVED|DONE)|\bRESOLVED\b|✝|\bRETIRED\b|\bREFUTED\b|\bSUPERSEDED\b|\bVINDICATED\b|OFF FOR GOOD|"
                    r"\bSTRUCK\b|\bREVERTED\b")
LEAD = re.compile(r"^(?:\d+[a-z]?(?:-\d+)?\.\s*|[A-Z]\d?\)\s*|\*\*|[①-⑳]\s*)+")
CLOSED_SCAN = 120   # chars of the item head scanned for a closing word (review 3: '—' no longer cuts the head; late prose is ignored)
# every item-opening shape in CURRENT_STATE (blockquoted or not, indented or not): bullets, numbered (11b-0.), circled ①, bold-led lines
ITEM = re.compile(r"^\s*(?:> )?\s*(?:[-*] |\d+[a-z]?(?:-\d+)?\. |[①-⑳]|(?:\S{1,3} )?\*\*)")
HEADING = re.compile(r"^(?:> )?#{1,4}\s+(.*)")

METHOD = [
    "STEP 0 — data first (operator 2026-10-26), before any sleeve: master WILLY re-priced at the live V2b exit · the batch in the master "
    "(provisional if still running) · size boundaries marked (paper top-up / sizing changes; compare in %) — then the rows below",
    "Save the batch: reports/BASELINE<N>_batch<dates>_<LIVE|PAPER>_orders.csv (bot fills only, MANUAL out) + the decisions journal beside it",
    "Paper = live: paper fills count for every gate (same exits) — state the mode of each fill in every table",
    "Decisions exports cover the whole batch with no ≥ 3-day gap (lite_ob_thin / green_retry / willy_hold need them)",
    "Era tags: every deploy in the batch → its fills tagged by era (deploy + 10 min); normalize across config changes",
    "Run scripts/validate_against_master.py (ALL CHECKS PASS before showing any analysis)",
    "Master pool: archive the batch (scripts/build_master_pool.py); re-run scripts/screen_pool.py if the stack changed",
    "Sleeve ORDER by stake: SPIKE_FADE → WILLY + WILLY_HT → FLATBASE → LITE / FRENZY_LONG → momentum long → momentum short → FAN flips → "
    "Bear-Run / SURGE / rest",
    "ONE SLEEVE AT A TIME (operator 2026-10-26): ① master batch before → after per batch · ② current batch before → after · ③ backtest "
    "before → after · ④ the quant recommendation (a clear call per watchlist item) · ⑤ STOP for the operator's OK before the next sleeve; "
    "new strategies: winners vs losers FIRST, then exits; momentum longs: W/L on the full master",
    "Every table carries the three blocks (current batch · master per batch · backtest), presented master → batch → backtest (the sleeve "
    "order above); before → after per batch; say whether a candidate ADDS or REMOVES a filter",
    "Before → after on TODAY's stack (re-simulate first); in-sample results labelled + 30–50 % haircut; market-wide effects in DAY units; "
    "per-pair / per-day concentration; the backtest refutes, never proves alone",
    "Classify every watchlist item: DECIDED by its frozen gate · OPERATOR CALL below the bar (say so) · KEEP OBSERVING (count + expected date)",
    "After all sleeves: PORTFOLIO view (shared slots, concurrent exposure, worst day, drawdown, sleeve overlap) → one-page decisions list + "
    "go-live criteria; ship ONE change at a time (risk-first; size/filter apart from exits)",
    "Operator-closed (MANUAL / MANUAL_* / EXTERNAL_CLOSE) fills excluded from every gate (scout_gatekit)",
    "Learnings box per sleeve first (plan §0a) · freeze predicted separators before reading (§0a+) · bug gate: two independent derivations "
    "+ an independent reviewer agent on any recommendation (§0c)",
    "Run the full opportunity scout (scripts/opportunity_scout.py) AND every standalone scout; re-run tick-first scouts once the aggTrades "
    "archives publish (≥ 6 h after the day ends)",
    "Any 'no separator' / sleeve-kill claim → scripts/sweep_separators.py output first (CLAUDE.md sleeve-kill checklist)",
    "EVERY sleeve (no exception — DECISION_LOG 398): ① WINNERS vs LOSERS on every variable — scripts/sweep_separators.py <SLEEVE> (real "
    "fills: master on today's stack + current batch) + the Scout 🔎 SEPARATOR SWEEP row (survivors vs shuffled null, blind spots) · "
    "② BEFORE → AFTER per batch (master per batch + current batch, with the daily compound return) for every candidate · ③ backtest "
    "cross-check only — sleeves too thin to sweep are listed as 'keep collecting' with their N",
    "Show BASE stats with and without the washed-out window (Jun-18 → Jul-2) where it applies",
    "'Why no X trades' → grep the server logs FIRST (absence is a logs question)",
    "Flag every read DECIDED or within ~20 % of its bar; list every read that could not run and why",
    "🧭 ENTRY × EXIT (DECISION_LOG 406) — per sleeve, BEFORE any study: the LOSER / WINNER DECOMPOSITION (scripts/path_labels.py · Scout 🧭 "
    "section, master + batch): NEVER_WORKED · STOPPED_RECOVERED · GAVE_BACK · DRIFT_LOSS · LEFT_ON_TABLE · CAPTURED with $ → the larger "
    "realised loss side goes first (ENTRY or EXIT)",
    "🧭 W/L on the ENTRY LABEL from the price path FIRST (sweep_separators.py <SLEEVE> --label path + the Scout ENTRY-LABEL sweep table), "
    "the booked-P&L screen second — never a booked-only W/L",
    "🧭 JOINT CHECK before any ship: an exit candidate re-read on the fills a pending entry candidate keeps (and on all); an entry "
    "candidate judged under the exit that will run; a decision that flips between the reads does not ship; every entry × exit combo "
    "tried counts in the luck test",
    "🧭 STALE LINES: the Scout 🧭 OBSERVATION LINES vs the LIVE EXIT table (scripts/observe_registry.py) — never read a ⚠ STALE line's "
    "verdict; each one → re-base (fresh count) or retire, operator decides",
    "🧮 LEAVE-ONE-OUT under every before → after (review_math.loo_text): the Δ with and without its biggest fill; ≥ 50 % from one fill = "
    "stated, never a ship on it alone",
    "📈 DCR with REAL sizing: each fill's ACTUAL notional (the leverage the exchange bracket really gave), on the account path "
    "(review_math.dcr_account: batch start + flows) — never the intended leverage",
]


def _read(path):
    try:
        return open(path, encoding="utf-8").read()
    except Exception:
        return None


def _clean(t, n=160):
    return re.sub(r"[`|]", "'", re.sub(r"\s+", " ", t)).strip()[:n]


def plan_items(txt=None):
    txt = _read(PLAN) if txt is None else txt
    if txt is None:
        return []
    return [(m.group(1), _clean(m.group(2), 150)) for m in re.finditer(r"^([①-⑳]\w?)\s+\*\*(.+?)\*\*", txt, re.M)]


def plan_sections(txt=None):
    """→ [(kind 'section'/'checkbox', section title, text, conditional)] for every ## / ### heading and every '- [ ]' line."""
    txt = _read(PLAN) if txt is None else txt
    if txt is None:
        return []
    out, sec, prev_head = [], "", False
    for line in txt.split("\n"):
        m = re.match(r"^(#{2,3})\s+(.*)", line)
        if m:
            if prev_head and out and out[-1][0] == "section":     # a heading wrapped over several lines → one row (review 3)
                sec = _clean(out[-1][1] + " " + m.group(2), 120)
                out[-1] = ("section", sec, sec, bool(CONDITIONAL.search(sec)))
            else:
                sec = _clean(m.group(2), 120)
                out.append(("section", sec, sec, bool(CONDITIONAL.search(sec))))
            prev_head = True
            continue
        prev_head = False
        skip = sec.startswith("⭐") or sec.startswith("0.")       # the ⭐ block is A; §0 is folded into D
        c = re.match(r"^\s*- \[[ xX]\]\s+(.*)", line)
        if c and not skip:
            out.append(("checkbox", sec, _clean(c.group(1), 140), bool(CONDITIONAL.search(sec))))
            continue
        b = re.match(r"^\*\*(.+?)\*\*", line)
        if b and not skip:
            out.append(("bold item", sec, _clean(line, 140), bool(CONDITIONAL.search(sec))))
    return [x for x in out if not (x[0] == "section" and (x[1].startswith("⭐") or x[1].startswith("0.")))]


def _importers():
    imp = {}
    for f in glob.glob(os.path.join(ROOT, "scripts", "*.py")):
        src = _read(f) or ""
        me = os.path.basename(f)[:-3]
        for name in set(re.findall(r"\b(?:from|import)\s+(scout_\w+)", src)):
            if name != me:
                imp.setdefault(name, set()).add(me)
    return imp


def scouts():
    imp = _importers()
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "scripts", "scout_*.py"))):
        name = os.path.basename(f)[:-3]
        if name in HELPERS:
            continue
        src = _read(f) or ""
        m = re.search(r'"""(.*?)\n', src, re.S)
        doc = _clean(m.group(1) if m else "", 140)
        by = imp.get(name, set())
        if name in LIBRARIES:
            how = "library (no tracker)"
        elif name in PARITY:
            how = "parity check"
        elif "opportunity_scout" in by:
            how = "opportunity scout"
        elif by:
            how = "via " + ", ".join(sorted(by))
        else:
            how = "standalone"
        rows.append((name, how, doc))
    return rows


def state_items(txt=None):
    """→ [(line no, section, text, closed)] for every item-opening line of CURRENT_STATE (no keyword filter)."""
    txt = _read(STATE) if txt is None else txt
    if txt is None:
        return []
    out, sec = [], "(top)"
    for i, line in enumerate(txt.split("\n"), 1):
        m = HEADING.match(line)
        if m:
            sec = _clean(m.group(1), 60)
            continue
        if ITEM.match(line):
            title = re.sub(r"^\s*(?:> )?\s*(?:[-*] )?", "", line)
            core = LEAD.sub("", title)
            closed = core.startswith("~~") or bool(CLOSED.search(title[:CLOSED_SCAN]))
            out.append((i, sec, _clean(title, 150), closed))
    return out


def state_gates():
    """back-compat: the open (not closed) CURRENT_STATE items as (line, text)."""
    return [(i, t) for i, _, t, c in state_items() if not c]


def referenced_scripts():
    """every scripts/*.py (not scout_*) named in the plan or CURRENT_STATE → [(script, exists, first source)]."""
    refs = {}
    for label, path in (("plan", PLAN), ("state", STATE)):
        txt = _read(path)
        if txt is None:
            continue
        for i, line in enumerate(txt.split("\n"), 1):
            names = set(re.findall(r"scripts/([A-Za-z0-9_]+\.py)", line))
            names |= {b for b in re.findall(r"\b([a-z0-9_]+\.py)\b", line) if os.path.exists(os.path.join(ROOT, "scripts", b))}
            for s in sorted(names):
                if s.startswith("scout_"):
                    continue
                refs.setdefault(s, f"{label}:L{i}")
    return [(s, os.path.exists(os.path.join(ROOT, "scripts", s)), src) for s, src in sorted(refs.items())]


def gate_states():
    """→ [(file, key or '')] — every persisted gate verdict / state file."""
    out = []
    try:
        with open(REVERT_GATES, encoding="utf-8") as fh:
            st = json.load(fh)
        g = st.get("gates", {}) if isinstance(st, dict) else {}   # (review 2) the verdicts are nested under "gates"
        for k in sorted(g):
            v = g[k] if isinstance(g[k], dict) else {}
            stt = v.get("state") or v.get("verdict") or ""
            out.append(("SCOUT_REVERT_GATES.json", _clean(f"gate {k}" + (f" · {stt}" if stt else "") + f" · {v.get('progress', '')}", 160)))
    except Exception:
        out.append(("SCOUT_REVERT_GATES.json", "✗ source missing / unreadable"))
    pats = ["SCOUT_*.json", ".*gate*.json", ".scout_*.json"]
    files = sorted({os.path.basename(f) for p in pats for f in glob.glob(os.path.join(ROOT, "reports", p))} - {"SCOUT_REVERT_GATES.json"})
    out += [(f, "") for f in files]
    return out


def scout_report_sections():
    txt = _read(SCOUT_REPORT)
    if txt is None:
        return None
    return [_clean(m.group(1), 140) for m in re.finditer(r"^#{2,3}\s+(.*)", txt, re.M)]


def build(batch):
    L = [f"# 📋 Batch review checklist — {batch}", "",
         "Generated by scripts/batch_review_checklist.py from the live sources (plan · scouts · CURRENT_STATE · referenced scripts · gate "
         "states · scout report). Tick EVERY row: ☑ result · ⏳ collecting (n / bar) · ✗ not runnable / obsolete (why). Nothing is "
         "skipped silently. Rows are cut at 150 chars — open the source line (L<n>) for the full bar. Order: method → priorities → live "
         "gate states + scout report → CURRENT_STATE → plan sections → scripts → closed items.", ""]
    for path in (PLAN, STATE, OPP, SCOUT_REPORT):
        if _read(path) is None:
            L.append(f"- ✗ source missing: {os.path.relpath(path, ROOT)}")
    L += ["## D. Method", ""] + [f"- ☐ {m}" for m in METHOD]
    pi = plan_items()
    L += ["", "## A. ⭐ Operator priority items (NEXT_BATCH_REVIEW_PLAN.md)", ""] + [f"- ☐ {k} {t}" for k, t in pi]
    gs = gate_states()
    L += ["", f"## F. Every persisted gate verdict / state file ({len(gs)})", "", "| ☐ | file | gate key |", "|---|---|---|"]
    L += [f"| ☐ | {f} | {k} |" for f, k in gs]
    sr = scout_report_sections()
    L += ["", "## G. Every section of the latest scout report (plan 0b: read each with a recommendation)", ""]
    L += [f"- ✗ source missing: {os.path.relpath(SCOUT_REPORT, ROOT)} (run scripts/opportunity_scout.py first)"] if sr is None else \
        [f"- ☐ {s}" for s in sr]
    st = state_items()
    op = [x for x in st if not x[3]]
    cl = [x for x in st if x[3]]
    L += ["", f"## C. Every CURRENT_STATE item — open ({len(op)})", "", "| ☐ | line | section | item |", "|---|---|---|---|"]
    L += [f"| ☐ | {i} | {s} | {t} |" for i, s, t, _ in op]
    ps = plan_sections()
    L += ["", f"## A2. Every other plan section + checkbox ({len(ps)})", "", "| ☐ | kind | section | item |", "|---|---|---|---|"]
    L += [f"| ☐ | {k}{' (conditional — check if yr6 is done)' if c else ''} | {s} | {t if k != 'section' else ''} |" for k, s, t, c in ps]
    sc = scouts()
    L += ["", f"## B. Every scout script ({len(sc)})", "", "| ☐ | scout | runs from | what it tracks |", "|---|---|---|---|"]
    L += [f"| ☐ | {n} | {w} | {d} |" for n, w, d in sc]
    rs = referenced_scripts()
    L += ["", f"## E. Every other script named in the plan / CURRENT_STATE ({len(rs)})", "", "| ☐ | script | exists | first named at |",
          "|---|---|---|---|"]
    L += [f"| ☐ | {s} | {'yes' if e else '✗ missing'} | {src} |" for s, e, src in rs]
    L += ["", f"## C2. CURRENT_STATE items marked closed / retired — confirm still closed ({len(cl)})", "", "| ☐ | line | section | item |",
          "|---|---|---|---|"]
    L += [f"| ☐ | {i} | {s} | {t} |" for i, s, t, _ in cl]
    return "\n".join(L) + "\n", dict(method=len(METHOD), plan=len(pi), plan_sections=len(ps), scouts=len(sc), state_open=len(op),
                                      state_closed=len(cl), scripts=len(rs), gate_states=len(gs), report_sections=len(sr or []))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="(next batch)")
    a = ap.parse_args(argv)
    txt, n = build(a.batch)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(f"{OUT}: " + " · ".join(f"{k} {v}" for k, v in n.items()))


if __name__ == "__main__":
    main()
