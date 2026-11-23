"""Route B signal-level loader — the WORKED / REACH entry labels and the path decomposition (weekly-scan PREREG §3, build item 3;
CLAUDE.md entry × exit ①). Pure (no I/O); unit-tested in tests/test_route_b_scan_labels.py.

The 1m path is the one the loader walks for the sleeve's outcome (exits.walk input: the bars from the entry minute, a taken fill's own
minute made pessimistic first), from the same entry price, over the sleeve's max hold:
  WORKED (primary) = the price went +X_PCT (1.0 %) our way BEFORE the sleeve's STOP distance (the same price-% stop the outcome walk
      uses; exit-independent: the TP is ignored). A minute that touches both → AMBIG → 0 (conservative). Neither within the hold → FLAT
      → 0. A path that ends before either and before the hold → unknown (NaN, lbl_final False).
      The FLATBASE stop's funding offset (DL 607) is NOT applied here: the label reads the price path only.
  REACH = the best favourable gross % from the entry up to and including the stop minute (scripts/path_labels.py's convention), else
      over the whole hold; NaN while the path is not finished.
  Decomposition (descriptive, per sleeve): path_labels.finish() on the label, the booked NET %, the best move to the exit + 45 min and
      "recovered" (+X within 120 min after the exit minute) → NEVER_WORKED · STOPPED_RECOVERED · GAVE_BACK · DRIFT_LOSS ·
      LEFT_ON_TABLE · CAPTURED · AMBIG · PENDING. The REAL stop is the −Y passage for the decomposition (path_labels review H3): a walk
      that STOPPED before the label's +X passage (e.g. FLATBASE's funding-offset stop, DL 607) is grouped as NEVER, while lbl_worked
      stays the exit-independent price label. A walk that is not finished (CENSORED) → PENDING.
  Minutes are counted in BARS like exits.walk (the same path); a path whose bars never arrive is closed by the build after a grace.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # scripts/ (path_labels)
import path_labels as PL  # noqa: E402

X_PCT = 1.0
MIN_MS = 60_000


def fav_adv(h, l, entry, side):
    h, l, e = np.asarray(h, float), np.asarray(l, float), float(entry)
    if side > 0:
        return (h / e - 1) * 100, (l / e - 1) * 100
    return (1 - l / e) * 100, (1 - h / e) * 100


def worked_reach(h, l, entry, side, sl_pct, hold_min, x=X_PCT):
    """→ dict(lbl_path, lbl_worked (1 / 0 / NaN), lbl_reach_pct, lbl_pass_min, lbl_final, lbl_ambig). h / l = the bars from the entry
    minute. Pure."""
    out = dict(lbl_path="PENDING", lbl_worked=math.nan, lbl_reach_pct=math.nan, lbl_pass_min=math.nan, lbl_final=False, lbl_ambig=False)
    if not (entry and entry > 0) or not (sl_pct and sl_pct > 0) or hold_min is None or not hold_min > 0:
        out["lbl_path"] = "NO_LEVELS"
        return out
    n = min(len(h), int(hold_min))
    if n <= 0:
        return out
    fav, adv = fav_adv(h[:n], l[:n], entry, side)
    lab, i = PL.first_passage(fav, adv - 1e-9, x, sl_pct)
    complete = n >= int(hold_min)
    if lab == "FLAT" and not complete:
        lab = "PENDING"
    out["lbl_path"] = lab
    out["lbl_ambig"] = lab == "AMBIG"
    out["lbl_worked"] = 1.0 if lab == "WORKED" else (0.0 if lab in ("NEVER", "AMBIG", "FLAT") else math.nan)
    if i is not None and lab in ("WORKED", "NEVER", "AMBIG"):
        out["lbl_pass_min"] = float(i + 1)
    dn = np.flatnonzero(adv <= -sl_pct + 1e-9)
    if len(dn):
        out["lbl_reach_pct"] = float(np.max(fav[:int(dn[0]) + 1]))
        out["lbl_final"] = True
    elif complete:
        out["lbl_reach_pct"] = float(np.max(fav))
        out["lbl_final"] = True
    return out


def decomposition_label(lbl_path, lbl_pass_min, walk_reason, walk_minutes):
    """the label the decomposition reads: the walk's real STOP is the −Y passage when it came before the label's own passage (or the
    label never passed). Pure."""
    if walk_reason != "STOP_LOSS":
        return lbl_path
    if lbl_path in ("WORKED", "AMBIG") and lbl_pass_min == lbl_pass_min and lbl_pass_min <= walk_minutes:
        return lbl_path
    return "NEVER"


def decomposition(h, l, t, entry, side, exit_ms, booked_pct, lbl_path, x=X_PCT, walk_final=True):
    """→ dict(lbl_group, lbl_best_pct, lbl_recovered) via path_labels.finish (best to the exit minute's end + 45 min; recovered = +x in
    the 120 min after the exit minute). PENDING until the bars cover what is needed, or while the walk is not final. Pure."""
    out = dict(lbl_group="PENDING", lbl_best_pct=math.nan, lbl_recovered=None)
    if not walk_final or exit_ms is None or exit_ms != exit_ms or booked_pct is None or booked_pct != booked_pct or not len(t):
        return out
    t = np.asarray(t, np.int64)
    fav, _ = fav_adv(h, l, entry, side)
    tc = int(exit_ms) + MIN_MS - 1                       # the exit minute's end (exits.walk books inside that minute)
    covered = int(t[-1]) + MIN_MS
    best = rec = None
    be = tc + PL.BEST_POST_MIN * MIN_MS
    if covered >= be:
        m = t < be
        best = float(np.max(fav[m])) if m.any() else None
    pe = tc + PL.POST_EXIT_MIN * MIN_MS
    if covered >= pe:
        m = (t > (tc // MIN_MS) * MIN_MS) & (t < pe)
        rec = bool(m.any() and np.max(fav[m]) >= x)
    lab = {"WORKED": "WORKED", "NEVER": "NEVER", "AMBIG": "AMBIG", "FLAT": "FLAT"}.get(lbl_path, "PENDING")
    f = PL.finish(dict(label=lab, best=best, recovered=rec), float(booked_pct))
    out.update(lbl_group=f["group"], lbl_best_pct=(best if best is not None else math.nan), lbl_recovered=rec)
    return out
