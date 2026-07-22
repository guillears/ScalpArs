"""HARD_TP LADDER helpers (Jul 23, 2026) — pure functions, no dependencies.

Extracted from the realtime engine block so the parse/floor semantics are unit-testable
(review finding: the shipped '11/11 tests' were scratch-only). Used by BOTH the live
ladder exit and the post-exit mechanism shadow, so live and shadow can never drift.

Rung format: "trigger:offset,trigger:offset,..." — a rung locks a profit FLOOR at
(trigger - offset) once the trade's peak pnl crosses the trigger. Floors are monotone
(max over reached rungs). There is NO upper cap: floors only fire on the way DOWN.
"""

DEFAULT_LADDER_RUNGS = [(1.0, 0.25), (1.5, 0.30), (2.0, 0.40), (3.0, 0.60), (4.0, 0.80)]


def parse_hard_tp_ladder(raw):
    """Parse a ladder config string into a sorted list of (trigger, offset) rungs.

    Invalid tokens are skipped (garbage-tolerant: a typo in one rung must not
    disable the others). A rung is valid only if trigger > 0 and 0 < offset < trigger
    (guarantees every floor is strictly positive -> profit-lock only).
    Returns [] for empty/blank/fully-invalid input (= legacy flat-cap mode).
    """
    rungs = []
    for tok in str(raw or '').split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        try:
            t_s, o_s = tok.split(':', 1)
            t, o = float(t_s), float(o_s)
            if t > 0 and 0 < o < t:
                rungs.append((t, o))
        except (ValueError, TypeError):
            continue
    rungs.sort()
    return rungs


def hard_tp_ladder_floor(rungs, peak):
    """Return (floor, level) for the given peak pnl.

    floor = highest (trigger - offset) among rungs whose trigger the peak has
    crossed (monotone — a trade never ratchets below an earned level), or None
    if no rung is reached yet. level = 1-based index of the highest rung reached
    (rungs assumed sorted ascending, as parse_hard_tp_ladder returns them).
    """
    floor = None
    lvl = 0
    for i, (t, o) in enumerate(rungs, 1):
        if peak >= t:
            f = t - o
            if floor is None or f > floor:
                floor = f
            lvl = i
    return floor, lvl
