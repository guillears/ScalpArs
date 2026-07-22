"""HARD_TP LADDER unit tests (Jul 23, 2026 — review finding: the shipped '11/11 tests'
were scratch-only; these check the exact semantics in as importable code).

Covers parse_hard_tp_ladder (garbage tolerance, validation, sorting, empty->flat-mode)
and hard_tp_ladder_floor (monotone floors, MIRA no-fire on the way up, DEXE-class
collapse capture, per-side L1 values, sub-first-rung inertness).

Run with:
    ./venv/bin/python tests/test_hard_tp_ladder.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.hard_tp_ladder import parse_hard_tp_ladder, hard_tp_ladder_floor, DEFAULT_LADDER_RUNGS

LONG_LADDER = "1.25:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80"
SHORT_LADDER = "1.0:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80"


def fire(ladder_raw, peak, pnl):
    """Replicates the live engine decision: (reason, floor)."""
    rungs = parse_hard_tp_ladder(ladder_raw)
    if not rungs:
        return ("FLAT-MODE", None)
    floor, lvl = hard_tp_ladder_floor(rungs, peak)
    if floor is not None and pnl <= floor:
        return (f"HARD_TP_LADDER L{lvl}", floor)
    return (None, floor)


CASES = [
    # (name, ladder, peak, pnl, expected_reason, expected_floor)
    ("L sub-arm: peak 1.1, pnl 0.3 -> no fire (no rung reached)", LONG_LADDER, 1.1, 0.3, None, None),
    ("L wick: peak 1.3 falls to 1.0 -> fire L1 floor 1.00", LONG_LADDER, 1.3, 1.0, "HARD_TP_LADDER L1", 1.0),
    ("L runner: peak 1.9, pnl 1.5 -> no fire (floor 1.20)", LONG_LADDER, 1.9, 1.5, None, 1.2),
    ("L runner collapse: peak 1.9 falls to 1.2 -> fire L2", LONG_LADDER, 1.9, 1.2, "HARD_TP_LADDER L2", 1.2),
    ("L DEXE: peak 3.64 falls to 2.4 -> fire L4 floor 2.40", LONG_LADDER, 3.64, 2.4, "HARD_TP_LADDER L4", 2.4),
    ("L MIRA: peak 19, pnl 18 -> NO fire (no cap; floor 3.20)", LONG_LADDER, 19.0, 18.0, None, 3.2),
    ("L MIRA full collapse: peak 19 falls to 3.2 -> fire L5", LONG_LADDER, 19.0, 3.2, "HARD_TP_LADDER L5", 3.2),
    ("S wick: peak 1.05 falls to 0.75 -> fire L1 floor 0.75", SHORT_LADDER, 1.05, 0.75, "HARD_TP_LADDER L1", 0.75),
    ("empty ladder -> flat mode", "", 1.5, 1.5, "FLAT-MODE", None),
    ("all-garbage tokens -> flat mode", "abc,1.0", 1.5, 0.1, "FLAT-MODE", None),
    ("partial garbage keeps valid rung", "abc:x,1.0:0.25", 1.1, 0.7, "HARD_TP_LADDER L1", 0.75),
    ("invalid rung offset>=trigger rejected", "1.0:1.0,2.0:0.4", 2.5, 1.5, "HARD_TP_LADDER L1", 1.6),
    ("unsorted input is sorted (levels 1-based ascending)", "2.0:0.40,1.25:0.25", 2.1, 1.55, "HARD_TP_LADDER L2", 1.6),
    ("negative pnl never fires below first rung", LONG_LADDER, 1.0, -0.5, None, None),
    ("exact floor touch fires", SHORT_LADDER, 1.5, 1.2, "HARD_TP_LADDER L2", 1.2),
]


def main():
    failures = 0
    for name, lad, pk, pnl, exp_r, exp_f in CASES:
        r, fl = fire(lad, pk, pnl)
        ok = (r == exp_r) and (fl == exp_f or (fl is not None and exp_f is not None and abs(fl - exp_f) < 1e-9))
        print(f"{'OK  ' if ok else 'FAIL'} {name:58s} -> {r} (floor={fl})")
        if not ok:
            failures += 1
    # parse invariants
    assert parse_hard_tp_ladder(None) == []
    assert parse_hard_tp_ladder("  ") == []
    assert parse_hard_tp_ladder("1.0:0.25") == [(1.0, 0.25)]
    assert all(0 < o < t for t, o in DEFAULT_LADDER_RUNGS)
    # floors strictly positive for any valid parse
    for t, o in parse_hard_tp_ladder(LONG_LADDER):
        assert t - o > 0
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURES'} ({len(CASES)} cases + parse invariants)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
