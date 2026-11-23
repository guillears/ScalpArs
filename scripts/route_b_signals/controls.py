"""Route B signal-level loader — the MATCHED RANDOM-MINUTE CONTROLS of the weekly-scan PREREG §6 (Q3 null; build item 5).

For every trigger opener (every key, parents and "|c" variants): N_CTRL (5) control minutes on the same coin, the same UTC day and the
same side, drawn without replacement with a generator seeded by (SEED 20261210, crc32(trigger_id)) — the same draw on every run once the
eligible set is final. Eligible control minute c (a trigger MINUTE, like the trigger's own m):
  · the coin's 20-level book was recorded at c (a book row stamped c — the same cost source the triggers use: the $6k walk at the state);
  · c is not within ± EXCL_MIN (60) minutes of ANY hit minute of the same family (triggers.family: a parent and its "|c" variant share it);
  · c is in the same UTC day as the trigger minute and ≥ C.REC_START.
A control is priced exactly like a trigger (build.price_h: entry at the open of minute c + 2, the TP / SL grid, the recorded cost or
the default slip, funding). The DRAW is final (ctrl_final, day_status) only when every hourly summary file from D−1 23:00 to D+1 01:00
LOADED this run, or is permanently absent (still missing 48 h after the hour ended — counted in ctrl_day_missing_hours); hours before
the recording start count as present. Before that it is redrawn on every run (deterministic: the same eligible set gives the same
draw). ctrl_final is about the draw only — the priced outcome has its own h_final; Q3 reads ctrl_final rows only.
A control needs a 20-level book row stamped exactly at c; a trigger's state accepts the book row ≤ m up to 2 min old (the cost
source can differ slightly; the Q3 code drops default-cost triggers together with their controls). Fewer than N_CTRL eligible minutes → all of them; ZERO eligible → one stub row
(ctrl_k −1, minute NaN, n_eligible 0) so a trigger never silently leaves the matched set (reported, excluded from Δ by construction).
Draws of two openers of the same coin-day and family are independent (the same minute can be drawn twice; it then counts twice).
Controls are NOT depth-conditioned for the "|c" variants (the null compares the variant to the coin-day's random minutes).
The generator: numpy default_rng([SEED, crc32(trigger_id)]).choice — pinned by a golden draw in the tests (a numpy upgrade that changes
the stream fails the suite instead of silently redrawing). Pure here (no I/O); the build prices the rows.
"""
import zlib

import numpy as np
import pandas as pd

MIN_MS = 60_000
DAY_MS = 1440 * MIN_MS
SEED = 20261210
N_CTRL = 5
EXCL_MIN = 60


def rng_for(trigger_id):
    return np.random.default_rng([SEED, zlib.crc32(str(trigger_id).encode())])


def family_hits(hits, family_of):
    """{(family, sym): sorted int64 array of hit minutes} from {key: [(sym, minute)]}. Pure."""
    acc = {}
    for key, lst in hits.items():
        f = family_of(key)
        for s, m in lst:
            acc.setdefault((f, s), []).append(int(m))
    return {k: np.unique(np.asarray(v, np.int64)) for k, v in acc.items()}


def eligible(book_minutes, day_start, fam_hits, rec_start_ms):
    """eligible control minutes of one coin-day. book_minutes = the coin's sorted recorded book minutes (int64). Pure."""
    bm = np.asarray(book_minutes, np.int64)
    c = bm[(bm >= max(day_start, rec_start_ms)) & (bm < day_start + DAY_MS)]
    if len(c) and fam_hits is not None and len(fam_hits):
        k = np.searchsorted(fam_hits, c, "left")
        lo = fam_hits[np.clip(k - 1, 0, len(fam_hits) - 1)]
        hi = fam_hits[np.clip(k, 0, len(fam_hits) - 1)]
        near = (np.abs(c - lo) <= EXCL_MIN * MIN_MS) | (np.abs(hi - c) <= EXCL_MIN * MIN_MS)
        c = c[~near]
    return c


def draw(trigger_id, elig, n=N_CTRL):
    """n minutes (sorted) drawn without replacement from elig with the trigger's own seeded generator. Pure."""
    elig = np.asarray(elig, np.int64)
    if len(elig) <= n:
        return [int(x) for x in elig]
    pick = rng_for(trigger_id).choice(len(elig), size=n, replace=False)
    return sorted(int(x) for x in elig[np.sort(pick)])


HOUR_MS = 3_600_000
GAP_PERMANENT_MS = 48 * HOUR_MS    # an hour file still absent this long after the hour ended is a permanent gap (recorder down /
                                   # feature build gave up) — counted in ctrl_day_missing_hours, the day can then finalise


def day_status(day_start, hours_present, rec_start_ms, now_ms):
    """(final, missing_hours) of a UTC day's control draw: every hourly file from the previous day's last hour to the next day's first
    hour (D−1 23:00 … D+1 01:00 — late hits there can exclude this day's minutes) is present, or permanently absent; hours before
    the recording start count as present. Pure."""
    missing, pending = 0, False
    for h in range(-1, 26):
        hs = day_start + h * HOUR_MS
        if hs < rec_start_ms // HOUR_MS * HOUR_MS:
            continue                                               # before the recording's first hour: nothing could exist
        if hs in hours_present:
            continue
        if now_ms >= hs + HOUR_MS + GAP_PERMANENT_MS:
            missing += 1
        else:
            pending = True
    return (not pending), missing


def controls_for(T, hits, book_minutes_by_sym, family_of, rec_start_ms, hours_present, now_ms):
    """T = trigger openers (trigger, direction, pair, minute, trigger_id, family). → DataFrame(trigger_id, trigger, family, pair,
    direction, ctrl_k, minute, n_eligible, ctrl_final) — one row per drawn control (a stub row when none is eligible). Pure."""
    fh = family_hits(hits, family_of)
    rows = []
    for r in T.itertuples():
        day = (int(r.minute) // DAY_MS) * DAY_MS
        bm = book_minutes_by_sym.get(r.pair)
        el = eligible(bm, day, fh.get((r.family, r.pair)), rec_start_ms) if bm is not None else np.array([], np.int64)
        fin, miss = day_status(day, hours_present, rec_start_ms, now_ms)
        base = dict(trigger_id=r.trigger_id, trigger=r.trigger, family=r.family, pair=r.pair, direction=r.direction,
                    n_eligible=len(el), ctrl_final=bool(fin), ctrl_day_missing_hours=miss)
        picks = draw(r.trigger_id, el)
        if not picks:
            rows.append(dict(base, ctrl_k=-1, minute=np.nan))
        for k, m in enumerate(picks):
            rows.append(dict(base, ctrl_k=k, minute=m))
    C = pd.DataFrame(rows, columns=["trigger_id", "trigger", "family", "pair", "direction", "ctrl_k", "minute", "n_eligible", "ctrl_final",
                                    "ctrl_day_missing_hours"])
    C["signal_time"] = pd.to_datetime(C.minute + MIN_MS, unit="ms", utc=True)
    C["ctrl_id"] = C.trigger_id + "|ctrl" + C.ctrl_k.astype(str) + "|" + C.minute.map(lambda x: "none" if x != x else str(int(x)))
    return C
