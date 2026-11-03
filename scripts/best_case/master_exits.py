"""Fills priced on TODAY's live exit where the data keeps them as traded (operator 2026-11-03: "Frenzy Long so much time? … re check";
DECISION_LOG 448).

MASTER — the builder keeps FRENZY_LONG at its traded stop on purpose (build_master_pool.py, DECISION_LOG 317 note): fills opened before
the DL 317 deploy traded the −3 stop, later ones the live −4. Priced at −4 here, else 5 of the 9 −3 losers that recover to the +1.5 TP
count as losers (29 · 69 % · +0.13 instead of 29 · 86 % · +0.74 %/fill). Source: the 10-19 stop study's tick re-walk of every real −3
touch (reports/LONG_STOP_SAVED_VS_DEEPER_2026-10-19_touches.csv, block REAL, why4 = TP / STOP at −4).
Every other sleeve's master exit = today's live exit (checked 2026-11-03: WILLY / HT V2b 10-26a · LITE / RETRY +3 / −3 10-30a ·
SPIKE_FADE, MOMENTUM, SURGE unchanged since their stack; FAN flips priced in flip_exits.py; Flatbase has no master era).

CURRENT BATCH — exits retired DURING the batch: LITE C +2 take-profits (until DL 420) → their +3 / −3 walk; Flatbase fills opened before
the DL 408 switch (+8 / −15) → their +5 / −10 walk from entry, close time moved to the walk's exit. Fills that ran today's exit keep their
real result. A fill a re-walk does not cover stops the run (never kept on the old exit silently).
Caveat: re-priced LITE and master LONG fills keep their as-traded close time (the walks carry no exit time) — a day-boundary shift is
possible in the daily compound; Flatbase moves its close time.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOUCHES = os.path.join(ROOT, "reports", "LONG_STOP_SAVED_VS_DEEPER_2026-10-19_touches.csv")
LITE_CF = os.path.join(ROOT, "reports", "LITE_C_REVERT_CF.csv")          # LITE C +2 closes walked on 1m at +3 / −3 / 12 h (DECISION_LOG 420)
LONG_TP, LONG_SL_LIVE, LONG_SL_OLD = 1.5, 4.0, 3.0
LONG_STOP4_UTC = pd.Timestamp("2026-10-19 17:00")                        # DL 317 deploy (scout_long_stop4.COHORT_START): later fills ran −4
FB_EXIT_UTC = pd.Timestamp("2026-10-28 21:37")                           # DL 408 commit (18:27 −03:00) + 10 min: later fills ran +5 / −10
LITE_TP_LIVE = 3.0                                                       # today's LITE / RETRY take profit — a TP close below it is LITE C
LITE = ("FRENZY_LITE", "FRENZY_LITE_RETRY")


def key(ts, pair):
    """one key format on every side: 'YYYY-MM-DDTHH:MM:SS|PAIR' (space or T, any sub-second part dropped)."""
    return str(ts).replace(" ", "T")[:19] + "|" + str(pair)


def _is_stop(reason):
    return "STOP" in str(reason).upper()


def reprice_long_stop4(M):
    """M = master rows (opened_at, o, pair, sl, pct, close_reason[, entry_frenzy_stop_pct]) → (M with −3-era FRENZY_LONG stops on the
    −4 exit, n moved)."""
    T = pd.read_csv(TOUCHES, low_memory=False)
    T = T[(T.block == "REAL") & (T.sleeve == "FRENZY_LONG")]
    why = {key(*k.split("|")[:2]): w for k, w in zip(T.key, T.why4)}
    stop = pd.to_numeric(M["entry_frenzy_stop_pct"], errors="coerce") if "entry_frenzy_stop_pct" in M else pd.Series(float("nan"), index=M.index)
    old = ((M.sl == "FRENZY_LONG") & (M.o < LONG_STOP4_UTC) & ~(stop >= LONG_SL_LIVE) & (M.pct < 0)
           & M.close_reason.map(_is_stop))
    M = M.copy()
    miss = []
    for i in M.index[old]:
        k = key(M.at[i, "opened_at"], M.at[i, "pair"])
        w = why.get(k)
        if w == "TP":
            M.at[i, "pct"] = LONG_TP
        elif w == "STOP":
            M.at[i, "pct"] = M.at[i, "pct"] - (LONG_SL_LIVE - LONG_SL_OLD)   # its own fill slippage kept, one more point of stop
        else:
            miss.append(k)
    if miss:
        sys.exit("FRENZY_LONG −3-era stops without a −4 re-walk (extend the 10-19 touches study):\n" + "\n".join(miss))
    return M, int(old.sum())


def reprice_b25_exits(A, fb_csv):
    """A = current-batch rows (o, opened_at, closed_at, close_reason, pair, sl, pct, u, notional_value) → (A on today's exits,
    {sleeve: n moved}). LITE / RETRY: a take-profit closed below today's +3 (= LITE C +2) → its +3 / −3 walk (LITE_C_REVERT_CF.csv).
    FLATBASE_SHORT opened before the DL 408 switch → its +5 / −10 walk from entry (fb_combo.csv now510, close at entry + h_now510)."""
    cf = pd.read_csv(LITE_CF)
    lite = {key(a, p): v for a, p, v in zip(cf.opened_at, cf.pair, cf.cf_pct)}
    fb = pd.read_csv(fb_csv)
    fbx = {key(k, p): (v, h) for k, p, v, h in zip(fb.k, fb.pair, fb.now510, fb.h_now510)}
    A = A.copy()
    moved, miss = {}, []
    for i in A.index:
        sl, k = A.at[i, "sl"], key(A.at[i, "opened_at"], A.at[i, "pair"])
        h = None
        if sl in LITE and "TP" in str(A.at[i, "close_reason"]).upper() and 0 < A.at[i, "pct"] < LITE_TP_LIVE - 0.5:
            v = lite.get(k)
        elif sl == "FLATBASE_SHORT" and A.at[i, "o"] < FB_EXIT_UTC:
            v, h = fbx.get(k, (None, None))
        else:
            continue
        if v is None or v != v or (h is not None and h != h):
            miss.append(f"{sl} {k}")
            continue
        A.at[i, "pct"] = float(v)
        A.at[i, "u"] = float(v) / 100 * float(A.at[i, "notional_value"])
        if h is not None:
            A.at[i, "closed_at"] = (A.at[i, "o"] + pd.Timedelta(minutes=float(h))).isoformat()
        moved[sl] = moved.get(sl, 0) + 1
    if miss:
        sys.exit("current-batch fills without a today's-exit re-walk (rebuild LITE_C_REVERT_CF / fb_combo):\n" + "\n".join(miss))
    return A, moved
