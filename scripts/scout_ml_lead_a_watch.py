#!/usr/bin/env python3
"""👁 Scout — ML_LEAD_A multiplier watch (registered 2026-11-06, DECISION_LOG 481; OBSERVE ONLY — never changes config or size, no bot API; public 1h
klines only for the ML_MKT_HOT breadth of fills the best-case zones.csv does not score, through the budgeted cache of scripts/scout_ml_mkt_hot_watch.py).

QUESTION   Inside the momentum longs ML_MKT_HOT keeps (market NOT hot), is the 2D winner cell "coin EMA20 slope > 0.1481 ∧ coin RSI prev ≤ 60.9" a
           real winner cell worth a SIZE MULTIPLIER (caps for losers, multipliers for winners — never a block on the rest)? Lead from the 11-04 not-hot
           2D read (DECISION_LOG 449 note: the exhaustive 3,280-cell search did NOT beat luck, best-cell p 0.57); operator 11-06: register it.
CELL       FROZEN at the 11-04 read: `entry_ema20_slope` > 0.1481 ∧ `entry_rsi_prev` ≤ 60.9 (stamped on every fill; either NaN → unscored, never counted).
COHORT     full-size momentum longs, HISTORY + FORWARD re-read every run (operator 11-05: "the decision will be based on forward + history"): MASTER =
           reports/MASTER_POOL_stacked.csv kept on today's stack (booked = stack_pct) · B25 = export fills opened from the B25 start to this line's
           registration (DL 481 commit + 10 min) · FORWARD = after it. Exports = Downloads + exports_archive/, dedup (opened_at[:19], pair, direction,
           entry_strategy), CLOSED beats OPEN, newest export by its download stamp; probes out (both flags); operator-closed out; not already in the master
           (raw master keys). MASTER + B25 are IN-SAMPLE (the cell was found on them) — shown, never enough alone.
VERDICT    (operator 11-06: "i dont think is correct not to consider all the trades of the batch and to do it forward looking") — HISTORY + B25 +
           FORWARD pooled, re-read every run, never forward-only: RETIRE iff the pooled cell avg ≤ the pooled rest-of-not-hot avg · MULTIPLIER CANDIDATE
           (operator decides; never automatic) iff the pooled cell meets Pattern W (N ≥ 30, WR ≥ 70 %, avg ≥ +0.10 %, Σ > 0) ∧ day-clustered bootstrap
           P(cell avg > rest avg) ≥ 0.95 (4,000, seed 7) ∧ no day / pair ≥ 50 % of the cell's gain ∧ the NEWEST trades do not contradict (B25 + FORWARD
           cell ≥ 5 fills → its avg must be above the B25 + FORWARD rest) · else KEEP OBSERVING. Built-on caveat: the cell was the best of a 3,280-cell
           2D search that did NOT beat luck (p 0.57) and the bootstrap is not search-corrected — a live multiplier carries the 30–50 % haircut, first
           step 1.5×, a pre-committed tight revert, and needs the engine to score market breadth (the cell on the HOT side loses).
Usage: venv/bin/python -I scripts/scout_ml_lead_a_watch.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import scout_gatekit as _GK                # noqa: E402  (operator-closed fills out · the DL commit time)
import scout_ml_fixed_tp_watch as FT       # noqa: E402  (export loader + breadth scoring of unscored fills, reused unchanged)
import study_ml_fixed_tp_recorded as RT    # noqa: E402  (master cohort keys, ML_MKT_HOT from zones.csv)
import sweep_separators as SW              # noqa: E402  (probe flags)

DL = 481
SLOPE_MIN, RSI_PREV_MAX = 0.1481, 60.9
P_MIN, NEWEST_MIN = 0.95, 5
BOOT_N, BOOT_SEED = 4000, 7
W_N, W_WR, W_AVG = 30, 0.70, 0.10
SHARE_MAX = 0.50
BLOCKS = ("MASTER", "B25", "FORWARD")
CANDIDATE = "MULTIPLIER CANDIDATE (operator decides; never automatic)"


def cell(d):
    """→ object Series: True in the cell, False out, NaN unscored. PURE."""
    s = pd.to_numeric(d["entry_ema20_slope"], errors="coerce")
    r = pd.to_numeric(d["entry_rsi_prev"], errors="coerce")
    v = (s > SLOPE_MIN) & (r <= RSI_PREV_MAX)
    return v.astype(object).where(s.notna() & r.notna(), np.nan)


def cohort(paths=None, since=None, master=None):
    """→ (fills MASTER + B25 + FORWARD with pct / block / day / pair / mkt_hot / cell, newest export time, operator-closed fills)."""
    if master is None:
        M = pd.read_csv(RT.MASTER, low_memory=False)
        M = M[RT._ml(M) & RT._true(M.stack_keep) & ~SW._is_probe(M)].copy()
        M["pct"] = pd.to_numeric(M.stack_pct, errors="coerce")
        keys = RT.master_keys()
    else:
        M = master.copy()
        keys = set(RT._key(M.opened_at) + M.pair) if len(M) else set()
    M["block"] = "MASTER"
    E, newest = FT.load_exports(paths)
    ox = pd.DataFrame(columns=["opened_at", "pair"])
    if len(E):
        E = E[E.status.astype(str).str.upper().eq("CLOSED") & RT._ml(E)]
        E, ox = _GK.split_operator_closed(E) if len(E) else (E, ox)
        E = E[(pd.to_datetime(E.opened_at, format="mixed") >= RT.B25_OPEN) & ~SW._is_probe(E)].copy()
        E = E[~(RT._key(E.opened_at) + E.pair).isin(keys)]
        E["pct"] = pd.to_numeric(E.pnl_percentage, errors="coerce")
        E["block"] = "B25"
        if since is not None:
            E.loc[pd.to_datetime(E.opened_at, format="mixed") >= since, "block"] = "FORWARD"
    cols = ["block", "opened_at", "pair", "pct", "entry_ema20_slope", "entry_rsi_prev"]
    F = pd.concat([M.reindex(columns=cols), E.reindex(columns=cols)], ignore_index=True)
    F = F[F.pct.notna()].reset_index(drop=True)
    F["day"] = pd.to_datetime(F.opened_at, format="mixed").dt.strftime("%Y-%m-%d")
    F["mkt_hot"] = RT.attach_mkt_hot(F)
    F["cell"] = cell(F)
    return F, newest, ox


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "1.0")


def _share(x, key):
    """largest key's share of the summed POSITIVE contribution (NaN when nothing is positive). PURE."""
    s = pd.Series(np.asarray(x, float)).groupby(np.asarray(key)).sum()
    pos = s[s > 0]
    return float(pos.max() / pos.sum()) if len(pos) else float("nan")


def boot_p(F, n=BOOT_N, seed=BOOT_SEED):
    """day-clustered bootstrap P(cell avg > rest avg) on scored fills (cell True / False). PURE."""
    days = F.day.unique()
    if len(days) < 2:
        return float("nan")
    idx = {d: np.flatnonzero(F.day.to_numpy() == d) for d in days}
    pct, cel = F.pct.to_numpy(float), (F.cell == True).to_numpy()   # noqa: E712
    rng, win, tot = np.random.default_rng(seed), 0, 0
    for _ in range(n):
        ii = np.concatenate([idx[d] for d in rng.choice(days, len(days))])
        a, b = pct[ii][cel[ii]], pct[ii][~cel[ii]]
        if len(a) and len(b):
            tot += 1
            win += a.mean() > b.mean()
    return win / tot if tot else float("nan")


def verdict(F):
    """F = scored not-hot fills (cell True / False) of every block → (state, detail). History + B25 + forward pooled. PURE."""
    def _s(g):
        return f"{len(g)} · {(g.pct > 0).mean():.0%} · {g.pct.mean():+.2f}" if len(g) else "0"
    c, r = F[F.cell == True], F[F.cell == False]   # noqa: E712 (object column)
    nw = F[F.block != "MASTER"]
    nc, nr = nw[nw.cell == True], nw[nw.cell == False]   # noqa: E712
    p = boot_p(F)
    sh = max(_share(c.pct, c.day), _share(c.pct, c.pair)) if len(c) else float("nan")
    det = (f"pooled cell {_s(c)} on {c.day.nunique()} d vs rest {_s(r)} · P(cell > rest) {p:.3f} · biggest day / pair {sh:.0%} of the gain · "
           f"newest (B25 + forward) cell {_s(nc)} vs rest {_s(nr)}")
    if not len(c) or not len(r):
        return "COLLECTING", det
    if c.pct.mean() <= r.pct.mean():
        return "RETIRE", det
    newest_ok = len(nc) < NEWEST_MIN or (len(nr) and nc.pct.mean() > nr.pct.mean())
    ok = (len(c) >= W_N and (c.pct > 0).mean() >= W_WR and c.pct.mean() >= W_AVG and c.pct.sum() > 0
          and np.isfinite(p) and p >= P_MIN and np.isfinite(sh) and sh < SHARE_MAX and newest_ok)
    return (CANDIDATE if ok else "KEEP OBSERVING"), det


def _row(label, g):
    if not len(g):
        return f"| {label} | 0 | — | — | — | — |"
    return f"| {label} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.2f} | {g.pct.sum():+.2f} |"


def run(now_ms=None, paths=None, since_ms=None, breadth_fn=None, master=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 481 commit + 10 min splits B25 from FORWARD (not in git → no FORWARD split yet)."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 ML_LEAD_A — momentum longs, not-hot winner cell (coin EMA20 slope > {SLOPE_MIN} ∧ RSI prev ≤ {RSI_PREV_MAX}), multiplier watch, "
         f"HISTORY + FORWARD (registered 2026-11-06, DECISION_LOG {DL}, OBSERVE only)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="ML_LEAD_A watch line")
    since = pd.Timestamp(int(since_ms), unit="ms") if since_ms is not None else None
    F, newest, ox = cohort(paths, since, master)
    F, complete = FT.score_missing(F, now_ms, breadth_fn)
    n = {b: int((F.block == b).sum()) for b in BLOCKS}
    L += [f"Cohort: MASTER {n['MASTER']} · B25 {n['B25']} · FORWARD {n['FORWARD']} full-size momentum longs"
          + (f" (FORWARD = opened ≥ {since:%Y-%m-%d %H:%M} UTC)" if since is not None else f" (DL {DL} commit not in git: no FORWARD split yet)")
          + f" · {_GK.operator_closed_line(ox)}. MASTER + B25 are in-sample (the cell was found on them).", ""]
    hot = F.mkt_hot
    unscored = hot.isna() | F.cell.isna()
    if not complete:
        L += [f"⚠ breadth fetch incomplete this run: {int(hot.isna().sum())} fills unscored (never counted on either side).", ""]
    nh = F[~unscored & hot.map(lambda v: not _truthy(v) if pd.notna(v) else False).astype(bool)]
    L += ["| Block · group | N | days | WR | avg % | Σ % |", "|---|---|---|---|---|---|"]
    for b in BLOCKS + ("ALL",):
        g = nh if b == "ALL" else nh[nh.block == b]
        if b != "ALL" and not len(g):
            continue
        L += [_row(f"**{b}** · cell (not hot)", g[g.cell == True]), _row(f"{b} · rest of not-hot", g[g.cell == False])]   # noqa: E712
    hh = F[~unscored & hot.map(lambda v: _truthy(v) if pd.notna(v) else False).astype(bool)]
    L += [_row("_hot side (ML_MKT_HOT zone), cell — display only_", hh[hh.cell == True]),   # noqa: E712
          f"| _unscored (breadth or a cell leg missing) — never counted_ | {int(unscored.sum())} | | | | |", ""]
    st, det = verdict(nh)
    L += [f"- **{st}** — {det}", "",
          f"VERDICT rule (DL {DL}, history + B25 + forward pooled, re-read every run): RETIRE if the pooled cell avg ≤ the rest · {CANDIDATE} iff the "
          f"pooled cell meets Pattern W (N ≥ {W_N}, WR ≥ {W_WR:.0%}, avg ≥ +{W_AVG:.2f} %, Σ > 0) ∧ day-bootstrap P(cell > rest) ≥ {P_MIN} ∧ no day / pair "
          f"≥ {SHARE_MAX:.0%} of the gain ∧ the newest trades (B25 + forward, ≥ {NEWEST_MIN} cell fills) above their rest · else KEEP OBSERVING. "
          "Built on a 2D search that did not beat luck (p 0.57): a live step = 1.5× first, haircut, tight revert, engine breadth scoring.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
