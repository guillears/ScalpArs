#!/usr/bin/env python3
"""STOP WIDTH (+ WILLY TP cap) on TODAY'S FILTERS — operator 2026-11-14: "do it with the current filters applied please".

Cohort = the master's kept fills (reports/MASTER_POOL_stacked.csv, stack_keep ∧ non-probe, every archived batch B18 → B25, each tagged by its
era) + B26 (the export passed, opened ≥ 2026-11-11 15:11 UTC — it already runs today's filters). Sleeves each apart: FRENZY_WILLY · HT ·
LITE · LITE RETRY.
BEFORE = the master's stack_pct (today's stack exit — the CLAUDE.md master ruler), the booked % for B26.
AFTER = BEFORE + (variant − today's re-walk) — a stop-only change moves only the fills that STOP under today's exit (B25 / B26: only fills
that stopped live; a re-walk stop on a fill live did not stop = a wick live rode through → Δ 0, counted "replica-only"; older eras ran other
stops, so today's re-walk decides which fills stop there — stated); a TP-cap change moves every fill.
"No stop" (LIQ) = liquidated at TODAY's leverage (WILLY / HT / LITE 8× ≈ −12 %, RETRY 10× ≈ −9.5 %).
Walker, grid, paths, slip and one size = scripts/study_stop_width_1114.py.
Usage: venv/bin/python -I scripts/study_stop_width_kept_1114.py <orders export.csv>"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_stop_width_1114 as M          # noqa: E402
import study_b19_fzexit_common as FX       # noqa: E402

LEV_TODAY = {"FRENZY_WILLY": 8, "FRENZY_WILLY_HT": 8, "FRENZY_LITE": 8, "FRENZY_LITE_RETRY": 10}
OUT = os.path.join(ROOT, "reports", "STOP_WIDTH_KEPT_2026-11-14")
KEY_CELLS = {"W": [(8, 4), (8, 5), (8, 6), (8, 8), (8, "LIQ"), (6, 4), (5, 4), (4, 4), (3, 4), (5, 6), (5, 8), (3, 6)],
             "L": [(0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, "LIQ")]}


def load(export):
    m = pd.read_csv(FX.MASTER, low_memory=False)
    m = m[m.entry_strategy.isin(M.SLEEVES) & (m.stack_keep == True)                                     # noqa: E712
          & ~m.cell_multiplier_source.astype(str).str.endswith("_PROBE")
          & ~m.close_reason.astype(str).str.upper().str.startswith(("MANUAL", "EXTERNAL"))].copy()
    m["batch"] = m.era.astype(str)
    m["before"] = pd.to_numeric(m.stack_pct, errors="coerce").fillna(pd.to_numeric(m.pnl_percentage, errors="coerce"))
    b = pd.read_csv(export, low_memory=False)
    b = b[(b.status.astype(str) == "CLOSED") & b.entry_strategy.isin(M.SLEEVES) & (b.opened_at.astype(str) >= M.B26_FROM)
          & ~b.cell_multiplier_source.astype(str).str.endswith("_PROBE")
          & ~b.close_reason.astype(str).str.upper().str.startswith(("MANUAL", "EXTERNAL"))].copy()
    b["batch"] = "B26"
    b["before"] = pd.to_numeric(b.pnl_percentage, errors="coerce")
    d = pd.concat([m, b], ignore_index=True)
    d["opened_at"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]          # review 546: one key format before the dedup / merge
    d = d.drop_duplicates(["opened_at", "pair", "entry_strategy"], keep="last")
    d["traded_lev"] = d.leverage
    d["leverage"] = d.entry_strategy.map(LEV_TODAY)          # LIQ at today's leverage
    return d.sort_values("opened_at").reset_index(drop=True)


def spans(d):
    s = {}
    for e, g in d.groupby("batch"):
        days = pd.to_datetime(g.closed_at.fillna(g.opened_at).astype(str).str[:10])
        s[e] = (days.min().strftime("%Y-%m-%d"), days.max().strftime("%Y-%m-%d"))
    s["B25"], s["B26"] = M.SPAN["B25"], M.SPAN["B26"]
    return s


def after(g, col, today, stop_only):
    dl = (g[col] - g[today]).fillna(0.0)
    if stop_only:
        stops_today = g[today + "_why"].eq("STOP")
        live = np.where(g.batch.isin(["B25", "B26"]), g.close.astype(str).str.upper().eq("STOP_LOSS"), stops_today)
        dl = dl.where(live & stops_today, 0.0)
    return g.before.values + dl.values


def main(export):
    th = M.CF.load_th()
    d = load(export)
    R = M.price(d, th)
    meta = d[["opened_at", "pair", "entry_strategy", "before", "traded_lev"]].assign(opened_at=lambda x: x.opened_at.astype(str).str[:19])
    R = R.merge(meta.rename(columns={"entry_strategy": "sleeve"}), on=["opened_at", "pair", "sleeve"], how="left")
    R.to_csv(f"{OUT}_rows.csv", index=False)
    SP = spans(d)
    L = ["# Stop width (+ WILLY TP cap) on TODAY'S FILTERS — 2026-11-14", "",
         "Kept fills only (master stack_keep, every archived batch) + B26. BEFORE = master stack_pct / B26 booked. AFTER = BEFORE + (variant − "
         "today's re-walk), stop-only changes on fills that stop under today's exit (B25 / B26: live stops only). LIQ = liquidation at today's "
         "leverage. Pts = Σ % per fill. DCR = %/day at today's one size over the batch's calendar days.", ""]
    for sl in M.SLEEVES:
        w = sl.startswith("FRENZY_WILLY"); kind = "W" if w else "L"; ts = 4 if w else 3; tc = 8 if w else 0
        today = f"c{tc}_s{ts}"; sz = M.SIZE[sl]
        G = R[(R.sleeve == sl) & R[today].notna()].copy() if today in R else R.iloc[0:0]
        L += ["", f"## {sl} — kept fills: " + " · ".join(f"{e} {n}" for e, n in G.batch.value_counts().sort_index().items())
              + f" · one stop today ≈ {ts * sz:.1f} % of the account", ""]
        if not len(G):
            continue
        par = (G[today] - G.before).abs() <= 0.5
        L.append(f"Re-walk of today's exit matches BEFORE (±0.5) on {int(par.sum())} of {len(G)}. Stop under today's exit: "
                 f"{int(G[today + '_why'].eq('STOP').sum())} of {len(G)}.")
        L += ["", "| TP cap | Stop | one stop = % acct | ALL kept N · WR · avg · Σ pts | Δ pts ALL [95 % day CI] | B25 Δ pts · DCR before → after | "
              "B26 Δ pts · DCR before → after | per-batch Δ pts |", "|---|---|---|---|---|---|---|---|"]
        cells = [(c, s) for c in (M.TPCAPS if w else [0]) for s in M.STOPS[kind]]
        for c, s in cells:
            col = f"c{c}_s{s}"
            a = np.asarray(after(G, col, today, c == tc), float)
            dl = a - G.before.values
            lo, hi = M.boot(dl, G.day)
            one = (M.liq(LEV_TODAY[sl]) if s == "LIQ" else s) * sz
            blk = []
            for bt in ("B25", "B26"):
                mk = (G.batch == bt).values
                if mk.any():
                    blk.append(f"{dl[mk].sum():+.1f} · {M.dcr(G.before.values[mk], G.day[mk], sz, SP[bt]):+.2f} → "
                               f"{M.dcr(a[mk], G.day[mk], sz, SP[bt]):+.2f}")
                else:
                    blk.append("–")
            per = " ".join(f"{e} {dl[(G.batch == e).values].sum():+.1f}" for e in sorted(G.batch.unique()))
            L.append(f"| {('+' + str(c)) if w else 'LITE TP'} | {'liq' if s == 'LIQ' else '−' + str(s)} | {one:.1f} | {len(a)} · {(a > 0).mean():.0%} · "
                     f"{a.mean():+.2f} · {a.sum():+.1f} | {dl.sum():+.1f} [{lo:+.1f}, {hi:+.1f}] | {blk[0]} | {blk[1]} | {per} |")
        L.append("")
    open(f"{OUT}.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
