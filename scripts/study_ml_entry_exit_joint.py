#!/usr/bin/env python3
"""🎯×🚪 MOMENTUM-LONG ENTRY × EXIT JOINT READ (DECISION_LOG 477, operator 2026-11-05: "after applying a potential filter of winners vs
losers, we also have different ideas of exit strategies to maximize the gains of those winners" → "why dont we have fix TP as
alternatives in momentum longs too? 1% 2% 3% … or even 0.5%" → "considering all momentum longs in the batch, not only moving forward").

The CLAUDE.md ENTRY × EXIT rule ③ — an exit candidate is re-read on the fills a pending entry line KEEPS — as one table. Observe-only:
no trading change, no gate moves; the registered exit lines (DL 380 OBS-1/2/3) keep their own frozen cohorts and bars.

COHORT: every momentum-long fill the master keeps on today's stack (MASTER_POOL_stacked stack_keep, non-probe) + EVERY B25 momentum-long
fill of the latest orders export (closed, non-probe, opened ≥ 2026-10-23 18:00, not already in the master) — the whole batch, not only
fills after a freeze.
ENTRY COHORTS (rows): ALL · kept by each momentum-long watch line (fills NOT in its block zone: ML_MKT_HOT · ML_W4 · STOP_COOLDOWN ·
PAIR_1H_DOWNTREND, scripts/watch_zones.py; MKT_HOT / STOP_COOLDOWN read from the best-case zones.csv, unscored fills stay out of that
line's row — unscored ≠ kept) · kept by EVERY line.
EXITS (columns), each priced on the fill's own path through the VALIDATED replica (scripts/study_ml_exit_b25_run.py evaluator, real
ticks where archived else 1m bars, fees in, recovery hold as live) vs BASE = today's exit through the SAME replica (Δ is replica −
replica, never replica − booked):
  registered (DL 380): OBS-1 W_LAD_off · OBS-2 W_LOCK0.2 · OBS-3 W_TW30_k1.5 / W_K1.25 / W_LAD_loose
  NEW, pre-declared here 2026-11-05 before any run (operator's fixed take-profits):
    FTP0.5 / FTP1 / FTP2 / FTP3 = a fixed TP at +X % net with today's stop, 180-min no-expansion clock and recovery hold — the
                                  runner trail and the hard-TP ladder OFF; booked AT +X (a resting limit), never at the bar's high
    FTP1_R / FTP2_R / FTP3_R    = today's runner trail kept, ladder off, capped by a fixed TP at +X %
Not priced (blind spots, listed on the report): the BTC-reactive lines OBS-B1 / B2 (need the BTC tick path — their own scout).
LUCK: per entry cohort, the day-block sign-flip test of scripts/best_case/luck_test.gain_line — each exit's Δ alone, and the best exit
after the search over all priced exits of that cohort; the page states how many entry × exit combinations were looked at.

    venv/bin/python -W ignore scripts/study_ml_entry_exit_joint.py [--export PATH] [--no-fetch] [--jobs 8]
Outputs: reports/ML_ENTRY_EXIT_JOINT.md · reports/ML_ENTRY_EXIT_JOINT_fills.csv (fill × exit, with the entry-line flags).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "best_case"))
import study_ml_wl_b25prep_data as LD  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT_MD = os.path.join(ROOT, "reports", "ML_ENTRY_EXIT_JOINT.md")
OUT_CSV = os.path.join(ROOT, "reports", "ML_ENTRY_EXIT_JOINT_fills.csv")
ZONES = os.path.join(ROOT, "reports", "best_case", "zones.csv")
LINES = ("ML_MKT_HOT", "ML_W4", "STOP_COOLDOWN", "PAIR_1H_DOWNTREND")
REGISTERED = {"OBS-1 ladder off": "W_LAD_off", "OBS-2 lock +0.20": "W_LOCK0.2", "OBS-3 trail ×1.5 after 30 min": "W_TW30_k1.5",
              "OBS-3 trail ×1.25": "W_K1.25", "OBS-3 looser ladder": "W_LAD_loose"}
FTP_LEVELS = (0.5, 1.0, 2.0, 3.0)
FTP_R_LEVELS = (1.0, 2.0, 3.0)


def latest_export():
    fs = glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-1*.csv")) + \
        glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-1*.csv"))
    return max(fs, key=lambda f: os.path.basename(f)) if fs else None


def _patch_b25(path):
    """the shared loader globs the late-October exports only → point it at the chosen export (every B25 fill, not a stale file)."""
    LD._b25_file = lambda: path


def ftp_variants(V):
    """the operator's fixed take-profits, PRE-DECLARED (2026-11-05). V = the run module's variant builder. PURE."""
    out = {}
    for x in FTP_LEVELS:
        out[f"FTP{x:g}"] = V(tp=x, ladder=[], run_arm=99.0)                 # plain TP + today's stop (runner never arms)
    for x in FTP_R_LEVELS:
        out[f"FTP{x:g}_R"] = V(tp=x, ladder=[])                             # today's runner trail, no ladder, capped at +x
    return out


def exit_columns():
    return list(REGISTERED.values()) + [f"FTP{x:g}" for x in FTP_LEVELS] + [f"FTP{x:g}_R" for x in FTP_R_LEVELS]


def label(col):
    rev = {v: k for k, v in REGISTERED.items()}
    if col in rev:
        return rev[col]
    if col.endswith("_R"):
        return f"TP +{col[3:-2]} % + runner"
    return f"fixed TP +{col[3:]} %"


def split_rows(F, cols):
    """per exit: fills NEWLY STOPPED vs today's exit (the CLAUDE.md stop rule — a replica stop on a 1m bar may be a wick live polling
    rode through) and Σ Δ on tick paths vs 1m-bar paths. F carries <col>_why and path_src. PURE."""
    out = []
    base_stop = F["BASE_why"].astype(str).str.contains("STOP")
    for c in cols:
        new_stop = F[f"{c}_why"].astype(str).str.contains("STOP") & ~base_stop
        d = F[c] - F.BASE
        tick = F.path_src.astype(str).str.contains("tick")
        out.append(dict(exit=c, new_stops=int(new_stop.sum()), new_stops_tick=int((new_stop & tick).sum()),
                        new_stops_1m=int((new_stop & ~tick).sum()), sum_d_new_stops=float(d[new_stop].sum()),
                        n_tick=int((tick & d.notna()).sum()), sum_d_tick=float(d[tick].sum()),
                        n_1m=int((~tick & d.notna()).sum()), sum_d_1m=float(d[~tick].sum())))
    return pd.DataFrame(out)


def entry_cohorts(F):
    """F = fills with the line flags (True = in the block zone, NaN = unscored) → {row name: boolean mask}. PURE."""
    out = {"ALL fills": pd.Series(True, index=F.index)}
    kept_all = pd.Series(True, index=F.index)
    for ln in LINES:
        z = F[ln]
        scored = z.notna()
        out[f"kept by {ln}"] = scored & ~z.fillna(False).astype(bool)
        kept_all &= scored & ~z.fillna(False).astype(bool)
    out["kept by EVERY line"] = kept_all
    return out


def summarise(F, cohorts, cols, r=2000):
    """→ DataFrame (cohort × exit): n, days, base avg %, exit avg %, Σ Δ, better %, worse %, p alone, p after the exit search. PURE-ish
    (calls luck_test)."""
    import luck_test as LT
    rows = []
    for cname, m in cohorts.items():
        S = F[m & F.BASE.notna()]
        if not len(S):
            continue
        D = pd.DataFrame({c: S[c] - S.BASE for c in cols})
        days = S.day.astype(str).to_numpy()
        for c in cols:
            ok = D[c].notna()
            n = int(ok.sum())
            g = LT.gain_line(D[cols], days, c, r=r) if n >= LT.MIN_N else dict(p_single=np.nan, p_search=np.nan, verdict=LT.FEW, dropped=0)
            rows.append(dict(cohort=cname, exit=c, n=n, days=int(S.day[ok].nunique()), base_avg=float(S.BASE[ok].mean()) if n else np.nan,
                             exit_avg=float(S[c][ok].mean()) if n else np.nan, sum_d=float(D[c][ok].sum()),
                             better=float((D[c][ok] > 0.005).mean() * 100) if n else np.nan,
                             worse=float((D[c][ok] < -0.005).mean() * 100) if n else np.nan,
                             p_alone=g["p_single"], p_search=g["p_search"], verdict=g["verdict"],
                             n_master=int((ok & S.block.eq("MASTER")).sum()), n_b25=int((ok & S.block.eq("B25")).sum()),
                             sum_d_master=float(D[c][ok & S.block.eq("MASTER")].sum()), sum_d_b25=float(D[c][ok & S.block.eq("B25")].sum())))
    return pd.DataFrame(rows)


def attach_lines(C):
    """the four entry-line flags per fill: ML_W4 / PAIR_1H_DOWNTREND from the fill's own stamps (watch_zones), ML_MKT_HOT / STOP_COOLDOWN
    from the best-case zones.csv (they need market data the row does not carry) — NaN where not scored."""
    import watch_zones as WZ
    z = WZ.zones(C)["MOMENTUM_LONG"]
    C = C.copy()
    C["ML_W4"] = z["ML_W4"].astype(object)
    gap = pd.to_numeric(C.get("entry_pair_1h_ema20_200_gap_pct"), errors="coerce")
    C["PAIR_1H_DOWNTREND"] = np.where(gap.notna(), (gap <= 0), np.nan).astype(object)
    C["ML_MKT_HOT"] = np.nan
    C["STOP_COOLDOWN"] = np.nan
    if os.path.exists(ZONES):
        Z = pd.read_csv(ZONES)
        Z = Z[Z.sl == "MOMENTUM_LONG"].copy()
        Z["k"] = Z.o.astype(str).str[:19].str.replace(" ", "T") + "|" + Z.pair
        for ln in ("ML_MKT_HOT", "STOP_COOLDOWN"):
            mp = dict(zip(Z.k, Z[ln]))
            v = (C.key_open + "|" + C.pair).map(mp)
            C[ln] = v.map(lambda x: np.nan if pd.isna(x) else str(x).lower() in ("true", "1", "1.0")).astype(object)
    return C


def fmt(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def report(T, F, cols, export, n_combos, X=None):
    L = []
    L.append("# Momentum longs — entry × exit joint read (observe-only)\n")
    L.append(f"Cohort: master kept on today's stack + every B25 momentum-long fill of `{os.path.basename(export or '?')}` · "
             f"{len(F)} fills ({int(F.block.eq('MASTER').sum())} master · {int(F.block.eq('B25').sum())} B25) · priced on "
             f"{int(F.BASE.notna().sum())}. Δ = exit − today's exit, both through the validated replica, % at 1× per fill (fees in). "
             f"**{n_combos} entry × exit combinations were looked at** — the luck p after the exit search counts the exits of a row only; "
             "across rows, expect ~1 in 20 to show p ≤ 0.05 by chance.\n")
    L.append("Entry-line coverage (unscored fills sit in no 'kept by' row of that line — unscored ≠ kept): " + " · ".join(
        f"{ln} scored {int(F[ln].notna().sum())} (in zone {int(F[ln].fillna(False).astype(bool).sum())})" for ln in LINES) + ".\n")
    L.append("Exits: " + " · ".join(f"`{c}` = {label(c)}" for c in cols) + ". Fixed TP = a take-profit at +X % net with today's stop "
             "with the 180-min clock and recovery hold as today (runner and ladder off), booked AT +X; `_R` keeps today's runner trail with the "
             "ladder off and the TP as a cap.\n")
    for cname in T.cohort.unique():
        S = T[T.cohort == cname]
        b = S.iloc[0]
        L.append(f"\n## {cname} — {b.n} fills on {b.days} days · today's exit {fmt(b.base_avg)} % avg\n")
        L.append("| Exit | Avg % | Σ Δ (master · B25) | Better / worse | p alone | p after exit search | Verdict |")
        L.append("|---|---|---|---|---|---|---|")
        for r in S.sort_values("sum_d", ascending=False).itertuples():
            L.append(f"| {label(r.exit)} | {fmt(r.exit_avg)} | {fmt(r.sum_d, 1)} ({fmt(r.sum_d_master, 1)} · {fmt(r.sum_d_b25, 1)}) | "
                     f"{r.better:.0f} % / {r.worse:.0f} % | {fmt(r.p_alone, 3, False)} | {fmt(r.p_search, 3, False)} | {r.verdict} |")
    if X is not None and len(X):
        L.append("\n## Path quality — ALL fills (read before trusting a gain)\n")
        L.append("Fills newly stopped = stopped under the exit but not under today's exit (a 1m-bar stop may be a wick live polling rode "
                 "through — CLAUDE.md stop rule). Σ Δ split by path: real ticks vs 1m bars (1m resolves a bar's high / low order by its "
                 "colour, so it can over- or under-state either side).\n")
        L.append("| Exit | Newly stopped (tick · 1m) | Σ Δ on them | Σ Δ tick paths (n) | Σ Δ 1m paths (n) |")
        L.append("|---|---|---|---|---|")
        for r in X.itertuples():
            L.append(f"| {label(r.exit)} | {r.new_stops} ({r.new_stops_tick} · {r.new_stops_1m}) | {fmt(r.sum_d_new_stops, 1)} | "
                     f"{fmt(r.sum_d_tick, 1)} ({r.n_tick}) | {fmt(r.sum_d_1m, 1)} ({r.n_1m}) |")
    L.append("\n## Blind spots\n")
    L.append("- BTC-reactive exits (OBS-B1 / B2) not priced here (BTC tick path) — read in their own scout.")
    L.append("- ML_MKT_HOT / STOP_COOLDOWN are scored only where the best-case zones.csv scored the fill (latest best-case run); "
             "PAIR_1H_DOWNTREND only where the fill carries its 1h EMA20/200 stamp (stamped late in the master) — see the coverage line; "
             "'kept by EVERY line' needs all four scored, so it is the smallest row.")
    L.append(f"- Fills without a price path or with a replica error: {int(F.BASE.isna().sum())}; per exit, a path that ran out before "
             "the exit (END_OF_DATA) is left out, never priced at its last tick.")
    L.append("- Same-bar fixed TP vs stop on 1m bars follows the bar colour (green: low first, red: high first) — a heuristic.")
    L.append("- Exit lines registered on ATR ≥ 0.50 fills from 2026-10-27 (DL 380) keep THEIR frozen cohort and bar; this table is wider "
             "(every kept fill, master + B25) and decides nothing on its own.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    export = a.export or latest_export()
    if not export:
        sys.exit("no orders export found (~/Downloads or exports_archive)")
    _patch_b25(export)
    import study_ml_exit_b25_data as DD
    if not a.no_fetch:
        import study_ml_exit_b25_fetch as FE
        FE.main()                                   # public 1m klines only for fills without a path; budgeted, honours the cooldown
    import study_ml_entry_exit_joint_worker as W
    C = DD.cohort()
    C = C[C.block.isin(["MASTER", "B25"])].copy()
    C["day"] = C.ts.dt.strftime("%Y-%m-%d")
    items = [(DD.fill_dict(r), r.key) for r in C.itertuples()]
    res = W.price(items, a.jobs)
    P = res.pivot_table(index="key", columns="variant", values="pct", aggfunc="first")
    R = res.pivot_table(index="key", columns="variant", values="reason", aggfunc="first").add_suffix("_why")
    src = res[res.variant == "BASE"].set_index("key").src.rename("path_src")
    cols = exit_columns()
    F = attach_lines(C).set_index("key").join(P[["BASE"] + cols], how="left").join(R).join(src).reset_index()
    F.to_csv(OUT_CSV, index=False, columns=["key", "block", "pair", "ts", "day", "y", "path_src"] + list(LINES) + ["BASE"] + cols
             + ["BASE_why"] + [c + "_why" for c in cols])
    cohorts = entry_cohorts(F)
    T = summarise(F, cohorts, cols)
    open(OUT_MD, "w").write(report(T, F, cols, export, len(cohorts) * len(cols), split_rows(F, cols)))
    print(f"{len(F)} fills → {OUT_MD}")


if __name__ == "__main__":
    main()
