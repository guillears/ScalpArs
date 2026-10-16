#!/usr/bin/env python3
"""WILLY LOW-VWAP / SHORT-RUN zone — deep backtest review (2026-10-16). Research only, cache-only.

Hypothesis frozen from the live batch 2026-10-15/16 (3/3 WILLY losers matched, 1/5 winners):
  ZONE = FRENZY_WILLY entry with entry_frenzy_vs_vwap_pct <= 0.10  AND  entry_frenzy_run_pct <= 10.  Legs reported alone too.
Cuts are NOT re-fitted.

Pricing = TODAY's WILLY exit: resting maker TP +1.25 net (books the line), stop -4 booked at min(print, -4) - 0.2 slip,
90-min cap (market close at the last print), late TP +0.25 after 30 min (DL 297, books the line).
Walkers reused (never rewritten): scout_frenzy_willy_exit.load_path / evaluate (via study_tick_maker_lock_common patches for 2025
data) and scout_willy_timecap.evaluate_late. Parity asserted vs reports/WILLY_DURATION_cell_rows.csv (base_pct, T30_TP0.25).
Exit-modifier arm: zone fills TP +1.0 instead of +1.25 (late TP unchanged).

Usage: venv/bin/python scripts/study_willy_lowvwap_shortrun.py [--jobs 8]
"""
import os, sys, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_tick_maker_lock_common as C  # noqa: E402  (patches WT / FX loaders for 2025 data, blocks network)
FX, WT = C.FX, C.WT
WT._NET_BLOCKED = True

TP, TP_MOD, SL, CAP, LATE, LTP, SLIP = 1.25, 1.0, 4.0, 90, 30, 0.25, 0.2
VW_CUT, RUN_CUT = 0.10, 10.0
FILLS = os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
DUR = os.path.join(REP, "WILLY_DURATION_cell_rows.csv")
MASTER = os.path.join(REP, "MASTER_POOL_stacked.csv")
LIVE = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-16_13-10-21 (1).csv")
OUT_ROWS = os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_rows.csv")


def ms(ts):
    return pd.to_datetime(ts, format="mixed").values.astype("datetime64[ms]").astype("int64")


def book(ev):
    w = ev["why"]
    if w in ("TP", "LATE_TP"):
        return float(ev["pct_line"])
    if w == "STOP":
        return min(float(ev["pct"]), -SL) - SLIP
    if w == "CAP":
        return float(ev["pct"])
    return np.nan


def walk(r, tc):
    P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), CAP, tc)
    plain = FX.evaluate(P, TP, SL, CAP)
    tod = WT.evaluate_late(P, TP, SL, CAP, LATE, LTP)
    mod = WT.evaluate_late(P, TP_MOD, SL, CAP, LATE, LTP)
    src = FX.path_src(P)
    on_t = bool(P["n_tick"] > 0 and src == "ticks" and not tod["on_bar"] and not mod["on_bar"]
                and tod["why"] != "UNRES" and mod["why"] != "UNRES")
    return dict(key=r["key"], src=src, tick_priced=on_t, plain_why=plain["why"], plain_pct=plain["pct"],
                raw_why=tod["why"], raw_pct=tod["pct"], why=tod["why"], xmin=tod["xmin"], y=book(tod),
                mod_why=mod["why"], y_mod=book(mod))


def job(rows):
    WT._NET_BLOCKED = True
    tc, out = {}, []
    for r in rows:
        try:
            out.append(walk(r, tc))
        except Exception as e:  # noqa: BLE001
            out.append(dict(key=r["key"], err=str(e)[:160]))
        if len(tc) > 400:
            tc.clear()
    return out


def run_walks(U, jobs):
    from multiprocessing import Pool
    items = U[["key", "pair", "te", "E"]].to_dict("records")
    pairs = sorted({r["pair"] for r in items}); g = {p: k % (jobs * 4) for k, p in enumerate(pairs)}
    chunks = [sorted([r for r in items if g[r["pair"]] == k], key=lambda r: (r["pair"], r["te"])) for k in range(jobs * 4)]
    with Pool(jobs) as Pp:
        res = sum(Pp.map(job, [c for c in chunks if c], chunksize=1), [])
    return pd.DataFrame(res)


def load_bt():
    D = pd.read_csv(FILLS, low_memory=False)
    D = D[(D.sleeve == "FRENZY_WILLY") & (D.status == "CLOSED") & (~D.is_probe.astype(bool))].copy()
    assert not D.duplicated(["seed", "pair", "opened_at"]).any()
    D["te"] = ms(D.opened_at.astype(str)); D["E"] = D.entry_price.astype(float)
    D["key"] = D.pair + "|" + D.te.astype(str) + "|" + D.E.map(lambda x: f"{x:.10g}")
    D["ts"] = pd.to_datetime(D.te, unit="ms")
    D["day"] = D.ts.dt.strftime("%Y-%m-%d"); D["month"] = D.ts.dt.strftime("%Y-%m")
    D["era"] = np.where(D.ts < "2026-01-01", "2025", np.where(D.ts < "2026-07-01", "2026-H1", "2026-H2"))
    import study_yr6wl_frenzy_common as YC
    T = YC.top50_pairdays()
    D["top50"] = [(d, p) in T for d, p in zip(D.day, D.pair)]
    return D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    D = load_bt()
    M = pd.read_csv(MASTER, low_memory=False)
    M = M[M.entry_strategy == "FRENZY_WILLY"].copy()
    M["te"] = ms(M.opened_at.astype(str).str[:19]); M["E"] = M.entry_price.astype(float)
    M["key"] = M.pair + "|" + M.te.astype(str) + "|" + M.E.map(lambda x: f"{x:.10g}")
    U = pd.concat([D[["key", "pair", "te", "E"]], M[["key", "pair", "te", "E"]]]).drop_duplicates("key")
    print(f"walking {len(U)} unique paths ({len(D)} BT fills over 3 seeds + {len(M)} master)", flush=True)
    R = run_walks(U, a.jobs)
    if "err" in R.columns:
        print("walk errors:", R.err.notna().sum(), R[R.err.notna()].head())
    B = D.merge(R, on="key", how="left"); B["block"] = "BT"
    MM = M.merge(R, on="key", how="left"); MM["block"] = "MASTER"
    # parity vs the stored WILLY_DURATION walk (seed-1 BT + MASTER)
    S = pd.read_csv(DUR, low_memory=False)
    S["key"] = S.pair + "|" + S.te.astype(str) + "|" + S.E.map(lambda x: f"{float(x):.10g}")
    J = S.merge(R, on="key", how="inner")
    ok1 = np.isclose(J.base_pct.astype(float), J.plain_pct.astype(float), atol=1e-6, equal_nan=True) & (J.base_why.astype(str) == J.plain_why.astype(str))
    ok2 = np.isclose(J["T30_TP0.25"].astype(float), J.raw_pct.astype(float), atol=1e-6, equal_nan=True) & (J["T30_TP0.25_why"].astype(str) == J.raw_why.astype(str))
    print(f"PARITY vs WILLY_DURATION_cell_rows: matched {len(J)}/{len(S)} · plain exit {ok1.sum()}/{len(J)} · today late-TP walk {ok2.sum()}/{len(J)}")
    assert ok1.mean() >= 0.999 and ok2.mean() >= 0.999, "walker parity broken"
    cols = ["block", "seed", "era", "pair", "opened_at", "te", "E", "day", "month", "top50", "entry_frenzy_vs_vwap_pct", "entry_frenzy_run_pct",
            "pnl_percentage", "pct_new", "src", "tick_priced", "plain_why", "plain_pct", "raw_why", "raw_pct", "why", "xmin", "y", "mod_why", "y_mod"]
    MM["seed"] = 0; MM["era"] = MM["era"].astype(str); MM["day"] = pd.to_datetime(MM.te, unit="ms").dt.strftime("%Y-%m-%d")
    MM["month"] = MM.day.str[:7]; MM["top50"] = True; MM["pct_new"] = np.nan
    out = pd.concat([B.reindex(columns=cols), MM.reindex(columns=cols + ["pnl"])], ignore_index=True)
    out.to_csv(OUT_ROWS, index=False)
    print("wrote", OUT_ROWS, len(out))


if __name__ == "__main__":
    main()
