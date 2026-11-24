#!/usr/bin/env python3
"""EXH before → after on the real kept fills (today's filters + exits; BATCH_TODAY_RULES_2026-11-23 — every real EXH fill to date,
so master = batch), the observe-line rule: at the TP (7 %) touch no TP; trail k × 15m ATR14 (at the touch) from the running low;
NO floor, the stop (1.5 × ATR ≤ 6 %) kept; 24 h from the touch; taker 0.05 + slip 0.10; funding while held. One size X 0.61 × $10k."""
import os, sys, glob, importlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
BA = importlib.import_module("TP_RUNNER_2026-11-24_beforeafter"); DV = importlib.import_module("TP_RUNNER_DIVE_2026-11-24")
T, C, DP = DV.T, DV.C, DV.DP
MIN, X, ACC = 60_000, 0.61, 10_000
SPECS = {"4×ATR · no floor · stop kept (observe line)": 4, "5×ATR · no floor · stop kept": 5}


def main():
    # master check: every real EXH fill across all exports (dedup) vs the batch file
    n = set()
    for f in glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_*.csv")):
        h = pd.read_csv(f, nrows=0).columns
        if "entry_strategy" not in h: continue
        x = pd.read_csv(f, low_memory=False, usecols=["opened_at", "pair", "entry_strategy", "status"])
        n |= {(str(a)[:16], p) for a, p in zip(x[(x.entry_strategy == "EXHAUSTION_SHORT") & (x.status == "CLOSED")].opened_at, x[(x.entry_strategy == "EXHAUSTION_SHORT") & (x.status == "CLOSED")].pair)}
    F = pd.read_csv(os.path.join(ROOT, "reports", "BATCH_TODAY_RULES_2026-11-23_fills.csv"))
    print("real closed EXH fills in all exports:", len(n), "· first", min(n)[0] if n else "-",
          "· batch-file EXH rows", int((F.sleeve == "EXH").sum()), "kept", int(((F.sleeve == "EXH") & F.kept.astype(bool)).sum()))
    F = F[(F.sleeve == "EXH") & F.kept.astype(bool)].copy()
    ex = pd.read_csv(os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-11-23_22-20-57.csv"), low_memory=False)
    ex["om"] = ex.opened_at.astype(str).str[:16]; F["om"] = F.opened_at.astype(str).str[:16]
    F = F.merge(ex[["om", "pair", "entry_price", "exit_price"]].drop_duplicates(["om", "pair"]), on=["om", "pair"], how="left")
    F["day"] = pd.to_datetime(F.today_closed_at, utc=True, format="mixed").dt.strftime("%Y-%m-%d")
    for c in SPECS: F[c] = F.today_pct; F[c + "_why"] = ""
    tpr = F[F.today_reason.astype(str).isin(["TP", "FRENZY_TP"])]
    T.fetch(pd.DataFrame(dict(pair=tpr.pair, t_touch=[BA.MS(a if isinstance(a, str) else b) for a, b in zip(tpr.today_closed_at, tpr.closed_at)])))
    for i, r in F.iterrows():
        if str(r.today_reason) not in ("TP", "FRENZY_TP") or not np.isfinite(r.entry_price): continue
        E = float(r.entry_price)
        if np.isfinite(r.today_tp): tp = E * (1 - float(r.today_tp) / 100); tt = BA.MS(r.today_closed_at)
        else: tp = float(r.exit_price); tt = BA.MS(r.closed_at)
        atr = T.atr15_pct(r.pair, tt // MIN * MIN); p = DP.path(r.pair, tt)
        if p is None: print("  no 1m path", r.pair, r.today_closed_at); continue
        try: C.fetch_funding(r.pair, sorted({r.day[:7], (pd.Timestamp(r.day) + pd.Timedelta(days=2)).strftime("%Y-%m")}))
        except SystemExit: pass
        sl = E * (1 + min(1.5 * atr, 6) / 100)
        for nm, k in SPECS.items():
            px, j, why = DV.trail_ix(p, tp, -1, k * atr, sl); px *= 1.001
            fund = DV.fund_between(r.pair, tt, tt + (j + 1) * MIN, E, px, -1)
            F.at[i, nm] = (1 - px / E) * 100 - 0.10 + fund; F.at[i, nm + "_why"] = {"floor": "stop"}.get(why, why)
    F.to_csv(os.path.join(T.OUT, "exh_ba_observe.csv"), index=False)
    days = pd.date_range("2026-11-18", "2026-11-23").strftime("%Y-%m-%d")
    tp = F[F.today_reason.astype(str).isin(["TP", "FRENZY_TP"])]
    print(f"\nEXH: {len(F)} kept fills · {len(days)} days · exits today {F.today_reason.value_counts().to_dict()}")
    for c, nm in [("today_pct", "BEFORE · TP 7 (today)")] + [(k, "AFTER · " + k) for k in SPECS]:
        x = F[c]; srt = np.sort(x.values)[::-1]
        print(f"  {nm:46s} N {len(x):2d} · WR {100*(x>0).mean():3.0f}% · avg {x.mean():+5.2f}% · ${(x*X*ACC/100).sum():+7,.0f} · "
              f"DCR {BA.dcr(F, c, days):+5.2f}%/day · w/o best fill avg {srt[1:].mean():+5.2f}%")
        if c != "today_pct":
            d = tp[c] - tp.today_pct
            print(f"  {'':46s} per TP hit Δ {d.mean():+.2f} pp · median {d.median():+.2f} · better {100*(d>0).mean():.0f}% · exits {tp[c+'_why'].value_counts().to_dict()} · best Δ {d.max():+.1f} ({tp.loc[d.idxmax(),'pair']}) · w/o best Δ {d.drop(d.idxmax()).mean():+.2f}")


if __name__ == "__main__":
    main()
