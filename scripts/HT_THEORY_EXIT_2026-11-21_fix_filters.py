#!/usr/bin/env python3
"""HT_THEORY_EXIT — CORRECTION (operator 11-21): today's HT entry filters = dip ≥ 1.4 % (DL 594) AND WILLY_HT_FAILED_SPIKE
(entry_frenzy_vs_vwap_pct ≤ 0 blocked, watch_block_willy_ht_failed_spike true, DL 527). The first eval applied the dip only.
Re-reads every HT cell on master and current batch with both filters. → reports/HT_THEORY_EXIT_2026-11-21_cells_fixed.csv"""
import glob, importlib.util, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sp = importlib.util.spec_from_file_location("HTE", os.path.join(HERE, "HT_THEORY_EXIT_2026-11-21_eval.py")); HTE = importlib.util.module_from_spec(sp); sp.loader.exec_module(HTE)
O = pd.read_pickle(os.path.join(HTE.D, "sim.pkl")); O = O[O.REF.notna() & (O.sleeve == "HT") & (O.kind == "REAL")].copy()
parts = []
for f in sorted(glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_*.csv"))):
    h = pd.read_csv(f, nrows=0).columns
    if "entry_frenzy_vs_vwap_pct" not in h:
        continue
    x = pd.read_csv(f, usecols=["opened_at", "pair", "entry_strategy", "entry_frenzy_vs_vwap_pct"], low_memory=False)
    parts.append(x[x.entry_strategy == "FRENZY_WILLY_HT"].assign(src=f))
V = pd.concat(parts); V["k"] = V.opened_at.astype(str).str[:19]; V = V.sort_values("src").drop_duplicates(["k", "pair"], keep="last")
O = O.merge(V[["k", "pair", "entry_frenzy_vs_vwap_pct"]], on=["k", "pair"], how="left")
assert O.entry_frenzy_vs_vwap_pct.notna().all()
O["keep2"] = O.today_keep.astype(bool) & (O.entry_frenzy_vs_vwap_pct > 0)
rows = []
for cname, G in (("MASTER_today", O[(O.period == "MASTER") & O.keep2]), ("BATCH_today", O[(O.period == "BATCH") & O.keep2])):
    for cell in ["REF"] + HTE.HTS.HT_CELLS + HTE.HTS.HT_FIXED:
        r = HTE.read(G, cell)
        if r is not None:
            rows.append(dict(cohort=cname, cell=cell, **r))
T = pd.DataFrame(rows); T.to_csv(os.path.join(ROOT, "reports", "HT_THEORY_EXIT_2026-11-21_cells_fixed.csv"), index=False)
pd.set_option("display.width", 250)
for c in ("MASTER_today", "BATCH_today"):
    t = T[T.cohort == c].sort_values("d", ascending=False)
    print("\n==", c, "N", int(t.N.iloc[0]))
    print(t[["cell", "fired", "WR", "avg", "d", "P", "saved", "cut", "usd", "dcr_b", "dcr_a", "avg_loss", "loo"]].round(2).to_string(index=False))
