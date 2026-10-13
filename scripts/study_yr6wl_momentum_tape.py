"""YR6 WL MOMENTUM — step 7: tape context per regime (sleeve-kill checklist ④) and a MONTH-unit regime read.
Day-level medians of every macro stamp on MOM-long fill days per regime, plus the month-level (22 windows) Spearman
correlation of the sleeve's monthly avg % with each month's median macro stamp, and its sign inside each half.

    venv/bin/python scripts/study_yr6wl_momentum_tape.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_momentum_common import REGIMES, is_macro, load_replay, signal_columns

for SLV in ["MOM-long", "MOM-short"]:
    d = load_replay([SLV]); d["mon"] = d.ts.dt.strftime("%Y-%m")
    mac = [c for c in signal_columns(d) if is_macro(c) and not c.startswith("cat__") and pd.to_numeric(d[c], errors="coerce").notna().mean() > 0.8]
    day = d.groupby(["regime", "day"]).agg(**{c: (c, "median") for c in mac}, y=("y", "mean"), n=("y", "size"))
    print(f"\n## {SLV} — day-level medians of macro stamps on fill days (sleeve outcome per regime: "
          + " · ".join(f"{rg} {d[d.regime == rg].y.mean():+.3f}" for rg in REGIMES) + ")")
    t = day.groupby("regime")[mac + ["n"]].median().T[REGIMES]
    print(t.to_string(float_format=lambda v: f"{v:+.3f}"))
    mo = d.groupby("mon").agg(**{c: (c, "median") for c in mac}, y=("y", "mean"), n=("y", "size"), regime=("regime", "first"))
    rk = lambda s: s.rank()
    rows = []
    for c in mac:
        r_all = np.corrcoef(rk(mo[c]), rk(mo.y))[0, 1]
        a = mo[mo.index < "2026-01"]; b = mo[mo.index >= "2026-01"]
        rows.append((c, r_all, np.corrcoef(rk(a[c]), rk(a.y))[0, 1], np.corrcoef(rk(b[c]), rk(b.y))[0, 1]))
    r = pd.DataFrame(rows, columns=["macro", "rho_22m", "rho_2025(12m)", "rho_2026(10m)"]).sort_values("rho_22m", key=np.abs, ascending=False)
    r["same_sign"] = np.sign(r["rho_2025(12m)"]) == np.sign(r["rho_2026(10m)"])
    print(f"   month-unit Spearman (22 windows; |rho| > 0.42 ≈ p<0.05 at n=22; within halves |rho| > 0.58 / 0.64):")
    print(r.head(12).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
