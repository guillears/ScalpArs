#!/usr/bin/env python3
"""WILLY: TP +0.3 or close at 30 min vs today's +1.25 / −4 / 90 (operator 2026-10-13, next-batch item) — master WILLY fills (B19, B20)
+ the current batch (B21 export), each walked on its own path (ticks → 1m tail, scout_frenzy_willy_exit.load_path; --fetch allows the
budgeted 1m / archive fetch for days not cached). $ at today's size (10×): master $/pt = stack_pnl / stack_pct; B21 = its own pnl / pct.
Output reports/WILLY_TP03_CAP30_MASTER.txt"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_timecap as WT  # noqa: E402
import scout_frenzy_willy_exit as FX  # noqa: E402
import study_willy_late_sl as S  # noqa: E402
VARS = {"TODAY +1.25/−4/90": lambda P: S.ev(P),
        "TP +0.3 from entry": lambda P: S.ev(P, 0, None, 0.3),
        "TP +0.3 after 30 min": lambda P: S.ev(P, 30, None, 0.3),
        "TP +0.25 after 30 min": lambda P: S.ev(P, 30, None, 0.25),
        "close at 30 min": lambda P: FX.evaluate(P, 1.25, 4.0, 30)}
ERAS = [("B19", "2026-10-08 18:30", "2026-10-12 03:34"), ("B20", "2026-10-12 05:00", "2026-10-12 16:00"), ("B21", "2026-10-12 16:00", "2099-01-01")]


def main():
    fetch = "--fetch" in sys.argv
    o, _ = WT.load_orders()
    o["era"] = None
    for e, a, b in ERAS:
        o.loc[(o.ts >= pd.Timestamp(a)) & (o.ts < pd.Timestamp(b)), "era"] = e
    o = o[o.era.notna() & ~o.why.astype(str).str.upper().str.startswith("MANUAL")].reset_index(drop=True)
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    M = M[(M.entry_strategy == "FRENZY_WILLY") & (M.stack_keep == True)]   # noqa: E712
    upp = {(str(a)[:19].replace("T", " "), p): s / c for a, p, s, c in zip(M.opened_at, M.pair, M.stack_pnl, M.stack_pct) if c}
    o["upp"] = [upp.get((str(t)[:19], p), (u / q if q else np.nan)) for t, p, u, q in zip(o.ts, o.pair, o.usd, o.pct)]
    if fetch:
        o["src"] = None; WT._NET_BLOCKED = False
        print("fetch:", WT.ensure_data(o, 90, int(time.time() * 1000), fetch=True))
    WT._NET_BLOCKED = True
    tc, rows = {}, []
    for r in o.itertuples():
        P = FX.load_path(r.pair, int(r.te), float(r.E), 90, tc)
        rec = dict(era=r.era, ts=str(r.ts)[:16], pair=r.pair, live=r.pct, live_why=r.why, upp=r.upp, src=FX.path_src(P))
        for k, f in VARS.items():
            v = f(P); rec[k] = v["pct"]; rec[k + "|why"] = v["why"]
        rows.append(rec)
    R = pd.DataFrame(rows)
    L = [__doc__.splitlines()[0], ""]
    L.append("Per fill (net %): " + " | ".join(VARS))
    for r in R.itertuples():
        d = r._asdict()
        L.append(f"  {r.era} {r.ts} {r.pair:10} [{r.src}] live {r.live:+.2f} ({str(r.live_why)[:12]}) → " +
                 " · ".join(f"{d[k.replace(' ', '_')] if False else R.loc[r.Index, k]:+.2f} {R.loc[r.Index, k + '|why']}" for k in VARS))
    L += ["", "Per batch: N · WR · avg % · $ at today's size (UNRES fills excluded and counted)"]
    for e, g in list(R.groupby("era")) + [("TOTAL", R)]:
        parts = []
        for k in VARS:
            v = g[k]; ok = v.notna()
            parts.append(f"{k}: {ok.sum()} · {(v[ok] > 0).mean() * 100:3.0f} % · {v[ok].mean():+.3f} · ${np.nansum(v[ok] * g.upp[ok]):+,.0f}" + (f" ({(~ok).sum()} unres)" if (~ok).sum() else ""))
        L.append(f"  {e}:\n    " + "\n    ".join(parts))
    txt = "\n".join(L); open(os.path.join(ROOT, "reports", "WILLY_TP03_CAP30_MASTER.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
