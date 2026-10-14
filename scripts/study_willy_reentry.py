#!/usr/bin/env python3
"""WILLY re-entry after a good / bad exit (operator 2026-10-14). Today's WILLY exit +1.25 / −4 / 90 min, 10×.
① HYPOTHETICAL immediate re-entry: right after each WILLY fill closes, open a new WILLY long on the same pair at the first print after the
   exit (same exit rule), walked on the pair's own path (ticks → 1m tail; scout_frenzy_willy_exit.load_path / evaluate). Split by the
   first fill's exit: after TP (good) · after STOP (bad) · after CAP. Variants: re-enter at once, or after a 5 / 15 min wait.
② NATURAL repeats: WILLY fills on the same pair within 6 h of a previous WILLY fill (the replay / live already re-entered), by the
   previous fill's outcome.
Cohorts: yr6 2026 replay seed 1 (reports/ENGINE_REPLAY_YR6_fills.csv) and the master WILLY fills (B19 / B20 + current export).
Output reports/WILLY_REENTRY.txt + _rows.csv."""
import os, sys, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_timecap as WT  # noqa: E402
import scout_frenzy_willy_exit as FX  # noqa: E402
TP, SL, CAP, MIN = 1.25, 4.0, 90, WT.MIN
WAITS = (0, 5, 15)
rng = np.random.default_rng(7)


def job(rows):
    WT._NET_BLOCKED = True
    tc, out = {}, []
    for r in rows:
        try:
            P0 = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), CAP, tc)
            b = FX.evaluate(P0, TP, SL, CAP)
        except Exception:
            continue
        if b["why"] == "UNRES":
            continue
        rec = dict(r, first_why=b["why"], first_pct=b["pct"], first_min=b["xmin"])
        t_exit = int(r["te"] + b["xmin"] * MIN)
        for w in WAITS:
            t1 = t_exit + w * MIN
            try:
                tt, pp = None, None
                P = FX.load_path(r["pair"], t1, 1.0, CAP + 1, tc)   # E placeholder: need the price at t1 first
                # entry price = the first print / bar open after t1: reconstruct from the net path at E = 1 (net = (p/1 − 1)·100 − fees)
                if not len(P["t"]):
                    rec[f"re{w}"] = np.nan; continue
                k = 0
                cl = P["op"][k] if P["bar"][k] and np.isfinite(P["op"][k]) else P["cl"][k]
                E1 = (float(cl) + 100 + WT.TAKER) / (100 - WT.TAKER)   # exact inverse of WT.net_pct at E = 1: the re-entry price
                P1 = FX.load_path(r["pair"], int(P["t"][k]), float(E1), CAP, tc)
                v = FX.evaluate(P1, TP, SL, CAP)
                rec[f"re{w}"] = v["pct"]; rec[f"re{w}_why"] = v["why"]
            except Exception:
                rec[f"re{w}"] = np.nan
        out.append(rec)
    return out


def main():
    F = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YR6_fills.csv"), low_memory=False)
    F = F[(F.sleeve == "FRENZY_WILLY") & (F.seed == 1) & (F.year.astype(str) == "2026")].copy()
    F["t"] = pd.to_datetime(F.t, format="mixed")
    items = [dict(cohort="BT 2026", pair=r.pair, te=int(pd.Timestamp(r.t).value // 10**6), E=float(r.entry_price), ts=str(r.t)[:16], day=str(r.t)[:10])
             for r in F.itertuples()]
    o, _ = WT.load_orders()
    o = o[(o.ts >= pd.Timestamp("2026-10-08 18:30")) & ~o.why.astype(str).str.upper().str.startswith("MANUAL")]
    items += [dict(cohort="MASTER", pair=r.pair, te=int(r.te), E=float(r.E), ts=str(r.ts)[:16], day=r.day) for r in o.itertuples()]
    ps = sorted({i["pair"] for i in items}); g = {p: k % 6 for k, p in enumerate(ps)}
    with Pool(6) as P:
        R = pd.DataFrame(sum(P.map(job, [[i for i in items if g[i["pair"]] == k] for k in range(6)]), []))
    R.to_csv(os.path.join(ROOT, "reports", "WILLY_REENTRY_rows.csv"), index=False)
    L = [__doc__.splitlines()[0], "", "① HYPOTHETICAL re-entry on the same pair after each WILLY exit (same exit rule) — N · WR · avg % net per re-entry"]
    for coh, D in R.groupby("cohort"):
        L.append(f"■ {coh}: {len(D)} first fills · first exits {D.first_why.value_counts().to_dict()} · first avg {D.first_pct.mean():+.3f}")
        for fw in ("TP", "STOP", "CAP"):
            d = D[D.first_why == fw]
            if not len(d):
                continue
            parts = []
            for w in WAITS:
                v = d[f"re{w}"].dropna()
                if len(v):
                    gg = pd.DataFrame(dict(v=v, day=d.loc[v.index, "day"])).groupby("day").v.agg(["sum", "size"])
                    idx = rng.integers(0, len(gg), size=(4000, len(gg))) if len(gg) > 2 else None
                    p = float(((gg["sum"].values[idx].sum(1) / gg["size"].values[idx].sum(1)) > 0).mean()) if idx is not None else float("nan")
                    parts.append(f"wait {w:2} min: {len(v):4} · {(v > 0).mean() * 100:3.0f} % · {v.mean():+.3f} (P>0 {p:.2f})")
            L.append(f"   after {fw:4} ({len(d):4}): " + " | ".join(parts))
    L += ["", "② NATURAL repeats (same pair, previous WILLY fill closed ≤ 6 h before this one's open) — by the previous fill's outcome"]
    for coh, D in R.groupby("cohort"):
        D = D.sort_values(["pair", "te"]).copy()
        D["prev_why"] = D.groupby("pair").first_why.shift(); D["prev_end"] = D.groupby("pair").apply(lambda x: (x.te + x.first_min * MIN).shift()).reset_index(level=0, drop=True)
        rep = D[D.prev_why.notna() & ((D.te - D.prev_end) <= 6 * 3600 * 1000) & ((D.te - D.prev_end) >= 0)]
        L.append(f"■ {coh}: {len(rep)} repeats of {len(D)} fills")
        for fw, d in rep.groupby("prev_why"):
            L.append(f"   after a {fw:4}: {len(d):3} · WR {(d.first_pct > 0).mean() * 100:3.0f} % · avg {d.first_pct.mean():+.3f}  (all WILLY avg {D.first_pct.mean():+.3f})")
    txt = "\n".join(L); open(os.path.join(ROOT, "reports", "WILLY_REENTRY.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
