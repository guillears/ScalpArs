#!/usr/bin/env python3
"""FRENZY_LITE durations + late take-profit / time cap (operator 2026-10-14, the WILLY 30-min analog). Today's LITE exit: +3 / −3 / 12 h
at 10×. Variants: LTx_z = TP +3 until minute x, then +z (x 30 / 60 / 120 / 240, z 0.5 / 1 / 1.5 / 2); Cx = close at market at minute x.
Cohorts walked on their OWN path (ticks → 1m tail, scout_frenzy_willy_exit.load_path; same fill rules as FX.evaluate):
  ③ BACKTEST yr6 2026 replay seed 1 (reports/ENGINE_REPLAY_YR6_fills.csv) and yr6a 2025 seed 1 (2025 non-top-50 pairs exit on 1m bars)
  ① master kept LITE fills B18–B20 + B21 export ($/pt from stack_pnl / stack_pct; B21 its own pnl / pct).
A fill whose base exit is ≤ minute x is identical by construction (asserted). Output reports/LITE_LATE_TP.txt + _rows.csv."""
import os, sys, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_timecap as WT  # noqa: E402
import scout_frenzy_willy_exit as FX  # noqa: E402
TP, SL, CAP, MIN = 3.0, 3.0, 720, WT.MIN
XS, ZS = (30, 60, 120, 240), (0.5, 1.0, 1.5, 2.0)
VARS = {f"LT{x}_{z:g}": (x, z) for x in XS for z in ZS}
VARS.update({f"C{x}": (x, None) for x in XS})
rng = np.random.default_rng(7)


def ev(P, x=None, z=None, cap=CAP):
    te = P["te"]; n = int(np.searchsorted(P["t"], te + cap * MIN, side="right"))
    t, hi, lo = P["t"][:n], P["hi"][:n], P["lo"][:n]
    late = (t > te + x * MIN) if (x is not None and z is not None) else np.zeros(n, bool)
    tpl = np.where(late, z if z is not None else TP, TP)
    a = np.flatnonzero(hi >= tpl); b = np.flatnonzero(lo <= -SL)
    a = int(a[0]) if len(a) else None; b = int(b[0]) if len(b) else None
    if b is not None and (a is None or b <= a):
        op = P["op"][b]
        v = (min(-SL, float(op)) if np.isfinite(op) else -SL) if P["bar"][b] else float(lo[b])
        return dict(why="STOP", pct=v, xmin=(t[b] - te) / MIN, trough=float(lo[:b + 1].min()))
    if a is not None:
        v = float(tpl[a]) if P["bar"][a] else float(hi[a])
        return dict(why="LATE_TP" if late[a] else "TP", pct=v, xmin=(t[a] - te) / MIN, trough=float(lo[:a + 1].min()))
    if P["end_ms"] < te + cap * MIN or n == 0:
        return dict(why="UNRES", pct=np.nan, xmin=np.nan, trough=np.nan)
    return dict(why="CAP", pct=float(P["cl"][n - 1]), xmin=float(cap), trough=float(lo.min()))


def job(rows):
    WT._NET_BLOCKED = True
    tc, out = {}, []
    for r in rows:
        try:
            P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), CAP, tc)
        except Exception as e:  # noqa: BLE001
            out.append(dict(r, err=str(e)[:60])); continue
        b = ev(P); rec = dict(r, src=FX.path_src(P), base=b["pct"], base_why=b["why"], base_min=b["xmin"], base_trough=b["trough"])
        for k, (x, z) in VARS.items():
            v = ev(P, x, z) if z is not None else ev(P, cap=x)
            rec[k] = v["pct"]
            if z is not None and b["why"] != "UNRES" and b["xmin"] <= x and v["why"] != "UNRES":
                assert abs(v["pct"] - b["pct"]) < 1e-9
        out.append(rec)
    return out


def day_p(d, day):
    g = pd.DataFrame(dict(d=d, day=day)).groupby("day").d.agg(["sum", "size"])
    if len(g) < 3:
        return float("nan")
    idx = rng.integers(0, len(g), size=(4000, len(g))); s, n = g["sum"].values, g["size"].values
    return float(((s[idx].sum(1) / n[idx].sum(1)) > 0).mean())


def main():
    items = []
    for fn, coh in (("ENGINE_REPLAY_YR6_fills.csv", "BT 2026"), ("ENGINE_REPLAY_YR6_2025_fills.csv", "BT 2025")):
        F = pd.read_csv(os.path.join(ROOT, "reports", fn), low_memory=False)
        F = F[(F.sleeve == "FRENZY_LITE") & (F.seed == 1)].copy(); F["t"] = pd.to_datetime(F.t, format="mixed")
        for r in F.itertuples():
            items.append(dict(cohort=coh, pair=r.pair, te=int(pd.Timestamp(r.t).value // 10**6), E=float(r.entry_price), ts=str(r.t)[:16],
                              day=str(r.t)[:10], live=float(r.pct), upp=float(r.notional_fixed) / 100 * 5 if False else np.nan))
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    M = M[(M.entry_strategy == "FRENZY_LITE") & (M.stack_keep == True)]   # noqa: E712
    for r in M.itertuples():
        t = pd.Timestamp(str(r.opened_at)[:19])
        items.append(dict(cohort=f"M {r.era}", pair=r.pair, te=int(t.value // 10**6), E=float(r.entry_price), ts=str(t)[:16], day=str(t)[:10],
                          live=float(r.stack_pct), upp=float(r.stack_pnl) / float(r.stack_pct) if r.stack_pct else np.nan))
    E = sorted([f for f in os.listdir(os.path.expanduser("~/Downloads")) if f.startswith("scalpars_orders_paper_2026-10-1")])[-1]
    X = pd.read_csv(os.path.expanduser("~/Downloads/" + E), low_memory=False)
    X = X[(X.entry_strategy == "FRENZY_LITE") & (X.status == "CLOSED") & (X.opened_at.astype(str) >= "2026-10-12T16")]
    for r in X.itertuples():
        t = pd.Timestamp(str(r.opened_at)[:19])
        items.append(dict(cohort="M B21", pair=r.pair, te=int(t.value // 10**6), E=float(r.entry_price), ts=str(t)[:16], day=str(t)[:10],
                          live=float(r.pnl_percentage), upp=float(r.pnl) / float(r.pnl_percentage)))
    if "--fetch" in sys.argv:
        o = pd.DataFrame([dict(pair=i["pair"], ts=pd.Timestamp(i["ts"]), te=i["te"], src=None) for i in items if i["cohort"].startswith("M")])
        WT._NET_BLOCKED = False; print("fetch:", WT.ensure_data(o, CAP, int(time.time() * 1000), fetch=True))
    ps = sorted({i["pair"] for i in items}); g = {p: k % 6 for k, p in enumerate(ps)}
    with Pool(6) as P:
        R = pd.DataFrame(sum(P.map(job, [[i for i in items if g[i["pair"]] == k] for k in range(6)]), []))
    R.to_csv(os.path.join(ROOT, "reports", "LITE_LATE_TP_rows.csv"), index=False)
    L = [__doc__.splitlines()[0], ""]
    for coh in ("BT 2026", "BT 2025"):
        D = R[(R.cohort == coh) & R.base.notna()]
        L.append(f"■ {coh}: {len(D)} fills walked · parity vs replay pct (|Δ| < 0.10): {((D.base - D.live).abs() < 0.10).mean() * 100:.0f} % · src {D.src.value_counts().to_dict()}")
        for k, gg in D.groupby("base_why"):
            L.append(f"   {k:5} {len(gg):4} ({len(gg) / len(D) * 100:4.1f} %) · minutes mean {gg.base_min.mean():6.1f} / median {gg.base_min.median():6.1f} · trough median {gg.base_trough.median():+.2f} · avg {gg.base.mean():+.2f}")
        W = D[D.base_why == "TP"]
        L.append("   winners minutes-to-TP: " + " · ".join(f"≤ {m}: {(W.base_min <= m).mean() * 100:.0f} %" for m in (15, 30, 60, 120, 240, 480, 720)))
        S = D[D.base_why == "STOP"]
        L.append("   losers minutes-to-stop: " + " · ".join(f"≤ {m}: {(S.base_min <= m).mean() * 100:.0f} %" for m in (15, 30, 60, 120, 240, 480, 720)))
        L.append(f"   base avg {D.base.mean():+.3f} %/fill · variant: avg · Δ · P(Δ>0) · H1/H2 Δ (2026 only)")
        rows = []
        for k in VARS:
            v = D[k]; ok = v.notna(); d = (v - D.base)[ok]
            rows.append((k, v[ok].mean(), d.mean(), day_p(d.values, D.day[ok].values)))
        for k, a, dm, p in sorted(rows, key=lambda x: -x[2]):
            L.append(f"   {k:9} {a:+.3f} · Δ {dm:+.3f} · P {p:.2f}")
        L.append("")
    D = R[R.cohort.str.startswith("M") & R.base.notna()]
    L.append(f"■ MASTER LITE (B18–B21) own paths: {len(D)} fills · src {D.src.value_counts().to_dict()}")
    for r in D.itertuples():
        L.append(f"   {r.cohort[2:]} {r.ts} {r.pair:10} live {r.live:+.2f} base {r.base:+.2f} {r.base_why} @ {r.base_min:.0f} min · trough {r.base_trough:+.2f}")
    L.append("   per batch $ at today's size — TODAY vs top variants:")
    keys = ["base"] + [k for k in ("LT60_1.5", "LT120_1.5", "LT120_1", "LT240_1.5", "LT30_1.5", "C120", "C240")]
    for e, gg in list(D.groupby("cohort")) + [("TOTAL", D)]:
        L.append(f"   {e:7} " + " · ".join(f"{k} ${np.nansum(gg[k] * gg.upp):+,.0f}" for k in keys))
    txt = "\n".join(L); open(os.path.join(ROOT, "reports", "LITE_LATE_TP.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
