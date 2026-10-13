#!/usr/bin/env python3
"""WILLY LATE STOP study (operator 2026-10-13): "a new SL after X minutes, like spike fade".

Today's WILLY exit: +1.25 TP / −4 stop / 90-min cap, 10× (DECISION_LOG 281). Variant LATE_SL(X, Y): until minute X the stop is −4; from
minute X on it is −Y (if the fill already sits below −Y at minute X it closes on the first print after X). Secondary family LATE_TP(X, Z),
the spike-fade analog (DECISION_LOG 111 late-arm: bank a small bounce on a slow trade): from minute X the take profit drops to +Z.

Cohorts (each walked on its OWN path — ticks first, then the 1m tail; scout_frenzy_willy_exit.load_path / evaluate; this study's backtest parity is printed — 89 % on yr6 2026):
  ③ BACKTEST  yr6 2026 replay (reports/ENGINE_REPLAY_YR6_fills.csv), FRENZY_WILLY, seed 1 (the FRENZY seeds are near-identical paths), the
              replay already ran today's WILLY exit and 10× → baseline walk must reproduce the replay (parity line).
  ① B20 / B19 live fills from the orders exports (dedup, CLOSED), priced at TODAY's exit on their own path (B19 traded older exits) and at
              today's size ($/pt from the master pool's stack_pnl / stack_pct).
Two-sided by construction: the same path for base and variant; a fill that exits before minute X must show Δ = 0 (asserted).
Usage: venv/bin/python scripts/study_willy_late_sl.py [--fetch]   (--fetch: allow the scout's budgeted archive / 1m fetch for B20 days)
Output: reports/WILLY_LATE_SL_STUDY.txt + reports/WILLY_LATE_SL_rows.csv
"""
import argparse, os, sys, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_timecap as WT  # noqa: E402
import scout_frenzy_willy_exit as FX  # noqa: E402

TP, SL, CAP = 1.25, 4.0, 90
MIN = WT.MIN
XS = (10, 15, 20, 30, 45, 60)
YS = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
TPZ = (0.25, 0.5)
VARS = {f"SL{x}_{y:g}": (x, y, None) for x in XS for y in YS}
VARS.update({f"TP{x}_{z:g}": (x, None, z) for x in (20, 30, 45, 60) for z in TPZ})
B19 = (pd.Timestamp("2026-10-08 18:30"), pd.Timestamp("2026-10-12 03:34"))
B20 = (pd.Timestamp("2026-10-12 05:00"), pd.Timestamp("2026-10-12 16:00"))
SPLIT = pd.Timestamp("2026-07-01")      # operator regimes: 2026-H1 bear · 2026-H2 bull
rng = np.random.default_rng(20261013)


def ev(P, x=None, y=None, z=None):
    """first-touch on the net path; stop −SL until minute x then −y; TP +TP until minute x then +z. Same fill rules as FX.evaluate."""
    te = P["te"]
    n = int(np.searchsorted(P["t"], te + CAP * MIN, side="right"))
    t, hi, lo = P["t"][:n], P["hi"][:n], P["lo"][:n]
    late = (t > te + x * MIN) if x is not None else np.zeros(n, bool)
    line = np.where(late & (y is not None), -(y if y is not None else SL), -SL)
    tpl = np.where(late & (z is not None), z if z is not None else TP, TP)
    itp = np.flatnonzero(hi >= tpl)
    ist = np.flatnonzero(lo <= line)
    a = int(itp[0]) if len(itp) else None
    b = int(ist[0]) if len(ist) else None
    if b is not None and (a is None or b <= a):
        if P["bar"][b]:
            op = P["op"][b]
            v = min(float(line[b]), float(op)) if np.isfinite(op) else float(line[b])
        else:
            v = float(lo[b])
        return dict(why="LATE_SL" if late[b] and y is not None and line[b] > -SL else "STOP", pct=v, xmin=(t[b] - te) / MIN,
                    trough=float(lo[:b + 1].min()))
    if a is not None:
        v = float(tpl[a]) if P["bar"][a] else float(hi[a])
        return dict(why="LATE_TP" if late[a] and z is not None else "TP", pct=v, xmin=(t[a] - te) / MIN, trough=float(lo[:a + 1].min()))
    if P["end_ms"] < te + CAP * MIN or n == 0:
        return dict(why="UNRES", pct=np.nan, xmin=np.nan, trough=np.nan)
    return dict(why="CAP", pct=float(P["cl"][n - 1]), xmin=float(CAP), trough=float(lo.min()))


def path_stats(P, base):
    """descriptives on the base path: trough before exit, trough AFTER minute m (m in XS) for a fill still open at m, pnl at minute m."""
    te = P["te"]; out = {}
    if base["why"] == "UNRES":
        return out
    end = te + base["xmin"] * MIN
    k = int(np.searchsorted(P["t"], end, side="right"))
    t, lo, cl = P["t"][:k], P["lo"][:k], P["cl"][:k]
    for m in XS:
        if base["xmin"] > m:
            j = int(np.searchsorted(t, te + m * MIN, side="right"))
            out[f"open_at_{m}"] = True
            out[f"pnl_at_{m}"] = float(cl[j - 1]) if j > 0 else 0.0
            out[f"trough_after_{m}"] = float(lo[j:].min()) if j < len(lo) else np.nan
        else:
            out[f"open_at_{m}"] = False
    return out


def job(rows):
    WT._NET_BLOCKED = True
    tc, out = {}, []
    for r in rows:
        try:
            P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), CAP, tc)
        except Exception as e:  # noqa: BLE001
            out.append(dict(r, err=str(e)[:80])); continue
        base = ev(P)
        rec = dict(r, src=FX.path_src(P), base_why=base["why"], base_pct=base["pct"], base_min=base["xmin"], base_trough=base["trough"])
        rec.update(path_stats(P, base))
        for name, (x, y, z) in VARS.items():
            v = ev(P, x, y, z)
            rec[name] = v["pct"]; rec[f"{name}_why"] = v["why"]
            if base["why"] != "UNRES" and base["xmin"] <= x and v["why"] != "UNRES":   # exits before X: identical by construction
                assert abs(v["pct"] - base["pct"]) < 1e-9, (r["pair"], name, v, base)
        out.append(rec)
    return out


def day_p(d, day):
    g = pd.DataFrame(dict(d=d, day=day)).groupby("day").d.agg(["sum", "size"])
    if len(g) < 3:
        return np.nan, (np.nan, np.nan)
    s, n = g["sum"].values, g["size"].values
    idx = rng.integers(0, len(g), size=(4000, len(g)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float((m > 0).mean()), tuple(np.percentile(m, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true"); ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--from-rows", action="store_true")
    A = ap.parse_args()
    # ③ backtest cohort
    F = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YR6_fills.csv"), low_memory=False)
    F = F[(F.sleeve == "FRENZY_WILLY") & (F.year.astype(str) == "2026")].copy()
    F["t"] = pd.to_datetime(F.t, format="mixed")
    allseeds = F.copy()
    F = F[F.seed == 1].reset_index(drop=True)
    F["te"] = F.t.values.astype("datetime64[ms]").astype("int64")
    bt = [dict(cohort="BACKTEST", pair=r.pair, te=int(r.te), E=float(r.entry_price), ts=str(r.t), day=str(r.t)[:10], live_pct=float(r.pct),
               live_why=str(r.close_reason), usd_per_pt=float(r.notional_fixed) / 100.0) for r in F.itertuples()]
    # ① live cohorts
    o, _ = WT.load_orders()
    o = o[((o.ts >= B19[0]) & (o.ts < B19[1])) | ((o.ts >= B20[0]) & (o.ts < B20[1]))].copy()
    o["cohort"] = np.where(o.ts >= B20[0], "B20", "B19")
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    M = M[(M.entry_strategy == "FRENZY_WILLY") & (M.stack_keep == True)]   # noqa: E712
    upp = {(str(a)[:19].replace("T", " "), p): (s / c if c else np.nan) for a, p, s, c in zip(M.opened_at, M.pair, M.stack_pnl, M.stack_pct)}
    if A.fetch:
        o["src"] = None
        WT._NET_BLOCKED = False
        notes = WT.ensure_data(o, CAP, int(time.time() * 1000), fetch=True)
        print("fetch notes:", notes)
    lv = [dict(cohort=r.cohort, pair=r.pair, te=int(r.te), E=float(r.E), ts=str(r.ts), day=r.day, live_pct=float(r.pct), live_why=r.why,
               usd_per_pt=upp.get((str(r.ts)[:19], r.pair), np.nan)) for r in o.itertuples()]
    items = bt + lv
    if A.from_rows:
        R = pd.read_csv(os.path.join(ROOT, "reports", "WILLY_LATE_SL_rows.csv"), low_memory=False)
        _lv = R.cohort != "BACKTEST"
        R.loc[_lv, "usd_per_pt"] = [upp.get((str(t)[:19], p), np.nan) for t, p in zip(R.ts[_lv], R.pair[_lv])]
    else:
        pairs = sorted({r["pair"] for r in items}); grp = {p: k % A.jobs for k, p in enumerate(pairs)}
        with Pool(A.jobs) as P:
            res = sum(P.map(job, [[r for r in items if grp[r["pair"]] == k] for k in range(A.jobs)]), [])
        R = pd.DataFrame(res)
        R.to_csv(os.path.join(ROOT, "reports", "WILLY_LATE_SL_rows.csv"), index=False)

    L = ["WILLY LATE STOP study (2026-10-13) — today's exit +1.25 / −4 / 90 min, 10×; variant SLx_y = stop −4 until minute x, then −y "
         "(y = 0 → exit on the first print ≤ 0 after x); TPx_z = take profit drops to +z after minute x (the spike-fade late-arm analog).",
         "Every cohort walked on its own path (ticks → 1m tail); a fill exiting before minute x is identical by construction (asserted).", ""]
    # ── descriptives (backtest, seed 1, base walk) ──
    B = R[(R.cohort == "BACKTEST") & (R.base_why != "UNRES") & R.base_pct.notna()].copy()
    par = (B.live_why.str.contains("TP") & (B.base_why == "TP")) | (B.live_why.str.contains("STOP|SL") & (B.base_why == "STOP")) | \
          (B.live_why.str.contains("HOLD|CAP|TIME") & (B.base_why == "CAP"))
    pv = (B.base_pct - B.live_pct).abs() < 0.10
    L.append(f"③ BACKTEST yr6 2026 seed 1: {len(R[R.cohort == 'BACKTEST'])} fills · walked {len(B)} · source "
             f"{B.src.value_counts().to_dict()} · parity vs the replay (same exit kind ∧ |Δ%| < 0.10): {(par & pv).mean() * 100:.0f} %")
    L.append("")
    L.append("Duration and trough (base walk) — minutes to exit · trough = worst net % before the exit")
    for k, g in B.groupby("base_why"):
        L.append(f"  {k:5} {len(g):4} ({len(g) / len(B) * 100:4.1f} %) · minutes mean {g.base_min.mean():5.1f} / median {g.base_min.median():5.1f}"
                 f" · trough mean {g.base_trough.mean():+.2f} / median {g.base_trough.median():+.2f} / p10 {g.base_trough.quantile(.1):+.2f}"
                 f" · avg % {g.base_pct.mean():+.2f}")
    W = B[B.base_why == "TP"]
    L.append("  Winners (TP) trough distribution: " + " · ".join(f"≤ {q:g}: {(W.base_trough <= q).mean() * 100:.0f} %" for q in (-0.5, -1, -1.5, -2, -2.5, -3, -3.5)))
    L.append("  Winners (TP) minutes-to-TP: " + " · ".join(f"≤ {m}: {(W.base_min <= m).mean() * 100:.0f} %" for m in XS + (90,)))
    _st = allseeds[allseeds.close_reason.astype(str).str.contains("TP")]
    _du = (pd.to_datetime(_st.closed_at.astype(str).str[:19]) - pd.to_datetime(_st.opened_at.astype(str).str[:19])).dt.total_seconds() / 60
    L.append(f"  cross-check (replay stamps, all 3 seeds, TP fills {len(_st)}): minutes mean {_du.mean():.1f} / median {_du.median():.1f} · "
             f"trough_pnl mean {pd.to_numeric(_st.trough_pnl, errors='coerce').mean():+.2f} / median {pd.to_numeric(_st.trough_pnl, errors='coerce').median():+.2f}")
    L.append("")
    L.append("Still open at minute m (base walk): how the fill ends, and the worst it goes AFTER m")
    for m in XS:
        g = B[B.get(f"open_at_{m}") == True]   # noqa: E712
        if not len(g):
            continue
        wv = g[g.base_why == "TP"]; lv_ = g[g.base_why == "STOP"]; cp = g[g.base_why == "CAP"]
        L.append(f"  m={m:>2}: open {len(g):4} ({len(g) / len(B) * 100:4.1f} %) → TP {len(wv)} · stop {len(lv_)} · cap {len(cp)} · avg % {g.base_pct.mean():+.2f}"
                 f" | winners: pnl@m median {wv[f'pnl_at_{m}'].median():+.2f}, worst after m median {wv[f'trough_after_{m}'].median():+.2f} / p10 "
                 f"{wv[f'trough_after_{m}'].quantile(.1):+.2f} | stopped: pnl@m median {lv_[f'pnl_at_{m}'].median():+.2f}")
    L.append("")

    def table(D, title, usd=True):
        D = D[(D.base_why != "UNRES") & D.base_pct.notna()]
        L.append(f"{title}: {len(D)} fills · base {(D.base_pct > 0).mean() * 100:.0f} % WR · avg {D.base_pct.mean():+.3f} % · Σ {D.base_pct.sum():+.1f} pts"
                 + (f" · ${(D.base_pct * D.usd_per_pt).sum():+,.0f}" if usd else ""))
        L.append(f"  {'variant':10} {'WR':>4} {'avg %':>7} {'Δavg':>7} {'ΣΔpts':>7} {'saved':>12} {'killed':>12} {'P(Δ>0)':>6} {'95% CI Δ':>17} "
                 f"{'H1 Δ':>6} {'H2 Δ':>6}" + (f" {'Δ$':>8}" if usd else ""))
        rows = []
        for name in VARS:
            v = D[name]; ok = v.notna()
            d = (v - D.base_pct)[ok]; dd = D[ok]
            sv = d > 1e-9; kl = d < -1e-9
            p, ci = day_p(d.values, dd.day.values)
            h1 = d[pd.to_datetime(dd.ts, format="mixed") < SPLIT].mean(); h2 = d[pd.to_datetime(dd.ts, format="mixed") >= SPLIT].mean()
            usd_d = (d * dd.usd_per_pt).sum()
            rows.append((name, (v[ok] > 0).mean() * 100, v[ok].mean(), d.mean(), d.sum(), sv.sum(), d[sv].sum(), kl.sum(), d[kl].sum(), p, ci, h1, h2, usd_d))
        rows.sort(key=lambda x: -x[4])
        for (name, wr, av, da, ds, ns_, ss, nk, sk, p, ci, h1, h2, ud) in rows:
            L.append(f"  {name:10} {wr:4.0f} {av:+7.3f} {da:+7.3f} {ds:+7.1f} {ns_:4}·{ss:+6.1f} {nk:4}·{sk:+6.1f} {p:6.2f} [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                     f"{h1:+6.3f} {h2:+6.3f}" + (f" {ud:+8,.0f}" if usd else ""))
        return rows

    rb = table(R[R.cohort == "BACKTEST"], "③ BACKTEST yr6 2026 (seed 1) — $ at 10× on a fixed $2,900 book (no compounding)")
    L.append("")
    for c in ("B20", "B19"):
        D = R[R.cohort == c]
        if len(D):
            L.append(f"① {c} fills (own path, today's exit, today's size): " + " · ".join(
                f"{r.ts[5:16]} {r.pair} live {r.live_pct:+.2f} ({r.live_why[:12]}) base {r.base_pct:+.2f} {r.base_why} [{r.src}]" for r in D.itertuples()))
            table(D, f"① {c}")
            L.append("")
    txt = "\n".join(L)
    open(os.path.join(ROOT, "reports", "WILLY_LATE_SL_STUDY.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
