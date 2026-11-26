#!/usr/bin/env python3
"""HT_FILTER_BT_2026-11-26 — WILLY_HT entry filter candidates on the 2-year backtest (research only; no bot / config change).
(operator 2026-11-26: "on HT, which filter has more chances to become a filter CROWDED_LONGS, or PRE_VOL_SURGE, or both together?" → "run the test")

PRIOR: DL 580 CROWDED_LONGS preview (real HT fills since 10-01: the zone held the WINNERS) · DL 596 PRE_VOL_SURGE (master 22 · −1.21 vs
+2.93, backtest had no volume → unchecked) · DL 618 / 619 BURST_REJECTION (= PRE_VOL_SURGE ∧ cloc15 < 0.50) armed. NEW: both variables on
the backtest; PRE_VOL_SURGE OUTSIDE the armed burst zone; the two together.

COHORT  reports/STATE_EXIT_2026-11-21_data/fills_out.csv kind BT, sleeve HT (today's HT filters bar burst), net = tp6 (TP +6 / SL −4 / 24 h,
        fees + slip + funding). Today's stack = minus the DL 619 burst zone (fail-open when unreadable, like the engine).
ZONES   (the frozen live definitions, scripts/scout_willy_ht_lines.py)
  CROWDED_LONGS  top-trader ACCOUNT L/S > 1.19 OR global ACCOUNT L/S > 1.0624 — data.binance.vision daily metrics
                 (count_toptrader_long_short_ratio / count_long_short_ratio), the last row with create_time ≤ fill − 1 min.
                 ⚠ vision create_time vs the API period end: DL 614 found vision OI one period fresher → a ≤ 5-min look-ahead risk is stated.
  PRE_VOL        pre_vol15_vs60 > 1.60 (services.frenzy.pre_vol15_vs60 on 1m BASE volume, bars closed before the fill minute)
  PRE_VOL_OUT    PRE_VOL ∧ NOT the burst zone (a burst that closed in the upper half — what is NOT already blocked)
  BOTH           CROWDED_LONGS ∧ PRE_VOL_OUT
READ    zone vs rest (on today's stack): N · WR · avg · per half-year · DESIGN (< 2025-07-01) / HOLDOUT · day-bootstrap P(zone mean < 0) ·
        luck = within-month label shuffle (2,000) of (rest − zone) · sleeve before → after if blocked: avg, DCR (one size 0.61), max DD, worst day.
        Locked bar for a FILTER: zone WR < HT breakeven 40 % ∧ P(mean<0) ≥ 0.95 ∧ ≥ 8 windows ∧ N ≥ 15 ∧ no day / pair ≥ 50 % of the loss.
Out: reports/HT_FILTER_BT_2026-11-26.csv (per fill) · _summary.txt"""
import io, os, sys, zipfile
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import importlib
C = importlib.import_module("STATE_EXIT_2026-11-21_common")
from services.frenzy import pre_vol15_vs60, cloc15
from services.watch_filters import ht_burst_state
D = os.path.join(R, "reports", "HT_FILTER_BT_2026-11-26_data"); MIN, DAY = 60_000, 86_400_000
HO = int(pd.Timestamp("2025-07-01").value // 10**6); SZ = 0.61
TOP_CUT, GLOB_CUT, PV_CUT = 1.19, 1.0624, 1.60


def acct_day(p, ds):
    f = os.path.join(D, p, ds + ".npz")
    if not os.path.exists(f):
        os.makedirs(os.path.dirname(f), exist_ok=True)
        z = C._get(f"{C.VIS}/daily/metrics/{p}/{p}-metrics-{ds}.zip")
        if not z:
            np.savez_compressed(f, t=np.array([], "int64"), top=np.array([]), glob=np.array([])); return None
        zz = zipfile.ZipFile(io.BytesIO(z)); x = pd.read_csv(zz.open(zz.namelist()[0]))
        t = (pd.to_datetime(x.create_time) - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
        g = lambda c: pd.to_numeric(x[c], errors="coerce").values.astype(float) if c in x else np.full(len(x), np.nan)
        np.savez_compressed(f, t=t.values.astype("int64"), top=g("count_toptrader_long_short_ratio"), glob=g("count_long_short_ratio"))
    z = np.load(f)
    return z if len(z["t"]) else None


def crowded(p, te):
    parts = [acct_day(p, ds) for ds in C.days_of(te - 2 * 3_600_000, te)]
    parts = [z for z in parts if z is not None]
    if not parts: return None, None, None
    t = np.concatenate([z["t"] for z in parts]); top = np.concatenate([z["top"] for z in parts]); gl = np.concatenate([z["glob"] for z in parts])
    m = t <= te - MIN
    if not m.any() or te - t[m].max() > 30 * MIN: return None, None, None
    i = np.nonzero(m)[0][np.argmax(t[m])]
    a, b = top[i], gl[i]
    if not (np.isfinite(a) or np.isfinite(b)): return None, None, None
    return (bool((np.isfinite(a) and a > TOP_CUT) or (np.isfinite(b) and b > GLOB_CUT))), a, b


def prevol(p, te):
    m0 = te // MIN * MIN
    a = C.k1m(p, m0 - 80 * MIN, m0)
    if a is None or not len(a["t"]): return None, None
    rows = [[int(t), o, h, l, c, v] for t, o, h, l, c, v in zip(a["t"], a["o"], a["h"], a["l"], a["c"], a["v"]) if int(t) < m0]
    return pre_vol15_vs60(rows, te), cloc15(rows, te)


def boot_p(z, n=2000, seed=1126):
    """P(zone mean < 0), day-block bootstrap (resample days)."""
    if len(z) < 3: return np.nan
    g = z.groupby("day").net.agg(["sum", "count"]); rng = np.random.default_rng(seed); out = 0
    for _ in range(n):
        s = g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
        out += (s["sum"].sum() / s["count"].sum()) < 0
    return out / n


def luck_p(A, col, n=2000, seed=1126):
    """within-month shuffle of the zone label → P(rest − zone ≥ observed)."""
    k = A[col].astype(bool).values
    if k.sum() < 3 or (~k).sum() < 3: return np.nan
    obs = A.net[~k].mean() - A.net[k].mean(); rng = np.random.default_rng(seed); cnt = 0
    mo = A.month.values; net = A.net.values
    for _ in range(n):
        kk = k.copy()
        for m in np.unique(mo):
            ii = np.nonzero(mo == m)[0]; kk[ii] = rng.permutation(kk[ii])
        cnt += (net[~kk].mean() - net[kk].mean()) >= obs
    return cnt / n


def dcr(A, keep):
    B = A[keep]; days = pd.date_range(pd.to_datetime(A.te.min(), unit="ms").normalize(), pd.to_datetime(A.te.max(), unit="ms").normalize()).date
    r = B.groupby("dday").net.sum().reindex(days, fill_value=0).values * SZ / 100; eq = np.cumprod(1 + r)
    return (eq[-1] ** (1 / len(days)) - 1) * 100, (eq / np.maximum.accumulate(eq) - 1).min() * 100, r.min() * 100


def main():
    f = pd.read_csv(os.path.join(R, "reports", "STATE_EXIT_2026-11-21_data", "fills_out.csv"), low_memory=False)
    A = f[(f.kind == "BT") & (f.sleeve == "HT")][["pair", "te", "tp6", "period"]].rename(columns={"tp6": "net"}).reset_index(drop=True)
    need = sorted({(p, ds) for p, te in zip(A.pair, A.te) for ds in C.days_of(int(te) - 2 * 3_600_000, int(te))})
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex: list(ex.map(lambda x: acct_day(*x), need))
    cr, pv, cl, ta, ga = [], [], [], [], []
    for r in A.itertuples():
        z, a, b = crowded(r.pair, int(r.te)); cr.append(z); ta.append(a); ga.append(b)
        x, y = prevol(r.pair, int(r.te)); pv.append(x); cl.append(y)
    A["crowded"], A["top_acc"], A["glob_acc"], A["pre_vol"], A["cloc15"] = cr, ta, ga, pv, cl
    A["burst"] = [ht_burst_state(x, y) for x, y in zip(pv, cl)]
    A["pv"] = [None if x is None else bool(x > PV_CUT) for x in pv]
    A["pv_out"] = [None if v is None else (bool(v) and b != "zone") for v, b in zip(A.pv, A.burst)]
    A["both"] = [None if (c is None or v is None) else (bool(c) and bool(v)) for c, v in zip(A.crowded, A.pv_out)]
    ts = pd.to_datetime(A.te, unit="ms"); A["day"] = ts.dt.floor("D"); A["dday"] = ts.dt.date; A["month"] = ts.dt.strftime("%Y-%m")
    A["half"] = ts.dt.year.astype(str) + "-H" + ((ts.dt.month > 6) + 1).astype(str); A["blk"] = np.where(A.te < HO, "DESIGN", "HOLDOUT")
    A.to_csv(os.path.join(R, "reports", "HT_FILTER_BT_2026-11-26.csv"), index=False)
    S = A[A.burst != "zone"].copy()               # today's stack
    L = [f"HT backtest {len(A)} fills · burst zone {int((A.burst == 'zone').sum())} removed (today's stack) → {len(S)} · coverage: crowded {S.crowded.notna().mean():.0%} · pre_vol {S.pv.notna().mean():.0%}",
         f"today's stack: N {len(S)} · WR {(S.net > 0).mean():.0%} · avg {S.net.mean():+.2f}", ""]
    for col, nm in (("crowded", "CROWDED_LONGS"), ("pv", "PRE_VOL_SURGE (all)"), ("pv_out", "PRE_VOL_SURGE outside burst"), ("both", "BOTH")):
        T = S[S[col].notna()].copy(); T[col] = T[col].astype(bool); z, rest = T[T[col]], T[~T[col]]
        L.append(f"=== {nm}: scored {len(T)} · zone {len(z)} · {z.day.nunique()} days · WR {(z.net > 0).mean():.0%} · avg {z.net.mean():+.2f} vs rest {len(rest)} · WR {(rest.net > 0).mean():.0%} · avg {rest.net.mean():+.2f}")
        for b in ("DESIGN", "HOLDOUT"):
            zz, rr = z[z.blk == b], rest[rest.blk == b]
            L.append(f"   {b}: zone {len(zz)} · {zz.net.mean():+.2f} vs rest {len(rr)} · {rr.net.mean():+.2f}")
        hh = T.groupby(["half", col]).net.agg(["count", "mean"]).unstack(col)
        L.append("   half-years (zone N/avg | rest N/avg): " + " · ".join(
            f"{h} {int(hh.loc[h, ('count', True)]) if ('count', True) in hh and pd.notna(hh.loc[h, ('count', True)]) else 0}/{hh.loc[h, ('mean', True)] if ('mean', True) in hh else float('nan'):+.2f} | {int(hh.loc[h, ('count', False)]) if pd.notna(hh.loc[h, ('count', False)]) else 0}/{hh.loc[h, ('mean', False)]:+.2f}"
            for h in hh.index))
        loss = z[z.net < 0]; top_pair = (loss.groupby("pair").net.sum().min() / loss.net.sum()) if len(loss) else np.nan
        top_day = (loss.groupby("day").net.sum().min() / loss.net.sum()) if len(loss) else np.nan
        L.append(f"   P(zone mean<0) {boot_p(z):.2f} · luck p {luck_p(T, col):.3f} · biggest pair {top_pair:.0%} / day {top_day:.0%} of zone loss")
        b0, b1 = dcr(S, np.ones(len(S), bool)), dcr(S, ~(S[col].fillna(False).astype(bool)).values)
        L.append(f"   sleeve if blocked: avg {S.net.mean():+.2f} → {S[~S[col].fillna(False).astype(bool)].net.mean():+.2f} · DCR {b0[0]:+.3f} → {b1[0]:+.3f} %/d · maxDD {b0[1]:.0f} → {b1[1]:.0f} % · worst day {b0[2]:.1f} → {b1[2]:.1f}")
        for b in ("HOLDOUT",):
            Sb = S[S.blk == b]; kb = ~(Sb[col].fillna(False).astype(bool)).values
            x0, x1 = dcr(Sb, np.ones(len(Sb), bool)), dcr(Sb, kb)
            L.append(f"   HOLDOUT if blocked: avg {Sb.net.mean():+.2f} → {Sb[kb].net.mean():+.2f} · DCR {x0[0]:+.3f} → {x1[0]:+.3f} · maxDD {x0[1]:.0f} → {x1[1]:.0f} % · worst {x0[2]:.1f} → {x1[2]:.1f}")
        L.append("")
    out = "\n".join(L); print(out); open(os.path.join(R, "reports", "HT_FILTER_BT_2026-11-26_summary.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
