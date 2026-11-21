#!/usr/bin/env python3
"""NEWSLEEVE 2026-11-21 — EVAL (PREREG reports/NEWSLEEVE_2026-11-21_PREREG.md).
  need   → metrics_need.csv (pair-days touched by the H1 / H2 raw candidates)
  run    → DESIGN grid + pick + shuffled null, HOLDOUT read once, controls, splits, charts → reports/NEWSLEEVE_2026-11-21_data/*.csv,
           reports/NEWSLEEVE_2026-11-21_tables.md, reports/NEWSLEEVE_2026-11-21_<idea>.png"""
import os, sys, pickle
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "NEWSLEEVE_2026-11-21_data")
REP = os.path.join(R, "reports")
sys.path.insert(0, os.path.join(R, "scripts"))
import importlib.util as _u
_s = _u.spec_from_file_location("nsb", os.path.join(R, "scripts", "NEWSLEEVE_2026-11-21_build.py"))
B = _u.module_from_spec(_s); _s.loader.exec_module(B)

TD = B.TD
DAY = 86_400_000
H1X, H1Y = (0.05, 0.08), (0.01, 0.03)
rng = np.random.default_rng(20261121)


def load(k):
    return pickle.load(open(os.path.join(D, f"build_{k}.pkl"), "rb"))


# ------------------------------------------------------------------ metrics (OI)
def stage_need():
    rows = set()
    for k in ("H1", "H2"):
        b = load(k)
        for s, t in zip(b["sym"], b["t_trig"]):
            for tt in (t, t - 20 * 60_000):
                rows.add((s, pd.Timestamp(int(tt), unit="ms").strftime("%Y-%m-%d")))
    pd.DataFrame(sorted(rows), columns=["sym", "day"]).to_csv(os.path.join(D, "metrics_need.csv"), index=False)
    print("pair-days", len(rows))


_met = {}


_k5 = {}
OI_EXTRA_LAG = 0          # ms; sensitivity read sets 5 min
OI_COL = 1                # 1 = OI value (as pre-registered) · 2 = OI contracts (theory-correct; disclosed post-hoc correction)
ALIGN = {}                # (sym, day) → offset ms found (or None)


def _closes(sym):
    if sym not in _k5:
        z = np.load(os.path.join(D, "k5m", sym + ".npz"))
        _k5[sym] = pd.Series(z["c"], index=z["t"] + 300_000)       # close indexed by bar END time
    return _k5[sym]


def oi_at(sym, t):
    """OI value measured at the latest TRUE snapshot time ≤ t − OI_EXTRA_LAG. The true time of each daily file is found from the
    implied price (value / contracts) vs the 5m closes: offset ∈ {−5, 0, +5, +10} min, best median match, must be < 0.15 %."""
    t = t - OI_EXTRA_LAG
    day = pd.Timestamp(int(t), unit="ms").strftime("%Y-%m-%d")
    key = (sym, day)
    if key not in _met:
        f = os.path.join(D, "met", sym, day + ".npy")
        a = np.load(f) if os.path.exists(f) else np.zeros((0, 3))
        if len(a) and a.shape[1] >= 3:
            cl = _closes(sym)
            px = a[:, 1] / np.where(a[:, 2] > 0, a[:, 2], np.nan)
            best, off = np.inf, None
            for o in (-300_000, 0, 300_000, 600_000):
                ref = cl.reindex(a[:, 0] + o).values
                err = np.nanmedian(np.abs(px / ref - 1)) if np.isfinite(ref).sum() >= 20 else np.inf
                if err < best:
                    best, off = err, o
            if off is None or best > 0.0015:
                a = np.zeros((0, 3)); off = None
            else:
                a = a.copy(); a[:, 0] += off
            ALIGN[key] = off
        else:
            a = np.zeros((0, 3)); ALIGN[key] = None
        _met[key] = a
    a = _met[key]
    if not len(a):
        return np.nan
    j = np.searchsorted(a[:, 0], t, side="right") - 1
    if j < 0 or t - a[j, 0] > 10 * 60_000:
        return np.nan
    return a[j, OI_COL]


def add_doi(b):
    d = np.full(len(b["trig"]), np.nan)
    for i, (s, t) in enumerate(zip(b["sym"], b["t_trig"])):
        a, z = oi_at(s, t), oi_at(s, t - 15 * 60_000)
        if np.isfinite(a) and np.isfinite(z) and z > 0:
            d[i] = a / z - 1
    b["doi"] = d
    return b


# ------------------------------------------------------------------ helpers
def subset(b, m):
    out = {}
    n = len(b["trig"])
    for k, v in b.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            out[k] = v[m]
    return out


def dedup_t(b, gap_ms):
    o = np.lexsort((b["t_trig"], b["sym"]))
    keep = np.zeros(len(o), bool)
    last_s, last_t = None, -1e18
    for j in o:
        s, t = b["sym"][j], b["t_trig"][j]
        if s != last_s or t >= last_t + gap_ms:
            keep[j] = True; last_s, last_t = s, t
    return keep


def wilson_lo(k, n, z=1.645):
    if n == 0:
        return np.nan
    p = k / n
    den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def kelly(p, w, l):
    if not (l > 0) or not np.isfinite(p):
        return np.nan
    return p - (1 - p) / (w / l)


def day_boot(x, days, nb=2000, q=0.05):
    """one-sided lower bound of the mean, resampling UTC days."""
    m = np.isfinite(x)
    x, days = x[m], days[m]
    if len(x) < 2:
        return np.nan, np.nan
    u, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=x); c = np.bincount(inv)
    k = len(u)
    idx = rng.integers(0, k, (nb, k))
    bm = s[idx].sum(1) / c[idx].sum(1)
    return np.quantile(bm, q), np.quantile(bm, 1 - q)


def half(t):
    ts = pd.to_datetime(t, unit="ms")
    return np.array([f"{y % 100}-H{1 if m <= 6 else 2}" for y, m in zip(ts.year, ts.month)])


def stats(x, days, syms=None, t=None):
    m = np.isfinite(x)
    x = x[m]; days = days[m]
    n = len(x)
    if n == 0:
        return dict(N=0)
    w = x[x > 0]; l = x[x <= 0]
    wr = len(w) / n
    aw, al = (w.mean() if len(w) else 0.0), (-l.mean() if len(l) else np.nan)
    lo, hi = day_boot(x, days)
    wl = wilson_lo(len(w), n)
    r = dict(N=n, days=len(np.unique(days)), WR=wr, avg=x.mean(), sum=x.sum(), avg_win=aw, avg_loss=-al if np.isfinite(al) else np.nan,
             be_WR=al / (aw + al) if np.isfinite(al) and aw + al > 0 else np.nan, wilson_lo=wl,
             kelly=kelly(wr, aw, al), kelly_floor=kelly(wl, aw, al), ci_lo=lo, ci_hi=hi, stress=(x - 0.20).mean())
    if syms is not None:
        s = pd.Series(x).groupby(syms[m]).sum()
        dd = pd.Series(x).groupby(days).sum()
        tot = x.sum()
        r["top_pair"] = s.idxmax(); r["top_pair_share"] = s.max() / tot if tot > 0 else np.nan
        r["top_day_share"] = dd.max() / tot if tot > 0 else np.nan
    return r


def md_table(df):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else f"{v:+.2f}"
        return str(v)
    h = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False))


def btc_regime():
    z = np.load(os.path.join(R, "reports", "HIGH_ATR_2026-11-18_data", "k5m", "BTCUSDT.npy"))   # cols t,o,h,l,c,…
    fut = pd.Series(z[:, 4], index=pd.to_datetime(z[:, 0].astype(np.int64), unit="ms")).resample("1D").last().dropna()
    sp = pd.read_csv(os.path.join(R, "reports", "LEADS_DEEP_2026-11-18_data", "BTCUSDT_spot_1d.csv"))   # spot daily since 2017 (EMA200 warm-up)
    s = pd.Series(sp.c.values, index=pd.to_datetime(sp.open_time, unit="ms"))
    s = pd.concat([s, fut[fut.index > s.index.max()]])
    e50, e200 = s.ewm(span=50, adjust=False).mean(), s.ewm(span=200, adjust=False).mean()
    reg = np.sign(e50 - e200).shift(1)                       # state at the previous day's close
    return {int(ts.value // 1e6 // DAY): v for ts, v in reg.items()}


# ------------------------------------------------------------------ idea sets
def idea_sets():
    """{idea: [(variant_label, event-dict)] } — possibilities (deduped)."""
    S = {}
    for nm in ("H5B", "H5S"):
        S[nm] = []
        for vi, (M, Sh) in enumerate(B.H5V):
            f = os.path.join(D, f"build_{nm}_v{vi}.pkl")
            if os.path.exists(f):
                S[nm].append((f"ratio≥{M:g} share{'≥' if nm == 'H5B' else '≤'}{Sh if nm == 'H5B' else 1 - Sh:.2f}", load(f"{nm}_v{vi}")))
    for nm, d in (("H1", 1), ("H2", -1)):
        b = add_doi(load(nm))
        S[nm], S[nm + "_noOI"] = [], []
        for X in H1X:
            mv = (b["r15_at"] <= -X) if d > 0 else (b["r15_at"] >= X)
            for Y in H1Y:
                m = mv & (b["doi"] <= -Y)
                sb = subset(b, m)
                S[nm].append((f"|r15|≥{X:.0%} dOI≤−{Y:.0%}", subset(sb, dedup_t(sb, 6 * 3600_000))))
            sb = subset(b, mv)
            S[nm + "_noOI"].append((f"|r15|≥{X:.0%} (no OI leg)", subset(sb, dedup_t(sb, 6 * 3600_000))))
            m2 = mv & np.isfinite(b["doi"])
            sb = subset(b, m2)
            S[nm + "_noOI"].append((f"|r15|≥{X:.0%} OI known (any)", subset(sb, dedup_t(sb, 6 * 3600_000))))
        S[nm + "_raw"] = b
    for nm in ("H6L", "H6S"):
        f = os.path.join(D, f"build_{nm}.pkl")
        S[nm] = [("frozen", load(nm))] if os.path.exists(f) else []
    return S


def cells_for(nm):
    return B.EXIT_F if nm.startswith("H6") else B.EXIT_A


def cell_label(c):
    tp, sl, hb = c
    return f"TP{tp*100:g}/SL{sl*100:g}/{hb*5/60:g}h"


SUF = ""


def run():
    S = idea_sets()
    al = pd.DataFrame([(k[0], k[1], v) for k, v in ALIGN.items()], columns=["sym", "day", "offset_ms"])
    al["month"] = al.day.str[:7]
    pd.crosstab(al.month, al.offset_ms.fillna(-1).astype(int).map(lambda v: "unaligned" if v == -1 else f"{v // 60000:+d} min")).to_csv(os.path.join(D, f"oi_alignment_by_month{SUF}.csv"))
    REG = btc_regime()
    out_md = []
    summary = []
    for nm in ("H5B", "H5S", "H1", "H2", "H6L", "H6S"):
        vs = S.get(nm, [])
        if not vs:
            continue
        cells = cells_for(nm)
        # ---------- counts
        cnt_rows = []
        for lab, b in vs:
            t = b["t_trig"]
            for per, m in (("DESIGN", t < TD), ("HOLDOUT", t >= TD)):
                cnt_rows.append(dict(variant=lab, period=per, possibilities=int(m.sum()),
                                     windows15=len(np.unique(t[m] // 900_000)), days=len(np.unique(t[m] // DAY)),
                                     coins=len(np.unique(b["sym"][m]))))
        # ---------- DESIGN grid
        grid = []
        for vi, (lab, b) in enumerate(vs):
            des = b["t_trig"] < TD
            days = b["t_trig"] // DAY
            for ci, c in enumerate(cells):
                x = b["net"][des, ci].astype(float)
                cx = b["cnet"][des, ci].astype(float)
                st = stats(x, days[des])
                grid.append(dict(variant=lab, vi=vi, cell=cell_label(c), ci=ci, N=st.get("N", 0), days=st.get("days", 0),
                                 WR=st.get("WR"), avg=st.get("avg"), ctrl_avg=np.nanmean(cx) if len(cx) else np.nan))
        G = pd.DataFrame(grid)
        elig = G[(G.N >= 30) & (G.days >= 15)]
        if not len(elig):
            out_md.append(f"## {nm}\nNo DESIGN cell reaches N ≥ 30 and 15 days → nothing to pick.\n")
            summary.append(dict(idea=nm, verdict="REFUTED (too rare on DESIGN)"))
            continue
        pick = elig.sort_values("avg", ascending=False).iloc[0]
        # ---------- shuffled null (max-cell avg over the full grid, DESIGN)
        null_max = np.full(B.NREP, -np.inf)
        for vi, (lab, b) in enumerate(vs):
            des = b["t_trig"] < TD
            if "null_sum" in b:
                ns, nc = b["null_sum"], b["null_cnt"]
                av = ns / np.maximum(nc, 1)
                ok = (G[(G.vi == vi)].sort_values("ci").pipe(lambda g: ((g.N >= 30) & (g.days >= 15)).values))
                for ci in range(len(cells)):
                    if ok[ci]:
                        null_max = np.maximum(null_max, av[:, ci])
            elif "nnet" in b:
                nn = b["nnet"][des].astype(np.float32)
                av = np.nanmean(nn, axis=0)                        # (NREP, ncells)
                ok = (G[(G.vi == vi)].sort_values("ci").pipe(lambda g: ((g.N >= 30) & (g.days >= 15)).values))
                for ci in range(len(cells)):
                    if ok[ci]:
                        null_max = np.maximum(null_max, av[:, ci])
        null95 = np.quantile(null_max, 0.95)
        null_pass = pick.avg > null95
        # ---------- HOLDOUT once at the pick
        lab, b = vs[int(pick.vi)]
        ci = int(pick.ci)
        ho = b["t_trig"] >= TD
        x = b["net"][ho, ci].astype(float)
        cx = b["cnet"][ho, ci].astype(float)
        t = b["t_trig"][ho]; days = t // DAY; syms = b["sym"][ho]
        st = stats(x, days, syms, t)
        diff = x - cx
        dlo, dhi = day_boot(diff, days)
        hy = half(t)
        hy_tab = []
        for h in sorted(np.unique(hy)):
            m = hy == h
            s2 = stats(x[m], days[m])
            hy_tab.append(dict(half=h, N=s2["N"], days=s2.get("days"), WR=s2.get("WR"), avg=s2.get("avg"),
                               ctrl=np.nanmean(cx[m]), ci_lo=s2.get("ci_lo")))
        HY = pd.DataFrame(hy_tab)
        n_pos_half = int((HY.avg > 0).sum())
        reg = np.array([REG.get(int(d), np.nan) for d in days])
        reg_tab = []
        for rv, rn in ((1, "BTC bull (EMA50>EMA200)"), (-1, "BTC bear (EMA50<EMA200)")):
            for per, mm in (("DESIGN", b["t_trig"] < TD), ("HOLDOUT", ho)):
                dd = b["t_trig"][mm] // DAY
                rr = np.array([REG.get(int(d_), np.nan) for d_ in dd])
                xx = b["net"][mm, ci].astype(float)[rr == rv]
                s3 = stats(xx, dd[rr == rv])
                reg_tab.append(dict(regime=rn, period=per, N=s3["N"], days=s3.get("days"), WR=s3.get("WR"), avg=s3.get("avg"),
                                    ci_lo=s3.get("ci_lo")))
        v24 = b["v24"][ho]
        cap_tab = []
        for lo_, hi_, tn in ((10e6, 50e6, "$10–50M"), (50e6, 200e6, "$50–200M"), (200e6, 1e15, "≥ $200M")):
            m = (v24 >= lo_) & (v24 < hi_)
            s4 = stats(x[m], days[m])
            cap_tab.append(dict(tier=tn, N=s4["N"], WR=s4.get("WR"), avg=s4.get("avg"), ctrl=np.nanmean(cx[m]) if m.any() else np.nan))
        # also all HOLDOUT cells (information only)
        hall = []
        for vi2, (lab2, b2) in enumerate(vs):
            ho2 = b2["t_trig"] >= TD
            for ci2, c in enumerate(cells):
                xx = b2["net"][ho2, ci2].astype(float)
                hall.append(dict(variant=lab2, cell=cell_label(c), N=int(np.isfinite(xx).sum()), avg=np.nanmean(xx) if len(xx) else np.nan,
                                 ctrl=np.nanmean(b2["cnet"][ho2, ci2]) if ho2.any() else np.nan))
        HA_ = pd.DataFrame(hall)
        checks = {
            "1 Kelly>0 at Wilson floor": st.get("kelly_floor", np.nan) > 0,
            "2 day-bootstrap lower 95% > 0": st.get("ci_lo", np.nan) > 0,
            "3 ≥2/3 half-years positive": n_pos_half >= 2,
            "4 survives +0.20% slip": st.get("stress", np.nan) > 0,
            "5 no pair/day ≥50% of profit": (st.get("top_pair_share", 1) < 0.5) and (st.get("top_day_share", 1) < 0.5),
            "6 beats random control (lower 95% > 0)": dlo > 0,
            "7 DESIGN pick beat shuffled null": bool(null_pass),
        }
        if nm in ("H1", "H2", "H6L", "H6S"):
            checks["8 market-wide: ≥30 HOLDOUT days"] = st.get("days", 0) >= 30
        npass = sum(bool(v) for v in checks.values())
        allp = all(checks.values())
        if allp:
            verdict = "CANDIDATE (5m) — needs the 1m re-walk"
        elif st.get("avg", -1) > 0 and np.nanmean(diff) > 0 and checks["4 survives +0.20% slip"] and checks["5 no pair/day ≥50% of profit"]:
            verdict = "OBSERVE"
        else:
            verdict = "REFUTED"
        summary.append(dict(idea=nm, vi=int(pick.vi), ci=ci, pick=f"{lab} · {pick.cell}", design_avg=pick.avg, design_N=pick.N, null95=null95,
                            ho_N=st["N"], ho_days=st.get("days"), ho_WR=st.get("WR"), ho_avg=st.get("avg"), ho_ci_lo=st.get("ci_lo"),
                            ho_ctrl=np.nanmean(cx), ho_diff_lo=dlo, kelly_floor=st.get("kelly_floor"), halves_pos=f"{n_pos_half}/{len(HY)}",
                            stress=st.get("stress"), passes=f"{npass}/{len(checks)}", verdict=verdict))
        # ---------- save tables
        pd.DataFrame(cnt_rows).to_csv(os.path.join(D, f"{nm}_counts{SUF}.csv"), index=False)
        G.to_csv(os.path.join(D, f"{nm}_design_grid{SUF}.csv"), index=False)
        HA_.to_csv(os.path.join(D, f"{nm}_holdout_allcells_info{SUF}.csv"), index=False)
        HY.to_csv(os.path.join(D, f"{nm}_holdout_halves{SUF}.csv"), index=False)
        pd.DataFrame(reg_tab).to_csv(os.path.join(D, f"{nm}_regime{SUF}.csv"), index=False)
        pd.DataFrame(cap_tab).to_csv(os.path.join(D, f"{nm}_capacity{SUF}.csv"), index=False)
        # ---------- markdown
        f2 = lambda v, p=2: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:+.{p}f}" if isinstance(v, (float, np.floating)) else str(v))
        md = [f"## {nm}", "", "### Trigger counts (events = raw trigger bars; possibilities = one per coin per holding period)", "",
              "| variant | period | possibilities | 15-min windows | days | coins |", "|---|---|---|---|---|---|"]
        for r in cnt_rows:
            md.append(f"| {r['variant']} | {r['period']} | {r['possibilities']} | {r['windows15']} | {r['days']} | {r['coins']} |")
        if "n_raw" in vs[0][1]:
            md.append(f"\nRaw trigger bars (events) for variant 1: {vs[0][1].get('n_raw', '—')}.")
        md += ["", "### DESIGN grid (avg net % per possibility; control = random entry same coin / day / direction / exit)", "",
               "| variant | exit | N | days | WR | avg | control avg |", "|---|---|---|---|---|---|---|"]
        for r in G.itertuples():
            md.append(f"| {r.variant} | {r.cell} | {r.N} | {r.days} | {r.WR:.0%} | {r.avg:+.2f} | {r.ctrl_avg:+.2f} |" if r.N else
                      f"| {r.variant} | {r.cell} | 0 | 0 | — | — | — |")
        md += ["", f"**DESIGN pick:** {lab} · {pick.cell} → N {pick.N} · avg {pick.avg:+.2f} %. Shuffled-null 95th pct of the max-cell avg: "
               f"{null95:+.2f} % → {'PASS' if null_pass else 'FAIL'}.", "",
               "### HOLDOUT (read once at the pick)", "",
               "| N | days | WR | avg win | avg loss | breakeven WR | Wilson floor | Kelly obs | Kelly floor | avg % | day-boot 90 % CI | +0.20 slip | control avg | paired Δ lower 95 % | top pair (share) | top day share |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
               f"| {st['N']} | {st['days']} | {st['WR']:.0%} | {st['avg_win']:+.2f} | {st['avg_loss']:+.2f} | {st['be_WR']:.0%} | {st['wilson_lo']:.0%} | "
               f"{f2(st['kelly'])} | {f2(st['kelly_floor'])} | **{st['avg']:+.2f}** | {st['ci_lo']:+.2f} … {st['ci_hi']:+.2f} | {st['stress']:+.2f} | "
               f"{np.nanmean(cx):+.2f} | {dlo:+.2f} | {st.get('top_pair')} ({f2(st.get('top_pair_share'))}) | {f2(st.get('top_day_share'))} |", "",
               "| half-year | N | days | WR | avg | control | day-boot lower 95 % |", "|---|---|---|---|---|---|---|"]
        for r in HY.itertuples():
            md.append(f"| {r.half} | {r.N} | {r.days} | {r.WR:.0%} | {r.avg:+.2f} | {r.ctrl:+.2f} | {f2(r.ci_lo)} |")
        md += ["", "| BTC daily regime | period | N | days | WR | avg | lower 95 % |", "|---|---|---|---|---|---|---|"]
        for r in reg_tab:
            md.append(f"| {r['regime']} | {r['period']} | {r['N']} | {r['days'] or 0} | {f2(r['WR'] if r['WR'] is None else r['WR']*100, 0)} % | {f2(r['avg'])} | {f2(r['ci_lo'])} |")
        md += ["", "| 24 h volume at trigger (HOLDOUT) | N | WR | avg | control |", "|---|---|---|---|---|"]
        for r in cap_tab:
            md.append(f"| {r['tier']} | {r['N']} | {f2(r['WR'] if r['WR'] is None else r['WR']*100, 0)} % | {f2(r['avg'])} | {f2(r['ctrl'])} |")
        md += ["", "| gate | pass |", "|---|---|"] + [f"| {k} | {'✓' if v else '✗'} |" for k, v in checks.items()]
        md += ["", f"**Verdict (mechanical): {verdict}**", ""]
        out_md += md
        if not SUF:
            chart(nm, lab, pick, b, ci, cells)
    # no-OI comparison for H1/H2
    for nm in ("H1_noOI", "H2_noOI"):
        vs = S.get(nm, [])
        md = [f"## {nm[:2]} — the same price + volume trigger without / with the OI leg (descriptive, DESIGN | HOLDOUT, best exit of the H grid not re-picked: cell TP2/SL4/6h)", "",
              "| variant | DESIGN N | DESIGN avg | HOLDOUT N | HOLDOUT avg | HOLDOUT control |", "|---|---|---|---|---|---|"]
        ci = [i for i, c in enumerate(B.EXIT_A) if c == (0.02, 0.04, 72)][0]
        for lab, b in vs + [(l2, b2) for l2, b2 in S[nm[:2]]]:
            de = b["t_trig"] < TD
            md.append(f"| {lab} | {de.sum()} | {np.nanmean(b['net'][de, ci]):+.2f} | {(~de).sum()} | {np.nanmean(b['net'][~de, ci]):+.2f} | "
                      f"{np.nanmean(b['cnet'][~de, ci]):+.2f} |")
        raw = S[nm[:2] + "_raw"]
        md.append(f"\nOI coverage on the raw candidates: {np.isfinite(raw['doi']).mean():.0%} of {len(raw['doi'])} trigger bars.\n")
        out_md += md
    SM = pd.DataFrame(summary)
    SM.to_csv(os.path.join(D, f"summary{SUF}.csv"), index=False)
    open(os.path.join(REP if not SUF else D, f"NEWSLEEVE_2026-11-21_tables{SUF}.md"), "w").write("# NEWSLEEVE 2026-11-21 — tables (generated)\n\n" +
                                                                         md_table(SM) + "\n\n" + "\n".join(out_md))
    print(SM.to_string())


def chart(nm, lab, pick, b, ci, cells):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = -1 if nm in ("H5B", "H2", "H6S") else 1
    P, CP = b["path"].astype(float), b["cpath"].astype(float)
    x = (np.arange(P.shape[1]) - B.PRE) * 5 / 60
    ret = d * (P - 1) * 100; cret = d * (CP - 1) * 100
    net = b["net"][:, ci].astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for k, (per, m) in enumerate((("DESIGN", b["t_trig"] < TD), ("HOLDOUT", b["t_trig"] >= TD))):
        a = ax[k]
        a.plot(x, np.nanmedian(ret[m], 0), color="#1f4e9c", lw=2, label=f"all triggers (N {m.sum()})")
        a.plot(x, np.nanmedian(ret[m & (net > 0)], 0), color="#2e8b57", lw=1.2, label="winners at the pick")
        a.plot(x, np.nanmedian(ret[m & (net <= 0)], 0), color="#c0392b", lw=1.2, label="losers at the pick")
        a.plot(x, np.nanmedian(cret[m], 0), color="#888", lw=1.5, ls="--", label="random control (same coin / day)")
        a.axvline(0, color="k", lw=0.6); a.axhline(0, color="k", lw=0.6)
        a.set_title(f"{nm} {per} — MEDIAN price in trade direction vs entry open (%)", fontsize=10)
        a.set_xlabel("hours from entry"); a.legend(fontsize=7)
    fig.suptitle(f"{nm} · {lab} · pick {pick.cell} (5m bars, before costs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(REP, f"NEWSLEEVE_2026-11-21_{nm}.png"), dpi=110)
    plt.close(fig)


def stage_rewalk():
    """1m re-walk of the HOLDOUT possibilities at the pick, for every idea the 5m run marked CANDIDATE (PREREG §2 / §9)."""
    import io, zipfile, urllib.request
    SM = pd.read_csv(os.path.join(D, "summary.csv"))
    S = idea_sets()
    K1 = os.path.join(D, "k1m"); os.makedirs(K1, exist_ok=True)
    rows = []
    for r in SM.itertuples():
        if not str(r.verdict).startswith("CANDIDATE"):
            continue
        nm = r.idea; lab, b = S[nm][int(r.vi)]; cell = cells_for(nm)[int(r.ci)]
        d = -1 if nm in ("H5B", "H2", "H6S") else 1
        ho = np.where(b["t_trig"] >= TD)[0]
        nets = []
        for j in ho:
            sym, te = b["sym"][j], int(b["t_trig"][j])          # entry = trigger close = next 1m open
            hold = cell[2] * 5
            days = sorted({pd.Timestamp(te + k * 60_000, unit="ms").strftime("%Y-%m-%d") for k in (0, hold)})
            parts = []
            for ds in days:
                f = os.path.join(K1, f"{sym}_{ds}.npy")
                if not os.path.exists(f):
                    z = B.__dict__.get("_dummy")
                    try:
                        with urllib.request.urlopen(urllib.request.Request(
                                f"https://data.binance.vision/data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{ds}.zip",
                                headers={"User-Agent": "Mozilla/5.0 (research client)"}), timeout=60) as rr:
                            zz = zipfile.ZipFile(io.BytesIO(rr.read()))
                            x = pd.read_csv(zz.open(zz.namelist()[0]), header=None)
                            if not str(x.iloc[0, 0]).isdigit():
                                x = x.iloc[1:]
                            a = x.iloc[:, :5].astype(float).values
                            a[a[:, 0] > 1e14, 0] //= 1000
                    except Exception:
                        a = np.zeros((0, 5))
                    np.save(f, a)
                parts.append(np.load(f))
            a = np.concatenate(parts) if parts else np.zeros((0, 5))
            k = np.searchsorted(a[:, 0], te) if len(a) else 0
            if not len(a) or k >= len(a) or a[k, 0] != te:
                nets.append(np.nan); continue
            w = a[k:k + hold]
            P0 = w[0, 1]
            fu = B.load_funding(sym)
            ts = w[:, 0]
            G = np.array([fu[(fu[:, 0] > te) & (fu[:, 0] <= t), 1].sum() for t in ts]) if len(fu) else np.zeros(len(ts))
            pad = lambda v: np.r_[v, np.full(hold - len(v), np.nan)][None, :]
            net, _ = B.walk(pad(w[:, 1] / P0), pad(w[:, 2] / P0), pad(w[:, 3] / P0), pad(w[:, 4] / P0), pad(G), d,
                            np.array([b["v24"][j]]), [(cell[0], cell[1], hold)])
            nets.append(net[0, 0])
        x = np.array(nets, float); x5 = b["net"][ho, int(r.ci)].astype(float)
        st = stats(x, b["t_trig"][ho] // DAY)
        rows.append(dict(idea=nm, N=st["N"], WR=st.get("WR"), avg_1m=st.get("avg"), ci_lo=st.get("ci_lo"), avg_5m=np.nanmean(x5),
                         kelly_floor=st.get("kelly_floor")))
    pd.DataFrame(rows).to_csv(os.path.join(D, "rewalk_1m.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "lag5":
        OI_EXTRA_LAG, SUF = 300_000, "_lag5"
    if len(sys.argv) > 2 and sys.argv[2] == "contracts":
        OI_COL, SUF = 2, "_contracts"
    {"need": stage_need, "run": run, "rewalk": stage_rewalk}[sys.argv[1]]()
