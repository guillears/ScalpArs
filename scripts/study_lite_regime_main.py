#!/usr/bin/env python3
"""LITE MACRO-REGIME study — screen, null, discover/confirm, ATR reads, policies, equity-curve switches, BTC grids.
Per reports/LITE_REGIME_PREREG.txt (+ grid addendum + deviation 1). Reads reports/lite_regime/day_features.csv.
Outputs reports/lite_regime/*.csv and reports/lite_regime/tables.md.
Usage: venv/bin/python scripts/study_lite_regime_main.py [--R 500]
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "lite_regime")
LIVE = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-15_19-55-15.csv")
R = int(sys.argv[sys.argv.index("--R") + 1]) if "--R" in sys.argv else 500
RNG = np.random.default_rng(20261015)
TICK = "--tick" in sys.argv          # pricing-controlled re-run: BT fills priced on ticks only (2025 1m-priced non-top-50 fills dropped)
SUF = "_tick" if TICK else ""
DAY = 86_400_000
SIGNV = {"r1", "r3", "r7", "r30", "dma20", "dma50", "dma200", "alt_rs7", "ftr12", "lite_n10", "lite_n20", "lite_d3", "lite_d7",
         "willy_n10", "willy_n20", "willy_d3", "willy_d7"}
VARS = ["r1", "r3", "r7", "r30", "dma20", "dma50", "dma200", "rv7", "rvtr", "off30",
        "br_up3", "br_up7", "br_ma7", "br_ma20", "alt_rs7", "voltr", "disp7",
        "spk24", "spk72", "flags96", "ft2", "ft12", "ftr12",
        "lite_n10", "lite_n20", "lite_d3", "lite_d7", "willy_n10", "willy_n20", "willy_d3", "willy_d7"]   # 31 cross-year; fund3 H1->H2
UNITS = ["1d", "3d", "7d", "week"]
TABLES = []


def T(s=""):
    TABLES.append(s)


# ───────────────────────────── data
def load():
    P = pd.read_csv(os.path.join(ROOT, "reports", "LITE_PROTECT_rows.csv"), usecols=["fid", "block", "era", "pair", "te", "day", "rm_base_pct", "src"])
    if TICK:
        P = P[(P.block != "BT") | P.src.astype(str).str.startswith("ticks")]
    M = pd.read_csv(os.path.join(ROOT, "reports", "lite_wlsweep", "matrix.csv"), usecols=["fid", "entry_atr_pct"])
    P = P.merge(M, on="fid", how="left")
    lv = pd.read_csv(LIVE)
    lv = lv[(lv.entry_strategy == "FRENZY_LITE") & (lv.status == "CLOSED")]
    L = pd.DataFrame({"fid": "LIVE|" + lv.pair + "|" + lv.opened_at, "block": "LIVE", "era": "LIVE", "pair": lv.pair,
                      "te": pd.to_datetime(lv.opened_at).values.astype("datetime64[ms]").astype(np.int64),
                      "day": lv.opened_at.str[:10], "rm_base_pct": lv.pnl_percentage, "entry_atr_pct": lv.entry_atr_pct})
    P = pd.concat([P, L], ignore_index=True)
    P["y"] = P.rm_base_pct
    P["yr"] = np.where(P.era == "2025", "2025", np.where(P.era.isin(["2026-H1", "2026-H2"]), "2026", P.era))
    F = pd.read_csv(os.path.join(OUT, "day_features.csv")).set_index("day")
    q1, q2 = P[(P.block == "BT") & (P.era == "2025")].entry_atr_pct.quantile([1 / 3, 2 / 3])
    P["atr3"] = np.where(P.entry_atr_pct <= q1, "low", np.where(P.entry_atr_pct <= q2, "mid", "high"))
    P["dt"] = pd.to_datetime(P.day)
    P["week"] = (P.dt - pd.to_timedelta(P.dt.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    P["month"] = P.day.str[:7]
    return P, F, (q1, q2)


# ───────────────────────────── per-period day arrays
class Period:
    """per-day (or per-week) sums for a set of fills; days = calendar range of the fills' days."""
    def __init__(self, fills, F, unit):
        self.unit = unit
        d0, d1 = fills.dt.min(), fills.dt.max()
        if unit == "week":
            d0 = d0 - pd.Timedelta(days=d0.weekday())
            keys = pd.date_range(d0, d1, freq="7D")
            kf = fills.week.values
            self.keys = keys.strftime("%Y-%m-%d").values
            idx = {k: i for i, k in enumerate(self.keys)}
            S = np.zeros(len(keys)); N = np.zeros(len(keys)); Wn = np.zeros(len(keys))
            for k, y in zip(kf, fills.y.values):
                i = idx[k]; S[i] += y; N[i] += 1; Wn[i] += y > 0
            self.block = np.arange(len(keys))                     # 1 week = 1 block
        else:
            H = int(unit[:-1])
            keys = pd.date_range(d0, d1, freq="D")
            self.keys = keys.strftime("%Y-%m-%d").values
            idx = {k: i for i, k in enumerate(self.keys)}
            n = len(keys)
            s1 = np.zeros(n); n1 = np.zeros(n); w1 = np.zeros(n)
            for k, y in zip(fills.day.values, fills.y.values):
                i = idx[k]; s1[i] += y; n1[i] += 1; w1[i] += y > 0
            cs = lambda a: np.concatenate([[0], np.cumsum(a)])
            c_s, c_n, c_w = cs(s1), cs(n1), cs(w1)
            hi = np.minimum(np.arange(n) + H, n)
            S = c_s[hi] - c_s[:n]; N = c_n[hi] - c_n[:n]; Wn = c_w[hi] - c_w[:n]
            self.block = np.arange(n) // 7
        self.S, self.N, self.W = S, N, Wn

    def feat(self, F, v):
        return F[v].reindex(self.keys).values.astype(float)


def nat_cut(v, train_vals):
    return 0.0 if v in SIGNV else float(np.nanmedian(train_vals))


def split_diff(f, S, N, cut):
    """fill-weighted avg(f > cut) - avg(f <= cut); f may be 2-D (R x n)."""
    ok = ~np.isnan(f)
    hi = (f > cut) & ok; lo = (f <= cut) & ok
    sh, nh = (hi * S).sum(-1), (hi * N).sum(-1); sl, nl = (lo * S).sum(-1), (lo * N).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sh / nh - sl / nl


def t3_diff(f, S, N, c1, c2):
    ok = ~np.isnan(f)
    hi = (f > c2) & ok; lo = (f <= c1) & ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return (hi * S).sum(-1) / (hi * N).sum(-1) - (lo * S).sum(-1) / (lo * N).sum(-1)


def rank(a):
    return pd.DataFrame(a).rank(axis=-1).values if a.ndim == 2 else pd.Series(a).rank().values


def spearman(f, S, N):
    m = (N > 0)
    avg = np.where(m, S / np.where(N > 0, N, 1), np.nan)
    if f.ndim == 1:
        ok = m & ~np.isnan(f)
        if ok.sum() < 10:
            return np.nan
        a, b = pd.Series(f[ok]).rank().values, pd.Series(avg[ok]).rank().values
        return np.corrcoef(a, b)[0, 1]
    ok = m[None, :] & ~np.isnan(f)
    out = np.full(f.shape[0], np.nan)
    ya = avg
    for r in range(f.shape[0]):
        o = ok[r]
        if o.sum() >= 10:
            out[r] = np.corrcoef(pd.Series(f[r, o]).rank().values, pd.Series(ya[o]).rank().values)[0, 1]
    return out


def block_perm(block, R):
    """R permutations of the day index that move whole 7-day blocks."""
    ub = np.unique(block)
    members = [np.where(block == b)[0] for b in ub]
    out = np.empty((R, len(block)), dtype=int)
    for r in range(R):
        order = RNG.permutation(len(ub))
        idx = np.concatenate([members[i] for i in order])
        if len(idx) < len(block):
            idx = np.concatenate([idx, idx[: len(block) - len(idx)]])
        out[r] = idx[: len(block)]
    return out


def block_boot(block, R):
    ub = np.unique(block)
    W = np.zeros((R, len(block)))
    for r in range(R):
        pick = RNG.choice(ub, len(ub), replace=True)
        cnt = np.bincount(pick, minlength=ub.max() + 1)
        W[r] = cnt[block]
    return W


# ───────────────────────────── screen
def screen(P, F):
    bt = P[P.block == "BT"]
    per = {"2025": bt[bt.yr == "2025"], "2026": bt[bt.yr == "2026"], "H1": bt[bt.era == "2026-H1"], "H2": bt[bt.era == "2026-H2"]}
    rows = []
    nulls = {}
    PER = {}
    for pn in ("2025", "2026"):
        for u in UNITS:
            pr = Period(per[pn], F, u)
            PER[(pn, u)] = pr
            perm = block_perm(pr.block, R)
            boot = block_boot(pr.block, R)
            for v in VARS:
                f = pr.feat(F, v)
                if np.isnan(f).all():
                    continue
                c = nat_cut(v, f); c1, c2 = np.nanquantile(f, [1 / 3, 2 / 3])
                fp = f[perm]
                obs = {"rho": spearman(f, pr.S, pr.N), "nat": split_diff(f, pr.S, pr.N, c), "t3": t3_diff(f, pr.S, pr.N, c1, c2)}
                nul = {"rho": spearman(fp, pr.S, pr.N), "nat": split_diff(fp, pr.S, pr.N, c), "t3": t3_diff(fp, pr.S, pr.N, c1, c2)}
                ok = ~np.isnan(f)
                hi = (f > c) & ok; lo = (f <= c) & ok
                bs = ((boot * hi * pr.S).sum(1) / (boot * hi * pr.N).sum(1)) - ((boot * lo * pr.S).sum(1) / (boot * lo * pr.N).sum(1))
                for st in ("rho", "nat", "t3"):
                    nv = nul[st][~np.isnan(nul[st])]
                    sd = nv.std() if len(nv) > 10 else np.nan
                    p = float((np.abs(nv) >= abs(obs[st])).mean()) if len(nv) else np.nan
                    rows.append(dict(period=pn, unit=u, var=v, stat=st, obs=obs[st], z=obs[st] / sd if sd else np.nan, p_null=p,
                                     P_boot_sign=float((np.sign(bs) == np.sign(obs["nat"])).mean()) if st == "nat" else np.nan,
                                     n_hi=int((hi * pr.N).sum()), n_lo=int((lo * pr.N).sum()), cut=c))
                    nulls[(pn, u, v, st)] = (nul[st], sd)
    S = pd.DataFrame(rows)
    # family-wise p per period x stat over the 31 x 4 tests (same permutation index per period-unit; max|z| across vars within unit,
    # then across units by permutation number)
    fw = []
    for pn in ("2025", "2026"):
        for st in ("rho", "nat", "t3"):
            mx = np.zeros(R)
            for (p2, u, v, s2), (nv, sd) in nulls.items():
                if p2 == pn and s2 == st and sd and len(nv) == R:
                    mx = np.maximum(mx, np.nan_to_num(np.abs(nv / sd)))
            sub = S[(S.period == pn) & (S.stat == st)]
            for i in sub.index:
                S.loc[i, "fw_p"] = float((mx >= abs(S.loc[i, "z"])).mean()) if S.loc[i, "z"] == S.loc[i, "z"] else np.nan
            fw.append((pn, st, float(np.nanmax(np.abs(sub.z))), float(np.nanmin(S.loc[sub.index, "fw_p"]))))
    return S, fw, per, PER


def confirm(per, F, A, B, units=("1d", "3d", "7d", "week")):
    """cut + direction on A, applied to B: same sign & B bootstrap P >= 0.90."""
    out = []
    for u in units:
        pa, pb = Period(per[A], F, u), Period(per[B], F, u)
        boot = block_boot(pb.block, R)
        for v in VARS:
            fa, fb = pa.feat(F, v), pb.feat(F, v)
            if np.isnan(fa).all() or np.isnan(fb).all():
                continue
            for st in ("nat", "t3"):
                if st == "nat":
                    c = nat_cut(v, fa)
                    da = split_diff(fa, pa.S, pa.N, c); hb = fb > c; lb = fb <= c
                else:
                    c1, c2 = np.nanquantile(fa, [1 / 3, 2 / 3])
                    da = t3_diff(fa, pa.S, pa.N, c1, c2); hb = fb > c2; lb = fb <= c1
                ok = ~np.isnan(fb); hb &= ok; lb &= ok
                db = (hb * pb.S).sum() / max((hb * pb.N).sum(), 1) - (lb * pb.S).sum() / max((lb * pb.N).sum(), 1)
                bs = (boot * hb * pb.S).sum(1) / (boot * hb * pb.N).sum(1) - (boot * lb * pb.S).sum(1) / (boot * lb * pb.N).sum(1)
                Pb = float(np.nanmean(np.sign(bs) == np.sign(da)))
                out.append(dict(train=A, test=B, unit=u, var=v, stat=st, d_train=da, d_test=db, P_test=Pb,
                                confirm=bool(np.sign(da) == np.sign(db) and Pb >= 0.90)))
    return pd.DataFrame(out)


# ───────────────────────────── policy accounting
def boot_total(fills, skip, R2=2000):
    """7-day-block bootstrap of delta total (= -sum skipped y) over the fills' days."""
    d = fills.dt.values.astype("datetime64[D]").astype(np.int64)
    b = (d - d.min()) // 7
    contrib = np.where(skip, -fills.y.values, 0.0)
    ub = np.unique(b)
    per_b = np.array([contrib[b == x].sum() for x in ub])
    sims = per_b[RNG.integers(0, len(ub), (R2, len(ub)))].sum(1)
    return float((sims > 0).mean())


def account(fills, skip, label):
    y = fills.y.values
    keep = ~skip
    dtot = -y[skip].sum()
    wk = pd.Series(np.where(skip, -y, 0.0)).groupby(fills.week.values).sum()
    pos = wk[wk > 0]
    lomo = [(-(y[skip & (fills.month.values != m)]).sum()) for m in sorted(fills.month.unique())]
    return dict(policy=label, N=len(y), kept=int(keep.sum()), avg_all=y.mean(), avg_kept=y[keep].mean() if keep.any() else np.nan,
                d_fill=(y[keep].mean() - y.mean()) if keep.any() else np.nan, d_total=dtot,
                winners_lost=y[skip & (y > 0)].sum(), losers_avoided=-y[skip & (y < 0)].sum(),
                P_boot=boot_total(fills, skip), weeks_pos=int(len(pos)), max_week_share=float(pos.max() / pos.sum()) if len(pos) else np.nan,
                lomo_min=float(min(lomo)) if lomo else np.nan, lomo_all_pos=bool(all(x > 0 for x in lomo)))


def gate(a25, a26):
    ok = all(a["d_total"] > 0 and a["P_boot"] >= 0.90 and a["lomo_all_pos"] and a["weeks_pos"] >= 8 and a["max_week_share"] < 0.5
             for a in (a25, a26))
    return ok


def regime_skip(fills, F, v, cut, good_hi, atr_only=False):
    f = F[v].reindex(fills.day.values).values.astype(float)
    bad = np.where(np.isnan(f), False, (f <= cut) if good_hi else (f > cut))
    if atr_only:
        bad &= fills.atr3.values == "high"
    return bad


def monthly_wf(bt, F, v, atr_only=False, unit="1d"):
    """expanding monthly WF: each month's cut + side from all BT fills before it (first 3 months train-only)."""
    months = sorted(bt.month.unique())
    skip = np.zeros(len(bt), bool)
    test = np.zeros(len(bt), bool)
    for i, m in enumerate(months):
        if i < 3:
            continue
        tr = bt[bt.month < m]
        f = F[v].reindex(tr.day.values).values.astype(float)
        if np.isnan(f).all():
            continue
        c = nat_cut(v, F[v].reindex(sorted(tr.day.unique())).values.astype(float))
        ok = ~np.isnan(f)
        good_hi = tr.y.values[ok & (f > c)].mean() > tr.y.values[ok & (f <= c)].mean()
        mm = (bt.month == m).values
        skip[mm] = regime_skip(bt[mm], F, v, c, good_hi, atr_only)
        test[mm] = True
    return skip, test


# ───────────────────────────── equity curve
def equity_curve(bt, N, K=None, dayvar=None, F=None):
    bt = bt.sort_values("te")
    if dayvar:
        f = F[dayvar].reindex(bt.day.values).values.astype(float)
        return pd.Series(np.where(np.isnan(f), False, f < 0), index=bt.index)
    P = pd.read_csv(os.path.join(ROOT, "reports", "LITE_PROTECT_rows.csv"), usecols=["fid", "base_xm"])
    cl = bt.te.values + bt.fid.map(P.set_index("fid").base_xm).values * 60_000
    order = np.argsort(cl)
    cls, ys = cl[order], bt.y.values[order]
    paused = False; pause_t = None
    skip = np.zeros(len(bt), bool)
    for i, t in enumerate(bt.te.values):
        k = np.searchsorted(cls, t, side="left")
        done = ys[:k]
        if not paused:
            if len(done) >= N and done[-N:].mean() < 0:
                paused = True; pause_t = t
        if paused:
            since = ys[(cls >= pause_t) & (cls < t)]
            if len(since) >= K and since[-K:].mean() >= 0:
                paused = False
                if len(done) >= N and done[-N:].mean() < 0:   # re-check immediately with the full window (resume = K recovered)
                    pass
        skip[i] = paused
    return pd.Series(skip, index=bt.index)


# ───────────────────────────── BTC grids
def grid_labels(F, days, train_days, g):
    f = lambda v: F[v].reindex(days).values.astype(float)
    med = lambda v: float(np.nanmedian(F[v].reindex(train_days).values.astype(float)))
    sgn = lambda v: np.where(np.isnan(f(v)), "?", np.where(f(v) > 0, "+", "-"))
    hl = lambda v: np.where(np.isnan(f(v)), "?", np.where(f(v) > med(v), "hi", "lo"))
    if g == "G1":
        return np.char.add(np.char.add(np.char.add("r7", sgn("r7")), np.char.add("|ma50", sgn("dma50"))), np.char.add("|rv", hl("rv7")))
    if g == "G2":
        o = f("off30"); m = med("off30")
        band = np.where(np.isnan(o), "?", np.where(o > m, "near", np.where(o > -15, "far", "deep")))
        return np.char.add(np.char.add(np.char.add("r30", sgn("r30")), np.char.add("|", band)), np.char.add("|r3", sgn("r3")))
    if g == "G3":
        return np.char.add(np.char.add(np.char.add("1d", sgn("r1")), np.char.add("|7d", sgn("r7"))), np.char.add("|30d", sgn("r30")))
    if g == "G3c":
        a = np.char.add(np.char.add(sgn("r1"), sgn("r7")), sgn("r30"))
        return np.where(a == "+++", "aligned_up", np.where(a == "---", "aligned_down", np.where(np.char.find(a, "?") >= 0, "?", "mixed")))
    if g == "G4":
        return np.char.add(np.char.add("ma200", sgn("dma200")), np.char.add("|br20", hl("br_ma20")))
    if g == "G5":
        return np.char.add(np.char.add("r7", sgn("r7")), np.char.add("|ft12", hl("ft12")))
    if g == "G6":
        return np.char.add(np.char.add("rv", hl("rv7")), np.char.add("|altrs", sgn("alt_rs7")))


GRIDS = ["G1", "G2", "G3", "G3c", "G4", "G5", "G6"]


def cell_table(fills, lab):
    df = fills.assign(cell=lab)
    g = df.groupby("cell")
    t = pd.DataFrame({"N": g.size(), "days": g.day.nunique(), "weeks": g.week.nunique(), "WR": g.y.apply(lambda s: (s > 0).mean() * 100),
                      "avg": g.y.mean(), "tot": g.y.sum()})
    h = df[df.atr3 == "high"].groupby("cell")
    t["hiN"] = h.size(); t["hi_avg"] = h.y.mean()
    return t.fillna({"hiN": 0})


def eta(fills_y, lab):
    df = pd.DataFrame({"y": fills_y, "c": lab})
    m = df.groupby("c").y.agg(["mean", "size"])
    return float(((m["mean"] - df.y.mean()) ** 2 * m["size"]).sum() / ((df.y - df.y.mean()) ** 2).sum())


def grid_null(fills, F, g, own_days):
    days = sorted(fills.day.unique())
    alld = pd.date_range(min(days), max(days), freq="D").strftime("%Y-%m-%d").values
    lab_all = grid_labels(F, alld, own_days, g)
    pos = {d: i for i, d in enumerate(alld)}
    fi = np.array([pos[d] for d in fills.day.values])
    obs = eta(fills.y.values, lab_all[fi])
    blk = np.arange(len(alld)) // 7
    perm = block_perm(blk, R)
    nv = np.array([eta(fills.y.values, lab_all[perm[r]][fi]) for r in range(R)])
    return obs, float((nv >= obs).mean())


def fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:+.{d}f}" if isinstance(x, float) else str(x))


def main():
    P, F, (q1, q2) = load()
    bt = P[P.block == "BT"].copy()
    T(f"# LITE regime tables{' — TICK-PRICED BT ONLY' if TICK else ''} (generated {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC, R = {R})\n")
    T(f"ATR terciles (2025 BT, frozen): low ≤ {q1:.3f} < mid ≤ {q2:.3f} < high\n")
    # baseline per year
    T("## Baseline (today's exit, rm_base_pct)\n\n| block | N | days | weeks | WR % | avg %/fill | high-ATR N | high-ATR avg |\n|---|---|---|---|---|---|---|---|")
    for lab, d in (("BT 2025", bt[bt.yr == "2025"]), ("BT 2026", bt[bt.yr == "2026"]), ("BT 2026-H1", bt[bt.era == "2026-H1"]),
                   ("BT 2026-H2", bt[bt.era == "2026-H2"]), ("MASTER", P[P.block == "MASTER"]), ("LIVE", P[P.block == "LIVE"])):
        h = d[d.atr3 == "high"]
        T(f"| {lab} | {len(d)} | {d.day.nunique()} | {d.week.nunique()} | {(d.y > 0).mean()*100:.1f} | {d.y.mean():+.3f} | {len(h)} | {h.y.mean():+.3f} |")
    T()
    # ── screen
    S, fw, per, PER = screen(P, F)
    S.to_csv(os.path.join(OUT, "screen.csv".replace(".", SUF + ".", 1)), index=False)
    T("## Screen family-wise (block-shuffle null, 31 vars × 4 units per stat)\n\n| period | stat | max |z| | min fw-p |\n|---|---|---|---|")
    for pn, st, mz, mp in fw:
        T(f"| {pn} | {st} | {mz:.2f} | {mp:.3f} |")
    T()
    nsig = S.groupby(["period", "stat"]).apply(lambda d: int((d.p_null <= 0.05).sum()), include_groups=False)
    T("Tests with per-test null p ≤ 0.05 (expected by chance ≈ 6 of 124 per period×stat): " +
      " · ".join(f"{a}/{b} {n}" for (a, b), n in nsig.items()) + "\n")
    # ── confirm
    per["H1"] = bt[bt.era == "2026-H1"]; per["H2"] = bt[bt.era == "2026-H2"]
    C = pd.concat([confirm(per, F, "2025", "2026"), confirm(per, F, "2026", "2025"), confirm(per, F, "H1", "H2")], ignore_index=True)
    C.to_csv(os.path.join(OUT, "confirm.csv".replace(".", SUF + ".", 1)), index=False)
    both = []
    for (v, u, st), d in C[C.unit.isin(["1d", "3d"])].groupby(["var", "unit", "stat"]):
        a = d[(d.train == "2025")]; b = d[(d.train == "2026")]
        if len(a) and len(b) and a.confirm.iloc[0] and b.confirm.iloc[0]:
            p25 = S[(S.period == "2025") & (S["var"] == v) & (S.unit == u) & (S.stat == st)].p_null
            p26 = S[(S.period == "2026") & (S["var"] == v) & (S.unit == u) & (S.stat == st)].p_null
            both.append(dict(var=v, unit=u, stat=st, d25=a.d_test.iloc[0] if False else b.d_test.iloc[0], d26=a.d_test.iloc[0],
                             p25=float(p25.iloc[0]) if len(p25) else np.nan, p26=float(p26.iloc[0]) if len(p26) else np.nan,
                             h1h2=bool(C[(C.train == "H1") & (C["var"] == v) & (C.unit == u) & (C.stat == st)].confirm.any())))
    B = pd.DataFrame(both)
    surv = B[(B.p25 <= 0.10) & (B.p26 <= 0.10)] if len(B) else B
    T(f"## Discover → confirm\n\nconfirms 2025→2026: {int(C[(C.train=='2025')].confirm.sum())} / {int((C.train=='2025').sum())} · "
      f"2026→2025: {int(C[(C.train=='2026')].confirm.sum())} / {int((C.train=='2026').sum())} · H1→H2: {int(C[(C.train=='H1')].confirm.sum())} / "
      f"{int((C.train=='H1').sum())}\n")
    T("Confirmed BOTH ways at 1d/3d (any null p):\n\n| var | unit | stat | Δ 2025 (cut from 2026) | Δ 2026 (cut from 2025) | null p 2025 | null p 2026 | H1→H2 |\n|---|---|---|---|---|---|---|---|")
    for _, r in B.iterrows():
        T(f"| {r['var']} | {r.unit} | {r.stat} | {r.d25:+.3f} | {r.d26:+.3f} | {r.p25:.3f} | {r.p26:.3f} | {'yes' if r.h1h2 else 'no'} |")
    T(f"\nSURVIVORS (both ways + null p ≤ 0.10 in both years): {', '.join(sorted(set(surv['var']))) if len(surv) else 'none'}\n")
    # null expectation of 'both ways' count
    nb = []
    for r in range(min(R, 100)):
        sh = {}
        for k in ("2025", "2026"):
            d = per[k]
            days = sorted(d.day.unique())
            alld = pd.date_range(min(days), max(days), freq="D").strftime("%Y-%m-%d").values
            blk = np.arange(len(alld)) // 7
            pm = block_perm(blk, 1)[0]
            FF = F.reindex(alld)[VARS].copy()
            FF.iloc[:, :] = FF.values[pm]
            sh[k] = FF
        cnt = 0
        Fsh = pd.concat([sh["2025"], sh["2026"][~sh["2026"].index.isin(sh["2025"].index)]])
        for u in ("1d", "3d"):
            pa25, pa26 = Period(per["2025"], Fsh, u), Period(per["2026"], Fsh, u)
            for v in VARS:
                f25, f26 = pa25.feat(Fsh, v), pa26.feat(Fsh, v)
                if np.isnan(f25).all() or np.isnan(f26).all():
                    continue
                c25, c26 = nat_cut(v, f25), nat_cut(v, f26)
                d25 = split_diff(f25, pa25.S, pa25.N, c25); d26x = split_diff(f26, pa26.S, pa26.N, c25)
                d26 = split_diff(f26, pa26.S, pa26.N, c26); d25x = split_diff(f25, pa25.S, pa25.N, c26)
                if np.sign(d25) == np.sign(d26x) and np.sign(d26) == np.sign(d25x):
                    cnt += 1
        nb.append(cnt)
    nat_both = int(len(B[B.stat == "nat"]))
    T(f"Null check (100 block shuffles, NAT 1d/3d, sign-agreement both ways WITHOUT the P ≥ 0.90 bar): null mean {np.mean(nb):.1f}, "
      f"p95 {np.percentile(nb, 95):.0f}; observed sign-agreement-both-ways count = "
      f"{int(sum(1 for (v,u), d in C[(C.unit.isin(['1d','3d'])) & (C.stat=='nat')].groupby(['var','unit']) if len(d[d.train.isin(['2025','2026'])])==2 and (np.sign(d[d.train=='2025'].d_train.iloc[0])==np.sign(d[d.train=='2025'].d_test.iloc[0])) and (np.sign(d[d.train=='2026'].d_train.iloc[0])==np.sign(d[d.train=='2026'].d_test.iloc[0]))))}"
      f" (of {2*len(VARS)}); with P ≥ 0.90 both ways: {nat_both}\n")
    # ── per-variable readable table (NAT, 1d, per-fill) with cut from 2025
    T("## Per-variable NAT split (per fill, day-start regime, cut = 2025 train) — good/bad side per year\n")
    T("| var | cut | 2025 hi: N·WR·avg | 2025 lo: N·WR·avg | 2026 hi: N·WR·avg | 2026 lo: N·WR·avg | 2026 Δ(hi−lo) | 2025 Δ | high-ATR Δ 2025 | high-ATR Δ 2026 |\n|---|---|---|---|---|---|---|---|---|---|")
    vtab = []
    for v in VARS + ["fund3"]:
        tr = F[v].reindex(sorted(per["2025"].day.unique())).values.astype(float)
        if np.isnan(tr).all():
            tr = F[v].reindex(sorted(per["H1"].day.unique())).values.astype(float)
        c = nat_cut(v, tr)
        row = {"var": v, "cut": c}
        cells = []
        for yn in ("2025", "2026"):
            d = bt[bt.yr == yn]
            f = F[v].reindex(d.day.values).values.astype(float)
            for side, m in (("hi", f > c), ("lo", f <= c)):
                x = d[m & ~np.isnan(f)]
                cells.append(f"{len(x)}·{(x.y>0).mean()*100:.0f}·{x.y.mean():+.2f}" if len(x) else "—")
                row[f"{yn}_{side}_avg"] = x.y.mean() if len(x) else np.nan; row[f"{yn}_{side}_N"] = len(x)
            row[f"{yn}_d"] = row[f"{yn}_hi_avg"] - row[f"{yn}_lo_avg"]
            h = d[d.atr3 == "high"]; fh = F[v].reindex(h.day.values).values.astype(float)
            row[f"{yn}_hd"] = h.y[fh > c].mean() - h.y[fh <= c].mean()
        vtab.append(row)
        T(f"| {v} | {c:.3g} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {fmt(row['2026_d'])} | {fmt(row['2025_d'])} | {fmt(row['2025_hd'])} | {fmt(row['2026_hd'])} |")
    pd.DataFrame(vtab).to_csv(os.path.join(OUT, "var_splits.csv".replace(".", SUF + ".", 1)), index=False)
    T()
    # ATR-amplification summary: |Δ high| vs |Δ low|
    T("## Does the regime matter more for high-ATR coins? (NAT Δ per ATR tercile, 2025 cut)\n\n| var | 2025 low | 2025 mid | 2025 high | 2026 low | 2026 mid | 2026 high |\n|---|---|---|---|---|---|---|")
    amp = []
    for v in VARS:
        c = nat_cut(v, F[v].reindex(sorted(per["2025"].day.unique())).values.astype(float))
        vals = []
        for yn in ("2025", "2026"):
            for t3 in ("low", "mid", "high"):
                d = bt[(bt.yr == yn) & (bt.atr3 == t3)]
                f = F[v].reindex(d.day.values).values.astype(float)
                vals.append(d.y[f > c].mean() - d.y[f <= c].mean())
        amp.append([v] + vals)
        T(f"| {v} | " + " | ".join(fmt(x, 2) for x in vals) + " |")
    A = pd.DataFrame(amp, columns=["var", "l25", "m25", "h25", "l26", "m26", "h26"])
    T(f"\nmedian |Δ| across vars — low {np.nanmedian(np.abs(A[['l25','l26']].values)):.2f} · mid {np.nanmedian(np.abs(A[['m25','m26']].values)):.2f} · "
      f"high {np.nanmedian(np.abs(A[['h25','h26']].values)):.2f}\n")
    # ── policies for survivors
    pol = []
    def run_policy(label, skip_fn):
        r25 = account(bt[bt.yr == "2025"], skip_fn("2025"), label + " test2025")
        r26 = account(bt[bt.yr == "2026"], skip_fn("2026"), label + " test2026")
        pol.append(dict(**r25, gate=None)); pol.append(dict(**r26, gate=gate(r25, r26)))
    cand_vars = sorted(set(surv["var"])) if len(surv) else []
    for v in cand_vars:
        for atr_only in (False, True):
            lab = f"{v} {'P2 skip-highATR-when-bad' if atr_only else 'P1 trade-only-good'}"
            def sk(yn, v=v, atr_only=atr_only):
                tr = bt[bt.yr == ("2026" if yn == "2025" else "2025")]
                f = F[v].reindex(tr.day.values).values.astype(float)
                c = nat_cut(v, F[v].reindex(sorted(tr.day.unique())).values.astype(float))
                good_hi = tr.y.values[f > c].mean() > tr.y.values[f <= c].mean()
                te = bt[bt.yr == yn]
                return regime_skip(te, F, v, c, good_hi, atr_only)
            run_policy(lab + " [cross-year]", sk)
            skw, tst = monthly_wf(bt, F, v, atr_only)
            run_policy(lab + " [monthly WF]", lambda yn, skw=skw: skw[(bt.yr == yn).values])
    # equity-curve switches (always)
    for N_, K_ in ((10, 5), (10, 10), (20, 5), (20, 10)):
        sk = equity_curve(bt, N_, K_)
        run_policy(f"E pause last{N_}<0 resume last{K_}>=0", lambda yn, sk=sk: sk.reindex(bt[bt.yr == yn].index).values)
    for dv in ("lite_d3", "lite_d7"):
        sk = equity_curve(bt, 0, dayvar=dv, F=F)
        run_policy(f"E skip day when {dv}<0", lambda yn, sk=sk: sk.reindex(bt[bt.yr == yn].index).values)
    # ── BTC grids
    T("## BTC regime map (2D / 3D) — cells with cuts from the OTHER year (train) \n")
    grid_rows = []
    days25 = sorted(per["2025"].day.unique()); days26 = sorted(per["2026"].day.unique())
    for g in GRIDS:
        o25, p25 = grid_null(per["2025"], F, g, days25)
        o26, p26 = grid_null(per["2026"], F, g, days26)
        T(f"### {g} — eta² 2025 {o25:.4f} (null p {p25:.3f}) · 2026 {o26:.4f} (null p {p26:.3f})\n")
        T("| cell | 2025 N·days·wks | 2025 WR·avg | 2025 hiATR N·avg | 2026 N·days·wks | 2026 WR·avg | 2026 hiATR N·avg | sign agree | test-P (26 w/ 25 cut) |\n|---|---|---|---|---|---|---|---|---|")
        l25 = grid_labels(F, per["2025"].day.values, days25, g)
        l26c = grid_labels(F, per["2026"].day.values, days25, g)        # 2026 fills, 2025 cuts
        t25 = cell_table(per["2025"], l25); t26 = cell_table(per["2026"], l26c)
        cells = sorted(set(t25.index) | set(t26.index))
        d26 = per["2026"].assign(cell=l26c)
        for cl in cells:
            a = t25.loc[cl] if cl in t25.index else None; b = t26.loc[cl] if cl in t26.index else None
            agree = (a is not None and b is not None and np.sign(a.avg) == np.sign(b.avg))
            Pt = np.nan
            if b is not None and a is not None and b.weeks >= 8 and a.weeks >= 8:
                x = d26[d26.cell == cl]
                wsum = x.groupby("week").y.agg(["sum", "size"])
                bs = RNG.integers(0, len(wsum), (2000, len(wsum)))
                Pt = float((np.sign(wsum["sum"].values[bs].sum(1) / wsum["size"].values[bs].sum(1)) == np.sign(a.avg)).mean())
            thin = lambda z: z is None or z.weeks < 8
            s = lambda z: "—" if z is None else f"{int(z.N)}·{int(z.days)}·{int(z.weeks)}"
            w = lambda z: "—" if z is None else f"{z.WR:.0f}·{z.avg:+.2f}" + (" (thin)" if z.weeks < 8 else "")
            h = lambda z: "—" if z is None or not z.hiN else f"{int(z.hiN)}·{z.hi_avg:+.2f}"
            T(f"| {cl} | {s(a)} | {w(a)} | {h(a)} | {s(b)} | {w(b)} | {h(b)} | {'yes' if agree else 'no'} | {fmt(Pt, 2)} |")
            grid_rows.append(dict(grid=g, cell=cl, n25=None if a is None else a.N, avg25=None if a is None else a.avg,
                                  wk25=None if a is None else a.weeks, n26=None if b is None else b.N, avg26=None if b is None else b.avg,
                                  wk26=None if b is None else b.weeks, agree=agree, P26=Pt,
                                  confirm=bool(agree and not thin(a) and not thin(b) and Pt >= 0.90)))
        T()
        # policies
        for mode in ("trade", "skipATR"):
            def sk(yn, g=g, mode=mode):
                trn = per["2026"] if yn == "2025" else per["2025"]
                td = sorted(trn.day.unique())
                ltr = grid_labels(F, trn.day.values, td, g)
                tt = cell_table(trn, ltr)
                te = bt[bt.yr == yn]
                lte = grid_labels(F, te.day.values, td, g)
                if mode == "trade":
                    bad = {c for c in tt.index if tt.loc[c, "weeks"] >= 8 and tt.loc[c, "avg"] <= 0}
                    return np.isin(lte, list(bad))
                bad = {c for c in tt.index if tt.loc[c, "hiN"] >= 15 and tt.loc[c, "hi_avg"] < 0}
                return np.isin(lte, list(bad)) & (te.atr3.values == "high")
            run_policy(f"{g} {mode} [cross-year]", sk)
            # monthly WF
            months = sorted(bt.month.unique()); skw = np.zeros(len(bt), bool)
            for i, m in enumerate(months):
                if i < 3:
                    continue
                trn = bt[bt.month < m]; td = sorted(trn.day.unique())
                tt = cell_table(trn, grid_labels(F, trn.day.values, td, g))
                mm = (bt.month == m).values
                lte = grid_labels(F, bt[mm].day.values, td, g)
                if mode == "trade":
                    bad = {c for c in tt.index if tt.loc[c, "weeks"] >= 8 and tt.loc[c, "avg"] <= 0}
                    skw[mm] = np.isin(lte, list(bad))
                else:
                    bad = {c for c in tt.index if tt.loc[c, "hiN"] >= 15 and tt.loc[c, "hi_avg"] < 0}
                    skw[mm] = np.isin(lte, list(bad)) & (bt[mm].atr3.values == "high")
            run_policy(f"{g} {mode} [monthly WF]", lambda yn, skw=skw: skw[(bt.yr == yn).values])
    G = pd.DataFrame(grid_rows); G.to_csv(os.path.join(OUT, "grid_cells.csv".replace(".", SUF + ".", 1)), index=False)
    T(f"Grid cells confirmed 2025→2026 (≥ 8 weeks both, same sign, P ≥ 0.90): {int(G.confirm.sum())} of {len(G)}\n")
    # ── H1 -> H2 grid sign agreement
    T("### Grid cells H1 → H2 (cuts from H1)\n\n| grid | cells ≥8 wks both | sign agree |\n|---|---|---|")
    dH1 = sorted(per["H1"].day.unique())
    for g in GRIDS:
        a = cell_table(per["H1"], grid_labels(F, per["H1"].day.values, dH1, g)); b = cell_table(per["H2"], grid_labels(F, per["H2"].day.values, dH1, g))
        cc = [c for c in a.index if c in b.index and a.loc[c, "weeks"] >= 8 and b.loc[c, "weeks"] >= 8]
        T(f"| {g} | {len(cc)} | {sum(np.sign(a.loc[c,'avg'])==np.sign(b.loc[c,'avg']) for c in cc)} |")
    T()
    # ── policy table
    PO = pd.DataFrame(pol); PO.to_csv(os.path.join(OUT, "policies.csv".replace(".", SUF + ".", 1)), index=False)
    T("## Policies (skips = 0; test year shown; gate on the 2026 row = both years)\n\n| policy | N | kept | avg all | avg kept | Δ %/fill | Δ total % | winners lost | losers avoided | P(Δ>0) | weeks + | max wk share | LOMO min | gate |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in PO.iterrows():
        T(f"| {r.policy} | {r.N} | {r.kept} | {r.avg_all:+.3f} | {fmt(r.avg_kept)} | {fmt(r.d_fill)} | {r.d_total:+.1f} | {r.winners_lost:+.1f} | {r.losers_avoided:+.1f} | {r.P_boot:.2f} | {r.weeks_pos} | {fmt(r.max_week_share,2)} | {r.lomo_min:+.1f} | {'' if r.gate is None else ('PASS' if r.gate else 'fail')} |")
    T()
    # ── master + live refute-only: grids + survivors, cuts from all BT
    T("## Master + live (refute-only) — cells / sides with cuts from all BT\n\n| fill | day | y | " + " | ".join(GRIDS) + " | r7 | dma50 | off30 | ft12 | spk72 |\n|---|---|---|" + "---|" * (len(GRIDS) + 5))
    ML = P[P.block.isin(["MASTER", "LIVE"])].sort_values("te")
    alld = sorted(bt.day.unique())
    labs = {g: grid_labels(F, ML.day.values, alld, g) for g in GRIDS}
    for i, (_, r) in enumerate(ML.iterrows()):
        fv = F.reindex([r.day]).iloc[0]
        T(f"| {r.block} {r.pair} | {r.day} | {r.y:+.2f} | " + " | ".join(labs[g][i] for g in GRIDS) +
          f" | {fv.r7:+.1f} | {fv.dma50:+.1f} | {fv.off30:+.1f} | {fv.ft12:.2f} | {fv.spk72:.0f} |")
    T()
    # ── washed-out band descriptive
    T("## Washed-out band (BTC 15–24 % below 30d high at day start) — descriptive\n\n| block | in band N·WR·avg | outside N·WR·avg | band days |\n|---|---|---|---|")
    for lab, d in (("BT 2025", bt[bt.yr == "2025"]), ("BT 2026", bt[bt.yr == "2026"])):
        o = F.off30.reindex(d.day.values).values
        m = (o <= -15) & (o >= -24)
        x, y_ = d[m], d[~m]
        T(f"| {lab} | {len(x)}·{(x.y>0).mean()*100 if len(x) else 0:.0f}·{x.y.mean() if len(x) else float('nan'):+.2f} | {len(y_)}·{(y_.y>0).mean()*100:.0f}·{y_.y.mean():+.2f} | {x.day.nunique()} |")
    T()
    # ── weekly / monthly timeline
    W = bt.groupby("month").agg(N=("y", "size"), WR=("y", lambda s: (s > 0).mean() * 100), avg=("y", "mean"))
    Fm = F.copy(); Fm["month"] = Fm.index.str[:7]
    fm = Fm.groupby("month")[["r30", "dma50", "off30", "rv7", "br_ma20", "alt_rs7", "spk72", "ft12"]].mean()
    W = W.join(fm)
    T("## Monthly timeline (LITE outcome vs mean day-start regime)\n\n| month | N | WR | avg | BTC r30 | dma50 | off30 | rv7 | br_ma20 | alt_rs7 | spk72 | ft12 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, r in W.iterrows():
        T(f"| {m} | {r.N} | {r.WR:.0f} | {r.avg:+.2f} | {r.r30:+.1f} | {r.dma50:+.1f} | {r.off30:+.1f} | {r.rv7:.2f} | {r.br_ma20:.2f} | {r.alt_rs7:+.1f} | {r.spk72:.0f} | {r.ft12:.2f} |")
    mm = W.dropna()
    T("\nMonth-level Spearman (N months = %d) of avg vs: " % len(mm) + " · ".join(
        f"{c} {np.corrcoef(mm[c].rank(), mm['avg'].rank())[0,1]:+.2f}" for c in ["r30", "dma50", "off30", "rv7", "br_ma20", "alt_rs7", "spk72", "ft12"]))
    T()
    open(os.path.join(OUT, "tables.md".replace(".", SUF + ".", 1)), "w").write("\n".join(TABLES))
    json.dump({"survivors": cand_vars, "fw": fw, "grid_confirm": int(G.confirm.sum()), "gates_passed": [r["policy"] for r in pol if r.get("gate")]},
              open(os.path.join(OUT, "summary.json".replace(".", SUF + ".", 1)), "w"), indent=1, default=str)
    print("\n".join(TABLES[-80:]))
    print("SURVIVORS", cand_vars, "GATES", [r["policy"] for r in pol if r.get("gate")])


if __name__ == "__main__":
    main()
