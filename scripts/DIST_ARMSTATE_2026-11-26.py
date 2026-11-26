#!/usr/bin/env python3
"""DIST_ARMSTATE — at the minute a DISTRIBUTION_SHORT first reaches +2 %, does the coin / BTC STATE tell a runner from a give-back?
(operator 2026-11-26, after DIST_TRAIL1: "Trail start +2 %, 0.5×ATR — nothing separates winners vs losers?")

PRIOR (not repeated): ENTRY_WL 2026-11-18 — entry-time RUNNER vs GAVE_BACK on DIST: nothing (thin-coin observe only). STATE_EXIT 2026-11-21 —
in-trade state exits on EXH / FB / HT: nothing. NEW: DIST, in-trade state AT THE +2 % ARM, as a switch for the DIST_TRAIL1 trail.

PRE-REGISTERED (frozen before any feature × outcome is read):
  cohort   = DIST_EXIT PRIMARY one-position fills under Exit B (REF) that reach +2 % (1m low ≤ entry × 0.98) before the Exit-B stop.
  decision = the close of the first such 1m bar (t_a). Only data with open < t_a + 1 min is used.
  outcome  = D = net(trail arm 2 % / 0.5 × ATR15) − net(Exit B) per fill (DIST_TRAIL1 engine, same costs). A rule = "arm the trail only
             when the state is in zone"; its value = mean D over the zone fills (> 0 = the trail helps there).
  features = in-trade (minutes to +2 %, adverse excursion before it, speed); coin 1m (returns 5 / 15 / 60 m, quote-volume 15 m ÷ prior
             60 m and ÷ the entry-hour mean, taker-buy share 15 / 60 m, RSI14 on 1m / 5m / 15m closes, 5m EMA20 distance, 15 m range ÷ ATR15);
             5m metrics (OI change since entry, top-trader position L/S change since entry, taker L/S last 15 m); BTC (return 15 / 60 m, 1m RSI).
  screen   = each feature at DESIGN-frozen sign (where signed) / median / terciles / quintiles (zone = one bin), + every 2D median quadrant;
             statistic t of mean D in the zone (zone N ≥ 30). Null = 300 shuffles of D across fills within the same calendar month, family-wise
             max t (DESIGN). CANDIDATE only if: beats the DESIGN null · HOLDOUT mean D > 0 with 10-day-block 95 % lower bound > 0 · ≥ 5 / 7
             half-years D > 0 in zone · real DIST fills (batch) not contradicting. Else: "no state switch" + coverage / blind spots.
Phases: --phase arm (arm minutes + D) · --phase fetch (vision 1m w/ volume, metrics, BTC) · --phase eval
Out: reports/DIST_ARMSTATE_2026-11-26{_screen.csv,_forward.csv} (+ _arm.pkl / _rows.pkl, not committed); write-up in reports/DIST_TRAIL1_2026-11-26.md.
Coverage: a fill whose sim fell back to 5m (no 1m) has no arm minute and is left out of the armed cohort. BTC returns assume no 1m gaps."""
import argparse, os, sys, time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dist_exit_sim as DS      # noqa: E402
import dist_exit_eval as EV     # noqa: E402
import dist_wl_survivors as SV  # noqa: E402
import importlib.util as _iu
_s = _iu.spec_from_file_location("SXC", os.path.join(HERE, "STATE_EXIT_2026-11-21_common.py")); SX = _iu.module_from_spec(_s); _s.loader.exec_module(SX)
R = DS.R; RP = os.path.join(R, "reports"); OUT = os.path.join(RP, "DIST_ARMSTATE_2026-11-26")
DAY, H1, MIN = 86_400_000, 3_600_000, 60_000
CL2 = [dict(name="REF_B", fam="REF", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="none", arm=np.inf, par=0.0, hold=24),
       dict(name="ATR_arm2_x0.5", fam="TA", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="trailatr", arm=2.0, par=0.5, hold=24)]


def arm_minute(p, rows):
    """per row: first 1m bar open time with low ≤ E0 × 0.98 before the Exit-B stop (stop checked first on a shared bar); -1 if none."""
    days = sorted({int(T) // DAY + k for T in rows.entry_ms.values for k in range(2)})
    one, _ = DS.load1m(p, days)
    out = np.full(len(rows), -1, np.int64); adv = np.full(len(rows), np.nan)
    if one is None:
        return out, adv
    t, o, h, l, c = one
    for i, (T, atrp) in enumerate(zip(rows.entry_ms.values.astype(np.int64), rows.atrp.values)):
        a = int(np.searchsorted(t, T)); b = int(np.searchsorted(t, T + 24 * H1))
        if a >= len(t) or t[a] != T:
            continue
        E = o[a]; S = E * (1 + min(1.5 * atrp, 6.0) / 100)
        hs = h[a:b] >= S; ar = l[a:b] <= E * 0.98
        fs = np.argmax(hs) if hs.any() else 10**9; fa = np.argmax(ar) if ar.any() else 10**9
        if fa < fs:
            out[i] = t[a + fa]; adv[i] = (h[a:a + fa + 1].max() / E - 1) * 100
    return out, adv


def w_arm(a):
    p, rows = a
    out, _ = DS.run_rows(p, rows, CL2)
    am, adv = arm_minute(p, rows)
    return rows.index.values, out, am, adv


def phase_arm(procs):
    S = pd.read_pickle(os.path.join(RP, "DIST_WL_2026-11-18_data", "rows.pkl"))
    F = DS.frames()["L1"]; prim = SV.keep_mask(S, EV.REP)
    Fp = F[prim].copy(); Sp = S[prim].reset_index(drop=True); Fp = Fp.reset_index(drop=True)
    n = len(Fp); r = np.full((n, 2), np.nan); k = np.full((n, 2), -1); xm = np.zeros((n, 2), np.int64); f = np.zeros((n, 2)); ok = np.zeros(n, bool)
    am = np.full(n, -1, np.int64); adv = np.full(n, np.nan)
    jobs = [(p, Fp[Fp.pair == p]) for p in Fp.pair.unique()]
    with mp.get_context("fork").Pool(procs) as pool:
        for ix, out, a_, d_ in pool.imap_unordered(w_arm, jobs, chunksize=2):
            r[ix] = out["r"]; k[ix] = out["k"]; xm[ix] = out["xm"]; f[ix] = out["f"]; ok[ix] = out["ok"]; am[ix] = a_; adv[ix] = d_
    Z = dict(r=r, k=k.astype(np.int8), f=f, ok=ok)
    slip = Fp.slip.values
    tk = Z["k"] != 0; s = slip[:, None]
    V = r - s - np.where(tk, s, 0) - 0.05 - np.where(tk, 0.05, 0.02) + f
    V = np.where((~ok[:, None]) | (Z["k"] == 9) | (Z["k"] < 0), np.nan, V)
    T = Fp.entry_ms.values.astype(np.int64)
    keep = EV.one_pos(Fp.pair.values, T, xm[:, 0], np.isfinite(V[:, 0]) & (T < EV.HOL[1]))
    X = pd.DataFrame(dict(pair=Fp.pair.values, entry_ms=T, atrp=Fp.atrp.values, ref=V[:, 0], trail=V[:, 1], D=V[:, 1] - V[:, 0],
                          k_ref=Z["k"][:, 0], k_tr=Z["k"][:, 1], arm_ms=am, adv_pre=adv, keep=keep, vol24=Sp.vol24.values))
    X.to_pickle(OUT + "_arm.pkl")
    A = X[X.keep & (X.arm_ms > 0) & np.isfinite(X.D)]
    print("kept", int(X.keep.sum()), "armed", len(A), "mean D armed", round(A.D.mean(), 3), "| never-armed |D| max",
          round(X[X.keep & (X.arm_ms < 0)].D.abs().max(), 4))


def phase_fetch(th):
    X = pd.read_pickle(OUT + "_arm.pkl"); A = X[X.keep & (X.arm_ms > 0)]
    need = set(); btc = set()
    for p, T, ta in zip(A.pair, A.entry_ms, A.arm_ms):
        for ds in SX.days_of(T - 2 * H1, ta + MIN):
            need.add((p, ds)); btc.add(("BTCUSDT", ds))
    jobs = sorted(need | btc); print("kline days", len(jobs), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(th) as ex:
        for i, _ in enumerate(ex.map(lambda j: SX.fetch_kday(*j), jobs)):
            if i % 500 == 0:
                print("k", i, f"{time.time() - t0:.0f}s", flush=True)
    mj = sorted(need)
    with ThreadPoolExecutor(th) as ex:
        for i, _ in enumerate(ex.map(lambda j: SX.fetch_mday(*j), mj)):
            if i % 500 == 0:
                print("m", i, f"{time.time() - t0:.0f}s", flush=True)
    print("fetch done", flush=True)


def rsi(c, n=14):
    d = np.diff(c)
    if len(d) < n:
        return np.nan
    g = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().values[-1]
    lo = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().values[-1]
    return 100.0 if lo == 0 else 100 - 100 / (1 + g / lo)


def feats(row):
    p, T, ta, atrp = row.pair, int(row.entry_ms), int(row.arm_ms), float(row.atrp)
    tend = ta + MIN                                         # bars with open < tend (the arm bar included)
    a = SX.k1m(p, T - 2 * H1, tend)
    o = dict()
    if a is None or len(a["t"]) < 30:
        return o
    g = SX.grid(a, T - 2 * H1, tend); c = g["c"]; qv = g["qv"]; tb = g["tbq"]
    if not np.isfinite(c[-1]):
        return o
    ie = int(np.searchsorted(g["t"], T))
    o["min_to_arm"] = (ta - T) / MIN + 1
    o["adv_pre"] = float(row.adv_pre)
    o["speed"] = 2.0 / max(o["min_to_arm"], 1)
    for m in (5, 15, 60):
        if len(c) > m and np.isfinite(c[-1 - m]):
            o[f"ret{m}"] = (c[-1] / c[-1 - m] - 1) * 100
    q15, q60 = qv[-15:].sum(), qv[-75:-15].sum() / 4
    o["qv15_vs_prev60"] = q15 / q60 if q60 > 0 else np.nan
    pre = qv[max(ie - 60, 0):ie].sum() / 4
    o["qv15_vs_entryhr"] = q15 / pre if pre > 0 else np.nan
    o["tb15"] = tb[-15:].sum() / q15 if q15 > 0 else np.nan
    o["tb60"] = tb[-60:].sum() / qv[-60:].sum() if qv[-60:].sum() > 0 else np.nan
    o["rsi1m"] = rsi(c[-60:])
    c5 = c[(len(c) - 1) % 5::5]; o["rsi5m"] = rsi(c5[-40:])
    c15 = c[(len(c) - 1) % 15::15]; o["rsi15m"] = rsi(c15)
    e20 = pd.Series(c5).ewm(span=20, adjust=False).mean().values[-1]; o["ema20_5m_dist"] = (c[-1] / e20 - 1) * 100
    o["rng15_atr"] = (g["h"][-15:].max() / g["l"][-15:].min() - 1) * 100 / atrp if atrp > 0 else np.nan
    m = SX.metrics(p, T - H1, tend)
    if m is not None and len(m["t"]):
        i0 = np.searchsorted(m["t"], T, "right") - 1; i1 = np.searchsorted(m["t"], ta, "right") - 1   # create_time ≤ t (known)
        if i0 >= 0 and i1 >= 0:
            o["oi_chg"] = (m["oi"][i1] / m["oi"][i0] - 1) * 100 if m["oi"][i0] > 0 else np.nan
            o["toppos_chg"] = m["top_pos"][i1] - m["top_pos"][i0]
            o["taker_ls"] = np.nanmean(m["taker_ls"][max(i1 - 2, 0):i1 + 1])
    b = SX.k1m("BTCUSDT", ta - 2 * H1, tend)
    if b is not None and len(b["t"]) > 61:
        bc = b["c"]; o["btc_ret15"] = (bc[-1] / bc[-16] - 1) * 100; o["btc_ret60"] = (bc[-1] / bc[-61] - 1) * 100; o["btc_rsi1m"] = rsi(bc[-60:])
    return o


def w_feat(rows):
    return [(i, feats(r)) for i, r in rows.iterrows()]


def tstat(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def phase_eval(procs):
    X = pd.read_pickle(OUT + "_arm.pkl"); A = X[X.keep & (X.arm_ms > 0) & np.isfinite(X.D)].copy()
    chunks = [A.iloc[i::procs * 4] for i in range(procs * 4)]
    with mp.get_context("fork").Pool(procs) as pool:
        res = [x for part in pool.map(w_feat, chunks) for x in part]
    Fd = pd.DataFrame({i: o for i, o in res}).T
    A = A.drop(columns=["adv_pre"]).join(Fd.apply(pd.to_numeric, errors="coerce"))
    A["day"] = A.entry_ms // DAY; A["half"] = EV.half_of(A.entry_ms.values); A["des"] = A.entry_ms < EV.DES[1]
    A["month"] = pd.to_datetime(A.entry_ms, unit="ms").dt.strftime("%Y-%m")
    A["runner"] = A.k_ref == 0
    A.to_pickle(OUT + "_rows.pkl")
    FE = [c for c in Fd.columns]
    cov = {c: float(A[c].notna().mean()) for c in FE}
    print("armed", len(A), "DESIGN", int(A.des.sum()), "runner share", round(A.runner.mean(), 3), "mean D", round(A.D.mean(), 3))
    print("coverage", {k: round(v, 2) for k, v in cov.items()})
    De = A[A.des]
    zones = []   # (name, function(df) -> bool mask)
    for c in FE:
        v = De[c].dropna()
        if len(v) < 100:
            continue
        if c in ("ret5", "ret15", "ret60", "ema20_5m_dist", "oi_chg", "toppos_chg", "btc_ret15", "btc_ret60"):
            zones += [(f"{c}>0", c, 0, np.inf), (f"{c}<=0", c, -np.inf, 0)]
        for q in (2, 3, 5):
            ed = np.quantile(v, np.linspace(0, 1, q + 1)); ed[0], ed[-1] = -np.inf, np.inf
            for j in range(q):
                zones.append((f"{c} Q{j + 1}/{q}", c, ed[j], ed[j + 1]))
    med = {c: De[c].median() for c in FE if De[c].notna().sum() >= 100}
    MC = list(med)
    two = []
    for i in range(len(MC)):
        for j in range(i + 1, len(MC)):
            for s1 in (0, 1):
                for s2 in (0, 1):
                    two.append((MC[i], s1, MC[j], s2))

    def masks(df):
        M = {}
        for nm, c, lo, hi in zones:
            M[nm] = ((df[c] > lo) & (df[c] <= hi)).values
        for c1, s1, c2, s2 in two:
            a_ = (df[c1] > med[c1]) if s1 else (df[c1] <= med[c1]); b_ = (df[c2] > med[c2]) if s2 else (df[c2] <= med[c2])
            M[f"{c1}{'>' if s1 else '<='}med & {c2}{'>' if s2 else '<='}med"] = (a_ & b_).values
        return M
    MD = masks(De); Dv = De.D.values
    names = [k for k, m in MD.items() if m.sum() >= 30]
    Mm = np.array([MD[k] for k in names])
    def tvec(d):
        n = Mm.sum(1); s = (Mm * d).sum(1); ss = (Mm * d * d).sum(1)
        mu = s / n; sd = np.sqrt(np.maximum(ss / n - mu ** 2, 1e-12) * n / (n - 1))
        return mu / (sd / np.sqrt(n)), mu, n
    t_obs, mu, nz = tvec(Dv)
    rng = np.random.default_rng(1126); mon = De.month.values; mx = []
    for _ in range(300):
        d = Dv.copy()
        for mth in np.unique(mon):
            ii = np.nonzero(mon == mth)[0]; d[ii] = d[rng.permutation(ii)]
        mx.append(tvec(d)[0].max())
    bar = np.percentile(mx, 95)
    Sc = pd.DataFrame(dict(zone=names, N=nz, meanD=mu, t=t_obs)).sort_values("t", ascending=False)
    Sc["beats_null"] = Sc.t > bar
    # forward reads for the top 10 DESIGN zones (positive mean D)
    MH = masks(A); Ho = ~A.des.values
    rows = []
    for _, z in Sc.head(10).iterrows():
        m = MH[z.zone]; mh = m & Ho
        lo, hi, _ = EV.block_boot_mean(A.day.values[mh], A.D.values[mh], EV.HOL, nb=2000, rng=np.random.default_rng(7))
        hv = {h: A.D.values[m & (A.half.values == h)].mean() for h, _, _ in EV.HALVES}
        rows.append(dict(zone=z.zone, des_N=int(z.N), des_D=z.meanD, des_t=z.t, beats=z.beats_null,
                         hol_N=int(mh.sum()), hol_D=A.D.values[mh].mean(), hol_lo=lo, hol_hi=hi,
                         halves_pos=int(sum(v > 0 for v in hv.values() if np.isfinite(v))), **{f"h_{k}": v for k, v in hv.items()},
                         runner_share_zone=A.runner.values[m].mean(), runner_share_rest=A.runner.values[~m].mean()))
    FR = pd.DataFrame(rows)
    Sc.to_csv(OUT + "_screen.csv", index=False); FR.to_csv(OUT + "_forward.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print("tests", len(names), "null 95% max t", round(bar, 2), "best t", round(t_obs.max(), 2), "survivors", int(Sc.beats_null.sum()))
    print(FR.round(2).to_string())
    # 1D AUC-ish: runner share by feature tercile (descriptive)
    for c in FE:
        if c in med:
            q = pd.qcut(A[c], 3, duplicates="drop")
            print(c, A.groupby(q, observed=True).agg(N=("D", "size"), runner=("runner", "mean"), D=("D", "mean")).round(2).to_dict("index"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    A = ap.parse_args()
    {"arm": lambda: phase_arm(A.procs), "fetch": lambda: phase_fetch(A.threads), "eval": lambda: phase_eval(A.procs)}[A.phase]()


if __name__ == "__main__":
    main()
