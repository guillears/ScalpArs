#!/usr/bin/env python3
"""🪶 Oct-8 FRENZY_LITE entry-signs study (live loser SKLUSDT 2026-10-08 21:40) — three PRE-REGISTERED signs, read-only research.

PRE-REGISTRATION (written before any outcome was read; the grid below is the whole family — no other thresholds are mined):
  Cohort MAIN  = FRENZY_LITE study cohort (scratch lite_streak/kept.pkl K[12], 724 fills) minus bearish-day fills = 548 fills,
                 outcome = tp34 walk_c 'fix3' (live ruler: first print ≥ signal close + 8 s, E × 1.001 slip, bot fees, +3/−3/12 h).
  Cohort ROBUST = all 724 (bearish days included).
  Features on the SIGNAL bar (closed 5m bar the engine judges; k5m_full, no look-ahead):
    S1  dd30 / dd60 = signal close vs the max HIGH of the 6 / 12 5m bars ending at the signal bar (incl.), % below (≥ 0)
        burst60 = max over 3-bar (15-min) windows fully inside the last 12 bars of (max high in the window ÷ the window's first open − 1)
        tests: dd30 ≥ 2 / ≥ 3 / ≥ 4 · dd60 ≥ 2 / ≥ 3 / ≥ 4 · burst60 ≥ 3          buckets: dd30 0-1-2-3-4-∞
    S2  above_streak == 12 vs > 12 · 12–14 vs ≥ 15 · vs-VWAP % terciles (T1 low / T2 / T3 high, cut on MAIN)
        buckets: streak 12 / 13–14 / 15–24 / ≥ 25
    S3  vol_trend (engine frenzy_vol_trend: Σ v·c of the last 12 closed 5m bars ÷ the 12 before) > 1 (sign) / ≥ 2 / ≥ 3
        v1h24 = quote volume of the last 12 bars ÷ (quote volume of the last 288 bars / 24) > 1 (sign) / ≥ 2 / ≥ 3
        buckets: vol_trend <0.7 / 0.7–1 / 1–2 / 2–3 / ≥3 · v1h24 quartiles
  Primary flags for combinations: S1p = dd30 ≥ 3 · S2p = streak == 12 · S3p = vol_trend ≥ 2 → S1p∧S2p, S1p∧S3p, S2p∧S3p, all three.
  Per test: N, WR, avg %, Σ %, days, day-clustered bootstrap 95 % CI + P(mean < 0), Δ vs rest, top pair/day share of the losers' loss,
  H1/H2, leave-one-month-out. Family null: day-block shuffle of the joint flag vectors (each day's outcomes get another day's flag
  block, resampled to size) → per-test p and family-wise p on max |z|.
  Block verdict = locked expectancy bar (CLAUDE.md, Sep-25) on MAIN: WR < LITE breakeven WR · P(mean<0) ≥ 0.95 (day-clustered) ·
  ≥ 8 days · N ≥ 15 · no day/pair ≥ 50 % of the cohort loss; then 30–50 % haircut.
  Stops: share of 'flush' stops (largest 60-s drop inside the 5 min before the stop print ≥ 2 %) vs gradual; recovery to E within 60 min.
Outputs: reports/LITE_ENTRY_SIGNS_STUDY_2026-10-08.csv (per fill) + _tests.csv; scratch lite_signs/tables.md.
Usage: venv/bin/python scripts/study_lite_signs_main.py"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_tp34_common as C                                   # noqa: E402

SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
OUT = os.path.join(SCR, "lite_signs"); os.makedirs(OUT, exist_ok=True)
REP = os.path.join(ROOT, "reports")
K5 = os.path.join(REP, "backtest_cache", "k5m_full")
B5 = 300_000
RNG = np.random.default_rng(20261008)


def bar_feats(d, bar_open):
    t = d.open_time.values.astype("int64")
    o, h, c, v, q = (d[k].values.astype(float) for k in ("o", "h", "c", "vol", "qvol"))
    out = []
    for bo in bar_open:
        i = int(np.searchsorted(t, bo))
        if i >= len(t) or t[i] != bo or i < 300:
            out.append(dict(feat_ok=False)); continue
        hi30, hi60 = h[i - 5:i + 1].max(), h[i - 11:i + 1].max()
        bur = max((h[s:s + 3].max() / o[s] - 1) * 100 for s in range(i - 11, i - 1))
        vc = v * c
        vt = vc[i - 11:i + 1].sum() / vc[i - 23:i - 11].sum() if vc[i - 23:i - 11].sum() > 0 else np.nan
        v24 = q[i - 287:i + 1].sum() / 24.0
        out.append(dict(feat_ok=True, sig_close=c[i], dd30=(1 - c[i] / hi30) * 100, dd60=(1 - c[i] / hi60) * 100, burst60=bur,
                        vol_trend=vt, v1h24=(q[i - 11:i + 1].sum() / v24 if v24 > 0 else np.nan)))
    return pd.DataFrame(out)


def build():
    K = pd.read_pickle(os.path.join(SCR, "lite_streak", "kept.pkl"))["K"][12].copy().reset_index(drop=True)
    c = pd.read_pickle(os.path.join(SCR, "tp34", "cohort_c.pkl")); w = pd.read_pickle(os.path.join(SCR, "tp34", "walk_c.pkl"))
    c["key"] = c.pair + "|" + c.sig.astype("int64").astype(str)
    K["key"] = K.pair + "|" + K.entry_ts.astype("int64").astype(str)
    K = K.merge(c[["key", "bear"]], on="key", how="left").merge(w[["key", "fix3", "fix3_how", "fix3_xms", "E", "entry_ms", "pk_before_stop"]], on="key")
    assert len(K) == 724 and K.bear.notna().all()
    Y = pd.read_pickle(os.path.join(SCR, "manualall", "yw_all.pkl"))[["pair", "bar_open", "vs_vwap", "vol_trend", "above_streak"]]
    Y = Y.rename(columns={"vol_trend": "yw_vol_trend", "above_streak": "yw_streak"})
    K = K.merge(Y, on=["pair", "bar_open"], how="left")
    parts = []
    for pair, g in K.groupby("pair"):
        d = pd.read_csv(os.path.join(K5, f"{pair}.csv")).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        f = bar_feats(d, g.bar_open.values); f.index = g.index; parts.append(f)
    K = K.join(pd.concat(parts))
    return K


def flush_stats(K):
    rows = []
    for r in K[K.fix3_how == "SL"].itertuples():
        xs = int(r.fix3_xms)
        x = C.ticks(r.pair, xs - 5 * C.MIN, xs + 60 * C.MIN)
        if x is None or len(x[0]) < 5:
            rows.append(dict(key=r.key, fl_ok=False)); continue
        t, p = x
        pre = t <= xs
        s = pd.Series(p[pre], index=pd.to_datetime(t[pre], unit="ms"))
        rmax = s.rolling("60s").max().values
        drop60 = float(np.max(1 - s.values / rmax) * 100)
        post = p[t > xs]
        rec = bool(len(post) and post.max() >= r.E)
        trec = (int(t[t > xs][np.argmax(post >= r.E)]) - xs) / 60000 if rec else np.nan
        rows.append(dict(key=r.key, fl_ok=True, drop60_max=drop60, flush=drop60 >= 2.0, recover60=rec, rec_min=trec,
                         min_to_stop=(xs - r.entry_ms) / 60000))
    return pd.DataFrame(rows)


def boot(v, d, B=4000):
    v = np.asarray(v, float); d = np.asarray(d)
    g = pd.DataFrame({"v": v, "d": d}).groupby("d").v.agg(["sum", "count"])
    s, c, k = g["sum"].values, g["count"].values, len(g)
    i = RNG.integers(0, k, (B, k))
    m = s[i].sum(1) / c[i].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), float((m < 0).mean())


def conc(z):
    L = z[z.y < 0]
    tot = -L.y.sum()
    if tot <= 0:
        return np.nan, "", np.nan, ""
    p = (-L.groupby("pair").y.sum()).sort_values(ascending=False); dd = (-L.groupby("day").y.sum()).sort_values(ascending=False)
    return p.iloc[0] / tot, p.index[0], dd.iloc[0] / tot, dd.index[0]


def stat(Z, m, name, be_wr):
    z, r = Z[m], Z[~m]
    n = len(z)
    if n < 3:
        return dict(test=name, N=n)
    lo, hi, p0 = boot(z.y, z.day)
    pc, pp, dc, dp = conc(z)
    h = {hh: z[z.half == hh].y for hh in ("H1", "H2")}
    lomo = []
    for mo in sorted(Z.month.unique()):
        k = Z.month != mo
        a, b = Z[k & m].y, Z[k & ~m].y
        if len(a) and len(b):
            lomo.append((a.mean(), a.mean() - b.mean()))
    lomo = np.array(lomo)
    wr = (z.y > 0).mean() * 100
    bar = dict(wr=wr < be_wr, conf=p0 >= 0.95, days=z.day.nunique() >= 8, n=n >= 15, conc=(pc < 0.5 and dc < 0.5))
    return dict(test=name, N=n, share=n / len(Z) * 100, WR=wr, avg=z.y.mean(), sum=z.y.sum(), days=z.day.nunique(), ci_lo=lo, ci_hi=hi,
                p_mean_lt0=p0, rest_N=len(r), rest_avg=r.y.mean(), delta=z.y.mean() - r.y.mean(),
                H1_n=len(h["H1"]), H1_avg=h["H1"].mean(), H2_n=len(h["H2"]), H2_avg=h["H2"].mean(),
                lomo_avg_min=lomo[:, 0].min(), lomo_avg_max=lomo[:, 0].max(), lomo_delta_min=lomo[:, 1].min(), lomo_delta_max=lomo[:, 1].max(),
                top_pair=pp, top_pair_loss_share=pc, top_day=dp, top_day_loss_share=dc,
                bar_pass=all(bar.values()), bar_detail=" ".join(f"{k}:{'Y' if v else 'n'}" for k, v in bar.items()))


def flags(Z, t1, t2):
    F = {
        "S1 dd30>=2": Z.dd30 >= 2, "S1 dd30>=3": Z.dd30 >= 3, "S1 dd30>=4": Z.dd30 >= 4,
        "S1 dd60>=2": Z.dd60 >= 2, "S1 dd60>=3": Z.dd60 >= 3, "S1 dd60>=4": Z.dd60 >= 4, "S1 burst60>=3": Z.burst60 >= 3,
        "S2 streak==12": Z.above_streak == 12, "S2 streak12-14": Z.above_streak <= 14,
        "S2 vsVWAP T1(low)": Z.vs_vwap < t1, "S2 vsVWAP T2": (Z.vs_vwap >= t1) & (Z.vs_vwap < t2), "S2 vsVWAP T3(high)": Z.vs_vwap >= t2,
        "S3 vol_trend>1": Z.vol_trend > 1, "S3 vol_trend>=2": Z.vol_trend >= 2, "S3 vol_trend>=3": Z.vol_trend >= 3,
        "S3 v1h24>1": Z.v1h24 > 1, "S3 v1h24>=2": Z.v1h24 >= 2, "S3 v1h24>=3": Z.v1h24 >= 3,
    }
    a, b, c = Z.dd30 >= 3, Z.above_streak == 12, Z.vol_trend >= 2
    F.update({"S1p&S2p": a & b, "S1p&S3p": a & c, "S2p&S3p": b & c, "S1p&S2p&S3p": a & b & c})
    return {k: v.fillna(False).values.astype(bool) for k, v in F.items()}


def buckets(Z, q4):
    B = {}
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 1e9)):
        B[f"dd30 {lo}-{hi if hi < 1e9 else '∞'}"] = (Z.dd30 >= lo) & (Z.dd30 < hi)
    for lo, hi in ((12, 13), (13, 15), (15, 25), (25, 1e9)):
        B[f"streak {lo}-{hi - 1 if hi < 1e9 else '∞'}"] = (Z.above_streak >= lo) & (Z.above_streak < hi)
    for lo, hi in ((0, 0.7), (0.7, 1), (1, 2), (2, 3), (3, 1e9)):
        B[f"vol_trend {lo}-{hi if hi < 1e9 else '∞'}"] = (Z.vol_trend >= lo) & (Z.vol_trend < hi)
    e = [-np.inf] + list(q4) + [np.inf]
    for j in range(4):
        B[f"v1h24 Q{j + 1} [{e[j]:.2f},{e[j + 1]:.2f})"] = (Z.v1h24 >= e[j]) & (Z.v1h24 < e[j + 1])
    return {k: v.fillna(False).values.astype(bool) for k, v in B.items()}


def null(Z, F, B=2000):
    """day-block shuffle: each day's outcome block receives the JOINT flag block of a random day (resampled to size)."""
    names = list(F)
    M = np.column_stack([F[k] for k in names]).astype(float)
    y = Z.y.values; days = Z.day.values
    ud = np.unique(days); idx = {d: np.flatnonzero(days == d) for d in ud}
    def zs(Mx):
        out = []
        for j in range(Mx.shape[1]):
            f = Mx[:, j] > 0.5
            if f.sum() < 3 or (~f).sum() < 3:
                out.append(0.0); continue
            a, b = y[f], y[~f]
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            out.append((a.mean() - b.mean()) / se if se > 0 else 0.0)
        return np.array(out)
    obs = zs(M)
    Nz = np.zeros((B, len(names)))
    for b in range(B):
        src = RNG.permutation(ud)
        Mx = np.empty_like(M)
        for d, s in zip(ud, src):
            ti, si = idx[d], idx[s]
            Mx[ti] = M[RNG.choice(si, len(ti), replace=True)]
        Nz[b] = zs(Mx)
    p_test = [(np.abs(Nz[:, j]) >= abs(obs[j])).mean() for j in range(len(names))]
    fw = np.abs(Nz).max(1)
    p_fw = [(fw >= abs(obs[j])).mean() for j in range(len(names))]
    return pd.DataFrame(dict(test=names, z=obs, p_null=p_test, p_family=p_fw))


def skl():
    d = pd.read_csv(os.path.join(OUT, "SKLUSDT_5m.csv")).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    bo = int(pd.Timestamp("2026-10-08 21:35", tz="UTC").timestamp() * 1000)
    f = bar_feats(d, [bo]).iloc[0].to_dict()
    f.update(above_streak=12, vs_vwap=3.832, stamp_vol_trend=3.882)
    m = pd.read_csv(os.path.join(OUT, "SKLUSDT_1m.csv"))
    m = m[(m.open_time >= bo + B5) & (m.open_time <= int(pd.Timestamp("2026-10-08 21:49", tz="UTC").timestamp() * 1000))]
    f["worst_1m_bar_drop"] = float(((1 - m.l / m.h) * 100).max())
    return f


def fmt_table(D, cols):
    def f(x):
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else f"{x:+.3f}" if abs(x) < 100 else f"{x:.0f}"
        return str(x)
    h = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return h + "".join("| " + " | ".join(f(r[c]) for c in cols) + " |\n" for _, r in D.iterrows())


def main():
    K = build()
    assert K.feat_ok.all(), K[~K.feat_ok][["pair", "bar_open"]]
    par = dict(streak_eq=(K.above_streak == K.yw_streak).mean(), vt_corr=np.corrcoef(K.vol_trend, K.yw_vol_trend)[0, 1],
               vt_maxdiff=float(np.nanmax(np.abs(K.vol_trend - K.yw_vol_trend))), vs_vwap_na=int(K.vs_vwap.isna().sum()))
    K["y"] = K.fix3
    Fl = flush_stats(K); K = K.merge(Fl, on="key", how="left")
    md = [f"parity: streak==yw {par['streak_eq']:.3f} · vol_trend recompute vs yw corr {par['vt_corr']:.4f} maxdiff {par['vt_maxdiff']:.2e} · vs_vwap NA {par['vs_vwap_na']}\n"]
    S = skl(); md.append("SKL: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in S.items()) + "\n")
    tests_all = []
    for coh, Z in (("MAIN548", K[~K.bear.astype(bool)].reset_index(drop=True)), ("ROBUST724", K.reset_index(drop=True))):
        W, L = Z[Z.y > 0].y, Z[Z.y <= 0].y
        be = abs(L.mean()) / (W.mean() + abs(L.mean())) * 100
        lo, hi, p0 = boot(Z.y, Z.day)
        md.append(f"\n## {coh}: N {len(Z)} · WR {(Z.y > 0).mean() * 100:.1f} % · avg {Z.y.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] · Σ {Z.y.sum():+.1f} · "
                  f"days {Z.day.nunique()} · breakeven WR {be:.1f} % (avg win {W.mean():+.2f} / avg loss {L.mean():+.2f})\n")
        t1, t2 = (np.nanpercentile(K[~K.bear.astype(bool)].vs_vwap, [100 / 3, 200 / 3]))
        q4 = np.nanpercentile(K[~K.bear.astype(bool)].v1h24, [25, 50, 75])
        md.append(f"vs_vwap tercile cuts (MAIN) {t1:.2f} / {t2:.2f} · v1h24 quartile cuts {q4.round(2).tolist()}\n")
        F = flags(Z, t1, t2)
        T = pd.DataFrame([stat(Z, F[k], k, be) for k in F])
        Nl = null(Z, F)
        T = T.merge(Nl, on="test"); T.insert(0, "cohort", coh)
        tests_all.append(T)
        md.append(fmt_table(T, ["test", "N", "WR", "avg", "ci_lo", "ci_hi", "p_mean_lt0", "days", "rest_avg", "delta", "z", "p_null", "p_family",
                                "H1_avg", "H2_avg", "lomo_delta_min", "lomo_delta_max", "top_pair_loss_share", "top_day_loss_share", "bar_detail"]))
        Bk = buckets(Z, q4)
        TB = pd.DataFrame([stat(Z, Bk[k], k, be) for k in Bk]); TB.insert(0, "cohort", coh + "_bucket")
        tests_all.append(TB)
        md.append("\nbuckets\n" + fmt_table(TB, ["test", "N", "WR", "avg", "ci_lo", "ci_hi", "days", "H1_avg", "H2_avg"]))
        # where SKL sits (percentile within cohort)
        md.append("SKL percentile in cohort: " + ", ".join(
            f"{k} {np.mean(Z[k] <= S[k]) * 100:.0f}th" for k in ("dd30", "dd60", "burst60", "vol_trend", "v1h24", "vs_vwap")) + "\n")
        # stops
        Sx = Z[(Z.fix3_how == "SL") & Z.fl_ok.fillna(False).astype(bool)]
        fl = Sx[Sx.flush.astype(bool)]; gr = Sx[~Sx.flush.astype(bool)]
        md.append(f"\nstops {coh}: {len(Z[Z.fix3_how == 'SL'])} SL ({len(Sx)} with ticks) · flush (≥2 % in 60 s within 5 min of the stop) "
                  f"{len(fl)} = {len(fl) / len(Sx) * 100:.1f} % · gradual {len(gr)}\n")
        for nm, g in (("flush", fl), ("gradual", gr)):
            md.append(f"  {nm}: back to entry price within 60 min {g.recover60.mean() * 100:.1f} % (median {g.rec_min.median():.0f} min when it does) · "
                      f"median minutes entry→stop {g.min_to_stop.median():.0f} · stop ≤ 15 min after entry {(g.min_to_stop <= 15).mean() * 100:.0f} % · "
                      f"median largest 60-s drop {g.drop60_max.median():.2f} %\n")
        md.append(f"  flush share among S1p (dd30≥3) stops {Sx[Sx.dd30 >= 3].flush.mean() * 100:.0f} % (N {int((Sx.dd30 >= 3).sum())}) vs rest "
                  f"{Sx[Sx.dd30 < 3].flush.mean() * 100:.0f} %; among S3p (vol_trend≥2) {Sx[Sx.vol_trend >= 2].flush.mean() * 100:.0f} % "
                  f"(N {int((Sx.vol_trend >= 2).sum())}) vs rest {Sx[Sx.vol_trend < 2].flush.mean() * 100:.0f} %\n")
    pd.concat(tests_all).to_csv(os.path.join(REP, "LITE_ENTRY_SIGNS_STUDY_2026-10-08_tests.csv"), index=False)
    cols = ["pair", "bar_open", "entry_ts", "day", "month", "half", "bear", "hours", "vol_mult", "above_streak", "vs_vwap", "dd30", "dd60", "burst60",
            "vol_trend", "v1h24", "gvol", "atr5", "bar_red", "E", "fix3", "fix3_how", "fix3_xms", "pk_before_stop", "drop60_max", "flush",
            "recover60", "rec_min", "min_to_stop"]
    K[cols].rename(columns={"fix3": "pct_today_exit", "fix3_how": "exit_how", "fix3_xms": "exit_ms"}).to_csv(
        os.path.join(REP, "LITE_ENTRY_SIGNS_STUDY_2026-10-08.csv"), index=False)
    open(os.path.join(OUT, "tables.md"), "w").write("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
