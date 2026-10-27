#!/usr/bin/env python3
"""FADE REGIME W/L study 2026-10-26 — steps 4–5: top regime splits → within-regime entry W/L (1D + exhaustive 2D with a rotation null)
→ candidates on the CLAUDE.md expectancy bar (fade breakeven WR from the master kept fills) → before → after per batch (+ B25 provisional
apart) · DL 374 lines ①–⑥ · 41h · DL 165 · PREP_B25_PREREG fade rows · size implications.
Outputs reports/FADE_REGIME_WL_2026-10-26_{splits,within_1d,within_2d,within_null,candidates,per_batch,dl374,prereg_rows,size}.csv + _within.txt
    venv/bin/python -W ignore scripts/study_fade_regime_wl_within.py
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_fade_regime_wl_common as C    # noqa: E402

PFX = C.PFX
OUT = []
RNG = np.random.default_rng(7)
R_NULL = 300


def P(*a):
    s = " ".join(str(x) for x in a); OUT.append(s); print(s, flush=True)


A = C.load(with_blocked=True)
K = A[A.blk == "MASTER"].sort_values("t").reset_index(drop=True)
B25 = A[A.blk == "B25"].sort_values("t").reset_index(drop=True)
BLK = A[A.blk == "MASTER_BLK"].reset_index(drop=True)
Bt = C.load_bt()
BE = C.breakeven(K.pct.values)
market_own, mstamp, pair_own, pstamp = C.columns(A[A.blk != "MASTER_BLK"])
ENTRY = [c for c in pair_own + pstamp if not c.startswith("entry_pattern_")] + \
        [c for c in pstamp if c.startswith("entry_pattern_") and C.num(K, c).sum() >= 5] + ["d_fanr", "d_adx_rising", "d_rsi_rising"]
ENTRY = list(dict.fromkeys(c for c in ENTRY if c in A.columns))
P(f"breakeven WR {BE:.1f} % · entry vars {len(ENTRY)}")

beta_med = float(np.median(C.num(K, "br_beta_btc_24h").dropna()))
SPLITS = [("S1 HB ∧ BTC trend gap ≥ −0.13 (41h zone)", "hb_gap_zone", "eq1", 1.0),
          ("S2 alt beta to BTC (24 h) > master median", "br_beta_btc_24h", "gt", beta_med),
          ("S3 BTC 1d EMA50 above EMA200", "btc_e50_e200_1d", "gt", 0.0)]


def zmask(df, c, kind, thr):
    return C.apply_zone(C.num(df, c).values.astype(float), kind, thr)


def episodes(days):
    d = sorted(pd.to_datetime(pd.Series(list(set(days)))))
    if not d:
        return 0
    return 1 + sum((b - a).days > 3 for a, b in zip(d[:-1], d[1:]))


# ───────────────────────── splits table ─────────────────────────
rows = []
for nm, c, kind, thr in SPLITS:
    for side in ("zone", "rest"):
        r = {"split": nm, "side": side}
        for lab, df in (("master", K), ("B25 (prov.)", B25)):
            z, s = zmask(df, c, kind, thr); m = z if side == "zone" else (s & ~z)
            r[lab] = C.st(df.pct.values[m])
            if lab == "master":
                r["master days"] = df.day[m].nunique(); r["master episodes (gap > 3 d)"] = episodes(df.day[m])
                r["master $ (2×, stack)"] = df.usd.values[m].sum()
                r["stops"] = int(df.is_stop.values[m].sum())
                ds, dk = C.loss_share(df.pct.values[m], df.day.values[m]); ps, pk = C.loss_share(df.pct.values[m], df.pair.values[m])
                r["top day loss share"] = f"{ds:.0%} {dk}"; r["top pair loss share"] = f"{ps:.0%} {pk}"
                r["P(avg<0) day-boot"] = C.day_boot_p_neg(df.pct.values[m], df.day.values[m])
                for g in C.GROUPS:
                    r[g] = C.st(df.pct.values[m & (df.eg.values == g)])
        for sd in (1, 2, 3):
            bb = Bt[Bt.seed == sd]; z, s = zmask(bb, c, kind, thr); m = z if side == "zone" else (s & ~z)
            r[f"BT s{sd}"] = C.st(bb.pct.values[m])
        z, s = zmask(BLK, c, kind, thr); m = z if side == "zone" else (s & ~z)
        r["stack-blocked fades (as traded, completeness)"] = C.st(BLK.pct.values[m])
        rows.append(r)
SPL = pd.DataFrame(rows); SPL.to_csv(PFX + "_splits.csv", index=False)
P(SPL[["split", "side", "master", "master days", "master episodes (gap > 3 d)", "B25 (prov.)", "BT s1", "BT s2", "BT s3", "P(avg<0) day-boot"]].to_string())
# overlap of the splits
zz = {nm: zmask(K, c, k, t)[0] for nm, c, k, t in SPLITS}
P("split overlap (master, share of S2 zone also in S3):", round((zz[SPLITS[1][0]] & zz[SPLITS[2][0]]).sum() / max(zz[SPLITS[1][0]].sum(), 1), 2),
  "· S1 in S3:", round((zz[SPLITS[0][0]] & zz[SPLITS[2][0]]).sum() / max(zz[SPLITS[0][0]].sum(), 1), 2))


# ───────────────────────── within-regime 1D + 2D ─────────────────────────
def null_y(y, R):
    """rotation null on the time-sorted side: EVERY circular shift k = 2 … n − 2 when that is fewer than R, else R random shifts."""
    n = len(y); ks = np.arange(2, n - 1) if n - 3 <= R else RNG.choice(np.arange(2, n - 1), size=R)
    return np.stack([np.roll(y, k) for k in ks])

def screen_1d(S, cols, R=R_NULL):
    y = S.pct.values; n = len(y)
    masks = []
    for c in cols:
        x = C.num(S, c).values.astype(float)
        if np.isfinite(x).sum() < max(12, 0.6 * n):
            continue
        for lab, kind, thr, z in C.zones(x, c):
            s = np.isfinite(x)
            if z.sum() >= 5 and (s & ~z).sum() >= 5:
                masks.append((c, lab, kind, thr, z, s))
    if not masks:
        return pd.DataFrame(), None
    Zm = np.stack([m[4] for m in masks]).astype(float); Rm = np.stack([m[5] & ~m[4] for m in masks]).astype(float)
    Y = null_y(y, R)
    obs = Zm @ y / Zm.sum(1) - Rm @ y / Rm.sum(1)
    null = Y @ Zm.T / Zm.sum(1) - Y @ Rm.T / Rm.sum(1)
    p = ((np.abs(null) >= np.abs(obs)).sum(0) + 1) / (len(Y) + 1)
    Yp = np.stack([RNG.permutation(y) for _ in range(1000)])
    nullp = Yp @ Zm.T / Zm.sum(1) - Yp @ Rm.T / Rm.sum(1)
    pp = ((np.abs(nullp) >= np.abs(obs)).sum(0) + 1) / 1001
    # same-statistic null for the expectancy bar (block = zone): N ≥ 8 ∧ WR < BE ∧ avg < 0 (cheap pre-screen) count
    def cheap(yv):
        W = (yv[None, :] > 0) @ Zm.T / Zm.sum(1) * 100
        M = yv[None, :] @ Zm.T / Zm.sum(1)
        return ((W < BE) & (M < 0) & (Zm.sum(1) >= 8)).sum(1)
    real_bar = int(cheap(y)[0]); null_bar = np.array([cheap(Y[i])[0] for i in range(len(Y))])
    null_barp = np.array([cheap(Yp[i])[0] for i in range(len(Yp))])
    rows = []
    for i, (c, lab, kind, thr, z, s) in enumerate(masks):
        yz = y[z]
        b = C.bar(yz, S.day.values[z], S.pair.values[z], BE)
        rows.append({"variable": c, "zone": lab, "kind": kind, "thr": thr, "zone_stat": C.st(yz), "rest_stat": C.st(y[s & ~z]),
                     "delta": obs[i], "p_shift": p[i], "p_perm": pp[i], "zone_days": S.day[z].nunique(), "bar_pass": b["passes"], "bar_fails": b["fails"],
                     "P_neg": b["P_neg"], "top_pair_share": b["top_pair_share"], "top2_pair_share": b["top2_pair_share"]})
    D = pd.DataFrame(rows)
    nd = dict(tests=len(D), p05=int((D.p_shift < 0.05).sum()), p05_chance=0.05 * len(D), cheapbar_real=real_bar,
              cheapbar_null_median=float(np.median(null_bar)), cheapbar_P=float((null_bar >= real_bar).mean()),
              cheapbar_perm_median=float(np.median(null_barp)), cheapbar_perm_P=float((null_barp >= real_bar).mean()),
              p05_perm=int((D.p_perm < 0.05).sum()), rotations=len(Y), fw_min_p=float(D.p_shift.min()))
    return D, nd


def screen_2d(S, cols, R=R_NULL, nmin=6):
    y = S.pct.values; n = len(y)
    H = {}
    for c in cols:
        x = C.num(S, c).values.astype(float)
        if np.isfinite(x).sum() < max(12, 0.7 * n) or np.unique(x[np.isfinite(x)]).size < 2:
            continue
        v = x[np.isfinite(x)]
        thr = 0.5 if set(np.unique(v)) <= {0.0, 1.0} else float(np.median(v))
        H[c] = (np.isfinite(x) & (x > thr), np.isfinite(x) & (x <= thr), thr)
    names, Q = [], []
    for a, b in itertools.combinations(sorted(H), 2):
        for qa, qb in ((0, 0), (0, 1), (1, 0), (1, 1)):
            m = H[a][qa] & H[b][qb]
            if m.sum() >= nmin and (n - m.sum()) >= nmin:
                names.append((a, "hi" if qa == 0 else "lo", b, "hi" if qb == 0 else "lo", H[a][2], H[b][2])); Q.append(m)
    if not Q:
        return pd.DataFrame(), None
    Q = np.stack(Q).astype(float); nq = Q.sum(1); nr = n - nq
    def tstats(yv):
        s1 = yv @ Q.T; s2 = (yv ** 2) @ Q.T
        T1, T2 = yv.sum(-1, keepdims=True), (yv ** 2).sum(-1, keepdims=True)
        m1, m0 = s1 / nq, (T1 - s1) / nr
        v1 = np.maximum(s2 / nq - m1 ** 2, 1e-9); v0 = np.maximum((T2 - s2) / nr - m0 ** 2, 1e-9)
        return (m1 - m0) / np.sqrt(v1 / nq + v0 / nr), m1
    t, m1 = tstats(y[None, :]); t, m1 = t[0], m1[0]
    wr = ((y > 0).astype(float) @ Q.T) / nq * 100
    Y = null_y(y, R)
    tn, mn = tstats(Y)
    wn = ((Y > 0).astype(float) @ Q.T) / nq * 100
    Yp = np.stack([RNG.permutation(y) for _ in range(300)])
    tp, mp = tstats(Yp); wp = ((Yp > 0).astype(float) @ Q.T) / nq * 100
    cnt = lambda tt: (np.abs(tt) >= 3.0).sum(-1)
    cheap = lambda W, M: ((W < BE) & (M < 0) & (nq >= 8)).sum(-1)
    nd = dict(quadrants=len(Q), vars=len(H), real_abs_t3=int(cnt(t)), null_abs_t3_median=float(np.median(cnt(tn))),
              P_count=float((cnt(tn) >= cnt(t)).mean()), real_max_abs_t=float(np.abs(t).max()),
              null_max_abs_t_p95=float(np.quantile(np.abs(tn).max(1), 0.95)), P_max=float((np.abs(tn).max(1) >= np.abs(t).max()).mean()),
              cheapbar_real=int(cheap(wr, m1)), cheapbar_null_median=float(np.median(cheap(wn, mn))),
              cheapbar_P=float((cheap(wn, mn) >= cheap(wr, m1)).mean()), rotations=len(Y),
              perm_abs_t3_median=float(np.median(cnt(tp))), perm_P_count=float((cnt(tp) >= cnt(t)).mean()),
              perm_max_abs_t_p95=float(np.quantile(np.abs(tp).max(1), 0.95)), perm_P_max=float((np.abs(tp).max(1) >= np.abs(t).max()).mean()),
              cheapbar_perm_median=float(np.median(cheap(wp, mp))), cheapbar_perm_P=float((cheap(wp, mp) >= cheap(wr, m1)).mean()))
    rows = []
    order = np.argsort(-np.abs(t))[:40]
    for i in order:
        a, qa, b, qb, ta, tb = names[i]; m = Q[i].astype(bool)
        bb = C.bar(y[m], S.day.values[m], S.pair.values[m], BE)
        rows.append({"var_a": a, "side_a": qa, "thr_a": ta, "var_b": b, "side_b": qb, "thr_b": tb, "quadrant": C.st(y[m]),
                     "rest": C.st(y[~m]), "t": t[i], "days": S.day[m].nunique(), "bar_pass": bb["passes"], "bar_fails": bb["fails"],
                     "P_neg": bb["P_neg"], "top_pair_share": bb["top_pair_share"]})
    return pd.DataFrame(rows), nd


W1, W2, WN = [], [], []
for nm, c, kind, thr in SPLITS:
    for side in ("zone", "rest"):
        for lab, S in (("master", K), ("master+B25", pd.concat([K, B25]).sort_values("t").reset_index(drop=True))):
            z, s = zmask(S, c, kind, thr); m = z if side == "zone" else (s & ~z)
            Ss = S[m].reset_index(drop=True)
            if len(Ss) < 14:
                WN.append({"split": nm, "side": side, "pool": lab, "N": len(Ss), "note": "N < 14 → 1D/2D not run"}); continue
            d1, n1 = screen_1d(Ss, ENTRY)
            if len(d1):
                d1.insert(0, "pool", lab); d1.insert(0, "side", side); d1.insert(0, "split", nm); W1.append(d1)
            d2, n2 = (screen_2d(Ss, ENTRY) if len(Ss) >= 20 else (pd.DataFrame(), None))
            if len(d2):
                d2.insert(0, "pool", lab); d2.insert(0, "side", side); d2.insert(0, "split", nm); W2.append(d2)
            WN.append({"split": nm, "side": side, "pool": lab, "N": len(Ss), "days": Ss.day.nunique(),
                       **{f"1D_{k}": v for k, v in (n1 or {}).items()}, **{f"2D_{k}": v for k, v in (n2 or {}).items()}})
W1 = pd.concat(W1, ignore_index=True); W2 = pd.concat(W2, ignore_index=True) if W2 else pd.DataFrame(); WN = pd.DataFrame(WN)
W1.to_csv(PFX + "_within_1d.csv", index=False); W2.to_csv(PFX + "_within_2d.csv", index=False); WN.to_csv(PFX + "_within_null.csv", index=False)
P(WN.to_string())


# ───────────────────────── candidates · per-batch before → after ─────────────────────────
def per_batch(name, zone_fn):
    rows = []
    for era, g in list(K.groupby("era", sort=False)) + [("B25 (prov.)", B25)]:
        z = zone_fn(g)
        rows.append({"candidate": name, "batch": era, "before": C.st(g.pct), "before $": g.usd.sum(), "blocked": C.st(g.pct[z]),
                     "blocked $": g.usd[z].sum(), "after": C.st(g.pct[~z]), "after $": g.usd[~z].sum(),
                     "Δ$ (block)": -g.usd[z].sum(), "Δ avg%/fill": (g.pct[~z].mean() if (~z).any() else np.nan) - g.pct.mean()})
    return rows


def zone_fn_factory(c, kind, thr, also=None):
    def f(df):
        z, s = zmask(df, c, kind, thr)
        if also is not None:
            z = z & also(df)
        return pd.Series(z, index=df.index)
    return f


hb = lambda df: (df.entry_btc_regime == "HEALTHY_BEAR").values
CANDS = [
    ("C1 block HB ∧ trend gap ≥ −0.13 (41h zone)", zone_fn_factory("hb_gap_zone", "eq1", 1.0)),
    ("C2 block HEALTHY_BEAR (all)", zone_fn_factory("c_regime_HEALTHY_BEAR", "eq1", 1.0)),
    (f"C3 block alt beta > {beta_med:.3f}", zone_fn_factory("br_beta_btc_24h", "gt", beta_med)),
    ("C4 block BTC 1d EMA50 > EMA200", zone_fn_factory("btc_e50_e200_1d", "gt", 0.0)),
    ("DL374 ① block ADX Δ < 2.359", zone_fn_factory("entry_adx_delta", "lt", 2.359)),
    ("DL374 ② block pair ADX ≤ 19.2", zone_fn_factory("entry_adx", "le", 19.2)),
    ("DL374 ⑤ block HB ∧ ADX Δ ≤ 2.4341", zone_fn_factory("entry_adx_delta", "le", 2.4341, also=hb)),
    ("DL374 ⑥ block HB ∧ BTC ADX prev > 24.5874", zone_fn_factory("entry_btc_adx_prev", "gt", 24.5874, also=hb)),
]
# the within-regime survivors (added below after inspection is NOT allowed to move thresholds: only the frozen medians of the 2D screen)
extra = []
if len(W2):
    top = W2[(W2.side == "zone") & (W2.pool == "master") & (W2.t < 0)].sort_values("t").head(3)
    for _, r in top.iterrows():
        sp = [s for s in SPLITS if s[0] == r.split][0]
        def mk(sp=sp, r=r):
            def f(df):
                z, _ = zmask(df, sp[1], sp[2], sp[3])
                a = C.num(df, r.var_a).values.astype(float); b = C.num(df, r.var_b).values.astype(float)
                za = (a > r.thr_a) if r.side_a == "hi" else (a <= r.thr_a); zb = (b > r.thr_b) if r.side_b == "hi" else (b <= r.thr_b)
                return pd.Series(z & za & zb & np.isfinite(a) & np.isfinite(b), index=df.index)
            return f
        extra.append((f"W2D {r.split.split()[0]} zone ∧ {r.var_a} {r.side_a} {r.thr_a:.4g} ∧ {r.var_b} {r.side_b} {r.thr_b:.4g}", mk()))
if len(W1):
    top1 = W1[(W1.side == "zone") & (W1.pool == "master") & (W1.delta < 0)].sort_values("p_shift").head(3)
    for _, r in top1.iterrows():
        sp = [s for s in SPLITS if s[0] == r.split][0]
        def mk1(sp=sp, r=r):
            def f(df):
                z, _ = zmask(df, sp[1], sp[2], sp[3]); zz, _ = zmask(df, r.variable, r.kind, r.thr)
                return pd.Series(z & zz, index=df.index)
            return f
        extra.append((f"W1D {r.split.split()[0]} zone ∧ {r.variable} {r.zone}", mk1()))

crow, prow = [], []
ALL = pd.concat([K, B25]).reset_index(drop=True)
for name, fn in CANDS + extra:
    zk = fn(K).values; za = fn(ALL).values; zb = fn(B25).values
    bk = C.bar(K.pct.values[zk], K.day.values[zk], K.pair.values[zk], BE)
    ba = C.bar(ALL.pct.values[za], ALL.day.values[za], ALL.pair.values[za], BE)
    btd = []
    for sd in (1, 2, 3):
        bb = Bt[Bt.seed == sd].reset_index(drop=True)
        try:
            zz = fn(bb).values
            btd.append(f"{C.st(bb.pct.values[zz])} vs {C.st(bb.pct.values[~zz])}")
        except Exception as e:
            btd.append("n/a")
    dk = -K.usd.values[zk].sum(); da = -ALL.usd.values[za].sum()
    crow.append({"candidate": name, "master blocked": C.st(K.pct.values[zk]), "master kept": C.st(K.pct.values[~zk]),
                 "master Δ$ in-sample (2×)": dk, "Δ$ haircut 30–50 %": f"{dk * 0.5:+.0f} … {dk * 0.7:+.0f}",
                 "B25 blocked (prov.)": C.st(B25.pct.values[zb]), "B25 Δ$": -B25.usd.values[zb].sum(),
                 "ALL blocked": C.st(ALL.pct.values[za]), "ALL Δ$": da,
                 "bar master": "PASS" if bk["passes"] else f"fail ({bk['fails']})", "P_neg master": bk["P_neg"],
                 "days master": bk["days"], "top day share": bk["top_day_share"], "top pair share": bk["top_pair_share"],
                 "top-2 pairs share": bk["top2_pair_share"],
                 "bar ALL": "PASS" if ba["passes"] else f"fail ({ba['fails']})", "P_neg ALL": ba["P_neg"],
                 "BT s1 blocked vs kept": btd[0], "BT s2": btd[1], "BT s3": btd[2]})
    prow += per_batch(name, fn)
CA = pd.DataFrame(crow); CA.to_csv(PFX + "_candidates.csv", index=False)
PB = pd.DataFrame(prow); PB.to_csv(PFX + "_per_batch.csv", index=False)
P(CA[["candidate", "master blocked", "master kept", "master Δ$ in-sample (2×)", "B25 blocked (prov.)", "bar master", "P_neg master",
      "days master", "top pair share", "bar ALL", "BT s1 blocked vs kept"]].to_string())

# ───────────────────────── DL 374 / 41h / DL 165 / size ─────────────────────────
reg7 = B25[B25.t < pd.Timestamp("2026-10-26 12:39:59")]
P(f"DL 374 registration B25 fills: {len(reg7)} (expected 7) · out-of-registration B25: {list(B25[B25.t >= pd.Timestamp('2026-10-26 12:39:59')].pair)}")
P("DL 374 forward cohort floor = commit 20c482b9 (2026-10-26 13:19:33 UTC) + 10 min → B25 fills at/after 13:29:33:",
  int((B25.t >= pd.Timestamp("2026-10-26 13:29:33")).sum()))
# 41h zone: registration 2026-09-18 → forward tally
reg41h = pd.Timestamp("2026-09-18 23:59:59")
fz = ALL[(ALL.t > reg41h) & (ALL.hb_gap_zone == 1)]; fr = ALL[(ALL.t > reg41h) & (ALL.hb_gap_zone == 0)]
P(f"41h zone FORWARD since registration (opened > 09-18): zone {C.st(fz.pct)} · ${fz.usd.sum():+.0f} · {fz.day.nunique()} days vs rest {C.st(fr.pct)}")
z41 = ALL[ALL.hb_gap_zone == 1]
P(f"41h zone ALL (master + B25): {C.st(z41.pct)} · Σ$ {z41.usd.sum():+.0f} · {z41.day.nunique()} days · SHIP bar N≥20 ∧ WR≤55 ∧ Σ<0 → "
  f"{'MET' if len(z41) >= 20 and (z41.pct > 0).mean() <= 0.55 and z41.usd.sum() < 0 else 'NOT MET'}")
# DL 165: fades on pairs with 24 h vol ≥ $10M opened after 2026-10-01 deploy
v = C.num(ALL, "entry_pair_volume_24h_usd")
d165 = ALL[(ALL.t >= pd.Timestamp("2026-10-01 18:00")) & (v >= 10e6)]
P("DL 165 ($10–20M band since 10-01):", d165[["opened_at", "pair", "entry_pair_volume_24h_usd", "pct", "usd", "liquidity_capped", "notional_value"]].to_string())

# size implications (fills at 2× vs 1× by regime split; 4 %-of-account test on $2,900)
srow = []
for nm, c, kind, thr in SPLITS + [("HEALTHY_BEAR (DL 374 ④)", "c_regime_HEALTHY_BEAR", "eq1", 1.0)]:
    for lab, df in (("master", K), ("B25 (prov.)", B25)):
        z, s = zmask(df, c, kind, thr)
        for side, m in (("zone", z), ("rest", s & ~z)):
            u = df.usd.values[m]
            srow.append({"split": nm, "pool": lab, "side": side, "fills": C.st(df.pct.values[m]), "$ at 2× (traded/stack)": u.sum(),
                         "$ at 1×": (u / C.num(df, "cell_multiplier").fillna(2.0).values[m]).sum(),
                         "worst fill $ (2×)": u.min() if len(u) else np.nan,
                         "fills losing > 4 % of $2,900 at 2×": int((u < -116).sum()), "at 1×": int((u / 2 < -116).sum())})
SZ = pd.DataFrame(srow); SZ.to_csv(PFX + "_size.csv", index=False)
P(SZ.to_string())

# PREP_B25_PREREG fade rows on the full data
pr = pd.read_csv(os.path.join(C.R, "reports/PREP_B25_PREREG.csv"))
pr = pr[pr.sleeve == "SPIKE_FADE"]
prow2 = []
for _, r in pr.iterrows():
    col, d, thr = str(r.column_name), str(r.direction), r.threshold
    if d == "NONE" or col.startswith("(") or d not in ("W_HIGHER", "W_LOWER", "W_FALSE"):
        prow2.append({"variable": r.variable, "column": col, "direction": d, "verdict": "no prediction / not stamped"}); continue
    def xval(df):
        if ">" in col and "+" not in col:
            a, b = col.split(">"); return (C.num(df, a) > C.num(df, b)).astype(float).where(C.num(df, b).notna()).values
        if col == "entry_btc_regime":
            return (df.entry_btc_regime == thr).astype(float).where(df.entry_btc_regime.notna()).values
        if "+" in col:
            return (((df.entry_btc_regime == "HEALTHY_BEAR") & (C.num(df, "entry_adx_delta") <= 2.4341)).astype(float)
                    .where(df.entry_btc_regime.notna()).values)
        return C.num(df, col).values.astype(float) if col in df.columns else np.full(len(df), np.nan)
    out = {"variable": r.variable, "column": col, "direction": d, "threshold": thr, "contains_B25": "CONTAINS_B25" in str(r.evidence_source)}
    for lab, df in (("master", K), ("B25", B25), ("B25 ex-registration", B25[B25.t >= pd.Timestamp("2026-10-26 12:39:59")])):
        x = xval(df); s = np.isfinite(x)
        if d == "W_FALSE":
            hi = s & (x == 1)
        else:
            t_ = float(thr) if pd.notna(thr) and str(thr) not in ("", "nan") else 0.0
            hi = s & (x > t_)
        y = df.pct.values
        out[f"{lab} high"] = C.st(y[hi]); out[f"{lab} low"] = C.st(y[s & ~hi])
        out[f"{lab} Δ high−low"] = (y[hi].mean() - y[s & ~hi].mean()) if hi.sum() and (s & ~hi).sum() else np.nan
        if lab == "master":
            out["master days hi/lo"] = f"{df.day[hi].nunique()}/{df.day[s & ~hi].nunique()}"
    want = 1 if d == "W_HIGHER" else -1
    dm = out["master Δ high−low"]
    b = B25; xb = xval(b); sb = np.isfinite(xb)
    hb_ = sb & ((xb == 1) if d == "W_FALSE" else (xb > (float(thr) if pd.notna(thr) and str(thr) not in ("", "nan") else 0.0)))
    nb_ok = hb_.sum() >= 5 and (sb & ~hb_).sum() >= 5
    out["master direction"] = "n/a" if not np.isfinite(dm) else ("as predicted" if np.sign(dm) == want else "OPPOSITE")
    out["B25 refute-only read (N ≥ 5 per side)"] = ("not readable (N < 5 per side)" if not nb_ok else
                                                   ("REFUTED" if np.sign(out["B25 Δ high−low"]) != want else "not refuted"))
    prow2.append(out)
PR = pd.DataFrame(prow2); PR.to_csv(PFX + "_prereg_rows.csv", index=False)
P(PR[["variable", "direction", "threshold", "master high", "master low", "master direction", "B25 refute-only read (N ≥ 5 per side)"]].to_string())
open(PFX + "_within.txt", "w").write("\n".join(OUT) + "\n")
