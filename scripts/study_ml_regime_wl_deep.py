#!/usr/bin/env python3
"""ML REGIME W/L study 2026-10-26 — deep panel on the shortlisted regime splits, never-green outcome screen, era-level table,
step 4 (within-regime entry screen 1D + exhaustive 2D with a circular-shift null + regime × entry interaction) and step 5 candidates
(expectancy bar, before → after per era group, B25 apart). Run after study_ml_regime_wl_analyse.py (reads its _D.pkl).
    venv/bin/python -W ignore scripts/study_ml_regime_wl_deep.py
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_ml_wl_b25prep_data as L                       # noqa: E402
from study_ml_regime_wl_analyse import (PFX, GROUPS, GOOD, BAD, st, zones, shifts, day_boot, strat_delta,  # noqa: E402
                                        regime_columns)
from study_yr6wl_momentum_common import load_replay        # noqa: E402

RNG = np.random.default_rng(7)
BE = 61.5
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a); OUT.append(s); print(s, flush=True)


D = pd.read_pickle(PFX + "_D.pkl")
K = D[D.y.notna() & D.era_group.isin(GROUPS)].reset_index(drop=True)          # 158 KEEP master
B25 = D[D.era_group == "B25 (prov.)"].reset_index(drop=True)
TR = D[D.y_trad.notna() & D.era_group.isin(GROUPS)].reset_index(drop=True)      # 210 TRAD


def replay():
    R = load_replay(["MOM-long"])
    G = pd.read_csv(PFX + "_replay_regime.csv"); G["ts"] = pd.to_datetime(G.ts)
    return R.merge(G, on="ts", how="left")


RP = replay()


def zmask(df, c, op, thr):
    x = pd.to_numeric(df[c], errors="coerce")
    return (x > thr) if op == ">" else (x <= thr) if op == "<=" else (x == thr)


# ───────────────────── shortlist panel (regime) ─────────────────────
SHORT = [("br_corr24h_top20", ">", 0.3164, "PC primary · T5"), ("br_beta_btc_24h", ">", 1.372, "T5 (alias of corr, ρ 0.82)"),
         ("br_disp_24h", "<=", 8.737, "T5 dispersion"), ("btc_rv_7d", "<=", 0.1122, "T6 low 7d vol"),
         ("btc_e20_e50_1d", ">", 2.494, "T1 BTC 1d uptrend"), ("btc_e50_e200_1d", ">", -7.497, "T1/T2 BTC 1d macro"),
         ("btc_r30d", ">", 0, "T2 BTC 30d up"), ("btc_offhi_30d", ">", -3.0, "PB primary · T2 near 30d high"),
         ("btc_eff_24h", "<=", 0.05162, "PA primary · T1 chop"), ("alt_minus_btc_7d", "<=", 0, "PD primary · T3"),
         ("br_funding_med", ">", 5e-05, "PE primary · T8 (partial)"), ("entry_global_volume_ratio", ">", 0.8217, "T7 volume"),
         ("cat__entry_btc_regime==STRONG_BULL", "==", 1.0, "seen (PREP H1) winner"), ("btc_r24h", "<=", 0, "BTC24H_DOWN analogue"),
         ("br_above_ema50_1h", "<=", 33.79, "T4 narrow breadth"), ("weekend", "==", 1.0, "T7 weekend"),
         ("btc_vr15_7d", "<=", 1.0, "T9 mean-reverting BTC")]


def panel():
    rows = []
    for c, op, thr, why in SHORT:
        z = zmask(K, c, op, thr).values; s = pd.to_numeric(K[c], errors="coerce").notna().values
        y = K.y.values
        zb = day_boot(y[z], K.day.values[z]); rb = day_boot(y[s & ~z], K.day.values[s & ~z])
        dz = K[z].groupby("day").y.mean()
        loss = -K.y[z & (K.y < 0)].sum()
        wd = (K[z & (K.y < 0)].groupby("day").y.sum().min() * -1 / loss * 100) if loss > 0 else np.nan
        wp = (K[z & (K.y < 0)].groupby("pair").y.sum().min() * -1 / loss * 100) if loss > 0 else np.nan
        sd, per = strat_delta(y, z, K.era_group.values)
        zt = zmask(TR, c, op, thr).values; st_ = pd.to_numeric(TR[c], errors="coerce").notna().values
        zb25 = zmask(B25, c, op, thr).values
        r = {"variable": c, "zone": f"{op}{thr:g}", "why": why, "KEEP scored": int(s.sum()),
             "zone": st(y[z]), "rest": st(y[s & ~z]), "Δ": y[z].mean() - y[s & ~z].mean(),
             "zone P(avg<0) day-boot": (zb < 0).mean(), "Δ P(<0) day-boot": np.nan,
             "zone days": int(K.day[z].nunique()), "zone scan windows": int(K.win_id[z].nunique()),
             "zone day-mean": dz.mean(), "zone neg-day %": (dz < 0).mean() * 100,
             "worst day % of zone loss": wd, "worst pair % of zone loss": wp,
             "era-strat Δ": sd, **{f"Δ {g}": per.get(g, np.nan) for g in GROUPS},
             "zone share good/bad %": f"{z[K.era_group.isin(GOOD)].mean() * 100:.0f}/{z[K.era_group.isin(BAD)].mean() * 100:.0f}",
             "ex-washed Δ": y[z & ~K.washed.values].mean() - y[s & ~z & ~K.washed.values].mean(),
             "TRAD zone/rest": f"{st(TR.y_trad[zt])} / {st(TR.y_trad[st_ & ~zt])}",
             "B25 zone/rest": f"{st(B25.y[zb25])} / {st(B25.y[~zb25 & pd.to_numeric(B25[c], errors='coerce').notna().values])}"}
        # Δ bootstrap (days resampled jointly)
        dd = pd.DataFrame({"d": K.day.values[s], "y": y[s], "z": z[s]})
        days = dd.d.unique(); g = {d: x for d, x in dd.groupby("d")}
        bs = []
        for _ in range(2000):
            pick = pd.concat([g[d] for d in RNG.choice(days, len(days))])
            if pick.z.sum() >= 3 and (~pick.z).sum() >= 3:
                bs.append(pick.y[pick.z].mean() - pick.y[~pick.z].mean())
        r["Δ P(<0) day-boot"] = (np.array(bs) < 0).mean()
        # replay per regime
        if c in RP:
            for rg in ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]:
                q = RP[RP.regime == rg]; zz = zmask(q, c, op, thr)
                r[f"replay {rg} Δ"] = q.y[zz].mean() - q.y[~zz & pd.to_numeric(q[c], errors='coerce').notna()].mean() if zz.sum() >= 10 else np.nan
                r[f"replay {rg} zone share %"] = zz.mean() * 100
        rows.append(r)
    return pd.DataFrame(rows)


def era_level():
    """n = 7 era groups (+ replay 3 eras): group avg vs group median regime value."""
    own, stamps, cats, d16 = regime_columns(D)
    g = K.groupby("era_group")
    ya = g.y.mean()
    rows = []
    for c in own + stamps + d16:
        x = pd.to_numeric(K[c], errors="coerce").groupby(K.era_group).median()
        ok = x.notna() & ya.notna()
        if ok.sum() < 6:
            continue
        rho = pd.Series(x[ok]).rank().corr(pd.Series(ya[ok]).rank())
        r = {"variable": c, "groups": int(ok.sum()), "spearman_eras": rho,
             "good median": pd.to_numeric(K[c][K.era_group.isin(GOOD)], errors="coerce").median(),
             "bad median": pd.to_numeric(K[c][K.era_group.isin(BAD)], errors="coerce").median()}
        if c in RP:
            xr = RP.groupby("regime")[c].median(); yr = RP.groupby("regime").y.mean()
            r["replay eras spearman (3)"] = xr.rank().corr(yr.rank())
        rows.append(r)
    E = pd.DataFrame(rows)
    # null: how many of the variables reach |rho| ≥ 0.9 by chance when the era ORDER is shuffled (7! perms sampled)
    return E


def ng_screen(R=1000):
    """secondary outcome: never-green loser (peak < +0.10 and loss) — which regime states raise it?"""
    own, stamps, cats, d16 = regime_columns(D)
    y = K.ng_loser_keep.astype(float).values
    Y = shifts(y, R)
    rows, M = [], []
    for c in own + stamps + d16:
        x = pd.to_numeric(K[c], errors="coerce").values
        if np.isfinite(x).sum() < 30:
            continue
        for lab, z in zones(x, c):
            s = np.isfinite(x)
            if z.sum() >= 10 and (s & ~z).sum() >= 10:
                M.append((c, lab, z, s))
    Z = np.stack([m[2] for m in M]).astype(float); Rr = np.stack([m[3] & ~m[2] for m in M]).astype(float)
    obs = Z @ y / Z.sum(1) - Rr @ y / Rr.sum(1)
    nul = Y @ Z.T / Z.sum(1) - Y @ Rr.T / Rr.sum(1)
    p = ((np.abs(nul) >= np.abs(obs)).sum(0) + 1) / (R + 1)
    pfw = ((np.abs(nul).max(1)[:, None] >= np.abs(obs)[None, :]).sum(0) + 1) / (R + 1)
    for i, (c, lab, z, s) in enumerate(M):
        sd, per = strat_delta(y, z, K.era_group.values)
        rows.append({"variable": c, "zone": lab, "N": int(z.sum()), "NG% zone": y[z].mean() * 100, "NG% rest": y[s & ~z].mean() * 100,
                     "Δpp": obs[i] * 100, "p_null": p[i], "p_fw": pfw[i], "era-strat Δpp": sd * 100,
                     "groups same sign": f"{sum(np.sign(v) == np.sign(obs[i]) for v in per.values())}/{len(per)}"})
    return pd.DataFrame(rows).sort_values("p_null")


# ───────────────────── step 4 within-regime entry screen ─────────────────────
def entry_columns():
    cols = [c for c in L.feature_columns(K) if not c.startswith("cat__entry_btc_regime==") or True]
    out = []
    for c in cols:
        x = pd.to_numeric(K[c], errors="coerce")
        if x.notna().sum() >= 30 and x.nunique() >= 2:
            out.append(c)
    return out


def screen_1d(S, cols, R=1000):
    y = S.y.values; Y = shifts(y, R)
    M = []
    for c in cols:
        x = pd.to_numeric(S[c], errors="coerce").values
        if np.isfinite(x).sum() < 20:
            continue
        for lab, z in zones(x, c):
            s = np.isfinite(x)
            if z.sum() >= 6 and (s & ~z).sum() >= 6:
                M.append((c, lab, z, s))
    Z = np.stack([m[2] for m in M]).astype(float); Rr = np.stack([m[3] & ~m[2] for m in M]).astype(float)
    obs = Z @ y / Z.sum(1) - Rr @ y / Rr.sum(1)
    nul = Y @ Z.T / Z.sum(1) - Y @ Rr.T / Rr.sum(1)
    p = ((np.abs(nul) >= np.abs(obs)).sum(0) + 1) / (R + 1)
    pfw = ((np.abs(nul).max(1)[:, None] >= np.abs(obs)[None, :]).sum(0) + 1) / (R + 1)
    rows = []
    for i, (c, lab, z, s) in enumerate(M):
        rows.append({"variable": c, "zone": lab, "N": int(z.sum()), "zone": st(y[z]), "rest": st(y[s & ~z]), "Δ": obs[i],
                     "p_null": p[i], "p_fw": pfw[i], "zone_days": int(S.day[z].nunique()), "scored": int(s.sum())})
    return pd.DataFrame(rows)


def screen_2d(S, cols, R=200, nrep=20, minz=8):
    y = S.y.values; n = len(y)
    full = [c for c in cols if pd.to_numeric(S[c], errors="coerce").notna().mean() >= 0.8 and pd.to_numeric(S[c], errors="coerce").nunique() >= 3]
    X = {c: pd.to_numeric(S[c], errors="coerce").values for c in full}
    hi = {c: X[c] > np.nanmedian(X[c]) for c in full}
    sc = {c: np.isfinite(X[c]) for c in full}
    M = []
    for a, b in itertools.combinations(full, 2):
        s = sc[a] & sc[b]
        for qa, qb in [(1, 1), (1, 0), (0, 1), (0, 0)]:
            z = s & (hi[a] == bool(qa)) & (hi[b] == bool(qb))
            if z.sum() >= minz and (s & ~z).sum() >= minz:
                M.append((a, b, qa, qb, z, s))
    Z = np.stack([m[4] for m in M]).astype(np.float32); Rr = np.stack([m[5] & ~m[4] for m in M]).astype(np.float32)
    zs, rs = Z.sum(1), Rr.sum(1)
    obs = Z @ y / zs - Rr @ y / rs
    Y = shifts(y, R).astype(np.float32)
    nul = (Y @ Z.T) / zs - (Y @ Rr.T) / rs
    p = ((np.abs(nul) >= np.abs(obs)).sum(0) + 1) / (R + 1)
    pfw = ((np.abs(nul).max(1)[:, None] >= np.abs(obs)[None, :]).sum(0) + 1) / (R + 1)
    # replicate the count of p<0.01 under shifted outcomes
    cnt = []
    for k in RNG.choice(np.arange(5, n - 5), nrep, replace=False):
        ys = np.roll(y, k).astype(np.float32)
        o2 = Z @ ys / zs - Rr @ ys / rs
        Y2 = shifts(ys, R).astype(np.float32)
        n2 = (Y2 @ Z.T) / zs - (Y2 @ Rr.T) / rs
        p2 = ((np.abs(n2) >= np.abs(o2)).sum(0) + 1) / (R + 1)
        cnt.append(int((p2 < 0.01).sum()))
    rows = [{"a": m[0], "b": m[1], "a_hi": m[2], "b_hi": m[3], "N": int(m[4].sum()), "zone": st(y[m[4]]),
             "rest": st(y[m[5] & ~m[4]]), "Δ": obs[i], "p_null": p[i], "p_fw": pfw[i], "zone_days": int(S.day[m[4]].nunique())}
            for i, m in enumerate(M)]
    return pd.DataFrame(rows), len(full), cnt


def interaction(Rmask, cols, R=1000):
    """entry zone Δ inside the bad regime minus inside the good regime (pooled cut); circular-shift null on the whole sample."""
    y = K.y.values; Y = shifts(y, R)
    M = []
    for c in cols:
        x = pd.to_numeric(K[c], errors="coerce").values
        for lab, z in zones(x, c):
            s = np.isfinite(x)
            cells = [(z & Rmask), (s & ~z & Rmask), (z & ~Rmask), (s & ~z & ~Rmask)]
            if min(cc.sum() for cc in cells) >= 5:
                M.append((c, lab, cells))
    C = [np.stack([mm[2][k] for mm in M]).astype(float) for k in range(4)]
    cnt = [cc.sum(1) for cc in C]
    def stat(Yv):
        m = [(Yv @ C[k].T) / cnt[k] for k in range(4)]
        return (m[0] - m[1]) - (m[2] - m[3])
    obs = stat(y[None, :])[0]
    nul = stat(Y)
    p = ((np.abs(nul) >= np.abs(obs)).sum(0) + 1) / (R + 1)
    pfw = ((np.abs(nul).max(1)[:, None] >= np.abs(obs)[None, :]).sum(0) + 1) / (R + 1)
    rows = []
    for i, (c, lab, cells) in enumerate(M):
        rows.append({"variable": c, "zone": lab, "bad-regime zone": st(y[cells[0]]), "bad-regime rest": st(y[cells[1]]),
                     "good-regime zone": st(y[cells[2]]), "good-regime rest": st(y[cells[3]]),
                     "Δ bad": y[cells[0]].mean() - y[cells[1]].mean(), "Δ good": y[cells[2]].mean() - y[cells[3]].mean(),
                     "interaction": obs[i], "p_null": p[i], "p_fw": pfw[i]})
    return pd.DataFrame(rows)


def bar(df, z, label):
    """expectancy bar on the blocked cohort z (KEEP), day = window."""
    y = df.y.values
    yz = y[z]
    if len(yz) == 0:
        return {}
    bs = day_boot(yz, df.day.values[z])
    loss = -yz[yz < 0].sum()
    wd = (-df[z & (df.y < 0)].groupby("day").y.sum().min() / loss * 100) if loss > 0 else np.nan
    wp = (-df[z & (df.y < 0)].groupby("pair").y.sum().min() / loss * 100) if loss > 0 else np.nan
    wins = yz[yz > 0]; los = yz[yz < 0]
    r = {"candidate": label, "N": len(yz), "WR %": (yz > 0).mean() * 100, "avg %": yz.mean(), "P(avg<0) day-boot": (bs < 0).mean(),
         "windows (days)": int(df.day[z].nunique()), "scan windows": int(df.win_id[z].nunique()),
         "worst day % of loss": wd, "worst pair % of loss": wp}
    r["① WR<61.5"] = r["WR %"] < BE
    r["② P≥0.95"] = r["P(avg<0) day-boot"] >= 0.95
    r["③ ≥8 windows & no day/pair ≥50%"] = (r["windows (days)"] >= 8) and (wd < 50) and (wp < 50)
    r["④ N≥15"] = len(yz) >= 15
    r["PASS"] = r["① WR<61.5"] and r["② P≥0.95"] and r["③ ≥8 windows & no day/pair ≥50%"] and r["④ N≥15"]
    return r


def before_after(df_all, zfun, label):
    rows = []
    for g in GROUPS + ["B25 (prov.)"]:
        d = df_all[(df_all.era_group == g) & df_all.y.notna()]
        z = zfun(d)
        rows.append({"candidate": label, "group": g, "before": st(d.y), "after (zone removed)": st(d.y[~z]), "zone": st(d.y[z])})
    d = df_all[df_all.y.notna() & df_all.era_group.isin(GROUPS)]; z = zfun(d)
    rows.append({"candidate": label, "group": "ALL KEEP", "before": st(d.y), "after (zone removed)": st(d.y[~z]), "zone": st(d.y[z]),
                 "Δavg": d.y[~z].mean() - d.y.mean(),
                 "Δavg haircut 30–50%": f"{(d.y[~z].mean() - d.y.mean()) * 0.5:+.3f}…{(d.y[~z].mean() - d.y.mean()) * 0.7:+.3f}"})
    d = df_all[df_all.y_trad.notna() & df_all.era_group.isin(GROUPS)]; z = zfun(d)
    rows.append({"candidate": label, "group": "TRAD (as traded)", "before": st(d.y_trad), "after (zone removed)": st(d.y_trad[~z]), "zone": st(d.y_trad[z])})
    return rows


if __name__ == "__main__":
    Pn = panel(); Pn.to_csv(PFX + "_regime_panel.csv", index=False)
    E = era_level(); E.to_csv(PFX + "_era_level.csv", index=False)
    NG = ng_screen(); NG.to_csv(PFX + "_ng_screen.csv", index=False)
    P(f"NG screen: {len(NG)} tests · p<0.05 {int((NG.p_null < 0.05).sum())} (chance {0.05 * len(NG):.0f}) · p_fw<0.05 {int((NG.p_fw < 0.05).sum())}")
    cols = entry_columns()
    P(f"entry columns screened: {len(cols)}")
    SPLITS = {
        "A_HICORR": (zmask(K, "br_corr24h_top20", ">", 0.3164).values, "top-20 24 h corr > 0.3164 (bad) vs ≤"),
        "C_LATEERA": (K.era_group.isin(BAD).values, "G5–G6 (bad, observational) vs G0–G4"),
        "D_LOWRV7": (zmask(K, "btc_rv_7d", "<=", 0.1122).values, "BTC 7 d realised vol ≤ 0.1122 (bad) vs >"),
    }
    allw1, allw2, allint, meta = [], [], [], []
    for nm, (m, desc) in SPLITS.items():
        for side, mm in [("bad", m), ("good", ~m)]:
            S = K[mm].reset_index(drop=True)
            o1 = screen_1d(S, cols); o1.insert(0, "side", side); o1.insert(0, "split", nm); allw1.append(o1)
            o2, nf, cnt = screen_2d(S, cols); o2.insert(0, "side", side); o2.insert(0, "split", nm); allw2.append(o2)
            meta.append({"split": nm, "desc": desc, "side": side, "N": len(S), "stat": st(S.y), "days": S.day.nunique(),
                         "1D tests": len(o1), "1D p<0.05": int((o1.p_null < 0.05).sum()), "1D p_fw<0.05": int((o1.p_fw < 0.05).sum()),
                         "2D full-cov vars": nf, "2D tests": len(o2), "2D p<0.01": int((o2.p_null < 0.01).sum()),
                         "2D null count p<0.01 median / p95": f"{np.median(cnt):.0f} / {np.percentile(cnt, 95):.0f}",
                         "2D p_fw<0.05": int((o2.p_fw < 0.05).sum())})
            P(meta[-1])
        it = interaction(m, cols); it.insert(0, "split", nm); allint.append(it)
        P(f"{nm} interaction tests {len(it)} · p<0.05 {int((it.p_null < 0.05).sum())} · p_fw<0.05 {int((it.p_fw < 0.05).sum())}")
    pd.concat(allw1).to_csv(PFX + "_within_1d.csv", index=False)
    pd.concat(allw2).to_csv(PFX + "_within_2d.csv.gz", index=False)
    pd.concat(allint).to_csv(PFX + "_interaction.csv", index=False)
    pd.DataFrame(meta).to_csv(PFX + "_within_meta.csv", index=False)
