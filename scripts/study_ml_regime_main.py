#!/usr/bin/env python3
"""ML REGIME study 2026-10-16 — Q1 (era anatomy) · Q2 (day/week screen, block null, walk-forward, persistence) · Q3 (conditional entry
effects inside the selected split) · Q4 (policy, three blocks). Pre-registration: reports/ML_REGIME_PREREG.txt (frozen first).
Inputs read-only. Output reports/ml_regime/{tables.md, screen.csv, wf.csv, policies.csv, summary.json}.
    venv/bin/python -W ignore scripts/study_ml_regime_main.py [--R 500] [--tick]
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_yr6wl_momentum_common import load_replay, load_master_mom, day_boot, breakeven_wr   # noqa: E402

R_SHUF = int(sys.argv[sys.argv.index("--R") + 1]) if "--R" in sys.argv else 500
TICK = "--tick" in sys.argv
SUF = "_tick" if TICK else ""
OUT = os.path.join(ROOT, "reports", "ml_regime")
ERAS = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
EA = {"2025 BULL": "2025", "2026-H1 BEAR": "H1", "2026-H2 BULL": "H2"}
DAY = pd.Timedelta(days=1)
LINES = []


def P(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s); print(s, flush=True)


def f3(x, p=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{p}f}"


# ───────────────────────── data ─────────────────────────
def load():
    R = load_replay(["MOM-long"])
    R["key"] = "r" + R.seed.astype(int).astype(str) + "|" + R.opened_at.astype(str) + "|" + R.pair
    R["era"] = R.regime
    R["usd"] = R.usd_new.astype(float)
    R["tick"] = R.top50.astype(bool)
    X = pd.read_csv(os.path.join(ROOT, "reports/ML_BTCFWD_exit_rows.csv"), usecols=["key", "src"])
    R = R.merge(X, on="key", how="left")
    R["tick"] = R.src.astype(str).eq("tick")          # DEVIATION (declared): pricing flag = exit_rows src (2026 is 100 % tick even off top-50)
    B = pd.read_csv(os.path.join(ROOT, "reports/ML_BTCFWD_fill_features.csv"), usecols=["key", "btc_fwd15", "fwd15"])
    R = R.merge(B, on="key", how="left")
    D = pd.read_pickle(os.path.join(ROOT, "reports/ML_DT2D_features.pkl"))
    D["key"] = "r" + D.seed.astype("Int64").astype(str) + "|" + D.opened_at.astype(str) + "|" + D.pair
    D = D[D.blk.isin(ERAS)][["key", "DT", "k_btc_ret24h"]]
    R = R.merge(D, on="key", how="left")
    M = load_master_mom(); M = M[M.sleeve == "MOM-long"].copy()
    M["batch"] = M.era; M["usd"] = M.pnl.astype(float); M["tick"] = True; M["win"] = M.y > 0
    import scout_ml_trend_lines as S
    Fz = S.load_orders()
    Fz = Fz[(Fz.ts >= pd.Timestamp("2026-09-30")) & ~Fz.washed.astype(bool)].copy()
    Fz["y"] = Fz.pct.astype(float); Fz["day"] = Fz.ts.dt.floor("D"); Fz["win"] = Fz.y > 0
    Fz["oos"] = Fz.ts > M.ts.max()
    return R, M, Fz


def feats():
    F = pd.read_csv(os.path.join(OUT, "day_features.csv"))
    F["day"] = pd.to_datetime(F.day)
    return F.set_index("day")


NAT0 = {"btc_ac5_1", "btc_ac5_3", "btc_ac5_7", "btc_ac15_7", "btc_ft_7", "alt_ac5_3", "alt_ac5_7", "alt_ft_3", "alt_ft_7",
        "ml_d3", "ml_d7", "ml_n30", "r1", "r3", "r7", "r30", "dma20", "dma50", "dma200", "alt_rs7", "ftr12"}
NAT1 = {"btc_vr3_1", "btc_vr3_3", "btc_vr3_7", "btc_vr12_3", "btc_vr12_7", "alt_vr3_1", "alt_vr3_3", "alt_vr3_7", "btc_rvchg"}
PRIMARY = [("btc_vr3_7", +1), ("alt_ft_7", +1), ("btc_ac5_3", +1)]


def cut_of(v, train_vals):
    if v in NAT0:
        return 0.0
    if v in NAT1:
        return 1.0
    return float(np.nanmedian(train_vals))


# ───────────────────────── Q1 ─────────────────────────
def q1(R, M, Fz, F):
    P("\n## Q1 · era anatomy\n")
    P("| block | N | days | fills/day (per seed) | WR % | avg win | avg loss | BE WR % | avg %/fill | tick-priced share |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    blocks = [(e, R[R.era == e]) for e in ERAS] + [("master", M), ("fresh", Fz)]
    for nm, d in blocks:
        y = d.y.values
        nseed = 3 if nm in ERAS else 1
        P(f"| {nm} | {len(d)} | {d.day.nunique()} | {len(d) / d.day.nunique() / nseed:.2f} | {100 * (y > 0).mean():.1f} | "
          f"{y[y > 0].mean():+.3f} | {y[y <= 0].mean():+.3f} | {100 * breakeven_wr(y):.1f} | {y.mean():+.3f} | "
          f"{d.tick.mean() * 100 if 'tick' in d else float('nan'):.0f} % |")
    # exit kind mix
    P("\n**Exit mix by era** (share of fills · avg %):\n")
    kinds = R.exit_kind.value_counts().index[:8]
    P("| exit kind | " + " | ".join(EA[e] for e in ERAS) + " |"); P("|---|---|---|---|")
    for k in kinds:
        row = []
        for e in ERAS:
            d = R[R.era == e]; s = d[d.exit_kind == k]
            row.append(f"{100 * len(s) / len(d):.0f} % · {s.y.mean():+.2f}" if len(s) else "—")
        P(f"| {k} | " + " | ".join(row) + " |")
    # pricing split
    P("\n**Pricing split** (tick = top-50 pairs priced on ticks; 1m = others priced on 1-minute bars, stop-first):\n")
    P("| era | tick N · avg | 1m N · avg | era avg | era avg tick-only | gap vs H2 (all → tick-only) | parity top50 vs exit_rows src=='tick' |")
    P("|---|---|---|---|---|---|---|")
    h2a = R[R.era == ERAS[2]].y.mean(); h2t = R[(R.era == ERAS[2]) & R.tick].y.mean()
    for e in ERAS:
        d = R[R.era == e]; t = d[d.tick]; o = d[~d.tick]
        par = ((d.src == "tick") == d.tick)[d.src.notna()].mean()
        P(f"| {EA[e]} | {len(t)} · {t.y.mean():+.3f} | {len(o)} · {o.y.mean():+.3f} | {d.y.mean():+.3f} | {t.y.mean():+.3f} | "
          f"{d.y.mean() - h2a:+.3f} → {t.y.mean() - h2t:+.3f} | {100 * par:.1f} % |")
    # month table
    P("\n**Month by month** (BT, seeds pooled; with month means of the primary persistence variables and BTC 30d return):\n")
    P("| month | era | N | days | WR % | avg %/fill | btc_vr3_7 | alt_ft_7 | btc_ac5_3 | btc_rv5_7 | r30 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    R["mon"] = R.ts.dt.to_period("M").astype(str)
    for m, d in R.groupby("mon"):
        ff = F[(F.index >= pd.Timestamp(m + "-01")) & (F.index < pd.Timestamp(m + "-01") + pd.offsets.MonthBegin(1))]
        P(f"| {m} | {EA[d.era.iloc[0]]} | {len(d)} | {d.day.nunique()} | {100 * d.win.mean():.0f} | {d.y.mean():+.3f} | "
          f"{ff.btc_vr3_7.mean():.3f} | {ff.alt_ft_7.mean():+.3f} | {ff.btc_ac5_3.mean():+.3f} | {ff.btc_rv5_7.mean():.3f} | {ff.r30.mean():+.1f} |")
    # month eta2 vs day-shuffle null
    P("\n**Is the month level real?** month eta² (share of DAY-mean variance explained by month) vs a null that shuffles days across months inside the era:\n")
    rng = np.random.default_rng(3)
    for e in ERAS:
        d = R[R.era == e]
        g = d.groupby("day").y.mean(); mo = g.index.to_period("M").astype(str).values
        def eta(v, lab):
            s = pd.Series(v).groupby(lab).agg(["mean", "size"]); gm = v.mean()
            return ((s["mean"] - gm) ** 2 * s["size"]).sum() / ((v - gm) ** 2).sum()
        ob = eta(g.values, mo); nl = [eta(rng.permutation(g.values), mo) for _ in range(1000)]
        P(f"- {EA[e]}: eta² {ob:.3f} vs null mean {np.mean(nl):.3f} · p {np.mean(np.array(nl) >= ob):.3f} ({len(set(mo))} months, {len(g)} days)")
    # uniformity across cells
    P("\n**Uniformity** — the era gap inside every cell (cells read only with ≥ 8 days in each era):\n")
    atr = pd.to_numeric(R.entry_atr_pct, errors="coerce"); q = atr[R.era == ERAS[0]].quantile([1 / 3, 2 / 3]).values
    adx = pd.to_numeric(R.entry_adx, errors="coerce")
    cells = {
        "pattern": np.where(R.entry_pattern_w_any_match.fillna(0).astype(float) > 0, "W-any",
                            np.where(R.entry_pattern_c_any_match.fillna(0).astype(float) > 0, "C-any", "none")),
        "hour block (UTC)": (R.hour // 4 * 4).astype(str) + "h",
        "ATR tercile (2025 cuts)": np.where(atr <= q[0], "low", np.where(atr <= q[1], "mid", "high")),
        "pricing": np.where(R.tick, "tick", "1m"),
        "ADX": pd.cut(adx, [-1, 20, 30, 40, 200], labels=["<20", "20-30", "30-40", "≥40"]).astype(str),
        "PAIR_1H_DOWNTREND": np.where(R.DT.fillna(False).astype(bool), "DT", "not DT"),
        "BTC 24h at entry": np.where(R.k_btc_ret24h < 0, "BTC24H down", "BTC24H up"),
    }
    top = R.pair.value_counts().index[:15]
    cells["pair (top-15 by count)"] = np.where(R.pair.isin(top), R.pair, "other")
    P("| dimension | cell | 2025 N · avg | H1 N · avg | H2 N · avg | H1−2025 | H2−2025 |")
    P("|---|---|---|---|---|---|---|")
    cnt = {"h1": [0, 0], "h2": [0, 0]}
    for dim, lab in cells.items():
        R["_c"] = lab
        for c in sorted(set(lab)):
            s = [R[(R.era == e) & (R._c == c)] for e in ERAS]
            if min(x.day.nunique() for x in s) < 8:
                continue
            a = [x.y.mean() for x in s]
            for kk, j in (("h1", 1), ("h2", 2)):
                cnt[kk][0] += a[j] > a[0]; cnt[kk][1] += 1
            P(f"| {dim} | {c} | {len(s[0])} · {a[0]:+.3f} | {len(s[1])} · {a[1]:+.3f} | {len(s[2])} · {a[2]:+.3f} | {a[1] - a[0]:+.3f} | {a[2] - a[0]:+.3f} |")
    P(f"\n- Cells where H1 beat 2025: {cnt['h1'][0]} of {cnt['h1'][1]} · where H2 beat 2025: {cnt['h2'][0]} of {cnt['h2'][1]}.")
    # BTC-forward decomposition
    P("\n**What decided the era gap: BTC's path after entry, or the coin's response to it?** (diagnostic — btc_fwd15 is NOT knowable at entry)\n")
    P("| era | N with btc_fwd15 | mean btc_fwd15 % | share btc_fwd15 > 0 | slope b (%/fill per 1 % BTC) | intercept a (BTC flat) | avg y |")
    P("|---|---|---|---|---|---|---|")
    coef = {}
    for e in ERAS:
        d = R[(R.era == e) & R.btc_fwd15.notna()]
        b, a = np.polyfit(d.btc_fwd15.values, d.y.values, 1)
        coef[e] = (a, b, d.btc_fwd15.mean())
        P(f"| {EA[e]} | {len(d)} | {d.btc_fwd15.mean():+.4f} | {100 * (d.btc_fwd15 > 0).mean():.1f} % | {b:+.3f} | {a:+.3f} | {d.y.mean():+.3f} |")
    a0, b0, m0 = coef[ERAS[0]]; a2, b2, m2 = coef[ERAS[2]]
    P(f"\n- Gap 2025 vs H2 = {a0 + b0 * m0 - (a2 + b2 * m2):+.3f}: BTC-path part (H2 slope × Δ mean btc_fwd15) {b2 * (m0 - m2):+.3f} · "
      f"response part (Δ intercept + Δ slope × 2025 mean) {(a0 - a2) + (b0 - b2) * m0:+.3f}.")
    # per-pair concentration of 2025 deficit vs H2 level
    d = R[R.era == ERAS[0]]
    exc = (d.y - h2a).groupby(d.pair).sum().sort_values()
    P(f"- 2025 deficit vs the H2 level is spread over {d.pair.nunique()} pairs; the worst 2 pairs carry {100 * exc.iloc[:2].sum() / exc.sum():.0f} % of it "
      f"({', '.join(exc.index[:2])}).")


# ───────────────────────── Q2 screen ─────────────────────────
def era_grid(R, e):
    d = R[R.era == e]
    days = pd.date_range(d.day.min(), d.day.max(), freq="D")
    s = d.groupby("day").y.agg(["sum", "size"]).reindex(days)
    return days, s["sum"].values, s["size"].values


def horizon(sm, n, H):
    S = np.nan_to_num(sm); N = np.nan_to_num(n)
    cs, cn = np.r_[0, np.cumsum(S)], np.r_[0, np.cumsum(N)]
    L = len(S); i = np.arange(L); j = np.minimum(i + H, L)
    num, den = cs[j] - cs[i], cn[j] - cn[i]
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def block_perm(L, rng, b=7):
    nb = int(np.ceil(L / b)); order = rng.permutation(nb)
    idx = np.concatenate([np.arange(k * b, min((k + 1) * b, L)) for k in order])
    return idx


def stats_vec(X, y, cuts):
    """X: (days, vars) features; y: (days,) outcome. Returns rho, nat delta, t3 delta per var (nan-safe)."""
    ok = np.isfinite(y)
    rho, nat, t3 = [], [], []
    for j in range(X.shape[1]):
        x = X[:, j]; m = ok & np.isfinite(x)
        if m.sum() < 20:
            rho.append(np.nan); nat.append(np.nan); t3.append(np.nan); continue
        xr = pd.Series(x[m]).rank().values; yr = pd.Series(y[m]).rank().values
        rho.append(np.corrcoef(xr, yr)[0, 1])
        hi = x[m] > cuts[j]
        nat.append(y[m][hi].mean() - y[m][~hi].mean() if 3 <= hi.sum() <= m.sum() - 3 else np.nan)
        q1, q2 = np.quantile(x[m], [1 / 3, 2 / 3])
        t3.append(y[m][x[m] > q2].mean() - y[m][x[m] <= q1].mean())
    return np.array(rho), np.array(nat), np.array(t3)


def q2(R, F):
    P("\n## Q2 · regime screen (day / week units, 7-day-block shuffle null)\n")
    V = [c for c in F.columns if c not in ("day_ms",) and F[c].dtype != object]
    rng = np.random.default_rng(17)
    rows = []
    era_obj = {}
    for e in ERAS:
        days, sm, n = era_grid(R, e)
        X = F.reindex(days)[V].values.astype(float)
        cuts = np.array([cut_of(v, X[np.isfinite(sm), k]) for k, v in enumerate(V)])
        Y = {H: horizon(sm, n, H) for H in (1, 3, 7)}
        # weekly unit: Monday features, week outcome
        wk = pd.Series(days).dt.to_period("W-SUN").values
        ws = pd.DataFrame({"wk": wk, "s": np.nan_to_num(sm), "n": np.nan_to_num(n)}).groupby("wk").sum()
        wy = np.where(ws.n > 0, ws.s / ws.n.replace(0, 1), np.nan)
        wdays = pd.DatetimeIndex([p.start_time for p in ws.index])
        WX = F.reindex(wdays)[V].values.astype(float)
        obs = {}
        for H in (1, 3, 7):
            obs[H] = stats_vec(X, Y[H], cuts)
        obs["wk"] = stats_vec(WX, wy, cuts)
        null = {u: [] for u in obs}
        for r in range(R_SHUF):
            idx = block_perm(len(days), rng)
            sm_p, n_p = sm[idx], n[idx]
            for H in (1, 3, 7):
                null[H].append(stats_vec(X, horizon(sm_p, n_p, H), cuts))
            null["wk"].append(stats_vec(WX, wy[rng.permutation(len(wy))], cuts))
        era_obj[e] = dict(days=days, sm=sm, n=n, X=X, cuts=cuts, obs=obs, null=null)
        for u in obs:
            NL = np.array(null[u])                 # (R, 3, vars)
            for si, st in enumerate(("rho", "nat", "t3")):
                o = obs[u][si]; nl = np.abs(NL[:, si, :])
                p = (1 + (nl >= np.abs(o)[None, :]).sum(0)) / (R_SHUF + 1)
                for k, v in enumerate(V):
                    rows.append(dict(era=EA[e], unit=str(u), stat=st, var=v, obs=o[k], p=p[k] if np.isfinite(o[k]) else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, f"screen{SUF}.csv"), index=False)
    # family-wise p: min-p over vars × units per stat, null distribution of min-p from the same shuffles
    P("**Family-wise p** (min per-test p across all variables × units {1d, 3d, 7d, week}, against the null distribution of that minimum):\n")
    P("| family | " + " | ".join(EA[e] for e in ERAS) + " |"); P("|---|---|---|---|")
    fw = {}
    for si, st in enumerate(("rho", "nat", "t3")):
        row = []
        for e in ERAS:
            o = era_obj[e]
            obs_p, null_minp = [], np.ones(R_SHUF)
            for u in o["obs"]:
                NL = np.abs(np.array(o["null"][u])[:, si, :]); ob = np.abs(o["obs"][u][si])
                pobs = (1 + (NL >= ob[None, :]).sum(0)) / (R_SHUF + 1); pobs[~np.isfinite(ob)] = 1
                obs_p.append(pobs.min())
                # per-shuffle p of each null draw (rank within the null)
                srt = np.sort(NL, axis=0)
                pn = (R_SHUF - np.array([np.searchsorted(srt[:, k], NL[:, k], side="left") for k in range(NL.shape[1])]).T) / R_SHUF
                pn[~np.isfinite(NL)] = 1
                null_minp = np.minimum(null_minp, pn.min(1))
            mp = min(obs_p); fwp = (null_minp <= mp).mean()
            fw[(st, e)] = fwp
            row.append(f"{fwp:.2f} (min p {mp:.3f})")
        P(f"| {st} | " + " | ".join(row) + " |")
    nsig = S[S.p <= 0.05].groupby(["era", "stat"]).size().unstack(fill_value=0)
    P(f"\nTests with per-test p ≤ 0.05 (≈ {0.05 * len(V) * 4:.0f} expected by chance per era × family):\n")
    P("| era | " + " | ".join(nsig.columns) + " |"); P("|---" * (len(nsig.columns) + 1) + "|")
    for i, r in nsig.iterrows():
        P(f"| {i} | " + " | ".join(str(x) for x in r.values) + " |")
    # cross-era sign consistency (NAT, 1d) vs null
    nat1 = {e: era_obj[e]["obs"][1][1] for e in ERAS}
    same = np.array([np.isfinite(nat1[ERAS[0]][k]) and all(np.isfinite(nat1[e][k]) for e in ERAS) and
                     len({np.sign(nat1[e][k]) for e in ERAS}) == 1 for k in range(len(V))])
    nulls = []
    for r in range(R_SHUF):
        sg = [np.sign(np.array(era_obj[e]["null"][1][r][1])) for e in ERAS]
        nulls.append(int(np.sum((sg[0] == sg[1]) & (sg[1] == sg[2]) & np.isfinite(sg[0] * sg[1] * sg[2]))))
    P(f"\n**Cross-era sign agreement (NAT split, 1 d):** {same.sum()} of {len(V)} variables point the same way in all three eras; "
      f"shuffle null mean {np.mean(nulls):.1f}, 95th pct {np.percentile(nulls, 95):.0f} → p {np.mean(np.array(nulls) >= same.sum()):.2f}.")
    # era-demeaned pooled test
    P("\n**Era-demeaned pooled test** (outcome minus its era mean, 3 eras pooled, 7-day-block shuffle within era; NAT split 1 d and 7 d; cut = natural or pooled median):\n")
    pooled = []
    Xs, Ys, ns, Y7 = [], [], [], []
    for e in ERAS:
        o = era_obj[e]; y1 = horizon(o["sm"], o["n"], 1); mu = np.nansum(o["sm"]) / np.nansum(o["n"])
        Xs.append(o["X"]); Ys.append(y1 - mu); Y7.append(horizon(o["sm"], o["n"], 7) - mu)
    Xp = np.vstack(Xs); cutsp = np.array([cut_of(v, Xp[:, k]) for k, v in enumerate(V)])
    ob1 = stats_vec(Xp, np.concatenate(Ys), cutsp); ob7 = stats_vec(Xp, np.concatenate(Y7), cutsp)
    NL1, NL7 = [], []
    for r in range(R_SHUF):
        y1s, y7s = [], []
        for e in ERAS:
            o = era_obj[e]; idx = block_perm(len(o["days"]), rng); mu = np.nansum(o["sm"]) / np.nansum(o["n"])
            y1s.append(horizon(o["sm"][idx], o["n"][idx], 1) - mu); y7s.append(horizon(o["sm"][idx], o["n"][idx], 7) - mu)
        NL1.append(stats_vec(Xp, np.concatenate(y1s), cutsp)[1]); NL7.append(stats_vec(Xp, np.concatenate(y7s), cutsp)[1])
    NL1, NL7 = np.abs(np.array(NL1)), np.abs(np.array(NL7))
    p1 = (1 + (NL1 >= np.abs(ob1[1])[None, :]).sum(0)) / (R_SHUF + 1); p7 = (1 + (NL7 >= np.abs(ob7[1])[None, :]).sum(0)) / (R_SHUF + 1)
    minp_null = np.array([min(((NL1 >= NL1[r][None, :]).mean(0)).min(), ((NL7 >= NL7[r][None, :]).mean(0)).min()) for r in range(R_SHUF)])
    for k, v in enumerate(V):
        pooled.append(dict(var=v, cut=cutsp[k], d1=ob1[1][k], p1=p1[k], d7=ob7[1][k], p7=p7[k], rho1=ob1[0][k],
                           nat_2025=nat1[ERAS[0]][k], nat_H1=nat1[ERAS[1]][k], nat_H2=nat1[ERAS[2]][k], same=bool(same[k])))
    Pd = pd.DataFrame(pooled).sort_values("p1")
    fw_pool = (minp_null <= min(Pd.p1.min(), Pd.p7.min())).mean()
    P(f"Family-wise p of the best pooled test (min over {len(V)} vars × 2 horizons): **{fw_pool:.2f}**\n")
    P("| var | cut | Δ 1d pooled (p) | Δ 7d pooled (p) | NAT Δ 1d 2025 / H1 / H2 | same sign 3 eras |")
    P("|---|---|---|---|---|---|")
    for r in Pd.head(15).itertuples():
        P(f"| {r.var} | {r.cut:.3g} | {r.d1:+.3f} ({r.p1:.3f}) | {r.d7:+.3f} ({r.p7:.3f}) | {r.nat_2025:+.3f} / {r.nat_H1:+.3f} / {r.nat_H2:+.3f} | {'yes' if r.same else 'no'} |")
    P("\n**Theory primaries** (pre-registered sign +, single tests):\n")
    for v, sg in PRIMARY:
        r = Pd[Pd["var"] == v].iloc[0]
        P(f"- {v}: pooled Δ 1d {r.d1:+.3f} (p {r.p1:.3f}) · 7d {r.d7:+.3f} (p {r.p7:.3f}) · by era {r.nat_2025:+.3f} / {r.nat_H1:+.3f} / {r.nat_H2:+.3f} · "
          f"predicted sign {'held' if (np.sign(r.d1) == sg and r.same) else 'NOT held'}")
    Pd.to_csv(os.path.join(OUT, f"pooled{SUF}.csv"), index=False)
    # walk-forward all directed era pairs (NAT 1d; cut from train; direction from train)
    P("\n**Walk-forward** (cut + direction learnt on era A, applied to era B). DEVIATION (declared, stricter): every era loses, so ANY skip saves Σ %; "
      "the honest score is SELECTIVITY = avg %/fill of kept fills − avg of skipped fills (> 0 = the switch picks the better days):\n")
    wf = []
    for k, v in enumerate(V):
        res = []
        for a in ERAS:
            oa = era_obj[a]
            ya = horizon(oa["sm"], oa["n"], 1); xa = oa["X"][:, k]; m = np.isfinite(ya) & np.isfinite(xa)
            if m.sum() < 20:
                continue
            c = cut_of(v, xa[m]); hi = xa[m] > c
            if hi.sum() < 3 or (~hi).sum() < 3:
                continue
            good_hi = ya[m][hi].mean() > ya[m][~hi].mean()
            for b in ERAS:
                if b == a:
                    continue
                ob_ = era_obj[b]; xb = ob_["X"][:, k]
                bad = np.isfinite(xb) & ((xb <= c) if good_hi else (xb > c))
                sk = np.nansum(ob_["sm"][bad]); nf = np.nansum(ob_["n"])
                nb = np.nansum(ob_["n"][bad]); nk = nf - nb
                selv = ((np.nansum(ob_["sm"]) - sk) / nk - sk / nb) if (nb > 0 and nk > 0) else np.nan
                res.append(dict(var=v, train=EA[a], test=EA[b], d100=-sk / nf * 100, skipped=nb / nf, sel=selv))
        wf += res
    W = pd.DataFrame(wf); W.to_csv(os.path.join(OUT, f"wf{SUF}.csv"), index=False)
    allpos = W.groupby("var").d100.apply(lambda s: (len(s) == 6) and (s > 0).all())
    P(f"- Σ-% view (trivial for a losing sleeve): {int(allpos.sum())} of {W['var'].nunique()} variables 'save' % in all 6 directions.")
    selpos = W.groupby("var").sel.apply(lambda s: (len(s) == 6) and (s > 0).all())
    nsel = W.groupby("var").sel.apply(lambda s: int((s > 0).sum()))
    # null: same count when each test era's day outcomes are 7-day-block shuffled (R = 200)
    rng2 = np.random.default_rng(41); nulls = []
    for r in range(200):
        perm = {e: block_perm(len(era_obj[e]["days"]), rng2) for e in ERAS}
        cntv = 0
        for k, v in enumerate(V):
            ok = 0; tot = 0
            for a in ERAS:
                oa = era_obj[a]; ya = horizon(oa["sm"], oa["n"], 1); xa = oa["X"][:, k]; m = np.isfinite(ya) & np.isfinite(xa)
                if m.sum() < 20:
                    continue
                c = cut_of(v, xa[m]); hi = xa[m] > c
                if hi.sum() < 3 or (~hi).sum() < 3:
                    continue
                gh = ya[m][hi].mean() > ya[m][~hi].mean()
                for b in ERAS:
                    if b == a:
                        continue
                    ob_ = era_obj[b]; xb = ob_["X"][:, k]; smb = ob_["sm"][perm[b]]; nbv = ob_["n"][perm[b]]
                    bad = np.isfinite(xb) & ((xb <= c) if gh else (xb > c))
                    sk = np.nansum(smb[bad]); nb = np.nansum(nbv[bad]); nk = np.nansum(nbv) - nb
                    tot += 1
                    if nb > 0 and nk > 0 and ((np.nansum(smb) - sk) / nk - sk / nb) > 0:
                        ok += 1
            cntv += (tot == 6 and ok == 6)
        nulls.append(cntv)
    P(f"- **Selectivity view:** {int(selpos.sum())} of {W['var'].nunique()} variables pick the better days in all 6 directed train→test pairs "
      f"({', '.join(selpos[selpos].index) or 'none'}); shuffle null mean {np.mean(nulls):.1f}, 95th pct {np.percentile(nulls, 95):.0f} → "
      f"p {np.mean(np.array(nulls) >= selpos.sum()):.2f}.")
    P(f"- How many of 6 directions select correctly: " + ", ".join(f"{k}: {v}" for k, v in nsel.value_counts().sort_index().items()))
    top = W.groupby("var").sel.agg(["mean", "min", "size"]).sort_values("mean", ascending=False).head(10)
    P("\n| var | mean selectivity (6 dirs) | worst direction |"); P("|---|---|---|")
    for vv, r in top.iterrows():
        P(f"| {vv} | {r['mean']:+.3f} | {r['min']:+.3f} |")
    return Pd, W, era_obj, V, fw, fw_pool


def persistence(R):
    P("\n## Persistence — does a good week predict the next week?\n")
    P("| era | weeks | weekly lag-1 / lag-2 / lag-4 AC (p) | daily lag-1 AC (p) |"); P("|---|---|---|---|")
    rng = np.random.default_rng(5)
    out = {}
    for e in ERAS:
        d = R[R.era == e]
        w = d.groupby(d.ts.dt.to_period("W-SUN")).y.mean()
        g = d.groupby("day").y.mean()
        def ac(s, L):
            s = np.asarray(s); return np.corrcoef(s[L:], s[:-L])[0, 1]
        txt = []
        for L in (1, 2, 4):
            o = ac(w.values, L); nl = [ac(rng.permutation(w.values), L) for _ in range(2000)]
            txt.append(f"{o:+.2f} ({np.mean(np.array(nl) >= o):.2f})")
        o = ac(g.values, 1); nl = [ac(rng.permutation(g.values), 1) for _ in range(2000)]
        out[e] = ac(w.values, 1)
        P(f"| {EA[e]} | {len(w)} | {' / '.join(txt)} | {o:+.2f} ({np.mean(np.array(nl) >= o):.2f}) |")
    return out


# ───────────────────────── Q3 / Q4 ─────────────────────────
def day_lab(F, v, cut, good_hi, days):
    x = F.reindex(pd.DatetimeIndex(days))[v].values
    good = (x > cut) if good_hi else (x <= cut)
    return np.where(np.isfinite(x), good, np.nan)


def q3(R, F, v, cut, good_hi, rng):
    P(f"\n## Q3 · entry effects inside each half of the split **{v} {'>' if good_hi else '≤'} {cut:.3g} = good regime**\n")
    R = R.copy()
    R["good"] = day_lab(F, v, cut, good_hi, R.day)
    adx = pd.to_numeric(R.entry_adx, errors="coerce"); adxp = pd.to_numeric(R.entry_adx_prev, errors="coerce")
    DT = R.DT.fillna(False).astype(bool)
    E = {"E1 ADX 20–30 inside DT (vs rest of DT)": (adx.between(20, 30, inclusive="left"), DT),
         "E2 PAIR_1H_DOWNTREND (vs not)": (DT, pd.Series(True, index=R.index)),
         "E3 BTC24H_DOWN (vs up)": (R.k_btc_ret24h < 0, R.k_btc_ret24h.notna()),
         "E4 ADX falling (vs rising)": (adx < adxp, adx.notna() & adxp.notna())}
    P("| effect | half | 2025 Δ (N in, p) | H1 Δ (N in, p) | H2 Δ (N in, p) | same sign 3 eras | pooled Δ (N-weighted) · p | pass (same sign ∧ pooled p ≤ 0.00625) |")
    P("|---|---|---|---|---|---|---|---|")
    for nm, (flag, base) in E.items():
        for half in (1.0, 0.0):
            ds, ps, ns, nls = [], [], [], []
            for e in ERAS:
                s = R[(R.era == e) & (R.good == half) & base]
                f = flag[s.index].values.astype(bool); y = s.y.values
                if f.sum() < 5 or (~f).sum() < 5:
                    ds.append(np.nan); ps.append(np.nan); ns.append(int(f.sum())); nls.append(None); continue
                ob = y[f].mean() - y[~f].mean()
                blk = ((s.day - s.day.min()).dt.days // 7).values
                nl = []
                for _ in range(300):
                    yy = y.copy()
                    for b in np.unique(blk):
                        ii = np.where(blk == b)[0]; yy[ii] = rng.permutation(yy[ii])
                    nl.append(yy[f].mean() - yy[~f].mean())
                ds.append(ob); ps.append((1 + np.sum(np.abs(nl) >= abs(ob))) / 301); ns.append(int(f.sum())); nls.append(np.array(nl))
            same = all(np.isfinite(ds)) and len({np.sign(x) for x in ds}) == 1
            if all(x is not None for x in nls):
                w = np.array(ns, float) / sum(ns); pob = float(np.dot(w, ds)); pnl = sum(w[i] * nls[i] for i in range(3))
                pp = (1 + np.sum(np.abs(pnl) >= abs(pob))) / 301
            else:
                pob = pp = np.nan
            P(f"| {nm} | {'good' if half == 1 else 'bad'} | {f3(ds[0])} ({ns[0]}, {ps[0]:.3f}) | {f3(ds[1])} ({ns[1]}, {ps[1]:.3f}) | {f3(ds[2])} ({ns[2]}, {ps[2]:.3f}) | "
              f"{'yes' if same else 'no'} | {f3(pob)} · {pp:.3f} | {'PASS' if (same and pp <= 0.00625) else 'fail'} |")


def policy_rows(d, good, label, rng):
    """skip-bad and size-0.5x-bad rows for one block. good: bool array (nan = unknown -> kept)."""
    bad = (good == 0)
    y, u = d.y.values, d.usd.values
    sk = y[bad]
    dsk = -sk.sum()
    days = d.day.values
    # day bootstrap of Δ
    if bad.any():
        dl = np.where(bad, -y, 0.0)
        u_, inv = np.unique(days, return_inverse=True); s = np.bincount(inv, weights=dl)
        idx = rng.integers(0, len(u_), size=(2000, len(u_))); bs = s[idx].sum(1)
        pp = (bs > 0).mean()
        mon = pd.to_datetime(days).to_period("M").astype(str)
        lomo = min((dl[mon != m].sum()) for m in np.unique(mon)) if len(np.unique(mon)) > 1 else np.nan
        wk = pd.to_datetime(days).to_period("W-SUN").astype(str)
        ws = pd.Series(dl).groupby(wk).sum(); maxw = ws.max() / ws[ws > 0].sum() if (ws > 0).any() and dsk > 0 else np.nan
        nwk = len(set(wk[bad]))
    else:
        pp = lomo = maxw = np.nan; nwk = 0
    kept = ~bad
    selv = y[kept].mean() - y[bad].mean() if (kept.any() and bad.any()) else np.nan
    if kept.any() and bad.any():
        u_, inv = np.unique(days, return_inverse=True)
        sk_ = np.bincount(inv, weights=y * kept, minlength=len(u_)); nk_ = np.bincount(inv, weights=kept.astype(float), minlength=len(u_))
        sb_ = np.bincount(inv, weights=y * bad, minlength=len(u_)); nb_ = np.bincount(inv, weights=bad.astype(float), minlength=len(u_))
        idx = rng.integers(0, len(u_), size=(2000, len(u_)))
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = sk_[idx].sum(1) / nk_[idx].sum(1) - sb_[idx].sum(1) / nb_[idx].sum(1)
        psel = float(np.mean(bs[np.isfinite(bs)] > 0))
    else:
        psel = np.nan
    return dict(sel=selv, P_sel=psel, block=label, N=len(d), N_kept=int((~bad).sum()), N_skip=int(bad.sum()), wk_skip=nwk,
                sum_before=y.sum(), avg_before=y.mean(), sum_after=y[~bad].sum(), avg_after=y[~bad].mean() if (~bad).any() else np.nan,
                avg_skipped=sk.mean() if len(sk) else np.nan, d_sum=dsk, d_100=dsk / len(d) * 100,
                usd_before=np.nansum(u), usd_after=np.nansum(u[~bad]), d_usd=-np.nansum(u[bad]),
                win_skipped=int((sk > 0).sum()), loss_avoided=int((sk <= 0).sum()),
                half_size_d=-0.5 * sk.sum(), P=pp, lomo_min=lomo, max_week_share=maxw)


def q4(R, M, Fz, F, v, rng, tag):
    P(f"\n## Q4 · policy on **{v}** ({tag})\n")
    rows = []
    # BT leave-one-era-out: cut and direction from the other two eras (era-demeaned day means)
    for e in ERAS:
        tr = R[R.era != e].copy()
        tr["yd"] = tr.y - tr.groupby("era").y.transform("mean")
        g = tr.groupby("day").yd.mean(); x = F.reindex(g.index)[v]
        m = x.notna(); c = cut_of(v, x[m].values); hi = x[m] > c
        good_hi = g[m][hi].mean() > g[m][~hi].mean()
        d = R[R.era == e]
        rows.append(dict(**policy_rows(d, day_lab(F, v, c, good_hi, d.day), f"BT {EA[e]} (train other eras)", rng),
                         cut=c, good="high" if good_hi else "low"))
    # all-BT training for master / fresh
    tr = R.copy(); tr["yd"] = tr.y - tr.groupby("era").y.transform("mean")
    g = tr.groupby("day").yd.mean(); x = F.reindex(g.index)[v]; m = x.notna()
    c = cut_of(v, x[m].values); good_hi = g[m][x[m] > c].mean() > g[m][x[m] <= c].mean()
    rows.append(dict(**policy_rows(M, day_lab(F, v, c, good_hi, M.day), "MASTER (all)", rng), cut=c, good="high" if good_hi else "low"))
    for b, d in M.groupby("batch"):
        rows.append(dict(**policy_rows(d, day_lab(F, v, c, good_hi, d.day), f"  master {b}", rng), cut=c, good="high" if good_hi else "low"))
    Fz = Fz.copy(); Fz["usd"] = Fz.usd.astype(float)
    rows.append(dict(**policy_rows(Fz, day_lab(F, v, c, good_hi, Fz.day), "FRESH (≥ 09-30)", rng), cut=c, good="high" if good_hi else "low"))
    o = Fz[Fz.oos]
    if len(o):
        rows.append(dict(**policy_rows(o, day_lab(F, v, c, good_hi, o.day), "FRESH after last master fill", rng), cut=c, good="high" if good_hi else "low"))
    T = pd.DataFrame(rows)
    P("| block | cut · good side | N → kept | skipped (weeks) | avg before → after | avg skipped | **selectivity kept − skipped (P>0)** | Δ Σ % (per 100 fills) | Δ $ | winners skipped / losers avoided | 0.5× sizing Δ Σ % | P(Δ>0) | LOMO min | max week share |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in T.itertuples():
        P(f"| {r.block} | {r.cut:.3g} · {r.good} | {r.N} → {r.N_kept} | {r.N_skip} ({r.wk_skip}) | {f3(r.avg_before)} → {f3(r.avg_after)} | {f3(r.avg_skipped)} | {f3(r.sel)} ({f3(r.P_sel, 2)}) | "
          f"{r.d_sum:+.1f} ({r.d_100:+.1f}) | {r.d_usd:+,.0f} | {r.win_skipped} / {r.loss_avoided} | {r.half_size_d:+.1f} | {f3(r.P, 2)} | {f3(r.lomo_min, 1)} | {f3(r.max_week_share, 2)} |")
    T["var"] = v
    return T


def main():
    os.makedirs(OUT, exist_ok=True)
    R, M, Fz = load()
    if TICK:
        R = R[R.tick].copy()
    F = feats()
    P(f"# ML REGIME study tables{' (TICK-PRICED BT FILLS ONLY)' if TICK else ''} — R = {R_SHUF}\n")
    P("Parity: " + " · ".join(f"{EA[e]} {len(R[R.era == e])} @ {R[R.era == e].y.mean():+.3f}" for e in ERAS) +
      f" · master {len(M)} · {100 * M.win.mean():.1f} % · {M.y.mean():+.3f} · fresh {len(Fz)} ({int(Fz.oos.sum())} after master) @ {Fz.y.mean():+.3f}")
    P(f"Join coverage: btc_fwd15 {R.btc_fwd15.notna().mean():.3f} · DT {R.DT.notna().mean():.3f} · exit_rows src {R.src.notna().mean():.3f}")
    q1(R, M, Fz, F)
    Pd, W, era_obj, V, fw, fw_pool = q2(R, F)
    pers = persistence(R)
    # selection rule (prereg §8)
    cand = Pd[Pd.same].sort_values("p1")
    prim = [v for v, _ in PRIMARY]
    sel = None
    if len(cand) and cand.p1.iloc[0] <= 0.05:
        best = cand[cand.p1 == cand.p1.iloc[0]]
        sel = next((v for v in prim if v in set(best["var"])), best["var"].iloc[0])
    sel_lab = sel if sel else (cand["var"].iloc[0] if len(cand) else Pd["var"].iloc[0])
    P(f"\n## Selection (prereg §8)\n\n- Best same-sign variable: **{sel_lab}** (pooled p 1d "
      f"{Pd.set_index('var').loc[sel_lab, 'p1']:.3f}); passes the p ≤ 0.05 bar: {'yes' if sel else 'no — descriptive only'}.")
    rng = np.random.default_rng(29)
    for v in dict.fromkeys([sel_lab, "btc_vr3_7"]):
        pr = Pd.set_index("var").loc[v]
        q3(R, F, v, pr.cut, pr.d1 > 0, rng)
    Ts = [q4(R, M, Fz, F, v, rng, "selected" if v == sel_lab else "theory primary") for v in dict.fromkeys([sel_lab, "btc_vr3_7", "alt_ft_7", "btc_ac5_3"])]
    pd.concat(Ts).to_csv(os.path.join(OUT, f"policies{SUF}.csv"), index=False)
    json.dump(dict(family_wise={f"{k[0]}|{EA[k[1]]}": v for k, v in fw.items()}, fw_pooled=fw_pool, selected=sel_lab, selected_pass=bool(sel),
                   weekly_ac1={EA[k]: v for k, v in pers.items()}), open(os.path.join(OUT, f"summary{SUF}.json"), "w"), indent=1)
    open(os.path.join(OUT, f"tables{SUF}.md"), "w").write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
