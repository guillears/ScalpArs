#!/usr/bin/env python3
"""FRENZY winners-vs-losers screen 2026-10-19 (operator: "analyse deeply all FRENZY longs here and in the master batch … ALL variables …
then cross check vs backtest"). Read-only on every input; writes reports/FRENZY_WL_SCREEN_2026-10-19_*.csv only.

Inputs: argv[1] = real-fill pool pickle (scripts/study_frenzy_wl_screen_build.py: master STACK 2026-10-19a kept non-probe + live B23).
Backtest = reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv seed 1 (scripts/study_yr6wl_frenzy_common.py loader; y = pct_new at the
10-13 exits: LONG +1.5/−3 · LITE +3/−3 (pre-LITE-C) · WILLY +1.25/−4/90 (no late TP)).

Method per sleeve (FRENZY_LONG / FRENZY_LITE / FRENZY_WILLY, separately):
  coverage per variable × block (master / live) — unscored never counted as "rest";
  1D: every numeric entry_* / x_* (derived) / j_* (journal order book) column at sign / median / low-tercile / high-tercile;
  2D: exhaustive all-pairs median quadrants over variables scored on ≥ 70 % of the sleeve's fills;
  statistic = pooled-variance t of y (zone − rest, scored fills only) + a second outcome (stop-hit indicator);
  null = day-block shuffle (each UTC day's outcome vector replaced by a permuted donor day's) → familywise p of max |t| and the count
  of |t| ≥ 3 hits; OOS = date halves, master→live, then the backtest at the SAME frozen numeric cut (per regime, top-50 split).
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_yr6wl_frenzy_common as C

POOL = sys.argv[1]
OUTP = os.path.join(ROOT, "reports", "FRENZY_WL_SCREEN_2026-10-19")
SLEEVES = ["FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY"]
N1, N2 = 2000, 300
EXCL = {"entry_price", "entry_fee", "entry_strategy", "entry_order_type", "entry_desired_notional", "entry_liquidity_cap_notional",
        "entry_bracket_cap_notional", "entry_bracket_max_leverage", "entry_frenzy_vwap", "entry_btc_regime_started_at",
        "entry_frenzy_spike_at", "entry_surge_trigger_at"}
EXEC = {"entry_slippage_pct", "entry_book_drift_pct", "entry_book_spread_pct", "entry_fill_vs_book_pct", "entry_order_lag_s", "j_depth1_to_notional"}


def numcol(s):
    if s.dtype == object:
        xs = s.astype(str).str.lower()
        if xs.isin(["true", "false", "nan", "none"]).all():
            return xs.map({"true": 1.0, "false": 0.0})
    return pd.to_numeric(s, errors="coerce").astype(float)


def candidates(D):
    out = {}
    for c in D.columns:
        if c in EXCL or not (c.startswith("entry_") or c.startswith("x_") or c.startswith("j_")):
            continue
        x = numcol(D[c])
        if x.notna().sum() == 0 or x.nunique() < 2:
            continue
        out[c] = x
    return out


def zones_1d(x):
    """[(label, cut description, zone mask over scored, scored mask, frozen rule)] — frozen rule = (op, value) re-applied to the backtest."""
    sc = x.notna().values; v = x.values
    res = []
    u = np.unique(v[sc])
    if len(u) == 2:
        hi = u[1]
        res.append(("bin", f"== {hi:g}", sc & (v == hi), sc, ("==", hi)))
        return res
    if np.nanmin(v) < 0 < np.nanmax(v):
        res.append(("sign", "< 0", sc & (v < 0), sc, ("<", 0.0)))
    med = np.nanmedian(v)
    res.append(("median", f"<= {med:.4g}", sc & (v <= med), sc, ("<=", med)))
    q1, q2 = np.nanpercentile(v[sc], [100 / 3, 200 / 3])
    res.append(("T1", f"<= {q1:.4g}", sc & (v <= q1), sc, ("<=", q1)))
    res.append(("T3", f"> {q2:.4g}", sc & (v > q2), sc, (">", q2)))
    return res


def apply_rule(x, rule):
    op, val = rule
    v = x.values; sc = ~np.isnan(v)
    with np.errstate(invalid="ignore"):
        z = {"==": v == val, "<": v < val, "<=": v <= val, ">": v > val}[op]
    return sc & z, sc


def welch(Z, R, Y):
    """Z, R: n×k 0/1 (zone / rest among scored); Y: n×p → k×p pooled-variance t."""
    Z = Z.astype(float); R = R.astype(float)
    n1 = Z.sum(0)[:, None]; n0 = R.sum(0)[:, None]
    m1 = (Z.T @ Y) / n1; m0 = (R.T @ Y) / n0
    v1 = ((Z.T @ (Y ** 2)) - n1 * m1 ** 2) / np.maximum(n1 - 1, 1)
    v0 = ((R.T @ (Y ** 2)) - n0 * m0 ** 2) / np.maximum(n0 - 1, 1)
    # pooled-variance (Student) t: a Welch t explodes on all-TP zones (zero variance) at N ≈ 30 — discrete TP/SL outcomes
    sp2 = ((n1 - 1) * v1 + (n0 - 1) * v0) / np.maximum(n1 + n0 - 2, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (m1 - m0) / np.sqrt(sp2 * (1 / n1 + 1 / n0) + 1e-12)
    return np.nan_to_num(t, nan=0.0)


def stats_zone(y, z, sc, day, pair):
    yz, yr = y[z], y[sc & ~z]
    lo, hi, p = C.day_boot(yz, day[z]) if z.sum() else (np.nan,) * 3
    loss = yz[yz < 0]
    dz = pd.Series(yz, index=day[z]); pz = pd.Series(yz, index=pair[z])
    neg = -dz[dz < 0].groupby(level=0).sum(); negp = -pz[pz < 0].groupby(level=0).sum()
    return dict(N=int(z.sum()), days=int(len(set(day[z]))), pairs=int(len(set(pair[z]))), WR=round(100 * (yz > 0).mean(), 1) if len(yz) else np.nan,
                avg=round(yz.mean(), 3) if len(yz) else np.nan, sum=round(yz.sum(), 2), ci=f"[{lo:+.2f}, {hi:+.2f}]", P_pos=round(p, 3),
                rest_N=int(len(yr)), rest_WR=round(100 * (yr > 0).mean(), 1) if len(yr) else np.nan, rest_avg=round(yr.mean(), 3) if len(yr) else np.nan,
                unscored=int((~sc).sum()),
                top_day_loss_share=round(neg.max() / neg.sum(), 2) if len(neg) and neg.sum() > 0 else np.nan,
                top_pair_loss_share=round(negp.max() / negp.sum(), 2) if len(negp) and negp.sum() > 0 else np.nan)


# ── backtest (yr6 seed 1) with derived features ─────────────────────────────────────────────────────────────────────────
BT_END = "2026-10-03"   # the real-fill era starts 2026-10-03 → backtest days from then on replay the SAME days: excluded (independence)


def load_bt():
    B = C.load()
    B = B[B.day < BT_END].copy()
    # WILLY: today's live entry also needs the ARMED turnover filter (24 h vol / mcap < 1.0; missing mcap → no entry, DL 314 kept it)
    R = pd.to_numeric(B.entry_vol_mcap_ratio, errors="coerce")
    B = B[~((B.sleeve == "FRENZY_WILLY") & ~(R < 1.0))].copy()
    full = pd.read_csv(C.FILLS, low_memory=False, usecols=["seed", "status", "is_probe", "pair", "opened_at", "closed_at", "sleeve", "in_window"])
    full = full[(full.seed == 1) & (full.status == "CLOSED")]
    full["ts"] = pd.to_datetime(full.opened_at, format="mixed"); full["tc"] = pd.to_datetime(full.closed_at, format="mixed")
    B["tc"] = pd.to_datetime(B.closed_at, format="mixed")
    B["x_hour_utc"] = B.ts.dt.hour + B.ts.dt.minute / 60
    B["x_weekday"] = B.ts.dt.weekday
    B["x_weekend"] = (B.ts.dt.weekday >= 5).astype(float)
    B["x_log_mcap"] = np.log10(pd.to_numeric(B.entry_mcap_usd, errors="coerce").where(lambda s: s > 0))
    B["x_log_vol24"] = np.log10(pd.to_numeric(B.entry_pair_volume_24h_usd, errors="coerce").where(lambda s: s > 0))
    sp = pd.to_datetime(B.entry_frenzy_spike_at.astype(str).str[:19], errors="coerce")
    B["x_minutes_since_spike"] = (B.ts - sp).dt.total_seconds() / 60
    B = B.sort_values("ts").reset_index(drop=True)
    kth, since, prevr, p7 = [], [], [], []
    for pair, g in B.groupby("pair", sort=False):
        ts = g.ts.values; tc = g.tc.values; ep = g.ep.values; yv = g.y.values
        for i in range(len(g)):
            prior = ts < ts[i]
            kth.append((g.index[i], int((prior & (ep == ep[i])).sum()) + 1))
            if prior.any():
                j = np.flatnonzero(prior)[-1]
                since.append((g.index[i], (ts[i] - ts[j]) / np.timedelta64(1, "h")))
                prevr.append((g.index[i], yv[j] if tc[j] <= ts[i] else np.nan))
            p7.append((g.index[i], int((prior & (ts >= ts[i] - np.timedelta64(7, "D"))).sum())))
    for name, lst in [("x_kth_in_episode", kth), ("x_hours_since_prev_fz_fill_pair", since), ("x_prev_fz_fill_pct_pair", prevr), ("x_pair_fz_fills_prev7d", p7)]:
        s = pd.Series(dict(lst)); B[name] = s.reindex(B.index)
    fo = full.ts.values; fc = full.tc.values; fz = full.sleeve.astype(str).str.startswith("FRENZY").values
    o = np.argsort(fo); fo, fc, fz = fo[o], fc[o], fz[o]
    nopen, nfz = [], []
    for t in B.ts.values:
        k = np.searchsorted(fo, t)
        m = fc[:k] > t
        nopen.append(int(m.sum())); nfz.append(int((m & fz[:k]).sum()))
    B["x_open_positions_at_entry"] = nopen; B["x_frenzy_open_at_entry"] = nfz
    B["stopped"] = (B.y <= -2.5).astype(float)
    return B


def bt_check(B, sleeve, var, rule, var2=None, rule2=None):
    S = B[B.sleeve == sleeve]
    if var not in S.columns or (var2 and var2 not in S.columns):
        return [dict(regime="(not in backtest)")]
    x = numcol(S[var]); z, sc = apply_rule(x, rule)
    if var2:
        x2 = numcol(S[var2]); z2, sc2 = apply_rule(x2, rule2); z = z & z2; sc = sc & sc2
    y = S.y.values; day = S.day.values; pair = S.pair.values
    out = []
    for reg in C.REGIMES + ["ALL"]:
        m = np.ones(len(S), bool) if reg == "ALL" else (S.regime.values == reg)
        for t50 in (None, True):
            mm = m & (S.top50.values if t50 else True)
            zz, ss = z & mm, sc & mm
            if zz.sum() < 3 or (ss & ~zz).sum() < 3:
                out.append(dict(regime=reg + (" top50" if t50 else ""), N=int(zz.sum()), cov=round(ss.sum() / max(mm.sum(), 1), 2)))
                continue
            st = stats_zone(y, zz, ss, day, pair)
            st["P_zone_lt_rest"] = round(C.boot_diff_p(y[ss], zz[ss], day[ss], -1), 3)
            st["cov"] = round(ss.sum() / max(mm.sum(), 1), 2)
            st["regime"] = reg + (" top50" if t50 else "")
            out.append(st)
    return out


def main():
    D = pd.read_pickle(POOL)
    B = load_bt()
    cov_rows, res_1d, res_2d, nulls, oos_rows = [], [], [], [], []
    for sleeve in SLEEVES:
        S = D[D.sleeve == sleeve].reset_index(drop=True)
        y = S.y.values.astype(float); st = S.stopped.values.astype(float)
        day = S.day.values; pair = S.pair.values
        cand = candidates(S)
        # coverage
        for c, x in cand.items():
            r = dict(sleeve=sleeve, var=c, kind="journal" if c.startswith("j_") else "derived" if c.startswith("x_") else "stamp",
                     macro=C.is_macro(c) or c in ("x_hour_utc", "x_weekday", "x_weekend", "x_open_positions_at_entry", "x_frenzy_open_at_entry", "j_n_frenzy_pairs_in_book"),
                     execution=c in EXEC, N=len(S), scored=int(x.notna().sum()), cov_all=round(x.notna().mean(), 2))
            for blk in ["master", "live"]:
                m = (S.src == blk).values
                r[f"cov_{blk}"] = round(x[m].notna().mean(), 2) if m.any() else np.nan
            r["in_backtest"] = c in B.columns and numcol(B[B.sleeve == sleeve][c]).notna().mean() > 0 if c in B.columns else False
            r["bt_cov"] = round(numcol(B[B.sleeve == sleeve][c]).notna().mean(), 2) if c in B.columns else 0.0
            cov_rows.append(r)
        # 1D tests
        T = []
        for c, x in cand.items():
            if x.notna().sum() < 12:
                continue
            for lab, cut, z, sc, rule in zones_1d(x):
                if z.sum() < 5 or (sc & ~z).sum() < 5:
                    continue
                T.append((c, lab, cut, z, sc & ~z, rule))
        Z = np.array([t[3] for t in T]).T; R = np.array([t[4] for t in T]).T
        Y = np.column_stack([y, st])
        tob = welch(Z, R, Y)
        Yp = C.block_draws(y, day, N1, 11); Sp = C.block_draws(st, day, N1, 11)
        tnull = welch(Z, R, Yp); snull = welch(Z, R, Sp)
        mx = np.abs(tnull).max(0); smx = np.abs(snull).max(0)
        hits_obs = int((np.abs(tob[:, 0]) >= 3).sum()); hits_null = (np.abs(tnull) >= 3).sum(0)
        nulls.append(dict(sleeve=sleeve, layer="1D", tests=len(T), outcome="pct", obs_max_t=round(np.abs(tob[:, 0]).max(), 2),
                          null_p_max=round((mx >= np.abs(tob[:, 0]).max()).mean(), 3), null_max_t_95=round(np.percentile(mx, 95), 2),
                          hits_t3_obs=hits_obs, hits_t3_null_mean=round(hits_null.mean(), 1), hits_t3_null_95=int(np.percentile(hits_null, 95)),
                          p_hits=round((hits_null >= hits_obs).mean(), 3)))
        hs = int((np.abs(tob[:, 1]) >= 3).sum()); hsn = (np.abs(snull) >= 3).sum(0)
        nulls.append(dict(sleeve=sleeve, layer="1D", tests=len(T), outcome="stop-hit", obs_max_t=round(np.abs(tob[:, 1]).max(), 2),
                          null_p_max=round((smx >= np.abs(tob[:, 1]).max()).mean(), 3), null_max_t_95=round(np.percentile(smx, 95), 2),
                          hits_t3_obs=hs, hits_t3_null_mean=round(hsn.mean(), 1), hits_t3_null_95=int(np.percentile(hsn, 95)), p_hits=round((hsn >= hs).mean(), 3)))
        for i, (c, lab, cut, z, r_, rule) in enumerate(T):
            row = dict(sleeve=sleeve, var=c, gran=lab, cut=cut, t=round(tob[i, 0], 2), p_fw=round((mx >= abs(tob[i, 0])).mean(), 3),
                       t_stop=round(tob[i, 1], 2), p_fw_stop=round((smx >= abs(tob[i, 1])).mean(), 3), rule=rule)
            row.update(stats_zone(y, z, z | r_, day, pair))
            res_1d.append(row)
        # 2D exhaustive median quadrants
        v70 = [c for c, x in cand.items() if x.notna().mean() >= 0.70 and x.nunique() >= 2]
        Q = []
        for a in range(len(v70)):
            xa = cand[v70[a]]; ua = np.unique(xa.dropna())
            ca = ua[0] if len(ua) == 2 else np.nanmedian(xa)
            for b in range(a + 1, len(v70)):
                xb = cand[v70[b]]; ub = np.unique(xb.dropna())
                cb = ub[0] if len(ub) == 2 else np.nanmedian(xb)
                sc = (xa.notna() & xb.notna()).values
                for sa in ("lo", "hi"):
                    za = (xa.values <= ca) if sa == "lo" else (xa.values > ca)
                    for sb in ("lo", "hi"):
                        zb = (xb.values <= cb) if sb == "lo" else (xb.values > cb)
                        z = sc & za & zb
                        if z.sum() < 5 or (sc & ~z).sum() < 5:
                            continue
                        Q.append((v70[a], sa, ca, v70[b], sb, cb, z, sc & ~z))
        Z2 = np.array([q[6] for q in Q]).T; R2 = np.array([q[7] for q in Q]).T
        t2 = welch(Z2, R2, Y)
        mx2 = np.zeros(N2); hn2 = np.zeros(N2); smx2 = np.zeros(N2)
        for k in range(0, N2, 50):
            tt = welch(Z2, R2, Yp[:, k:k + 50]); mx2[k:k + 50] = np.abs(tt).max(0); hn2[k:k + 50] = (np.abs(tt) >= 3.5).sum(0)
            ts_ = welch(Z2, R2, Sp[:, k:k + 50]); smx2[k:k + 50] = np.abs(ts_).max(0)
        h2 = int((np.abs(t2[:, 0]) >= 3.5).sum())
        nulls.append(dict(sleeve=sleeve, layer="2D", tests=len(Q), outcome="pct", obs_max_t=round(np.abs(t2[:, 0]).max(), 2),
                          null_p_max=round((mx2 >= np.abs(t2[:, 0]).max()).mean(), 3), null_max_t_95=round(np.percentile(mx2, 95), 2),
                          hits_t3_obs=h2, hits_t3_null_mean=round(hn2.mean(), 1), hits_t3_null_95=int(np.percentile(hn2, 95)), p_hits=round((hn2 >= h2).mean(), 3)))
        nulls.append(dict(sleeve=sleeve, layer="2D", tests=len(Q), outcome="stop-hit", obs_max_t=round(np.abs(t2[:, 1]).max(), 2),
                          null_p_max=round((smx2 >= np.abs(t2[:, 1]).max()).mean(), 3), null_max_t_95=round(np.percentile(smx2, 95), 2)))
        order = np.argsort(-np.abs(t2[:, 0]))[:400]
        for i in order:
            a, sa, ca, b_, sb, cb, z, r_ = Q[i]
            row = dict(sleeve=sleeve, var=a, side_a=sa, cut_a=round(ca, 5), var2=b_, side_b=sb, cut_b=round(cb, 5), t=round(t2[i, 0], 2),
                       p_fw=round((mx2 >= abs(t2[i, 0])).mean(), 3), t_stop=round(t2[i, 1], 2))
            row.update(stats_zone(y, z, z | r_, day, pair))
            res_2d.append(row)
        # OOS on the top candidates: halves, master→live, backtest at the frozen cut
        r1 = sorted([r for r in res_1d if r["sleeve"] == sleeve], key=lambda r: -abs(r["t"]))[:12]
        r2 = [r for r in res_2d if r["sleeve"] == sleeve][:8]
        cut_day = np.sort(np.unique(day))[len(np.unique(day)) // 2]
        for rk, r in enumerate(r1 + r2):
            two = "var2" in r
            if two:
                z, sc = apply_rule(cand[r["var"]], ("<=" if r["side_a"] == "lo" else ">", r["cut_a"]))
                z2, sc2 = apply_rule(cand[r["var2"]], ("<=" if r["side_b"] == "lo" else ">", r["cut_b"]))
                z, sc = z & z2, sc & sc2
                label = f"{r['var']} {r['side_a']}({r['cut_a']:.4g}) × {r['var2']} {r['side_b']}({r['cut_b']:.4g})"
                bt = bt_check(B, sleeve, r["var"], ("<=" if r["side_a"] == "lo" else ">", r["cut_a"]), r["var2"], ("<=" if r["side_b"] == "lo" else ">", r["cut_b"]))
            else:
                z, sc = apply_rule(cand[r["var"]], r["rule"])
                label = f"{r['var']} {r['cut']} [{r['gran']}]"
                bt = bt_check(B, sleeve, r["var"], r["rule"])
            sign = np.sign(r["t"])
            base = dict(sleeve=sleeve, rank=rk + 1, layer="2D" if two else "1D", test=label, t_real=r["t"], p_fw=r["p_fw"])
            for nm, m in [("H1(real)", day < cut_day), ("H2(real)", day >= cut_day), ("master", (S.src == "master").values), ("live", (S.src == "live").values)]:
                zz, ss = z & m, sc & m
                oos_rows.append({**base, "block": nm, "N": int(zz.sum()), "avg": round(y[zz].mean(), 3) if zz.sum() else np.nan,
                                 "rest_N": int((ss & ~zz).sum()), "rest_avg": round(y[ss & ~zz].mean(), 3) if (ss & ~zz).sum() else np.nan,
                                 "same_sign": (np.sign(y[zz].mean() - y[ss & ~zz].mean()) == sign) if zz.sum() and (ss & ~zz).sum() else np.nan})
            for b in bt:
                d = {**base, "block": "BT " + b.get("regime", "")}
                if "avg" in b:
                    d.update(N=b["N"], days=b["days"], WR=b["WR"], avg=b["avg"], rest_N=b["rest_N"], rest_avg=b["rest_avg"], ci=b["ci"],
                             P_zone_lt_rest=b["P_zone_lt_rest"], cov=b["cov"], same_sign=np.sign(b["avg"] - b["rest_avg"]) == sign)
                else:
                    d.update(N=b.get("N"), cov=b.get("cov"))
                oos_rows.append(d)
    pd.DataFrame(cov_rows).to_csv(OUTP + "_coverage.csv", index=False)
    pd.DataFrame(res_1d).drop(columns=["rule"]).sort_values(["sleeve", "p_fw", "t"]).to_csv(OUTP + "_tests_1d.csv", index=False)
    pd.DataFrame(res_2d).to_csv(OUTP + "_tests_2d_top400.csv", index=False)
    pd.DataFrame(nulls).to_csv(OUTP + "_null.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(OUTP + "_oos_backtest.csv", index=False)
    print(pd.DataFrame(nulls).to_string())


if __name__ == "__main__":
    main()
