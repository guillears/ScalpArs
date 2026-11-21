#!/usr/bin/env python3
"""STATE_EXIT_2026-11-21 — evaluation (research only). Reads fills_out.csv + dec_<SLEEVE>.pkl (STATE_EXIT_2026-11-21_build.py).

Rule simulation: an extra exit on top of today's exit = the FIRST decision minute where (gate ∧ condition) holds → taker exit at the next
minute's open (label = exit-now − hold). A fill where it never fires keeps today's result (Δ = 0). One FILL = one unit.
Gates: G1 gross now ≥ +1 % · G2 now ≥ +2 % · G3 in-trade peak so far ≥ +2 % (any P&L now — a give-back protection that can exit at a loss).
Layer 1 = the PRE-REGISTERED thesis family (frozen thresholds, written in the report before outcomes) with its OWN null; layer 2 = the broad
screen (every feature at sign / terciles / quintiles with thresholds frozen on the BACKTEST DESIGN decision points, + every 2D sign quadrant)
with a family-wise null. Null = circular shift of each fill's condition sequence within its own eligible minutes (same shift for every
condition of that fill → joint structure kept), 300 shuffles, statistic = per-fill mean Δ (layer 1) / t (layer 2), family-wise max.
Verdict (pre-committed, coordinator-revised): CANDIDATE iff on MASTER real fills Δ ≥ 0, SAVED $ ≥ CHOPPED $, day-block bootstrap P(Δ > 0)
≥ 0.80 and Δ > the best fixed-TP alternative's Δ on master; the backtest DESIGN / HOLDOUT is the cross-check (caveat, not a veto).
$ at one size: notional $3,050 per fill (0.61 × a $5,000 account); DCR at x = 0.61 of equity per fill (ONE size).
→ reports/STATE_EXIT_2026-11-21_data/eval_<SLEEVE>.pkl (tables) — the report script renders them."""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_sp = importlib.util.spec_from_file_location("SEC", os.path.join(HERE, "STATE_EXIT_2026-11-21_common.py"))
C = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C)
NOT = 3050.0
X = 0.61
NSH = 300
PERIODS = ("MASTER", "BATCH", "DESIGN", "HOLDOUT")
NONFEAT = {"fid", "k", "label", "now", "rsi_max_in", "rsi_min_in", "below_peak", "climax_near", "climax_bar", "x_ema58"}


def thesis(s):
    """sleeve → {name: (description, fn(dec) → bool array)} — FROZEN (see the report's design section)."""
    L = s == "HT"
    T = {}
    if L:
        T["RSI_ROLL"] = ("5m RSI14 in-trade max ≥ 75, now ≥ 10 pts below it", lambda d: (d.rsi_max_in >= 75) & (d.rsi_5m <= d.rsi_max_in - 10))
    else:
        T["RSI_ROLL"] = ("5m RSI14 in-trade min ≤ 25, now ≥ 10 pts above it", lambda d: (d.rsi_min_in <= 25) & (d.rsi_5m >= d.rsi_min_in + 10))
    T["LH2"] = ("≥ 2 consecutive lower highs (short: higher lows) on closed 5m bars, close off the in-trade extreme",
                lambda d: (d.lh_run_5m >= 2) & (d.below_peak == 1))
    T["EMA_X"] = ("5m EMA5 crosses EMA8 against us this bar and EMA5 beyond EMA13 against us", lambda d: d.x_ema58 == 1)
    T["CLIMAX"] = ("an against-us 5m candle on quote volume ≥ 2 × its prior-20 mean, its extreme within 1 % of the in-trade extreme",
                   lambda d: (d.climax_bar == 1) & (d.climax_near == 1))
    T["FUND_RECV"] = ("the position RECEIVES ≥ 0.05 % per settlement (opposite side crowded)", lambda d: d.fund_recv >= 0.05)
    T["FUND_PAY"] = ("the position PAYS ≥ 0.05 % per settlement", lambda d: d.fund_recv <= -0.05)
    if s == "FB":
        T["FB_VWAP"] = ("last closed 1h close back ABOVE the spike VWAP", lambda d: d.dvwap_1h > 0)
        T["FB_4H"] = ("last closed 4h close back above the entry price", lambda d: d.d4h_entry < 0)
        T["FB_BASE"] = ("price within 1 % of the pre-pump base (thesis complete)", lambda d: d.dbase <= 1.0)
        T["FB_VOL"] = ("rolling 1h quote volume ≥ 50 % of the episode's peak hourly volume (second pump)", lambda d: d.vd_now >= 0.5)
    if s == "HT":
        T["HT_DI"] = ("5m +DI − −DI < 0 (down-moves dominate)", lambda d: d.di_5m < 0)
        T["HT_ADX"] = ("5m ADX lower than 3 bars ago", lambda d: d.adxd_5m < 0)
        T["HT_VOLFADE"] = ("5m volume trend (last 12 bars ÷ 12 before) < 0.5", lambda d: d.voltr_5m < 0.5)
        T["HT_NEWHIGH"] = ("price at / above the spike high H (thesis complete)", lambda d: d.dH >= 0)
        T["HT_VWAP"] = ("price below the spike VWAP", lambda d: d.dvwap < 0)
        T["HT_EMAGAP"] = ("5m EMA5 at / below EMA20", lambda d: d.g5_20_5m <= 0)
    if s == "EXH":
        T["EXH_RSI_OS"] = ("15m RSI14 ≤ 30 (exhaustion played out)", lambda d: d.rsi_15m <= 30)
        T["EXH_RSI_UP"] = ("15m RSI rising 2 closed bars in a row while < 50", lambda d: (d.rsi_15m_d1 > 0) & (d.rsi_15m_d2 > 0) & (d.rsi_15m < 50))
        T["EXH_VOLBACK"] = ("1h volume back ≥ 80 % of the 24 h / episode peak with the last hour UP (buyers returning)",
                            lambda d: (d.vd_now >= 0.8) & (d.r60 > 0))
        T["EXH_OI_UP"] = ("open interest +5 % or more in the last hour", lambda d: d.oi_chg_1h >= 5)
        T["EXH_TB_UP"] = ("taker-buy share of the last 15 min ≥ 55 %", lambda d: d.tb15 >= 0.55)
    return T


GATES = {"G1": lambda d: d.pnl >= 1.0, "G2": lambda d: d.pnl >= 2.0, "G3": lambda d: d.peak >= 2.0}


def first_fire(dec, m):
    idx = np.flatnonzero(m)
    if not len(idx):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    f = dec.fid.values[idx]
    u, first = np.unique(f, return_index=True)
    return pd.Series(dec.label.values[idx[first]], index=u), pd.Series(dec.k.values[idx[first]], index=u)


def per_fill_delta(dec, m, fids):
    lab, kk = first_fire(dec, m)
    dl = pd.Series(0.0, index=fids)
    lab = lab[lab.index.isin(fids)]
    dl.loc[lab.index] = lab.values
    return dl, kk


def boot(delta, cl, B=3000, seed=7):
    d = np.asarray(delta, float)
    if len(d) < 3:
        return np.nan, np.nan, np.nan
    u, inv = np.unique(np.asarray(cl), return_inverse=True)
    s = np.bincount(inv, d, len(u)); n = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(u), (B, len(u)))
    m = s[ix].sum(1) / n[ix].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean()


def dcr(net, xt, d0, d1):
    s = pd.Series(np.asarray(net, float) * X / 100).groupby(pd.to_datetime(np.asarray(xt), unit="ms").normalize()).sum()
    days = pd.date_range(pd.Timestamp(d0).normalize(), pd.Timestamp(d1).normalize(), freq="D")
    r = s.reindex(days, fill_value=0.0).values
    return ((np.prod(1 + r)) ** (1 / len(r)) - 1) * 100 if len(r) else np.nan


def rule_row(O, dl, kk, name, per):
    o = O.set_index("fid").loc[dl.index]
    after = o.base_net + dl
    fired = dl.index.isin(kk.index) & (dl != 0) | dl.index.isin(kk.index)
    fired = pd.Series(dl.index.isin(kk.index), index=dl.index)
    why = o.base_why
    tp = why == "TP"; loss = o.base_net < 0
    lo, hi, pp = boot(dl.values, o.day.values)
    loo = ((dl.sum() - dl) / max(len(dl) - 1, 1))
    xt_after = o.base_xt.copy()
    kk2 = kk[kk.index.isin(dl.index)]
    xt_after.loc[kk2.index] = o.loc[kk2.index, "t0"] + (kk2.values + 1) * C.MIN
    d0, d1 = pd.to_datetime(o.te.min(), unit="ms"), pd.to_datetime(max(o.base_xt.max(), xt_after.max()), unit="ms")
    pair_abs = dl.groupby(o.pair).sum()
    top_share = (pair_abs.abs().max() / dl.abs().sum() * 100) if dl.abs().sum() > 0 else np.nan
    return dict(rule=name, period=per, N=len(dl), fired=int(fired.sum()), fired_tp=int((fired & tp).sum()), fired_loss=int((fired & loss).sum()),
                WR_before=(o.base_net > 0).mean() * 100, WR_after=(after > 0).mean() * 100, avg_before=o.base_net.mean(), avg_after=after.mean(),
                d_mean=dl.mean(), ci_lo=lo, ci_hi=hi, p_pos=pp,
                saved_pts=dl[~tp & (dl > 0)].sum(), saved_usd=dl[~tp & (dl > 0)].sum() * NOT / 100,
                chopped_pts=-dl[tp & (dl < 0)].sum(), chopped_usd=-dl[tp & (dl < 0)].sum() * NOT / 100,
                other_pts=dl[~((~tp & (dl > 0)) | (tp & (dl < 0)))].sum(), sum_usd=dl.sum() * NOT / 100,
                loo_min=loo.min(), loo_max=loo.max(), top_pair=pair_abs.abs().idxmax() if len(pair_abs) else "", top_pair_share=top_share,
                dcr_before=dcr(o.base_net, o.base_xt, d0, d1), dcr_after=dcr(after, xt_after, d0, d1))


def shift_perms(dec, gate_mask, nsh, seed):
    """per shuffle: an index array mapping each gated row to a circularly-shifted gated row of the SAME fill."""
    idx = np.flatnonzero(gate_mask)
    f = dec.fid.values[idx]
    u, start, cnt = np.unique(f, return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    pos = np.arange(len(idx)) - np.repeat(start, cnt)
    out = []
    for _ in range(nsh):
        off = np.repeat(rng.integers(0, cnt), cnt)
        out.append(idx[np.repeat(start, cnt) + (pos + off) % np.repeat(cnt, cnt)])
    return idx, out


def fast_mean_delta(dec_f, lab, idx_sel, nfills):
    """mean per-fill Δ when the rule fires at the gated rows idx_sel (sorted by fill, k)."""
    if not len(idx_sel):
        return 0.0
    f = dec_f[idx_sel]
    _, first = np.unique(f, return_index=True)
    return lab[idx_sel[first]].sum() / nfills


def evaluate(s):
    O = pd.read_csv(os.path.join(C.D, "fills_out.csv"))
    O = O[(O.sleeve == s) & O.base_net.notna()].copy()
    O["day"] = pd.to_datetime(O.te, unit="ms").dt.strftime("%Y-%m-%d")
    O["t0"] = (O.te // C.MIN) * C.MIN + np.where(O.start == "next", C.MIN, 0)
    dec = pd.read_pickle(os.path.join(C.D, f"dec_{s}.pkl")).sort_values(["fid", "k"]).reset_index(drop=True)
    dec = dec.replace([np.inf, -np.inf], np.nan)
    res = {"O": O}
    # ── layer 1: thesis family
    TH = thesis(s)
    rows = []
    for per in PERIODS:
        fids = O[O.period == per].fid.values
        if not len(fids):
            continue
        dp = dec[dec.fid.isin(fids)]
        for gn, gf in GATES.items():
            gm = gf(dp).values
            for nm, (_, fn) in TH.items():
                try:
                    m = gm & fn(dp).fillna(False).values
                except AttributeError:
                    continue
                dl, kk = per_fill_delta(dp, m, fids)
                rows.append(rule_row(O, dl, kk, f"{nm}·{gn}", per))
            # fixed alternatives and the plain gate (exit at the first gated minute = 'take any +1 / +2')
            dl, kk = per_fill_delta(dp, gm, fids)
            rows.append(rule_row(O, dl, kk, f"ANY·{gn}", per))
        o = O[O.period == per].set_index("fid")
        for c_ in [c for c in O.columns if ((c.startswith("tp") and c[2:3].isdigit()) or c.startswith("late")) and o[c].notna().any()]:
            dl = (o[c_] - o.base_net)
            r = rule_row(O, dl, pd.Series(dtype=float), f"FIXED_{c_.upper()}", per)
            r["fired"] = int((dl != 0).sum()); r["dcr_after"] = dcr(o[c_], o.base_xt, pd.to_datetime(o.te.min(), unit="ms"), pd.to_datetime(o.base_xt.max(), unit="ms"))
            rows.append(r)
    R1 = pd.DataFrame(rows)
    res["thesis"] = R1
    # thesis-family null (per period with fills: MASTER and the backtest blocks), statistic = per-fill mean Δ, family-wise max
    nulls = {}
    for per in PERIODS:
        fids = O[O.period == per].fid.values
        if len(fids) < 8:
            continue
        dp = dec[dec.fid.isin(fids)].reset_index(drop=True)
        fv, lab = dp.fid.values, dp.label.values
        mx = np.zeros(NSH)
        obs = {}
        for gi, (gn, gf) in enumerate(GATES.items()):
            gm = gf(dp).values
            idx, perms = shift_perms(dp, gm, NSH, 11 + gi)
            for nm, (_, fn) in TH.items():
                try:
                    cm = fn(dp).fillna(False).values
                except AttributeError:
                    continue
                obs[f"{nm}·{gn}"] = fast_mean_delta(fv, lab, idx[cm[idx]], len(fids))
                for b in range(NSH):
                    sel = idx[cm[perms[b]]]
                    v = fast_mean_delta(fv, lab, np.sort(sel), len(fids))
                    mx[b] = max(mx[b], v)
        nulls[per] = dict(q95=float(np.percentile(mx, 95)), obs=obs, p_fw={k: float((mx >= v).mean()) for k, v in obs.items()})
    res["thesis_null"] = nulls
    # ── layer 2: broad screen (G1), thresholds frozen on the backtest DESIGN decision points
    des = dec[dec.fid.isin(O[O.period == "DESIGN"].fid)]
    des = des[des.pnl >= 1.0]
    feats = [c for c in dec.columns if c not in NONFEAT and dec[c].dtype.kind in "fi" and dec[c].notna().mean() > 0.5]
    conds = {}
    for v in feats:
        x = des[v].astype(float).dropna()
        if len(x) < 100:
            continue
        q05, q95 = x.quantile([0.05, 0.95])
        if q05 < 0 < q95:
            conds[f"{v}>0"] = (v, ">", 0.0); conds[f"{v}<=0"] = (v, "<=", 0.0)
        else:
            md = float(x.median()); conds[f"{v}>med({md:.3g})"] = (v, ">", md); conds[f"{v}<=med({md:.3g})"] = (v, "<=", md)
        for nq, lab_ in ((3, "T"), (5, "Q")):
            qs = x.quantile(np.linspace(0, 1, nq + 1)[1:-1]).values
            edges = np.r_[-np.inf, qs, np.inf]
            for i in range(nq):
                conds[f"{v}∈{lab_}{i + 1}/{nq}({edges[i]:.3g},{edges[i + 1]:.3g}]"] = (v, "in", (edges[i], edges[i + 1]))
    bin_ = {k: c for k, c in conds.items() if c[1] in (">", "<=")}
    res["coverage"] = pd.DataFrame([dict(variable=v, coverage_design=float(dec[dec.fid.isin(O[O.period == 'DESIGN'].fid)][v].notna().mean()),
                                         coverage_master=float(dec[dec.fid.isin(O[O.period == 'MASTER'].fid)][v].notna().mean()) if (O.period == 'MASTER').any() else np.nan,
                                         n_conditions=sum(1 for c in conds.values() if c[0] == v)) for v in feats])

    def cmask(dp, c):
        v, op, th = c
        x = dp[v].values.astype(float)
        if op == ">":
            return x > th
        if op == "<=":
            return x <= th
        return (x > th[0]) & (x <= th[1])
    screen = {}
    decide = "MASTER" if (O.period == "MASTER").sum() >= 8 else "DESIGN"
    for per in [p for p in PERIODS if (O.period == p).sum() >= 8]:
        fids = O[O.period == per].fid.values
        dp = dec[dec.fid.isin(fids)].reset_index(drop=True)
        gm = (dp.pnl >= 1.0).values
        nsh = 150 if per == decide else 0
        idx, perms = shift_perms(dp, gm, nsh, 99)
        fv, lab = dp.fid.values, dp.label.values
        n = len(fids)
        if not len(idx):
            continue
        fg = fv[idx]; lg = lab[idx]
        _, starts = np.unique(fg, return_index=True)
        G = len(idx); BIG = G + 10; ar = np.arange(G)
        M = {k: cmask(dp, c)[idx] for k, c in conds.items()}
        # perms → positions in the gated space
        pos_of = np.full(len(dp), -1); pos_of[idx] = ar
        permg = [pos_of[pp] for pp in perms]
        keys2 = list(bin_.keys())
        tests = [(k,) for k in M] + [(a_, b_) for i, a_ in enumerate(keys2) for b_ in keys2[i + 1:] if bin_[a_][0] != bin_[b_][0]]

        def tstat(cg):
            first = np.minimum.reduceat(np.where(cg, ar, BIG), starts)
            ok = first < BIG
            nf = int(ok.sum())
            if nf == 0:
                return 0.0, 0.0, 0
            d = np.zeros(n); d[:nf] = lg[first[ok]]
            sd = d.std(ddof=1)
            return (d.mean() / (sd / np.sqrt(n)) if sd > 0 else 0.0), d.mean(), nf
        out = []
        for tst in tests:
            cg = M[tst[0]] if len(tst) == 1 else (M[tst[0]] & M[tst[1]])
            t_, m_, nf = tstat(cg)
            out.append((" ∧ ".join(tst), len(tst), t_, m_, nf))
        S = pd.DataFrame(out, columns=["test", "dim", "t", "d_mean", "fired"])
        if per == decide:
            mx = np.zeros(nsh)
            for bi in range(nsh):
                Mb = {k: v[permg[bi]] for k, v in M.items()}
                best = 0.0
                for tst in tests:
                    cg = Mb[tst[0]] if len(tst) == 1 else (Mb[tst[0]] & Mb[tst[1]])
                    t_, _, nf = tstat(cg)
                    if nf >= 5 and t_ > best:
                        best = t_
                mx[bi] = best
            res["screen_null_q95"] = float(np.percentile(mx, 95)); res["screen_null"] = mx
        screen[per] = S
    res["screen"] = screen; res["screen_decide"] = decide; res["conds"] = conds
    pd.to_pickle(res, os.path.join(C.D, f"eval_{s}.pkl"))
    return res


if __name__ == "__main__":
    for s in sys.argv[1:] or ["FB", "HT", "EXH"]:
        evaluate(s)
        print("done", s, flush=True)
