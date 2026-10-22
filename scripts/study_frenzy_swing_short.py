#!/usr/bin/env python3
"""FRENZY swing short (operator 2026-10-20: "FRENZY pairs fall below the spike average — isn't there a short?").
Pre-registration (frozen before results): reports/FRENZY_SWING_SHORT_2026-10-20_PREREG.md. Episodes: study_frenzy_swing_short_walk.py
(live-engine frenzy_walk re-implemented; parity: study_frenzy_swing_short_validate.py). READ-ONLY on the bot.

Pipeline:  --need   → trades of every cell, writes data/fund_needed.csv (pair, month) for the funding fetch
           (default) → full study: 7 entries × 64 exits, three blocks, null, family-wise max-statistic, LOMO, concentration, tails,
                       interactions with the FRENZY long family, sizing at the $2,600 account → reports/FRENZY_SWING_SHORT_2026-10-20.md + csv
Usage: venv/bin/python scripts/study_frenzy_swing_short.py [--need] [--shuffles 200]
"""
import argparse, glob, os, pickle, sys, time
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "FRENZY_SWING_SHORT_2026-10-20_data")
OUT = os.path.join(R, "reports", "FRENZY_SWING_SHORT_2026-10-20")
DL = os.path.expanduser("~/Downloads")
BAR, HR = 300_000, 3_600_000
SLIP, FEES = 0.0005, 0.10            # 0.05 % slippage per side (in price) · taker 0.05 % × 2 (in % of notional)
FUND_MISSING = -0.10                 # % per 8 h assumed when no funding history (short pays)
HOLDS = [4, 12, 24, 48]; STOPS = [5.0, 7.5, 10.0, 15.0]; TPS = ["none", "base", "tp10", "trail"]
ENTRIES = ["E1", "E2-6", "E2-12", "E2-24", "E3", "E4-10", "E4-30"]
L = 48 * 12
PRIMARY = ("E1", 24, 10.0, "none"); SECONDARY = ("E2-12", 24, 10.0, "none")
B_LIVE0, B_MAST0, B_YEAR0 = pd.Timestamp("2026-10-14"), pd.Timestamp("2026-10-02"), pd.Timestamp("2025-01-01")
FAMILY = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_LITE_RETRY", "FRENZY_WILLY")
TODAY7 = ["ERAUSDT", "ALTUSDT", "AUSDT", "PHAUSDT", "GTCUSDT", "CTRUSDT", "BRUSDT"]
ap = argparse.ArgumentParser(); ap.add_argument("--need", action="store_true"); ap.add_argument("--shuffles", type=int, default=200)
A = ap.parse_args()
rng = np.random.default_rng(20261020)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cell_name(e, h, s, tp):
    return f"{e}|{h}h|stop{s:g}|{tp}"


# ───────────────────────────── real-world inputs ─────────────────────────────
def real_fills():
    m = pd.read_csv(os.path.join(R, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[m.entry_strategy.astype(str).str.startswith("FRENZY")][["pair", "entry_strategy", "opened_at", "closed_at", "pnl_percentage"]].assign(src="master")
    fs = sorted(glob.glob(os.path.join(DL, "scalpars_orders_*2026-10-*.csv")))
    parts = []
    for f in fs:
        x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("pair", "entry_strategy", "opened_at", "closed_at", "pnl_percentage", "status", "is_paper"))
        x = x[x.entry_strategy.astype(str).str.startswith("FRENZY")]
        parts.append(x.assign(src="live" if "_live_" in f else "paper", f=os.path.basename(f)))
    o = pd.concat(parts) if parts else pd.DataFrame()
    o = o.sort_values("f").drop_duplicates(["opened_at", "pair"], keep="last")
    allf = pd.concat([m, o[["pair", "entry_strategy", "opened_at", "closed_at", "pnl_percentage", "src"]]])
    allf["opened"] = pd.to_datetime(allf.opened_at, format="mixed"); allf["closed"] = pd.to_datetime(allf.closed_at, format="mixed", errors="coerce")
    allf = allf.sort_values("src").drop_duplicates(["pair", "opened"], keep="first")   # master wins on overlap (paper 10-13 / 10-14)
    return allf.reset_index(drop=True)


def replay_fills():
    d = pd.read_csv(os.path.join(R, "reports", "ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv"), low_memory=False,
                    usecols=["pair", "entry_strategy", "opened_at", "closed_at", "pnl_percentage", "status"])
    d = d[d.entry_strategy.astype(str).str.startswith("FRENZY") & (d.status == "CLOSED")].copy()
    d["opened"] = pd.to_datetime(d.opened_at, format="mixed"); d["closed"] = pd.to_datetime(d.closed_at, format="mixed")
    return d.reset_index(drop=True)


def journal_evidence():
    rows = []
    for f in sorted(glob.glob(os.path.join(DL, "scalpars_decisions_*2026-10-*.csv"))):
        try:
            x = pd.read_csv(f, usecols=["t", "pair", "gate", "strategy"], low_memory=False)
        except Exception:
            continue
        m = x.gate.astype(str).str.startswith("FRENZY") | x.strategy.astype(str).str.startswith("FRENZY")
        rows.append(x[m][["t", "pair"]])
    j = pd.concat(rows).drop_duplicates(); j["t"] = pd.to_datetime(j.t, format="mixed")
    return j


def funding_table():
    F = {}
    for d in (os.path.join(R, "reports", "backtest_cache", "funding"), os.path.join(D, "funding")):
        for f in glob.glob(os.path.join(d, "*.csv")):
            p = os.path.basename(f)[:-4]
            try:
                x = pd.read_csv(f)
            except Exception:
                continue
            F[p] = pd.concat([F[p], x]) if p in F else x
    return {p: x.drop_duplicates("t").sort_values("t") for p, x in F.items()}


# ───────────────────────────── signals ─────────────────────────────
def signals(eps, fills_by_pair):
    """the first qualifying bar per episode per entry (segment index), frozen definitions (PREREG)."""
    out = []
    for k, e in enumerate(eps):
        on, off, n = e["on_i"], e["off_i"], len(e["t"])
        c, vw, vx, v24, se, bs = e["c"], e["vwap"], e["volx"], e["vol24"], e["seen"], e["bstreak"]
        ok = lambda i: bool(se[min(i, off - 1)] and v24[i] >= 20e6)

        def first(mask_idx):
            for i in mask_idx:
                if ok(i):
                    return int(i)
            return None
        rng_in = np.arange(on + 24, off)
        sig = {}
        has_off = off < n and not e["new_ep_at_end"]
        sig["E1"] = int(off) if has_off and ok(off) else None
        for N in (6, 12, 24):
            sig[f"E2-{N}"] = first(rng_in[bs[rng_in] == N])
        below = rng_in[c[rng_in] < vw[rng_in]]
        sig["E4-10"] = first(below[vx[below] < 10]); sig["E4-30"] = first(below[vx[below] < 30])
        sig["E3"] = None
        f = fills_by_pair.get(e["pair"])
        if f is not None and len(f):
            t_on = e["t"][on]; t_off = e["t"][off] if off < n else e["t"][-1] + BAR
            ff = f[(f.o_ms >= t_on) & (f.o_ms < t_off) & f.c_ms.notna()]
            if len(ff):
                x_ms = ff.c_ms.min()     # the first family exit inside the episode, then the first close below the VWAP after it
                cand = np.arange(on, off)
                cand = cand[(e["t"][cand] + BAR >= x_ms) & (c[cand] < vw[cand])]
                sig["E3"] = first(cand)
        for en, i in sig.items():
            if i is not None:
                out.append((k, en, i))
    return out


# ───────────────────────────── exits (vectorised) ─────────────────────────────
def paths(eps, ks, idx):
    n = len(ks); H = np.full((n, L), np.nan); Lo = H.copy(); O = H.copy(); C = H.copy(); T = np.zeros((n, L), dtype="int64")
    for r, (k, i) in enumerate(zip(ks, idx)):
        e = eps[k]; a, b = i + 1, min(len(e["t"]), i + 1 + L); m = b - a
        if m > 0:
            H[r, :m] = e["h"][a:b]; Lo[r, :m] = e["l"][a:b]; O[r, :m] = e["o"][a:b]; C[r, :m] = e["c"][a:b]; T[r, :m] = e["t"][a:b]
    return H, Lo, O, C, T


def simulate(E, base, H, Lo, O, C, hold, stop, tp):
    """→ (ret % before funding & fees, exit bar index, complete?). E = entry fill (after slippage)."""
    n = len(E); hb = hold * 12
    h, l, o, c = H[:, :hb], Lo[:, :hb], O[:, :hb], C[:, :hb]
    SL = E * (1 + stop / 100)
    if tp == "trail":
        pl = np.where(np.isnan(l), np.inf, l)
        rm = np.minimum.accumulate(np.minimum(pl, E[:, None]), axis=1)
        rm_prev = np.concatenate([E[:, None], rm[:, :-1]], axis=1)
        armed = rm_prev <= E[:, None] * 0.95
        lev = np.where(armed, np.minimum(SL[:, None], rm_prev * 1.03), SL[:, None])
    else:
        lev = np.broadcast_to(SL[:, None], h.shape)
    hit_s = h >= lev
    ks = np.where(hit_s.any(1), hit_s.argmax(1), hb + 10)
    if tp in ("base", "tp10"):
        TPL = np.where(base < E * 0.99, base, np.nan) if tp == "base" else E * 0.90
        hit_t = l <= TPL[:, None]
        kt = np.where(hit_t.any(1), hit_t.argmax(1), hb + 10)
    else:
        TPL = np.full(n, np.nan); kt = np.full(n, hb + 10)
    r = np.arange(n)
    kx = np.minimum(np.minimum(ks, kt), hb - 1)
    X = c[r, kx].copy()
    s_first = (ks <= kt) & (ks < hb)
    X[s_first] = np.maximum(lev[r, ks.clip(max=hb - 1)], o[r, ks.clip(max=hb - 1)])[s_first]
    t_first = (kt < ks) & (kt < hb)
    X[t_first] = np.minimum(TPL, o[r, kt.clip(max=hb - 1)])[t_first]
    X = X * (1 + SLIP)
    complete = ~np.isnan(c[r, kx]) & ~np.isnan(c[:, :kx.max() + 1]).any(1) if False else ~np.isnan(c[r, kx])
    # incomplete path (data ends): mark-to-market at the last close available
    inc = np.isnan(c[r, kx])
    if inc.any():
        lastc = np.array([cc[~np.isnan(cc)][-1] if (~np.isnan(cc)).any() else np.nan for cc in c[inc]])
        X[inc] = lastc * (1 + SLIP)
        kx = kx.copy(); kx[inc] = np.array([np.nonzero(~np.isnan(cc))[0][-1] if (~np.isnan(cc)).any() else 0 for cc in c[inc]])
    ret = (E - X) / E * 100
    reason = np.where(s_first & ~inc, "stop", np.where(t_first & ~inc, "tp", np.where(inc, "open", "time")))
    return ret, kx, complete, reason


def funding_for(F, pair, t0, t1):
    x = F.get(pair)
    if x is None or not len(x) or x.t.max() < t1 - 9 * HR or x.t.min() > t0:
        return (t1 - t0) / (8 * HR) * FUND_MISSING, True
    s = x[(x.t > t0) & (x.t <= t1)].rate.sum() * 100
    return float(s), False


# ───────────────────────────── stats ─────────────────────────────
def boot_ci(g, B=2000, key="day"):
    d = g.groupby(key).net.agg(["sum", "count"])
    if len(d) < 3:
        return np.nan, np.nan
    s, c = d["sum"].values, d["count"].values; i = rng.integers(0, len(d), (B, len(d)))
    m = s[i].sum(1) / c[i].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def summ(g):
    if not len(g):
        return dict(N=0)
    lo, hi = boot_ci(g)
    pos = g.net.sum()
    pc = g.groupby("pair").net.sum(); dc = g.groupby("day").net.sum()
    top5 = g.sort_values("net", ascending=False).iloc[int(np.ceil(len(g) * 0.05)):].net.mean() if len(g) >= 20 else np.nan
    return dict(N=len(g), eps=g.ep.nunique(), pairs=g.pair.nunique(), days=g.day.nunique(), WR=(g.net > 0).mean() * 100, avg=g.net.mean(),
                med=g.net.median(), lo=lo, hi=hi, tot=g.net.sum(), worst=g.net.min(), best=g.net.max(),
                top_pair=(pc.idxmax() if len(pc) else ""), top_pair_share=(pc.max() / pos * 100 if pos > 0 else np.nan),
                top_day_share=(dc.max() / pos * 100 if pos > 0 else np.nan), ex_top5=top5,
                stop_rate=(g.reason == "stop").mean() * 100, fund=g.fund.mean())


def fmt(s, cols=("N", "eps", "days", "WR", "avg", "med", "lo", "hi", "tot", "worst")):
    if not s or s.get("N", 0) == 0:
        return "0 | | | | | | | |"
    f = {"N": "{:.0f}", "eps": "{:.0f}", "days": "{:.0f}", "WR": "{:.0f}%", "avg": "{:+.2f}", "med": "{:+.2f}", "lo": "{:+.2f}", "hi": "{:+.2f}",
         "tot": "{:+.0f}", "worst": "{:+.1f}"}
    return " | ".join(f[c].format(s[c]) if pd.notna(s.get(c)) else "–" for c in cols)


# ───────────────────────────── main ─────────────────────────────
if __name__ == "__main__":
    eps = pickle.load(open(os.environ.get("EPS", os.path.join(D, "episodes.pkl")), "rb"))
    log("episodes", len(eps))
    RF, YF = real_fills(), replay_fills()
    J = journal_evidence()
    for f in (RF, YF):
        f["o_ms"] = f.opened.astype("datetime64[ms]").astype("int64"); f["c_ms"] = f.closed.astype("datetime64[ms]").astype("int64").astype(float)
        f.loc[f.closed.isna(), "c_ms"] = np.nan
    # E3 needs a fill source per period: real fills from 2026-10-02, replay fills before
    fb = {}
    for p, g in pd.concat([YF[YF.opened < B_MAST0], RF[RF.opened >= B_MAST0]]).groupby("pair"):
        fb[p] = g
    S = signals(eps, fb)
    log("signals", len(S))
    sig = pd.DataFrame(S, columns=["ep", "entry", "i"])
    sig["pair"] = [eps[k]["pair"] for k in sig.ep]; sig["t_ms"] = [int(eps[k]["t"][i]) + BAR for k, i in zip(sig.ep, sig.i)]
    sig["entry_dt"] = pd.to_datetime(sig.t_ms, unit="ms")
    sig["c"] = [eps[k]["c"][i] for k, i in zip(sig.ep, sig.i)]; sig["vwap"] = [eps[k]["vwap"][i] for k, i in zip(sig.ep, sig.i)]
    sig["base"] = [eps[k]["base"] for k in sig.ep]; sig["spike"] = pd.to_datetime([eps[k]["spike_ts"] for k in sig.ep], unit="ms")
    sig["E"] = sig.c * (1 - SLIP)
    sig = sig[sig.entry_dt >= B_YEAR0].reset_index(drop=True)
    H, Lo, O, C, T = paths(eps, sig.ep.values, sig.i.values)
    # all 448 cells
    rows = []
    for hold in HOLDS:
        for stop in STOPS:
            for tp in TPS:
                ret, kx, comp, reason = simulate(sig.E.values, sig.base.values, H, Lo, O, C, hold, stop, tp)
                x_ms = T[np.arange(len(sig)), kx] + BAR
                rows.append(pd.DataFrame({"row": sig.index, "hold": hold, "stop": stop, "tp": tp, "gross": ret, "x_ms": x_ms, "reason": reason}))
    X = pd.concat(rows, ignore_index=True)
    X = X.merge(sig[["ep", "entry", "pair", "t_ms", "entry_dt", "spike", "c", "vwap", "base"]], left_on="row", right_index=True)
    X.loc[X.x_ms <= X.t_ms, "x_ms"] = X.t_ms + BAR
    if A.need:
        mm = set()
        for p, a, b in X[["pair", "t_ms", "x_ms"]].drop_duplicates().itertuples(index=False):
            for m in pd.period_range(pd.to_datetime(a, unit="ms"), pd.to_datetime(b, unit="ms"), freq="M").strftime("%Y-%m"):
                mm.add((p, m))
        F0 = funding_table(); need = []
        for p, m in sorted(mm):
            x = F0.get(p)
            have = set(pd.to_datetime(x.t, unit="ms").dt.strftime("%Y-%m")) if x is not None and len(x) else set()
            if m not in have or m == "2026-10":
                need.append((p, m))
        pd.DataFrame(need, columns=["pair", "month"]).to_csv(os.path.join(D, "fund_needed.csv"), index=False)
        log("fund needed", len(need), "pairs", len({p for p, _ in need})); sys.exit(0)
    F = funding_table()
    fu = {}
    for p, a, b in X[["pair", "t_ms", "x_ms"]].drop_duplicates().itertuples(index=False):
        fu[(p, a, b)] = funding_for(F, p, a, b)
    X["fund"] = [fu[(p, a, b)][0] for p, a, b in zip(X.pair, X.t_ms, X.x_ms)]
    X["fund_missing"] = [fu[(p, a, b)][1] for p, a, b in zip(X.pair, X.t_ms, X.x_ms)]
    X["net"] = X.gross - FEES + X.fund
    X["net_nofund"] = X.gross - FEES
    X["day"] = X.entry_dt.dt.strftime("%Y-%m-%d"); X["month"] = X.entry_dt.dt.strftime("%Y-%m")
    X["block"] = np.where(X.entry_dt >= B_LIVE0, "1_live", np.where(X.entry_dt >= B_MAST0, "2_master", "3_year"))
    X["period"] = np.where(X.entry_dt < pd.Timestamp("2026-01-01"), "2025", np.where(X.entry_dt < pd.Timestamp("2026-07-01"), "2026H1", "2026H2"))
    X["cell"] = [cell_name(e, h, s, t) for e, h, s, t in zip(X.entry, X.hold, X.stop, X.tp)]
    # real-episode evidence for blocks ① ②
    ev = {}
    for k in X.ep.unique():
        e = eps[k]; t_on = pd.to_datetime(e["t"][e["on_i"]], unit="ms"); t_off = pd.to_datetime(e["t"][min(e["off_i"], len(e["t"]) - 1)], unit="ms") + pd.Timedelta(minutes=5)
        jj = J[(J.pair == e["pair"]) & (J.t >= t_on) & (J.t <= t_off)]
        ff = RF[(RF.pair == e["pair"]) & (RF.opened >= t_on) & (RF.opened <= t_off)]
        ev[k] = (len(jj) > 0) or (len(ff) > 0)
    X["real_ep"] = X.ep.map(ev)
    X.to_pickle(os.path.join(D, "cells.pkl"))
    log("cells", X.cell.nunique(), "rows", len(X))

    # ───── cell table (year block) ─────
    Y = X[X.block == "3_year"]
    cs = []
    for cell, g in Y.groupby("cell"):
        s = summ(g); s["cell"] = cell
        s["p2025"], s["p26H1"], s["p26H2"] = (g[g.period == p].net.mean() for p in ("2025", "2026H1", "2026H2"))
        lomo = [g[g.month != m].net.mean() for m in g.month.unique()]; s["lomo_min"] = min(lomo) if lomo else np.nan
        s["avg_nofund"] = g.net_nofund.mean()
        cs.append(s)
    CS = pd.DataFrame(cs).set_index("cell")
    CS.to_csv(OUT + "_cells_year.csv")

    # ───── shuffled-entry null (random bar inside the same episode, pre-funding) + family-wise max-statistic ─────
    epi = sorted(Y.ep.unique())
    lo_b = np.array([eps[k]["on_i"] + 24 for k in epi]); hi_b = np.array([eps[k]["off_i"] for k in epi])
    valid = hi_b > lo_b
    cell_eps = {c: g.ep.values for c, g in Y.groupby("cell")}
    real_nf = Y.groupby("cell").net_nofund.mean()
    pos_of = {k: j for j, k in enumerate(epi)}
    NULL = {c: [] for c in cell_eps}
    cidx = {c: np.array([pos_of[k] for k in v]) for c, v in cell_eps.items()}
    for b in range(A.shuffles):
        ri = np.where(valid, rng.integers(lo_b, np.maximum(hi_b, lo_b + 1) + 1), lo_b)
        ri = np.minimum(ri, np.array([len(eps[k]["t"]) - 2 for k in epi]))
        ks = np.array(epi)
        Hh, Ll, Oo, Cc, _ = paths(eps, ks, ri)
        Ee = np.array([eps[k]["c"][i] for k, i in zip(ks, ri)]) * (1 - SLIP); Bb = np.array([eps[k]["base"] for k in ks])
        for hold in HOLDS:
            for stop in STOPS:
                for tp in TPS:
                    ret, _, _, _ = simulate(Ee, Bb, Hh, Ll, Oo, Cc, hold, stop, tp)
                    net = ret - FEES
                    for en in ENTRIES:
                        c = cell_name(en, hold, stop, tp)
                        if c in cell_eps:
                            NULL[c].append(np.nanmean(net[cidx[c]]))
        if b % 20 == 0:
            log("shuffle", b)
    NM = pd.DataFrame(NULL)                                   # shuffles × cells
    mu, sd = NM.mean(), NM.std(ddof=1).replace(0, np.nan)
    Z = (NM - mu) / sd; zmax = Z.max(axis=1).values
    CS["null_mean"] = mu; CS["real_nofund"] = real_nf
    CS["p_own"] = [(NM[c] >= real_nf[c]).mean() for c in CS.index]
    CS["z"] = (real_nf - mu) / sd
    CS["p_fw"] = [(zmax >= z).mean() if pd.notna(z) else np.nan for z in CS.z]
    CS.to_csv(OUT + "_cells_year.csv")
    pickle.dump(dict(NM=NM), open(os.path.join(D, "null.pkl"), "wb"))
    log("null done")
