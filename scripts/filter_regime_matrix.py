#!/usr/bin/env python3
"""FILTER x BTC-REGIME MATRIX for the momentum-LONG sleeve (2026-10-05, operator: "there must be a MACRO BTC variable that says which
rule makes sense in each period"). Read-only research — step 3 (after filter_regime_extract.py + filter_regime_price.py).

Input : reports/FILTER_REGIME_MATRIX_signals_priced.csv (every momentum-LONG refusal of the yr5 replay, re-priced with the live exit
        replica), BTC caches (k1d / k4h / btc_1h / btc_5m), the yr5 journal SCAN lines (frm_scan.pkl in $S).
Method: each signal tagged with BTC macro state at the signal (last CLOSED bar / last scan ≤ t — no look-ahead). For each gate with
        ≥ 15 signals per side x each pre-declared split (binary: yes/no · tercile: T1 vs T3, cut-points fixed on the year's 5-min grid):
        blocked-cohort avg % per signal in each state (weights n_seeds/3). "Helps" = blocked cohort NEGATIVE (removed losers);
        "hurts" = POSITIVE. A REGIME-CONDITIONAL finding needs ALL of:
          ① opposite signs (helps in one state, hurts in the other) · ② N ≥ 15 signals and ≥ 8 distinct days per state
          ③ the same sign pattern in BOTH halves (Jan–Apr / May–Oct), each state with ≥ 3 days in each half
          ④ day-block bootstrap 95 % CI of the state gap (hurt − help) excluding 0 (days resampled jointly, 2000 reps)
        STRONG = also each state's own 95 % CI excludes 0 on its side.
        SHUFFLED NULL: 200 rounds; each day's tags are replaced by the tags of another day of the SAME month at the same time of day
        (permutation within month); the whole matrix is re-run (bootstrap 300 reps) and the number of passing cells counted.
Out   : reports/FILTER_REGIME_MATRIX_2026-10-05.csv (full matrix), _null.csv, and the tables consumed by the .md writer (this script).
"""
import json, os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
S = os.environ.get("S", "/tmp")
C = "reports/backtest_cache"
DAY, H, MIN = 86_400_000, 3_600_000, 60_000
RNG = np.random.default_rng(20261005)
Y0, Y1 = pd.Timestamp("2026-01-04").value // 10**6, pd.Timestamp("2026-10-04").value // 10**6
HALF = pd.Timestamp("2026-05-01").value // 10**6
OUTCSV = "reports/FILTER_REGIME_MATRIX_2026-10-05.csv"
EXCLUDED_NOTE = None

# ─────────────────────────── BTC macro series ───────────────────────────
def _load(fp):
    d = pd.read_csv(fp).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return d


D1 = _load(f"{C}/k1d/BTCUSDT.csv")
D1["e50"] = D1.c.ewm(span=50, adjust=False).mean(); D1["e200"] = D1.c.ewm(span=200, adjust=False).mean()
D1["qv30"] = D1.qvol.shift(1).rolling(30).mean()
H4 = _load(f"{C}/k4h/BTCUSDT.csv")
H4["e50"] = H4.c.ewm(span=50, adjust=False).mean(); H4["e200"] = H4.c.ewm(span=200, adjust=False).mean()
H1 = _load(f"{C}/btc_1h.csv")
H1["e20"] = H1.c.ewm(span=20, adjust=False).mean(); H1["sl"] = H1.e20.diff()
M5 = _load(f"{C}/btc_5m.csv")
_tr = np.maximum(M5.h - M5.l, np.maximum((M5.h - M5.c.shift()).abs(), (M5.l - M5.c.shift()).abs()))
M5["atrp"] = _tr.ewm(alpha=1 / 14, adjust=False).mean() / M5.c * 100
SC = pd.read_pickle(f"{S}/frm_scan.pkl").sort_values("t").reset_index(drop=True)


def _last_closed(df, period, t):
    """index of the last bar CLOSED at or before t (open_time + period ≤ t)."""
    return np.searchsorted(df.open_time.values, t - period, side="right") - 1


def raw_tags(t):
    """continuous + boolean macro readings at times t (ms array) — last CLOSED bar / last scan ≤ t."""
    t = np.asarray(t, dtype=np.int64)
    i = _last_closed(D1, DAY, t); j = _last_closed(H4, 4 * H, t); k = _last_closed(H1, H, t); m = _last_closed(M5, 5 * MIN, t)
    s = np.searchsorted(SC.t.values, t, side="right") - 1
    c = D1.c.values
    out = dict(
        D1_GOLDEN=D1.e50.values[i] > D1.e200.values[i],
        ABOVE_D200=c[i] > D1.e200.values[i],
        H4_GOLDEN=H4.e50.values[j] > H4.e200.values[j],
        H1_EMA20_UP=H1.sl.values[k] > 0,
        R1D=(c[i] / c[i - 1] - 1) * 100, R3D=(c[i] / c[i - 3] - 1) * 100, R7D=(c[i] / c[i - 7] - 1) * 100, R30D=(c[i] / c[i - 30] - 1) * 100,
        OFF30=SC.off30d.values[s], ADX5=SC.btc_adx.values[s], ADX5_RISING=SC.btc_adx.values[s] > SC.btc_adx_prev.values[s],
        ATR5=M5.atrp.values[m], EFF72_CHOP=SC.eff.values[s] <= 0.007, BULL=SC.bull.values[s],
        BULL_GT_BEAR=SC.bull.values[s] > SC.bear.values[s], BTC_RSI5=SC.btc_rsi.values[s], BTC_SLOPE5_UP=SC.btc_slope.values[s] > 0,
        ABOVE72=SC.above.values[s], VOLR=D1.qvol.values[i] / D1.qv30.values[i],
    )
    return pd.DataFrame(out)


BIN = ["D1_GOLDEN", "ABOVE_D200", "H4_GOLDEN", "H1_EMA20_UP", "ADX5_RISING", "EFF72_CHOP", "BULL_GT_BEAR", "BTC_SLOPE5_UP"]
SIGN = ["R1D", "R3D", "R7D", "R30D"]                      # split at 0
TERC = ["R7D", "R30D", "OFF30", "ADX5", "ATR5", "BULL", "BTC_RSI5", "ABOVE72", "VOLR"]   # T1 vs T3
LABEL = {"D1_GOLDEN": "BTC daily EMA50>EMA200", "ABOVE_D200": "BTC close > daily EMA200", "H4_GOLDEN": "BTC 4h EMA50>EMA200",
         "H1_EMA20_UP": "BTC 1h EMA20 rising", "ADX5_RISING": "BTC 5m ADX rising", "EFF72_CHOP": "72h efficiency ≤0.007 (chop)",
         "BULL_GT_BEAR": "breadth bull% > bear%", "BTC_SLOPE5_UP": "BTC 5m EMA20 slope > 0", "R1D": "BTC 1d return", "R3D": "BTC 3d return",
         "R7D": "BTC 7d return", "R30D": "BTC 30d return", "OFF30": "BTC distance below 30d high", "ADX5": "BTC 5m ADX",
         "ATR5": "BTC 5m ATR %", "BULL": "breadth bull %", "BTC_RSI5": "BTC 5m RSI", "ABOVE72": "% pairs above 72h ref (bull-run 'above')",
         "VOLR": "BTC daily quote-vol / 30d mean"}

GRID = np.arange(Y0, Y1, 5 * MIN)
_G = raw_tags(GRID)
CUTS = {v: tuple(np.nanquantile(_G[v].astype(float), [1 / 3, 2 / 3])) for v in TERC}


def splits(R):
    """→ {split_name: (labels array with 'A'/'B'/None, labelA, labelB)}; A/B are the two compared states."""
    out = {}
    for v in BIN:
        x = R[v].values
        out[f"{v}"] = (np.where(x, "A", "B"), "yes", "no")
    for v in SIGN:
        x = R[v].values.astype(float)
        out[f"{v}>0"] = (np.where(np.isnan(x), None, np.where(x > 0, "A", "B")), ">0", "≤0")
    for v in TERC:
        x = R[v].values.astype(float); a, b = CUTS[v]
        out[f"{v}_T"] = (np.where(np.isnan(x), None, np.where(x <= a, "B", np.where(x > b, "A", None))), f"T3 >{b:.3g}", f"T1 ≤{a:.3g}")
    return out


# ─────────────────────────── statistics ───────────────────────────
def poisson_w(nd, reps, rng):
    return rng.poisson(1.0, size=(reps, nd)).astype(float)


def cell_stats(pct, w, dix, lab, nd, halfmask, PW, full=True):
    """pct/w/dix (day index 0..nd-1) of one gate; lab = 'A'/'B'/None per signal; PW = Poisson bootstrap weights (reps x nd)."""
    res = {}
    sums = {}
    for st in ("A", "B"):
        m = lab == st
        n = int(m.sum())
        Sd = np.bincount(dix[m], weights=w[m] * pct[m], minlength=nd); Wd = np.bincount(dix[m], weights=w[m], minlength=nd)
        days = int((Wd > 0).sum())
        avg = Sd.sum() / Wd.sum() if Wd.sum() > 0 else np.nan
        res[st] = dict(n=n, days=days, avg=avg)
        sums[st] = (Sd, Wd)
        if full and n:
            res[st]["wr"] = float(np.average(pct[m] > 0, weights=w[m]))
            res[st]["w"] = float(w[m].sum())
            for hn, hm in (("h1", halfmask), ("h2", ~halfmask)):
                mm = m & hm
                dd = len(np.unique(dix[mm]))
                res[st][hn] = (np.average(pct[mm], weights=w[mm]) if mm.any() else np.nan, dd)
    # bootstrap (joint day resampling)
    with np.errstate(invalid="ignore", divide="ignore"):
        ba = (PW @ sums["A"][0]) / (PW @ sums["A"][1]); bb = (PW @ sums["B"][0]) / (PW @ sums["B"][1])
    res["ciA"] = np.nanquantile(ba, [0.025, 0.975]) if res["A"]["n"] else (np.nan, np.nan)
    res["ciB"] = np.nanquantile(bb, [0.025, 0.975]) if res["B"]["n"] else (np.nan, np.nan)
    res["ciD"] = np.nanquantile(ba - bb, [0.025, 0.975]) if res["A"]["n"] and res["B"]["n"] else (np.nan, np.nan)
    return res


def verdict(r, full=True):
    A, B = r["A"], r["B"]
    if min(A["n"], B["n"]) < 15:
        return "n<15", None
    if not (np.sign(A["avg"]) * np.sign(B["avg"]) < 0):
        return "same sign", None
    helps = "A" if A["avg"] < 0 else "B"
    if min(A["days"], B["days"]) < 8:
        return "days<8", helps
    if full:
        for hn in ("h1", "h2"):
            for st in ("A", "B"):
                v, dd = r[st][hn]
                if dd < 3 or not np.isfinite(v):
                    return f"half {hn} thin", helps
                if (st == helps and v >= 0) or (st != helps and v <= 0):
                    return f"half {hn} flips", helps
    lo, hi = r["ciD"]
    if not (lo > 0 or hi < 0):
        return "gap CI ∋ 0", helps
    strong = (r["ciA"][1] < 0 if helps == "A" else r["ciA"][0] > 0) and (r["ciB"][1] < 0 if helps == "B" else r["ciB"][0] > 0)
    return ("PASS-STRONG" if strong else "PASS"), helps


def gap_screen(r, pct, w, dix, lab, halfmask):
    """SECONDARY (state-gap) screen: N ≥ 15 and ≥ 8 days per state, gap CI excludes 0, and the gap has the same sign in BOTH halves
    (each state ≥ 3 days per half) — 'the filter is worth more in one state', whatever the signs."""
    A, B = r["A"], r["B"]
    if min(A["n"], B["n"]) < 15 or min(A["days"], B["days"]) < 8:
        return False
    lo, hi = r["ciD"]
    if not (lo > 0 or hi < 0):
        return False
    sg = np.sign(A["avg"] - B["avg"])
    for hm in (halfmask, ~halfmask):
        v = []
        for st in ("A", "B"):
            m = (lab == st) & hm
            if len(np.unique(dix[m])) < 3:
                return False
            v.append(np.average(pct[m], weights=w[m]))
        if np.sign(v[0] - v[1]) != sg:
            return False
    return True


def halves_ok(pct, w, dix, lab, halfmask, helps):
    for hm in (halfmask, ~halfmask):
        for st in ("A", "B"):
            m = (lab == st) & hm
            if len(np.unique(dix[m])) < 3:
                return False
            v = np.average(pct[m], weights=w[m])
            if (st == helps and v >= 0) or (st != helps and v <= 0):
                return False
    return True


# ─────────────────────────── main ───────────────────────────
def main():
    P = pd.read_csv("reports/FILTER_REGIME_MATRIX_signals_priced.csv")
    P = P[(P.t >= Y0) & (P.t < Y1)]
    cov = P.groupby("gate").agg(n=("pct", "size"), priced=("pct", lambda x: x.notna().mean()),
                                tick=("src", lambda x: (x == "tick").mean()), n_all=("n_signals_all", "first"), kind=("kind", "first"))
    P = P[P.pct.notna()].copy()
    P["w"] = P.n_seeds / 3.0
    P["day"] = (P.t // DAY).astype(np.int64)
    days_all = np.sort(P.day.unique()); dpos = {d: i for i, d in enumerate(days_all)}; nd = len(days_all)
    P["dix"] = P.day.map(dpos).values
    R = raw_tags(P.t.values); R.index = P.index
    SP = splits(R)
    halfmask_all = (P.t.values < HALF)
    PW = poisson_w(nd, 2000, RNG)
    rows = []
    gates = [g for g in P.gate.unique()]
    G = {g: np.flatnonzero(P.gate.values == g) for g in gates}
    pct_all, w_all, dix_all = P.pct.values, P.w.values, P.dix.values
    for g in gates:
        ix = G[g]
        for sn, (lab, la, lb) in SP.items():
            r = cell_stats(pct_all[ix], w_all[ix], dix_all[ix], lab[ix], nd, halfmask_all[ix], PW)
            v, helps = verdict(r)
            rows.append(dict(gate=g, split=sn, stateA=la, stateB=lb, nA=r["A"]["n"], daysA=r["A"]["days"], avgA=r["A"]["avg"],
                             wrA=r["A"].get("wr"), ciA_lo=r["ciA"][0], ciA_hi=r["ciA"][1],
                             h1A=r["A"].get("h1", (np.nan, 0))[0], h1A_days=r["A"].get("h1", (np.nan, 0))[1],
                             h2A=r["A"].get("h2", (np.nan, 0))[0], h2A_days=r["A"].get("h2", (np.nan, 0))[1],
                             nB=r["B"]["n"], daysB=r["B"]["days"], avgB=r["B"]["avg"], wrB=r["B"].get("wr"), ciB_lo=r["ciB"][0], ciB_hi=r["ciB"][1],
                             h1B=r["B"].get("h1", (np.nan, 0))[0], h1B_days=r["B"].get("h1", (np.nan, 0))[1],
                             h2B=r["B"].get("h2", (np.nan, 0))[0], h2B_days=r["B"].get("h2", (np.nan, 0))[1],
                             gap=(r["A"]["avg"] - r["B"]["avg"]), gap_ci_lo=r["ciD"][0], gap_ci_hi=r["ciD"][1],
                             helps_in=(la if helps == "A" else lb) if helps else None, verdict=v,
                             gap_screen=gap_screen(r, pct_all[ix], w_all[ix], dix_all[ix], lab[ix], halfmask_all[ix])))
    MX = pd.DataFrame(rows)
    MX.to_csv(OUTCSV, index=False)
    obs_pass = int(MX.verdict.str.startswith("PASS").sum()); obs_strong = int((MX.verdict == "PASS-STRONG").sum())
    print("observed PASS", obs_pass, "STRONG", obs_strong, flush=True)

    # ── shuffled null: permute day → day within month, same time of day ──
    NR = int(os.environ.get("NULL_ROUNDS", 200))
    month = pd.to_datetime(days_all * DAY, unit="ms").month.values
    PWn = poisson_w(nd, 300, RNG)
    nulls = []
    t0 = time.time()
    for k in range(NR):
        perm = np.arange(nd)
        for mo in np.unique(month):
            idx = np.flatnonzero(month == mo)
            perm[idx] = RNG.permutation(idx)
        shift = (days_all[perm] - days_all) * DAY          # per day index
        Rn = raw_tags(P.t.values + shift[dix_all]); SPn = splits(Rn)
        npass = 0; nstrong = 0; ngap = 0
        for g in gates:
            ix = G[g]
            for sn, (lab, la, lb) in SPn.items():
                r = cell_stats(pct_all[ix], w_all[ix], dix_all[ix], lab[ix], nd, halfmask_all[ix], PWn, full=False)
                v, helps = verdict(r, full=False)
                ngap += gap_screen(r, pct_all[ix], w_all[ix], dix_all[ix], lab[ix], halfmask_all[ix])
                if v.startswith("PASS") and halves_ok(pct_all[ix], w_all[ix], dix_all[ix], lab[ix], halfmask_all[ix], helps):
                    npass += 1; nstrong += v == "PASS-STRONG"
        nulls.append((k, npass, nstrong, ngap))
        if k % 20 == 0:
            print("null", k, npass, nstrong, f"{time.time() - t0:.0f}s", flush=True)
    N = pd.DataFrame(nulls, columns=["round", "n_pass", "n_strong", "n_gap"]); N.to_csv(OUTCSV.replace(".csv", "_null.csv"), index=False)
    cov.to_csv(OUTCSV.replace(".csv", "_coverage.csv"))
    json.dump(dict(obs_pass=obs_pass, obs_strong=obs_strong, null_mean=float(N.n_pass.mean()), null_p95=float(N.n_pass.quantile(.95)),
                   null_ge_obs=float((N.n_pass >= obs_pass).mean()), null_strong_mean=float(N.n_strong.mean()),
                   null_strong_ge_obs=float((N.n_strong >= obs_strong).mean()),
                   obs_gap=int(MX.gap_screen.sum()), null_gap_mean=float(N.n_gap.mean()), null_gap_p95=float(N.n_gap.quantile(.95)),
                   null_gap_ge_obs=float((N.n_gap >= MX.gap_screen.sum()).mean()), rounds=NR, cuts={k: list(v) for k, v in CUTS.items()},
                   n_cells=int(len(MX)), n_cells_testable=int((~MX.verdict.isin(["n<15"])).sum()), days=int(nd)),
              open(OUTCSV.replace(".csv", "_summary.json"), "w"), indent=1)
    print(open(OUTCSV.replace(".csv", "_summary.json")).read())


if __name__ == "__main__":
    main()
