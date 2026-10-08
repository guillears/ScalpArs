#!/usr/bin/env python3
"""💼 Oct-8 operator study "7-sleeve portfolio" — the 5-sleeve shared compounding book of scripts/study_tp34_portfolio.py (REUSED: its
fills(), size schedules, metrics(), caps) PLUS MOM_LONG and SPIKE_FADE. ONE shared $3,000 book, 2026-01-04 → 2026-10-04, compounding,
today's live sizing (trading_config.json), global max_open_positions 4 shared, per-sleeve slot caps, one position per pair, ≤ 3 FRENZY
entries per pair-day, capacity skips counted, 3 seeds (mean + range).

Added sleeves (read-only sources):
  MOM_LONG    yr5 replay kept fills of study_yr5_halves_today.build_replay() — today's LONG_HEAT 3-leg + LONG_CHOP_BURST filters, today's
              cell sizing (build_master_pool.today_size_rule) split into invest mult × leverage mult (re-priced cells run lev mult 1×;
              unchanged cells keep cell_lev_multiplier). FIX-A (study_fade_h1_fixed_halves.py) is applied DYNAMICALLY: the replay's own
              fill ratio is NOT used; the engine's 0.1 % × 24 h volume notional cap (ceiling $500k) is applied at the book's size at entry
              (a throttle below the $100 min investment = skip, like the engine).
              + HEAT re-admits (HEAT_YR5_SIGNALS_priced.csv, the signals today's 3-leg rule admits again, labelled MOM_LONG_READMIT): the
              file only gives n_seeds (1–3), so seed k (0,1,2) takes the signals with n_seeds > k (expected count = the halves study's
              weight); exit time = entry + the minutes in `how`; size = the mean ML today multiplier at 20×; no 24 h volume → no liq cap.
  SPIKE_FADE  yr5 replay kept fades; invest mult 2 × 20× (lev mult 1); ticket = min(desired notional, 0.5 % × 24 h volume, $500k) at the
              book's size. Three variants (no trades invented):
                F0 replay pnl % as-is
                F1 every fade's pnl % + Δ, Δ = +0.04 − (mean replay pnl % of all offered fades, 3 seeds pooled, in window) → calibrated
                F2 same uniform shift to the stress mean −0.05
MOM_SHORT / FLIP / SURGE / WILLY are OFF. BULLRUN / BEARRUN / FRENZY_* priced exactly as study_tp34_portfolio (no liq cap there — parity).
Writes reports/PORTFOLIO_7SLEEVE_2026-10-08.csv (+ _sleeves.csv, _skips.csv) and prints the tables.
Usage: venv/bin/python scripts/study_port7_portfolio.py"""
import json, os, re, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import study_tp34_portfolio as P                                  # noqa: E402  (read-only reuse)
import study_yr5_halves_today as YH                               # noqa: E402

REP = os.path.join(ROOT, "reports")
T0, TS, T1 = P.T0, P.TS, P.T1
INV = P.INV
MIN_INV = 100.0                                                   # config.py min_investment_size default (not overridden in json)
LIQ_ML = float(INV["max_notional_pct_of_pair_volume"]) / 100      # 0.1 %
LIQ_FADE = float(INV["spike_lowvol_liq_cap_pct"]) / 100           # 0.5 % (threshold 1e12 → every fade)
CEIL = float(INV["max_notional_hard_ceiling"])
CAPS = dict(P.CAPS, MOM_LONG=99, MOM_LONG_READMIT=99, SPIKE_FADE=99)
SLOTKEY = {"MOM_LONG_READMIT": "MOM_LONG"}                        # readmits share the MOM_LONG identity
FADE_TARGET = {"F0": None, "F1": 0.04, "F2": -0.05}
ms = lambda s: pd.to_datetime(pd.Series(s).astype(str).str[:23].str.replace("T", " "), format="mixed").values.astype("datetime64[ms]").astype("int64")


def load():
    F = YH.build_replay(False)
    # FRENZY cohorts exactly as study_tp34_portfolio.main
    A = pd.read_pickle(os.path.join(P.OUT, "cohort_a.pkl")); W = pd.read_pickle(os.path.join(P.OUT, "walk_a.pkl"))
    A = A.assign(key=A.pair + "|" + A.sig.astype("int64").astype(str)).merge(W, on="key", how="left")
    st = F[F.S.isin(["FRENZY_LONG", "FRENZY_WIDE"]) & F.keep][["seed", "pair", "tms", "entry_frenzy_adx_delta", "entry_frenzy_di_spread"]]
    st = st.assign(strong=(pd.to_numeric(st.entry_frenzy_adx_delta, errors="coerce") > 0) & (pd.to_numeric(st.entry_frenzy_di_spread, errors="coerce") > 0))
    A = A.merge(st[["seed", "pair", "tms", "strong"]].rename(columns={"tms": "rep_open"}), on=["seed", "pair", "rep_open"], how="left")
    A["strong"] = A.strong.fillna(False).astype(bool)
    Cc = pd.read_pickle(os.path.join(P.OUT, "cohort_c.pkl")); Wc = pd.read_pickle(os.path.join(P.OUT, "walk_c.pkl"))
    Cc = Cc.assign(key=Cc.pair + "|" + Cc.sig.astype("int64").astype(str)).merge(Wc, on="key", how="left")
    Cc = Cc[(Cc.st == "ok") & ~Cc.bear]
    # MOM_LONG
    K = F[F.keep & (F.S == "MOM_LONG")].copy()
    cm = pd.to_numeric(K.cell_multiplier, errors="coerce").fillna(1.0); clm = pd.to_numeric(K.cell_lev_multiplier, errors="coerce").fillna(1.0)
    same = np.isclose(K.size_factor, 1.0)
    K["lev_mult"] = np.where(same, clm, 1.0)
    K["inv_mult"] = np.where(same, cm, K.inv_today)              # inv_today = cm × clm × factor; re-priced → lev 1×
    ml_inv_mean = float(F[F.keep & (F.S == "MOM_LONG")].inv_today.mean())
    Hr, _, _ = YH.heat_readmits(ml_inv_mean)
    Hr["mins"] = Hr.how.str.extract(r"(\d+)m\s*$")[0].astype(float)
    # SPIKE_FADE
    D = F[F.keep & (F.S == "SPIKE_FADE")].copy()
    return F, A, Cc, K, Hr, D, ml_inv_mean


def fills7(seed, k_seed, F, A, Cc, K, Hr, D, fade_shift, ml_shift=0.0):
    X = P.fills(seed, F, A, Cc)
    X = X.assign(mult=1.0, liq=np.nan, vol=np.nan)
    rows = []
    k = K[K.seed == seed]
    for r, t_out in zip(k.itertuples(), ms(k.closed_at)):
        rows.append(dict(S="MOM_LONG", pair=r.pair, t_in=int(r.tms), t_out=int(t_out), pct=float(r.pct_today) + ml_shift,
                         lev=float(max(1, round(20 * r.lev_mult))), mult=float(r.inv_mult), liq=LIQ_ML,
                         vol=float(pd.to_numeric(r.entry_pair_volume_24h_usd, errors="coerce"))))
    h = Hr[Hr.n_seeds > k_seed]
    for r in h.itertuples():
        rows.append(dict(S="MOM_LONG_READMIT", pair=r.pair, t_in=int(r.t.value // 1_000_000), t_out=int(r.t.value // 1_000_000 + r.mins * 60_000),
                         pct=float(r.pct) + ml_shift, lev=20.0, mult=float(r.mult), liq=np.nan, vol=np.nan))
    d = D[D.seed == seed]
    if fade_shift is not None:
        for r, t_out in zip(d.itertuples(), ms(d.closed_at)):
            rows.append(dict(S="SPIKE_FADE", pair=r.pair, t_in=int(r.tms), t_out=int(t_out), pct=float(r.pct) + fade_shift, lev=20.0,
                             mult=float(pd.to_numeric(r.cell_multiplier, errors="coerce") or 2.0), liq=LIQ_FADE,
                             vol=float(pd.to_numeric(r.entry_pair_volume_24h_usd, errors="coerce"))))
    Y = pd.DataFrame(rows)
    if len(Y):
        Y = Y[(Y.t_in >= T0.value // 1_000_000) & (Y.t_in < T1.value // 1_000_000)]
    X = pd.concat([X, Y], ignore_index=True)
    return X.sort_values(["t_in", "S"], kind="stable").reset_index(drop=True)


def run(X, start=3000.0):
    """study_tp34_portfolio.run + invest multiplier + engine liquidity cap (only rows with liq set). Identical for the 5 original sleeves."""
    eq = start; openp = []; skipped = {}; daycnt = {}; taken = []; curve = [(T0.value // 1_000_000, eq)]

    def close_until(t):
        nonlocal eq
        openp.sort(key=lambda p: p["t_out"])
        while openp and openp[0]["t_out"] <= t:
            p = openp.pop(0); eq += p["usd"]; curve.append((p["t_out"], eq))
    for r in X.itertuples():
        close_until(r.t_in)
        sk = SLOTKEY.get(r.S, r.S)
        why = None
        if len(openp) >= P.MAXPOS:
            why = "global_slots"
        elif sum(SLOTKEY.get(p["S"], p["S"]) == sk for p in openp) >= CAPS[r.S]:
            why = "sleeve_slots"
        elif any(p["pair"] == r.pair for p in openp):
            why = "pair_held"
        elif r.S.startswith("FRENZY") and daycnt.get((r.S, r.pair, r.t_in // 86_400_000), 0) >= P.DAYCAP:
            why = "pair_day_cap"
        if why is None:
            margin = sum(p["inv"] for p in openp)
            split, levcap = P.size(eq, 0.0)                      # equal split (uncapped by free balance)
            tradeable_inv, _ = P.size(eq, margin)                 # = min(split, tradeable)
            if r.mult == 1.0:
                inv = tradeable_inv
            else:
                tgt = P.tier(P.RES, eq); sres = max(0.0, eq - tgt) if tgt else 0.0
                fee_eq = min(eq, tgt) if tgt else eq
                fres = max(float(INV["fee_reserve_usd"]), fee_eq * float(INV["fee_reserve_pct"]) / 100)
                inv = min(split * r.mult, max(0.0, eq - margin - sres - fres))
            if inv < 5.0 or eq <= 0:
                why = "no_balance"
        if why is None:
            lev = min(r.lev, levcap) if levcap else r.lev
            if not np.isnan(r.liq):
                cap = CEIL if not (r.vol > 0) else min(r.liq * r.vol, CEIL)
                if inv * lev > cap:
                    inv = cap / lev
                    if inv < MIN_INV:
                        why = "liq_cap_below_min"
        if why:
            skipped[(r.S, why)] = skipped.get((r.S, why), 0) + 1; continue
        usd = inv * lev * r.pct / 100
        p = dict(S=r.S, pair=r.pair, t_in=r.t_in, t_out=r.t_out, inv=inv, usd=usd, pct=r.pct, eq_in=eq, lev=lev)
        openp.append(p); taken.append(p)
        if r.S.startswith("FRENZY"):
            kk = (r.S, r.pair, r.t_in // 86_400_000); daycnt[kk] = daycnt.get(kk, 0) + 1
    close_until(2**62)
    T = pd.DataFrame(taken); Cv = pd.DataFrame(curve, columns=["t", "eq"]).sort_values("t", kind="stable")
    return eq, T, Cv, skipped


def main():
    F, A, Cc, K, Hr, D, ml_inv_mean = load()
    Hr["mult"] = ml_inv_mean
    seeds = sorted(F.seed.unique())
    win = (D.t >= T0) & (D.t < T1)
    fade_mean = float(D[win].pct.mean())
    shifts = {v: (0.0 if t is None else t - fade_mean) for v, t in FADE_TARGET.items()}
    print(f"fade replay mean (offered, 3 seeds pooled, N {win.sum()}): {fade_mean:+.4f} → shifts {shifts}")
    print(f"ML inv mean today {ml_inv_mean:.3f}; readmits {len(Hr)} signals, per-seed counts", [int((Hr.n_seeds > k).sum()) for k in range(3)])
    ALL = ["BULLRUN", "BEARRUN", "FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "MOM_LONG", "MOM_LONG_READMIT", "SPIKE_FADE"]
    scen = {"ALL 7": [], "without BULLRUN": ["BULLRUN"], "without LITE": ["FRENZY_LITE"],
            "without MOM_LONG": ["MOM_LONG", "MOM_LONG_READMIT"], "without SPIKE_FADE": ["SPIKE_FADE"],
            "5-sleeve (without MOM_LONG and SPIKE_FADE)": ["MOM_LONG", "MOM_LONG_READMIT", "SPIKE_FADE"],
            "ALL 7 without heat re-admits": ["MOM_LONG_READMIT"]}
    out, srows, krows = [], [], []
    for var, sh in shifts.items():
        for name, drop in scen.items():
            if var != "F0" and "SPIKE_FADE" in drop:
                continue                                         # fade-free rows are variant-independent → computed once under F0
            for k, sd in enumerate(seeds):
                X = fills7(sd, k, F, A, Cc, K, Hr, D, sh)
                X = X[~X.S.isin(drop)]
                eq, T, Cv, sk = run(X)
                m = P.metrics(eq, T, Cv)
                T["t"] = pd.to_datetime(T.t_in, unit="ms")
                vlab = "—" if "SPIKE_FADE" in drop else var
                out.append(dict(variant=vlab, scenario=name, seed=sd, **{f: m[f] for f in m}))
                for s in ALL:
                    x = T[T.S == s] if len(T) else T
                    off = int((X.S == s).sum())
                    if off == 0:
                        continue
                    srows.append(dict(variant=vlab, scenario=name, seed=sd, sleeve=s, offered=off, N=len(x),
                                      WR=(x.pct > 0).mean() * 100 if len(x) else np.nan, avg=x.pct.mean() if len(x) else np.nan,
                                      usd=x.usd.sum() if len(x) else 0.0, H1_usd=x[x.t < TS].usd.sum() if len(x) else 0.0,
                                      H2_usd=x[x.t >= TS].usd.sum() if len(x) else 0.0))
                for (s, why), v in sk.items():
                    krows.append(dict(variant=vlab, scenario=name, seed=sd, sleeve=s, reason=why, n=v))
                if name == "ALL 7":
                    T.to_csv(os.path.join(P.OUT, f"port7_trades_{var}_seed{sd}.csv"), index=False)
    # SENSITIVITY (not evidence): MOM_LONG pnl % shifted uniformly (replay + re-admits, same Δ) so the replay ML mean (3 seeds pooled, in
    # window) equals a target — answers "what per-trade ML result would this book need"; fades at F1.
    kw = (K.t >= T0) & (K.t < T1); ml_mean = float(K[kw].pct_today.mean())
    print(f"ML replay mean (offered, pooled, N {kw.sum()}): {ml_mean:+.4f}")
    for tgt in (0.0, 0.05, 0.10, 0.20):
        name = f"SENS ML mean set to {tgt:+.2f}"
        for k, sd in enumerate(seeds):
            X = fills7(sd, k, F, A, Cc, K, Hr, D, shifts["F1"], tgt - ml_mean)
            eq, T, Cv, sk = run(X)
            m = P.metrics(eq, T, Cv)
            out.append(dict(variant="F1", scenario=name, seed=sd, **{f: m[f] for f in m}))
            T["t"] = pd.to_datetime(T.t_in, unit="ms")
            for s in ALL:
                x = T[T.S == s]
                srows.append(dict(variant="F1", scenario=name, seed=sd, sleeve=s, offered=int((X.S == s).sum()), N=len(x),
                                  WR=(x.pct > 0).mean() * 100 if len(x) else np.nan, avg=x.pct.mean() if len(x) else np.nan,
                                  usd=x.usd.sum(), H1_usd=x[x.t < TS].usd.sum(), H2_usd=x[x.t >= TS].usd.sum()))
            for (s, why), v in sk.items():
                krows.append(dict(variant="F1", scenario=name, seed=sd, sleeve=s, reason=why, n=v))
    O = pd.DataFrame(out); S = pd.DataFrame(srows); Kk = pd.DataFrame(krows)
    base = os.path.join(REP, "PORTFOLIO_7SLEEVE_2026-10-08")
    O.to_csv(base + ".csv", index=False); S.to_csv(base + "_sleeves.csv", index=False); Kk.to_csv(base + "_skips.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    g = O.groupby(["variant", "scenario"], sort=False)
    agg = g.agg(end=("end", "mean"), end_lo=("end", "min"), end_hi=("end", "max"), ret=("ret", "mean"), ret_lo=("ret", "min"), ret_hi=("ret", "max"),
                dcal=("daily_cal", "mean"), dcal_lo=("daily_cal", "min"), dcal_hi=("daily_cal", "max"), dtr=("daily_trade", "mean"),
                tdays=("trade_days", "mean"), mdd=("mdd", "mean"), mdd_lo=("mdd", "min"), mdd_hi=("mdd", "max"), worst=("worst_day", "mean"),
                worst_lo=("worst_day", "min"), H1=("H1_ret", "mean"), H2=("H2_ret", "mean"), H2_lo=("H2_ret", "min"), H2_hi=("H2_ret", "max"))
    print(agg.round(2).to_string())
    print(O.groupby(["variant", "scenario"], sort=False).worst_day_date.agg(list).to_string())
    sg = S.groupby(["variant", "scenario", "sleeve"], sort=False).agg(offered=("offered", "mean"), N=("N", "mean"), WR=("WR", "mean"), avg=("avg", "mean"),
                                                                       usd=("usd", "mean"), usd_lo=("usd", "min"), usd_hi=("usd", "max"),
                                                                       H1=("H1_usd", "mean"), H2=("H2_usd", "mean"))
    print(sg.round(3).to_string())
    kg = Kk.groupby(["variant", "scenario", "sleeve", "reason"], sort=False).n.sum().div(len(seeds))
    print(kg.round(1).to_string())


if __name__ == "__main__":
    main()
