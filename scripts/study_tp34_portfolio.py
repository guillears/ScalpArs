#!/usr/bin/env python3
"""💼 Oct-8 operator study "5-sleeve portfolio" — ONE shared compounding book, $3,000 on 2026-01-04 → 2026-10-04, only BULLRUN_LONG,
BEARRUN_SHORT, FRENZY_LONG, FRENZY_WIDE, FRENZY_LITE active.

Fills (each sleeve's own source, never re-derived here):
  BULLRUN / BEARRUN    yr5 engine replay, kept fills of study_yr5_halves_today.build_replay() (per seed; replay pct = the replay's own exit)
  FRENZY_LONG / WIDE   the same replay fills (per seed) re-priced on TICKS at the live exit +3/−3/12 h (study_tp34_walk, live ruler:
                       first print ≥ signal close + 8 s, +0.10 % entry slip, taker 0.045 % each side) — entry/exit times from the tick walk
  FRENZY_LITE          the 724-fill study cohort, bearish-day block, same tick walk (single cohort; identical in every seed run)
Sizing (services/trading_engine sizing, trading_config.json today): equal split = (equity − schedule reserve − fee reserve) / max_open_positions
(4); schedule reserve from reserve_schedule (none below $10k; above: equity − tier target); fee reserve = max($15, 2.5 % × min(equity, tier
target)); investment ≤ tradeable = free balance − reserves; × invest mult (1 for all five); leverage = max(1, round(20 × lev mult)) capped by
leverage_balance_schedule (20 below $25k, 15 above): FRENZY_LONG 6× (10× when entry ADX Δ > 0 ∧ DI spread > 0), WIDE 4×, LITE 6×,
BEARRUN 5×, BULLRUN 20×. P&L $ = investment × leverage × net % / 100 (net % already carries the fees). Equity = realized book (marked at
closes; open P&L not marked). Capacity: global 4 open positions shared; per-sleeve caps FRENZY_LONG 2 · WIDE 2 · LITE 2 · BULLRUN 4 ·
BEARRUN (global only); one position per pair; ≤ 3 entries per pair per UTC day per FRENZY sleeve. Fills processed in entry-time order;
a fill that cannot open is skipped (counted per sleeve and reason). Not modelled: liquidation (paper), funding, the WILLY global hold (WILLY
is not in this book), exchange min-notional, slippage beyond the entry 0.10 % (FRENZY family) / the replay's own fills (BULL/BEAR).
Usage: venv/bin/python scripts/study_tp34_portfolio.py"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
OUT = os.path.join(SCR, "tp34")
T0, TS, T1 = pd.Timestamp("2026-01-04"), pd.Timestamp("2026-05-20"), pd.Timestamp("2026-10-04")
CFG = json.load(open(os.path.join(ROOT, "trading_config.json")))
INV = CFG["investment"]; TH = CFG.get("thresholds", CFG)
MAXPOS = int(INV["max_open_positions"])
CAPS = {"FRENZY_LONG": int(TH["frenzy_max_slots"]), "FRENZY_WIDE": int(TH["frenzy_wide_max_slots"]), "FRENZY_LITE": int(TH["frenzy_lite_max_slots"]),
        "BULLRUN": int(TH["bullrun_max_slots"]), "BEARRUN": 99}
DAYCAP = int(TH["frenzy_max_entries_per_pair_day"])


def sched(s):
    return sorted((float(a), float(b)) for a, b in (p.split(":") for p in s.split(",") if p.strip()))


RES, LEVS = sched(INV["reserve_schedule"]), sched(INV["leverage_balance_schedule"])


def tier(tab, x):
    v = None
    for a, b in tab:
        if x >= a:
            v = b
    return v


def size(equity, margin_used):
    tgt = tier(RES, equity)
    sres = max(0.0, equity - tgt) if tgt else 0.0
    fee_eq = min(equity, tgt) if tgt else equity
    fres = max(float(INV["fee_reserve_usd"]), fee_eq * float(INV["fee_reserve_pct"]) / 100)
    inv = max(0.0, equity - sres - fres) / MAXPOS
    tradeable = max(0.0, equity - margin_used - sres - fres)
    return min(inv, tradeable), tier(LEVS, equity)


def fills(seed, F, A, Cc):
    rows = []
    br = F[(F.seed == seed) & F.keep & F.S.isin(["BULLRUN", "BEARRUN"])]
    for r in br.itertuples():
        rows.append(dict(S=r.S, pair=r.pair, t_in=int(r.tms), t_out=int(pd.Timestamp(str(r.closed_at)[:23]).value // 1_000_000), pct=float(r.pct),
                         lev=20.0 if r.S == "BULLRUN" else max(1, round(20 * float(TH["bearrun_lev_mult"])))))
    a = A[(A.seed == seed) & (A.st == "ok")]
    for r in a.itertuples():
        lev = 4 if r.sleeve == "FRENZY_WIDE" else (10 if r.strong else 6)
        rows.append(dict(S=r.sleeve, pair=r.pair, t_in=int(r.entry_ms), t_out=int(r.fix3_xms), pct=float(r.fix3), lev=float(lev)))
    for r in Cc.itertuples():
        rows.append(dict(S="FRENZY_LITE", pair=r.pair, t_in=int(r.entry_ms), t_out=int(r.fix3_xms), pct=float(r.fix3), lev=6.0))
    X = pd.DataFrame(rows)
    X = X[(X.t_in >= T0.value // 1_000_000) & (X.t_in < T1.value // 1_000_000)]
    return X.sort_values(["t_in", "S"], kind="stable").reset_index(drop=True)


def run(X, start=3000.0):
    eq = start; openp = []; skipped = {}; daycnt = {}; taken = []; curve = [(T0.value // 1_000_000, eq)]
    def close_until(t):
        nonlocal eq
        openp.sort(key=lambda p: p["t_out"])
        while openp and openp[0]["t_out"] <= t:
            p = openp.pop(0); eq += p["usd"]; curve.append((p["t_out"], eq))
    for r in X.itertuples():
        close_until(r.t_in)
        why = None
        if len(openp) >= MAXPOS:
            why = "global_slots"
        elif sum(p["S"] == r.S for p in openp) >= CAPS[r.S]:
            why = "sleeve_slots"
        elif any(p["pair"] == r.pair for p in openp):
            why = "pair_held"
        elif r.S.startswith("FRENZY") and daycnt.get((r.S, r.pair, r.t_in // 86_400_000), 0) >= DAYCAP:
            why = "pair_day_cap"
        if why is None:
            inv, levcap = size(eq, sum(p["inv"] for p in openp))
            if inv < 5.0 or eq <= 0:
                why = "no_balance"
        if why:
            skipped[(r.S, why)] = skipped.get((r.S, why), 0) + 1; continue
        lev = min(r.lev, levcap) if levcap else r.lev
        usd = inv * lev * r.pct / 100
        p = dict(S=r.S, pair=r.pair, t_in=r.t_in, t_out=r.t_out, inv=inv, usd=usd, pct=r.pct, eq_in=eq)
        openp.append(p); taken.append(p)
        if r.S.startswith("FRENZY"):
            k = (r.S, r.pair, r.t_in // 86_400_000); daycnt[k] = daycnt.get(k, 0) + 1
    close_until(2**62)
    T = pd.DataFrame(taken); Cv = pd.DataFrame(curve, columns=["t", "eq"]).sort_values("t", kind="stable")
    return eq, T, Cv, skipped


def metrics(eq, T, Cv, start=3000.0):
    Cv = Cv.copy(); Cv["d"] = pd.to_datetime(Cv.t, unit="ms").dt.floor("D")
    peak = Cv["eq"].cummax(); mdd = ((Cv["eq"] / peak - 1) * 100).min()
    de = Cv.groupby("d")["eq"].last()
    days = pd.date_range(T0, T1 - pd.Timedelta(days=1), freq="D")
    de = de.reindex(days).ffill().fillna(start)
    prev = de.shift(1).fillna(start); dr = (de / prev - 1) * 100
    nd = len(days); tdays = pd.to_datetime(T.t_in, unit="ms").dt.floor("D").nunique() if len(T) else 0
    g = eq / start
    out = dict(end=eq, ret=(g - 1) * 100, daily_cal=(g ** (1 / nd) - 1) * 100 if g > 0 else np.nan,
               daily_trade=(g ** (1 / max(tdays, 1)) - 1) * 100 if g > 0 else np.nan, trade_days=tdays, mdd=mdd, worst_day=dr.min(),
               worst_day_date=str(dr.idxmin().date()))
    h1 = de[de.index < TS]; out["H1_ret"] = (h1.iloc[-1] / start - 1) * 100; out["H2_ret"] = (de.iloc[-1] / h1.iloc[-1] - 1) * 100
    out["H1_usd"] = h1.iloc[-1] - start; out["H2_usd"] = de.iloc[-1] - h1.iloc[-1]
    return out


def main():
    import study_yr5_halves_today as YH
    F = YH.build_replay(False)
    A = pd.read_pickle(os.path.join(OUT, "cohort_a.pkl")); W = pd.read_pickle(os.path.join(OUT, "walk_a.pkl"))
    A = A.assign(key=A.pair + "|" + A.sig.astype("int64").astype(str)).merge(W, on="key", how="left")
    st = F[F.S.isin(["FRENZY_LONG", "FRENZY_WIDE"]) & F.keep][["seed", "pair", "tms", "entry_frenzy_adx_delta", "entry_frenzy_di_spread"]]
    st = st.assign(strong=(pd.to_numeric(st.entry_frenzy_adx_delta, errors="coerce") > 0) & (pd.to_numeric(st.entry_frenzy_di_spread, errors="coerce") > 0))
    A = A.merge(st[["seed", "pair", "tms", "strong"]].rename(columns={"tms": "rep_open"}), on=["seed", "pair", "rep_open"], how="left")
    A["strong"] = A.strong.fillna(False).astype(bool)
    Cc = pd.read_pickle(os.path.join(OUT, "cohort_c.pkl")); Wc = pd.read_pickle(os.path.join(OUT, "walk_c.pkl"))
    Cc = Cc.assign(key=Cc.pair + "|" + Cc.sig.astype("int64").astype(str)).merge(Wc, on="key", how="left")
    Cc = Cc[(Cc.st == "ok") & ~Cc.bear]
    scen = {"ALL 5": None, "without BULLRUN": "BULLRUN", "without LITE": "FRENZY_LITE", "without BULLRUN and LITE": ("BULLRUN", "FRENZY_LITE")}
    res = {}
    for name, drop in scen.items():
        per = []
        for sd in sorted(F.seed.unique()):
            X = fills(sd, F, A, Cc)
            if drop:
                X = X[~X.S.isin([drop] if isinstance(drop, str) else list(drop))]
            eq, T, Cv, sk = run(X)
            m = metrics(eq, T, Cv)
            T["t"] = pd.to_datetime(T.t_in, unit="ms")
            m["sleeves"] = {s: dict(N=len(x), WR=(x.pct > 0).mean() * 100, avg=x.pct.mean(), usd=x.usd.sum(),
                                    H1_usd=x[x.t < TS].usd.sum(), H2_usd=x[x.t >= TS].usd.sum()) for s, x in T.groupby("S")}
            m["skipped"] = {f"{a}|{b}": v for (a, b), v in sk.items()}
            m["offered"] = X.S.value_counts().to_dict()
            per.append(m)
            if name == "ALL 5":
                T.to_csv(os.path.join(OUT, f"portfolio_trades_seed{sd}.csv"), index=False)
        res[name] = per
    json.dump(res, open(os.path.join(OUT, "portfolio.json"), "w"), indent=1, default=float)
    for name, per in res.items():
        k = lambda f: (np.mean([p[f] for p in per]), min(p[f] for p in per), max(p[f] for p in per))
        print(f"\n== {name}")
        for f in ("end", "ret", "daily_cal", "daily_trade", "mdd", "worst_day", "H1_ret", "H2_ret", "H1_usd", "H2_usd"):
            m, lo, hi = k(f); print(f"  {f:<12} mean {m:12,.3f}  range [{lo:,.3f} … {hi:,.3f}]")
        print("  worst day dates", [p["worst_day_date"] for p in per], "trade days", [p["trade_days"] for p in per])
        for s in sorted({s for p in per for s in p["sleeves"]}):
            v = [p["sleeves"].get(s) for p in per if s in p["sleeves"]]
            print(f"  {s:<12} N {np.mean([x['N'] for x in v]):6.1f} WR {np.mean([x['WR'] for x in v]):5.1f} avg {np.mean([x['avg'] for x in v]):+.3f} "
                  f"$ {np.mean([x['usd'] for x in v]):+12,.0f} [{min(x['usd'] for x in v):+,.0f} … {max(x['usd'] for x in v):+,.0f}] H1 $ {np.mean([x['H1_usd'] for x in v]):+,.0f} H2 $ {np.mean([x['H2_usd'] for x in v]):+,.0f}")
        sk = {}
        for p in per:
            for kk, v in p["skipped"].items():
                sk[kk] = sk.get(kk, 0) + v / len(per)
        print("  skipped/seed", {k_: round(v, 1) for k_, v in sorted(sk.items())})
        print("  offered/seed", {s: round(np.mean([p["offered"].get(s, 0) for p in per]), 1) for s in per[0]["offered"]})


if __name__ == "__main__":
    main()
