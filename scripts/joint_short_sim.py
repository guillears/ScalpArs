#!/usr/bin/env python3
"""JOINT SIM (2026-11-18, research only): every short sleeve now running, on ONE account, at today's size (0.5 × 5× = 0.609 × equity).

Sleeves (trade files, nothing rebuilt):
  EXH   reports/BASELINE_TRADES_EXHAUSTION_2026-11-18.csv, rows with dl582_block False (FRENZY universe, live exit L2)
  FB    reports/BASELINE_TRADES_FLATBASE_2026-11-18.csv (NEAR_BASE, 10 % stop / 5 % TP / 72 h)
  L4    reports/BASELINE_TRADES_EXH_WIDE_L4_2026-11-18.csv.gz: set C ∧ key not in BASE (ADDED) ∧ not DL 582-blocked, EX_* = live exit
  DIST  reports/DIST_WL_2026-11-18_data/rows.pkl (full row frame behind DIST_WL_2026-11-18_trades.csv.gz): keep RSI1h ≥ 74.07, LIVE exit
  WILLY reports/BASELINE_TRADES_WILLY_HT_2026-11-18.csv (FRENZY_WILLY + FRENZY_WILLY_HT, net as recorded) — second line only
Costs: fees + flat 0.10 % slip per taker leg (entry + taker exit) + real funding; stresses add 0.10 / 0.20 % per taker leg.
Account: one position per coin across sleeves (tie → EXH > L4 > FB > DIST > WILLY), book cap max_open_positions_hard 10,
free-margin cap (margin = notional ÷ 5 ≤ 0.975 × equity − open margin), size 0.609 × realised equity at entry, P&L booked at exit.
Capacity: SIZING depth model (fillable ≈ 3,505 × q5^0.054 ≈ $7k): fits → full, ≥ ½ fits → shrink to fill, else skip.
Usage: venv/bin/python -I scripts/joint_short_sim.py"""
import heapq, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sizing_exh_flatbase as SZ  # noqa: E402

R = SZ.R; RP = os.path.join(R, "reports")
OUT = os.path.join(RP, "JOINT_SIM_2026-11-18_data"); os.makedirs(OUT, exist_ok=True)
DAY = 86_400_000
DAYS = SZ.DAYS; ND = SZ.ND; D0ms, D1ms = SZ.TS(SZ.D0), SZ.TS(SZ.D1)
X = 0.24375 * 0.5 * 5          # 0.609
LEV = 5.0; RESERVE = 0.025; BOOK = 10
PRIO = dict(EXH=0, L4=1, FB=2, DIST=3, WILLY=4)
RNG = np.random.default_rng(20261118)
XMAP_CUR = {}
XREC = dict(EXH=0.183, L4=0.183, DIST=0.183, FB=0.305)   # SIZING_EXH_FLATBASE recommended (EXH 0.15 × 5×, FB 0.25 × 5×); L4 / DIST at EXH's
depth_of = lambda q5: np.exp(SZ.LOGC) * np.asarray(q5, float) ** SZ.BETA


def load():
    E = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"))
    E = E[~E.dl582_block.astype(bool)]
    tk = (E.exit_kind != "TP").values
    exh = pd.DataFrame(dict(sleeve="EXH", pair=E.pair.values, entry_ms=E.entry_ms.values, exit_ms=E.exit_ms.values,
                            net=E.n0.values - 0.10 * (1 + tk), nt=1 + tk, stop=(E.exit_kind == "SL").values, depth=depth_of(E.q5)))
    F = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_FLATBASE_2026-11-18.csv"))
    tk = F.taker_exit.astype(bool).values
    fb = pd.DataFrame(dict(sleeve="FB", pair=F.pair.values, entry_ms=F.entry_ms.values, exit_ms=F.exit_ms.values,
                           net=F.n0.values - 0.10 * (1 + tk), nt=1 + tk, stop=F.stop.astype(bool).values, depth=depth_of(F.q5)))
    L = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_EXH_WIDE_L4_2026-11-18.csv.gz"))
    bk = set(L[L.set == "BASE"].key); A = L[(L.set == "C") & ~L.key.isin(bk)]
    blk = np.isfinite(A.oi_v24) & (A.ssc > -0.0366) & (A.oi_v24 > 0.128); A = A[~blk]
    tk = (A.EX_k != 0).values
    l4 = pd.DataFrame(dict(sleeve="L4", pair=A.pair.values, entry_ms=A.entry_ms.values, exit_ms=A.EX_x.values.astype("int64"),
                           net=A.EX_r.values - 0.10 * (1 + tk) - 0.05 - np.where(tk, 0.05, 0.02) + A.EX_f.values, nt=1 + tk,
                           stop=(A.EX_k == 1).values, depth=depth_of(A.q5), net_src=A.net.values))
    import dist_wl_analyze as DA, dist_wl_survivors as SV  # noqa: E402
    S = pd.read_pickle(os.path.join(RP, "DIST_WL_2026-11-18_data", "rows.pkl"))
    REP = dict(kind="1D", var="rsi1h", lo=74.07103, hi=np.inf, mode="keep_only", var2="", lo2=np.nan, hi2=np.nan)
    km = SV.keep_mask(S, REP) & (S.LIVE_k.values != 3)
    n_onepos = int(DA.one_pos(S, SV.keep_mask(S, REP)).sum())
    S = S[km]; tk = (S.LIVE_k != 0).values
    dist = pd.DataFrame(dict(sleeve="DIST", pair=S.pair.values, entry_ms=S.entry_ms.values, exit_ms=S.LIVE_x.values.astype("int64"),
                             net=S.net_LIVE.values, nt=1 + tk, stop=(S.LIVE_k == 1).values, depth=depth_of(S.q5)))
    W = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_WILLY_HT_2026-11-18.csv"))
    tk = ~W.reason.isin(["FRENZY_TP", "WILLY_LATE_TP"]).values
    wil = pd.DataFrame(dict(sleeve="WILLY", sub=W.sleeve.values, pair=W.pair.values, entry_ms=W.entry_ms.values, exit_ms=W.exit_ms.values,
                            net=W.net.values, nt=1 + tk, stop=(W.reason == "STOP_LOSS").values, depth=np.median(exh.depth)))
    T = pd.concat([exh, fb, l4, dist, wil], ignore_index=True)
    T = T[(T.entry_ms >= D0ms) & (T.entry_ms < D1ms)].copy()
    T["prio"] = T.sleeve.map(PRIO)
    T = T.sort_values(["entry_ms", "prio"]).reset_index(drop=True)
    return T, n_onepos


def sim(T, cap=False, E0=10_000.0, minfrac=0.5, margin=True, book=BOOK, xmap=None, coin=True):
    """event sim → dict(eq_day, r_day, rec, skips, ...). Exits at the same ms as an entry are booked first."""
    e = T.entry_ms.values; x = T.exit_ms.values; net = T.net.values; pr = T.pair.values; dep = T.depth.values; sl = T.sleeve.values
    eq = E0; openh = []; held = set(); mopen = 0.0; held_n = {}
    taken = np.zeros(len(T), bool); notional = np.zeros(len(T)); fracs = np.zeros(len(T)); pnl = np.zeros(len(T))
    why = np.array([""] * len(T), dtype=object); nopen_sig = np.zeros(len(T), int)

    def close_until(t):
        nonlocal eq, mopen
        while openh and openh[0][0] <= t:
            _, j = heapq.heappop(openh)
            p = notional[j] * net[j] / 100.0
            pnl[j] = p; eq += p; mopen -= notional[j] / LEV
            held_n[pr[j]] -= 1
            if held_n[pr[j]] == 0:
                held.discard(pr[j])
    for i in range(len(T)):
        close_until(e[i])
        nopen_sig[i] = len(openh)
        if eq <= 0:
            why[i] = "ruin"; continue
        if coin and pr[i] in held:
            why[i] = "coin"; continue
        if len(openh) >= book:
            why[i] = "book"; continue
        N = (xmap.get(sl[i], X) if xmap else X) * eq
        if margin:
            free = eq * (1 - RESERVE) - mopen
            if free * LEV < N:
                if free * LEV < 0.25 * N:
                    why[i] = "margin"; continue
                N = free * LEV; why[i] = "margin_shrunk"
        if cap:
            if dep[i] < N:
                if dep[i] >= minfrac * N or minfrac == 0:
                    N = dep[i]; why[i] = "depth_shrunk"
                else:
                    why[i] = "depth"; continue
        taken[i] = True; notional[i] = N; fracs[i] = N / eq; mopen += N / LEV; held.add(pr[i])
        heapq.heappush(openh, (x[i], i)); held_n[pr[i]] = held_n.get(pr[i], 0) + 1
    close_until(np.iinfo(np.int64).max)
    xd = x // DAY - DAYS[0]
    inwin = taken & (xd >= 0) & (xd < ND)
    dpnl = np.zeros(ND); np.add.at(dpnl, xd[inwin], pnl[inwin])
    eq_day = E0 + np.cumsum(dpnl)
    prev = np.r_[E0, eq_day[:-1]]
    r_day = np.where(prev > 0, dpnl / np.maximum(prev, 1e-9), 0.0)
    return dict(taken=taken, notional=notional, frac=fracs, pnl=pnl, why=why, nopen_sig=nopen_sig, eq_day=eq_day, r_day=r_day,
                dpnl=dpnl, prev=prev, xd=xd, inwin=inwin)


def stats(r, eq0=1.0):
    eq = np.cumprod(1 + r)
    if eq[-1] <= 0:
        return dict(dcr=-100, cagr=-100, maxdd=-100, worst_day=r.min() * 100, worst7=-100, uw=len(r))
    dcr = (eq[-1] ** (1 / len(r)) - 1) * 100
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]; dd = eq / peak - 1
    uw = best = 0
    for v in dd:
        uw = uw + 1 if v < -1e-12 else 0; best = max(best, uw)
    w7 = np.expm1(pd.Series(np.log1p(r)).rolling(7).sum().min()) * 100
    return dict(dcr=dcr, cagr=(eq[-1] ** (365 / len(r)) - 1) * 100, maxdd=dd.min() * 100, worst_day=r.min() * 100, worst7=w7, uw=best)


IDX10 = SZ.boot_idx(ND, nb=2000, block=10, rng=np.random.default_rng(7))


def ddp(r):
    P = r[IDX10]; eq = np.cumprod(1 + P, 1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(P), 1)), eq], 1), 1)[:, 1:]
    m = (eq / peak - 1).min(1)
    return (m <= -0.30).mean() * 100, (m <= -0.50).mean() * 100


def open_profile(T, s):
    m = s["taken"]; e = np.sort(T.entry_ms.values[m]); x = np.sort(T.exit_ms.values[m])
    grid = np.arange(D0ms, D1ms, 300_000)
    c = np.searchsorted(e, grid, "right") - np.searchsorted(x, grid, "right")
    return np.percentile(c, 95), c.max(), (c >= BOOK).mean() * 100


def stops_per_day(T, s):
    m = s["taken"] & T.stop.values & s["inwin"]
    c = np.bincount(s["xd"][m], minlength=ND)
    act = c[c > 0]
    return (np.percentile(act, 99) if len(act) else 0), c.max(), c


def one_size_dcr(T, s):
    m = s["taken"] & s["inwin"]; r = np.zeros(ND); xs_ = T.sleeve.map(lambda z: XMAP_CUR.get(z, X)).values
    np.add.at(r, s["xd"][m], xs_[m] * T.net.values[m] / 100)
    return stats(r)["dcr"]


def row(name, T, s):
    m = s["taken"]; st = stats(s["r_day"]); p30, p50 = ddp(s["r_day"])
    op95, opmx, opfull = open_profile(T, s); sp99, spmx, _ = stops_per_day(T, s)
    return dict(config=name, tpd=m.sum() / ND, avg=T.net.values[m].mean(), dcr=st["dcr"], dcr_1size=one_size_dcr(T, s), cagr=st["cagr"],
                maxdd=st["maxdd"], worst_day=st["worst_day"], worst7=st["worst7"], uw=st["uw"], p30=p30, p50=p50,
                open_p95=op95, open_max=opmx, book_full_pct=opfull, stops_p99=sp99, stops_max=spmx, final_x=np.prod(1 + s["r_day"]))


CONFIGS = [("EXH + FB (before)", ["EXH", "FB"]), ("+ L4", ["EXH", "FB", "L4"]), ("+ DIST", ["EXH", "FB", "DIST"]),
           ("all four", ["EXH", "FB", "L4", "DIST"]), ("all four + WILLY/HT", ["EXH", "FB", "L4", "DIST", "WILLY"])]
HALVES = SZ.HALVES


def variant(T, hc=0.0, stress=0.0, avgs=None):
    T = T.copy()
    for sl in ("L4", "DIST"):
        m = T.sleeve.values == sl
        T.loc[m, "net"] = T.net.values[m] - hc * avgs[sl]
    T["net"] = T.net.values - stress * T.nt.values
    return T


def balance_paths(T, s, E0s=(1_000, 5_000, 10_000), H=(30, 60, 90), nb=5000, block=10, cap=True):
    """day-block bootstrap of the joint days; each sampled day re-sizes its trades at the path's start-of-day balance
    (frac × balance, depth rule) — selection (coin lock / book / margin) taken from the full-history sim."""
    m = s["taken"] & s["inwin"]; d = s["xd"][m]
    fr = s["frac"][m]; nt = T.net.values[m]; dp = T.depth.values[m]
    M = np.bincount(d, minlength=ND).max()
    Fm = np.zeros((ND, M)); Nm = np.zeros((ND, M)); Dm = np.full((ND, M), 1e18); k = np.zeros(ND, int)
    for a, b, c, dd in zip(fr, nt, dp, d):
        Fm[dd, k[dd]] = a; Nm[dd, k[dd]] = b; Dm[dd, k[dd]] = c; k[dd] += 1
    out = {}
    Hmax = max(H)
    idx = SZ.boot_idx(ND, nb=nb, block=block, rng=RNG)[:, :Hmax]
    for E0 in E0s:
        E = np.full(nb, float(E0)); traj = np.zeros((nb, Hmax))
        for t in range(Hmax):
            dd = idx[:, t]; N = Fm[dd] * E[:, None]
            if cap:
                Dd = Dm[dd]; N = np.where(Dd >= N, N, np.where(Dd >= 0.5 * N, Dd, 0.0))
            E = np.maximum(E + (N * Nm[dd] / 100).sum(1), 0.0); traj[:, t] = E
        for h in H:
            v = traj[:, h - 1]
            out[(E0, h)] = dict(p10=np.percentile(v, 10), med=np.median(v), p90=np.percentile(v, 90), p_loss=(v < E0).mean() * 100)
    return out


def kelly_joint(T, s):
    m = s["taken"] & s["inwin"]; r1 = np.zeros(ND); np.add.at(r1, s["xd"][m], T.net.values[m] / 100)   # per 1× notional
    xs = np.round(np.arange(0.02, 3.0, 0.01), 3)
    def best(r):
        g = [np.log1p(a * r).mean() if (1 + a * r).min() > 0 else -np.inf for a in xs]
        return xs[int(np.argmax(g))] if max(g) > 0 else 0.0
    mb = r1[IDX10].mean(1); sh = r1.mean() - np.percentile(mb, 5)
    return dict(obs=best(r1), lb10=best(r1 - sh), hc50=best(r1 - 0.5 * r1.mean()))


def main():
    T, n_onepos = load()
    res = dict(n_onepos_dist=n_onepos)
    L = []; P = lambda *a: L.append(" ".join(str(z) for z in a))
    # coverage
    cov = T.groupby("sleeve").agg(N=("net", "size"), first=("entry_ms", "min"), last=("entry_ms", "max"), avg=("net", "mean"),
                                  wr=("net", lambda v: (v > 0).mean() * 100), stop_pct=("stop", "mean"))
    cov["first"] = pd.to_datetime(cov["first"], unit="ms").dt.date; cov["last"] = pd.to_datetime(cov["last"], unit="ms").dt.date
    cov["avg_stop"] = T[T.stop].groupby("sleeve").net.mean()
    P("COVERAGE (candidate trades, flat 0.10 slip)\n" + cov.round(3).to_string())
    P("L4 source net (walk slip model) avg", T[T.sleeve == "L4"].net_src.mean().round(3))
    P("WILLY by sub:", T[T.sleeve == "WILLY"].groupby("sub").net.agg(["count", "mean"]).round(3).to_dict())
    avgs = cov.avg.to_dict()

    # 1. check: reproduce SIZING EXH + FB (L2 all 3,998, model slip, depth cap at $10k one size)
    E = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv")); E["taker_exit"] = E.exit_kind != "TP"
    F = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_FLATBASE_2026-11-18.csv"))
    for D_ in (E, F):
        D_.drop(D_[(D_.entry_ms < D0ms) | (D_.entry_ms >= D1ms)].index, inplace=True)
    chk = {}
    for a, b in ((X, X), (0.183, 0.305)):
        r = SZ.daily(E, a) + SZ.daily(F, b); st = SZ.path_stats(r); p = SZ.dd_probs(r)
        chk[f"{a:.3f}/{b:.3f}"] = dict(N_exh=len(E), N_fb=len(F), dcr=st["dcr"], maxdd=st["maxdd"], worst_day=st["worst_day"], p30=p[0], p50=p[1])
    for lab, Ex in (("flat 0.10 slip, no depth cap, EXH unfiltered (one size)", E), ("… + DL 582 filter (one size)", E[~E.dl582_block.astype(bool)])):
        r = np.zeros(ND)
        for D_, tkc in ((Ex, Ex.taker_exit.values), (F, F.taker_exit.astype(bool).values)):
            xd = D_.exit_ms.values // DAY - DAYS[0]; ok = (xd >= 0) & (xd < ND)
            np.add.at(r, xd[ok], X * (D_.n0.values - 0.10 * (1 + tkc))[ok] / 100)
        st = stats(r); p = ddp(r)
        chk[lab] = dict(N_exh=len(Ex), N_fb=len(F), dcr=st["dcr"], maxdd=st["maxdd"], worst_day=st["worst_day"], p30=p[0], p50=p[1])
    res["check"] = chk; P("CHECK vs SIZING\n" + pd.DataFrame(chk).T.round(2).to_string())
    # same check through the joint engine: EXH (all 3,998, flat slip) + FB, no book / margin / coin lock, uncapped
    # (isolates the engine; differences vs the SIZING line = flat slip vs model slip + per-trade compounding)

    # 2. main table — uncapped (edge), then capped from $10k / $1k
    sims = {}; rows = []
    for name, sl in CONFIGS:
        Tc = T[T.sleeve.isin(sl)].reset_index(drop=True)
        s = sim(Tc); sims[name] = (Tc, s); rows.append(row(name, Tc, s))
    global XMAP_CUR
    XMAP_CUR = XREC
    for vn, hc in (("in-sample", 0), ("haircut 50 % (L4+DIST)", 0.5)):
        Tc = variant(T[T.sleeve.isin(["EXH", "FB", "L4", "DIST"])], hc, 0, avgs).reset_index(drop=True)
        s = sim(Tc, xmap=XREC); rr = row(f"all four at SIZING sizes | {vn}", Tc, s); rows.append(rr); sims[f"rec | {vn}"] = (Tc, s)
    XMAP_CUR = {}
    MT = pd.DataFrame(rows); res["main"] = MT
    # bridge from the SIZING check to the joint headline (EXH + FB at 0.609)
    br = []
    Eb = T[T.sleeve.isin(["EXH", "FB"])].reset_index(drop=True)
    for lab, kw in (("no coin lock, no margin cap, no book", dict(coin=False, margin=False, book=10**9)),
                    ("+ coin lock", dict(margin=False, book=10**9)), ("+ book 10", dict(margin=False)), ("+ free-margin cap (headline)", dict())):
        s = sim(Eb, **kw); st = stats(s["r_day"]); br.append(dict(step=lab, N=int(s["taken"].sum()), dcr=st["dcr"], dcr_1size=one_size_dcr(Eb, s), maxdd=st["maxdd"], worst_day=st["worst_day"]))
    res["bridge"] = pd.DataFrame(br); P("\nBRIDGE\n" + res["bridge"].round(3).to_string())
    P("\nMAIN (uncapped, compounding per trade)\n" + MT.round(3).to_string())

    # skip reasons per sleeve (all four, and + WILLY)
    for name in ("all four", "all four + WILLY/HT", "EXH + FB (before)"):
        Tc, s = sims[name]
        sk = pd.crosstab(Tc.sleeve, s["why"]); sk["signals"] = Tc.groupby("sleeve").size(); sk["taken"] = Tc[s["taken"]].groupby("sleeve").size()
        sk["taken_avg"] = Tc[s["taken"]].groupby("sleeve").net.mean(); sk["skipped_avg"] = Tc[~s["taken"]].groupby("sleeve").net.mean()
        res[f"skips_{name}"] = sk; P(f"\nSKIPS {name}\n" + sk.round(3).to_string())

    # 3. haircut + stress
    var_rows = []
    for vn, hc, stv in (("in-sample", 0, 0), ("haircut 30 % (L4+DIST)", 0.3, 0), ("haircut 50 % (L4+DIST)", 0.5, 0),
                        ("+0.10 %/side slip", 0, 0.10), ("+0.20 %/side slip", 0, 0.20), ("haircut 50 % + 0.10 slip", 0.5, 0.10)):
        Tv = variant(T, hc, stv, avgs)
        for name, sl in CONFIGS:
            if vn != "in-sample" and name == "EXH + FB (before)" and hc > 0 and stv == 0:
                pass
            Tc = Tv[Tv.sleeve.isin(sl)].reset_index(drop=True); s = sim(Tc)
            rr = row(name, Tc, s); rr["variant"] = vn; var_rows.append(rr)
            if name == "all four":
                sims[f"all four | {vn}"] = (Tc, s)
    VT = pd.DataFrame(var_rows); res["variants"] = VT
    P("\nVARIANTS\n" + VT.round(3).to_string())

    # capped full-history paths
    cap_rows = []
    for name, sl in CONFIGS:
        Tc = T[T.sleeve.isin(sl)].reset_index(drop=True)
        for E0 in (1_000, 10_000):
            for mf in (0.5, 0.25, 0.0):
                if mf != 0.5 and E0 != 10_000:
                    continue
                s = sim(Tc, cap=True, E0=E0, minfrac=mf); st = stats(s["r_day"])
                w = pd.Series(s["why"]); m = s["taken"]
                cap_rows.append(dict(config=name, E0=E0, minfrac=mf, final=s["eq_day"][-1], dcr=st["dcr"], maxdd=st["maxdd"], worst_day=st["worst_day"],
                                     uw=st["uw"], depth_skip=(w == "depth").sum(), depth_shrunk=(w == "depth_shrunk").sum(), taken=m.sum(),
                                     days_to_25k=int(np.argmax(s["eq_day"] >= 25_000)) if (s["eq_day"] >= 25_000).any() else None,
                                     last365_dcr=stats(s["r_day"][-365:])["dcr"]))
                if name == "all four" and E0 == 10_000 and mf == 0.5:
                    sims["all four | capped $10k"] = (Tc, s)
    CT = pd.DataFrame(cap_rows); res["capped"] = CT
    P("\nCAPPED full history\n" + CT.round(3).to_string())

    # 4. ten worst joint days (all four, uncapped, in-sample) + the same with WILLY
    for name in ("all four", "all four + WILLY/HT"):
        Tc, s = sims[name]
        m = s["taken"] & s["inwin"]
        by = pd.DataFrame(dict(d=s["xd"][m], sl=Tc.sleeve.values[m], p=s["pnl"][m], stop=Tc.stop.values[m]))
        wd = np.argsort(s["r_day"])[:10]; out = []
        for d in wd:
            g = by[by.d == d]; prev = s["prev"][d]
            part = g.groupby("sl").agg(p=("p", "sum"), n=("p", "size"), st=("stop", "sum"))
            out.append(dict(date=str(pd.to_datetime((DAYS[0] + d) * DAY, unit="ms").date()), day_pct=s["r_day"][d] * 100,
                            stops=int(g.stop.sum()), closes=len(g),
                            sleeves=" · ".join(f"{k} {v.p / prev * 100:+.1f} % ({int(v.st)}/{int(v.n)} stops)" for k, v in part.iterrows())))
        res[f"worst_{name}"] = pd.DataFrame(out); P(f"\nWORST DAYS {name}\n" + pd.DataFrame(out).round(2).to_string())

    # 5. per half
    hrows = []
    for name, _ in CONFIGS + [("all four | haircut 50 % (L4+DIST)", None)]:
        Tc, s = sims[name]
        for h, a, b in HALVES:
            mm = (DAYS >= SZ.TS(a) // DAY) & (DAYS < SZ.TS(b) // DAY)
            st = stats(s["r_day"][mm]); hrows.append(dict(config=name, half=h, dcr=st["dcr"], maxdd=st["maxdd"]))
    HT = pd.DataFrame(hrows).pivot(index="half", columns="config", values=["dcr", "maxdd"]); res["halves"] = HT
    P("\nHALVES\n" + HT.round(2).to_string())
    # per sleeve contribution per half (all four, % of start-of-day equity summed)
    Tc, s = sims["all four"]; m = s["taken"] & s["inwin"]
    contrib = pd.DataFrame(dict(d=s["xd"][m], sl=Tc.sleeve.values[m], r=s["pnl"][m] / s["prev"][s["xd"][m]]))
    contrib["half"] = pd.cut(DAYS[contrib.d] * DAY, [SZ.TS(a) for _, a, _ in HALVES] + [SZ.TS(HALVES[-1][2])], labels=[h for h, _, _ in HALVES], right=False)
    CH = contrib.groupby(["half", "sl"], observed=True).r.sum().unstack() * 100
    nd_h = pd.Series({h: ((DAYS >= SZ.TS(a) // DAY) & (DAYS < SZ.TS(b) // DAY)).sum() for h, a, b in HALVES})
    CH = CH.div(nd_h, axis=0); res["half_contrib"] = CH; P("\nHALF contribution (simple %/day)\n" + CH.round(3).to_string())

    # 6. balance paths
    bp = {}
    for nm in ("all four", "all four | haircut 50 % (L4+DIST)", "all four | haircut 30 % (L4+DIST)", "EXH + FB (before)", "rec | in-sample", "rec | haircut 50 % (L4+DIST)"):
        Tc, s = sims[nm]
        bp[nm] = balance_paths(Tc, s)
        bp[nm + " | no cap"] = balance_paths(Tc, s, cap=False)
    res["balance"] = bp
    for k, v in bp.items():
        P(f"\nBALANCE {k}\n" + pd.DataFrame(v).T.round(0).to_string())

    # 7. Kelly / per stop
    kj = {}
    for nm in ("EXH + FB (before)", "all four", "all four | haircut 50 % (L4+DIST)", "all four + WILLY/HT"):
        Tc, s = sims[nm]; kj[nm] = kelly_joint(Tc, s)
    Tc, s = sims["all four"]
    ps = Tc[s["taken"] & Tc.stop.values].groupby("sleeve").net.mean()
    sp99, spmx, cday = stops_per_day(Tc, s)
    avg_stop = Tc[s["taken"] & Tc.stop.values].net.mean()
    act = cday[cday > 0]
    res["kelly"] = dict(joint=kj, per_stop=(ps * X).to_dict(), avg_stop=avg_stop, stops_p50_active=np.percentile(act, 50),
                        stops_p95_active=np.percentile(act, 95), stops_p99=sp99, stops_max=spmx)
    P("\nKELLY\n" + json.dumps(res["kelly"], indent=1, default=float))

    # 8. chart
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
    dts = pd.to_datetime(DAYS * DAY, unit="ms")
    for (name, _), c in zip(CONFIGS, cols):
        _, s = sims[name]; eq = np.cumprod(1 + s["r_day"])
        ax.plot(dts, eq, color=c, lw=1.6 if name == "all four" else 1.2, label=name)
    _, s = sims["all four | haircut 50 % (L4+DIST)"]; ax.plot(dts, np.cumprod(1 + s["r_day"]), color="#6250d6", lw=1.2, ls="--", label="all four, 50 % haircut on L4+DIST")
    _, s = sims["all four | capped $10k"]; ax.plot(dts, s["eq_day"] / 10_000, color="#52514e", lw=1.4, ls=":", label="all four, depth cap, $10k start")
    ax.set_yscale("log"); ax.set_ylabel("equity multiple (log)"); ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Joint short book at 0.609 × equity per trade — BACKTEST 2023-07 → 2026-11", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(RP, "JOINT_SIM_2026-11-18.png")); plt.close(fig)

    open(os.path.join(OUT, "tables.txt"), "w").write("\n".join(L))
    pd.to_pickle(res, os.path.join(OUT, "res.pkl"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
