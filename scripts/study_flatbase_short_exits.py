#!/usr/bin/env python3
"""FRENZY flat-base short — EXIT follow-up (2026-10-22, post-hoc with an OOS split). Pre-registration addendum:
reports/FRENZY_FLATBASE_SHORT_2026-10-22_EXITS_PREREG.md. READ-ONLY on the bot; no network (uses the main study's data folder).

FLAT cohort (frozen FLAT14 ≤ 44.39 %) · entries S2 / S3 / S4 (the main study's signal bars, data/signals.pkl) · 72 exits = TP −5/−8/−10/−15/−20 %
or trail (arm −8 %, trail 4 %) × max hold 12/24/48/72 h × stop 10/15/25 % · 0.20 % costs + real funding · gap-aware stops · one short per pair
per cell · random-bar null with the SAME exit, costs, funding and one-short-per-pair rule (200 shuffles).
Selection on 2025 only (top 3 by the 2025 day-bootstrap lower bound, N ≥ 300) → OOS 2026-H1 / H2 / ② master / ① live.
Writes reports/FRENZY_FLATBASE_SHORT_2026-10-22_EXITS_cells.csv, _EXITS_trades.csv, _EXITS_tables.md.
Usage: venv/bin/python scripts/study_flatbase_short_exits.py [--procs 7]
"""
import argparse, io, os, sys, zlib
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_flatbase_short_common as C  # noqa: E402

HM = 72 * 12
NB = 200
TPS = [5, 8, 10, 15, 20, "trail"]
HOLDS = [12, 24, 48, 72]
STOPS = [10, 15, 25]
EXITS = [(tp, hd, s) for tp in TPS for hd in HOLDS for s in STOPS]
NX = len(EXITS)
EXN = [f"{'trail8-4' if tp == 'trail' else f'tp{tp}'}|{hd}h|stop{s}" for tp, hd, s in EXITS]
ENTRIES = ["S2", "S3", "S4", "D2", "D6", "D12", "D24"]   # D = first eligible bar ≥ spike + D h (non-anticipating)
FOLDS = ["2025H1", "2025H2", "2026H1", "2026H2"]
ACCOUNT, RISK = 3200.0, 0.05


def first_idx(mask):
    return np.where(mask.any(1), mask.argmax(1), HM + 10)


def paths(arr, idx, n):
    t, o, h, l, c = arr
    m = len(idx)
    H = np.full((m, HM), np.nan); Lo = H.copy(); O = H.copy(); Cc = H.copy(); T = np.zeros((m, HM), dtype="int64"); av = np.zeros(m, dtype="int64")
    for r, i in enumerate(idx):
        a, b = i + 1, min(n, i + 1 + HM); k = b - a; av[r] = max(k, 0)
        if k > 0:
            H[r, :k] = h[a:b]; Lo[r, :k] = l[a:b]; O[r, :k] = o[a:b]; Cc[r, :k] = c[a:b]; T[r, :k] = t[a:b]
            if k < HM:
                T[r, k:] = t[b - 1] + C.BAR * np.arange(1, HM - k + 1)
    return H, Lo, O, Cc, T, av


def simulate(E, H, Lo, O, Cc, T, av, data_open):
    n = len(E); r = np.arange(n)
    hh = np.where(np.isnan(H), -np.inf, H); ll = np.where(np.isnan(Lo), np.inf, Lo)
    cmh = np.maximum.accumulate(hh, 1); cml = np.minimum.accumulate(ll, 1)
    rm = np.minimum.accumulate(np.minimum(ll, E[:, None]), 1)
    rm_prev = np.concatenate([E[:, None], rm[:, :-1]], 1)
    armed = rm_prev <= E[:, None] * 0.92
    kS = {s: first_idx(cmh >= (E * (1 + s / 100))[:, None]) for s in STOPS}
    kTR = {}
    for s in STOPS:
        lev = np.where(armed, np.minimum(E[:, None] * (1 + s / 100), rm_prev * 1.04), E[:, None] * (1 + s / 100))
        k = first_idx(hh >= lev)
        kTR[s] = (k, lev[r, np.clip(k, 0, HM - 1)])
    kT = {tp: first_idx(cml <= (E * (1 - tp / 100))[:, None]) for tp in TPS if tp != "trail"}
    G = np.zeros((n, NX)); XT = np.zeros((n, NX), dtype="int64"); RS = np.zeros((n, NX), dtype="int8")
    for j, (tp, hd, s) in enumerate(EXITS):
        hb = hd * 12
        kmax = np.minimum(hb, av)
        if tp == "trail":
            ks, lvl = kTR[s]; kt = np.full(n, HM + 10)
        else:
            ks = kS[s]; lvl = E * (1 + s / 100); kt = kT[tp]
        ev = np.minimum(ks, kt); has = ev < kmax
        kx = np.where(has, ev, np.maximum(kmax - 1, 0)); kc = np.clip(kx, 0, HM - 1)
        g = (E - Cc[r, kc] * (1 + C.SLIP)) / E * 100
        rs = np.where(kmax < hb, np.where(data_open, 4, 5), 0).astype("int8")
        st = has & (ks == ev)
        Sx = np.maximum(lvl, O[r, np.clip(ks, 0, HM - 1)]) * (1 + C.SLIP)
        g = np.where(st, np.maximum((E - Sx) / E * 100, -100.0), g); rs = np.where(st, 1, rs)
        if tp != "trail":
            tt = has & (kt == ev) & ~st
            Tx = np.minimum(E * (1 - tp / 100), O[r, np.clip(kt, 0, HM - 1)]) * (1 + C.SLIP)
            g = np.where(tt, (E - Tx) / E * 100, g); rs = np.where(tt, 3, rs)
        G[:, j] = g; RS[:, j] = rs; XT[:, j] = T[r, kc] + C.BAR
    return G, XT, RS


def one_per_pair(entry_ms, XT):
    """rows already in time order → taken mask (rows × exits)."""
    busy = np.full(XT.shape[1], -1, dtype="int64"); TK = np.zeros(XT.shape, bool)
    for k in range(len(entry_ms)):
        tk = entry_ms[k] >= busy; TK[k] = tk; busy = np.where(tk, XT[k], busy)
    return TK


def worker(args):
    pair, eps, sig, now_ms = args
    full = C.load_full(pair)
    if full is None:
        return None
    t, o, h, l, c, v, qv, real = full
    n = len(t); data_open = bool(t[-1] + C.BAR >= now_ms - 2 * C.HR)
    F = C.funding_table(pair)
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    pos = lambda ts: int((ts - t[0]) // C.BAR)
    # fixed-delay entries: the first bar ≥ spike + D h whose gates hold ON DATA UP TO THAT BAR (vol24 is trailing; first-seen ≤ bar)
    extra = []
    for e in eps.itertuples():
        on = pos(e.on_t)
        fs = pos(e.first_seen_t) if e.first_seen_t is not None and not pd.isna(e.first_seen_t) else 10**12
        hi = min(n - 2, on + C.SEARCH)
        for D in (2, 6, 12, 24):
            for i in range(on + D * 12, hi + 1):
                if i >= fs and vol24[i] >= 20e6:
                    extra.append(dict(pair=pair, ep_on_t=e.on_t, entry=f"D{D}", i_t=int(t[i]), entry_ms=int(t[i]) + C.BAR, close=float(c[i]),
                                      E=float(c[i]) * (1 - C.SLIP), base=float(e.base)))
                    break
    cols = ["pair", "ep_on_t", "entry", "i_t", "entry_ms", "close", "E", "base"]
    sig = pd.concat([sig[cols], pd.DataFrame(extra, columns=cols)], ignore_index=True)
    out_sig = None
    if len(sig):
        idx = np.array([pos(x) for x in sig.i_t])
        H, Lo, O, Cc, T, av = paths((t, o, h, l, c), idx, n)
        G, XT, RS = simulate(sig.E.values, H, Lo, O, Cc, T, av, data_open)
        fu, ms = C.funding_sum(F, np.repeat(sig.entry_ms.values[:, None], NX, 1).ravel(), XT.ravel())
        NET = G - C.FEES + fu.reshape(G.shape)
        TK = np.zeros(G.shape, bool)
        for en in ENTRIES:
            rr = np.nonzero(sig.entry.values == en)[0]
            rr = rr[np.argsort(sig.entry_ms.values[rr], kind="stable")]
            if len(rr):
                TK[rr] = one_per_pair(sig.entry_ms.values[rr], XT[rr])
        MK = np.full((len(sig), 14), np.nan)
        for d in range(1, 4):
            k = d * 288 - 1; okk = av > k
            MK[okk, d - 1] = (sig.E.values[okk] - Cc[okk, k]) / sig.E.values[okk] * 100
        out_sig = dict(meta=sig, NET=NET.astype(np.float32), NOF=(G - C.FEES).astype(np.float32), XT=XT, RS=RS, TK=TK,
                       MISS=ms.reshape(G.shape), MK=MK.astype(np.float32))
    # null: one random bar per flat episode per shuffle, same exits / costs / funding, one-short-per-pair across the pair's episodes
    rng = np.random.default_rng(zlib.crc32(pair.encode()) + 7)
    keys, NN, NT, NE = [], [], [], []
    busy = np.full((NB, NX), -1, dtype="int64")
    for e in eps.sort_values("on_t").itertuples():
        on = pos(e.on_t)
        fs = pos(e.first_seen_t) if e.first_seen_t is not None and not pd.isna(e.first_seen_t) else 10**12
        hi = min(n - 2, on + C.SEARCH)
        cand = np.arange(on + 24, hi + 1)
        if len(cand):
            cand = cand[(cand >= fs) & (np.nan_to_num(vol24[cand]) >= 20e6)]
        if not len(cand):
            continue
        dd = rng.integers(on + 24, hi + 1, NB)                 # NON-ANTICIPATING: draw the delay first,
        pp = np.searchsorted(cand, dd)                          # then the first eligible bar at / after it
        okd = pp < len(cand)
        if not okd.any():
            continue
        ri = cand[np.minimum(pp, len(cand) - 1)]
        H, Lo, O, Cc, T, av = paths((t, o, h, l, c), ri, n)
        E0 = c[ri] * (1 - C.SLIP)
        G, XT, _ = simulate(E0, H, Lo, O, Cc, T, av, data_open)
        em = t[ri] + C.BAR
        fu, _ = C.funding_sum(F, np.repeat(em[:, None], NX, 1).ravel(), XT.ravel())
        net = G - C.FEES + fu.reshape(G.shape)
        tk = (em[:, None] >= busy) & okd[:, None]
        busy = np.where(tk, XT, busy)
        keys.append((pair, e.on_t)); NN.append(net.astype(np.float32)); NT.append(tk); NE.append(em)
    return pair, out_sig, keys, NN, NT, NE


def boot(day, x, B=2000, rng=np.random.default_rng(9)):
    d = pd.DataFrame({"d": day, "x": x}).groupby("d").x.agg(["sum", "count"])
    if len(d) < 3:
        return np.nan, np.nan
    s, cn = d["sum"].values, d["count"].values; i = rng.integers(0, len(d), (B, len(d)))
    m = s[i].sum(1) / cn[i].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=7); A = ap.parse_args()
    buf = io.StringIO()

    def P(*a):
        print(*a); print(*a, file=buf)
    E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    thr = E[(E.spike >= C.B_YEAR0) & (E.spike < C.FREEZE_CUT)].FLAT14.quantile(1 / 3)
    assert abs(thr - 44.39) < 0.01, thr
    E = E[(E.spike >= C.B_YEAR0) & (E.FLAT14 <= thr)].copy()
    S0 = pd.read_pickle(os.path.join(C.D, "signals.pkl"))
    S0 = S0[S0.entry.isin(ENTRIES)].copy()
    fk = set(zip(E.pair, E.on_t))
    S0 = S0[[k in fk for k in zip(S0.pair, S0.ep_on_t)]].reset_index(drop=True)
    now_ms = pd.read_pickle(os.path.join(C.D, "meta.pkl"))["now_ms"]
    jobs = [(p, g, S0[S0.pair == p], now_ms) for p, g in E.groupby("pair")]
    PARTS, NK, NNET, NTK, NEM = [], [], [], [], []
    with mp.Pool(A.procs) as pool:
        for res in pool.imap_unordered(worker, jobs):
            if res is None:
                continue
            pair, o, keys, nn, nt, ne = res
            if o is not None:
                PARTS.append(o)
            NK += keys; NNET += nn; NTK += nt; NEM += ne
    S = pd.concat([o["meta"] for o in PARTS], ignore_index=True)
    NET = np.concatenate([o["NET"] for o in PARTS]); NOF = np.concatenate([o["NOF"] for o in PARTS]); XT = np.concatenate([o["XT"] for o in PARTS])
    RS = np.concatenate([o["RS"] for o in PARTS]); TK = np.concatenate([o["TK"] for o in PARTS]); MISS = np.concatenate([o["MISS"] for o in PARTS])
    S["mtm"] = list(np.concatenate([o["MK"] for o in PARTS]))
    NNET = np.stack(NNET); NTK = np.stack(NTK); NEM = np.stack(NEM)
    nrow = {k: i for i, k in enumerate(NK)}
    # real-episode flag for ① ② (from the main study's trades csv)
    LMr = pd.read_csv(C.OUTP + "_live_master.csv", usecols=["pair", "spike", "real"])
    rmap = dict(zip(zip(LMr.pair, LMr.spike.astype(str).str[:16]), LMr.real))
    spk = dict(zip(zip(E.pair, E.on_t), E.spike.astype(str).str[:16]))
    S["real_ep"] = [bool(rmap.get((p, spk.get((p, k), "")), False)) for p, k in zip(S.pair, S.ep_on_t)]
    S["dt"] = pd.to_datetime(S.entry_ms, unit="ms"); S["day"] = S.dt.dt.strftime("%Y-%m-%d")
    S["per"] = np.select([S.dt >= C.B_LIVE0, S.dt >= C.B_MAST0, S.dt >= pd.Timestamp("2026-07-01"), S.dt >= pd.Timestamp("2026-01-01")],
                         ["1_live", "2_master", "2026H2", "2026H1"], "2025")
    S["fold"] = np.where(S.per == "2025", np.where(S.dt < pd.Timestamp("2025-07-01"), "2025H1", "2025H2"), S.per)
    S["nrow"] = [nrow.get(k, -1) for k in zip(S.pair, S.ep_on_t)]

    def stats(m, j, null=True):
        mm = m & TK[:, j]
        x = NET[mm, j].astype(float)
        if not len(x):
            return dict(N=0)
        sub = S[mm]
        lo, hi = boot(sub.day.values, x)
        k = int(np.ceil(len(x) * 0.05))
        pos_ = x.sum(); pc = pd.Series(x).groupby(sub.pair.values).sum(); dc = pd.Series(x).groupby(sub.day.values).sum()
        out = dict(N=len(x), days=sub.day.nunique(), WR=(x > 0).mean() * 100, avg=x.mean(), lo=lo, hi=hi, nofund=NOF[mm, j].mean(),
                   ex_top5=np.sort(x)[:-k].mean() if len(x) >= 20 else np.nan, worst=x.min(),
                   top_pair_sh=pc.max() / pos_ * 100 if pos_ > 0 else np.nan, top_day_sh=dc.max() / pos_ * 100 if pos_ > 0 else np.nan,
                   fund_missing=MISS[mm, j].mean() * 100)
        if null:
            ks = sorted(set(sub.nrow[sub.nrow >= 0]))
            if ks:
                nv = NNET[ks, :, j].astype(float); nt = NTK[ks, :, j]
                nm = np.nansum(np.where(nt, nv, 0), 0) / np.maximum(nt.sum(0), 1)
                out["null"] = nm.mean(); out["p_null"] = (nm >= x.mean()).mean()
        return out

    def sizing(m, j):
        mm = m & TK[:, j]
        idx = np.nonzero(mm)[0]
        x = NET[idx, j].astype(float)
        p1 = np.percentile(x, 1); notional = ACCOUNT * RISK / abs(p1) * 100
        order = idx[np.argsort(S.entry_ms.values[idx])]
        openx, taken = [], []
        for k in order:
            openx = [(a, b) for a, b in openx if a > S.entry_ms[k]]
            if len(openx) >= 5 or any(b == S.pair[k] for _, b in openx):
                continue
            openx.append((XT[k, j], S.pair[k])); taken.append(k)
        taken = np.array(taken)
        pnl = NET[taken, j].astype(float) / 100 * notional
        ex = pd.to_datetime(XT[taken, j], unit="ms").normalize()
        days = pd.date_range(S.dt[taken].min().normalize(), ex.max(), freq="D")
        MT = np.stack(S.mtm.values[taken]).astype(float)
        ent = S.entry_ms.values[taken]; xt = XT[taken, j]; eq = []
        for d in days:
            de = d.value // 10**6 + C.DAY
            cl = xt <= de; op = ~cl & (ent <= de)
            k = np.clip(((de - ent[op]) // C.DAY).astype(int) - 1, 0, 13)
            eq.append(pnl[cl].sum() + np.nansum(MT[op][np.arange(op.sum()), k] / 100 * notional))
        eq = pd.Series(eq, index=days)
        return dict(p1=p1, notional=notional, taken=len(taken), sum_usd=pnl.sum(), worst_usd=pnl.min(), maxdd_usd=(eq - eq.cummax()).min())

    # ───── all cells, every period ─────
    rows = []
    for en in ENTRIES:
        me = (S.entry == en).values
        for j, nm in enumerate(EXN):
            r = dict(cell=f"{en}|{nm}", entry=en, exit=nm)
            for per in ("2025", "2026H1", "2026H2", "OOS26", "YEAR", "2_master", "1_live", "2025H1", "2025H2",
                        "x2025H1", "x2025H2", "x2026H1", "x2026H2"):
                if per.startswith("x"):
                    m = me & S.fold.isin([f for f in FOLDS if f != per[1:]]).values
                elif per in ("2025H1", "2025H2"):
                    m = me & (S.fold == per).values
                elif per == "OOS26":
                    m = me & S.per.isin(["2026H1", "2026H2"]).values
                elif per == "YEAR":
                    m = me & S.per.isin(["2025", "2026H1", "2026H2"]).values
                elif per in ("2_master", "1_live"):
                    m = me & (S.per == per).values & S.real_ep.values
                else:
                    m = me & (S.per == per).values
                s = stats(m, j, null=per in ("2025", "OOS26", "2026H1", "2026H2", "2025H1", "2025H2"))
                for k, v in s.items():
                    r[f"{per}_{k}"] = v
            rows.append(r)
    CE = pd.DataFrame(rows)
    # ───── walk-forward by half-year ─────
    P("## WALK-FORWARD (select on the other 3 half-years pooled by the day-bootstrap lower bound, N ≥ 300; score on the held-out half)")
    P("| held-out fold | selected cell | selection: N · avg [lo] | held-out N | held-out WR | held-out avg [95 %] | held-out null · p | best 5 % removed |")
    P("|---|---|---|---|---|---|---|---|")
    sel = {}
    for f in FOLDS:
        el = CE[CE[f"x{f}_N"] >= 300].sort_values(f"x{f}_lo", ascending=False)
        c = el.cell.iloc[0]; sel[f] = c; r = CE[CE.cell == c].iloc[0]
        P(f"| {f}{' (2025: flagged)' if f.startswith('2025') else ''} | {c} | {r[f'x{f}_N']:.0f} · {r[f'x{f}_avg']:+.2f} [{r[f'x{f}_lo']:+.2f}] | "
          f"{r[f'{f}_N']:.0f} | {r[f'{f}_WR']:.0f}% | {r[f'{f}_avg']:+.2f} [{r[f'{f}_lo']:+.2f}, {r[f'{f}_hi']:+.2f}] | {r[f'{f}_null']:+.2f} · {r[f'{f}_p_null']:.2f} | "
          f"{r[f'{f}_ex_top5']:+.2f} |")
    cnt = pd.Series(list(sel.values())).value_counts()
    P("\nselection counts: " + " · ".join(f"{k} ×{v}" for k, v in cnt.items()))
    P("top 5 per selection set (stability view): ")
    for f in FOLDS:
        el = CE[CE[f"x{f}_N"] >= 300].sort_values(f"x{f}_lo", ascending=False).head(5)
        P(f"- without {f}: " + " · ".join(f"{r.cell} [{r[f'x{f}_lo']:+.2f}]" for _, r in el.iterrows()))
    cands = [c for c, v in cnt.items() if v >= 3 and CE[CE.cell == c].iloc[0]["2026H1_avg"] > 0 and CE[CE.cell == c].iloc[0]["2026H2_avg"] > 0]
    P(f"\nCANDIDATES (selected ≥ 3 of 4 folds AND 2026-H1 > 0 AND 2026-H2 > 0): {cands if cands else 'NONE'}")
    picks = list(dict.fromkeys(list(cnt.index)))            # every selected cell gets the full OOS table
    P("\n## Every walk-forward-selected cell, all blocks (2025 rows flagged: lower-reliability data)")
    for c in picks:
        r = CE[CE.cell == c].iloc[0]
        en = r.entry; j = EXN.index(r.exit); me = (S.entry == en).values
        P(f"\n### {c}")
        P("| period | N | days | WR | avg % | 95 % day-boot | pre-funding | best 5 % removed | worst | random-bar null (same exit, costs, funding, one-per-pair) | p | top pair / day share |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for per, lab in (("2025H1", "⚠ 2025-H1"), ("2025H2", "⚠ 2025-H2"), ("2026H1", "2026-H1 OOS"), ("2026H2", "2026-H2 OOS"), ("OOS26", "2026 OOS total"),
                         ("2_master", "② master (real eps)"), ("1_live", "① live (real eps, open = MTM)"), ("YEAR", "③ full year")):
            g = lambda k: r.get(f"{per}_{k}", np.nan)
            if not g("N"):
                P(f"| {lab} | 0 | | | | | | | | | | |"); continue
            P(f"| {lab} | {g('N'):.0f} | {g('days'):.0f} | {g('WR'):.0f}% | {g('avg'):+.2f} | [{g('lo'):+.2f}, {g('hi'):+.2f}] | {g('nofund'):+.2f} | "
              f"{g('ex_top5'):+.2f} | {g('worst'):+.1f} | {g('null'):+.2f} | {g('p_null'):.3f} | {g('top_pair_sh'):.0f} % / {g('top_day_sh'):.0f} % |")
        mw = me & S.per.isin(["1_live", "2_master"]).values & S.real_ep.values & TK[:, j]
        wv = pd.DataFrame({"d": S.day[mw], "x": NET[mw, j]}).groupby("d").x.mean()
        P(f"\nwindows ①②: {len(wv)} · mean of window means {wv.mean() if len(wv) else np.nan:+.2f} · positive {(wv > 0).sum()} of {len(wv)}")
        for per, lab in (("YEAR", "full year"), ("OOS26", "2026 OOS")):
            m = me & ((S.per.isin(["2025", "2026H1", "2026H2"])) if per == "YEAR" else S.per.isin(["2026H1", "2026H2"])).values
            sz = sizing(m, j)
            P(f"sizing at $3,200 ({lab}): 1st-pct loss {sz['p1']:+.1f} % → ${sz['notional']:.0f}/short · taken {sz['taken']} · sum ${sz['sum_usd']:+,.0f} · worst trade "
              f"${sz['worst_usd']:+,.0f} · max drawdown (daily MTM) ${sz['maxdd_usd']:+,.0f} ({sz['maxdd_usd'] / ACCOUNT * 100:+.0f} %)")
        # bars
        ok = dict(N=r["YEAR_N"] >= 300 and r["YEAR_days"] >= 30, lo=r["OOS26_lo"] > 0, null=r["OOS26_p_null"] < 0.05,
                  h1=r["2026H1_avg"] > 0, h2=r["2026H2_avg"] > 0, win=(len(wv) >= 8 and wv.mean() > 0 and (wv > 0).mean() >= 0.5),
                  haircut=r["OOS26_avg"] * 0.6 >= 0.5, top5=r["OOS26_ex_top5"] > 0,
                  conc=(r["OOS26_top_pair_sh"] <= 25) and (r["OOS26_top_day_sh"] <= 25))
        wm = me & (S.per == "2_master").values & S.real_ep.values & TK[:, j]
        wv2 = pd.DataFrame({"d": S.day[wm], "x": NET[wm, j]}).groupby("d").x.mean()
        ok["master_not_negative"] = len(wv2) > 0 and wv2.mean() >= 0
        ok["candidate"] = c in cands
        P(f"② master windows: {len(wv2)} · mean of window means {wv2.mean() if len(wv2) else np.nan:+.2f}")
        P("SHIP BARS (OOS): " + " · ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in ok.items()) + f" → {'PASS' if all(ok.values()) else 'FAIL'}")
    # ───── every cell, compact: 2025 vs OOS ─────
    P("\n## All 504 cells (7 entries × 72 exits) — 2025 vs 2026 (2026 is OOS only for a pick selected without it), sorted by 2025 lower bound (top 25)")
    cols = ["cell", "2025_N", "2025_avg", "2025_lo", "2026H1_avg", "2026H2_avg", "OOS26_N", "OOS26_avg", "OOS26_lo", "OOS26_null", "OOS26_p_null", "OOS26_ex_top5"]
    tb = CE.sort_values("2025_lo", ascending=False).head(25)[cols]
    P("| " + " | ".join(cols) + " |"); P("|" + "---|" * len(cols))
    for rr in tb.itertuples(index=False):
        P("| " + " | ".join(f"{v:+.2f}" if isinstance(v, float) else str(v) for v in rr) + " |")
    P(f"\nOOS (2026) cells with lo > 0: {(CE.OOS26_lo > 0).sum()} of {len(CE)} · OOS mean > 0: {(CE.OOS26_avg > 0).sum()} · H1 > 0 and H2 > 0: "
      f"{((CE['2026H1_avg'] > 0) & (CE['2026H2_avg'] > 0)).sum()} · OOS beats its null at p < 0.05: {(CE.OOS26_p_null < 0.05).sum()}")
    best_oos = CE.sort_values("OOS26_lo", ascending=False).head(5)
    P("Best 5 cells by the OOS lower bound (hindsight — NOT a valid pick): " + " · ".join(
        f"{r.cell} OOS {r.OOS26_avg:+.2f} [{r.OOS26_lo:+.2f}] H2 {r['2026H2_avg']:+.2f} null {r.OOS26_null:+.2f} p {r.OOS26_p_null:.2f} (2025 {r['2025_avg']:+.2f})"
        for _, r in best_oos.iterrows()))
    # by TP level: avg across entries/holds/stops (shape)
    P("\n## Shape: mean of the OOS (2026) avg by TP / hold (averaged over entries and stops)")
    CE["tp"] = CE.exit.str.split("|").str[0]; CE["hold"] = CE.exit.str.split("|").str[1]
    P(CE.pivot_table(index="tp", columns="hold", values="OOS26_avg", aggfunc="mean").round(2).to_string())
    P("\n2025 (in-sample) same shape:")
    P(CE.pivot_table(index="tp", columns="hold", values="2025_avg", aggfunc="mean").round(2).to_string())
    # ───── 2025 reliability diagnostics (per half-year) ─────
    import json, glob as _g
    P("\n## Data reliability by half-year (why 2025 is flagged)")
    SY = json.load(open(os.path.join(C.D, "symbols.json"))); TRD = set(SY["trading"])
    EA = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl")); EA = EA[(EA.spike >= C.B_YEAR0) & (EA.spike < C.B_YEAR1)]
    EA["fold"] = np.select([EA.spike < pd.Timestamp("2025-07-01"), EA.spike < pd.Timestamp("2026-01-01"), EA.spike < pd.Timestamp("2026-07-01")],
                           ["2025H1", "2025H2", "2026H1"], "2026H2")
    jm = EXN.index("tp10|24h|stop15")
    btc = pd.read_csv(os.path.join(C.R, "reports", "backtest_cache_2025", "k5m_full", "BTCUSDT.csv"), usecols=["open_time", "c"])
    b2 = pd.read_csv(os.path.join(C.R, "reports", "backtest_cache", "k5m_full", "BTCUSDT.csv"), usecols=["open_time", "c"])
    btc = pd.concat([btc, b2]).drop_duplicates("open_time").set_index("open_time").c
    btc.index = pd.to_datetime(btc.index, unit="ms")
    P("| half-year | episodes (all) | pairs | flat share | episodes on pairs now delisted | flat signals | funding history missing (signals, 24 h exit) | funding paid avg (24 h, flat) | BTC change | BTC 30-d realised vol (median) |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for f, (a, b) in zip(FOLDS, [("2025-01-01", "2025-07-01"), ("2025-07-01", "2026-01-01"), ("2026-01-01", "2026-07-01"), ("2026-07-01", "2026-10-01")]):
        g = EA[EA.fold == f]
        m = (S.fold == f).values & TK[:, jm]
        fu = (NET[m, jm] - NOF[m, jm]).astype(float)
        bb = btc[a:b]; rv = bb.resample("1h").last().pct_change().rolling(720).std().median() * np.sqrt(24 * 365) * 100
        P(f"| {f} | {len(g)} | {g.pair.nunique()} | {(g.FLAT14 <= thr).mean() * 100:.0f} % | {(~g.pair.isin(TRD)).mean() * 100:.0f} % | {m.sum()} | "
          f"{MISS[m, jm].mean() * 100:.1f} % | {fu.mean():+.3f} | {(bb.iloc[-1] / bb.iloc[0] - 1) * 100:+.0f} % | {rv:.0f} % |")
    CE.to_csv(C.OUTP + "_EXITS_cells.csv", index=False)
    TW = S[["pair", "entry", "dt", "per", "real_ep", "FLAT14" if "FLAT14" in S else "pair", "close", "base"]].copy() if False else S[["pair", "entry", "dt", "per", "real_ep", "close", "base"]].copy()
    for c in picks:
        j = EXN.index(c.split("|", 1)[1]); en = c.split("|")[0]
        TW[f"net|{c}"] = np.where((S.entry == en).values & TK[:, j], NET[:, j], np.nan).round(3)
        TW[f"reason|{c}"] = np.where((S.entry == en).values & TK[:, j], pd.Series(RS[:, j]).map({0: "time", 1: "stop", 3: "tp", 4: "OPEN(mtm)", 5: "settle"}).values, "")
    TW.sort_values("dt").to_csv(C.OUTP + "_EXITS_trades.csv", index=False)
    open(C.OUTP + "_EXITS_tables.md", "w").write(buf.getvalue())
    print("done")
