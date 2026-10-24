#!/usr/bin/env python3
"""FLATBASE_SHORT — TP −3 % / stop 5 % exit test (2026-10-24). Pre-registration: reports/FRENZY_FLATBASE_TP3_STOP5_2026-10-24_PREREG.md.
READ-ONLY on the bot; no network (the exit study's cached data folder + locally cached 1m bars for the wick check).

Entry S4 only (the shipped DL 348 rule) · TP {3, 5, 8} × stop {5, 10, 15} × hold {72 (primary), 48 (context)} · the exit study's simulator,
costs, funding, gap-aware stop-first convention, one-short-per-pair and non-anticipating random-bar null — all IMPORTED from
scripts/study_flatbase_short_exits.py (worker / simulate take the exit grid as a parameter).
Writes reports/FRENZY_FLATBASE_TP3_STOP5_2026-10-24_cells.csv, _tables.md, _trades.csv.
Usage: venv/bin/python scripts/study_flatbase_tp3_stop5.py [--procs 8]
"""
import argparse, io, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_flatbase_short_common as C  # noqa: E402
import study_flatbase_short_exits as X   # noqa: E402

TPS, STOPS, HOLDS = [3, 5, 8], [5, 10, 15], [72, 48]
EXITS = [(tp, hd, s) for hd in HOLDS for tp in TPS for s in STOPS]          # 18
EXN = [f"tp{tp}|{hd}h|stop{s}" for tp, hd, s in EXITS]
BASE = {hd: EXN.index(f"tp8|{hd}h|stop15") for hd in HOLDS}
HALVES = ["2025H1", "2025H2", "2026H1", "2026H2"]
NBOOT, SEED = 4000, 24
BONF_Q = 100 * 0.05 / 8 / 2                                                  # 0.3125th percentile (two-sided 1 − 0.05/8)
K1M = os.path.join(C.R, "reports", "WILLY_LATE_JOIN_2026-10-22_data", "k1m")
OUT = os.path.join(C.R, "reports", "FRENZY_FLATBASE_TP3_STOP5_2026-10-24")
ACCOUNT, RISK = X.ACCOUNT, X.RISK


def wk(args):
    """X.worker on the 18-exit grid (S4 signals; X adds its D-entries, dropped later) + the stop bar's 5m bar (o, h, l, c) for every stop."""
    pair = args[0]
    full = C.load_full(pair)
    orig = C.load_full
    C.load_full = lambda p: full if p == pair else orig(p)                   # one CSV load per pair
    try:
        res = X.worker(args)
    finally:
        C.load_full = orig
    if res is None or res[1] is None:
        return res, None
    t, o, h, l, c = full[:5]
    o_ = res[1]
    XT, RS = o_["XT"], o_["RS"]
    kb = ((XT - C.BAR - t[0]) // C.BAR).astype(np.int64)
    ok = (RS == 1) & (kb >= 0) & (kb < len(t))
    kk = np.clip(kb, 0, len(t) - 1)
    SB = np.stack([np.where(ok, a[kk], np.nan) for a in (o, h, l, c)], -1).astype(np.float64)   # rows × exits × (o h l c)
    return res, SB


def boot_mean(day, x, B=NBOOT, seed=SEED):
    d = pd.DataFrame({"d": day, "x": x}).groupby("d").x.agg(["sum", "count"])
    if len(d) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed); i = rng.integers(0, len(d), (B, len(d)))
    m = d["sum"].values[i].sum(1) / d["count"].values[i].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_delta(dx, x, dy, y, B=NBOOT, seed=SEED):
    """day-block bootstrap of mean(x) − mean(y): resample UTC days jointly (both cells' trades of a drawn day come together)."""
    a = pd.DataFrame({"d": dx, "x": x}).groupby("d").x.agg(["sum", "count"])
    b = pd.DataFrame({"d": dy, "x": y}).groupby("d").x.agg(["sum", "count"])
    j = a.join(b, how="outer", lsuffix="_a", rsuffix="_b").fillna(0.0)
    if len(j) < 3:
        return dict(lo=np.nan, hi=np.nan, lo_bonf=np.nan, p_le0=np.nan)
    rng = np.random.default_rng(seed); i = rng.integers(0, len(j), (B, len(j)))
    sa, ca, sb, cb = (j[k].values[i].sum(1) for k in ("sum_a", "count_a", "sum_b", "count_b"))
    d = sa / np.maximum(ca, 1) - sb / np.maximum(cb, 1)
    return dict(lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), lo_bonf=np.percentile(d, BONF_Q), p_le0=(d <= 0).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); A = ap.parse_args()
    T0 = time.time()
    buf = io.StringIO()

    def P(*a):
        print(*a); print(*a, file=buf)
    E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    thr = E[(E.spike >= C.B_YEAR0) & (E.spike < C.FREEZE_CUT)].FLAT14.quantile(1 / 3)
    assert abs(thr - 44.39) < 0.01, thr
    E = E[(E.spike >= C.B_YEAR0) & (E.FLAT14 <= thr)].copy()
    S0 = pd.read_pickle(os.path.join(C.D, "signals.pkl"))
    S0 = S0[S0.entry == "S4"].copy()
    fk = set(zip(E.pair, E.on_t))
    S0 = S0[[k in fk for k in zip(S0.pair, S0.ep_on_t)]].reset_index(drop=True)
    now_ms = pd.read_pickle(os.path.join(C.D, "meta.pkl"))["now_ms"]
    jobs = [(p, g, S0[S0.pair == p], now_ms, EXITS) for p, g in E.groupby("pair")]
    PARTS, SBS, NK, NNET, NTK = [], [], [], [], []
    with mp.Pool(A.procs) as pool:
        for res, sb in pool.imap_unordered(wk, jobs, chunksize=4):
            if res is None:
                continue
            pair, o, keys, nn, nt, ne = res
            if o is not None:
                PARTS.append(o); SBS.append(sb)
            NK += keys; NNET += nn; NTK += nt
    t_sim = time.time() - T0
    S = pd.concat([o["meta"] for o in PARTS], ignore_index=True)
    cat = lambda k: np.concatenate([o[k] for o in PARTS])
    NET, NOF, XT, RS, TK, MISS = cat("NET"), cat("NOF"), cat("XT"), cat("RS"), cat("TK"), cat("MISS")
    SB = np.concatenate(SBS)
    S["mtm"] = list(cat("MK"))
    keep = (S.entry == "S4").values                                            # drop X's D-entries
    S = S[keep].reset_index(drop=True); NET, NOF, XT, RS, TK, MISS, SB = (a[keep] for a in (NET, NOF, XT, RS, TK, MISS, SB))
    NNET = np.stack(NNET); NTK = np.stack(NTK)
    nrow = {k: i for i, k in enumerate(NK)}
    LMr = pd.read_csv(C.OUTP + "_live_master.csv", usecols=["pair", "spike", "real"])
    rmap = dict(zip(zip(LMr.pair, LMr.spike.astype(str).str[:16]), LMr.real))
    spk = dict(zip(zip(E.pair, E.on_t), E.spike.astype(str).str[:16]))
    S["real_ep"] = [bool(rmap.get((p, spk.get((p, k), "")), False)) for p, k in zip(S.pair, S.ep_on_t)]
    S["dt"] = pd.to_datetime(S.entry_ms, unit="ms"); S["day"] = S.dt.dt.strftime("%Y-%m-%d")
    S["per"] = np.select([S.dt >= C.B_LIVE0, S.dt >= C.B_MAST0, S.dt >= pd.Timestamp("2026-07-01"), S.dt >= pd.Timestamp("2026-01-01")],
                         ["1_live", "2_master", "2026H2", "2026H1"], "2025")
    S["fold"] = np.where(S.per == "2025", np.where(S.dt < pd.Timestamp("2025-07-01"), "2025H1", "2025H2"), S.per)
    S["nrow"] = [nrow.get(k, -1) for k in zip(S.pair, S.ep_on_t)]
    PER = {"2025H1": (S.fold == "2025H1").values, "2025H2": (S.fold == "2025H2").values, "2026H1": (S.per == "2026H1").values,
           "2026H2": (S.per == "2026H2").values, "OOS26": S.per.isin(["2026H1", "2026H2"]).values,
           "YEAR": S.per.isin(["2025", "2026H1", "2026H2"]).values,
           "2_master": ((S.per == "2_master") & S.real_ep).values, "1_live": ((S.per == "1_live") & S.real_ep).values}

    def sizing(m, j):
        idx = np.nonzero(m & TK[:, j])[0]
        if len(idx) < 20:
            return dict(notional=np.nan, maxdd_usd=np.nan, sum_usd=np.nan)
        x = NET[idx, j].astype(float)
        p1 = np.percentile(x, 1); notional = ACCOUNT * RISK / abs(p1) * 100
        order = idx[np.lexsort((S.pair.values[idx], S.entry_ms.values[idx]))]   # deterministic: time, then pair (pool order varies)
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

    def null_of(m, j):
        mm = m & TK[:, j]
        ks = sorted(set(S.nrow[mm][S.nrow[mm] >= 0]))
        if not ks:
            return None
        nv = NNET[ks, :, j].astype(float); nt = NTK[ks, :, j]
        return np.nansum(np.where(nt, nv, 0), 0) / np.maximum(nt.sum(0), 1)          # per-shuffle mean

    rows = []
    for j, nm in enumerate(EXN):
        tp, hd, s = EXITS[j]; jb = BASE[hd]
        r = dict(cell=nm, tp=tp, hold=hd, stop=s, baseline=(j == jb))
        for per, m in PER.items():
            mm = m & TK[:, j]; x = NET[mm, j].astype(float)
            r[f"{per}_N"] = len(x)
            if not len(x):
                continue
            sub = S[mm]
            mb = m & TK[:, jb]; y = NET[mb, jb].astype(float)
            lo, hi = boot_mean(sub.day.values, x)
            r.update({f"{per}_days": sub.day.nunique(), f"{per}_WR": (x > 0).mean() * 100, f"{per}_avg": x.mean(), f"{per}_lo": lo, f"{per}_hi": hi,
                      f"{per}_nofund": NOF[mm, j].mean(), f"{per}_worst": x.min(), f"{per}_stop_rate": (RS[mm, j] == 1).mean() * 100,
                      f"{per}_tp_rate": (RS[mm, j] == 3).mean() * 100, f"{per}_delta": x.mean() - y.mean() if len(y) else np.nan})
            if j != jb and len(y):
                bd = boot_delta(sub.day.values, x, S.day.values[mb], y)
                r.update({f"{per}_d_lo": bd["lo"], f"{per}_d_hi": bd["hi"], f"{per}_d_lo_bonf": bd["lo_bonf"], f"{per}_d_p": bd["p_le0"]})
            if per in HALVES + ["OOS26"]:
                nmn = null_of(m, j)
                if nmn is not None:
                    r[f"{per}_null"] = nmn.mean(); r[f"{per}_p_null"] = (nmn >= x.mean()).mean()
                    nb = null_of(m, jb)
                    r[f"{per}_null_delta"] = nmn.mean() - nb.mean() if nb is not None else np.nan
            if per in HALVES + ["OOS26", "YEAR"]:
                sz = sizing(m, j)
                r[f"{per}_notional"] = sz["notional"]; r[f"{per}_maxdd_usd"] = sz["maxdd_usd"]; r[f"{per}_sum_usd"] = sz["sum_usd"]
            if per == "OOS26" and j != jb:
                g = pd.Series(x).groupby(sub.pair.values).sum().sub(pd.Series(y).groupby(S.pair.values[mb]).sum(), fill_value=0)
                gd = pd.Series(x).groupby(sub.day.values).sum().sub(pd.Series(y).groupby(S.day.values[mb]).sum(), fill_value=0)
                tot = x.sum() - y.sum()
                r["OOS26_gain_sum"] = tot
                r["OOS26_top_pair_share"] = g.max() / tot * 100 if tot > 0 else np.nan
                r["OOS26_top_pair"] = g.idxmax()
                r["OOS26_top_day_share"] = gd.max() / tot * 100 if tot > 0 else np.nan
                r["OOS26_top_day"] = gd.idxmax()
        rows.append(r)
    CE = pd.DataFrame(rows)

    # ── verdict per 72 h non-baseline cell ──
    def verdict(r):
        ok = {}
        ok["every_half"] = all((r.get(f"{h}_delta", np.nan) > 0) for h in HALVES)
        ok["2026_bonf_lb"] = r.get("OOS26_d_lo_bonf", np.nan) > 0
        ok["null_p"] = r.get("OOS26_p_null", np.nan) < 0.05
        ok["conc"] = (r.get("OOS26_top_pair_share", np.nan) < 50) and (r.get("OOS26_top_day_share", np.nan) < 50)
        ok["haircut"] = r.get("OOS26_delta", np.nan) * 0.5 > 0
        return ok
    VER = {}
    for _, r in CE[(CE.hold == 72) & ~CE.baseline].iterrows():
        VER[r.cell] = verdict(r.to_dict())
    CE["candidate"] = [all(VER[c].values()) if c in VER else np.nan for c in CE.cell]

    f2 = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2f}"
    P(f"# FLATBASE TP3 / stop5 test — tables (S4 entries, FLAT cohort; sim {t_sim:.0f} s)")
    P(f"S4 signals {len(S)} · taken baseline 72 h: " + " · ".join(f"{p} {int((PER[p] & TK[:, BASE[72]]).sum())}" for p in PER))
    for hd in HOLDS:
        P(f"\n## {hd} h hold{' (PRIMARY)' if hd == 72 else ' (context only)'} — per half: N · WR · avg net % (funding incl.) · Δ vs TP−8/stop15 [95 % day-boot]")
        P("| cell | " + " | ".join(HALVES) + " | 2026 total (H1+H2) | 2026 Bonf. LB of Δ | 2026 null · p | ② master | ① live (study 10-14→10-22) |")
        P("|---|" + "---|" * (len(HALVES) + 5))
        for _, r in CE[CE.hold == hd].iterrows():
            cells = []
            for per in HALVES + ["OOS26"]:
                if not r.get(f"{per}_N"):
                    cells.append("0"); continue
                d = "" if r.baseline else f" · Δ {f2(r[f'{per}_delta'])} [{f2(r[f'{per}_d_lo'])}, {f2(r[f'{per}_d_hi'])}]"
                cells.append(f"{r[f'{per}_N']:.0f} · {r[f'{per}_WR']:.0f}% · {f2(r[f'{per}_avg'])}{d}")
            lb = "baseline" if r.baseline else f2(r.get("OOS26_d_lo_bonf"))
            nl = f"{f2(r.get('OOS26_null'))} · {r.get('OOS26_p_null', np.nan):.3f}"
            bl = []
            for per in ("2_master", "1_live"):
                bl.append(f"{r[f'{per}_N']:.0f} · {f2(r.get(f'{per}_avg'))}" + ("" if r.baseline or not r[f"{per}_N"] else f" (Δ {f2(r.get(f'{per}_delta'))})"))
            P(f"| {r.cell}{' **LIVE**' if r.baseline else ''} | " + " | ".join(cells) + f" | {lb} | {nl} | " + " | ".join(bl) + " |")
    P("\n## Detail per 72 h cell × half: pre-funding avg · worst trade · stop-out % · TP % · max drawdown at the study's sizing ($3,200; 1st-pct loss = 5 %)")
    P("| cell | period | N | avg | pre-funding | worst | stop % | TP % | $/short | sum $ | max DD $ (% acct) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in CE[CE.hold == 72].iterrows():
        for per in HALVES + ["OOS26", "YEAR"]:
            if not r.get(f"{per}_N"):
                continue
            dd = r.get(f"{per}_maxdd_usd", np.nan)
            P(f"| {r.cell} | {per} | {r[f'{per}_N']:.0f} | {f2(r[f'{per}_avg'])} | {f2(r[f'{per}_nofund'])} | {r[f'{per}_worst']:+.1f} | "
              f"{r[f'{per}_stop_rate']:.0f} | {r[f'{per}_tp_rate']:.0f} | {r.get(f'{per}_notional', np.nan):.0f} | {r.get(f'{per}_sum_usd', np.nan):+,.0f} | "
              f"{dd:+,.0f} ({dd / ACCOUNT * 100:+.0f} %) |")
    P("\n## Verdict vs the pre-registered bar (72 h cells)")
    P("| cell | Δ>0 every half | 2026 Bonf. LB > 0 | null p < 0.05 | no pair/day ≥ 50 % | haircut > 0 | CANDIDATE | 2026 gain top pair / day share |")
    P("|---|---|---|---|---|---|---|---|")
    for c, ok in VER.items():
        r = CE[CE.cell == c].iloc[0]
        P(f"| {c} | " + " | ".join("PASS" if v else "FAIL" for v in ok.values()) + f" | **{'YES' if all(ok.values()) else 'no'}** | "
          + (f"{r['OOS26_top_pair_share']:.0f} % ({r['OOS26_top_pair']}) / {r['OOS26_top_day_share']:.0f} % ({r['OOS26_top_day']}) |"
             if r.get("OOS26_gain_sum", 0) > 0 else f"no 2026 gain (Σ cell − Σ live = {r.get('OOS26_gain_sum', np.nan):+.0f} % pts) |"))

    # ── saved / cut split vs the baseline on signals taken by both (72 h) ──
    P("\n## Saved vs cut split (72 h, 2026 H1+H2 and the full year; signals taken by BOTH the cell and the baseline; Δ in % points)")
    P("| cell | period | both | live STOPPED: n · saved / same / deeper · ΣΔ | live TP: n · cut by stop / capped / same · ΣΔ | live TIME: n · ΣΔ | ΣΔ all |")
    P("|---|---|---|---|---|---|---|")
    jb = BASE[72]
    for j, nm in enumerate(EXN):
        if EXITS[j][1] != 72 or j == jb:
            continue
        for per in ("OOS26", "YEAR"):
            m = PER[per] & TK[:, j] & TK[:, jb]
            d = (NET[m, j] - NET[m, jb]).astype(float); rb = RS[m, jb]; rc = RS[m, j]
            st, tpm, tm = rb == 1, rb == 3, ~np.isin(rb, [1, 3])
            P(f"| {nm} | {per} | {m.sum()} | {st.sum()} · {(d[st] > 0.005).sum()} / {(abs(d[st]) <= 0.005).sum()} / {(d[st] < -0.005).sum()} · {d[st].sum():+.0f} | "
              f"{tpm.sum()} · {(tpm & (rc == 1)).sum()} / {(tpm & (rc != 1) & (d < -0.005)).sum()} / {(tpm & (abs(d) <= 0.005)).sum()} · {d[tpm].sum():+.0f} | "
              f"{tm.sum()} · {d[tm].sum():+.0f} | {d.sum():+.0f} |")

    # ── 5 % stop wick risk ──
    P("\n## 5 % stop wick risk (72 h cells with stop 5; every stop-out on the taken set, all periods)")
    P("| cell | stop-outs | gap fills (bar opened ≥ stop) | 5m close back below ENTRY (wick stop) | 5m close below the stop level | stop bar also touched the TP |")
    P("|---|---|---|---|---|---|")
    EVT = []
    for j, nm in enumerate(EXN):
        tp, hd, s = EXITS[j]
        if s != 5 or hd != 72:
            continue
        m = TK[:, j] & (RS[:, j] == 1) & ~np.isnan(SB[:, j, 3])
        Ev = S.E.values[m]; lvl = Ev * 1.05; so, sh, sl, sc = (SB[m, j, k] for k in range(4))
        P(f"| {nm} | {m.sum()} | {(so >= lvl).sum()} | {(sc < Ev).sum()} ({(sc < Ev).mean() * 100:.0f} %) | {(sc < lvl).sum()} ({(sc < lvl).mean() * 100:.0f} %) | "
          f"{(sl <= Ev * (1 - tp / 100)).sum()} |")
        if tp == 8:                                                              # the stop-5 events are the same bars for every TP until TP fires
            for i, k in enumerate(np.nonzero(m)[0]):
                EVT.append(dict(pair=S.pair[k], per=S.fold[k], E=S.E[k], lvl=lvl[i], bar=int(XT[k, j] - C.BAR), o=so[i], c=sc[i]))
    for s_ in (10, 15):
        j = EXN.index(f"tp8|72h|stop{s_}")
        m = TK[:, j] & (RS[:, j] == 1) & ~np.isnan(SB[:, j, 3]); Ev = S.E.values[m]
        P(f"| (ref) tp8|72h|stop{s_} | {m.sum()} | {(SB[m, j, 0] >= Ev * (1 + s_ / 100)).sum()} | {(SB[m, j, 3] < Ev).sum()} ({(SB[m, j, 3] < Ev).mean() * 100:.0f} %) | "
          f"{(SB[m, j, 3] < Ev * (1 + s_ / 100)).sum()} ({(SB[m, j, 3] < Ev * (1 + s_ / 100)).mean() * 100:.0f} %) | — |")
    # 1m re-check (locally cached 1m bars only)
    EV = pd.DataFrame(EVT)
    res1 = []
    for e in EV.itertuples():
        d = pd.Timestamp(e.bar, unit="ms").strftime("%Y-%m-%d")
        f = os.path.join(K1M, e.pair, d + ".npz")
        if not os.path.exists(f):
            continue
        try:
            a = np.load(f)["a"]
        except Exception:
            continue
        w = a[(a[:, 0] >= e.bar) & (a[:, 0] < e.bar + C.BAR)]
        if len(w) < 5:
            continue
        hit = w[:, 2] >= e.lvl
        res1.append(dict(pair=e.pair, per=e.per, bar=e.bar, n_hi=int(hit.sum()), close_at_stop=bool((w[:, 4] >= e.lvl).any()),
                         gap=bool(e.o >= e.lvl), wick5=bool(e.c < e.E), max1m_close_vs_lvl=float((w[:, 4].max() / e.lvl - 1) * 100),
                         hi_ok=bool(hit.any())))
    R1 = pd.DataFrame(res1)
    P(f"\n### 1m re-check of the 5 % stop bars (tp8|72h|stop5 stop-outs; locally cached 1m in {os.path.relpath(K1M, C.R)})")
    if len(R1):
        P(f"covered {len(R1)} of {len(EV)} stop-outs ({len(R1) / max(len(EV), 1) * 100:.0f} %) · 1m high reached the stop in {R1.hi_ok.sum()} "
          f"(data consistency) · a 1m CLOSE at/above the stop (price held there ≥ 1 min): {R1.close_at_stop.sum()} ({R1.close_at_stop.mean() * 100:.0f} %) · "
          f"ONLY 1m highs touched (pure wick, < 1 min above the stop): {(~R1.close_at_stop).sum()} ({(~R1.close_at_stop).mean() * 100:.0f} %) · "
          f"of the pure wicks, one single 1m bar above the stop: {((~R1.close_at_stop) & (R1.n_hi == 1)).sum()}")
        P(f"among the 5m 'wick stops' (5m close < entry) covered: {R1.wick5.sum()} · of which pure 1m wick: {(R1.wick5 & ~R1.close_at_stop).sum()}")
        P("by period: " + " · ".join(f"{p} {len(g)} covered, pure wick {(~g.close_at_stop).mean() * 100:.0f} %" for p, g in R1.groupby("per")))
    else:
        P("no 5 % stop bar is covered by locally cached 1m data — cannot be checked offline.")
    CE.to_csv(OUT + "_cells.csv", index=False)
    TW = S[["pair", "dt", "fold", "per", "real_ep", "E"]].copy()
    for j, nm in enumerate(EXN):
        if EXITS[j][1] != 72:
            continue
        TW[f"net|{nm}"] = np.where(TK[:, j], NET[:, j], np.nan).round(3)
        TW[f"reason|{nm}"] = np.where(TK[:, j], pd.Series(RS[:, j]).map({0: "time", 1: "stop", 3: "tp", 4: "OPEN(mtm)", 5: "settle"}).values, "")
    TW.sort_values("dt").to_csv(OUT + "_trades.csv", index=False)
    if len(R1):
        R1.to_csv(OUT + "_wick1m.csv", index=False)
    P(f"\nruntime {time.time() - T0:.0f} s")
    open(OUT + "_tables.md", "w").write(buf.getvalue())
    print("done")
