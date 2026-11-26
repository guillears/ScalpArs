#!/usr/bin/env python3
"""DIST_TRAIL1 — early ATR trail on DISTRIBUTION_SHORT (operator 2026-11-26: "almost all distribution shorts have been positive before
ending really bad … evaluate a trailing, starting at 1 %, based on ATR, similar to momentum longs").

PRIOR STUDY (DIST_EXIT, DL 595-ish, reports/DIST_EXIT_2026-11-18.md, same exit era = Exit B, still live): trail arm {2, 3, 4} % ×
give-back {1, 1.5, 2, 3} % and arm 3 % × {1, 2} × ATR all LOSE to Exit B (DESIGN and HOLDOUT, white-reality t ≤ 0). NEW here: arm
below 2 % (+1 / +1.5 %) and ATR give-backs 0.5–2 × ATR15 at those arms (the momentum-long trail uses an ATR-floored pull-back, 0.5 ×).
Re-open note in that report: in the BTC-daily-bull HOLDOUT a trail sat +0.07…+0.09 above Exit B; the current batch's DIST is losing.

PRE-REGISTERED (frozen before any number of this run is read):
  cohort  = DIST_EXIT PRIMARY (DIST_WL rows, RSI1h ≥ 74.07 keep cut), one position per pair per cell, Exit B = REF_B.
  cells   = REF_B · trailatr arm {1, 1.5, 2} × {0.5, 1, 1.5, 2} × ATR15 · trail arm 1 × {0.5, 1, 1.5, 2} % · D_T3_A2 (parity with 11-18).
            Exit B otherwise unchanged (TP min(20 × ATR, 10 %) maker, SL min(1.5 × ATR, 6 %), 24 h); trail stop = market order.
  costs   = DIST_EXIT net_mat (fee 0.05 + 0.05 taker / 0.02 maker, slip max(0.10, model) both legs for market exits, real funding).
  primary = paired Δ (cell − REF) on the rows REF keeps (mechanism-only; never-armed rows must show Δ = 0, asserted).
  SHIP CANDIDATE only if ALL: DESIGN Δ > 0 with 10-day-block bootstrap 95 % lower bound > 0 · HOLDOUT Δ > 0 · ≥ 5 / 7 half-years Δ > 0 ·
            the RND control does NOT show the same Δ (else it is a generic exit effect) · one-size DCR and max DD not worse on HOLDOUT.
            With 17 cells, a single DESIGN pass is luck-prone: the HOLDOUT read is the decider and is read once.
  Batch   = the real DIST fills of the export (1m klines, real fill price, the fill's own ATR stop) → same cells, descriptive.
Usage: venv/bin/python -I scripts/DIST_TRAIL1_2026-11-26.py [--procs 8] [--export orders.csv]
Out:   reports/DIST_TRAIL1_2026-11-26{.md,_grid.csv,_halves.csv,_batch.csv}"""
import argparse, json, os, sys, time, urllib.request
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dist_exit_sim as DS      # noqa: E402
import dist_exit_eval as EV     # noqa: E402
import dist_wl_survivors as SV  # noqa: E402
R = DS.R; RP = os.path.join(R, "reports"); OUT = os.path.join(RP, "DIST_TRAIL1_2026-11-26")
DAY, H1, MIN = 86_400_000, 3_600_000, 60_000
SIZE = 0.61


def cells():
    C = [dict(name="REF_B", fam="REF", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="none", arm=np.inf, par=0.0, hold=24)]
    for a in (1, 1.5, 2):
        for m in (0.5, 1, 1.5, 2):
            C.append(dict(name=f"ATR_arm{a:g}_x{m:g}", fam="TA", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="trailatr", arm=float(a), par=float(m), hold=24))
    for g in (0.5, 1, 1.5, 2):
        C.append(dict(name=f"PCT_arm1_g{g:g}", fam="TP", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="trail", arm=1.0, par=float(g), hold=24))
    C.append(dict(name="D_T3_A2", fam="PAR", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="trailatr", arm=3.0, par=2.0, hold=24))
    return C


CL = cells(); NAMES = [c["name"] for c in CL]; NC = len(CL)


def net(Z, slip):
    k = Z["k"]; tk = k != 0
    s = slip[:, None] * np.ones((1, NC))
    v = Z["r"].astype(float) - s - np.where(tk, s, 0.0) - 0.05 - np.where(tk, 0.05, 0.02) + Z["f"].astype(float)
    return np.where((~Z["ok"][:, None]) | (k == 9) | (k < 0), np.nan, v)


def work(a):
    p, rows = a
    out, _ = DS.run_rows(p, rows, CL, keep_path=None)
    return rows.index.values, out


def simulate(F, procs):
    n = len(F)
    res = dict(r=np.full((n, NC), np.nan, np.float32), k=np.full((n, NC), -1, np.int8), xm=np.zeros((n, NC), np.int64),
               f=np.zeros((n, NC), np.float32), armhit=np.zeros((n, NC), np.int8), reach=np.zeros((n, NC), np.float32), ok=np.zeros(n, bool))
    jobs = [(p, F[F.pair == p]) for p in F.pair.unique()]; jobs.sort(key=lambda j: -len(j[1]))
    pos = {ix: i for i, ix in enumerate(F.index.values)}
    with mp.get_context("fork").Pool(procs) as pool:
        for ix, out in pool.imap_unordered(work, jobs, chunksize=2):
            ii = [pos[x] for x in ix]
            for kk in res:
                res[kk][ii] = out[kk]
    return res


def halves(T):
    return EV.half_of(T)


def batch(export):
    d = pd.read_csv(export, low_memory=False)
    x = d[(d.entry_strategy == "DISTRIBUTION_SHORT") & (d.status == "CLOSED")].copy()
    rows = []
    for _, r in x.iterrows():
        T0 = pd.Timestamp(r.opened_at).value // 10**6
        Tm = T0 // MIN * MIN
        kl = []; st = Tm
        while st < Tm + 26 * H1:
            u = f"https://fapi.binance.com/fapi/v1/klines?symbol={r.pair}&interval=1m&startTime={st}&limit=1500"
            kk = None
            for _try in range(4):
                try:
                    kk = json.load(urllib.request.urlopen(u, timeout=20)); break
                except Exception:
                    time.sleep(2)
            if not kk:
                break
            kl += kk; st = kk[-1][0] + MIN; time.sleep(0.3)
        k = np.array([[float(v) for v in z[:5]] for z in kl])
        t, o, h, l, c = k[:, 0].astype(np.int64), k[:, 1].copy(), k[:, 2].copy(), k[:, 3].copy(), k[:, 4].copy()
        assert len(kl) and int(kl[0][0]) == Tm, f"{r.pair}: no 1m bar at the entry minute"
        E = float(r.entry_price); o[0] = E
        # the entry bar = the fill price only (the bar's earlier part is before the fill; a stop inside the fill minute is missed — neutral
        # for the Δ, slightly flatters every cell in absolute terms). No funding term on the batch (the backtest has it).
        h[0] = l[0] = c[0] = E
        atrp = float(r.entry_dist_atr_pct)
        CC = DS.cell_arrays(CL)
        tick = DS.NS.tick_of(o, h, l, c)
        rr, kk_, xm, *_ = DS.sim_trade(Tm, t, o, h, l, c, atrp, 0.10, tick, CC, 1)
        s = 0.10
        v = rr - s - np.where(kk_ != 0, s, 0) - 0.05 - np.where(kk_ != 0, 0.05, 0.02)
        rows.append(dict(pair=r.pair, opened=r.opened_at, booked=float(r.pnl_percentage), peak=float(r.peak_pnl), trough=float(r.trough_pnl),
                         close=r.close_reason, **{NAMES[j]: float(v[j]) for j in range(NC)}, **{"k_" + NAMES[j]: int(kk_[j]) for j in range(NC)}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--export", default="")
    A = ap.parse_args()
    S = pd.read_pickle(os.path.join(RP, "DIST_WL_2026-11-18_data", "rows.pkl"))
    F = DS.frames()["L1"]
    prim = SV.keep_mask(S, EV.REP)
    Fp = F[prim].copy()
    t0 = time.time(); Z = simulate(Fp, A.procs); print("sim", f"{time.time() - t0:.0f}s", int(Z["ok"].sum()), "of", len(Fp), flush=True)
    Rn = DS.frames()["RND"]; ZR = simulate(Rn, A.procs); print("rnd done", flush=True)
    V = net(Z, Fp.slip.values); VR = net(ZR, Rn.slip.values)
    pair = Fp.pair.values; T = Fp.entry_ms.values.astype(np.int64); day = T // DAY
    full = (T >= EV.DES[0]) & (T < EV.HOL[1]); des = T < EV.DES[1]
    keep = {j: EV.one_pos(pair, T, Z["xm"][:, j], full & np.isfinite(V[:, j])) for j in range(NC)}
    ref = keep[0]
    # assert: never-armed rows Δ = 0
    for j in range(1, NC):
        na = ref & (Z["armhit"][:, j] == 0) & np.isfinite(V[:, j])
        assert np.nanmax(np.abs(V[na, j] - V[na, 0])) < 1e-4, NAMES[j]
    H = halves(T); HN = [h[0] for h in EV.HALVES]
    G = []; HV = []
    for j in range(NC):
        m = ref & np.isfinite(V[:, j]); dlt = V[:, j] - V[:, 0]
        o = dict(cell=NAMES[j])
        for nm, sp, sel in (("DES", EV.DES, des), ("HOL", EV.HOL, ~des)):
            mm = m & sel
            o[f"{nm}_dN"] = int(mm.sum()); o[f"{nm}_d"] = dlt[mm].mean()
            lo, hi, _ = EV.block_boot_mean(day[mm], dlt[mm], sp, nb=2000, rng=np.random.default_rng(1126))
            o[f"{nm}_lo"], o[f"{nm}_hi"] = lo, hi
            kc = keep[j] & sel
            st = EV.cell_stats(V[kc, j], Z["xm"][kc, j], day[kc], Z["k"][kc, j], sp)
            for kk in ("N", "WR", "avg", "tp", "stop", "dyn", "dcr", "maxdd", "worst"):
                o[f"{nm}_{kk}"] = st.get(kk, np.nan)
        hv = {h: dlt[m & (H == h)].mean() for h in HN}
        o["halves_pos"] = int(sum(v > 0 for v in hv.values() if np.isfinite(v)))
        HV.append(dict(cell=NAMES[j], **hv))
        # saved vs cut, on REF fills: REF stopped (k=1) → Δ>0 saved; REF TP (k=0) → Δ<0 cut
        st_ = m & (Z["k"][:, 0] == 1); tp_ = m & (Z["k"][:, 0] == 0)
        o["saved_n"] = int((st_ & (dlt > 1e-6)).sum()); o["saved_sum"] = dlt[st_ & (dlt > 1e-6)].sum()
        o["deeper_sum"] = dlt[st_ & (dlt < -1e-6)].sum()
        o["cut_tp_n"] = int((tp_ & (dlt < -1e-6)).sum()); o["cut_tp_sum"] = dlt[tp_ & (dlt < -1e-6)].sum()
        o["other_sum"] = dlt[m & ~st_ & ~tp_].sum()
        # RND control (all draws, mechanism Δ)
        mr = np.isfinite(VR[:, j]) & np.isfinite(VR[:, 0])
        o["RND_d"] = (VR[mr, j] - VR[mr, 0]).mean()
        G.append(o)
    G = pd.DataFrame(G); HVd = pd.DataFrame(HV)
    G.to_csv(OUT + "_grid.csv", index=False); HVd.to_csv(OUT + "_halves.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(G.round(3).to_string()); print(HVd.round(2).to_string())
    if A.export:
        B = batch(A.export); B.to_csv(OUT + "_batch.csv", index=False); print(B.round(2).to_string())


if __name__ == "__main__":
    main()
