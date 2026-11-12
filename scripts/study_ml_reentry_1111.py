#!/usr/bin/env python3
"""ML RE-ENTRY (second impulse) study — MASTER (2026-11-11, operator: "yes run the re-entry study on master").

THEORY (from the impulse-end study, reports/ML_EXIT_IMPULSE_2026-11-11_*): today's exit already catches the end of the FIRST impulse as
early as 1m data allows; the "money on the table" is a SECOND impulse after a pause (66 % of winners trade ≥ +0.5 above our exit within
60 min, median +1.21 at 37 min). So: keep today's exit, RE-ENTER the same coin when it re-accelerates.

PRE-DECLARED (written before any re-entry was priced):
  cohort   C_WIN = re-entries after a WINNING exit (primary; the impulse was proven) · C_ALL = after any exit (secondary line)
  window   W ∈ {60, 120} min after the original's exit (today's live exit = the RP.BASE replica, parity 0.000 on these 123 fills)
  trigger  (1m CLOSED bars, known at close + 10 s, first bar opening at/after the exit; one re-entry per original fill)
    BREAK    1m close > the original trade's highest price between its entry and exit (new high = the next leg)
    REACCEL  1m EMA5 > EMA8 ∧ the EMA5−EMA8 gap rose on the last 2 closes ∧ close > 1m EMA20, AFTER a pause bar (a bar since the exit
             where EMA5 ≤ EMA8 or the gap fell) — the entry signal, on 1m, re-armed
    REACCEL_UP  REACCEL ∧ close ≥ our exit price
  entry    first pair print ≥ trigger + 8 s, + 0.10 % slip, taker fee 0.045 (scout_blocked_filters ruler)
  exit     today's ML exit (RP.BASE replica) on the re-entry, ATR = the original's entry ATR, BTC RSI ruler at the re-entry
  read     per variant: N · WR · avg % · Σ · per era group · day concentration · leave-one-out · day-block 95 % CI and Bonferroni (12
           variants) · the sleeve's ONE-SIZE DCR before (BASE) → after (BASE + re-entries; re-entries are ADDED, nothing displaced) ·
           collisions (re-entry still open when the next original ML fill on the same pair opened — live would refuse that fill).
  blind spots: today's entry gates (ML_MKT_HOT, ML_BULL90, regime bands, heat, max-open slots) are NOT re-checked at the re-entry; the live
           ML stop-cooldown (30 min after a STOP) is not modelled for C_ALL.
Writes reports/ML_REENTRY_2026-11-11_{fills,summary}.csv + .txt
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study_ml_btcexit_run as BX  # noqa: E402,F401
import study_ml_exit_macro_1111 as S  # noqa: E402
import study_ml_exit_macro_1111_report as R  # noqa: E402
import scout_ml_matched_refusals as MM  # noqa: E402
import review_math as RM  # noqa: E402

OUT = os.path.join(S.ROOT, "reports", "ML_REENTRY_2026-11-11")
MIN, LAT, ENTRY_LAT, SLIP = 60_000, 10_000, 8_000, 0.10
WINDOWS = (60, 120)
TRIGS = ("BREAK", "REACCEL", "REACCEL_UP")


def triggers(RP, pair, o_ms, x_ms, hi_px, x_px):
    """→ {(trig, W): trigger known ms or None}"""
    k1 = RP._k1(pair)
    out = {(g, w): None for g in TRIGS for w in WINDOWS}
    if k1 is None:
        return out
    ot = k1.open_time.values.astype(np.int64)
    a, b = np.searchsorted(ot, x_ms - 90 * MIN), np.searchsorted(ot, x_ms + max(WINDOWS) * MIN)
    d = k1.iloc[a:b]
    if len(d) < 40:
        return out
    ot, c = d.open_time.values.astype(np.int64), d.c.values.astype(float)
    e5 = pd.Series(c).ewm(span=5, adjust=False).mean().values
    e8 = pd.Series(c).ewm(span=8, adjust=False).mean().values
    e20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    g = e5 - e8
    dg = np.r_[0, np.diff(g)]
    up2 = (dg > 0) & (np.r_[0, dg[:-1]] > 0)
    sig = (e5 > e8) & up2 & (c > e20)
    pause = (e5 <= e8) | (dg < 0)
    kn = ot + MIN + LAT
    after = ot >= (x_ms // MIN) * MIN                        # bars opening at / after the exit minute
    for w in WINDOWS:
        inw = after & (kn <= x_ms + w * MIN)
        idx = np.flatnonzero(inw)
        if not len(idx):
            continue
        br = idx[c[idx] > hi_px]
        out[("BREAK", w)] = int(kn[br[0]]) if len(br) else None
        p = idx[pause[idx]]
        if len(p):
            cand = idx[(idx > p[0]) & sig[idx]]
            out[("REACCEL", w)] = int(kn[cand[0]]) if len(cand) else None
            cu = idx[(idx > p[0]) & sig[idx] & (c[idx] >= x_px)]
            out[("REACCEL_UP", w)] = int(kn[cu[0]]) if len(cu) else None
    return out


def main():
    C, _ = S.cohort()
    V = pd.read_csv(os.path.join(S.ROOT, "reports", "ML_EXIT_MACRO_2026-11-11_variants.csv"))
    B = V[V.variant == "BASE"].set_index("key")
    C = C[C.key.isin(B.index)].copy()
    C["base_pct"] = C.key.map(B.pct)
    C["base_x"] = C.key.map(B.exit_ms).astype(np.int64)
    now = int(time.time() * 1000)
    trig_rows = []
    with MM.replica_with_topup() as RP:
        # ① triggers from each original's path (entry → BASE exit)
        for r in C.itertuples():
            fx = dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in), atr=None, rsi_entry=None)
            P = RP.prep_path(fx, int(r.base_x - r.o_ms) + 2 * MIN)
            if P is None:
                continue
            k = int(np.searchsorted(P["t"], r.base_x))
            k = min(k, len(P["t"]) - 1)
            hi_px, x_px = float(np.max(P["p"][:k + 1])), float(P["p"][k])
            for (g, w), tk in triggers(RP, r.pair, int(r.o_ms), int(r.base_x), hi_px, x_px).items():
                if tk is not None:
                    trig_rows.append(dict(key=r.key, pair=r.pair, trig=g, W=w, t_trig=tk, hi_px=hi_px, x_px=x_px, atr=r.atr))
        T = pd.DataFrame(trig_rows)
        # ② top up paths for the re-entries (20 h horizon from each)
        fills = [dict(key=f"{x.key}|{x.trig}|{x.W}", pair=x.pair, o_ms=int(x.t_trig + ENTRY_LAT)) for x in T.itertuples()]
        uniq = {(f["pair"], f["o_ms"] // (6 * 3600_000)): f for f in fills}
        for i in range(12):
            bud = MM.Budget(max_calls=200)
            MM.ensure_paths(list(uniq.values()), now, bud, RP.HORIZON_MS)
            print(f"top-up {i}: calls {bud.n} stop={bud.stop}", flush=True)
            if bud.n == 0:
                break
            if bud.stop and "weight" in str(bud.stop):
                time.sleep(61)
            RP._K1.clear(); RP._K5.clear(); RP._BTC = None
        # ③ price each re-entry with today's exit
        out = []
        for x in T.itertuples():
            te = int(x.t_trig + ENTRY_LAT)
            bp = RP.build_path(x.pair, te, 5 * MIN)
            if bp is None or not len(bp[0]):
                continue
            E = float(bp[1][0]) * (1 + SLIP / 100)
            rsi = RP.btc_rsi_at(te)
            fx = dict(key=f"{x.key}|{x.trig}|{x.W}", pair=x.pair, o_ms=te, E=E, fee_in=RP.TAKER,
                      atr=float(x.atr) if (x.atr == x.atr and x.atr > 0) else None, rsi_entry=None if rsi is None else float(rsi))
            k, why, pct, tex, src, pk = RP.run_fill(fx, {"BASE": RP.BASE})["BASE"]
            out.append(dict(orig=x.key, pair=x.pair, trig=x.trig, W=x.W, t_re=te, E=E, re_pct=pct, re_reason=why, re_exit=tex, src=src,
                            ent_vs_exit=(E / x.x_px - 1) * 100))
    Rf = pd.DataFrame(out).merge(C[["key", "era", "o_ms", "base_pct", "base_x"]], left_on="orig", right_on="key")
    # collisions: the re-entry still open when the NEXT original ML fill on the same pair opened
    nxt = C.sort_values("o_ms").groupby("pair").o_ms.shift(-1)
    Rf["next_open"] = Rf.orig.map(dict(zip(C.key, nxt)))
    Rf["collide"] = Rf.next_open.notna() & (Rf.re_exit > Rf.next_open)
    Rf.to_csv(OUT + "_fills.csv", index=False)
    report(C, Rf)


def report(C, Rf):
    L = []
    p = L.append
    C = C.copy()
    C["grp"] = C.era.map(R.GROUPS)
    C["day"] = pd.to_datetime(C.o_ms, unit="ms").dt.strftime("%Y-%m-%d")
    Rf = Rf.copy()
    Rf["grp"] = Rf.era.map(R.GROUPS)
    Rf["day"] = pd.to_datetime(Rf.t_re, unit="ms").dt.strftime("%Y-%m-%d")
    win = C.set_index("key").base_pct > 0
    Rf["after_win"] = Rf.orig.map(win)
    eras = sorted(C.era.unique())
    base_dcr, nd = R.dcr_group(C.assign(pp=C.base_pct, ex=C.base_x, t0=C.t0, t1=C.t1), "pp", "ex", eras)
    p(f"# ML RE-ENTRY (second impulse) — master, {len(C)} kept ML fills ({int(win.sum())} winners) · today's exit BASE avg "
      f"{C.base_pct.mean():+.3f} · one-size DCR {base_dcr:+.3f} %/day over {nd} days (X {R.X})\n")
    p(f"priced re-entries: {Rf.re_pct.notna().sum()} / {len(Rf)} triggers · END_OF_DATA {int((Rf.re_reason == 'END_OF_DATA').sum())}\n")
    K = 12
    rows = []
    for coh, mcoh in (("C_WIN", Rf.after_win == True), ("C_ALL", Rf.after_win.notna())):  # noqa: E712
        for g in TRIGS:
            for w in WINDOWS:
                s = Rf[mcoh & (Rf.trig == g) & (Rf.W == w) & Rf.re_pct.notna()]
                if not len(s):
                    continue
                y = s.re_pct.values
                lo, hi, _ = R.boot_ci(y, s.day.values, 0.05)
                lB, hB, _ = R.boot_ci(y, s.day.values, 0.05 / K)
                add = C[["key", "era", "base_pct", "base_x", "t0", "t1"]].rename(columns={"base_pct": "pp", "base_x": "ex"})
                extra = s.merge(C[["key", "t0", "t1"]], left_on="orig", right_on="key", suffixes=("", "_c"))
                extra = extra.assign(pp=extra.re_pct, ex=extra.re_exit)[["era", "pp", "ex", "t0", "t1"]]
                dcr_after, _ = R.dcr_group(pd.concat([add, extra]), "pp", "ex", eras)
                dd = pd.Series(y).groupby(s.day.values).sum()
                lo_ = RM.loo(list(y), list(s.pair + " " + s.day))
                rows.append(dict(cohort=coh, trig=g, W=w, n=len(s), days=s.day.nunique(), wr=(y > 0).mean(), avg=y.mean(), sum=y.sum(),
                                 lo95=lo, hi95=hi, loBonf=lB, dcr_before=base_dcr, dcr_after=dcr_after,
                                 top_day=(dd.max() / y.sum() if y.sum() > 0 else np.nan), sum_wo_top=lo_["without_top"],
                                 collide=int(s.collide.sum()), stop_share=s.re_reason.astype(str).str.contains("STOP").mean(),
                                 **{f"avg_{k}": s[s.grp == k].re_pct.mean() for k in sorted(Rf.grp.dropna().unique())},
                                 **{f"n_{k}": int((s.grp == k).sum()) for k in sorted(Rf.grp.dropna().unique())}))
    S_ = pd.DataFrame(rows)
    S_.to_csv(OUT + "_summary.csv", index=False)
    p(S_.round(3).to_string(index=False))
    open(OUT + ".txt", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
