#!/usr/bin/env python3
"""🎯 Oct-8 study (+3 vs +4 tick check) — build the four cohorts (never mixed) → scratch pickles + a missing-tick pair-day list.

(a) yr5 replay FRENZY_LONG + FRENZY_WIDE kept fills under TODAY's entry stack, exactly as scripts/study_yr5_halves_today.build_replay()
    builds them (WIDE→LONG move of the 2.5–3.0 red setups, WIDE hold-green streak > 12, bearish-day block, gvol gate as replayed) — 3 seeds.
    Signal close = opened_at floored to the 5-min grid (replay opens 12–16 s after the close).
(b) the Oct-4/5 tick cohort (frenzy_gvr_trades = scratch frenzy_gvr_exits.csv, gvr < 1 = the 850 quiet-market FRENZY first candles of
    frenzy_exit_ticks_botexact.py / frenzy_trail_v2.py); t = the signal close. Today's-stack flags: LONG = ATR ≤ 3.0 ∧ red/flat; WIDE =
    green ∧ ATR ≤ 3.0 ∧ above_streak > 12 (engine cohort join, unknown → refused, as the yr5 builder); bearish-day block (BTC rebuild).
(c) the FRENZY_LITE study cohort (scratch lite_streak/kept.pkl K[12], 724 fills, entry_ts = signal close) + bearish-day block.
(d) live master FRENZY-family fills (reports/MASTER_POOL_stacked.csv, entry_strategy FRENZY_*; B18 included if present).
Usage: venv/bin/python scripts/study_tp34_cohorts.py"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import study_tp34_common as C                                   # noqa: E402
import study_yr5_halves_btc as BT                               # noqa: E402

SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
OUT = os.path.join(SCR, "tp34"); os.makedirs(OUT, exist_ok=True)
REP = os.path.join(ROOT, "reports")


def ms(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:23].str.replace("T", " "), format="mixed").values.astype("datetime64[ms]").astype("int64")


def cohort_a():
    import study_yr5_halves_today as YH
    F = YH.build_replay(False)
    z = F[F.S.isin(["FRENZY_LONG", "FRENZY_WIDE"]) & F.keep].copy()
    z["sig"] = (z.tms // 300_000) * 300_000
    z["lag_replay_s"] = (z.tms - z.sig) / 1000
    a = pd.DataFrame(dict(cohort="a", seed=z.seed.values, pair=z.pair.values, sig=z.sig.values, sleeve=z.S.values,
                          rep_pct=z.pct.values, rep_pk=z.pk.values, rep_reason=z.close_reason.values, rep_E=z.entry_price.astype(float).values,
                          rep_open=z.tms.values, rep_close=ms(z.closed_at), atr=pd.to_numeric(z.entry_atr_pct, errors="coerce").values,
                          bar_ret=pd.to_numeric(z.entry_frenzy_bar_ret_pct, errors="coerce").values,
                          usd_per_pct=(YH.BOOK * YH.SLOT * z.inv_today * z.lev_today * z.fr / 100).values,
                          rep_usd=z.usd.values, rep_pct_today=z.pct_today.values))
    return a


def cohort_b():
    Y = pd.read_csv(os.path.join(SCR, "frenzy_gvr_exits.csv"))
    Y = Y[Y.gvr < 1.0].reset_index(drop=True)
    Cc = pd.read_csv(os.path.join(REP, "FRENZY_ENGINE_COHORT_2026-10-05.csv"))[["pair", "t_signal_close", "above_streak"]].sort_values("t_signal_close")
    Z = Y[["pair", "t"]].reset_index().sort_values("t")
    J = pd.merge_asof(Z, Cc, left_on="t", right_on="t_signal_close", by="pair", direction="backward", tolerance=15 * 60 * 1000).set_index("index")
    Y["above_streak"] = J.above_streak.reindex(Y.index)
    R = BT.btc_at(Y.t.values, "closed")
    Y["bear"] = [BT.bearish(a, b) is True for a, b in zip(R.ret1d, R.gap)]
    red = Y.body <= 0
    Y["today_long"] = red & (Y.atr <= 3.0)
    Y["today_wide"] = ~red & (Y.atr <= 3.0) & (Y.above_streak > 12)
    Y["today_keep"] = (Y.today_long | Y.today_wide) & ~Y.bear
    b = pd.DataFrame(dict(cohort="b", seed=0, pair=Y.pair, sig=Y.t.astype("int64"), sleeve=np.where(Y.fz, "FRENZY", "WIDE"),
                          atr=Y.atr, body=Y.body, gvr=Y.gvr, above_streak=Y.above_streak, bear=Y.bear, today_long=Y.today_long,
                          today_wide=Y.today_wide, today_keep=Y.today_keep, gvrcsv_fix3=Y["fixed +3 / −3"], gvrcsv_fix4=Y["fixed +4 / −3"]))
    return b


def cohort_c():
    K = pd.read_pickle(os.path.join(SCR, "lite_streak", "kept.pkl"))["K"][12].copy()
    R = BT.btc_at(K.entry_ts.values, "closed")
    K["bear"] = [BT.bearish(a, b) is True for a, b in zip(R.ret1d, R.gap)]
    c = pd.DataFrame(dict(cohort="c", seed=0, pair=K.pair.values, sig=K.entry_ts.astype("int64").values, sleeve="FRENZY_LITE",
                          bear=K.bear.values, pri=K.PRI.values, why_pri=K.why_pri.values))
    return c


def cohort_d():
    M = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
    f = M[M.entry_strategy.astype(str).str.startswith("FRENZY") & (M.status.astype(str).str.upper() == "CLOSED")].copy()
    f["o_ms"] = ms(f.opened_at); f["c_ms"] = ms(f.closed_at)
    d = pd.DataFrame(dict(cohort="d", seed=0, pair=f.pair.values, sig=(f.o_ms.values // 300_000) * 300_000, sleeve=f.entry_strategy.values,
                          live_open=f.o_ms.values, live_close=f.c_ms.values, live_E=f.entry_price.astype(float).values,
                          live_exit_px=f.exit_price.astype(float).values, live_pct=f.pnl_percentage.astype(float).values,
                          live_peak=pd.to_numeric(f.peak_pnl, errors="coerce").values, live_reason=f.close_reason.values,
                          batch=f.get("batch", pd.Series([""] * len(f))).values))
    return d


def main():
    A, B, Cc, D = cohort_a(), cohort_b(), cohort_c(), cohort_d()
    for k, v in dict(a=A, b=B, c=Cc, d=D).items():
        v.to_pickle(os.path.join(OUT, f"cohort_{k}.pkl"))
        print(k, len(v), v.sleeve.value_counts().to_dict())
    need = set()
    for v in (A, B, Cc, D):
        for p, s in zip(v.pair, v.sig):
            for ds in C.days_for(int(s), int(s) + C.HOLD + 10 * C.MIN):
                if not C.has_day(p, ds):
                    need.add((p, ds))
    nd = pd.DataFrame(sorted(need), columns=["pair", "date"])
    nd.to_csv(os.path.join(OUT, "missing_pairdays.csv"), index=False)
    print("missing pair-days:", len(nd))
    for k, v in dict(a=A, b=B, c=Cc, d=D).items():
        miss = [any((p, ds) in need for ds in C.days_for(int(s), int(s) + C.HOLD + 10 * C.MIN)) for p, s in zip(v.pair, v.sig)]
        print(k, "fills with a missing day:", int(np.sum(miss)), "of", len(v))


if __name__ == "__main__":
    main()


def cohort_a0():
    """(a0) the ORIGINAL yr5 FRENZY_LONG fills (replay label, no today's-stack re-coding / blocks) — the cohort the halves report's
    waterfall step 'TP +4 → +3: +0.174 → +0.011' was computed on."""
    import study_yr5_halves_today as YH
    F = YH.build_replay(False)
    z = F[F.sleeve == "FRENZY_LONG"].copy()
    z["sig"] = (z.tms // 300_000) * 300_000
    return pd.DataFrame(dict(cohort="a0", seed=z.seed.values, pair=z.pair.values, sig=z.sig.values, sleeve="FRENZY_LONG_orig",
                             rep_pct=z.pct.values, rep_pk=z.pk.values, rep_reason=z.close_reason.values, rep_E=z.entry_price.astype(float).values,
                             rep_open=z.tms.values, kept_today=z.keep.values))
