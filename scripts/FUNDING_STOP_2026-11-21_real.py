#!/usr/bin/env python3
"""FUNDING_STOP_2026-11-21 — REAL fills: arm A (price stop, today) vs arm B (price + accrued funding stop) vs Bp (paid-only), per sleeve.

Cohorts (CLOSED only, no probes — no probe column / *_PROBE strategy in any of these sleeves' rows; dedup (opened_at[:19], pair, direction)):
  FLATBASE_SHORT  = reports/TP_GRID_2026-11-20_FB_data/cohort.csv (master stack_keep on today's stack + B28 batch, 31) + KERNELUSDT #25 added by
                    hand (opened 2026-11-20 20:00:35 at 0.04419, stopped 11-21 01:07:22 at 0.04861, booked −11.69 % incl. ≈ −$108 funding on
                    $6,777; not in the newest export). Entries: REAL fills = the fill price, walk from the minute AFTER the fill; fills before the
                    FB_CONFIRM deploy = the master's confirm entry rebuilt on 1m (TP_GRID_2026-11-20_FB_real.entry_of, +0.25 % drift), walk from
                    the entry bar. TP 5 / SL 10 / 72 h. Data: TP_GRID FB k1m + funding (KERNEL: FUNDING_STOP_2026-11-21_data).
  EXHAUSTION_SHORT / DISTRIBUTION_SHORT = the 5 + 5 closed fills of exports_archive/scalpars_orders_paper_2026-11-20_16-01-39.csv (the master
                    holds none), each with its OWN stamped TP / SL (entry_exh_* / entry_dist_*), 24 h. Data: TP_GRID EXH real 1m, DIST_FB_LOSERS
                    k1m, KERNEL 1m + all funding from FUNDING_STOP_2026-11-21_data (the DIST TP_GRID funding cache was empty).
Costs (primary, live): entry taker 0.05 % · maker TP 0.02 % filled only when the low trades ≥ 0.05 % through · stop / time taker 0.05 % + slip
  (FB 0.25 % = the real FB slip used in TP_GRID FB; EXH / DIST 0.10 %) · funding mark-scaled at each settlement.
Parity (arm A vs booked): paper costs 0.045 / 0.018 %, TP ≥ 1 tick through, the row's own exit_slippage_pct (else 0), funding only on rows that
  fold it into pnl (FB master pre-11-14 = price only, as the stack_pct re-price) → mean |gap| reported.
Output: reports/FUNDING_STOP_2026-11-21_data/real_rows.csv
Usage: venv/bin/python -I scripts/FUNDING_STOP_2026-11-21_real.py"""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


C = _imp("FSC", "FUNDING_STOP_2026-11-21_common.py")
FBR = _imp("FBR", "TP_GRID_2026-11-20_FB_real.py")
OUT = os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data")
FBD = os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_FB_data")
EXPORT = os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-11-20_16-01-39.csv")
HR, M1 = 3_600_000, 60_000


def k_kernel():
    k = pd.read_csv(os.path.join(OUT, "k1m_KERNELUSDT.csv"))
    return k.t.values.astype("int64"), k.o.values, k.h.values, k.l.values, k.c.values


def k1m(pair, src):
    if pair == "KERNELUSDT":
        return k_kernel()
    if src == "FB":
        k = pd.read_csv(os.path.join(FBD, "k1m", f"{pair}.csv"))
        return k.open_time.values.astype("int64"), k.o.values, k.h.values, k.l.values, k.c.values
    if src == "DIST":
        k = pd.read_csv(os.path.join(ROOT, "reports", "DIST_FB_LOSERS_2026-11-20_data", "k1m", f"{pair}.csv"))
        return k.open_time.values.astype("int64"), k.o.values, k.h.values, k.l.values, k.c.values
    d = os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_EXH_data", "real")
    f = [x for x in os.listdir(d) if x.startswith(pair + "_") and x.endswith("_1m.csv")][0]
    k = pd.read_csv(os.path.join(d, f))
    return k.t.values.astype("int64"), k.o.values, k.h.values, k.l.values, k.c.values


def fund_series(pair, src):
    f = os.path.join(OUT, f"funding_{pair}.csv")
    if src == "FB" and pair != "KERNELUSDT":
        f = os.path.join(FBD, "funding", f"{pair}.csv")
    x = pd.read_csv(f).sort_values("t")
    return x.t.values.astype("int64"), x.rate.values.astype(float)


def fb_cohort():
    R = pd.read_csv(os.path.join(FBD, "cohort.csv"))
    te = int(pd.Timestamp("2026-11-20 20:00:35").value // 10**6)
    kern = dict(k="2026-11-20T20:00:35", pair="KERNELUSDT", direction="SHORT", batch="B28", entry_price=0.04419, booked=-11.69,
                pnl_percentage=-11.69, close_reason="STOP_LOSS", closed_at="2026-11-21T01:07:22", entry_funding_rate=np.nan,
                funding_fee_usd=-108.0, notional_value=6777.0, te=te, kind="REAL")
    R = pd.concat([R, pd.DataFrame([kern])], ignore_index=True)
    R["tp"], R["sl"], R["hold"], R["slip"] = 5.0, 10.0, 72, 0.25
    R["fold"] = R.batch.isin(["B27", "B28"]) | (R.k >= "2026-11-14")
    R["exit_slip_row"] = 0.0
    R["sleeve"] = "FLATBASE_SHORT"; R["src"] = "FB"
    return R


def ed_cohort():
    o = pd.read_csv(EXPORT, low_memory=False)
    x = o[o.entry_strategy.isin(["EXHAUSTION_SHORT", "DISTRIBUTION_SHORT"]) & (o.status == "CLOSED")].copy()
    x = x[~x.entry_strategy.astype(str).str.endswith("_PROBE")]
    x["k"] = x.opened_at.astype(str).str[:19]
    x = x.drop_duplicates(["k", "pair", "direction"])
    x["te"] = (pd.to_datetime(x.opened_at) - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    ex = x.entry_strategy == "EXHAUSTION_SHORT"
    x["tp"] = np.where(ex, x.entry_exh_tp_pct, x.entry_dist_tp_pct)
    x["sl"] = np.where(ex, x.entry_exh_sl_pct, x.entry_dist_sl_pct)
    x["hold"], x["slip"], x["kind"], x["fold"] = 24, 0.10, "REAL", True
    x["booked"] = x.pnl / (x.quantity * x.entry_price) * 100
    x["exit_slip_row"] = pd.to_numeric(x.exit_slippage_pct, errors="coerce").fillna(0.0)
    x["batch"] = "B28"
    x["sleeve"] = x.entry_strategy; x["src"] = np.where(ex, "EXH", "DIST")
    return x[["k", "pair", "direction", "batch", "entry_price", "booked", "close_reason", "closed_at", "entry_funding_rate", "funding_fee_usd",
              "notional_value", "te", "kind", "tp", "sl", "hold", "slip", "fold", "exit_slip_row", "sleeve", "src"]]


def main():
    R = pd.concat([fb_cohort(), ed_cohort()], ignore_index=True)
    rows = []
    for r in R.itertuples():
        t, o, h, l, c = k1m(r.pair, r.src)
        fs_t, fs_r = fund_series(r.pair, r.src)
        if r.src == "FB" and r.kind != "REAL":
            te, E, i, note = FBR.entry_of(r, t, o, c)
            if te is None:
                print("no entry", r.pair, r.k, note); continue
            i0 = i
        else:
            te, E, note = int(r.te), float(r.entry_price), "real fill"
            i0 = int(np.searchsorted(t, te // M1 * M1 + M1))     # the minute after the fill
        end_t = te + int(r.hold) * HR
        if t[-1] + M1 < min(end_t, int(pd.to_datetime(r.closed_at).value // 10**6)):
            print("⚠ path ends before the booked close", r.pair, r.k)
        F = C.settle_on_bars(t, o, E, te, fs_t, fs_r, 1, markscale=True)
        Fp = C.settle_on_bars(t, o, E, te, fs_t, fs_r, 1, markscale=False)
        d = dict(sleeve=r.sleeve, k=r.k, pair=r.pair, batch=r.batch, kind=r.kind, E=E, te=te, tp=r.tp, sl=r.sl, booked=r.booked,
                 booked_reason=r.close_reason, entry_rate=r.entry_funding_rate, fund_usd=r.funding_fee_usd, notional=r.notional_value,
                 interval_h=C.interval_h(fs_t, te), rate_at_entry=C.last_rate(fs_t, fs_r, te))
        for arm in ("A", "B", "Bp"):
            w = C.walk(t, o, h, l, c, i0, E, r.tp, r.sl, end_t, F, arm, r.slip, 0.05, 0.05, 0.05, 0.02, "through")
            d.update({f"{arm}_net": w["net"], f"{arm}_why": w["why"], f"{arm}_xt": int(t[w["j"]]), f"{arm}_fund": w["fund"]})
        Fq = Fp if r.fold else np.zeros(len(t))
        pa = C.walk(t, o, h, l, c, i0, E, r.tp, r.sl, end_t, Fq, "A", r.exit_slip_row, 0.0, 0.045, 0.045, 0.018, "tick", tick=1e-12)
        d.update(par_net=pa["net"], par_why=pa["why"], par_xt=int(t[pa["j"]]))
        rows.append(d)
    X = pd.DataFrame(rows)
    X["day"] = pd.to_datetime(X.te, unit="ms").dt.strftime("%Y-%m-%d")
    for arm in ("A", "B", "Bp"):
        X[f"{arm}_day"] = pd.to_datetime(X[f"{arm}_xt"], unit="ms").dt.strftime("%Y-%m-%d")
    X.to_csv(os.path.join(OUT, "real_rows.csv"), index=False)
    pd.set_option("display.width", 260)
    X["par_gap"] = X.par_net - X.booked
    print(X[["sleeve", "k", "pair", "kind", "booked", "booked_reason", "par_net", "par_why", "par_gap", "A_net", "A_why", "A_fund", "B_net", "B_why",
             "Bp_net", "Bp_why", "interval_h", "rate_at_entry"]].round(3).to_string())
    for s, g in X.groupby("sleeve"):
        print(s, "parity mean |gap|", round(g.par_gap.abs().mean(), 3), "median", round(g.par_gap.abs().median(), 3), "max", round(g.par_gap.abs().max(), 3),
              "reason match", int(((g.par_why == "TP") == (g.booked_reason == "FRENZY_TP")).sum()), "/", len(g))


if __name__ == "__main__":
    main()
