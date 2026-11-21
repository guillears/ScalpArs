#!/usr/bin/env python3
"""STATE_EXIT_2026-11-21 — unified fill table (research only).
Per sleeve FB = FLATBASE_SHORT · HT = FRENZY_WILLY_HT · EXH = EXHAUSTION_SHORT:
  REAL  : reports/STATE_EXIT_2026-11-21_data/real_cohort.csv (every archived export, CLOSED, no probes, operator-closed out, dedup
          (opened_at[:19], pair, direction)); period MASTER (opened < CUR_BATCH) / BATCH (current batch, opened ≥ CUR_BATCH). Entry = the REAL
          fill price; the walk starts the minute AFTER the fill (the fill minute's range holds pre-fill prints).
  BT    : FB = reports/FUNDING_STOP_2026-11-21_data/bt_FB.csv minus the DL 601 funding-above-base entries (entry_rate > 0.00005) = today's
          stack (545); HT = reports/HT_WILLY_2026-11-20_cohort_HT.csv (300, dip ≥ 1.4 = DL 594 stack); EXH = bt_EXH.csv stratified sample
          (half-year × recorded exit kind, 30 %, seed 7). Period DESIGN (< 2025-07-01) / HOLDOUT (≥ 2025-07-01).
  CASE  : operator-named live give-backs (illustration only, never counted): ENSO / EVAA HT, RARE EXH.
Today's exit: FB TP −5 % price (maker, resting) · stop +10 % price counting the funding PAID (DL 607) · 72 h. HT TP +5 net (maker) · stop −4
net · 24 h (DL 587; late TP off, high target off). EXH TP / SL = the row's own (min(20 × ATR15, 10) / min(1.5 × ATR15, 6)) price · 24 h · price
stop. Costs: entry taker 0.05 (+ the row's entry slip for EXH backtest rows), maker TP 0.02, stop / time taker 0.05 + slip (FB real 0.25,
FB backtest the row's slip_x, HT 0.10, EXH the row's slip ≥ 0.10).
→ reports/STATE_EXIT_2026-11-21_data/fills.csv"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "STATE_EXIT_2026-11-21_data")
CUR_BATCH = "2026-11-17T17:00"           # the current paper batch starts after the 11-17 live stop (B28)
SPLIT = pd.Timestamp("2025-07-01").value // 10**6
TAKER, MAKER = 0.05, 0.02


def ms(s):
    return int(pd.Timestamp(str(s).replace("T", " ")[:19]).value // 10**6)


def ht_px(E, tp_net, sl_net):
    return E * (100 + tp_net + TAKER) / (100 - MAKER), E * (100 - sl_net + TAKER) / (100 - TAKER)


def main():
    rows = []
    R = pd.read_csv(os.path.join(D, "real_cohort.csv"), low_memory=False)
    for r in R.itertuples():
        te = ms(r.opened_at); E = float(r.entry_price)
        per = "BATCH" if r.k >= CUR_BATCH else "MASTER"
        d = dict(sleeve={"FLATBASE_SHORT": "FB", "FRENZY_WILLY_HT": "HT", "EXHAUSTION_SHORT": "EXH"}[r.entry_strategy], kind="REAL",
                 period=per, pair=r.pair, te=te, E=E, start="next", booked=r.net_pct, booked_reason=r.close_reason, k=r.k,
                 spike_ms=(ms(r.entry_frenzy_spike_at) if isinstance(r.entry_frenzy_spike_at, str) else np.nan),
                 H=getattr(r, "entry_frenzy_willy_spike_high", np.nan), entry_cost=TAKER, rec_net=r.net_pct, rec_why=r.close_reason)
        if d["sleeve"] == "FB":
            d.update(dirn=-1, tp_px=E * 0.95, sl_px=E * 1.10, sl_pct=10.0, stop_mode="paid", hold_h=72, slip=0.25,
                     par_ok=r.k >= "2026-10-28")            # booked under −5 / +10 (DL 408) from ~10-28; earlier rows ran +8 / −15
        elif d["sleeve"] == "HT":
            tp, sl = ht_px(E, 5.0, 4.0)
            d.update(dirn=1, tp_px=tp, sl_px=sl, sl_pct=4.0, stop_mode="price", hold_h=24, slip=0.10,
                     par_ok=(abs(float(r.frenzy_willy_tp_target_pct or 0) - 5.0) < 1e-6))
        else:
            d.update(dirn=-1, tp_px=E * (1 - r.entry_exh_tp_pct / 100), sl_px=E * (1 + r.entry_exh_sl_pct / 100), sl_pct=r.entry_exh_sl_pct,
                     stop_mode="price", hold_h=24, slip=0.10, par_ok=True)
        rows.append(d)
    # ── backtest
    F = pd.read_csv(os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data", "bt_FB.csv"), low_memory=False)
    F = F[~(F.entry_rate > 0.00005)]
    for r in F.itertuples():
        te = int(r.cf_entry_t); E = float(r.cf_entry_px)
        rows.append(dict(sleeve="FB", kind="BT", period="DESIGN" if te < SPLIT else "HOLDOUT", pair=r.pair, te=te, E=E, start="bar",
                         spike_ms=int(r.spike_ts) - 300_000, H=np.nan, entry_cost=TAKER, rec_net=r.Bp_net, rec_why=r.Bp_why, dirn=-1,
                         tp_px=E * 0.95, sl_px=E * 1.10, sl_pct=10.0, stop_mode="paid", hold_h=72, slip=float(r.slip_x), par_ok=True,
                         dist_base=r.dist_base, k=str(pd.Timestamp(te, unit="ms"))[:19]))
    Hh = pd.read_csv(os.path.join(ROOT, "reports", "HT_WILLY_2026-11-20_cohort_HT.csv"))
    for r in Hh.itertuples():
        te = int(r.ts); E = float(r.E); tp, sl = ht_px(E, 5.0, 4.0)
        rows.append(dict(sleeve="HT", kind="BT", period="DESIGN" if te < SPLIT else "HOLDOUT", pair=r.pair, te=te, E=E, start="bar",
                         spike_ms=int(r.spike), H=float(r.H), entry_cost=TAKER, rec_net=np.nan, rec_why=r.ref_kind, dirn=1, tp_px=tp, sl_px=sl,
                         sl_pct=4.0, stop_mode="price", hold_h=24, slip=0.10, par_ok=True, k=str(pd.Timestamp(te, unit="ms"))[:19],
                         ht_half=r.half))
    X = pd.read_csv(os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data", "bt_EXH.csv"))
    X["half"] = pd.to_datetime(X.te, unit="ms").dt.year.astype(str) + np.where(pd.to_datetime(X.te, unit="ms").dt.month <= 6, "-H1", "-H2")
    ix = [i for _, g in X.groupby(["half", "A_why"]) for i in g.sample(frac=0.30, random_state=7).index]
    X = X.loc[sorted(ix)]
    for r in X.itertuples():
        te = int(r.te); E = float(r.E); sl = float(r.slip)
        rows.append(dict(sleeve="EXH", kind="BT", period="DESIGN" if te < SPLIT else "HOLDOUT", pair=r.pair, te=te, E=E, start="bar",
                         spike_ms=np.nan, H=np.nan, entry_cost=TAKER + sl, rec_net=r.A_net, rec_why=r.A_why, dirn=-1, tp_px=E * (1 - r.tp / 100),
                         sl_px=E * (1 + r.sl / 100), sl_pct=r.sl, stop_mode="price", hold_h=24, slip=max(sl, 0.10), par_ok=True,
                         k=str(pd.Timestamp(te, unit="ms"))[:19]))
    # ── named live cases (illustration only)
    for pair, k, E, sl_, tp_net in (("ENSOUSDT", "2026-11-21T02:30:14", 0.5034, "HT", 5.0), ("EVAAUSDT", "2026-11-20T23:20:12", 6.20, "HT", 5.0)):
        tp, sl = ht_px(E, 5.0, 4.0)
        rows.append(dict(sleeve="HT", kind="CASE", period="CASE", pair=pair, te=ms(k), E=E, start="next", spike_ms=np.nan, H=np.nan,
                         entry_cost=TAKER, rec_net=np.nan, rec_why="", dirn=1, tp_px=tp, sl_px=sl, sl_pct=4.0, stop_mode="price", hold_h=24,
                         slip=0.10, par_ok=False, k=k))
    E = 0.01386
    rows.append(dict(sleeve="EXH", kind="CASE", period="CASE", pair="RAREUSDT", te=ms("2026-11-20T20:00:39"), E=E, start="next", spike_ms=np.nan,
                     H=np.nan, entry_cost=TAKER, rec_net=np.nan, rec_why="", dirn=-1, tp_px=E * 0.90, sl_px=E * 1.06, sl_pct=6.0,
                     stop_mode="price", hold_h=24, slip=0.10, par_ok=False, k="2026-11-20T20:00:39"))
    Z = pd.DataFrame(rows)
    Z["fid"] = np.arange(len(Z))
    Z.to_csv(os.path.join(D, "fills.csv"), index=False)
    print(Z.groupby(["sleeve", "kind", "period"]).size())


if __name__ == "__main__":
    main()
