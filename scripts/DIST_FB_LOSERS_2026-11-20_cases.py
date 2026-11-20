#!/usr/bin/env python3
"""DIST_FB_LOSERS_2026-11-20 — per-fill forensic case files for every DISTRIBUTION_SHORT / FLATBASE_SHORT fill of the current
paper batch (export exports_archive/scalpars_orders_paper_2026-11-20_16-01-39.csv, CLOSED, probes out).

For each fill, from its own 1m path (reports/DIST_FB_LOSERS_2026-11-20_data/k1m, built by DIST_FB_LOSERS_2026-11-20_fetch.py):
signal close → fill (lag, drift), the stop / TP lines, MFE / MAE before the stop, minutes to the stop, booked vs stop-line slip,
funding, where the price went after the stop (did it reach the TP inside the sleeve's hold?), wider / tighter stop what-ifs on
this one path (descriptive only — the decision read is the backtest cohort in DIST_FB_LOSERS_2026-11-20_stopwidth.py), the
coin's prior move, BTC over the same minutes and the other bot fills in the same window.
Output: reports/DIST_FB_LOSERS_2026-11-20_data/cases.csv + cases.json + paths.pkl (anchored paths for the chart).
Usage: venv/bin/python -I scripts/DIST_FB_LOSERS_2026-11-20_cases.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "DIST_FB_LOSERS_2026-11-20_data")
EXPORT = os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-11-20_16-01-39.csv")
MIN = 60_000
HOLD_H = {"DISTRIBUTION_SHORT": 24, "FLATBASE_SHORT": 72}
SIG_MIN = {"DISTRIBUTION_SHORT": 15, "FLATBASE_SHORT": 240}


def k1(sym):
    k = pd.read_csv(os.path.join(D, "k1m", f"{sym}.csv"))
    return k.astype({"open_time": "int64"}).set_index("open_time").sort_index()


def k1h(sym):
    k = pd.read_csv(os.path.join(D, "k1h", f"{sym}.csv"))
    return k.astype({"open_time": "int64"}).set_index("open_time").sort_index()


def ms(ts):
    return int(pd.Timestamp(ts).tz_localize("UTC").value // 10**6) if pd.Timestamp(ts).tzinfo is None else int(pd.Timestamp(ts).value // 10**6)


def first_hit(series_bool):
    a = np.asarray(series_bool)
    return int(a.argmax()) if a.any() else None


def main():
    o = pd.read_csv(EXPORT, low_memory=False)
    o = o[(o.status == "CLOSED") & o.entry_strategy.isin(list(HOLD_H))].copy()
    if "is_probe" in o:
        o = o[o.is_probe.astype(str).str.lower() != "true"]
    allf = pd.read_csv(EXPORT, low_memory=False)
    allf["t_open"] = allf.opened_at.map(ms)
    btc = k1("BTCUSDT")
    rows, paths = [], {}
    for _, r in o.sort_values("opened_at").iterrows():
        sl_ = r.entry_strategy
        sym = r.pair
        k = k1(sym)
        t_fill = ms(r.opened_at)
        t_close = ms(r.closed_at)
        sig_ms = SIG_MIN[sl_] * MIN
        t_sig = (t_fill // sig_ms) * sig_ms                       # the signal candle closes at this boundary
        sig_close = float(k.loc[t_sig - MIN, "c"]) if (t_sig - MIN) in k.index else np.nan
        E = float(r.entry_price)
        if sl_ == "DISTRIBUTION_SHORT":
            sl_pct, tp_pct = float(r.entry_dist_sl_pct), float(r.entry_dist_tp_pct)
        else:
            sl_pct, tp_pct = 10.0, 5.0
        stop_line, tp_line = E * (1 + sl_pct / 100), E * (1 - tp_pct / 100)
        t0 = (t_fill // MIN) * MIN
        t_end = t_fill + HOLD_H[sl_] * 3_600_000
        w = k[(k.index >= t0) & (k.index < t_end)]
        fav = (1 - w.l.values / E) * 100
        adv = (w.h.values / E - 1) * 100
        i_stop = first_hit(w.h.values >= stop_line)
        i_tp = first_hit(w.l.values <= tp_line)
        booked_stop = r.close_reason == "STOP_LOSS"
        # minutes / extremes up to the booked close
        n_close = int(np.searchsorted(w.index.values, t_close, "right"))
        mfe = float(fav[:max(n_close, 1)].max())
        mae = float(adv[:max(n_close, 1)].max())
        t_mfe = int((w.index.values[int(fav[:max(n_close, 1)].argmax())] - t0) // MIN)
        after = w[w.index > t_close]
        tp_after = bool((after.l <= tp_line).any()) if len(after) else False
        tp_after_min = int((after.index.values[int((after.l <= tp_line).values.argmax())] - t_fill) // MIN) if tp_after else None
        post_hi = float((after.h.max() / E - 1) * 100) if len(after) else np.nan
        post_lo = float((1 - after.l.min() / E) * 100) if len(after) else np.nan
        end_px = float(w.c.values[-1]) if len(w) else np.nan
        hold_end_pct = float((1 - end_px / E) * 100)
        hold_complete = bool(len(w) and w.index.values[-1] >= t_end - 2 * MIN)
        # one-path what-ifs: first passage of TP vs a stop at s % (descriptive, this path only)
        whatif = {}
        for s in (4, 6, 8, 10, 12, 15, 20):
            j_s = first_hit(w.h.values >= E * (1 + s / 100))
            j_t = i_tp
            if j_t is not None and (j_s is None or j_t < j_s):
                whatif[s] = f"TP {tp_pct:g} @{int((w.index.values[j_t] - t_fill) // MIN)}m"
            elif j_s is not None:
                whatif[s] = f"stop @{int((w.index.values[j_s] - t_fill) // MIN)}m"
            else:
                whatif[s] = f"open/time {hold_end_pct:+.1f}"
        # prior move (1h klines)
        h1 = k1h(sym)
        pre = h1[h1.index < t_sig]
        ret24 = float((sig_close / pre.c.iloc[-25] - 1) * 100) if len(pre) > 25 else np.nan
        ret7d = float((sig_close / pre.c.iloc[-169] - 1) * 100) if len(pre) > 169 else np.nan
        spike_ms = ms(r.entry_frenzy_spike_at) if pd.notna(r.get("entry_frenzy_spike_at")) else None
        if spike_ms is not None:
            kk = k[(k.index >= spike_ms - 35 * MIN) & (k.index < t_sig)]
            base_px = float(k.loc[spike_ms - 30 * MIN, "c"]) if (spike_ms - 30 * MIN) in k.index else np.nan
            hi_since = float(kk.h.max()) if len(kk) else np.nan
            off_hi = float((sig_close / hi_since - 1) * 100) if len(kk) else np.nan
            above_base = float((sig_close / base_px - 1) * 100) if base_px == base_px else np.nan
            run = float((hi_since / base_px - 1) * 100) if base_px == base_px else np.nan
        else:
            off_hi = above_base = run = np.nan
        # BTC over the same minutes
        def bret(a, b):
            try:
                pa = float(btc.loc[(a // MIN) * MIN, "o"]); pb = float(btc.loc[(b // MIN) * MIN, "c"])
                return (pb / pa - 1) * 100
            except KeyError:
                return np.nan
        btc_pre4h = bret(t_sig - 4 * 3_600_000, t_sig - MIN)
        btc_hold = bret(t_fill, t_close)
        # same-window bot fills (± 60 min of the fill, any sleeve)
        same = allf[(abs(allf.t_open - t_fill) <= 3_600_000) & (allf.id != r.id)]
        same_txt = "; ".join(f"{x.pair[:-4]} {x.entry_strategy.replace('_SHORT','').replace('FRENZY_','')} {x.pnl_percentage:+.1f}" for _, x in same.iterrows())
        notional = float(r.notional_value)
        fund_pct = float(r.funding_fee_usd) / notional * 100 if notional else np.nan
        row = dict(
            sleeve=sl_, pair=sym, id=int(r.id), opened=r.opened_at, closed=str(r.closed_at)[:19], reason=r.close_reason,
            booked_pct=float(r.pnl_percentage), booked_usd=float(r.pnl), notional=notional,
            signal_close_utc=pd.Timestamp(t_sig, unit="ms").strftime("%m-%d %H:%M"), sig_close=sig_close,
            decision_px=float(r.entry_dist_decision_price) if pd.notna(r.get("entry_dist_decision_price")) else np.nan,
            fill=E, lag_s=float(r.entry_order_lag_s),
            drift_cost_pct=float((sig_close - E) / sig_close * 100) if sig_close == sig_close else np.nan,   # + = the short filled lower = cost
            book_drift_pct=float(r.entry_book_drift_pct), fill_vs_book_pct=float(r.entry_fill_vs_book_pct),
            sl_pct=sl_pct, tp_pct=tp_pct, atr_pct=float(r.entry_dist_atr_pct) if pd.notna(r.get("entry_dist_atr_pct")) else np.nan,
            stop_line=stop_line, tp_line=tp_line, exit_px=float(r.exit_price),
            exit_slip_vs_line_pct=float((r.exit_price / stop_line - 1) * 100) if booked_stop else np.nan,
            min_to_close=round((t_close - t_fill) / MIN, 1), mfe_pct=mfe, t_mfe_min=t_mfe, mae_pct=mae,
            tp_after_stop=tp_after, tp_after_min=tp_after_min, post_max_adverse_pct=post_hi, post_max_fav_pct=post_lo,
            hold_end_pct=hold_end_pct, hold_complete=hold_complete,
            funding_usd=float(r.funding_fee_usd), funding_pct=fund_pct,
            cap_frac=float(r.entry_cap_frac), cap_binding=r.entry_cap_binding_side, book_usd_02=float(r.entry_book_usd_02),
            book_exit_usd_02=float(r.entry_book_exit_usd_02), vol24_musd=float(r.entry_pair_volume_24h_usd) / 1e6,
            hours_after_spike=float(r.entry_frenzy_hours), ret24h_pre=ret24, ret7d_pre=ret7d, off_spike_high_pct=off_hi,
            above_base_pct=above_base, run_pct=run, btc_pre4h_pct=btc_pre4h, btc_during_pct=btc_hold,
            same_window=same_txt,
            **{f"whatif_stop{s}": v for s, v in whatif.items()},
        )
        rows.append(row)
        # anchored path from the signal close: −60 min … +hold, close-to-close, short-positive (% vs the SIGNAL close)
        ts = np.arange(-60, HOLD_H[sl_] * 60 + 1) * MIN + t_sig
        kk = k.reindex(ts - MIN)        # the bar whose close is at ts
        paths[f"{sl_}|{sym}|{r.id}"] = dict(t_min=(ts - t_sig) / MIN, p=(1 - kk.c.values / sig_close) * 100,
                                           fill_min=(t_fill - t_sig) / MIN, close_min=(t_close - t_sig) / MIN,
                                           stop=-((stop_line / sig_close) - 1) * 100, tp=(1 - tp_line / sig_close) * 100,
                                           won=float(r.pnl_percentage) > 0)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(D, "cases.csv"), index=False)
    pd.to_pickle(paths, os.path.join(D, "paths.pkl"))
    pd.set_option("display.width", 250)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
