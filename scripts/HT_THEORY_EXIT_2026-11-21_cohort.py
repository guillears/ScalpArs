#!/usr/bin/env python3
"""HT_THEORY_EXIT_2026-11-21 — REAL fills cohort (HT + plain WILLY; adapted from STATE_EXIT_2026-11-21_real_cohort.py) (research only).
Every exports_archive/scalpars_orders_*.csv (paper + live), rows of FLATBASE_SHORT / FRENZY_WILLY_HT / EXHAUSTION_SHORT, CLOSED, no probes
(is_probe / *_PROBE), operator-closed out (scout_gatekit.split_operator_closed), dedup (opened_at[:19], pair, direction) keeping the row from
the NEWEST export, opened ≥ 2026-10-01. → reports/HT_THEORY_EXIT_2026-11-21_data/real_cohort.csv"""
import glob, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_gatekit as GK  # noqa: E402
OUT = os.path.join(ROOT, "reports", "HT_THEORY_EXIT_2026-11-21_data")
SLV = ("FRENZY_WILLY_HT", "FRENZY_WILLY")
KEEP = ["opened_at", "closed_at", "pair", "direction", "entry_strategy", "entry_price", "exit_price", "quantity", "pnl", "pnl_percentage",
        "close_reason", "status", "notional_value", "funding_fee_usd", "entry_funding_rate", "exit_slippage_pct", "is_probe",
        "frenzy_willy_tp_target_pct", "entry_exh_tp_pct", "entry_exh_sl_pct", "peak_pnl", "high_price_since_entry", "low_price_since_entry",
        "leverage", "investment", "entry_frenzy_spike_at", "entry_frenzy_vwap", "entry_frenzy_willy_spike_high",
        "entry_flatbase_flat14_pct", "entry_flatbase_off_peak_pct", "entry_flatbase_peak_age_h", "entry_exh_rsi_prev", "entry_exh_rsi",
        "entry_exh_vol_decay", "entry_exh_atr_pct", "entry_exh_oi_vol", "entry_frenzy_di_spread", "entry_frenzy_adx_delta", "entry_frenzy_vol_trend",
        "entry_frenzy_willy_dip_pct", "frenzy_willy_runner_k", "entry_atr_pct"]


def main():
    fs = sorted(glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_*.csv")))
    parts = []
    for f in fs:
        try:
            h = pd.read_csv(f, nrows=0).columns
            if "entry_strategy" not in h:
                continue
            x = pd.read_csv(f, low_memory=False, usecols=[c for c in KEEP if c in h])
        except Exception as e:
            print("skip", os.path.basename(f), e); continue
        x = x[x.entry_strategy.isin(SLV) & (x.status == "CLOSED")]
        if not len(x):
            continue
        x["src"] = os.path.basename(f)
        x["mode"] = "live" if "_live_" in f else "paper"
        parts.append(x)
    X = pd.concat(parts, ignore_index=True)
    if "is_probe" in X:
        X = X[~X.is_probe.fillna(False).astype(str).str.lower().isin(["true", "1"])]
    X = X[~X.entry_strategy.astype(str).str.endswith("_PROBE")]
    X["k"] = X.opened_at.astype(str).str[:19]
    # (no date floor: every archived fill back to each sleeve's first one)
    X = X.sort_values("src").drop_duplicates(["k", "pair", "direction"], keep="last")
    X, oc = GK.split_operator_closed(X)
    print(GK.operator_closed_line(oc))
    X["net_pct"] = X.pnl / (X.quantity * X.entry_price) * 100
    X = X.sort_values("k")
    X.to_csv(os.path.join(OUT, "real_cohort.csv"), index=False)
    for s, g in X.groupby("entry_strategy"):
        print(s, len(g), "losers", int((g.net_pct < 0).sum()), "modes", g["mode"].value_counts().to_dict(), "reasons", g.close_reason.value_counts().to_dict())


if __name__ == "__main__":
    main()
