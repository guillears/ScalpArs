#!/usr/bin/env python3
"""Master-pool support (DECISION_LOG 282): every archived FRENZY_WILLY fill re-walked under TODAY's WILLY exit (trading_config.json
frenzy_willy_tp_pct / frenzy_willy_stop_pct / frenzy_willy_max_hold_minutes; +1.25 / −4 / 90 at DL 281) on its own price path — ticks first,
then the 1m tail (scripts/study_b19_fzexit_common.load_path / evaluate, tick ruler = first print through the line; 28/28 live parity on
B19). WILLY's exit is path-dependent, so build_master_pool reads this table (reports/WILLY_EXIT_CF.csv) instead of the as-traded pct;
an unpriced fill keeps its as-traded pct (the builder lists those) and the builder refuses a CF priced under another exit. Cache-only. Re-run whenever the WILLY exit changes or a batch is archived.
Usage: venv/bin/python -I scripts/build_willy_exit_cf.py
"""
import glob, json, os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_b19_fzexit_common as FX
OUT = os.path.join(ROOT, "reports", "WILLY_EXIT_CF.csv")


def main():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    tp = float(th.get("frenzy_willy_tp_pct") or 1.0); st = abs(float(th.get("frenzy_willy_stop_pct") or 0)) or None
    cap = int(th.get("frenzy_willy_max_hold_minutes") or 120)
    rows, cache = [], {}
    for f in sorted(glob.glob(os.path.join(ROOT, "reports", "BASELINE[0-9]*_*orders*.csv"))):
        d = pd.read_csv(f, low_memory=False)
        d = d[(d.get("entry_strategy").astype(str) == "FRENZY_WILLY") & (d.status == "CLOSED")] if "entry_strategy" in d else d.iloc[:0]
        for r in d.itertuples():
            te = int(pd.Timestamp(r.opened_at).value // 1_000_000)
            P = FX.load_path(r.pair, te, float(r.entry_price), cap, cache)
            ev = FX.evaluate(P, tp, st or 1e9, cap) if len(P["t"]) else dict(why="UNRES", pct=float("nan"), xmin=float("nan"))
            src = "none" if not len(P["t"]) else ("ticks" if P["n_bar"] == 0 else ("1m" if P["n_tick"] == 0 else "ticks+1m"))
            rows.append(dict(opened_at=str(r.opened_at)[:19], pair=r.pair, entry_strategy="FRENZY_WILLY", live_pct=float(r.pnl_percentage),
                             cf_pct=None if ev["why"] == "UNRES" else round(float(ev["pct"]), 4), cf_why=ev["why"], cf_min=ev["xmin"], src=src,
                             exit_rule=f"+{tp:g}/-{st:g}/{cap}" if st else f"+{tp:g}/none/{cap}"))
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    print(f"{len(R)} WILLY fills → {OUT}")
    if len(R):
        print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
