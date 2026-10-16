#!/usr/bin/env python3
"""Master-pool support (DECISION_LOG 282): every archived FRENZY_WILLY fill re-walked under TODAY's WILLY exit (trading_config.json
frenzy_willy_tp_pct / frenzy_willy_stop_pct / frenzy_willy_max_hold_minutes; +1.25 / −4 / 90 at DL 281) on its own price path — ticks first,
then the 1m tail (scripts/study_b19_fzexit_common.load_path / evaluate, tick ruler = first print through the line; 28/28 live parity on
B19). WILLY's exit is path-dependent, so build_master_pool reads this table (reports/WILLY_EXIT_CF.csv) instead of the as-traded pct;
an unpriced fill keeps its as-traded pct (the builder lists those) and the builder refuses a CF priced under another exit. Cache-only. Re-run whenever the WILLY exit changes or a batch is archived.
STACK 2026-10-16a (B22 archive): today's WILLY exit also carries ⏰ the late TP (DECISION_LOG 297: from minute frenzy_willy_late_tp_minutes the
TP line is frenzy_willy_late_tp_pct) and 🎯 the resting maker TP (DECISION_LOG 296, resting_tp_willy_enabled): walked with
scout_willy_timecap.evaluate_late(book="live") — the walker the Scout uses for post-297 fills (DECISION_LOG 304): a TP on the +tp line books
EXACTLY the line (the resting limit fills at its price — before 296 the first print through it was booked), a late-line TP books the line,
or the print when the first event after the late minute is already through it (the engine market-closes then); stop = first print through
−stop (unchanged), cap = last event ≤ the cap (unchanged). Fill assumption: a touch of the line fills the resting limit (paper books ≥ 1 tick
through; queue / partial fills not modelled — optimistic by ≤ 1 tick). exit_rule = "+tp/-stop/cap[/late<min>@+<pct>][/rtp]".
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
    late_min = float(th.get("frenzy_willy_late_tp_minutes") or 0); late_pct = float(th.get("frenzy_willy_late_tp_pct") or 0)
    late_on = late_min > 0 and 0 < late_pct < tp                     # services.frenzy.frenzy_willy_late_tp: ≤ 0 or ≥ the TP = off
    rtp = bool(th.get("resting_tp_willy_enabled"))                   # DECISION_LOG 296
    rule = (f"+{tp:g}/-{st:g}/{cap}" if st else f"+{tp:g}/none/{cap}") + (f"/late{late_min:g}@+{late_pct:g}" if late_on else "") + ("/rtp" if rtp else "")

    def walk(P):
        if not late_on and not rtp:
            return FX.evaluate(P, tp, st or 1e9, cap)                # the pre-296/297 walker (print booking)
        lm, lp = (late_min, late_pct) if late_on else (float(cap) + 1.0, tp)
        return FX.WT.evaluate_late(P, tp, st or 1e9, cap, lm, lp, book="live" if rtp else "print")
    rows, cache = [], {}
    for f in sorted(glob.glob(os.path.join(ROOT, "reports", "BASELINE[0-9]*_*orders*.csv"))):
        d = pd.read_csv(f, low_memory=False)
        d = d[(d.get("entry_strategy").astype(str) == "FRENZY_WILLY") & (d.status == "CLOSED")] if "entry_strategy" in d else d.iloc[:0]
        for r in d.itertuples():
            te = int(pd.Timestamp(r.opened_at).value // 1_000_000)
            P = FX.load_path(r.pair, te, float(r.entry_price), cap, cache)
            ev = walk(P) if len(P["t"]) else dict(why="UNRES", pct=float("nan"), xmin=float("nan"))
            src = "none" if not len(P["t"]) else ("ticks" if P["n_bar"] == 0 else ("1m" if P["n_tick"] == 0 else "ticks+1m"))
            rows.append(dict(opened_at=str(r.opened_at)[:19], pair=r.pair, entry_strategy="FRENZY_WILLY", live_pct=float(r.pnl_percentage),
                             cf_pct=None if ev["why"] == "UNRES" else round(float(ev["pct"]), 4), cf_why=ev["why"], cf_min=ev["xmin"], src=src,
                             exit_rule=rule))
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    print(f"{len(R)} WILLY fills → {OUT}")
    if len(R):
        print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
