#!/usr/bin/env python3
"""🪫 Scout — EXHAUSTION_SHORT revert gate (DECISION_LOG 578, 2026-11-18, operator-approved PAPER run, declared override — ALERT-only,
never changes config).

RULE (engine)   services/exhaustion.py: a coin in a FRENZY episode (≤ 72 h after the spike bar, 24 h volume ≥ $20M, Binance SPOT market) is
                shorted at market after a CLOSED 15m bar whose Wilder RSI(14) falls from > 70 to ≤ 70, when vol_decay ≥ 0.6366 and the spot
                share of volume is falling (change < −0.0066). Exit TP min(3 × ATR, 5 %) (resting maker) · SL min(1.5 × ATR, 5 %) · 24 h.
COHORT          CLOSED EXHAUSTION_SHORT fills from the orders exports (paper + live via export_glob; dedup (opened_at, pair, direction), CLOSED
                wins); operator-closed (MANUAL / MANUAL_* / EXTERNAL_CLOSE) excluded.
GATE (frozen)   services.exhaustion.exhaustion_gate (the SAME function as the dashboard FRENZY Fills row): the first 20 fills by OPEN time, each
                re-priced with the study's modelled slippage (stamped per fill: entry_exh_model_slip_pct = max(0, −0.042 + 0.276 · √(notional
                ÷ Q5)), charged on the entry and on a taker exit, not on the maker TP) → average ≤ 0 → OFF · mean modelled slippage > 0.25 %
                → OFF · else KEEP (paper). A fill without the stamp is re-priced from its notional and entry_exh_q5_usd.
JOURNAL         e=EXHAUSTION rows of the decisions exports: every evaluated signal with its status / block — the tally per block is printed
                (what the rule refused, what the open path refused).
NETWORK         none. Usage: venv/bin/python scripts/scout_exhaustion_short.py"""
import os
import sys

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import export_glob as _EG                                    # noqa: E402  🗄 (DECISION_LOG 409) Downloads + exports_archive/
from services.exhaustion import EXHAUSTION_SHORT, exhaustion_gate, model_slip_pct, gate_net_pct, GATE_N, exit_era_579   # noqa: E402

DL = 578
ORDER_PATTERN = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
DEC_PATTERNS = (os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"), os.path.join(_R, "reports", "BASELINE*_decisions.csv"))
ORDER_COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason",
              "notional_value", "entry_exh_model_slip_pct", "entry_exh_q5_usd", "entry_exh_tp_pct", "entry_exh_sl_pct", "entry_slippage_pct", "entry_exh_atr_pct")
DEC_COLS = ("e", "pair", "exh_signal_at", "exh_status", "exh_block", "exh_reason",
            # 🧯 (DECISION_LOG 582) what scripts/scout_exh_ssc_oivol.py prices an EXH_SSC_OIVOL block with
            "exh_decision_price", "exh_atr_pct", "exh_tp_pct", "exh_sl_pct", "exh_q5_usd", "exh_vol24_usd", "exh_spot_share_chg",
            "exh_oi_usd", "exh_oi_vol", "exh_model_slip_pct", "exh_shadow_notional_usd")


def _operator_closed(r) -> bool:
    s = str(r or '').upper()
    return s.split(' ')[0] in ('MANUAL', 'EXTERNAL_CLOSE') or s.startswith('MANUAL_')


def load_fills(paths=None, extra=()) -> pd.DataFrame:
    """CLOSED EXHAUSTION_SHORT fills, dedup (opened_at, pair, direction) — a CLOSED row wins, then the newest export.
    `extra` = more order columns to keep (the DL 579 exit-override scout reads entry_price / entry_exh_atr_pct)."""
    paths = paths if paths is not None else sorted(_EG.glob_exports(ORDER_PATTERN), key=os.path.getmtime)
    cols = tuple(ORDER_COLS) + tuple(c for c in extra if c not in ORDER_COLS)
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(cols))
    a = pd.concat(fr, ignore_index=True)
    for c in cols:
        if c not in a:
            a[c] = None
    a = a[a.entry_strategy.astype(str) == EXHAUSTION_SHORT]
    a = a.assign(_c=(a.status.astype(str) == "CLOSED").astype(int), _i=range(len(a))).sort_values(["_c", "_i"])
    a = a.drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a = a[(a.status.astype(str) == "CLOSED") & a.pnl_percentage.notna() & ~a.close_reason.map(_operator_closed).astype(bool)]
    a["o"] = pd.to_datetime(a.opened_at.astype(str).str[:19], errors="coerce")
    a = a[a.o.notna()].sort_values("o").reset_index(drop=True)

    def _slip(r):
        s = pd.to_numeric(r.entry_exh_model_slip_pct, errors="coerce")
        if pd.notna(s):
            return float(s)
        return model_slip_pct(pd.to_numeric(r.notional_value, errors="coerce"), pd.to_numeric(r.entry_exh_q5_usd, errors="coerce"))
    a["slip"] = [(_slip(r)) for r in a.itertuples()] if len(a) else []
    return a.drop(columns=["_c", "_i"])


def load_journal(paths=None) -> pd.DataFrame:
    paths = paths if paths is not None else sorted({p for pat in DEC_PATTERNS for p in _EG.glob_exports(pat)}, key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in DEC_COLS)
        except Exception:
            continue
        if "e" in x and "exh_status" in x:
            fr.append(x[x.e.astype(str) == "EXHAUSTION"])
    if not fr:
        return pd.DataFrame(columns=list(DEC_COLS))
    return pd.concat(fr, ignore_index=True).drop_duplicates(["pair", "exh_signal_at"], keep="last")


def report(fills: pd.DataFrame, journal: pd.DataFrame) -> str:
    out = [f"🪫 EXHAUSTION_SHORT revert gate (DECISION_LOG {DL}, paper, declared override) — 🔁 re-based 2026-11-18 on the DL 579 exit "
           "(TP ≈ 10 % / SL ≤ 6 %): the count starts at the first fill of that exit"]
    if len(fills):
        era = fills.apply(lambda r: exit_era_579(r.entry_exh_tp_pct, r.entry_exh_atr_pct), axis=1).astype(bool)
        for r in fills[~era].itertuples():
            tag = "shipped-exit era" if pd.notna(pd.to_numeric(r.entry_exh_tp_pct, errors="coerce")) else "unstamped"
            out.append(f"  ({tag}, not counted) {str(r.o)[:16]} {r.pair} {float(r.pnl_percentage):+.2f}% {r.close_reason}")
        fills = fills[era]
    rows = [(r.o, float(r.pnl_percentage), r.close_reason, r.slip) for r in fills.itertuples()]
    v, txt = exhaustion_gate(rows)
    out.append(f"  verdict: {v} — {txt}")
    if rows:
        nets = [gate_net_pct(p, r, s) for _, p, r, s in rows]
        out.append(f"  fills {len(rows)} · raw avg {sum(p for _, p, _, _ in rows) / len(rows):+.3f}% · re-priced avg {sum(nets) / len(nets):+.3f}% · "
                   f"WR {100.0 * sum(1 for x in nets if x > 0) / len(nets):.0f}% (re-priced) · first {GATE_N} frozen")
        for r, n in zip(fills.itertuples(), nets):
            out.append(f"    {str(r.o)[:16]} {r.pair:<14} {float(r.pnl_percentage):+6.2f}% → {n:+6.2f}% (slip {r.slip if r.slip is not None else float('nan'):.3f}) "
                       f"{r.close_reason} · TP {r.entry_exh_tp_pct} / SL {r.entry_exh_sl_pct}")
    if len(journal):
        tally = journal.assign(k=journal.exh_block.fillna(journal.exh_status)).k.value_counts()
        out.append("  journal (every evaluated signal): " + " · ".join(f"{k} {v}" for k, v in tally.items()))
    else:
        out.append("  journal: no e=EXHAUSTION rows in the decisions exports yet")
    return "\n".join(out)


def main():
    print(report(load_fills(), load_journal()))


if __name__ == "__main__":
    main()
