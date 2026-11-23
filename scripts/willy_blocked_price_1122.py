#!/usr/bin/env python3
"""💲 One-off (DECISION_LOG 628, 2026-11-22): price the WILLY / WILLY_HT entries refused on 2026-11-22 on today's exits, with the SAME pricer
as the revert gates (scripts/scout_blocked_filters.price_all — 8 s latency, 0.10 % slip against the trade, taker fee, 1m adverse-first walk).

① the 5 filter refusals (decisions e=BLOCKED rows: EVAA / BULLA / BLESS HT · XVG / PROM WILLY) — real decision price + entry snapshot.
② ELSA (09:00, gate FRENZY_WILLY_TURNOVER_UNREAD — turnover unreadable, never re-read, expired 09:55): NOT a filter block, so no gate
   prices it; shown as a SENSITIVITY only — priced both as plain WILLY and as HT (its turnover R, hence its sleeve, is unknown, and whether
   WILLY_FAILED_SPIKE would have blocked it is unknown).
Reads the export from exports_archive/ (permanent copy, DL 409), else ~/Downloads. Prices against a COPY of the pricer's store
(argv[1] = a scratch folder holding sbf_store.csv) — never the real store / gate state.
Usage: venv/bin/python -I scripts/willy_blocked_price_1122.py <scratch_dir>
"""
import json
import os
import sys
import time

_D = os.path.dirname(os.path.abspath(__file__))
for _p in (_D, os.path.dirname(_D)):
    _p in sys.path or sys.path.insert(0, _p)
import pandas as pd                       # noqa: E402
import scout_blocked_filters as S         # noqa: E402

_NAME = "scalpars_decisions_paper_2026-11-23_00-04-24.csv"
DEC = next((p for p in (os.path.join(os.path.dirname(_D), "exports_archive", _NAME), os.path.expanduser(f"~/Downloads/{_NAME}"))
            if os.path.exists(p)), _NAME)
DAY0, DAY1 = "2026-11-22", "2026-11-23"


def _ms(t):
    return int(pd.Timestamp(t).value // 1_000_000)


def elsa_rows():
    """ELSA's refusal row (gate FRENZY_WILLY_TURNOVER_UNREAD) → one shadow per possible sleeve."""
    d = pd.read_csv(DEC, low_memory=False, usecols=lambda c: c in ("t", "e", "pair", "price", "gate"))
    r = d[(d.pair == "ELSAUSDT") & (d.e.astype(str) == "WILLY") & (d.gate.astype(str) == "FRENZY_WILLY_TURNOVER_UNREAD")
          & d.t.astype(str).str.startswith(DAY0)].iloc[0]          # the 11-22 09:00 refusal (11-21 05:20 had one too)
    t, px = S._ms(r.t), float(r.price)
    return [dict(pair="ELSAUSDT", sleeve=sl, direction="LONG", t_ms=int(t), te=int(t) + S.LAT_MS, price=px, E=px * (1 + S.SLIP / 100),
                 lines="TURNOVER_UNREAD", sole=True, snap={"entry_frenzy_spike_at": _ms("2026-11-22 08:50")},   # spike per the live log
                 bar=str(r.t)[:19], day=DAY0, key=f"ELSAUSDT|{sl}|{str(r.t)[:19]}|SENS") for sl in ("FRENZY_WILLY", "FRENZY_WILLY_HT")]


def main(sp):
    B = S.load_blocked([DEC], since_ms=_ms(DAY0))
    B = B[B.sleeve.str.contains("WILLY") & (B.t_ms < _ms(DAY1))]
    P = S.price_all(pd.concat([B, pd.DataFrame(elsa_rows())], ignore_index=True), int(time.time() * 1000), fetch=True,
                    store_path=os.path.join(sp, "sbf_store.csv"))
    print("| coin | sleeve | refused (UTC) | lines | priced exit | net % |\n|---|---|---|---|---|---|")
    for r in P.itertuples():
        print(f"| {r.pair} | {r.sleeve} | {pd.Timestamp(r.t_ms, unit='ms'):%H:%M} | {r.lines} | {r.why} ({r.status}) | "
              f"{'–' if pd.isna(r.pct) else f'{r.pct:+.2f}'} |")
    F = P[(P.status == "PRICED") & (P.lines != "TURNOVER_UNREAD")]
    for sl, g in F.groupby("sleeve"):
        print(f"filter blocks {sl}: {len(g)} · WR {(g.pct > 0).mean():.0%} · avg {g.pct.mean():+.2f} · Σ {g.pct.sum():+.2f} %")
    print(f"filter blocks ALL: {len(F)} · Σ {F.pct.sum():+.2f} % · avg {F.pct.mean():+.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))
