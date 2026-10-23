"""🎲² DECISION_LOG 356 (WILLY exit V2) — group 2 scouts: scout_frenzy_exits (tracker 15 FRENZY_WILLY + WILLY_TURNOVER_BLOCKED),
scout_stop_slip (the WILLY stop line), scout_flatbase_short (③ blocked WILLY longs). Proves: the exit-era line prints, a V2 RUNNER fill
(frenzy_willy_tp_target_pct 3.5, closed FRENZY_TP +3.5 or WILLY_FLOOR) is kept OUT of every old-exit comparison, a V2 fill with target
+1.25 stays IN, and the frozen v2 gate's walker stays PINNED to its registered exit. Synthetic, hermetic, no network."""
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, HERE)
import scout_frenzy_exits as SF  # noqa: E402
import willy_exit_era as WEra  # noqa: E402

WR_LABEL = "+1.25 / −4 / 90 min · late +0.25 from min 30 · resting TP"


def _export(rows):
    """orders-export rows: (opened_at, pair, pnl %, close_reason, V2 target or None)."""
    return pd.DataFrame([dict(opened_at=t, pair=p, direction="LONG", entry_strategy="FRENZY_WILLY", status="CLOSED", pnl_percentage=x,
                              pnl=x * 10, close_reason=r, entry_frenzy_willy_trigger="A", entry_price=1.0,
                              frenzy_willy_tp_target_pct=(np.nan if tp is None else tp)) for t, p, x, r, tp in rows])


FILLS = [("2026-10-20T10:00:00", "PREUSDT", 1.25, "FRENZY_TP", None),           # pre-V2 (no stamp)
         ("2026-10-23T10:00:00", "V2FLATUSDT", 1.25, "FRENZY_TP", 1.25),        # V2, target +1.25 → the old exit exactly → stays IN
         ("2026-10-23T11:00:00", "RUNUSDT", 3.5, "FRENZY_TP", 3.5),             # V2 runner at its target → OUT
         ("2026-10-23T12:00:00", "FLOORUSDT", 1.0, "WILLY_FLOOR", 3.5)]         # V2 runner closed on the floor → OUT


def _J():
    return pd.DataFrame(dict(t=["2026-10-23T09:00:00"], e=["WILLY"], pair=["X"], gate=["FRENZY_WILLY_ARMED_A"], strategy=[""]))


# ─────────────── tracker 15 store: the V2 stamp rides the rows ───────────────
def test_willy_rows_carry_the_v2_target_and_merge_never_wipes_it():
    r = SF.willy_rows(_export(FILLS))
    t = r.set_index("pair").tpt
    assert np.isnan(t["PREUSDT"]) and (t["V2FLATUSDT"], t["RUNUSDT"], t["FLOORUSDT"]) == (1.25, 3.5, 3.5)
    newer = r.assign(tpt=np.nan)                                     # an older export without the column surfaces later
    assert SF.willy_merge(r, newer).set_index("pair").tpt["RUNUSDT"] == 3.5


def test_willy_lines_print_the_era_line_and_split_the_all_row():
    w = SF.willy_merge(None, SF.willy_rows(_export(FILLS)))
    txt = "\n".join(SF.willy_lines(w, 0, _J()))
    assert "🎲² exit era (FRENZY_WILLY): pre-V2 1 · V2 3 (runner target / floor 2)" in txt
    assert "| All · old exit (pre-V2 + V2 target +1.25) | 2 | 2 | 100 % | +1.250 |" in txt   # runner + floor fills NOT in it
    assert "| All · 🎲² V2 runner / floor (DL 356) | 2 | 1 | 100 % | +2.250 |" in txt


def test_willy_lines_no_v2_fill_no_split_row_but_era_line():
    w = SF.willy_merge(None, SF.willy_rows(_export(FILLS[:2])))
    txt = "\n".join(SF.willy_lines(w, 0, _J()))
    assert "pre-V2 1 · V2 1 (runner target / floor 0)" in txt and "All · old exit" not in txt


# ─────────────── WILLY_TURNOVER_BLOCKED: as-booked rows + parity keep runners out ───────────────
def _blk(n):
    return pd.DataFrame([dict(t=f"2026-10-{10 + (i % 10):02d}T17:00:{i:02d}", pair=f"P{i}USDT", gate="FRENZY_WILLY_TURNOVER", pct=0.2,
                              how="take profit", final=True, wpct=0.2, wfinal=True, trig="A", R=2.0, price=1.0, src="ticks", entry=1.0)
                         for i in range(n)])


def test_tb_lines_let_through_as_booked_excludes_v2_runners():
    w = SF.willy_merge(None, SF.willy_rows(_export(FILLS))).assign(mcap=np.nan)
    txt = "\n".join(SF.tb_lines(_blk(3), w, {}, ltw=None, par=None, wr_label=WR_LABEL))
    assert "🎲² exit era (WILLY let-through): pre-V2 1 · V2 3 (runner target / floor 2)" in txt
    assert "| let through — WILLY fills (as traded, old exit) | 2 | 2 | 2 | 100 % | +1.250 |" in txt
    assert "| let through — 🎲² V2 runner / floor fills (DL 356, as booked — apart, never in the old-exit rows) | 2 |" in txt
    assert "let-through AS BOOKED 2 · 100 % · +1.250 %" in txt                  # the old as-booked read: runner +3.5 / floor +1.0 out
    assert "2 V2 runner / floor let-through fill(s) re-walked under this PINNED old exit" in txt


def test_lt_parity_drops_v2_runner_keeps_v2_target_125():
    ltw = pd.DataFrame([dict(t="2026-10-23T10:00:00", pair="V2FLATUSDT", booked=1.26, reason="FRENZY_TP", wpct=1.25, wwhy="take profit",
                             wfinal=True, tpt=1.25),
                        dict(t="2026-10-23T11:00:00", pair="RUNUSDT", booked=3.5, reason="FRENZY_TP", wpct=1.25, wwhy="take profit",
                             wfinal=True, tpt=3.5),
                        dict(t="2026-10-23T12:00:00", pair="FLOORUSDT", booked=1.0, reason="WILLY_FLOOR", wpct=1.25, wwhy="take profit",
                             wfinal=True, tpt=3.5),
                        dict(t="2026-10-22T12:00:00", pair="PREUSDT", booked=0.27, reason="WILLY_LATE_TP", wpct=0.32, wwhy="late TP",
                             wfinal=True, tpt=np.nan)])
    par = SF.tb_lt_parity(ltw, SF._ms("2026-10-15T00:00:00"))
    assert par["n"] == 2 and par["same"] == 2 and abs(par["booked"] - (1.26 + 0.27) / 2) < 1e-9
    legacy = ltw.drop(columns=["tpt"]).iloc[[0, 3]]                  # a store walked before 356 (no tpt column) → unchanged behaviour
    assert SF.tb_lt_parity(legacy, SF._ms("2026-10-15T00:00:00"))["n"] == 2


def test_tb_lt_price_carries_the_target_stamp(tmp_path, monkeypatch):
    """a final stored walk keeps its price; the V2 target refreshes onto the row (needed by the parity split)."""
    store = str(tmp_path / "lt.csv")
    w = SF.willy_merge(None, SF.willy_rows(_export(FILLS[2:3])))
    wr = SF.tb_walker_rule_pinned()
    monkeypatch.setattr(SF, "tb_walk", lambda *a, **k: dict(wpct=1.25, wwhy="take profit", wsrc="ticks", wfinal=True))
    out = SF.tb_lt_price(w, SF._ms("2026-10-24T00:00:00"), budget={"dl": 0}, wr=wr, store=store, entries={})
    assert out.tpt.iloc[0] == 3.5 and out.wpct.iloc[0] == 1.25 and out.booked.iloc[0] == 3.5
    out2 = SF.tb_lt_price(w, SF._ms("2026-10-24T00:00:00"), budget={"dl": 0}, wr=wr, store=store, entries={})
    assert float(out2.tpt.iloc[0]) == 3.5


def test_v2_walker_rule_is_pinned_to_its_registered_exit(monkeypatch):
    import scout_willy_hold as WH
    moved = NS(frenzy_willy_tp_pct=3.0, frenzy_willy_stop_pct=2.0, frenzy_willy_max_hold_minutes=240, frenzy_willy_late_tp_minutes=0,
               frenzy_willy_late_tp_pct=0, resting_tp_willy_enabled=False)
    monkeypatch.setattr(WH, "_live_th", lambda: moved)              # the live config moves …
    p = SF.tb_walker_rule_pinned()                                  # … the frozen gate's walker does not
    assert (p["tp"], p["sl"], p["cap"], p["late"], p["rtp"]) == (WEra.OLD_TP, WEra.OLD_STOP, WEra.OLD_HOLD, WEra.OLD_LATE, True)
    assert p["label"] == WR_LABEL                                   # same label → every stored walk stays valid (no re-walk)
    assert SF.tb_walker_rule()["tp"] == 3.0                         # the live reader itself is untouched (flatbase ③ uses it)


# ─────────────── scout_stop_slip: the WILLY stop line (V2 keeps −4; WILLY_HT runs WILLY's exit) ───────────────
def test_stop_slip_willy_and_ht_stop_line_and_floor_not_a_stop():
    import scout_stop_slip as SS
    cfg = {"thresholds": {"frenzy_willy_stop_pct": 4.0, "frenzy_stop_pct": 3.0}}
    for es in ("FRENZY_WILLY", "FRENZY_WILLY_HT"):
        assert SS.stop_line(dict(strategy=es, kind="STOP_LOSS", direction="LONG"), cfg) == (-4.0, 0.0, "frenzy_willy_stop_pct")
    assert SS.classify("WILLY_FLOOR") is None                       # the V2 profit floor is never priced as a stop


# ─────────────── scout_flatbase_short ③: blocked WILLY priced under the fixed exit — said so ───────────────
def test_flatbase_blocked_willy_line_carries_the_v2_note(monkeypatch, tmp_path):
    import scout_flatbase_short as SC
    monkeypatch.setattr(SC, "CACHE", str(tmp_path / "cache"))
    b = pd.DataFrame(dict(t_ms=[1_790_000_000_000, 1_790_000_000_000], pair=["AUSDT", "BUSDT"],
                          gate=["FRENZY_WILLY_PAIR_HELD", "FRENZY_LITE_PAIR_HELD"], sleeve=["FRENZY_WILLY", "FRENZY_LITE"]))
    monkeypatch.setattr(SC, "load_blocks", lambda frame, since: b)
    monkeypatch.setattr(SC, "holder_of", lambda o, pair, t: NS(opened_at="2026-10-23T00:00:00", pair=pair, status="OPEN", net=None, usd=None,
                                                                   funding_fee_usd=None))
    monkeypatch.setattr(SC, "price_long", lambda *a, **k: dict(state="PRICED", pct=1.15, exit_ms=1_790_000_600_000))
    ctx = dict(frame=pd.DataFrame(), since_ms=0, orders=None, bud=NS(stop=False, why=""), now_ms=1_800_000_000_000, fetch=False,
               th_long=NS())
    L = SC.blocked_lines(ctx)
    wl = [x for x in L if x.startswith("- FRENZY_WILLY:")]
    ll = [x for x in L if x.startswith("- FRENZY_LITE:")]
    assert wl and "DL 356 V2 runner target" in wl[0] and ll and "DL 356" not in ll[0]
