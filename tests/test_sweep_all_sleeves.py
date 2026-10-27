"""🔎 DECISION_LOG 398: scripts/sweep_separators.py runs on ANY sleeve's real fills (master on today's stack + current batch) and the Scout
SEPARATOR SWEEP section (scripts/scout_sweep_all.py) — real-fill loader, dedup, blind spots, thin-cohort handling, shuffled null, cache.
⚡🎯 DECISION_LOG 397: the SURGE exit-grid walker (scripts/study_surge_exit_grid.py) — TODAY / LOCK lines = the engine, resting TP books its line."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts")); sys.path.insert(0, HERE)
import sweep_separators as SW  # noqa: E402
import scout_sweep_all as SA  # noqa: E402


def _rows(n, strat="SURGE_LONG", start="2026-08-01", sep=True, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="1D")
    x = rng.normal(size=n)
    pnl = np.where(x > 0, 1.0, -1.0) * (1.0 if sep else 0.0) + rng.normal(scale=0.05 if sep else 1.0, size=n)
    return pd.DataFrame(dict(opened_at=t.strftime("%Y-%m-%dT%H:%M:%S"), pair=[f"P{i}USDT" for i in range(n)], direction="LONG",
                             entry_strategy=strat, status="CLOSED", pnl_percentage=pnl, entry_x=x,
                             entry_noise=rng.normal(size=n), entry_sparse=np.where(np.arange(n) % 2 == 0, x, np.nan)))


def _master(tmp, df, keep=True, stack_pct=None):
    m = df.copy()
    m["stack_keep"] = keep
    m["stack_pct"] = m.pnl_percentage if stack_pct is None else stack_pct
    m["pnl_percentage"] = 99.0                     # the as-traded pct must NOT be used for master rows
    p = os.path.join(tmp, "master.csv"); m.to_csv(p, index=False)
    return p


def test_load_real_master_stack_pct_plus_batch_dedup(tmp_path):
    m = _rows(10, start="2026-08-01")
    mp = _master(str(tmp_path), m)
    b = _rows(4, start="2026-10-24", seed=2)
    dup = m.iloc[[0]].assign(pnl_percentage=5.0)    # an export row already in the master → dropped: the master's stack_pct wins (review)
    blocked = _rows(1, start="2026-10-25", seed=3).assign(status="OPEN")
    other = _rows(2, strat="FRENZY_LITE", start="2026-10-24", seed=4)
    ep = os.path.join(str(tmp_path), "scalpars_orders_paper_x.csv")
    pd.concat([b, blocked, other, dup]).to_csv(ep, index=False)
    d = SW.load_real("SURGE_LONG", master=mp, exports=[ep], batch_start="2026-01-01")
    assert len(d) == 14                                                    # 10 master + 4 batch (OPEN out, other sleeve out)
    assert (d.pnl_percentage != 99.0).all()                                # stack_pct, never the as-traded pct
    assert abs(d[d.opened_at == m.opened_at.iloc[0]].pnl_percentage.iloc[0] - m.pnl_percentage.iloc[0]) < 1e-9
    blk = SW.load_real("SURGE_LONG", master=_master(str(tmp_path), m, keep=False), exports=[ep], batch_start="2026-01-01")
    assert m.opened_at.iloc[0] not in set(blk.opened_at)                    # a master row today's stack BLOCKS never re-enters via an export
    assert set(d._src) == {"master", "batch"}
    nk = SW.load_real("SURGE_LONG", master=_master(str(tmp_path), m, keep=False), exports=[], batch_start="2026-01-01")   # noqa
    assert len(nk) == 0                                                    # rows today's stack blocks are not fills of today's sleeve


def test_momentum_direction_split_and_batch_start(tmp_path):
    a = _rows(6, strat="MOMENTUM").assign(direction="LONG")
    s = _rows(5, strat="MOMENTUM", seed=5).assign(direction="SHORT")
    ep = os.path.join(str(tmp_path), "e.csv"); pd.concat([a, s]).to_csv(ep, index=False)
    assert len(SW.load_real("MOMENTUM:LONG", master=None, exports=[ep], batch_start="2026-01-01")) == 6
    assert len(SW.load_real("MOMENTUM:SHORT", master=None, exports=[ep], batch_start="2026-01-01")) == 5
    assert len(SW.load_real("MOMENTUM:LONG", master=None, exports=[ep], batch_start="2027-01-01")) == 0   # before the batch → master's job


def test_run_finds_a_real_separator_and_lists_blind_spots(tmp_path):
    mp = _master(str(tmp_path), _rows(60))
    R = SW.run("SURGE_LONG", master=mp, exports=[])
    assert R["n"] == 60 and "entry_x" in R["cands"] and "entry_sparse" in R["blind"]
    top = R["cons"].iloc[0]
    assert top.col == "entry_x" and top.dA > 0.5 and top.dB > 0.5
    thin = SW.run("SURGE_LONG", master=_master(str(tmp_path), _rows(5)), exports=[])
    assert thin["n"] == 0 and "too few" in thin["note"]


def test_scout_row_beats_null_only_for_a_real_signal(tmp_path):
    sig = SA.sleeve_row("SURGE_LONG", master=_master(str(tmp_path), _rows(80)), exports=[], null_1d=5, null_2d=0)
    assert sig["s1"] >= 1 and sig["top"][0][0] == "x" and sig["best"] > sig["null_best"]
    assert sig["pmax"] == 1 / 6 and "ABOVE LUCK" not in SA.verdict(sig)               # 5 shuffles → p ≥ 1/6: never "above" on 5
    sig50 = SA.sleeve_row("SURGE_LONG", master=_master(str(tmp_path), _rows(80)), exports=[], null_1d=50, null_2d=0)
    assert sig50["pmax"] <= 0.05 and "ABOVE LUCK" in SA.verdict(sig50)                # a real separator beats luck's best split
    noise = SA.sleeve_row("SURGE_LONG", master=_master(str(tmp_path), _rows(80, sep=False, seed=9)), exports=[], null_1d=50, null_2d=0)
    assert "not above luck" in SA.verdict(noise)
    assert "too few" in SA.verdict(dict(n=0))


def test_scout_lines_and_cache(tmp_path):
    mp = _master(str(tmp_path), _rows(40))
    cp = os.path.join(str(tmp_path), "cache.json")
    L = "\n".join(SA.run(["SURGE_LONG", "BEARRUN_SHORT"], master=mp, exports=[], cache_path=cp))
    assert "SEPARATOR SWEEP" in L and "| SURGE_LONG | 40 (40 · 0)" in L and "BEARRUN_SHORT" in L and "keep collecting" in L
    assert os.path.exists(cp)
    calls = []
    orig = SA.sleeve_row
    SA.sleeve_row = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        SA.run(["SURGE_LONG"], master=mp, exports=[], cache_path=cp)
        assert not calls                                                   # same cohort → cached, not re-swept
        _master(str(tmp_path), _rows(41))
        SA.run(["SURGE_LONG"], master=mp, exports=[], cache_path=cp)
        assert calls                                                       # cohort changed → re-swept
    finally:
        SA.sleeve_row = orig


def test_surge_exit_grid_walker():
    import study_surge_exit_grid as G
    G.check_engine_parity()                                                 # TODAY / LOCK_LADDER lines == _bullrun_exit_for
    t0 = 0
    ts = np.arange(1, 7) * 60_000
    pnl = np.array([0.5, 1.32, 0.4, -0.5, -1.25, -1.3])                     # the EVAA shape: armed at +1.32, falls to the stop
    stop, tpf, hold, _ = G.cells(2.69)["TODAY"]
    assert G.walk(ts, pnl, t0, stop, tpf, hold)[:2] == (-1.25, "STOP")
    stop, tpf, hold, _ = G.cells(2.69)["LOCK_ONLY"]
    assert G.walk(ts, pnl, t0, stop, tpf, hold)[:2] == (-0.5, "TRAIL")      # lock line +0.2 after the arm: first print ≤ +0.2 (gaps through)
    stop, tpf, hold, _ = G.cells(2.69)["TP1_SL1.2"]
    assert G.walk(ts, pnl, t0, stop, tpf, hold)[:2] == (1.0, "TP")          # a resting TP books its line, not the print through it


def test_era_split_uses_the_opened_at_format(tmp_path):
    """review: the default split must be 'YYYY-MM-DDTHH:MM:SS' like opened_at — a ' ' split put every fill of the split day in era B."""
    d = _rows(7)
    df, split, _ = SW.prepare("SURGE_LONG", None, master=_master(str(tmp_path), d), exports=[])
    assert "T" in split and split == d.opened_at.iloc[3]
    assert (df.era == "A").sum() == 3 and (df.era == "B").sum() == 4            # fills strictly before the median → A


def test_cache_key_sees_a_backfilled_stamp(tmp_path):
    """review: a COALESCE stamp backfill changes values, not N / Σ P&L — the fingerprint must change."""
    d = _rows(40)
    mp = _master(str(tmp_path), d)
    f1 = SA._fingerprint("SURGE_LONG", mp, [])
    d2 = d.copy(); d2.loc[3, "entry_sparse"] = 1.234
    _master(str(tmp_path), d2)
    assert SA._fingerprint("SURGE_LONG", mp, []) != f1


def test_budget_shows_unswept_rows(tmp_path):
    mp = _master(str(tmp_path), _rows(40))
    L = "\n".join(SA.run(["SURGE_LONG"], master=mp, exports=[], cache_path=os.path.join(str(tmp_path), "c.json"), budget_s=0))
    assert "not swept this run (time budget)" in L


def test_level_columns_are_tested_and_empty_ones_listed(tmp_path):
    d = _rows(60)
    d["entry_flag"] = (d.entry_x > 0).astype(int)                           # a 0 / 1 flag carrying the whole signal
    d["entry_regime"] = np.where(d.entry_x > 0, "BULL", "BEAR")               # text category
    d["entry_never"] = np.nan
    R = SW.run("SURGE_LONG", master=_master(str(tmp_path), d), exports=[])
    assert {"entry_flag", "entry_regime"} <= set(R["levels"]) and "entry_never" in R["empty"]
    assert {"entry_flag", "entry_regime"} <= set(R["cons"].col)
    assert "entry_flag" not in R["res2"].pair.str.cat()                      # 2D stays numeric
