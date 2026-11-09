"""🏦 Nov-9: the MEGACAP_READMIT watch line (scripts/scout_megacap_readmit_watch.py, DECISION_LOG 506) — frozen constants, cohort, every branch of the bar,
the early stop, the not-evaluable hook and the registry / wiring."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_megacap_readmit_watch as W  # noqa: E402


def _f(rows):
    """rows = (day, pair, rank, zone, pct)"""
    return pd.DataFrame([dict(day=d, pair=p, rank=r, zone=z, pct=x, o=pd.Timestamp(d)) for d, p, r, z, x in rows])


EMPTY = pd.DataFrame(columns=["day", "pair", "rank", "zone", "pct", "o"])


def _top(nh, h=(), days=8, pairs=("AAA", "BBB", "CCC")):
    rows = [(f"2026-11-{10 + i % days:02d}", pairs[i % len(pairs)], 5, "not_hot", x) for i, x in enumerate(nh)]
    rows += [(f"2026-11-{10 + i % days:02d}", "DDD", 6, "hot", x) for i, x in enumerate(h)]
    return _f(rows)


def test_frozen_constants():
    assert (W.DL, W.RANK_TOP, W.RANK_CMP, W.HOT_CUT) == (506, 10, 50, 0.50)
    assert (W.N_MIN, W.DAYS_MIN, W.NH_MIN, W.NH_DAYS, W.N_MAX) == (15, 8, 8, 5, 25)
    assert (W.BE_WR, W.CMP_TOL, W.EARLY_N, W.EARLY_AVG, W.SHARE_MAX) == (0.615, 0.10, 8, -0.30, 0.50)


def test_collecting_until_both_floors():
    top = _top([0.3] * 10, h=[0.2] * 6)
    assert W.verdict(top.iloc[:10], EMPTY)[0] == "COLLECTING"          # 10 fills < 15
    assert W.verdict(top, EMPTY)[0] != "COLLECTING"                     # 16 fills, 10 not hot on 8 days → decidable


def test_restore_keep_off_pair_and_collect():
    cmp_ = _f([("2026-11-10", "ZZZ", 30, "not_hot", 0.20)] * 10)
    lose = _top([0.5, -0.6, -0.6, 0.5, -0.6, -0.6, 0.5, -0.6, -0.6, 0.5], h=[0.1] * 6)
    assert W.verdict(lose, cmp_)[0] == W.RESTORE                         # not-hot WR 40 % < 61.5 % ∧ avg < 0
    one_pair = _top([0.5, -0.6, -0.6, 0.5, -0.6, -0.6, 0.5, -0.6, -0.6, 0.5], h=[0.1] * 6, pairs=("HYPE",))
    assert W.verdict(one_pair, cmp_)[0] == W.PAIR_BL                     # one pair carries the loss → blacklist, not the rank block
    good = _top([0.3] * 10, h=[0.1] * 6)
    assert W.verdict(good, cmp_)[0] == W.KEEP_OFF                        # avg ≥ 0 and ≥ comparator − 0.10
    meh = _top([0.6, -0.5] * 5, h=[0.1] * 6)                             # WR 50 %, avg +0.05 but < comparator +0.20 − 0.10
    assert W.verdict(meh, cmp_)[0] == "KEEP COLLECTING"
    assert W.verdict(_top([0.6, -0.5] * 13, h=[]), cmp_)[0] == W.RESTORE   # still undecided at N 25 → restore


def test_early_stop():
    top = _top([-0.6] * 8)
    assert W.verdict(top, EMPTY)[0].startswith(W.RESTORE) and "EARLY STOP" in W.verdict(top, EMPTY)[0]


def test_not_evaluable_wiring_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_megacap_readmit_watch as _mcr")
    assert "_mcr.run(now_ms)" in src[j:j + 200] and "except Exception as _mcr_e" in src[j:j + 400]
    import observe_registry as OR
    assert "scout_megacap_readmit_watch.py" in OR.BOOKED


def test_cohort_from_export(tmp_path):
    p = tmp_path / "scalpars_orders_paper_2026-11-20_10-00-00.csv"
    pd.DataFrame([
        dict(opened_at="2026-11-12T10:00:00", closed_at="2026-11-12T11:00:00", pair="SOLUSDT", direction="LONG", entry_strategy="MOMENTUM", status="CLOSED",
             pnl_percentage=0.4, pnl=4, close_reason="TP", cell_multiplier_source="UNMATCHED", entry_pair_rank=4, entry_mkt_up4h_pct=0.30),
        dict(opened_at="2026-11-12T12:00:00", closed_at="2026-11-12T13:00:00", pair="XRPUSDT", direction="LONG", entry_strategy="MOMENTUM", status="CLOSED",
             pnl_percentage=-0.7, pnl=-7, close_reason="STOP_LOSS", cell_multiplier_source="UNMATCHED", entry_pair_rank=5, entry_mkt_up4h_pct=0.70),
        dict(opened_at="2026-11-12T14:00:00", closed_at="2026-11-12T15:00:00", pair="ADAUSDT", direction="LONG", entry_strategy="MOMENTUM", status="CLOSED",
             pnl_percentage=0.2, pnl=2, close_reason="TP", cell_multiplier_source="UNMATCHED_PROBE", entry_pair_rank=8, entry_mkt_up4h_pct=0.30),
    ]).to_csv(p, index=False)
    c, _ = W.load([str(p)])
    assert list(c.zone) == ["not_hot", "hot"] and list(c["rank"]) == [4, 5]       # probe out, zone from the stamp
    out = "\n".join(W.run(now_ms=int(pd.Timestamp("2026-11-21").value // 10**6), paths=[str(p)], state_path=None,
                          since_ms=int(pd.Timestamp("2026-11-10").value // 10**6)))
    assert "top-10, not hot** | 1 |" in out and "COLLECTING" in out
