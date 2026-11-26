"""⏱ DECISION_LOG 648 — EXH_EARLY_SPIKE observe line: zone rule, forward cohort, frozen two-step bar, registry + scout hook."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import scout_exh_early_spike_648 as B                  # noqa: E402
import observe_registry as OR                     # noqa: E402

DAY = 86_400_000


def _src(p):
    with open(os.path.join(ROOT, p)) as f:
        return f.read()


def test_frozen_constants():
    assert (B.ZONE_MAX, B.BE_WR, B.READS, B.COL, B.DL) == (0.5, 0.4328, ((15, 8), (30, 10)), "entry_frenzy_hours", 648)


def test_zone_is_strictly_below_half_an_hour():
    assert B.zone_of(0.0) == "zone" and B.zone_of(0.25) == "zone" and B.zone_of(0.4999) == "zone"
    assert B.zone_of(0.5) == "rest" and B.zone_of(0.75) == "rest" and B.zone_of(48) == "rest"
    assert B.zone_of(None) == "unscored" and B.zone_of("x") == "unscored" and B.zone_of(float("nan")) == "unscored"
    assert B.zone_of(-0.1) == "unscored"


def _csv(tmp_path, rows):
    base = dict(direction="SHORT", status="CLOSED", entry_strategy="EXHAUSTION_SHORT", close_reason="FRENZY_TP", notional_value=1000.0,
                entry_exh_model_slip_pct=0.0, entry_exh_q5_usd=1e6, entry_exh_tp_pct=7.0, entry_exh_sl_pct=5.0, entry_exh_universe="FRENZY",
                closed_at="2026-12-01T00:00:00", pnl=1.0)
    p = tmp_path / "scalpars_orders_paper_x.csv"
    pd.DataFrame([dict(base, **r) for r in rows]).to_csv(p, index=False)
    return [str(p)]


def _rows(n_days, zone_pct, rest_pct, per_day=2, day0=1):
    r = []
    for d in range(n_days):
        for h in range(per_day):
            r.append(dict(opened_at=f"2026-12-{d + day0:02d}T0{h + 1}:00:00", pair=f"Z{d}{h}USDT", pnl_percentage=zone_pct, entry_frenzy_hours=0.25))
        r.append(dict(opened_at=f"2026-12-{d + day0:02d}T07:00:00", pair=f"R{d}USDT", pnl_percentage=rest_pct, entry_frenzy_hours=3.0))
    return r


def test_forward_cohort_freezes_candidate_once(tmp_path):
    rows = _rows(9, -2.0, 4.0)
    rows.append(dict(opened_at="2026-12-20T05:00:00", pair="LATEUSDT", pnl_percentage=1.0, entry_frenzy_hours=None))   # newest ≥ +24 h, unscored
    rows.append(dict(opened_at="2026-11-01T05:00:00", pair="OLDUSDT", pnl_percentage=-9.0, entry_frenzy_hours=0.1))    # before the floor
    rows.append(dict(opened_at="2026-12-20T04:00:00", pair="NEGUSDT", pnl_percentage=-3.0, entry_frenzy_hours=-0.2))   # (review) negative → unscored
    paths, st = _csv(tmp_path, rows), tmp_path / "state.json"
    since = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    out = "\n".join(B.run(paths=paths, state_path=str(st), since_ms=since, now_ms=since + 40 * DAY))
    assert "| zone (would block: < 0.5 h after the spike) | 18 | 9 | 0% | -2.000 % |" in out
    assert "| UNSCORED (no stamp) — never counted | 2 |" in out and "OLDUSDT" not in out
    assert "Frozen verdict @ 15 fills" in out and B.CANDIDATE in out and "@ 30" not in out    # a candidate is never re-read
    out2 = "\n".join(B.run(paths=paths, state_path=str(st), since_ms=since, now_ms=since + 41 * DAY))
    assert out2.count("Frozen verdict") == 1


def test_no_freeze_until_every_counted_fill_has_closed(tmp_path):
    rows = _rows(8, -2.0, 4.0)                    # 16 zone fills on 8 days: crossing 12-08 01:00, newest 12-08 07:00 → < 24 h later → not frozen yet
    out = "\n".join(B.run(paths=_csv(tmp_path, rows), state_path=str(tmp_path / "s.json"),
                          since_ms=int(pd.Timestamp("2026-11-30").value // 1_000_000), now_ms=int(pd.Timestamp("2026-12-31").value // 1_000_000)))
    assert "Frozen verdict" not in out and "Now:" in out


def test_second_read_only_after_keep_observing(tmp_path):
    # zone fills alternate −3 / +2 (WR 50–53 % ≥ the 43.3 % breakeven, mean < 0, below the rest) → KEEP OBSERVING at 15 / 8 days; at 30 / 10 days → second read
    rows = []
    for d in range(16):
        for h in range(2):
            pct = -3.0 if (d + h) % 2 == 0 else 2.0
            rows.append(dict(opened_at=f"2026-12-{d + 1:02d}T0{h + 1}:00:00", pair=f"Z{d}{h}USDT", pnl_percentage=pct, entry_frenzy_hours=0.4))
        rows.append(dict(opened_at=f"2026-12-{d + 1:02d}T07:00:00", pair=f"R{d}USDT", pnl_percentage=4.0, entry_frenzy_hours=6.0))
    rows.append(dict(opened_at="2026-12-28T05:00:00", pair="LATEUSDT", pnl_percentage=1.0, entry_frenzy_hours=12.0))
    st = tmp_path / "s.json"
    out = "\n".join(B.run(paths=_csv(tmp_path, rows), state_path=str(st), since_ms=int(pd.Timestamp("2026-11-30").value // 1_000_000),
                          now_ms=int(pd.Timestamp("2027-01-05").value // 1_000_000)))
    assert "Frozen verdict @ 15 fills" in out and "Frozen verdict @ 30 fills" in out
    import json
    s = json.load(open(st))
    assert s["EXH_EARLY_SPIKE"]["state"] == "KEEP OBSERVING" and s["EXH_EARLY_SPIKE@30"]["state"] == "KEEP OBSERVING"


def test_retired_first_read_blocks_the_second_and_live_bar_stays(tmp_path):
    rows = _rows(16, 2.0, -1.0)                   # zone wins → RETIRED at the first read
    rows.append(dict(opened_at="2026-12-28T05:00:00", pair="LATEUSDT", pnl_percentage=1.0, entry_frenzy_hours=12.0))
    st = tmp_path / "s.json"
    out = "\n".join(B.run(paths=_csv(tmp_path, rows), state_path=str(st), since_ms=int(pd.Timestamp("2026-11-30").value // 1_000_000),
                          now_ms=int(pd.Timestamp("2027-01-05").value // 1_000_000)))
    assert "Frozen verdict @ 15 fills" in out and "RETIRED" in out and "@ 30 fills" not in out and "/30" not in out


def test_malformed_record_is_flagged_and_never_rejudged(tmp_path):
    import json
    st = tmp_path / "s.json"
    good = dict(state="KEEP OBSERVING", detail="d", at="x", run_at="y")
    st.write_text(json.dumps({"EXH_EARLY_SPIKE": good, "EXH_EARLY_SPIKE@30": {"oops": 1}}))
    rows = _rows(16, -3.0, 4.0) + [dict(opened_at="2026-12-28T05:00:00", pair="LATEUSDT", pnl_percentage=1.0, entry_frenzy_hours=12.0)]
    out = "\n".join(B.run(paths=_csv(tmp_path, rows), state_path=str(st), since_ms=int(pd.Timestamp("2026-11-30").value // 1_000_000),
                          now_ms=int(pd.Timestamp("2027-01-05").value // 1_000_000)))
    assert "Frozen verdict @ 15 fills" in out and "malformed" in out and json.load(open(st))["EXH_EARLY_SPIKE@30"] == {"oops": 1}


def test_verdict_retires_a_winning_or_not_worse_zone():
    z = pd.DataFrame(dict(pct=[1.0] * 15, day=[f"d{i % 8}" for i in range(15)], pair=["P"] * 15))
    r = pd.DataFrame(dict(pct=[0.5] * 5, day=["d0"] * 5, pair=["Q"] * 5))
    assert B.verdict(z, r)[0] == "RETIRED"
    zl = z.assign(pct=-1.0)
    assert B.verdict(zl, r.assign(pct=-2.0))[0] == "RETIRED"          # zone not below the rest
    assert B.verdict(z.iloc[:5], r)[0] == "COLLECTING"
    assert B.verdict(zl, r.iloc[:0])[0] == "KEEP OBSERVING"           # no rest fills → never a candidate


def test_not_evaluable_without_the_commit(monkeypatch):
    monkeypatch.setattr(B._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert any("NOT EVALUABLE" in x for x in B.run(state_path=None))


def test_registry_and_scout_hook():
    assert "scout_exh_early_spike_648.py" in OR.classified()
    assert OR.BOOKED["scout_exh_early_spike_648.py"]["exhaustion_tp_cap_pct"] == 7.0
    op = _src("scripts/opportunity_scout.py")
    assert "import scout_exh_early_spike_648 as _ees" in op and "_rg_sec += _ees.run(now_ms)" in op
    assert "reports/SCOUT_EXH_EARLY_SPIKE_648.json" in _src(".gitignore")
