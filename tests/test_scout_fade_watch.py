"""🩸 SPIKE_FADE watch lines (DECISION_LOG 374): the frozen weak / strong split (ADX Δ < 2.359 · pair ADX ≤ 19.2; missing stamp =
unscored), the CLAUDE.md expectancy filter bar (COLLECTING → FILTER_CANDIDATE / KEEP_OBSERVING at N ≥ 15 on ≥ 8 UTC days), the high-mcap
1.5× shadow (incl. the liquidity-capped re-ticket) and its first-10 read, decide-once persistence, the not-evaluable branch, and no writes
outside tmp. Synthetic CSVs — no network."""
import datetime as dt
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_fade_watch as S  # noqa: E402

T0 = dt.datetime(2026, 10, 27, 1, 0)
SINCE = int(pd.Timestamp(T0).value // 1_000_000)


def _row(i, y, adx=25.0, dadx=3.0, day=None, pair=None, mcap=None, mult=2.0, notional=20000.0, desired=None, liqcap=None, capped=False,
         strat="SPIKE_FADE", reason=None, status="CLOSED", paper=True, offset_min=0):
    o = T0 + dt.timedelta(days=i if day is None else day, minutes=i + offset_min)
    return dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), pair=pair or f"P{i}USDT", direction="SHORT", status=status, entry_strategy=strat,
                pnl=y * notional / 100.0, pnl_percentage=y, close_reason=reason or ("RUNNER_TRAIL" if y > 0 else "STOP_LOSS L1"),
                is_paper=paper, entry_adx=adx, entry_adx_delta=dadx, entry_mcap_usd=mcap, cell_multiplier=mult, notional_value=notional,
                entry_desired_notional=desired if desired is not None else notional, entry_liquidity_cap_notional=liqcap, liquidity_capped=capped)


def _csv(tmp_path, rows, name="o.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def _fw(tmp_path, rows):
    fw, _ = S.forward(S.load([_csv(tmp_path, rows)]), pd.Timestamp(SINCE, unit="ms"))
    return fw


def test_frozen_split_and_unscored():
    df = pd.DataFrame(dict(entry_adx_delta=[2.359, 2.3589, None, -1.0], entry_adx=[19.2, 19.21, 5.0, None]))
    assert list(S.side(df, "ADX_RISING")) == ["strong", "weak", "unscored", "weak"]
    assert list(S.side(df, "PAIR_ADX_LOW")) == ["weak", "strong", "weak", "unscored"]


def test_forward_cohort_filters(tmp_path):
    rows = [_row(0, 0.4), _row(1, -1.5, reason="MANUAL"), _row(2, 0.3, strat="SPIKE_CHASE"), _row(3, 0.3, status="OPEN"),
            _row(4, 0.3, day=-1), _row(5, 0.2, reason="EXTERNAL_CLOSE")]
    rows.append(dict(rows[0], status="OPEN"))                        # duplicate key: the CLOSED row wins
    out = "\n".join(S.run(paths=[_csv(tmp_path, rows)], since_ms=SINCE, state_path=None, mcap_since_ms=SINCE))
    assert "N 1 · 1 days" in out and "operator-closed: 2 — excluded" in out


def test_not_evaluable_counts_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: None)
    st = tmp_path / "st.json"
    out = "\n".join(S.run(paths=[_csv(tmp_path, [_row(0, -1.5)])], state_path=str(st)))
    assert "NOT EVALUABLE" in out and "Forward cohort" not in out and not st.exists()


def test_collecting_below_the_bar(tmp_path):
    fw = _fw(tmp_path, [_row(i, -1.5, dadx=0.0, day=i) for i in range(14)])
    v, txt = S.filter_read(fw, "ADX_RISING")
    assert v == "COLLECTING" and "14/15" in txt
    fw = _fw(tmp_path, [_row(i, -1.5, dadx=0.0, day=i % 7) for i in range(20)])      # N 20 but only 7 days
    assert S.filter_read(fw, "ADX_RISING")[0] == "COLLECTING"


def test_filter_candidate_when_the_weak_side_loses(tmp_path):
    weak = [_row(i, -1.5 if i % 4 else 0.5, adx=15.0, day=i // 2, pair=f"W{i % 8}USDT") for i in range(16)]   # 16 · 8 days · 25 %
    strong = [_row(100 + i, 0.5 if i % 6 else -1.5, adx=30.0, day=i // 3) for i in range(24)]
    v, txt = S.filter_read(_fw(tmp_path, weak + strong), "PAIR_ADX_LOW")
    assert v == "FILTER_CANDIDATE", txt
    assert "haircut" in txt and "first 10 blocked fades" in txt


def test_keep_observing_when_the_weak_side_wins(tmp_path):
    weak = [_row(i, 0.5 if i % 6 else -1.5, dadx=0.0, day=i // 2) for i in range(16)]
    v, txt = S.filter_read(_fw(tmp_path, weak), "ADX_RISING")
    assert v == "KEEP_OBSERVING", txt


def test_keep_observing_when_one_pair_carries_the_loss(tmp_path):
    weak = [_row(i, (-1.5 if i % 4 else 0.5), dadx=0.0, day=i // 2, pair="ONEUSDT" if i % 4 else f"X{i}USDT") for i in range(16)]
    v, txt = S.filter_read(_fw(tmp_path, weak), "ADX_RISING")
    assert v == "KEEP_OBSERVING" and "pair 100%" in txt


def test_breakeven_fallback_and_sleeve_formula():
    assert S.breakeven([0.5, 0.5, -1.5])[0] == 72.0                   # < 5 losses → registration breakeven
    be, fb = S.breakeven([0.5] * 10 + [-1.5] * 5)
    assert not fb and abs(be - 75.0) < 1e-9                          # 1.5 / (0.5 + 1.5)


def test_shadow_plain_and_liquidity_capped():
    plain = dict(usd=-300.0, cell_multiplier=2.0, notional_value=20000.0, entry_desired_notional=20000.0, liquidity_capped=False)
    assert S.shadow_usd(plain) == (-225.0, "")
    assert S.shadow_usd(plain, 1.0)[0] == -150.0                     # the 1× equivalent
    one = dict(plain, cell_multiplier=1.0)
    assert S.shadow_usd(one)[0] == -300.0                            # never above the traded size
    # liquidity-capped: desired 21,000 × 0.75 = 15,750 < cap 20,000 → −300 × 15,750 / 20,000
    lc = dict(usd=-300.0, cell_multiplier=2.0, notional_value=20000.0, entry_desired_notional=21000.0, entry_liquidity_cap_notional=20000.0,
              liquidity_capped=True)
    s, lab = S.shadow_usd(lc)
    assert lab == "liquidity-capped" and abs(s - (-236.25)) < 1e-9
    # a tight cap binds: desired 40,000 × 0.75 = 30,000 but the book cap is 10,000 = the traded notional → unchanged
    tight = dict(lc, notional_value=10000.0, entry_desired_notional=40000.0, entry_liquidity_cap_notional=10000.0, usd=-150.0)
    assert S.shadow_usd(tight)[0] == -150.0


def test_shadow_labels_sized_below_and_desired_unknown():
    # notional < desired WITHOUT the flag (PARTI 10-23 class) → re-ticketed, labelled 'sized below desired', never 'liquidity-capped'
    below = dict(usd=-150.0, cell_multiplier=2.0, notional_value=10000.0, entry_desired_notional=40000.0, entry_liquidity_cap_notional=None,
                 liquidity_capped=False)
    assert S.shadow_usd(below) == (-150.0, "sized below desired")
    # flag true but desired unknown → pnl unchanged (no saving), flagged — never × 0.75
    unk = dict(usd=-300.0, cell_multiplier=2.0, notional_value=20000.0, entry_desired_notional=None, entry_liquidity_cap_notional=20000.0,
               liquidity_capped=True)
    s, lab = S.shadow_usd(unk)
    assert s == -300.0 and "desired unknown" in lab


def test_mcap_line_unscored_and_avg_pct_read(tmp_path):
    rows = [_row(i, 0.4, mcap=300e6, day=i) for i in range(9)] + [_row(50, 0.3, mcap=None), _row(51, 0.3, mcap=50e6)]
    fw = _fw(tmp_path, rows)
    eq, _ = S.ref_equity([_csv(tmp_path, rows, "e.csv")])
    h, uns = S.mcap_rows(fw, eq)
    assert len(h) == 9 and uns == 1
    assert S.mcap_read(h)[0] == "COLLECTING"
    rows.append(_row(9, 0.4, mcap=250e6, day=9))
    h, _ = S.mcap_rows(_fw(tmp_path, rows), eq)
    assert S.mcap_read(h)[0] == "KEEP_2X"
    # one −1.5 % fill (−$300 = 10.3 % of $2,900) but the first-10 avg is still +0.21 % → KEEP_2X (DL 311: avg P&L % < 0, not $ / one loss)
    rows[3] = _row(3, -1.5, mcap=300e6, day=3)
    h, _ = S.mcap_rows(_fw(tmp_path, rows), eq)
    v, txt = S.mcap_read(h)
    assert v == "KEEP_2X" and "1 lost > 4 %" in txt
    assert h.loss_pct15.max() > 4.0 and h.loss_pct1.max() > 4.0       # 1.5× and 1× still breach 4 % at the −1.5 % stop
    # avg P&L % < 0 on the first 10 → CAP_CANDIDATE
    neg = [_row(i, -0.3 if i % 2 else 0.1, mcap=300e6, day=i, notional=1000.0) for i in range(10)]
    h, _ = S.mcap_rows(_fw(tmp_path, neg), eq)
    assert S.mcap_read(h)[0] == "CAP_CANDIDATE"


def test_mcap_cohort_starts_at_the_dl311_floor(tmp_path, monkeypatch):
    # a high-mcap fade opened BEFORE the DL 374 floor (but after the DL 311 floor) counts in ③, never in ① ②
    early = _row(0, -1.5, mcap=300e6, day=-3)
    late = _row(1, 0.4, mcap=300e6, day=1)
    p = _csv(tmp_path, [early, late])
    floor311 = SINCE - 5 * 86_400_000
    out = "\n".join(S.run(paths=[p], since_ms=SINCE, state_path=None, mcap_since_ms=floor311))
    assert "Forward cohort ① ② ④ ⑤ ⑥: closed SPIKE_FADE fills opened ≥ 2026-10-27 01:00 UTC (override) · paper + live · N 1 " in out
    assert out.count("mcap $300M") == 2 and "(override)" in out
    # the floor resolves from the DL 311 commit; not in git → the pinned 2026-10-19 05:08 UTC
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: None)
    ms, how = S.mcap_floor_ms()
    assert pd.Timestamp(ms, unit="ms") == pd.Timestamp("2026-10-19 05:08") and "pinned" in how


def test_line3_all_nan_usd_never_blanks_lines_1_2(tmp_path):
    rows = [dict(_row(i, 0.4, mcap=300e6, day=i), pnl=None) for i in range(3)]
    out = "\n".join(S.run(paths=[_csv(tmp_path, rows)], since_ms=SINCE, state_path=None, mcap_since_ms=SINCE))
    assert "① ADX rising" in out and "② Pair ADX" in out and "no high-mcap fade yet" in out


def test_ref_equity_balance_column_and_assumption(tmp_path):
    eq, src = S.ref_equity([_csv(tmp_path, [_row(0, 0.4)])])
    assert eq == {"paper": 2900.0, "live": 1225.0} and "ASSUMED" in src


def test_ref_equity_keyed_by_book(tmp_path):
    r = [dict(_row(0, 0.4), balance=5000.0), dict(_row(1, 0.4), balance=6000.0), dict(_row(2, 0.4, paper=False), balance=1300.0)]
    eq, src = S.ref_equity([_csv(tmp_path, r, "b.csv")])
    assert eq == {"paper": 6000.0, "live": 1300.0} and "balance" in src
    eq, src = S.ref_equity([_csv(tmp_path, r[:2], "c.csv")])                        # paper only → live falls back to the assumption
    assert eq == {"paper": 6000.0, "live": 1225.0} and "live ASSUMED" in src


def test_decide_once_persists_and_never_moves(tmp_path):
    st = tmp_path / "st.json"
    weak = [_row(i, 0.5 if i % 6 else -1.5, dadx=0.0, day=i // 2) for i in range(16)]
    p1 = _csv(tmp_path, weak, "a.csv")
    out = "\n".join(S.run(paths=[p1], since_ms=SINCE, state_path=str(st), mcap_since_ms=SINCE))
    assert "KEEP_OBSERVING" in out and st.exists()
    import json
    saved = json.loads(st.read_text())
    assert set(saved) == {"ADX_RISING"}                              # COLLECTING lines (② / ③) never persisted
    bad = [_row(i, -1.5, dadx=0.0, day=i // 2, pair=f"Z{i}USDT") for i in range(40)]
    out2 = "\n".join(S.run(paths=[_csv(tmp_path, bad, "b.csv")], since_ms=SINCE, state_path=str(st), mcap_since_ms=SINCE))
    assert "FILTER_CANDIDATE" not in out2 and "(frozen " in out2


def test_run_writes_nothing_outside_tmp(tmp_path):
    before = set(os.listdir(os.path.join(ROOT, "reports")))
    st = tmp_path / "st.json"
    out = S.run(paths=[_csv(tmp_path, [_row(i, 0.4, mcap=300e6, day=i) for i in range(12)])], since_ms=SINCE, state_path=str(st), mcap_since_ms=SINCE)
    assert any("KEEP_2X" in x for x in out)
    assert set(os.listdir(os.path.join(ROOT, "reports"))) == before
    assert set(os.listdir(tmp_path)) <= {"o.csv", "st.json"}


# ── ④ HEALTHY_BEAR size shadow ──
def _hb(i, y, day, regime="HEALTHY_BEAR", gap=0.0, **kw):
    return dict(_row(i, y, day=day, **kw), entry_btc_regime=regime, entry_btc_trend_gap_pct=gap)


def test_hb_split_and_shadow_lines(tmp_path):
    rows = [_hb(0, -1.5, 0), _hb(1, 0.4, 1, gap=-0.5), _hb(2, 0.4, 2, regime="STRONG_BEAR"), _hb(3, 0.4, 3, regime=None)]
    fw = _fw(tmp_path, rows)
    assert list(S.hb_split(fw)) == ["hb", "hb", "other", "unscored"]
    out = "\n".join(S.hb_lines(fw, None))
    assert "- HEALTHY_BEAR: N 2" in out and "- other regimes: N 1" in out and "- unscored (no regime stamp): N 1" in out
    assert "41h sub-zone (HEALTHY_BEAR ∧ entry_btc_trend_gap_pct ≥ -0.13): N 1" in out
    assert "traded $-300.00 vs 1× $-150.00" in out                     # 2× → 1× halves the $
    assert "paper: Σ traded $-220.00 vs 1× $-110.00 (Δ $+110.00)" in out
    assert S.shadow_usd(dict(usd=-300.0, cell_multiplier=2.0, notional_value=20000.0, entry_desired_notional=21000.0,
                             entry_liquidity_cap_notional=20000.0, liquidity_capped=True), S.HB_MULT)[0] == -300.0 * 10500 / 20000


def test_hb_read_collecting_candidate_keep(tmp_path):
    few = _fw(tmp_path, [_hb(i, -1.5, i) for i in range(14)])
    assert S.hb_read(few)[0] == "COLLECTING"
    bad = _fw(tmp_path, [_hb(i, -1.5 if i % 3 else 0.4, i // 2) for i in range(16)])     # 16 · 8 days · avg < 0, spread
    v, txt = S.hb_read(bad)
    assert v == "SIZE_CUT_CANDIDATE" and "0.95 filter bar" in txt, txt
    good = _fw(tmp_path, [_hb(i, 0.4 if i % 4 else -1.5, i // 2) for i in range(16)])
    assert S.hb_read(good)[0] == "KEEP_2X"
    one_day = _fw(tmp_path, [_hb(i, 0.4, i) for i in range(15)] + [_hb(20, -9.0, 3)])      # loss concentrated on one day
    assert S.hb_read(one_day)[0] == "KEEP_2X"


def test_hb_decide_once(tmp_path):
    st = tmp_path / "st.json"
    good = [_hb(i, 0.4 if i % 4 else -1.5, i // 2) for i in range(16)]
    out = "\n".join(S.run(paths=[_csv(tmp_path, good, "g.csv")], since_ms=SINCE, state_path=str(st), mcap_since_ms=SINCE))
    assert "④" in out and "KEEP_2X" in out
    import json
    assert "HB_SIZE" in json.loads(st.read_text())
    bad = [_hb(i, -1.5, i // 2, pair=f"Q{i}USDT") for i in range(30)]
    out2 = "\n".join(S.run(paths=[_csv(tmp_path, bad, "b.csv")], since_ms=SINCE, state_path=str(st), mcap_since_ms=SINCE))
    assert "else KEEP_2X): KEEP_2X — read at 15 fades" in out2 and "→ 🚨 SIZE-CUT" not in out2 and "(frozen " in out2


# ── ⑤ HB + ADX not rising · ⑥ HB + BTC ADX high (entry-filter candidates, expectancy bar) ──
def _z(i, y, day, regime="HEALTHY_BEAR", dadx=0.0, badx=30.0, **kw):
    return dict(_row(i, y, day=day, dadx=dadx, **kw), entry_btc_regime=regime, entry_btc_trend_gap_pct=0.0, entry_btc_adx_prev=badx)


def test_hb_zone_split_at_frozen_thresholds_and_unscored(tmp_path):
    rows = [_z(0, 0.4, 0, dadx=2.4341), _z(1, 0.4, 1, dadx=2.4342), _z(2, 0.4, 2, regime="STRONG_BEAR", dadx=0.0),
            _z(3, 0.4, 3, dadx=None), _z(4, 0.4, 4, regime=None), _z(5, 0.4, 5, badx=24.5874), _z(6, 0.4, 6, badx=None)]
    fw = _fw(tmp_path, rows)
    assert list(S.side(fw, "HB_ADXFLAT")) == ["weak", "strong", "strong", "unscored", "unscored", "weak", "weak"]
    assert list(S.side(fw, "HB_BTCADX")) == ["weak", "weak", "strong", "weak", "unscored", "strong", "unscored"]


def test_hb_zone_reads(tmp_path):
    few = _fw(tmp_path, [_z(i, -1.5, i) for i in range(14)])
    assert S.filter_read(few, "HB_ADXFLAT")[0] == "COLLECTING"
    zone = [_z(i, -1.5 if i % 4 else 0.5, i // 2, pair=f"W{i % 8}USDT") for i in range(16)]           # 16 · 8 days · 25 %
    rest = [_z(100 + i, 0.5 if i % 6 else -1.5, i // 3, regime="CHOPPY_WEAK") for i in range(24)]
    fw = _fw(tmp_path, zone + rest)
    for key in ("HB_ADXFLAT", "HB_BTCADX"):
        v, txt = S.filter_read(fw, key)
        assert v == "FILTER_CANDIDATE" and "first 10 blocked fades" in txt, (key, txt)
    win = _fw(tmp_path, [_z(i, 0.5 if i % 6 else -1.5, i // 2) for i in range(16)])
    assert S.filter_read(win, "HB_BTCADX")[0] == "KEEP_OBSERVING"


def test_hb_zone_decide_once(tmp_path):
    import json
    st = tmp_path / "st.json"
    win = [_z(i, 0.5 if i % 6 else -1.5, i // 2) for i in range(16)]
    out = "\n".join(S.run(paths=[_csv(tmp_path, win, "w.csv")], since_ms=SINCE, state_path=str(st), mcap_since_ms=SINCE))
    assert "⑤ HEALTHY_BEAR + ADX not rising" in out and "⑥ HEALTHY_BEAR + BTC ADX high" in out
    saved = json.loads(st.read_text())
    assert saved["HB_ADXFLAT"]["verdict"] == "KEEP_OBSERVING" and saved["HB_BTCADX"]["verdict"] == "KEEP_OBSERVING"
    lose = [_z(i, -1.5, i // 2, pair=f"Q{i}USDT") for i in range(30)]
    out2 = "\n".join(S.run(paths=[_csv(tmp_path, lose, "l.csv")], since_ms=SINCE, state_path=str(st), mcap_since_ms=SINCE))
    assert "→ 🚨 FILTER CANDIDATE" not in out2 and out2.count("KEEP_OBSERVING — read at") >= 2 and "(frozen " in out2


# ── final review: HOLD while a prefix fade is still OPEN · notional-unknown shadow ──
def test_hold_while_a_prefix_fade_is_open(tmp_path):
    import json
    st = tmp_path / "st.json"
    zone = [_z(i, 0.5 if i % 6 else -1.5, i // 2) for i in range(16)]
    zone[15]["opened_at"] = (T0 + dt.timedelta(days=7, hours=12)).strftime("%Y-%m-%dT%H:%M:%S")   # crossing fill (day 7)
    open_early = dict(_z(50, 0.0, 3, mcap=300e6), status="OPEN")                                   # opened before the crossing
    out = "\n".join(S.run(paths=[_csv(tmp_path, zone + [open_early], "h.csv")], since_ms=SINCE, state_path=str(st), mcap_since_ms=SINCE))
    assert out.count("HOLD — ⏸ HOLD") >= 4 and not st.exists()        # ① ⑤ ⑥ ④ held, nothing persisted
    fw = _fw(tmp_path, zone)
    pend = pd.Series([pd.Timestamp(T0 + dt.timedelta(days=3))])
    assert S.filter_read(fw, "HB_ADXFLAT", pend)[0] == "HOLD"
    assert S.hb_read(fw, pend)[0] == "HOLD"
    late = pd.Series([pd.Timestamp(T0 + dt.timedelta(days=20))])                                  # opened after the crossing → no hold
    assert S.filter_read(fw, "HB_ADXFLAT", late)[0] == "KEEP_OBSERVING"
    hi = [_row(i, 0.4, mcap=300e6, day=i) for i in range(10)]
    eq, _ = S.ref_equity([_csv(tmp_path, hi, "m.csv")])
    h, _ = S.mcap_rows(_fw(tmp_path, hi), eq)
    assert S.mcap_read(h, pend)[0] == "HOLD" and S.mcap_read(h, late)[0] == "KEEP_2X"
    out2 = "\n".join(S.run(paths=[_csv(tmp_path, zone, "h2.csv")], since_ms=SINCE, state_path=str(st), mcap_since_ms=SINCE))
    assert "HOLD — ⏸" not in out2 and "HB_ADXFLAT" in json.loads(st.read_text())   # closed out → decided and persisted


def test_shadow_notional_unknown_no_saving():
    r = dict(usd=-300.0, cell_multiplier=2.0, notional_value=None, entry_desired_notional=21000.0, entry_liquidity_cap_notional=20000.0,
             liquidity_capped=True)
    assert S.shadow_usd(r) == (-300.0, "liquidity-capped, notional unknown — no saving")
    assert S.shadow_usd(dict(r, notional_value=0.0))[1] == "liquidity-capped, notional unknown — no saving"
