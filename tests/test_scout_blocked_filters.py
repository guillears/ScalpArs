"""🧱💲 (DECISION_LOG 529) the blocked-trade pricer + per-line revert gates (scripts/scout_blocked_filters.py): BLOCKED-row loading (dedup,
latency / slip ruler, snapshot), the fade pricer on the engine's own exit, the gate (collecting → keep / revert, frozen once, kept fills on the
same pricer), the hook and the registry. Hermetic: temp exports and state, pricers stubbed where they would need network."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_blocked_filters as S  # noqa: E402

T0 = pd.Timestamp("2026-11-12 00:00")
SINCE = int(T0.value // 10**6)


def _dec(tmp_path, rows, name="scalpars_decisions_paper_2026-12-01_00-00-00.csv"):
    out = []
    for i, (d, pair, sleeve, lines, px, sole, dr) in enumerate(rows):
        t = T0 + pd.Timedelta(days=d, minutes=i * 7)
        out.append(dict(t=t.strftime("%Y-%m-%dT%H:%M:%S.000"), e="BLOCKED", pair=pair, dir=dr, price=px, gate=lines, strategy=sleeve,
                        bl_sleeve=sleeve, bl_bar_at=t.strftime("%Y-%m-%dT%H:%M:00.000"), bl_lines=lines, bl_sole=sole, bl_is_paper=True,
                        bl_entry_fields=json.dumps({"entry_atr_pct": 0.5})))
    out.append(dict(t=T0.strftime("%Y-%m-%dT%H:%M:%S.000"), e="BLOCK", pair="XUSDT", dir="LONG", gate="OTHER"))   # not a BLOCKED row
    p = tmp_path / name
    pd.DataFrame(out).to_csv(p, index=False)
    return [str(p)]


FADE_T = dict(spike_fade_sl_pct=-1.5, max_holding_time_minutes=1200, spike_fade_late_arm_after_min=15.0, spike_fade_late_arm_peak=0.3,
              runner_trail_short_arm_peak=0.4, runner_trail_short_atr_mult=0.5, runner_trail_short_giveback_frac=0.35,
              runner_trail_short_use_atr=True, runner_trail_short_be_ratchet_enabled=False, hard_tp_enabled=True,
              hard_tp_ladder_short="1.0:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80")


def test_constants_and_config_exits():
    assert (S.DL, S.FLOOR_DL, S.LAT_MS, S.SLIP, S.N_MIN, S.DAYS_MIN, S.K_MIN) == (529, 527, 8000, 0.10, 15, 8, 5)
    t = dict(frenzy_stop_pct=3.0, frenzy_lite_tp_pct=3.0, frenzy_tp_pct=1.5, frenzy_long_stop_pct=3.0, frenzy_max_hold_minutes=720,
             frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, spike_fade_sl_pct=-1.5, max_holding_time_minutes=1200)
    assert S.exit_of("FRENZY_LITE", t) == dict(tp=3.0, sl=3.0, cap=720) and S.exit_of("FRENZY_LONG", t) == dict(tp=1.5, sl=3.0, cap=720)
    assert S.exit_of("FRENZY_WILLY", t)["sl"] == 4.0 and S.exit_of("SPIKE_FADE", t) == dict(sl=-1.5, cap=1200)
    live = S.th()
    assert "max_holding_time_minutes" in live                                            # read from the investment block (review L1)
    gi = open(os.path.join(ROOT, ".gitignore")).read().split("\n")
    assert "reports/SCOUT_BLOCKED_FILTERS.csv" in gi and "reports/.scout_blocked_filters_gate.json" in gi
    with pytest.raises(AttributeError):
        S.SimpleTh({}).runner_trail_short_use_atr                                        # missing key → the engine's own default applies


def test_load_blocked_ruler_and_dedup(tmp_path):
    paths = _dec(tmp_path, [(0, "AUSDT", "FRENZY_LITE", "LITE_BTC_EMA20_DOWN", 1.0, True, "LONG"),
                            (0, "BUSDT", "SPIKE_FADE", "FADE_LATE_WEAK", 2.0, True, "SHORT")])
    paths += _dec(tmp_path, [(0, "AUSDT", "FRENZY_LITE", "LITE_BTC_EMA20_DOWN", 1.0, True, "LONG")], "scalpars_decisions_paper_2026-12-02_00-00-00.csv")
    B = S.load_blocked(paths, SINCE)
    assert len(B) == 2                                                                   # the same (pair, sleeve, bar) in two exports → one
    a = B[B.pair == "AUSDT"].iloc[0]; b = B[B.pair == "BUSDT"].iloc[0]
    assert a.te == a.t_ms + 8000 and a.E == pytest.approx(1.001) and b.E == pytest.approx(2 * 0.999)   # slip against the trade
    assert a.snap["entry_atr_pct"] == 0.5 and bool(a.sole) is True
    assert len(S.load_blocked(paths, SINCE + 10 * 86_400_000)) == 0                     # the floor


def _bars(first, rows):
    return pd.DataFrame(dict(open_time=[first + i * 60_000 for i in range(len(rows))], o=[r[0] for r in rows], h=[r[1] for r in rows],
                             l=[r[2] for r in rows], c=[r[3] for r in rows]))


def test_fade_pricer_live_exit_line_gap_and_late_arm():
    te = SINCE + 8000
    r = dict(pair="X", sleeve="SPIKE_FADE", te=te, E=100.0, snap={"entry_atr_pct": 0.5})
    first = -(-te // 60_000) * 60_000
    up = _bars(first, [(100.2, 100.5, 99.9, 100.2), (100.2, 101.0, 100.2, 100.8), (100.8, 102.0, 101.0, 101.5)])
    p, why, st, ex = S.price_fade(r, FADE_T, te + 3 * 86_400_000, k1m=up)
    assert st == "PRICED" and why == "STOP" and p == -1.5 and ex == first + 3 * 60_000            # books the LINE, not the bar high
    gap = _bars(first, [(100.2, 100.5, 99.9, 100.2), (102.5, 103.0, 102.4, 102.8)])
    p, why, _, _ = S.price_fade(r, FADE_T, te + 3 * 86_400_000, k1m=gap)
    assert why == "STOP" and p == pytest.approx(-2.59)                                    # opened through the stop → the open
    flat = [(99.91, 99.95, 99.9, 99.91)] * 16
    late = _bars(first, flat + [(99.91, 99.95, 99.6, 99.6), (99.6, 100.0, 99.6, 99.9)])
    p, why, st, _ = S.price_fade(r, FADE_T, te + 3 * 86_400_000, k1m=late)
    assert (st, why) == ("PRICED", "TRAIL_LATE") and p == pytest.approx(0.31 - 0.35 * 0.31, abs=1e-6)   # the post-X peak 0.31 armed late
    p2, why2, st2, _ = S.price_fade(r, dict(FADE_T, spike_fade_late_arm_after_min=0.0), te + 3 * 86_400_000, k1m=late)
    assert why2 != "TRAIL_LATE" and st2 != "PRICED"                                      # no late arm → nothing binds, path incomplete
    p3, _, st3, _ = S.price_fade(r, FADE_T, te + 60_000, k1m=up.iloc[:1])
    assert st3 == "PENDING" and p3 is None                                               # the path is not complete yet


def test_episodes_shadow_repeat_refusals():
    rows = [dict(pair="A", sleeve="MOMENTUM_LONG", te=SINCE + i * 300_000, status="PRICED", exit_ms=SINCE + i * 300_000 + 20 * 60_000) for i in range(6)]
    rows += [dict(pair="B", sleeve="MOMENTUM_LONG", te=SINCE, status="PENDING", exit_ms=None),
             dict(pair="B", sleeve="MOMENTUM_LONG", te=SINCE + 86_400_000, status="PRICED", exit_ms=SINCE + 86_500_000)]
    out = S.shadow_episodes(pd.DataFrame(rows), t={"max_holding_time_minutes": 1200})
    a = out[out.pair == "A"].sort_values("te").status.tolist()
    assert a == ["PRICED", "SHADOWED", "SHADOWED", "SHADOWED", "PRICED", "SHADOWED"]      # one shadow per pair until its exit (20 min)
    assert out[out.pair == "B"].sort_values("te").status.tolist() == ["PENDING", "SHADOWED"]   # a pending shadow is still open


def test_store_era_reprices_and_ml_tops_up(tmp_path, monkeypatch):
    B = pd.DataFrame([dict(pair="AUSDT", sleeve="FRENZY_LITE", te=SINCE, E=1.0, snap={}, key="k1")])
    calls = []
    monkeypatch.setattr(S, "price_frenzy", lambda r, *a: (calls.append(r["key"]) or (1.0, "TP", "PRICED", SINCE + 60_000)))
    sp = str(tmp_path / "store.csv")
    assert S.price_all(B, SINCE + 86_400_000, fetch=False, store_path=sp).pct.iloc[0] == 1.0 and calls == ["k1"]
    S.price_all(B, SINCE + 86_400_000, fetch=False, store_path=sp)
    assert calls == ["k1"]                                                               # cached: same era → not re-priced
    monkeypatch.setattr(S, "era_of", lambda sl, t: "another exit")
    S.price_all(B, SINCE + 86_400_000, fetch=False, store_path=sp)
    assert calls == ["k1", "k1"]                                                         # the exit changed → re-priced
    open(sp, "w").write("garbage\x00,\n\"")
    assert S.load_store(sp) == {} and not os.path.exists(sp) and any(f.startswith("store.csv.bad-") for f in os.listdir(tmp_path))
    import scout_ml_matched_refusals as MM
    seen = {}
    monkeypatch.setattr(MM, "ensure_paths", lambda fills, now, bud, h: seen.setdefault("n", len(fills)))
    monkeypatch.setattr(MM, "price_replica", lambda fills, now: {f["key"]: dict(pct=0.5, reason="TP", status="PRICED", exit_ms=1) for f in fills})
    r = S.price_ml([dict(key="m", pair="X", te=SINCE, E=1.0, snap={})], SINCE, fetch=True)
    assert seen == {"n": 1} and r["m"] == (0.5, "TP", "PRICED", 1)                       # the blocked ML row is fetched first (review H1)


def _priced(rows):
    df = pd.DataFrame(rows)
    df["day"] = pd.to_datetime(df.te, unit="ms").dt.strftime("%Y-%m-%d")
    return df


def test_gate_collecting_keep_revert_and_guards():
    blk = _priced([dict(sleeve="FRENZY_LITE", lines="LITE_BTC_EMA20_DOWN", te=SINCE + i * 86_400_000 // 2, pct=-1.0, status="PRICED", sole=True)
                   for i in range(16)])
    kept = _priced([dict(sleeve="FRENZY_LITE", te=SINCE + i * 3_600_000, pct=0.8, status="PRICED") for i in range(20)])
    st, det, _ = S.gate_read(blk.iloc[:10], kept, "LITE_BTC_EMA20_DOWN")
    assert st == "COLLECTING"
    st, det, pre = S.gate_read(blk, kept, "LITE_BTC_EMA20_DOWN")
    assert st == "KEEP" and len(pre) == 15 and "avg -1.000" in det                      # frozen at the 15th trade (its 8th day)
    st, det, _ = S.gate_read(blk.assign(pct=0.5), kept, "LITE_BTC_EMA20_DOWN")
    assert st.startswith("REVERT CANDIDATE")                                             # the blocked trades would have earned
    st, det, _ = S.gate_read(blk.assign(pct=-0.2), kept.assign(pct=-0.5), "LITE_BTC_EMA20_DOWN")
    assert st.startswith("REVERT CANDIDATE")                                             # better than what the sleeve kept
    early = pd.concat([blk, _priced([dict(sleeve="FRENZY_LITE", lines="LITE_BTC_EMA20_DOWN", te=SINCE + 1, pct=None, status="PENDING", sole=True)])])
    st, det, _ = S.gate_read(early, kept, "LITE_BTC_EMA20_DOWN")
    assert st == "COLLECTING" and "pending" in det                                       # an earlier refusal unpriced → not decidable
    st, det, _ = S.gate_read(blk, kept.iloc[:4], "LITE_BTC_EMA20_DOWN")
    assert st == "COLLECTING" and "4/5" in det                                           # never judged against < 5 kept fills
    other = pd.concat([blk, _priced([dict(sleeve="FRENZY_LITE", lines="OTHER_LINE", te=SINCE + 1, pct=None, status="PENDING", sole=True)])])
    assert S.gate_read(other, kept, "LITE_BTC_EMA20_DOWN")[0] == "COLLECTING"           # a pending refusal of ANOTHER line holds too (it can shadow)
    shadow = blk.assign(status=["SHADOWED"] * 8 + ["PRICED"] * 8)
    assert S.gate_read(shadow, kept, "LITE_BTC_EMA20_DOWN")[0] == "COLLECTING"           # shadowed repeats never count


def _orders(tmp_path, n):
    rows = [dict(opened_at=(T0 + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S"), closed_at="", pair=f"K{i}USDT", direction="LONG",
                 entry_strategy="FRENZY_LONG", status="CLOSED", pnl_percentage=0.5, close_reason="TP", entry_price=1.0, is_probe=False)
            for i in range(n)]
    rows.append(dict(rows[0], pair="PRBUSDT", is_probe=True))                           # a probe never enters the kept side
    p = tmp_path / "scalpars_orders_paper_2026-12-01_00-00-00.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return [str(p)]


def test_run_freezes_once_and_hook_registry(tmp_path, monkeypatch):
    rows = [(i // 2, f"P{i}USDT", "FRENZY_LONG", "LONG_WEAK_FRENZY", 1.0, True, "LONG") for i in range(16)]
    paths = _dec(tmp_path, rows)
    ords = _orders(tmp_path, 6)
    assert len(S.load_kept(ords, SINCE)) == 6                                            # is_probe out (review)

    def fake_price(B, now_ms, fetch=True, store_path=None):
        return B.assign(pct=-1.0, why="STOP", status="PRICED", exit_ms=B.te + 60_000)
    monkeypatch.setattr(S, "price_all", fake_price)
    st = tmp_path / "g.json"
    out = "\n".join(S.run(now_ms=SINCE + 30 * 86_400_000, dec_paths=paths, ord_paths=ords, state_path=str(st), store_path=None, since_ms=SINCE))
    assert "| FRENZY_LONG · LONG_WEAK_FRENZY | 16 | 8 |" in out and json.load(open(st))["FRENZY_LONG:LONG_WEAK_FRENZY"]["state"] == "KEEP"
    monkeypatch.setattr(S, "price_all", lambda B, now_ms, fetch=True, store_path=None: B.assign(pct=2.0, why="TP", status="PRICED", exit_ms=B.te + 60_000))
    out2 = "\n".join(S.run(now_ms=SINCE + 31 * 86_400_000, dec_paths=paths, ord_paths=ords, state_path=str(st), store_path=None, since_ms=SINCE))
    assert "**KEEP** (frozen" in out2                                                    # never re-decided
    for bad in ("[1, 2]", '{"X:Y": {"state": "KEEP"}}', "not json"):
        st.write_text(bad)
        o3 = "\n".join(S.run(now_ms=SINCE + 31 * 86_400_000, dec_paths=paths, ord_paths=ords, state_path=str(st), store_path=None, since_ms=SINCE))
        assert "gate state unreadable" in o3 and st.read_text() == bad                   # a wrong shape warns, never crashes or overwrites
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(S.run(now_ms=0, dec_paths=[], ord_paths=[], state_path=None, store_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_blocked_filters as _sbf")
    assert "_sbf.run(now_ms)" in src[j:j + 200] and "except Exception as _sbf_e" in src[j:j + 400]
    import observe_registry as OR
    w = OR.EXIT["scout_blocked_filters.py"]["walks"]
    _ml = {k: v for k, v in OR.EXIT["scout_ml_fixed_tp_watch.py"]["walks"].items() if k != "ml_second_chance_enabled"}
    assert _ml.items() <= w.items()                                                                   # the pinned ML replica (M2)
    assert w["ml_second_chance_enabled"] is True                                                      # 🔂 (532) re-based on the second chance
    import scout_frenzy_v2b_shadow as VB
    assert all(w[k] == getattr(VB.V2B_TH, k) for k in ("frenzy_willy_tp_pct", "frenzy_willy_high_k_pct", "frenzy_willy_high_tp_max_pct"))
    assert {r[0]: r[4] for r in OR.status()}["scout_blocked_filters.py"] == "OK"


def _topup_env(tmp_path, monkeypatch):
    import pandas as pd
    import scout_blocked_filters as M
    import scout_willy_timecap as WT
    import scout_b1h_negflank as NF
    monkeypatch.setattr(WT, "MY_CACHE", str(tmp_path))
    monkeypatch.setattr(WT, "_K1", {})
    monkeypatch.setattr(NF, "_cooldown_until", lambda: 0)
    return M, WT, NF, pd


def test_topup_1m_fetches_missing_bars_for_unpriced_frenzy_rows(tmp_path, monkeypatch):
    """🧱 (DL 538) an unpriced blocked FRENZY / fade row gets its missing closed 1m bars up to NOW (not only after the 20 h window);
    an existing cache file is merged (old rows kept)."""
    M, WT, NF, pd = _topup_env(tmp_path, monkeypatch)
    MIN = 60_000
    te = 1_800_000_000_000 - 30 * MIN
    first = (te // MIN) * MIN
    (tmp_path / "1m").mkdir()
    pd.DataFrame([(first - 5 * MIN, 9.0, 9.0, 9.0, 9.0)], columns=["open_time", "o", "h", "l", "c"]).to_csv(tmp_path / "1m" / "ABCUSDT.csv", index=False)
    calls = []

    def fake(sym, start, lim, now):
        calls.append((sym, start, lim))
        return pd.DataFrame([(start + i * MIN, 1.0, 1.0, 1.0, 1.0) for i in range(lim)], columns=["open_time", "o", "h", "l", "c"]), 10
    monkeypatch.setattr(WT, "_fetch_1m", fake)
    fz = pd.DataFrame([dict(te=te, pair="ABCUSDT", sleeve="FRENZY_LITE")])
    notes = M._topup_1m(fz, M.th(), te + 25 * MIN, state_path=str(tmp_path / "a.json"))
    assert calls == [("ABCUSDT", first, 25)]                 # the entry minute + 24 closed bars after it
    k = WT._read_k1(str(tmp_path / "1m" / "ABCUSDT.csv"))
    assert len(k) == 26 and (k.open_time == first - 5 * MIN).any() and "1 public kline call" in " ".join(notes)
    monkeypatch.setattr(NF, "_cooldown_until", lambda: 10**15)
    assert "cooldown" in M._topup_1m(fz, M.th(), te + 25 * MIN)[0]


def test_topup_rate_limit_cap_and_final_misses(tmp_path, monkeypatch):
    import urllib.error
    M, WT, NF, pd = _topup_env(tmp_path, monkeypatch)
    MIN = 60_000
    te = 1_800_000_000_000 - 30 * MIN
    rows = pd.DataFrame([dict(te=te + i * MIN, pair=f"P{i}USDT", sleeve="FRENZY_LITE") for i in range(25)])
    calls = []

    def limited(sym, start, lim, now):
        calls.append(sym)
        raise NF.RateLimited("Binance 429 — stopped")
    monkeypatch.setattr(WT, "_fetch_1m", limited)
    assert "stopped" in " ".join(M._topup_1m(rows, M.th(), te + 60 * MIN, state_path=str(tmp_path / "a.json"))) and len(calls) == 1
    calls.clear()

    def empty(sym, start, lim, now):
        calls.append(sym)
        return pd.DataFrame(columns=["open_time", "o", "h", "l", "c"]), 10
    monkeypatch.setattr(WT, "_fetch_1m", empty)
    sp = str(tmp_path / "b.json")
    M._topup_1m(rows, M.th(), te + 60 * MIN, state_path=sp)
    assert len(calls) == M.TOPUP_MAX_CALLS                   # the per-run cap
    for _ in range(3):
        M._topup_1m(rows.iloc[:1], M.th(), te + 60 * MIN, state_path=sp)
    calls.clear()
    M._topup_1m(rows.iloc[:1], M.th(), te + 60 * MIN, state_path=sp)
    assert calls == []                                       # answered empty 3× → given up, never re-fetched

    def gone(sym, start, lim, now):
        calls.append(sym)
        raise urllib.error.HTTPError("u", 400, "bad symbol", {}, None)
    monkeypatch.setattr(WT, "_fetch_1m", gone)
    sp2 = str(tmp_path / "c.json")
    M._topup_1m(rows.iloc[1:2], M.th(), te + 60 * MIN, state_path=sp2)
    calls.clear()
    M._topup_1m(rows.iloc[1:2], M.th(), te + 60 * MIN, state_path=sp2)
    assert calls == []                                       # a 4xx symbol is final


def test_1m_price_is_provisional_until_ticks_cover(monkeypatch):
    """🧱 (DL 538 review) a FRENZY row priced on 1m bars carries the era suffix and is re-priced once the tick archive covers its days."""
    import scout_blocked_filters as M
    import scout_willy_timecap as WT
    te = int(pd.Timestamp("2026-11-11 20:55").value // 10**6)
    ex = int(pd.Timestamp("2026-11-12 00:30").value // 10**6)
    have = {"2026-11-11"}
    monkeypatch.setattr(WT, "_tick_file", lambda pair, d: ("x" if d in have else None))
    assert M._ticks_cover("IDOLUSDT", te, ex) is False                 # the exit day has no ticks yet
    have.add("2026-11-12")
    assert M._ticks_cover("IDOLUSDT", te, ex) is True
    assert M.ERA_1M == "|1m"
    src = open(M.__file__).read()
    assert 'eras.get(sl, "") + (ERA_1M if _SRC.get(k) == "1m" else "")' in src and "return not _ticks_cover(pair, te, store[k][2])" in src
