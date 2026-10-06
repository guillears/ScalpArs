"""🌊 Scout market-volume reading (scripts/scout_gvol.py, 2026-10-06 fix of FRENZY_GVOL_GATE_REVALIDATION §5) — per-bar universe, the freeze,
the live-value preference, and engine parity on the 18 live readings of 10-03 → 10-06 (compact fixture from the report's rebuild). No network."""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_gvol as SG  # noqa: E402
import scout_frenzy as FZ  # noqa: E402

BAR = SG.BAR
CFG = dict(new_listing_filter_days=90, alpha_subtype_filter_enabled=True, coin_underlying_only=True)
FIX = os.path.join(ROOT, "tests", "fixtures", "scout_gvol_parity_1003_1006.csv")
T0 = int(pd.Timestamp("2026-10-07", tz="UTC").value // 1_000_000)   # after the gate deploy (gate labels are judged)
OLD = T0 - 400 * SG.DAY                     # onboard long ago


def _series(n_bars, v=100.0, q=1000.0, start=T0, bump=None):
    """n 5m rows [t, o, h, l, c, v, q] from start; bump {bar index: (v, q)} overrides."""
    out = []
    for k in range(n_bars):
        vv, qq = (bump or {}).get(k, (v, q))
        out.append([start + k * BAR, 1, 1, 1, 1, vv, qq])
    return out


def _market(n_pairs=40, n_bars=700):
    rows = {f"P{i:02d}USDT": _series(n_bars, v=100.0 + i, q=1e6 * (n_pairs - i)) for i in range(n_pairs)}
    meta = {p: dict(onboard=OLD, alpha=False, coin=True) for p in rows}
    return rows, meta


# ─────────────────────────── per-bar universe ───────────────────────────
def test_a_coin_entering_later_does_not_change_an_earlier_bar():
    rows, meta = _market(n_pairs=60)
    early, late = T0 + 400 * BAR, T0 + 650 * BAR
    base = SG.gvol_at(SG.Bars(rows), meta, early, CFG)
    # a low-priced coin with HUGE base volume becomes the #1 by quote volume only after bar 500
    rows2 = dict(rows); meta2 = dict(meta, VTHOUSDT=dict(onboard=OLD, alpha=False, coin=True))
    rows2["VTHOUSDT"] = _series(700, v=3e8, q=1.0, bump={k: (3e8 * (2 if k == 650 else 1), 1e12) for k in range(500, 700)})
    B2 = SG.Bars(rows2)
    assert SG.gvol_at(B2, meta2, early, CFG)[0] == base[0]                       # earlier bar untouched
    assert "VTHOUSDT" not in SG.gvol_at(B2, meta2, early, CFG)[2]
    v_late, _, top_late = SG.gvol_at(B2, meta2, late, CFG)
    assert top_late[0] == "VTHOUSDT" and len(top_late) == SG.TOP_N             # it IS in the later bar's top-50 …
    assert v_late != SG.gvol_at(SG.Bars(rows), meta, late, CFG)[0]              # … and moves that bar only


def test_rank_is_the_288_bars_ending_at_the_signal_bar():
    rows, meta = _market(n_pairs=55)
    sig = T0 + 600 * BAR
    # P54 (rank 55) gets a quote spike exactly 288 bars before sig+1 → outside the window; at the window's first bar → inside
    rows["P54USDT"] = _series(700, v=154.0, q=1e6, bump={600 - 288: (154.0, 1e12)})
    assert "P54USDT" not in SG.top_pairs(SG.Bars(rows), list(meta), sig)
    rows["P54USDT"] = _series(700, v=154.0, q=1e6, bump={600 - 287: (154.0, 1e12)})
    assert SG.top_pairs(SG.Bars(rows), list(meta), sig)[0] == "P54USDT"


def test_pair_without_the_signal_bar_is_not_ranked_and_min_pairs():
    rows, meta = _market(n_pairs=31)
    sig = T0 + 600 * BAR
    rows["P00USDT"] = [r for r in rows["P00USDT"] if r[0] != sig]
    B = SG.Bars(rows)
    assert "P00USDT" not in SG.top_pairs(B, list(meta), sig)
    assert SG.gvol_at(B, meta, sig, CFG)[0] == 1.0                               # 30 pairs, flat volume
    rows["P01USDT"] = rows["P01USDT"][:100]
    assert SG.gvol_at(SG.Bars(rows), meta, sig, CFG)[0] is None                 # 29 < min_pairs


def test_eligibility_is_as_of_the_signal_close():
    sig = T0 + 600 * BAR
    meta = dict(A=dict(onboard=sig + BAR - 90 * SG.DAY, alpha=False, coin=True),          # exactly 90 d at the close → excluded (engine: ≥ cutoff)
                B=dict(onboard=sig + BAR - 90 * SG.DAY - 1, alpha=False, coin=True),      # older by 1 ms → kept
                C=dict(onboard=None, alpha=False, coin=True),                             # missing → kept (fail-open)
                D=dict(onboard=OLD, alpha=True, coin=True), E=dict(onboard=OLD, alpha=False, coin=False))
    assert sorted(SG.eligible(meta, sig + BAR, CFG)) == ["B", "C"]
    assert sorted(SG.eligible(meta, sig + BAR + SG.DAY, CFG)) == ["A", "B", "C"]          # a day later A qualifies
    assert sorted(SG.eligible(meta, sig + BAR, dict(CFG, alpha_subtype_filter_enabled=False, coin_underlying_only=False))) == ["B", "C", "D", "E"]


def test_prescreen_keeps_a_coin_that_collapsed_since():
    """the PUMPBTC / 1000000BOB class: top-5 at the bar, ~0 volume by the run — the current ranking would lose it, the hourly prescreen keeps it."""
    H = SG.HOUR
    hourly = {f"P{i:03d}": [[T0 + h * H, 1e6 * (200 - i)] for h in range(60)] for i in range(200)}
    hourly["PUMP"] = [[T0 + h * H, (5e8 if h < 30 else 0.0)] for h in range(60)]
    meta = {p: dict(onboard=OLD, alpha=False, coin=True) for p in hourly}
    sig = T0 + 30 * H - BAR                               # closes at hour 30: the 24 h before are PUMP's big hours
    keep = SG.prescreen(hourly, meta, [sig], CFG, k=80)
    assert "PUMP" in keep and len(keep) == 80
    assert "PUMP" not in SG.prescreen(hourly, meta, [T0 + 58 * H - BAR], CFG, k=80)   # a day+ later it has really left


# ─────────────────────────── freeze ───────────────────────────
def test_freeze_first_value_wins_and_old_version_is_replaced(tmp_path):
    p = str(tmp_path / "bars.csv")
    pd.DataFrame([dict(bar_ts=T0, gvol=0.833, n_pairs=50, ver=1, computed_ms=0, top3="")]).to_csv(p, index=False)
    assert SG.cached(path=p) == {}                                   # a v1 row is not a v2 value
    SG.freeze({T0: (1.1149, 50, "x"), T0 + BAR: (None, 0, "")}, 1, path=p)
    assert SG.cached(path=p) == {T0: 1.1149}                         # v1 replaced once; None not frozen
    SG.freeze({T0: (0.5, 50, "y"), T0 + BAR: (0.9, 50, "z")}, 2, path=p)
    assert SG.cached(path=p) == {T0: 1.1149, T0 + BAR: 0.9}          # never overwritten; the unread bar fills later
    assert len(pd.read_csv(p)) == 2


def test_ensure_computes_each_bar_once(tmp_path, monkeypatch):
    p = str(tmp_path / "bars.csv")
    calls = []

    monkeypatch.setattr(SG, "universe_meta", lambda EX, retry: {"X": {}})

    def fake_compute(meta, cfg, sigs, pre=None, deadline=None):
        calls.append(list(sigs))
        return {s: (0.7 + len(calls), 50, "") for s in sigs}
    monkeypatch.setattr(SG, "compute", fake_compute)
    now = T0 + 100 * BAR
    a = SG.ensure(None, None, CFG, [T0, T0 + BAR, now], now, path=p)          # `now` is not closed yet → not asked
    assert a == {T0: 1.7, T0 + BAR: 1.7} and calls == [[T0, T0 + BAR]]
    b = SG.ensure(None, None, CFG, [T0, T0 + BAR, T0 + 2 * BAR], now, path=p)
    assert b == {T0: 1.7, T0 + BAR: 1.7, T0 + 2 * BAR: 2.7} and calls[-1] == [T0 + 2 * BAR]
    assert SG.ensure(None, None, CFG, [now - SG.MAX_AGE_MS - BAR], now, path=p) == {} and len(calls) == 2   # too old: not recomputed


def test_ensure_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(SG, "universe_meta", lambda EX, retry: {"X": {}})
    monkeypatch.setattr(SG, "compute", lambda *a, **k: 1 / 0)
    assert SG.ensure(None, None, CFG, [T0], T0 + 10 * BAR, path=str(tmp_path / "b.csv")) == {}


def test_scout_row_keeps_its_frozen_value_and_old_rows_are_labelled():
    live = {}
    r = dict(bar_ts=T0, gvol_scout=1.1149, gvol_ver=2, gvol=1.1149)
    FZ._apply_gvol(r, {T0: 0.833}, live, 1.0, prev=dict(r))                    # a different later computation must not replace it
    assert r["gvol_scout"] == 1.1149 and r["gvol_src"] == SG.SRC_SCOUT
    old = dict(bar_ts=T0 + BAR, gvol=0.831)                                      # stored by the old method, bar not (yet) recomputed
    FZ._apply_gvol(old, {}, live, 1.0, prev=dict(old))
    assert old["gvol_src"] == SG.SRC_OLD and old["gvol_ver"] == 1 and old["gvol_gate"].endswith("(old method)")
    FZ._apply_gvol(old, {T0 + BAR: 1.1149}, live, 1.0, prev=dict(old))          # recomputed once the cache has it
    assert old["gvol_src"] == SG.SRC_SCOUT and old["gvol"] == 1.1149 and old["gvol_ver"] == SG.VER


# ─────────────────────────── live-value preference ───────────────────────────
def test_live_value_wins_and_the_source_is_shown():
    close = T0 + BAR
    live = SG.live_map(pd.DataFrame([dict(pair="API3USDT", close_ms=close, value=1.11, sleeve="FRENZY", source="log"),
                                     dict(pair="ORCAUSDT", close_ms=close, value=1.1149, sleeve="FRENZY", source="stamp")]))
    assert live[close][:2] == (1.1149, SG.SRC_STAMP)                            # a stamp (4 dp) before a log line (2 dp)
    r = dict(bar_ts=T0, gvol=None)
    FZ._apply_gvol(r, {T0: 0.95}, live, 1.0)
    assert (r["gvol"], r["gvol_src"], r["gvol_scout"], r["gvol_gate"]) == (1.1149, SG.SRC_STAMP, 0.95, "BLOCK")
    assert SG.resolve(0.95, None) == (0.95, SG.SRC_SCOUT) and SG.resolve(None, None) == (None, "unread")


def test_server_log_gate_line_parsing():
    ok = ("Oct  6 06:35:09 ip web[1]: 2026-10-06 06:35:09,034 - services.trading_engine - INFO - [FRENZY_LONG] API3USDT: setup ON but market "
          "volume 1.11× normal ≥ 1× — skipped")
    wide = ok.replace("[FRENZY_LONG] API3USDT", "[FRENZY_WIDE] CAPUSDT").replace("1.11×", "1.30×")
    late = ok.replace("06:35:09,034", "06:37:09,034")                            # 2 min after the close: maybe a catch-up → not used
    unread = ok.replace("market volume 1.11× normal ≥ 1×", "market volume unreadable")
    got = SG.parse_log_lines([ok, wide, late, unread, "noise"])
    c = int(pd.Timestamp("2026-10-06 06:35", tz="UTC").value // 1_000_000)
    assert got == [("API3USDT", c, 1.11, "FRENZY"), ("CAPUSDT", c, 1.30, "WIDE")]


def test_fill_stamps_map_to_the_signal_close():
    o = pd.DataFrame(dict(opened_at=["2026-10-06T09:40:08", "2026-10-06T10:20:30", "2026-10-06T11:00:05", "2026-10-06T12:00:05"],
                          pair=["ORCAUSDT", "XUSDT", "YUSDT", "ZUSDT"], entry_strategy=["FRENZY_LONG", "FRENZY_WIDE", "MOMENTUM", "FRENZY_LONG"],
                          entry_frenzy_gvol=[0.9539, 0.7, 0.5, None]))
    o.loc[1, "opened_at"] = "2026-10-06T10:23:30"   # 3.5 min past its 5m floor (10:20) > LIVE_FILL_MAX_MS: not attributed to any bar
    got = SG.fills_live(o)
    assert got == [("ORCAUSDT", int(pd.Timestamp("2026-10-06 09:40", tz="UTC").value // 1_000_000), 0.9539, "FRENZY")]


def test_update_live_registry_merges_and_survives(tmp_path):
    p = str(tmp_path / "live.csv"); lg = tmp_path / "web.log"
    lg.write_text("2026-10-06 06:35:09,034 - services.trading_engine - INFO - [FRENZY_LONG] API3USDT: setup ON but market volume 1.11× normal ≥ 1×\n")
    o = pd.DataFrame(dict(opened_at=["2026-10-06T09:40:08"], pair=["ORCAUSDT"], entry_strategy=["FRENZY_LONG"], entry_frenzy_gvol=[0.9539]))
    r = SG.update_live(1, orders=o, log_paths=[str(lg)], path=p)
    assert len(r) == 2 and os.path.exists(p)
    r2 = SG.update_live(2, orders=pd.DataFrame(), log_paths=[], path=p)        # the export / log left ~/Downloads: the registry keeps them
    assert len(r2) == 2


# ─────────────────────────── engine parity (18 live readings) ───────────────────────────
def _fixture_event(F, close):
    """the fixture's per-pair (q24, signal-bar volume, Σ 48 bars) → synthetic rows that reproduce the same rank and the same 48-bar mean."""
    sig = int(pd.Timestamp(close, tz="UTC").value // 1_000_000) - BAR
    rows, meta = {}, {}
    for _, r in F[F.signal_close == close].iterrows():
        rest = (r.sum48 - r.v_sig) / 47.0
        rows[r.pair] = [[sig - (47 - k) * BAR, 1, 1, 1, 1, rest, 0.0] for k in range(47)] + [[sig, 1, 1, 1, 1, r.v_sig, r.q24]]
        meta[r.pair] = dict(onboard=(None if pd.isna(r.onboard) else int(r.onboard)), alpha=bool(r.alpha), coin=bool(r.coin))
    return SG.Bars(rows), meta, sig


def test_parity_with_the_18_live_readings():
    F = pd.read_csv(FIX)
    got = []
    for close, g in F.groupby("signal_close", sort=False):
        B, meta, sig = _fixture_event(F, close)
        v, n, _ = SG.gvol_at(B, meta, sig, CFG)
        got.append((close, g.pair.iloc[0], v, g.live.iloc[0]))
    G = pd.DataFrame(got, columns=["close", "pair", "scout", "live"])
    lv = G[G.live.notna()]
    pr = SG.parity(list(zip(lv.scout, lv.live)))
    assert pr["n"] == 18 and pr["same_side"] == 18 and pr["max"] <= 0.005 and pr["mae"] <= 0.002
    at = dict(zip(G.close, G.scout))
    assert at["2026-10-06 06:35"] == pytest.approx(1.115, abs=0.002)          # API3: was 0.833 "pass" on the run-time list
    assert at["2026-10-06 13:25"] == pytest.approx(1.286, abs=0.002)          # CAP: was 1.014
    assert at["2026-10-06 13:40"] == pytest.approx(2.106, abs=0.002)          # NMR: was 1.699


def test_parity_summary_and_lines():
    pr = SG.parity([(1.115, 1.11), (0.979, 0.979), (0.98, 1.01), (None, 1.0)])
    assert pr["n"] == 3 and pr["same_side"] == 2 and pr["flips"] == [2]
    hist = pd.DataFrame(dict(bar_ts=[T0, T0, T0 + BAR], pair=["A", "B", "C"], signal_close_utc=["2026-10-06 06:35"] * 3,
                             gvol_scout=[1.115, 1.115, 0.98], gvol_live=[1.11, 1.11, 1.01], gvol_src=[SG.SRC_STAMP] * 3))
    L = FZ.gvol_parity_lines(hist, 1.0)
    assert "2 bars with both" in L[0] and "same side of the 1× gate on 1/2" in L[0] and "opposite side" in L[0]


def test_groups_newest_first_and_all_or_nothing(tmp_path, monkeypatch):
    p = str(tmp_path / "bars.csv")
    monkeypatch.setattr(SG, "universe_meta", lambda EX, retry: {"X": {}})
    calls = []
    monkeypatch.setattr(SG, "compute", lambda meta, cfg, sigs, pre=None, deadline=None: calls.append(list(sigs)) or {s: (0.9, 50, "") for s in sigs})
    now = T0 + 1000 * BAR
    sigs = [T0, T0 + 100 * BAR, T0 + 500 * BAR, T0 + 900 * BAR]
    SG.ensure(None, None, CFG, sigs, now, path=p)
    assert calls[0] == [T0 + 900 * BAR] and calls[1] == [T0 + 500 * BAR] and calls[2] == [T0, T0 + 100 * BAR]   # ≤ 288-bar groups, newest first

    def boom(jobs, deadline):
        raise SG.Abort("Binance 429")
    monkeypatch.undo()                                                           # the real compute, a failing fetch
    monkeypatch.setattr(SG, "_fetch_all", boom)
    meta = {"A": dict(id="A", onboard=OLD, alpha=False, coin=True)}
    assert SG.compute(meta, CFG, [T0]) == {}                                     # a failed fetch freezes nothing


def test_group_span_is_measured_from_the_newest_bar(tmp_path, monkeypatch):
    monkeypatch.setattr(SG, "universe_meta", lambda EX, retry: {"X": {}})
    calls = []
    monkeypatch.setattr(SG, "compute", lambda meta, cfg, sigs, pre=None, deadline=None: calls.append(list(sigs)) or {})
    sigs = [T0 + k * 10 * BAR for k in range(100)]                               # dense: 1,000 bars of span, 10 apart
    SG.ensure(None, None, CFG, sigs, T0 + 2000 * BAR, path=str(tmp_path / "b.csv"))
    assert all(g[-1] - g[0] < SG.GROUP_BARS * BAR for g in calls) and sum(map(len, calls)) == 100 and len(calls) == 4


def test_save_end_to_end_old_row_then_v2_then_live(tmp_path, monkeypatch):
    monkeypatch.setattr(FZ, "CSV", str(tmp_path / "fz.csv"))
    monkeypatch.setattr(SG, "CACHE_CSV", str(tmp_path / "bars.csv"))
    monkeypatch.setattr(SG, "LIVE_CSV", str(tmp_path / "live.csv"))
    cfg = dict(frenzy_gvol_max=1.0)
    pd.DataFrame([dict(pair="API3USDT", bar_ts=T0, gvol=0.831, gvol_gate="pass", pnl=1.0, exit="trail", superseded=False)]).to_csv(FZ.CSV, index=False)
    assert FZ._old_method_bars([FZ.CSV]) == {T0}
    row = dict(pair="API3USDT", bar_ts=T0, gvol=None, pnl=1.0, exit="trail")
    FZ._apply_gvol(row, {}, {}, 1.0)                                             # this run could not read it
    a = FZ.save([row], ["API3USDT"], T0 + 50 * BAR, cfg=cfg)
    assert a.iloc[0].gvol_src == SG.SRC_OLD and a.iloc[0].gvol == 0.831 and a.iloc[0].gvol_gate == "pass (old method)"
    SG.freeze({T0: (1.1149, 50, "")}, 1)                                         # a later run computes it
    a = FZ.save([], ["API3USDT"], T0 + 60 * BAR, cfg=cfg)
    assert (a.iloc[0].gvol_src, a.iloc[0].gvol, a.iloc[0].gvol_gate) == (SG.SRC_SCOUT, 1.1149, "BLOCK")
    assert FZ._old_method_bars([FZ.CSV]) == set()
    pd.DataFrame([dict(pair="API3USDT", close_ms=T0 + BAR, value=1.11, sleeve="FRENZY", source="log", seen_ms=1)]).to_csv(SG.LIVE_CSV, index=False)
    a = FZ.save([], ["API3USDT"], T0 + 70 * BAR, cfg=cfg)
    assert (a.iloc[0].gvol_src, a.iloc[0].gvol, a.iloc[0].gvol_scout, a.iloc[0].gvol_live) == (SG.SRC_LOG, 1.11, 1.1149, 1.11)
    L = FZ.gvol_parity_lines(a.assign(signal_close_utc="2026-10-07 00:05"), 1.0)
    assert "1 bars with both" in L[0] and "same side of the 1× gate on 1/1" in L[0]



# ─────────────────────────── review fixes (2026-10-06 dual review) ───────────────────────────
def test_catchup_fill_is_not_attributed():
    o = pd.DataFrame(dict(opened_at=["2026-10-06T09:40:08", "2026-10-06T10:10:08"], pair=["ORCAUSDT", "UMAUSDT"],
                          entry_strategy=["FRENZY_LONG", "FRENZY_LONG"], entry_frenzy_gvol=[0.9539, 0.8155], entry_frenzy_catchup=[True, None]))
    assert [x[0] for x in SG.fills_live(o)] == ["UMAUSDT"]
    o["entry_frenzy_catchup"] = ["False", "1"]
    assert [x[0] for x in SG.fills_live(o)] == ["ORCAUSDT"]


def test_catchup_log_line_shadows_the_pairs_gate_lines():
    pre = "2026-10-06 {t},034 - services.trading_engine - {lv} - "
    cu = pre.format(t="06:35:05", lv="WARNING") + "[FRENZY_CATCHUP] API3USDT: ON bar 06:15 was not judged (bot paused/restarting) → opening"
    g = pre.format(t="06:35:09", lv="INFO") + "[FRENZY_LONG] API3USDT: setup ON but market volume 1.11× normal ≥ 1× — skipped"
    other = pre.format(t="06:35:10", lv="INFO") + "[FRENZY_LONG] NMRUSDT: setup ON but market volume 1.11× normal ≥ 1× — skipped"
    far = pre.format(t="08:00:09", lv="INFO") + "[FRENZY_LONG] API3USDT: setup ON but market volume 0.90× normal ≥ 1× — skipped"
    got = SG.parse_log_lines([g, cu, other, far])                                  # order-independent: the catch-up line may come after
    assert [(p, v) for p, _, v, _ in got] == [("NMRUSDT", 1.11), ("API3USDT", 0.90)]


def test_live_map_is_deterministic():
    rows = [dict(pair=p, close_ms=T0, value=v, sleeve="FRENZY", source="log") for p, v in (("ZZZUSDT", 1.2), ("AAAUSDT", 1.1))]
    a = SG.live_map(pd.DataFrame(rows)); b = SG.live_map(pd.DataFrame(rows[::-1]))
    assert a == b and a[T0][2] == "AAAUSDT"


class _HTTPErr(Exception):
    pass


def _http_error(code, body=b"", headers=None):
    import io
    import urllib.error
    return urllib.error.HTTPError("u", code, "x", headers or {}, io.BytesIO(body))


def test_http_400_invalid_symbol_is_empty_other_400_aborts(monkeypatch):
    import urllib.request
    monkeypatch.setattr(SG, "_BAN_UNTIL", [0.0]); monkeypatch.setattr(SG, "_STOP", []); monkeypatch.setattr(SG, "_PAUSE", [0.0])
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http_error(400, b'{"code":-1121,"msg":"Invalid symbol."}')))
    assert SG._get("GONEUSDT", "5m", T0, 10) == []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http_error(400, b'{"code":-1100,"msg":"Illegal"}')))
    with pytest.raises(SG.Abort):
        SG._get("XUSDT", "5m", T0, 10)


def test_429_stops_the_run_and_backs_off(monkeypatch, tmp_path):
    import urllib.request
    monkeypatch.setattr(SG, "_BAN_UNTIL", [0.0]); monkeypatch.setattr(SG, "_STOP", []); monkeypatch.setattr(SG, "_PAUSE", [0.0])
    monkeypatch.setattr(SG, "_DEADLINES", {})
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise _http_error(429, headers={"Retry-After": "120"})
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(SG.Banned):
        SG._get("XUSDT", "1h", T0, 10)
    assert SG._BAN_UNTIL[0] >= __import__("time").time() + 100                  # Retry-After honoured (≥ 60 s floor)
    with pytest.raises(SG.Banned):
        SG._get("YUSDT", "1h", T0, 10)                                            # no new call during the back-off
    assert len(calls) == 1
    # ensure(): compute raising Banned stops every group; a second ensure in the same run makes no read
    monkeypatch.setattr(SG, "_BAN_UNTIL", [0.0])
    monkeypatch.setattr(SG, "universe_meta", lambda EX, retry: {"X": {}})
    seen = []

    def banned(meta, cfg, sigs, pre=None, deadline=None):
        seen.append(sigs); SG._BAN_UNTIL[0] = __import__("time").time() + 60
        raise SG.Banned("Binance 429")
    monkeypatch.setattr(SG, "compute", banned)
    now = T0 + 2000 * BAR
    p = str(tmp_path / "b.csv")
    assert SG.ensure(None, None, CFG, [T0 + 100 * BAR, T0 + 900 * BAR], now, path=p) == {} and len(seen) == 1
    assert SG.ensure(None, None, CFG, [T0 + 1500 * BAR], now, path=p) == {} and len(seen) == 1


def test_one_time_budget_per_run_shared_by_both_callers(monkeypatch, tmp_path):
    monkeypatch.setattr(SG, "_DEADLINES", {}); monkeypatch.setattr(SG, "_BAN_UNTIL", [0.0])
    monkeypatch.setattr(SG, "universe_meta", lambda EX, retry: {"X": {}})
    calls = []
    monkeypatch.setattr(SG, "compute", lambda meta, cfg, sigs, pre=None, deadline=None: calls.append(deadline) or {s: (0.9, 50, "") for s in sigs})
    now = T0 + 2000 * BAR
    SG.ensure(None, None, CFG, [T0 + 10 * BAR], now, path=str(tmp_path / "a.csv"))
    SG.ensure(None, None, CFG, [T0 + 20 * BAR], now, path=str(tmp_path / "b.csv"))   # the SURGE call of the same run
    assert len(calls) == 2 and calls[0] == calls[1]
    SG._DEADLINES[int(now)] = 0.0                                                   # budget spent → no read
    SG.ensure(None, None, CFG, [T0 + 30 * BAR], now, path=str(tmp_path / "c.csv"))
    assert len(calls) == 2


def test_throttle_honours_deadline_and_stop(monkeypatch):
    monkeypatch.setattr(SG, "_PAUSE", [__import__("time").monotonic() + 3600]); monkeypatch.setattr(SG, "_STOP", [])
    monkeypatch.setattr(SG, "_BAN_UNTIL", [0.0])
    with pytest.raises(SG.Abort):
        SG._throttle(1, deadline=__import__("time").monotonic() - 1)
    monkeypatch.setattr(SG, "_PAUSE", [0.0]); monkeypatch.setattr(SG, "_STOP", [1])
    with pytest.raises(SG.Abort):
        SG._throttle(1)


def test_freeze_needs_45_pairs_and_tries_cap(tmp_path, monkeypatch):
    p = str(tmp_path / "bars.csv")
    SG.freeze({T0: (0.95, 44, "")}, 1, path=p)                                    # short read: not frozen, one try
    assert SG.cached(path=p) == {} and SG.tried_out(path=p) == set()
    SG.freeze({T0: (None, 0, "")}, 2, path=p); SG.freeze({T0: (None, 0, "")}, 3, path=p)
    assert SG.tried_out(path=p) == {T0}                                            # 3 complete-but-unreadable reads → given up
    monkeypatch.setattr(SG, "universe_meta", lambda EX, retry: {"X": {}}); monkeypatch.setattr(SG, "_DEADLINES", {})
    monkeypatch.setattr(SG, "_BAN_UNTIL", [0.0])
    calls = []
    monkeypatch.setattr(SG, "compute", lambda meta, cfg, sigs, pre=None, deadline=None: calls.append(sigs) or {})
    SG.ensure(None, None, CFG, [T0], T0 + 100 * BAR, path=p)
    assert calls == []
    SG.freeze({T0 + BAR: (0.95, 45, "")}, 4, path=p)
    assert SG.cached(path=p) == {T0 + BAR: 0.95}


def test_compute_does_not_freeze_when_a_candidate_came_back_empty(monkeypatch):
    meta = {f"P{i}": dict(id=f"P{i}", onboard=OLD, alpha=False, coin=True) for i in range(2)}
    monkeypatch.setattr(SG, "_fetch_all", lambda jobs, deadline: {k: ([] if k == "P1" else [[T0, 0, 0, 0, 0, 1, 1]]) for k in jobs})
    monkeypatch.setattr(SG, "prescreen", lambda *a, **k: {"P0", "P1"})
    assert SG.compute(meta, CFG, [T0]) == {}


def test_corrupt_cache_is_moved_aside_loudly(tmp_path, capsys):
    p = tmp_path / "bars.csv"; p.write_text("garbage\n\x00,,\n")
    SG.load_cache(str(p))
    assert not p.exists() and list(tmp_path.glob("bars.csv.*.bad")) and "WARNING" in capsys.readouterr().err


def test_save_crashes_keeps_the_old_method_value(tmp_path, monkeypatch):
    monkeypatch.setattr(FZ, "CRASH_CSV", str(tmp_path / "cr.csv"))
    monkeypatch.setattr(SG, "CACHE_CSV", str(tmp_path / "bars.csv")); monkeypatch.setattr(SG, "LIVE_CSV", str(tmp_path / "live.csv"))
    pd.DataFrame([dict(pair="AINUSDT", bar_ts=T0, gvol=0.77, pnl=1.0, exit="trail", held_min=5, gone=False)]).to_csv(FZ.CRASH_CSV, index=False)
    c = dict(pair="AINUSDT", bar_ts=T0, gvol=None, pnl=None, exit="no 1m data", held_min=0)
    FZ._apply_gvol(c, {}, {}, None)
    a = FZ.save_crashes([c], ["AINUSDT"], T0 + 10 * BAR)
    assert a.iloc[0].gvol == 0.77 and a.iloc[0].gvol_src == SG.SRC_OLD


def test_parity_judged_at_the_gate_threshold():
    assert SG.parity([(1.2, 1.4)], 1.3)["flips"] == [0] and SG.parity([(1.2, 1.4)], 1.0)["flips"] == []


def test_surge_unread_gvol_is_not_a_failed_leg():
    import scout_surge_obs as SO
    assert SO.gmap_unread([T0, T0 + BAR], {T0: 1.2}) == {T0 + BAR}
    assert SO.gmap_unread([T0], {T0: 0.4}) == set()


def test_surge_scan_with_unread_gvol_fires_nothing_and_keeps_stored_triggers(tmp_path, monkeypatch):
    import scout_surge_obs as SO
    monkeypatch.setattr(SO, "CSV", str(tmp_path / "surge.csv"))
    n = 700
    t = [T0 + k * BAR for k in range(n)]
    c = [100.0] * (n - 1) + [101.0]                                            # +1 % on the last bar, a new 24 h high
    v = [10.0] * (n - 1) + [100.0]                                             # 10× volume
    btc = pd.DataFrame(dict(o=c, h=c, l=c, c=c, v=v), index=t)
    last = t[-1]; now = last + BAR + 1000
    stored = pd.DataFrame([dict(trig=last - 10 * BAR, pair="SOLUSDT", pnl=1.0, exit="TRAIL", held_min=30, close_utc="x")])
    stored.to_csv(SO.CSV, index=False)
    import scout_gvol as G
    monkeypatch.setattr(G, "ensure", lambda *a, **k: {})                        # the read failed this run
    bars, obs = SO.scan(None, None, {}, last, {}, btc, [], now)
    row = [b for b in bars if b["trig"] == last][0]
    assert row["fired"] is False and row["ok_gvol"] is None and "unread" in row["note"] and obs == []
    assert SO.LAST_WINDOW is None
    a = SO.save(obs)
    assert len(a) == 1 and int(a.iloc[0].trig) == last - 10 * BAR               # the stored in-window trigger is NOT deleted
    monkeypatch.setattr(G, "ensure", lambda *a, **k: {last: 0.5})               # read, below 1.0: a failed leg (no fire), window authoritative
    bars, obs = SO.scan(None, None, {}, last, {}, btc, [], now)
    row = [b for b in bars if b["trig"] == last][0]
    assert row["ok_gvol"] is False and row["fired"] is False and SO.LAST_WINDOW is not None
