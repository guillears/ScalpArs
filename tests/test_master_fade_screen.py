"""10-08c — the master builder's SPIKE_FADE screen must refuse what today's live rules refuse (SPIKE_FADE recall trace §E).

① FADE_FRESHBREAK on pre-ship fades (opened before the Aug-10 gate ship, no rsi_prev1 stamp) is judged on the engine's own
rsi_prev1 rebuilt from 5m klines (99 closed bars + forming close, bar anchored by the fill's rsi_prev2 stamp) — the old
rsi_prev2 stamp proxy kept 5 named fades the live gate refuses. ② Today's pair blacklists apply to every era's rows
(龙虾USDT B3 fade was kept). Fixtures: tests/fixtures/master_fade_fb/ (cached klines + stamps; no network).
Falsifiable: reverting the builder to the stamp proxy, dropping the bar anchor, or skipping the blacklist fails here.
"""
import importlib.util
import json
import pathlib

import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_master_pool", _ROOT / "scripts" / "build_master_pool.py")
bmp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bmp)
FX = _ROOT / "tests" / "fixtures" / "master_fade_fb"
FILLS = pd.read_csv(FX / "fills.csv")
NAMED_FB = [("ZENUSDT", "2026-08-02T11:59:50"), ("KSMUSDT", "2026-08-04T08:10:39"), ("FHEUSDT", "2026-08-05T00:13:07"),
            ("XANUSDT", "2026-08-08T16:34:55"), ("GRIFFAINUSDT", "2026-08-09T02:48:09")]
NAMED_BL = ("龙虾USDT", "2026-08-17T00:25:37")


def _row(pair, t):
    r = FILLS[(FILLS.pair == pair) & (FILLS.opened_at.str[:19] == t)]
    assert len(r) == 1, (pair, t)
    return r.iloc[0]


def _rec(r):
    k5 = bmp.fade_fb_load_klines(r.pair, dirs=(str(FX),))
    return bmp.fade_fb_recompute(k5, r.opened_at, r.entry_price, r.entry_rsi_prev, r.entry_pair_rank)


def test_five_named_pre_ship_fades_refused_by_freshbreak_recompute():
    for pair, t in NAMED_FB:
        r = _row(pair, t)
        assert bool(r.keep_pre1008c), "fixture = the pre-10-08c keep (the bug)"
        assert not bmp.fade_freshbreak_stamp_block(r.opened_at, r.entry_rsi_prev, r[bmp.G]), "the legacy proxy passed it"
        rec = _rec(r)
        assert rec is not None and rec["shift"] == 0 and rec["path"] == "scanner", (pair, rec)
        assert rec["rsi_prev1"] < bmp.FADE_FB_RSI_PREV_MIN and r[bmp.G] > bmp.FADE_FB_PGAP_MIN
        assert bmp.fade_freshbreak_block(r.opened_at, rec, r.entry_rsi_prev, r[bmp.G]), (pair, t)


def test_recompute_matches_the_live_truth_rsi_prev1():
    """rsi_prev1 values the recall trace rebuilt independently from real trades (study_fade_trace_signals_raw live_truth_prev)."""
    truth = {"ZENUSDT": 42.180476, "GRIFFAINUSDT": 41.785449}
    for pair, t in NAMED_FB:
        if pair in truth:
            assert abs(_rec(_row(pair, t))["rsi_prev1"] - truth[pair]) < 1e-4


def test_bar_anchor_uses_the_stamp():
    """A fill opened seconds after a 5m boundary was decided in the previous bar — the rsi_prev2 stamp picks it, both ways."""
    tag = _rec(_row("TAGUSDT", "2026-08-03T07:55:02"))     # prev bar: rsi_prev1 40.4 → refused (floor bar would read 80.1)
    assert tag["shift"] == 1 and tag["rsi_prev1"] < 44
    met = _row("METUSDT", "2026-07-29T11:50:08")            # proxy refused it (rsi_prev2 39.2); live rsi_prev1 44.7 → admitted
    rm = _rec(met)
    assert rm["shift"] == 1 and rm["rsi_prev1"] >= 44
    assert not bmp.fade_freshbreak_block(met.opened_at, rm, met.entry_rsi_prev, met[bmp.G])
    assert bmp.fade_freshbreak_stamp_block(met.opened_at, met.entry_rsi_prev, met[bmp.G])


def test_post_ship_stamped_fill_same_verdict_from_stamps_and_recompute():
    r = _row("XANUSDT", "2026-08-16T22:19:44")              # B3, live gate admitted it
    rec = _rec(r)
    assert rec is not None
    assert abs(rec["pgap"] - r[bmp.G]) < 0.02                # recomputed EMA13/50 gap = the stamp the gate read
    from_stamps = bmp.fade_freshbreak_rule(rec["rsi_prev1"], r[bmp.G])
    from_recompute = bmp.fade_freshbreak_rule(rec["rsi_prev1"], rec["pgap"])
    assert from_stamps is False and from_recompute is False  # = the live verdict (admitted)
    assert not bmp.fade_freshbreak_block(r.opened_at, rec, r.entry_rsi_prev, r[bmp.G])   # post-ship: never re-screened


def test_unanchored_falls_back_to_the_stamp_proxy():
    r = _row("ZENUSDT", "2026-08-02T11:59:50")
    assert bmp.fade_fb_recompute(None, r.opened_at, r.entry_price, r.entry_rsi_prev, r.entry_pair_rank) is None
    bad = bmp.fade_fb_recompute(bmp.fade_fb_load_klines(r.pair, dirs=(str(FX),)), r.opened_at, r.entry_price, r.entry_rsi_prev + 1.0, 105)
    assert bad is None                                       # a stamp that matches no bar never anchors
    assert bmp.fade_freshbreak_block(r.opened_at, None, 40.0, 0.1) == bmp.fade_freshbreak_stamp_block(r.opened_at, 40.0, 0.1)


def test_rule_semantics_match_the_engine():
    rule = bmp.fade_freshbreak_rule
    assert rule(43.99, -0.39) and not rule(44.0, 0.5) and not rule(40.0, -0.40)
    assert not rule(None, 0.5) and not rule(40.0, None) and not rule(float("nan"), 0.5)


def test_blacklist_applies_across_sleeves_and_eras():
    f = bmp.stack_pair_blacklist_reason
    assert f("SPIKE_FADE", NAMED_BL[0]) == "PAIR_BLACKLIST"
    assert f("MOMENTUM", "龙虾USDT") == "PAIR_BLACKLIST" and f("FLIP:FAN_RATIO_GATE", "USDCUSDT") == "PAIR_BLACKLIST"
    assert f("SPIKE_FADE", "BTCUSDT") == "PAIR_NO_TRADE" and f("MOMENTUM", "BTCUSDT") == ""   # momentum majors = probe-only path
    assert f("BULLRUN_LONG", "ONGUSDT") == "SLEEVE_PAIR_BLACKLIST" and f("BULLRUN_LONG", "SOLUSDT") == ""
    assert f("FRENZY_WIDE", "ETHUSDT") == "PAIR_NO_TRADE" and f("SPIKE_FADE", "ZENUSDT") == ""


def test_frozen_blacklists_match_live_config():
    c = json.loads((_ROOT / "trading_config.json").read_text())
    th = c["thresholds"]
    s = lambda v: {x.strip() for x in str(v or "").split(",") if x.strip()}
    assert s(c["pair_blacklist"]) == set(bmp.PAIR_BLACKLIST_FROZEN)
    assert s(c["no_trade_pairs"]) == set(bmp.NO_TRADE_PAIRS_FROZEN)
    for key, sleeve in (("bullrun_pair_blacklist", "BULLRUN_LONG"), ("bearrun_pair_blacklist", "BEARRUN_SHORT"),
                        ("surge_long_pair_blacklist", "SURGE_LONG"), ("surge_short_pair_blacklist", "SURGE_SHORT"),
                        ("frenzy_pair_blacklist", "FRENZY")):
        assert s(th[key]) == set(bmp.SLEEVE_BLACKLISTS_FROZEN[sleeve]), key


def test_master_refuses_the_six_and_keeps_no_blacklisted_pair():
    m = pd.read_csv(_ROOT / "reports" / "MASTER_POOL_stacked.csv", low_memory=False)
    assert (m.stack_version == bmp.STACK_VERSION).all() and bmp.STACK_VERSION == "2026-10-08c"
    key = m.pair + "@" + m.opened_at.astype(str).str[:19]
    for pair, t in NAMED_FB:
        r = m[key == f"{pair}@{t}"].iloc[0]
        assert not r.stack_keep and r.stack_block_reason == "FADE_FRESHBREAK", (pair, t)
    r = m[key == f"{NAMED_BL[0]}@{NAMED_BL[1]}"].iloc[0]
    assert not r.stack_keep and r.stack_block_reason == "PAIR_BLACKLIST"
    kept = m[m.stack_keep.astype(bool) & ~m.is_probe.astype(bool)]
    bad = [f"{p}({s})" for p, s in zip(kept.pair, kept.entry_strategy) if bmp.stack_pair_blacklist_reason(s, p)]
    assert not bad, bad


# ── caveman review (10-08c): seconds-into-bar choice, stamp confirmation, tie-break / ambiguity flags, no silent proxy, cache dedup ──
def test_seconds_rule_picks_previous_bar_near_the_boundary():
    for pair, t in (("TAGUSDT", "2026-08-03T07:55:02"), ("METUSDT", "2026-07-29T11:50:08")):
        rec = _rec(_row(pair, t))
        assert rec["shift"] == 1 and not rec["anchor_override"], (pair, rec)
    rec = _rec(_row("ZENUSDT", "2026-08-02T11:59:50"))       # 290 s into the bar → own bar
    assert rec["shift"] == 0 and not rec["anchor_override"]


def test_flat_closes_both_bars_confirm_pgap_breaks_the_tie():
    """PROM 07-29 (21 s in) and SNX 07-31: rsi_prev2 matches the stamp on BOTH bars (flat closes)."""
    prom = _row("PROMUSDT", "2026-07-29T15:35:21")
    rp = bmp.fade_fb_recompute(bmp.fade_fb_load_klines(prom.pair, dirs=(str(FX),)), prom.opened_at, prom.entry_price,
                               prom.entry_rsi_prev, prom.entry_pair_rank, stamp_pgap=prom[bmp.G])
    assert rp["shift"] == 1 and rp["pgap_tiebreak"] and abs(rp["pgap"] - prom[bmp.G]) < 1e-3 and not rp["ambiguous"]
    assert not bmp.fade_freshbreak_block(prom.opened_at, rp, prom.entry_rsi_prev, prom[bmp.G])   # admitted on either bar
    snx = _row("SNXUSDT", "2026-07-31T10:09:34")
    rs = bmp.fade_fb_recompute(bmp.fade_fb_load_klines(snx.pair, dirs=(str(FX),)), snx.opened_at, snx.entry_price,
                               snx.entry_rsi_prev, snx.entry_pair_rank, stamp_pgap=snx[bmp.G])
    assert rs["shift"] == 0 and not rs["pgap_tiebreak"] and not rs["ambiguous"]
    assert bmp.fade_freshbreak_block(snx.opened_at, rs, snx.entry_rsi_prev, snx[bmp.G])          # refused on either bar


def test_ambiguous_flag_when_the_two_bars_disagree(monkeypatch):
    snx = _row("SNXUSDT", "2026-07-31T10:09:34")               # rsi_prev1 39.57 (own bar) vs 42.01 (previous bar)
    monkeypatch.setattr(bmp, "FADE_FB_RSI_PREV_MIN", 41.0)     # a line between the two → the verdicts differ
    rs = bmp.fade_fb_recompute(bmp.fade_fb_load_klines(snx.pair, dirs=(str(FX),)), snx.opened_at, snx.entry_price,
                               snx.entry_rsi_prev, snx.entry_pair_rank, stamp_pgap=snx[bmp.G])
    assert rs["ambiguous"]


def test_stamp_overrides_the_seconds_rule(monkeypatch):
    monkeypatch.setattr(bmp, "FADE_FB_PREV_BAR_MAX_S", 299)    # seconds rule now prefers the previous bar for ZEN (290 s)
    rec = _rec(_row("ZENUSDT", "2026-08-02T11:59:50"))
    assert rec["shift"] == 0 and rec["anchor_override"]       # only the own bar matches the stamp


def test_missing_cache_is_a_build_error_not_a_silent_proxy(tmp_path):
    pre = FILLS[FILLS.opened_at.str[:19] < bmp.FADE_FB_SHIP_UTC].reset_index(drop=True)
    import pytest
    with pytest.raises(bmp.FadeFreshbreakUnanchored):
        bmp.fade_freshbreak_inputs(pre, kline_dirs=(str(tmp_path),), verbose=False)
    out = bmp.fade_freshbreak_inputs(pre, kline_dirs=(str(tmp_path),), verbose=False, allow_proxy=True)
    assert all(v is None for v in out.values())
    assert bmp.fade_freshbreak_inputs(pre, kline_dirs=(str(FX),), verbose=False)   # fixtures anchor every pre-ship row


def test_cache_dedup_keeps_the_complete_bar_and_rejects_a_corrupt_open(tmp_path):
    import pytest
    a, b = tmp_path / "a", tmp_path / "b"; a.mkdir(); b.mkdir()
    hdr = "open_time,o,h,l,c,vol,qvol\n"
    (a / "XUSDT.csv").write_text(hdr + "0,1,2,1,1.5,100,150\n300000,1.5,2,1.4,1.8,500,900\n")
    (b / "XUSDT.csv").write_text(hdr + "300000,1.5,1.6,1.5,1.55,40,62\n")      # snapshot taken while that bar was forming
    k = bmp.fade_fb_load_klines("XUSDT", dirs=(str(b), str(a)))
    assert len(k) == 2 and float(k.c.iloc[-1]) == 1.8 and float(k.vol.iloc[-1]) == 500
    (b / "XUSDT.csv").write_text(hdr + "300000,9.9,9.9,9.9,9.9,40,62\n")       # different OPEN = corrupt cache
    with pytest.raises(SystemExit):
        bmp.fade_fb_load_klines("XUSDT", dirs=(str(a), str(b)))


def test_no_trade_pairs_cover_every_scanner_fed_sleeve():
    for s in ("SPIKE_FADE", "SPIKE_CHASE", "SPIKE_BOUNCE"):
        assert bmp.stack_pair_blacklist_reason(s, "ETHUSDT") == "PAIR_NO_TRADE", s
