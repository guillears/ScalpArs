"""🩳 DL 348 addendum — the FLATBASE scout's three observe lines (scripts/scout_flatbase_short.py): ① NON-FLAT pumps priced like the
sleeve · ② MISSED / refused setups · ③ FRENZY longs blocked PAIR_HELD by an open FLATBASE short. Synthetic exports + injected klines /
funding fetchers — no network; the per-setup cache lives in tmp_path."""
import datetime as dt
import os
import sys
from types import SimpleNamespace as NS

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from services import flatbase as FB  # noqa: E402
from tests.test_flatbase_short import _series, T0, BAR, HOUR, DAY  # noqa: E402

MIN = 60_000


def _iso(ms):
    return pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%dT%H:%M:%S")


def _k5(series):
    """an injectable 5m fetcher over {pair: rows}: (sym, start, limit) → the rows from start (calls recorded)."""
    calls = []

    def kl(sym, start, limit):
        calls.append((sym, int(start), int(limit)))
        rows = series.get(sym) or []
        return [list(r) for r in rows if int(r[0]) >= int(start)][:int(limit)]
    return kl, calls


@pytest.fixture
def SC(tmp_path, monkeypatch):
    import scout_flatbase_short as SC
    monkeypatch.setattr(SC, "CACHE", str(tmp_path / "cache"))
    return SC


def _engine_expected(rows, on, th, fs_ms=None, slip=0.03, fund=0.0):
    """the engine's own S4 on the FULL series + the sleeve exit on the 72 h after it — the scout must reproduce it from its window."""
    sp = int(rows[on][0]) + BAR
    full = [list(r) for r in rows[: on + 1153]]
    sc = FB.s4_scan(full, sp, th, fs_ms)
    assert sc["status"] in (FB.ACT_MISSED, FB.ACT_ENTER)
    i = sc["idx"]
    r = FB.price_path_exit(sc["close"], rows[i + 1: i + 1 + 864], th, slip_pct=slip)
    return sp, sc, r["net_pct"] + fund


def test_nonflat_line_cohort_s4_pricing_turnover_and_cache(SC, tmp_path):
    th = SC._frozen_th()
    rows, on = _series(after=1300)
    sp, sc, want = _engine_expected(rows, on, th, fund=-0.2)
    rows2 = [[r[0] + 10 * DAY] + r[1:] for r in rows]                  # NF2: the same shape 10 days later, no market-cap source
    series = {"NF1USDT": rows, "NF2USDT": rows2, "NF3USDT": rows2}
    kl, calls = _k5(series)
    now = sp + 40 * DAY
    s4_close = sc["close"]
    supply = sc["vol24"] / (2.0 * s4_close)                             # → turnover 2.0 at the S4 bar (inferred)
    dec = tmp_path / "scalpars_decisions_paper_a.csv"
    led = [dict(t=_iso(sp), e="FLATBASE", pair="NF1USDT", fb_spike_at=_iso(sp), fb_status="TRACKING", fb_first_seen_at=_iso(sp - BAR)),
           dict(t=_iso(sp), e="FLATBASE", pair="NF1USDT", fb_spike_at=_iso(sp), fb_status="NOT_FLAT", fb_first_seen_at=_iso(sp - BAR),
                fb_flat14_pct=80.0, fb_reason="14-day range 80.0 % > 44.4 %"),
           dict(t=_iso(sp), e="FLATBASE", pair="NF2USDT", fb_spike_at=_iso(sp + 10 * DAY), fb_status="NOT_FLAT",
                fb_first_seen_at=_iso(sp + 10 * DAY - BAR)),
           dict(t=_iso(now), e="FLATBASE", pair="NF3USDT", fb_spike_at=_iso(now - 2 * DAY), fb_status="NOT_FLAT",   # window open → waits
                fb_first_seen_at=_iso(now - 2 * DAY)),
           dict(t=_iso(sp), e="FLATBASE", pair="OLDUSDT", fb_spike_at=_iso(T0), fb_status="NOT_FLAT"),            # before the deploy: out
           dict(t=_iso(sp), e="FLATBASE", pair="FLATUSDT", fb_spike_at=_iso(sp), fb_status="OPENED", fb_flat14_pct=20.0)]
    pd.DataFrame(led).to_csv(dec, index=False)
    orders = tmp_path / "scalpars_orders_paper_a.csv"
    pd.DataFrame([dict(opened_at=_iso(sp - 2 * DAY), closed_at=_iso(sp - 2 * DAY + HOUR), pair="NF1USDT", direction="LONG", status="CLOSED",
                       entry_strategy="FRENZY_WILLY", pnl=1.0, pnl_percentage=1.0, close_reason="FRENZY_TP", entry_price=1.0, quantity=10.0,
                       entry_mcap_usd=supply * 1.0),
                  dict(opened_at=_iso(sp + 5 * HOUR), closed_at=_iso(sp + 50 * HOUR), pair="FLATUSDT", direction="SHORT", status="CLOSED",
                       entry_strategy="FLATBASE_SHORT", pnl=2.0, pnl_percentage=7.5, close_reason="FRENZY_TP", entry_price=1.0, quantity=50.0)
                  ]).to_csv(orders, index=False)
    since = sp - 3 * DAY
    L = SC.run(now_ms=now, paths=[str(orders)], dec_paths=[str(dec)], since_ms=since, klines=kl, funding=lambda s, a, b: -0.2)
    txt = "\n".join(L)
    sec = txt[txt.index("## 🩳①"):txt.index("## 🩳②")]
    assert "Cohort: 3 NOT_FLAT episodes" in sec and "priced 2 · waiting 1" in sec                     # TRACKING folded into its decision
    # the scout's S4 window = 20 h before the spike bar → its last candidate; the exit path = the 72 h after the S4 bar (the shadows' call)
    start = sp - BAR - SC.S4_PRE_BARS * BAR
    assert ("NF1USDT", start, SC.S4_PRE_BARS + 1153) in calls and ("NF1USDT", sc["bar_open"] + BAR, 864) in calls
    assert not any(c[0] == "NF3USDT" for c in calls)                                                  # an open window is never read
    assert f"> 1 N 1 · WR {100 if want > 0 else 0}% · avg {want:+.3f} %" in sec and "1 inferred" in sec
    assert "unscored (no mcap source) N 1" in sec and "FLAT real N 1" in sec
    # cached per setup: a second run with no network gives the same numbers
    L2 = SC.run(now_ms=now, paths=[str(orders)], dec_paths=[str(dec)], since_ms=since, klines=lambda *a: pytest.fail("refetched"),
                funding=lambda s, a, b: pytest.fail("refetched"))
    t2 = "\n".join(L2)
    assert f"> 1 N 1 · WR {100 if want > 0 else 0}% · avg {want:+.3f} %" in t2
    assert os.path.exists(os.path.join(SC.CACHE, "setups", f"nonflat_NF1USDT_{sp}_s0.030.json"))


def test_turnover_inference_and_s4_states(SC):
    pts = {"AUSDT": [(0, 1e6), (40 * DAY, 2e6)]}
    assert SC.turnover_of(0.7, "AUSDT", 0, 1.0, 5e6, pts) == (0.7, "stamped")
    v, how = SC.turnover_of(None, "AUSDT", 1 * DAY, 2.0, 4e6, pts)
    assert how == "inferred" and v == pytest.approx(2.0)                                               # 4e6 / (1e6 × 2.0)
    assert SC.turnover_of(None, "AUSDT", 100 * DAY, 2.0, 4e6, pts) == (None, None)                    # nearest stamp > 30 days away
    assert SC.turnover_of(None, "BUSDT", 0, 2.0, 4e6, pts) == (None, None)
    th = SC._frozen_th()
    rows, on = _series(after=1300, stay_above=True, early_dump=False)                                                    # never closes below the VWAP
    sp = int(rows[on][0]) + BAR
    kl, _ = _k5({"X": rows})
    assert SC.s4_from_klines("X", sp, None, None, sp + 30 * DAY, klines=kl)["state"] == "NO_S4"
    assert SC.s4_from_klines("X", sp, None, None, sp + 50 * HOUR, klines=kl)["state"] == "WAIT"
    assert SC.s4_from_klines("Y", sp, None, None, sp + 30 * DAY, klines=kl)["state"] == "UNPRICED"
    rows, on = _series(after=1300)
    sp = int(rows[on][0]) + BAR
    kl, _ = _k5({"X": rows})
    got = SC.s4_from_klines("X", sp, None, None, sp + 30 * DAY, klines=kl)
    want = FB.s4_scan([list(r) for r in rows[: on + 1153]], sp, th)
    assert got["state"] == "S4" and got["s4_ms"] == want["bar_open"] and got["px"] == want["close"]
    # the first-seen gate (the engine's): a first-seen bar after the natural S4 pushes it to the next qualifying boundary
    late = SC.s4_from_klines("X", sp, int(want["bar_open"]) + BAR, None, sp + 30 * DAY, klines=kl)
    assert late["state"] == "S4" and late["s4_ms"] > got["s4_ms"]


def test_missed_line_per_reason(SC, tmp_path):
    th = SC._frozen_th()
    rows, on = _series(after=1300)
    sp, sc, want = _engine_expected(rows, on, th)
    flat_rows, on2 = _series(after=1300, stay_above=True, early_dump=False)
    kl, calls = _k5({"M1USDT": rows, "CAPUSDT": rows, "UNRUSDT": flat_rows})
    now = sp + 40 * DAY
    dec = tmp_path / "scalpars_decisions_paper_b.csv"
    s4 = int(sc["bar_open"])
    pd.DataFrame([
        dict(e="FLATBASE", pair="M1USDT", price=sc["close"], fb_spike_at=_iso(sp), fb_status="MISSED", fb_first_seen_at=_iso(sp - BAR),
             fb_flat14_pct=20.0, fb_reason="the first qualifying 4-h close (…) was not judged live (outage / restart) — never caught up"),
        dict(e="FLATBASE", pair="CAPUSDT", price=sc["close"], fb_spike_at=_iso(sp), fb_status="REFUSED", fb_entry_bar_at=_iso(s4),
             fb_reason="10 FLATBASE_SHORT open (flatbase_max_open 10) at the S4 bar", fb_vol24_usd=sc["vol24"]),
        dict(e="FLATBASE", pair="UNRUSDT", fb_spike_at=_iso(int(flat_rows[on2][0]) + BAR), fb_status="UNREADABLE",
             fb_reason="14-day history unreadable after 3 reads (fail-closed)"),
        dict(e="FLATBASE", pair="SHUSDT", price=1.0, fb_spike_at=_iso(sp), fb_status="SHADOW", fb_entry_bar_at=_iso(s4))]).to_csv(dec, index=False)
    L = SC.run(now_ms=now, paths=[], dec_paths=[str(dec)], since_ms=sp - DAY, klines=kl, funding=lambda s, a, b: 0.0)
    txt = "\n".join(L)
    sec = txt[txt.index("## 🩳②"):txt.index("## 🩳③")]
    assert "MISSED 1 · FLATBASE_MAX_OPEN 1 · FLAT_UNREADABLE 1" in sec
    exp = f"N 1 · WR {100 if want > 0 else 0}% · avg {want:+.3f} %"
    assert f"- MISSED: 1 · {exp}" in sec and "FLAT 1 · not yet read 0" in sec
    assert f"- FLATBASE_MAX_OPEN: 1 · {exp}" in sec
    assert "- FLAT_UNREADABLE: 1 · N 0 · priced 0 · waiting 0 (S4 window / 72 h hold unfinished) · no S4 bar 1" in sec
    assert not any(c[0] == "CAPUSDT" and c[2] != 864 for c in calls)                                  # stamped S4 → only the exit path read
    assert any(c[0] == "M1USDT" and c[2] == SC.S4_PRE_BARS + 1153 for c in calls)                     # MISSED: S4 recomputed
    assert SC._reason_class("REFUSED", "pair held (MOMENTUM LONG open) at the S4 bar") == "REFUSED_PAIR_HELD"
    assert SC._reason_class("REFUSED", "reached the order 400s after the 4-h bar closed (max 300s — same 5m bar) at the S4 bar") == "REFUSED_LATE"


def _th_long():
    return NS(frenzy_tp_pct=1.5, frenzy_stop_pct=3.0, frenzy_long_stop_pct=4.0, frenzy_lite_tp_pct=3.0, frenzy_lite_c_enabled=True,
              frenzy_lite_c_tp_pct=2.0, frenzy_lite_c_vol_mult=20.0, frenzy_lite_c_btc_slope_min=0.0, frenzy_max_hold_minutes=720,
              frenzy_lock_arm_pct=0.0, frenzy_trail_arm_pct=5.0, frenzy_trail_giveback_pct=1.5, frenzy_willy_tp_pct=1.25,
              frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, frenzy_willy_late_tp_minutes=30.0, frenzy_willy_late_tp_pct=0.25,
              resting_tp_willy_enabled=True)


def _m1(paths):
    """(sym, start) → [(o, h, l, c)] per minute, padded flat at the last close to `limit` rows."""
    def f(sym, start, limit):
        spec = paths.get((sym, int(start)))
        if spec is None:
            return []
        out = [[int(start) + i * MIN, o, h, l, c, 1.0] for i, (o, h, l, c) in enumerate(spec)]
        last = spec[-1][3]
        out += [[int(start) + i * MIN, last, last, last, last, 1.0] for i in range(len(spec), int(limit))]
        return out[:int(limit)]
    return f


def test_blocked_line_sleeves_holder_fold_and_net(SC, tmp_path):
    ta = int(pd.Timestamp("2026-11-01T00:00:00").value // 10**6)
    t_w, t_w2, t_lite, t_long, t_gen = ta + HOUR, ta + HOUR + 5 * MIN, ta + 5 * HOUR, ta + 9 * HOUR, ta + 10 * HOUR
    flat = (1.0, 1.0, 1.0, 1.0)
    t_b4 = ta + 2 * HOUR
    paths = {("B1USDT", t_w + MIN): [flat] * 15 + [(1.0, 1.03, 1.0, 1.03)],                         # WILLY: TP at minute 15 → +1.25 line
             ("B1USDT", t_lite + MIN): [(1.0, 1.04, 1.0, 1.04)],                                     # LITE: both TPs touch on minute 0
             ("B1USDT", t_long + MIN): [(1.0, 1.0, 0.965, 0.97), (0.97, 1.03, 0.97, 1.03)],          # −3.5 % then +3 %: the −4 stop holds
             ("B2USDT", t_w + MIN): [(1.0, 1.03, 1.0, 1.03)],
             ("B4USDT", t_b4 + MIN): [(1.0, 1.03, 1.0, 1.03)],                                       # B4: an OPEN holder
             # the bucket's own first minute spikes: entry is bucket + 1 min, so this +10 % print is never seen
             ("B1USDT", t_w): [(1.0, 1.10, 1.0, 1.10)]}
    m1 = _m1(paths)
    orders = tmp_path / "scalpars_orders_paper_c.csv"
    pd.DataFrame([dict(opened_at=_iso(ta), closed_at=_iso(ta + 72 * HOUR), pair="B1USDT", direction="SHORT", status="CLOSED",
                       entry_strategy="FLATBASE_SHORT", pnl=3.0, pnl_percentage=5.0, close_reason="MAX_HOLD_TIME", entry_price=1.0,
                       quantity=60.0, funding_fee_usd=-0.5),
                  dict(opened_at=_iso(ta - 5 * DAY), closed_at=_iso(ta - 5 * DAY + HOUR), pair="ZZUSDT", direction="LONG", status="CLOSED",
                       entry_strategy="FRENZY_WILLY", pnl=1.0, pnl_percentage=1.0, close_reason="FRENZY_TP", entry_price=1.0, quantity=100.0),
                  dict(opened_at=_iso(ta - 4 * DAY), closed_at=_iso(ta - 4 * DAY + HOUR), pair="ZZUSDT", direction="LONG", status="CLOSED",
                       entry_strategy="FRENZY_LONG", pnl=1.0, pnl_percentage=1.0, close_reason="FRENZY_TP", entry_price=1.0, quantity=200.0),
                  dict(opened_at=_iso(ta + HOUR), closed_at=None, pair="B4USDT", direction="SHORT", status="OPEN",
                       entry_strategy="FLATBASE_SHORT", entry_price=1.0, quantity=60.0)
                  ]).to_csv(orders, index=False)
    blk = lambda t, pair, gate: dict(t=_iso(t)[:16] + ":00", e="BLOCK", pair=pair, dir="LONG", gate=gate, n=1)
    dec = tmp_path / "scalpars_decisions_paper_c.csv"
    pd.DataFrame([blk(t_w, "B1USDT", "FRENZY_WILLY_PAIR_HELD"), blk(t_w2, "B1USDT", "FRENZY_WILLY_PAIR_HELD"),
                  blk(t_w, "B1USDT", "FRENZY_WILLY_PAIR_HELD"),                                         # the same bucket in a 2nd export
                  blk(t_lite, "B1USDT", "FRENZY_LITE_PAIR_HELD"), blk(t_long, "B1USDT", "FRENZY_PAIR_HELD"),
                  blk(t_gen, "B1USDT", "PAIR_HELD"), blk(t_w, "B2USDT", "FRENZY_WILLY_PAIR_HELD"),       # B2: no FLATBASE holder
                  blk(ta - 10 * DAY, "B1USDT", "FRENZY_WILLY_PAIR_HELD"),
                  blk(t_b4, "B4USDT", "FRENZY_WILLY_PAIR_HELD"), blk(t_b4 + 20 * MIN, "B4USDT", "FRENZY_LITE_RETRY_PAIR_HELD"),   # no path → UNPRICED
                  blk(t_b4 + 25 * MIN, "B4USDT", "FRENZY_LITE_RETRY_PAIR_HELD"),                           # …its repeat folds on the cap                                  # before the deploy
                  dict(t=_iso(t_w), e="BLOCK", pair="B1USDT", dir="SHORT", gate="PAIR_HELD", n=1)]).to_csv(dec, index=False)
    now = ta + 20 * DAY
    L = SC.run(now_ms=now, paths=[str(orders)], dec_paths=[str(dec)], since_ms=ta - DAY, klines=lambda *a: [], funding=lambda *a: 0.0,
               klines1m=m1, th_long=_th_long())
    txt = "\n".join(L)
    sec = txt[txt.index("## 🩳③"):]
    assert ("Cohort: 9 PAIR_HELD LONG blocks ≥ deploy (any holder) · held by a FLATBASE_SHORT (inferred from the orders exports): 7 sleeve-named · "
            "other / unknown holder: 1 · repeats inside a priced trade folded: 2 · generic PAIR_HELD on a FLATBASE-held pair: 1 UNATTRIBUTED") in sec
    assert "- FRENZY_WILLY: 2 · N 2 · WR 100% · avg +1.150 % · Σ +2.30 pts · ≈ $+2.30 (× median FRENZY_WILLY notional $100)" in sec
    assert "- FRENZY_LONG: 1 · N 1 · WR 100% · avg +1.400 % · Σ +1.40 pts · ≈ $+2.80" in sec          # −4 (DL 317), not the shared −3
    assert "- FRENZY_LITE_RETRY: 1 · N 0" in sec and "unpriced (no data / fetch) 1" in sec              # its repeat folded on the cap
    assert "- FRENZY_LITE: 1 · N 0 · unpriced 1 (LITE C" in sec and "bracket avg [+1.900, +2.900] %" in sec
    assert "2 FLATBASE_SHORT holds · closed 1 N 1 · WR 100% · avg +4.167 %" in sec and "$+2.50 (pnl + funding) · open 1" in sec
    # ONE blocked position per CLOSED hold (WILLY beats FRENZY_LONG; the open B4 hold's WILLY is out of the $ net)
    assert "FLATBASE $+2.50 − blocked longs ≈ $+1.15 (1 positions) = $+1.35" in sec
    # LITE C off → LITE is faithful and priced at its own TP
    off = SC.price_long("FRENZY_LITE", "B1USDT", t_lite, None, now, m1=m1, th=NS(**dict(vars(_th_long()), frenzy_lite_c_enabled=False)))
    assert off["state"] == "PRICED" and off["pct"] == pytest.approx(2.9)
    # an unfinished exit window waits; a missing 1m path is unpriced (never guessed)
    assert SC.price_long("FRENZY_WILLY", "B1USDT", now - 10 * MIN, None, now, m1=m1, th=_th_long())["state"] == "WAIT"
    assert SC.price_long("FRENZY_WILLY", "B9USDT", t_w, None, now, m1=m1, th=_th_long())["state"] == "UNPRICED"


def test_sections_crash_isolated(SC, tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "missed_lines", lambda ctx: 1 / 0)
    L = SC.run(now_ms=T0 + 30 * DAY, paths=[], dec_paths=[], since_ms=T0, klines=lambda *a: [], funding=lambda *a: 0.0)
    txt = "\n".join(L)
    assert "## 🩳① FLATBASE observe" in txt and "## 🩳② FLATBASE observe — MISSED setups\n\nUnavailable this run" in txt
    assert "## 🩳③ FLATBASE observe" in txt


def test_ctx_failure_keeps_the_gate_and_no_block_skips_config(SC, tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "read_decisions", lambda *a, **k: 1 / 0)
    L = SC.run(now_ms=T0 + 30 * DAY, paths=[], dec_paths=[], since_ms=T0, klines=lambda *a: [], funding=lambda *a: 0.0)
    txt = "\n".join(L)
    assert "Gate verdict:" in txt and "observe lines unavailable (division by zero)" in txt and "## 🩳① FLATBASE observe —" not in txt
    monkeypatch.undo()
    monkeypatch.setattr(SC, "CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(SC, "_live_th", lambda: pytest.fail("config read with no block"))
    L = SC.run(now_ms=T0 + 30 * DAY, paths=[], dec_paths=[], since_ms=T0, klines=lambda *a: [], funding=lambda *a: 0.0)
    assert "no PAIR_HELD LONG block yet" in "\n".join(L)


def test_cache_keys_and_final_no_klines(SC, tmp_path):
    th = SC._frozen_th()
    rows, on = _series(after=1300)
    sp, sc, _ = _engine_expected(rows, on, th)
    kl, _ = _k5({"P1USDT": rows})
    r = NS(pair="P1USDT", spike_ms=sp, fs_ms=None, s4_ms=None, px=None, vol24=None)
    x = SC.price_setup("nonflat", r, None, sp + 40 * DAY, klines=kl, funding=lambda *a: 0.0, slip=0.05)
    assert x["state"] == "PRICED" and os.path.exists(os.path.join(SC.CACHE, "setups", f"nonflat_P1USDT_{sp}_s0.050.json"))
    y = SC.price_setup("nonflat", r, None, sp + 40 * DAY, klines=kl, funding=lambda *a: 0.0, slip=0.03)   # another slippage → its own entry
    assert y["state"] == "PRICED" and os.path.exists(os.path.join(SC.CACHE, "setups", f"nonflat_P1USDT_{sp}_s0.030.json"))
    # a delisted / invalid symbol: "no klines" on a window closed > 1 day ago is cached as final (no weight spent next run)
    d = NS(pair="GONEUSDT", spike_ms=sp, fs_ms=None, s4_ms=None, px=None, vol24=None)
    z = SC.price_setup("missed", d, None, sp + 40 * DAY, klines=kl, funding=lambda *a: 0.0)
    assert z["state"] == "UNPRICED" and SC.price_setup("missed", d, None, sp + 40 * DAY, klines=lambda *a: pytest.fail("refetched")) == z
    fresh = NS(pair="GONEUSDT", spike_ms=sp + 30 * DAY, fs_ms=None, s4_ms=None, px=None, vol24=None)   # closed < 1 day ago: retried
    assert SC.price_setup("missed", fresh, None, sp + 30 * DAY + 102 * HOUR, klines=kl)["state"] == "UNPRICED"
    assert not os.path.exists(os.path.join(SC.CACHE, "setups", f"missed_GONEUSDT_{sp + 30 * DAY}_s0.030.json"))


def test_resize_ladder_frozen_steps_and_step_down():
    """📈 (DECISION_LOG 363) FLATBASE re-size ladder 0.2 → 0.5 → 1.0 (5× unchanged): every rung is judged by the sleeve's own frozen gate
    (flatbase_gate) on the CURRENT run of fills at that size — STEP_UP → step-up candidate (+ re-decide flatbase_max_open); OFF / REVIEW at a
    rung above 0.2 → step-down candidate. One stop-out never triggers a step-down (deep review I1)."""
    import pandas as pd
    import scout_flatbase_short as S
    def fills(n, net, cm, days=None, reason="TAKE_PROFIT"):
        days = days or n
        return pd.DataFrame(dict(o=pd.to_datetime([f"2026-11-{1 + (i % days):02d}T10:00:{i % 60:02d}" for i in range(n)]),
                                 net=[net] * n, cell_multiplier=[cm] * n, close_reason=[reason] * n)).sort_values("o", ignore_index=True)
    t = " ".join(S.resize_lines(fills(5, 1.0, 0.2), inv_now=0.2))
    assert "next step 0.2 → 0.5" in t and "now COLLECTING" in t and "CANDIDATE" not in t
    t = " ".join(S.resize_lines(fills(30, 1.0, 0.2, 10), inv_now=0.2))
    assert "✅ STEP-UP candidate 0.2 → 0.5" in t and "re-decide flatbase_max_open" in t
    assert "✅ STEP-UP candidate 0.5 → 1" in " ".join(S.resize_lines(fills(30, 0.8, 0.5, 10), inv_now=0.5))
    assert "next step 0.5 → 1" in " ".join(S.resize_lines(fills(12, 0.8, 0.5, 10), inv_now=0.5))      # collecting at the new size
    assert "🛑 STEP-DOWN candidate to 0.2" in " ".join(S.resize_lines(fills(15, -0.5, 0.5, 10), inv_now=0.5))   # first 10 avg < 0
    one_stop = pd.concat([fills(5, 1.0, 0.5, 5), fills(1, -15.1, 0.5, 1, "STOP_LOSS").assign(o=lambda d: d.o + pd.Timedelta(days=9))],
                         ignore_index=True)
    assert "STEP-DOWN" not in " ".join(S.resize_lines(one_stop, inv_now=0.5))                          # I1: one stop ≠ step down
    three = pd.concat([fills(3, 1.0, 0.5, 3), fills(3, -15.1, 0.5, 3, "STOP_LOSS").assign(o=lambda d: d.o + pd.Timedelta(days=5))],
                      ignore_index=True)
    assert "🛑 STEP-DOWN" in " ".join(S.resize_lines(three, inv_now=0.5))                             # 3 stops in a row / Σ ≤ −30
    assert "STEP-DOWN" not in " ".join(S.resize_lines(fills(15, -0.5, 0.2, 10), inv_now=0.2))          # no rung below 0.2
    assert "top of the ladder" in " ".join(S.resize_lines(fills(3, 1.0, 1.0), inv_now=1.0))
    assert "OFF the ladder" in " ".join(S.resize_lines(fills(3, 1.0, 0.3), inv_now=0.3))
    assert "unreadable" in " ".join(S.resize_lines(None, inv_now=0.5))                                 # never raises
    # after a step-down (0.5 fills exist, back at 0.2) only the new 0.2 run counts — the old 0.2 STEP_UP does not re-arm
    back = pd.concat([fills(30, 1.0, 0.2, 10), fills(15, -0.5, 0.5, 10).assign(o=lambda d: d.o + pd.Timedelta(days=20)),
                      fills(3, 1.0, 0.2, 3).assign(o=lambda d: d.o + pd.Timedelta(days=60))], ignore_index=True)
    t = " ".join(S.resize_lines(back, inv_now=0.2))
    assert "CANDIDATE" not in t and "3 fills on 3 days" in t
    run2 = pd.concat([fills(15, 1.0, 0.5, 10), fills(2, 1.0, 0.2, 2).assign(o=lambda d: d.o + pd.Timedelta(days=20)),
                      fills(4, 1.0, 0.5, 4).assign(o=lambda d: d.o + pd.Timedelta(days=40))], ignore_index=True)
    assert "4 fills on 4 days" in " ".join(S.resize_lines(run2, inv_now=0.5))                         # 0.5 → 0.2 → 0.5
    # caveman review: off the ladder (0.75 / 0.25) → no step verdict even with a good prior 0.5 run
    t = " ".join(S.resize_lines(fills(30, 1.0, 0.5, 10), inv_now=0.75))
    assert "OFF the ladder" in t and "CANDIDATE" not in t
    assert "OFF the ladder" in " ".join(S.resize_lines(fills(3, 1.0, 0.25), inv_now=0.25))
    # unknown size → counted aloud, never silently dropped; newest fill ≠ config → sync warning
    t = " ".join(S.resize_lines(fills(4, 1.0, float("nan")), inv_now=0.2))
    assert "4 fill(s) with unknown size not counted" in t and "0 fills on 0 days" in t
    assert "sync the config" in " ".join(S.resize_lines(fills(4, 1.0, 0.5), inv_now=0.2))
    # OFF from the main read (30 on 10 days avg < 0, no early warning: wins then losses) at 0.5 → step down
    mixed = pd.concat([fills(10, 2.0, 0.5, 10), fills(20, -1.5, 0.5, 10).assign(o=lambda d: d.o + pd.Timedelta(days=10))], ignore_index=True)
    assert "🛑 STEP-DOWN candidate to 0.2" in " ".join(S.resize_lines(mixed, inv_now=0.5))
    # the config-read path prints the cap note on a step-up
    t = " ".join(S.resize_lines(fills(30, 1.0, S.flatbase_params(S._frozen_th())["invest"], 10)))
    assert "re-decide flatbase_max_open" in t or "OFF the ladder" in t
