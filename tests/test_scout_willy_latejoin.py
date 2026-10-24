"""🕰 tests for scripts/scout_willy_latejoin.py (DECISION_LOG 366 — WILLY late-join refusals priced on the live V2b exit; observe-only) —
synthetic decisions rows, synthetic 5m klines and net paths, no network, state / CSV in tmp."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_latejoin as S  # noqa: E402
import scout_willy_exit_v2 as V2  # noqa: E402
import scout_b1h_negflank as NF  # noqa: E402
import scout_willy_timecap as WT  # noqa: E402

BAR, MIN = S.BAR, S.MIN
CFG = dict(wait=60.0, floor=20e6, taker=0.00045)
NOW = int(pd.Timestamp("2026-10-26 00:00:00").value // 1_000_000)
SINCE = int(pd.Timestamp("2026-10-23 03:04:00").value // 1_000_000)


FAR = NOW + 365 * 86_400_000                                  # a "newest export" stamp well past every test spike (freeze allowed)


@pytest.fixture(autouse=True)
def _no_net(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "_NET_BLOCKED", True)
    monkeypatch.setattr(NF, "_cooldown_until", lambda: 0)     # the real shared cooldown file is read before the network guard
    monkeypatch.setattr(S, "CACHE", str(tmp_path / "k5"))
    monkeypatch.setattr(S, "BAD_SYMBOLS", str(tmp_path / "bad_symbols.json"))
    monkeypatch.setattr(WT, "_NET_BLOCKED", True)              # guard: no test may reach the 1m fetch or the real timecap caches
    monkeypatch.setattr(WT, "MY_CACHE", str(tmp_path / "timecap"))


def _ms(s):
    return int(pd.Timestamp(s).value // 1_000_000)


def _dec(tmp_path, name, rows):
    p = tmp_path / name
    pd.DataFrame(rows, columns=["t", "e", "pair", "dir", "gate", "reason"]).to_csv(p, index=False)
    return str(p)


def _row(t, pair, L, gate="FRENZY_WILLY_A_NOMISS_MISSED"):
    return [t, "WILLY", pair, "LONG", gate, f"read {L} bars after the spike · a red bar already closed"]


def _k5(start, n, qv=50_000.0, px=1.0, red=(), spike=None, spike_h=None):
    """n 5m bars from start: green flat bars (o 1.0 → c 1.001) except opens in `red` (c < o); spike bar → its high spike_h."""
    o = np.array([start + i * BAR for i in range(n)], dtype="int64")
    rows = []
    for t in o:
        if t in red:
            rows.append((t, px * 1.01, px * 1.012, px * 0.99, px * 1.0, qv))
        else:
            rows.append((t, px, px * 1.002, px * 0.999, px * 1.001, qv))
    k = pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c", "qv"])
    if spike is not None:
        k.loc[k.open_time == spike, "h"] = spike_h
    return k


def _P(te, ev, cap=90, bar=False):
    return V2._P(te, ev, cap=cap, bar=bar)


# ─────────────────────────── parsing / timing ───────────────────────────
def test_constants_pinned():
    assert (S.DL, S.NOMISS_DL, S.LAG_MIN) == (366, 350, 10)
    assert (S.GATE_N, S.GATE_DAYS, S.BOOT_N, S.BOOT_SEED, S.GATE_BUCKET) == (30, 10, 4000, 7, "4-12")
    assert S.EVENTS == ("FRENZY_WILLY_A_NOMISS_MISSED", "FRENZY_WILLY_A_NOMISS_LATE", "FRENZY_WILLY_A_NOMISS_OFF")
    assert (S.VOL_BARS, S.LATENCY_MS, S.SLIP_PCT) == (288, 8000, 0.10)
    assert S.STATE.endswith(os.path.join("reports", ".willy_latejoin_gate.json"))
    assert S.CSV_OUT.endswith(os.path.join("reports", "SCOUT_WILLY_LATEJOIN.csv"))


def test_parse_L_and_bucket():
    assert S.parse_L("read 28 bars after the spike · a red bar already closed") == 28
    assert S.parse_L("read 4 bars after the spike · no red bar yet, the red wait from the spike is over") == 4
    assert S.parse_L("read 7 bars after the spike · no red bar missed · switch OFF") == 7
    assert S.parse_L("garbage") is None and S.parse_L(None) is None and S.parse_L(float("nan")) is None
    assert [S.bucket(x) for x in (4, 12, 13, 36, 37, 288, 3)] == ["4-12", "4-12", "13-36", "13-36", "37+", "37+", "<=3"]


def test_event_bars_match_the_engine_convention():
    # the engine's allowed INIT event: t 16:25:08, "read 4 bars after the spike 10-23 16:05 UTC" (16:05 = the spike bar's CLOSE)
    b = S.event_bars(_ms("2026-10-23 16:25:08.688"), 4)
    assert b["B"] == _ms("2026-10-23 16:25") and b["join_open"] == _ms("2026-10-23 16:20") and b["join_close"] == _ms("2026-10-23 16:25")
    assert b["spike_close"] == _ms("2026-10-23 16:05") and b["spike_open"] == _ms("2026-10-23 16:00")
    # frenzy_willy_late_bars inverse: (B − spike close) // 5 min = L
    from services.frenzy import frenzy_willy_late_bars
    assert frenzy_willy_late_bars(b["spike_close"], b["B"]) == 4
    # LIGHTUSDT shape: read at the 13:05 pass as 28 bars late → spike bar 10:40–10:45 (close 10:45 UTC)
    lt = S.event_bars(_ms("2026-10-24 13:05:07"), 28)
    assert lt["spike_open"] == _ms("2026-10-24 10:40") and lt["spike_close"] == _ms("2026-10-24 10:45")


def test_read_refusals_dedup_modes_since_and_bad_rows(tmp_path):
    rows = [_row("2026-10-23T10:55:09.867", "SOPHUSDT", 38),
            _row("2026-10-23T11:00:09.000", "SOPHUSDT", 39),                        # same spike, a later read → dropped (first read kept)
            _row("2026-10-23T12:00:00", "AAAUSDT", 6, "FRENZY_WILLY_A_NOMISS_LATE"),
            _row("2026-10-23T12:05:00", "OFFUSDT", 5, "FRENZY_WILLY_A_NOMISS_OFF"),
            ["2026-10-23T12:10:00", "WILLY", "BBBUSDT", "LONG", "FRENZY_WILLY_A_NOMISS", "read 4 bars after the spike"],   # allowed → not a refusal
            ["2026-10-23T12:10:00", "BLOCK", "CCCUSDT", "LONG", "FRENZY_WILLY_A_NOMISS_MISSED", "read 9 bars after the spike"],  # e ≠ WILLY
            ["2026-10-23T12:15:00", "WILLY", "BADUSDT", "LONG", "FRENZY_WILLY_A_NOMISS_MISSED", "no number here"],
            ["not-a-time", "WILLY", "BAD2USDT", "LONG", "FRENZY_WILLY_A_NOMISS_MISSED", "read 9 bars after the spike"],
            _row("2026-10-22T10:00:00", "OLDUSDT", 9)]                              # before the deploy
    live = _dec(tmp_path, "scalpars_decisions_live_2026-10-23_14-43-15.csv", rows)
    paper = _dec(tmp_path, "scalpars_decisions_paper_2026-10-24_11-21-29.csv", rows[:4] + [_row("2026-10-24T07:50:08", "KATUSDT", 54)])
    d, bad, newest = S.read_refusals([paper, live], pd.Timestamp(SINCE, unit="ms"))
    assert bad == 2 and newest == _ms("2026-10-24 11:21:29")
    assert list(d.pair) == ["SOPHUSDT", "AAAUSDT", "OFFUSDT", "KATUSDT"]
    assert list(d["mode"]) == ["live", "live", "live", "paper"]                   # the oldest export holding the row
    assert int(d[d.pair == "SOPHUSDT"].L.iloc[0]) == 38
    assert set(d.gate) == {"FRENZY_WILLY_A_NOMISS_MISSED", "FRENZY_WILLY_A_NOMISS_LATE", "FRENZY_WILLY_A_NOMISS_OFF"}
    # an unreadable / missing file never raises
    d2, _, _ = S.read_refusals([str(tmp_path / "nope.csv")], None)
    assert len(d2) == 0


# ─────────────────────────── per-refusal math ───────────────────────────
def test_vol24_lowvol_liquid_unread():
    sp = _ms("2026-10-24 10:45")
    k = _k5(sp - 287 * BAR, 300, qv=50_000.0)                    # 288 bars to the spike × $50k = $14.4M → LOWVOL
    v, n = S.vol24(k, sp)
    assert n == 288 and abs(v - 288 * 50_000.0) < 1e-6 and S.vol_class(v, n, 20e6) == "LOWVOL"
    k2 = _k5(sp - 287 * BAR, 300, qv=100_000.0)                  # $28.8M → LIQUID
    assert S.vol_class(*S.vol24(k2, sp), 20e6) == "LIQUID"
    k3 = _k5(sp - 100 * BAR, 120, qv=1e9)                        # a new listing: < 288 bars → UNREAD (never a guessed class)
    assert S.vol_class(*S.vol24(k3, sp), 20e6) == "UNREAD"
    assert S.vol24(None, sp) == (None, 0) and S.vol_class(None, 0, 20e6) == "UNREAD"


def test_red_entry_first_red_join_bar_counts_and_expiry():
    jo = _ms("2026-10-24 13:00")
    k = _k5(jo - 10 * BAR, 40, red={jo})                         # the join bar itself is red → entry on it
    e = S.red_entry(k, jo, 60, NOW)
    assert e["status"] == "RED" and e["red_open"] == jo
    k = _k5(jo - 10 * BAR, 40, red={jo - BAR, jo + 3 * BAR, jo + 5 * BAR})   # a red BEFORE the join bar does not count
    assert S.red_entry(k, jo, 60, NOW)["red_open"] == jo + 3 * BAR
    # last allowed: the bar whose close = join close + 60 min (open jo + 60 min); one later → EXPIRED
    k = _k5(jo, 20, red={jo + 12 * BAR})
    assert S.red_entry(k, jo, 60, NOW)["status"] == "RED"
    k = _k5(jo, 20, red={jo + 13 * BAR})
    assert S.red_entry(k, jo, 60, NOW)["status"] == "EXPIRED"
    # the window is not closed yet → PENDING; a gap in closed bars → NODATA
    k = _k5(jo, 5)
    assert S.red_entry(k, jo, 60, jo + 5 * BAR + 1)["status"] == "PENDING"            # bar jo + 5 not closed yet
    assert S.red_entry(k[k.open_time != jo + 2 * BAR], jo, 60, NOW)["status"] == "NODATA"
    assert S.red_entry(None, jo, 60, NOW)["status"] == "NODATA"
    flat = _k5(jo, 20)
    flat.loc[flat.open_time == jo, "c"] = flat.loc[flat.open_time == jo, "o"]     # flat bar = NOT red (engine rule)
    assert S.red_entry(flat, jo, 60, NOW)["status"] == "EXPIRED"


def test_spike_high_and_target_follow_the_engine_convention():
    from services.exec_shadow import net_pct_at
    sp = _ms("2026-10-24 10:45")
    red = sp + 30 * BAR
    k = _k5(sp - 5 * BAR, 50, red={red}, spike=sp + 3 * BAR, spike_h=1.06)
    k.loc[k.open_time == sp - BAR, "h"] = 5.0                    # a high BEFORE the spike bar never counts
    k.loc[k.open_time == red + BAR, "h"] = 9.0                   # nor one after the red bar
    assert S.spike_high(k, sp, red) == 1.06
    E = 1.0 * 1.001
    T = S.target(E, 1.06, 0.00045, maker=0.00018)
    want = net_pct_at("LONG", E, 1.06 * 0.99, 0.045, 0.00018)
    assert abs(T - round(max(1.25, min(8.0, want)), 4)) < 1e-9 and T > 1.25
    assert S.target(E, 1.0, 0.00045, maker=0.00018) == 1.25      # H at the entry → the +1.25 floor of the target
    assert S.target(E, 3.0, 0.00045, maker=0.00018) == 8.0       # capped at +8


def _ref(pair, t, L, gate="FRENZY_WILLY_A_NOMISS_MISSED", mode="paper"):
    r = dict(t=t, pair=pair, L=L, gate=gate, mode=mode, reason=f"read {L} bars after the spike")
    r.update(S.event_bars(_ms(t), L))
    return r


def _scene(t, L, red_after=2, qv=50_000.0, spike_h=1.03):
    """a refusal + 5m klines: 288 bars of history to the spike, the spike high, a red bar `red_after` bars after the join bar."""
    r = _ref("XUSDT", t, L)
    k = _k5(r["spike_open"] - 287 * BAR, 288 + L + 1 + 13, qv=qv, red={r["join_open"] + red_after * BAR} if red_after is not None else (),
            spike=r["spike_open"], spike_h=spike_h)
    return r, k


def test_price_one_tp_stop_unres_unpriced_expired():
    r, k = _scene("2026-10-24T13:05:07", 6)
    te_want = r["join_open"] + 2 * BAR + BAR + S.LATENCY_MS
    seen = {}

    def tp_path(pair, te, E):
        seen.update(te=te, E=E)
        return _P(te, [(3, 0.4, 0.4), (10, 9.0, 0.2)])
    o = S.price_one(r, k, CFG, NOW, tp_path)
    assert o["status"] == "PRICED" and o["exit_why"] == "TP" and abs(o["net_pct"] - o["target"]) < 1e-9
    assert seen["te"] == te_want and abs(seen["E"] - 1.0 * 1.001) < 1e-12        # red close × (1 + 0.10 %) at close + 8 s
    assert o["cls"] == "LOWVOL" and o["bucket"] == "4-12" and o["day"] == "2026-10-24"
    o = S.price_one(r, k, CFG, NOW, lambda p, te, E: _P(te, [(3, 0.5, -4.2)]))
    assert o["status"] == "PRICED" and o["exit_why"] == "STOP" and o["net_pct"] == -4.2
    o = S.price_one(r, k, CFG, NOW, lambda p, te, E: dict(_P(te, [(3, 0.1, 0.1)]), end_ms=te + 20 * MIN))
    assert o["status"] == "UNRES" and o["net_pct"] is None                            # unresolved path → unscored
    o = S.price_one(r, k, CFG, NOW, lambda p, te, E: None)
    assert o["status"] == "UNPRICED"
    r2, k2 = _scene("2026-10-24T13:05:07", 6, red_after=None)
    o = S.price_one(r2, k2, CFG, NOW, tp_path)
    assert o["status"] == "EXPIRED" and o["net_pct"] is None
    # a path_fn that raises → a NODATA row, never an exception
    o = S.price_one(r, k, CFG, NOW, lambda p, te, E: 1 / 0)
    assert o["status"] == "NODATA"


def test_late_tp_and_cap_on_the_v2b_walk():
    r, k = _scene("2026-10-24T13:05:07", 8, spike_h=1.10)
    o = S.price_one(r, k, CFG, NOW, lambda p, te, E: _P(te, [(5, 1.3, 1.3), (35, 0.3, 0.3)]))
    assert o["target"] > 1.3 and o["exit_why"] == "LATE_TP" and abs(o["net_pct"] - 0.3) < 1e-9
    o = S.price_one(r, k, CFG, NOW, lambda p, te, E: _P(te, [(5, 0.1, -0.5), (89, 0.1, -1.0)]))
    assert o["exit_why"] == "CAP"


# ─────────────────────────── gate ───────────────────────────
def _frame(vals, days, bucket="4-12", status="PRICED"):
    d = pd.DataFrame(dict(pair=[f"P{i}USDT" for i in range(len(vals))], bucket=bucket, status=status, net_pct=vals,
                          day=[f"2026-11-{x:02d}" for x in days]))
    d["spike_utc"] = [f"2026-11-{x:02d} 10:{i % 60:02d}:00" for i, x in enumerate(days)]
    d["_spk"] = pd.to_datetime(d.spike_utc)
    return d.sort_values("_spk").reset_index(drop=True)


def test_gate_collecting_candidate_keep_hold():
    v, txt, done = S.gate(_frame([1.0] * 29, [1 + i % 10 for i in range(29)]))
    assert v == "COLLECTING" and not done
    v, _, _ = S.gate(_frame([1.0] * 40, [1 + i % 9 for i in range(40)]))      # 40 priced but only 9 days
    assert v == "COLLECTING"
    days = [1 + i // 3 for i in range(30)]                                     # 10 days × 3
    v, txt, done = S.gate(_frame([1.25, 1.25, -0.5] * 10, days), FAR)
    assert v == "RE-OPEN CANDIDATE" and done and "first 30" in txt
    v, txt, done = S.gate(_frame([1.25, 0.25, -4.0] * 10, days), FAR)
    assert v == "KEEP" and done
    # a positive average whose day-boot lower bound is ≤ 0 → KEEP (both legs needed)
    vals = [1.25] * 27 + [-9.0, -9.0, -9.0]
    v, txt, _ = S.gate(_frame(vals, days), FAR)
    assert np.mean(vals) > 0 and v == "KEEP"
    # a held row inside the prefix holds the freeze; outside the prefix it does not
    f = _frame([1.0] * 31, [1 + i // 3 for i in range(31)])
    f.loc[5, "status"] = "UNPRICED"
    assert S.gate(f, FAR)[0] == "HOLD"
    f = _frame([1.0] * 31, [1 + i // 3 for i in range(31)])
    f.loc[30, "status"] = "PENDING"
    assert S.gate(f, FAR)[0] == "RE-OPEN CANDIDATE"
    # other buckets / expired rows never count toward the 30
    f = pd.concat([_frame([1.0] * 30, days, bucket="13-36"), _frame([1.0] * 5, days[:5], status="EXPIRED")])
    f = f.sort_values("_spk").reset_index(drop=True)
    assert S.gate(f, FAR)[0] == "COLLECTING"
    assert S.gate(pd.DataFrame())[0] in ("COLLECTING", "ERROR")


def test_boot_lb_is_seeded_and_day_clustered():
    p, d = [1.0, 2.0, -1.0, 0.5], ["a", "a", "b", "c"]
    assert S.boot_lb(p, d) == S.boot_lb(p, d)
    assert S.boot_lb([1.0, 2.0], ["a", "a"]) is None


def test_freeze_once(tmp_path):
    st = str(tmp_path / "g.json")
    days = [1 + i // 3 for i in range(30)]
    v, txt, done = S.gate(_frame([1.25, 1.25, -0.5] * 10, days), FAR)
    assert V2.frozen(st, S.GATE_KEY, v, txt, done, True)[0] == "RE-OPEN CANDIDATE"
    v2, txt2, done2 = S.gate(_frame([-4.0] * 30, days), FAR)                         # later data never re-fits a frozen verdict
    fv, ft = V2.frozen(st, S.GATE_KEY, v2, txt2, done2, True)
    assert fv == "RE-OPEN CANDIDATE" and "(frozen " in ft
    assert list(json.load(open(st))) == [S.GATE_KEY]
    st2 = str(tmp_path / "g2.json")
    v, txt, done = S.gate(_frame([1.0] * 3, [1, 2, 3]))
    V2.frozen(st2, S.GATE_KEY, v, txt, done, True)
    assert not os.path.exists(st2)                                              # COLLECTING is never persisted


# ─────────────────────────── run (end to end, offline) ───────────────────────────
def test_run_end_to_end_offline(tmp_path):
    rows = [_row("2026-10-24T13:05:07", "LOWUSDT", 6), _row("2026-10-24T14:05:07", "LIQUSDT", 9, "FRENZY_WILLY_A_NOMISS_LATE"),
            _row("2026-10-24T15:05:07", "LATEUSDT", 28), _row("2026-10-24T16:05:07", "EXPUSDT", 5),
            ["2026-10-24T17:00:00", "WILLY", "JUNKUSDT", "LONG", "FRENZY_WILLY_A_NOMISS_MISSED", ""]]
    dec = _dec(tmp_path, "scalpars_decisions_paper_2026-10-24_20-00-00.csv", rows)
    ref, _, _ = S.read_refusals([dec], None)
    scen = {}
    for r in ref.to_dict("records"):
        qv = 100_000.0 if r["pair"] == "LIQUSDT" else 50_000.0
        red = None if r["pair"] == "EXPUSDT" else 1
        scen[r["pair"]] = _k5(r["spike_open"] - 287 * BAR, 288 + r["L"] + 1 + 13, qv=qv,
                              red={r["join_open"] + red * BAR} if red is not None else (), spike=r["spike_open"], spike_h=1.03)

    def k5(pair, start, end):
        k = scen[pair]
        return k[(k.open_time >= start) & (k.open_time < end)]

    def path(pair, te, E):
        return _P(te, [(4, 9.0, 0.5)]) if pair != "LATEUSDT" else _P(te, [(4, 0.2, -4.5)])
    csv = str(tmp_path / "out.csv")
    st = str(tmp_path / "state.json")
    L = S.run(now_ms=NOW, dec_paths=[dec], since_ms=SINCE, k5_fn=k5, path_fn=path, fetch=False, state_path=st, csv_path=csv,
              orders_paths=[], cfg=CFG)
    txt = "\n".join(L)
    assert L[0].startswith("## 🕰 WILLY LATE-JOIN (low-volume late spikes) — DECISION_LOG 366, observe-only")
    assert "| 4-12 | LOWVOL | 2 | 1 |" in txt and "| 4-12 | LIQUID | 1 | 1 |" in txt and "| 13-36 | LOWVOL | 1 | 1 |" in txt
    assert "COLLECTING" in txt and "L 13–36 (LIGHTUSDT's case was L 28): N 1 · priced 1" in txt
    assert "Could NOT test" in txt and "Comparator" in txt and "EXPUSDT" in txt and "EXPIRED" in txt
    out = pd.read_csv(csv)
    assert list(out.columns) == list(S.CSV_COLS) and len(out) == 4
    assert set(out.status) == {"PRICED", "EXPIRED"} and set(out.cls) == {"LOWVOL", "LIQUID"}
    assert not os.path.exists(st)                                               # nothing frozen while collecting


def test_run_never_raises_on_bad_inputs(tmp_path):
    bad = tmp_path / "scalpars_decisions_paper_2026-10-24_20-00-00.csv"
    bad.write_text("garbage,,\n\x00\x01,,,\n")
    L = S.run(now_ms=NOW, dec_paths=[str(bad), str(tmp_path / "missing.csv")], since_ms=SINCE, fetch=False, state_path=None,
              csv_path=None, orders_paths=[], cfg=CFG)
    assert L[0].startswith("## 🕰 WILLY LATE-JOIN") and any("COLLECTING" in x for x in L)
    L = S.run(now_ms=NOW, dec_paths=[], since_ms=None, fetch=False, state_path=None, csv_path=None)
    assert any("NOT EVALUABLE" in x for x in L)
    # a k5 reader that raises → NODATA rows, never an exception
    dec = _dec(tmp_path, "scalpars_decisions_paper_2026-10-24_21-00-00.csv", [_row("2026-10-24T13:05:07", "XUSDT", 6)])
    L = S.run(now_ms=NOW, dec_paths=[dec], since_ms=SINCE, k5_fn=lambda *a: 1 / 0, path_fn=lambda *a: None, fetch=False,
              state_path=None, csv_path=None, orders_paths=[], cfg=CFG)
    assert any("NODATA" in x for x in L)


def test_fetch_budget_cache_and_no_network(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CACHE", str(tmp_path))
    bud = S._Bud(fetch=False)
    assert S.fetch_5m("XUSDT", 0, 10 * BAR, NOW, bud) is None                   # nothing cached, fetch off → not read (held), no request
    rows = [[i * BAR, "1", "1.1", "0.9", "1.05", "0", 0, "12345.5"] for i in range(10)]
    with open(os.path.join(str(tmp_path), f"XUSDT_0_10.json"), "w") as fh:
        json.dump(rows, fh)
    k = S.fetch_5m("XUSDT", 0, 10 * BAR, NOW, bud)
    assert len(k) == 10 and k.qv.iloc[0] == 12345.5 and bud.n == 0
    bud = S._Bud(fetch=True)
    with pytest.raises(RuntimeError):                                           # the test guard: no request ever leaves
        S._fetch_chunk("YUSDT", 0, 10, NOW, bud)
    bud = S._Bud(fetch=True, max_req=0)
    assert S._fetch_chunk("YUSDT", 0, 10, NOW, bud) is None and "request cap" in bud.stop


def test_read_cfg_defaults_and_values(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"taker_fee": 0.0005, "thresholds": {"frenzy_willy_red_max_wait_minutes": 45, "frenzy_min_volume_usd": 1e7}}))
    assert S.read_cfg(str(p)) == dict(wait=45.0, floor=1e7, taker=0.0005)
    assert S.read_cfg(str(tmp_path / "none.json")) == dict(wait=60.0, floor=20e6, taker=0.00045)
    for w, want in ((0, 0.0), (-5, 60.0), ("", 60.0), (None, 60.0), ("abc", 60.0)):           # 0 = the join bar only, like the engine
        p.write_text(json.dumps({"thresholds": {"frenzy_willy_red_max_wait_minutes": w}}))
        assert S.read_cfg(str(p))["wait"] == want, w
    c = S.read_cfg()
    assert c["wait"] > 0 and c["floor"] > 0 and 0 <= c["taker"] < 0.01


def test_wait_zero_is_the_join_bar_only():
    jo = _ms("2026-10-24 13:00")
    assert S.red_entry(_k5(jo, 5, red={jo}), jo, 0, NOW)["status"] == "RED"
    assert S.red_entry(_k5(jo, 5, red={jo + BAR}), jo, 0, NOW)["status"] == "EXPIRED"


def test_nan_pair_never_leaks(tmp_path):
    dec = _dec(tmp_path, "scalpars_decisions_paper_2026-10-24_20-00-00.csv",
               [[ "2026-10-24T13:05:07", "WILLY", None, "LONG", "FRENZY_WILLY_A_NOMISS_MISSED", "read 6 bars after the spike"],
                _row("2026-10-24T13:05:07", "XUSDT", 6)])
    d, bad, _ = S.read_refusals([dec], None)
    assert list(d.pair) == ["XUSDT"] and bad == 1 and "nan" not in set(d.pair.astype(str))


# ─────────────────────────── timing guard ───────────────────────────
def test_timing_guard_boundary(tmp_path):
    ok = S.price_one(dict(_ref("XUSDT", "2026-10-24T13:07:00", 6), timing_off_s=120.0), *_scene_k("2026-10-24T13:07:00", 6), CFG, NOW,
                     lambda p, te, E: _P(te, [(4, 9.0, 0.5)]))
    assert ok["status"] == "PRICED"                                             # exactly 120 s into the bar → still trusted
    late = S.price_one(dict(_ref("XUSDT", "2026-10-24T13:07:00.001", 6), timing_off_s=120.001), *_scene_k("2026-10-24T13:07:00", 6), CFG,
                       NOW, lambda p, te, E: _P(te, [(4, 9.0, 0.5)]))
    assert late["status"] == "TIMING?" and late["net_pct"] is None and late["cls"] == "LOWVOL"
    # read_refusals stamps the offset from the bar the event floors to
    dec = _dec(tmp_path, "scalpars_decisions_paper_2026-10-24_20-00-00.csv",
               [_row("2026-10-24T13:07:00", "AUSDT", 6), _row("2026-10-24T14:07:00.500", "BUSDT", 6)])
    d, _, _ = S.read_refusals([dec], None)
    assert list(d.timing_off_s) == [120.0, 120.5]
    # out of the gate, counted in its text
    f = _frame([1.0] * 3, [1, 2, 3])
    f.loc[1, "status"] = "TIMING?"
    v, txt, _ = S.gate(f, FAR)
    assert v == "COLLECTING" and "2/30" in txt and "TIMING? 1" in txt


def test_spike_ok_on_the_fetched_bars():
    r, k = _scene("2026-10-24T13:05:07", 6)
    assert S.spike_ok(k, r["spike_open"]) is True
    k.loc[k.open_time == r["spike_open"], "c"] = 0.9
    assert S.spike_ok(k, r["spike_open"]) is False
    assert S.spike_ok(k, r["spike_open"] - 10_000 * BAR) is None and S.spike_ok(None, 0) is None


def _scene_k(t, L):
    return (_scene(t, L)[1],)


# ─────────────────────────── terminal statuses ───────────────────────────
def test_bad_symbol_is_remembered_and_final(monkeypatch):
    monkeypatch.setattr(S, "_NET_BLOCKED", False)
    monkeypatch.setattr(WT, "_NET_BLOCKED", False)            # _http_get is mocked below — never the real network
    monkeypatch.setattr(S, "PACE_S", 0.0)
    import urllib.error
    calls = []

    def boom(url, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, 400, "Invalid symbol", {}, None)
    monkeypatch.setattr(WT, "_http_get", boom)
    bud = S._Bud(fetch=True)
    assert S.fetch_5m("DEADUSDT", 0, 10 * BAR, NOW, bud) == S.BAD and len(calls) == 1
    assert S.fetch_5m("DEADUSDT", 0, 10 * BAR, NOW, S._Bud(fetch=True)) == S.BAD and len(calls) == 1     # never re-requested
    assert json.load(open(S.BAD_SYMBOLS)) == {"DEADUSDT": 400}
    r = _ref("DEADUSDT", "2026-10-24T13:05:07", 6)
    assert S.price_one(r, S.BAD, CFG, NOW, lambda *a: None)["status"] == "NODATA_FINAL"


def test_finalize_after_seven_days():
    r, k = _scene("2026-10-01T13:05:07", 6)
    old_now = r["join_close"] + 60 * MIN + S.FINAL_AGE_MS + S.PATH_MIN * MIN + 3 * BAR
    x = S.price_one(r, k, CFG, old_now, lambda p, te, E: None)
    assert x["status"] == "UNPRICED"
    assert S.finalize(dict(x), r, CFG, old_now, lambda p, te: False)["status"] == "UNPRICED"      # fetch not final → still held
    assert S.finalize(dict(x), r, CFG, old_now, lambda p, te: True)["status"] == "UNPRICED_FINAL"
    assert S.finalize(dict(x), r, CFG, r["join_close"] + 2 * 86_400_000, lambda p, te: True)["status"] == "UNPRICED"   # < 7 days
    u = S.price_one(r, k, CFG, old_now, lambda p, te, E: dict(_P(te, [(3, 0.1, 0.1)]), end_ms=te + 20 * MIN))
    assert S.finalize(dict(u), r, CFG, old_now, lambda p, te: True)["status"] == "UNRES_FINAL"
    gap = S.price_one(r, k[k.open_time != r["join_open"] + BAR], CFG, old_now, lambda *a: None)
    assert gap["status"] == "NODATA" and S.finalize(dict(gap), r, CFG, old_now, lambda p, te: False)["status"] == "NODATA_FINAL"
    unread = S.price_one(r, None, CFG, old_now, lambda *a: None)                 # bars never read (budget) → never final
    assert S.finalize(dict(unread), r, CFG, old_now, lambda p, te: True)["status"] == "NODATA"


def test_final_rows_never_hold_the_gate():
    days = [1 + i // 3 for i in range(31)]
    f = _frame([1.0] * 31, days)
    f.loc[4, "status"] = "UNRES_FINAL"
    v, txt, done = S.gate(f, FAR)
    assert v == "RE-OPEN CANDIDATE" and done and "UNRES_FINAL 1" in txt and "first 30" in txt


# ─────────────────────────── overlap ───────────────────────────
def test_overlap_same_pump_dropped():
    te = _ms("2026-10-24 12:00")
    rows = [dict(pair="XUSDT", status="PRICED", _te=te, _xm=te + 40 * MIN, net_pct=1.0, exit_why="TP"),
            dict(pair="XUSDT", status="PRICED", _te=te + 20 * MIN, _xm=te + 50 * MIN, net_pct=-4.0, exit_why="STOP"),
            dict(pair="XUSDT", status="PRICED", _te=te + 40 * MIN, _xm=te + 60 * MIN, net_pct=0.25, exit_why="LATE_TP"),   # at the exit → kept
            dict(pair="YUSDT", status="PRICED", _te=te + 10 * MIN, _xm=te + 30 * MIN, net_pct=1.0, exit_why="TP")]
    S.mark_overlaps(rows)
    assert [x["status"] for x in rows] == ["PRICED", "OVERLAP", "PRICED", "PRICED"] and rows[1]["net_pct"] is None


def test_exit_ms_matches_the_walk():
    te = _ms("2026-10-24 12:00")
    P = _P(te, [(3, 0.4, 0.4), (10, 9.0, 0.2), (12, 9.0, 0.2)])
    assert S.exit_ms(P, 2.0) == te + 10 * MIN
    assert S.exit_ms(_P(te, [(5, 0.1, -0.5), (89, 0.1, -1.0)]), 2.0) == te + 90 * MIN                         # CAP
    assert S.exit_ms(dict(_P(te, [(3, 0.1, 0.1)]), end_ms=te + 20 * MIN), 2.0) is None                       # UNRES
    assert S.exit_ms(_P(te, [(5, 1.3, 1.3), (35, 0.3, 0.3)]), 2.0) == te + 35 * MIN                           # late TP


# ─────────────────────────── freeze timing ───────────────────────────
def test_freeze_waits_for_the_export_stamp():
    days = [1 + i // 3 for i in range(30)]
    f = _frame([1.25, 1.25, -0.5] * 10, days)
    last = int(f._spk.max().value // 1_000_000)
    v, txt, done = S.gate(f, last + 64 * MIN)
    assert v == "HOLD" and not done and "would read RE-OPEN CANDIDATE" in txt
    assert S.gate(f, None)[0] == "HOLD"
    v, _, done = S.gate(f, last + 65 * MIN)
    assert v == "RE-OPEN CANDIDATE" and done


# ─────────────────────────── budget (mocked Binance) ───────────────────────────
def _hdr(w):
    return {"X-MBX-USED-WEIGHT-1M": str(w)}


def _rows(start, n):
    return [[start + i * BAR, "1", "1.1", "0.9", "1.05", "0", 0, "100"] for i in range(n)]


def test_budget_weight_cooldown_and_bans(monkeypatch):
    monkeypatch.setattr(S, "_NET_BLOCKED", False)
    monkeypatch.setattr(WT, "_NET_BLOCKED", False)            # _http_get is mocked below — never the real network
    monkeypatch.setattr(S, "PACE_S", 0.0)
    calls = []

    def ok(url, timeout):
        calls.append(url)
        return _rows(0, 10), _hdr(950)
    monkeypatch.setattr(WT, "_http_get", ok)
    bud = S._Bud(fetch=True)
    assert len(S._fetch_chunk("AUSDT", 0, 10, NOW, bud)) == 10 and bud.used == 950
    assert S._fetch_chunk("BUSDT", 0, 10, NOW, bud) is None and "used weight 950" in bud.stop and len(calls) == 1
    # cooldown active → stop before any request
    monkeypatch.setattr(NF, "_cooldown_until", lambda: NOW + 60_000)
    bud = S._Bud(fetch=True)
    assert S._fetch_chunk("CUSDT", 0, 10, NOW, bud) is None and "cooldown" in bud.stop and len(calls) == 1
    monkeypatch.setattr(NF, "_cooldown_until", lambda: 0)
    import urllib.error
    armed = []
    monkeypatch.setattr(NF, "_arm_cooldown", lambda code, ra, now: armed.append(code) or 60)
    for code in (418, 429, 403):
        def ban(url, timeout, code=code):
            raise urllib.error.HTTPError(url, code, "ban", {}, None)
        monkeypatch.setattr(WT, "_http_get", ban)
        bud = S._Bud(fetch=True)
        assert S._fetch_chunk(f"D{code}USDT", 0, 10, NOW, bud) is None and bud.stop, code
        assert S._fetch_chunk("EUSDT", 0, 10, NOW, bud) is None and bud.n == 1          # nothing more this run
    assert armed == [418, 429]                                                          # 403 stops without arming the cooldown
    assert not os.path.exists(S.BAD_SYMBOLS)                                            # a ban is never a bad symbol


def test_partial_page_not_cached_and_unclosed_bar_filtered(monkeypatch):
    monkeypatch.setattr(S, "_NET_BLOCKED", False)
    monkeypatch.setattr(WT, "_NET_BLOCKED", False)            # _http_get is mocked below — never the real network
    monkeypatch.setattr(S, "PACE_S", 0.0)
    start = _ms("2026-10-25 23:00")
    monkeypatch.setattr(WT, "_http_get", lambda url, timeout: (_rows(start, 12), _hdr(5)))
    now = start + 10 * BAR + 1                                                          # bars 10, 11 not closed yet
    bud = S._Bud(fetch=True)
    rows = S._fetch_chunk("FUSDT", start, 12, now, bud)
    assert len(rows) == 10 and max(int(x[0]) for x in rows) + BAR <= now
    assert not os.path.exists(os.path.join(S.CACHE, f"FUSDT_{start}_12.json"))         # partial page → not cached
    rows = S._fetch_chunk("FUSDT", start, 12, start + 12 * BAR, S._Bud(fetch=True))
    assert len(rows) == 12 and os.path.exists(os.path.join(S.CACHE, f"FUSDT_{start}_12.json"))


def test_block_stops_the_1m_catchup_and_only_400_is_permanent(monkeypatch):
    """verification fixes: a 403 / 451 stops the run incl. the 1m catch-up; only 400 marks a bad symbol; 5xx / timeout / 408 are transient;
    a blocked 1m attempt never counts as a final give-up."""
    import urllib.error
    monkeypatch.setattr(S, "_NET_BLOCKED", False)
    monkeypatch.setattr(WT, "_NET_BLOCKED", False)
    monkeypatch.setattr(S, "PACE_S", 0.0)
    for code, stop, bad in ((403, True, False), (451, True, False), (500, False, False), (408, False, False), (400, False, True)):
        def boom(url, timeout, _c=code):
            raise urllib.error.HTTPError(url, _c, "x", {}, None)
        monkeypatch.setattr(WT, "_http_get", boom)
        bud = S._Bud(fetch=True)
        r = S.fetch_5m(f"C{code}USDT", 0, 10 * BAR, NOW, bud)
        assert bool(bud.stop) is stop and (r == S.BAD) is bad, code
    monkeypatch.setattr(WT, "load_attempts", lambda: {"done": {WT._kk("AUSDT", 5): "HTTP 403", WT._kk("BUSDT", 5): "ok"}})
    ff = S._default_fetch_final()
    assert ff("AUSDT", 5) is False and ff("BUSDT", 5) is True and ff("CUSDT", 5) is False


def test_overlap_unknown_while_an_earlier_same_pair_refusal_is_unresolved():
    rows = [dict(pair="AUSDT", status="UNRES", _te=1_000, _xm=None, net_pct=None, exit_why=None),
            dict(pair="AUSDT", status="PRICED", _te=2_000, _xm=3_000, net_pct=1.0, exit_why="TP"),
            dict(pair="BUSDT", status="PRICED", _te=2_000, _xm=3_000, net_pct=1.0, exit_why="TP")]
    S.mark_overlaps(rows)
    assert rows[1]["status"] == "OVERLAP?" and "OVERLAP?" in S.HOLDS and rows[2]["status"] == "PRICED"
