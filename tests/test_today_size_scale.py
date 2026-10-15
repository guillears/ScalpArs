"""📏 TODAY's cell sizing — ONE rule, scripts/build_master_pool.today_size_rule (master STACK 2026-10-06c, DECISION_LOG 233), used by
the master's stack_pnl AND scripts/screen_pool.pnl_current (SCREENED_BASELINE `pnl_current_sizing`). Engine order: FLIP / NONEXP_CALM3D /
momentum SHORT (C1, W2+W1) → 1× · UNMATCHED long: crowd-sprint → 1×, else pair-vol ≥ 0.90 → 1×, else min(cell, 1.5×) (206).
History: v19 (2026-10-06) found the screen still pricing UNMATCHED longs at 2× and W2+W1 shorts at 2×; 10-06c then found the master
pricing the same 3+1 W2+W1 shorts at 2× and the 5 PVR ≥ 0.90 longs at 1.5× — two copies of one rule drift, so there is one copy now."""
import csv, json, os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from scripts import build_master_pool as B
from scripts.build_master_pool import today_size_rule, today_size_scale

MOM, FLIP = "MOMENTUM", "FLIP:FAN_RATIO_GATE"


@pytest.mark.parametrize("args,want,tag", [
    ((MOM, "SHORT", "C1", 2.0), 0.5, "SHORT_1X"),                              # C1 short de-mux (Jun-29)
    ((MOM, "SHORT", "C1+C6", 1.5), 1 / 1.5, "SHORT_1X"),                       # any multiplied momentum short → 1×
    ((MOM, "SHORT", "W1+W2+W1", 2.0), 0.5, "SHORT_1X"),                        # W2+W1 short de-muxed 2026-07-30
    ((MOM, "SHORT", "W1+W2+W1", 2.0, None, None, None, 1.5), 1 / 3, "SHORT_1X"),  # SOL 06-18: 2× at 30× lev → 1× at 20×
    ((MOM, "SHORT", "W1", 1.0, None, None, None, 1.5), 1 / 1.5, "SHORT_1X"),      # a lev-only boost is stripped too
    ((MOM, "LONG", "UNMATCHED", 1.0, 0.50, 0.01, 0.50, 1.5), 1 / 1.5 * 0.05, "UNMATCHED_INV+ML_LEV"),
    ((MOM, "LONG", "ADXMAX_PROBE", 0.5, None, None, None, 0.05), 1.0, "ML_LEV"),       # probes never re-priced
    ((MOM, "SHORT", "UNMATCHED", 2.0), 0.5, "SHORT_1X"),                       # not the 1.5× long rule
    ((MOM, "SHORT", "W1", 1.0), 1.0, ""),
    ((MOM, "LONG", "NONEXP_CALM3D", 2.0), 0.5 * 0.05, "CALM3D_1X+ML_LEV"),                   # DECISION_LOG 225
    ((MOM, "LONG", "NONEXP_CALM3D", 1.0), 1.0 * 0.05, "ML_LEV"),
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE[NEGDI15]×2", 2.0), 0.25, "FLIP_FAN_LEV"),  # 220 size → 1× · 252 FAN lev 20× → 10×
    (("nan", "SHORT", "FLIP:FAN_RATIO_GATE×2", 2.0), 0.25, "FLIP_FAN_LEV"),          # flip known by its source alone
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0), 0.5, "FLIP_FAN_LEV"),              # a 1× FAN fill at 20× → 10× (252)
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0, None, None, None, 1.0, 10), 1.0, "FLIP_FAN_LEV"),   # already bracket-capped at 10× → unchanged
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0, None, None, None, 1.0, 20), 0.5, "FLIP_FAN_LEV"),
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0, None, None, None, 0.5, 10), 1.0, "FLIP_FAN_LEV"),   # post-252 fill stamps 0.5, traded 10× → unchanged (B20 MUBARAK; was halved)
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0, None, None, None, 0.5, None), 1.0, "FLIP_FAN_LEV"), # post-252 stamp, no traded lev → no re-scale
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0, None, None, None, 1.0, 15), 10 / 15, "FLIP_FAN_LEV"),  # schedule-capped 15× → 10×
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE+FGP_QS", 0.5, None, None, None, 0.05, 1), 1.0, "FLIP_FAN_LEV"),  # probe 1× stays 1× (0.05 floor)
    (("FLIP:PAIR_RSI_OB", "SHORT", "FLIP:PAIR_RSI_OB×2", 2.0), 0.5, "FLIP_1X"),      # other flip sources: 220 rule unchanged
    (("FLIP:PAIR_RSI_OB", "SHORT", "FLIP:PAIR_RSI_OB", 1.0), 1.0, ""),
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.50, 0.01, 0.50), 0.75 * 0.05, "UNMATCHED_INV+ML_LEV"),  # DECISION_LOG 206 — 2× → 1.5×
    ((MOM, "LONG", "UNMATCHED", 2.5, 0.50, 0.01, 0.50), 0.6 * 0.05, "UNMATCHED_INV+ML_LEV"),   # old 2.5× quiet boost → 1.5×
    ((MOM, "LONG", "UNMATCHED", 1.5, 0.50, 0.01, 0.50), 1.0 * 0.05, "UNMATCHED_INV+ML_LEV"),   # already at today's size
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.80, 0.08, 0.50), 0.5 * 0.05, "SPRINT_DEMUX+ML_LEV"),    # crowd-sprint → 1×
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.80, 0.08, 1.20), 0.5 * 0.05, "SPRINT_DEMUX+ML_LEV"),    # sprint first (engine order)
    ((MOM, "LONG", "UNMATCHED", 1.5, 0.80, 0.08, 0.50), 1 / 1.5 * 0.05, "SPRINT_DEMUX+ML_LEV"),
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.74, 0.08, 0.50), 0.75 * 0.05, "UNMATCHED_INV+ML_LEV"),  # strict > on global vol
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.80, 0.07, 0.50), 0.75 * 0.05, "UNMATCHED_INV+ML_LEV"),  # strict > on BTC slope
    ((MOM, "LONG", "UNMATCHED", 2.0, None, 0.08, 0.50), 0.75 * 0.05, "UNMATCHED_INV+ML_LEV"),  # missing stamp = fail-open
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.50, 0.01, 0.90), 0.5 * 0.05, "PVR_DEMUX+ML_LEV"),       # crowded entry (≥)
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.50, 0.01, float("nan")), 0.75 * 0.05, "UNMATCHED_INV+ML_LEV"),
    ((MOM, "LONG", "UNMATCHED", 1.0, 0.80, 0.08, 1.20), 1.0 * 0.05, "ML_LEV"),               # a 1× fill is never re-priced
    (("SPIKE_FADE", "SHORT", "SPIKE_FADE", 2.0), 1.0, ""),                      # fades: ticket scale, not a cell
    (("FRENZY_LONG", "LONG", "FRENZY_LONG", 2.0), 1.0, ""),                     # sleeves: re-priced by _today
    (("BEARRUN_SHORT", "SHORT", "", 2.0), 1.0, ""),
    # ⚖️ 10-12d (DECISION_LOG 283) → 10-14a (289): momentum_long_lev_mult 0.05 — a 20× momentum long → 1×, never double-counted on a fill already lower
    ((MOM, "LONG", "UNMATCHED", 1.0, None, None, None, 1.0, 20), 0.05, "ML_LEV"),
    ((MOM, "LONG", "UNMATCHED", 1.0, None, None, None, 1.0, 10), 0.1, "ML_LEV"),          # traded at 10× (bracket cap or the 283 factor) → 1× (289)
    ((MOM, "LONG", "UNMATCHED", 1.0, None, None, None, 1.0, 15), 1 / 15, "ML_LEV"),       # schedule-capped 15× → 1×
    ((MOM, "LONG", "UNMATCHED", 1.0, 0.5, 0.01, 0.5, 1.5, 30), 1 / 30, "UNMATCHED_INV+ML_LEV"),  # 30× (cell lev 1.5) → 20× → 1×
    ((MOM, "LONG", "UNMATCHED", 1.0, 0.5, 0.01, 0.5, 1.5, 15), 1 / 15, "UNMATCHED_INV+ML_LEV"),  # traded under the 283 factor (20 × 1.5 × 0.5) → 10× base → 1×
    ((MOM, "LONG", "GAPMIN_PROBE", 0.5, None, None, None, 0.05, 1), 1.0, "ML_LEV"),       # a 1× probe stays 1×
    (("", "LONG", "UNMATCHED", 1.0), 0.05, "ML_LEV"),                                     # blank strategy = momentum (old exports)
    (("MOMENTUM", "SHORT", "UNMATCHED", 1.0, None, None, None, 1.0, 20), 1.0, ""),          # shorts never
    (("FRENZY_LITE", "LONG", "FRENZY_LITE", 1.0, None, None, None, 0.4, 8), 1.0, ""),      # sleeves never (re-priced by _today)
    (("FLIP:FAN_RATIO_GATE", "LONG", "FLIP:FAN_RATIO_GATE", 1.0, None, None, None, 1.0, 20), 0.5, "FLIP_FAN_LEV"),   # flips: their own rule only
    (("BULLRUN_LONG", "LONG", "BULLRUN", 1.0, None, None, None, 1.0, 20), 1.0, ""),
])

def test_today_size_rule(args, want, tag):
    f, t = today_size_rule(*args)
    assert f == pytest.approx(want) and t == tag
    assert today_size_scale(*args) == pytest.approx(want)


def test_momentum_long_lev_frozen_matches_live_json():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))
    th = th.get("thresholds", th)
    # DECISION_LOG 298: the master prices the NOTIONAL — the live notional factor = lev mult × invest mult (0.25 × 0.2 = DL 289's 0.05)
    assert th["momentum_long_lev_mult"] * th.get("momentum_long_invest_mult", 1.0) == pytest.approx(B.MOMENTUM_LONG_LEV_FROZEN) and B.MOMENTUM_LONG_LEV_FROZEN == 0.05
    assert B.momentum_long_lev_scale(1.0, 1.0, 20) == pytest.approx(0.05) and B.momentum_long_lev_scale(1.0, 1.0, 20, mll=1.0) == 1.0
    assert B.momentum_long_lev_scale(2.0, 2.0, 40) == pytest.approx(2 / 40)          # a 2.0-lev cell → 2× (round(20 × 2 × 0.05))
    assert B.momentum_long_lev_scale(0.05, 0.05, 1) == 1.0                           # a 1× probe stays 1×


def test_frozen_constants_match_live_json():
    """Frozen with STACK_VERSION — must equal the live config at ship time (a retune = new STACK_VERSION + these values)."""
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))
    th = th.get("thresholds", th)
    assert th["long_unmatched_sprint_demux_gvr_min"] == B.UNMATCHED_SPRINT_GVR_MIN_FROZEN
    assert th["long_unmatched_sprint_demux_b20slope_min"] == B.UNMATCHED_SPRINT_B20SLOPE_MIN_FROZEN
    assert th["long_unmatched_mult_pvr_max"] == B.UNMATCHED_PVR_MAX_FROZEN
    assert th["long_unmatched_demux_inv_mult"] == 1.0                           # the PVR de-mux target is 1×
    assert th["long_unmatched_quiet_mult"] <= B.UNMATCHED_LONG_INV_FROZEN       # the quiet boost never sizes above the cell
    rules = th["pattern_cell_rules"]
    um = [r for r in rules if r.get("pattern") == "UNMATCHED" and r.get("direction") == "LONG"][0]
    assert float(um["inv_mult"]) == B.UNMATCHED_LONG_INV_FROZEN and float(um["lev_mult"]) == B.UNMATCHED_LONG_LEV_FROZEN
    assert th["long_unmatched_demux_lev_mult"] == 1.0 and th["long_unmatched_quiet_lev_mult"] <= B.UNMATCHED_LONG_LEV_FROZEN
    assert th["nonexp_calm3d_lev_mult"] <= 1.0
    assert th["flip_short_negdi_lev_mult"] <= 1.0 and th["flip_short_tg_shallow_lev_mult"] <= 1.0
    hot = [r["pattern"] for r in rules if r.get("direction") == "SHORT"
           and max(float(r.get("inv_mult") or 1.0), float(r.get("lev_mult") or 1.0)) > 1.0]
    assert not hot, f"a SHORT cell sizes above 1× again ({hot}) — today_size_rule de-muxes every momentum short; re-scope it"
    assert th["nonexp_calm3d_invest_mult"] <= 1.0
    assert th["flip_short_negdi_mult"] <= 1.0 and th["flip_short_tg_shallow_mult"] <= 1.0


def test_screen_uses_the_shared_rule():
    import inspect
    from scripts import screen_pool
    assert screen_pool.today_size_scale is today_size_scale
    body = inspect.getsource(screen_pool.pnl_current)
    assert "today_size_scale(" in body and "cell_lev_multiplier" in body and "getattr(" not in body


def _rows(name):
    p = os.path.join(ROOT, "reports", name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not built")
    return list(csv.DictReader(open(p, encoding="utf-8")))


def test_frozen_baseline_is_at_todays_sizing():
    """A sizing change without a screen_pool.py re-run leaves the frozen column stale."""
    from scripts.screen_pool import pnl_current
    stale = [(r["pair"], r["opened_at"]) for r in _rows("SCREENED_BASELINE.csv")
             if abs(float(r["pnl_current_sizing"]) - pnl_current(r)) > 0.01]
    assert not stale, f"SCREENED_BASELINE.csv not re-frozen at today's sizing: {stale[:5]}"


def test_screen_and_master_agree_on_kept_rows():
    """Every screened row the master keeps carries the same today's-sizing $ (±1 cent of rounding)."""
    m = {(r["opened_at"][:19], r["pair"], r["direction"]): r for r in _rows("MASTER_POOL_stacked.csv")}
    assert next(iter(m.values())).get("stack_version") == B.STACK_VERSION, "master not rebuilt after a STACK_VERSION bump — run scripts/build_master_pool.py"
    diff = []
    for r in _rows("SCREENED_BASELINE.csv"):
        mm = m.get((r["opened_at"][:19], r["pair"], r["direction"]))
        assert mm is not None, f"screened row missing from the master: {r['pair']} {r['opened_at']}"
        if mm["stack_keep"] == "True" and abs(float(mm["stack_pnl"]) - float(r["pnl_current_sizing"])) > 0.011:
            diff.append((r["pair"], r["opened_at"], r["pnl_current_sizing"], mm["stack_pnl"]))
    assert not diff, f"screen vs master sizing drift: {diff[:5]}"



@pytest.mark.parametrize("cl", [1.0, 0.5, 1.5, 2.0, 0.05])
def test_dl298_post_fill_is_not_rescaled(monkeypatch, cl):
    """⚖️ DECISION_LOG 298: a post-298 momentum long trades 5× at 0.2 margin — the SAME notional as its paired leverage at the full margin.
    With the stamp (or, for an unstamped row opened after the 298 commit, the fallback) the size rule prices it EXACTLY like a pre-298 fill
    that traded at the paired leverage — the 298 step adds no factor; a pre-298 fill is unchanged."""
    ref = max(1, int(round(20 * cl * 0.05)))
    traded = max(1, int(round(20 * cl * 0.25)))
    row = dict(entry_strategy="MOMENTUM", direction="LONG", leverage=traded, cell_lev_multiplier=cl, opened_at="2026-10-16T10:00:00",
               momentum_long_ref_leverage=ref)
    assert B.momentum_long_traded_lev(row) == ref
    assert B.momentum_long_lev_scale(cl, cl, B.momentum_long_traded_lev(row)) == pytest.approx(1.0)        # the ML_LEV part = 1.0
    pre = B.today_size_rule("MOMENTUM", "LONG", "UNMATCHED", 1.0, lev=cl, traded_lev=ref)
    post = B.today_size_rule("MOMENTUM", "LONG", "UNMATCHED", 1.0, lev=cl, traded_lev=B.momentum_long_traded_lev(row))
    assert post == pre
    # unstamped row opened after the 298 commit → the fallback gives the same leverage
    monkeypatch.setattr(B, "_DL298_FLOOR", [__import__("pandas").Timestamp("2026-10-15 20:00:00")])
    row2 = dict(row); row2.pop("momentum_long_ref_leverage")
    assert B.momentum_long_traded_lev(row2) == pytest.approx(ref)
    # a pre-298 fill (opened before the commit) and every non-momentum-long row: the as-traded leverage (today's behaviour)
    row3 = dict(row2, opened_at="2026-10-14T10:00:00", leverage=1)
    assert B.momentum_long_traded_lev(row3) == 1
    assert B.momentum_long_traded_lev(dict(row2, direction="SHORT")) == traded and B.momentum_long_traded_lev(dict(row2, entry_strategy="FRENZY_LONG")) == traded


def test_dl298_builders_read_the_margin_scaled_leverage():
    src_b = open(os.path.join(ROOT, "scripts", "build_master_pool.py"), encoding="utf-8").read()
    src_s = open(os.path.join(ROOT, "scripts", "screen_pool.py"), encoding="utf-8").read()
    assert 'r.get("cell_lev_multiplier"), momentum_long_traded_lev(r))' in src_b
    assert "r.get('cell_lev_multiplier'), momentum_long_traded_lev(r))" in src_s
