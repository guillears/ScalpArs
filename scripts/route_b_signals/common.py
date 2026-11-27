"""Route B signal-level loader — shared constants, paths and TODAY's exits read from the live trading_config.json (research only)."""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "reports", "ROUTE_B_SIGNALS_data")          # gitignored: caches + datasets
REPORTS = os.path.join(ROOT, "reports")
REC_START = pd.Timestamp("2026-11-19 23:23:30", tz="UTC")           # Route B recording began (DL 597)
MIN_MS = 60_000

# ── booking costs (% of notional) — the LIVE fee rates from trading_config.json (taker_fee / maker_fee), the same ones that place the HT
#    net levels: taker legs pay the fee + modelled slip; a resting maker TP pays the maker fee only
def _fees():
    try:
        with open(os.path.join(ROOT, "trading_config.json")) as f:
            c = json.load(f)
        return float(c.get("taker_fee", 0.00045)) * 100, float(c.get("maker_fee", 0.00018)) * 100
    except (OSError, ValueError):
        return 0.045, 0.018


TAKER_PCT, MAKER_PCT = _fees()
TP_THROUGH_PCT = 0.05          # a resting maker TP fills only when price trades this % THROUGH the level (DL 602's trade-through rule)
NOTIONAL_USD = 6000.0          # modelled-cost size (≈ today's ticket)
DEFAULT_SLIP_PCT = 0.10        # per taker leg when no recorded book cost exists (flagged slip_src = 'default')
BOOK_MAX_AGE_MIN = 2           # a book row older than this before the state minute is not used
LATENCY_S = 1                  # the entry is the open of the first minute starting ≥ signal time + this. The signal time of a blocked /
                               # refused EXH / DIST row is the 15m bar CLOSE → its entry = bar close + 60 s (the bot's real fills land
                               # ~30–40 s after the close, so blocked rows enter ~20–30 s later than a real fill would). H1–H6: signal =
                               # the END of trigger minute m → entry = the open of minute m + 2.
STOP_FILL = "worse_of_level_and_close"   # non-gapped stop fill (see exits.walk) — pessimistic on purpose (review 7)

SLEEVES_PRICED = ("FLATBASE_SHORT", "EXHAUSTION_SHORT", "DISTRIBUTION_SHORT", "FRENZY_WILLY_HT")


def load_thresholds(path=None):
    path = path or os.path.join(ROOT, "trading_config.json")
    with open(path) as f:
        c = json.load(f)
    th = dict(c.get("thresholds") or {})
    th["_taker_fee"] = float(c.get("taker_fee", 0.00045))
    th["_maker_fee"] = float(c.get("maker_fee", 0.00018))
    return th


def exit_specs(th):
    """{sleeve: spec | None}. Spec kinds: 'price' (fixed price % levels), 'atr' (min(k × ATR, cap) price % levels), 'net' (net-% levels on
    the engine's taker-net P&L). None = NOT PRICED (reason in 'why')."""
    g = lambda k, d=None: th.get(k, d)
    tk, mk = th["_taker_fee"] * 100, th["_maker_fee"] * 100
    out = {
        "FLATBASE_SHORT": dict(kind="price", side=-1, tp=float(g("flatbase_tp_pct")), sl=float(g("flatbase_stop_pct")),
                               hold_min=int(round(float(g("flatbase_max_hold_hours")) * 60)),
                               stop_counts_funding=bool(g("flatbase_stop_counts_funding", False)),
                               tp_counts_funding=bool(g("flatbase_tp_counts_funding", False))),   # 💸 (DECISION_LOG 653)
        "EXHAUSTION_SHORT": dict(kind="atr", side=-1, tp_atr=float(g("exhaustion_tp_atr_mult")), tp_cap=float(g("exhaustion_tp_cap_pct")),
                                 sl_atr=float(g("exhaustion_sl_atr_mult")), sl_cap=float(g("exhaustion_sl_cap_pct")),
                                 hold_min=int(round(float(g("exhaustion_max_hold_hours")) * 60)), stop_counts_funding=False),
        "DISTRIBUTION_SHORT": dict(kind="atr", side=-1, tp_atr=float(g("distribution_tp_atr_mult")), tp_cap=float(g("distribution_tp_cap_pct")),
                                   sl_atr=float(g("distribution_sl_atr_mult")), sl_cap=float(g("distribution_sl_cap_pct")),
                                   hold_min=int(round(float(g("distribution_max_hold_hours")) * 60)), stop_counts_funding=False),
    }
    ht_tp = float(g("frenzy_willy_ht_tp_pct", 0) or 0)
    if g("frenzy_willy_ht_late_tp_enabled", True) is False and g("frenzy_willy_ht_high_target_enabled", True) is False and ht_tp > 0:
        hm = float(g("frenzy_willy_ht_max_hold_minutes", 0) or 0) or float(g("frenzy_willy_max_hold_minutes"))
        # engine P&L is taker-net: TP books at the maker limit where net = +tp (entry taker + exit maker); stop when net ≤ −stop (both taker)
        out["FRENZY_WILLY_HT"] = dict(kind="net", side=1, tp_net=ht_tp, sl_net=float(g("frenzy_willy_stop_pct")), hold_min=int(round(hm)),
                                      tp_gross=ht_tp + tk + mk, sl_gross=float(g("frenzy_willy_stop_pct")) - 2 * tk, stop_counts_funding=False)
    else:
        out["FRENZY_WILLY_HT"] = None
    out["FRENZY_WILLY"] = None     # V2b: spike-high target + late TP + runner trail (needs the spike high / 1h EMA gap) → NOT PRICED
    return out


NOT_PRICED_WHY = {
    "FRENZY_WILLY": "plain WILLY V2b exit (spike-high target, late TP from min 30, runner trail on falling coins) needs inputs not journaled",
    "FRENZY_WIDE": "sleeve OFF — no live exit to price",
}


def levels_for(spec, atr_pct=None):
    """(tp_pct, sl_pct) PRICE distances from the entry for a spec (+ the ATR for 'atr'); (None, None) when undefined."""
    if spec is None:
        return None, None
    if spec["kind"] == "price":
        return spec["tp"], spec["sl"]
    if spec["kind"] == "net":
        return spec["tp_gross"], spec["sl_gross"]
    try:
        a = float(atr_pct)
    except (TypeError, ValueError):
        return None, None
    if not a > 0:
        return None, None
    return min(spec["tp_atr"] * a, spec["tp_cap"]), min(spec["sl_atr"] * a, spec["sl_cap"])


def spec_hash(spec):
    """a short stable hash of one sleeve's exit spec + the booking costs + the stop-fill rule (carry-over invalidation)."""
    import hashlib
    blob = json.dumps(dict(spec=spec, taker=TAKER_PCT, maker=MAKER_PCT, through=TP_THROUGH_PCT, slip=DEFAULT_SLIP_PCT, stop=STOP_FILL,
                           lat=LATENCY_S), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def ensure_dirs():
    for d in (DATA, os.path.join(DATA, "s3"), os.path.join(DATA, "klines"), os.path.join(DATA, "funding"), os.path.join(DATA, "minutes")):
        os.makedirs(d, exist_ok=True)
