"""Route B signal-level loader — TODAY'S FILTER STACK on every signal (research only; DECISION_LOG 635).

A signal's journal `status` is the bot's decision AT THAT MOMENT. A filter armed AFTER the recording began (2026-11-19 23:23 UTC) never
judged the signals before it went live, so without this module a "what would today's bot trade" read counts trades today's stack refuses.
Every such filter is listed in RECHECKS and re-applied — with the ENGINE'S OWN pure functions (one definition) — to the signals that
reached it before `trust_from`; after `trust_from` the engine's own decision stands (it already ran the filter).

Columns added to every signal row:
  today_recheck         the re-checked filter's name ('' when no re-check applies to the row)
  today_recheck_state   'zone' (today's filter refuses) · 'rest' (passes) · 'unread' (the engine's own input was unreadable → it FAILS
                        OPEN, so the row passes, flagged) · 'nofetch' (THIS build could not read the input — Binance stopped / network:
                        unknown, today_filters_pass NA, re-tried next build) · ''
  today_recheck_value   the input(s) read, in words (audit)
  today_block           the filter today's stack would refuse the row with ('' = none)
  today_block_class     block_class, with 'filter' where today_block is set (taken → filter; capacity → filter: the filter sits before
                        the order)
  today_filters_pass    True when the row passed every filter of TODAY's stack (today_block_class taken / capacity); NA on 'nofetch'
  (CSV note: the '' strings read back as NaN from signals.csv.gz — readers use .fillna(""))

The re-checks (each only while its switch is ON in trading_config.json):
  · FLATBASE_FUNDING_ABOVE_BASE (DL 601, committed 2026-11-20 17:27:18 UTC): services.flatbase.funding_above_read(rate, cut) at the ORDER
    decision. Rate = the journal's fb_funding_rate (the engine's premiumIndex read at that decision; stamped since 11-18, before the
    recording began) — a missing stamp means the engine's read failed → 'unread' (fail-open, as the engine); the recorder's rate is shown
    in the audit text only. Reached = the rows that got to the order decision: OPENED, REFUSED:OTHER on the open path (engine L11109),
    or the open path's own "pair held" refusal (same reason text as the pre-check one at L11021, told apart by the funding stamp: the
    pre-check refusal is written before the funding read, so its stamp is empty) — NOT the "one-per-episode check unreadable" refusal
    (L10981), the pre-check PAIR_HELD, nor EXPIRED / SHADOW / SKIPPED_CONFIRM / NOT_FLAT (all before the check).
  · WILLY_HT_BURST_REJECTION (DL 619, committed 2026-11-22 00:09:59 UTC): services.frenzy.pre_vol15_vs60 + cloc15 on the 75 one-minute
    bars opened before the decision minute (Binance 1m klines with BASE volume — the engine's rows; cached per (coin, minute) once final),
    services.watch_filters.ht_burst_state → zone / rest / unread. Reached = OPENED (the engine judges it last, only when no other HT line
    refused). Decision minute = the journal's open time (the engine reads at the decision, ~1 s before the open; the 8 HT fills open
    8–15 s into their minute, so the minute matches — a row within the first 3 s of a minute is flagged in the audit text).
trust_from = the commit time + DEPLOY_MARGIN (AWS auto-deploy + restart). A row between the commit and its deploy is re-checked too: if
the new code already judged it, the re-check reads the same inputs and agrees.

Guard: ENTRY_SNAPSHOT pins every ENTRY setting of the Route B sleeves — switches and cuts under GUARD_PREFIXES (incl. the WILLY watch
blocks and the shared funding guard), minus exit / size / sleeve-master keys (exits are read live by common.exit_specs) — as of DL 635.
The test suite fails when trading_config.json differs: a changed entry setting must get a RECHECKS decision (signals the engine judged
with the old setting are re-judged, or the reason none is needed is stated) before the snapshot moves.
Not re-checked (DL 650, 2026-11-27): EXH_BEAR80 / EXH_EARLY_SPIKE — the Route B signal rows carry neither the market breadth nor the
spike hours, so EXHAUSTION rows TAKEN before the DL 650 deploy in those zones stay counted as taken (the batch had 7 such fills of 210 EXH
signals); their own priced revert gate is scripts/scout_exh_regime_650.py. Re-check them here if the journal ever stamps both inputs.
Not re-checked, by design: DL 629 (commit 2026-11-23 01:39 UTC, WILLY turnover-unreadable TIMING bug fix — a routing delay, not a filter; refused
setups were re-judged on a later red bar and journaled as such). Pure except `fetch_bars_1m` (Binance, through the shared limiter).
"""
import math
import os
import re
import sys
from collections import namedtuple

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT in sys.path or sys.path.insert(0, _ROOT)
from services.flatbase import funding_above_read          # noqa: E402  (DL 601 — the engine's own predicate)
from services.frenzy import PRE_VOL_BARS_BASE, PRE_VOL_BARS_RECENT, cloc15, pre_vol15_vs60   # noqa: E402  (DL 596 / 618 / 619)
from services.watch_filters import ht_burst_state          # noqa: E402  (DL 619)

MIN_MS = 60_000
DEPLOY_MARGIN = pd.Timedelta(minutes=60)
N_BARS = PRE_VOL_BARS_RECENT + PRE_VOL_BARS_BASE           # 75 — the engine's read window

Recheck = namedtuple("Recheck", "name sleeve switch dl committed reached")
RECHECKS = (
    Recheck("FLATBASE_FUNDING_ABOVE_BASE", "FLATBASE_SHORT", "flatbase_funding_above_base_block", 601,
            pd.Timestamp("2026-11-20 17:27:18", tz="UTC"),
            lambda st, det, r: (st == "OPENED" or (st == "REFUSED:OTHER" and not det.startswith("one-per-episode"))
                                or (st == "REFUSED:PAIR_HELD" and _num(r.get("journal_funding_rate")) is not None))),
    Recheck("WILLY_HT_BURST_REJECTION", "FRENZY_WILLY_HT", "frenzy_willy_ht_burst_rejection_enabled", 619,
            pd.Timestamp("2026-11-22 00:09:59", tz="UTC"), lambda st, det, r: st == "OPENED"),
)

# every ENTRY setting of the Route B sleeves (switches AND cuts: filters, universe, routing, the shared funding guard) as of DL 635 — all
# keys under GUARD_PREFIXES minus exit / size / sleeve-master keys (GUARD_EXCLUDE: exits are read live by common.exit_specs). A changed
# entry setting fails the suite (see the module docstring); EXH / DIST cuts are included too — gate_eval re-scores BLOCKED rows with
# today's cuts, but a TAKEN row judged under an older cut is not re-scored.
GUARD_PREFIXES = ('flatbase_', 'exhaustion_', 'distribution_', 'frenzy_willy_', 'watch_block_willy', 'funding_guard_')
GUARD_EXCLUDE = r"(invest_mult|lev_mult|max_hold|_sl_|_tp_|tp_pct|stop_pct|stop_counts_funding|runner|late_tp|high_target|high_k_pct|high_tp_max|max_slots|max_open|size_frac|shares_bot_slots|global_hold)|^(distribution_short_enabled|exhaustion_short_enabled|flatbase_short_enabled|frenzy_willy_enabled|frenzy_willy_ht_enabled)$"
ENTRY_SNAPSHOT = {
    'distribution_max_hours_after_spike': 72.0,
    'distribution_min_vol_usd': 20000000.0,
    'distribution_pos_chg_4h_max': -0.01097,
    'distribution_rsi1h_min': 74.07,
    'distribution_vsw_vol_mult': 5.0,
    'exhaustion_max_hours_after_spike': 72.0,
    'exhaustion_min_vol_usd': 20000000.0,
    'exhaustion_rsi_level': 70.0,
    'exhaustion_rsi_period': 14,
    'exhaustion_bear80_block_enabled': True,
    'exhaustion_bear_pct_max': 80.0,
    'exhaustion_early_spike_block_enabled': True,
    'exhaustion_min_hours_after_spike': 0.5,
    'exhaustion_spot_share_chg_max': -0.0066,
    'exhaustion_ssc_oivol_block_enabled': True,
    'exhaustion_ssc_oivol_oivol_min': 0.128,
    'exhaustion_ssc_oivol_ssc_min': -0.0366,
    'exhaustion_vol_decay_min': 0.6366,
    'exhaustion_wide_enabled': True,
    'exhaustion_wide_min_vol_usd': 5000000.0,
    'exhaustion_wide_spike_vol_mult': 15.0,
    'flatbase_confirm_hours': 24.0,
    'flatbase_flat_max': 0.444,
    'flatbase_funding_above_base_block': True,
    'flatbase_higher_high_block': True,
    'flatbase_max_entry_funding_rate': 5e-05,
    'flatbase_max_hours_after_spike': 96.0,
    'flatbase_min_dist_above_base_pct': 2.09,
    'flatbase_min_hours_after_spike': 4.0,
    'flatbase_min_vol_usd': 20000000.0,
    'flatbase_turnover_min': 1.0,
    'frenzy_willy_a_cold_max_bars': 3,
    'frenzy_willy_a_nomiss': True,
    'frenzy_willy_entry_a': True,
    'frenzy_willy_entry_b': False,
    'frenzy_willy_ht_burst_rejection_enabled': True,
    'frenzy_willy_ht_min_dip_pct': 1.4,
    'frenzy_willy_max_vol_mcap_ratio': 1.0,
    'frenzy_willy_plain_entries_enabled': True,
    'frenzy_willy_red_max_wait_minutes': 60,
    'funding_guard_enabled': True,
    'funding_guard_minutes': 15.0,
    'funding_guard_rate_min_pct': 0.05,
    'watch_block_willy_failed_spike': True,
    'watch_block_willy_gap': True,
    'watch_block_willy_ht_failed_spike': True,
}


def entry_settings(th):
    """{key: value} of every setting ENTRY_SNAPSHOT covers, read from a thresholds dict. Pure."""
    return {k: v for k, v in th.items() if k.startswith(GUARD_PREFIXES) and not re.search(GUARD_EXCLUDE, k)}


def trust_from(rc):
    return rc.committed + DEPLOY_MARGIN


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fb_funding(r, th):
    """(state, value text) of FLATBASE_FUNDING_ABOVE_BASE for one row. Pure."""
    rate, cut = _num(r.get("journal_funding_rate")), th.get("flatbase_max_entry_funding_rate", 0.00005)
    hit = funding_above_read(rate, cut)
    if rate is None:
        p = _num(r.get("rb_funding_pct"))
        return "unread", ("journal funding stamp missing (the engine's read failed → fail-open)"
                          + (f"; recorder {p:+.4f} % (audit only)" if p is not None else ""))
    if hit is None:
        return "unread", f"funding {rate * 100:+.4f} % (journal) · cut unreadable ({cut!r}) → fail-open"
    return ("zone" if hit else "rest"), f"funding {rate * 100:+.4f} % (journal) vs cut {float(cut) * 100:+.4f} %"


def ht_burst(rows_1m, decision_ms):
    """(state, value text) of WILLY_HT_BURST_REJECTION from raw 1m rows [open_ms, o, h, l, c, base_volume, …]. Pure."""
    pv, cl = pre_vol15_vs60(rows_1m, decision_ms), cloc15(rows_1m, decision_ms)
    st = ht_burst_state(pv, cl)
    return st, f"pre_vol {pv if pv is not None else 'n/a'} · cloc15 {cl if cl is not None else 'n/a'}"


def fetch_bars_1m(sym, decision_ms, now_ms, cache_dir):
    """the N_BARS Binance 1m klines opened before the decision minute (raw rows, BASE volume at [5]) — the engine's read. Cached once the
    window is final (all 75 bars back, or the minute ≥ 1 h old). [] = Binance answered with no bars (→ 'unread', as the engine would
    read it); None = THIS build could not read Binance (limiter stopped / network) → 'nofetch' (unknown, re-tried next build)."""
    m0 = int(decision_ms) // MIN_MS * MIN_MS
    path = os.path.join(cache_dir, f"{sym}_{m0}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    import sources as SRC
    if SRC.limiter().stopped:
        return None
    js = SRC.limiter().get("/fapi/v1/klines", dict(symbol=sym, interval="1m", startTime=m0 - N_BARS * MIN_MS, endTime=m0 - 1,
                                                   limit=N_BARS + 5), 1)
    if js is None:
        return None
    rows = [[int(x[0])] + [float(v) for v in x[1:6]] for x in js if int(x[0]) < m0]
    if m0 + 2 * MIN_MS <= now_ms and (len(rows) == N_BARS or m0 + 3600_000 <= now_ms):
        os.makedirs(cache_dir, exist_ok=True)
        pd.to_pickle(rows, path)
    return rows


def apply(S, th, bars_for=None):
    """S + the today_* columns (module docstring). bars_for(sym, decision_ms) → raw 1m rows or None (the build passes fetch_bars_1m).
    Pure apart from bars_for."""
    S = S.copy()
    n = len(S)
    rc_name, rc_state, rc_val, block = [""] * n, [""] * n, [""] * n, [""] * n
    for i, (_, r) in enumerate(S.iterrows()):
        st = str(r.get("status"))
        for rc in RECHECKS:
            if r.get("sleeve") != rc.sleeve or th.get(rc.switch) is not True:
                continue
            t = r.get("signal_time")
            if not rc.reached(st, str(r.get("detail") or ""), r) or not (isinstance(t, pd.Timestamp) and t < trust_from(rc)):
                continue
            if rc.name == "FLATBASE_FUNDING_ABOVE_BASE":
                state, txt = fb_funding(r, th)
            else:
                ms = int(r.signal_time.value // 1_000_000)
                rows = bars_for(r.pair, ms) if bars_for is not None else None
                state, txt = ht_burst(rows, ms) if rows is not None else ("nofetch", "1m bars not fetched this build (Binance stopped / network)")
                if t.second < 3:
                    txt += " · ⚠ open in the first 3 s of its minute: the engine's decision may sit in the minute before"
            rc_name[i], rc_state[i], rc_val[i] = rc.name, state, txt
            if state == "zone":
                block[i] = rc.name
            break
    S["today_recheck"], S["today_recheck_state"], S["today_recheck_value"], S["today_block"] = rc_name, rc_state, rc_val, block
    bc = S["block_class"].astype(str) if "block_class" in S.columns else pd.Series([""] * n, index=S.index)
    S["today_block_class"] = [("filter" if b else c) for b, c in zip(block, bc)]
    S["today_filters_pass"] = pd.array([pd.NA if s_ == "nofetch" else (c in ("taken", "capacity"))
                                        for s_, c in zip(rc_state, S["today_block_class"])], dtype="boolean")
    return S
