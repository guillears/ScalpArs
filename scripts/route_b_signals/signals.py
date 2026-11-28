"""Route B signal-level loader — decision journal → ONE ROW PER SIGNAL (taken, blocked or refused), fill linkage, possibilities.

Mapping (journal event → sleeve · status · signal time · decision price):
  EXHAUSTION  exh_status OPENED / BLOCKED:<exh_block> / REFUSED:<block or OPEN_PATH> (the open path refused it — capacity); t = the 15m signal bar close (exh_signal_at); price = exh_decision_price.
  DISTRIBUTION dist_status OPENED / BLOCKED:<gate> / REFUSED:<block or OPEN_PATH>; t = dist_signal_at; price = dist_decision_price. DIST_NOT_VSW = the pre-screen passed
              but the exchange klines miss the ≥ 5× volume bar → core_rule_met False.
  FLATBASE    final fb_status per episode (pair, fb_spike_at): OPENED · REFUSED (pair held) · SHADOW (turnover ≤ 1, observe-only) →
              REFUSED:SHADOW_TURNOVER · SKIPPED → REFUSED:SKIPPED_<tag> (no confirm close → signal = the S4 bar close, core_rule_met
              False) · NOT_FLAT → BLOCKED:NOT_FLAT (no entry bar: unpriced) · TRACKING only → not a signal yet (counted). The BLOCK counter
              FLATBASE_VOL24_LOW has no journal row → BLOCKED:VOL24_LOW (unpriced, no entry bar).
  WILLY       OPEN rows of FRENZY_WILLY / FRENZY_WILLY_HT (taken) · BLOCKED rows (bl_sleeve, bl_lines → BLOCKED:<lines>; price = the
              decision price) · BLOCK FRENZY_WILLY_PAIR_HELD (sleeve from the last turnover read: R > cap → HT, else plain WILLY) ·
              WILLY RED_EXPIRED / A_NOMISS_MISSED → REFUSED:<x> (armed setup without an entry bar: unpriced) · BLOCK FRENZY_<S>_OFF →
              sleeve FRENZY_<S>, REFUSED:SLEEVE_OFF (unpriced).
  orders      a taken fill (orders export) with no journal row (the journal export ends earlier) → OPENED, source 'orders'.
Not signals (counted in data quality): EXH_WIDE_READ_CAP (shortlist coins not read — no episode known), FLATBASE TRACKING, BOOK / SCAN.
"""
import math
import re

import numpy as np
import pandas as pd

from common import MIN_MS

WILLY_OPEN = ("FRENZY_WILLY", "FRENZY_WILLY_HT")
BASE_COLS = ["sleeve", "pair", "direction", "signal_time", "status", "decision_price", "core_rule_met", "priceable", "journal_atr_pct",
             "journal_tp_pct", "journal_sl_pct", "journal_funding_rate", "journal_model_slip_pct", "journal_t", "detail", "source"]
OPERATOR_CLOSE = re.compile(r"^(EXTERNAL_CLOSE|BACKSTOP_STOP|BACKSTOP|MANUAL)")


def _ts(x):
    try:
        t = pd.Timestamp(x)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except (ValueError, TypeError):
        return pd.NaT


def _s(*xs, default=""):
    """the first non-empty string among xs (NaN-safe)."""
    for x in xs:
        if isinstance(x, str) and x and x.lower() != "nan":
            return x
    return default


def _b(x):
    if isinstance(x, str):
        return {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}.get(x.strip().lower())
    try:
        return None if x != x else bool(x)
    except (TypeError, ValueError):
        return None


def _f(x):
    try:
        v = float(x)
        return v if v == v else math.nan
    except (TypeError, ValueError):
        return math.nan


def _status(status, block, gate):
    """EXH / DIST journal status → OPENED · REFUSED:<block or OPEN_PATH> (the open path refused it: book full, late, open refused …) ·
    BLOCKED:<block or gate> (a rule / filter)."""
    stt = _s(status)
    if stt == "OPENED":
        return "OPENED"
    if stt == "REFUSED":
        b = _s(block)
        return f"REFUSED:{b if b and b != 'REFUSED' else 'OPEN_PATH'}"
    return f"BLOCKED:{_s(block, gate, default='UNKNOWN')}"


def build_signals(J, turnover_cap=1.0):
    """J = deduplicated journal rows → DataFrame of signals (before fill linkage / window cut)."""
    rows = []

    def add(**k):
        k.setdefault("core_rule_met", True)
        k.setdefault("priceable", True)
        k.setdefault("journal_atr_pct", math.nan)
        k.setdefault("journal_tp_pct", math.nan)
        k.setdefault("journal_sl_pct", math.nan)
        k.setdefault("journal_funding_rate", math.nan)
        k.setdefault("journal_model_slip_pct", math.nan)
        k.setdefault("detail", "")
        rows.append(k)

    for r in J[J.e == "EXHAUSTION"].itertuples():
        st = _status(r.exh_status, r.exh_block, r.gate)
        add(sleeve="EXHAUSTION_SHORT", pair=r.pair, direction="SHORT", signal_time=_ts(r.exh_signal_at if isinstance(r.exh_signal_at, str) else r.t),
            status=st, decision_price=_f(r.exh_decision_price) if _f(r.exh_decision_price) == _f(r.exh_decision_price) else _f(r.price),
            journal_atr_pct=_f(r.exh_atr_pct), journal_tp_pct=_f(r.exh_tp_pct), journal_sl_pct=_f(r.exh_sl_pct),
            journal_funding_rate=_f(r.exh_funding_rate), journal_model_slip_pct=_f(r.exh_model_slip_pct), journal_t=str(r.t),
            detail=str(r.exh_reason)[:160], source="journal", j_exh_vol24_usd=_f(r.exh_vol24_usd), j_exh_has_spot=_b(r.exh_has_spot),
            j_exh_vol_decay=_f(r.exh_vol_decay), j_exh_ssc=_f(r.exh_spot_share_chg), j_exh_oi_vol=_f(r.exh_oi_vol),
            j_exh_universe=_s(r.exh_universe))
    for r in J[J.e == "DISTRIBUTION"].itertuples():
        st = _status(r.dist_status, r.dist_block, r.gate)
        add(sleeve="DISTRIBUTION_SHORT", pair=r.pair, direction="SHORT",
            signal_time=_ts(r.dist_signal_at if isinstance(r.dist_signal_at, str) else r.t), status=st,
            decision_price=_f(r.dist_decision_price) if _f(r.dist_decision_price) == _f(r.dist_decision_price) else _f(r.price),
            journal_atr_pct=_f(r.dist_atr_pct), journal_tp_pct=_f(r.dist_tp_pct), journal_sl_pct=_f(r.dist_sl_pct),
            journal_funding_rate=_f(r.dist_funding_rate), core_rule_met=(r.gate != "DIST_NOT_VSW"), journal_t=str(r.t),
            detail=str(r.dist_reason)[:160], source="journal", j_dist_vsw_mult=_f(r.dist_vsw_mult), j_dist_bar_ret_pct=_f(r.dist_bar_ret_pct),
            j_dist_vol24_usd=_f(r.dist_vol24_usd), j_dist_pos_chg_4h=_f(r.dist_pos_chg_4h), j_dist_rsi1h=_f(r.dist_rsi1h))
    # FLATBASE: the final status per episode
    fb = J[J.e == "FLATBASE"].copy()
    if len(fb):
        fb["ts_"] = fb.t.map(_ts)
        fb = fb.sort_values("ts_")
        last = fb.groupby(["pair", "fb_spike_at"], dropna=False).tail(1)
        for r in last.itertuples():
            stt, reason = str(r.fb_status), str(r.fb_reason or "")
            if stt == "TRACKING":
                continue
            core, priceable, t, px = True, True, r.ts_, _f(r.price)
            if stt != "OPENED" and isinstance(r.fb_entry_bar_at, str) and r.fb_entry_bar_at:
                t = _ts(r.fb_entry_bar_at) + pd.Timedelta(minutes=5)     # the decision bar's close, not the journal write time
            if stt == "OPENED":
                st = "OPENED"
            elif stt == "REFUSED":
                st = "REFUSED:PAIR_HELD" if "pair held" in reason else "REFUSED:OTHER"
            elif stt == "SHADOW":
                st = "REFUSED:SHADOW_TURNOVER"
            elif stt == "SKIPPED":
                m_ = re.match(r"^([A-Z][A-Z_]+):", reason)
                tag = m_.group(1) if m_ else "CONFIRM"
                st = f"REFUSED:SKIPPED_{tag}"
                if tag == "CONFIRM":                                  # no confirm close: the would-be entry = the S4 bar close (above)
                    core = False
            elif stt == "NOT_FLAT":
                st, core, priceable = "BLOCKED:NOT_FLAT", False, False
            else:
                st, priceable = f"REFUSED:{stt}", px == px
            add(sleeve="FLATBASE_SHORT", pair=r.pair, direction="SHORT", signal_time=t, status=st, decision_price=px, core_rule_met=core,
                priceable=priceable and px == px, journal_funding_rate=_f(r.fb_funding_rate), journal_t=str(r.t), detail=reason[:160],
                source="journal")
    # BLOCK counters with no journal row
    B = J[J.e == "BLOCK"]
    for r in B[B.gate == "FLATBASE_VOL24_LOW"].itertuples():
        add(sleeve="FLATBASE_SHORT", pair=r.pair, direction="SHORT", signal_time=_ts(r.t), status="BLOCKED:VOL24_LOW", decision_price=math.nan,
            priceable=False, journal_t=str(r.t), detail="BLOCK counter (24 h volume below the floor) — no entry bar journaled", source="block")
    for r in B[B.gate.astype(str).str.match(r"^FRENZY_[A-Z_]+_OFF$")].itertuples():
        sl = "FRENZY_" + r.gate[len("FRENZY_"):-len("_OFF")]
        add(sleeve=sl, pair=r.pair, direction=_s(r.dir, default="LONG"), signal_time=_ts(r.t), status="REFUSED:SLEEVE_OFF", decision_price=math.nan,
            priceable=False, journal_t=str(r.t), detail="sleeve switched off", source="block")
    # WILLY family
    W = J[J.e == "WILLY"].copy()
    W["ts_"] = W.t.map(_ts)
    turn = W[W.gate.isin(["FRENZY_WILLY_TURNOVER", "FRENZY_WILLY_TURNOVER_UNREAD"])]

    def willy_sleeve(pair, t):
        x = turn[(turn.pair == pair) & (turn.ts_ <= t) & (turn.ts_ >= t - pd.Timedelta(hours=2))]
        if not len(x):
            return "FRENZY_WILLY", "no turnover read"
        rr = str(x.iloc[-1].reason or "")
        if rr.startswith("R="):
            try:
                R = float(rr[2:].split(" ")[0])
                return ("FRENZY_WILLY_HT" if R > turnover_cap else "FRENZY_WILLY"), f"R={R:g}"
            except ValueError:
                pass
        return "FRENZY_WILLY", "turnover unread"
    for r in J[(J.e == "OPEN") & J.strategy.isin(WILLY_OPEN)].itertuples():
        add(sleeve=r.strategy, pair=r.pair, direction=str(r.dir), signal_time=_ts(r.t), status="OPENED", decision_price=_f(r.price),
            journal_t=str(r.t), detail="", source="journal")
    _jb = J[(J.e == "BLOCKED") & J.bl_sleeve.astype(str).str.startswith("FRENZY_WILLY")]
    _jb = _jb[~_jb.bl_lines.astype(str).str.split("+").map(lambda ls: any(x.startswith("EXEC_") for x in ls))]   # 🚦 (DL 660) execution refusals are not filter blocks
    for r in _jb.itertuples():
        add(sleeve=str(r.bl_sleeve), pair=r.pair, direction=_s(r.dir, default="LONG"), signal_time=_ts(r.t), status=f"BLOCKED:{r.bl_lines}",
            decision_price=_f(r.price), journal_t=str(r.t), detail=f"sole={r.bl_sole}", source="journal", j_bl_sole=_b(r.bl_sole))
    for r in B[B.gate == "FRENZY_WILLY_PAIR_HELD"].itertuples():
        t = _ts(r.t)
        sl, why = willy_sleeve(r.pair, t)
        add(sleeve=sl, pair=r.pair, direction="LONG", signal_time=t, status="BLOCKED:PAIR_HELD", decision_price=math.nan,
            journal_t=str(r.t), detail=f"sleeve from {why}; decision price not journaled → next-minute open", source="block")
    for r in W[W.gate.isin(["FRENZY_WILLY_RED_EXPIRED", "FRENZY_WILLY_A_NOMISS_MISSED"])].itertuples():
        add(sleeve="FRENZY_WILLY", pair=r.pair, direction="LONG", signal_time=r.ts_, status="REFUSED:" + r.gate[len("FRENZY_WILLY_"):],
            decision_price=math.nan, priceable=False, core_rule_met=False, journal_t=str(r.t), detail=str(r.reason)[:160], source="journal")
    S = pd.DataFrame(rows)
    if not len(S):
        return pd.DataFrame(columns=BASE_COLS)
    S["signal_time"] = pd.to_datetime(S.signal_time, utc=True)
    S = S.dropna(subset=["signal_time"])
    S = S.drop_duplicates(subset=["sleeve", "pair", "direction", "signal_time", "status"]).reset_index(drop=True)
    return S


def link_fills(S, O, tol_min=20):
    """attach the taken fill (orders export) to OPENED signals: same sleeve, pair, direction, opened within [t − 2, t + tol] min.
    Taken fills with no OPENED signal → added as OPENED rows (source 'orders'). → (S, n_added)."""
    O = O.copy()
    if "signal_time" not in S.columns or not len(O):
        S = S.copy()
        for c in ("fill_id", "fill_key", "fill_entry_price", "fill_opened_at", "fill_status", "fill_close_reason", "fill_pnl_pct",
                  "fill_funding_usd", "fill_notional_usd", "fill_closed_at", "fill_operator_closed"):
            S[c] = None
        return S, 0
    O["open_ts"] = pd.to_datetime(O.opened_at.astype(str).str[:19], utc=True)
    for c in ("fill_id", "fill_key", "fill_entry_price", "fill_opened_at", "fill_status", "fill_close_reason", "fill_pnl_pct",
              "fill_funding_usd", "fill_notional_usd", "fill_closed_at", "fill_operator_closed"):
        S[c] = np.nan if c not in ("fill_key", "fill_status", "fill_close_reason", "fill_opened_at", "fill_closed_at", "fill_operator_closed") else None
    used = set()
    for i, r in S[S.status == "OPENED"].iterrows():
        c = O[(O.entry_strategy == r.sleeve) & (O.pair == r.pair) & (O.direction == r.direction) & (O.open_ts >= r.signal_time - pd.Timedelta(minutes=2))
              & (O.open_ts <= r.signal_time + pd.Timedelta(minutes=tol_min)) & ~O.k.isin(used)]
        if len(c):
            f = c.sort_values("open_ts").iloc[0]
            used.add(f.k)
            _set_fill(S, i, f)
    extra = []
    sleeves = set(S.sleeve)
    for f in O[~O.k.isin(used)].itertuples():
        if f.entry_strategy not in sleeves and f.entry_strategy not in ("FRENZY_WILLY", "FRENZY_WILLY_HT", "EXHAUSTION_SHORT",
                                                                         "DISTRIBUTION_SHORT", "FLATBASE_SHORT"):
            continue
        pre = {"EXHAUSTION_SHORT": "entry_exh_", "DISTRIBUTION_SHORT": "entry_dist_"}.get(f.entry_strategy)
        g = (lambda c: _f(getattr(f, pre + c, np.nan))) if pre else (lambda c: math.nan)
        st = f.open_ts.floor("15min") if pre else f.open_ts          # EXH / DIST decide on the 15m bar close before the fill
        extra.append(dict(sleeve=f.entry_strategy, pair=f.pair, direction=f.direction, signal_time=st, status="OPENED",
                          decision_price=g("decision_price") if pre else _f(f.entry_price), core_rule_met=True, priceable=True,
                          journal_atr_pct=g("atr_pct"), journal_tp_pct=g("tp_pct"), journal_sl_pct=g("sl_pct"),
                          journal_funding_rate=_f(getattr(f, "entry_funding_rate", np.nan)), journal_model_slip_pct=g("model_slip_pct"),
                          detail="taken fill with no journal row (journal export ends earlier); stamps from the order row",
                          source="orders", journal_t=None, _k=f.k))
    n_add = 0
    if extra:
        E = pd.DataFrame(extra)
        S = pd.concat([S, E], ignore_index=True)
        for i in range(len(S) - len(E), len(S)):
            _set_fill(S, i, O[O.k == S.at[i, "_k"]].iloc[0])
        S = S.drop(columns=["_k"])
        n_add = len(E)
    return S, n_add


def _set_fill(S, i, f):
    S.at[i, "fill_id"] = f.id
    S.at[i, "fill_key"] = f.k
    S.at[i, "fill_entry_price"] = _f(f.entry_price)
    S.at[i, "fill_opened_at"] = str(f.opened_at)
    S.at[i, "fill_status"] = str(f.status)
    S.at[i, "fill_close_reason"] = str(f.close_reason) if isinstance(f.close_reason, str) else None
    S.at[i, "fill_pnl_pct"] = _f(f.pnl_percentage)
    S.at[i, "fill_funding_usd"] = _f(f.funding_fee_usd)
    S.at[i, "fill_notional_usd"] = _f(f.investment) * _f(f.leverage)
    S.at[i, "fill_closed_at"] = str(f.closed_at) if isinstance(f.closed_at, str) else None
    S.at[i, "fill_operator_closed"] = bool(OPERATOR_CLOSE.match(str(f.close_reason))) if isinstance(f.close_reason, str) else False


UNPRICED_DEDUP_MIN = 60        # dedup window for unpriced rows of a sleeve with no hold (sleeve off, plain WILLY counters …)


def possibilities(S, exit_ms_col="exit_ms", hold_col="hold_min"):
    """one POSSIBILITY per sleeve × coin per holding period. Two separate groups per sleeve × coin, each in time order:
      · PRICED rows (an exit time): a TAKEN row (status OPENED*) always opens its own possibility (never a duplicate of an earlier
        blocked row's hypothetical hold); any other row opens one unless it falls inside the current possibility (→ poss_dup);
      · UNPRICED rows (no exit): deduplicated only among themselves over the sleeve's hold, else UNPRICED_DEDUP_MIN — they never open a
        period that absorbs a priced row.
    Also the 15-min market window id. Pure on the columns it reads."""
    S = S.copy()
    ex_all = S[exit_ms_col] if exit_ms_col in S.columns else pd.Series(np.nan, index=S.index)
    S["grp_"] = np.where(ex_all.notna(), "priced", "unpriced")
    S = S.sort_values(["sleeve", "pair", "grp_", "signal_time"])
    pid, dup = [], []
    cur_key, cur_id, cur_end = None, None, -math.inf
    for r in S.itertuples():
        k = (r.sleeve, r.pair, r.grp_)
        t = r.signal_time.value // 1_000_000
        taken = str(r.status).startswith("OPENED") if hasattr(r, "status") else False
        if k != cur_key or t >= cur_end or (taken and k[2] == "priced"):
            cur_key, cur_id = k, r.signal_id
            ex = getattr(r, exit_ms_col, math.nan)
            hm = getattr(r, hold_col, math.nan)
            if ex == ex and ex is not None:
                cur_end = ex
            else:
                cur_end = t + ((hm if (hm == hm and hm is not None and hm > 0) else UNPRICED_DEDUP_MIN) * MIN_MS)
            pid.append(cur_id)
            dup.append(False)
        else:
            pid.append(cur_id)
            dup.append(True)
    S["poss_id"], S["poss_dup"] = pid, dup
    S = S.drop(columns=["grp_"])
    S["window15"] = S.signal_time.dt.floor("15min").dt.strftime("%Y-%m-%dT%H:%M")
    return S


# ───────────────────────────────────────────────────────── gate audit (engine gates are FAIL-FAST: a row names only the FIRST failing gate)
# live EXH order: NO_SPOT is checked BEFORE the klines read (then NO_DATA / NO_SIGNAL), then VOL24, VOL_DECAY, SPOT_SHARE; the OI read
# behind EXH_SSC_OIVOL happens only after every other gate passed → on an earlier block it is NOT REACHED (structurally never scored)
EXH_FILTERS = ("EXH_NO_SPOT", "EXH_VOL24_LOW", "EXH_VOL_DECAY", "EXH_SPOT_SHARE", "EXH_SSC_OIVOL")
DIST_FILTERS = ("DIST_NOT_VSW", "DIST_VOL24_LOW", "DIST_POS_CHG", "DIST_RSI1H")
NOT_REACHED_IF_EARLIER = {"EXH_SSC_OIVOL"}
CAPACITY = ("PAIR_HELD", "SLEEVE_OFF", "OPEN_REFUSED", "REFUSED", "OPEN_PATH", "LATE", "BOOK_FULL", "READ_CAP")
DATA = ("NO_DATA", "_FAILED", "SPOT_MISMATCH", "_UNREAD", "UNREADABLE")
CORE = ("NO_SIGNAL", "NOT_VSW", "NOT_FLAT", "A_NOMISS_MISSED", "RED_EXPIRED", "SKIPPED_CONFIRM")


def gate_class(status):
    """taken / capacity / data / core_rule / filter for a status string. Pure."""
    st = str(status)
    if st.startswith("OPENED"):
        return "taken"
    gate = st.split(":", 1)[1] if ":" in st else st
    if st.startswith("REFUSED:") and not any(gate.endswith(c) for c in DATA + CORE) and not gate.startswith(("SKIPPED_", "SHADOW")):
        return "capacity"
    if any(gate.endswith(c) for c in CAPACITY):
        return "capacity"
    if any(gate.endswith(c) for c in DATA):         # suffix match: WILLY_FAILED_SPIKE is a filter, EXH_FAILED is data
        return "data"
    if any(gate.endswith(c) for c in CORE):
        return "core_rule"
    return "filter"


def _fails(v, bad):
    """None when the stamp is missing (unevaluated), else bool(bad(v))."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    return bool(bad(v))


def gate_eval(r, th):
    """{filter: True (fails) / False (passes) / None (not evaluable from the journal stamps)} for an EXH / DIST signal row. Pure."""
    g = lambda k, d=None: th.get(k, d)
    if r["sleeve"] == "EXHAUSTION_SHORT":
        floor = float(g("exhaustion_wide_min_vol_usd")) if str(r.get("j_exh_universe")) == "WIDE" else float(g("exhaustion_min_vol_usd"))
        ssc, oiv = r.get("j_exh_ssc"), r.get("j_exh_oi_vol")
        ssc_oi = None
        if g("exhaustion_ssc_oivol_block_enabled", False):
            ssc_oi = (None if (ssc is None or ssc != ssc or oiv is None or oiv != oiv)
                      else bool(ssc > float(g("exhaustion_ssc_oivol_ssc_min")) and oiv > float(g("exhaustion_ssc_oivol_oivol_min"))))
        hs = r.get("j_exh_has_spot")
        return {"EXH_VOL24_LOW": _fails(r.get("j_exh_vol24_usd"), lambda v: v < floor),
                "EXH_NO_SPOT": (None if hs is None or hs != hs else (not hs)),
                "EXH_VOL_DECAY": _fails(r.get("j_exh_vol_decay"), lambda v: v < float(g("exhaustion_vol_decay_min"))),
                "EXH_SPOT_SHARE": _fails(ssc, lambda v: not v < float(g("exhaustion_spot_share_chg_max"))),
                "EXH_SSC_OIVOL": ssc_oi}
    if r["sleeve"] == "DISTRIBUTION_SHORT":
        vsw, ret = r.get("j_dist_vsw_mult"), r.get("j_dist_bar_ret_pct")
        nv = None if (vsw is None or vsw != vsw) else bool(vsw < float(g("distribution_vsw_vol_mult")) or (ret == ret and ret is not None and ret >= 0))
        rmin = float(g("distribution_rsi1h_min", 74.07) if g("distribution_rsi1h_min", 74.07) is not None else 74.07)   # engine default
        return {"DIST_NOT_VSW": nv,
                "DIST_VOL24_LOW": _fails(r.get("j_dist_vol24_usd"), lambda v: v < float(g("distribution_min_vol_usd"))),
                "DIST_POS_CHG": _fails(r.get("j_dist_pos_chg_4h"), lambda v: v > float(g("distribution_pos_chg_4h_max"))),
                "DIST_RSI1H": (_fails(r.get("j_dist_rsi1h"), lambda v: v < rmin) if rmin > 0 else False)}
    return {}


def gate_audit(S, th):
    """adds block_class (taken / capacity / data / core_rule / filter / unpriceable), also_blocked_by (other filters that ALSO fail,
    recomputed from the journal stamps), filters_unevaluated (filters with a missing stamp), filters_not_reached (EXH_SSC_OIVOL on an
    earlier block: the engine reads OI only after every other gate passed), sole_block (a FILTER row whose named gate is a known filter
    that the stamps do not contradict, with no other failing / unevaluated / not-reached filter). Only sole_block rows are a clean
    single-filter counterfactual."""
    S = S.copy()
    cls, also, unev, nreach, sole = [], [], [], [], []
    for _, r in S.iterrows():
        st = str(r["status"])
        gate = st.split(":", 1)[1] if ":" in st else ""
        c = gate_class(st)
        if c == "filter" and not bool(r.get("core_rule_met", True)):
            c = "core_rule"
        if c != "filter":
            cls.append(c); also.append(""); unev.append(""); nreach.append(""); sole.append(False)
            continue
        ev = gate_eval(r, th)
        if ev:
            known = gate in ev
            fails = [k for k, v in ev.items() if v is True and k != gate]
            nr = [k for k, v in ev.items() if v is None and k != gate and k in NOT_REACHED_IF_EARLIER]
            un = [k for k, v in ev.items() if v is None and k != gate and k not in NOT_REACHED_IF_EARLIER]
            also.append("+".join(fails)); unev.append("+".join(un) if known else "named gate not a scored filter"); nreach.append("+".join(nr))
            cls.append("filter" if r.get("priceable", True) else "unpriceable")
            sole.append(bool(known and ev[gate] is not False and not fails and not un and not nr))
        elif r.get("j_bl_sole") is not None and r.get("j_bl_sole") == r.get("j_bl_sole"):      # WILLY watch-line blocks: the engine's own sole flag
            lines = gate.split("+")
            also.append("+".join(lines[1:]) if len(lines) > 1 else ""); unev.append(""); nreach.append("")
            cls.append("filter"); sole.append(bool(r.get("j_bl_sole")) and len(lines) == 1)
        else:
            also.append(""); unev.append("n/a (no stamps for this sleeve)"); nreach.append("")
            cls.append("filter" if r.get("priceable", True) else "unpriceable")
            sole.append(False)
    S["block_class"], S["also_blocked_by"], S["filters_unevaluated"], S["filters_not_reached"], S["sole_block"] = cls, also, unev, nreach, sole
    return S


def pair_held_links(S):
    """a PAIR_HELD row is capacity, not a new possibility: it joins the possibility of the position that held the coin (any sleeve: the
    taken signal on that pair whose [signal, exit] covers it) → poss_dup True, poss_id = that signal (or its own id when none is found)."""
    S = S.copy()
    taken = S[S.status.astype(str).str.startswith("OPENED")]
    for i, r in S[S.status.astype(str).str.endswith("PAIR_HELD")].iterrows():
        t = r.signal_time.value // 1_000_000
        c = taken[(taken.pair == r.pair) & (taken.signal_time <= r.signal_time)]
        hit = None
        for _, x in c.sort_values("signal_time", ascending=False).iterrows():
            end = x.get("out_exit_ms")
            if end is None or end != end or t <= end:
                hit = x.signal_id
                break
        S.at[i, "poss_dup"] = True
        S.at[i, "poss_id"] = hit or r.signal_id
        S.at[i, "poss_note"] = "pair held by " + (hit.split("|")[0] if hit else "an unknown position")
    return S
