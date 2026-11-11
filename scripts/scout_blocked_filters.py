#!/usr/bin/env python3
"""🧱💲 Scout — BLOCKED-TRADE PRICER + per-line REVERT GATES for the ARMED watch-line filters (DECISION_LOG 529; the filters: DL 527).
OBSERVE / ALERT only — never changes config, never calls the bot API; writes only reports/.

WHAT     Every entry an armed filter refused is a decisions-CSV e=BLOCKED row (blocked_signals: pair, sleeve, refusal time, decision price,
         the refusing lines, sole flag, the entry_* snapshot). This scout prices each one AS IF IT HAD OPENED, on TODAY's exit of its sleeve,
         and compares the blocked trades with the trades the same sleeve really kept over the same period.
RULER    entry = the refusal + 8 s at the decision price moved 0.10 % against the trade (the live latency / slip ruler) · taker entry fee ·
         the sleeve's live exit read from trading_config.json each run:
           FRENZY_LITE / FRENZY_LITE_RETRY   TP frenzy_lite_tp_pct / stop frenzy_stop_pct / cap frenzy_max_hold_minutes
           FRENZY_LONG                       TP frenzy_tp_pct / stop frenzy_long_stop_pct (or frenzy_stop_pct) / the same cap
           FRENZY_WILLY / FRENZY_WILLY_HT    the V2b exit: target from the spike high (the engine's frenzy_willy_high_target), stop
                                             frenzy_willy_stop_pct, late TP, cap frenzy_willy_max_hold_minutes
             (all FRENZY: the validated net-path walker scout_willy_exit_v2.walk_v2 on scout_frenzy_willy_exit.load_path — the walker the
              v2b shadow's parity check uses against live fills)
           SPIKE_FADE                        today's live fade exit from the ENGINE's pure pieces (fade_base_shadow_seed for the stop /
                                             ladder, short_runner_arm + next_fade_late_peak for the normal AND the late arm, short_trail_floor)
                                             on 1m bars from the entry minute (each bar: high, low, close — adverse first); a stop / a floor
                                             from an earlier bar books its LINE (the bar open on a gap), as live polling fills ≈ at the line;
                                             a floor raised by a peak inside the same bar books the reading (1m cannot see the in-bar order);
                                             the order's ATR from the snapshot; cap max_holding_time_minutes (investment block)
           MOMENTUM_LONG                     the validated ML exit replica (scripts/ml_exit_optimize_yr5.py RP.BASE via
                                             scout_ml_matched_refusals.price_replica — DL 380 parity)
         A path not complete yet → PENDING (re-priced next run); no path ≥ 1 day after the horizon → NOT_EVALUABLE (listed, never counted).
         Final prices are cached (reports/SCOUT_BLOCKED_FILTERS.csv) WITH the exit era they were priced on: when a sleeve's exit changes,
         its stored prices are re-priced on the new exit (never two exit eras in one gate prefix — CLAUDE.md entry × exit ④).
EPISODES The engine records one refusal per (pair, sleeve, bar) — a setup that stays blocked for 2 h gives ~24 ML rows. Only one shadow per
         (pair, sleeve) is open at a time (live holds one position per pair): a refusal inside an earlier blocked shadow's life (its priced
         exit; still PENDING = open) is SHADOWED — listed, never counted.
COHORT   BLOCKED rows from every decisions export (export_glob: Downloads + exports_archive/), dedup (pair, sleeve, bar), refused at ≥ the
         DECISION_LOG 527 commit + 10 min. KEPT = the same sleeve's CLOSED full-size fills (orders exports, is_probe / *_PROBE out,
         operator-closed out) opened in the same window, re-priced by the SAME pricer on the SAME ruler (real entry price moved 0.10 %
         against the trade — the blocked side's slip; the latency is already in the real fill), booked % shown beside.
🔒 REVERT GATE (pre-committed in DL 527, decided ONCE per line, persisted in reports/.scout_blocked_filters_gate.json — never re-decided):
         per line, every priced (not shadowed) blocked trade the line refused (sole or shared), in refusal order; at the first prefix with
         N ≥ 15 on ≥ 8 UTC days — and only when no earlier refusal of that line is still PENDING and ≥ 5 kept fills of the sleeve in the same
         window are priced (none of them pending) — the blocked trades' average > 0 OR above the sleeve's kept average over the same window
         → REVERT CANDIDATE (operator switches the line off — never automatic); else KEEP. Before that: COLLECTING. The sole-blocked subset
         is shown beside (the line's own effect).
PARITY   on B25 real CLOSED fills (deep review 2026-11-11): FRENZY LITE / RETRY / WILLY / HT exact (corr 1.000, bias ≤ 0.01 %/fill) ·
         momentum long 53 / 53 priced with the top-up, corr 0.97, bias −0.01 · SPIKE_FADE 29 / 30, same sign 90 %, corr 0.71, bias +0.07,
         median |Δ| 0.08 — without TLM (+0.31 booked, +3.44 priced: an intra-minute trail exit 1m bars cannot see) corr 0.78, bias −0.04,
         median |Δ| 0.06. 1m resolution misses intra-minute exits both ways (RAYSOL booked −1.49, priced −0.03); ticks exist for few fades.
Usage: venv/bin/python -I scripts/scout_blocked_filters.py [--no-fetch]
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_ROOT = os.path.dirname(_D)
_ROOT in sys.path or sys.path.insert(0, _ROOT)
import export_glob as _EG              # noqa: E402
import scout_gatekit as _GK            # noqa: E402

DL, FLOOR_DL = 529, 527
LAT_MS, SLIP = 8_000, 0.10
TAKER = 0.045
N_MIN, DAYS_MIN = 15, 8
K_MIN = 5                      # kept fills of the sleeve priced in the gate window before a verdict may freeze (review: never judge vs 0 kept)
MIN, DAY = 60_000, 86_400_000
DEC_GLOB = os.path.expanduser("~/Downloads/scalpars_decisions_*.csv")
ORD_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_*.csv")
STORE = os.path.join(_ROOT, "reports", "SCOUT_BLOCKED_FILTERS.csv")
STATE = os.path.join(_ROOT, "reports", ".scout_blocked_filters_gate.json")
CONFIG = os.path.join(_ROOT, "trading_config.json")
FRENZY = ("FRENZY_LITE", "FRENZY_LITE_RETRY", "FRENZY_LONG", "FRENZY_WILLY", "FRENZY_WILLY_HT")
SHORTS = ("SPIKE_FADE",)
DEC_COLS = ("t", "e", "pair", "dir", "price", "gate", "strategy", "bl_sleeve", "bl_bar_at", "bl_lines", "bl_sole", "bl_is_paper", "bl_entry_fields")
ORD_COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "close_reason", "cell_multiplier_source",
            "pattern_cell_source", "is_probe", "entry_price")


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ms(v):
    if v is None or (isinstance(v, float) and v != v) or str(v) in ("", "nan", "None"):
        return None
    try:
        return int(pd.Timestamp(str(v).replace(" ", "T")[:23]).value // 10**6)
    except Exception:
        return None


def th():
    with open(CONFIG, encoding="utf-8") as fh:
        c = json.load(fh)
    t = dict(c.get("thresholds", c))
    for blk in (c, c.get("investment") or {}):                 # max_holding_time_minutes lives in the investment block
        for k in ("max_holding_time_minutes",):
            if k in blk and k not in t:
                t[k] = blk[k]
    return t


def exit_of(sleeve, t):
    """the sleeve's live exit (dict) from the config each run."""
    st = float(t.get("frenzy_stop_pct", 3.0) or 3.0)
    cap = int(t.get("frenzy_max_hold_minutes", 720) or 720)
    if sleeve in ("FRENZY_LITE", "FRENZY_LITE_RETRY"):
        return dict(tp=float(t.get("frenzy_lite_tp_pct") or t.get("frenzy_tp_pct") or 3.0), sl=st, cap=cap)
    if sleeve == "FRENZY_LONG":
        return dict(tp=float(t.get("frenzy_tp_pct", 1.5) or 1.5), sl=float(t.get("frenzy_long_stop_pct") or st), cap=cap)
    if sleeve in ("FRENZY_WILLY", "FRENZY_WILLY_HT"):
        return dict(sl=float(t.get("frenzy_willy_stop_pct", 4.0) or 4.0), cap=int(t.get("frenzy_willy_max_hold_minutes", 90) or 90),
                    late=float(t.get("frenzy_willy_late_tp_minutes", 30) or 30), ltp=float(t.get("frenzy_willy_late_tp_pct", 0.25) or 0.25))
    if sleeve == "SPIKE_FADE":
        return dict(sl=float(t.get("spike_fade_sl_pct", -1.5) or -1.5), cap=int(t.get("max_holding_time_minutes", 1200) or 1200))
    if sleeve == "MOMENTUM_LONG":
        try:
            import scout_ml_matched_refusals as MM
            return dict(cap=int(MM._rp().HORIZON_MS // MIN))           # the replica's own horizon
        except Exception:
            return dict(cap=2 * 1440)
    return dict(cap=1200)


# ─────────────────────────── load ───────────────────────────
def load_blocked(paths=None, since_ms=None):
    """→ one row per (pair, sleeve, bar): te (refusal + 8 s), E (price 0.10 % against the trade), lines, sole, snapshot dict."""
    fr = []
    for p in (_EG.glob_exports(DEC_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(p, low_memory=False, usecols=lambda c: c in DEC_COLS)
        except Exception:
            continue
        if "e" not in d or "bl_sleeve" not in d:
            continue
        d = d[d.e.astype(str) == "BLOCKED"]
        if len(d):
            fr.append(d)
    cols = ["pair", "sleeve", "direction", "t_ms", "te", "price", "E", "lines", "sole", "snap", "bar", "day", "key"]
    if not fr:
        return pd.DataFrame(columns=cols)
    d = pd.concat(fr, ignore_index=True)
    d["t_ms"] = [_ms(x) for x in d.t]
    d["bar"] = d.bl_bar_at.astype(str).str[:19]
    d = d[d.t_ms.notna()].sort_values("t_ms").drop_duplicates(["pair", "bl_sleeve", "bar"], keep="first")
    if since_ms is not None:
        d = d[d.t_ms >= since_ms]
    out = []
    for r in d.itertuples():
        px = _f(r.price)
        if not (px and px > 0):
            continue
        short = str(r.dir).upper() == "SHORT"
        try:
            snap = json.loads(r.bl_entry_fields) if isinstance(r.bl_entry_fields, str) else {}
        except Exception:
            snap = {}
        te = int(r.t_ms) + LAT_MS
        out.append(dict(pair=str(r.pair), sleeve=str(r.bl_sleeve), direction="SHORT" if short else "LONG", t_ms=int(r.t_ms), te=te, price=px,
                        E=px * (1 - SLIP / 100 if short else 1 + SLIP / 100), lines=str(r.bl_lines if pd.notna(r.bl_lines) and str(r.bl_lines) else (r.gate if pd.notna(r.gate) else "")),
                        sole=str(r.bl_sole).lower() in ("true", "1", "1.0"), snap=snap, bar=r.bar,
                        day=pd.Timestamp(te, unit="ms").strftime("%Y-%m-%d"), key=f"{r.pair}|{r.bl_sleeve}|{r.bar}"))
    return pd.DataFrame(out, columns=cols)


def sleeve_of(es, d):
    es = str(es)
    if es in ("MOMENTUM", "", "nan"):
        return "MOMENTUM_LONG" if str(d).upper() == "LONG" else "MOMENTUM_SHORT"
    return es


def load_kept(paths=None, since_ms=None):
    """CLOSED full-size bot fills (sleeve, opened ms, pct, day), dedup (opened_at[:19], pair), newest export wins; operator-closed out."""
    fr = []
    for i, p in enumerate(_EG.glob_exports(ORD_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(p, low_memory=False, usecols=lambda c: c in ORD_COLS or str(c).startswith("entry_"))
        except Exception:
            continue
        if {"opened_at", "pair", "entry_strategy", "status", "pnl_percentage"} <= set(d.columns):
            if since_ms is not None:                              # the floor right after the read (an export holds months of fills)
                d = d[pd.to_datetime(d.opened_at.astype(str).str.replace(" ", "T").str[:19], errors="coerce") >= pd.Timestamp(int(since_ms), unit="ms")]
            fr.append(d.copy().assign(_m=os.path.getmtime(p) if os.path.exists(p) else i))
    if not fr:
        return pd.DataFrame(columns=["sleeve", "o_ms", "pct", "day"])
    d = pd.concat(fr, ignore_index=True)
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d = d.sort_values("_m").drop_duplicates(["_k", "pair"], keep="last")
    d = d[d.status.astype(str).str.upper() == "CLOSED"]
    prb = pd.Series(False, index=d.index)
    for c in ("cell_multiplier_source", "pattern_cell_source"):
        if c in d:
            prb |= d[c].astype(str).str.upper().str.contains("PROBE")
    if "is_probe" in d:
        prb |= d.is_probe.astype(str).str.lower().isin(("true", "1", "1.0"))
    d = d[~prb]
    d, _ox = _GK.split_operator_closed(d) if len(d) else (d, d)
    d["o_ms"] = [_ms(x) for x in d._k]
    d = d[d.o_ms.notna()]
    if since_ms is not None:
        d = d[d.o_ms >= since_ms]
    d["sleeve"] = [sleeve_of(e, x) for e, x in zip(d.entry_strategy, d.get("direction", pd.Series("LONG", index=d.index)))]
    d["pct"] = pd.to_numeric(d.pnl_percentage, errors="coerce")
    d["day"] = pd.to_datetime(d.o_ms, unit="ms").dt.strftime("%Y-%m-%d")
    return d[d.pct.notna()].reset_index(drop=True)


def kept_as_rows(K):
    """kept fills → pricer rows at their REAL entry moved 0.10 % against the trade (the blocked side's slip ruler — review M1: the paper
    fills carry ≈ 0 slip, so without it every blocked trade would start 0.10 % behind what it is compared with); the same pricer prices
    both sides of the gate."""
    cols = ["pair", "sleeve", "direction", "t_ms", "te", "price", "E", "lines", "sole", "snap", "bar", "day", "key", "booked"]
    out = []
    for r in K.to_dict("records"):
        E = _f(r.get("entry_price"))
        if not (E and E > 0):
            continue
        snap = {k: v for k, v in r.items() if str(k).startswith("entry_")}
        sp = snap.get("entry_frenzy_spike_at")
        if isinstance(sp, str) and sp:
            snap["entry_frenzy_spike_at"] = _ms(sp)
        short = str(r.get("direction", "LONG")).upper() == "SHORT"
        out.append(dict(pair=str(r["pair"]), sleeve=r["sleeve"], direction="SHORT" if short else "LONG", t_ms=int(r["o_ms"]), te=int(r["o_ms"]),
                        price=E, E=E * (1 - SLIP / 100 if short else 1 + SLIP / 100), lines="KEPT", sole=False, snap=snap, bar=pd.Timestamp(int(r["o_ms"]), unit="ms").strftime("%Y-%m-%dT%H:%M:%S"),
                        day=r["day"], key=f"K|{r['pair']}|{r['sleeve']}|{int(r['o_ms'])}", booked=r["pct"]))
    return pd.DataFrame(out, columns=cols)


# ─────────────────────────── pricing ───────────────────────────
def _late_over(now_ms, end_ms):
    """a path that is still missing ≥ 1 day after its horizon will not come — NOT_EVALUABLE; before that PENDING."""
    return "NOT_EVALUABLE" if now_ms > end_ms + DAY else "PENDING"


def _frenzy_exit_ms(P, x, T=None):
    """the event time the walk_v2 exit fired (the same stop / TP / late-TP conditions, no floor) — the shadow's life for EPISODES."""
    te = int(P["te"])
    t, hi, lo = P["t"], P["hi"], P["lo"]
    n = int(np.searchsorted(t, te + x["cap"] * MIN, side="right"))
    tp = float(T if T is not None else x["tp"])
    late, ztp = x.get("late"), x.get("ltp")
    use_late = T is not None and late is not None and ztp is not None and float(ztp) < tp
    for i in range(n):
        line = float(ztp) if (use_late and t[i] > te + late * MIN) else tp
        if lo[i] <= -x["sl"] or hi[i] >= line:
            return int(t[i])
    return te + x["cap"] * MIN


def price_frenzy(r, t, now_ms, path_cache, tcache, bud):
    """FRENZY family on the validated net-path walker. → (pct, why, status, exit_ms)."""
    import scout_frenzy_willy_exit as FX
    import scout_willy_exit_v2 as SV
    import scout_frenzy_v2b_shadow as VB
    x = exit_of(r["sleeve"], t)
    end = int(r["te"]) + (x["cap"] + 1) * MIN
    P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), x["cap"], path_cache)
    if P is None or not len(P.get("t", [])):
        return None, "NO_PATH", _late_over(now_ms, end), None
    T = None
    if r["sleeve"] in ("FRENZY_WILLY", "FRENZY_WILLY_HT"):
        rr = dict(pair=r["pair"], opened_at=pd.Timestamp(r["te"], unit="ms").strftime("%Y-%m-%dT%H:%M:%S"),
                  entry_frenzy_spike_at=(pd.Timestamp(int(r["snap"]["entry_frenzy_spike_at"]), unit="ms").strftime("%Y-%m-%dT%H:%M:%S")
                                         if _f(r["snap"].get("entry_frenzy_spike_at")) else r["snap"].get("entry_frenzy_spike_at")),
                  entry_price=r["E"], quantity=1.0, entry_fee=r["E"] * TAKER / 100)
        H, _src = VB.resolve_h(rr, now_ms, bud, tcache)
        T = VB.v2b_target(rr, H)[0] if H else None
        if T is None:
            return None, "NO_TARGET", _late_over(now_ms, end), None
        w = SV.walk_v2(P, T, sl=x["sl"], cap=x["cap"], late=x["late"], ztp=x["ltp"])
    else:
        w = SV.walk_v2(P, x["tp"], sl=x["sl"], cap=x["cap"], late=None, ztp=None)
    if w["why"] == "UNRES":
        return None, "UNRES", _late_over(now_ms, end), None
    return float(w["pct"]), str(w["why"]), "PRICED", _frenzy_exit_ms(P, x, T)


def price_fade(r, t, now_ms, k1m=None):
    """SPIKE_FADE short on today's live fade exit, fed 1m bars (high → low → close per bar). → (pct, why, status, exit_ms).
    Per reading, the live order (services/trading_engine.py, the realtime non-flip SHORT block): stop first (engine epsilon +0.01), the
    flat HARD_TP cap if no ladder, then the floor from the peak BEFORE this reading — the ATR trail armed by short_runner_arm (normal arm
    on the since-entry peak, or the LATE arm on the post-X peak) and the HARD_TP ladder on the since-entry peak, the higher binding —
    then the peaks update. A crossed stop / floor books the LINE, or the bar open when the bar opened through it (a gap)."""
    import scout_willy_timecap as WT
    from services.trading_engine import (fade_base_shadow_seed, hard_tp_ladder_floor, next_fade_late_peak, short_runner_arm,
                                         short_trail_floor)
    x = exit_of("SPIKE_FADE", t)
    k = WT.load_1m(r["pair"]) if k1m is None else k1m
    te = int(r["te"])
    end = te + x["cap"] * MIN
    first = (te // MIN) * MIN                                          # the ENTRY minute too (review: a stop inside it was missed)
    b = k[(k.open_time >= first) & (k.open_time < end)] if len(k) else k
    if not len(b):
        return None, "NO_PATH", _late_over(now_ms, end), None
    E = float(r["E"])
    fees = 2 * TAKER
    T = SimpleTh(t)
    st = fade_base_shadow_seed(T, 0.0, _f(r["snap"].get("entry_atr_pct")) or 0.0, stop=x["sl"])
    pk, lpk = st["peak"], None
    net = lambda px: (E - float(px)) / E * 100 - fees
    has_o = "o" in b.columns
    pk_new = False                                                    # the floor's peak was set inside THIS bar
    for bar in b.itertuples():
        age = (int(bar.open_time) + MIN - te) / MIN                # the bar's readings are judged at its close age (conservative for X)
        op = net(bar.o) if (has_o and _f(bar.o) is not None and int(bar.open_time) > first) else None
        pk_new = False
        for j, px in enumerate((bar.h, bar.l, bar.c)):
            v = net(px)
            # a stop / a floor from an EARLIER bar books its line (the bar OPEN on a gap, first reading of a later bar only); a floor raised by a
            # peak inside this same bar books the reading itself — 1m bars cannot tell how far the in-bar peak / give-back went (deep review:
            # the line there over-credits, the open under-credits)
            hit = (lambda line: min(line, op)) if (j == 0 and op is not None) else (lambda line: line)
            if v <= st["stop"] + 0.01:
                return float(hit(st["stop"])), "STOP", "PRICED", int(bar.open_time) + MIN
            if st.get("flat_cap") and v >= st["flat_cap"]:
                return float(st["flat_cap"]), "HARD_TP", "PRICED", int(bar.open_time) + MIN
            armed, pkf, late = short_runner_arm(T, "SPIKE_FADE", age, pk, lpk)
            tf = short_trail_floor(T, pkf, st["atr"])[0] if (armed and st["atr"] > 0 and st.get("use_atr", True)) else None
            lf = hard_tp_ladder_floor(st["rungs"], pk)[0] if st.get("rungs") else None
            cand = [(f, w) for f, w in ((tf, "TRAIL_LATE" if late else "TRAIL"), (lf, "LADDER")) if f is not None]
            if cand:
                fl, why = max(cand, key=lambda q: q[0])
                if fl >= 0 and v <= fl:
                    return float(v if pk_new else hit(fl)), why, "PRICED", int(bar.open_time) + MIN
            if v > pk:
                pk, pk_new = v, True
            lpk = next_fade_late_peak(T, "SPIKE_FADE", age, lpk, v)
    if int(b.open_time.iloc[-1]) + MIN >= end:
        return float(net(b.c.iloc[-1])), "CAP", "PRICED", end
    return None, "UNRES", _late_over(now_ms, end), None


class SimpleTh:
    """the config dict as the attribute namespace the engine's pure functions read — a missing key raises AttributeError, so the
    engine's own getattr defaults apply (review: returning None silently switched the ATR trail / ladder off)."""
    def __init__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, k):
        raise AttributeError(k)


def price_ml(rows, now_ms, fetch=True):
    """momentum long on the RP.BASE replica; with fetch, tops up the 1m / 5m / BTC paths first (scout_ml_matched_refusals.ensure_paths,
    budgeted, the folder the replica reads — review H1: nothing else fetches a blocked ML row). → {key: (pct, why, status, exit_ms)}."""
    import scout_ml_matched_refusals as MM
    fills = [dict(key=r["key"], pair=r["pair"], o_ms=int(r["te"]), E=float(r["E"]), atr=_f(r["snap"].get("entry_atr_pct"))) for r in rows]
    if not fills:
        return {}
    if fetch:
        try:
            MM.ensure_paths(fills, now_ms, MM.Budget(), MM._rp().HORIZON_MS)
        except Exception:
            pass
    out = MM.price_replica(fills, now_ms)
    hz = int(MM._rp().HORIZON_MS)
    te = {f["key"]: f["o_ms"] for f in fills}
    return {k: ((v["pct"], v["reason"], v["status"], v.get("exit_ms")) if (v.get("pct") is not None or v["status"] != "PRICED")
                else (None, v["reason"], _late_over(now_ms, te.get(k, 0) + hz), None)) for k, v in out.items()}   # no pct: times out


def era_of(sleeve, t):
    """the exit a stored price was made on — a different era = re-price (never mix two exits in one gate)."""
    if sleeve == "MOMENTUM_LONG":
        try:
            import scout_ml_matched_refusals as MM
            import hashlib
            return "ML:" + hashlib.sha1(json.dumps(MM._rp().BASE, sort_keys=True, default=str).encode()).hexdigest() + f"|{SLIP}|{LAT_MS}|{TAKER}"
        except Exception:
            return "ML:?"
    x = exit_of(sleeve, t)
    if sleeve in FRENZY:                                                 # keys that shape the FRENZY / WILLY exit beyond exit_of (review)
        x = dict(x, **{k: t.get(k) for k in ("frenzy_lock_arm_pct", "frenzy_lite_c_enabled", "frenzy_willy_high_target_enabled",
                                              "frenzy_willy_high_k_pct", "frenzy_willy_high_tp_max_pct", "frenzy_willy_tp_pct")})
    x = dict(x, _ruler=(SLIP, LAT_MS, TAKER))                            # the slip / latency / fee ruler is part of the price
    if sleeve in SHORTS:
        x = dict(x, **{k: t.get(k) for k in ("hard_tp_enabled", "hard_tp_ladder_short", "hard_tp_pct", "runner_trail_short_arm_peak",
                                              "runner_trail_short_atr_mult", "runner_trail_short_giveback_frac", "runner_trail_short_use_atr",
                                              "runner_trail_short_be_ratchet_enabled", "runner_trail_short_be_lock_pct",
                                              "spike_fade_late_arm_after_min", "spike_fade_late_arm_peak")})
    return json.dumps(x, sort_keys=True, default=str)


def load_store(path=STORE):
    """{key: (pct, why, exit_ms, era)}; an unreadable store is moved aside (.bad-<ts>), never silently overwritten."""
    if not path or not os.path.exists(path):
        return {}
    try:
        s = pd.read_csv(path, dtype={"era": str})
        ex = s.exit_ms if "exit_ms" in s else pd.Series([None] * len(s))
        er = s.era if "era" in s else pd.Series([""] * len(s))
        return {k: (float(p), w, (int(e) if _f(e) is not None else None), (a if isinstance(a, str) else ""))
                for k, p, w, e, a in zip(s.key, s.pct, s.why, ex, er) if pd.notna(p)}
    except Exception:
        try:
            os.replace(path, f"{path}.bad-{int(time.time())}")
        except Exception:
            pass
        return {}


def save_store(store, path=STORE):
    tmp = f"{path}.{os.getpid()}.tmp"
    pd.DataFrame([dict(key=k, pct=v[0], why=v[1], exit_ms=v[2], era=v[3]) for k, v in sorted(store.items())],
                 columns=["key", "pct", "why", "exit_ms", "era"]).to_csv(tmp, index=False)
    os.replace(tmp, path)


def price_all(B, now_ms, fetch=True, store_path=STORE):
    """→ B + pct / why / status / exit_ms. Final prices are cached with their exit era; pending ones (and another era's) re-priced."""
    t = th()
    store = load_store(store_path) if store_path else {}
    B = B.copy()
    eras = {sl: era_of(sl, t) for sl in set(B.sleeve)} if len(B) else {}
    fresh = [k in store and store[k][3] == eras.get(sl) for k, sl in zip(B.key, B.sleeve)]
    res = {}
    todo = B[[not f for f in fresh]]
    fz = todo[todo.sleeve.isin(FRENZY + SHORTS)]
    if fetch and len(fz):
        try:
            import scout_willy_timecap as WT
            span = max(exit_of(sl, t)["cap"] for sl in set(fz.sleeve))       # the fade cap (1,200 min) is longer than FRENZY's
            WT.ensure_data(pd.DataFrame(dict(ts=pd.to_datetime(fz.te, unit="ms"), pair=fz.pair.values, te=fz.te.values, src=None)),
                           span, now_ms, fetch=True, max_req=40)
        except Exception:
            pass
    path_cache, tcache = {}, {}
    try:
        import scout_frenzy_v2b_shadow as VB
        bud = VB._Bud(fetch)
    except Exception:
        bud = None
    for r in todo.to_dict("records"):
        try:
            if r["sleeve"] in FRENZY:
                res[r["key"]] = price_frenzy(r, t, now_ms, path_cache, tcache, bud)
            elif r["sleeve"] in SHORTS:
                res[r["key"]] = price_fade(r, t, now_ms)
        except Exception as e:
            res[r["key"]] = (None, f"ERR:{str(e)[:60]}", _late_over(now_ms, int(r["te"]) + (exit_of(r["sleeve"], t)["cap"] + 1) * MIN), None)
    ml = todo[todo.sleeve == "MOMENTUM_LONG"]
    try:
        res.update(price_ml(ml.to_dict("records"), now_ms, fetch=fetch))
    except Exception as e:
        for k, te in zip(ml.key, ml.te):
            res[k] = (None, f"ERR:{str(e)[:60]}", _late_over(now_ms, int(te) + 2 * DAY), None)
    pct, why, stt, exm = [], [], [], []
    for k, sl, ok in zip(B.key, B.sleeve, fresh):
        if ok:
            p, w, e, _a = store[k]; s = "PRICED"
        else:
            p, w, s, e = res.get(k, (None, "UNKNOWN_SLEEVE", "NOT_EVALUABLE", None))
            if s == "PRICED" and p is not None:
                store[k] = (p, w, e, eras.get(sl, ""))
        pct.append(p); why.append(w); stt.append(s); exm.append(e)
    B["pct"], B["why"], B["status"], B["exit_ms"] = pct, why, stt, exm
    if store_path:
        save_store(store, store_path)
    return B


def shadow_episodes(B, t=None):
    """one blocked shadow per (pair, sleeve) at a time: a refusal while an earlier counted shadow on that pair / sleeve is still open (its
    priced exit; PENDING = open until priced; NOT_EVALUABLE = its cap) → status SHADOWED (listed, never counted)."""
    if not len(B):
        return B
    t = th() if t is None else t
    B = B.sort_values("te").copy()
    stt = B.status.tolist()
    busy = {}
    for i, (p, sl, te, s, e) in enumerate(zip(B.pair, B.sleeve, B.te, B.status, B.exit_ms)):
        k = (p, sl)
        if te < busy.get(k, -1):
            stt[i] = "SHADOWED"
            continue
        cap_end = int(te) + exit_of(sl, t)["cap"] * MIN
        busy[k] = (int(e) if (s == "PRICED" and _f(e) is not None) else (float("inf") if s == "PENDING" else cap_end))
    B["status"] = stt
    return B


# ─────────────────────────── gate ───────────────────────────
def _has_line(lines, line):
    return lines.astype(str).str.split("+").map(lambda ls: line in ls)


def gate_read(B, K, line):
    """the line's blocked trades (priced, not shadowed, refusal order) vs the sleeve's kept fills → (state, detail, prefix)."""
    a = B[_has_line(B.lines, line)].sort_values("te") if len(B) else B
    g = a[a.status == "PRICED"]
    seen, n, cut = set(), 0, None
    for i, dd in enumerate(g.day):
        n += 1; seen.add(dd)
        if n >= N_MIN and len(seen) >= DAYS_MIN:
            cut = i + 1
            break
    if cut is None:
        return "COLLECTING", f"{len(g)}/{N_MIN} priced on {g.day.nunique() if len(g) else 0}/{DAYS_MIN} days", g
    pre = g.iloc[:cut]
    end = int(pre.te.max())
    pend = B[(B.status == "PENDING") & (B.te <= end)]  # ANY pending refusal of the sleeve (review: a pending row of another line can
    if len(pend):                                    # shadow this line's refusals on its pair) → the first 15 REFUSALS are not decidable yet
        return "COLLECTING", f"{N_MIN} priced, but {len(pend)} earlier refusal(s) still pending", g
    sleeves = set(pre.sleeve)
    kw = K[K.sleeve.isin(sleeves) & (K.te <= end)] if len(K) and "status" in K else K.iloc[0:0]
    k = kw[kw.status == "PRICED"] if len(kw) else kw
    if len(kw[kw.status == "PENDING"]) if len(kw) else 0:
        return "COLLECTING", f"{N_MIN} priced; {int((kw.status == 'PENDING').sum())} kept fill(s) in the window still pending", g
    if len(k) < K_MIN:
        return "COLLECTING", f"{N_MIN} priced; kept fills priced in the window {len(k)}/{K_MIN}", g
    ba, ka = float(pre.pct.mean()), float(k.pct.mean())
    det = (f"blocked {len(pre)} on {pre.day.nunique()} d · WR {(pre.pct > 0).mean():.0%} · avg {ba:+.3f} % vs kept {len(k)} · "
           f"{(k.pct > 0).mean():.0%} · {ka:+.3f} %")
    if ba > 0 or ba > ka:
        return "REVERT CANDIDATE (operator switches the line off — never automatic)", det, pre
    return "KEEP", det, pre


LINES = {"FRENZY_WILLY": ("WILLY_GAP", "WILLY_FAILED_SPIKE"), "FRENZY_WILLY_HT": ("WILLY_FAILED_SPIKE",), "FRENZY_LITE_RETRY": ("RETRY_BTC1H",),
         "FRENZY_LITE": ("LITE_BTC_EMA20_DOWN",), "FRENZY_LONG": ("LONG_WEAK_FRENZY",), "SPIKE_FADE": ("FADE_LATE_WEAK",),
         "MOMENTUM_LONG": ("ML_MKT_HOT", "ML_BULL90")}


def _load_state(path):
    """→ (state dict, usable path or None, warning or None). Valid JSON of the wrong shape counts as unreadable (never crashes the run)."""
    if not (path and os.path.exists(path)):
        return {}, path, None
    try:
        st = json.load(open(path))
        if not isinstance(st, dict) or not all(isinstance(v, dict) and {"state", "detail", "at"} <= set(v) for v in st.values()):
            raise ValueError("shape")
        return st, path, None
    except Exception:
        return {}, None, "⚠ gate state unreadable — nothing frozen this run (operator restore needed)."


def run(now_ms=None, dec_paths=None, ord_paths=None, state_path=STATE, store_path=STORE, since_ms=None, fetch=True):
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🧱💲 BLOCKED TRADES — the armed filters' refusals priced on today's exit + per-line revert gates (DECISION_LOG {DL}; filters DL {FLOOR_DL})", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(FLOOR_DL, lag_min=10, label="armed filters (blocked-trade pricer)")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {FLOOR_DL} commit is not in git; nothing is counted.", ""]
    B0 = load_blocked(dec_paths, since_ms)
    K0 = load_kept(ord_paths, since_ms)
    K0 = K0[K0.sleeve.isin(list(LINES))] if len(K0) else K0
    KR = kept_as_rows(K0) if len(K0) else kept_as_rows(pd.DataFrame())
    ALL = pd.concat([B0.assign(booked=np.nan), KR], ignore_index=True) if (len(B0) or len(KR)) else B0.assign(booked=[])
    ALL = price_all(ALL, now_ms, fetch=fetch, store_path=store_path) if len(ALL) else ALL.assign(pct=[], why=[], status=[], exit_ms=[])
    isk = ALL.key.astype(str).str.startswith("K|")
    B = shadow_episodes(ALL[~isk].reset_index(drop=True))
    K = ALL[isk].reset_index(drop=True)
    st, state_path, warn = _load_state(state_path)
    if warn:
        L.append(warn)
    n_sh = int((B.status == "SHADOWED").sum()) if len(B) else 0
    L += [f"Cohort: BLOCKED rows refused ≥ {pd.Timestamp(since_ms, unit='ms'):%Y-%m-%d %H:%M} UTC · priced {int((B.status == 'PRICED').sum())} · "
          f"pending {int((B.status == 'PENDING').sum())} · not evaluable {int((B.status == 'NOT_EVALUABLE').sum())} · shadowed (inside an earlier "
          f"blocked shadow of the same pair) {n_sh} · kept fills {len(K)} (both sides 0.10 % slip against the trade, taker; blocked entry +8 s; "
          f"each sleeve's live exit).", "",
          "| Sleeve · line | Blocked N | days | WR | avg % | Σ % | sole: N · avg | kept (same pricer): N · avg · booked avg | 🔒 revert gate |",
          "|---|---|---|---|---|---|---|---|---|"]
    changed = False
    for sl, lines in LINES.items():
        Bs = B[B.sleeve == sl] if len(B) else B
        Ks = K[K.sleeve == sl] if len(K) else K
        Kk = Ks[Ks.status == "PRICED"] if len(Ks) else Ks
        kept = f"{len(Kk)} · {Kk.pct.mean():+.3f} · {pd.to_numeric(Kk.booked, errors='coerce').mean():+.3f}" if len(Kk) else "0"
        for ln in lines:
            P = Bs[_has_line(Bs.lines, ln) & (Bs.status == "PRICED")] if len(Bs) else Bs
            S = P[P.sole.astype(bool)] if len(P) else P
            key = f"{sl}:{ln}"
            rec = st.get(key)
            if not rec:
                state, det, _ = gate_read(Bs, Ks, ln) if len(Bs) else ("COLLECTING", "0 priced", None)
                if state != "COLLECTING" and state_path:
                    st[key] = rec = dict(state=state, detail=det, at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
                    changed = True
                gtxt = f"COLLECTING ({det})" if state == "COLLECTING" else f"**{state}**"
            else:
                gtxt = f"**{rec['state']}** (frozen {rec['at']}: {rec['detail']})"
            sole = f"{len(S)} · {S.pct.mean():+.3f}" if len(S) else "0"
            if len(P):
                L.append(f"| {sl} · {ln} | {len(P)} | {P.day.nunique()} | {(P.pct > 0).mean():.0%} | {P.pct.mean():+.3f} | {P.pct.sum():+.2f} | {sole} | {kept} | {gtxt} |")
            else:
                L.append(f"| {sl} · {ln} | 0 | – | – | – | – | 0 | {kept} | {gtxt} |")
    if changed and state_path:
        tmp = f"{state_path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(st, fh, indent=1)
        os.replace(tmp, state_path)
    pend = B[B.status.isin(("PENDING", "NOT_EVALUABLE"))] if len(B) else B
    if len(pend):
        L += ["", "Not priced: " + " · ".join(f"{r.sleeve} {r.pair} {pd.Timestamp(r.te, unit='ms'):%m-%d %H:%M} ({r.status}: {r.why})"
                                              for r in pend.head(12).itertuples()) + (" …" if len(pend) > 12 else "")]
    L += ["", f"🔒 Gate (DL {FLOOR_DL}, frozen once per line): at {N_MIN} priced blocked trades on {DAYS_MIN} UTC days (no earlier refusal pending, "
          f"≥ {K_MIN} kept fills priced in the window) — blocked avg > 0 OR above the sleeve's kept avg over the same window (the kept fills "
          "priced by the SAME pricer on the SAME slip ruler) → REVERT CANDIDATE (the operator switches that line's watch_block_* toggle off); "
          "else KEEP. Never automatic.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run(fetch="--no-fetch" not in sys.argv)))
