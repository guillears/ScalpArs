#!/usr/bin/env python3
"""ML EXIT × MACRO study (2026-11-11, operator: "check different exit strategies for all remaining momentum longs with current filters in
master (incl. B25) — trailing starting later (at +1 %), fixed TP 0.5 / 1 / 1.5 / 2, others? — and a macro variable: only exit once the
macro that marked the entry changes direction, like a BTC move. I have the strong feeling we are leaving money on the table.")

STUDY ONLY — no engine / config / template edit, no git, no bot API. Writes reports/ML_EXIT_MACRO_2026-11-11_* only.

COHORT   every MOMENTUM LONG, non-probe, stack_keep row of reports/MASTER_POOL_stacked.csv (STACK 2026-11-11a = today's real stack, ML_MKT_HOT
         + ML_BULL90 armed, B25 archived). % at 1×. Probes OUT (memory: full-size only).
PRICING  the VALIDATED ML exit replica scripts/ml_exit_optimize_yr5.py (RP.evaluate, untouched — DL 380 parity 104/104) with the
         scout_ml_matched_refusals top-up readers (1m / 5m / BTC 5m, public klines, budgeted). Ticks where the day archive exists, else 1m.
         Every variant = an RP params dict; the MACRO exits ride the replica's EMA13 hook (P['e13'] = ±inf, P['flip'] = the macro mask) so
         the evaluator itself is never edited — BASE is RP.BASE bit-exact.
MACRO    a signal is KNOWN at its bar close + LAT (60 s ≈ half a scan cycle + the bar close; the engine scans every ~112 s), read on CLOSED
         bars only (no look-ahead). Two semantics, pre-declared:
           M_*  (pure macro hold, from entry): runner trail / ladder / 180-min clock OFF, live stop + recovery hold ON, exit at the first
                CHANGE of the signal to the opposite side after entry (it must have been on our side at least once since entry — a signal
                already against us at entry has to flip back and then turn again). Max hold 20 h (live max_holding).
           H<a>_* (hybrid): today's exit until the running peak reaches +a; from then the runner trail + ladder are replaced by the
                macro exit (STATE: exit at the first known bar where the signal is against us) with the +0.10 lock floor kept; the
                180-min clock applies only before the arm.
         Signals: BTC_E20 (BTC 5m close < its EMA20) · BTC_SLOPE (BTC 5m EMA20 falling) · BTC_DD30 / BTC_DD50 (BTC −0.30 / −0.50 % below its
                  highest 5m close since entry — flip-only by construction) · BTC_1H (BTC 1h EMA20 falling, hourly closes) ·
                  PAIR_E5E8 (pair 5m EMA5 < EMA8 = the entry trigger reversed) · PAIR_E20 (pair 5m close < EMA20 = signal_active lost) ·
                  PAIR_E13 (pair 5m close < EMA13) · PAIR_E8E13 (pair 5m EMA8 < EMA13).
GRID     (pre-declared, written before any variant was run on this cohort — see VARIANTS). Multiple testing: Bonferroni over every
         pre-declared variant on the Δ day-block bootstrap.
Usage:   venv/bin/python -W ignore scripts/study_ml_exit_macro_1111.py [--no-fetch]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import scout_ml_matched_refusals as MM  # noqa: E402

OUT = os.path.join(ROOT, "reports", "ML_EXIT_MACRO_2026-11-11")
MIN, BAR5, HOUR, DAY = 60_000, 300_000, 3_600_000, 86_400_000
LAT = 60_000
INF = 999.0


def _rp():
    import ml_exit_optimize_yr5 as RP
    return RP


def V(**kw):
    p = dict(_rp().BASE)
    p.update(kw)
    return p


# ─────────────── PRE-DECLARED GRID (2026-11-11, before any run on this cohort) ───────────────
def variants():
    RP = _rp()
    off = dict(run_arm=INF, ladder=[])                         # runner + ladder off
    G = {"BASE": dict(RP.BASE)}
    # T — fixed TP with the live stop, no runner / ladder; live 180-min clock (T) or held to 20 h (TH)
    for tp in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        G[f"T{tp:g}"] = V(**off, tp=tp)
    for tp in (1.0, 1.5, 2.0):
        G[f"TH{tp:g}"] = V(**off, tp=tp, noexp=1200)
    # A — trailing starts later: arm at +a, trail a FIXED d % below the peak (lock +0.10 floor), ladder off, live clock
    for a in (0.75, 1.0, 1.5):
        for d in (0.3, 0.5, 0.75):
            G[f"A{a:g}_d{d:g}"] = V(run_arm=a, run_g_fixed=d, ladder=[])
    # A* — same, but the trail is held to 20 h once armed (clock only before the arm)  — via the hybrid machinery (mask = none)
    for a in (1.0,):
        for d in (0.5, 0.75):
            G[f"AH{a:g}_d{d:g}"] = dict(V(run_arm=a, run_g_fixed=d, ladder=[], noexp=1200), _clock_pre=a)
    # R — the 10-26 winners re-read on today's filtered cohort
    G["R_LAD_off"] = V(ladder=[])
    G["R_LOCK0.2"] = V(run_lock=0.20)
    G["R_K1.5"] = V(run_g_atr=1.5)
    G["R_ARM1.0_atr"] = V(run_arm=1.0)
    # M — pure macro hold (flip semantics)
    for s in SIGNALS:
        G[f"M_{s}"] = dict(V(**off, noexp=1200, ema13=True), _mx=s, _sem="flip")
    # H — hybrid: today's exit until +a, then macro STATE exit with the +0.10 lock floor; clock only before the arm
    for a in (0.40, 1.0):
        for s in ("BTC_E20", "BTC_SLOPE", "BTC_DD30", "BTC_1H", "PAIR_E5E8", "PAIR_E20", "PAIR_E13"):
            G[f"H{a:g}_{s}"] = dict(V(run_arm=a, run_g_fixed=INF, ladder=[], noexp=1200, ema13=True), _mx=s, _sem="state",
                                    _arm=a, _clock_pre=a)
    return G


SIGNALS = ("BTC_E20", "BTC_SLOPE", "BTC_DD30", "BTC_DD50", "BTC_1H", "PAIR_E5E8", "PAIR_E20", "PAIR_E13", "PAIR_E8E13")


# ─────────────── cohort ───────────────
def _ms(s):
    """UTC epoch ms (pandas 3 parses to datetime64[s|us] — never astype('int64') a parsed column directly)."""
    return pd.to_datetime(s, format="mixed", utc=True).dt.tz_convert(None).values.astype("datetime64[ms]").astype(np.int64)


def cohort():
    d = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    k = d[(d.entry_strategy == "MOMENTUM") & (d.direction == "LONG") & (d.stack_keep == True) & (d.is_probe != True)].copy()  # noqa: E712
    k["o_ms"] = _ms(k.opened_at)
    k["c_ms"] = _ms(k.closed_at)
    k["key"] = k.opened_at.astype(str).str[:19] + "|" + k.pair
    k["E"] = pd.to_numeric(k.entry_price, errors="coerce")
    k["atr"] = pd.to_numeric(k.entry_atr_pct, errors="coerce")
    RP = _rp()
    k["fee_in"] = np.where(k.entry_order_type.astype(str).str.upper().str.startswith("MAKER"), RP.MAKER, RP.TAKER)
    k["y"] = pd.to_numeric(k.stack_pct, errors="coerce")
    # era span from EVERY master row of the era (all sleeves) — the batch's calendar days for the DCR
    dd = d.assign(o=pd.to_datetime(d.opened_at, format="mixed", utc=True), c=pd.to_datetime(d.closed_at, format="mixed", utc=True))
    span = dd.groupby("era").agg(t0=("o", "min"), t1=("c", "max"))
    k = k.join(span, on="era")
    return k.sort_values("o_ms").reset_index(drop=True), d


# ─────────────── macro signals ───────────────
def _btc_frames(RP):
    b = RP._btc()
    ot, c = b[:, 0].astype(np.int64), b[:, 4].astype(float)
    e20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    h = pd.DataFrame({"ot": ot, "c": c})
    h = h[(h.ot % HOUR) == HOUR - BAR5]                       # the 5m bar that closes the hour → hourly close
    h1_ot = (h.ot.values - (HOUR - BAR5)).astype(np.int64)
    h1_c = h.c.values
    h1_e = pd.Series(h1_c).ewm(span=20, adjust=False).mean().values
    return dict(ot=ot, c=c, e20=e20, h1_ot=h1_ot, h1_e=h1_e)


def _known(ot, step):
    return ot + step + LAT                                    # bar open → time the closed bar is known to the engine


def bar_signal(name, B, k5, o_ms):
    """→ (known_ms array, against-us bool array) on that signal's own bar grid, bars known at or after the entry's known bar."""
    if name.startswith("BTC_") and name != "BTC_1H":
        kn = _known(B["ot"], BAR5)
        if name == "BTC_E20":
            ag = B["c"] < B["e20"]
        elif name == "BTC_SLOPE":
            ag = np.r_[False, np.diff(B["e20"]) < 0]
        else:                                                 # BTC_DD30 / DD50 — vs the highest close known since entry
            x = 0.30 if name == "BTC_DD30" else 0.50
            j0 = max(int(np.searchsorted(kn, o_ms, side="right")) - 1, 0)
            c = B["c"][j0:]
            ag = np.zeros(len(B["c"]), bool)
            ag[j0:] = c <= np.maximum.accumulate(c) * (1 - x / 100.0)
        return kn, ag
    if name == "BTC_1H":
        kn = _known(B["h1_ot"], HOUR)
        return kn, np.r_[False, np.diff(B["h1_e"]) < 0]
    if k5 is None:
        return None, None
    kn = _known(k5["ot"], BAR5)
    c = k5["c"]
    if name == "PAIR_E5E8":
        ag = k5["e5"] < k5["e8"]
    elif name == "PAIR_E20":
        ag = c < k5["e20"]
    elif name == "PAIR_E13":
        ag = c < k5["e13"]
    elif name == "PAIR_E8E13":
        ag = k5["e8"] < k5["e13"]
    else:
        raise KeyError(name)
    return kn, ag


def tick_mask(t, o_ms, kn, ag, sem):
    """per tick: state = the latest known bar is against us; flip = from the first known bar after entry that is against us AND
    preceded (since the entry's known bar) by a bar on our side. Also returns whether the signal was against us at entry."""
    if kn is None:
        return np.zeros(len(t), bool), None
    j0 = max(int(np.searchsorted(kn, o_ms, side="right")) - 1, 0)
    at_entry = bool(ag[j0])
    if sem == "state":
        j = np.searchsorted(kn, t, side="right") - 1
        return (j >= j0) & ag[np.clip(j, 0, len(ag) - 1)], at_entry
    seg = ag[j0:]
    ours = np.flatnonzero(~seg)
    if not len(ours):
        return np.zeros(len(t), bool), at_entry
    after = np.flatnonzero(seg[ours[0]:])
    if not len(after):
        return np.zeros(len(t), bool), at_entry
    tk = kn[j0 + ours[0] + after[0]]
    return t >= tk, at_entry


# ─────────────── pricing ───────────────
def price_fill(RP, fx, G, B):
    P = RP.prep_path(fx, RP.HORIZON_MS)
    if P is None:
        return None, None
    t, net = P["t"], P["net"]
    pk = np.maximum.accumulate(np.maximum(net, 0.0))
    rel = t - fx["o_ms"]
    k5 = RP._k5(fx["pair"])
    sig_cache, out, info = {}, {}, {}
    end_ok = (t[-1] - fx["o_ms"]) >= RP.HORIZON_MS - 10 * MIN
    for name, pr in G.items():
        prm = {k: v for k, v in pr.items() if not k.startswith("_")}
        Q = P
        mx, cpre = pr.get("_mx"), pr.get("_clock_pre")
        if mx is not None or cpre is not None:
            m = np.zeros(len(t), bool)
            if mx is not None:
                ck = (mx, pr["_sem"])
                if ck not in sig_cache:
                    kn, ag = bar_signal(mx, B, k5, fx["o_ms"])
                    sig_cache[ck] = tick_mask(t, fx["o_ms"], kn, ag, pr["_sem"])
                m, at_entry = sig_cache[ck]
                info[f"against_at_entry_{mx}"] = at_entry
                if pr.get("_arm") is not None:
                    m = m & (pk >= pr["_arm"] - 0.005)
            if cpre is not None:                              # the 180-min clock only while not armed
                m = m | ((rel >= 180 * MIN) & (pk < cpre - 0.005))
            Q = dict(P, e13=np.where(m, np.inf, -np.inf), flip=m)
            prm["ema13"] = True
        k, why, pct = RP.evaluate(Q, fx, prm)
        if why == "EMA13_CROSS":
            why = "MACRO" if mx is not None else "NO_EXPANSION"
            if mx is not None and cpre is not None and pk[k] < cpre - 0.005 and rel[k] >= 180 * MIN:
                why = "NO_EXPANSION"
        if why == "FIXED_TP":                                  # a resting TP fills AT its price, never at a gapped tick above it
            pct = min(pct, float(prm["tp"]))
        if why == "END_OF_DATA" and not end_ok:
            pct = np.nan
        out[name] = (why, pct, int(t[k]), float(pk[k]))
    # path facts for the decomposition (BASE exit)
    kb = np.searchsorted(t, out["BASE"][2])
    post = (t > t[kb]) & (t <= t[kb] + 4 * HOUR)
    info.update(src=P["src"], mfe_to_exit=float(pk[kb]), mfe_20h=float(pk[-1]), mae_to_exit=float(np.min(net[:kb + 1])),
                post4h_max=float(np.max(net[post])) if post.any() else np.nan,
                post60_max=float(np.max(net[(t > t[kb]) & (t <= t[kb] + HOUR)])) if ((t > t[kb]) & (t <= t[kb] + HOUR)).any() else np.nan,
                horizon_complete=bool(end_ok))
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    C, M = cohort()
    G = variants()
    now = int(time.time() * 1000)
    fills = [dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), atr=float(r.atr) if r.atr == r.atr else None)
             for r in C.itertuples()]
    with MM.replica_with_topup() as RP:
        if not a.no_fetch:
            for i in range(12):
                bud = MM.Budget(max_calls=200)
                MM.ensure_paths(fills, now, bud, RP.HORIZON_MS)
                print(f"top-up round {i}: calls {bud.n} stop={bud.stop}", flush=True)
                if bud.n == 0 or (bud.stop and "call cap" not in str(bud.stop) and "weight" not in str(bud.stop)):
                    break
                if bud.stop and "weight" in str(bud.stop):
                    time.sleep(61)
                RP._K1.clear(); RP._K5.clear(); RP._BTC = None
        B = _btc_frames(RP)
        rows, facts = [], []
        for r in C.itertuples():
            rsi = RP.btc_rsi_at(int(r.o_ms))
            fx = dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in),
                      atr=float(r.atr) if (r.atr == r.atr and r.atr > 0) else None, rsi_entry=None if rsi is None else float(rsi))
            try:
                out, info = price_fill(RP, fx, G, B)
            except Exception as e:
                print("ERR", r.key, e, flush=True)
                out, info = None, None
            if out is None:
                facts.append(dict(key=r.key, src="none"))
                continue
            for v, (why, pct, tex, pkx) in out.items():
                rows.append(dict(key=r.key, variant=v, reason=why, pct=pct, exit_ms=tex, peak_at_exit=pkx))
            facts.append(dict(key=r.key, **info))
    Vv = pd.DataFrame(rows)
    F = C[["key", "era", "pair", "o_ms", "c_ms", "y", "close_reason", "peak_pnl", "trough_pnl", "atr", "t0", "t1", "fee_in",
           "entry_btc_regime"]].merge(pd.DataFrame(facts), on="key", how="left")
    Vv.to_csv(OUT + "_variants.csv", index=False)
    F.to_csv(OUT + "_fills.csv", index=False)
    print("fills", len(F), "priced", Vv.key.nunique(), "rows", len(Vv))


if __name__ == "__main__":
    main()
