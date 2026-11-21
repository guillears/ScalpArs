#!/usr/bin/env python3
"""📡 Route B — SIGNAL-LEVEL loader (research only; DL 597 phase 2 input). One row per signal the bot evaluated since recording began
(2026-11-19 23:23 UTC) — taken, blocked or refused — with the recorded book / tape / liquidation / funding STATE at the signal (no
look-ahead), the market row, and the outcome priced on the coin's own 1-minute path under TODAY's exit for that sleeve.
A separate table holds the new-sleeve triggers H1–H6 (outcomes stored, NEVER printed — pre-registration first).

README
------
Usage (from the repo root):
    venv/bin/python -I scripts/route_b_signals/build.py            # incremental daily run → datasets + reports/ROUTE_B_SIGNALS_<UTC date>.md
    venv/bin/python -I scripts/route_b_signals/build.py --full     # re-price everything (after a code change)
    venv/bin/python -I scripts/route_b_signals/build.py --no-report
Daily: run it after the newest dashboard decisions + orders exports are in ~/Downloads or exports_archive/ (the journal is the bot's
export — signals after the newest export's last row are not visible until the next export; taken fills from the orders export are added).
Incremental: hourly summaries (S3, read-only), complete-day klines and funding are cached under reports/ROUTE_B_SIGNALS_data/
(gitignored); a signal / trigger whose outcome is FINAL is carried over from the previous run unchanged; CENSORED ones (path not
finished yet) are re-priced. Binance: data.binance.vision zips for complete days (no weight); fapi only for the current day + funding,
≤ 1,000 weight / minute, stops on 418 / 429.

Outputs (reports/ROUTE_B_SIGNALS_data/):
    signals.csv.gz / .pkl.gz   one row per signal (identity · status · fill link · possibility / window · rb_* coin state · mk_* market
                               state · out_* outcome under today's exit · flags)
    triggers_H.csv.gz / .pkl.gz H1–H6 openers with the same state columns + fwd_{5,15,60,240}m_pct + a fixed TP/SL grid (g_tp*_sl*_net)
                               — DO NOT summarise outcomes before the pre-registration file is frozen (operator)
    meta.json                  run stamps, coverage, data-quality counters
Rules honoured: no look-ahead (state.py), master over backtest (real journal signals, real fills linked and flagged), window units
(window15 on every row; the report counts windows), full-size only (probe fills dropped), possibilities not events (poss_id / poss_dup).
Module map: common.py (constants, today's exits from trading_config.json) · sources.py (S3 / journals / orders / klines / funding) ·
state.py (state at T) · signals.py (journal → signals, fills, possibilities) · exits.py (walkers) · triggers.py (H1–H6) · report.py.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402
import exits as X           # noqa: E402
import signals as SG        # noqa: E402
import sources as SRC       # noqa: E402
import state as ST          # noqa: E402
import triggers as TR       # noqa: E402

CODE_VERSION = 2            # bump when pricing / state logic changes (forces a re-price of carried-over rows)
OUT_COLS_PREFIX = ("out_",)


def atr_from_klines(k, T_ms, n=14):
    """Wilder ATR(14) % on the 15m bars that CLOSED at or before T (EXHAUSTION's definition: first TR = h − l, EWM α 1/14, ÷ the last
    close). NaN with fewer than 30 bars."""
    if k is None or not len(k):
        return math.nan
    kk = k[k.open_ms + C.MIN_MS <= T_ms]
    if not len(kk):
        return math.nan
    b = kk.assign(b=kk.open_ms // (15 * C.MIN_MS)).groupby("b").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), nb=("c", "size"))
    last_b = (T_ms // (15 * C.MIN_MS)) - 1                                  # the last bar whose close ≤ T
    b = b[b.index <= last_b]
    if len(b) < 30:
        return math.nan
    pc = b.c.shift(1).fillna(b.c.iloc[0])
    tr = np.maximum(b.h - b.l, np.maximum((b.h - pc).abs(), (b.l - pc).abs()))
    tr.iloc[0] = b.h.iloc[0] - b.l.iloc[0]
    a = tr.ewm(alpha=1 / n, adjust=False).mean()
    return float(a.iloc[-1] / b.c.iloc[-1] * 100)


def pessimistic_fill_minute(k, i, e, side):
    """the fill minute (whose pre-fill part is unknown) made PESSIMISTIC: open = the fill price; the ADVERSE extreme keeps the whole minute's
    (it may have come after the fill); the FAVOURABLE extreme is capped at the better of the fill price and the minute's close. Pure."""
    k = k.copy()
    j = k.index[i]
    c = float(k.at[j, "c"])
    k.at[j, "o"] = e
    if side > 0:                     # long: adverse = low, favourable = high
        k.at[j, "h"] = max(e, c)
        k.at[j, "l"] = min(float(k.at[j, "l"]), e, c)
    else:
        k.at[j, "l"] = min(e, c)
        k.at[j, "h"] = max(float(k.at[j, "h"]), e, c)
    return k


def entry_index(k, T_ms):
    """the first kline minute whose open ≥ T + LATENCY_S (the next-minute open)."""
    return int(np.searchsorted(k.open_ms.to_numpy(np.int64), T_ms + C.LATENCY_S * 1000, "left"))


def book_cost_at(coin, minute_ms, side):
    """recorded $6k cost (bps) for a market order `side` at the book row ≤ minute_ms (≤ 2 min old) — used for EXIT legs only."""
    if coin is None or not len(coin.bm):
        return math.nan
    k = np.searchsorted(coin.bm, minute_ms, "right") - 1
    if k < 0 or minute_ms - coin.bm[k] > C.BOOK_MAX_AGE_MIN * C.MIN_MS:
        return math.nan
    b = coin.book.iloc[k]
    return ST.cost_at(b[f"cost_{side}_2500"], b[f"cost_{side}_10000"])[0]


def price_signal(r, spec, k, coin, now_ms, data_end_ms):
    """outcome dict (out_*) for one signal row under today's exit."""
    side = 1 if r["direction"] == "LONG" else -1
    T = int(r["signal_time"].value // 1_000_000)
    o = dict(out_priced=False, out_why="")
    if spec is None:
        o["out_why"] = C.NOT_PRICED_WHY.get(r["sleeve"], "no exit for this sleeve")
        return o
    if not r["priceable"]:
        o["out_why"] = "no entry decision (no entry bar journaled)"
        return o
    if k is None or not len(k):
        o["out_why"] = "no 1m klines"
        return o
    taken = r["status"] == "OPENED" and r.get("fill_entry_price") == r.get("fill_entry_price") and r.get("fill_entry_price") is not None
    unlinked = r["status"] == "OPENED_UNLINKED"            # taken, but its fill is in no orders export yet (still open) → never final
    fill_minute = ""
    if taken:
        f_ms = int(pd.Timestamp(r["fill_opened_at"][:19], tz="UTC").value // 1_000_000)
        i0 = int(np.searchsorted(k.open_ms.to_numpy(np.int64), (f_ms // C.MIN_MS) * C.MIN_MS, "left"))
        e, e_ms, e_src = float(r["fill_entry_price"]), f_ms, "real_fill"
        if i0 < len(k) and int(k.open_ms.iloc[i0]) == (f_ms // C.MIN_MS) * C.MIN_MS:
            k = pessimistic_fill_minute(k, i0, e, side)              # the fill minute is walked, pessimistically
            fill_minute = "pessimistic"
        else:
            fill_minute = "missing"
    else:
        i0 = entry_index(k, T)
        if i0 >= len(k):
            o["out_why"] = "entry minute not in klines yet"
            return o
        e, e_ms, e_src = float(k.o.iloc[i0]), int(k.open_ms.iloc[i0]), "next_minute_open"
    # levels
    atr_src = ""
    atr = math.nan
    if spec["kind"] == "atr":
        atr = r.get("journal_atr_pct", math.nan)
        atr_src = "journal"
        if not (atr == atr and atr > 0):
            atr, atr_src = atr_from_klines(k, T), "computed_15m"
    tp, sl = C.levels_for(spec, atr)
    if tp is None:
        o["out_why"] = "levels undefined (ATR missing)"
        return o
    # entry slip
    ein = "buy" if side > 0 else "sell"
    eout = "sell" if side > 0 else "buy"
    if taken:
        eslip, eslip_src = 0.0, "in_fill_price"
    elif r.get(f"rb_cost6k_{ein}_bps") == r.get(f"rb_cost6k_{ein}_bps") and r.get(f"rb_cost6k_{ein}_bps") is not None:
        eslip, eslip_src = max(0.0, r[f"rb_cost6k_{ein}_bps"]) / 100.0, "route_b_book"
    elif r.get("journal_model_slip_pct") == r.get("journal_model_slip_pct") and r.get("journal_model_slip_pct") is not None:
        eslip, eslip_src = float(r["journal_model_slip_pct"]), "journal_model"
    else:
        eslip, eslip_src = C.DEFAULT_SLIP_PCT, "default"
    fund, fund_ok = SRC.funding_history(r["pair"], e_ms, e_ms + spec["hold_min"] * C.MIN_MS, now_ms)
    pk = k.iloc[i0:]
    w = X.walk(pk.o.to_numpy(), pk.h.to_numpy(), pk.l.to_numpy(), pk.c.to_numpy(), pk.open_ms.to_numpy(), e, side, tp, sl, spec["hold_min"],
               funding=fund, stop_counts_funding=spec["stop_counts_funding"], entry_slip_pct=eslip, exit_slip_pct=0.0, entry_ms=e_ms)
    xslip, xslip_src = 0.0, "maker_tp"
    if w["reason"] not in ("TP", "NO_PATH", "NO_LEVELS"):
        xs = book_cost_at(coin, int(w["exit_ms"]), eout) if w["exit_ms"] == w["exit_ms"] else math.nan
        if xs == xs:
            xslip, xslip_src = max(0.0, xs) / 100.0, "route_b_book_at_exit"
        elif r.get(f"rb_cost6k_{eout}_bps") == r.get(f"rb_cost6k_{eout}_bps") and r.get(f"rb_cost6k_{eout}_bps") is not None:
            xslip, xslip_src = max(0.0, r[f"rb_cost6k_{eout}_bps"]) / 100.0, "route_b_book_at_signal"
        else:
            xslip, xslip_src = C.DEFAULT_SLIP_PCT, "default"
    net = w["net_pct"] - xslip if w["net_pct"] == w["net_pct"] else math.nan
    if w["exit_ms"] == w["exit_ms"] and w["exit_ms"] > now_ms - SRC.FUNDING_PUBLISH_LAG_MS:
        fund_ok = False                                               # a settlement inside the last minutes may not be published yet
    slip_pending = xslip_src in ("route_b_book_at_signal", "default") and w["exit_ms"] == w["exit_ms"] and w["exit_ms"] >= data_end_ms
    final = bool(w["final"]) and bool(fund_ok) and not unlinked and not slip_pending   # funding / fill link / exit book still to come → not final
    why = ("" if final else ("taken, fill not in an orders export yet — priced from the next-minute open, re-priced each run" if unlinked
                             else "funding history not fetched" if (w["final"] and not fund_ok)
                             else "exit after the Route B data end (exit book not recorded yet)" if (w["final"] and slip_pending)
                             else "path not finished" if w["reason"] == "CENSORED" else w["reason"]))
    o.update(out_priced=w["reason"] not in ("NO_PATH", "NO_LEVELS"), out_why=why,
             out_entry_src=e_src, out_entry_px=e, out_entry_ms=e_ms, out_tp_pct=tp, out_sl_pct=sl, out_atr_pct=atr, out_atr_src=atr_src,
             out_hold_min=spec["hold_min"], out_reason=w["reason"], out_final=final, out_path_final=w["final"], out_exit_px=w["exit_px"], out_exit_ms=w["exit_ms"],
             out_minutes=w["minutes"], out_gross_pct=w["gross_pct"], out_fees_pct=w["fees_pct"], out_entry_slip_pct=eslip,
             out_entry_slip_src=eslip_src, out_exit_slip_pct=xslip, out_exit_slip_src=xslip_src, out_funding_pct=w["funding_pct"],
             out_funding_ok=fund_ok, out_net_pct=net, out_mfe_pct=w["mfe_pct"], out_mae_pct=w["mae_pct"], out_min_to_mfe=w["min_to_mfe"],
             out_min_to_mae=w["min_to_mae"], out_min_to_exit=w["minutes"], out_fill_minute=fill_minute, out_data_end_ms=data_end_ms)
    if spec["kind"] == "atr" and atr_src == "journal":
        o["out_atr_check_computed"] = atr_from_klines(k, T)
    return o


def klines_for(sym, a_ms, b_ms, now_ms):
    """the exact [a, b) slice for THIS call (whole days are cached per symbol inside sources.klines)."""
    return SRC.klines(sym, a_ms, b_ms, now_ms)


def carry_key(spec, fill_key, status):
    """what a carried-over outcome depends on besides the signal itself: the sleeve's exit spec + costs (hash), the linked fill, the status."""
    return f"{C.spec_hash(spec)}|{fill_key if isinstance(fill_key, str) else ''}|{status}"


PERMANENT_WHY = tuple(C.NOT_PRICED_WHY.values()) + ("no entry decision (no entry bar journaled)", "no exit for this sleeve")


def _true(x):
    """True only for a real boolean True (NaN / None / strings are not)."""
    return isinstance(x, (bool, np.bool_)) and bool(x)


def can_carry(prev, cur_key):
    """carry a previous outcome only when code version, exit spec / costs, fill link and status match AND either ① it was priced, final
    and its funding read (real True values — NaN never counts), with no fallback exit slip past the then data end, or ② it is unpriced
    for a PERMANENT reason (sleeve off, no entry bar by design). Temporary reasons (no klines yet, entry minute not in klines, levels
    undefined, path not finished) are always re-tried. prev = dict-like of the previous row. Pure."""
    try:
        if int(prev.get("code_version", -1)) != CODE_VERSION or prev.get("carry_key") != cur_key:
            return False
    except (TypeError, ValueError):
        return False
    if _true(prev.get("out_priced")):
        if not (_true(prev.get("out_final")) and _true(prev.get("out_funding_ok"))):
            return False
        xm, de = prev.get("out_exit_ms"), prev.get("out_data_end_ms")
        if prev.get("out_exit_slip_src") in ("route_b_book_at_signal", "default") and (de is None or de != de or not (xm < de)):
            return False
        return True
    return str(prev.get("out_why") or "") in PERMANENT_WHY


def fb_tracking_episodes(J):
    """FLATBASE episodes (pair, spike) whose LAST journaled status is TRACKING (no final status yet)."""
    if not len(J) or "fb_status" not in J.columns:
        return 0
    fb = J[J.e == "FLATBASE"].sort_values("t")
    return int((fb.groupby(["pair", "fb_spike_at"], dropna=False).tail(1).fb_status == "TRACKING").sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    t_run = time.time()
    C.ensure_dirs()
    now_ms = int(time.time() * 1000)
    th = C.load_thresholds()
    specs = C.exit_specs(th)
    # ── Route B minutes
    keys = SRC.feature_keys()
    if not keys:
        sys.exit("no Route B summaries listed on S3 (aws credentials / network?)")
    book, mark, liq, n_hours = SRC.load_minutes(keys)
    data_end_ms = int(max(book.minute.max(), mark.minute.max())) + C.MIN_MS
    print(f"Route B: {n_hours} hours · {book.sym.nunique()} book coins · {mark.sym.nunique()} mark coins · data to "
          f"{pd.Timestamp(data_end_ms, unit='ms', tz='UTC')}")
    # ── journal → signals
    J, spans = SRC.load_journal(C.REC_START - pd.Timedelta(hours=2))
    journal_end = str(J.t.max()) if len(J) else ""
    S = SG.build_signals(J, turnover_cap=float(th.get("frenzy_willy_max_vol_mcap_ratio", 1.0) or 1.0))
    O, n_probe = SRC.load_orders()
    O = O[pd.to_datetime(O.opened_at.astype(str).str[:19], utc=True) >= C.REC_START - pd.Timedelta(minutes=30)]
    S, n_orders_only = SG.link_fills(S, O)
    S.loc[(S.status == "OPENED") & S.fill_key.isna(), "status"] = "OPENED_UNLINKED"
    S = S[(S.signal_time >= C.REC_START) & (S.signal_time.map(lambda x: x.value // 1_000_000) < data_end_ms)].reset_index(drop=True)
    S["signal_id"] = S.sleeve + "|" + S.pair + "|" + S.direction + "|" + S.signal_time.dt.strftime("%Y-%m-%dT%H:%M:%S") + "|" + S.status
    S["hold_min"] = S.sleeve.map(lambda s: specs[s]["hold_min"] if specs.get(s) else (int(th.get("frenzy_willy_max_hold_minutes", 90)) if s == "FRENZY_WILLY" else np.nan))
    # ── coin objects + market row
    trig_syms = set(book.sym) | set(mark[mark.funding.abs() >= 0.001].sym)
    coins = ST.coins_from(book, mark, liq, sorted(set(S.pair) | trig_syms))
    market = ST.Market(mark, liq, book[book.sym == "BTCUSDT"])
    st_rows = []
    for r in S.itertuples():
        T = r.signal_time.value // 1_000_000
        c = coins.get(r.pair)
        st_rows.append({**(c.state(T) if c is not None else {"rb_has_book": False, "rb_has_mark": False}), **market.state(T)})
    S = pd.concat([S, pd.DataFrame(st_rows)], axis=1)
    # ── outcomes (carry FINAL rows from the previous run)
    prev_p = os.path.join(C.DATA, "signals.pkl.gz")
    prev = pd.read_pickle(prev_p) if (os.path.exists(prev_p) and not a.full) else None
    carry = {}
    if prev is not None and "carry_key" in prev.columns:
        carry = {r["signal_id"]: r for r in prev.to_dict("records")}
    S["carry_key"] = [carry_key(specs.get(r.sleeve), r.fill_key, r.status) for r in S.itertuples()]
    outs, n_carried = [], 0
    for _, r in S.iterrows():
        pr = carry.get(r.signal_id)
        if pr is not None and can_carry(pr, r.carry_key):
            outs.append({c: pr[c] for c in pr if c.startswith("out_")})
            n_carried += 1
            continue
        spec = specs.get(r.sleeve)
        k = None
        if spec is not None and r.priceable:
            T = r.signal_time.value // 1_000_000
            k = klines_for(r.pair, T - 30 * 3600_000, T + (spec["hold_min"] + 60) * C.MIN_MS, now_ms)
        outs.append(price_signal(r, spec, k, coins.get(r.pair), now_ms, data_end_ms))
    S = pd.concat([S, pd.DataFrame(outs)], axis=1)
    S["code_version"] = CODE_VERSION
    S["exit_ms_for_poss"] = S.get("out_exit_ms")
    S = SG.possibilities(S, exit_ms_col="exit_ms_for_poss", hold_col="hold_min").drop(columns=["exit_ms_for_poss"])
    S["poss_note"] = ""
    S = SG.pair_held_links(S)                    # PAIR_HELD = capacity: joins the holding position's possibility (any sleeve)
    S = SG.gate_audit(S, th)                     # also_blocked_by · filters_unevaluated · sole_block · block_class
    S = S.sort_values("signal_time").reset_index(drop=True)
    S.to_pickle(prev_p)
    S.assign(signal_time=S.signal_time.dt.strftime("%Y-%m-%dT%H:%M:%S")).to_csv(os.path.join(C.DATA, "signals.csv.gz"), index=False)
    print(f"signals: {len(S)} rows ({int(S.get('out_priced', pd.Series(dtype=bool)).fillna(False).sum())} priced, {n_carried} carried over)")
    # ── H1–H6 triggers (outcomes stored, never printed)
    Tt = TR.detect(book, mark, coins)
    Tt = Tt[Tt.signal_time >= C.REC_START].reset_index(drop=True)
    Tt["trigger_id"] = Tt.trigger + "|" + Tt.pair + "|" + Tt.signal_time.dt.strftime("%Y-%m-%dT%H:%M")
    tprev_p = os.path.join(C.DATA, "triggers_H.pkl.gz")
    tprev = pd.read_pickle(tprev_p) if (os.path.exists(tprev_p) and not a.full) else None
    hkey = C.spec_hash({"grid": "tp1-5 x sl1-5 / 240 min", "fwd": [5, 15, 60, 240]})
    tcarry = {}
    if tprev is not None and "h_carry_key" in tprev.columns:
        tcarry = {r.trigger_id: r for r in tprev[(tprev.code_version == CODE_VERSION) & (tprev.h_final == True)   # noqa: E712
                                                  & (tprev.h_carry_key == hkey)].itertuples()}
    rows = []
    for r in Tt.itertuples():
        T = r.signal_time.value // 1_000_000
        c = coins.get(r.pair)
        st = {**(c.state(T) if c is not None else {}), **market.state(T)}
        if r.trigger_id in tcarry:
            pr = tcarry[r.trigger_id]
            rows.append({**st, **{cc: getattr(pr, cc) for cc in tprev.columns if cc.startswith(("fwd_", "g_", "h_"))}})
            continue
        side = 1 if r.direction == "LONG" else -1
        k = klines_for(r.pair, T - 3600_000, T + 300 * C.MIN_MS, now_ms)
        res = {"h_final": False, "h_entry_px": math.nan, "h_carry_key": hkey}
        if k is not None and len(k):
            i0 = entry_index(k, T)
            pk = k.iloc[i0:i0 + 240]
            if len(pk):
                ein, eout = ("buy", "sell") if side > 0 else ("sell", "buy")
                es = st.get(f"rb_cost6k_{ein}_bps")
                xs = st.get(f"rb_cost6k_{eout}_bps")
                es = es / 100 if (es is not None and es == es) else C.DEFAULT_SLIP_PCT
                xs = xs / 100 if (xs is not None and xs == xs) else C.DEFAULT_SLIP_PCT
                e_ms = int(pk.open_ms.iloc[0])
                fund, fok = SRC.funding_history(r.pair, e_ms, e_ms + 240 * C.MIN_MS, now_ms)
                res.update(X.forward_returns(pk.o.to_numpy(), pk.c.to_numpy(), side, t=pk.open_ms.to_numpy(), funding=fund))
                res.update(X.grid(pk.o.to_numpy(), pk.h.to_numpy(), pk.l.to_numpy(), pk.c.to_numpy(), pk.open_ms.to_numpy(), side,
                                  entry_slip_pct=max(es, 0), exit_slip_pct=max(xs, 0), funding=fund))
                res.update(h_final=(len(pk) >= 240) and bool(fok), h_funding_ok=bool(fok), h_entry_px=float(pk.o.iloc[0]), h_entry_ms=e_ms,
                           h_slip_entry_pct=es, h_slip_exit_pct=xs,
                           h_funding_240_pct=X.funding_pct(fund, side, float(pk.o.iloc[0]), e_ms, e_ms + 240 * C.MIN_MS))
        rows.append({**st, **res})
    Tt = pd.concat([Tt, pd.DataFrame(rows)], axis=1)
    Tt["code_version"] = CODE_VERSION
    Tt.to_pickle(tprev_p)
    Tt.assign(signal_time=Tt.signal_time.dt.strftime("%Y-%m-%dT%H:%M:%S")).to_csv(os.path.join(C.DATA, "triggers_H.csv.gz"), index=False)
    print(f"H1–H6 triggers: {len(Tt)} openers (outcomes stored, not printed)")
    meta = dict(run_utc=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), code_version=CODE_VERSION, s3_hours=n_hours,
                data_end=str(pd.Timestamp(data_end_ms, unit="ms", tz="UTC")), journal_end=journal_end, journal_files=spans,
                orders_only_fills=n_orders_only, probe_fills_dropped=n_probe, exh_wide_read_cap=int((J.gate == "EXH_WIDE_READ_CAP").sum()) if len(J) else 0,
                fb_tracking_only=fb_tracking_episodes(J),
                binance_stopped=SRC.limiter().stopped, runtime_s=round(time.time() - t_run, 1),
                exits={k: v for k, v in specs.items()})
    with open(os.path.join(C.DATA, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    if not a.no_report:
        import report as RP
        p = RP.write(S, Tt, J, meta, book, mark)
        print("report →", os.path.relpath(p, C.ROOT))


if __name__ == "__main__":
    main()
