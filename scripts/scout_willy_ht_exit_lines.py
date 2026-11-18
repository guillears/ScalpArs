#!/usr/bin/env python3
"""🎯👁 Scout — WILLY HT take profit +2 % (SHIPPED in paper, DECISION_LOG 537 declared override) revert gate + the RUN15 watch line (observe
only — never changes config, no bot API; public 1m klines only, budgeted).
⛔ RETIRED 2026-11-18 (DECISION_LOG 587): HT's take profit moved +2 → +5 (stop −4) — every line here priced the +2 exit; unhooked from the
opportunity scout and classified NONE in observe_registry. Kept as a library / history; re-register a line on the new exit if wanted.

WHY        INIT HT 11-12 closed at +1.25 in 38 s, then ran to +6.64 % in 10 min (≈ $1,000 left); ONE HT the same day ran +2.5 then −4.2. Study
           reports/WILLY_HT_EXIT_STUDY_2026-11-12.md: TP2 kept master HT 8 fills +3.0 pts (4 better / 0 worse) + B26 3 fills +2.25, luck p 0.64,
           WILLY backtest −15.7 (19 of 609 reach +1.25 then stop, mostly Jul–Sep 2026); RUN15 +3.5 but one fill = 76–112 % of the gain.
           Operator 11-12: "switch HT to 2 %" — frenzy_willy_ht_tp_pct 2.0 (T_used = max(2, the spike-high target)); WILLY unchanged.
PRICED     every HT fill on its OWN 1m path with the study's walker (scripts/study_willy_ht_exit_1112.leg — the same ruler), from the V2b target
           rebuilt from the spike high (T_old = build_willy_exit_cf.resolve_target T_calc, the pre-537 exit) and T_new = max(T_old, HT TP):
             OLD125  the pre-537 exit at T_old (what +1.25 would have booked)      NEW2  the live exit at T_new (+2 / the spike target)
             RUN15   at the first touch of T_old: trail = highest net since − 1.5 × entry ATR %, −4 floor, 90-min cap
COHORT     CLOSED full-size FRENZY_WILLY_HT fills (*_PROBE out, operator closes out) opened ≥ the DL 537 commit + 10 min, from the orders exports
           (export_glob, dedup newest wins). A fill whose 90-min path is not complete is PENDING. The 3 B26 HT fills before the ship (KGEN · ONE ·
           INIT) are shown as context, never counted.
🔒 GATE    (pre-committed 2026-11-12, never re-fit; Δ = NEW2 − OLD125 per fill, one size %, open-time order, decided ONCE)
           TP2 (shipped): EARLY STOP — the first 8 fills Σ Δ < 0 → REVERT (frenzy_willy_ht_tp_pct → 0, operator applies) · at 15 fills Σ Δ ≤ 0 →
           REVERT · else KEEP (a go-live is a separate decision).
           RUN15 (observe): at 15 fills, Δ = RUN15 − NEW2: Σ Δ > 0 ∧ day-block bootstrap 95 % CI low > 0 ∧ day sign-flip p ≤ 0.10 → OPERATOR CALL
           · else NO CHANGE. Both wait while an earlier fill is pending and need ≥ 2 days. Frozen in reports/SCOUT_WILLY_HT_EXIT_LINES.json.
🛑 STOP    (DECISION_LOG 546, operator 2026-11-14 — OBSERVE only, pre-registered, never re-fit) would a WIDER HT stop pay? Study
LINES      reports/STOP_WIDTH_KEPT_2026-11-14.md: on today's-filter kept HT fills both stop-outs recovered — ICNT (B25) at a −6 stop, EVAA (B26)
           only at −10 — 2 fills = watch only. Lines STOP6 / STOP8 / STOP10 = today's HT exit (TP T_new, late +0.25 from 30, cap 90) with the
           stop at −6 / −8 / −10 instead of −4, walked on each fill's path. COHORT = forward HT fills opened ≥ the DL 546 commit + 10 min (EVAA
           / ICNT are the motivating fills — never counted). Δ_i = variant − NEW2 ONLY on fills that stopped LIVE (close_reason STOP_LOSS); a
           fill live did not stop counts Δ 0 (CLAUDE.md exit/stop CF rule — a replica-only stop is a wick live rode through). RISK-EQUAL
           Δeq_i = variant_i × 4 / s − NEW2_i on EVERY fill (a wider stop must ship at size × 4 / s to keep one stop ≈ 7.8 % of the account).
           🔒 decided ONCE per line when ≥ 6 live-stopped HT fills on ≥ 4 days are counted and no earlier fill is pending: Σ Δ > 0 ∧
           day-block bootstrap 95 % CI low > 0 ∧ Σ Δeq > 0 → OPERATOR CALL (a ship = size × 4 / s + its own revert gate) · else NO CHANGE.
🛑🧱 BLOCKED (DECISION_LOG 548, operator 2026-11-14 — OBSERVE only, a SEPARATE line, never pooled) the same −6 / −8 / −10 lines on HT
STOP LINES signals an armed filter BLOCKED (decisions-CSV e=BLOCKED, refused ≥ the 548 commit + 10 min), priced by the blocked-trade pricer
           (scout_blocked_filters.price_frenzy: refusal + 8 s, 0.10 % slip, taker, today's HT exit with the stop moved; one shadow per pair).
           'Stopped' = the −4 walk's STOP (no live fill — a wick live might ride counts, tilts wider). Same frozen bar as above, own state keys
           BLK_STOP6 / 8 / 10; a pass = SUPPORTING evidence beside the kept-fill line. In-sample at registration (never counted): 17 HT −4 stops
           incl. blocked, B25 + B26 → Σ Δ −6 +4.9 · −8 +7.3 · −10 +32.4 pts.
PATHS      the study's 1m cache + this scout's own reports/backtest_cache/scout_willy_ht_exit/1m (public klines, ≤ MAX_CALLS / run, stop at
           X-MBX-USED-WEIGHT-1M ≥ 800, the shared Binance cooldown honoured and armed on 418 / 429).
Usage: venv/bin/python -I scripts/scout_willy_ht_exit_lines.py [--no-fetch]
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(_D)
for _p in (_D, _R):
    _p in sys.path or sys.path.insert(0, _p)
import export_glob as _EG                  # noqa: E402
import scout_gatekit as _GK                # noqa: E402
import study_b19_fzexit_common as _FX      # noqa: E402  (the shared 1m / tick loader the study walks on)

_TRUE_LOAD_1M = _FX.WT.load_1m             # review 537: captured BEFORE the study import patches it — restored after every run
import study_willy_ht_exit_1112 as S       # noqa: E402  (the study's walker, path loader and stats — the same ruler)
S.FX.WT.load_1m = _TRUE_LOAD_1M            # review 537: the study's import-time patch is undone; this scout patches only inside run()

DL = 537
N_GATE = 15
LINES = ("TP2", "RUN15")          # TP2 = the shipped exit's revert gate · RUN15 = the observe line
EARLY_N = 8
HT_TP_SHIPPED = 2.0                # review 537: the gate measures the SHIPPED rule (a later revert to 0 never re-prices it) — registry-pinned
BASE_TP_PRE537 = 1.25              # WILLY's base TP the pre-537 HT exit ran (frenzy_willy_tp_pct, registry-pinned)
CTX_KEYS = ("2026-11-11T23:50:07|KGENUSDT", "2026-11-12T08:20:10|ONEUSDT", "2026-11-12T09:00:09|INITUSDT")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
STATE = os.path.join(_R, "reports", "SCOUT_WILLY_HT_EXIT_LINES.json")
K1_DIR = os.path.join(_R, "reports", "backtest_cache", "scout_willy_ht_exit", "1m")
MAX_CALLS, WEIGHT_STOP = 20, 800
DL_STOP = 546
STOP_LINES = (6, 8, 10)            # 🛑 (546) the wider HT stops watched (today −4 = frenzy_willy_stop_pct, registry-pinned)
STOP_TODAY = 4.0
STOP_MIN_STOPPED, STOP_MIN_DAYS = 6, 4
LIVE_STOP_REASONS = ("STOP_LOSS", "BACKSTOP_STOP", "BACKSTOP")   # review 546: a live exchange-backstop stop is a live stop too
HT_ACCT_PER_PCT = 1.95             # today's HT one size: % of the account per 1 % move (1.0 × 9,750 / 4 / 10,000 × 8×) — report text only
DL_BLK = 548                       # 🛑🧱 (548) the same stop lines on HT signals an armed filter BLOCKED — a SEPARATE line, own gate
BLK_PREFIX = "BLK_"                # its frozen verdicts: state keys BLK_STOP6 / BLK_STOP8 / BLK_STOP10
MIN = 60_000

def _load_1m(pair):
    """the shared 1m bars + the study's and this scout's own caches (installed only for the duration of run())."""
    k = _TRUE_LOAD_1M(pair)
    for f in (os.path.join(S.K1_EXTRA, f"{pair}.csv"), os.path.join(K1_DIR, f"{pair}.csv")):
        if os.path.exists(f):
            x = S.FX.WT._read_k1(f)
            k = pd.concat([k, x]).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    S.FX.WT._K1[pair] = k
    return k


def _have(pair, a_ms, b_ms):
    """review 537: EVERY minute of [a, b − 1m] cached (a gap between two older fetches is not coverage)."""
    have = set()
    for g in (os.path.join(K1_DIR, f"{pair}.csv"), os.path.join(S.K1_EXTRA, f"{pair}.csv")):
        if os.path.exists(g):
            have |= set(S.FX.WT._read_k1(g).open_time.astype("int64"))
    a0 = -(-int(a_ms) // MIN) * MIN
    return all(t in have for t in range(a0, int(b_ms) - MIN + 1, MIN))


def fetch(fills, now_ms):
    """public 1m klines for each fill's [entry − 5 min, entry + 95 min] not yet cached. → note."""
    try:
        import binance_cooldown as BC
        if BC.active(now_ms=now_ms):
            return "shared Binance cooldown active — no fetch"
    except Exception:
        BC = None
    n = 0
    os.makedirs(K1_DIR, exist_ok=True)
    for r in fills:
        te = S.CF._ms(r.get("opened_at"))
        if te is None or te + 95 * MIN > now_ms:
            continue
        a, b = te - 5 * MIN, te + 95 * MIN
        if _have(r["pair"], a, b):
            continue
        if n >= MAX_CALLS:
            return f"budget {MAX_CALLS} calls reached"
        u = f"https://fapi.binance.com/fapi/v1/klines?symbol={r['pair']}&interval=1m&startTime={a}&endTime={b}&limit=110"
        try:
            with urllib.request.urlopen(u, timeout=15) as resp:
                n += 1
                rows = json.load(resp)
                w = int(resp.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                try:
                    import scout_ml_matched_refusals as _MM   # the shared cooldown writer (never shortens another script's cooldown)
                    _MM.arm_cooldown(600_000 if e.code == 418 else 60_000, err=e, now_ms=now_ms)   # review 537: 418 = an IP ban (minutes–hours)
                except Exception:
                    pass
            if e.code in (418, 429):
                return f"HTTP {e.code} — stopped"
            n += 1
            continue                                           # review 537: a delisted / bad symbol (4xx) skips that pair only
        except Exception as e:
            return f"fetch failed ({str(e)[:60]})"
        new = pd.DataFrame([[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in rows if int(k[0]) + MIN <= now_ms],
                           columns=["open_time", "o", "h", "l", "c"])          # review 537: a still-forming candle is never cached
        f = os.path.join(K1_DIR, f"{r['pair']}.csv")
        old = S.FX.WT._read_k1(f) if os.path.exists(f) else new.iloc[0:0]
        pd.concat([old, new]).drop_duplicates("open_time", keep="last").sort_values("open_time").to_csv(f, index=False)
        S.FX.WT._K1.pop(r["pair"], None)
        if w >= WEIGHT_STOP:
            return f"weight {w} ≥ {WEIGHT_STOP} — stopped"
        time.sleep(0.2)
    return f"public kline calls {n}"


def load(paths=None):
    fr = []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame()
    d = pd.concat(fr, ignore_index=True)
    d = d.reindex(columns=sorted(set(d.columns) | {"cell_multiplier_source", "close_reason"}))
    d = d[(d.entry_strategy.astype(str) == "FRENZY_WILLY_HT") & d.status.astype(str).str.upper().eq("CLOSED")]
    d = d[~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")]
    if len(d):
        d = d[~np.array([_GK.is_operator_closed(s, r) for s, r in zip(d.entry_strategy, d.close_reason)], dtype=bool)]
    d = d.sort_values("_m").drop_duplicates(["opened_at", "pair", "entry_strategy"], keep="last")
    return d.reset_index(drop=True)


def gate(D, days, held=False):
    """D = DataFrame with columns TP2 (Δ = NEW2 − OLD125) and RUN15 (Δ = RUN15 − NEW2; NaN = unpriced, e.g. no ATR stamp), open-time order,
    days aligned; held = an earlier forward fill still pending → {line: (verdict, detail)}. Pure."""
    out = {}
    for c in LINES:
        ok_rows = D[c].notna().to_numpy()
        x, dx = D[c].to_numpy()[ok_rows], [d for d, k in zip(days, ok_rows) if k]
        if c == "TP2":
            if len(x) >= EARLY_N and not held and x[:EARLY_N].sum() < 0:
                out[c] = ("REVERT", f"early stop: the first {EARLY_N} HT fills Σ Δ {x[:EARLY_N].sum():+.2f} pts < 0 (the +1.25 exit would have made more)")
                continue
            if len(x) >= N_GATE and not held:
                s_ = float(x[:N_GATE].sum())
                out[c] = (("KEEP" if s_ > 0 else "REVERT"), f"the first {N_GATE} HT fills Σ Δ {s_:+.2f} pts ({'> 0' if s_ > 0 else '≤ 0'})")
                continue
            out[c] = ("OBSERVING", ("an earlier fill is still pending · " if held else "") + f"{len(x)} of {N_GATE} · Σ Δ {x.sum():+.2f} pts"
                      + (f" (early stop read at {EARLY_N})" if len(x) < EARLY_N else ""))
            continue
        if len(x) < N_GATE or held or len(set(dx[:N_GATE])) < 2:
            why = ("an earlier fill is still pending" if held and len(x) >= N_GATE else
                   "the first 15 fall on one day — waits for a 2nd day" if len(x) >= N_GATE else f"{len(x)} of {N_GATE} priced HT fills")
            out[c] = ("OBSERVING", f"{why} · Σ Δ {x.sum():+.2f} pts" if len(x) else why)
            continue
        q, dq = x[:N_GATE], dx[:N_GATE]
        pc = S.luck_p(pd.DataFrame({c: q}), dq)[c]
        lo, hi = S.boot_ci(q, dq)
        s_ = float(q.sum())
        ok = s_ > 0 and lo == lo and lo > 0 and pc <= 0.10
        out[c] = (("OPERATOR CALL" if ok else "NO CHANGE"), f"first {N_GATE}: Σ Δ {s_:+.2f} pts · 95 % CI [{lo:+.2f}, {hi:+.2f}] · sign-flip p {pc:.2f}")
    return out


def _leg_stop(P, tp, stop):
    """the study's walker with its stop at −stop (S.STOP is a module constant — set and always restored)."""
    old = S.STOP
    try:
        S.STOP = float(stop)
        return S.leg(P, tp)
    finally:
        S.STOP = old


def stop_cut(F, days):
    """🛑 (546) the frozen cohort: the shortest open-time prefix holding ≥ STOP_MIN_STOPPED live-stopped fills on ≥ STOP_MIN_DAYS days → its
    length, or None while not reached (review 546: the decision never depends on how often the scout runs). Pure."""
    ls = F.live_stop.fillna(False).astype(bool).to_numpy() if len(F) else np.array([], bool)
    seen = set()
    for i, (k, dd) in enumerate(zip(ls, days)):
        if k:
            seen.add(dd)
            if int(ls[:i + 1].sum()) >= STOP_MIN_STOPPED and len(seen) >= STOP_MIN_DAYS:
                return i + 1
    return None


def gate_stop(F, days, held=False, pend=()):
    """🛑 (546) F = per counted fill (open-time order): live_stop (bool), NEW2, STOP6 / STOP8 / STOP10 (net %), opened_at (optional); days
    aligned; pend = opened_at of forward fills whose path is still filling in → {line: (verdict, detail, Σ Δ, Σ Δeq)}. Decided on the
    frozen prefix (stop_cut) only; held while a pending fill opened before its last fill. Pure."""
    out = {}
    n = stop_cut(F, days)
    if n is not None:
        F, days = F.iloc[:n], list(days)[:n]
        if "opened_at" in F and any(str(p) < str(F.opened_at.iloc[-1]) for p in pend):
            held = True
    ls = F.live_stop.fillna(False).astype(bool).to_numpy() if len(F) else np.array([], bool)
    n_st = int(ls.sum()); d_st = len({d for d, k in zip(days, ls) if k})
    for s_ in STOP_LINES:
        c = f"STOP{s_}"
        v = F[c].to_numpy(float) if len(F) else np.array([])
        nw = F.NEW2.to_numpy(float) if len(F) else np.array([])
        dl = np.where(ls, v - nw, 0.0)
        deq = np.where(ls, v, nw) * (STOP_TODAY / s_) - nw
        sd, se = float(np.nansum(dl)), float(np.nansum(deq))
        if n is None or held:
            why = ("an earlier fill is still pending · " if held else "") + f"{n_st} of {STOP_MIN_STOPPED} live-stopped fills on {d_st} of {STOP_MIN_DAYS} days"
            out[c] = ("OBSERVING", f"{why} · Σ Δ {sd:+.2f} pts · risk-equal Σ Δ {se:+.2f}", sd, se)
            continue
        lo, hi = S.boot_ci(dl, days)
        ok = sd > 0 and lo == lo and lo > 0 and se > 0
        out[c] = (("OPERATOR CALL" if ok else "NO CHANGE"), f"{len(F)} fills · {n_st} live-stopped · Σ Δ {sd:+.2f} pts · 95 % CI [{lo:+.2f}, {hi:+.2f}] · "
                  f"risk-equal (size × {STOP_TODAY / s_:.2f}) Σ Δ {se:+.2f}", sd, se)
    return out


def price(use):
    """→ DataFrame per fill: T_old / T_new, OLD125 · NEW2 · RUN15 (net %, NaN = unpriced) + why, booked, status. The study's walker."""
    rows, cache, tcache = [], {}, {}
    ht_tp = HT_TP_SHIPPED
    for r in use.to_dict("records"):
        base = dict(key=S._key(r.get("opened_at"), r.get("pair")), opened_at=str(r.get("opened_at"))[:19], pair=r.get("pair"),
                    booked=S.CF._num(r.get("pnl_percentage")), atr=S.CF._num(r.get("entry_atr_pct")), stamped_T=S.CF._num(r.get("frenzy_willy_tp_target_pct")))
        te, E = S.CF._ms(r.get("opened_at")), S.CF._num(r.get("entry_price"))
        if te is None or not E:
            rows.append(dict(base, status="no entry")); continue
        P = S.FX.load_path(r.get("pair"), te, E, S.CAP, cache)
        if not len(P["t"]):
            rows.append(dict(base, status="no path")); continue
        tg = S.CF.resolve_target(S.TH, r, tcache)
        t_src = "rebuilt"
        if tg["T_calc"] is not None:
            t_old = tg["T_calc"]
        elif base["stamped_T"] is not None and base["stamped_T"] > ht_tp + 1e-9:
            t_old, t_src = base["stamped_T"], "stamp"           # a spike target above the HT TP: identical under both exits
        elif base["stamped_T"] is not None and base["stamped_T"] < ht_tp - 1e-9:
            t_old, t_src = base["stamped_T"], "stamp"           # a pre-537 row (context)
        else:
            t_old, t_src = BASE_TP_PRE537, "assumed"            # review 537: no spike high and stamped = the HT TP → pre-537 target unknowable
        t_new = max(t_old, ht_tp) if ht_tp else t_old
        old, new = S.leg(P, t_old), S.leg(P, t_new)
        run = S.leg(P, t_old, run_k=1.5, atr=base["atr"]) if (base["atr"] and base["atr"] > 0) else (float("nan"), "NO_ATR")
        st = "UNRES" if (old[1] == "UNRES" or new[1] == "UNRES") else "priced"
        st = "T_UNKNOWN" if (st == "priced" and t_src == "assumed") else st   # never counted (it would bias Δ toward +2)
        wide = {f"STOP{s_}": _leg_stop(P, t_new, s_) for s_ in STOP_LINES}       # 🛑 (546)
        rows.append(dict(base, status=st, T_old=t_old, T_new=t_new, T_src=t_src, OLD125=old[0], OLD125_why=old[1], NEW2=new[0], NEW2_why=new[1],
                         RUN15=(run[0] if run[1] != "UNRES" else float("nan")), RUN15_why=run[1],
                         live_stop=str(r.get("close_reason")).upper().split(":")[0].strip() in LIVE_STOP_REASONS,
                         **{k: (v[0] if v[1] != "UNRES" else float("nan")) for k, v in wide.items()}, **{k + "_why": v[1] for k, v in wide.items()}))
    return pd.DataFrame(rows)


def _stop_section(R, d, st, state_path):
    """🛑 (546) the wider-stop watch lines → report lines; verdicts frozen in the same state file (keys STOP6 / STOP8 / STOP10)."""
    L = ["", "### 🛑 HT wider-stop watch lines (DECISION_LOG 546, observe only)"]
    fl = _GK.dl_deploy_ms(DL_STOP, label="WILLY_HT_STOP_LINES")
    if fl is None:
        return L + ["Not evaluable yet: the DECISION_LOG 546 commit is not in git history (counted from its commit + 10 min).",
                    "Motivating fills (never counted): ICNT B25 11-08 −4.05 (−6 → +0.25) · EVAA B26 11-14 −4.33 (−10 → +8.00)."]
    o_ms = dict(zip(d.k, [S.CF._ms(x) for x in d.opened_at]))
    R = R.assign(fwd2=[(o_ms.get(k) is not None and o_ms.get(k) >= fl) for k in R.key])
    tu = R.status.eq("T_UNKNOWN") & (pd.to_numeric(R.get("stamped_T"), errors="coerce") - HT_TP_SHIPPED).abs().lt(1e-6)
    ok_st = R.status.eq("priced") | tu                      # review 546: T_new = +2 is known when the row stamped the HT TP (not when unstamped)
    F = R[R.fwd2 & ok_st].sort_values("opened_at").copy()
    for c in [f"STOP{s_}" for s_ in STOP_LINES]:
        if c not in F:
            F[c] = np.nan
    F["live_stop"] = F["live_stop"].fillna(False).astype(bool) if "live_stop" in F else False
    wide_unres = F[~F[[f"STOP{s_}" for s_ in STOP_LINES]].notna().all(axis=1)]   # review 546: a wider walk still filling in = pending
    F = F[F[[f"STOP{s_}" for s_ in STOP_LINES]].notna().all(axis=1)]
    pend = R[R.fwd2 & ~ok_st]
    pu = pend[pend.status.eq("UNRES")]
    days = list(pd.to_datetime(F.opened_at).dt.strftime("%Y-%m-%d"))
    g = gate_stop(F, days, pend=list(pu.opened_at) + list(wide_unres.opened_at))
    changed = False
    for c, (v, det, _, _) in list(g.items()):
        if c in st:
            g[c] = (st[c]["verdict"], st[c]["detail"] + f" (frozen {st[c]['at']})", None, None)
        elif v != "OBSERVING":
            st[c] = dict(verdict=v, detail=det, at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")); changed = True
    if changed and state_path:
        try:
            _save_state(st, state_path)
        except Exception:
            L.append("⚠ stop-line state not saved this run.")
    L.append(f"Forward since {pd.Timestamp(int(fl), unit='ms'):%Y-%m-%d %H:%M} UTC: {len(F)} priced HT fills, {int(F.live_stop.astype(bool).sum())} "
             f"stopped live · pending {len(pend) + len(wide_unres)}. Motivating fills (never counted): ICNT B25 −4.05 (−6 → +0.25) · EVAA B26 −4.33 (−10 → +8.00).")
    for s_ in STOP_LINES:
        c = f"STOP{s_}"
        L.append(f"- **Stop −{s_}** (one stop ≈ {s_ * HT_ACCT_PER_PCT:.1f} % of the account at today's size): **{g[c][0]}** — {g[c][1]}")
    stp = F[F.live_stop.astype(bool)]
    off = stp[stp.NEW2_why.astype(str) != "STOP"] if "NEW2_why" in stp else stp.iloc[0:0]
    if len(off):
        L.append(f"⚠ {len(off)} live-stopped fill(s) the replica did NOT stop at −4 ({', '.join(off.pair)}) — their Δ is measured from the replica, read with care.")
    nop = pend[pend.status.astype(str).isin(["no path", "no entry"])]
    if len(nop):
        L.append(f"⚠ {len(nop)} forward fill(s) could not be priced ({', '.join(nop.pair.astype(str))}) — outside the stop cohort.")
    if len(stp):
        L += ["", "| Opened (UTC) | Pair | live −4 (booked) | −6 | −8 | −10 |", "|---|---|---|---|---|---|"]
        for r in stp.itertuples():
            L.append(f"| {str(r.opened_at)[5:16].replace('T', ' ')} | {r.pair} | {r.booked:+.2f} | {r.STOP6:+.2f} ({r.STOP6_why}) | "
                     f"{r.STOP8:+.2f} ({r.STOP8_why}) | {r.STOP10:+.2f} ({r.STOP10_why}) |")
    if any(g[c][0] == "OPERATOR CALL" for c in g):
        L.append("👁 A wider-stop line passed → OPERATOR CALL: ship only at size × 4 / s (one stop stays ≈ 7.8 % of the account) with its own revert gate.")
    return L + [f"Gate (frozen 2026-11-14): ≥ {STOP_MIN_STOPPED} live-stopped HT fills on ≥ {STOP_MIN_DAYS} days, none pending → Σ Δ > 0 ∧ 95 % "
                "CI low > 0 ∧ risk-equal Σ Δ > 0 → operator call, else no change (decided once)."]


def price_blocked(B, now_ms, fetch_ok=True):
    """🛑🧱 (548) blocked HT signals (scout_blocked_filters.load_blocked rows) priced with the blocked-trade pricer at today's stop (−4) and at
    each STOP_LINES stop → DataFrame: key, pair, opened_at, day, status (PRICED / PENDING / NOT_EVALUABLE / SHADOWED), NEW2 (today's −4
    price), STOP6 / 8 / 10, live_stop (= the −4 walk STOPPED: no live fill exists, the replica decides). Shadowing (one shadow per pair at a
    time) is read on the −4 walk and applied to every stop (one cohort). A row is PRICED only when every stop priced."""
    import scout_blocked_filters as SB
    import scout_frenzy_v2b_shadow as VB
    cols = ["key", "pair", "opened_at", "day", "status", "NEW2", "NEW2_why", "live_stop"] + [f"STOP{s_}" for s_ in STOP_LINES] \
        + [f"STOP{s_}_why" for s_ in STOP_LINES]
    price_blocked.last_budget = ""
    if not len(B):
        return pd.DataFrame(columns=cols)
    bud = VB._Bud(bool(fetch_ok))
    rec = B.to_dict("records")
    stops = (STOP_TODAY,) + tuple(float(x) for x in STOP_LINES)
    ths = {}
    for s_ in stops:
        ths[s_] = SB.th()
        ths[s_]["frenzy_willy_stop_pct"] = s_
    res = {s_: [] for s_ in stops}
    pc, tc = {}, {}                                            # review 548: the path and the target do not depend on the stop — load once
    for r in rec:                                              # review 548: rows outer, stops inner — a budget stop drops whole rows,
        for s_ in stops:                                       # never only their wider walks
            try:
                res[s_].append(SB.price_frenzy(r, ths[s_], now_ms, pc, tc, bud))
            except Exception as e:                             # review 548: one bad row never kills the section (price_all's pattern)
                end = int(r["te"]) + (SB.exit_of(r["sleeve"], ths[s_])["cap"] + 1) * MIN
                res[s_].append((None, f"ERR:{type(e).__name__}", SB._late_over(now_ms, end), None))
    price_blocked.last_budget = bud.stop or ""
    r4 = res[STOP_TODAY]
    C = B.assign(pct=[x[0] for x in r4], why=[x[1] for x in r4], status=[x[2] for x in r4], exit_ms=[x[3] for x in r4])
    t4 = SB.th()
    t4["frenzy_willy_stop_pct"] = STOP_TODAY
    C = SB.shadow_episodes(C, t4)
    order = {k: i for i, k in enumerate(B.key)}                # res lists follow B's row order; shadow_episodes re-sorts C by te
    out = []
    for r in C.itertuples():
        i = order[r.key]
        st = r.status
        if st == "PRICED" and any(res[float(s_)][i][2] != "PRICED" for s_ in STOP_LINES):
            st = next(res[float(s_)][i][2] for s_ in STOP_LINES if res[float(s_)][i][2] != "PRICED")   # a wider walk still filling in
        row = dict(key=r.key, pair=r.pair, opened_at=pd.Timestamp(int(r.te), unit="ms").strftime("%Y-%m-%dT%H:%M:%S"), day=r.day, status=st,
                   NEW2=r.pct, NEW2_why=r.why, live_stop=(str(r.why) == "STOP"))
        for s_ in STOP_LINES:
            p, w = res[float(s_)][i][0], res[float(s_)][i][1]
            row[f"STOP{s_}"], row[f"STOP{s_}_why"] = p, w
        out.append(row)
    return pd.DataFrame(out, columns=cols).sort_values(["opened_at", "key"], kind="stable").reset_index(drop=True)


def _blocked_stop_section(st, state_path, now_ms, fetch_ok, dec_paths=None):
    """🛑🧱 (548) the wider-stop lines on BLOCKED HT signals — a separate population (failed spikes the filter refuses), its own frozen gate
    (same bar as the kept-fill lines), never pooled with them. → report lines."""
    import scout_blocked_filters as SB
    L = ["", "### 🛑🧱 HT wider-stop lines on BLOCKED signals (DECISION_LOG 548, observe only — a separate line, never pooled with the kept fills)"]
    fl = _GK.dl_deploy_ms(DL_BLK, label="WILLY_HT_BLOCKED_STOP_LINES")
    if fl is None:
        return L + ["Not evaluable yet: the DECISION_LOG 548 commit is not in git history (counted from its commit + 10 min). "
                    "In-sample at registration (never counted): 17 HT −4 stops incl. blocked, B25 + B26 — Σ Δ vs −4: −6 +4.9 · −8 +7.3 · −10 +32.4 pts."]
    B = SB.load_blocked(paths=dec_paths, since_ms=fl)
    B = B[B.sleeve.astype(str) == "FRENZY_WILLY_HT"] if len(B) else B
    R = price_blocked(B, now_ms, fetch_ok)
    F = R[R.status.eq("PRICED")].copy()
    pend = R[R.status.eq("PENDING")]
    days = list(F.day)
    g = gate_stop(F, days, pend=list(pend.opened_at))
    changed = False
    for c, (v, det, _, _) in list(g.items()):
        k = BLK_PREFIX + c
        if k in st:
            g[c] = (st[k]["verdict"], st[k]["detail"] + f" (frozen {st[k]['at']})", None, None)
        elif v != "OBSERVING":
            st[k] = dict(verdict=v, detail=det, at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")); changed = True
    if changed and state_path:
        try:
            _save_state(st, state_path)
        except Exception:
            L.append("⚠ blocked stop-line state not saved this run.")
    n_sh = int(R.status.eq("SHADOWED").sum()); n_ne = int(R.status.isin(["NOT_EVALUABLE"]).sum())
    n_np = int((R.status.isin(["PENDING", "NOT_EVALUABLE"]) & R.NEW2_why.astype(str).eq("NO_PATH")).sum())
    L.append(f"Blocked HT signals refused since {pd.Timestamp(int(fl), unit='ms'):%Y-%m-%d %H:%M} UTC: {len(R)} · priced {len(F)} "
             f"({int(F.live_stop.astype(bool).sum())} stopped by the −4 walk) · pending {len(pend)} · shadowed {n_sh} · not evaluable {n_ne}. "
             "Ruler = the blocked-trade pricer (refusal + 8 s, 0.10 % slip, taker, today's HT exit); 'stopped' is the replica's −4 stop — no live "
             "fill exists, so a wick live polling might ride through counts here (tilts toward a wider stop); shadows (one per pair at a time) are "
             "read on the −4 walk, so a refusal a wider-stop shadow would still cover counts as fresh (slight tilt wider). Paths come from the "
             f"blocked-trade pricer's hourly fetch — without a path: {n_np}"
             + (f" · ⚠ data budget stopped: {getattr(price_blocked, 'last_budget', '')}" if getattr(price_blocked, "last_budget", "") else "") + ".")
    for s_ in STOP_LINES:
        c = f"STOP{s_}"
        L.append(f"- **Blocked · stop −{s_}**: **{g[c][0]}** — {g[c][1]}")
    stp = F[F.live_stop.astype(bool)]
    if len(stp):
        L += ["", "| Refused + 8 s (UTC) | Pair | −4 | −6 | −8 | −10 |", "|---|---|---|---|---|---|"]
        for r in stp.itertuples():
            L.append(f"| {str(r.opened_at)[5:16].replace('T', ' ')} | {r.pair} | {r.NEW2:+.2f} | {r.STOP6:+.2f} ({r.STOP6_why}) | "
                     f"{r.STOP8:+.2f} ({r.STOP8_why}) | {r.STOP10:+.2f} ({r.STOP10_why}) |")
    if any(g[c][0] == "OPERATOR CALL" for c in g):
        L.append("👁 A blocked-signal stop line passed — SUPPORTING evidence only (a different population): read it beside the kept-fill line above; "
                 "a ship still needs size × 4 / s and its own revert gate.")
    return L + [f"Gate (frozen 2026-11-14, DL 548): the same bar as the kept-fill lines — ≥ {STOP_MIN_STOPPED} −4-stopped blocked signals on ≥ "
                f"{STOP_MIN_DAYS} days, none pending → Σ Δ > 0 ∧ 95 % CI low > 0 ∧ risk-equal Σ Δ > 0 → operator call (supporting), else no change (decided once)."]


def _blocked_safe(st, state_path, now_ms, fetch_ok):
    """🛑🧱 (548) the blocked-signal section in its own try (review 548: shown even when no kept HT fill is exported)."""
    try:
        return _blocked_stop_section(st, state_path, now_ms, fetch_ok)
    except Exception as e:
        return ["", f"### 🛑🧱 HT wider-stop lines on blocked signals — unavailable this run ({type(e).__name__}: {str(e)[:100]})."]


def _state(path):
    try:
        st = json.load(open(path)) if (path and os.path.exists(path)) else {}
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def _save_state(st, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, path)                                      # review 537: atomic — a crash never leaves a corrupt frozen verdict


def run(now_ms=None, paths=None, floor_ms=None, fetch_ok=True, state_path=STATE):
    """→ report lines. The scout's 1m loader is installed only for this call and the shared loader / its cache restored after."""
    k1_snap = dict(S.FX.WT._K1)
    S.FX.WT.load_1m = _load_1m
    try:
        return _run(now_ms, paths, floor_ms, fetch_ok, state_path)
    finally:
        S.FX.WT.load_1m = _TRUE_LOAD_1M
        S.FX.WT._K1.clear(); S.FX.WT._K1.update(k1_snap)


def _run(now_ms, paths, floor_ms, fetch_ok, state_path):
    now_ms = int(now_ms or time.time() * 1000)
    L = ["## 🎯👁 WILLY HT EXIT — TP +2 % revert gate (shipped in paper) + RUN15 watch line (DECISION_LOG 537)", ""]
    try:
        S.TH = S.CF.load_th()
        floor = floor_ms if floor_ms is not None else _GK.dl_deploy_ms(DL, label="WILLY_HT_EXIT_LINES")
        if floor is None:
            return L + ["Not evaluable yet: the DECISION_LOG 537 commit is not in git history (counted from its commit + 10 min).", ""] \
                + _blocked_safe(_state(state_path), state_path, now_ms, fetch_ok)
        d = load(paths)
        if not len(d):
            return L + ["No closed WILLY HT fill in the exports yet.", ""] + _blocked_safe(_state(state_path), state_path, now_ms, fetch_ok)
        d["k"] = [S._key(o, p) for o, p in zip(d.opened_at, d.pair)]
        o_ms = [S.CF._ms(x) for x in d.opened_at]
        d["fwd"] = [(t is not None and t >= floor) for t in o_ms]
        use = d[d.fwd | d.k.isin(CTX_KEYS)]
        note = fetch(use.to_dict("records"), now_ms) if fetch_ok else "no fetch"
        R = price(use)
        R["fwd"] = R.key.map(dict(zip(d.k, d.fwd))).fillna(False)
        for c in ("OLD125", "NEW2", "RUN15", "status"):
            if c not in R:
                R[c] = np.nan
        priced = R[R.status.eq("priced")]
        pend = R[~R.status.eq("priced")]
        F = priced[priced.fwd].sort_values("opened_at")
        D = pd.DataFrame({"TP2": (F.NEW2 - F.OLD125).to_numpy(), "RUN15": (F.RUN15 - F.NEW2).to_numpy()})
        days = list(pd.to_datetime(F.opened_at).dt.strftime("%Y-%m-%d"))
        fp = pend[pend.fwd]
        fu = fp[fp.status.eq("UNRES")]                         # only a path still filling in holds a decision (no path / T unknown never price)
        held = bool(len(fu) and len(F) and fu.opened_at.min() < F.opened_at.iloc[min(len(F), N_GATE) - 1])
        st = _state(state_path)
        g = gate(D, days, held)
        changed = False
        for c in LINES:
            if c in st:
                g[c] = (st[c]["verdict"], st[c]["detail"] + f" (frozen {st[c]['at']})")
            elif g[c][0] != "OBSERVING":
                st[c] = dict(verdict=g[c][0], detail=g[c][1], at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")); changed = True
        if changed and state_path:
            try:
                _save_state(st, state_path)
            except Exception:
                L.append("⚠ gate state not saved this run.")
        L.append(f"Forward since {pd.Timestamp(int(floor), unit='ms'):%Y-%m-%d %H:%M} UTC: {len(F)} priced HT fills · pending {len(fp)} "
                 f"({', '.join(f'{k} {v}' for k, v in fp.status.value_counts().items()) or '—'}) · RUN15 without an ATR stamp {int(F.RUN15.isna().sum())} ({note}).")
        if len(F):
            L.append(f"Live +2 % vs the +1.25 exit: Σ {F.NEW2.sum():+.2f} vs {F.OLD125.sum():+.2f} pts (Δ {(F.NEW2 - F.OLD125).sum():+.2f}; "
                     f"{int(((F.NEW2 - F.OLD125) > 1e-9).sum())} better / {int(((F.NEW2 - F.OLD125) < -1e-9).sum())} worse) · win rate "
                     f"{(F.OLD125 > 0).mean():.0%} → {(F.NEW2 > 0).mean():.0%}.")
        L.append(f"- **TP +2 % (shipped)**: **{g['TP2'][0]}** — {g['TP2'][1]}")
        L.append(f"- **RUN15 (watch line)**: **{g['RUN15'][0]}** — {g['RUN15'][1]}")
        show = priced.sort_values("opened_at")
        if len(show):
            f = lambda v, w: "–" if v != v else f"{v:+.2f} ({w})"   # noqa: E731
            L += ["", "| Opened (UTC) | Pair | counted | T old → new | +1.25 exit | live +2 % | RUN15 | booked |", "|---|---|---|---|---|---|---|---|"]
            for r in show.itertuples():
                L.append(f"| {str(r.opened_at)[5:16].replace('T', ' ')} | {r.pair} | {'yes' if r.fwd else 'context'} | {r.T_old:+.2f} → {r.T_new:+.2f} | "
                         f"{f(r.OLD125, r.OLD125_why)} | {f(r.NEW2, r.NEW2_why)} | {f(r.RUN15, r.RUN15_why)} | {r.booked:+.2f} |")
        live = priced[priced.fwd]
        bad = live[(live.NEW2 - live.booked).abs() > 0.15]
        if len(bad):
            L.append(f"⚠ replica ≠ booked on {len(bad)} forward fill(s) by > 0.15 pt ({', '.join(bad.pair)}) — read those with care.")
        try:                                                   # review 546: the stop lines can never hide the TP2 / RUN15 banners
            L += _stop_section(R, d, st, state_path)
        except Exception as _se:
            L += ["", f"### 🛑 HT wider-stop watch lines — unavailable this run ({type(_se).__name__}: {str(_se)[:100]})."]
        L += _blocked_safe(st, state_path, now_ms, fetch_ok)    # 🛑🧱 (548) its own try — never hides the lines above
        if g["TP2"][0] == "REVERT":
            L.append("⛔ TP2 GATE FIRED → REVERT: set frenzy_willy_ht_tp_pct 0 (dashboard WILLY_HT take profit) and log it in DECISION_LOG — operator applies.")
        if g["RUN15"][0] == "OPERATOR CALL":
            L.append("👁 RUN15 passed its gate → OPERATOR CALL (a live exit change needs the haircut + its own revert gate).")
    except Exception as e:
        L.append(f"Unavailable this run ({type(e).__name__}: {str(e)[:120]}).")
    return L + ["", f"Gates (frozen 2026-11-12): TP +2 % — first {EARLY_N} HT fills Σ (live − the +1.25 exit) < 0 → revert; at {N_GATE} Σ ≤ 0 → revert. "
                    f"RUN15 — at {N_GATE} fills (≥ 2 days, none pending): Σ Δ vs the live exit > 0 ∧ 95 % CI low > 0 ∧ p ≤ 0.10 → operator call. "
                    "Study: reports/WILLY_HT_EXIT_STUDY_2026-11-12.md.", ""]


if __name__ == "__main__":
    print("\n".join(run(fetch_ok="--no-fetch" not in sys.argv)))
