#!/usr/bin/env python3
"""🎯👁 Scout — WILLY HT take profit +2 % (SHIPPED in paper, DECISION_LOG 537 declared override) revert gate + the RUN15 watch line (observe
only — never changes config, no bot API; public 1m klines only, budgeted).

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
        rows.append(dict(base, status=st, T_old=t_old, T_new=t_new, T_src=t_src, OLD125=old[0], OLD125_why=old[1], NEW2=new[0], NEW2_why=new[1],
                         RUN15=(run[0] if run[1] != "UNRES" else float("nan")), RUN15_why=run[1]))
    return pd.DataFrame(rows)


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
            return L + ["Not evaluable yet: the DECISION_LOG 537 commit is not in git history (counted from its commit + 10 min).", ""]
        d = load(paths)
        if not len(d):
            return L + ["No closed WILLY HT fill in the exports yet.", ""]
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
