#!/usr/bin/env python3
"""🧯 Scout — EXH_SSC_OIVOL shadow revert gate (DECISION_LOG 582, 2026-11-18, operator "yes build it" — DECLARED OVERRIDE, ALERT-only,
never changes config).

RULE (engine)   services/exhaustion.ssc_oivol_block: at an EXHAUSTION_SHORT signal the rule already passed, BLOCK when spot_share_chg >
                −0.0366 (with the rule's < −0.0066: the spot share falls only a little) AND oi_vol = OI_usd ÷ the 24 h quote volume > 0.128.
                OI_usd = openInterestHist sumOpenInterestValue of the latest row with T − 30 min ≤ timestamp ≤ T − 5 min (the API timestamp
                = the metrics create_time, checked live — the study's exact window); unreadable → FAIL-OPEN (trade, EXH_SSC_OIVOL_UNREAD).
EVIDENCE        reports/EXH_BADDAY_2026-11-18.md (2023-07 → 2026-11, the live exit): blocked 630 of 3,998 (16 %), 31 % WR, −0.43 %/trade;
                daily compound at 0.183 × equity +0.356 → +0.399 %/day; design 511 · −0.49, unseen 2026-H2 119 · −0.16 (P(avg<0) 0.63);
                blocked negative in 5 of 7 half-years (2023-H2 +0.66, 2024-H1 +0.26); family-wise luck p 0.45 → DECLARED OVERRIDE.
COHORT          BLOCKED EXH_SSC_OIVOL signals (exh_block = EXH_SSC_OIVOL) from the decisions exports' e=EXHAUSTION journal rows
                (scout_exhaustion_short.load_journal, dedup (pair, signal time)), signal time ≥ the DL 582 commit + 10 min
                (scout_gatekit.dl_deploy_ms — before the commit is found: NOT EVALUABLE, nothing counted). Ordered by signal time.
SHADOW PRICE    each blocked signal priced as if traded with the LIVE exit (scout_exh_exit_override's machinery): entry = the journaled
                decision price (the signal 15m close ≈ the next 5m open the study filled at) · the path = the 1,440 1m bars from the decision
                time T (the entry bar) · TP / SL = the journaled distances (services.exhaustion.exit_levels with the live config at the
                decision; missing → recomputed from the journaled ATR at the DL 579 exit) · TP = a resting maker limit filled only when a 1m
                low trades THROUGH it · SL taker, gap-aware (the worse of the stop and the bar open) · both in one bar → the stop · time exit
                at the close of minute 1,440 · taker 0.05 % on the entry and on a taker exit, maker 0.02 % on the TP · the modelled slip
                (exh_model_slip_pct, at the notional the sleeve's sizing gave at the decision) on the ENTRY and on a taker EXIT (never on
                the maker TP). A row without a modelled slip is priced at 0 slip and flagged. Funding is NOT priced.
GATE (frozen)   read ONCE: at the first prefix of ≥ 30 priced blocked signals spanning ≥ 10 UTC days (by signal time) → the one-sided
                80 % LOWER bound of the blocked mean (day-block bootstrap, seed 582) > 0 → REVERT (false REVERT ≈ 10–16 %, power ≈ 40 %) →
                REVERT (the filter blocked winners; switch exhaustion_ssc_oivol_block_enabled off — operator decides) · else KEEP. The verdict
                is frozen in reports/SCOUT_EXH_SSC_OIVOL.json the first time it is reached (never recomputed). Counted signals are always a
                signal-time PREFIX: pricing halts at the first signal it cannot price this run (path not over / network); a path that is
                NO_DATA for good (gapped / delisted) is listed and left out. Also printed: running N · WR · mean of the blocked cohort and
                (information) the kept EXHAUSTION fills' booked mean over the same window.
NETWORK         fapi /fapi/v1/klines (1m, 1,440 bars, weight 10 per signal), ≥ 0.6 s apart, stops above 1,000 used weight / on 418 / 429;
                priced results cached in reports/.exh_ssc_oivol_cache.json (gitignored; a path still short 2 h after it should be complete
                is cached NO_DATA).
Usage: venv/bin/python scripts/scout_exh_ssc_oivol.py"""
import json
import math
import os
import sys
import time
import urllib.error

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)

import scout_exh_exit_override as XO          # noqa: E402  price_short / path_bars / fetch_1m / NetStop / the live exit (OVERRIDE)

DL = 582
BLOCK = "EXH_SSC_OIVOL"
GATE_N, GATE_DAYS = 30, 10
HOLD_MIN = XO.HOLD_MIN
LIVE_EXIT = XO.OVERRIDE                       # the DL 579 exit the sleeve runs (pinned in observe_registry → ⚠ STALE on a change)
CACHE = os.path.join(_R, "reports", ".exh_ssc_oivol_cache.json")
STATE = os.path.join(_R, "reports", "SCOUT_EXH_SSC_OIVOL.json")


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ms(ts):
    import pandas as pd
    t = pd.to_datetime(str(ts)[:19], errors="coerce")
    return None if pd.isna(t) else int(t.value // 1_000_000)


def _load(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return None                                               # unreadable → never freeze over it


def _save(d, path):
    try:
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
    except Exception:
        pass


def row_levels(tp, sl, atr):
    """(TP %, SL %) — the journaled distances; either missing → both from the journaled ATR at the live exit. None when neither. Pure."""
    t, s = _f(tp), _f(sl)
    if t is not None and t > 0 and s is not None and s > 0:
        return t, s
    return XO.levels(atr, LIVE_EXIT)


def price_signal(pair, t_ms, entry, tp, sl, slip, cache, now_ms=None):
    """→ (net %, why) · 'NO_DATA' (unpriceable for good) · None (the 24 h path is not over / complete yet). The path starts at the decision
    time t_ms (the entry 5m bar's open = the 1m bar opening at t_ms). Net = XO.price_short (taker entry fee + exit fee + slip on a taker
    exit) − slip once more for the ENTRY (the decision price is not a real fill). The cache keeps the priced result (never the bars)."""
    t1 = int(t_ms) // 60_000 * 60_000
    s = _f(slip) or 0.0
    key = f"{pair}|{t1}|{round(float(tp), 6)}|{round(float(sl), 6)}|{round(s, 6)}"
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    hit = cache.get(key)
    if hit == "NO_DATA":
        return "NO_DATA"
    if isinstance(hit, list) and len(hit) == 2:
        return float(hit[0]), str(hit[1])
    if _f(entry) is None or _f(entry) <= 0:
        cache[key] = "NO_DATA"
        return "NO_DATA"
    if now_ms < t1 + HOLD_MIN * 60_000 + 60_000:
        return None
    bars = XO.path_bars(XO.fetch_1m(pair, t1, HOLD_MIN), t1)
    if bars is None:
        if now_ms > t1 + HOLD_MIN * 60_000 + 2 * 3_600_000:
            cache[key] = "NO_DATA"
            return "NO_DATA"
        return None
    net, why = XO.price_short(entry, bars, float(tp), float(sl), HOLD_MIN, s)
    if net is None:
        if why == "NO_DATA":
            cache[key] = "NO_DATA"
            return "NO_DATA"
        return None
    net -= max(0.0, s)
    cache[key] = [net, why]
    return net, why


LB_Q, BOOT_N, BOOT_SEED = 0.80, 2000, 582


def lower_bound(rows, q=LB_Q, n_boot=BOOT_N, seed=BOOT_SEED):
    """one-sided q LOWER bound of the mean by a day-block bootstrap (resample whole days; deterministic seed). Pure."""
    import random
    by = {}
    for d, x in rows:
        by.setdefault(d, []).append(x)
    days = list(by.values())
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [days[rng.randrange(len(days))] for _ in range(len(days))]
        means.append(sum(sum(v) for v in pick) / sum(len(v) for v in pick))
    means.sort()
    return means[max(0, int(math.floor((1 - q) * len(means))) - 1)]


def gate(rows):
    """rows = [(utc day, net)] in signal-time order → (verdict, text, n used or None). The first prefix with ≥ GATE_N rows spanning ≥ GATE_DAYS
    days decides: the one-sided 80 % LOWER bound of the blocked mean (day-block bootstrap) > 0 → REVERT (the blocked signals are shown to be
    winners) · else KEEP. A raw "mean > 0" would flag a false REVERT ≈ 1 time in 3 at the study's own effect (SD ≈ 5.6 % per signal, deep
    review) — this rule ≈ 10 % at the study's −0.43, ≈ 16 % at the unseen −0.16; power ≈ 40 % if the blocked signals really earn the sleeve's
    +0.65 (low power, stated). Before that: COLLECTING. Pure."""
    rows = [(d, x) for d, x in rows if x is not None and math.isfinite(x)]
    rule = (f"gate: the first {GATE_N}+ priced blocked signals spanning ≥ {GATE_DAYS} UTC days → the one-sided {LB_Q:.0%} lower bound of the "
            f"blocked mean (day-block bootstrap) > 0 → REVERT (the filter blocked winners) · else KEEP — read once, frozen; false REVERT ≈ 10–16 %")
    for n in range(GATE_N, len(rows) + 1):
        if len({d for d, _ in rows[:n]}) >= GATE_DAYS:
            cut = [x for _, x in rows[:n]]
            m = sum(cut) / n
            wr = 100.0 * sum(1 for x in cut if x > 0) / n
            lb = lower_bound(rows[:n])
            info = (f"first {n} blocked signals ({len({d for d, _ in rows[:n]})} days): mean {m:+.3f} % · {LB_Q:.0%} lower bound {lb:+.3f} % · "
                    f"WR {wr:.0f} % (slip = an upper bound: the shadow notional skips the depth / liquidity caps)")
            if lb > 0:
                return "REVERT", f"🔴 {info} → the filter blocked winners: flag switch EXH_SSC_OIVOL off (operator decides)", n
            return "KEEP", f"🟢 {info} → not shown to block winners: keep the filter", n
    days = len({d for d, _ in rows})
    return "COLLECTING", f"⏳ {len(rows)}/{GATE_N} priced on {days}/{GATE_DAYS} days — {rule}", None


def cohort(journal, floor_ms):
    """the BLOCKED EXH_SSC_OIVOL journal rows at / after floor_ms, signal-time order (list of dicts). floor_ms None → [] (NOT EVALUABLE)."""
    if floor_ms is None or journal is None or not len(journal):
        return []
    out = []
    for r in journal.to_dict("records"):
        if str(r.get("exh_block") or "") != BLOCK:
            continue
        t = _ms(r.get("exh_signal_at"))
        if t is None or t < floor_ms:
            continue
        r["_t"] = t
        out.append(r)
    return sorted(out, key=lambda r: (r["_t"], str(r.get("pair"))))


def report(journal, fills=None, now_ms=None, state_path=STATE, floor_ms="auto"):
    import pandas as pd
    out = [f"🧯 EXH_SSC_OIVOL shadow revert gate (DECISION_LOG {DL}, declared override) — blocked = spot-share chg > −0.0366 ∧ OI ÷ 24 h "
           "vol > 0.128 · evidence: blocked 630 / 3,998 · 31 % WR · −0.43 %/trade (2023-07 → 2026-11, live exit), DCR +0.356 → +0.399 %/day, "
           "unseen 2026-H2 119 · −0.16, 5 of 7 half-years negative, luck p 0.45"]
    if floor_ms == "auto":
        import scout_gatekit as GK
        floor_ms = GK.dl_deploy_ms(DL, label=f"EXH_SSC_OIVOL (DECISION_LOG {DL})")
    if floor_ms is None:
        out.append(f"  ⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet: nothing counted")
        return "\n".join(out)
    XO._weight_hit[0] = False
    cache = _load(CACHE) or {}
    rows, lines, nodata, noslip = [], [], [], []
    halted = None
    coh = cohort(journal, floor_ms)
    for r in coh:
        lv = row_levels(r.get("exh_tp_pct"), r.get("exh_sl_pct"), r.get("exh_atr_pct"))
        tag = f"{r.get('pair')} {str(r.get('exh_signal_at'))[:16]}"
        if lv is None:
            nodata.append(tag + " (no TP / SL / ATR)")
            continue
        slip = _f(r.get("exh_model_slip_pct"))
        if slip is None:
            noslip.append(tag)
        try:
            p = price_signal(str(r.get("pair")), r["_t"], _f(r.get("exh_decision_price")), lv[0], lv[1], slip, cache, now_ms)
        except XO.NetStop as e:
            halted = f"{tag}: {e}"
            break
        except (urllib.error.URLError, OSError, ValueError, TypeError) as e:
            halted = f"{tag}: {type(e).__name__}"
            break
        if p is None:
            halted = f"{tag}: path not complete yet"
            break
        if p == "NO_DATA":
            nodata.append(tag)
            continue
        day = time.strftime("%Y-%m-%d", time.gmtime(r["_t"] / 1000))
        rows.append((day, p[0]))
        lines.append(f"    {tag} @ {_f(r.get('exh_decision_price'))} · TP {lv[0]:.2f} / SL {lv[1]:.2f} · slip {slip if slip is not None else 0:.3f} → "
                     f"{p[0]:+6.2f} % ({p[1]}) · ssc {r.get('exh_spot_share_chg')} · OI/vol {r.get('exh_oi_vol')}")
    _save(cache, CACHE)
    st = _load(state_path) if state_path else {}
    if st is not None and "verdict" not in st:
        v, txt, n = gate(rows)
        if n is not None:
            st = dict(verdict=v, detail=txt, n=n, run_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime((now_ms or time.time() * 1000) / 1000)))
            if state_path:
                _save(st, state_path)
    if st and st.get("verdict"):
        out.append(f"  FROZEN ({st.get('run_at')}): {st.get('verdict')} — {st.get('detail')} (alert only · operator switches)")
    else:
        v, txt, _ = gate(rows)
        out.append(f"  verdict: {v} — {txt}" + (" (⚠ state file unreadable — nothing frozen this run)" if st is None else ""))
    if rows:
        xs = [x for _, x in rows]
        out.append(f"  running: N {len(xs)} · WR {100.0 * sum(1 for x in xs if x > 0) / len(xs):.0f} % · mean {sum(xs) / len(xs):+.3f} % "
                   f"(shadow-priced, live exit, modelled slip) · {len({d for d, _ in rows})} days")
        if fills is not None and len(fills):
            a, b = min(r["_t"] for r in coh), max(r["_t"] for r in coh)
            o = pd.to_datetime(fills.opened_at.astype(str).str[:19], errors="coerce")
            k = fills[(o >= pd.Timestamp(a, unit="ms")) & (o <= pd.Timestamp(b, unit="ms") + pd.Timedelta(hours=24))]
            if len(k):
                kp = pd.to_numeric(k.pnl_percentage, errors="coerce").dropna()
                out.append(f"  (information) kept EXHAUSTION fills over the same window: N {len(kp)} · mean {kp.mean():+.3f} % booked")
    else:
        out.append(f"  running: no priced blocked signal yet ({len(coh)} in the cohort)")
    if halted:
        out.append(f"  ⏸ pricing halted at {halted} — later signals wait for the next run (the counted signals stay a prefix)")
    if nodata:
        out.append(f"  NO_DATA for good (left out): {', '.join(nodata)}")
    if noslip:
        out.append(f"  ⚠ no modelled slip journaled (priced at 0 slip): {', '.join(noslip)}")
    out.append("  ⚠ funding not priced")
    out.extend(lines)
    return "\n".join(out)


def load():
    import scout_exhaustion_short as SC
    return SC.load_journal(), SC.load_fills()


def run(now_ms=None):
    j, f = load()
    return ["## 🧯 EXH_SSC_OIVOL (DL 582)", "", "```", report(j, f, now_ms=now_ms), "```", ""]


def main():
    j, f = load()
    print(report(j, f))


if __name__ == "__main__":
    main()
