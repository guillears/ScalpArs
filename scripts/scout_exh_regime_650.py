#!/usr/bin/env python3
"""🌡⏱ Scout — EXH_BEAR80 + EXH_EARLY_SPIKE shadow revert gates (DECISION_LOG 650, 2026-11-27, operator "yes build it" — DECLARED OVERRIDE,
ALERT-only, never changes config).

RULE (engine)   services/exhaustion.regime_block: a FRENZY-universe EXHAUSTION_SHORT signal that passed every other RULE gate (execution
                refusals — book full / late / depth — are not modelled: a few priced rows may be unreachable) is BLOCKED when the
                market is > 80 % bearish (EXH_BEAR80, the engine's breadth = the entry_bear_pct stamp) or comes < 0.5 h after the spike
                (EXH_EARLY_SPIKE, = entry_frenzy_hours). BEAR80 is judged first.
EVIDENCE        2-year backtest at TP 7 (reports/BATCH28_REVIEW_2026-11-27_data/exh_lines_backtest.csv): zones 482 · 42 % · −0.15 %/fill;
                blocking both DCR 1.03 → 1.12 %/day, max DD −62.7 → −55.3 %; the zones are ≈ 0 since 2025-H2 and the latest half-year
                (2026-H2) loses DCR (2.44 → 2.09). Below the locked gates (observe lines 2 / 15 and 1 / 15) → DECLARED OVERRIDE, tight gate.
COHORT          per line, the BLOCKED journal rows (exh_block = EXH_BEAR80 / EXH_EARLY_SPIKE, FRENZY universe) from the decisions exports,
                signal time ≥ the DL 650 commit + 10 min (scout_gatekit.dl_deploy_ms — before it: NOT EVALUABLE). Signal-time order.
SHADOW PRICE    scout_exh_ssc_oivol.price_signal (the SAME machinery as the DL 582 gate): entry = the journaled decision price, 1m path from
                the decision time, journaled TP / SL (the live exit), maker TP through the level, gap-aware taker stop, 24 h, taker / maker
                fees, the modelled slip on the entry and a taker exit. Funding NOT priced.
GATE (frozen)   per line, read ONCE at the first prefix of ≥ 10 priced blocked signals spanning ≥ 5 UTC days: blocked mean ≥ REF → REVERT
                (flag switch the line off — operator decides) · else KEEP. REF = the re-priced (modelled slip) mean of the kept FRENZY-universe EXHAUSTION fills
                opened from the first to the last counted signal + 24 h when ≥ 5 such fills, else +0.66 (the 2-year sleeve mean at TP 7).
                Frozen in reports/SCOUT_EXH_REGIME_650.json (never recomputed) once the window has settled (last counted signal + 48 h);
                a failed state write is reported, never treated as frozen. Pricing halts at the first signal it cannot price (prefix); a
                signal that can NEVER be priced (NO_DATA) is listed and left out, as in the DL 582 gate. Bias: blocked = ex-funding vs
                REF = booked incl. funding → leans toward REVERT (the safe side for a declared override).
NETWORK         via scout_exh_exit_override (fapi 1m klines, weight-capped, stops on 418 / 429); cache shared with the DL 582 gate.
Usage: venv/bin/python scripts/scout_exh_regime_650.py"""
import math
import os
import sys
import time
import urllib.error

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)

import scout_exh_ssc_oivol as SO               # noqa: E402  price_signal / row_levels / _load / _save / _ms / _f / cache
XO = SO.XO

DL = 650
LINES = ("EXH_BEAR80", "EXH_EARLY_SPIKE")
GATE_N, GATE_DAYS, REF_MIN_FILLS, REF_FALLBACK = 10, 5, 5, 0.66
SETTLE_MS = 48 * 3_600_000    # (review) freeze only once the REF window (… + 24 h) has closed AND its last kept fill could have exited (24 h hold)


def _save(d, path):
    """→ True when written (review: a failed write must not pretend the verdict is frozen)."""
    try:
        import json
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
        return True
    except Exception:
        return False
STATE = os.path.join(_R, "reports", "SCOUT_EXH_REGIME_650.json")


def cohort(journal, floor_ms, line):
    """the BLOCKED rows of one line at / after floor_ms (FRENZY universe), signal-time order. floor_ms None → []."""
    if floor_ms is None or journal is None or not len(journal):
        return []
    out = []
    for r in journal.to_dict("records"):
        if str(r.get("exh_block") or "") != line or str(r.get("exh_universe") or "FRENZY").upper() == "WIDE":
            continue
        t = SO._ms(r.get("exh_signal_at"))
        if t is None or t < floor_ms:
            continue
        r["_t"] = t
        out.append(r)
    return sorted(out, key=lambda r: (r["_t"], str(r.get("pair"))))


def kept_ref(fills, a_ms, b_ms):
    """(REF mean, n kept, source) — mean of the kept FRENZY-universe EXHAUSTION fills opened in [a, b + 24 h], each re-priced with the
    modelled slip (scout_exh_wide_585.net_of — the same slip model the blocked side pays); < 5 → the fallback."""
    import pandas as pd
    if fills is not None and len(fills):
        f = fills
        if "entry_exh_universe" in f:
            f = f[f.entry_exh_universe.fillna("FRENZY").astype(str).str.upper() != "WIDE"]
        o = pd.to_datetime(f.opened_at.astype(str).str[:19], errors="coerce")
        k = f[(o >= pd.Timestamp(a_ms, unit="ms")) & (o <= pd.Timestamp(b_ms, unit="ms") + pd.Timedelta(hours=24))]
        import scout_exh_wide_585 as WD     # (deep review M2) the kept fills on the SAME cost measure as the blocked shadow price
        kp = pd.Series([WD.net_of(r) for r in k.itertuples()], dtype="float64").dropna()
        if len(kp) >= REF_MIN_FILLS:
            return float(kp.mean()), int(len(kp)), "kept EXH fills, same window, re-priced with the modelled slip"
        return REF_FALLBACK, int(len(kp)), f"fallback +{REF_FALLBACK} (2-year sleeve mean; only {len(kp)} kept fills in the window)"
    return REF_FALLBACK, 0, f"fallback +{REF_FALLBACK} (no kept fills read)"


def gate(rows, ref_fn):
    """rows = [(utc day, net, t_ms)] in signal-time order; ref_fn(a_ms, b_ms) → (ref, n, src). → (verdict, text, n used or None). Pure
    apart from ref_fn."""
    rows = [r for r in rows if r[1] is not None and math.isfinite(r[1])]
    for n in range(GATE_N, len(rows) + 1):
        days = {d for d, _, _ in rows[:n]}
        if len(days) >= GATE_DAYS:
            xs = [x for _, x, _ in rows[:n]]
            m = sum(xs) / n
            wr = 100.0 * sum(1 for x in xs if x > 0) / n
            ref, nk, src = ref_fn(rows[0][2], rows[n - 1][2])
            info = (f"first {n} blocked signals ({len(days)} days): mean {m:+.3f} % · WR {wr:.0f} % vs REF {ref:+.3f} % ({src}; blocked = "
                    f"ex-funding, REF = booked funding incl. → leans toward REVERT)")
            if m >= ref:
                return "REVERT", f"🔴 {info} → the block removed trades at least as good as the ones kept: flag the switch off (operator decides)", n
            return "KEEP", f"🟢 {info} → the blocked signals did worse than the kept ones: keep the block", n
    return "COLLECTING", (f"⏳ {len(rows)}/{GATE_N} priced on {len({d for d, _, _ in rows})}/{GATE_DAYS} days — gate: first {GATE_N}+ priced "
                          f"blocked signals on ≥ {GATE_DAYS} UTC days, mean ≥ the kept EXH fills' mean (≥ {REF_MIN_FILLS} fills, else "
                          f"+{REF_FALLBACK}) → REVERT · else KEEP — read once, frozen"), None


def report(journal, fills=None, now_ms=None, state_path=STATE, floor_ms="auto"):
    out = [f"🌡⏱ EXH_BEAR80 + EXH_EARLY_SPIKE shadow revert gates (DECISION_LOG {DL}, declared override) — FRENZY universe · blocked = "
           "market > 80 % bearish · < 0.5 h after the spike · evidence: 2-year zones 482 · 42 % · −0.15 %/fill, DCR 1.03 → 1.12, max DD "
           "−62.7 → −55.3 %; latest half-year (2026-H2) DCR 2.44 → 2.09"]
    if floor_ms == "auto":
        import scout_gatekit as GK
        floor_ms = GK.dl_deploy_ms(DL, label=f"EXH_BEAR80 / EXH_EARLY_SPIKE (DECISION_LOG {DL})")
    if floor_ms is None:
        out.append(f"  ⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet: nothing counted")
        return "\n".join(out)
    cache = SO._load(SO.CACHE) or {}   # (deep review L4) the shared weight flag is NOT reset here — the DL 582 scout's reset covers the run
    st = SO._load(state_path) if state_path else {}
    for line in LINES:
        rows, nodata, halted = [], [], None
        coh = cohort(journal, floor_ms, line)
        for r in coh:
            lv = SO.row_levels(r.get("exh_tp_pct"), r.get("exh_sl_pct"), r.get("exh_atr_pct"))
            tag = f"{r.get('pair')} {str(r.get('exh_signal_at'))[:16]}"
            if lv is None:
                nodata.append(tag + " (no TP / SL / ATR)")
                continue
            try:
                p = SO.price_signal(str(r.get("pair")), r["_t"], SO._f(r.get("exh_decision_price")), lv[0], lv[1], SO._f(r.get("exh_model_slip_pct")),
                                    cache, now_ms)
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
            rows.append((time.strftime("%Y-%m-%d", time.gmtime(r["_t"] / 1000)), p[0], r["_t"]))
        ref_fn = lambda a, b: kept_ref(fills, a, b)   # noqa: E731
        rec = (st or {}).get(line) if st is not None else None
        unsaved = settling = False
        if st is not None and not rec:
            v, txt, n = gate(rows, ref_fn)
            _now = now_ms if now_ms is not None else int(time.time() * 1000)
            if n is not None and _now < rows[n - 1][2] + SETTLE_MS:
                settling = True                       # the REF window's kept fills may still be open / unexported → not frozen yet
            elif n is not None:
                rec = dict(verdict=v, detail=txt, n=n, run_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(_now / 1000)))
                if state_path:
                    st[line] = rec
                    if not _save(st, state_path):
                        st.pop(line, None)
                        rec, unsaved = None, True
        if rec and rec.get("verdict"):
            out.append(f"  {line}: FROZEN ({rec.get('run_at')}): {rec.get('verdict')} — {rec.get('detail')} (alert only · operator switches)"
                       + ("" if state_path else " (not saved — no state file)"))
        else:
            v, txt, _ = gate(rows, ref_fn)
            out.append(f"  {line}: {v} — {txt}" + (" (⚠ state file unreadable — nothing frozen this run)" if st is None else "")
                       + (" (⏳ settling: frozen once the window's kept fills have had 48 h to close)" if settling else "")
                       + (" (⚠ state not saved — not frozen, re-read next run)" if unsaved else ""))
        if rows:
            xs = [x for _, x, _ in rows]
            out.append(f"    running: N {len(xs)} · WR {100.0 * sum(1 for x in xs if x > 0) / len(xs):.0f} % · mean {sum(xs) / len(xs):+.3f} % · "
                       f"{len({d for d, _, _ in rows})} days ({len(coh)} blocked in the cohort)")
        else:
            out.append(f"    running: no priced blocked signal yet ({len(coh)} in the cohort)")
        if halted:
            out.append(f"    ⏸ pricing halted at {halted} — later signals wait for the next run (the counted signals stay a prefix)")
        if nodata:
            out.append(f"    NO_DATA for good (left out): {', '.join(nodata)}")
    SO._save(cache, SO.CACHE)
    out.append("  ⚠ funding not priced · slip = an upper bound (the shadow notional skips the depth / liquidity caps)")
    return "\n".join(out)


def run(now_ms=None):
    j, f = SO.load()
    return [f"## 🌡⏱ EXH_BEAR80 + EXH_EARLY_SPIKE (DL {DL})", "", "```", report(j, f, now_ms=now_ms), "```", ""]


def main():
    j, f = SO.load()
    print(report(j, f))


if __name__ == "__main__":
    main()
