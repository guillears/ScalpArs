#!/usr/bin/env python3
"""📉🌡 Scout — DIST_CRASH_FUNDING revert gate (DECISION_LOG 651, 2026-11-27, operator "yes build it"; ALERT-only, never changes config).

RULE (engine)   services/distribution.crash_funding_block: a DISTRIBUTION_SHORT signal that passed every rule incl. the RSI1h filter is
                BLOCKED when BTC's last closed daily close is ≤ −20 % below the max daily high of the 30 closed days ending with it AND the
                coin's funding at the decision ≥ +0.005 % (longs still pay). Unreadable → trade (DIST_CRASH_FUNDING_UNREAD).
EVIDENCE        reports/DIST_DEEP_DIVE_2026-11-27.md: backtest blocked 43 · 23 % · −2.01 %/fill (both blocks, 8 / 9 crash episodes negative,
                family-wise p 0.026); live 11-19 → 11-27 out of sample: 4 blocked, all losers.
COHORT          BLOCKED DIST_CRASH_FUNDING journal rows (decisions exports, e = DISTRIBUTION), signal time ≥ the DL 651 commit + 10 min.
SHADOW PRICE    scout_distribution_586.price_blocked (Exit B on the 1m path from the decision price, modelled slip at the median kept-fill
                notional, taker / maker fees) — the same pricer as the RSI1h filter's blocked-side gate. Funding NOT priced (a blocked short
                here would have PAID funding → the shadow flatters the blocked side → leans toward REVERT, the safe side).
GATE (frozen)   read ONCE at the first prefix of ≥ 15 priced blocked signals spanning ≥ 3 UTC days: blocked mean ≥ 0 → REVERT (flag: switch
                distribution_crash_funding_block_enabled off — operator decides) · else KEEP. State reports/SCOUT_DIST_CRASH_FUNDING_651.json.
Usage: venv/bin/python scripts/scout_dist_crash_funding_651.py"""
import os
import sys

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)

import scout_distribution_586 as DS            # noqa: E402  load_journal / price_blocked / _freeze / _W
import scout_gatekit as GK                     # noqa: E402

DL = 651
BLOCK = "DIST_CRASH_FUNDING"
GATE_N, GATE_DAYS = 15, 3
STATE = os.path.join(_R, "reports", "SCOUT_DIST_CRASH_FUNDING_651.json")
CACHE = os.path.join(_R, "reports", ".dist_crash_funding_cache.json")


def one_per_coin(journal, hold_ms=24 * 3_600_000):
    """(deep review M2) drop a blocked DIST_CRASH_FUNDING row when the same pair already has a KEPT blocked row whose 24 h hold is still
    open (the backtest's one-position-per-coin). Other rows untouched. Pure."""
    import pandas as pd
    if journal is None or not len(journal):
        return journal
    j = journal.copy()
    t = pd.to_datetime(j.dist_signal_at.astype(str).str[:19], errors="coerce")
    keep, last = [], {}
    for i, (b, p, ts) in enumerate(zip(j.dist_block.astype(str), j.pair.astype(str), t)):
        if b != BLOCK or pd.isna(ts):
            keep.append(True); continue
        lt = last.get(p)
        if lt is not None and (ts - lt).total_seconds() * 1000 < hold_ms:
            keep.append(False); continue
        last[p] = ts; keep.append(True)
    order = t.argsort(kind="stable")
    return j.iloc[order][pd.Series(keep).iloc[order].values]


def gate(rows):
    """rows = [(utc day, net %)] in signal-time order → (verdict, text, n). Pure."""
    rows = [(d, x) for d, x in rows if x is not None]
    for n in range(GATE_N, len(rows) + 1):
        days = {d for d, _ in rows[:n]}
        if len(days) >= GATE_DAYS:
            xs = [x for _, x in rows[:n]]
            m = sum(xs) / n
            wr = 100.0 * sum(1 for x in xs if x > 0) / n
            info = f"first {n} blocked signals ({len(days)} days): mean {m:+.3f} % · WR {wr:.0f} % (Exit B, modelled slip, ex-funding)"
            if m >= 0:
                return "REVERT", f"🔴 {info} ≥ 0 → the filter removed trades that did not lose: flag the switch off (operator decides)", n
            return "KEEP", f"🟢 {info} < 0 → the blocked signals lost: keep the filter", n
    return "COLLECTING", (f"⏳ {len(rows)}/{GATE_N} priced on {len({d for d, _ in rows})}/{GATE_DAYS} days — gate: first {GATE_N}+ blocked "
                          f"signals on ≥ {GATE_DAYS} days, mean ≥ 0 → REVERT · else KEEP — read once, frozen"), None


def report(journal=None, frame=None, now_ms=None, state_path=STATE, floor_ms="auto", pricer=None, cache_path=CACHE):
    out = [f"## 📉🌡 DIST_CRASH_FUNDING (DECISION_LOG {DL}) — revert gate (alert only)",
           "blocked = BTC ≤ −20 % below its 30 d high ∧ funding ≥ +0.005 % · evidence: backtest 43 · 23 % · −2.01 %/fill, 8 / 9 crash episodes "
           "negative, luck p 0.026; live 11-19 → 11-27: 4 / 4 blocked were losers"]
    if floor_ms == "auto":
        floor_ms = GK.dl_deploy_ms(DL, lag_min=10, label=f"DIST_CRASH_FUNDING (DECISION_LOG {DL})")
    if floor_ms is None:
        out.append(f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.")
        return "\n".join(out)
    j = DS.load_journal() if journal is None else journal
    a = DS._W.load(sleeve=DS.DISTRIBUTION_SHORT, cols=DS.COLS) if frame is None else frame
    nots = sorted(x for x in (DS._W._f(v) for v in (a.notional_value.tolist() if len(a) and "notional_value" in a.columns else [])) if x)
    med = nots[len(nots) // 2] if nots else None
    j = one_per_coin(j)
    rows, lines, halted = DS.price_blocked(j, floor_ms, med, now_ms, pricer, cache_path, block=BLOCK)
    g = gate(rows)
    st = DS._freeze(state_path, g if med is not None else (g[0], g[1], None), now_ms) if state_path else None
    if st and st.get("verdict"):
        out.append(f"FROZEN ({st.get('run_at')}): {st['verdict']} — {st.get('detail')}")
    else:
        out.append(f"now: {g[0]} — {g[1]}" + (" (⚠ state file unreadable — nothing frozen)" if state_path and st is None else ""))
    if med is None:
        out.append("  ⚠ no kept DIST fill notional yet: blocked signals priced at 0 slip (flatters them) — not frozen until one exists")
    if halted:
        out.append(f"  ⏸ pricing halted at {halted} — later signals wait for the next run")
    if j is not None and len(j):
        jb = j[j.dist_block.astype(str) == BLOCK]
        out.append(f"  cohort: {len(jb)} blocked rows after one-per-coin · {jb.pair.nunique()} distinct pairs")
    out.append("  ⚠ funding not priced — a blocked short here would have RECEIVED it (≈ +0.03–0.06 %) → leans toward KEEP")
    out.extend(lines[-20:])
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
