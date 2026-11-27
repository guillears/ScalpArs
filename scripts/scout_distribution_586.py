#!/usr/bin/env python3
"""📉 Scout — DISTRIBUTION_SHORT revert gate (DECISION_LOG 586, 2026-11-18, operator-approved PAPER run, DECLARED OVERRIDE — ALERT-only,
never changes config).

RULE (engine)   services/distribution.py: on a FRENZY-universe coin EXHAUSTION tracks (≤ 72 h after the spike), a CLOSED red 15m candle with
                quote volume ≥ 5 × the mean of the 20 before it (VSW15) while the top-trader POSITION long/short ratio fell ≥ 0.01097 over
                4 h → SHORT at market; Exit B = EXHAUSTION's live exit (TP min(20 × ATR, 10 %) resting maker · SL min(1.5 × ATR, 6 %) · 24 h).
COHORT          CLOSED DISTRIBUTION_SHORT fills opened ≥ the DL 586 commit + 10 min (orders exports, paper + live; dedup; CLOSED wins);
                operator-closed out. NET = each fill re-priced with EXHAUSTION's modelled slippage (entry_dist_model_slip_pct).
GATE (frozen)   services.distribution.upper_bound_gate: the first 30 fills by OPEN time, extended until they span ≥ 10 days → the one-sided
                80 % UPPER bound of their mean net (day-block bootstrap) < 0 → flag "switch distribution_short_enabled off" · else KEEP.
JOURNAL         e=DISTRIBUTION rows of the decisions exports: the tally per status / block (what the rule refused, what the open path refused).
🔒 RSI1h FILTER (DECISION_LOG 586, operator 11-18, DECLARED OVERRIDE: keep only 1h RSI ≥ 74.07) — two TIGHT revert gates, alert only, each
                frozen once (its own state file):
  (a) KEPT      the kept fills (the cohort above): the first ≥ 60 by open time spanning ≥ 10 days → mean net (re-priced) ≤ +0.30 % → flag
                "disarm the RSI1h filter / review the sleeve" (the study's kept side: HOLDOUT +0.79, DESIGN +1.05) · else KEEP.
  (b) BLOCKED   journal rows blocked DIST_RSI1H (the L1 signals the cut removed) at / after the floor, each priced as if traded with Exit B on
                1m klines (scout_exh_ssc_oivol.price_signal: entry = the journaled decision price, TP / SL = the journaled distances, TP a
                maker limit filled only through it, SL taker gap-aware, 24 h, taker / maker fees, the modelled slip at the median kept notional
                on the entry and a taker exit — 0 when no kept fill exists yet, stated): the first ≥ 30 spanning ≥ 10 UTC days → the one-sided
                80 % LOWER bound of the blocked mean (day-block bootstrap) > +0.30 % → flag "the filter blocks winners" · else KEEP.
STATE           reports/SCOUT_DISTRIBUTION_586.json · SCOUT_DIST_RSI1H_KEPT_586.json · SCOUT_DIST_RSI1H_BLOCKED_586.json (gitignored) — frozen once;
                priced blocked signals cached in reports/.dist_rsi1h_cache.json (gitignored).
NETWORK         (b) only: fapi 1m klines (1,440 bars, weight 10 per blocked signal; scout_exh_exit_override's budget: ≥ 0.6 s apart, stops above
                1,000 used weight / on 418 / 429). Usage: venv/bin/python -I scripts/scout_distribution_586.py"""
import json
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402
import scout_exh_wide_585 as _W    # noqa: E402  (shared loader / re-pricing / frozen-state helpers)
from services.distribution import upper_bound_gate, DISTRIBUTION_SHORT   # noqa: E402

DL = 586
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "notional_value",
        "entry_dist_model_slip_pct", "entry_dist_q5_usd", "entry_depth_mode", "entry_depth_walk_pct")
DEC_PATTERNS = (os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"),)
STATE = os.path.join(_R, "reports", "SCOUT_DISTRIBUTION_586.json")
STATE_KEPT = os.path.join(_R, "reports", "SCOUT_DIST_RSI1H_KEPT_586.json")
STATE_BLOCKED = os.path.join(_R, "reports", "SCOUT_DIST_RSI1H_BLOCKED_586.json")
CACHE_BLOCKED = os.path.join(_R, "reports", ".dist_rsi1h_cache.json")
KEPT_N, KEPT_DAYS, KEPT_MIN = 60, 10, 0.30
BLK_N, BLK_DAYS, BLK_LB = 30, 10, 0.30
EXIT_B = dict(tp_atr=20.0, tp_cap=10.0, sl_atr=1.5, sl_cap=6.0, hold_h=24.0)
JCOLS = ("e", "pair", "dist_signal_at", "dist_status", "dist_block", "dist_decision_price", "dist_tp_pct", "dist_sl_pct", "dist_atr_pct",
         "dist_q5_usd", "dist_rsi1h")


def kept_gate(rows):
    """rows = [(opened_at datetime, net %)] by open time → (verdict, text, n). The first prefix of ≥ KEPT_N spanning ≥ KEPT_DAYS days decides:
    mean ≤ +0.30 → DISARM · else KEEP. Before: COLLECTING. Pure."""
    rows = sorted([(t, x) for t, x in rows if t is not None and x is not None], key=lambda r: r[0])
    for n in range(KEPT_N, len(rows) + 1):
        span = (rows[n - 1][0] - rows[0][0]).total_seconds() / 86400.0
        if span >= KEPT_DAYS:
            m = sum(x for _, x in rows[:n]) / n
            info = f"first {n} kept fills ({span:.1f} days): mean {m:+.3f} % (re-priced)"
            if m <= KEPT_MIN:
                return "DISARM", f"🔴 {info} ≤ +{KEPT_MIN:.2f} → flag: disarm the RSI1h filter / review the sleeve (operator decides)", n
            return "KEEP", f"🟢 {info} > +{KEPT_MIN:.2f} → keep the filter", n
    return "COLLECTING", f"⏳ {len(rows)}/{KEPT_N} kept fills — gate: ≥ {KEPT_N} over ≥ {KEPT_DAYS} days, mean ≤ +{KEPT_MIN:.2f} → flag disarm", None


def blocked_gate(rows):
    """rows = [(utc day, net %)] in signal-time order → (verdict, text, n). The first prefix of ≥ BLK_N spanning ≥ BLK_DAYS days decides:
    one-sided 80 % LOWER bound of the blocked mean (day-block bootstrap, scout_exh_ssc_oivol.lower_bound) > +0.30 → BLOCKS_WINNERS · else KEEP."""
    import scout_exh_ssc_oivol as SO
    rows = [(d, x) for d, x in rows if x is not None]
    for n in range(BLK_N, len(rows) + 1):
        if len({d for d, _ in rows[:n]}) >= BLK_DAYS:
            m = sum(x for _, x in rows[:n]) / n
            lb = SO.lower_bound(rows[:n])
            info = f"first {n} blocked signals ({len({d for d, _ in rows[:n]})} days): mean {m:+.3f} % · 80 % lower bound {lb:+.3f} %"
            if lb > BLK_LB:
                return "BLOCKS_WINNERS", f"🔴 {info} > +{BLK_LB:.2f} → flag: the RSI1h filter blocks winners (operator decides)", n
            return "KEEP", f"🟢 {info} ≤ +{BLK_LB:.2f} → keep the filter", n
    return "COLLECTING", (f"⏳ {len(rows)}/{BLK_N} priced on {len({d for d, _ in rows})}/{BLK_DAYS} days — gate: 80 % lower bound of the "
                          f"blocked mean > +{BLK_LB:.2f} → flag blocks winners"), None


def _freeze(state_path, gate_out, now_ms=None):
    """the verdict of gate_out = (v, txt, n) is written ONCE (when n is not None) → the state dict (None = unreadable)."""
    v, txt, n = gate_out
    try:
        with open(state_path) as fh:
            st = json.load(fh)
    except FileNotFoundError:
        st = {}
    except Exception:
        return None
    if "verdict" not in st and n is not None:
        st = dict(verdict=v, detail=txt, n=n, run_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime((now_ms or time.time() * 1000) / 1000)))
        try:
            with open(state_path, "w") as fh:
                json.dump(st, fh)
        except Exception:
            pass
    return st


def load_journal(paths=None):
    """the e=DISTRIBUTION decision rows (dedup pair × signal time, the newest export wins). Pure over files."""
    paths = paths if paths is not None else sorted({p for pat in DEC_PATTERNS for p in _EG.glob_exports(pat)}, key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in JCOLS)
            fr.append(x[x.e.astype(str) == "DISTRIBUTION"])
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(JCOLS))
    return pd.concat(fr, ignore_index=True).drop_duplicates(["pair", "dist_signal_at"], keep="last")


def price_blocked(journal, floor_ms, slip_notional=None, now_ms=None, pricer=None, cache_path=CACHE_BLOCKED, block="DIST_RSI1H"):
    """→ (rows [(day, net)], lines, halted) for the journal rows blocked with `block` (default DIST_RSI1H) ≥ floor_ms, priced with Exit B (pricer = the injectable
    scout_exh_ssc_oivol.price_signal). Stops at the first unpriceable-yet signal (the counted rows stay a prefix)."""
    import scout_exh_ssc_oivol as SO
    import scout_exh_exit_override as XO
    from services.exhaustion import model_slip_pct
    pricer = pricer or SO.price_signal
    cache = (SO._load(cache_path) or {}) if cache_path else {}
    out, lines, halted = [], [], None
    if journal is None or not len(journal):
        return out, lines, halted
    j = journal[journal.dist_block.astype(str) == block].copy()   # (651) the block code is a parameter — DIST_CRASH_FUNDING reuses this pricer
    j["t_"] = pd.to_datetime(j.dist_signal_at.astype(str).str[:19], errors="coerce")
    j = j[j.t_.notna()]
    j = j[j.t_ >= pd.Timestamp(int(floor_ms), unit="ms")].sort_values(["t_", "pair"])
    for r in j.itertuples():
        t_ms = int(r.t_.value // 10**6)
        tp, sl = _W._f(r.dist_tp_pct), _W._f(r.dist_sl_pct)
        if not (tp and sl):
            lv = XO.levels(r.dist_atr_pct, EXIT_B)
            if lv is None:
                lines.append(f"    {r.pair} {str(r.dist_signal_at)[:16]}: NO_DATA (no TP / SL / ATR)")
                continue
            tp, sl = lv
        slip = model_slip_pct(slip_notional, _W._f(r.dist_q5_usd)) if slip_notional else 0.0
        try:
            p = pricer(str(r.pair), t_ms, _W._f(r.dist_decision_price), tp, sl, slip, cache, now_ms)
        except Exception as e:   # NetStop / network: halt, the prefix stays
            halted = f"{r.pair}: {type(e).__name__}"
            break
        if p is None:
            halted = f"{r.pair} {str(r.dist_signal_at)[:16]}: path not complete yet"
            break
        if p == "NO_DATA":
            lines.append(f"    {r.pair} {str(r.dist_signal_at)[:16]}: NO_DATA")
            continue
        out.append((r.t_.strftime("%Y-%m-%d"), float(p[0])))
        lines.append(f"    {r.pair} {str(r.dist_signal_at)[:16]} RSI1h {getattr(r, 'dist_rsi1h', None)} · TP {tp:.2f} / SL {sl:.2f} → {p[0]:+.2f} % ({p[1]})")
    if cache_path:
        SO._save(cache, cache_path)
    return out, lines, halted


def journal_tally(paths=None):
    """{status/block: count} of the e=DISTRIBUTION decision rows (dedup pair × signal time). Pure over files."""
    paths = paths if paths is not None else sorted({p for pat in DEC_PATTERNS for p in _EG.glob_exports(pat)}, key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("e", "pair", "dist_signal_at", "dist_status", "dist_block"))
            fr.append(x[x.e.astype(str) == "DISTRIBUTION"])
        except Exception:
            continue
    if not fr:
        return {}
    a = pd.concat(fr, ignore_index=True).drop_duplicates(["pair", "dist_signal_at"], keep="last")
    return a.dist_block.fillna(a.dist_status).astype(str).value_counts().to_dict()


def report(frame=None, since_ms=None, state_path=STATE, now_ms=None, tally=None, journal=None, pricer=None,
           state_kept=STATE_KEPT, state_blocked=STATE_BLOCKED, cache_path=CACHE_BLOCKED):
    out = [f"## 📉 DL {DL} DISTRIBUTION_SHORT — revert gate (alert only)"]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="DL 586 DISTRIBUTION_SHORT")
    if since_ms is None:
        return "\n".join(out + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted."])
    a = _W.load(sleeve=DISTRIBUTION_SHORT, cols=COLS) if frame is None else frame
    a = a[a.o >= pd.Timestamp(int(since_ms), unit="ms")] if len(a) else a
    rows = []
    for r in a.itertuples():
        x = _W.net_of(r, "entry_dist_model_slip_pct", "entry_dist_q5_usd")
        if x is not None:
            rows.append((r.o.to_pydatetime(), x))
    st, v, txt = _W.frozen(rows, state_path, lambda rr: upper_bound_gate(rr, "DISTRIBUTION_SHORT", "distribution_short_enabled", DL), now_ms)
    if st and st.get("verdict"):
        out.append(f"FROZEN ({st.get('run_at')}): {st['verdict']} — {st.get('detail')}")
    out.append(f"now: {v} — {txt}" + (" (information; the frozen read decides)" if st and st.get("verdict") else ""))
    out.append(_W._row("fills", [x for _, x in rows]))
    t = journal_tally() if tally is None else tally
    if t:
        out.append("  journal: " + " · ".join(f"{k} {n}" for k, n in sorted(t.items(), key=lambda kv: -kv[1])))
    # 🔒 the RSI1h filter's two tight revert gates (declared override, frozen once each)
    out.append("### 🔒 RSI1h ≥ 74.07 filter (declared override) — revert gates")
    for lbl, sp_, g in (("(a) KEPT", state_kept, kept_gate(rows)),):
        st2 = _freeze(sp_, g, now_ms)
        out.append(f"  {lbl}: " + (f"FROZEN ({st2.get('run_at')}): {st2['verdict']} — {st2.get('detail')}" if st2 and st2.get("verdict") else f"{g[0]} — {g[1]}"))
    j = load_journal() if journal is None else journal
    nots = [n_ for n_ in a.notional_value.tolist()] if len(a) and "notional_value" in a.columns else []
    nots = sorted(x for x in (_W._f(v) for v in nots) if x)
    med = nots[len(nots) // 2] if nots else None
    brows, blines, halted = price_blocked(j, since_ms, med, now_ms, pricer, cache_path)
    gb = blocked_gate(brows)
    # (586 review M3) never frozen before ≥ 1 kept fill exists: until then the blocked side is priced at 0 slip (it would flatter them)
    st3 = _freeze(state_blocked, gb if med is not None else (gb[0], gb[1], None), now_ms)
    out.append("  (b) BLOCKED: " + (f"FROZEN ({st3.get('run_at')}): {st3['verdict']} — {st3.get('detail')}" if st3 and st3.get("verdict") else f"{gb[0]} — {gb[1]}"))
    if med is None:
        out.append("    ⚠ no kept fill yet: blocked signals priced at 0 modelled slip (a slip-free price flatters them) — the gate does not freeze until ≥ 1 kept fill")
    out.append("    ⚠ DIST_RSI1H_UNREAD blocks are not priced (their RSI is unknown — not part of the blocked-side cohort)")
    if halted:
        out.append(f"    ⏸ pricing halted at {halted} — later signals wait for the next run")
    out.append("    ⚠ funding not priced")
    out.extend(blines[-30:])
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
