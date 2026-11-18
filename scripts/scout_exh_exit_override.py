#!/usr/bin/env python3
"""🎯 Scout — EXHAUSTION_SHORT exit-override revert gate (DECISION_LOG 579, 2026-11-18, operator-approved PAPER, DECLARED OVERRIDE —
ALERT-only, never changes config).

OVERRIDE        trading_config.json runs TP min(20 × ATR, 10 %) ≈ 10 % (resting maker) · SL min(1.5 × ATR, 6 %) · 24 h, instead of the shipped
                study exit TP min(3 × ATR, 5 %) · SL min(1.5 × ATR, 5 %) · 24 h. Evidence reports/EXH_EXIT_GRID_2026-11-18.md: +0.65 vs
                +0.46 %/trade, worst half-year +0.17, holdout 2023-H1 +0.46 vs +0.11; fails the family-wise null (p 0.37) → declared override.
COHORT          CLOSED EXHAUSTION_SHORT fills of the override era (services.exhaustion.exit_era_579: stamped TP > 5 % or > 3 × the stamped
                ATR — the shipped exit can do neither), operator-closed rows excluded (scout_exhaustion_short.load_fills).
PAIRED PRICE    each fill is re-priced from Binance 1m klines under BOTH exits with the same model: entry at the booked entry price (already
                the real fill — taker fee only, no extra slippage) · the path starts at the first WHOLE minute after the fill (the fill
                minute's earlier high / low is never used) · TP = a resting maker limit that fills only when a 1m low trades THROUGH it
                (strictly below) · SL taker, gap-aware (the worse of the stop and the bar open) · both in one bar → the stop · time exit at
                the close of minute 1,440 · taker 0.05 % + the fill's modelled slippage on taker exits, maker 0.02 % on the TP.
                ⚠ Funding is NOT priced: the override leg usually holds longer (it rides past the 5 % TP), so on squeezed coins with
                hourly funding the Δ is biased in the override's FAVOUR — read the booked funding_fee_usd beside a KEEP verdict.
                Δ = override − shipped, per fill.
GATE (frozen, deep-review redesign 2026-11-18)  read ONCE at each checkpoint (the first 40 · 80 · 120 priced fills by open time,
                each also needing ≥ 10 distinct days) — never re-judged between checkpoints (no optional stopping). At a checkpoint:
                the one-sided 90 % UPPER bound of mean Δ (day-block bootstrap, 2,000 resamples, seed 579) < 0 → REVERT (the override is
                shown worse) · else KEEP. Why not "mean Δ ≤ 0": per-fill Δ has SD ≈ 4.8 % against a studied edge of +0.28 %, so at N 40
                that rule flags REVERT ≈ 35–48 % of the time even when the override is exactly as good as studied (deep review,
                bootstrap of the study's 4,140 paired fills). This rule, simulated (Δ ~ N(μ, 4.8), 12 days): P(REVERT) at μ +0.28 ≈ 8 %
                (40 / 120 fills) per look — ≈ 10–13 % across the three frozen looks (deep-review normal-approx sim: 9.5 %; at μ 0 ≈ 21 %)
                · μ 0 ≈ 12 % per look · μ −0.28 ≈ 22 % / 31 % · μ −1.0 ≈ 57 % / 81 %. Low power on small losses — stated.
                REPLICA CHECK (Sep-23 rule): the override leg is what the bot ran, so the model must reproduce the booked fills — exit
                class match ≥ 80 % and mean |model − booked| ≤ 0.30 pts on the checkpoint's fills, else that checkpoint freezes
                UNRELIABLE (no verdict). Time exit lands ≈ 1 min after the engine's fill + 24 h (stated).
                The mean Δ without the top 5 % (by Δ) is printed as information only (the study itself reads ≈ 0 there).
                REVERT = restore 3 / 5 / 1.5 / 5 on a FLAT EXHAUSTION book (an open override row would be clamped to the old caps
                mid-trade) — operator decides; alert only. Each checkpoint verdict is FROZEN in reports/SCOUT_EXH_EXIT_OVERRIDE.json the
                first time it is reached (never recomputed); the counted fills are always an OPEN-TIME PREFIX (pricing stops at the first
                fill it cannot price this run — no gaps); a fill that is NO_DATA for good (a gapped / delisted path) is listed and left out.
                Levels are anchored to the booked fill price, as the engine does (row_levels / the resting TP at fill × (1 − TP %)).
NETWORK         fapi /fapi/v1/klines (1m, 1,440 bars, weight 10 per fill), ≥ 0.6 s apart, stops above 1,000 used weight / on 418 / 429;
                complete paths cached in reports/.exh_exit_override_cache.json (a path still short 2 h after it should be complete is
                cached as NO_DATA and counted).
Usage: venv/bin/python scripts/scout_exh_exit_override.py"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)

DL = 579
CHECKPOINTS, GATE_DAYS, TOP_TRIM, BOOT_N, BOOT_SEED, UPPER_Q = (40, 80, 120), 10, 0.05, 2000, 579, 0.90
SHIPPED = dict(tp_atr=3.0, tp_cap=5.0, sl_atr=1.5, sl_cap=5.0, hold_h=24.0)
OVERRIDE = dict(tp_atr=20.0, tp_cap=10.0, sl_atr=1.5, sl_cap=6.0, hold_h=24.0)
TAKER, MAKER = 0.05, 0.02
CACHE = os.path.join(_R, "reports", ".exh_exit_override_cache.json")
STATE = os.path.join(_R, "reports", "SCOUT_EXH_EXIT_OVERRIDE.json")
UA = "Mozilla/5.0"
HOLD_MIN = int(SHIPPED["hold_h"] * 60)
WEIGHT_STOP = 1000


class NetStop(Exception):
    """418 / 429 / the used-weight ceiling — stop fetching for this run."""


_weight_hit = [False]


def levels(atr_pct, ex):
    """(TP %, SL %) price distances: min(k × ATR, cap). None on a bad ATR. Pure."""
    try:
        a = float(atr_pct)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or a <= 0:
        return None
    return min(ex["tp_atr"] * a, ex["tp_cap"]), min(ex["sl_atr"] * a, ex["sl_cap"])


def price_short(entry, bars, tp_pct, sl_pct, hold_bars, slip_pct=0.0):
    """Net % of a SHORT filled at `entry` (the real fill — taker fee only) over 1m bars [(open, high, low, close), …] starting at the
    first whole minute after the fill. TP maker through (low < TP price) · SL taker gap-aware + slip · same bar → the stop · time exit
    at the close of bar hold_bars−1 (+ slip). Pure."""
    try:
        entry = float(entry)
    except (TypeError, ValueError):
        return None, "NO_DATA"
    if not bars or not math.isfinite(entry) or entry <= 0:
        return None, "NO_DATA"
    tp_px = entry * (1 - tp_pct / 100.0)
    sl_px = entry * (1 + sl_pct / 100.0)
    s = float(slip_pct) if (slip_pct is not None and math.isfinite(float(slip_pct)) and float(slip_pct) > 0) else 0.0
    n = min(len(bars), hold_bars)
    for o, h, l, c in bars[:n]:
        if h >= sl_px:                                            # the stop first (also when the TP traded in the same bar)
            px = max(sl_px, o)
            return (1 - px / entry) * 100 - TAKER - TAKER - s, "SL"
        if l < tp_px:
            return (1 - tp_px / entry) * 100 - TAKER - MAKER, "TP"
    if len(bars) < hold_bars:
        return None, "OPEN"                                       # the path is not complete yet
    c = bars[hold_bars - 1][3]
    return (1 - c / entry) * 100 - TAKER - TAKER - s, "TIME"


def upper_bound(rows, q=UPPER_Q, n_boot=BOOT_N, seed=BOOT_SEED):
    """one-sided q upper bound of the mean Δ by a day-block bootstrap (resample whole days, deterministic seed). Pure."""
    import random
    by = {}
    for d, x in rows:
        by.setdefault(d, []).append(x)
    days = list(by.values())
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [days[rng.randrange(len(days))] for _ in range(len(days))]
        tot = sum(sum(v) for v in pick); n = sum(len(v) for v in pick)
        means.append(tot / n)
    means.sort()
    return means[min(len(means) - 1, int(math.ceil(q * len(means))) - 1)]


def gate(rows):
    """rows = [(day, delta)] in OPEN-time order → (verdict, text). Read once per checkpoint: the latest checkpoint the cohort has
    reached decides (its first N fills only). Pure. Non-finite deltas are dropped."""
    rows = [(d, x) for d, x in rows if x is not None and math.isfinite(x)]
    rule = (f"gate: read once at {' · '.join(map(str, CHECKPOINTS))} fills (each ≥ {GATE_DAYS} days) → the one-sided "
            f"{UPPER_Q:.0%} upper bound of mean Δ (day-block bootstrap) < 0 → REVERT · else KEEP (false REVERT ≈ 8 % per look, ≈ 10–13 % across the 3 looks at the studied +0.28)")
    n = len(rows)
    reached = [c for c in CHECKPOINTS if n >= c and len({d for d, _ in rows[:c]}) >= GATE_DAYS]
    mean_all = (sum(x for _, x in rows) / n) if n else None
    if not reached:
        days = len({d for d, _ in rows})
        run = f" · running mean Δ {mean_all:+.3f} (not judged)" if n else ""
        return "COLLECTING", f"⏳ {n}/{CHECKPOINTS[0]} fills on {days}/{GATE_DAYS} days{run} — {rule}"
    c = reached[-1]
    cut = rows[:c]
    mean = sum(x for _, x in cut) / c
    ub = upper_bound(cut)
    k = max(1, int(math.floor(c * TOP_TRIM)))
    tmean = sum(sorted(x for _, x in cut)[:c - k]) / (c - k)
    info = f"checkpoint {c}: mean Δ {mean:+.3f} · {UPPER_Q:.0%} upper bound {ub:+.3f} · without the top {k} Δ {tmean:+.3f} (info)"
    if ub < 0:
        return "REVERT", f"🔴 {info} — the override is shown worse: flag restore the shipped exit on a flat book (operator decides)"
    return "KEEP", f"🟢 {info} — not shown worse; next read at {next((x for x in CHECKPOINTS if x > c), 'none (final)')} fills"


def is_override_era(tp_stamp, atr_stamp) -> bool:
    from services.exhaustion import exit_era_579
    return exit_era_579(tp_stamp, atr_stamp)


def _load_cache():
    try:
        with open(CACHE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(c):
    try:
        with open(CACHE, "w") as f:
            json.dump(c, f)
    except Exception:
        pass


_last_call = [0.0]


def fetch_1m(pair, start_ms, n=HOLD_MIN):
    """[(t_ms, o, h, l, c)] 1m klines from start_ms. 418 / 429 → NetStop; above the used-weight ceiling the rows already paid for are
    returned and the NEXT call raises NetStop. Other HTTP / URL errors raise."""
    if _weight_hit[0]:
        raise NetStop(f"used weight > {WEIGHT_STOP}")
    wait = 0.6 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=1m&startTime={int(start_ms)}&limit={int(min(1500, n))}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            _last_call[0] = time.time()
            used = r.headers.get("X-MBX-USED-WEIGHT-1M")
            k = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        _last_call[0] = time.time()
        u = e.headers.get("X-MBX-USED-WEIGHT-1M") if e.headers else None
        if e.code in (418, 429) or (u is not None and str(u).isdigit() and int(u) > WEIGHT_STOP):
            raise NetStop(f"HTTP {e.code}")
        raise
    if used is not None and str(used).isdigit() and int(used) > WEIGHT_STOP:
        _weight_hit[0] = True
    return [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])) for x in k]


def path_bars(rows, t1, hold=HOLD_MIN):
    """the first `hold` 1m bars from t1 as (o, h, l, c); None when any minute is missing (a gap would shift the time exit). Pure."""
    want = {t1 + i * 60_000 for i in range(hold)}
    got = {int(r[0]): r for r in rows if int(r[0]) in want}
    if len(got) < hold:
        return None
    return [tuple(got[t1 + i * 60_000][1:5]) for i in range(hold)]


def price_fill(pair, opened_at, entry, atr, slip, cache, now_ms=None):
    """→ {'override': (net, why), 'shipped': (net, why)} · 'NO_DATA' (unpriceable for good) · None (the 24 h path is not over / complete yet).
    The cache keeps the two PRICED results per fill (never the bars)."""
    import pandas as pd
    t1 = int(pd.Timestamp(str(opened_at)[:19]).floor("1min").value // 1_000_000) + 60_000   # the first whole minute after the fill
    key = f"{pair}|{t1}"
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    hit = cache.get(key)
    if hit == "NO_DATA":
        return "NO_DATA"
    if isinstance(hit, dict) and {"override", "shipped"} <= set(hit):
        return {k: tuple(v) for k, v in hit.items()}
    if now_ms < t1 + HOLD_MIN * 60_000 + 60_000:
        return None                                               # the 24 h path is not over yet
    out = {}
    for name, ex in (("override", OVERRIDE), ("shipped", SHIPPED)):
        if levels(atr, ex) is None:
            cache[key] = "NO_DATA"
            return "NO_DATA"
    bars = path_bars(fetch_1m(pair, t1, HOLD_MIN), t1)
    if bars is None:
        if now_ms > t1 + HOLD_MIN * 60_000 + 2 * 3_600_000:
            cache[key] = "NO_DATA"
            return "NO_DATA"
        return None
    for name, ex in (("override", OVERRIDE), ("shipped", SHIPPED)):
        lv = levels(atr, ex)
        net, why = price_short(entry, bars, lv[0], lv[1], HOLD_MIN, slip)
        if net is None:
            if why == "NO_DATA":
                cache[key] = "NO_DATA"
                return "NO_DATA"
            return None
        out[name] = (net, why)
    cache[key] = {k: list(v) for k, v in out.items()}
    return out


def _load_state(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return None                                               # unreadable → never freeze over it


def _save_state(st, path):
    try:
        with open(path, "w") as f:
            json.dump(st, f, indent=1)
    except Exception:
        pass


REPLICA_MATCH_MIN, REPLICA_GAP_MAX = 0.80, 0.30


def booked_kind(close_reason) -> str:
    """the booked exit class: TP / SL / TIME (anything else, e.g. a max-hold or backstop close, counts as TIME). Pure."""
    s = str(close_reason or "").upper()
    if "TP" in s or "TAKE_PROFIT" in s:
        return "TP"
    if "STOP" in s or s.startswith("SL") or "_SL" in s:
        return "SL"
    return "TIME"


def booked_price_pct(r):
    """the booked % WITHOUT funding (the engine folds funding_fee_usd ÷ notional × 100 into pnl_percentage; the model prices price only).
    None on a bad row. Pure."""
    try:
        pc = float(r.pnl_percentage)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pc):
        return None
    try:
        fu, n = float(getattr(r, "funding_fee_usd", None)), float(getattr(r, "notional_value", None))
        if math.isfinite(fu) and math.isfinite(n) and n > 0:
            pc -= fu / n * 100.0
    except (TypeError, ValueError):
        pass
    return pc


def replica_check(pairs):
    """pairs = [(model override net, model why, booked pnl %, booked close_reason)] → (match rate, mean |gap|, reliable?). Pure.
    The Sep-23 rule: the override leg is what LIVE ran, so the model must reproduce the booked fill before Δ is trusted."""
    pairs = [p for p in pairs if p[2] is not None and math.isfinite(p[2])]
    if not pairs:
        return None, None, False
    m = sum(1 for n, w, b, r in pairs if w == booked_kind(r)) / len(pairs)
    g = sum(abs(n - b) for n, w, b, r in pairs) / len(pairs)
    return m, g, (m >= REPLICA_MATCH_MIN and g <= REPLICA_GAP_MAX)


def report(fills, now_ms=None, state_path=STATE):
    out = [f"🎯 EXHAUSTION_SHORT exit-override revert gate (DECISION_LOG {DL}, paper, declared override) · 🌐 FRENZY universe only (WIDE: DL 585 gate)"]
    import scout_exhaustion_short as _SC
    fills = _SC.frenzy_universe(fills, "entry_exh_universe")   # 🌐 (585 review I2) WIDE fills out
    _weight_hit[0] = False
    cache = _load_cache()
    rows, lines, nodata, reps = [], [], [], []
    halted = None                                                 # the first fill this run could not price → everything after waits
    era = [r for r in fills.itertuples()
           if is_override_era(getattr(r, "entry_exh_tp_pct", None), getattr(r, "entry_exh_atr_pct", None))]
    for r in era:
        if halted:
            break
        try:
            p = price_fill(r.pair, r.opened_at, r.entry_price, r.entry_exh_atr_pct, r.slip, cache, now_ms)
        except NetStop as e:
            halted = f"{r.pair} {str(r.opened_at)[:16]}: {e}"
            break
        except (urllib.error.URLError, OSError, ValueError, TypeError) as e:
            halted = f"{r.pair} {str(r.opened_at)[:16]}: {type(e).__name__}"
            break
        if p is None:
            halted = f"{r.pair} {str(r.opened_at)[:16]}: path not complete yet"
            break
        if p == "NO_DATA":
            nodata.append(f"{r.pair} {str(r.opened_at)[:16]}")
            continue
        d = p["override"][0] - p["shipped"][0]
        rows.append((str(r.o)[:10], d))
        bk = booked_price_pct(r)
        sl_back = (r.slip or 0.0) if p["override"][1] in ("SL", "TIME") and r.slip is not None and math.isfinite(float(r.slip)) else 0.0
        reps.append((p["override"][0] + sl_back, p["override"][1], bk, getattr(r, "close_reason", None)))   # paper books no exit slip
        lines.append(f"    {str(r.o)[:16]} {r.pair:<14} booked {(bk if bk is not None else float('nan')):+6.2f}% (price only) · model override {p['override'][0]:+6.2f}% "
                     f"({p['override'][1]}) vs shipped {p['shipped'][0]:+6.2f}% ({p['shipped'][1]}) · Δ {d:+6.2f}")
    _save_cache(cache)
    st = _load_state(state_path) if state_path else {}
    frozen_now = False
    if st is not None:
        for c in CHECKPOINTS:                                     # freeze each checkpoint ONCE, on its own first-c prefix
            k = str(c)
            if k in st or len(rows) < c or len({d for d, _ in rows[:c]}) < GATE_DAYS:
                continue
            v, txt = gate(rows[:c])
            m, g, ok_rep = replica_check(reps[:c])
            if not ok_rep:
                v = "UNRELIABLE"
                txt = (f"the model does not reproduce the booked override fills (exit match {m if m is not None else float('nan'):.0%} "
                       f"vs ≥ {REPLICA_MATCH_MIN:.0%}, mean |model − booked| {g if g is not None else float('nan'):.2f} vs ≤ {REPLICA_GAP_MAX} pts) "
                       f"— no verdict; fix the replica first (gate read was: {txt})")
            st[k] = dict(verdict=v, detail=txt, run_at=(time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime((now_ms or time.time() * 1000) / 1000))))
            frozen_now = True
        if frozen_now and state_path:
            _save_state(st, state_path)
    for c in CHECKPOINTS:
        rec = (st or {}).get(str(c))
        if rec:
            out.append(f"  FROZEN checkpoint {c} ({rec.get('run_at')}): {rec.get('verdict')} — {rec.get('detail')}")
        elif len(rows) >= c and len({d for d, _ in rows[:c]}) < GATE_DAYS:
            out.append(f"  checkpoint {c} SKIPPED for good: its first {c} fills span {len({d for d, _ in rows[:c]})} days (< {GATE_DAYS})")
    m, g, ok_rep = replica_check(reps)
    if m is not None:
        out.append(f"  replica check (override leg vs booked, all priced fills): exit match {m:.0%} · mean |model − booked| {g:.2f} pts · "
                   f"{'OK' if ok_rep else 'NOT RELIABLE — a checkpoint reached now freezes UNRELIABLE'}")
    if st is None:
        out.append(f"  ⚠ state file unreadable ({os.path.basename(state_path)}) — nothing frozen this run (operator restore)")
    v, txt = gate(rows)
    out.append(f"  now: {v} — {txt}" + (" (information only — the state file is unreadable, nothing is decided)" if st is None else
                                        "" if not any(st.get(str(c)) for c in CHECKPOINTS) else " (information; the frozen reads above decide)"))
    if halted:
        out.append(f"  ⏸ pricing halted at {halted} — later fills wait for the next run (the counted fills stay an open-time prefix)")
    if nodata:
        out.append(f"  NO_DATA for good (left out): {', '.join(nodata)}")
    out.append("  ⚠ funding not priced — biased toward the override (it holds longer); compare booked funding_fee_usd before acting on KEEP")
    out.extend(lines)
    return "\n".join(out)


def load(paths=None):
    import scout_exhaustion_short as SC
    return SC.load_fills(paths, extra=("entry_price", "entry_exh_atr_pct", "funding_fee_usd"))


def main():
    print(report(load()))


if __name__ == "__main__":
    main()
