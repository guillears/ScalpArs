#!/usr/bin/env python3
"""🏃👁 Scout — TP_RUNNER observe lines: HT + EXH "don't sell at the TP, trail" (DECISION_LOG 636, operator 2026-11-24 "for both build it as
observe only, we will decide once analysing this batch"; OBSERVE only — alert-only, never changes config, public fapi data only).

QUESTION   when a fill reaches its resting maker TP, would a runner (no TP order; a trail armed at the TP level) have booked more?
           Study: reports/TP_RUNNER_2026-11-24.md (scripts/TP_RUNNER_DIVE*_2026-11-24.py, TP_RUNNER_{HT,EXH}_BA_2026-11-24.py).
           HT backtest (today's filters, 118 TP hits): +2.46 pp per hit, positive in all 4 half-years, a lottery edge (9 in 10 hits fall to
           the floor at ≈ −1.3 pp, 1 in 10 runs ≈ +39 pp; the best 3 trades carry the whole gain). Real fills to 11-23: 16 hits, 2 small
           runners, −$893 at one size. EXH backtest (stop kept, no floor): +0.65 pp per hit, today's half-year negative (−0.9), funding ≈ −0.45
           per hit; real fills 8 hits −3.03 pp per hit. Neither is armed: both are tracked here and decided by the operator.
RULES      (each priced from the REAL booked TP fill: touch = closed_at, TP price = exit_price; the 1m bars from the minute AFTER the touch minute)
  HT_RUNNER   FRENZY_WILLY_HT fills on the +6 target (frenzy_willy_tp_target_pct ≥ 6) booked FRENZY_TP: trail 4 × the 15m ATR14 % (at the touch,
              from the 1m bars closed before the touch minute — the study's ruler, whose last 15m bucket is the one still forming at the touch)
              below the running HIGH (best up to the previous bar, starting at the TP price),
              floor at +5 net (E × (100 + 5 + taker) ÷ (100 − taker), taker from trading_config.json); the stop = max(trail, floor); a gap books the open; 24 h after the touch →
              the last bar's close.
  EXH_RUNNER  EXHAUSTION_SHORT TP-7-era fills (services.exhaustion.exit_era_609) booked FRENZY_TP / TP: trail 4 × ATR above the running LOW, NO
              floor, the fill's own stop kept (entry_exh_sl_pct above the entry); the stop = min(trail, own stop); gap → open; 24 h after the touch.
  Δ (pp of entry, vs the booked maker TP) = side × (runner exit after slip − TP price) ÷ entry × 100 − (taker − maker fee, trading_config.json)
              + the funding settled between the touch and the runner exit (public fundingRate; long pays a positive rate, short receives it).
              Exit slip 0.10 % (the study's model slip); the fee gap is charged on entry notional and funding is mark-scaled by the exit price
              ÷ entry (both ≈ the research's convention). Unreadable funding → pending until 48 h, then counted without it AND the freeze is held
              while any deciding fill lacks funding. ⚠ STATED DEVIATION: the runner is walked 24 h from the TOUCH (the studied rule); today's
              engine force-closes HT / EXH 24 h from the OPEN — an armed runner would need its own hold.
🔒 READ    frozen ONCE per line (reports/SCOUT_TP_RUNNER_636.json, gitignored) on the FORWARD cohort only (fills opened ≥ the DL 636 commit):
  HT_RUNNER   the first 40 priced TP hits: Σ Δ > 0 → ARM CANDIDATE (operator decides) · else KEEP TP +6. Info (never deciding): after ≥ 10 floor
              exits a mean floor Δ < −2.0 pp = floor slippage worse than modelled; Σ Δ < −25 pp at any read = the runner is far behind.
              Priced power (backtest distribution): P(Σ < 0 | the edge is real) at 40 hits ≈ 23 %.
  EXH_RUNNER  the first 50 priced TP hits spanning ≥ 10 UTC days: the one-sided 90 % LOWER bound of mean Δ (day-block bootstrap, 2,000, seed 636)
              > 0 → ARM CANDIDATE · else KEEP TP 7. (At the backtest's +0.65 / hit a 50-hit read has ≈ 31 % chance of a negative mean.)
  Freezing waits while a fetch stopped this run, the newest orders export is < the deciding (k-th) fill's open + 24 h (both sleeves
  force-close 24 h after the open, so no earlier-opened fill can still be open), a deciding fill lacks funding, or > 10 % of the forward
  fills up to the k-th are NOT EVALUABLE. The study period
  (fills since each exit era's deploy — DL 608 HT / DL 609 EXH — to the DL 636 commit) is shown as an IN-SAMPLE block, never deciding.
OUTPUT     summaries only (operator 11-23): N · Σ / mean / median Δ · $ at one size (0.61 × $10k) · exit mix · best · without the best · funding.
Usage: venv/bin/python -I scripts/scout_tp_runner_636.py"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import scout_gatekit as _GK          # noqa: E402
import scout_exh_wide_585 as _W      # noqa: E402  (shared orders loader)
import scout_long_stop4 as LS        # noqa: E402  (budgeted, cached public fapi klines)
import scout_dist_tp_shadow as _DT   # noqa: E402  (newest_export)
import scout_exh_exit_override as XO  # noqa: E402  (upper_bound: day-block one-sided bound)
import scout_flatbase_short as _FB   # noqa: E402  (funding_sum_pct: budgeted public fundingRate)
from services.exhaustion import EXHAUSTION_SHORT, exit_era_609   # noqa: E402

DL = 636
STATE = os.path.join(_R, "reports", "SCOUT_TP_RUNNER_636.json")
CACHE = os.path.join(_R, "reports", "backtest_cache", "scout_tp_runner")
HT, EXH = "FRENZY_WILLY_HT", EXHAUSTION_SHORT
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "pnl_percentage", "close_reason", "entry_price", "exit_price",
        "closed_at", "frenzy_willy_tp_target_pct", "entry_exh_tp_pct", "entry_exh_atr_pct", "entry_exh_sl_pct")
TP_REASONS = ("FRENZY_TP", "TP")
K_ATR, HT_FLOOR_NET, HT_TP_MIN, SLIP = 4.0, 5.0, 6.0, 0.10
ATR_N, ATR_TF_MIN, ATR_LOOK_H = 14, 15, 5
MIN1, HOUR, DAY, PAGE = 60_000, 3_600_000, 86_400_000, 900
HOLD_H, STALE_H, EXPORT_HOLD_H = 24, 48, 24
HT_N, HT_FLOOR_MIN_N, HT_FLOOR_WARN, HT_BEHIND = 40, 10, -2.0, -25.0
EXH_N, EXH_DAYS, EXH_Q, BOOT_N, SEED = 50, 10, 0.90, 2000, 636
ONE_SIZE_USD = 0.61 * 10_000
LINES = {HT: "HT_RUNNER", EXH: "EXH_RUNNER"}


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _fees():
    """(maker, taker) exit fee in % from trading_config.json; (0.018, 0.045) default. Never raises."""
    try:
        with open(os.path.join(_R, "trading_config.json")) as fh:
            c = json.load(fh)
        mk, tk = float(c.get("maker_fee", 0.00018)) * 100.0, float(c.get("taker_fee", 0.00045)) * 100.0
        return (mk, tk) if 0.0 <= mk <= tk < 0.2 else (0.018, 0.045)
    except Exception:
        return 0.018, 0.045


MAKER, TAKER = _fees()


def atr15_pct(bars, t_touch):
    """15m ATR14 as % of the last close from the 1m bars [[open ms, o, h, l, c, …]] opening in [touch minute − 5 h, touch minute) — the
    study's ruler (TP_HOLD_2026-11-24.atr15_pct): 15m buckets on epoch boundaries, TR vs the previous bucket's close, mean of the last 14.
    None when < 225 bars or < 14 TRs. PURE."""
    m0 = int(t_touch) // MIN1 * MIN1
    b = [q for q in (bars or []) if m0 - ATR_LOOK_H * HOUR <= int(q[0]) < m0]
    if len(b) < ATR_TF_MIN * ATR_TF_MIN:
        return None
    g = {}
    for q in b:
        k = int(q[0]) // (ATR_TF_MIN * MIN1)
        h, lo, c = float(q[2]), float(q[3]), float(q[4])
        if k in g:
            g[k] = (max(g[k][0], h), min(g[k][1], lo), c)
        else:
            g[k] = (h, lo, c)
    ks = sorted(g)
    tr = []
    for a, b2 in zip(ks[:-1], ks[1:]):
        pc = g[a][2]
        h, lo, _ = g[b2]
        tr.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    last = g[ks[-1]][2]
    if len(tr) < ATR_N or not last > 0:
        return None
    return float(np.mean(tr[-ATR_N:]) / last * 100.0)


def runner_walk(bars, t_touch, tp_px, side, w_pct, guard_px=None):
    """the runner from the TP touch: bars = 1m [[open ms, o, h, l, c, …]]; the walk uses the bars opening in [touch minute + 1 min,
    + 24 h). side +1 (long): best = running HIGH from tp_px (the stop for bar i uses the best up to bar i − 1), trail = best × (1 − w %),
    stop = max(trail, guard_px) (guard = the floor); side −1 (short): best = running LOW, trail = best × (1 + w %), stop = min(trail, guard)
    (guard = the fill's own stop). A bar through the stop books the stop, or its open on a gap. No hit in 24 h → the last bar's close.
    → (why 'trail' | 'guard' | 'time' | 'PENDING', raw exit px or None, exit bar close ms or None). PENDING = the path ends before an exit
    and before 24 h. PURE."""
    m0 = int(t_touch) // MIN1 * MIN1
    end = m0 + MIN1 + HOLD_H * HOUR
    b = [q for q in (bars or []) if m0 + MIN1 <= int(q[0]) < end]
    if not (tp_px and tp_px > 0 and w_pct and w_pct > 0 and side in (1, -1)):
        return "PENDING", None, None
    best = float(tp_px)
    for q in b:
        o, h, lo = float(q[1]), float(q[2]), float(q[3])
        if side > 0:
            tr = best * (1 - w_pct / 100.0)
            st = max(tr, guard_px) if guard_px is not None else tr
            if lo <= st:
                return ("guard" if guard_px is not None and st == guard_px else "trail"), min(st, o), int(q[0]) + MIN1
            best = max(best, h)
        else:
            tr = best * (1 + w_pct / 100.0)
            st = min(tr, guard_px) if guard_px is not None else tr
            if h >= st:
                return ("guard" if guard_px is not None and st == guard_px else "trail"), max(st, o), int(q[0]) + MIN1
            best = min(best, lo)
    if b and int(b[-1][0]) >= end - MIN1:
        return "time", float(b[-1][4]), int(b[-1][0]) + MIN1
    return "PENDING", None, None


def delta_pp(side, E, tp_px, exit_px, funding_pct=0.0, slip=SLIP, maker=None, taker=None):
    """runner Δ vs the booked maker TP, pp of entry: side × (exit after slip − TP) ÷ E × 100 − (taker − maker) + funding to the position.
    PURE."""
    mk, tk = (MAKER if maker is None else maker), (TAKER if taker is None else taker)
    px = exit_px * (1 - side * slip / 100.0)
    return float(side * (px - tp_px) / E * 100.0 - (tk - mk) + (funding_pct or 0.0))


def ht_floor_px(E, floor_net=HT_FLOOR_NET, fee=None):
    """the +5-net floor price (study: E × (100 + 5 + 0.05) ÷ (100 − 0.05); fee = the taker fee %). PURE."""
    fe = TAKER if fee is None else fee
    return E * (100 + floor_net + fe) / (100 - fe)


def ht_read(deltas, whys):
    """HT_RUNNER read on the forward priced prefix (open order) → (state, text, k). Σ Δ of the first HT_N > 0 → ARM_CANDIDATE else KEEP_TP;
    fewer → COLLECTING. PURE."""
    d = list(deltas)
    if len(d) < HT_N:
        return "COLLECTING", f"⏳ {len(d)}/{HT_N} TP hits · Σ Δ so far {sum(d):+.2f} pp (not judged)", None
    s = sum(d[:HT_N])
    ran = sum(1 for w in list(whys)[:HT_N] if w != "guard")
    return (("ARM_CANDIDATE" if s > 0 else "KEEP_TP"),
            f"first {HT_N} TP hits: Σ Δ {s:+.2f} pp · {ran} ran past the floor — " +
            ("runner ahead of TP +6: arm candidate (operator decides)" if s > 0 else "runner not ahead: keep TP +6"), HT_N)


def ht_info(deltas, whys):
    """the never-deciding HT warnings (floor mechanics · far behind). PURE."""
    out = []
    fl = [x for x, w in zip(deltas, whys) if w == "guard"]
    if len(fl) >= HT_FLOOR_MIN_N and np.mean(fl) < HT_FLOOR_WARN:
        out.append(f"⚠ floor exits average {np.mean(fl):+.2f} pp vs TP over {len(fl)} (model ≈ −1.3; < {HT_FLOOR_WARN:+.1f} = floor slippage worse than modelled)")
    if deltas and sum(deltas) < HT_BEHIND:
        out.append(f"⚠ runner {sum(deltas):+.2f} pp behind TP +6 (< {HT_BEHIND:+.0f}) — far behind; the line can be retired (operator decides)")
    return out


def lower_bound(rows, q=EXH_Q, n_boot=BOOT_N, seed=SEED):
    """one-sided q LOWER bound of mean Δ, day-block bootstrap = −(upper bound of −Δ). PURE."""
    return -XO.upper_bound([(d, -x) for d, x in rows], q=q, n_boot=n_boot, seed=seed)


def exh_read(rows):
    """EXH_RUNNER read: rows = [(day, Δ)] forward priced prefix → (state, text, k). The first prefix with ≥ EXH_N hits spanning ≥ EXH_DAYS days:
    90 % lower bound > 0 → ARM_CANDIDATE else KEEP_TP. PURE."""
    for k in range(EXH_N, len(rows) + 1):
        if len({d for d, _ in rows[:k]}) >= EXH_DAYS:
            cut = rows[:k]
            m, lb = float(np.mean([x for _, x in cut])), lower_bound(cut)
            info = f"first {k} TP hits on {len({d for d, _ in cut})} days: mean Δ {m:+.2f} pp · {EXH_Q:.0%} lower bound {lb:+.2f}"
            return (("ARM_CANDIDATE", info + " — runner shown better than TP 7: arm candidate (operator decides)", k) if lb > 0 else
                    ("KEEP_TP", info + " — not shown better: keep TP 7", k))
    n = len(rows)
    return "COLLECTING", (f"⏳ {n}/{EXH_N} TP hits on {len({d for d, _ in rows})}/{EXH_DAYS} days" +
                          (f" · running mean Δ {np.mean([x for _, x in rows]):+.2f} (not judged)" if n else "")), None


def summary(rows):
    """aggregate text for priced rows [dict(d=Δ, why=…, fund=…|None)]. PURE."""
    if not rows:
        return "no priced TP hits yet"
    d = np.array([r["d"] for r in rows], float)
    mix = pd.Series([r["why"] for r in rows]).value_counts().to_dict()
    srt = np.sort(d)[::-1]
    fu = [r["fund"] for r in rows if r.get("fund") is not None]
    nf = sum(1 for r in rows if r.get("fund") is None)
    return (f"N {len(d)} · Σ Δ {d.sum():+.2f} pp (${d.sum() * ONE_SIZE_USD / 100:+,.0f} at one size) · mean {d.mean():+.2f} · median {np.median(d):+.2f} · "
            f"better {100 * (d > 0).mean():.0f} % · exits {mix} · best {srt[0]:+.2f} · mean w/o best {(srt[1:].mean() if len(srt) > 1 else float('nan')):+.2f}"
            + (f" · funding avg {np.mean(fu):+.2f}" if fu else "") + (f" · {nf} without funding (unread)" if nf else ""))


def _bars(pair, t_touch, now_ms, bud):
    """1m bars from touch minute − 5 h to + 24 h + 1 bar (pages of 900, cached when full). None on a budget / error stop."""
    start = int(t_touch) // MIN1 * MIN1 - ATR_LOOK_H * HOUR
    end, out, e0 = int(t_touch) // MIN1 * MIN1 + HOLD_H * HOUR + 2 * MIN1, [], getattr(bud, "errors", 0)
    while start <= min(end, now_ms):
        pg = LS.fapi_klines(pair, "1m", start, PAGE, bud, CACHE, now_ms, fetch=True)
        if pg is None or getattr(bud, "errors", 0) > e0:
            return None
        out += [q[:5] for q in pg]
        if len(pg) < PAGE:
            break
        start = int(pg[-1][0]) + MIN1                                 # (deep review) a gap in page 1 must not overlap page 2
    seen, uniq = set(), []
    for q in out:
        if int(q[0]) not in seen:
            seen.add(int(q[0]))
            uniq.append(q)
    return sorted(uniq, key=lambda q: int(q[0]))


def price_fill(r, side, now_ms, bud, bars_fn=None, funding_fn=None):
    """→ dict(d, why, fund, day) | 'PENDING' | 'NOT_EVALUABLE: …'. bars_fn(pair, t_touch, now) / funding_fn(pair, t0, t1) → short-receives %
    (None = unread) are injectable (tests)."""
    E, tp_px = _f(r.entry_price), _f(getattr(r, "exit_price", None))
    try:
        ts = pd.Timestamp(str(r.closed_at)[:19])
    except Exception:
        ts = pd.NaT
    if pd.isna(ts):
        return "NOT_EVALUABLE: no closed_at"
    t_touch = int(ts.value // 1_000_000)
    if not (E and E > 0 and tp_px and tp_px > 0):
        return "NOT_EVALUABLE: no entry / exit price"
    bars = bars_fn(r.pair, t_touch, now_ms) if bars_fn else _bars(r.pair, t_touch, now_ms, bud)
    if bars is None:
        return "PENDING"
    atr = atr15_pct(bars, t_touch)
    if atr is None:
        return "PENDING" if now_ms < t_touch + STALE_H * HOUR else "NOT_EVALUABLE: no ATR (too few 1m bars before the touch)"
    if side > 0:
        guard = ht_floor_px(E)
    else:
        sl = _f(getattr(r, "entry_exh_sl_pct", None))
        if not sl:
            return "NOT_EVALUABLE: no entry_exh_sl_pct stamp"
        guard = E * (1 + sl / 100.0)
    why, px, t_exit = runner_walk(bars, t_touch, tp_px, side, K_ATR * atr, guard)
    if why == "PENDING":
        return "PENDING" if now_ms < t_touch + STALE_H * HOUR else "NOT_EVALUABLE: no complete price path"
    t_fe = t_exit - MIN1                                            # the exit bar's open (the research's window end)
    fs = funding_fn(r.pair, t_touch, t_fe) if funding_fn else _FB.funding_sum_pct(r.pair, t_touch, t_fe, bud, now_ms)
    if fs is None and now_ms < t_touch + STALE_H * HOUR:
        return "PENDING"
    fund = None if fs is None else float(-side * fs * px / E)       # a short receives fs; a long pays it (mark-scaled ≈ exit px ÷ E)
    return dict(d=delta_pp(side, E, tp_px, px, fund or 0.0), why=why, fund=fund, day=t_touch // DAY, atr=atr, o=r.o)


def _cohort(sleeve, frame):
    a = _W.load(sleeve=sleeve, cols=COLS) if frame is None else frame
    if not len(a):
        return a
    a = a[a.close_reason.astype(str).str.upper().str.split(" ").str[0].isin(TP_REASONS)]
    if sleeve == HT:
        tgt = pd.to_numeric(_col(a, "frenzy_willy_tp_target_pct"), errors="coerce")
        a = a[((tgt - HT_TP_MIN).abs() < 0.05).values]                 # exactly the +6 target (a high-target era stamps 7.69 / 8)
    else:
        a = a[[bool(exit_era_609(t, x, o)) for t, x, o in zip(_col(a, "entry_exh_tp_pct"), _col(a, "entry_exh_atr_pct"), a.opened_at)]]
    return a


def _col(a, c):
    return a[c] if c in a.columns else pd.Series([None] * len(a), index=a.index)


def report(frames=None, since_ms=None, era_ms=None, state_path=STATE, now_ms=None, bars_fn=None, funding_fn=None, newest=None):
    """frames / era_ms = {sleeve: …} (tests); since_ms = the DL 636 forward floor."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🏃👁 TP_RUNNER observe lines (DECISION_LOG {DL}) — HT 4 × ATR + floor +5 · EXH 4 × ATR, stop kept, no floor — "
         "runner Δ vs the booked maker TP (pp of entry, funding incl.)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=0, label="TP_RUNNER")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    try:
        with open(state_path) as fh:
            st = json.load(fh)
    except FileNotFoundError:
        st = {}
    except (OSError, ValueError):
        st = None
    if st is not None and not isinstance(st, dict):
        st = None
    bud, stopped, dirty = LS.Budget(), False, False
    newest = _DT.newest_export() if newest is None else newest
    for sleeve, side, era_dl in ((HT, 1, 608), (EXH, -1, 609)):
        key = LINES[sleeve]
        e0 = (era_ms or {}).get(sleeve) if era_ms is not None else _GK.dl_deploy_ms(era_dl, lag_min=10, label=key + " era")
        a = _cohort(sleeve, frames.get(sleeve, pd.DataFrame(columns=list(COLS) + ["o"])) if frames is not None else None)
        fwd, ins, pend, ne, ne_fwd_o = [], [], 0, {}, []
        for r in (a.itertuples() if len(a) else []):
            is_fwd = r.o >= pd.Timestamp(int(since_ms), unit="ms")
            if not is_fwd and (e0 is None or r.o < pd.Timestamp(int(e0), unit="ms")):
                continue                        # an older exit era (e.g. the pre-DL 608 high-target HT fills) — in neither block
            pr = price_fill(r, side, now_ms, bud, bars_fn, funding_fn)
            stopped |= bud.stop
            if pr == "PENDING":
                pend += 1
                if is_fwd:
                    break                       # open-time prefix: the forward read waits at the first unpriced fill
                continue
            if isinstance(pr, str):
                ne[pr] = ne.get(pr, 0) + 1
                if is_fwd:
                    ne_fwd_o.append(r.o)
                continue
            if is_fwd:
                fwd.append(pr)
            else:
                ins.append(pr)
        L.append(f"### {key} — {'WILLY_HT' if sleeve == HT else 'EXHAUSTION_SHORT'} TP hits")
        L.append(f"  IN-SAMPLE (study period, exit era DL {era_dl} → DL {DL}; never deciding): {summary(ins)}")
        L.append(f"  FORWARD (opened ≥ {pd.Timestamp(int(since_ms), unit='ms'):%Y-%m-%d %H:%M} UTC): {summary(fwd)}"
                 + (f" · ≥ {pend} pending (the forward read waits at the first)" if pend else "") + (f" · not evaluable {ne}" if ne else ""))
        if sleeve == HT:
            v, txt, k = ht_read([x["d"] for x in fwd], [x["why"] for x in fwd])
            for w in ht_info([x["d"] for x in fwd], [x["why"] for x in fwd]):
                L.append("  " + w)
        else:
            v, txt, k = exh_read([(x["day"], x["d"]) for x in fwd])
        hold = []
        if stopped:
            hold.append("a fetch stopped this run")
        if v != "COLLECTING":
            dec = fwd[:k]
            last_o = dec[-1]["o"]
            if newest is None or newest < last_o + pd.Timedelta(hours=EXPORT_HOLD_H):
                hold.append("newest export < the deciding fill's open + 24 h")
            nf = sum(1 for x in dec if x.get("fund") is None)
            if nf:
                hold.append(f"funding unread on {nf} deciding fill(s)")
            nne = sum(1 for o in ne_fwd_o if o <= last_o)
            if nne > 0.10 * (k + nne):
                hold.append(f"{nne} forward fills NOT EVALUABLE (> 10 % of the prefix) — check before reading")
        rec = st.get(key) if isinstance(st, dict) else None
        bad_rec = rec is not None and not (isinstance(rec, dict) and rec.get("verdict"))
        if st is None or bad_rec:
            L.append("  ⚠ state file corrupt" + (f" ({key} entry)" if bad_rec else "") + " — operator restore needed; not frozen this run.")
            rec = None
        elif v != "COLLECTING" and not hold and rec is None:
            rec = dict(verdict=v, detail=txt, run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
            st[key], dirty = rec, True
        if rec:
            L.append(f"  **FROZEN ({rec.get('run_at')}): {rec.get('verdict')}** — {rec.get('detail')}")
        else:
            L.append(f"  Now: {v} — {txt}" + (" · ⏸ freezing deferred: " + " · ".join(hold) if hold else ""))
        L.append("")
    if dirty:
        try:
            tmp = state_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(st, fh, indent=1)
            os.replace(tmp, state_path)
        except OSError as e:
            L.append(f"⚠ state not saved ({e})")
    return L


if __name__ == "__main__":
    print("\n".join(report()))
