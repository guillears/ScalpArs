#!/usr/bin/env python3
"""⏱ Best-case step: every FRENZY fill priced under a SHORTER TIME CAP (DECISION_LOG 497, operator 2026-11-07: "we should study this in
best case, like if it helps to winners on the filters") → reports/best_case/frenzy_caps.csv. best_dcr.py reads it for table ①c.

Per fill (FRENZY_LITE · FRENZY_LITE_RETRY · FRENZY_LONG · FRENZY_WILLY · FRENZY_WILLY_HT; current batch + master stack_keep), for each cap
X in scout_frenzy_duration.XS below the sleeve's live cap: CAP<X> = the fill's net % on TODAY's exit with a cap at X, h_CAP<X> its hold.
Base = the % best_dcr.py books on today's exit (current batch after master_exits.reprice_b25_exits; master stack_pct after
reprice_long_stop4).

TWO METHODS, never mixed inside one fill:
  REAL (current-batch fills that ran today's exit — not re-priced): the real path before X is the trade itself → closed before the cap
       fires (X + 8 s) = base; still open = a market close at the last 1m bar closed by then, taker + 0.10 slip
       (scout_frenzy_duration.cap_at — the live-stopped rule holds by construction).
  WALK (master fills + current-batch fills re-priced on today's exit): paired walk on the fill's 1m path from the first full minute after
       the entry — today's exit with today's cap (BASE walk) and with a cap at X; CAP<X> = base + (walk_X − walk_base). Both walks share
       the path, so a replica artefact (a wick live polling rode through) cancels unless the cap itself changes the fill; only fills the
       walk still holds at X move — winners and losers alike (two-sided). Master fills today's stack left as traded take the REAL method;
       a walked fill whose exit time under today's exit is KNOWN (WILLY_EXIT_CF.csv cf_min) and came before the cap fired keeps Δ 0.
  Today's exit per sleeve (trading_config.json, re-read): LITE / RETRY TP frenzy_lite_tp_pct (resting maker) / stop frenzy_stop_pct ·
       LONG TP frenzy_tp_pct / stop frenzy_long_stop_pct · WILLY / HT TP = the fill's own V2b target (stamped frenzy_willy_tp_target_pct;
       master: WILLY_EXIT_CF.csv target_pct — the engine's target on the spike high, the file the master stack prices WILLY from), stop
       frenzy_willy_stop_pct, late TP frenzy_willy_late_tp_pct for bars starting ≥ frenzy_willy_late_tp_minutes after entry; stops ≤ 0
       load as the engine's STOP0_GUARD (3 / 4). A bar touching both lines = the stop (conservative). Stop / TP book their lines exactly —
       the cap close pays taker + 0.10 slip, a stop no slip (a small bias against a cap that saves a loser; stated in ①c).
  PARITY per fill: the base walk matches the booked base when |walk − base| ≤ 0.15, or both are the stop (walk STOP and the base within
       1 pt of the stop line — fill slip / a gap). A MISS (e.g. a manual close, a tick re-walk the 1m path reads differently) keeps its
       booked base at every cap (Δ 0), is flagged and listed — never priced on a path that does not reproduce it. Where the fill's hold
       under today's exit is known (WILLY_EXIT_CF.csv cf_min), the base walk must also exit within 2 min of it. A sleeve whose walked
       fills match < 80 % (PARITY_MIN) → best_dcr.py prints that block as "not read (walk parity)" (current batch or master).
⏬ LATE TP (DECISION_LOG 498, operator 11-07) — LT<X>_<L> on the sleeves on the long FRENZY cap (LITE · RETRY · LONG): today's exit and cap,
the resting TP lowered to L after X minutes. REAL: late_real() on the real path (the lower line arms on 1m bars that start ≥ entry + X and
closed before the real close; a bar touching it and the stop books min(base, L) — live rode that wick; a real exit after X above L → L).
WALK: the same paired walk with
late_min = X, late_tp = L. A late hole sets late_status (best_dcr.py shows that sleeve's ①d "not read"), never status (①c keeps running).
A fill whose 1m minutes are missing → UNPRICED (status), never guessed; best_dcr.py stops on an unpriced FRENZY fill unless the sleeve
is shown as "not read". --fetch fills missing minutes through path_labels' budgeted public fetcher (≤ 120 requests / 60 s per run).

Usage: venv/bin/python -I scripts/best_case/frenzy_caps.py <orders export> [--fetch]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(D))
SC = os.path.join(ROOT, "reports", "best_case")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, D)
import scout_frenzy_duration as FD  # noqa: E402
import master_exits as MX  # noqa: E402

import sweep_separators as _SW  # noqa: E402
B0 = pd.Timestamp(_SW.BATCH_START)      # the current batch start — ONE source (best_dcr.py asserts its own B0 equals it)
WILLY_CF = os.path.join(ROOT, "reports", "WILLY_EXIT_CF.csv")   # master WILLY on today's V2b exit: per-fill target + hold (build_willy_exit_cf.py)
HOLD_TOL = 2.0                           # a walked fill's base hold must sit within 2 min of its known hold under today's exit
MIN = FD.MIN
PARITY_TOL, PARITY_MIN = 0.15, 0.80
STOP_SLIP = 1.0                          # a booked stop within 1 pt of the stop line (fill slip / gap) matches a STOP walk
FRENZY = FD.SLEEVES
WILLY = ("FRENZY_WILLY", "FRENZY_WILLY_HT")
# ⏬ LATE TP (operator 2026-11-07: "for the sleeves with 12h max duration, what happens if after X time we lower the TP? … only for the
# winners in the best case filters") — the sleeves on the long FRENZY cap (≥ LATE_MIN_CAP: LITE · RETRY · LONG), after X minutes the
# resting TP drops from today's target to L (only L below it). Column LT<X>_<L>. WILLY / HT already carry their own late line.
LATE_XS = (60, 120, 240, 360)
LATE_LS = (0.25, 0.5, 1.0, 1.5, 2.0)
LATE_MIN_CAP = 480


def late_variants(sleeve, cap, cfg_path=FD.CONFIG):
    """[(X, L)] for a sleeve on the long cap: X below the cap, L below today's TP; [] for WILLY / HT or a short cap."""
    if sleeve in WILLY or cap < LATE_MIN_CAP:
        return []
    tp = exit_spec(sleeve, cfg_path)["tp"]
    return [(x, l_) for x in LATE_XS if x < cap for l_ in LATE_LS if l_ < tp]


def late_col(x, l_):
    return f"LT{int(x)}_{l_:g}"


def late_real(bars, te, close_ms, entry, fee_in, sl, x, l_, base):
    """a current-batch fill on today's exit, its REAL path: the late line arms for bars that START ≥ te + X min and closed before the real
    close (the trade was live on them) → the first touch books L (resting maker); a bar touching the stop line too is ambiguous — live rode
    that wick (the bar closed before the real exit), so it books min(base, L), never the replica stop (review: live-stopped rule); never
    touched → base, EXCEPT a fill that really exited after X above L (base > L, a net %): the price crossed L by its exit, so the resting L
    filled → L (deep review: the exit bar itself is not read, and a TP / cap close above L there would otherwise keep base). A missing
    minute in that span → None (unpriced)."""
    if close_ms is None:
        return None
    first = -(-(int(te) + int(x) * MIN) // MIN) * MIN
    last = (int(close_ms) // MIN) * MIN - MIN
    if last < first:
        return float(l_) if float(base) > l_ and int(close_ms) >= int(te) + int(x) * MIN else float(base)   # no full bar: the exit itself
    if bars is None or not len(bars):
        return None
    sub = bars[(bars.open_time >= first) & (bars.open_time <= last)].drop_duplicates("open_time", keep="last")
    hl = dict(zip(sub.open_time.values.astype("int64").tolist(), zip(sub.h.values, sub.l.values)))
    lt = entry * ratio_for(l_, fee_in, FD.MAKER)
    dn = entry * ratio_for(-sl, fee_in, FD.TAKER)
    for ot in range(first, last + 1, MIN):
        if ot not in hl:
            return None
        h, l = hl[ot]
        if h >= lt:
            return min(float(base), float(l_)) if l <= dn else float(l_)
    return float(l_) if float(base) > l_ and int(close_ms) >= int(te) + int(x) * MIN else float(base)   # exited after X above L → L filled


def exit_spec(sleeve, cfg_path=FD.CONFIG):
    """today's exit for a sleeve → dict(tp, sl, late_min, late_tp) (tp None = per-fill WILLY target)."""
    with open(cfg_path) as f:
        th = json.load(f)["thresholds"]
    def g(k, d):
        try:
            v = float(th.get(k))
        except (TypeError, ValueError):
            return d
        return v
    fz_stop = g("frenzy_stop_pct", 3.0)
    fz_stop = fz_stop if fz_stop > 0 else 3.0              # engine STOP0_GUARD (DL 292): a stop ≤ 0 loads as 3.0
    if sleeve in ("FRENZY_LITE", "FRENZY_LITE_RETRY"):
        lt = g("frenzy_lite_tp_pct", 0.0)
        return dict(tp=lt if lt > 0 else g("frenzy_tp_pct", 1.5), sl=fz_stop, late_min=None, late_tp=None)   # ≤ 0 → frenzy_tp_pct (engine)
    if sleeve == "FRENZY_LONG":
        ls = g("frenzy_long_stop_pct", 0.0)
        return dict(tp=g("frenzy_tp_pct", 1.5), sl=ls if ls > 0 else fz_stop, late_min=None, late_tp=None)   # 0 → frenzy_stop_pct (frenzy_stop_for)
    ws = g("frenzy_willy_stop_pct", 4.0)
    return dict(tp=None, sl=ws if ws > 0 else 4.0, late_min=g("frenzy_willy_late_tp_minutes", 30.0),     # STOP0_GUARD: ≤ 0 loads as 4.0
                late_tp=g("frenzy_willy_late_tp_pct", 0.25))


def ratio_for(net, fee_in, exit_fee):
    """the price ratio (exit / entry) at which a LONG books `net` % with entry fee % fee_in and exit fee % exit_fee (on the exit notional)."""
    return (100.0 + fee_in + net) / (100.0 - exit_fee)


def walk(bars, te, entry, fee_in, tp, sl, cap_min, late_min=None, late_tp=None):
    """today's fixed exit on 1m bars (open_time ms, h, l, c), LONG, from the first full minute after te → (net %, hold min, why) or None
    when a minute is missing before the exit. tp = the net TP (resting maker), sl = the net stop distance (taker); after late_min the TP line
    is late_tp when lower; a bar touching both = STOP; no touch → a market close at the cap (FD.price_at + FD.net_pct, taker + slip)."""
    if bars is None or not len(bars):
        return None
    fire = FD.cap_fires_ms(te, cap_min)
    first = (int(te) // MIN + 1) * MIN
    last = (fire // MIN) * MIN - MIN                    # the last bar that closed by the time the cap fires
    if last < first:
        px = FD.price_at(bars, fire)
        return None if px is None else (FD.net_pct(entry, px, "LONG", fee_in), cap_min, "CAP")
    sub = bars[(bars.open_time >= first) & (bars.open_time <= last)].drop_duplicates("open_time", keep="last")
    hl = dict(zip(sub.open_time.values.astype("int64").tolist(), zip(sub.h.values, sub.l.values)))
    up = entry * ratio_for(tp, fee_in, FD.MAKER)
    dn = entry * ratio_for(-sl, fee_in, FD.TAKER)
    lt = None if late_min is None or late_tp is None or late_tp >= tp else entry * ratio_for(late_tp, fee_in, FD.MAKER)
    for ot in range(first, last + 1, MIN):              # minute by minute: a hole BEFORE the exit → unpriced; after it, irrelevant
        if ot not in hl:
            return None
        h, l = hl[ot]
        mid = (int(ot) + MIN / 2 - te) / MIN
        if l <= dn:
            return (-float(sl), mid, "STOP")
        late = lt is not None and int(ot) >= te + late_min * MIN   # a bar that STARTS before the late minute keeps the full target
        line, net = (lt, late_tp) if late else (up, tp)
        if h >= line:
            return (float(net), mid, "LATE_TP" if late else "TP")
    px = FD.price_at(bars, fire)
    return None if px is None else (FD.net_pct(entry, px, "LONG", fee_in), float(cap_min), "CAP")


def match(walk_pct, why, base, sl, walk_hold=None, ref_hold=None):
    """the base walk reproduces the booked base: within PARITY_TOL, or both the stop (booked within STOP_SLIP of the stop line) — and, when
    the fill's hold under today's exit is known (ref_hold), the walk exits within HOLD_TOL of it (a TP the 1m walk finds only later would
    otherwise let a short cap cut a winner that really closed at once)."""
    ok = abs(float(walk_pct) - float(base)) <= PARITY_TOL or (why == "STOP" and abs(float(base) + float(sl)) <= STOP_SLIP)
    if ok and ref_hold is not None and walk_hold is not None and ref_hold == ref_hold:
        ok = abs(float(walk_hold) - float(ref_hold)) <= HOLD_TOL
    return ok


def willy_target(row, wcf=None):
    """the fill's V2b take-profit: the stamped target, else the master's re-walk target (WILLY_EXIT_CF.csv target_pct — the engine's
    target on the spike high, the same file the master stack prices WILLY from) → (target, hold under today's exit or None) / (None, None)."""
    t = pd.to_numeric(row.get("frenzy_willy_tp_target_pct"), errors="coerce")
    c = (wcf or {}).get(MX.key(row.get("opened_at"), row.get("pair")))
    if pd.notna(t) and t > 0:
        return float(t), (c[1] if c else None)
    if c and c[0] == c[0] and c[0] > 0:
        return float(c[0]), c[1]
    if c:   # a re-walk row with no target: the master builder found it target-invariant (it stops / caps before any +1.25) → out of reach
        return float("inf"), c[1]
    return None, None


def load_willy_cf(path=WILLY_CF):
    """{key: (target %, hold min under today's exit)} from the master's WILLY re-walk."""
    if not os.path.exists(path):
        return {}
    w = pd.read_csv(path)
    return {MX.key(a, p): (float(t) if t == t else float("nan"), float(m) if m == m else float("nan"))
            for a, p, t, m in zip(w.opened_at, w.pair, pd.to_numeric(w.target_pct, errors="coerce"), pd.to_numeric(w.cf_min, errors="coerce"))}


def bars_for(pair, te, span_min, fetch, bud, attempts, now_ms, notes):
    """the pair's cached 1m bars; with fetch, the missing minutes of [te − 1 min, te + span + 2 min) through path_labels' budgeted fetcher."""
    import scout_willy_timecap as WT
    k = WT.load_1m(pair)
    if not fetch or bud.get("stopped"):
        return k
    a, b = (int(te) // MIN) * MIN - MIN, ((int(te) + int(span_min * MIN) + FD.LAT_MS) // MIN) * MIN + 2 * MIN
    have = set(k.open_time.values.astype("int64").tolist()) if len(k) else set()
    if all(m in have for m in range(a, min(b, (now_ms // MIN) * MIN - MIN), MIN)):
        return k
    try:
        import path_labels as PL
        if PL._fetch_window(WT, pair, a, b, now_ms, bud, attempts):
            WT._K1.pop(pair, None)
            k = WT.load_1m(pair)
    except Exception as e:
        notes.append(f"{pair}: 1m fetch failed ({type(e).__name__}: {str(e)[:60]})")
    return k


def price_rows(F, src, caps, fetch, bud, notes, now_ms, wcf=None):
    """F: fills of one source with sl, opened_at, closed_at, entry_price, pct (base), fee_in, method → list of output dicts."""
    out = []
    attempts = {}
    for r in F.to_dict("records"):
        sl, base = r["sl"], float(r["pct"])
        cap = caps[sl]
        xs = [x for x in FD.XS if x < cap]
        te = FD.to_ms(r["opened_at"])
        rec = dict(src=src, sl=sl, k=str(r["opened_at"]).replace(" ", "T")[:19], pair=r["pair"], base=base, method=r["method"],
                   status="PRICED", walk_base=np.nan, why_base="")
        if str(r.get("direction", "LONG")).upper() != "LONG" or te is None:
            out.append({**rec, "status": "UNPRICED (not a long / no time)"})
            continue
        hold = FD.hold_minutes(r["opened_at"], r["closed_at"])
        lv = late_variants(sl, cap)
        span = float(cap) if r["method"] == "WALK" else min(float(hold or 0), float(max(xs + [x for x, _ in lv]) if xs or lv else 0))
        if r["method"] == "REAL" and lv:
            span = float(hold or 0)                     # the late line needs the real path up to the real close
        k = bars_for(str(r["pair"]), te, span, fetch, bud, attempts, now_ms, notes)
        ex = exit_spec(sl)
        if r["method"] == "REAL":
            for x in xs:
                px = FD.price_at(k, FD.cap_fires_ms(te, x)) if FD.open_at_cap(hold, x) else None
                v = FD.cap_at(base, hold, x, px, float(r["entry_price"]), "LONG", float(r["fee_in"]))
                rec[f"CAP{x}"] = np.nan if v is None else v
                rec[f"h_CAP{x}"] = min(float(hold), x) if hold is not None else np.nan
                if v is None:
                    rec["status"] = "UNPRICED (1m bars missing)"
            for x, l_ in lv:
                v = late_real(k, te, FD.to_ms(r["closed_at"]), float(r["entry_price"]), float(r["fee_in"]), ex["sl"], x, l_, base)
                rec[late_col(x, l_)] = np.nan if v is None else v
                if v is None:
                    rec["late_status"] = "UNPRICED (1m bars missing)"   # ①d only: never aborts ①c (review)
            out.append(rec)
            continue
        ref_hold = None
        if ex["tp"] is not None:
            tp = ex["tp"]
        else:
            tp, ref_hold = willy_target(r, wcf)
        if tp is None:
            out.append({**rec, "status": "UNPRICED (no WILLY target)"})
            continue
        args = (te, float(r["entry_price"]), float(r["fee_in"]), tp, ex["sl"])
        wb = walk(k, *args, cap, ex["late_min"], ex["late_tp"])
        if wb is None:
            out.append({**rec, "status": "UNPRICED (1m bars missing)"})
            continue
        rec["walk_base"], rec["why_base"], rec["walk_base_hold"], rec["ref_hold"] = wb[0], wb[2], wb[1], ref_hold
        if not match(wb[0], wb[2], base, ex["sl"], wb[1], ref_hold):
            rec["status"] = "PARITY MISS (kept as booked)"
            for x in xs:
                rec[f"CAP{x}"] = base
                rec[f"h_CAP{x}"] = np.nan
            for x, l_ in lv:
                rec[late_col(x, l_)] = base
            out.append(rec)
            continue
        for x in xs:
            if ref_hold is not None and ref_hold == ref_hold and not FD.open_at_cap(ref_hold, x):
                rec[f"CAP{x}"], rec[f"h_CAP{x}"] = base, float(ref_hold)   # (deep review) its known exit came before the cap fired → Δ 0
                continue
            w = walk(k, *args, x, ex["late_min"], ex["late_tp"])
            if w is None:
                rec["status"] = "UNPRICED (1m bars missing)"
                rec[f"CAP{x}"] = rec[f"h_CAP{x}"] = np.nan
                continue
            rec[f"CAP{x}"] = base + (w[0] - wb[0])
            rec[f"h_CAP{x}"] = w[1]
        for x, l_ in lv:                                # ⏬ the same paired walk with the TP lowered to L after X (today's cap kept)
            w = walk(k, *args, cap, x, l_)
            if w is None:
                rec["late_status"] = "UNPRICED (1m bars missing)"
                rec[late_col(x, l_)] = np.nan
                continue
            rec[late_col(x, l_)] = base + (w[0] - wb[0])
        out.append(rec)
    return out


def load(exp):
    """→ (current-batch FRENZY fills, master FRENZY fills), each with sl, pct (best_dcr's base), fee_in, method."""
    A = pd.read_csv(exp, low_memory=False)
    A = A[(A.status == "CLOSED") & (pd.to_datetime(A.opened_at, format="mixed") >= B0)].copy()
    A["o"] = pd.to_datetime(A.opened_at, format="mixed", utc=True).dt.tz_convert(None)
    A["pct"] = pd.to_numeric(A.pnl_percentage, errors="coerce")
    A["u"] = pd.to_numeric(A.pnl, errors="coerce")
    A["sl"] = A.entry_strategy.astype(str)
    p0 = A.pct.copy()
    A, _ = MX.reprice_b25_exits(A, os.path.join(SC, "fb_combo.csv"))
    A["method"] = np.where((A.pct - p0).abs() > 1e-9, "WALK", "REAL")
    A = A[A.sl.isin(FRENZY)]
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    M = M[(M.stack_keep == True) & (M.is_probe != True)].copy()
    M["o"] = pd.to_datetime(M.opened_at, format="mixed", utc=True).dt.tz_convert(None)
    M["pct"] = pd.to_numeric(M.stack_pct, errors="coerce")
    M["sl"] = M.entry_strategy.astype(str).where(M.entry_strategy.astype(str) != "MOMENTUM", "MOMENTUM_" + M.direction.astype(str))
    M, _ = MX.reprice_long_stop4(M)
    M = M[M.sl.isin(FRENZY)].copy()
    # a master fill today's stack left as traded (same %, same exit) has its REAL path → exact, like the current batch; re-priced ones walk
    M["method"] = np.where((M.pct - pd.to_numeric(M.pnl_percentage, errors="coerce")).abs() <= 1e-9, "REAL", "WALK")
    for d in (A, M):
        n = len(d)
        col = (lambda c: d[c] if c in d.columns else [None] * n)
        d["fee_in"] = [FD.entry_fee(t, fu, nv) for t, fu, nv in zip(col("entry_order_type"), col("entry_fee"), col("notional_value"))]
    return A, M


def parity(rows):
    """per (src, sleeve) share of walked fills whose base walk is within PARITY_TOL of the booked base → dict."""
    d = pd.DataFrame(rows)
    d = d[(d.method == "WALK") & d.walk_base.notna()]
    return {k: float((g.status != "PARITY MISS (kept as booked)").mean()) for k, g in d.groupby(["src", "sl"])}


def main(argv):
    exp = argv[1]
    fetch = "--fetch" in argv
    caps = FD.caps()
    A, M = load(exp)
    notes, bud = [], {"req": 0, "used": 0, "t0": time.monotonic()}
    now_ms = int(time.time() * 1000)
    wcf = load_willy_cf()
    rows = price_rows(A, "B25", caps, fetch, bud, notes, now_ms, wcf) + price_rows(M, "MASTER", caps, fetch, bud, notes, now_ms, wcf)
    P = parity(rows)
    D_ = pd.DataFrame(rows)
    os.makedirs(SC, exist_ok=True)
    D_.to_csv(os.path.join(SC, "frenzy_caps.csv"), index=False)
    print(D_.groupby(["src", "sl", "method", "status"]).size().to_string())
    print("walk parity (share of walked fills whose base walk reproduces the booked % on today's exit): "
          + " · ".join(f"{s}/{sl} {v:.0%}" for (s, sl), v in sorted(P.items())))
    for r in D_[D_.status.str.startswith("PARITY MISS")].itertuples():
        print(f"- parity miss (kept as booked): {r.src} {r.sl} {r.k} {r.pair} booked {r.base:+.2f} vs walk {r.walk_base:+.2f} {r.why_base}"
              + (f" · hold {r.walk_base_hold:.0f} vs {r.ref_hold:.0f} min" if r.ref_hold == r.ref_hold and r.ref_hold is not None else ""))
    if bud.get("req"):
        print(f"- 1m fetch: {bud['req']} request(s), used weight {bud['used']}" + (f" · stopped: {bud['stopped']}" if bud.get("stopped") else ""))
    for n in notes:
        print(f"- {n}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv)
