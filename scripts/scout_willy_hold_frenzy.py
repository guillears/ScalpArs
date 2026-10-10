#!/usr/bin/env python3
"""🔓 Scout — HOLD_FRENZY_EXEMPT (2026-10-10, DECISION_LOG 266; OBSERVE only — never changes config, no bot API).

HYPOTHESIS  FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE should be EXEMPT from the FRENZY_WILLY global hold (DECISION_LOG 251): their own −3 stop
            bounds what they add next to an open WILLY, so if the trades the hold refuses them make money, the hold is costing money there.
            Every other sleeve (and WILLY itself) keeps the hold — not judged here.
COHORT      the WILLY_HOLD tracker's store (reports/SCOUT_WILLY_HOLD.csv — the bot's own e=WILLY_HOLD refusals, priced AS IF OPENED by
            scripts/scout_willy_hold.py at the sleeve's live exit, +3 / −3 / 12 h on 1m klines, FEE 0.09 + SLIP 0.10, with the exit minute),
            FRENZY_LONG / WIDE / LITE rows only, one per (pair, sleeve, signal bar). This line prices nothing. The hold is the FIRST check in
            _frenzy_open, so a refused setup was never judged by the gates after it. Re-checked in signal order (the bot's own fills, stamps
            and decision journal; EARLIER KEPT refusals count as if open from their signal to their priced exit — 12 h when not known — so the
            counterfactual also occupies its own pair / slot / day count):
              · UNREAD — refused because the open-WILLY check itself was unreadable (wh_reason UNREAD), not because a WILLY was open → out
              · TAKEN LATER — the same pair + sleeve opened within frenzy_catchup_max_bars (6) + 1 bars after the signal (a refused fresh ON bar /
                LITE stretch is NOT consumed: the catch-up / stretch retry took it once the WILLY closed) → not a lost trade, out
              · PAIR HELD — another open position on the pair at the signal (any strategy: open_position's PAIR_HELD) → out
              · SLOTS — the sleeve's own open positions ≥ its max slots at the signal → out
              · COOLDOWN — a position on the pair closed ≤ cooldown_after_loss_minutes (5) before the signal (open_position) → out
              · PAIR-DAY CAP — FRENZY_LONG / WIDE / LITE (WILLY alone is exempt in the engine): same pair + sleeve fills that UTC day ≥
                frenzy_max_entries_per_pair_day → out
              · WIDE HOLD-GREEN (frenzy_wide_hold_green_streak armed) — a WIDE row whose FRENZY leg was refused FRENZY_ATR_HIGH on that bar (the
                decision journal) is always refused by WIDE → out; a FRENZY_GREEN_BAR leg passes only on a reclaim streak > the setting, which
                is NOT rebuilt (flagged on the row)
              · BEARISH DAY (frenzy_bearish_day_block on) — services.frenzy.frenzy_bearish_day(1d, gap): 1d = entry_btc_1d_ret_pct of any fill
                that UTC day, else the last closed BTC daily return from the B1H line's cache; gap = BTC 5m EMA13 − EMA50 rebuilt as the engine
                reads it (100 bars, the forming one ≈ the last close; vs 121 fill stamps: r 0.999, sign 98 %), else a fill stamp ≤ 5 min away;
                undecidable → kept and flagged (the engine fails OPEN there too)
              · UNPRICEABLE — kept but still unpriced 2 days after its 12 h window (delisted / no klines) → out, listed
            BLIND SPOTS (not re-checkable for a refused setup): entry lateness, price dislocation, WIDE's reclaim streak, open_position's book /
            margin caps and PAIR_NO_TRADE, and a second signal of the same pair + sleeve during ONE WILLY (the engine keeps one row per WILLY ×
            pair × sleeve with a repeat count). Gates are judged with TODAY's config for every row (fine while the cohort starts after the
            Oct-8 ships); any gate armed later must be added here. An OPEN fill counts as open only up to its export's mtime.
            Network: only the BTC 5m / 1d top-up through scout_b1h_negflank.load_btc (cache first, shared cooldown, ≤ 900 weight, 25 s).
VERDICT     (FROZEN, pre-registered — computed ONCE on the first prefix by signal time of the kept, priced rows reaching N ≥ 15 on ≥ 8 UTC days;
            deferred while a FRENZY-family row signalled ≤ the prefix end is unpriced or not yet covered by the newest orders AND decisions
            exports (catch-up window + 10 min); persisted in reports/SCOUT_HOLD_FRENZY_EXEMPT.json;
            never re-fit; one re-read at N ≥ 30). DAY units (market-wide windows, CLAUDE.md WINDOW-UNITS rule).
            EXEMPT CANDIDATE (operator decides) iff mean % > 0 ∧ day-clustered bootstrap P(mean > 0) ≥ 0.90 (4,000, seed 7) ∧ no single day
            AND no single pair > 50 % of the positive Σ % (registered before any priced row); mean ≤ 0 → KEEP THE HOLD; else KEEP OBSERVING.
REVERT      pre-committed if the exemption is ever armed: the first 15 FRENZY-family fills opened while a WILLY was open average < 0 % → restore
            the hold for them.
Usage: venv/bin/python scripts/scout_willy_hold_frenzy.py [--selftest]
"""
import glob
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # services.frenzy (the engine's pure rule)
import scout_b1h_negflank as NF                     # noqa: E402  (bootstrap, freeze hold, state I/O)
import scout_willy_hold as WH                       # noqa: E402  (the tracker's store + config reader)

FRENZY3 = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
N_MIN, DAYS_MIN, REREAD_N = 15, 8, 30
P_MIN, SHARE_MAX = 0.90, 0.50
BOOT_N, BOOT_SEED = 4000, 7
BAR_MS, GAP_NEAR_MS, COVER_SLACK_MS = 300_000, 5 * 60_000, 10 * 60_000
HOLD_MS, UNPRICEABLE_MS, M5_STALE_MS = 720 * 60_000, 2 * 86_400_000, 15 * 60_000
DEC_GLOB = os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(_ROOT, "reports", "SCOUT_HOLD_FRENZY_EXEMPT.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
OCOLS = ("opened_at", "closed_at", "pair", "entry_strategy", "status", "entry_btc_1d_ret_pct", "entry_btc_trend_gap_pct")
REVERT = ("Pre-committed revert if the exemption is ever armed: the first 15 FRENZY-family fills opened while a WILLY was open average "
          "< 0 % → restore the hold for them.")


def _ms(s):
    try:
        t = pd.Timestamp(str(s)[:19])
        return None if pd.isna(t) else int(t.value // 1_000_000)
    except Exception:
        return None


def _th():
    th = WH._live_th()
    g = lambda k, d: (lambda v: d if v in (None, "") else v)(getattr(th, k, None))   # noqa: E731
    return dict(catchup=int(float(g("frenzy_catchup_max_bars", 6) or 0)), day_cap=int(float(g("frenzy_max_entries_per_pair_day", 3) or 0)),
                bearish=bool(g("frenzy_bearish_day_block", False)), cooldown=_cooldown_min(),
                hold_green=float(g("frenzy_wide_hold_green_streak", 0) or 0),
                slots={"FRENZY_LONG": int(float(g("frenzy_max_slots", 2) or 2)), "FRENZY_WIDE": int(float(g("frenzy_wide_max_slots", 2) or 2)),
                       "FRENZY_LITE": int(float(g("frenzy_lite_max_slots", 2) or 2))})


def _cooldown_min():
    try:
        return float((json.load(open(os.path.join(_ROOT, "trading_config.json"))).get("investment") or {}).get("cooldown_after_loss_minutes", 0) or 0)
    except Exception:
        return 0.0


def load_decisions(floor=0):
    """the decision journal's FRENZY-leg refusal codes (BLOCK FRENZY_ATR_HIGH / FRENZY_GREEN_BAR) from exports written ≥ floor; attrs newest_ms."""
    fr, newest = [], 0
    for f in glob.glob(DEC_GLOB):
        m = os.path.getmtime(f)
        newest = max(newest, int(m * 1000))
        if m * 1000 < floor:
            continue
        try:
            d = pd.read_csv(f, usecols=lambda c: c in ("t", "e", "pair", "gate"), dtype=str, low_memory=False)
        except Exception:
            continue
        if {"t", "e", "pair", "gate"} <= set(d.columns):
            fr.append(d[(d.e == "BLOCK") & d.gate.isin(("FRENZY_ATR_HIGH", "FRENZY_GREEN_BAR"))])
    d = pd.concat(fr, ignore_index=True).drop_duplicates() if fr else pd.DataFrame(columns=["t", "e", "pair", "gate"])
    d["ms"] = d.t.map(_ms) if len(d) else pd.Series(dtype=float)
    d.attrs["newest_ms"] = newest
    return d


def btc_gap_at(m5, t_ms):
    """BTC 5m (EMA13 − EMA50) / EMA50 % as the engine reads it at t: 100 bars ending with the FORMING one (≈ its last close) — None when the
    cache does not reach t (last closed bar > 15 min before t) or holds < 99 bars."""
    if m5 is None or not len(m5):
        return None
    g = m5[m5["T"] <= t_ms]
    if len(g) < 99 or t_ms - int(g["T"].iloc[-1]) > M5_STALE_MS:
        return None
    from ta.trend import EMAIndicator
    c = g.c.astype(float).tail(99).tolist()
    sr = pd.Series(c + [c[-1]])
    e13, e50 = EMAIndicator(sr, 13).ema_indicator().iloc[-1], EMAIndicator(sr, 50).ema_indicator().iloc[-1]
    return float((e13 - e50) / e50 * 100) if e50 else None


def btc_1d_at(d1, t_ms):
    """the last CLOSED BTC daily candle's return vs the one before (indicators.last_closed_bar_ret_pct) at t, from the B1H daily series."""
    if d1 is None or not len(d1):
        return None
    day0 = (int(t_ms) // 86_400_000) * 86_400_000
    a, b = d1.get(day0 - 86_400_000), d1.get(day0 - 2 * 86_400_000)
    return float((a / b - 1) * 100) if a is not None and b is not None and b > 0 else None


def load_orders():
    fr = []
    for f in glob.glob(EXPORT_GLOB):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in OCOLS)
        except Exception:
            continue
        if {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame(columns=list(OCOLS) + ["o_ms", "c_ms", "_c"])
    o = pd.concat(fr, ignore_index=True).reindex(columns=list(OCOLS) + ["_m"]).sort_values("_m", kind="stable")
    o["_k"] = o.opened_at.astype(str).str[:19]
    o["_c"] = o.status.astype(str).str.upper().eq("CLOSED")
    o = o.sort_values("_c", kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    o["o_ms"] = o.opened_at.map(_ms)
    o["c_ms"] = o.closed_at.map(_ms)
    still = ~o._c & o.c_ms.isna()                                   # OPEN in its export → open until that export's time, unknown after
    o.loc[still, "c_ms"] = (o.loc[still, "_m"].astype(float) * 1000).astype("int64")
    o.attrs["newest_ms"] = int(float(o._m.max()) * 1000) if len(o) else 0
    return o[o.o_ms.notna()].reset_index(drop=True).pipe(lambda d: (d.attrs.update(o.attrs), d)[1])


# ─────────────────────────── the re-checks (pure) ───────────────────────────
def recheck(r, o, th, ctx=None):
    """→ (status, note). status: KEEP · UNREAD · TAKEN_LATER · PAIR_HELD · SLOTS · COOLDOWN · PAIR_DAY_CAP · WIDE_HOLD_GREEN · BEARISH_DAY.
    ctx: synth = [(pair, sleeve, open_ms, close_ms)] earlier KEPT refusals as if open · dec = decision-journal FRENZY-leg codes · m5 / d1 = BTC."""
    ctx = ctx or {}
    if str(getattr(r, "reason", "") or "").upper() == "UNREAD":
        return "UNREAD", "the open-WILLY check was unreadable"
    sig = _ms(r.signal_at) or _ms(r.t)
    if sig is None:
        return "KEEP", "signal time unreadable"
    pair, sl = str(r.pair), str(r.sleeve)
    es = o.entry_strategy.astype(str) if len(o) else pd.Series(dtype=str)
    syn = pd.DataFrame(ctx.get("synth") or [], columns=["pair", "sleeve", "o_ms", "c_ms"])
    if len(o):
        later = o[(o.pair == pair) & (es == sl) & (o.o_ms > sig) & (o.o_ms <= sig + (max(th["catchup"], 0) + 1) * BAR_MS)]
        if len(later):
            return "TAKEN_LATER", f"opened {pd.Timestamp(int(later.o_ms.min()), unit='ms'):%H:%M}"
    open_real = ((o.o_ms <= sig) & ((o.c_ms.isna() & ~o._c) | (o.c_ms > sig))) if len(o) else pd.Series(dtype=bool)
    open_syn = (syn.o_ms <= sig) & (syn.c_ms > sig)
    if (len(o) and (open_real & (o.pair == pair)).any()) or (open_syn & (syn.pair == pair)).any():
        return "PAIR_HELD", ""
    n_open = (int((open_real & (es == sl)).sum()) if len(o) else 0) + int((open_syn & (syn.sleeve == sl)).sum())
    if n_open >= th["slots"].get(sl, 2):
        return "SLOTS", ""
    cd = float(th.get("cooldown", 0) or 0) * 60_000
    if cd > 0:
        rc = len(o) and (o._c & (o.pair == pair) & (o.c_ms > sig - cd) & (o.c_ms <= sig)).any()
        if rc or ((syn.pair == pair) & (syn.c_ms > sig - cd) & (syn.c_ms <= sig)).any():
            return "COOLDOWN", ""
    day0 = (sig // 86_400_000) * 86_400_000
    if th["day_cap"] > 0:
        nd = (int(((o.pair == pair) & (es == sl) & (o.o_ms >= day0) & (o.o_ms < sig)).sum()) if len(o) else 0) + \
            int(((syn.pair == pair) & (syn.sleeve == sl) & (syn.o_ms >= day0) & (syn.o_ms < sig)).sum())
        if nd >= th["day_cap"]:
            return "PAIR_DAY_CAP", ""
    notes = []
    if sl == "FRENZY_WIDE" and th.get("hold_green", 0):
        dec = ctx.get("dec")
        dp = dec[dec.pair == pair] if dec is not None and len(dec) else None
        codes = set(dp[(dp.ms - sig).abs() < 60_000].gate) if dp is not None else set()      # the journal stamps the bar itself (t = signal_at)
        if "FRENZY_ATR_HIGH" in codes:
            return "WIDE_HOLD_GREEN", "FRENZY leg ATR_HIGH — WIDE refuses it while hold-green is armed"
        near = set(dp[(dp.ms - sig).abs() <= BAR_MS].gate) if dp is not None and not codes else set()   # neighbours: a flag, never a ruling
        notes.append("WIDE reclaim streak unverified" + ("" if "FRENZY_GREEN_BAR" in codes else
                                                          f" (no FRENZY-leg code on this bar{'; a neighbouring bar has ' + '/'.join(sorted(near)) if near else ''})"))
    if th["bearish"]:
        from services.frenzy import frenzy_bearish_day
        d = o[(o.o_ms >= day0) & (o.o_ms < day0 + 86_400_000)] if len(o) else o
        r1 = pd.to_numeric(d.entry_btc_1d_ret_pct, errors="coerce").dropna() if len(d) else pd.Series(dtype=float)
        v1 = float(r1.median()) if len(r1) else btc_1d_at(ctx.get("d1"), sig)
        gv = btc_gap_at(ctx.get("m5"), sig)
        if gv is None and len(o):
            near = o[(o.o_ms - sig).abs() <= GAP_NEAR_MS]
            g = pd.to_numeric(near.entry_btc_trend_gap_pct, errors="coerce").dropna()
            gv = float(g.loc[(near.o_ms[g.index] - sig).abs().idxmin()]) if len(g) else None
        b = frenzy_bearish_day(v1, gv)
        bt = f"BTC 1d {'?' if v1 is None else f'{v1:+.2f}'} % · 5m gap {'?' if gv is None else f'{gv:+.3f}'} %"
        if b is True:
            return "BEARISH_DAY", bt
        notes.append(bt)
        if b is None:
            notes.append("bearish-day undecidable (BTC 1d / 5m gap unreadable) — kept, the engine fails open")
    return "KEEP", " · ".join(notes)


# ─────────────────────────── verdict / freeze ───────────────────────────
def judge(c):
    p = c.pct.values.astype(float)
    s = float(p.sum())
    pr = NF.p_mean_neg(-p, c.day.values, BOOT_N, BOOT_SEED) if len(c) else None
    share = float(c.groupby("day").pct.sum().max() / s) if s > 0 else float("nan")
    pshare = float(c.groupby("pair").pct.sum().max() / s) if s > 0 else float("nan")
    m = float(p.mean()) if len(p) else float("nan")
    return dict(mean=m, sum=s, p=pr, share=share, pshare=pshare,
                q=bool(m > 0 and pr is not None and pr >= P_MIN and share <= SHARE_MAX and pshare <= SHARE_MAX))


def verdict(c):
    n, nd = len(c), c.day.nunique() if len(c) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"N {n}/{N_MIN} · {nd}/{DAYS_MIN} days", {}
    j = judge(c)
    det = (f"N {n} · {nd} d · WR {(c.pct > 0).mean() * 100:.0f} % · mean {j['mean']:+.3f} % · Σ {j['sum']:+.2f} % · P(mean>0) "
           f"{(j['p'] if j['p'] is not None else float('nan')):.2f} · top day {(j['share'] * 100 if j['share'] == j['share'] else float('nan')):.0f} % / top pair "
           f"{(j['pshare'] * 100 if j['pshare'] == j['pshare'] else float('nan')):.0f} % of Σ")
    if j["q"]:
        return "EXEMPT CANDIDATE (operator decides)", det, j
    if j["mean"] <= 0:
        return "KEEP THE HOLD", det, j
    return "KEEP OBSERVING", det, j


def prefix(c, n_min):
    z = c.sort_values(["sig", "pair"], kind="stable")
    seen = set()
    for k, d in enumerate(z.day.values, 1):
        seen.add(d)
        if k >= n_min and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def freeze(st, c, now_iso, hold=None, why=None):
    key = "reread" if "first" in st else "first"
    if key in st:
        return st, False
    pre = prefix(c, REREAD_N if key == "reread" else N_MIN)
    if pre is None:
        return st, False
    last = pd.Timestamp(int(pre.sig.max()), unit="ms")
    if NF.freeze_hold(last, hold, why):
        return st, False
    state, det, _ = verdict(pre)
    st[key] = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso, n=len(pre), days=int(pre.day.nunique()),
                   keys=[f"{a}|{b}|{s}" for a, b, s in zip(pre.signal_at.astype(str), pre.pair.astype(str), pre.sleeve.astype(str))])
    return st, True


# ─────────────────────────── run ───────────────────────────
def cohort(rows, o, th, ctx=None, now_ms=None):
    r = rows[rows.sleeve.astype(str).isin(FRENZY3)].copy()
    if not len(r):
        return r.assign(sig=pd.Series(dtype="int64"), day=pd.Series(dtype=object), status=pd.Series(dtype=object), note=pd.Series(dtype=object))
    r["sig"] = [(_ms(a) or _ms(b) or 0) for a, b in zip(r.signal_at, r.t)]
    r = r[r.sig > 0]                                              # a row with no readable time can never be judged (never a 1970 day)
    r = r.sort_values("sig", kind="stable").drop_duplicates(["pair", "sleeve", "sig"], keep="first").reset_index(drop=True)
    r["day"] = pd.to_datetime(r.sig, unit="ms").dt.strftime("%Y-%m-%d")
    r["pct"] = pd.to_numeric(r.pct, errors="coerce")
    ctx = dict(ctx or {}, synth=[])
    now_ms = now_ms or int(time.time() * 1000)
    xm = pd.to_numeric(r.exit_ms, errors="coerce") if "exit_ms" in r else pd.Series(np.nan, index=r.index)
    st, nt = [], []
    for k, x in enumerate(r.itertuples()):                         # signal order: earlier KEPT refusals occupy pair / slot / day count
        a, b = recheck(x, o, th, ctx)
        if a == "KEEP" and pd.isna(x.pct) and now_ms > int(x.sig) + HOLD_MS + UNPRICEABLE_MS:
            a, b = "UNPRICEABLE", "no price 2 days after its 12 h window"
        if a == "KEEP":
            ctx["synth"].append((str(x.pair), str(x.sleeve), int(x.sig),
                                 int(xm.iloc[k]) if pd.notna(xm.iloc[k]) else int(x.sig) + HOLD_MS))
        st.append(a)
        nt.append(b)
    r["status"], r["note"] = st, nt
    return r


def run(now_ms=None, rows=None, orders=None, state_path=None, th=None, ctx=None):
    now_ms = now_ms or int(time.time() * 1000)
    state_path = state_path or STATE
    th = th or _th()
    if rows is None:
        rows = pd.read_csv(WH.STORE, dtype={"willy_id": str}) if os.path.exists(WH.STORE) else pd.DataFrame(columns=WH.COLS)
        fl = WH.floor_ms()
        rows = rows[rows.t.map(lambda x: (_ms(x) or 0) >= fl)] if len(rows) else rows
    o = load_orders() if orders is None else orders
    if "_c" not in o:
        o = o.assign(_c=o.status.astype(str).str.upper().eq("CLOSED"))
    ctx = dict(ctx or {})
    if "dec" not in ctx:
        ctx["dec"] = load_decisions(WH.floor_ms() - 86_400_000)
    fz = rows[rows.sleeve.astype(str).isin(FRENZY3)] if len(rows) else rows
    notes = []
    if "m5" not in ctx:
        sigs = [x for x in (_ms(a) or _ms(b) for a, b in zip(fz.signal_at, fz.t)) if x] if len(fz) else []
        if sigs and th["bearish"]:
            ctx["m5"], ctx["d1"], notes = NF.load_btc(max(sigs), now_ms, fetch=True)
    r = cohort(rows, o, th, ctx, now_ms)
    keep = r[r.status == "KEEP"] if len(r) else r
    c = keep[keep.pct.notna()] if len(keep) else keep
    pend = keep[keep.pct.isna()] if len(keep) else keep
    L = ["## 🔓 HOLD_FRENZY_EXEMPT — should FRENZY / WIDE / LITE be exempt from the WILLY global hold? (DECISION_LOG 266, OBSERVE only)", "",
         "The bot's own WILLY_HOLD refusals of FRENZY_LONG / WIDE / LITE (the WILLY_HOLD tracker's store, priced AS IF OPENED at +3 / −3 / 12 h on "
         "1m klines, fees + slip), re-checked in signal order against the gates the hold pre-empted (unreadable check · catch-up took it later · pair held · slots · "
         "cooldown · pair-day cap · WIDE hold-green · bearish day on rebuilt BTC values · unpriceable); earlier kept refusals occupy pair / slot / "
         "day count. Blind spots: lateness, dislocation, WIDE's reclaim streak, book / margin caps, a 2nd signal during one WILLY.", ""]
    if not len(r):
        L.append("No FRENZY-family setup refused by the hold yet.")
    else:
        L += ["| Kept / out | N | Days | WR | avg % | Σ % |", "|---|---|---|---|---|---|"]
        for lab, g in [("**KEPT (lost trades)**", c), ("kept — pending price", pend)] + [(f"out — {s}", r[r.status == s]) for s in
                                                         ("UNREAD", "TAKEN_LATER", "PAIR_HELD", "SLOTS", "COOLDOWN", "PAIR_DAY_CAP",
                                                          "WIDE_HOLD_GREEN", "BEARISH_DAY", "UNPRICEABLE")]:
            if not lab.startswith("**") and not len(g):
                continue
            p = pd.to_numeric(g.pct, errors="coerce").dropna()
            L.append(f"| {lab} | {len(g)} | {g.day.nunique() if len(g) else 0} | "
                     + (f"{(p > 0).mean() * 100:.0f} % | {p.mean():+.3f} % | {p.sum():+.2f} % |" if len(p) else "– | – | – |"))
        L.append("")
        L.append("Per setup: " + " · ".join(
            f"{pd.Timestamp(int(x.sig), unit='ms'):%m-%d %H:%M} {x.pair} {x.sleeve} (WILLY {x.willy_pair if isinstance(x.willy_pair, str) and x.willy_pair else 'check unreadable'} open) → {x.status}"
            + (f" {x.note}" if x.note else "") + (f" · {x.pct:+.2f} % ({x.exit_how})" if pd.notna(x.pct) else " · unpriced (window not over)")
            for x in r.itertuples()))
        bys = c.groupby("sleeve").pct.agg(["count", "mean"]) if len(c) else None
        if bys is not None and len(bys):
            L.append("By sleeve (kept, priced): " + " · ".join(f"{k} {int(v['count'])} · {v['mean']:+.3f} %" for k, v in bys.iterrows()))
        L.append("")
    hold = [(pd.Timestamp(int(x.sig), unit="ms"), f"unpriced {x.pair} {x.sleeve}") for x in pend.itertuples()] if len(pend) else []
    newest = min(int(o.attrs.get("newest_ms", 0) or 0), int(ctx["dec"].attrs.get("newest_ms", 0) or 0) if "dec" in ctx else 0)
    need = (max(th["catchup"], 0) + 1) * BAR_MS + COVER_SLACK_MS
    hold += [(pd.Timestamp(int(x.sig), unit="ms"), f"orders / decisions exports not past the catch-up window {x.pair} {x.sleeve}")
             for x in (r.itertuples() if len(r) else []) if newest < int(x.sig) + need]
    why = []
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok:
        changed = False
        for _ in range(2):
            had = set(st)
            st, ch = freeze(st, c, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), hold, why)
            changed |= ch
            if not ch or "first" not in had:
                break
        if changed:
            NF.du_save_state(st, state_path)
    live_state, live_det, _ = verdict(c)
    bar = (f"Bar (pre-registered, frozen once at the first prefix of N ≥ {N_MIN} kept priced setups on ≥ {DAYS_MIN} UTC days, never re-fit; "
           f"one re-read at N ≥ {REREAD_N}): EXEMPT CANDIDATE (operator decides) iff mean % > 0 ∧ day-clustered P(mean > 0) ≥ {P_MIN:.2f} "
           f"({BOOT_N:,}, seed {BOOT_SEED}) ∧ no day and no pair > {SHARE_MAX * 100:.0f} % of Σ; mean ≤ 0 → KEEP THE HOLD; else keep observing.")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    if "first" not in st:
        L.append(f"{bar} Now: ⏳ {live_state} ({live_det}).")
    else:
        for k in ("first", "reread"):
            if k in st:
                f0 = st[k]
                L.append(f"**Frozen {'verdict' if k == 'first' else 're-read'} (prefix to {f0['at']}, frozen on the run of {f0['run_at']}, "
                         f"N {f0['n']} · {f0['days']} d): {f0['state']}** ({f0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    L.append(REVERT)
    if notes:
        L.append("BTC data: " + " · ".join(notes) + ".")
    return L + [""]


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    global EXPORT_GLOB, DEC_GLOB
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved = (EXPORT_GLOB, DEC_GLOB)
    dec0 = pd.DataFrame(columns=["t", "e", "pair", "gate", "ms"])
    dec0.attrs["newest_ms"] = 4_000_000_000_000
    CTX = dict(dec=dec0, m5=None, d1=None)                         # run() never reads the real journals or the network here
    try:
        chk((FRENZY3, N_MIN, DAYS_MIN, REREAD_N, P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED) ==
            (("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE"), 15, 8, 30, 0.90, 0.50, 4000, 7), "pre-registered constants pinned")
        th = dict(catchup=6, day_cap=3, bearish=True, cooldown=5.0, hold_green=12.0, slots={"FRENZY_LONG": 2, "FRENZY_WIDE": 2, "FRENZY_LITE": 2})
        S = "2026-10-09T14:50:00"
        sig = _ms(S)
        row = lambda **k: pd.Series(dict(dict(t=S, signal_at=S, pair="MAGICUSDT", sleeve="FRENZY_WIDE"), **k))   # noqa: E731

        def od(rows):
            o = pd.DataFrame(rows, columns=list(OCOLS))
            o["o_ms"], o["c_ms"] = o.opened_at.map(_ms), o.closed_at.map(_ms)
            o["_c"] = o.status.astype(str).str.upper().eq("CLOSED")
            return o
        stamp = ["2026-10-09T14:40:00", "2026-10-09T15:00:00", "BTCUSDT", "MOMENTUM", "CLOSED", 0.5, -0.1]
        o = od([stamp])
        chk(recheck(row(), o, th)[0] == "KEEP" and "BTC 1d +0.50" in recheck(row(), o, th)[1], "a clean refusal on a non-bearish day is kept")
        chk(recheck(row(reason="UNREAD"), o, th)[0] == "UNREAD", "an unreadable-check refusal is not a WILLY hold")
        cdn = ["2026-10-09T14:00:00", "2026-10-09T14:47:00", "MAGICUSDT", "MOMENTUM", "CLOSED", 0.5, 0.1]
        chk(recheck(row(), od([stamp, cdn]), th)[0] == "COOLDOWN" and recheck(row(), od([stamp, cdn]), dict(th, cooldown=0))[0] == "KEEP",
            "a close on the pair 3 min before → cooldown")
        syn = dict(synth=[("MAGICUSDT", "FRENZY_LONG", sig - 3_600_000, sig + 60_000)])
        chk(recheck(row(), o, th, syn)[0] == "PAIR_HELD", "an earlier KEPT refusal still open on the pair → pair held (self-occupancy)")
        syn2 = dict(synth=[(f"Y{i}USDT", "FRENZY_WIDE", sig - 60_000, sig + 60_000) for i in range(2)])
        chk(recheck(row(), o, th, syn2)[0] == "SLOTS", "earlier KEPT refusals fill the slots")
        syn3 = dict(synth=[("MAGICUSDT", "FRENZY_WIDE", sig - 7_200_000 + 60_000 * k, sig - 7_000_000 + 60_000 * k) for k in range(3)])
        chk(recheck(row(), o, th, syn3)[0] == "PAIR_DAY_CAP", "earlier KEPT refusals count toward the pair-day cap")
        dg = pd.DataFrame(dict(t=[S], e=["BLOCK"], pair=["MAGICUSDT"], gate=["FRENZY_ATR_HIGH"]))
        dg["ms"] = dg.t.map(_ms)
        chk(recheck(row(), o, th, dict(dec=dg))[0] == "WIDE_HOLD_GREEN" and recheck(row(sleeve="FRENZY_LONG"), o, th, dict(dec=dg))[0] == "KEEP"
            and recheck(row(), o, dict(th, hold_green=0), dict(dec=dg))[0] == "KEEP", "WIDE with an ATR_HIGH FRENZY leg → hold-green refuses it")
        chk("streak unverified" in recheck(row(), o, th, dict(dec=dg.assign(gate="FRENZY_GREEN_BAR")))[1], "a GREEN_BAR leg is flagged, kept")
        nb = pd.concat([dg.assign(gate="FRENZY_GREEN_BAR"), dg.assign(ms=sig + BAR_MS)], ignore_index=True)
        chk(recheck(row(), o, th, dict(dec=nb))[0] == "KEEP", "an ATR_HIGH on the NEXT bar never rules this bar out")
        chk("neighbouring bar has FRENZY_ATR_HIGH" in recheck(row(), o, th, dict(dec=dg.assign(ms=sig + BAR_MS)))[1], "a neighbour's code is a flag only")
        T0 = sig - 200 * BAR_MS
        up = pd.DataFrame(dict(open_time=T0 + np.arange(200) * BAR_MS, c=100 + np.arange(200) * 0.1))
        up["T"] = up.open_time + BAR_MS
        chk((btc_gap_at(up, sig) or 0) > 0 and btc_gap_at(up.assign(c=100 - np.arange(200) * 0.1), sig) < 0 and btc_gap_at(up, sig + 3_600_000) is None
            and btc_gap_at(up.iloc[:50], sig - 150 * BAR_MS) is None, "BTC 5m gap: sign, stale cache → None, short → None")
        d1 = pd.Series({(sig // 86_400_000 - 1) * 86_400_000: 99.0, (sig // 86_400_000 - 2) * 86_400_000: 100.0})
        chk(abs(btc_1d_at(d1, sig) + 1.0) < 1e-9 and btc_1d_at(d1, sig + 86_400_000) is None, "BTC 1d = the last closed day vs the one before")
        no1d = ["2026-10-08T14:45:00", "2026-10-08T15:00:00", "BTCUSDT", "MOMENTUM", "CLOSED", 0.5, 0.1]
        dn = up.assign(c=100 - np.arange(200) * 0.1)
        chk(recheck(row(), od([no1d]), th, dict(m5=dn, d1=d1))[0] == "BEARISH_DAY", "no same-day stamp → BTC daily + rebuilt gap decide")
        chk(recheck(row(), od([stamp, ["2026-10-09T15:10:00", "", "MAGICUSDT", "FRENZY_WIDE", "OPEN", 0.5, 0.1]]), th)[0] == "TAKEN_LATER",
            "the catch-up opened it 20 min later → not a lost trade")
        chk(recheck(row(), od([stamp, ["2026-10-09T15:30:00", "", "MAGICUSDT", "FRENZY_WIDE", "OPEN", 0.5, 0.1]]), th)[0] != "TAKEN_LATER",
            "a fill 40 min later is beyond 6 catch-up bars")
        chk(recheck(row(), od([stamp, ["2026-10-09T14:00:00", "", "MAGICUSDT", "MOMENTUM", "OPEN", 0.5, 0.1]]), th)[0] == "PAIR_HELD",
            "the pair already held → refused anyway")
        two = [["2026-10-09T14:00:00", "", f"X{i}USDT", "FRENZY_WIDE", "OPEN", 0.5, 0.1] for i in range(2)]
        chk(recheck(row(), od([stamp] + two), th)[0] == "SLOTS", "both WIDE slots busy → refused anyway")
        chk(recheck(row(), od([stamp] + [two[0]]), th)[0] == "KEEP", "one slot free → kept")
        cap = [[f"2026-10-09T0{h}:00:00", f"2026-10-09T0{h}:30:00", "MAGICUSDT", "FRENZY_WIDE", "CLOSED", 0.5, 0.1] for h in (1, 2, 3)]
        chk(recheck(row(), od([stamp] + cap), th)[0] == "PAIR_DAY_CAP" and recheck(row(sleeve="FRENZY_LITE"), od([stamp] + [
            [a, b, c, "FRENZY_LITE", e, f, g] for a, b, c, _, e, f, g in cap]), th)[0] == "PAIR_DAY_CAP", "pair-day cap: LONG / WIDE / LITE")
        bear = ["2026-10-09T14:45:00", "2026-10-09T15:00:00", "BTCUSDT", "MOMENTUM", "CLOSED", -0.8, -0.2]
        chk(recheck(row(), od([bear]), th)[0] == "BEARISH_DAY", "bearish day on the bot's own stamps → refused anyway")
        chk(recheck(row(), od([bear]), dict(th, bearish=False))[0] == "KEEP", "bearish gate off → kept")
        far = ["2026-10-09T10:00:00", "2026-10-09T10:30:00", "BTCUSDT", "MOMENTUM", "CLOSED", -0.8, -0.2]
        st_, note = recheck(row(), od([far]), th)
        chk(st_ == "KEEP" and "undecidable" in note, "no trend-gap stamp within 5 min, no BTC cache → kept, flagged (engine fails open)")
        # verdict / freeze
        n = 15

        def coh(p):
            days = [f"2026-10-{10 + i % 8:02d}" for i in range(n)]
            return pd.DataFrame(dict(pair=[f"P{i}" for i in range(n)], sleeve="FRENZY_LONG", day=days, pct=p,
                                     sig=[_ms(d + "T12:00:00") + i * 60_000 for i, d in enumerate(days)], signal_at=days))
        chk(verdict(coh([3.0] * 12 + [-3.0] * 3))[0] == "EXEMPT CANDIDATE (operator decides)", "mostly +3 → candidate")
        chk(verdict(coh([3.0] * 6 + [-3.0] * 9))[0] == "KEEP THE HOLD", "mostly −3 → keep the hold")
        chk(verdict(coh([3.0] * 8 + [-3.0] * 7))[0] == "KEEP OBSERVING", "barely positive → keep observing")
        chk(verdict(coh([3.0] * n).iloc[:5])[0] == "COLLECTING", "N < 15 → collecting")
        chk(verdict(coh([3.0] * 12 + [-3.0] * 3).assign(pair=["P0"] * 8 + [f"P{i}" for i in range(1, 8)]))[0] == "KEEP OBSERVING",
            "one pair carrying > 50 % of Σ → not a candidate")
        stt, ch = freeze({}, coh([3.0] * 12 + [-3.0] * 3), "now")
        chk(ch and stt["first"]["state"].startswith("EXEMPT") and not freeze(stt, coh([-3.0] * n), "later")[0]["first"]["state"].startswith("KEEP"),
            "frozen once, never re-fit")
        chk(not freeze({}, coh([3.0] * n), "x", [(pd.Timestamp("2026-10-10"), "unpriced X")], [])[1], "an unpriced row inside the prefix defers")
        # end-to-end on a temp tree
        with tempfile.TemporaryDirectory() as td:
            EXPORT_GLOB = os.path.join(td, "scalpars_orders_paper_*.csv")
            DEC_GLOB = os.path.join(td, "scalpars_decisions_paper_*.csv")
            pd.DataFrame(dict(t=[S, S], e=["BLOCK", "OPEN"], pair=["MAGICUSDT", "X"], gate=["FRENZY_GREEN_BAR", ""])).to_csv(
                os.path.join(td, "scalpars_decisions_paper_a.csv"), index=False)
            dd = load_decisions()
            chk(len(dd) == 1 and dd.gate.iloc[0] == "FRENZY_GREEN_BAR" and dd.attrs["newest_ms"] > sig, "journal loader: FRENZY-leg codes only")
            pd.DataFrame([dict(zip(OCOLS, stamp))]).to_csv(os.path.join(td, "scalpars_orders_paper_a.csv"), index=False)
            rows = pd.DataFrame([dict(t=S, pair="MAGICUSDT", dir="LONG", sleeve="FRENZY_WIDE", signal_at=S, willy_pair="API3USDT", pct=-3.0,
                                      exit_how="stop"),
                                 dict(t=S, pair="MAGICUSDT", dir="LONG", sleeve="FRENZY_WIDE", signal_at=S, willy_pair="API3USDT", pct=-3.0,
                                      exit_how="stop"),
                                 dict(t=S, pair="BATUSDT", dir="LONG", sleeve="FRENZY_WILLY", signal_at=S, willy_pair="API3USDT", pct=1.0,
                                      exit_how="take profit")])
            sp = os.path.join(td, "s.json")
            out = "\n".join(run(sig + 86_400_000, rows=rows, state_path=sp, th=th, ctx=CTX))
            chk("| **KEPT (lost trades)** | 1 | 1 | 0 % | -3.000 % | -3.00 % |" in out and "BATUSDT" not in out and "COLLECTING" in out
                and not os.path.exists(sp), f"end-to-end: deduped, WILLY rows out, collecting\n{out}")
            # an OPEN row from an old export is open only until that export's time; the newest export bounds freezing
            fo = os.path.join(td, "scalpars_orders_paper_b.csv")
            pd.DataFrame([dict(zip(OCOLS, ["2026-10-09T13:00:00", "", "MAGICUSDT", "MOMENTUM", "OPEN", 0.5, 0.1]))]).to_csv(fo, index=False)
            os.utime(fo, (sig / 1000 - 600, sig / 1000 - 600))       # that export was taken 10 min BEFORE the signal
            lo = load_orders()
            chk(recheck(row(), lo, th)[0] == "KEEP" and lo.attrs.get("newest_ms", 0) > sig, "an OPEN row is not held past its export's time")
            os.utime(fo, (sig / 1000 + 60, sig / 1000 + 60))
            os.utime(os.path.join(td, "scalpars_orders_paper_a.csv"), (sig / 1000 + 60, sig / 1000 + 60))
            lo = load_orders()
            out = "\n".join(run(sig + 86_400_000, rows=rows, orders=lo, state_path=sp, th=th, ctx=CTX))
            chk(recheck(row(), lo, th)[0] == "PAIR_HELD" and "PAIR_HELD" in out, "open at the export after the signal → pair held")
            chk(lo.attrs["newest_ms"] < sig + 7 * BAR_MS + COVER_SLACK_MS, "newest export before the catch-up window → coverage hold applies")
            unp = rows.iloc[:1].assign(pct=np.nan)
            out = "\n".join(run(sig + 5 * 86_400_000, rows=unp, orders=od([stamp]), state_path=sp, th=th, ctx=CTX))
            chk("out — UNPRICEABLE" in out, f"unpriced 2 days after its window → unpriceable\n{out}")
            out = "\n".join(run(sig + 3_600_000, rows=unp, orders=od([stamp]), state_path=sp, th=th, ctx=CTX))
            chk("kept — pending price | 1" in out, "a kept unpriced row shows as pending")
            chk("No FRENZY-family" in "\n".join(run(sig, rows=rows.iloc[:1].assign(signal_at="x", t="y"), state_path=sp, th=th, ctx=CTX)),
                "a row with no readable time is dropped")
            chk("No FRENZY-family" in "\n".join(run(sig, rows=rows.iloc[2:], state_path=sp, th=th, ctx=CTX)), "no FRENZY rows → says so")
    finally:
        EXPORT_GLOB, DEC_GLOB = saved
    print(f"selftest HOLD_FRENZY_EXEMPT OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
