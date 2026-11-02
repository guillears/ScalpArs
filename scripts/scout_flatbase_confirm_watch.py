#!/usr/bin/env python3
"""👁 Scout — FB_CONFIRM watch line (pre-registered 2026-10-30, DECISION_LOG 417; OBSERVE ONLY — never changes config, no bot API; public 5m klines only).

QUESTION   Would FLATBASE_SHORT do better entering after a 4-hour close below the signal price, instead of at the signal? Operator 10-30: the trades go negative a lot first. Theory: if the coin cannot even
           close below where we wanted to short it, the drop has not started.
RULE       From each real fill (signal price E = its entry price, time te): at each UTC 4-hour candle close within 24 h after te, first close < E → shadow short at the next 5m open. None → NO TRADE. Same exit as live: stop
           first on a bar touching both, stop gross +10 % → net −10.09 (a gap books the open), TP gross −5 % → net +4.94 (the live booked values), else market at 72 h after te. No funding (paper rows carry none).
EVIDENCE   (in-sample, NEVER counted) B25 16 signals re-priced at +5 / −10: before 16 · 62 % · −0.62 %, real-account daily compound −0.43 → after 12 finished · 92 % · +3.75 %, +1.19, gross targets (at the net constants −0.70 → +3.69) (3 skipped, 1 unresolved); it removes 5 of the 6 stops. Only 3 of the
           16 really ran on +5 / −10. AGAINST: 2026 backtest 535 → 461 · +1.25 → +1.30 %/trade, total +670 → +601 pts, median hold 4.4 → 7.3 h. Found on these same fills → observe only.
COHORT     CLOSED FLATBASE_SHORT fills from the orders exports (export_glob: Downloads + exports_archive/), dedup (opened_at[:19], pair) — CLOSED beats OPEN, newest export wins; operator-closed out (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 417
           commit + 10 min (forward only). A fill is priced once its shadow is decided: skipped, TP, stop or 72 h; undecided → PENDING (holds the freeze). No klines → UNPRICED (holds the freeze).
FROZEN BAR (operator's aim: well above breakeven, same style as DL 416), read per SIGNAL, a skip counts 0 %: at the first prefix (by open time) of N ≥ 15 priced signals on ≥ 8 UTC days, decided ONCE, Δ = shadow − booked: ① mean Δ ≤ 0 → RETIRED; ② ENTRY CANDIDATE (operator decides; never
           automatic) iff mean Δ ≥ +0.50 %/signal ∧ shadow mean > 0 ∧ day-clustered bootstrap P(mean Δ > 0) ≥ 0.90 (4,000 resamples, seed 7) ∧ no single signal ≥ 50 % of the positive Σ Δ; ③ else KEEP OBSERVING. Deferred until the newest export was downloaded ≥ the prefix end + 72 h (a long-held fill
           could be missing). The read shows before → after with the daily compound return. observe_registry pins today's exit (a change → ⚠ STALE). A candidate ships only after the 30–50 % in-sample haircut, real-sizing daily compound, leave-one-out and a pre-committed revert.
✅ SHIPPED   2026-11-02, DECISION_LOG 428 (operator: "ship FB_CONFIRM with the revert gate" — DECLARED OVERRIDE at 4 of 15 signals): the engine
           (flatbase_confirm_hours 24) now opens every TRADE-routed S4 signal only at the first 4-h close below the S4 close within 24 h, else
           SKIPPED. The observe line above CLOSES at the DL 428 commit + 10 min (after it the real fills are already confirmed — a shadow would
           confirm twice); its frozen bar no longer decides. Fills from then on feed the REVERT GATE:
🔒 REVERT   the first 8 CLOSED confirmed fills (opened ≥ the DL 428 commit + 10 min, by open time, operator-closed out), decided ONCE: avg booked
           % < 0 OR ≥ 3 of them stopped (close_reason STOP_LOSS / BACKSTOP*) → REVERT (operator sets flatbase_confirm_hours 0 — waits already
           running finish on their own 24 h clock; never automatic) ·
           else KEEP. 3 stops among the first 8 decide early (no later fill can undo them). Deferred until the newest export is ≥ the 8th fill +
           72 h (an earlier-opened fill could still be open). Side line (no verdict): every SKIPPED episode (decisions export, e=FLATBASE
           SKIPPED) priced as if entered at its S4 close (next 5m open, +5 / −10, 72 h) = what the confirm gave up.
🏦 BANK-TYPE (DECISION_LOG 429, operator 2026-11-02: "keep any close below and track BANK-type trades"): every confirmed fill split by DEPTH =
           how far its confirm close sits below the S4 close, % (the ledger's OPENED price = the confirm close since DL 428; S4 close = the order's
           entry_frenzy_vwap × (1 + entry_frenzy_vs_vwap_pct / 100); joined on pair + spike). SHALLOW = depth < 1 % (BANK 10-26: 0.81 %, −10.09 —
           the registration, never counted). 🔒 frozen read, once, at the first 10 shallow fills: shallow avg < 0 ∧ shallow avg < the deep side's
           avg (deep N ≥ 5; fewer → the read waits) → 1 % MARGIN CANDIDATE (operator decides; never automatic) · else KEEP "any close below".
           Scored only when the ledger's OPENED reason names FB_CONFIRM and depth > 0.001 % (an S4-priced row — a revert to 0, a recovered
           order — is "not a confirm entry"); no ledger row → unscored.
📈 RE-SIZE  (operator 2026-11-02: "if FB_CONFIRM performs well, the re-sizing is next") a frozen KEEP opens the DL 363 ladder step
           0.2 → 0.5 invest for the operator (confirmed fills only, real-path DCR before → after, flatbase_max_open re-decided).
STATE      reports/SCOUT_FB_CONFIRM.json (gitignored; the revert verdict under key FLATBASE_SHORT_REVERT).
Usage: venv/bin/python -I scripts/scout_flatbase_confirm_watch.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK        # noqa: E402  (operator-closed fills out · the DL commit time)
import scout_b1h_negflank as NF    # noqa: E402  (state I/O)
import scout_long_stop4 as LS      # noqa: E402  (budgeted, cached public fapi klines)

DL = 417
DL_SHIP = 428                       # ✅ FB_CONFIRM shipped (engine flatbase_confirm_hours) — closes the observe line, opens the revert gate
SLEEVE = "FLATBASE_SHORT"
REVERT_KEY, REVERT_N, REVERT_STOPS = "FLATBASE_SHORT_REVERT", 8, 3
RESIZE_FROM, RESIZE_TO = 0.2, 0.5   # 📈 a KEEP opens the DL 363 ladder step (operator 2026-11-02) — never automatic
BANK_KEY, BANK_DEPTH, BANK_N, BANK_DEEP_MIN = "FLATBASE_SHORT_BANK1", 1.0, 10, 5   # 🏦 (DECISION_LOG 429) shallow = confirm close < 1 % below S4
BANK_EPS = 1e-3                     # depth ≤ this = an S4-priced ledger row (a revert to 0, a recovered order) — never a confirm entry
STOP_REASONS = ("STOP_LOSS", "BACKSTOP_STOP", "BACKSTOP")
CONFIRM_H, MAX_H = 24, 72
STOP_GROSS, TP_GROSS = 10.0, 5.0
STOP_NET, TP_NET, TIME_FEE = -10.09, 4.94, 0.09
N_MIN, DAYS_MIN = 15, 8
D_MIN, P_MIN, SHARE_MAX = 0.50, 0.90, 0.50
BOOT_N, BOOT_SEED = 4000, 7
WAIT_MIN = MAX_H * 60
M5, H4, HOUR = 300_000, 14_400_000, 3_600_000
PAGE_BARS = 900                    # 75 h of 5m bars
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_FB_CONFIRM.json")
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_fb_confirm")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "pnl", "close_reason",
        "entry_frenzy_spike_at", "entry_frenzy_vwap", "entry_frenzy_vs_vwap_pct")   # 🏦 (429) the S4 close = vwap × (1 + vs_vwap %)
CANDIDATE = "ENTRY CANDIDATE (operator decides; never automatic)"
REGISTRATION = ("in-sample at registration (NEVER counted): B25 16 signals at +5 / −10 · −0.62 % → 12 finished · 92 % · +3.75 %, removes 5 of 6 stops; 2026 backtest +1.25 → +1.30 %/trade, total +670 → +601, holds longer")


def load(paths=None):
    """→ (closed FLATBASE fills — dedup, CLOSED beats OPEN, newest export wins —, the newest read export's download time or None)."""
    fr, read = [], []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
            read.append(f)
    newest = None
    for f in read:
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "day", "pct", "te"]), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[d.entry_strategy.astype(str) == SLEEVE].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c["E"] = pd.to_numeric(c.entry_price, errors="coerce")
    c = c[c.pct.notna() & c.E.notna()]
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    c["te"] = c.o.values.astype("datetime64[ms]").astype("int64")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


def price_shadow(bars, te, E, now_ms):
    """bars = [[open ms, o, h, l, c], …] 5m CLOSED, ascending, from te's bar. → (why, net %). why: SKIP (0 %) · TP · STOP · TIME · PENDING · UNPRICED. PURE."""
    if not len(bars):
        return "UNPRICED", np.nan
    b = np.asarray(bars, float)
    t, o, h, l, c = b[:, 0].astype("int64"), b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    closes = (t + M5) % H4 == 0
    in_window = t + M5 <= te + CONFIRM_H * HOUR
    after = t + M5 > te
    hit = np.flatnonzero(closes & in_window & after & (c < E))
    last_close = int(t[-1]) + M5
    if not len(hit):
        return ("SKIP", 0.0) if last_close >= te + CONFIRM_H * HOUR else ("PENDING", np.nan)
    i0 = int(hit[0]) + 1
    if i0 >= len(t):
        return "PENDING", np.nan
    Ee = float(o[i0])
    for i in range(i0, len(t)):
        if t[i] >= te + MAX_H * HOUR:
            return "TIME", float(-(o[i] / Ee - 1) * 100 - TIME_FEE)
        line = Ee * (1 + STOP_GROSS / 100)
        if h[i] >= line:
            return "STOP", float(STOP_NET if o[i] < line else -(o[i] / Ee - 1) * 100 - TIME_FEE)
        if l[i] < Ee * (1 - TP_GROSS / 100):     # the paper resting TP needs a trade THROUGH the limit
            return "TP", TP_NET
    return "PENDING", np.nan


def fetch_bars(pair, te, now_ms, bud):
    """5m closed bars from te's bar for up to 75 h. → (bars or None on a budget / error stop)."""
    start = int(te) // M5 * M5
    pg = LS.fapi_klines(pair, "5m", start, PAGE_BARS, bud, CACHE, now_ms, fetch=True)
    if pg is None or getattr(bud, "errors", 0):
        return None
    return [q[:5] for q in pg]


def verdict(rows):
    """rows = priced signals (booked pct, shadow net, day, pair) → (state, detail). PURE."""
    n, nd = len(rows), rows.day.nunique() if len(rows) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"{n}/{N_MIN} priced signals · {nd}/{DAYS_MIN} days"
    d = (rows.shadow - rows.pct).values
    md, ms = float(d.mean()), float(rows.shadow.mean())
    days = rows.day.values
    uniq = pd.unique(days)
    g = [d[days == u] for u in uniq]
    rng = np.random.default_rng(BOOT_SEED)
    s = np.array([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(BOOT_N)])
    p = float((s > 0).mean())
    pos = d[d > 0].sum()
    top = float(d.max() / pos) if pos > 0 else 0.0
    det = (f"{n} signals · {nd} d · booked avg {rows.pct.mean():+.3f} % → shadow {ms:+.3f} % (skip = 0) · mean Δ {md:+.3f} · P(Δ > 0) {p:.2f} · top signal {top:.0%} of + Σ Δ · "
           f"skipped {(rows.why == 'SKIP').sum()}")
    if md <= 0:
        return "RETIRED", det
    if md >= D_MIN and ms > 0 and p >= P_MIN and top < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def before_after(rows):
    """booked vs shadow: N · WR · avg · the daily compound return with each trade = the account (comparison only). PURE."""
    if not len(rows):
        return "no priced signals"
    span = pd.date_range(rows.o.min().normalize(), rows.o.max().normalize(), freq="D").strftime("%Y-%m-%d")

    def dcr(col):
        return (((1 + (rows.groupby("day")[col].sum() / 100).reindex(span).fillna(0)).prod()) ** (1 / len(span)) - 1) * 100
    k = rows[rows.why != "SKIP"]
    a = f"{len(k)} · {(k.shadow > 0).mean():.0%} · {k.shadow.mean():+.3f} %" if len(k) else "0 trades"
    return f"booked {len(rows)} · {(rows.pct > 0).mean():.0%} · {rows.pct.mean():+.3f} % · {dcr('pct'):+.3f} %/day → confirmed {a} · {dcr('shadow'):+.3f} %/day ({(rows.why == 'SKIP').sum()} skipped)"


def crossing_prefix(rows):
    seen = set()
    for k, d in enumerate(rows.day.values, 1):
        seen.add(d)
        if k >= N_MIN and len(seen) >= DAYS_MIN:
            return rows.iloc[:k]
    return None


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, bars_fn=None, ship_ms=None, dec_paths=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 417 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write. bars_fn(pair, te, now_ms) → bars or None; None → public klines.
    ship_ms None → the DECISION_LOG 428 commit + 10 min (not in git yet → the observe line still runs, no revert gate); False → not shipped.
    dec_paths → the decisions exports for the SKIPPED side line (None → Downloads + exports_archive/)."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 FB_CONFIRM — FLATBASE_SHORT entered after a 4-hour close below the signal, vs today's entry (pre-registered 2026-10-30, DECISION_LOG {DL}, OBSERVE only)", ""]
    since_given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="FB_CONFIRM watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    if ship_ms is None:
        ship_ms = _GK.dl_deploy_ms(DL_SHIP, lag_min=10, label="FB_CONFIRM ship")
    ship = pd.Timestamp(int(ship_ms), unit="ms") if (ship_ms is not None and ship_ms is not False) else None   # False = not shipped (tests)
    c, newest = load(paths)
    allc = c
    c = c[c.o >= since] if len(c) else c
    if ship is not None and len(c):
        c = c[c.o < ship]   # ✅ (428) after the ship the real fills are confirmed already — the observe line is closed
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    bud = LS.Budget()
    why, sh, stopped = [], [], False
    for r in keep.itertuples():
        bars = bars_fn(r.pair, r.te, now_ms) if bars_fn else (None if stopped else fetch_bars(r.pair, r.te, now_ms, bud))
        if bars is None:
            stopped = True
            why.append("UNPRICED")
            sh.append(np.nan)
            continue
        w, v = price_shadow(bars, r.te, r.E, now_ms)
        why.append(w)
        sh.append(v)
    keep = keep.assign(why=why, shadow=sh) if len(keep) else keep.assign(why=pd.Series(dtype=object), shadow=pd.Series(dtype=float))
    priced = keep[keep.shadow.notna()]
    waiting = keep[keep.shadow.isna()]
    if ship is not None:
        L += [f"✅ SHIPPED — DECISION_LOG {DL_SHIP} (engine flatbase_confirm_hours): this observe line closed at {ship:%Y-%m-%d %H:%M} UTC; its frozen bar no longer decides — the REVERT gate below does.", ""]
    L += [f"Cohort: CLOSED {SLEEVE} fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if since_given is None else 'floor given'}) · {_GK.operator_closed_line(ox)} · priced {len(priced)} · "
          f"skipped {(priced.why == 'SKIP').sum()} · pending / unpriced {len(waiting)}" + (" (" + ", ".join(f"{a} {b:%m-%d %H:%M}" for a, b in zip(waiting.pair, waiting.o)) + ")" if len(waiting) else "") + ".", "", "Before → after now: " + before_after(priced) + ".", ""]
    if stopped:
        L += ["⚠ kline fetch stopped this run (budget, cooldown or a network error): the rest stay unpriced; nothing freezes.", ""]
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    rec = st.get(SLEEVE) if ok else None
    damaged = rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec))
    rec = None if damaged else rec
    if damaged:
        L.append("⚠ frozen record damaged — operator restore needed; this line does not freeze until then.")
    hold = []
    if not rec and ok and not damaged and ship is None:
        pre = crossing_prefix(priced) if len(priced) else None
        if pre is not None:
            last = pre.o.max()
            if (waiting.o <= last).any() or stopped:
                hold.append("a signal inside the prefix is not priced yet")
            elif newest is None or newest < last + pd.Timedelta(minutes=WAIT_MIN):
                hold.append(f"newest export < the crossing signal + {WAIT_MIN} min (a fill could still be open)")
            else:
                state, det = verdict(pre)
                rec = dict(state=state, detail=det, before_after=before_after(pre), at=f"{last:%Y-%m-%d %H:%M} UTC",
                           run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                           keys=[f"{a:%Y-%m-%dT%H:%M:%S}|{b}|{w}" for a, b, w in zip(pre.o, pre.pair, pre.why)])
                if state_path:
                    st[SLEEVE] = rec
                    NF.du_save_state(st, state_path)
    live_state, live_det = verdict(priced)
    if rec and ok and state_path:
        L.append(f"**Frozen verdict (crossing at signal {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
        L.append(f"Before → after at the crossing: {rec.get('before_after', '–')}")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    else:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    if hold:
        L.append("⏸ freezing deferred: " + " · ".join(hold) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += ["", f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}, per signal, skip = 0): at N ≥ {N_MIN} priced signals on ≥ {DAYS_MIN} UTC days, decided ONCE — mean Δ ≤ 0 → RETIRED; {CANDIDATE} iff mean Δ ≥ {D_MIN:+.2f} ∧ shadow mean > 0 ∧ P(Δ > 0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ "
          f"no signal ≥ {SHARE_MAX:.0%} of + Σ Δ; else KEEP OBSERVING.", ""]
    if ship is not None:
        post = allc[allc.o >= ship] if len(allc) else allc
        for nm, fn in (("revert gate", lambda: revert_lines(post, newest, now_ms, st if ok else None, state_path if ok else None)),
                       ("skipped side line", lambda: skipped_lines(ship, now_ms, dec_paths, bars_fn)),
                       ("BANK-TYPE", lambda: bank_lines(post, now_ms, st if ok else None, state_path if ok else None, dec_paths, newest))):
            try:   # one section's failure never hides the others (the revert gate above all)
                L += fn()
            except Exception as e:
                L += [f"⚠ {nm} unavailable this run ({str(e)[:80]})", ""]
    return L


def bank_depth(fills, led):
    """→ fills + depth (% the confirm close sits below the S4 close; NaN = unscored). led = the FLATBASE ledger (OPENED rows: price = the
    confirm close). Join on pair + spike (± 1 s). PURE."""
    f = fills.copy()
    s4 = pd.to_numeric(f.get("entry_frenzy_vwap"), errors="coerce") * (1 + pd.to_numeric(f.get("entry_frenzy_vs_vwap_pct"), errors="coerce") / 100)
    sp = pd.to_datetime(f.get("entry_frenzy_spike_at"), format="mixed", errors="coerce")
    f["spike_s"] = (sp.values.astype("datetime64[s]").astype("int64")) if len(f) else []
    f.loc[sp.isna(), "spike_s"] = -1
    op = led[led.fb_status.astype(str) == "OPENED"] if len(led) else led
    px = {}
    for r in op.itertuples():   # only an OPENED row decided at the confirm bar carries the confirm close (DL 428)
        if r.spike_ms == r.spike_ms and r.px is not None and r.px == r.px and "FB_CONFIRM" in str(getattr(r, "fb_reason", "") or ""):
            px[(r.pair, int(round(r.spike_ms / 1000)))] = float(r.px)
    cc = [px.get((p, int(s))) or px.get((p, int(s) - 1)) or px.get((p, int(s) + 1)) for p, s in zip(f.pair, f.spike_s)]
    cc = pd.to_numeric(pd.Series(cc, index=f.index, dtype=object), errors="coerce")
    d = (1 - cc / s4) * 100
    f["depth"] = d.where(d > BANK_EPS)   # ≤ 0.001 % (S4-priced) or below 0 → never a confirm entry → unscored
    return f


def bank_lines(c, now_ms, st, state_path, dec_paths=None, newest=None):
    """🏦 (429) the BANK-type tracker on the confirmed fills (opened ≥ the ship)."""
    L = [f"### 🏦 BANK-TYPE — confirm close < {BANK_DEPTH:g} % below the S4 close (DECISION_LOG 429; registration BANK 10-26 0.81 % · −10.09, never counted)", ""]
    keep, _ = _GK.split_operator_closed(c) if len(c) else (c, c)
    if not len(keep):
        return L + ["  no confirmed fills yet.", ""]
    try:
        import scout_flatbase_short as FS
        led = FS.load_ledger(FS.read_decisions(dec_paths))
    except Exception as e:
        return L + [f"  ledger unreadable ({str(e)[:80]})", ""]
    try:
        f = bank_depth(keep.sort_values(["o", "pair"]), led)
    except Exception as e:   # never takes the revert gate down with it
        return L + [f"  depth unreadable ({str(e)[:80]})", ""]
    sh, dp, un = f[f.depth < BANK_DEPTH], f[f.depth >= BANK_DEPTH], f[f.depth.isna()]

    def _s(g):
        return f"N {len(g)} · WR {(g.pct > 0).mean():.0%} · avg {g.pct.mean():+.3f} %" if len(g) else "N 0"
    L += [f"  shallow (< {BANK_DEPTH:g} %): {_s(sh)}", f"  deep (≥ {BANK_DEPTH:g} %): {_s(dp)}",
          f"  unscored (not a confirm entry / no OPENED ledger row / stamp): N {len(un)}"]
    for r in sh.itertuples():
        L.append(f"    · {r.pair} {r.o:%m-%d %H:%M} depth {r.depth:.2f} % · {r.pct:+.2f} %")
    rec = (st or {}).get(BANK_KEY)
    if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
        return L + [f"⚠ frozen BANK record damaged — operator restore needed ({BANK_KEY}); not re-decided.", ""]
    wait = None
    if rec is None and st is not None and len(sh) >= BANK_N and (newest is None or newest < sh.iloc[:BANK_N].o.max() + pd.Timedelta(minutes=WAIT_MIN)):
        wait = f"⏸ freezing deferred: newest export < the {BANK_N}th shallow fill + {WAIT_MIN} min."
    if rec is None and st is not None and len(sh) >= BANK_N and wait is None and len(dp[dp.o <= sh.iloc[:BANK_N].o.max()]) < BANK_DEEP_MIN:
        wait = f"⏸ freezing deferred: fewer than {BANK_DEEP_MIN} deep fills by the {BANK_N}th shallow one — no comparison yet."
    if rec is None and st is not None and len(sh) >= BANK_N and wait is None:
        pre = sh.iloc[:BANK_N]
        d_before = dp[dp.o <= pre.o.max()]
        cand = pre.pct.mean() < 0 and pre.pct.mean() < d_before.pct.mean()
        rec = dict(state=("1 % MARGIN CANDIDATE (operator decides; never automatic)" if cand else 'KEEP "any close below"'),
                   detail=f"shallow {_s(pre)} vs deep {_s(d_before)}", at=f"{pre.o.max():%Y-%m-%d %H:%M} UTC",
                   run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
        if state_path:
            st[BANK_KEY] = rec
            NF.du_save_state(st, state_path)
    if wait:
        L.append(wait)
    if rec:
        L.append(f"**Frozen verdict ({rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
    else:
        L.append(f"Now: ⏳ COLLECTING ({len(sh)}/{BANK_N} shallow fills) — at {BANK_N}: shallow avg < 0 ∧ below the deep side (deep N ≥ 5) → 1 % MARGIN CANDIDATE, else KEEP.")
    return L + [""]


def revert_verdict(g):
    """g = the confirmed fills (pct, close_reason) in open order → (state, detail). PURE. 3 stops decide early; else below N → COLLECTING."""
    n = len(g)
    stops = int(g.close_reason.astype(str).str.upper().str.split(" ").str[0].isin(STOP_REASONS).sum()) if n else 0
    det = f"{n} fills · WR {(g.pct > 0).mean():.0%} · avg {g.pct.mean():+.3f} % · stops {stops}" if n else "0 fills"
    if stops >= REVERT_STOPS:
        return "REVERT", det + f" (≥ {REVERT_STOPS} stops)"
    if n < REVERT_N:
        return "COLLECTING", f"{n}/{REVERT_N} fills · stops {stops}/{REVERT_STOPS}"
    return ("REVERT", det + " (avg < 0)") if g.pct.mean() < 0 else ("KEEP", det)


def revert_lines(c, newest, now_ms, st, state_path):
    """the 🔒 revert gate on the confirmed fills (opened ≥ the ship). st None = state unreadable → no freeze. An early REVERT (3 stops) freezes
    at once: a fill opened earlier that lands in a later export could push a stop out of the first 8 — accepted, the error is toward safety."""
    L = [f"### 🔒 FB_CONFIRM revert gate (DECISION_LOG {DL_SHIP}) — the first {REVERT_N} confirmed fills: avg < 0 or ≥ {REVERT_STOPS} stopped → REVERT "
         "(operator sets flatbase_confirm_hours 0 — waits already running finish on their own clock; never automatic) · else KEEP", ""]
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    g = keep.sort_values(["o", "pair"]).iloc[:REVERT_N] if len(keep) else keep
    rec = (st or {}).get(REVERT_KEY)
    if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
        return L + [f"⚠ frozen REVERT record damaged — operator restore needed ({REVERT_KEY}); not re-decided.", ""]
    if rec is None and st is not None and len(g):
        state, det = revert_verdict(g)
        if state != "COLLECTING":
            early = state == "REVERT" and len(g) < REVERT_N
            last = g.o.max()
            if not early and (newest is None or newest < last + pd.Timedelta(minutes=WAIT_MIN)):
                L.append(f"⏸ freezing deferred: newest export < the {REVERT_N}th fill + {WAIT_MIN} min (an earlier-opened fill could still be open).")
            else:
                rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
                if state_path:
                    st[REVERT_KEY] = rec
                    NF.du_save_state(st, state_path)
    for r in g.itertuples():
        L.append(f"  · {r.pair} {r.o:%m-%d %H:%M} · {r.pct:+.2f} % · {r.close_reason}")
    ls, ld = revert_verdict(g)
    L.append(f"Confirmed fills: {len(keep)} closed · {_GK.operator_closed_line(ox)}.")
    if rec:
        L.append(f"**Frozen verdict ({rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
        if rec['state'] == "KEEP":
            L.append(f"📈 RE-SIZE NEXT (operator 2026-11-02: \"if FB_CONFIRM performs well, the re-sizing is next\"): the DL 363 ladder step "
                     f"{RESIZE_FROM:g} → {RESIZE_TO:g} invest is OPEN for the operator — read it on the confirmed fills only (the ladder line in "
                     "scout_flatbase_short counts the whole run at this size), with real-path DCR before → after and flatbase_max_open re-decided.")
    else:
        L.append(f"Now: ⏳ {ls} ({ld})." + (f" A KEEP opens the re-size step {RESIZE_FROM:g} → {RESIZE_TO:g} (operator)." if ls != "REVERT" else ""))
    return L + [""]


def skipped_lines(ship, now_ms, dec_paths=None, bars_fn=None):
    """side line, no verdict: SKIPPED episodes (no 4-h close below the S4 close in the wait) priced as if entered at the S4 bar."""
    L = ["### FB_CONFIRM — what the skips gave up (no verdict): SKIPPED episodes priced as if shorted at the S4 bar (next 5m open, +5 / −10, 72 h)", ""]
    try:
        import scout_flatbase_short as FS
        led = FS.load_ledger(FS.read_decisions(dec_paths))
        sk = led[(led.fb_status.astype(str) == "SKIPPED") & led.s4_ms.notna() & (led.s4_ms >= ship.value // 10**6 - 30 * 3_600_000)]
    except Exception as e:
        return L + [f"  unreadable ({str(e)[:80]})", ""]
    if not len(sk):
        return L + ["  none yet.", ""]
    bud, got = LS.Budget(), []
    for r in sk.itertuples():
        te = int(r.s4_ms) + M5                                            # the S4 bar's close = the old entry time
        bars = bars_fn(r.pair, te, now_ms) if bars_fn else fetch_bars(r.pair, te, now_ms, bud)
        v = np.nan
        if bars:
            b = [q for q in bars if int(q[0]) >= te]
            if b:
                Ee = float(b[0][1])
                for q in b:
                    if int(q[0]) >= te + MAX_H * HOUR:
                        v = float(-(q[1] / Ee - 1) * 100 - TIME_FEE); break
                    if q[2] >= Ee * (1 + STOP_GROSS / 100):
                        v = STOP_NET if q[1] < Ee * (1 + STOP_GROSS / 100) else float(-(q[1] / Ee - 1) * 100 - TIME_FEE); break
                    if q[3] < Ee * (1 - TP_GROSS / 100):
                        v = TP_NET; break
        got.append(v)
        L.append(f"  · {r.pair} S4 {pd.Timestamp(int(r.s4_ms), unit='ms'):%m-%d %H:%M} at {r.px} → " + (f"{v:+.2f} %" if v == v else "pending / unpriced"))
    x = pd.Series(got).dropna()
    if len(x):
        L.append(f"  priced {len(x)} · WR {(x > 0).mean():.0%} · avg {x.mean():+.2f} % (positive = the skip cost money)")
    return L + [""]


if __name__ == "__main__":
    print("\n".join(run()))
