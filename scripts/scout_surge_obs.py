#!/usr/bin/env python3
"""⚡ Scout — SURGE_LONG comparison OBSERVATION (pre-registered 2026-10-04, operator; DECISION_LOG 202). OBSERVE only — never a trade.

Live SURGE_LONG runs option B (0.3 % · 5× · strict 24 h high · market volume ≥ 1 · spacing only after a fill). This section tracks the
runner-up, option A, on the same tape so the two can be compared at 30 triggers. Called by scripts/opportunity_scout.py every run.

RULE A (frozen): on a CLOSED BTC 5m bar ① BTC 30-min return ≥ +0.50 % ② bar quote volume ≥ 3× the median of the prior 288 bars
  ③ close ≥ the prior 24 h high (strict) ④ market volume ≥ 1.0 (the engine's reading on the trigger bar: scripts/scout_gvol.py — top-50
  ranked per bar by 24 h quote volume over the engine's universe, frozen once computed; 2026-10-06 fix, was the scout's top-80 frames) → a
  trigger; the next needs 4 h from ANY trigger (a bar failing ④ never starts it). ADX / breadth are shown, never used.
PICKS   the live SURGE_LONG selection (services.surge.surge_pair_pick): top-20 by 24 h volume at the bar (minus surge_long_pair_blacklist),
        ATR ≥ surge_atr_min_pct, outrunning BTC, ≤ surge_max_slots, in rank order.
OUTCOME entry 60 s after the bar closes (1m open), today's SURGE_LONG exit (the Bull-Run exit replica of scripts/surge_bearrun_review.py —
        BR_NO_BE_LOCK while surge_long_exit_no_lock is on, else LIVE) on 1m bars — a stop / trail fills at the crossing (the line) unless the minute OPENED through it (then that open) — net of
        0.09 % fees, 4 h hold. "open" until 4 h have passed. Also shown: whether the LIVE rule (option B's trigger legs) fired on that bar.
YEAR    scripts/surge_trigger_grid_report.py: A 92 triggers +0.113 %/trigger CI [−0.16, +0.40] (3 triggers carry it) · B 131 triggers
        +0.008 %/trigger → neither proven.
REVIEW  at 30 recorded A triggers: compare with the live B triggers of the same period (mean/trigger, ≥ 8 days, no trigger ≥ 50 % of the gain).
"""
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
BAR, MIN = 300_000, 60_000
CSV = os.path.join(ROOT, "reports", "SCOUT_SURGE_OBS.csv")
MOVE, VOLX, GVOL, COOLDOWN_MS, HOLD_MIN, REVIEW_N, FEE = 0.50, 3.0, 1.0, 4 * 3600_000, 240, 30, 0.09
YEAR_REF = "A 92 triggers · +0.113 %/trigger · CI −0.16 to +0.40 (3 triggers carry it) · live B 131 triggers · +0.008 %/trigger"


def _start_ms():
    try:
        import scout_revert_gates as _rg
        return _rg.deploy_ms("SURGE_B")
    except Exception:
        return int(pd.Timestamp("2026-10-05 01:40", tz="UTC").value // 1_000_000)


def _adx(d):
    h, l, c = d.h, d.l, d.c
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), d.index); ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), d.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr; ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    return (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14, adjust=False).mean()


def _market(frames, t):
    """(breadth %, market volume ratio) at bar t over the top-50 frames by rolling 24 h c·v volume. Only the BREADTH is used (shown, never a
    leg); the market volume leg reads scout_gvol (the engine's universe + per-bar rank) — this ratio is the scout frames' approximation."""
    snap = []
    for d in frames.values():
        if t in d.index:
            r = d.loc[t]
            if np.isfinite(r.qv24) and np.isfinite(r.m48) and r.m48 > 0:
                snap.append((r.qv24, r.c > r.e20, r.v, r.m48))
    top = sorted(snap, key=lambda x: -x[0])[:50]
    if len(top) < 30:
        return np.nan, np.nan
    return 100.0 * np.mean([x[1] for x in top]), sum(x[2] for x in top) / sum(x[3] for x in top)


def _m1(EX, retry, pair, start_ms, n=250):
    raw = retry(EX.fapiPublicGetKlines, {"symbol": pair, "interval": "1m", "startTime": int(start_ms), "limit": int(n)}) or []
    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in raw]


def _walk(m1, t_entry, atr, now_ms):
    """today's SURGE_LONG exit on 1m bars → (net %, reason, minutes) or (None, 'open', minutes so far)."""
    import surge_bearrun_review as R
    rows = [r for r in m1 if r[0] >= t_entry - MIN + 1]
    if not rows:
        return None, "no data", 0
    e = rows[0][1]
    ts, px = [], []
    for t0, o, h, l, c in rows:
        for k, p in enumerate((o, l, h, c)):          # low before high: conservative for a long
            ts.append(t0 + k * 15_000); px.append(p)
    ts, px = np.array(ts, dtype=float), np.array(px, dtype=float)
    pnl = (px / e - 1) * 100 - FEE
    _nl = bool((R.TH or {}).get("surge_long_exit_no_lock", False))   # the live SURGE_LONG exit (DECISION_LOG 203: no +0.2 % lock)
    fn, tp, hold = R.variants("LONG", atr)["BR_NO_BE_LOCK" if _nl else "LIVE"]
    m = ts <= t_entry + HOLD_MIN * MIN
    ts, pnl = ts[m], pnl[m]
    if not len(pnl):
        return None, "no data", 0
    peak = np.maximum.accumulate(np.maximum(pnl, 0.0))
    line = R._lines(fn, peak)
    hit = np.where(pnl <= line)[0]
    if len(hit):
        j = int(hit[0])
        # the bot fills at the CROSSING print: inside a minute the price passes the line on its way to the bar's low — fill at the line,
        # unless the minute OPENED through it (a gap → that open). Without this the 1m low over-states every stop (GTC 19:15: −2.58 vs −1.2).
        fill = float(pnl[j]) if j % 4 == 0 else float(line[j])   # samples per minute = open, low, high, close
        return round(fill, 3), ("STOP" if line[j] < 0 else "TRAIL"), round((ts[j] - t_entry) / MIN)
    if now_ms < t_entry + HOLD_MIN * MIN:
        return None, "open", round((now_ms - t_entry) / MIN)
    return round(float(pnl[-1]), 3), "TIME", HOLD_MIN


def scan(EX, retry, cfg, last_closed, alts, btc_full, in_now, now_ms):
    """→ (bars: every bar of the last 24 h whose BTC move ∧ volume legs held, with each other leg; observations with picks / outcomes)."""
    from services.surge import surge_pair_pick, surge_trigger
    th = SimpleNamespace(**cfg)
    b = btc_full.copy()
    b["qv"] = b.c * b.v
    b["r30"] = (b.c / b.c.shift(6) - 1) * 100
    b["volx"] = b.qv / b.qv.shift(1).rolling(288).median()
    b["hi"] = b.h.shift(1).rolling(288).max()
    b["adx"] = _adx(b); b["d_adx"] = b.adx - b.adx.shift(6)
    frames = {}
    for p, d in list(alts.items()) + [("BTCUSDT", btc_full)]:
        if d is None or len(d) < 300:
            continue
        x = d.copy(); x["qv24"] = (x.c * x.v).rolling(288).sum(); x["e20"] = x.c.ewm(span=20, adjust=False).mean(); x["m48"] = x.v.rolling(48).mean()
        frames[p] = x
    old = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
    t0 = last_closed - 288 * BAR
    _prev = old[old.trig.astype("int64") <= t0] if len(old) and "trig" in old else pd.DataFrame()   # the window is re-judged from scratch each run
    last_fire = int(_prev.trig.max()) if len(_prev) else -10**15
    bars, obs = [], []
    global LAST_WINDOW
    LAST_WINDOW = (t0, last_closed)   # save() drops recorded A triggers inside this window the run no longer reproduces (review)
    cand = [int(t) for t in b.index[b.index > t0] if b.at[t, "r30"] >= MOVE and b.at[t, "volx"] >= VOLX]
    gmap = {}
    if cand:
        import scout_gvol as SG
        gmap = SG.ensure(EX, retry, cfg, cand, now_ms, pre={**(alts or {}), "BTCUSDT": btc_full})
    unread = gmap_unread(cand, gmap)
    if unread:   # an unread market volume is NOT a failed leg: nothing fires this run, the stored triggers stay, the window is re-judged next run
        LAST_WINDOW = None
    for t in b.index[b.index > t0]:
        r = b.loc[t]
        if not (r.r30 >= MOVE and r.volx >= VOLX):
            continue
        br, _ = _market(frames, t)
        gv = float(gmap.get(int(t), np.nan))
        br6, _ = _market(frames, t - 6 * BAR)
        legs = dict(ok_high=bool(r.c >= r.hi), ok_gvol=bool(np.isfinite(gv) and gv >= GVOL))   # ADX / breadth: shown, never used
        i = b.index.get_loc(t)
        view = [[int(k), float(x.o), float(x.h), float(x.l), float(x.c), float(x.v)] for k, x in b.iloc[max(0, i - 305):i + 1].iterrows()] + [[int(t) + BAR, 0, 0, 0, 0, 0]]
        live = surge_trigger(view, th, "LONG") is not None and (float(cfg.get("surge_long_gvol_min", 0) or 0) <= 0 or (np.isfinite(gv) and gv >= float(cfg.get("surge_long_gvol_min"))))
        row = dict(trig=int(t), close_utc=pd.to_datetime(int(t) + BAR, unit="ms").strftime("%Y-%m-%d %H:%M"), r30=round(r.r30, 3),
                   volx=round(r.volx, 1), vs_hi=round((r.c / r.hi - 1) * 100, 3), gvol=(round(gv, 2) if np.isfinite(gv) else None),
                   d_adx=round(r.d_adx, 2), breadth=(round(br, 0) if np.isfinite(br) else None),
                   d_breadth=(round(br - br6, 0) if np.isfinite(br) and np.isfinite(br6) else None), live_rule=bool(live), **legs)
        row["all"] = all(legs.values())
        row["fired"] = bool(row["all"] and t - last_fire >= COOLDOWN_MS)
        if row["all"] and not row["fired"]:
            row["note"] = "cooldown"
        if unread:   # the cooldown chain is not advanced on a partial read
            row.update(fired=False, all=False, note=("gvol unread — judged next run" if int(t) in unread else "pending (gvol unread in the window)"))
            if int(t) in unread:
                row["ok_gvol"] = None
        bars.append(row)
        if not row["fired"]:
            continue
        last_fire = int(t)
        if int(t) + BAR < _start_ms():   # the comparison counts only from the option-B deploy (same period as the live B triggers)
            row["note"] = "before the B deploy"
            continue
        bl = {x.strip().upper() for x in str(cfg.get("surge_long_pair_blacklist") or "").split(",") if x.strip()} | {"BTCUSDT", "ETHUSDT"}
        cand = sorted((p for p in in_now if p not in bl and p in frames and t in frames[p].index and np.isfinite(frames[p].at[t, "qv24"])),
                      key=lambda p: -frames[p].at[t, "qv24"])[:int(cfg.get("surge_universe_size", 20) or 20)]
        picks = []
        for p in cand:
            d = alts[p]
            rows = [[int(k), float(x.o), float(x.h), float(x.l), float(x.c), float(x.v)] for k, x in d[d.index <= t].tail(60).iterrows()]
            ok, why, atr, pm = surge_pair_pick(rows, int(t), float(r.r30), th, "LONG")
            if ok:
                picks.append((p, atr, pm))
            if len(picks) >= int(cfg.get("surge_max_slots", 4) or 4):
                break
        t_entry = int(t) + BAR + MIN
        if not picks:
            obs.append(dict(row, pair="", atr=None, pair_move=None, pnl=None, exit="no pick", held_min=None))
        for p, atr, pm in picks:
            try:
                pnl, why, held = _walk(_m1(EX, retry, p, t_entry - MIN), t_entry, atr, now_ms)
            except Exception as e:
                pnl, why, held = None, f"error {str(e)[:40]}", None
            obs.append(dict(row, pair=p, atr=round(atr, 2), pair_move=round(pm, 2), pnl=pnl, exit=why, held_min=held))
            time.sleep(0.05)
    return bars, obs


LAST_WINDOW = None


def gmap_unread(cand, gmap):
    """the candidate bars whose market volume could not be read this run (empty set = complete)."""
    return {int(t) for t in cand if int(t) not in (gmap or {})}


def save(obs):
    """reports/SCOUT_SURGE_OBS.csv — one row per trigger × pick (pair '' = no pick); the latest run wins; a finished outcome is never
    replaced by an unfinished one."""
    new = pd.DataFrame(obs)
    old = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
    if len(old) and LAST_WINDOW is not None:   # this run re-judged (t0, last_closed] from scratch → its triggers are authoritative there
        keep_trigs = set(new.trig.astype("int64")) if len(new) else set()
        tr = old.trig.astype("int64")
        old = old[~((tr > LAST_WINDOW[0]) & (tr <= LAST_WINDOW[1]) & ~tr.isin(keep_trigs))]
    if len(old) and len(new):
        old["pair"] = old["pair"].fillna("")
        kept = old.drop_duplicates(["trig", "pair"], keep="last").set_index(["trig", "pair"])
        for i, r in new.iterrows():
            k = (int(r.trig), r.pair)
            if pd.isna(r.pnl) and k in kept.index and pd.notna(kept.loc[k, "pnl"]):
                for c in ("pnl", "exit", "held_min"):
                    new.at[i, c] = kept.loc[k, c]
    a = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(a):
        a["pair"] = a["pair"].fillna("")
        a = a.drop_duplicates(["trig", "pair"], keep="last").sort_values("trig")
        tmp = CSV + ".tmp"; a.to_csv(tmp, index=False); os.replace(tmp, CSV)
    return a


def lines(bars, hist):
    ok = lambda v: "✓" if v else "✗"
    L = ["## ⚡ SURGE option-A comparison (pre-registered, OBSERVE only — never a trade)", "",
         f"Live SURGE_LONG = option B (0.3 % · 5× · market volume ≥ 1 · spacing only after a fill). Tracked here, the runner-up A (frozen): "
         f"BTC 30-min ≥ +{MOVE} % ∧ bar volume ≥ {VOLX:g}× ∧ close ≥ the 24 h high ∧ market volume ≥ {GVOL:g}; 4 h from any trigger. Picks / exit = "
         f"today's SURGE_LONG rules (1m bars, fills at the crossing, net of fees, 4 h). Year: {YEAR_REF} → neither proven. Review at "
         f"{REVIEW_N} A triggers vs the live B triggers of the same period.", ""]
    if bars:
        L += [f"Bars of the last 24 h where BTC moved ≥ +{MOVE} % on ≥ {VOLX:g}× volume (ADX / breadth for information only):", "",
              "| Bar close UTC | BTC 30m | vol × | vs 24h high | mkt vol | ADX Δ30m | breadth (Δ30m) | A trigger | live rule B legs |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in bars:
            L.append(f"| {r['close_utc'][5:]} | {r['r30']:+.2f}% | {r['volx']:.1f} | {r['vs_hi']:+.2f}% {ok(r['ok_high'])} | "
                     f"{r['gvol'] if r['gvol'] is not None else '–'} {ok(r['ok_gvol']) if r['ok_gvol'] is not None else 'unread'} | {r['d_adx']:+.1f} | "
                     f"{r['breadth'] if r['breadth'] is not None else '–'}% ({r['d_breadth'] if r['d_breadth'] is not None else '–'}) | "
                     f"{('🟢 FIRED' + (' (before B deploy — not counted)' if r.get('note') == 'before the B deploy' else '')) if r['fired'] else (r.get('note') or '–')} | {'met' if r['live_rule'] else '–'} |")
    else:
        L.append(f"No bar in the last 24 h with BTC ≥ +{MOVE} % in 30 min on ≥ {VOLX:g}× volume.")
    if hist is not None and len(hist):
        L += ["", "**Recorded A triggers** (latest first):", "", "| Bar close UTC | Pick | ATR | pair 30m | result | exit | min |", "|---|---|---|---|---|---|---|"]
        for r in hist.sort_values("trig", ascending=False).head(20).itertuples():
            L.append(f"| {str(r.close_utc)[5:]} | {r.pair or '—'} | {r.atr if pd.notna(r.atr) else '–'} | "
                     f"{f'{r.pair_move:+.2f}%' if pd.notna(r.pair_move) else '–'} | {f'{r.pnl:+.2f}%' if pd.notna(r.pnl) else '–'} | {r.exit} | "
                     f"{int(r.held_min) if pd.notna(r.held_min) else '–'} |")
        fin = hist[hist.pnl.notna()]
        per = fin.groupby("trig").pnl.mean()
        n_trig = hist.trig.nunique(); n_open = int((hist.exit.astype(str) == "open").sum())
        if len(per):
            tot = per.sum(); days = pd.to_datetime(per.index, unit="ms").normalize().nunique()
            top = per.max() / tot * 100 if tot > 0 else None
            L += ["", f"**Tally: {len(per)} of {REVIEW_N} A triggers finished** ({n_trig} recorded, {n_open} picks still open) · mean "
                      f"{per.mean():+.3f}%/trigger · {(per > 0).mean() * 100:.0f}% of triggers positive · {days} days · no trigger ≥ 50 % of the gain "
                      f"{ok(top is not None and top < 50)}" + (" · 📋 REVIEW DUE" if len(per) >= REVIEW_N else "")]
        else:
            L += ["", f"Tally: {n_trig} A triggers recorded, none finished yet."]
    return L + [""]
