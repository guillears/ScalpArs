#!/usr/bin/env python3
"""👁 Scout — WILLY_FAILED_SPIKE watch line for WILLY (registered 2026-11-09, DECISION_LOG 509; OBSERVE ONLY — never changes config, no bot API, no network).

QUESTION   WILLY lost −1.52 %/day on B25 (one size). Does refusing a dip whose red signal bar closed AT OR BELOW the spike-anchored VWAP fix it?
           zone = entry_frenzy_vs_vwap_pct ≤ 0 (frozen). Theory (written after the data — explains, never proves): below the average price
           paid since the spike, the spike has failed and the buyers since are underwater — no bounce to buy.
EVIDENCE   (in-sample, NEVER counted; reports/WILLY_REVIEW_2026-11-09.md) master on today's stack 38 · 82 % · +0.71 → kept 15 · 93 % · +1.87,
           daily compound (one size) +1.73 → +2.00; B25 33 · 48 % · −1.33 → 11 · 82 % · +0.21, −1.52 → +0.07. Pooled zone 45 · 53 % · −1.05
           on 26 days, P(mean < 0) 0.99. AGAINST: found by the 11-09 sweep (8,320 2D + 1D cells; luck search p 0.70 → NOT ABOVE LUCK); master
           alone does not separate (zone 23 · 74 % · −0.05); knife-edge cut (B25 kept −0.14 / +0.21 / −0.67 at −0.25 / 0 / +0.25). → observe only.
COHORT     CLOSED FRENZY_WILLY and FRENZY_WILLY_HT fills, each read and decided SEPARATELY (never pooled; HT added by DL 511 after the
           pre-registered theory test: B25 zone 29 · 62 % · −0.31 vs rest 6 · 83 % · +3.00, no master — TOO FEW, so observe; HT counts from the
           DL 511 commit + 10 min, WILLY from DL 509), from the orders exports (export_glob: Downloads +
           exports_archive/), dedup (opened_at[:19], pair, entry_strategy) — CLOSED beats OPEN, newest export wins; operator-closed out (scout_gatekit).
           COUNTED: opened ≥ the DECISION_LOG 509 commit + 10 min (WILLY_HT: ≥ the DL 587 commit + 10 min since the 11-19 re-base; DL 511 before) — forward only. rest = stamp present and > 0 · stamp missing → UNSCORED.
FROZEN BAR the WILLY_GAP bar (DL 414), never re-fit: at the first prefix where the ZONE reaches N ≥ 15 on ≥ 8 UTC days, decided ONCE —
           ① zone avg ≥ rest avg, OR zone avg ≥ 0 → RETIRED; ② else FILTER CANDIDATE (operator decides; never automatic) iff zone WR < 69.0 %
           ∧ day-bootstrap P(mean < 0) ≥ 0.95 ∧ no single day / pair ≥ 50 % of the zone's gross loss; ③ else KEEP OBSERVING. Pinned to the V2b
           exit keys in observe_registry.
🔁 HT RE-BASED 2026-11-21 (DECISION_LOG 608 exit, CLAUDE.md entry × exit ④; first re-base 11-19 at DL 587): WILLY_HT runs its OWN exit — flat TP +6 net
           (resting maker) · stop −4 net · 24 h cap · no late TP · no spike-high target. The HT half now counts only HT fills opened ≥ the
           DL 608 commit + 10 min, judges WR against the FROZEN HT breakeven 4 ÷ 10 = 40.0 % (WILLY keeps 69.0 %), freezes under the state key
           FRENZY_WILLY_HT@608 (the old FRENZY_WILLY_HT record of the +2 / V2b era stays in the file for history and is never read as the new
           verdict) and defers freezing until the newest export is ≥ the crossing fill + HT's own max hold. The registry pin carries the HT exit
           keys → a later HT exit change shows ⚠ STALE. Plain FRENZY_WILLY entries are OFF since DL 587 → the WILLY half collects no new fills.
           ⚠ FAILED_SPIKE is ARMED for WILLY_HT since DL 527 (zone entries are REFUSED) → forward HT zone fills should be ~0 and this line mostly
           shows rest-side fills; the filter's forward verdict is the blocked-trade pricer (scout_blocked_filters.py, FRENZY_WILLY_HT ·
           WILLY_FAILED_SPIKE), which prices the refused zone trades on HT's live exit.
           ⚠ HT's 40.0 % breakeven is frozen from THEORY (+6 / −4 net = 4 ÷ 10 — no stop slippage, no 24 h time exits), while WILLY's 69.0 %
           came from real fills (V2b, N 50).
🔒 WILLY OFF TRIGGER (DL 509, pre-committed, operator 11-09): read when WILLY_GAP's FRENZY_WILLY verdict freezes (reports/SCOUT_WILLY_GAP.json) —
           the forward fills KEPT by WILLY_GAP (outside its zone, since DL 414) and KEPT by WILLY_FAILED_SPIKE (stamp > 0, since DL 509): if BOTH have
           WR < 69.0 % and avg < 0 → WILLY OFF is the recommendation (operator switches it). FAILED_SPIKE-kept N < 5 at that moment → the GAP-kept read
           alone decides. Printed below, never acts.
           ⛔ RETIRED 2026-11-11 (DECISION_LOG 531, operator): it fired MET (GAP-kept 11 · 55 % · −1.31 / FAILED_SPIKE-kept 5 · 40 % · −2.10) on
           single-filter cohorts, but since DL 527 WILLY runs BOTH filters — the fills passing both since DL 414 were 4 · 100 % · +1.00 %. Its
           frozen record is shown as history only (off_trigger(), kept for the record).
🔒 WILLY OFF TRIGGER v2 (DL 531, pre-committed, operator 11-11): the first 15 CLOSED FRENZY_WILLY fills opened ≥ the DECISION_LOG 527 commit
           + 10 min (both filters armed — every one passes WILLY_GAP and WILLY_FAILED_SPIKE), operator-closed out: WR < 69.0 % ∧ avg < 0 →
           recommend WILLY OFF (operator switches it) · else WILLY STAYS. Decided ONCE (OFF_TRIGGER_531), after the newest export is ≥ the
           15th fill + WILLY's max hold. Never acts.
STATE      reports/SCOUT_WILLY_FAILED_SPIKE.json (gitignored).
Usage: venv/bin/python -I scripts/scout_willy_failed_spike_watch.py
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
import scout_b1h_negflank as NF    # noqa: E402  (day-bootstrap, gross-loss share, freeze hold, state I/O)
import scout_willy_gap_watch as WG  # noqa: E402  (the WILLY_GAP zone cuts, for the off trigger — one definition)

DL = 509
SLEEVES = ("FRENZY_WILLY", "FRENZY_WILLY_HT")   # each read and decided SEPARATELY (never pooled); HT added by DL 511, counted from ITS commit
HT_DL = 511                        # HT's registration (history)
HT_EXIT_DL = 608                   # 🔁 HT re-based on its own TP +6 / SL −4 / 24 h exit (DL 608; first 587 at +5): HT counts from THIS commit + 10 min
VWAP_CUT = 0.0
MIN_VWAP_KEPT = 5
N_MIN, DAYS_MIN = 15, 8
BE_WR = 0.690                      # FROZEN 2026-10-29: WILLY V2b breakeven WR (avg win +1.806 / loss −4.009, N 50)
HT_BE_WR = 0.400                   # FROZEN 2026-11-21 (DL 608 exit): HT breakeven = |loss| ÷ (win + |loss|) = 4 ÷ 10 at TP +6 / SL −4 net (0.444 at +5)
BE_NOTE = "HT's 40.0 % breakeven is FROZEN FROM THEORY (+6 / −4 net = 4 ÷ 10; no stop slippage, no 24 h time exits counted), while WILLY's 69.0 % came from real fills (V2b, N 50)"
BE_BY = {"FRENZY_WILLY": BE_WR, "FRENZY_WILLY_HT": HT_BE_WR}
STATE_KEY = {"FRENZY_WILLY": "FRENZY_WILLY", "FRENZY_WILLY_HT": "FRENZY_WILLY_HT@608"}   # the pre-608 HT verdicts are never read as the new one
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
CAP_DEFAULT = 90                   # WILLY's live max hold (frenzy_willy_max_hold_minutes) — read from the config at run time


def cap_minutes(path=None, es=None):
    """frenzy_willy_max_hold_minutes from trading_config.json (unreadable → 90). es = FRENZY_WILLY_HT → HT's own
    frenzy_willy_ht_max_hold_minutes when 1 – 10,080 (DL 587; 0 / blank / out of range → WILLY's, as the engine's frenzy_willy_levels)."""
    import json
    try:
        with open(path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_config.json")) as f:
            c = json.load(f)
        th = c.get("thresholds", c)
        cap = float(th.get("frenzy_willy_max_hold_minutes", CAP_DEFAULT))
        if es == "FRENZY_WILLY_HT":
            try:
                hm = float(th.get("frenzy_willy_ht_max_hold_minutes", 0) or 0)
                if 1 <= hm <= 10080:
                    cap = float(round(hm))
            except (TypeError, ValueError):
                pass
        return cap
    except Exception:
        return float(CAP_DEFAULT)


ZONE_COL = "entry_frenzy_vs_vwap_pct"
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_FAILED_SPIKE.json")
GAP_STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_GAP.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason", ZONE_COL, "entry_adx", "entry_ema_gap_8_13")
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
REGISTRATION = "in-sample at registration (NEVER counted): pooled zone 45 · 53 % · −1.05 vs kept 26 · 88 %; master 38 → 15 kept, +1.73 → +2.00 %/day; B25 33 → 11 kept, −1.52 → +0.07; luck search p 0.70 (NOT ABOVE LUCK); master alone does not separate · WILLY_HT (DL 511, B25 only, no master): zone 29 · 62 % · −0.31 vs rest 6 · 83 % · +3.00"


def load(paths=None, extra=()):
    """→ (closed WILLY / WILLY_HT fills — dedup, CLOSED beats OPEN, newest export wins —, the newest read export's download time or None).
    `extra` = more order columns to keep (scout_willy_ht_lines, DECISION_LOG 580)."""
    cols = tuple(COLS) + tuple(c for c in extra if c not in COLS)
    fr, read = [], []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
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
        return pd.DataFrame(columns=list(cols) + ["o", "day", "zone", "pct"]), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(cols) + ["_m"])
    d = d[d.entry_strategy.astype(str).isin(SLEEVES) & (d.direction.astype(str) == "LONG")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    vw = pd.to_numeric(c[ZONE_COL], errors="coerce")
    c["zone"] = np.where(vw.isna(), "unscored", np.where(vw <= VWAP_CUT, "zone", "rest"))
    adx, gap = pd.to_numeric(c["entry_adx"], errors="coerce"), pd.to_numeric(c["entry_ema_gap_8_13"], errors="coerce")
    c["gap_zone"] = np.where(adx.isna() | gap.isna(), "unscored", np.where((adx > WG.ADX_CUT) | (gap <= WG.GAP_CUT), "zone", "rest"))   # WILLY_GAP (DL 414), for the off trigger
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


def verdict(zone, rest, be_wr=BE_WR):
    """zone / rest = counted fills (pct, day, pair) → (state, detail); be_wr = the sleeve's FROZEN breakeven (BE_BY). PURE."""
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz, mr = float(zone.pct.mean()), (float(rest.pct.mean()) if len(rest) else float("nan"))
    wr = float((zone.pct > 0).mean())
    p = NF.p_mean_neg(zone.pct.values, zone.day.values, BOOT_N, BOOT_SEED)
    sd, sp = NF.gross_loss_share(zone.pct.values, zone.day.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {wr:.0%} (breakeven {be_wr:.1%}) · avg {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · "
           f"P(mean<0) {(p if p is not None else float('nan')):.2f} · top day {sd:.0%} / top pair {sp:.0%} of the zone's gross loss")
    if mz >= 0 or (len(rest) and mz >= mr):
        return "RETIRED", det
    if wr < be_wr and mz < 0 and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def crossing_prefix(sc):
    """scored counted fills by open time → the prefix ending at the zone fill where the zone first reaches N ≥ 15 on ≥ 8 days."""
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(sc.zone.values, sc.day.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(d)
        if nz >= N_MIN and len(seen) >= DAYS_MIN:
            return sc.iloc[:k]
    return None


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % | "
            f"{pd.to_numeric(g.pnl, errors='coerce').sum():+,.0f} |")


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 509 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 WILLY_FAILED_SPIKE — WILLY entries whose signal bar closed at or below the spike VWAP ({ZONE_COL} ≤ {VWAP_CUT:g}; registered 2026-11-09, "
         f"DECISION_LOG {DL}, OBSERVE only)", ""]
    since_given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="WILLY_FAILED_SPIKE watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c0 = c
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    changed = False
    L += [f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if since_given is None else 'floor given'}) · zone = {ZONE_COL} ≤ {VWAP_CUT:g} · WILLY and WILLY_HT decided separately (HT re-based: from the DL {HT_EXIT_DL} commit + 10 min, its TP +6 / SL −4 / 24 h exit) · {_GK.operator_closed_line(ox)}.", ""]
    ht_since = (_GK.dl_deploy_ms(HT_EXIT_DL, lag_min=10, label="WILLY_FAILED_SPIKE (WILLY_HT, re-based DL 608) watch line") if since_given is None else since_ms)
    for es in SLEEVES:
        s = keep[keep.entry_strategy.astype(str) == es] if len(keep) else keep
        cap, be = cap_minutes(es=es), BE_BY[es]
        notes = []
        if es == "FRENZY_WILLY_HT":
            if ht_since is None:
                L += [f"### {es}", "", f"⚠ HT re-based on the DECISION_LOG {HT_EXIT_DL} exit (TP +6 / SL −4 / 24 h) — NOT EVALUABLE until that commit is in git; nothing is counted for WILLY_HT.", ""]
                continue
            s = s[s.o >= pd.Timestamp(int(ht_since), unit="ms")] if len(s) else s
            notes = [f"(re-based: HT fills opened ≥ {pd.Timestamp(int(ht_since), unit='ms'):%Y-%m-%d %H:%M} UTC — the DL {HT_EXIT_DL} exit TP +6 net "
                     f"(resting maker) / SL −4 / {cap:.0f} min cap / no late TP; frozen HT breakeven {be:.1%} = 4 ÷ 10; verdict key {STATE_KEY[es]})",
                     f"⚠ {BE_NOTE}.",
                     f"⚠ FAILED_SPIKE is ARMED for WILLY_HT since DECISION_LOG 527 — zone entries are REFUSED, so forward zone fills should be ~0 "
                     "and this table mostly shows rest-side fills. The filter's forward verdict is the blocked-trade pricer "
                     "(scout_blocked_filters: FRENZY_WILLY_HT · WILLY_FAILED_SPIKE — the refused zone trades priced on HT's live exit)."]
        else:
            notes = ["(plain WILLY entries back ON since DECISION_LOG 595 (2026-11-19) — OFF 11-18 → 11-19 (DL 587), no fills in that gap)"]
        sc = s[s.zone != "unscored"] if len(s) else s
        zone, rest = sc[sc.zone == "zone"], sc[sc.zone == "rest"]
        uns = s[s.zone == "unscored"] if len(s) else s
        L += [f"### {es}", ""] + notes + ["", "| Group | N | days | WR | avg % | Σ$ as traded |", "|---|---|---|---|---|---|",
              _row("**zone**", zone), _row("rest", rest),
              _row("UNSCORED (a stamp missing) — never counted", uns), ""]
        why = []
        rec = st.get(STATE_KEY[es]) if ok else None
        if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
            rec, ok = None, False      # a damaged record never re-freezes silently
        if not rec and ok:
            pre = crossing_prefix(sc) if len(sc) else None
            if pre is not None:
                last = pre.o.max()
                if newest is None or newest < last + pd.Timedelta(minutes=cap):
                    why.append(f"newest export < the crossing fill + {cap:.0f} min (a fill could still be open)")
                else:
                    state, det = verdict(pre[pre.zone == "zone"], pre[pre.zone == "rest"], be)
                    rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC",
                               run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                               keys=[f"{a:%Y-%m-%dT%H:%M:%S}|{b}|{g}" for a, b, g in zip(pre.o, pre.pair, pre.zone)])
                    if state_path:
                        st[STATE_KEY[es]] = rec
                        changed = True
        live_state, live_det = verdict(zone, rest, be)
        if rec and ok and state_path:
            L.append(f"**Frozen verdict (crossing at fill {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
            L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
        else:
            L.append(f"Now: ⏳ {live_state} ({live_det}).")
        if why:
            L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
        old = st.get(es) if (ok and STATE_KEY[es] != es) else None
        if isinstance(old, dict) and old.get("state"):
            L.append(f"History only (pre-DL {HT_EXIT_DL} exit era, never read as this verdict): {old.get('state')} frozen {old.get('run_at', '?')}.")
        L.append("")
    if changed and ok and state_path:
        NF.du_save_state(st, state_path)
    L += ["### ⛔ WILLY OFF trigger (DL 509) — RETIRED (DL 531): superseded by both filters armed; history only", ""]
    _old, _ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    if _ok and isinstance(_old.get("OFF_TRIGGER"), dict):
        L += [f"Frozen {_old['OFF_TRIGGER'].get('at')}: {_old['OFF_TRIGGER'].get('result')} · {_old['OFF_TRIGGER'].get('detail')} — never acted on.", ""]
    L += off_trigger_v2(now_ms, state_path=state_path, frame=(c0, newest))
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += [f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}, the WILLY_GAP bar): at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} UTC days, decided ONCE — "
          f"zone avg ≥ rest avg or ≥ 0 → RETIRED; {CANDIDATE} iff zone WR < the sleeve's frozen breakeven (WILLY {BE_WR:.1%} · HT {HT_BE_WR:.1%} since DL {HT_EXIT_DL}) ∧ zone mean < 0 at day-bootstrap P(mean < 0) ≥ "
          f"{P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no day / pair ≥ {SHARE_MAX:.0%} of the zone's gross loss; else KEEP OBSERVING. "
          "Haircut, before → after per batch with the daily compound return and a pre-committed revert before any ship.", ""]
    return L


def trigger(gap_kept, vwap_kept):
    """the pre-committed WILLY OFF trigger on the two kept cohorts (pct). PURE → (met: bool, detail)."""
    def bad(g):
        return len(g) > 0 and float((g.pct > 0).mean()) < BE_WR and float(g.pct.mean()) < 0
    use_vwap = len(vwap_kept) >= MIN_VWAP_KEPT
    met = bad(gap_kept) and (bad(vwap_kept) if use_vwap else True)
    f = lambda g: f"{len(g)} · {(g.pct > 0).mean():.0%} · {g.pct.mean():+.3f} %" if len(g) else "0"
    return met, (f"GAP-kept {f(gap_kept)} · FAILED_SPIKE-kept {f(vwap_kept)}" + ("" if use_vwap else f" (FAILED_SPIKE-kept < {MIN_VWAP_KEPT}: GAP-kept alone decides)"))


def _read_json(path):
    """plain, side-effect-free read of ANOTHER line's state file → (dict, readable). Missing → ({}, True); damaged → ({}, False)."""
    import json
    if not path or not os.path.exists(path):
        return {}, True
    try:
        with open(path) as f:
            d = json.load(f)
        return (d if isinstance(d, dict) else {}), isinstance(d, dict)
    except Exception:
        return {}, False


def off_trigger(now_ms, paths=None, gap_state=None, state_path=STATE, frame=None, gap_since=None, vw_since=None):
    """🔒 (509) the WILLY OFF trigger — decided ONCE, on the first run after WILLY_GAP's FRENZY_WILLY verdict froze and the newest export is
    ≥ the last kept fill + WILLY's max hold; frozen under OFF_TRIGGER in this line's state; never acts. WILLY_GAP's state is only READ."""
    L = ["### 🔒 WILLY OFF trigger (DL 509, pre-committed — decided once)", ""]
    rule = (f"Rule: when WILLY_GAP's FRENZY_WILLY verdict freezes — the forward fills KEPT by WILLY_GAP (since DL 414) and KEPT by WILLY_FAILED_SPIKE "
            f"(since DL {DL}) both WR < {BE_WR:.1%} and avg < 0 → recommend WILLY OFF (FAILED_SPIKE-kept N < {MIN_VWAP_KEPT} → GAP-kept alone).")
    gap_since = _GK.dl_deploy_ms(414, lag_min=10, label="WILLY_GAP watch line") if gap_since is None else gap_since
    vw_since = _GK.dl_deploy_ms(DL, lag_min=10, label="WILLY_FAILED_SPIKE watch line") if vw_since is None else vw_since
    if gap_since is None or vw_since is None:
        return L + [f"⚠ NOT EVALUABLE — the DL {'414' if gap_since is None else DL} commit is not in git; the trigger is not read.", rule, ""]
    c, newest = frame if frame is not None else load(paths)
    c = c[c.entry_strategy.astype(str) == "FRENZY_WILLY"] if len(c) else c
    if len(c):
        c, _ox = _GK.split_operator_closed(c)
    gk = c[(c.gap_zone == "rest") & (c.o >= pd.Timestamp(int(gap_since), unit="ms"))] if len(c) else c
    vk = c[(c.zone == "rest") & (c.o >= pd.Timestamp(int(vw_since), unit="ms"))] if len(c) else c
    met, det = trigger(gk, vk)
    own, own_ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    rec = own.get("OFF_TRIGGER") if own_ok else None
    if rec:
        return L + [f"**Frozen ({rec.get('at')}): {rec.get('result')}** · {rec.get('detail')}",
                    f"Live (information only, never re-decides): {'MET' if met else 'not met'} · {det}", rule, ""]
    if state_path and not own_ok:
        return L + [f"⚠ own state corrupt ({os.path.basename(state_path)}.*.bad) — trigger not frozen this run (operator restore). Now: {det}.", rule, ""]
    gst, gok = _read_json(gap_state or GAP_STATE)
    if not gok:
        return L + [f"⚠ WILLY_GAP state unreadable ({os.path.basename(gap_state or GAP_STATE)}) — operator restore needed; trigger not read. Now: {det}.", rule, ""]
    g = gst.get("FRENZY_WILLY")
    if not (isinstance(g, dict) and g.get("state")):
        return L + [f"⏳ waiting for WILLY_GAP's FRENZY_WILLY verdict to freeze — now: {det} (would be {'MET' if met else 'not met'}).", rule, ""]
    last = max([x for x in (gk.o.max() if len(gk) else None, vk.o.max() if len(vk) else None) if x is not None], default=None)
    if last is not None and (newest is None or newest < last + pd.Timedelta(minutes=cap_minutes())):
        return L + [f"⏸ WILLY_GAP frozen ({g.get('state')}); deciding deferred — the newest export is older than the last kept fill + the max hold. Now: {det}.", rule, ""]
    result = "MET — recommend WILLY OFF (operator switches it)" if met else "NOT MET — WILLY stays"
    if state_path and own_ok:
        own["OFF_TRIGGER"] = dict(result=result, detail=det, gap_state=g.get("state"),
                                  at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
        NF.du_save_state(own, state_path)
    return L + [f"**WILLY_GAP frozen ({g.get('state')}) → {result}** · {det}", rule, ""]


V2_DL, V2_N = 527, 15


def off_trigger_v2(now_ms, state_path=STATE, frame=None, since_ms=None, paths=None):
    """🔒 (DL 531) WILLY OFF v2 on the both-filters cohort: the first V2_N CLOSED FRENZY_WILLY fills ≥ the DL 527 commit + 10 min →
    WR < BE_WR ∧ avg < 0 → recommend OFF. Decided once under OFF_TRIGGER_531; never acts."""
    L = [f"### 🔒 WILLY OFF trigger v2 (DL 531, pre-committed — decided once)", ""]
    rule = (f"Rule: the first {V2_N} CLOSED FRENZY_WILLY fills since the DL {V2_DL} commit + 10 min (both filters armed): WR < {BE_WR:.1%} ∧ "
            "avg < 0 → recommend WILLY OFF (operator switches it); else WILLY stays.")
    since_ms = _GK.dl_deploy_ms(V2_DL, lag_min=10, label="WILLY OFF v2 (armed filters)") if since_ms is None else since_ms
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DL {V2_DL} commit is not in git; the trigger is not read.", rule, ""]
    c, newest = frame if frame is not None else load(paths)
    c = c[c.entry_strategy.astype(str) == "FRENZY_WILLY"] if len(c) else c
    if len(c):
        c, _ox = _GK.split_operator_closed(c)
    c = c[c.o >= pd.Timestamp(int(since_ms), unit="ms")].sort_values(["o", "pair"]) if len(c) else c
    pre = c.iloc[:V2_N] if len(c) else c
    f = lambda g: f"{len(g)} · {(g.pct > 0).mean():.0%} · {g.pct.mean():+.3f} %" if len(g) else "0"
    met = len(pre) > 0 and float((pre.pct > 0).mean()) < BE_WR and float(pre.pct.mean()) < 0
    own, own_ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    rec = own.get("OFF_TRIGGER_531") if own_ok else None
    if rec:
        return L + [f"**Frozen ({rec.get('at')}): {rec.get('result')}** · {rec.get('detail')}", f"Live (never re-decides): {f(c)}", rule, ""]
    if len(pre) < V2_N:
        return L + [f"⏳ COLLECTING — {f(pre)} of {V2_N} (would be {'MET' if met else 'not met'} now).", rule, ""]
    last = pre.o.max()
    if newest is None or newest < last + pd.Timedelta(minutes=cap_minutes()):
        return L + [f"⏸ deciding deferred — the newest export is older than the 15th fill + the max hold. Now: {f(pre)}.", rule, ""]
    result = "MET — recommend WILLY OFF (operator switches it)" if met else "NOT MET — WILLY stays"
    if state_path and own_ok:
        own["OFF_TRIGGER_531"] = dict(result=result, detail=f(pre), at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
        NF.du_save_state(own, state_path)
    elif not own_ok:
        return L + [f"⚠ own state corrupt — not frozen this run. Now: {f(pre)}.", rule, ""]
    return L + [f"**{result}** · {f(pre)}", rule, ""]


if __name__ == "__main__":
    print("\n".join(run()))
