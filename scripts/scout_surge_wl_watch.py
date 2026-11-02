#!/usr/bin/env python3
"""👁 Scout — SURGE_LONG winners vs losers watch lines (pre-registered 2026-11-01, DECISION_LOG 427, operator: "register both as observe-only lines"; OBSERVE ONLY — never changes config,
no bot API, no network).

QUESTION   Two entry readings split SURGE_LONG's first 6 real fills perfectly. Do they hold on new fills? Two separate lines, each decided on its own:
           SURGE_BTC_HOT  = entry_btc_rsi ≥ 78 (a market-wide reading: a sleeve on/off switch, evidence in trigger windows) · theory: the trigger buys a BTC move that is already stretched.
           SURGE_THIN     = entry_pair_volume_24h_usd < $150M · theory: a thinner coin whipsaws through a −1.2 % stop before it runs.
EVIDENCE   (in-sample, NEVER counted) both lines mark the same fills: zone 3 on 2 windows · 0 % · −1.20 %/fill (BTW, EVAA, HAEDAL, B25) vs rest 3 · 100 % · +4.40 (STRK, AGT, GTC, master).
           AGAINST: BEAMX 11-02 (RSI 71, $196M, −0.22, before the floor, never counted) lost in "rest" on both lines · 11 of 69 complete columns split these 6 fills perfectly, as luck expects at 3 against 3; the cuts sit in the gap, chosen after seeing it (RSI 76.3 | 78.9, volume $157M | $131M).
COHORT     CLOSED SURGE_LONG fills from the orders exports (export_glob), dedup (opened_at[:19], pair) — CLOSED beats OPEN, newest export wins; operator-closed out (scout_gatekit).
           COUNTED: opened ≥ the DECISION_LOG 427 commit + 10 min. Stamp missing → UNSCORED (listed, never counted). Window = entry_surge_trigger_at, else the open time's 5-minute bar.
FROZEN BAR (CLAUDE.md expectancy bar, never re-fit), per line: at the first prefix (by open time) where the ZONE reaches N ≥ 15 on ≥ 8 windows, decided ONCE — ① zone avg ≥ rest avg, OR zone avg
           ≥ 0 → RETIRED; ② else FILTER CANDIDATE (operator decides; never automatic) iff zone WR < the frozen breakeven WR 31.5 % (SURGE_LONG, DL 287) ∧ zone mean < 0 at a window-clustered
           bootstrap P(mean < 0) ≥ 0.95 (4,000 resamples, seed 7) ∧ no single window and no single pair carries ≥ 50 % of the zone's gross loss; ③ else KEEP OBSERVING. Deferred until the newest
           export is ≥ the prefix end + the max hold. Pinned to SURGE_LONG's exit keys (a change → ⚠ STALE). A candidate ships only after the 30–50 % haircut, a before → after per batch with the
           daily compound return, and a pre-committed revert.
STATE      reports/SCOUT_SURGE_WL.json (gitignored).
Usage: venv/bin/python -I scripts/scout_surge_wl_watch.py
"""
import json
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402
import scout_b1h_negflank as NF    # noqa: E402  (bootstrap, gross-loss share, state I/O)

DL = 427
SLEEVE = "SURGE_LONG"
RSI_CUT = 78.0
VOL_CUT = 150e6
LINES = (("SURGE_BTC_HOT", "entry_btc_rsi", "ge", RSI_CUT, "BTC RSI ≥ 78"),
         ("SURGE_THIN", "entry_pair_volume_24h_usd", "lt", VOL_CUT, "pair 24 h volume < $150M"))
N_MIN, WINDOWS_MIN = 15, 8
BE_WR = 0.315                      # FROZEN: SURGE_LONG's breakeven WR, the DL 287 revert bar
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
CAP_DEFAULT = 240
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_SURGE_WL.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "entry_strategy", "status", "pnl_percentage", "close_reason", "entry_surge_trigger_at", "entry_btc_rsi", "entry_pair_volume_24h_usd")
REGISTRATION = "in-sample, never counted: both lines zone 3 on 2 windows · 0 % · −1.20 vs rest 3 · 100 % · +4.40, then BEAMX −0.22 lost in rest; 11 of 69 columns split these 6 fills perfectly, as luck expects"


def cap_minutes(path=None):
    try:
        with open(path or os.path.join(_ROOT, "trading_config.json")) as f:
            c = json.load(f)
        return float(c.get("thresholds", c).get("surge_max_hold_minutes", CAP_DEFAULT)) or float(CAP_DEFAULT)   # 0 = the global max hold: never no deferral
    except Exception:
        return float(CAP_DEFAULT)


def load(paths=None):
    """→ (closed SURGE_LONG fills in open order with their window; the newest read export's time or None)."""
    fr, newest = [], None
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if not {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            continue
        fr.append(d.assign(_m=os.path.getmtime(f)))
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "window", "pct"]), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[d.entry_strategy.astype(str) == SLEEVE].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    trig = c.entry_surge_trigger_at.astype(object).where(c.entry_surge_trigger_at.notna())
    c["window"] = trig.where(trig.notna(), c.o.dt.floor("5min").dt.strftime("%Y-%m-%dT%H:%M:%S")).astype(str)
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


def zone_of(c, col, op, cut):
    """→ Series zone / rest / unscored. PURE."""
    x = pd.to_numeric(c[col], errors="coerce")
    z = (x >= cut) if op == "ge" else (x < cut)
    return z.map({True: "zone", False: "rest"}).where(x.notna(), "unscored")


def verdict(zone, rest):
    """zone / rest = counted fills (pct, window, pair) → (state, detail). PURE."""
    n, nw = len(zone), zone.window.nunique() if len(zone) else 0
    if n < N_MIN or nw < WINDOWS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nw}/{WINDOWS_MIN} windows"
    mz, mr = float(zone.pct.mean()), (float(rest.pct.mean()) if len(rest) else float("nan"))
    wr = float((zone.pct > 0).mean())
    p = NF.p_mean_neg(zone.pct.values, zone.window.values, BOOT_N, BOOT_SEED)
    sw, sp = NF.gross_loss_share(zone.pct.values, zone.window.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nw} windows · WR {wr:.0%} (breakeven {BE_WR:.1%}) · avg {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · "
           f"P(mean<0) {(p if p is not None else float('nan')):.2f} · top window {sw:.0%} / top pair {sp:.0%} of the zone's gross loss")
    if mz >= 0 or (len(rest) and mz >= mr):
        return "RETIRED", det
    if wr < BE_WR and p is not None and p >= P_MIN and sw < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def crossing_prefix(sc):
    """scored counted fills by open time → the prefix ending where the zone first reaches N ≥ 15 on ≥ 8 windows."""
    seen, nz = set(), 0
    for k, (g, w) in enumerate(zip(sc.zone.values, sc.window.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(w)
        if nz >= N_MIN and len(seen) >= WINDOWS_MIN:
            return sc.iloc[:k]
    return None


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – |"
    return f"| {lab} | {len(g)} | {g.window.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % |"


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 427 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 SURGE_WL — SURGE_LONG winners vs losers: BTC RSI ≥ {RSI_CUT:g} · pair volume < $150M (pre-registered 2026-11-01, DECISION_LOG {DL}, OBSERVE only)", ""]
    given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="SURGE_WL watch lines")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    cap = cap_minutes()
    L += [f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if given is None else 'floor given'}) · each line decided separately · {_GK.operator_closed_line(ox)}.", ""]
    for name, col, op, cut, lab in LINES:
        s = keep.assign(zone=zone_of(keep, col, op, cut)) if len(keep) else keep.assign(zone=[])
        sc = s[s.zone != "unscored"]
        zone, rest = sc[sc.zone == "zone"], sc[sc.zone == "rest"]
        L += [f"### {name} ({lab})", "", "| Group | N | windows | WR | avg % |", "|---|---|---|---|---|",
              _row("**zone**", zone), _row("rest", rest), _row("UNSCORED (stamp missing) — never counted", s[s.zone == "unscored"]), ""]
        why = []
        rec = st.get(name) if ok else None
        if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
            rec, ok = None, False
        if not rec and ok:
            pre = crossing_prefix(sc) if len(sc) else None
            if pre is not None:
                last = pre.o.max()
                if newest is None or newest < last + pd.Timedelta(minutes=cap):
                    why.append(f"newest export < the crossing fill + {cap:.0f} min (a fill could still be open)")
                else:
                    state, det = verdict(pre[pre.zone == "zone"], pre[pre.zone == "rest"])
                    rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                               keys=[f"{a:%Y-%m-%dT%H:%M:%S}|{b}|{g}" for a, b, g in zip(pre.o, pre.pair, pre.zone)])
                    if state_path:
                        st[name] = rec
                        NF.du_save_state(st, state_path)   # saved at once: a damaged record on the next line never leaves this verdict printed but unsaved
        ls, ld = verdict(zone, rest)
        if rec and ok and state_path:
            L += [f"**Frozen verdict (crossing at fill {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})", f"Live (information only, never re-decides): {ls} — {ld}"]
        else:
            L.append(f"Now: ⏳ {ls} ({ld}).")
        if why:
            L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
        L.append("")
    if not ok and state_path:
        L.append(f"⚠ frozen state corrupt or damaged — operator restore needed ({os.path.basename(state_path)}); freezing skipped this run.")
    L += [f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}, per line): at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {WINDOWS_MIN} windows, decided ONCE — zone avg ≥ rest avg or ≥ 0 → RETIRED; {CANDIDATE} iff zone WR < {BE_WR:.1%} "
          f"∧ zone mean < 0 at window-bootstrap P(mean < 0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no window / pair ≥ {SHARE_MAX:.0%} of the zone's gross loss; else KEEP OBSERVING.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
