#!/usr/bin/env python3
"""🏦 Scout — MEGACAP_READMIT watch line (DECISION_LOG 506, operator 2026-11-09: "if we remove it, we keep collecting trades, and we can decide whether
the mkt hot is a good filter or not, or whether we need to go back and filter the top 10"). OBSERVE ONLY — never changes config, no bot API, no network.

WHY        The momentum-long mega-cap block (`long_megacap_rank_max` 10, DL 110) was switched OFF (→ 0) on 11-09 as a DECLARED OVERRIDE of its own re-admit
           gate (which had not fired: 20 refused signals 55 % · Σ −2.33 %). Its losses looked like MARKET-HOT losses: master re-admitted cohort (stack basis)
           hot 3 · 33 % · −0.45 vs not hot 7 · 71 % · +0.05; live refusals since 09-23 hot 9 · 33 % · −0.49 vs not hot 10 · 70 % · +0.20; pooled not hot
           17 · 71 % · +0.14 vs hot 12 · 33 % · −0.48. This line reads the re-admitted top-10 fills and decides ONCE.
COHORT     CLOSED full-size MOMENTUM LONG fills (*_PROBE out, operator-closed out), orders exports (Downloads + exports_archive/), dedup (opened_at[:19], pair),
           CLOSED beats OPEN, newest export wins, opened ≥ the DECISION_LOG 506 commit + 10 min. TOP10 = entry_pair_rank ≤ 10 · COMPARATOR = ranks 11–50
           in the same period. Market-hot zone = the fill's own stamp entry_mkt_up4h_pct > 0.50 (the ML_MKT_HOT cut, DL 416 / 489); stamp missing →
           UNSCORED (listed, never counted). P&L = booked pnl_percentage (leverage-invariant). Market-hot is MARKET-WIDE → days are the unit.
FROZEN BAR (DL 506, never re-fit): decided ONCE at the first prefix where TOP10 reaches N ≥ 15 on ≥ 8 UTC days AND its NOT-HOT part N ≥ 8 on ≥ 5 days —
           ① NOT-HOT top-10 WR < 61.5 % (momentum-long breakeven, stack basis) ∧ avg < 0 → RESTORE `long_megacap_rank_max` 10 — unless one pair carries
             ≥ 50 % of their gross loss → PAIR BLACKLIST candidate instead (operator decides)
           ② NOT-HOT top-10 avg ≥ 0 ∧ ≥ the comparator's not-hot avg − 0.10 → KEEP OFF (Market-hot is the right filter; the hot top-10 rows are ML_MKT_HOT
             evidence — decided by its own gate)
           ③ else → KEEP COLLECTING to TOP10 N 25, then RESTORE 10 unless ② holds by then.
EARLY STOP the first 8 counted TOP10 fills average < −0.30 %/fill (hot and not hot) → RESTORE 10 at once (paper guard).
STATE      reports/SCOUT_MEGACAP_READMIT.json (gitignored).
Usage: venv/bin/python -I scripts/scout_megacap_readmit_watch.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402
import scout_b1h_negflank as NF    # noqa: E402  (gross-loss share, state I/O)

DL = 506
FLOOR_DL = 506
RANK_TOP, RANK_CMP = 10, 50
HOT_CUT = 0.50
N_MIN, DAYS_MIN, NH_MIN, NH_DAYS = 15, 8, 8, 5
N_MAX = 25
BE_WR = 0.615                      # FROZEN 2026-11-09: momentum-long breakeven on kept fills (stack basis, DL 506 deep review)
CMP_TOL = 0.10
EARLY_N, EARLY_AVG = 8, -0.30
SHARE_MAX = 0.50
WAIT_MIN = 300                     # a momentum long closes within hours — export ≥ the crossing fill + 5 h before freezing
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_MEGACAP_READMIT.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason",
        "cell_multiplier_source", "entry_pair_rank", "entry_mkt_up4h_pct")
RESTORE, KEEP_OFF, PAIR_BL = "RESTORE long_megacap_rank_max 10", "KEEP OFF (Market-hot is the filter)", "PAIR BLACKLIST candidate (operator decides)"


def load(paths=None):
    """→ (closed full-size momentum longs with rank / zone / day, the newest export time or None)."""
    fr, read = [], []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
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
    cols = list(COLS) + ["o", "day", "rank", "zone", "pct"]
    if not fr:
        return pd.DataFrame(columns=cols), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[(d.entry_strategy.astype(str) == "MOMENTUM") & (d.direction.astype(str) == "LONG")
          & ~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    c["rank"] = pd.to_numeric(c.entry_pair_rank, errors="coerce")
    b = pd.to_numeric(c.entry_mkt_up4h_pct, errors="coerce")
    c["zone"] = np.where(b.isna(), "unscored", np.where(b > HOT_CUT, "hot", "not_hot"))
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


def verdict(top, cmp_):
    """top / cmp_ = counted scored fills (pct, day, pair, zone) → (state, detail). PURE."""
    n, nd = len(top), (top.day.nunique() if len(top) else 0)
    nh = top[top.zone == "not_hot"]
    nhn, nhd = len(nh), (nh.day.nunique() if len(nh) else 0)
    if len(top) >= EARLY_N and float(top.pct.iloc[:EARLY_N].mean()) < EARLY_AVG:
        return RESTORE + " — EARLY STOP", f"first {EARLY_N} top-10 fills avg {float(top.pct.iloc[:EARLY_N].mean()):+.3f} % < {EARLY_AVG:+.2f}"
    if n < N_MIN or nd < DAYS_MIN or nhn < NH_MIN or nhd < NH_DAYS:
        return "COLLECTING", f"top-10 N {n}/{N_MIN} on {nd}/{DAYS_MIN} days · not-hot N {nhn}/{NH_MIN} on {nhd}/{NH_DAYS} days"
    cmp_nh = cmp_[cmp_.zone == "not_hot"]
    mnh, wr = float(nh.pct.mean()), float((nh.pct > 0).mean())
    mc = float(cmp_nh.pct.mean()) if len(cmp_nh) else float("nan")
    sp = NF.gross_loss_share(nh.pct.values, nh.pair.values)
    det = (f"top-10 not-hot N {nhn} · {nhd} d · WR {wr:.0%} (breakeven {BE_WR:.1%}) · avg {mnh:+.3f} % vs ranks 11–50 not-hot {mc:+.3f} % "
           f"(N {len(cmp_nh)}) · top pair {sp:.0%} of their gross loss · hot top-10 N {len(top[top.zone == 'hot'])} avg "
           f"{(float(top[top.zone == 'hot'].pct.mean()) if (top.zone == 'hot').any() else float('nan')):+.3f} %")
    if wr < BE_WR and mnh < 0:
        return (PAIR_BL if sp >= SHARE_MAX else RESTORE), det
    if mnh >= 0 and (not len(cmp_nh) or mnh >= mc - CMP_TOL):
        return KEEP_OFF, det
    return ("KEEP COLLECTING" if n < N_MAX else RESTORE), det


def crossing_prefix(sc):
    """counted scored fills by open time → the prefix ending where the bar first becomes decidable (or the early stop fires), else None."""
    for k in range(1, len(sc) + 1):
        pre = sc.iloc[:k]
        top = pre[pre["rank"] <= RANK_TOP]
        st, _ = verdict(top, pre[(pre["rank"] > RANK_TOP) & (pre["rank"] <= RANK_CMP)])
        if st not in ("COLLECTING", "KEEP COLLECTING"):
            return pre
    return None


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % | "
            f"{pd.to_numeric(g.pnl, errors='coerce').sum():+,.0f} |")


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 506 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🏦 MEGACAP_READMIT — momentum longs on top-10 coins with the mega-cap block OFF (DECISION_LOG {DL}, OBSERVE only)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(FLOOR_DL, lag_min=10, label="MEGACAP_READMIT watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {FLOOR_DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    sc = keep[(keep.zone != "unscored") & keep["rank"].notna()] if len(keep) else keep
    top = sc[sc["rank"] <= RANK_TOP] if len(sc) else sc
    cmp_ = sc[(sc["rank"] > RANK_TOP) & (sc["rank"] <= RANK_CMP)] if len(sc) else sc
    uns = keep[(keep.zone == "unscored") | keep["rank"].isna()] if len(keep) else keep
    L += [f"Cohort: CLOSED full-size momentum longs opened ≥ {since:%Y-%m-%d %H:%M} UTC · {_GK.operator_closed_line(ox)}.", "",
          "| Group | N | days | WR | avg % | Σ$ as traded |", "|---|---|---|---|---|---|",
          _row("**top-10, not hot**", top[top.zone == "not_hot"] if len(top) else top),
          _row("top-10, market-hot", top[top.zone == "hot"] if len(top) else top),
          _row("ranks 11–50, not hot (comparator)", cmp_[cmp_.zone == "not_hot"] if len(cmp_) else cmp_),
          _row("ranks 11–50, market-hot", cmp_[cmp_.zone == "hot"] if len(cmp_) else cmp_),
          _row("UNSCORED (no rank / breadth stamp) — never counted", uns), ""]
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    rec = st.get("MEGACAP") if ok else None
    if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
        rec, ok = None, False
    why = []
    if not rec and ok and len(sc):
        pre = crossing_prefix(sc)
        if pre is not None:
            last = pre.o.max()
            if newest is None or newest < last + pd.Timedelta(minutes=WAIT_MIN):
                why.append(f"newest export < the crossing fill + {WAIT_MIN} min")
            else:
                p_top = pre[pre["rank"] <= RANK_TOP]
                state, det = verdict(p_top, pre[(pre["rank"] > RANK_TOP) & (pre["rank"] <= RANK_CMP)])
                rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
                if state_path:
                    st["MEGACAP"] = rec
                    NF.du_save_state(st, state_path)
    live_state, live_det = verdict(top, cmp_)
    if rec and ok and state_path:
        L.append(f"**Frozen verdict ({rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
        L.append(f"Live (information only): {live_state} — {live_det}")
    else:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    if why:
        L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += ["", f"FROZEN bar (DL {DL}): at top-10 N ≥ {N_MIN} on ≥ {DAYS_MIN} days and its not-hot part N ≥ {NH_MIN} on ≥ {NH_DAYS} days, decided ONCE — "
          f"not-hot WR < {BE_WR:.1%} ∧ avg < 0 → {RESTORE} (one pair ≥ {SHARE_MAX:.0%} of their gross loss → {PAIR_BL}) · not-hot avg ≥ 0 ∧ ≥ ranks 11–50 "
          f"not-hot − {CMP_TOL:.2f} → {KEEP_OFF} · else keep collecting to N {N_MAX}, then restore. Early stop: first {EARLY_N} top-10 avg < {EARLY_AVG:+.2f} % → restore.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
