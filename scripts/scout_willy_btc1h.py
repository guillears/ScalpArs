#!/usr/bin/env python3
"""🧭🎲 Scout — WILLY_BTC1H observe line (pre-registered 2026-10-12, DECISION_LOG 275; OBSERVE ONLY — never changes config, no bot API).

HYPOTHESIS     FRENZY_WILLY does worse when BTC's 1-hour EMA20 slope is ≤ 0 at entry. Market-wide variable → a WILLY on/off switch, not
               a trade filter (CLAUDE.md WINDOW-UNITS RULE): evidence counts in DAY units (same-day fills = one observation).
FEATURE        EXACTLY scripts/study_b19_port_conc_regime.btc_feats: BTCUSDT 5m closes (bar close time = open_time + 5 min; caches
               k5m_full + b19_5m + negflank2d_ext/5m + scout_dailyup, the last one kept fresh by NEG_DAILYUP's fetcher — reused here, cache
               first) resampled to 1 h bars (label right, closed right, last close); EMA20 = ewm(span=20, adjust=False) on the 1 h closes;
               slope = (EMA20 of the last CLOSED 1 h bar at the entry ÷ the previous one − 1) × 100 — no look-ahead (a bar labelled H only
               holds 5m bars closed ≤ H ≤ entry). UNSCORED (never counted on either side) when: fewer than 100 one-hour bars up to that bar
               (EMA warm-up) · the last closed bar is not the entry's hour or is incomplete, or its predecessor is missing (BTC cache behind
               / gap — such a fill holds the freeze for 3 days while the cache can still catch up).
EVIDENCE       reports/B19_WILLY_BTC1H.txt (scripts/study_b19_willy_btc1h.py), yr5 live-reachable WILLY cohort, exit +1 / −4:
               kept (slope > 0) 286 fills · 142 days · 83 % · +0.185 %/fill (H1 +0.188 / H2 +0.181);
               blocked (slope ≤ 0) 246 · 145 days · 77 % · −0.098 (H1 −0.100 / H2 −0.097), P(mean < 0) 0.76;
               found in a 16-test time / regime screen, Bonferroni 0.24 → OBSERVE ONLY (CLAUDE.md OBSERVE-FIRST RULE);
               B19: all 14 WILLY fills had slope > 0, so 0 blocked.
COHORT         CLOSED FRENZY_WILLY LONG fills, non-MANUAL (MANUAL_* closes listed, excluded), from the ~/Downloads orders exports
               (scout_willy_timecap's reader / column set and scout_frenzy_willy_exit's preparation), dedup (opened_at[:19], pair,
               entry_strategy) — never id; a CLOSED row beats an OPEN one. COUNTED (DECISION_LOG 281): fills opened ≥ the DECISION_LOG
               281 deploy (scout_willy_timecap.dl281_floor = the "(DECISION_LOG 281)" commit + 10 min; commit not found → its placeholder
               pin 2026-10-12 12:00 UTC + 10 min), so the counted cohort runs ONE live exit (+1.25 / −4 / 90); it was 2026-10-12 00:00
               before (0 forward fills then, nothing frozen). Fills from the WILLY_TIMECAP deploy floor (the "(DECISION_LOG 251)" commit) to
               that deploy = the B19 "pre-registration context" line — shown, NEVER counted.
PRICING        every fill re-walked under WILLY's REGISTERED exit, PINNED since DECISION_LOG 356 (willy_rule → +1.25 / −4 / 90 min, the
               DECISION_LOG 281 live exit; never read from the config again — V2 per-order targets / floor never enter the walk; V2 runner /
               floor fills stay in the groups, walked like every fill, and are kept out of the parity line) with the
               ticks-first hybrid walker (scout_frenzy_willy_exit.load_path / evaluate, DECISION_LOG 272; tick ruler, bot accounting).
               Cache only (WILLY_TIMECAP's fetch pass fills the tick / 1m caches for the same fills). Walker parity vs the actual exit
               printed. Unpriced fills are listed and hold the freeze while their data is still due; a fill with a 1m part (provisional)
               holds it while its tick archive is pending. Group stats use the WALKED %.
GROUPS         kept (slope > 0) / blocked (slope ≤ 0) / unscored (never counted on either side).
FROZEN BAR     (pre-registered 2026-10-12, DECISION_LOG 275 — never changed later; written in the report output too)
               At the first prefix (by open time) where the BLOCKED side reaches N ≥ 15 on ≥ 8 distinct UTC days, decided ONCE:
               ① blocked avg ≥ kept avg → RETIRED;
               ② else ARM CANDIDATE (operator decides; never automatic) iff all of: blocked mean < 0 with day-clustered bootstrap
                  P(mean < 0) ≥ 0.95 (4,000 resamples, seed 7) ∧ no single day and no single pair carries ≥ 50 % of the blocked side's
                  gross loss;
               ③ else KEEP OBSERVING.
               Deferred while any fill ≤ the prefix end is OPEN, unpriced, or provisional (1m part with ticks still due).
               Counted from the DECISION_LOG 281 deploy (one live exit: today's config, +1.25 / −4 / 90).
               (An empty kept side cannot trigger ①; ② still applies.)
               The frozen record keeps n_unscored_le_prefix (warm-up / lag / gap) so the excluded fills stay on record.
SUB-READ       🩸 DIP_SUB (pre-registered 2026-10-12, DECISION_LOG 279; OBSERVE only, operator may approve; own state key "dip_sub"):
               "BTC 1 h rising ∧ WILLY signal red candle ≤ −1.16 %". Among the COUNTED (from the DECISION_LOG 281 deploy, one live exit),
               priced, BTC-rising (slope > 0) fills: deep =
               entry_frenzy_bar_ret_pct ≤ −1.16 (inclusive) · shallow = > −1.16 · missing candle stamp → unscored (never counted).
               Evidence reports/B19_WILLY_2D_BTC1H.txt: year — kept (rising ∧ deep) 119 · 88 % · +0.471 %/fill vs the rest −0.066,
               H1 +0.51 / H2 +0.42; family-wise p 0.23 over 169 tests → NOT a survivor; B19 contradicts it (7 deep-dip fills +0.30 vs 7
               shallow +0.52).
               FROZEN BAR: at the first prefix (by open time) where EACH side reaches N ≥ 15 on ≥ 8 distinct UTC days, decided ONCE:
               ① deep avg ≤ shallow avg → RETIRED; ② else CANDIDATE (operator decides; never automatic) iff day-clustered bootstrap
               P(deep mean − shallow mean > 0) ≥ 0.95 (4,000 resamples, seed 7) ∧ no single day and no single pair carries ≥ 50 % of the
               deep side's gross gain; ③ else KEEP OBSERVING. Same deferral as the main bar.
               State reports/SCOUT_WILLY_BTC1H.json (gitignored) · rows reports/SCOUT_WILLY_BTC1H.csv.
Usage: venv/bin/python scripts/scout_willy_btc1h.py [--selftest]
"""
import os
import os as _os_eg, sys as _sys_eg; _d_eg = _os_eg.path.dirname(_os_eg.path.abspath(__file__)); _d_eg in _sys_eg.path or _sys_eg.path.insert(0, _d_eg)
import export_glob as _EG   # 📦 (DECISION_LOG 294) paper + post-go-live LIVE exports
import tempfile
import time

import numpy as np
import pandas as pd
import os as _os_gk, sys as _sys_gk; _sys_gk.path.insert(0, _os_gk.path.dirname(_os_gk.path.abspath(__file__))) if _os_gk.path.dirname(_os_gk.path.abspath(__file__)) not in _sys_gk.path else None
import scout_gatekit as _GK   # noqa: E402  🧰 (DECISION_LOG 331) operator-closed fills (MANUAL / MANUAL_* / EXTERNAL_CLOSE) out of every gate

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (BTC 5m cache + rate-limited fetcher, bootstrap, loss share, freeze hold, state I/O)
import scout_frenzy_willy_exit as FWX               # noqa: E402  (the 272 hybrid walker, thresholds namespace, order prep)
import scout_willy_hold as WH                       # noqa: E402  (WILLY's live exit from trading_config.json)
import willy_exit_era as WEra                       # noqa: E402  🎲² (DECISION_LOG 356) pinned old exit + exit-era split / header

WT = FWX.WT
# FROZEN 2026-10-12 (DECISION_LOG 275) — never re-fit
N_MIN, DAYS_MIN = 15, 8
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
EMA_SPAN, WARM_BARS = 20, 100                       # EMA20 on 1 h closes; ≥ 100 one-hour bars up to the scored bar, else UNSCORED
COUNT_FROM = "the DECISION_LOG 281 deploy"          # (281) was COUNT_START = "2026-10-12 00:00"; now one live exit per cohort


def count_start():
    """the counted floor = scout_willy_timecap.dl281_floor() (the "(DECISION_LOG 281)" commit + 10 min, or its placeholder pin + 10 min)."""
    return WT.dl281_floor()


STRATEGY = "FRENZY_WILLY"
H_MS, M5_MS = 3_600_000, 300_000
LAG_PENDING_MS = 3 * 86_400_000                     # a fill unscored because the BTC cache is behind holds the freeze ≤ 3 days
PROVISIONAL = ("1m", "ticks+1m")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, "reports", "backtest_cache")
BTC_FILES = (("k5m_full", "BTCUSDT.csv"), ("b19_5m", "BTCUSDT.csv"), ("negflank2d_ext", "5m", "BTCUSDT.csv"), ("scout_dailyup", "BTCUSDT_5m.csv"))
STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_BTC1H.json")
ROWS_CSV = os.path.join(_ROOT, "reports", "SCOUT_WILLY_BTC1H.csv")
EVIDENCE = ("yr5 live-reachable WILLY cohort, +1 / −4 (reports/B19_WILLY_BTC1H.txt): kept (slope > 0) 286 fills · 142 days · 83 % · +0.185 "
            "(H1 +0.188 / H2 +0.181) · blocked (slope ≤ 0) 246 · 145 days · 77 % · −0.098 (H1 −0.100 / H2 −0.097), P(mean < 0) 0.76 · found "
            "in a 16-test time / regime screen, Bonferroni 0.24 → OBSERVE ONLY · market-wide → DAY units, a WILLY on/off switch · B19: all 14 "
            "WILLY fills had slope > 0 (0 blocked)")
BAR = (f"FROZEN bar (pre-registered 2026-10-12, DECISION_LOG 275): at the first prefix (by open time) where the BLOCKED side reaches N ≥ "
       f"{N_MIN} on ≥ {DAYS_MIN} distinct UTC days, decided ONCE — ① blocked avg ≥ kept avg → RETIRED; ② else ARM CANDIDATE (operator "
       f"decides; never automatic) iff blocked mean < 0 with day-clustered bootstrap P(mean < 0) ≥ {P_MIN:.2f} ({BOOT_N:,} resamples, seed "
       f"{BOOT_SEED}) ∧ no single day and no single pair carries ≥ {SHARE_MAX * 100:.0f} % of the blocked side's gross loss; ③ else KEEP "
       f"OBSERVING. Deferred while any fill ≤ the prefix end is OPEN, unpriced, or provisional (1m part with ticks still due). "
       f"Counted from the DECISION_LOG 281 deploy (one live exit: today's config, +1.25 / −4 / 90).")
ARM_STATE = "ARM CANDIDATE (operator decides; never automatic)"
# 🩸 DIP_SUB — FROZEN 2026-10-12 (DECISION_LOG 279) — never re-fit
DIP_COL, DIP_CUT, SUB_KEY = "entry_frenzy_bar_ret_pct", -1.16, "dip_sub"
SUB_STATE = "CANDIDATE (operator decides; never automatic)"
SUB_EVIDENCE = ("reports/B19_WILLY_2D_BTC1H.txt: year — kept (BTC rising ∧ red candle ≤ −1.16 %) 119 · 88 % · +0.471 vs blocked −0.066, "
                "H1 +0.51 / H2 +0.42 · family-wise p 0.23 (169 tests) → not a survivor · B19 contradicts it (7 deep-dip fills +0.30 vs 7 "
                "shallow +0.52)")
SUB_BAR = (f"DIP_SUB frozen bar (pre-registered 2026-10-12, DECISION_LOG 279): among the BTC-rising fills (slope > 0), deep = WILLY signal "
           f"red candle ({DIP_COL}) ≤ {DIP_CUT:g} % vs shallow = > {DIP_CUT:g} %, missing stamp → unscored; at the first prefix (by open "
           f"time) where EACH side reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} distinct UTC days, decided ONCE — ① deep avg ≤ shallow avg → "
           f"RETIRED; ② else {SUB_STATE} iff day-clustered bootstrap P(deep mean − shallow mean > 0) ≥ {P_MIN:.2f} ({BOOT_N:,} resamples, "
           f"seed {BOOT_SEED}) ∧ no single day and no single pair carries ≥ {SHARE_MAX * 100:.0f} % of the deep side's gross gain; ③ else "
           f"KEEP OBSERVING. Same deferral as the main bar. Counted from the DECISION_LOG 281 deploy (one live exit: today's config, "
           f"+1.25 / −4 / 90).")


# ─────────────────────────── BTC 1 h EMA20 slope ───────────────────────────
def load_btc5m():
    """the union of the cached BTCUSDT 5m closes (open_time, c) — study order, later files win on a duplicate bar."""
    fr = [NF._read_k(os.path.join(_CACHE, *f)) for f in BTC_FILES]
    fr = [f for f in fr if len(f)]
    if not fr:
        return pd.DataFrame({"open_time": pd.Series(dtype="int64"), "c": pd.Series(dtype=float)})
    return pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)


def hourly(m5):
    """→ DataFrame(H = 1 h bar close ms, c, e20, full) — study btc_feats: 5m closes at t_close = open_time + 5 min, resampled 1 h (label
    right, closed right, last), dropna, EMA20 adjust=False. full = the bar's last 5m close time == its label (a complete hour)."""
    if not len(m5):
        return pd.DataFrame({"H": pd.Series(dtype="int64"), "c": pd.Series(dtype=float), "e20": pd.Series(dtype=float), "full": pd.Series(dtype=bool)})
    tc = m5.open_time.astype("int64").values + M5_MS
    idx = pd.to_datetime(tc, unit="ms")
    h = pd.Series(m5.c.astype(float).values, index=idx).resample("1h", label="right", closed="right").last().dropna()
    last_tc = pd.Series(tc, index=idx).resample("1h", label="right", closed="right").max().reindex(h.index)
    e20 = h.ewm(span=EMA_SPAN, adjust=False).mean()
    H = h.index.values.astype("datetime64[ms]").astype("int64")
    return pd.DataFrame(dict(H=H, c=h.values, e20=e20.values, full=(last_tc.values == H)))


def slope_at(hb, t_ms):
    """→ (slope %, status) per entry ms. status 'ok' / 'warmup' (< 100 one-hour bars up to the scored bar) / 'lag' (the last closed bar is
    not the entry's own hour, or is incomplete — BTC cache behind) / 'gap' (the previous hour is missing)."""
    t_ms = np.asarray(t_ms, dtype="int64")
    H = hb.H.values.astype("int64") if len(hb) else np.array([], dtype="int64")
    out_s, out_st = [], []
    for t in t_ms:
        k = int(np.searchsorted(H, t, side="right")) - 1
        if k < 0:
            out_s.append(np.nan); out_st.append("lag" if not len(H) or H[-1] < t else "warmup")
            continue
        if H[k] != (t // H_MS) * H_MS or not bool(hb.full.values[k]):
            out_s.append(np.nan); out_st.append("lag")
            continue
        if k + 1 < WARM_BARS:
            out_s.append(np.nan); out_st.append("warmup")
            continue
        if H[k - 1] != H[k] - H_MS:
            out_s.append(np.nan); out_st.append("gap")
            continue
        e = hb.e20.values
        out_s.append((e[k] / e[k - 1] - 1) * 100); out_st.append("ok")
    return np.array(out_s, dtype=float), out_st


def group(slope):
    """FROZEN: slope > 0 → kept · slope ≤ 0 → blocked · NaN → unscored."""
    if slope is None or pd.isna(slope):
        return "unscored"
    return "kept" if slope > 0 else "blocked"


# ─────────────────────────── orders ───────────────────────────
def load_orders(orders=None):
    """→ (closed FRENZY_WILLY fills prepared, [(opened_at, 'OPEN <pair>')]) — WILLY_TIMECAP's export reader, dedup (opened_at[:19], pair,
    entry_strategy) (CLOSED beats OPEN; never id), FRENZY_WILLY_EXIT's preparation."""
    if orders is None:
        fr = _read_exports()
        raw = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable") if fr else pd.DataFrame(columns=list(COLS))
    else:
        raw = orders
    o = raw.reindex(columns=list(dict.fromkeys(list(COLS) + list(raw.columns)))).copy()
    if not len(o):
        return FWX._prep(o).assign(bar_ret=pd.Series(dtype=float)), []
    o["_k"] = o.opened_at.astype(str).str[:19]
    o["_c"] = o.status.astype(str).str.upper().eq("CLOSED")
    o = o.sort_values("_c", kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    o = o[(o.direction.astype(str) == "LONG") & (o.entry_strategy.astype(str) == STRATEGY)]
    t = pd.to_datetime(o.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    op = o.status.astype(str).str.upper().eq("OPEN")
    opens = [(a, f"OPEN {p}") for a, p in zip(t[op], o.pair[op].astype(str)) if pd.notna(a)]
    cl = o[o.status.astype(str).str.upper() == "CLOSED"]
    d = FWX._prep(cl)
    br = pd.DataFrame({"ts": pd.to_datetime(cl.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce").values,
                       "pair": cl.pair.astype(str).values, "bar_ret": pd.to_numeric(cl[DIP_COL], errors="coerce").values,
                       WEra.TP_COL: pd.to_numeric(cl[WEra.TP_COL], errors="coerce").values})   # DL 356: the fill's stored V2 target (NaN = pre-V2)
    d = d.drop(columns=[WEra.TP_COL], errors="ignore")
    d = (d.merge(br.drop_duplicates(["ts", "pair"]), on=["ts", "pair"], how="left") if len(d)
         else d.assign(bar_ret=pd.Series(dtype=float), **{WEra.TP_COL: pd.Series(dtype=float)}))
    return d, opens


COLS = tuple(dict.fromkeys(tuple(WT.COLS) + (DIP_COL, "close_reason") + tuple(WEra.V2_COLS)))   # DL 356: the V2 stamps ride the reader


def _read_exports():
    """WILLY_TIMECAP's export reader (same glob, same required columns) plus the candle stamp column the DIP_SUB read needs."""
    import glob
    fr = []
    for f in _EG.glob_exports(WT.EXPORT_GLOB):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    return fr


# ─────────────────────────── pricing ───────────────────────────
# 🔒 DECISION_LOG 356 (WILLY exit V2): the frozen DL 275 / 279 bars were registered on ONE live exit, +1.25 / −4 / 90 (DL 281; the walker
# never modelled the DL 297 late TP). That exit is PINNED here — never read from trading_config.json / services.frenzy any more, so a later
# config change (or V2's per-order targets) can never move the frozen walk. (Was: scout_willy_hold.exit_rule('FRENZY_WILLY') on the config.)
PINNED_RULE = (float(WEra.OLD_TP), float(WEra.OLD_STOP), int(WEra.OLD_HOLD))


def willy_rule(th=None):
    """the REGISTERED WILLY exit (tp %, stop % positive, cap min) = PINNED_RULE (+1.25 / −4 / 90) — `th` is accepted for call-compat and
    IGNORED (DECISION_LOG 356: a frozen gate keeps the exit it was registered with)."""
    return PINNED_RULE


def _v2ch(c):
    """DL 356 → boolean array: V2 fills whose live exit differs from the pinned one (runner target > +1.25 or a WILLY_FLOOR close)."""
    if not len(c):
        return np.zeros(0, dtype=bool)
    x = c if "close_reason" in c.columns else c.assign(close_reason=c.why)
    return WEra.masks(x)[1].values.astype(bool)


def price(r, rule, tcache=None):
    """→ dict(walk, walk_why, walk_min, src, par_ok) — the 272 hybrid walk from the fill's own entry; walk NaN (src None) when undecided."""
    tp, st, cap = rule
    P = FWX.load_path(r.pair, int(r.te), float(r.E), cap, tcache)
    ev = FWX.evaluate(P, tp, st, cap)
    if ev["why"] == "UNRES":
        return dict(walk=np.nan, walk_why="UNRES", walk_min=np.nan, src=None, par_ok=False)
    ok, _ = WT.parity(r.why, r.pct, {"v": {"LIVE": ev}})
    src = "ticks" if (P["n_tick"] > 0 and not ev["on_bar"]) else FWX.path_src(P)     # decided on a tick print = final
    return dict(walk=float(ev["pct"]), walk_why=ev["why"], walk_min=float(ev["xmin"]), src=src, par_ok=bool(ok))


def pending(src, ticks_pend, page_due):
    """unpriced → holds while its data is still due (a tick archive can arrive or its 1m page is not fetched yet); a 1m part (provisional)
    → holds while its tick archive can arrive; decided on ticks = final."""
    if not isinstance(src, str):
        return bool(ticks_pend or page_due)
    return bool(src in PROVISIONAL and ticks_pend)


# ─────────────────────────── verdict / freeze ───────────────────────────
def verdict(kept, blocked):
    """kept / blocked = priced, scored fills (walk, day, pair) → (state, detail)."""
    n, nd = len(blocked), blocked.day.nunique() if len(blocked) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"blocked N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mb = float(blocked.walk.mean())
    mk = float(kept.walk.mean()) if len(kept) else float("nan")
    p = NF.p_mean_neg(blocked.walk.values, blocked.day.values, BOOT_N, BOOT_SEED)
    sd = NF.gross_loss_share(blocked.walk.values, blocked.day.values)
    sp = NF.gross_loss_share(blocked.walk.values, blocked.pair.values)
    det = (f"blocked N {n} · {nd} d · WR {(blocked.walk > 0).mean() * 100:.0f} % · avg {mb:+.3f} % vs kept {mk:+.3f} % (N {len(kept)}) · "
           f"P(mean<0) {(p if p is not None else float('nan')):.2f} · top day {sd * 100:.0f} % / top pair {sp * 100:.0f} % of the blocked gross loss")
    if len(kept) and mb >= mk:
        return "RETIRED", det
    if mb < 0 and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return ARM_STATE, det
    return "KEEP OBSERVING", det


def crossing_prefix(sc):
    """priced scored COUNTED fills by (open time, pair) → the prefix ending at the blocked fill where blocked first reaches N ≥ 15 on ≥ 8 days."""
    z = sc.sort_values(["ts", "pair"], kind="stable")
    seen, nb = set(), 0
    for k, (g, d) in enumerate(zip(z.grp.values, z.day.values), 1):
        if g != "blocked":
            continue
        nb += 1
        seen.add(d)
        if nb >= N_MIN and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def _uns_rec(uns, last):
    """(L3) counted fills left UNSCORED that were opened ≤ the prefix end, by status — kept on record in the frozen verdict."""
    u = uns[uns.ts <= last] if uns is not None and len(uns) else None
    if u is None or not len(u):
        return dict(total=0)
    col = "slope_status" if "slope_status" in u else None
    vc = u[col].value_counts() if col else pd.Series(dtype=int)
    return dict(total=int(len(u)), **{str(k): int(v) for k, v in vc.items()})


def freeze(st, sc, now_iso, hold=None, why=None, uns=None):
    """ONCE at the first crossing; never recomputed. uns = counted fills unscored on the slope (recorded, never counted). → (st, changed)."""
    if "first" in st:
        return st, False
    pre = crossing_prefix(sc)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if NF.freeze_hold(last, hold, why):
        return st, False
    state, det = verdict(pre[pre.grp == "kept"], pre[pre.grp == "blocked"])
    st["first"] = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso,
                       n=int((pre.grp == "blocked").sum()), days=int(pre[pre.grp == "blocked"].day.nunique()), n_kept=int((pre.grp == "kept").sum()),
                       n_unscored_le_prefix=_uns_rec(uns, last),
                       keys=[f"{a}|{b}|{g}" for a, b, g in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair, pre.grp)])
    return st, True


# ─────────────────────────── 🩸 DIP_SUB (DECISION_LOG 279) ───────────────────────────
def sub_group(slope, bar_ret):
    """FROZEN: only BTC-rising fills (slope > 0) are in the read → deep (bar_ret ≤ −1.16) / shallow / unscored (no stamp); else None."""
    if group(slope) != "kept":
        return None
    if bar_ret is None or pd.isna(bar_ret):
        return "unscored"
    return "deep" if float(bar_ret) <= DIP_CUT else "shallow"


def p_diff_pos(deep, shallow, n=BOOT_N, seed=BOOT_SEED):
    """day-clustered bootstrap over the UNION of both sides' days → P(deep mean − shallow mean > 0); a resample missing a side counts as
    not > 0 (conservative)."""
    days = sorted(set(deep.day) | set(shallow.day))
    if len(days) < 2:
        return None
    pos = {d: i for i, d in enumerate(days)}
    sd, cd, ss, cs = (np.zeros(len(days)) for _ in range(4))
    for d, v in zip(deep.day, deep.walk.astype(float)):
        sd[pos[d]] += v; cd[pos[d]] += 1
    for d, v in zip(shallow.day, shallow.walk.astype(float)):
        ss[pos[d]] += v; cs[pos[d]] += 1
    idx = np.random.default_rng(seed).integers(0, len(days), size=(n, len(days)))
    a, b = cd[idx].sum(1), cs[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = sd[idx].sum(1) / a - ss[idx].sum(1) / b
    return float(((a > 0) & (b > 0) & (diff > 0)).mean())


def sub_verdict(deep, shallow):
    nd_, ns_ = (deep.day.nunique() if len(deep) else 0), (shallow.day.nunique() if len(shallow) else 0)
    if len(deep) < N_MIN or nd_ < DAYS_MIN or len(shallow) < N_MIN or ns_ < DAYS_MIN:
        return "COLLECTING", f"deep N {len(deep)}/{N_MIN} · {nd_}/{DAYS_MIN} d · shallow N {len(shallow)}/{N_MIN} · {ns_}/{DAYS_MIN} d"
    md, ms = float(deep.walk.mean()), float(shallow.walk.mean())
    p = p_diff_pos(deep, shallow)
    gd = NF.gross_loss_share(-deep.walk.values.astype(float), deep.day.values)     # share of the deep side's GROSS GAIN
    gp = NF.gross_loss_share(-deep.walk.values.astype(float), deep.pair.values)
    det = (f"deep N {len(deep)} · {nd_} d · WR {(deep.walk > 0).mean() * 100:.0f} % · avg {md:+.3f} % vs shallow {ms:+.3f} % (N {len(shallow)} · "
           f"{ns_} d) · P(deep − shallow > 0) {(p if p is not None else float('nan')):.2f} · top day {gd * 100:.0f} % / top pair {gp * 100:.0f} % "
           f"of the deep gross gain")
    if md <= ms:
        return "RETIRED", det
    if p is not None and p >= P_MIN and gd < SHARE_MAX and gp < SHARE_MAX:
        return SUB_STATE, det
    return "KEEP OBSERVING", det


def sub_prefix(ss):
    """deep / shallow fills by (open time, pair) → the prefix ending where BOTH sides first reach N ≥ 15 on ≥ 8 days."""
    z = ss.sort_values(["ts", "pair"], kind="stable")
    n = {"deep": 0, "shallow": 0}
    seen = {"deep": set(), "shallow": set()}
    for k, (g, d) in enumerate(zip(z.sgrp.values, z.day.values), 1):
        if g not in n:
            continue
        n[g] += 1
        seen[g].add(d)
        if all(n[x] >= N_MIN and len(seen[x]) >= DAYS_MIN for x in n):
            return z.iloc[:k]
    return None


def sub_freeze(st, ss, now_iso, hold=None, why=None, uns=None):
    """ONCE under st['dip_sub']['first']; never recomputed. → (st, changed)."""
    s0 = st.get(SUB_KEY, {})
    if "first" in s0:
        return st, False
    pre = sub_prefix(ss)
    if pre is None:
        return st, False
    last = pre.ts.max()
    w = [] if why is not None else None
    if NF.freeze_hold(last, hold, w):
        if why is not None:
            why += [f"DIP_SUB: {x}" for x in w]
        return st, False
    state, det = sub_verdict(pre[pre.sgrp == "deep"], pre[pre.sgrp == "shallow"])
    u = uns[uns.ts <= last] if uns is not None and len(uns) else []
    st[SUB_KEY] = {"first": dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso,
                                 n_deep=int((pre.sgrp == "deep").sum()), n_shallow=int((pre.sgrp == "shallow").sum()),
                                 n_unscored_le_prefix=dict(total=int(len(u)), no_candle_stamp=int(len(u))),
                                 keys=[f"{a}|{b}|{g}" for a, b, g in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair, pre.sgrp)])}
    return st, True


def hold_list(F, opens, now_ms):
    """fills that can still change a prefix: OPEN rows, unpriced / provisional fills while their data is due, and a fill unscored
    only because the BTC cache is behind (≤ 3 days after its open)."""
    hold = list(opens) + [(r.ts, f"{'unpriced' if not isinstance(r.src, str) else 'provisional'} {r.pair}") for r in F.itertuples() if r.pend]
    hold += [(r.ts, f"BTC slope pending {r.pair}") for r in F.itertuples()
             if r.slope_status == "lag" and now_ms - r.ts.value // 1_000_000 < LAG_PENDING_MS]
    return hold


# ─────────────────────────── rendering ───────────────────────────
HDR = ["| Group | N | days | pairs | WR | avg % (walked) | Σ % | Σ$ at the fill's size |", "|---|---|---|---|---|---|---|---|"]


def _row(lab, g):
    g = g[g.walk.notna()] if len(g) else g
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {g.pair.nunique()} | {(g.walk > 0).mean() * 100:.0f} % | {g.walk.mean():+.3f} % | "
            f"{g.walk.sum():+.2f} % | {(g.walk * g.notional / 100).sum():+,.0f} |")


def _brief(g):
    g = g[g.walk.notna()] if len(g) else g
    if not len(g):
        return "0"
    return f"{len(g)} · {g.day.nunique()} d · {(g.walk > 0).mean() * 100:.0f} % · {g.walk.mean():+.3f} · Σ {g.walk.sum():+.2f}"


def _store(rows, path):
    """merge into the CSV store (key opened_at|pair; this run's row wins) — every fill ever seen stays even after its export is gone."""
    cols = ["k", "pair", "day", "era", "slope", "slope_status", "grp", "live_pct", "live_why", "walk", "walk_why", "walk_min", "src", "par_ok", "notional", "bar_ret", "sgrp"]
    new = rows.reindex(columns=cols)
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            new = pd.concat([old.reindex(columns=cols), new], ignore_index=True).drop_duplicates(["k", "pair"], keep="last")
        except Exception:
            pass
    tmp = f"{path}.{os.getpid()}.tmp"
    new.sort_values(["k", "pair"]).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _fmt_uns(d):
    if not isinstance(d, dict):
        return "?"
    rest = " / ".join(f"{k} {v}" for k, v in d.items() if k != "total")
    return f"{d.get('total', 0)}" + (f" ({rest})" if rest else "")


# ─────────────────────────── run ───────────────────────────
def run(now_ms=None, orders=None, fetch=True, state_path=None, rows_path=None, open_ts=None, th=None, floor=None, cstart=None, rule=None):
    """→ markdown lines (the caller wraps it in its own try). Ticks / 1m: cache only (WILLY_TIMECAP's fetch pass). BTC 5m: cache first,
    then NEG_DAILYUP's rate-limited fetcher (shared cooldown, used-weight stop at 900, 418/429 stop) only when fetch and the cache is
    behind the newest counted fill. DL 356: the walk uses PINNED_RULE (th is ignored for the exit); `rule` = a synthetic exit for the
    hermetic selftest harness ONLY — production callers never pass it."""
    now_ms = now_ms or int(time.time() * 1000)
    state_path, rows_path = state_path or STATE, rows_path or ROWS_CSV
    WT._K1.clear()
    rule = willy_rule(th) if rule is None else (float(rule[0]), (abs(float(rule[1])) if rule[1] else None), int(rule[2]))
    o, opens = load_orders(orders)
    if open_ts is not None:
        opens = list(open_ts)
    floor = WT.deploy_ts() if floor is None else pd.Timestamp(floor)
    cstart = count_start() if cstart is None else pd.Timestamp(cstart)
    o = o[o.ts >= floor].reset_index(drop=True)
    opens = [x for x in opens if pd.notna(x[0]) and x[0] >= cstart]             # only a counted-era OPEN row can hold the freeze
    man = o[_GK.reason_mask(o.why)]
    c = o[~_GK.reason_mask(o.why)].reset_index(drop=True)
    c["era"] = np.where(c.ts >= cstart, "forward", "context")
    notes = []
    fw = c[c.era == "forward"]
    if fetch and len(fw):
        need = int(fw.ts.max().value // 10**6)
        m5 = load_btc5m()
        if not len(m5) or int(m5.open_time.max()) + M5_MS < need:
            try:
                _, _, notes = NF.load_btc(need, now_ms, fetch=True)
            except Exception as e:
                notes = [f"BTC 5m fetch failed ({str(e)[:80]})"]
    m5c = load_btc5m()
    hb = hourly(m5c)
    ts_ms = c.ts.values.astype("datetime64[ms]").astype("int64") if len(c) else np.array([], dtype="int64")
    sl, sst = slope_at(hb, ts_ms)
    c["slope"], c["slope_status"] = sl, sst
    c["grp"] = [group(x) for x in c.slope]
    c["sgrp"] = [sub_group(a, b) for a, b in zip(c.slope, c.bar_ret)] if len(c) else []
    tc = {}
    pr = [price(r, rule, tc) for r in c.itertuples()]
    for k in ("walk", "walk_why", "walk_min", "src", "par_ok"):
        c[k] = [x[k] for x in pr] if pr else []
    done = WT.load_attempts()["done"]
    c["pend"] = [pending(s, WT.ticks_pending(r, rule[2], done, now_ms, tc), WT._kk(r.pair, r.te, WT._span(rule[2])) not in done)
                 for s, r in zip(c.src, c.itertuples())]
    if len(c):
        _store(c.assign(k=c.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), live_pct=c.pct, live_why=c.why), rows_path)
    F, X = c[c.era == "forward"], c[c.era == "context"]
    priced = F[F.walk.notna()]
    sc = priced[priced.grp != "unscored"]
    kept, blocked, uns = sc[sc.grp == "kept"], sc[sc.grp == "blocked"], F[F.grp == "unscored"]
    ss = priced[priced.sgrp.isin(["deep", "shallow"])] if len(priced) else priced.assign(sgrp=pd.Series(dtype=object))
    deep, shallow, suns = ss[ss.sgrp == "deep"], ss[ss.sgrp == "shallow"], F[F.sgrp == "unscored"]
    why = []
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok:
        hold = hold_list(F, opens, now_ms)
        now_iso = pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC")
        st, ch1 = freeze(st, sc, now_iso, hold, why, uns=uns)
        st, ch2 = sub_freeze(st, ss, now_iso, hold, why, uns=suns)
        if ch1 or ch2:
            NF.du_save_state(st, state_path)
    live_state, live_det = verdict(kept, blocked)
    lab = f"+{rule[0]:g} / " + (f"−{rule[1]:g}" if rule[1] else "no stop") + f" / {rule[2]:g} min"
    L = ["## 🧭🎲 WILLY_BTC1H — FRENZY_WILLY vs BTC 1 h EMA20 slope at entry (pre-registered 2026-10-12, DECISION_LOG 275, OBSERVE only)", "",
         f"Split (frozen): BTC 1 h EMA20 slope of the last CLOSED 1 h bar at entry (5m closes resampled on the hour, no look-ahead; < "
         f"{WARM_BARS} h warm-up or BTC cache behind → UNSCORED, never counted) > 0 → kept, ≤ 0 → blocked. Cohort: CLOSED non-MANUAL "
         f"FRENZY_WILLY fills opened ≥ {cstart:%Y-%m-%d %H:%M} UTC ({COUNT_FROM} + 10 min: one live exit), every fill re-walked under WILLY's registered exit ({lab}, pinned — DECISION_LOG 356) with the ticks-first ruler "
         f"(DECISION_LOG 272). Market-wide variable → DAY units (a WILLY on/off switch).", "",
         WEra.era_line(c.assign(close_reason=c.why) if len(c) else c, "WILLY_BTC1H",
                       note=("V2 runner / floor fills stay in the groups (their ENTRY is the tested variable; every fill is walked under the "
                             "pinned exit, never its booked V2 %) and are kept OUT of the walker-parity line") if _v2ch(c).any() else None), "",
         *HDR, _row("kept (slope > 0)", kept), _row("**blocked (slope ≤ 0)**", blocked),
         _row("UNSCORED (no warmed / current BTC 1 h bar) — never counted", uns[uns.walk.notna()] if len(uns) else uns), ""]
    nx = len(X)
    if nx:
        Xp = X[X.walk.notna()]
        L.append(f"B19 pre-registration context (fills {floor:%m-%d %H:%M} → {cstart:%m-%d %H:%M} UTC, before {COUNT_FROM}, NEVER counted): kept "
                 f"{_brief(Xp[Xp.grp == 'kept'])} · blocked {_brief(Xp[Xp.grp == 'blocked'])} · unscored {int((X.grp == 'unscored').sum())} · "
                 f"unpriced {int(X.walk.isna().sum())}.")
    cov = c.slope_status.value_counts() if len(c) else pd.Series(dtype=int)
    L.append(f"Slope coverage: scored {int(cov.get('ok', 0))}/{len(c)} (forward {int((F.slope_status == 'ok').sum())}/{len(F)}) · warm-up "
             f"{int(cov.get('warmup', 0))} · BTC cache behind {int(cov.get('lag', 0))} · hour gap {int(cov.get('gap', 0))}"
             + (f" · BTC 5m closes to {pd.Timestamp(int(m5c.open_time.max()) + M5_MS, unit='ms'):%m-%d %H:%M} UTC, last full 1 h bar "
                + (f"{pd.Timestamp(int(hb.H[hb.full].max()), unit='ms'):%m-%d %H:%M} UTC" if hb.full.any() else "none")
                if len(m5c) else " · no BTC 5m cache") + ".")
    if len(c):
        L.append("Per fill: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair}{'' if r.era == 'forward' else ' (ctx)'} slope {('–' + ('/' + r.slope_status) if np.isnan(r.slope) else f'{r.slope:+.3f}')} "
            f"[{r.grp}] candle {('–' if pd.isna(r.bar_ret) else f'{r.bar_ret:+.2f}')}{'' if r.sgrp is None or r.sgrp != r.sgrp else f' ({r.sgrp})'} · live {r.pct:+.2f} → walked "
            + (f"{r.walk:+.2f} ({r.walk_why} @ {r.walk_min:.0f} min) [{r.src}{'ᵖ' if r.src != 'ticks' else ''}]" if r.walk == r.walk else "UNPRICED")
            for r in c.itertuples()))
        _ch = _v2ch(c)
        pz = c[c.walk.notna() & ~_ch]                                # DL 356: a V2 runner / floor fill booked another exit → not a parity case
        pv = c[c.walk.notna() & _ch]
        off = pz[~pz.par_ok.astype(bool)]
        L.append(f"Walker parity (walked vs the actual exit, reason + % within {WT.PARITY_TOL} pts): {int(pz.par_ok.astype(bool).sum())}/{len(pz)}"
                 + (f" (+ {len(pv)} V2 runner / floor fill(s) out — booked the V2 exit, not the pinned one: "
                    + " · ".join(f"{r.pair} target +{getattr(r, WEra.TP_COL):g} {r.why} {r.pct:+.2f}" for r in pv.itertuples()) + ")" if len(pv) else "")
                 + ("" if not len(off) else " — off: " + " · ".join(f"{r.pair} live {r.why} {r.pct:+.3f} vs walk {r.walk_why} {r.walk:+.3f}" for r in off.itertuples()))
                 + f" · unpriced {int(c.walk.isna().sum())}" + (": " + " · ".join(
                     f"{r.ts:%m-%d %H:%M} {r.pair}" + ("" if r.pend else " (final — no data)") for r in c[c.walk.isna()].itertuples()) if c.walk.isna().any() else "")
                 + " · ᵖ = provisional 1m part.")
    if len(man):
        L.append(f"Operator-closed: {len(man)} — excluded, manually closed fill(s): " + ", ".join(f"{r.ts:%m-%d %H:%M} {r.pair} ({r.why})" for r in man.itertuples()) + ".")
    L.append(f"Evidence (not part of the verdict): {EVIDENCE}.")
    L.append(BAR)
    if ok and "first" in st:
        f0 = st["first"]
        L.append(f"**Frozen verdict (crossing at fill {f0['at']}, frozen on the run of {f0.get('run_at', '?')}, blocked N {f0['n']} · "
                 f"{f0['days']} d, kept N {f0.get('n_kept', '?')}, unscored ≤ prefix {_fmt_uns(f0.get('n_unscored_le_prefix'))}): {f0['state']}** "
                 f"({f0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    else:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    # 🩸 DIP_SUB
    s_state, s_det = sub_verdict(deep, shallow)
    Xr = X[X.walk.notna() & X.sgrp.isin(["deep", "shallow"])] if len(X) else X
    L += ["", "### 🩸 DIP_SUB — BTC 1 h rising ∧ WILLY signal red candle ≤ −1.16 % (pre-registered 2026-10-12, DECISION_LOG 279, OBSERVE only)", "",
          *HDR, _row(f"**deep (rising ∧ candle ≤ {DIP_CUT:g} %)**", deep), _row(f"shallow (rising ∧ candle > {DIP_CUT:g} %)", shallow),
          _row("UNSCORED (rising, no candle stamp) — never counted", suns[suns.walk.notna()] if len(suns) else suns), ""]
    if len(X):
        L.append(f"B19 pre-registration context (NEVER counted): deep {_brief(Xr[Xr.sgrp == 'deep'])} · shallow {_brief(Xr[Xr.sgrp == 'shallow'])} · "
                 f"no stamp {int((X.sgrp == 'unscored').sum())}.")
    L.append(f"Evidence (not part of the verdict): {SUB_EVIDENCE}.")
    L.append(SUB_BAR)
    s0 = st.get(SUB_KEY, {}) if ok else {}
    if "first" in s0:
        f1 = s0["first"]
        L.append(f"**DIP_SUB frozen verdict (crossing at fill {f1['at']}, frozen on the run of {f1.get('run_at', '?')}, deep N {f1['n_deep']} · "
                 f"shallow N {f1['n_shallow']}, unscored ≤ prefix {_fmt_uns(f1.get('n_unscored_le_prefix'))}): {f1['state']}** ({f1['detail']})")
        L.append(f"DIP_SUB live (information only, never re-decides): {s_state} — {s_det}")
    else:
        L.append(f"DIP_SUB now: ⏳ {s_state} ({s_det}).")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    if notes:
        L.append(f"Data: {' · '.join(notes)}.")
    return L + [""]


# ─────────────────────────── self-test (hermetic, offline) ───────────────────────────
def _study_slope_ref(m5, ts):
    """the study formula (study_b19_port_conc_regime.btc_feats, slope leg) — used when the untracked study module is absent."""
    b = m5.copy()
    b["t_close"] = b.open_time.astype("int64") + 300_000
    h = b.set_index(pd.to_datetime(b.t_close, unit="ms")).c.resample("1h", label="right", closed="right").last().dropna()
    e20 = h.ewm(span=20, adjust=False).mean()
    out = []
    for t in ts:
        k = np.searchsorted(h.index.values, np.datetime64(pd.Timestamp(t)), side="right") - 1
        out.append((e20.iloc[k] / e20.iloc[k - 1] - 1) * 100 if k >= 1 else np.nan)
    return np.array(out)


def selftest():
    global _CACHE
    import socket
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved = (_CACHE, WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT._NET_BLOCKED, NF._NET_BLOCKED, socket.socket,
             NF._CACHE, NF.DU_CACHE, NF.SHARED_COOLDOWN, NF.DU_COOLDOWN, NF.load_btc, WT.S4_DEPLOY_GREP, WT.S4_DEPLOY_PIN)

    class _NoNet(socket.socket):
        def connect(self, *a, **k):
            raise RuntimeError("network blocked (selftest)")
    socket.socket = _NoNet
    WT._NET_BLOCKED = NF._NET_BLOCKED = True
    study_ran = False
    try:
        chk((N_MIN, DAYS_MIN, P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED, EMA_SPAN, WARM_BARS, COUNT_FROM, STRATEGY) ==
            (15, 8, 0.95, 0.50, 4000, 7, 20, 100, "the DECISION_LOG 281 deploy", "FRENZY_WILLY"), "pre-registered constants pinned")
        g281 = WT._git_ct("(DECISION_LOG 281)")
        chk(count_start() == WT.dl281_floor() == (pd.Timestamp("2026-10-12 12:00") if g281 is None else g281) + pd.Timedelta(minutes=10),
            f"(281) the counted floor = the DL 281 deploy shared with WILLY_TIMECAP ({count_start()})")
        chk(all("Counted from the DECISION_LOG 281 deploy (one live exit" in x for x in (BAR, SUB_BAR, __doc__)) and "+1.25 / −4 / 90" in BAR,
            "(281) bar / DIP_SUB / doc texts carry the DL 281 cohort sentence")
        WT.S4_DEPLOY_GREP, WT.S4_DEPLOY_PIN = "(no such commit — selftest)", "2026-10-11 23:50:00"     # fixtures: counted from 10-12 00:00
        chk(count_start() == pd.Timestamp("2026-10-12 00:00"), "fixture floor pinned through the shared WILLY_TIMECAP constants")
        chk("pre-registered 2026-10-12, DECISION_LOG 275" in BAR and "pre-registered 2026-10-12, DECISION_LOG 275" in __doc__, "bar text")
        # ── synthetic BTC 5m: 200 h, a slow sine + trend
        t0 = pd.Timestamp("2026-09-01").value // 10**6
        n = 200 * 12
        ot = t0 + np.arange(n) * M5_MS
        cl = 80000 + 2000 * np.sin(np.arange(n) / 150.0) + np.arange(n) * 0.3
        m5 = pd.DataFrame(dict(open_time=ot.astype("int64"), c=cl))
        hb = hourly(m5)
        qs = [pd.Timestamp(t0 + x, unit="ms") for x in (150 * H_MS + 17 * 60_000, 120 * H_MS, 180 * H_MS + 59 * 60_000 + 59_000, 199 * H_MS + 1)]
        mine, st_ = slope_at(hb, [q.value // 10**6 for q in qs])
        ref = _study_slope_ref(m5, qs)
        chk(st_ == ["ok"] * 4 and np.allclose(mine, ref, rtol=0, atol=1e-12), f"slope = the study formula ({mine} vs {ref})")
        try:                                                           # the study module itself (untracked; offline, cache patched)
            import study_b19_port_conc_regime as CR
            import study_b19_port_common as C
            with tempfile.TemporaryDirectory() as td:
                os.makedirs(os.path.join(td, "k5m_full"))
                m5.to_csv(os.path.join(td, "k5m_full", "BTCUSDT.csv"), index=False)
                sv = (C.CACHE, CR._BTC)
                C.CACHE, CR._BTC = td, None
                try:
                    sref = CR.btc_feats([q.strftime("%Y-%m-%dT%H:%M:%S") for q in qs])[:, 1]
                finally:
                    C.CACHE, CR._BTC = sv
            chk(np.allclose(mine, sref, rtol=0, atol=1e-12), f"slope parity vs study_b19_port_conc_regime.btc_feats ({mine} vs {sref})")
            study_ran = True
        except ImportError:
            print("  (study_b19_port_conc_regime not importable — parity checked against the inlined study formula only)")
        # ── closed bar / no look-ahead: changing every 5m close AFTER the entry changes nothing; changing the bar closing AT the hour does
        te = t0 + 150 * H_MS + 17 * 60_000
        fut = m5.copy()
        fut.loc[fut.open_time + M5_MS > te, "c"] *= 1.5
        a, _ = slope_at(hourly(fut), [te])
        chk(abs(a[0] - mine[0]) < 1e-15, "no look-ahead: closes after the entry never move the slope")
        same = m5.copy()
        same.loc[same.open_time + M5_MS > t0 + 150 * H_MS, "c"] *= 1.5             # bars of the FORMING hour 150→151 only matter after 151:00
        b_, _ = slope_at(hourly(same), [te])
        chk(abs(b_[0] - mine[0]) < 1e-15, "the forming hour (5m bars closed after the last hour label) is never used")
        at_h = m5.copy()
        at_h.loc[at_h.open_time + M5_MS == t0 + 150 * H_MS, "c"] *= 1.01           # the 5m bar closing exactly at the hour = the closed 1 h bar
        c_, _ = slope_at(hourly(at_h), [te])
        chk(abs(c_[0] - mine[0]) > 1e-9, "the bar closed exactly at the hour is the scored bar")
        cut = m5[m5.open_time + M5_MS <= t0 + 150 * H_MS - 30 * 60_000]           # cache ends mid-hour before the entry
        _, s2 = slope_at(hourly(cut), [te])
        chk(s2 == ["lag"], "BTC cache behind (incomplete / old last bar) → lag (unscored)")
        # ── warm-up
        _, s3 = slope_at(hb, [t0 + 99 * H_MS + 5 * 60_000, t0 + 100 * H_MS + 5 * 60_000, t0 - 1])
        chk(s3 == ["warmup", "ok", "warmup"], f"< 100 one-hour bars → warm-up unscored ({s3})")
        gp = m5[~((m5.open_time + M5_MS > t0 + 149 * H_MS) & (m5.open_time + M5_MS <= t0 + 150 * H_MS))]
        _, s4 = slope_at(hourly(gp), [t0 + 151 * H_MS + 60_000])
        chk(s4 == ["gap"], f"previous hour missing → gap unscored ({s4})")
        # ── grouping
        chk([group(x) for x in (0.01, 0.0, -0.2, np.nan, None)] == ["kept", "blocked", "blocked", "unscored", "unscored"], "grouping")

        # ── verdict branches
        def coh(grp, w, days=None, pairs=None):
            m = len(w)
            days = days or [f"2026-10-{12 + i % 8:02d}" for i in range(m)]
            pairs = pairs or [f"P{i}" for i in range(m)]
            return pd.DataFrame(dict(grp=grp, walk=np.asarray(w, float), day=days, pair=pairs,
                                     ts=pd.to_datetime(days) + pd.to_timedelta(np.arange(m), "min")))
        K = coh("kept", [1.0] * 10)
        chk(verdict(K, coh("blocked", [-1.0] * 14))[0] == "COLLECTING", "N 14 → collecting")
        chk(verdict(K, coh("blocked", [-1.0] * 15, days=["2026-10-12"] * 8 + [f"2026-10-{13 + i}" for i in range(6)] + ["2026-10-12"]))[0]
            == "COLLECTING", "7 days → collecting")
        chk(verdict(K, coh("blocked", [1.0] * 15))[0] == "RETIRED", "blocked avg ≥ kept avg → RETIRED")
        chk(verdict(K, coh("blocked", [-1.0] * 15))[0] == ARM_STATE, "negative, confident, spread → ARM CANDIDATE")
        chk(verdict(K.iloc[:0], coh("blocked", [-1.0] * 15))[0] == ARM_STATE, "empty kept side: ① cannot fire, ② still applies")
        chk(verdict(K.iloc[:0], coh("blocked", [1.0] * 15))[0] == "KEEP OBSERVING", "empty kept side, blocked positive → keep observing")
        w = [-1.0] * 14 + [-14.0]                                       # one pair = exactly 50 % of the gross loss (14 / 28)
        chk(abs(NF.gross_loss_share(np.array(w), np.array([f"P{i}" for i in range(15)])) - 0.5) < 1e-12
            and verdict(K, coh("blocked", w))[0] == "KEEP OBSERVING", "exactly 50 % on one pair → not a candidate")
        dd = [f"2026-10-{12 + i % 8:02d}" for i in range(15)]
        wd = [(-6.5 if d == "2026-10-12" else -1.0) for d in dd]        # 10-12 carries 2 × 6.5 = 13 of 26 → exactly 50 %
        chk(abs(NF.gross_loss_share(np.array(wd), np.array(dd)) - 0.5) < 1e-12
            and verdict(K, coh("blocked", wd, days=dd))[0] == "KEEP OBSERVING", "exactly 50 % on one day → not a candidate")
        chk(verdict(coh("kept", [5.0] * 10), coh("blocked", [0.5] * 8 + [-0.2] * 7))[0] == "KEEP OBSERVING",
            "blocked < kept but blocked mean ≥ 0 → keep observing")
        mixed = coh("blocked", [-0.5, 0.4] * 7 + [-0.5])
        chk(verdict(K, mixed)[0] == "KEEP OBSERVING" and NF.p_mean_neg(mixed.walk.values, mixed.day.values, BOOT_N, BOOT_SEED) < P_MIN,
            "negative but not confident → keep observing")
        # ── unscored never counted / prefix / freeze
        sc = pd.concat([coh("blocked", [-1.0] * 15), coh("kept", [1.0] * 5)]).reset_index(drop=True)
        mx = pd.concat([coh("blocked", [-1.0] * 16), coh("kept", [1.0] * 3, days=["2026-10-12"] * 3)])
        mx["ts"] = pd.Timestamp("2026-10-12") + pd.to_timedelta(np.r_[np.arange(16) * 60 + 30, [0, 61, 10_000]], "min")
        pre = crossing_prefix(mx)
        chk(pre is not None and (pre.grp == "blocked").sum() == 15 and pre.grp.iloc[-1] == "blocked" and (pre.grp == "kept").sum() == 2,
            "prefix ends at the 15th blocked fill")
        st, ch = freeze({}, sc, "now", [], [])
        chk(ch and st["first"]["state"] == ARM_STATE and st["first"]["n"] == 15, "frozen once at the crossing")
        st2, ch2 = freeze(st, coh("blocked", [5.0] * 40), "later", [], [])
        chk(not ch2 and st2["first"]["state"] == ARM_STATE and st2["first"]["run_at"] == "now", "never re-frozen")
        why = []
        chk(not freeze({}, sc, "x", [(pd.Timestamp("2026-10-12 00:30"), "provisional XUSDT")], why)[1] and "provisional XUSDT" in why[0],
            "a provisional fill ≤ the prefix end holds the freeze")
        chk(freeze({}, sc, "x", [(pd.Timestamp("2026-12-01"), "provisional LATE")], [])[1], "a pending fill after the prefix end does not hold")
        chk(pending(None, False, True) and pending(None, True, False) and not pending(None, False, False) and pending("1m", True, False)
            and pending("ticks+1m", True, False) and not pending("ticks+1m", False, True) and not pending("ticks", True, True),
            "pending: unpriced while data is due · 1m part while ticks are due · ticks = final")

        # ── 🩸 DIP_SUB unit checks
        chk([sub_group(*x) for x in ((0.1, -1.16), (0.1, -1.15), (0.1, np.nan), (0.0, -3.0), (np.nan, -3.0))]
            == ["deep", "shallow", "unscored", None, None], "DIP_SUB grouping: −1.16 inclusive = deep · no stamp = unscored · not rising = out")

        def sd_(w, days=None, pairs=None, g="deep"):
            return coh(g, w, days, pairs).rename(columns={"grp": "sgrp"})
        D, S = sd_([1.0] * 15), sd_([0.2] * 15, g="shallow")
        chk(sub_verdict(D, S.iloc[:14])[0] == "COLLECTING" and sub_verdict(D.iloc[:14], S)[0] == "COLLECTING", "each side needs N ≥ 15")
        chk(sub_verdict(sd_([0.2] * 15), S)[0] == "RETIRED" and sub_verdict(sd_([0.1] * 15), S)[0] == "RETIRED", "deep ≤ shallow → RETIRED")
        chk(sub_verdict(D, S)[0] == SUB_STATE, "deep beats shallow, confident, spread → CANDIDATE")
        wg = [1.0] * 14 + [14.0]                                        # one pair = exactly 50 % of the deep gross gain
        chk(sub_verdict(sd_(wg), S)[0] == "KEEP OBSERVING", "exactly 50 % of the deep gain on one pair → keep observing")
        noisy = sd_([2.0, -1.5] * 7 + [2.0])
        chk(sub_verdict(noisy, S)[0] == "KEEP OBSERVING" and p_diff_pos(noisy, S) < P_MIN, "better but not confident → keep observing")
        both = pd.concat([D, S]).reset_index(drop=True)
        both["ts"] = pd.Timestamp("2026-10-12") + pd.to_timedelta(np.arange(len(both)), "min")
        stS, chS = sub_freeze({}, both, "now", [], [])
        chk(chS and stS[SUB_KEY]["first"]["state"] == SUB_STATE and stS[SUB_KEY]["first"]["n_deep"] == 15 and "first" not in stS,
            "DIP_SUB frozen under its own key")
        chk(not sub_freeze(stS, both.assign(walk=-9.0), "later", [], [])[1], "DIP_SUB never re-frozen")
        wS = []
        chk(not sub_freeze({}, both, "x", [(pd.Timestamp("2026-10-12"), "provisional Q")], wS)[1] and wS and wS[0].startswith("DIP_SUB:"),
            "DIP_SUB deferred by a pending fill ≤ its prefix end")

        # ── end-to-end on synthetic exports + ticks (temp dirs; no repo state)
        with tempfile.TemporaryDirectory() as td:
            _CACHE = os.path.join(td, "cache")
            WT._CACHE = _CACHE
            WT.MY_CACHE = os.path.join(_CACHE, "scout_willy_timecap")
            os.makedirs(os.path.join(_CACHE, "k5m_full"))
            b0 = pd.Timestamp("2026-10-01").value // 10**6
            nb = (12 * 24 + 1) * 12                                    # 10-01 → 10-13 00:05
            bc = 80000 + np.arange(nb) * 1.0                            # rising BTC → slope > 0 …
            ix = (b0 + np.arange(nb) * M5_MS + M5_MS) > pd.Timestamp("2026-10-12 06:00").value // 10**6
            bc[ix] = bc[ix][0] - np.arange(ix.sum()) * 30.0              # … falling after 10-12 06:00 → slope ≤ 0
            pd.DataFrame(dict(open_time=b0 + np.arange(nb) * M5_MS, c=bc)).to_csv(os.path.join(_CACHE, "k5m_full", "BTCUSDT.csv"), index=False)
            dl = os.path.join(td, "dl")
            os.makedirs(dl)
            WT.EXPORT_GLOB = os.path.join(dl, "scalpars_orders_paper_*.csv")
            E = 100.0
            px = lambda net: E * (1 + (net + WT.TAKER) / 100) / (1 - WT.TAKER / 100)   # noqa: E731
            base = dict(direction="LONG", status="CLOSED", notional_value=1000.0, entry_price=E, pnl=1.0, closed_at="x", entry_strategy=STRATEGY)
            rows = [dict(base, opened_at="2026-10-11T10:00:00", pair="CTXUSDT", pnl_percentage=1.0, close_reason="FRENZY_TP"),
                    dict(base, opened_at="2026-10-12T03:00:00", pair="KEPUSDT", pnl_percentage=1.0, close_reason="FRENZY_TP", **{DIP_COL: -1.16}),
                    dict(base, opened_at="2026-10-12T09:00:00", pair="BLKUSDT", pnl_percentage=-4.0, close_reason="STOP_LOSS"),
                    dict(base, opened_at="2026-10-12T09:00:00", pair="BLKUSDT", pnl_percentage=0.0, close_reason="", status="OPEN"),
                    dict(base, opened_at="2026-10-13T03:00:00", pair="LAGUSDT", pnl_percentage=1.0, close_reason="FRENZY_TP"),
                    dict(base, opened_at="2026-10-12T10:00:00", pair="MANUSDT", pnl_percentage=0.5, close_reason="MANUAL_CLOSE"),
                    dict(base, opened_at="2026-10-12T11:00:00", pair="NOPUSDT", pnl_percentage=1.0, close_reason="FRENZY_TP"),
                    dict(base, opened_at="2026-10-12T12:00:00", pair="LITUSDT", pnl_percentage=1.0, close_reason="FRENZY_TP", entry_strategy="FRENZY_LITE")]
            pd.DataFrame(rows).to_csv(os.path.join(dl, "scalpars_orders_paper_a.csv"), index=False)
            for pair, ts, nets in (("CTXUSDT", "2026-10-11 10:00:00", [0.3, 1.02]), ("KEPUSDT", "2026-10-12 03:00:00", [-1.0, 1.05]),
                                   ("BLKUSDT", "2026-10-12 09:00:00", [-2.0, -4.0]), ("LAGUSDT", "2026-10-13 03:00:00", [1.01])):
                tk = pd.Timestamp(ts).value // 10**6
                os.makedirs(os.path.join(_CACHE, "ticks_q", pair), exist_ok=True)
                np.savez_compressed(os.path.join(_CACHE, "ticks_q", pair, f"{ts[:10]}.npz"),
                                    t=(tk + np.array([60_000, 300_000])[:len(nets)]).astype(np.int64), p=np.array([px(x) for x in nets]))
            th = {"frenzy_willy_tp_pct": 1.0, "frenzy_willy_stop_pct": 4.0, "frenzy_willy_max_hold_minutes": 120}
            chk(willy_rule(th) == willy_rule() == PINNED_RULE == (1.25, 4.0, 90), "DL 356: the registered exit is pinned — a config never moves it")
            th = (1.0, 4.0, 120)                                         # the harness's synthetic exit (run(rule=…), selftest only)
            sp, rp = os.path.join(td, "s.json"), os.path.join(td, "rows.csv")
            now = pd.Timestamp("2026-10-13 05:00").value // 10**6
            out = "\n".join(run(now, fetch=False, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12"))
            chk("pre-registered 2026-10-12, DECISION_LOG 275" in out and "B19 pre-registration context" in out
                and "kept 1 · 1 d · 100 % · +1.020" in out.split("B19 pre-registration context")[1].split("\n")[0], f"context line\n{out}")
            chk("| kept (slope > 0) | 1 | 1 | 1 | 100 % | +1.050 %" in out and "| **blocked (slope ≤ 0)** | 1 | 1 | 1 | 0 % | -4.000 %" in out,
                f"forward groups on the walked % (OPEN row superseded by its CLOSED twin)\n{out}")
            chk("LAGUSDT slope –/lag [unscored]" in out and "| UNSCORED" in out and "Operator-closed: 1 — excluded, manually closed" in out and "LITUSDT" not in out
                and "NOPUSDT" in out and "UNPRICED" in out, f"unscored listed (never counted) · manual excluded · non-WILLY out · unpriced listed\n{out}")
            chk("reason + % within 0.05 pts): 4/4 · unpriced 1: 10-12 11:00 NOPUSDT ·" in out and "Now: ⏳ COLLECTING (blocked N 1/15" in out and not os.path.exists(sp)
                and os.path.exists(rp) and len(pd.read_csv(rp)) == 5, f"parity + collecting + rows stored, no state written\n{out}")

            o2 = "\n".join(run(now, fetch=False, state_path=os.path.join(td, "s_cs.json"), rows_path=os.path.join(td, "rows_cs.csv"), rule=th,
                                floor="2026-10-08 15:13:12", cstart="2026-10-12 05:00"))
            chk("KEPUSDT (ctx)" in o2 and "| kept (slope > 0) | 0 |" in o2 and "opened ≥ 2026-10-12 05:00 UTC (the DECISION_LOG 281 deploy" in o2,
                f"(281) a fill before the counted floor is context only (never counted)\n{o2}")
            chk("KEPUSDT slope +" in out and "[kept] candle -1.16 (deep)" in out and "| **deep (rising ∧ candle ≤ -1.16 %)** | 1 | 1 | 1 | 100 % | +1.050 %" in out
                and "DIP_SUB now: ⏳ COLLECTING (deep N 1/15" in out and "pre-registered 2026-10-12, DECISION_LOG 279" in out
                and "CTXUSDT (ctx) slope +0.0" in out and "no stamp 1" in out, f"DIP_SUB e2e (−1.16 inclusive = deep; ctx fill without a stamp)\n{out}")
            # ── (L4) coverage prints the max 5m close and the last FULL hour, never an incomplete hour label
            bk = pd.read_csv(os.path.join(_CACHE, "k5m_full", "BTCUSDT.csv"))
            lo_ = int(bk.open_time.max())
            pd.concat([bk, pd.DataFrame(dict(open_time=[lo_ + M5_MS, lo_ + 2 * M5_MS], c=[1.0, 1.0]))]).to_csv(
                os.path.join(_CACHE, "k5m_full", "BTCUSDT.csv"), index=False)
            out = "\n".join(run(now, fetch=False, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12"))
            chk("BTC 5m closes to 10-13 01:10 UTC, last full 1 h bar 10-13 01:00 UTC" in out and "02:00 UTC" not in out.split("Slope coverage")[1].split("\n")[0],
                f"coverage: max 5m close + last full hour\n{out}")
            bk.to_csv(os.path.join(_CACHE, "k5m_full", "BTCUSDT.csv"), index=False)
            # ── (L5) fetch=True: NF.load_btc called only when the cache is behind the newest counted fill
            calls = []
            NF.load_btc = lambda need, now_ms, fetch=True: (calls.append(need), (None, None, ["stub fetch"]))[1]   # noqa: E731
            out = "\n".join(run(now, fetch=True, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12"))
            chk(calls == [pd.Timestamp("2026-10-13 03:00").value // 10**6] and "Data: stub fetch." in out, f"cache behind → one fetch ({calls})")
            calls.clear()
            fresh = pd.DataFrame([r for r in rows if r["pair"] != "LAGUSDT"])
            run(now, orders=fresh, fetch=True, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12")
            chk(calls == [], "cache covers the newest counted fill → no fetch")
            run(now, fetch=False, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12")
            chk(calls == [], "fetch=False → never")
            NF.load_btc = saved[11]
            # the REAL fetcher: shared cooldown honoured; with no cooldown the network block fails the request cleanly (never raises)
            NF._CACHE, NF.DU_CACHE = os.path.join(td, "nfcache"), os.path.join(td, "nfcache", "scout_dailyup")
            NF.SHARED_COOLDOWN = os.path.join(NF._CACHE, ".binance_cooldown_until")
            NF.DU_COOLDOWN = os.path.join(NF.DU_CACHE, ".ratelimited_until")
            os.makedirs(NF.DU_CACHE)
            with open(NF.SHARED_COOLDOWN, "w") as f:
                f.write(str(now + 3_600_000))
            out = "\n".join(run(now, fetch=True, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12"))
            chk("fetching stopped: rate-limit cooldown until" in out and "fetch failed" not in out, f"cooldown → no request\n{out}")
            os.remove(NF.SHARED_COOLDOWN)
            out = "\n".join(run(now, fetch=True, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12"))
            chk("5m fetch failed (network blocked (selftest))" in out and not os.listdir(NF.DU_CACHE), f"network blocked → clean failure, no cache write\n{out}")
            chk(not os.path.exists(sp), "still nothing frozen")
            # ── (L5) 3-day lag hold
            Fh = pd.DataFrame(dict(ts=[pd.Timestamp("2026-10-14 06:30")], pair=["LGX"], src=["ticks"], pend=[False], slope_status=["lag"]))
            t_l = pd.Timestamp("2026-10-14 06:30").value // 10**6
            chk(hold_list(Fh, [], t_l + LAG_PENDING_MS - 1) == [(pd.Timestamp("2026-10-14 06:30"), "BTC slope pending LGX")]
                and hold_list(Fh, [], t_l + LAG_PENDING_MS) == [] and hold_list(Fh.assign(slope_status="warmup"), [], t_l) == [],
                "lag holds < 3 days only; warm-up never holds")

        # ── (L5) full freeze cycle through run(): deferred by a lag fill → frozen → persisted → never re-frozen; unscored on record
        with tempfile.TemporaryDirectory() as td:
            _CACHE = os.path.join(td, "cache")
            WT._CACHE, WT.MY_CACHE = _CACHE, os.path.join(_CACHE, "scout_willy_timecap")
            os.makedirs(os.path.join(_CACHE, "k5m_full"))
            b0 = pd.Timestamp("2026-10-01").value // 10**6
            nb = 21 * 24 * 12                                           # 10-01 → 10-22
            k = np.arange(nb)
            k0 = int((pd.Timestamp("2026-10-11 12:00").value // 10**6 - b0) // M5_MS)
            bc = np.where(k < k0, 80000 * (1 + 1e-4 * k), 80000 * (1 + 1e-4 * k0) * 0.9998 ** (k - k0))   # rising, then falling (slope ≤ 0)
            bt = pd.DataFrame(dict(open_time=b0 + k * M5_MS, c=bc))
            bt = bt[bt.open_time + M5_MS != pd.Timestamp("2026-10-14 06:00").value // 10**6]          # incomplete 06:00 hour → a 'lag' fill
            bt.to_csv(os.path.join(_CACHE, "k5m_full", "BTCUSDT.csv"), index=False)
            dl = os.path.join(td, "dl")
            os.makedirs(dl)
            WT.EXPORT_GLOB = os.path.join(dl, "scalpars_orders_paper_*.csv")
            fills = [(f"2026-10-{12 + d:02d} {h}:00:00", f"B{d}{h}USDT") for d in range(8) for h in (12, 13)][:15]
            fills.append(("2026-10-14 06:30:00", "LGXUSDT"))
            rr = []
            for i, (ts, pair) in enumerate(fills):
                win = i % 3 == 0
                rr.append(dict(base, opened_at=ts.replace(" ", "T"), pair=pair, pnl_percentage=1.02 if win else -4.05,
                               close_reason="FRENZY_TP" if win else "STOP_LOSS", **{DIP_COL: -2.0}))
                tk = pd.Timestamp(ts).value // 10**6
                os.makedirs(os.path.join(_CACHE, "ticks_q", pair), exist_ok=True)
                np.savez_compressed(os.path.join(_CACHE, "ticks_q", pair, f"{ts[:10]}.npz"), t=np.array([tk + 60_000], dtype=np.int64),
                                    p=np.array([px(1.02 if win else -4.05)]))
            pd.DataFrame(rr).to_csv(os.path.join(dl, "scalpars_orders_paper_a.csv"), index=False)
            sp, rp = os.path.join(td, "s.json"), os.path.join(td, "rows.csv")
            n1 = pd.Timestamp("2026-10-15 06:00").value // 10**6                                # < 3 days after the lag fill
            out = "\n".join(run(n1, fetch=False, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12"))
            chk("LGXUSDT slope –/lag [unscored]" in out and "BTC slope pending LGXUSDT" in out and not os.path.exists(sp),
                f"a lag fill ≤ the prefix end defers the freeze\n{out}")
            n2 = pd.Timestamp("2026-10-25 00:00").value // 10**6
            out = "\n".join(run(n2, fetch=False, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12"))
            st_ = NF.du_load_state(sp)[0]
            f0 = st_.get("first", {})
            chk(os.path.exists(sp) and f0.get("n") == 15 and f0.get("days") == 8 and f0.get("n_unscored_le_prefix") == {"total": 1, "lag": 1}
                and "**Frozen verdict" in out and "unscored ≤ prefix 1 (lag 1)" in out and SUB_KEY not in st_,
                f"frozen + persisted, unscored on record; DIP_SUB untouched (no rising fills)\n{st_}\n{out}")
            raw0 = open(sp).read()
            pd.DataFrame(rr + [dict(rr[0], opened_at="2026-10-20T12:00:00", pair="ZZZUSDT")]).to_csv(os.path.join(dl, "scalpars_orders_paper_a.csv"), index=False)
            out = "\n".join(run(n2 + 86_400_000, fetch=False, state_path=sp, rows_path=rp, rule=th, floor="2026-10-08 15:13:12"))
            chk(open(sp).read() == raw0 and "Live (information only, never re-decides)" in out, "never re-frozen on a later run")
    finally:
        (_CACHE, WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT._NET_BLOCKED, NF._NET_BLOCKED, socket.socket,
         NF._CACHE, NF.DU_CACHE, NF.SHARED_COOLDOWN, NF.DU_COOLDOWN, NF.load_btc, WT.S4_DEPLOY_GREP, WT.S4_DEPLOY_PIN) = saved
        WT._K1.clear()
    print(f"selftest WILLY_BTC1H OK ({ok} checks{', study parity ran' if study_ran else ''})")
    return study_ran


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
