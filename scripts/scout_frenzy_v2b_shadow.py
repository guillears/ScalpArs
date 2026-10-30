#!/usr/bin/env python3
"""🎲🪶 Scout — FRENZY_V2B_SHADOW (DECISION_LOG 379, operator 2026-10-26; OBSERVE-ONLY — never changes config, never calls the bot API,
never writes the DB; writes only reports/).

OPERATOR QUESTION  Should FRENZY_LITE (incl. FRENZY_LITE_RETRY) and FRENZY_LONG run WILLY's V2b exit instead of their own?
  + (scope addition, same day) do FIXED higher take profits pay: LITE TP4 / TP5 / TP6 (own −3 stop) · LONG TP2 / TP3 / TP4 (own −4 stop)?

THEORY (pre-registered with the line — printed in every run)
  T1 V2b: the spike-high target pays only when H is NEAR the entry and the volume is still alive. LITE enters hours into an episode with
     faded volume → far-H LITE fills should NOT gain. LONG enters on the hot volume state → closer to WILLY. Check: near-H (< 3 %) bucket
     Δ > far-H (≥ 6 %) bucket Δ.
  T2 fixed TPs (DL 284): LITE is all-or-nothing — its winners all ran to +3, its losers never passed +0.68 → a higher fixed TP costs only
     the winners that stall between +3 and the new TP; the read answers "how far do LITE winners keep running". LONG is bounce-then-
     collapse (why it runs +1.5) → PREDICTION: higher fixed TPs are WORSE for LONG; a LONG shadow beating +1.5 REFUTES the theory (reported).

COHORT  CLOSED FRENZY_LITE / FRENZY_LITE_RETRY / FRENZY_LONG fills, orders exports (~/Downloads/scalpars_orders_*.csv, paper + live via
        export_glob) ∪ the archived batches (reports/BASELINE<N>_*orders*.csv), dedup (opened_at[:19], pair, direction), CLOSED beats OPEN,
        newest export wins; operator-closed fills out (scout_gatekit). GATE cohort = opened ≥ scout_gatekit.dl_deploy_ms(379) (commit + 10
        min); everything older = the MASTER / ARCHIVED REFERENCE block — IN-SAMPLE (the line was registered after seeing it), never counted.
PER FILL
  H         the episode spike high = frenzy_walk's peak at the entry pass: max 5m high from the spike bar (entry_frenzy_spike_at = its CLOSE
            time) through the last 5m bar CLOSED at the fill — build_willy_exit_cf.h_window / reconstruct_high (5m → 1m → ticks caches),
            else a budgeted public 5m page (scout_willy_latejoin's cached fetcher). No look-ahead.
  dist_H    T raw = net % at the PRICE H × 0.99 (entry taker fee % of the fill + TODAY's config maker exit fee, trading_config.json
            maker_fee — the fee the engine would use at a new entry; the other V2b params are pinned) — services.exec_shadow.net_pct_at via
            services.frenzy.frenzy_willy_high_target, uncapped. Buckets near < 3 · mid 3–6 · far ≥ 6. Also: hours since the spike, entry vol
            mult, LITE C zone (vol mult ≥ 20 ∧ BTC 1h slope > 0).
  TODAY     the BASELINE every Δ is measured against (review fix, DL 379): TODAY's live exit re-walked on every fill's own path
            (scout_frenzy_willy_exit.load_path: ticks → 1m tail; scout_willy_exit_v2.walk_v2: resting TP books the NET line, stop = first
            print / bar open through −stop, 12 h cap): LITE (+ RETRY) TP = the fill's stored frenzy_lite_tp_used (the LITE C TP it ran), else
            the LITE C rule on its stamps (entry_frenzy_vol_mult, entry_btc_1h_slope), stop −3 · LONG +1.5 / −4. ⚠ The LONG TP is HARDCODED
            +1.5 (there is no stored LONG TP column): a future LONG TP change needs an update here — the parity line guards it (gate-cohort
            fills would stop replicating).
  REPLICA   (PARITY ONLY) the fill's own ERA exit: LITE stored TP, else the era rule (pre-283 +3 · 283→284 +1.5 · 284→301 +3 · ≥ 301 LITE C);
            LONG pre-283 +3 else +1.5; stop = stamped entry_frenzy_stop_pct, else LONG −4 from DL 317 / −3, LITE −3. A resting TP is assumed
            in EVERY era — the pre-296 (LITE) / pre-310 (LONG) market TPs are known parity misses (in-sample REF only). PARITY = |replica −
            booked| ≤ 0.15 on ≥ 90 % of the priced fills (the gate reads it on its frozen prefix).
  SHADOWS   V2B      T = max(+1.25, min(+8, net at H × 0.99)) (frenzy_willy_high_target on PINNED V2b params), resting TP at T, stop −4,
                     late TP +0.25 after 30 min, cap 90, no floor (DL 356 → 358)
            V2B_OWN  the same with LITE's stop −3 (LONG's stop IS −4 → not a separate shadow)
            TPx      fixed +x / TODAY's sleeve stop (LITE −3 · LONG −4) / 12 h (LITE TP4 TP5 TP6 · LONG TP2 TP3 TP4)
            Δ = shadow − TODAY, paired on the same path.
  COMPLETE  a fill counts for a sleeve's gate only when TODAY and EVERY shadow of that sleeve resolved (H found) — the SAME fills for every
            shadow. An incomplete fill < 3 days old → the read HOLDs; older → counted and printed as "not evaluable" (never a silent drop).
🔒 FROZEN GATE (set 2026-10-26 before any gate-cohort fill exists; per sleeve × shadow; decided ONCE — NO_VERDICT included — persisted
   atomically in reports/.frenzy_v2b_shadow_gate.json (an unreadable file is never overwritten); NEVER an automatic switch):
   read at the shortest prefix (open order) of ≥ 30 complete fills on ≥ 10 UTC days →
   CANDIDATE — OPERATOR CALL iff avg Δ ≥ +0.10 %/fill ∧ day-block bootstrap one-sided lower bound > 0 at α = 0.05 / m (Bonferroni, m = the
   number of shadows of that sleeve: LITE 5, LONG 4) ∧ replica parity on the prefix ≥ 90 % (else NO_VERDICT) ∧ no UTC day carries ≥ 50 %
   of Σ Δ; else KEEP the live exit. HOLD while a cohort fill opened ≤ the prefix end is still OPEN or incomplete and < 3 days old.
Binance: caches first; public endpoints only (5m pages ≤ 15 per run, stop at X-MBX-USED-WEIGHT-1M ≥ 500, shared cooldown file honoured,
418 / 429 arm it via the shared fetchers; ticks / 1m via scout_willy_timecap.ensure_data, ≤ 2 requests). Numbers move as paths arrive.
Usage: venv/bin/python scripts/scout_frenzy_v2b_shadow.py [--no-fetch]
"""
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import export_glob as _EG              # noqa: E402
import scout_gatekit as _GK            # noqa: E402
import scout_willy_exit_v2 as SV       # noqa: E402  (walk_v2 — the V2b walker)

DL, LAG_MIN = 379, 10
MIN, HOUR, DAY, BAR5 = 60_000, 3_600_000, 86_400_000, 300_000
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_*.csv")
ARCHIVE_GLOB = os.path.join(_ROOT, "reports", "BASELINE[0-9]*_*orders*.csv")
STATE = os.path.join(_ROOT, "reports", ".frenzy_v2b_shadow_gate.json")
CSV_OUT = os.path.join(_ROOT, "reports", "SCOUT_FRENZY_V2B_SHADOW.csv")
SLEEVE_OF = {"FRENZY_LITE": "LITE", "FRENZY_LITE_RETRY": "LITE", "FRENZY_LONG": "LONG"}
SLEEVES = ("LITE", "LONG")
# ── pinned exits (never read from config) ──
CAP_LIVE = 720
V2B_STOP, V2B_CAP, V2B_LATE_MIN, V2B_LATE_TP = 4.0, 90, 30.0, 0.25
V2B_TH = SimpleNamespace(frenzy_willy_high_target_enabled=True, frenzy_willy_tp_pct=1.25, frenzy_willy_stop_pct=4.0,
                         frenzy_willy_max_hold_minutes=90, frenzy_willy_high_k_pct=1.0, frenzy_willy_high_tp_max_pct=8.0)
OWN_STOP = {"LITE": 3.0, "LONG": 4.0}                     # TODAY's sleeve stops (frenzy_stop_pct 3 · frenzy_long_stop_pct 4, DL 317)
TODAY_LONG = (1.5, 4.0)                                   # TODAY's LONG exit — HARDCODED (no stored LONG TP column; parity guards it)
FIXED_TP = {"TP2": 2.0, "TP3": 3.0, "TP4": 4.0, "TP5": 5.0, "TP6": 6.0}
SHADOWS = {"LITE": ("V2B", "V2B_OWN", "TP4", "TP5", "TP6"), "LONG": ("V2B", "TP2", "TP3", "TP4")}
ALL_SHADOWS = ("V2B", "V2B_OWN", "TP2", "TP3", "TP4", "TP5", "TP6")
LITE_C = SimpleNamespace(frenzy_lite_c_enabled=True, frenzy_lite_c_vol_mult=20.0, frenzy_lite_c_btc_slope_min=0.0, frenzy_lite_c_tp_pct=2.0,
                         frenzy_lite_tp_pct=3.0, frenzy_tp_pct=1.5)
# ── frozen gate ──
GATE_N, GATE_DAYS, MIN_GAIN, ALPHA, DAY_SHARE = 30, 10, 0.10, 0.05, 0.50
PARITY_TOL, PARITY_MIN = 0.15, 0.90
BOOT_N, BOOT_SEED = 4000, 7
PENDING_DAYS = 3
DIST_EDGES = (3.0, 6.0)
# ── Binance budget (5m H pages) ──
MAX_REQ_5M, WEIGHT_STOP_5M, BUDGET_S = 15, 500, 25.0
MAKER_FEE_DEF = 0.00018
LIST_CAP = 25
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "entry_price", "quantity", "entry_fee", "pnl_percentage",
        "close_reason", "frenzy_lite_tp_used", "entry_frenzy_spike_at", "entry_frenzy_vol_mult", "entry_frenzy_hours", "entry_frenzy_stop_pct",
        "entry_btc_1h_slope")
THEORY = [
    "THEORY T1 (V2b): the spike-high target pays only when H is NEAR the entry and the volume is still alive. LITE enters hours into an "
    "episode with faded volume → far-H LITE fills should NOT gain; LONG enters on the hot volume state → closer to WILLY. Check: near-H "
    "(< 3 %) Δ > far-H (≥ 6 %) Δ.",
    "THEORY T2 (fixed TPs, DL 284): LITE is all-or-nothing (winners all ran to +3, losers never passed +0.68) → a higher TP costs only the "
    "winners that stall between +3 and the new TP (reads how far LITE winners keep running). LONG is bounce-then-collapse (why +1.5) → "
    "PREDICTION: higher TPs are WORSE for LONG; a LONG TPx beating TODAY's +1.5 / −4 REFUTES the theory.",
]


# ─────────────────────────── small helpers ───────────────────────────
def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ms(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    try:
        ts = pd.Timestamp(str(v)[:19])
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return int(ts.value // 1_000_000)


def _maker_fee():
    try:
        with open(os.path.join(_ROOT, "trading_config.json"), encoding="utf-8") as fh:
            v = float(json.load(fh).get("maker_fee", MAKER_FEE_DEF))
        return v if math.isfinite(v) and 0 <= v < 0.01 else MAKER_FEE_DEF
    except Exception:
        return MAKER_FEE_DEF


def _fmt(v, f="{:+.2f}"):
    x = _f(v)
    return "n/a" if x is None else f.format(x)


def dist_bucket(d):
    if d is None or not math.isfinite(d):
        return "noH"
    return "near<3" if d < DIST_EDGES[0] else ("mid3-6" if d < DIST_EDGES[1] else "far>=6")


def entry_fee_pct(r):
    e, q, fee = _f(r.get("entry_price")), _f(r.get("quantity")), _f(r.get("entry_fee"))
    if not (e and e > 0 and q and q > 0) or fee is None or fee < 0:
        return None
    return fee / (e * q) * 100.0


def v2b_target(r, H, maker=None):
    """→ (T_used, T_raw) on the PINNED V2b params via the engine's frenzy_willy_high_target; (None, None) when H / fees are unusable."""
    from services.frenzy import frenzy_willy_high_target
    ef = entry_fee_pct(r)
    if H is None or ef is None:
        return None, None
    tu, traw = frenzy_willy_high_target(V2B_TH, _f(r.get("entry_price")), H, ef, _maker_fee() if maker is None else maker)
    return (tu, traw) if traw is not None else (None, None)


# ─────────────────────────── live exit replica (era-aware) ───────────────────────────
_ERAS = {}


def eras():
    """deploy floors (ms, commit + 10 min) of the exit changes the replica needs: 250 (fixed TP replaces the DL 205 lock / trail — older
    fills are not replicable and stay out) · 283 (TP +1.5) · 284 (LITE own +3) · 301 (LITE C) · 317 (LONG −4) · 420 (LITE C off). Missing → None (the current
    rule is used on that side)."""
    if not _ERAS:
        for n in (250, 283, 284, 301, 317, 420):
            try:
                _ERAS[n] = _GK.dl_deploy_ms(n, lag_min=LAG_MIN, label=f"FRENZY_V2B_SHADOW replica era DL {n}")
            except Exception:
                _ERAS[n] = None
    return _ERAS


def _before(te, ms):
    return ms is not None and te is not None and te < ms


def live_exit(r, era=None):
    """→ (tp, stop, tp_src) of the fill's own live exit."""
    from services.frenzy import frenzy_lite_c_tp
    era = eras() if era is None else era
    sl = SLEEVE_OF.get(str(r.get("entry_strategy")), "LONG")
    te = _ms(r.get("opened_at"))
    st = _f(r.get("entry_frenzy_stop_pct"))
    if not (st and st > 0):
        st = 4.0 if (sl == "LONG" and not _before(te, era.get(317))) else 3.0
    if sl == "LONG":
        return (3.0 if _before(te, era.get(283)) else 1.5), st, "era"
    tp = _f(r.get("frenzy_lite_tp_used"))
    if tp and tp > 0:
        return tp, st, "stored"
    if _before(te, era.get(283)):
        return 3.0, st, "era"
    if _before(te, era.get(284)):
        return 1.5, st, "era"
    if _before(te, era.get(301)):
        return 3.0, st, "era"
    if era.get(420) is not None and not _before(te, era.get(420)):   # LITE C off: new fills store no TP and run +3
        return 3.0, st, "era"
    return frenzy_lite_c_tp(LITE_C, r.get("entry_frenzy_vol_mult"), r.get("entry_btc_1h_slope"))[0], st, "liteC"


def today_exit(r):
    """→ (tp, stop) of TODAY's live exit for this fill — the Δ BASELINE: LITE (+ RETRY) +3 / −3 (LITE C off since DL 420: old fills are compared with today's +3 too) ·
    LONG +1.5 / −4 (HARDCODED — no stored LONG TP column; update on a LONG TP change)."""
    if SLEEVE_OF.get(str(r.get("entry_strategy")), "LONG") == "LONG":
        return TODAY_LONG
    return float(LITE_C.frenzy_lite_tp_pct), OWN_STOP["LITE"]


def lite_zone(r):
    from services.frenzy import frenzy_lite_c_tp
    vm, sl = _f(r.get("entry_frenzy_vol_mult")), _f(r.get("entry_btc_1h_slope"))
    if vm is None or sl is None:
        return "unread"
    return "zone" if frenzy_lite_c_tp(LITE_C, vm, sl)[1] else "out"


# ─────────────────────────── walks ───────────────────────────
def walk_live(P, tp, sl):
    return SV.walk_v2(P, tp, sl=sl, cap=CAP_LIVE, late=None, ztp=None)


def walk_v2b(P, T, sl=V2B_STOP):
    return SV.walk_v2(P, T, sl=sl, cap=V2B_CAP, late=V2B_LATE_MIN, ztp=V2B_LATE_TP)


def shadow_pct(P, name, sleeve, T, own_stop=None):
    """one shadow's booked % on the path (nan = undecided / unpriceable). own_stop default = TODAY's sleeve stop (LITE −3 / LONG −4) —
    V2B_OWN and the TPx shadows run it, so Δ vs TODAY isolates the take-profit change."""
    if name not in SHADOWS[sleeve]:
        return float("nan")
    own = OWN_STOP[sleeve] if own_stop is None else float(own_stop)
    if name in ("V2B", "V2B_OWN"):
        if T is None:
            return float("nan")
        w = walk_v2b(P, T, V2B_STOP if name == "V2B" else own)
    else:
        w = walk_live(P, FIXED_TP[name], own)
    return float(w["pct"]) if w["why"] != "UNRES" else float("nan")


# ─────────────────────────── H (spike high) ───────────────────────────
class _Bud:
    def __init__(self, fetch=True):
        self.fetch, self.max_req, self.n, self.used, self.stop = fetch, MAX_REQ_5M, 0, 0, None
        self.deadline = time.monotonic() + BUDGET_S
        self.notes = []


def _h_from_5m(pair, b0, last, now_ms, bud):
    """max 5m high over open times [b0, last] from scout_willy_latejoin's cached public 5m pages → float or None (cache first; a fetch only
    while the budget holds; stops at X-MBX-USED-WEIGHT-1M ≥ WEIGHT_STOP_5M; the shared cooldown is checked inside the fetcher)."""
    import scout_willy_latejoin as LJ
    n = (last - b0) // BAR5 + 1
    if n <= 0 or n > LJ.KL_MAX:
        return None
    if bud.used >= WEIGHT_STOP_5M and not bud.stop:
        bud.stop = f"used weight {bud.used} ≥ {WEIGHT_STOP_5M}"
    if bud.stop:
        bud.fetch = False                                     # cache reads only from here
    rows = LJ._fetch_chunk(pair, b0, n, now_ms, bud)
    if rows is None or isinstance(rows, str):
        return None
    k = LJ._k5_frame(rows)
    k = k[(k.open_time >= b0) & (k.open_time <= last)]
    if k.open_time.nunique() != n:
        return None
    h = float(k.h.max())
    return h if math.isfinite(h) and h > 0 else None


def resolve_h(r, now_ms, bud, tcache):
    """→ (H, src). Caches (5m → 1m → ticks, build_willy_exit_cf.reconstruct_high), then the budgeted 5m page."""
    import build_willy_exit_cf as CF
    sp, te = _ms(r.get("entry_frenzy_spike_at")), _ms(r.get("opened_at"))
    if sp is None or te is None:
        return None, "no_spike_stamp"
    H, src = CF.reconstruct_high(r.get("pair"), sp, te, tcache)
    if H:
        return float(H), src
    w = CF.h_window(sp, te)
    if w is None:
        return None, "none"
    try:
        H = _h_from_5m(str(r.get("pair")), w[0], w[1], now_ms, bud)
    except Exception as e:
        bud.notes.append(f"{r.get('pair')} 5m H failed ({str(e)[:60]})")
        H = None
    return (H, "5m_fetch") if H else (None, "none")


# ─────────────────────────── cohort ───────────────────────────
def read_sources(export_glob=EXPORT_GLOB, archive_glob=ARCHIVE_GLOB):
    import glob
    fr = []
    for f in _EG.glob_exports(export_glob):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        fr.append(d.assign(_m=os.path.getmtime(f), _src=os.path.basename(f)))
    for f in sorted(glob.glob(archive_glob)):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        fr.append(d.assign(_m=0.0, _src=os.path.basename(f)))           # an archive never beats a fresh export row
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=list(COLS) + ["_m", "_src"])


def dedup(raw):
    o = raw.reindex(columns=list(dict.fromkeys(list(COLS) + list(raw.columns))))
    if not len(o):
        return o.assign(_k=pd.Series(dtype=object))
    o = o[o.entry_strategy.astype(str).isin(SLEEVE_OF) & (o.direction.astype(str) == "LONG")].copy()
    if "_m" not in o:
        o["_m"] = 0.0
    o["_k"] = o.opened_at.astype(str).str[:19]
    o["_c"] = o.status.astype(str).str.upper().eq("CLOSED")
    return o.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last")


def cohort(raw, floor_ms):
    """→ dict(gate = closed gate-cohort fills · ref = closed reference fills (before the floor; all when floor_ms is None) · open = OPEN
    gate-cohort rows · ox = operator-closed rows)."""
    o = dedup(raw)
    if len(o):
        o["te"] = [_ms(x) for x in o._k]
        o = o[o.te.notna()].copy()
        o["te"] = o.te.astype("int64")
        o["sleeve"] = o.entry_strategy.astype(str).map(SLEEVE_OF)
    c = o[o._c] if len(o) else o
    op = o[~o._c & (o.status.astype(str).str.upper() == "OPEN")] if len(o) else o
    c, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    if floor_ms is None:
        return dict(gate=c.iloc[0:0], ref=c, open=op.iloc[0:0], ox=ox)
    return dict(gate=c[c.te >= floor_ms], ref=c[c.te < floor_ms], open=op[op.te >= floor_ms], ox=ox)


# ─────────────────────────── pricing ───────────────────────────
def price(c, now_ms, path_fn=None, h_fn=None, fetch=True, era=None):
    """→ c + per-fill columns: H, h_src, T_v2b, dist_H, hours, vm, zone, tp_today / stop_today / rep / rep_why (TODAY — the Δ baseline),
    tp_live / stop_live / rep_era / rep_era_why (the era replica — parity only), live, pstatus, sh_<X>, d_<X> (= shadow − TODAY), complete."""
    c = c.copy().reset_index(drop=True)
    rows = c.to_dict("records")
    bud, tcache, cache = _Bud(fetch), {}, {}
    notes = []
    if path_fn is None:
        import scout_frenzy_willy_exit as FX
        import scout_willy_timecap as WT

        def path_fn(r):
            E = _f(r.get("entry_price"))
            if not (E and E > 0):
                return None
            return FX.load_path(str(r["pair"]), int(r["te"]), E, CAP_LIVE, cache)
        if fetch and len(c):
            try:
                fr = pd.DataFrame(dict(ts=pd.to_datetime(c.te, unit="ms"), pair=c.pair.astype(str), te=c.te))
                fr = fr[(fr.te + (CAP_LIVE + 2) * MIN <= now_ms) & (fr.te >= now_ms - 7 * DAY)]     # recent fills only (caches serve the rest)
                src = []
                for r in c.loc[fr.index].to_dict("records"):                                     # the REAL path source (ticks / 1m) when known
                    try:
                        P0 = path_fn(r)
                        src.append(FX.path_src(P0) if P0 is not None and len(P0["t"]) else None)
                    except Exception:
                        src.append(None)
                fr["src"] = src
                notes += WT.ensure_data(fr, CAP_LIVE, now_ms, fetch=True, max_req=2)
                cache.clear()                                                                     # re-read what the fetch added
            except Exception as e:
                notes.append(f"tick / 1m fetch pass skipped: {str(e)[:80]}")
    if h_fn is None:
        def h_fn(r):
            return resolve_h(r, now_ms, bud, tcache)
    res = {}
    for i in sorted(range(len(rows)), key=lambda i: -int(rows[i]["te"])):     # newest first: the 5m budget serves the newest fills
        r = rows[i]
        rec = dict(live=_f(r.get("pnl_percentage")))
        tp, st, tsrc = live_exit(r, era)
        tt, ts = today_exit(r)
        rec.update(tp_live=tp, stop_live=st, tp_src=tsrc, tp_today=tt, stop_today=ts,
                   zone=(lite_zone(r) if r["sleeve"] == "LITE" else ""), vm=_f(r.get("entry_frenzy_vol_mult")))
        sp = _ms(r.get("entry_frenzy_spike_at"))
        rec["hours"] = (int(r["te"]) - sp) / HOUR if sp is not None else float("nan")
        try:
            H, hs = h_fn(r)
        except Exception:
            H, hs = None, "none"
        T, traw = v2b_target(r, H)
        rec.update(H=H, h_src=hs, T_v2b=T, dist_H=(traw if traw is not None else float("nan")))
        try:
            P = path_fn(r)
        except Exception:
            P = None
        if P is None or not len(P.get("t", [])):
            rec.update(rep=float("nan"), rep_why="NO_PATH", rep_era=float("nan"), rep_era_why="NO_PATH", pstatus="NOT_EVALUABLE")
            for k in ALL_SHADOWS:
                rec[f"sh_{k}"] = float("nan")
            res[i] = rec
            continue
        w = walk_live(P, tt, ts)                         # TODAY (the Δ baseline)
        we = walk_live(P, tp, st)                        # the era replica (parity only)
        rec.update(rep=(float(w["pct"]) if w["why"] != "UNRES" else float("nan")), rep_why=w["why"],
                   rep_era=(float(we["pct"]) if we["why"] != "UNRES" else float("nan")), rep_era_why=we["why"])
        for k in ALL_SHADOWS:
            rec[f"sh_{k}"] = shadow_pct(P, k, r["sleeve"], T)
        rec["pstatus"] = "PRICED" if w["why"] != "UNRES" else "PENDING"
        res[i] = rec
    R = pd.DataFrame([res[i] for i in range(len(rows))], index=c.index) if rows else pd.DataFrame(index=c.index)
    c = pd.concat([c, R], axis=1)
    if len(c):
        for k in ALL_SHADOWS:
            c[f"d_{k}"] = pd.to_numeric(c[f"sh_{k}"], errors="coerce") - pd.to_numeric(c.rep, errors="coerce")
        c["complete"] = [bool(pd.notna(r["rep"]) and all(pd.notna(r[f"sh_{k}"]) for k in SHADOWS[r["sleeve"]]))
                         for r in c.to_dict("records")]
        c["bucket"] = [dist_bucket(_f(x)) for x in c.dist_H]
        c["day"] = pd.to_datetime(c.te, unit="ms").dt.strftime("%Y-%m-%d")
    notes += bud.notes + ([f"5m H pages: {bud.n} requested · stop: {bud.stop}"] if bud.n or bud.stop else [])
    c.attrs["notes"] = notes
    return c


# ─────────────────────────── stats ───────────────────────────
def day_boot(d, day, n=BOOT_N, seed=BOOT_SEED):
    """day-block bootstrap of the mean → array of resampled means (empty when < 2 days)."""
    d, day = np.asarray(d, float), np.asarray(day)
    u = np.unique(day)
    if len(d) == 0 or len(u) < 2:
        return np.array([])
    groups = [d[day == x] for x in u]
    sums, cnts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n, len(u)))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def bonf_lb(d, day, m):
    b = day_boot(d, day)
    return float(np.quantile(b, ALPHA / max(1, m))) if len(b) else float("nan")


def top_share(d, key):
    """share of Σ Δ carried by the largest contributor (day / pair); nan when Σ Δ ≤ 0."""
    s = pd.Series(np.asarray(d, float)).groupby(np.asarray(key)).sum()
    tot = float(s.sum())
    return float(s.max() / tot) if tot > 0 else float("nan")


def parity(g):
    """the ERA replica (the exit the fill ran) vs the booked % → (share or None, text)."""
    p = g[g.rep_era.notna() & g.live.notna()] if len(g) else g
    if not len(p):
        return None, "parity: no priced fill"
    ok = (pd.to_numeric(p.rep_era) - pd.to_numeric(p.live)).abs() <= PARITY_TOL
    sh = float(ok.mean())
    bad = p[~ok.values]
    who = "" if not len(bad) else " · off: " + ", ".join(
        f"{r.pair} {str(r.opened_at)[5:16]} booked {_fmt(r.live)} {r.close_reason} vs replica {_fmt(r.rep_era)} {r.rep_era_why}"
        for r in bad.head(6).itertuples())
    return sh, (f"parity (era replica vs booked, ±{PARITY_TOL}): {int(ok.sum())}/{len(p)} = {sh * 100:.0f} %"
                + ("" if sh >= PARITY_MIN else f" < {PARITY_MIN * 100:.0f} % → no verdict (parity)") + who)


def _complete(g):
    """the fills on which TODAY and EVERY shadow of the sleeve resolved — the SAME set for every shadow (review fix 4)."""
    return g[g.complete.astype(bool)] if len(g) else g


def _paired(g, k):
    return _complete(g)


def stat_row(g, k, m):
    p = _paired(g, k)
    if not len(p):
        return f"    {k:8s} n 0"
    sh, rp, d = pd.to_numeric(p[f"sh_{k}"]), pd.to_numeric(p.rep), pd.to_numeric(p[f"d_{k}"])
    b = day_boot(d.values, p.day.values)
    ci = f"[{np.quantile(b, 0.025):+.2f}, {np.quantile(b, 0.975):+.2f}]" if len(b) else "[n/a]"
    lb = bonf_lb(d.values, p.day.values, m)
    tsd, tsp = top_share(d.values, p.day.values), top_share(d.values, p.pair.astype(str).values)
    return (f"    {k:8s} N {len(p)} · {p.day.nunique()} d | shadow WR {(sh > 0).mean() * 100:.0f} % · avg {sh.mean():+.3f} · Σ {sh.sum():+.2f} | "
            f"TODAY WR {(rp > 0).mean() * 100:.0f} % · avg {rp.mean():+.3f} · Σ {rp.sum():+.2f} | Δ avg {d.mean():+.3f} · Σ {d.sum():+.2f} · "
            f"better {int((d > 1e-9).sum())} / worse {int((d < -1e-9).sum())} · day-boot 95 % {ci} · Bonferroni LB (α {ALPHA}/{m}) "
            + (f"{lb:+.3f}" if lb == lb else "n/a") + " · top day "
            + (f"{tsd:.0%}" if tsd == tsd else "n/a") + " · top pair " + (f"{tsp:.0%}" if tsp == tsp else "n/a") + " of Σ Δ")


def split_line(g, k, col, labels):
    parts = []
    for lab in labels:
        p = _paired(g[g[col] == lab], k)
        d = pd.to_numeric(p[f"d_{k}"]) if len(p) else pd.Series(dtype=float)
        parts.append(f"{lab} n {len(p)} Δ " + (f"{d.mean():+.3f}" if len(d) else "n/a"))
    return f"      {k} by {col}: " + " · ".join(parts)


def theory_check(g, k="V2B"):
    near, far = _paired(g[g.bucket == "near<3"], k), _paired(g[g.bucket == "far>=6"], k)
    if not len(near) or not len(far):
        return f"      T1 check ({k}): near n {len(near)} / far n {len(far)} — n/a"
    a, b = float(near[f"d_{k}"].mean()), float(far[f"d_{k}"].mean())
    return f"      T1 check ({k}): near-H Δ {a:+.3f} (n {len(near)}) vs far-H Δ {b:+.3f} (n {len(far)}) → " + (
        "near > far (consistent with T1)" if a > b else "near ≤ far (T1 NOT supported)")


def t2_check(g, sleeve):
    ks = [k for k in SHADOWS[sleeve] if k.startswith("TP")]
    p = _complete(g)
    if not len(p):
        return "      T2 check: n/a"
    av = {k: float(pd.to_numeric(p[f"d_{k}"]).mean()) for k in ks}
    if sleeve == "LONG":
        bad = [k for k, v in av.items() if v > 0]
        return "      T2 check (LONG, predicted worse than TODAY's +1.5 / −4): " + " · ".join(f"{k} Δ {v:+.3f}" for k, v in av.items()) + (
            f" → REFUTES T2 ({', '.join(bad)} beat +1.5)" if bad else " → CONSISTENT with T2 (no higher TP beats +1.5)")
    return "      T2 check (LITE, how far winners run vs TODAY's +3 / −3): " + " · ".join(f"{k} Δ {v:+.3f}" for k, v in av.items())


def block_lines(c, title):
    L = [f"### {title}"]
    if not len(c):
        return L + ["  0 fills"]
    for s in SLEEVES:
        g = c[c.sleeve == s]
        if not len(g):
            L.append(f"  {s}: 0 fills")
            continue
        st = g.pstatus.value_counts().to_dict()
        hn = int(pd.to_numeric(g["T_v2b"], errors="coerce").notna().sum())
        L.append(f"  {s}: {len(g)} fills on {g.day.nunique()} days · complete (TODAY + every shadow) {int(g.complete.sum())} · priced "
                 f"{st.get('PRICED', 0)} · pending {st.get('PENDING', 0)} · not evaluable (no path) {st.get('NOT_EVALUABLE', 0)} · H found {hn} ("
                 + ", ".join(f"{k} {v}" for k, v in g.h_src.value_counts().items()) + ") · booked avg "
                 + (f"{pd.to_numeric(g.live).mean():+.3f}" if g.live.notna().any() else "n/a"))
        L.append("    " + parity(g)[1])
        m = len(SHADOWS[s])
        for k in SHADOWS[s]:
            L.append(stat_row(g, k, m))
        L.append("    splits (avg Δ vs TODAY, complete fills):")
        for k in SHADOWS[s]:
            L.append(split_line(g, k, "bucket", ("near<3", "mid3-6", "far>=6")))
            if s == "LITE":
                L.append(split_line(g, k, "zone", ("zone", "out", "unread")))
        L.append(theory_check(g, "V2B"))
        if s == "LITE":
            L.append(theory_check(g, "V2B_OWN"))
        L.append(t2_check(g, s))
    L.append(f"  per fill (newest {LIST_CAP}): " + " · ".join(
        f"{r.sleeve}{'-R' if str(r.entry_strategy).endswith('RETRY') else ''} {r.pair} {str(r.opened_at)[5:16]} distH {_fmt(r.dist_H)} "
        + f"{_fmt(r.hours, '{:.1f}')}h vm {_fmt(r.vm, '{:.0f}')} {r.zone} | booked {_fmt(r.live)} era {_fmt(r.rep_era)} "
        + f"({r.tp_live:g}/-{r.stop_live:g}) TODAY {_fmt(r.rep)} ({r.tp_today:g}/-{r.stop_today:g}) | "
        + " ".join(f"{k} {getattr(r, f'sh_{k}'):+.2f}" for k in SHADOWS[r.sleeve] if getattr(r, f"sh_{k}") == getattr(r, f"sh_{k}"))
        + ("" if r.pstatus == "PRICED" else f" [{r.pstatus}]") + ("" if r.complete else " [incomplete]")
        for r in c.sort_values("te", ascending=False).head(LIST_CAP).itertuples()))
    return L


# ─────────────────────────── frozen gate ───────────────────────────
def gate_reads(c, open_rows, now_ms):
    """→ {f'{sleeve}_{shadow}': (verdict, text, complete)} on the GATE cohort. ONE prefix per sleeve (the complete fills, open order) —
    identical for every shadow; parity read on that prefix; NO_VERDICT is a decision (persisted)."""
    out = {}
    for s in SLEEVES:
        g = (c[c.sleeve == s] if len(c) else c)
        g = g.sort_values(["te", "pair"], kind="stable") if len(g) else g
        m = len(SHADOWS[s])
        op = open_rows[open_rows.sleeve == s] if len(open_rows) else open_rows
        p = _complete(g)
        pre = None
        for j in range(len(p)):
            q = p.iloc[:j + 1]
            if len(q) >= GATE_N and q.day.nunique() >= GATE_DAYS:
                pre = q
                break
        inc = g[~g.complete.astype(bool)] if len(g) else g
        if pre is None:
            for k in SHADOWS[s]:
                out[f"{s}_{k}"] = ("COLLECTING", f"{len(p)}/{GATE_N} complete fills on {p.day.nunique() if len(p) else 0}/{GATE_DAYS} days · "
                                                 f"{len(inc)} incomplete (pending / not evaluable)", False)
            continue
        end = int(pre.te.max())
        inc = inc[inc.te <= end] if len(inc) else inc
        young = inc[now_ms - inc.te < PENDING_DAYS * DAY] if len(inc) else inc
        n_open = int((op.te <= end).sum()) if len(op) else 0
        if n_open or len(young):
            for k in SHADOWS[s]:
                out[f"{s}_{k}"] = ("HOLD", f"prefix {len(pre)} complete fills: {n_open} still open · {len(young)} incomplete < {PENDING_DAYS} "
                                           f"days old (path / H pending) — held", False)
            continue
        ne = (f" · {len(inc)} not evaluable (incomplete ≥ {PENDING_DAYS} days: "
              + ", ".join(f"{r.pair} {str(r.opened_at)[5:16]}" for r in inc.head(5).itertuples()) + ")") if len(inc) else ""
        sh, _ = parity(pre)
        par_ok = sh is not None and sh >= PARITY_MIN
        for k in SHADOWS[s]:
            d = pd.to_numeric(pre[f"d_{k}"])
            lb = bonf_lb(d.values, pre.day.values, m)
            ts = top_share(d.values, pre.day.values)
            ok = float(d.mean()) >= MIN_GAIN and lb == lb and lb > 0 and par_ok and ts == ts and ts < DAY_SHARE
            txt = (f"{len(pre)} fills on {pre.day.nunique()} days: avg Δ vs TODAY {d.mean():+.3f} · Bonferroni LB (α {ALPHA}/{m}) {lb:+.3f} · "
                   f"top day " + (f"{ts:.0%}" if ts == ts else "n/a") + f" · prefix parity {sh * 100 if sh is not None else float('nan'):.0f} %"
                   + ne + " → ")
            if not par_ok:
                out[f"{s}_{k}"] = ("NO_VERDICT", txt + "no verdict (parity)", True)
                continue
            out[f"{s}_{k}"] = (("CANDIDATE" if ok else "KEEP"),
                               txt + ("CANDIDATE — OPERATOR CALL (never automatic)" if ok else "KEEP the live exit"), True)
    return out


DECIDED = ("CANDIDATE", "KEEP", "NO_VERDICT")


def frozen(state_path, name, v, txt, complete):
    """🔒 decided ONCE: the first complete CANDIDATE / KEEP / NO_VERDICT is persisted (atomic temp + os.replace). An existing state file
    that cannot be parsed is NEVER overwritten (warning; nothing frozen this run). → (verdict, text)."""
    st = {}
    if state_path and os.path.exists(state_path):
        try:
            with open(state_path) as fh:
                st = json.load(fh)
            if not isinstance(st, dict):
                raise ValueError("not a JSON object")
        except Exception as e:
            print(f"⚠ FRENZY_V2B_SHADOW: state file {state_path} unreadable ({str(e)[:60]}) — left untouched, nothing frozen this run",
                  file=sys.stderr)
            return v, txt + " (⚠ state file unreadable — not frozen)"
    if name in st:
        return st[name]["verdict"], st[name]["text"] + f" (frozen {str(st[name].get('at', ''))[:16]} UTC)"
    if state_path and complete and v in DECIDED:
        try:
            st[name] = {"verdict": v, "text": txt, "at": pd.Timestamp.now("UTC").isoformat()}
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            tmp = f"{state_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(st, fh, indent=1)
            os.replace(tmp, state_path)
        except Exception as e:
            print(f"⚠ FRENZY_V2B_SHADOW: could not persist {name} ({str(e)[:60]})", file=sys.stderr)
    return v, txt


def write_csv(c, path=CSV_OUT):
    if not path or not len(c):
        return
    if not os.path.abspath(path).startswith(os.path.join(_ROOT, "reports") + os.sep):
        raise ValueError("FRENZY_V2B_SHADOW writes only under reports/")
    keep = ["_k", "pair", "entry_strategy", "sleeve", "era", "day", "pnl_percentage", "close_reason", "tp_today", "stop_today", "rep",
            "rep_why", "tp_live", "stop_live", "tp_src", "rep_era", "rep_era_why", "complete", "H", "h_src", "T_v2b", "dist_H", "bucket", "hours", "vm", "zone", "pstatus"] + [f"sh_{k}" for k in ALL_SHADOWS] + [
            f"d_{k}" for k in ALL_SHADOWS]
    c.reindex(columns=keep).to_csv(path, index=False)


# ─────────────────────────── run ───────────────────────────
def run(now_ms=None, raw=None, floor_ms="auto", path_fn=None, h_fn=None, fetch=True, state_path=STATE, csv_path=CSV_OUT, era=None):
    now_ms = int(now_ms or pd.Timestamp.now("UTC").value // 1_000_000)
    L = [f"## 🎲🪶 FRENZY_V2B_SHADOW — LITE / LONG on WILLY's V2b exit + fixed higher TPs (DECISION_LOG {DL}, observe-only; frozen "
         f"gate, OPERATOR CALL)", ""] + THEORY + [
        f"Shadows: LITE {', '.join(SHADOWS['LITE'])} · LONG {', '.join(SHADOWS['LONG'])}. V2B = T max(+1.25, min(+8, net@H×0.99)) / −4 / "
        f"late +0.25 @30 / 90 · V2B_OWN = same with −3 · TPx = fixed +x / today's sleeve stop (LITE −3 · LONG −4) / 12 h. Δ = shadow − "
        f"TODAY's live exit re-walked on the same path (LITE: +3 / −3 · LONG +1.5 / −4); the era replica is the parity check "
        f"only. Numbers move as paths arrive (1m tails → ticks, H pages)."]
    if raw is None:
        raw = read_sources()
    if floor_ms == "auto":
        floor_ms = _GK.dl_deploy_ms(DL, lag_min=LAG_MIN, label="FRENZY_V2B_SHADOW")
    co = cohort(raw, floor_ms)
    e250 = (eras() if era is None else era).get(250)
    ref0 = co["ref"]
    trail = ref0[ref0.te < e250] if (e250 is not None and len(ref0)) else ref0.iloc[0:0]
    ref0 = ref0[ref0.te >= e250] if (e250 is not None and len(ref0)) else ref0
    both = pd.concat([co["gate"].assign(era="GATE"), ref0.assign(era="REF")], ignore_index=True)
    both = price(both, now_ms, path_fn=path_fn, h_fn=h_fn, fetch=fetch, era=era) if len(both) else both
    for n in both.attrs.get("notes", []) if len(both) else []:
        L.append(f"  (data) {n}")
    L.append(f"Operator-closed (excluded): {len(co['ox'])} · pre-DL 250 lock / trail-era fills (exit not replicable, reference only, "
             f"excluded): {len(trail)}")
    g = both[both.era == "GATE"] if len(both) else both
    ref = both[both.era == "REF"] if len(both) else both
    L.append("")
    if floor_ms is None:
        L.append(f"GATE cohort: NOT evaluable yet — no DECISION_LOG {DL} commit found (the cohort starts at its deploy + {LAG_MIN} min).")
    else:
        L.append(f"GATE cohort (opened ≥ {pd.Timestamp(floor_ms, unit='ms'):%Y-%m-%d %H:%M} UTC = the DL {DL} commit + {LAG_MIN} min): "
                 f"{len(g)} closed · {len(co['open'])} open")
        L += block_lines(g, "GATE cohort (counted)")
    L.append("")
    L += block_lines(ref, "MASTER / ARCHIVED REFERENCE — IN-SAMPLE (before the gate floor; NOT counted, never frozen)")
    L += ["", f"🔒 FROZEN gate (per sleeve × shadow: ≥ {GATE_N} complete fills (same set for every shadow) on ≥ {GATE_DAYS} days · avg Δ ≥ +{MIN_GAIN} · day-boot LB > 0 "
               f"at α {ALPHA}/m (LITE m {len(SHADOWS['LITE'])}, LONG m {len(SHADOWS['LONG'])}) · parity ≥ {PARITY_MIN:.0%} · top day < "
               f"{DAY_SHARE:.0%} of Σ Δ → OPERATOR CALL, never automatic):"]
    if floor_ms is None:
        L.append("  not evaluable (no DL 379 commit yet)")
    else:
        for name, (v, txt, comp) in gate_reads(g, co["open"], now_ms).items():
            fv, ft = frozen(state_path, name, v, txt, comp)
            L.append(f"  {name}: {fv} — {ft}")
    try:
        write_csv(both, csv_path)
    except Exception as e:
        L.append(f"  (csv not written: {str(e)[:80]})")
    L.append("")
    return L


if __name__ == "__main__":
    print("\n".join(run(fetch="--no-fetch" not in sys.argv)))
