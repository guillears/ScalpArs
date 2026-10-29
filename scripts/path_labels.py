#!/usr/bin/env python3
"""🧭 PATH LABELS + LOSER / WINNER DECOMPOSITION (DECISION_LOG 406 — the analysis methodology, operator 2026-10-28: "exits cannot be
independently looked at from entries … we should optimize both").

The booked P&L mixes two questions — was the ENTRY right, and did the EXIT take what was there. Every fill is read from its own price path
(1m bars, GROSS % from the entry price, direction-aware), never from the exit it happened to run:

  ENTRY label (exit-independent; the winners-vs-losers screens' primary target):
    WORKED  — +X % our way (X = ENTRY_X 1.0, operator 10-28: "+1 % is real money a small target could take") BEFORE −Y against us
    NEVER   — −Y first, OR the fill's own booked STOP closed it (a loss) before any +X (review H3: a live stop is the −Y passage)
    FLAT    — neither within HORIZON_MIN (24 h) · AMBIG — one 1m bar touched both, or the ENTRY bar alone crossed a line and the fill
              carries no peak / trough stamp to bound it (review H2: the entry minute holds prints from BEFORE the fill) · PENDING
    Y = the sleeve's stop distance, gross (net − FEE_RT): TODAY's config stop for the fixed-stop sleeves (STOP_KEYS), else the median
        |booked %| of the sleeve's LOSING stop closes (STOP_REASONS, as traded — never re-priced %, never a trailing stop) over the master
        + every export (one ruler per sleeve per run, cohort-independent — review H1); DEFAULT_Y only when neither exists.
  REACH (continuous, operator 10-28 "how far it went"): the best gross move our way before the −Y passage / the stop close / 24 h.
  EXIT label: best = the best move our way to the close + BEST_POST_MIN (45 min); left = best − FEE_RT − booked; capture = booked ÷ net best.

  DECOMPOSITION group (one per fill — which side to study first):
    losers (booked ≤ 0):  NEVER_WORKED · STOPPED_RECOVERED (NEVER, but +X from the entry within POST_EXIT_MIN after the close) ·
                          GAVE_BACK (+X first, still lost) · DRIFT_LOSS (neither) · AMBIG / PENDING
    winners (booked > 0): LEFT_ON_TABLE (left ≥ max(LEFT_MIN_PTS, booked)) · CAPTURED

Entry bar (bar 0): its favourable / adverse extreme is bounded by the fill's own stamped peak_pnl / trough_pnl (+ FEE_RT, gross) when
present; no stamps and bar 0 alone crosses a line → AMBIG.
Bars: the shared 1m caches (scout_willy_timecap.load_1m) — missing windows fetched only with fetch=True (public fapi klines, limit
1000 = weight 5; per-run budget: ≤ MAX_REQ requests, ≤ BUDGET_S wall-clock, stop at used weight ≥ WEIGHT_STOP, on a missing weight
header, on the first network error or a 418 / 429 cooldown; an empty window is retried at most twice ever). The cache
(reports/.path_labels_cache.json) holds PATH FACTS only (label · pass · reach · best · recovered), keyed by LABEL_VERSION + the fill +
its close + X / Y + its stop flag; the group / left / capture are recomputed from the CURRENT booked % on every call (review M1).

Usage: venv/bin/python -I scripts/path_labels.py [orders.csv …] [--fetch] [--since 2026-10-23T18:30:00]
       (no file → the newest ~/Downloads/scalpars_orders_*.csv; --since defaults to sweep_separators.BATCH_START)
"""
import functools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import review_math as RM  # noqa: E402

LABEL_VERSION = 3
ENTRY_X = 1.0
DEFAULT_Y = 3.0
FEE_RT = 0.09                  # taker in + out ≈ 0.045 × 2: net booked % ≈ gross path % − FEE_RT
MIN_STOPS_FOR_Y = 3
HORIZON_MIN = 1440
POST_EXIT_MIN = 120
BEST_POST_MIN = 45
LEFT_MIN_PTS = 1.0
MIN = 60_000
MAX_REQ = 120
BUDGET_S = 60.0
WEIGHT_STOP = 900
KLINE_LIMIT = 1000
CACHE = os.path.join(ROOT, "reports", ".path_labels_cache.json")
CONFIG = os.path.join(ROOT, "trading_config.json")
MASTER = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
GROUPS = ("NEVER_WORKED", "STOPPED_RECOVERED", "GAVE_BACK", "DRIFT_LOSS", "LEFT_ON_TABLE", "CAPTURED", "AMBIG", "PENDING")
LOSER_GROUPS = GROUPS[:4]
PATH_COLS = ("path_label", "pass_min", "reach", "best", "capture", "left", "recovered", "group", "path_y", "path_y_src")
STOP_REASONS = ("STOP_LOSS", "BR_STOP_LOSS", "FLIP_STOP_LOSS", "RH_HARD_STOP", "STOP_LOSS_WIDE", "PATTERN_FIXED_SL", "SPIKE_SL")   # never a TRAILING stop
GAP_TOL_MIN = 5          # an exchange-missing minute (≤ 5 in a row) never ends the path — no fill stays PENDING forever on one hole
Y_ROUND = 0.05           # Y rounded so a new export cannot nudge every cached key
STOP_KEYS = {"FRENZY_WILLY": "frenzy_willy_stop_pct", "FRENZY_WILLY_HT": "frenzy_willy_stop_pct", "FRENZY_LITE": "frenzy_stop_pct",
             "FRENZY_LITE_RETRY": "frenzy_stop_pct", "FRENZY_WIDE": "frenzy_stop_pct", "FRENZY_LONG": "frenzy_long_stop_pct",
             "FLATBASE_SHORT": "flatbase_stop_pct", "SPIKE_FADE": "spike_fade_sl_pct", "SPIKE_BOUNCE": "spike_bounce_sl_pct",
             "SPIKE_CHASE": "spike_sl_pct"}
Y_FAMILY = {"BEARRUN_SHORT": "MOMENTUM", "BULL_LONG": "MOMENTUM", "BOUNCE_LONG": "MOMENTUM"}   # momentum-stack sleeves: no stop of their own


# ───────────────────────────── pure core ─────────────────────────────
def fav_adv(o, h, l, entry, direction):
    """1m bars → (favourable, adverse) gross % per bar from the entry price (LONG: high / low · SHORT: low / high)."""
    h, l, e = np.asarray(h, float), np.asarray(l, float), float(entry)
    if str(direction).upper() == "SHORT":
        return (1 - l / e) * 100, (1 - h / e) * 100
    return (h / e - 1) * 100, (l / e - 1) * 100


def first_passage(fav, adv, x, y):
    """→ (label, bar index) — WORKED: fav ≥ x before adv ≤ −y · NEVER: the reverse · AMBIG: the same bar · FLAT: neither."""
    up = np.flatnonzero(np.asarray(fav) >= x)
    dn = np.flatnonzero(np.asarray(adv) <= -y)
    iu = int(up[0]) if len(up) else None
    idn = int(dn[0]) if len(dn) else None
    if iu is None and idn is None:
        return "FLAT", None
    if idn is None or (iu is not None and iu < idn):
        return "WORKED", iu
    if iu is None or idn < iu:
        return "NEVER", idn
    return "AMBIG", iu


def group(label, booked, best, recovered):
    """the decomposition group of one fill. Pure."""
    if label in ("PENDING", "AMBIG"):
        return label
    if booked is None or not np.isfinite(booked):
        return "PENDING"
    if booked > 0:
        if best is not None and np.isfinite(best) and best - booked >= max(LEFT_MIN_PTS, booked):
            return "LEFT_ON_TABLE"
        return "CAPTURED"
    if label == "WORKED":
        return "GAVE_BACK"
    if label == "NEVER":
        return "STOPPED_RECOVERED" if recovered else "NEVER_WORKED"
    return "DRIFT_LOSS"


def next_study(entry_loss, exit_loss):
    """the DL 406 pointer: the side with the larger realised loss goes first (losses are ≤ 0). Pure."""
    if not entry_loss and not exit_loss:
        return "—"
    return "ENTRY first" if entry_loss <= exit_loss else "EXIT first"


def is_stop(close_reason):
    return str(close_reason or "").strip().upper().split(" ")[0] in STOP_REASONS


@functools.lru_cache(maxsize=4)
def _stop_table(master=MASTER, cfg=CONFIG, exports_key=()):
    """sleeve → (Y gross, source): today's config stop for STOP_KEYS sleeves, else the median |as-traded booked %| of the sleeve's
    LOSING stop closes over the master + the given exports (≥ MIN_STOPS_FOR_Y), minus FEE_RT. Cached per run."""
    out = {}
    try:
        with open(cfg) as f:
            th = json.load(f).get("thresholds", {})
        for sl, k in STOP_KEYS.items():
            v = th.get(k)
            if k == "frenzy_long_stop_pct" and (v is None or not abs(float(v)) > 0):   # 0 = the engine falls back to frenzy_stop_pct
                k, v = "frenzy_stop_pct", th.get("frenzy_stop_pct")
            if v is not None and abs(float(v)) > 0:
                out[sl] = (_yround(abs(float(v)) - FEE_RT), f"config {k}")
    except Exception:
        pass
    fr = []
    for f in ([master] if master and os.path.exists(master) else []) + list(exports_key):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("entry_strategy", "close_reason", "pnl_percentage", "status",
                                                                                "opened_at", "pair", "direction")))
        except Exception:
            continue
    if fr:
        a = pd.concat(fr, ignore_index=True)
        if {"opened_at", "pair", "direction"} <= set(a.columns):   # every export snapshot repeats fills → one row per fill (CLAUDE.md dedup key)
            a = a.assign(_k=a.opened_at.astype(str).str.replace(" ", "T").str[:19]).drop_duplicates(["_k", "pair", "direction"], keep="first")
        a["pnl_percentage"] = pd.to_numeric(a.pnl_percentage, errors="coerce")
        a = a[a.close_reason.map(is_stop) & (a.pnl_percentage < 0)]
        for sl, g in a.groupby(a.entry_strategy.astype(str)):
            if sl not in out and len(g) >= MIN_STOPS_FOR_Y:
                out[sl] = (_yround(float(g.pnl_percentage.abs().median()) - FEE_RT), f"its {len(g)} losing stops")
    return out


def _yround(v):
    return round(max(Y_ROUND, round(float(v) / Y_ROUND) * Y_ROUND), 2)


def stop_y(sleeve, table=None):
    """the sleeve's Y (gross) and its source — cohort-independent (review H1). Pure on `table`."""
    t = table if table is not None else _stop_table(exports_key=tuple(_exports()))
    sl = str(sleeve)
    if sl not in t and Y_FAMILY.get(sl) in t:            # review 2 F7: a momentum-stack sleeve borrows the momentum ruler
        y, src = t[Y_FAMILY[sl]]
        return max(0.05, float(y)), f"{src} (momentum stack)"
    y, src = t.get(sl, (DEFAULT_Y, "default"))
    return max(0.05, float(y)), src


def label_from_bars(bars, te, tc, entry, direction, booked, x, y, now_ms=None, stopped=False, peak=None, trough=None):
    """bars = DataFrame open_time / h / l (1m, sorted). te / tc = open / close ms. stopped = the fill closed on its own STOP at a loss;
    peak / trough = its stamped peak_pnl / trough_pnl (net %) or None. → dict(label, pass_min, reach, best, recovered) — path facts only."""
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    te, tc = int(te), int(tc)
    b0 = (te // MIN) * MIN
    h_end = te + HORIZON_MIN * MIN
    end_need = max(h_end, tc + POST_EXIT_MIN * MIN)
    b = bars[(bars.open_time >= b0) & (bars.open_time < end_need)] if len(bars) else bars
    ot = b.open_time.values.astype("int64") if len(b) else np.array([], dtype="int64")
    k = 0                                               # the prefix from the entry bar; a hole ≤ GAP_TOL_MIN minutes is skipped
    if len(ot) and ot[0] == b0:
        k = 1
        while k < len(ot) and ot[k] - ot[k - 1] <= (GAP_TOL_MIN + 1) * MIN:
            k += 1
    out = dict(label="PENDING", pass_min=None, reach=None, best=None, recovered=None)
    if not k:
        return out
    b = b.iloc[:k]
    t = b.open_time.values.astype("int64")
    fav, adv = fav_adv(None, b.h.values, b.l.values, entry, direction)
    fav, adv = fav.copy(), adv.copy()
    stamped = peak is not None and trough is not None and np.isfinite(peak) and np.isfinite(trough)
    if stamped:                                          # bar 0 bounded by what the fill itself saw (prints before the fill excluded)
        fav[0] = min(fav[0], float(peak) + FEE_RT)
        adv[0] = max(adv[0], float(trough) + FEE_RT)
    covered_to = int(t[-1]) + MIN
    hz = t < h_end
    lab, i = first_passage(fav[hz], adv[hz] - 1e-9, x, y)
    if i == 0 and lab in ("WORKED", "NEVER", "AMBIG"):   # the entry minute alone decided it: never trusted on its own (review 2)
        if not stamped:
            lab = "AMBIG"                                # nothing bounds the pre-fill prints
        else:                                            # review 2 F2: the fill's own stamps settle the order first
            up = float(peak) + FEE_RT >= x
            tr_dn = float(trough) + FEE_RT <= -y + 1e-9
            if up and (not tr_dn or stopped):
                lab = "WORKED"                           # it saw +X; a stop ends a trade, so +X came before it
            elif (tr_dn or stopped) and not up:
                lab = "NEVER"
            else:
                l2, i2 = first_passage(fav[hz][1:], adv[hz][1:] - 1e-9, x, y)
                if l2 != lab:
                    lab = "AMBIG"                        # both lines inside the fill's life and bars 1+ disagree → unknown
    if stopped and tc < h_end:                           # review H3: the live stop IS the −Y passage when it came first (inside 24 h)
        if covered_to > tc:
            i_stop = int(np.searchsorted(t, tc, side="right")) - 1
            after_close = i is not None and t[min(i, len(t) - 1)] > tc
            if lab in ("FLAT", "PENDING") or (lab in ("WORKED", "AMBIG") and after_close):
                lab, i = "NEVER", max(0, i_stop)
        elif lab in ("FLAT", "AMBIG") or (lab == "WORKED" and i is not None and t[i] > tc):
            lab = "PENDING"                              # bars end before the stop close — cannot place it yet
    if lab == "FLAT" and covered_to < h_end:
        lab = "PENDING"
    out.update(label=lab, pass_min=(None if i is None or lab in ("PENDING", "FLAT") else round((int(t[i]) - te) / MIN, 1)))   # F4: real bar time
    # reach: the best move our way before the adverse event (the −Y touch, the stop close) or the horizon
    r0 = 0 if stamped else 1                            # an unstamped entry minute never sets the reach (pre-fill prints)
    i_dn = np.flatnonzero(adv[hz] <= -y + 1e-9)
    stop_ix = (int(np.searchsorted(t, tc, side="right")) - 1) if (stopped and covered_to > tc and tc < h_end) else None
    ends = [int(i_dn[0])] if len(i_dn) else []
    if stop_ix is not None:
        ends.append(stop_ix)
    if ends:
        e = min(ends)
        seg = fav[r0:e + 1]
        out["reach"] = round(float(np.max(seg)), 3) if len(seg) else 0.0
    elif covered_to >= h_end:
        seg = fav[hz][r0:]
        out["reach"] = round(float(np.max(seg)), 3) if len(seg) else 0.0
    best_end = tc + BEST_POST_MIN * MIN
    if covered_to >= best_end:
        m = t < best_end
        out["best"] = round(float(np.max(fav[m])), 3) if m.any() else None
    post_end = tc + POST_EXIT_MIN * MIN
    if covered_to >= post_end:
        m = (t > (tc // MIN) * MIN) & (t < post_end)              # after the close minute (its pre-stop prints excluded)
        out["recovered"] = bool(m.any() and np.max(fav[m]) >= x)
    return out


def finish(facts, booked):
    """path facts + the CURRENT booked % → group / left / capture. Pure (review M1: never cached)."""
    lab, best, rec = facts.get("label"), facts.get("best"), facts.get("recovered")
    out = dict(left=None, capture=None)
    ok = booked is not None and np.isfinite(booked)
    if best is not None and ok:
        out["left"] = round(best - FEE_RT - booked, 3)
        nb = best - FEE_RT
        out["capture"] = round(booked / nb, 3) if nb > 0 else None
    if not ok:
        out["group"] = "PENDING"
    elif booked > 0:                                     # a winner's group is about the EXIT only — the entry label is not needed
        out["group"] = "PENDING" if best is None else group("WORKED", booked, best - FEE_RT, False)
    elif lab == "PENDING":
        out["group"] = "PENDING"
    elif lab == "AMBIG":
        out["group"] = "AMBIG"
    elif lab == "NEVER" and rec is None:
        out["group"] = "PENDING"
    else:
        out["group"] = group(lab, booked, None, bool(rec))
    return out


# ───────────────────────────── data ─────────────────────────────
def _ms(v):
    try:
        return int(pd.Timestamp(str(v).replace(" ", "T")[:26]).tz_localize(None).value // 1_000_000)
    except TypeError:
        return int(pd.Timestamp(str(v).replace(" ", "T")[:26]).tz_convert(None).value // 1_000_000)
    except Exception:
        return None


def _load_cache():
    try:
        with open(CACHE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(c):
    tmp = f"{CACHE}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(c, f)
    os.replace(tmp, CACHE)


def _key(r, x, y, stopped, stamped):
    return (f"v{LABEL_VERSION}|{str(r['opened_at']).replace(' ', 'T')[:19]}|{r['pair']}|{r['direction']}|"
            f"{str(r.get('closed_at')).replace(' ', 'T')[:19]}|{x:g}|{y:g}|{int(bool(stopped))}|{int(bool(stamped))}")


def _fnum(v):
    v = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(v) else float(v)


def _fetch_window(WT, pair, start, end, now_ms, bud, attempts):
    """fetch the missing 1m minutes of [start, end) (capped at the last closed minute) in KLINE_LIMIT chunks within the shared budget.
    → True when something was appended. Sets bud['stopped'] on any stop reason; never raises."""
    end = min(int(end), (int(now_ms) // MIN) * MIN - MIN)
    have = WT.load_1m(pair)
    got = set(have.open_time.values.astype("int64").tolist()) if have is not None and len(have) else set()
    need = [m for m in range(int(start), end, MIN) if m not in got]
    chunks, appended = [], False
    while need and not bud.get("stopped"):
        s0 = need[0]
        ak = f"{pair}|{s0}"
        if int(attempts.get(ak, 0)) >= 2:                # an empty window twice → never again (delisted / exchange gap)
            need = need[1:]                              # F5: skip only that minute, never the next 1000
            continue
        if bud["req"] >= MAX_REQ:
            bud["stopped"] = f"request cap {MAX_REQ}/run"
            break
        if time.monotonic() - bud["t0"] > BUDGET_S:
            bud["stopped"] = f"wall-clock budget {BUDGET_S:g} s"
            break
        if bud["used"] >= WEIGHT_STOP:
            bud["stopped"] = f"used weight {bud['used']} ≥ {WEIGHT_STOP}"
            break
        cd = WT.NF._cooldown_until()
        if cd > now_ms:
            bud["stopped"] = f"rate-limit cooldown until {pd.Timestamp(cd, unit='ms'):%m-%d %H:%M} UTC"
            break
        bud["req"] += 1
        try:
            d, used = WT._fetch_1m(pair, s0, KLINE_LIMIT, now_ms)
        except Exception as e:                           # review M3: the first network error stops the run's fetching
            bud["stopped"] = f"network error ({type(e).__name__}: {str(e)[:60]})"
            break
        if not used:
            bud["stopped"] = "no X-MBX-USED-WEIGHT-1M header — weight unknown, fetching stopped"
        bud["used"] = max(int(bud["used"]), int(used or 0))
        if len(d):
            chunks.append(d)
        if not len(d) or s0 not in set(d.open_time.values.astype("int64").tolist()):
            attempts[ak] = int(attempts.get(ak, 0)) + 1   # empty, or the exchange has no bar at s0 → at most twice ever
        need = [m for m in need if m >= s0 + KLINE_LIMIT * MIN]
    if chunks:
        WT._append_1m(pair, pd.concat(chunks, ignore_index=True))
        appended = True
    return appended


def label_frame(fills, fetch=False, now_ms=None, notes=None, budget=None):
    """fills (closed orders rows, any sleeves) → the same rows + PATH_COLS. Y per sleeve from stop_y (cohort-independent). Cache-first;
    fetch=True fills missing 1m windows within the SHARED per-run `budget` dict. Never raises on a bad row (→ PENDING)."""
    import scout_willy_timecap as WT
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    notes = notes if notes is not None else []
    if fills is None or not len(fills):
        return pd.DataFrame(columns=list(fills.columns if fills is not None else []) + list(PATH_COLS))
    cache = _load_cache()
    for _k in [k_ for k_ in cache if not k_.startswith("_") and not k_.startswith(f"v{LABEL_VERSION}|")]:
        cache.pop(_k, None)                              # prune keys of older label versions
    attempts = cache.setdefault("_attempts", {})
    ytab = _stop_table(exports_key=tuple(_exports()))    # once per call, not per row (review 2)
    bud = budget if budget is not None else {}
    bud.setdefault("req", 0)
    bud.setdefault("used", 0)
    bud.setdefault("t0", time.monotonic())
    req0, was_stopped = int(bud["req"]), bud.get("stopped")
    out, dirty = [], False
    for r in fills.to_dict("records"):
        s = str(r.get("entry_strategy"))
        y, ysrc = stop_y(s, ytab)
        rec = dict(path_label="PENDING", pass_min=None, reach=None, best=None, capture=None, left=None, recovered=None, group="PENDING",
                   path_y=y, path_y_src=ysrc)
        try:
            bk = _fnum(r.get("pnl_percentage"))
            stopped = bool(is_stop(r.get("close_reason")) and bk is not None and bk < 0)
            pk, tr = _fnum(r.get("peak_pnl")), _fnum(r.get("trough_pnl"))
            k = _key(r, ENTRY_X, y, stopped, pk is not None and tr is not None)
            facts = cache.get(k)
            if facts is None:
                te, tc = _ms(r.get("opened_at")), _ms(r.get("closed_at"))
                E = float(r.get("entry_price"))
                pair = str(r.get("pair"))
                args = (te, tc, E, r.get("direction"), bk, ENTRY_X, y, now_ms)
                facts = label_from_bars(WT.load_1m(pair), *args, stopped=stopped, peak=pk, trough=tr)
                incomplete = (facts["label"] == "PENDING" or facts["best"] is None or facts["reach"] is None
                              or (facts["label"] == "NEVER" and facts["recovered"] is None))
                if incomplete and fetch and not bud.get("stopped"):
                    if _fetch_window(WT, pair, (te // MIN) * MIN, max(te + HORIZON_MIN * MIN, tc + POST_EXIT_MIN * MIN), now_ms, bud, attempts):
                        facts = label_from_bars(WT.load_1m(pair), *args, stopped=stopped, peak=pk, trough=tr)
                    dirty = True                         # attempts may have moved
                done = (facts["label"] != "PENDING" and facts["best"] is not None and facts["reach"] is not None
                        and (facts["label"] != "NEVER" or facts["recovered"] is not None))
                if done:
                    cache[k] = facts
                    dirty = True
            fin = finish(facts, bk)
            rec.update(path_label=facts["label"], pass_min=facts["pass_min"], reach=facts["reach"], best=facts["best"],
                       recovered=facts["recovered"], **fin)
        except Exception as e:   # one bad row never kills the read
            notes.append(f"{r.get('pair')} {str(r.get('opened_at'))[:16]}: path label failed ({type(e).__name__}: {str(e)[:80]})")
        out.append(rec)
    if dirty:
        try:
            _save_cache(cache)
        except Exception as e:
            notes.append(f"label cache not saved ({e})")
    if bud.get("stopped") and not was_stopped:
        notes.append(f"fetching stopped: {bud['stopped']} — the rest catches up on later runs")
    if bud["req"] > req0:
        notes.append(f"1m fetch: {bud['req'] - req0} request(s), used weight {bud['used']}")
    return pd.concat([fills.reset_index(drop=True), pd.DataFrame(out, columns=list(PATH_COLS))], axis=1)


def entry_target(fills, label="path", **kw):
    """the winners-vs-losers screens' targets (DL 406): 'path' = 1.0 WORKED / 0.0 NEVER (FLAT / AMBIG / PENDING → NaN, dropped, counted) ·
    'reach' = the continuous best move our way before the stop (operator 10-28), in % points. → (labelled frame, Series)."""
    lab = label_frame(fills, **kw)
    if label == "reach":
        t = pd.to_numeric(lab.reach, errors="coerce").where(~lab.path_label.isin(("AMBIG", "PENDING")))
    else:
        t = lab.path_label.map({"WORKED": 1.0, "NEVER": 0.0})
    return lab, t


# ───────────────────────────── report ─────────────────────────────
def report_lines(lab, title="current batch"):
    """the scout / review section: per sleeve — entry label counts, then the decomposition table with $ and the next-study pointer."""
    L = [f"## 🧭 LOSER / WINNER DECOMPOSITION — {title} (DECISION_LOG 406: entry and exit read from the price path, not the booked exit)", "",
         f"ENTRY label: WORKED = +{ENTRY_X:g} % our way (gross) before −Y (Y = the sleeve's stop, gross: today's config stop, else its "
         f"losing stop closes) within {HORIZON_MIN // 60} h · NEVER = −Y first or its own stop closed it first · REACH = the best move our "
         "way before the stop (median, winners / losers) · FLAT / AMBIG (one 1m bar touched both, or an unstamped entry minute decided "
         "it) / PENDING (bars missing) are not counted. "
         f"Groups: losers NEVER_WORKED · STOPPED_RECOVERED (+{ENTRY_X:g} within {POST_EXIT_MIN} min after the close) · GAVE_BACK · "
         f"DRIFT_LOSS; winners LEFT_ON_TABLE (best to close + {BEST_POST_MIN} min, net of fees, − booked ≥ max({LEFT_MIN_PTS:g}, booked)) · CAPTURED.", ""]
    if not len(lab):
        return L + ["_no closed bot fills_"]
    L += ["| Sleeve | N | Y gross (src) | entry WORKED / NEVER / other | reach median W / L | NEVER_WORKED | STOPPED_RECOVERED | GAVE_BACK | "
          "DRIFT_LOSS | LEFT_ON_TABLE (upside left) | CAPTURED | pending / ambig | next study |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, g in lab.groupby("entry_strategy", sort=True):
        def cell(gr):
            x = g[g.group == gr]
            return "–" if not len(x) else f"{len(x)} · ${pd.to_numeric(x.pnl, errors='coerce').sum():+,.0f}"
        w, n = int((g.path_label == "WORKED").sum()), int((g.path_label == "NEVER").sum())
        lose = g[g.group.isin(LOSER_GROUPS)]
        ent = float(pd.to_numeric(lose[lose.group.isin(("NEVER_WORKED", "STOPPED_RECOVERED"))].pnl, errors="coerce").sum())
        ext = float(pd.to_numeric(lose[lose.group == "GAVE_BACK"].pnl, errors="coerce").sum())
        lt = g[g.group == "LEFT_ON_TABLE"]
        left_usd = float((pd.to_numeric(lt.left, errors="coerce").fillna(0) * pd.to_numeric(lt.notional_value, errors="coerce").fillna(0)).sum() / 100.0)
        nxt = next_study(ent, ext)
        bk_ = pd.to_numeric(g.pnl_percentage, errors="coerce")
        rch = pd.to_numeric(g.reach, errors="coerce").where(~g.path_label.isin(("AMBIG", "PENDING")))
        rw, rl = rch[bk_ > 0].median(), rch[bk_ <= 0].median()
        rtxt = f"{'–' if pd.isna(rw) else f'{rw:+.2f}'} / {'–' if pd.isna(rl) else f'{rl:+.2f}'}"
        L.append(f"| {s} | {len(g)} | {g.path_y.iloc[0]:g} ({g.path_y_src.iloc[0]}) | {w} / {n} / {len(g) - w - n} | {rtxt} | {cell('NEVER_WORKED')} | "
                 f"{cell('STOPPED_RECOVERED')} | {cell('GAVE_BACK')} | {cell('DRIFT_LOSS')} | {cell('LEFT_ON_TABLE')}"
                 f"{f' (${left_usd:,.0f} left)' if len(lt) else ''} | {cell('CAPTURED')} | "
                 f"{int(g.group.isin(('PENDING', 'AMBIG')).sum())} | {nxt} (entry-side loss ${ent:+,.0f} vs exit-side loss ${ext:+,.0f}) |")
    L += ["", "_Entry-side loss = the booked $ of NEVER_WORKED + STOPPED_RECOVERED losers (what a better entry could avoid); exit-side "
          "loss = the booked $ of GAVE_BACK losers (+X reached, still lost). The larger REALISED loss decides which study goes first "
          "(DL 406 step ②; upside left on winners is shown, never added — no exit captures the whole best); the other follows on the "
          "surviving fills (joint check ③). Pending fills are not counted — the pointer can move as they resolve._"]
    return L


def _exports():
    import export_glob as _EG   # 🗄 (DECISION_LOG 409) Downloads + the permanent exports_archive/
    return sorted(_EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_*.csv")), key=os.path.getmtime)


def main(argv):
    fetch = "--fetch" in argv
    since = None
    if "--since" in argv:
        since = argv[argv.index("--since") + 1]
    files = [a for a in argv[1:] if a.endswith(".csv")]
    if not files:
        ex = _exports()
        files = ex[-1:] if ex else []
    if since is None:
        import sweep_separators as SW
        since = SW.BATCH_START
    fr = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d = d[(d.status.astype(str) == "CLOSED") & (d.entry_strategy.astype(str) != "MANUAL")]
        d = d[d.opened_at.astype(str).str.replace(" ", "T").str[:19] >= since]
        fr.append(d)
    if not fr:
        print("no orders export found")
        return
    d = pd.concat(fr, ignore_index=True).drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    notes = []
    lab = label_frame(d, fetch=fetch, notes=notes)
    print("\n".join(report_lines(lab, f"fills opened ≥ {since[:16]} UTC ({', '.join(os.path.basename(f) for f in files)})")))
    for n in notes:
        print(f"  · {n}")


if __name__ == "__main__":
    main(sys.argv)
