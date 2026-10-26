#!/usr/bin/env python3
"""🧷 Scout — ML_MATCHED_REFUSALS (DECISION_LOG 383, operator 2026-10-26; OBSERVE-ONLY — never changes config, never calls the bot API, never
writes the DB; writes only reports/).

QUESTION  Should any refused C/W-MATCHED momentum long (the LONG_UNMATCHED_ONLY block) be allowed? The 2026-10-26 study
          (reports/ML_MATCHED_SHADOW_2026-10-26.md) said NO for every cell on 318 live shadows; two cells were FROZEN as forward observe lines.
SOURCE    e=ML_MATCHED rows of the Decisions CSV exports (~/Downloads/scalpars_decisions_*.csv, paper + post-go-live live via export_glob) —
          written by the engine's refusal recorder (TradingEngine._ml_refusal_record → ml_matched_refusals), at most one row per pair per 5m
          bar: pair, refusal ms, decision price, confidence, EXACT labels (e.g. "C6+W2"), BTC regime label, slot state, post_gate (heat /
          mega-cap would ALSO have refused it), the entry_* snapshot (JSON). Merged across downloads, dedup (pair, t).
COHORT    refusals at ≥ scout_gatekit.dl_deploy_ms(383) (commit + 10 min); post_gate rows and probe rows are OUT (they could never have
          opened); LONG_CHOP_BURST is not applied (needs the fill book — listed as a blind spot, as in the study).
PRICING   the VALIDATED ML exit replica scripts/ml_exit_optimize_yr5.py at RP.BASE (today's live ML exit; DL 380 parity), TAKER entry fee,
          entry = the recorded decision price, ATR = the recorded entry_atr_pct, BTC closed-RSI ruler rebuilt from 5m — exactly as
          scripts/study_ml_matched_build.py priced its shadows. Paths: caches first (RP readers), then this scout's own budgeted public 1m / 5m
          top-up (reports/SCOUT_ML_MATCHED_REFUSALS_fetch/, ≤ MAX_CALLS calls / run, ≤ 400 weight / min of OUR calls, stop above
          X-MBX-USED-WEIGHT-1M 800, the shared cooldown file honoured and armed on 418 / 429). A refusal whose path is not complete yet is
          PENDING (re-priced next run); final prices are cached write-once in reports/SCOUT_ML_MATCHED_REFUSALS.csv.
EPISODES  one shadow position per pair at a time (a refusal while the previous shadow on that pair is still open by the replica's exit is
          dropped) — study_ml_matched_common.episodes. A refusal behind a PENDING shadow on the same pair waits.
CONTROL   the SAME-DAYS UNMATCHED control: closed momentum LONG bot fills from the orders exports (entry_strategy MOMENTUM / empty, no C / W
          pattern flag, non-probe, not operator-closed), opened ≥ the same floor, priced by the SAME replica at the same taker fee.
🔒 FROZEN CELLS (pre-registered in the study §4, never re-fit) — per cell, decided ONCE (atomic freeze file
   reports/.scout_ml_matched_refusals_gate.json; an unreadable file is never overwritten):
   ① W2_CFREE        labels contain W2 and no C pattern (any regime)
   ② W2_CFREE_SBULL  the same ∧ BTC regime label STRONG_BULL (recorded at the refusal)
   read at the shortest open-ordered prefix of priced EPISODES with N ≥ 30 on ≥ 10 UTC days (HOLD while an earlier refusal of the cell is
   still pending) → OPERATOR_CALL iff WR ≥ 61.5 % ∧ avg ≥ +0.10 %/fill ∧ day-block bootstrap lower bound > 0 at α = 0.025 (Bonferroni over
   the 2 cells) ∧ avg ≥ the same-days control avg ∧ no single day / pair carries ≥ 50 % of the net; else DECIDED_NO (recorder can go off).
   OPERATOR_CALL is never automatic — the probe would be 1× with its own revert gate (study §4).
Usage: venv/bin/python scripts/scout_ml_matched_refusals.py [--no-fetch]
"""
import contextlib
import glob
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import export_glob as _EG              # noqa: E402
import scout_gatekit as _GK            # noqa: E402

DL, LAG_MIN = 383, 10
MIN, HOUR, DAY, BAR5 = 60_000, 3_600_000, 86_400_000, 300_000
DEC_GLOB = os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")
ORD_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
REPORTS = os.path.join(_ROOT, "reports")
FETCH = os.path.join(REPORTS, "SCOUT_ML_MATCHED_REFUSALS_fetch")
STORE = os.path.join(REPORTS, "SCOUT_ML_MATCHED_REFUSALS.csv")
STATE = os.path.join(REPORTS, ".scout_ml_matched_refusals_gate.json")
# ── frozen gate (study §4) ──
GATE_N, GATE_DAYS, BE_WR, MIN_AVG, ALPHA, M_CELLS, CONC = 30, 10, 61.5, 0.10, 0.05, 2, 0.50
BOOT_N, BOOT_SEED = 4000, 20261026
CELLS = ("W2_CFREE", "W2_CFREE_SBULL")
DECIDED = ("OPERATOR_CALL", "DECIDED_NO")
# ── Binance budget ──
MAX_CALLS, BUDGET_W, IP_PAUSE_W = 150, 400, 800
URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&endTime={b}&limit={n}"
PRE_5M_BARS = 110
DEC_COLS = ["t", "e", "pair", "price", "conf", "mr_bar_at", "mr_labels", "mr_family", "mr_btc_regime", "mr_post_gate", "mr_room",
            "mr_open_count", "mr_max_positions", "mr_alt_corr24", "mr_alt_beta24", "mr_is_paper", "mr_entry_fields"]
ORD_COLS = ["opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "entry_price", "entry_atr_pct", "pnl_percentage",
            "close_reason", "cell_multiplier_source", "pattern_cell_source", "entry_pattern_c_any_match", "entry_pattern_w_any_match", "is_paper"]
STORE_COLS = ["kind", "key", "pair", "o_ms", "E", "atr", "pct", "reason", "exit_ms", "src", "status", "priced_at"]
TERMINAL = ("PRICED", "NOT_EVALUABLE")   # stored write-once; NOT_EVALUABLE is never re-fetched (383 review)


# ─────────────────────────── helpers ───────────────────────────
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
        ts = pd.Timestamp(str(v).strip().replace(" ", "T")[:23])
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.value // 1_000_000)


def _true(v):
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def tokens(labels):
    return [t for t in str(labels or "").replace(" ", "").split("+") if t and t.lower() != "nan"]


def cell_flags(labels, regime):
    tk = tokens(labels)
    w2_cfree = ("W2" in tk) and not any(t.startswith("C") for t in tk)
    return {"W2_CFREE": w2_cfree, "W2_CFREE_SBULL": w2_cfree and str(regime or "").upper() == "STRONG_BULL"}


def _day(ms):
    return pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%d")


# ─────────────────────────── sources ───────────────────────────
def read_refusals(paths=None):
    """every e=ML_MATCHED row of the decisions exports → one row per (pair, refusal ms), newest export wins."""
    paths = paths if paths is not None else sorted(_EG.glob_exports(DEC_GLOB), key=lambda p: os.path.getmtime(p))
    fr = []
    for f in paths:
        try:
            x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in DEC_COLS)
        except Exception:
            continue
        if "e" not in x:
            continue
        x = x[x.e.astype(str) == "ML_MATCHED"]
        if len(x):
            fr.append(x)
    if not fr:
        return pd.DataFrame(columns=DEC_COLS + ["o_ms"])
    a = pd.concat(fr, ignore_index=True)
    for c in DEC_COLS:
        if c not in a:
            a[c] = None
    a["o_ms"] = [_ms(t) for t in a.t]
    a = a[a.o_ms.notna()].copy()
    a["o_ms"] = a.o_ms.astype("int64")
    return a.drop_duplicates(["pair", "o_ms"], keep="last").sort_values(["o_ms", "pair"]).reset_index(drop=True)


def read_orders(paths=None):
    paths = paths if paths is not None else sorted(_EG.glob_exports(ORD_GLOB), key=lambda p: os.path.getmtime(p))
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in ORD_COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=ORD_COLS)
    a = pd.concat(fr, ignore_index=True)
    for c in ORD_COLS:
        if c not in a:
            a[c] = None
    a["_k"] = a.opened_at.astype(str).str[:19] + "|" + a.pair.astype(str) + "|" + a.direction.astype(str)
    a["_closed"] = (a.status.astype(str) == "CLOSED").astype(int)
    a = a.assign(_i=range(len(a))).sort_values(["_closed", "_i"]).drop_duplicates("_k", keep="last")
    return a.drop(columns=["_closed", "_i"]).reset_index(drop=True)


def shadow_cohort(R, floor_ms):
    """→ (kept shadows, excluded counts). Adds ef (dict), atr, cells."""
    if not len(R):
        return R.assign(atr=[], W2_CFREE=[], W2_CFREE_SBULL=[]), {}
    R = R[R.o_ms >= int(floor_ms)].copy()
    ef = []
    for s in R.mr_entry_fields:
        try:
            d = json.loads(s) if isinstance(s, str) and s.strip() else {}
        except Exception:
            d = {}
        ef.append(d if isinstance(d, dict) else {})
    R["ef"] = ef
    R["atr"] = [_f(d.get("entry_atr_pct")) for d in ef]
    R["probe"] = [bool(d.get("probe")) for d in ef]
    R["E"] = [_f(p) for p in R.price]
    pg = R.mr_post_gate.fillna("").astype(str).replace({"nan": ""})
    ex = {"post_gate": int((pg != "").sum()), "probe": int(R.probe.sum()), "no_price": int(R.E.isna().sum())}
    R = R[(pg == "") & ~R.probe & R.E.notna() & (R.E > 0)].copy()
    for c in CELLS:
        R[c] = [cell_flags(l, g)[c] for l, g in zip(R.mr_labels, R.mr_btc_regime)]
    return R.reset_index(drop=True), ex


def control_cohort(O, floor_ms):
    """closed unmatched non-probe momentum LONG bot fills opened ≥ floor (operator-closed out)."""
    if not len(O):
        return O.assign(o_ms=[], E=[], atr=[]), 0
    O = O[(O.direction.astype(str) == "LONG") & (O.status.astype(str) == "CLOSED")].copy()
    es = O.entry_strategy.fillna("").astype(str).replace({"nan": ""})
    O = O[es.isin(["", "MOMENTUM"])]
    probe = (O.cell_multiplier_source.fillna("").astype(str).str.contains("PROBE")
             | O.pattern_cell_source.fillna("").astype(str).str.contains("PROBE"))
    matched = O.entry_pattern_c_any_match.map(_true) | O.entry_pattern_w_any_match.map(_true)
    O = O[~probe & ~matched].copy()
    O, ox = _GK.split_operator_closed(O)
    O["o_ms"] = [_ms(t) for t in O.opened_at]
    O = O[O.o_ms.notna()].copy()
    O["o_ms"] = O.o_ms.astype("int64")
    O = O[O.o_ms >= int(floor_ms)].copy()
    O["E"] = [_f(x) for x in O.entry_price]
    O["atr"] = [_f(x) for x in O.entry_atr_pct]
    return O[O.E.notna() & (O.E > 0)].reset_index(drop=True), len(ox)


# ─────────────────────────── Binance top-up (public, budgeted) ───────────────────────────
class Budget:
    def __init__(self, max_calls=MAX_CALLS):
        self.max_calls, self.n, self.stop, self.recent = max_calls, 0, None, []

    def ok(self):
        if self.stop:
            return False
        try:
            import binance_cooldown as BC
            if BC.active():
                self.stop = "shared Binance cooldown active"
                return False
        except Exception:
            pass
        if self.n >= self.max_calls:
            self.stop = f"call cap {self.max_calls} reached (the rest next run)"
            return False
        return True

    def wait(self, w):
        while True:
            now = time.time()
            self.recent = [(t, x) for t, x in self.recent if now - t < 60]
            if sum(x for _, x in self.recent) + w <= BUDGET_W:
                break
            time.sleep(2)
        self.recent.append((time.time(), w))


def arm_cooldown(min_ms, err=None, path=None, now_ms=None):
    """write the shared cooldown as max(existing, now + max(min_ms, Retry-After)) — never shortens a cooldown another script armed."""
    try:
        import binance_cooldown as BC
        path = path or BC.SHARED
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        ra = 0
        try:
            ra = int(float((getattr(err, "headers", None) or {}).get("Retry-After") or 0) * 1000)
        except (TypeError, ValueError, AttributeError):
            ra = 0
        until = max(BC.read_until_ms(path), now + max(int(min_ms), ra))
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            fh.write(str(until))
        os.replace(tmp, path)
        return until
    except Exception:
        return None


def _get(sym, iv, a, b, bud):
    if not bud.ok():
        return None
    n = 499
    bud.wait(2)
    bud.n += 1
    req = urllib.request.Request(URL.format(s=sym, i=iv, a=int(a), b=int(b), n=n), headers={"User-Agent": "scalpars-scout"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            used = int(r.headers.get("x-mbx-used-weight-1m", "0") or 0)
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            arm_cooldown(15 * MIN, e)
            bud.stop = f"HTTP {e.code} — shared cooldown armed"
        return None
    except Exception:
        return None
    if used > IP_PAUSE_W:
        bud.stop = f"IP used weight {used} > {IP_PAUSE_W}"
    return data


def _have(sub, sym, a, b, step):
    f = os.path.join(FETCH, sub, sym + ".csv")
    if not os.path.exists(f):
        return False
    try:
        t = pd.read_csv(f, usecols=["open_time"]).open_time
    except Exception:
        return False
    return int(((t >= a) & (t <= b)).sum()) >= (b - a) // step - 1


def _save(sub, sym, rows, now_ms):
    d = os.path.join(FETCH, sub)
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, sym + ".csv")
    step = MIN if sub == "k1m" else BAR5
    df = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows
                       if int(r[0]) + step <= now_ms], columns=["open_time", "o", "h", "l", "c", "vol"])   # CLOSED bars only
    if os.path.exists(f):
        try:
            df = pd.concat([pd.read_csv(f), df])
        except Exception:
            pass
    tmp = f"{f}.{os.getpid()}.tmp"
    df.drop_duplicates("open_time", keep="last").sort_values("open_time").to_csv(tmp, index=False)
    os.replace(tmp, f)


def _span(sym, iv, a, end, now_ms, bud):
    step = MIN if iv == "1m" else BAR5
    sub = "k1m" if iv == "1m" else "k5m"
    end = min(int(end), int(now_ms) - step)
    a = int(a) // step * step
    while a <= end:
        b = min(a + 498 * step, end)
        if not _have(sub, sym, a, b, step):
            rows = _get(sym, iv, a, b + step - 1, bud)
            if rows is None:
                return
            if rows:
                _save(sub, sym, rows, now_ms)
        a = b + step


def ensure_paths(fills, now_ms, bud, horizon_ms):
    """1m [o − 10 min, o + horizon] and 5m [o − 110 bars, o] per fill + BTC 5m over the whole window. Budgeted; stops quietly."""
    if not fills:
        return
    lo = min(f["o_ms"] for f in fills)
    _span("BTCUSDT", "5m", lo - PRE_5M_BARS * BAR5, now_ms, now_ms, bud)
    for f in sorted(fills, key=lambda x: x["o_ms"]):
        if bud.stop:
            return
        _span(f["pair"], "5m", f["o_ms"] - PRE_5M_BARS * BAR5, f["o_ms"], now_ms, bud)
        _span(f["pair"], "1m", f["o_ms"] - 10 * MIN, f["o_ms"] + horizon_ms, now_ms, bud)


# ─────────────────────────── replica (RP.BASE) with this scout's top-up ───────────────────────────
def _rp():
    import ml_exit_optimize_yr5 as RP
    return RP


def _read_fetch(sub, pair):
    f = os.path.join(FETCH, sub, pair + ".csv")
    if not os.path.exists(f):
        return None
    try:
        return pd.read_csv(f)
    except Exception:
        return None


@contextlib.contextmanager
def replica_with_topup():
    """RP's readers extended with this scout's fetch dir for the duration of the pricing (restored + caches cleared after)."""
    RP = _rp()
    orig = (RP._k1, RP._k5, RP._btc)

    def clear():
        RP._K1.clear(); RP._K5.clear(); RP._DAYS.clear(); RP._BTC = None

    def k1(pair):
        if pair in RP._K1:
            return RP._K1[pair]
        base = orig[0](pair)
        mine = _read_fetch("k1m", pair)
        fr = [x for x in (base, mine) if x is not None and len(x)]
        df = (pd.concat([x[["open_time", "o", "h", "l", "c"]] for x in fr]).drop_duplicates("open_time", keep="last")
              .sort_values("open_time").reset_index(drop=True) if fr else None)
        RP._K1[pair] = df
        return df

    def k5(pair):
        if pair in RP._K5:
            return RP._K5[pair]
        fr = []
        fp = os.path.join(RP.CACHE, "k5m_full", f"{pair}.csv")
        if os.path.exists(fp):
            fr.append(pd.read_csv(fp, usecols=["open_time", "c"]))
        mine = _read_fetch("k5m", pair)
        if mine is not None and len(mine):
            fr.append(mine[["open_time", "c"]])
        df = pd.concat(fr) if fr else None
        k = k1(pair)
        if k is not None and len(k):
            g = k.assign(b=(k.open_time // BAR5) * BAR5).groupby("b").c.last().reset_index().rename(columns={"b": "open_time"})
            df = g if df is None else pd.concat([df, g[~g.open_time.isin(df.open_time)]])
        if df is None or not len(df):
            RP._K5[pair] = None
            return None
        df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        c = df.c.astype(float)
        out = dict(ot=df.open_time.values.astype(np.int64), c=c.values)
        for n in (5, 8, 13, 20):
            out[f"e{n}"] = c.ewm(span=n, adjust=False).mean().values
        RP._K5[pair] = out
        return out

    def btc():
        if RP._BTC is None:
            fr = []
            fp = os.path.join(RP.CACHE, "btc_5m.csv")
            if os.path.exists(fp):
                fr.append(pd.read_csv(fp, usecols=["open_time", "o", "h", "l", "c"]))
            mine = _read_fetch("k5m", "BTCUSDT")
            if mine is not None and len(mine):
                fr.append(mine[["open_time", "o", "h", "l", "c"]])
            b = (pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time") if fr
                 else pd.DataFrame(columns=["open_time", "o", "h", "l", "c"]))
            RP._BTC = b[["open_time", "o", "h", "l", "c"]].values.astype(float)
        return RP._BTC

    clear()
    RP._k1, RP._k5, RP._btc = k1, k5, btc
    try:
        yield RP
    finally:
        RP._k1, RP._k5, RP._btc = orig
        clear()


def price_replica(fills, now_ms):
    """→ {key: dict(pct, reason, exit_ms, src, status)} at RP.BASE, taker fee. status PRICED / PENDING (path not complete yet) /
    NOT_EVALUABLE (no path ≥ 1 day after the horizon). One fill at a time, in-process (no worker pool)."""
    out = {}
    with replica_with_topup() as RP:
        for f in fills:
            try:
                rsi = RP.btc_rsi_at(int(f["o_ms"]))
                fx = dict(key=f["key"], pair=f["pair"], o_ms=int(f["o_ms"]), E=float(f["E"]), fee_in=RP.TAKER,
                          atr=(float(f["atr"]) if f.get("atr") and f["atr"] > 0 else None), rsi_entry=(None if rsi is None else float(rsi)))
                k, why, pct, tex, src, _pk = RP.run_fill(fx, {"BASE": RP.BASE})["BASE"]
            except Exception as e:
                k, why, pct, tex, src = None, f"ERR:{str(e)[:60]}", float("nan"), None, "err"
            done_by = int(f["o_ms"]) + RP.HORIZON_MS
            if why in ("END_OF_DATA", "NO_DATA") or str(why).startswith("ERR"):
                st = "NOT_EVALUABLE" if now_ms > done_by + DAY else "PENDING"
            else:
                st = "PRICED"
            out[f["key"]] = dict(pct=(None if pct is None or pct != pct else float(pct)), reason=why, exit_ms=tex, src=src, status=st)
    return out


# ─────────────────────────── store (write-once for final prices) ───────────────────────────
def load_store(path=STORE):
    if not path or not os.path.exists(path):
        return {}
    try:
        d = pd.read_csv(path)
        return {(r.kind, r.key): dict(pct=_f(r.pct), reason=r.reason, exit_ms=(int(r.exit_ms) if _f(r.exit_ms) is not None else None),
                                      src=r.src, status=(r.status if getattr(r, "status", None) in TERMINAL else "PRICED"), pair=r.pair, o_ms=r.o_ms, E=r.E, atr=r.atr, priced_at=r.priced_at)
                for r in d.itertuples()}
    except Exception:
        bad = f"{path}.bad-{int(time.time())}"
        try:
            os.replace(path, bad)
        except Exception:
            pass
        return {}


def save_store(cache, rows_by_key, path=STORE):
    if not path:
        return
    if not os.path.abspath(path).startswith(REPORTS + os.sep):
        raise ValueError("ML_MATCHED_REFUSALS writes only under reports/")
    recs = []
    for (kind, key), v in cache.items():
        if v.get("status") not in TERMINAL:
            continue
        r = rows_by_key.get((kind, key)) or v          # a stored row whose export is gone keeps its own fields (write-once)
        recs.append(dict(kind=kind, key=key, pair=r.get("pair"), o_ms=r.get("o_ms"), E=r.get("E"), atr=r.get("atr"), pct=v.get("pct"),
                         reason=v.get("reason"), exit_ms=v.get("exit_ms"), src=v.get("src"), status=v.get("status"), priced_at=v.get("priced_at") or int(time.time())))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    pd.DataFrame(recs, columns=STORE_COLS).to_csv(tmp, index=False)
    os.replace(tmp, path)


# ─────────────────────────── stats / episodes ───────────────────────────
def episodes(S):
    """one shadow per pair at a time (study_ml_matched_common.episodes); a PENDING shadow keeps its pair busy (later refusals wait)."""
    if not len(S):
        return S.assign(episode=[], waiting=[])
    S = S.sort_values(["o_ms", "pair"]).reset_index(drop=True)
    keep, wait = np.zeros(len(S), bool), np.zeros(len(S), bool)
    for _, g in S.groupby("pair", sort=False):
        busy, blocked = -1, False
        for i, r in g.iterrows():
            if blocked:
                wait[i] = True
                continue
            if r.o_ms >= busy:
                if r.status == "PRICED":
                    keep[i] = True
                    busy = int(r.exit_ms) if r.exit_ms is not None and r.exit_ms == r.exit_ms else int(r.o_ms) + 3 * HOUR
                elif r.status == "PENDING":
                    wait[i] = True
                    blocked = True
    return S.assign(episode=keep, waiting=wait)


def day_boot(y, day, n=BOOT_N, seed=BOOT_SEED):
    y, day = np.asarray(y, float), np.asarray(day)
    u, inv = np.unique(day, return_inverse=True)
    if len(y) == 0 or len(u) < 2:
        return np.array([])
    s, c = np.bincount(inv, y), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    return s[idx].sum(1) / c[idx].sum(1)


def top_share(y, key):
    s = pd.Series(np.asarray(y, float)).groupby(np.asarray(key)).sum()
    tot = float(s.sum())
    return float(s.max() / tot) if tot > 0 else float("nan")


def stat(g, ctl):
    y = pd.to_numeric(g.pct).values.astype(float)
    days = g.day.unique()
    b = day_boot(y, g.day.values)
    c = ctl[ctl.day.isin(set(days))] if len(ctl) else ctl
    cy = pd.to_numeric(c.pct).values.astype(float) if len(c) else np.array([])
    return dict(N=len(y), days=len(days), WR=(float((y > 0).mean() * 100) if len(y) else float("nan")),
                avg=(float(y.mean()) if len(y) else float("nan")), net=float(y.sum()),
                lb=(float(np.quantile(b, ALPHA / M_CELLS)) if len(b) else float("nan")),
                ctl_n=len(cy), ctl_avg=(float(cy.mean()) if len(cy) else float("nan")),
                top_day=top_share(y, g.day.values), top_pair=top_share(y, g.pair.astype(str).values))


def fmt_stat(s):
    f = lambda v, p="{:+.3f}": ("n/a" if v is None or v != v else p.format(v))
    return (f"N {s['N']} · {s['days']} d · WR {f(s['WR'], '{:.1f}')} % · avg {f(s['avg'])} %/fill · Σ {f(s['net'], '{:+.2f}')} · "
            f"day-boot LB (α {ALPHA}/{M_CELLS}) {f(s['lb'])} · same-days control N {s['ctl_n']} avg {f(s['ctl_avg'])} · Δ vs ctl "
            f"{f(s['avg'] - s['ctl_avg'] if s['ctl_avg'] == s['ctl_avg'] else float('nan'))} · top day {f(s['top_day'], '{:.0%}')} · "
            f"top pair {f(s['top_pair'], '{:.0%}')}")


def gate_reads(E, W, ctl):
    """E = priced episodes, W = waiting / pending rows (both with the cell flags) → {cell: (verdict, text, decided)}."""
    out = {}
    for c in CELLS:
        g = E[E[c].astype(bool)].sort_values(["o_ms", "pair"]) if len(E) else E
        pre = None
        for j in range(len(g)):
            q = g.iloc[:j + 1]
            if len(q) >= GATE_N and q.day.nunique() >= GATE_DAYS:
                pre = q
                break
        if pre is None:
            out[c] = ("COLLECTING", f"{len(g)}/{GATE_N} priced episodes on {g.day.nunique() if len(g) else 0}/{GATE_DAYS} days", False)
            continue
        end = int(pre.o_ms.max())
        w = W[W[c].astype(bool) & (W.o_ms <= end)] if len(W) else W
        if len(w):
            out[c] = ("HOLD", f"prefix {len(pre)} episodes: {len(w)} earlier refusal(s) still pending — held", False)
            continue
        s = stat(pre, ctl)
        fails = []
        if not s["WR"] >= BE_WR: fails.append(f"WR < {BE_WR}")
        if not s["avg"] >= MIN_AVG: fails.append(f"avg < +{MIN_AVG}")
        if not (s["lb"] == s["lb"] and s["lb"] > 0): fails.append("LB ≤ 0")
        if not (s["ctl_avg"] == s["ctl_avg"] and s["avg"] >= s["ctl_avg"]): fails.append("< same-days control")
        if (s["top_day"] == s["top_day"] and s["top_day"] >= CONC) or (s["top_pair"] == s["top_pair"] and s["top_pair"] >= CONC):
            fails.append("day / pair ≥ 50 % of net")
        txt = fmt_stat(s) + " → "
        out[c] = (("OPERATOR_CALL", txt + "OPERATOR CALL on a 1× probe (never automatic)", True) if not fails
                  else ("DECIDED_NO", txt + "DECIDED no (" + "; ".join(fails) + ")", True))
    return out


def frozen(state_path, name, v, txt, decided):
    """🔒 decided ONCE (atomic temp + os.replace); an unreadable state file is never overwritten. → (verdict, text)."""
    st = {}
    if state_path and os.path.exists(state_path):
        try:
            with open(state_path) as fh:
                st = json.load(fh)
            if not isinstance(st, dict):
                raise ValueError("not a JSON object")
        except Exception as e:
            print(f"⚠ ML_MATCHED_REFUSALS: state file {state_path} unreadable ({str(e)[:60]}) — left untouched", file=sys.stderr)
            return v, txt + " (⚠ state file unreadable — not frozen)"
    if name in st:
        return st[name]["verdict"], st[name]["text"] + f" (frozen {str(st[name].get('at', ''))[:16]} UTC)"
    if state_path and decided and v in DECIDED:
        try:
            st[name] = {"verdict": v, "text": txt, "at": pd.Timestamp.now("UTC").isoformat()}
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            tmp = f"{state_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(st, fh, indent=1)
            os.replace(tmp, state_path)
        except Exception as e:
            print(f"⚠ ML_MATCHED_REFUSALS: could not persist {name} ({str(e)[:60]})", file=sys.stderr)
    return v, txt


# ─────────────────────────── run ───────────────────────────
def run(now_ms=None, refusals=None, orders=None, floor_ms="auto", price_fn=None, fetch=True, state_path=STATE, store_path=STORE):
    """→ report lines. Never raises."""
    H = "## 🧷 ML_MATCHED_REFUSALS — refused C/W-matched momentum longs, priced at today's ML exit (DECISION_LOG 383, observe-only; frozen cells, OPERATOR CALL)"
    try:
        return _run(now_ms, refusals, orders, floor_ms, price_fn, fetch, state_path, store_path, H)
    except Exception as e:
        return [H, "", f"Unavailable this run ({type(e).__name__}: {str(e)[:120]}).", ""]


def _run(now_ms, refusals, orders, floor_ms, price_fn, fetch, state_path, store_path, H):
    now_ms = int(now_ms or pd.Timestamp.now("UTC").value // 1_000_000)
    L = [H, "", "Cells (frozen 2026-10-26, study §4): ① W2_CFREE = W2 and no C pattern · ② W2_CFREE_SBULL = ① ∧ BTC regime STRONG_BULL. "
         f"Gate per cell at the first {GATE_N} priced episodes on ≥ {GATE_DAYS} days: WR ≥ {BE_WR} % ∧ avg ≥ +{MIN_AVG} ∧ day-boot LB > 0 at "
         f"α {ALPHA}/{M_CELLS} ∧ ≥ same-days unmatched control ∧ no day / pair ≥ {CONC:.0%} of net → OPERATOR CALL; else DECIDED no. "
         "Replica = RP.BASE (today's ML exit), taker fee, one shadow per pair at a time; post-gate (heat / mega-cap) and probe refusals out."]
    if floor_ms == "auto":
        floor_ms = _GK.dl_deploy_ms(DL, lag_min=LAG_MIN, label="ML_MATCHED_REFUSALS")
    if floor_ms is None:
        return L + [f"NOT EVALUABLE — no DECISION_LOG {DL} commit found yet (the cohort starts at its deploy + {LAG_MIN} min).", ""]
    R = read_refusals() if refusals is None else refusals
    O = read_orders() if orders is None else orders
    S, ex = shadow_cohort(R, floor_ms)
    C, n_ox = control_cohort(O, floor_ms)
    L.append(f"Cohort (refusals ≥ {pd.Timestamp(int(floor_ms), unit='ms'):%Y-%m-%d %H:%M} UTC): {len(S)} recorded · excluded post-gate "
             f"{ex.get('post_gate', 0)} · probe {ex.get('probe', 0)} · no price {ex.get('no_price', 0)} | control fills {len(C)} "
             f"(operator-closed out {n_ox})")
    if not len(S) and not len(C):
        return L + ["  nothing to price yet.", ""]
    rows = {}
    for i, r in S.iterrows():
        rows[("S", f"{r.pair}|{int(r.o_ms)}")] = dict(key=f"{r.pair}|{int(r.o_ms)}", pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), atr=r.atr)
    for i, r in C.iterrows():
        rows[("C", f"{r.pair}|{int(r.o_ms)}")] = dict(key=f"{r.pair}|{int(r.o_ms)}", pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), atr=r.atr)
    cache = load_store(store_path) if store_path else {}
    todo = [k for k in rows if k not in cache]
    notes = []
    if todo:
        fills = [dict(rows[k], key=f"{k[0]}:{k[1]}") for k in todo]
        if price_fn is None:
            RP = _rp()
            if fetch:
                bud = Budget()
                ensure_paths(fills, now_ms, bud, RP.HORIZON_MS)
                notes.append(f"public kline calls {bud.n}" + (f" · stopped: {bud.stop}" if bud.stop else ""))
            res = price_replica(fills, now_ms)
        else:
            res = price_fn(fills, now_ms)
        for k in todo:
            v = res.get(f"{k[0]}:{k[1]}") or dict(pct=None, reason="NO_DATA", exit_ms=None, src="none", status="PENDING")
            v = dict(v, priced_at=int(time.time()))
            cache[k] = v
        try:
            save_store(cache, rows, store_path)
        except Exception as e:
            notes.append(f"store not written ({str(e)[:60]})")
    for n in notes:
        L.append(f"  (data) {n}")

    def attach(df, kind):
        if not len(df):
            return df.assign(pct=[], reason=[], exit_ms=[], status=[], day=[])
        v = [cache.get((kind, f"{r.pair}|{int(r.o_ms)}"), {}) for r in df.itertuples()]
        return df.assign(pct=[x.get("pct") for x in v], reason=[x.get("reason") for x in v], exit_ms=[x.get("exit_ms") for x in v],
                         status=[x.get("status", "PENDING") for x in v], day=[_day(t) for t in df.o_ms])
    S, C = attach(S, "S"), attach(C, "C")
    ctl = C[C.status == "PRICED"] if len(C) else C
    Sx = episodes(S)
    E = Sx[Sx.episode] if len(Sx) else Sx
    W = Sx[Sx.waiting] if len(Sx) else Sx
    ne = int((S.status == "NOT_EVALUABLE").sum()) if len(S) else 0
    L.append(f"Shadows: priced {int((S.status == 'PRICED').sum()) if len(S) else 0} · pending {int((S.status == 'PENDING').sum()) if len(S) else 0} · "
             f"not evaluable {ne} → episodes (one per pair at a time) {len(E)} · waiting {len(W)}")
    if len(ctl):
        cy = pd.to_numeric(ctl.pct)
        L.append(f"Control (unmatched momentum longs, same replica / fee): N {len(ctl)} · {ctl.day.nunique()} d · WR {(cy > 0).mean() * 100:.1f} % · "
                 f"avg {cy.mean():+.3f} %/fill")
    L.append("Descriptive (not gated):")
    if len(E):
        fam = E.mr_family.fillna("").astype(str)
        for name, m in (("ALL matched", np.ones(len(E), bool)), ("C-only", (fam == "C").values), ("W-only", (fam == "W").values),
                        ("C+W", (fam == "C+W").values)):
            g = E[m]
            if len(g):
                L.append(f"  {name}: " + fmt_stat(stat(g, ctl)))
    else:
        L.append("  no priced episode yet")
    L.append("🔒 FROZEN cells:")
    for name, (v, txt, dec) in gate_reads(E, W, ctl).items():
        fv, ft = frozen(state_path, name, v, txt, dec)
        g = E[E[name].astype(bool)] if len(E) else E
        L.append(f"  {name}: {fv} — {ft}")
        if len(g) and fv in ("COLLECTING", "HOLD"):
            L.append("     running: " + fmt_stat(stat(g, ctl)))
    L.append("Blind spots: LONG_CHOP_BURST not applied (needs the fill book) · slot competition ignored (a matched long would take a slot) · "
             "entry at the recorded decision price, taker, no queue / slippage model · funding / order book not recorded.")
    L.append("")
    return L


if __name__ == "__main__":
    print("\n".join(run(fetch="--no-fetch" not in sys.argv)))
