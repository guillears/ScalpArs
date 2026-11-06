#!/usr/bin/env python3
"""📐 Study — ML_MKT_HOT on a ROLLING universe (2026-11-06; research only — never changes config, no bot API; Binance PUBLIC fapi only).

The watch line ML_MKT_HOT (DECISION_LOG 416, scripts/scout_ml_mkt_hot_watch.py) measures breadth on a FROZEN 108-coin list. This study
pre-registers six rolling definitions (V1..V6, see reports/ML_MKT_HOT_ROLLING_2026-11-06.md §0 — written before any fetch) and selects ONE by
the pre-registered priority rules (live cost → adapts → faithful/same sign → stability → simplicity), never by the biggest gap.

Measure (frozen): breadth = share of universe coins whose last CLOSED 1h close > the close 4 h earlier; ≥ 30 coins with data; last bar ≤ 2 h old;
hot = breadth > 0.50. Cohort: master full-size momentum longs (158, stack_pct) + B25 (39, pnl_percentage).

Budget: ≤ 1,200 weight/min — reads X-MBX-USED-WEIGHT-1M, pauses to the next minute past 900, stops on 418/429 (and honours the shared cooldown).
Cache: reports/backtest_cache/mkt_hot_rolling/.
Usage: venv/bin/python -I scripts/study_ml_mkt_hot_rolling.py [--no-fetch]
"""
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import scout_ml_mkt_hot_watch as MH   # noqa: E402  (FROZEN universe, MIN_COINS, HOT_CUT, bootstrap)
import sweep_separators as SW         # noqa: E402  (_is_probe)
import scout_long_stop4 as LS         # noqa: E402  (shared Binance cooldown)

ROOT = os.path.dirname(_D)
CACHE = os.path.join(ROOT, "reports", "backtest_cache", "mkt_hot_rolling")
OLD_EI = os.path.join(ROOT, "reports", "backtest_cache", "exchange_info.json")
OLD_DAILY = os.path.join(ROOT, "reports", "backtest_cache", "daily")
MASTER = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
ZONES = os.path.join(ROOT, "reports", "best_case", "zones.csv")
REPORT = os.path.join(ROOT, "reports", "ML_MKT_HOT_ROLLING_2026-11-06.md")
OUT_FILLS = os.path.join(ROOT, "reports", "ML_MKT_HOT_ROLLING_2026-11-06_fills.csv")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
B25_START = pd.Timestamp("2026-10-23 18:30")
# pinned to the export every other 11-06 read used (zones.csv / best case / replace analysis: 39 B25 fills); a newer export only adds fills
# zones.csv does not score yet — set SCALPARS_EXPORT to override
EXPORT_PIN = os.environ.get("SCALPARS_EXPORT") or os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-06_13-03-41.csv")


def _export():
    return EXPORT_PIN if os.path.exists(EXPORT_PIN) else max(sorted(glob.glob(EXPORT_GLOB)), key=os.path.getmtime)
FAPI = "https://fapi.binance.com"
HOUR, DAY = 3_600_000, 86_400_000
D1_START = pd.Timestamp("2026-05-01")       # 1d klines from here (7d averages + listing age need history)
H1_START = pd.Timestamp("2026-06-16")       # 1h klines from here (cohort starts 2026-06-18)
PAGE = 1000                                  # 1h page (weight 5)
WEIGHT_PAUSE, PACE_S = 900, 0.06
HOT, MINC = MH.HOT_CUT, MH.MIN_COINS
REG_DAY = pd.Timestamp("2026-10-30")        # ML_MKT_HOT registration (DL 416)
V4_CAP, V5_AGE_D, V6_AGE_D = 150, 30, 90
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "BUSD", "USD1", "USDE", "DAI", "PYUSD", "RLUSD", "U", "XUSD", "USDS", "EUR", "EURI", "AEUR", "USDD", "BFUSD"}
EXCL = {"BTCUSDT", "ETHUSDT"}


# ───────────────────────── fetch (budgeted, cached) ─────────────────────────
class Halt(Exception):
    pass


class Api:
    def __init__(self, fetch=True):
        self.fetch, self.used, self.req = fetch, 0, 0

    def _get(self, url):
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    self.used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or self.used)
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError:
                raise
            except (TimeoutError, OSError, urllib.error.URLError) as e:      # transient network error: back off and retry, then stop
                if attempt == 3:
                    raise Halt(f"network error after 4 tries: {e}")
                time.sleep(5 * (attempt + 1))

    def get(self, path, params):
        if not self.fetch:
            raise Halt("--no-fetch")
        now = int(time.time() * 1000)
        if LS._cooling(now):
            raise Halt("shared Binance cooldown active")
        if self.used >= WEIGHT_PAUSE:
            wait = 61 - time.gmtime().tm_sec
            print(f"  weight {self.used} ≥ {WEIGHT_PAUSE}: pausing {wait}s", flush=True)
            time.sleep(wait)
            self.used = 0
        time.sleep(PACE_S)
        self.req += 1
        url = FAPI + path + ("?" + urllib.parse.urlencode(params) if params else "")
        try:
            return self._get(url)
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                try:
                    LS.NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now)
                except Exception:
                    pass
                raise Halt(f"HTTP {e.code} — stopped")
            if e.code == 400:
                return None                     # no such symbol (delisted / never on fapi)
            raise Halt(f"HTTP {e.code}")


def _jload(p):
    with open(p) as f:
        return json.load(f)


def _jsave(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, p)


def exchange_info(api):
    """→ {symbol: meta} = live exchangeInfo (USDT PERPETUAL, any status) ∪ the 2026-09-16 snapshot (keeps since-delisted coins' metadata)."""
    p = os.path.join(CACHE, "exchangeInfo_live.json")
    if not os.path.exists(p):
        d = api.get("/fapi/v1/exchangeInfo", {})
        _jsave(p, d)
    live = _jload(p)
    meta = {}
    old = _jload(OLD_EI) if os.path.exists(OLD_EI) else {}
    for s, m in old.items():
        if s.endswith("USDT") and m.get("contractType") == "PERPETUAL":
            meta[s] = dict(m, src="snapshot-09-16")
    for m in live.get("symbols", []):
        if m.get("quoteAsset") == "USDT" and m.get("contractType") == "PERPETUAL":
            meta[m["symbol"]] = dict(onboardDate=m.get("onboardDate"), underlyingType=m.get("underlyingType"),
                                     underlyingSubType=m.get("underlyingSubType"), status=m.get("status"),
                                     baseAsset=m.get("baseAsset"), src="live")
    return meta


def klines_1d(api, sym, now_ms):
    """→ rows [open_ms, close, quote_vol] from D1_START (closed bars). Cached per day of fetch."""
    p = os.path.join(CACHE, "1d", f"{sym}_{time.strftime('%Y%m%d', time.gmtime(now_ms / 1000))}.json")
    if os.path.exists(p):
        return _jload(p)
    if not api.fetch:                       # --no-fetch: the newest cached fetch of this symbol
        old = sorted(glob.glob(os.path.join(CACHE, "1d", f"{sym}_2*.json")))
        return _jload(old[-1]) if old else []
    st = int(D1_START.value // 10**6)
    d = api.get("/fapi/v1/klines", dict(symbol=sym, interval="1d", startTime=st, limit=499))   # weight 2
    rows = [] if d is None else [[int(x[0]), float(x[4]), float(x[7])] for x in d if int(x[0]) + DAY <= now_ms]
    _jsave(p, rows)
    return rows


def klines_1h(api, sym, now_ms):
    """→ rows [open_ms, close, quote_vol] from H1_START. Complete 1,000-bar pages cached forever; the tail page per run-hour."""
    out, t = [], int(H1_START.value // 10**6)
    while t < now_ms:
        full = t + PAGE * HOUR <= now_ms
        p = os.path.join(CACHE, "1h", f"{sym}_{t}.json" if full else f"{sym}_{t}_tail_{now_ms // HOUR}.json")
        tails = [] if full else sorted(glob.glob(os.path.join(CACHE, "1h", f"{sym}_{t}_tail_*.json")))
        if os.path.exists(p):
            pg = _jload(p)
        elif not api.fetch and (full or tails):     # --no-fetch: the newest cached tail
            pg = _jload(tails[-1]) if tails else []
        else:
            d = api.get("/fapi/v1/klines", dict(symbol=sym, interval="1h", startTime=t, limit=PAGE))   # weight 5
            pg = [] if d is None else [[int(x[0]), float(x[4]), float(x[7])] for x in d if int(x[0]) + HOUR <= now_ms]
            if full and d is not None:
                _jsave(p, pg)
            elif not full:
                _jsave(p, pg)
        out += pg
        t += PAGE * HOUR
    return out


def old_daily(sym):
    """fallback for coins fapi no longer serves: the repo's 2026 daily quote-volume cache (sym, open_ms, qv)."""
    p = os.path.join(OLD_DAILY, f"{sym}.csv")
    if not os.path.exists(p):
        return []
    d = pd.read_csv(p, header=None, names=["s", "o", "qv"])
    return [[int(o), np.nan, float(q)] for o, q in zip(d.o, d.qv)]


# ───────────────────────── cohort ─────────────────────────
def _key(s):
    return pd.Series(s).astype(str).str.replace(" ", "T").str[:19].values


def load_cohort():
    m = pd.read_csv(MASTER, low_memory=False)
    m = m[m.stack_keep.astype(str).eq("True") & ~SW._is_probe(m) & (m.entry_strategy == "MOMENTUM") & (m.direction == "LONG")]
    a = pd.DataFrame(dict(src="MASTER", k=_key(m.opened_at), pair=m.pair.values, pct=pd.to_numeric(m.stack_pct, errors="coerce").values,
                          rank=pd.to_numeric(m.get("entry_pair_rank"), errors="coerce").values))
    newest = _export()
    e = pd.read_csv(newest, low_memory=False)
    e = e[(e.entry_strategy == "MOMENTUM") & (e.direction == "LONG") & (e.status.astype(str).str.upper() == "CLOSED") & ~SW._is_probe(e)]
    e = e[pd.to_datetime(_key(e.opened_at), format="ISO8601") >= B25_START]
    b = pd.DataFrame(dict(src="B25", k=_key(e.opened_at), pair=e.pair.values, pct=pd.to_numeric(e.pnl_percentage, errors="coerce").values,
                          rank=pd.to_numeric(e.get("entry_pair_rank"), errors="coerce").values))
    c = pd.concat([a, b], ignore_index=True)
    c["o"] = pd.to_datetime(c.k, format="ISO8601")
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    z = pd.read_csv(ZONES)
    z = z[z.sl == "MOMENTUM_LONG"]
    z = pd.DataFrame(dict(src=z.src.values, k=_key(z.o), pair=z.pair.values, F108_zones=z.ML_MKT_HOT.astype(str).map({"True": True, "False": False}).values))
    c = c.merge(z, on=["src", "k", "pair"], how="left")
    return c.sort_values("o").reset_index(drop=True), os.path.basename(newest)


# ───────────────────────── universes ─────────────────────────
def is_stable(sym, meta, closes_d):
    base = (meta.get(sym, {}) or {}).get("baseAsset") or sym[:-4]
    if base in STABLE_BASES:
        return True
    c = closes_d.get(sym)
    if c is not None:
        c = c.dropna()
        if len(c) >= 10 and (c.between(0.97, 1.03)).mean() > 0.95:     # empirically USD-pegged
            return True
    return False


def onboard_ms(sym, meta, vol_d):
    ob = (meta.get(sym, {}) or {}).get("onboardDate")
    if ob:
        return int(ob)
    s = vol_d[sym].dropna() if sym in vol_d else pd.Series(dtype=float)
    return int(s.index.min().value // 10**6) if len(s) else None


def daily_universes(vol_d, eligible, meta, days, floor):
    """→ {variant: {day: [syms]}} for V1..V5. vol_d: day-index × sym quote volume (index = UTC day open)."""
    v = vol_d[eligible]
    avg7 = v.rolling(7, min_periods=1).mean()
    U = {k: {} for k in ("V1", "V2", "V3", "V4", "V5")}
    obs = {s: onboard_ms(s, meta, vol_d) for s in eligible}
    for d in days:
        prev = d - pd.Timedelta(days=1)
        if prev not in v.index:
            continue
        pv = v.loc[prev].dropna()
        pv = pv[pv > 0].sort_values(ascending=False)
        a7 = avg7.loc[prev].dropna()
        a7 = a7[a7 > 0].sort_values(ascending=False)
        U["V1"][d] = list(pv.index[:50])
        U["V2"][d] = list(pv.index[:100])
        U["V3"][d] = list(a7.index[:50])
        U["V4"][d] = list(a7[a7 >= floor].index[:V4_CAP])
        dm = int(d.value // 10**6)
        old = [s for s in a7.index if obs.get(s) is not None and obs[s] <= dm - V5_AGE_D * DAY]
        U["V5"][d] = old[:50]
    return U


# ───────────────────────── scoring ─────────────────────────
def breadth_at(closes, t, members):
    """last CLOSED 1h bar ≤ t (≤ 2 h old) → (breadth or nan, coins with data, bar close)."""
    idx = closes.index
    i = idx.searchsorted(t, side="right") - 1
    if i < 4 or (t - idx[i]) > pd.Timedelta(hours=2):
        return np.nan, 0, None
    H = idx[i]
    H4 = H - pd.Timedelta(hours=4)
    if H4 not in idx:
        return np.nan, 0, H
    cols = [s for s in members if s in closes.columns]
    r = closes.loc[H, cols] / closes.loc[H4, cols] - 1
    r = r.dropna()
    n = len(r)
    return ((r > 0).sum() / n if n >= MINC else np.nan), n, H


def v6_members(qv1h, closes, t, meta, eligible_set):
    """bot-scan reconstruction (TODAY's config): top-50 by rolling 24 h quote volume over the 24 closed 1h bars ≤ t; COIN-only, no Alpha, onboard ≥ 90 d.
    → (members, rank of every eligible sym incl. BTC/ETH for the rank validation)."""
    idx = qv1h.index
    i = idx.searchsorted(t, side="right") - 1
    if i < 24:
        return [], {}
    win = qv1h.iloc[i - 23:i + 1]
    s = win.sum(min_count=20).dropna()
    tm = int(t.value // 10**6)

    def ok(sym):
        m = meta.get(sym) or {}
        ut, sub, ob = m.get("underlyingType"), m.get("underlyingSubType") or [], m.get("onboardDate")
        if ut is not None and ut != "COIN":
            return False
        if any("alpha" in str(x).lower() for x in sub):
            return False
        if ob and int(ob) > tm - V6_AGE_D * DAY:
            return False
        return True
    s = s[[x for x in s.index if (x in eligible_set or x in EXCL) and ok(x)]].sort_values(ascending=False)
    rank = {x: k + 1 for k, x in enumerate(s.index)}
    return [x for x in s.index if x not in EXCL][:50], rank


def boot_p(blk, kpt, n=MH.BOOT_N, seed=MH.BOOT_SEED):
    z = pd.DataFrame(dict(pct=blk.pct.values, day=blk.day.values))
    r = pd.DataFrame(dict(pct=kpt.pct.values, day=kpt.day.values))
    return MH.p_zone_below(z, r, n, seed) if len(z) and len(r) else None


def split_row(g, col):
    s = g[g[col].notna()]
    b, k = s[s[col] == True], s[s[col] == False]  # noqa: E712
    f = lambda x: f"{len(x)} · {(x.pct > 0).mean():.0%} · {x.pct.mean():+.3f}" if len(x) else "0 · – · –"
    gap = (k.pct.mean() - b.pct.mean()) if len(b) and len(k) else np.nan
    p = boot_p(b, k)
    return dict(n=len(s), unscored=len(g) - len(s), blocked=f(b), kept=f(k), gap=gap, p=p, nb=len(b), nk=len(k),
                ab=b.pct.mean() if len(b) else np.nan, ak=k.pct.mean() if len(k) else np.nan)


# ───────────────────────── main ─────────────────────────
def main():
    fetch = "--no-fetch" not in sys.argv
    api = Api(fetch)
    now_ms = int(time.time() * 1000)
    now_ms = now_ms // HOUR * HOUR
    os.makedirs(CACHE, exist_ok=True)
    cohort, export_name = load_cohort()
    print(f"cohort: {cohort.src.value_counts().to_dict()} · export {export_name}", flush=True)

    meta = exchange_info(api)
    cands = set(meta) | set(MH.UNIVERSE_F108_RETIRED) | set(cohort.pair)
    cands |= {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(OLD_DAILY, "*.csv"))}
    cands = sorted(s for s in cands if s.endswith("USDT"))
    print(f"candidate symbols: {len(cands)}", flush=True)

    vol, cls, src_d = {}, {}, {}
    for k, s in enumerate(cands):
        rows = klines_1d(api, s, now_ms)
        if not rows:
            rows, src_d[s] = old_daily(s), "repo-daily-cache"
        else:
            src_d[s] = "fapi"
        if rows:
            ix = pd.to_datetime([r[0] for r in rows], unit="ms")
            dup = ~ix.duplicated(keep="last")
            vol[s] = pd.Series([r[2] for r in rows], index=ix)[dup]
            cls[s] = pd.Series([r[1] for r in rows], index=ix)[dup]
        if k % 100 == 0:
            print(f"  1d {k}/{len(cands)} weight {api.used}", flush=True)
    vol_d = pd.DataFrame(vol).sort_index()
    vol_d = vol_d[~vol_d.index.duplicated()]
    cls_d = pd.DataFrame(cls).sort_index()
    nodata = [s for s in cands if s not in vol]
    stable = {s for s in vol_d.columns if is_stable(s, meta, cls_d)}
    eligible = [s for s in vol_d.columns if s not in EXCL and s not in stable]

    # V4 floor — pre-registered: median over F108 of the 7-day avg quote volume 2026-10-23 … 10-29, frozen
    w = vol_d.loc[(vol_d.index >= REG_DAY - pd.Timedelta(days=7)) & (vol_d.index < REG_DAY), [s for s in MH.UNIVERSE_F108_RETIRED if s in vol_d]]
    floor = float(w.mean().median())
    print(f"V4 floor = ${floor/1e6:.1f}M", flush=True)

    days = pd.date_range(pd.Timestamp("2026-06-01"), pd.Timestamp(now_ms, unit="ms").normalize(), freq="D")
    U = daily_universes(vol_d, eligible, meta, days, floor)

    # which symbols need 1h data: anything that is ever a member + top-150 of the same/previous day (V6 rolling-24 h pool) + F108 + fills
    need = set(MH.UNIVERSE_F108_RETIRED) | set(cohort.pair)
    for var in U.values():
        for mem in var.values():
            need |= set(mem)
    vd = vol_d[[s for s in eligible if s in vol_d]]
    for d in days:
        for dd in (d, d - pd.Timedelta(days=1)):
            if dd in vd.index:
                need |= set(vd.loc[dd].dropna().sort_values(ascending=False).index[:150])
    need |= EXCL
    need = sorted(need)
    print(f"1h symbols: {len(need)}", flush=True)
    c1, q1, miss1 = {}, {}, []
    for k, s in enumerate(need):
        rows = klines_1h(api, s, now_ms)
        if rows:
            ix = pd.to_datetime([r[0] + HOUR for r in rows], unit="ms")     # index = bar CLOSE time
            dup = ~ix.duplicated(keep="last")
            c1[s] = pd.Series([r[1] for r in rows], index=ix)[dup]
            q1[s] = pd.Series([r[2] for r in rows], index=ix)[dup]
        else:
            miss1.append(s)
        if k % 50 == 0:
            print(f"  1h {k}/{len(need)} weight {api.used} req {api.req}", flush=True)
    closes = pd.DataFrame(c1).sort_index()
    closes = closes[~closes.index.duplicated()]
    qv1h = pd.DataFrame(q1).sort_index()
    qv1h = qv1h[~qv1h.index.duplicated()]
    print(f"1h matrix {closes.shape}, requests {api.req}", flush=True)

    # ── score every fill under every variant
    elig_set = set(eligible)
    rec = []
    for r in cohort.itertuples():
        d = r.o.normalize()
        row = dict(src=r.src, k=r.k, pair=r.pair)
        b, n, _ = breadth_at(closes, r.o, MH.UNIVERSE_F108_RETIRED)
        row.update(F108_b=b, F108_n=n)
        for v in ("V1", "V2", "V3", "V4", "V5"):
            mem = U[v].get(d, [])
            b, n, _ = breadth_at(closes, r.o, mem)
            row.update({f"{v}_b": b, f"{v}_n": n, f"{v}_size": len(mem), f"{v}_nodata": sum(1 for s in mem if s not in closes.columns)})
        mem6, rk = v6_members(qv1h, closes, r.o, meta, elig_set)
        b, n, _ = breadth_at(closes, r.o, mem6)
        row.update(V6_b=b, V6_n=n, V6_size=len(mem6), V6_rank=rk.get(r.pair, np.nan))
        rec.append(row)
    S = cohort.merge(pd.DataFrame(rec), on=["src", "k", "pair"], how="left")
    VARS = ["F108", "V1", "V2", "V3", "V4", "V5", "V6"]
    for v in VARS:
        S[f"{v}_hot"] = np.where(S[f"{v}_b"].isna(), None, S[f"{v}_b"] > HOT)
        S[f"{v}_hot"] = S[f"{v}_hot"].astype(object)
    S.to_csv(OUT_FILLS, index=False)

    # ── V6 validation vs the fills' own entry_pair_rank stamp (bot rank of the traded pair)
    vv = S[S["rank"].notna() & S.V6_rank.notna()]
    v6_val = dict(n=len(vv), med_abs=float((vv["rank"] - vv.V6_rank).abs().median()) if len(vv) else np.nan,
                  within5=float(((vv["rank"] - vv.V6_rank).abs() <= 5).mean()) if len(vv) else np.nan,
                  spear=float(vv["rank"].rank().corr(vv.V6_rank.rank())) if len(vv) > 3 else np.nan,
                  stamped=int(S["rank"].notna().sum()))

    # ── turnover per variant (daily membership; V6 at 00:00 UTC each day)
    def turnover(mdict):
        ds = sorted(mdict)
        ins = [len(set(mdict[b]) - set(mdict[a])) for a, b in zip(ds, ds[1:]) if mdict[a] and mdict[b]]
        wk = [len(set(mdict[ds[i]]) - set(mdict[ds[i - 7]])) for i in range(7, len(ds)) if mdict[ds[i]] and mdict[ds[i - 7]]]
        size = np.mean([len(m) for m in mdict.values() if m])
        return np.mean(ins) * 7, np.mean(wk), size
    U6 = {}
    for d in days:
        if d >= H1_START + pd.Timedelta(days=2) and d <= closes.index.max():
            U6[d] = v6_members(qv1h, closes, d, meta, elig_set)[0]
    TURN = {v: turnover(U[v]) for v in U}
    TURN["V6"] = turnover(U6)
    TURN["F108"] = (0.0, 0.0, 108)

    # dead members: a member with no 1h bar in the 24 h before the day start — the longest run of consecutive member-days it stays in
    def dead(mdict):
        run, best = {}, 0
        for d in sorted(mdict):
            mem = mdict[d]
            if d < H1_START + pd.Timedelta(days=1) or d > closes.index.max():
                continue
            win = closes.loc[(closes.index > d - pd.Timedelta(hours=24)) & (closes.index <= d)]
            alive = set(win.columns[win.notna().any()])
            deadnow = {s for s in mem if s not in alive}
            run = {s: run.get(s, 0) + 1 for s in deadnow}
            best = max([best] + list(run.values()))
        return best
    DEAD = {v: dead(U[v]) for v in U}
    DEAD["V6"] = dead(U6)
    DEAD["F108"] = dead({d: list(MH.UNIVERSE_F108_RETIRED) for d in days})

    # ── hourly breadth distribution per variant (Jun-18 → now, every closed hour)
    r4 = closes / closes.shift(4) - 1
    r4 = r4[r4.index >= pd.Timestamp("2026-06-18")]
    up = (r4 > 0)
    has = r4.notna()

    def hourly(mem_of_hour):
        vals = []
        for H in r4.index:
            mem = [s for s in mem_of_hour(H) if s in r4.columns]
            n = int(has.loc[H, mem].sum()) if mem else 0
            vals.append(up.loc[H, mem].sum() / n if n >= MINC else np.nan)
        return pd.Series(vals, index=r4.index).dropna()
    DIST = {"F108": hourly(lambda H: MH.UNIVERSE_F108_RETIRED)}
    for v in U:
        DIST[v] = hourly(lambda H, v=v: U[v].get((H - pd.Timedelta(minutes=1)).normalize(), []))
    DIST["V6"] = hourly(lambda H: U6.get((H - pd.Timedelta(minutes=1)).normalize(), []))

    # ── F108 drift by month (monthly total quote vol, ranks among eligible alts)
    drift = []
    for mo in pd.period_range("2026-06", pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m"), freq="M"):
        sl = vd[(vd.index >= mo.start_time) & (vd.index <= mo.end_time)]
        tot = sl.sum(min_count=1).dropna().sort_values(ascending=False)
        rk = {s: i + 1 for i, s in enumerate(tot.index)}
        f = list(MH.UNIVERSE_F108_RETIRED)
        no = [s for s in f if s not in rk or tot.get(s, 0) <= 0]
        o100 = [s for s in f if rk.get(s, 10**6) > 100]
        o50 = [s for s in f if rk.get(s, 10**6) > 50]
        ov = [len(set(MH.UNIVERSE_F108_RETIRED) & set(U["V1"][d])) for d in U["V1"] if mo.start_time <= d <= mo.end_time]
        drift.append(dict(month=str(mo), out100=len(o100), out50=len(o50), nodata=len(no), v1_overlap=np.mean(ov) if ov else np.nan,
                          days=len(sl)))

    # ── delisted / not-on-fapi
    live_ei = {m["symbol"]: m.get("status") for m in _jload(os.path.join(CACHE, "exchangeInfo_live.json")).get("symbols", [])}
    f108_status = {s: live_ei.get(s, "not in live exchangeInfo") for s in MH.UNIVERSE_F108_RETIRED}
    f108_bad = {s: st for s, st in f108_status.items() if st != "TRADING"}
    fallback = sorted(s for s, x in src_d.items() if x == "repo-daily-cache" and s in vol)

    extra = impact_section(S, VARS, TURN, DEAD, v6_val)
    write_report(S, VARS, TURN, DEAD, DIST, drift, floor, v6_val, f108_bad, nodata, fallback, miss1, stable, export_name, api, need, closes, extra)
    print("done", flush=True)


def _block(S, name):
    return S if name == "all" else S[S.src == name]


def write_report(S, VARS, TURN, DEAD, DIST, drift, floor, v6_val, f108_bad, nodata, fallback, miss1, stable, export_name, api, need, closes, extra=()):
    head = open(REPORT).read().split("\n---\n", 1)[0]
    L = [head.rstrip(), "", "---", "", f"_Results generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `scripts/study_ml_mkt_hot_rolling.py` (export {export_name}; "
         f"{api.req} API requests this run; 1h matrix {closes.shape[1]} coins × {closes.shape[0]} bars to {closes.index.max():%Y-%m-%d %H:%M} UTC)._", "",
         "**Clarifications fixed before any result was seen (coded 2026-11-06 ~16:10 UTC, ahead of the first complete run):** B4 is read as the share of members replaced "
         "week-over-week (new members vs 7 days earlier ÷ universe size — absolute counts favour small lists); a B4 tie = within 2 points; B5 order V1 < V2 < V3 < V5 < V4 < V6; "
         "B2 'dead coin' = a member with no 1h bar in the 24 h before the day start (its stay in days is reported; breadth drops it anyway since it has no close).", "",
         f"**V4 floor (pre-registered rule, computed before the split):** median F108 member 7-day avg quote volume 2026-10-23…29 = **${floor/1e6:,.1f} M/day**, frozen.", ""]
    n_m, n_b = (S.src == "MASTER").sum(), (S.src == "B25").sum()
    L += [f"Cohort: master {n_m} (expect 158) · B25 {n_b} (expect 39). zones.csv F108 flag present on {S.F108_zones.notna().sum()} / {len(S)}.", ""]

    # pipeline check
    ag = S[S.F108_zones.notna() & S.F108_hot.notna()]
    L += ["## 1 · Pipeline check — F108 recomputed vs zones.csv", "",
          f"Recomputed F108 hot flag agrees with zones.csv on **{(ag.F108_zones == ag.F108_hot).mean():.1%}** of {len(ag)} fills "
          f"({int((ag.F108_zones != ag.F108_hot).sum())} differ). All split tables below use zones.csv for F108 (the registered flag) and this run's klines for V1…V6.", ""]
    S["F108_ref"] = S.F108_zones.where(S.F108_zones.notna(), S.F108_hot)

    # per-variant tables
    L += ["## 2 · Per-variant split (blocked = hot > 0.50 · kept = ≤ 0.50) — N · WR · avg %", "",
          "| Variant | Block | scored / unscored | blocked | kept | gap kept − blocked | P(blocked < kept) | agree w/ F108 | flips (F108 hot→not / not→hot) |", "|---|---|---|---|---|---|---|---|---|"]
    SUM = {}
    for v in VARS:
        col = "F108_ref" if v == "F108" else f"{v}_hot"
        for blk in ("MASTER", "B25", "all"):
            g = _block(S, blk)
            r = split_row(g, col)
            both = g[g.F108_ref.notna() & g[col].notna()]
            agree = (both.F108_ref == both[col]).mean() if len(both) else np.nan
            f1 = int(((both.F108_ref == True) & (both[col] == False)).sum())  # noqa: E712
            f2 = int(((both.F108_ref == False) & (both[col] == True)).sum())  # noqa: E712
            SUM[(v, blk)] = dict(r, agree=agree, f1=f1, f2=f2)
            L.append(f"| {v} | {blk} | {r['n']} / {r['unscored']} | {r['blocked']} | {r['kept']} | {r['gap']:+.3f} | "
                     f"{(r['p'] if r['p'] is not None else float('nan')):.2f} | {agree:.0%} | {f1} / {f2} |")
    L.append("")

    # breadth distribution + cost + turnover
    L += ["## 3 · Breadth distribution, coverage, turnover, live cost", "",
          "Hourly breadth, every closed hour 2026-06-18 → now (is 0.50 still 'hot'? share of hours hot should sit near F108's):", "",
          "| Variant | universe size (avg) | hours | median | p25 / p75 | share of hours hot | fills: median breadth | breadth corr w/ F108 (fills) | "
          "turnover ins/week (day-over-day summed) | week-over-week new members | longest stay of a dead member (days) | weight / hour (live) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARS:
        d = DIST[v]
        ins_w, wk, size = TURN[v]
        fb = S[f"{v}_b"]
        corr = fb.corr(S.F108_b) if v != "F108" else 1.0
        wph = {"F108": "≈108 (1 kline call/coin, limit 5, w 1)", "V1": "≈50 + 40/day ticker", "V2": "≈100 + 40/day ticker",
               "V3": "≈50 + 40/day ticker", "V4": f"≈{size:.0f} + 40/day ticker", "V5": "≈50 + 40/day ticker (+exchangeInfo 1/day)",
               "V6": "≈50 (+ tickers the scan already pulls)"}[v]
        L.append(f"| {v} | {size:.0f} | {len(d)} | {d.median():.2f} | {d.quantile(.25):.2f} / {d.quantile(.75):.2f} | {(d > HOT).mean():.0%} | "
                 f"{fb.median():.2f} | {corr:.2f} | {ins_w:.1f} | {wk:.1f} | {DEAD[v]} | {wph} |")
    L += ["", "Weight / hour = breadth needs the last 5 closed 1h bars of every member once per hour (`/fapi/v1/klines` limit 5 = weight 1 each); "
          "membership needs ONE `/fapi/v1/ticker/24hr` (all symbols, weight 40) at 00:00:30 UTC per day — at that moment the 24 h rolling volume IS the previous UTC day "
          "(V1/V2); V3/V4/V5 keep the last 7 of those daily snapshots (or one 1d-kline call per perp on cold start, ≈550 × weight 1). "
          "The engine already pulls 5m/1h data for its scan pairs, so the incremental cost is lower still. Budget headroom: 1,200 weight/min = 72,000/h.", ""]

    # V6
    L += ["## 4 · V6 (bot scan universe) reconstruction check", "",
          f"Fills carrying entry_pair_rank: {v6_val['stamped']}; with a V6 rank for the pair: {v6_val['n']} · median |bot rank − reconstructed rank| "
          f"{v6_val['med_abs']:.1f} · within ±5: {v6_val['within5']:.0%} · Spearman {v6_val['spear']:.2f}. "
          "The reconstruction applies TODAY's scan config (COIN-only, no Alpha, onboard ≥ 90 d, top-50 by 24 h volume); "
          "the window ran with other settings at times (pairs limit / listing filter / blacklist changes), so V6 on old fills is an approximation, not the as-was scan.", ""]

    # drift
    L += ["## 5 · F108 drift by month (ranks by monthly total quote volume, BTC/ETH/stables out)", "",
          "| Month | days | F108 outside top-100 | outside top-50 | no volume (delisted / not listed) | avg daily overlap F108 ∩ V1 (of 50) |", "|---|---|---|---|---|---|"]
    for d in drift:
        L.append(f"| {d['month']} | {d['days']} | {d['out100']} | {d['out50']} | {d['nodata']} | {d['v1_overlap']:.1f} |")
    L += ["", f"F108 members not TRADING in today's exchangeInfo: {len(f108_bad)}" + (f" — {', '.join(f'{s} ({st})' for s, st in sorted(f108_bad.items()))}" if f108_bad else ""), ""]

    # coverage / blind spots
    unsc = {v: int(S[f'{v}_b'].isna().sum()) for v in VARS}
    L += ["## 6 · Coverage & blind spots", "",
          f"- Unscored fills per variant: " + " · ".join(f"{v} {n}" for v, n in unsc.items()) + ".",
          f"- Candidate symbols with no 1d data anywhere (fapi 400 and not in the repo daily cache): {len(nodata)}" + (f" — {', '.join(nodata[:40])}{' …' if len(nodata) > 40 else ''}" if nodata else "") + ".",
          f"- Delisted coins whose daily volume came from the repo cache (fapi no longer serves them; cache ends 2026-10-11): {len(fallback)}" + (f" — {', '.join(fallback[:40])}{' …' if len(fallback) > 40 else ''}" if fallback else "") + ".",
          f"- 1h symbols with no 1h klines (delisted before / not served): {len(miss1)} of {len(need)}" + (f" — {', '.join(miss1[:40])}" if miss1 else "") + ". "
          "A coin delisted mid-window that fapi no longer serves cannot enter breadth → slight survivorship in V1…V6 (and in F108's recompute) for those hours.",
          f"- Stablecoin / pegged perps excluded: {', '.join(sorted(stable)) or 'none'}.",
          "- V6 uses today's scan config and the current exchangeInfo metadata (onboardDate / subtype) — not the as-was config per day.",
          "- Breadth reads the 1h close series only; no volume weighting (same as the registered measure).", ""]
    L += list(extra)
    open(REPORT, "w").write("\n".join(L) + "\n")
    json.dump({f"{k[0]}|{k[1]}": {a: (None if (isinstance(b, float) and np.isnan(b)) else b) for a, b in v.items()} for k, v in SUM.items()},
              open(os.path.join(CACHE, "summary.json"), "w"), default=str, indent=1)



# ───────────────────────── selection (pre-registered B1→B5) + impact on every Market-hot-dependent line ─────────────────────────
SIMPLE_ORDER = ["V1", "V2", "V3", "V5", "V4", "V6"]     # B5 simplicity, fixed before results: one daily list < bigger list < 7-day avg < + age < floor < hourly bot scan
TIE_REL = 0.02                                          # B4: relative weekly turnover within 2 points of the best = a tie → B5 decides
REPLACE_DIR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/ml"
DL_HOT, DL_LEADA, DL_TP = 416, 481, 479


def _hotcol(v):
    return "F108_ref" if v == "F108" else f"{v}_hot"


def _fmap(S, v):
    """(k 'YYYY-MM-DDTHH:MM:SS', pair) → True / False / None for the variant's hot flag."""
    out = {}
    for k, p, x in zip(S.k, S.pair, S[_hotcol(v)]):
        out[(k, p)] = None if (x is None or (isinstance(x, float) and np.isnan(x))) else bool(x)
    return out


def _st(g, col="pct"):
    return f"{len(g)} · {(g[col] > 0).mean():.0%} · {g[col].mean():+.3f}" if len(g) else "0"


def select(S, TURN, DEAD, v6_val):
    L = ["## 7 · Selection walk-through (pre-registered order B1 → B5; impact numbers in §8 are NOT used)", "",
         "| Variant | B1 live-cheap | B2 adapts (turnover > 0 · dead member stay) | B3 agree w/ F108 · sign master · sign B25 | B4 week-over-week replaced (% of members) | B5 rank | passes B1–B3 |",
         "|---|---|---|---|---|---|---|"]
    ok = []
    for v in SIMPLE_ORDER:
        b1 = True
        b1_txt = "yes"
        if v == "V6":
            b1 = bool(v6_val["n"] and v6_val["med_abs"] <= 5)
            b1_txt = "yes (scan already pulls tickers)" if b1 else f"NOT reconstructable honestly (median rank error {v6_val['med_abs']:.0f})"
        ins_w, wk, size = TURN[v]
        b2 = wk > 0
        both = S[S.F108_ref.notna() & S[_hotcol(v)].notna()]
        agree = (both.F108_ref == both[_hotcol(v)]).mean()
        sgn = {}
        for blk in ("MASTER", "B25"):
            g = S[(S.src == blk) & S[_hotcol(v)].notna()]
            b, k = g[g[_hotcol(v)] == True], g[g[_hotcol(v)] == False]  # noqa: E712
            sgn[blk] = bool(len(b) and len(k) and b.pct.mean() < k.pct.mean())
        b3 = sgn["MASTER"] and sgn["B25"]
        rel = wk / size if size else np.nan
        if b1 and b2 and b3:
            ok.append((v, rel))
        L.append(f"| {v} | {b1_txt} | {'yes' if b2 else 'NO'} ({wk:.1f} new / week · dead stay {DEAD[v]} d) | {agree:.0%} · "
                 f"{'blocked<kept' if sgn['MASTER'] else 'FLIPPED'} · {'blocked<kept' if sgn['B25'] else 'FLIPPED'} | {rel:.1%} | {SIMPLE_ORDER.index(v) + 1} | {'✅' if b1 and b2 and b3 else '✗'} |")
    if not ok:
        L += ["", "**No variant passes B1–B3 → no rolling universe is recommended; keep F108 and re-freeze (see verdict).**", ""]
        return L, None
    best = min(r for _, r in ok)
    tied = [v for v, r in ok if r <= best + TIE_REL]
    win = min(tied, key=SIMPLE_ORDER.index)
    L += ["", f"B4: lowest relative turnover {best:.1%}; within {TIE_REL:.0%} points: {', '.join(tied)} → B5 simplicity → **winner = {win}**.", ""]
    return L, win


def _load_b25_account():
    ex = _export()
    A = pd.read_csv(ex, low_memory=False)
    A = A[(A.status.astype(str).str.upper() == "CLOSED")].copy()
    A["o"] = pd.to_datetime(A.opened_at, format="mixed", utc=True).dt.tz_convert(None)
    A = A[A.o >= B25_START].copy()
    A["u"] = pd.to_numeric(A.pnl, errors="coerce").fillna(0.0)
    A["k"] = A.o.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return A, ex


def _master_ml():
    M0 = pd.read_csv(MASTER, low_memory=False)
    nv = lambda s: pd.to_datetime(s, format="mixed", utc=True).dt.tz_convert(None)
    span = M0.assign(o=nv(M0.opened_at), c=nv(M0.closed_at)).groupby("era").agg(a=("o", "min"), b=("c", "max"))
    M = M0[M0.stack_keep.astype(str).eq("True") & ~SW._is_probe(M0) & (M0.entry_strategy == "MOMENTUM") & (M0.direction == "LONG")].copy()
    M["o"] = nv(M.opened_at)
    M["c"] = nv(M.closed_at)
    M["k"] = M.o.dt.strftime("%Y-%m-%dT%H:%M:%S")
    M["pct"] = pd.to_numeric(M.stack_pct, errors="coerce")
    return M, span


def impact_section(S, VARS, TURN, DEAD, v6_val):
    import review_math as RM
    import scout_gatekit as GK
    S["F108_ref"] = S.F108_zones.where(S.F108_zones.notna(), S.F108_hot)
    sel, win = select(S, TURN, DEAD, v6_val)
    L = list(sel)
    show = ["F108"] + [v for v in VARS if v != "F108"]
    star = lambda v: f"**{v} ★**" if v == win else v

    # ── ① ML_MKT_HOT split + daily compound
    A, ex = _load_b25_account()
    t0, t1 = A.o.min(), pd.to_datetime(A.closed_at, format="mixed", utc=True).dt.tz_convert(None).max()
    FLOWS = [("2026-10-26T13:50:00", 7100.0)]
    mlk = set(zip(S[S.src == "B25"].k, S[S.src == "B25"].pair))
    A["is_ml"] = [(k, p) in mlk for k, p in zip(A.k, A.pair)]
    g = A[A.is_ml]
    eq = np.where(g.o < pd.Timestamp("2026-10-26 13:50"), 2900.0, 10000.0)
    x1 = float(np.nanmedian(pd.to_numeric(g.notional_value, errors="coerce") / eq / np.where(pd.to_numeric(g.cell_multiplier, errors="coerce") == 1.5, 1.5, 1.0)))
    M, span = _master_ml()

    def b25_dcr(fm):
        acc = A.copy()
        acc["ua"] = [0.0 if (m and fm.get((k, p)) is True) else u for k, p, u, m in zip(acc.k, acc.pair, acc.u, acc.is_ml)]
        return RM.dcr_account(acc, acc.is_ml, "ua", 2900, t0, t1, FLOWS)

    def m_dcr(fm):
        vals = []
        for era, (a, b) in span.iterrows():
            h = M[M.era == era]
            if not len(h):
                continue
            days = pd.date_range(a.normalize(), b.normalize())
            keep = h[[fm.get((k, p)) is not True for k, p in zip(h.k, h.pair)]] if fm is not None else h
            r = (keep.pct * x1 / 100).groupby(keep.c.dt.normalize()).sum().reindex(days).fillna(0)
            vals.append(((1 + r).prod() ** (1 / len(days)) - 1) * 100)
        return float(np.mean(vals))
    base_b, base_m = b25_dcr({}), m_dcr(None)
    L += ["## 8 · Impact of switching the list — every Market-hot-dependent momentum-long line (F108 = before · ★ = the B1–B5 winner)", "",
          f"### 8.1 ML_MKT_HOT itself (DL {DL_HOT})", "",
          f"Daily compound = the sleeve's DCR with the block ON (B25: real account path, review_math.dcr_account, start 2,900, flow 2026-10-26 13:50 +7,100; "
          f"master: per-era calendar-day DCR at the 1× notional ruler X1 = {x1:.3f} of equity, mean over eras). No block: B25 {base_b:+.3f} %/day · master {base_m:+.3f} %/day.", "",
          "| List | Block | blocked N · WR · avg % | kept N · WR · avg % | gap | P(blocked < kept) | DCR block ON (%/day) |", "|---|---|---|---|---|---|---|"]
    for v in show:
        fm = _fmap(S, v)
        for blk in ("MASTER", "B25"):
            r = split_row(S[S.src == blk], _hotcol(v))
            d = m_dcr(fm) if blk == "MASTER" else b25_dcr(fm)
            L.append(f"| {star(v)} | {blk} | {r['blocked']} | {r['kept']} | {r['gap']:+.3f} | {(r['p'] if r['p'] is not None else float('nan')):.2f} | {d:+.3f} |")
    L.append("")
    # frozen gate (forward fills ≥ DL 416 + 10 min, the scout's own cohort)
    since = GK.dl_deploy_ms(DL_HOT, lag_min=10, label="ML_MKT_HOT")
    c, newest = MH.load([ex])
    c = c[c.entry_strategy.astype(str) == "MOMENTUM"]
    c = c[c.o >= pd.Timestamp(int(since), unit="ms")] if since else c.iloc[0:0]
    c, _ox = GK.split_operator_closed(c) if len(c) else (c, c)
    c = c.copy()
    c["_kk"] = c.o.dt.strftime("%Y-%m-%dT%H:%M:%S")
    L += ["Frozen gate (CURRENT_STATE 👁 ML_MKT_HOT: forward fills ≥ DL 416 commit + 10 min; zone N ≥ 15 on ≥ 8 days → decided once):", "",
          "| List | forward zone / rest / unscored | crossing reached? | gate verdict (at crossing, else live) |", "|---|---|---|---|"]
    for v in show:
        fm = _fmap(S, v)
        z = [fm.get((k, p)) for k, p in zip(c._kk, c.pair)]
        cc = c.assign(zone=["unscored" if x is None else ("zone" if x else "rest") for x in z])
        sc = cc[cc.zone != "unscored"].sort_values("o")
        pre = MH.crossing_prefix(sc) if len(sc) else None
        if pre is not None:
            stt, det = MH.verdict(pre[pre.zone == "zone"], pre[pre.zone == "rest"])
            cr = f"yes, at {pre.o.max():%m-%d %H:%M}"
        else:
            stt, det = MH.verdict(sc[sc.zone == "zone"], sc[sc.zone == "rest"])
            cr = "no"
        L.append(f"| {star(v)} | {(cc.zone == 'zone').sum()} / {(cc.zone == 'rest').sum()} / {(cc.zone == 'unscored').sum()} | {cr} | {stt} — {det} |")
    L.append("")

    # ── ② ML_LEAD_A
    import scout_ml_lead_a_watch as LAW
    import study_ml_fixed_tp_recorded as RT
    s481 = GK.dl_deploy_ms(DL_LEADA, lag_min=10, label="ML_LEAD_A")
    F, _n, _ox = LAW.cohort(paths=[ex], since=pd.Timestamp(int(s481), unit="ms") if s481 else None)
    F["_kk"] = RT._key(F.opened_at)
    L += [f"### 8.2 ML_LEAD_A (DL {DL_LEADA}) — cell (EMA20 slope > {LAW.SLOPE_MIN} ∧ RSI prev ≤ {LAW.RSI_PREV_MAX}) inside the NOT-hot side, history + B25 + forward pooled", "",
          "| List | not-hot scored | cell N · WR · avg % | rest of not-hot N · WR · avg % | newest (B25+fwd) cell vs rest | verdict |", "|---|---|---|---|---|---|"]
    for v in show:
        fm = _fmap(S, v)
        h = [fm.get((k, p)) for k, p in zip(F._kk, F.pair)]
        if v == "F108":     # the scout's own flag (zones.csv) where present
            h = [(bool(o) if pd.notna(o) else None) if pd.notna(o) else x for o, x in zip(F.mkt_hot, h)]
        G = F.assign(mkt_hot=h)
        nh = G[G.mkt_hot.map(lambda x: x is False) & G.cell.notna()]
        stt, det = LAW.verdict(nh)
        nw = nh[nh.block != "MASTER"]
        L.append(f"| {star(v)} | {len(nh)} | {_st(nh[nh.cell == True])} | {_st(nh[nh.cell == False])} | "  # noqa: E712
                 f"{_st(nw[nw.cell == True])} vs {_st(nw[nw.cell == False])} | **{stt}** — {det.split(' · newest')[0]} |")  # noqa: E712
    L.append("")

    # ── ③ replace analysis (scratchpad ml_lines.py → D with leadA / m15)
    L += ["### 8.3 Replace analysis (reports/ML_CELL_REPLACE_LEADA_2026-11-06.md) — daily compound %/day", ""]
    try:
        import runpy, io, contextlib
        old_argv = sys.argv
        sys.argv = [os.path.join(REPLACE_DIR, "ml_lines.py"), ex]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Gm = runpy.run_path(os.path.join(REPLACE_DIR, "ml_lines.py"), run_name="ml")
        sys.argv = old_argv
        D, ALL, X1r, spanr, naive = Gm["D"], Gm["ALL"], Gm["X1"], Gm["span"], Gm["naive"]
        t0r, t1r, FLr = Gm["t0"], Gm["t1"], Gm["FLOWS"]
        D = D.copy()
        D["_kk"] = D.o.dt.strftime("%Y-%m-%dT%H:%M:%S")
        B = D[D.src == "B25"].copy()
        Mm = D[D.src == "MASTER"].copy()
        eras = [e for e in spanr.index if (Mm.era == e).any()]

        def dcr_b(size):
            acc = ALL.copy()
            cur = acc.k.map(dict(zip(B.k, np.where(B.m15, 1.5, 1.0))))
            nw = acc.k.map(dict(zip(B.k, size)))
            acc["ua"] = np.where(cur.notna(), acc.u / cur * nw, acc.u)
            return RM.dcr_account(acc, acc.is_ml, "ua", 2900, t0r, t1r, FLr)

        def dcr_m(size):
            vals = []
            for e in eras:
                a, b = spanr.loc[e, "a"], spanr.loc[e, "b"]
                days = pd.date_range(a.normalize(), b.normalize())
                m = (Mm.era == e).values
                h = Mm[m]
                r = (h.pct * size[m] * X1r / 100).groupby(naive(h.closed_at).dt.normalize()).sum().reindex(days).fillna(0)
                vals.append(((1 + r).prod() ** (1 / len(days)) - 1) * 100)
            return float(np.mean(vals))
        L += ["| List | Block | before (today: 1.5× UNMATCHED, no block) | Market-hot only | + replace Lead A ×1.5 | + replace Lead A ×2 | unscored → kept |", "|---|---|---|---|---|---|---|"]
        for v in show:
            fm = _fmap(S, v)
            for blk, X, f in (("MASTER", Mm, dcr_m), ("B25", B, dcr_b)):
                hv = [fm.get((k, p)) for k, p in zip(X._kk, X.pair)]
                if v == "F108":
                    hot = X.hot.values.astype(bool)
                    nun = 0
                else:
                    nun = sum(1 for x in hv if x is None)
                    hot = np.array([x is True for x in hv])
                Aa = (X.leadA.values == 1) & ~hot
                cur = np.where(X.m15.values, 1.5, 1.0)
                vals = [f(cur), f(np.where(hot, 0.0, cur)), f(np.where(hot, 0.0, np.where(Aa, 1.5, 1.0))), f(np.where(hot, 0.0, np.where(Aa, 2.0, 1.0)))]
                L.append(f"| {star(v)} | {blk} | " + " | ".join(f"{x:+.3f}" for x in vals) + f" | {nun} |")
        L.append("")
    except Exception as e:   # the replace scripts live in the session scratchpad — a missing file is reported, never hidden
        sys.argv = old_argv if "old_argv" in dir() else sys.argv
        L += [f"⚠ replace analysis not re-run: {type(e).__name__}: {e}", ""]

    # ── ④ ML_FIXED_TP line B
    import scout_ml_fixed_tp_watch as FT
    s479 = GK.dl_deploy_ms(DL_TP, lag_min=10, label="ML_FIXED_TP")
    P, _nw, _wt, _ox = FT.cohort(paths=[ex], since=pd.Timestamp(int(s479), unit="ms") if s479 else None)
    P = P[P[[f"TP{x:g}" for x in FT.TPS]].notna().all(axis=1) & P.before.notna()].copy()
    P["_kk"] = RT._key(P.opened_at)
    L += [f"### 8.4 ML_FIXED_TP line B (DL {DL_TP}) — fixed TP on the fills KEPT by Market-hot: avg %/fill today's exit → TP +0.5 / +1 / +2 / +3", "",
          "| List | Block | N kept | today's exit | TP +0.5 | TP +1 | TP +2 | TP +3 | verdict per TP (pooled) |", "|---|---|---|---|---|---|---|---|---|"]
    for v in show:
        fm = _fmap(S, v)
        h = [fm.get((k, p)) for k, p in zip(P._kk, P.pair)]
        if v == "F108":
            h = [(bool(o) if pd.notna(o) else None) if pd.notna(o) else x for o, x in zip(P.mkt_hot, h)]
        Gp = P.assign(mkt_hot=h)
        gB = Gp[Gp.mkt_hot.map(lambda x: x is False)]
        vd = FT.verdict(gB)
        vtxt = " · ".join(f"+{k[2:]}: {st.split(' (')[0]}" for k, (st, _d) in vd.items())
        for blk, m in (("MASTER", gB.block == "MASTER"), ("B25+fwd", gB.block != "MASTER")):
            q = gB[m]
            L.append(f"| {star(v)} | {blk} | {len(q)} | {q.before.mean():+.3f} | " + " | ".join(f"{q[f'TP{x:g}'].mean():+.3f}" for x in FT.TPS)
                     + f" | {vtxt if blk == 'MASTER' else ''} |")
    L.append("")

    # ── ⑤ fills that change side under the winner
    if win:
        a, b = S.F108_ref, S[_hotcol(win)]
        ch = S[(a.notna() & b.notna() & (a != b)) | (a.notna() & b.isna())].sort_values("o")
        L += [f"### 8.5 Fills that change side F108 → {win} ({len(ch)} of {len(S)})", "", "| Block | opened (UTC) | pair | pct % | F108 | " + win + " | F108 breadth | " + win + " breadth |", "|---|---|---|---|---|---|---|---|"]
        lab = lambda x: "unscored" if x is None or (isinstance(x, float) and np.isnan(x)) else ("HOT (blocked)" if x else "not hot (kept)")
        for r in ch.itertuples():
            L.append(f"| {r.src} | {r.o:%Y-%m-%d %H:%M} | {r.pair} | {r.pct:+.3f} | {lab(getattr(r, 'F108_ref'))} | {lab(getattr(r, _hotcol(win)))} | "
                     f"{r.F108_b:.2f} | {getattr(r, win + '_b'):.2f} |")
        L.append("")
    S.attrs["winner"] = win
    L += final_section(S, win, TURN)
    return L


def final_section(S, win, TURN):
    """§9 verdict + §10 engine SPEC for the winner (text; numbers read from this run)."""
    if not win:
        return ["## 9 · Verdict", "", "No rolling variant passed B1–B3: keep F108, re-freeze it from the current top-50 and add the drift alarm.", ""]
    r = {(v, b): split_row(S[S.src == b] if b != "all" else S, _hotcol(v)) for v in ("F108", win) for b in ("MASTER", "B25", "all")}
    both = S[S.F108_ref.notna() & S[_hotcol(win)].notna()]
    agree = (both.F108_ref == both[_hotcol(win)]).mean()
    g = lambda v, b: f"{r[(v, b)]['gap']:+.3f} (P {r[(v, b)]['p']:.2f})"
    ins_w, wk, size = TURN[win]
    return [
        "## 9 · Verdict (plain words)", "",
        f"- **The split survives a rolling universe.** Every rolling variant keeps blocked < kept in master AND in B25. Winner {win}: master gap {g(win, 'MASTER')} vs F108 "
        f"{g('F108', 'MASTER')}; B25 {g(win, 'B25')} vs {g('F108', 'B25')}; all {g(win, 'all')} vs {g('F108', 'all')}. Same sign, similar size — the Market-hot effect is "
        f"not an artefact of the frozen list. It agrees with F108's hot flag on {agree:.0%} of the {len(both)} fills.",
        "- **But every rolling list shows a SMALLER master gap than F108** (+0.10 … +0.21 vs +0.23; the 50-coin lists are weakest, V1 +0.10 P 0.75). F108 was the list the "
        "split was found on, so part of its master edge is in-sample; the rolling numbers are the more honest estimate. In B25 every rolling list is stronger (+0.63 vs +0.48): "
        "3 fills on 11-06 09:11–09:47 (TAO / SUI / FET, all winners) read 0.62 on F108 but 0.48 on every rolling list.",
        "- **F108 is drifting fast:** members outside that month's top-100 by volume rose from 32 / 108 (June) to 53 (October) and 50 (November, 5 days); outside the top-50 "
        "61 → 73; 2 members are no longer trading (IPUSDT, TONUSDT SETTLING); on an average October day only ~36 of the 50 most-traded coins are in F108. Breadth on F108 is "
        "increasingly the breadth of coins nobody trades.",
        "- **The 0.50 cut still means 'hot' on every list:** median hourly breadth 0.46–0.48, 44–45 % of hours hot on all variants and on F108 — the cut does not need "
        "(and was not given) a re-fit.",
        "- **Decisions that move with the list (see §8):** ML_MKT_HOT — numbers move a little, gate still COLLECTING (forward zone 8 → 5 of 15), no verdict changes today. "
        "ML_LEAD_A — on V2 it stays MULTIPLIER CANDIDATE (cell 31 · 90 % · +0.78 vs F108 30 · 90 % · +0.81) but on every OTHER rolling list it drops to KEEP OBSERVING only "
        "because the cell falls to N 27–29 (< 30): the verdict sits on the N ≥ 30 edge and is list-fragile — treat it as borderline, not robust. Replace analysis — "
        "ranking unchanged (×2 > ×1.5 ≈ Market-hot only > before) on both lists; master levels drop ~0.005 %/day, B25 levels rise ~0.03 %/day. ML_FIXED_TP line B — "
        "still no fixed TP beats today's exit on master; +2 moves from NO EDGE to KEEP OBSERVING (master Σ Δ shrinks to ≈ 0), no candidate appears.",
        "", "### Recommendation", "",
        f"- **Switch the live measure to the rolling list {win}** (engine computes and stamps it on every trade, spec below). Do NOT re-freeze a new fixed list — it would "
        "drift again (≈ 5 of 108 members a month leave the top-100).",
        "- **Keep the DL 416 frozen gate as registered** (read on F108, pre-committed bars do not move), and add one condition before any ship: a FILTER CANDIDATE on F108 "
        f"must also show blocked < kept on {win} over the same prefix (else no ship). Once the engine stamp exists, every NEW read uses the stamped {win} value.",
        f"- **Drift alarm** (scout line, report every batch): ⚠ when > 40 % of F108 is outside the trailing-30-day top-100 (already true: 53 / 108 in October) OR the trailing-30-day "
        f"hot-flag agreement F108 vs {win} on momentum-long fills falls below 85 % (today {agree:.0%}). For the rolling stamp itself: ⚠ when < 90 members have data on an hour "
        "(the stamp goes NULL below 30).",
        "",
        f"## 10 · SPEC — `entry_mkt_breadth_r100` (winner {win}; precise enough for the engine)", "",
        "1. **Universe rule:** the 100 USDT-margined `contractType == PERPETUAL`, `status == TRADING` symbols with the highest quote volume of the PREVIOUS UTC day, "
        "after exclusions. One list per UTC day.",
        "2. **Refresh time:** 00:00:30 UTC daily, from ONE `GET /fapi/v1/ticker/24hr` (all symbols, weight 40) — at that moment `quoteVolume` = the previous UTC day "
        "(equivalently the 1d kline of D−1, field 7). Keep yesterday's list if the refresh fails; ⚠ log after 2 missed refreshes. (Study: day-over-day ≈ "
        f"{ins_w / 7:.0f} coins swap at the rank-100 boundary; week-over-week {wk:.0f} of 100 are new.)",
        "3. **Exclusions:** BTCUSDT, ETHUSDT; stablecoin / USD-pegged perps (base in USDC, FDUSD, TUSD, USDP, BUSD, USD1, USDE, DAI, PYUSD, RLUSD, U, XUSD, USDS, EUR, "
        "EURI, AEUR, USDD, BFUSD, or a 30-day close within 0.97–1.03 on > 95 % of days); non-PERPETUAL; non-USDT. NOT excluded (as tested): the pair blacklist, Alpha "
        "sub-type coins (~21 of 100 a day), new listings, non-COIN underlyings — the measure is the MARKET, not the tradable list.",
        "4. **Bar at entry:** the last CLOSED 1h bar at the entry time (bar close time ≤ entry; never the forming bar). If that bar closed more than 2 h before the entry → NULL.",
        "5. **Breadth formula:** for every member with both closes, r = close(last closed 1h bar) / close(the 1h bar 4 h earlier) − 1; breadth = #(r > 0) ÷ #(members with both "
        "closes). Strict `>` (flat counts as not up). Hot = breadth > 0.50 (frozen DL 416 cut).",
        "6. **Minimum coins:** ≥ 30 members with both closes, else the stamp is NULL (never 0, never counted on either side).",
        "7. **Data / cost:** once per hour at HH:00:05+ fetch `GET /fapi/v1/klines?interval=1h&limit=6` for the 100 members (weight 1 each → ≈ 100 weight / hour + 40 / day; "
        "budget 72,000 / hour). Cache per hour; every entry in that hour reads the same value (pure function of the hour → identical across sleeves).",
        "8. **Stamps (D13 class: universal):** `entry_mkt_breadth_r100` (float 0–1, NULL as above) + `entry_mkt_breadth_r100_n` (members with data). Also keep stamping the "
        "F108 value (`entry_mkt_breadth_f108`) during the transition so the frozen gate can be read from stamps.",
        "",
    ]

if __name__ == "__main__":
    try:
        main()
    except Halt as e:
        print(f"HALTED: {e}", flush=True)
        sys.exit(2)
