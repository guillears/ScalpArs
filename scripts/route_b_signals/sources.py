"""Route B signal-level loader — inputs (all cached under reports/ROUTE_B_SIGNALS_data/, gitignored).

  · Route B per-minute summaries: s3://<recorder bucket>/features/<date>/<HH>.csv.gz — READ-ONLY (aws s3 ls / cp through
    scripts/route_b/daily_report.aws). A closed hour is immutable → cached once (the daily report's cache is reused when it has the file).
  · decision journals: exports_archive/ + ~/Downloads scalpars_decisions_*.csv (read directly — no copy / trash side effects), deduplicated.
  · orders exports (taken fills): same places, dedup key (opened_at[:19], pair, direction), latest export wins; probes dropped.
  · 1m klines: data.binance.vision daily zips for complete UTC days (no API weight); the current / missing day from fapi /klines with a
    weight limiter (≤ 1,000 weight per minute, read X-MBX-USED-WEIGHT-1M, stop on 418 / 429).
  · funding settlements: fapi /fundingRate (real settled rates, times and marks), refreshed when the cached copy ends before the need.
"""
import collections
import glob
import re
import gzip
import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")
import requests  # noqa: E402

from common import DATA, MIN_MS, ROOT

sys.path.insert(0, os.path.join(ROOT, "scripts", "route_b"))
import daily_report as DR  # noqa: E402  (aws CLI wrapper + bucket name; S3 read-only)

FAPI = "https://fapi.binance.com"
VISION = "https://data.binance.vision/data/futures/um/daily/klines/{s}/1m/{s}-1m-{d}.zip"
DOWNLOADS = os.path.expanduser("~/Downloads")
ARCHIVE = os.path.join(ROOT, "exports_archive")
WEIGHT_CAP = 1000


# ───────────────────────────────────────────────────────────── S3 features (read-only)
def feature_keys():
    return sorted(k for k in DR.s3_keys("features/") if k.endswith(".csv.gz"))


def fetch_feature(key):
    """bytes of one hourly summary; cache = DATA/s3/<key> (copied from the daily report's cache when present)."""
    path = os.path.join(DATA, "s3", key)
    if os.path.exists(path):
        return open(path, "rb").read()
    alt = os.path.join(DR.CACHE, key)
    b = open(alt, "rb").read() if os.path.exists(alt) else None
    if b is None:
        rc, out, _ = DR.aws("s3", "cp", "--only-show-errors", f"s3://{DR.BUCKET}/{key}", "-")
        if rc != 0 or not out:
            return None
        b = out
    try:
        gzip.decompress(b)
    except (OSError, EOFError):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        f.write(b)
    os.replace(path + ".tmp", path)
    return b


BOOK_COLS = ["n_book", "mid", "mid_hi", "mid_lo", "spread_bps", "bid_usd_02", "ask_usd_02", "bid_usd_05", "ask_usd_05", "bid_usd_1",
             "ask_usd_1", "imb_05", "reach_bid_pct", "reach_ask_pct", "cost_buy_2500", "cost_sell_2500", "cost_buy_10000", "cost_sell_10000"]
SUM_COLS = ["n_trades", "buy_usd", "sell_usd", "liq_long_usd", "liq_short_usd"]
MARK_COLS = ["mark", "index", "funding"]


FB_BAND_TAGS = ("01", "02", "05", "1", "2", "5")          # the recorder's full-book layout (scripts/route_b/fullbook.py, DL 611)
FB_SIZES = (1000, 3000, 6000, 10000, 25000)
FB_COLS = (["n_fbook", "fb_spread_bps", "fb_reach_bid_pct", "fb_reach_ask_pct"]
           + [f"fb_{s}_usd_{t}" for t in FB_BAND_TAGS for s in ("bid", "ask")]
           + [f"fb_cost_{s}_{z}" for z in FB_SIZES for s in ("buy", "sell")])
OI_COLS = ["oi", "oi_chg_1m", "oi_chg_5m"]
RATIO_COLS = ["ratio_ts", "top_ls_acct", "top_ls_pos", "global_ls_acct", "taker_bs_ratio"]
MIN_CACHE_V = 2                # v2 (build item 1): the per-hour pickle also keeps the DL 611 full-book / OI / positioning rows


_HOUR_MEMO = {}                 # one read of each hourly pickle per run (load_minutes and load_fb_pos share it)


def compact_hour(key):
    """one hourly file → (book rows, mark rows, liq rows, full-book rows, OI / positioning rows) as float frames, cached as a pickle per
    hour (DATA/minutes/, versioned: a v1 pickle without the DL 611 frames is rebuilt from the cached raw file). Hours written before the
    DL 611 columns existed give empty fb / pos frames."""
    pk = os.path.join(DATA, "minutes", key.replace("/", "_").replace(".csv.gz", f".v{MIN_CACHE_V}.pkl.gz"))
    if pk in _HOUR_MEMO:
        return _HOUR_MEMO[pk]
    if os.path.exists(pk):
        _HOUR_MEMO[pk] = pd.read_pickle(pk)
        return _HOUR_MEMO[pk]
    b = fetch_feature(key)
    if b is None:
        return None
    d = pd.read_csv(io.BytesIO(b), compression="gzip")
    d = d[d.sym.str.endswith("USDT") & ~d.sym.str.contains("_")]          # coin-margined / dated contracts out (size in contracts)
    num = [c for c in d.columns if c != "sym"]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    book = d[d.n_book > 0][["sym", "minute"] + BOOK_COLS + ["n_trades", "buy_usd", "sell_usd", "max_trade_usd"]].copy()
    mark = d[d[MARK_COLS].notna().any(axis=1)][["sym", "minute"] + MARK_COLS].copy()
    liq = d[(d.liq_long_usd.fillna(0) > 0) | (d.liq_short_usd.fillna(0) > 0)][["sym", "minute", "liq_long_usd", "liq_short_usd"]].copy()
    for x in (book, mark, liq):
        for c in x.columns:
            if c not in ("sym", "minute"):
                x[c] = x[c].astype("float64" if c in ("mid", "mid_hi", "mid_lo", "mark", "index") else "float32")
    fcols = [c for c in FB_COLS if c in d.columns]
    fb = (d[d.n_fbook.fillna(0) > 0][["sym", "minute"] + fcols].astype({c: "float64" for c in fcols}).copy()
          if "n_fbook" in d.columns else pd.DataFrame(columns=["sym", "minute"] + FB_COLS))
    pcols = [c for c in OI_COLS + RATIO_COLS if c in d.columns]
    pos = (d[d[pcols].notna().any(axis=1)][["sym", "minute"] + pcols].astype({c: "float64" for c in pcols}).copy()
           if pcols else pd.DataFrame(columns=["sym", "minute"] + OI_COLS + RATIO_COLS))
    for x, cols in ((fb, FB_COLS), (pos, OI_COLS + RATIO_COLS)):
        for c in cols:
            if c not in x.columns:
                x[c] = np.nan
    res = dict(book=book, mark=mark, liq=liq, fb=fb[["sym", "minute"] + FB_COLS], pos=pos[["sym", "minute"] + OI_COLS + RATIO_COLS],
               key=key)
    pd.to_pickle(res, pk)
    old = pk.replace(f".v{MIN_CACHE_V}.pkl.gz", ".pkl.gz")         # the v1 pickle of this hour is superseded
    if os.path.exists(old):
        os.remove(old)
    _HOUR_MEMO[pk] = res
    return res


def load_minutes(keys):
    """all hours → merged frames (a coin-minute split across two hourly files: book from the fuller row, trades / liqs summed, last mark)."""
    parts = [p for p in (compact_hour(k) for k in keys) if p is not None]
    if not parts:
        raise SystemExit("no Route B hourly summary could be read (S3 access / network?) — nothing built")
    book = pd.concat([p["book"] for p in parts], ignore_index=True)
    mark = pd.concat([p["mark"] for p in parts], ignore_index=True)
    liq = pd.concat([p["liq"] for p in parts], ignore_index=True)
    book = book.sort_values(["sym", "minute", "n_book"])
    g = book.groupby(["sym", "minute"], sort=False)
    sums = g[["n_trades", "buy_usd", "sell_usd"]].sum(min_count=1)
    mx = g[["max_trade_usd"]].max()
    bk = g[BOOK_COLS].last()
    book = pd.concat([bk, sums, mx], axis=1).reset_index().sort_values(["sym", "minute"]).reset_index(drop=True)
    mark = mark.groupby(["sym", "minute"], sort=True)[MARK_COLS].last().reset_index()
    liq = liq.groupby(["sym", "minute"], sort=True)[["liq_long_usd", "liq_short_usd"]].sum().reset_index()
    if not len(mark) or not len(book):
        raise SystemExit("Route B summaries hold no mark / book rows — nothing built")
    return book, mark, liq, len(parts)


def merge_fb_pos(fb, pos):
    """the DL 611 frames across hourly files: a coin-minute split over two files keeps the full-book row with more full-book seconds
    (first_view.merge_split_minutes' rule); the OI family comes whole from the first row that has an oi, the positioning family whole
    from the row with the newest ratio_ts (those lines are filed by write time, so a minute's value may sit in the next hour's file;
    a per-column merge could attach a newer read to an older ratio_ts and pass the 10-min lag early). Pure."""
    if len(fb):
        fb = (fb.sort_values(["sym", "minute", "n_fbook"]).drop_duplicates(["sym", "minute"], keep="last")
              .sort_values(["sym", "minute"]).reset_index(drop=True))
    if len(pos):                              # each FAMILY from ONE row (never a ratio value glued to another row's ratio_ts)
        key = ["sym", "minute"]
        o = (pos[pos.oi.notna()].sort_values(key).drop_duplicates(key, keep="first")[key + OI_COLS])
        r = (pos[pos.ratio_ts.notna()].sort_values(key + ["ratio_ts"]).drop_duplicates(key, keep="last")[key + RATIO_COLS])
        pos = o.merge(r, on=key, how="outer").sort_values(key).reset_index(drop=True)
    return fb, pos


def load_fb_pos(keys):
    """the DL 611 full-book and OI / positioning rows of every readable hour (the same per-hour cache as load_minutes) → (fb, pos)."""
    parts = [p for p in (compact_hour(k) for k in keys) if p is not None]
    fb = pd.concat([p["fb"] for p in parts if len(p["fb"])] or [pd.DataFrame(columns=["sym", "minute"] + FB_COLS)], ignore_index=True)
    pos = pd.concat([p["pos"] for p in parts if len(p["pos"])] or [pd.DataFrame(columns=["sym", "minute"] + OI_COLS + RATIO_COLS)],
                    ignore_index=True)
    for x in (fb, pos):
        x["minute"] = pd.to_numeric(x["minute"]).astype("int64")
    return merge_fb_pos(fb, pos)


# ───────────────────────────────────────────────────────────── journals + orders
_STAMP = re.compile(r"^scalpars_(?:orders|decisions)_(?:paper|live)_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")


def export_stamp(name):
    """the export's UTC download stamp parsed from its file name (paper AND live) → pd.Timestamp, or None."""
    m = _STAMP.match(os.path.basename(name))
    if not m:
        return None
    return pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", tz="UTC")


def _export_files(kind, since=None):
    """paper + live exports of `kind`, one copy per name, whose name stamp ≥ since, sorted OLDEST → NEWEST by that stamp (so a
    keep='last' dedup lets the newest export win, whatever its mode)."""
    seen = {}
    for d in (ARCHIVE, DOWNLOADS):
        for f in sorted(glob.glob(os.path.join(d, f"scalpars_{kind}_*.csv"))):
            seen.setdefault(os.path.basename(f), f)                     # one copy per export name
    out = [(export_stamp(k), seen[k]) for k in seen]
    out = [(t, f) for t, f in out if t is not None and (since is None or t >= since)]
    return [f for t, f in sorted(out, key=lambda x: (x[0], os.path.basename(x[1])))]


JOURNAL_KEY = ["t", "e", "pair", "dir", "gate", "reason", "exh_status", "dist_status", "fb_status", "bl_sleeve", "strategy"]
JOURNAL_E = ("EXHAUSTION", "DISTRIBUTION", "FLATBASE", "WILLY", "BLOCKED", "BLOCK", "OPEN", "POSITION")


def load_journal(since):
    """the decision journal rows (signal-relevant event kinds) with t ≥ since, deduplicated across overlapping exports.
    → (frame, list of (file, first t, last t))."""
    fr, spans = [], []
    since_s = since.strftime("%Y-%m-%dT%H:%M:%S")
    for f in _export_files("decisions", since=since.floor("D")):    # exports downloaded before recording cannot hold its signals
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        spans.append((os.path.basename(f), str(d.t.min()), str(d.t.max())))
        d = d[(d.t.astype(str) >= since_s) & d.e.isin(JOURNAL_E)].copy()
        d["src_file"] = os.path.basename(f)
        fr.append(d)
    if not fr:
        return pd.DataFrame(), spans
    a = pd.concat(fr, ignore_index=True)
    if not len(a):
        return pd.DataFrame(columns=["t", "e", "pair", "gate"]), spans
    a = a.drop_duplicates(subset=[c for c in JOURNAL_KEY if c in a.columns], keep="last").reset_index(drop=True)
    return a, spans


ORDER_COLS = ["id", "pair", "direction", "entry_strategy", "status", "opened_at", "closed_at", "entry_price", "pnl_percentage",
              "close_reason", "funding_fee_usd", "investment", "leverage", "k", "src_file"]


def load_orders(since=pd.Timestamp("2026-11-19", tz="UTC")):
    """taken fills from every orders export (paper + live) downloaded since `since`; the NEWEST export wins a key. → (frame, probes)."""
    fr = []
    for f in _export_files("orders", since=since):
        try:
            d = pd.read_csv(f, low_memory=False).copy()
        except Exception:
            continue
        d["src_file"] = os.path.basename(f)
        fr.append(d)
    if not fr:
        return pd.DataFrame(columns=ORDER_COLS), 0
    a = pd.concat(fr, ignore_index=True).copy()               # fr is already oldest → newest by the name stamp
    if not len(a):
        return pd.DataFrame(columns=ORDER_COLS), 0
    a["k"] = a.opened_at.astype(str).str[:19] + "|" + a.pair + "|" + a.direction
    a = a.drop_duplicates("k", keep="last")
    probe = a.entry_strategy.astype(str).str.endswith("_PROBE")
    if "is_probe" in a.columns:
        probe |= a.is_probe.astype(str).str.lower().isin(("true", "1", "1.0"))
    return a[~probe].reset_index(drop=True), int(probe.sum())


# ───────────────────────────────────────────────────────────── Binance public data (rate-limited)
FUNDING_CALLS_5MIN = 400     # /fundingRate has its own 500 per 5 min per IP limit (shared with the bot's host only if same IP)


class Limiter:
    def __init__(self, cap=WEIGHT_CAP):
        self.cap, self.used, self.stopped = cap, 0, False
        self.s = requests.Session()
        self.fcalls = collections.deque()
        self.errors = 0

    def get(self, path, params, weight):
        if self.stopped:
            return None
        if self.used + weight > self.cap:
            time.sleep(61 - time.time() % 60)
            self.used = 0
        if path.endswith("/fundingRate"):
            while self.fcalls and time.time() - self.fcalls[0] > 300:
                self.fcalls.popleft()
            if len(self.fcalls) >= FUNDING_CALLS_5MIN:
                time.sleep(max(0.0, 301 - (time.time() - self.fcalls[0])))
            self.fcalls.append(time.time())
        try:
            r = self.s.get(FAPI + path, params=params, timeout=30)
        except requests.RequestException:
            self.errors += 1                                       # one network blip never kills the build: the caller sees None
            return None
        try:
            self.used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", self.used + weight))
        except ValueError:
            self.used += weight
        if r.status_code in (418, 429):
            self.stopped = True
            print(f"⛔ Binance {r.status_code} — stopping API calls for this run", file=sys.stderr)
            return None
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            self.errors += 1
            return None


_LIM = None


def limiter():
    global _LIM
    if _LIM is None:
        _LIM = Limiter()
    return _LIM


KCOLS = ["open_ms", "o", "h", "l", "c", "qv"]


def _klines_rest(sym, start_ms, end_ms):
    rows, s = [], start_ms
    while s < end_ms:
        js = limiter().get("/fapi/v1/klines", dict(symbol=sym, interval="1m", startTime=s, endTime=end_ms - 1, limit=1000), 5)
        if not js:
            break
        rows += [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7])] for r in js]
        if len(js) < 1000:
            break
        s = int(js[-1][0]) + MIN_MS
    return pd.DataFrame(rows, columns=KCOLS)


def klines_day(sym, day, now_ms):
    """1m klines of one UTC day. Complete days: vision zip (cached forever; REST fallback when the zip is not published yet, cached
    as final only once the day is > 2 h old). The current day: REST, cached with a fetch stamp and re-fetched each run."""
    d0 = int(pd.Timestamp(day, tz="UTC").value // 1_000_000)
    d1 = d0 + 1440 * MIN_MS
    final = os.path.join(DATA, "klines", sym, f"{day}.pkl.gz")
    if os.path.exists(final):
        return pd.read_pickle(final)
    part = os.path.join(DATA, "klines", sym, f"{day}.partial.pkl.gz")
    os.makedirs(os.path.dirname(final), exist_ok=True)
    if d1 <= now_ms:
        try:
            r = requests.get(VISION.format(s=sym, d=day), timeout=60)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
                if str(raw.iloc[0, 0]).startswith("open"):
                    raw = raw.iloc[1:]
                k = pd.DataFrame({"open_ms": raw[0].astype("int64"), "o": raw[1].astype(float), "h": raw[2].astype(float),
                                  "l": raw[3].astype(float), "c": raw[4].astype(float), "qv": raw[7].astype(float)})
                k.to_pickle(final)
                if os.path.exists(part):
                    os.remove(part)
                return k
        except (requests.RequestException, zipfile.BadZipFile, ValueError, KeyError):
            pass
    if os.path.exists(part):
        k = pd.read_pickle(part)
        if len(k) and (k.open_ms.max() + 2 * MIN_MS >= min(d1, now_ms) - MIN_MS):
            return k
    k = _klines_rest(sym, d0, min(d1, now_ms - MIN_MS))                  # closed minutes only
    if len(k):
        k = k[k.open_ms + MIN_MS <= now_ms]
        (k.to_pickle(final) if (d1 + 2 * 3600_000 <= now_ms and len(k) >= 1430) else k.to_pickle(part))
    return k


_DAY_CACHE = {}


def klines(sym, start_ms, end_ms, now_ms):
    """1m klines of `sym` with start ≤ open < end. Whole UTC days are cached per (symbol, day) for the run and every call slices its own
    range (never another caller's slice — the earlier (sym, first day, last day) key returned the FIRST caller's window)."""
    days = pd.date_range(pd.Timestamp(start_ms, unit="ms", tz="UTC").floor("D"), pd.Timestamp(min(end_ms, now_ms), unit="ms", tz="UTC").floor("D"),
                         freq="D")
    fr = []
    for d in days:
        key = (sym, d.strftime("%Y-%m-%d"))
        if key not in _DAY_CACHE:
            _DAY_CACHE[key] = klines_day(sym, key[1], now_ms)
        fr.append(_DAY_CACHE[key])
    fr = [f for f in fr if f is not None and len(f)]
    if not fr:
        return pd.DataFrame(columns=KCOLS)
    k = pd.concat(fr, ignore_index=True).drop_duplicates("open_ms").sort_values("open_ms")
    return k[(k.open_ms >= start_ms) & (k.open_ms < end_ms)].reset_index(drop=True)


FUNDING_PUBLISH_LAG_MS = 10 * 60_000


def merge_spans(spans):
    """union of [a, b] intervals (sorted, overlapping / touching merged). Pure."""
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def funding_history(sym, start_ms, end_ms, now_ms):
    """settled funding [(fundingTime ms, rate, mark)] in (start, end]; cached per coin with the fetched span, extended when the cache does
    not cover the need. → (list, ok) — ok False when the span could not be fetched (funding then counts 0 and the row is flagged)."""
    path = os.path.join(DATA, "funding", f"{sym}.pkl.gz")
    cols = ["t", "rate", "mark"]
    have, spans = (pd.read_pickle(path) if os.path.exists(path) else (pd.DataFrame(columns=cols), []))
    need_end = min(end_ms, now_ms - FUNDING_PUBLISH_LAG_MS)        # a settlement is published a few minutes after its time
    covered = any(a <= start_ms and b >= need_end for a, b in spans)
    if not covered and not limiter().stopped:
        a0 = int(start_ms)
        js = limiter().get("/fapi/v1/fundingRate", dict(symbol=sym, startTime=a0, endTime=int(need_end), limit=1000), 1)
        if js is not None:
            new = pd.DataFrame([(int(r["fundingTime"]), float(r["fundingRate"]), float(r.get("markPrice") or "nan")) for r in js], columns=cols)
            have = pd.concat([have, new], ignore_index=True).drop_duplicates("t", keep="last").sort_values("t")
            spans = merge_spans(spans + [(a0, int(need_end))])
            pd.to_pickle((have, spans), path)
            covered = True
    h = have[(have.t > start_ms) & (have.t <= end_ms)]
    return [(int(r.t), float(r.rate), float(r.mark) if r.mark == r.mark else np.nan) for r in h.itertuples()], bool(covered)
