"""STATE_EXIT_2026-11-21 — shared data access + walker (research only; no bot code / config touched).

Data (all cached under reports/STATE_EXIT_2026-11-21_data/):
  k1m/<PAIR>/<YYYY-MM-DD>.npz   1m klines WITH volume: t, o, h, l, c, v (base), qv (quote), tbq (taker-buy quote) — data.binance.vision daily zips
                                (0 Binance weight); days not yet on vision → fapi /fapi/v1/klines (weight read from X-MBX-USED-WEIGHT-1M, pause
                                at 900, stop on 418 / 429 / 403).
  met/<PAIR>/<YYYY-MM-DD>.npz   5m metrics (vision daily 'metrics'): t (create_time ms), oi (sum_open_interest_value USD), top_pos
                                (sum_toptrader_long_short_ratio), taker_ls (sum_taker_long_short_vol_ratio). Empty npz = not on the server.
  fund/<PAIR>.csv               funding settlements t, rate (vision monthly fundingRate zips; the current month from fapi /fapi/v1/fundingRate).
Walker = today's exit per sleeve on 1m bars (stop first on a bar touching both; resting maker TP needs 0.05 % through; time exit at the close
of the last bar inside the hold + slip), plus funding accrued at settlements (mark-scaled)."""
import io, os, time, json, zipfile, urllib.request, urllib.error, threading
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "STATE_EXIT_2026-11-21_data")
K1, MET, FUN = os.path.join(D, "k1m"), os.path.join(D, "met"), os.path.join(D, "fund")
for _d in (K1, MET, FUN):
    os.makedirs(_d, exist_ok=True)
VIS = "https://data.binance.vision/data/futures/um"
FAPI = "https://fapi.binance.com"
UA = {"User-Agent": "research-client/1.0"}
MIN, HR, DAY = 60_000, 3_600_000, 86_400_000
TAKER, MAKER = 0.05, 0.02
_lock = threading.Lock()
_W = {"used": 0}


def _get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 418, 429):
                raise SystemExit(f"HTTP {e.code} on {url} — stopping")
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def _fapi(path):
    """fapi GET with the used-weight header read; pause when > 900 / min."""
    with _lock:
        if _W["used"] > 900:
            time.sleep(61); _W["used"] = 0
    req = urllib.request.Request(FAPI + path, headers=UA)
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                w = r.headers.get("X-MBX-USED-WEIGHT-1M")
                with _lock:
                    _W["used"] = int(w) if w else _W["used"] + 5
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 418, 429):
                raise SystemExit(f"fapi HTTP {e.code} — stopping")
            return None
        except Exception:
            time.sleep(3 * (i + 1))
    return None


def _csv_rows(raw):
    lines = raw.decode().strip().split("\n")
    if lines and not lines[0][:1].isdigit():
        lines = lines[1:]
    return [ln.split(",") for ln in lines if ln]


def fetch_kday(p, ds):
    f = os.path.join(K1, p, ds + ".npz")
    if os.path.exists(f):
        return 0
    os.makedirs(os.path.dirname(f), exist_ok=True)
    z = _get(f"{VIS}/daily/klines/{p}/1m/{p}-1m-{ds}.zip")
    rows = None
    if z:
        zz = zipfile.ZipFile(io.BytesIO(z))
        rows = _csv_rows(zz.open(zz.namelist()[0]).read())
    else:
        t0 = int(pd.Timestamp(ds).value // 10**6)
        if t0 > int(time.time() * 1000) - 3 * DAY:          # recent day not yet on vision → fapi (weight 5 per ≤ 1,000 bars)
            rows = []; t = t0; now = int(time.time() * 1000)
            while t < min(t0 + DAY, now - MIN):
                x = _fapi(f"/fapi/v1/klines?symbol={p}&interval=1m&startTime={t}&endTime={t0 + DAY - 1}&limit=1000")
                if not x:
                    break
                rows += [[str(v) for v in r] for r in x]
                t = int(x[-1][0]) + MIN
                if len(x) < 1000:
                    break
            rows = [r for r in rows if int(r[0]) + MIN <= now]          # closed bars only
    if not rows:
        np.savez_compressed(f, t=np.array([], "int64")); return 0
    a = np.array([[float(r[k]) for k in (0, 1, 2, 3, 4, 5, 7, 10)] for r in rows])
    if a[:, 0].max() > 1e14:
        a[:, 0] //= 1000
    np.savez_compressed(f, t=a[:, 0].astype("int64"), o=a[:, 1], h=a[:, 2], l=a[:, 3], c=a[:, 4], v=a[:, 5].astype("float32"),
                        qv=a[:, 6].astype("float32"), tbq=a[:, 7].astype("float32"))
    return 1


def fetch_mday(p, ds):
    f = os.path.join(MET, p, ds + ".npz")
    if os.path.exists(f):
        return 0
    os.makedirs(os.path.dirname(f), exist_ok=True)
    z = _get(f"{VIS}/daily/metrics/{p}/{p}-metrics-{ds}.zip")
    if not z:
        np.savez_compressed(f, t=np.array([], "int64")); return 0
    zz = zipfile.ZipFile(io.BytesIO(z))
    x = pd.read_csv(zz.open(zz.namelist()[0]))
    t = (pd.to_datetime(x.create_time) - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    g = lambda c: pd.to_numeric(x[c], errors="coerce").values.astype("float64") if c in x else np.full(len(x), np.nan)
    np.savez_compressed(f, t=t.values.astype("int64"), oi=g("sum_open_interest_value"), top_pos=g("sum_toptrader_long_short_ratio"),
                        taker_ls=g("sum_taker_long_short_vol_ratio"))
    return 1


def fetch_funding(p, months):
    f = os.path.join(FUN, p + ".csv")
    have = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["t", "rate", "m"])
    got = set(have.m.astype(str)) if len(have) else set()
    parts = [have]
    cur = pd.Timestamp.utcnow().strftime("%Y-%m")
    for m in sorted(set(months) - got):
        if m >= cur:
            t0 = int(pd.Timestamp(m + "-01").value // 10**6)
            x = _fapi(f"/fapi/v1/fundingRate?symbol={p}&startTime={t0}&limit=1000") or []
            rows = [(int(r["fundingTime"]), float(r["fundingRate"])) for r in x]
            if not rows:
                continue          # retry next run (current month is never marked complete)
            parts.append(pd.DataFrame(rows, columns=["t", "rate"]).assign(m="cur:" + m))
            continue
        z = _get(f"{VIS}/monthly/fundingRate/{p}/{p}-fundingRate-{m}.zip")
        if not z:
            parts.append(pd.DataFrame([[0, np.nan]], columns=["t", "rate"]).assign(m=m)); continue
        zz = zipfile.ZipFile(io.BytesIO(z))
        rr = _csv_rows(zz.open(zz.namelist()[0]).read())
        parts.append(pd.DataFrame([(int(r[0]), float(r[2])) for r in rr], columns=["t", "rate"]).assign(m=m))
    x = pd.concat(parts, ignore_index=True)
    x.to_csv(f, index=False)


_KC = {}


def kday(p, ds):
    k = (p, ds)
    if k in _KC:
        return _KC[k]
    f = os.path.join(K1, p, ds + ".npz")
    a = None
    if os.path.exists(f):
        z = np.load(f)
        if len(z["t"]):
            a = {c: z[c] for c in ("t", "o", "h", "l", "c", "v", "qv", "tbq")}
    if len(_KC) > 4000:
        _KC.clear()
    _KC[k] = a
    return a


def days_of(t0, t1):
    return [pd.Timestamp(d * DAY, unit="ms").strftime("%Y-%m-%d") for d in range(int(t0) // DAY, (int(t1) - 1) // DAY + 1)]


def k1m(p, t0, t1, need_all=False):
    """dict of arrays for 1m bars with open in [t0, t1); a missing day is skipped (need_all → None)."""
    parts = []
    for ds in days_of(t0, t1):
        a = kday(p, ds)
        if a is None:
            if need_all:
                return None
            continue
        parts.append(a)
    if not parts:
        return None
    out = {c: np.concatenate([x[c] for x in parts]) for c in parts[0]}
    m = (out["t"] >= t0) & (out["t"] < t1)
    out = {c: v[m] for c, v in out.items()}
    _, ix = np.unique(out["t"], return_index=True)
    return {c: v[ix] for c, v in out.items()}


def grid(a, t0, t1):
    """continuous 1m grid [t0, t1): missing bars filled flat at the previous close with 0 volume."""
    T = np.arange(t0 // MIN * MIN, t1, MIN, dtype="int64")
    idx = np.searchsorted(a["t"], T)
    ok = (idx < len(a["t"])) & (a["t"][np.minimum(idx, len(a["t"]) - 1)] == T)
    pos = np.where(ok, idx, -1)
    last = np.maximum.accumulate(np.where(ok, np.arange(len(T)), -1))
    src = np.where(last >= 0, pos[np.maximum(last, 0)], -1)
    c = np.where(src >= 0, a["c"][np.maximum(src, 0)], np.nan)
    g = {"t": T, "c": c, "present": ok}
    for k in ("o", "h", "l"):
        g[k] = np.where(ok, a[k][np.maximum(idx, 0).clip(max=len(a["t"]) - 1)], c)
    for k in ("v", "qv", "tbq"):
        g[k] = np.where(ok, a[k][np.maximum(idx, 0).clip(max=len(a["t"]) - 1)], 0.0).astype("float64")
    return g


def metrics(p, t0, t1):
    parts = []
    for ds in days_of(t0, t1):
        f = os.path.join(MET, p, ds + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            if len(z["t"]):
                parts.append({c: z[c] for c in ("t", "oi", "top_pos", "taker_ls")})
    if not parts:
        return None
    out = {c: np.concatenate([x[c] for x in parts]) for c in parts[0]}
    o = np.argsort(out["t"], kind="stable")
    return {c: v[o] for c, v in out.items()}


_FC = {}


def funding(p):
    if p in _FC:
        return _FC[p]
    f = os.path.join(FUN, p + ".csv")
    if not os.path.exists(f):
        _FC[p] = (np.array([], "int64"), np.array([])); return _FC[p]
    x = pd.read_csv(f).dropna(subset=["rate"])
    x = x[x.t > 0].drop_duplicates("t").sort_values("t")
    _FC[p] = (x.t.values.astype("int64"), x.rate.values.astype(float))
    return _FC[p]


def fund_cum(T, o, E, dirn, fs_t, fs_r, T_fill):
    """per-bar cumulative funding % to the position (short receives a positive rate), settlement s with T_fill < s assigned to the first
    bar with open ≥ s (mark-scaled by that bar's open ÷ E)."""
    F = np.zeros(len(T))
    if not len(fs_t):
        return F
    m = (fs_t > T_fill) & (fs_t <= T[-1])
    if not m.any():
        return F
    j = np.searchsorted(T, fs_t[m], "left")
    ok = j < len(T)
    add = np.zeros(len(T))
    np.add.at(add, j[ok], -dirn * fs_r[m][ok] * o[j[ok]] / E * 100.0)
    return np.cumsum(add)


def walk(g, i0, end_i, E, dirn, tp_px, sl_px, F, slip_stop, entry_cost, stop_mode="price", sl_pct=None):
    """today's exit from bar i0 (inclusive) to bar end_i (exclusive, the last bar inside the hold = end_i − 1).
    tp_px / sl_px: PRICE levels. Short: stop on high ≥ line (line moves with PAID funding if stop_mode == 'paid': line = E × (1 + (sl_pct +
    min(F, 0)) / 100)); maker TP on low ≤ tp × (1 − 0.05 %). Long mirrored. → dict(net, why, j, gross, fund)."""
    h, l, o, c = g["h"][i0:end_i], g["l"][i0:end_i], g["o"][i0:end_i], g["c"][i0:end_i]
    Fj = F[i0:end_i]
    n = len(h)
    if n <= 0:
        return None
    if dirn < 0:
        line = E * (1 + (sl_pct + np.minimum(Fj, 0.0)) / 100.0) if stop_mode == "paid" else np.full(n, sl_px)
        hs = h >= line
        tl = l <= tp_px * (1 - 0.0005)
    else:
        line = np.full(n, sl_px)
        hs = l <= line
        tl = h >= tp_px * (1 + 0.0005)
    ks = int(np.argmax(hs)) if hs.any() else n
    kt = int(np.argmax(tl)) if tl.any() else n
    if ks < n and ks <= kt:
        k = ks
        X = (max(line[k], o[k]) * (1 + slip_stop / 100.0)) if dirn < 0 else (min(line[k], o[k]) * (1 - slip_stop / 100.0))
        fee = TAKER * X / E; why = "STOP"
    elif kt < n:
        k = kt; X = tp_px; fee = MAKER * X / E; why = "TP"
    else:
        k = n - 1; X = c[k] * (1 + slip_stop / 100.0) if dirn < 0 else c[k] * (1 - slip_stop / 100.0)
        fee = TAKER * X / E; why = "TIME"
    gross = dirn * (X / E - 1) * 100.0
    fu = float(Fj[k])
    return dict(net=gross - entry_cost - fee + fu, why=why, j=i0 + k, gross=gross, fund=fu)


def md(df, fl=2):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{v:+.{fl}f}" if abs(v) < 10000 else f"{v:,.0f}"))
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)
