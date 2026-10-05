#!/usr/bin/env python3
"""⏳ Scout — pre-committed REVERT / ARM gate tracker (operator, 2026-10-04: "the hourly scout tracks every open pre-committed revert
gate automatically and flags the moment one fires").

READ-ONLY. Public Binance market data (1m / 5m klines over REST, aggTrades daily archives via scripts/backtest_fetch_ticks.py) + the
operator's exports in ~/Downloads (orders: scalpars_orders_paper_*.csv, decisions journal: scalpars_decisions_paper_*.csv) +
reports/MASTER_POOL_stacked.csv. Never talks to the bot, never changes config. Called by scripts/opportunity_scout.py every run
(own try/except there; every gate here has its own try/except too). State: reports/SCOUT_REVERT_GATES.json (atomic writes).

GATES (frozen definitions — quoted from CLAUDE_CURRENT_STATE.md / DECISION_LOG; never re-tuned here):
  CHOP_BURST (201) momentum LONG refused (BLOCK LONG_CHOP_BURST): first 6 refused signals re-priced → WR ≥ 50 % ∨ Σ > 0 → FIRES
                   (long_chop_burst_block_enabled false).
  FRENZY_TP3 (199) first 20 FRENZY_LONG + FRENZY_WIDE fills opened after the +3 deploy re-priced with fixed +4/−3 on ticks (bot accounting:
                   net levels, 0.09 % fees, fill at the crossing print, 12 h cap) → +4/−3 beats the actual average → FIRES (frenzy_tp_pct 4).
  FRENZY_STRONG (197) first 10 sized-up FRENZY_LONG fills (entry_frenzy_adx_delta > 0 ∧ entry_frenzy_di_spread > 0) average below the
                   other FRENZY_LONG fills of the same period, or below 0 → FIRES (frenzy_long_lev_mult_strong 0).
  FRENZY_GVOL (194) first 20 FRENZY + WIDE fills under the market-volume gate average < 0 → FIRES (frenzy_gvol_max 0).
  SURGE_LONG (202) option B at full size (operator override): the first 15 SURGE_LONG triggers that FILLED after the deploy (a closed
                   prefix; one trigger = the mean of its fills) mean ≤ 0 → FIRES (surge_long_lev_mult 0.05). Supersedes the 200 probe gate.
  BEARRUN (200)    windows (fills ≤ 180 min apart = one window) started after 2026-10-04 22:00 UTC: ≥ 5 windows, ≥ 3 positive ∧ Σ > 0 →
                   ARM bar met (bearrun_lev_mult 1.0) — a positive event, not a revert.
  LOADX (126)      first 30 (extended from 8 on 2026-10-04, operator) PAIR_RSI_MOMENTUM_LOADX-blocked LONG signals (journal FAILS lines whose COMPLETE fail set is LOADX alone,
                   rank ≤ 10 pairs excluded = mega-cap gate), WINDOW units (one 5-min journal bucket = one scan = one window, value =
                   mean) → WR ≥ 60 % ∨ net > 0 → long_rsi_momentum_adx_max 0.
  HEAT (116)       first 30 (extended from 6 on 2026-10-04, operator) LONG_HEAT_BLOCK fires re-priced → WR ≥ 60 % → legs back to 0.07 / 64 / 80 (second leg — the Jan–Jun
                   engine replay failing the expectancy bar — is manual).
  MEGACAP (110)    LONG_MEGACAP_BLOCK refusals re-priced → ≥ 60 % WR ∧ Σ > 0 on N ≥ 8 across ≥ 3 windows → long_megacap_rank_max 0.

REFUSED-SIGNAL PRICING (CHOP_BURST / LOADX / HEAT / MEGACAP): journal bucket t = the 5-min START of the refusal → entries at t+1 min
(primary) and t+5 min (sensitivity); one signal per pair-episode (same pair refused again ≤ 60 min later = the same signal). Price = the
live momentum-LONG exit REPLICA of scripts/ml_exit_optimize.py (evaluate/run_fill, BASE params = today's live stack incl. recovery hold;
reproduces live exits 104/104 on ticks), taker entry fee, ATR = 14-bar ATR % of the closed 5m bars before entry, recovery-hold RSI ruler =
BTC closed-5m RSI at entry. Path = aggTrades ticks where the daily archive exists (fetched on demand), 1m klines otherwise →
'provisional' until every day of the path is on ticks (accepted as final on 1m after 4 days without an archive). A gate only FIRES on
final prices; a provisional verdict is shown as such. The journal hides the exact refusal second / price → the two timings bracket it.

Usage:  venv/bin/python scripts/scout_revert_gates.py            # print the section (and update the state)
        venv/bin/python scripts/scout_revert_gates.py --selftest # synthetic checks of every gate's fire logic
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
REPORTS = os.path.join(ROOT, "reports")
CACHE = os.path.join(REPORTS, "backtest_cache")
STATE_JSON = os.path.join(REPORTS, "SCOUT_REVERT_GATES.json")
DL = os.path.expanduser("~/Downloads")
MIN, H, DAY = 60_000, 3_600_000, 86_400_000
EPISODE_MIN = 60                     # same pair refused again ≤ 60 min later = the same signal
FINAL_AFTER_MS = 4 * DAY             # no tick archive after 4 days → the 1m price is accepted as final
TICK_RETRY_MS = 3 * H                # an archive that is not published yet is retried at most every 3 h
FETCH_TIMEOUT_S = 240                # tick-archive fetch budget per run
PRICE_BUDGET_S = 240                 # pricing budget per run (the rest waits for the next run)
FRENZY_FEES, FRENZY_HOLD = 0.09, 12 * H
PROBE_START = "2026-10-04 22:00"     # DECISION_LOG 200: SURGE_LONG / BEARRUN probe windows count from here
SURGE_R72_MAX = 2.7
# (commit, fallback UTC push time) — deploy = push + 10 min
DEPLOYS = {"FRENZY_TP3": ("2e36c26", "2026-10-04 19:23:15"), "FRENZY_STRONG": ("181131e", "2026-10-04 14:06:09"),
           "FRENZY_GVOL": ("0d79904", "2026-10-03 22:14:13"),
           "SURGE_B": ("grep:(DECISION_LOG 202)", "2026-10-05 01:30:00")}   # ⚡ Oct-4 option B (found by its commit message)
SHIPS = {"HEAT": "2026-09-25", "LOADX": "2026-09-29", "MEGACAP": "2026-09-23"}
# Oct-4 operator: "keep collecting" → trackers extended to 30; (new N, frozen N, frozen verdict) — the frozen first-N verdict stays on record
EXT_N = {"LOADX": (30, 8, "FIRED (fragile at t+5m)"), "HEAT": (30, 6, "FIRED (6/6 won)")}   # ship dates (journal coverage notes)


def log(msg):
    print(f"[revert-gates] {msg}", file=sys.stderr, flush=True)


def _ms(ts):
    """UTC epoch ms of a string / datetime / Timestamp (naive = UTC)."""
    t = pd.Timestamp(ts)
    return int((t.tz_localize("UTC") if t.tzinfo is None else t).value // 1_000_000)


def _ms_series(s):
    t = pd.to_datetime(pd.Series(s).astype(str).str[:23].str.replace("T", " ", regex=False), errors="coerce", format="mixed")
    return ((t - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("float")


def _fmt_t(ms, full=False):
    if ms is None or (isinstance(ms, float) and not np.isfinite(ms)):
        return "–"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M" if full else "%m-%d %H:%M")


def _f(v, fmt="+.2f"):
    try:
        v = float(v)
        return format(v, fmt) if np.isfinite(v) else "–"
    except (TypeError, ValueError):
        return "–"


def atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_state():
    try:
        with open(STATE_JSON) as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:                                # a corrupt file must be visible: the frozen first-N sets live in it
        log(f"state unreadable ({e}) — starting fresh; old file kept as .bad")
        try:
            os.replace(STATE_JSON, STATE_JSON + ".bad")
        except OSError:
            pass
        return {}


def deploy_ms(name):
    """push time of the gate's commit (git) + 10 min; the pinned fallback when git is unavailable."""
    h, fb = DEPLOYS[name]
    try:
        args = ["--grep=" + h[5:], "--fixed-strings", "--reverse", "--since=2026-10-04"] if h.startswith("grep:") else [h]   # grep: the FIRST commit naming it
        out = subprocess.run(["git", "-C", ROOT, "log", "--format=%ct"] + args, capture_output=True, text=True, timeout=10)
        out.stdout = (out.stdout.strip().splitlines() or [""])[0]
        ct = int(out.stdout.strip())
        return ct * 1000 + 10 * MIN
    except Exception:
        return _ms(fb) + 10 * MIN


def cfg_value(key):
    """current value of a config key in trading_config.json (top level or any nested dict), else None."""
    try:
        cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))
    except Exception:
        return None
    stack = [cfg]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            if key in d:
                return d[key]
            stack.extend(v for v in d.values() if isinstance(v, dict))
    return None


# ═══════════════════════════════ pure decision rules (self-tested) ═══════════════════════════════
def _wr(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return (100.0 * sum(1 for x in v if x > 0) / len(v)) if v else float("nan")


def decide_first_n(vals, n, wr_min, need_sum_pos=False, or_sum_pos=True):
    """first-N rule on re-priced refused signals (vals in order). Returns (state, wr, sum) with state 'collecting' | 'fired' | 'holds'.
    or_sum_pos: fires when WR ≥ wr_min OR Σ > 0 · need_sum_pos: fires when WR ≥ wr_min AND Σ > 0 · neither: WR alone."""
    v = list(vals)[:n]
    if len(v) < n:
        return "collecting", _wr(v), float(np.sum(v)) if v else 0.0
    wr, s = _wr(v), float(np.sum(v))
    if need_sum_pos:
        fired = wr >= wr_min and s > 0
    elif or_sum_pos:
        fired = wr >= wr_min or s > 0
    else:
        fired = wr >= wr_min
    return ("fired" if fired else "holds"), wr, s


def decide_megacap(vals, windows, n_min=8, w_min=3, wr_min=60.0):
    """cumulative re-admit bar: ≥ 60 % WR ∧ Σ > 0 on N ≥ 8 across ≥ 3 windows (fires; otherwise collecting — no frozen N)."""
    v = list(vals)
    wr, s = _wr(v), float(np.sum(v)) if v else 0.0
    if len(v) < n_min or len(set(windows)) < w_min:
        return "collecting", wr, s
    return ("fired" if (wr >= wr_min and s > 0) else "collecting"), wr, s


def decide_tp(actual, alt, n=20):
    """FRENZY TP: on the first n fills, the alternative exit's average beats the actual average → fired."""
    a, b = list(actual)[:n], list(alt)[:n]
    if len(a) < n or len(b) < n:
        return "collecting", float(np.mean(a)) if a else float("nan"), float(np.mean(b)) if b else float("nan")
    ma, mb = float(np.mean(a)), float(np.mean(b))
    return ("fired" if mb > ma else "holds"), ma, mb


def decide_strong(sized, normal, n=10):
    """FRENZY strong leverage: first n sized-up fills average below the normal fills' average, or below 0 → fired."""
    s = list(sized)[:n]
    ms = float(np.mean(s)) if s else float("nan")
    mn = float(np.mean(normal)) if len(normal) else float("nan")
    if len(s) < n:
        return "collecting", ms, mn
    fired = ms < 0 or (np.isfinite(mn) and ms < mn)
    return ("fired" if fired else "holds"), ms, mn


def decide_mean_neg(vals, n=20):
    v = list(vals)[:n]
    m = float(np.mean(v)) if v else float("nan")
    if len(v) < n:
        return "collecting", m
    return ("fired" if m < 0 else "holds"), m


def boot_ci(x, n=5000, seed=7):
    """95 % bootstrap CI of the mean (window units) — same ruler as scripts/surge_bearrun_review.py."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan")
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def decide_surge(win, min_windows=8, r72_max=SURGE_R72_MAX):   # RETIRED Oct-5 (superseded by decide_surge_b, DECISION_LOG 202) — kept for the record
    """win = [(btc_r72, window_mean)]. Frozen rule (DECISION_LOG 200) after ≥ 8 windows with fills:
    low group (r72 ≤ +2.7) mean > 0 with CI > 0 ∧ rest ≤ 0 → 'arm_group' · low group ≤ 0 → 'fired' (LONG off) · else 'open'."""
    lo = [m for r, m in win if r is not None and np.isfinite(r) and r <= r72_max]
    hi = [m for r, m in win if r is not None and np.isfinite(r) and r > r72_max]
    info = dict(n=len(win), n_lo=len(lo), n_hi=len(hi), m_lo=float(np.mean(lo)) if lo else float("nan"),
                m_hi=float(np.mean(hi)) if hi else float("nan"), ci_lo=boot_ci(lo))
    if len(win) < min_windows:
        return "collecting", info
    if any(r is None or not np.isfinite(r) for r, _ in win):
        return "collecting", info                    # an unreadable BTC 3-day return blocks the split — never guess a group
    if not lo:
        return "open", info
    if info["m_lo"] <= 0:
        return "fired", info
    if info["m_lo"] > 0 and np.isfinite(info["ci_lo"][0]) and info["ci_lo"][0] > 0 and (not hi or info["m_hi"] <= 0):
        return "arm_group", info
    return "open", info


def decide_surge_b(trigger_means, n=15):
    """⚡ Oct-4 option B (DECISION_LOG 202) revert gate, pre-committed: the first 15 SURGE_LONG triggers that FILLED since the B deploy
    (one trigger = the mean of its fills' pnl %) — mean ≤ 0 → 'fired' (leverage back to the 0.05 probe); mean > 0 → 'holds'."""
    v = list(trigger_means)[:n]
    if len(v) < n:
        return "collecting"
    return "fired" if float(np.mean(v)) <= 0 else "holds"


def decide_bearrun(win_means, total_sum, min_windows=5, min_pos=3):
    """ARM bar (DECISION_LOG 200): ≥ 5 windows, ≥ 3 positive ∧ Σ > 0 → 'armbar'."""
    if len(win_means) < min_windows:
        return "collecting"
    return "armbar" if (sum(1 for m in win_means if m > 0) >= min_pos and total_sum > 0) else "open"


def window_chain(ts_ms, gap_ms):
    """window index per (sorted) timestamp: a new window when the gap to the previous one exceeds gap_ms."""
    out, cur, prev = [], -1, None
    for t in ts_ms:
        if prev is None or t - prev > gap_ms:
            cur += 1
        out.append(cur)
        prev = t
    return out


def episodes(rows, gap_min=EPISODE_MIN):
    """first journal line of each pair-episode (same pair refused again ≤ gap_min later = the same signal). rows: DataFrame(ms, pair)."""
    rows = rows.sort_values(["ms", "pair"])
    keep, last = [], {}
    for i, r in zip(rows.index, rows.itertuples()):
        if r.pair not in last or r.ms - last[r.pair] > gap_min * MIN:
            keep.append(i)
        last[r.pair] = r.ms
    return rows.loc[keep]


# ═══════════════════════════════ data: exports ═══════════════════════════════
ORDER_COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "entry_price", "exit_price", "pnl_percentage",
              "leverage", "entry_frenzy_adx_delta", "entry_frenzy_di_spread", "entry_surge_trigger_at", "close_reason", "entry_pair_rank")


def _export_ms(path):
    try:
        b = os.path.basename(path).rsplit("_paper_", 1)[1][:19]
        return _ms(datetime.strptime(b, "%Y-%m-%d_%H-%M-%S"))
    except Exception:
        return int(os.path.getmtime(path) * 1000)


def load_orders():
    """bot fills: every orders export (dedupe opened_at·pair·direction, newest export wins) + the master pool (lowest priority);
    MANUAL excluded. Returns (df, newest_export_ms, n_files)."""
    fs = sorted(glob.glob(os.path.join(DL, "scalpars_orders_paper_*.csv")), key=_export_ms)
    fr = []
    pool = os.path.join(REPORTS, "MASTER_POOL_stacked.csv")
    if os.path.exists(pool):
        try:
            fr.append(pd.read_csv(pool, low_memory=False, usecols=lambda c: c in ORDER_COLS).assign(_rank=-1))
        except Exception as e:
            log(f"master pool unreadable: {e}")
    newest = None
    for i, f in enumerate(fs):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in ORDER_COLS).assign(_rank=i))
            newest = max(newest or 0, _export_ms(f))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(ORDER_COLS) + ["o_ms"]), None, 0
    A = pd.concat(fr, ignore_index=True)
    for c in ORDER_COLS:
        if c not in A:
            A[c] = np.nan
    A = A.dropna(subset=["opened_at", "pair"])
    A["_k"] = A.opened_at.astype(str).str.replace(" ", "T", regex=False).str[:19]   # pool and exports may differ in the separator
    A = A.sort_values("_rank").drop_duplicates(["_k", "pair", "direction"], keep="last")
    A = A[A.entry_strategy.astype(str) != "MANUAL"].copy()
    A["o_ms"] = _ms_series(A.opened_at).values
    A = A[np.isfinite(A.o_ms)].sort_values("o_ms").reset_index(drop=True)
    A["o_ms"] = A.o_ms.astype("int64")
    A["pnl_percentage"] = pd.to_numeric(A.pnl_percentage, errors="coerce")
    return A, newest, len(fs)


def load_journal():
    """decision-journal lines needed by the refused-signal gates + coverage. Returns (J, (t_min, t_max, n_files, gaps_h))."""
    fs = sorted(glob.glob(os.path.join(DL, "scalpars_decisions_paper_*.csv")))
    fr, beats = [], []
    for f in fs:
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("t", "e", "pair", "dir", "gate", "src"))
        except Exception:
            continue
        if not {"t", "e"} <= set(d.columns):
            continue
        for c in ("pair", "dir", "gate", "src"):
            if c not in d:
                d[c] = None
        beats.append(d.loc[d.e == "SCAN", "t"])
        g = d.gate.astype(str)
        keep = (d.dir.astype(str) == "LONG") & (((d.e == "BLOCK") & g.isin(["LONG_CHOP_BURST", "LONG_HEAT_BLOCK", "LONG_MEGACAP_BLOCK"]))
                                               | ((d.e == "FAILS") & g.str.contains("PAIR_RSI_MOMENTUM_LOADX", regex=False)))
        fr.append(d[keep])
    if not fr:
        return pd.DataFrame(columns=["t", "e", "pair", "dir", "gate", "src", "ms"]), None
    J = pd.concat(fr, ignore_index=True).drop_duplicates(["t", "e", "pair", "dir", "gate"])
    J["ms"] = _ms_series(J.t).values
    J = J[np.isfinite(J.ms)].copy()
    J["ms"] = J.ms.astype("int64")
    b = pd.concat(beats) if beats else pd.Series(dtype=str)
    bm = np.sort(_ms_series(b.drop_duplicates()).dropna().values.astype("int64")) if len(b) else np.array([], dtype="int64")
    if not len(bm):
        return J, None
    gaps = np.diff(bm)
    return J, (int(bm[0]), int(bm[-1]) + 5 * MIN, len(fs), int((gaps > H).sum()))


def rank_map(orders):
    """latest stamped entry_pair_rank per pair (any sleeve) — the mega-cap exclusion for the LOADX cohort."""
    r = orders.dropna(subset=["entry_pair_rank"]).sort_values("o_ms") if "entry_pair_rank" in orders else orders.iloc[0:0]
    return r.groupby("pair").entry_pair_rank.last().astype(float).to_dict() if len(r) else {}


# ═══════════════════════════════ data: market ═══════════════════════════════
_K1_MEM = {}


def _get_json(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def klines(pair, tf, t0, t1):
    """CLOSED klines [t0, t1) from Binance USDⓈ-M public REST → DataFrame(open_time, o, h, l, c). In-memory cache per run."""
    key = (pair, tf, t0, t1)
    if key in _K1_MEM:
        return _K1_MEM[key]
    step = {"1m": MIN, "5m": 5 * MIN}[tf]
    now = int(time.time() * 1000)
    rows, since = [], t0
    end = min(t1, now)
    while since < end:
        q = urllib.parse.urlencode(dict(symbol=pair, interval=tf, startTime=since, endTime=end - 1, limit=1500))
        r = _get_json(f"https://fapi.binance.com/fapi/v1/klines?{q}")
        if not r:
            break
        rows += [x for x in r if int(x[0]) + step <= now]        # forming bar dropped
        nxt = int(r[-1][0]) + step
        if nxt <= since or len(r) < 1500:
            break
        since = nxt
        time.sleep(0.05)
    d = pd.DataFrame([[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in rows],
                     columns=["open_time", "o", "h", "l", "c"]).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    _K1_MEM[key] = d
    return d


def btc5m_array(t_from):
    """BTC 5m [open_time, o, h, l, c] — reports/backtest_cache/btc_5m.csv merged with REST bars up to now (closed only)."""
    fr = []
    fp = os.path.join(CACHE, "btc_5m.csv")
    if os.path.exists(fp):
        try:
            fr.append(pd.read_csv(fp, usecols=["open_time", "o", "h", "l", "c"]))   # whole cache (~90k rows): any stored signal's RSI ruler
        except Exception:
            pass
    last = int(fr[0].open_time.max()) if fr and len(fr[0]) else t_from - 4 * DAY
    now = int(time.time() * 1000)
    try:
        fr.append(klines("BTCUSDT", "5m", max(last - 2 * H, t_from - 4 * DAY), now))
    except Exception as e:
        log(f"BTC 5m fetch failed: {e}")
    if not fr:
        return None
    b = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time")
    return b[["open_time", "o", "h", "l", "c"]].values.astype(float)


def btc_r72_at(btc, trig_close_ms):
    """BTC 3-day return at the SURGE trigger bar (entry_surge_trigger_at = the bar's CLOSE): close(bar) / close(864 bars earlier) − 1."""
    if btc is None or trig_close_ms is None:
        return None
    ot = btc[:, 0]
    i = np.searchsorted(ot, trig_close_ms - 5 * MIN)
    j = np.searchsorted(ot, trig_close_ms - 5 * MIN - 864 * 5 * MIN)
    if i >= len(ot) or ot[i] != trig_close_ms - 5 * MIN or j >= len(ot) or ot[j] != trig_close_ms - 5 * MIN - 864 * 5 * MIN:
        return None
    return float((btc[i, 4] / btc[j, 4] - 1) * 100)


def _tick_path(pair, day_ms):
    ds = time.strftime("%Y-%m-%d", time.gmtime(day_ms / 1000))
    return [os.path.join(CACHE, sub, pair, f"{ds}.npz") for sub in ("ticks_q", "ticks")]


def has_ticks(pair, day_ms):
    return any(os.path.exists(p) for p in _tick_path(pair, day_ms))


def days_of(t0, t1):
    return list(range((t0 // DAY) * DAY, t1, DAY))


def ensure_ticks(pairdays, st, now_ms):
    """fetch missing aggTrades daily archives (completed UTC days only) through scripts/backtest_fetch_ticks.py; a missing archive is
    retried at most every 3 h (state 'tick_tries')."""
    tries = st.setdefault("tick_tries", {})
    want = []
    for pair, dms in sorted(set(pairdays)):
        if dms + DAY > now_ms or has_ticks(pair, dms):
            continue
        k = f"{pair}|{dms}"
        if now_ms - int(tries.get(k, 0)) < TICK_RETRY_MS:
            continue
        tries[k] = now_ms
        want.append((pair, time.strftime("%Y-%m-%d", time.gmtime(dms / 1000))))
    for k in [k for k, v in tries.items() if now_ms - int(v) > 10 * DAY]:
        tries.pop(k, None)
    if not want:
        return 0
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=REPORTS)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("pair,date\n" + "".join(f"{p},{d}\n" for p, d in want))
        subprocess.run([sys.executable, os.path.join(SCRIPTS, "backtest_fetch_ticks.py"), tmp], cwd=ROOT,
                       capture_output=True, text=True, timeout=FETCH_TIMEOUT_S)
    except Exception as e:
        log(f"tick fetch: {e}")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if _M is not None:
        _M._DAYS.clear()                                  # a day cached as 'no archive' must be re-read
    return sum(1 for p, d in want if has_ticks(p, _ms(d)))


# ═══════════════════════════════ pricing ═══════════════════════════════
_M = None


def _replica(btc):
    """scripts/ml_exit_optimize.py (the live momentum-long exit replica) with this run's BTC 5m injected (recovery-hold ruler)."""
    global _M
    if _M is None:
        for p in (ROOT, SCRIPTS):
            if p not in sys.path:
                sys.path.insert(0, p)
        import ml_exit_optimize as M
        _M = M
    if btc is not None:
        _M._BTC = btc
    return _M


def _inject_k1(M, pair, k1):
    M._K1[pair] = k1
    M._K5.pop(pair, None)


def _atr_pct(k1, entry_ms):
    """14-bar ATR % (ta / Wilder, as services.indicators) on the CLOSED 5m bars before entry, built from the 1m klines."""
    from ta.volatility import AverageTrueRange
    d = k1[k1.open_time < entry_ms].copy()
    d["b"] = (d.open_time // (5 * MIN)) * (5 * MIN)
    g = d.groupby("b").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), n=("o", "size"))
    g = g[(g.index + 5 * MIN <= entry_ms) & (g.n == 5)]
    if len(g) < 30:
        return None
    a = AverageTrueRange(high=g.h, low=g.l, close=g.c, window=14).average_true_range()
    v = float(a.iloc[-1] / g.c.iloc[-1] * 100)
    return v if np.isfinite(v) and v > 0 else None


def price_ml(pair, entry_ms, btc, kwin=None):
    """live momentum-LONG exit replica from entry_ms → dict(pct, how, src) or dict(pending=reason). kwin = the 1m kline span to fetch
    (one span shared by both entry timings of a signal); it must cover ≥ 1 day before and ≥ 7 h after the entry."""
    M = _replica(btc)
    a, b = kwin or (entry_ms - DAY, entry_ms + 8 * H)
    try:
        k1 = klines(pair, "1m", a, b)
    except Exception as e:
        return dict(pending=f"klines: {str(e)[:60]}")
    if k1 is None or not len(k1) or k1.open_time.max() < entry_ms:
        return dict(pending="no klines yet")
    _inject_k1(M, pair, k1)
    bp = M.build_path(pair, entry_ms, 2 * H)
    if bp is None or not len(bp[0]):
        return dict(pending="no path")
    atr = _atr_pct(k1, entry_ms)
    rsi = M.btc_rsi_at(entry_ms)
    fx = dict(key=f"{pair}|{entry_ms}", pair=pair, o_ms=int(entry_ms), E=float(bp[1][0]), fee_in=M.TAKER, atr=atr, rsi_entry=rsi)
    k, why, pct, tex, src, pk = M.run_fill(fx, {"BASE": M.BASE})["BASE"]
    if why in ("END_OF_DATA", "NO_DATA") or str(why).startswith("ERR") or not np.isfinite(pct):
        return dict(pending=("still running" if why == "END_OF_DATA" else str(why)[:60]))
    return dict(pct=round(float(pct), 3), how=f"{why} {int((tex - entry_ms) / MIN)}m", src=src)


def price_fixed(pair, entry_ms, E, levels):
    """FRENZY bot-exact fixed exits on one path: levels = [(tp, sl)] → {f'{tp}/{sl}': (pct, why)} + src, or dict(pending=…).
    Net P&L at every print = (p / E − 1) · 100 − 0.09; first print with net ≤ −sl or ≥ +tp closes there; else the last print ≤ 12 h."""
    M = _replica(None)
    try:
        k1 = klines(pair, "1m", entry_ms - H, entry_ms + FRENZY_HOLD + 5 * MIN)
    except Exception as e:
        return dict(pending=f"klines: {str(e)[:60]}")
    if k1 is None or not len(k1):
        return dict(pending="no klines yet")
    _inject_k1(M, pair, k1)
    bp = M.build_path(pair, entry_ms, FRENZY_HOLD)
    if bp is None or not len(bp[0]):
        return dict(pending="no path")
    t, p, src = bp
    net = (p / float(E) - 1) * 100 - FRENZY_FEES
    out = dict(src=src)
    complete = t[-1] >= entry_ms + FRENZY_HOLD - 2 * MIN
    for tp, sl in levels:
        hit = np.flatnonzero((net <= -sl) | (net >= tp))
        if len(hit):
            i = int(hit[0])
            v = float(net[i])
            if src != "tick":   # a 1m o→l/h→c path jumps to the bar extreme — the provisional price fills AT the level instead
                v = float(tp) if v >= tp else -float(sl)
            out[f"{tp:g}/{sl:g}"] = (round(v, 3), "TP" if v >= tp else "SL")
        elif complete:
            out[f"{tp:g}/{sl:g}"] = (round(float(net[-1]), 3), "12h cap")
        else:
            return dict(pending="still running")
    return out


def _is_final(src_list, t_ms, now_ms):
    return all(s == "tick" for s in src_list) or now_ms - t_ms > FINAL_AFTER_MS


class Budget:
    def __init__(self, s):
        self.end = time.time() + s

    def ok(self):
        return time.time() < self.end


# ═══════════════════════════════ gates ═══════════════════════════════
def _signals_for(J, gate_name, ranks=None, store=None):
    """refused LONG signals of one gate → DataFrame(ms, pair, key) — pair-episodes, in time order. Signals already stored in the state
    are kept even when their export has left ~/Downloads (a frozen first-N set never shrinks)."""
    old = pd.DataFrame([dict(ms=int(v["t"]), pair=v["pair"]) for v in (store or {}).values()], columns=["ms", "pair"])
    if J is None or not len(J):
        ep = old
    elif gate_name == "LOADX":
        f = J[J.e == "FAILS"].copy()
        f = f[f.gate.astype(str).str.replace("MACRO:", "", regex=False) == "PAIR_RSI_MOMENTUM_LOADX"]
        if "src" in f:
            f = f[f.src.astype(str).isin(["MOMENTUM", "nan", "None"])]
        ep = episodes(pd.concat([old, f[["ms", "pair"]]], ignore_index=True))
    else:
        g = {"CHOP_BURST": "LONG_CHOP_BURST", "HEAT": "LONG_HEAT_BLOCK", "MEGACAP": "LONG_MEGACAP_BLOCK"}[gate_name]
        ep = episodes(pd.concat([old, J[(J.e == "BLOCK") & (J.gate.astype(str) == g)][["ms", "pair"]]], ignore_index=True))
    if ranks and gate_name in ("LOADX", "HEAT"):    # these gates run BEFORE the mega-cap block: a rank ≤ 10 pair is refused there anyway
        ep = ep[~ep.pair.map(lambda p: ranks.get(p, 999) <= 10)]
    ep = ep.astype({"ms": "int64"}).sort_values(["ms", "pair"]).reset_index(drop=True)
    ep["key"] = ep.pair.astype(str) + "|" + ep.ms.astype(str)
    return ep


def _price_signals(sig, store, n_needed, btc, budget, now_ms, need_days):
    """price (or reuse stored prices for) the first n_needed signals (None = all). Both entry timings. Returns the list of items."""
    live = set(sig.key)
    for k in [k for k in store if k not in live]:      # e.g. a pair whose rank now marks it mega-cap — no longer in the cohort
        store.pop(k, None)
    out = []
    todo = sig if n_needed is None else sig.head(n_needed)
    for r in todo.itertuples():
        it = store.get(r.key) or dict(pair=r.pair, t=int(r.ms))
        if not it.get("final"):
            if need_days is not None:                           # phase 1: collect the tick days this signal needs
                for off in (1, 5):
                    need_days.update((r.pair, d) for d in days_of(int(r.ms) + off * MIN, int(r.ms) + off * MIN + 7 * H))
            elif budget.ok():
                res = {off: price_ml(r.pair, int(r.ms) + off * MIN, btc, (int(r.ms) - DAY, int(r.ms) + 8 * H)) for off in (1, 5)}
                for off, x in res.items():
                    if "pending" in x:
                        it[f"p{off}"] = x["pending"]
                        it.pop(f"sim{off}", None)
                    else:
                        it.update({f"sim{off}": x["pct"], f"how{off}": x["how"], f"src{off}": x["src"]})
                        it.pop(f"p{off}", None)
                srcs = [it.get("src1"), it.get("src5")]
                # final = both timings on ticks, or the primary priced and 4 days passed (no archive / a timing that never prices)
                it["final"] = bool((all(s is not None for s in srcs) and all(s == "tick" for s in srcs))
                                   or (srcs[0] is not None and now_ms - int(r.ms) > FINAL_AFTER_MS))
            store[r.key] = it
        out.append(it)
    return out


def _sig_progress(items, n, unit="signals"):
    pr = [x for x in items if x.get("sim1") is not None]
    fin = [x for x in pr if x.get("final")]
    s1 = [x["sim1"] for x in pr]
    s5 = [x["sim5"] for x in pr if x.get("sim5") is not None]
    txt = (f"{len(pr)}/{n} {unit} re-priced" + (f" ({len(pr) - len(fin)} provisional)" if len(pr) > len(fin) else "")
           + (f" · {sum(1 for v in s1 if v > 0)} won · Σ {sum(s1):+.2f} %" if s1 else "")
           + (f" (t+5m: {sum(1 for v in s5 if v > 0)} won · Σ {sum(s5):+.2f})" if s5 else ""))
    return txt, pr, fin


def _priced_txt(items):
    pr = [x for x in items if x.get("sim1") is not None]
    return f"priced {sum(1 for x in pr if x.get('final'))} final / {sum(1 for x in pr if not x.get('final'))} on 1m" if pr else ""


def _sig_detail(items, k=8):
    return " · ".join(f"{_fmt_t(x['t'])} {str(x['pair']).replace('USDT', '')} "
                      + (f"{x['sim1']:+.2f}/{_f(x.get('sim5'))}" if x.get("sim1") is not None else f"pending ({x.get('p1', '?')})")
                      + ("" if x.get("final") or x.get("sim1") is None else "ᵖ")
                      for x in items[:k])


def gate_first_n_signals(code, J, st, n, wr_min, mode, btc, budget, now_ms, need_days, ranks=None, windows=False):
    """CHOP_BURST / HEAT (signal units) and LOADX (window units: one 5-min bucket = one window, value = mean of its signals)."""
    G = st.setdefault("gates", {}).setdefault(code, {})
    store = G.setdefault("items", {})
    sig = _signals_for(J, code, ranks, store)
    if windows:   # the first n windows = the signals of the first n distinct buckets
        bk = sorted(sig.ms.unique())[:n]
        sig = sig[sig.ms.isin(bk)]
        n_sig = None
    else:
        n_sig = n
    items = _price_signals(sig, store, n_sig, btc, budget, now_ms, need_days)
    if need_days is not None:
        return None
    if windows:
        byb = {}
        for x in items:
            byb.setdefault(x["t"], []).append(x)
        wins = [byb[b] for b in sorted(byb)]
        ready = [w for w in wins if all(x.get("sim1") is not None for x in w)]
        v1 = [float(np.mean([x["sim1"] for x in w])) for w in ready]
        v5 = [float(np.mean([x["sim5"] for x in w])) for w in ready if all(x.get("sim5") is not None for x in w)]
        all_final = len(ready) == len(wins) and all(x.get("final") for w in wins for x in w)
        npr = len(ready)
        prog = (f"{npr}/{n} windows re-priced ({len(items)} signals)" + ("" if all_final or not npr else " (provisional)")
                + (f" · {sum(1 for v in v1 if v > 0)} won · Σ {sum(v1):+.2f} %" if v1 else "")
                + (f" (t+5m: {sum(1 for v in v5 if v > 0)} won · Σ {sum(v5):+.2f})" if v5 else ""))
    else:
        prog, pr, fin = _sig_progress(items, n)
        v1 = [x["sim1"] for x in pr]
        v5 = [x["sim5"] for x in pr if x.get("sim5") is not None]
        all_final = len(fin) == len(pr)
    kw = dict(or_sum_pos=(mode == "or"), need_sum_pos=(mode == "and"))
    state, wr, s = decide_first_n(v1, n, wr_min, **kw)
    state5 = decide_first_n(v5, n, wr_min, **kw)[0] if len(v5) >= n else None
    G.update(progress=prog, detail=_sig_detail(items, len(items) if windows else max(8, n)), priced=_priced_txt(items))
    if state != "collecting" and not all_final:
        G["provisional"] = state
        state = "collecting"
    else:
        G.pop("provisional", None)
    G["fragile"] = bool(state in ("fired", "holds") and state5 is not None and state5 != state)
    return state


def gate_megacap(J, st, btc, budget, now_ms, need_days):
    G = st.setdefault("gates", {}).setdefault("MEGACAP", {})
    store = G.setdefault("items", {})
    sig = _signals_for(J, "MEGACAP", None, store)
    items = _price_signals(sig, store, None, btc, budget, now_ms, need_days)
    if need_days is not None:
        return None
    prog, pr, fin = _sig_progress(items, 8)
    fin_items = [x for x in pr if x.get("final")]
    state, wr, s = decide_megacap([x["sim1"] for x in fin_items], [x["t"] for x in fin_items])
    G.update(progress=prog + f" · {len({x['t'] for x in pr})} windows", detail=_sig_detail(items), priced=_priced_txt(items))
    return state


def gate_frenzy_tp(orders, st, budget, now_ms, need_days, n=20):
    G = st.setdefault("gates", {}).setdefault("FRENZY_TP3", {})
    store = G.setdefault("items", {})
    t0 = deploy_ms("FRENZY_TP3")
    f = orders[orders.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (orders.o_ms >= t0)].head(n)
    items = []
    for r in f.itertuples():
        key = f"{str(r.opened_at)[:19]}|{r.pair}"
        it = store.get(key) or dict(pair=r.pair, t=int(r.o_ms), sleeve=r.entry_strategy)
        closed = str(r.status) == "CLOSED" and np.isfinite(r.pnl_percentage)
        it["actual"] = float(r.pnl_percentage) if closed else None
        if not it.get("final") and np.isfinite(pd.to_numeric(r.entry_price, errors="coerce")):
            if need_days is not None:
                need_days.update((r.pair, d) for d in days_of(int(r.o_ms), int(r.o_ms) + FRENZY_HOLD))
            elif budget.ok():
                x = price_fixed(r.pair, int(r.o_ms), float(r.entry_price), [(4, 3), (3, 3)])
                if "pending" in x:
                    it["p"] = x["pending"]
                else:
                    it.update(alt4=x["4/3"][0], alt4_why=x["4/3"][1], rep3=x["3/3"][0], src=x["src"])
                    it.pop("p", None)
                    it["final"] = _is_final([x["src"]], int(r.o_ms), now_ms)
        store[key] = it
        items.append(it)
    if need_days is not None:
        return None
    done = [x for x in items if x.get("actual") is not None and x.get("alt4") is not None]
    fin = [x for x in done if x.get("final")]
    state, ma, mb = decide_tp([x["actual"] for x in fin], [x["alt4"] for x in fin], n)
    pa = [x["actual"] for x in done]
    pb = [x["alt4"] for x in done]
    pr3 = [x["rep3"] for x in done if x.get("rep3") is not None]
    G["progress"] = (f"{len(done)}/{n} fills re-priced" + (f" ({len(done) - len(fin)} provisional)" if len(done) > len(fin) else "")
                     + (f" · actual avg {np.mean(pa):+.2f} % vs +4/−3 {np.mean(pb):+.2f} %" if done else "")
                     + (f" (tick +3/−3 replica {np.mean(pr3):+.2f} — fidelity check)" if pr3 else "")
                     + (f" · {len(items) - len(done)} open/pending" if len(items) > len(done) else ""))
    G["detail"] = " · ".join(f"{_fmt_t(x['t'])} {x['pair'].replace('USDT', '')} {_f(x.get('actual'))}→{_f(x.get('alt4'))}"
                             + ("" if x.get("final") or x.get("alt4") is None else "ᵖ") for x in items[:10])
    if state == "collecting" and len(done) >= n and len(fin) < n:
        G["provisional"] = decide_tp(pa, pb, n)[0]
    else:
        G.pop("provisional", None)
    return state, t0


def _closed_prefix(f):
    """pnl % of the leading CLOSED fills (in open order) — a first-N set waits for an earlier fill that is still open."""
    out = []
    for r in f.itertuples():
        if str(r.status) != "CLOSED" or not np.isfinite(r.pnl_percentage):
            break
        out.append(float(r.pnl_percentage))
    return out


def gate_frenzy_strong(orders, st, n=10):
    G = st.setdefault("gates", {}).setdefault("FRENZY_STRONG", {})
    t0 = deploy_ms("FRENZY_STRONG")
    f = orders[(orders.entry_strategy.astype(str) == "FRENZY_LONG") & (orders.o_ms >= t0)].copy()
    f["sized"] = (pd.to_numeric(f.entry_frenzy_adx_delta, errors="coerce") > 0) & (pd.to_numeric(f.entry_frenzy_di_spread, errors="coerce") > 0)
    sz = f[f.sized].head(n)
    vals = _closed_prefix(sz)
    nm = f[~f.sized & (f.status.astype(str) == "CLOSED") & f.pnl_percentage.notna()]
    if len(sz) == n:                                                  # "the same period" = deploy → the 10th sized-up fill
        nm = nm[nm.o_ms <= int(sz.o_ms.iloc[-1])]
    state, ms, mn = decide_strong(vals, nm.pnl_percentage.tolist(), n)
    lev = pd.to_numeric(sz.leverage, errors="coerce")
    G["progress"] = (f"{len(vals)}/{n} sized-up fills closed" + (f" · avg {ms:+.2f} %" if vals else "")
                     + f" vs normal {len(nm)} · " + (f"avg {mn:+.2f} %" if len(nm) else "none yet")
                     + (f" · sized-up leverage seen {', '.join(sorted({f'{v:g}×' for v in lev.dropna()}))}" if lev.notna().any() else ""))
    G["detail"] = " · ".join(f"{_fmt_t(r.o_ms)} {r.pair.replace('USDT', '')} {'⬆' if r.sized else ''}{_f(r.pnl_percentage)}"
                             for r in f.head(12).itertuples())
    return state, t0


def gate_frenzy_gvol(orders, st, n=20):
    G = st.setdefault("gates", {}).setdefault("FRENZY_GVOL", {})
    t0 = deploy_ms("FRENZY_GVOL")
    f = orders[orders.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (orders.o_ms >= t0)].head(n)
    vals = _closed_prefix(f)
    state, m = decide_mean_neg(vals, n)
    G["progress"] = (f"{len(vals)}/{n} fills closed" + (f" · avg {m:+.2f} % · {sum(1 for v in vals if v > 0)} won" if vals else "")
                     + (f" · {len(f) - len(vals)} later/open" if len(f) > len(vals) else ""))
    G["detail"] = " · ".join(f"{_fmt_t(r.o_ms)} {r.pair.replace('USDT', '')} {str(r.entry_strategy).replace('FRENZY_', '')[:4]} {_f(r.pnl_percentage)}"
                             for r in f.head(10).itertuples())
    return state, t0


def gate_surge(orders, st, btc):   # RETIRED Oct-5 (DECISION_LOG 202) — the SURGE_LONG row is gate_surge_b
    G = st.setdefault("gates", {}).setdefault("SURGE_LONG", {})
    t0 = _ms(PROBE_START)
    f = orders[(orders.entry_strategy.astype(str) == "SURGE_LONG") & (orders.o_ms >= t0)].copy()
    f["trig"] = _ms_series(f.entry_surge_trigger_at).values
    f["trig"] = f.trig.where(np.isfinite(f.trig), f.o_ms)            # unstamped → the fill time (window = the fill)
    win, n_open = [], 0
    for trig, g in f.groupby("trig"):
        if not ((g.status.astype(str) == "CLOSED") & g.pnl_percentage.notna()).all():
            n_open += 1                                               # a window counts once every fill of it has closed
            continue
        r72 = btc_r72_at(btc, int(trig))
        win.append((r72, float(g.pnl_percentage.mean()), int(trig), len(g)))
    state, info = decide_surge([(r, m) for r, m, _, _ in win])
    G["progress"] = (f"{len(win)}/8 windows · BTC 3d ≤ +2.7 %: {info['n_lo']} (avg {_f(info['m_lo'])}, CI {_f(info['ci_lo'][0])}…{_f(info['ci_lo'][1])})"
                     f" · rest {info['n_hi']} (avg {_f(info['m_hi'])})"
                     + (f" · {sum(1 for r, *_ in win if r is None)} unreadable 3d return" if any(r is None for r, *_ in win) else "")
                     + (f" · {n_open} window(s) still open" if n_open else ""))
    G["detail"] = " · ".join(f"{_fmt_t(t)} 3d {_f(r, '+.1f')} n{k} {m:+.2f}" for r, m, t, k in win[:10])
    return state


def gate_surge_b(orders, st):
    """⚡ option B: SURGE_LONG triggers (fills grouped by entry_surge_trigger_at) opened after the B deploy; a trigger counts once all its fills closed."""
    G = st.setdefault("gates", {}).setdefault("SURGE_LONG", {})
    t0 = deploy_ms("SURGE_B")
    f = orders[(orders.entry_strategy.astype(str) == "SURGE_LONG") & (orders.o_ms >= t0)].copy()
    f["trig"] = _ms_series(f.entry_surge_trigger_at).values
    f["trig"] = f.trig.where(np.isfinite(f.trig), f.o_ms)
    done, n_open = [], 0
    for trig, g in f.sort_values("o_ms").groupby("trig", sort=True):
        if not ((g.status.astype(str) == "CLOSED") & g.pnl_percentage.notna()).all():
            n_open += 1
            break   # review: the FIRST 15 = a closed prefix — never let a later trigger take the place of one still open
        done.append((int(trig), float(g.pnl_percentage.mean()), len(g)))
    vals = [m for _, m, _ in done]
    state = decide_surge_b(vals)
    G["progress"] = (f"{min(len(vals), 15)}/15 filled triggers closed" + (f" · mean {np.mean(vals[:15]):+.3f} %/trigger" if vals else "")
                     + (f" · {sum(1 for v in vals[:15] if v > 0)} positive" if vals else "") + (f" · {n_open} still open" if n_open else ""))
    G["detail"] = " · ".join(f"{_fmt_t(t)} n{k} {m:+.2f}" for t, m, k in done[:15])
    return state, t0


def gate_bearrun(orders, st):
    G = st.setdefault("gates", {}).setdefault("BEARRUN", {})
    t0 = _ms(PROBE_START)
    f = orders[(orders.entry_strategy.astype(str) == "BEARRUN_SHORT") & (orders.o_ms >= t0)].copy()
    f = f.sort_values("o_ms")
    f["win"] = window_chain(f.o_ms.tolist(), 180 * MIN)
    wins = []
    for w, g in f.groupby("win"):
        done = (g.status.astype(str) == "CLOSED").all() and g.pnl_percentage.notna().all()
        wins.append((int(g.o_ms.min()), len(g), float(g.pnl_percentage.mean()) if done else None, float(g.pnl_percentage.sum()) if done else None))
    closed = [w for w in wins if w[2] is not None]                    # a window with a fill still open is not counted yet
    state = decide_bearrun([w[2] for w in closed], sum(w[3] for w in closed))
    G["progress"] = (f"{len(closed)}/5 windows closed · {sum(1 for w in closed if w[2] > 0)} positive · Σ {sum(w[3] for w in closed):+.2f} %"
                     + (f" · {len(wins) - len(closed)} window(s) still open" if len(wins) > len(closed) else ""))
    G["detail"] = " · ".join(f"{_fmt_t(t)} n{k} {_f(m)}" for t, k, m, _ in wins[:10])
    return state


# ═══════════════════════════════ assembly ═══════════════════════════════
DEFS = {
    "CHOP_BURST": ("🌀👥 Chop∧burst block (201)", "first 6 refused momentum-LONG signals (LONG_CHOP_BURST) re-priced with the live exit replica: "
                   "WR ≥ 50 % ∨ Σ > 0", "set long_chop_burst_block_enabled false", "long_chop_burst_block_enabled"),
    "FRENZY_TP3": ("🎯 FRENZY TP +3 (199)", "first 20 FRENZY + WIDE fills after the +3 deploy re-priced with fixed +4/−3 on ticks (bot accounting): "
                   "+4/−3 avg > actual avg", "set frenzy_tp_pct 4", "frenzy_tp_pct"),
    "FRENZY_STRONG": ("💪 FRENZY strong leverage (197)", "first 10 sized-up FRENZY_LONG fills (ADX Δ > 0 ∧ DI spread > 0) avg < the normal "
                      "FRENZY_LONG fills of the same period, or < 0", "set frenzy_long_lev_mult_strong 0", "frenzy_long_lev_mult_strong"),
    "FRENZY_GVOL": ("🌊 FRENZY market-volume gate (194)", "first 20 FRENZY + WIDE fills under the gate: avg pnl % < 0", "set frenzy_gvol_max 0",
                    "frenzy_gvol_max"),
    "SURGE_LONG": ("⚡ SURGE_LONG option B (202)", "trigger 0.3 % · 5× · market vol ≥ 1 · spacing after a fill, FULL size (operator override, "
                   "unproven: year +0.01 %/trigger): the first 15 triggers that filled, mean pnl %/trigger ≤ 0 → revert (supersedes the 200 probe gate)",
                   "set surge_long_lev_mult 0.05 (back to the probe)", "surge_long_lev_mult"),
    "BEARRUN": ("🐻 BEARRUN probe arm bar (200)", "≥ 5 windows started after 10-04 22:00 with fills, ≥ 3 positive ∧ Σ > 0 (positive event)",
                "set bearrun_lev_mult 1.0", "bearrun_lev_mult"),
    "LOADX": ("🧭 LOADX gate (126)", "first 30 LOADX-only refused LONG signals (journal FAILS, rank ≤ 10 excluded), WINDOW units: WR ≥ 60 % ∨ net > 0 · extended from 8 on 10-04 (first 8 had FIRED, fragile at t+5m)",
              "set long_rsi_momentum_adx_max 0", "long_rsi_momentum_adx_max"),
    "HEAT": ("🫧 Heat re-scope (116)", "first 30 LONG_HEAT_BLOCK fires re-priced: WR ≥ 60 % (2nd leg — Jan–Jun replay expectancy — manual) · extended from 6 on 10-04 (first 6 had FIRED, 6/6 won)",
             "legs back to long_heat_btc_slope_min 0.07 · long_heat_btc_rsi_prev_min 64 · long_heat_bull_pct_min 80", "long_heat_bull_pct_min"),
    "MEGACAP": ("🏦 Mega-cap exclusion (110)", "LONG_MEGACAP_BLOCK refusals re-priced: ≥ 60 % WR ∧ Σ > 0 on N ≥ 8 across ≥ 3 windows",
                "set long_megacap_rank_max 0", "long_megacap_rank_max"),
}
ORDER = ["CHOP_BURST", "FRENZY_TP3", "FRENZY_STRONG", "FRENZY_GVOL", "SURGE_LONG", "BEARRUN", "LOADX", "HEAT", "MEGACAP"]


def _status_text(code, state, G):
    action = DEFS[code][2]
    if state == "fired":
        return f"🔔 FIRED → {action}" + (" (t+5m entry disagrees — fragile)" if G.get("fragile") else "")
    if state == "arm_group":
        return "🔔 ARM BAR MET → normal size for the BTC 3d ≤ +2.7 % group only (needs a group-scoped size switch)"
    if state == "armbar":
        return f"🔔 ARM BAR MET → {action}"
    if state == "holds":
        return "✅ holds — bar resolved, keep" + (" (t+5m entry disagrees — fragile)" if G.get("fragile") else "")
    if state == "open":
        return "⏳ bar not met — keep probing (neither branch)"
    if state == "nodata":
        return "⚠ no data"
    if state == "error":
        return "⚠ error"
    prov = G.get("provisional")
    return "⏳ collecting" + (f" (provisional on 1m prices: would {'FIRE' if prov == 'fired' else 'hold'})" if prov else "")


def run_section(now_ms=None, noted=None, record_notes=True):
    """→ (markdown lines, [(note_key, note_line)]). Never raises (each gate guarded); persists reports/SCOUT_REVERT_GATES.json.
    Notes are once per gate-state for good: the key is also kept in this state file (the scout's own note keys expire after
    10 days). record_notes=False (the standalone CLI preview) leaves them for the scout run to deliver."""
    now_ms = int(now_ms or time.time() * 1000)
    noted = noted or {}
    st = load_state()
    st.setdefault("gates", {})
    res, cov = {}, {}
    try:
        orders, newest, n_ord = load_orders()
    except Exception as e:
        log(f"orders: {e}")
        orders, newest, n_ord = pd.DataFrame(columns=list(ORDER_COLS) + ["o_ms"]), None, 0
    try:
        J, jcov = load_journal()
    except Exception as e:
        log(f"journal: {e}")
        J, jcov = None, None
    try:
        ranks = rank_map(orders) if len(orders) else {}
    except Exception as e:
        log(f"ranks: {e}")
        ranks = {}
    jtxt = (f"journal {_fmt_t(jcov[0])}→{_fmt_t(jcov[1])} ({jcov[2]} exports{', ' + str(jcov[3]) + ' gaps > 1 h' if jcov[3] else ''})"
            if jcov else "no decisions export in ~/Downloads")
    otxt = f"orders to {_fmt_t(newest)} ({n_ord} exports)" if newest else "no orders export in ~/Downloads"
    btc = None
    try:
        btc = btc5m_array(now_ms - 12 * DAY)
    except Exception as e:
        log(f"btc: {e}")
    budget = Budget(PRICE_BUDGET_S)
    sig_specs = [("CHOP_BURST", 6, 50.0, "or", None, False), ("LOADX", EXT_N["LOADX"][0], 60.0, "or", ranks, True),
                 ("HEAT", EXT_N["HEAT"][0], 60.0, "wr", ranks, False)]
    # phase 1 — which tick days do the unpriced items need? fetch them once
    need = set()
    for code, n, wr, mode, rk, win in sig_specs:
        try:
            gate_first_n_signals(code, J, st, n, wr, mode, btc, budget, now_ms, need, rk, win)
        except Exception as e:
            log(f"{code} phase 1: {e}")
    for fn in (lambda: gate_megacap(J, st, btc, budget, now_ms, need), lambda: gate_frenzy_tp(orders, st, budget, now_ms, need)):
        try:
            fn()
        except Exception as e:
            log(f"phase 1: {e}")
    fetched = 0
    try:
        fetched = ensure_ticks(need, st, now_ms)
    except Exception as e:
        log(f"ticks: {e}")
    # phase 2 — price + decide
    for code, n, wr, mode, rk, win in sig_specs:
        try:
            if (J is None or not len(J)) and not st["gates"].get(code, {}).get("items"):
                res[code] = "nodata"
                st["gates"].setdefault(code, {})["progress"] = "no decisions export covers it"
            else:
                res[code] = gate_first_n_signals(code, J, st, n, wr, mode, btc, budget, now_ms, None, rk, win)
            cov[code] = jtxt + (f" · shipped {SHIPS[code]}: earlier refusals only in the EB logs" if code in SHIPS and jcov and
                                _ms(SHIPS[code]) < jcov[0] - DAY else "")
            if code == "LOADX":
                cov[code] += " · FAILS sets start ≈ 09-30 19:00"
        except Exception as e:
            log(f"{code}: {e}")
            res[code] = "error"
            st["gates"].setdefault(code, {})["progress"] = f"error: {str(e)[:100]}"
    try:
        res["MEGACAP"] = gate_megacap(J, st, btc, budget, now_ms, None)
        cov["MEGACAP"] = jtxt + " · shipped 09-23: earlier refusals only in the EB logs"
    except Exception as e:
        log(f"MEGACAP: {e}")
        res["MEGACAP"] = "error"
        st["gates"].setdefault("MEGACAP", {})["progress"] = f"error: {str(e)[:100]}"
    for code, fn in (("FRENZY_TP3", lambda: gate_frenzy_tp(orders, st, budget, now_ms, None)),
                     ("FRENZY_STRONG", lambda: gate_frenzy_strong(orders, st)), ("FRENZY_GVOL", lambda: gate_frenzy_gvol(orders, st))):
        try:
            state, t0 = fn()
            res[code] = state
            cov[code] = otxt + f" · counts fills opened ≥ {_fmt_t(t0)} (deploy = push + 10 min)"
            if newest is None or newest < t0:
                res[code] = "collecting"
                st["gates"][code]["progress"] = "no export covers it yet"
        except Exception as e:
            log(f"{code}: {e}")
            res[code] = "error"
            st["gates"].setdefault(code, {})["progress"] = f"error: {str(e)[:100]}"
    try:
        res["SURGE_LONG"], _t0 = gate_surge_b(orders, st)
        cov["SURGE_LONG"] = otxt + f" · counts triggers from {_fmt_t(_t0)} (deploy = push + 10 min)"
        if newest is None or newest < _t0:
            res["SURGE_LONG"] = "collecting"
            st["gates"]["SURGE_LONG"]["progress"] = "no export covers it yet"
    except Exception as e:
        log(f"SURGE_LONG: {e}")
        res["SURGE_LONG"] = "error"
        st["gates"].setdefault("SURGE_LONG", {})["progress"] = f"error: {str(e)[:100]}"
    for code, fn in (("BEARRUN", lambda: gate_bearrun(orders, st)),):
        try:
            res[code] = fn()
            cov[code] = otxt + " · windows from 10-04 22:00 UTC"
            if newest is None or newest < _ms(PROBE_START):
                st["gates"][code]["progress"] = "no export covers it yet (probe windows count from 10-04 22:00 UTC)"
        except Exception as e:
            log(f"{code}: {e}")
            res[code] = "error"
            st["gates"].setdefault(code, {})["progress"] = f"error: {str(e)[:100]}"
    ftp = st["gates"].get("FRENZY_TP3", {}).get("items", {})
    if ftp:
        srcs = [x.get("src") for x in ftp.values() if x.get("src")]
        cov["FRENZY_TP3"] += f" · paths: {sum(1 for s in srcs if s == 'tick')} tick / {sum(1 for s in srcs if s != 'tick')} 1m"
    for code in ("CHOP_BURST", "LOADX", "HEAT", "MEGACAP"):
        if st["gates"].get(code, {}).get("priced"):
            cov[code] = cov.get(code, "") + " · " + st["gates"][code]["priced"]
    # render
    L = ["## ⏳ Revert gates (every open pre-committed revert / arm gate, tracked each run)", "",
         "Refused signals are re-priced with the live momentum-LONG exit replica (scripts/ml_exit_optimize.py) at the journal bucket "
         "+1 min (primary) and +5 min (in brackets); FRENZY fills on aggTrades ticks with the bot's accounting. A gate fires only on final "
         "(tick) prices; ᵖ = provisional 1m price. Decisions are the operator's — this table never changes config.", "",
         "| Gate | Definition (frozen) | Progress | Status | Config now | Data coverage |", "|---|---|---|---|---|---|"]
    notes = []
    for code in ORDER:
        G = st["gates"].setdefault(code, {})
        state = res.get(code, "error")
        prev = G.get("state")
        if state in ("fired", "armbar", "arm_group") and prev != state:
            G["state_at"] = now_ms
        G["state"] = state
        title, d, action, ck = DEFS[code]
        stt = _status_text(code, state, G)
        cv = cfg_value(ck)
        shown = (f"first {EXT_N[code][1]} (frozen gate): {EXT_N[code][2]} · first {EXT_N[code][0]} (extension): {stt}"
                 if code in EXT_N and state not in ("nodata", "error") else stt)
        L.append(f"| {title} | {d} | {G.get('progress', '–')} | {shown} | {ck} = {cv if cv is not None else '–'} | {cov.get(code, '–')} |")
        if state in ("fired", "armbar", "arm_group"):
            k = f"RG|{code}|n{EXT_N[code][0]}|{state}" if code in EXT_N else f"RG|{code}|{state}"   # N in the key: the frozen-N alert must not mute the extension
            if k not in noted and k not in G.get("noted", []):
                notes.append((k, f"🔔 Revert gate {title}: {stt[2:].strip()} — {G.get('progress', '')}"))
                if record_notes:
                    G.setdefault("noted", []).append(k)
    L += [""]
    for code in ORDER:
        dt = st["gates"].get(code, {}).get("detail")
        if dt:
            L.append(f"- {DEFS[code][0]}: {dt}")
    L += ["", f"_Tick archives fetched this run: {fetched}. State: reports/SCOUT_REVERT_GATES.json._", ""]
    st["updated_utc"] = _fmt_t(now_ms, True)
    try:
        atomic_write(STATE_JSON, json.dumps(st, indent=1, default=str))
    except Exception as e:
        log(f"state write failed: {e}")
    return L, notes


# ═══════════════════════════════ self-test ═══════════════════════════════
def selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    # CHOP_BURST — first 6, WR ≥ 50 ∨ Σ > 0
    chk(decide_first_n([0.3, -0.5, -0.5], 6, 50)[0] == "collecting", "chop: < 6 collects")
    chk(decide_first_n([0.3, 0.2, 0.1, -0.5, -0.5, -0.5], 6, 50)[0] == "fired", "chop: 3/6 = 50 % fires")
    chk(decide_first_n([0.9, -0.1, -0.1, -0.1, -0.1, -0.1], 6, 50)[0] == "fired", "chop: Σ > 0 fires at WR 17 %")
    chk(decide_first_n([0.1, 0.1, -0.5, -0.5, -0.5, -0.5], 6, 50)[0] == "holds", "chop: 2/6 Σ<0 holds")
    chk(decide_first_n([0.1, 0.1, -0.5, -0.5, -0.5, -0.5, 9, 9, 9], 6, 50)[0] == "holds", "chop: only the FIRST 6 count")
    # HEAT — WR ≥ 60 alone
    chk(decide_first_n([1, 1, 1, 1, -9, -9], 6, 60, or_sum_pos=False)[0] == "fired", "heat: 67 % fires even with Σ<0")
    chk(decide_first_n([1, 1, 1, -0.1, -0.1, -0.1], 6, 60, or_sum_pos=False)[0] == "holds", "heat: 50 % with Σ>0 holds")
    # LOADX — windows, WR ≥ 60 ∨ net > 0
    chk(decide_first_n([-0.1] * 7 + [1.0], 8, 60)[0] == "fired", "loadx: net > 0 fires")
    chk(decide_first_n([-0.1] * 8, 8, 60)[0] == "holds", "loadx: all losers holds")
    # MEGACAP — cumulative N ≥ 8, ≥ 3 windows, WR ≥ 60 ∧ Σ > 0
    chk(decide_megacap([1] * 8, [1, 1, 1, 1, 2, 2, 2, 2])[0] == "collecting", "mega: 2 windows not enough")
    chk(decide_megacap([1] * 5 + [-0.1] * 3, [1, 2, 3, 1, 2, 3, 1, 2])[0] == "fired", "mega: 62.5 % Σ>0 3 windows fires")
    chk(decide_megacap([5] * 4 + [-0.1] * 4, [1, 2, 3, 4] * 2)[0] == "collecting", "mega: 50 % does not fire")
    chk(decide_megacap([1] * 7, [1, 2, 3, 4, 5, 6, 7])[0] == "collecting", "mega: N 7 collects")
    # FRENZY TP — +4/−3 beats actual on average over the first 20
    chk(decide_tp([1.0] * 19, [2.0] * 19)[0] == "collecting", "tp: 19 collects")
    chk(decide_tp([0.5] * 20, [0.6] * 20)[0] == "fired", "tp: alt better fires")
    chk(decide_tp([0.5] * 20, [0.5] * 20)[0] == "holds", "tp: tie holds (must BEAT)")
    chk(decide_tp([0.5] * 20 + [-9] * 5, [0.4] * 20 + [9] * 5)[0] == "holds", "tp: only the first 20")
    # FRENZY strong — first 10 sized-up < normal or < 0
    chk(decide_strong([1.0] * 9, [0.5])[0] == "collecting", "strong: 9 collects")
    chk(decide_strong([0.4] * 10, [0.5] * 5)[0] == "fired", "strong: below normal fires")
    chk(decide_strong([-0.1] * 10, [])[0] == "fired", "strong: below 0 fires with no normal fills")
    chk(decide_strong([0.6] * 10, [0.5] * 5)[0] == "holds", "strong: above normal and > 0 holds")
    chk(decide_strong([0.6] * 10, [])[0] == "holds", "strong: > 0, no normal fills holds")
    # FRENZY gvol — first 20 avg < 0
    chk(decide_mean_neg([-0.1] * 19)[0] == "collecting", "gvol: 19 collects")
    chk(decide_mean_neg([-0.1] * 20)[0] == "fired", "gvol: negative fires")
    chk(decide_mean_neg([0.0] * 20)[0] == "holds", "gvol: zero holds (bar is < 0)")
    # SURGE_LONG — ≥ 8 windows, split at BTC 3d ≤ +2.7
    w = [(1.0, 0.5), (2.0, 0.6), (0.5, 0.4), (2.7, 0.7), (3.5, -0.2), (4.0, -0.1), (5.0, -0.3), (1.5, 0.55)]
    chk(decide_surge(w[:7])[0] == "collecting", "surge: 7 windows collects")
    chk(decide_surge(w)[0] == "arm_group", "surge: low group + CI > 0, rest ≤ 0 → arm group (2.7 inclusive)")
    chk(decide_surge([(1.0, -0.2), (2.0, 0.1), (0.5, -0.3)] + [(4.0, 0.5)] * 5)[0] == "fired", "surge: low group ≤ 0 → LONG off")
    chk(decide_surge([(1.0, 0.5)] * 5 + [(4.0, 0.5)] * 3)[0] == "open", "surge: rest > 0 → neither branch")
    chk(decide_surge([(None, 0.5)] + w[1:])[0] == "collecting", "surge: unreadable 3d return never guesses a group")
    chk(decide_surge([(4.0, 0.5)] * 8)[0] == "open", "surge: no low-group windows → open")
    # BEARRUN — ≥ 5 windows, ≥ 3 positive ∧ Σ > 0
    chk(decide_bearrun([0.1, 0.2, 0.3, -0.1], 0.5) == "collecting", "bear: 4 windows collects")
    chk(decide_bearrun([0.1, 0.2, 0.3, -0.1, -0.2], 0.3) == "armbar", "bear: 3/5 positive Σ>0 → arm bar")
    chk(decide_bearrun([0.1, 0.2, 0.3, -1.0, -1.0], -1.4) == "open", "bear: Σ<0 → not met")
    chk(decide_bearrun([0.1, 0.2, -0.3, -0.1, -0.2], 0.1) == "open", "bear: 2 positive → not met")
    # helpers
    chk(window_chain([0, 60, 400, 401], 180) == [0, 0, 1, 1], "window chain")
    ep = episodes(pd.DataFrame(dict(ms=[0, 10 * MIN, 70 * MIN, 135 * MIN, 0], pair=["A", "A", "A", "A", "B"])))
    chk(list(zip(ep.pair, ep.ms)) == [("A", 0), ("B", 0), ("A", 135 * MIN)], "episodes: chained ≤ 60 min = one signal")
    btc = np.array([[i * 5 * MIN, 1, 1, 1, 100.0 + (3 if i == 900 else 0)] for i in range(901)])
    chk(abs(btc_r72_at(btc, 900 * 5 * MIN + 5 * MIN) - 3.0) < 1e-9, "btc r72 at the trigger bar (close-stamped)")
    chk(btc_r72_at(btc, 10 * 5 * MIN) is None, "btc r72 unreadable without 3 days of history")
    of = pd.DataFrame(dict(status=["CLOSED", "CLOSED", "OPEN", "CLOSED"], pnl_percentage=[1.0, -2.0, np.nan, 3.0]))
    chk(_closed_prefix(of) == [1.0, -2.0], "first-N waits behind an earlier open fill")
    chk(_is_final(["tick"], 0, 1) and not _is_final(["1m"], 0, 1) and _is_final(["1m"], 0, FINAL_AFTER_MS + 1), "final rule")
    # status text / notes are once per gate-state
    chk(_status_text("CHOP_BURST", "fired", {}).startswith("🔔 FIRED → set long_chop_burst_block_enabled false"), "status text")
    chk(EXT_N["LOADX"][0] == 30 and EXT_N["HEAT"][0] == 30, "extension: trackers at 30")
    chk(f"RG|HEAT|n{EXT_N['HEAT'][0]}|fired" != "RG|HEAT|fired", "extension: note key differs from the frozen-N key")
    chk(decide_surge_b([0.5] * 14) == "collecting", "surge B: < 15 collects")
    chk(decide_surge_b([-0.1] * 15) == "fired", "surge B: mean ≤ 0 fires")
    chk(decide_surge_b([1.0] + [-0.05] * 14) == "holds", "surge B: mean > 0 holds")
    chk(decide_surge_b([-1.0] * 15 + [9.0] * 5) == "fired", "surge B: only the FIRST 15 count")
    print(f"selftest OK — {ok} checks")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        lines, notes = run_section(record_notes=False)
        print("\n".join(lines))
        for k, ln in notes:
            print("NOTE", k, ln)
