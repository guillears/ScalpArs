#!/usr/bin/env python3
"""archive_batch.py — ONE command to add a batch export to the master pool.

Usage:
  venv/bin/python scripts/archive_batch.py --orders <orders.csv> --decisions <decisions.csv> [--n <batch number>] [--dry-run]
                                           [--allow-same-stack]

Steps (stops on the first failure with a clear message; exit 2 = refused / failed, 3 = STACK_VERSION bump needed):
  ① batch number = max existing reports/BASELINE<n>_* + 1 (or --n) · window = first / last opened_at of the export · refuses an empty
     export, an export whose window overlaps an existing archive, or a destination that already holds a DIFFERENT file
  ② copy → reports/BASELINE<n>_batch<MMDD>-<MMDD>[_LIVE]_orders.csv / _decisions.csv (a *_live_* export gets _LIVE; a *_paper_* export is
     archived plain, like B6–B21). Re-running after a stop RESUMES: an archive byte-identical to the export counts as already copied.
  ③ refresh the BTC 5m cache reports/backtest_cache/k5m_full/BTCUSDT.csv to the last CLOSED bar (public klines; honours
     scripts/binance_cooldown.py, reads X-MBX-USED-WEIGHT-1M and pauses above WEIGHT_PAUSE, stops + arms the shared cooldown on 418 / 429)
  ④ scripts/build_willy_exit_cf.py (the WILLY exit CF covers the new fills; cache-only)
  ⑤ STACK_VERSION in scripts/build_master_pool.py must mention B<n> — else prints the exact suggested line prefix and stops (exit 3)
     unless --allow-same-stack. Source is NEVER edited here.
  ⑥ scripts/build_master_pool.py → scripts/screen_pool.py → scripts/validate_against_master.py
  ⑦ the new era's per-sleeve table (bot only: no MANUAL, no *_PROBE) — as traded N · WR · avg % · Σ% · $ and stack-kept N · avg % · $ —
     plus MANUAL count, EXTERNAL_CLOSE rows (any sleeve), bot fills the operator closed (scout_gatekit.operator_closed_mask:
     close_reason MANUAL / MANUAL_* / EXTERNAL_CLOSE on a bot sleeve — the gates' own exclusion rule), bracket-capped rows
  ⑧ DRAFT batch-boundary paragraph (CLAUDE_CURRENT_STATE.md) + DRAFT DECISION_LOG entry — printed only, never written
--dry-run runs ① and ⑦ only (no copy, no fetch, no build); the stack-kept columns show only if the master already holds the era.
"""
import argparse, datetime as dt, filecmp, glob, json, os, re, shutil, subprocess, sys, time
import urllib.error, urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
REPORTS = os.path.join(ROOT, "reports")
BTC_CACHE = os.path.join(REPORTS, "backtest_cache", "k5m_full", "BTCUSDT.csv")
BUILDER = os.path.join(ROOT, "scripts", "build_master_pool.py")
MASTER = os.path.join(REPORTS, "MASTER_POOL_stacked.csv")
DECISION_LOG = os.path.join(ROOT, "CLAUDE_ARCHIVE", "DECISION_LOG.md")
PY = sys.executable

KLINES_URL = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&startTime={start}&limit=1500"
BAR_MS = 300_000
WEIGHT_PAUSE = 1000          # pause to the next minute above this X-MBX-USED-WEIGHT-1M (hard budget 1,200 / min)
MAX_REQUESTS = 400           # ≈ 600k bars — a stale cache this old means something else is wrong


class ArchiveError(Exception):
    """a refusal / failed step — main() prints it and exits 2"""


class StackBumpNeeded(Exception):
    """⑤ — main() prints the suggestion and exits 3"""


# ───────────────────────────── ① number, window, overlap ─────────────────────────────
_ARCH_RE = re.compile(r"^BASELINE(\d+)_.*\.csv$")


def _is_orders_archive(name):
    return bool(_ARCH_RE.match(name)) and "_decisions" not in name and "_split_report" not in name


def existing_archives(reports=None):
    """[(n, path, first_opened, last_opened)] for every orders archive reports/BASELINE<n>_*.csv (journals / split reports skipped).
    A file without opened_at keeps its number but no window."""
    out = []
    for p in sorted(glob.glob(os.path.join(reports or REPORTS, "BASELINE*_*.csv"))):
        name = os.path.basename(p)
        m = _ARCH_RE.match(name)
        if not m or not _is_orders_archive(name):
            continue
        lo = hi = None
        try:
            d = pd.read_csv(p, usecols=lambda c: c == "opened_at", low_memory=False)
            if "opened_at" in d and d.opened_at.notna().any():
                o = d.opened_at.dropna().astype(str).str[:19]
                lo, hi = o.min(), o.max()
        except Exception:
            pass
        out.append((int(m.group(1)), p, lo, hi))
    return out


def max_baseline_n(reports=None):
    ns = [int(m.group(1)) for p in glob.glob(os.path.join(reports or REPORTS, "BASELINE*_*"))
          for m in [re.match(r"^BASELINE(\d+)_", os.path.basename(p))] if m]
    return max(ns) if ns else 0


def load_export(path):
    if not os.path.exists(path):
        raise ArchiveError(f"orders export not found: {path}")
    try:
        d = pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise ArchiveError(f"cannot read the orders export {path}: {e}")
    if "opened_at" not in d.columns:
        raise ArchiveError(f"{path} has no opened_at column — not an orders export")
    d = d[d.opened_at.notna()]
    if d.empty:
        raise ArchiveError(f"{path} is empty — nothing to archive")
    return d


def export_window(d):
    o = d.opened_at.astype(str).str[:19]
    return o.min(), o.max()


def export_mode(path):
    b = os.path.basename(path).lower()
    if "_live_" in b:
        return "LIVE"
    if "_paper_" in b:
        return "PAPER"
    return None


def archive_names(n, first, last, mode):
    a, b = first[5:7] + first[8:10], last[5:7] + last[8:10]
    span = a if a == b else f"{a}-{b}"
    tag = "_LIVE" if mode == "LIVE" else ""
    stem = f"BASELINE{n}_batch{span}{tag}"
    return f"{stem}_orders.csv", f"{stem}_decisions.csv"


def plan(orders, n=None, reports=None):
    """① → dict(n, first, last, mode, orders_dest, decisions_dest, resumed, export). Raises ArchiveError on any refusal."""
    reports = reports or REPORTS
    d = load_export(orders)
    first, last = export_window(d)
    arch = existing_archives(reports)
    same = [a for a in arch if filecmp.cmp(a[1], orders, shallow=False)]
    if same:
        if n is not None and int(n) != same[0][0]:
            raise ArchiveError(f"this export is already archived as BASELINE{same[0][0]} ({os.path.basename(same[0][1])}) — not as B{n}")
        n = same[0][0]
    elif n is None:
        n = max_baseline_n(reports) + 1
    n = int(n)
    if not same and any(a[0] == n for a in arch):
        clash = [os.path.basename(a[1]) for a in arch if a[0] == n]
        raise ArchiveError(f"BASELINE{n} already has an orders archive ({', '.join(clash)}) — pick another --n")
    for an, p, lo, hi in arch:
        if same and p == same[0][1]:
            continue
        if lo is not None and first <= hi and lo <= last:
            raise ArchiveError(f"export window {first} → {last} overlaps BASELINE{an} ({os.path.basename(p)}: {lo} → {hi}) — "
                               f"a fill would be archived twice (the master dies on the dedup key)")
    mode = export_mode(orders)
    od, dd = archive_names(n, first, last, mode)
    if same:
        od = os.path.basename(same[0][1])
        dd = od.replace("_orders", "_decisions") if "_orders" in od else dd
    return dict(n=n, first=first, last=last, mode=mode, orders_dest=os.path.join(reports, od),
                decisions_dest=os.path.join(reports, dd), resumed=bool(same), export=d)


# ───────────────────────────── ② copy ─────────────────────────────
def copy_archives(p, orders, decisions):
    for src, dst in ((orders, p["orders_dest"]), (decisions, p["decisions_dest"])):
        if not os.path.exists(src):
            raise ArchiveError(f"export not found: {src}")
        if os.path.exists(dst):
            if filecmp.cmp(src, dst, shallow=False):
                print(f"② {os.path.basename(dst)} already archived (identical) — kept")
                continue
            raise ArchiveError(f"{dst} exists with DIFFERENT content — refusing to overwrite an archive")
        shutil.copyfile(src, dst)
        print(f"② {os.path.basename(src)} → reports/{os.path.basename(dst)}")


# ───────────────────────────── ③ BTC 5m cache ─────────────────────────────
def _http_get(url, timeout=15):
    """→ (parsed JSON, headers dict). HTTPError propagates (418 / 429 handled by the caller)."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read()), {k.lower(): v for k, v in r.headers.items()}


def _last_open_ms(path):
    try:
        with open(path, "rb") as f:
            f.seek(max(-600, -os.path.getsize(path)), 2)
            for line in reversed(f.read().decode().strip().split("\n")):
                try:
                    return int(line.split(",")[0])
                except ValueError:
                    pass
    except Exception:
        pass
    return None


def _arm_cooldown(code, retry_after, now_ms):
    import binance_cooldown as BC
    try:
        secs = int(float(retry_after)) if retry_after else (7200 if code == 418 else 600)
    except (TypeError, ValueError):
        secs = 7200 if code == 418 else 600
    until = max(now_ms + secs * 1000, BC.read_until_ms())
    os.makedirs(os.path.dirname(BC.SHARED), exist_ok=True)
    with open(BC.SHARED, "w") as f:
        f.write(str(int(until)))                                   # ms (the reader accepts both units); never shortens a longer ban
    return secs


def _now_ms():
    return int(time.time() * 1000)


def refresh_btc_cache(path=None, now_ms=None, fetch=None, sleep=time.sleep):
    """append every CLOSED 5m BTCUSDT bar after the cache's last row → number of bars added. Budget-safe (see module doc)."""
    import binance_cooldown as BC
    path = path or BTC_CACHE
    fetch = fetch or _http_get
    now = _now_ms() if now_ms is None else int(now_ms)
    if BC.active(now_ms=now):
        raise ArchiveError(f"Binance cooldown active until {BC.read_until_ms()} ms ({BC.SHARED}) — retry after it expires")
    last = _last_open_ms(path)
    if last is None:
        raise ArchiveError(f"BTC cache {path} missing or unreadable — rebuild it with scripts/backtest_fetch_full5m.py first")
    last_closed = now - BAR_MS                                        # a bar is CLOSED once open_time + 5 min ≤ now
    added, cur, reqs = 0, last + BAR_MS, 0
    rows = []
    while cur <= last_closed:
        if reqs >= MAX_REQUESTS:
            raise ArchiveError(f"BTC cache refresh would need > {MAX_REQUESTS} requests — the cache is far too old, rebuild it")
        try:
            kl, hdr = fetch(KLINES_URL.format(start=cur))
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                secs = _arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now)
                raise ArchiveError(f"Binance {e.code} — stopped, shared cooldown armed for {secs} s")
            raise ArchiveError(f"Binance klines HTTP {e.code}: {e}")
        except Exception as e:
            raise ArchiveError(f"Binance klines request failed: {e}")
        reqs += 1
        if not kl:
            break
        for k in kl:
            t = int(k[0])
            if cur <= t <= last_closed:
                rows.append(f"{t},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{k[7]}\n")
                added += 1
        nxt = int(kl[-1][0]) + BAR_MS
        if nxt <= cur:
            break
        cur = nxt
        try:
            used = int((hdr or {}).get("x-mbx-used-weight-1m") or 0)
        except (TypeError, ValueError):
            used = 0
        if used > WEIGHT_PAUSE and cur <= last_closed:
            wait = 61 - (time.time() % 60)
            print(f"③ used weight {used}/min > {WEIGHT_PAUSE} — pausing {wait:.0f} s")
            sleep(wait)
    if rows:
        with open(path, "a") as f:
            f.writelines(rows)
    return added


# ───────────────────────────── ⑤ STACK_VERSION ─────────────────────────────
_SV_RE = re.compile(r'^STACK_VERSION\s*=\s*"([^"]+)"(.*)$', re.M)


def stack_line(builder=None):
    src = open(builder or BUILDER, encoding="utf-8").read()
    m = _SV_RE.search(src)
    if not m:
        raise ArchiveError("STACK_VERSION line not found in build_master_pool.py")
    return m.group(1), m.group(0)


def stack_mentions(line, n):
    return re.search(rf"(?<![A-Za-z0-9])B{int(n)}(?![0-9])", line) is not None


def suggest_stack(version, n, first, last, n_bot, n_manual, export_name, orders_dest, today=None):
    today = today or dt.datetime.utcnow().strftime("%Y-%m-%d")
    if version[:10] == today and len(version) > 10 and version[10:].isalpha():
        nxt = today + chr(ord(version[-1]) + 1)
    else:
        nxt = today + "a"
    short, prior = nxt[5:], version[5:]
    return (f'STACK_VERSION = "{nxt}"  # {short} (+ B{n} archived: fills {first[5:16].replace("T", " ")} → {last[5:16].replace("T", " ")} '
            f'UTC, {n_bot} bot fills + {n_manual} MANUAL, export {export_name} → reports/{os.path.basename(orders_dest)}). '
            f'Prior {prior} — ')


def check_stack(p, export_name, allow_same=False, builder=None, today=None):
    version, line = stack_line(builder)
    if stack_mentions(line, p["n"]):
        print(f"⑤ STACK_VERSION {version} mentions B{p['n']} ✓")
        return version
    d = p["export"]
    man = int((d.entry_strategy.astype(str) == "MANUAL").sum()) if "entry_strategy" in d else 0
    sug = suggest_stack(version, p["n"], p["first"], p["last"], len(d) - man, man, export_name, p["orders_dest"], today)
    if allow_same:
        print(f"⑤ STACK_VERSION {version} does not mention B{p['n']} — continuing (--allow-same-stack)")
        return version
    raise StackBumpNeeded(
        f"⑤ STACK_VERSION {version} does not mention B{p['n']}. Edit scripts/build_master_pool.py — replace the start of the line\n"
        f'     STACK_VERSION = "{version}"  # \n  with\n     {sug}\n'
        f"  (keep the rest of the line = the prior chain), then re-run the same command (it resumes). "
        f"Or pass --allow-same-stack if the stack truly does not change.")


# ───────────────────────────── ⑥ build chain ─────────────────────────────
def run_step(label, args):
    print(f"\n── {label}: {' '.join(os.path.relpath(a, ROOT) if os.path.isabs(a) and a.startswith(ROOT) else a for a in args)}")
    r = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-25:])
    print(tail)
    if r.returncode != 0:
        raise ArchiveError(f"{label} failed (exit {r.returncode}) — fix it, then re-run this command (it resumes)")
    return out


# ───────────────────────────── ⑦ report ─────────────────────────────
def _truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _sleeve(d):
    s = d.entry_strategy.astype(str)
    dr = d.direction.astype(str).str.lower()
    return pd.Series([("MOM-" + x) if st in ("MOMENTUM", "nan", "", "None") else ("FLIP-" + x if st.startswith("FLIP") else st)
                      for st, x in zip(s, dr)], index=d.index)


REDUCED_SIZE_SLEEVES = ("FRENZY_WILLY_HT",)   # 🎲¼ (DECISION_LOG 341) quarter-size lines — full-size-only rule: off the headline table, own line
# 🩳 (DECISION_LOG 348) FLATBASE_SHORT is FULL size (the momentum-long mults) → its own headline row via _sleeve (entry_strategy); its
# observe-only shadows ride the decisions export (e=FLATBASE rows) archived beside the orders file — scripts/scout_flatbase_short.py prices them.


def _probe(d):
    """reduced-size rows kept off the headline sleeve table: *_PROBE cell fires + REDUCED_SIZE_SLEEVES (FRENZY_WILLY_HT, ¼ size)."""
    src = (d.get("cell_multiplier_source", pd.Series("", index=d.index)).fillna("").astype(str)
           + d.get("pattern_cell_source", pd.Series("", index=d.index)).fillna("").astype(str))
    es = d.get("entry_strategy", pd.Series("", index=d.index)).astype(str)
    return src.str.upper().str.contains("PROBE") | es.isin(REDUCED_SIZE_SLEEVES)


def batch_report(d, era, master=None):
    """⑦ → (text, stats dict). d = the export; master = MASTER_POOL_stacked rows (or None)."""
    d = d.copy()
    if "status" in d:
        d = d[d.status.astype(str) == "CLOSED"]
    d["key"] = d.opened_at.astype(str).str[:19] + "|" + d.pair.astype(str) + "|" + d.direction.astype(str)
    d["sleeve"] = _sleeve(d)
    manual = d.entry_strategy.astype(str) == "MANUAL"
    bot = d[~manual & ~_probe(d)].copy()
    pnl, pct = pd.to_numeric(bot.pnl, errors="coerce"), pd.to_numeric(bot.pnl_percentage, errors="coerce")
    bot["_pnl"], bot["_pct"] = pnl, pct
    m = None
    if master is not None and len(master):
        mm = master[master.era.astype(str) == era].copy()
        if len(mm):
            mm["key"] = mm.opened_at.astype(str).str[:19] + "|" + mm.pair.astype(str) + "|" + mm.direction.astype(str)
            m = mm.set_index("key")
    L = [f"⑦ {era} per sleeve — bot only (no MANUAL, no *_PROBE); window {d.opened_at.astype(str).str[:16].min()} → "
         f"{d.opened_at.astype(str).str[:16].max()} UTC",
         f"{'sleeve':20s} {'N':>3s} {'WR%':>5s} {'avg%':>7s} {'Σ%':>8s} {'$':>9s}   |  {'kept':>4s} {'avg%':>7s} {'stack$':>9s}"]
    stats = {}

    def line(name, g):
        n = len(g)
        wr = 100 * (g._pnl > 0).mean() if n else float("nan")
        row = dict(n=n, wr=wr, avg=g._pct.mean(), sum=g._pct.sum(), usd=g._pnl.sum())
        if m is not None:
            k = m.reindex(g.key)
            k = k[_truthy(k.stack_keep.fillna(False))] if "stack_keep" in k else k.iloc[:0]
            row.update(kn=len(k), kavg=pd.to_numeric(k.stack_pct, errors="coerce").mean(), kusd=pd.to_numeric(k.stack_pnl, errors="coerce").sum())
            kept = f"{row['kn']:4d} {row['kavg']:+7.2f} {row['kusd']:+9.2f}" if row["kn"] else f"{0:4d} {'—':>7s} {0:+9.2f}"
        else:
            kept = f"{'—':>4s} {'—':>7s} {'—':>9s}"
        stats[name] = row
        L.append(f"{name:20s} {n:3d} {wr:5.0f} {row['avg']:+7.2f} {row['sum']:+8.2f} {row['usd']:+9.2f}   |  {kept}")

    for s, g in bot.groupby("sleeve"):
        line(s, g)
    line("TOTAL (bot)", bot)
    if m is None:
        L.append("  (stack-kept columns: the master does not hold this era yet — run without --dry-run)")
    cr = d.close_reason.astype(str) if "close_reason" in d else pd.Series("", index=d.index)
    ext = d[cr.str.startswith("EXTERNAL_CLOSE")]
    import scout_gatekit as GK                       # the shared operator-closed rule: bot fill ∧ close MANUAL / MANUAL_* / EXTERNAL_CLOSE
    opc = d[GK.operator_closed_mask(d) & ~manual]
    brk = d[_truthy(d.bracket_capped)] if "bracket_capped" in d else d.iloc[:0]
    probes = int((~manual & _probe(d)).sum())
    L.append(f"MANUAL fills: {int(manual.sum())}" + (f" · *_PROBE fills (excluded): {probes}" if probes else ""))   # (341: FRENZY_WILLY_HT counted here too — its own line below)
    for _rs in REDUCED_SIZE_SLEEVES:   # 🎲¼ (341) its own line, as traded (never blended into FRENZY_WILLY's full-size row)
        _g = d[(d.entry_strategy.astype(str) == _rs) & ~manual]
        if len(_g):
            _p = pd.to_numeric(_g.pnl_percentage, errors="coerce"); _u = pd.to_numeric(_g.pnl, errors="coerce")
            L.append(f"{_rs} (reduced size, separate line): N {len(_g)} · WR {100 * (_u > 0).mean():.0f}% · avg {_p.mean():+.2f} % · "
                     f"Σ {_p.sum():+.2f} % · ${_u.sum():+.2f}")

    def _lst(g, extra=None):
        return ", ".join(f"{p} {str(o)[5:16].replace('T', ' ')} {s}" + (f" ({extra(r)})" if extra else "")
                         for (_, r), p, o, s in zip(g.iterrows(), g.pair, g.opened_at, g.entry_strategy.astype(str))) or "none"
    L.append(f"EXTERNAL_CLOSE rows: {len(ext)} — {_lst(ext)}")
    L.append(f"bot fills closed by the operator (MANUAL / MANUAL_* / EXTERNAL_CLOSE — scout_gatekit): {len(opc)} — "
             + _lst(opc, lambda r: str(r.get('close_reason'))))
    L.append(f"bracket-capped rows: {len(brk)} — " + _lst(brk, lambda r: f"{r.get('leverage')}× of max {r.get('entry_bracket_max_leverage')}×"))
    stats.update(_manual=int(manual.sum()), _ext=len(ext), _opclose=len(opc), _brk=len(brk), _ext_rows=ext, _op_rows=opc, _brk_rows=brk,
                 _kept=m is not None)
    return "\n".join(L), stats


# ───────────────────────────── ⑧ drafts ─────────────────────────────
def next_dl_number(path=None):
    try:
        ns = [int(x) for x in re.findall(r"^## \((\d+)\)", open(path or DECISION_LOG, encoding="utf-8").read(), re.M)]
        return max(ns) + 1 if ns else 1
    except Exception:
        return None


def _fmt(o):
    return pd.Timestamp(o).strftime("%b-%d %H:%M")


def drafts(p, stats, export_name, version, dl_n=None, today=None):
    today = today or dt.datetime.utcnow().strftime("%Y-%m-%d")
    t = stats.get("TOTAL (bot)", {})
    kind = "LIVE" if p["mode"] == "LIVE" else "paper"
    sleeves = " · ".join(f"{s} {r['n']} · {r['avg']:+.2f} % · {r['usd']:+.0f} $" for s, r in stats.items()
                         if not s.startswith("_") and s != "TOTAL (bot)")
    kept = (f" → TODAY'S STACK {version} {t.get('kn', 0)} kept · {t.get('kavg', float('nan')):+.2f} % · {t.get('kusd', 0):+.0f} $"
            if stats.get("_kept") else " → (stack-kept: rebuild pending)")
    inter = []
    if stats["_opclose"]:
        inter.append("operator-closed: " + ", ".join(f"{a} {str(b)[5:16].replace('T', ' ')}" for a, b in zip(stats["_op_rows"].pair, stats["_op_rows"].opened_at)))
    if stats["_ext"]:
        inter.append("EXTERNAL_CLOSE: " + ", ".join(f"{a} {str(b)[5:16].replace('T', ' ')}" for a, b in zip(stats["_ext_rows"].pair, stats["_ext_rows"].opened_at)))
    if stats["_brk"]:
        inter.append("bracket-capped: " + ", ".join(f"{a} {str(b)[5:16].replace('T', ' ')}" for a, b in zip(stats["_brk_rows"].pair, stats["_brk_rows"].opened_at)))
    od = os.path.basename(p["orders_dest"])
    para = (f"> **📦 BATCH BOUNDARY {today} — B{p['n']} ({kind} batch: fills {_fmt(p['first'])} → {_fmt(p['last'])} UTC) ARCHIVED "
            f"`reports/{od}` (export {export_name}; journal beside it `{os.path.basename(p['decisions_dest'])}`), "
            f"{t.get('n', 0)} bot fills + {stats['_manual']} MANUAL. Bot as traded {t.get('n', 0)} · {t.get('wr', float('nan')):.0f} % · "
            f"avg {t.get('avg', float('nan')):+.2f} % · {t.get('usd', 0):+.0f} $ ({sleeves}){kept}."
            + (f" Interference: {'; '.join(inter)}." if inter else "") + " BTC 5m cache refreshed (validate F2). DECISION_LOG "
            + (f"{dl_n}" if dl_n else "<n>") + ".**")
    dl = [f"## ({dl_n if dl_n else '<n>'}) {today} — 📦 B{p['n']} archived ({kind} batch) + master STACK {version}",
          f"- **Archive:** export {export_name} → reports/{od} (+ the decisions journal beside it).",
          f"  - **Window:** fills {_fmt(p['first'])} → {_fmt(p['last'])} UTC.",
          f"  - **Rows:** {t.get('n', 0) + stats['_manual']} = {t.get('n', 0)} bot + {stats['_manual']} MANUAL (MANUAL_EXEMPT in the master).",
          f"- **Master:** build_willy_exit_cf.py → build_master_pool.py (STACK {version}) → screen_pool.py → validate_against_master.py."
          + (f" B{p['n']} stack-kept {t.get('kn', 0)} · {t.get('kavg', float('nan')):+.2f} % · {t.get('kusd', 0):+.0f} $." if stats.get("_kept") else ""),
          "- **Per sleeve, bot only, as traded:**"]
    for s, r in stats.items():
        if s.startswith("_") or s == "TOTAL (bot)":
            continue
        dl.append(f"  - {s} {r['n']} · {r['wr']:.0f} % · {r['avg']:+.2f} % · {r['usd']:+.0f} $")
    if inter:
        dl.append("- **Interference:** " + "; ".join(inter) + ".")
    return para, "\n".join(dl)


# ───────────────────────────── main ─────────────────────────────
def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--orders", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-same-stack", action="store_true")
    a = ap.parse_args(argv)
    export_name = os.path.splitext(os.path.basename(a.orders))[0]
    p = plan(a.orders, a.n)
    print(f"① B{p['n']}: {len(p['export'])} rows, window {p['first']} → {p['last']} UTC, mode {p['mode'] or 'unknown (archived plain)'}"
          + (" — RESUMING (already archived)" if p["resumed"] else "")
          + f"\n   → reports/{os.path.basename(p['orders_dest'])} · reports/{os.path.basename(p['decisions_dest'])}")
    if p["mode"] is None:
        print("   ⚠ the filename says neither _live_ nor _paper_ — archived without the _LIVE tag")
    if "status" in p["export"] and (p["export"].status.astype(str) != "CLOSED").any():
        print(f"   ⚠ {int((p['export'].status.astype(str) != 'CLOSED').sum())} non-CLOSED rows — the master ignores them (still open at export?)")
    if not a.dry_run and not os.path.exists(a.decisions):
        raise ArchiveError(f"decisions export not found: {a.decisions}")
    era = f"B{p['n']}"
    version = stack_line()[0]
    if not a.dry_run:
        copy_archives(p, a.orders, a.decisions)
        n_add = refresh_btc_cache()
        print(f"③ BTC 5m cache +{n_add} closed bars → {os.path.relpath(BTC_CACHE, ROOT)}")
        run_step("④ WILLY exit CF", ["-I", os.path.join(ROOT, "scripts", "build_willy_exit_cf.py")])
        version = check_stack(p, export_name, a.allow_same_stack)
        run_step("⑥ master build", [os.path.join(ROOT, "scripts", "build_master_pool.py")])
        run_step("⑥ screened baseline", [os.path.join(ROOT, "scripts", "screen_pool.py")])
        run_step("⑥ validate against master", [os.path.join(ROOT, "scripts", "validate_against_master.py")])
    master = pd.read_csv(MASTER, low_memory=False) if os.path.exists(MASTER) else None
    text, stats = batch_report(p["export"], era, master)
    print("\n" + text)
    if not a.dry_run:
        para, dl = drafts(p, stats, export_name, version, next_dl_number())
        print("\n⑧ DRAFT — CLAUDE_CURRENT_STATE.md batch boundary (not written):\n" + para)
        print("\n⑧ DRAFT — CLAUDE_ARCHIVE/DECISION_LOG.md entry (not written):\n" + dl)
    return 0


def main(argv=None):
    try:
        return run(argv)
    except StackBumpNeeded as e:
        print(str(e))
        return 3
    except ArchiveError as e:
        print(f"✗ {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
