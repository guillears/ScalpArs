#!/usr/bin/env python3
"""📡 Route B — DAILY report (DECISION_LOG 597): health + per-sleeve coverage + inactive-sleeve coverage + the new-sleeve-hunt countdown.

Reads only SMALL files from s3://scalpars-route-b-recorder-euw1 (never the raw book / trades): health/status.json + health/daily/*,
events/* (the recorder's log — incl. every bot list it read: open positions with their sleeve, Frenzy-flagged coins, sleeves on / off) and
features/<date>/<HH>.csv.gz (one row per coin per minute, built on the server every hour). Downloads are cached in
reports/.route_b_cache/ (gitignored), so each run only fetches the new files.

  ① Health (scripts/route_b/health_check.verdict)
  ② ACTIVE sleeves: positions opened on the report day (UTC) and since recording began · how many have the full order book recorded AT
     ENTRY (a features row with book data in the entry minute) · the real book's cost of a $2.5k / $10k market order at entry (median)
  ③ INACTIVE sleeves: Frenzy-flagged coins that day and how many were recorded in full (their setups can be priced later from the journal)
  ④ Market cost at our size that day (bot coins) · ⑤ new-sleeve hunt: days recorded vs the 21 days the weekly scan waits for
Winners vs losers need results → read from the orders export at the weekly scan; the daily line counts toward the minimums.
Usage: venv/bin/python -I scripts/route_b/daily_report.py [YYYY-MM-DD]   (default: yesterday UTC) → prints + reports/route_b_daily/<day>.md"""
import gzip
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import health_check as H      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_D))
CACHE = os.path.join(ROOT, "reports", ".route_b_cache")
OUTDIR = os.path.join(ROOT, "reports", "route_b_daily")
REGION, BUCKET = H.REGION, H.BUCKET
SCAN_DAYS = 21
WL_MIN = 30              # fills per sleeve before a winners-vs-losers read is even attempted (≥ 15 per side at the scan)
ACTIVE_ORDER = ("EXHAUSTION_SHORT", "DISTRIBUTION_SHORT", "FLATBASE_SHORT", "FRENZY_WILLY_HT", "FRENZY_WILLY")


def aws(*args):
    try:
        p = subprocess.run(["aws", "--region", REGION, *args], capture_output=True, timeout=300)
        return p.returncode, p.stdout, p.stderr.decode(errors="replace")
    except (subprocess.TimeoutExpired, OSError) as e:
        return 1, b"", str(e)


def s3_keys(prefix):
    rc, out, _ = aws("s3", "ls", "--recursive", f"s3://{BUCKET}/{prefix}")
    if rc != 0:
        return []
    return [ln.split(None, 3)[3] for ln in out.decode().splitlines() if len(ln.split(None, 3)) == 4]


def fetch(key):
    """the object's bytes, cached under CACHE (immutable data files; status.json is never cached)."""
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        return open(path, "rb").read()
    rc, out, _ = aws("s3", "cp", "--only-show-errors", f"s3://{BUCKET}/{key}", "-")
    if rc != 0 or not out:
        return None
    if key.endswith(".gz"):
        try:
            gzip.decompress(out)                                     # a truncated download is never cached
        except (OSError, EOFError):
            return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, path)
    return out


def read_events(blobs):
    ev = []
    for b in blobs:
        try:
            with gzip.open(io.BytesIO(b), "rt") as f:
                for ln in f:
                    try:
                        ev.append(json.loads(ln))
                    except ValueError:
                        continue
        except (EOFError, OSError):
            continue
    return ev


def positions(ev, paper=None):
    """bot-list events → one row per position (pair, sleeve, direction, opened ISO, paper), first seen — from the OPEN map and the
    RECENTLY CLOSED map (fills that opened and closed between two publishes). paper True / False → that mode only. Pure."""
    seen = {}
    for e in ev:
        if e.get("ev") != "bot" or (paper is not None and e.get("paper") is not None and bool(e.get("paper")) != paper):
            continue
        items = [(p, v) for p, v in (e.get("open") or {}).items()] if isinstance(e.get("open"), dict) else []
        items += [(v.get("p"), v) for v in (e.get("recent") or {}).values() if isinstance(v, dict)] if isinstance(e.get("recent"), dict) else []
        for pair, v in items:
            if not pair or not isinstance(v, dict) or not v.get("o"):
                continue
            k = (pair, v["o"])
            if k not in seen:
                seen[k] = dict(pair=pair, sleeve=v.get("s") or "OTHER", direction=v.get("d"), opened=v["o"], paper=e.get("paper"), seen_ms=e.get("r"))
    return pd.DataFrame(list(seen.values()), columns=["pair", "sleeve", "direction", "opened", "paper", "seen_ms"])


def current_paper(ev):
    for e in reversed(ev):
        if e.get("ev") == "bot" and e.get("paper") is not None:
            return bool(e["paper"])
    return None


def feature_keys_needed(all_keys, day, pos):
    """only the summaries the report reads: the report day's 24 hours + the entry hour of every position. Pure."""
    want = {f"/{day}/"}
    hours = set()
    for o in (pos.opened if len(pos) else []):
        try:
            t = pd.Timestamp(o, tz="UTC")
            hours.add(f"features/{t:%Y-%m-%d}/{t:%H}.csv.gz")
        except (ValueError, TypeError):
            continue
    return [k for k in all_keys if any(w in k for w in want) or k in hours]


def last_sleeves(ev):
    for e in reversed(ev):
        if e.get("ev") == "bot" and isinstance(e.get("sleeves"), dict) and e["sleeves"]:
            return e["sleeves"]
    return {}


def flagged_on(ev, day):
    out = set()
    for e in ev:
        if e.get("ev") == "bot" and e.get("r") and datetime.fromtimestamp(e["r"] / 1000, timezone.utc).strftime("%Y-%m-%d") == day:
            out.update(e.get("flagged") or [])
    return out


def entry_book(pos, feats):
    """per position: was the book recorded in the entry minute (features row with n_book > 0) + the $2.5k / $10k cost there. Pure."""
    if not len(pos):
        return pos.assign(recorded=[], cost25=[], cost10=[])
    f = feats[feats.n_book > 0].set_index(["sym", "minute"]) if len(feats) else None
    rec, c25, c10 = [], [], []
    for r in pos.itertuples():
        try:
            m = int(pd.Timestamp(r.opened, tz="UTC").value // 1_000_000) // 60_000 * 60_000
        except (ValueError, TypeError):
            m = None
        row = None
        for mm in ((m, m - 60_000, m + 60_000) if m is not None else ()):   # ±1 minute: opened_at is the DB transaction time
            if f is not None and (r.pair, mm) in f.index:
                row = f.loc[(r.pair, mm)]
                row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                break
        rec.append(row is not None)
        side_in = "buy" if str(r.direction).upper() == "LONG" else "sell"
        c25.append(row[f"cost_{side_in}_2500"] if row is not None else None)
        c10.append(row[f"cost_{side_in}_10000"] if row is not None else None)
    return pos.assign(recorded=rec, cost25=pd.to_numeric(pd.Series(c25, index=pos.index), errors="coerce"),
                      cost10=pd.to_numeric(pd.Series(c10, index=pos.index), errors="coerce"))


def load_features(keys):
    fr = []
    for k in keys:
        b = fetch(k)
        if not b:
            continue
        try:
            d = pd.read_csv(io.BytesIO(b), compression="gzip")
            fr.append(d[d.n_book > 0])                               # mark-only rows (~92 %) are not read here
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=["sym", "minute", "n_book"])
    a = pd.concat(fr, ignore_index=True)
    # a minute split across two hourly files (exchange time vs receive hour) → keep the fuller row
    return a.sort_values("n_book").drop_duplicates(["sym", "minute"], keep="last").reset_index(drop=True)


GREEN, YELLOW, RED, WHITE = "🟢", "🟡", "🔴", "⚪"
LEGEND = ("Colours (fixed rules, data quality only — a day never makes a finding): 🟢 recorder healthy and ≥ 90 % of the sleeve's trades "
          "had the order book recorded at entry · 🟡 recorder degraded, or 50–90 % recorded · 🔴 recorder down, or < 50 % recorded "
          "(that day's data can't be used) · ⚪ no trades that day (Frenzy line: no flagged coins). Red wins over yellow. The recorder's "
          "health is THAT DAY's (its daily record: > 1 h of gaps 🔴 · > 5 min, a restart or no bot list 🟡), the live one for today. "
          "Hourly summaries: 🟢 24 · 🟡 20–23 · 🔴 fewer · ⚪ day still running.")


def coverage_colour(health_code, n, recorded):
    """the daily data-quality colour of one row. health_code = health_check.verdict code (0 ok · 1 degraded · 2 down). Pure."""
    if health_code == 2:
        return RED
    if not n:
        return WHITE
    share = recorded / n
    if share < 0.5:
        return RED
    if health_code == 1 or share < 0.9:
        return YELLOW
    return GREEN


GAP_RED_S, GAP_YELLOW_S = 3600, 300


def day_health(day, today, daily, now_code):
    """the recorder's health code FOR THAT DAY (0 ok · 1 degraded · 2 down) + why. Today → the live verdict. A past day → its own daily
    summary: none → 2 (the recorder never reached that day's rollover) · reconnect gaps > 1 h → 2 · > 5 min, a restart during the day
    (the process started after its midnight — time down before a restart is not in the gap counters) or no bot list at rollover → 1. Pure."""
    if day >= today:
        return now_code, "live"
    if not daily:
        return 2, "no daily record (recorder not running at that day's end)"
    gaps = sum(float(v or 0) for v in (daily.get("day_gap_s") or {}).values())
    if gaps > GAP_RED_S:
        return 2, f"{gaps / 60:.0f} min of gaps"
    why = []
    if gaps > GAP_YELLOW_S:
        why.append(f"{gaps / 60:.0f} min of gaps")
    try:
        d0 = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        if int(daily.get("started") or 0) > d0:
            why.append(f"recorder (re)started {datetime.fromtimestamp(int(daily['started']) / 1000, timezone.utc):%H:%M} UTC")
    except (TypeError, ValueError):
        pass
    if not int(daily.get("bot_n") or 0):
        why.append("no bot list at the day's end")
    return (1, "; ".join(why)) if why else (0, "no gaps, ran all day")


def summaries_colour(n_hours, day_complete=True):
    """a finished day: 24 hourly summaries 🟢 · 20–23 🟡 · fewer 🔴 · a day still running ⚪ (not judged). Pure."""
    if not day_complete:
        return WHITE
    if n_hours >= 24:
        return GREEN
    return YELLOW if n_hours >= 20 else RED


def build(day=None):
    day = day or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    now_ms = int(time.time() * 1000)
    st = None
    rc, out, _ = aws("s3", "cp", "--only-show-errors", f"s3://{BUCKET}/health/status.json", "-")
    if rc == 0:
        try:
            st = json.loads(out)
        except ValueError:
            st = None
    code, label, why = H.verdict(st, now_ms)
    ev_keys = [k for k in s3_keys("data/events/") if k.endswith(".jsonl.gz")]
    ev = read_events([b for b in (fetch(k) for k in ev_keys) if b])
    days = sorted({k.split("/")[2] for k in ev_keys})
    first_day = days[0] if days else day
    feat_keys = s3_keys("features/")
    mode = current_paper(ev)
    pos0 = positions(ev, mode)
    feats_all = load_features(feature_keys_needed(feat_keys, day, pos0))
    pos = entry_book(pos0, feats_all)
    pos["day"] = pos.opened.astype(str).str[:10] if len(pos) else []
    sleeves = last_sleeves(ev)
    active = [s for s in ACTIVE_ORDER if sleeves.get(s)] + [s for s, on in sleeves.items() if on and s not in ACTIVE_ORDER]
    inactive = [s for s, on in sleeves.items() if not on]
    daily = H._json(f"health/daily/{day}.json") if hasattr(H, "_json") else None
    L = [f"# 📡 Route B — daily report {day} (UTC)", "",
         f"**Health now: {label}**" + ("" if not why else " — " + "; ".join(why)), "", f"*{LEGEND}*", ""]
    if daily:
        gaps = daily.get("day_gap_s") or {}
        L.append(f"- {day}: gaps (s) {gaps} · coins in full detail {daily.get('day_hot_unique')} · messages {daily.get('day_n')}")
    if st:
        L.append(f"- now: {st.get('hot_n')} coins in full detail ({st.get('bot_n')} from the bot's list) · summaries built "
                 f"{st.get('features_done')} (failed {st.get('features_fail')}, newest {st.get('features_last')}) · disk {st.get('disk_free_gb')} GB free")
    fd = {k for k in feat_keys if f"/{day}/" in k and k.endswith(".csv.gz")}
    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dcode, dwhy = day_health(day, _today, daily, code)
    L += [f"- {[GREEN, YELLOW, RED][dcode]} recorder on {day}: {dwhy}",
          f"- {summaries_colour(len(fd), day < _today)} per-minute summaries for {day}: {len(fd)} of 24 hours"
          + (" (day still running)" if day >= _today else "") + (" (recording began this day)" if day == first_day else ""), "",
          "## Active sleeves — trades with the order book recorded at entry", "",
          f"Mode: {'PAPER' if mode else 'LIVE' if mode is not None else '—'} fills only. Trades = positions seen in the bot's list (open, or "
          "closed within the last 30 min of a publish) — a lower bound if the bot paused publishing.", "",
          f"| | Sleeve | trades {day} | book at entry | since {first_day} | book at entry (total) | median entry cost $2.5k / $10k (bps) | toward W/L ({WL_MIN}) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in active or ["(no bot list yet)"]:
        p = pos[pos.sleeve == s] if len(pos) else pos
        pd_ = p[p.day == day] if len(p) else p
        c25 = p.cost25.median() if len(p) and p.cost25.notna().any() else float("nan")
        c10 = p.cost10.median() if len(p) and p.cost10.notna().any() else float("nan")
        n_rec = int(pd_.recorded.sum()) if len(pd_) else 0
        L.append(f"| {coverage_colour(dcode, len(pd_), n_rec)} | {s} | {len(pd_)} | {n_rec}/{len(pd_)} | {len(p)} | "
                 f"{int(p.recorded.sum()) if len(p) else 0}/{len(p)} | "
                 + ("–" if c25 != c25 else f"{c25:.1f}") + " / " + ("–" if c10 != c10 else f"{c10:.1f}")
                 + f" | {int(p.recorded.sum()) if len(p) else 0}/{WL_MIN} |")
    other = pos[~pos.sleeve.isin(active)] if len(pos) else pos
    for lab, g in ((("manual (your own trades)"), other[other.sleeve == "MANUAL"] if len(other) else other),
                   ("sleeves off now / other", other[other.sleeve != "MANUAL"] if len(other) else other)):
        if len(g):
            gd = g[g.day == day]
            L.append(f"| {coverage_colour(dcode, len(gd), int(gd.recorded.sum()))} | {lab} | {len(gd)} | {int(gd.recorded.sum())}/{len(gd)} | "
                     f"{len(g)} | {int(g.recorded.sum())}/{len(g)} | – | – |")
    fl = flagged_on(ev, day)
    fday = feats_all[(pd.to_datetime(feats_all.minute, unit="ms", utc=True).dt.strftime("%Y-%m-%d") == day) & (feats_all.n_book > 0)] if len(feats_all) else feats_all
    rec_syms = set(fday.sym) if len(fday) else set()
    L += ["", "## Inactive sleeves — setups the bot saw but no sleeve trades", "",
          f"- Off now: {', '.join(inactive) if inactive else '—'}.",
          f"- {coverage_colour(dcode, len(fl), len(fl & rec_syms))} Frenzy-flagged coins on {day}: {len(fl)} · recorded in full detail: {len(fl & rec_syms)} "
          + (f"(missing: {', '.join(sorted(fl - rec_syms))})" if fl - rec_syms else "(none missing)") + ". Their signals sit in the decision journal and can be priced "
          "with the recorded book at the weekly scan.", ""]
    if len(fday):
        botc = fday[fday.sym.isin(fl)]
        if len(botc):
            L += ["## Cost at our size on the real book (Frenzy-flagged coins, median of the day)", "",
                  f"- spread {botc.spread_bps.median():.1f} bps · $2.5k market order {botc.cost_buy_2500.median():.1f} / "
                  f"{botc.cost_sell_2500.median():.1f} bps (buy / sell) · $10k {botc.cost_buy_10000.median():.1f} / {botc.cost_sell_10000.median():.1f} bps", ""]
    n_days = (datetime.strptime(day, "%Y-%m-%d") - datetime.strptime(first_day, "%Y-%m-%d")).days + 1
    L += ["## New-sleeve hunt", "",
          f"- {n_days} of {SCAN_DAYS} days recorded — the weekly research scan starts at day {SCAN_DAYS} "
          f"(≈ {(datetime.strptime(first_day, '%Y-%m-%d') + timedelta(days=SCAN_DAYS)).strftime('%Y-%m-%d')}). Nothing is read daily.", ""]
    os.makedirs(OUTDIR, exist_ok=True)
    txt = "\n".join(L)
    with open(os.path.join(OUTDIR, f"{day}.md"), "w") as f:
        f.write(txt)
    return code, txt


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    code, txt = build(argv[0] if argv else None)
    print(txt)
    return code


if __name__ == "__main__":
    sys.exit(main())
