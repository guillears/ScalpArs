#!/usr/bin/env python3
"""📡 Route B health check (DECISION_LOG 597) — run from the operator's Mac (aws CLI); the daily Claude routine reads its output.

Reads s3://<BUCKET>/health/status.json (written by the recorder every 5 min) and health/daily/*.json, lists the data stored, and prints a
plain-English verdict:
  ✅ HEALTHY   status ≤ 15 min old · mark / book / trade data ≤ 2 min old at the status time · bot Frenzy list read (bot_n > 0)
  ⚠ DEGRADED   recording, but the bot list is missing / stale (only the recorder's own surge list is used) or the disk is low (< 3 GB)
  ❌ DOWN      status older than 15 min, or a main stream silent > 2 min
Liquidations can be quiet for minutes in calm markets — their age is shown, never judged.
FULL BOOK (④, the bot's priority coins): ⚠ DEGRADED (never ❌ — the top-20 recording is the core) when, ≥ 10 min after the start, fewer
  than 80 % of the priority coins have a synced book, the full-book stream is down / its data > 2 min old, snapshot requests are paused
  by Binance (418 / 429 — the CURRENT pause's cause), today's coverage (synced coin-seconds ÷ wanted) is < 90 % after ≥ 1 h of
  uptime, or fullbook.py failed to load while FULL_MAX > 0. "Synced" = an fbook line in the last 10 s. Reported: synced / priority coins
  (wanted before the cap), syncs · gaps (each one = a resync) · stale snapshots · crossed books · coverage today · REST weight used.
OPEN INTEREST / POSITIONING (⑤): ⚠ when (≥ 10 min after the start) the last OI sweep is > 5 min old or read < 80 % of its coins, the last
  ratio sweep WITH DATA is > 15 min old, the last one read 0 coins or < 80 % of those it tried, the budget skipped > 20 % of the
  coins, Binance paused them, or positioning.py failed to load without POS_OFF. Reported: coins · read · today's coverage ·
  coins the ratio budget skipped.
--restart  when ❌ DOWN: send `systemctl restart scalpars-recorder` through SSM (no SSH) and say so.
Usage: venv/bin/python -I scripts/route_b/health_check.py [--restart]   → exit 0 HEALTHY · 1 DEGRADED · 2 DOWN"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

REGION = "eu-west-1"
BUCKET = "scalpars-route-b-recorder-euw1"
NAME = "scalpars-recorder"
STALE_STATUS_S, STALE_STREAM_S, LOW_DISK_GB, BACKLOG_S = 900, 120, 3.0, 7200
MAIN = ("mark", "book", "trade")
FULL_WARMUP_S, FULL_MIN_SYNCED, FULL_MIN_COVERAGE = 600, 0.8, 90.0
OI_SLOW_S = 45
PRICE_GB_MONTH = {"STANDARD": 0.023, "DEEP_ARCHIVE": 0.00099}   # eu-west-1; data/ moves to DEEP_ARCHIVE after 30 days


def aws(*args, text=True):
    try:
        p = subprocess.run(["aws", "--region", REGION, *args], capture_output=True, text=text, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as e:
        return 1, "", f"{type(e).__name__}: {e}"
    return p.returncode, p.stdout, p.stderr


def verdict(st, now_ms):
    """status.json dict → (code 0/1/2, label, reasons). Pure."""
    if not st:
        return 2, "❌ DOWN", ["no status file in S3 (the recorder has never reported)"]
    why = []
    age = (now_ms - int(st.get("now") or 0)) / 1000.0
    if age > STALE_STATUS_S:
        return 2, "❌ DOWN", [f"last status {age / 60:.0f} min ago (> {STALE_STATUS_S // 60} min)"]
    ages = st.get("age_s") or {}
    dead = [k for k in MAIN if ages.get(k) is None or float(ages[k]) > STALE_STREAM_S]
    if dead:
        return 2, "❌ DOWN", [f"{k} silent ({ages.get(k)} s)" for k in dead]
    if not int(st.get("bot_n") or 0):
        why.append("bot Frenzy list not read (missing or > 30 min old) — recording on the recorder's own surge list only")
    if st.get("disk_free_gb") is not None and float(st["disk_free_gb"]) < LOW_DISK_GB:
        why.append(f"disk low: {st.get('disk_free_gb')} GB free")
    if st.get("writes_paused"):
        why.append(f"writes PAUSED (disk nearly full) — {st.get('writes_dropped')} messages dropped")
    fl = st.get("features_last")
    if "features_done" in st and (now_ms - int(st.get("started") or now_ms)) > 3 * 3_600_000:
        try:
            fl_ms = int(datetime.strptime(fl, "%Y-%m-%d %H").replace(tzinfo=timezone.utc).timestamp() * 1000) if fl else None
        except (TypeError, ValueError):
            fl_ms = None
        if fl_ms is None or now_ms - fl_ms > 3 * 3_600_000:
            why.append(f"hourly summaries behind (newest {fl or 'none'}, failed {st.get('features_fail')})")
    bl = st.get("upload_backlog_age_s")
    if bl is not None and float(bl) > BACKLOG_S:
        why.append(f"uploads are behind: the oldest finished file has waited {float(bl) / 3600:.1f} h (upload_fail {st.get('upload_fail')})")
    why += full_issues(st, now_ms) + pos_issues(st, now_ms)
    if why:
        return 1, "⚠ DEGRADED", why
    return 0, "✅ HEALTHY", []


def full_issues(st, now_ms):
    """status.json → the full-book (④) warnings; [] when healthy, not running yet or absent (an older recorder). Pure."""
    fb = st.get("full")
    if not isinstance(fb, dict):
        return []
    out = []
    rest = fb.get("rest") or {}                                     # the REST pause is shared with OI / the ticker: checked even when
    if float(rest.get("paused_s") or 0) > 0 and rest.get("pause_why") in (418, 429):   # the full book itself is off
        out.append(f"REST: Binance answered {rest.get('pause_why')} — every weight-counted read paused {rest.get('paused_s')} s "
                   f"more (lifetime 418: {rest.get('n418')} · 429: {rest.get('n429')})")
    if not fb.get("on"):                                            # asked for (FULL_MAX > 0) but fullbook.py did not load
        if int(fb.get("requested") or 0) > 0 and fb.get("module") is False:
            out.append("full book: OFF although FULL_MAX > 0 — fullbook.py failed to load")
        return out
    up_s = (now_ms - int(st.get("started") or now_ms)) / 1000.0
    n, syn = int(fb.get("n") or 0), int(fb.get("synced_n") or 0)
    if up_s < FULL_WARMUP_S or n == 0:
        return out
    if not fb.get("conn"):
        out.append("full book: the depth-diff stream is disconnected")
    elif syn < FULL_MIN_SYNCED * n:
        out.append(f"full book: only {syn} / {n} priority coins emitting a synced book (not: {', '.join((fb.get('not_synced') or [])[:8])})")
    cov, d = fb.get("coverage_day_pct"), fb.get("day") or {}
    if cov is not None and up_s >= 3600 and float(cov) < FULL_MIN_COVERAGE:   # uptime ≥ 1 h (the 2-min warm-up sync never trips it)
        out.append(f"full book: today's coverage {cov} % (< {FULL_MIN_COVERAGE:.0f} %) — gaps {d.get('gaps')} · stale {d.get('stale')} · "
                   f"silent streams {d.get('silent')} · snapshot errors {d.get('snap_err')}")
    return out


def pos_issues(st, now_ms):
    """status.json → the open-interest / positioning (⑤) warnings; [] when healthy, warming up or absent. Pure."""
    p = st.get("pos")
    if not isinstance(p, dict):
        return []
    if not p.get("on"):
        return ["open interest / positioning: OFF although POS_OFF is not set — positioning.py failed to load"] \
            if p.get("module") is False and not p.get("off") else []
    if (now_ms - int(st.get("started") or now_ms)) / 1000.0 < FULL_WARMUP_S:
        return []
    out = []
    if p.get("oi_age_s") is None or float(p["oi_age_s"]) > 300:
        out.append(f"open interest: last sweep with data {p.get('oi_age_s')} s ago (> 300 s)")
    elif p.get("oi_coins") and int(p.get("oi_ok") or 0) < 0.8 * int(p["oi_coins"]):
        out.append(f"open interest: last sweep read {p.get('oi_ok')} / {p.get('oi_coins')} coins (skipped {p.get('oi_skipped')})")
    if p.get("oi_sweep_s") is not None and float(p["oi_sweep_s"]) > OI_SLOW_S:
        out.append(f"open interest: one sweep took {p.get('oi_sweep_s')} s (> {OI_SLOW_S} s of its 60 s slot)")
    coins, ok, sk = int(p.get("ratio_coins") or 0), int(p.get("ratio_ok") or 0), int(p.get("ratio_skipped") or 0)
    if p.get("ratio_age_s") is None or float(p["ratio_age_s"]) > 900:
        out.append(f"positioning ratios: last sweep with data {p.get('ratio_age_s')} s ago (> 900 s)")
    elif coins and p.get("ratio_tried_ms") is not None and ok == 0:
        out.append(f"positioning ratios: the last sweep read 0 / {coins} coins")
    elif coins and ok < 0.8 * (coins - sk):
        out.append(f"positioning ratios: last sweep read {ok} / {coins - sk} coins it tried")
    if coins and sk > 0.2 * coins:
        out.append(f"positioning ratios: the request budget skipped {sk} / {coins} coins (> 20 %)")
    if float(p.get("ratio_paused_s") or 0) > 0:
        out.append(f"positioning ratios: paused {p.get('ratio_paused_s')} s by Binance's /futures/data limit (429 × {p.get('ratio_n429')})")
    return out


def pos_lines(p, label="open interest / positioning"):
    """the report lines for a status / daily 'pos' block. Pure."""
    if not isinstance(p, dict):
        return [f"- {label}: not reported (recorder code older than this change)"]
    if not p.get("on"):
        return [f"- {label}: OFF ({'POS_OFF=1' if p.get('off') else 'positioning.py missing'})"]
    d = p.get("day") or {}
    return [f"- {label}: OI last sweep {p.get('oi_ok')} / {p.get('oi_coins')} coins in {p.get('oi_sweep_s')} s ({p.get('oi_age_s')} s ago) · "
            f"coverage today {p.get('oi_coverage_day_pct')} %",
            f"  - ratios last sweep {p.get('ratio_ok')} / {p.get('ratio_coins')} coins (budget skipped {p.get('ratio_skipped')}) "
            f"({p.get('ratio_age_s')} s ago) · coverage today {p.get('ratio_coverage_day_pct')} % · requests in 5 min {p.get('ratio_5m')} · "
            f"429 {p.get('ratio_n429')} · today OI {d.get('oi_ok')} / {d.get('oi_want')} · ratios {d.get('ratio_ok')} / {d.get('ratio_want')}"]


def full_lines(fb, label="full book"):
    """the report lines for a status / daily 'full' block. Pure."""
    if not isinstance(fb, dict):
        return [f"- {label}: not reported (recorder code older than the full-book change)"]
    if not fb.get("on"):
        return [f"- {label}: OFF (FULL_MAX 0 or fullbook.py missing)"]
    d, t, r = fb.get("day") or {}, fb.get("totals") or {}, fb.get("rest") or {}
    capped = int(fb.get("n_want") or 0) - int(fb.get("n") or 0)
    return [f"- {label}: synced {fb.get('synced_n')} / {fb.get('n')} priority coins (bot wanted {fb.get('n_want')}, cap {fb.get('cap')}"
            + (f" — {capped} left out by the cap" if capped > 0 else "") + f") · stream {'up' if fb.get('conn') else 'DOWN'}",
            f"  - today: coverage {fb.get('coverage_day_pct')} % of wanted coin-seconds · syncs {d.get('syncs')} · gaps → resyncs "
            f"{d.get('gaps')} · stale snapshots {d.get('stale')} · crossed {d.get('crossed')} · refreshes {d.get('refreshes')} · "
            f"silent streams {d.get('silent')} · empty sides {d.get('empty')} · bad events {d.get('bad')} · snapshot errors {d.get('snap_err')}",
            f"  - since start: syncs {t.get('syncs')} · gaps {t.get('gaps')} · stale {t.get('stale')} · snapshots {r.get('snapshots')} "
            f"(weight {r.get('weight')}, used 1m {r.get('used_1m')}) · 429 {r.get('n429')} · 418 {r.get('n418')} · errors {r.get('errors')}"]


def _json(key):
    rc, out, _ = aws("s3", "cp", "--only-show-errors", f"s3://{BUCKET}/{key}", "-")
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def stored(prefix="data/", now_s=None):
    """ONE recursive listing of data/ → (files, bytes, {kind: (files, bytes)}, bytes older than 30 days). Pure on the CLI output."""
    rc, out, _ = aws("s3", "ls", "--recursive", f"s3://{BUCKET}/{prefix}")
    if rc != 0:
        return None, None, {}, None
    return parse_listing(out, now_s)


def parse_listing(out, now_s=None):
    """`aws s3 ls --recursive` lines ('YYYY-MM-DD HH:MM:SS size key') → (files, bytes, {kind: (files, bytes)}, bytes ≥ 30 days old). Pure."""
    now_s = time.time() if now_s is None else now_s
    n = size = old = 0
    kinds = {}
    for ln in out.splitlines():
        parts = ln.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            sz = int(parts[2])
            t = datetime.strptime(parts[0] + " " + parts[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
        key = parts[3]
        k = key.split("/")[1] if key.startswith("data/") and key.count("/") >= 2 else "?"
        n, size = n + 1, size + sz
        kn, ks = kinds.get(k, (0, 0))
        kinds[k] = (kn + 1, ks + sz)
        if now_s - t >= 30 * 86400:
            old += sz
    return n, size, kinds, old


def daily_list():
    rc, out, _ = aws("s3", "ls", f"s3://{BUCKET}/health/daily/")
    return sorted(ln.split()[-1] for ln in out.splitlines() if ln.strip().endswith(".json")) if rc == 0 else []


def instance_id():
    rc, out, _ = aws("ec2", "describe-instances", "--filters", f"Name=tag:Name,Values={NAME}", "Name=instance-state-name,Values=running",
                     "--query", "Reservations[].Instances[].InstanceId", "--output", "text")
    return (out.split() or [""])[0] if rc == 0 else ""     # one instance id even if several carry the tag


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true")
    a = ap.parse_args(argv)
    now_ms = int(time.time() * 1000)
    st = _json("health/status.json")
    code, label, why = verdict(st, now_ms)
    L = [f"## 📡 Route B recorder — {label}  ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})"]
    L += [f"- {w}" for w in why]
    if st:
        ages = st.get("age_s") or {}
        L.append(f"- last status {(now_ms - int(st.get('now') or 0)) / 60000:.1f} min ago · running since "
                 f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(int(st.get('started') or 0) / 1000))} · host {st.get('host')}")
        L.append("- newest data (s): " + " · ".join(f"{k} {ages.get(k)}" for k in ("mark", "book", "trade", "liq")))
        L.append(f"- hot coins {st.get('hot_n')} (from the bot's Frenzy list: {st.get('bot_n')}) · unique today {st.get('day_hot_unique')} · "
                 f"reconnects {st.get('reconnects')} · gap seconds today {st.get('day_gap_s')}")
        L.append(f"- messages today {st.get('day_n')} · files uploaded {st.get('uploaded')} (failed {st.get('upload_fail')}) · disk free "
                 f"{st.get('disk_free_gb')} GB")
        L += full_lines(st.get("full"))
        L += pos_lines(st.get("pos"))
    n, size, kinds, old = stored()
    if size is not None:
        gb, gb_old = size / 1e9, (old or 0) / 1e9
        cost = (gb - gb_old) * PRICE_GB_MONTH["STANDARD"] + gb_old * PRICE_GB_MONTH["DEEP_ARCHIVE"]
        L.append(f"- stored in S3: {n} files · {gb:.2f} GB ({gb_old:.2f} GB > 30 days old → DEEP_ARCHIVE, 12–48 h to restore) · "
                 f"≈ ${cost:.2f}/month")
        parts = [f"{k} {kinds[k][1] / 1e9:.2f} GB" for k in ("fbook", "fsnap", "oi", "ratio") if k in kinds]
        if parts:
            L.append("  - of which full book / OI / positioning: " + " · ".join(parts))
    dl = daily_list()
    if dl:
        L.append(f"- daily summaries: {len(dl)} (latest {dl[-1]})")
        d = _json(f"health/daily/{dl[-1]}")
        if d:
            L.append(f"  - {d.get('day')}: messages {d.get('day_n')} · gap seconds {d.get('day_gap_s')} · unique hot coins {d.get('day_hot_unique')}")
            if isinstance(d.get("full"), dict):
                L += ["  " + x for x in full_lines(d["full"], f"{d.get('day')} full book")]
            if isinstance(d.get("pos"), dict):
                L += ["  " + x for x in pos_lines(d["pos"], f"{d.get('day')} open interest / positioning")]
    if code == 2 and a.restart:
        iid = instance_id()
        if iid:
            rc, out, err = aws("ssm", "send-command", "--instance-ids", iid, "--document-name", "AWS-RunShellScript", "--parameters",
                               'commands=["systemctl restart scalpars-recorder"]', "--query", "Command.CommandId", "--output", "text")
            L.append(f"- 🔁 restart sent to {iid} (command {out.strip()})" if rc == 0 else f"- restart FAILED: {err.strip()[:200]}")
        else:
            L.append("- no running scalpars-recorder instance found — the server itself is down (needs a look)")
    print("\n".join(L))
    return code


if __name__ == "__main__":
    sys.exit(main())
