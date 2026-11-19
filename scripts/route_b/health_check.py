#!/usr/bin/env python3
"""📡 Route B health check (DECISION_LOG 597) — run from the operator's Mac (aws CLI); the daily Claude routine reads its output.

Reads s3://<BUCKET>/health/status.json (written by the recorder every 5 min) and health/daily/*.json, lists the data stored, and prints a
plain-English verdict:
  ✅ HEALTHY   status ≤ 15 min old · mark / book / trade data ≤ 2 min old at the status time · bot Frenzy list read (bot_n > 0)
  ⚠ DEGRADED   recording, but the bot list is missing / stale (only the recorder's own surge list is used) or the disk is low (< 3 GB)
  ❌ DOWN      status older than 15 min, or a main stream silent > 2 min
Liquidations can be quiet for minutes in calm markets — their age is shown, never judged.
--restart  when ❌ DOWN: send `systemctl restart scalpars-recorder` through SSM (no SSH) and say so.
Usage: venv/bin/python -I scripts/route_b/health_check.py [--restart]   → exit 0 HEALTHY · 1 DEGRADED · 2 DOWN"""
import argparse
import json
import subprocess
import sys
import time

REGION = "eu-west-1"
BUCKET = "scalpars-route-b-recorder-euw1"
NAME = "scalpars-recorder"
STALE_STATUS_S, STALE_STREAM_S, LOW_DISK_GB, BACKLOG_S = 900, 120, 3.0, 7200
MAIN = ("mark", "book", "trade")


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
    bl = st.get("upload_backlog_age_s")
    if bl is not None and float(bl) > BACKLOG_S:
        why.append(f"uploads are behind: the oldest finished file has waited {float(bl) / 3600:.1f} h (upload_fail {st.get('upload_fail')})")
    if why:
        return 1, "⚠ DEGRADED", why
    return 0, "✅ HEALTHY", []


def _json(key):
    rc, out, _ = aws("s3", "cp", "--only-show-errors", f"s3://{BUCKET}/{key}", "-")
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def stored():
    rc, out, _ = aws("s3", "ls", "--recursive", "--summarize", f"s3://{BUCKET}/data/")
    if rc != 0:
        return None, None
    n = size = None
    for ln in out.splitlines():
        if "Total Objects:" in ln:
            n = int(ln.split(":")[1])
        if "Total Size:" in ln:
            size = int(ln.split(":")[1])
    return n, size


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
    n, size = stored()
    if size is not None:
        gb = size / 1e9
        L.append(f"- stored in S3: {n} files · {gb:.2f} GB (≈ ${gb * 0.023:.2f}/month at S3 Standard)")
    dl = daily_list()
    if dl:
        L.append(f"- daily summaries: {len(dl)} (latest {dl[-1]})")
        d = _json(f"health/daily/{dl[-1]}")
        if d:
            L.append(f"  - {d.get('day')}: messages {d.get('day_n')} · gap seconds {d.get('day_gap_s')} · unique hot coins {d.get('day_hot_unique')}")
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
