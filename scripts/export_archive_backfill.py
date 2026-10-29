#!/usr/bin/env python3
"""🗄 One-off backfill of the permanent export archive (DECISION_LOG 409, operator 2026-10-29).

The scouts read the dashboard exports from ~/Downloads; the operator cleaned Downloads and every export it held left the scouts' view.
scripts/export_glob.py now also reads exports_archive/ (gitignored) and copies each new Downloads export into it. This script fills the gap
from what the project already holds, idempotent (a name already in the archive is never overwritten):
  ① every export still in ~/Downloads (export_glob.sync_archive)
  ② the live orders exports' filtered copies (reports/.live_exports/v*/scalpars_orders_live_*.csv — post-go-live live rows only)
  ③ each archived batch reports/BASELINE<N>_*_orders.csv, written under an export name the readers glob:
       paper batch (every row is_paper true)  → scalpars_orders_paper_<stamp>_BASELINE<N>.csv
       live batch  (every row is_paper false, opened ≥ the go-live) → scalpars_orders_live_<stamp>_BASELINE<N>.csv
     <stamp> = the batch's last opened / closed time + 1 min (UTC), also set as the file mtime (newest-export-wins readers stay ordered).
     A mixed batch, or a live batch from before the go-live (the August tests, kept out by export_glob since DL 294), is skipped.
Decisions journals need no backfill: the kept Downloads files (①) cover 2026-09-28 → today, and reports/BASELINE*_decisions.csv are
already read by the scouts that need them.
Usage: venv/bin/python -I scripts/export_archive_backfill.py [--dry-run]
"""
import calendar
import glob
import os
import re
import shutil
import sys

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as EG  # noqa: E402

ROOT = os.path.dirname(_D)
_TRUE = ("true", "1", "1.0", "yes")


def _copy(src, dst, mtime=None, dry=False):
    if os.path.exists(dst):
        return False
    if not dry:
        tmp = f"{dst}.{os.getpid()}.tmp"
        shutil.copy2(src, tmp)
        if mtime is not None:
            os.utime(tmp, (mtime, mtime))
        os.replace(tmp, dst)
    return True


def baseline_target(path):
    """→ (archive name, mtime) for one BASELINE orders file, or (None, why)."""
    m = re.search(r"BASELINE(\d+)_", os.path.basename(path))
    if not m:
        return None, "not a BASELINE<N> file"
    d = pd.read_csv(path, low_memory=False, usecols=lambda c: c in ("opened_at", "closed_at", "is_paper"))
    if not len(d) or "is_paper" not in d:
        return None, "empty or no is_paper column"
    ip = d.is_paper.astype(str).str.strip().str.lower()
    if d.is_paper.isna().any() or ip.isin(("", "nan", "none")).any():
        return None, "blank is_paper (mode unknown)"
    paper = ip.isin(_TRUE)
    ts = [pd.to_datetime(d[c], format="mixed", errors="coerce", utc=True) for c in ("opened_at", "closed_at") if c in d]
    if not ts:
        return None, "no timestamp columns"
    t = pd.concat(ts)
    end = t.max()
    if pd.isna(end):
        return None, "no timestamps"
    end = (end + pd.Timedelta(minutes=1)).floor("s")
    stamp = end.strftime("%Y-%m-%d_%H-%M-%S")
    if paper.all():
        mode = "paper"
    elif (~paper).all():
        first = pd.to_datetime(d.opened_at, format="mixed", errors="coerce", utc=True).min()
        if pd.isna(first) or first.strftime("%Y-%m-%dT%H:%M:%S") < EG.LIVE_START_UTC:
            return None, "live batch from before the go-live (August tests — kept out since DL 294)"
        mode = "live"
    else:
        return None, "mixed paper / live rows"
    return f"scalpars_orders_{mode}_{stamp}_BASELINE{m.group(1)}.csv", float(calendar.timegm(end.timetuple()))


def main(argv):
    dry = "--dry-run" in argv
    os.makedirs(EG.ARCHIVE_DIR, exist_ok=True)
    n1 = 0 if dry else EG.sync_archive()
    print(f"① Downloads → archive: {n1} new export(s){' (dry run: skipped)' if dry else ''}")
    n2 = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "reports", ".live_exports", "v*", "scalpars_orders_live_*.csv"))):
        n2 += _copy(f, os.path.join(EG.ARCHIVE_DIR, os.path.basename(f)), dry=dry)
    print(f"② live filtered copies → archive: {n2} new")
    for f in sorted(glob.glob(os.path.join(ROOT, "reports", "BASELINE*_orders.csv"))):
        name, x = baseline_target(f)
        if name is None:
            print(f"③ skip {os.path.basename(f)}: {x}")
            continue
        new = _copy(f, os.path.join(EG.ARCHIVE_DIR, name), mtime=x, dry=dry)
        print(f"③ {os.path.basename(f)} → {name} ({'new' if new else 'already archived'})")
    print(f"archive: {EG.ARCHIVE_DIR} · {len(os.listdir(EG.ARCHIVE_DIR))} file(s)")


if __name__ == "__main__":
    main(sys.argv)
