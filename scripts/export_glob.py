"""📦 Export finder for the Scout / tracker scripts — PAPER and LIVE exports (DECISION_LOG 294).

The dashboard names its downloads by the bot's mode: scalpars_orders_{paper|live}_<UTC stamp>.csv and
scalpars_decisions_{paper|live}_<UTC stamp>.csv. The trackers were written for the paper bot and globbed "_paper_" only — after the
2026-10-14 go-live every live trade would have been invisible to them. glob_exports(pattern) is a drop-in for glob.glob(pattern):

- a pattern without "scalpars_orders_paper_" / "scalpars_decisions_paper_" → plain glob.glob (self-test temp dirs etc. unchanged);
- otherwise → the paper matches PLUS the live exports of the same kind in the same directory that were downloaded at/after the
  go-live (filename stamp ≥ LIVE_START_STAMP). The August 2026 live-test exports stay out.
- ORDERS live exports are returned as a filtered copy under reports/.live_exports/ (same basename and mtime) holding only rows with
  is_paper false and opened_at ≥ LIVE_START_UTC — a live export also carries every older live row (the August tests).
- DECISIONS exports are the mode-agnostic decision journal (same content whichever mode named the file) → returned as they are.

Rows keep their is_paper column, so any table can split paper vs live. Never raises: an unreadable live file is skipped.

🗄 PERMANENT ARCHIVE (DECISION_LOG 409, operator 2026-10-29 — Downloads was cleaned and the scouts lost every export it held): a pattern in
~/Downloads ALSO reads exports_archive/ in the project (gitignored, local only). Each call first copies any dashboard export in ~/Downloads
(scalpars_orders_* / scalpars_decisions_*, paper or live) that the archive does not hold yet (same name, mtime kept), so an export survives
being deleted from Downloads. A name in both places is read once (the Downloads copy). Patterns outside ~/Downloads (self-test temp dirs,
reports/) are unchanged. Gaps from before the archive existed were backfilled from reports/BASELINE*_orders.csv and the live filtered
copies (scripts/export_archive_backfill.py) — note: those live copies are already filtered to post-go-live live rows under the raw export
names (a no-filter reader of scalpars_orders_*.csv gets only those rows from them — the August test rows are never wanted).
Removing an export on purpose (a bad / test file): delete it from BOTH ~/Downloads and exports_archive/ — the archive never forgets.
"""
import calendar
import csv
import glob
import os
import re
import shutil
import sys
import time

LIVE_START_UTC = "2026-10-14T18:36:30"          # [MODE_SWITCH] PAPER → LIVE (pre-flight passed), EB log 2026-10-14 18:36:30 UTC
LIVE_START_STAMP = "2026-10-14_18-36-30"        # same instant in the export filename format (UTC)
_KINDS = ("scalpars_orders_", "scalpars_decisions_")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTER_VERSION = 1                              # bump when the row filter changes → old cached copies are never reused
CACHE_DIR = os.path.join(_ROOT, "reports", ".live_exports", f"v{FILTER_VERSION}_{LIVE_START_STAMP}")
_STAMP = re.compile(r"_live_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
DOWNLOADS = os.path.expanduser("~/Downloads")
ARCHIVE_DIR = os.path.join(_ROOT, "exports_archive")
_EXPORT_NAME = re.compile(r"^scalpars_(?:orders|decisions)_(?:paper|live)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.*\.csv$")
_ACCOUNTING_NAME = re.compile(r"^scalpars_accounting_\d{4}-\d{2}\.xlsx$")   # 📒 (DECISION_LOG 457) monthly statements → exports_archive/accounting/
# 📒 (DECISION_LOG 463) the 📒 button's From / To files — live "scalpars_accounting_<from>_<to>.xlsx" and paper "scalpars_PAPER_check_…" (a browser
# " (n)" copy kept under its own name) — archived beside the monthly statements
_ACCOUNTING_RANGE = re.compile(r"^scalpars_(?:accounting|PAPER_check)_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}(?: \(\d+\))?\.xlsx$")
ACCOUNTING_SUBDIR = "accounting"
# 🧹 (DECISION_LOG 463, operator 2026-11-04 "leave only the last 2 days") after the copy pass on ~/Downloads: an orders / decisions export
# whose filename stamp is older than KEEP_DAYS, and every accounting file, goes to the Trash — ONLY when its archive copy is verified
# (same size, same whole-second mtime). The archive is the permanent store (DL 409); nothing is ever deleted outright.
KEEP_DAYS = 2
ACCOUNTING_MIN_AGE_S = 86400                    # an accounting file stays in Downloads for a day (time to e-mail it) before it moves
TRASH = os.path.expanduser("~/.Trash")
_TRASH_FAILED = set()                           # a refused move (no Full Disk Access) is reported once per process, then retried quietly
_STAMP_ANY = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")


def _is_accounting(f):
    return bool(_ACCOUNTING_NAME.match(f) or _ACCOUNTING_RANGE.match(f))


SYNC_TTL_S = 60.0
_LAST_SYNC = {}


def sync_archive(src=None, dst=None, force=False, tidy=None, trash=None, now=None):
    """copy every dashboard export in src (default ~/Downloads) that dst (default exports_archive/) does not hold yet — or holds with a
    different size / mtime (a truncated earlier copy is repaired) → number copied. Once per SYNC_TTL_S per (src, dst) unless force.
    Stale *.tmp files (> 1 h, a killed copy) are removed. Never raises (a failed copy is reported on stderr and retried next call).
    tidy (default: only when src is the real ~/Downloads) → then tidy_downloads(src, dst, trash, now)."""
    if tidy is None:
        try:
            tidy = src is None or os.path.realpath(src) == os.path.realpath(DOWNLOADS)
        except OSError:
            tidy = False
    src, dst = src or DOWNLOADS, dst or ARCHIVE_DIR
    now = time.time() if now is None else float(now)
    if not force and now - _LAST_SYNC.get((src, dst), -1e18) < SYNC_TTL_S:
        return 0
    _LAST_SYNC[(src, dst)] = now
    n = 0
    try:
        names = [f for f in os.listdir(src) if _EXPORT_NAME.match(f) or _is_accounting(f)]
    except OSError:
        return 0
    try:
        os.makedirs(dst, exist_ok=True)
    except OSError as e:
        print(f"[export_glob] archive folder unavailable ({e!r})", file=sys.stderr)
        return 0
    try:
        for t in os.listdir(dst):
            if t.endswith(".tmp") and now - os.path.getmtime(os.path.join(dst, t)) > 3600:
                os.remove(os.path.join(dst, t))
    except OSError:
        pass
    for f in names:
        sub = os.path.join(dst, ACCOUNTING_SUBDIR) if _is_accounting(f) else dst
        if sub != dst:
            try:
                os.makedirs(sub, exist_ok=True)
            except OSError as e:
                print(f"[export_glob] accounting archive folder unavailable ({e!r})", file=sys.stderr)
                continue
        a, b = os.path.join(src, f), os.path.join(sub, f)
        try:
            sa = os.stat(a)
            if os.path.exists(b):
                sb = os.stat(b)
                if sb.st_size == sa.st_size and int(sb.st_mtime) == int(sa.st_mtime):
                    continue
        except OSError:
            continue
        tmp = f"{b}.{os.getpid()}.tmp"
        try:
            shutil.copy2(a, tmp)
            os.replace(tmp, b)
            n += 1
        except OSError as e:
            print(f"[export_glob] could not archive {f} ({e!r})", file=sys.stderr)
            try:
                os.remove(tmp)
            except OSError:
                pass
    if tidy:
        tidy_downloads(src, dst, trash, now)
    return n


def _same(a, b):
    """b is a verified copy of a: both exist, same size, same whole-second mtime."""
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)


def _to_trash(path, trash):
    """move path into the trash folder (a name already there gets a ' <unix time>' suffix) → True when moved. Never raises."""
    try:
        os.makedirs(trash, exist_ok=True)
        base = os.path.basename(path)
        target = os.path.join(trash, base)
        if os.path.exists(target):
            stem, ext = os.path.splitext(base)
            target = os.path.join(trash, f"{stem} {int(time.time() * 1000)}{ext}")
        shutil.move(path, target)
        return True
    except OSError as e:
        if path not in _TRASH_FAILED:
            _TRASH_FAILED.add(path)
            print(f"[export_glob] could not move {os.path.basename(path)} to the Trash ({e!r})", file=sys.stderr)
        return False


def tidy_downloads(src=None, dst=None, trash=None, now=None, keep_days=None):
    """🧹 (DECISION_LOG 463) src (default ~/Downloads) → the Trash: orders / decisions exports stamped more than keep_days (KEEP_DAYS) before
    now, and every accounting file older than ACCOUNTING_MIN_AGE_S (by mtime) — each ONLY when its archive copy in dst is verified (_same). An unstamped / unverified file stays.
    → names moved. Never raises."""
    src, dst = src or DOWNLOADS, dst or ARCHIVE_DIR
    trash = trash or TRASH
    now = time.time() if now is None else float(now)
    keep = KEEP_DAYS if keep_days is None else keep_days
    moved = []
    try:
        names = sorted(os.listdir(src))
    except OSError:
        return moved
    for f in names:
        if _is_accounting(f):
            try:
                if now - os.path.getmtime(os.path.join(src, f)) < ACCOUNTING_MIN_AGE_S:
                    continue
            except OSError:
                continue
            b = os.path.join(dst, ACCOUNTING_SUBDIR, f)
        elif _EXPORT_NAME.match(f):
            m = _STAMP_ANY.search(f)
            if not m:
                continue
            try:
                ts = calendar.timegm(time.strptime(f"{m[1]} {m[2]}:{m[3]}:{m[4]}", "%Y-%m-%d %H:%M:%S"))   # filename stamp = UTC
            except (ValueError, OverflowError):
                continue
            if now - ts <= keep * 86400:
                continue
            b = os.path.join(dst, f)
        else:
            continue
        a = os.path.join(src, f)
        if _same(a, b) and _to_trash(a, trash):
            moved.append(f)
    return moved


def _with_archive(pattern):
    """glob.glob(pattern) + the same file pattern in the archive when pattern is in ~/Downloads (synced first); one path per name."""
    d = os.path.dirname(pattern)
    try:
        in_dl = os.path.realpath(d) == os.path.realpath(DOWNLOADS)
    except OSError:
        in_dl = False
    if not in_dl:
        return glob.glob(pattern)
    sync_archive()                                       # (463 review) BEFORE listing: the tidy may move a Downloads file to the Trash
    out = [p for p in glob.glob(pattern) if os.path.exists(p)]
    seen = {os.path.basename(p) for p in out}
    out += [p for p in glob.glob(os.path.join(ARCHIVE_DIR, os.path.basename(pattern))) if os.path.basename(p) not in seen]
    return out


def _norm_ts(v):
    s = str(v or "").strip().replace(" ", "T")
    return s[:19]


def _is_true(v):
    """Only an explicit true drops a row (a live export holds is_paper 'False'; blank / nan = unknown → kept)."""
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def is_live_export(path) -> bool:
    return "_live_" in os.path.basename(str(path))


def _live_stamp_ok(path) -> bool:
    m = _STAMP.search(os.path.basename(path))
    return bool(m) and m.group(1) >= LIVE_START_STAMP


def _filtered_orders_copy(src, cache_dir=None):
    """The live orders export reduced to post-go-live live rows → cached path (rebuilt when the source is newer). None on failure."""
    cache_dir = cache_dir or CACHE_DIR
    try:
        os.makedirs(cache_dir, exist_ok=True)
        dst = os.path.join(cache_dir, os.path.basename(src))
        st = os.stat(src)
        if os.path.exists(dst) and os.path.getmtime(dst) == st.st_mtime:
            return dst
        tmp = f"{dst}.{os.getpid()}.tmp"   # unique per process: a manual tracker run beside the scout never interleaves
        with open(src, newline="", encoding="utf-8") as fi, open(tmp, "w", newline="", encoding="utf-8") as fo:
            r = csv.reader(fi)
            head = next(r)
            i_open = head.index("opened_at")
            i_paper = head.index("is_paper") if "is_paper" in head else None
            w = csv.writer(fo)
            w.writerow(head)
            for row in r:
                if len(row) <= i_open:
                    continue
                if i_paper is not None and len(row) > i_paper and _is_true(row[i_paper]):
                    continue
                if _norm_ts(row[i_open]) >= LIVE_START_UTC:
                    w.writerow(row)
        os.replace(tmp, dst)
        os.utime(dst, (st.st_atime, st.st_mtime))   # newest-export-wins logic keyed on mtime keeps working
        return dst
    except Exception as e:   # never raises — but never silently: a dropped live export would hide every live fill
        print(f"[export_glob] live export SKIPPED (could not filter {os.path.basename(str(src))}: {e!r})", file=sys.stderr)
        return None


def glob_exports(pattern, cache_dir=None):
    """Drop-in for glob.glob(pattern) that adds the post-go-live LIVE exports (see the module docstring)."""
    pattern = os.path.expanduser(str(pattern))
    out = _with_archive(pattern)
    base = os.path.basename(pattern)
    kind = next((k for k in _KINDS if base.startswith(k + "paper_")), None)
    if kind is None:
        return out
    live_pattern = os.path.join(os.path.dirname(pattern), base.replace(kind + "paper_", kind + "live_", 1))
    for f in sorted(_with_archive(live_pattern)):
        if not _live_stamp_ok(f):
            continue
        if kind == "scalpars_orders_":
            c = _filtered_orders_copy(f, cache_dir)
            if c:
                out.append(c)
        else:
            out.append(f)
    return out


def export_stamp(path):
    """The UTC download instant in an export's filename → 'YYYY-MM-DDTHH:MM:SS' (paper or live; None if absent)."""
    m = re.search(r"_(?:paper|live)_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", os.path.basename(str(path)))
    return f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}" if m else None


def newest_per_mode(paths, key=os.path.getmtime):
    """For readers that use only the NEWEST export: the newest paper file and the newest live file (each export holds one mode's
    rows only, so the newest of all would drop the other mode). A paper export downloaded before the go-live is dropped once a live
    export exists: the paper bot has stopped, so it would pin its last state forever. → list of 0–2 paths, oldest first."""
    best = {}
    for p in paths or ():
        m = "live" if is_live_export(p) else "paper"
        if m not in best or key(p) > key(best[m]):
            best[m] = p
    if "live" in best and "paper" in best and (export_stamp(best["paper"]) or "9") < LIVE_START_UTC:
        best.pop("paper")
    return sorted(best.values(), key=key)
