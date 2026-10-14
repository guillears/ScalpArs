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
"""
import csv
import glob
import os
import re
import sys

LIVE_START_UTC = "2026-10-14T18:36:30"          # [MODE_SWITCH] PAPER → LIVE (pre-flight passed), EB log 2026-10-14 18:36:30 UTC
LIVE_START_STAMP = "2026-10-14_18-36-30"        # same instant in the export filename format (UTC)
_KINDS = ("scalpars_orders_", "scalpars_decisions_")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTER_VERSION = 1                              # bump when the row filter changes → old cached copies are never reused
CACHE_DIR = os.path.join(_ROOT, "reports", ".live_exports", f"v{FILTER_VERSION}_{LIVE_START_STAMP}")
_STAMP = re.compile(r"_live_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


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
    out = glob.glob(pattern)
    base = os.path.basename(pattern)
    kind = next((k for k in _KINDS if base.startswith(k + "paper_")), None)
    if kind is None:
        return out
    live_pattern = os.path.join(os.path.dirname(pattern), base.replace(kind + "paper_", kind + "live_", 1))
    for f in sorted(glob.glob(live_pattern)):
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
