"""📦 (DECISION_LOG 294) the Scout / tracker scripts read PAPER and post-go-live LIVE exports: live orders rows are filtered to
is_paper false and opened_at ≥ the go-live, pre-go-live live exports (the August tests) are ignored, decisions exports pass as they
are, and readers that take only the newest export get the newest of EACH mode."""
import csv, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))

import export_glob as EG

HEAD = ["id", "pair", "direction", "opened_at", "status", "is_paper"]


def _w(path, rows, mtime=None):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEAD); w.writerows(rows)
    if mtime:
        os.utime(path, (mtime, mtime))


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_paper_plus_filtered_live_orders(tmp_path):
    dl, cache = tmp_path / "dl", tmp_path / "cache"; dl.mkdir()
    _w(dl / "scalpars_orders_paper_2026-10-14_17-24-40.csv", [[1, "AUSDT", "LONG", "2026-10-14T15:55:08", "CLOSED", "True"]], 1000)
    _w(dl / "scalpars_orders_live_2026-08-26_14-16-33.csv", [[2, "BUSDT", "LONG", "2026-08-26T10:00:00", "CLOSED", "False"]], 900)
    live = dl / "scalpars_orders_live_2026-10-15_10-00-00.csv"
    _w(live, [[2, "BUSDT", "LONG", "2026-08-26T10:00:00", "CLOSED", "False"],          # August live test → out
              [3, "CUSDT", "SHORT", "2026-10-14 18:40:00.5", "CLOSED", "False"],        # after the go-live → kept
              [4, "DUSDT", "LONG", "2026-10-14T18:36:29", "CLOSED", "False"],           # one second before → out
              [5, "EUSDT", "LONG", "2026-10-14T19:00:00", "CLOSED", "True"]], 2000)     # paper row (never in a live export) → out
    out = EG.glob_exports(str(dl / "scalpars_orders_paper_*.csv"), cache_dir=str(cache))
    names = sorted(os.path.basename(p) for p in out)
    assert names == ["scalpars_orders_live_2026-10-15_10-00-00.csv", "scalpars_orders_paper_2026-10-14_17-24-40.csv"]
    cp = next(p for p in out if EG.is_live_export(p))
    assert os.path.dirname(cp) == str(cache) and os.path.getmtime(cp) == 2000                # cached copy keeps the source mtime
    assert [r["pair"] for r in _read(cp)] == ["CUSDT"]
    assert os.path.exists(live) and len(_read(live)) == 4                                    # the operator's download is never touched
    # rebuilt when the export is replaced by a newer one with more rows
    _w(live, [[3, "CUSDT", "SHORT", "2026-10-14T18:40:00", "CLOSED", "False"], [6, "FUSDT", "LONG", "2026-10-15T09:00:00", "CLOSED", "False"]], 3000)
    cp2 = next(p for p in EG.glob_exports(str(dl / "scalpars_orders_paper_*.csv"), cache_dir=str(cache)) if EG.is_live_export(p))
    assert [r["pair"] for r in _read(cp2)] == ["CUSDT", "FUSDT"]
    # an unreadable live export is skipped LOUDLY, never silently
    bad = dl / "scalpars_orders_live_2026-10-16_10-00-00.csv"; bad.write_text("id,pair\n1,X\n")   # no opened_at column
    import io, contextlib
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        out3 = EG.glob_exports(str(dl / "scalpars_orders_paper_*.csv"), cache_dir=str(cache))
    assert not any(os.path.basename(p) == bad.name for p in out3) and "SKIPPED" in err.getvalue()


def test_decisions_pass_through_and_other_patterns_unchanged(tmp_path):
    (tmp_path / "scalpars_decisions_paper_2026-10-14_17-24-45.csv").write_text("t,e\n")
    (tmp_path / "scalpars_decisions_live_2026-10-14_20-17-01.csv").write_text("t,e\n")
    (tmp_path / "scalpars_decisions_live_2026-08-25_15-51-56.csv").write_text("t,e\n")      # pre-go-live → out
    (tmp_path / "none_1.csv").write_text("x\n")
    out = sorted(os.path.basename(p) for p in EG.glob_exports(str(tmp_path / "scalpars_decisions_paper_*.csv")))
    assert out == ["scalpars_decisions_live_2026-10-14_20-17-01.csv", "scalpars_decisions_paper_2026-10-14_17-24-45.csv"]
    assert [os.path.basename(p) for p in EG.glob_exports(str(tmp_path / "none_*.csv"))] == ["none_1.csv"]
    assert EG.glob_exports(str(tmp_path / "scalpars_orders_paper_*.csv")) == []


def test_newest_per_mode_and_stamp(tmp_path):
    fs = []
    for name, mt in (("scalpars_orders_paper_2026-10-13_10-00-00.csv", 10), ("scalpars_orders_paper_2026-10-14_19-00-00.csv", 20),
                     ("scalpars_orders_live_2026-10-14_20-00-00.csv", 30), ("scalpars_orders_live_2026-10-15_20-00-00.csv", 40)):
        p = tmp_path / name; p.write_text("x\n"); os.utime(p, (mt, mt)); fs.append(str(p))
    assert [os.path.basename(p) for p in EG.newest_per_mode(fs)] == ["scalpars_orders_paper_2026-10-14_19-00-00.csv",
                                                                     "scalpars_orders_live_2026-10-15_20-00-00.csv"]
    assert EG.export_stamp(fs[2]) == "2026-10-14T20:00:00" and EG.export_stamp(fs[0]) == "2026-10-13T10:00:00"
    assert EG.export_stamp("x.csv") is None and EG.newest_per_mode([]) == []
    # a paper export from BEFORE the go-live is dropped once a live export exists (the stopped paper bot would pin its last state)
    assert [os.path.basename(p) for p in EG.newest_per_mode([fs[0], fs[3]])] == ["scalpars_orders_live_2026-10-15_20-00-00.csv"]
    assert [os.path.basename(p) for p in EG.newest_per_mode(fs[:1])] == ["scalpars_orders_paper_2026-10-13_10-00-00.csv"]


def test_every_tracker_reads_live_exports():
    """No Scout / tracker glob of the dashboard exports is left paper-only."""
    import glob, re
    files = [os.path.join(ROOT, "scripts", n) for n in ("opportunity_scout.py", "ms_pvr_shadow.py", "ml_watchlist_adjudicate.py")]
    files = [f for f in files if os.path.exists(f)] + glob.glob(os.path.join(ROOT, "scripts", "scout_*.py"))   # some tools are local-only
    bad = []
    for f in files:
        for i, ln in enumerate(open(f, encoding="utf-8").read().split("\n"), 1):
            if re.search(r"\bglob\.glob\(", ln) and ("_paper_" in ln or "EXPORT_GLOB" in ln or "DEC_GLOB" in ln):
                bad.append(f"{os.path.basename(f)}:{i}")
            if 'rsplit("_paper_"' in ln or 'len("scalpars_decisions_paper_")' in ln:
                bad.append(f"{os.path.basename(f)}:{i} (paper-only filename parse)")
    assert bad == []


# ── 🗄 permanent export archive (DECISION_LOG 409) ────────────────────────────────────────────────────────────────────────────────
def test_downloads_exports_are_archived_and_survive_deletion(tmp_path, monkeypatch):
    """a Downloads pattern copies new exports into the archive (mtime kept) and keeps reading them after they leave Downloads; a name in
    both places is read once (the Downloads copy); a non-export file is never archived."""
    dl, ar, cache = tmp_path / "dl", tmp_path / "archive", tmp_path / "cache"; dl.mkdir()
    monkeypatch.setattr(EG, "DOWNLOADS", str(dl)); monkeypatch.setattr(EG, "ARCHIVE_DIR", str(ar)); monkeypatch.setattr(EG, "SYNC_TTL_S", 0.0)
    p1 = dl / "scalpars_orders_paper_2026-10-20_10-00-00.csv"
    _w(p1, [[1, "AUSDT", "LONG", "2026-10-20T09:00:00", "CLOSED", "True"]], 1000)
    (dl / "notes.csv").write_text("x\n")
    out = EG.glob_exports(str(dl / "scalpars_orders_paper_*.csv"), cache_dir=str(cache))
    assert out == [str(p1)]                                                             # once, from Downloads
    assert sorted(os.listdir(ar)) == ["scalpars_orders_paper_2026-10-20_10-00-00.csv"]  # archived, the non-export file left alone
    assert os.path.getmtime(ar / p1.name) == 1000
    p1.unlink()
    assert EG.glob_exports(str(dl / "scalpars_orders_paper_*.csv"), cache_dir=str(cache)) == [str(ar / p1.name)]   # survives deletion
    assert EG.sync_archive() == 0                                                       # nothing new → no copy


def test_archive_live_and_decisions_and_other_dirs(tmp_path, monkeypatch):
    """archived LIVE orders still go through the live filter; decisions (paper or live) are archived too; a pattern outside Downloads
    never reads or writes the archive (self-test temp dirs stay hermetic)."""
    dl, ar, cache, other = tmp_path / "dl", tmp_path / "archive", tmp_path / "cache", tmp_path / "other"
    dl.mkdir(); other.mkdir()
    monkeypatch.setattr(EG, "DOWNLOADS", str(dl)); monkeypatch.setattr(EG, "ARCHIVE_DIR", str(ar)); monkeypatch.setattr(EG, "SYNC_TTL_S", 0.0)
    ar.mkdir()
    _w(ar / "scalpars_orders_live_2026-10-16_12-15-07_BASELINE22.csv",
       [[2, "BUSDT", "LONG", "2026-08-26T10:00:00", "CLOSED", "False"], [3, "CUSDT", "LONG", "2026-10-15T10:00:00", "CLOSED", "False"]], 900)
    out = EG.glob_exports(str(dl / "scalpars_orders_paper_*.csv"), cache_dir=str(cache))
    assert len(out) == 1 and [r["pair"] for r in _read(out[0])] == ["CUSDT"]           # the live filter still applies
    (dl / "scalpars_decisions_live_2026-10-29_12-01-24.csv").write_text("t,e\n")
    assert EG.glob_exports(str(dl / "scalpars_decisions_*.csv")) == [str(dl / "scalpars_decisions_live_2026-10-29_12-01-24.csv")]
    assert (ar / "scalpars_decisions_live_2026-10-29_12-01-24.csv").exists()
    _w(other / "scalpars_orders_paper_2026-10-21_10-00-00.csv", [[4, "DUSDT", "LONG", "2026-10-21T09:00:00", "CLOSED", "True"]])
    n_ar = len(os.listdir(ar))
    assert EG.glob_exports(str(other / "scalpars_orders_paper_*.csv"), cache_dir=str(cache)) == [str(other / "scalpars_orders_paper_2026-10-21_10-00-00.csv")]
    assert len(os.listdir(ar)) == n_ar                                                  # nothing archived from a non-Downloads dir


def test_backfill_names_baselines_as_exports(tmp_path):
    """export_archive_backfill: a paper batch → a paper export name stamped after its last fill (mtime = the stamp); a post-go-live live
    batch → a live name; a pre-go-live live batch (August tests) and a mixed batch are skipped."""
    import export_archive_backfill as BF
    def bl(name, rows):
        p = tmp_path / name
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["opened_at", "closed_at", "is_paper"]); w.writerows(rows)
        return str(p)
    n, mt = BF.baseline_target(bl("BASELINE19_batch1008-1012_orders.csv", [["2026-10-08T21:40:00", "2026-10-12T00:58:28", "True"]]))
    assert n == "scalpars_orders_paper_2026-10-12_00-59-28_BASELINE19.csv" and mt == 1791766768.0
    assert EG.export_stamp(n) == "2026-10-12T00:59:28"
    n, _ = BF.baseline_target(bl("BASELINE22_batch_LIVE_orders.csv", [["2026-10-14T21:40:00", "2026-10-16T12:14:07", "False"]]))
    assert n == "scalpars_orders_live_2026-10-16_12-15-07_BASELINE22.csv" and EG._live_stamp_ok(n)
    assert BF.baseline_target(bl("BASELINE5_batch_orders.csv", [["2026-08-26T14:32:00", "2026-08-27T16:10:00", "False"]]))[0] is None
    assert BF.baseline_target(bl("BASELINE7_batch_orders.csv", [["2026-09-01T00:00:00", "", "True"], ["2026-09-01T01:00:00", "", "False"]]))[0] is None


def test_scout_readers_use_the_archive_aware_finder():
    """(409) every scout-side reader of the Downloads exports goes through export_glob (so exports_archive/ is read)."""
    import re
    for fn in ("sweep_separators.py", "path_labels.py", "scout_flatbase_short.py", "scout_willy_nomiss.py"):
        src = open(os.path.join(ROOT, "scripts", fn), encoding="utf-8").read()
        assert not re.search(r"glob\.glob\([^)]*(Downloads|DEC_PATTERNS|pat\b)", src), fn
    assert "exports_archive/" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_sync_repairs_a_truncated_copy_throttles_and_cleans_tmp(tmp_path, monkeypatch):
    """a copy whose size / mtime differs from Downloads is re-copied; within the TTL no second pass runs; a stale *.tmp is removed."""
    import time
    dl, ar = tmp_path / "dl", tmp_path / "archive"; dl.mkdir(); ar.mkdir()
    f = "scalpars_decisions_paper_2026-10-29_12-01-24.csv"
    (dl / f).write_text("t,e\n1,2\n"); os.utime(dl / f, (1000, 1000))
    (ar / f).write_text("t,e\n"); os.utime(ar / f, (1000, 1000))                     # truncated earlier copy
    old = ar / "x.csv.1.tmp"; old.write_text(""); os.utime(old, (time.time() - 7200,) * 2)
    assert EG.sync_archive(str(dl), str(ar), force=True) == 1 and (ar / f).read_text() == "t,e\n1,2\n" and not old.exists()
    assert EG.sync_archive(str(dl), str(ar), force=True) == 0                         # identical now
    os.utime(dl / f, (2000, 2000))
    monkeypatch.setattr(EG, "SYNC_TTL_S", 3600.0); EG._LAST_SYNC.clear()
    assert EG.sync_archive(str(dl), str(ar)) == 1 and EG.sync_archive(str(dl), str(ar)) == 0   # second call throttled
