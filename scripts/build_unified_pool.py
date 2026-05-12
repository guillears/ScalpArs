#!/usr/bin/env python3
"""Build a unified deduped cross-batch CSV pool from all archived report snapshots
plus the latest live CSV.

Usage:
    python3 scripts/build_unified_pool.py
    python3 scripts/build_unified_pool.py --from-date 2026-05-04
    python3 scripts/build_unified_pool.py --output /tmp/pool.csv
    python3 scripts/build_unified_pool.py --quiet

DEDUP METHODOLOGY (see CLAUDE.md "Cross-batch CSV dedup methodology"):
- DO NOT dedupe by order `id` — IDs reset on paper-balance resets (autoincrement-from-1)
- DO dedupe by `(opened_at, pair, direction)` — microsecond timestamp + pair + direction
  is a stable cross-batch unique key

INPUTS:
- All CSVs in <repo>/reports/ matching orders_*.csv
- Latest scalpars_orders_paper_*.csv in ~/Downloads/ (most recent by mtime)

OUTPUT:
- Default: <repo>/reports/dedupe_pool.csv (deduped, CLOSED-only, opened_at >= from-date)
- Prints summary stats (N per direction, N per batch contribution, date range)
"""
from __future__ import annotations
import csv
import os
import glob
import argparse
import sys
from pathlib import Path
from collections import defaultdict


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FROM_DATE = "2026-05-04"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "dedupe_pool.csv"


def find_input_files() -> list[Path]:
    """Find all archived report CSVs + the latest live CSV from Downloads."""
    files: list[Path] = []
    # Archived report snapshots
    reports_dir = REPO_ROOT / "reports"
    files.extend(sorted(reports_dir.glob("orders_*.csv")))
    # Latest live CSV (most recent by mtime) from Downloads
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        live_csvs = sorted(
            downloads.glob("scalpars_orders_paper_*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if live_csvs:
            files.append(live_csvs[-1])  # most recent only
    return files


def build_pool(from_date: str, output_path: Path, quiet: bool = False) -> dict:
    """Build the unified deduped pool. Returns summary stats dict."""
    input_files = find_input_files()
    if not input_files:
        print("ERROR: no input CSVs found", file=sys.stderr)
        sys.exit(1)

    seen: set[tuple] = set()
    all_rows: list[dict] = []
    fieldnames: list[str] | None = None
    per_file_added: dict[str, int] = {}

    for path in input_files:
        added = 0
        try:
            with path.open() as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    if row.get("status") != "CLOSED":
                        continue
                    opened = (row.get("opened_at") or "").strip()
                    if not opened:
                        continue  # defensive: skip rows with no timestamp
                    if opened < from_date:
                        continue
                    pair = (row.get("pair") or "").strip()
                    direction = (row.get("direction") or "").strip()
                    if not pair or not direction:
                        continue  # defensive
                    key = (opened, pair, direction)
                    if key in seen:
                        continue
                    seen.add(key)
                    all_rows.append(row)
                    added += 1
        except (FileNotFoundError, PermissionError) as e:
            if not quiet:
                print(f"  WARN: skipping {path.name}: {e}", file=sys.stderr)
            continue
        per_file_added[path.name] = added

    # Sort by opened_at ascending for chronological CSV
    all_rows.sort(key=lambda r: r.get("opened_at", ""))

    # Compute stats
    n_long = sum(1 for r in all_rows if r.get("direction") == "LONG")
    n_short = sum(1 for r in all_rows if r.get("direction") == "SHORT")
    date_min = all_rows[0].get("opened_at", "")[:10] if all_rows else ""
    date_max = all_rows[-1].get("opened_at", "")[:10] if all_rows else ""

    # Per-date breakdown
    by_date: dict[str, dict[str, int]] = defaultdict(lambda: {"L": 0, "S": 0})
    for r in all_rows:
        d = (r.get("opened_at") or "")[:10]
        if r.get("direction") == "LONG":
            by_date[d]["L"] += 1
        elif r.get("direction") == "SHORT":
            by_date[d]["S"] += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    stats = {
        "total": len(all_rows),
        "long": n_long,
        "short": n_short,
        "date_min": date_min,
        "date_max": date_max,
        "per_file": per_file_added,
        "by_date": dict(by_date),
        "output": str(output_path),
        "files_processed": len(input_files),
    }

    if not quiet:
        print(f"=== Unified Cross-Batch Pool (deduped by opened_at+pair+direction) ===\n")
        print(f"Filter: status='CLOSED', opened_at >= {from_date}")
        print(f"Date range: {date_min} → {date_max}")
        print(f"Total: {len(all_rows)} trades ({n_long}L / {n_short}S)\n")
        print("=== Per source file (new trades added after dedup) ===")
        for fname in sorted(per_file_added.keys()):
            n = per_file_added[fname]
            print(f"  {n:>4} new trades from {fname}")
        print(f"\n=== Per date breakdown ===")
        for d in sorted(by_date.keys()):
            e = by_date[d]
            print(f"  {d}: {e['L']}L / {e['S']}S  (total {e['L'] + e['S']})")
        print(f"\nWritten to: {output_path}")
        print(f"\nIMPORTANT: For cross-batch comparisons, ALWAYS use Avg P&L %")
        print(f"(leverage-invariant per CLAUDE.md core principle). Raw $ values mix")
        print(f"different leverages/configs and are NOT directly comparable across batches.")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build unified deduped CSV pool")
    parser.add_argument("--from-date", default=DEFAULT_FROM_DATE,
                        help=f"Earliest opened_at to include (YYYY-MM-DD, default {DEFAULT_FROM_DATE})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Output CSV path (default {DEFAULT_OUTPUT})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stdout summary")
    args = parser.parse_args()

    build_pool(
        from_date=args.from_date,
        output_path=Path(args.output),
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
