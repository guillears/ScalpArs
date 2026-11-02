#!/usr/bin/env python3
"""🗄 (DECISION_LOG 436) map a code fingerprint (Order.code_version, BackupRun / ExchangeDay.code_version) to its git commit(s).

The fingerprint is services.track_record.code_fingerprint over the runtime source (top-level *.py, services/*.py, live_terminal/*.py,
templates/*.html) — this script hashes the same files of each commit's tree with the same rule.

Usage:
  venv/bin/python -I scripts/code_version.py                     # the working tree's fingerprint and HEAD's
  venv/bin/python -I scripts/code_version.py <fingerprint> [N]   # the commits (newest first, the last N, default 400) with that fingerprint
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.track_record import code_fingerprint, code_paths, code_version  # noqa: E402


def _git(*a) -> bytes:
    return subprocess.run(["git", "-C", ROOT, *a], check=True, capture_output=True).stdout


def commit_fingerprint(sha: str) -> str:
    names = _git("ls-tree", "-r", "--name-only", sha).decode().splitlines()
    return code_fingerprint(lambda p: _git("show", f"{sha}:{p}"), code_paths(names))


def main(argv):
    if len(argv) < 2:
        print(f"working tree {code_version(ROOT)} · HEAD {_git('rev-parse', '--short', 'HEAD').decode().strip()} {commit_fingerprint('HEAD')}")
        return 0
    want, n = argv[1].strip(), int(argv[2]) if len(argv) > 2 else 400
    hits = []
    for line in _git("log", f"-{n}", "--format=%H %cI %s").decode().splitlines():
        sha, when, subj = line.split(" ", 2)
        if commit_fingerprint(sha) == want:
            hits.append(f"{sha[:10]} {when} {subj[:110]}")
    print("\n".join(hits) if hits else f"no commit among the last {n} has fingerprint {want}")
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
