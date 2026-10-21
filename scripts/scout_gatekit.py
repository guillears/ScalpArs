"""🧰 Shared gate helpers for the Scout / tracker scripts (DECISION_LOG 331, 2026-10-21).

1. DEPLOY LOOKUP BY DECISION_LOG NUMBER — dl_commit_ts(n)
   A gate counts fills from the commit that shipped its DECISION_LOG entry. The scouts used to `git log --grep --fixed-strings` for the
   literal tag "(DECISION_LOG 263)" — but commits often ship several entries, "(DECISION_LOG 261, 262, 263)" (ec98aea), so the literal
   never matched and the gate silently fell back to NOW (GVOL_SPLIT counted 0 fills; the REENTRY / STRETCHED era notes were wrong).
   Here the number may appear ANYWHERE in a "DECISION_LOG a, b, c" / "DL a, b" list (comma / "&" / "and" / "+" separated). The FIRST
   (oldest) commit whose SUBJECT carries the number wins; only if no subject has it, the oldest commit whose full message has it.
   Not found → None, with a LOUD warning (warn_not_evaluable) — callers treat the gate as NOT EVALUABLE, never as "counted from now".

2. OPERATOR-CLOSED BOT FILLS — operator_closed_mask(df)
   A bot fill (entry_strategy ≠ MANUAL) that the OPERATOR closed — close_reason MANUAL (dashboard close; MANUAL_* = the paper-era
   MANUAL_CLOSE and any other manual-close reason) or EXTERNAL_CLOSE (closed in the Binance app) — does not carry the bot's exit, so it
   never counts in a step-up / revert / kill gate. Gates drop these rows with split_operator_closed() and print operator_closed_line()
   ("operator-closed: N — excluded") next to their tally. Operator-edited exits the BOT executed (OVERRIDE_TP / OVERRIDE_SL) are NOT
   operator closes and stay in. MANUAL entries (entry_strategy MANUAL) are never bot fills — the sleeves' own filters drop them.
"""
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════ 1. deploy lookup by DECISION_LOG number ═══════════════════════════════
_DL_LIST = re.compile(r"(?:\bDECISION_LOG|\bDL)\s*#?\s*(\d{1,4}(?:\s*(?:,|&|\+|\band\b|[\u2013\u2014-])\s*\d{1,4})*)")
_DL_TOK = re.compile(r"(\d{1,4})|([\u2013\u2014-])")
DL_RANGE_MAX = 50   # a 'DL a–b' range wider than this is not a range of decisions (read as two numbers)
_TAG_NUM = re.compile(r"(?:DECISION_LOG|\bDL)\s*#?\s*(\d{1,4})")
_LOG_CACHE = {}
_WARNED = set()


class GateNotEvaluable(Exception):
    """raised by a deploy lookup whose commit is not in git — the gate is NOT evaluable (never 'counted from now')."""


def parse_dl_numbers(text):
    """every DECISION_LOG number named in text: '(DECISION_LOG 261, 262, 263)' → [261, 262, 263]; 'DL 316 & 317' → [316, 317];
    'DL 329–336' / 'DL 329-336' (en / em dash or hyphen) → 329 … 336 (a range wider than DL_RANGE_MAX or running backwards is two numbers).
    A number after the first is part of the list only when it has at least as many digits as the first — a trailing small number is
    prose, not a decision ('(DECISION_LOG 281, 3 legs)' → [281]); same-magnitude lists behave as before."""
    out = []

    def add(v):
        if v not in out:
            out.append(v)
    for m in _DL_LIST.finditer(str(text or "")):
        toks = _DL_TOK.findall(m.group(1))
        first = toks[0][0]
        prev, dash = None, False
        for num, sep in toks:
            if sep:
                dash = True
                continue
            if prev is not None and len(num) < len(first):
                break                                    # prose after the list — stop absorbing
            v = int(num)
            if dash and prev is not None and prev < v <= prev + DL_RANGE_MAX:
                for x in range(prev + 1, v + 1):
                    add(x)
            else:
                add(v)
            prev, dash = v, False
    return out


def dl_from_tag(tag):
    """'(DECISION_LOG 263)' / 'grep:(DECISION_LOG 263)' / 'DL 263' / 263 → 263; anything else → None."""
    if isinstance(tag, int):
        return tag
    m = _TAG_NUM.search(str(tag or ""))
    return int(m.group(1)) if m else None


def git_commits(root=None):
    """[(sha, commit unix seconds, full message)] oldest first — memoised per root; git unavailable → []."""
    root = root or _ROOT
    if root not in _LOG_CACHE:
        try:
            out = subprocess.run(["git", "-C", root, "log", "--reverse", "--format=%H%x1f%ct%x1f%B%x1e"],
                                 capture_output=True, text=True, timeout=20)
            recs = []
            for rec in out.stdout.split("\x1e"):
                parts = rec.strip("\n").split("\x1f")
                if len(parts) >= 3 and parts[1].strip().isdigit():
                    recs.append((parts[0].strip(), int(parts[1]), parts[2]))
            _LOG_CACHE[root] = recs
        except Exception:
            _LOG_CACHE[root] = []
    return _LOG_CACHE[root]


def find_dl_commit(dl, commits=None, root=None):
    """(sha, unix seconds) of the deploy commit for DECISION_LOG dl — the oldest commit whose SUBJECT names it, else the oldest whose
    full message names it — or None."""
    dl = dl_from_tag(dl)
    if dl is None:
        return None
    commits = git_commits(root) if commits is None else commits
    for use_subject in (True, False):
        for sha, ct, msg in commits:
            txt = (msg.strip().splitlines() or [""])[0] if use_subject else msg
            if dl in parse_dl_numbers(txt):
                return sha, int(ct)
    return None


def dl_commit_ts(dl, commits=None, root=None):
    """commit unix seconds of the DECISION_LOG dl deploy, or None (not in git / git unavailable)."""
    hit = find_dl_commit(dl, commits, root)
    return None if hit is None else hit[1]


def warn_not_evaluable(label, dl=None, stream=None):
    """the LOUD warning for a gate whose deploy commit cannot be found (once per label per process)."""
    key = (str(label), dl)
    if key in _WARNED:
        return
    _WARNED.add(key)
    msg = (f"⚠⚠⚠ GATE NOT EVALUABLE — {label}: the deploy commit for DECISION_LOG {dl if dl is not None else '?'} was NOT found in git "
           f"(no commit message names it in a 'DECISION_LOG …' / 'DL …' list). Nothing is counted for this gate until it is found.")
    print(msg, file=stream or sys.stderr, flush=True)


def dl_deploy_ms(dl, lag_min=10, label=None, commits=None, root=None):
    """deploy floor in ms = the DECISION_LOG dl commit time + lag_min; not found → None after a LOUD warning (NOT evaluable)."""
    n = dl_from_tag(dl)
    ct = dl_commit_ts(n, commits, root)
    if ct is None:
        warn_not_evaluable(label or f"DECISION_LOG {n}", n)
        return None
    return int(ct) * 1000 + int(lag_min) * 60_000


def tag_commit_ts(tag, commits=None, root=None):
    """drop-in for the old `git log --grep=<tag> --fixed-strings` readers: a DL tag → its commit unix seconds via dl_commit_ts; a tag that
    names no DL number (self-test sentinels such as '(no such commit — selftest)') → None."""
    n = dl_from_tag(tag)
    return None if n is None else dl_commit_ts(n, commits, root)


# ═══════════════════════════════ 2. operator-closed bot fills ═══════════════════════════════
OPERATOR_CLOSE_EXACT = ("MANUAL", "EXTERNAL_CLOSE")
OPERATOR_CLOSE_PREFIX = ("MANUAL_",)


def _base_reason(cr):
    s = "" if cr is None else str(cr).strip().upper()
    if s in ("", "NAN", "NONE"):
        return ""
    return s.split()[0]                     # a ladder suffix (" L1") never changes who closed it


def is_operator_closed(entry_strategy, close_reason):
    """True for a BOT fill the operator closed: entry_strategy ≠ MANUAL ∧ close_reason MANUAL / MANUAL_* / EXTERNAL_CLOSE."""
    es = "" if entry_strategy is None else str(entry_strategy).strip().upper()
    if es == "MANUAL":
        return False
    return is_operator_close_reason(close_reason)


def is_operator_close_reason(close_reason):
    """the close_reason half alone (for frames already restricted to bot sleeves): MANUAL / MANUAL_* / EXTERNAL_CLOSE."""
    b = _base_reason(close_reason)
    return b in OPERATOR_CLOSE_EXACT or any(b.startswith(p) for p in OPERATOR_CLOSE_PREFIX)


def reason_mask(reasons):
    """numpy bool array over a close_reason sequence — True = an operator close (callers have already dropped MANUAL entries)."""
    import numpy as np
    return np.array([is_operator_close_reason(r) for r in reasons], dtype=bool)


def operator_closed_mask(df, strategy_col="entry_strategy", reason_col="close_reason"):
    """boolean pandas Series (df's index): True = an operator-closed bot fill. A frame without close_reason → all False."""
    import pandas as pd
    if df is None or not len(df) or reason_col not in df:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    es = df[strategy_col] if strategy_col in df else pd.Series("", index=df.index)
    return pd.Series([is_operator_closed(a, b) for a, b in zip(es, df[reason_col])], index=df.index, dtype=bool)


def split_operator_closed(df, strategy_col="entry_strategy", reason_col="close_reason"):
    """→ (kept rows, operator-closed rows) — the gate counts only the first."""
    m = operator_closed_mask(df, strategy_col, reason_col)
    if df is None or not len(df):
        return df, (df.iloc[0:0] if df is not None else df)
    return df[~m.values], df[m.values]


def operator_closed_line(excluded, cap=8, opened_col=None, prefix="operator-closed"):
    """'operator-closed: N — excluded (AUSDT 10-20 14:50 FRENZY_LITE_RETRY MANUAL · …)' — printed next to a gate's tally (N = 0 too)."""
    n = 0 if excluded is None else len(excluded)
    if not n:
        return f"{prefix}: 0 — excluded"
    oc = opened_col or next((c for c in ("_k", "opened_at", "o_ts", "key") if c in excluded), None)
    items = []
    for _, r in excluded.head(cap).iterrows():
        t = str(r.get(oc, ""))[:16].replace("T", " ")[5:] if oc else ""
        items.append(" ".join(x for x in (str(r.get("pair", "")), t, str(r.get("entry_strategy", "") or ""),
                                          str(r.get("close_reason", "") or "")) if x and x != "nan"))
    more = f" · +{n - cap} more" if n > cap else ""
    return f"{prefix}: {n} — excluded ({' · '.join(items)}{more})"
