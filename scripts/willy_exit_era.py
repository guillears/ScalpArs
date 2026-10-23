"""🎲² WILLY exit-era classifier (DECISION_LOG 356) — shared by every scout that prices or gates FRENZY_WILLY / FRENZY_WILLY_HT fills.

Why: DECISION_LOG 356 (WILLY exit V2) changed the WILLY exit PER ORDER — the take profit is the order's stored spike-high target
(Order.frenzy_willy_tp_target_pct, ≥ +1.25) and an order with a runner target (> the floor arm, +1.25) gets a profit floor (close reason
WILLY_FLOOR). Scouts built before 356 walk / compare against the old fixed exit (+1.25 / −4 / late +0.25 @30 / 90). A V2 fill with a
runner target can book something that exit never could → a fake delta in a before/after or a parity line. Every WILLY scout therefore:
  (a) keeps the exit its frozen gate was registered with (pinned, never read from config),
  (b) recognises V2 fills by their STAMPS and either re-walks them under V2 or splits them out — never mixes them silently,
  (c) prints era_line(...) — "exit era: pre-V2 N · V2 N (runner target / floor N)" — so batch reviews cannot mix eras.

Classification (from the orders export alone; restart-proof, no config read):
  V2 era        = frenzy_willy_tp_target_pct is a finite number (stamped by the V2 code on every WILLY / HT fill, switch on or off)
  V2 changed    = V2 era AND (target > OLD_TP + EPS  OR  close reason WILLY_FLOOR) — the only fills whose exit can differ from the old one.
                  A V2 fill with target == +1.25 runs the old exit exactly (the floor needs a target above the arm) → safe in old cohorts.
Pure pandas / stdlib; every helper is total (never raises on a missing column)."""
import math

OLD_TP, OLD_STOP, OLD_HOLD, OLD_LATE = 1.25, 4.0, 90, (30.0, 0.25)   # the pre-356 live WILLY exit (DECISION_LOG 281 / 297) — PINNED
EPS = 1e-6
TP_COL, ARM_COL, HIGH_COL = "frenzy_willy_tp_target_pct", "willy_floor_armed_at", "entry_frenzy_willy_spike_high"
FLOOR_REASON = "WILLY_FLOOR"
V2_COLS = (HIGH_COL, TP_COL, ARM_COL)   # ask the export reader for these (usecols) so the era is never silently unknown


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def row_tp(row):
    """the stored V2 target of a row (dict / namedtuple / Series) or None (pre-V2 / missing)."""
    try:
        v = row.get(TP_COL) if hasattr(row, "get") else getattr(row, TP_COL, None)
    except Exception:
        v = None
    return _num(v)


def row_reason(row):
    try:
        v = row.get("close_reason") if hasattr(row, "get") else getattr(row, "close_reason", None)
    except Exception:
        v = None
    return str(v or "")


def is_v2(row) -> bool:
    return row_tp(row) is not None


def is_v2_changed(row) -> bool:
    """a V2 fill whose exit may differ from the old fixed exit (runner target, or a floor exit)."""
    t = row_tp(row)
    return t is not None and (t > OLD_TP + EPS or row_reason(row).endswith(FLOOR_REASON))


def masks(df):
    """→ (v2 mask, v2-changed mask) as boolean pandas Series aligned to df (all False when the column is missing)."""
    import pandas as pd
    if df is None or not len(df):
        e = pd.Series([], dtype=bool)
        return e, e
    tp = pd.to_numeric(df[TP_COL], errors="coerce") if TP_COL in df.columns else pd.Series(float("nan"), index=df.index)
    rs = df["close_reason"].astype(str) if "close_reason" in df.columns else pd.Series("", index=df.index)
    v2 = tp.notna()
    ch = v2 & ((tp > OLD_TP + EPS) | rs.str.endswith(FLOOR_REASON))
    return v2.fillna(False).astype(bool), ch.fillna(False).astype(bool)


def split(df):
    """→ (old-exit cohort = pre-V2 + V2 fills that ran the old exit exactly, V2-changed fills). Use the first for any comparison against
    the pinned old exit; report the second separately (or re-walk it under V2)."""
    if df is None or not len(df):
        return df, df
    _, ch = masks(df)
    return df[~ch], df[ch]


def counts(df):
    """→ (pre-V2 N, V2 N, V2-changed N)."""
    if df is None or not len(df):
        return 0, 0, 0
    v2, ch = masks(df)
    return int((~v2).sum()), int(v2.sum()), int(ch.sum())


def era_line(df, label="WILLY", note=None) -> str:
    """the one-line era header every WILLY scout prints (DECISION_LOG 356 (c))."""
    pre, v2, ch = counts(df)
    tail = f" — {note}" if note else (" — runner / floor fills are kept OUT of old-exit comparisons" if ch else "")
    return f"  🎲² exit era ({label}): pre-V2 {pre} · V2 {v2} (runner target / floor {ch}){tail}"
