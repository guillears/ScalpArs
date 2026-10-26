#!/usr/bin/env python3
"""Load engine-replay fills (default prefix yr5) TRIMMED to each chunk's own window — the one correct way to read them.

Why (reports/YR5_REREAD_WARMUP_TRIM_2026-10-06.md): every chunk run yr5_<chunk>_s<seed> also saves the fills it opened
during its 3-day WARM-UP (meta warm_ms → start_ms). Those repeat time the previous chunk already covers (a second, different
realisation of the same days) — 1,504 of 9,365 yr5 fills (16 %). Keep only opened_at in [start_ms, end_ms) of the chunk's
<tag>_meta.json. Also: the `pnl` column is each chunk's OWN $5k book (compounding inside the chunk) — never sum it across
chunks or seeds. Use per-trade % (pnl_percentage, fee-net, leverage-invariant) and, for dollars, `fixed_book_usd()`.

Usage (library):
    import sys; sys.path.insert(0, "scripts"); import yr5_fills_trimmed as YT
    F = YT.load()                                  # every sleeve, trimmed, CLOSED, probes + MANUAL out
    ML = YT.load(sleeves=["MOM-long"])            # one sleeve (names = engine_replay_report.sleeve_of)
    F["usd"] = YT.fixed_book_usd(F)                 # $ on a fixed $3,000 book at live sizing (cell multipliers on)
CLI:
    venv/bin/python scripts/yr5_fills_trimmed.py [--prefix yr5] [--out reports/x.csv] [--untrimmed]
Columns added: seed, chunk, tag, t (opened_at, naive UTC), in_window, sleeve, pct (= pnl_percentage), cm (cell_multiplier ×
cell_lev_multiplier, ≥ 0.1), is_probe. Asserts no duplicate (seed, pair, opened_at) after trimming.
"""
import argparse, glob, json, os, re

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REP = os.path.join(ROOT, "reports", "backtest_cache", "replay")
BOOK_USD = 3000.0
# Notional as a fraction of equity for a 1× fill at full book — the engine's own calculate_position_size on the shared-book
# ledger (scripts/engine_replay_report.py), median over yr5 fills (identical for seeds 1/2/3). fixed $ = pct/100 × BASE × cm × BOOK.
BASE_NOTIONAL_FRAC = 4.875


def sleeve_of(strat, direction):
    """Same labels as scripts/engine_replay_report.py sleeve_of."""
    s = strat if isinstance(strat, str) and strat else "MOMENTUM"
    if s.startswith("FLIP"):
        return "FLIP-short"
    if s == "MOMENTUM":
        return "MOM-long" if direction == "LONG" else "MOM-short"
    return {"BULLRUN_LONG": "BullRun-Long", "BEARRUN_SHORT": "BearRun-Short", "SPIKE_CHASE": "Spike-Chase",
            "SPIKE_FADE": "Spike-Fade", "SPIKE_BOUNCE": "Spike-Bounce"}.get(s, s)


def chunk_windows(prefix="yr5"):
    """[(tag, chunk, seed, start, end, warm)] for every finished chunk run (has a meta json)."""
    rx = re.compile(re.escape(prefix) + r"_(\d{4}-\d{2}(?:-\d{2})?)_s(\d+)_meta\.json$")
    out = []
    for f in sorted(glob.glob(os.path.join(REP, f"{prefix}_*_meta.json"))):
        m = rx.search(os.path.basename(f))
        if not m:
            continue
        meta = json.load(open(f))
        out.append((os.path.basename(f)[:-len("_meta.json")], m.group(1), int(m.group(2)),
                    pd.Timestamp(meta["start_ms"], unit="ms"), pd.Timestamp(meta["end_ms"], unit="ms"),
                    pd.Timestamp(meta.get("warm_ms", meta["start_ms"]), unit="ms")))
    return out


def load(prefix="yr5", sleeves=None, trim=True, closed_only=True, drop_probes=True, drop_manual=True, seeds=None):
    parts = []
    for tag, chunk, seed, a, z, _w in chunk_windows(prefix):
        if seeds and seed not in seeds:
            continue
        of = os.path.join(REP, f"{tag}_orders.csv")
        if not os.path.exists(of):
            continue
        o = pd.read_csv(of, low_memory=False).copy()   # .copy(): the wide orders frame is fragmented
        if not len(o):
            continue
        o["t"] = pd.to_datetime(o.opened_at.astype(str).str[:23].str.replace("T", " "), format="mixed")
        o["in_window"] = (o.t >= a) & (o.t < z)
        if trim:
            o = o[o.in_window]
        parts.append(o.assign(seed=seed, chunk=chunk, tag=tag))
    if not parts:
        raise SystemExit(f"no {prefix} chunk outputs found in {REP}")
    F = pd.concat(parts, ignore_index=True)
    if closed_only:
        F = F[F.status == "CLOSED"]
    F["sleeve"] = [sleeve_of(s, d) for s, d in zip(F.entry_strategy, F.direction)]
    F["is_probe"] = F.cell_multiplier_source.fillna("").astype(str).str.contains("PROBE")
    if drop_probes:
        F = F[~F.is_probe]
    if drop_manual:
        F = F[F.entry_strategy.fillna("MOMENTUM").astype(str) != "MANUAL"]
    if sleeves:
        F = F[F.sleeve.isin(sleeves)]
    F["pct"] = pd.to_numeric(F.pnl_percentage, errors="coerce")
    F["cm"] = (pd.to_numeric(F.cell_multiplier, errors="coerce").fillna(1.0)
               * pd.to_numeric(F.cell_lev_multiplier, errors="coerce").fillna(1.0)).clip(lower=0.1)
    if trim:
        assert not F.duplicated(["seed", "pair", "direction", "opened_at"]).any(), "duplicate fills after trimming"
    return F.sort_values(["seed", "t"]).reset_index(drop=True)


def fixed_book_usd(F, book=BOOK_USD, base=BASE_NOTIONAL_FRAC, cells=True):
    """$ per fill on a FIXED (non-compounding) book at live sizing. cells=False → every fill at 1×."""
    return F.pct / 100.0 * base * (F.cm if cells else 1.0) * book


def summary(F, by="sleeve"):
    ns = F.seed.nunique()
    g = F.assign(usd=fixed_book_usd(F), usd1=fixed_book_usd(F, cells=False)).groupby(by)
    return pd.DataFrame({"N_per_seed": g.size() / ns, "WR": g.pct.apply(lambda x: (x > 0).mean() * 100),
                         "avg_pct": g.pct.mean(), "usd_per_seed_live": g.usd.sum() / ns, "usd_per_seed_1x": g.usd1.sum() / ns})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="yr5"); ap.add_argument("--out", default=None)
    ap.add_argument("--untrimmed", action="store_true", help="keep warm-up fills (for comparison only)")
    A = ap.parse_args()
    F = load(A.prefix, trim=not A.untrimmed)
    pd.set_option("display.width", 200)
    print(f"{A.prefix}: {len(F)} fills · seeds {sorted(F.seed.unique())} · chunks {F.chunk.nunique()} · "
          f"{'UNTRIMMED (warm-up kept)' if A.untrimmed else 'trimmed to [start_ms, end_ms)'}")
    print(summary(F).round(3).to_string())
    if A.out:
        F.to_csv(A.out, index=False); print("wrote", A.out)
