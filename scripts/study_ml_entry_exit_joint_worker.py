"""🎯×🚪 (DECISION_LOG 477) worker for scripts/study_ml_entry_exit_joint.py — prices every exit of the joint read on each fill's own path
through the validated replica (study_ml_exit_b25_run.evaluate). A separate module so process-pool children import the same variant set
(macOS spawns: globals added in the parent would not reach them)."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_exit_b25_run as RUN  # noqa: E402  (imports study_ml_exit_b25_data → patches the replica readers)
import study_ml_entry_exit_joint as J  # noqa: E402

VARIANTS = {"BASE": RUN.VARIANTS["BASE"], **{c: RUN.VARIANTS[c] for c in J.REGISTERED.values()}, **J.ftp_variants(RUN.V)}


def booked(variant, why, pct):
    """the % a variant books. A FIXED_TP is a resting limit at +tp: it fills AT the level, never at the bar's high (deep review 477 —
    on 1m paths the replica returns the bar extreme, ~70 % of FTP1's first-read gain was that overshoot). ERR / NO_DATA /
    END_OF_DATA (the path ran out, not an exit) → NaN. PURE."""
    w = str(why)
    if w.startswith(("ERR", "NO_DATA")) or w == "END_OF_DATA":
        return np.nan
    if w.endswith("FIXED_TP"):
        return min(float(pct), float(VARIANTS[variant]["tp"]))
    return float(pct)


def _work(batch):
    out = []
    for fx, key in batch:
        try:
            res, _ = RUN.run_fill(fx, VARIANTS)
            P = RUN.RP.prep_path(fx, 2 * 3600_000)                 # the bit-exact guard RUN._work carries: BASE == RP.evaluate
            if P is not None:
                k0, w0, p0 = RUN.RP.evaluate(P, fx, RUN.RP.BASE)
                k1, w1, p1 = RUN.evaluate(P, fx, VARIANTS["BASE"])
                if w0 != "END_OF_DATA" and (w0 != w1 or abs(p0 - p1) > 1e-12):
                    res["BASE"] = (f"ERR:guard {w0}|{w1}", np.nan, None, "err")
        except Exception as ex:                                     # one bad path never stops the read
            res = {v: (f"ERR:{ex}", np.nan, None, "err") for v in VARIANTS}
        for v, (why, pct, _t, src) in res.items():
            out.append(dict(key=key, variant=v, reason=why, pct=booked(v, why, pct), src=src))
    return out


def price(items, jobs=8):
    items = sorted(items, key=lambda x: (x[0]["pair"], x[0]["o_ms"]))
    n = max(1, len(items) // (max(jobs, 1) * 4))
    batches = [items[i:i + n] for i in range(0, len(items), n)]
    rows = []
    if jobs <= 1:
        for b in batches:
            rows.extend(_work(b))
    else:
        with ProcessPoolExecutor(jobs) as ex:
            for r in ex.map(_work, batches):
                rows.extend(r)
    return pd.DataFrame(rows)
