#!/usr/bin/env python3
"""Bug gate for the yr6a 2025 replay (2026-10-13): 2025 FRENZY / WILLY exits off the day's top-50 walk on 1m bars (no ticks). Measure the
1m-path bias on 2026, where every fill has ticks: re-walk the SAME 2026 fills (seed 1) on the tick path and on a 1m-only path (ticks hidden),
same exit rule, same fill rules (scout_frenzy_willy_exit.load_path / evaluate). Output reports/YR6_1M_BIAS.txt + _rows.csv."""
import os, sys, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_timecap as WT  # noqa: E402
import scout_frenzy_willy_exit as FX  # noqa: E402
EXIT = {"FRENZY_WILLY": (1.25, 4.0, 90), "FRENZY_LITE": (3.0, 3.0, 720), "FRENZY_LONG": (3.0, 3.0, 720), "FRENZY_WIDE": (3.0, 3.0, 720)}


class NoTicks(dict):
    def __contains__(self, k):
        return True

    def __getitem__(self, k):
        return None


def job(rows):
    WT._NET_BLOCKED = True
    tc, out = {}, []
    for r in rows:
        tp, sl, cap = EXIT[r["sleeve"]]
        try:
            Pt = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), cap, tc)
            Pm = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), cap, NoTicks())
        except Exception as e:  # noqa: BLE001
            out.append(dict(r, err=str(e)[:60])); continue
        a, b = FX.evaluate(Pt, tp, sl, cap), FX.evaluate(Pm, tp, sl, cap)
        out.append(dict(r, src=FX.path_src(Pt), tick_pct=a["pct"], tick_why=a["why"], m1_pct=b["pct"], m1_why=b["why"], m1_bars=Pm["n_bar"]))
    return out


def main():
    F = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YR6_fills.csv"), low_memory=False)
    F = F[F.sleeve.isin(list(EXIT)) & (F.seed == 1) & (F.year.astype(str) == "2026")].copy()
    F["t"] = pd.to_datetime(F.t, format="mixed")
    F["te"] = F.t.values.astype("datetime64[ms]").astype("int64")
    items = [dict(sleeve=r.sleeve, pair=r.pair, te=int(r.te), E=float(r.entry_price), live=float(r.pct), day=str(r.t)[:10]) for r in F.itertuples()]
    ps = sorted({r["pair"] for r in items}); g = {p: k % 6 for k, p in enumerate(ps)}
    with Pool(6) as P:
        R = pd.DataFrame(sum(P.map(job, [[r for r in items if g[r["pair"]] == k] for k in range(6)]), []))
    R.to_csv(os.path.join(ROOT, "reports", "YR6_1M_BIAS_rows.csv"), index=False)
    L = ["1m-path bias on 2026 fills (seed 1): the same fill walked on ticks vs on 1m bars only — same exit rule (FRENZY +3/−3/12 h at the "
         "replay config, WILLY +1.25/−4/90).", ""]
    for s, d in R.groupby("sleeve"):
        d = d[(d.tick_why != "UNRES") & (d.m1_why != "UNRES") & (d.src == "ticks") & (d.m1_bars > 0)]
        diff = d.m1_pct - d.tick_pct
        L.append(f"{s:13} N {len(d):4} · ticks: WR {(d.tick_pct > 0).mean() * 100:4.1f} % avg {d.tick_pct.mean():+.3f} stop {(d.tick_why == 'STOP').mean() * 100:4.1f} % · "
                 f"1m: WR {(d.m1_pct > 0).mean() * 100:4.1f} % avg {d.m1_pct.mean():+.3f} stop {(d.m1_why == 'STOP').mean() * 100:4.1f} % · bias (1m − ticks) "
                 f"{diff.mean():+.3f} %/fill · exit kind differs on {(d.tick_why != d.m1_why).mean() * 100:.1f} %")
    txt = "\n".join(L); open(os.path.join(ROOT, "reports", "YR6_1M_BIAS.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
