#!/usr/bin/env python3
"""FAST_TP — real-fill BEFORE → AFTER tables (per sleeve × batch) + the two motivating trades. Research only.
One size: every fill at $6,100 notional (today's ≈ 0.61 × a $10k account; the current-batch median notional is $5.9–6.3k for every
sleeve) → $ = net % × 61; DCR-cal over every calendar day of the batch (master: the era's first open → last close over all master rows;
current: the export's span), r_d = Σ fill % × 0.61 ÷ 100 by close day. Reads reports/FAST_TP_2026-11-20_real.pkl.
Usage: venv/bin/python -I scripts/fast_tp_real_tables.py"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_tp_sim as S   # noqa: E402
RP = S.RP; OUT = os.path.join(RP, "FAST_TP_2026-11-20")
MIN, DAY = 60_000, 86_400_000
X = 0.61
NAME = {"EXH": "EXHAUSTION_SHORT", "DIST": "DISTRIBUTION_SHORT", "FLATBASE": "FLATBASE_SHORT", "WILLY": "FRENZY_WILLY", "HT": "FRENZY_WILLY_HT"}


def dcr(net, xday, d0, d1):
    r = pd.Series(np.asarray(net) * X / 100).groupby(np.asarray(xday)).sum().reindex(np.arange(d0, d1 + 1), fill_value=0.0)
    eq = np.cumprod(1 + r.values)
    return (eq[-1] ** (1 / len(r)) - 1) * 100


def main():
    Z = pd.read_pickle(OUT + "_real.pkl")
    summ = json.load(open(OUT + "_summary.json"))
    m = pd.read_csv(os.path.join(RP, "MASTER_POOL_stacked.csv"), low_memory=False, usecols=["era", "opened_at", "closed_at"])
    m["o"] = (pd.to_datetime(m.opened_at, format="mixed") - pd.Timestamp(0)) // pd.Timedelta(days=1)
    m["c"] = (pd.to_datetime(m.closed_at, format="mixed") - pd.Timestamp(0)) // pd.Timedelta(days=1)
    span = m.groupby(m.era.astype(str)).agg(d0=("o", "min"), d1=("c", "max")).to_dict("index")
    cur = Z[Z.src == "batch"]
    span["CURRENT(11-18→11-20)"] = dict(d0=int(cur.open_ms.min() // DAY), d1=int(cur.close_ms.max() // DAY))
    rows = []
    key = {"EXH": "EXH", "DIST": "DIST", "FLATBASE": "FLATBASE", "WILLY": "WILLY", "HT": "HT"}
    for sl in ("EXH", "DIST", "FLATBASE", "WILLY", "HT"):
        r = summ[key[sl]]
        cells = [("frozen pre-reg pick" if r["pick"] else "DESIGN best (no pick: Δ ≤ 0)", r["pick"] or r["design_best_cell"]),
                 ("full-span best (hindsight)", r["full_best_cell_hindsight"])]
        g = Z[Z.sleeve == sl]
        if not len(g):
            rows.append(dict(sleeve=NAME[sl], batch="—", cell="—", N=0)); continue
        groups = [(b, gg) for b, gg in g.groupby("batch")]
        if g.src.eq("master").any():
            groups.append(("MASTER all batches", g[g.src == "master"]))
        for lab, c in cells:
            j = S.CELLS.index(c)
            for b, gg in groups:
                ref = gg.ref_net.values.astype(float)
                for suf in ("", "_b0"):
                    v = np.array([x[j] for x in gg["nets" + suf]], float); fi = np.array([int(x[j]) for x in gg["fire" + suf]], np.int64)
                    f = fi >= 0
                    xb = (gg.close_ms.values // DAY)
                    xa = np.where(f, (gg.open_ms.values // MIN * MIN + (fi + 1) * MIN) // DAY, xb)
                    if b == "MASTER all batches":
                        d0 = min(span[e]["d0"] for e in gg.batch.unique()); d1 = max(span[e]["d1"] for e in gg.batch.unique())
                    else:
                        d0, d1 = span[b]["d0"], span[b]["d1"]
                    rows.append(dict(sleeve=NAME[sl], batch=b, which=lab, cell=c, first_bar=("excluded" if suf == "" else "included"),
                                     N=len(gg), WR_b=(ref > 0).mean() * 100, WR_a=(v > 0).mean() * 100, avg_b=ref.mean(), avg_a=v.mean(),
                                     usd_b=ref.sum() * 61, usd_a=v.sum() * 61, dcr_b=dcr(ref, xb, d0, d1), dcr_a=dcr(v, xa, d0, d1),
                                     changed=int(f.sum()), saved=int((f & (ref <= 0)).sum()), cut=int((f & (ref > 0)).sum()),
                                     d_usd=(v - ref).sum() * 61, ref_src=gg.ref_src.iloc[0]))
    T = pd.DataFrame(rows)
    T.to_csv(OUT + "_real_tables.csv", index=False)
    # anecdotes
    A = []
    for p, sl in (("KERNELUSDT", "DIST"), ("NAORISUSDT", "HT")):
        a = Z[(Z.pair == p) & (Z.sleeve == sl) & (Z.opened_at.astype(str).str.startswith("2026-11-20"))]
        for _, rw in a.iterrows():
            fired = {c: (int(rw.fire_b0[i]), round(float(rw.nets_b0[i]), 2), int(rw.fire[i]), round(float(rw.nets[i]), 2))
                     for i, c in enumerate(S.CELLS) if rw.fire_b0[i] >= 0 or rw.fire[i] >= 0}
            A.append(dict(pair=p, sleeve=NAME[sl], opened=rw.opened_at, entry=rw.entry_price, ref_net=rw.ref_net, peak_fav=rw.peak_fav,
                          min_to_peak=rw.min_to_peak, cpath15=[round(float(x), 2) for x in rw.cpath[:15]], fired=fired))
    json.dump(A, open(OUT + "_anecdotes.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(T[T.first_bar == "excluded"].drop(columns=["first_bar", "ref_src"]).round(3).to_string())
    print(json.dumps(A, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
