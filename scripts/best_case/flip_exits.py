import sys, os, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); SC = os.path.join(ROOT, "reports", "best_case")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_exit_shadows as ES
NOW = int(time.time() * 1000)
b, _ = ES.load([sys.argv[1]]); b = b[b.o >= pd.Timestamp("2026-10-23 18:30")].assign(src="B25", era="B25")
M = pd.read_csv(os.path.join(ROOT, "reports/MASTER_POOL_stacked.csv"), low_memory=False)
M = M[(M.stack_keep == True) & (M.is_probe != True) & M.entry_strategy.astype(str).str.startswith("FLIP")]
os.makedirs(SC, exist_ok=True)
mp = os.path.join(SC, "master_flips.csv"); M.to_csv(mp, index=False)
m, _ = ES.load([mp])
m = m.merge(M.assign(_k=M.opened_at.astype(str).str.replace(" ", "T").str[:19])[["_k", "pair", "era", "stack_pct"]], on=["_k", "pair"], how="left").assign(src="MASTER")
P = pd.concat([b, m], ignore_index=True); P["kk"] = P["_k"]
USED = ("TP1.5", "TP2", "TP3")   # = best_dcr.EXITS: the exits the best case reads (TP2_SL3 is the scout's, NaN when still undecided)
bud = ES.LS.Budget(max_req=400) if "max_req" in ES.LS.Budget.__init__.__code__.co_varnames else ES.LS.Budget()
def early(r):
    """(11-06) a flip still inside its 12 h window: price it on the path so far when EVERY exit is already decided (TP or stop touched —
    walk_v2 books the first touch, more path cannot change it); any exit still undecided → stays PENDING (the run stops, as before)."""
    hz = int((NOW - int(r.te)) // ES.MIN) - 2
    if hz < 1:
        return None
    try:
        raw, st = ES.raw_path(r, hz, bud, NOW, fetch=True)
    except Exception:
        return None
    if raw is None or st == "gap":
        return None
    Pn = ES.to_net(raw, r.direction, getattr(r, "fin", ES.TAKER))
    ex = ES.exits_on(Pn, r)
    if any(ex.get(x, np.nan) != ex.get(x, np.nan) for x in USED):   # (operator 11-11) only the exits the best case prices must be decided; TP2_SL3's wider
        return None                                  # stop can stay undecided for 12 h on a CLOSED flip (AVAX 11-11) and is not read here
    return dict(status="PRICED", exits=ex, parity=ES.parity_ok(Pn, r.sl, r.pct, r.close_reason, r.tc), hz=hz, path=Pn)


def holds(r, hz=None, Pn=None):
    """(operator 11-06, best-case hold time) minutes from entry to each fixed exit on the fill's net path (the early-priced path when given,
    else the price_fill horizon): the first event touching the TP or the stop (exits_on's stop rule), else the FIX_CAP time exit. 1m bars
    are stamped at their END → a bar hit counts as its middle (−0.5 min, fb_combo's convention); ticks as stamped. No path / short page
    (gap) / no hit before a path that ends early → NaN, and best_dcr.py stops on a NaN hold (deep review 480)."""
    nan = {f"h_{x}": np.nan for x in ES.FLIP_EXITS}
    if Pn is None:
        try:
            raw, st = ES.raw_path(r, hz or ES.horizon_of(r), bud, NOW, fetch=True)
        except Exception:
            return nan
        if raw is None or st == "gap":
            return nan
        Pn = ES.to_net(raw, r.direction, getattr(r, "fin", ES.TAKER))
    out = {}
    for x in ES.FLIP_EXITS:
        tp, _, st = x[2:].partition("_SL")
        sl = max(float(st), float(r.sl)) if st else float(r.sl)
        hit = np.flatnonzero((Pn["hi"] >= float(tp)) | (Pn["lo"] <= -sl))
        if len(hit):
            m = (Pn["t"][hit[0]] - Pn["te"]) / 60000 - (0.5 if Pn["bar"][hit[0]] else 0.0)
        elif int(Pn["end_ms"]) >= int(Pn["te"]) + ES.FIX_CAP * ES.MIN:
            m = ES.FIX_CAP
        else:
            m = np.nan
        out[f"h_{x}"] = float(min(max(m, 0.0), ES.FIX_CAP)) if m == m else np.nan
    return out
cache, out = {}, []
for r in P.itertuples():
    res = ES.price_fill(r, bud, NOW, cache, fetch=True)
    if res["status"] == "PENDING" and res.get("err") == "window open":
        res = early(r) or res
    ex = res.get("exits") or {}
    out.append(dict(src=r.src, era=r.era, k=r.kk, pair=r.pair, booked=r.pct, stack=getattr(r, "stack_pct", np.nan), status=res["status"],
                    parity=res.get("parity"), **{x: ex.get(x, np.nan) for x in ES.FLIP_EXITS},
                    **(holds(r, res.get("hz"), res.get("path")) if res["status"] == "PRICED" else {f"h_{x}": np.nan for x in ES.FLIP_EXITS})))
D = pd.DataFrame(out); D.to_csv(os.path.join(SC, "flip_exits.csv"), index=False)
print(D.groupby(["src", "status"]).size())
for s, g in D[D.status == "PRICED"].groupby("src"):
    base = g["booked"] if s == "B25" else g["stack"]
    print(s, f"N {len(g)} · booked/stack {base.mean():+.3f} · " + " · ".join(f"{x} {g[x].mean():+.3f} (WR {(g[x] > 0).mean():.0%})" for x in ES.FLIP_EXITS), "· parity", g.parity.mean())
