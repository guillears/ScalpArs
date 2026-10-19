#!/usr/bin/env python3
"""SPIKE_FADE winners vs losers — exhaustive 1D + all-pairs 2D screen with a day-stratified permutation null and OOS confirmation.

Research only (2026-10-19, operator question on the KMNO 10-17 loss). Reads, never writes code/config.
Blocks:  MASTER = MASTER_POOL_stacked.csv kept SPIKE_FADE (stack_pct, STACK 2026-10-19a)
         BT     = ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv SPIKE_FADE (today's stack, seed 1), pct = tick-tail-corrected value from
                  reports/FADE_SIZE_MCAP_2026-10-19_rows.csv
         LIVE   = KMNO 2026-10-17 (scored only, not in any statistic)
Extra per-fill features: infl (24h vol / prior-7d avg daily vol), mcap (stamp or rebuilt), turnover = V/mcap — from the fade_infl.csv
scratch file built in the size/mcap study.
Outputs: reports/FADE_WL_SCREEN_2026-10-19_{coverage,screen_<block>_<y>,confirm,survivors,prereg,kmno}.csv + _summary.txt
"""
import sys, os, json, itertools
import numpy as np, pandas as pd

R = "/Users/guillearslanian/Downloads/NOFA AI"
SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
OUT = R + "/reports/FADE_WL_SCREEN_2026-10-19"
NSHUF = int(os.environ.get("NSHUF", 300))
RNG = np.random.default_rng(20261019)
MIN_N, MIN_D = 15, 8

# ---------------------------------------------------------------- load
m = pd.read_csv(R + "/reports/MASTER_POOL_stacked.csv", low_memory=False)
m = m[(m.entry_strategy == "SPIKE_FADE") & (m.stack_keep == True)].copy()
m["pct"] = m.stack_pct; m["blk"] = "MASTER"; m["era"] = m.era
b = pd.read_csv(R + "/reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv", low_memory=False)
b = b[b.entry_strategy == "SPIKE_FADE"].copy(); b["blk"] = "BT"; b["era"] = b.regime
rows = pd.read_csv(R + "/reports/FADE_SIZE_MCAP_2026-10-19_rows.csv")
rb = rows[rows.src == "BT"][["pair", "opened_at", "pct"]].rename(columns={"pct": "pct_tick"})
b = b.merge(rb, on=["pair", "opened_at"], how="left")
assert b.pct_tick.notna().all(), "BT rows did not all match the size/mcap rows file"
b["pct_1m"] = b.pct; b["pct"] = b.pct_tick
l = pd.read_csv("/Users/guillearslanian/Downloads/scalpars_orders_live_2026-10-19_03-22-01.csv", low_memory=False)
l = l[(l.entry_strategy == "SPIKE_FADE") & (l.pair == "KMNOUSDT")].copy(); l["pct"] = l.pnl_percentage; l["blk"] = "LIVE"; l["era"] = "B23-live"
A = pd.concat([m, b, l], ignore_index=True, sort=False)
A["is_stop"] = A.close_reason.astype(str).str.startswith("STOP_LOSS") | (A.pct <= -1.49)
A["loser"] = A.pct < 0
A["t"] = pd.to_datetime(A.opened_at, format="mixed")
A["day"] = A.t.dt.floor("D")
inf = pd.read_csv(SCR + "/fade_infl.csv")
srcmap = {"MASTER": "MASTER", "BT": "BT", "LIVE": "LIVE"}
inf = inf.rename(columns={"src": "blk"})[["blk", "pair", "opened_at", "infl", "mcap", "norm"]]
inf["t"] = pd.to_datetime(inf.opened_at, format="mixed"); inf = inf.drop(columns="opened_at")
A = A.merge(inf, on=["blk", "pair", "t"], how="left")
# ---------------------------------------------------------------- derived
V = pd.to_numeric(A.entry_pair_volume_24h_usd, errors="coerce")
A["d_log_vol24h"] = np.log10(V)
A["d_log_mcap"] = np.log10(A.mcap)
A["d_turnover"] = V / A.mcap
A["d_infl"] = A.infl
A["d_log_norm7d"] = np.log10(A.norm)
A["d_hour_utc"] = A.t.dt.hour + A.t.dt.minute / 60
A["d_weekend"] = (A.t.dt.dayofweek >= 5).astype(float)
A["d_di_spread"] = pd.to_numeric(A.entry_pos_di, errors="coerce") - pd.to_numeric(A.entry_neg_di, errors="coerce")
A["d_rsi_jump"] = pd.to_numeric(A.entry_rsi, errors="coerce") - pd.to_numeric(A.entry_rsi_prev, errors="coerce")
A["d_breadth_diff"] = pd.to_numeric(A.entry_bull_pct, errors="coerce") - pd.to_numeric(A.entry_bear_pct, errors="coerce")
A["d_btc_rsi_1h_delta"] = pd.to_numeric(A.entry_btc_rsi_1h, errors="coerce") - pd.to_numeric(A.entry_btc_rsi_1h_prev, errors="coerce")
A["d_btc_rsi_delta6"] = pd.to_numeric(A.entry_btc_rsi, errors="coerce") - pd.to_numeric(A.entry_btc_rsi_prev6, errors="coerce")
for lv in ["HEALTHY_BEAR", "STRONG_BEAR", "CHOPPY_WEAK", "CHOPPY_FLAT", "HEALTHY_BULL", "STRONG_BULL", "CHOPPY_STRONG"]:
    A["c_regime_" + lv] = (A.entry_btc_regime == lv).astype(float).where(A.entry_btc_regime.notna())
for lv in ["BEARISH", "BULLISH", "NEUTRAL"]:
    A["c_macro_" + lv] = (A.entry_macro_trend == lv).astype(float).where(A.entry_macro_trend.notna())

EXCL = {"entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional", "entry_liquidity_cap_notional",
        "entry_bracket_max_leverage", "entry_bracket_cap_notional", "entry_order_type", "entry_strategy", "entry_btc_regime",
        "entry_macro_trend", "entry_btc_regime_started_at", "entry_frenzy_spike_at", "entry_surge_trigger_at",
        "entry_mcap_usd", "entry_vol_mcap_ratio", "entry_cmc_rank"}  # mcap family replaced by the rebuilt d_* versions
cand = [c for c in A.columns if (c.startswith("entry_") or c.startswith("d_") or c.startswith("c_")) and c not in EXCL]
NUM = {}
for c in cand:
    x = A[c]
    if x.dtype == object or x.dtype == bool:
        x = x.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0}) if x.dtype != object or x.dropna().isin(["True", "False", True, False]).all() else pd.to_numeric(x, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    NUM[c] = x
F = pd.DataFrame(NUM)
A = pd.concat([A[["blk", "era", "pair", "opened_at", "t", "day", "pct", "is_stop", "loser", "close_reason"]].reset_index(drop=True),
               F.reset_index(drop=True)], axis=1)

# market-wide (window-unit) family
MARKET = lambda c: (c.startswith("entry_btc") or c in ("entry_bull_pct", "entry_bear_pct", "entry_eth_5m_ret1_pct", "entry_global_volume_ratio",
                    "d_breadth_diff", "d_btc_rsi_1h_delta", "d_btc_rsi_delta6", "d_hour_utc", "d_weekend") or c.startswith("c_regime"))

# ---------------------------------------------------------------- coverage table
cov = []
for c in F.columns:
    r = {"col": c, "family": "market(day)" if MARKET(c) else "pair"}
    for blk in ["MASTER", "BT"]:
        x = A.loc[A.blk == blk, c]
        r[f"cov_{blk}"] = round(x.notna().mean(), 3); r[f"nuniq_{blk}"] = x.nunique()
    r["KMNO"] = A.loc[A.blk == "LIVE", c].iloc[0] if (A.blk == "LIVE").any() else np.nan
    cov.append(r)
cov = pd.DataFrame(cov)


def usable(c, blk, thr=0.9):
    r = cov.set_index("col").loc[c]
    return r[f"cov_{blk}"] >= thr and r[f"nuniq_{blk}"] >= 2


cov["screened_MASTER"] = [usable(c, "MASTER") for c in cov.col]
cov["screened_BT"] = [usable(c, "BT") for c in cov.col]
cov["scored_only_BT"] = [(not usable(c, "BT")) and cov.set_index("col").loc[c, "cov_BT"] >= 0.5 and cov.set_index("col").loc[c, "nuniq_BT"] >= 2 for c in cov.col]
cov.to_csv(OUT + "_coverage.csv", index=False)


# ---------------------------------------------------------------- zone builders (thresholds frozen on the discovery block)
def zones_1d(D, cols):
    """returns list of (name, col, op, thr) ; ops: lt, ge, le, eq1"""
    Z = []
    for c in cols:
        x = D[c].dropna()
        if x.nunique() <= 2:
            if x.nunique() == 2:
                Z.append((f"{c}==1", c, "eq1", 1.0))
            continue
        if x.min() < 0 < x.max():
            Z.append((f"{c}<0", c, "lt", 0.0))
        med = x.median(); Z.append((f"{c}<med({med:.4g})", c, "lt", med))
        t1, t2 = x.quantile([1 / 3, 2 / 3])
        Z.append((f"{c}<=T1({t1:.4g})", c, "le", t1))
        Z.append((f"{c}>=T2({t2:.4g})", c, "ge", t2))
    return Z


def ind(x, op, thr):
    s = x.notna().values.astype(np.float32)
    v = x.fillna(0).values
    if op == "lt": z = v < thr
    elif op == "le": z = v <= thr
    elif op == "ge": z = v >= thr
    elif op == "eq1": z = v == 1
    elif op == "gt": z = v > thr
    return (z & (s > 0)).astype(np.float32), s


def build_matrix(D, cols, two_d=True):
    """1D zones + 2D median quadrants. returns meta list, Z (n,K), S (n,K)"""
    meta, Zs, Ss = [], [], []
    for (nm, c, op, thr) in zones_1d(D, cols):
        z, s = ind(D[c], op, thr); meta.append(dict(test=nm, kind="1D", col_a=c, op_a=op, thr_a=thr, col_b="", op_b="", thr_b=np.nan,
                                                     family="market" if MARKET(c) else "pair"))
        Zs.append(z); Ss.append(s)
    if two_d:
        meds = {}
        for c in cols:
            x = D[c].dropna()
            if x.nunique() <= 2: meds[c] = ("eq1", 1.0)
            else: meds[c] = ("lt", x.median())
        for a, bb in itertools.combinations(cols, 2):
            za, sa = ind(D[a], *meds[a]); zb, sb = ind(D[bb], *meds[bb]); s = sa * sb
            for qa, qb in [(1, 1), (1, 0), (0, 1), (0, 0)]:
                z = ((za if qa else (1 - za)) * (zb if qb else (1 - zb)) * s).astype(np.float32)
                opa = meds[a][0] if qa else {"lt": "ge", "eq1": "ne1"}[meds[a][0]]
                opb = meds[bb][0] if qb else {"lt": "ge", "eq1": "ne1"}[meds[bb][0]]
                fam = "market" if (MARKET(a) and MARKET(bb)) else ("pair" if not (MARKET(a) or MARKET(bb)) else "mixed")
                meta.append(dict(test=f"{a} {opa} {meds[a][1]:.4g} & {bb} {opb} {meds[bb][1]:.4g}", kind="2D", col_a=a, op_a=opa,
                                 thr_a=meds[a][1], col_b=bb, op_b=opb, thr_b=meds[bb][1], family=fam))
                Zs.append(z); Ss.append(s)
    return meta, np.stack(Zs, 1), np.stack(Ss, 1)


def apply_zone(D, r):
    """frozen-threshold zone on another block → (z, s)"""
    def one(c, op, thr):
        if op == "ne1":
            z, s = ind(D[c], "eq1", 1.0); return (1 - z) * s, s
        if op == "ge" and r["kind"] == "2D":
            z, s = ind(D[c], "ge", thr); return z, s
        return ind(D[c], op, thr)
    z, s = one(r["col_a"], r["op_a"], r["thr_a"])
    if r["kind"] == "2D":
        z2, s2 = one(r["col_b"], r["op_b"], r["thr_b"]); z = z * z2; s = s * s2
    return z, s


# ---------------------------------------------------------------- cluster-robust z, vectorised
class Screen:
    def __init__(self, D, Z, S):
        self.n = len(D)
        days, g = np.unique(D.day.values, return_inverse=True)
        self.G = np.zeros((len(days), self.n), np.float32); self.G[g, np.arange(self.n)] = 1
        self.g = g; self.ndays = len(days)
        self.Z, self.S = Z, S
        self.n1 = (Z * S).sum(0); self.n0 = ((1 - Z) * S).sum(0)
        self.zbar = self.n1 / np.maximum(self.n1 + self.n0, 1)
        self.Xt = (Z - self.zbar) * S
        self.GX = self.G @ self.Xt; self.GXZ = self.G @ (self.Xt * Z)
        self.sxx = (self.Xt ** 2).sum(0)
        self.d1 = ((self.G @ (Z * S)) > 0).sum(0); self.d0 = ((self.G @ ((1 - Z) * S)) > 0).sum(0)
        self.ok = (self.n1 >= MIN_N) & (self.n0 >= MIN_N) & (self.d1 >= MIN_D) & (self.d0 >= MIN_D)
        # day-size strata for the permutation
        sizes = np.bincount(g); self.day_idx = [np.where(g == k)[0] for k in range(self.ndays)]
        self.strata = {}
        for k, s in enumerate(sizes): self.strata.setdefault(s, []).append(k)

    def z(self, y):
        y = y.astype(np.float32)
        ZY = (self.Z * self.S * y[:, None]).sum(0); SY = (self.S * y[:, None]).sum(0)
        y1 = ZY / np.maximum(self.n1, 1); y0 = (SY - ZY) / np.maximum(self.n0, 1)
        bb = y1 - y0
        GXY = self.G @ (self.Xt * y[:, None])
        sc = GXY - y0 * self.GX - bb * self.GXZ
        Gc = self.ndays
        var = (sc ** 2).sum(0) / np.maximum(self.sxx, 1e-12) ** 2 * Gc / max(Gc - 1, 1)
        # floor = iid pooled-variance SE (a zone with zero outcome variance would otherwise give an infinite cluster z)
        SYY = (self.S * (y ** 2)[:, None]).sum(0)
        sse = SYY - self.n1 * y1 ** 2 - self.n0 * y0 ** 2
        s2 = np.maximum(sse, 0) / np.maximum(self.n1 + self.n0 - 2, 1)
        var = np.maximum(var, s2 * (1 / np.maximum(self.n1, 1) + 1 / np.maximum(self.n0, 1)))
        z = bb / np.sqrt(np.maximum(var, 1e-12))
        z[~self.ok] = 0
        return z, y1, y0

    def perm(self, y):
        """day-stratified permutation: whole days' outcome vectors swapped among days of equal size"""
        yp = np.empty_like(y)
        for s, ks in self.strata.items():
            pk = RNG.permutation(ks)
            for src, dst in zip(ks, pk):
                yp[self.day_idx[dst]] = y[self.day_idx[src]]
        return yp


def run_screen(blkname, D, cols, ys, two_d=True):
    meta, Z, S = build_matrix(D, cols, two_d)
    sc = Screen(D, Z, S)
    M = pd.DataFrame(meta); M["n_zone"] = sc.n1; M["n_rest"] = sc.n0; M["days_zone"] = sc.d1; M["days_rest"] = sc.d0; M["ok"] = sc.ok
    res = {}
    for yn in ys:
        y = D[yn].values.astype(float)
        z, y1, y0 = sc.z(y)
        Mi = M.copy(); Mi["y"] = yn; Mi["zone_mean"] = y1; Mi["rest_mean"] = y0; Mi["z"] = z
        is1 = (Mi.kind == "1D").values
        nmax1, nmax2, ncnt1, ncnt2 = [], [], [], []
        for _ in range(NSHUF):
            zp, _, _ = sc.z(sc.perm(y))
            a = np.abs(zp)
            nmax1.append(a[is1].max()); nmax2.append(a[~is1].max() if (~is1).any() else 0)
            ncnt1.append((a[is1] > 2).sum()); ncnt2.append((a[~is1] > 3).sum() if (~is1).any() else 0)
        nmax1, nmax2 = np.array(nmax1), np.array(nmax2)
        az = np.abs(z)
        Mi["fw_p"] = np.where(is1, [(nmax1 >= v).mean() for v in az], [(nmax2 >= v).mean() for v in az])
        Mi = Mi[Mi.ok]
        summ = dict(block=blkname, y=yn, n=len(D), days=sc.ndays, tests_1d=int((Mi.kind == "1D").sum()), tests_2d=int((Mi.kind == "2D").sum()),
                    real_1d_gt2=int((np.abs(Mi[Mi.kind == "1D"].z) > 2).sum()), null_1d_gt2_mean=float(np.mean(ncnt1)), null_1d_gt2_p95=float(np.percentile(ncnt1, 95)),
                    real_2d_gt3=int((np.abs(Mi[Mi.kind == "2D"].z) > 3).sum()), null_2d_gt3_mean=float(np.mean(ncnt2)), null_2d_gt3_p95=float(np.percentile(ncnt2, 95)),
                    p_count_1d=float((np.array(ncnt1) >= (np.abs(Mi[Mi.kind == "1D"].z) > 2).sum()).mean()),
                    p_count_2d=float((np.array(ncnt2) >= (np.abs(Mi[Mi.kind == "2D"].z) > 3).sum()).mean()) if two_d else np.nan,
                    max_1d=float(np.abs(Mi[Mi.kind == "1D"].z).max()), null_max_1d_p95=float(np.percentile(nmax1, 95)),
                    max_2d=float(np.abs(Mi[Mi.kind == "2D"].z).max()) if two_d else np.nan, null_max_2d_p95=float(np.percentile(nmax2, 95)) if two_d else np.nan)
        res[yn] = (Mi.sort_values("z", key=np.abs, ascending=False), summ)
        print(blkname, yn, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)
    return res


_BW = {}
def day_boot(D, z, s, y, nb=2000):
    """day-cluster bootstrap of zone mean, rest mean and delta (vectorised: multinomial day weights)"""
    key = (D.day.values.astype("datetime64[ns]").tobytes(), nb)
    if key not in _BW:
        ud, g = np.unique(D.day.values, return_inverse=True)
        G = np.zeros((len(ud), len(D))); G[g, np.arange(len(D))] = 1
        W = RNG.multinomial(len(ud), np.ones(len(ud)) / len(ud), size=nb).astype(float)
        _BW[key] = (G, W)
    G, W = _BW[key]
    zs, rs = z * s, (1 - z) * s
    per = G @ np.stack([zs * y, zs, rs * y, rs], 1)
    p = W @ per
    ok = (p[:, 1] > 0) & (p[:, 3] > 0); p = p[ok]
    a, r = p[:, 0] / p[:, 1], p[:, 2] / p[:, 3]
    return np.stack([a, r, a - r], 1)


def describe(D, z, s):
    zz = (z * s) > 0
    x = D[zz]; r = D[(((1 - z) * s) > 0)]
    def st(x):
        if len(x) == 0: return dict(N=0)
        return dict(N=len(x), days=x.day.nunique(), WR=round((x.pct > 0).mean() * 100, 1), stop=round(x.is_stop.mean() * 100, 1),
                    avg=round(x.pct.mean(), 3), sum=round(x.pct.sum(), 2), top_pair=x.pair.value_counts().index[0], top_pair_n=int(x.pair.value_counts().iloc[0]))
    return st(x), st(r)


if __name__ == "__main__":
    MA = A[A.blk == "MASTER"].reset_index(drop=True); BT = A[A.blk == "BT"].reset_index(drop=True); KM = A[A.blk == "LIVE"].reset_index(drop=True)
    for D in (MA, BT):
        D["y_pct"] = D.pct.astype(float); D["y_stop"] = D.is_stop.astype(float); D["y_loser"] = D.loser.astype(float)
    colsB = [c for c in cov.col if usable(c, "BT")]
    colsM = [c for c in cov.col if usable(c, "MASTER")]
    common = [c for c in colsB if c in colsM]
    print("screened cols BT", len(colsB), "MASTER", len(colsM), "common", len(common), flush=True)
    YS = ["y_pct", "y_stop", "y_loser"]
    summ = []
    allres = {}
    # discovery blocks: BT full (all 2D), BT 2025, BT 2026, MASTER
    BT25 = BT[BT.era == "2025 BULL"].reset_index(drop=True); BT26 = BT[BT.era != "2025 BULL"].reset_index(drop=True)
    for name, D, cols in [("BT", BT, colsB), ("BT2025", BT25, colsB), ("BT2026", BT26, colsB), ("MASTER", MA, colsM)]:
        res = run_screen(name, D, cols, YS)
        for yn, (Mi, s) in res.items():
            Mi.to_csv(f"{OUT}_screen_{name}_{yn}.csv", index=False); summ.append(s); allres[(name, yn)] = Mi
    pd.DataFrame(summ).to_csv(OUT + "_null_summary.csv", index=False)

    # ---------------------------------------------------- confirmation with frozen thresholds + confirmation-count null
    blocks = {"BT": BT, "BT2025": BT25, "BT2026": BT26, "MASTER": MA}
    dirs = [("BT2025", "BT2026"), ("BT2026", "BT2025"), ("BT", "MASTER"), ("MASTER", "BT")]
    conf_rows, conf_summ = [], []
    for (dsc, cnf) in dirs:
        for yn in YS:
            Mi = allres[(dsc, yn)]
            surv = Mi[((Mi.kind == "1D") & (Mi.z.abs() > 2)) | ((Mi.kind == "2D") & (Mi.z.abs() > 3))]
            Dc = blocks[cnf]
            if not all(c in Dc.columns for c in set(surv.col_a) | set(surv.col_b) - {""}): pass
            yc = Dc[yn].values.astype(float)
            prepared = []
            for _, r in surv.iterrows():
                if Dc[r.col_a].notna().mean() < 0.5 or (r.kind == "2D" and Dc[r.col_b].notna().mean() < 0.5):
                    continue
                z, s = apply_zone(Dc, r)
                dz = len(np.unique(Dc.day.values[(z * s) > 0])); dr = len(np.unique(Dc.day.values[((1 - z) * s) > 0]))
                if dz < 5 or dr < 5: continue
                prepared.append((r, z, s))
            def count_conf(y, keep_rows=False):
                n = 0
                for (r, z, s) in prepared:
                    zs, rs = z * s, (1 - z) * s
                    if zs.sum() == 0 or rs.sum() == 0: continue
                    d = (zs * y).sum() / zs.sum() - (rs * y).sum() / rs.sum()
                    if np.sign(d) != np.sign(r.zone_mean - r.rest_mean): continue
                    bt = day_boot(Dc, z, s, y, nb=300 if not keep_rows else 2000)
                    P = (np.sign(bt[:, 2]) == np.sign(r.zone_mean - r.rest_mean)).mean()
                    if P >= 0.90:
                        n += 1
                        if keep_rows:
                            zst, rst = describe(Dc, z, s)
                            conf_rows.append(dict(discover=dsc, confirm=cnf, y=yn, test=r.test, kind=r.kind, family=r.family, z_disc=round(r.z, 2), fw_p_disc=r.fw_p,
                                                  disc_zone=round(r.zone_mean, 3), disc_rest=round(r.rest_mean, 3), conf_delta=round(d, 3), P_conf=round(P, 3),
                                                  conf_zone=json.dumps(zst), conf_rest=json.dumps(rst)))
                return n
            real = count_conf(yc, keep_rows=True)
            # null: day-stratified shuffle of the confirmation block's outcome (cheap bootstrap inside)
            scC = Screen(Dc, np.ones((len(Dc), 1), np.float32), np.ones((len(Dc), 1), np.float32))
            nulls = [count_conf(scC.perm(yc)) for _ in range(min(NSHUF, 200))] if prepared else [0]
            conf_summ.append(dict(discover=dsc, confirm=cnf, y=yn, survivors=len(surv), testable=len(prepared), confirmed=real,
                                  null_mean=round(float(np.mean(nulls)), 2), null_p95=float(np.percentile(nulls, 95)), P=float((np.array(nulls) >= real).mean())))
            print("CONFIRM", conf_summ[-1], flush=True)
    pd.DataFrame(conf_rows).to_csv(OUT + "_confirm.csv", index=False)
    pd.DataFrame(conf_summ).to_csv(OUT + "_confirm_summary.csv", index=False)
    A.to_csv(OUT + "_features.csv", index=False)
